"""Empirical Bayes Self-Calibration of Smart Tagging Log-Likelihood Ratios.

Translates optional automatic Smart Tagging intervals into parameter-free Bayesian log-odds
adjustments calibrated directly against the match's unlabeled tracking posteriors:
  - FIFA Law 8 centre-circle (9.15 m) Kick-Off start-of-half localization.
  - Empirical Bayes Dirichlet-shrunk log-likelihood ratios for tag categories.
  - Team-phase log-odds gated strictly by ball unobservability (1 - local_ball_visibility),
    so coarse tactical tags never override observed short ball turnovers.
"""

import re
from typing import Dict, List, Tuple
import numpy as np

from possession_model.signal_ops import AWAY, HOME, NOBODY, box_mean_blocks, logit

# FIFA Law 8 official centre-circle radius (metres).
FIFA_CENTRE_CIRCLE_RADIUS_M = 9.15


def _tokenize_name(name: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", (name or "").lower())
    return {t for t in tokens if len(t) >= 3 and t not in ("club", "futbol", "football")}


def resolve_team_label(tag_val: str, home_name: str, away_name: str) -> int:
    """Match a tag's team string to HOME (0), AWAY (1), or NOBODY (-1) via token/prefix overlap."""
    if not tag_val:
        return NOBODY
    val = tag_val.strip().lower()
    if val in ("n/a", "na", "none", ""):
        return NOBODY
    if val == home_name.strip().lower():
        return HOME
    if val == away_name.strip().lower():
        return AWAY

    v_toks = _tokenize_name(val)
    h_toks = _tokenize_name(home_name)
    a_toks = _tokenize_name(away_name)
    if not v_toks:
        return NOBODY

    def _overlap(t_set: set) -> int:
        score = len(v_toks & t_set) * 2
        for vt in v_toks:
            for tt in t_set:
                if vt[:4] == tt[:4] and len(vt) >= 4 and len(tt) >= 4:
                    score += 1
        return score

    sh = _overlap(h_toks)
    sa = _overlap(a_toks)
    if sh > sa and sh > 0:
        return HOME
    if sa > sh and sa > 0:
        return AWAY
    return NOBODY


def compute_empirical_bayes_tag_adjustments(
    events: List[dict],
    team_names: Tuple[str, str],
    ball_data: Dict[str, np.ndarray],
    p_dead_track: np.ndarray,
    p_home_track: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    pitch: Tuple[float, float],
    fps: float = 25.0,
) -> Dict[str, np.ndarray]:
    """Compute Empirical Bayes log-odds adjustments (llr_dead, llr_team_h) from Smart Tagging."""
    n_frames = len(p_dead_track)
    llr_dead = np.zeros(n_frames, dtype=np.float64)
    llr_team_h = np.zeros(n_frames, dtype=np.float64)
    pre_kickoff_mask = np.zeros(n_frames, dtype=bool)

    if not events:
        return {
            "llr_dead": llr_dead,
            "llr_team_h": llr_team_h,
            "pre_kickoff_mask": pre_kickoff_mask,
        }

    length, width = pitch
    home_name, away_name = team_names
    obs_xy = ball_data["obs_xy"]
    obs_mask = ball_data["obs_mask"]
    kin_speed = ball_data["kin_speed"]

    # 1. FIFA Law 8 Start-of-Half Kick-Off Localization (pre-kickoff walk from tunnel to kick)
    kickoff_starts = [
        ev for ev in events
        if (ev.get("code") or "").strip().upper() == "SET PIECES"
        and "kick off (start)" in ((ev.get("tags") or {}).get("Type") or "").strip().lower()
    ]
    for half_num, h_start, h_end in half_blocks:
        half_label = "1st half" if half_num == 1 else "2nd half"
        ko = next(
            (ev for ev in kickoff_starts if ((ev.get("tags") or {}).get("Half") or "").strip().lower() == half_label),
            None,
        )
        if ko is not None:
            sf = max(h_start, int(ko["start_frame"]))
            ef = min(h_end, int(ko["end_frame"]))
            bx = obs_xy[sf : ef + 1, 0]
            by = obs_xy[sf : ef + 1, 1]
            bsp = kin_speed[sf : ef + 1]
            dist_c = np.hypot(bx - 0.5 * length, by - 0.5 * width)
            moving = np.flatnonzero(
                np.isfinite(dist_c)
                & (dist_c <= FIFA_CENTRE_CIRCLE_RADIUS_M)
                & ((bsp > 1.8) | (dist_c > 1.2))
            )
            if len(moving) > 0:
                ko_frame = sf + int(moving[0])
            else:
                pd_seg = np.clip(p_dead_track[sf : ef + 1], 1e-4, 1.0 - 1e-4)
                ll_dead = np.cumsum(np.log(pd_seg))
                ll_alive = np.cumsum(np.log(1.0 - pd_seg)[::-1])[::-1]
                ko_frame = sf + int(np.argmax(ll_dead[:-1] + ll_alive[1:])) + 1
            if ko_frame > h_start:
                pre_kickoff_mask[h_start:ko_frame] = True

    # 2. Graded-frame mask and Empirical Bayes shrinkage pseudocount (10 seconds of frames)
    graded = np.zeros(n_frames, dtype=bool)
    for _, s_f, e_f in half_blocks:
        graded[s_f : e_f + 1] = True
    if not graded.any():
        return {"llr_dead": llr_dead, "llr_team_h": llr_team_h, "pre_kickoff_mask": pre_kickoff_mask}

    prior_n0 = 10.0 * fps
    base_p_dead = float(np.mean(p_dead_track[graded]))
    base_logit_dead = float(logit(base_p_dead))

    # Estimate team-attributed phase polarity per tag code from alive tracking posteriors
    alive_weight = graded.astype(np.float64) * (1.0 - p_dead_track)
    code_stats: Dict[str, List[float]] = {}  # code -> [weighted_p_for_tagged_team, total_weight]

    parsed_events = []
    for ev in events:
        code = (ev.get("code") or "").strip().upper()
        sf = max(0, int(ev.get("start_frame", 0)))
        ef = min(n_frames, int(ev.get("end_frame", 0)))
        if ef <= sf or not code:
            continue
        tags = ev.get("tags") or {}
        tm = resolve_team_label(tags.get("Team", ""), home_name, away_name)
        sub_type = (tags.get("Type") or "").strip().lower()
        parsed_events.append((code, sub_type, tm, sf, ef))

        if tm != NOBODY and code != "SET PIECES":
            w_sl = alive_weight[sf:ef]
            w_sum = float(w_sl.sum())
            if w_sum > 1.0:
                p_same = p_home_track[sf:ef] if tm == HOME else (1.0 - p_home_track[sf:ef])
                acc = code_stats.setdefault(code, [0.0, 0.0])
                acc[0] += float((w_sl * p_same).sum())
                acc[1] += w_sum

    # Chronologically sort events to ensure sequential boundary inference and non-leaking restart boundaries
    parsed_events.sort(key=lambda x: (x[3], x[4]))

    # Shrunk log-likelihood ratio per team-attributed tag code (positive => tagged team has ball;
    # negative => opponent has ball, e.g. DEFENDING tags automatically invert without manual lists!)
    code_llr: Dict[str, float] = {}
    for code, (sum_p, sum_w) in code_stats.items():
        post_p = (sum_p + 0.5 * prior_n0) / (sum_w + prior_n0)
        code_llr[code] = float(np.clip(logit(post_p), -1.5, 1.5))

    # Accumulate frame-level team log-odds gated by ball unobservability (1 - local_vis)
    w_vis = int(round(1.0 * fps))
    local_vis = box_mean_blocks(obs_mask.astype(np.float32), w_vis, w_vis, half_blocks)
    unobs_gate = np.clip(1.0 - local_vis, 0.0, 1.0).astype(np.float64)

    raw_team_sum = np.zeros(n_frames, dtype=np.float64)
    raw_team_cnt = np.zeros(n_frames, dtype=np.float64)
    for code, _, tm, sf, ef in parsed_events:
        if tm != NOBODY and code in code_llr:
            sign = 1.0 if tm == HOME else -1.0
            raw_team_sum[sf:ef] += sign * code_llr[code]
            raw_team_cnt[sf:ef] += 1.0

    has_tc = raw_team_cnt > 0
    mean_team_llr = np.zeros(n_frames, dtype=np.float64)
    mean_team_llr[has_tc] = raw_team_sum[has_tc] / np.sqrt(raw_team_cnt[has_tc])
    llr_team_h = unobs_gate * np.clip(mean_team_llr, -1.5, 1.5)

    # Empirical Bayes stoppage evidence on SET PIECES / GOALS intervals
    # Localizes the exact dead-ball period [t_stop, t_restart) via Maximum-Likelihood Change-Point Estimation
    exit_sig = ball_data["boundary_exit_signal"]
    oob = ball_data["out_of_bounds"]
    in_bounds_kick = obs_mask & (~oob) & np.isfinite(kin_speed) & (kin_speed > 2.5)

    sp_events = [ev for ev in parsed_events if ev[0] in ("SET PIECES", "GOALS") and "kick off" not in ev[1]]
    sp_dead_frames = np.zeros(n_frames, dtype=bool)

    prev_restart_frame = 0
    for code, sub_type, tm, sf, ef in sp_events:
        h_block = next(((h, hs, he) for h, hs, he in half_blocks if sf < he and ef > hs), None)
        if h_block is None:
            continue
        hs, he = h_block[1], h_block[2]
        sf_c = max(hs, sf)
        ef_c = min(he, ef)
        if ef_c <= sf_c + 2:
            continue

        pd_seg = np.clip(p_dead_track[sf_c:ef_c], 1e-4, 1.0 - 1e-4)
        ll_dead = np.cumsum(np.log(pd_seg))
        ll_alive = np.cumsum(np.log(1.0 - pd_seg)[::-1])[::-1]
        r_ml = sf_c + int(np.argmax(ll_dead[:-1] + ll_alive[1:])) + 1

        win_lo = max(sf_c, r_ml - int(2.5 * fps))
        win_hi = min(ef_c, r_ml + int(2.5 * fps))
        kicks = np.flatnonzero(in_bounds_kick[win_lo:win_hi])
        r_final = (win_lo + int(kicks[0])) if len(kicks) > 0 else r_ml

        back_limit = max(hs, prev_restart_frame + int(2.5 * fps), r_final - (ef_c - sf_c))
        exit_hits = np.flatnonzero((exit_sig[back_limit:r_final] > 0.5) | oob[back_limit:r_final])
        if len(exit_hits) > 0:
            s_final = back_limit + int(exit_hits[0])
        else:
            s_final = max(back_limit, sf_c)

        if r_final > s_final:
            sp_dead_frames[s_final:r_final] = True
            prev_restart_frame = r_final

            if tm != NOBODY:
                sign = 1.0 if tm == HOME else -1.0
                post_win = min(ef_c, r_final + int(4.0 * fps))
                llr_team_h[r_final:post_win] += sign * 1.25

    sp_graded = sp_dead_frames & graded
    if sp_graded.sum() > prior_n0:
        sp_p_dead = (float(p_dead_track[sp_graded].sum()) + prior_n0 * base_p_dead) / (
            float(sp_graded.sum()) + prior_n0
        )
        sp_delta_llr = max(0.0, float(logit(sp_p_dead)) - base_logit_dead)
        llr_dead[sp_graded] = np.maximum(llr_dead[sp_graded], sp_delta_llr)

    return {
        "llr_dead": llr_dead,
        "llr_team_h": llr_team_h,
        "pre_kickoff_mask": pre_kickoff_mask,
    }
