"""Tactical Phase Priors & Set-Piece Stoppage/Restart Localization from Smart Tagging + Tracking.

Translates automatic Smart Tagging intervals into:
  1. Soft per-frame team-possession priors for active tactical phases (inverting defensive tags).
  2. Precise tracking-based localization of stoppage windows [t_stop .. t_restart) and restarts:
     - Guards against false out-of-bounds sideline/ball-boy movements before the true restart.
     - Walks back across the full pre-restart stoppage (up to 35 s) to avoid splitting dead-ball intervals.
     - Snaps t_stop to the exact boundary-exit frame when the ball crosses a touchline or goal-line.
"""

from typing import Dict, List, Tuple
import numpy as np

HOME = 0
AWAY = 1
NOBODY = -1

OFFENSIVE_CODES = {
    "BUILD UP": 0.90,
    "PROGRESSION": 0.95,
    "CREATING CHANCES": 0.70,
    "BALL IN FINAL THIRD": 0.75,
    "BALL IN THE BOX": 0.80,
    "PLAYERS IN THE BOX": 0.65,
    "LONG BALL": 0.85,
    "ATTACKING TRANSITION": 1.00,
}

DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD": 0.90,
    "DEFENDING IN MIDDLE THIRD": 0.90,
    "DEFENDING IN DEFENSIVE THIRD": 0.70,
    "DEFENSIVE TRANSITION": 1.00,
}


def resolve_team_label(tag_val: str, home_name: str, away_name: str) -> int:
    """Resolve a Smart Tagging team string to HOME (0), AWAY (1), or NOBODY (-1)."""
    if not tag_val:
        return NOBODY
    val = tag_val.strip().lower()
    if val in ("n/a", "na", "none", ""):
        return NOBODY
    if val == home_name.strip().lower() or "barcelona" in val or "barca" in val:
        return HOME
    if val == away_name.strip().lower() or "atletic" in val or "athletic" in val:
        return AWAY
    return NOBODY


def infer_throwin_team(
    player_xy: np.ndarray,
    slot_teams: np.ndarray,
    ball_obs_xy: np.ndarray,
    t_center: int,
    width: float,
) -> int:
    """Infer which team takes a Throw-in (where Smart Tagging sets Team='N/A')
    using touchline geometry: the thrower stands directly on y ~ 0 or y ~ width.
    """
    n_frames = len(player_xy)
    s = max(0, t_center - 65)
    e = min(n_frames, t_center + 20)

    bx = ball_obs_xy[s : min(n_frames, t_center + 75)]
    vis_b = np.isfinite(bx[:, 1])
    if vis_b.any():
        med_by = float(np.median(bx[vis_b, 1]))
        use_low_y = med_by < (0.5 * width)
        ref_x = float(np.median(bx[vis_b, 0]))
    else:
        py = player_xy[s:e, :, 1]
        med_py = float(np.nanmedian(py)) if np.isfinite(py).any() else 0.5 * width
        use_low_y = med_py < (0.5 * width)
        px = player_xy[s:e, :, 0]
        ref_x = float(np.nanmedian(px)) if np.isfinite(px).any() else 52.5

    px_win = player_xy[s:e, :, 0]
    py_win = player_xy[s:e, :, 1]
    vis = np.isfinite(px_win) & (np.abs(px_win - ref_x) < 18.0)
    if not vis.any():
        return NOBODY

    dist_line = py_win if use_low_y else (width - py_win)
    dist_line = np.where(vis, dist_line, 1e6)

    h_slots = slot_teams == HOME
    a_slots = slot_teams == AWAY
    min_h = np.min(dist_line[:, h_slots], axis=1)
    min_a = np.min(dist_line[:, a_slots], axis=1)

    score_h = float(np.sum((min_h < 1.6) & (min_h + 0.6 < min_a)))
    score_a = float(np.sum((min_a < 1.6) & (min_a + 0.6 < min_h)))

    if score_h > score_a + 2.0:
        return HOME
    if score_a > score_h + 2.0:
        return AWAY
    return NOBODY


def compute_tactical_tag_priors(
    events: List[dict],
    team_names: Tuple[str, str],
    player_xy: np.ndarray,
    slot_teams: np.ndarray,
    ball_data: Dict[str, np.ndarray],
    team_dynamics: Dict[str, np.ndarray],
    half_blocks: List[Tuple[int, int, int]],
    pitch: Tuple[float, float],
    fps: float = 25.0,
) -> Dict[str, np.ndarray]:
    """Compute per-frame tactical priors and stoppage/restart signals from Smart Tagging + tracking."""
    n_frames = len(player_xy)
    length, width = pitch
    home_name, away_name = team_names

    obs_xy = ball_data["obs_xy"]
    obs_mask = ball_data["obs_mask"]
    kin_speed = ball_data["kin_speed"]
    oob = ball_data["out_of_bounds"]
    exit_sig = ball_data["boundary_exit_signal"]
    exit_imp = ball_data["exit_impulse"]

    walk_ratio_sm = team_dynamics["walking_ratio_smooth"]
    rms_spd_sm = team_dynamics["rms_speed_smooth"]
    kickoff_setup = team_dynamics["kickoff_formation_mask"]

    team_prior_h = np.zeros(n_frames, dtype=np.float32)
    active_phase_score = np.zeros(n_frames, dtype=np.float32)
    stoppage_prior = np.zeros(n_frames, dtype=np.float32)
    restart_impulse = np.zeros(n_frames, dtype=bool)
    stop_impulse = exit_imp.copy()
    restart_team = np.full(n_frames, NOBODY, dtype=np.int8)

    # 1. Pre-half Kick Off localization
    kickoff_starts = [
        ev for ev in events
        if (ev["code"] or "").strip().upper() == "SET PIECES"
        and "kick off (start)" in (ev["tags"].get("Type") or "").strip().lower()
    ]

    for half_num, h_start, h_end in half_blocks:
        half_label = "1st half" if half_num == 1 else "2nd half"
        ko = next(
            (ev for ev in kickoff_starts if (ev["tags"].get("Half") or "").strip().lower() == half_label),
            None,
        )
        if ko is not None:
            sf = max(h_start, ko["start_frame"])
            ef = min(h_end, ko["end_frame"])
            bx = obs_xy[sf : ef + 1, 0]
            by = obs_xy[sf : ef + 1, 1]
            bsp = kin_speed[sf : ef + 1]
            dist_center = np.hypot(bx - 0.5 * length, by - 0.5 * width)
            near_center = np.flatnonzero(
                np.isfinite(dist_center) & (dist_center < 8.0) & ((bsp > 2.2) | (dist_center > 0.9))
            )
            if len(near_center) > 0:
                ko_frame = sf + int(near_center[0])
            else:
                ko_frame = min(h_end, max(h_start, ko["start_frame"] + int(round(3.4 * fps))))

            if ko_frame > h_start:
                stoppage_prior[h_start:ko_frame] = np.maximum(
                    stoppage_prior[h_start:ko_frame], 4.8
                )
            restart_impulse[ko_frame] = True

            w_s = max(h_start, ko_frame - 15)
            w_e = min(h_end + 1, ko_frame + 15)
            dc_all = np.hypot(player_xy[w_s:w_e, :, 0] - 0.5 * length, player_xy[w_s:w_e, :, 1] - 0.5 * width)
            dc_all = np.where(np.isfinite(dc_all), dc_all, 1e6)
            min_dc_h = float(np.min(dc_all[:, slot_teams == HOME]))
            min_dc_a = float(np.min(dc_all[:, slot_teams == AWAY]))
            if min_dc_h + 0.8 < min_dc_a:
                restart_team[ko_frame : min(h_end + 1, ko_frame + 75)] = HOME
            elif min_dc_a + 0.8 < min_dc_h:
                restart_team[ko_frame : min(h_end + 1, ko_frame + 75)] = AWAY

    # 2. Active Tactical Phases (soft team & live-play priors)
    for ev in events:
        code = (ev["code"] or "").strip().upper()
        sf = max(0, ev["start_frame"])
        ef = min(n_frames, ev["end_frame"])
        if ef <= sf:
            continue
        tm = resolve_team_label(ev["tags"].get("Team", ""), home_name, away_name)

        if code in OFFENSIVE_CODES and tm != NOBODY:
            w = OFFENSIVE_CODES[code]
            sign = 1.0 if tm == HOME else -1.0
            team_prior_h[sf:ef] += sign * w
            if code not in ("CREATING CHANCES", "BALL IN FINAL THIRD", "PLAYERS IN THE BOX"):
                active_phase_score[sf:ef] += 0.45 * w
            else:
                active_phase_score[sf:ef] += 0.25 * w
        elif code in DEFENSIVE_CODES and tm != NOBODY:
            w = DEFENSIVE_CODES[code]
            poss_tm = AWAY if tm == HOME else HOME
            sign = 1.0 if poss_tm == HOME else -1.0
            team_prior_h[sf:ef] += sign * w
            if code != "DEFENDING IN DEFENSIVE THIRD":
                active_phase_score[sf:ef] += 0.45 * w
            else:
                active_phase_score[sf:ef] += 0.25 * w

    team_prior_h = np.clip(team_prior_h, -1.6, 1.6)
    active_phase_score = np.clip(active_phase_score, 0.0, 1.1)

    # 3. Post-Goal Stoppages & Kick Off (After Goal)
    after_goal_kos = [
        ev for ev in events
        if (ev["code"] or "").strip().upper() == "SET PIECES"
        and "kick off (after goal)" in (ev["tags"].get("Type") or "").strip().lower()
    ]
    goal_tags = [ev for ev in events if (ev["code"] or "").strip().upper() == "GOALS"]

    for ko in after_goal_kos:
        sf = max(0, ko["start_frame"])
        ef = min(n_frames - 1, ko["end_frame"])
        bx = obs_xy[sf : ef + 1, 0]
        by = obs_xy[sf : ef + 1, 1]
        bsp = kin_speed[sf : ef + 1]
        dist_center = np.hypot(bx - 0.5 * length, by - 0.5 * width)
        near_center = np.flatnonzero(
            np.isfinite(dist_center) & (dist_center < 8.5) & ((bsp > 2.2) | (dist_center > 1.0))
        )
        if len(near_center) > 0:
            ko_frame = sf + int(near_center[0])
        else:
            ko_frame = min(n_frames - 1, sf + int(round(3.4 * fps)))

        restart_impulse[ko_frame] = True

        matching_goal = next(
            (g for g in goal_tags if abs(g["end_frame"] - sf) <= 150),
            None,
        )
        search_back_limit = (
            matching_goal["start_frame"] + int(8 * fps)
            if matching_goal
            else max(0, ko_frame - int(90 * fps))
        )
        t_goal_stop = ko_frame - int(15 * fps)
        scan_s = max(0, search_back_limit)
        scan_e = max(scan_s + 1, ko_frame - int(4 * fps))
        bx_g = obs_xy[scan_s:scan_e, 0]
        by_g = obs_xy[scan_s:scan_e, 1]
        near_goal_mouth = (
            np.isfinite(bx_g)
            & ((bx_g < 6.0) | (bx_g > length - 6.0))
            & (by_g > 0.30 * width)
            & (by_g < 0.70 * width)
        )
        hits = np.flatnonzero(near_goal_mouth)
        if len(hits) > 0:
            for h_idx in hits:
                cand = scan_s + int(h_idx) + 10
                if cand < ko_frame - 100 and float(np.nanmean(walk_ratio_sm[cand : min(ko_frame, cand + 250)])) > 0.62:
                    t_goal_stop = cand
                    break
        else:
            curr = ko_frame - 1
            while curr > scan_s and (walk_ratio_sm[curr] > 0.56 or kickoff_setup[curr] or not obs_mask[curr]):
                curr -= 1
            t_goal_stop = min(curr + 1, ko_frame - int(10 * fps))

        stoppage_prior[t_goal_stop:ko_frame] = np.maximum(
            stoppage_prior[t_goal_stop:ko_frame], 4.5
        )
        active_phase_score[t_goal_stop:ko_frame] = 0.0
        stop_impulse[t_goal_stop] = True

        w_s = max(0, ko_frame - 15)
        w_e = min(n_frames, ko_frame + 15)
        dc_all = np.hypot(player_xy[w_s:w_e, :, 0] - 0.5 * length, player_xy[w_s:w_e, :, 1] - 0.5 * width)
        dc_all = np.where(np.isfinite(dc_all), dc_all, 1e6)
        min_dc_h = float(np.min(dc_all[:, slot_teams == HOME]))
        min_dc_a = float(np.min(dc_all[:, slot_teams == AWAY]))
        if min_dc_h + 0.8 < min_dc_a:
            restart_team[ko_frame : min(n_frames, ko_frame + 75)] = HOME
        elif min_dc_a + 0.8 < min_dc_h:
            restart_team[ko_frame : min(n_frames, ko_frame + 75)] = AWAY
        elif matching_goal is not None:
            scorer = resolve_team_label(matching_goal["tags"].get("Team", ""), home_name, away_name)
            if scorer != NOBODY:
                restart_team[ko_frame : min(n_frames, ko_frame + 75)] = AWAY if scorer == HOME else HOME

    # 4. In-Match Set Pieces (Corner Kick, Goal Kick, Free Kick, Throw-in — sorted chronologically)
    in_match_sps = sorted(
        [
            ev for ev in events
            if (ev["code"] or "").strip().upper() == "SET PIECES"
            and "kick off" not in (ev["tags"].get("Type") or "").strip().lower()
        ],
        key=lambda x: x["start_frame"],
    )

    prev_restart_frame = 0
    for ev in in_match_sps:
        sp_type = (ev["tags"].get("Type") or "").strip().lower()
        sf = max(0, ev["start_frame"])
        ef = min(n_frames, ev["end_frame"])
        if ef - sf < 50:
            continue

        tm = resolve_team_label(ev["tags"].get("Team", ""), home_name, away_name)

        # Locate pre-restart stillness peak inside [sf .. sf+24s]
        search_end = min(ef, sf + int(24 * fps))
        sub_walk = walk_ratio_sm[sf:search_end]
        sub_vrms = rms_spd_sm[sf:search_end]
        sub_bsp = kin_speed[sf:search_end]
        sub_oob = oob[sf:search_end]

        still_end = min(len(sub_vrms), int(18 * fps))
        if still_end < 10 or not np.isfinite(sub_vrms[:still_end]).any():
            continue

        ball_dead_cue = (~np.isfinite(sub_bsp[:still_end])) | (sub_bsp[:still_end] < 2.2) | sub_oob[:still_end]
        stillness_score = (
            sub_walk[:still_end]
            - 0.22 * sub_vrms[:still_end]
            + 0.40 * ball_dead_cue.astype(np.float32)
        )
        t_still_rel = int(np.nanargmax(stillness_score))
        t_still = sf + t_still_rel

        if stillness_score[t_still_rel] < 0.22 and sub_walk[t_still_rel] < 0.46 and not sub_oob[:still_end].any():
            continue

        # Locate exact IN-BOUNDS restart strike t_restart
        t_restart = None
        post_s = max(sf + 12, t_still - 8)
        post_e = min(ef, max(sf + int(16 * fps), t_still + int(12 * fps)))
        for t_cand in range(post_s, post_e):
            sp_b = kin_speed[t_cand]
            bx_c, by_c = obs_xy[t_cand]
            if not (np.isfinite(sp_b) and sp_b > 4.0 and np.isfinite(bx_c)):
                continue

            fut_e = min(n_frames, t_cand + 6)
            if oob[t_cand + 1 : fut_e].any():
                continue

            if "corner" in sp_type:
                if ((bx_c < 15.0) or (bx_c > length - 15.0)) and ((by_c < 15.0) or (by_c > width - 15.0)):
                    t_restart = max(sf + 5, t_cand - 2)
                    break
            elif "goal kick" in sp_type:
                if (bx_c < 21.0) or (bx_c > length - 21.0):
                    t_restart = max(sf + 5, t_cand - 2)
                    break
            elif "throw" in sp_type:
                if (not oob[t_cand]) and ((by_c < 11.0) or (by_c > width - 11.0)):
                    t_restart = max(sf + 5, t_cand - 2)
                    break
            else:
                if not oob[t_cand]:
                    t_restart = max(sf + 5, t_cand - 2)
                    break

        if t_restart is None:
            for t_cand in range(t_still + 5, post_e):
                if oob[t_cand]:
                    continue
                if rms_spd_sm[t_cand] > max(2.05, rms_spd_sm[t_still] + 0.55) and walk_ratio_sm[t_cand] < 0.58:
                    t_restart = t_cand
                    break
        if t_restart is None:
            t_restart = min(ef - 1, max(t_still + int(2.0 * fps), sf + int(5.0 * fps)))

        # Walk t_stop backwards across the entire pre-restart stoppage (up to 35 s before sf)
        back_limit = max(prev_restart_frame + int(2.5 * fps), sf - int(35 * fps))
        t_stop = min(t_still, t_restart - 1)
        while t_stop > back_limit:
            prev = t_stop - 1
            is_slow_or_out = (
                oob[prev]
                or (exit_sig[prev] > 0.5)
                or (walk_ratio_sm[prev] > 0.50)
                or (not obs_mask[prev] and (rms_spd_sm[prev] < 2.35 or walk_ratio_sm[prev] > 0.44))
                or (np.isfinite(kin_speed[prev]) and kin_speed[prev] < 1.9 and walk_ratio_sm[prev] > 0.44)
            )
            if is_slow_or_out:
                t_stop = prev
            else:
                break

        # Snap t_stop to exact boundary-exit frame when the ball crossed out-of-bounds nearby
        snap_lo = max(back_limit, t_stop - int(3.5 * fps))
        snap_hi = min(t_restart - int(2.5 * fps), t_stop + int(4.0 * fps))
        if snap_hi > snap_lo:
            imp_hits = np.flatnonzero(exit_imp[snap_lo:snap_hi])
            if len(imp_hits) > 0:
                t_stop = snap_lo + int(imp_hits[0])

        if t_restart - t_stop < int(3.0 * fps):
            t_stop = max(back_limit, min(sf, t_restart - int(4.0 * fps)))

        stoppage_prior[t_stop:t_restart] = np.maximum(
            stoppage_prior[t_stop:t_restart], 3.5
        )
        active_phase_score[t_stop:t_restart] = 0.0
        stop_impulse[t_stop] = True
        restart_impulse[t_restart] = True
        prev_restart_frame = t_restart

        if tm == NOBODY and "throw" in sp_type:
            tm = infer_throwin_team(player_xy, slot_teams, obs_xy, t_restart, width)

        if tm != NOBODY:
            post_win = min(n_frames, t_restart + int(3.5 * fps))
            restart_team[t_restart:post_win] = tm
            sign = 1.0 if tm == HOME else -1.0
            team_prior_h[t_restart:post_win] = np.clip(
                team_prior_h[t_restart:post_win] + sign * 1.25, -2.4, 2.4
            )

    return {
        "team_prior_h": team_prior_h,
        "active_phase_score": active_phase_score,
        "stoppage_prior": stoppage_prior,
        "stop_impulse": stop_impulse,
        "restart_impulse": restart_impulse,
        "restart_team": restart_team,
    }
