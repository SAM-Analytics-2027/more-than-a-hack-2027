"""External DFL / IDSSE Training Dataset Loader, Physical Ricochet Filter,
Domain-Adaptation Augmentation, and Half-Aware Official Metric Evaluation.
"""

import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL_DIR = os.path.join(ROOT, "evaluation")
for _p in (ROOT, EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import possession_metrics as pm  # noqa: E402
from possession_model.ball_tracking import (  # noqa: E402
    PASS_SPEED_SCALE_MPS,
    PITCH_LINE_TOLERANCE_M,
    REACH_SCALE_M,
)
from possession_model.data_loader import compute_player_kinematics  # noqa: E402
from possession_model.signal_ops import AWAY, HOME, runs_of_labels  # noqa: E402

# Human visual-motor reaction/control time (~0.24 s = 6 frames at 25 Hz) distinguishing an
# instantaneous single-player grazing deflection from sustained ball control.
MIN_SINGLE_TOUCH_CONTROL_S = 0.24


def default_training_dir(data_dir: Optional[str] = None) -> str:
    base = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
    return os.path.join(base, "training")


def list_dfl_matches(training_dir: Optional[str] = None, kind: str = "dfl_sporcate") -> List[str]:
    tdir = training_dir or default_training_dir()
    kdir = os.path.join(tdir, kind)
    if not os.path.isdir(kdir):
        raise FileNotFoundError(f"Training directory not found: {kdir}")
    return sorted(
        d for d in os.listdir(kdir)
        if os.path.isdir(os.path.join(kdir, d)) and not d.startswith(".")
    )


def parse_dfl_metadata(meta_path: str) -> dict:
    root = ET.parse(meta_path).getroot()
    gp = {}
    for pp in root.iter("ProviderParameter"):
        name = pp.findtext("Name")
        val = pp.findtext("Value")
        if name is not None:
            gp[name] = val

    fps = float(root.findtext(".//FrameRate") or 25.0)
    pitch_len = float(root.findtext(".//PitchLength") or root.findtext(".//pitch_length") or 105.0)
    pitch_wid = float(root.findtext(".//PitchWidth") or root.findtext(".//pitch_width") or 68.0)

    teams = [(t.get("id"), t.findtext("Name") or "") for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, got {len(teams)}")
    home_id = teams[0][0]
    slot_teams = np.array(
        [HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
        dtype=np.int8,
    )

    h1_end = int(gp["first_half_end"]) - 1 if "first_half_end" in gp else None
    h2_start = int(gp["second_half_start"]) - 1 if "second_half_start" in gp else None

    return {
        "fps": fps,
        "pitch": (pitch_len, pitch_wid),
        "team_names": (teams[0][1], teams[1][1]),
        "slot_teams": slot_teams,
        "h1_end": h1_end,
        "h2_start": h2_start,
    }


def _parse_raw_dfl_txt(raw_path: str, n_slots: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stream normalized DFL ATD text file into (N, n_slots, 3) and (N, 2) float32 arrays."""
    with open(raw_path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ":" in ln]

    n_frames = len(lines)
    pl = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    for idx, line in enumerate(lines):
        parts = line.rstrip("\r\n").split(":")
        if len(parts) != 3:
            continue
        f_idx = int(parts[0]) - 1
        if f_idx < 0 or f_idx >= n_frames:
            f_idx = idx

        b_str = parts[2]
        if b_str and b_str[0] != "N":
            c1 = b_str.find(",")
            c2 = b_str.find(",", c1 + 1)
            ball[f_idx, 0] = float(b_str[:c1])
            ball[f_idx, 1] = float(b_str[c1 + 1 : c2])

        for s_idx, group in enumerate(parts[1].split(";")):
            if not group or group[0] == "N" or s_idx >= n_slots:
                continue
            c1 = group.find(",")
            c2 = group.find(",", c1 + 1)
            pl[f_idx, s_idx, 0] = float(group[:c1])
            pl[f_idx, s_idx, 1] = float(group[c1 + 1 : c2])
            s_str = group[c2 + 1 :]
            if s_str and s_str[0] != "N":
                pl[f_idx, s_idx, 2] = float(s_str)

    return pl, ball


def load_dfl_match(
    match_id: str,
    kind: str = "dfl_sporcate",
    training_dir: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[Tuple[int, int, int]], Tuple[float, float], float]:
    """Load a DFL match into metric arrays matching `load_atd_arrays`, plus half_blocks, pitch, fps."""
    tdir = training_dir or default_training_dir()
    mdir = os.path.join(tdir, kind, match_id)
    meta = parse_dfl_metadata(os.path.join(mdir, f"{match_id}_atd.xml"))
    slot_teams = meta["slot_teams"]
    length, width = meta["pitch"]
    fps = meta["fps"]

    cache_dir = os.path.join(tdir, ".cache")
    cache_path = os.path.join(cache_dir, f"{kind}_{match_id}_v1.npz")

    if use_cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        pl_norm, ball_norm = z["pl"], z["ball"]
    else:
        pl_norm, ball_norm = _parse_raw_dfl_txt(os.path.join(mdir, f"{match_id}_atd.txt"), len(slot_teams))
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache_path, pl=pl_norm, ball=ball_norm)

    scale = np.array([length, width], dtype=np.float32)
    player_xy = pl_norm[:, : len(slot_teams), :2] * scale
    raw_speed = pl_norm[:, : len(slot_teams), 2]
    ball_xy = ball_norm[:, :2] * scale
    player_vel, player_speed = compute_player_kinematics(player_xy, raw_speed, fps=fps)

    n_frames = len(ball_xy)
    if meta["h1_end"] is not None and meta["h2_start"] is not None:
        half_blocks = [(1, 0, int(meta["h1_end"])), (2, int(meta["h2_start"]), n_frames - 1)]
    else:
        mid = n_frames // 2
        half_blocks = [(1, 0, mid - 1), (2, mid, n_frames - 1)]

    tac = {
        "player_xy": player_xy,
        "player_speed": player_speed,
        "player_vel": player_vel,
        "ball_xy": ball_xy,
        "slot_teams": slot_teams,
        "team_names": meta["team_names"],
    }
    return tac, half_blocks, (length, width), fps


def load_dfl_gt(match_id: str, training_dir: Optional[str] = None) -> np.ndarray:
    """Return (N,) <U1 array of ground-truth labels in {'H', 'A', 'D'}."""
    tdir = training_dir or default_training_dir()
    for sub in ("dfl_pulite", "dfl_sporcate"):
        p = os.path.join(tdir, sub, match_id, f"{match_id}_possession_gt.csv")
        if os.path.isfile(p):
            df = pd.read_csv(p, usecols=["label"])
            return df["label"].astype(str).str.strip().str.upper().values.astype("<U1")
    raise FileNotFoundError(f"GT CSV not found for {match_id} under {tdir}")


def filter_ricochets_by_physics(
    gt_raw: np.ndarray,
    clean_tac: Dict[str, np.ndarray],
    half_blocks: List[Tuple[int, int, int]],
    fps: float = 25.0,
) -> np.ndarray:
    """Align touch-by-touch DFL flags with CHALLENGE.md controlled possession using clean physical kinematics:
      - Dead-ball frames ('D') and standalone restarts ('D -> Team -> D') are never modified.
      - Any short spell where >= 2 distinct teammates touch the ball (completed pass) is preserved.
      - Only single-player uncontrolled deflections (< MIN_SINGLE_TOUCH_CONTROL_S of reach proximity
        at controlled ball speed, or parries straight into stoppage) are absorbed into the live spell.
    """
    out = gt_raw.copy()
    px = clean_tac["player_xy"]
    bx = clean_tac["ball_xy"]
    st = clean_tac["slot_teams"]
    h_slots = st == HOME
    a_slots = st == AWAY

    min_ctrl = max(2, int(round(MIN_SINGLE_TOUCH_CONTROL_S * fps)))
    min_ctrl_pre_dead = max(3, int(round(1.5 * MIN_SINGLE_TOUCH_CONTROL_S * fps)))
    max_probe_frames = int(round(1.60 * fps))

    for _, s_f, e_f in half_blocks:
        sub = out[s_f : e_f + 1].copy()
        changed = True
        while changed:
            changed = False
            runs = runs_of_labels(sub)
            for i, (a, b, lbl) in enumerate(runs):
                if lbl not in ("H", "A") or (b - a) > max_probe_frames:
                    continue
                prev_r = runs[i - 1] if i > 0 else None
                next_r = runs[i + 1] if i + 1 < len(runs) else None
                prev_live = prev_r is not None and prev_r[2] in ("H", "A")
                next_live = next_r is not None and next_r[2] in ("H", "A")
                if not prev_live and not next_live:
                    continue

                fa, fb = s_f + a, s_f + b
                b_seg = bx[fa:fb]
                p_seg = px[fa:fb][:, h_slots if lbl == "H" else a_slots, :]
                vis_b = np.isfinite(b_seg[:, 0])

                ctrl_frames = 0
                n_touchers = 0
                if vis_b.any():
                    dd = np.hypot(p_seg[:, :, 0] - b_seg[:, None, 0], p_seg[:, :, 1] - b_seg[:, None, 1])
                    near_mat = np.isfinite(dd) & (dd <= REACH_SCALE_M)
                    n_touchers = int(near_mat.any(axis=0).sum())
                    min_d = np.where(np.isfinite(dd), dd, 99.0).min(axis=1)
                    if len(b_seg) >= 3:
                        bsp = np.hypot(np.gradient(b_seg[:, 0]) * fps, np.gradient(b_seg[:, 1]) * fps)
                    else:
                        bsp = np.zeros(len(b_seg), dtype=np.float32)
                    ctrl_frames = int(((min_d <= REACH_SCALE_M) & (bsp <= PASS_SPEED_SCALE_MPS)).sum())

                # Preserve any spell with a multi-player pass (>= 2 teammates within reach)
                if n_touchers >= 2:
                    continue

                req_ctrl = min_ctrl_pre_dead if (prev_live and not next_live) else min_ctrl
                if ctrl_frames < req_ctrl:
                    if prev_live and not next_live:
                        tgt = prev_r[2]
                    else:
                        nbrs = [r for r in (prev_r, next_r) if r is not None and r[2] in ("H", "A")]
                        tgt = max(nbrs, key=lambda r: r[1] - r[0])[2]
                    sub[a:b] = tgt
                    changed = True
                    break

        out[s_f : e_f + 1] = sub
    return out


def augment_dfl_to_mtah_domain(
    tac: Dict[str, np.ndarray],
    pitch: Tuple[float, float],
    fps: float = 25.0,
    seed: int = 2027,
) -> Dict[str, np.ndarray]:
    """Simulate target optical tracking domain perturbations (camera jitter, player tracklet dropouts,
    duplicate slot detections, and extended ball occlusions) with deterministic RNG seeding.
    """
    rng = np.random.default_rng(seed)
    length, width = pitch
    player_xy = tac["player_xy"].copy()
    ball_xy = tac["ball_xy"].copy()
    slot_teams = tac["slot_teams"]
    n_frames, n_slots, _ = player_xy.shape

    # 1. Binomial-smoothed optical homography jitter
    raw_noise = rng.normal(0.0, 0.36, size=player_xy.shape).astype(np.float32)
    smooth_noise = raw_noise.copy()
    smooth_noise[1:-1] = 0.25 * raw_noise[:-2] + 0.50 * raw_noise[1:-1] + 0.25 * raw_noise[2:]
    vis_p = np.isfinite(player_xy[:, :, 0])
    player_xy[vis_p] += smooth_noise[vis_p]

    # 2. Contiguous player tracklet dropouts (stadium blind-spot / occlusion bursts)
    n_drop = max(1, int(round(0.048 * float(vis_p.sum()) / (3.5 * fps))))
    starts = rng.integers(0, max(1, n_frames - 25), size=n_drop)
    slots = rng.integers(0, n_slots, size=n_drop)
    durs = rng.integers(int(1.2 * fps), int(6.0 * fps), size=n_drop)
    for st, sl, du in zip(starts, slots, durs):
        player_xy[st : min(n_frames, st + du), sl, :] = np.nan

    # 3. Occasional duplicate player detections
    n_dup = max(1, int(round(0.055 * n_frames / 30.0)))
    dup_starts = rng.integers(0, max(1, n_frames - 40), size=n_dup)
    dup_durs = rng.integers(12, 48, size=n_dup)
    h_slots = np.flatnonzero(slot_teams == HOME)
    a_slots = np.flatnonzero(slot_teams == AWAY)
    for st, du in zip(dup_starts, dup_durs):
        en = min(n_frames, st + du)
        pool = h_slots if rng.random() < 0.5 else a_slots
        vis_sub = np.isfinite(player_xy[st:en, pool, 0])
        active_cols = np.flatnonzero(vis_sub.all(axis=0))
        empty_cols = np.flatnonzero((~vis_sub).all(axis=0))
        if len(active_cols) > 0 and len(empty_cols) > 0:
            src = pool[int(rng.choice(active_cols))]
            dst = pool[int(rng.choice(empty_cols))]
            offset = rng.normal(0.0, 0.35, size=(1, 2)).astype(np.float32)
            player_xy[st:en, dst, :] = player_xy[st:en, src, :] + offset

    # 4. Out-of-bounds ball dropout & extended occlusion gaps
    b_vis = np.flatnonzero(np.isfinite(ball_xy[:, 0]))
    if len(b_vis) > 0:
        oob = (
            (ball_xy[:, 0] < -PITCH_LINE_TOLERANCE_M)
            | (ball_xy[:, 0] > length + PITCH_LINE_TOLERANCE_M)
            | (ball_xy[:, 1] < -PITCH_LINE_TOLERANCE_M)
            | (ball_xy[:, 1] > width + PITCH_LINE_TOLERANCE_M)
        )
        oob_idx = np.flatnonzero(oob & np.isfinite(ball_xy[:, 0]))
        if len(oob_idx) > 0:
            chg = np.flatnonzero(np.diff(oob_idx) > 5) + 1
            for grp in np.split(oob_idx, chg):
                if rng.random() < 0.50:
                    ball_xy[grp] = np.nan

        vis_idx = np.flatnonzero(np.isfinite(ball_xy[:, 0]))
        for k in range(len(vis_idx) - 1):
            g_s = int(vis_idx[k]) + 1
            g_e = int(vis_idx[k + 1])
            if (g_e - g_s) >= int(2.0 * fps) and rng.random() < 0.28:
                ext_pre = int(rng.integers(10, 45))
                ext_post = int(rng.integers(10, 45))
                ball_xy[max(0, g_s - ext_pre) : min(n_frames, g_e + ext_post)] = np.nan

    nan_spd = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    player_vel, player_speed = compute_player_kinematics(player_xy, nan_spd, fps=fps)
    return {
        "player_xy": player_xy,
        "player_speed": player_speed,
        "player_vel": player_vel,
        "ball_xy": ball_xy,
        "slot_teams": slot_teams.copy(),
        "team_names": tac["team_names"],
    }


def evaluate_dfl_predictions(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
) -> dict:
    """Evaluate predictions on a two-half DFL match using the official scorer with a halftime separator."""
    (_, s1, e1), (_, s2, e2) = half_blocks
    sep = np.array(["X"], dtype="<U1")
    gt_sep = np.concatenate([gt_labels[s1 : e1 + 1], sep, gt_labels[s2 : e2 + 1]])
    pr_sep = np.concatenate([pred_labels[s1 : e1 + 1], sep, pred_labels[s2 : e2 + 1]])
    return pm.evaluate(pm.label_codes(gt_sep), pm.label_codes(pr_sep))
