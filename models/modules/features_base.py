"""Stage-1 feature builder for possession_arm_v1 (feature set v2).

Everything here is computed from the AUTOMATIC tracking of a match (players + ball),
identically for external training matches and for the competition match. Rules that
shape the design:

1.  NaN-native. Where the ball or a player is missing the feature stays NaN and
    XGBoost routes the missing value -- no interpolation of the competition match.
2.  Symmetric reuse. Ball-trajectory reconstruction, the verified-control detector
    (`detect_ball_control_and_passes`) and team dynamics are the PROVIDED
    possession_model implementations, so the arm shares one code path with the
    rule-based model instead of drifting from it. Their outputs (control scores,
    pass-chain bridges, occlusion bridges, proximity, verified-control mask) enter
    the feature matrix directly instead of being hand-scaled into logits.

Feature list (see FEATURE_NAMES order in the cache files):
  - ball:    observed flag, speed, 1 s net displacement, flight bridge,
             out-of-bounds, boundary-exit signal/impulse, unseen streak,
             normalized position and distance to each goal, velocity components
  - control: per-team control score, pass bridge, occlusion bridge, proximity
             score, min distance + relative speed to ball, verified-control mask,
             same-team-occlusion mask, all with signed H-A diff features
  - players: visible counts, attack direction per half
  - teams:   RMS speed, walk/run ratios, centroid speed, velocity alignment,
             attacking drift, lateral width diff, territory, kickoff mask,
             velocity-projected pitch control
  - smoothed: NaN-aware centered moving averages (0.8 s, 3.2 s, 9.6 s) of the
              most informative raw features
"""

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/modules/ -> models/ -> repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import half_video_blocks  # noqa: E402
from possession_model.ball_tracking import (  # noqa: E402
    detect_ball_control_and_passes,
    reconstruct_ball_trajectory,
)
from possession_model.team_dynamics import compute_team_dynamics, moving_average_1d  # noqa: E402

FEATURE_SET_TAG = "v2"

# Moving-average half-windows in FRAMES: 0.8 s, 3.2 s, 9.6 s at 25 fps. Round physical
# times, fixed for both cameras, not tuned to the competition match.
SMOOTH_HALF_WINDOWS = (10, 40, 120)

# Features fed through the smoothing stack (context windows into the classifier):
SMOOTHED_KEYS = (
    "ball_speed", "ball_net_disp_1s", "ball_x_norm", "ball_y_norm",
    "ball_exit_signal", "ball_frames_unseen",
    "ctrl_min_dist_h", "ctrl_min_dist_a", "ctrl_rel_speed_h", "ctrl_rel_speed_a",
    "ctrl_score_diff", "ctrl_pass_diff", "ctrl_occl_diff",
    "tm_rms", "tm_walk_ratio", "tm_centroid_speed", "tm_drift_h", "tm_width_diff_h",
    "tm_territory_h", "tm_pitch_control_h",
)


def _f32(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def _ball_position_features(ball_xy: np.ndarray, pitch: Tuple[float, float]) -> Dict[str, np.ndarray]:
    length, width = pitch
    xs = ball_xy[:, 0].astype(np.float32)
    ys = ball_xy[:, 1].astype(np.float32)
    finite_x = np.isfinite(xs)
    finite_y = np.isfinite(ys)
    return {
        "ball_x_norm": np.where(finite_x, xs / length, np.nan).astype(np.float32),
        "ball_y_norm": np.where(finite_y, ys / width, np.nan).astype(np.float32),
        "ball_dist_h_goal": np.where(finite_x, xs / length, np.nan).astype(np.float32),
        "ball_dist_a_goal": np.where(finite_x, (length - xs) / length, np.nan).astype(np.float32),
    }


def _unseen_streak(obs_mask: np.ndarray) -> np.ndarray:
    """Consecutive frames with no observed ball, counting back including the current frame.

    Streaks cross half blocks only during dead time, where "long streak" is exactly
    the right feature; half-time gaps inflate it by construction, like any stoppage.
    """
    out = np.zeros(len(obs_mask), dtype=np.float32)
    run = 0.0
    for i, seen in enumerate(obs_mask):
        run = 0.0 if seen else run + 1.0
        out[i] = run
    return out


def _attack_direction(player_xy: np.ndarray, slot_teams: np.ndarray,
                      blocks: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """+1 where the home team attacks towards increasing x (median-position rule)."""
    attack = np.zeros(len(player_xy), dtype=np.float32)
    for _, s_f, e_f in blocks:
        h_x = player_xy[s_f : e_f + 1, slot_teams == 0, 0]
        a_x = player_xy[s_f : e_f + 1, slot_teams == 1, 0]
        with np.errstate(all="ignore"):
            med_h = float(np.nanmedian(h_x))
            med_a = float(np.nanmedian(a_x))
        if not np.isfinite(med_h) or not np.isfinite(med_a):
            raise ValueError("attack direction undefined: a whole half has no visible player")
        attack[s_f : e_f + 1] = 1.0 if med_h < med_a else -1.0
    return attack


def build_feature_matrix(
    match: Dict,
    blocks: Optional[Sequence[Tuple[int, int, int]]],
    use_broadcast: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    """Assemble the Stage-1 feature matrix for one loaded match.

    match  : dict from data_external.load_external_match / target_match.load_target_match
    blocks : per-half decode blocks [(num, s, e), ...] -- required (attack direction
             needs them); external matches get theirs from the detected half-time
             split, the competition match from games.py.
    use_broadcast : keep False. The broadcast camera exists only for the competition
             match, so a broadcast-fused ball track would push the target outside
             the feature distribution the external training built; v1 stays
             tactical-camera-only on both sides.

    Returns (X (N, F) float32, names, side) -- side keeps ball_data/dynamics/control
    for the subsequent context build (no recompute).
    """
    if blocks is None:
        raise ValueError("blocks are required: pass per-half (num, s, e) tuples")

    ball_data = reconstruct_ball_trajectory(
        match["ball_xy"], None, pitch=tuple(match["pitch"]), fps=float(match["fps"])
    )
    dynamics = compute_team_dynamics(
        match["player_xy"],
        match["player_speed"],
        match["player_vel"],
        match["slot_teams"],
        ball_data["kin_xy"],
        list(blocks),
        tuple(match["pitch"]),
    )
    # Provided verified-control machinery -- same path as the rule-based model.
    control = detect_ball_control_and_passes(
        match["player_xy"],
        match["player_vel"],
        match["slot_teams"],
        ball_data,
        team_dynamics=dynamics,
        fps=float(match["fps"]),
    )

    feats: Dict[str, np.ndarray] = {}

    # --- ball evidence ---
    obs = np.asarray(ball_data["obs_mask"], dtype=bool)
    feats["ball_observed"] = obs.astype(np.float32)
    feats["ball_speed"] = _f32(ball_data["kin_speed"])
    feats["ball_net_disp_1s"] = _f32(ball_data["net_disp_1s"])
    feats["ball_is_flight"] = np.asarray(ball_data["is_flight_bridge"], dtype=bool).astype(np.float32)
    feats["ball_oob"] = np.asarray(ball_data["out_of_bounds"], dtype=bool).astype(np.float32)
    feats["ball_exit_signal"] = _f32(ball_data["boundary_exit_signal"])
    feats["ball_exit_impulse"] = np.asarray(ball_data["exit_impulse"], dtype=bool).astype(np.float32)
    feats["ball_frames_unseen"] = _unseen_streak(obs)
    feats.update(_ball_position_features(ball_data["kin_xy"], tuple(match["pitch"])))
    feats["ball_vx_ms"] = _f32(ball_data["kin_vel"][:, 0])
    feats["ball_vy_ms"] = _f32(ball_data["kin_vel"][:, 1])

    # --- verified-control evidence (provided detector outputs) ---
    for key, arr in {
        "ctrl_score_h": control["control_score_h"],
        "ctrl_score_a": control["control_score_a"],
        "ctrl_pass_h": control["pass_bridge_h"],
        "ctrl_pass_a": control["pass_bridge_a"],
        "ctrl_occl_h": control["occlusion_bridge_h"],
        "ctrl_occl_a": control["occlusion_bridge_a"],
        "ctrl_prox_h": control["proximity_score_h"],
        "ctrl_prox_a": control["proximity_score_a"],
        "ctrl_min_dist_h": control["min_dist_h"],
        "ctrl_min_dist_a": control["min_dist_a"],
        "ctrl_rel_speed_h": control["rel_speed_h"],
        "ctrl_rel_speed_a": control["rel_speed_a"],
    }.items():
        feats[key] = _f32(arr)
    feats["ctrl_score_diff"] = feats["ctrl_score_h"] - feats["ctrl_score_a"]
    feats["ctrl_pass_diff"] = feats["ctrl_pass_h"] - feats["ctrl_pass_a"]
    feats["ctrl_occl_diff"] = feats["ctrl_occl_h"] - feats["ctrl_occl_a"]
    feats["ctrl_prox_diff"] = feats["ctrl_prox_h"] - feats["ctrl_prox_a"]
    feats["ctrl_verified"] = np.asarray(control["verified_control_mask"], dtype=bool).astype(np.float32)
    feats["ctrl_in_occl"] = np.asarray(control["in_same_team_occlusion"], dtype=bool).astype(np.float32)
    feats["ctrl_impact_h"] = np.asarray(control["impact_h"], dtype=bool).astype(np.float32)
    feats["ctrl_impact_a"] = np.asarray(control["impact_a"], dtype=bool).astype(np.float32)
    feats["ctrl_impact_strength"] = _f32(control["impact_strength"])

    # --- visibility / direction ---
    visible = np.isfinite(match["player_xy"][:, :, 0])
    feats["n_visible"] = visible.sum(axis=1).astype(np.float32)
    feats["n_visible_h"] = visible[:, match["slot_teams"] == 0].sum(axis=1).astype(np.float32)
    feats["n_visible_a"] = visible[:, match["slot_teams"] == 1].sum(axis=1).astype(np.float32)
    feats["attack_dir_h"] = _attack_direction(match["player_xy"], match["slot_teams"], blocks)

    # --- team dynamics (provided implementation) ---
    for key, src in (
        ("tm_rms", "rms_speed_smooth"),
        ("tm_walk_ratio", "walking_ratio_smooth"),
        ("tm_run_ratio", "running_ratio_smooth"),
        ("tm_centroid_speed", "centroid_speed_smooth"),
        ("tm_vel_alignment", "velocity_alignment_smooth"),
        ("tm_drift_h", "attacking_drift_h"),
        ("tm_width_diff_h", "width_diff_h"),
        ("tm_territory_h", "territory_h"),
        ("tm_pitch_control_h", "pitch_control_h"),
    ):
        feats[key] = _f32(dynamics[src])
    feats["tm_kickoff_mask"] = np.asarray(dynamics["kickoff_formation_mask"], dtype=bool).astype(np.float32)

    # --- smoothed context copies ---
    for key in SMOOTHED_KEYS:
        base = np.asarray(feats[key], dtype=np.float32)
        for w in SMOOTH_HALF_WINDOWS:
            feats[f"sm{w}_{key}"] = moving_average_1d(base, half_win=w)

    names = list(feats.keys())
    X = np.column_stack([np.asarray(feats[n], dtype=np.float32) for n in names])
    side = {"ball_data": ball_data, "dynamics": dynamics, "control": control}
    return X, names, side


def target_blocks(game: str = "BAR-ATH") -> List[Tuple[int, int, int]]:
    """Published half blocks of the competition match (frame boundaries, not labels)."""
    return half_video_blocks(game)
