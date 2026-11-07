"""Decoder context masks: coarse per-frame evidence tiers for the transition costs.

The transition penalties of the duration-constrained Viterbi are context-dependent
(flipping possession right after a detected ball impact costs far less than flipping
while the ball is unseen inside a scrum). This module derives those coarse masks
purely from tracking, with NO Smart Tagging inputs, so external training matches and
the competition match run through one identical path:

    impact_loud    a kinematic kick/impact was verified near a player (dilated)
    control_loud   the verified-control mask of the provided control detector
    in_occlusion   live-play same-team occlusion without any observed ball contact
    stop_strong    ball out of pitch or boundary-exit signal (pure geometry)
    stop_lowtempo  ball absent/unseen while the teams move at walking tempo
    restart_cue    ball re-enters live play right after a strong stop window

Threshold provenance: every fixed threshold is pitch geometry (out-of-bounds,
boundary-exit) or a biomechanics-scale value copied from the provided reference
pipeline (team_dynamics uses the same 1.45/3.2 m/s walk/run cut; the low-tempo gate
reuses the same walking-ratio scale as its z_dead recipe). They only select WHICH
tier a frame falls into; how expensive each tier's transition is remains a free
decoder parameter learned on the external matches, so a coarse cue boundary does not
encode competition-match knowledge.
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/modules/ -> models/ -> repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession_model.ball_tracking import detect_ball_control_and_passes  # noqa: E402

# Tier selector thresholds (see provenance note above):
WALK_RATIO_LOWTEMPO = 0.55      # collective walking-dominant (walking_ratio_smooth)
RMS_SPEED_LOWTEMPO = 1.90       # collective m/s ceiling for "nobody is sprinting"
EXIT_SIGNAL_THRESHOLD = 0.0     # provided boundary_exit_signal is a soft 0..1 signal
CONTEXT_DILATION = {            # dilation half-windows in FRAMES (copied from pipeline.py)
    "impact": 2,                # pipeline dilates impact over window 5
    "control": 3,               # pipeline dilates control over window 7
    "restart": 5,               # pipeline dilates restart impulse over window 11
}
RESTART_LOOKBACK_S = 5.0        # stop window the restart cue reacts to
RESTART_SKIP_S = 2.0            # do not count the most recent skip seconds (reactivity)

CONTEXT_KEYS = (
    "impact_loud", "control_loud", "in_occlusion",
    "stop_strong", "stop_lowtempo", "restart_cue",
)


def dilate_bool(mask: np.ndarray, half_win: int) -> np.ndarray:
    """Binary dilation by a centred box window (O(n), same op as pipeline.py)."""
    if half_win <= 0:
        return mask.astype(bool)
    conv = np.convolve(mask.astype(np.int32), np.ones(2 * half_win + 1, dtype=np.int32), mode="same")
    return conv > 0


def _prev_fraction(mask: np.ndarray, skip: int, length: int) -> np.ndarray:
    """Fraction of True frames in the window [t - skip - length, t - skip) for each t.

    O(n) cumsum implementation; returns 0.0 where the window is empty.
    """
    m = np.asarray(mask, dtype=np.float64)
    cs = np.concatenate(([0.0], np.cumsum(m)))
    idx = np.arange(len(m), dtype=np.int64)
    end = np.maximum(idx - skip, 0)
    start = np.maximum(end - length, 0)
    cnt = (end - start).astype(np.float64)
    total = cs[end] - cs[start]
    out = np.divide(total, np.maximum(cnt, 1.0))
    out[cnt <= 0] = 0.0
    return out


def build_context(
    match: Dict,
    blocks: List[Tuple[int, int, int]],
    side: Dict,
) -> Dict[str, np.ndarray]:
    """Build the six context masks for one feature-built match.

    Reuses the provided `detect_ball_control_and_passes` (same path as the provided
    rule-based model) so the tiers agree with the control evidence the GBDT sees.
    """
    ball_data = side["ball_data"]
    dynamics = side["dynamics"]
    control = side.get("control")
    if control is None:  # features v1 path (no control reuse): compute here once
        control = detect_ball_control_and_passes(
            match["player_xy"],
            match["player_vel"],
            match["slot_teams"],
            ball_data,
            team_dynamics=dynamics,
            fps=float(match["fps"]),
        )

    obs_mask = np.asarray(ball_data["obs_mask"], dtype=bool)
    flight = np.asarray(ball_data["is_flight_bridge"], dtype=bool)
    oob = np.asarray(ball_data["out_of_bounds"], dtype=bool)
    exit_sig = np.asarray(ball_data["boundary_exit_signal"], dtype=np.float32)

    ctx: Dict[str, np.ndarray] = {}
    impact_any = np.asarray(control["impact_h"], dtype=bool) | np.asarray(control["impact_a"], dtype=bool)
    ctx["impact_loud"] = dilate_bool(impact_any, CONTEXT_DILATION["impact"])
    ctx["control_loud"] = dilate_bool(
        np.asarray(control["verified_control_mask"], dtype=bool), CONTEXT_DILATION["control"]
    )
    ctx["in_occlusion"] = np.asarray(control["in_same_team_occlusion"], dtype=bool)

    ctx["stop_strong"] = oob | (exit_sig > EXIT_SIGNAL_THRESHOLD)

    walk = np.asarray(dynamics["walking_ratio_smooth"], dtype=np.float32)
    rms = np.asarray(dynamics["rms_speed_smooth"], dtype=np.float32)
    with np.errstate(invalid="ignore"):
        lowtempo = (walk > WALK_RATIO_LOWTEMPO) & (rms < RMS_SPEED_LOWTEMPO)
    absent = (~obs_mask) & (~flight)
    lowtempo = np.where(np.isfinite(lowtempo), lowtempo, False)
    ctx["stop_lowtempo"] = dilate_bool(absent & lowtempo, CONTEXT_DILATION["control"])

    # Restart cue without Smart Tagging: live ball observed right after a stop window.
    stop_any = ctx["stop_strong"] | ctx["stop_lowtempo"]
    fps = float(match["fps"])
    prev_stop = _prev_fraction(
        stop_any, skip=int(RESTART_SKIP_S * fps), length=int(RESTART_LOOKBACK_S * fps)
    )
    live = (~oob) & obs_mask
    ctx["restart_cue"] = dilate_bool(live & (prev_stop >= 0.4), CONTEXT_DILATION["restart"])
    return ctx


def context_badge(ctx: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Small summary for logs: fraction of frames in each tier."""
    return {k: float(np.asarray(v, dtype=bool).mean()) for k, v in ctx.items()}
