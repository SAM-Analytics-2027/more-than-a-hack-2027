"""Stage 2 of possession_arm_v1: duration-constrained Viterbi sequence decoding.

Reuses the provided exact dynamic program (`possession_model.sequence_decoder`)
unchanged — one code path, no re-derivation — and supplies it with:

  1. frame costs C_frame(t, s) = -log P_model(s | x_t)^stretched, optionally shifted
     by w_prior * (-log P_prior(s)), where P_prior comes from the aggregate external
     ground truths only (never from the competition match);
  2. context-dependent transition penalties organized in tiers (loud/controlled/
     occluded/quiet for turnovers, strong/quiet for stoppage entry, loud/quiet for
     restarts). EVERY tier cost is a parameter of this module (`DecoderParams`); the
     start values copy the provided pipeline's constants purely as a search
     starting point, and the real values are learned by grid/coordinate search over
     the external training matches (never on the competition match).

The decoded states are {0=H, 1=A, 2=D}. Frames outside a decode block stay X (-1).
"""

import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession_model.sequence_decoder import decode_possession_sequence  # noqa: E402

HOME, AWAY, DEAD = 0, 1, 2
EPS = 1e-9


@dataclass
class DecoderParams:
    """Every scalar Stage 2 needs. Learned on external matches, never on the target."""

    # turnover tier costs -- transition cost for a direct H<->A flip at frame t:
    turn_loud: float = 4.5      # right after a verified kinematic impact
    turn_control: float = 13.5  # inside verified controlled possession
    turn_occlusion: float = 54.0  # inside a same-team live-play occlusion
    turn_quiet: float = 42.0    # elsewhere (open play, ball seen)

    # stoppage-entry tier costs -- live play -> D at frame t:
    stop_loud: float = 5.5      # with a strong stop cue (oob / exit / absent&lowtempo)
    stop_quiet: float = 23.0    # elsewhere

    # restart tier costs -- D -> live play at frame t:
    restart_loud: float = 2.0   # right after a restart cue
    restart_quiet: float = 18.5

    min_pos_frames: int = 45    # minimum H/A spell length
    min_stop_frames: int = 85   # minimum dead-ball interval length
    w_prior: float = 0.0        # weight of -log P_prior on frame costs


# Search start = the provided pipeline's hand-set constants, kept ONLY so the search
# begins from a known, documented point; they are not trusted or used as-is.
DEFAULT_PARAMS = DecoderParams()

PARAM_GRID = {
    "turn_loud": (1.5, 3.0, 4.5, 7.0),
    "turn_control": (5.0, 9.0, 13.5, 19.0),
    "turn_occlusion": (30.0, 42.0, 54.0, 70.0),
    "turn_quiet": (20.0, 30.0, 42.0, 54.0),
    "stop_loud": (0.0, 3.0, 5.5, 9.0),
    "stop_quiet": (8.0, 15.0, 23.0, 32.0),
    "restart_loud": (0.0, 2.0, 4.5, 7.0),
    "restart_quiet": (7.0, 13.0, 18.5, 25.0),
    # DFL ground truth counts touches shorter than 2 s, so the minimum H/A spell
    # length can fall well below the provided pipeline's hand-set 45 frames.
    "min_pos_frames": (10, 18, 25, 35, 50, 70, 90),
    "min_stop_frames": (35, 55, 85, 120, 160),
    "w_prior": (0.0, 0.25, 0.5, 0.75),
}


def frame_costs_from_posteriors(
    P: np.ndarray,
    prior: Optional[Sequence[float]] = None,
    w_prior: float = 0.0,
) -> np.ndarray:
    """(T, 3) posterior -> (T, 3) negative-log costs, optionally prior-shifted.

    P rows need not be normalized; they are clipped and renormalized here. With
    w_prior > 0 the cost gets a per-class constant w_prior * (-log prior), which
    biases frames whose evidence is weak toward nonzero-prior classes.
    """
    P = np.clip(np.asarray(P, dtype=np.float64), EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    costs = -np.log(P)
    if w_prior > 0.0 and prior is not None:
        pr = np.clip(np.asarray(prior, dtype=np.float64), EPS, 1.0)
        costs = costs + w_prior * (-np.log(pr))[None, :]
    return costs


def transition_costs(ctx: Dict[str, np.ndarray], p: DecoderParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatially resolve the three per-frame context penalty arrays.

    Returns
    -------
    (turnover_cost, stoppage_entry_cost, restart_cost), each (T,) float64.
    """
    context_impact = np.asarray(ctx["impact_loud"], dtype=bool)
    context_control = np.asarray(ctx["control_loud"], dtype=bool)
    in_occlusion = np.asarray(ctx["in_occlusion"], dtype=bool)
    stop_strong = np.asarray(ctx["stop_strong"], dtype=bool)
    stop_lowtempo = np.asarray(ctx["stop_lowtempo"], dtype=bool)
    restart_cue = np.asarray(ctx["restart_cue"], dtype=bool)
    n = len(context_impact)

    turnover = np.full(n, p.turn_quiet, dtype=np.float64)
    turnover[in_occlusion] = p.turn_occlusion
    turnover[context_control] = p.turn_control
    turnover[context_impact] = p.turn_loud

    stoppage = np.where(stop_strong | stop_lowtempo, p.stop_loud, p.stop_quiet)
    restart = np.where(restart_cue, p.restart_loud, p.restart_quiet)
    return turnover, stoppage, restart


def decode_match(
    P: np.ndarray,
    ctx: Dict[str, np.ndarray],
    blocks: List[Tuple[int, int, int]],
    params: DecoderParams,
    prior: Optional[Sequence[float]] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Decode one whole match per block and stitch; -1 outside blocks.

    P : (T, 3) model posteriors in (H, A, D); NaN rows are neutralized to uniform.
    The class prior may be passed explicitly or attached to `params.prior` (the
    decoder search does the latter); with w_prior == 0 it is unused anyway.
    """
    pri = prior if prior is not None else getattr(params, "prior", None)
    n = len(P)
    P = np.where(np.isfinite(P), P, 1.0 / 3.0)  # NaN boxes -> uniform posterior
    frame_costs = frame_costs_from_posteriors(P, prior=pri, w_prior=params.w_prior)
    turnover, stoppage, restart = transition_costs(ctx, params)

    out = np.full(n, -1, dtype=np.int8)
    code_map = np.array((HOME, AWAY, DEAD), dtype=np.int8)
    for _, s_f, e_f in blocks:
        states = decode_possession_sequence(
            frame_costs[s_f : e_f + 1],
            turnover[s_f : e_f + 1],
            stoppage[s_f : e_f + 1],
            restart[s_f : e_f + 1],
            min_possession_frames=int(params.min_pos_frames),
            min_stoppage_frames=int(params.min_stop_frames),
        )
        out[s_f : e_f + 1] = code_map[states]
        if verbose:
            print(f"  block {s_f}..{e_f} decoded ({e_f - s_f + 1} frames)")
    return out


def params_to_dict(p: DecoderParams) -> Dict:
    return asdict(p)


def params_from_dict(d: Dict) -> DecoderParams:
    clean = {k: v for k, v in d.items() if k in DecoderParams.__dataclass_fields__}
    return DecoderParams(**clean)
