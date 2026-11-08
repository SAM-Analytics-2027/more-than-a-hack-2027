"""Rule-pipeline posteriors on the shared v2 feature base (family 9, fit-free).

The provided pipeline documented its per-frame negative log-likelihood costs
P(H|A|D | x_t) exactly (possession_model.pipeline.compute_frame_and_transition_costs
docstring): frame_costs = -log P(s | x_t), therefore P = exp(-frame_costs) and the
rows already sum to 1 in exact arithmetic.

Inputs we reuse WITHOUT recomputation risks:
  side = build_feature_matrix(match, blocks)[2] = {"ball_data", "dynamics", "control"}
  -- built by the SAME call sequence as possession_model.pipeline.run_possession_inference
     (reconstruct_ball_trajectory -> compute_team_dynamics -> detect_ball_control_and_passes),
     with use_broadcast=False on both sides (feature-base policy: the broadcast camera
     exists only for the competition match).

Smart Tagging is absent on the external matches, so the tactical arrays carry the
EXACT no-events defaults of possession_model.tactical_priors.compute_tactical_tag_priors
(zeros everywhere except stop_impulse = exit_impulse.copy() and restart_team =
NOBODY). We pass the same defaults at competition time: one identical rule both
for training files and for the target. The with-tags rule remains available as a
separate experiment (possession_v2_repro_head csv), never as a blend ingredient.
"""

from typing import Dict

import numpy as np

from possession_model.pipeline import compute_frame_and_transition_costs

NOBODY = -1

MODEL_TAG = "possession_arm_rule"


def tagless_tactical(ball_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """The exact compute_tactical_tag_priors output when `events == []`."""
    n = len(ball_data["obs_mask"])
    return {
        "team_prior_h": np.zeros(n, dtype=np.float32),
        "active_phase_score": np.zeros(n, dtype=np.float32),
        "stoppage_prior": np.zeros(n, dtype=np.float32),
        "stop_impulse": np.asarray(ball_data["exit_impulse"], dtype=bool).copy(),
        "restart_impulse": np.zeros(n, dtype=bool),
        "restart_team": np.full(n, NOBODY, dtype=np.int8),
    }


def rule_posterior(side: Dict) -> np.ndarray:
    """P = exp(-frame_costs) [n, 3] float32 for one match's `side` dicts."""
    costs, _turnover, _stop, _restart = compute_frame_and_transition_costs(
        side["ball_data"], side["control"], side["dynamics"],
        tagless_tactical(side["ball_data"]))
    costs = np.asarray(costs, dtype=np.float64)
    P = np.exp(-(costs - costs.min(axis=1, keepdims=True)))
    P /= P.sum(axis=1, keepdims=True)
    return P.astype(np.float32)
