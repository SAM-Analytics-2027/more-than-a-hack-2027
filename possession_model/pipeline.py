"""End-to-End Algorithmic Ball Possession Inference Pipeline.

Stages:
  1. Load Tactical ATD player & ball trajectories and pitch geometry.
  2. Self-calibrated Broadcast-to-Tactical synchronization, TV replay rejection, and blind-spot recovery.
  3. Reconstruct physical ball trajectory, 22-player collective dynamics, and continuous ball-player kinematics.
  4. Score the 3-stage LightGBM cascade (Dead-Ball -> Equivariant Team -> Turnover Boundary).
  5. Fuse self-calibrated Empirical Bayes Smart Tagging LLRs and decode the globally optimal sequence via Viterbi DP.
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from games import GAME, GAMES, half_video_blocks
from possession_model.ball_tracking import (
    compute_ball_player_kinematics,
    reconstruct_ball_trajectory,
)
from possession_model.data_loader import (
    load_atd_arrays,
    load_smart_tags,
    sync_broadcast_to_tactical,
)
from possession_model.gbm_features import (
    extract_dead_features,
    extract_team_features,
    extract_transition_features,
)
from possession_model.gbm_model import PossessionGBMBundle, decode_match_sequence
from possession_model.tactical_priors import compute_empirical_bayes_tag_adjustments
from possession_model.team_dynamics import compute_team_dynamics


def run_possession_inference(
    game: str = GAME,
    data_dir: Optional[str] = None,
    use_broadcast: bool = True,
    model_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Execute the deterministic 3-GBM + Viterbi possession inference pipeline on `game`."""
    t0 = time.time()
    resolved_model_dir = model_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    bundle = PossessionGBMBundle.load(resolved_model_dir)

    if verbose:
        print(f"[1/5] Loading Tactical ATD & metadata for {game}...")
    tac = load_atd_arrays(game=game, data_dir=data_dir, source="tactical")
    pitch = GAMES[game]["pitch"]
    fps = float(GAMES[game]["fps"])
    blocks = half_video_blocks(game)

    tac_vis_graded = sum(int(np.isfinite(tac["ball_xy"][s : e + 1, 0]).sum()) for _, s, e in blocks)
    total_graded = sum(e - s + 1 for _, s, e in blocks)

    brd_ball = None
    tac_eff = tac
    if use_broadcast:
        if verbose:
            print("[2/5] Self-calibrating Broadcast camera sync & recovering blind-spot players...")
        sync_res = sync_broadcast_to_tactical(tac, game=game, data_dir=data_dir)
        if sync_res is not None:
            brd_ball = sync_res["brd_ball_xy"]
            tac_eff = {
                **tac,
                "player_xy": sync_res["fused_player_xy"],
                "player_speed": sync_res["fused_player_speed"],
                "player_vel": sync_res["fused_player_vel"],
            }
            if verbose:
                print(
                    f"      Recovered blind-spot sideline players on "
                    f"{sync_res['n_backfilled_frames']:,} live-verified frames."
                )

    if verbose:
        print("[3/5] Computing ball flight kinematics & 22-player spatial dynamics...")
    ball_data = reconstruct_ball_trajectory(tac["ball_xy"], brd_ball, pitch=pitch, fps=fps)
    fused_vis_graded = sum(int(ball_data["obs_mask"][s : e + 1].sum()) for _, s, e in blocks)
    kin_vis_graded = sum(int(np.isfinite(ball_data["kin_xy"][s : e + 1, 0]).sum()) for _, s, e in blocks)
    if verbose:
        print(
            f"      Graded ball coverage: Tactical {100.0 * tac_vis_graded / total_graded:.1f}% "
            f"-> Dual-Cam Observed {100.0 * fused_vis_graded / total_graded:.1f}% "
            f"-> With Ballistic Arcs {100.0 * kin_vis_graded / total_graded:.1f}%"
        )

    team_dynamics = compute_team_dynamics(
        tac["player_xy"],
        tac["player_speed"],
        tac["player_vel"],
        tac["slot_teams"],
        ball_data["kin_xy"],
        blocks,
        pitch,
        fps=fps,
    )
    ball_kin = compute_ball_player_kinematics(
        tac_eff["player_xy"],
        tac_eff["player_vel"],
        tac_eff["slot_teams"],
        ball_data,
        fps=fps,
    )

    if verbose:
        print("[4/5] Scoring 3-stage LightGBM cascade (Dead-Ball -> Equivariant Team -> Turnover)...")
    dead_ft = extract_dead_features(ball_data, team_dynamics, blocks, fps=fps)
    p_dead = bundle.predict_dead_posterior(dead_ft["X_dead"], blocks)

    team_ft = extract_team_features(
        ball_data, ball_kin, team_dynamics, p_dead, blocks, pitch, fps=fps
    )
    p_home = bundle.predict_team_posterior(team_ft["X_team"], blocks)

    turn_ft = extract_transition_features(ball_data, ball_kin, p_dead, p_home, blocks, fps=fps)
    p_turn = bundle.predict_turn_posterior(turn_ft["X_turn"], blocks)

    if verbose:
        print("[5/5] Calibrating Empirical Bayes tag LLRs & decoding exact Viterbi sequence...")
    events = load_smart_tags(game=game, data_dir=data_dir)
    tag_adj = compute_empirical_bayes_tag_adjustments(
        events, tac["team_names"], ball_data, p_dead, p_home, blocks, pitch, fps=fps
    )
    labels = decode_match_sequence(
        p_dead, p_home, p_turn, ball_data, blocks, bundle.calibration, tag_adjustments=tag_adj, fps=fps
    )

    elapsed = time.time() - t0
    if verbose:
        print(f"Done in {elapsed:.1f}s.")

    stats = {
        "elapsed_s": elapsed,
        "tac_ball_pct": 100.0 * tac_vis_graded / total_graded,
        "fused_ball_pct": 100.0 * fused_vis_graded / total_graded,
        "kin_ball_pct": 100.0 * kin_vis_graded / total_graded,
    }
    return labels, stats
