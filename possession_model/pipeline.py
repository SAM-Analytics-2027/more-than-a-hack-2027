"""End-to-End Ball Possession Inference Pipeline.

Combines:
  1. Dual-camera player-constellation ball synchronization & flight trajectory reconstruction
  2. Verified controlled possession vs uncontrolled clearances/deflections/duels + occlusion continuity
  3. Collective team movement & spatial dynamics (RMS speed, walking ratio, drift, width, pitch control)
  4. Smart Tagging tactical priors & set-piece boundary-exit / restart localization
  5. Exact duration-constrained Viterbi dynamic-programming sequence decoding
"""

import time
from typing import Dict, Optional, Tuple
import numpy as np

from games import GAME, GAMES, half_video_blocks
from possession_model.data_loader import (
    load_atd_arrays,
    load_smart_tags,
    sync_broadcast_to_tactical,
)
from possession_model.ball_tracking import (
    detect_ball_control_and_passes,
    reconstruct_ball_trajectory,
)
from possession_model.team_dynamics import compute_team_dynamics
from possession_model.tactical_priors import compute_tactical_tag_priors
from possession_model.sequence_decoder import AWAY, DEAD, HOME, decode_possession_sequence


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 + exp(x))."""
    return np.logaddexp(0.0, x)


def compute_frame_and_transition_costs(
    ball_data: Dict[str, np.ndarray],
    ball_control: Dict[str, np.ndarray],
    team_dynamics: Dict[str, np.ndarray],
    tactical: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute factorized per-frame negative log-likelihood costs C_frame(t, s) = -log P(s | x_t)
    and context-dependent transition penalties:

        P(D | x_t) = sigmoid(z_dead)
        P(H | x_t) = (1 - P(D | x_t)) * sigmoid(+delta_team)
        P(A | x_t) = (1 - P(D | x_t)) * sigmoid(-delta_team)
    """
    n_frames = len(ball_data["obs_mask"])

    obs_mask = ball_data["obs_mask"]
    kin_speed = np.where(np.isfinite(ball_data["kin_speed"]), ball_data["kin_speed"], 0.0)
    net_disp_1s = ball_data["net_disp_1s"]
    oob = ball_data["out_of_bounds"]
    exit_sig = ball_data["boundary_exit_signal"]
    flight = ball_data["is_flight_bridge"]
    in_occlusion = ball_control["in_same_team_occlusion"]

    # --- 1. Dead-Ball vs Live-Play Logit z_dead (z_dead > 0 => Dead ball favored) ---
    walk_ratio = np.where(
        np.isfinite(team_dynamics["walking_ratio_smooth"]),
        team_dynamics["walking_ratio_smooth"],
        0.7,
    )
    rms_speed = np.where(
        np.isfinite(team_dynamics["rms_speed_smooth"]),
        team_dynamics["rms_speed_smooth"],
        1.4,
    )
    centroid_speed = np.where(
        np.isfinite(team_dynamics["centroid_speed_smooth"]),
        team_dynamics["centroid_speed_smooth"],
        0.3,
    )
    run_ratio = np.where(
        np.isfinite(team_dynamics["running_ratio_smooth"]),
        team_dynamics["running_ratio_smooth"],
        0.05,
    )

    low_tempo_score = (
        np.clip((walk_ratio - 0.59) / 0.18, -1.3, 1.6)
        + np.clip((1.88 - rms_speed) / 0.58, -1.4, 1.5)
    )

    collective_motion_alive = (
        np.clip((centroid_speed - 0.78) / 0.58, 0.0, 1.5)
        + np.clip((run_ratio - 0.12) / 0.13, 0.0, 1.1)
    )

    max_pass_bridge = np.maximum(ball_control["pass_bridge_h"], ball_control["pass_bridge_a"])
    max_control = np.maximum(ball_control["control_score_h"], ball_control["control_score_a"])
    no_tagged_stoppage = tactical["stoppage_prior"] < 0.1

    ball_rolling_in_pitch = obs_mask & (~oob) & ((kin_speed > 2.0) | (net_disp_1s > 1.8))
    players_active = (rms_speed > 1.75) & (walk_ratio < 0.62)

    ball_active = (
        1.20 * ball_rolling_in_pitch.astype(np.float32)
        + 0.85 * (flight & players_active).astype(np.float32)
        + 0.80 * np.where(players_active, np.clip(max_pass_bridge, 0.0, 1.4), 0.0)
        + 0.85 * (in_occlusion & players_active).astype(np.float32)
        + 0.55 * np.where(no_tagged_stoppage & obs_mask & (~oob) & (max_control > 0.35), 1.0, 0.0).astype(np.float32)
    )

    stationary_ball_setup = (
        obs_mask & (kin_speed < 1.1) & (net_disp_1s < 0.9) & (walk_ratio > 0.64) & (rms_speed < 1.72)
    ).astype(np.float32)

    unseen_ball_stoppage = (
        (~obs_mask) & (~flight) & (~in_occlusion) & (low_tempo_score > 0.25)
    ).astype(np.float32) * (0.68 + 0.55 * np.maximum(0.0, low_tempo_score))

    z_dead = (
        0.95 * low_tempo_score
        + unseen_ball_stoppage
        + 0.90 * stationary_ball_setup
        + 1.20 * tactical["stoppage_prior"]
        + 3.10 * oob.astype(np.float32)
        + 1.55 * exit_sig
        + 2.85 * team_dynamics["kickoff_formation_mask"].astype(np.float32)
        - 1.15 * collective_motion_alive
        - 1.10 * ball_active
        - 0.50 * tactical["active_phase_score"]
        - 0.45
    ).astype(np.float64)
    z_dead = np.clip(z_dead, -8.0, 8.0)

    # --- 2. Team Possession Logit delta_team (delta_team > 0 => Home favored over Away) ---
    ball_control_diff_h = (
        3.35 * (ball_control["control_score_h"] - ball_control["control_score_a"])
        + 2.25 * (ball_control["pass_bridge_h"] - ball_control["pass_bridge_a"])
        + 1.90 * (ball_control["occlusion_bridge_h"] - ball_control["occlusion_bridge_a"])
        + 0.75 * (ball_control["proximity_score_h"] - ball_control["proximity_score_a"])
    )

    max_occlusion = np.maximum(ball_control["occlusion_bridge_h"], ball_control["occlusion_bridge_a"])
    has_strong_ball_evidence = (max_control > 0.35) | (max_pass_bridge > 0.75) | (max_occlusion > 0.5)
    tag_weight = np.where(has_strong_ball_evidence, 0.30, 0.85).astype(np.float32)

    team_context_diff_h = (
        0.80 * team_dynamics["pitch_control_h"]
        + 0.42 * np.clip(team_dynamics["width_diff_h"], -1.3, 1.3)
        + 0.35 * np.clip(team_dynamics["attacking_drift_h"] / 2.2, -1.1, 1.1)
        + 0.18 * np.clip(team_dynamics["territory_h"], -1.1, 1.1)
        + tag_weight * tactical["team_prior_h"]
    )

    delta_team = np.clip((ball_control_diff_h + team_context_diff_h).astype(np.float64), -8.0, 8.0)

    c_alive = _softplus(z_dead)
    c_dead = _softplus(-z_dead)
    c_h = c_alive + _softplus(-delta_team)
    c_a = c_alive + _softplus(+delta_team)

    frame_costs = np.empty((n_frames, 3), dtype=np.float64)
    frame_costs[:, HOME] = c_h
    frame_costs[:, AWAY] = c_a
    frame_costs[:, DEAD] = c_dead

    # --- 3. Context-Dependent State-Transition Penalties ---
    ctrl = ball_control["verified_control_mask"]
    ctrl_dilated = np.convolve(ctrl.astype(np.int32), np.ones(7, dtype=np.int32), mode="same") > 0

    turnover_cost = np.where(
        ctrl_dilated,
        13.5,
        np.where(in_occlusion, 54.0, 42.0),
    ).astype(np.float64)

    stop_imp_dil = (
        np.convolve(tactical["stop_impulse"].astype(np.int32), np.ones(9, dtype=np.int32), mode="same") > 0
    )
    stop_cue = stop_imp_dil | oob | (exit_sig > 0.5) | (tactical["stoppage_prior"] > 2.0)
    stoppage_entry_cost = np.where(stop_cue, 5.5, 23.0).astype(np.float64)

    restart_cue = (
        np.convolve(tactical["restart_impulse"].astype(np.int32), np.ones(11, dtype=np.int32), mode="same") > 0
    )
    restart_cost = np.where(restart_cue, 2.0, 18.5).astype(np.float64)

    return frame_costs, turnover_cost, stoppage_entry_cost, restart_cost


def run_possession_inference(
    game: str = GAME,
    data_dir: Optional[str] = None,
    use_broadcast: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Execute the full ball possession inference pipeline on a match."""
    t0 = time.time()
    if verbose:
        print(f"[1/5] Loading Tactical ATD & metadata for {game}...")
    tac = load_atd_arrays(game=game, data_dir=data_dir, source="tactical")
    n_frames = len(tac["ball_xy"])
    pitch = GAMES[game]["pitch"]
    fps = float(GAMES[game]["fps"])
    blocks = half_video_blocks(game)

    tac_vis_graded = 0
    total_graded = 0
    for _, s, e in blocks:
        tac_vis_graded += int(np.isfinite(tac["ball_xy"][s : e + 1, 0]).sum())
        total_graded += (e - s + 1)

    brd_ball = None
    if use_broadcast:
        if verbose:
            print("[2/5] Synchronizing Broadcast camera & filtering TV replays...")
        sync_res = sync_broadcast_to_tactical(tac, game=game, data_dir=data_dir)
        if sync_res is not None:
            brd_ball = sync_res["brd_ball_xy"]

    if verbose:
        print("[3/5] Reconstructing ball trajectory & collective team dynamics...")
    ball_data = reconstruct_ball_trajectory(tac["ball_xy"], brd_ball, pitch=pitch, fps=fps)

    fused_vis_graded = 0
    kin_vis_graded = 0
    for _, s, e in blocks:
        fused_vis_graded += int(ball_data["obs_mask"][s : e + 1].sum())
        kin_vis_graded += int(np.isfinite(ball_data["kin_xy"][s : e + 1, 0]).sum())

    if verbose:
        print(
            f"      Graded ball coverage: Tactical {100.0 * tac_vis_graded / total_graded:.1f}% "
            f"-> Dual-Cam Observed {100.0 * fused_vis_graded / total_graded:.1f}% "
            f"-> With Flight Bridges {100.0 * kin_vis_graded / total_graded:.1f}%"
        )

    team_dynamics = compute_team_dynamics(
        tac["player_xy"],
        tac["player_speed"],
        tac["player_vel"],
        tac["slot_teams"],
        ball_data["kin_xy"],
        blocks,
        pitch,
    )

    if verbose:
        print("[4/5] Detecting ball control, pass bridges & tactical tag priors...")
    ball_control = detect_ball_control_and_passes(
        tac["player_xy"],
        tac["player_vel"],
        tac["slot_teams"],
        ball_data,
        team_dynamics=team_dynamics,
        fps=fps,
    )

    events = load_smart_tags(game=game, data_dir=data_dir)
    tactical = compute_tactical_tag_priors(
        events,
        tac["team_names"],
        tac["player_xy"],
        tac["slot_teams"],
        ball_data,
        team_dynamics,
        blocks,
        pitch,
        fps=fps,
    )

    if verbose:
        print("[5/5] Decoding optimal possession sequence per half (Viterbi DP)...")
    frame_costs, turnover_cost, stoppage_entry_cost, restart_cost = compute_frame_and_transition_costs(
        ball_data, ball_control, team_dynamics, tactical
    )

    labels = np.full(n_frames, "X", dtype="<U1")
    code_map = np.array(["H", "A", "D"], dtype="<U1")

    for half_num, s_f, e_f in blocks:
        init_offset = np.zeros(3, dtype=np.float64)
        if tactical["stoppage_prior"][s_f] > 2.0:
            init_offset[HOME] = 20.0
            init_offset[AWAY] = 20.0

        states = decode_possession_sequence(
            frame_costs[s_f : e_f + 1],
            turnover_cost[s_f : e_f + 1],
            stoppage_entry_cost[s_f : e_f + 1],
            restart_cost[s_f : e_f + 1],
            min_possession_frames=45,  # 1.80 s minimum controlled possession duration
            min_stoppage_frames=85,    # 3.40 s minimum dead-ball stoppage duration
            initial_cost_offset=init_offset,
        )
        labels[s_f : e_f + 1] = code_map[states]

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
