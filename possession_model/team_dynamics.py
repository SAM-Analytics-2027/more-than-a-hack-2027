"""Collective 22-Player Kinematics, Team Spatial Geometry, and Continuous Pitch Control.

Computes permutation-invariant team-level and match-level kinematic descriptors from
anonymous player trajectories in a single vectorized pass.
"""

from typing import Dict, List, Tuple
import numpy as np

from possession_model.signal_ops import AWAY, HOME, box_mean_blocks

# Standard biomechanical human locomotion gait transitions (m/s):
# walk-to-jog ~ 1.5 m/s (5.4 km/h), jog-to-run ~ 3.3 m/s (11.9 km/h).
WALK_GAIT_SPEED_MPS = 1.50
RUN_GAIT_SPEED_MPS = 3.30
# Human visual-motor reaction time (s) for off-ball spatial pitch control (Spearman 2017).
REACTION_TIME_S = 0.50


def compute_team_dynamics(
    player_xy: np.ndarray,
    player_speed: np.ndarray,
    player_vel: np.ndarray,
    slot_teams: np.ndarray,
    kin_ball_xy: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    pitch: Tuple[float, float],
    fps: float = 25.0,
) -> Dict[str, np.ndarray]:
    """Compute collective movement, outfield spatial geometry, set-piece clustering, and pitch control."""
    n_frames = len(player_xy)
    length, width = pitch
    h_mask = slot_teams == HOME
    a_mask = slot_teams == AWAY

    # 1. Infer each team's attacking x-direction (+1 or -1) per half from median x position
    with np.errstate(all="ignore"):
        g_med_xh = float(np.nanmedian(player_xy[:, h_mask, 0]))
        g_med_xa = float(np.nanmedian(player_xy[:, a_mask, 0]))
    default_dir = 1.0 if g_med_xh < g_med_xa else -1.0
    attack_dir_h = np.full(n_frames, default_dir, dtype=np.float32)
    for _, s_f, e_f in half_blocks:
        with np.errstate(all="ignore"):
            med_xh = float(np.nanmedian(player_xy[s_f : e_f + 1, h_mask, 0]))
            med_xa = float(np.nanmedian(player_xy[s_f : e_f + 1, a_mask, 0]))
        attack_dir_h[s_f : e_f + 1] = 1.0 if med_xh < med_xa else -1.0

    # 2. Match-wide 22-player speed moments & gait fractions
    vis_spd = np.isfinite(player_speed)
    n_vis_spd = vis_spd.sum(axis=1)
    spd_zero = np.where(vis_spd, player_speed, 0.0)
    ok_s = n_vis_spd > 0

    rms_speed = np.zeros(n_frames, dtype=np.float32)
    rms_speed[ok_s] = np.sqrt((spd_zero[ok_s] ** 2).sum(axis=1) / n_vis_spd[ok_s])

    walk_ratio = np.ones(n_frames, dtype=np.float32)
    walk_ratio[ok_s] = (vis_spd & (player_speed < WALK_GAIT_SPEED_MPS)).sum(axis=1)[ok_s] / n_vis_spd[ok_s]

    run_ratio = np.zeros(n_frames, dtype=np.float32)
    run_ratio[ok_s] = (vis_spd & (player_speed > RUN_GAIT_SPEED_MPS)).sum(axis=1)[ok_s] / n_vis_spd[ok_s]

    # Upper-tail player speed (85th percentile approximated cleanly via order statistics)
    spd_neg = np.where(vis_spd, player_speed, -1.0)
    k85 = max(0, spd_neg.shape[1] - 4)
    p85_speed = np.maximum(0.0, np.partition(spd_neg, kth=k85, axis=1)[:, k85]).astype(np.float32)

    # 3. Collective centre-of-mass velocity & directional alignment ||sum v_i|| / sum ||v_i||
    vis_vel = np.isfinite(player_vel[:, :, 0]) & np.isfinite(player_vel[:, :, 1])
    n_vis_vel = vis_vel.sum(axis=1)
    vx_z = np.where(vis_vel, player_vel[:, :, 0], 0.0)
    vy_z = np.where(vis_vel, player_vel[:, :, 1], 0.0)
    vmag_z = np.hypot(vx_z, vy_z)

    sum_vx = vx_z.sum(axis=1)
    sum_vy = vy_z.sum(axis=1)
    sum_vmag = vmag_z.sum(axis=1)

    centroid_speed = np.zeros(n_frames, dtype=np.float32)
    vel_alignment = np.zeros(n_frames, dtype=np.float32)
    ok_v = n_vis_vel >= 6
    centroid_speed[ok_v] = np.hypot(sum_vx[ok_v], sum_vy[ok_v]) / n_vis_vel[ok_v]
    vel_alignment[ok_v] = np.hypot(sum_vx[ok_v], sum_vy[ok_v]) / (sum_vmag[ok_v] + 1e-5)

    # 4. Per-team outfield spatial moments (excluding deepest player = goalkeeper) & attacking drift
    def _team_shape(pxy_t: np.ndarray, pvel_t: np.ndarray, att_dir: np.ndarray):
        x = pxy_t[:, :, 0]
        y = pxy_t[:, :, 1]
        vis = np.isfinite(x)
        cnt = vis.sum(axis=1).astype(np.float32)

        prog = np.where(att_dir[:, None] >= 0, x, length - x)
        prog_safe = np.where(vis, prog, np.inf)
        gk_idx = np.argmin(prog_safe, axis=1)
        out_vis = vis.copy()
        has_gk = np.flatnonzero(cnt >= 6)
        out_vis[has_gk, gk_idx[has_gk]] = False
        out_cnt = out_vis.sum(axis=1)
        ok_o = out_cnt >= 4

        mean_prog = np.full(n_frames, 0.5 * length, dtype=np.float32)
        mean_prog[ok_o] = (np.where(out_vis, prog, 0.0)[ok_o].sum(axis=1) / out_cnt[ok_o]).astype(np.float32)

        dp2 = np.where(out_vis, (prog - mean_prog[:, None]) ** 2, 0.0)
        std_x = np.full(n_frames, length / 8.0, dtype=np.float32)
        std_x[ok_o] = np.sqrt(dp2[ok_o].sum(axis=1) / out_cnt[ok_o]).astype(np.float32)

        mean_y = np.full(n_frames, 0.5 * width, dtype=np.float32)
        mean_y[ok_o] = (np.where(out_vis, y, 0.0)[ok_o].sum(axis=1) / out_cnt[ok_o]).astype(np.float32)

        dy2 = np.where(out_vis, (y - mean_y[:, None]) ** 2, 0.0)
        std_y = np.full(n_frames, width / 6.0, dtype=np.float32)
        std_y[ok_o] = np.sqrt(dy2[ok_o].sum(axis=1) / out_cnt[ok_o]).astype(np.float32)

        # Team longitudinal velocity along its own attacking direction
        vv = np.isfinite(pvel_t[:, :, 0])
        nv = vv.sum(axis=1)
        ok_tv = nv >= 3
        drift = np.zeros(n_frames, dtype=np.float32)
        vx_t = np.where(vv, pvel_t[:, :, 0], 0.0)
        drift[ok_tv] = ((vx_t[ok_tv].sum(axis=1) / nv[ok_tv]) * att_dir[ok_tv]).astype(np.float32)

        # Count of players strictly inside opponent half (beyond centre circle tolerance)
        opp_half_cnt = (vis & (prog > 0.5 * length + 1.5)).sum(axis=1).astype(np.float32)
        return cnt, mean_prog, std_x, std_y, drift, opp_half_cnt

    n_vis_h, prog_h, std_x_h, std_y_h, drift_h, inv_h = _team_shape(
        player_xy[:, h_mask, :], player_vel[:, h_mask, :], attack_dir_h
    )
    n_vis_a, prog_a, std_x_a, std_y_a, drift_a, inv_a = _team_shape(
        player_xy[:, a_mask, :], player_vel[:, a_mask, :], -attack_dir_h
    )

    # 5. Geometric Set-Piece Spatial Concentration (FIFA penalty area 16.5m x 40.32m, touchline, centre)
    px_all = player_xy[:, :, 0]
    py_all = player_xy[:, :, 1]
    v_all = np.isfinite(px_all)
    n_vis_tot = v_all.sum(axis=1).astype(np.float32)

    # FIFA penalty box dimensions: 16.5 m from goal line, +/- 20.16 m from goal centre (0.5 * width)
    in_box_l = v_all & (px_all <= 16.5) & (np.abs(py_all - 0.5 * width) <= 20.16)
    in_box_r = v_all & (px_all >= length - 16.5) & (np.abs(py_all - 0.5 * width) <= 20.16)
    max_in_penalty_box = np.maximum(in_box_l.sum(axis=1), in_box_r.sum(axis=1)).astype(np.float32)

    near_touchline = (v_all & ((py_all <= 3.0) | (py_all >= width - 3.0))).sum(axis=1).astype(np.float32)
    half_separation_frac = 1.0 - np.minimum(1.0, (inv_h + inv_a) / np.maximum(1.0, n_vis_tot))

    # 6. Continuous Off-Ball Pitch Control using geometric 22-player Voronoi radius
    voronoi_radius = float(0.5 * np.sqrt((length * width) / (22.0 * np.pi)))
    pitch_control_h = np.zeros(n_frames, dtype=np.float32)
    kb_idx = np.flatnonzero(np.isfinite(kin_ball_xy[:, 0]))

    chunk = 12000
    for s0 in range(0, len(kb_idx), chunk):
        idx = kb_idx[s0 : s0 + chunk]
        b = kin_ball_xy[idx]
        pv = np.where(np.isfinite(player_vel[idx]), player_vel[idx], 0.0)
        proj = player_xy[idx] + REACTION_TIME_S * pv
        d2 = (proj[:, :, 0] - b[:, None, 0]) ** 2 + (proj[:, :, 1] - b[:, None, 1]) ** 2
        w = np.where(np.isfinite(d2), np.exp(-0.5 * d2 / (voronoi_radius ** 2)), 0.0)
        wh = w[:, h_mask].sum(axis=1)
        wa = w[:, a_mask].sum(axis=1)
        pitch_control_h[idx] = ((wh - wa) / (wh + wa + 0.1)).astype(np.float32)

    w1s = int(round(1.0 * fps))
    return {
        "attack_dir_h": attack_dir_h,
        "rms_player_speed": rms_speed,
        "rms_speed_smooth": box_mean_blocks(rms_speed, w1s, w1s, half_blocks),
        "p85_player_speed": p85_speed,
        "walking_ratio": walk_ratio,
        "walking_ratio_smooth": box_mean_blocks(walk_ratio, w1s, w1s, half_blocks, fill=1.0),
        "running_ratio": run_ratio,
        "centroid_speed": centroid_speed,
        "velocity_alignment": vel_alignment,
        "n_vis_tot": n_vis_tot,
        "n_vis_h": n_vis_h,
        "n_vis_a": n_vis_a,
        "prog_h": prog_h,
        "prog_a": prog_a,
        "std_x_h": std_x_h,
        "std_x_a": std_x_a,
        "std_y_h": std_y_h,
        "std_y_a": std_y_a,
        "drift_h": drift_h,
        "drift_a": drift_a,
        "max_in_penalty_box": max_in_penalty_box,
        "near_touchline": near_touchline,
        "half_separation_frac": half_separation_frac,
        "pitch_control_h": pitch_control_h,
    }
