"""Collective Team Movement, Spatial Shape, Territorial Tilt, and Pitch Control.

Computes permutation-invariant team-level movement and spatial features from anonymous
player trajectories:
  1. RMS player speed, walking/standing player fraction (< 1.45 m/s), and sprinting fraction (> 3.2 m/s)
  2. Collective centre-of-mass speed and directional velocity alignment
  3. Longitudinal attacking drift along each team's attacking direction
  4. Outfield lateral width expansion (excluding goalkeepers) and territorial tilt
  5. Pre-kickoff / post-goal half-field formation separation detector
  6. Velocity-projected continuous pitch control around the ball
"""

from typing import Dict, List, Tuple
import numpy as np

HOME = 0
AWAY = 1


def moving_average_1d(arr: np.ndarray, half_win: int) -> np.ndarray:
    """Fast O(N) centred moving average handling NaNs gracefully."""
    if half_win <= 0:
        return arr.copy()
    valid = np.isfinite(arr)
    vals = np.where(valid, arr, 0.0).astype(np.float64)
    cs_v = np.concatenate(([0.0], np.cumsum(vals)))
    cs_c = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    idx = np.arange(len(arr))
    lo = np.maximum(0, idx - half_win)
    hi = np.minimum(len(arr), idx + half_win + 1)
    cnt = cs_c[hi] - cs_c[lo]
    out = np.full(len(arr), np.nan, dtype=np.float32)
    ok = cnt > 0
    out[ok] = ((cs_v[hi[ok]] - cs_v[lo[ok]]) / cnt[ok]).astype(np.float32)
    return out


def compute_team_dynamics(
    player_xy: np.ndarray,
    player_speed: np.ndarray,
    player_vel: np.ndarray,
    slot_teams: np.ndarray,
    kin_ball_xy: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    pitch: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    """Compute collective movement, spatial width, territorial, and pitch-control features per frame."""
    n_frames = len(player_xy)
    length, width = pitch
    h_mask = slot_teams == HOME
    a_mask = slot_teams == AWAY

    # 1. Infer each team's attacking x-direction (+1 or -1) per half from median x position
    attack_dir_h = np.zeros(n_frames, dtype=np.float32)
    for _, s_f, e_f in half_blocks:
        with np.errstate(all="ignore"):
            med_xh = float(np.nanmedian(player_xy[s_f : e_f + 1, h_mask, 0]))
            med_xa = float(np.nanmedian(player_xy[s_f : e_f + 1, a_mask, 0]))
        attack_dir_h[s_f : e_f + 1] = 1.0 if med_xh < med_xa else -1.0

    # 2. Collective Player Speed & Walking / Running Fractions
    vis_spd = np.isfinite(player_speed)
    n_vis_spd = vis_spd.sum(axis=1)
    spd_zero = np.where(vis_spd, player_speed, 0.0)

    rms_player_speed = np.zeros(n_frames, dtype=np.float32)
    ok_s = n_vis_spd > 0
    rms_player_speed[ok_s] = np.sqrt((spd_zero[ok_s] ** 2).sum(axis=1) / n_vis_spd[ok_s])

    walk_cnt = (vis_spd & (player_speed < 1.45)).sum(axis=1)
    walking_ratio = np.ones(n_frames, dtype=np.float32)
    walking_ratio[ok_s] = walk_cnt[ok_s] / n_vis_spd[ok_s]

    run_cnt = (vis_spd & (player_speed > 3.2)).sum(axis=1)
    running_ratio = np.zeros(n_frames, dtype=np.float32)
    running_ratio[ok_s] = run_cnt[ok_s] / n_vis_spd[ok_s]

    n_visible_players = np.isfinite(player_xy[:, :, 0]).sum(axis=1).astype(np.int16)

    # 3. Collective Centre-of-Mass Drift & Velocity Alignment
    vis_vel = np.isfinite(player_vel[:, :, 0]) & np.isfinite(player_vel[:, :, 1])
    n_vis_vel = vis_vel.sum(axis=1)
    vx_z = np.where(vis_vel, player_vel[:, :, 0], 0.0)
    vy_z = np.where(vis_vel, player_vel[:, :, 1], 0.0)
    vmag_z = np.hypot(vx_z, vy_z)

    sum_vx = vx_z.sum(axis=1)
    sum_vy = vy_z.sum(axis=1)
    sum_vmag = vmag_z.sum(axis=1)

    centroid_speed = np.zeros(n_frames, dtype=np.float32)
    centroid_vx = np.zeros(n_frames, dtype=np.float32)
    velocity_alignment = np.zeros(n_frames, dtype=np.float32)

    ok_v = n_vis_vel >= 6
    centroid_vx[ok_v] = sum_vx[ok_v] / n_vis_vel[ok_v]
    centroid_vy = np.zeros(n_frames, dtype=np.float32)
    centroid_vy[ok_v] = sum_vy[ok_v] / n_vis_vel[ok_v]
    centroid_speed[ok_v] = np.hypot(centroid_vx[ok_v], centroid_vy[ok_v])
    velocity_alignment[ok_v] = np.hypot(sum_vx[ok_v], sum_vy[ok_v]) / (sum_vmag[ok_v] + 1e-5)

    # Directed longitudinal drift (>0 means collective movement toward Away's goal -> H attacking)
    attacking_drift_h = moving_average_1d(centroid_vx * attack_dir_h, half_win=18)

    # 4. Outfield Lateral Width & Territorial Progression (excluding Goalkeepers)
    def compute_team_width_and_territory(pxy_team: np.ndarray, att_dir: np.ndarray):
        """Return (transverse_std_y, outfield_mean_progression, opponent_half_count)."""
        x = pxy_team[:, :, 0]
        y = pxy_team[:, :, 1]
        vis = np.isfinite(x)
        cnt = vis.sum(axis=1)

        prog = np.where(att_dir[:, None] >= 0, x, length - x)
        prog_safe = np.where(vis, prog, np.inf)
        # Identify goalkeeper as the deepest player (min progression) when >= 6 players are visible
        gk_idx = np.argmin(prog_safe, axis=1)
        outfield_vis = vis.copy()
        rows_gk = np.flatnonzero(cnt >= 6)
        outfield_vis[rows_gk, gk_idx[rows_gk]] = False
        out_cnt = outfield_vis.sum(axis=1)

        ok_o = out_cnt >= 5
        prog_z = np.where(outfield_vis, prog, 0.0)
        mean_prog = np.full(n_frames, length * 0.5, dtype=np.float32)
        mean_prog[ok_o] = (prog_z[ok_o].sum(axis=1) / out_cnt[ok_o]).astype(np.float32)

        y_z = np.where(outfield_vis, y, 0.0)
        mean_y = np.zeros(n_frames, dtype=np.float32)
        mean_y[ok_o] = y_z[ok_o].sum(axis=1) / out_cnt[ok_o]
        dy2 = np.where(outfield_vis, (y - mean_y[:, None]) ** 2, 0.0)
        std_y = np.full(n_frames, 10.0, dtype=np.float32)
        std_y[ok_o] = np.sqrt(dy2[ok_o].sum(axis=1) / out_cnt[ok_o]).astype(np.float32)

        invading = (vis & (prog > (0.5 * length + 1.8))).sum(axis=1).astype(np.int16)
        return std_y, mean_prog, invading

    std_y_h, prog_h, inv_h = compute_team_width_and_territory(
        player_xy[:, h_mask, :], attack_dir_h
    )
    std_y_a, prog_a, inv_a = compute_team_width_and_territory(
        player_xy[:, a_mask, :], -attack_dir_h
    )

    # Possessing team expands laterally across the pitch while defending team compacts centrally
    width_diff_h = moving_average_1d((std_y_h - std_y_a) / 4.5, half_win=30)
    territory_h = moving_average_1d(((prog_h - prog_a) / (0.35 * length)), half_win=30)

    # 5. Pre-Kickoff / Post-Goal Half-Field Formation Detector:
    # Neither team crosses into the opponent's half (<= 1 player) while tempo is low or players are entering.
    kickoff_formation_mask = (
        ((inv_h + inv_a) <= 1) & ((walking_ratio > 0.65) | (n_visible_players < 14))
    )

    # 6. Velocity-Projected Continuous Pitch Control around Ball Position
    pitch_control_h = np.zeros(n_frames, dtype=np.float32)
    has_kin_ball = np.isfinite(kin_ball_xy[:, 0])
    kb_idx = np.flatnonzero(has_kin_ball)

    tau_react = 0.45  # seconds forward velocity projection
    sigma_pc = 4.8    # spatial influence radius in metres

    chunk = 10000
    for start in range(0, len(kb_idx), chunk):
        idx = kb_idx[start : start + chunk]
        b = kin_ball_xy[idx]

        px = player_xy[idx]
        pv = np.where(np.isfinite(player_vel[idx]), player_vel[idx], 0.0)
        proj = px + tau_react * pv

        d2 = (proj[:, :, 0] - b[:, None, 0]) ** 2 + (proj[:, :, 1] - b[:, None, 1]) ** 2
        w = np.where(np.isfinite(d2), np.exp(-0.5 * d2 / (sigma_pc ** 2)), 0.0)

        wh = w[:, h_mask].sum(axis=1)
        wa = w[:, a_mask].sum(axis=1)
        pitch_control_h[idx] = ((wh - wa) / (wh + wa + 0.25)).astype(np.float32)

    return {
        "attack_dir_h": attack_dir_h,
        "rms_player_speed": rms_player_speed,
        "rms_speed_smooth": moving_average_1d(rms_player_speed, half_win=25),
        "walking_ratio": walking_ratio,
        "walking_ratio_smooth": moving_average_1d(walking_ratio, half_win=25),
        "running_ratio_smooth": moving_average_1d(running_ratio, half_win=25),
        "centroid_speed_smooth": moving_average_1d(centroid_speed, half_win=18),
        "velocity_alignment_smooth": moving_average_1d(velocity_alignment, half_win=18),
        "attacking_drift_h": attacking_drift_h,
        "width_diff_h": width_diff_h,
        "territory_h": territory_h,
        "n_visible_players": n_visible_players,
        "kickoff_formation_mask": kickoff_formation_mask,
        "pitch_control_h": pitch_control_h,
    }
