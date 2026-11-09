"""Physical Ball Trajectory Reconstruction and Continuous Player-Ball Kinematics.

Computes dual-camera ball trajectories, ballistic air-flight interpolation, pitch-boundary
exit kinematics, and continuous player-ball interaction kernels without heuristic rule loops.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from possession_model.signal_ops import AWAY, HOME

# Physical upper bound for football flight speed (~151 km/h); faster 1-frame jumps are optical spikes.
MAX_BALL_SPEED_MPS = 42.0
# Physical upper bound for interpolated pass flight speed (~115 km/h) over short unseen air arcs (<= 1.4 s).
MAX_PASS_ARC_SPEED_MPS = 32.0
MAX_PASS_ARC_SECONDS = 1.40
# FIFA ball radius (0.11 m) + optical line/homography projection tolerance (0.34 m).
PITCH_LINE_TOLERANCE_M = 0.45
# Characteristic human ball-reach scale (m) and ground-pass speed scale (m/s) for continuous kernels.
REACH_SCALE_M = 1.60
PASS_SPEED_SCALE_MPS = 16.0


def despike_ball_track(ball_xy: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """Remove isolated 1-frame and 2-frame optical false detections exceeding MAX_BALL_SPEED_MPS."""
    xy = ball_xy.copy()
    max_step = MAX_BALL_SPEED_MPS / fps

    for _ in range(2):
        vis = np.flatnonzero(np.isfinite(xy[:, 0]))
        if len(vis) < 3:
            break
        prev_i = vis[:-2]
        curr_i = vis[1:-1]
        next_i = vis[2:]

        dt1 = curr_i - prev_i
        dt2 = next_i - curr_i
        dt_skip = next_i - prev_i

        d1 = np.hypot(xy[curr_i, 0] - xy[prev_i, 0], xy[curr_i, 1] - xy[prev_i, 1])
        d2 = np.hypot(xy[next_i, 0] - xy[curr_i, 0], xy[next_i, 1] - xy[curr_i, 1])
        d_skip = np.hypot(xy[next_i, 0] - xy[prev_i, 0], xy[next_i, 1] - xy[prev_i, 1])

        spike = (
            (dt1 <= 4)
            & (dt2 <= 4)
            & (d1 > max_step * dt1)
            & (d2 > max_step * dt2)
            & (d_skip <= max_step * dt_skip * 1.2)
        )
        if not spike.any():
            break
        xy[curr_i[spike]] = np.nan

    return xy


def reconstruct_ball_trajectory(
    tac_ball_xy: np.ndarray,
    brd_ball_xy: Optional[np.ndarray] = None,
    pitch: Tuple[float, float] = (105.0, 68.0),
    fps: float = 25.0,
) -> Dict[str, np.ndarray]:
    """Fuse Tactical and live-synced Broadcast ball tracks, interpolate short ballistic arcs,
    and compute velocity, vector impulse, 1-second displacement, and pitch-line crossing instants.
    """
    length, width = pitch
    tac_clean = despike_ball_track(tac_ball_xy, fps=fps)
    fused = tac_clean.copy()

    if brd_ball_xy is not None:
        brd_clean = despike_ball_track(brd_ball_xy, fps=fps)
        tac_ok = np.isfinite(tac_clean[:, 0])
        brd_ok = np.isfinite(brd_clean[:, 0])

        only_brd = (~tac_ok) & brd_ok
        fused[only_brd] = brd_clean[only_brd]

        both = tac_ok & brd_ok
        agree = np.zeros(len(fused), dtype=bool)
        agree[both] = (
            np.hypot(tac_clean[both, 0] - brd_clean[both, 0], tac_clean[both, 1] - brd_clean[both, 1])
            < 2.0 * REACH_SCALE_M
        )
        # Tactical fixed-camera homography has ~2x higher precision than panning Broadcast homography (4:1 precision weight)
        fused[agree] = 0.80 * tac_clean[agree] + 0.20 * brd_clean[agree]
        fused = despike_ball_track(fused, fps=fps)

    n_frames = len(fused)
    obs_mask = np.isfinite(fused[:, 0])

    out_of_bounds = np.zeros(n_frames, dtype=bool)
    out_of_bounds[obs_mask] = (
        (fused[obs_mask, 0] < -PITCH_LINE_TOLERANCE_M)
        | (fused[obs_mask, 0] > length + PITCH_LINE_TOLERANCE_M)
        | (fused[obs_mask, 1] < -PITCH_LINE_TOLERANCE_M)
        | (fused[obs_mask, 1] > width + PITCH_LINE_TOLERANCE_M)
    )

    kin_xy = fused.copy()
    is_flight_bridge = np.zeros(n_frames, dtype=bool)
    vis_idx = np.flatnonzero(obs_mask)
    max_arc_frames = int(round(MAX_PASS_ARC_SECONDS * fps))

    for k in range(len(vis_idx) - 1):
        i0 = int(vis_idx[k])
        i1 = int(vis_idx[k + 1])
        gap = i1 - i0 - 1
        if gap <= 0 or gap > max_arc_frames or out_of_bounds[i0] or out_of_bounds[i1]:
            continue
        p0, p1 = fused[i0], fused[i1]
        avg_spd = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) / ((i1 - i0) / fps)
        if avg_spd <= MAX_PASS_ARC_SPEED_MPS:
            alphas = np.linspace(0.0, 1.0, i1 - i0 + 1, dtype=np.float32)[1:-1, None]
            kin_xy[i0 + 1 : i1] = p0[None, :] * (1.0 - alphas) + p1[None, :] * alphas
            is_flight_bridge[i0 + 1 : i1] = True

    # Binomial [0.25, 0.50, 0.25] smoothing kernel for low-noise central-difference velocity
    kin_smooth = kin_xy.copy()
    ok3 = np.isfinite(kin_xy[:-2, 0]) & np.isfinite(kin_xy[1:-1, 0]) & np.isfinite(kin_xy[2:, 0])
    kin_smooth[1:-1][ok3] = 0.25 * kin_xy[:-2][ok3] + 0.50 * kin_xy[1:-1][ok3] + 0.25 * kin_xy[2:][ok3]

    kin_vel = np.full_like(kin_smooth, np.nan)
    ok_v = np.isfinite(kin_smooth[2:, 0]) & np.isfinite(kin_smooth[:-2, 0])
    kin_vel[1:-1][ok_v] = (kin_smooth[2:][ok_v] - kin_smooth[:-2][ok_v]) * (0.5 * fps)
    ok_f = np.isnan(kin_vel[:-1, 0]) & np.isfinite(kin_smooth[1:, 0]) & np.isfinite(kin_smooth[:-1, 0])
    kin_vel[:-1][ok_f] = (kin_smooth[1:][ok_f] - kin_smooth[:-1][ok_f]) * fps
    ok_b = np.isnan(kin_vel[1:, 0]) & np.isfinite(kin_smooth[1:, 0]) & np.isfinite(kin_smooth[:-1, 0])
    kin_vel[1:][ok_b] = (kin_smooth[1:][ok_b] - kin_smooth[:-1][ok_b]) * fps

    kin_speed = np.hypot(kin_vel[:, 0], kin_vel[:, 1])
    bad_v = kin_speed > MAX_BALL_SPEED_MPS
    kin_vel[bad_v] = np.nan
    kin_speed[bad_v] = np.nan

    # Vector velocity change magnitude |v(t + dt) - v(t - dt)| over +/- 3 frames (~0.24 s)
    vel_zero = np.where(np.isfinite(kin_vel), kin_vel, 0.0)
    ball_impulse = np.zeros(n_frames, dtype=np.float32)
    has_vpair = np.isfinite(kin_vel[6:, 0]) & np.isfinite(kin_vel[:-6, 0])
    dv = np.hypot(vel_zero[6:, 0] - vel_zero[:-6, 0], vel_zero[6:, 1] - vel_zero[:-6, 1])
    ball_impulse[3 : n_frames - 3] = np.where(has_vpair, dv, 0.0)
    ball_impulse[~obs_mask] = 0.0

    # 1-second net ball displacement
    net_disp_1s = np.zeros(n_frames, dtype=np.float32)
    hw = max(1, int(round(0.5 * fps)))
    if n_frames > 2 * hw:
        p_lo = kin_smooth[: -2 * hw]
        p_hi = kin_smooth[2 * hw :]
        ok_d = np.isfinite(p_lo[:, 0]) & np.isfinite(p_hi[:, 0])
        disp_mid = np.zeros(len(p_lo), dtype=np.float32)
        disp_mid[ok_d] = np.hypot(p_hi[ok_d, 0] - p_lo[ok_d, 0], p_hi[ok_d, 1] - p_lo[ok_d, 1])
        net_disp_1s[hw : n_frames - hw] = disp_mid

    # Kinematic boundary-crossing projection when ball disappears moving toward a pitch line
    boundary_exit = np.zeros(n_frames, dtype=np.float32)
    exit_impulse = np.zeros(n_frames, dtype=bool)
    oob_onset = np.flatnonzero(out_of_bounds[1:] & (~out_of_bounds[:-1])) + 1
    exit_impulse[oob_onset] = True

    min_stoppage_gap = int(round(2.0 * fps))
    decay_frames = int(round(10.0 * fps))
    for k in range(len(vis_idx) - 1):
        i0 = int(vis_idx[k])
        i1 = int(vis_idx[k + 1])
        if (i1 - i0 - 1) < min_stoppage_gap:
            continue
        x0, y0 = fused[i0]
        vx, vy = kin_vel[i0] if np.isfinite(kin_vel[i0, 0]) else (0.0, 0.0)
        if out_of_bounds[i0]:
            t_cross = i0
        else:
            cross_dt = None
            if y0 < 2.2 * REACH_SCALE_M and vy < -1.5:
                cross_dt = max(0.0, y0 / (-vy))
            elif y0 > width - 2.2 * REACH_SCALE_M and vy > 1.5:
                cross_dt = max(0.0, (width - y0) / vy)
            elif x0 < 2.2 * REACH_SCALE_M and vx < -1.5:
                cross_dt = max(0.0, x0 / (-vx))
            elif x0 > length - 2.2 * REACH_SCALE_M and vx > 1.5:
                cross_dt = max(0.0, (length - x0) / vx)
            if cross_dt is None or cross_dt > 0.85:
                continue
            t_cross = min(i1 - 1, i0 + max(1, int(round(cross_dt * fps))))

        boundary_exit[t_cross : min(i1, t_cross + decay_frames)] = 1.0
        exit_impulse[t_cross] = True

    return {
        "obs_xy": fused,
        "obs_mask": obs_mask,
        "kin_xy": kin_xy,
        "kin_vel": kin_vel,
        "kin_speed": kin_speed,
        "ball_impulse": ball_impulse,
        "net_disp_1s": net_disp_1s,
        "is_flight_bridge": is_flight_bridge,
        "out_of_bounds": out_of_bounds,
        "boundary_exit_signal": boundary_exit,
        "exit_impulse": exit_impulse,
    }


def compute_ball_player_kinematics(
    player_xy: np.ndarray,
    player_vel: np.ndarray,
    slot_teams: np.ndarray,
    ball_data: Dict[str, np.ndarray],
    fps: float = 25.0,
) -> Dict[str, np.ndarray]:
    """Compute continuous player-to-ball distances, relative speeds, exponential interaction
    affinities, impulse-weighted touches, and ballistic pass-arc continuity per team.
    """
    n_frames = len(player_xy)
    h_slots = slot_teams == HOME
    a_slots = slot_teams == AWAY

    kin_xy = ball_data["kin_xy"]
    kin_vel = np.where(np.isfinite(ball_data["kin_vel"]), ball_data["kin_vel"], 0.0)
    obs_in = ball_data["obs_mask"] & (~ball_data["out_of_bounds"])
    oob = ball_data["out_of_bounds"]
    impulse = ball_data["ball_impulse"]

    d1_h = np.full(n_frames, 25.0, dtype=np.float32)
    d2_h = np.full(n_frames, 30.0, dtype=np.float32)
    d1_a = np.full(n_frames, 25.0, dtype=np.float32)
    d2_a = np.full(n_frames, 30.0, dtype=np.float32)
    rv_h = np.full(n_frames, 20.0, dtype=np.float32)
    rv_a = np.full(n_frames, 20.0, dtype=np.float32)
    near_dx_h = np.zeros(n_frames, dtype=np.float32)
    near_dx_a = np.zeros(n_frames, dtype=np.float32)

    valid_b = np.flatnonzero(np.isfinite(kin_xy[:, 0]) & (~oob))
    chunk = 12000
    for s0 in range(0, len(valid_b), chunk):
        idx = valid_b[s0 : s0 + chunk]
        b = kin_xy[idx]
        bv = kin_vel[idx]
        rows = np.arange(len(idx))

        for mask, d1_out, d2_out, rv_out, dx_out in [
            (h_slots, d1_h, d2_h, rv_h, near_dx_h),
            (a_slots, d1_a, d2_a, rv_a, near_dx_a),
        ]:
            p_sub = player_xy[idx][:, mask, :]
            v_sub = np.where(np.isfinite(player_vel[idx][:, mask, :]), player_vel[idx][:, mask, :], 0.0)
            dd = np.hypot(p_sub[:, :, 0] - b[:, None, 0], p_sub[:, :, 1] - b[:, None, 1])
            dd = np.where(np.isfinite(dd), dd, 99.0)
            part = np.partition(dd, kth=min(1, dd.shape[1] - 1), axis=1)
            d1_out[idx] = np.minimum(part[:, 0], 30.0)
            if dd.shape[1] > 1:
                d2_out[idx] = np.minimum(part[:, 1], 35.0)
            arg1 = np.argmin(dd, axis=1)
            v1 = v_sub[rows, arg1]
            rv_out[idx] = np.minimum(np.hypot(bv[:, 0] - v1[:, 0], bv[:, 1] - v1[:, 1]), 35.0)
            p1_x = p_sub[rows, arg1, 0]
            dx_out[idx] = np.where(np.isfinite(p1_x), b[:, 0] - p1_x, 0.0)

    # Continuous physical interaction affinity K_T(t) in [0, 1] on observed in-bounds ball frames
    obs_f = obs_in.astype(np.float32)
    aff_h = (obs_f * np.exp(-d1_h / REACH_SCALE_M) * np.exp(-rv_h / PASS_SPEED_SCALE_MPS)).astype(np.float32)
    aff_a = (obs_f * np.exp(-d1_a / REACH_SCALE_M) * np.exp(-rv_a / PASS_SPEED_SCALE_MPS)).astype(np.float32)

    # Impulse-weighted contact signal (peaks on passes, interceptions, headers, tackles)
    imp_scale = np.log1p(impulse)
    imp_aff_h = (aff_h * imp_scale).astype(np.float32)
    imp_aff_a = (aff_a * imp_scale).astype(np.float32)

    # Continuous ballistic pass-arc bridge across unobserved in-bounds gaps (attenuated by gap duration)
    arc_h = aff_h.copy()
    arc_a = aff_a.copy()
    vis_in = np.flatnonzero(obs_in)
    max_bridge_frames = int(round(5.0 * fps))
    tau_bridge = 2.5 * fps

    for k in range(len(vis_in) - 1):
        i0 = int(vis_in[k])
        i1 = int(vis_in[k + 1])
        gap = i1 - i0 - 1
        if gap <= 0 or gap > max_bridge_frames or oob[i0 : i1 + 1].any():
            continue
        decay = float(np.exp(-gap / tau_bridge))
        alphas = np.linspace(0.0, 1.0, i1 - i0 + 1, dtype=np.float32)[1:-1]
        arc_h[i0 + 1 : i1] = decay * ((1.0 - alphas) * aff_h[i0] + alphas * aff_h[i1])
        arc_a[i0 + 1 : i1] = decay * ((1.0 - alphas) * aff_a[i0] + alphas * aff_a[i1])

    # Clear dominant contact indicators for barrier-clipped elapsed-time timers
    # exp(-1.0) ~ 0.368 corresponds physically to player within 1 characteristic reach scale
    e_minus_1 = float(np.exp(-1.0))
    dom_ratio = float(np.exp(0.35))
    contact_h = obs_in & (aff_h >= e_minus_1) & (aff_h >= dom_ratio * aff_a)
    contact_a = obs_in & (aff_a >= e_minus_1) & (aff_a >= dom_ratio * aff_h)

    return {
        "d1_h": d1_h,
        "d2_h": d2_h,
        "d1_a": d1_a,
        "d2_a": d2_a,
        "rv_h": rv_h,
        "rv_a": rv_a,
        "near_dx_h": near_dx_h,
        "near_dx_a": near_dx_a,
        "aff_h": aff_h,
        "aff_a": aff_a,
        "imp_aff_h": imp_aff_h,
        "imp_aff_a": imp_aff_a,
        "arc_h": arc_h,
        "arc_a": arc_a,
        "contact_h": contact_h,
        "contact_a": contact_a,
    }
