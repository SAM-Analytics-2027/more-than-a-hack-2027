"""Ball Trajectory Reconstruction, Boundary-Exit Detection, Verified Ball Control,
Pass-Chain Bridges, and Live-Play Occlusion Continuity.
"""

from typing import Dict, Optional, Tuple
import numpy as np

HOME = 0
AWAY = 1

MAX_BALL_SPEED_MPS = 38.0       # ~137 km/h physical upper bound for professional football shots
MAX_INTERP_GAP_FRAMES = 75      # 3.0 s max ballistic/ground gap bridge
MAX_INTERP_SPEED_MPS = 28.0     # max plausible average horizontal speed across an unseen gap


def despike_ball_track(ball_xy: np.ndarray, fps: float = 25.0) -> np.ndarray:
    """Remove isolated optical false positives (e.g. white boots, hoardings) violating max ball speed."""
    clean = ball_xy.copy()
    vis = np.flatnonzero(np.isfinite(clean[:, 0]))
    if len(vis) < 3:
        return clean

    for _ in range(2):
        vis = np.flatnonzero(np.isfinite(clean[:, 0]))
        if len(vis) < 3:
            break
        pts = clean[vis]
        dt_prev = (vis[1:-1] - vis[:-2]) / fps
        dt_next = (vis[2:] - vis[1:-1]) / fps
        d_prev = np.hypot(pts[1:-1, 0] - pts[:-2, 0], pts[1:-1, 1] - pts[:-2, 1])
        d_next = np.hypot(pts[2:, 0] - pts[1:-1, 0], pts[2:, 1] - pts[1:-1, 1])
        d_skip = np.hypot(pts[2:, 0] - pts[:-2, 0], pts[2:, 1] - pts[:-2, 1])
        dt_skip = (vis[2:] - vis[:-2]) / fps

        v_prev = d_prev / np.maximum(dt_prev, 1.0 / fps)
        v_next = d_next / np.maximum(dt_next, 1.0 / fps)
        v_skip = d_skip / np.maximum(dt_skip, 1.0 / fps)

        is_spike = (
            (v_prev > MAX_BALL_SPEED_MPS)
            & (v_next > MAX_BALL_SPEED_MPS)
            & ((v_skip < MAX_BALL_SPEED_MPS * 0.8) | (dt_skip > 2.0))
        )
        isolated = (dt_prev > 0.8) & (dt_next > 0.8) & (d_prev > 18.0) & (d_next > 18.0)
        bad_idx = vis[1:-1][is_spike | isolated]
        if len(bad_idx) == 0:
            break
        clean[bad_idx] = np.nan

    return clean


def reconstruct_ball_trajectory(
    tac_ball_xy: np.ndarray,
    brd_ball_xy: Optional[np.ndarray],
    pitch: Tuple[float, float],
    fps: float = 25.0,
) -> Dict[str, np.ndarray]:
    """Fuse Tactical + live-verified Broadcast ball tracks, despike, interpolate short in-field flights,
    and compute 1-second net displacement plus boundary line-exit instants.
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
            < 3.0
        )
        fused[agree] = 0.8 * tac_clean[agree] + 0.2 * brd_clean[agree]
        fused = despike_ball_track(fused, fps=fps)

    n_frames = len(fused)
    obs_mask = np.isfinite(fused[:, 0])

    # Out-of-bounds check (0.45 m tolerance for ball radius + homography error)
    margin = 0.45
    out_of_bounds = np.zeros(n_frames, dtype=bool)
    out_of_bounds[obs_mask] = (
        (fused[obs_mask, 0] < -margin)
        | (fused[obs_mask, 0] > length + margin)
        | (fused[obs_mask, 1] < -margin)
        | (fused[obs_mask, 1] > width + margin)
    )

    # Linear interpolation across plausible short in-field gaps (passes in air / brief occlusions)
    kin_xy = fused.copy()
    is_flight_bridge = np.zeros(n_frames, dtype=bool)
    vis_idx = np.flatnonzero(obs_mask)

    for k in range(len(vis_idx) - 1):
        i0 = int(vis_idx[k])
        i1 = int(vis_idx[k + 1])
        gap = i1 - i0 - 1
        if gap <= 0 or gap > MAX_INTERP_GAP_FRAMES:
            continue
        if out_of_bounds[i0] or out_of_bounds[i1]:
            continue
        p0 = fused[i0]
        p1 = fused[i1]
        dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        avg_spd = dist / ((i1 - i0) / fps)
        if avg_spd <= MAX_INTERP_SPEED_MPS:
            alphas = np.linspace(0.0, 1.0, i1 - i0 + 1, dtype=np.float32)[1:-1, None]
            kin_xy[i0 + 1 : i1] = p0[None, :] * (1.0 - alphas) + p1[None, :] * alphas
            is_flight_bridge[i0 + 1 : i1] = True

    # Smooth interpolated positions lightly for stable velocity derivatives
    kin_smooth = kin_xy.copy()
    ok3 = np.isfinite(kin_xy[:-2, 0]) & np.isfinite(kin_xy[1:-1, 0]) & np.isfinite(kin_xy[2:, 0])
    kin_smooth[1:-1][ok3] = (
        0.25 * kin_xy[:-2][ok3] + 0.50 * kin_xy[1:-1][ok3] + 0.25 * kin_xy[2:][ok3]
    )

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

    # 1-second net ball displacement inside pitch (separates slow dribbling from a stationary dead ball)
    net_disp_1s = np.zeros(n_frames, dtype=np.float32)
    half_w = 12  # +/- 12 frames = 0.96 s window
    p_lo = kin_smooth[:-2 * half_w]
    p_hi = kin_smooth[2 * half_w :]
    ok_d = np.isfinite(p_lo[:, 0]) & np.isfinite(p_hi[:, 0])
    disp_mid = np.zeros(len(p_lo), dtype=np.float32)
    disp_mid[ok_d] = np.hypot(p_hi[ok_d, 0] - p_lo[ok_d, 0], p_hi[ok_d, 1] - p_lo[ok_d, 1])
    net_disp_1s[half_w : n_frames - half_w] = disp_mid

    # Boundary-exit detection: project exact frame ball crosses touchline or goal-line
    boundary_exit = np.zeros(n_frames, dtype=np.float32)
    exit_impulse = np.zeros(n_frames, dtype=bool)

    oob_onset = np.flatnonzero(out_of_bounds[1:] & (~out_of_bounds[:-1])) + 1
    exit_impulse[oob_onset] = True

    for k in range(len(vis_idx) - 1):
        i0 = int(vis_idx[k])
        i1 = int(vis_idx[k + 1])
        gap = i1 - i0 - 1
        if gap < 50:
            continue
        x0, y0 = fused[i0]
        vx, vy = kin_vel[i0] if np.isfinite(kin_vel[i0, 0]) else (0.0, 0.0)

        if out_of_bounds[i0]:
            t_cross = i0
            end_mark = min(i1, t_cross + 250)
            boundary_exit[t_cross:end_mark] = 1.0
            exit_impulse[t_cross] = True
            continue

        cross_dt = None
        if y0 < 3.5 and vy < -1.5:
            cross_dt = max(0.0, y0 / (-vy))
        elif y0 > width - 3.5 and vy > 1.5:
            cross_dt = max(0.0, (width - y0) / vy)
        elif x0 < 3.5 and vx < -1.5:
            cross_dt = max(0.0, x0 / (-vx))
        elif x0 > length - 3.5 and vx > 1.5:
            cross_dt = max(0.0, (length - x0) / vx)

        if cross_dt is not None and cross_dt <= 0.8:
            t_cross = min(i1 - 1, i0 + max(1, int(round(cross_dt * fps))))
            end_mark = min(i1, t_cross + 250)
            boundary_exit[t_cross:end_mark] = 1.0
            exit_impulse[t_cross] = True

    return {
        "obs_xy": fused,
        "obs_mask": obs_mask,
        "kin_xy": kin_xy,
        "kin_vel": kin_vel,
        "kin_speed": kin_speed,
        "net_disp_1s": net_disp_1s,
        "is_flight_bridge": is_flight_bridge,
        "out_of_bounds": out_of_bounds,
        "boundary_exit_signal": boundary_exit,
        "exit_impulse": exit_impulse,
    }


def detect_ball_impact_events(
    player_xy: np.ndarray,
    slot_teams: np.ndarray,
    ball_data: Dict[str, np.ndarray],
    fps: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect sudden kinematic kicks and impact events from ball acceleration and directional shifts,
    attributed to the nearest player within physical reaching distance (< 2.2 m).
    Returns:
        impact_h : (N,) bool
        impact_a : (N,) bool
        impact_strength : (N,) float32
    """
    kin_xy = ball_data["kin_xy"]
    n_frames = len(kin_xy)

    v3 = np.full_like(kin_xy, np.nan)
    v3[2:-2] = (kin_xy[4:] - kin_xy[:-4]) * (fps / 4.0)
    sp3 = np.hypot(v3[:, 0], v3[:, 1])

    v_fwd = v3[3:]
    v_back = v3[:-3]
    ok_dv = np.isfinite(v_fwd[:, 0]) & np.isfinite(v_back[:, 0])
    dv_sub = np.hypot(v_fwd[:, 0] - v_back[:, 0], v_fwd[:, 1] - v_back[:, 1])
    dv_vec = np.zeros(n_frames, dtype=np.float32)
    dv_vec[3:] = np.where(ok_dv, dv_sub, 0.0)

    sp_fwd = sp3[3:]
    sp_back = sp3[:-3]
    ok_a = np.isfinite(sp_fwd) & np.isfinite(sp_back)
    a_sub = (sp_fwd - sp_back) * (fps / 3.0)
    a_mag = np.zeros(n_frames, dtype=np.float32)
    a_mag[3:] = np.where(ok_a, a_sub, 0.0)

    h_slots = slot_teams == HOME
    a_slots = slot_teams == AWAY

    vis_ball = np.isfinite(kin_xy[:, 0])
    idx_vis = np.flatnonzero(vis_ball)

    dist_h = np.full(n_frames, 999.0, dtype=np.float32)
    dist_a = np.full(n_frames, 999.0, dtype=np.float32)

    chunk = 10000
    for start in range(0, len(idx_vis), chunk):
        sub = idx_vis[start : start + chunk]
        bx = kin_xy[sub]
        ph = player_xy[sub][:, h_slots]
        pa = player_xy[sub][:, a_slots]

        dh = np.hypot(ph[:, :, 0] - bx[:, None, 0], ph[:, :, 1] - bx[:, None, 1])
        da = np.hypot(pa[:, :, 0] - bx[:, None, 0], pa[:, :, 1] - bx[:, None, 1])

        dist_h[sub] = np.nanmin(np.where(np.isfinite(dh), dh, 999.0), axis=1)
        dist_a[sub] = np.nanmin(np.where(np.isfinite(da), da, 999.0), axis=1)

    is_active_impulse = (a_mag > 5.0) | (dv_vec > 3.5)

    impact_h = (
        vis_ball
        & (dist_h < 2.2)
        & (dist_h < dist_a - 0.2)
        & (is_active_impulse | (dist_h < 0.8))
    )
    impact_a = (
        vis_ball
        & (dist_a < 2.2)
        & (dist_a < dist_h - 0.2)
        & (is_active_impulse | (dist_a < 0.8))
    )

    impact_strength = np.maximum(
        np.clip(dv_vec / 5.0, 0.0, 2.0),
        np.clip(a_mag / 8.0, 0.0, 2.0),
    ).astype(np.float32)

    return impact_h, impact_a, impact_strength


def detect_ball_control_and_passes(
    player_xy: np.ndarray,
    player_vel: np.ndarray,
    slot_teams: np.ndarray,
    ball_data: Dict[str, np.ndarray],
    team_dynamics: Optional[Dict[str, np.ndarray]] = None,
    fps: float = 25.0,
) -> Dict[str, np.ndarray]:
    """Detect verified controlled ball contacts vs uncontrolled clearances/deflections/aerial duels,
    pass-chain flight bridges, and live-play occlusion continuity.
    """
    n_frames = len(player_xy)
    obs_xy = ball_data["obs_xy"]
    obs_mask = ball_data["obs_mask"]
    kin_vel = ball_data["kin_vel"]
    oob = ball_data["out_of_bounds"]
    exit_sig = ball_data["boundary_exit_signal"]
    kin_speed = ball_data["kin_speed"]

    h_slots = slot_teams == HOME
    a_slots = slot_teams == AWAY

    impact_h, impact_a, impact_strength = detect_ball_impact_events(
        player_xy, slot_teams, ball_data, fps=fps
    )

    min_dist_h = np.full(n_frames, np.nan, dtype=np.float32)
    min_dist_a = np.full(n_frames, np.nan, dtype=np.float32)
    rel_speed_h = np.full(n_frames, np.nan, dtype=np.float32)
    rel_speed_a = np.full(n_frames, np.nan, dtype=np.float32)

    in_bounds_obs = obs_mask & (~oob)
    obs_idx = np.flatnonzero(in_bounds_obs)

    chunk = 10000
    for start in range(0, len(obs_idx), chunk):
        idx = obs_idx[start : start + chunk]
        b = obs_xy[idx]
        bv = kin_vel[idx]

        ph = player_xy[idx][:, h_slots, :]
        vh = player_vel[idx][:, h_slots, :]
        dh = np.hypot(ph[:, :, 0] - b[:, None, 0], ph[:, :, 1] - b[:, None, 1])
        dh_safe = np.where(np.isfinite(dh), dh, 1e6)
        arg_h = np.argmin(dh_safe, axis=1)
        min_h = dh_safe[np.arange(len(idx)), arg_h]
        ok_h = min_h < 1e5
        min_dist_h[idx[ok_h]] = min_h[ok_h]

        near_vh = vh[np.arange(len(idx)), arg_h]
        vr_h = np.hypot(bv[:, 0] - near_vh[:, 0], bv[:, 1] - near_vh[:, 1])
        vr_h = np.where(np.isfinite(vr_h), vr_h, np.hypot(bv[:, 0], bv[:, 1]))
        rel_speed_h[idx[ok_h]] = vr_h[ok_h]

        pa = player_xy[idx][:, a_slots, :]
        va = player_vel[idx][:, a_slots, :]
        da = np.hypot(pa[:, :, 0] - b[:, None, 0], pa[:, :, 1] - b[:, None, 1])
        da_safe = np.where(np.isfinite(da), da, 1e6)
        arg_a = np.argmin(da_safe, axis=1)
        min_a = da_safe[np.arange(len(idx)), arg_a]
        ok_a = min_a < 1e5
        min_dist_a[idx[ok_a]] = min_a[ok_a]

        near_va = va[np.arange(len(idx)), arg_a]
        vr_a = np.hypot(bv[:, 0] - near_va[:, 0], bv[:, 1] - near_va[:, 1])
        vr_a = np.where(np.isfinite(vr_a), vr_a, np.hypot(bv[:, 0], bv[:, 1]))
        rel_speed_a[idx[ok_a]] = vr_a[ok_a]

    r_ctrl = 1.85
    prox_h = np.where(np.isfinite(min_dist_h), np.clip(1.0 - min_dist_h / r_ctrl, 0.0, 1.0), 0.0)
    prox_a = np.where(np.isfinite(min_dist_a), np.clip(1.0 - min_dist_a / r_ctrl, 0.0, 1.0), 0.0)

    sep_h = np.where(
        np.isfinite(min_dist_h) & np.isfinite(min_dist_a),
        np.clip((min_dist_a - min_dist_h + 0.5) / 2.0, 0.15, 1.25),
        1.0,
    )
    sep_a = np.where(
        np.isfinite(min_dist_h) & np.isfinite(min_dist_a),
        np.clip((min_dist_h - min_dist_a + 0.5) / 2.0, 0.15, 1.25),
        1.0,
    )

    vel_factor_h = np.where(
        np.isfinite(rel_speed_h),
        np.clip(1.25 - rel_speed_h / 18.0, 0.30, 1.25),
        0.75,
    )
    vel_factor_a = np.where(
        np.isfinite(rel_speed_a),
        np.clip(1.25 - rel_speed_a / 18.0, 0.30, 1.25),
        0.75,
    )

    raw_touch_h = (prox_h * sep_h * vel_factor_h).astype(np.float32)
    raw_touch_a = (prox_a * sep_a * vel_factor_a).astype(np.float32)

    # Group contiguous/near-contiguous (gap <= 3 frames) touches of the same team into touch episodes
    contact_team = np.full(n_frames, -1, dtype=np.int8)
    contact_team[(raw_touch_h > 0.18) & (raw_touch_h > raw_touch_a + 0.10)] = HOME
    contact_team[(raw_touch_a > 0.18) & (raw_touch_a > raw_touch_h + 0.10)] = AWAY

    touch_episodes = []  # [start, end, team, integrated_strength, med_rel_speed]
    t = 0
    while t < n_frames:
        tm = int(contact_team[t])
        if tm < 0:
            t += 1
            continue
        e = t + 1
        while e < n_frames:
            if contact_team[e] == tm:
                e += 1
            elif e + 1 < n_frames and contact_team[e] < 0 and contact_team[e + 1] == tm:
                e += 2
            elif (
                e + 2 < n_frames
                and contact_team[e] < 0
                and contact_team[e + 1] < 0
                and contact_team[e + 2] == tm
            ):
                e += 3
            else:
                break
        strength = float(raw_touch_h[t:e].sum() if tm == HOME else raw_touch_a[t:e].sum())
        vr_slice = rel_speed_h[t:e] if tm == HOME else rel_speed_a[t:e]
        med_vr = float(np.nanmedian(vr_slice)) if np.isfinite(vr_slice).any() else 10.0
        touch_episodes.append([t, e, tm, strength, med_vr])
        t = e

    # Official "Controlled Possession" filter (CHALLENGE.md):
    # Isolated deflections, blocks, parries, or won aerial duels without subsequent team control
    # do NOT change ball possession.
    walk_ratio_sm = team_dynamics["walking_ratio_smooth"] if team_dynamics is not None else None
    rms_spd_sm = team_dynamics["rms_speed_smooth"] if team_dynamics is not None else None

    def _has_stoppage_between(f0: int, f1: int) -> bool:
        if f1 <= f0:
            return False
        if oob[f0:f1].any() or (exit_sig[f0:f1].mean() > 0.45):
            return True
        if (f1 - f0) >= 65 and walk_ratio_sm is not None and rms_spd_sm is not None:
            vis_ratio = float(obs_mask[f0:f1].mean())
            if vis_ratio < 0.25:
                mw = float(np.nanmean(walk_ratio_sm[f0:f1]))
                mv = float(np.nanmean(rms_spd_sm[f0:f1]))
                if mw > 0.58 or mv < 1.88:
                    return True
        return False

    def _is_sustained_carry(ep_item) -> bool:
        s_i, e_i, _, str_i, mvr_i = ep_item
        return (e_i - s_i) >= 28 and str_i >= 11.0 and mvr_i < 6.5

    def _is_clearance(ep_item) -> bool:
        s_i, e_i, tm_i, str_i, _ = ep_item
        bsp_sub = kin_speed[s_i : min(n_frames, e_i + 15)]
        has_kick = (bsp_sub > 15.0).any() if len(bsp_sub) > 0 else False
        opp_dist = min_dist_h[s_i:e_i] if tm_i == AWAY else min_dist_a[s_i:e_i]
        is_uncontested = (np.nanmean(opp_dist) > 2.5) if (len(opp_dist) > 0 and np.isfinite(opp_dist).any()) else False
        return has_kick and is_uncontested

    is_uncontrolled = np.zeros(len(touch_episodes), dtype=bool)
    for _ in range(2):
        active_indices = [i for i in range(len(touch_episodes)) if not is_uncontrolled[i]]
        if not active_indices:
            break
        # Group active episodes into runs of identical team
        runs = []
        i = 0
        while i < len(active_indices):
            tm = touch_episodes[active_indices[i]][2]
            j = i + 1
            while j < len(active_indices) and touch_episodes[active_indices[j]][2] == tm:
                j += 1
            runs.append((tm, i, j))
            i = j

        # Check runs sandwiched between the same possessing team (T -> Opp ... Opp -> T)
        # Rule (CHALLENGE.md): Deflections, tackles, and duels without sustained team control
        # do not change possession.
        for r_idx in range(1, len(runs) - 1):
            prev_tm, p_s, p_e = runs[r_idx - 1]
            curr_tm, c_s, c_e = runs[r_idx]
            next_tm, n_s, n_e = runs[r_idx + 1]

            if prev_tm == next_tm and prev_tm != curr_tm:
                t_prev_team = touch_episodes[active_indices[p_e - 1]][1]
                t_next_team = touch_episodes[active_indices[n_s]][0]
                gap_span = t_next_team - t_prev_team
                if gap_span <= 210 and not _has_stoppage_between(t_prev_team, t_next_team):
                    has_carry = any(_is_sustained_carry(touch_episodes[active_indices[k]]) for k in range(c_s, c_e))
                    has_clearance = any(_is_clearance(touch_episodes[active_indices[k]]) for k in range(c_s, c_e))
                    total_opp_frames = sum(
                        touch_episodes[active_indices[k]][1] - touch_episodes[active_indices[k]][0]
                        for k in range(c_s, c_e)
                    )
                    total_opp_strength = sum(
                        touch_episodes[active_indices[k]][3] for k in range(c_s, c_e)
                    )
                    if not has_carry and not has_clearance and (total_opp_frames < 32 or total_opp_strength < 15.0):
                        for k in range(c_s, c_e):
                            is_uncontrolled[active_indices[k]] = True

        for pos, k in enumerate(active_indices):
            s_curr, e_curr, tm_curr, str_curr, mvr_curr = touch_episodes[k]
            prev_ep = touch_episodes[active_indices[pos - 1]] if pos > 0 else None
            # Shot/cross block or goalkeeper parry deflected directly out of play
            if (
                prev_ep is not None
                and prev_ep[2] != tm_curr
                and (s_curr - prev_ep[1]) <= 100
                and (e_curr - s_curr) <= 16
                and _has_stoppage_between(e_curr, min(n_frames, e_curr + 65))
            ):
                is_uncontrolled[k] = True

    control_score_h = raw_touch_h.copy()
    control_score_a = raw_touch_a.copy()
    uncontrolled_mask = np.zeros(n_frames, dtype=bool)
    for k, (s_c, e_c, tm_c, _, _) in enumerate(touch_episodes):
        if is_uncontrolled[k]:
            uncontrolled_mask[s_c:e_c] = True
            if tm_c == HOME:
                control_score_h[s_c:e_c] *= 0.05
            else:
                control_score_a[s_c:e_c] *= 0.05

    # Boost control score at kinematic impact/kick frames ONLY if not an uncontrolled deflection
    control_score_h += 2.0 * (impact_h & (~uncontrolled_mask)).astype(np.float32)
    control_score_a += 2.0 * (impact_a & (~uncontrolled_mask)).astype(np.float32)

    valid_episodes = [ep for k, ep in enumerate(touch_episodes) if not is_uncontrolled[k]]
    pass_bridge_h = np.zeros(n_frames, dtype=np.float32)
    pass_bridge_a = np.zeros(n_frames, dtype=np.float32)
    verified_control_mask = np.zeros(n_frames, dtype=bool)

    def _is_substantial(ep) -> bool:
        s_i, e_i, _, str_i, _ = ep
        bsp_sub = kin_speed[s_i : min(n_frames, e_i + 8)]
        has_kick = (bsp_sub > 12.0).any() if len(bsp_sub) > 0 else False
        return _is_sustained_carry(ep) or ((e_i - s_i) >= 12 and str_i >= 4.5) or (has_kick and (e_i - s_i) >= 5)

    for idx_v, ep_v in enumerate(valid_episodes):
        s_c, e_c, tm_c, str_c, _ = ep_v
        is_sub_curr = _is_substantial(ep_v)
        has_sub_prev = idx_v > 0 and _is_substantial(valid_episodes[idx_v - 1])
        has_sub_next = idx_v + 1 < len(valid_episodes) and _is_substantial(valid_episodes[idx_v + 1])

        linked_prev = (
            idx_v > 0
            and valid_episodes[idx_v - 1][2] == tm_c
            and (s_c - valid_episodes[idx_v - 1][1]) <= 185
            and not _has_stoppage_between(valid_episodes[idx_v - 1][1], s_c)
            and (is_sub_curr or has_sub_prev)
        )
        linked_next = (
            idx_v + 1 < len(valid_episodes)
            and valid_episodes[idx_v + 1][2] == tm_c
            and (valid_episodes[idx_v + 1][0] - e_c) <= 185
            and not _has_stoppage_between(e_c, valid_episodes[idx_v + 1][0])
            and (is_sub_curr or has_sub_next)
        )
        is_verified = (
            is_sub_curr
            or linked_prev
            or linked_next
        )
        if is_verified:
            verified_control_mask[s_c:e_c] = True

        boost = min(2.2, 0.55 + 0.18 * (e_c - s_c) + 0.25 * str_c)
        if tm_c == HOME:
            pass_bridge_h[s_c:e_c] = np.maximum(pass_bridge_h[s_c:e_c], boost)
        else:
            pass_bridge_a[s_c:e_c] = np.maximum(pass_bridge_a[s_c:e_c], boost)

        if idx_v + 1 < len(valid_episodes):
            s_n, e_n, tm_n, str_n, _ = valid_episodes[idx_v + 1]
            gap_len = s_n - e_c
            if 0 < gap_len <= 190 and not _has_stoppage_between(e_c, s_n):
                sub_spd = kin_speed[e_c:s_n]
                sub_dh = min_dist_h[e_c:s_n]
                sub_da = min_dist_a[e_c:s_n]
                min_both = np.minimum(sub_dh, sub_da)
                loose = (sub_spd < 2.0) & (min_both > 2.5)
                loose_hits = np.flatnonzero(loose)
                bridge_end = s_n
                if len(loose_hits) > 0 and loose_hits[0] > 10:
                    bridge_end = e_c + int(loose_hits[0])

                if tm_c == tm_n:
                    decay = max(0.55, 1.35 - gap_len / 220.0)
                    if tm_c == HOME:
                        pass_bridge_h[e_c:bridge_end] = np.maximum(pass_bridge_h[e_c:bridge_end], decay)
                    else:
                        pass_bridge_a[e_c:bridge_end] = np.maximum(pass_bridge_a[e_c:bridge_end], decay)
                elif gap_len <= 140 and is_verified:
                    decay = max(0.30, 0.78 - gap_len / 210.0)
                    if tm_c == HOME:
                        pass_bridge_h[e_c:bridge_end] = np.maximum(pass_bridge_h[e_c:bridge_end], decay)
                    else:
                        pass_bridge_a[e_c:bridge_end] = np.maximum(pass_bridge_a[e_c:bridge_end], decay)

    # Live-play occlusion continuity across unseen in-bounds ball flights (3.0 s .. 10.4 s)
    # when the possessing team before and after the occlusion matches and players stay at live match tempo.
    occlusion_bridge_h = np.zeros(n_frames, dtype=np.float32)
    occlusion_bridge_a = np.zeros(n_frames, dtype=np.float32)
    in_same_team_occlusion = np.zeros(n_frames, dtype=bool)

    if len(valid_episodes) >= 2:
        ep_starts = np.array([ep[0] for ep in valid_episodes])
        ep_ends = np.array([ep[1] for ep in valid_episodes])

        vis_idx = np.flatnonzero(obs_mask)
        for k in range(len(vis_idx) - 1):
            g_s = int(vis_idx[k]) + 1
            g_e = int(vis_idx[k + 1])
            gap = g_e - g_s
            if gap <= MAX_INTERP_GAP_FRAMES or gap > 260:
                continue
            if _has_stoppage_between(max(0, g_s - 5), min(n_frames, g_e + 5)):
                continue

            before_candidates = np.flatnonzero((ep_ends <= g_s + 10) & (ep_ends >= g_s - 90))
            after_candidates = np.flatnonzero((ep_starts >= g_e - 10) & (ep_starts <= g_e + 90))
            if len(before_candidates) == 0 or len(after_candidates) == 0:
                continue

            ep_in = valid_episodes[int(before_candidates[-1])]
            ep_out = valid_episodes[int(after_candidates[0])]
            if ep_in[2] == ep_out[2]:
                tm_occ = ep_in[2]
                span_s = ep_in[1]
                span_e = ep_out[0]
                if not _has_stoppage_between(span_s, span_e):
                    in_same_team_occlusion[span_s:span_e] = True
                    if tm_occ == HOME:
                        occlusion_bridge_h[span_s:span_e] = 0.95
                    else:
                        occlusion_bridge_a[span_s:span_e] = 0.95

    proximity_score_h = np.where(
        np.isfinite(min_dist_h), np.exp(-0.5 * (min_dist_h / 2.8) ** 2), 0.0
    ).astype(np.float32)
    proximity_score_a = np.where(
        np.isfinite(min_dist_a), np.exp(-0.5 * (min_dist_a / 2.8) ** 2), 0.0
    ).astype(np.float32)

    verified_control_mask |= ((impact_h | impact_a) & (~uncontrolled_mask))

    return {
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
        "rel_speed_h": rel_speed_h,
        "rel_speed_a": rel_speed_a,
        "control_score_h": control_score_h,
        "control_score_a": control_score_a,
        "pass_bridge_h": pass_bridge_h,
        "pass_bridge_a": pass_bridge_a,
        "occlusion_bridge_h": occlusion_bridge_h,
        "occlusion_bridge_a": occlusion_bridge_a,
        "in_same_team_occlusion": in_same_team_occlusion,
        "proximity_score_h": proximity_score_h,
        "proximity_score_a": proximity_score_a,
        "verified_control_mask": verified_control_mask,
        "impact_h": impact_h & (~uncontrolled_mask),
        "impact_a": impact_a & (~uncontrolled_mask),
        "impact_strength": impact_strength,
    }
