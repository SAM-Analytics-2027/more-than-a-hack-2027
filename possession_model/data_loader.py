"""Streaming loaders for Tactical & Broadcast ATD, Homography, and Smart Tagging,
with Self-Calibrated Dual-Camera Constellation Synchronization, Replay Rejection,
and Camera Blind-Spot Player Recovery.
"""

import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from possession_model.signal_ops import AWAY, HOME  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Biomechanical human sprint ceiling (m/s, ~43.9 km/h). Consecutive-frame displacements
# exceeding MAX_HUMAN_SPRINT_MPS / fps correspond to tracker slot re-assignments.
MAX_HUMAN_SPRINT_MPS = 12.2


def parse_atd_metadata(meta_path: str) -> Tuple[np.ndarray, Tuple[str, str]]:
    """Return (slot_teams, (home_name, away_name)) where slot_teams is 0 for Home, 1 for Away."""
    root = ET.parse(meta_path).getroot()
    teams = [(t.get("id"), t.findtext("Name") or "") for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, got {len(teams)}")
    home_id = teams[0][0]
    slot_teams = np.array(
        [HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
        dtype=np.int8,
    )
    return slot_teams, (teams[0][1], teams[1][1])


def compute_player_kinematics(
    player_xy: np.ndarray,
    raw_speed: np.ndarray,
    fps: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute identity-switch-filtered player velocity vectors and speeds in m/s."""
    player_vel = np.full_like(player_xy, np.nan)
    disp = player_xy[1:] - player_xy[:-1]
    step_vel = disp * fps
    step_spd = np.hypot(step_vel[:, :, 0], step_vel[:, :, 1])
    valid_step = np.isfinite(step_spd) & (step_spd <= MAX_HUMAN_SPRINT_MPS)

    both_ok = valid_step[:-1] & valid_step[1:]
    player_vel[1:-1] = np.where(
        both_ok[:, :, None],
        0.5 * (step_vel[:-1] + step_vel[1:]),
        np.where(
            valid_step[1:, :, None],
            step_vel[1:],
            np.where(valid_step[:-1, :, None], step_vel[:-1], np.nan),
        ),
    )
    player_vel[0] = np.where(valid_step[0, :, None], step_vel[0], np.nan)
    player_vel[-1] = np.where(valid_step[-1, :, None], step_vel[-1], np.nan)

    vel_mag = np.hypot(player_vel[:, :, 0], player_vel[:, :, 1])
    valid_raw = np.isfinite(raw_speed) & (raw_speed >= 0.0) & (raw_speed <= MAX_HUMAN_SPRINT_MPS)
    player_speed = np.where(valid_raw, raw_speed, vel_mag)
    return player_vel, player_speed


def load_atd_arrays(
    game: str = GAME,
    data_dir: Optional[str] = None,
    source: str = "tactical",
) -> Dict[str, np.ndarray]:
    """Stream raw ATD tracking into metric float32 arrays (metres and m/s)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    slot_teams, names = parse_atd_metadata(meta_path)
    n_frames = source_frames(source, game)
    n_slots = len(slot_teams)
    length, width = GAMES[game]["pitch"]
    fps = float(GAMES[game]["fps"])

    player_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    raw_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            f_idx = int(parts[0]) - 1
            if f_idx < 0 or f_idx >= n_frames:
                continue

            ball_str = parts[2]
            if ball_str and ball_str[0] != "N":
                c1 = ball_str.find(",")
                c2 = ball_str.find(",", c1 + 1)
                ball_xy[f_idx, 0] = float(ball_str[:c1]) * length
                ball_xy[f_idx, 1] = float(ball_str[c1 + 1 : c2]) * width

            for s_idx, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N" or s_idx >= n_slots:
                    continue
                c1 = group.find(",")
                c2 = group.find(",", c1 + 1)
                player_xy[f_idx, s_idx, 0] = float(group[:c1]) * length
                player_xy[f_idx, s_idx, 1] = float(group[c1 + 1 : c2]) * width
                s_str = group[c2 + 1 :]
                if s_str and s_str[0] != "N":
                    raw_speed[f_idx, s_idx] = float(s_str)

    player_vel, player_speed = compute_player_kinematics(player_xy, raw_speed, fps=fps)
    return {
        "player_xy": player_xy,
        "player_speed": player_speed,
        "player_vel": player_vel,
        "ball_xy": ball_xy,
        "slot_teams": slot_teams,
        "team_names": names,
    }


def load_homography_quality(
    game: str = GAME,
    data_dir: Optional[str] = None,
    source: str = "tactical",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (valid_bool, reprojection_error) arrays from `_homography.csv`."""
    path = require(["homography"], game, data_dir, source)[0]
    df = pd.read_csv(path, usecols=["reprojection_error", "valid"])
    return df["valid"].to_numpy(dtype=bool), df["reprojection_error"].to_numpy(dtype=np.float32)


def _compute_team_centroids_4d(pxy: np.ndarray, slot_teams: np.ndarray) -> np.ndarray:
    out = np.full((len(pxy), 4), np.nan, dtype=np.float32)
    h_mask = slot_teams == HOME
    a_mask = slot_teams == AWAY
    h_cnt = np.isfinite(pxy[:, h_mask, 0]).sum(axis=1)
    a_cnt = np.isfinite(pxy[:, a_mask, 0]).sum(axis=1)
    hx = np.nansum(pxy[:, h_mask, 0], axis=1) / np.maximum(h_cnt, 1)
    hy = np.nansum(pxy[:, h_mask, 1], axis=1) / np.maximum(h_cnt, 1)
    ax = np.nansum(pxy[:, a_mask, 0], axis=1) / np.maximum(a_cnt, 1)
    ay = np.nansum(pxy[:, a_mask, 1], axis=1) / np.maximum(a_cnt, 1)
    ok = (h_cnt >= 4) & (a_cnt >= 4)
    out[ok, 0] = hx[ok]
    out[ok, 1] = hy[ok]
    out[ok, 2] = ax[ok]
    out[ok, 3] = ay[ok]
    return out


def _detect_blind_spot_bands(tac_player_xy: np.ndarray, width: float, n_bands: int = 6) -> Tuple[bool, bool]:
    """Detect whether the near (y ~ 0) or far (y ~ width) lateral pitch strip suffers from
    stadium-stand camera occlusion by comparing player detection symmetry across opposite strips.
    """
    vis = np.isfinite(tac_player_xy[:, :, 0])
    ys = tac_player_xy[:, :, 1][vis]
    if len(ys) == 0:
        return False, False
    band_w = width / float(n_bands)
    c_near = int(((ys >= 0.0) & (ys < band_w)).sum())
    c_far = int(((ys >= width - band_w) & (ys <= width)).sum())
    max_c = max(1, c_near, c_far)
    return (c_near < 0.70 * max_c), (c_far < 0.70 * max_c)


def sync_broadcast_to_tactical(
    tac: Dict[str, np.ndarray],
    game: str = GAME,
    data_dir: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Self-calibrating Broadcast-to-Tactical camera synchronization:
      1. Global coarse centroid search + local chunk fine alignment across asynchronous video timelines.
      2. Estimates the live-camera vs TV-replay gate directly from the lower-mode MAD of the
         constellation error distribution.
      3. Recovers missing ball-contending players strictly inside symmetric-deficit camera blind-spot
         strips without ingesting central-corridor referees or outside-the-line officials.
    """
    try:
        require(["atd_meta", "atd_raw", "homography"], game, data_dir, "broadcast")
        brd = load_atd_arrays(game=game, data_dir=data_dir, source="broadcast")
        brd_valid, brd_reproj = load_homography_quality(game=game, data_dir=data_dir, source="broadcast")
    except (SystemExit, IOError, KeyError):
        return None

    valid_rp_all = brd_reproj[brd_valid & np.isfinite(brd_reproj)]
    cam_reproj_cap = float(np.percentile(valid_rp_all, 95)) if len(valid_rp_all) > 200 else 4.0
    strict_reproj_thr = float(np.percentile(valid_rp_all, 85)) if len(valid_rp_all) > 200 else 3.2

    usable_cam = brd_valid & np.isfinite(brd_reproj) & (brd_reproj <= cam_reproj_cap)
    brd_ball = brd["ball_xy"].copy()
    brd_players = brd["player_xy"].copy()
    brd_ball[~usable_cam] = np.nan
    brd_players[~usable_cam] = np.nan

    tac_ball = tac["ball_xy"]
    tac_players = tac["player_xy"]
    n_tac = len(tac_ball)
    n_brd = len(brd_ball)
    length, width = GAMES[game]["pitch"]

    tac_cent = _compute_team_centroids_4d(tac_players, tac["slot_teams"])
    brd_cent = _compute_team_centroids_4d(brd_players, brd["slot_teams"])

    tac_h_slots = np.flatnonzero(tac["slot_teams"] == HOME)
    tac_a_slots = np.flatnonzero(tac["slot_teams"] == AWAY)
    brd_h_slots = np.flatnonzero(brd["slot_teams"] == HOME)
    brd_a_slots = np.flatnonzero(brd["slot_teams"] == AWAY)

    # Pass 1: Coarse + fine chunk alignment & constellation error collection
    blocks = half_video_blocks(game)
    chunk_size = 5000
    chunk_alignments: List[Tuple[int, int, int]] = []
    constellation_err = np.full(n_tac, np.nan, dtype=np.float32)
    n_brd_vis_arr = np.zeros(n_tac, dtype=np.int16)
    reproj_aligned = np.full(n_tac, np.inf, dtype=np.float32)

    for _, t_start, t_end in blocks:
        mid_tac = (t_start + t_end) // 2
        win_len = 5000
        w_tac_s = max(t_start, mid_tac - win_len // 2)
        w_tac_e = min(t_end, w_tac_s + win_len)
        sub_tac_c = tac_cent[w_tac_s:w_tac_e]
        tac_ok = np.isfinite(sub_tac_c[:, 0])

        best_coarse_off = None
        best_coarse_err = 1e9
        span = w_tac_e - w_tac_s
        for b_start in range(0, max(1, n_brd - span), 25):
            sub_brd_c = brd_cent[b_start : b_start + span]
            m = tac_ok & np.isfinite(sub_brd_c[:, 0])
            if int(m.sum()) < 600:
                continue
            err = float(np.mean(np.abs(sub_tac_c[m] - sub_brd_c[m])))
            if err < best_coarse_err:
                best_coarse_err = err
                best_coarse_off = w_tac_s - b_start

        if best_coarse_off is None:
            continue

        for c_start in range(t_start, t_end + 1, chunk_size):
            c_end = min(t_end + 1, c_start + chunk_size)
            c_len = c_end - c_start

            best_off = best_coarse_off
            best_ball_med = 1e9
            for off in range(best_coarse_off - 40, best_coarse_off + 41):
                b_s = c_start - off
                b_e = b_s + c_len
                if b_s < 0 or b_e > n_brd:
                    continue
                tb = tac_ball[c_start:c_end]
                bb = brd_ball[b_s:b_e]
                both = np.isfinite(tb[:, 0]) & np.isfinite(bb[:, 0])
                if int(both.sum()) >= 40:
                    dist = np.hypot(tb[both, 0] - bb[both, 0], tb[both, 1] - bb[both, 1])
                    med = float(np.median(dist))
                    if med < best_ball_med:
                        best_ball_med = med
                        best_off = off

            if best_ball_med < 1.8:
                best_coarse_off = best_off

            b_s = c_start - best_off
            b_e = b_s + c_len
            if b_s < 0 or b_e > n_brd:
                continue

            chunk_alignments.append((c_start, c_end, b_s))
            tac_h = tac_players[c_start:c_end, tac_h_slots]
            tac_a = tac_players[c_start:c_end, tac_a_slots]
            brd_h = brd_players[b_s:b_e, brd_h_slots]
            brd_a = brd_players[b_s:b_e, brd_a_slots]

            dh = np.hypot(brd_h[:, :, None, 0] - tac_h[:, None, :, 0], brd_h[:, :, None, 1] - tac_h[:, None, :, 1])
            da = np.hypot(brd_a[:, :, None, 0] - tac_a[:, None, :, 0], brd_a[:, :, None, 1] - tac_a[:, None, :, 1])
            vis_bh = np.isfinite(brd_h[:, :, 0])
            vis_ba = np.isfinite(brd_a[:, :, 0])
            min_dh = np.where(vis_bh, np.where(np.isfinite(dh), dh, 1e6).min(axis=2), np.nan)
            min_da = np.where(vis_ba, np.where(np.isfinite(da), da, 1e6).min(axis=2), np.nan)

            all_min = np.concatenate([min_dh, min_da], axis=1)
            n_vis = np.isfinite(all_min).sum(axis=1)
            n_brd_vis_arr[c_start:c_end] = n_vis
            reproj_aligned[c_start:c_end] = brd_reproj[b_s:b_e]
            has_b = n_vis >= 4
            if has_b.any():
                with np.errstate(all="ignore"):
                    constellation_err[c_start:c_end][has_b] = np.nanmedian(all_min[has_b], axis=1)

    # Self-calibrate live registration threshold from lower-half mode MAD
    valid_ce = constellation_err[np.isfinite(constellation_err) & (constellation_err < 1e4)]
    if len(valid_ce) > 200:
        q50 = float(np.median(valid_ce))
        low_mode = valid_ce[valid_ce <= q50]
        mode_med = float(np.median(low_mode))
        mode_mad = 1.4826 * float(np.median(np.abs(low_mode - mode_med)))
        live_err_thr = float(np.clip(mode_med + 3.5 * max(mode_mad, 0.15), 1.4, 2.6))
    else:
        live_err_thr = 2.2

    live_verified = (n_brd_vis_arr >= 4) & np.isfinite(constellation_err) & (constellation_err <= live_err_thr)
    brd_ball_aligned = np.full((n_tac, 2), np.nan, dtype=np.float32)

    blind_near, blind_far = _detect_blind_spot_bands(tac_players, width, n_bands=6)
    band_depth = width / 6.0

    fused_player_xy = tac_players.copy()
    fused_player_speed = tac["player_speed"].copy()
    fused_player_vel = tac["player_vel"].copy()
    brd_spd = brd["player_speed"]
    brd_vel = brd["player_vel"]
    n_backfilled = 0

    for c_start, c_end, b_s in chunk_alignments:
        b_e = b_s + (c_end - c_start)
        is_live = live_verified[c_start:c_end]
        bb_chunk = brd_ball[b_s:b_e]
        take = is_live[:, None] & np.isfinite(bb_chunk)
        brd_ball_aligned[c_start:c_end] = np.where(take, bb_chunk, np.nan)

        if not (blind_near or blind_far):
            continue

        is_strict = is_live & (n_brd_vis_arr[c_start:c_end] >= 5) & (reproj_aligned[c_start:c_end] <= strict_reproj_thr)
        if not is_strict.any():
            continue

        tac_h = tac_players[c_start:c_end, tac_h_slots]
        tac_a = tac_players[c_start:c_end, tac_a_slots]
        brd_h = brd_players[b_s:b_e, brd_h_slots]
        brd_a = brd_players[b_s:b_e, brd_a_slots]
        tb_chunk = tac_ball[c_start:c_end]
        has_ball = np.isfinite(tb_chunk[:, 0]) | np.isfinite(bb_chunk[:, 0])
        did_fill = np.zeros(c_end - c_start, dtype=bool)

        for brd_t, tac_same, tac_opp, t_slots, b_slots in [
            (brd_h, tac_h, tac_a, tac_h_slots, brd_h_slots),
            (brd_a, tac_a, tac_h, tac_a_slots, brd_a_slots),
        ]:
            vis_bt = np.isfinite(brd_t[:, :, 0])
            vis_tt = np.isfinite(tac_same[:, :, 0])
            n_tt = vis_tt.sum(axis=1)

            d_same = np.hypot(
                brd_t[:, :, None, 0] - tac_same[:, None, :, 0],
                brd_t[:, :, None, 1] - tac_same[:, None, :, 1],
            )
            min_same = np.where(np.isfinite(d_same), d_same, 1e6).min(axis=2)

            in_strip = np.zeros_like(vis_bt)
            if blind_near:
                in_strip |= brd_t[:, :, 1] <= band_depth
            if blind_far:
                in_strip |= brd_t[:, :, 1] >= (width - band_depth)

            cand_mask = (
                vis_bt
                & in_strip
                & (brd_t[:, :, 0] > 0.35)
                & (brd_t[:, :, 0] < length - 0.35)
                & (brd_t[:, :, 1] > 0.35)
                & (brd_t[:, :, 1] < width - 0.35)
                & (min_same > max(1.8, 1.15 * live_err_thr))
            )
            rows = np.flatnonzero(is_strict & has_ball & (n_tt <= 11) & cand_mask.any(axis=1))
            if len(rows) == 0:
                continue

            d_opp = np.hypot(
                brd_t[rows, :, None, 0] - tac_opp[rows, None, :, 0],
                brd_t[rows, :, None, 1] - tac_opp[rows, None, :, 1],
            )
            min_opp = np.where(np.isfinite(d_opp), d_opp, 1e6).min(axis=2)
            ok_rows = cand_mask[rows] & (min_opp > 0.55)

            for idx_r, r in enumerate(rows):
                c_cols = np.flatnonzero(ok_rows[idx_r])
                empty_slots = np.flatnonzero(~vis_tt[r])
                if len(c_cols) == 0 or len(empty_slots) == 0:
                    continue
                rx, ry = tb_chunk[r] if np.isfinite(tb_chunk[r, 0]) else bb_chunk[r]
                d_ball = np.hypot(brd_t[r, c_cols, 0] - rx, brd_t[r, c_cols, 1] - ry)
                near = np.flatnonzero(d_ball <= 1.8 * live_err_thr)
                if len(near) == 0:
                    continue
                b_local = int(c_cols[near[int(np.argmin(d_ball[near]))]])
                f_t = c_start + int(r)
                f_b = b_s + int(r)
                dst = int(t_slots[empty_slots[0]])
                src = int(b_slots[b_local])
                fused_player_xy[f_t, dst] = brd_players[f_b, src]
                fused_player_speed[f_t, dst] = brd_spd[f_b, src]
                fused_player_vel[f_t, dst] = brd_vel[f_b, src]
                did_fill[r] = True

        n_backfilled += int(did_fill.sum())

    return {
        "brd_ball_xy": brd_ball_aligned,
        "live_verified": live_verified,
        "fused_player_xy": fused_player_xy,
        "fused_player_speed": fused_player_speed,
        "fused_player_vel": fused_player_vel,
        "n_backfilled_frames": n_backfilled,
    }


def load_smart_tags(game: str = GAME, data_dir: Optional[str] = None) -> List[dict]:
    """Load Smart Tagging instances mapped to tactical video frames."""
    return load_smart_tagging.load(game=game, data_dir=data_dir)
