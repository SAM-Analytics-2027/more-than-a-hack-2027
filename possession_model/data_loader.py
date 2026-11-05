"""Streaming loaders for Tactical & Broadcast ATD, Homography, and Smart Tagging,
including automatic Dual-Camera Player-Constellation Synchronization and TV Replay Rejection.
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
from starter import load_smart_tagging  # noqa: E402

HOME = 0
AWAY = 1

# Physical sprint ceiling (m/s): jumps above this between consecutive frames (0.04 s)
# correspond to tracker slot re-assignments (identity switches), not human motion.
MAX_HUMAN_SPEED_MPS = 12.0


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


def load_atd_arrays(
    game: str = GAME,
    data_dir: Optional[str] = None,
    source: str = "tactical",
) -> Dict[str, np.ndarray]:
    """Stream raw ATD into metric float32 arrays (metres and m/s).

    Returns dict with:
      - player_xy: (N, S, 2) float32 in metres (NaN where unseen)
      - player_speed: (N, S) float32 in m/s from tracker or finite differences
      - player_vel: (N, S, 2) float32 velocity vector (vx, vy) in m/s (identity-switch filtered)
      - ball_xy: (N, 2) float32 in metres (NaN where unseen)
      - slot_teams: (S,) int8 (0=Home, 1=Away)
      - team_names: (home_name, away_name)
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    slot_teams, names = parse_atd_metadata(meta_path)
    n_frames = source_frames(source, game)
    n_slots = len(slot_teams)
    length, width = GAMES[game]["pitch"]
    fps = float(GAMES[game]["fps"])

    player_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    player_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
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
                    player_speed[f_idx, s_idx] = float(s_str)

    # Compute slot velocity vectors (vx, vy) via forward/backward differences, masking ID switches
    player_vel = np.full_like(player_xy, np.nan)
    disp = (player_xy[1:] - player_xy[:-1]) * fps  # (N-1, S, 2) in m/s
    step_speed = np.hypot(disp[:, :, 0], disp[:, :, 1])
    valid_step = np.isfinite(step_speed) & (step_speed <= MAX_HUMAN_SPEED_MPS)

    disp_clean = np.where(valid_step[:, :, None], disp, np.nan)
    player_vel[1:] = disp_clean
    # Back-fill first frame of a valid tracklet segment
    first_only = np.isnan(player_vel[:-1, :, 0]) & np.isfinite(disp_clean[:, :, 0])
    player_vel[:-1] = np.where(first_only[:, :, None], disp_clean, player_vel[:-1])

    # Fill missing scalar speeds from vector speed where available
    vec_speed = np.hypot(player_vel[:, :, 0], player_vel[:, :, 1])
    missing_spd = np.isnan(player_speed) & np.isfinite(vec_speed)
    player_speed[missing_spd] = vec_speed[missing_spd]
    # Clamp unphysical optical tracker speed spikes
    player_speed = np.where(player_speed <= MAX_HUMAN_SPEED_MPS, player_speed, np.nan)

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
    """Return (valid_bool, reprojection_error) arrays of length n_frames."""
    path = require(["homography"], game, data_dir, source)[0]
    df = pd.read_csv(path, usecols=["reprojection_error", "valid"])
    return df["valid"].values.astype(bool), df["reprojection_error"].values.astype(np.float32)


def sync_broadcast_to_tactical(
    tac: Dict[str, np.ndarray],
    game: str = GAME,
    data_dir: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Automatically synchronize Broadcast ATD onto the Tactical timeline per half,
    rejecting TV replays and close-ups via player-constellation nearest-neighbour matching.

    Returns None if broadcast data is unavailable, otherwise a dict aligned to tactical frames:
      - brd_ball_xy: (N_tac, 2) float32 live-verified broadcast ball coordinates in metres
      - live_verified: (N_tac,) bool where broadcast frame is live-synced to tactical
    """
    try:
        brd = load_atd_arrays(game=game, data_dir=data_dir, source="broadcast")
        brd_valid, brd_reproj = load_homography_quality(game=game, data_dir=data_dir, source="broadcast")
    except (IOError, KeyError):
        return None

    usable_cam = brd_valid & (brd_reproj < 4.0)
    brd_ball = brd["ball_xy"].copy()
    brd_players = brd["player_xy"].copy()
    brd_ball[~usable_cam] = np.nan
    brd_players[~usable_cam] = np.nan

    tac_ball = tac["ball_xy"]
    tac_players = tac["player_xy"]
    n_tac = len(tac_ball)
    n_brd = len(brd_ball)

    def compute_team_centroids(pxy: np.ndarray, steams: np.ndarray) -> np.ndarray:
        out = np.full((len(pxy), 4), np.nan, dtype=np.float32)
        h_mask = steams == HOME
        a_mask = steams == AWAY
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

    tac_cent = compute_team_centroids(tac_players, tac["slot_teams"])
    brd_cent = compute_team_centroids(brd_players, brd["slot_teams"])

    brd_ball_aligned = np.full((n_tac, 2), np.nan, dtype=np.float32)
    live_verified = np.zeros(n_tac, dtype=bool)

    blocks = half_video_blocks(game)
    for half_idx, t_start, t_end in blocks:
        # 1. Coarse search over possible offsets (tac_frame - brd_frame) using a central window
        mid_tac = (t_start + t_end) // 2
        win_len = 5000  # 200 seconds
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

        # 2. Fine frame-exact alignment across 5000-frame chunks using ball & player positions
        chunk_size = 5000
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

            # 3. Per-frame Replay Rejection via Player-Constellation Nearest-Neighbour Match:
            #    A broadcast frame is genuinely LIVE iff visible broadcast players coincide
            #    with same-team tactical players at that exact tactical frame.
            tac_h = tac_players[c_start:c_end, tac["slot_teams"] == HOME]
            tac_a = tac_players[c_start:c_end, tac["slot_teams"] == AWAY]
            brd_h = brd_players[b_s:b_e, brd["slot_teams"] == HOME]
            brd_a = brd_players[b_s:b_e, brd["slot_teams"] == AWAY]

            dh = np.hypot(
                brd_h[:, :, None, 0] - tac_h[:, None, :, 0],
                brd_h[:, :, None, 1] - tac_h[:, None, :, 1],
            )
            dh = np.where(np.isfinite(dh), dh, 1e6)
            min_dh = dh.min(axis=2)
            vis_bh = np.isfinite(brd_h[:, :, 0])

            da = np.hypot(
                brd_a[:, :, None, 0] - tac_a[:, None, :, 0],
                brd_a[:, :, None, 1] - tac_a[:, None, :, 1],
            )
            da = np.where(np.isfinite(da), da, 1e6)
            min_da = da.min(axis=2)
            vis_ba = np.isfinite(brd_a[:, :, 0])

            all_min = np.concatenate(
                [np.where(vis_bh, min_dh, np.nan), np.where(vis_ba, min_da, np.nan)],
                axis=1,
            )
            n_vis_b = vis_bh.sum(axis=1) + vis_ba.sum(axis=1)
            med_constellation_err = np.full(c_len, np.nan, dtype=np.float32)
            has_b = n_vis_b >= 4
            if has_b.any():
                med_constellation_err[has_b] = np.nanmedian(all_min[has_b], axis=1)

            is_live = (n_vis_b >= 4) & np.isfinite(med_constellation_err) & (med_constellation_err < 2.2)
            live_verified[c_start:c_end] = is_live

            bb_chunk = brd_ball[b_s:b_e]
            take = is_live[:, None] & np.isfinite(bb_chunk)
            brd_ball_aligned[c_start:c_end] = np.where(take, bb_chunk, np.nan)

    return {
        "brd_ball_xy": brd_ball_aligned,
        "live_verified": live_verified,
    }


def load_smart_tags(game: str = GAME, data_dir: Optional[str] = None) -> List[dict]:
    """Load Smart Tagging instances mapped to tactical video frames."""
    return load_smart_tagging.load(game=game, data_dir=data_dir)
