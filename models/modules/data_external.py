"""Loaders for the EXTERNAL training matches under data/altre_partite/.

Every external match is stored in exactly the same Barca format as `data/tactical/`:
    <ID>_atd.txt     raw ATD (one row per frame, frame ids 1..N)
    <ID>_atd.xml     EPTS metadata (teams, slots, pitch)
plus, for the CLEAN set, the ground-truth labels:
    <ID>_possession_gt.csv   frame,label with H/A/D (X where ungraded)

The DIRTIED set shares the same ids and flushes only the tracking (atd.txt/xml); the
labels always come from the CLEAN folder, per data/analisi/LEGGIMI.md.

The competition match must never be mixed into these arrays: this module refuses it.
"""

import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/modules/ -> models/ -> repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Identity-switch ceiling for humans between consecutive frames (0.04 s): the same value
# the provided loader uses (possession_model.data_loader.MAX_HUMAN_SPEED_MPS); it is a
# physical constant for the tracker format, not a tuned parameter.
MAX_HUMAN_SPEED_MPS = 12.0

DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "altre_partite")

EXTERNAL_IDS = [
    "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9",
    "Sample_Game_1", "Sample_Game_2", "Sample_Game_3",
]

HOME = 0
AWAY = 1


def parse_atd_metadata(meta_path: str) -> Tuple[np.ndarray, Tuple[str, str]]:
    """Return (slot_teams, (home_name, away_name)), 0 = Home, 1 = Away.

    Reuses the provided parser so the two code paths cannot drift apart.
    """
    from possession_model.data_loader import parse_atd_metadata as _parse
    return _parse(meta_path)


def _pitch_dims(meta_path: str) -> Tuple[float, float]:
    """Pitch size from the EPTS XML, else the documented default 105 x 68 m."""
    root = ET.parse(meta_path).getroot()
    for node in root.iter():
        if not node.tag.lower().endswith("pitchdimensions"):
            continue
        xl = yw = None
        for child in node.iter():
            tag = child.tag.split("}")[-1].lower()
            text = (child.text or "").strip()
            if not text or not text.replace(".", "", 1).replace("-", "", 1).isdigit():
                continue
            if tag in ("sizex", "length3d", "length2d", "pitchlength", "length"):
                xl = max(xl or 0.0, float(text))
            elif tag in ("sizey", "width3d", "width2d", "pitchwidth", "width"):
                yw = max(yw or 0.0, float(text))
        if xl and yw and 80 < xl < 120 and 55 < yw < 85:
            return float(xl), float(yw)
    return 105.0, 68.0


def count_lines(path: str) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def read_atd_arrays(atd_raw: str, n_slots: int, pitch: Tuple[float, float],
                    n_frames: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Stream a Barca-format ATD text file into metre arrays.

    Mirrors possession_model.data_loader.load_atd_arrays exactly (same math, same
    velocity-derivation and same identity-switch filter), BUT with explicit paths so
    external matches can be read without touching the games.py registry. Returns:
      player_xy (N, S, 2)   player_speed (N, S)   player_vel (N, S, 2)
      ball_xy (N, 2)        -- all float32, metres, NaN where unseen
    """
    if n_frames is None:
        n_frames = count_lines(atd_raw)
    length, width = pitch
    fps = 25.0

    player_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    player_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(atd_raw, "r", encoding="utf-8") as handle:
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
                ball_xy[f_idx, 1] = float(ball_str[c1 + 1: c2]) * width

            for s_idx, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N" or s_idx >= n_slots:
                    continue
                c1 = group.find(",")
                c2 = group.find(",", c1 + 1)
                player_xy[f_idx, s_idx, 0] = float(group[:c1]) * length
                player_xy[f_idx, s_idx, 1] = float(group[c1 + 1: c2]) * width
                s_str = group[c2 + 1:]
                if s_str and s_str[0] != "N":
                    player_speed[f_idx, s_idx] = float(s_str)

    # Velocity vectors via central/one-sided differences with the identity-switch mask
    player_vel = np.full_like(player_xy, np.nan)
    disp = (player_xy[1:] - player_xy[:-1]) * fps
    step_speed = np.hypot(disp[:, :, 0], disp[:, :, 1])
    valid_step = np.isfinite(step_speed) & (step_speed <= MAX_HUMAN_SPEED_MPS)
    disp_clean = np.where(valid_step[:, :, None], disp, np.nan)
    player_vel[1:] = disp_clean
    first_only = np.isnan(player_vel[:-1, :, 0]) & np.isfinite(disp_clean[:, :, 0])
    player_vel[:-1] = np.where(first_only[:, :, None], disp_clean, player_vel[:-1])

    vec_speed = np.hypot(player_vel[:, :, 0], player_vel[:, :, 1])
    missing_spd = np.isnan(player_speed) & np.isfinite(vec_speed)
    player_speed[missing_spd] = vec_speed[missing_spd]
    player_speed = np.where(player_speed <= MAX_HUMAN_SPEED_MPS, player_speed, np.nan)

    return {
        "player_xy": player_xy,
        "player_speed": player_speed,
        "player_vel": player_vel,
        "ball_xy": ball_xy,
        "fps": fps,
        "pitch": pitch,
    }


def match_paths(match_id: str, dirty: bool, data_root: str = DEFAULT_DATA_ROOT) -> Dict[str, str]:
    family = "dfl" if match_id.startswith("J03") else "metrica"
    subset = f"{family}_sporcate" if dirty else f"{family}_pulite"
    folder = os.path.join(data_root, subset, match_id)
    return {
        "atd_txt": os.path.join(folder, f"{match_id}_atd.txt"),
        "atd_xml": os.path.join(folder, f"{match_id}_atd.xml"),
        "gt_csv": os.path.join(data_root, f"{family}_pulite", match_id, f"{match_id}_possession_gt.csv"),
        "name": match_id,
    }


def guard_external_paths(paths: Dict[str, str]) -> None:
    """Fail fast if a path points anywhere inside data/tactical or data/broadcast."""
    for key in ("atd_txt", "atd_xml", "gt_csv"):
        low = paths[key].lower().replace("\\", "/")
        if "tactical" in low or "broadcast" in low:
            raise ValueError(f"refusing to use competition asset as external data: {paths[key]}")


LABEL_TO_CODE = {"H": 0, "A": 1, "D": 2, "X": -1}
CODE_TO_LABEL = np.array(["H", "A", "D"])


def read_gt_labels(gt_csv: str) -> np.ndarray:
    """frame,label CSV -> int8 codes in {-1..2}; validates that frame ids are 0..N-1."""
    df = pd.read_csv(gt_csv)
    if list(df.columns[:2]) != ["frame", "label"]:
        raise ValueError(f"{gt_csv}: bad header {list(df.columns[:2])}")
    frames = df["frame"].values
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError(f"{gt_csv}: frame column is not 0..N-1")
    labels = df["label"].astype(str).str.strip().str.upper().values
    known = np.isin(labels, list(LABEL_TO_CODE))
    if not known.all():
        raise ValueError(f"{gt_csv}: unknown labels {set(labels[~known])}")
    out = np.array([LABEL_TO_CODE[v] for v in labels], dtype=np.int8)
    return out


def detect_half_blocks(labels: np.ndarray, fps: float = 25.0) -> List[Tuple[int, int, int]]:
    """Split the external-match timeline into consecutive blocks.

    External matches publish no calendar, so half-time breaks are detected from the
    labels alone: no gameplay stoppage in a professional match lasts 8 minutes, so any
    D-or-X run at least that long is a break. A missed break costs only the
    attack-direction features on one half -- the decoder treats a long dead interval
    as one stoppage regardless, so no label depends on break boundaries.

    Break frames are assigned to the FOLLOWING block (the re-setup), and consecutive
    blocks are numbered 1..K.

    Returns
    -------
    blocks : list of (block_num, start_frame, end_frame) inclusive.
    """
    code = np.asarray(labels)
    gap = (code == 2) | (code == -1)
    n = len(code)
    runs: List[Tuple[int, int]] = []
    start = None
    for i in range(n):
        if gap[i] and start is None:
            start = i
        elif not gap[i] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))

    min_break_frames = int(8 * 60 * fps)
    edges = [0]
    for s, e in sorted(runs, key=lambda p: p[0]):
        if (e - s + 1) >= min_break_frames:
            edges.append(e + 1)
    if edges[-1] != n:
        edges.append(n)

    blocks = []
    for k in range(len(edges) - 1):
        s, e = edges[k], edges[k + 1] - 1
        if s <= e:
            blocks.append((k + 1, int(s), int(e)))
    return blocks


def load_external_match(match_id: str, dirty: bool, data_root: str = DEFAULT_DATA_ROOT) -> Dict:
    """Full external match: cleaned tracking (+ frictions when dirty) and always the CLEAN labels."""
    paths = match_paths(match_id, dirty, data_root)
    guard_external_paths(paths)
    slot_teams, team_names = parse_atd_metadata(paths["atd_xml"])
    pitch = _pitch_dims(paths["atd_xml"])
    n_slots = len(slot_teams)
    tracking = read_atd_arrays(paths["atd_txt"], n_slots, pitch)
    labels = read_gt_labels(paths["gt_csv"])
    if len(labels) != len(tracking["ball_xy"]):
        raise ValueError(
            f"{match_id}: labels={len(labels)} frames vs tracking={len(tracking['ball_xy'])}"
        )
    blocks = detect_half_blocks(labels)
    return {
        "name": match_id,
        "dirty": dirty,
        "paths": paths,
        "slot_teams": slot_teams,
        "team_names": team_names,
        "pitch": pitch,
        **tracking,
        "labels": labels,
        "blocks": blocks,
    }
