"""Load the competition match through the SAME interface as external matches.

The competition match keeps using the provided loaders exclusively (possession_model
data_loader + games.py), so no second path exists for the match itself. The returned
dict has the exact keys data_external.load_external_match returns.
"""

import os
import sys
from typing import Dict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/modules/ -> models/ -> repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAMES, GAME  # noqa: E402
from possession_model.data_loader import load_atd_arrays  # noqa: E402
from models.modules.features_base import target_blocks  # noqa: E402


def load_target_match(game: str = GAME, use_broadcast: bool = False) -> Dict:
    """Tactical-camera ATD of the competition match, in the arm's neutral schema."""
    arrays = load_atd_arrays(game=game, data_dir=None, source="tactical")
    entry = GAMES[game]
    match = {
        "name": game,
        "dirty": False,
        "paths": None,
        "slot_teams": arrays["slot_teams"],
        "team_names": arrays["team_names"],
        "pitch": tuple(entry["pitch"]),
        "player_xy": arrays["player_xy"],
        "player_speed": arrays["player_speed"],
        "player_vel": arrays["player_vel"],
        "ball_xy": arrays["ball_xy"],
        "fps": float(entry["fps"]),
        "labels": None,                      # never write a competition ground truth here
        "blocks": target_blocks(game),
    }
    if use_broadcast:
        raise NotImplementedError(
            "v1 trains on the tactical camera alone to keep the target inside the "
            "external feature distribution; broadcast fusion is a later stage"
        )
    return match


def embed_block_labels(n_frames: int, blocks, per_block_states) -> np.ndarray:
    """Place each decoded per-block state sequence into a full-length frame array
    (X outside blocks). `per_block_states` is aligned with `blocks`."""
    out = np.full(n_frames, "X", dtype="<U1")
    code_map = np.array(["H", "A", "D"], dtype="<U1")
    for (num, s_f, e_f), states in zip(blocks, per_block_states):
        out[s_f: e_f + 1] = code_map[states]
    return out
