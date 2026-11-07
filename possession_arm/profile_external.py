"""Print a one-line profile for every external training match: a sanity check that the
loaders read every file and that the label/block detection behaves sensibly.

    python possession_arm/profile_external.py [--dirty]
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession_arm.data_external import EXTERNAL_IDS, load_external_match
from possession_arm.half_blocks import detect_half_split


def profile(match_id: str, dirty: bool) -> None:
    m = load_external_match(match_id, dirty=dirty)
    labels = m["labels"]
    n = len(labels)
    counts = {c: int((labels == c).sum()) for c in (-1, 0, 1, 2)}
    changes = int((labels[1:] != labels[:-1]).sum())
    split = detect_half_split(m["player_xy"], m["slot_teams"], m["pitch"])
    split_txt = "none" if split is None else f"{split}({split / n:.0%})"
    home, away = m["team_names"]
    print(
        f"{m['name']:<14} dirty={int(dirty)} frames={n:>7} slots={len(m['slot_teams']):>2} "
        f"pitch={m['pitch'][0]:.1f}x{m['pitch'][1]:.1f} "
        f"X/H/A/D = {counts[-1]}/{counts[0]}/{counts[1]}/{counts[2]} "
        f"changes={changes:>4} split={split_txt} home='{home}' away='{away}'"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirty", action="store_true", help="load the dirtied tracking")
    parser.add_argument("--ids", nargs="*", default=EXTERNAL_IDS)
    args = parser.parse_args()
    deps = 0
    for match_id in args.ids:
        try:
            profile(match_id, dirty=args.dirty)
        except Exception as error:  # noqa: BLE001 -- the point is reporting every id
            print(f"{match_id:<14} FAILED: {error}")
            deps += 1
    return 1 if deps else 0


if __name__ == "__main__":
    sys.exit(main())
