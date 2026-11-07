"""Decoder parameter search for possession_arm_seg (STRICTLY external data).

Same shared engine models/modules/decoder_search on THIS family's posteriors
(they are step-constant per segment, so the search tends to relax the minimum
durations the per-frame families need -- that is the paradigm, not a hack).

    python models/possession_arm_seg/search_decoder.py --rounds 4
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_seg/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.decoder_search import Holdouts, coordinate_search
from models.possession_arm_seg.train import ARTIFACT_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    holdouts = Holdouts("possession_arm_seg")
    holdouts.tags = sorted(holdouts.P)
    if not holdouts.P:
        raise SystemExit("no LOO posteriors: run models/possession_arm_seg/eval_external.py first")
    print(f"holdouts: {len(holdouts.tags)} LOO posterior files "
          f"({sum(1 for t in holdouts.tags if not t[1])} clean / "
          f"{sum(1 for t in holdouts.tags if t[1])} dirty)")
    coordinate_search(holdouts, ARTIFACT_DIR, rounds=args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
