"""Decoder parameter search for possession_arm_gate (STRICTLY external data).

Shared engine models/modules/decoder_search on the gate's LOO posteriors
(looP_v2_possession_arm_gate_* written by eval_external.py).

    python models/possession_arm_gate/search_decoder.py --rounds 4
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_gate/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.decoder_search import Holdouts, coordinate_search
from models.possession_arm_gate.train import ARTIFACT_DIR, MODEL_TAG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    holdouts = Holdouts(MODEL_TAG)
    holdouts.tags = sorted(holdouts.P)
    if not holdouts.P:
        raise SystemExit("no gate posteriors: run models/possession_arm_gate/eval_external.py first")
    print(f"holdouts: {len(holdouts.tags)} gate posterior files")
    coordinate_search(holdouts, ARTIFACT_DIR, rounds=args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
