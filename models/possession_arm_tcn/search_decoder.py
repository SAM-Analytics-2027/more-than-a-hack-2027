"""Decoder parameter search for possession_arm_tcn (STRICTLY external data).

Thin wrapper around the shared engine models/modules/decoder_search: loads THIS
family's LOO posteriors (looP_v2_possession_arm_tcn_*), runs the same coordinate
descent as possession_arm_v1 over the shared PARAM_GRID, and writes
decoder_params.json + decoder_search_log.json into this model's artifacts dir.

    python models/possession_arm_tcn/search_decoder.py --rounds 4
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_tcn/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.decoder_search import Holdouts, coordinate_search
from models.possession_arm_tcn.train import ARTIFACT_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    holdouts = Holdouts("possession_arm_tcn")
    holdouts.tags = sorted(holdouts.P)
    if not holdouts.P:
        raise SystemExit("no LOO posteriors: run models/possession_arm_tcn/eval_external.py first")
    print(f"holdouts: {len(holdouts.tags)} LOO posterior files "
          f"({sum(1 for t in holdouts.tags if not t[1])} clean / "
          f"{sum(1 for t in holdouts.tags if t[1])} dirty)")
    coordinate_search(holdouts, ARTIFACT_DIR, rounds=args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
