"""Decoder parameter search for possession_arm_ens (STRICTLY external data).

Shared engine models/modules/decoder_search on the COMPOSED ensemble
posteriors (looP_v2_possession_arm_ens_* written by train.py --mode loo).

    python models/possession_arm_ens/search_decoder.py --rounds 4
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_ens/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.decoder_search import Holdouts, coordinate_search
from models.possession_arm_ens.train import ARTIFACT_DIR, MODEL_TAG


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--artifact-dir", default=None,
                        help="variant output dir (default: the family artifacts; "
                             "use a subdir to avoid clobbering the production params)")
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir is not None else ARTIFACT_DIR

    holdouts = Holdouts(MODEL_TAG)
    holdouts.tags = sorted(holdouts.P)
    if not holdouts.P:
        raise SystemExit("no composed posteriors: run models/possession_arm_ens/train.py --mode loo first")
    print(f"holdouts: {len(holdouts.tags)} composed posterior files")
    print(f"artifact dir: {artifact_dir}")
    coordinate_search(holdouts, artifact_dir, rounds=args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
