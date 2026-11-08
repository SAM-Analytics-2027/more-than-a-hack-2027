"""Leave-one-match-out evaluation of possession_arm_rule (external matches).

IDENTICAL protocol via models/modules/training_protocol: the rule family is
fit-free, so the per-fold posterior equals the static one -- the loop exists to
score it under the shared decoder/metric and write the protocol-tagged LOO
posteriors for the decoder search.

    python models/possession_arm_rule/eval_external.py            # all folds (fast)
    python models/possession_arm_rule/eval_external.py --folds 0
"""

import argparse
import json
import os
import sys
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_rule/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.cache import load_all_files  # noqa: E402
from models.modules.data_external import EXTERNAL_IDS  # noqa: E402
from models.modules.decoder_support import DEFAULT_PARAMS, DecoderParams  # noqa: E402
from models.modules.training_protocol import run_nested_loo  # noqa: E402
from models.possession_arm_rule.train import ARTIFACT_DIR, MODEL_TAG, prediction_only_adapter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="subset of folds (default all)")
    parser.add_argument("--min-pos-frames", type=int, default=DEFAULT_PARAMS.min_pos_frames)
    parser.add_argument("--min-stop-frames", type=int, default=DEFAULT_PARAMS.min_stop_frames)
    parser.add_argument("--w-prior", type=float, default=DEFAULT_PARAMS.w_prior)
    args = parser.parse_args()

    params = DecoderParams(min_pos_frames=args.min_pos_frames,
                           min_stop_frames=args.min_stop_frames, w_prior=args.w_prior)
    folds = args.folds if args.folds is not None else list(range(len(EXTERNAL_IDS)))

    files = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")

    t0 = time.time()
    res = run_nested_loo(files, MODEL_TAG, prediction_only_adapter(), params, folds)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "rounds": [int(r if r is not None else 0) for r in res["best_iterations"]],
            "spell_scores": {f"{t[0]}{'_d' if t[1] else ''}": s for _, t, s in res["rows"]},
            "params": {"min_pos_frames": params.min_pos_frames,
                       "min_stop_frames": params.min_stop_frames,
                       "w_prior": params.w_prior},
            "mean_spell_f1": res["mean_spell_f1"],
            "mean_macro_f1": res["mean_macro_f1"],
            "note": "rule family: fit-free (deterministic provided-pipeline posterior, tagless)",
        }, handle, indent=2)
    print(f"saved -> {summary_path}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
