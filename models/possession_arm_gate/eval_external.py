"""Leave-one-match-out evaluation of possession_arm_gate over the EXTERNAL matches.

The IDENTICAL protocol of every family (models/modules/training_protocol): per
fold i keep out fold i AND the next fold (i+1) % 10, fit the meta-gate on the
18 remaining files using each ingredient's cached protocol-compliant LOO
posterior, score every held-out file with the repository metric (taus 12/25).
Posteriors of the held-out files go to data/_arm_cache/looP_v2_possession_arm_gate_*
so no other family's files are touched; the summary lands in artifacts/.

    python models/possession_arm_gate/eval_external.py            # all folds
    python models/possession_arm_gate/eval_external.py --folds 0 1
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_gate/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.cache import load_all_files
from models.modules.data_external import EXTERNAL_IDS
from models.modules.decoder_support import DEFAULT_PARAMS, DecoderParams
from models.modules.training_protocol import run_nested_loo
from models.possession_arm_gate.train import (
    ARTIFACT_DIR, MODEL_TAG, make_fit_predict, preload_ingredient_posteriors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="subset of folds (default all)")
    parser.add_argument("--min-pos-frames", type=int, default=DEFAULT_PARAMS.min_pos_frames)
    parser.add_argument("--min-stop-frames", type=int, default=DEFAULT_PARAMS.min_stop_frames)
    parser.add_argument("--w-prior", type=float, default=DEFAULT_PARAMS.w_prior)
    parser.add_argument("--C", type=float, default=None,
                        help="gate regularization (default: train.GATE_C)")
    parser.add_argument("--interactions", action="store_true", default=None,
                        help="add tier x log-posterior cross features (MoE mode)")
    parser.add_argument("--blend", choices=("loglinear", "mixture"), default=None,
                        help="gate blend mode (default: loglinear)")
    args = parser.parse_args()

    import models.possession_arm_gate.train as G
    if args.C is not None:
        G.GATE_C = float(args.C)
    if args.interactions:
        G.GATE_CROSS = True
    if args.blend is not None:
        G.GATE_BLEND = args.blend

    params = DecoderParams(min_pos_frames=args.min_pos_frames,
                           min_stop_frames=args.min_stop_frames, w_prior=args.w_prior)
    folds = args.folds if args.folds is not None else list(range(len(EXTERNAL_IDS)))

    files = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")
    post_cache = preload_ingredient_posteriors(files)

    t0 = time.time()
    res = run_nested_loo(files, MODEL_TAG, make_fit_predict(post_cache), params, folds)

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
            "note": "gate: no boosting rounds (meta fit is regularized logistic regression)",
        }, handle, indent=2)
    print(f"saved -> {summary_path}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
