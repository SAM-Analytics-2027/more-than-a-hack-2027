"""Leave-one-match-out evaluation of possession_arm_v1 over the external matches.

For every fold i (one logical match = its clean AND dirtied file):
  - trains XGBoost on the files of all other folds;
  - uses fold (i+1) % 10 for early stopping, so the held-out match i is touched by
    neither the gradient data nor the round-count selection;
  - decodes the held-out files with the given DecoderParams and scores each against
    its external ground truth via the repository metric (evaluation.possession_metrics),
    spellF1 mean at IoU 0.3..0.7 primary;
  - saves the fold model and the per-file posteriors for the decoder parameter search.

    python possession_arm/eval_external.py            # all folds (long: GPU fits)
    python possession_arm/eval_external.py --folds 0 1

    New location: python models/possession_arm_v1/eval_external.py --folds 0 1
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm  # noqa: E402
from models.modules.cache import CACHE_DIR, loo_posterior_path  # noqa: E402
from models.modules.data_external import EXTERNAL_IDS  # noqa: E402
from models.modules.decoder_support import (  # noqa: E402
    DEFAULT_PARAMS,
    DecoderParams,
    decode_match,
)
from models.possession_arm_v1.train import (  # noqa: E402
    ARTIFACT_DIR,
    fit_fold,
    fold_of,
    load_all,
    save_model,
)


def to_eval_codes(codes: np.ndarray) -> np.ndarray:
    """Internal codes {-1, 0, 1, 2} -> metric codes {0=H, 1=A, 2=D, 3=X}."""
    return np.where(codes < 0, 3, codes).astype(np.int8)


def score_file(gt_labels: np.ndarray, pred_states: np.ndarray) -> Dict:
    gt_codes = to_eval_codes(gt_labels.astype(np.int8))
    pred_codes = to_eval_codes(pred_states.astype(np.int8))
    result = pm.evaluate(gt_codes, pred_codes, taus=(12, 25), bootstrap_n=0)
    t25 = result["transitions"][25]
    return {
        "spell_f1": float(result["spell_f1_mean"]),
        "macro_f1": float(result["macro_f1"]),
        "acc": float(result["accuracy"]),
        "trans_f1_25": float(t25["f1"]),
        "gt_spells": int(result["n_gt_spells"]),
        "pred_spells": int(result["n_pred_spells"]),
        "f1_h": float(result["per_class"]["H"]["f1"]),
        "f1_a": float(result["per_class"]["A"]["f1"]),
        "f1_d": float(result["per_class"]["D"]["f1"]),
    }


def run_fold(fold: int, files: List[Dict], params: DecoderParams) -> Dict:
    """Fit one nested LOO fold, decode+score its held-out files, save posteriors."""
    import xgboost as xgb  # local import: only GPU fitting/predicting needs it

    stop_fold = (fold + 1) % len(EXTERNAL_IDS)
    train_files = [f for f in files if fold_of(f) != fold and fold_of(f) != stop_fold]
    stop_files = [f for f in files if fold_of(f) == stop_fold]
    hold_files = [f for f in files if fold_of(f) == fold]

    t0 = time.time()
    bst, names, best_it = fit_fold(train_files, eval_files=stop_files)
    fit_s = time.time() - t0

    fold_dir = os.path.join(ARTIFACT_DIR, "loo")
    os.makedirs(fold_dir, exist_ok=True)
    save_model(bst, names, {"mode": "loo", "fold": fold, "held_out": EXTERNAL_IDS[fold],
                            "best_iteration": best_it},
               os.path.join(fold_dir, f"xgb_fold{fold:02d}.json"))

    rows, posteriors = [], {}
    for f in hold_files:
        dmatrix = xgb.DMatrix(f["X"], feature_names=names)
        P = np.asarray(bst.predict(dmatrix)).reshape(-1, 3).astype(np.float32)
        del dmatrix
        tags = (f["match_id"], bool(f["dirty"]))
        posteriors[tags] = P
        meta = {"labels": f["labels"], "blocks": [tuple(map(int, b)) for b in f["blocks"]],
                "names": names, "ctx": f["ctx"]}
        states = decode_match(P, meta["ctx"], meta["blocks"], params)
        rows.append((fold, tags, score_file(f["labels"], states)))
        p_path = loo_posterior_path(tags[0], tags[1])
        np.savez_compressed(p_path, P=P, labels=f["labels"],
                            blocks=np.asarray(meta["blocks"], dtype=np.int64))
    return {"rows": rows, "best_iteration": best_it, "fit_s": fit_s}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="subset of folds (default all)")
    parser.add_argument("--min-pos-frames", type=int, default=DEFAULT_PARAMS.min_pos_frames)
    parser.add_argument("--min-stop-frames", type=int, default=DEFAULT_PARAMS.min_stop_frames)
    parser.add_argument("--w-prior", type=float, default=DEFAULT_PARAMS.w_prior)
    args = parser.parse_args()

    params = DecoderParams(min_pos_frames=args.min_pos_frames, min_stop_frames=args.min_stop_frames,
                           w_prior=args.w_prior)
    folds = args.folds if args.folds is not None else list(range(len(EXTERNAL_IDS)))

    files = load_all(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")

    all_rows: List = []
    rounds: List = []
    for fold in folds:
        res = run_fold(fold, files, params)
        for _, tags, s in res["rows"]:
            all_rows.append((fold, tags, s))
            label = f"{tags[0]}{'_d' if tags[1] else ''}"
            print(f"  fold {fold} hold {label:<20}: spellF1={s['spell_f1']:.3f} "
                  f"macro={s['macro_f1']:.3f} spells {s['gt_spells']}/{s['pred_spells']}")
        rounds.append(res["best_iteration"])
        print(f"fold {fold} ({EXTERNAL_IDS[fold]}) fit={res['fit_s']:.0f}s rounds={res['best_iteration']}")

    spell = np.array([s["spell_f1"] for _, _, s in all_rows])
    macro = np.array([s["macro_f1"] for _, _, s in all_rows])
    print(f"\n mean spellF1 = {spell.mean():.4f}  macroF1 = {macro.mean():.4f}  (n={len(spell)})")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({"rounds": rounds, "spell_scores": {f"{t[0]}{'_d' if t[1] else ''}": s
                                                      for _, t, s in all_rows},
                   "params": {"min_pos_frames": params.min_pos_frames,
                              "min_stop_frames": params.min_stop_frames,
                              "w_prior": params.w_prior}}, handle, indent=2)
    print(f"rounds={rounds}\nsaved -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
