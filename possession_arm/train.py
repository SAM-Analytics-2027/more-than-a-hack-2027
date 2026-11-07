"""Stage-1 training: XGBoost GBDT (native NaNs, GPU) for possession_arm_v1.

Split discipline (rules compliance):
  - Folds are per MATCH, and the clean + dirtied versions of the same match always
    share one fold, so no held-out match leaks through a near-duplicate file.
  - Class weights come from the training folds only.
  - The competition match is never part of any fit here.

    python possession_arm/train.py --mode final      train on all 10 external matches
    python possession_arm/train.py --mode loo --fold 3   one leave-one-match-out fold
    python possession_arm/train.py --help
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession_arm.cache import CACHE_DIR, cache_path, is_cached  # noqa: E402
from possession_arm.context import CONTEXT_KEYS  # noqa: E402
from possession_arm.data_external import EXTERNAL_IDS, LABEL_TO_CODE  # noqa: E402

ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "possession_arm_v1")

# XGBoost hyper-parameters. Standard multiclass defaults with GPU hist; nothing here
# is tuned on the competition data. n_estimators is picked by early stopping per fold.
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cuda",
    "eta": 0.08,
    "max_depth": 8,
    "min_child_weight": 5.0,
    "subsample": 0.95,
    "colsample_bytree": 0.75,
    "reg_lambda": 2.0,
    "max_bin": 512,
}
EARLY_STOP_ROUNDS = 40
MIN_BOOST_ROUNDS = 150          # floor so tiny folds do not stop absurdly early
MAX_BOOST_ROUNDS = 1200


def _load_file(match_id: str, dirty: bool) -> Dict:
    path = cache_path(match_id, dirty)
    with np.load(path, allow_pickle=False) as z:
        names = list(json.loads(str(z["names"])))
        return {
            "match_id": match_id,
            "dirty": dirty,
            "X": z["X"],
            "labels": z["labels"],
            "names": names,
            "blocks": z["blocks"],
            "ctx": {k: z[f"ctx_{k}"].astype(bool) for k in CONTEXT_KEYS},
        }


def load_all(clean_and_dirty: bool = True) -> List[Dict]:
    files = []
    for mid in EXTERNAL_IDS:
        if is_cached(mid, False):
            files.append(_load_file(mid, False))
        if clean_and_dirty and is_cached(mid, True):
            files.append(_load_file(mid, True))
    if not files:
        raise FileNotFoundError("no cache files: run python possession_arm/cache.py first")
    if len({tuple(f["names"]) for f in files}) != 1:
        raise RuntimeError("feature names differ between cache files: rebuild the cache")
    return files


def fold_of(f: Dict) -> int:
    """Fold index = position of the logical match in EXTERNAL_IDS (clean/dirty share it)."""
    return EXTERNAL_IDS.index(f["match_id"])


def class_weights(labels: np.ndarray) -> np.ndarray:
    """Balanced per-class sample weights from THIS training set only (alpha scheme)."""
    codes = labels[labels >= 0]
    n = len(codes)
    counts = {c: max(1, int((codes == c).sum())) for c in (0, 1, 2)}
    weights = np.empty(n, dtype=np.float32)
    for c, cnt in counts.items():
        mask = codes == c
        weights[mask] = (n / (3.0 * cnt))
    return weights


def fit_fold(train_files: List[Dict], eval_files: Optional[List[Dict]] = None,
             n_estimators: Optional[int] = None, verbose: bool = True) -> Tuple:
    """Fit one XGBoost multiclass model; returns (model, feature_names, best_iteration)."""
    import xgboost as xgb  # local import: only training needs it

    X = np.concatenate([f["X"] for f in train_files]).astype(np.float32)
    y = np.concatenate([f["labels"] for f in train_files]).astype(np.int32)
    keep = y >= 0                       # X (ungraded) rows never enter the fit
    X, y = X[keep], y[keep]
    w = class_weights(y)

    dtrain = xgb.DMatrix(X, label=y, weight=w, feature_names=train_files[0]["names"])
    del X, y, w
    evals = [(dtrain, "train")]
    best_iteration = None
    if eval_files and n_estimators is None:
        dval = xgb.DMatrix(
            np.concatenate([f["X"][f["labels"] >= 0] for f in eval_files]).astype(np.float32),
            label=np.concatenate([f["labels"][f["labels"] >= 0] for f in eval_files]).astype(np.int32),
            feature_names=train_files[0]["names"],
        )
        evals.append((dval, "eval"))
    params = dict(XGB_PARAMS)
    n_rounds = n_estimators or MAX_BOOST_ROUNDS
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        evals=evals,
        early_stopping_rounds=(EARLY_STOP_ROUNDS if n_estimators is None else None),
        verbose_eval=False,
    )
    if n_estimators is None:
        best_iteration = max(MIN_BOOST_ROUNDS, int(bst.best_iteration) + 1)
        if best_iteration != bst.best_iteration + 1:
            if verbose:
                print(f"  early-stop best_iter={bst.best_iteration + 1} -> floor raised to {best_iteration}")
        if verbose:
            print(f"  fit rounds={best_iteration} (best_iter={bst.best_iteration + 1}, "
                  f"eval_mlogloss={bst.best_score:.4f})")
    del dtrain
    return bst, train_files[0]["names"], best_iteration


def model_path(mode: str, fold: Optional[int] = None) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    tag = "final" if mode == "final" else f"loo_fold{fold:02d}"
    return os.path.join(ARTIFACT_DIR, f"xgb_{tag}.json")


def save_model(bst, names: List[str], meta: Dict, path: str) -> None:
    bst.save_model(path)
    meta_path = path.replace(".json", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"feature_names": names, **meta}, handle, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loo", "final"), default="final")
    parser.add_argument("--fold", type=int, default=None, help="held-out fold for loo mode (0..9)")
    parser.add_argument("--singles-camera", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    files = load_all(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")

    if args.mode == "final":
        t0 = time.time()
        # n_estimators from the LOO fold summary if it exists (written by eval_external)
        n_final = None
        summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                rounds = json.load(handle)["rounds"]
            n_final = int(np.percentile(rounds, 80))
            print(f"final rounds from LOO p80: {n_final}")
        bst, names, _ = fit_fold(files, n_estimators=n_final)
        save_model(
            bst, names,
            {"mode": "final", "xgb_params": XGB_PARAMS, "training_files":
                [f"{f['match_id']}{'_dirty' if f['dirty'] else ''}" for f in files],
             "n_estimators": n_final or MAX_BOOST_ROUNDS},
            model_path("final"),
        )
        print(f"final model saved ({time.time() - t0:.1f}s)")
    else:
        fold = args.fold if args.fold is not None else -1
        if not 0 <= fold < len(EXTERNAL_IDS):
            parser.error("--mode loo needs --fold in 0..9")
        train_files = [f for f in files if fold_of(f) != fold]
        eval_files = [f for f in files if fold_of(f) == fold]
        print(f"fold {fold} ({EXTERNAL_IDS[fold]}): train on {len(train_files)} files, "
              f"hold out {len(eval_files)}")
        bst, names, best_it = fit_fold(train_files, eval_files=eval_files)
        save_model(
            bst, names,
            {"mode": "loo", "fold": fold, "held_out": EXTERNAL_IDS[fold],
             "xgb_params": XGB_PARAMS},
            model_path("loo", fold),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
