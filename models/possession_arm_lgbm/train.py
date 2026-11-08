"""Stage-1 training for possession_arm_lgbm: LightGBM GBDT (leaf-wise sibling of
possession_arm_v1's depth-wise XGBoost), same v2 features, same external-only rules.

    python models/possession_arm_lgbm/train.py --mode final                all-10-matches fit
    python models/possession_arm_lgbm/train.py --mode loo --fold 3        one LOO fold
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_lgbm/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from models.modules.cache import load_all_files
from models.modules.data_external import EXTERNAL_IDS
from models.modules.training_protocol import fold_of, stack_labeled

MODEL_TAG = "possession_arm_lgbm"

ARTIFACT_DIR = os.path.join(HERE, "artifacts")

# LightGBM multiclass set on the same scale as v1's XGBoost (hist, eta 0.08,
# depth-8-equivalent leaf count, subsample 0.95, feature_fraction 0.75, lambda 2,
# min_data_in_leaf 5, max_bin 512). Nothing is tuned on the competition data:
# n_rounds comes from the shared LOO early-stopping protocol, not guesses.
LGBM_BASE_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.95,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 512,
    "verbosity": -1,
}
EARLY_STOP_ROUNDS = 40      # same stopping constant as v1 (identical protocol)
MIN_BOOST_ROUNDS = 150      # floor so tiny folds do not stop absurdly early
MAX_BOOST_ROUNDS = 1200

_DEVICE: Dict[str, str] = {}


def resolve_device(verbose: bool = True) -> str:
    """Probe LightGBM CUDA once; any failure falls back to CPU.

    Same accelerator policy as v1 (GPU when available) so the GBDT comparison
    stays about the learning algorithm, not about the hardware.
    """
    if "device" in _DEVICE:
        return _DEVICE["device"]
    # CPU only: the pip Windows wheel ships WITHOUT the CUDA/ROCm learners
    # (a GPU build needs a custom -DUSE_CUDA=1 compilation). The inherited
    # GPU-probe promise degrades to a documented CPU policy.
    _DEVICE["device"] = "cpu"
    if verbose:
        print("device: cpu (pip LightGBM wheel has no CUDA learner)")
    return _DEVICE["device"]


def fit_fold(train_files: List[Dict], eval_files: Optional[List[Dict]] = None,
             n_estimators: Optional[int] = None, verbose: bool = True) -> Tuple:
    """Fit one LightGBM multiclass booster; returns (booster, names, best_iteration).

    Early stopping mirrors v1 exactly: multi_logloss on the supplied eval files
    (the NEXT fold), 40-round patience, best_iteration floored at 150.
    """
    import lightgbm as lgb  # local import: only training needs it

    X, y, w = stack_labeled(train_files)
    names = train_files[0]["names"]
    bin_opt = {"max_bin": LGBM_BASE_PARAMS["max_bin"]}
    train_set = lgb.Dataset(X, label=y, weight=w, feature_name=names, params=bin_opt)
    del X

    valid_sets, cb = [], None
    if eval_files and n_estimators is None:
        Xv, yv, wv = stack_labeled(eval_files)
        valid_sets = [lgb.Dataset(Xv, label=yv, weight=wv, feature_name=names,
                                  params=bin_opt, reference=train_set)]
        del Xv, yv
        cb = lgb.early_stopping(EARLY_STOP_ROUNDS, min_delta=0.0, verbose=False)

    params = dict(LGBM_BASE_PARAMS, device_type=resolve_device(False))
    n_rounds = n_estimators or MAX_BOOST_ROUNDS
    if valid_sets:
        booster = lgb.train(params, train_set, num_boost_round=n_rounds,
                            valid_sets=[train_set] + valid_sets,
                            valid_names=["train", "eval"],
                            callbacks=[cb] if cb else [])
        raw = int(booster.best_iteration or n_rounds)
        best_iteration = max(MIN_BOOST_ROUNDS, raw)
        if best_iteration != raw and verbose:
            print(f"  early-stop best_iter={raw} -> floor raised to {best_iteration}")
        if verbose:
            score = booster.best_score
            if isinstance(score, dict):
                per_set = score.get("eval", next(iter(score.values())))
                if isinstance(per_set, dict):
                    per_set = next(iter(per_set.values()))
                score = per_set
            print(f"  fit rounds={best_iteration} (best_iter={raw}, "
                  f"eval_multi_logloss={float(score):.4f})")
    else:
        booster = lgb.train(params, train_set, num_boost_round=n_rounds,
                            valid_sets=[train_set], valid_names=["train"])
        best_iteration = None
        if verbose:
            print(f"  fit rounds={n_rounds} (no early stopping: fixed count)")
    return booster, names, best_iteration


def make_fit_predict():
    """Adapter for models/modules/training_protocol.run_nested_loo.

    Returns fit_predict(train_files, stop_files) -> (predict_fn, info) where
    predict_fn(file_dict) -> [n, 3] float32 posterior (the FULL file-dict
    contract: the dict carries X, labels, names, blocks, ctx).
    """

    def fit_predict(train_files: List[Dict], stop_files: List[Dict]) -> Tuple:
        import lightgbm as lgb_predict  # noqa: F401  (keeps the import lazy)

        booster, names, best_it = fit_fold(train_files, eval_files=stop_files)

        def predict_fn(file_dict: Dict) -> np.ndarray:
            X = np.asarray(file_dict["X"], dtype=np.float32)
            P = booster.predict(X, num_iteration=booster.best_iteration or None)
            return np.asarray(P, dtype=np.float32).reshape(-1, 3)

        return predict_fn, {"best_iteration": best_it}

    return fit_predict


def predict_final(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Competition-time posterior: load the final artifact and predict.

    Kept in one place so predict.py cannot drift from the training protocol.
    """
    import lightgbm as lgb  # local import: only inference needs it

    path = model_path("final")
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    with open(meta_path, "r", encoding="utf-8") as handle:
        names = json.load(handle)["feature_names"]
    booster = lgb.Booster(model_file=path)
    P = booster.predict(X, num_iteration=booster.best_iteration or None)
    return np.asarray(P, dtype=np.float32).reshape(-1, 3), names


def model_path(mode: str, fold: Optional[int] = None) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    tag = "final" if mode == "final" else f"loo_fold{fold:02d}"
    return os.path.join(ARTIFACT_DIR, f"lgbm_{tag}.txt")


def save_model(booster, names: List[str], meta: Dict, path: str) -> None:
    booster.save_model(path)
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"feature_names": names, **meta}, handle, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loo", "final"), default="final")
    parser.add_argument("--fold", type=int, default=None, help="held-out fold for loo mode (0..9)")
    args = parser.parse_args()

    print(f"device: {resolve_device()}")
    files = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")

    if args.mode == "final":
        t0 = time.time()
        # n_estimators from the LOO summary if it exists (written by eval_external)
        n_final = None
        summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                rounds = json.load(handle)["rounds"]
            n_final = int(np.percentile(rounds, 80))
            print(f"final rounds from LOO p80: {n_final}")
        booster, names, _ = fit_fold(files, n_estimators=n_final)
        save_model(
            booster, names,
            {"mode": "final", "lgbm_params": LGBM_BASE_PARAMS, "model_tag": MODEL_TAG,
             "training_files":
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
        booster, names, best_it = fit_fold(train_files, eval_files=eval_files)
        save_model(
            booster, names,
            {"mode": "loo", "fold": fold, "held_out": EXTERNAL_IDS[fold],
             "lgbm_params": LGBM_BASE_PARAMS, "model_tag": MODEL_TAG},
            model_path("loo", fold),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
