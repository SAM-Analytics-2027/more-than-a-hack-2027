"""Stage-1 training for possession_arm_catboost: CatBoost multiclass (oblivious
trees sibling of possession_arm_v1's depth-wise XGBoost), same v2 features.

    python models/possession_arm_catboost/train.py --mode final             all-10-matches fit
    python models/possession_arm_catboost/train.py --mode loo --fold 3     one LOO fold
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_catboost/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from models.modules.cache import load_all_files
from models.modules.data_external import EXTERNAL_IDS
from models.modules.training_protocol import fold_of, stack_labeled

MODEL_TAG = "possession_arm_catboost"

ARTIFACT_DIR = os.path.join(HERE, "artifacts")

# CatBoost multiclass on the same scale as v1's XGBoost (eta 0.08, depth 8,
# l2 2.0, subsample 0.95, border_count 512). Class weights reuse the SAME
# balanced alpha scheme via `class_weights`. n_rounds = LOO early stopping.
CATBOOST_BASE_PARAMS = {
    "loss_function": "MultiClass",
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 2.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.95,
    "border_count": 512,
    "random_seed": 2027,
    "allow_writing_files": False,
    "logging_level": "Silent",
}
EARLY_STOP_ROUNDS = 40      # same stopping constant as v1 (identical protocol)
MIN_BOOST_ROUNDS = 150
MAX_BOOST_ROUNDS = 1200

_DEVICE: Dict[str, str] = {}


def resolve_device(verbose: bool = True) -> str:
    """Probe CatBoost CUDA once; any failure falls back to CPU (same policy as v1)."""
    if "device" in _DEVICE:
        return _DEVICE["device"]
    try:
        from catboost import CatBoostClassifier

        _probe = CatBoostClassifier(
            iterations=2, depth=2, loss_function="MultiClass", task_type="GPU",
            logging_level="Silent", allow_writing_files=False)
        rng = np.random.default_rng(0)
        _probe.fit(rng.random((256, 4)), rng.integers(0, 3, 256), logging_level="Silent")
        del _probe
        _DEVICE["device"] = "GPU"
        if verbose:
            print("device: cuda probe OK")
    except Exception as error:  # noqa: BLE001 -- the probe must never kill the run
        _DEVICE["device"] = "CPU"
        if verbose:
            print(f"device: cuda probe failed -> CPU ({error!r})")
    return _DEVICE["device"]


def fit_fold(train_files: List[Dict], eval_files: Optional[List[Dict]] = None,
             n_estimators: Optional[int] = None, verbose: bool = True) -> Tuple:
    """Fit one CatBoost multiclass model; returns (model, names, best_iteration).

    Early stopping mirrors v1: MultiClass loss on the NEXT fold's files,
    40-round patience, best_iteration floored at 150 (balanced class_weights).
    """
    from catboost import CatBoostClassifier

    X, y, _w_unused = stack_labeled(train_files)
    names = train_files[0]["names"]
    counts = [max(1, int((y == c).sum())) for c in (0, 1, 2)]
    n_total = len(y)
    class_weights = [n_total / (3.0 * counts[c]) for c in (0, 1, 2)]

    params = dict(CATBOOST_BASE_PARAMS)
    params["task_type"] = resolve_device(False)
    params["class_weights"] = class_weights
    n_rounds = n_estimators or MAX_BOOST_ROUNDS

    if eval_files and n_estimators is None:
        Xv, yv, _ = stack_labeled(eval_files)
        model = CatBoostClassifier(iterations=n_rounds,
                                   early_stopping_rounds=EARLY_STOP_ROUNDS,
                                   use_best_model=True, **params)
        model.fit(X, y, eval_set=(Xv, yv), verbose=False)
        raw = int(model.get_best_iteration()) + 1 if model.get_best_iteration() is not None else n_rounds
        best_iteration = max(MIN_BOOST_ROUNDS, raw)
        if verbose:
            print(f"  fit rounds={best_iteration} (best_iter={raw})")
    else:
        model = CatBoostClassifier(iterations=n_rounds, **params)
        model.fit(X, y, verbose=False)
        best_iteration = None
        if verbose:
            print(f"  fit rounds={n_rounds} (no early stopping: fixed count)")
    return model, names, best_iteration


def make_fit_predict():
    """Adapter for models/modules/training_protocol.run_nested_loo (file-dict contract)."""

    def fit_predict(train_files: List[Dict], stop_files: List[Dict]) -> Tuple:
        model, names, best_it = fit_fold(train_files, eval_files=stop_files)

        def predict_fn(file_dict: Dict) -> np.ndarray:
            P = model.predict_proba(file_dict["X"].astype(np.float32))
            return np.asarray(P, dtype=np.float32).reshape(-1, 3)

        return predict_fn, {"best_iteration": best_it}

    return fit_predict


def model_path(mode: str) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    return os.path.join(ARTIFACT_DIR, f"catboost_{mode}.cbm")


def save_model(model, names: List[str], meta: Dict, path: str) -> None:
    model.save_model(path, format="cbm")
    with open(os.path.splitext(path)[0] + "_meta.json", "w", encoding="utf-8") as handle:
        json.dump({"feature_names": names, **meta}, handle, indent=2, sort_keys=True)


def predict_final(X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Competition-time posterior from the final artifact (kept next to training)."""
    from catboost import CatBoostClassifier

    path = model_path("final")
    with open(os.path.splitext(path)[0] + "_meta.json", "r", encoding="utf-8") as handle:
        names = json.load(handle)["feature_names"]
    model = CatBoostClassifier()
    model.load_model(path, format="cbm")
    P = model.predict_proba(X.astype(np.float32))
    return np.asarray(P, dtype=np.float32).reshape(-1, 3), names


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
        n_final = None
        summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                rounds = json.load(handle)["rounds"]
            n_final = int(np.percentile([r for r in rounds if r], 80))
            print(f"final rounds from LOO p80: {n_final}")
        model, names, _ = fit_fold(files, n_estimators=n_final)
        save_model(
            model, names,
            {"mode": "final", "catboost_params": CATBOOST_BASE_PARAMS, "model_tag": MODEL_TAG,
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
        model, names, best_it = fit_fold(train_files, eval_files=eval_files)
        save_model(
            model, names,
            {"mode": "loo", "fold": fold, "held_out": EXTERNAL_IDS[fold],
             "catboost_params": CATBOOST_BASE_PARAMS, "model_tag": MODEL_TAG},
            model_path(f"loo_fold{fold:02d}"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
