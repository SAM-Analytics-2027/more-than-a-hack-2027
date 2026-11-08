"""possession_arm_gate — situation-gated meta-blend of the production trio (family 8).

Stage-1 ingredient posteriors stay EXACTLY as produced by each family's
protocol-compliant nested LOO (loaded from data/_arm_cache). On top of them a
small per-frame meta-model learns to WEIGHT the ingredients as a function of
the situation (the six decoder-context tiers) and of each ingredient's own
uncertainty, instead of the ensemble's fixed equal-weight mean.

Nesting (double LOO, leakage-free by construction):
  - the ingredient posterior for training file j comes from models that were
    themselves leave-one-match-out on j, so the meta-fit never sees a
    posterior that is in-sample for its own match;
  - the gate itself is nested leave-one-match-out through the shared engine:
    fold i fits on 18 files and scores fold i only.

Meta features: log-posterior (3) + normalized entropy (1) per ingredient +
the six context masks. The meta-model is multinomial logistic regression with
fixed C (heavily regularized by design: the effective sample is 10 matches,
not millions of autocorrelated frames) -- no extra hyper-parameter tuning
beyond the shared external decoder-parameter search every family already uses.

    python models/possession_arm_gate/eval_external.py          # all folds LOO
    python models/possession_arm_gate/train.py --mode final     # fit on all 20 files
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_gate/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.context import CONTEXT_KEYS  # noqa: E402
from models.modules.data_external import EXTERNAL_IDS  # noqa: E402
from models.modules.decoder_support import DecoderParams  # noqa: E402
from models.modules.training_protocol import fold_of, run_nested_loo  # noqa: E402
from models.modules.cache import loo_posterior_path  # noqa: E402

MODEL_TAG = "possession_arm_gate"

# Production composition of possession_arm_ens (chosen on external LOO only).
GATE_FAMILIES = ("possession_arm_lgbm", "possession_arm_catboost", "possession_arm_crf")
_SHORT = {"possession_arm_lgbm": "lgb", "possession_arm_catboost": "cb",
          "possession_arm_crf": "crf"}

# Fixed by design (see module docstring); NOT an externally-tuned free lunch.
# GATE_C / GATE_CROSS / GATE_BLEND are settable via CLI (eval_external --C
# --interactions --blend); the artifact records them so predict.py reproduces
# the exact training setup.
GATE_C = 0.3
GATE_CROSS = False
GATE_BLEND = "loglinear"
GATE_MAX_ITER = 400
_LOG_EPS = 1e-7

ARTIFACT_DIR = os.path.join(HERE, "artifacts")


def family_to_cache_tag(family: str):
    """v1 keeps the LEGACY untagged looP names; every other family is tagged."""
    return None if family == "possession_arm_v1" else family


def load_member_posterior(match_id: str, dirty: bool, family: str,
                          allow_missing: bool = False) -> Optional[np.ndarray]:
    path = loo_posterior_path(match_id, dirty, model_tag=family_to_cache_tag(family))
    if not os.path.isfile(path):
        if allow_missing:
            return None
        raise FileNotFoundError(f"missing ingredient LOO posterior: {path}")
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["P"], dtype=np.float64)


def meta_features(Ps: Sequence[np.ndarray], ctx: Dict[str, np.ndarray],
                  use_cross: bool = False) -> Tuple[np.ndarray, List[str]]:
    """One file's gate feature matrix + the feature names.

    Base block: per-ingredient log-posterior (3) + normalized entropy (1), plus
    the six context tiers. With use_cross, ALSO the tier-by-log-posterior
    products (six tiers x three ingredients x three classes) -- these are what
    let the gate shift per-ingredient trust as a function of the situation
    (the actual mixture-of-experts behaviour).
    """
    feats: List[np.ndarray] = []
    names: List[str] = []
    logps: List[np.ndarray] = []
    for family, P in zip(GATE_FAMILIES, Ps):
        short = _SHORT[family]
        P = np.asarray(P, dtype=np.float64)
        logp = np.log(np.clip(P, _LOG_EPS, 1.0))
        logps.append(logp)
        feats.append(logp)
        names += [f"logp_{short}_{c}" for c in ("H", "A", "D")]
        ent = -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(1) / np.log(3.0)
        feats.append(ent[:, None])
        names.append(f"ent_{short}")
    if use_cross:
        for key in CONTEXT_KEYS:
            col = np.asarray(ctx[key], dtype=np.float64).reshape(-1, 1)
            for fam_i, family in enumerate(GATE_FAMILIES):
                feats.append(logps[fam_i] * col)
                names += [f"x_{key}_{_SHORT[family]}_{c}"
                          for c in ("H", "A", "D")]
    for key in CONTEXT_KEYS:
        col = np.asarray(ctx[key], dtype=np.float64)
        feats.append(col.reshape(-1, 1))
        names.append(key)
    return np.concatenate(feats, axis=1), names


def preload_ingredient_posteriors(files) -> Dict[Tuple[str, bool], Dict[str, np.ndarray]]:
    """All ingredient LOO posteriors for all cache files, once, into RAM."""
    out: Dict[Tuple[str, bool], Dict[str, np.ndarray]] = {}
    for f in files:
        key = (f["match_id"], bool(f["dirty"]))
        out[key] = {fam: load_member_posterior(*key, fam) for fam in GATE_FAMILIES}
    return out


def _gate_matrix(f, post_cache) -> Tuple[np.ndarray, np.ndarray]:
    """(X_meta, labels) for one cache-file dict using the preloaded posteriors."""
    ids = (f["match_id"], bool(f["dirty"]))
    Ps = [post_cache[ids][fam] for fam in GATE_FAMILIES]
    X, names = meta_features(Ps, f["ctx"], use_cross=GATE_CROSS)
    return X, np.asarray(f["labels"]), names


def fit_gate_model(train_files, post_cache) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, List[str], int]:
    """Standardized multinomial logistic regression over the meta features.

    Fits ONLY on rows with a ground-truth label. Returns
    (coef, intercept, scaler_mean, scaler_scale, feature_names, n_rows_used).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_parts, y_parts = [], []
    for f in train_files:
        X, y, _ = _gate_matrix(f, post_cache)
        X_parts.append(X)
        y_parts.append(y)
    X = np.concatenate(X_parts).astype(np.float64)
    y = np.concatenate(y_parts).astype(np.int32)
    keep = y >= 0
    X, y = X[keep], y[keep]

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(C=GATE_C, max_iter=GATE_MAX_ITER)
    model.fit(Xs, y)
    return (np.asarray(model.coef_, dtype=np.float64),
            np.asarray(model.intercept_, dtype=np.float64),
            np.asarray(scaler.mean_, dtype=np.float64),
            np.asarray(scaler.scale_, dtype=np.float64),
            _gate_matrix(train_files[0], post_cache)[2], int(keep.sum()))

def gate_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis of a [n, 3] logit matrix."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    P = e / e.sum(axis=1, keepdims=True)
    return P.astype(np.float32)


def gate_posterior(features: np.ndarray, coef, intercept, mean, scale) -> np.ndarray:
    return gate_softmax(((features - mean) / scale) @ coef.T + intercept)


def make_fit_predict(post_cache):
    """Adapter for models/modules/training_protocol.run_nested_loo.

    fit_predict(train_files, stop_files) -> (predict_fn, info):
      - blend='loglinear': regularized multinomial logistic regression on the
        ingredient log-posteriors (+entropy, context tiers, optional cross);
        stop_files are NOT used (no early stopping in sklearn lbfgs) -- kept in
        the signature to satisfy the shared protocol contract;
      - blend='mixture': per-class convex mixture re-router trained with Adam
        (probability space), early-stopped on stop_files exactly like the
        family-level folds (NEXT fold as val).

    In both cases the meta-fit sees only posteriors that are themselves
    out-of-sample for their own match (ingredient cached LOO artifacts).
    """
    if GATE_BLEND == "mixture":
        from models.possession_arm_gate.mixture import apply_mixture, fit_mixture

        def fit_predict(train_files, stop_files):
            params = fit_mixture(train_files, post_cache, stop_files=stop_files)
            W, b, mean, scale = params["W"], params["b"], params["mean"], params["scale"]
            info = {"best_iteration": params["best_epoch"]}

            def predict_fn(file_dict) -> np.ndarray:
                key = (file_dict["match_id"], bool(file_dict["dirty"]))
                Ps = [post_cache[key][fam] for fam in GATE_FAMILIES]
                return apply_mixture(Ps, file_dict["ctx"], mean, scale, W, b)

            return predict_fn, info

        return fit_predict

    def fit_predict(train_files, stop_files):
        coef, intercept, mean, scale, names, n_rows = fit_gate_model(train_files, post_cache)
        info = {"best_iteration": None, "n_train_rows": n_rows}

        def predict_fn(file_dict) -> np.ndarray:
            X, y, _ = _gate_matrix(file_dict, post_cache)
            return gate_posterior(X, coef, intercept, mean, scale)

        return predict_fn, info

    return fit_predict


def fit_final(all_files, post_cache) -> None:
    """Fit the gate on ALL 20 external files (no holdout) and save the artifact.

    The meta inputs are the ingredient families' cached LOO posteriors (always
    out-of-sample for their own match). At competition time predict.py feeds the
    gate the ingredients' FINAL artifacts instead -- the same assumption the
    posterior-mean ensemble already makes and validates end to end.
    """
    coef, intercept, mean, scale, names, n_rows = fit_gate_model(all_files, post_cache)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    npz_path = os.path.join(ARTIFACT_DIR, "gate_final.npz")
    np.savez(npz_path, coef=coef, intercept=intercept, mean=mean, scale=scale)
    meta = {"tag": MODEL_TAG, "families": list(GATE_FAMILIES),
            "feature_names": names, "C": GATE_C, "max_iter": GATE_MAX_ITER,
            "use_cross": GATE_CROSS, "n_train_rows": n_rows,
            "note": "multinomial logistic gate over ingredient log-posteriors + entropy + context tiers"}
    with open(os.path.splitext(npz_path)[0] + "_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(f"final gate saved ({n_rows} labeled rows, {len(names)} features)")


def main() -> int:
    global GATE_C, GATE_CROSS, GATE_BLEND
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loo", "final"), default="final")
    parser.add_argument("--folds", nargs="*", type=int, default=None,
                        help="subset of LOO folds 0..9 (loo mode)")
    parser.add_argument("--min-pos-frames", type=int, default=None)
    parser.add_argument("--min-stop-frames", type=int, default=None)
    parser.add_argument("--w-prior", type=float, default=None)
    parser.add_argument("--C", type=float, default=None, help="loglinear regularization")
    parser.add_argument("--interactions", action="store_true",
                        help="loglinear cross features (tier x log-posterior)")
    parser.add_argument("--blend", choices=("loglinear", "mixture"), default=None)
    args = parser.parse_args()
    if args.C is not None:
        GATE_C = float(args.C)
    if args.interactions:
        GATE_CROSS = True
    if args.blend is not None:
        GATE_BLEND = args.blend

    files_all = None
    post_cache = None
    t0 = time.time()

    if args.mode == "final":
        from models.modules.cache import load_all_files
        files_all = load_all_files(clean_and_dirty=True)
        post_cache = preload_ingredient_posteriors(files_all)
        if GATE_BLEND == "mixture":
            import json as _json  # noqa: F401  (kept local: meta read below)
            from models.possession_arm_gate.mixture import fit_final_and_save
            summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
            if not os.path.isfile(summary_path):
                raise SystemExit("run the mixture LOO first (eval_external.py) -- "
                                 "the final epochs come from its p80")
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = _json.load(handle)
            has_blend_note = "mixture" in str(summary.get("note", ""))
            if not has_blend_note:
                raise SystemExit("loo_rounds_summary.json is not from a mixture LOO run; re-run it")
            best = [e for e in summary["rounds"] if e]
            import math
            epochs_final = max(int(math.floor(0.8 * max(best))), 80)
            print(f"final epochs from LOO p80: {epochs_final}")
            fit_final_and_save(files_all, post_cache, epochs_final)
        else:
            fit_final(files_all, post_cache)
        return 0

    # ---- loo mode: identical nested protocol through the shared engine ----
    from models.modules.cache import load_all_files
    params = DecoderParams(
        min_pos_frames=args.min_pos_frames if args.min_pos_frames is not None else 45,
        min_stop_frames=args.min_stop_frames if args.min_stop_frames is not None else 85,
        w_prior=args.w_prior if args.w_prior is not None else 0.0)
    folds = args.folds if args.folds is not None else list(range(len(EXTERNAL_IDS)))

    files_all = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files_all)} files, feature-dim={len(files_all[0]['names'])}")
    post_cache = preload_ingredient_posteriors(files_all)

    res = run_nested_loo(files_all, MODEL_TAG, make_fit_predict(post_cache), params, folds)

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
            "note": "gate: no boosting rounds (meta fit is regularized softmax regression); "
                    f"C={GATE_C} fixed",
        }, handle, indent=2)
    print(f"saved -> {summary_path}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
