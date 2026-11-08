"""Run the situation-gated meta-blend on the competition match (submission CSV).

The target posterior = the gate's blend of the production trio's FINAL
artifacts on the shared v2 features (same situation features as in training:
per-ingredient log-posteriors + entropies + the six context tiers); the CSV +
public-GT sanity follow the same shape as every other family's predict script.
The gate itself was validated the identical external-LOO way (double nesting:
ingredient posteriors are per-family out-of-fold by construction), so nothing
here tunes on the target.

    python models/possession_arm_gate/predict.py                       # write + sanity
"""

import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_gate/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME  # noqa: E402
from possession_model.run import write_submission  # noqa: E402
from models.modules.context import context_badge  # noqa: E402
from models.modules.context import build_context  # noqa: E402
from models.modules.decoder_support import decode_match, params_from_dict  # noqa: E402
from models.modules.features_base import build_feature_matrix  # noqa: E402
from models.modules.target_match import load_target_match  # noqa: E402
from models.possession_arm_gate.train import (
    ARTIFACT_DIR, GATE_FAMILIES, gate_posterior, meta_features)

PARAMS_PATH = os.path.join(ARTIFACT_DIR, "decoder_params.json")
SANITY_PATH = os.path.join(ARTIFACT_DIR, "target_sanity.json")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def v1_posterior(X: np.ndarray, names) -> np.ndarray:
    """Port of possession_arm_v1/predict.py's posterior (only needed if v1 joins the blend)."""
    import xgboost as xgb

    from models.possession_arm_v1.predict import load_model_and_params
    booster, inames, _params, _prior = load_model_and_params()
    if list(inames) != list(names):
        raise RuntimeError(f"v1 feature mismatch: {inames[:3]}...")
    dmatrix = xgb.DMatrix(X, feature_names=names)
    return np.asarray(booster.predict(dmatrix)).reshape(-1, 3).astype(np.float32)


def ingredient_posterior(family: str, X: np.ndarray, names, f_dict) -> np.ndarray:
    """One ingredient family's FINAL-artifact posterior on the target matrix."""
    if family == "possession_arm_v1":
        return v1_posterior(X, names)
    if family == "possession_arm_lgbm":
        from models.possession_arm_lgbm.train import predict_final
        P, inames = predict_final(X)
    elif family == "possession_arm_catboost":
        from models.possession_arm_catboost.train import predict_final
        P, inames = predict_final(X)
    elif family == "possession_arm_tcn":
        from models.possession_arm_tcn.train import predict_final
        P, inames = predict_final(f_dict)
    elif family == "possession_arm_crf":
        from models.possession_arm_crf.train import predict_final
        P, inames = predict_final(f_dict)
    elif family == "possession_arm_seg":
        from models.possession_arm_seg.train import predict_final
        P, inames = predict_final(f_dict), names
    else:
        raise RuntimeError(f"unknown ingredient family: {family}")
    if list(inames) != list(names):
        raise RuntimeError(f"{family} feature mismatch: {inames[:3]}...")
    return np.asarray(P, dtype=np.float32).reshape(-1, 3)


def load_gate_artifact() -> Tuple[np.ndarray, ...]:
    npz_path = os.path.join(ARTIFACT_DIR, "gate_final.npz")
    if not os.path.isfile(npz_path):
        raise SystemExit(f"missing gate artifact: {npz_path}\n"
                         "run: python models/possession_arm_gate/train.py --mode final")
    meta_path = os.path.splitext(npz_path)[0] + "_meta.json"
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(npz_path, allow_pickle=False) as z:
        coef = np.asarray(z["coef"], dtype=np.float64)
        intercept = np.asarray(z["intercept"], dtype=np.float64)
        mean = np.asarray(z["mean"], dtype=np.float64)
        scale = np.asarray(z["scale"], dtype=np.float64)
    print(f"gate artifact: {len(meta['families'])} ingredients, "
          f"{coef.shape[1]} features (C={meta['C']}, cross={meta.get('use_cross', False)})")
    return coef, intercept, mean, scale, bool(meta.get("use_cross", False))


def load_params():
    params = params_from_dict({"min_pos_frames": 45, "min_stop_frames": 85, "w_prior": 0.0})
    prior = None
    if os.path.isfile(PARAMS_PATH):
        with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        params = params_from_dict(saved["params"])
        prior = saved.get("prior")
        print(f"decoder params from {PARAMS_PATH}: "
              f"min_pos={params.min_pos_frames} min_stop={params.min_stop_frames} "
              f"w_prior={params.w_prior} "
              f"prior={[round(float(x), 4) for x in prior] if prior is not None else None}")
    else:
        print("decoder params: no learned file found -> defaults copied from the provided pipeline")
    return params, prior


def main() -> int:
    t0 = time.time()
    game = GAME
    match = load_target_match(game)
    blocks = match["blocks"]
    n = len(match["ball_xy"])
    print(f"target {game}: {n} frames, blocks={[(int(a), int(b)) for _, a, b in blocks]} "
          f"pitch={match['pitch']}, slots={len(match['slot_teams'])}")

    print("[1/4] features...")
    X, names, side = build_feature_matrix(match, blocks)
    print(f"      X={X.shape} nan-rate={float(np.isnan(X).mean()):.3f}")
    print("[2/4] context masks...")
    ctx = build_context(match, blocks, side)
    print(f"      tiers: {context_badge(ctx)}")

    print("[3/4] ingredient posteriors + gated blend...")
    f_dict = {"match_id": GAME, "dirty": False, "X": X, "labels": None,
              "names": names, "blocks": np.asarray(blocks, dtype=np.int64), "ctx": ctx}
    Ps = []
    for family in GATE_FAMILIES:
        P = ingredient_posterior(family, X, names, f_dict)
        if P.shape != (n, 3):
            raise RuntimeError(f"{family}: posterior shape {P.shape} != ({n}, 3)")
        Ps.append(P)
        print(f"      {family:<25} P ok (rowsum {float(np.abs(np.asarray(P, np.float64).sum(1) - 1).max()):.2e})")

    artifact_meta = json.load(open(os.path.join(ARTIFACT_DIR, "gate_final_meta.json"),
                                   "r", encoding="utf-8"))
    blend = artifact_meta.get("blend", "loglinear")
    print(f"      gate blend: {blend}")
    if blend == "mixture":
        from models.possession_arm_gate.mixture import apply_mixture, load_final
        gate = load_final()
        P = apply_mixture(Ps, ctx, gate["mean"], gate["scale"], gate["W"], gate["b"])
    else:
        coef, intercept, mean, scale, gate_use_cross = load_gate_artifact()
        gate_feats, gate_names = meta_features(Ps, ctx, use_cross=gate_use_cross)
        P = gate_posterior(gate_feats, coef, intercept, mean, scale)
    print(f"      gate blend ok (rowsum {float(np.abs(P.sum(1) - 1).max()):.2e})")

    params, prior = load_params()
    states = decode_match(P, ctx, blocks, params, prior=prior)
    labels = np.full(n, "X", dtype="<U1")
    code_map = np.array(("H", "A", "D"), dtype="<U1")
    live = states >= 0
    labels[live] = code_map[states[live]]

    print("[4/4] writing evaluation/results/possession_arm_gate.csv")
    out = write_submission(os.path.join(ROOT, "evaluation", "results", "possession_arm_gate.csv"),
                           labels)
    print(f"      rows={len(labels)}  live={int(live.sum())}  X={n - int(live.sum())}")

    # ---- public-GT sanity: only a report, never a tuning signal ----
    try:
        from evaluation import possession_metrics as pm
        gt_codes = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        pred_eval = np.where(states < 0, 3, states).astype(np.int8)
        if len(pred_eval) == len(gt_codes):
            result = pm.evaluate(gt_codes, pred_eval, taus=(12, 25), bootstrap_n=0)
            t25 = result["transitions"][25]
            sanity = {
                "path": out,
                "spell_f1_mean": float(result["spell_f1_mean"]),
                "macro_f1": float(result["macro_f1"]),
                "acc": float(result["accuracy"]),
                "trans_f1_25": float(t25["f1"]),
                "gt_spells": int(result["n_gt_spells"]),
                "pred_spells": int(result["n_pred_spells"]),
                "per_class": {k: float(v["f1"]) for k, v in result["per_class"].items()},
                "note": "sanity report on the PUBLIC slice only; no parameter was chosen here",
            }
            with open(SANITY_PATH, "w", encoding="utf-8") as handle:
                json.dump(sanity, handle, indent=2)
            print(f"sanity (public GT): spellF1={sanity['spell_f1_mean']:.4f} "
                  f"macro={sanity['macro_f1']:.4f} spells {sanity['gt_spells']}/{sanity['pred_spells']}")
    except Exception as error:  # noqa: BLE001 -- the sanity check must never kill the run
        print(f"sanity check skipped: {error}")

    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
