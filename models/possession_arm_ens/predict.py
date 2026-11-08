"""Run the posterior-averaging ensemble on the competition match (submission CSV).

The target posterior = equal-weight mean of every ingredient family's FINAL
artifact on the shared v2 features (each family loaded via its own predict
entry point); the CSV + public-GT sanity follow the same shape as every other
family's predict script. The ensemble itself was validated the identical
external-LOO way (no folding shortcuts), so nothing here tunes on the target.
Default blend = the validated production trio (lgbm+catboost+crf).

    python models/possession_arm_ens/predict.py                       # write + sanity
    python models/possession_arm_ens/predict.py --out PATH --families A B
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_ens/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from games import GAME, GAMES  # noqa: E402
from possession_model.run import write_submission  # noqa: E402
from models.modules.context import build_context  # noqa: E402
from models.modules.context import context_badge  # noqa: E402
from models.modules.decoder_support import params_from_dict, decode_match  # noqa: E402
from models.modules.features_base import build_feature_matrix  # noqa: E402
from models.modules.target_match import load_target_match  # noqa: E402
from models.possession_arm_ens.train import ARTIFACT_DIR, INGREDIENTS

PARAMS_PATH = os.path.join(ARTIFACT_DIR, "decoder_params.json")
SANITY_PATH = os.path.join(ARTIFACT_DIR, "target_sanity.json")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def v1_posterior(X: np.ndarray, names) -> np.ndarray:
    """Faithful port of possession_arm_v1/predict.py's posterior computation."""
    import xgboost as xgb

    from models.possession_arm_v1.predict import load_model_and_params
    booster, inames, _params, _prior = load_model_and_params()
    if list(inames) != list(names):
        raise RuntimeError(f"v1 feature mismatch: {inames[:3]}...")
    dmatrix = xgb.DMatrix(X, feature_names=names)
    P = np.asarray(booster.predict(dmatrix)).reshape(-1, 3).astype(np.float32)
    return P


def ingredient_posterior(family: str, X: np.ndarray, names, f_dict, side=None):
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
    elif family == "possession_arm_rule":
        from models.possession_arm_rule.train import predict_final as rule_final
        P, inames = rule_final(X, side)
    else:
        raise RuntimeError(f"unknown ingredient family: {family}")
    if list(inames) != list(names):
        raise RuntimeError(f"{family} feature mismatch: {inames[:3]}...")
    return np.asarray(P, dtype=np.float32).reshape(-1, 3)


def load_params(params_path: str) -> tuple:
    params = params_from_dict({"min_pos_frames": 45, "min_stop_frames": 85, "w_prior": 0.0})
    prior = None
    if os.path.isfile(params_path):
        with open(params_path, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        params = params_from_dict(saved["params"])
        prior = saved.get("prior")
        print(f"decoder params from {params_path}: "
              f"min_pos={params.min_pos_frames} min_stop={params.min_stop_frames} "
              f"w_prior={params.w_prior} "
              f"prior={[round(float(x), 4) for x in prior] if prior is not None else None}")
    else:
        print(f"decoder params: {params_path} not found -> defaults copied from the provided pipeline")
    return params, prior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=os.path.join(ROOT, "evaluation", "results", "possession_arm_ens.csv"))
    parser.add_argument("--families", nargs="*", default=list(INGREDIENTS))
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="per-family blend weights (same order as --families; default equal)")
    parser.add_argument("--log-blend", action="store_true",
                        help="log-space (geometric) blend, same combiner as train.py")
    parser.add_argument("--params", default=PARAMS_PATH,
                        help="decoder_params.json to decode with (default production)")
    args = parser.parse_args()
    families = list(dict.fromkeys(args.families))
    if not families:
        parser.error("--families cannot be empty")
    weights = list(args.weights) if args.weights is not None else None
    if weights is not None and len(weights) != len(families):
        parser.error("--weights count must match --families")

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

    print(f"[3/4] {len(families)}-ingredient posteriors + decode...")
    f_dict = {"match_id": GAME, "dirty": False, "X": X, "labels": None,
              "names": names, "blocks": np.asarray(blocks, dtype=np.int64), "ctx": ctx}
    Ps = []
    for family in families:
        P = ingredient_posterior(family, X, names, f_dict, side)
        if P.shape != (n, 3):
            raise RuntimeError(f"{family}: posterior shape {P.shape} != ({n}, 3)")
        Ps.append(P)
        print(f"      {family:<25} P ok (rowsum {float(np.abs(P.sum(1) - 1).max()):.2e})")
    stacked = np.stack(Ps, 0)
    blend = ("log-space geometric" if args.log_blend else "arithmetic mean")
    print(f"      blend: {blend}" + (f" weights={weights}" if weights is not None else " equal"))
    if args.log_blend:
        if weights is None:
            weights = np.ones(len(Ps)) / len(Ps)
        W = np.asarray(weights, dtype=np.float64)
        P_mix = np.exp((np.log(np.clip(stacked, 1e-12, None)) * W[:, None, None]).sum(axis=0))
        P_mix /= P_mix.sum(axis=1, keepdims=True)
    else:
        W = np.ones(len(Ps)) if weights is None else np.asarray(weights, dtype=np.float64)
        P_mix = (stacked * W[:, None, None]).sum(axis=0)
    P = P_mix.astype(np.float32)

    params, prior = load_params(args.params)
    states = decode_match(P, ctx, blocks, params, prior=prior)
    labels = np.full(n, "X", dtype="<U1")
    code_map = np.array(("H", "A", "D"), dtype="<U1")
    live = states >= 0
    labels[live] = code_map[states[live]]

    print(f"[4/4] writing {args.out}")
    out = write_submission(args.out, labels)
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
