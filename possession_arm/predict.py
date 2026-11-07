"""Run possession_arm_v1 on the competition match and write the submission CSV.

Full chain: load tactical ATD through the provided loader -> build the Stage-1
features -> Stage-1 XGBoost posteriors (final artifact) -> Stage-2 duration-
constrained Viterbi with the LEARNED decoder parameters -> frame,label CSV.

The public 5-minute ground truth is used only at the end as a pipeline sanity
check: its score is reported to a JSON, never fed back into anything that adjusts
parameters or the model (RULES.md compliance).

    python possession_arm/predict.py                 # write + sanity check
    python possession_arm/predict.py --out PATH     # custom CSV path
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
from possession_model.run import write_submission  # noqa: E402
from possession_arm.context import build_context  # noqa: E402
from possession_arm.context import context_badge  # noqa: E402
from possession_arm.decode import params_from_dict, decode_match  # noqa: E402
from possession_arm.features import build_feature_matrix  # noqa: E402
from possession_arm.target_match import load_target_match  # noqa: E402

ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "possession_arm_v1")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_final.json")
PARAMS_PATH = os.path.join(ARTIFACT_DIR, "decoder_params.json")
SANITY_PATH = os.path.join(ARTIFACT_DIR, "target_sanity.json")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def load_model_and_params() -> tuple:
    import xgboost as xgb

    with open(MODEL_PATH.replace(".json", "_meta.json"), "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    names = meta["feature_names"]
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    prior = None
    params = params_from_dict({"min_pos_frames": 45, "min_stop_frames": 85, "w_prior": 0.0})
    if os.path.isfile(PARAMS_PATH):
        with open(PARAMS_PATH, "r", encoding="utf-8") as handle:
            saved = json.load(handle)
        params = params_from_dict(saved["params"])
        prior = saved.get("prior")
        print(f"decoder params from {PARAMS_PATH}: "
              f"min_pos={params.min_pos_frames} min_stop={params.min_stop_frames} "
              f"w_prior={params.w_prior} prior={[round(float(x), 4) for x in prior] if prior is not None else None}")
    else:
        print("decoder params: no learned file found -> defaults copied from the provided pipeline")
    return booster, names, params, prior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "possession_arm_v1.csv"))
    args = parser.parse_args()

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

    print("[3/4] posterior + decode...")
    booster, model_names, params, prior = load_model_and_params()
    if model_names != names:
        raise RuntimeError("feature-name mismatch: model expects different feature order")
    import xgboost as xgb
    dmatrix = xgb.DMatrix(X, feature_names=names)
    P = np.asarray(booster.predict(dmatrix)).reshape(-1, 3).astype(np.float32)
    del dmatrix
    states = decode_match(P, ctx, blocks, params, prior=prior)
    labels = np.full(n, "X", dtype="<U1")
    code_map = np.array(("H", "A", "D"), dtype="<U1")
    live = states >= 0
    labels[live] = code_map[states[live]]

    print(f"[4/4] writing {args.out}")
    out = write_submission(args.out, labels)
    print(f"      rows={len(labels)}  live={int(live.sum())}  X={n - int(live.sum())}")

    # ---- public-GT sanity checkonly a report, never a tuning signal ----
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
