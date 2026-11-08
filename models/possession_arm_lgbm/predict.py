"""Run possession_arm_lgbm on the competition match and write the submission CSV.

Same shape as possession_arm_v1/predict.py: shared v2 features + context on the
target match, the final LightGBM artifact, the family's own decoder parameters,
the provided Viterbi per half block, then the format-checked CSV plus a public-GT
sanity report (a report only -- never a tuning signal).

    python models/possession_arm_lgbm/predict.py                 # write + sanity check
    python models/possession_arm_lgbm/predict.py --out PATH     # custom CSV path
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_lgbm/ -> models/ -> repo
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
from models.possession_arm_lgbm.train import ARTIFACT_DIR, predict_final

MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgbm_final.txt")
PARAMS_PATH = os.path.join(ARTIFACT_DIR, "decoder_params.json")
SANITY_PATH = os.path.join(ARTIFACT_DIR, "target_sanity.json")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def load_params() -> tuple:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=os.path.join(ROOT, "evaluation", "results", "possession_arm_lgbm.csv"))
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
    P, model_names = predict_final(X)
    if model_names != names:
        raise RuntimeError("feature-name mismatch: model expects different feature order")
    params, prior = load_params()
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
