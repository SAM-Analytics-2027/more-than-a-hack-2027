"""Ensemble LOO composition: average the ingredient families' held-out
posteriors fold-by-fold, decode + score with the shared decoder, save the
composed posteriors as looP_v2_possession_arm_ens_* (search input).

    python models/possession_arm_ens/train.py --mode loo                     # production trio
    python models/possession_arm_ens/train.py --mode loo --families possession_arm_v1 possession_arm_lgbm
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

from models.modules.cache import (
    load_all_files, loo_posterior_path,
)
from models.modules.data_external import EXTERNAL_IDS
from models.modules.decoder_support import DEFAULT_PARAMS, DecoderParams, decode_match
from models.modules.scoring import score_file
from models.modules.training_protocol import fold_of

MODEL_TAG = "possession_arm_ens"

ARTIFACT_DIR = os.path.join(HERE, "artifacts")

FAMILIES = ("possession_arm_v1", "possession_arm_lgbm", "possession_arm_catboost",
            "possession_arm_tcn", "possession_arm_seg", "possession_arm_crf")

# PRODUCTION composition: equal-weight mean of the three strongest external-LOO
# families. The blend was chosen under the same external-only rule as every
# other tuning decision here -- see README for the screen table.The full
# six-family mean dilutes the signal (0.6029 default / 0.6314 post-search vs
# the trio's 0.6345 / 0.6539-at-1-round).
INGREDIENTS = ("possession_arm_lgbm", "possession_arm_catboost", "possession_arm_crf")


def family_to_cache_tag(family: str):
    """v1 keeps the LEGACY untagged looP names; every other family is tagged."""
    return None if family == "possession_arm_v1" else family


def validate_families(families, files) -> None:
    missing = []
    for family in families:
        for f in files:
            path = loo_posterior_path(f["match_id"], bool(f["dirty"]),
                                      model_tag=family_to_cache_tag(family))
            if not os.path.isfile(path):
                missing.append(f"{family}: {os.path.basename(path)}")
                break
    if missing:
        raise SystemExit("missing ingredient LOO posteriors ->\n  " + "\n  ".join(missing))


def compose_one(match_id: str, dirty: bool, families, weights=None,
                log_blend: bool = False) -> np.ndarray:
    """Blend the ingredient families' held-out posteriors [n, 3].

    Default: arithmetic weighted mean. --log-blend switches to the log-space
    (geometric) blend rows ∝ prod_f P_f**w_f — each ingredient's log-dominance is
    preserved instead of being diluted by probability averaging, so a peaky D
    ingredient (the tagless rule) suppresses wrong H/A mass instead of washing out.
    Weights must sum to 1 (a proper convex combination in either space).
    """
    Ps = []
    for family in families:
        with np.load(loo_posterior_path(match_id, dirty,
                                        model_tag=family_to_cache_tag(family)),
                     allow_pickle=False) as z:
            Ps.append(np.asarray(z["P"], dtype=np.float64))
    stacked = np.stack(Ps, axis=0)
    if weights is None:
        weights = np.ones(len(Ps), dtype=np.float64) / len(Ps)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(Ps),):
            raise ValueError(f"--weights {weights} does not match {len(Ps)} families")
    if log_blend:
        logs = np.log(np.clip(stacked, 1e-12, None))
        P = np.exp((logs * weights[:, None, None]).sum(axis=0))
        P /= P.sum(axis=1, keepdims=True)
        return P.astype(np.float32)
    return (stacked * weights[:, None, None]).sum(axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loo",), default="loo")
    parser.add_argument("--families", nargs="*", default=list(INGREDIENTS))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="per-family blend weights (same order as --families; default equal)")
    parser.add_argument("--log-blend", action="store_true",
                        help="log-space (geometric) blend instead of the arithmetic mean")
    parser.add_argument("--min-pos-frames", type=int, default=DEFAULT_PARAMS.min_pos_frames)
    parser.add_argument("--min-stop-frames", type=int, default=DEFAULT_PARAMS.min_stop_frames)
    parser.add_argument("--w-prior", type=float, default=DEFAULT_PARAMS.w_prior)
    args = parser.parse_args()

    params = DecoderParams(min_pos_frames=args.min_pos_frames,
                           min_stop_frames=args.min_stop_frames, w_prior=args.w_prior)
    folds = args.folds if args.folds is not None else list(range(len(EXTERNAL_IDS)))
    families = tuple(args.families)
    if not families:
        parser.error("--families cannot be empty")
    weights = list(args.weights) if args.weights is not None else None
    if weights is not None and len(weights) != len(families):
        parser.error("--weights count must match --families")

    files = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")
    print(f"ingredients: {', '.join(families)}")
    print(f"blend: {'log-space geometric' if args.log_blend else 'arithmetic mean'}"
          f"{f'  weights={weights}' if weights is not None else '  equal'}")
    validate_families(families, files)

    all_rows = []
    t0 = time.time()
    for fold in folds:
        hold = [f for f in files if fold_of(f) == fold]
        for f in hold:
            match_id, dirty = f["match_id"], bool(f["dirty"])
            P = compose_one(match_id, dirty, families, weights, args.log_blend)
            blocks = [tuple(map(int, b)) for b in f["blocks"]]
            states = decode_match(P, f["ctx"], blocks, params)
            s = score_file(f["labels"], states)
            all_rows.append((fold, (match_id, dirty), s))
            np.savez_compressed(loo_posterior_path(match_id, dirty, model_tag=MODEL_TAG),
                                P=P, labels=f["labels"],
                                blocks=np.asarray(blocks, dtype=np.int64))
            print(f"  fold {fold} hold {match_id}{'_d' if dirty else '':<2}: "
                  f"spellF1={s['spell_f1']:.3f} macro={s['macro_f1']:.3f} "
                  f"spells {s['gt_spells']}/{s['pred_spells']}")

    spell = np.array([s["spell_f1"] for _, _, s in all_rows])
    macro = np.array([s["macro_f1"] for _, _, s in all_rows])
    mean_spell, mean_macro = float(spell.mean()), float(macro.mean())
    print(f"\n mean spellF1 = {mean_spell:.4f}  macroF1 = {mean_macro:.4f}  (n={len(spell)})")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump({
            "rounds": [],
            "spell_scores": {f"{t[0]}{'_d' if t[1] else ''}": s for _, t, s in all_rows},
            "params": {"min_pos_frames": params.min_pos_frames,
                       "min_stop_frames": params.min_stop_frames,
                       "w_prior": params.w_prior},
            "families": list(families),
            "mean_spell_f1": mean_spell,
            "mean_macro_f1": mean_macro,
            "note": "pure composition of per-family protocol-compliant LOO posteriors; no new fit",
        }, handle, indent=2)
    print(f"saved -> {summary_path}  (total {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
