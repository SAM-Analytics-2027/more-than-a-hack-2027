"""Decoder parameter search for possession_arm_v1 (STRICTLY external data).

Coordinate/grid search over the duration-constrained Viterbi parameters, driven by
the leave-one-out posteriors saved by eval_external.py, so every configuration is
judged on matches the classifier never saw. The objective is the mean spellF1 over
all external files (clean AND dirtied, since the competition tracking is imperfect
too); macro-F1 mean is a reported tiebreaker.

No competition asset is involved anywhere in this search: the posteriors come from
external LOO folds and the context masks come from the external tracking caches.

    python possession_arm/search_decoder.py            # full coordinate descent
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_v1/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation import possession_metrics as pm  # noqa: E402
from models.modules.cache import (  # noqa: E402
    CACHE_DIR,
    cache_path,
    loo_posterior_names,
    parse_posterior_name,
)
from models.modules.decoder_support import (  # noqa: E402
    DEFAULT_PARAMS,
    PARAM_GRID,
    DecoderParams,
    decode_match,
    params_from_dict,
    params_to_dict,
)
from models.possession_arm_v1.train import ARTIFACT_DIR  # noqa: E402


def loo_sample_names() -> List[str]:
    """The looP_<tag>_* posterior files in the cache dir.

    Posteriors do not share the cache_names() vocabulary: they are model artifacts
    appended under the same directory, so the directory is listed directly.
    """
    return loo_posterior_names()      # model_tag=None -> the legacy untagged v1 files

SEARCH_LOG = os.path.join(ARTIFACT_DIR, "decoder_search_log.json")
BEST_PATH = os.path.join(ARTIFACT_DIR, "decoder_params.json")

# Coordinate order: shortest path to the dense spellF1 levers first.
SEARCH_ORDER = (
    "min_pos_frames", "min_stop_frames",
    "turn_quiet", "stop_quiet", "restart_quiet", "w_prior",
    "turn_loud", "turn_control", "turn_occlusion",
    "stop_loud", "restart_loud",
)


class Holdouts:
    """All LOO posteriors + metadata held in RAM, indexed by file tag."""

    def __init__(self) -> None:
        self.tags: List[Tuple[str, bool]] = []
        self.P: Dict[Tuple[str, bool], np.ndarray] = {}
        self.labels: Dict[Tuple[str, bool], np.ndarray] = {}
        self.blocks: Dict[Tuple[str, bool], List[Tuple[int, int, int]]] = {}
        self.ctx: Dict[Tuple[str, bool], Dict[str, np.ndarray]] = {}
        self.gt_codes: Dict[Tuple[str, bool], np.ndarray] = {}
        for name in loo_sample_names():
            match_id, dirty = parse_posterior_name(name)
            tags = (match_id, dirty)
            with np.load(os.path.join(CACHE_DIR, name), allow_pickle=False) as z:
                self.P[tags] = z["P"]
                labels = z["labels"]
                self.labels[tags] = labels
                self.gt_codes[tags] = np.where(labels < 0, 3, labels).astype(np.int8)
                self.blocks[tags] = [tuple(map(int, row)) for row in z["blocks"]]
        # contexts come from the feature caches (same tags)
        for tag in list(self.P):
            path = cache_path(tag[0], tag[1])
            with np.load(path, allow_pickle=False) as z:
                self.ctx[tag] = {k[4:]: z[k].astype(bool) for k in z.files if k.startswith("ctx_")}

    def score(self, params: DecoderParams) -> Tuple[float, float, Dict[str, float]]:
        spell_scores = []
        macro_scores = []
        for tag in self.tags:
            P = self.P[tag]
            states = decode_match(P, self.ctx[tag], self.blocks[tag], params)
            codes = np.where(states < 0, 3, states).astype(np.int8)
            result = pm.evaluate(self.gt_codes[tag], codes, taus=(25,), bootstrap_n=0)
            spell_scores.append(result["spell_f1_mean"])
            macro_scores.append(result["macro_f1"])
        return float(np.mean(spell_scores)), float(np.mean(macro_scores)), dict(
            spell_f1=spell_scores)


def external_class_prior(holdouts: Holdouts) -> np.ndarray:
    """Aggregate H/A/D prior over all EXTERNAL GT files (never the competition GT)."""
    totals = np.zeros(3, dtype=np.float64)
    for tag in holdouts.tags:
        flat = holdouts.gt_codes[tag]
        _flat = flat[flat < 3]
        for c in (0, 1, 2):
            totals[c] += float((_flat == c).sum())
    return totals / totals.sum()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--starts", type=int, default=0, help="random-restart count on top")
    args = parser.parse_args()

    t0 = time.time()
    holdouts = Holdouts()
    holdouts.tags = sorted(holdouts.P)
    n_files = len(holdouts.tags)
    print(f"holdouts: {n_files} LOO posterior files "
          f"({sum(1 for t in holdouts.tags if not t[1])} clean / "
          f"{sum(1 for t in holdouts.tags if t[1])} dirty)")
    prior = external_class_prior(holdouts)
    print(f"external class prior H/A/D = {prior.round(4)}")

    current = params_to_dict(DEFAULT_PARAMS)

    def evaluate(d: Dict) -> Tuple[float, float]:
        params = params_from_dict(d)
        params.prior = prior
        spell, macro, _ = holdouts.score(params)
        return spell, macro

    best_spell, best_macro = evaluate(current)
    print(f"start (default) spellF1={best_spell:.4f}  macroF1={best_macro:.4f}")

    history = [{"step": "init", "spell_f1": best_spell, "params": dict(current)}]
    for round_num in range(args.rounds):
        round_best = best_spell
        for param in SEARCH_ORDER:
            added = tuple(v for v in PARAM_GRID[param] if v not in (current[param],)) \
                    + (current[param],)
            for value in added:
                candidate = dict(current)
                candidate[param] = value
                spell, macro = evaluate(candidate)
                adopted = spell > best_spell + 1e-4
                flag = " ADOPT" if adopted else ""
                print(f"  r{round_num} {param:>16}={value:<6} spell={spell:.4f} "
                      f"macro={macro:.4f}{flag}")
                history.append({"step": f"r{round_num}.{param}", "value": value,
                                "spell_f1": spell, "macro_f1": macro,
                                "adopted": bool(adopted)})
                if adopted:
                    current = candidate
                    best_spell, best_macro = spell, macro
            if best_spell > round_best:
                round_best = best_spell
            os.makedirs(os.path.dirname(SEARCH_LOG), exist_ok=True)
            with open(SEARCH_LOG, "w", encoding="utf-8") as handle:
                json.dump({"best_spell_f1": best_spell, "best_params": current,
                           "elapsed_s": time.time() - t0, "history": history}, handle, indent=2)
        if best_spell <= round_best - 1e-9 and round_num > 0:
            break
        print(f"[round {round_num}] best spellF1={best_spell:.4f} "
              f"(elapsed {time.time() - t0:.0f}s)")

    # final report with the full tau set
    params = params_from_dict(current)
    params.prior = prior
    full = 0.0
    print("\nfinal per-file spellF1 (taus 12 & 25):")
    per_file = {}
    for tag in holdouts.tags:
        states = decode_match(holdouts.P[tag], holdouts.ctx[tag], holdouts.blocks[tag], params)
        codes = np.where(states < 0, 3, states).astype(np.int8)
        result = pm.evaluate(holdouts.gt_codes[tag], codes, taus=(12, 25), bootstrap_n=0)
        f1 = result["spell_f1_mean"]
        per_file[f"{tag[0]}{'_d' if tag[1] else ''}"] = float(f1)
        full += f1
        print(f"  {tag[0]:<14} dirty={int(tag[1])}: spellF1={f1:.3f}  "
              f"macro={result['macro_f1']:.3f}")
    best_spell = full / len(holdouts.tags)
    print(f"\nFINAL mean spellF1 = {best_spell:.4f} over {len(holdouts.tags)} files")

    update = {
        "spell_f1_mean": best_spell,
        "params": current,
        "prior": [float(x) for x in prior],
        "provenance": "coordinate descent on LOO external posteriors; no competition asset",
        "per_file": per_file,
    }
    with open(BEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(update, handle, indent=2)
    with open(SEARCH_LOG, "w", encoding="utf-8") as handle:
        json.dump({"best_spell_f1": best_spell, "best_params": current,
                   "elapsed_s": time.time() - t0, "history": history}, handle, indent=2)
    print(f"saved -> {BEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
