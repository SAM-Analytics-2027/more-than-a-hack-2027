"""Generic external-only decoder-parameter search (coordinate descent).

Faithful port of possession_arm_v1/search_decoder.py to the shared protocol so
every family tunes its Stage-2 decoder the same STRICTLY-external way:

  - inputs: the family's LOO posteriors saved by models/modules/training_protocol
    (files 'looP_v2_<model_tag>_*.npz' under data/_arm_cache/) plus the shared v2
    context masks read from the feature caches;
  - objective: mean spellF1 over the 20 held-out files at tau=25 (v1 consistency);
  - search: coordinate descent over SEARCH_ORDER x PARAM_GRID from
    models/modules/decoder_support, adopting a value only when spellF1 improves
    by more than 1e-4;
  - outputs: '<artifact_dir>/decoder_params.json' (provenance-tagged) and
    'decoder_search_log.json'.

The competition match and its ground truth are NEVER involved. The class prior
comes from the external GT files themselves.
"""

import json
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

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

# Coordinate order: shortest path to the dense spellF1 levers first (same as v1).
SEARCH_ORDER = (
    "min_pos_frames", "min_stop_frames",
    "turn_quiet", "stop_quiet", "restart_quiet", "w_prior",
    "turn_loud", "turn_control", "turn_occlusion",
    "stop_loud", "restart_loud",
)


class Holdouts:
    """All LOO posteriors + metadata held in RAM, indexed by file tag."""

    def __init__(self, model_tag: str | None = None) -> None:
        self.tags: List[Tuple[str, bool]] = []
        self.P: Dict[Tuple[str, bool], np.ndarray] = {}
        self.labels: Dict[Tuple[str, bool], np.ndarray] = {}
        self.blocks: Dict[Tuple[str, bool], List[Tuple[int, int, int]]] = {}
        self.ctx: Dict[Tuple[str, bool], Dict[str, np.ndarray]] = {}
        self.gt_codes: Dict[Tuple[str, bool], np.ndarray] = {}
        for name in loo_posterior_names(model_tag):
            match_id, dirty = parse_posterior_name(name, model_tag)
            tags = (match_id, dirty)
            with np.load(os.path.join(CACHE_DIR, name), allow_pickle=False) as z:
                self.P[tags] = z["P"]
                labels = z["labels"]
                self.labels[tags] = labels
                self.gt_codes[tags] = np.where(labels < 0, 3, labels).astype(np.int8)
                self.blocks[tags] = [tuple(map(int, row)) for row in z["blocks"]]
        # contexts come from the feature caches (same tags)
        for tag in list(self.P):
            with np.load(cache_path(tag[0], tag[1]), allow_pickle=False) as z:
                self.ctx[tag] = {k[4:]: z[k].astype(bool) for k in z.files
                                 if k.startswith("ctx_")}

    def score(self, params: DecoderParams) -> Tuple[float, float]:
        spell_scores, macro_scores = [], []
        for tag in self.tags:
            states = decode_match(self.P[tag], self.ctx[tag], self.blocks[tag], params)
            codes = np.where(states < 0, 3, states).astype(np.int8)
            result = pm.evaluate(self.gt_codes[tag], codes, taus=(25,), bootstrap_n=0)
            spell_scores.append(result["spell_f1_mean"])
            macro_scores.append(result["macro_f1"])
        return float(np.mean(spell_scores)), float(np.mean(macro_scores))


def external_class_prior(holdouts: Holdouts) -> np.ndarray:
    """Aggregate H/A/D prior over all EXTERNAL GT files (never the competition GT)."""
    totals = np.zeros(3, dtype=np.float64)
    for tag in holdouts.tags:
        flat = holdouts.gt_codes[tag]
        flat = flat[flat < 3]
        for c in (0, 1, 2):
            totals[c] += float((flat == c).sum())
    return totals / totals.sum()


def coordinate_search(holdouts: Holdouts, artifact_dir: str, rounds: int = 4,
                      log: Callable[[str], None] = print,
                      start_params: Dict | None = None) -> Dict:
    """Coordinate descent identical to possession_arm_v1/search_decoder.py.

    Returns {'best_params': dict, 'best_spell_f1': float, 'prior': np.ndarray,
    'history': [...]} and writes decoder_params.json + decoder_search_log.json
    into artifact_dir. start_params overrides the DecoderParams() start point
    (used to warm-start from another family's addressable optimum).
    """
    os.makedirs(artifact_dir, exist_ok=True)
    log_path = os.path.join(artifact_dir, "decoder_search_log.json")
    best_path = os.path.join(artifact_dir, "decoder_params.json")
    t0 = time.time()

    prior = external_class_prior(holdouts)
    log(f"external class prior H/A/D = {prior.round(4)}")

    current = dict(params_to_dict(DEFAULT_PARAMS)) if start_params is None else dict(start_params)

    def evaluate(d: Dict) -> Tuple[float, float]:
        params = params_from_dict(d)
        params.prior = prior
        spell, macro = holdouts.score(params)
        return spell, macro

    best_spell, best_macro = evaluate(current)
    log(f"start (default) spellF1={best_spell:.4f}  macroF1={best_macro:.4f}")

    history = [{"step": "init", "spell_f1": best_spell, "params": dict(current)}]
    for round_num in range(rounds):
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
                log(f"  r{round_num} {param:>16}={value:<6} spell={spell:.4f} "
                    f"macro={macro:.4f}{flag}")
                history.append({"step": f"r{round_num}.{param}", "value": value,
                                "spell_f1": spell, "macro_f1": macro,
                                "adopted": bool(adopted)})
                if adopted:
                    current = candidate
                    best_spell, best_macro = spell, macro
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as handle:
                json.dump({"best_spell_f1": best_spell, "best_params": current,
                           "elapsed_s": time.time() - t0, "history": history},
                          handle, indent=2)
        if best_spell <= round_best - 1e-9 and round_num > 0:
            break
        log(f"[round {round_num}] best spellF1={best_spell:.4f} "
            f"(elapsed {time.time() - t0:.0f}s)")

    # final report with the full tau set (verbatim v1 semantics)
    params = params_from_dict(current)
    params.prior = prior
    full = 0.0
    log("\nfinal per-file spellF1 (taus 12 & 25):")
    per_file = {}
    for tag in holdouts.tags:
        states = decode_match(holdouts.P[tag], holdouts.ctx[tag], holdouts.blocks[tag], params)
        codes = np.where(states < 0, 3, states).astype(np.int8)
        result = pm.evaluate(holdouts.gt_codes[tag], codes, taus=(12, 25), bootstrap_n=0)
        f1 = result["spell_f1_mean"]
        per_file[f"{tag[0]}{'_d' if tag[1] else ''}"] = float(f1)
        full += f1
        log(f"  {tag[0]:<14} dirty={int(tag[1])}: spellF1={f1:.3f}  "
            f"macro={result['macro_f1']:.3f}")
    best_spell = full / len(holdouts.tags)
    log(f"\nFINAL mean spellF1 = {best_spell:.4f} over {len(holdouts.tags)} files")

    update = {
        "spell_f1_mean": best_spell,
        "params": current,
        "prior": [float(x) for x in prior],
        "provenance": "coordinate descent on external LOO posteriors; no competition asset",
        "per_file": per_file,
    }
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump(update, handle, indent=2)
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump({"best_spell_f1": best_spell, "best_params": current,
                   "elapsed_s": time.time() - t0, "history": history}, handle, indent=2)
    log(f"saved -> {best_path}")
    return update
