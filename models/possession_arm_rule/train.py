"""possession_arm_rule (family 9): the provided rule pipeline as a fit-free
posterior family under the IDENTICAL shared protocol.

The rule posterior is DETERMINISTIC and per-file: the LOO folds exist only to
score it with the shared decoder/metric and produce the protocol-tagged LOO
posteriors (`looP_v2_possession_arm_rule_*`, equal to the static ones). The
fit_predict adapter ignores training/stop files by construction -- exactly the
adaptation the rule model needs, no learnt parameter anywhere.

    python models/possession_arm_rule/train.py --mode final     # no-op (kept for CLI parity)
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_rule/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.cache import loo_posterior_path  # noqa: E402

MODEL_TAG = "possession_arm_rule"

ARTIFACT_DIR = os.path.join(HERE, "artifacts")


def static_posterior(match_id: str, dirty: bool) -> np.ndarray:
    path = loo_posterior_path(match_id, dirty, model_tag=MODEL_TAG)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing rule posterior: {path}\n"
            "run: python models/possession_arm_rule/build_posteriors.py")
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["P"], dtype=np.float32)


def prediction_only_adapter() -> Callable:
    """run_nested_loo adapter for a fit-free family.

    fit_predict(train_files, stop_files) -> (predict_fn, info) where predict_fn
    reads the family's static per-file posterior (already out-of-sample by
    construction: no fold-time fitting happens anywhere).
    """

    def fit_predict(train_files, stop_files):
        info = {"best_iteration": None, "note": "fit-free rule family"}

        def predict_fn(file_dict) -> np.ndarray:
            return static_posterior(file_dict["match_id"], bool(file_dict["dirty"]))

        return predict_fn, info

    return fit_predict


def predict_final(X=None, side=None, names=None) -> Tuple[np.ndarray, list]:
    """Target-time posterior for ensemble predict scripts.

    Two supported forms (mirroring the other families' predict_final):
      - predict_final(f_dict) with the loaded target / cache file dict: reads the
        static per-file posterior;
      - predict_final(X, side) with the target's fresh feature dicts: computes
        the rule costs directly (identical function, no fit).
    Returns (P [n, 3], names) -- names echoes the feature list for the drift check.
    """
    if side is not None:
        import models.possession_arm_rule.rule_costs as rc
        P = rc.rule_posterior(side)
        return P, list(names)
    raise TypeError("rule predict_final needs (X, side) with the target's side dicts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("final",), default="final")
    args = parser.parse_args()
    # There is no final artifact: the rule is code, not a fit. Landing the artifact
    # folder keeps the family layout uniform for the search/predict scripts.
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    meta_path = os.path.join(ARTIFACT_DIR, "rule_final_meta.json")
    meta = {"tag": MODEL_TAG,
            "form": "P = exp(-frame_costs), provided pipeline cost recipe (tagless tactical)",
            "fit": "none (deterministic rule; the LOO posteriors equal the static ones)",
            "note": "external matches and the competition target share the same tagless rule"}
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(f"rule family: no fit -- {meta['form']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
