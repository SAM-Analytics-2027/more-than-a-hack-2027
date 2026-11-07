"""Repository scoring helper shared by every model.

`score_file` is the exact scoring block of possession_arm_v1's eval_external.py,
extracted so new families evaluate with the SAME metric wiring: spellF1/acc/macroF1
(u, taus 12 and 25, bootstrap off for speed) plus the transition-F1 at tau=25.
Nothing here is model-specific; comparison between families stays apples-to-apples.
"""

from typing import Dict

import numpy as np

from evaluation import possession_metrics as pm  # noqa: E402


def to_eval_codes(codes: np.ndarray) -> np.ndarray:
    """Internal codes {-1, 0, 1, 2} -> metric codes {0=H, 1=A, 2=D, 3=X}."""
    return np.where(codes < 0, 3, codes).astype(np.int8)


def score_file(gt_labels: np.ndarray, pred_states: np.ndarray) -> Dict:
    """Score one decoded match against its clean GT with the repository metric."""
    gt_codes = to_eval_codes(gt_labels.astype(np.int8))
    pred_codes = to_eval_codes(pred_states.astype(np.int8))
    result = pm.evaluate(gt_codes, pred_codes, taus=(12, 25), bootstrap_n=0)
    t25 = result["transitions"][25]
    return {
        "spell_f1": float(result["spell_f1_mean"]),
        "macro_f1": float(result["macro_f1"]),
        "acc": float(result["accuracy"]),
        "trans_f1_25": float(t25["f1"]),
        "gt_spells": int(result["n_gt_spells"]),
        "pred_spells": int(result["n_pred_spells"]),
        "f1_h": float(result["per_class"]["H"]["f1"]),
        "f1_a": float(result["per_class"]["A"]["f1"]),
        "f1_d": float(result["per_class"]["D"]["f1"]),
    }
