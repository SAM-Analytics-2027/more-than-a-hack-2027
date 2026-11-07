"""Feature-coverage diagnostic: target match vs pooled external (Dirtied) training data.

For every Stage-1 feature, how far does the competition match's distribution sit
outside the external training distribution? No ground truth involved: this is a
tracking-only distribution-coverage report used to fix train/serve feature skew.

    python possession_arm/diagnose_shift.py

    New location: python models/possession_arm_v1/diagnose_shift.py
"""

import os
import sys
from typing import Dict, List

import numpy as np
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_v1/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.cache import CACHE_DIR, cache_names  # noqa: E402
from models.modules.features_base import build_feature_matrix  # noqa: E402
from models.modules.target_match import load_target_match  # noqa: E402


def pooled_external(names: List[str], dirty_only: bool = True) -> Dict[str, np.ndarray]:
    chunks: Dict[str, List[np.ndarray]] = {n: [] for n in names}
    for name in cache_names():
        if not name.endswith(".npz") or name.startswith("looP_"):
            continue
        if dirty_only and "_pulite_" in name:
            continue
        with np.load(os.path.join(CACHE_DIR, name), allow_pickle=False) as z:
            X = z["X"]
            for j, feat in enumerate(json.loads(str(z["names"]))):
                chunks[feat].append(X[:, j])
    return {k: np.concatenate(v) for k, v in chunks.items()}


def coverage_report(ext: np.ndarray, tgt: np.ndarray) -> Dict:
    ext_f = ext[np.isfinite(ext)]
    tgt_f = tgt[np.isfinite(tgt)]
    if len(ext_f) < 100 or len(tgt_f) < 100:
        return {"sev": 0.0, "cov": 1.0, "note": "insufficient finite values"}
    q1, q3 = np.percentile(ext_f, [25, 75])
    iqr = max(1e-9, q3 - q1)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    cov = float(np.mean((tgt_f >= lo) & (tgt_f <= hi)))
    z = abs(float(np.median(tgt_f)) - float(np.median(ext_f))) / (0.7413 * iqr)
    frac_ext_below = float(np.mean(ext_f < np.percentile(tgt_f, 5)))
    return {"cov": cov, "shift_z": z, "ext_lo": float(np.percentile(ext_f, 5)),
            "ext_hi": float(np.percentile(ext_f, 95)),
            "tgt_lo": float(np.percentile(tgt_f, 5)),
            "tgt_hi": float(np.percentile(tgt_f, 95)),
            "ext_zero_frac": 1.0 - len(ext_f) / len(ext), "sev": float(max(0.0, 1.0 - cov) + max(0.0, z - 1))}


def main() -> int:
    ext = pooled_external([None], dirty_only=True) if False else None
    # derive names first
    with np.load(os.path.join(CACHE_DIR, cache_names()[0]), allow_pickle=False) as z:
        names = list(json.loads(str(z["names"])))
    ext = pooled_external(names, dirty_only=True)

    match = load_target_match()
    blocks = match["blocks"]
    X, tnames, _ = build_feature_matrix(match, blocks)
    assert tnames == names, "feature name order mismatch vs cache"
    print(f"target X={X.shape}; external dirty files pooled; report sorted by worst coverage\n")

    rows = []
    for j, feat in enumerate(names):
        rep = coverage_report(ext[feat], X[:, j])
        rep["feature"] = feat
        rows.append(rep)
    rows.sort(key=lambda r: (r["cov"],))

    print(f"{'feature':>28} {'cov%':>6} {'shiftZ':>7}   ext95 range        tgt95 range")
    for r in rows[: 22]:
        print(f"{r['feature']:>28} {100 * r['cov']:>6.1f} {r['shift_z']:>7.2f}   "
              f"[{r['ext_lo']:8.2f},{r['ext_hi']:8.2f}] [{r['tgt_lo']:8.2f},{r['tgt_hi']:8.2f}]")
    full_cov = sum(1 for r in rows if r["cov"] >= 0.9)
    print(f"\nfeatures with coverage >= 90%: {full_cov}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
