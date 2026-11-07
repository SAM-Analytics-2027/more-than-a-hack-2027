"""Smoke test of the Stage-1 feature builder on ONE external match (not cached).

    python possession_arm/smoke_features.py <match_id> [--dirty]

    New location: python models/possession_arm_v1/smoke_features.py <match_id> [--dirty]
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_v1/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from models.modules.data_external import load_external_match  # noqa: E402
from models.modules.features_base import build_feature_matrix  # noqa: E402
from models.modules.half_blocks import detect_half_split, half_blocks_from_tracking  # noqa: E402


def main() -> int:
    match_id = sys.argv[1] if len(sys.argv) > 1 else "J03WN1"
    dirty = "--dirty" in sys.argv

    t0 = time.time()
    match = load_external_match(match_id, dirty=dirty)
    split = detect_half_split(match["player_xy"], match["slot_teams"], match["pitch"])
    blocks = half_blocks_from_tracking(len(match["labels"]), split)
    print(f"load: {time.time() - t0:.1f}s  frames={len(match['labels'])}  blocks={blocks}")

    t1 = time.time()
    X, names, side = build_feature_matrix(match, blocks)
    dt = time.time() - t1
    n_nan = float(np.isnan(X).mean())
    print(f"features: X={X.shape} dtype={X.dtype}  in {dt:.1f}s  NaN-rate={n_nan:.3f}")
    for k in range(0, len(names), 6):
        print("  ", " ".join(f"{n:>26}" for n in names[k : k + 6]))
    col = dict(zip(names, X.T))
    for probe in ("ball_observed", "contact_min_dist_h", "contact_impact_h", "tm_pitch_control_h"):
        v = col[probe]
        print(f"  {probe:>26}: mean={np.nanmean(v):.3f} frac-nonnan={np.mean(np.isfinite(v)):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
