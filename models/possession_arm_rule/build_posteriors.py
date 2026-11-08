"""Build the static rule-pipeline posteriors for all 20 external files.

For every (match_id, dirty): load the external match, run the SAME feature build
the cache used (build_feature_matrix), then compute the exact provided-pipeline
posterior P = exp(-frame_costs) via possession_model.pipeline and save it to the
family's protocol-tagged cache path (looP_v2_possession_arm_rule_*). The LOO
engine re-writes the same arrays later (the rule is fit-free, so per-fold P is
identical to the static one) -- this build is just the one-time expensive pass.

    python models/possession_arm_rule/build_posteriors.py          # all 20 files
    python models/possession_arm_rule/build_posteriors.py --ids J03WMX
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_rule/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.cache import cache_path, loo_posterior_path  # noqa: E402
from models.modules.data_external import EXTERNAL_IDS, load_external_match  # noqa: E402
from models.modules.features_base import build_feature_matrix  # noqa: E402
from models.modules.half_blocks import detect_half_split, half_blocks_from_tracking  # noqa: E402
from models.possession_arm_rule.rule_costs import MODEL_TAG, rule_posterior  # noqa: E402


def build_one(match_id: str, dirty: bool, check_cache: bool = True) -> float:
    match = load_external_match(match_id, dirty)
    # EXACT cache.py block semantics: tracking-based half split (the labels-based
    # detector is too coarse and would skew the per-half attack features).
    split = detect_half_split(match["player_xy"], match["slot_teams"], match["pitch"])
    blocks = half_blocks_from_tracking(len(match["labels"]), split)
    X, _names, side = build_feature_matrix(match, blocks)
    if check_cache:
        with np.load(cache_path(match_id, dirty), allow_pickle=False) as z:
            cached = np.asarray(z["X"], dtype=np.float32)
        same = (np.asarray(X, dtype=np.float32).shape == cached.shape
                and np.allclose(np.asarray(X, dtype=np.float32), cached,
                                rtol=0, atol=2e-5, equal_nan=True))
        if not same:
            nan_diff = int(np.isnan(X).sum())
            raise RuntimeError(f"{match_id}{'_d' if dirty else ''}: "
                               f"recomputed X drifts from the cached feature matrix "
                               f"(shape {np.shape(X)} vs {cached.shape}, nan={nan_diff})")
    P = rule_posterior(side)
    labels = np.asarray(match["labels"], dtype=np.int8)
    out = loo_posterior_path(match_id, dirty, model_tag=MODEL_TAG)
    np.savez_compressed(out, P=P, labels=labels,
                        blocks=np.asarray(blocks, dtype=np.int64))
    share = float(P[:6000][:, 2].mean())
    return share


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=None,
                        help="subset of external match ids (default all)")
    parser.add_argument("--no-cache-check", action="store_true")
    args = parser.parse_args()
    ids = args.ids if args.ids is not None else EXTERNAL_IDS

    t0 = time.time()
    for mid in ids:
        for dirty in (False, True):
            share = build_one(mid, dirty, check_cache=not args.no_cache_check)
            suffix = "_d" if dirty else ""
            print(f"  built {mid}{suffix:<3}  rule D-share(6000f)={share:.3f}  "
                  f"({time.time() - t0:.0f}s elapsed)")
    print(f"done ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
