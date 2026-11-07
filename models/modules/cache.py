"""Feature + context cache for the possession_arm_v1 training matches.

build_cache computes, for every (match_id, dirty) file, the Stage-1 feature matrix
X, the ground-truth labels, and the six decoder-context masks, and stores them in
one compressed npz per file under data/_arm_cache/. Everything downstream (XGBoost
fitting, LOO evaluation, decoder parameter search, the final competition decode)
reads from this cache only, so feature code runs exactly once per change.

    python possession_arm/cache.py                 # build all missing deltas
    python possession_arm/cache.py --force         # rebuild everything
    python possession_arm/cache.py --ids J03WN1    # a subset
"""

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/modules/ -> models/ -> repo root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.modules.context import CONTEXT_KEYS, build_context  # noqa: E402
from models.modules.data_external import EXTERNAL_IDS, load_external_match  # noqa: E402
from models.modules.features_base import FEATURE_SET_TAG, build_feature_matrix  # noqa: E402
from models.modules.half_blocks import detect_half_split, half_blocks_from_tracking  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "_arm_cache")


def cache_path(match_id: str, dirty: bool) -> str:
    branch = "dfl" if match_id.startswith("J03") else "metrica"
    subset = "sporcate" if dirty else "pulite"
    return os.path.join(CACHE_DIR, f"{FEATURE_SET_TAG}_{branch}_{subset}_{match_id}.npz")


def parse_cache_name(name: str) -> tuple:
    """'<tag>_<branch>_<subset>_<match_id>.npz' -> (match_id, dirty); assert current tag."""
    stem = name[:-len(".npz")] if name.endswith(".npz") else name
    if not stem.startswith(FEATURE_SET_TAG + "_"):
        raise ValueError(f"not a {FEATURE_SET_TAG} cache file: {name}")
    rest = stem[len(FEATURE_SET_TAG) + 1:]
    if "_sporcate_" in rest:
        dirty, match_id = True, rest.split("_sporcate_", 1)[1]
    else:
        dirty, match_id = False, rest.split("_pulite_", 1)[1]
    return match_id, dirty


def cache_names() -> List[str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    prefix = FEATURE_SET_TAG + "_"
    return [f for f in sorted(os.listdir(CACHE_DIR))
            if f.endswith(".npz") and f.startswith(prefix)]


def loo_posterior_path(match_id: str, dirty: bool, model_tag: str | None = None) -> str:
    """Path of the leave-one-match-out posterior npz for one (match_id, dirty) file.

    The legacy naming (model_tag=None) is 'looP_<tag>_<branch>_<subset>_<match_id>.npz';
    every other model appends its own tag so families never overwrite each other's
    posteriors. Passing model_tag=None reproduces the exact legacy name, so the v1
    files already in the cache keep working untouched.
    """
    base = os.path.basename(cache_path(match_id, dirty))
    stem = base[len(FEATURE_SET_TAG) + 1:]
    prefix = "looP_" + FEATURE_SET_TAG + "_"
    if model_tag:
        prefix += model_tag + "_"
    return os.path.join(CACHE_DIR, prefix + stem)


def loo_posterior_names(model_tag: str | None = None) -> List[str]:
    """Sorted LOO posterior npz names (model_tag=None -> the legacy untagged set)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    prefix = "looP_" + FEATURE_SET_TAG + "_"
    if model_tag:
        prefix += model_tag + "_"
    return [f for f in sorted(os.listdir(CACHE_DIR))
            if f.startswith(prefix) and f.endswith(".npz")]


def parse_posterior_name(name: str, model_tag: str | None = None) -> tuple:
    """'looP_<tag>_<branch>_<subset>_<match_id>.npz' -> (match_id, dirty).

    With model_tag set the name must carry that tag ('looP_<tag>_<model_tag>_...')
    and it is stripped before parsing; ValueError otherwise.
    """
    stem = name[:-len(".npz")] if name.endswith(".npz") else name
    prefix = "looP_" + FEATURE_SET_TAG + "_"
    if not stem.startswith(prefix):
        raise ValueError(f"not a LOO posterior file: {name}")
    rest = stem[len(prefix):]
    if model_tag:
        tag_prefix = model_tag + "_"
        if not rest.startswith(tag_prefix):
            raise ValueError(f"not a {model_tag} posterior file: {name}")
        rest = rest[len(tag_prefix):]
    if "_sporcate_" in rest:
        dirty, match_id = True, rest.split("_sporcate_", 1)[1]
    else:
        dirty, match_id = False, rest.split("_pulite_", 1)[1]
    return match_id, dirty


def is_cached(match_id: str, dirty: bool) -> bool:
    return os.path.isfile(cache_path(match_id, dirty))


def build_one(match_id: str, dirty: bool) -> str:
    """Compute and save one cache file; returns the path."""
    t0 = time.time()
    match = load_external_match(match_id, dirty=dirty)
    split = detect_half_split(match["player_xy"], match["slot_teams"], match["pitch"])
    blocks = half_blocks_from_tracking(len(match["labels"]), split)
    X, names, side = build_feature_matrix(match, blocks)
    ctx = build_context(match, blocks, side)

    out = {
        "X": X.astype(np.float32),
        "labels": match["labels"].astype(np.int8),
        "names": json.dumps(names),
        "blocks": np.asarray(blocks, dtype=np.int64),
        "pitch": np.asarray(match["pitch"], dtype=np.float64),
        "fps": np.float64(match["fps"]),
    }
    bad = [k for k in CONTEXT_KEYS if k not in ctx]
    if bad:
        raise RuntimeError(f"context masks missing {bad}")
    for key in CONTEXT_KEYS:
        out[f"ctx_{key}"] = np.asarray(ctx[key], dtype=np.uint8)

    path = cache_path(match_id, dirty)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **out)
    nan_rate = float(np.isnan(X).mean())
    print(
        f"{match_id:<14} dirty={int(dirty)}  X={X.shape}  nan={nan_rate:.3f}  "
        f"blocks={[(int(a), int(b)) for _, a, b in blocks]}  {time.time() - t0:.1f}s"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=EXTERNAL_IDS)
    parser.add_argument("--force", action="store_true", help="rebuild even cached files")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--dirty-only", action="store_true")
    args = parser.parse_args()

    jobs: List = []
    for match_id in args.ids:
        for dirty in (False, True):
            if args.clean_only and dirty:
                continue
            if args.dirty_only and not dirty:
                continue
            if not args.force and is_cached(match_id, dirty):
                print(f"{match_id:<14} dirty={int(dirty)}  cached (skip)")
                continue
            jobs.append((match_id, dirty))

    failures = 0
    for match_id, dirty in jobs:
        try:
            build_one(match_id, dirty)
        except Exception as error:  # noqa: BLE001 -- report every failing id
            failures += 1
            print(f"{match_id:<14} dirty={int(dirty)}  FAILED: {error!r}")
    print(f"done: {len(jobs) - failures}/{len(jobs)} built, cache={CACHE_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
