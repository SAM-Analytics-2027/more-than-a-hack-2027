"""Benchmark the provided duration-constrained Viterbi on realistic CPU arrays.

   python possession_arm/bench_decoder.py [frames] [min_pos] [min_stop]

   New location: python models/possession_arm_v1/bench_decoder.py [frames] [min_pos] [min_stop]
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_v1/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession_model.sequence_decoder import decode_possession_sequence  # noqa: E402


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 71_000
    min_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    min_stop = int(sys.argv[3]) if len(sys.argv) > 3 else 85
    rng = np.random.default_rng(7)
    fc = -np.log(rng.dirichlet([1.0, 1.0, 0.35], size=n).astype(np.float64) + 1e-9)
    tc = np.full(n, 23.0)
    sc = np.where(rng.random(n) < 0.05, 5.5, 23.0)
    rc = np.where(rng.random(n) < 0.02, 2.0, 18.5)
    t0 = time.time()
    states = decode_possession_sequence(fc, tc, sc, rc, min_pos, min_stop)
    dt = time.time() - t0
    k = 2 * min_pos + min_stop
    print(f"frames={n} K={k} -> {dt:.2f}s ({n / dt:.0f} frames/s)")
    print(f"seg changes: {int((states[1:] != states[:-1]).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
