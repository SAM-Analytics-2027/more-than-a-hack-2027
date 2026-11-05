"""CLI entry-point to run the Ball Possession Inference Model and write a submission CSV.

Usage:
    python possession_model/run.py
    python possession_model/run.py --out evaluation/results/possession_v2.csv --no-broadcast
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME  # noqa: E402
from possession_model.pipeline import run_possession_inference  # noqa: E402


def write_submission(path: str, labels: np.ndarray) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ball Possession Inference Model")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "evaluation", "results", "possession_v2.csv"),
        help="Output submission CSV path (default: evaluation/results/possession_v2.csv)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional override for MTAH_DATA_DIR",
    )
    parser.add_argument(
        "--no-broadcast",
        action="store_true",
        help="Disable automatic Broadcast ATD camera fusion",
    )
    args = parser.parse_args()

    labels, _ = run_possession_inference(
        game=GAME,
        data_dir=args.data_dir,
        use_broadcast=not args.no_broadcast,
        verbose=True,
    )
    out_path = write_submission(args.out, labels)
    print(f"\nWrote submission ({len(labels)} rows) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
