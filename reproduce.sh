#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Checking Python environment and BAR-ATH input file hashes ==="
python modelli_mv/riproduci.py controlla --solo-barca

echo "=== [2/4] Generating predictions from frozen weights (mv_hgb_rete3) ==="
python modelli_mv/riproduci.py prevedi

echo "=== [3/4] Copying output to SAM-Analytics-2027.csv ==="
cp evaluation/results/mv_hgb_rete3.csv SAM-Analytics-2027.csv

echo "=== [4/4] Validating final submission file ==="
python evaluation/validate_submission.py SAM-Analytics-2027.csv

echo "Done. SAM-Analytics-2027.csv is ready for submission."