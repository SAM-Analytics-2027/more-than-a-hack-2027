@echo off
setlocal enabledelayedexpansion

echo === [1/4] Checking Python environment and BAR-ATH input file hashes ===
python modelli_mv\riproduci.py controlla --solo-barca
if errorlevel 1 exit /b 1

echo === [2/4] Generating predictions from frozen weights (mv_hgb_rete3) ===
python modelli_mv\riproduci.py prevedi
if errorlevel 1 exit /b 1

echo === [3/4] Copying output to SAM-Analytics-2027.csv ===
copy /Y evaluation\results\mv_hgb_rete3.csv SAM-Analytics-2027.csv >nul
if errorlevel 1 exit /b 1

echo === [4/4] Validating final submission file ===
python evaluation\validate_submission.py SAM-Analytics-2027.csv
if errorlevel 1 exit /b 1

echo Done. SAM-Analytics-2027.csv is ready for submission.