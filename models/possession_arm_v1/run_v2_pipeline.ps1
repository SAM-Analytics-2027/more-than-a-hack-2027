# Full v2 feature-set chain: rebuild cache -> nested LOO -> final fit -> decoder search.
# Steps run strictly in order; every step's stdout prefixes with [step] for readability.
$ErrorActionPreference = 'Stop'
.\.venv\Scripts\python.exe models\modules\cache.py --force
if ($LASTEXITCODE -ne 0) { Write-Host '[cache] FAILED'; exit 1 }
.\.venv\Scripts\python.exe models\possession_arm_v1\eval_external.py
if ($LASTEXITCODE -ne 0) { Write-Host '[loo] FAILED'; exit 1 }
.\.venv\Scripts\python.exe models\possession_arm_v1\train.py --mode final
if ($LASTEXITCODE -ne 0) { Write-Host '[final] FAILED'; exit 1 }
.\.venv\Scripts\python.exe models\possession_arm_v1\search_decoder.py --rounds 4
if ($LASTEXITCODE -ne 0) { Write-Host '[search] FAILED'; exit 1 }
Write-Host 'V2 PIPELINE COMPLETE'
