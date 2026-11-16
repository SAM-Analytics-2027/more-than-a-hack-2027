@echo off
setlocal
cd /d "%~dp0.."

if "%MTAH_DATA_DIR%"=="" (
    if exist "%~dp0..\..\data" (
        set "MTAH_DATA_DIR=%~dp0..\..\data"
    )
)

echo ========================================================
echo   More than a Hack 2027 - Visual Comparison Dashboard
echo   (100%% read-only - does not change data or algorithms)
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "dashboard\app.py" %*
) else (
    python "dashboard\app.py" %*
)
pause
