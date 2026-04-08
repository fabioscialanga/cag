@echo off
REM Windows convenience launcher for the CAG preview.
REM This script is optional and not the primary documented setup path.

setlocal enableextensions enabledelayedexpansion

set REPO_DIR=%~dp0
echo Starting CAG preview services from %REPO_DIR%

start "CAG API" cmd /k "cd /d "%REPO_DIR%" && .venv\Scripts\activate.bat && python -m uvicorn cag.api.upload:app --reload --host 0.0.0.0 --port 8000"
start "CAG Frontend" cmd /k "cd /d "%REPO_DIR%frontend" && npm install && npm run dev -- --host 0.0.0.0"

echo Started two windows: API and React frontend.
echo Use these launchers only as local Windows conveniences.

endlocal
pause
