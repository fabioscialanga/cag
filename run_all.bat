@echo off
REM Windows convenience launcher for the CAG preview.
REM This script is optional and not the primary documented setup path.

setlocal enableextensions enabledelayedexpansion

set REPO_DIR=%~dp0
echo Starting CAG preview services from %REPO_DIR%

start "CAG API" "%REPO_DIR%run_api.bat"
start "CAG Frontend" "%REPO_DIR%run_frontend.bat"

echo Started two windows: API and React frontend.
echo Backend:  http://127.0.0.1:8010
echo Frontend: http://127.0.0.1:5176
echo Use these launchers only as local Windows conveniences.

endlocal
pause
