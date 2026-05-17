@echo off
REM Windows convenience launcher for the React preview frontend.
REM This script is optional and not the primary documented setup path.

SET SCRIPT_DIR=%~dp0
PUSHD "%SCRIPT_DIR%frontend"

where node >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
  echo Node.js was not found. Install Node.js and try again: https://nodejs.org/
  POPD
  exit /b 1
)

echo Installing frontend dependencies...
npm install

echo Starting Vite dev server on http://127.0.0.1:5176 ...
echo It will call the FastAPI backend on http://127.0.0.1:8010.
npm run dev -- --host 127.0.0.1 --port 5176

POPD

echo.
echo Frontend stopped. If this was unexpected, copy the error above.
pause
