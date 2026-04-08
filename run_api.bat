@echo off
REM Windows convenience launcher for the FastAPI preview backend.
REM This script is optional and not the primary documented setup path.

SET SCRIPT_DIR=%~dp0
SET PYTHON=%SCRIPT_DIR%\.venv\Scripts\python.exe

IF NOT EXIST "%PYTHON%" (
  echo Virtualenv Python not found at: %PYTHON%
  echo Create it with: python -m venv .venv
  echo Install dependencies with: pip install -r requirements.txt
  pause
  exit /b 1
)

echo Starting FastAPI on http://localhost:8000 ...
"%PYTHON%" -m uvicorn cag.api.upload:app --reload --port 8000
