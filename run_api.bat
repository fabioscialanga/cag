@echo off
REM Windows convenience launcher for the FastAPI preview backend.
REM This script is optional and not the primary documented setup path.

SET SCRIPT_DIR=%~dp0
SET PYTHON=%SCRIPT_DIR%\.venv\Scripts\python.exe
SET PYTHONPATH=%SCRIPT_DIR%src

IF NOT EXIST "%PYTHON%" (
  echo Virtualenv Python not found at: %PYTHON%
  echo Create it with: python -m venv .venv
  echo Install dependencies with: pip install -r requirements.txt
  pause
  exit /b 1
)

echo Starting FastAPI on http://127.0.0.1:8010 ...
echo Keep this window open while using the app.
"%PYTHON%" -m uvicorn cag.api.upload:app --host 127.0.0.1 --port 8010

echo.
echo FastAPI stopped. If this was unexpected, copy the error above.
pause
