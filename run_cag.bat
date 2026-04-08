@echo off
REM Windows convenience launcher for the Streamlit debug UI.
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

echo Starting Streamlit on the local development UI...
"%PYTHON%" -m streamlit run "%SCRIPT_DIR%src\cag\ui\app.py"

pause
