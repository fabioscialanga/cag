#!/usr/bin/env bash
# Convenience launcher for the FastAPI preview backend.
# This script is optional and not the primary documented setup path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
  echo "Virtualenv Python not found at: $PYTHON"
  echo "Create it with: python -m venv .venv"
  echo "Install dependencies with: pip install -r requirements.txt"
  exit 1
fi

echo "Starting FastAPI on http://localhost:8000 ..."
"$PYTHON" -m uvicorn cag.api.upload:app --reload --port 8000
