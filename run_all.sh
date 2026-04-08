#!/usr/bin/env bash
# Convenience launcher for the CAG preview.
# This script is optional and not the primary documented setup path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting CAG preview services from $SCRIPT_DIR"

# Start API in background
"$SCRIPT_DIR/run_api.sh" &
API_PID=$!

# Start frontend in background
"$SCRIPT_DIR/run_frontend.sh" &
FRONTEND_PID=$!

echo "Started API (PID $API_PID) and React frontend (PID $FRONTEND_PID)."
echo "Use these launchers only as local conveniences."

wait
