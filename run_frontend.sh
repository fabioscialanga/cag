#!/usr/bin/env bash
# Convenience launcher for the React preview frontend.
# This script is optional and not the primary documented setup path.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v node &>/dev/null; then
  echo "Node.js was not found. Install Node.js and try again: https://nodejs.org/"
  exit 1
fi

cd "$SCRIPT_DIR/frontend"

echo "Installing frontend dependencies..."
npm install

echo "Starting Vite dev server on http://localhost:5173 ..."
npm run dev
