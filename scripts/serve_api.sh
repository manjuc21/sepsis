#!/usr/bin/env bash
# Runs the FastAPI app directly on the host (no Docker), against whatever
# trained model artifact exists in models/. If Postgres isn't reachable at
# DATABASE_URL, prediction logging is skipped automatically (src/api/main.py
# degrades gracefully rather than crashing) — a local Postgres is optional
# for this script, only needed if you want the audit log to actually persist.
#
# Usage: scripts/serve_api.sh [--reload]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT_DIR"

if [ ! -d models ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
  echo "Warning: models/ is empty — run scripts/train.sh first, or the API will" >&2
  echo "start but /predict will 503 until a model artifact exists." >&2
fi

uvicorn src.api.main:app --host 0.0.0.0 --port "${API_PORT:-8000}" "$@"
