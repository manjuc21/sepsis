#!/usr/bin/env bash
# Brings up the full demo stack (Postgres + API + React dashboard) via
# docker-compose. Run scripts/train.sh first so models/ has a trained
# artifact for the API to serve. Dashboard: http://localhost:5173
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d models ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
  echo "Error: models/ is empty. Run scripts/train.sh first." >&2
  exit 1
fi

if [ ! -f data/processed/demo_patients.json ]; then
  echo "Generating demo patient data..."
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python scripts/prepare_demo_patients.py
fi

docker compose up --build
