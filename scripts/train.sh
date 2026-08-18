#!/usr/bin/env bash
# Runs the full pipeline: raw data -> preprocessed -> trained -> evaluated.
# Requires data/raw/{training_setA,training_setB} to already be populated
# (see scripts/download_data.sh). Trained artifacts land in models/,
# metrics in results/comparison.csv.
#
# Usage: scripts/train.sh [--models logreg,xgboost,lstm] [--threshold 0.5]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate
python scripts/run_pipeline.py "$@"
