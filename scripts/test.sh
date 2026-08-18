#!/usr/bin/env bash
# Runs the test suite. Passes any extra args straight to pytest,
# e.g. scripts/test.sh tests/test_metrics.py -k auroc
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate
pytest "$@"
