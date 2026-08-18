#!/usr/bin/env bash
# Stops the docker-compose demo stack started by scripts/demo.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker compose down
