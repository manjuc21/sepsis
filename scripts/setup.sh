#!/usr/bin/env bash
# One-command environment setup: creates .venv and installs dependencies.
# Auto-detects an NVIDIA GPU (via nvidia-smi) and installs the matching
# torch build — CUDA-enabled if a GPU is present, CPU-only otherwise.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  echo "Creating virtualenv at .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  echo "Installing requirements with CUDA-enabled torch..."
  pip install -r requirements.txt
else
  echo "No NVIDIA GPU detected — installing CPU-only torch."
  pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
fi

echo
echo "Setup complete."
echo "Activate the venv in new shells with: source .venv/bin/activate"
