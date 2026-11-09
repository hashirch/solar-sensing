#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required but was not found." >&2
    exit 1
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "Error: pip for python3 is required but was not found." >&2
    exit 1
fi

echo "Installing project requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install -r "${ROOT_DIR}/requirements-local.txt"

if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch is missing; installing PyTorch and torchvision..."
    python3 -m pip install torch torchvision
fi

if ! python3 -c "import detectron2" >/dev/null 2>&1; then
    echo "Detectron2 is missing; installing Detectron2..."
    python3 -m pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git"
fi
