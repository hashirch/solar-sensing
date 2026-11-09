#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP=("${PYTHON}" -m pip)

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required but was not found." >&2
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

echo "Installing project requirements..."
"${PIP[@]}" install --upgrade pip
"${PIP[@]}" install -r "${ROOT_DIR}/requirements-local.txt"

if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch is missing; installing PyTorch and torchvision..."
    "${PIP[@]}" install torch torchvision
fi

if ! "${PYTHON}" -c "import detectron2" >/dev/null 2>&1; then
    echo "Detectron2 is missing; installing Detectron2..."
    "${PIP[@]}" install --no-build-isolation \
        "git+https://github.com/facebookresearch/detectron2.git"
fi
