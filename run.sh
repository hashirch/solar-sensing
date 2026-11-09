#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Error: virtual environment not found. Run ./install.sh first." >&2
    exit 1
fi

cd "${ROOT_DIR}"
exec "${PYTHON}" -m streamlit run frontend/app.py "$@"
