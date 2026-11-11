#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"

echo "=========================================="
echo " Starting Solar Sensing Platform          "
echo "=========================================="

if [[ -f "${ROOT_DIR}/.env" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
        [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
        key="${line%%=*}"
        value="${line#*=}"
        export "${key}=${value}"
    done < "${ROOT_DIR}/.env"
fi

# Ensure Node/NPM from environment or NVM
if ! command -v npm >/dev/null 2>&1; then
    if [[ -d "$HOME/.nvm" ]]; then
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    fi
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm / node is required but not found in PATH." >&2
    exit 1
fi

# Start PyTorch model server on port 8000 if not already running
if ! lsof -i :8000 >/dev/null 2>&1; then
    echo "Starting PyTorch Model Inference Server (port 8000)..."
    (cd "${ROOT_DIR}" && python3 -m backend.server) &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
    sleep 2
fi

echo "Starting React Frontend on port 5173..."
cd "${FRONTEND_DIR}"
exec npm run dev -- --host
