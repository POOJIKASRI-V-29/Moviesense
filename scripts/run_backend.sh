#!/usr/bin/env bash
# Start the FastAPI backend. Loads settings from .env if present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -f .env ]] && set -a && source .env && set +a
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-.venv/bin/python}" -m uvicorn moviesense.api.main:app \
  --host "${MOVIESENSE_HOST:-127.0.0.1}" --port "${MOVIESENSE_PORT:-8000}"
