#!/usr/bin/env bash
# Serve the redesigned UI (FastAPI + static/index.html) on http://localhost:8000
cd "$(dirname "$0")/.."
exec .venv/bin/python -m uvicorn webapp:app --host 127.0.0.1 --port 8000 "$@"
