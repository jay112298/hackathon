#!/usr/bin/env bash
# One-command launch for the demo app.
# Usage: ./scripts/serve.sh
set -e
cd "$(dirname "$0")/.."
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
exec streamlit run app.py
