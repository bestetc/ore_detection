#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
# Equivalent minimal form: PYTHONPATH=src py -3.13 scripts/run_backend_ui.py
py -3.13 scripts/run_backend_ui.py --host 127.0.0.1 --port 7860 --datasets-root datasets --predictions-root data_work/predictions/ui
