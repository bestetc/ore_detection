#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_CMD=(.venv/bin/python)
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON_CMD=(.venv/Scripts/python.exe)
elif command -v python3.13 >/dev/null 2>&1; then
  PYTHON_CMD=(python3.13)
elif command -v py >/dev/null 2>&1; then
  PYTHON_CMD=(py -3.13)
else
  echo "Python 3.13 is required. Run setup_ui_cpu.sh first or install Python 3.13." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" scripts/run_backend_ui.py --host 127.0.0.1 --port 7860 --datasets-root datasets --predictions-root data_work/predictions/ui
