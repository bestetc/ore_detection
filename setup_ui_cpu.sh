#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v python3.13 >/dev/null 2>&1; then
  PYTHON_BASE=(python3.13)
elif command -v py >/dev/null 2>&1; then
  PYTHON_BASE=(py -3.13)
else
  echo "Python 3.13 is required. Install it, then rerun this script." >&2
  exit 1
fi

if [ ! -x ".venv/bin/python" ] && [ ! -x ".venv/Scripts/python.exe" ]; then
  "${PYTHON_BASE[@]}" -m venv .venv
fi

if [ -x ".venv/bin/python" ]; then
  PYTHON_EXE=".venv/bin/python"
else
  PYTHON_EXE=".venv/Scripts/python.exe"
fi

"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install -r requirements.txt

if [ "$(uname -s)" = "Darwin" ]; then
  "$PYTHON_EXE" -m pip install torch
else
  "$PYTHON_EXE" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "CPU UI environment is ready."
echo "Start the UI with ./run_ui.sh, then open http://127.0.0.1:7860"
