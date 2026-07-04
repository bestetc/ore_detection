@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=src;%PYTHONPATH%"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\run_backend_ui.py --host 127.0.0.1 --port 7860 --datasets-root datasets --predictions-root data_work\predictions\ui
) else (
    py -3.13 scripts\run_backend_ui.py --host 127.0.0.1 --port 7860 --datasets-root datasets --predictions-root data_work\predictions\ui
)
