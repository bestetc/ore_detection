@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv
    if errorlevel 1 exit /b 1
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
"%PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1

echo.
echo CPU UI environment is ready.
echo Start the UI with run_ui.cmd, then open http://127.0.0.1:7860
