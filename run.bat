@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3.10+ from https://python.org
    echo and make sure "Add python.exe to PATH" is checked during install.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a virtual environment in .venv ...
    python -m venv .venv
)

echo Installing/updating dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency install failed - see the error above.
    pause
    exit /b 1
)

echo.
echo Starting Anima Studio ...
REM pythonw.exe (not python.exe) runs with no console window at all, and
REM "start" launches it as its own detached process - so this setup
REM window closes itself right after, instead of a command prompt
REM staying open behind the app for as long as it's running.
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" desktop.py
) else (
    start "" pythonw desktop.py
)
