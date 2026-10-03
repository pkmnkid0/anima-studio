#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found on PATH. Install Python 3.10+ first."
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "Creating a virtual environment in .venv ..."
    python3 -m venv .venv
fi

echo "Installing/updating dependencies ..."
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Starting Anima Studio ..."
.venv/bin/python run.py
