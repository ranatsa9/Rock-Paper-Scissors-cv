#!/bin/zsh
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 was not found. Install Python 3.11 or 3.12 first."
    read -k 1 "?Press any key to close..."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating the local Python environment..."
    python3 -m venv .venv || exit 1
fi

echo "Installing or checking dependencies..."
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt || exit 1

echo "Starting RPS Vision Arena..."
".venv/bin/python" -m streamlit run app.py

read -k 1 "?Press any key to close..."
