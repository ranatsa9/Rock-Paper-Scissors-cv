#!/bin/zsh
cd "$(dirname "$0")" || exit 1

if [ ! -x ".venv/bin/python" ]; then
    echo "The environment is not installed yet. Running first-time setup..."
    exec ./run_mac.command
fi

echo "Starting RPS Vision Arena..."
".venv/bin/python" -m streamlit run app.py

read -k 1 "?Press any key to close..."
