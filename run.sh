#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python" ]; then
    .venv/bin/python gesture_controller.py "$@"
else
    python3 gesture_controller.py "$@"
fi
