#!/usr/bin/env bash
set -e

# Navigate to script directory
cd "$(dirname "$0")"

echo "=================================================="
echo "    Launching Scrawl — Neural Handwriting OCR    "
echo "        Engineered by Trigun Parvathapuram        "
echo "=================================================="

# Check for virtual environment
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
    UVICORN_BIN=".venv/bin/uvicorn"
else
    PYTHON_BIN="python3"
    UVICORN_BIN="uvicorn"
fi

echo "[*] Using Python: $($PYTHON_BIN --version)"
echo "[*] Verifying model weights..."
if [ ! -f "emnist_alphanumeric_cnn.keras" ]; then
    echo "[!] Error: emnist_alphanumeric_cnn.keras not found in $(pwd)"
    exit 1
fi

echo "[*] Launching application and opening Chrome..."
echo "--------------------------------------------------"

$PYTHON_BIN app.py
