#!/bin/bash
# setup.sh — Run this once to set up the project
# Usage: bash setup.sh

set -e
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   CrossFakeNet — Environment Setup       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Create virtual environment
echo "[1/4] Creating virtual environment..."
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
echo "[2/4] Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "[3/4] Installing dependencies (this may take 5–10 mins)..."
pip install -r requirements.txt

# Create uploads directory
echo "[4/4] Creating uploads directory..."
mkdir -p uploads

echo ""
echo "✅  Setup complete!"
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "Then open: http://127.0.0.1:5000"
echo ""
