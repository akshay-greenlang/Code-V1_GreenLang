#!/bin/bash
# Setup pip cache and create offline wheel directory for greenlang-cli (Linux/Mac)

echo "GreenLang CLI - Cache Setup"
echo "============================"
echo

# Show current pip cache location
echo "[INFO] Current pip cache location:"
pip cache dir
echo

# Create wheels directory
echo "Creating wheels directory for offline installations..."
mkdir -p wheels

# Download all dependencies to wheels directory
echo "Downloading all dependencies to wheels directory..."
pip wheel greenlang-cli -w wheels

echo
echo "==============================================="
echo "Setup complete!"
echo "==============================================="
echo
echo "To install greenlang-cli without downloading:"
echo
echo "Option 1: Use the locked requirements (requires internet but faster):"
echo "  pip install -r requirements-lock.txt"
echo
echo "Option 2: Install from local wheels (offline):"
echo "  pip install --no-index --find-links wheels greenlang-cli"
echo
echo "Option 3: Use pip's download cache:"
echo "  pip install greenlang-cli  # Will use cached files automatically"
echo