#!/bin/bash
# Quick installation script for greenlang-cli without re-downloading (Linux/Mac)

echo "GreenLang CLI Quick Installer"
echo "============================="
echo

# Check if wheels directory exists
if [ -d "wheels" ]; then
    echo "[FAST MODE] Installing from local wheels directory..."
    pip install --no-index --find-links wheels greenlang-cli
    if [ $? -eq 0 ]; then
        echo
        echo "Installation successful!"
        gl --version
        exit 0
    else
        echo "Local installation failed, falling back to online mode..."
    fi
fi

# Check if requirements-lock.txt exists
if [ -f "requirements-lock.txt" ]; then
    echo "Installing from locked requirements..."
    pip install -r requirements-lock.txt
    if [ $? -eq 0 ]; then
        echo
        echo "Installation successful!"
        gl --version
        exit 0
    fi
fi

# Fallback to regular installation
echo "Installing from PyPI..."
pip install greenlang-cli
if [ $? -eq 0 ]; then
    echo
    echo "Installation successful!"
    gl --version
fi