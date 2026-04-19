#!/bin/bash
# GreenLang Agent Factory CLI - Unix/Linux/macOS Installation Script

set -e

echo ""
echo "============================================================"
echo " GreenLang Agent Factory CLI - Installation"
echo "============================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

echo "Checking Python version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" || {
    echo "ERROR: Python 3.11 or higher is required"
    python3 --version
    exit 1
}

echo "Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel
echo ""

# Install CLI
echo "Installing GreenLang Agent Factory CLI..."
pip install -e .
echo ""

# Verify installation
echo "Verifying installation..."
gl --version
echo ""

echo "============================================================"
echo " Installation Complete!"
echo "============================================================"
echo ""
echo "The 'gl' command is now available."
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "Quick start:"
echo "  gl --help"
echo "  gl init"
echo "  gl agent list"
echo ""
echo "Documentation:"
echo "  README.md - Complete guide"
echo "  QUICKSTART.md - 5-minute tutorial"
echo "  INSTALL.md - Installation details"
echo ""
