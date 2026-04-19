#!/bin/bash
# GreenLang Agent Factory CLI - Unix/Linux/macOS Test Script

set -e

echo ""
echo "============================================================"
echo " GreenLang Agent Factory CLI - Running Tests"
echo "============================================================"
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found"
    echo "Please run install.sh first"
    exit 1
fi

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"
echo ""

# Run tests
echo "Running tests..."
pytest tests/ -v --cov=cli --cov-report=term-missing --cov-report=html
echo ""

# Run linting
echo "Running linters..."
echo ""
echo "--- Black (formatting check) ---"
black --check cli/
echo ""

echo "--- Ruff (linting) ---"
ruff check cli/
echo ""

echo "--- MyPy (type checking) ---"
mypy cli/
echo ""

echo "============================================================"
echo " Tests Complete!"
echo "============================================================"
echo ""
echo "Coverage report: htmlcov/index.html"
echo ""
