#!/bin/bash
# Demo Validation Script
# Performs basic validation of the demo environment and structure

set -e

echo "🔍 Validating GreenLang Demo Setup..."
echo "======================================="

# Check script exists and is executable
if [[ -x "run_demo.sh" ]]; then
    echo "✅ Demo script is executable"
else
    echo "❌ Demo script not found or not executable"
    exit 1
fi

# Check syntax
if bash -n run_demo.sh; then
    echo "✅ Demo script syntax is valid"
else
    echo "❌ Demo script has syntax errors"
    exit 1
fi

# Check README exists
if [[ -f "README.md" ]]; then
    echo "✅ README.md documentation exists"
else
    echo "❌ README.md not found"
    exit 1
fi

# Check for required commands
commands=("python3" "git")
for cmd in "${commands[@]}"; do
    if command -v "$cmd" &> /dev/null; then
        echo "✅ $cmd is available"
    else
        echo "⚠️  $cmd not found (required for demo)"
    fi
done

# Validate directory structure
if [[ -d "data" ]]; then
    echo "✅ Data directory exists"
else
    echo "ℹ️  Data directory will be created during demo"
fi

# Test basic GreenLang installation (if available)
if command -v gl &> /dev/null; then
    echo "✅ GreenLang CLI is already installed"
    gl --version 2>/dev/null || echo "ℹ️  GreenLang available but version check failed"
else
    echo "ℹ️  GreenLang CLI not installed (will be installed during demo)"
fi

echo ""
echo "📋 Demo Validation Summary:"
echo "• Script: Ready ✅"
echo "• Documentation: Ready ✅"
echo "• Prerequisites: Checked ✅"
echo ""
echo "🚀 Ready to run: ./run_demo.sh"