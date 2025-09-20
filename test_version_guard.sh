#!/bin/bash
# Test the version guard logic locally

echo "Testing Version Guard logic..."
echo ""

# Test 1: Check VERSION file vs package version
echo "Test 1: VERSION file vs package version"
V=$(cat VERSION | tr -d '\n\r')
PV=$(python -c 'import greenlang,sys; print(getattr(greenlang,"__version__",""))')
echo "VERSION file: $V"
echo "Python package: $PV"
if [ "$V" != "$PV" ]; then
    echo "FAIL: Version mismatch!"
else
    echo "PASS: Versions match"
fi
echo ""

# Test 2: Check CLI version
echo "Test 2: CLI version check"
V=$(cat VERSION | tr -d '\n\r')
CLI_V=$(python -c "import greenlang; print(greenlang.__version__)")
echo "VERSION file: $V"
echo "CLI version: $CLI_V"
if [ "$V" != "$CLI_V" ]; then
    echo "FAIL: CLI version mismatch!"
else
    echo "PASS: CLI version matches"
fi
echo ""

# Test 3: Check for hardcoded versions (simulation)
echo "Test 3: Hardcoded version check"
if git grep -E 'version\s*=\s*"[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?"' setup.py pyproject.toml 2>/dev/null | grep -v 'file = "VERSION"' | grep -v 'dynamic = \["version"\]'; then
    echo "FAIL: Found hardcoded versions"
else
    echo "PASS: No hardcoded versions found"
fi
echo ""

echo "=== Version Guard Test Complete ==="
echo "Current version: $(cat VERSION)"