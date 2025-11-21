#!/bin/bash
# GreenLang Acceptance Test Runner
# Run this script to validate all acceptance criteria

set -e

echo "=================================================="
echo "    GreenLang Acceptance Test Suite"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python version..."
python --version || python3 --version

# Check required tools
echo ""
echo "Checking required tools..."

check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 is not installed (some tests may be skipped)"
        return 1
    fi
}

check_tool "gl"
check_tool "cosign"
check_tool "oras"
check_tool "opa"
check_tool "kubectl"

echo ""
echo "=================================================="
echo "Running Acceptance Tests"
echo "=================================================="
echo ""

# Run all tests
if [ "$1" == "--quick" ]; then
    echo "Running quick tests only..."
    python acceptance_test.py --test scaffolding
    python acceptance_test.py --test determinism
else
    echo "Running full test suite..."
    python acceptance_test.py --verbose --export-results acceptance-results.json
fi

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=================================================="
    echo "✅ ALL ACCEPTANCE TESTS PASSED!"
    echo "==================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review test results in acceptance-results.json"
    echo "2. Check performance metrics"
    echo "3. Create PR with results"
else
    echo ""
    echo -e "${RED}=================================================="
    echo "❌ SOME TESTS FAILED"
    echo "==================================================${NC}"
    echo ""
    echo "Please review the failures above and fix before merging."
    exit 1
fi

# Performance summary
echo ""
echo "Performance Summary:"
echo "-------------------"

if [ -f "acceptance-results.json" ]; then
    python -c "
import json
with open('acceptance-results.json') as f:
    data = json.load(f)
    timings = data.get('timings', {})
    if timings:
        avg = sum(timings.values()) / len(timings)
        max_time = max(timings.values())
        print(f'Average test time: {avg:.2f}s')
        print(f'Slowest test: {max_time:.2f}s')
        
        # Check performance requirements
        if max_time <= 60:
            print('✅ All tests within 60s limit')
        else:
            print('⚠️  Some tests exceeded 60s limit')
    "
fi

echo ""
echo "=================================================="
echo "Test run complete!"
echo "==================================================