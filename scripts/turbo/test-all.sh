#!/bin/bash
# Test all applications in the GreenLang monorepo using Turborepo
# This script runs tests in parallel across all packages

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  GreenLang Monorepo Test Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if turbo is installed
if ! command -v turbo &> /dev/null; then
    echo -e "${YELLOW}Turborepo not found. Installing...${NC}"
    npm install -g turbo
fi

# Change to project root
cd "${PROJECT_ROOT}"

# Parse command line arguments
FILTER=""
TEST_TYPE="test"
COVERAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            FILTER="--filter=$2"
            shift 2
            ;;
        --unit)
            TEST_TYPE="test:unit"
            shift
            ;;
        --integration)
            TEST_TYPE="test:integration"
            shift
            ;;
        --e2e)
            TEST_TYPE="test:e2e"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --filter <package>    Test only specified package"
            echo "  --unit                Run only unit tests"
            echo "  --integration         Run only integration tests"
            echo "  --e2e                 Run only e2e tests"
            echo "  --coverage            Generate coverage reports"
            echo "  --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all tests"
            echo "  $0 --unit --coverage                  # Unit tests with coverage"
            echo "  $0 --filter=@greenlang/frontend       # Test only frontend"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build turbo command
if [ "$COVERAGE" = true ]; then
    TURBO_CMD="turbo run coverage"
else
    TURBO_CMD="turbo run ${TEST_TYPE}"
fi

if [ -n "$FILTER" ]; then
    TURBO_CMD="$TURBO_CMD $FILTER"
fi

echo -e "${BLUE}Command: ${TURBO_CMD}${NC}"
echo ""

# Run the tests
if eval "$TURBO_CMD"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Tests completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"

    # Show coverage summary if enabled
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${BLUE}Coverage reports generated in coverage/ directories${NC}"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Tests failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
