#!/bin/bash
# ============================================================================
# GL-CBAM-APP - Comprehensive Test Execution Script
# ============================================================================
#
# Purpose: Execute all 320+ tests with coverage reporting and validation
#
# Usage:
#   ./scripts/run_all_tests.sh              # Run all tests with coverage
#   ./scripts/run_all_tests.sh --fast       # Skip coverage (faster)
#   ./scripts/run_all_tests.sh --unit       # Unit tests only
#   ./scripts/run_all_tests.sh --smoke      # Quick smoke tests
#   ./scripts/run_all_tests.sh --compliance # Compliance tests only
#   ./scripts/run_all_tests.sh --parallel   # Parallel execution
#
# Outputs:
#   - Terminal test results
#   - HTML coverage report: htmlcov/index.html
#   - HTML test report: test-report.html
#   - Test execution log: test-execution.log
#   - JUnit XML: test-results.xml
#
# Version: 1.0.0
# ============================================================================

set -e  # Exit on error

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results directory
RESULTS_DIR="$PROJECT_ROOT/test-results"
mkdir -p "$RESULTS_DIR"

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ----------------------------------------------------------------------------
# Pre-flight Checks
# ----------------------------------------------------------------------------
print_header "GL-CBAM Test Execution Suite"

# Check if in correct directory
cd "$PROJECT_ROOT" || exit 1
print_info "Working directory: $PROJECT_ROOT"

# Check Python version
print_info "Python version:"
python --version

# Check pytest installation
if ! command -v pytest &> /dev/null; then
    print_error "pytest not found. Install dependencies:"
    echo "  pip install -r requirements-test.txt"
    exit 1
fi

print_success "pytest found: $(pytest --version | head -n 1)"

# Check if required packages are installed
python -c "import pytest_cov, pytest_html" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Some pytest plugins missing. Installing..."
    pip install -q pytest-cov pytest-html pytest-xdist
fi

# ----------------------------------------------------------------------------
# Parse Command Line Arguments
# ----------------------------------------------------------------------------
MODE="full"
PARALLEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            MODE="fast"
            shift
            ;;
        --unit)
            MODE="unit"
            shift
            ;;
        --integration)
            MODE="integration"
            shift
            ;;
        --performance)
            MODE="performance"
            shift
            ;;
        --compliance)
            MODE="compliance"
            shift
            ;;
        --smoke)
            MODE="smoke"
            shift
            ;;
        --parallel|-n)
            PARALLEL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fast         Skip coverage (faster execution)"
            echo "  --unit         Run unit tests only"
            echo "  --integration  Run integration tests only"
            echo "  --performance  Run performance benchmarks"
            echo "  --compliance   Run compliance tests only (CRITICAL)"
            echo "  --smoke        Run quick smoke tests"
            echo "  --parallel     Run tests in parallel (faster)"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ----------------------------------------------------------------------------
# Test Execution
# ----------------------------------------------------------------------------
print_header "Test Configuration"
print_info "Mode: $MODE"
print_info "Parallel: $PARALLEL"
print_info "Results: $RESULTS_DIR"

# Build pytest command
PYTEST_CMD="pytest"
PYTEST_ARGS=""

# Add parallel execution if requested
if [ "$PARALLEL" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -n auto"
    print_info "Parallel execution enabled (using all CPU cores)"
fi

# Add mode-specific arguments
case $MODE in
    fast)
        print_info "Fast mode: Skipping coverage"
        PYTEST_ARGS="$PYTEST_ARGS --tb=short"
        ;;
    unit)
        print_info "Running unit tests only"
        PYTEST_ARGS="$PYTEST_ARGS -m unit --cov --cov-report=html --cov-report=term"
        ;;
    integration)
        print_info "Running integration tests only"
        PYTEST_ARGS="$PYTEST_ARGS -m integration --cov --cov-report=html --cov-report=term"
        ;;
    performance)
        print_info "Running performance benchmarks"
        PYTEST_ARGS="$PYTEST_ARGS -m performance --benchmark-only"
        ;;
    compliance)
        print_info "Running CRITICAL compliance tests"
        PYTEST_ARGS="$PYTEST_ARGS -m compliance --cov --cov-report=html --cov-report=term"
        ;;
    smoke)
        print_info "Running quick smoke tests"
        PYTEST_ARGS="$PYTEST_ARGS -m smoke --tb=short --maxfail=1"
        ;;
    full)
        print_info "Running FULL test suite with coverage"
        PYTEST_ARGS="$PYTEST_ARGS --cov --cov-report=html --cov-report=term --cov-report=json"
        PYTEST_ARGS="$PYTEST_ARGS --html=$RESULTS_DIR/test-report-$TIMESTAMP.html --self-contained-html"
        PYTEST_ARGS="$PYTEST_ARGS --junitxml=$RESULTS_DIR/test-results-$TIMESTAMP.xml"
        ;;
esac

# ----------------------------------------------------------------------------
# Execute Tests
# ----------------------------------------------------------------------------
print_header "Executing Tests"

# Record start time
START_TIME=$(date +%s)

# Run tests
set +e  # Don't exit on test failures
$PYTEST_CMD $PYTEST_ARGS
TEST_EXIT_CODE=$?
set -e

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# ----------------------------------------------------------------------------
# Results Summary
# ----------------------------------------------------------------------------
print_header "Test Execution Summary"

# Calculate duration
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))
print_info "Execution time: ${MINUTES}m ${SECONDS}s"

# Check results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "ALL TESTS PASSED!"

    # Additional validation for full mode
    if [ "$MODE" = "full" ]; then
        print_info ""
        print_info "Generated Reports:"

        if [ -d "htmlcov" ]; then
            print_success "Coverage report: htmlcov/index.html"
        fi

        if [ -f "$RESULTS_DIR/test-report-$TIMESTAMP.html" ]; then
            print_success "Test report: test-results/test-report-$TIMESTAMP.html"
        fi

        if [ -f "$RESULTS_DIR/test-results-$TIMESTAMP.xml" ]; then
            print_success "JUnit XML: test-results/test-results-$TIMESTAMP.xml"
        fi

        # Check coverage threshold
        if [ -f "coverage.json" ]; then
            COVERAGE=$(python -c "import json; data=json.load(open('coverage.json')); print(f\"{data['totals']['percent_covered']:.1f}\")")
            print_info ""
            print_info "Code Coverage: ${COVERAGE}%"

            # Compare against target
            if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
                print_success "Coverage meets target (≥80%)"
            else
                print_warning "Coverage below target (${COVERAGE}% < 80%)"
            fi
        fi
    fi

    print_info ""
    print_success "Test execution completed successfully!"
    exit 0
else
    print_error "TESTS FAILED!"
    print_info ""
    print_info "Exit code: $TEST_EXIT_CODE"

    if [ -f "test-execution.log" ]; then
        print_info "Check test-execution.log for details"
    fi

    if [ "$MODE" = "full" ] && [ -f "$RESULTS_DIR/test-report-$TIMESTAMP.html" ]; then
        print_info "HTML report: test-results/test-report-$TIMESTAMP.html"
    fi

    print_info ""
    print_error "Please fix failing tests and re-run"
    exit $TEST_EXIT_CODE
fi

# ----------------------------------------------------------------------------
# Post-execution Actions
# ----------------------------------------------------------------------------
if [ "$MODE" = "full" ]; then
    print_info ""
    print_info "Additional Validation:"

    # Count tests executed
    if [ -f "$RESULTS_DIR/test-results-$TIMESTAMP.xml" ]; then
        TEST_COUNT=$(grep -o 'tests="[0-9]*"' "$RESULTS_DIR/test-results-$TIMESTAMP.xml" | grep -o '[0-9]*' | head -1)
        print_info "Tests executed: $TEST_COUNT"

        # Validate we ran expected number of tests
        if [ "$TEST_COUNT" -ge 300 ]; then
            print_success "Test count meets expectations (≥300 tests)"
        else
            print_warning "Test count lower than expected ($TEST_COUNT < 300)"
        fi
    fi

    # Generate coverage badge (if coverage-badge installed)
    if command -v coverage-badge &> /dev/null && [ -f "coverage.json" ]; then
        coverage-badge -o coverage.svg -f
        print_success "Coverage badge generated: coverage.svg"
    fi
fi

print_info ""
print_info "Test execution completed at $(date)"

# ============================================================================
# END OF SCRIPT
# ============================================================================
