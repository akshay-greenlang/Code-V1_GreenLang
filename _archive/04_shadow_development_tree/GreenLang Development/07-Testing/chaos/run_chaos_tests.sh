#!/bin/bash
# Chaos Engineering Test Suite Runner
#
# Usage: ./run_chaos_tests.sh [options]
#
# Options:
#   --all           Run all chaos tests (default)
#   --failover      Run failover tests only
#   --database      Run database resilience tests only
#   --latency       Run latency/timeout tests only
#   --resource      Run resource pressure tests only
#   --integration   Run integration tests only
#   --quick         Run quick tests only (no slow tests)
#   --verbose       Show detailed output with logging
#   --report        Generate JSON and HTML reports
#   --help          Show this help message

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
REPORT_DIR="${PROJECT_DIR}/chaos-reports"

# Create report directory
mkdir -p "$REPORT_DIR"

# Default options
RUN_MODE="all"
VERBOSE=false
GENERATE_REPORT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_MODE="all"
            shift
            ;;
        --failover)
            RUN_MODE="failover"
            shift
            ;;
        --database)
            RUN_MODE="database"
            shift
            ;;
        --latency)
            RUN_MODE="latency"
            shift
            ;;
        --resource)
            RUN_MODE="resource"
            shift
            ;;
        --integration)
            RUN_MODE="integration"
            shift
            ;;
        --quick)
            RUN_MODE="quick"
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/bin/bash"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest $SCRIPT_DIR"
PYTEST_ARGS="-v -m chaos"

case $RUN_MODE in
    failover)
        PYTEST_ARGS="$PYTEST_ARGS and chaos_failover"
        echo "Running Failover Tests..."
        ;;
    database)
        PYTEST_ARGS="$PYTEST_ARGS and chaos_database"
        echo "Running Database Resilience Tests..."
        ;;
    latency)
        PYTEST_ARGS="$PYTEST_ARGS and chaos_latency"
        echo "Running Latency/Timeout Tests..."
        ;;
    resource)
        PYTEST_ARGS="$PYTEST_ARGS and chaos_resource"
        echo "Running Resource Pressure Tests..."
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD/test_process_heat_agent_chaos.py"
        echo "Running Integration Tests..."
        ;;
    quick)
        PYTEST_ARGS="$PYTEST_ARGS and not chaos_slow"
        echo "Running Quick Tests (excluding slow tests)..."
        ;;
    *)
        echo "Running All Chaos Tests..."
        ;;
esac

# Add verbose logging if requested
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -s --log-cli-level=DEBUG"
fi

# Add report generation if requested
if [ "$GENERATE_REPORT" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --junitxml=$REPORT_DIR/chaos-results.xml --html=$REPORT_DIR/chaos-report.html"
    echo "Reports will be saved to: $REPORT_DIR"
fi

# Add timeout and other options
PYTEST_ARGS="$PYTEST_ARGS --timeout=300 --tb=short"

# Run tests
echo "Command: $PYTEST_CMD $PYTEST_ARGS"
echo "---"

$PYTEST_CMD $PYTEST_ARGS

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "===== CHAOS TESTS PASSED ====="

    if [ "$GENERATE_REPORT" = true ]; then
        echo "Reports available:"
        echo "  - JSON: $REPORT_DIR/chaos-results.xml"
        echo "  - HTML: $REPORT_DIR/chaos-report.html"
    fi

    exit 0
else
    echo ""
    echo "===== CHAOS TESTS FAILED ====="
    exit 1
fi
