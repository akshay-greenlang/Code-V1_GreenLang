#!/bin/bash
################################################################################
# GL-CSRD Parallel Test Execution Script
################################################################################
#
# Purpose: Execute 975 tests in parallel for faster execution
# Execution Strategy: pytest-xdist with intelligent worker distribution
# Expected Speedup: 4-8x faster than sequential execution
#
# Usage:
#   ./scripts/run_tests_parallel.sh                # Auto-detect CPU cores
#   ./scripts/run_tests_parallel.sh --workers 8    # Specify worker count
#   ./scripts/run_tests_parallel.sh --fast         # Skip slow tests
#   ./scripts/run_tests_parallel.sh --coverage     # With coverage
#
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORTS_DIR="${BASE_DIR}/test-reports"
COVERAGE_DIR="${BASE_DIR}/htmlcov"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Auto-detect CPU cores
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    CPU_CORES=${NUMBER_OF_PROCESSORS:-4}
else
    CPU_CORES=4
fi

# Use 75% of cores for testing (leave some for system)
DEFAULT_WORKERS=$((CPU_CORES * 3 / 4))
if [ ${DEFAULT_WORKERS} -lt 2 ]; then
    DEFAULT_WORKERS=2
fi

WORKERS=${DEFAULT_WORKERS}

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  GL-CSRD Parallel Test Execution${NC}"
    echo -e "${MAGENTA}  975 Tests | ${WORKERS} Parallel Workers | ${CPU_CORES} CPU Cores${NC}"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_section() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_dependencies() {
    print_section "Checking Parallel Test Dependencies"

    # Check pytest-xdist
    python -c "import xdist" 2>/dev/null
    if [ $? -eq 0 ]; then
        XDIST_VERSION=$(python -c "import xdist; print(xdist.__version__)")
        print_success "pytest-xdist ${XDIST_VERSION} installed"
    else
        print_error "pytest-xdist not found"
        echo "  Install with: pip install pytest-xdist"
        exit 1
    fi

    # Check pytest-parallel (alternative)
    python -c "import pytest_parallel" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_info "pytest-parallel also available (alternative mode)"
    fi

    echo ""
}

optimize_worker_distribution() {
    print_section "Optimizing Worker Distribution"

    echo "  System Information:"
    echo "    CPU Cores:        ${CPU_CORES}"
    echo "    Workers:          ${WORKERS}"
    echo "    Tests per Worker: ~$((975 / WORKERS))"
    echo ""

    # Calculate estimated execution time
    # Assuming average 0.5s per test, with parallelization overhead
    SEQUENTIAL_TIME=$((975 * 1 / 2))  # 487.5 seconds = ~8 minutes
    PARALLEL_TIME=$((SEQUENTIAL_TIME / WORKERS + 30))  # Add 30s overhead

    echo "  Estimated Execution Time:"
    echo "    Sequential: ~$((SEQUENTIAL_TIME / 60)) minutes"
    echo "    Parallel:   ~$((PARALLEL_TIME / 60)) minutes (${WORKERS}x speedup)"
    echo ""
}

run_parallel_tests() {
    local mode=$1
    local extra_args=$2

    print_section "Running Parallel Test Suite"

    cd "${BASE_DIR}"

    # Base pytest arguments with xdist
    PYTEST_ARGS="-v --tb=short --color=yes"
    PYTEST_ARGS="${PYTEST_ARGS} -n ${WORKERS}"  # Parallel workers
    PYTEST_ARGS="${PYTEST_ARGS} --dist=loadscope"  # Smart test distribution

    # Add coverage if requested
    if [[ "${extra_args}" == *"--coverage"* ]]; then
        # Use coverage with parallel mode
        PYTEST_ARGS="${PYTEST_ARGS} --cov=agents --cov=cli --cov=sdk --cov=provenance"
        PYTEST_ARGS="${PYTEST_ARGS} --cov-report=html:${COVERAGE_DIR}"
        PYTEST_ARGS="${PYTEST_ARGS} --cov-report=term-missing"
        PYTEST_ARGS="${PYTEST_ARGS} --cov-report=json:${REPORTS_DIR}/coverage_parallel.json"

        # Enable parallel coverage
        export COVERAGE_PROCESS_START="${BASE_DIR}/.coveragerc"
    fi

    # Add JUnit XML
    PYTEST_ARGS="${PYTEST_ARGS} --junitxml=${REPORTS_DIR}/junit/parallel_results_${TIMESTAMP}.xml"

    # Mode-specific filters
    case "${mode}" in
        "fast")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'not slow and not performance'"
            print_info "Running fast tests only (excluding slow and performance tests)"
            ;;
        "critical")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'critical'"
            print_info "Running critical tests only"
            ;;
        "unit")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'unit'"
            print_info "Running unit tests only"
            ;;
        "integration")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'integration'"
            print_info "Running integration tests only"
            ;;
        *)
            print_info "Running all 975 tests in parallel"
            ;;
    esac

    echo ""
    echo "  Executing: pytest ${PYTEST_ARGS} tests/"
    echo ""

    START_TIME=$(date +%s)

    # Run pytest with parallel execution
    pytest ${PYTEST_ARGS} tests/ || {
        print_error "Parallel test execution failed!"
        exit 1
    }

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    print_success "Parallel test execution completed in ${DURATION} seconds"
    echo ""
}

run_parallel_by_group() {
    print_section "Running Tests by Groups (Parallel within Groups)"

    # Group tests by category for better organization
    local groups=(
        "agents:Calculator,Reporting,Audit,Intake,Materiality,Aggregator"
        "infrastructure:Pipeline,Provenance,Validation"
        "interfaces:CLI,SDK"
        "security:Encryption,Security,E2E"
    )

    for group_info in "${groups[@]}"; do
        IFS=':' read -r group_name test_names <<< "${group_info}"

        echo -e "${BLUE}Testing ${group_name} components in parallel...${NC}"

        # Convert comma-separated names to test file patterns
        test_files=""
        IFS=',' read -ra NAMES <<< "${test_names}"
        for name in "${NAMES[@]}"; do
            name_lower=$(echo "${name}" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
            # Find matching test file
            if [ -f "tests/test_${name_lower}_agent.py" ]; then
                test_files="${test_files} tests/test_${name_lower}_agent.py"
            elif [ -f "tests/test_${name_lower}.py" ]; then
                test_files="${test_files} tests/test_${name_lower}.py"
            fi
        done

        if [ -n "${test_files}" ]; then
            pytest -v -n ${WORKERS} --dist=loadscope ${test_files} -q

            if [ $? -eq 0 ]; then
                print_success "${group_name}: PASSED"
            else
                print_error "${group_name}: FAILED"
            fi
        fi

        echo ""
    done
}

run_load_balanced_tests() {
    print_section "Running Load-Balanced Test Distribution"

    cd "${BASE_DIR}"

    # Use loadfile distribution for better balance with known slow tests
    PYTEST_ARGS="-v --tb=short --color=yes"
    PYTEST_ARGS="${PYTEST_ARGS} -n ${WORKERS}"
    PYTEST_ARGS="${PYTEST_ARGS} --dist=loadfile"  # Distribute by file
    PYTEST_ARGS="${PYTEST_ARGS} --maxfail=5"  # Stop after 5 failures
    PYTEST_ARGS="${PYTEST_ARGS} --junitxml=${REPORTS_DIR}/junit/loadbalanced_${TIMESTAMP}.xml"

    print_info "Using loadfile distribution for optimal load balancing"
    echo ""

    pytest ${PYTEST_ARGS} tests/

    echo ""
}

generate_parallel_report() {
    print_section "Generating Parallel Execution Report"

    cat > "${REPORTS_DIR}/parallel_summary_${TIMESTAMP}.txt" <<EOF
GL-CSRD Parallel Test Execution Summary
Generated: $(date)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARALLEL EXECUTION CONFIGURATION
────────────────────────────────────────────────────────────────

CPU Cores:              ${CPU_CORES}
Parallel Workers:       ${WORKERS}
Distribution Strategy:  loadscope (smart distribution)
Tests per Worker:       ~$((975 / WORKERS))

PERFORMANCE METRICS
────────────────────────────────────────────────────────────────

Total Tests:            975
Sequential Est.:        ~8 minutes
Parallel Execution:     ~${1} seconds
Speedup Factor:         ~$((480 / $1))x

WORKER EFFICIENCY
────────────────────────────────────────────────────────────────

Worker utilization:     Optimal (load-balanced)
Test distribution:      Automatic by pytest-xdist
Overhead:               Minimal (shared fixtures)

REPORTS GENERATED
────────────────────────────────────────────────────────────────

JUnit XML:              ${REPORTS_DIR}/junit/parallel_results_${TIMESTAMP}.xml
Coverage Report:        ${COVERAGE_DIR}/index.html

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

    cat "${REPORTS_DIR}/parallel_summary_${TIMESTAMP}.txt"
    echo ""
    print_success "Parallel execution report saved"
    echo ""
}

show_performance_comparison() {
    print_section "Performance Comparison"

    echo "  Execution Mode Comparison:"
    echo ""
    echo "  ┌─────────────────┬──────────────┬──────────────┬───────────┐"
    echo "  │ Mode            │ Workers      │ Est. Time    │ Use Case  │"
    echo "  ├─────────────────┼──────────────┼──────────────┼───────────┤"
    echo "  │ Sequential      │ 1            │ ~8 min       │ Debug     │"
    echo "  │ Parallel (2x)   │ 2            │ ~4 min       │ Basic     │"
    echo "  │ Parallel (4x)   │ 4            │ ~2 min       │ Standard  │"
    echo "  │ Parallel (8x)   │ 8            │ ~1 min       │ Fast CI   │"
    echo "  │ Auto (Current)  │ ${WORKERS}            │ ~$((480 / WORKERS / 60)) min       │ Optimal   │"
    echo "  └─────────────────┴──────────────┴──────────────┴───────────┘"
    echo ""
}

################################################################################
# Main Execution
################################################################################

main() {
    # Parse arguments
    MODE="all"
    EXTRA_ARGS=""

    for arg in "$@"; do
        case $arg in
            --workers=*)
                WORKERS="${arg#*=}"
                ;;
            --fast)
                MODE="fast"
                ;;
            --critical)
                MODE="critical"
                ;;
            --unit)
                MODE="unit"
                ;;
            --integration)
                MODE="integration"
                ;;
            --by-group)
                MODE="by-group"
                ;;
            --load-balanced)
                MODE="load-balanced"
                ;;
            --coverage)
                EXTRA_ARGS="${EXTRA_ARGS} --coverage"
                ;;
            *)
                ;;
        esac
    done

    print_header
    check_dependencies
    optimize_worker_distribution
    show_performance_comparison

    mkdir -p "${REPORTS_DIR}/junit"

    START_TIME=$(date +%s)

    case "${MODE}" in
        "by-group")
            run_parallel_by_group
            ;;
        "load-balanced")
            run_load_balanced_tests
            ;;
        *)
            run_parallel_tests "${MODE}" "${EXTRA_ARGS}"
            ;;
    esac

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    generate_parallel_report "${DURATION}"

    print_success "Parallel test execution complete!"
    echo ""
    echo "  Execution Time: ${DURATION} seconds (~$((DURATION / 60)) minutes)"
    echo "  Speedup: ~$((480 / DURATION))x faster than sequential"
    echo ""
}

main "$@"
