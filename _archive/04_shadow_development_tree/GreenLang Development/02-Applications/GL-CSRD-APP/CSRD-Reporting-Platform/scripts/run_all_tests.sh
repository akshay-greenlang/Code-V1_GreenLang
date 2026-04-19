#!/bin/bash
################################################################################
# GL-CSRD Test Execution Script - Run All 975 Tests
################################################################################
#
# Purpose: Execute complete CSRD test suite with coverage reporting
# Test Count: 975 tests across 14 test files
# Target Coverage: 90%+
#
# Usage:
#   ./scripts/run_all_tests.sh                    # Run all tests
#   ./scripts/run_all_tests.sh --fast             # Skip slow tests
#   ./scripts/run_all_tests.sh --critical         # Only critical tests
#   ./scripts/run_all_tests.sh --coverage         # With coverage report
#   ./scripts/run_all_tests.sh --html             # Generate HTML report
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="${BASE_DIR}/tests"
REPORTS_DIR="${BASE_DIR}/test-reports"
COVERAGE_DIR="${BASE_DIR}/htmlcov"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test counts by agent (975 total)
CALCULATOR_TESTS=109
INTAKE_TESTS=107
REPORTING_TESTS=133
AUDIT_TESTS=115
PROVENANCE_TESTS=101
AGGREGATOR_TESTS=75
CLI_TESTS=69
SDK_TESTS=61
PIPELINE_TESTS=59
VALIDATION_TESTS=55
MATERIALITY_TESTS=45
ENCRYPTION_TESTS=24
SECURITY_TESTS=16
E2E_TESTS=6

################################################################################
# Functions
################################################################################

print_header() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  GL-CSRD Test Execution Suite${NC}"
    echo -e "${BLUE}  975 Tests | 14 Test Files | Target: 90%+ Coverage${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
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

check_environment() {
    print_section "Checking Test Environment"

    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "  Python Version: ${PYTHON_VERSION}"

    # Check pytest installation
    if ! command -v pytest &> /dev/null; then
        print_error "pytest not found. Install with: pip install -r requirements-test.txt"
        exit 1
    fi

    PYTEST_VERSION=$(pytest --version | head -n1)
    echo "  ${PYTEST_VERSION}"

    # Check for test dependencies
    python -c "import pytest_cov, pytest_asyncio, pytest_mock, pytest_html" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "All test dependencies installed"
    else
        print_error "Missing test dependencies. Run: pip install -r requirements-test.txt"
        exit 1
    fi

    echo ""
}

create_directories() {
    print_section "Creating Report Directories"

    mkdir -p "${REPORTS_DIR}"
    mkdir -p "${COVERAGE_DIR}"
    mkdir -p "${REPORTS_DIR}/junit"
    mkdir -p "${REPORTS_DIR}/html"

    print_success "Directories created"
    echo ""
}

show_test_summary() {
    print_section "Test Suite Summary"

    echo "  Test Breakdown by Agent:"
    echo "    ├── Calculator Agent:        ${CALCULATOR_TESTS} tests (11.2%)"
    echo "    ├── Reporting Agent:         ${REPORTING_TESTS} tests (13.6%)"
    echo "    ├── Audit Agent:             ${AUDIT_TESTS} tests (11.8%)"
    echo "    ├── Intake Agent:            ${INTAKE_TESTS} tests (11.0%)"
    echo "    ├── Provenance System:       ${PROVENANCE_TESTS} tests (10.4%)"
    echo "    ├── Aggregator Agent:        ${AGGREGATOR_TESTS} tests (7.7%)"
    echo "    ├── CLI Interface:           ${CLI_TESTS} tests (7.1%)"
    echo "    ├── SDK:                     ${SDK_TESTS} tests (6.3%)"
    echo "    ├── Pipeline Integration:    ${PIPELINE_TESTS} tests (6.1%)"
    echo "    ├── Validation System:       ${VALIDATION_TESTS} tests (5.6%)"
    echo "    ├── Materiality Agent:       ${MATERIALITY_TESTS} tests (4.6%)"
    echo "    ├── Encryption:              ${ENCRYPTION_TESTS} tests (2.5%)"
    echo "    ├── Security:                ${SECURITY_TESTS} tests (1.6%)"
    echo "    └── E2E Workflows:           ${E2E_TESTS} tests (0.6%)"
    echo "  ───────────────────────────────────────────────"
    echo "  Total:                       975 tests (100%)"
    echo ""
}

run_test_suite() {
    local mode=$1
    local extra_args=$2

    print_section "Running Test Suite: ${mode}"

    cd "${BASE_DIR}"

    # Base pytest arguments
    PYTEST_ARGS="-v --tb=short --color=yes"

    # Add coverage if requested
    if [[ "${mode}" == *"coverage"* ]] || [[ "${extra_args}" == *"--coverage"* ]]; then
        PYTEST_ARGS="${PYTEST_ARGS} --cov=agents --cov=cli --cov=sdk --cov=provenance"
        PYTEST_ARGS="${PYTEST_ARGS} --cov-report=html:${COVERAGE_DIR}"
        PYTEST_ARGS="${PYTEST_ARGS} --cov-report=term-missing"
        PYTEST_ARGS="${PYTEST_ARGS} --cov-report=json:${REPORTS_DIR}/coverage.json"
    fi

    # Add HTML report if requested
    if [[ "${extra_args}" == *"--html"* ]]; then
        PYTEST_ARGS="${PYTEST_ARGS} --html=${REPORTS_DIR}/html/report_${TIMESTAMP}.html --self-contained-html"
    fi

    # Add JUnit XML for CI/CD
    PYTEST_ARGS="${PYTEST_ARGS} --junitxml=${REPORTS_DIR}/junit/results_${TIMESTAMP}.xml"

    # Mode-specific filters
    case "${mode}" in
        "fast")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'not slow and not performance'"
            ;;
        "critical")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'critical'"
            ;;
        "unit")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'unit'"
            ;;
        "integration")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'integration'"
            ;;
        "performance")
            PYTEST_ARGS="${PYTEST_ARGS} -m 'performance'"
            ;;
        *)
            # Run all tests
            ;;
    esac

    echo "  Executing: pytest ${PYTEST_ARGS} tests/"
    echo ""

    # Run pytest
    pytest ${PYTEST_ARGS} tests/ || {
        print_error "Test execution failed!"
        exit 1
    }

    echo ""
}

run_tests_by_agent() {
    print_section "Running Tests by Agent (Sequential)"

    local agents=(
        "calculator_agent:109:Calculator Agent (Zero Hallucination)"
        "reporting_agent:133:Reporting Agent (XBRL/ESEF)"
        "audit_agent:115:Audit Agent (Compliance)"
        "intake_agent:107:Intake Agent (Data Ingestion)"
        "provenance:101:Provenance System"
        "aggregator_agent:75:Aggregator Agent"
        "cli:69:CLI Interface"
        "sdk:61:SDK"
        "pipeline_integration:59:Pipeline Integration"
        "validation:55:Validation System"
        "materiality_agent:45:Materiality Agent"
        "encryption:24:Encryption"
        "automated_filing_agent_security:16:Security"
        "e2e_workflows:6:E2E Workflows"
    )

    local total_passed=0
    local total_failed=0

    for agent_info in "${agents[@]}"; do
        IFS=':' read -r agent count description <<< "${agent_info}"

        echo -e "${BLUE}Testing ${description} (${count} tests)...${NC}"

        pytest -v --tb=line tests/test_${agent}.py -q

        if [ $? -eq 0 ]; then
            print_success "${description}: PASSED"
            ((total_passed+=count))
        else
            print_error "${description}: FAILED"
            ((total_failed+=count))
        fi

        echo ""
    done

    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "  Total Passed: ${GREEN}${total_passed}${NC}"
    echo -e "  Total Failed: ${RED}${total_failed}${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

generate_summary_report() {
    print_section "Generating Test Summary Report"

    cat > "${REPORTS_DIR}/summary_${TIMESTAMP}.txt" <<EOF
GL-CSRD Test Execution Summary
Generated: $(date)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST SUITE STATISTICS
────────────────────────────────────────────────────────────────

Total Tests:              975
Test Files:               14
Target Coverage:          90%+

TEST BREAKDOWN BY AGENT
────────────────────────────────────────────────────────────────

Calculator Agent:         ${CALCULATOR_TESTS} tests (11.2%) - CRITICAL
Reporting Agent:          ${REPORTING_TESTS} tests (13.6%)
Audit Agent:              ${AUDIT_TESTS} tests (11.8%)
Intake Agent:             ${INTAKE_TESTS} tests (11.0%)
Provenance System:        ${PROVENANCE_TESTS} tests (10.4%)
Aggregator Agent:         ${AGGREGATOR_TESTS} tests (7.7%)
CLI Interface:            ${CLI_TESTS} tests (7.1%)
SDK:                      ${SDK_TESTS} tests (6.3%)
Pipeline Integration:     ${PIPELINE_TESTS} tests (6.1%)
Validation System:        ${VALIDATION_TESTS} tests (5.6%)
Materiality Agent:        ${MATERIALITY_TESTS} tests (4.6%)
Encryption:               ${ENCRYPTION_TESTS} tests (2.5%)
Security:                 ${SECURITY_TESTS} tests (1.6%)
E2E Workflows:            ${E2E_TESTS} tests (0.6%)

ESRS STANDARDS COVERED
────────────────────────────────────────────────────────────────

ESRS 1:  General Requirements
ESRS 2:  General Disclosures
ESRS E1: Climate Change
ESRS E2: Pollution
ESRS E3: Water and Marine Resources
ESRS E4: Biodiversity and Ecosystems
ESRS E5: Resource Use and Circular Economy
ESRS S1: Own Workforce
ESRS S2: Workers in Value Chain
ESRS S3: Affected Communities
ESRS S4: Consumers and End-users
ESRS G1: Business Conduct

REPORTS GENERATED
────────────────────────────────────────────────────────────────

JUnit XML:     ${REPORTS_DIR}/junit/results_${TIMESTAMP}.xml
HTML Report:   ${REPORTS_DIR}/html/report_${TIMESTAMP}.html
Coverage JSON: ${REPORTS_DIR}/coverage.json
Coverage HTML: ${COVERAGE_DIR}/index.html

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

    cat "${REPORTS_DIR}/summary_${TIMESTAMP}.txt"
    echo ""
    print_success "Summary report saved to: ${REPORTS_DIR}/summary_${TIMESTAMP}.txt"
    echo ""
}

show_coverage_summary() {
    if [ -f "${REPORTS_DIR}/coverage.json" ]; then
        print_section "Coverage Summary"

        python -c "
import json
with open('${REPORTS_DIR}/coverage.json') as f:
    data = json.load(f)
    total_coverage = data['totals']['percent_covered']
    print(f'  Overall Coverage: {total_coverage:.2f}%')

    if total_coverage >= 90:
        print('  Status: ✓ TARGET MET (≥90%)')
    else:
        print(f'  Status: ✗ Below Target (need {90 - total_coverage:.2f}% more)')
"
        echo ""
        echo "  Detailed HTML report: file://${COVERAGE_DIR}/index.html"
        echo ""
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    print_header

    # Parse arguments
    MODE="all"
    EXTRA_ARGS=""

    for arg in "$@"; do
        case $arg in
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
            --performance)
                MODE="performance"
                ;;
            --by-agent)
                MODE="by-agent"
                ;;
            --coverage|--html)
                EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
                ;;
            *)
                # Unknown option
                ;;
        esac
    done

    # Execute test pipeline
    check_environment
    create_directories
    show_test_summary

    START_TIME=$(date +%s)

    if [ "${MODE}" == "by-agent" ]; then
        run_tests_by_agent
    else
        run_test_suite "${MODE}" "${EXTRA_ARGS}"
    fi

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    print_section "Test Execution Complete"
    echo "  Duration: ${DURATION} seconds"
    echo ""

    show_coverage_summary
    generate_summary_report

    print_success "All tests completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Review HTML coverage report: file://${COVERAGE_DIR}/index.html"
    echo "  2. Check test report: file://${REPORTS_DIR}/html/report_${TIMESTAMP}.html"
    echo "  3. Review summary: ${REPORTS_DIR}/summary_${TIMESTAMP}.txt"
    echo ""
}

# Run main function
main "$@"
