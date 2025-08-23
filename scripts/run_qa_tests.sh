#!/bin/bash
# GreenLang Comprehensive QA Test Suite
# Automated testing script for production readiness

set -e  # Exit on first error
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "PASS")
            echo -e "${GREEN}[PASS]${NC} $message"
            ((PASSED_TESTS++))
            ;;
        "FAIL")
            echo -e "${RED}[FAIL]${NC} $message"
            ((FAILED_TESTS++))
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message"
            ((WARNINGS++))
            ;;
        "SECTION")
            echo -e "\n${BLUE}========================================${NC}"
            echo -e "${BLUE}$message${NC}"
            echo -e "${BLUE}========================================${NC}"
            ;;
    esac
    ((TOTAL_TESTS++))
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start time
START_TIME=$(date +%s)

print_status "SECTION" "GreenLang v0.0.1 - QA Test Suite"
echo "Date: $(date)"
echo "Platform: $(uname -s)"
echo "Python: $(python --version 2>&1)"
echo ""

# ============================================
# 1. ENVIRONMENT CHECKS
# ============================================
print_status "SECTION" "1. Environment Verification"

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" == "3.8" || "$PYTHON_VERSION" == "3.9" || "$PYTHON_VERSION" == "3.10" || "$PYTHON_VERSION" == "3.11" || "$PYTHON_VERSION" == "3.12" ]]; then
    print_status "PASS" "Python version $PYTHON_VERSION is supported"
else
    print_status "FAIL" "Python version $PYTHON_VERSION is not supported"
fi

# Check required tools
for tool in pip pytest mypy ruff black; do
    if command_exists $tool; then
        print_status "PASS" "$tool is installed"
    else
        print_status "FAIL" "$tool is not installed"
    fi
done

# ============================================
# 2. DEPENDENCY CHECKS
# ============================================
print_status "SECTION" "2. Dependency Installation"

# Install dependencies
print_status "INFO" "Installing dependencies..."
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    print_status "PASS" "Dependencies installed successfully"
else
    print_status "FAIL" "Failed to install dependencies"
    exit 1
fi

# Install package in development mode
pip install -q -e .
if [ $? -eq 0 ]; then
    print_status "PASS" "GreenLang installed in development mode"
else
    print_status "FAIL" "Failed to install GreenLang"
    exit 1
fi

# ============================================
# 3. SECURITY SCANNING
# ============================================
print_status "SECTION" "3. Security Scanning"

# Install security tools if not present
pip install -q pip-audit safety bandit

# Run pip-audit
print_status "INFO" "Running pip-audit for vulnerability scanning..."
pip-audit --desc > security_audit.txt 2>&1
if [ $? -eq 0 ]; then
    print_status "PASS" "No dependency vulnerabilities found"
else
    print_status "WARN" "Dependency vulnerabilities detected (see security_audit.txt)"
fi

# Run bandit
print_status "INFO" "Running bandit security linter..."
bandit -r greenlang/ -ll -f txt > bandit_report.txt 2>&1
if grep -q "No issues identified" bandit_report.txt; then
    print_status "PASS" "No security issues in code"
else
    print_status "WARN" "Security issues found (see bandit_report.txt)"
fi

# ============================================
# 4. CODE QUALITY CHECKS
# ============================================
print_status "SECTION" "4. Code Quality Analysis"

# Type checking with mypy
print_status "INFO" "Running mypy type checker..."
mypy greenlang/ --strict --ignore-missing-imports > mypy_report.txt 2>&1
if [ $? -eq 0 ]; then
    print_status "PASS" "Type checking passed (strict mode)"
else
    print_status "WARN" "Type checking issues found (see mypy_report.txt)"
fi

# Linting with ruff
print_status "INFO" "Running ruff linter..."
ruff check greenlang/ tests/ --statistics > ruff_report.txt 2>&1
if [ $? -eq 0 ]; then
    print_status "PASS" "Code linting passed"
else
    print_status "WARN" "Linting issues found (see ruff_report.txt)"
fi

# Code formatting check
print_status "INFO" "Checking code formatting with black..."
black --check greenlang/ tests/ > black_report.txt 2>&1
if [ $? -eq 0 ]; then
    print_status "PASS" "Code formatting is correct"
else
    print_status "WARN" "Code formatting issues (run: black greenlang/ tests/)"
fi

# ============================================
# 5. UNIT TESTS
# ============================================
print_status "SECTION" "5. Unit Tests"

print_status "INFO" "Running unit tests..."
pytest tests/unit/ -v --tb=short --timeout=30 -q > unit_test_results.txt 2>&1
if [ $? -eq 0 ]; then
    print_status "PASS" "All unit tests passed"
else
    print_status "FAIL" "Unit tests failed (see unit_test_results.txt)"
fi

# ============================================
# 6. INTEGRATION TESTS
# ============================================
print_status "SECTION" "6. Integration Tests"

print_status "INFO" "Running integration tests..."
pytest tests/integration/ -v --tb=short --timeout=60 -q > integration_test_results.txt 2>&1
if [ $? -eq 0 ]; then
    print_status "PASS" "All integration tests passed"
else
    print_status "FAIL" "Integration tests failed (see integration_test_results.txt)"
fi

# ============================================
# 7. PROPERTY-BASED TESTS
# ============================================
print_status "SECTION" "7. Property-Based Tests"

if [ -d "tests/property" ]; then
    print_status "INFO" "Running property-based tests..."
    pytest tests/property/ -v --tb=short -q > property_test_results.txt 2>&1
    if [ $? -eq 0 ]; then
        print_status "PASS" "Property tests passed"
    else
        print_status "WARN" "Property tests failed (see property_test_results.txt)"
    fi
else
    print_status "INFO" "No property tests found"
fi

# ============================================
# 8. CLI TESTING
# ============================================
print_status "SECTION" "8. CLI Command Testing"

# Test basic CLI commands
CLI_COMMANDS=(
    "greenlang --version"
    "greenlang --help"
    "greenlang agents"
)

for cmd in "${CLI_COMMANDS[@]}"; do
    print_status "INFO" "Testing: $cmd"
    eval $cmd > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_status "PASS" "Command: $cmd"
    else
        print_status "FAIL" "Command failed: $cmd"
    fi
done

# ============================================
# 9. PERFORMANCE TESTS
# ============================================
print_status "SECTION" "9. Performance Benchmarks"

if [ -d "tests/performance" ]; then
    print_status "INFO" "Running performance benchmarks..."
    pytest tests/performance/ --benchmark-only --benchmark-autosave -q > performance_results.txt 2>&1
    if [ $? -eq 0 ]; then
        print_status "PASS" "Performance benchmarks completed"
    else
        print_status "WARN" "Performance tests failed (see performance_results.txt)"
    fi
else
    print_status "INFO" "No performance tests found"
fi

# ============================================
# 10. TEST COVERAGE
# ============================================
print_status "SECTION" "10. Test Coverage Analysis"

print_status "INFO" "Calculating test coverage..."
pytest --cov=greenlang --cov-report=term --cov-report=html --cov-fail-under=85 -q > coverage_report.txt 2>&1
if [ $? -eq 0 ]; then
    print_status "PASS" "Test coverage ≥ 85%"
    echo "Coverage report generated in htmlcov/"
else
    print_status "FAIL" "Test coverage < 85% (see coverage_report.txt)"
fi

# ============================================
# 11. DOCUMENTATION VALIDATION
# ============================================
print_status "SECTION" "11. Documentation Validation"

# Check if key documentation files exist
DOCS=(
    "README.md"
    "GREENLANG_DOCUMENTATION.md"
    "requirements.txt"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        print_status "PASS" "Documentation: $doc exists"
    else
        print_status "FAIL" "Missing documentation: $doc"
    fi
done

# ============================================
# 12. JSON SCHEMA VALIDATION
# ============================================
print_status "SECTION" "12. Data Schema Validation"

# Check if schema files exist
if [ -d "schemas" ]; then
    for schema in schemas/*.json; do
        if [ -f "$schema" ]; then
            print_status "INFO" "Validating schema: $(basename $schema)"
            python -m json.tool "$schema" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                print_status "PASS" "Valid JSON schema: $(basename $schema)"
            else
                print_status "FAIL" "Invalid JSON schema: $(basename $schema)"
            fi
        fi
    done
else
    print_status "WARN" "No schema directory found"
fi

# ============================================
# FINAL REPORT
# ============================================
print_status "SECTION" "QA Test Suite Summary"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "Test Execution Time: ${DURATION} seconds"
echo "Total Tests Run: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo ""

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Pass Rate: ${PASS_RATE}%"
fi

# Generate detailed report
REPORT_FILE="qa_report_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "GreenLang QA Report"
    echo "==================="
    echo "Date: $(date)"
    echo "Duration: ${DURATION} seconds"
    echo ""
    echo "Results Summary:"
    echo "- Total Tests: $TOTAL_TESTS"
    echo "- Passed: $PASSED_TESTS"
    echo "- Failed: $FAILED_TESTS"
    echo "- Warnings: $WARNINGS"
    echo "- Pass Rate: ${PASS_RATE}%"
    echo ""
    echo "Generated Reports:"
    ls -la *.txt 2>/dev/null || echo "No report files generated"
} > "$REPORT_FILE"

echo ""
echo "Detailed report saved to: $REPORT_FILE"

# Exit code based on failures
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✅ QA SUITE PASSED - Ready for production${NC}"
    exit 0
else
    echo -e "\n${RED}❌ QA SUITE FAILED - $FAILED_TESTS tests failed${NC}"
    echo "Please review the detailed reports and fix issues before release."
    exit 1
fi