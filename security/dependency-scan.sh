#!/bin/bash
# GreenLang Dependency Vulnerability Scanner
# Phase 3 Security Hardening
# ==========================================
# This script runs pip-audit to scan for known CVEs in dependencies

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$PROJECT_ROOT/security"
REPORT_FILE="$REPORT_DIR/pip-audit-report.txt"
JSON_REPORT="$REPORT_DIR/pip-audit-report.json"
REQUIREMENTS_FILE="$PROJECT_ROOT/pyproject.toml"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GreenLang Dependency Security Scan${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if pip-audit is installed
if ! command -v pip-audit &> /dev/null; then
    echo -e "${RED}ERROR: pip-audit is not installed${NC}"
    echo "Install with: pip install pip-audit"
    exit 1
fi

echo -e "${BLUE}Scanning dependencies for known vulnerabilities...${NC}"
echo ""

# Run pip-audit and capture output
echo "Project: GreenLang CLI"
echo "Scan Date: $(date)"
echo "pip-audit Version: $(pip-audit --version)"
echo ""

# Create report directory if it doesn't exist
mkdir -p "$REPORT_DIR"

# Run pip-audit with multiple output formats
EXIT_CODE=0

# Text report
echo -e "${BLUE}Running pip-audit (text report)...${NC}"
if pip-audit --desc --requirement "$REQUIREMENTS_FILE" > "$REPORT_FILE" 2>&1; then
    echo -e "${GREEN}✓ No vulnerabilities found!${NC}"
else
    EXIT_CODE=$?
    echo -e "${YELLOW}⚠ Vulnerabilities detected - see report${NC}"
fi

# JSON report for CI/CD integration
echo -e "${BLUE}Running pip-audit (JSON report)...${NC}"
pip-audit --desc --requirement "$REQUIREMENTS_FILE" --format json > "$JSON_REPORT" 2>&1 || true

# Display summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Scan Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Parse and display results
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Status: PASSED${NC}"
    echo -e "${GREEN}✓ No known vulnerabilities found in dependencies${NC}"
else
    echo -e "${YELLOW}⚠ Status: VULNERABILITIES FOUND${NC}"
    echo ""
    echo "Detailed reports saved to:"
    echo "  - Text: $REPORT_FILE"
    echo "  - JSON: $JSON_REPORT"
    echo ""

    # Show vulnerability count if available
    if [ -f "$JSON_REPORT" ]; then
        VULN_COUNT=$(python -c "import json; data=json.load(open('$JSON_REPORT')); print(len(data.get('dependencies', [])))" 2>/dev/null || echo "unknown")
        echo -e "${YELLOW}Found $VULN_COUNT vulnerable dependencies${NC}"
    fi

    echo ""
    echo "Review the reports and update dependencies as needed:"
    echo "  1. Check $REPORT_FILE for details"
    echo "  2. Update vulnerable packages in pyproject.toml"
    echo "  3. Run: pip install -e .[all]"
    echo "  4. Re-run this scan to verify"
fi

echo ""
echo -e "${BLUE}========================================${NC}"

# Additional checks
echo ""
echo -e "${BLUE}Additional Security Checks:${NC}"
echo ""

# Check for outdated packages
echo "Checking for outdated packages..."
pip list --outdated --format=columns || true

echo ""
echo -e "${BLUE}Scan complete!${NC}"
echo ""

exit $EXIT_CODE
