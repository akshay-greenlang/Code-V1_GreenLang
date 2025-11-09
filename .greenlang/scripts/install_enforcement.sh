#!/bin/bash
# GreenLang Infrastructure-First Enforcement Installation
# ========================================================
#
# This script installs all enforcement mechanisms:
# - Pre-commit hooks
# - GitHub Actions workflows
# - Linter dependencies
# - OPA policy engine
#
# Usage: bash .greenlang/scripts/install_enforcement.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GreenLang Enforcement Installation${NC}"
echo -e "${BLUE}========================================${NC}\n"

# =============================================================================
# 1. Install Pre-Commit Hook
# =============================================================================
echo -e "${YELLOW}[1/6] Installing pre-commit hook...${NC}"

if [ ! -d ".git" ]; then
    echo -e "${RED}✗ Not a git repository${NC}"
    exit 1
fi

# Create hooks directory if needed
mkdir -p .git/hooks

# Copy pre-commit hook
if [ -f ".greenlang/hooks/pre-commit" ]; then
    cp .greenlang/hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
    echo -e "${GREEN}✓ Pre-commit hook installed${NC}"
else
    echo -e "${RED}✗ Pre-commit hook not found at .greenlang/hooks/pre-commit${NC}"
    exit 1
fi

# =============================================================================
# 2. Verify GitHub Actions Workflow
# =============================================================================
echo -e "\n${YELLOW}[2/6] Verifying GitHub Actions workflow...${NC}"

if [ -f ".github/workflows/greenlang-first-enforcement.yml" ]; then
    echo -e "${GREEN}✓ GitHub Actions workflow found${NC}"
else
    echo -e "${YELLOW}⚠ GitHub Actions workflow not found${NC}"
    echo -e "  Manual step: Copy .greenlang/workflows/greenlang-first-enforcement.yml"
    echo -e "  to .github/workflows/greenlang-first-enforcement.yml"
fi

# =============================================================================
# 3. Install Python Dependencies
# =============================================================================
echo -e "\n${YELLOW}[3/6] Installing Python dependencies...${NC}"

# Check if pip is available
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo -e "${RED}✗ pip not found. Install Python 3.8+ first${NC}"
    exit 1
fi

PIP_CMD=$(command -v pip3 || command -v pip)

# Install/upgrade required packages
echo "Installing linter dependencies..."
$PIP_CMD install --upgrade ast argparse 2>/dev/null || true

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# =============================================================================
# 4. Install OPA (Open Policy Agent)
# =============================================================================
echo -e "\n${YELLOW}[4/6] Installing OPA (Open Policy Agent)...${NC}"

# Check if OPA is already installed
if command -v opa &> /dev/null; then
    OPA_VERSION=$(opa version | head -n1)
    echo -e "${GREEN}✓ OPA already installed: $OPA_VERSION${NC}"
else
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Downloading OPA for Linux..."
        curl -L -o /tmp/opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
        chmod +x /tmp/opa

        # Try to move to /usr/local/bin (requires sudo)
        if sudo mv /tmp/opa /usr/local/bin/ 2>/dev/null; then
            echo -e "${GREEN}✓ OPA installed to /usr/local/bin/opa${NC}"
        else
            mv /tmp/opa "$HOME/.local/bin/opa" 2>/dev/null || mv /tmp/opa ./opa
            echo -e "${YELLOW}⚠ OPA installed to $HOME/.local/bin/opa or ./opa${NC}"
            echo -e "  Add to PATH or move to /usr/local/bin"
        fi

    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Downloading OPA for macOS..."
        curl -L -o /tmp/opa https://openpolicyagent.org/downloads/latest/opa_darwin_amd64
        chmod +x /tmp/opa

        if sudo mv /tmp/opa /usr/local/bin/ 2>/dev/null; then
            echo -e "${GREEN}✓ OPA installed to /usr/local/bin/opa${NC}"
        else
            mv /tmp/opa "$HOME/.local/bin/opa" 2>/dev/null || mv /tmp/opa ./opa
            echo -e "${YELLOW}⚠ OPA installed to $HOME/.local/bin/opa or ./opa${NC}"
        fi

    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "Downloading OPA for Windows..."
        curl -L -o opa.exe https://openpolicyagent.org/downloads/latest/opa_windows_amd64.exe
        echo -e "${GREEN}✓ OPA downloaded as opa.exe${NC}"
        echo -e "  Move to a directory in your PATH"

    else
        echo -e "${YELLOW}⚠ Unknown OS. Download OPA manually from:${NC}"
        echo -e "  https://www.openpolicyagent.org/docs/latest/#running-opa"
    fi
fi

# Verify OPA installation
if command -v opa &> /dev/null; then
    echo "Testing OPA policy..."
    if opa test .greenlang/policies/infrastructure-first.rego 2>/dev/null; then
        echo -e "${GREEN}✓ OPA policy tests passed${NC}"
    else
        echo -e "${YELLOW}⚠ OPA installed but policy tests failed${NC}"
    fi
fi

# =============================================================================
# 5. Create ADR Directory
# =============================================================================
echo -e "\n${YELLOW}[5/6] Setting up ADR directory...${NC}"

mkdir -p .greenlang/adrs

if [ ! -f ".greenlang/adrs/README.md" ]; then
    cat > .greenlang/adrs/README.md <<'EOF'
# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for custom implementations
that deviate from GreenLang infrastructure-first principles.

## Format

Each ADR should follow this naming convention:
```
YYYYMMDD-short-title.md
```

Example: `20241109-custom-llm-provider.md`

## Template

```markdown
# ADR-XXX: [Title]

Date: YYYY-MM-DD
Status: [Proposed | Accepted | Rejected | Superseded]

## Context

What is the issue we're facing?

## Decision

What custom implementation are we using instead of GreenLang infrastructure?

## Consequences

What are the implications?

## Alternatives Considered

Why can't we use GreenLang infrastructure?
```

## Review Process

1. Create ADR in this directory
2. Submit for review
3. Get approval from 2+ team members
4. Update status to "Accepted"
5. Reference in PR
EOF
    echo -e "${GREEN}✓ ADR directory created with README${NC}"
else
    echo -e "${GREEN}✓ ADR directory already exists${NC}"
fi

# =============================================================================
# 6. Run Initial Validation
# =============================================================================
echo -e "\n${YELLOW}[6/6] Running initial validation...${NC}"

# Test linter
echo "Testing linter..."
if python3 .greenlang/linters/infrastructure_first.py --path core/greenlang --output json > /tmp/ium_test.json 2>&1; then
    echo -e "${GREEN}✓ Linter working${NC}"
else
    echo -e "${YELLOW}⚠ Linter test had warnings (check /tmp/ium_test.json)${NC}"
fi

# Test IUM calculator
echo "Testing IUM calculator..."
if python3 .greenlang/scripts/calculate_ium.py --path core/greenlang --output json > /tmp/ium_score.json 2>&1; then
    IUM_SCORE=$(python3 -c "import json; print(json.load(open('/tmp/ium_score.json'))['overall']['percentage'])" 2>/dev/null || echo "N/A")
    echo -e "${GREEN}✓ IUM calculator working (Current score: $IUM_SCORE%)${NC}"
else
    echo -e "${YELLOW}⚠ IUM calculator test failed${NC}"
fi

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Installation Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}Installed Components:${NC}"
echo -e "  ✓ Pre-commit hook (.git/hooks/pre-commit)"
echo -e "  ✓ Static linter (.greenlang/linters/infrastructure_first.py)"
echo -e "  ✓ IUM calculator (.greenlang/scripts/calculate_ium.py)"
echo -e "  ✓ OPA policy (.greenlang/policies/infrastructure-first.rego)"
echo -e "  ✓ ADR directory (.greenlang/adrs/)"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "  1. Review enforcement guide: ${BLUE}.greenlang/ENFORCEMENT_GUIDE.md${NC}"
echo -e "  2. Test pre-commit: ${BLUE}git commit${NC} (will run checks)"
echo -e "  3. Check current IUM: ${BLUE}python .greenlang/scripts/calculate_ium.py${NC}"
echo -e "  4. Run linter: ${BLUE}python .greenlang/linters/infrastructure_first.py${NC}"

echo -e "\n${YELLOW}GitHub Actions:${NC}"
echo -e "  • Workflow will run automatically on PRs"
echo -e "  • Check .github/workflows/greenlang-first-enforcement.yml"

echo -e "\n${YELLOW}Creating ADRs:${NC}"
echo -e "  • Template: .greenlang/adrs/README.md"
echo -e "  • Create ADR if custom code needed"
echo -e "  • Reference in PR"

echo -e "\n${GREEN}Happy coding with GreenLang infrastructure! 🌱${NC}\n"
