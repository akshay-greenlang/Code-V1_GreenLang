#!/bin/bash
# ==============================================================================
# GreenLang Release Smoke Test Script
# ==============================================================================
#
# Run after PyPI publish to verify the release works correctly.
#
# Usage:
#   ./scripts/smoke_test.sh                    # Test latest version
#   ./scripts/smoke_test.sh 0.3.0              # Test specific version
#   ./scripts/smoke_test.sh --from-testpypi    # Test from TestPyPI
#   ./scripts/smoke_test.sh --local            # Test local installation
#
# Environment Variables:
#   GL_SMOKE_TIMEOUT    - Timeout for tests (default: 300s)
#   GL_SMOKE_STRICT     - Set to "1" for strict mode
#   GL_SMOKE_KEEP_ENV   - Set to "1" to keep virtual environment after tests
#
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${TMPDIR:-/tmp}/gl-smoke-test-$$"
TIMEOUT="${GL_SMOKE_TIMEOUT:-300}"
STRICT="${GL_SMOKE_STRICT:-0}"
KEEP_ENV="${GL_SMOKE_KEEP_ENV:-0}"
LOG_FILE="${TMPDIR:-/tmp}/gl-smoke-test-$$.log"

# Parse arguments
VERSION=""
SOURCE="pypi"
LOCAL_INSTALL=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-testpypi)
            SOURCE="testpypi"
            shift
            ;;
        --local)
            LOCAL_INSTALL=1
            shift
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --keep-env)
            KEEP_ENV=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [VERSION] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  VERSION             Version to test (e.g., 0.3.0)"
            echo ""
            echo "Options:"
            echo "  --from-testpypi     Install from TestPyPI instead of PyPI"
            echo "  --local             Test local installation (editable)"
            echo "  --strict            Fail on warnings"
            echo "  --keep-env          Keep virtual environment after tests"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  GL_SMOKE_TIMEOUT    Test timeout in seconds (default: 300)"
            echo "  GL_SMOKE_STRICT     Set to '1' for strict mode"
            echo "  GL_SMOKE_KEEP_ENV   Set to '1' to keep environment"
            exit 0
            ;;
        *)
            if [[ -z "$VERSION" ]]; then
                VERSION="$1"
            else
                echo -e "${RED}Error: Unknown argument: $1${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Functions
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[OK]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[FAIL]${NC} $1"
}

cleanup() {
    if [[ "$KEEP_ENV" == "0" ]] && [[ -d "$VENV_DIR" ]]; then
        log_info "Cleaning up virtual environment..."
        rm -rf "$VENV_DIR"
    else
        log_info "Keeping virtual environment at: $VENV_DIR"
    fi
}

trap cleanup EXIT

# ==============================================================================
# Main Script
# ==============================================================================

echo ""
log "========================================================================"
log "          GreenLang Release Smoke Test"
log "========================================================================"
echo ""

# Get version from pyproject.toml if not specified
if [[ -z "$VERSION" ]] && [[ "$LOCAL_INSTALL" == "0" ]]; then
    if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        VERSION=$(grep '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/')
    fi
fi

# Display configuration
log_info "Configuration:"
log_info "  Version:        ${VERSION:-latest}"
log_info "  Source:         ${SOURCE}"
log_info "  Local Install:  ${LOCAL_INSTALL}"
log_info "  Strict Mode:    ${STRICT}"
log_info "  Timeout:        ${TIMEOUT}s"
log_info "  Log File:       ${LOG_FILE}"
echo ""

# Step 1: Create virtual environment
log "========================================================================"
log "Step 1: Creating Virtual Environment"
log "========================================================================"

log_info "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log_success "Virtual environment created and activated"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip >> "$LOG_FILE" 2>&1
log_success "pip upgraded"

# Step 2: Install GreenLang
log ""
log "========================================================================"
log "Step 2: Installing GreenLang"
log "========================================================================"

if [[ "$LOCAL_INSTALL" == "1" ]]; then
    log_info "Installing from local source (editable)..."
    pip install -e "$PROJECT_ROOT[test]" >> "$LOG_FILE" 2>&1
    log_success "Local installation complete"
elif [[ "$SOURCE" == "testpypi" ]]; then
    log_info "Installing from TestPyPI..."
    if [[ -n "$VERSION" ]]; then
        pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "greenlang-cli==$VERSION" >> "$LOG_FILE" 2>&1
    else
        pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            greenlang-cli >> "$LOG_FILE" 2>&1
    fi
    log_success "TestPyPI installation complete"
else
    log_info "Installing from PyPI..."
    if [[ -n "$VERSION" ]]; then
        pip install "greenlang-cli==$VERSION" >> "$LOG_FILE" 2>&1
    else
        pip install greenlang-cli >> "$LOG_FILE" 2>&1
    fi
    log_success "PyPI installation complete"
fi

# Install test dependencies
log_info "Installing test dependencies..."
pip install pytest pytest-timeout >> "$LOG_FILE" 2>&1
log_success "Test dependencies installed"

# Step 3: Verify Installation
log ""
log "========================================================================"
log "Step 3: Verifying Installation"
log "========================================================================"

# Check CLI is accessible
log_info "Checking CLI availability..."
if command -v gl &> /dev/null; then
    log_success "gl command is available"
else
    log_error "gl command not found in PATH"
    exit 1
fi

# Check version
log_info "Checking version..."
INSTALLED_VERSION=$(gl --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [[ -n "$INSTALLED_VERSION" ]]; then
    log_success "Installed version: $INSTALLED_VERSION"
    if [[ -n "$VERSION" ]] && [[ "$INSTALLED_VERSION" != "$VERSION" ]]; then
        log_warning "Version mismatch: expected $VERSION, got $INSTALLED_VERSION"
        if [[ "$STRICT" == "1" ]]; then
            exit 1
        fi
    fi
else
    log_error "Could not determine installed version"
    if [[ "$STRICT" == "1" ]]; then
        exit 1
    fi
fi

# Step 4: Run Basic CLI Tests
log ""
log "========================================================================"
log "Step 4: Running Basic CLI Tests"
log "========================================================================"

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_cmd="$2"

    log_info "Testing: $test_name"
    if eval "$test_cmd" >> "$LOG_FILE" 2>&1; then
        log_success "$test_name"
        ((TESTS_PASSED++))
    else
        log_error "$test_name"
        ((TESTS_FAILED++))
        if [[ "$STRICT" == "1" ]]; then
            exit 1
        fi
    fi
}

run_test "gl --version" "gl --version"
run_test "gl --help" "gl --help"
run_test "gl doctor" "gl doctor"
run_test "gl version" "gl version"
run_test "gl pack --help" "gl pack --help"
run_test "gl pack list" "gl pack list"

# Step 5: Run Python Import Tests
log ""
log "========================================================================"
log "Step 5: Running Python Import Tests"
log "========================================================================"

run_test "Import greenlang" "python -c 'import greenlang; print(greenlang.__version__)'"
run_test "Import BaseAgent" "python -c 'from greenlang.agents.base import BaseAgent; print(BaseAgent)'"
run_test "Import PackLoader" "python -c 'from greenlang.ecosystem.packs.loader import PackLoader; print(PackLoader)'"
run_test "Import CLI" "python -c 'from greenlang.cli.main import app; print(app)'"

# Step 6: Run Pytest Smoke Tests (if available)
log ""
log "========================================================================"
log "Step 6: Running Pytest Smoke Tests"
log "========================================================================"

SMOKE_TEST_FILE="$PROJECT_ROOT/tests/smoke/test_release_smoke.py"
if [[ -f "$SMOKE_TEST_FILE" ]]; then
    log_info "Running pytest smoke tests..."

    export GL_EXPECTED_VERSION="${VERSION:-$INSTALLED_VERSION}"
    export GL_SMOKE_STRICT="$STRICT"

    PYTEST_ARGS="-v --tb=short --timeout=$TIMEOUT"
    if [[ "$STRICT" == "1" ]]; then
        PYTEST_ARGS="$PYTEST_ARGS -x"
    fi

    if python -m pytest $PYTEST_ARGS "$SMOKE_TEST_FILE" >> "$LOG_FILE" 2>&1; then
        log_success "Pytest smoke tests passed"
        ((TESTS_PASSED++))
    else
        log_error "Pytest smoke tests failed"
        ((TESTS_FAILED++))
        # Show last 50 lines of log on failure
        log_warning "Last 50 lines of log:"
        tail -50 "$LOG_FILE"
    fi
else
    log_warning "Smoke test file not found: $SMOKE_TEST_FILE"
fi

# Step 7: Summary
log ""
log "========================================================================"
log "                        Test Summary"
log "========================================================================"
echo ""
log_info "Tests Passed: $TESTS_PASSED"
log_info "Tests Failed: $TESTS_FAILED"
log_info "Log File:     $LOG_FILE"
echo ""

if [[ "$TESTS_FAILED" -gt 0 ]]; then
    log_error "SMOKE TESTS FAILED"
    log_warning "Check the log file for details: $LOG_FILE"
    exit 1
else
    log_success "ALL SMOKE TESTS PASSED"
    log ""
    log_success "GreenLang ${INSTALLED_VERSION:-$VERSION} is ready for use!"
fi

log "========================================================================"
exit 0
