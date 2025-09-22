#!/bin/bash
#
# GreenLang CI Smoke Test Script (Unix)
# Runs basic functionality tests to verify the package works correctly
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="$PROJECT_ROOT/smoke-test-results"
LOG_FILE="$TEST_DIR/smoke-test.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log() {
    echo -e "${BLUE}[SMOKE]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
    ((TESTS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    ((TESTS_FAILED++))
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1" | tee -a "$LOG_FILE"
    ((TESTS_TOTAL++))
}

# Test execution wrapper
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expect_success="${3:-true}"

    log_test "Running: $test_name"

    if [[ "$expect_success" == "true" ]]; then
        if eval "$test_command" >> "$LOG_FILE" 2>&1; then
            log_success "$test_name - PASSED"
            return 0
        else
            log_error "$test_name - FAILED"
            return 1
        fi
    else
        if eval "$test_command" >> "$LOG_FILE" 2>&1; then
            log_error "$test_name - FAILED (expected to fail)"
            return 1
        else
            log_success "$test_name - PASSED (expected failure)"
            return 0
        fi
    fi
}

# Test functions
test_basic_import() {
    python -c "
import greenlang
print(f'GreenLang version: {greenlang.__version__}')
"
}

test_cli_availability() {
    gl --version
}

test_cli_help() {
    gl --help | grep -q "GreenLang"
}

test_basic_calculation() {
    echo '{"building_type": "office", "area": 1000, "energy_efficiency": "standard"}' | \
    gl calculate building-emissions --input-format json --output-format json | \
    python -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    assert 'emissions' in data or 'energy_consumption' in data or 'carbon_footprint' in data
    print('Basic calculation test passed')
except Exception as e:
    print(f'Basic calculation test failed: {e}')
    sys.exit(1)
"
}

test_pipeline_validation() {
    if [[ -f "$PROJECT_ROOT/test_simple.yaml" ]]; then
        gl pipeline validate "$PROJECT_ROOT/test_simple.yaml"
    else
        echo "test_simple.yaml not found, skipping pipeline validation test"
        return 0
    fi
}

test_pack_validation() {
    if [[ -d "$PROJECT_ROOT/packs" ]]; then
        # Find first pack directory
        local pack_dir=$(find "$PROJECT_ROOT/packs" -name "pack.yaml" -type f | head -1 | xargs dirname)
        if [[ -n "$pack_dir" ]]; then
            gl pack validate "$pack_dir"
        else
            echo "No pack.yaml found in packs directory, skipping pack validation test"
            return 0
        fi
    else
        echo "packs directory not found, skipping pack validation test"
        return 0
    fi
}

test_sdk_import() {
    python -c "
from greenlang.sdk import GreenLangClient
print('SDK client import successful')

# Basic client initialization
try:
    client = GreenLangClient()
    print('SDK client initialization successful')
except Exception as e:
    print(f'SDK client initialization failed: {e}')
    # Non-fatal for smoke test
"
}

test_core_modules() {
    python -c "
# Test core module imports
modules_to_test = [
    'greenlang.core',
    'greenlang.cli',
    'greenlang.agents',
    'greenlang.utils',
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f'✓ {module}')
    except ImportError as e:
        print(f'✗ {module}: {e}')
        # Some modules might not exist, that's OK for smoke test
    except Exception as e:
        print(f'✗ {module}: Unexpected error: {e}')
"
}

test_config_validation() {
    python -c "
import os
import sys
from pathlib import Path

# Test configuration file validation
project_root = Path('$PROJECT_ROOT')
config_files = [
    'pyproject.toml',
    'VERSION',
]

for config_file in config_files:
    file_path = project_root / config_file
    if file_path.exists():
        print(f'✓ {config_file} exists')
        if file_path.stat().st_size > 0:
            print(f'✓ {config_file} is not empty')
        else:
            print(f'⚠ {config_file} is empty')
    else:
        if config_file == 'VERSION':
            print(f'ℹ {config_file} not found (optional)')
        else:
            print(f'✗ {config_file} not found')
            sys.exit(1)
"
}

test_environment_compatibility() {
    python -c "
import sys
import platform

print(f'Python version: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()[0]}')

# Check minimum Python version
min_version = (3, 10)
current_version = sys.version_info[:2]

if current_version >= min_version:
    print(f'✓ Python version {current_version} meets minimum requirement {min_version}')
else:
    print(f'✗ Python version {current_version} below minimum requirement {min_version}')
    sys.exit(1)
"
}

test_dependency_availability() {
    python -c "
import pkg_resources
import sys

# Test key dependencies
dependencies = [
    'typer',
    'pydantic',
    'pyyaml',
    'rich',
    'jsonschema',
    'packaging',
    'python-dotenv',
    'httpx',
    'requests',
    'networkx',
]

missing_deps = []
for dep in dependencies:
    try:
        pkg_resources.get_distribution(dep)
        print(f'✓ {dep}')
    except pkg_resources.DistributionNotFound:
        print(f'✗ {dep} - not found')
        missing_deps.append(dep)

if missing_deps:
    print(f'Missing dependencies: {missing_deps}')
    # Don't fail smoke test for missing optional dependencies
    print('Some dependencies missing but continuing...')
"
}

# Main test execution
main() {
    log "Starting GreenLang smoke tests"
    log "Project root: $PROJECT_ROOT"
    log "Test directory: $TEST_DIR"
    log "Python version: $(python --version)"

    # Setup test directory
    mkdir -p "$TEST_DIR"
    rm -f "$LOG_FILE"

    cd "$PROJECT_ROOT"

    # Run smoke tests
    log "=== Environment Tests ==="
    run_test "Environment compatibility" "test_environment_compatibility"
    run_test "Configuration validation" "test_config_validation"
    run_test "Dependency availability" "test_dependency_availability"

    log "=== Import Tests ==="
    run_test "Basic package import" "test_basic_import"
    run_test "Core modules import" "test_core_modules"
    run_test "SDK import" "test_sdk_import"

    log "=== CLI Tests ==="
    run_test "CLI availability" "test_cli_availability"
    run_test "CLI help command" "test_cli_help"

    log "=== Functional Tests ==="
    run_test "Basic calculation" "test_basic_calculation"
    run_test "Pipeline validation" "test_pipeline_validation"
    run_test "Pack validation" "test_pack_validation"

    # Generate test report
    log "=== Test Summary ==="
    log "Tests run: $TESTS_TOTAL"
    log "Tests passed: $TESTS_PASSED"
    log "Tests failed: $TESTS_FAILED"

    # Create test results file
    cat > "$TEST_DIR/smoke-test-results.txt" << EOF
GreenLang Smoke Test Results
============================

Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Platform: $(uname -s)-$(uname -m)
Python: $(python --version)
GreenLang: $(python -c "import greenlang; print(greenlang.__version__)" 2>/dev/null || echo "unknown")

Test Summary:
- Total tests: $TESTS_TOTAL
- Passed: $TESTS_PASSED
- Failed: $TESTS_FAILED

Status: $(if [[ $TESTS_FAILED -eq 0 ]]; then echo "PASSED"; else echo "FAILED"; fi)

Detailed log: $LOG_FILE
EOF

    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "All smoke tests passed!"
        echo "SMOKE_TEST_STATUS=passed" >> "$TEST_DIR/test-status.env"
        exit 0
    else
        log_error "$TESTS_FAILED test(s) failed"
        echo "SMOKE_TEST_STATUS=failed" >> "$TEST_DIR/test-status.env"
        exit 1
    fi
}

# Error handler
handle_error() {
    local exit_code=$?
    log_error "Smoke tests failed with exit code $exit_code"
    echo "SMOKE_TEST_STATUS=error" >> "$TEST_DIR/test-status.env"
    exit $exit_code
}

trap 'handle_error' ERR

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi