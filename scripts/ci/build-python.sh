#!/bin/bash
#
# GreenLang CI Build Script (Unix)
# Builds Python packages (wheels and source distribution) for CI
#

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/dist"
LOG_FILE="$PROJECT_ROOT/build.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[BUILD]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Error handler
handle_error() {
    local exit_code=$?
    log_error "Build failed with exit code $exit_code"
    log_error "Check $LOG_FILE for details"
    exit $exit_code
}

trap 'handle_error' ERR

# Main build function
main() {
    log "Starting GreenLang Python package build"
    log "Project root: $PROJECT_ROOT"
    log "Build directory: $BUILD_DIR"
    log "Python version: $(python --version)"
    log "Pip version: $(pip --version)"

    cd "$PROJECT_ROOT"

    # Clean previous builds
    log "Cleaning previous build artifacts..."
    rm -rf "$BUILD_DIR" build/ *.egg-info/
    mkdir -p "$BUILD_DIR"

    # Verify project structure
    log "Verifying project structure..."
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found in project root"
        exit 1
    fi

    if [[ ! -d "greenlang" ]]; then
        log_error "greenlang package directory not found"
        exit 1
    fi

    # Extract version from pyproject.toml
    VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
    log "Building version: $VERSION"

    # Install build dependencies
    log "Installing build dependencies..."
    python -m pip install --upgrade pip
    pip install build wheel setuptools

    # Validate package configuration
    log "Validating package configuration..."
    python -c "
import toml
import sys
try:
    with open('pyproject.toml', 'r') as f:
        config = toml.load(f)

    # Validate required fields
    project = config.get('project', {})
    required_fields = ['name', 'version', 'description']

    for field in required_fields:
        if field not in project:
            print(f'ERROR: Missing required field: project.{field}')
            sys.exit(1)

    print('Package configuration is valid')
except Exception as e:
    print(f'ERROR: Invalid pyproject.toml: {e}')
    sys.exit(1)
"

    # Build source distribution
    log "Building source distribution..."
    python -m build --sdist --outdir "$BUILD_DIR"

    # Build wheel
    log "Building wheel..."
    python -m build --wheel --outdir "$BUILD_DIR"

    # Verify build artifacts
    log "Verifying build artifacts..."
    local wheel_count=$(find "$BUILD_DIR" -name "*.whl" | wc -l)
    local sdist_count=$(find "$BUILD_DIR" -name "*.tar.gz" | wc -l)

    if [[ $wheel_count -eq 0 ]]; then
        log_error "No wheel files found in build output"
        exit 1
    fi

    if [[ $sdist_count -eq 0 ]]; then
        log_error "No source distribution files found in build output"
        exit 1
    fi

    log_success "Built $wheel_count wheel(s) and $sdist_count source distribution(s)"

    # List build artifacts
    log "Build artifacts:"
    find "$BUILD_DIR" -type f | while read -r file; do
        local size=$(du -h "$file" | cut -f1)
        log "  $(basename "$file") ($size)"
    done

    # Basic package validation
    log "Performing basic package validation..."

    # Check wheel contents
    for wheel in "$BUILD_DIR"/*.whl; do
        log "Validating wheel: $(basename "$wheel")"
        python -m zipfile -l "$wheel" | grep -q "greenlang/" || {
            log_error "Wheel does not contain greenlang package"
            exit 1
        }
    done

    # Check source distribution contents
    for sdist in "$BUILD_DIR"/*.tar.gz; do
        log "Validating source distribution: $(basename "$sdist")"
        tar -tzf "$sdist" | grep -q "greenlang/" || {
            log_error "Source distribution does not contain greenlang package"
            exit 1
        }
    done

    # Test wheel installation in temporary environment
    log "Testing wheel installation..."
    local temp_venv=$(mktemp -d)
    python -m venv "$temp_venv"
    source "$temp_venv/bin/activate"

    pip install --upgrade pip
    pip install "$BUILD_DIR"/*.whl

    # Basic import test
    python -c "
import greenlang
print(f'Successfully imported GreenLang {greenlang.__version__}')

# Test CLI availability
import subprocess
result = subprocess.run(['gl', '--version'], capture_output=True, text=True)
if result.returncode == 0:
    print(f'CLI available: {result.stdout.strip()}')
else:
    print('CLI not available')
    exit(1)
"

    deactivate
    rm -rf "$temp_venv"

    log_success "Wheel installation test passed"

    # Generate build metadata
    log "Generating build metadata..."
    cat > "$BUILD_DIR/build-metadata.json" << EOF
{
    "build_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "$VERSION",
    "python_version": "$(python --version | cut -d' ' -f2)",
    "platform": "$(uname -s)-$(uname -m)",
    "build_script": "$0",
    "git_sha": "${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}",
    "artifacts": [
        $(find "$BUILD_DIR" -name "*.whl" -o -name "*.tar.gz" | sed 's/.*\///' | sed 's/^/        "/' | sed 's/$/"/' | paste -sd,)
    ]
}
EOF

    log_success "Build completed successfully!"
    log "Build summary:"
    log "  Version: $VERSION"
    log "  Artifacts: $(find "$BUILD_DIR" -name "*.whl" -o -name "*.tar.gz" | wc -l) files"
    log "  Output directory: $BUILD_DIR"
    log "  Log file: $LOG_FILE"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi