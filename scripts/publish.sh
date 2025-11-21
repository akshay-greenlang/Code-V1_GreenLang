#!/usr/bin/env bash
#
# Manual publishing script for GreenLang to PyPI
# Use this for manual releases when GitHub Actions is not available
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="greenlang-cli"
VERSION="0.3.0"

echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   GreenLang PyPI Publishing Script${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"

# Function to print colored messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
log_info "Checking Python version..."
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.11"

if [[ $(echo "$python_version < $required_version" | bc) -eq 1 ]]; then
    log_error "Python $required_version or higher is required (found $python_version)"
    exit 1
fi

log_info "Python $python_version detected ✓"

# Check if we're in the project root
if [[ ! -f "pyproject.toml" ]]; then
    log_error "pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Parse command line arguments
TARGET="testpypi"
if [[ "$1" == "pypi" ]]; then
    TARGET="pypi"
    log_warn "Publishing to PRODUCTION PyPI"
elif [[ "$1" == "testpypi" ]] || [[ -z "$1" ]]; then
    TARGET="testpypi"
    log_info "Publishing to TEST PyPI (default)"
else
    log_error "Invalid target. Use 'testpypi' or 'pypi'"
    exit 1
fi

# Install build tools
log_info "Installing build tools..."
pip install --upgrade pip build twine wheel setuptools check-wheel-contents

# Clean previous builds
log_info "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info

# Extract version from pyproject.toml
log_info "Extracting version from pyproject.toml..."
ACTUAL_VERSION=$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
log_info "Building version: $ACTUAL_VERSION"

# Build the package
log_info "Building distribution packages..."
python3 -m build

# List built files
log_info "Built packages:"
ls -la dist/

# Check the wheel contents
log_info "Checking wheel contents..."
check-wheel-contents dist/*.whl || log_warn "Wheel check reported warnings"

# Run twine check
log_info "Running twine check..."
python3 -m twine check dist/*

# Generate SHA256 checksums
log_info "Generating SHA256 checksums..."
cd dist
sha256sum * > SHA256SUMS
cat SHA256SUMS
cd ..

# Configure upload based on target
if [[ "$TARGET" == "testpypi" ]]; then
    REPO_URL="https://test.pypi.org/legacy/"
    VIEW_URL="https://test.pypi.org/project/${PROJECT_NAME}/"
    INSTALL_CMD="pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ${PROJECT_NAME}"
else
    REPO_URL="https://upload.pypi.org/legacy/"
    VIEW_URL="https://pypi.org/project/${PROJECT_NAME}/"
    INSTALL_CMD="pip install ${PROJECT_NAME}"
fi

# Check for credentials
log_info "Checking for PyPI credentials..."

if [[ "$TARGET" == "testpypi" ]]; then
    if [[ -z "$TESTPYPI_API_TOKEN" ]]; then
        log_warn "TESTPYPI_API_TOKEN environment variable not set"
        log_info "You can set it with: export TESTPYPI_API_TOKEN=pypi-..."
        log_info "Or you will be prompted for username/password"
    fi
else
    if [[ -z "$PYPI_API_TOKEN" ]]; then
        log_warn "PYPI_API_TOKEN environment variable not set"
        log_info "You can set it with: export PYPI_API_TOKEN=pypi-..."
        log_info "Or you will be prompted for username/password"
    fi
fi

# Confirm before upload
echo -e "\n${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Ready to upload to $TARGET${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════${NC}"
echo "Package: ${PROJECT_NAME}"
echo "Version: ${ACTUAL_VERSION}"
echo "Target: ${REPO_URL}"
echo ""
read -p "Continue with upload? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Upload cancelled"
    exit 0
fi

# Upload to PyPI/TestPyPI
log_info "Uploading to $TARGET..."

if [[ "$TARGET" == "testpypi" ]] && [[ -n "$TESTPYPI_API_TOKEN" ]]; then
    python3 -m twine upload --repository-url "$REPO_URL" \
        --username __token__ \
        --password "$TESTPYPI_API_TOKEN" \
        --verbose \
        dist/*
elif [[ "$TARGET" == "pypi" ]] && [[ -n "$PYPI_API_TOKEN" ]]; then
    python3 -m twine upload --repository-url "$REPO_URL" \
        --username __token__ \
        --password "$PYPI_API_TOKEN" \
        --verbose \
        dist/*
else
    # Fall back to interactive authentication
    python3 -m twine upload --repository-url "$REPO_URL" \
        --verbose \
        dist/*
fi

# Check upload status
if [[ $? -eq 0 ]]; then
    log_info "Upload successful! ✓"
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}   Package published successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "View at: $VIEW_URL"
    echo ""
    echo "Install with:"
    echo "  $INSTALL_CMD"
    echo ""

    if [[ "$TARGET" == "testpypi" ]]; then
        echo "After testing, publish to production PyPI with:"
        echo "  ./scripts/publish.sh pypi"
    fi
else
    log_error "Upload failed"
    exit 1
fi

# Test installation (optional)
echo ""
read -p "Test installation from $TARGET? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Waiting 60 seconds for package to be available..."
    sleep 60

    log_info "Creating test virtual environment..."
    python3 -m venv test_env
    source test_env/bin/activate

    log_info "Installing from $TARGET..."
    eval "$INSTALL_CMD"

    log_info "Testing import..."
    python3 -c "import greenlang; print(f'Successfully imported greenlang {greenlang.__version__}')"

    log_info "Testing CLI..."
    gl --version || greenlang --version

    deactivate
    rm -rf test_env

    log_info "Installation test successful! ✓"
fi

echo -e "\n${GREEN}Publishing complete!${NC}"