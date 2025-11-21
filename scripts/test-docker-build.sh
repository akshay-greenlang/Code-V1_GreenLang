#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GL_VERSION="${GL_VERSION:-0.3.0}"
REGISTRY="${REGISTRY:-ghcr.io/akshay-greenlang}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

echo -e "${GREEN}🐳 GreenLang Docker Multi-Arch Build Test${NC}"
echo "================================================"
echo "Version: ${GL_VERSION}"
echo "Registry: ${REGISTRY}"
echo "Platforms: ${PLATFORMS}"
echo ""

# Function to print status
print_status() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! command -v docker buildx &> /dev/null; then
    print_error "Docker buildx is not available"
    exit 1
fi

print_success "Prerequisites check passed"

# Setup buildx builder
print_status "Setting up Docker buildx builder..."

BUILDER_NAME="greenlang-builder"
if ! docker buildx inspect ${BUILDER_NAME} &> /dev/null; then
    docker buildx create --name ${BUILDER_NAME} --use --platform=${PLATFORMS}
    docker buildx inspect --bootstrap
else
    docker buildx use ${BUILDER_NAME}
fi

print_success "Builder '${BUILDER_NAME}' is ready"

# Get build metadata
BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

print_status "Build metadata:"
echo "  Build Date: ${BUILD_DATE}"
echo "  VCS Ref: ${VCS_REF}"

# Test core image build (replacing runner with core)
print_status "Building core image (local test)..."

if [ -f "Dockerfile.core" ]; then
    docker buildx build \
        --platform=${PLATFORMS} \
        --file=Dockerfile.core \
        --build-arg GL_VERSION=${GL_VERSION} \
        --build-arg BUILD_DATE=${BUILD_DATE} \
        --build-arg VCS_REF=${VCS_REF} \
        --tag ${REGISTRY}/greenlang-core:test \
        --load \
        . || {
            print_error "Core image build failed"
            exit 1
        }
    print_success "Core image built successfully"
else
    print_status "Dockerfile.core not found, skipping core image build"
fi

# Test full image build
print_status "Building full image (local test)..."

docker buildx build \
    --platform=${PLATFORMS} \
    --file=Dockerfile.full \
    --build-arg GL_VERSION=${GL_VERSION} \
    --build-arg BUILD_DATE=${BUILD_DATE} \
    --build-arg VCS_REF=${VCS_REF} \
    --tag ${REGISTRY}/greenlang-full:test \
    --load \
    . || {
        print_error "Full image build failed"
        exit 1
    }

print_success "Full image built successfully"

# Test core image
if [ -f "Dockerfile.core" ]; then
    print_status "Testing core image..."

    # Test basic command
    docker run --rm ${REGISTRY}/greenlang-core:test gl version || {
        print_error "Core image test failed: gl version"
        exit 1
    }

    # Test help command
    docker run --rm ${REGISTRY}/greenlang-core:test gl --help > /dev/null || {
        print_error "Core image test failed: gl --help"
        exit 1
    }

    # Verify user ID
    USER_ID=$(docker run --rm ${REGISTRY}/greenlang-core:test id -u)
    if [ "${USER_ID}" != "10001" ]; then
        print_error "Core image has wrong user ID: ${USER_ID} (expected 10001)"
        exit 1
    fi

    print_success "Core image tests passed"
fi

# Test full image
print_status "Testing full image..."

# Test gl command
docker run --rm ${REGISTRY}/greenlang-full:test gl --version || {
    print_error "Full image test failed: gl --version"
    exit 1
}

# Test development tools
docker run --rm ${REGISTRY}/greenlang-full:test python -c "import pytest, mypy, black; print('Dev tools OK')" || {
    print_error "Full image test failed: dev tools import"
    exit 1
}

# Verify user ID
USER_ID=$(docker run --rm ${REGISTRY}/greenlang-full:test id -u)
if [ "${USER_ID}" != "10001" ]; then
    print_error "Full image has wrong user ID: ${USER_ID} (expected 10001)"
    exit 1
fi

print_success "Full image tests passed"

# Security scan with Docker Scout (if available)
if command -v docker-scout &> /dev/null; then
    print_status "Running security scan with Docker Scout..."

    [ -f "Dockerfile.core" ] && docker scout cves ${REGISTRY}/greenlang-core:test --only-severity critical,high || true
    docker scout cves ${REGISTRY}/greenlang-full:test --only-severity critical,high || true
else
    print_status "Docker Scout not available, skipping security scan"
fi

# Image size report
print_status "Image size report:"

if [ -f "Dockerfile.core" ] && docker image inspect ${REGISTRY}/greenlang-core:test &> /dev/null; then
    CORE_SIZE=$(docker image inspect ${REGISTRY}/greenlang-core:test --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo "N/A")
    echo "  Core image: ${CORE_SIZE}"
fi

if docker image inspect ${REGISTRY}/greenlang-full:test &> /dev/null; then
    FULL_SIZE=$(docker image inspect ${REGISTRY}/greenlang-full:test --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo "N/A")
    echo "  Full image: ${FULL_SIZE}"
fi

# Test docker compose (if docker-compose.yml exists)
if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
    print_status "Testing docker compose setup..."

    # Use new docker compose command (v2) if available, fallback to docker-compose
    if docker compose version &> /dev/null; then
        docker compose build || {
            print_error "docker compose build failed"
            exit 1
        }
    elif command -v docker-compose &> /dev/null; then
        docker-compose build || {
            print_error "docker-compose build failed"
            exit 1
        }
    else
        print_status "Docker Compose not available, skipping compose tests"
    fi

    print_success "docker compose build successful"
else
    print_status "No docker-compose.yml found, skipping compose tests"
fi

# Cleanup test images
print_status "Cleaning up test images..."

docker rmi ${REGISTRY}/greenlang-core:test 2>/dev/null || true
docker rmi ${REGISTRY}/greenlang-full:test 2>/dev/null || true

print_success "Cleanup complete"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✅ All Docker build tests passed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Tag and push images: git tag v${GL_VERSION} && git push --tags"
echo "  2. Trigger CI workflow: gh workflow run release-docker.yml"
echo "  3. Verify signatures: cosign verify ${REGISTRY}/greenlang-core:${GL_VERSION}"