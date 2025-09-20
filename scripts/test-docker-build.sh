#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GL_VERSION="${GL_VERSION:-0.2.0}"
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

# Test runner image build
print_status "Building runner image (local test)..."

docker buildx build \
    --platform=${PLATFORMS} \
    --file=Dockerfile.runner \
    --build-arg GL_VERSION=${GL_VERSION} \
    --build-arg BUILD_DATE=${BUILD_DATE} \
    --build-arg VCS_REF=${VCS_REF} \
    --tag ${REGISTRY}/greenlang-runner:test \
    --load \
    . || {
        print_error "Runner image build failed"
        exit 1
    }

print_success "Runner image built successfully"

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

# Test runner image
print_status "Testing runner image..."

# Test basic command
docker run --rm ${REGISTRY}/greenlang-runner:test --version || {
    print_error "Runner image test failed: gl --version"
    exit 1
}

# Test help command
docker run --rm ${REGISTRY}/greenlang-runner:test --help > /dev/null || {
    print_error "Runner image test failed: gl --help"
    exit 1
}

# Verify user ID
USER_ID=$(docker run --rm ${REGISTRY}/greenlang-runner:test id -u)
if [ "${USER_ID}" != "10001" ]; then
    print_error "Runner image has wrong user ID: ${USER_ID} (expected 10001)"
    exit 1
fi

print_success "Runner image tests passed"

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

    docker scout cves ${REGISTRY}/greenlang-runner:test --only-severity critical,high || true
    docker scout cves ${REGISTRY}/greenlang-full:test --only-severity critical,high || true
else
    print_status "Docker Scout not available, skipping security scan"
fi

# Image size report
print_status "Image size report:"

RUNNER_SIZE=$(docker image inspect ${REGISTRY}/greenlang-runner:test --format='{{.Size}}' | numfmt --to=iec)
FULL_SIZE=$(docker image inspect ${REGISTRY}/greenlang-full:test --format='{{.Size}}' | numfmt --to=iec)

echo "  Runner image: ${RUNNER_SIZE}"
echo "  Full image: ${FULL_SIZE}"

# Test docker-compose
print_status "Testing docker-compose setup..."

docker-compose build || {
    print_error "docker-compose build failed"
    exit 1
}

print_success "docker-compose build successful"

# Cleanup test images
print_status "Cleaning up test images..."

docker rmi ${REGISTRY}/greenlang-runner:test || true
docker rmi ${REGISTRY}/greenlang-full:test || true

print_success "Cleanup complete"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✅ All Docker build tests passed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Tag and push images: git tag v${GL_VERSION} && git push --tags"
echo "  2. Trigger CI workflow: gh workflow run release-docker.yml"
echo "  3. Verify signatures: cosign verify ${REGISTRY}/greenlang-runner:${GL_VERSION}"