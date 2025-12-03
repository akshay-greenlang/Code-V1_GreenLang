#!/bin/bash
# =============================================================================
# GreenLang Agents - Build Script
# =============================================================================
# Builds Docker images for all 3 agents
# Usage: ./scripts/build-agents.sh [tag] [--push] [--scan]
# =============================================================================

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-ghcr.io/greenlang}"
TAG="${1:-latest}"
PUSH="${2:-}"
SCAN="${3:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Agents to build
AGENTS=(
    "fuel-analyzer:generated/fuel_analyzer_agent"
    "carbon-intensity:generated/carbon_intensity_v1"
    "energy-performance:generated/energy_performance_v1"
    "eudr-compliance:generated/eudr_compliance_v1"
)

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Build base image first
build_base() {
    log_info "Building base image..."
    docker build \
        -t "${REGISTRY}/greenlang-base:${TAG}" \
        -f docker/base/Dockerfile.base \
        docker/base/

    if [[ "$PUSH" == "--push" ]]; then
        log_info "Pushing base image..."
        docker push "${REGISTRY}/greenlang-base:${TAG}"
    fi
}

# Build agent image
build_agent() {
    local name=$1
    local path=$2

    log_info "Building ${name}..."

    # Build the image
    docker build \
        --build-arg BASE_IMAGE="${REGISTRY}/greenlang-base:${TAG}" \
        -t "${REGISTRY}/${name}:${TAG}" \
        -t "${REGISTRY}/${name}:latest" \
        -f "${path}/Dockerfile" \
        .

    # Run Trivy security scan if requested
    if [[ "$SCAN" == "--scan" ]] || [[ "$PUSH" == "--scan" ]]; then
        log_info "Running Trivy security scan for ${name}..."
        if command -v trivy &> /dev/null; then
            trivy image \
                --severity HIGH,CRITICAL \
                --exit-code 0 \
                "${REGISTRY}/${name}:${TAG}"
        else
            log_warn "Trivy not installed, skipping security scan"
        fi
    fi

    # Push if requested
    if [[ "$PUSH" == "--push" ]]; then
        log_info "Pushing ${name}..."
        docker push "${REGISTRY}/${name}:${TAG}"
        docker push "${REGISTRY}/${name}:latest"
    fi

    log_info "Successfully built ${name}"
}

# Main
main() {
    log_info "=== GreenLang Agent Build ==="
    log_info "Registry: ${REGISTRY}"
    log_info "Tag: ${TAG}"
    log_info ""

    # Enable BuildKit
    export DOCKER_BUILDKIT=1

    # Build base image
    build_base

    # Build each agent
    for agent in "${AGENTS[@]}"; do
        IFS=':' read -r name path <<< "$agent"
        build_agent "$name" "$path"
    done

    log_info ""
    log_info "=== Build Complete ==="
    log_info "Built images:"
    for agent in "${AGENTS[@]}"; do
        IFS=':' read -r name path <<< "$agent"
        echo "  - ${REGISTRY}/${name}:${TAG}"
    done
}

# Run main
main "$@"
