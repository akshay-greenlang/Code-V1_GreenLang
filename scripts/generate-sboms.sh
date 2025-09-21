#!/bin/bash
# Generate comprehensive SBOMs for GreenLang packages and Docker images
# This script produces CycloneDX (primary) and SPDX (secondary) format SBOMs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION="${1:-0.2.0}"
ARTIFACTS_DIR="artifacts/sbom"
SYFT_VERSION="v1.0.0"
COSIGN_VERSION="v2.2.4"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Syft if not present
install_syft() {
    if command_exists syft; then
        log_info "Syft is already installed: $(syft version)"
    else
        log_info "Installing Syft ${SYFT_VERSION}..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command_exists brew; then
                brew install syft
            else
                curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin ${SYFT_VERSION}
            fi
        else
            # Linux
            curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sudo sh -s -- -b /usr/local/bin ${SYFT_VERSION}
        fi
        log_success "Syft installed successfully"
    fi
}

# Install Cosign if not present
install_cosign() {
    if command_exists cosign; then
        log_info "Cosign is already installed: $(cosign version)"
    else
        log_info "Installing Cosign ${COSIGN_VERSION}..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command_exists brew; then
                brew install cosign
            else
                curl -sSfL https://github.com/sigstore/cosign/releases/download/${COSIGN_VERSION}/cosign-darwin-amd64 -o cosign
                chmod +x cosign
                sudo mv cosign /usr/local/bin/
            fi
        else
            # Linux
            curl -sSfL https://github.com/sigstore/cosign/releases/download/${COSIGN_VERSION}/cosign-linux-amd64 -o cosign
            chmod +x cosign
            sudo mv cosign /usr/local/bin/
        fi
        log_success "Cosign installed successfully"
    fi
}

# Generate SBOM for Python wheel
generate_wheel_sbom() {
    local wheel_file="$1"
    if [[ ! -f "$wheel_file" ]]; then
        log_error "Wheel file not found: $wheel_file"
        return 1
    fi

    local wheel_name=$(basename "$wheel_file")
    local pkg_name=$(echo "$wheel_name" | cut -d'-' -f1)
    local pkg_version=$(echo "$wheel_name" | cut -d'-' -f2)

    log_info "Generating SBOM for wheel: $wheel_name"

    # CycloneDX format (PRIMARY)
    syft "$wheel_file" \
        -o cyclonedx-json="${ARTIFACTS_DIR}/sbom-${pkg_name}-${pkg_version}-wheel.cdx.json" \
        --name "${pkg_name}-wheel" \
        --source-name "${wheel_name}" \
        --source-version "${pkg_version}"

    # SPDX format (SECONDARY)
    syft "$wheel_file" \
        -o spdx-json="${ARTIFACTS_DIR}/sbom-${pkg_name}-${pkg_version}-wheel.spdx.json" \
        --name "${pkg_name}-wheel" \
        --source-name "${wheel_name}" \
        --source-version "${pkg_version}"

    log_success "Generated wheel SBOMs (CycloneDX + SPDX)"
}

# Generate SBOM for Python sdist
generate_sdist_sbom() {
    local sdist_file="$1"
    if [[ ! -f "$sdist_file" ]]; then
        log_error "Sdist file not found: $sdist_file"
        return 1
    fi

    local sdist_name=$(basename "$sdist_file")
    local pkg_name=$(echo "$sdist_name" | cut -d'-' -f1)
    local pkg_version=$(echo "$sdist_name" | cut -d'-' -f2 | sed 's/.tar.gz$//')

    log_info "Generating SBOM for sdist: $sdist_name"

    # CycloneDX format (REQUIRED)
    syft "$sdist_file" \
        -o cyclonedx-json="${ARTIFACTS_DIR}/sbom-${pkg_name}-${pkg_version}-sdist.cdx.json" \
        --name "${pkg_name}-sdist" \
        --source-name "${sdist_name}" \
        --source-version "${pkg_version}"

    # SPDX format (OPTIONAL)
    syft "$sdist_file" \
        -o spdx-json="${ARTIFACTS_DIR}/sbom-${pkg_name}-${pkg_version}-sdist.spdx.json" \
        --name "${pkg_name}-sdist" \
        --source-name "${sdist_name}" \
        --source-version "${pkg_version}"

    log_success "Generated sdist SBOMs (CycloneDX + SPDX)"
}

# Generate SBOM for Docker image
generate_docker_sbom() {
    local image_name="$1"
    local image_tag="$2"
    local image_type="$3"  # runner or full

    log_info "Generating SBOM for Docker image: ${image_name}:${image_tag}"

    # Sanitize image name for filename
    local safe_name=$(echo "$image_name" | tr '/' '-' | tr ':' '-')

    # CycloneDX format (PRIMARY)
    syft "docker:${image_name}:${image_tag}" \
        -o cyclonedx-json="${ARTIFACTS_DIR}/sbom-image-${safe_name}-${image_tag}.cdx.json" \
        --name "greenlang-${image_type}" \
        --source-name "${image_name}" \
        --source-version "${image_tag}"

    # SPDX format (SECONDARY)
    syft "docker:${image_name}:${image_tag}" \
        -o spdx-json="${ARTIFACTS_DIR}/sbom-image-${safe_name}-${image_tag}.spdx.json" \
        --name "greenlang-${image_type}" \
        --source-name "${image_name}" \
        --source-version "${image_tag}"

    log_success "Generated Docker image SBOMs (CycloneDX + SPDX)"
}

# Create Docker SBOM attestation
create_docker_attestation() {
    local image_name="$1"
    local image_tag="$2"
    local sbom_file="$3"

    if ! command_exists cosign; then
        log_warn "Cosign not available, skipping attestation creation"
        return
    fi

    log_info "Creating CycloneDX attestation for ${image_name}:${image_tag}"

    # Note: This requires the image to be pushed to a registry first
    cosign attest --yes \
        --predicate "${sbom_file}" \
        --type cyclonedx \
        "${image_name}:${image_tag}" 2>/dev/null || {
        log_warn "Could not create attestation (image may not be in registry yet)"
        log_info "To create attestation after pushing, run:"
        log_info "  cosign attest --predicate ${sbom_file} --type cyclonedx ${image_name}:${image_tag}"
    }
}

# Verify Docker SBOM attestation
verify_docker_attestation() {
    local image_name="$1"
    local image_tag="$2"

    if ! command_exists cosign; then
        log_warn "Cosign not available, skipping attestation verification"
        return
    fi

    log_info "Verifying CycloneDX attestation for ${image_name}:${image_tag}"

    cosign verify-attestation \
        --type cyclonedx \
        --certificate-identity-regexp ".*" \
        --certificate-oidc-issuer https://token.actions.githubusercontent.com \
        "${image_name}:${image_tag}" 2>/dev/null || {
        log_warn "Could not verify attestation (image may not have attestation yet)"
    }
}

# Display SBOM summary
display_summary() {
    log_info "SBOM Generation Summary"
    echo "========================"

    if [[ -d "$ARTIFACTS_DIR" ]]; then
        echo ""
        echo "Generated SBOMs:"
        for sbom in "$ARTIFACTS_DIR"/*.json; do
            if [[ -f "$sbom" ]]; then
                local size=$(du -h "$sbom" | cut -f1)
                local name=$(basename "$sbom")

                if [[ "$name" == *"cdx.json" ]]; then
                    local components=$(jq -r '.components | length' "$sbom" 2>/dev/null || echo "?")
                    echo "  ✓ $name ($size, $components components)"
                else
                    local packages=$(jq -r '.packages | length' "$sbom" 2>/dev/null || echo "?")
                    echo "  ✓ $name ($size, $packages packages)"
                fi
            fi
        done

        echo ""
        local total=$(find "$ARTIFACTS_DIR" -name "*.json" | wc -l)
        log_success "Total SBOMs generated: $total"
    else
        log_warn "No SBOMs generated"
    fi
}

# Main execution
main() {
    echo "======================================"
    echo "GreenLang SBOM Generation Tool v${VERSION}"
    echo "======================================"
    echo ""

    # Parse arguments
    case "$1" in
        --help|-h)
            echo "Usage: $0 [version] [options]"
            echo ""
            echo "Options:"
            echo "  --python     Generate SBOMs for Python packages only"
            echo "  --docker     Generate SBOMs for Docker images only"
            echo "  --verify     Verify existing attestations"
            echo "  --clean      Clean artifacts directory before generation"
            echo ""
            echo "Examples:"
            echo "  $0                    Generate all SBOMs for version 0.2.0"
            echo "  $0 0.3.0              Generate all SBOMs for version 0.3.0"
            echo "  $0 0.2.0 --python     Generate Python SBOMs only"
            echo "  $0 0.2.0 --docker     Generate Docker SBOMs only"
            echo "  $0 0.2.0 --verify     Verify Docker attestations"
            exit 0
            ;;
    esac

    # Setup
    if [[ "$2" == "--clean" ]] || [[ "$3" == "--clean" ]]; then
        log_info "Cleaning artifacts directory..."
        rm -rf "$ARTIFACTS_DIR"
    fi

    mkdir -p "$ARTIFACTS_DIR"

    # Install required tools
    install_syft
    if [[ "$2" == "--docker" ]] || [[ -z "$2" ]]; then
        install_cosign
    fi

    # Generate Python package SBOMs
    if [[ "$2" == "--python" ]] || [[ -z "$2" ]]; then
        log_info "Building Python packages..."

        if command_exists python; then
            python -m pip install --quiet --upgrade pip build
            python -m build

            # Generate SBOMs for wheel
            for wheel in dist/*.whl; do
                if [[ -f "$wheel" ]]; then
                    generate_wheel_sbom "$wheel"
                    break  # Only process first wheel
                fi
            done

            # Generate SBOMs for sdist
            for sdist in dist/*.tar.gz; do
                if [[ -f "$sdist" ]]; then
                    generate_sdist_sbom "$sdist"
                    break  # Only process first sdist
                fi
            done
        else
            log_error "Python not found. Skipping Python package SBOMs."
        fi
    fi

    # Generate Docker image SBOMs
    if [[ "$2" == "--docker" ]] || [[ -z "$2" ]]; then
        log_info "Processing Docker images..."

        # Build Docker images if not present
        if [[ -f "Dockerfile.runner" ]]; then
            if ! docker images | grep -q "greenlang-runner.*${VERSION}"; then
                log_info "Building runner image..."
                docker build -t "greenlang-runner:${VERSION}" \
                    --build-arg GL_VERSION="${VERSION}" \
                    -f Dockerfile.runner .
            fi
            generate_docker_sbom "greenlang-runner" "${VERSION}" "runner"
        fi

        if [[ -f "Dockerfile.full" ]]; then
            if ! docker images | grep -q "greenlang-full.*${VERSION}"; then
                log_info "Building full image..."
                docker build -t "greenlang-full:${VERSION}" \
                    --build-arg GL_VERSION="${VERSION}" \
                    -f Dockerfile.full .
            fi
            generate_docker_sbom "greenlang-full" "${VERSION}" "full"
        fi
    fi

    # Verify attestations if requested
    if [[ "$2" == "--verify" ]]; then
        log_info "Verifying Docker attestations..."
        verify_docker_attestation "ghcr.io/${GITHUB_REPOSITORY_OWNER:-akshay-greenlang}/greenlang-runner" "${VERSION}"
        verify_docker_attestation "ghcr.io/${GITHUB_REPOSITORY_OWNER:-akshay-greenlang}/greenlang-full" "${VERSION}"
    fi

    # Display summary
    echo ""
    display_summary

    echo ""
    echo "======================================"
    log_success "SBOM generation complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review generated SBOMs in ${ARTIFACTS_DIR}/"
    echo "  2. Sign SBOMs with: cosign sign-blob --output-signature <sbom>.sig <sbom>.json"
    echo "  3. Push images and create attestations in CI/CD"
    echo "======================================"
}

# Run main function
main "$@"