#!/bin/bash

# Security Artifacts Verification Script for GreenLang Docker Images
# This script verifies cosign signatures, SBOM attachments, and vulnerability scan results

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REGISTRY="${REGISTRY:-ghcr.io}"
REPO_OWNER="${REPO_OWNER:-$(git config --get remote.origin.url | sed 's/.*github\.com[:/]\([^/]*\).*/\1/')}"
DOCKERHUB_ORG="${DOCKERHUB_ORG:-greenlang}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS] VERSION

Verify security artifacts for GreenLang Docker images.

OPTIONS:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -r, --registry REG  Registry to check (default: ghcr.io)
    -o, --owner OWNER   Repository owner (default: auto-detect from git)
    --dockerhub-org ORG Docker Hub organization (default: greenlang)
    --skip-cosign       Skip cosign signature verification
    --skip-sbom         Skip SBOM verification
    --skip-trivy        Skip Trivy scan result verification
    --output-format FMT Output format: text, json, markdown (default: text)

EXAMPLES:
    # Verify all security artifacts for version 0.2.0
    $0 0.2.0

    # Verify with custom registry and owner
    $0 -r ghcr.io -o myorg 0.2.0

    # Skip SBOM verification
    $0 --skip-sbom 0.2.0

    # Output results in JSON format
    $0 --output-format json 0.2.0

REQUIREMENTS:
    - cosign: Install from https://docs.sigstore.dev/cosign/installation/
    - docker: For pulling and inspecting images
    - jq: For JSON processing
    - curl: For API calls

EOF
}

# Default options
VERBOSE=false
SKIP_COSIGN=false
SKIP_SBOM=false
SKIP_TRIVY=false
OUTPUT_FORMAT="text"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -o|--owner)
            REPO_OWNER="$2"
            shift 2
            ;;
        --dockerhub-org)
            DOCKERHUB_ORG="$2"
            shift 2
            ;;
        --skip-cosign)
            SKIP_COSIGN=true
            shift
            ;;
        --skip-sbom)
            SKIP_SBOM=true
            shift
            ;;
        --skip-trivy)
            SKIP_TRIVY=true
            shift
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -*)
            log_error "Unknown option $1"
            usage
            exit 1
            ;;
        *)
            VERSION="$1"
            shift
            ;;
    esac
done

# Validate requirements
if [[ -z "${VERSION:-}" ]]; then
    log_error "Version is required"
    usage
    exit 1
fi

# Check required tools
check_requirements() {
    local missing_tools=()

    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v jq &> /dev/null; then
        missing_tools+=("jq")
    fi

    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi

    if [[ "$SKIP_COSIGN" == "false" ]] && ! command -v cosign &> /dev/null; then
        missing_tools+=("cosign")
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi
}

# Initialize results tracking
declare -A RESULTS
RESULTS["cosign_runner_ghcr"]="skipped"
RESULTS["cosign_runner_dockerhub"]="skipped"
RESULTS["cosign_full_ghcr"]="skipped"
RESULTS["cosign_full_dockerhub"]="skipped"
RESULTS["sbom_runner_ghcr"]="skipped"
RESULTS["sbom_runner_dockerhub"]="skipped"
RESULTS["sbom_full_ghcr"]="skipped"
RESULTS["sbom_full_dockerhub"]="skipped"
RESULTS["trivy_scan"]="skipped"

# Verify cosign signatures
verify_cosign_signature() {
    local image="$1"
    local image_type="$2"
    local registry_type="$3"

    log_info "Verifying cosign signature for $image ($image_type on $registry_type)"

    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Running: cosign verify $image --certificate-identity-regexp ..."
    fi

    # Construct expected certificate identity
    local cert_identity="https://github.com/${REPO_OWNER}/greenlang/.github/workflows/release-docker.yml@refs/tags/v${VERSION}"

    if cosign verify "$image" \
        --certificate-identity-regexp "https://github.com/${REPO_OWNER}/.*/.github/workflows/.*" \
        --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
        &> /dev/null; then

        log_success "Cosign signature verified for $image"
        RESULTS["cosign_${image_type}_${registry_type}"]="pass"
        return 0
    else
        log_error "Cosign signature verification failed for $image"
        RESULTS["cosign_${image_type}_${registry_type}"]="fail"
        return 1
    fi
}

# Verify SBOM attachments
verify_sbom_attachment() {
    local image="$1"
    local image_type="$2"
    local registry_type="$3"

    log_info "Verifying SBOM attachment for $image ($image_type on $registry_type)"

    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Running: cosign download sbom $image"
    fi

    # Try to download and validate SBOM
    local sbom_output
    if sbom_output=$(cosign download sbom "$image" 2>/dev/null); then
        # Validate SBOM is valid JSON and contains expected fields
        if echo "$sbom_output" | jq -e '.spdxVersion and .creationInfo and .packages' &> /dev/null; then
            log_success "SBOM verified for $image (SPDX format)"
            RESULTS["sbom_${image_type}_${registry_type}"]="pass"

            if [[ "$VERBOSE" == "true" ]]; then
                local package_count=$(echo "$sbom_output" | jq '.packages | length')
                log_info "SBOM contains $package_count packages"
            fi

            return 0
        else
            log_warning "SBOM found but format validation failed for $image"
            RESULTS["sbom_${image_type}_${registry_type}"]="partial"
            return 1
        fi
    else
        log_error "No SBOM attachment found for $image"
        RESULTS["sbom_${image_type}_${registry_type}"]="fail"
        return 1
    fi
}

# Check Trivy scan results
verify_trivy_results() {
    log_info "Checking Trivy vulnerability scan results"

    # Check if GitHub Security tab has Trivy results
    local repo_url="https://api.github.com/repos/${REPO_OWNER}/greenlang/code-scanning/alerts"

    if curl -s -H "Accept: application/vnd.github+json" "$repo_url" | jq -e '.[] | select(.tool.name == "Trivy")' &> /dev/null; then
        log_success "Trivy scan results found in GitHub Security tab"
        RESULTS["trivy_scan"]="pass"
        return 0
    else
        log_warning "No Trivy scan results found in GitHub Security tab"
        RESULTS["trivy_scan"]="partial"
        return 1
    fi
}

# Test image functionality
test_image_functionality() {
    local image="$1"
    local image_type="$2"

    log_info "Testing functionality of $image"

    if [[ "$image_type" == "runner" ]]; then
        if docker run --rm "$image" --version &> /dev/null; then
            log_success "Runner image functionality test passed"
            return 0
        else
            log_error "Runner image functionality test failed"
            return 1
        fi
    elif [[ "$image_type" == "full" ]]; then
        if docker run --rm "$image" gl --version &> /dev/null; then
            log_success "Full image functionality test passed"
            return 0
        else
            log_error "Full image functionality test failed"
            return 1
        fi
    fi
}

# Generate output in different formats
generate_output() {
    case "$OUTPUT_FORMAT" in
        "json")
            generate_json_output
            ;;
        "markdown")
            generate_markdown_output
            ;;
        *)
            generate_text_output
            ;;
    esac
}

generate_json_output() {
    local json_results='{'
    local first=true

    for key in "${!RESULTS[@]}"; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            json_results+=','
        fi
        json_results+="\"$key\":\"${RESULTS[$key]}\""
    done

    json_results+='}'

    echo "$json_results" | jq -r '{
        version: "'$VERSION'",
        timestamp: "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",
        registry: "'$REGISTRY'",
        owner: "'$REPO_OWNER'",
        results: . as $results | {
            cosign: {
                runner: {
                    ghcr: $results.cosign_runner_ghcr,
                    dockerhub: $results.cosign_runner_dockerhub
                },
                full: {
                    ghcr: $results.cosign_full_ghcr,
                    dockerhub: $results.cosign_full_dockerhub
                }
            },
            sbom: {
                runner: {
                    ghcr: $results.sbom_runner_ghcr,
                    dockerhub: $results.sbom_runner_dockerhub
                },
                full: {
                    ghcr: $results.sbom_full_ghcr,
                    dockerhub: $results.sbom_full_dockerhub
                }
            },
            trivy: $results.trivy_scan
        },
        summary: {
            total_checks: (. | to_entries | length),
            passed: (. | to_entries | map(select(.value == "pass")) | length),
            failed: (. | to_entries | map(select(.value == "fail")) | length),
            partial: (. | to_entries | map(select(.value == "partial")) | length),
            skipped: (. | to_entries | map(select(.value == "skipped")) | length)
        }
    }'
}

generate_markdown_output() {
    cat << EOF
# Security Artifacts Verification Report

**Version:** $VERSION
**Registry:** $REGISTRY
**Owner:** $REPO_OWNER
**Timestamp:** $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Summary

| Check Type | Runner (GHCR) | Runner (Docker Hub) | Full (GHCR) | Full (Docker Hub) |
|------------|---------------|---------------------|-------------|-------------------|
| **Cosign Signatures** | ${RESULTS[cosign_runner_ghcr]^^} | ${RESULTS[cosign_runner_dockerhub]^^} | ${RESULTS[cosign_full_ghcr]^^} | ${RESULTS[cosign_full_dockerhub]^^} |
| **SBOM Attachments** | ${RESULTS[sbom_runner_ghcr]^^} | ${RESULTS[sbom_runner_dockerhub]^^} | ${RESULTS[sbom_full_ghcr]^^} | ${RESULTS[sbom_full_dockerhub]^^} |

| Security Scan | Status |
|---------------|--------|
| **Trivy Vulnerability Scan** | ${RESULTS[trivy_scan]^^} |

## Legend

- ✅ **PASS**: Verification successful
- ❌ **FAIL**: Verification failed
- ⚠️ **PARTIAL**: Partially successful with warnings
- ⏭️ **SKIPPED**: Check was skipped

## Images Verified

### Runner Images (Minimal Production)
- GHCR: \`$REGISTRY/$REPO_OWNER/greenlang-runner:$VERSION\`
- Docker Hub: \`$DOCKERHUB_ORG/core-runner:$VERSION\`

### Full Images (Developer/CI)
- GHCR: \`$REGISTRY/$REPO_OWNER/greenlang-full:$VERSION\`
- Docker Hub: \`$DOCKERHUB_ORG/core-full:$VERSION\`

EOF
}

generate_text_output() {
    echo
    echo "=================================="
    echo "Security Artifacts Verification Report"
    echo "=================================="
    echo "Version: $VERSION"
    echo "Registry: $REGISTRY"
    echo "Owner: $REPO_OWNER"
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo

    echo "Cosign Signature Verification:"
    echo "  Runner (GHCR):     ${RESULTS[cosign_runner_ghcr]}"
    echo "  Runner (DockerHub): ${RESULTS[cosign_runner_dockerhub]}"
    echo "  Full (GHCR):       ${RESULTS[cosign_full_ghcr]}"
    echo "  Full (DockerHub):   ${RESULTS[cosign_full_dockerhub]}"
    echo

    echo "SBOM Attachment Verification:"
    echo "  Runner (GHCR):     ${RESULTS[sbom_runner_ghcr]}"
    echo "  Runner (DockerHub): ${RESULTS[sbom_runner_dockerhub]}"
    echo "  Full (GHCR):       ${RESULTS[sbom_full_ghcr]}"
    echo "  Full (DockerHub):   ${RESULTS[sbom_full_dockerhub]}"
    echo

    echo "Security Scans:"
    echo "  Trivy Results:     ${RESULTS[trivy_scan]}"
    echo

    # Count results
    local total=0 passed=0 failed=0 partial=0 skipped=0
    for result in "${RESULTS[@]}"; do
        ((total++))
        case "$result" in
            "pass") ((passed++)) ;;
            "fail") ((failed++)) ;;
            "partial") ((partial++)) ;;
            "skipped") ((skipped++)) ;;
        esac
    done

    echo "Summary: $passed passed, $failed failed, $partial partial, $skipped skipped (out of $total total)"

    if [[ $failed -gt 0 ]]; then
        echo
        log_error "Some security verification checks failed. Please review the results above."
        return 1
    elif [[ $partial -gt 0 ]]; then
        echo
        log_warning "Some security verification checks had warnings. Please review the results above."
        return 0
    else
        echo
        log_success "All enabled security verification checks passed!"
        return 0
    fi
}

# Main verification function
main() {
    log_info "Starting security artifacts verification for GreenLang v$VERSION"

    # Check requirements
    check_requirements

    # Define image URLs
    local runner_ghcr="$REGISTRY/$REPO_OWNER/greenlang-runner:$VERSION"
    local runner_dockerhub="$DOCKERHUB_ORG/core-runner:$VERSION"
    local full_ghcr="$REGISTRY/$REPO_OWNER/greenlang-full:$VERSION"
    local full_dockerhub="$DOCKERHUB_ORG/core-full:$VERSION"

    # Verify cosign signatures
    if [[ "$SKIP_COSIGN" == "false" ]]; then
        log_info "=== Verifying Cosign Signatures ==="
        verify_cosign_signature "$runner_ghcr" "runner" "ghcr" || true
        verify_cosign_signature "$runner_dockerhub" "runner" "dockerhub" || true
        verify_cosign_signature "$full_ghcr" "full" "ghcr" || true
        verify_cosign_signature "$full_dockerhub" "full" "dockerhub" || true
    else
        log_info "=== Skipping Cosign Signature Verification ==="
    fi

    # Verify SBOM attachments
    if [[ "$SKIP_SBOM" == "false" ]]; then
        log_info "=== Verifying SBOM Attachments ==="
        verify_sbom_attachment "$runner_ghcr" "runner" "ghcr" || true
        verify_sbom_attachment "$runner_dockerhub" "runner" "dockerhub" || true
        verify_sbom_attachment "$full_ghcr" "full" "ghcr" || true
        verify_sbom_attachment "$full_dockerhub" "full" "dockerhub" || true
    else
        log_info "=== Skipping SBOM Verification ==="
    fi

    # Verify Trivy scan results
    if [[ "$SKIP_TRIVY" == "false" ]]; then
        log_info "=== Verifying Trivy Scan Results ==="
        verify_trivy_results || true
    else
        log_info "=== Skipping Trivy Verification ==="
    fi

    # Test image functionality
    log_info "=== Testing Image Functionality ==="
    test_image_functionality "$runner_ghcr" "runner" || true
    test_image_functionality "$full_ghcr" "full" || true

    # Generate output
    log_info "=== Generating Verification Report ==="
    generate_output
}

# Run main function
main "$@"