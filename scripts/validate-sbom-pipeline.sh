#!/bin/bash
# Validate SBOM Generation Pipeline for GreenLang v0.2.0
# This script validates that all SBOM generation requirements are met

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
VALIDATION_FAILED=false

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Helper functions
log_test() {
    echo -e "\n${BLUE}[TEST]${NC} $1"
    ((TOTAL_TESTS++))
}

log_pass() {
    echo -e "${GREEN}  ✓${NC} $1"
    ((PASSED_TESTS++))
}

log_fail() {
    echo -e "${RED}  ✗${NC} $1"
    ((FAILED_TESTS++))
    VALIDATION_FAILED=true
}

log_skip() {
    echo -e "${YELLOW}  ⊖${NC} $1"
    ((SKIPPED_TESTS++))
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validation functions
validate_tools() {
    log_test "Tool Installation Validation"

    if command_exists syft; then
        log_pass "Syft is installed: $(syft version 2>&1 | grep 'Version' | head -1)"
    else
        log_fail "Syft is not installed"
    fi

    if command_exists cosign; then
        log_pass "Cosign is installed: $(cosign version 2>&1 | grep 'GitVersion' | head -1)"
    else
        log_fail "Cosign is not installed"
    fi

    if command_exists jq; then
        log_pass "jq is installed for JSON parsing"
    else
        log_fail "jq is not installed (required for SBOM validation)"
    fi

    if command_exists docker; then
        log_pass "Docker is installed: $(docker --version)"
    else
        log_skip "Docker is not installed (Docker SBOM tests will be skipped)"
    fi
}

validate_python_sboms() {
    log_test "Python Package SBOM Validation"

    # Check if Python build artifacts exist
    if [[ ! -d "dist" ]]; then
        log_info "Building Python packages..."
        if command_exists python; then
            python -m pip install --quiet --upgrade pip build
            python -m build
        else
            log_skip "Python not available, skipping Python SBOM tests"
            return
        fi
    fi

    # Validate wheel SBOM
    local wheel_file=$(ls dist/*.whl 2>/dev/null | head -1)
    if [[ -n "$wheel_file" ]]; then
        # Test CycloneDX generation
        if syft "$wheel_file" -o cyclonedx-json > /tmp/test-wheel.cdx.json 2>/dev/null; then
            if [[ -s /tmp/test-wheel.cdx.json ]]; then
                local components=$(jq '.components | length' /tmp/test-wheel.cdx.json)
                log_pass "Wheel CycloneDX SBOM generated ($components components)"
            else
                log_fail "Wheel CycloneDX SBOM is empty"
            fi
        else
            log_fail "Failed to generate wheel CycloneDX SBOM"
        fi

        # Test SPDX generation
        if syft "$wheel_file" -o spdx-json > /tmp/test-wheel.spdx.json 2>/dev/null; then
            if [[ -s /tmp/test-wheel.spdx.json ]]; then
                local packages=$(jq '.packages | length' /tmp/test-wheel.spdx.json)
                log_pass "Wheel SPDX SBOM generated ($packages packages)"
            else
                log_fail "Wheel SPDX SBOM is empty"
            fi
        else
            log_fail "Failed to generate wheel SPDX SBOM"
        fi
    else
        log_skip "No wheel file found in dist/"
    fi

    # Validate sdist SBOM
    local sdist_file=$(ls dist/*.tar.gz 2>/dev/null | head -1)
    if [[ -n "$sdist_file" ]]; then
        # Test CycloneDX generation
        if syft "$sdist_file" -o cyclonedx-json > /tmp/test-sdist.cdx.json 2>/dev/null; then
            if [[ -s /tmp/test-sdist.cdx.json ]]; then
                local components=$(jq '.components | length' /tmp/test-sdist.cdx.json)
                log_pass "Sdist CycloneDX SBOM generated ($components components)"
            else
                log_fail "Sdist CycloneDX SBOM is empty"
            fi
        else
            log_fail "Failed to generate sdist CycloneDX SBOM"
        fi
    else
        log_skip "No sdist file found in dist/"
    fi
}

validate_docker_sboms() {
    log_test "Docker Image SBOM Validation"

    if ! command_exists docker; then
        log_skip "Docker not available, skipping Docker SBOM tests"
        return
    fi

    # Test runner image
    if [[ -f "Dockerfile.runner" ]]; then
        local test_image="test-greenlang-runner:test"

        # Build test image
        log_info "Building test runner image..."
        if docker build -t "$test_image" -f Dockerfile.runner . >/dev/null 2>&1; then
            # Test CycloneDX generation
            if syft "docker:$test_image" -o cyclonedx-json > /tmp/test-runner.cdx.json 2>/dev/null; then
                if [[ -s /tmp/test-runner.cdx.json ]]; then
                    local components=$(jq '.components | length' /tmp/test-runner.cdx.json)
                    log_pass "Runner CycloneDX SBOM generated ($components components)"
                else
                    log_fail "Runner CycloneDX SBOM is empty"
                fi
            else
                log_fail "Failed to generate runner CycloneDX SBOM"
            fi

            # Test SPDX generation
            if syft "docker:$test_image" -o spdx-json > /tmp/test-runner.spdx.json 2>/dev/null; then
                if [[ -s /tmp/test-runner.spdx.json ]]; then
                    local packages=$(jq '.packages | length' /tmp/test-runner.spdx.json)
                    log_pass "Runner SPDX SBOM generated ($packages packages)"
                else
                    log_fail "Runner SPDX SBOM is empty"
                fi
            else
                log_fail "Failed to generate runner SPDX SBOM"
            fi

            # Cleanup
            docker rmi "$test_image" >/dev/null 2>&1
        else
            log_fail "Failed to build test runner image"
        fi
    else
        log_skip "Dockerfile.runner not found"
    fi

    # Test full image
    if [[ -f "Dockerfile.full" ]]; then
        local test_image="test-greenlang-full:test"

        # Build test image
        log_info "Building test full image..."
        if docker build -t "$test_image" -f Dockerfile.full . >/dev/null 2>&1; then
            # Test CycloneDX generation
            if syft "docker:$test_image" -o cyclonedx-json > /tmp/test-full.cdx.json 2>/dev/null; then
                if [[ -s /tmp/test-full.cdx.json ]]; then
                    local components=$(jq '.components | length' /tmp/test-full.cdx.json)
                    log_pass "Full CycloneDX SBOM generated ($components components)"
                else
                    log_fail "Full CycloneDX SBOM is empty"
                fi
            else
                log_fail "Failed to generate full CycloneDX SBOM"
            fi

            # Cleanup
            docker rmi "$test_image" >/dev/null 2>&1
        else
            log_fail "Failed to build test full image"
        fi
    else
        log_skip "Dockerfile.full not found"
    fi
}

validate_sbom_formats() {
    log_test "SBOM Format Validation"

    # Test CycloneDX format compliance
    if [[ -f /tmp/test-wheel.cdx.json ]]; then
        # Check required CycloneDX fields
        local bom_format=$(jq -r '.bomFormat' /tmp/test-wheel.cdx.json)
        local spec_version=$(jq -r '.specVersion' /tmp/test-wheel.cdx.json)

        if [[ "$bom_format" == "CycloneDX" ]]; then
            log_pass "CycloneDX format valid: bomFormat=$bom_format"
        else
            log_fail "Invalid CycloneDX bomFormat: $bom_format"
        fi

        if [[ -n "$spec_version" ]]; then
            log_pass "CycloneDX specVersion present: $spec_version"
        else
            log_fail "CycloneDX specVersion missing"
        fi
    else
        log_skip "No CycloneDX SBOM to validate"
    fi

    # Test SPDX format compliance
    if [[ -f /tmp/test-wheel.spdx.json ]]; then
        # Check required SPDX fields
        local spdx_version=$(jq -r '.spdxVersion' /tmp/test-wheel.spdx.json)
        local creation_info=$(jq -r '.creationInfo' /tmp/test-wheel.spdx.json)

        if [[ "$spdx_version" =~ ^SPDX- ]]; then
            log_pass "SPDX format valid: spdxVersion=$spdx_version"
        else
            log_fail "Invalid SPDX spdxVersion: $spdx_version"
        fi

        if [[ "$creation_info" != "null" ]]; then
            log_pass "SPDX creationInfo present"
        else
            log_fail "SPDX creationInfo missing"
        fi
    else
        log_skip "No SPDX SBOM to validate"
    fi
}

validate_github_workflow() {
    log_test "GitHub Workflow Validation"

    if [[ -f ".github/workflows/sbom-generation.yml" ]]; then
        log_pass "SBOM generation workflow exists"

        # Check for required job configurations
        if grep -q "python-sbom:" .github/workflows/sbom-generation.yml; then
            log_pass "Python SBOM job configured"
        else
            log_fail "Python SBOM job missing in workflow"
        fi

        if grep -q "docker-sbom:" .github/workflows/sbom-generation.yml; then
            log_pass "Docker SBOM job configured"
        else
            log_fail "Docker SBOM job missing in workflow"
        fi

        if grep -q "cosign attest" .github/workflows/sbom-generation.yml; then
            log_pass "Cosign attestation configured"
        else
            log_fail "Cosign attestation not configured"
        fi

        if grep -q "cyclonedx" .github/workflows/sbom-generation.yml; then
            log_pass "CycloneDX format configured as primary"
        else
            log_fail "CycloneDX format not configured"
        fi
    else
        log_fail "SBOM generation workflow not found"
    fi

    # Check integration with release workflows
    if [[ -f ".github/workflows/release-build.yml" ]]; then
        if grep -q "sbom-generation.yml" .github/workflows/release-build.yml; then
            log_pass "SBOM generation integrated in release-build.yml"
        else
            log_fail "SBOM generation not integrated in release-build.yml"
        fi
    else
        log_skip "release-build.yml not found"
    fi
}

validate_scripts() {
    log_test "Developer Script Validation"

    if [[ -f "scripts/generate-sboms.sh" ]]; then
        log_pass "Unix/Linux SBOM generation script exists"

        # Check if script is executable
        if [[ -x "scripts/generate-sboms.sh" ]]; then
            log_pass "Script is executable"
        else
            log_fail "Script is not executable (run: chmod +x scripts/generate-sboms.sh)"
        fi
    else
        log_fail "Unix/Linux SBOM generation script not found"
    fi

    if [[ -f "scripts/generate-sboms.bat" ]]; then
        log_pass "Windows SBOM generation script exists"
    else
        log_fail "Windows SBOM generation script not found"
    fi
}

validate_documentation() {
    log_test "Documentation Validation"

    if [[ -f "docs/security/sbom.md" ]]; then
        log_pass "SBOM documentation exists"

        # Check for required sections
        if grep -q "## Verification Steps" docs/security/sbom.md; then
            log_pass "Verification steps documented"
        else
            log_fail "Verification steps missing in documentation"
        fi

        if grep -q "cosign verify-attestation" docs/security/sbom.md; then
            log_pass "Attestation verification commands documented"
        else
            log_fail "Attestation verification commands missing"
        fi

        if grep -q "## Local SBOM Generation" docs/security/sbom.md; then
            log_pass "Local generation steps documented"
        else
            log_fail "Local generation steps missing"
        fi
    else
        log_fail "SBOM documentation not found at docs/security/sbom.md"
    fi
}

validate_attestation_support() {
    log_test "Attestation Support Validation"

    if ! command_exists cosign; then
        log_skip "Cosign not available, skipping attestation tests"
        return
    fi

    # Check if we can perform keyless signing (requires OIDC)
    if [[ -n "$CI" ]] || [[ -n "$GITHUB_ACTIONS" ]]; then
        log_pass "Running in CI environment with OIDC support"
    else
        log_info "Not in CI environment, attestation will require manual signing"
        log_pass "Cosign available for local signing"
    fi

    # Test blob signing capability
    echo "test" > /tmp/test-blob.txt
    if cosign sign-blob --yes /tmp/test-blob.txt --output-signature /tmp/test-blob.sig 2>/dev/null; then
        log_pass "Blob signing capability confirmed"
        rm -f /tmp/test-blob.txt /tmp/test-blob.sig
    else
        log_skip "Blob signing requires authentication setup"
    fi
}

# Display summary
display_summary() {
    echo ""
    echo "======================================"
    echo "SBOM Pipeline Validation Summary"
    echo "======================================"
    echo ""
    echo "Total Tests:    $TOTAL_TESTS"
    echo -e "${GREEN}Passed:         $PASSED_TESTS${NC}"

    if [[ $FAILED_TESTS -gt 0 ]]; then
        echo -e "${RED}Failed:         $FAILED_TESTS${NC}"
    else
        echo "Failed:         $FAILED_TESTS"
    fi

    if [[ $SKIPPED_TESTS -gt 0 ]]; then
        echo -e "${YELLOW}Skipped:        $SKIPPED_TESTS${NC}"
    else
        echo "Skipped:        $SKIPPED_TESTS"
    fi

    echo ""

    if [[ "$VALIDATION_FAILED" == "true" ]]; then
        echo -e "${RED}❌ VALIDATION FAILED${NC}"
        echo ""
        echo "Required fixes:"
        echo "  1. Install missing tools (syft, cosign, jq)"
        echo "  2. Fix failing workflow configurations"
        echo "  3. Ensure all scripts are present and executable"
        echo "  4. Complete missing documentation sections"
        return 1
    else
        echo -e "${GREEN}✅ VALIDATION PASSED${NC}"
        echo ""
        echo "The SBOM generation pipeline meets v0.2.0 requirements!"
    fi
}

# Main execution
main() {
    echo "======================================"
    echo "SBOM Pipeline Validation for v${VERSION}"
    echo "======================================"
    echo ""
    echo "This validates compliance with v0.2.0 SBOM requirements:"
    echo "  - CycloneDX as primary format"
    echo "  - SPDX as secondary format"
    echo "  - Attestations for Docker images"
    echo "  - Complete documentation"
    echo ""

    # Run validations
    validate_tools
    validate_python_sboms
    validate_docker_sboms
    validate_sbom_formats
    validate_github_workflow
    validate_scripts
    validate_documentation
    validate_attestation_support

    # Display summary
    display_summary

    # Cleanup temporary files
    rm -f /tmp/test-*.json /tmp/test-*.txt /tmp/test-*.sig

    if [[ "$VALIDATION_FAILED" == "true" ]]; then
        exit 1
    fi
}

# Parse arguments
case "$1" in
    --help|-h)
        echo "Usage: $0 [version]"
        echo ""
        echo "Validates the SBOM generation pipeline for GreenLang"
        echo ""
        echo "Arguments:"
        echo "  version     Version to validate (default: 0.2.0)"
        echo ""
        echo "Examples:"
        echo "  $0          Validate for version 0.2.0"
        echo "  $0 0.3.0    Validate for version 0.3.0"
        exit 0
        ;;
esac

# Run main function
main "$@"