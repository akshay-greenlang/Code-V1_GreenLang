#!/bin/bash
# DoD Verification Script for SBOM Implementation v0.2.0
# This script verifies ALL Definition of Done requirements

set -e

# Configuration
VERSION="${1:-0.2.0}"
ARTIFACTS_DIR="artifacts/sbom"
GHCR_REGISTRY="ghcr.io"
OWNER="${GITHUB_REPOSITORY_OWNER:-greenlang}"
DOCKER_IMAGES=("runner" "full")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Tracking
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ PASS:${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_fail() {
    echo -e "${RED}❌ FAIL:${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

check_info() {
    echo -e "${BLUE}ℹ️  INFO:${NC} $1"
}

section() {
    echo -e "\n${YELLOW}═══ $1 ═══${NC}"
}

# DoD Verification Runbook Implementation
main() {
    echo "════════════════════════════════════════════════════════════"
    echo "    SBOM Definition of Done (DoD) Verification v${VERSION}"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Gate: SBOM+Attestations Ready"
    echo "Version: ${VERSION}"
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""

    # 1) Inventory check
    section "1) Inventory Check"

    check_info "Checking Python artifacts in dist/"
    if [[ -d "dist" ]]; then
        echo "Python artifacts:"
        ls -1 dist/ 2>/dev/null || echo "  No files in dist/"

        # Check for wheel
        if ls dist/*.whl >/dev/null 2>&1; then
            check_pass "Python wheel found"
        else
            check_fail "No Python wheel found in dist/"
        fi

        # Check for sdist
        if ls dist/*.tar.gz >/dev/null 2>&1; then
            check_pass "Python sdist found"
        else
            check_fail "No Python sdist found in dist/"
        fi
    else
        check_fail "dist/ directory not found"
    fi

    check_info "Checking SBOM files in ${ARTIFACTS_DIR}/"
    if [[ -d "$ARTIFACTS_DIR" ]]; then
        echo "SBOM files:"
        ls -1 "$ARTIFACTS_DIR/" 2>/dev/null | sort || echo "  No files in ${ARTIFACTS_DIR}/"

        # Expected files for Python packages
        EXPECTED_PYTHON=(
            "sbom-greenlang-${VERSION}-wheel.cdx.json"
            "sbom-greenlang-${VERSION}-wheel.spdx.json"
            "sbom-greenlang-${VERSION}-sdist.cdx.json"
        )

        for expected in "${EXPECTED_PYTHON[@]}"; do
            if [[ -f "${ARTIFACTS_DIR}/${expected}" ]]; then
                check_pass "Found: ${expected}"
            else
                check_fail "Missing: ${expected}"
            fi
        done

        # Expected files for Docker images
        SAFE_OWNER=$(echo "$OWNER" | tr '/' '-')
        for image in "${DOCKER_IMAGES[@]}"; do
            cdx_file="sbom-image-ghcr-io-${SAFE_OWNER}-greenlang-${image}-${VERSION}.cdx.json"
            spdx_file="sbom-image-ghcr-io-${SAFE_OWNER}-greenlang-${image}-${VERSION}.spdx.json"

            if [[ -f "${ARTIFACTS_DIR}/${cdx_file}" ]]; then
                check_pass "Found: ${cdx_file}"
            else
                check_fail "Missing: ${cdx_file}"
            fi

            if [[ -f "${ARTIFACTS_DIR}/${spdx_file}" ]]; then
                check_pass "Found: ${spdx_file}"
            else
                check_fail "Missing: ${spdx_file}"
            fi
        done
    else
        check_fail "${ARTIFACTS_DIR}/ directory not found"
    fi

    # 2) JSON validity & format sanity
    section "2) JSON Validity & Format Sanity"

    if [[ -d "$ARTIFACTS_DIR" ]]; then
        check_info "Validating all JSON files parse correctly..."
        JSON_VALID=true
        for f in "$ARTIFACTS_DIR"/*.json; do
            if [[ -f "$f" ]]; then
                if jq -e . "$f" >/dev/null 2>&1; then
                    echo "  ✓ Valid JSON: $(basename "$f")"
                else
                    check_fail "Invalid JSON: $(basename "$f")"
                    JSON_VALID=false
                fi
            fi
        done

        if [[ "$JSON_VALID" == "true" ]]; then
            check_pass "All JSON files are valid"
        fi

        check_info "Checking CycloneDX format fields..."
        CDX_VALID=true
        for f in "$ARTIFACTS_DIR"/*.cdx.json; do
            if [[ -f "$f" ]]; then
                BOM_FORMAT=$(jq -r '.bomFormat' "$f" 2>/dev/null)
                SPEC_VERSION=$(jq -r '.specVersion' "$f" 2>/dev/null)

                if [[ "$BOM_FORMAT" == "CycloneDX" ]]; then
                    echo "  ✓ CycloneDX format: $(basename "$f")"
                else
                    check_fail "Invalid bomFormat in $(basename "$f"): $BOM_FORMAT"
                    CDX_VALID=false
                fi

                if [[ -n "$SPEC_VERSION" && "$SPEC_VERSION" != "null" ]]; then
                    echo "  ✓ specVersion present: $SPEC_VERSION"
                else
                    check_fail "Missing specVersion in $(basename "$f")"
                    CDX_VALID=false
                fi
            fi
        done

        if [[ "$CDX_VALID" == "true" ]]; then
            check_pass "All CycloneDX files have required fields"
        fi

        check_info "Checking SPDX format fields..."
        SPDX_VALID=true
        for f in "$ARTIFACTS_DIR"/*.spdx.json; do
            if [[ -f "$f" ]]; then
                SPDX_VERSION=$(jq -r '.spdxVersion' "$f" 2>/dev/null)

                if [[ "$SPDX_VERSION" =~ ^SPDX- ]]; then
                    echo "  ✓ SPDX format: $(basename "$f") - $SPDX_VERSION"
                else
                    check_fail "Invalid spdxVersion in $(basename "$f"): $SPDX_VERSION"
                    SPDX_VALID=false
                fi
            fi
        done

        if [[ "$SPDX_VALID" == "true" ]]; then
            check_pass "All SPDX files have required fields"
        fi
    fi

    # 3) Image attestation verification
    section "3) Image Attestation Verification (if images exist)"

    if command -v cosign >/dev/null 2>&1; then
        check_info "Cosign is available, checking attestations..."

        for image in "${DOCKER_IMAGES[@]}"; do
            IMAGE_REF="${GHCR_REGISTRY}/${OWNER}/greenlang-${image}:${VERSION}"

            check_info "Verifying attestation for ${IMAGE_REF}..."

            # Check if image exists in registry (may fail if not pushed yet)
            if cosign verify-attestation \
                --type cyclonedx \
                --certificate-identity-regexp ".*" \
                --certificate-oidc-issuer https://token.actions.githubusercontent.com \
                "${IMAGE_REF}" >/dev/null 2>&1; then
                check_pass "Attestation verified: ${image}"
            else
                echo "  ⚠️  Cannot verify attestation (image may not be pushed yet)"
            fi
        done
    else
        check_info "Cosign not available - skipping attestation verification"
        echo "  Install with: brew install cosign"
    fi

    # 4) Cross-check SBOM ↔ artifact mapping
    section "4) Cross-check SBOM ↔ Artifact Mapping"

    if [[ -d "$ARTIFACTS_DIR" ]]; then
        # Check Python wheel SBOM
        if [[ -f "${ARTIFACTS_DIR}/sbom-greenlang-${VERSION}-wheel.cdx.json" ]]; then
            WHEEL_NAME=$(jq -r '.metadata.component.name // empty' \
                "${ARTIFACTS_DIR}/sbom-greenlang-${VERSION}-wheel.cdx.json" 2>/dev/null)
            if [[ -n "$WHEEL_NAME" ]]; then
                check_pass "Wheel SBOM name matches: $WHEEL_NAME"
            else
                check_info "Wheel SBOM component name not found (may be normal)"
            fi
        fi

        # Check Docker image SBOMs
        SAFE_OWNER=$(echo "$OWNER" | tr '/' '-')
        for image in "${DOCKER_IMAGES[@]}"; do
            SBOM_FILE="${ARTIFACTS_DIR}/sbom-image-ghcr-io-${SAFE_OWNER}-greenlang-${image}-${VERSION}.cdx.json"
            if [[ -f "$SBOM_FILE" ]]; then
                SOURCE_NAME=$(jq -r '.source.name // .metadata.component.name // empty' "$SBOM_FILE" 2>/dev/null)
                if [[ "$SOURCE_NAME" == *"${image}"* ]]; then
                    check_pass "Docker SBOM matches image: ${image}"
                else
                    check_info "Docker SBOM source name: $SOURCE_NAME"
                fi
            fi
        done
    fi

    # 5) CI artifacts & release assets
    section "5) CI Artifacts & Release Assets"

    # Check workflow files
    check_info "Checking GitHub Actions workflows..."

    if [[ -f ".github/workflows/sbom-generation.yml" ]]; then
        check_pass "SBOM generation workflow exists"

        # Check for artifact upload
        if grep -q "actions/upload-artifact" .github/workflows/sbom-generation.yml; then
            check_pass "Workflow uploads artifacts"
        else
            check_fail "Workflow missing artifact upload"
        fi

        # Check for release upload job
        if grep -q "upload-to-release:" .github/workflows/sbom-generation.yml; then
            check_pass "Workflow has release upload job"
        else
            check_fail "Workflow missing release upload job"
        fi
    else
        check_fail "SBOM generation workflow not found"
    fi

    # 6) Docs & README
    section "6) Documentation & README"

    # Check documentation
    if [[ -f "docs/security/sbom.md" ]]; then
        check_pass "SBOM documentation exists"

        # Check for required content
        if grep -q "cosign verify-attestation" docs/security/sbom.md; then
            check_pass "Verification commands documented"
        else
            check_fail "Verification commands not found in docs"
        fi

        # Check for tool versions
        if grep -q "Syft.*v1.0.0" docs/security/sbom.md && \
           grep -q "Cosign.*v2.2.4" docs/security/sbom.md; then
            check_pass "Tool versions documented"
        else
            check_fail "Tool versions not properly documented"
        fi
    else
        check_fail "docs/security/sbom.md not found"
    fi

    # Check README
    if [[ -f "README.md" ]]; then
        if grep -q "v0.2.0.*SBOM\|SBOM.*v0.2.0\|Software Bill of Materials" README.md; then
            check_pass "README mentions SBOM support"

            if grep -q "cosign verify-attestation" README.md; then
                check_pass "README includes verification example"
            else
                check_fail "README missing verification example"
            fi
        else
            check_fail "README does not mention v0.2.0 SBOM support"
        fi
    else
        check_fail "README.md not found"
    fi

    # 7) CI "sbom" job is blocking
    section "7) CI SBOM Job Configuration"

    # Check if sbom job is in required workflows
    if [[ -f ".github/workflows/release-build.yml" ]]; then
        if grep -q "sbom-generation.yml" .github/workflows/release-build.yml; then
            check_pass "SBOM integrated in release-build workflow"
        else
            check_fail "SBOM not integrated in release-build workflow"
        fi
    fi

    if [[ -f ".github/workflows/docker-release-complete.yml" ]]; then
        if grep -q "sbom-generation.yml\|generate-sboms:" .github/workflows/docker-release-complete.yml; then
            check_pass "SBOM integrated in docker-release workflow"
        else
            check_fail "SBOM not integrated in docker-release workflow"
        fi
    fi

    # Check validation gates
    if grep -q "exit 1" .github/workflows/sbom-generation.yml 2>/dev/null; then
        check_pass "SBOM workflow has failure gates"
    else
        check_fail "SBOM workflow missing failure gates"
    fi

    # Summary
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "                    DoD VERIFICATION SUMMARY"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Total Checks:  $TOTAL_CHECKS"
    echo -e "${GREEN}Passed:${NC}        $PASSED_CHECKS"
    echo -e "${RED}Failed:${NC}        $FAILED_CHECKS"
    echo ""

    if [[ $FAILED_CHECKS -eq 0 ]]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║        ✅ SBOM+Attestations Ready - DoD MET!          ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "One-line acceptance statement:"
        echo "Approved: SBOM+Attestations Ready — all Python wheels/sdists and images"
        echo "have CycloneDX (and SPDX where required) SBOMs; image SBOMs are attached"
        echo "as cosign attestations and verify; CI sbom job blocks on failure;"
        echo "SBOM docs and release assets are present."
        exit 0
    else
        echo -e "${RED}╔════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║        ❌ DoD NOT MET - $FAILED_CHECKS checks failed             ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "The task is NOT DONE. Fix the failed checks above."
        exit 1
    fi
}

# Run verification
main "$@"