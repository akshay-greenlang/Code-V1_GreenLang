#!/bin/bash

# CI-specific security verification script
# This is a simplified version for use in GitHub Actions

set -euo pipefail

VERSION="${1:-}"
REGISTRY="${REGISTRY:-ghcr.io}"
REPO_OWNER="${GITHUB_REPOSITORY_OWNER:-$(git config --get remote.origin.url | sed 's/.*github\.com[:/]\([^/]*\).*/\1/')}"

if [[ -z "$VERSION" ]]; then
    echo "❌ ERROR: Version is required"
    echo "Usage: $0 <version>"
    exit 1
fi

echo "🔍 Verifying security artifacts for GreenLang v$VERSION"
echo "📦 Registry: $REGISTRY"
echo "👤 Owner: $REPO_OWNER"
echo ""

RUNNER_IMAGE="$REGISTRY/$REPO_OWNER/greenlang-runner:$VERSION"
FULL_IMAGE="$REGISTRY/$REPO_OWNER/greenlang-full:$VERSION"

# Function to check if image exists
check_image_exists() {
    local image="$1"
    if docker manifest inspect "$image" >/dev/null 2>&1; then
        echo "✅ Image exists: $image"
        return 0
    else
        echo "❌ Image not found: $image"
        return 1
    fi
}

# Function to verify cosign signature
verify_cosign() {
    local image="$1"
    local image_type="$2"

    echo "🔐 Verifying cosign signature for $image_type image..."

    if cosign verify "$image" \
        --certificate-identity-regexp "https://github.com/$REPO_OWNER/.*/.github/workflows/.*" \
        --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
        >/dev/null 2>&1; then
        echo "✅ Cosign signature verified for $image_type image"
        return 0
    else
        echo "❌ Cosign signature verification failed for $image_type image"
        return 1
    fi
}

# Function to verify SBOM
verify_sbom() {
    local image="$1"
    local image_type="$2"

    echo "📋 Verifying SBOM for $image_type image..."

    local sbom_output
    if sbom_output=$(cosign download sbom "$image" 2>/dev/null); then
        if echo "$sbom_output" | jq -e '.spdxVersion and .creationInfo and .packages' >/dev/null 2>&1; then
            local package_count=$(echo "$sbom_output" | jq '.packages | length')
            echo "✅ SBOM verified for $image_type image ($package_count packages)"
            return 0
        else
            echo "⚠️ SBOM found but format validation failed for $image_type image"
            return 1
        fi
    else
        echo "❌ No SBOM attachment found for $image_type image"
        return 1
    fi
}

# Function to test image functionality
test_image() {
    local image="$1"
    local image_type="$2"

    echo "🧪 Testing $image_type image functionality..."

    if [[ "$image_type" == "runner" ]]; then
        if timeout 30 docker run --rm "$image" --version >/dev/null 2>&1; then
            echo "✅ Runner image functionality test passed"
            return 0
        else
            echo "❌ Runner image functionality test failed"
            return 1
        fi
    elif [[ "$image_type" == "full" ]]; then
        if timeout 30 docker run --rm "$image" gl --version >/dev/null 2>&1; then
            echo "✅ Full image functionality test passed"
            return 0
        else
            echo "❌ Full image functionality test failed"
            return 1
        fi
    fi
}

# Main verification
echo "== Image Existence Check =="
RUNNER_EXISTS=false
FULL_EXISTS=false

if check_image_exists "$RUNNER_IMAGE"; then
    RUNNER_EXISTS=true
fi

if check_image_exists "$FULL_IMAGE"; then
    FULL_EXISTS=true
fi

echo ""

# Verify runner image
if [[ "$RUNNER_EXISTS" == "true" ]]; then
    echo "== Runner Image Security Verification =="
    verify_cosign "$RUNNER_IMAGE" "runner"
    verify_sbom "$RUNNER_IMAGE" "runner"
    test_image "$RUNNER_IMAGE" "runner"
    echo ""
fi

# Verify full image
if [[ "$FULL_EXISTS" == "true" ]]; then
    echo "== Full Image Security Verification =="
    verify_cosign "$FULL_IMAGE" "full"
    verify_sbom "$FULL_IMAGE" "full"
    test_image "$FULL_IMAGE" "full"
    echo ""
fi

# Summary
echo "== Verification Summary =="
if [[ "$RUNNER_EXISTS" == "true" && "$FULL_EXISTS" == "true" ]]; then
    echo "✅ All security artifact verifications completed"
    echo "🎉 GreenLang v$VERSION Docker images are properly secured!"
else
    echo "⚠️ Some images were not found, partial verification completed"
fi

echo ""
echo "🔗 For detailed verification, use: ./scripts/verify-security-artifacts.sh $VERSION"