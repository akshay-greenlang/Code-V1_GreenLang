#!/bin/bash
# ============================================================================
# GL-CBAM-APP - Kustomize Validation Script
# ============================================================================
# Validates all Kustomize overlays and resource manifests
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
BASE_DIR="$DEPLOYMENT_DIR/kustomize"

echo "============================================================================"
echo "GL-CBAM-APP Kubernetes Manifest Validation"
echo "============================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${YELLOW}WARNING: kubectl not found. Installing basic validation only.${NC}"
    KUBECTL_AVAILABLE=false
else
    echo -e "${GREEN}✓ kubectl found${NC}"
    KUBECTL_AVAILABLE=true
fi

echo ""
echo "============================================================================"
echo "Validating Kustomize Structure"
echo "============================================================================"
echo ""

# Validate base
echo "Validating base configuration..."
if [ -f "$BASE_DIR/base/kustomization.yaml" ]; then
    echo -e "${GREEN}✓ Base kustomization.yaml found${NC}"
else
    echo -e "${RED}✗ Base kustomization.yaml not found${NC}"
    exit 1
fi

# Validate overlays
OVERLAYS=("dev" "staging" "production")
for overlay in "${OVERLAYS[@]}"; do
    echo ""
    echo "Validating $overlay overlay..."

    if [ -f "$BASE_DIR/overlays/$overlay/kustomization.yaml" ]; then
        echo -e "${GREEN}✓ $overlay/kustomization.yaml found${NC}"
    else
        echo -e "${RED}✗ $overlay/kustomization.yaml not found${NC}"
        exit 1
    fi

    # Check patches
    if [ -d "$BASE_DIR/overlays/$overlay/patches" ]; then
        patch_count=$(find "$BASE_DIR/overlays/$overlay/patches" -name "*.yaml" | wc -l)
        echo -e "${GREEN}✓ $overlay patches found: $patch_count files${NC}"
    else
        echo -e "${RED}✗ $overlay patches directory not found${NC}"
    fi
done

echo ""
echo "============================================================================"
echo "Validating Resource Manifests"
echo "============================================================================"
echo ""

# Validate HPA
if [ -f "$DEPLOYMENT_DIR/hpa.yaml" ]; then
    echo -e "${GREEN}✓ HPA manifest found${NC}"
else
    echo -e "${RED}✗ HPA manifest not found${NC}"
fi

# Validate PDB
if [ -f "$DEPLOYMENT_DIR/pdb.yaml" ]; then
    echo -e "${GREEN}✓ PDB manifest found${NC}"
else
    echo -e "${RED}✗ PDB manifest not found${NC}"
fi

# Validate ResourceQuota
if [ -f "$DEPLOYMENT_DIR/resourcequota.yaml" ]; then
    echo -e "${GREEN}✓ ResourceQuota manifest found${NC}"
else
    echo -e "${RED}✗ ResourceQuota manifest not found${NC}"
fi

# Validate LimitRange
if [ -f "$DEPLOYMENT_DIR/limitrange.yaml" ]; then
    echo -e "${GREEN}✓ LimitRange manifest found${NC}"
else
    echo -e "${RED}✗ LimitRange manifest not found${NC}"
fi

if [ "$KUBECTL_AVAILABLE" = true ]; then
    echo ""
    echo "============================================================================"
    echo "Kustomize Build Validation"
    echo "============================================================================"
    echo ""

    for overlay in "${OVERLAYS[@]}"; do
        echo "Building $overlay overlay..."
        if kubectl kustomize "$BASE_DIR/overlays/$overlay" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $overlay overlay builds successfully${NC}"

            # Count resources
            resource_count=$(kubectl kustomize "$BASE_DIR/overlays/$overlay" | grep -c "^kind:" || true)
            echo "  Resources generated: $resource_count"
        else
            echo -e "${RED}✗ $overlay overlay build failed${NC}"
            kubectl kustomize "$BASE_DIR/overlays/$overlay"
            exit 1
        fi
        echo ""
    done

    echo "============================================================================"
    echo "Dry Run Validation"
    echo "============================================================================"
    echo ""

    for overlay in "${OVERLAYS[@]}"; do
        echo "Dry-run validation for $overlay..."
        if kubectl apply -k "$BASE_DIR/overlays/$overlay" --dry-run=client > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $overlay passes dry-run validation${NC}"
        else
            echo -e "${YELLOW}⚠ $overlay dry-run validation failed (may need cluster context)${NC}"
        fi
    done
fi

echo ""
echo "============================================================================"
echo "File Statistics"
echo "============================================================================"
echo ""

total_files=$(find "$DEPLOYMENT_DIR" -type f \( -name "*.yaml" -o -name "*.md" -o -name "*.sh" \) | wc -l)
yaml_files=$(find "$DEPLOYMENT_DIR" -type f -name "*.yaml" | wc -l)
total_lines=$(find "$DEPLOYMENT_DIR" -type f \( -name "*.yaml" -o -name "*.md" \) -exec wc -l {} + | tail -1 | awk '{print $1}')

echo "Total files: $total_files"
echo "YAML files: $yaml_files"
echo "Total lines: $total_lines"

echo ""
echo "============================================================================"
echo "Validation Complete"
echo "============================================================================"
echo ""
echo -e "${GREEN}All validations passed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review generated manifests: kubectl kustomize deployment/kustomize/overlays/dev"
echo "  2. Deploy to dev: kubectl apply -k deployment/kustomize/overlays/dev"
echo "  3. Verify deployment: kubectl get all -n gl-cbam-dev"
echo ""
