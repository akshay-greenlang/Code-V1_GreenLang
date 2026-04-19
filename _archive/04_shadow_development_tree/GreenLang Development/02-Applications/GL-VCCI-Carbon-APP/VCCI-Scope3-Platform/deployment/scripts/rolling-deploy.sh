#!/bin/bash
# ==============================================================================
# GL-VCCI Rolling Deployment Script
# ==============================================================================

set -e

ENVIRONMENT="$1"
VERSION="$2"
DRY_RUN="$3"

NAMESPACE="vcci-${ENVIRONMENT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "[INFO] Starting rolling deployment for $NAMESPACE with version $VERSION"

# Use kustomize to apply environment-specific configurations
cd "$PROJECT_ROOT/k8s/overlays/${ENVIRONMENT}"

# Update image tags in kustomization.yaml
kustomize edit set image \
    YOUR_REGISTRY/vcci-backend:${VERSION} \
    YOUR_REGISTRY/vcci-worker:${VERSION}

# Apply with kustomize
echo "[INFO] Applying manifests with kustomize..."
kubectl apply -k . $DRY_RUN

if [ -z "$DRY_RUN" ]; then
    # Wait for rollout to complete
    echo "[INFO] Waiting for backend-api rollout..."
    kubectl rollout status deployment/vcci-backend-api -n "$NAMESPACE" --timeout=10m

    echo "[INFO] Waiting for worker rollout..."
    kubectl rollout status deployment/vcci-worker -n "$NAMESPACE" --timeout=10m

    echo "[INFO] Rolling deployment completed successfully!"
else
    echo "[INFO] Dry run complete"
fi
