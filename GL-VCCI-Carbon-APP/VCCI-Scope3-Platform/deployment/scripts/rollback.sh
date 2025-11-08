#!/bin/bash
# ==============================================================================
# GL-VCCI Rollback Script
# ==============================================================================

set -e

ENVIRONMENT="$1"
REVISION="$2"

if [ -z "$ENVIRONMENT" ]; then
    echo "[ERROR] Environment is required"
    echo "Usage: $0 <environment> [revision]"
    exit 1
fi

NAMESPACE="vcci-${ENVIRONMENT}"

echo "[INFO] Rolling back deployments in $NAMESPACE"

# If no revision specified, rollback to previous
if [ -z "$REVISION" ]; then
    echo "[INFO] Rolling back to previous revision..."

    # Rollback backend API
    echo "[INFO] Rolling back backend-api..."
    kubectl rollout undo deployment/vcci-backend-api -n "$NAMESPACE"

    # Rollback worker
    echo "[INFO] Rolling back worker..."
    kubectl rollout undo deployment/vcci-worker -n "$NAMESPACE"
else
    echo "[INFO] Rolling back to revision $REVISION..."

    # Rollback to specific revision
    kubectl rollout undo deployment/vcci-backend-api -n "$NAMESPACE" --to-revision="$REVISION"
    kubectl rollout undo deployment/vcci-worker -n "$NAMESPACE" --to-revision="$REVISION"
fi

# Wait for rollback to complete
echo "[INFO] Waiting for rollback to complete..."
kubectl rollout status deployment/vcci-backend-api -n "$NAMESPACE" --timeout=10m
kubectl rollout status deployment/vcci-worker -n "$NAMESPACE" --timeout=10m

# Run smoke tests
echo "[INFO] Running smoke tests..."
if bash "$(dirname "$0")/smoke-test.sh" "$NAMESPACE"; then
    echo "[INFO] Rollback completed successfully!"
else
    echo "[ERROR] Smoke tests failed after rollback!"
    exit 1
fi

# Display current status
echo "[INFO] Current deployment status:"
kubectl get deployments -n "$NAMESPACE"
kubectl get pods -n "$NAMESPACE"

echo "[INFO] Rollback history:"
kubectl rollout history deployment/vcci-backend-api -n "$NAMESPACE"
