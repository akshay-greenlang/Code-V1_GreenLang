#!/bin/bash
# ==============================================================================
# GL-VCCI Canary Deployment Script
# ==============================================================================

set -e

ENVIRONMENT="$1"
VERSION="$2"
DRY_RUN="$3"

NAMESPACE="vcci-${ENVIRONMENT}"
APP="vcci-backend-api"

echo "[INFO] Starting canary deployment for $APP:$VERSION"

# Deploy canary
echo "[INFO] Deploying canary (5% traffic)..."
kubectl set image deployment/${APP}-canary \
    ${APP}=YOUR_REGISTRY/${APP}:${VERSION} -n "$NAMESPACE" $DRY_RUN

if [ -z "$DRY_RUN" ]; then
    # Wait for canary to be ready
    kubectl rollout status deployment/${APP}-canary -n "$NAMESPACE" --timeout=5m

    # Stage 1: 5%
    echo "[INFO] Stage 1: 5% traffic to canary"
    kubectl scale deployment/${APP}-canary --replicas=1 -n "$NAMESPACE"
    kubectl scale deployment/${APP}-stable --replicas=19 -n "$NAMESPACE"
    sleep 600  # Monitor for 10 minutes

    # Check metrics
    if ! bash "$(dirname "$0")/check-canary-metrics.sh" "$NAMESPACE" "$APP"; then
        echo "[ERROR] Canary metrics unhealthy, rolling back"
        kubectl scale deployment/${APP}-canary --replicas=0 -n "$NAMESPACE"
        kubectl scale deployment/${APP}-stable --replicas=20 -n "$NAMESPACE"
        exit 1
    fi

    # Stage 2: 10%
    echo "[INFO] Stage 2: 10% traffic to canary"
    kubectl scale deployment/${APP}-canary --replicas=2 -n "$NAMESPACE"
    kubectl scale deployment/${APP}-stable --replicas=18 -n "$NAMESPACE"
    sleep 600

    # Stage 3: 25%
    echo "[INFO] Stage 3: 25% traffic to canary"
    kubectl scale deployment/${APP}-canary --replicas=5 -n "$NAMESPACE"
    kubectl scale deployment/${APP}-stable --replicas=15 -n "$NAMESPACE"
    sleep 600

    # Stage 4: 50%
    echo "[INFO] Stage 4: 50% traffic to canary"
    kubectl scale deployment/${APP}-canary --replicas=10 -n "$NAMESPACE"
    kubectl scale deployment/${APP}-stable --replicas=10 -n "$NAMESPACE"
    sleep 600

    # Stage 5: 75%
    echo "[INFO] Stage 5: 75% traffic to canary"
    kubectl scale deployment/${APP}-canary --replicas=15 -n "$NAMESPACE"
    kubectl scale deployment/${APP}-stable --replicas=5 -n "$NAMESPACE"
    sleep 600

    # Stage 6: 100% (Promote)
    echo "[INFO] Stage 6: Promoting canary to 100%"
    kubectl scale deployment/${APP}-canary --replicas=20 -n "$NAMESPACE"
    kubectl scale deployment/${APP}-stable --replicas=0 -n "$NAMESPACE"

    # Update stable to new version
    kubectl set image deployment/${APP}-stable \
        ${APP}=YOUR_REGISTRY/${APP}:${VERSION} -n "$NAMESPACE"

    echo "[INFO] Canary deployment completed successfully!"
else
    echo "[INFO] Dry run complete"
fi
