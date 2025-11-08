#!/bin/bash
# ==============================================================================
# GL-VCCI Blue-Green Deployment Script
# ==============================================================================

set -e

ENVIRONMENT="$1"
VERSION="$2"
DRY_RUN="$3"

NAMESPACE="vcci-${ENVIRONMENT}"

echo "[INFO] Starting blue-green deployment"

# Determine current color
CURRENT_COLOR=$(kubectl get service vcci-backend-api -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")

if [ "$CURRENT_COLOR" == "blue" ]; then
    NEW_COLOR="green"
    OLD_COLOR="blue"
else
    NEW_COLOR="blue"
    OLD_COLOR="green"
fi

echo "[INFO] Current active: $OLD_COLOR"
echo "[INFO] Deploying to: $NEW_COLOR"

# Deploy new version to inactive color
echo "[INFO] Deploying version $VERSION to $NEW_COLOR environment..."
kubectl set image deployment/vcci-backend-api-${NEW_COLOR} \
    vcci-backend-api=YOUR_REGISTRY/vcci-backend:${VERSION} \
    -n "$NAMESPACE" $DRY_RUN

if [ -z "$DRY_RUN" ]; then
    # Wait for deployment
    kubectl rollout status deployment/vcci-backend-api-${NEW_COLOR} -n "$NAMESPACE" --timeout=10m

    # Run smoke tests on new environment
    echo "[INFO] Running smoke tests on $NEW_COLOR..."
    kubectl port-forward service/vcci-backend-api-${NEW_COLOR} 8001:8000 -n "$NAMESPACE" &
    PF_PID=$!
    sleep 5

    if curl -f http://localhost:8001/health/live; then
        echo "[INFO] Health check passed on $NEW_COLOR"
        kill $PF_PID
    else
        echo "[ERROR] Health check failed on $NEW_COLOR"
        kill $PF_PID
        exit 1
    fi

    # Confirm switch
    read -p "Switch traffic to $NEW_COLOR? (yes/no): " CONFIRM
    if [ "$CONFIRM" == "yes" ]; then
        echo "[INFO] Switching traffic to $NEW_COLOR..."
        kubectl patch service vcci-backend-api -n "$NAMESPACE" \
            -p "{\"spec\":{\"selector\":{\"color\":\"$NEW_COLOR\"}}}"

        echo "[INFO] Traffic switched successfully!"

        # Monitor for 5 minutes
        echo "[INFO] Monitoring $NEW_COLOR for 5 minutes..."
        sleep 300

        # Scale down old environment
        read -p "Scale down $OLD_COLOR? (yes/no): " CONFIRM2
        if [ "$CONFIRM2" == "yes" ]; then
            kubectl scale deployment/vcci-backend-api-${OLD_COLOR} --replicas=0 -n "$NAMESPACE"
            echo "[INFO] Scaled down $OLD_COLOR environment"
        fi
    else
        echo "[INFO] Deployment cancelled"
        exit 1
    fi
else
    echo "[INFO] Dry run complete"
fi
