#!/bin/bash
# ==============================================================================
# GL-VCCI Smoke Test Script
# ==============================================================================

set -e

NAMESPACE="$1"

if [ -z "$NAMESPACE" ]; then
    echo "[ERROR] Namespace is required"
    echo "Usage: $0 <namespace>"
    exit 1
fi

echo "[INFO] Running smoke tests for namespace: $NAMESPACE"

# Get backend API pod
BACKEND_POD=$(kubectl get pods -n "$NAMESPACE" -l app=vcci-backend-api -o jsonpath='{.items[0].metadata.name}')

if [ -z "$BACKEND_POD" ]; then
    echo "[ERROR] No backend API pod found"
    exit 1
fi

echo "[INFO] Testing backend pod: $BACKEND_POD"

# Test 1: Health check
echo "[TEST] Health check..."
if kubectl exec -n "$NAMESPACE" "$BACKEND_POD" -- curl -sf http://localhost:8000/health/live > /dev/null; then
    echo "[PASS] Liveness check passed"
else
    echo "[FAIL] Liveness check failed"
    exit 1
fi

if kubectl exec -n "$NAMESPACE" "$BACKEND_POD" -- curl -sf http://localhost:8000/health/ready > /dev/null; then
    echo "[PASS] Readiness check passed"
else
    echo "[FAIL] Readiness check failed"
    exit 1
fi

# Test 2: API endpoint
echo "[TEST] API endpoint..."
if kubectl exec -n "$NAMESPACE" "$BACKEND_POD" -- curl -sf http://localhost:8000/api/v1/public/health > /dev/null; then
    echo "[PASS] API endpoint accessible"
else
    echo "[FAIL] API endpoint not accessible"
    exit 1
fi

# Test 3: Database connectivity
echo "[TEST] Database connectivity..."
if kubectl exec -n "$NAMESPACE" "$BACKEND_POD" -- python -c "from sqlalchemy import create_engine; import os; engine = create_engine(os.getenv('DATABASE_URL')); engine.connect()" 2>/dev/null; then
    echo "[PASS] Database connection successful"
else
    echo "[FAIL] Database connection failed"
    exit 1
fi

# Test 4: Redis connectivity
echo "[TEST] Redis connectivity..."
if kubectl exec -n "$NAMESPACE" "$BACKEND_POD" -- python -c "import redis; import os; r = redis.from_url(os.getenv('REDIS_URL')); r.ping()" 2>/dev/null; then
    echo "[PASS] Redis connection successful"
else
    echo "[FAIL] Redis connection failed"
    exit 1
fi

echo "[INFO] All smoke tests passed!"
exit 0
