#!/bin/bash
# ==============================================================================
# GL-VCCI Canary Metrics Check Script
# ==============================================================================

NAMESPACE="$1"
APP="$2"

if [ -z "$NAMESPACE" ] || [ -z "$APP" ]; then
    echo "[ERROR] Usage: $0 <namespace> <app>"
    exit 1
fi

echo "[INFO] Checking canary metrics for $APP in $NAMESPACE"

# Get canary and stable pod counts
CANARY_PODS=$(kubectl get pods -n "$NAMESPACE" -l app="$APP",track=canary -o name | wc -l)
STABLE_PODS=$(kubectl get pods -n "$NAMESPACE" -l app="$APP",track=stable -o name | wc -l)

echo "[INFO] Canary pods: $CANARY_PODS, Stable pods: $STABLE_PODS"

# Check if Prometheus is available
if ! kubectl get svc prometheus -n monitoring &>/dev/null; then
    echo "[WARN] Prometheus not found, skipping metric checks"
    echo "[INFO] Performing basic health checks only..."

    # Check pod status
    CANARY_READY=$(kubectl get pods -n "$NAMESPACE" -l app="$APP",track=canary -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -c "True" || echo "0")

    if [ "$CANARY_READY" -eq "$CANARY_PODS" ]; then
        echo "[PASS] All canary pods are ready"
        exit 0
    else
        echo "[FAIL] Some canary pods are not ready"
        exit 1
    fi
fi

# Query Prometheus for error rates
PROMETHEUS_URL="http://prometheus.monitoring.svc.cluster.local:9090"

# Canary error rate
CANARY_ERROR_RATE=$(curl -s "${PROMETHEUS_URL}/api/v1/query" \
    --data-urlencode "query=rate(http_requests_total{app=\"$APP\",track=\"canary\",status=~\"5..\"}[5m])/rate(http_requests_total{app=\"$APP\",track=\"canary\"}[5m])" \
    | jq -r '.data.result[0].value[1] // 0')

# Stable error rate
STABLE_ERROR_RATE=$(curl -s "${PROMETHEUS_URL}/api/v1/query" \
    --data-urlencode "query=rate(http_requests_total{app=\"$APP\",track=\"stable\",status=~\"5..\"}[5m])/rate(http_requests_total{app=\"$APP\",track=\"stable\"}[5m])" \
    | jq -r '.data.result[0].value[1] // 0')

echo "[INFO] Canary error rate: $CANARY_ERROR_RATE"
echo "[INFO] Stable error rate: $STABLE_ERROR_RATE"

# Check if canary error rate is more than 2x stable
THRESHOLD=$(echo "$STABLE_ERROR_RATE * 2" | bc -l)

if (( $(echo "$CANARY_ERROR_RATE > $THRESHOLD" | bc -l) )); then
    echo "[FAIL] Canary error rate is more than 2x stable error rate"
    exit 1
fi

# Check response time
CANARY_P99=$(curl -s "${PROMETHEUS_URL}/api/v1/query" \
    --data-urlencode "query=histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{app=\"$APP\",track=\"canary\"}[5m]))" \
    | jq -r '.data.result[0].value[1] // 0')

STABLE_P99=$(curl -s "${PROMETHEUS_URL}/api/v1/query" \
    --data-urlencode "query=histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{app=\"$APP\",track=\"stable\"}[5m]))" \
    | jq -r '.data.result[0].value[1] // 0')

echo "[INFO] Canary p99 latency: ${CANARY_P99}s"
echo "[INFO] Stable p99 latency: ${STABLE_P99}s"

# Check if canary latency is more than 1.5x stable
LATENCY_THRESHOLD=$(echo "$STABLE_P99 * 1.5" | bc -l)

if (( $(echo "$CANARY_P99 > $LATENCY_THRESHOLD" | bc -l) )); then
    echo "[FAIL] Canary latency is more than 1.5x stable latency"
    exit 1
fi

echo "[PASS] Canary metrics are healthy"
exit 0
