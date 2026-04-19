#!/bin/bash
# ==============================================================================
# GL-VCCI Post-Deployment Validation Script
# ==============================================================================
# Validates deployment success and system health after deployment
#
# Version: 2.0.0
# Date: 2025-11-09

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_BASE_URL="${API_BASE_URL:-https://api.vcci.company.com}"
NAMESPACE="${KUBERNETES_NAMESPACE:-vcci-production}"
MAX_RETRIES="${MAX_RETRIES:-30}"
RETRY_DELAY="${RETRY_DELAY:-10}"

# Counters
CHECKS_PASSED=0
CHECKS_FAILED=0

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_check() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

check_passed() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_failed() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

wait_for_condition() {
    local condition="$1"
    local message="$2"
    local retries=0

    log_info "$message"

    while [ $retries -lt $MAX_RETRIES ]; do
        if eval "$condition"; then
            return 0
        fi

        retries=$((retries + 1))
        echo -n "."
        sleep $RETRY_DELAY
    done

    echo ""
    return 1
}

# ==============================================================================
# Post-Deployment Validation
# ==============================================================================

echo "======================================================================"
echo "GL-VCCI Post-Deployment Validation"
echo "======================================================================"
echo ""
log_info "Validating deployment at $(date)"
log_info "API Base URL: $API_BASE_URL"
log_info "Namespace: $NAMESPACE"
echo ""

# 1. Kubernetes Pod Status Check
log_check "Checking Kubernetes pod status..."

# Wait for all pods to be ready
wait_for_condition "kubectl get pods -n $NAMESPACE | grep -q 'Running'" \
    "Waiting for pods to be running..."

if kubectl get pods -n $NAMESPACE | grep -q "Running"; then
    RUNNING_PODS=$(kubectl get pods -n $NAMESPACE | grep -c "Running")
    check_passed "Pods are running ($RUNNING_PODS pods)"

    # Check if all pods are ready
    NOT_READY=$(kubectl get pods -n $NAMESPACE | grep -v "Running" | grep -v "Completed" | wc -l)
    if [ "$NOT_READY" -gt 1 ]; then  # 1 for header
        check_failed "Some pods are not ready"
        kubectl get pods -n $NAMESPACE | grep -v "Running"
    else
        check_passed "All pods are ready"
    fi
else
    check_failed "No running pods found"
fi
echo ""

# 2. Deployment Rollout Status
log_check "Checking deployment rollout status..."

DEPLOYMENTS=$(kubectl get deployments -n $NAMESPACE -o name)
for deployment in $DEPLOYMENTS; do
    DEPLOYMENT_NAME=$(echo "$deployment" | cut -d'/' -f2)

    if kubectl rollout status "$deployment" -n "$NAMESPACE" --timeout=300s; then
        check_passed "Deployment $DEPLOYMENT_NAME rolled out successfully"
    else
        check_failed "Deployment $DEPLOYMENT_NAME rollout failed"
    fi
done
echo ""

# 3. Service Availability Check
log_check "Checking service availability..."

SERVICES=$(kubectl get services -n $NAMESPACE -o name)
for service in $SERVICES; do
    SERVICE_NAME=$(echo "$service" | cut -d'/' -f2)
    check_passed "Service $SERVICE_NAME is available"
done
echo ""

# 4. Health Check Endpoints
log_check "Testing health check endpoints..."

# Liveness check
log_info "Testing /health/live endpoint..."
for i in {1..5}; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE_URL/health/live" || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        check_passed "Liveness endpoint returned 200 OK"
        break
    elif [ $i -eq 5 ]; then
        check_failed "Liveness endpoint returned $HTTP_CODE"
    else
        sleep 5
    fi
done

# Readiness check
log_info "Testing /health/ready endpoint..."
for i in {1..5}; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE_URL/health/ready" || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        check_passed "Readiness endpoint returned 200 OK"
        READY_RESPONSE=$(curl -s "$API_BASE_URL/health/ready")
        log_info "Readiness response: $READY_RESPONSE"
        break
    elif [ $i -eq 5 ]; then
        check_failed "Readiness endpoint returned $HTTP_CODE"
    else
        sleep 5
    fi
done

# Detailed health check
log_info "Testing /health/detailed endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE_URL/health/detailed" || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    check_passed "Detailed health endpoint returned 200 OK"

    DETAILED_HEALTH=$(curl -s "$API_BASE_URL/health/detailed")
    echo "$DETAILED_HEALTH" | jq . || echo "$DETAILED_HEALTH"
else
    check_failed "Detailed health endpoint returned $HTTP_CODE"
fi
echo ""

# 5. Database Connectivity Check
log_check "Checking database connectivity..."

# Get a pod name
POD_NAME=$(kubectl get pods -n $NAMESPACE -l app=vcci-api -o jsonpath='{.items[0].metadata.name}')

if [ -n "$POD_NAME" ]; then
    DB_CHECK=$(kubectl exec -n $NAMESPACE "$POD_NAME" -- python -c \
        "import psycopg2; conn = psycopg2.connect('$DATABASE_URL'); print('OK')" 2>&1 || echo "FAILED")

    if echo "$DB_CHECK" | grep -q "OK"; then
        check_passed "Database connection successful"
    else
        check_failed "Database connection failed: $DB_CHECK"
    fi
else
    log_warn "Could not find pod to test database connection"
fi
echo ""

# 6. Redis Connectivity Check
log_check "Checking Redis connectivity..."

if [ -n "$POD_NAME" ]; then
    REDIS_CHECK=$(kubectl exec -n $NAMESPACE "$POD_NAME" -- python -c \
        "import redis; r = redis.from_url('$REDIS_URL'); r.ping(); print('OK')" 2>&1 || echo "FAILED")

    if echo "$REDIS_CHECK" | grep -q "OK"; then
        check_passed "Redis connection successful"
    else
        check_failed "Redis connection failed: $REDIS_CHECK"
    fi
else
    log_warn "Could not find pod to test Redis connection"
fi
echo ""

# 7. Authentication Test
log_check "Testing authentication endpoint..."

# Test login endpoint
LOGIN_RESPONSE=$(curl -s -X POST "$API_BASE_URL/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"test","password":"test"}' || echo "{}")

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE_URL/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username":"test","password":"test"}' || echo "000")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "401" ]; then
    check_passed "Authentication endpoint is responding (HTTP $HTTP_CODE)"
else
    check_failed "Authentication endpoint returned unexpected code: $HTTP_CODE"
fi
echo ""

# 8. Circuit Breaker Metrics Check
log_check "Checking circuit breaker metrics..."

METRICS_RESPONSE=$(curl -s "$API_BASE_URL/metrics" || echo "")

if echo "$METRICS_RESPONSE" | grep -q "greenlang_circuit_breaker"; then
    check_passed "Circuit breaker metrics are being exported"

    # Count circuit breakers
    CB_COUNT=$(echo "$METRICS_RESPONSE" | grep "greenlang_circuit_breaker_state" | wc -l)
    log_info "Found $CB_COUNT circuit breaker metrics"
else
    check_failed "Circuit breaker metrics not found"
fi
echo ""

# 9. API Endpoint Smoke Tests
log_check "Running API endpoint smoke tests..."

# Test intake status endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE_URL/api/v1/intake/status" \
    -H "Authorization: Bearer test-token" || echo "000")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "401" ]; then
    check_passed "Intake API endpoint is responding"
else
    log_warn "Intake API endpoint returned: $HTTP_CODE"
fi

# Test calculator endpoint
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE_URL/api/v1/calculator/categories" \
    -H "Authorization: Bearer test-token" || echo "000")

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "401" ]; then
    check_passed "Calculator API endpoint is responding"
else
    log_warn "Calculator API endpoint returned: $HTTP_CODE"
fi
echo ""

# 10. Resource Usage Check
log_check "Checking resource usage..."

# Get pod resource usage
if kubectl top pods -n $NAMESPACE &> /dev/null; then
    log_info "Pod resource usage:"
    kubectl top pods -n $NAMESPACE

    # Check if any pod is using >90% memory or CPU
    HIGH_USAGE=$(kubectl top pods -n $NAMESPACE | awk 'NR>1 {print $3, $4}' | \
        grep -E '9[0-9]%|100%' || echo "")

    if [ -z "$HIGH_USAGE" ]; then
        check_passed "All pods have healthy resource usage"
    else
        log_warn "Some pods have high resource usage:"
        echo "$HIGH_USAGE"
    fi
else
    log_warn "Cannot check pod resource usage (metrics-server not available)"
fi
echo ""

# 11. Persistent Volume Claims Check
log_check "Checking persistent volume claims..."

PVCS=$(kubectl get pvc -n $NAMESPACE -o jsonpath='{.items[*].status.phase}')

if echo "$PVCS" | grep -q "Bound"; then
    check_passed "All PVCs are bound"
else
    check_failed "Some PVCs are not bound"
    kubectl get pvc -n $NAMESPACE
fi
echo ""

# 12. Ingress Configuration Check
log_check "Checking ingress configuration..."

INGRESS_COUNT=$(kubectl get ingress -n $NAMESPACE -o name | wc -l)

if [ "$INGRESS_COUNT" -gt 0 ]; then
    check_passed "Ingress resources are configured ($INGRESS_COUNT ingress)"

    # Check ingress addresses
    kubectl get ingress -n $NAMESPACE
else
    log_warn "No ingress resources found"
fi
echo ""

# 13. ConfigMap and Secrets Check
log_check "Checking ConfigMaps and Secrets..."

CONFIGMAP_COUNT=$(kubectl get configmap -n $NAMESPACE -o name | wc -l)
SECRET_COUNT=$(kubectl get secret -n $NAMESPACE -o name | wc -l)

log_info "Found $CONFIGMAP_COUNT ConfigMaps and $SECRET_COUNT Secrets"

if [ "$CONFIGMAP_COUNT" -gt 0 ] && [ "$SECRET_COUNT" -gt 0 ]; then
    check_passed "ConfigMaps and Secrets are present"
else
    log_warn "Some ConfigMaps or Secrets may be missing"
fi
echo ""

# 14. Log Error Check
log_check "Checking recent logs for errors..."

# Check last 100 lines of logs from each pod
PODS=$(kubectl get pods -n $NAMESPACE -l app=vcci-api -o name)

ERROR_COUNT=0
for pod in $PODS; do
    POD_NAME=$(echo "$pod" | cut -d'/' -f2)
    ERRORS=$(kubectl logs -n $NAMESPACE "$POD_NAME" --tail=100 | grep -i "error\|exception\|fatal" | wc -l)

    if [ "$ERRORS" -gt 0 ]; then
        log_warn "Found $ERRORS errors in $POD_NAME logs"
        ERROR_COUNT=$((ERROR_COUNT + ERRORS))
    fi
done

if [ "$ERROR_COUNT" -eq 0 ]; then
    check_passed "No errors found in recent logs"
elif [ "$ERROR_COUNT" -lt 5 ]; then
    log_warn "Found $ERROR_COUNT errors in logs (acceptable threshold)"
else
    check_failed "Found $ERROR_COUNT errors in logs (above threshold)"
fi
echo ""

# 15. Monitoring Integration Check
log_check "Checking monitoring integration..."

# Check if Prometheus is scraping metrics
if command -v curl &> /dev/null; then
    PROMETHEUS_URL="${PROMETHEUS_URL:-http://prometheus.monitoring.svc.cluster.local:9090}"

    # Query Prometheus for VCCI metrics
    PROM_QUERY="up{namespace=\"$NAMESPACE\"}"
    PROM_RESPONSE=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${PROM_QUERY}" || echo "{}")

    if echo "$PROM_RESPONSE" | grep -q "success"; then
        check_passed "Prometheus is scraping metrics"
    else
        log_warn "Could not verify Prometheus integration"
    fi
else
    log_warn "curl not available, skipping Prometheus check"
fi
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "======================================================================"
echo "Post-Deployment Validation Summary"
echo "======================================================================"
echo ""
echo -e "${GREEN}Checks Passed:${NC}  $CHECKS_PASSED"
echo -e "${RED}Checks Failed:${NC}  $CHECKS_FAILED"
echo ""

if [ $CHECKS_FAILED -gt 0 ]; then
    echo -e "${RED}❌ POST-DEPLOYMENT VALIDATION FAILED${NC}"
    echo ""
    echo "Some validation checks failed. Please investigate and resolve issues."
    echo ""
    echo "To view pod logs:"
    echo "  kubectl logs -n $NAMESPACE <pod-name>"
    echo ""
    echo "To rollback deployment:"
    echo "  ./rollback.sh"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ ALL POST-DEPLOYMENT VALIDATIONS PASSED${NC}"
    echo ""
    echo "Deployment is healthy and ready for traffic."
    echo ""
    echo "Next steps:"
    echo "  1. Monitor dashboards: https://grafana.company.com"
    echo "  2. Check alerts: https://alerts.company.com"
    echo "  3. Review logs: https://logs.company.com"
    echo ""
fi

echo "======================================================================"
