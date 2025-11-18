#!/bin/bash
# GL-002 Health Check Script
# Comprehensive health validation for deployments
# Usage: ./health-check.sh [ENVIRONMENT] [URL]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENVIRONMENT="${1:-staging}"
BASE_URL="${2:-}"
RETRIES=5
RETRY_INTERVAL=10
TIMEOUT=30

# Auto-detect URL if not provided
if [ -z "${BASE_URL}" ]; then
    case "${ENVIRONMENT}" in
        production|prod)
            BASE_URL="https://api.boiler.greenlang.io"
            ;;
        staging|stage)
            BASE_URL="https://api.staging.greenlang.io"
            ;;
        development|dev)
            BASE_URL="http://localhost:8000"
            ;;
        *)
            BASE_URL="http://localhost:8000"
            ;;
    esac
fi

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}GL-002 Health Check${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}URL: ${BASE_URL}${NC}"
echo -e "${BLUE}======================================${NC}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Track results
PASSED=0
FAILED=0
WARNINGS=0

# Make HTTP request with retries
http_request() {
    local endpoint=$1
    local description=$2
    local retries=${3:-1}

    log_info "Checking: ${description}"

    for i in $(seq 1 $retries); do
        if [ $retries -gt 1 ]; then
            log_info "Attempt ${i}/${retries}..."
        fi

        local response=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time $TIMEOUT \
            "${BASE_URL}${endpoint}" 2>/dev/null)

        if [ "${response}" = "200" ]; then
            log_pass "${description} - HTTP ${response}"
            ((PASSED++))
            return 0
        fi

        if [ $i -lt $retries ]; then
            log_warn "Retrying in ${RETRY_INTERVAL}s..."
            sleep $RETRY_INTERVAL
        fi
    done

    log_fail "${description} - HTTP ${response}"
    ((FAILED++))
    return 1
}

# Check service availability
check_availability() {
    log_info "=== Service Availability ==="

    if ! curl -s --max-time 5 "${BASE_URL}" > /dev/null 2>&1; then
        log_fail "Service is not reachable at ${BASE_URL}"
        ((FAILED++))
        return 1
    fi

    log_pass "Service is reachable"
    ((PASSED++))
    return 0
}

# Check health endpoint
check_health() {
    log_info "=== Health Endpoint ==="
    http_request "/api/v1/health" "Health endpoint" $RETRIES
}

# Check readiness endpoint
check_readiness() {
    log_info "=== Readiness Endpoint ==="
    http_request "/api/v1/ready" "Readiness endpoint" $RETRIES
}

# Check liveness
check_liveness() {
    log_info "=== Liveness Check ==="
    http_request "/api/v1/health/live" "Liveness endpoint" 1
}

# Check metrics endpoint
check_metrics() {
    log_info "=== Metrics Endpoint ==="

    local response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time $TIMEOUT \
        "${BASE_URL}/metrics" 2>/dev/null)

    if [ "${response}" = "200" ]; then
        log_pass "Metrics endpoint - HTTP ${response}"
        ((PASSED++))
    else
        log_warn "Metrics endpoint not available - HTTP ${response}"
        ((WARNINGS++))
    fi
}

# Check API documentation
check_docs() {
    log_info "=== API Documentation ==="

    local response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time $TIMEOUT \
        "${BASE_URL}/docs" 2>/dev/null)

    if [ "${response}" = "200" ]; then
        log_pass "API docs endpoint - HTTP ${response}"
        ((PASSED++))
    else
        log_warn "API docs not available - HTTP ${response}"
        ((WARNINGS++))
    fi
}

# Check response time
check_response_time() {
    log_info "=== Response Time Check ==="

    local start_time=$(date +%s%N)
    curl -s -o /dev/null --max-time $TIMEOUT "${BASE_URL}/api/v1/health" 2>/dev/null
    local end_time=$(date +%s%N)

    local response_time=$(( (end_time - start_time) / 1000000 ))

    log_info "Response time: ${response_time}ms"

    if [ $response_time -lt 1000 ]; then
        log_pass "Response time is acceptable (< 1000ms)"
        ((PASSED++))
    elif [ $response_time -lt 3000 ]; then
        log_warn "Response time is slow (< 3000ms)"
        ((WARNINGS++))
    else
        log_fail "Response time is too slow (>= 3000ms)"
        ((FAILED++))
    fi
}

# Check Kubernetes deployment (if kubectl available)
check_k8s_deployment() {
    log_info "=== Kubernetes Deployment ==="

    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl not available, skipping K8s checks"
        ((WARNINGS++))
        return
    fi

    local namespace="greenlang"
    local deployment="gl-002-boiler-efficiency"

    # Check deployment exists
    if ! kubectl get deployment "${deployment}" -n "${namespace}" &> /dev/null; then
        log_warn "Deployment not found in K8s"
        ((WARNINGS++))
        return
    fi

    # Check pod status
    local running_pods=$(kubectl get pods -n "${namespace}" \
        -l app="${deployment}" \
        -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)

    local desired_replicas=$(kubectl get deployment "${deployment}" \
        -n "${namespace}" \
        -o jsonpath='{.spec.replicas}')

    log_info "Running pods: ${running_pods}/${desired_replicas}"

    if [ "$running_pods" -eq "$desired_replicas" ]; then
        log_pass "All pods are running"
        ((PASSED++))
    else
        log_fail "Not all pods are running"
        ((FAILED++))
    fi
}

# Check database connectivity
check_database() {
    log_info "=== Database Connectivity ==="

    local response=$(curl -s --max-time $TIMEOUT \
        "${BASE_URL}/api/v1/health/db" 2>/dev/null | jq -r '.status' 2>/dev/null)

    if [ "${response}" = "healthy" ] || [ "${response}" = "ok" ]; then
        log_pass "Database connection OK"
        ((PASSED++))
    else
        log_warn "Database health check inconclusive"
        ((WARNINGS++))
    fi
}

# Check cache connectivity
check_cache() {
    log_info "=== Cache Connectivity ==="

    local response=$(curl -s --max-time $TIMEOUT \
        "${BASE_URL}/api/v1/health/cache" 2>/dev/null | jq -r '.status' 2>/dev/null)

    if [ "${response}" = "healthy" ] || [ "${response}" = "ok" ]; then
        log_pass "Cache connection OK"
        ((PASSED++))
    else
        log_warn "Cache health check inconclusive"
        ((WARNINGS++))
    fi
}

# Print summary
print_summary() {
    echo ""
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}Health Check Summary${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo -e "Environment: ${ENVIRONMENT}"
    echo -e "URL: ${BASE_URL}"
    echo ""
    echo -e "${GREEN}Passed:   ${PASSED}${NC}"
    echo -e "${YELLOW}Warnings: ${WARNINGS}${NC}"
    echo -e "${RED}Failed:   ${FAILED}${NC}"
    echo -e "${BLUE}======================================${NC}"

    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}Overall Status: HEALTHY ✓${NC}"
        return 0
    else
        echo -e "${RED}Overall Status: UNHEALTHY ✗${NC}"
        return 1
    fi
}

# Main health check flow
main() {
    log_info "Starting comprehensive health check..."

    check_availability
    check_health
    check_readiness
    check_liveness
    check_metrics
    check_docs
    check_response_time
    check_k8s_deployment
    check_database
    check_cache

    if print_summary; then
        exit 0
    else
        exit 1
    fi
}

main
