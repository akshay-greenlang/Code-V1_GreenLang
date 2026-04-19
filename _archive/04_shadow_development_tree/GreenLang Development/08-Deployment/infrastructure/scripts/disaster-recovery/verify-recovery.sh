#!/bin/bash
# =============================================================================
# GreenLang Recovery Verification Script
# Comprehensive validation of disaster recovery operations
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/greenlang/verify-recovery.log"
ENVIRONMENT="${ENVIRONMENT:-production}"
API_ENDPOINT="${API_ENDPOINT:-https://api.greenlang.io}"
API_TOKEN="${API_TOKEN:-}"

# Test configuration
QUICK_MODE="${QUICK_MODE:-false}"
VERBOSE="${VERBOSE:-false}"
FAIL_FAST="${FAIL_FAST:-false}"

# Results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
TOTAL_TESTS=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$*"; }
log_warn() { log "WARN" "${YELLOW}$*${NC}"; }
log_error() { log "ERROR" "${RED}$*${NC}"; }
log_success() { log "SUCCESS" "${GREEN}$*${NC}"; }

# Test result tracking
test_pass() {
    local test_name=$1
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "  ${GREEN}[PASS]${NC} ${test_name}"
}

test_fail() {
    local test_name=$1
    local reason="${2:-}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "  ${RED}[FAIL]${NC} ${test_name}"
    [[ -n "${reason}" ]] && echo -e "        ${RED}Reason: ${reason}${NC}"

    if [[ "${FAIL_FAST}" == "true" ]]; then
        echo ""
        echo "FAIL_FAST enabled. Stopping tests."
        print_summary
        exit 1
    fi
}

test_skip() {
    local test_name=$1
    local reason="${2:-}"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "  ${YELLOW}[SKIP]${NC} ${test_name}"
    [[ -n "${reason}" ]] && echo -e "        Reason: ${reason}"
}

# Print test summary
print_summary() {
    echo ""
    echo "=========================================="
    echo "RECOVERY VERIFICATION SUMMARY"
    echo "=========================================="
    echo -e "Total Tests:  ${TOTAL_TESTS}"
    echo -e "Passed:       ${GREEN}${TESTS_PASSED}${NC}"
    echo -e "Failed:       ${RED}${TESTS_FAILED}${NC}"
    echo -e "Skipped:      ${YELLOW}${TESTS_SKIPPED}${NC}"
    echo ""

    if [[ ${TESTS_FAILED} -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}${TESTS_FAILED} test(s) failed!${NC}"
        return 1
    fi
}

# =============================================================================
# Infrastructure Tests
# =============================================================================

test_kubernetes_connectivity() {
    echo ""
    echo "${CYAN}=== Kubernetes Connectivity Tests ===${NC}"

    # Cluster info
    if kubectl cluster-info &>/dev/null; then
        test_pass "Kubernetes cluster accessible"
    else
        test_fail "Kubernetes cluster accessible" "Cannot connect to cluster"
        return 1
    fi

    # Node health
    local unhealthy_nodes=$(kubectl get nodes --no-headers 2>/dev/null | \
        grep -v "Ready" | wc -l)
    if [[ "${unhealthy_nodes}" -eq 0 ]]; then
        test_pass "All Kubernetes nodes healthy"
    else
        test_fail "All Kubernetes nodes healthy" "${unhealthy_nodes} unhealthy nodes"
    fi

    # Namespace exists
    if kubectl get namespace greenlang-production &>/dev/null; then
        test_pass "Production namespace exists"
    else
        test_fail "Production namespace exists"
    fi
}

test_deployments() {
    echo ""
    echo "${CYAN}=== Deployment Tests ===${NC}"

    local deployments=("greenlang-api-blue" "greenlang-worker" "greenlang-frontend")

    for deployment in "${deployments[@]}"; do
        local ready=$(kubectl get deployment "${deployment}" \
            -n greenlang-production \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

        local desired=$(kubectl get deployment "${deployment}" \
            -n greenlang-production \
            -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

        if [[ "${ready}" -eq "${desired}" && "${ready}" -gt 0 ]]; then
            test_pass "Deployment ${deployment} (${ready}/${desired} ready)"
        else
            test_fail "Deployment ${deployment}" "${ready}/${desired} ready"
        fi
    done
}

test_pods() {
    echo ""
    echo "${CYAN}=== Pod Health Tests ===${NC}"

    # Count running pods
    local running_pods=$(kubectl get pods -n greenlang-production \
        --field-selector=status.phase=Running \
        --no-headers 2>/dev/null | wc -l)

    if [[ "${running_pods}" -gt 0 ]]; then
        test_pass "Running pods found (${running_pods})"
    else
        test_fail "Running pods found" "No running pods"
    fi

    # Check for crash loops
    local crash_loops=$(kubectl get pods -n greenlang-production \
        -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}' 2>/dev/null | \
        tr ' ' '\n' | awk '$1 > 5' | wc -l)

    if [[ "${crash_loops}" -eq 0 ]]; then
        test_pass "No pods in crash loop"
    else
        test_fail "No pods in crash loop" "${crash_loops} pods with high restart count"
    fi

    # Check for pending pods
    local pending_pods=$(kubectl get pods -n greenlang-production \
        --field-selector=status.phase=Pending \
        --no-headers 2>/dev/null | wc -l)

    if [[ "${pending_pods}" -eq 0 ]]; then
        test_pass "No pending pods"
    else
        test_fail "No pending pods" "${pending_pods} pods pending"
    fi
}

test_services() {
    echo ""
    echo "${CYAN}=== Service Tests ===${NC}"

    local services=("greenlang-api-service" "greenlang-frontend-service")

    for service in "${services[@]}"; do
        if kubectl get service "${service}" -n greenlang-production &>/dev/null; then
            local endpoints=$(kubectl get endpoints "${service}" \
                -n greenlang-production \
                -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null | wc -w)

            if [[ "${endpoints}" -gt 0 ]]; then
                test_pass "Service ${service} (${endpoints} endpoints)"
            else
                test_fail "Service ${service}" "No endpoints"
            fi
        else
            test_fail "Service ${service}" "Service not found"
        fi
    done
}

# =============================================================================
# Database Tests
# =============================================================================

test_database_connectivity() {
    echo ""
    echo "${CYAN}=== Database Connectivity Tests ===${NC}"

    # RDS status
    local rds_status=$(aws rds describe-db-instances \
        --db-instance-identifier "${RDS_INSTANCE_ID:-greenlang-production}" \
        --query 'DBInstances[0].DBInstanceStatus' \
        --output text 2>/dev/null || echo "unknown")

    if [[ "${rds_status}" == "available" ]]; then
        test_pass "RDS instance available"
    else
        test_fail "RDS instance available" "Status: ${rds_status}"
    fi

    # Database connection via API
    local db_health=$(curl -s "${API_ENDPOINT}/api/v1/health/db" 2>/dev/null | \
        jq -r '.status' 2>/dev/null || echo "unknown")

    if [[ "${db_health}" == "healthy" ]]; then
        test_pass "Database connection via API"
    else
        test_fail "Database connection via API" "Status: ${db_health}"
    fi
}

test_database_data() {
    echo ""
    echo "${CYAN}=== Database Data Integrity Tests ===${NC}"

    if [[ "${QUICK_MODE}" == "true" ]]; then
        test_skip "Detailed data integrity checks" "Quick mode enabled"
        return 0
    fi

    # Check critical tables have data
    local tables_check=$(curl -s "${API_ENDPOINT}/api/v1/health/db/tables" \
        -H "Authorization: Bearer ${API_TOKEN}" 2>/dev/null | \
        jq -r '.tables_with_data // 0' 2>/dev/null || echo "0")

    if [[ "${tables_check}" -gt 5 ]]; then
        test_pass "Critical tables have data (${tables_check} tables)"
    else
        test_fail "Critical tables have data" "Only ${tables_check} tables with data"
    fi

    # Check recent data exists (within last hour)
    local recent_data=$(curl -s "${API_ENDPOINT}/api/v1/health/db/recent" \
        -H "Authorization: Bearer ${API_TOKEN}" 2>/dev/null | \
        jq -r '.has_recent_data' 2>/dev/null || echo "false")

    if [[ "${recent_data}" == "true" ]]; then
        test_pass "Recent data exists (last hour)"
    else
        test_warn "Recent data exists" "No data in last hour (may be expected)"
    fi
}

# =============================================================================
# Cache Tests
# =============================================================================

test_cache_connectivity() {
    echo ""
    echo "${CYAN}=== Cache Connectivity Tests ===${NC}"

    # ElastiCache status
    local cache_status=$(aws elasticache describe-cache-clusters \
        --cache-cluster-id "${ELASTICACHE_ID:-greenlang-redis}" \
        --query 'CacheClusters[0].CacheClusterStatus' \
        --output text 2>/dev/null || echo "unknown")

    if [[ "${cache_status}" == "available" ]]; then
        test_pass "ElastiCache cluster available"
    else
        test_fail "ElastiCache cluster available" "Status: ${cache_status}"
    fi

    # Redis connection via API
    local cache_health=$(curl -s "${API_ENDPOINT}/api/v1/health/cache" 2>/dev/null | \
        jq -r '.status' 2>/dev/null || echo "unknown")

    if [[ "${cache_health}" == "healthy" ]]; then
        test_pass "Redis connection via API"
    else
        test_fail "Redis connection via API" "Status: ${cache_health}"
    fi
}

# =============================================================================
# API Tests
# =============================================================================

test_api_health() {
    echo ""
    echo "${CYAN}=== API Health Tests ===${NC}"

    # Health endpoint
    local health_status=$(curl -s -o /dev/null -w "%{http_code}" \
        "${API_ENDPOINT}/api/v1/health" 2>/dev/null || echo "000")

    if [[ "${health_status}" == "200" ]]; then
        test_pass "Health endpoint returns 200"
    else
        test_fail "Health endpoint returns 200" "HTTP ${health_status}"
    fi

    # Readiness endpoint
    local ready_status=$(curl -s -o /dev/null -w "%{http_code}" \
        "${API_ENDPOINT}/api/v1/ready" 2>/dev/null || echo "000")

    if [[ "${ready_status}" == "200" ]]; then
        test_pass "Readiness endpoint returns 200"
    else
        test_fail "Readiness endpoint returns 200" "HTTP ${ready_status}"
    fi

    # Version endpoint
    local version=$(curl -s "${API_ENDPOINT}/api/v1/version" 2>/dev/null | \
        jq -r '.version' 2>/dev/null || echo "unknown")

    if [[ "${version}" != "unknown" && -n "${version}" ]]; then
        test_pass "Version endpoint returns version (${version})"
    else
        test_fail "Version endpoint returns version"
    fi
}

test_api_response_time() {
    echo ""
    echo "${CYAN}=== API Response Time Tests ===${NC}"

    local total_time=0
    local requests=5

    for i in $(seq 1 ${requests}); do
        local time=$(curl -s -o /dev/null -w "%{time_total}" \
            "${API_ENDPOINT}/api/v1/health" 2>/dev/null || echo "10")
        total_time=$(echo "${total_time} + ${time}" | bc)
    done

    local avg_time=$(echo "scale=3; ${total_time} / ${requests}" | bc)
    local avg_ms=$(echo "scale=0; ${avg_time} * 1000" | bc)

    if (( $(echo "${avg_time} < 0.5" | bc -l) )); then
        test_pass "Average response time < 500ms (${avg_ms}ms)"
    elif (( $(echo "${avg_time} < 1.0" | bc -l) )); then
        test_warn "Average response time" "${avg_ms}ms (acceptable but slow)"
    else
        test_fail "Average response time" "${avg_ms}ms exceeds 1000ms threshold"
    fi
}

test_api_functionality() {
    echo ""
    echo "${CYAN}=== API Functionality Tests ===${NC}"

    if [[ -z "${API_TOKEN}" ]]; then
        test_skip "API functionality tests" "No API token provided"
        return 0
    fi

    # Test carbon calculation endpoint
    local calc_response=$(curl -s -X POST "${API_ENDPOINT}/api/v1/carbon/calculate" \
        -H "Authorization: Bearer ${API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{"type":"test","scope":"scope1","value":100,"unit":"kWh"}' 2>/dev/null)

    local calc_status=$(echo "${calc_response}" | jq -r '.status // .error' 2>/dev/null || echo "error")

    if [[ "${calc_status}" == "success" || "${calc_status}" == "calculated" ]]; then
        test_pass "Carbon calculation endpoint functional"
    else
        test_fail "Carbon calculation endpoint functional" "${calc_status}"
    fi

    # Test data retrieval
    local data_response=$(curl -s "${API_ENDPOINT}/api/v1/emissions/summary" \
        -H "Authorization: Bearer ${API_TOKEN}" 2>/dev/null)

    local data_status=$(echo "${data_response}" | jq -r 'type' 2>/dev/null || echo "error")

    if [[ "${data_status}" == "object" || "${data_status}" == "array" ]]; then
        test_pass "Data retrieval endpoint functional"
    else
        test_fail "Data retrieval endpoint functional"
    fi
}

# =============================================================================
# Security Tests
# =============================================================================

test_security() {
    echo ""
    echo "${CYAN}=== Security Tests ===${NC}"

    # TLS/SSL
    local ssl_status=$(curl -s -o /dev/null -w "%{ssl_verify_result}" \
        "${API_ENDPOINT}/api/v1/health" 2>/dev/null || echo "1")

    if [[ "${ssl_status}" == "0" ]]; then
        test_pass "SSL certificate valid"
    else
        test_fail "SSL certificate valid" "Verification result: ${ssl_status}"
    fi

    # Check for sensitive headers
    local headers=$(curl -s -I "${API_ENDPOINT}/api/v1/health" 2>/dev/null)

    if echo "${headers}" | grep -qi "x-content-type-options: nosniff"; then
        test_pass "Security headers present (X-Content-Type-Options)"
    else
        test_warn "Security headers" "X-Content-Type-Options missing"
    fi

    # Authentication required for protected endpoints
    local unauth_response=$(curl -s -o /dev/null -w "%{http_code}" \
        "${API_ENDPOINT}/api/v1/admin/users" 2>/dev/null || echo "000")

    if [[ "${unauth_response}" == "401" || "${unauth_response}" == "403" ]]; then
        test_pass "Protected endpoints require authentication"
    else
        test_fail "Protected endpoints require authentication" "HTTP ${unauth_response}"
    fi
}

# =============================================================================
# Monitoring Tests
# =============================================================================

test_monitoring() {
    echo ""
    echo "${CYAN}=== Monitoring Tests ===${NC}"

    # Metrics endpoint
    local metrics_status=$(curl -s -o /dev/null -w "%{http_code}" \
        "${API_ENDPOINT}/metrics" 2>/dev/null || echo "000")

    if [[ "${metrics_status}" == "200" ]]; then
        test_pass "Metrics endpoint accessible"
    else
        test_fail "Metrics endpoint accessible" "HTTP ${metrics_status}"
    fi

    # Check Prometheus is scraping
    if kubectl get servicemonitor greenlang-api \
        -n greenlang-production &>/dev/null; then
        test_pass "ServiceMonitor configured"
    else
        test_skip "ServiceMonitor configured" "Not using Prometheus Operator"
    fi
}

# =============================================================================
# Load Test (Optional)
# =============================================================================

test_load() {
    echo ""
    echo "${CYAN}=== Load Tests ===${NC}"

    if [[ "${QUICK_MODE}" == "true" ]]; then
        test_skip "Load tests" "Quick mode enabled"
        return 0
    fi

    if ! command -v ab &>/dev/null; then
        test_skip "Load tests" "Apache Bench (ab) not installed"
        return 0
    fi

    # Light load test
    local ab_output=$(ab -n 100 -c 10 -q "${API_ENDPOINT}/api/v1/health" 2>&1)

    local failed_requests=$(echo "${ab_output}" | grep "Failed requests:" | awk '{print $3}')
    local requests_per_sec=$(echo "${ab_output}" | grep "Requests per second:" | awk '{print $4}')

    if [[ "${failed_requests:-0}" -eq 0 ]]; then
        test_pass "Load test: 0 failed requests"
    else
        test_fail "Load test" "${failed_requests} failed requests"
    fi

    if (( $(echo "${requests_per_sec:-0} > 50" | bc -l) )); then
        test_pass "Load test: ${requests_per_sec} req/sec"
    else
        test_warn "Load test throughput" "${requests_per_sec} req/sec (may be slow)"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    mkdir -p "$(dirname "${LOG_FILE}")"

    echo "=========================================="
    echo "GREENLANG RECOVERY VERIFICATION"
    echo "=========================================="
    echo "Environment: ${ENVIRONMENT}"
    echo "API Endpoint: ${API_ENDPOINT}"
    echo "Quick Mode: ${QUICK_MODE}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # Run test suites
    test_kubernetes_connectivity
    test_deployments
    test_pods
    test_services
    test_database_connectivity
    test_database_data
    test_cache_connectivity
    test_api_health
    test_api_response_time
    test_api_functionality
    test_security
    test_monitoring
    test_load

    # Print summary and exit with appropriate code
    print_summary
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE="true"
            shift
            ;;
        --verbose|-v)
            VERBOSE="true"
            shift
            ;;
        --fail-fast|-f)
            FAIL_FAST="true"
            shift
            ;;
        --api-endpoint)
            API_ENDPOINT="$2"
            shift 2
            ;;
        --api-token)
            API_TOKEN="$2"
            shift 2
            ;;
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -q, --quick         Run quick tests only"
            echo "  -v, --verbose       Verbose output"
            echo "  -f, --fail-fast     Stop on first failure"
            echo "  --api-endpoint URL  API endpoint to test"
            echo "  --api-token TOKEN   API authentication token"
            echo "  -e, --environment   Environment name"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
