#!/bin/bash
# GL-002 Production Deployment Script
# Blue-Green deployment to production with automated health checks
# Usage: ./deploy-production.sh [IMAGE_TAG]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NAMESPACE="greenlang"
DEPLOYMENT_NAME="gl-002-boiler-efficiency"
IMAGE_REGISTRY="ghcr.io/greenlang"
IMAGE_NAME="gl-002"
IMAGE_TAG="${1:-latest}"
TIMEOUT="10m"
HEALTH_CHECK_RETRIES=5
HEALTH_CHECK_INTERVAL=10

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}GL-002 Production Deployment${NC}"
echo -e "${BLUE}Blue-Green Strategy${NC}"
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

# Confirmation prompt
confirm_production_deployment() {
    echo ""
    echo -e "${YELLOW}WARNING: This will deploy to PRODUCTION${NC}"
    echo -e "Image: ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo -e "Namespace: ${NAMESPACE}"
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_error "Deployment cancelled by user"
        exit 1
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not installed"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        log_error "curl not installed"
        exit 1
    fi

    log_info "Prerequisites OK"
}

verify_cluster() {
    log_info "Verifying production cluster..."

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to cluster"
        exit 1
    fi

    # Verify we're on production cluster
    CURRENT_CONTEXT=$(kubectl config current-context)
    log_info "Current context: ${CURRENT_CONTEXT}"

    if [[ ! "${CURRENT_CONTEXT}" =~ "production" ]] && [[ ! "${CURRENT_CONTEXT}" =~ "prod" ]]; then
        log_warn "Current context does not appear to be production"
        read -p "Continue anyway? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            exit 1
        fi
    fi

    log_info "Cluster verified"
}

# Get current deployment state (Blue)
get_current_state() {
    log_info "Getting current deployment state..."

    CURRENT_IMAGE=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.template.spec.containers[0].image}' 2>/dev/null || echo "none")

    CURRENT_REPLICAS=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "3")

    log_info "Current image: ${CURRENT_IMAGE}"
    log_info "Current replicas: ${CURRENT_REPLICAS}"
}

# Create green deployment
create_green_deployment() {
    log_info "Creating green deployment..."

    kubectl set image deployment/"${DEPLOYMENT_NAME}" \
        "${DEPLOYMENT_NAME}=${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
        -n "${NAMESPACE}" \
        --record

    log_info "Green deployment created"
}

# Wait for green deployment
wait_for_green_ready() {
    log_info "Waiting for green deployment (timeout: ${TIMEOUT})..."

    if ! kubectl rollout status deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        --timeout="${TIMEOUT}"; then
        log_error "Green deployment failed"
        return 1
    fi

    log_info "Green deployment ready"
    return 0
}

# Verify all pods healthy
verify_pod_health() {
    log_info "Verifying pod health..."

    RUNNING_PODS=$(kubectl get pods -n "${NAMESPACE}" \
        -l app="${DEPLOYMENT_NAME}" \
        -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)

    DESIRED_REPLICAS=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.replicas}')

    log_info "Running pods: ${RUNNING_PODS}/${DESIRED_REPLICAS}"

    if [ "$RUNNING_PODS" -ne "$DESIRED_REPLICAS" ]; then
        log_error "Not all pods running"
        return 1
    fi

    # Check each pod's health endpoint
    POD_NAMES=$(kubectl get pods -n "${NAMESPACE}" \
        -l app="${DEPLOYMENT_NAME}" \
        -o jsonpath='{.items[*].metadata.name}')

    for pod in $POD_NAMES; do
        log_info "Checking health of pod: ${pod}"
        if ! kubectl exec -n "${NAMESPACE}" "${pod}" -- \
            curl -f -s http://localhost:8000/api/v1/health > /dev/null; then
            log_error "Health check failed for pod: ${pod}"
            return 1
        fi
    done

    log_info "All pods healthy"
    return 0
}

# Run production smoke tests
run_production_tests() {
    log_info "Running production smoke tests..."

    PROD_URL="${PROD_URL:-https://api.boiler.greenlang.io}"

    # Health check with retries
    for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
        log_info "Health check attempt ${i}/${HEALTH_CHECK_RETRIES}..."

        if curl -f -s "${PROD_URL}/api/v1/health" > /dev/null; then
            log_info "Health check passed"
            break
        fi

        if [ $i -eq $HEALTH_CHECK_RETRIES ]; then
            log_error "Health check failed after ${HEALTH_CHECK_RETRIES} attempts"
            return 1
        fi

        sleep $HEALTH_CHECK_INTERVAL
    done

    # Readiness check
    if ! curl -f -s "${PROD_URL}/api/v1/ready" > /dev/null; then
        log_error "Readiness check failed"
        return 1
    fi
    log_info "Readiness check passed"

    # Metrics endpoint check
    if ! curl -f -s "${PROD_URL}/metrics" > /dev/null; then
        log_warn "Metrics endpoint check failed (non-critical)"
    else
        log_info "Metrics endpoint OK"
    fi

    log_info "Production tests passed"
    return 0
}

# Rollback to blue deployment
rollback() {
    log_error "DEPLOYMENT FAILED - Initiating rollback..."

    kubectl rollout undo deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}"

    kubectl rollout status deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        --timeout="${TIMEOUT}"

    log_warn "Rollback completed - service restored to previous version"
}

# Send deployment notification
send_notification() {
    local status=$1
    local message=$2

    log_info "Sending deployment notification: ${status}"

    # Placeholder for Slack/PagerDuty/Email notification
    # curl -X POST webhook_url -d "{\"status\": \"${status}\", \"message\": \"${message}\"}"
}

# Main deployment flow
main() {
    log_info "Starting production deployment for GL-002..."

    confirm_production_deployment
    check_prerequisites
    verify_cluster
    get_current_state

    log_info "Beginning blue-green deployment..."

    create_green_deployment

    if ! wait_for_green_ready; then
        log_error "Green deployment failed to become ready"
        rollback
        send_notification "FAILED" "Green deployment failed"
        exit 1
    fi

    if ! verify_pod_health; then
        log_error "Pod health verification failed"
        rollback
        send_notification "FAILED" "Health checks failed"
        exit 1
    fi

    if ! run_production_tests; then
        log_error "Production tests failed"
        rollback
        send_notification "FAILED" "Production tests failed"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Production Deployment Successful!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "Image: ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo -e "Previous: ${CURRENT_IMAGE}"
    echo -e "Replicas: ${CURRENT_REPLICAS}"
    echo ""

    kubectl get pods -n "${NAMESPACE}" -l app="${DEPLOYMENT_NAME}"

    send_notification "SUCCESS" "Production deployment completed"

    exit 0
}

main
