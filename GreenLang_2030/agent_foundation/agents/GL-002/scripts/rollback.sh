#!/bin/bash
# GL-002 Rollback Script
# Automated rollback to previous deployment version
# Usage: ./rollback.sh [ENVIRONMENT]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
ENVIRONMENT="${1:-production}"
NAMESPACE="greenlang"
DEPLOYMENT_NAME="gl-002-boiler-efficiency"
TIMEOUT="5m"

echo -e "${YELLOW}======================================${NC}"
echo -e "${YELLOW}GL-002 Rollback Script${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"
echo -e "${YELLOW}======================================${NC}"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Confirm rollback
confirm_rollback() {
    echo ""
    echo -e "${RED}WARNING: This will rollback the deployment to the previous version${NC}"
    echo -e "Environment: ${ENVIRONMENT}"
    echo -e "Deployment: ${DEPLOYMENT_NAME}"
    echo ""
    read -p "Are you sure you want to rollback? (yes/no): " -r
    echo

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_error "Rollback cancelled by user"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not installed"
        exit 1
    fi

    log_info "Prerequisites OK"
}

# Verify cluster connection
verify_cluster() {
    log_info "Verifying cluster connection..."

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to cluster"
        exit 1
    fi

    log_info "Connected to cluster"
}

# Get current deployment info
get_deployment_info() {
    log_info "Getting current deployment information..."

    CURRENT_IMAGE=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.template.spec.containers[0].image}')

    CURRENT_REPLICAS=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.replicas}')

    log_info "Current image: ${CURRENT_IMAGE}"
    log_info "Current replicas: ${CURRENT_REPLICAS}"
}

# Get rollback history
show_rollback_history() {
    log_info "Deployment history:"
    echo ""
    kubectl rollout history deployment/"${DEPLOYMENT_NAME}" -n "${NAMESPACE}"
    echo ""
}

# Perform rollback
perform_rollback() {
    log_warn "Initiating rollback..."

    kubectl rollout undo deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}"

    log_info "Rollback initiated, waiting for completion..."
}

# Wait for rollback to complete
wait_for_rollback() {
    log_info "Waiting for rollback to complete (timeout: ${TIMEOUT})..."

    if ! kubectl rollout status deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        --timeout="${TIMEOUT}"; then
        log_error "Rollback failed or timed out"
        return 1
    fi

    log_info "Rollback completed"
    return 0
}

# Verify rollback success
verify_rollback() {
    log_info "Verifying rollback..."

    # Get new deployment info
    NEW_IMAGE=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.template.spec.containers[0].image}')

    NEW_REPLICAS=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.replicas}')

    log_info "New image: ${NEW_IMAGE}"
    log_info "New replicas: ${NEW_REPLICAS}"

    # Check pod status
    RUNNING_PODS=$(kubectl get pods -n "${NAMESPACE}" \
        -l app="${DEPLOYMENT_NAME}" \
        -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)

    log_info "Running pods: ${RUNNING_PODS}/${NEW_REPLICAS}"

    if [ "$RUNNING_PODS" -ne "$NEW_REPLICAS" ]; then
        log_error "Not all pods are running"
        return 1
    fi

    log_info "All pods are running"
    return 0
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."

    # Get pod names
    POD_NAMES=$(kubectl get pods -n "${NAMESPACE}" \
        -l app="${DEPLOYMENT_NAME}" \
        -o jsonpath='{.items[*].metadata.name}')

    for pod in $POD_NAMES; do
        log_info "Checking health of pod: ${pod}"
        if ! kubectl exec -n "${NAMESPACE}" "${pod}" -- \
            curl -f -s http://localhost:8000/api/v1/health > /dev/null; then
            log_warn "Health check failed for pod: ${pod}"
        else
            log_info "Health check passed for pod: ${pod}"
        fi
    done

    log_info "Health checks completed"
}

# Send notification
send_notification() {
    local status=$1

    log_info "Sending rollback notification..."

    # Placeholder for notification
    # curl -X POST webhook_url -d "{\"status\": \"${status}\"}"
}

# Main rollback flow
main() {
    log_info "Starting rollback for GL-002 deployment..."

    if [[ "${ENVIRONMENT}" == "production" ]]; then
        confirm_rollback
    fi

    check_prerequisites
    verify_cluster
    get_deployment_info
    show_rollback_history

    perform_rollback

    if ! wait_for_rollback; then
        log_error "Rollback failed"
        send_notification "FAILED"
        exit 1
    fi

    if ! verify_rollback; then
        log_error "Rollback verification failed"
        send_notification "FAILED"
        exit 1
    fi

    run_health_checks

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Rollback Successful!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "Environment: ${ENVIRONMENT}"
    echo -e "Previous image: ${CURRENT_IMAGE}"
    echo -e "Current image: ${NEW_IMAGE}"
    echo ""

    kubectl get pods -n "${NAMESPACE}" -l app="${DEPLOYMENT_NAME}"

    send_notification "SUCCESS"

    exit 0
}

main
