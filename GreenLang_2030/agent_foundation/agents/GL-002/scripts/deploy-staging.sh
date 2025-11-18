#!/bin/bash
# GL-002 Staging Deployment Script
# Deploys GL-002 BoilerEfficiencyOptimizer to staging environment
# Usage: ./deploy-staging.sh [IMAGE_TAG]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="greenlang"
DEPLOYMENT_NAME="gl-002-boiler-efficiency"
IMAGE_REGISTRY="ghcr.io/greenlang"
IMAGE_NAME="gl-002"
IMAGE_TAG="${1:-latest}"
TIMEOUT="5m"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}GL-002 Staging Deployment${NC}"
echo -e "${GREEN}======================================${NC}"

# Function to log messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Verify cluster connectivity
verify_cluster() {
    log_info "Verifying Kubernetes cluster connectivity..."

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warn "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi

    log_info "Cluster connectivity verified"
}

# Pull and verify Docker image
verify_image() {
    log_info "Verifying Docker image: ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

    if ! docker pull "${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"; then
        log_error "Failed to pull Docker image"
        exit 1
    fi

    log_info "Docker image verified"
}

# Update deployment
update_deployment() {
    log_info "Updating deployment..."

    kubectl set image deployment/"${DEPLOYMENT_NAME}" \
        "${DEPLOYMENT_NAME}=${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
        -n "${NAMESPACE}" \
        --record

    log_info "Deployment updated, waiting for rollout..."
}

# Wait for rollout to complete
wait_for_rollout() {
    log_info "Waiting for rollout to complete (timeout: ${TIMEOUT})..."

    if ! kubectl rollout status deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        --timeout="${TIMEOUT}"; then
        log_error "Rollout failed or timed out"
        return 1
    fi

    log_info "Rollout completed successfully"
    return 0
}

# Verify deployment health
verify_deployment() {
    log_info "Verifying deployment health..."

    # Check pod status
    RUNNING_PODS=$(kubectl get pods -n "${NAMESPACE}" \
        -l app="${DEPLOYMENT_NAME}" \
        -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)

    DESIRED_REPLICAS=$(kubectl get deployment "${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        -o jsonpath='{.spec.replicas}')

    log_info "Running pods: ${RUNNING_PODS}/${DESIRED_REPLICAS}"

    if [ "$RUNNING_PODS" -ne "$DESIRED_REPLICAS" ]; then
        log_error "Not all pods are running"
        return 1
    fi

    log_info "All pods are running"
    return 0
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."

    STAGING_URL="${STAGING_URL:-https://api.staging.greenlang.io}"

    # Health check
    if ! curl -f -s "${STAGING_URL}/api/v1/health" > /dev/null; then
        log_error "Health check failed"
        return 1
    fi
    log_info "Health check passed"

    # Readiness check
    if ! curl -f -s "${STAGING_URL}/api/v1/ready" > /dev/null; then
        log_error "Readiness check failed"
        return 1
    fi
    log_info "Readiness check passed"

    log_info "All smoke tests passed"
    return 0
}

# Rollback on failure
rollback_deployment() {
    log_warn "Rolling back deployment..."

    kubectl rollout undo deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}"

    kubectl rollout status deployment/"${DEPLOYMENT_NAME}" \
        -n "${NAMESPACE}" \
        --timeout="${TIMEOUT}"

    log_warn "Rollback completed"
}

# Main deployment flow
main() {
    log_info "Starting staging deployment for GL-002..."
    log_info "Image: ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

    check_prerequisites
    verify_cluster
    verify_image
    update_deployment

    if ! wait_for_rollout; then
        log_error "Deployment failed during rollout"
        rollback_deployment
        exit 1
    fi

    if ! verify_deployment; then
        log_error "Deployment verification failed"
        rollback_deployment
        exit 1
    fi

    if ! run_smoke_tests; then
        log_warn "Smoke tests failed (deployment continues)"
    fi

    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Deployment Successful!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "Image: ${IMAGE_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    echo -e "Namespace: ${NAMESPACE}"
    echo -e "Deployment: ${DEPLOYMENT_NAME}"
    echo ""

    # Show pod status
    kubectl get pods -n "${NAMESPACE}" -l app="${DEPLOYMENT_NAME}"

    exit 0
}

# Execute main function
main
