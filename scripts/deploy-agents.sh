#!/bin/bash
# =============================================================================
# GreenLang Agents - Kubernetes Deployment Script
# =============================================================================
# Deploys all agents to Kubernetes cluster
# Usage: ./scripts/deploy-agents.sh [namespace] [--dry-run]
# =============================================================================

set -euo pipefail

# Configuration
NAMESPACE="${1:-greenlang-dev}"
DRY_RUN="${2:-}"
K8S_DIR="k8s/agents"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

log_section() {
    echo -e "${BLUE}[====]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_section "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Apply manifests
apply_manifests() {
    local dry_run_flag=""
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        dry_run_flag="--dry-run=client"
        log_warn "Running in dry-run mode"
    fi

    log_section "Applying Kubernetes manifests..."

    # Apply in order
    local manifests=(
        "namespace.yaml"
        "rbac.yaml"
        "configmap.yaml"
        "services.yaml"
        "deployment-fuel-analyzer.yaml"
        "deployment-carbon-intensity.yaml"
        "deployment-energy-performance.yaml"
        "hpa.yaml"
    )

    for manifest in "${manifests[@]}"; do
        local file="${K8S_DIR}/${manifest}"
        if [[ -f "$file" ]]; then
            log_info "Applying ${manifest}..."
            kubectl apply -f "$file" $dry_run_flag
        else
            log_warn "File not found: ${file}"
        fi
    done
}

# Wait for deployments
wait_for_deployments() {
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        log_info "Skipping wait in dry-run mode"
        return
    fi

    log_section "Waiting for deployments to be ready..."

    local deployments=(
        "fuel-analyzer"
        "carbon-intensity"
        "energy-performance"
    )

    for deployment in "${deployments[@]}"; do
        log_info "Waiting for ${deployment}..."
        kubectl rollout status deployment/${deployment} \
            -n ${NAMESPACE} \
            --timeout=300s || {
            log_error "Deployment ${deployment} failed to become ready"
            return 1
        }
    done

    log_info "All deployments are ready"
}

# Verify deployment
verify_deployment() {
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        log_info "Skipping verification in dry-run mode"
        return
    fi

    log_section "Verifying deployment..."

    echo ""
    log_info "Pods:"
    kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/part-of=greenlang-platform

    echo ""
    log_info "Services:"
    kubectl get svc -n ${NAMESPACE}

    echo ""
    log_info "HPA:"
    kubectl get hpa -n ${NAMESPACE}
}

# Health check
health_check() {
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        log_info "Skipping health check in dry-run mode"
        return
    fi

    log_section "Running health checks..."

    local deployments=(
        "fuel-analyzer"
        "carbon-intensity"
        "energy-performance"
    )

    for deployment in "${deployments[@]}"; do
        local pod=$(kubectl get pods -n ${NAMESPACE} -l app=${deployment} -o jsonpath='{.items[0].metadata.name}')
        if [[ -n "$pod" ]]; then
            log_info "Checking health for ${deployment}..."
            kubectl exec -n ${NAMESPACE} ${pod} -- curl -sf http://localhost:8000/health || {
                log_warn "Health check failed for ${deployment}"
            }
        fi
    done
}

# Main
main() {
    log_info "=== GreenLang Agent Deployment ==="
    log_info "Namespace: ${NAMESPACE}"
    log_info "K8s Directory: ${K8S_DIR}"
    log_info ""

    check_prerequisites
    apply_manifests
    wait_for_deployments
    verify_deployment
    health_check

    log_info ""
    log_info "=== Deployment Complete ==="
    log_info "To access agents, use kubectl port-forward:"
    log_info "  kubectl port-forward svc/fuel-analyzer 8001:80 -n ${NAMESPACE}"
    log_info "  kubectl port-forward svc/carbon-intensity 8002:80 -n ${NAMESPACE}"
    log_info "  kubectl port-forward svc/energy-performance 8003:80 -n ${NAMESPACE}"
}

# Run main
main "$@"
