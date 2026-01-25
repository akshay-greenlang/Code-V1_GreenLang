#!/bin/bash
# GL-005 CombustionControlAgent - Rollback Script
# Rolls back GL-005 deployment to previous version
# Usage: ./rollback.sh [environment] [revision]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
ENVIRONMENT="${1:-production}"
REVISION="${2:-0}"  # 0 means previous revision

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo "=========================================="
    echo "  GL-005 CombustionControlAgent Rollback"
    echo "  Environment: $ENVIRONMENT"
    echo "=========================================="
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check kubeconfig."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

get_namespace() {
    case $ENVIRONMENT in
        dev)
            echo "greenlang-dev"
            ;;
        staging)
            echo "greenlang-staging"
            ;;
        production)
            echo "greenlang"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

show_rollout_history() {
    local namespace
    namespace=$(get_namespace)

    log_info "Showing deployment rollout history..."
    echo ""

    kubectl rollout history deployment/gl-005-combustion-control -n "$namespace"

    echo ""
}

confirm_rollback() {
    local namespace
    namespace=$(get_namespace)

    log_warning "This will rollback GL-005 deployment in $ENVIRONMENT environment"
    log_warning "Namespace: $namespace"

    if [ "$REVISION" -eq 0 ]; then
        log_warning "Revision: Previous (auto-detect)"
    else
        log_warning "Revision: $REVISION"
    fi

    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " -r
    echo

    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
}

perform_rollback() {
    local namespace
    namespace=$(get_namespace)

    log_info "Performing rollback..."

    if [ "$REVISION" -eq 0 ]; then
        # Rollback to previous revision
        if kubectl rollout undo deployment/gl-005-combustion-control -n "$namespace"; then
            log_success "Rollback initiated successfully"
        else
            log_error "Rollback failed"
            exit 1
        fi
    else
        # Rollback to specific revision
        if kubectl rollout undo deployment/gl-005-combustion-control \
            -n "$namespace" --to-revision="$REVISION"; then
            log_success "Rollback to revision $REVISION initiated successfully"
        else
            log_error "Rollback failed"
            exit 1
        fi
    fi
}

wait_for_rollback() {
    local namespace
    namespace=$(get_namespace)

    log_info "Waiting for rollback to complete..."

    if kubectl rollout status deployment/gl-005-combustion-control \
        -n "$namespace" --timeout=5m; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed or timed out"
        log_error "Run: kubectl describe deployment gl-005-combustion-control -n $namespace"
        exit 1
    fi
}

verify_rollback() {
    local namespace
    namespace=$(get_namespace)

    log_info "Verifying rollback..."

    # Check pods
    log_info "Checking pods status..."
    kubectl get pods -n "$namespace" -l app=gl-005-combustion-control

    # Check deployment
    log_info "Checking deployment status..."
    kubectl get deployment gl-005-combustion-control -n "$namespace"

    log_success "Rollback verification completed"
}

check_health() {
    local namespace
    namespace=$(get_namespace)

    log_info "Checking application health..."

    # Get pod name
    POD_NAME=$(kubectl get pods -n "$namespace" \
        -l app=gl-005-combustion-control \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "$POD_NAME" ]; then
        log_error "No pods found after rollback"
        exit 1
    fi

    log_info "Testing health endpoint on pod: $POD_NAME"

    # Wait for pod to be ready
    sleep 10

    # Check health endpoint
    if kubectl exec -n "$namespace" "$POD_NAME" -- \
        curl -f http://localhost:8000/api/v1/health &> /dev/null; then
        log_success "Health check passed"
    else
        log_warning "Health check failed - pod may still be starting"
    fi
}

print_summary() {
    local namespace
    namespace=$(get_namespace)

    echo ""
    echo "=========================================="
    echo "  Rollback Summary"
    echo "=========================================="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $namespace"
    echo "Status: Completed"
    echo ""
    echo "Next Steps:"
    echo "1. Monitor logs: kubectl logs -n $namespace -l app=gl-005-combustion-control -f"
    echo "2. Check metrics: kubectl top pods -n $namespace -l app=gl-005-combustion-control"
    echo "3. Review rollout history: kubectl rollout history deployment/gl-005-combustion-control -n $namespace"
    echo ""
    echo "If issues persist, check:"
    echo "- kubectl describe deployment gl-005-combustion-control -n $namespace"
    echo "- kubectl describe pods -n $namespace -l app=gl-005-combustion-control"
    echo "- kubectl get events -n $namespace --sort-by='.lastTimestamp'"
    echo "=========================================="
}

# Main execution
main() {
    print_banner
    check_prerequisites
    show_rollout_history
    confirm_rollback
    perform_rollback
    wait_for_rollback
    verify_rollback
    check_health
    print_summary
}

# Run main function
main
