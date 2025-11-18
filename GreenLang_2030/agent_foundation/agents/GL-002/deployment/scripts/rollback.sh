#!/bin/bash
# GL-002 BoilerEfficiencyOptimizer - Rollback Script
# Rolls back GL-002 deployment to previous revision

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
NAMESPACE="greenlang"
REVISION="${2:-}"  # Empty means rollback to previous
DRY_RUN="${DRY_RUN:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${BLUE}==>${NC} $1"
}

# Banner
print_banner() {
    echo ""
    echo "=================================================="
    echo "  GL-002 Kubernetes Rollback"
    echo "=================================================="
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Namespace:   $NAMESPACE"
    echo "Revision:    ${REVISION:-previous}"
    echo "Dry Run:     $DRY_RUN"
    echo ""
}

# Print usage
usage() {
    echo "Usage: $0 [ENVIRONMENT] [REVISION] [OPTIONS]"
    echo ""
    echo "Environments:"
    echo "  dev          Rollback in development environment"
    echo "  staging      Rollback in staging environment"
    echo "  production   Rollback in production environment (default)"
    echo ""
    echo "Revision:"
    echo "  <number>     Specific revision number (e.g., 3)"
    echo "  (empty)      Rollback to previous revision (default)"
    echo ""
    echo "Options:"
    echo "  DRY_RUN=true             Perform dry-run only (no actual rollback)"
    echo "  WAIT_TIMEOUT=300         Timeout for rollout status (seconds)"
    echo ""
    echo "Examples:"
    echo "  $0 production              # Rollback to previous revision"
    echo "  $0 production 5            # Rollback to revision 5"
    echo "  DRY_RUN=true $0 staging    # Dry-run rollback"
    echo ""
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    log_success "kubectl installed"

    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    log_success "Connected to Kubernetes cluster"

    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace '$NAMESPACE' does not exist."
        exit 1
    fi
    log_success "Namespace '$NAMESPACE' exists"

    # Check deployment exists
    if ! kubectl get deployment gl-002-boiler-efficiency -n "$NAMESPACE" &> /dev/null; then
        log_error "Deployment 'gl-002-boiler-efficiency' not found in namespace '$NAMESPACE'."
        exit 1
    fi
    log_success "Deployment exists"
}

# Set environment-specific variables
set_environment() {
    log_step "Setting environment-specific configuration..."

    case "$ENVIRONMENT" in
        dev|development)
            ENVIRONMENT="dev"
            NAMESPACE="greenlang-dev"
            ;;
        staging)
            ENVIRONMENT="staging"
            NAMESPACE="greenlang-staging"
            ;;
        production|prod)
            ENVIRONMENT="production"
            NAMESPACE="greenlang"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            usage
            ;;
    esac

    log_success "Environment set: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
}

# Show rollout history
show_rollout_history() {
    log_step "Deployment rollout history..."

    kubectl rollout history deployment/gl-002-boiler-efficiency -n "$NAMESPACE"

    echo ""
    log_info "Current deployment status:"
    kubectl get deployment gl-002-boiler-efficiency -n "$NAMESPACE"
}

# Get revision details
get_revision_details() {
    log_step "Analyzing revision details..."

    # Get current revision
    local current_revision=$(kubectl rollout history deployment/gl-002-boiler-efficiency -n "$NAMESPACE" | tail -n 1 | awk '{print $1}')
    log_info "Current revision: $current_revision"

    # If no revision specified, use previous
    if [ -z "$REVISION" ]; then
        REVISION=$((current_revision - 1))
        log_info "Rolling back to previous revision: $REVISION"
    else
        log_info "Rolling back to specified revision: $REVISION"
    fi

    # Validate revision exists
    if ! kubectl rollout history deployment/gl-002-boiler-efficiency -n "$NAMESPACE" --revision="$REVISION" &> /dev/null; then
        log_error "Revision $REVISION does not exist."
        echo ""
        log_info "Available revisions:"
        kubectl rollout history deployment/gl-002-boiler-efficiency -n "$NAMESPACE"
        exit 1
    fi

    # Show revision details
    echo ""
    log_info "Revision $REVISION details:"
    kubectl rollout history deployment/gl-002-boiler-efficiency -n "$NAMESPACE" --revision="$REVISION"
}

# Confirm rollback
confirm_rollback() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "DRY RUN MODE - Rollback will not be executed"
        return
    fi

    if [ "$ENVIRONMENT" = "production" ]; then
        echo ""
        log_warning "⚠️  PRODUCTION ROLLBACK WARNING ⚠️"
        echo ""
        echo "You are about to rollback GL-002 in PRODUCTION environment."
        echo "This will affect live traffic and customer operations."
        echo ""
        echo "Environment:  $ENVIRONMENT"
        echo "Namespace:    $NAMESPACE"
        echo "Revision:     $REVISION"
        echo ""

        read -p "Are you sure you want to proceed? (type 'yes' to confirm): " confirm

        if [ "$confirm" != "yes" ]; then
            log_error "Rollback cancelled by user."
            exit 0
        fi

        log_success "Rollback confirmed"
    fi
}

# Execute rollback
execute_rollback() {
    log_step "Executing rollback..."

    local dry_run_flag=""
    if [ "$DRY_RUN" = "true" ]; then
        dry_run_flag="--dry-run=client"
        log_info "DRY RUN MODE - No changes will be applied"
    fi

    # Perform rollback
    if [ -n "$REVISION" ]; then
        log_info "Rolling back to revision $REVISION..."
        if kubectl rollout undo deployment/gl-002-boiler-efficiency -n "$NAMESPACE" --to-revision="$REVISION" $dry_run_flag; then
            log_success "Rollback initiated"
        else
            log_error "Rollback failed"
            exit 1
        fi
    else
        log_info "Rolling back to previous revision..."
        if kubectl rollout undo deployment/gl-002-boiler-efficiency -n "$NAMESPACE" $dry_run_flag; then
            log_success "Rollback initiated"
        else
            log_error "Rollback failed"
            exit 1
        fi
    fi
}

# Wait for rollback
wait_for_rollback() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping rollback wait (DRY_RUN=true)"
        return
    fi

    log_step "Waiting for rollback to complete..."

    log_info "Waiting for deployment rollout (timeout: ${WAIT_TIMEOUT}s)..."
    if kubectl rollout status deployment/gl-002-boiler-efficiency -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s"; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed or timed out"

        # Show pod status for debugging
        log_info "Current pod status:"
        kubectl get pods -n "$NAMESPACE" -l app=gl-002-boiler-efficiency

        log_info "Recent events:"
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -n 10

        exit 1
    fi
}

# Verify rollback
verify_rollback() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping rollback verification (DRY_RUN=true)"
        return
    fi

    log_step "Verifying rollback..."

    # Check deployment status
    local desired_replicas=$(kubectl get deployment gl-002-boiler-efficiency -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    local ready_replicas=$(kubectl get deployment gl-002-boiler-efficiency -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')

    log_info "Desired replicas: $desired_replicas"
    log_info "Ready replicas: $ready_replicas"

    if [ "$ready_replicas" = "$desired_replicas" ]; then
        log_success "All replicas are ready"
    else
        log_error "Not all replicas are ready ($ready_replicas/$desired_replicas)"
        exit 1
    fi

    # Check pod health
    log_info "Checking pod health..."
    local unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" -l app=gl-002-boiler-efficiency --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)

    if [ "$unhealthy_pods" -eq 0 ]; then
        log_success "All pods are healthy"
    else
        log_error "$unhealthy_pods unhealthy pod(s) found"
        kubectl get pods -n "$NAMESPACE" -l app=gl-002-boiler-efficiency
        exit 1
    fi

    # Show current revision
    local new_revision=$(kubectl rollout history deployment/gl-002-boiler-efficiency -n "$NAMESPACE" | tail -n 1 | awk '{print $1}')
    log_success "Rolled back to revision: $new_revision"
}

# Post-rollback actions
post_rollback() {
    log_step "Post-rollback actions..."

    # Show deployment info
    log_info "Deployment summary:"
    kubectl get deployment gl-002-boiler-efficiency -n "$NAMESPACE"

    log_info "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app=gl-002-boiler-efficiency

    log_info "Rollout history (after rollback):"
    kubectl rollout history deployment/gl-002-boiler-efficiency -n "$NAMESPACE"

    # Create incident log
    local incident_file="$DEPLOYMENT_DIR/.rollback-log-$(date +%Y%m%d-%H%M%S).json"
    cat > "$incident_file" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": "$ENVIRONMENT",
  "namespace": "$NAMESPACE",
  "revision": "$REVISION",
  "rolled_back_by": "${USER:-unknown}",
  "reason": "Manual rollback triggered"
}
EOF
    log_success "Rollback log saved to $incident_file"

    # Send alert (optional)
    log_warning "Remember to:"
    echo "  1. Notify team about rollback"
    echo "  2. Create incident ticket"
    echo "  3. Investigate root cause"
    echo "  4. Update runbook if needed"
}

# Emergency rollback (skip confirmations)
emergency_rollback() {
    log_error "EMERGENCY ROLLBACK INITIATED"

    check_prerequisites
    set_environment
    show_rollout_history
    get_revision_details

    log_warning "Skipping confirmation (EMERGENCY=true)"

    execute_rollback
    wait_for_rollback
    verify_rollback
    post_rollback

    echo ""
    log_success "Emergency rollback completed!"
    echo ""
}

# Main execution
main() {
    # Parse arguments
    if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
        usage
    fi

    # Check for emergency mode
    if [ "${EMERGENCY:-false}" = "true" ]; then
        emergency_rollback
        return
    fi

    print_banner
    check_prerequisites
    set_environment
    show_rollout_history
    get_revision_details
    confirm_rollback
    execute_rollback
    wait_for_rollback
    verify_rollback
    post_rollback

    echo ""
    log_success "Rollback completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor deployment: kubectl get pods -n $NAMESPACE -w"
    echo "  2. Check logs: kubectl logs -n $NAMESPACE -l app=gl-002-boiler-efficiency -f"
    echo "  3. Verify health: kubectl port-forward -n $NAMESPACE svc/gl-002-boiler-efficiency 8000:80"
    echo "  4. Create incident report"
    echo ""
}

# Run main
main "$@"
