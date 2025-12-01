#!/bin/bash
# GL-011 FUELCRAFT - Deployment Script
# Deploys GL-011 to Kubernetes with pre-flight checks and verification

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
DRY_RUN="${DRY_RUN:-false}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${BLUE}==>${NC} $1"
}

# Banner
print_banner() {
    echo ""
    echo "=================================================="
    echo "  GL-011 FUELCRAFT Kubernetes Deployment"
    echo "  Fuel Mix Optimization Agent"
    echo "=================================================="
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Namespace:   $NAMESPACE"
    echo "Dry Run:     $DRY_RUN"
    echo ""
}

# Print usage
usage() {
    echo "Usage: $0 [ENVIRONMENT] [OPTIONS]"
    echo ""
    echo "Environments:"
    echo "  dev          Deploy to development environment"
    echo "  staging      Deploy to staging environment"
    echo "  production   Deploy to production environment (default)"
    echo ""
    echo "Options:"
    echo "  DRY_RUN=true             Perform dry-run only (no actual deployment)"
    echo "  SKIP_VALIDATION=true     Skip pre-flight validation checks"
    echo "  WAIT_TIMEOUT=300         Timeout for rollout status (seconds)"
    echo ""
    echo "Examples:"
    echo "  $0 production"
    echo "  DRY_RUN=true $0 staging"
    echo "  SKIP_VALIDATION=true $0 dev"
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
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_success "Namespace '$NAMESPACE' exists"
    else
        log_warning "Namespace '$NAMESPACE' does not exist. Creating..."
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace '$NAMESPACE' created"
    fi

    # Check for kustomize
    if command -v kustomize &> /dev/null; then
        log_success "kustomize installed"
    else
        log_warning "kustomize not found. Will use kubectl kustomize instead."
    fi
}

# Set environment-specific variables
set_environment() {
    log_step "Setting environment-specific configuration..."

    case "$ENVIRONMENT" in
        dev|development)
            ENVIRONMENT="dev"
            NAMESPACE="greenlang-dev"
            REPLICAS=1
            IMAGE_TAG="dev-latest"
            ;;
        staging)
            ENVIRONMENT="staging"
            NAMESPACE="greenlang-staging"
            REPLICAS=2
            IMAGE_TAG="staging-1.0.0-rc.1"
            ;;
        production|prod)
            ENVIRONMENT="production"
            NAMESPACE="greenlang"
            REPLICAS=3
            IMAGE_TAG="1.0.0"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            usage
            ;;
    esac

    log_success "Environment set: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Replicas: $REPLICAS"
    log_info "Image Tag: $IMAGE_TAG"
}

# Run validation
run_validation() {
    if [ "$SKIP_VALIDATION" = "true" ]; then
        log_warning "Skipping validation (SKIP_VALIDATION=true)"
        return
    fi

    log_step "Running manifest validation..."

    if [ -f "$SCRIPT_DIR/validate-manifests.sh" ]; then
        if bash "$SCRIPT_DIR/validate-manifests.sh"; then
            log_success "Validation passed"
        else
            log_error "Validation failed. Fix errors before deploying."
            exit 1
        fi
    else
        log_warning "Validation script not found. Skipping validation."
    fi
}

# Pre-flight checks
pre_flight_checks() {
    log_step "Running pre-flight checks..."

    # Check if secrets exist
    if kubectl get secret gl-011-secrets -n "$NAMESPACE" &> /dev/null; then
        log_success "Secrets exist"
    else
        log_error "Secrets not found. Please create gl-011-secrets before deploying."
        echo ""
        echo "To create secrets, run:"
        echo "  kubectl create secret generic gl-011-secrets \\"
        echo "    --from-literal=database_url=<DB_URL> \\"
        echo "    --from-literal=redis_url=<REDIS_URL> \\"
        echo "    --from-literal=api_key=<API_KEY> \\"
        echo "    --from-literal=jwt_secret=<JWT_SECRET> \\"
        echo "    --from-literal=market_data_api_key=<MARKET_API_KEY> \\"
        echo "    -n $NAMESPACE"
        echo ""
        exit 1
    fi

    # Check for existing deployment
    if kubectl get deployment gl-011-fuelcraft -n "$NAMESPACE" &> /dev/null; then
        log_warning "Existing deployment found. This will trigger a rolling update."

        # Show current deployment info
        CURRENT_REPLICAS=$(kubectl get deployment gl-011-fuelcraft -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        CURRENT_IMAGE=$(kubectl get deployment gl-011-fuelcraft -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')

        log_info "Current replicas: $CURRENT_REPLICAS"
        log_info "Current image: $CURRENT_IMAGE"
    else
        log_info "No existing deployment found. This will be a fresh deployment."
    fi
}

# Deploy manifests
deploy_manifests() {
    log_step "Deploying manifests..."

    local dry_run_flag=""
    if [ "$DRY_RUN" = "true" ]; then
        dry_run_flag="--dry-run=client"
        log_info "DRY RUN MODE - No changes will be applied"
    fi

    # Deploy in order (dependencies first)
    local manifests=(
        "serviceaccount.yaml"
        "resourcequota.yaml"
        "limitrange.yaml"
        "configmap.yaml"
        "secret.yaml"
        "service.yaml"
        "deployment.yaml"
        "hpa.yaml"
        "pdb.yaml"
        "networkpolicy.yaml"
        "ingress.yaml"
        "servicemonitor.yaml"
    )

    for manifest in "${manifests[@]}"; do
        local file="$DEPLOYMENT_DIR/$manifest"

        if [ ! -f "$file" ]; then
            log_warning "Manifest not found: $manifest (skipping)"
            continue
        fi

        log_info "Applying $manifest..."
        if kubectl apply -f "$file" -n "$NAMESPACE" $dry_run_flag; then
            log_success "Applied $manifest"
        else
            log_error "Failed to apply $manifest"
            exit 1
        fi
    done
}

# Deploy with Kustomize (alternative)
deploy_with_kustomize() {
    log_step "Deploying with Kustomize..."

    local overlay_dir="$DEPLOYMENT_DIR/kustomize/overlays/$ENVIRONMENT"

    if [ ! -d "$overlay_dir" ]; then
        log_error "Kustomize overlay not found: $ENVIRONMENT"
        exit 1
    fi

    local dry_run_flag=""
    if [ "$DRY_RUN" = "true" ]; then
        dry_run_flag="--dry-run=client"
        log_info "DRY RUN MODE - No changes will be applied"
    fi

    log_info "Building kustomize overlay for $ENVIRONMENT..."
    if kubectl apply -k "$overlay_dir" $dry_run_flag; then
        log_success "Kustomize deployment successful"
    else
        log_error "Kustomize deployment failed"
        exit 1
    fi
}

# Wait for rollout
wait_for_rollout() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping rollout wait (DRY_RUN=true)"
        return
    fi

    log_step "Waiting for rollout to complete..."

    log_info "Waiting for deployment rollout (timeout: ${WAIT_TIMEOUT}s)..."
    if kubectl rollout status deployment/gl-011-fuelcraft -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s"; then
        log_success "Deployment rollout completed"
    else
        log_error "Deployment rollout failed or timed out"

        # Show pod status for debugging
        log_info "Current pod status:"
        kubectl get pods -n "$NAMESPACE" -l app=gl-011-fuelcraft

        log_info "Recent events:"
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -n 10

        exit 1
    fi
}

# Verify deployment
verify_deployment() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "Skipping deployment verification (DRY_RUN=true)"
        return
    fi

    log_step "Verifying deployment..."

    # Check deployment status
    local desired_replicas=$(kubectl get deployment gl-011-fuelcraft -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    local ready_replicas=$(kubectl get deployment gl-011-fuelcraft -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')

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
    local unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" -l app=gl-011-fuelcraft --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)

    if [ "$unhealthy_pods" -eq 0 ]; then
        log_success "All pods are healthy"
    else
        log_error "$unhealthy_pods unhealthy pod(s) found"
        kubectl get pods -n "$NAMESPACE" -l app=gl-011-fuelcraft
        exit 1
    fi
}

# Post-deployment actions
post_deployment() {
    log_step "Post-deployment actions..."

    # Show deployment info
    log_info "Deployment summary:"
    kubectl get deployment gl-011-fuelcraft -n "$NAMESPACE"

    log_info "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app=gl-011-fuelcraft

    log_info "Service:"
    kubectl get service gl-011-fuelcraft -n "$NAMESPACE"

    log_info "HPA:"
    kubectl get hpa gl-011-fuelcraft-hpa -n "$NAMESPACE" 2>/dev/null || log_warning "HPA not found"

    log_info "PDB:"
    kubectl get pdb gl-011-fuelcraft-pdb -n "$NAMESPACE" 2>/dev/null || log_warning "PDB not found"

    # Save deployment metadata
    local metadata_file="$DEPLOYMENT_DIR/.deployment-metadata.json"
    cat > "$metadata_file" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": "$ENVIRONMENT",
  "namespace": "$NAMESPACE",
  "image_tag": "$IMAGE_TAG",
  "replicas": $REPLICAS,
  "deployed_by": "${USER:-unknown}",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
}
EOF
    log_success "Deployment metadata saved to $metadata_file"
}

# Main execution
main() {
    # Parse arguments
    if [[ "$1" =~ ^(-h|--help)$ ]] 2>/dev/null; then
        usage
    fi

    print_banner
    check_prerequisites
    set_environment
    run_validation
    pre_flight_checks

    # Choose deployment method
    if [ -d "$DEPLOYMENT_DIR/kustomize/overlays/$ENVIRONMENT" ]; then
        log_info "Using Kustomize for deployment"
        deploy_with_kustomize
    else
        log_info "Using direct manifest deployment"
        deploy_manifests
    fi

    wait_for_rollout
    verify_deployment
    post_deployment

    echo ""
    log_success "Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor deployment: kubectl get pods -n $NAMESPACE -w"
    echo "  2. Check logs: kubectl logs -n $NAMESPACE -l app=gl-011-fuelcraft -f"
    echo "  3. Verify health: kubectl port-forward -n $NAMESPACE svc/gl-011-fuelcraft 8080:8080"
    echo "     Then visit: http://localhost:8080/api/v1/health"
    echo ""
}

# Run main
main "$@"
