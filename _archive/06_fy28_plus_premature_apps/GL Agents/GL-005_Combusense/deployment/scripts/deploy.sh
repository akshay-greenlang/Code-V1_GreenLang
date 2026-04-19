#!/bin/bash
# GL-005 CombustionControlAgent - Deployment Script
# Deploys GL-005 to Kubernetes using Kustomize
# Usage: ./deploy.sh [environment] [options]

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
KUSTOMIZE_DIR="$DEPLOYMENT_DIR/kustomize"

# Default values
ENVIRONMENT="${1:-dev}"
DRY_RUN="${2:-false}"
SKIP_VALIDATION="${3:-false}"

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
    echo "  GL-005 CombustionControlAgent Deployment"
    echo "  Environment: $ENVIRONMENT"
    echo "=========================================="
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    # Check kustomize
    if ! command -v kustomize &> /dev/null; then
        log_error "kustomize not found. Please install kustomize."
        exit 1
    fi

    # Check kubectl cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check kubeconfig."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

validate_environment() {
    log_info "Validating environment: $ENVIRONMENT"

    case $ENVIRONMENT in
        dev|staging|production)
            log_success "Environment is valid: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: dev, staging, production"
            exit 1
            ;;
    esac
}

validate_manifests() {
    if [ "$SKIP_VALIDATION" == "true" ]; then
        log_warning "Skipping manifest validation"
        return 0
    fi

    log_info "Validating Kubernetes manifests..."

    OVERLAY_DIR="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"

    if [ ! -d "$OVERLAY_DIR" ]; then
        log_error "Overlay directory not found: $OVERLAY_DIR"
        exit 1
    fi

    # Build and validate manifests
    if kustomize build "$OVERLAY_DIR" > /tmp/gl-005-manifests.yaml; then
        log_success "Manifest validation passed"
    else
        log_error "Manifest validation failed"
        exit 1
    fi

    # Validate with kubectl
    if kubectl apply --dry-run=client -f /tmp/gl-005-manifests.yaml &> /dev/null; then
        log_success "kubectl validation passed"
    else
        log_error "kubectl validation failed"
        exit 1
    fi
}

check_namespace() {
    local namespace
    case $ENVIRONMENT in
        dev)
            namespace="greenlang-dev"
            ;;
        staging)
            namespace="greenlang-staging"
            ;;
        production)
            namespace="greenlang"
            ;;
    esac

    log_info "Checking namespace: $namespace"

    if kubectl get namespace "$namespace" &> /dev/null; then
        log_success "Namespace exists: $namespace"
    else
        log_warning "Namespace does not exist: $namespace"
        log_info "Creating namespace: $namespace"
        kubectl create namespace "$namespace"
        log_success "Namespace created: $namespace"
    fi
}

backup_current_deployment() {
    local namespace
    case $ENVIRONMENT in
        dev)
            namespace="greenlang-dev"
            ;;
        staging)
            namespace="greenlang-staging"
            ;;
        production)
            namespace="greenlang"
            ;;
    esac

    log_info "Backing up current deployment..."

    BACKUP_DIR="$DEPLOYMENT_DIR/backups"
    mkdir -p "$BACKUP_DIR"

    BACKUP_FILE="$BACKUP_DIR/gl-005-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).yaml"

    if kubectl get deployment -n "$namespace" -l app=gl-005-combustion-control &> /dev/null; then
        kubectl get all,configmap,secret,ingress,hpa,pdb -n "$namespace" \
            -l app=gl-005-combustion-control -o yaml > "$BACKUP_FILE"
        log_success "Backup saved to: $BACKUP_FILE"
    else
        log_warning "No existing deployment found to backup"
    fi
}

deploy() {
    local namespace
    case $ENVIRONMENT in
        dev)
            namespace="greenlang-dev"
            ;;
        staging)
            namespace="greenlang-staging"
            ;;
        production)
            namespace="greenlang"
            ;;
    esac

    log_info "Deploying GL-005 to $ENVIRONMENT..."

    OVERLAY_DIR="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"

    if [ "$DRY_RUN" == "true" ]; then
        log_warning "DRY RUN MODE - No changes will be applied"
        kustomize build "$OVERLAY_DIR" | kubectl apply --dry-run=server -f -
        log_success "Dry run completed successfully"
    else
        kustomize build "$OVERLAY_DIR" | kubectl apply -f -
        log_success "Deployment applied successfully"
    fi
}

wait_for_rollout() {
    local namespace
    case $ENVIRONMENT in
        dev)
            namespace="greenlang-dev"
            ;;
        staging)
            namespace="greenlang-staging"
            ;;
        production)
            namespace="greenlang"
            ;;
    esac

    if [ "$DRY_RUN" == "true" ]; then
        return 0
    fi

    log_info "Waiting for deployment rollout..."

    # Wait for deployment to be ready (timeout: 5 minutes)
    if kubectl rollout status deployment/gl-005-combustion-control \
        -n "$namespace" --timeout=5m; then
        log_success "Deployment rolled out successfully"
    else
        log_error "Deployment rollout failed or timed out"
        log_error "Run: kubectl describe deployment gl-005-combustion-control -n $namespace"
        exit 1
    fi
}

verify_deployment() {
    local namespace
    case $ENVIRONMENT in
        dev)
            namespace="greenlang-dev"
            ;;
        staging)
            namespace="greenlang-staging"
            ;;
        production)
            namespace="greenlang"
            ;;
    esac

    if [ "$DRY_RUN" == "true" ]; then
        return 0
    fi

    log_info "Verifying deployment..."

    # Check pods
    log_info "Checking pods..."
    kubectl get pods -n "$namespace" -l app=gl-005-combustion-control

    # Check services
    log_info "Checking services..."
    kubectl get svc -n "$namespace" -l app=gl-005-combustion-control

    # Check HPA
    log_info "Checking HPA..."
    kubectl get hpa -n "$namespace" -l app=gl-005-combustion-control

    # Check ingress
    log_info "Checking ingress..."
    kubectl get ingress -n "$namespace" -l app=gl-005-combustion-control

    log_success "Deployment verification completed"
}

print_summary() {
    local namespace
    case $ENVIRONMENT in
        dev)
            namespace="greenlang-dev"
            ;;
        staging)
            namespace="greenlang-staging"
            ;;
        production)
            namespace="greenlang"
            ;;
    esac

    echo ""
    echo "=========================================="
    echo "  Deployment Summary"
    echo "=========================================="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $namespace"
    echo "Dry Run: $DRY_RUN"
    echo ""
    echo "Next Steps:"
    echo "1. Check logs: kubectl logs -n $namespace -l app=gl-005-combustion-control -f"
    echo "2. Check metrics: kubectl top pods -n $namespace -l app=gl-005-combustion-control"
    echo "3. Access application: kubectl port-forward -n $namespace svc/gl-005-combustion-control 8000:80"
    echo ""
    echo "Rollback (if needed): ./rollback.sh $ENVIRONMENT"
    echo "=========================================="
}

# Main execution
main() {
    print_banner
    check_prerequisites
    validate_environment
    validate_manifests
    check_namespace
    backup_current_deployment
    deploy
    wait_for_rollout
    verify_deployment
    print_summary
}

# Run main function
main
