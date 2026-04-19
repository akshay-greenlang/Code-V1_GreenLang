#!/bin/bash
# ==============================================================================
# GL-VCCI Main Deployment Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Deploy GL-VCCI platform to Kubernetes

OPTIONS:
    -e, --environment ENV    Environment: dev, staging, prod (required)
    -v, --version VERSION    Version to deploy (required)
    -s, --strategy STRATEGY  Deployment strategy: rolling, blue-green, canary (default: rolling)
    -d, --dry-run           Perform a dry run without applying changes
    -h, --help              Show this help message

EXAMPLES:
    # Deploy to production with rolling update
    $0 -e prod -v v2.0.0

    # Deploy to staging with canary strategy
    $0 -e staging -v v2.0.1 -s canary

    # Dry run for production
    $0 -e prod -v v2.0.0 --dry-run
EOF
}

# Parse arguments
ENVIRONMENT=""
VERSION=""
STRATEGY="rolling"
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -s|--strategy)
            STRATEGY="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="--dry-run=client"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$ENVIRONMENT" ] || [ -z "$VERSION" ]; then
    log_error "Environment and version are required"
    usage
    exit 1
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT (must be dev, staging, or prod)"
    exit 1
fi

# Validate strategy
if [[ ! "$STRATEGY" =~ ^(rolling|blue-green|canary)$ ]]; then
    log_error "Invalid strategy: $STRATEGY (must be rolling, blue-green, or canary)"
    exit 1
fi

# Set namespace based on environment
NAMESPACE="vcci-${ENVIRONMENT}"

log_info "Deployment Configuration:"
log_info "  Environment: $ENVIRONMENT"
log_info "  Namespace: $NAMESPACE"
log_info "  Version: $VERSION"
log_info "  Strategy: $STRATEGY"
log_info "  Dry Run: ${DRY_RUN:-no}"

# Verify kubectl is configured
if ! kubectl cluster-info &>/dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

log_info "Connected to cluster: $(kubectl config current-context)"

# Verify namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    log_warn "Namespace $NAMESPACE does not exist, creating..."
    kubectl create namespace "$NAMESPACE" $DRY_RUN
fi

# Build and push Docker images
log_info "Building Docker images..."
$SCRIPT_DIR/build-images.sh "$VERSION"

# Apply base configurations
log_info "Applying base configurations..."
kubectl apply -f "$PROJECT_ROOT/k8s/namespace.yaml" $DRY_RUN

# Apply secrets (if using external secrets)
if [ -f "$PROJECT_ROOT/k8s/external-secrets.yaml" ]; then
    log_info "Applying external secrets configuration..."
    kubectl apply -f "$PROJECT_ROOT/k8s/external-secrets.yaml" $DRY_RUN
fi

# Apply network policies
log_info "Applying network policies..."
kubectl apply -f "$PROJECT_ROOT/k8s/network-policy.yaml" $DRY_RUN

# Apply resource quotas
log_info "Applying resource quotas..."
kubectl apply -f "$PROJECT_ROOT/k8s/resource-quota.yaml" $DRY_RUN

# Deploy based on strategy
case $STRATEGY in
    rolling)
        log_info "Deploying with rolling update strategy..."
        $SCRIPT_DIR/rolling-deploy.sh "$ENVIRONMENT" "$VERSION" "$DRY_RUN"
        ;;
    blue-green)
        log_info "Deploying with blue-green strategy..."
        $SCRIPT_DIR/blue-green-deploy.sh "$ENVIRONMENT" "$VERSION" "$DRY_RUN"
        ;;
    canary)
        log_info "Deploying with canary strategy..."
        $SCRIPT_DIR/canary-deploy.sh "$ENVIRONMENT" "$VERSION" "$DRY_RUN"
        ;;
esac

# Apply HPA
log_info "Applying Horizontal Pod Autoscalers..."
kubectl apply -f "$PROJECT_ROOT/k8s/hpa.yaml" $DRY_RUN

# Apply PDB
log_info "Applying Pod Disruption Budgets..."
kubectl apply -f "$PROJECT_ROOT/k8s/pdb.yaml" $DRY_RUN

# Wait for rollout (if not dry run)
if [ -z "$DRY_RUN" ]; then
    log_info "Waiting for deployments to be ready..."
    kubectl rollout status deployment/vcci-backend-api -n "$NAMESPACE" --timeout=10m
    kubectl rollout status deployment/vcci-worker -n "$NAMESPACE" --timeout=10m

    # Run smoke tests
    log_info "Running smoke tests..."
    if $SCRIPT_DIR/smoke-test.sh "$NAMESPACE"; then
        log_info "Smoke tests passed!"
    else
        log_error "Smoke tests failed!"
        log_warn "Consider rolling back the deployment"
        exit 1
    fi

    # Display deployment status
    log_info "Deployment complete!"
    log_info "Deployment status:"
    kubectl get deployments -n "$NAMESPACE"
    kubectl get pods -n "$NAMESPACE"
    kubectl get hpa -n "$NAMESPACE"
else
    log_info "Dry run complete. Review the output above."
fi

log_info "Deployment completed successfully!"
