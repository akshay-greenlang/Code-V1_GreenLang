#!/bin/bash
# GL-003 SteamSystemAnalyzer - Deployment Script
# Automated deployment with safety checks and rollback capability

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
KUSTOMIZE_DIR="$DEPLOYMENT_DIR/kustomize"
NAMESPACE="greenlang"
DEPLOYMENT_NAME="gl-003-steam-system-analyzer"
TIMEOUT_SECONDS=300

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}===========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}===========================================${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    print_success "kubectl found: $(kubectl version --client --short)"

    # Check kustomize
    if ! command -v kustomize &> /dev/null; then
        print_error "kustomize not found. Please install kustomize."
        exit 1
    fi
    print_success "kustomize found: $(kustomize version --short)"

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    print_success "Connected to cluster: $(kubectl config current-context)"
}

# Validate manifests
validate_manifests() {
    print_header "Validating Manifests"

    local environment=$1
    local overlay_path="$KUSTOMIZE_DIR/overlays/$environment"

    if [ ! -d "$overlay_path" ]; then
        print_error "Environment overlay not found: $environment"
        exit 1
    fi

    # Build and validate
    print_info "Building kustomize overlay for $environment..."
    if ! kustomize build "$overlay_path" > /tmp/gl-003-manifests.yaml; then
        print_error "Kustomize build failed"
        exit 1
    fi
    print_success "Kustomize build successful"

    # Dry-run validation
    print_info "Validating with kubectl dry-run..."
    if ! kubectl apply --dry-run=client -f /tmp/gl-003-manifests.yaml &> /dev/null; then
        print_error "Manifest validation failed"
        exit 1
    fi
    print_success "Manifest validation passed"
}

# Create backup
create_backup() {
    print_header "Creating Backup"

    local backup_dir="/tmp/gl-003-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup deployment
    if kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_info "Backing up current deployment..."
        kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o yaml > "$backup_dir/deployment.yaml"
        print_success "Deployment backed up to $backup_dir"
    else
        print_warning "No existing deployment to backup"
    fi

    echo "$backup_dir" > /tmp/gl-003-last-backup
    print_success "Backup location saved: $backup_dir"
}

# Deploy application
deploy_application() {
    print_header "Deploying Application"

    local environment=$1
    local overlay_path="$KUSTOMIZE_DIR/overlays/$environment"

    # Apply manifests
    print_info "Applying manifests for $environment environment..."
    if ! kubectl apply -k "$overlay_path"; then
        print_error "Deployment failed"
        return 1
    fi
    print_success "Manifests applied successfully"

    # Wait for rollout
    print_info "Waiting for rollout to complete (timeout: ${TIMEOUT_SECONDS}s)..."
    if ! kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"; then
        print_error "Rollout failed or timed out"
        return 1
    fi
    print_success "Rollout completed successfully"
}

# Health check
health_check() {
    print_header "Running Health Checks"

    # Check pod status
    print_info "Checking pod status..."
    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l "app=$DEPLOYMENT_NAME" -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -o "True" | wc -l)
    local total_pods=$(kubectl get pods -n "$NAMESPACE" -l "app=$DEPLOYMENT_NAME" --no-headers | wc -l)

    if [ "$ready_pods" -eq 0 ] || [ "$ready_pods" -lt "$total_pods" ]; then
        print_error "Not all pods are ready: $ready_pods/$total_pods"
        return 1
    fi
    print_success "All pods ready: $ready_pods/$total_pods"

    # Check service endpoints
    print_info "Checking service endpoints..."
    local endpoints=$(kubectl get endpoints "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)
    if [ "$endpoints" -eq 0 ]; then
        print_error "No service endpoints available"
        return 1
    fi
    print_success "Service endpoints: $endpoints"

    # Test health endpoint (if ingress is available)
    print_info "Testing health endpoint..."
    local service_ip=$(kubectl get svc "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    if kubectl run curl-test --image=curlimages/curl:latest --rm -i --restart=Never -- \
        curl -f -s "http://$service_ip:80/api/v1/health" &> /dev/null; then
        print_success "Health endpoint responding"
    else
        print_warning "Health endpoint check failed (may not be critical)"
    fi
}

# Show deployment status
show_status() {
    print_header "Deployment Status"

    echo -e "\n${BLUE}Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -l "app=$DEPLOYMENT_NAME"

    echo -e "\n${BLUE}Services:${NC}"
    kubectl get svc -n "$NAMESPACE" -l "app=$DEPLOYMENT_NAME"

    echo -e "\n${BLUE}Ingress:${NC}"
    kubectl get ingress -n "$NAMESPACE" -l "app=$DEPLOYMENT_NAME" 2>/dev/null || echo "No ingress found"

    echo -e "\n${BLUE}HPA:${NC}"
    kubectl get hpa -n "$NAMESPACE" -l "app=$DEPLOYMENT_NAME"
}

# Main deployment flow
main() {
    print_header "GL-003 SteamSystemAnalyzer Deployment"

    # Parse arguments
    local environment=${1:-dev}
    local skip_backup=${2:-false}

    print_info "Environment: $environment"
    print_info "Skip backup: $skip_backup"

    # Run deployment steps
    check_prerequisites

    validate_manifests "$environment"

    if [ "$skip_backup" != "true" ]; then
        create_backup
    fi

    if ! deploy_application "$environment"; then
        print_error "Deployment failed!"
        print_warning "To rollback, run: ./rollback.sh"
        exit 1
    fi

    if ! health_check; then
        print_error "Health checks failed!"
        print_warning "To rollback, run: ./rollback.sh"
        exit 1
    fi

    show_status

    print_success "Deployment completed successfully!"
    print_info "To monitor logs: kubectl logs -f -n $NAMESPACE -l app=$DEPLOYMENT_NAME"
    print_info "To rollback: cd $SCRIPT_DIR && ./rollback.sh"
}

# Run main function
main "$@"
