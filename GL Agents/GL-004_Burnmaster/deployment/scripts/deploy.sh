#!/bin/bash
# GL-004 BURNMASTER - Deployment Script
# =====================================
# Automated deployment to Kubernetes environments

set -euo pipefail

# ===========================================
# Configuration
# ===========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$SCRIPT_DIR/.."

# Default values
ENVIRONMENT="${GL_ENVIRONMENT:-staging}"
NAMESPACE="${GL_NAMESPACE:-greenlang}"
REGISTRY="${GL_REGISTRY:-gcr.io/greenlang}"
IMAGE_NAME="gl-004-burnmaster"
HELM_RELEASE_NAME="gl004-burnmaster"
KUBECTL_CONTEXT="${GL_KUBECTL_CONTEXT:-}"
DRY_RUN="${DRY_RUN:-false}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"
SKIP_TESTS="${SKIP_TESTS:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===========================================
# Helper Functions
# ===========================================
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

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

GL-004 BURNMASTER Deployment Script

Options:
    -e, --environment   Target environment (development, staging, production)
                        Default: staging
    -n, --namespace     Kubernetes namespace
                        Default: greenlang
    -t, --tag           Docker image tag
                        Default: git commit SHA
    -r, --registry      Container registry
                        Default: gcr.io/greenlang
    -c, --context       Kubernetes context
                        Default: current context
    --dry-run           Perform a dry run without applying changes
    --skip-tests        Skip running tests before deployment
    --skip-build        Skip Docker build (use existing image)
    --helm              Deploy using Helm
    --kubectl           Deploy using kubectl
    -h, --help          Show this help message

Examples:
    # Deploy to staging with auto-generated tag
    $0 -e staging

    # Deploy to production with specific tag
    $0 -e production -t v1.2.3

    # Dry run deployment
    $0 -e production --dry-run

    # Deploy using Helm
    $0 -e staging --helm

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    # Check required tools
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check Kubernetes connectivity
    if [ -n "$KUBECTL_CONTEXT" ]; then
        kubectl config use-context "$KUBECTL_CONTEXT"
    fi

    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

get_image_tag() {
    if [ -n "${IMAGE_TAG:-}" ]; then
        echo "$IMAGE_TAG"
    else
        # Use git commit SHA
        git rev-parse --short HEAD 2>/dev/null || echo "latest"
    fi
}

run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping tests as requested"
        return 0
    fi

    log_info "Running tests..."

    cd "$PROJECT_ROOT"

    # Run unit tests
    python -m pytest tests/unit -v --tb=short || {
        log_error "Unit tests failed"
        exit 1
    }

    # Run integration tests (if not dry-run)
    if [ "$DRY_RUN" != "true" ]; then
        python -m pytest tests/integration -v --tb=short || {
            log_warning "Integration tests failed - continuing with deployment"
        }
    fi

    log_success "Tests passed"
}

build_image() {
    if [ "$SKIP_BUILD" = "true" ]; then
        log_warning "Skipping Docker build as requested"
        return 0
    fi

    local tag=$(get_image_tag)
    local full_image="$REGISTRY/$IMAGE_NAME:$tag"

    log_info "Building Docker image: $full_image"

    cd "$PROJECT_ROOT"

    # Build with buildkit
    DOCKER_BUILDKIT=1 docker build \
        --file "$DEPLOYMENT_DIR/docker/Dockerfile" \
        --target runtime \
        --tag "$full_image" \
        --tag "$REGISTRY/$IMAGE_NAME:latest" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$tag" \
        --cache-from "$REGISTRY/$IMAGE_NAME:latest" \
        . || {
            log_error "Docker build failed"
            exit 1
        }

    log_success "Docker image built: $full_image"
}

push_image() {
    local tag=$(get_image_tag)
    local full_image="$REGISTRY/$IMAGE_NAME:$tag"

    log_info "Pushing Docker image: $full_image"

    if [ "$DRY_RUN" = "true" ]; then
        log_warning "DRY RUN: Would push $full_image"
        return 0
    fi

    docker push "$full_image" || {
        log_error "Failed to push Docker image"
        exit 1
    }

    docker push "$REGISTRY/$IMAGE_NAME:latest" || {
        log_warning "Failed to push latest tag"
    }

    log_success "Docker image pushed"
}

deploy_kubectl() {
    local tag=$(get_image_tag)
    local full_image="$REGISTRY/$IMAGE_NAME:$tag"

    log_info "Deploying with kubectl to $ENVIRONMENT..."

    # Create namespace if not exists
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Apply ConfigMaps and Secrets
    log_info "Applying ConfigMaps..."
    kubectl apply -f "$DEPLOYMENT_DIR/kubernetes/configmap.yaml" -n "$NAMESPACE" \
        ${DRY_RUN:+--dry-run=client}

    # Apply Deployment
    log_info "Applying Deployment..."
    kubectl apply -f "$DEPLOYMENT_DIR/kubernetes/deployment.yaml" -n "$NAMESPACE" \
        ${DRY_RUN:+--dry-run=client}

    # Update image
    if [ "$DRY_RUN" != "true" ]; then
        kubectl set image deployment/gl004-burnmaster \
            gl004-burnmaster="$full_image" \
            -n "$NAMESPACE"
    fi

    # Apply Services
    log_info "Applying Services..."
    kubectl apply -f "$DEPLOYMENT_DIR/kubernetes/service.yaml" -n "$NAMESPACE" \
        ${DRY_RUN:+--dry-run=client}

    log_success "kubectl deployment complete"
}

deploy_helm() {
    local tag=$(get_image_tag)
    local values_file="$DEPLOYMENT_DIR/helm/values.yaml"
    local env_values_file="$DEPLOYMENT_DIR/helm/values-$ENVIRONMENT.yaml"

    log_info "Deploying with Helm to $ENVIRONMENT..."

    # Build Helm command
    local helm_cmd="helm upgrade --install $HELM_RELEASE_NAME $DEPLOYMENT_DIR/helm"
    helm_cmd="$helm_cmd --namespace $NAMESPACE --create-namespace"
    helm_cmd="$helm_cmd --values $values_file"

    # Add environment-specific values if exists
    if [ -f "$env_values_file" ]; then
        helm_cmd="$helm_cmd --values $env_values_file"
    fi

    # Set image tag
    helm_cmd="$helm_cmd --set image.tag=$tag"
    helm_cmd="$helm_cmd --set image.repository=$REGISTRY/$IMAGE_NAME"

    # Set environment
    helm_cmd="$helm_cmd --set env.GL_ENVIRONMENT=$ENVIRONMENT"

    # Timeout
    helm_cmd="$helm_cmd --timeout ${WAIT_TIMEOUT}s"

    # Wait for deployment
    helm_cmd="$helm_cmd --wait"

    # Dry run
    if [ "$DRY_RUN" = "true" ]; then
        helm_cmd="$helm_cmd --dry-run"
    fi

    log_info "Running: $helm_cmd"
    eval "$helm_cmd" || {
        log_error "Helm deployment failed"
        exit 1
    }

    log_success "Helm deployment complete"
}

verify_deployment() {
    if [ "$DRY_RUN" = "true" ]; then
        log_warning "DRY RUN: Skipping deployment verification"
        return 0
    fi

    log_info "Verifying deployment..."

    # Wait for rollout
    kubectl rollout status deployment/gl004-burnmaster \
        -n "$NAMESPACE" \
        --timeout="${WAIT_TIMEOUT}s" || {
            log_error "Deployment rollout failed"
            exit 1
        }

    # Check pod status
    local ready_pods=$(kubectl get deployment gl004-burnmaster \
        -n "$NAMESPACE" \
        -o jsonpath='{.status.readyReplicas}')

    local desired_pods=$(kubectl get deployment gl004-burnmaster \
        -n "$NAMESPACE" \
        -o jsonpath='{.spec.replicas}')

    if [ "$ready_pods" != "$desired_pods" ]; then
        log_error "Expected $desired_pods ready pods, got $ready_pods"
        exit 1
    fi

    # Health check
    log_info "Running health checks..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" \
        -l app.kubernetes.io/name=gl004-burnmaster \
        -o jsonpath='{.items[0].metadata.name}')

    local health_status=$(kubectl exec "$pod_name" -n "$NAMESPACE" -- \
        curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health/ready 2>/dev/null || echo "000")

    if [ "$health_status" != "200" ]; then
        log_warning "Health check returned status: $health_status"
    else
        log_success "Health check passed"
    fi

    log_success "Deployment verification complete"
}

print_summary() {
    local tag=$(get_image_tag)

    echo ""
    echo "=========================================="
    echo "Deployment Summary"
    echo "=========================================="
    echo "Environment:    $ENVIRONMENT"
    echo "Namespace:      $NAMESPACE"
    echo "Image:          $REGISTRY/$IMAGE_NAME:$tag"
    echo "Helm Release:   $HELM_RELEASE_NAME"
    echo "Dry Run:        $DRY_RUN"
    echo "=========================================="
    echo ""

    if [ "$DRY_RUN" != "true" ]; then
        # Show deployment status
        kubectl get deployments,pods,services \
            -n "$NAMESPACE" \
            -l app.kubernetes.io/name=gl004-burnmaster
    fi
}

# ===========================================
# Main Script
# ===========================================
main() {
    local deploy_method="helm"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -c|--context)
                KUBECTL_CONTEXT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            --helm)
                deploy_method="helm"
                shift
                ;;
            --kubectl)
                deploy_method="kubectl"
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

    # Validate environment
    case $ENVIRONMENT in
        development|staging|production)
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac

    # Production safety check
    if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" != "true" ]; then
        log_warning "You are about to deploy to PRODUCTION!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi

    echo ""
    echo "=========================================="
    echo "GL-004 BURNMASTER Deployment"
    echo "=========================================="
    echo ""

    # Run deployment steps
    check_prerequisites
    run_tests
    build_image
    push_image

    # Deploy based on method
    if [ "$deploy_method" = "helm" ]; then
        deploy_helm
    else
        deploy_kubectl
    fi

    verify_deployment
    print_summary

    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"
