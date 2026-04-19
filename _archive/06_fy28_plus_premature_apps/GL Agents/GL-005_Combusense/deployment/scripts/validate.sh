#!/bin/bash
# GL-005 CombustionControlAgent - Validation Script
# Validates deployment configuration and Kubernetes manifests
# Usage: ./validate.sh [environment]

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
VALIDATION_ERRORS=0

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((VALIDATION_ERRORS++))
}

print_banner() {
    echo "=========================================="
    echo "  GL-005 Deployment Validation"
    echo "  Environment: $ENVIRONMENT"
    echo "=========================================="
}

check_tools() {
    log_info "Checking required tools..."

    local tools_ok=true

    # Check kubectl
    if command -v kubectl &> /dev/null; then
        log_success "kubectl found: $(kubectl version --client --short 2>/dev/null || kubectl version --client 2>/dev/null | head -n1)"
    else
        log_error "kubectl not found"
        tools_ok=false
    fi

    # Check kustomize
    if command -v kustomize &> /dev/null; then
        log_success "kustomize found: $(kustomize version --short 2>/dev/null || kustomize version 2>/dev/null)"
    else
        log_error "kustomize not found"
        tools_ok=false
    fi

    # Check docker (optional)
    if command -v docker &> /dev/null; then
        log_success "docker found: $(docker --version)"
    else
        log_warning "docker not found (optional)"
    fi

    if [ "$tools_ok" = false ]; then
        log_error "Some required tools are missing"
        exit 1
    fi
}

check_cluster_connection() {
    log_info "Checking Kubernetes cluster connection..."

    if kubectl cluster-info &> /dev/null; then
        log_success "Connected to Kubernetes cluster"
        kubectl cluster-info | head -n 2
    else
        log_error "Cannot connect to Kubernetes cluster"
        log_error "Check kubeconfig: kubectl config view"
    fi
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

validate_directory_structure() {
    log_info "Validating directory structure..."

    local dirs=(
        "$KUSTOMIZE_DIR/base"
        "$KUSTOMIZE_DIR/overlays/dev"
        "$KUSTOMIZE_DIR/overlays/staging"
        "$KUSTOMIZE_DIR/overlays/production"
    )

    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            log_success "Directory exists: $dir"
        else
            log_error "Directory missing: $dir"
        fi
    done
}

validate_base_manifests() {
    log_info "Validating base manifests..."

    local base_dir="$KUSTOMIZE_DIR/base"
    local required_files=(
        "kustomization.yaml"
    )

    for file in "${required_files[@]}"; do
        if [ -f "$base_dir/$file" ]; then
            log_success "Base file exists: $file"
        else
            log_error "Base file missing: $file"
        fi
    done
}

validate_overlay_manifests() {
    log_info "Validating overlay manifests for $ENVIRONMENT..."

    local overlay_dir="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"

    if [ ! -d "$overlay_dir" ]; then
        log_error "Overlay directory not found: $overlay_dir"
        return 1
    fi

    if [ ! -f "$overlay_dir/kustomization.yaml" ]; then
        log_error "kustomization.yaml not found in overlay"
        return 1
    fi

    log_success "Overlay manifests exist for $ENVIRONMENT"
}

build_and_validate() {
    log_info "Building and validating manifests with kustomize..."

    local overlay_dir="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"
    local output_file="/tmp/gl-005-$ENVIRONMENT-manifests.yaml"

    # Build manifests
    if kustomize build "$overlay_dir" > "$output_file" 2>&1; then
        log_success "Kustomize build successful"
        log_info "Manifest size: $(wc -l < "$output_file") lines"
    else
        log_error "Kustomize build failed"
        cat "$output_file"
        return 1
    fi

    # Validate with kubectl
    log_info "Validating with kubectl dry-run..."
    if kubectl apply --dry-run=client -f "$output_file" &> /dev/null; then
        log_success "kubectl client-side validation passed"
    else
        log_error "kubectl validation failed"
        kubectl apply --dry-run=client -f "$output_file"
    fi

    # Validate with kubectl server-side (if cluster is accessible)
    if kubectl cluster-info &> /dev/null; then
        log_info "Validating with kubectl server-side dry-run..."
        if kubectl apply --dry-run=server -f "$output_file" &> /dev/null; then
            log_success "kubectl server-side validation passed"
        else
            log_warning "kubectl server-side validation failed (cluster may have policies)"
        fi
    fi
}

check_resource_limits() {
    log_info "Checking resource limits..."

    local overlay_dir="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"
    local output_file="/tmp/gl-005-$ENVIRONMENT-manifests.yaml"

    kustomize build "$overlay_dir" > "$output_file"

    # Check if deployment has resource limits
    if grep -q "resources:" "$output_file"; then
        log_success "Resource limits are defined"
    else
        log_warning "No resource limits found (not recommended for production)"
    fi

    # Check specific limits
    if grep -q "limits:" "$output_file" && grep -q "requests:" "$output_file"; then
        log_success "Both limits and requests are defined"
    else
        log_warning "Missing either limits or requests"
    fi
}

check_security_context() {
    log_info "Checking security context..."

    local overlay_dir="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"
    local output_file="/tmp/gl-005-$ENVIRONMENT-manifests.yaml"

    kustomize build "$overlay_dir" > "$output_file"

    # Check for runAsNonRoot
    if grep -q "runAsNonRoot: true" "$output_file"; then
        log_success "runAsNonRoot is enabled"
    else
        log_error "runAsNonRoot is not enabled (security risk)"
    fi

    # Check for readOnlyRootFilesystem
    if grep -q "readOnlyRootFilesystem: true" "$output_file"; then
        log_success "readOnlyRootFilesystem is enabled"
    else
        log_warning "readOnlyRootFilesystem is not enabled (recommended)"
    fi
}

check_health_probes() {
    log_info "Checking health probes..."

    local overlay_dir="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"
    local output_file="/tmp/gl-005-$ENVIRONMENT-manifests.yaml"

    kustomize build "$overlay_dir" > "$output_file"

    # Check for liveness probe
    if grep -q "livenessProbe:" "$output_file"; then
        log_success "Liveness probe is configured"
    else
        log_error "Liveness probe is missing"
    fi

    # Check for readiness probe
    if grep -q "readinessProbe:" "$output_file"; then
        log_success "Readiness probe is configured"
    else
        log_error "Readiness probe is missing"
    fi
}

check_hpa() {
    log_info "Checking HorizontalPodAutoscaler..."

    local overlay_dir="$KUSTOMIZE_DIR/overlays/$ENVIRONMENT"
    local output_file="/tmp/gl-005-$ENVIRONMENT-manifests.yaml"

    kustomize build "$overlay_dir" > "$output_file"

    if grep -q "kind: HorizontalPodAutoscaler" "$output_file"; then
        log_success "HPA is configured"

        # Check min/max replicas
        if grep -q "minReplicas:" "$output_file" && grep -q "maxReplicas:" "$output_file"; then
            log_success "HPA min/max replicas are configured"
        else
            log_error "HPA min/max replicas are missing"
        fi
    else
        log_warning "HPA is not configured (may be intentional for $ENVIRONMENT)"
    fi
}

print_summary() {
    echo ""
    echo "=========================================="
    echo "  Validation Summary"
    echo "=========================================="
    echo "Environment: $ENVIRONMENT"
    echo "Validation Errors: $VALIDATION_ERRORS"
    echo ""

    if [ $VALIDATION_ERRORS -eq 0 ]; then
        log_success "All validation checks passed!"
        echo ""
        echo "Next steps:"
        echo "1. Deploy: ./deploy.sh $ENVIRONMENT"
        echo "2. Monitor: kubectl get pods -n greenlang -l app=gl-005-combustion-control"
        echo "=========================================="
        exit 0
    else
        log_error "$VALIDATION_ERRORS validation error(s) found"
        echo ""
        echo "Please fix the errors before deploying."
        echo "=========================================="
        exit 1
    fi
}

# Main execution
main() {
    print_banner
    check_tools
    check_cluster_connection
    validate_environment
    validate_directory_structure
    validate_base_manifests
    validate_overlay_manifests
    build_and_validate
    check_resource_limits
    check_security_context
    check_health_probes
    check_hpa
    print_summary
}

# Run main function
main
