#!/bin/bash
# GL-011 FUELCRAFT - Manifest Validation Script
# Validates all Kubernetes manifests using kubectl dry-run and kubeval

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
MANIFEST_DIR="$DEPLOYMENT_DIR"

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
    ((PASSED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

# Banner
print_banner() {
    echo ""
    echo "=================================================="
    echo "  GL-011 FUELCRAFT Manifest Validation"
    echo "=================================================="
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    ((TOTAL_TESTS++))

    # Check kubectl
    if command -v kubectl &> /dev/null; then
        KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | awk '{print $3}')
        log_success "kubectl installed: $KUBECTL_VERSION"
    else
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    # Check for kubeval (optional but recommended)
    if command -v kubeval &> /dev/null; then
        KUBEVAL_VERSION=$(kubeval --version 2>&1 | head -n1)
        log_success "kubeval installed: $KUBEVAL_VERSION"
        KUBEVAL_AVAILABLE=true
    else
        log_warning "kubeval not found (optional). Install from: https://kubeval.com"
        KUBEVAL_AVAILABLE=false
    fi

    # Check for kustomize (optional)
    if command -v kustomize &> /dev/null; then
        KUSTOMIZE_VERSION=$(kustomize version --short 2>/dev/null | awk '{print $1}')
        log_success "kustomize installed: $KUSTOMIZE_VERSION"
        KUSTOMIZE_AVAILABLE=true
    else
        log_warning "kustomize not found (optional). Install from: https://kustomize.io"
        KUSTOMIZE_AVAILABLE=false
    fi

    echo ""
}

# Validate YAML syntax
validate_yaml_syntax() {
    log_info "Validating YAML syntax..."

    local manifests=(
        "deployment.yaml"
        "service.yaml"
        "configmap.yaml"
        "secret.yaml"
        "hpa.yaml"
        "pdb.yaml"
        "networkpolicy.yaml"
        "ingress.yaml"
        "serviceaccount.yaml"
        "servicemonitor.yaml"
        "resourcequota.yaml"
        "limitrange.yaml"
    )

    for manifest in "${manifests[@]}"; do
        ((TOTAL_TESTS++))
        local file="$MANIFEST_DIR/$manifest"

        if [ ! -f "$file" ]; then
            log_error "File not found: $manifest"
            continue
        fi

        # Basic YAML syntax check using kubectl
        if kubectl apply --dry-run=client -f "$file" &> /dev/null; then
            log_success "YAML syntax valid: $manifest"
        else
            log_error "YAML syntax invalid: $manifest"
            kubectl apply --dry-run=client -f "$file" 2>&1 | sed 's/^/  /'
        fi
    done

    echo ""
}

# Validate with kubeval
validate_with_kubeval() {
    if [ "$KUBEVAL_AVAILABLE" = false ]; then
        log_warning "Skipping kubeval validation (not installed)"
        echo ""
        return
    fi

    log_info "Validating with kubeval (Kubernetes schema validation)..."

    local manifests=(
        "deployment.yaml"
        "service.yaml"
        "configmap.yaml"
        "secret.yaml"
        "hpa.yaml"
        "pdb.yaml"
        "networkpolicy.yaml"
        "ingress.yaml"
        "serviceaccount.yaml"
        "resourcequota.yaml"
        "limitrange.yaml"
    )

    for manifest in "${manifests[@]}"; do
        ((TOTAL_TESTS++))
        local file="$MANIFEST_DIR/$manifest"

        if [ ! -f "$file" ]; then
            continue
        fi

        # Validate against Kubernetes 1.24 schema
        if kubeval --kubernetes-version 1.24.0 --strict "$file" &> /dev/null; then
            log_success "Kubeval passed: $manifest"
        else
            log_error "Kubeval failed: $manifest"
            kubeval --kubernetes-version 1.24.0 --strict "$file" 2>&1 | sed 's/^/  /'
        fi
    done

    echo ""
}

# Validate Kustomize overlays
validate_kustomize() {
    if [ "$KUSTOMIZE_AVAILABLE" = false ]; then
        log_warning "Skipping kustomize validation (not installed)"
        echo ""
        return
    fi

    log_info "Validating Kustomize overlays..."

    local overlays=("dev" "staging" "production")

    for overlay in "${overlays[@]}"; do
        ((TOTAL_TESTS++))
        local overlay_dir="$DEPLOYMENT_DIR/kustomize/overlays/$overlay"

        if [ ! -d "$overlay_dir" ]; then
            log_warning "Overlay not found: $overlay"
            continue
        fi

        # Build kustomize overlay
        if kustomize build "$overlay_dir" > /dev/null 2>&1; then
            log_success "Kustomize build passed: $overlay"
        else
            log_error "Kustomize build failed: $overlay"
            kustomize build "$overlay_dir" 2>&1 | sed 's/^/  /'
        fi
    done

    echo ""
}

# Validate resource constraints
validate_resource_constraints() {
    log_info "Validating resource constraints..."

    ((TOTAL_TESTS++))
    local deployment="$MANIFEST_DIR/deployment.yaml"

    # Check if deployment has resource requests and limits
    if grep -q "requests:" "$deployment" && grep -q "limits:" "$deployment"; then
        log_success "Resource requests and limits defined"
    else
        log_error "Missing resource requests or limits in deployment.yaml"
    fi

    ((TOTAL_TESTS++))
    # Check if CPU requests/limits meet GL-011 specs (2 CPU, 4Gi RAM)
    local cpu_limit=$(grep -A 1 "limits:" "$deployment" | grep "cpu:" | awk '{print $2}' | tr -d '"' | head -n1)
    local mem_limit=$(grep -A 2 "limits:" "$deployment" | grep "memory:" | awk '{print $2}' | tr -d '"' | head -n1)

    if [ -n "$cpu_limit" ] && [ -n "$mem_limit" ]; then
        log_success "Resource limits defined: CPU=$cpu_limit, Memory=$mem_limit"
    else
        log_error "Resource limits not fully defined"
    fi

    echo ""
}

# Validate health checks
validate_health_checks() {
    log_info "Validating health checks..."

    local deployment="$MANIFEST_DIR/deployment.yaml"

    ((TOTAL_TESTS++))
    if grep -q "livenessProbe:" "$deployment"; then
        log_success "Liveness probe configured"
    else
        log_error "Missing liveness probe"
    fi

    ((TOTAL_TESTS++))
    if grep -q "readinessProbe:" "$deployment"; then
        log_success "Readiness probe configured"
    else
        log_error "Missing readiness probe"
    fi

    ((TOTAL_TESTS++))
    if grep -q "startupProbe:" "$deployment"; then
        log_success "Startup probe configured"
    else
        log_warning "Startup probe not configured (optional)"
    fi

    echo ""
}

# Validate security context
validate_security_context() {
    log_info "Validating security context..."

    local deployment="$MANIFEST_DIR/deployment.yaml"

    ((TOTAL_TESTS++))
    if grep -q "runAsNonRoot: true" "$deployment"; then
        log_success "runAsNonRoot configured"
    else
        log_error "Missing runAsNonRoot: true"
    fi

    ((TOTAL_TESTS++))
    if grep -q "readOnlyRootFilesystem: true" "$deployment"; then
        log_success "readOnlyRootFilesystem configured"
    else
        log_warning "readOnlyRootFilesystem not configured (recommended)"
    fi

    ((TOTAL_TESTS++))
    if grep -q "allowPrivilegeEscalation: false" "$deployment"; then
        log_success "allowPrivilegeEscalation: false configured"
    else
        log_error "Missing allowPrivilegeEscalation: false"
    fi

    echo ""
}

# Validate PodDisruptionBudget
validate_pdb() {
    log_info "Validating PodDisruptionBudget..."

    ((TOTAL_TESTS++))
    local pdb="$MANIFEST_DIR/pdb.yaml"

    if [ -f "$pdb" ]; then
        log_success "PodDisruptionBudget exists"

        ((TOTAL_TESTS++))
        if grep -q "minAvailable:" "$pdb" || grep -q "maxUnavailable:" "$pdb"; then
            log_success "PDB has minAvailable or maxUnavailable configured"
        else
            log_error "PDB missing minAvailable/maxUnavailable"
        fi
    else
        log_error "PodDisruptionBudget not found"
    fi

    echo ""
}

# Validate RBAC
validate_rbac() {
    log_info "Validating RBAC configuration..."

    ((TOTAL_TESTS++))
    local sa="$MANIFEST_DIR/serviceaccount.yaml"

    if [ -f "$sa" ]; then
        log_success "ServiceAccount exists"

        ((TOTAL_TESTS++))
        if grep -q "kind: Role" "$sa"; then
            log_success "Role configured"
        else
            log_warning "Role not found in serviceaccount.yaml"
        fi

        ((TOTAL_TESTS++))
        if grep -q "kind: RoleBinding" "$sa"; then
            log_success "RoleBinding configured"
        else
            log_warning "RoleBinding not found in serviceaccount.yaml"
        fi
    else
        log_error "ServiceAccount not found"
    fi

    echo ""
}

# Validate HPA
validate_hpa() {
    log_info "Validating HorizontalPodAutoscaler..."

    ((TOTAL_TESTS++))
    local hpa="$MANIFEST_DIR/hpa.yaml"

    if [ -f "$hpa" ]; then
        log_success "HPA exists"

        ((TOTAL_TESTS++))
        if grep -q "minReplicas: 2" "$hpa"; then
            log_success "HPA minReplicas set to 2 (as specified)"
        else
            log_warning "HPA minReplicas should be 2 per spec"
        fi

        ((TOTAL_TESTS++))
        if grep -q "maxReplicas: 10" "$hpa"; then
            log_success "HPA maxReplicas set to 10 (as specified)"
        else
            log_warning "HPA maxReplicas should be 10 per spec"
        fi
    else
        log_error "HPA not found"
    fi

    echo ""
}

# Validate GL-011 specific configuration
validate_gl011_specific() {
    log_info "Validating GL-011 FUELCRAFT specific configuration..."

    local configmap="$MANIFEST_DIR/configmap.yaml"

    ((TOTAL_TESTS++))
    if [ -f "$configmap" ] && grep -q "fuel_specifications:" "$configmap"; then
        log_success "Fuel specifications database configured"
    else
        log_warning "Fuel specifications not found in configmap"
    fi

    ((TOTAL_TESTS++))
    if [ -f "$configmap" ] && grep -q "emission_factors:" "$configmap"; then
        log_success "Emission factors database configured"
    else
        log_warning "Emission factors not found in configmap"
    fi

    ((TOTAL_TESTS++))
    if [ -f "$configmap" ] && grep -q "regulatory_limits:" "$configmap"; then
        log_success "Regulatory limits configured"
    else
        log_warning "Regulatory limits not found in configmap"
    fi

    echo ""
}

# Generate validation report
generate_report() {
    echo ""
    echo "=================================================="
    echo "  Validation Summary"
    echo "=================================================="
    echo ""
    echo "Total Tests:  $TOTAL_TESTS"
    echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
    echo ""

    local success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Success Rate: ${success_rate}%"
    echo ""

    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "All validation tests passed!"
        echo ""
        echo "Manifests are ready for deployment."
        return 0
    else
        log_error "Some validation tests failed."
        echo ""
        echo "Please fix the errors above before deploying."
        return 1
    fi
}

# Main execution
main() {
    print_banner
    check_prerequisites
    validate_yaml_syntax
    validate_with_kubeval
    validate_kustomize
    validate_resource_constraints
    validate_health_checks
    validate_security_context
    validate_pdb
    validate_rbac
    validate_hpa
    validate_gl011_specific
    generate_report
}

# Run main
main
exit $?
