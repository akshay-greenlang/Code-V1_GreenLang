#!/bin/bash
# ============================================================================
# GreenLang Agent Factory - Monitoring Stack Installation Script
# ============================================================================
# This script installs the complete monitoring stack:
# - Prometheus Operator (kube-prometheus-stack)
# - Grafana dashboards
# - ServiceMonitors for agents
# - Alerting rules
#
# Prerequisites:
# - kubectl configured with cluster access
# - Helm 3.x installed
# - Cluster admin permissions
#
# Usage:
#   ./install.sh [--dry-run] [--namespace NAMESPACE]
#
# Created: 2025-12-03
# Team: Monitoring & Observability
# ============================================================================

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-monitoring}"
HELM_RELEASE="prometheus"
CHART_VERSION="55.5.0"  # Update to latest stable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${1:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi

    # Check Helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed. Please install Helm 3.x first."
        exit 1
    fi

    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Add Helm repository
add_helm_repo() {
    log_info "Adding Prometheus Community Helm repository..."

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    log_success "Helm repository added and updated"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace: ${NAMESPACE}..."

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        kubectl apply -f "${SCRIPT_DIR}/namespace.yaml" --dry-run=client
    else
        kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"
    fi

    log_success "Namespace created"
}

# Create Grafana admin secret
create_grafana_secret() {
    log_info "Creating Grafana admin secret..."

    # Generate random password if not set
    GRAFANA_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-$(openssl rand -base64 12)}"

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        log_info "[DRY-RUN] Would create secret grafana-admin"
    else
        kubectl create secret generic grafana-admin \
            --from-literal=admin-user=admin \
            --from-literal=admin-password="${GRAFANA_PASSWORD}" \
            --namespace "${NAMESPACE}" \
            --dry-run=client -o yaml | kubectl apply -f -

        log_success "Grafana admin secret created"
        log_info "Grafana admin password: ${GRAFANA_PASSWORD}"
        log_warning "Save this password securely!"
    fi
}

# Install Prometheus Stack
install_prometheus_stack() {
    log_info "Installing Prometheus Stack (kube-prometheus-stack)..."
    log_info "  Release: ${HELM_RELEASE}"
    log_info "  Namespace: ${NAMESPACE}"
    log_info "  Chart Version: ${CHART_VERSION}"

    HELM_ARGS=(
        "upgrade" "--install" "${HELM_RELEASE}"
        "prometheus-community/kube-prometheus-stack"
        "--namespace" "${NAMESPACE}"
        "--create-namespace"
        "-f" "${SCRIPT_DIR}/prometheus-values.yaml"
        "--version" "${CHART_VERSION}"
        "--wait"
        "--timeout" "10m"
    )

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        HELM_ARGS+=("--dry-run")
    fi

    helm "${HELM_ARGS[@]}"

    log_success "Prometheus Stack installed successfully"
}

# Apply ServiceMonitors
apply_servicemonitors() {
    log_info "Applying ServiceMonitors for agents..."

    KUBECTL_ARGS=("apply" "-f")
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        KUBECTL_ARGS+=("--dry-run=client")
    fi

    # Fuel Analyzer Agent
    kubectl "${KUBECTL_ARGS[@]}" "${SCRIPT_DIR}/servicemonitor-fuel-analyzer.yaml"
    log_info "  - Fuel Analyzer ServiceMonitor applied"

    # CBAM Importer Agent
    kubectl "${KUBECTL_ARGS[@]}" "${SCRIPT_DIR}/servicemonitor-cbam.yaml"
    log_info "  - CBAM Importer ServiceMonitor applied"

    # Building Energy Agent
    kubectl "${KUBECTL_ARGS[@]}" "${SCRIPT_DIR}/servicemonitor-building-energy.yaml"
    log_info "  - Building Energy ServiceMonitor applied"

    log_success "ServiceMonitors applied"
}

# Apply Prometheus Rules
apply_prometheus_rules() {
    log_info "Applying Prometheus alerting rules..."

    KUBECTL_ARGS=("apply" "-f")
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        KUBECTL_ARGS+=("--dry-run=client")
    fi

    kubectl "${KUBECTL_ARGS[@]}" "${SCRIPT_DIR}/prometheus-rules.yaml"

    log_success "Prometheus alerting rules applied"
}

# Apply Grafana dashboards
apply_grafana_dashboards() {
    log_info "Creating Grafana dashboard ConfigMaps..."

    DASHBOARDS_DIR="${SCRIPT_DIR}/dashboards"

    if [[ ! -d "$DASHBOARDS_DIR" ]]; then
        log_warning "Dashboards directory not found: ${DASHBOARDS_DIR}"
        return
    fi

    for dashboard in "${DASHBOARDS_DIR}"/*.json; do
        if [[ -f "$dashboard" ]]; then
            DASHBOARD_NAME=$(basename "$dashboard" .json)

            if [[ "$DRY_RUN" == "--dry-run" ]]; then
                log_info "[DRY-RUN] Would create ConfigMap: grafana-dashboard-${DASHBOARD_NAME}"
            else
                kubectl create configmap "grafana-dashboard-${DASHBOARD_NAME}" \
                    --from-file="${DASHBOARD_NAME}.json=${dashboard}" \
                    --namespace "${NAMESPACE}" \
                    --dry-run=client -o yaml | \
                kubectl label --local -f - grafana_dashboard=1 -o yaml | \
                kubectl apply -f -

                log_info "  - Dashboard ${DASHBOARD_NAME} applied"
            fi
        fi
    done

    log_success "Grafana dashboards applied"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        log_info "[DRY-RUN] Skipping verification"
        return
    fi

    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=Ready pods \
        --all \
        --namespace "${NAMESPACE}" \
        --timeout=300s || true

    # List deployed resources
    log_info "Deployed resources:"
    echo ""
    echo "Pods:"
    kubectl get pods -n "${NAMESPACE}" -o wide
    echo ""
    echo "Services:"
    kubectl get svc -n "${NAMESPACE}"
    echo ""
    echo "ServiceMonitors:"
    kubectl get servicemonitors -n "${NAMESPACE}" 2>/dev/null || log_warning "No ServiceMonitors found"
    echo ""
    echo "PrometheusRules:"
    kubectl get prometheusrules -n "${NAMESPACE}" 2>/dev/null || log_warning "No PrometheusRules found"

    log_success "Installation verified"
}

# Print access information
print_access_info() {
    log_info ""
    log_info "============================================"
    log_info "Monitoring Stack Installation Complete!"
    log_info "============================================"
    log_info ""
    log_info "Access Prometheus:"
    log_info "  kubectl port-forward svc/prometheus-prometheus -n ${NAMESPACE} 9090:9090"
    log_info "  Open: http://localhost:9090"
    log_info ""
    log_info "Access Grafana:"
    log_info "  kubectl port-forward svc/prometheus-grafana -n ${NAMESPACE} 3000:80"
    log_info "  Open: http://localhost:3000"
    log_info "  Default credentials: admin / (see secret grafana-admin)"
    log_info ""
    log_info "Access Alertmanager:"
    log_info "  kubectl port-forward svc/prometheus-alertmanager -n ${NAMESPACE} 9093:9093"
    log_info "  Open: http://localhost:9093"
    log_info ""
    log_info "For Ingress access, configure your DNS to point to the cluster ingress."
    log_info "============================================"
}

# Main execution
main() {
    echo ""
    echo "============================================"
    echo "GreenLang Monitoring Stack Installation"
    echo "============================================"
    echo ""

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        log_warning "DRY-RUN MODE: No changes will be applied"
        echo ""
    fi

    check_prerequisites
    add_helm_repo
    create_namespace
    create_grafana_secret
    install_prometheus_stack
    apply_servicemonitors
    apply_prometheus_rules
    apply_grafana_dashboards
    verify_installation
    print_access_info
}

# Run main function
main "$@"
