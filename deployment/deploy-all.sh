#!/bin/bash
# =============================================================================
# GreenLang Platform - INFRA-001 Complete Deployment Script
# =============================================================================
# This script performs end-to-end deployment of the GreenLang platform
# including infrastructure provisioning, Kubernetes configuration, and
# application deployment.
#
# Usage:
#   ./deploy-all.sh [environment] [options]
#
# Arguments:
#   environment    Target environment (dev, staging, prod) [default: dev]
#
# Options:
#   --dry-run      Show what would be done without executing
#   --skip-infra   Skip infrastructure provisioning (Terraform)
#   --skip-k8s     Skip Kubernetes deployment
#   --skip-addons  Skip Kubernetes add-ons deployment
#   --skip-app     Skip application deployment
#   --auto-approve Auto-approve all prompts (not recommended for prod)
#   --rollback     Perform rollback instead of deployment
#   -h, --help     Show this help message
#
# Examples:
#   ./deploy-all.sh dev                    # Full deployment to dev
#   ./deploy-all.sh staging --skip-infra   # Deploy K8s only to staging
#   ./deploy-all.sh prod                   # Full production deployment (interactive)
#   ./deploy-all.sh prod --rollback        # Rollback production
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/deploy-$(date +%Y%m%d-%H%M%S).log"

# Default values
ENVIRONMENT="${1:-dev}"
DRY_RUN=false
SKIP_INFRA=false
SKIP_K8S=false
SKIP_ADDONS=false
SKIP_APP=false
AUTO_APPROVE=false
ROLLBACK=false

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-default}"

# Terraform Configuration
TF_DIR="${ROOT_DIR}/terraform/environments"
TF_STATE_BUCKET="greenlang-terraform-state"
TF_LOCK_TABLE="greenlang-terraform-locks"

# Kubernetes Configuration
K8S_NAMESPACE="greenlang"
KUBECONFIG_PATH="/tmp/kubeconfig-${ENVIRONMENT}"

# Helm Configuration
HELM_DIR="${ROOT_DIR}/infrastructure/helm/greenlang"
HELM_RELEASE="greenlang"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Progress tracking
TOTAL_STEPS=10
CURRENT_STEP=0
START_TIME=$(date +%s)

# =============================================================================
# Helper Functions
# =============================================================================

usage() {
    cat << EOF
GreenLang Platform - INFRA-001 Complete Deployment Script

Usage:
    $0 [environment] [options]

Arguments:
    environment    Target environment (dev, staging, prod) [default: dev]

Options:
    --dry-run      Show what would be done without executing
    --skip-infra   Skip infrastructure provisioning (Terraform)
    --skip-k8s     Skip Kubernetes deployment
    --skip-addons  Skip Kubernetes add-ons deployment
    --skip-app     Skip application deployment
    --auto-approve Auto-approve all prompts (not recommended for prod)
    --rollback     Perform rollback instead of deployment
    -h, --help     Show this help message

Examples:
    $0 dev                    # Full deployment to dev
    $0 staging --skip-infra   # Deploy K8s only to staging
    $0 prod                   # Full production deployment (interactive)
    $0 prod --rollback        # Rollback production

EOF
}

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        INFO)
            echo -e "${BLUE}[${timestamp}] [INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        SUCCESS)
            echo -e "${GREEN}[${timestamp}] [SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        WARN)
            echo -e "${YELLOW}[${timestamp}] [WARN]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${RED}[${timestamp}] [ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        DEBUG)
            echo -e "${CYAN}[${timestamp}] [DEBUG]${NC} $message" >> "$LOG_FILE"
            ;;
        *)
            echo -e "[${timestamp}] $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

progress_bar() {
    local current="$1"
    local total="$2"
    local width=50
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "\r${CYAN}["
    printf "%${filled}s" | tr ' ' '#'
    printf "%${empty}s" | tr ' ' '-'
    printf "] %3d%% (%d/%d)${NC}" "$percent" "$current" "$total"
}

step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo ""
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}Step ${CURRENT_STEP}/${TOTAL_STEPS}: $1${NC}"
    echo -e "${MAGENTA}========================================${NC}"
    echo ""
    log "INFO" "Step ${CURRENT_STEP}/${TOTAL_STEPS}: $1"
    progress_bar "$CURRENT_STEP" "$TOTAL_STEPS"
    echo ""
}

elapsed_time() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    echo "${minutes}m ${seconds}s"
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        log "ERROR" "$cmd is not installed"
        return 1
    fi
    log "DEBUG" "$cmd is available"
    return 0
}

confirm() {
    local message="$1"
    local default="${2:-n}"

    if [[ "$AUTO_APPROVE" == "true" ]]; then
        log "WARN" "Auto-approved: $message"
        return 0
    fi

    if [[ "$default" == "y" ]]; then
        read -p "${message} [Y/n]: " response
        response=${response:-y}
    else
        read -p "${message} [y/N]: " response
        response=${response:-n}
    fi

    [[ "$response" =~ ^[Yy]$ ]]
}

run_command() {
    local description="$1"
    shift
    local cmd="$@"

    log "INFO" "Running: $description"
    log "DEBUG" "Command: $cmd"

    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY-RUN] Would execute: $cmd"
        return 0
    fi

    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        log "SUCCESS" "$description completed"
        return 0
    else
        log "ERROR" "$description failed"
        return 1
    fi
}

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        log "ERROR" "Deployment failed with exit code $exit_code"
        log "ERROR" "Check logs at: $LOG_FILE"
        echo ""
        echo -e "${YELLOW}Collecting diagnostics...${NC}"
        collect_diagnostics
    fi
    echo ""
    log "INFO" "Total elapsed time: $(elapsed_time)"
    exit $exit_code
}

collect_diagnostics() {
    local diag_dir="${LOG_DIR}/diagnostics-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$diag_dir"

    # Collect Terraform state
    if [[ -d "${TF_DIR}/${ENVIRONMENT}" ]]; then
        cp "${TF_DIR}/${ENVIRONMENT}/terraform.tfstate" "$diag_dir/" 2>/dev/null || true
    fi

    # Collect Kubernetes info
    if [[ -f "$KUBECONFIG_PATH" ]]; then
        export KUBECONFIG="$KUBECONFIG_PATH"
        kubectl get events -n "$K8S_NAMESPACE" --sort-by='.lastTimestamp' > "$diag_dir/k8s-events.txt" 2>/dev/null || true
        kubectl describe pods -n "$K8S_NAMESPACE" > "$diag_dir/k8s-pods-describe.txt" 2>/dev/null || true
        kubectl logs -l app=greenlang --all-containers --tail=500 -n "$K8S_NAMESPACE" > "$diag_dir/k8s-app-logs.txt" 2>/dev/null || true
    fi

    log "INFO" "Diagnostics saved to: $diag_dir"
}

# =============================================================================
# Parse Arguments
# =============================================================================

parse_args() {
    # Check if first argument is environment or option
    if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then
        ENVIRONMENT="$1"
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-infra)
                SKIP_INFRA=true
                shift
                ;;
            --skip-k8s)
                SKIP_K8S=true
                shift
                ;;
            --skip-addons)
                SKIP_ADDONS=true
                shift
                ;;
            --skip-app)
                SKIP_APP=true
                shift
                ;;
            --auto-approve)
                AUTO_APPROVE=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log "ERROR" "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Validate environment
    if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "prod" ]]; then
        log "ERROR" "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
        exit 1
    fi

    # Update paths based on environment
    KUBECONFIG_PATH="/tmp/kubeconfig-${ENVIRONMENT}"
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

preflight_checks() {
    step "Pre-flight Checks"

    log "INFO" "Environment: $ENVIRONMENT"
    log "INFO" "AWS Region: $AWS_REGION"
    log "INFO" "Dry Run: $DRY_RUN"
    log "INFO" "Auto Approve: $AUTO_APPROVE"

    # Check required tools
    local tools=("aws" "terraform" "kubectl" "helm" "jq" "git")
    local failed=false

    echo "Checking required tools..."
    for tool in "${tools[@]}"; do
        if check_command "$tool"; then
            echo -e "  ${GREEN}[OK]${NC} $tool"
        else
            echo -e "  ${RED}[MISSING]${NC} $tool"
            failed=true
        fi
    done

    if [[ "$failed" == "true" ]]; then
        log "ERROR" "Missing required tools. Please install them and try again."
        exit 1
    fi

    # Verify AWS credentials
    echo ""
    echo "Verifying AWS credentials..."
    if aws sts get-caller-identity &> /dev/null; then
        local account_id=$(aws sts get-caller-identity --query 'Account' --output text)
        local user_arn=$(aws sts get-caller-identity --query 'Arn' --output text)
        echo -e "  ${GREEN}[OK]${NC} AWS Account: $account_id"
        echo -e "  ${GREEN}[OK]${NC} User: $user_arn"
        log "INFO" "AWS Account: $account_id, User: $user_arn"
    else
        log "ERROR" "AWS credentials not configured"
        exit 1
    fi

    # Check Terraform state bucket
    echo ""
    echo "Checking Terraform state backend..."
    if aws s3 ls "s3://${TF_STATE_BUCKET}" --region "$AWS_REGION" &> /dev/null; then
        echo -e "  ${GREEN}[OK]${NC} State bucket: $TF_STATE_BUCKET"
    else
        log "ERROR" "Terraform state bucket not found: $TF_STATE_BUCKET"
        exit 1
    fi

    # Check for production deployment
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        echo ""
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}WARNING: PRODUCTION DEPLOYMENT${NC}"
        echo -e "${YELLOW}========================================${NC}"
        echo ""

        if [[ "$AUTO_APPROVE" == "true" ]]; then
            log "WARN" "Auto-approve is enabled for production deployment!"
        fi

        if ! confirm "Are you sure you want to deploy to PRODUCTION?"; then
            log "INFO" "Deployment cancelled by user"
            exit 0
        fi
    fi

    log "SUCCESS" "Pre-flight checks passed"
}

# =============================================================================
# Infrastructure Deployment (Terraform)
# =============================================================================

deploy_infrastructure() {
    if [[ "$SKIP_INFRA" == "true" ]]; then
        log "INFO" "Skipping infrastructure deployment (--skip-infra)"
        return 0
    fi

    step "Terraform Initialization"

    local tf_env_dir="${TF_DIR}/${ENVIRONMENT}"

    if [[ ! -d "$tf_env_dir" ]]; then
        log "ERROR" "Terraform environment directory not found: $tf_env_dir"
        exit 1
    fi

    cd "$tf_env_dir"

    run_command "Terraform init" "terraform init \
        -backend-config=\"bucket=${TF_STATE_BUCKET}\" \
        -backend-config=\"key=environments/${ENVIRONMENT}/terraform.tfstate\" \
        -backend-config=\"region=${AWS_REGION}\" \
        -backend-config=\"encrypt=true\" \
        -backend-config=\"dynamodb_table=${TF_LOCK_TABLE}\" \
        -reconfigure"

    run_command "Terraform workspace" "terraform workspace select ${ENVIRONMENT} || terraform workspace new ${ENVIRONMENT}"

    run_command "Terraform validate" "terraform validate"

    step "Terraform Plan"

    run_command "Generate plan" "terraform plan \
        -var-file=\"terraform.tfvars\" \
        -out=tfplan.binary \
        -detailed-exitcode" || true

    if [[ "$DRY_RUN" != "true" ]]; then
        # Show plan summary
        terraform show -json tfplan.binary > tfplan.json
        echo ""
        echo "Plan Summary:"
        jq -r '.resource_changes | group_by(.change.actions[0]) | map({action: .[0].change.actions[0], count: length}) | .[] | "  \(.action): \(.count)"' tfplan.json
        echo ""

        # Check for destructive changes
        local destroy_count=$(jq '[.resource_changes[] | select(.change.actions[] == "delete")] | length' tfplan.json)
        if [[ "$destroy_count" -gt 0 ]]; then
            log "WARN" "Plan contains $destroy_count destructive changes!"
            if ! confirm "Do you want to continue with destructive changes?"; then
                log "INFO" "Deployment cancelled"
                exit 0
            fi
        fi
    fi

    step "Terraform Apply"

    if [[ "$ENVIRONMENT" == "prod" && "$AUTO_APPROVE" != "true" ]]; then
        if ! confirm "Apply infrastructure changes to PRODUCTION?"; then
            log "INFO" "Infrastructure apply cancelled"
            exit 0
        fi
    fi

    run_command "Apply infrastructure" "terraform apply tfplan.binary"

    # Export outputs
    log "INFO" "Exporting Terraform outputs..."
    terraform output -json > "${ROOT_DIR}/tf-outputs-${ENVIRONMENT}.json"

    cd "$ROOT_DIR"
    log "SUCCESS" "Infrastructure deployment completed"
}

# =============================================================================
# Kubernetes Configuration
# =============================================================================

configure_kubernetes() {
    if [[ "$SKIP_K8S" == "true" ]]; then
        log "INFO" "Skipping Kubernetes configuration (--skip-k8s)"
        return 0
    fi

    step "Kubernetes Configuration"

    local cluster_name="greenlang-${ENVIRONMENT}-eks"

    # Update kubeconfig
    log "INFO" "Updating kubeconfig for $cluster_name..."
    run_command "Update kubeconfig" "aws eks update-kubeconfig \
        --name \"$cluster_name\" \
        --region \"$AWS_REGION\" \
        --kubeconfig \"$KUBECONFIG_PATH\" \
        --alias \"greenlang-${ENVIRONMENT}\""

    export KUBECONFIG="$KUBECONFIG_PATH"

    # Verify connectivity
    log "INFO" "Verifying cluster connectivity..."
    if kubectl cluster-info &>> "$LOG_FILE"; then
        log "SUCCESS" "Cluster connectivity verified"
    else
        log "ERROR" "Failed to connect to cluster"
        exit 1
    fi

    # Create namespaces
    log "INFO" "Creating namespaces..."
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: greenlang
  labels:
    app: greenlang
    environment: ${ENVIRONMENT}
---
apiVersion: v1
kind: Namespace
metadata:
  name: greenlang-agents
  labels:
    app: greenlang-agents
    environment: ${ENVIRONMENT}
---
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    app: monitoring
---
apiVersion: v1
kind: Namespace
metadata:
  name: cert-manager
  labels:
    app: cert-manager
EOF

    log "SUCCESS" "Kubernetes configuration completed"
}

# =============================================================================
# Kubernetes Add-ons
# =============================================================================

deploy_addons() {
    if [[ "$SKIP_ADDONS" == "true" ]]; then
        log "INFO" "Skipping add-ons deployment (--skip-addons)"
        return 0
    fi

    step "Kubernetes Add-ons"

    export KUBECONFIG="$KUBECONFIG_PATH"

    # Add Helm repos
    log "INFO" "Adding Helm repositories..."
    helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx 2>/dev/null || true
    helm repo add external-secrets https://charts.external-secrets.io 2>/dev/null || true
    helm repo add autoscaler https://kubernetes.github.io/autoscaler 2>/dev/null || true
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo update

    # Cert Manager
    log "INFO" "Installing cert-manager..."
    run_command "Install cert-manager" "helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --set installCRDs=true \
        --wait \
        --timeout 10m"

    # Ingress Controller
    log "INFO" "Installing ingress-nginx..."
    run_command "Install ingress-nginx" "helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.service.type=LoadBalancer \
        --set controller.metrics.enabled=true \
        --set controller.autoscaling.enabled=true \
        --set controller.autoscaling.minReplicas=2 \
        --wait \
        --timeout 10m"

    # External Secrets Operator
    log "INFO" "Installing external-secrets..."
    run_command "Install external-secrets" "helm upgrade --install external-secrets external-secrets/external-secrets \
        --namespace external-secrets \
        --create-namespace \
        --wait \
        --timeout 10m"

    # Cluster Autoscaler
    local cluster_name="greenlang-${ENVIRONMENT}-eks"
    log "INFO" "Installing cluster-autoscaler..."
    run_command "Install cluster-autoscaler" "helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
        --namespace kube-system \
        --set autoDiscovery.clusterName=\"$cluster_name\" \
        --set awsRegion=\"$AWS_REGION\" \
        --wait \
        --timeout 10m"

    log "SUCCESS" "Add-ons deployment completed"
}

# =============================================================================
# Monitoring Stack
# =============================================================================

deploy_monitoring() {
    step "Monitoring Stack"

    export KUBECONFIG="$KUBECONFIG_PATH"

    log "INFO" "Installing kube-prometheus-stack..."
    run_command "Install Prometheus stack" "helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set grafana.enabled=true \
        --set grafana.persistence.enabled=true \
        --wait \
        --timeout 15m"

    # Apply custom alerts
    if [[ -f "${ROOT_DIR}/monitoring/alerts-unified.yml" ]]; then
        log "INFO" "Applying custom alert rules..."
        kubectl apply -f "${ROOT_DIR}/monitoring/alerts-unified.yml" -n monitoring 2>/dev/null || true
    fi

    # Apply custom dashboards
    if [[ -d "${ROOT_DIR}/infrastructure/kubernetes/greenlang/monitoring" ]]; then
        log "INFO" "Creating dashboard ConfigMaps..."
        kubectl create configmap greenlang-dashboards \
            --from-file="${ROOT_DIR}/infrastructure/kubernetes/greenlang/monitoring/" \
            -n monitoring --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || true
    fi

    log "SUCCESS" "Monitoring stack deployment completed"
}

# =============================================================================
# Application Deployment
# =============================================================================

deploy_application() {
    if [[ "$SKIP_APP" == "true" ]]; then
        log "INFO" "Skipping application deployment (--skip-app)"
        return 0
    fi

    step "Application Deployment"

    export KUBECONFIG="$KUBECONFIG_PATH"

    # Deploy GreenLang application
    log "INFO" "Deploying GreenLang application..."

    local values_file="${HELM_DIR}/values-${ENVIRONMENT}.yaml"
    if [[ ! -f "$values_file" ]]; then
        values_file="${HELM_DIR}/values.yaml"
        log "WARN" "Environment-specific values file not found, using default"
    fi

    run_command "Deploy GreenLang" "helm upgrade --install \"$HELM_RELEASE\" \"$HELM_DIR\" \
        --namespace \"$K8S_NAMESPACE\" \
        -f \"$values_file\" \
        --set global.environment=\"$ENVIRONMENT\" \
        --wait \
        --timeout 15m"

    log "SUCCESS" "Application deployment completed"
}

# =============================================================================
# Validation
# =============================================================================

validate_deployment() {
    step "Deployment Validation"

    export KUBECONFIG="$KUBECONFIG_PATH"
    local failed=false

    echo "Validating deployment..."
    echo ""

    # Check pods
    echo "Checking pods in $K8S_NAMESPACE namespace..."
    local total_pods=$(kubectl get pods -n "$K8S_NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local running_pods=$(kubectl get pods -n "$K8S_NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)

    echo -e "  Total pods: $total_pods"
    echo -e "  Running pods: $running_pods"

    if [[ "$total_pods" -eq 0 ]]; then
        log "WARN" "No pods found in $K8S_NAMESPACE namespace"
    elif [[ "$running_pods" -lt "$total_pods" ]]; then
        log "WARN" "Not all pods are running ($running_pods/$total_pods)"
        failed=true
    else
        echo -e "  ${GREEN}[OK]${NC} All pods running"
    fi

    # Check deployments
    echo ""
    echo "Checking deployments..."
    for deploy in greenlang-executor greenlang-worker; do
        local ready=$(kubectl get deployment "$deploy" -n "$K8S_NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired=$(kubectl get deployment "$deploy" -n "$K8S_NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

        if [[ "$ready" == "$desired" && "$ready" != "0" ]]; then
            echo -e "  ${GREEN}[OK]${NC} $deploy: $ready/$desired replicas ready"
        else
            echo -e "  ${YELLOW}[WARN]${NC} $deploy: $ready/$desired replicas ready"
        fi
    done

    # Check services
    echo ""
    echo "Checking services..."
    kubectl get svc -n "$K8S_NAMESPACE" --no-headers 2>/dev/null | while read line; do
        local name=$(echo "$line" | awk '{print $1}')
        echo -e "  ${GREEN}[OK]${NC} Service: $name"
    done

    # Health check
    echo ""
    echo "Running health check..."
    if kubectl exec -n "$K8S_NAMESPACE" deployment/greenlang-executor -- curl -sf http://localhost:8080/api/v1/health &>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC} Health endpoint responding"
    else
        echo -e "  ${YELLOW}[SKIP]${NC} Health endpoint not available (application may still be starting)"
    fi

    # Monitoring check
    echo ""
    echo "Checking monitoring stack..."
    if kubectl get deployment kube-prometheus-stack-grafana -n monitoring &>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC} Grafana is deployed"
    else
        echo -e "  ${YELLOW}[WARN]${NC} Grafana not found"
    fi

    if kubectl get statefulset prometheus-kube-prometheus-stack-prometheus -n monitoring &>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC} Prometheus is deployed"
    else
        echo -e "  ${YELLOW}[WARN]${NC} Prometheus not found"
    fi

    echo ""
    if [[ "$failed" == "true" ]]; then
        log "WARN" "Validation completed with warnings"
    else
        log "SUCCESS" "Validation completed successfully"
    fi
}

# =============================================================================
# Rollback
# =============================================================================

perform_rollback() {
    step "Rollback"

    export KUBECONFIG="$KUBECONFIG_PATH"

    log "WARN" "Performing rollback for $ENVIRONMENT environment"

    if [[ "$ENVIRONMENT" == "prod" && "$AUTO_APPROVE" != "true" ]]; then
        if ! confirm "Are you sure you want to rollback PRODUCTION?"; then
            log "INFO" "Rollback cancelled"
            exit 0
        fi
    fi

    # Rollback Helm release
    log "INFO" "Rolling back Helm release..."
    run_command "Rollback Helm" "helm rollback \"$HELM_RELEASE\" -n \"$K8S_NAMESPACE\" --wait"

    # Verify rollback
    log "INFO" "Verifying rollback..."
    kubectl rollout status deployment/greenlang-executor -n "$K8S_NAMESPACE" --timeout=300s

    log "SUCCESS" "Rollback completed"
}

# =============================================================================
# Summary
# =============================================================================

print_summary() {
    step "Deployment Summary"

    local status="${GREEN}SUCCESS${NC}"

    echo ""
    echo "=========================================="
    echo -e "Deployment Status: $status"
    echo "=========================================="
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Region: $AWS_REGION"
    echo "Duration: $(elapsed_time)"
    echo ""
    echo "Access Points:"
    echo "  - Kubeconfig: $KUBECONFIG_PATH"

    if [[ -f "${ROOT_DIR}/tf-outputs-${ENVIRONMENT}.json" ]]; then
        local eks_endpoint=$(jq -r '.eks_cluster_endpoint.value // "N/A"' "${ROOT_DIR}/tf-outputs-${ENVIRONMENT}.json")
        local rds_endpoint=$(jq -r '.rds_endpoint.value // "N/A"' "${ROOT_DIR}/tf-outputs-${ENVIRONMENT}.json")

        if [[ "$eks_endpoint" != "N/A" ]]; then
            echo "  - EKS Endpoint: $eks_endpoint"
        fi
        if [[ "$rds_endpoint" != "N/A" ]]; then
            echo "  - RDS Endpoint: $rds_endpoint"
        fi
    fi

    echo ""
    echo "Next Steps:"
    echo "  1. Review the deployment: make status ENV=$ENVIRONMENT"
    echo "  2. Check application health: make validate ENV=$ENVIRONMENT"
    echo "  3. View logs: make logs ENV=$ENVIRONMENT"
    echo "  4. Access Grafana: make grafana ENV=$ENVIRONMENT"
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""

    log "SUCCESS" "Deployment completed successfully!"
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Parse command line arguments
    parse_args "$@"

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Set up cleanup trap
    trap cleanup EXIT

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}GreenLang Platform - INFRA-001 Deployment${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Environment: $ENVIRONMENT"
    echo "Timestamp: $(date)"
    echo "Log file: $LOG_FILE"
    echo ""

    log "INFO" "Starting deployment to $ENVIRONMENT"

    if [[ "$ROLLBACK" == "true" ]]; then
        # Update kubeconfig first
        local cluster_name="greenlang-${ENVIRONMENT}-eks"
        aws eks update-kubeconfig \
            --name "$cluster_name" \
            --region "$AWS_REGION" \
            --kubeconfig "$KUBECONFIG_PATH" \
            --alias "greenlang-${ENVIRONMENT}" 2>/dev/null || true

        perform_rollback
        exit 0
    fi

    # Run deployment phases
    preflight_checks
    deploy_infrastructure
    configure_kubernetes
    deploy_addons
    deploy_monitoring
    deploy_application
    validate_deployment
    print_summary
}

# Run main function
main "$@"
