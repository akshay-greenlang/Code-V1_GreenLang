#!/bin/bash
# GL-003 SteamSystemAnalyzer - Rollback Script
# Automated rollback to previous deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
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
    print_success "kubectl found"

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    print_success "Connected to cluster: $(kubectl config current-context)"

    # Check if deployment exists
    if ! kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_error "Deployment $DEPLOYMENT_NAME not found in namespace $NAMESPACE"
        exit 1
    fi
    print_success "Deployment found"
}

# Show rollout history
show_history() {
    print_header "Rollout History"

    kubectl rollout history deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE"
}

# Perform rollback
perform_rollback() {
    print_header "Performing Rollback"

    local revision=${1:-0}

    if [ "$revision" -eq 0 ]; then
        print_info "Rolling back to previous revision..."
        kubectl rollout undo deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE"
    else
        print_info "Rolling back to revision $revision..."
        kubectl rollout undo deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --to-revision="$revision"
    fi

    print_info "Waiting for rollback to complete (timeout: ${TIMEOUT_SECONDS}s)..."
    if ! kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"; then
        print_error "Rollback failed or timed out"
        return 1
    fi

    print_success "Rollback completed successfully"
}

# Restore from backup
restore_from_backup() {
    print_header "Restoring from Backup"

    local backup_file=${1:-}

    if [ -z "$backup_file" ]; then
        # Try to find last backup
        if [ -f "/tmp/gl-003-last-backup" ]; then
            local backup_dir=$(cat /tmp/gl-003-last-backup)
            backup_file="$backup_dir/deployment.yaml"
        else
            print_error "No backup file specified and no automatic backup found"
            exit 1
        fi
    fi

    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        exit 1
    fi

    print_info "Restoring from backup: $backup_file"
    kubectl apply -f "$backup_file"

    print_info "Waiting for restoration to complete..."
    if ! kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout="${TIMEOUT_SECONDS}s"; then
        print_error "Restoration failed or timed out"
        return 1
    fi

    print_success "Restoration completed successfully"
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
}

# Show status
show_status() {
    print_header "Current Status"

    echo -e "\n${BLUE}Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -l "app=$DEPLOYMENT_NAME"

    echo -e "\n${BLUE}Recent Events:${NC}"
    kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$DEPLOYMENT_NAME" --sort-by='.lastTimestamp' | tail -10
}

# Main rollback flow
main() {
    print_header "GL-003 SteamSystemAnalyzer Rollback"

    # Parse arguments
    local method=${1:-auto}
    local revision=${2:-0}
    local backup_file=${3:-}

    print_info "Rollback method: $method"

    # Run rollback steps
    check_prerequisites

    show_history

    # Perform rollback based on method
    case "$method" in
        auto)
            if ! perform_rollback "$revision"; then
                print_error "Automatic rollback failed!"
                exit 1
            fi
            ;;
        backup)
            if ! restore_from_backup "$backup_file"; then
                print_error "Backup restoration failed!"
                exit 1
            fi
            ;;
        *)
            print_error "Unknown rollback method: $method"
            print_info "Usage: $0 [auto|backup] [revision] [backup_file]"
            exit 1
            ;;
    esac

    if ! health_check; then
        print_error "Health checks failed after rollback!"
        exit 1
    fi

    show_status

    print_success "Rollback completed successfully!"
    print_info "To monitor logs: kubectl logs -f -n $NAMESPACE -l app=$DEPLOYMENT_NAME"
}

# Run main function
main "$@"
