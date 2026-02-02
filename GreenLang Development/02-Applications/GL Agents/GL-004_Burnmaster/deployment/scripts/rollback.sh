#!/bin/bash
# GL-004 BURNMASTER - Rollback Script
# ====================================
# Automated rollback for failed deployments

set -euo pipefail

# ===========================================
# Configuration
# ===========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
NAMESPACE="${GL_NAMESPACE:-greenlang}"
HELM_RELEASE_NAME="gl004-burnmaster"
KUBECTL_CONTEXT="${GL_KUBECTL_CONTEXT:-}"
DRY_RUN="${DRY_RUN:-false}"
REVISION="${ROLLBACK_REVISION:-}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}"

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

GL-004 BURNMASTER Rollback Script

Options:
    -n, --namespace     Kubernetes namespace
                        Default: greenlang
    -r, --revision      Specific revision to rollback to
                        Default: previous revision
    -c, --context       Kubernetes context
                        Default: current context
    --dry-run           Perform a dry run without applying changes
    --helm              Rollback using Helm
    --kubectl           Rollback using kubectl
    --list              List available revisions
    -h, --help          Show this help message

Examples:
    # Rollback to previous version
    $0

    # Rollback to specific revision
    $0 -r 5

    # List available revisions
    $0 --list

    # Dry run rollback
    $0 --dry-run

EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Set Kubernetes context if specified
    if [ -n "$KUBECTL_CONTEXT" ]; then
        kubectl config use-context "$KUBECTL_CONTEXT"
    fi

    # Check Kubernetes connectivity
    if ! kubectl cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if deployment exists
    if ! kubectl get deployment gl004-burnmaster -n "$NAMESPACE" >/dev/null 2>&1; then
        log_error "Deployment gl004-burnmaster not found in namespace $NAMESPACE"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

list_revisions() {
    log_info "Listing available revisions..."

    echo ""
    echo "Helm Release History:"
    echo "====================="
    helm history "$HELM_RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null || \
        log_warning "Could not get Helm history (may not be a Helm deployment)"

    echo ""
    echo "Kubernetes Deployment Revisions:"
    echo "================================="
    kubectl rollout history deployment/gl004-burnmaster -n "$NAMESPACE"

    echo ""
    echo "Recent ReplicaSets:"
    echo "==================="
    kubectl get replicasets -n "$NAMESPACE" \
        -l app.kubernetes.io/name=gl004-burnmaster \
        --sort-by=.metadata.creationTimestamp \
        -o wide
}

get_current_revision() {
    kubectl get deployment gl004-burnmaster -n "$NAMESPACE" \
        -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}'
}

get_current_image() {
    kubectl get deployment gl004-burnmaster -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}'
}

get_previous_image() {
    local current_revision=$(get_current_revision)
    local previous_revision=$((current_revision - 1))

    if [ "$previous_revision" -lt 1 ]; then
        log_error "No previous revision available"
        exit 1
    fi

    # Get ReplicaSet for previous revision
    local rs_name=$(kubectl get replicasets -n "$NAMESPACE" \
        -l app.kubernetes.io/name=gl004-burnmaster \
        -o jsonpath="{.items[?(@.metadata.annotations.deployment\.kubernetes\.io/revision==\"$previous_revision\")].metadata.name}")

    if [ -z "$rs_name" ]; then
        log_error "Could not find ReplicaSet for revision $previous_revision"
        exit 1
    fi

    kubectl get replicaset "$rs_name" -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.spec.containers[0].image}'
}

rollback_helm() {
    log_info "Rolling back with Helm..."

    local target_revision="${REVISION:-}"

    # Get current revision info before rollback
    local current_revision=$(helm history "$HELM_RELEASE_NAME" -n "$NAMESPACE" --max 1 -o json | jq -r '.[0].revision')
    local current_status=$(helm history "$HELM_RELEASE_NAME" -n "$NAMESPACE" --max 1 -o json | jq -r '.[0].status')

    echo ""
    echo "Current Release Info:"
    echo "  Revision: $current_revision"
    echo "  Status:   $current_status"
    echo ""

    if [ -n "$target_revision" ]; then
        log_info "Rolling back to revision $target_revision"
    else
        log_info "Rolling back to previous revision"
    fi

    # Build rollback command
    local helm_cmd="helm rollback $HELM_RELEASE_NAME"

    if [ -n "$target_revision" ]; then
        helm_cmd="$helm_cmd $target_revision"
    fi

    helm_cmd="$helm_cmd -n $NAMESPACE"
    helm_cmd="$helm_cmd --wait"
    helm_cmd="$helm_cmd --timeout ${WAIT_TIMEOUT}s"

    if [ "$DRY_RUN" = "true" ]; then
        helm_cmd="$helm_cmd --dry-run"
    fi

    log_info "Running: $helm_cmd"
    eval "$helm_cmd" || {
        log_error "Helm rollback failed"
        exit 1
    }

    log_success "Helm rollback complete"
}

rollback_kubectl() {
    log_info "Rolling back with kubectl..."

    local current_image=$(get_current_image)
    log_info "Current image: $current_image"

    if [ -n "$REVISION" ]; then
        log_info "Rolling back to revision $REVISION"

        if [ "$DRY_RUN" = "true" ]; then
            kubectl rollout undo deployment/gl004-burnmaster \
                -n "$NAMESPACE" \
                --to-revision="$REVISION" \
                --dry-run=client
        else
            kubectl rollout undo deployment/gl004-burnmaster \
                -n "$NAMESPACE" \
                --to-revision="$REVISION"
        fi
    else
        local previous_image=$(get_previous_image)
        log_info "Previous image: $previous_image"
        log_info "Rolling back to previous revision"

        if [ "$DRY_RUN" = "true" ]; then
            kubectl rollout undo deployment/gl004-burnmaster \
                -n "$NAMESPACE" \
                --dry-run=client
        else
            kubectl rollout undo deployment/gl004-burnmaster \
                -n "$NAMESPACE"
        fi
    fi

    log_success "kubectl rollback initiated"
}

verify_rollback() {
    if [ "$DRY_RUN" = "true" ]; then
        log_warning "DRY RUN: Skipping rollback verification"
        return 0
    fi

    log_info "Verifying rollback..."

    # Wait for rollout to complete
    kubectl rollout status deployment/gl004-burnmaster \
        -n "$NAMESPACE" \
        --timeout="${WAIT_TIMEOUT}s" || {
            log_error "Rollback verification failed - deployment did not stabilize"
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
        log_error "Expected $desired_pods ready pods, got ${ready_pods:-0}"
        exit 1
    fi

    log_success "All $ready_pods pods are ready"

    # Health check
    log_info "Running health checks..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" \
        -l app.kubernetes.io/name=gl004-burnmaster \
        -o jsonpath='{.items[0].metadata.name}')

    # Try health check multiple times
    local max_attempts=5
    local attempt=1
    local health_ok=false

    while [ $attempt -le $max_attempts ]; do
        local health_status=$(kubectl exec "$pod_name" -n "$NAMESPACE" -- \
            curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health/ready 2>/dev/null || echo "000")

        if [ "$health_status" = "200" ]; then
            health_ok=true
            break
        fi

        log_warning "Health check attempt $attempt/$max_attempts returned: $health_status"
        sleep 5
        ((attempt++))
    done

    if [ "$health_ok" = "true" ]; then
        log_success "Health check passed"
    else
        log_warning "Health check did not pass after $max_attempts attempts"
    fi

    log_success "Rollback verification complete"
}

print_summary() {
    echo ""
    echo "=========================================="
    echo "Rollback Summary"
    echo "=========================================="
    echo "Namespace:      $NAMESPACE"
    echo "Dry Run:        $DRY_RUN"
    if [ -n "$REVISION" ]; then
        echo "Target Revision: $REVISION"
    else
        echo "Target Revision: Previous"
    fi
    echo "=========================================="
    echo ""

    if [ "$DRY_RUN" != "true" ]; then
        # Show current state
        echo "Current Deployment State:"
        kubectl get deployment gl004-burnmaster -n "$NAMESPACE" -o wide

        echo ""
        echo "Current Pods:"
        kubectl get pods -n "$NAMESPACE" \
            -l app.kubernetes.io/name=gl004-burnmaster \
            -o wide

        echo ""
        echo "Current Image:"
        get_current_image

        echo ""
        echo "Deployment History (last 5):"
        kubectl rollout history deployment/gl004-burnmaster -n "$NAMESPACE" | tail -6
    fi
}

create_incident_record() {
    if [ "$DRY_RUN" = "true" ]; then
        return 0
    fi

    log_info "Creating incident record..."

    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local current_image=$(get_current_image)
    local current_revision=$(get_current_revision)
    local initiator="${USER:-unknown}"

    # Create ConfigMap with rollback info
    kubectl create configmap "gl004-rollback-$timestamp" \
        -n "$NAMESPACE" \
        --from-literal=timestamp="$timestamp" \
        --from-literal=image="$current_image" \
        --from-literal=revision="$current_revision" \
        --from-literal=target_revision="${REVISION:-previous}" \
        --from-literal=initiator="$initiator" \
        --from-literal=reason="${ROLLBACK_REASON:-manual rollback}" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Label for easy querying
    kubectl label configmap "gl004-rollback-$timestamp" \
        -n "$NAMESPACE" \
        app.kubernetes.io/name=gl004-burnmaster \
        type=rollback-record \
        --overwrite

    log_success "Incident record created: gl004-rollback-$timestamp"
}

send_notifications() {
    if [ "$DRY_RUN" = "true" ]; then
        return 0
    fi

    local current_image=$(get_current_image)
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Slack notification (if webhook configured)
    local slack_webhook="${SLACK_WEBHOOK_URL:-}"
    if [ -n "$slack_webhook" ]; then
        log_info "Sending Slack notification..."

        local payload=$(cat <<EOF
{
    "text": "GL-004 BURNMASTER Rollback Initiated",
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "GL-004 BURNMASTER Rollback"
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": "*Namespace:*\n$NAMESPACE"},
                {"type": "mrkdwn", "text": "*Image:*\n$current_image"},
                {"type": "mrkdwn", "text": "*Target:*\n${REVISION:-Previous}"},
                {"type": "mrkdwn", "text": "*Initiator:*\n${USER:-unknown}"}
            ]
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "Timestamp: $timestamp"}
            ]
        }
    ]
}
EOF
)
        curl -s -X POST -H 'Content-type: application/json' \
            --data "$payload" \
            "$slack_webhook" >/dev/null || \
            log_warning "Failed to send Slack notification"
    fi

    log_success "Notifications sent"
}

# ===========================================
# Main Script
# ===========================================
main() {
    local rollback_method="helm"
    local list_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--revision)
                REVISION="$2"
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
            --helm)
                rollback_method="helm"
                shift
                ;;
            --kubectl)
                rollback_method="kubectl"
                shift
                ;;
            --list)
                list_only=true
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

    echo ""
    echo "=========================================="
    echo "GL-004 BURNMASTER Rollback"
    echo "=========================================="
    echo ""

    check_prerequisites

    # If list only, show revisions and exit
    if [ "$list_only" = "true" ]; then
        list_revisions
        exit 0
    fi

    # Confirmation for non-dry-run
    if [ "$DRY_RUN" != "true" ]; then
        log_warning "You are about to rollback the deployment!"
        echo ""
        echo "Current deployment info:"
        echo "  Image: $(get_current_image)"
        echo "  Revision: $(get_current_revision)"
        echo ""
        read -p "Are you sure you want to proceed? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Rollback cancelled"
            exit 0
        fi

        # Optional: Ask for reason
        read -p "Reason for rollback (optional): " ROLLBACK_REASON
    fi

    # Create incident record
    create_incident_record

    # Perform rollback
    if [ "$rollback_method" = "helm" ]; then
        rollback_helm
    else
        rollback_kubectl
    fi

    # Verify rollback
    verify_rollback

    # Send notifications
    send_notifications

    # Print summary
    print_summary

    log_success "Rollback completed successfully!"
}

# Run main function
main "$@"
