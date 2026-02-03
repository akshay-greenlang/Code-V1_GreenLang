#!/bin/bash
# =============================================================================
# Disaster Recovery Failover Script for GreenLang
# INFRA-001: Backup and Disaster Recovery
# Version: 1.0.0
# =============================================================================
#
# This script orchestrates disaster recovery failover operations:
# - Validates DR region readiness
# - Promotes DR database
# - Updates DNS records
# - Redirects traffic to DR region
# - Validates failover success
#
# CRITICAL: This script should only be executed during actual DR events
#           or scheduled DR drills.
#
# Usage: ./dr-failover.sh [--mode drill|failover] [--confirm] [--skip-dns]
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/greenlang/dr-failover-$(date +%Y%m%d%H%M%S).log"
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# Region Configuration
PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"
DR_REGION="${DR_REGION:-eu-west-1}"

# RDS Configuration
PRIMARY_RDS_INSTANCE="${PRIMARY_RDS_INSTANCE:-greenlang-postgres}"
DR_RDS_INSTANCE="${DR_RDS_INSTANCE:-greenlang-postgres-dr}"
DR_INSTANCE_CLASS="${DR_INSTANCE_CLASS:-db.t3.medium}"

# Kubernetes Configuration
PRIMARY_CLUSTER="${PRIMARY_CLUSTER:-greenlang-primary}"
DR_CLUSTER="${DR_CLUSTER:-greenlang-dr}"
KUBE_NAMESPACE="${KUBE_NAMESPACE:-greenlang}"

# DNS Configuration
DNS_ZONE="${DNS_ZONE:-greenlang.io}"
API_DOMAIN="${API_DOMAIN:-api.greenlang.io}"
DNS_TTL="${DNS_TTL:-60}"

# Notification Configuration
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
PAGERDUTY_ROUTING_KEY="${PAGERDUTY_ROUTING_KEY:-}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"

# Parse arguments
MODE="drill"
CONFIRMED=false
SKIP_DNS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --confirm)
            CONFIRMED=true
            shift
            ;;
        --skip-dns)
            SKIP_DNS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--mode drill|failover] [--confirm] [--skip-dns]"
            echo ""
            echo "Modes:"
            echo "  drill     - DR drill mode (creates isolated test environment)"
            echo "  failover  - Production failover (updates DNS, redirects traffic)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }
log_critical() { log "CRITICAL" "$@"; }

# =============================================================================
# Notification Functions
# =============================================================================

send_notification() {
    local title=$1
    local message=$2
    local severity=${3:-info}

    # Slack
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local color="good"
        case $severity in
            critical) color="danger" ;;
            warning) color="warning" ;;
            *) color="good" ;;
        esac

        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"$title\",
                    \"text\": \"$message\",
                    \"footer\": \"DR Failover | $(date)\",
                    \"mrkdwn_in\": [\"text\"]
                }]
            }" "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi

    # PagerDuty
    if [ -n "$PAGERDUTY_ROUTING_KEY" ] && [ "$severity" = "critical" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"routing_key\": \"$PAGERDUTY_ROUTING_KEY\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"$title: $message\",
                    \"severity\": \"$severity\",
                    \"source\": \"dr-failover.sh\"
                }
            }" \
            "https://events.pagerduty.com/v2/enqueue" > /dev/null 2>&1 || true
    fi

    # SNS
    if [ -n "$SNS_TOPIC_ARN" ]; then
        aws sns publish \
            --topic-arn "$SNS_TOPIC_ARN" \
            --subject "$title" \
            --message "$message" > /dev/null 2>&1 || true
    fi
}

# =============================================================================
# Pre-Failover Checks
# =============================================================================

preflight_checks() {
    log_info "=========================================="
    log_info "PRE-FAILOVER CHECKS"
    log_info "=========================================="

    local checks_passed=true

    # Check 1: Verify AWS credentials
    log_info "[Check 1/7] Verifying AWS credentials..."
    if aws sts get-caller-identity &>/dev/null; then
        log_success "AWS credentials valid"
    else
        log_error "AWS credentials invalid or expired"
        checks_passed=false
    fi

    # Check 2: Verify DR region access
    log_info "[Check 2/7] Verifying DR region access..."
    if aws ec2 describe-availability-zones --region "$DR_REGION" &>/dev/null; then
        log_success "DR region ($DR_REGION) accessible"
    else
        log_error "Cannot access DR region: $DR_REGION"
        checks_passed=false
    fi

    # Check 3: Check for replicated backups in DR region
    log_info "[Check 3/7] Checking replicated backups in DR region..."
    local dr_backup=$(aws rds describe-db-instance-automated-backups \
        --region "$DR_REGION" \
        --db-instance-identifier "$PRIMARY_RDS_INSTANCE" \
        --query 'DBInstanceAutomatedBackups[0].DBInstanceAutomatedBackupsArn' \
        --output text 2>/dev/null || echo "None")

    if [ "$dr_backup" != "None" ] && [ -n "$dr_backup" ]; then
        log_success "Replicated backup found in DR region"

        # Get latest restorable time
        local latest_restorable=$(aws rds describe-db-instance-automated-backups \
            --region "$DR_REGION" \
            --db-instance-identifier "$PRIMARY_RDS_INSTANCE" \
            --query 'DBInstanceAutomatedBackups[0].RestoreWindow.LatestTime' \
            --output text 2>/dev/null)

        log_info "Latest restorable time: $latest_restorable"
    else
        log_error "No replicated backup found in DR region"
        log_error "Cross-region backup replication may not be configured"
        checks_passed=false
    fi

    # Check 4: Verify DR Kubernetes cluster
    log_info "[Check 4/7] Verifying DR Kubernetes cluster..."
    if kubectl config get-contexts "$DR_CLUSTER" &>/dev/null; then
        log_success "DR Kubernetes cluster configured"
    else
        log_warn "DR Kubernetes cluster not configured in kubeconfig"
    fi

    # Check 5: Check primary region status
    log_info "[Check 5/7] Checking primary region status..."
    local primary_status=$(aws rds describe-db-instances \
        --region "$PRIMARY_REGION" \
        --db-instance-identifier "$PRIMARY_RDS_INSTANCE" \
        --query 'DBInstances[0].DBInstanceStatus' \
        --output text 2>/dev/null || echo "unavailable")

    log_info "Primary database status: $primary_status"
    if [ "$primary_status" = "available" ] && [ "$MODE" = "failover" ]; then
        log_warn "Primary database is still available. Confirm this is a real DR event."
    fi

    # Check 6: Verify DNS access (Route53)
    log_info "[Check 6/7] Verifying DNS access..."
    if [ "$SKIP_DNS" = false ]; then
        if aws route53 list-hosted-zones &>/dev/null; then
            log_success "Route53 access verified"
        else
            log_warn "Cannot access Route53. DNS updates may fail."
        fi
    else
        log_info "DNS updates will be skipped (--skip-dns flag)"
    fi

    # Check 7: Verify required tools
    log_info "[Check 7/7] Verifying required tools..."
    local tools=("aws" "kubectl" "jq" "psql")
    for tool in "${tools[@]}"; do
        if command -v "$tool" &>/dev/null; then
            log_success "$tool: available"
        else
            log_warn "$tool: not found (may be required)"
        fi
    done

    if [ "$checks_passed" = false ]; then
        log_error "Pre-flight checks failed. Cannot proceed with failover."
        return 1
    fi

    log_success "All pre-flight checks passed"
    return 0
}

# =============================================================================
# Database Failover
# =============================================================================

failover_database() {
    log_info "=========================================="
    log_info "DATABASE FAILOVER"
    log_info "=========================================="

    local start_time=$(date +%s)

    # Step 1: Get replicated backup ARN
    log_info "[Step 1/5] Getting replicated backup ARN..."
    local backup_arn=$(aws rds describe-db-instance-automated-backups \
        --region "$DR_REGION" \
        --db-instance-identifier "$PRIMARY_RDS_INSTANCE" \
        --query 'DBInstanceAutomatedBackups[0].DBInstanceAutomatedBackupsArn' \
        --output text)

    if [ -z "$backup_arn" ] || [ "$backup_arn" = "None" ]; then
        log_error "No replicated backup found!"
        return 1
    fi

    log_info "Backup ARN: $backup_arn"

    # Step 2: Check if DR instance already exists
    log_info "[Step 2/5] Checking for existing DR instance..."
    local existing=$(aws rds describe-db-instances \
        --region "$DR_REGION" \
        --db-instance-identifier "$DR_RDS_INSTANCE" \
        --query 'DBInstances[0].DBInstanceStatus' \
        --output text 2>/dev/null || echo "not-found")

    if [ "$existing" = "available" ]; then
        log_info "DR instance already exists and is available"
    elif [ "$existing" != "not-found" ]; then
        log_warn "DR instance exists with status: $existing"
        log_info "Waiting for existing instance..."
    else
        # Step 3: Create DR instance from backup
        log_info "[Step 3/5] Creating DR database instance..."
        send_notification "DR Failover Started" "Creating DR database in $DR_REGION" "warning"

        aws rds restore-db-instance-to-point-in-time \
            --region "$DR_REGION" \
            --source-db-instance-automated-backups-arn "$backup_arn" \
            --target-db-instance-identifier "$DR_RDS_INSTANCE" \
            --use-latest-restorable-time \
            --db-instance-class "$DR_INSTANCE_CLASS" \
            --multi-az \
            --storage-type gp3 \
            --deletion-protection \
            --enable-cloudwatch-logs-exports '["postgresql","upgrade"]' \
            --tags Key=Environment,Value=dr Key=FailoverTimestamp,Value="$TIMESTAMP"
    fi

    # Step 4: Wait for DR instance to be available
    log_info "[Step 4/5] Waiting for DR instance to be available..."
    local timeout=2400  # 40 minutes
    local wait_start=$(date +%s)

    while true; do
        local status=$(aws rds describe-db-instances \
            --region "$DR_REGION" \
            --db-instance-identifier "$DR_RDS_INSTANCE" \
            --query 'DBInstances[0].DBInstanceStatus' \
            --output text 2>/dev/null || echo "creating")

        local elapsed=$(($(date +%s) - wait_start))

        if [ "$status" = "available" ]; then
            log_success "DR instance is available (${elapsed}s)"
            break
        elif [ "$elapsed" -gt "$timeout" ]; then
            log_error "Timeout waiting for DR instance"
            return 1
        fi

        log_info "Status: $status (${elapsed}s elapsed)"
        sleep 60
    done

    # Step 5: Get DR endpoint
    log_info "[Step 5/5] Getting DR instance endpoint..."
    local dr_endpoint=$(aws rds describe-db-instances \
        --region "$DR_REGION" \
        --db-instance-identifier "$DR_RDS_INSTANCE" \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)

    log_success "DR Database Endpoint: $dr_endpoint"

    local total_duration=$(($(date +%s) - start_time))
    log_info "Database failover completed in ${total_duration}s"

    echo "$dr_endpoint"
}

# =============================================================================
# DNS Failover
# =============================================================================

failover_dns() {
    local dr_endpoint=$1

    if [ "$SKIP_DNS" = true ]; then
        log_info "DNS failover skipped (--skip-dns flag)"
        return 0
    fi

    log_info "=========================================="
    log_info "DNS FAILOVER"
    log_info "=========================================="

    # Step 1: Get hosted zone ID
    log_info "[Step 1/3] Getting hosted zone ID..."
    local zone_id=$(aws route53 list-hosted-zones-by-name \
        --dns-name "$DNS_ZONE" \
        --query 'HostedZones[0].Id' \
        --output text | sed 's|/hostedzone/||')

    if [ -z "$zone_id" ]; then
        log_error "Hosted zone not found for: $DNS_ZONE"
        return 1
    fi

    log_info "Hosted Zone ID: $zone_id"

    # Step 2: Update DNS record
    log_info "[Step 2/3] Updating DNS record..."
    local change_batch=$(cat <<EOF
{
    "Changes": [
        {
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "${API_DOMAIN}",
                "Type": "CNAME",
                "TTL": ${DNS_TTL},
                "ResourceRecords": [
                    {
                        "Value": "${dr_endpoint}"
                    }
                ]
            }
        }
    ],
    "Comment": "DR Failover at ${TIMESTAMP}"
}
EOF
)

    local change_id=$(aws route53 change-resource-record-sets \
        --hosted-zone-id "$zone_id" \
        --change-batch "$change_batch" \
        --query 'ChangeInfo.Id' \
        --output text)

    log_info "DNS change submitted: $change_id"

    # Step 3: Wait for DNS propagation
    log_info "[Step 3/3] Waiting for DNS propagation..."
    aws route53 wait resource-record-sets-changed --id "$change_id"

    log_success "DNS failover completed"
    log_info "New endpoint: $API_DOMAIN -> $dr_endpoint"
}

# =============================================================================
# Kubernetes Failover
# =============================================================================

failover_kubernetes() {
    local dr_endpoint=$1

    log_info "=========================================="
    log_info "KUBERNETES CONFIGURATION UPDATE"
    log_info "=========================================="

    # Switch to DR cluster
    log_info "[Step 1/3] Switching to DR cluster..."
    kubectl config use-context "$DR_CLUSTER" 2>/dev/null || {
        log_warn "Could not switch to DR cluster context"
        return 0
    }

    # Step 2: Update database secrets
    log_info "[Step 2/3] Updating database secrets..."
    kubectl create secret generic greenlang-db-credentials \
        --namespace "$KUBE_NAMESPACE" \
        --from-literal=host="$dr_endpoint" \
        --from-literal=username="${DB_USER:-greenlang_admin}" \
        --from-literal=database="${DB_NAME:-greenlang}" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Step 3: Restart deployments
    log_info "[Step 3/3] Restarting deployments..."
    kubectl rollout restart deployment -n "$KUBE_NAMESPACE" 2>/dev/null || true

    # Wait for rollout
    for deployment in $(kubectl get deployments -n "$KUBE_NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
        log_info "Waiting for deployment: $deployment"
        kubectl rollout status deployment/"$deployment" -n "$KUBE_NAMESPACE" --timeout=300s 2>/dev/null || true
    done

    log_success "Kubernetes configuration updated"
}

# =============================================================================
# Post-Failover Validation
# =============================================================================

validate_failover() {
    local dr_endpoint=$1

    log_info "=========================================="
    log_info "POST-FAILOVER VALIDATION"
    log_info "=========================================="

    local validation_passed=true

    # Test 1: Database connectivity
    log_info "[Test 1/4] Testing database connectivity..."
    if PGPASSWORD="${DB_PASSWORD:-}" psql -h "$dr_endpoint" -U "${DB_USER:-greenlang_admin}" \
        -d "${DB_NAME:-greenlang}" -c "SELECT 1;" &>/dev/null; then
        log_success "Database connectivity: OK"
    else
        log_error "Database connectivity: FAILED"
        validation_passed=false
    fi

    # Test 2: Database tables
    log_info "[Test 2/4] Validating database tables..."
    local table_count=$(PGPASSWORD="${DB_PASSWORD:-}" psql -h "$dr_endpoint" \
        -U "${DB_USER:-greenlang_admin}" -d "${DB_NAME:-greenlang}" -t \
        -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ')

    if [ "${table_count:-0}" -gt 0 ]; then
        log_success "Database tables: $table_count tables found"
    else
        log_error "Database tables: No tables found"
        validation_passed=false
    fi

    # Test 3: DNS resolution
    if [ "$SKIP_DNS" = false ]; then
        log_info "[Test 3/4] Validating DNS resolution..."
        local resolved=$(dig +short "$API_DOMAIN" | head -1)
        if [ -n "$resolved" ]; then
            log_success "DNS resolution: $API_DOMAIN -> $resolved"
        else
            log_warn "DNS resolution: Could not resolve $API_DOMAIN"
        fi
    else
        log_info "[Test 3/4] DNS validation skipped"
    fi

    # Test 4: Application health (if available)
    log_info "[Test 4/4] Checking application health..."
    if kubectl get pods -n "$KUBE_NAMESPACE" &>/dev/null; then
        local ready_pods=$(kubectl get pods -n "$KUBE_NAMESPACE" \
            -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null \
            | tr ' ' '\n' | grep -c "True" || echo "0")

        local total_pods=$(kubectl get pods -n "$KUBE_NAMESPACE" --no-headers 2>/dev/null | wc -l)

        if [ "$ready_pods" -gt 0 ]; then
            log_success "Application pods: $ready_pods/$total_pods ready"
        else
            log_warn "Application pods: No ready pods"
        fi
    else
        log_info "Kubernetes validation skipped (no cluster access)"
    fi

    if [ "$validation_passed" = true ]; then
        log_success "All validations passed"
        return 0
    else
        log_error "Some validations failed"
        return 1
    fi
}

# =============================================================================
# Rollback
# =============================================================================

rollback() {
    log_warn "=========================================="
    log_warn "ROLLBACK INITIATED"
    log_warn "=========================================="

    send_notification "DR Failover Rollback" "Rolling back DR failover due to validation failures" "warning"

    # Rollback DNS if changed
    if [ "$SKIP_DNS" = false ]; then
        log_info "Rolling back DNS changes..."
        # TODO: Implement DNS rollback if needed
    fi

    # Delete DR RDS instance if it was just created
    if [ "$MODE" = "drill" ]; then
        log_info "Cleaning up DR drill resources..."
        aws rds delete-db-instance \
            --region "$DR_REGION" \
            --db-instance-identifier "$DR_RDS_INSTANCE" \
            --skip-final-snapshot \
            --delete-automated-backups 2>/dev/null || true
    fi

    log_info "Rollback completed"
}

# =============================================================================
# Main
# =============================================================================

main() {
    mkdir -p "$(dirname "$LOG_FILE")"

    log_info "=========================================="
    log_critical "DISASTER RECOVERY FAILOVER"
    log_info "=========================================="
    log_info "Mode: $MODE"
    log_info "Primary Region: $PRIMARY_REGION"
    log_info "DR Region: $DR_REGION"
    log_info "Timestamp: $TIMESTAMP"
    log_info "=========================================="

    # Confirmation check
    if [ "$MODE" = "failover" ] && [ "$CONFIRMED" = false ]; then
        echo ""
        echo "!!! WARNING !!!"
        echo "You are about to execute a PRODUCTION DISASTER RECOVERY FAILOVER."
        echo "This will:"
        echo "  - Create/promote DR database in $DR_REGION"
        if [ "$SKIP_DNS" = false ]; then
            echo "  - Update DNS to point to DR region"
        fi
        echo "  - Update Kubernetes configuration"
        echo ""
        echo "This action CANNOT be easily undone."
        echo ""
        read -p "Type 'FAILOVER' to confirm: " confirm
        if [ "$confirm" != "FAILOVER" ]; then
            log_info "Failover cancelled by user"
            exit 0
        fi
    fi

    # Send start notification
    send_notification "DR Failover Initiated" "Mode: $MODE\nPrimary: $PRIMARY_REGION\nDR: $DR_REGION" "warning"

    # Run pre-flight checks
    if ! preflight_checks; then
        send_notification "DR Failover Failed" "Pre-flight checks failed" "critical"
        exit 1
    fi

    # Execute database failover
    local dr_endpoint
    dr_endpoint=$(failover_database)

    if [ -z "$dr_endpoint" ]; then
        log_error "Database failover failed"
        send_notification "DR Failover Failed" "Database failover failed" "critical"
        rollback
        exit 1
    fi

    # Execute DNS failover
    if ! failover_dns "$dr_endpoint"; then
        log_error "DNS failover failed"
        send_notification "DR Failover Warning" "DNS failover failed, manual intervention required" "warning"
    fi

    # Update Kubernetes
    failover_kubernetes "$dr_endpoint"

    # Validate failover
    if validate_failover "$dr_endpoint"; then
        log_success "=========================================="
        log_success "DISASTER RECOVERY FAILOVER COMPLETE"
        log_success "=========================================="
        log_success "DR Database: $dr_endpoint"
        log_success "Mode: $MODE"
        log_success "=========================================="

        send_notification "DR Failover Complete" "Successfully failed over to $DR_REGION\nEndpoint: $dr_endpoint" "info"
    else
        log_warn "Failover completed with validation warnings"
        send_notification "DR Failover Complete (with warnings)" "Failover completed but some validations failed" "warning"
    fi

    # Print next steps
    echo ""
    echo "=========================================="
    echo "NEXT STEPS"
    echo "=========================================="
    echo "1. Verify application functionality"
    echo "2. Check monitoring dashboards"
    echo "3. Update status page"
    echo "4. Notify stakeholders"
    echo "5. Document incident timeline"
    if [ "$MODE" = "drill" ]; then
        echo ""
        echo "DR DRILL CLEANUP:"
        echo "- Delete DR instance when drill is complete"
        echo "- Revert any DNS changes"
        echo "- Document drill results"
    fi
    echo "=========================================="
}

main "$@"
