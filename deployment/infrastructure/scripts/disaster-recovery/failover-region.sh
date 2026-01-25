#!/bin/bash
# =============================================================================
# GreenLang Regional Failover Script
# Production-grade multi-region disaster recovery failover
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/greenlang/failover.log"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Region configuration
PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"
SECONDARY_REGION="${SECONDARY_REGION:-us-west-2}"
CURRENT_ACTIVE_REGION=""

# Cluster configuration
PRIMARY_CLUSTER="${PRIMARY_CLUSTER:-greenlang-prod-east}"
SECONDARY_CLUSTER="${SECONDARY_CLUSTER:-greenlang-prod-west}"

# Database configuration
PRIMARY_RDS="${PRIMARY_RDS:-greenlang-prod-east}"
SECONDARY_RDS="${SECONDARY_RDS:-greenlang-prod-west}"

# DNS configuration
HOSTED_ZONE_ID="${HOSTED_ZONE_ID:-Z1234567890ABC}"
DNS_RECORD="${DNS_RECORD:-api.greenlang.io}"
DNS_TTL="${DNS_TTL:-60}"

# Notification
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
PAGERDUTY_ROUTING_KEY="${PAGERDUTY_ROUTING_KEY:-}"

# Failover options
FAILOVER_MODE="${FAILOVER_MODE:-manual}"  # manual, automatic
DRY_RUN="${DRY_RUN:-false}"

# Timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Logging
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$*"; }
log_warn() { log "WARN" "${YELLOW}$*${NC}"; }
log_error() { log "ERROR" "${RED}$*${NC}"; }
log_success() { log "SUCCESS" "${GREEN}$*${NC}"; }
log_step() { log "STEP" "${CYAN}>>> $*${NC}"; }
log_critical() { log "CRITICAL" "${MAGENTA}!!! $*${NC}"; }

# Cleanup
cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        send_alert "FAILURE" "Regional failover failed with exit code ${exit_code}"
    fi
    exit ${exit_code}
}

trap cleanup EXIT

# Send alert
send_alert() {
    local status=$1
    local message=$2
    local severity="${3:-high}"

    # Slack
    if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
        local color="danger"
        [[ "${status}" == "SUCCESS" ]] && color="good"
        [[ "${status}" == "WARNING" ]] && color="warning"

        curl -s -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"Regional Failover ${status}\",
                    \"text\": \"${message}\",
                    \"fields\": [
                        {\"title\": \"From Region\", \"value\": \"${PRIMARY_REGION}\", \"short\": true},
                        {\"title\": \"To Region\", \"value\": \"${SECONDARY_REGION}\", \"short\": true},
                        {\"title\": \"Timestamp\", \"value\": \"${TIMESTAMP}\", \"short\": true},
                        {\"title\": \"Mode\", \"value\": \"${FAILOVER_MODE}\", \"short\": true}
                    ]
                }]
            }" || true
    fi

    # PagerDuty
    if [[ -n "${PAGERDUTY_ROUTING_KEY}" ]]; then
        local pd_severity="critical"
        [[ "${severity}" == "low" ]] && pd_severity="warning"

        curl -s -X POST "https://events.pagerduty.com/v2/enqueue" \
            -H 'Content-type: application/json' \
            -d "{
                \"routing_key\": \"${PAGERDUTY_ROUTING_KEY}\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"Regional Failover: ${message}\",
                    \"severity\": \"${pd_severity}\",
                    \"source\": \"greenlang-failover\",
                    \"custom_details\": {
                        \"from_region\": \"${PRIMARY_REGION}\",
                        \"to_region\": \"${SECONDARY_REGION}\",
                        \"timestamp\": \"${TIMESTAMP}\"
                    }
                }
            }" || true
    fi
}

# Get current active region
get_active_region() {
    log_step "Determining current active region..."

    # Check Route53 DNS record
    local current_target=$(aws route53 list-resource-record-sets \
        --hosted-zone-id "${HOSTED_ZONE_ID}" \
        --query "ResourceRecordSets[?Name=='${DNS_RECORD}.'].ResourceRecords[0].Value" \
        --output text 2>/dev/null || echo "")

    if [[ "${current_target}" == *"east"* ]]; then
        CURRENT_ACTIVE_REGION="${PRIMARY_REGION}"
    elif [[ "${current_target}" == *"west"* ]]; then
        CURRENT_ACTIVE_REGION="${SECONDARY_REGION}"
    else
        # Fallback: check which cluster has active deployments
        if kubectl --context "${PRIMARY_CLUSTER}" get deployment greenlang-api-blue \
            -n greenlang-production &>/dev/null; then
            local primary_replicas=$(kubectl --context "${PRIMARY_CLUSTER}" \
                get deployment greenlang-api-blue -n greenlang-production \
                -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

            if [[ "${primary_replicas}" -gt 0 ]]; then
                CURRENT_ACTIVE_REGION="${PRIMARY_REGION}"
            else
                CURRENT_ACTIVE_REGION="${SECONDARY_REGION}"
            fi
        fi
    fi

    log_info "Current active region: ${CURRENT_ACTIVE_REGION}"
}

# Check regional health
check_region_health() {
    local region=$1
    local cluster=$2

    log_step "Checking health of region: ${region}"

    local health_score=0
    local max_score=100

    # Check Kubernetes cluster
    if kubectl --context "${cluster}" cluster-info &>/dev/null; then
        health_score=$((health_score + 25))
        log_info "  [OK] Kubernetes cluster accessible"
    else
        log_warn "  [FAIL] Kubernetes cluster not accessible"
    fi

    # Check API deployment
    local api_ready=$(kubectl --context "${cluster}" \
        get deployment greenlang-api-blue -n greenlang-production \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

    if [[ "${api_ready}" -gt 0 ]]; then
        health_score=$((health_score + 25))
        log_info "  [OK] API deployment healthy (${api_ready} replicas)"
    else
        log_warn "  [FAIL] API deployment not healthy"
    fi

    # Check database (RDS)
    local rds_instance="${PRIMARY_RDS}"
    [[ "${region}" == "${SECONDARY_REGION}" ]] && rds_instance="${SECONDARY_RDS}"

    local rds_status=$(AWS_DEFAULT_REGION="${region}" aws rds describe-db-instances \
        --db-instance-identifier "${rds_instance}" \
        --query 'DBInstances[0].DBInstanceStatus' \
        --output text 2>/dev/null || echo "unknown")

    if [[ "${rds_status}" == "available" ]]; then
        health_score=$((health_score + 25))
        log_info "  [OK] Database available"
    else
        log_warn "  [FAIL] Database status: ${rds_status}"
    fi

    # Check load balancer
    local lb_healthy=$(AWS_DEFAULT_REGION="${region}" aws elbv2 describe-target-health \
        --target-group-arn "arn:aws:elasticloadbalancing:${region}:123456789012:targetgroup/greenlang-api/xxxxx" \
        --query 'TargetHealthDescriptions[?TargetHealth.State==`healthy`] | length(@)' \
        --output text 2>/dev/null || echo "0")

    if [[ "${lb_healthy}" -gt 0 ]]; then
        health_score=$((health_score + 25))
        log_info "  [OK] Load balancer healthy (${lb_healthy} targets)"
    else
        log_warn "  [FAIL] No healthy load balancer targets"
    fi

    log_info "Region ${region} health score: ${health_score}/${max_score}"
    echo "${health_score}"
}

# Pre-failover checks
pre_failover_checks() {
    log_step "Running pre-failover checks..."

    local target_region="${1}"
    local target_cluster="${2}"

    # Check target region health
    local target_health=$(check_region_health "${target_region}" "${target_cluster}")

    if [[ "${target_health}" -lt 50 ]]; then
        log_error "Target region health is too low (${target_health}/100)"
        log_error "Cannot proceed with failover"
        exit 1
    fi

    # Check database replication lag
    if [[ "${target_region}" == "${SECONDARY_REGION}" ]]; then
        log_info "Checking database replication lag..."

        local replica_lag=$(AWS_DEFAULT_REGION="${target_region}" aws cloudwatch get-metric-statistics \
            --namespace AWS/RDS \
            --metric-name ReplicaLag \
            --dimensions Name=DBInstanceIdentifier,Value="${SECONDARY_RDS}" \
            --start-time "$(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%SZ)" \
            --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            --period 60 \
            --statistics Average \
            --query 'Datapoints[0].Average' \
            --output text 2>/dev/null || echo "0")

        if [[ "${replica_lag}" != "None" && "${replica_lag}" != "0" ]]; then
            local lag_seconds=$(echo "${replica_lag}" | awk '{printf "%.0f", $1}')
            log_warn "Database replication lag: ${lag_seconds} seconds"

            if [[ "${lag_seconds}" -gt 60 ]]; then
                log_error "Replication lag too high. Data loss possible."
                if [[ "${FORCE_FAILOVER:-false}" != "true" ]]; then
                    read -p "Continue anyway? (yes/no): " confirm
                    [[ "${confirm}" != "yes" ]] && exit 1
                fi
            fi
        fi
    fi

    log_success "Pre-failover checks passed"
}

# Scale up target region
scale_up_target() {
    local target_cluster=$1

    log_step "Scaling up target region..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY-RUN] Would scale up deployments in ${target_cluster}"
        return 0
    fi

    # Scale up API
    kubectl --context "${target_cluster}" scale deployment greenlang-api-blue \
        -n greenlang-production --replicas=5

    # Scale up workers
    kubectl --context "${target_cluster}" scale deployment greenlang-worker \
        -n greenlang-production --replicas=3

    # Scale up frontend
    kubectl --context "${target_cluster}" scale deployment greenlang-frontend \
        -n greenlang-production --replicas=3

    # Wait for rollout
    log_info "Waiting for deployments to scale up..."
    kubectl --context "${target_cluster}" rollout status deployment/greenlang-api-blue \
        -n greenlang-production --timeout=300s

    log_success "Target region scaled up"
}

# Promote read replica to primary
promote_database() {
    local target_region=$1
    local target_rds=$2

    log_step "Promoting database read replica to primary..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY-RUN] Would promote ${target_rds} to primary"
        return 0
    fi

    # Promote read replica
    AWS_DEFAULT_REGION="${target_region}" aws rds promote-read-replica \
        --db-instance-identifier "${target_rds}"

    log_info "Waiting for database promotion..."
    AWS_DEFAULT_REGION="${target_region}" aws rds wait db-instance-available \
        --db-instance-identifier "${target_rds}"

    log_success "Database promoted to primary"
}

# Update DNS records
update_dns() {
    local target_region=$1
    local target_lb=$2

    log_step "Updating DNS to point to ${target_region}..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY-RUN] Would update DNS record ${DNS_RECORD} to ${target_lb}"
        return 0
    fi

    # Get target load balancer hostname
    local lb_dns=$(AWS_DEFAULT_REGION="${target_region}" aws elbv2 describe-load-balancers \
        --names "greenlang-api-${target_region}" \
        --query 'LoadBalancers[0].DNSName' \
        --output text 2>/dev/null || echo "")

    if [[ -z "${lb_dns}" ]]; then
        log_error "Could not find load balancer in ${target_region}"
        exit 1
    fi

    # Update Route53 record
    local change_batch=$(cat <<EOF
{
    "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
            "Name": "${DNS_RECORD}",
            "Type": "CNAME",
            "TTL": ${DNS_TTL},
            "ResourceRecords": [{
                "Value": "${lb_dns}"
            }]
        }
    }]
}
EOF
)

    aws route53 change-resource-record-sets \
        --hosted-zone-id "${HOSTED_ZONE_ID}" \
        --change-batch "${change_batch}"

    log_info "Waiting for DNS propagation..."
    sleep 30

    # Verify DNS
    local resolved=$(dig +short "${DNS_RECORD}" | head -1)
    log_info "DNS now resolves to: ${resolved}"

    log_success "DNS updated successfully"
}

# Scale down source region
scale_down_source() {
    local source_cluster=$1

    log_step "Scaling down source region..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY-RUN] Would scale down deployments in ${source_cluster}"
        return 0
    fi

    # Scale down to 0 replicas
    kubectl --context "${source_cluster}" scale deployment greenlang-api-blue \
        -n greenlang-production --replicas=0

    kubectl --context "${source_cluster}" scale deployment greenlang-worker \
        -n greenlang-production --replicas=0

    kubectl --context "${source_cluster}" scale deployment greenlang-frontend \
        -n greenlang-production --replicas=0

    log_success "Source region scaled down"
}

# Verify failover
verify_failover() {
    local target_region=$1

    log_step "Verifying failover..."

    # Health check via DNS
    local health_check=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://${DNS_RECORD}/api/v1/health" 2>/dev/null || echo "000")

    if [[ "${health_check}" == "200" ]]; then
        log_success "API health check passed"
    else
        log_error "API health check failed (HTTP ${health_check})"
        return 1
    fi

    # Verify region header
    local region_header=$(curl -s -I "https://${DNS_RECORD}/api/v1/health" 2>/dev/null | \
        grep -i "x-served-by" | awk '{print $2}' || echo "")

    if [[ "${region_header}" == *"${target_region}"* ]]; then
        log_success "Traffic is now served from ${target_region}"
    else
        log_warn "Could not verify serving region"
    fi

    # Run smoke tests
    log_info "Running smoke tests..."
    "${SCRIPT_DIR}/verify-recovery.sh" --quick || {
        log_error "Smoke tests failed"
        return 1
    }

    log_success "Failover verification completed"
}

# Record failover event
record_failover_event() {
    local from_region=$1
    local to_region=$2
    local status=$3

    log_step "Recording failover event..."

    local event_file="/var/log/greenlang/failover-events.json"

    local event=$(cat <<EOF
{
    "timestamp": "${TIMESTAMP}",
    "from_region": "${from_region}",
    "to_region": "${to_region}",
    "status": "${status}",
    "mode": "${FAILOVER_MODE}",
    "initiated_by": "${USER:-system}"
}
EOF
)

    echo "${event}" >> "${event_file}"

    # Upload to S3 for audit
    aws s3 cp "${event_file}" \
        "s3://greenlang-audit-logs/failover-events/${TIMESTAMP}.json" \
        --only-show-errors 2>/dev/null || true

    log_info "Failover event recorded"
}

# Perform failover
perform_failover() {
    local from_region="${PRIMARY_REGION}"
    local to_region="${SECONDARY_REGION}"
    local from_cluster="${PRIMARY_CLUSTER}"
    local to_cluster="${SECONDARY_CLUSTER}"

    # Determine direction based on current active region
    get_active_region

    if [[ "${CURRENT_ACTIVE_REGION}" == "${SECONDARY_REGION}" ]]; then
        # Failback to primary
        from_region="${SECONDARY_REGION}"
        to_region="${PRIMARY_REGION}"
        from_cluster="${SECONDARY_CLUSTER}"
        to_cluster="${PRIMARY_CLUSTER}"
        log_info "Performing FAILBACK to primary region"
    else
        log_info "Performing FAILOVER to secondary region"
    fi

    log_critical "=========================================="
    log_critical "REGIONAL FAILOVER INITIATED"
    log_critical "=========================================="
    log_critical "From: ${from_region} (${from_cluster})"
    log_critical "To:   ${to_region} (${to_cluster})"
    log_critical "Mode: ${FAILOVER_MODE}"
    log_critical "=========================================="

    if [[ "${FAILOVER_MODE}" == "manual" && "${DRY_RUN}" != "true" ]]; then
        read -p "Proceed with failover? (yes/no): " confirm
        if [[ "${confirm}" != "yes" ]]; then
            log_info "Failover cancelled by user"
            exit 0
        fi
    fi

    send_alert "STARTED" "Regional failover initiated from ${from_region} to ${to_region}"

    # Execute failover steps
    pre_failover_checks "${to_region}" "${to_cluster}"
    scale_up_target "${to_cluster}"
    promote_database "${to_region}" "${SECONDARY_RDS}"
    update_dns "${to_region}" ""
    scale_down_source "${from_cluster}"
    verify_failover "${to_region}"

    record_failover_event "${from_region}" "${to_region}" "SUCCESS"

    log_critical "=========================================="
    log_success "FAILOVER COMPLETED SUCCESSFULLY"
    log_critical "=========================================="
    log_info "Active region is now: ${to_region}"

    send_alert "SUCCESS" "Regional failover completed. Active region: ${to_region}"
}

# Main
main() {
    mkdir -p "$(dirname "${LOG_FILE}")"

    case "${COMMAND:-failover}" in
        status)
            get_active_region
            echo ""
            echo "=== Regional Status ==="
            echo "Active Region: ${CURRENT_ACTIVE_REGION}"
            echo ""
            echo "Primary Region (${PRIMARY_REGION}):"
            check_region_health "${PRIMARY_REGION}" "${PRIMARY_CLUSTER}"
            echo ""
            echo "Secondary Region (${SECONDARY_REGION}):"
            check_region_health "${SECONDARY_REGION}" "${SECONDARY_CLUSTER}"
            ;;

        failover)
            perform_failover
            ;;

        failback)
            # Force failback to primary
            CURRENT_ACTIVE_REGION="${SECONDARY_REGION}"
            perform_failover
            ;;

        test)
            log_info "Running failover test (dry-run)..."
            DRY_RUN="true"
            perform_failover
            ;;

        *)
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  status    Show current regional status"
            echo "  failover  Perform regional failover"
            echo "  failback  Failback to primary region"
            echo "  test      Test failover (dry-run)"
            echo ""
            echo "Options:"
            echo "  --mode              Failover mode: manual, automatic"
            echo "  --dry-run           Perform dry-run without changes"
            echo "  --force             Force failover even with warnings"
            echo "  --primary-region    Primary region (default: us-east-1)"
            echo "  --secondary-region  Secondary region (default: us-west-2)"
            exit 0
            ;;
    esac
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        status|failover|failback|test)
            COMMAND="$1"
            shift
            ;;
        --mode)
            FAILOVER_MODE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --force)
            FORCE_FAILOVER="true"
            shift
            ;;
        --primary-region)
            PRIMARY_REGION="$2"
            shift 2
            ;;
        --secondary-region)
            SECONDARY_REGION="$2"
            shift 2
            ;;
        --help|-h)
            COMMAND="help"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
