#!/bin/bash
# ==============================================================================
# GL-001 ThermalCommand - Manual Failover Script
# ==============================================================================
# This script performs manual failover operations for GL-001:
# - Regional failover (primary to secondary region)
# - Database failover (PostgreSQL primary to replica)
# - Redis failover (master to replica via Sentinel)
# - Kafka broker failover
#
# Usage: ./failover.sh <operation> [options]
#
# Operations:
#   region           - Failover to secondary region
#   database         - Failover PostgreSQL to replica
#   redis            - Failover Redis via Sentinel
#   kafka            - Rebalance Kafka partitions
#   verify           - Verify current failover status
#   rollback         - Rollback to primary (after recovery)
#
# ==============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/gl-001-failover.log"

# Default values
PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"
SECONDARY_REGION="${SECONDARY_REGION:-us-west-2}"
PRIMARY_CONTEXT="${PRIMARY_CONTEXT:-eks-us-east-1-prod}"
SECONDARY_CONTEXT="${SECONDARY_CONTEXT:-eks-us-west-2-dr}"
NAMESPACE="${NAMESPACE:-greenlang}"

# Operation
OPERATION="${1:-help}"
shift || true

# ==============================================================================
# Logging Functions
# ==============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# ==============================================================================
# Notification Functions
# ==============================================================================

send_notification() {
    local status="$1"
    local message="$2"
    local color="${3:-#ffcc00}"

    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"GL-001 Failover ${status}\",
                    \"text\": \"${message}\",
                    \"footer\": \"Timestamp: ${TIMESTAMP}\"
                }]
            }" \
            "${SLACK_WEBHOOK}" || true
    fi

    # Also send to PagerDuty if critical
    if [ "${status}" == "INITIATED" ] || [ "${status}" == "FAILED" ]; then
        if [ -n "${PAGERDUTY_KEY:-}" ]; then
            curl -s -X POST -H 'Content-Type: application/json' \
                -d "{
                    \"routing_key\": \"${PAGERDUTY_KEY}\",
                    \"event_action\": \"trigger\",
                    \"payload\": {
                        \"summary\": \"GL-001 Failover ${status}: ${message}\",
                        \"severity\": \"critical\",
                        \"source\": \"gl-001-thermalcommand\"
                    }
                }" \
                'https://events.pagerduty.com/v2/enqueue' || true
        fi
    fi
}

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check kubectl contexts
    if ! kubectl config get-contexts "${PRIMARY_CONTEXT}" &> /dev/null; then
        log_error "Primary context ${PRIMARY_CONTEXT} not found"
        return 1
    fi

    if ! kubectl config get-contexts "${SECONDARY_CONTEXT}" &> /dev/null; then
        log_error "Secondary context ${SECONDARY_CONTEXT} not found"
        return 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        return 1
    fi

    log_success "Pre-flight checks passed"
}

# ==============================================================================
# Region Health Check
# ==============================================================================

check_region_health() {
    local context="$1"
    local region="$2"

    log_info "Checking health of region: ${region}"

    kubectl config use-context "${context}" > /dev/null 2>&1

    local healthy=true

    # Check namespace exists
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_error "Namespace ${NAMESPACE} not found in ${region}"
        healthy=false
    fi

    # Check application pods
    local ready_pods=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-thermalcommand \
        -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | \
        grep -c "True" || echo "0")

    if [ "${ready_pods}" -ge 2 ]; then
        log_success "Application pods healthy: ${ready_pods} ready"
    else
        log_warn "Application pods unhealthy: only ${ready_pods} ready"
        healthy=false
    fi

    # Check PostgreSQL
    local pg_leader=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -n "${pg_leader}" ]; then
        log_success "PostgreSQL leader found: ${pg_leader}"
    else
        log_warn "PostgreSQL leader not found"
        healthy=false
    fi

    # Check Redis
    local redis_master=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-redis,role=master \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -n "${redis_master}" ]; then
        log_success "Redis master found: ${redis_master}"
    else
        log_warn "Redis master not found"
        healthy=false
    fi

    $healthy && return 0 || return 1
}

# ==============================================================================
# Regional Failover
# ==============================================================================

failover_region() {
    log_info "========================================="
    log_info "INITIATING REGIONAL FAILOVER"
    log_info "From: ${PRIMARY_REGION} -> To: ${SECONDARY_REGION}"
    log_info "========================================="

    send_notification "INITIATED" "Regional failover from ${PRIMARY_REGION} to ${SECONDARY_REGION}" "#ffcc00"

    # Step 1: Verify primary is actually down
    log_info "Step 1: Verifying primary region status..."
    if check_region_health "${PRIMARY_CONTEXT}" "${PRIMARY_REGION}"; then
        log_warn "Primary region appears healthy!"
        read -p "Primary region is healthy. Continue anyway? (yes/no): " confirm
        if [ "${confirm}" != "yes" ]; then
            log_info "Failover cancelled by user"
            exit 0
        fi
    else
        log_info "Primary region confirmed unhealthy"
    fi

    # Step 2: Verify secondary is ready
    log_info "Step 2: Verifying secondary region readiness..."
    kubectl config use-context "${SECONDARY_CONTEXT}"

    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_error "Secondary region namespace not found"
        send_notification "FAILED" "Secondary region not ready" "#ff0000"
        exit 1
    fi

    # Step 3: Scale up secondary application
    log_info "Step 3: Scaling up secondary region..."
    kubectl scale deployment gl-001-thermalcommand \
        --replicas=5 \
        -n "${NAMESPACE}"

    # Step 4: Wait for pods to be ready
    log_info "Step 4: Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod \
        -l app=gl-001-thermalcommand \
        -n "${NAMESPACE}" \
        --timeout=300s

    # Step 5: Promote PostgreSQL replica
    log_info "Step 5: Promoting PostgreSQL replica..."
    failover_database

    # Step 6: Verify Redis Sentinel has elected master
    log_info "Step 6: Verifying Redis master..."
    local redis_master=$(kubectl exec -n "${NAMESPACE}" gl-001-redis-sentinel-0 -- \
        redis-cli -p 26379 SENTINEL get-master-addr-by-name gl-001-redis 2>/dev/null | head -1)
    log_info "Redis master: ${redis_master}"

    # Step 7: Update DNS
    log_info "Step 7: Updating DNS..."
    update_dns "${SECONDARY_REGION}"

    # Step 8: Verify services
    log_info "Step 8: Verifying services..."
    sleep 30  # Wait for DNS propagation

    if verify_services; then
        log_success "Regional failover completed successfully!"
        send_notification "SUCCESS" "Regional failover to ${SECONDARY_REGION} completed" "#36a64f"
    else
        log_error "Service verification failed!"
        send_notification "FAILED" "Service verification failed after failover" "#ff0000"
        exit 1
    fi
}

# ==============================================================================
# Database Failover
# ==============================================================================

failover_database() {
    log_info "Initiating PostgreSQL failover..."

    # Get Patroni leader
    local patroni_leader=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres \
        -o jsonpath='{.items[0].metadata.name}')

    if [ -z "${patroni_leader}" ]; then
        log_error "No PostgreSQL pods found"
        return 1
    fi

    # Check current cluster status
    log_info "Current Patroni cluster status:"
    kubectl exec -n "${NAMESPACE}" "${patroni_leader}" -- \
        patronictl list 2>/dev/null || true

    # Get replicas
    local replicas=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | \
        grep -v "${patroni_leader}" || true)

    if [ -z "${replicas}" ]; then
        log_error "No replicas available for failover"
        return 1
    fi

    local target_replica=$(echo "${replicas}" | head -1)
    log_info "Target replica for promotion: ${target_replica}"

    # Perform failover
    log_info "Executing Patroni failover..."
    kubectl exec -n "${NAMESPACE}" "${patroni_leader}" -- \
        patronictl switchover \
        --master "${patroni_leader}" \
        --candidate "${target_replica}" \
        --force

    # Wait for failover to complete
    sleep 10

    # Verify new leader
    local new_leader=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}')

    if [ "${new_leader}" == "${target_replica}" ]; then
        log_success "PostgreSQL failover completed. New leader: ${new_leader}"
    else
        log_warn "Failover completed but leader is: ${new_leader}"
    fi

    # Show new cluster status
    log_info "New Patroni cluster status:"
    kubectl exec -n "${NAMESPACE}" "${new_leader}" -- \
        patronictl list 2>/dev/null || true
}

# ==============================================================================
# Redis Failover
# ==============================================================================

failover_redis() {
    log_info "Initiating Redis Sentinel failover..."

    # Get Sentinel pod
    local sentinel_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-redis-sentinel \
        -o jsonpath='{.items[0].metadata.name}')

    if [ -z "${sentinel_pod}" ]; then
        log_error "No Sentinel pods found"
        return 1
    fi

    # Get current master
    local current_master=$(kubectl exec -n "${NAMESPACE}" "${sentinel_pod}" -- \
        redis-cli -p 26379 SENTINEL get-master-addr-by-name gl-001-redis 2>/dev/null)

    log_info "Current Redis master: ${current_master}"

    # Trigger failover
    log_info "Triggering Sentinel failover..."
    kubectl exec -n "${NAMESPACE}" "${sentinel_pod}" -- \
        redis-cli -p 26379 SENTINEL failover gl-001-redis

    # Wait for failover
    sleep 15

    # Verify new master
    local new_master=$(kubectl exec -n "${NAMESPACE}" "${sentinel_pod}" -- \
        redis-cli -p 26379 SENTINEL get-master-addr-by-name gl-001-redis 2>/dev/null)

    log_info "New Redis master: ${new_master}"

    if [ "${current_master}" != "${new_master}" ]; then
        log_success "Redis failover completed"
    else
        log_warn "Redis master unchanged (might already be optimal)"
    fi

    # Show Sentinel info
    log_info "Sentinel status:"
    kubectl exec -n "${NAMESPACE}" "${sentinel_pod}" -- \
        redis-cli -p 26379 SENTINEL master gl-001-redis 2>/dev/null | head -20
}

# ==============================================================================
# Kafka Rebalance
# ==============================================================================

failover_kafka() {
    log_info "Initiating Kafka partition rebalance..."

    # Get Kafka pod
    local kafka_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-kafka \
        -o jsonpath='{.items[0].metadata.name}')

    if [ -z "${kafka_pod}" ]; then
        log_error "No Kafka pods found"
        return 1
    fi

    # Check broker status
    log_info "Current broker status:"
    kubectl exec -n "${NAMESPACE}" "${kafka_pod}" -- \
        kafka-broker-api-versions --bootstrap-server localhost:9092 2>/dev/null | head -10

    # Check under-replicated partitions
    log_info "Checking under-replicated partitions..."
    local under_replicated=$(kubectl exec -n "${NAMESPACE}" "${kafka_pod}" -- \
        kafka-topics --bootstrap-server localhost:9092 \
        --describe --under-replicated-partitions 2>/dev/null)

    if [ -n "${under_replicated}" ]; then
        log_warn "Under-replicated partitions found:"
        echo "${under_replicated}"
    else
        log_success "No under-replicated partitions"
    fi

    # Trigger preferred leader election
    log_info "Triggering preferred leader election..."
    kubectl exec -n "${NAMESPACE}" "${kafka_pod}" -- \
        kafka-leader-election --bootstrap-server localhost:9092 \
        --all-topic-partitions --election-type preferred 2>/dev/null || true

    log_success "Kafka rebalance triggered"
}

# ==============================================================================
# DNS Update
# ==============================================================================

update_dns() {
    local target_region="$1"

    log_info "Updating DNS to point to ${target_region}..."

    # Determine ALB hostname based on region
    local alb_hostname=""
    case "${target_region}" in
        us-east-1)
            alb_hostname="gl-001-alb-us-east-1.greenlang.io"
            ;;
        us-west-2)
            alb_hostname="gl-001-alb-us-west-2.greenlang.io"
            ;;
        *)
            log_error "Unknown region: ${target_region}"
            return 1
            ;;
    esac

    # Update Route53
    cat > /tmp/dns-update.json << EOF
{
    "Changes": [
        {
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "gl-001.greenlang.io",
                "Type": "CNAME",
                "TTL": 60,
                "ResourceRecords": [
                    {
                        "Value": "${alb_hostname}"
                    }
                ]
            }
        }
    ]
}
EOF

    # Get hosted zone ID
    local hosted_zone_id="${ROUTE53_HOSTED_ZONE_ID:-}"
    if [ -z "${hosted_zone_id}" ]; then
        hosted_zone_id=$(aws route53 list-hosted-zones-by-name \
            --dns-name "greenlang.io" \
            --query 'HostedZones[0].Id' \
            --output text | sed 's|/hostedzone/||')
    fi

    # Apply DNS change
    aws route53 change-resource-record-sets \
        --hosted-zone-id "${hosted_zone_id}" \
        --change-batch file:///tmp/dns-update.json

    log_success "DNS updated to ${alb_hostname}"

    # Wait for propagation
    log_info "Waiting for DNS propagation..."
    sleep 30
}

# ==============================================================================
# Service Verification
# ==============================================================================

verify_services() {
    log_info "Verifying services..."

    local all_healthy=true

    # Check application health endpoint
    local health_status=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://gl-001.greenlang.io/api/v1/health" 2>/dev/null || echo "000")

    if [ "${health_status}" == "200" ]; then
        log_success "Application health check: OK"
    else
        log_error "Application health check: FAILED (${health_status})"
        all_healthy=false
    fi

    # Check deep health
    local deep_health=$(curl -s "https://gl-001.greenlang.io/api/v1/health/deep" 2>/dev/null)
    log_info "Deep health response: ${deep_health}"

    # Check database connectivity (via app)
    local db_check=$(echo "${deep_health}" | jq -r '.database // "unknown"' 2>/dev/null)
    if [ "${db_check}" == "healthy" ]; then
        log_success "Database connectivity: OK"
    else
        log_warn "Database connectivity: ${db_check}"
    fi

    # Check Redis connectivity (via app)
    local redis_check=$(echo "${deep_health}" | jq -r '.redis // "unknown"' 2>/dev/null)
    if [ "${redis_check}" == "healthy" ]; then
        log_success "Redis connectivity: OK"
    else
        log_warn "Redis connectivity: ${redis_check}"
    fi

    # Check Kafka connectivity (via app)
    local kafka_check=$(echo "${deep_health}" | jq -r '.kafka // "unknown"' 2>/dev/null)
    if [ "${kafka_check}" == "healthy" ]; then
        log_success "Kafka connectivity: OK"
    else
        log_warn "Kafka connectivity: ${kafka_check}"
    fi

    $all_healthy && return 0 || return 1
}

# ==============================================================================
# Status Check
# ==============================================================================

check_status() {
    log_info "========================================="
    log_info "GL-001 ThermalCommand Failover Status"
    log_info "========================================="

    # Check primary region
    log_info ""
    log_info "=== Primary Region (${PRIMARY_REGION}) ==="
    if check_region_health "${PRIMARY_CONTEXT}" "${PRIMARY_REGION}"; then
        log_success "Primary region: HEALTHY"
    else
        log_error "Primary region: UNHEALTHY"
    fi

    # Check secondary region
    log_info ""
    log_info "=== Secondary Region (${SECONDARY_REGION}) ==="
    if check_region_health "${SECONDARY_CONTEXT}" "${SECONDARY_REGION}"; then
        log_success "Secondary region: HEALTHY"
    else
        log_warn "Secondary region: NOT READY"
    fi

    # Check DNS
    log_info ""
    log_info "=== DNS Status ==="
    local dns_target=$(dig +short gl-001.greenlang.io CNAME 2>/dev/null | head -1)
    log_info "DNS currently points to: ${dns_target:-unknown}"

    # Check active region
    log_info ""
    log_info "=== Active Region ==="
    if echo "${dns_target}" | grep -q "us-east-1"; then
        log_info "Active region: ${PRIMARY_REGION} (primary)"
    elif echo "${dns_target}" | grep -q "us-west-2"; then
        log_info "Active region: ${SECONDARY_REGION} (secondary/DR)"
    else
        log_warn "Active region: unknown"
    fi
}

# ==============================================================================
# Rollback
# ==============================================================================

rollback_to_primary() {
    log_info "========================================="
    log_info "INITIATING ROLLBACK TO PRIMARY"
    log_info "From: ${SECONDARY_REGION} -> To: ${PRIMARY_REGION}"
    log_info "========================================="

    send_notification "INITIATED" "Rollback to ${PRIMARY_REGION}" "#ffcc00"

    # Verify primary is healthy
    if ! check_region_health "${PRIMARY_CONTEXT}" "${PRIMARY_REGION}"; then
        log_error "Primary region is not healthy. Cannot rollback."
        exit 1
    fi

    # Update DNS
    update_dns "${PRIMARY_REGION}"

    # Verify services
    if verify_services; then
        log_success "Rollback to primary completed!"
        send_notification "SUCCESS" "Rollback to ${PRIMARY_REGION} completed" "#36a64f"
    else
        log_error "Service verification failed after rollback!"
        send_notification "FAILED" "Service verification failed after rollback" "#ff0000"
        exit 1
    fi

    # Scale down secondary
    log_info "Scaling down secondary region..."
    kubectl config use-context "${SECONDARY_CONTEXT}"
    kubectl scale deployment gl-001-thermalcommand \
        --replicas=2 \
        -n "${NAMESPACE}"
}

# ==============================================================================
# Help
# ==============================================================================

show_help() {
    cat << EOF
GL-001 ThermalCommand - Manual Failover Script

Usage: $0 <operation> [options]

Operations:
    region          Failover to secondary region
    database        Failover PostgreSQL to replica
    redis           Failover Redis via Sentinel
    kafka           Rebalance Kafka partitions
    verify          Verify current failover status
    rollback        Rollback to primary region

Options:
    --primary-region    Primary AWS region (default: us-east-1)
    --secondary-region  Secondary AWS region (default: us-west-2)
    --namespace         Kubernetes namespace (default: greenlang)

Environment Variables:
    SLACK_WEBHOOK           Slack webhook for notifications
    PAGERDUTY_KEY           PagerDuty routing key
    ROUTE53_HOSTED_ZONE_ID  Route53 hosted zone ID

Examples:
    $0 verify                    # Check current status
    $0 database                  # Failover PostgreSQL
    $0 redis                     # Failover Redis
    $0 region                    # Full regional failover
    $0 rollback                  # Rollback to primary

EOF
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    case "${OPERATION}" in
        region)
            preflight_checks
            failover_region
            ;;
        database)
            preflight_checks
            failover_database
            ;;
        redis)
            preflight_checks
            failover_redis
            ;;
        kafka)
            preflight_checks
            failover_kafka
            ;;
        verify)
            check_status
            ;;
        rollback)
            preflight_checks
            rollback_to_primary
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown operation: ${OPERATION}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
