#!/bin/bash
#===============================================================================
# manual-failover.sh
#
# Manual failover script for PostgreSQL/TimescaleDB with Patroni
#
# Usage: ./manual-failover.sh [--candidate <node>] [--force] [--dry-run]
#
# This script performs a controlled failover of the PostgreSQL primary to
# a specified replica or the best available candidate.
#
# Author: GreenLang Database Operations Team
# Version: 1.0.0
# Date: 2026-02-03
#===============================================================================

set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
PATRONI_CONFIG="${PATRONI_CONFIG:-/etc/patroni/patroni.yml}"
CLUSTER_NAME="${CLUSTER_NAME:-greenlang-db}"
NAMESPACE="${NAMESPACE:-database}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
LOG_DIR="${LOG_DIR:-/var/log/database}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

# Script variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/manual_failover_${TIMESTAMP}.log"
DRY_RUN=false
FORCE=false
CANDIDATE=""

#-------------------------------------------------------------------------------
# Logging Functions
#-------------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

#-------------------------------------------------------------------------------
# Notification Functions
#-------------------------------------------------------------------------------
send_slack_notification() {
    local status="$1"
    local message="$2"
    local color=""

    case "$status" in
        "success") color="good" ;;
        "warning") color="warning" ;;
        "error")   color="danger" ;;
        *)         color="#808080" ;;
    esac

    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"Database Failover Notification\",
                    \"text\": \"${message}\",
                    \"fields\": [
                        {\"title\": \"Cluster\", \"value\": \"${CLUSTER_NAME}\", \"short\": true},
                        {\"title\": \"Timestamp\", \"value\": \"$(date -Iseconds)\", \"short\": true}
                    ],
                    \"footer\": \"GreenLang Database Operations\"
                }]
            }" 2>/dev/null || true
    fi
}

#-------------------------------------------------------------------------------
# Utility Functions
#-------------------------------------------------------------------------------
ensure_log_directory() {
    mkdir -p "$LOG_DIR"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for required commands
    local required_commands=("patronictl" "psql" "jq" "curl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Required command not found: $cmd"
            return 1
        fi
    done

    # Check Patroni config exists
    if [ ! -f "$PATRONI_CONFIG" ]; then
        log_error "Patroni configuration not found: $PATRONI_CONFIG"
        return 1
    fi

    # Check cluster is accessible
    if ! patronictl -c "$PATRONI_CONFIG" list &>/dev/null; then
        log_error "Cannot access Patroni cluster"
        return 1
    fi

    log_success "Prerequisites check passed"
    return 0
}

#-------------------------------------------------------------------------------
# Cluster Status Functions
#-------------------------------------------------------------------------------
get_cluster_status() {
    log_info "Retrieving cluster status..."

    local status
    status=$(patronictl -c "$PATRONI_CONFIG" list -f json 2>/dev/null)

    if [ -z "$status" ] || [ "$status" == "null" ]; then
        log_error "Failed to get cluster status"
        return 1
    fi

    echo "$status"
}

display_cluster_status() {
    log_info "Current cluster status:"
    patronictl -c "$PATRONI_CONFIG" list | tee -a "$LOG_FILE"
}

get_current_primary() {
    local status
    status=$(get_cluster_status)
    echo "$status" | jq -r '.[] | select(.Role == "Leader") | .Member'
}

get_sync_standby() {
    local status
    status=$(get_cluster_status)
    echo "$status" | jq -r '.[] | select(.Role == "Sync Standby") | .Member' | head -1
}

get_best_candidate() {
    local status
    status=$(get_cluster_status)

    # Priority: Sync Standby > Replica with zero lag > Any replica
    local candidate

    # Try sync standby first
    candidate=$(echo "$status" | jq -r '.[] | select(.Role == "Sync Standby") | .Member' | head -1)

    if [ -z "$candidate" ] || [ "$candidate" == "null" ]; then
        # Try replica with zero lag
        candidate=$(echo "$status" | jq -r '.[] | select(.Role == "Replica" and (.Lag == 0 or .Lag == null)) | .Member' | head -1)
    fi

    if [ -z "$candidate" ] || [ "$candidate" == "null" ]; then
        # Any replica
        candidate=$(echo "$status" | jq -r '.[] | select(.Role == "Replica") | .Member' | head -1)
    fi

    echo "$candidate"
}

#-------------------------------------------------------------------------------
# Health Check Functions
#-------------------------------------------------------------------------------
check_replication_lag() {
    log_info "Checking replication lag..."

    local primary
    primary=$(get_current_primary)

    if [ -z "$primary" ]; then
        log_warn "No primary found, skipping replication lag check"
        return 0
    fi

    local lag_info
    lag_info=$(patronictl -c "$PATRONI_CONFIG" list -f json | jq -r '.[] | select(.Role != "Leader") | "\(.Member): \(.Lag // 0) bytes"')

    log_info "Replication lag status:"
    echo "$lag_info" | tee -a "$LOG_FILE"

    # Check for significant lag
    local max_lag
    max_lag=$(patronictl -c "$PATRONI_CONFIG" list -f json | jq -r '[.[] | select(.Role != "Leader") | .Lag // 0] | max')

    if [ "$max_lag" != "0" ] && [ "$max_lag" != "null" ] && [ -n "$max_lag" ]; then
        log_warn "Replication lag detected: ${max_lag} bytes"
        if [ "$FORCE" != "true" ]; then
            log_error "Use --force to proceed with failover despite lag"
            return 1
        fi
    fi

    return 0
}

check_long_running_transactions() {
    log_info "Checking for long-running transactions..."

    local primary
    primary=$(get_current_primary)

    if [ -z "$primary" ]; then
        return 0
    fi

    local long_txns
    long_txns=$(psql -h "$primary" -U postgres -t -c "
        SELECT count(*)
        FROM pg_stat_activity
        WHERE xact_start IS NOT NULL
          AND now() - xact_start > interval '5 minutes'
          AND state != 'idle';
    " 2>/dev/null | tr -d ' ')

    if [ -n "$long_txns" ] && [ "$long_txns" -gt 0 ]; then
        log_warn "Found $long_txns long-running transactions"
        if [ "$FORCE" != "true" ]; then
            log_info "Consider waiting for transactions to complete or use --force"
        fi
    fi

    return 0
}

verify_candidate() {
    local candidate="$1"

    log_info "Verifying candidate: $candidate"

    local status
    status=$(get_cluster_status)

    # Check candidate exists
    local candidate_info
    candidate_info=$(echo "$status" | jq -r ".[] | select(.Member == \"$candidate\")")

    if [ -z "$candidate_info" ] || [ "$candidate_info" == "null" ]; then
        log_error "Candidate $candidate not found in cluster"
        return 1
    fi

    # Check candidate is not the current primary
    local candidate_role
    candidate_role=$(echo "$candidate_info" | jq -r '.Role')

    if [ "$candidate_role" == "Leader" ]; then
        log_error "Candidate $candidate is already the primary"
        return 1
    fi

    # Check candidate state
    local candidate_state
    candidate_state=$(echo "$candidate_info" | jq -r '.State')

    if [ "$candidate_state" != "running" ]; then
        log_error "Candidate $candidate is not in running state: $candidate_state"
        return 1
    fi

    # Check candidate lag
    local candidate_lag
    candidate_lag=$(echo "$candidate_info" | jq -r '.Lag // 0')

    if [ "$candidate_lag" != "0" ] && [ "$candidate_lag" != "null" ] && [ -n "$candidate_lag" ]; then
        log_warn "Candidate $candidate has replication lag: $candidate_lag bytes"
        if [ "$FORCE" != "true" ]; then
            return 1
        fi
    fi

    log_success "Candidate $candidate verified"
    return 0
}

#-------------------------------------------------------------------------------
# Failover Functions
#-------------------------------------------------------------------------------
execute_failover() {
    local candidate="$1"
    local current_primary
    current_primary=$(get_current_primary)

    log_info "Executing failover: $current_primary -> $candidate"

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would execute: patronictl failover --candidate $candidate --force"
        return 0
    fi

    # Send notification
    send_slack_notification "warning" "Initiating failover from $current_primary to $candidate"

    # Execute failover
    local failover_output
    if [ "$FORCE" == "true" ]; then
        failover_output=$(patronictl -c "$PATRONI_CONFIG" failover \
            --candidate "$candidate" \
            --force 2>&1)
    else
        failover_output=$(patronictl -c "$PATRONI_CONFIG" failover \
            --candidate "$candidate" 2>&1)
    fi

    log_info "Failover command output:"
    echo "$failover_output" | tee -a "$LOG_FILE"

    return 0
}

wait_for_failover() {
    local expected_primary="$1"
    local timeout=$TIMEOUT_SECONDS
    local elapsed=0
    local interval=5

    log_info "Waiting for failover to complete (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        local current_primary
        current_primary=$(get_current_primary)

        if [ "$current_primary" == "$expected_primary" ]; then
            log_success "Failover completed. New primary: $current_primary"
            return 0
        fi

        log_info "Waiting for $expected_primary to become primary... (${elapsed}s/${timeout}s)"
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log_error "Failover timeout exceeded. Expected primary: $expected_primary"
    return 1
}

verify_new_primary() {
    local expected_primary="$1"

    log_info "Verifying new primary..."

    # Check it's the leader
    local current_primary
    current_primary=$(get_current_primary)

    if [ "$current_primary" != "$expected_primary" ]; then
        log_error "Primary mismatch. Expected: $expected_primary, Got: $current_primary"
        return 1
    fi

    # Check it's accepting writes
    local write_test
    write_test=$(psql -h "$expected_primary" -U postgres -t -c "
        SELECT pg_is_in_recovery();
    " 2>/dev/null | tr -d ' ')

    if [ "$write_test" == "f" ]; then
        log_success "New primary is accepting writes"
    else
        log_error "New primary is still in recovery mode"
        return 1
    fi

    return 0
}

update_dns() {
    log_info "Updating DNS records..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would update DNS records"
        return 0
    fi

    # If using Kubernetes, the service will automatically update
    # For external DNS, add your DNS update logic here

    # Example for AWS Route53:
    # aws route53 change-resource-record-sets \
    #     --hosted-zone-id ${HOSTED_ZONE_ID} \
    #     --change-batch file://dns-update.json

    log_info "DNS update not configured (using Kubernetes service discovery)"
    return 0
}

#-------------------------------------------------------------------------------
# Post-Failover Checks
#-------------------------------------------------------------------------------
post_failover_checks() {
    local new_primary="$1"

    log_info "Running post-failover checks..."

    # Check replication is established
    log_info "Checking replication status..."
    local replication_count
    replication_count=$(psql -h "$new_primary" -U postgres -t -c "
        SELECT count(*) FROM pg_stat_replication;
    " 2>/dev/null | tr -d ' ')

    log_info "Active replication connections: $replication_count"

    # Check cluster status
    display_cluster_status

    # Check for any issues
    log_info "Checking for cluster issues..."
    local cluster_issues
    cluster_issues=$(patronictl -c "$PATRONI_CONFIG" list -f json | \
        jq -r '.[] | select(.State != "running") | "\(.Member): \(.State)"')

    if [ -n "$cluster_issues" ]; then
        log_warn "Nodes with issues:"
        echo "$cluster_issues" | tee -a "$LOG_FILE"
    else
        log_success "All nodes are running"
    fi

    return 0
}

#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
main() {
    local start_time=$(date +%s)

    ensure_log_directory

    log_info "=========================================="
    log_info "Starting manual failover procedure"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Dry run: $DRY_RUN"
    log_info "Force: $FORCE"
    log_info "Candidate: ${CANDIDATE:-auto}"
    log_info "=========================================="

    # Pre-flight checks
    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi

    # Display current status
    display_cluster_status

    # Get current primary
    local current_primary
    current_primary=$(get_current_primary)

    if [ -z "$current_primary" ]; then
        log_warn "No current primary found - cluster may already be in failover"
    else
        log_info "Current primary: $current_primary"
    fi

    # Determine candidate
    local target_candidate
    if [ -n "$CANDIDATE" ]; then
        target_candidate="$CANDIDATE"
    else
        target_candidate=$(get_best_candidate)
    fi

    if [ -z "$target_candidate" ]; then
        log_error "No suitable failover candidate found"
        send_slack_notification "error" "Failover failed: No suitable candidate"
        exit 1
    fi

    log_info "Selected failover candidate: $target_candidate"

    # Verify candidate
    if ! verify_candidate "$target_candidate"; then
        log_error "Candidate verification failed"
        send_slack_notification "error" "Failover failed: Candidate verification failed"
        exit 1
    fi

    # Check replication lag
    if ! check_replication_lag; then
        log_error "Replication lag check failed"
        exit 1
    fi

    # Check for long-running transactions
    check_long_running_transactions

    # Execute failover
    if ! execute_failover "$target_candidate"; then
        log_error "Failover execution failed"
        send_slack_notification "error" "Failover execution failed"
        exit 1
    fi

    # Wait for failover to complete
    if [ "$DRY_RUN" != "true" ]; then
        if ! wait_for_failover "$target_candidate"; then
            log_error "Failover did not complete within timeout"
            send_slack_notification "error" "Failover timeout"
            exit 1
        fi

        # Verify new primary
        if ! verify_new_primary "$target_candidate"; then
            log_error "New primary verification failed"
            send_slack_notification "error" "New primary verification failed"
            exit 1
        fi

        # Update DNS
        update_dns

        # Post-failover checks
        post_failover_checks "$target_candidate"
    fi

    # Calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "=========================================="
    log_success "Failover completed successfully"
    log_success "New primary: $target_candidate"
    log_success "Duration: ${duration} seconds"
    log_success "Log file: $LOG_FILE"
    log_success "=========================================="

    # Send success notification
    send_slack_notification "success" "Failover completed. New primary: $target_candidate (Duration: ${duration}s)"

    return 0
}

#-------------------------------------------------------------------------------
# Argument Parsing
#-------------------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Manual failover script for PostgreSQL/TimescaleDB with Patroni

OPTIONS:
    --candidate <node>    Specify failover target (default: auto-select)
    --force               Force failover even with replication lag
    --dry-run             Show what would be done without executing
    -h, --help            Show this help message

ENVIRONMENT VARIABLES:
    PATRONI_CONFIG        Path to Patroni configuration (default: /etc/patroni/patroni.yml)
    CLUSTER_NAME          Cluster name (default: greenlang-db)
    NAMESPACE             Kubernetes namespace (default: database)
    SLACK_WEBHOOK_URL     Slack webhook for notifications
    LOG_DIR               Log directory (default: /var/log/database)
    TIMEOUT_SECONDS       Failover timeout (default: 300)

EXAMPLES:
    # Auto-select best candidate
    $(basename "$0")

    # Specify candidate
    $(basename "$0") --candidate greenlang-db-1

    # Force failover despite lag
    $(basename "$0") --candidate greenlang-db-1 --force

    # Dry run
    $(basename "$0") --dry-run

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --candidate)
            CANDIDATE="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main
