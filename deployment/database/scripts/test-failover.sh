#!/bin/bash
#===============================================================================
# test-failover.sh
#
# Automated failover test script for PostgreSQL/TimescaleDB
#
# This script performs automated failover testing including:
# - Pre-test health validation
# - Controlled failover execution
# - RTO/RPO measurement
# - Data integrity verification
# - Automated rollback (switchback)
# - Test report generation
#
# Usage: ./test-failover.sh [--no-rollback] [--report-only] [--output <file>]
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
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-greenlang}"

# Test configuration
TEST_TABLE="${TEST_TABLE:-failover_test_data}"
TEST_DURATION_THRESHOLD_SECONDS="${TEST_DURATION_THRESHOLD_SECONDS:-300}"
LAG_THRESHOLD_BYTES="${LAG_THRESHOLD_BYTES:-0}"

# Output configuration
LOG_DIR="${LOG_DIR:-/var/log/database}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/failover_test_${TIMESTAMP}.log"
REPORT_FILE="${LOG_DIR}/failover_test_report_${TIMESTAMP}.json"

# Options
NO_ROLLBACK=false
REPORT_ONLY=false
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# Test metrics
declare -A METRICS

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
send_notification() {
    local status="$1"
    local message="$2"

    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local color
        case "$status" in
            "success") color="good" ;;
            "warning") color="warning" ;;
            "error")   color="danger" ;;
            "info")    color="#439FE0" ;;
            *)         color="#808080" ;;
        esac

        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"Failover Test Notification\",
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
ensure_directories() {
    mkdir -p "$LOG_DIR"
}

get_cluster_status() {
    patronictl -c "$PATRONI_CONFIG" list -f json 2>/dev/null || echo "[]"
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

run_query() {
    local query="$1"
    local host="${2:-localhost}"
    psql -h "$host" -U "$DB_USER" -d "$DB_NAME" -t -A -c "$query" 2>/dev/null
}

#-------------------------------------------------------------------------------
# Pre-Test Validation
#-------------------------------------------------------------------------------
validate_cluster_health() {
    log_info "Validating cluster health before test..."

    local status
    status=$(get_cluster_status)

    # Check for single primary
    local primary_count
    primary_count=$(echo "$status" | jq '[.[] | select(.Role == "Leader")] | length')

    if [ "$primary_count" -ne 1 ]; then
        log_error "Invalid primary count: $primary_count (expected 1)"
        return 1
    fi

    # Check all nodes are running
    local total_nodes
    total_nodes=$(echo "$status" | jq 'length')

    local running_nodes
    running_nodes=$(echo "$status" | jq '[.[] | select(.State == "running")] | length')

    if [ "$running_nodes" -ne "$total_nodes" ]; then
        log_error "Not all nodes are running: $running_nodes/$total_nodes"
        return 1
    fi

    # Check for sync standby
    local sync_standby
    sync_standby=$(get_sync_standby)

    if [ -z "$sync_standby" ]; then
        log_warn "No synchronous standby found"
    else
        log_success "Sync standby: $sync_standby"
    fi

    # Check replication lag
    local max_lag
    max_lag=$(echo "$status" | jq -r '[.[] | select(.Role != "Leader") | .Lag // 0] | max')

    if [ "${max_lag:-0}" -gt "$LAG_THRESHOLD_BYTES" ]; then
        log_error "Replication lag exceeds threshold: $max_lag bytes"
        return 1
    fi

    log_success "Cluster health validation passed"
    return 0
}

#-------------------------------------------------------------------------------
# Test Data Management
#-------------------------------------------------------------------------------
create_test_table() {
    log_info "Creating test table..."

    local primary
    primary=$(get_current_primary)

    run_query "
        DROP TABLE IF EXISTS ${TEST_TABLE};
        CREATE TABLE ${TEST_TABLE} (
            id SERIAL PRIMARY KEY,
            test_id VARCHAR(50) NOT NULL,
            write_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            data TEXT,
            verified BOOLEAN DEFAULT FALSE
        );
        CREATE INDEX idx_${TEST_TABLE}_test_id ON ${TEST_TABLE}(test_id);
    " "$primary"

    log_success "Test table created"
}

insert_test_data() {
    local test_id="$1"
    local primary
    primary=$(get_current_primary)

    local insert_time
    insert_time=$(date -Iseconds)

    run_query "
        INSERT INTO ${TEST_TABLE} (test_id, data)
        VALUES ('$test_id', 'Test data inserted at $insert_time');
    " "$primary"

    METRICS["last_write_time"]="$insert_time"
    log_info "Inserted test data with ID: $test_id"
}

verify_test_data() {
    local test_id="$1"
    local host="$2"

    local count
    count=$(run_query "SELECT count(*) FROM ${TEST_TABLE} WHERE test_id = '$test_id';" "$host")

    if [ "${count:-0}" -gt 0 ]; then
        log_success "Test data verified on $host"
        return 0
    else
        log_error "Test data not found on $host"
        return 1
    fi
}

cleanup_test_data() {
    log_info "Cleaning up test data..."

    local primary
    primary=$(get_current_primary)

    run_query "DROP TABLE IF EXISTS ${TEST_TABLE};" "$primary" || true

    log_success "Test data cleaned up"
}

#-------------------------------------------------------------------------------
# Failover Test Execution
#-------------------------------------------------------------------------------
measure_downtime() {
    local start_time="$1"
    local host="$2"
    local timeout="${3:-60}"

    local elapsed=0
    local interval=1

    while [ $elapsed -lt $timeout ]; do
        if psql -h "$host" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &>/dev/null; then
            echo "$elapsed"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo "$timeout"
    return 1
}

execute_failover_test() {
    log_info "=========================================="
    log_info "Starting Failover Test"
    log_info "=========================================="

    local test_id="test_${TIMESTAMP}"
    local original_primary
    original_primary=$(get_current_primary)

    local failover_target
    failover_target=$(get_sync_standby)

    if [ -z "$failover_target" ]; then
        # Fall back to any replica
        failover_target=$(get_cluster_status | jq -r '.[] | select(.Role == "Replica") | .Member' | head -1)
    fi

    if [ -z "$failover_target" ]; then
        log_error "No failover target available"
        return 1
    fi

    METRICS["original_primary"]="$original_primary"
    METRICS["failover_target"]="$failover_target"
    METRICS["test_id"]="$test_id"

    log_info "Original primary: $original_primary"
    log_info "Failover target: $failover_target"

    # Step 1: Create test table and insert data
    create_test_table
    insert_test_data "$test_id"

    # Step 2: Verify data exists before failover
    if ! verify_test_data "$test_id" "$original_primary"; then
        log_error "Test data verification failed before failover"
        return 1
    fi

    # Record pre-failover state
    local pre_failover_lsn
    pre_failover_lsn=$(run_query "SELECT pg_current_wal_lsn()::text;" "$original_primary")
    METRICS["pre_failover_lsn"]="$pre_failover_lsn"

    # Step 3: Execute failover
    log_info "Executing failover to $failover_target..."
    local failover_start_time
    failover_start_time=$(date +%s.%N)
    METRICS["failover_start_time"]="$(date -Iseconds)"

    # Trigger failover
    patronictl -c "$PATRONI_CONFIG" switchover \
        --master "$original_primary" \
        --candidate "$failover_target" \
        --scheduled now \
        --force 2>&1 | tee -a "$LOG_FILE"

    # Step 4: Measure RTO (time until new primary accepts writes)
    log_info "Measuring recovery time..."
    local write_available_time=0
    local max_wait=120
    local check_interval=1
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        local current_primary
        current_primary=$(get_current_primary)

        if [ "$current_primary" == "$failover_target" ]; then
            # Try to write
            if run_query "SELECT 1;" "$failover_target" &>/dev/null; then
                write_available_time=$elapsed
                break
            fi
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    local failover_end_time
    failover_end_time=$(date +%s.%N)
    METRICS["failover_end_time"]="$(date -Iseconds)"

    # Calculate RTO
    local rto_seconds
    rto_seconds=$(echo "$failover_end_time - $failover_start_time" | bc)
    METRICS["rto_seconds"]="$rto_seconds"

    log_info "RTO (Recovery Time Objective): ${rto_seconds} seconds"

    # Step 5: Verify new primary
    local new_primary
    new_primary=$(get_current_primary)

    if [ "$new_primary" != "$failover_target" ]; then
        log_error "Failover failed: expected $failover_target, got $new_primary"
        METRICS["failover_success"]="false"
        return 1
    fi

    METRICS["failover_success"]="true"
    METRICS["new_primary"]="$new_primary"
    log_success "Failover successful: new primary is $new_primary"

    # Step 6: Measure RPO (check for data loss)
    log_info "Verifying data integrity (RPO check)..."

    local data_verified=false
    if verify_test_data "$test_id" "$new_primary"; then
        data_verified=true
        METRICS["data_loss"]="false"
        METRICS["rpo_seconds"]="0"
        log_success "RPO (Recovery Point Objective): 0 (no data loss)"
    else
        METRICS["data_loss"]="true"
        log_error "Data loss detected!"
    fi

    # Step 7: Verify replication is re-established
    log_info "Verifying replication health..."
    sleep 10  # Wait for replication to stabilize

    local replication_count
    replication_count=$(run_query "SELECT count(*) FROM pg_stat_replication;" "$new_primary")

    METRICS["replication_count"]="$replication_count"
    log_info "Active replication connections: $replication_count"

    # Step 8: Insert new data on new primary
    log_info "Testing write capability on new primary..."
    local post_failover_test_id="${test_id}_post"
    insert_test_data "$post_failover_test_id"

    if verify_test_data "$post_failover_test_id" "$new_primary"; then
        METRICS["write_after_failover"]="true"
        log_success "Write capability verified on new primary"
    else
        METRICS["write_after_failover"]="false"
        log_error "Write capability test failed"
    fi

    return 0
}

#-------------------------------------------------------------------------------
# Rollback (Switchback)
#-------------------------------------------------------------------------------
execute_rollback() {
    log_info "=========================================="
    log_info "Executing Rollback (Switchback)"
    log_info "=========================================="

    local original_primary="${METRICS["original_primary"]}"
    local current_primary
    current_primary=$(get_current_primary)

    if [ "$current_primary" == "$original_primary" ]; then
        log_info "Already on original primary, no rollback needed"
        return 0
    fi

    # Wait for original primary to become a healthy replica
    log_info "Waiting for original primary to become healthy replica..."
    local max_wait=120
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        local original_state
        original_state=$(get_cluster_status | jq -r ".[] | select(.Member == \"$original_primary\") | .State")

        if [ "$original_state" == "running" ]; then
            break
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done

    # Execute switchback
    log_info "Switching back to original primary: $original_primary"
    local rollback_start_time
    rollback_start_time=$(date +%s.%N)

    patronictl -c "$PATRONI_CONFIG" switchover \
        --master "$current_primary" \
        --candidate "$original_primary" \
        --scheduled now \
        --force 2>&1 | tee -a "$LOG_FILE"

    # Wait for switchback
    max_wait=120
    elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        local new_primary
        new_primary=$(get_current_primary)

        if [ "$new_primary" == "$original_primary" ]; then
            local rollback_end_time
            rollback_end_time=$(date +%s.%N)
            local rollback_duration
            rollback_duration=$(echo "$rollback_end_time - $rollback_start_time" | bc)

            METRICS["rollback_success"]="true"
            METRICS["rollback_duration"]="$rollback_duration"

            log_success "Rollback successful in ${rollback_duration} seconds"
            return 0
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    METRICS["rollback_success"]="false"
    log_error "Rollback failed"
    return 1
}

#-------------------------------------------------------------------------------
# Report Generation
#-------------------------------------------------------------------------------
generate_report() {
    log_info "Generating test report..."

    local test_result="passed"
    if [ "${METRICS["failover_success"]}" != "true" ]; then
        test_result="failed"
    elif [ "${METRICS["data_loss"]}" == "true" ]; then
        test_result="failed"
    elif [ "$(echo "${METRICS["rto_seconds"]} > ${TEST_DURATION_THRESHOLD_SECONDS}" | bc)" -eq 1 ]; then
        test_result="warning"
    fi

    METRICS["test_result"]="$test_result"

    # Generate JSON report
    cat > "$REPORT_FILE" << EOF
{
    "test_metadata": {
        "timestamp": "$(date -Iseconds)",
        "test_id": "${METRICS["test_id"]:-unknown}",
        "cluster_name": "$CLUSTER_NAME",
        "log_file": "$LOG_FILE"
    },
    "cluster_info": {
        "original_primary": "${METRICS["original_primary"]:-unknown}",
        "failover_target": "${METRICS["failover_target"]:-unknown}",
        "new_primary": "${METRICS["new_primary"]:-unknown}"
    },
    "timing": {
        "failover_start": "${METRICS["failover_start_time"]:-unknown}",
        "failover_end": "${METRICS["failover_end_time"]:-unknown}",
        "rto_seconds": ${METRICS["rto_seconds"]:-0},
        "rto_threshold_seconds": $TEST_DURATION_THRESHOLD_SECONDS
    },
    "data_integrity": {
        "data_loss": ${METRICS["data_loss"]:-true},
        "rpo_seconds": ${METRICS["rpo_seconds"]:-0},
        "pre_failover_lsn": "${METRICS["pre_failover_lsn"]:-unknown}",
        "write_after_failover": ${METRICS["write_after_failover"]:-false}
    },
    "replication": {
        "replication_count_after_failover": ${METRICS["replication_count"]:-0}
    },
    "rollback": {
        "rollback_performed": $([ "$NO_ROLLBACK" == "true" ] && echo "false" || echo "true"),
        "rollback_success": ${METRICS["rollback_success"]:-false},
        "rollback_duration_seconds": ${METRICS["rollback_duration"]:-0}
    },
    "result": {
        "overall_status": "$test_result",
        "failover_success": ${METRICS["failover_success"]:-false},
        "rto_within_threshold": $([ "$(echo "${METRICS["rto_seconds"]:-999} <= ${TEST_DURATION_THRESHOLD_SECONDS}" | bc)" -eq 1 ] && echo "true" || echo "false"),
        "rpo_zero": $([ "${METRICS["rpo_seconds"]:-999}" == "0" ] && echo "true" || echo "false")
    }
}
EOF

    log_success "Report generated: $REPORT_FILE"

    # Print summary
    echo ""
    echo "=========================================="
    echo "FAILOVER TEST SUMMARY"
    echo "=========================================="
    echo "Test Result: $test_result"
    echo "RTO: ${METRICS["rto_seconds"]:-unknown} seconds (threshold: ${TEST_DURATION_THRESHOLD_SECONDS}s)"
    echo "RPO: ${METRICS["rpo_seconds"]:-unknown} seconds (target: 0)"
    echo "Data Loss: ${METRICS["data_loss"]:-unknown}"
    echo "Failover Success: ${METRICS["failover_success"]:-unknown}"
    if [ "$NO_ROLLBACK" != "true" ]; then
        echo "Rollback Success: ${METRICS["rollback_success"]:-unknown}"
    fi
    echo "Report: $REPORT_FILE"
    echo "Log: $LOG_FILE"
    echo "=========================================="

    return 0
}

#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
main() {
    ensure_directories

    log_info "=========================================="
    log_info "PostgreSQL/TimescaleDB Failover Test"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Cluster: $CLUSTER_NAME"
    log_info "=========================================="

    send_notification "info" "Starting failover test for cluster $CLUSTER_NAME"

    local test_failed=false

    # Step 1: Validate cluster health
    if ! validate_cluster_health; then
        log_error "Cluster health validation failed - aborting test"
        send_notification "error" "Failover test aborted: cluster health check failed"
        exit 1
    fi

    # Step 2: Execute failover test
    if ! execute_failover_test; then
        log_error "Failover test failed"
        test_failed=true
    fi

    # Step 3: Execute rollback (unless disabled)
    if [ "$NO_ROLLBACK" != "true" ] && [ "$test_failed" != "true" ]; then
        if ! execute_rollback; then
            log_error "Rollback failed"
        fi
    fi

    # Step 4: Cleanup test data
    cleanup_test_data

    # Step 5: Generate report
    generate_report

    # Step 6: Send final notification
    local test_result="${METRICS["test_result"]:-unknown}"
    if [ "$test_result" == "passed" ]; then
        send_notification "success" "Failover test PASSED. RTO: ${METRICS["rto_seconds"]:-unknown}s, RPO: ${METRICS["rpo_seconds"]:-unknown}s"
    elif [ "$test_result" == "warning" ]; then
        send_notification "warning" "Failover test completed with WARNINGS. RTO: ${METRICS["rto_seconds"]:-unknown}s exceeded threshold"
    else
        send_notification "error" "Failover test FAILED. Check report: $REPORT_FILE"
    fi

    # Exit with appropriate code
    if [ "$test_result" == "passed" ]; then
        exit 0
    else
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# Argument Parsing
#-------------------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Automated failover test for PostgreSQL/TimescaleDB

OPTIONS:
    --no-rollback             Skip rollback after test
    --report-only             Generate report from last test (no new test)
    --output <file>           Custom report output file
    --rto-threshold <seconds> RTO threshold in seconds (default: 300)
    -h, --help                Show this help message

ENVIRONMENT VARIABLES:
    PATRONI_CONFIG            Path to Patroni config
    CLUSTER_NAME              Cluster name
    DB_USER                   Database user
    DB_NAME                   Database name
    SLACK_WEBHOOK_URL         Slack webhook for notifications
    LOG_DIR                   Log directory

EXAMPLES:
    # Run full test with rollback
    $(basename "$0")

    # Run test without rollback
    $(basename "$0") --no-rollback

    # Custom RTO threshold
    $(basename "$0") --rto-threshold 60

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-rollback)
            NO_ROLLBACK=true
            shift
            ;;
        --report-only)
            REPORT_ONLY=true
            shift
            ;;
        --output)
            REPORT_FILE="$2"
            shift 2
            ;;
        --rto-threshold)
            TEST_DURATION_THRESHOLD_SECONDS="$2"
            shift 2
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
