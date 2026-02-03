#!/bin/bash
#===============================================================================
# restore-pitr.sh
#
# Point-in-Time Recovery (PITR) script for PostgreSQL/TimescaleDB
#
# This script performs a complete PITR restore including:
# - Pausing application traffic
# - Restoring from backup to specified timestamp
# - Applying WAL to target time
# - Validating restored data
# - Resuming traffic
#
# Usage: ./restore-pitr.sh --target-time "2026-02-03 14:30:00" [OPTIONS]
#
# Author: GreenLang Database Operations Team
# Version: 1.0.0
# Date: 2026-02-03
#===============================================================================

set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PGBACKREST_STANZA="${PGBACKREST_STANZA:-greenlang}"
PGBACKREST_CONFIG="${PGBACKREST_CONFIG:-/etc/pgbackrest/pgbackrest.conf}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-greenlang}"
NAMESPACE="${NAMESPACE:-database}"
API_DEPLOYMENT="${API_DEPLOYMENT:-greenlang-api}"

# Recovery options
RECOVERY_TYPE="time"  # time, xid, name, immediate
TARGET_TIME=""
TARGET_XID=""
TARGET_NAME=""
RECOVERY_TARGET_INCLUSIVE="true"
RECOVERY_TARGET_ACTION="promote"  # pause, promote, shutdown

# Script options
LOG_DIR="${LOG_DIR:-/var/log/database}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pitr_restore_${TIMESTAMP}.log"
DRY_RUN=false
SKIP_TRAFFIC_PAUSE=false
SKIP_VALIDATION=false
FORCE=false

# Slack notification
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

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
                    \"title\": \"PITR Restore Notification\",
                    \"text\": \"${message}\",
                    \"fields\": [
                        {\"title\": \"Target Time\", \"value\": \"${TARGET_TIME:-N/A}\", \"short\": true},
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
    mkdir -p "/tmp/pitr_restore_${TIMESTAMP}"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf "/tmp/pitr_restore_${TIMESTAMP}"
}

trap cleanup EXIT

validate_target_time() {
    local target="$1"

    # Validate timestamp format
    if ! date -d "$target" &>/dev/null; then
        log_error "Invalid timestamp format: $target"
        log_error "Expected format: YYYY-MM-DD HH:MM:SS"
        return 1
    fi

    # Check if target time is in the past
    local target_epoch
    target_epoch=$(date -d "$target" +%s)
    local now_epoch
    now_epoch=$(date +%s)

    if [ "$target_epoch" -ge "$now_epoch" ]; then
        log_error "Target time must be in the past"
        return 1
    fi

    # Check if target time is within backup retention
    local oldest_backup
    oldest_backup=$(pgbackrest info --stanza="$PGBACKREST_STANZA" --output=json | \
        jq -r '.[0].backup[0].timestamp.start // empty' 2>/dev/null)

    if [ -n "$oldest_backup" ]; then
        local oldest_epoch
        oldest_epoch=$(date -d "$oldest_backup" +%s 2>/dev/null || echo 0)

        if [ "$target_epoch" -lt "$oldest_epoch" ]; then
            log_error "Target time is before oldest available backup"
            log_error "Oldest backup: $oldest_backup"
            return 1
        fi
    fi

    log_success "Target time validated: $target"
    return 0
}

#-------------------------------------------------------------------------------
# Traffic Management
#-------------------------------------------------------------------------------
pause_application_traffic() {
    log_info "Pausing application traffic..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would scale down application deployments"
        return 0
    fi

    # Scale down API deployment
    log_info "Scaling down $API_DEPLOYMENT..."
    kubectl scale deployment "$API_DEPLOYMENT" --replicas=0 -n production 2>/dev/null || \
        log_warn "Could not scale deployment (may not exist or not using Kubernetes)"

    # Wait for pods to terminate
    log_info "Waiting for pods to terminate..."
    kubectl wait --for=delete pod -l app=greenlang-api -n production --timeout=120s 2>/dev/null || true

    # Pause PgBouncer connections (if available)
    if command -v psql &>/dev/null; then
        psql -h pgbouncer -U admin -p 6432 pgbouncer -c "PAUSE greenlang;" 2>/dev/null || \
            log_warn "Could not pause PgBouncer (may not be configured)"
    fi

    log_success "Application traffic paused"
}

resume_application_traffic() {
    log_info "Resuming application traffic..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would scale up application deployments"
        return 0
    fi

    # Resume PgBouncer connections (if available)
    psql -h pgbouncer -U admin -p 6432 pgbouncer -c "RESUME greenlang;" 2>/dev/null || true

    # Scale up API deployment
    log_info "Scaling up $API_DEPLOYMENT..."
    kubectl scale deployment "$API_DEPLOYMENT" --replicas=3 -n production 2>/dev/null || \
        log_warn "Could not scale deployment"

    # Wait for pods to be ready
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=greenlang-api -n production --timeout=300s 2>/dev/null || true

    log_success "Application traffic resumed"
}

#-------------------------------------------------------------------------------
# Backup Verification
#-------------------------------------------------------------------------------
verify_backup_availability() {
    log_info "Verifying backup availability..."

    # List available backups
    log_info "Available backups:"
    pgbackrest info --stanza="$PGBACKREST_STANZA" | tee -a "$LOG_FILE"

    # Verify backup integrity
    log_info "Verifying backup integrity..."
    if pgbackrest verify --stanza="$PGBACKREST_STANZA" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Backup verification passed"
    else
        log_error "Backup verification failed"
        return 1
    fi

    # Check WAL archive coverage
    log_info "Checking WAL archive coverage for target time..."
    local wal_check
    wal_check=$(pgbackrest info --stanza="$PGBACKREST_STANZA" --output=json | \
        jq -r '.[0].archive[] | select(.min != null) | "min: \(.min), max: \(.max)"' 2>/dev/null)

    if [ -n "$wal_check" ]; then
        log_info "WAL archive range: $wal_check"
    else
        log_warn "Could not determine WAL archive range"
    fi

    return 0
}

#-------------------------------------------------------------------------------
# Pre-Restore Backup
#-------------------------------------------------------------------------------
create_pre_restore_backup() {
    log_info "Creating pre-restore backup of current state..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would create pg_dump backup"
        return 0
    fi

    local backup_file="/tmp/pitr_restore_${TIMESTAMP}/pre_restore_backup.sql"

    pg_dump -h localhost -U "$DB_USER" -d "$DB_NAME" \
        --format=custom \
        --file="$backup_file" 2>&1 | tee -a "$LOG_FILE" || {
        log_warn "Could not create pre-restore backup (database may be down)"
    }

    if [ -f "$backup_file" ]; then
        log_success "Pre-restore backup created: $backup_file"
    fi
}

#-------------------------------------------------------------------------------
# Stop PostgreSQL
#-------------------------------------------------------------------------------
stop_postgresql() {
    log_info "Stopping PostgreSQL..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would stop PostgreSQL"
        return 0
    fi

    # Try systemctl first
    if systemctl stop postgresql 2>/dev/null; then
        log_success "PostgreSQL stopped via systemctl"
        return 0
    fi

    # Try pg_ctl
    if sudo -u postgres pg_ctl stop -D "$PGDATA" -m fast 2>/dev/null; then
        log_success "PostgreSQL stopped via pg_ctl"
        return 0
    fi

    # Try Kubernetes
    if kubectl scale statefulset greenlang-db --replicas=0 -n "$NAMESPACE" 2>/dev/null; then
        log_success "PostgreSQL stopped via Kubernetes"
        sleep 10
        return 0
    fi

    log_error "Could not stop PostgreSQL"
    return 1
}

#-------------------------------------------------------------------------------
# Restore from Backup
#-------------------------------------------------------------------------------
perform_restore() {
    log_info "Starting PITR restore..."
    log_info "Recovery type: $RECOVERY_TYPE"
    log_info "Target: ${TARGET_TIME:-${TARGET_XID:-${TARGET_NAME:-immediate}}}"

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would perform pgbackrest restore"
        return 0
    fi

    # Clear data directory
    log_info "Clearing data directory..."
    rm -rf "${PGDATA:?}/"*

    # Build restore command
    local restore_cmd="pgbackrest restore --stanza=$PGBACKREST_STANZA"
    restore_cmd+=" --config=$PGBACKREST_CONFIG"
    restore_cmd+=" --log-level-console=info"
    restore_cmd+=" --delta"

    # Add recovery target options
    case "$RECOVERY_TYPE" in
        "time")
            restore_cmd+=" --type=time"
            restore_cmd+=" --target=\"$TARGET_TIME\""
            ;;
        "xid")
            restore_cmd+=" --type=xid"
            restore_cmd+=" --target=$TARGET_XID"
            ;;
        "name")
            restore_cmd+=" --type=name"
            restore_cmd+=" --target=$TARGET_NAME"
            ;;
        "immediate")
            restore_cmd+=" --type=immediate"
            ;;
    esac

    restore_cmd+=" --target-action=$RECOVERY_TARGET_ACTION"

    if [ "$RECOVERY_TARGET_INCLUSIVE" == "true" ]; then
        restore_cmd+=" --target-inclusive"
    fi

    # Execute restore
    log_info "Executing: $restore_cmd"
    eval "$restore_cmd" 2>&1 | tee -a "$LOG_FILE"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log_error "Restore command failed"
        return 1
    fi

    log_success "Restore completed successfully"
    return 0
}

#-------------------------------------------------------------------------------
# Start PostgreSQL
#-------------------------------------------------------------------------------
start_postgresql() {
    log_info "Starting PostgreSQL..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would start PostgreSQL"
        return 0
    fi

    # Fix ownership
    chown -R postgres:postgres "$PGDATA"

    # Try systemctl first
    if systemctl start postgresql 2>/dev/null; then
        log_success "PostgreSQL started via systemctl"
        return 0
    fi

    # Try pg_ctl
    if sudo -u postgres pg_ctl start -D "$PGDATA" -l "$LOG_DIR/postgresql_recovery.log" 2>/dev/null; then
        log_success "PostgreSQL started via pg_ctl"
        return 0
    fi

    # Try Kubernetes
    if kubectl scale statefulset greenlang-db --replicas=1 -n "$NAMESPACE" 2>/dev/null; then
        log_success "PostgreSQL started via Kubernetes"
        return 0
    fi

    log_error "Could not start PostgreSQL"
    return 1
}

#-------------------------------------------------------------------------------
# Wait for Recovery
#-------------------------------------------------------------------------------
wait_for_recovery() {
    log_info "Waiting for recovery to complete..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would wait for recovery"
        return 0
    fi

    local timeout=600
    local elapsed=0
    local interval=5

    while [ $elapsed -lt $timeout ]; do
        # Check if PostgreSQL is responding
        if ! pg_isready -h localhost -U "$DB_USER" &>/dev/null; then
            log_info "Waiting for PostgreSQL to accept connections... (${elapsed}s/${timeout}s)"
            sleep $interval
            elapsed=$((elapsed + interval))
            continue
        fi

        # Check recovery status
        local is_in_recovery
        is_in_recovery=$(psql -h localhost -U "$DB_USER" -t -c "SELECT pg_is_in_recovery();" 2>/dev/null | tr -d ' ')

        if [ "$is_in_recovery" == "f" ]; then
            log_success "Recovery completed - database is accepting writes"
            return 0
        fi

        # Check recovery progress
        local replay_lsn
        replay_lsn=$(psql -h localhost -U "$DB_USER" -t -c "SELECT pg_last_wal_replay_lsn();" 2>/dev/null | tr -d ' ')

        local replay_time
        replay_time=$(psql -h localhost -U "$DB_USER" -t -c "SELECT pg_last_xact_replay_timestamp();" 2>/dev/null | tr -d ' ')

        log_info "Recovery in progress: LSN=$replay_lsn, Time=$replay_time (${elapsed}s/${timeout}s)"

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log_error "Recovery timeout exceeded"
    return 1
}

#-------------------------------------------------------------------------------
# Data Validation
#-------------------------------------------------------------------------------
validate_restored_data() {
    log_info "Validating restored data..."

    if [ "$DRY_RUN" == "true" ]; then
        log_info "[DRY RUN] Would validate restored data"
        return 0
    fi

    if [ "$SKIP_VALIDATION" == "true" ]; then
        log_warn "Skipping data validation (--skip-validation)"
        return 0
    fi

    local validation_errors=0

    # Check database exists
    log_info "Checking database exists..."
    if ! psql -h localhost -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &>/dev/null; then
        log_error "Database $DB_NAME does not exist or is not accessible"
        validation_errors=$((validation_errors + 1))
    fi

    # Check table counts
    log_info "Checking table counts..."
    local table_counts
    table_counts=$(psql -h localhost -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT schemaname, tablename,
               pg_relation_size(schemaname || '.' || tablename) as size
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY size DESC
        LIMIT 10;
    " 2>/dev/null)

    log_info "Top tables by size:"
    echo "$table_counts" | tee -a "$LOG_FILE"

    # Check TimescaleDB hypertables
    log_info "Checking TimescaleDB hypertables..."
    local hypertables
    hypertables=$(psql -h localhost -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT hypertable_name, num_chunks
        FROM timescaledb_information.hypertables;
    " 2>/dev/null || echo "TimescaleDB not installed")

    log_info "Hypertables:"
    echo "$hypertables" | tee -a "$LOG_FILE"

    # Check for data consistency
    log_info "Checking data consistency..."
    local latest_data_time
    latest_data_time=$(psql -h localhost -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT max(created_at) FROM (
            SELECT max(created_at) as created_at FROM users
            UNION ALL
            SELECT max(time) FROM energy_readings
        ) t;
    " 2>/dev/null | tr -d ' ')

    if [ -n "$latest_data_time" ]; then
        log_info "Latest data timestamp: $latest_data_time"

        # Compare with target time
        if [ -n "$TARGET_TIME" ]; then
            local target_epoch
            target_epoch=$(date -d "$TARGET_TIME" +%s)
            local data_epoch
            data_epoch=$(date -d "$latest_data_time" +%s 2>/dev/null || echo 0)

            if [ "$data_epoch" -gt "$target_epoch" ]; then
                log_warn "Latest data is after target time - may indicate issue"
            else
                log_success "Data timestamps are consistent with target time"
            fi
        fi
    fi

    # Check indexes
    log_info "Checking index health..."
    local invalid_indexes
    invalid_indexes=$(psql -h localhost -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT count(*) FROM pg_index WHERE NOT indisvalid;
    " 2>/dev/null | tr -d ' ')

    if [ "${invalid_indexes:-0}" -gt 0 ]; then
        log_warn "Found $invalid_indexes invalid indexes"
        validation_errors=$((validation_errors + 1))
    else
        log_success "All indexes are valid"
    fi

    # Check sequences
    log_info "Checking sequence values..."
    psql -h localhost -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT schemaname, sequencename, last_value
        FROM pg_sequences
        WHERE schemaname = 'public'
        ORDER BY sequencename;
    " 2>/dev/null | tee -a "$LOG_FILE"

    if [ $validation_errors -gt 0 ]; then
        log_warn "Validation completed with $validation_errors warnings"
        return 1
    fi

    log_success "Data validation completed successfully"
    return 0
}

#-------------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------------
main() {
    local start_time=$(date +%s)

    ensure_directories

    log_info "=========================================="
    log_info "Starting Point-in-Time Recovery"
    log_info "Timestamp: $TIMESTAMP"
    log_info "Target time: ${TARGET_TIME:-N/A}"
    log_info "Recovery type: $RECOVERY_TYPE"
    log_info "Dry run: $DRY_RUN"
    log_info "=========================================="

    send_notification "info" "Starting PITR to ${TARGET_TIME:-immediate}"

    # Step 1: Validate target time
    if [ -n "$TARGET_TIME" ]; then
        if ! validate_target_time "$TARGET_TIME"; then
            send_notification "error" "Invalid target time: $TARGET_TIME"
            exit 1
        fi
    fi

    # Step 2: Verify backup availability
    if ! verify_backup_availability; then
        send_notification "error" "Backup verification failed"
        exit 1
    fi

    # Step 3: Create pre-restore backup
    create_pre_restore_backup

    # Step 4: Pause application traffic
    if [ "$SKIP_TRAFFIC_PAUSE" != "true" ]; then
        pause_application_traffic
    fi

    # Step 5: Stop PostgreSQL
    if ! stop_postgresql; then
        send_notification "error" "Failed to stop PostgreSQL"
        exit 1
    fi

    # Step 6: Perform restore
    if ! perform_restore; then
        send_notification "error" "Restore failed"
        # Try to start PostgreSQL for recovery
        start_postgresql || true
        exit 1
    fi

    # Step 7: Start PostgreSQL
    if ! start_postgresql; then
        send_notification "error" "Failed to start PostgreSQL after restore"
        exit 1
    fi

    # Step 8: Wait for recovery
    if ! wait_for_recovery; then
        send_notification "error" "Recovery timeout"
        exit 1
    fi

    # Step 9: Validate data
    validate_restored_data || true

    # Step 10: Resume traffic
    if [ "$SKIP_TRAFFIC_PAUSE" != "true" ]; then
        resume_application_traffic
    fi

    # Calculate duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "=========================================="
    log_success "PITR completed successfully"
    log_success "Target time: ${TARGET_TIME:-immediate}"
    log_success "Duration: ${duration} seconds"
    log_success "Log file: $LOG_FILE"
    log_success "=========================================="

    send_notification "success" "PITR completed successfully (Duration: ${duration}s)"
}

#-------------------------------------------------------------------------------
# Argument Parsing
#-------------------------------------------------------------------------------
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Point-in-Time Recovery script for PostgreSQL/TimescaleDB

OPTIONS:
    --target-time <timestamp>     Target time (format: YYYY-MM-DD HH:MM:SS)
    --target-xid <xid>            Target transaction ID
    --target-name <name>          Target restore point name
    --immediate                   Recover to end of backup
    --target-action <action>      Action after recovery: promote|pause|shutdown (default: promote)
    --target-inclusive            Include target in recovery (default: true)
    --skip-traffic-pause          Skip pausing application traffic
    --skip-validation             Skip data validation
    --force                       Force recovery even with warnings
    --dry-run                     Show what would be done
    -h, --help                    Show this help message

ENVIRONMENT VARIABLES:
    PGDATA                        PostgreSQL data directory
    PGBACKREST_STANZA             pgBackRest stanza name
    PGBACKREST_CONFIG             pgBackRest configuration file
    DB_USER                       Database user
    DB_NAME                       Database name
    SLACK_WEBHOOK_URL             Slack webhook for notifications

EXAMPLES:
    # Restore to specific time
    $(basename "$0") --target-time "2026-02-03 14:30:00"

    # Restore to transaction ID
    $(basename "$0") --target-xid 12345678

    # Dry run
    $(basename "$0") --target-time "2026-02-03 14:30:00" --dry-run

    # Skip traffic management (for maintenance window)
    $(basename "$0") --target-time "2026-02-03 14:30:00" --skip-traffic-pause

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --target-time)
            TARGET_TIME="$2"
            RECOVERY_TYPE="time"
            shift 2
            ;;
        --target-xid)
            TARGET_XID="$2"
            RECOVERY_TYPE="xid"
            shift 2
            ;;
        --target-name)
            TARGET_NAME="$2"
            RECOVERY_TYPE="name"
            shift 2
            ;;
        --immediate)
            RECOVERY_TYPE="immediate"
            shift
            ;;
        --target-action)
            RECOVERY_TARGET_ACTION="$2"
            shift 2
            ;;
        --target-inclusive)
            RECOVERY_TARGET_INCLUSIVE="true"
            shift
            ;;
        --skip-traffic-pause)
            SKIP_TRAFFIC_PAUSE=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
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

# Validate required options
if [ "$RECOVERY_TYPE" == "time" ] && [ -z "$TARGET_TIME" ]; then
    echo "Error: --target-time is required for time-based recovery"
    usage
    exit 1
fi

# Run main function
main
