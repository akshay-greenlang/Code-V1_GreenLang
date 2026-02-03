#!/bin/bash
# =============================================================================
# pgBackRest Point-in-Time Recovery Script
# GreenLang Database Infrastructure
# =============================================================================
# This script performs point-in-time recovery to restore the PostgreSQL
# database to a specific timestamp. Designed for disaster recovery scenarios.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STANZA="${STANZA:-greenlang}"
RESTORE_TARGET_TIME="${RESTORE_TARGET_TIME:-}"
RESTORE_TARGET_ACTION="${RESTORE_TARGET_ACTION:-promote}"
RESTORE_TARGET_TYPE="${RESTORE_TARGET_TYPE:-time}"
RESTORE_DIR="${RESTORE_DIR:-/var/lib/postgresql/data-restored}"
PG_VERSION="${PG_VERSION:-15}"
LOG_FILE="${LOG_FILE:-/var/log/pgbackrest/restore-pitr.log}"
PROCESS_MAX="${PROCESS_MAX:-4}"

# Validation settings
VALIDATE_RESTORE="${VALIDATE_RESTORE:-true}"
VALIDATION_PORT="${VALIDATION_PORT:-5433}"

# Notification settings
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"

# Exit codes
EXIT_SUCCESS=0
EXIT_INVALID_PARAMS=1
EXIT_NO_BACKUP=2
EXIT_RESTORE_FAILED=3
EXIT_VALIDATION_FAILED=4
EXIT_CONFIG_ERROR=5

# -----------------------------------------------------------------------------
# Logging Functions
# -----------------------------------------------------------------------------
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date -Iseconds)
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
}

log_warn() {
    log "WARN" "$@"
}

log_error() {
    log "ERROR" "$@"
}

# -----------------------------------------------------------------------------
# Notification Functions
# -----------------------------------------------------------------------------
send_notification() {
    local status="$1"
    local message="$2"
    local details="${3:-}"

    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local color
        case "$status" in
            success) color="good" ;;
            warning) color="warning" ;;
            failure) color="danger" ;;
            progress) color="#439FE0" ;;
            *) color="#808080" ;;
        esac

        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"pgBackRest PITR Restore\",
                    \"text\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Stanza\", \"value\": \"$STANZA\", \"short\": true},
                        {\"title\": \"Target Time\", \"value\": \"$RESTORE_TARGET_TIME\", \"short\": true},
                        {\"title\": \"Details\", \"value\": \"$details\", \"short\": false}
                    ],
                    \"ts\": $(date +%s)
                }]
            }" 2>/dev/null || log_warn "Failed to send Slack notification"
    fi
}

# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------
validate_parameters() {
    log_info "Validating restore parameters..."

    # Check required parameters
    if [ -z "$RESTORE_TARGET_TIME" ]; then
        log_error "RESTORE_TARGET_TIME is required"
        echo ""
        echo "Usage: $0"
        echo "  Required environment variables:"
        echo "    RESTORE_TARGET_TIME - Target recovery timestamp (e.g., '2024-01-15 14:30:00+00')"
        echo ""
        echo "  Optional environment variables:"
        echo "    STANZA              - pgBackRest stanza name (default: greenlang)"
        echo "    RESTORE_DIR         - Directory for restored data (default: /var/lib/postgresql/data-restored)"
        echo "    RESTORE_TARGET_ACTION - Recovery action: promote, pause, shutdown (default: promote)"
        echo "    VALIDATE_RESTORE    - Validate restore after completion (default: true)"
        echo ""
        return $EXIT_INVALID_PARAMS
    fi

    # Validate timestamp format
    if ! date -d "$RESTORE_TARGET_TIME" > /dev/null 2>&1; then
        log_error "Invalid timestamp format: $RESTORE_TARGET_TIME"
        log_error "Expected format: YYYY-MM-DD HH:MM:SS+TZ (e.g., 2024-01-15 14:30:00+00)"
        return $EXIT_INVALID_PARAMS
    fi

    # Check if target time is in the future
    local target_epoch current_epoch
    target_epoch=$(date -d "$RESTORE_TARGET_TIME" +%s)
    current_epoch=$(date +%s)

    if [ "$target_epoch" -gt "$current_epoch" ]; then
        log_error "Target time cannot be in the future"
        return $EXIT_INVALID_PARAMS
    fi

    # Validate target action
    case "$RESTORE_TARGET_ACTION" in
        promote|pause|shutdown)
            log_info "Target action: $RESTORE_TARGET_ACTION"
            ;;
        *)
            log_error "Invalid target action: $RESTORE_TARGET_ACTION"
            log_error "Valid options: promote, pause, shutdown"
            return $EXIT_INVALID_PARAMS
            ;;
    esac

    log_info "Parameters validated successfully"
    return $EXIT_SUCCESS
}

check_backup_availability() {
    log_info "Checking backup availability..."

    # Get backup info
    local backup_info
    if ! backup_info=$(pgbackrest --stanza="$STANZA" info --output=json 2>&1); then
        log_error "Failed to get backup info: $backup_info"
        return $EXIT_NO_BACKUP
    fi

    # Check if any backups exist
    local backup_count
    backup_count=$(echo "$backup_info" | jq '.[0].backup | length')

    if [ "$backup_count" -eq 0 ]; then
        log_error "No backups found for stanza: $STANZA"
        return $EXIT_NO_BACKUP
    fi

    log_info "Found $backup_count backup(s)"

    # Check if target time is within backup range
    local oldest_backup_time
    oldest_backup_time=$(echo "$backup_info" | jq -r '.[0].backup[0].timestamp.start')
    local oldest_epoch
    oldest_epoch=$(date -d "$oldest_backup_time" +%s)
    local target_epoch
    target_epoch=$(date -d "$RESTORE_TARGET_TIME" +%s)

    if [ "$target_epoch" -lt "$oldest_epoch" ]; then
        log_error "Target time ($RESTORE_TARGET_TIME) is before oldest backup ($oldest_backup_time)"
        return $EXIT_NO_BACKUP
    fi

    log_info "Backup availability confirmed"

    # Display relevant backup info
    log_info "Available backups:"
    echo "$backup_info" | jq -r '.[0].backup[] | "  \(.label) (\(.type)) - \(.timestamp.start) to \(.timestamp.stop)"' | while read -r line; do
        log_info "$line"
    done

    return $EXIT_SUCCESS
}

# -----------------------------------------------------------------------------
# Restore Functions
# -----------------------------------------------------------------------------
prepare_restore_directory() {
    log_info "Preparing restore directory: $RESTORE_DIR"

    # Check if directory exists
    if [ -d "$RESTORE_DIR" ]; then
        if [ "$(ls -A "$RESTORE_DIR" 2>/dev/null)" ]; then
            log_warn "Restore directory is not empty"

            if [ "${FORCE_OVERWRITE:-false}" == "true" ]; then
                log_warn "FORCE_OVERWRITE is set, clearing directory..."
                rm -rf "${RESTORE_DIR:?}/"*
            else
                log_error "Directory not empty and FORCE_OVERWRITE is not set"
                log_error "Set FORCE_OVERWRITE=true to overwrite existing data"
                return $EXIT_CONFIG_ERROR
            fi
        fi
    else
        log_info "Creating restore directory..."
        mkdir -p "$RESTORE_DIR"
    fi

    # Set permissions
    chmod 700 "$RESTORE_DIR"

    log_info "Restore directory prepared"
    return $EXIT_SUCCESS
}

perform_restore() {
    log_info "=========================================="
    log_info "STARTING POINT-IN-TIME RECOVERY"
    log_info "=========================================="
    log_info "Stanza: $STANZA"
    log_info "Target Time: $RESTORE_TARGET_TIME"
    log_info "Target Action: $RESTORE_TARGET_ACTION"
    log_info "Restore Directory: $RESTORE_DIR"
    log_info "=========================================="

    send_notification "progress" "Starting PITR restore" "Target: $RESTORE_TARGET_TIME"

    local restore_start
    restore_start=$(date +%s)

    # Build restore command
    local restore_cmd="pgbackrest --stanza=$STANZA restore"
    restore_cmd+=" --target=\"$RESTORE_TARGET_TIME\""
    restore_cmd+=" --target-action=$RESTORE_TARGET_ACTION"
    restore_cmd+=" --type=$RESTORE_TARGET_TYPE"
    restore_cmd+=" --pg1-path=$RESTORE_DIR"
    restore_cmd+=" --delta"
    restore_cmd+=" --process-max=$PROCESS_MAX"
    restore_cmd+=" --log-level-console=info"
    restore_cmd+=" --log-level-file=detail"

    log_info "Executing restore command..."
    log_info "$restore_cmd"

    # Execute restore
    if ! eval "$restore_cmd" 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Restore command failed"
        send_notification "failure" "PITR restore failed" "Restore command returned error"
        return $EXIT_RESTORE_FAILED
    fi

    local restore_end restore_duration
    restore_end=$(date +%s)
    restore_duration=$((restore_end - restore_start))

    log_info "Restore completed in ${restore_duration}s"

    # Configure recovery settings
    configure_recovery

    return $EXIT_SUCCESS
}

configure_recovery() {
    log_info "Configuring recovery settings..."

    # Create or update postgresql.auto.conf with recovery settings
    cat >> "$RESTORE_DIR/postgresql.auto.conf" << EOF

# =============================================================================
# Recovery configuration added by pgBackRest PITR restore
# Restore timestamp: $(date -Iseconds)
# Target time: $RESTORE_TARGET_TIME
# =============================================================================
restore_command = 'pgbackrest --stanza=$STANZA archive-get %f "%p"'
recovery_target_time = '$RESTORE_TARGET_TIME'
recovery_target_action = '$RESTORE_TARGET_ACTION'
EOF

    # Create recovery signal file
    touch "$RESTORE_DIR/recovery.signal"

    # Set proper permissions
    chmod 600 "$RESTORE_DIR"/*.conf 2>/dev/null || true
    chmod 600 "$RESTORE_DIR/recovery.signal"

    log_info "Recovery configuration completed"
}

# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------
validate_restore() {
    if [ "$VALIDATE_RESTORE" != "true" ]; then
        log_info "Restore validation disabled, skipping..."
        return $EXIT_SUCCESS
    fi

    log_info "=========================================="
    log_info "VALIDATING RESTORED DATABASE"
    log_info "=========================================="

    # Check for required PostgreSQL files
    local required_files=("PG_VERSION" "postgresql.auto.conf" "recovery.signal")

    for file in "${required_files[@]}"; do
        if [ ! -f "$RESTORE_DIR/$file" ]; then
            log_error "Required file missing: $file"
            return $EXIT_VALIDATION_FAILED
        fi
    done

    log_info "Required files present"

    # Verify PostgreSQL version
    local pg_version
    pg_version=$(cat "$RESTORE_DIR/PG_VERSION")
    log_info "PostgreSQL version: $pg_version"

    # Optionally start PostgreSQL for validation
    if command -v pg_ctl > /dev/null 2>&1; then
        log_info "Starting PostgreSQL for validation..."

        # Create temporary configuration for validation
        local temp_conf="$RESTORE_DIR/postgresql.conf.validate"
        cat > "$temp_conf" << EOF
listen_addresses = 'localhost'
port = $VALIDATION_PORT
max_connections = 10
shared_buffers = 128MB
logging_collector = off
EOF

        # Attempt to start PostgreSQL
        if pg_ctl -D "$RESTORE_DIR" -o "-c config_file=$temp_conf" -l "$RESTORE_DIR/validation.log" start; then
            log_info "PostgreSQL started successfully"

            # Wait for startup
            sleep 5

            # Check if PostgreSQL is ready
            local max_attempts=30
            local attempt=0

            while [ $attempt -lt $max_attempts ]; do
                if pg_isready -h localhost -p "$VALIDATION_PORT" > /dev/null 2>&1; then
                    log_info "PostgreSQL is ready"
                    break
                fi
                attempt=$((attempt + 1))
                sleep 1
            done

            if [ $attempt -ge $max_attempts ]; then
                log_warn "PostgreSQL did not become ready within timeout"
            else
                # Run validation queries
                log_info "Running validation queries..."

                # Check recovery status
                local recovery_status
                recovery_status=$(psql -h localhost -p "$VALIDATION_PORT" -U postgres -t -c "SELECT pg_is_in_recovery();" 2>/dev/null || echo "unknown")
                log_info "Recovery status: $recovery_status"

                # Get database size
                local db_size
                db_size=$(psql -h localhost -p "$VALIDATION_PORT" -U postgres -t -c "SELECT pg_size_pretty(pg_database_size('postgres'));" 2>/dev/null || echo "unknown")
                log_info "Database size: $db_size"

                # List databases
                log_info "Available databases:"
                psql -h localhost -p "$VALIDATION_PORT" -U postgres -c "\l" 2>/dev/null | tee -a "$LOG_FILE" || true
            fi

            # Stop PostgreSQL
            log_info "Stopping PostgreSQL..."
            pg_ctl -D "$RESTORE_DIR" stop -m fast

            # Remove temporary config
            rm -f "$temp_conf"
        else
            log_warn "Failed to start PostgreSQL for validation"
            log_warn "Check $RESTORE_DIR/validation.log for details"
        fi
    else
        log_warn "pg_ctl not found, skipping PostgreSQL startup validation"
    fi

    log_info "Validation completed"
    return $EXIT_SUCCESS
}

# -----------------------------------------------------------------------------
# Reporting Functions
# -----------------------------------------------------------------------------
generate_report() {
    log_info "=========================================="
    log_info "POINT-IN-TIME RECOVERY REPORT"
    log_info "=========================================="

    echo ""
    echo "Recovery Summary:"
    echo "  Stanza:          $STANZA"
    echo "  Target Time:     $RESTORE_TARGET_TIME"
    echo "  Target Action:   $RESTORE_TARGET_ACTION"
    echo "  Restore Dir:     $RESTORE_DIR"
    echo ""
    echo "Next Steps:"
    echo "  1. Stop the current PostgreSQL instance (if running)"
    echo "  2. Backup the current data directory"
    echo "  3. Move restored data to the PostgreSQL data directory:"
    echo "     mv $RESTORE_DIR/* /var/lib/postgresql/data/"
    echo "  4. Start PostgreSQL"
    echo "  5. Monitor recovery progress in PostgreSQL logs"
    echo "  6. After recovery completes, verify data integrity"
    echo ""
    echo "To promote immediately after recovery:"
    echo "  pg_ctl -D /var/lib/postgresql/data promote"
    echo ""

    # Display restored files
    log_info "Restored files:"
    ls -la "$RESTORE_DIR" | head -20

    # Display disk usage
    local restore_size
    restore_size=$(du -sh "$RESTORE_DIR" 2>/dev/null | cut -f1)
    log_info "Total restore size: $restore_size"
}

# -----------------------------------------------------------------------------
# Cleanup Functions
# -----------------------------------------------------------------------------
cleanup() {
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        log_error "Restore process failed with exit code: $exit_code"
        send_notification "failure" "PITR restore failed" "Exit code: $exit_code"
    fi

    # Remove any temporary files
    rm -f /tmp/backup-info.json 2>/dev/null || true
    rm -f /tmp/wal-list.txt 2>/dev/null || true
}

trap cleanup EXIT

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
main() {
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"

    log_info "=========================================="
    log_info "pgBackRest Point-in-Time Recovery"
    log_info "Script started at: $(date -Iseconds)"
    log_info "=========================================="

    # Step 1: Validate parameters
    if ! validate_parameters; then
        return $EXIT_INVALID_PARAMS
    fi

    # Step 2: Check backup availability
    if ! check_backup_availability; then
        return $EXIT_NO_BACKUP
    fi

    # Step 3: Prepare restore directory
    if ! prepare_restore_directory; then
        return $EXIT_CONFIG_ERROR
    fi

    # Step 4: Perform restore
    if ! perform_restore; then
        return $EXIT_RESTORE_FAILED
    fi

    # Step 5: Validate restore
    if ! validate_restore; then
        return $EXIT_VALIDATION_FAILED
    fi

    # Step 6: Generate report
    generate_report

    log_info "=========================================="
    log_info "POINT-IN-TIME RECOVERY COMPLETED SUCCESSFULLY"
    log_info "=========================================="

    send_notification "success" "PITR restore completed successfully" "Database restored to $RESTORE_TARGET_TIME"

    return $EXIT_SUCCESS
}

# Run main function
main "$@"
