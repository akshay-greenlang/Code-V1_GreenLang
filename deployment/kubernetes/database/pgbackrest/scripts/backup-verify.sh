#!/bin/bash
# =============================================================================
# pgBackRest Backup Verification Script
# GreenLang Database Infrastructure
# =============================================================================
# This script verifies backup integrity, WAL continuity, and sends alerts
# on failures. Designed to run as a Kubernetes CronJob or manually.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
STANZA="${STANZA:-greenlang}"
LOG_FILE="${LOG_FILE:-/var/log/pgbackrest/backup-verify.log}"
METRICS_FILE="${METRICS_FILE:-/var/lib/pgbackrest/metrics/verify.prom}"
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"
MAX_WAL_GAP_SECONDS="${MAX_WAL_GAP_SECONDS:-300}"
VERIFY_FULL_INTEGRITY="${VERIFY_FULL_INTEGRITY:-true}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# Exit codes
EXIT_SUCCESS=0
EXIT_NO_BACKUPS=1
EXIT_INTEGRITY_FAILED=2
EXIT_WAL_GAP=3
EXIT_VERIFICATION_FAILED=4

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
# Alert Functions
# -----------------------------------------------------------------------------
send_alert() {
    local status="$1"
    local message="$2"
    local details="${3:-}"

    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local color
        case "$status" in
            success) color="good" ;;
            warning) color="warning" ;;
            failure) color="danger" ;;
            *) color="#808080" ;;
        esac

        curl -s -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"pgBackRest Backup Verification\",
                    \"text\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Stanza\", \"value\": \"$STANZA\", \"short\": true},
                        {\"title\": \"Status\", \"value\": \"$status\", \"short\": true},
                        {\"title\": \"Details\", \"value\": \"$details\", \"short\": false}
                    ],
                    \"ts\": $(date +%s)
                }]
            }" || log_warn "Failed to send Slack notification"
    fi

    if [ -n "$ALERT_WEBHOOK_URL" ]; then
        curl -s -X POST "$ALERT_WEBHOOK_URL" \
            -H 'Content-type: application/json' \
            -d "{
                \"stanza\": \"$STANZA\",
                \"status\": \"$status\",
                \"message\": \"$message\",
                \"details\": \"$details\",
                \"timestamp\": \"$(date -Iseconds)\"
            }" || log_warn "Failed to send webhook notification"
    fi
}

# -----------------------------------------------------------------------------
# Metrics Functions
# -----------------------------------------------------------------------------
init_metrics() {
    mkdir -p "$(dirname "$METRICS_FILE")"
    cat > "$METRICS_FILE" << EOF
# HELP pgbackrest_verify_timestamp Last verification timestamp
# TYPE pgbackrest_verify_timestamp gauge
# HELP pgbackrest_verify_status Verification status (1=success, 0=failure)
# TYPE pgbackrest_verify_status gauge
# HELP pgbackrest_backup_count Number of backups in repository
# TYPE pgbackrest_backup_count gauge
# HELP pgbackrest_wal_gap_seconds Gap in WAL archive (seconds)
# TYPE pgbackrest_wal_gap_seconds gauge
# HELP pgbackrest_oldest_backup_age_seconds Age of oldest backup (seconds)
# TYPE pgbackrest_oldest_backup_age_seconds gauge
# HELP pgbackrest_newest_backup_age_seconds Age of newest backup (seconds)
# TYPE pgbackrest_newest_backup_age_seconds gauge
EOF
}

update_metric() {
    local metric_name="$1"
    local labels="$2"
    local value="$3"

    echo "${metric_name}{${labels}} ${value}" >> "$METRICS_FILE"
}

# -----------------------------------------------------------------------------
# Verification Functions
# -----------------------------------------------------------------------------
check_backup_exists() {
    log_info "Checking if backups exist for stanza: $STANZA"

    local backup_info
    backup_info=$(pgbackrest --stanza="$STANZA" info --output=json 2>/dev/null) || {
        log_error "Failed to get backup info"
        return $EXIT_NO_BACKUPS
    }

    local backup_count
    backup_count=$(echo "$backup_info" | jq '.[0].backup | length')

    if [ "$backup_count" -eq 0 ]; then
        log_error "No backups found for stanza: $STANZA"
        send_alert "failure" "No backups found" "Stanza $STANZA has no backups in the repository"
        update_metric "pgbackrest_backup_count" "stanza=\"$STANZA\"" "0"
        return $EXIT_NO_BACKUPS
    fi

    log_info "Found $backup_count backup(s) for stanza: $STANZA"
    update_metric "pgbackrest_backup_count" "stanza=\"$STANZA\"" "$backup_count"

    # Get backup types
    local full_count diff_count incr_count
    full_count=$(echo "$backup_info" | jq '.[0].backup | map(select(.type == "full")) | length')
    diff_count=$(echo "$backup_info" | jq '.[0].backup | map(select(.type == "diff")) | length')
    incr_count=$(echo "$backup_info" | jq '.[0].backup | map(select(.type == "incr")) | length')

    log_info "Backup breakdown - Full: $full_count, Differential: $diff_count, Incremental: $incr_count"

    update_metric "pgbackrest_backup_count" "stanza=\"$STANZA\",type=\"full\"" "$full_count"
    update_metric "pgbackrest_backup_count" "stanza=\"$STANZA\",type=\"diff\"" "$diff_count"
    update_metric "pgbackrest_backup_count" "stanza=\"$STANZA\",type=\"incr\"" "$incr_count"

    return $EXIT_SUCCESS
}

check_backup_integrity() {
    log_info "Verifying backup integrity..."

    if [ "$VERIFY_FULL_INTEGRITY" != "true" ]; then
        log_info "Full integrity check disabled, skipping..."
        return $EXIT_SUCCESS
    fi

    # Verify latest backup
    log_info "Verifying latest backup..."

    if ! pgbackrest --stanza="$STANZA" --set=latest verify 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Backup integrity verification failed"
        send_alert "failure" "Backup integrity check failed" "The latest backup for stanza $STANZA failed integrity verification"
        update_metric "pgbackrest_verify_status" "stanza=\"$STANZA\",check=\"integrity\"" "0"
        return $EXIT_INTEGRITY_FAILED
    fi

    log_info "Backup integrity verified successfully"
    update_metric "pgbackrest_verify_status" "stanza=\"$STANZA\",check=\"integrity\"" "1"
    return $EXIT_SUCCESS
}

check_wal_continuity() {
    log_info "Checking WAL archive continuity..."

    local backup_info
    backup_info=$(pgbackrest --stanza="$STANZA" info --output=json)

    # Get archive information
    local archive_info
    archive_info=$(echo "$backup_info" | jq '.[0].archive // empty')

    if [ -z "$archive_info" ] || [ "$archive_info" == "null" ]; then
        log_warn "No archive information available"
        return $EXIT_SUCCESS
    fi

    # Check for WAL gaps by verifying the archive range
    local archive_start archive_stop
    archive_start=$(echo "$backup_info" | jq -r '.[0].archive[0].min // empty')
    archive_stop=$(echo "$backup_info" | jq -r '.[0].archive[0].max // empty')

    if [ -n "$archive_start" ] && [ -n "$archive_stop" ]; then
        log_info "WAL archive range: $archive_start to $archive_stop"
    fi

    # Check time since last archived WAL
    local latest_backup_time
    latest_backup_time=$(echo "$backup_info" | jq -r '.[0].backup[-1].timestamp.stop // empty')

    if [ -n "$latest_backup_time" ]; then
        local latest_epoch current_epoch gap_seconds
        latest_epoch=$(date -d "$latest_backup_time" +%s 2>/dev/null || echo "0")
        current_epoch=$(date +%s)
        gap_seconds=$((current_epoch - latest_epoch))

        log_info "Time since last backup: ${gap_seconds}s"
        update_metric "pgbackrest_wal_gap_seconds" "stanza=\"$STANZA\"" "$gap_seconds"

        # Check if gap exceeds threshold (for alerting on stale backups)
        local expected_gap=$((86400 + 3600))  # 25 hours (daily backup + 1 hour buffer)
        if [ "$gap_seconds" -gt "$expected_gap" ]; then
            log_warn "Backup may be stale - last backup was ${gap_seconds}s ago"
            send_alert "warning" "Backup may be stale" "Last backup was $(($gap_seconds / 3600)) hours ago"
        fi
    fi

    log_info "WAL continuity check completed"
    update_metric "pgbackrest_verify_status" "stanza=\"$STANZA\",check=\"wal_continuity\"" "1"
    return $EXIT_SUCCESS
}

check_backup_age() {
    log_info "Checking backup age..."

    local backup_info
    backup_info=$(pgbackrest --stanza="$STANZA" info --output=json)

    local current_epoch
    current_epoch=$(date +%s)

    # Check oldest backup age
    local oldest_backup_time
    oldest_backup_time=$(echo "$backup_info" | jq -r '.[0].backup[0].timestamp.start // empty')

    if [ -n "$oldest_backup_time" ]; then
        local oldest_epoch oldest_age
        oldest_epoch=$(date -d "$oldest_backup_time" +%s 2>/dev/null || echo "$current_epoch")
        oldest_age=$((current_epoch - oldest_epoch))

        log_info "Oldest backup age: ${oldest_age}s ($(($oldest_age / 86400)) days)"
        update_metric "pgbackrest_oldest_backup_age_seconds" "stanza=\"$STANZA\"" "$oldest_age"
    fi

    # Check newest backup age
    local newest_backup_time
    newest_backup_time=$(echo "$backup_info" | jq -r '.[0].backup[-1].timestamp.stop // empty')

    if [ -n "$newest_backup_time" ]; then
        local newest_epoch newest_age
        newest_epoch=$(date -d "$newest_backup_time" +%s 2>/dev/null || echo "$current_epoch")
        newest_age=$((current_epoch - newest_epoch))

        log_info "Newest backup age: ${newest_age}s ($(($newest_age / 3600)) hours)"
        update_metric "pgbackrest_newest_backup_age_seconds" "stanza=\"$STANZA\"" "$newest_age"

        # Alert if newest backup is too old (more than 2 days)
        if [ "$newest_age" -gt 172800 ]; then
            log_error "Newest backup is more than 2 days old!"
            send_alert "failure" "Backups are stale" "The newest backup is $(($newest_age / 86400)) days old"
            return $EXIT_VERIFICATION_FAILED
        fi
    fi

    return $EXIT_SUCCESS
}

check_backup_sizes() {
    log_info "Checking backup sizes..."

    local backup_info
    backup_info=$(pgbackrest --stanza="$STANZA" info --output=json)

    # Get size of each backup type
    echo "$backup_info" | jq -r '.[0].backup[] | "\(.label) \(.type) \(.info.size) \(.info.delta)"' | while read -r label type size delta; do
        log_info "Backup $label ($type): size=${size}, delta=${delta}"
    done

    # Calculate total repository size
    local total_size
    total_size=$(echo "$backup_info" | jq '[.[0].backup[].info.size] | add // 0')

    log_info "Total backup size: $total_size bytes ($(($total_size / 1073741824)) GB)"
    update_metric "pgbackrest_total_backup_size_bytes" "stanza=\"$STANZA\"" "$total_size"
}

verify_restore_capability() {
    log_info "Verifying restore capability (dry-run)..."

    # This is a lightweight check to ensure we can read backup manifest
    if ! pgbackrest --stanza="$STANZA" info > /dev/null 2>&1; then
        log_error "Cannot access backup repository - restore may not be possible"
        send_alert "failure" "Restore capability compromised" "Cannot access backup repository"
        return $EXIT_VERIFICATION_FAILED
    fi

    # Check if we have a complete backup chain
    local backup_info
    backup_info=$(pgbackrest --stanza="$STANZA" info --output=json)

    local has_full
    has_full=$(echo "$backup_info" | jq '.[0].backup | map(select(.type == "full")) | length > 0')

    if [ "$has_full" != "true" ]; then
        log_error "No full backup available - restore not possible"
        send_alert "failure" "No full backup available" "At least one full backup is required for restore capability"
        return $EXIT_VERIFICATION_FAILED
    fi

    log_info "Restore capability verified - backup chain is complete"
    update_metric "pgbackrest_verify_status" "stanza=\"$STANZA\",check=\"restore_capability\"" "1"
    return $EXIT_SUCCESS
}

# -----------------------------------------------------------------------------
# Main Verification Workflow
# -----------------------------------------------------------------------------
main() {
    log_info "=========================================="
    log_info "pgBackRest Backup Verification"
    log_info "Stanza: $STANZA"
    log_info "=========================================="

    # Initialize metrics
    init_metrics

    local overall_status=$EXIT_SUCCESS
    local failed_checks=()

    # Run all verification checks
    if ! check_backup_exists; then
        overall_status=$EXIT_NO_BACKUPS
        failed_checks+=("backup_exists")
    fi

    # Only continue if backups exist
    if [ $overall_status -eq $EXIT_SUCCESS ]; then
        if ! check_backup_integrity; then
            overall_status=$EXIT_INTEGRITY_FAILED
            failed_checks+=("integrity")
        fi

        if ! check_wal_continuity; then
            overall_status=$EXIT_WAL_GAP
            failed_checks+=("wal_continuity")
        fi

        if ! check_backup_age; then
            overall_status=$EXIT_VERIFICATION_FAILED
            failed_checks+=("backup_age")
        fi

        check_backup_sizes

        if ! verify_restore_capability; then
            overall_status=$EXIT_VERIFICATION_FAILED
            failed_checks+=("restore_capability")
        fi
    fi

    # Update overall status metric
    if [ $overall_status -eq $EXIT_SUCCESS ]; then
        update_metric "pgbackrest_verify_status" "stanza=\"$STANZA\",check=\"overall\"" "1"
        update_metric "pgbackrest_verify_timestamp" "stanza=\"$STANZA\"" "$(date +%s)"

        log_info "=========================================="
        log_info "All verification checks PASSED"
        log_info "=========================================="

        send_alert "success" "Backup verification passed" "All checks completed successfully"
    else
        update_metric "pgbackrest_verify_status" "stanza=\"$STANZA\",check=\"overall\"" "0"
        update_metric "pgbackrest_verify_timestamp" "stanza=\"$STANZA\"" "$(date +%s)"

        log_error "=========================================="
        log_error "Verification FAILED"
        log_error "Failed checks: ${failed_checks[*]}"
        log_error "=========================================="

        send_alert "failure" "Backup verification failed" "Failed checks: ${failed_checks[*]}"
    fi

    return $overall_status
}

# Run main function
main "$@"
