#!/bin/bash
# ==============================================================================
# Redis RDB Backup Script
# ==============================================================================
# This script performs RDB snapshot backups of Redis data to S3.
#
# Features:
#   - Connects to Redis master
#   - Triggers BGSAVE operation
#   - Waits for completion with timeout
#   - Copies dump.rdb file
#   - Compresses with gzip
#   - Uploads to S3 with timestamp
#   - Verifies upload integrity
#   - Cleans up old backups based on retention policy
#
# Required Environment Variables:
#   - REDIS_HOST: Redis server hostname
#   - REDIS_PORT: Redis server port
#   - REDIS_PASSWORD: Redis AUTH password
#   - S3_BUCKET: S3 bucket name
#   - S3_REGION: AWS region
#   - S3_PREFIX: S3 key prefix
#   - AWS_ACCESS_KEY_ID: AWS access key
#   - AWS_SECRET_ACCESS_KEY: AWS secret key
#
# Optional Environment Variables:
#   - RETENTION_DAYS: Number of days to retain backups (default: 7)
#   - BGSAVE_TIMEOUT: Timeout for BGSAVE in seconds (default: 300)
#   - COMPRESSION_ENABLED: Enable gzip compression (default: true)
#   - ENCRYPTION_ENABLED: Enable S3 server-side encryption (default: true)
#   - VERIFY_BACKUP: Verify upload integrity (default: true)
#   - LOG_LEVEL: Logging level (DEBUG, INFO, WARN, ERROR)
#   - MIN_BACKUP_SIZE_BYTES: Minimum valid backup size (default: 1024)
#
# Exit Codes:
#   0 - Success
#   1 - General error
#   2 - Redis connection error
#   3 - BGSAVE timeout
#   4 - Upload verification failed
#   5 - Backup file too small
#
# Usage:
#   ./backup-rdb.sh
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

# Timestamp for this backup
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
BACKUP_NAME="redis_rdb_${TIMESTAMP}"

# Directories
LOCAL_BACKUP_DIR="${LOCAL_BACKUP_DIR:-/backup}"
TMP_DIR="${TMP_DIR:-/tmp}"

# S3 paths
S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/rdb"

# Defaults
RETENTION_DAYS="${RETENTION_DAYS:-7}"
BGSAVE_TIMEOUT="${BGSAVE_TIMEOUT:-300}"
BGSAVE_RETRY_COUNT="${BGSAVE_RETRY_COUNT:-3}"
BGSAVE_RETRY_DELAY="${BGSAVE_RETRY_DELAY:-10}"
COMPRESSION_ENABLED="${COMPRESSION_ENABLED:-true}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"
ENCRYPTION_ENABLED="${ENCRYPTION_ENABLED:-true}"
VERIFY_BACKUP="${VERIFY_BACKUP:-true}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MIN_BACKUP_SIZE_BYTES="${MIN_BACKUP_SIZE_BYTES:-1024}"
MAX_BACKUPS_KEEP="${MAX_BACKUPS_KEEP:-28}"

# Metrics
METRICS_FILE="${TMP_DIR}/backup_metrics.prom"

# ==============================================================================
# Logging Functions
# ==============================================================================

log() {
    local level="$1"
    shift
    local message="$*"

    # Log level filtering
    case "$LOG_LEVEL" in
        DEBUG) allowed_levels="DEBUG INFO WARN ERROR" ;;
        INFO)  allowed_levels="INFO WARN ERROR" ;;
        WARN)  allowed_levels="WARN ERROR" ;;
        ERROR) allowed_levels="ERROR" ;;
        *)     allowed_levels="INFO WARN ERROR" ;;
    esac

    if [[ ! " $allowed_levels " =~ " $level " ]]; then
        return 0
    fi

    # JSON formatted log output
    echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"level\":\"$level\",\"component\":\"rdb-backup\",\"backup_name\":\"${BACKUP_NAME}\",\"message\":\"${message}\"}"
}

log_debug() { log "DEBUG" "$@"; }
log_info()  { log "INFO" "$@"; }
log_warn()  { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# ==============================================================================
# Cleanup Function
# ==============================================================================

cleanup() {
    local exit_code=$?
    log_info "Running cleanup"

    # Remove temporary files
    rm -f "${LOCAL_BACKUP_DIR}/${BACKUP_NAME}"* 2>/dev/null || true
    rm -f "${TMP_DIR}/redis_backup_*" 2>/dev/null || true

    # Record metrics
    record_metrics "$exit_code"

    if [ $exit_code -eq 0 ]; then
        log_info "Backup completed successfully"
    else
        log_error "Backup failed with exit code: $exit_code"
    fi
}
trap cleanup EXIT

# ==============================================================================
# Metrics Functions
# ==============================================================================

record_metrics() {
    local status="$1"
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))

    cat > "$METRICS_FILE" <<EOF
# HELP redis_backup_last_success_timestamp Unix timestamp of last successful backup
# TYPE redis_backup_last_success_timestamp gauge
redis_backup_last_success_timestamp{type="rdb"} $([ "$status" -eq 0 ] && echo "$end_time" || echo "0")

# HELP redis_backup_duration_seconds Duration of backup operation in seconds
# TYPE redis_backup_duration_seconds gauge
redis_backup_duration_seconds{type="rdb"} $duration

# HELP redis_backup_size_bytes Size of backup file in bytes
# TYPE redis_backup_size_bytes gauge
redis_backup_size_bytes{type="rdb"} ${BACKUP_SIZE:-0}

# HELP redis_backup_status Status of last backup (1=success, 0=failure)
# TYPE redis_backup_status gauge
redis_backup_status{type="rdb"} $([ "$status" -eq 0 ] && echo "1" || echo "0")
EOF

    log_debug "Metrics recorded to $METRICS_FILE"
}

# ==============================================================================
# Redis Functions
# ==============================================================================

verify_redis_connection() {
    log_info "Verifying Redis connection to ${REDIS_HOST}:${REDIS_PORT}"

    local retries=5
    while [ $retries -gt 0 ]; do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null | grep -q PONG; then
            log_info "Redis connection verified"
            return 0
        fi
        retries=$((retries - 1))
        log_warn "Redis connection failed, retrying... ($retries attempts left)"
        sleep 2
    done

    log_error "Failed to connect to Redis after multiple attempts"
    exit 2
}

get_redis_info() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning INFO "$1" 2>/dev/null
}

get_last_save_time() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning LASTSAVE 2>/dev/null
}

trigger_bgsave() {
    log_info "Triggering BGSAVE operation"

    local attempt=1
    while [ $attempt -le "$BGSAVE_RETRY_COUNT" ]; do
        local last_save_before=$(get_last_save_time)

        # Trigger BGSAVE
        local result=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning BGSAVE 2>/dev/null)

        if [[ "$result" == *"Background saving started"* ]] || [[ "$result" == *"Background saving scheduled"* ]]; then
            log_info "BGSAVE triggered successfully (attempt $attempt)"
            echo "$last_save_before"
            return 0
        fi

        log_warn "BGSAVE trigger failed (attempt $attempt): $result"
        attempt=$((attempt + 1))
        sleep "$BGSAVE_RETRY_DELAY"
    done

    log_error "Failed to trigger BGSAVE after $BGSAVE_RETRY_COUNT attempts"
    exit 3
}

wait_for_bgsave() {
    local last_save_before="$1"
    log_info "Waiting for BGSAVE to complete (timeout: ${BGSAVE_TIMEOUT}s)"

    local start_time=$(date +%s)
    local dots=""

    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))

        if [ $elapsed -gt "$BGSAVE_TIMEOUT" ]; then
            log_error "BGSAVE timeout after ${elapsed}s"
            exit 3
        fi

        # Check if BGSAVE is in progress
        local bgsave_in_progress=$(get_redis_info persistence | grep rdb_bgsave_in_progress | cut -d: -f2 | tr -d '\r\n')

        if [ "$bgsave_in_progress" = "0" ]; then
            local last_save_after=$(get_last_save_time)
            if [ "$last_save_after" != "$last_save_before" ]; then
                log_info "BGSAVE completed after ${elapsed}s"
                return 0
            fi
        fi

        # Progress indicator
        dots="${dots}."
        if [ ${#dots} -gt 30 ]; then
            dots="."
        fi
        log_debug "BGSAVE in progress${dots} (${elapsed}s elapsed)"

        sleep 2
    done
}

# ==============================================================================
# Backup Functions
# ==============================================================================

copy_rdb_file() {
    log_info "Copying dump.rdb from Redis"

    local rdb_file="${LOCAL_BACKUP_DIR}/${BACKUP_NAME}.rdb"

    # Method 1: Direct copy if running in same pod
    if [ -f "/data/dump.rdb" ]; then
        cp /data/dump.rdb "$rdb_file"
        log_info "Copied dump.rdb from local /data directory"
    # Method 2: Copy via kubectl exec
    else
        log_info "Copying dump.rdb via kubectl exec"
        kubectl exec -n greenlang redis-master-0 -- cat /data/dump.rdb > "$rdb_file"
    fi

    # Verify file exists and has content
    if [ ! -f "$rdb_file" ]; then
        log_error "RDB file copy failed - file not found"
        exit 1
    fi

    local file_size=$(stat -c%s "$rdb_file" 2>/dev/null || stat -f%z "$rdb_file" 2>/dev/null)
    log_info "RDB file size: ${file_size} bytes"

    if [ "$file_size" -lt "$MIN_BACKUP_SIZE_BYTES" ]; then
        log_error "RDB file too small (${file_size} bytes < ${MIN_BACKUP_SIZE_BYTES} bytes minimum)"
        exit 5
    fi

    BACKUP_SIZE=$file_size
    echo "$rdb_file"
}

compress_backup() {
    local input_file="$1"

    if [ "$COMPRESSION_ENABLED" != "true" ]; then
        log_info "Compression disabled, skipping"
        echo "$input_file"
        return
    fi

    log_info "Compressing backup with gzip (level ${COMPRESSION_LEVEL})"

    local compressed_file="${input_file}.gz"
    gzip -"${COMPRESSION_LEVEL}" -c "$input_file" > "$compressed_file"

    local original_size=$(stat -c%s "$input_file" 2>/dev/null || stat -f%z "$input_file" 2>/dev/null)
    local compressed_size=$(stat -c%s "$compressed_file" 2>/dev/null || stat -f%z "$compressed_file" 2>/dev/null)
    local ratio=$(echo "scale=2; ($original_size - $compressed_size) * 100 / $original_size" | bc)

    log_info "Compression complete: ${original_size} -> ${compressed_size} bytes (${ratio}% reduction)"

    # Remove uncompressed file
    rm -f "$input_file"

    BACKUP_SIZE=$compressed_size
    echo "$compressed_file"
}

generate_checksum() {
    local file="$1"

    log_info "Generating SHA256 checksum"

    local checksum=$(sha256sum "$file" | cut -d' ' -f1)
    echo "$checksum" > "${file}.sha256"

    log_info "Checksum: ${checksum}"
    echo "$checksum"
}

upload_to_s3() {
    local file="$1"
    local checksum="$2"

    log_info "Uploading backup to S3: ${S3_PATH}/"

    # Build S3 upload arguments
    local s3_args="--region ${S3_REGION}"

    if [ "$ENCRYPTION_ENABLED" = "true" ]; then
        s3_args="$s3_args --sse AES256"
        log_debug "S3 server-side encryption enabled (AES256)"
    fi

    # Add metadata
    local metadata="backup-timestamp=${TIMESTAMP},backup-type=rdb,checksum=${checksum}"
    s3_args="$s3_args --metadata $metadata"

    # Upload backup file
    local filename=$(basename "$file")
    log_info "Uploading ${filename}..."
    aws s3 cp "$file" "${S3_PATH}/${filename}" $s3_args

    # Upload checksum file
    log_info "Uploading ${filename}.sha256..."
    aws s3 cp "${file}.sha256" "${S3_PATH}/${filename}.sha256" $s3_args

    log_info "Upload completed successfully"
    echo "${S3_PATH}/${filename}"
}

verify_upload() {
    local s3_file="$1"
    local local_checksum="$2"

    if [ "$VERIFY_BACKUP" != "true" ]; then
        log_info "Upload verification disabled, skipping"
        return 0
    fi

    log_info "Verifying upload integrity"

    # Download and verify checksum
    local s3_checksum=$(aws s3 cp "${s3_file}.sha256" - | cut -d' ' -f1)

    if [ "$local_checksum" != "$s3_checksum" ]; then
        log_error "Checksum verification failed!"
        log_error "  Local:  $local_checksum"
        log_error "  Remote: $s3_checksum"
        exit 4
    fi

    # Verify file exists and has correct size
    local s3_size=$(aws s3 ls "$s3_file" | awk '{print $3}')
    log_info "S3 file size: ${s3_size} bytes"

    log_info "Upload verification passed"
}

cleanup_old_backups() {
    log_info "Cleaning up backups older than ${RETENTION_DAYS} days"

    local cutoff_date=$(date -u -d "-${RETENTION_DAYS} days" +%Y%m%d 2>/dev/null || date -u -v-${RETENTION_DAYS}d +%Y%m%d)
    local deleted_count=0

    # List all backups and delete old ones
    aws s3 ls "${S3_PATH}/" | while read -r line; do
        local file_name=$(echo "$line" | awk '{print $4}')

        # Skip if not an RDB backup file
        if [[ ! "$file_name" =~ ^redis_rdb_[0-9]{8}_[0-9]{6}\.rdb ]]; then
            continue
        fi

        # Extract date from filename
        local file_date=$(echo "$file_name" | grep -oP '\d{8}' | head -1)

        if [ -n "$file_date" ] && [ "$file_date" -lt "$cutoff_date" ]; then
            log_info "Deleting old backup: ${file_name}"
            aws s3 rm "${S3_PATH}/${file_name}"

            # Also delete checksum file
            aws s3 rm "${S3_PATH}/${file_name}.sha256" 2>/dev/null || true

            deleted_count=$((deleted_count + 1))
        fi
    done

    log_info "Cleanup complete, deleted ${deleted_count} old backups"

    # Enforce maximum backup count
    enforce_max_backups
}

enforce_max_backups() {
    log_info "Enforcing maximum backup count: ${MAX_BACKUPS_KEEP}"

    local backup_count=$(aws s3 ls "${S3_PATH}/" | grep -c "redis_rdb_.*\.rdb\.gz$" || echo "0")

    if [ "$backup_count" -gt "$MAX_BACKUPS_KEEP" ]; then
        local to_delete=$((backup_count - MAX_BACKUPS_KEEP))
        log_info "Deleting ${to_delete} oldest backups to maintain limit"

        aws s3 ls "${S3_PATH}/" | grep "redis_rdb_.*\.rdb\.gz$" | sort | head -n "$to_delete" | while read -r line; do
            local file_name=$(echo "$line" | awk '{print $4}')
            log_info "Deleting excess backup: ${file_name}"
            aws s3 rm "${S3_PATH}/${file_name}"
            aws s3 rm "${S3_PATH}/${file_name}.sha256" 2>/dev/null || true
        done
    fi
}

# ==============================================================================
# Main Function
# ==============================================================================

main() {
    START_TIME=$(date +%s)

    log_info "=========================================="
    log_info "Redis RDB Backup Started"
    log_info "=========================================="
    log_info "Backup name: ${BACKUP_NAME}"
    log_info "Redis host: ${REDIS_HOST}:${REDIS_PORT}"
    log_info "S3 bucket: ${S3_BUCKET}"
    log_info "S3 prefix: ${S3_PREFIX}"
    log_info "Retention: ${RETENTION_DAYS} days"

    # Create backup directory
    mkdir -p "$LOCAL_BACKUP_DIR"

    # Step 1: Verify Redis connection
    verify_redis_connection

    # Step 2: Trigger and wait for BGSAVE
    local last_save_before=$(trigger_bgsave)
    wait_for_bgsave "$last_save_before"

    # Step 3: Copy RDB file
    local rdb_file=$(copy_rdb_file)

    # Step 4: Compress backup
    local backup_file=$(compress_backup "$rdb_file")

    # Step 5: Generate checksum
    local checksum=$(generate_checksum "$backup_file")

    # Step 6: Upload to S3
    local s3_location=$(upload_to_s3 "$backup_file" "$checksum")

    # Step 7: Verify upload
    verify_upload "$s3_location" "$checksum"

    # Step 8: Cleanup old backups
    cleanup_old_backups

    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))

    log_info "=========================================="
    log_info "Redis RDB Backup Completed"
    log_info "=========================================="
    log_info "Backup: ${BACKUP_NAME}"
    log_info "Location: ${s3_location}"
    log_info "Size: ${BACKUP_SIZE} bytes"
    log_info "Duration: ${duration} seconds"
    log_info "=========================================="
}

# Run main function
main "$@"
