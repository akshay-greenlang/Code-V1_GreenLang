#!/bin/bash
# ==============================================================================
# Redis RDB Restore Script
# ==============================================================================
# This script restores Redis data from S3 backups.
#
# Features:
#   - Downloads backup from S3
#   - Verifies checksum integrity
#   - Safely stops Redis
#   - Replaces dump.rdb file
#   - Starts Redis and verifies data loaded
#   - Supports dry-run mode for validation
#
# Required Environment Variables:
#   - BACKUP_TIMESTAMP: Timestamp of backup to restore (YYYYMMDD_HHMMSS)
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
#   - BACKUP_TYPE: Type of backup (rdb or aof, default: rdb)
#   - DRY_RUN: Validate without restoring (default: false)
#   - VERIFY_CHECKSUM: Verify backup checksum (default: true)
#   - RESTORE_TIMEOUT: Timeout for restore operation (default: 600)
#   - RESTORE_STOP_TIMEOUT: Timeout for Redis shutdown (default: 60)
#   - RESTORE_VERIFY_DATA: Verify data after restore (default: true)
#   - LOG_LEVEL: Logging level (DEBUG, INFO, WARN, ERROR)
#
# Exit Codes:
#   0 - Success
#   1 - General error
#   2 - Redis connection error
#   3 - Backup not found
#   4 - Checksum verification failed
#   5 - Redis start failed
#   6 - Data verification failed
#
# Usage:
#   BACKUP_TIMESTAMP=20240115_060000 ./restore-rdb.sh
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# Configuration
# ==============================================================================

# Required parameter
if [ -z "${BACKUP_TIMESTAMP:-}" ]; then
    echo "ERROR: BACKUP_TIMESTAMP is required"
    echo "Usage: BACKUP_TIMESTAMP=20240115_060000 ./restore-rdb.sh"
    exit 1
fi

# Directories
RESTORE_DIR="${RESTORE_DIR:-/restore}"
TMP_DIR="${TMP_DIR:-/tmp}"

# Backup type
BACKUP_TYPE="${BACKUP_TYPE:-rdb}"

# S3 paths
S3_PATH="s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_TYPE}"

# Defaults
DRY_RUN="${DRY_RUN:-false}"
VERIFY_CHECKSUM="${VERIFY_CHECKSUM:-true}"
RESTORE_TIMEOUT="${RESTORE_TIMEOUT:-600}"
RESTORE_STOP_TIMEOUT="${RESTORE_STOP_TIMEOUT:-60}"
RESTORE_VERIFY_DATA="${RESTORE_VERIFY_DATA:-true}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# State tracking
REDIS_WAS_RUNNING=false
BACKUP_CREATED=false
HELPER_POD_CREATED=false

# ==============================================================================
# Logging Functions
# ==============================================================================

log() {
    local level="$1"
    shift
    local message="$*"

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

    echo "{\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"level\":\"$level\",\"component\":\"redis-restore\",\"backup_timestamp\":\"${BACKUP_TIMESTAMP}\",\"message\":\"${message}\"}"
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

    # Clean up helper pod if created
    if [ "$HELPER_POD_CREATED" = "true" ]; then
        log_info "Cleaning up helper pod"
        kubectl delete pod redis-restore-helper -n greenlang --ignore-not-found=true --wait=false 2>/dev/null || true
    fi

    # Remove temporary files
    rm -rf "${RESTORE_DIR}"/* 2>/dev/null || true
    rm -f "${TMP_DIR}/redis_restore_*" 2>/dev/null || true

    if [ $exit_code -eq 0 ]; then
        log_info "Restore operation completed successfully"
    else
        log_error "Restore operation failed with exit code: $exit_code"

        # Attempt recovery if Redis was running
        if [ "$REDIS_WAS_RUNNING" = "true" ] && [ "$exit_code" -ne 0 ]; then
            log_warn "Attempting to restart Redis..."
            kubectl scale statefulset redis-master -n greenlang --replicas=1 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT

# ==============================================================================
# Validation Functions
# ==============================================================================

validate_backup_exists() {
    log_info "Validating backup exists: ${BACKUP_TIMESTAMP}"

    local backup_file
    if [ "$BACKUP_TYPE" = "rdb" ]; then
        backup_file="redis_rdb_${BACKUP_TIMESTAMP}.rdb.gz"
    else
        backup_file="redis_aof_${BACKUP_TIMESTAMP}.aof.gz"
    fi

    local s3_file="${S3_PATH}/${backup_file}"

    if ! aws s3 ls "$s3_file" > /dev/null 2>&1; then
        log_error "Backup not found: $s3_file"
        log_info "Available backups:"
        aws s3 ls "${S3_PATH}/" | grep "${BACKUP_TYPE}_" | tail -20 | while read -r line; do
            log_info "  $line"
        done
        exit 3
    fi

    log_info "Backup validated: $backup_file"
    echo "$backup_file"
}

verify_redis_connection() {
    log_info "Verifying Redis connection"

    local retries=5
    while [ $retries -gt 0 ]; do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null | grep -q PONG; then
            log_info "Redis connection verified"
            REDIS_WAS_RUNNING=true
            return 0
        fi
        retries=$((retries - 1))
        sleep 2
    done

    log_warn "Could not connect to Redis - may already be stopped"
    return 0
}

# ==============================================================================
# Download Functions
# ==============================================================================

download_backup() {
    local backup_file="$1"
    local s3_file="${S3_PATH}/${backup_file}"
    local local_file="${RESTORE_DIR}/${backup_file}"

    log_info "Downloading backup from S3"
    log_info "  Source: ${s3_file}"
    log_info "  Destination: ${local_file}"

    # Create restore directory
    mkdir -p "$RESTORE_DIR"

    # Download backup file
    aws s3 cp "$s3_file" "$local_file"

    # Download checksum
    aws s3 cp "${s3_file}.sha256" "${local_file}.sha256" 2>/dev/null || {
        log_warn "Checksum file not found, skipping checksum verification"
        VERIFY_CHECKSUM="false"
    }

    local file_size=$(stat -c%s "$local_file" 2>/dev/null || stat -f%z "$local_file" 2>/dev/null)
    log_info "Downloaded backup: ${file_size} bytes"

    echo "$local_file"
}

verify_checksum() {
    local file="$1"

    if [ "$VERIFY_CHECKSUM" != "true" ]; then
        log_info "Checksum verification disabled"
        return 0
    fi

    log_info "Verifying backup checksum"

    local expected_checksum=$(cat "${file}.sha256" | cut -d' ' -f1)
    local actual_checksum=$(sha256sum "$file" | cut -d' ' -f1)

    if [ "$expected_checksum" != "$actual_checksum" ]; then
        log_error "Checksum verification failed!"
        log_error "  Expected: $expected_checksum"
        log_error "  Actual:   $actual_checksum"
        exit 4
    fi

    log_info "Checksum verified: $expected_checksum"
}

decompress_backup() {
    local file="$1"

    if [[ "$file" == *.gz ]]; then
        log_info "Decompressing backup"
        gunzip "$file"
        echo "${file%.gz}"
    else
        echo "$file"
    fi
}

# ==============================================================================
# Redis Control Functions
# ==============================================================================

get_redis_state() {
    log_info "Getting current Redis state"

    # Get database size
    local dbsize=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DBSIZE 2>/dev/null | grep -oP '\d+' || echo "0")

    # Get memory usage
    local memory=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning INFO memory 2>/dev/null | grep used_memory_human | cut -d: -f2 | tr -d '\r\n' || echo "unknown")

    log_info "Current state: ${dbsize} keys, ${memory} memory"

    echo "$dbsize"
}

create_pre_restore_backup() {
    log_info "Creating pre-restore backup of current data"

    # Trigger BGSAVE
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning BGSAVE 2>/dev/null || true

    # Wait briefly for save
    sleep 5

    BACKUP_CREATED=true
    log_info "Pre-restore backup triggered"
}

stop_redis_safely() {
    log_info "Stopping Redis safely"

    # Check if Redis is accessible
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null | grep -q PONG; then
        # Trigger final save
        log_info "Triggering final BGSAVE"
        redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning BGSAVE 2>/dev/null || true

        # Wait for BGSAVE to complete
        local start_time=$(date +%s)
        while true; do
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))

            if [ $elapsed -gt "$RESTORE_STOP_TIMEOUT" ]; then
                log_warn "BGSAVE timeout during shutdown"
                break
            fi

            local bgsave_in_progress=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning INFO persistence 2>/dev/null | grep rdb_bgsave_in_progress | cut -d: -f2 | tr -d '\r\n' || echo "0")

            if [ "$bgsave_in_progress" = "0" ]; then
                log_info "BGSAVE completed"
                break
            fi

            sleep 2
        done
    fi

    # Scale down Redis StatefulSet
    log_info "Scaling down Redis StatefulSet"
    kubectl scale statefulset redis-master -n greenlang --replicas=0

    # Wait for pods to terminate
    log_info "Waiting for Redis pods to terminate"
    kubectl wait --for=delete pod/redis-master-0 -n greenlang --timeout="${RESTORE_STOP_TIMEOUT}s" 2>/dev/null || true

    # Verify pod is gone
    sleep 5
    if kubectl get pod redis-master-0 -n greenlang 2>/dev/null; then
        log_warn "Pod still exists, force deleting"
        kubectl delete pod redis-master-0 -n greenlang --force --grace-period=0 2>/dev/null || true
        sleep 5
    fi

    log_info "Redis stopped"
}

replace_data_file() {
    local backup_file="$1"

    log_info "Replacing Redis data file"

    # Get PVC name
    local pvc_name="redis-data-redis-master-0"

    # Create helper pod
    log_info "Creating helper pod to access Redis data volume"
    HELPER_POD_CREATED=true

    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: redis-restore-helper
  namespace: greenlang
  labels:
    app: redis-restore-helper
spec:
  containers:
  - name: helper
    image: redis:7.2-alpine
    command: ["sleep", "3600"]
    volumeMounts:
    - name: redis-data
      mountPath: /data
    securityContext:
      runAsUser: 0
  volumes:
  - name: redis-data
    persistentVolumeClaim:
      claimName: ${pvc_name}
  restartPolicy: Never
EOF

    # Wait for helper pod
    log_info "Waiting for helper pod to be ready"
    kubectl wait --for=condition=Ready pod/redis-restore-helper -n greenlang --timeout=120s

    # Backup existing data file
    log_info "Backing up existing dump.rdb"
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    kubectl exec -n greenlang redis-restore-helper -- mv /data/dump.rdb "/data/dump.rdb.pre-restore.${backup_timestamp}" 2>/dev/null || {
        log_warn "No existing dump.rdb to backup"
    }

    # Determine filename based on backup type
    local data_file
    if [ "$BACKUP_TYPE" = "rdb" ]; then
        data_file="dump.rdb"
    else
        data_file="appendonly.aof"
    fi

    # Copy new data file
    log_info "Copying restored data to Redis volume"
    kubectl cp "$backup_file" "greenlang/redis-restore-helper:/data/${data_file}"

    # Set permissions
    log_info "Setting file permissions"
    kubectl exec -n greenlang redis-restore-helper -- chown 1000:1000 "/data/${data_file}"
    kubectl exec -n greenlang redis-restore-helper -- chmod 644 "/data/${data_file}"

    # Verify file
    local restored_size=$(kubectl exec -n greenlang redis-restore-helper -- stat -c%s "/data/${data_file}" 2>/dev/null || kubectl exec -n greenlang redis-restore-helper -- stat -f%z "/data/${data_file}")
    log_info "Restored file size: ${restored_size} bytes"

    # Delete helper pod
    log_info "Cleaning up helper pod"
    kubectl delete pod redis-restore-helper -n greenlang --wait=true
    HELPER_POD_CREATED=false

    log_info "Data file replaced successfully"
}

start_redis() {
    log_info "Starting Redis"

    # Scale up StatefulSet
    kubectl scale statefulset redis-master -n greenlang --replicas=1

    # Wait for pod to be ready
    log_info "Waiting for Redis pod to be ready"
    kubectl wait --for=condition=Ready pod/redis-master-0 -n greenlang --timeout="${RESTORE_TIMEOUT}s"

    # Wait for Redis to accept connections
    log_info "Waiting for Redis to accept connections"
    local retries=60
    while [ $retries -gt 0 ]; do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping 2>/dev/null | grep -q PONG; then
            log_info "Redis is accepting connections"
            return 0
        fi
        retries=$((retries - 1))
        sleep 2
    done

    log_error "Redis failed to start accepting connections"
    exit 5
}

verify_restore() {
    if [ "$RESTORE_VERIFY_DATA" != "true" ]; then
        log_info "Data verification disabled"
        return 0
    fi

    log_info "Verifying restored data"

    # Get database info
    local dbsize=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning DBSIZE | grep -oP '\d+' || echo "0")
    local memory=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r\n')
    local last_save=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning LASTSAVE)

    log_info "=========================================="
    log_info "Restore Verification"
    log_info "=========================================="
    log_info "  Database size: ${dbsize} keys"
    log_info "  Memory usage: ${memory}"
    log_info "  Last save: ${last_save}"
    log_info "=========================================="

    # Warn if database appears empty
    if [ "$dbsize" -eq 0 ]; then
        log_warn "Database is empty after restore"
        log_warn "This may be expected for an empty backup or may indicate a problem"
    fi

    # Run basic health check
    local ping_result=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" --no-auth-warning ping)
    if [ "$ping_result" != "PONG" ]; then
        log_error "Redis health check failed"
        exit 6
    fi

    log_info "Data verification completed successfully"
}

# ==============================================================================
# Main Function
# ==============================================================================

main() {
    local start_time=$(date +%s)

    log_info "=========================================="
    log_info "Redis Restore Operation Started"
    log_info "=========================================="
    log_info "Backup timestamp: ${BACKUP_TIMESTAMP}"
    log_info "Backup type: ${BACKUP_TYPE}"
    log_info "Dry run: ${DRY_RUN}"
    log_info "Redis host: ${REDIS_HOST}:${REDIS_PORT}"
    log_info "S3 bucket: ${S3_BUCKET}"
    log_info "S3 prefix: ${S3_PREFIX}"
    log_info "=========================================="

    # Step 1: Validate backup exists
    local backup_file=$(validate_backup_exists)

    # Step 2: Download backup
    local local_file=$(download_backup "$backup_file")

    # Step 3: Verify checksum
    verify_checksum "$local_file"

    # Step 4: Decompress
    local data_file=$(decompress_backup "$local_file")

    log_info "Backup downloaded and verified: ${data_file}"

    # Check for dry run
    if [ "$DRY_RUN" = "true" ]; then
        log_info "=========================================="
        log_info "DRY RUN MODE"
        log_info "=========================================="
        log_info "Backup validated successfully"
        log_info "No changes made to Redis"
        log_info "To perform actual restore, set DRY_RUN=false"
        log_info "=========================================="
        exit 0
    fi

    # Confirmation warning
    log_warn "=========================================="
    log_warn "WARNING: THIS WILL OVERWRITE EXISTING DATA"
    log_warn "=========================================="
    log_info "Restore will begin in 10 seconds..."
    log_info "Press Ctrl+C to abort"
    sleep 10

    # Step 5: Check Redis connection
    verify_redis_connection

    # Step 6: Get current state (for comparison)
    local dbsize_before="0"
    if [ "$REDIS_WAS_RUNNING" = "true" ]; then
        dbsize_before=$(get_redis_state)

        # Create pre-restore backup
        create_pre_restore_backup
    fi

    # Step 7: Stop Redis safely
    stop_redis_safely

    # Step 8: Replace data file
    replace_data_file "$data_file"

    # Step 9: Start Redis
    start_redis

    # Step 10: Verify restore
    verify_restore

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_info "=========================================="
    log_info "Redis Restore Completed Successfully"
    log_info "=========================================="
    log_info "Backup: redis_${BACKUP_TYPE}_${BACKUP_TIMESTAMP}"
    log_info "Duration: ${duration} seconds"
    log_info "Previous key count: ${dbsize_before}"
    log_info "=========================================="
}

# Run main
main "$@"
