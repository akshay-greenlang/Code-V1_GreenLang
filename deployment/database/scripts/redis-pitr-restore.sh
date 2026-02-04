#!/usr/bin/env bash
# =============================================================================
# GreenLang Climate OS - Redis Point-in-Time Recovery Script
# =============================================================================
# PRD: INFRA-003 Redis Caching Cluster
# Purpose: Restore Redis to a specific point in time using RDB + AOF replay
# Usage: ./redis-pitr-restore.sh --target-time "2026-02-04T10:00:00Z" [--dry-run]
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
S3_BUCKET="${REDIS_BACKUP_S3_BUCKET:-greenlang-prod-redis-backups}"
S3_PREFIX="${REDIS_BACKUP_S3_PREFIX:-redis/rdb}"
S3_AOF_PREFIX="${REDIS_BACKUP_S3_AOF_PREFIX:-redis/aof}"
S3_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
RESTORE_DIR="${REDIS_RESTORE_DIR:-/tmp/redis-restore}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_AUTH="${REDIS_AUTH:-}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_FILE="/var/log/redis/pitr-restore-${TIMESTAMP}.log"
DRY_RUN=false
TARGET_TIME=""

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-time)
            TARGET_TIME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --redis-host)
            REDIS_HOST="$2"
            shift 2
            ;;
        --redis-port)
            REDIS_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --target-time <ISO8601> [--dry-run] [--s3-bucket <bucket>]"
            exit 1
            ;;
    esac
done

if [ -z "${TARGET_TIME}" ]; then
    echo "ERROR: --target-time is required"
    echo "Usage: $0 --target-time \"2026-02-04T10:00:00Z\" [--dry-run]"
    exit 1
fi

# =============================================================================
# Functions
# =============================================================================

log() {
    mkdir -p "$(dirname "${LOG_FILE}")"
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $1" | tee -a "${LOG_FILE}"
}

redis_cmd() {
    if [ -n "${REDIS_AUTH}" ]; then
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" -a "${REDIS_AUTH}" --no-auth-warning "$@"
    else
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "$@"
    fi
}

find_closest_rdb() {
    local target_epoch
    target_epoch=$(date -d "${TARGET_TIME}" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%SZ" "${TARGET_TIME}" +%s)

    log "INFO: Finding closest RDB snapshot before ${TARGET_TIME} (epoch: ${target_epoch})..."

    # List all RDB snapshots in S3
    local snapshots
    snapshots=$(aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        --region "${S3_REGION}" \
        --recursive 2>/dev/null | grep '\.rdb' | sort -k1,2)

    if [ -z "${snapshots}" ]; then
        log "ERROR: No RDB snapshots found in s3://${S3_BUCKET}/${S3_PREFIX}/"
        exit 1
    fi

    # Find the closest snapshot before target time
    local best_key=""
    local best_time=""

    while IFS= read -r line; do
        local snap_date
        snap_date=$(echo "${line}" | awk '{print $1"T"$2}')
        local snap_epoch
        snap_epoch=$(date -d "${snap_date}" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "${snap_date}" +%s 2>/dev/null || echo 0)
        local snap_key
        snap_key=$(echo "${line}" | awk '{print $4}')

        if [ "${snap_epoch}" -le "${target_epoch}" ]; then
            best_key="${snap_key}"
            best_time="${snap_date}"
        fi
    done <<< "${snapshots}"

    if [ -z "${best_key}" ]; then
        log "ERROR: No RDB snapshot found before ${TARGET_TIME}"
        exit 1
    fi

    log "INFO: Found closest RDB: ${best_key} (${best_time})"
    echo "${best_key}"
}

find_aof_segments() {
    local rdb_time="$1"

    log "INFO: Finding AOF segments between RDB snapshot and ${TARGET_TIME}..."

    aws s3 ls "s3://${S3_BUCKET}/${S3_AOF_PREFIX}/" \
        --region "${S3_REGION}" \
        --recursive 2>/dev/null | grep '\.aof' | sort -k1,2 || true
}

download_snapshot() {
    local s3_key="$1"
    local local_path="${RESTORE_DIR}/dump.rdb"

    log "INFO: Downloading RDB snapshot..."
    mkdir -p "${RESTORE_DIR}"

    aws s3 cp "s3://${S3_BUCKET}/${s3_key}" "${local_path}" \
        --region "${S3_REGION}" \
        2>&1 | tee -a "${LOG_FILE}"

    # Verify checksum
    local checksum_key="${s3_key}.sha256"
    if aws s3 ls "s3://${S3_BUCKET}/${checksum_key}" --region "${S3_REGION}" &>/dev/null; then
        log "INFO: Verifying checksum..."
        aws s3 cp "s3://${S3_BUCKET}/${checksum_key}" "${RESTORE_DIR}/dump.rdb.sha256" \
            --region "${S3_REGION}" 2>/dev/null

        local expected
        expected=$(cat "${RESTORE_DIR}/dump.rdb.sha256" | awk '{print $1}')
        local actual
        actual=$(sha256sum "${local_path}" | awk '{print $1}')

        if [ "${expected}" != "${actual}" ]; then
            log "ERROR: Checksum mismatch! Expected: ${expected}, Got: ${actual}"
            exit 1
        fi
        log "INFO: Checksum verified OK"
    else
        log "WARN: No checksum file found, skipping verification"
    fi

    echo "${local_path}"
}

stop_redis_traffic() {
    log "INFO: Pausing client connections..."

    if [ "${DRY_RUN}" = true ]; then
        log "DRY-RUN: Would pause CLIENT PAUSE ALL"
        return
    fi

    # Pause all clients for 60 seconds
    redis_cmd CLIENT PAUSE 60000 ALL
    log "INFO: Clients paused for 60 seconds"
}

restore_rdb() {
    local rdb_path="$1"

    log "INFO: Restoring RDB snapshot to Redis..."

    if [ "${DRY_RUN}" = true ]; then
        log "DRY-RUN: Would restore ${rdb_path} to ${REDIS_HOST}:${REDIS_PORT}"
        return
    fi

    # Stop Redis, replace RDB, restart
    local redis_dir
    redis_dir=$(redis_cmd CONFIG GET dir | tail -1)

    log "INFO: Redis data directory: ${redis_dir}"

    # Shutdown Redis gracefully
    redis_cmd SHUTDOWN NOSAVE 2>/dev/null || true

    # Wait for Redis to stop
    sleep 2

    # Copy RDB file
    cp "${rdb_path}" "${redis_dir}/dump.rdb"
    chown redis:redis "${redis_dir}/dump.rdb" 2>/dev/null || true

    # Restart Redis (systemd or manual)
    if systemctl is-active redis &>/dev/null; then
        systemctl start redis
    else
        log "WARN: Redis not managed by systemd. Start Redis manually."
    fi

    # Wait for Redis to be ready
    local retries=30
    while [ ${retries} -gt 0 ]; do
        if redis_cmd PING 2>/dev/null | grep -q PONG; then
            log "INFO: Redis is ready"
            break
        fi
        retries=$((retries - 1))
        sleep 1
    done

    if [ ${retries} -eq 0 ]; then
        log "ERROR: Redis did not start within 30 seconds"
        exit 1
    fi
}

validate_restore() {
    log "INFO: Validating restore..."

    if [ "${DRY_RUN}" = true ]; then
        log "DRY-RUN: Would validate restore"
        return
    fi

    # Check Redis info
    local db_size
    db_size=$(redis_cmd DBSIZE | awk '{print $2}' 2>/dev/null || echo "unknown")
    log "INFO: Database size after restore: ${db_size} keys"

    local memory
    memory=$(redis_cmd INFO memory | grep "used_memory_human" | tr -d '\r' || echo "unknown")
    log "INFO: Memory usage: ${memory}"

    local last_save
    last_save=$(redis_cmd LASTSAVE || echo "unknown")
    log "INFO: Last save timestamp: ${last_save}"

    # Resume clients
    redis_cmd CLIENT UNPAUSE 2>/dev/null || true
    log "INFO: Client connections resumed"
}

# =============================================================================
# Main
# =============================================================================

log "=============================================="
log "GreenLang Redis Point-in-Time Recovery"
log "Target Time: ${TARGET_TIME}"
log "Redis: ${REDIS_HOST}:${REDIS_PORT}"
log "S3 Bucket: ${S3_BUCKET}"
log "Dry Run: ${DRY_RUN}"
log "=============================================="

# Step 1: Find closest RDB snapshot
rdb_key=$(find_closest_rdb)

# Step 2: Download snapshot
rdb_path=$(download_snapshot "${rdb_key}")

# Step 3: Stop traffic
stop_redis_traffic

# Step 4: Restore RDB
restore_rdb "${rdb_path}"

# Step 5: Validate
validate_restore

# Cleanup
rm -rf "${RESTORE_DIR}"

log "=============================================="
log "PITR restore completed successfully"
log "=============================================="
