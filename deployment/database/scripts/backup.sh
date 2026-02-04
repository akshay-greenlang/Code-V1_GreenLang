#!/usr/bin/env bash
# =============================================================================
# GreenLang Climate OS - PostgreSQL Backup Script
# =============================================================================
# PRD: INFRA-002 PostgreSQL + TimescaleDB Primary/Replica Configuration
# Purpose: Full and incremental database backups via pgBackRest
# Usage: ./backup.sh [full|incr|diff] [--stanza=greenlang]
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
BACKUP_TYPE="${1:-incr}"
STANZA="${PGBACKREST_STANZA:-greenlang}"
S3_BUCKET="${BACKUP_S3_BUCKET:-greenlang-prod-backups}"
S3_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
RETENTION_FULL="${RETENTION_FULL_COUNT:-4}"
RETENTION_DIFF="${RETENTION_DIFF_COUNT:-14}"
LOG_DIR="/var/log/pgbackrest"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_FILE="${LOG_DIR}/backup-${BACKUP_TYPE}-${TIMESTAMP}.log"

# Database connection (from environment or defaults)
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres_admin}"
PGDATABASE="${PGDATABASE:-greenlang}"

# =============================================================================
# Functions
# =============================================================================

log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $1" | tee -a "${LOG_FILE}"
}

check_prerequisites() {
    log "INFO: Checking prerequisites..."

    # Check pgBackRest is installed
    if ! command -v pgbackrest &>/dev/null; then
        log "ERROR: pgbackrest is not installed"
        exit 1
    fi

    # Check AWS CLI for S3 operations
    if ! command -v aws &>/dev/null; then
        log "ERROR: aws CLI is not installed"
        exit 1
    fi

    # Check stanza exists
    if ! pgbackrest --stanza="${STANZA}" info &>/dev/null; then
        log "WARN: Stanza '${STANZA}' not found, creating..."
        pgbackrest --stanza="${STANZA}" stanza-create
    fi

    log "INFO: Prerequisites OK"
}

run_backup() {
    local backup_type="$1"

    log "INFO: Starting ${backup_type} backup for stanza '${STANZA}'..."

    # Run pgBackRest backup
    pgbackrest \
        --stanza="${STANZA}" \
        --type="${backup_type}" \
        --repo1-type=s3 \
        --repo1-s3-bucket="${S3_BUCKET}" \
        --repo1-s3-region="${S3_REGION}" \
        --repo1-s3-endpoint="s3.${S3_REGION}.amazonaws.com" \
        --repo1-path="/pgbackrest/${STANZA}" \
        --repo1-cipher-type=aes-256-cbc \
        --repo1-retention-full="${RETENTION_FULL}" \
        --repo1-retention-diff="${RETENTION_DIFF}" \
        --compress-type=zst \
        --compress-level=3 \
        --process-max=4 \
        --log-level-console=info \
        --log-level-file=detail \
        backup \
        2>&1 | tee -a "${LOG_FILE}"

    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        log "INFO: ${backup_type} backup completed successfully"
    else
        log "ERROR: ${backup_type} backup failed with exit code ${exit_code}"
        return ${exit_code}
    fi
}

verify_backup() {
    log "INFO: Verifying latest backup..."

    pgbackrest \
        --stanza="${STANZA}" \
        --repo1-type=s3 \
        --repo1-s3-bucket="${S3_BUCKET}" \
        --repo1-s3-region="${S3_REGION}" \
        --repo1-s3-endpoint="s3.${S3_REGION}.amazonaws.com" \
        --repo1-path="/pgbackrest/${STANZA}" \
        verify \
        2>&1 | tee -a "${LOG_FILE}"

    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        log "INFO: Backup verification passed"
    else
        log "WARN: Backup verification had warnings (exit code ${exit_code})"
    fi
}

collect_stats() {
    log "INFO: Collecting backup statistics..."

    # Get backup info
    local info_output
    info_output=$(pgbackrest \
        --stanza="${STANZA}" \
        --repo1-type=s3 \
        --repo1-s3-bucket="${S3_BUCKET}" \
        --repo1-s3-region="${S3_REGION}" \
        --repo1-s3-endpoint="s3.${S3_REGION}.amazonaws.com" \
        --repo1-path="/pgbackrest/${STANZA}" \
        --output=json \
        info 2>/dev/null)

    # Record row counts for key tables
    log "INFO: Recording table row counts..."
    local row_counts
    row_counts=$(psql -h "${PGHOST}" -p "${PGPORT}" -U "${PGUSER}" -d "${PGDATABASE}" -t -A -c "
        SELECT json_build_object(
            'emission_measurements', (SELECT count(*) FROM emission_measurements),
            'sensor_readings', (SELECT count(*) FROM sensor_readings),
            'calculation_results', (SELECT count(*) FROM calculation_results),
            'audit_log', (SELECT count(*) FROM audit_log),
            'vector_embeddings', (SELECT count(*) FROM vector_embeddings),
            'timestamp', NOW()
        );
    " 2>/dev/null || echo '{"error": "could not collect row counts"}')

    # Upload metadata to S3
    local metadata_file="/tmp/backup-metadata-${TIMESTAMP}.json"
    cat > "${metadata_file}" << METADATA_EOF
{
    "backup_type": "${BACKUP_TYPE}",
    "stanza": "${STANZA}",
    "timestamp": "${TIMESTAMP}",
    "host": "${PGHOST}",
    "database": "${PGDATABASE}",
    "row_counts": ${row_counts},
    "pgbackrest_info": ${info_output:-"{}"}
}
METADATA_EOF

    aws s3 cp "${metadata_file}" \
        "s3://${S3_BUCKET}/metadata/backup-${BACKUP_TYPE}-${TIMESTAMP}.json" \
        --sse aws:kms \
        --region "${S3_REGION}" \
        2>&1 | tee -a "${LOG_FILE}"

    rm -f "${metadata_file}"
    log "INFO: Backup statistics collected and uploaded"
}

cleanup_old_logs() {
    log "INFO: Cleaning up old log files..."
    find "${LOG_DIR}" -name "backup-*.log" -mtime +30 -delete 2>/dev/null || true
}

# =============================================================================
# Main
# =============================================================================

mkdir -p "${LOG_DIR}"

log "=============================================="
log "GreenLang PostgreSQL Backup"
log "Type: ${BACKUP_TYPE}"
log "Stanza: ${STANZA}"
log "Timestamp: ${TIMESTAMP}"
log "=============================================="

# Validate backup type
case "${BACKUP_TYPE}" in
    full|incr|diff)
        ;;
    *)
        log "ERROR: Invalid backup type '${BACKUP_TYPE}'. Must be: full, incr, diff"
        exit 1
        ;;
esac

check_prerequisites
run_backup "${BACKUP_TYPE}"
verify_backup
collect_stats
cleanup_old_logs

log "=============================================="
log "Backup completed successfully"
log "=============================================="
