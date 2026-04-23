#!/bin/bash
# ==============================================================================
# GL-002 Flameguard - Backup Verification Script
# ==============================================================================
set -euo pipefail

AGENT_ID="gl-002"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
VERIFY_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

verify_backups() {
    log "Verifying backups for ${VERIFY_DATE}..."

    local status="PASS"

    # Check PostgreSQL backup
    if aws s3 ls s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${VERIFY_DATE}/ | grep -q ".tar.gz"; then
        log "PostgreSQL backup: FOUND"
    else
        log "PostgreSQL backup: MISSING"
        status="FAIL"
    fi

    # Check Redis backup
    if aws s3 ls s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${VERIFY_DATE}/ | grep -q ".rdb"; then
        log "Redis backup: FOUND"
    else
        log "Redis backup: MISSING"
        status="FAIL"
    fi

    # Check Config backup
    if aws s3 ls s3://${BACKUP_BUCKET}/${AGENT_ID}/config/${VERIFY_DATE}/ | grep -q ".yaml"; then
        log "Config backup: FOUND"
    else
        log "Config backup: MISSING"
        status="FAIL"
    fi

    log "Overall Status: ${status}"
    [ "${status}" == "PASS" ] && exit 0 || exit 1
}

verify_backups
