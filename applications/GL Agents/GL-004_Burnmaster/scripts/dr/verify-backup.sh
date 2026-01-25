#!/bin/bash
# GL-004 Burnmaster - Backup Verification Script
set -euo pipefail

AGENT_ID="gl-004"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
VERIFY_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

verify_backups() {
    log "Verifying backups for ${VERIFY_DATE}..."
    local status="PASS"

    if aws s3 ls s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${VERIFY_DATE}/ | grep -q ".tar.gz"; then
        log "PostgreSQL backup: FOUND"
    else
        log "PostgreSQL backup: MISSING"; status="FAIL"
    fi

    if aws s3 ls s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${VERIFY_DATE}/ | grep -q ".rdb"; then
        log "Redis backup: FOUND"
    else
        log "Redis backup: MISSING"; status="FAIL"
    fi

    log "Overall Status: ${status}"
    [ "${status}" == "PASS" ] && exit 0 || exit 1
}

verify_backups
