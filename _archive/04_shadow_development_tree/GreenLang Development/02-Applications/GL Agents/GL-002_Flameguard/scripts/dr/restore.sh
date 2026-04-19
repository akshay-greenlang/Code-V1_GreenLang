#!/bin/bash
# ==============================================================================
# GL-002 Flameguard - Restoration Script
# ==============================================================================
set -euo pipefail

AGENT_ID="gl-002"
AGENT_NAME="flameguard"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
NAMESPACE="${NAMESPACE:-greenlang}"
RESTORE_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

restore_postgresql() {
    log "Restoring PostgreSQL from ${RESTORE_DATE}..."

    # Download backup
    aws s3 cp s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${RESTORE_DATE}/ /tmp/restore/ --recursive

    # Stop application
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME} --replicas=0 -n ${NAMESPACE}

    # Restore
    local pg_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres -o jsonpath='{.items[0].metadata.name}')
    kubectl cp /tmp/restore/ ${NAMESPACE}/${pg_pod}:/tmp/restore/
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- pg_restore -d greenlang_${AGENT_ID} -c /tmp/restore/*.tar.gz || true

    # Start application
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME} --replicas=2 -n ${NAMESPACE}

    log "PostgreSQL restoration completed"
}

restore_redis() {
    log "Restoring Redis from ${RESTORE_DATE}..."

    aws s3 cp s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${RESTORE_DATE}/ /tmp/restore/ --recursive

    local redis_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-redis -o jsonpath='{.items[0].metadata.name}')
    kubectl cp /tmp/restore/*.rdb ${NAMESPACE}/${redis_pod}:/data/dump.rdb
    kubectl delete pod ${redis_pod} -n ${NAMESPACE}

    log "Redis restoration completed"
}

main() {
    log "========================================="
    log "GL-002 Flameguard Restore Starting"
    log "Restore Date: ${RESTORE_DATE}"
    log "========================================="

    restore_postgresql
    restore_redis

    kubectl wait --for=condition=ready pod -l app=${AGENT_ID}-${AGENT_NAME} -n ${NAMESPACE} --timeout=300s

    log "========================================="
    log "GL-002 Flameguard Restore Complete"
    log "========================================="
}

main "$@"
