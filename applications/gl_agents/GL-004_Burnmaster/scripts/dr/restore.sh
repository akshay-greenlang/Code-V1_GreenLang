#!/bin/bash
# GL-004 Burnmaster - Restoration Script
set -euo pipefail

AGENT_ID="gl-004"
AGENT_NAME="burnmaster"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
NAMESPACE="${NAMESPACE:-greenlang}"
RESTORE_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

restore_postgresql() {
    log "Restoring PostgreSQL from ${RESTORE_DATE}..."
    aws s3 cp s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${RESTORE_DATE}/ /tmp/restore/ --recursive
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME} --replicas=0 -n ${NAMESPACE}
    local pg_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres -o jsonpath='{.items[0].metadata.name}')
    kubectl cp /tmp/restore/ ${NAMESPACE}/${pg_pod}:/tmp/restore/
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- pg_restore -d greenlang_${AGENT_ID} -c /tmp/restore/*.tar.gz || true
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME} --replicas=2 -n ${NAMESPACE}
    log "PostgreSQL restoration completed"
}

main() {
    log "GL-004 Burnmaster Restore Starting (Date: ${RESTORE_DATE})"
    restore_postgresql
    kubectl wait --for=condition=ready pod -l app=${AGENT_ID}-${AGENT_NAME} -n ${NAMESPACE} --timeout=300s
    log "GL-004 Burnmaster Restore Complete"
}

main "$@"
