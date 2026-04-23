#!/bin/bash
# ==============================================================================
# GL-002 Flameguard - Automated Backup Script
# ==============================================================================
set -euo pipefail

AGENT_ID="gl-002"
AGENT_NAME="flameguard"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
NAMESPACE="${NAMESPACE:-greenlang}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y-%m-%d)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# PostgreSQL Backup
backup_postgresql() {
    log "Starting PostgreSQL backup for ${AGENT_NAME}..."

    local pg_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres,role=master -o jsonpath='{.items[0].metadata.name}')

    kubectl exec -n ${NAMESPACE} ${pg_pod} -- pg_basebackup -D /tmp/backup -Ft -z -P
    kubectl cp ${NAMESPACE}/${pg_pod}:/tmp/backup /tmp/${AGENT_NAME}_pg_${TIMESTAMP}.tar.gz

    aws s3 cp /tmp/${AGENT_NAME}_pg_${TIMESTAMP}.tar.gz \
        s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${DATE}/

    kubectl exec -n ${NAMESPACE} ${pg_pod} -- rm -rf /tmp/backup
    log "PostgreSQL backup completed"
}

# Redis Backup
backup_redis() {
    log "Starting Redis backup for ${AGENT_NAME}..."

    local redis_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-redis -o jsonpath='{.items[0].metadata.name}')
    local redis_pass=$(kubectl get secret -n ${NAMESPACE} ${AGENT_ID}-redis-credentials -o jsonpath='{.data.password}' | base64 -d)

    kubectl exec -n ${NAMESPACE} ${redis_pod} -- redis-cli -a ${redis_pass} BGSAVE
    sleep 10
    kubectl cp ${NAMESPACE}/${redis_pod}:/data/dump.rdb /tmp/${AGENT_NAME}_redis_${TIMESTAMP}.rdb

    aws s3 cp /tmp/${AGENT_NAME}_redis_${TIMESTAMP}.rdb \
        s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${DATE}/

    log "Redis backup completed"
}

# Configuration Backup
backup_config() {
    log "Starting configuration backup for ${AGENT_NAME}..."

    kubectl get configmaps -n ${NAMESPACE} -l app=${AGENT_ID}-${AGENT_NAME} -o yaml > /tmp/${AGENT_NAME}_config_${TIMESTAMP}.yaml

    aws s3 cp /tmp/${AGENT_NAME}_config_${TIMESTAMP}.yaml \
        s3://${BACKUP_BUCKET}/${AGENT_ID}/config/${DATE}/

    log "Configuration backup completed"
}

# Main
main() {
    log "========================================="
    log "GL-002 Flameguard Backup Starting"
    log "========================================="

    backup_postgresql
    backup_redis
    backup_config

    log "========================================="
    log "GL-002 Flameguard Backup Complete"
    log "========================================="
}

main "$@"
