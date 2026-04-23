#!/bin/bash
# GL-003 UnifiedSteam - Automated Backup Script
set -euo pipefail

AGENT_ID="gl-003"
AGENT_NAME="unifiedsteam"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
NAMESPACE="${NAMESPACE:-greenlang}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y-%m-%d)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

backup_postgresql() {
    log "Starting PostgreSQL backup..."
    local pg_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres,role=master -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- pg_basebackup -D /tmp/backup -Ft -z -P
    kubectl cp ${NAMESPACE}/${pg_pod}:/tmp/backup /tmp/${AGENT_NAME}_pg_${TIMESTAMP}.tar.gz
    aws s3 cp /tmp/${AGENT_NAME}_pg_${TIMESTAMP}.tar.gz s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${DATE}/
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- rm -rf /tmp/backup
    log "PostgreSQL backup completed"
}

backup_redis() {
    log "Starting Redis backup..."
    local redis_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-redis -o jsonpath='{.items[0].metadata.name}')
    local redis_pass=$(kubectl get secret -n ${NAMESPACE} ${AGENT_ID}-redis-credentials -o jsonpath='{.data.password}' | base64 -d)
    kubectl exec -n ${NAMESPACE} ${redis_pod} -- redis-cli -a ${redis_pass} BGSAVE
    sleep 10
    kubectl cp ${NAMESPACE}/${redis_pod}:/data/dump.rdb /tmp/${AGENT_NAME}_redis_${TIMESTAMP}.rdb
    aws s3 cp /tmp/${AGENT_NAME}_redis_${TIMESTAMP}.rdb s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${DATE}/
    log "Redis backup completed"
}

backup_influxdb() {
    log "Starting InfluxDB backup..."
    local influx_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-influxdb -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n ${NAMESPACE} ${influx_pod} -- influx backup /tmp/influx-backup
    kubectl cp ${NAMESPACE}/${influx_pod}:/tmp/influx-backup /tmp/${AGENT_NAME}_influx_${TIMESTAMP}/
    aws s3 sync /tmp/${AGENT_NAME}_influx_${TIMESTAMP}/ s3://${BACKUP_BUCKET}/${AGENT_ID}/influxdb/${DATE}/
    log "InfluxDB backup completed"
}

main() {
    log "GL-003 UnifiedSteam Backup Starting"
    backup_postgresql
    backup_redis
    backup_influxdb 2>/dev/null || log "InfluxDB backup skipped"
    log "GL-003 UnifiedSteam Backup Complete"
}

main "$@"
