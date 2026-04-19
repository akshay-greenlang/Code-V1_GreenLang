#!/bin/bash
# ==============================================================================
# GL-002 Flameguard - Failover Script
# ==============================================================================
set -euo pipefail

AGENT_ID="gl-002"
AGENT_NAME="flameguard"
NAMESPACE="${NAMESPACE:-greenlang}"
OPERATION="${1:-help}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

failover_database() {
    log "Initiating PostgreSQL failover..."

    local pg_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- patronictl switchover --force

    log "PostgreSQL failover completed"
}

failover_redis() {
    log "Initiating Redis Sentinel failover..."

    local sentinel_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-redis-sentinel -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n ${NAMESPACE} ${sentinel_pod} -- redis-cli -p 26379 SENTINEL failover ${AGENT_ID}-redis

    log "Redis failover completed"
}

verify_status() {
    log "Checking ${AGENT_NAME} status..."

    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-${AGENT_NAME}
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-redis
}

case "${OPERATION}" in
    database) failover_database ;;
    redis) failover_redis ;;
    verify) verify_status ;;
    *)
        echo "Usage: $0 {database|redis|verify}"
        exit 1
        ;;
esac
