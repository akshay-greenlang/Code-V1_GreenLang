#!/bin/bash
# GL-004 Burnmaster - Failover Script
set -euo pipefail

AGENT_ID="gl-004"
AGENT_NAME="burnmaster"
NAMESPACE="${NAMESPACE:-greenlang}"
OPERATION="${1:-help}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

failover_database() {
    log "Initiating PostgreSQL failover..."
    local pg_pod=$(kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- patronictl switchover --force
    log "PostgreSQL failover completed"
}

verify_status() {
    log "Checking ${AGENT_NAME} status..."
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-${AGENT_NAME}
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-redis
}

case "${OPERATION}" in
    database) failover_database ;;
    verify) verify_status ;;
    *) echo "Usage: $0 {database|verify}"; exit 1 ;;
esac
