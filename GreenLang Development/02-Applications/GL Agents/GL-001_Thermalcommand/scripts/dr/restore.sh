#!/bin/bash
# ==============================================================================
# GL-001 ThermalCommand - Restoration Script
# ==============================================================================
# This script performs complete restoration of GL-001 from backups including:
# - PostgreSQL database restoration
# - Redis cache restoration
# - Configuration restoration
# - Kafka topic restoration (optional)
#
# Usage: ./restore.sh --date YYYY-MM-DD [--component all|postgres|redis|config|kafka]
#
# Environment Variables:
#   BACKUP_BUCKET     - S3 bucket for backup storage (required)
#   BACKUP_REGION     - AWS region for backup (default: us-east-1)
#   NAMESPACE         - Kubernetes namespace (default: greenlang)
#   SLACK_WEBHOOK     - Slack webhook URL for notifications
#   DRY_RUN           - Set to "true" to simulate without making changes
#
# ==============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/gl-001-restore.log"
TEMP_DIR="/tmp/gl-001-restore-${TIMESTAMP}"

# Default values
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
BACKUP_REGION="${BACKUP_REGION:-us-east-1}"
NAMESPACE="${NAMESPACE:-greenlang}"
DRY_RUN="${DRY_RUN:-false}"

# Backup paths
BACKUP_ROOT="s3://${BACKUP_BUCKET}/gl-001"

# Parse arguments
RESTORE_DATE=""
COMPONENT="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --date)
            RESTORE_DATE="$2"
            shift 2
            ;;
        --component)
            COMPONENT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "${RESTORE_DATE}" ]; then
    echo "Error: --date is required"
    echo "Usage: $0 --date YYYY-MM-DD [--component all|postgres|redis|config|kafka]"
    exit 1
fi

# ==============================================================================
# Logging Functions
# ==============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# ==============================================================================
# Notification Functions
# ==============================================================================

send_notification() {
    local status="$1"
    local message="$2"
    local color="${3:-#36a64f}"

    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"GL-001 Restore ${status}\",
                    \"text\": \"${message}\",
                    \"footer\": \"Restore Date: ${RESTORE_DATE}\"
                }]
            }" \
            "${SLACK_WEBHOOK}" || true
    fi
}

# ==============================================================================
# Cleanup Functions
# ==============================================================================

cleanup() {
    local exit_code=$?
    log_info "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}" || true

    if [ ${exit_code} -ne 0 ]; then
        log_error "Restore failed with exit code: ${exit_code}"
        send_notification "FAILED" "Restore failed. Check logs for details." "#ff0000"
    fi

    exit ${exit_code}
}

trap cleanup EXIT

# ==============================================================================
# Prerequisite Checks
# ==============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check required tools
    local required_tools=("kubectl" "aws" "psql" "redis-cli" "tar" "gzip")
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" &> /dev/null; then
            log_error "Required tool not found: ${tool}"
            exit 1
        fi
    done

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi

    # Check if backup exists for the date
    if ! aws s3 ls "${BACKUP_ROOT}/postgresql/full/${RESTORE_DATE}/" --region "${BACKUP_REGION}" &> /dev/null; then
        log_error "No backup found for date: ${RESTORE_DATE}"
        exit 1
    fi

    # Create temp directory
    mkdir -p "${TEMP_DIR}"

    log_success "Prerequisites check passed"
}

# ==============================================================================
# Confirmation
# ==============================================================================

confirm_restore() {
    log_warn "========================================"
    log_warn "         RESTORE CONFIRMATION           "
    log_warn "========================================"
    log_warn "You are about to restore GL-001 ThermalCommand"
    log_warn "  Restore Date: ${RESTORE_DATE}"
    log_warn "  Component(s): ${COMPONENT}"
    log_warn "  Namespace: ${NAMESPACE}"
    log_warn "  Dry Run: ${DRY_RUN}"
    log_warn ""
    log_warn "This will REPLACE current data with backup data!"
    log_warn "========================================"

    if [ "${DRY_RUN}" == "true" ]; then
        log_info "Dry run mode - no changes will be made"
        return 0
    fi

    read -p "Type 'RESTORE' to confirm: " confirmation
    if [ "${confirmation}" != "RESTORE" ]; then
        log_error "Restore cancelled by user"
        exit 1
    fi
}

# ==============================================================================
# Stop Application
# ==============================================================================

stop_application() {
    log_info "Stopping GL-001 ThermalCommand application..."

    if [ "${DRY_RUN}" == "true" ]; then
        log_info "[DRY RUN] Would scale deployment to 0"
        return 0
    fi

    # Scale down application
    kubectl scale deployment gl-001-thermalcommand \
        --replicas=0 \
        -n "${NAMESPACE}"

    # Wait for pods to terminate
    kubectl wait --for=delete pod \
        -l app=gl-001-thermalcommand \
        -n "${NAMESPACE}" \
        --timeout=120s || true

    log_success "Application stopped"
}

# ==============================================================================
# PostgreSQL Restoration
# ==============================================================================

restore_postgresql() {
    log_info "Starting PostgreSQL restoration..."

    # Find backup file
    local backup_file=$(aws s3 ls "${BACKUP_ROOT}/postgresql/full/${RESTORE_DATE}/" \
        --region "${BACKUP_REGION}" | grep ".tar.gz$" | awk '{print $4}' | head -1)

    if [ -z "${backup_file}" ]; then
        log_error "PostgreSQL backup file not found for ${RESTORE_DATE}"
        return 1
    fi

    log_info "Found backup: ${backup_file}"

    # Download backup
    log_info "Downloading PostgreSQL backup..."
    aws s3 cp "${BACKUP_ROOT}/postgresql/full/${RESTORE_DATE}/${backup_file}" \
        "${TEMP_DIR}/${backup_file}" \
        --region "${BACKUP_REGION}"

    # Verify checksum
    local checksum_file="${backup_file%.tar.gz}.sha256"
    if aws s3 ls "${BACKUP_ROOT}/postgresql/full/${RESTORE_DATE}/${checksum_file}" --region "${BACKUP_REGION}" &> /dev/null; then
        log_info "Verifying checksum..."
        aws s3 cp "${BACKUP_ROOT}/postgresql/full/${RESTORE_DATE}/${checksum_file}" \
            "${TEMP_DIR}/${checksum_file}" \
            --region "${BACKUP_REGION}"

        local expected_checksum=$(cat "${TEMP_DIR}/${checksum_file}")
        local actual_checksum=$(sha256sum "${TEMP_DIR}/${backup_file}" | awk '{print $1}')

        if [ "${expected_checksum}" != "${actual_checksum}" ]; then
            log_error "Checksum verification failed!"
            log_error "Expected: ${expected_checksum}"
            log_error "Actual: ${actual_checksum}"
            return 1
        fi
        log_success "Checksum verified"
    fi

    if [ "${DRY_RUN}" == "true" ]; then
        log_info "[DRY RUN] Would restore PostgreSQL from ${backup_file}"
        return 0
    fi

    # Get PostgreSQL pod
    local pg_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres \
        -o jsonpath='{.items[0].metadata.name}')

    # Extract backup
    mkdir -p "${TEMP_DIR}/pg_restore"
    tar -xzf "${TEMP_DIR}/${backup_file}" -C "${TEMP_DIR}/pg_restore"

    # Stop PostgreSQL
    log_info "Stopping PostgreSQL..."
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- \
        patronictl pause gl-001-postgres

    # Copy backup to pod
    log_info "Copying backup to pod..."
    kubectl cp "${TEMP_DIR}/pg_restore" "${NAMESPACE}/${pg_pod}:/tmp/restore/"

    # Restore database
    log_info "Restoring database..."
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- bash -c "
        cd /tmp/restore
        pg_restore -d greenlang_gl001 -c -j 4 base.tar.gz || true
    "

    # Resume Patroni
    log_info "Resuming Patroni..."
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- \
        patronictl resume gl-001-postgres

    # Cleanup
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- rm -rf /tmp/restore

    log_success "PostgreSQL restoration completed"
}

# ==============================================================================
# Redis Restoration
# ==============================================================================

restore_redis() {
    log_info "Starting Redis restoration..."

    # Find backup file
    local backup_file=$(aws s3 ls "${BACKUP_ROOT}/redis/rdb/${RESTORE_DATE}/" \
        --region "${BACKUP_REGION}" | grep ".rdb$" | awk '{print $4}' | head -1)

    if [ -z "${backup_file}" ]; then
        log_error "Redis backup file not found for ${RESTORE_DATE}"
        return 1
    fi

    log_info "Found backup: ${backup_file}"

    # Download backup
    log_info "Downloading Redis backup..."
    aws s3 cp "${BACKUP_ROOT}/redis/rdb/${RESTORE_DATE}/${backup_file}" \
        "${TEMP_DIR}/${backup_file}" \
        --region "${BACKUP_REGION}"

    if [ "${DRY_RUN}" == "true" ]; then
        log_info "[DRY RUN] Would restore Redis from ${backup_file}"
        return 0
    fi

    # Get Redis master pod
    local redis_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-redis,role=master \
        -o jsonpath='{.items[0].metadata.name}')

    # Get Redis password
    local redis_password=$(kubectl get secret -n "${NAMESPACE}" \
        gl-001-redis-credentials \
        -o jsonpath='{.data.password}' | base64 -d)

    # Stop Redis (scale to 0)
    log_info "Stopping Redis..."
    kubectl scale statefulset gl-001-redis-master --replicas=0 -n "${NAMESPACE}"
    kubectl wait --for=delete pod -l app=gl-001-redis,role=master \
        -n "${NAMESPACE}" --timeout=60s || true

    # Create restore pod
    log_info "Creating restore pod..."
    kubectl run redis-restore-pod \
        --image=redis:7-alpine \
        --restart=Never \
        -n "${NAMESPACE}" \
        --overrides='{
            "spec": {
                "containers": [{
                    "name": "redis-restore-pod",
                    "image": "redis:7-alpine",
                    "command": ["sleep", "3600"],
                    "volumeMounts": [{
                        "name": "data",
                        "mountPath": "/data"
                    }]
                }],
                "volumes": [{
                    "name": "data",
                    "persistentVolumeClaim": {
                        "claimName": "data-gl-001-redis-master-0"
                    }
                }]
            }
        }'

    # Wait for pod
    kubectl wait --for=condition=ready pod/redis-restore-pod \
        -n "${NAMESPACE}" --timeout=60s

    # Copy RDB file
    log_info "Copying RDB file..."
    kubectl cp "${TEMP_DIR}/${backup_file}" \
        "${NAMESPACE}/redis-restore-pod:/data/dump.rdb"

    # Delete restore pod
    kubectl delete pod redis-restore-pod -n "${NAMESPACE}"

    # Start Redis
    log_info "Starting Redis..."
    kubectl scale statefulset gl-001-redis-master --replicas=1 -n "${NAMESPACE}"
    kubectl wait --for=condition=ready pod -l app=gl-001-redis,role=master \
        -n "${NAMESPACE}" --timeout=120s

    # Verify
    log_info "Verifying Redis..."
    local key_count=$(kubectl exec -n "${NAMESPACE}" gl-001-redis-master-0 -- \
        redis-cli -a "${redis_password}" DBSIZE 2>/dev/null | awk '{print $2}')
    log_info "Redis key count: ${key_count}"

    log_success "Redis restoration completed"
}

# ==============================================================================
# Configuration Restoration
# ==============================================================================

restore_configuration() {
    log_info "Starting configuration restoration..."

    # Find config backup
    local config_file=$(aws s3 ls "${BACKUP_ROOT}/config/${RESTORE_DATE}/" \
        --region "${BACKUP_REGION}" | grep "config_.*\.tar\.gz$" | awk '{print $4}' | head -1)

    if [ -z "${config_file}" ]; then
        log_error "Configuration backup not found for ${RESTORE_DATE}"
        return 1
    fi

    log_info "Found config backup: ${config_file}"

    # Download
    aws s3 cp "${BACKUP_ROOT}/config/${RESTORE_DATE}/${config_file}" \
        "${TEMP_DIR}/${config_file}" \
        --region "${BACKUP_REGION}"

    # Extract
    tar -xzf "${TEMP_DIR}/${config_file}" -C "${TEMP_DIR}"

    if [ "${DRY_RUN}" == "true" ]; then
        log_info "[DRY RUN] Would restore configuration from ${config_file}"
        log_info "ConfigMaps:"
        cat "${TEMP_DIR}"/configmaps_*.yaml | head -50
        return 0
    fi

    # Restore ConfigMaps
    log_info "Restoring ConfigMaps..."
    kubectl apply -f "${TEMP_DIR}"/configmaps_*.yaml -n "${NAMESPACE}"

    # Restore Secrets (if decrypted or decrypt first)
    if ls "${TEMP_DIR}"/secrets_*.yaml 2>/dev/null; then
        log_info "Restoring Secrets..."
        kubectl apply -f "${TEMP_DIR}"/secrets_*.yaml -n "${NAMESPACE}"
    elif ls "${TEMP_DIR}"/secrets_*.yaml.enc 2>/dev/null; then
        log_info "Decrypting and restoring Secrets..."
        sops --decrypt "${TEMP_DIR}"/secrets_*.yaml.enc > "${TEMP_DIR}/secrets.yaml"
        kubectl apply -f "${TEMP_DIR}/secrets.yaml" -n "${NAMESPACE}"
    fi

    log_success "Configuration restoration completed"
}

# ==============================================================================
# Kafka Topic Restoration
# ==============================================================================

restore_kafka() {
    log_info "Starting Kafka topic restoration..."
    log_warn "Note: Kafka topic restoration is optional and may take a long time"

    local kafka_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-kafka \
        -o jsonpath='{.items[0].metadata.name}')

    if [ -z "${kafka_pod}" ]; then
        log_error "Kafka pod not found"
        return 1
    fi

    # List available topic backups
    local topics=$(aws s3 ls "${BACKUP_ROOT}/kafka/topics/${RESTORE_DATE}/" \
        --region "${BACKUP_REGION}" | grep ".json.gz$" | awk '{print $4}')

    for topic_file in ${topics}; do
        log_info "Restoring topic from: ${topic_file}"

        # Download
        aws s3 cp "${BACKUP_ROOT}/kafka/topics/${RESTORE_DATE}/${topic_file}" \
            "${TEMP_DIR}/${topic_file}" \
            --region "${BACKUP_REGION}"

        # Decompress
        gunzip "${TEMP_DIR}/${topic_file}"

        local json_file="${topic_file%.gz}"
        local topic_name=$(echo "${json_file}" | sed 's/kafka_//' | sed "s/_${RESTORE_DATE}.*\.json$//" | tr '_' '.')

        if [ "${DRY_RUN}" == "true" ]; then
            log_info "[DRY RUN] Would restore topic: ${topic_name}"
            continue
        fi

        # Restore messages
        log_info "Restoring messages to topic: ${topic_name}"
        kubectl exec -n "${NAMESPACE}" "${kafka_pod}" -- \
            kafka-console-producer \
            --bootstrap-server localhost:9092 \
            --topic "${topic_name}" \
            < "${TEMP_DIR}/${json_file}"
    done

    log_success "Kafka restoration completed"
}

# ==============================================================================
# Start Application
# ==============================================================================

start_application() {
    log_info "Starting GL-001 ThermalCommand application..."

    if [ "${DRY_RUN}" == "true" ]; then
        log_info "[DRY RUN] Would scale deployment to 3"
        return 0
    fi

    # Scale up application
    kubectl scale deployment gl-001-thermalcommand \
        --replicas=3 \
        -n "${NAMESPACE}"

    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod \
        -l app=gl-001-thermalcommand \
        -n "${NAMESPACE}" \
        --timeout=300s

    log_success "Application started"
}

# ==============================================================================
# Verification
# ==============================================================================

verify_restoration() {
    log_info "Verifying restoration..."

    if [ "${DRY_RUN}" == "true" ]; then
        log_info "[DRY RUN] Would verify restoration"
        return 0
    fi

    local verification_status=0

    # Check application health
    log_info "Checking application health..."
    local health=$(kubectl exec -n "${NAMESPACE}" \
        -l app=gl-001-thermalcommand \
        -- curl -s http://localhost:8000/api/v1/health 2>/dev/null | head -1)

    if echo "${health}" | grep -q "healthy"; then
        log_success "Application health check passed"
    else
        log_error "Application health check failed"
        verification_status=1
    fi

    # Check database
    log_info "Checking database..."
    local pg_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}')

    local table_count=$(kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- \
        psql -U postgres -d greenlang_gl001 -t -c \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';")

    log_info "Database table count: ${table_count}"

    # Check Redis
    log_info "Checking Redis..."
    local redis_password=$(kubectl get secret -n "${NAMESPACE}" \
        gl-001-redis-credentials \
        -o jsonpath='{.data.password}' | base64 -d)

    local redis_ping=$(kubectl exec -n "${NAMESPACE}" gl-001-redis-master-0 -- \
        redis-cli -a "${redis_password}" PING 2>/dev/null)

    if [ "${redis_ping}" == "PONG" ]; then
        log_success "Redis health check passed"
    else
        log_error "Redis health check failed"
        verification_status=1
    fi

    return ${verification_status}
}

# ==============================================================================
# Main Function
# ==============================================================================

main() {
    log_info "========================================="
    log_info "GL-001 ThermalCommand Restore Starting"
    log_info "Restore Date: ${RESTORE_DATE}"
    log_info "Component(s): ${COMPONENT}"
    log_info "Dry Run: ${DRY_RUN}"
    log_info "========================================="

    # Check prerequisites
    check_prerequisites

    # Confirm restore
    confirm_restore

    # Send notification
    send_notification "STARTED" "Restore initiated for ${RESTORE_DATE}" "#ffcc00"

    # Stop application (except for config-only restore)
    if [ "${COMPONENT}" != "config" ]; then
        stop_application
    fi

    # Perform restoration based on component
    case "${COMPONENT}" in
        all)
            restore_postgresql
            restore_redis
            restore_configuration
            ;;
        postgres)
            restore_postgresql
            ;;
        redis)
            restore_redis
            ;;
        config)
            restore_configuration
            ;;
        kafka)
            restore_kafka
            ;;
        *)
            log_error "Unknown component: ${COMPONENT}"
            exit 1
            ;;
    esac

    # Start application
    if [ "${COMPONENT}" != "config" ]; then
        start_application
    fi

    # Verify restoration
    verify_restoration

    log_info "========================================="
    log_success "GL-001 ThermalCommand Restore Complete"
    log_info "========================================="

    # Send success notification
    send_notification "SUCCESS" "Restore completed successfully for ${RESTORE_DATE}" "#36a64f"
}

# Run main function
main "$@"
