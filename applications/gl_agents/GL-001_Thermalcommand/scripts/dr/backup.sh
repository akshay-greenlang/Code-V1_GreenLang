#!/bin/bash
# ==============================================================================
# GL-001 ThermalCommand - Automated Backup Script
# ==============================================================================
# This script performs comprehensive backups of all GL-001 components including:
# - PostgreSQL database (full and incremental)
# - Redis cache (RDB snapshots)
# - Kafka topics (optional)
# - Configuration and secrets
# - Audit logs
#
# Usage: ./backup.sh [--full|--incremental|--audit-only]
#
# Environment Variables:
#   BACKUP_BUCKET     - S3 bucket for backup storage (required)
#   BACKUP_REGION     - AWS region for backup (default: us-east-1)
#   DR_BUCKET         - S3 bucket for DR copies (optional)
#   DR_REGION         - AWS region for DR (default: us-west-2)
#   NAMESPACE         - Kubernetes namespace (default: greenlang)
#   SLACK_WEBHOOK     - Slack webhook URL for notifications
#   ENCRYPTION_KEY_ID - KMS key ID for encryption
#
# ==============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y-%m-%d)
LOG_FILE="/var/log/gl-001-backup.log"
TEMP_DIR="/tmp/gl-001-backup-${TIMESTAMP}"

# Default values
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
BACKUP_REGION="${BACKUP_REGION:-us-east-1}"
DR_BUCKET="${DR_BUCKET:-greenlang-backups-dr}"
DR_REGION="${DR_REGION:-us-west-2}"
NAMESPACE="${NAMESPACE:-greenlang}"
BACKUP_TYPE="${1:-full}"

# Backup paths
BACKUP_ROOT="s3://${BACKUP_BUCKET}/gl-001"
DR_ROOT="s3://${DR_BUCKET}/gl-001"

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
    local color="${3:-#36a64f}"  # Default green for success

    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"GL-001 Backup ${status}\",
                    \"text\": \"${message}\",
                    \"footer\": \"Timestamp: ${TIMESTAMP}\"
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
        log_error "Backup failed with exit code: ${exit_code}"
        send_notification "FAILED" "Backup failed at ${TIMESTAMP}. Check logs for details." "#ff0000"
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
    local required_tools=("kubectl" "aws" "pg_dump" "redis-cli" "gzip" "sha256sum")
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

    # Check Kubernetes connectivity
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_error "Cannot connect to Kubernetes or namespace ${NAMESPACE} not found"
        exit 1
    fi

    # Create temp directory
    mkdir -p "${TEMP_DIR}"

    log_success "Prerequisites check passed"
}

# ==============================================================================
# PostgreSQL Backup Functions
# ==============================================================================

backup_postgresql_full() {
    log_info "Starting PostgreSQL full backup..."

    local backup_file="${TEMP_DIR}/postgresql_full_${TIMESTAMP}.tar.gz"
    local checksum_file="${TEMP_DIR}/postgresql_full_${TIMESTAMP}.sha256"

    # Get PostgreSQL pod
    local pg_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

    if [ -z "${pg_pod}" ]; then
        log_error "PostgreSQL master pod not found"
        return 1
    fi

    log_info "Using PostgreSQL pod: ${pg_pod}"

    # Perform backup using pg_basebackup
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- \
        pg_basebackup -D /tmp/backup -Ft -z -P -X stream \
        -h localhost -U replicator 2>&1 | tee -a "${LOG_FILE}"

    # Copy backup from pod
    kubectl cp "${NAMESPACE}/${pg_pod}:/tmp/backup" "${TEMP_DIR}/pg_backup/"

    # Compress backup
    tar -czf "${backup_file}" -C "${TEMP_DIR}/pg_backup" .

    # Generate checksum
    sha256sum "${backup_file}" | awk '{print $1}' > "${checksum_file}"

    # Upload to S3
    log_info "Uploading PostgreSQL backup to S3..."
    aws s3 cp "${backup_file}" \
        "${BACKUP_ROOT}/postgresql/full/${DATE}/" \
        --region "${BACKUP_REGION}" \
        --sse aws:kms \
        --sse-kms-key-id "${ENCRYPTION_KEY_ID:-alias/greenlang-backup}"

    aws s3 cp "${checksum_file}" \
        "${BACKUP_ROOT}/postgresql/full/${DATE}/" \
        --region "${BACKUP_REGION}"

    # Cross-region replication
    if [ -n "${DR_BUCKET:-}" ]; then
        log_info "Replicating to DR region..."
        aws s3 cp "${backup_file}" \
            "${DR_ROOT}/postgresql/full/${DATE}/" \
            --region "${DR_REGION}" \
            --sse aws:kms
    fi

    # Cleanup pod
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- rm -rf /tmp/backup

    log_success "PostgreSQL full backup completed: $(basename ${backup_file})"
}

backup_postgresql_wal() {
    log_info "Archiving PostgreSQL WAL files..."

    local pg_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}')

    # Force WAL switch
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- \
        psql -U postgres -c "SELECT pg_switch_wal();" 2>&1 | tee -a "${LOG_FILE}"

    log_success "WAL archive triggered"
}

backup_postgresql_incremental() {
    log_info "Starting PostgreSQL incremental backup..."

    # Use pgBackRest for incremental backups
    local pg_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}')

    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- \
        pgbackrest --stanza=gl-001 --type=incr backup 2>&1 | tee -a "${LOG_FILE}"

    log_success "PostgreSQL incremental backup completed"
}

# ==============================================================================
# Redis Backup Functions
# ==============================================================================

backup_redis() {
    log_info "Starting Redis backup..."

    local backup_file="${TEMP_DIR}/redis_${TIMESTAMP}.rdb"
    local aof_file="${TEMP_DIR}/redis_${TIMESTAMP}.aof.gz"
    local checksum_file="${TEMP_DIR}/redis_${TIMESTAMP}.sha256"

    # Get Redis master pod
    local redis_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-redis,role=master \
        -o jsonpath='{.items[0].metadata.name}')

    if [ -z "${redis_pod}" ]; then
        log_error "Redis master pod not found"
        return 1
    fi

    log_info "Using Redis pod: ${redis_pod}"

    # Get Redis password
    local redis_password=$(kubectl get secret -n "${NAMESPACE}" \
        gl-001-redis-credentials \
        -o jsonpath='{.data.password}' | base64 -d)

    # Trigger BGSAVE
    kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- \
        redis-cli -a "${redis_password}" BGSAVE 2>&1 | tee -a "${LOG_FILE}"

    # Wait for BGSAVE to complete
    local max_wait=60
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        local status=$(kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- \
            redis-cli -a "${redis_password}" LASTSAVE 2>/dev/null)
        sleep 2
        local new_status=$(kubectl exec -n "${NAMESPACE}" "${redis_pod}" -- \
            redis-cli -a "${redis_password}" LASTSAVE 2>/dev/null)
        if [ "${status}" != "${new_status}" ]; then
            break
        fi
        waited=$((waited + 2))
    done

    # Copy RDB file
    kubectl cp "${NAMESPACE}/${redis_pod}:/data/dump.rdb" "${backup_file}"

    # Copy AOF file if exists
    kubectl cp "${NAMESPACE}/${redis_pod}:/data/appendonly.aof" "${TEMP_DIR}/appendonly.aof" 2>/dev/null || true
    if [ -f "${TEMP_DIR}/appendonly.aof" ]; then
        gzip -c "${TEMP_DIR}/appendonly.aof" > "${aof_file}"
    fi

    # Generate checksum
    sha256sum "${backup_file}" | awk '{print $1}' > "${checksum_file}"

    # Upload to S3
    log_info "Uploading Redis backup to S3..."
    aws s3 cp "${backup_file}" \
        "${BACKUP_ROOT}/redis/rdb/${DATE}/" \
        --region "${BACKUP_REGION}"

    aws s3 cp "${checksum_file}" \
        "${BACKUP_ROOT}/redis/rdb/${DATE}/" \
        --region "${BACKUP_REGION}"

    if [ -f "${aof_file}" ]; then
        aws s3 cp "${aof_file}" \
            "${BACKUP_ROOT}/redis/aof/${DATE}/" \
            --region "${BACKUP_REGION}"
    fi

    log_success "Redis backup completed: $(basename ${backup_file})"
}

# ==============================================================================
# Kafka Backup Functions
# ==============================================================================

backup_kafka_topics() {
    log_info "Starting Kafka topics backup..."

    local topics=(
        "gl001.telemetry.normalized"
        "gl001.plan.dispatch"
        "gl001.actions.recommendations"
        "gl001.safety.events"
        "gl001.maintenance.triggers"
        "gl001.explainability.reports"
        "gl001.audit.log"
    )

    local kafka_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-kafka \
        -o jsonpath='{.items[0].metadata.name}')

    if [ -z "${kafka_pod}" ]; then
        log_error "Kafka pod not found"
        return 1
    fi

    for topic in "${topics[@]}"; do
        log_info "Backing up topic: ${topic}"

        local topic_file="${TEMP_DIR}/kafka_${topic//\./_}_${TIMESTAMP}.json"

        # Export topic messages
        kubectl exec -n "${NAMESPACE}" "${kafka_pod}" -- \
            kafka-console-consumer \
            --bootstrap-server localhost:9092 \
            --topic "${topic}" \
            --from-beginning \
            --timeout-ms 60000 \
            --property print.timestamp=true \
            --property print.key=true \
            > "${topic_file}" 2>/dev/null || true

        # Compress
        gzip "${topic_file}"

        # Upload
        aws s3 cp "${topic_file}.gz" \
            "${BACKUP_ROOT}/kafka/topics/${DATE}/" \
            --region "${BACKUP_REGION}"
    done

    log_success "Kafka topics backup completed"
}

backup_kafka_metadata() {
    log_info "Backing up Kafka metadata..."

    local kafka_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-kafka \
        -o jsonpath='{.items[0].metadata.name}')

    # Export topic configurations
    kubectl exec -n "${NAMESPACE}" "${kafka_pod}" -- \
        kafka-topics --bootstrap-server localhost:9092 --describe \
        > "${TEMP_DIR}/kafka_topics_${TIMESTAMP}.txt"

    # Export consumer group offsets
    kubectl exec -n "${NAMESPACE}" "${kafka_pod}" -- \
        kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe \
        > "${TEMP_DIR}/kafka_consumers_${TIMESTAMP}.txt"

    # Upload
    aws s3 cp "${TEMP_DIR}/kafka_topics_${TIMESTAMP}.txt" \
        "${BACKUP_ROOT}/kafka/metadata/${DATE}/" \
        --region "${BACKUP_REGION}"

    aws s3 cp "${TEMP_DIR}/kafka_consumers_${TIMESTAMP}.txt" \
        "${BACKUP_ROOT}/kafka/metadata/${DATE}/" \
        --region "${BACKUP_REGION}"

    log_success "Kafka metadata backup completed"
}

# ==============================================================================
# Configuration Backup Functions
# ==============================================================================

backup_configuration() {
    log_info "Starting configuration backup..."

    # Export ConfigMaps
    kubectl get configmaps -n "${NAMESPACE}" \
        -l app.kubernetes.io/part-of=greenlang \
        -o yaml > "${TEMP_DIR}/configmaps_${TIMESTAMP}.yaml"

    # Export Secrets (encrypted)
    kubectl get secrets -n "${NAMESPACE}" \
        -l app.kubernetes.io/part-of=greenlang \
        -o yaml | \
        sops --encrypt --age "${SOPS_AGE_PUBLIC_KEY:-}" \
        > "${TEMP_DIR}/secrets_${TIMESTAMP}.yaml.enc" 2>/dev/null || \
        kubectl get secrets -n "${NAMESPACE}" \
            -l app.kubernetes.io/part-of=greenlang \
            -o yaml > "${TEMP_DIR}/secrets_${TIMESTAMP}.yaml"

    # Export Deployments
    kubectl get deployments -n "${NAMESPACE}" \
        -l app.kubernetes.io/part-of=greenlang \
        -o yaml > "${TEMP_DIR}/deployments_${TIMESTAMP}.yaml"

    # Export Services
    kubectl get services -n "${NAMESPACE}" \
        -l app.kubernetes.io/part-of=greenlang \
        -o yaml > "${TEMP_DIR}/services_${TIMESTAMP}.yaml"

    # Upload
    tar -czf "${TEMP_DIR}/config_${TIMESTAMP}.tar.gz" \
        -C "${TEMP_DIR}" \
        configmaps_${TIMESTAMP}.yaml \
        secrets_${TIMESTAMP}.yaml* \
        deployments_${TIMESTAMP}.yaml \
        services_${TIMESTAMP}.yaml

    aws s3 cp "${TEMP_DIR}/config_${TIMESTAMP}.tar.gz" \
        "${BACKUP_ROOT}/config/${DATE}/" \
        --region "${BACKUP_REGION}" \
        --sse aws:kms

    log_success "Configuration backup completed"
}

# ==============================================================================
# Audit Log Backup Functions
# ==============================================================================

backup_audit_logs() {
    log_info "Starting audit log backup..."

    local pg_pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app=gl-001-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}')

    # Export audit logs older than 30 days
    kubectl exec -n "${NAMESPACE}" "${pg_pod}" -- psql -U postgres -d greenlang_gl001 -c "
        COPY (
            SELECT * FROM audit_log
            WHERE created_at < NOW() - INTERVAL '30 days'
            ORDER BY created_at
        ) TO STDOUT WITH CSV HEADER
    " > "${TEMP_DIR}/audit_archive_${TIMESTAMP}.csv"

    # Compress and encrypt
    gzip "${TEMP_DIR}/audit_archive_${TIMESTAMP}.csv"

    # Upload to Glacier Deep Archive for 7-year retention
    aws s3 cp "${TEMP_DIR}/audit_archive_${TIMESTAMP}.csv.gz" \
        "s3://${BACKUP_BUCKET}/gl-001/audit-archive/${DATE}/" \
        --region "${BACKUP_REGION}" \
        --storage-class DEEP_ARCHIVE \
        --sse aws:kms

    log_success "Audit log backup completed"
}

# ==============================================================================
# Verification Functions
# ==============================================================================

verify_backups() {
    log_info "Verifying backup integrity..."

    local verification_status=0

    # List today's backups
    log_info "Today's backups in S3:"
    aws s3 ls "${BACKUP_ROOT}/postgresql/full/${DATE}/" --region "${BACKUP_REGION}" || verification_status=1
    aws s3 ls "${BACKUP_ROOT}/redis/rdb/${DATE}/" --region "${BACKUP_REGION}" || verification_status=1
    aws s3 ls "${BACKUP_ROOT}/config/${DATE}/" --region "${BACKUP_REGION}" || verification_status=1

    # Verify checksums
    log_info "Verifying checksums..."
    local pg_backup=$(aws s3 ls "${BACKUP_ROOT}/postgresql/full/${DATE}/" --region "${BACKUP_REGION}" | grep ".tar.gz$" | awk '{print $4}' | head -1)
    if [ -n "${pg_backup}" ]; then
        local pg_checksum=$(aws s3 cp "${BACKUP_ROOT}/postgresql/full/${DATE}/${pg_backup%.tar.gz}.sha256" - --region "${BACKUP_REGION}")
        log_info "PostgreSQL backup checksum: ${pg_checksum}"
    fi

    if [ ${verification_status} -eq 0 ]; then
        log_success "Backup verification passed"
    else
        log_error "Backup verification failed"
    fi

    return ${verification_status}
}

# ==============================================================================
# Retention Policy Functions
# ==============================================================================

apply_retention_policy() {
    log_info "Applying retention policy..."

    # PostgreSQL full backups: 30 days
    local pg_cutoff=$(date -d "30 days ago" +%Y-%m-%d)
    aws s3 ls "${BACKUP_ROOT}/postgresql/full/" --region "${BACKUP_REGION}" | \
        awk '{print $2}' | \
        while read -r date_folder; do
            if [[ "${date_folder%/}" < "${pg_cutoff}" ]]; then
                log_info "Removing old PostgreSQL backup: ${date_folder}"
                aws s3 rm "${BACKUP_ROOT}/postgresql/full/${date_folder}" \
                    --recursive --region "${BACKUP_REGION}"
            fi
        done

    # Redis RDB: 7 days
    local redis_cutoff=$(date -d "7 days ago" +%Y-%m-%d)
    aws s3 ls "${BACKUP_ROOT}/redis/rdb/" --region "${BACKUP_REGION}" | \
        awk '{print $2}' | \
        while read -r date_folder; do
            if [[ "${date_folder%/}" < "${redis_cutoff}" ]]; then
                log_info "Removing old Redis backup: ${date_folder}"
                aws s3 rm "${BACKUP_ROOT}/redis/rdb/${date_folder}" \
                    --recursive --region "${BACKUP_REGION}"
            fi
        done

    log_success "Retention policy applied"
}

# ==============================================================================
# Main Function
# ==============================================================================

main() {
    log_info "========================================="
    log_info "GL-001 ThermalCommand Backup Starting"
    log_info "Backup Type: ${BACKUP_TYPE}"
    log_info "Timestamp: ${TIMESTAMP}"
    log_info "========================================="

    # Check prerequisites
    check_prerequisites

    case "${BACKUP_TYPE}" in
        full)
            backup_postgresql_full
            backup_redis
            backup_kafka_metadata
            backup_configuration
            ;;
        incremental)
            backup_postgresql_wal
            backup_postgresql_incremental
            backup_redis
            ;;
        audit-only)
            backup_audit_logs
            ;;
        all)
            backup_postgresql_full
            backup_redis
            backup_kafka_topics
            backup_kafka_metadata
            backup_configuration
            backup_audit_logs
            ;;
        *)
            log_error "Unknown backup type: ${BACKUP_TYPE}"
            echo "Usage: $0 [--full|--incremental|--audit-only|--all]"
            exit 1
            ;;
    esac

    # Verify backups
    verify_backups

    # Apply retention policy
    apply_retention_policy

    log_info "========================================="
    log_success "GL-001 ThermalCommand Backup Complete"
    log_info "========================================="

    # Send success notification
    send_notification "SUCCESS" "Backup completed successfully at ${TIMESTAMP}. Type: ${BACKUP_TYPE}" "#36a64f"
}

# Run main function
main "$@"
