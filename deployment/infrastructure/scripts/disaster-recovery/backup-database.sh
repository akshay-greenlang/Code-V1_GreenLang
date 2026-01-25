#!/bin/bash
# =============================================================================
# GreenLang Database Backup Script
# Production-grade database backup with encryption and verification
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/greenlang/db-backup.log"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
BACKUP_PREFIX="${BACKUP_PREFIX:-database}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
KMS_KEY_ID="${KMS_KEY_ID:-alias/greenlang-backup-key}"
ENVIRONMENT="${ENVIRONMENT:-production}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
PAGERDUTY_ROUTING_KEY="${PAGERDUTY_ROUTING_KEY:-}"

# Database configuration
DB_HOST="${DB_HOST:-greenlang-production.cluster-xxxxx.us-east-1.rds.amazonaws.com}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-greenlang}"
DB_USER="${DB_USER:-greenlang_backup}"
DB_PASSWORD="${DB_PASSWORD:-}"
DB_SSL_MODE="${DB_SSL_MODE:-verify-full}"

# Timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="greenlang-${ENVIRONMENT}-${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$*"; }
log_warn() { log "WARN" "${YELLOW}$*${NC}"; }
log_error() { log "ERROR" "${RED}$*${NC}"; }
log_success() { log "SUCCESS" "${GREEN}$*${NC}"; }

# Cleanup function
cleanup() {
    local exit_code=$?
    log_info "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}" 2>/dev/null || true
    if [[ ${exit_code} -ne 0 ]]; then
        send_alert "FAILURE" "Database backup failed with exit code ${exit_code}"
    fi
    exit ${exit_code}
}

trap cleanup EXIT

# Create temp directory
TEMP_DIR=$(mktemp -d)
log_info "Created temporary directory: ${TEMP_DIR}"

# Send alert notification
send_alert() {
    local status=$1
    local message=$2

    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
        local color="danger"
        [[ "${status}" == "SUCCESS" ]] && color="good"
        [[ "${status}" == "WARNING" ]] && color="warning"

        curl -s -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"Database Backup ${status}\",
                    \"text\": \"${message}\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"${ENVIRONMENT}\", \"short\": true},
                        {\"title\": \"Database\", \"value\": \"${DB_NAME}\", \"short\": true},
                        {\"title\": \"Timestamp\", \"value\": \"${TIMESTAMP}\", \"short\": true}
                    ]
                }]
            }" || true
    fi

    # PagerDuty alert for failures
    if [[ -n "${PAGERDUTY_ROUTING_KEY}" && "${status}" == "FAILURE" ]]; then
        curl -s -X POST "https://events.pagerduty.com/v2/enqueue" \
            -H 'Content-type: application/json' \
            -d "{
                \"routing_key\": \"${PAGERDUTY_ROUTING_KEY}\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"Database Backup Failed: ${message}\",
                    \"severity\": \"critical\",
                    \"source\": \"greenlang-backup-${ENVIRONMENT}\",
                    \"custom_details\": {
                        \"environment\": \"${ENVIRONMENT}\",
                        \"database\": \"${DB_NAME}\",
                        \"timestamp\": \"${TIMESTAMP}\"
                    }
                }
            }" || true
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    for tool in pg_dump aws gzip sha256sum; do
        if ! command -v ${tool} &> /dev/null; then
            missing_tools+=("${tool}")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi

    # Check S3 bucket access
    if ! aws s3 ls "s3://${BACKUP_BUCKET}" &> /dev/null; then
        log_error "Cannot access S3 bucket: ${BACKUP_BUCKET}"
        exit 1
    fi

    # Check database connectivity
    export PGPASSWORD="${DB_PASSWORD}"
    if ! pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" &> /dev/null; then
        log_error "Cannot connect to database"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# Create database backup
create_backup() {
    log_info "Creating database backup..."

    local backup_file="${TEMP_DIR}/${BACKUP_NAME}.sql"
    local compressed_file="${backup_file}.gz"

    export PGPASSWORD="${DB_PASSWORD}"

    # Perform pg_dump with all data
    pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --format=plain \
        --no-owner \
        --no-acl \
        --clean \
        --if-exists \
        --verbose \
        --file="${backup_file}" \
        2>&1 | tee -a "${LOG_FILE}"

    if [[ ! -f "${backup_file}" ]]; then
        log_error "Backup file not created"
        exit 1
    fi

    local backup_size=$(stat -f%z "${backup_file}" 2>/dev/null || stat -c%s "${backup_file}")
    log_info "Backup size (uncompressed): $(numfmt --to=iec ${backup_size})"

    # Compress backup
    log_info "Compressing backup..."
    gzip -9 "${backup_file}"

    local compressed_size=$(stat -f%z "${compressed_file}" 2>/dev/null || stat -c%s "${compressed_file}")
    log_info "Backup size (compressed): $(numfmt --to=iec ${compressed_size})"

    # Calculate checksum
    local checksum=$(sha256sum "${compressed_file}" | awk '{print $1}')
    echo "${checksum}" > "${compressed_file}.sha256"
    log_info "Checksum: ${checksum}"

    echo "${compressed_file}"
}

# Upload to S3 with encryption
upload_backup() {
    local backup_file=$1
    local s3_path="s3://${BACKUP_BUCKET}/${BACKUP_PREFIX}/${ENVIRONMENT}/${BACKUP_NAME}.sql.gz"

    log_info "Uploading backup to S3: ${s3_path}"

    # Upload with server-side encryption
    aws s3 cp "${backup_file}" "${s3_path}" \
        --sse aws:kms \
        --sse-kms-key-id "${KMS_KEY_ID}" \
        --storage-class STANDARD_IA \
        --metadata "environment=${ENVIRONMENT},timestamp=${TIMESTAMP},database=${DB_NAME}" \
        --only-show-errors

    # Upload checksum
    aws s3 cp "${backup_file}.sha256" "${s3_path}.sha256" \
        --sse aws:kms \
        --sse-kms-key-id "${KMS_KEY_ID}" \
        --only-show-errors

    # Verify upload
    if aws s3 ls "${s3_path}" &> /dev/null; then
        log_success "Backup uploaded successfully: ${s3_path}"
    else
        log_error "Failed to verify backup upload"
        exit 1
    fi

    echo "${s3_path}"
}

# Verify backup integrity
verify_backup() {
    local s3_path=$1

    log_info "Verifying backup integrity..."

    local verify_dir="${TEMP_DIR}/verify"
    mkdir -p "${verify_dir}"

    # Download and verify checksum
    aws s3 cp "${s3_path}" "${verify_dir}/backup.sql.gz" --only-show-errors
    aws s3 cp "${s3_path}.sha256" "${verify_dir}/backup.sql.gz.sha256" --only-show-errors

    local expected_checksum=$(cat "${verify_dir}/backup.sql.gz.sha256")
    local actual_checksum=$(sha256sum "${verify_dir}/backup.sql.gz" | awk '{print $1}')

    if [[ "${expected_checksum}" != "${actual_checksum}" ]]; then
        log_error "Checksum verification failed!"
        log_error "Expected: ${expected_checksum}"
        log_error "Actual: ${actual_checksum}"
        exit 1
    fi

    log_success "Checksum verification passed"

    # Test decompression
    if ! gzip -t "${verify_dir}/backup.sql.gz"; then
        log_error "Backup file is corrupted (gzip test failed)"
        exit 1
    fi

    log_success "Backup verification completed successfully"
}

# Create RDS snapshot
create_rds_snapshot() {
    log_info "Creating RDS snapshot..."

    local snapshot_id="${BACKUP_NAME}"
    local db_instance_id="${RDS_INSTANCE_ID:-greenlang-production}"

    aws rds create-db-snapshot \
        --db-instance-identifier "${db_instance_id}" \
        --db-snapshot-identifier "${snapshot_id}" \
        --tags "Key=Environment,Value=${ENVIRONMENT}" \
               "Key=CreatedBy,Value=backup-script" \
               "Key=Timestamp,Value=${TIMESTAMP}"

    log_info "Waiting for RDS snapshot to complete..."
    aws rds wait db-snapshot-available \
        --db-snapshot-identifier "${snapshot_id}"

    log_success "RDS snapshot created: ${snapshot_id}"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than ${RETENTION_DAYS} days..."

    local cutoff_date=$(date -d "${RETENTION_DAYS} days ago" +%Y%m%d 2>/dev/null || \
                       date -v-${RETENTION_DAYS}d +%Y%m%d)

    # List and delete old S3 backups
    aws s3 ls "s3://${BACKUP_BUCKET}/${BACKUP_PREFIX}/${ENVIRONMENT}/" | \
    while read -r line; do
        local file_date=$(echo "${line}" | grep -oP 'greenlang-\w+-\K\d{8}' || true)
        if [[ -n "${file_date}" && "${file_date}" < "${cutoff_date}" ]]; then
            local file_name=$(echo "${line}" | awk '{print $4}')
            log_info "Deleting old backup: ${file_name}"
            aws s3 rm "s3://${BACKUP_BUCKET}/${BACKUP_PREFIX}/${ENVIRONMENT}/${file_name}" --only-show-errors
        fi
    done

    # Cleanup old RDS snapshots
    local db_instance_id="${RDS_INSTANCE_ID:-greenlang-production}"
    aws rds describe-db-snapshots \
        --db-instance-identifier "${db_instance_id}" \
        --query "DBSnapshots[?SnapshotCreateTime<='${cutoff_date}'].DBSnapshotIdentifier" \
        --output text | tr '\t' '\n' | \
    while read -r snapshot_id; do
        if [[ -n "${snapshot_id}" && "${snapshot_id}" != "None" ]]; then
            log_info "Deleting old RDS snapshot: ${snapshot_id}"
            aws rds delete-db-snapshot \
                --db-snapshot-identifier "${snapshot_id}" 2>/dev/null || true
        fi
    done

    log_success "Cleanup completed"
}

# Record backup metadata
record_metadata() {
    local s3_path=$1
    local backup_size=$2

    local metadata_file="${TEMP_DIR}/backup-metadata.json"

    cat > "${metadata_file}" <<EOF
{
    "backup_id": "${BACKUP_NAME}",
    "timestamp": "${TIMESTAMP}",
    "environment": "${ENVIRONMENT}",
    "database": "${DB_NAME}",
    "s3_path": "${s3_path}",
    "size_bytes": ${backup_size},
    "retention_days": ${RETENTION_DAYS},
    "kms_key": "${KMS_KEY_ID}",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    # Upload metadata
    aws s3 cp "${metadata_file}" \
        "s3://${BACKUP_BUCKET}/${BACKUP_PREFIX}/${ENVIRONMENT}/metadata/${BACKUP_NAME}.json" \
        --sse aws:kms \
        --sse-kms-key-id "${KMS_KEY_ID}" \
        --only-show-errors

    log_info "Backup metadata recorded"
}

# Main execution
main() {
    log_info "=========================================="
    log_info "GreenLang Database Backup - ${ENVIRONMENT}"
    log_info "=========================================="
    log_info "Timestamp: ${TIMESTAMP}"
    log_info "Database: ${DB_HOST}:${DB_PORT}/${DB_NAME}"
    log_info "Backup bucket: ${BACKUP_BUCKET}"

    # Run backup steps
    check_prerequisites

    local backup_file=$(create_backup)
    local backup_size=$(stat -f%z "${backup_file}" 2>/dev/null || stat -c%s "${backup_file}")

    local s3_path=$(upload_backup "${backup_file}")
    verify_backup "${s3_path}"

    # Create RDS snapshot (optional, can be disabled)
    if [[ "${CREATE_RDS_SNAPSHOT:-true}" == "true" ]]; then
        create_rds_snapshot
    fi

    record_metadata "${s3_path}" "${backup_size}"
    cleanup_old_backups

    log_info "=========================================="
    log_success "Backup completed successfully!"
    log_info "S3 Path: ${s3_path}"
    log_info "Size: $(numfmt --to=iec ${backup_size})"
    log_info "=========================================="

    send_alert "SUCCESS" "Database backup completed successfully. Size: $(numfmt --to=iec ${backup_size})"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --bucket|-b)
            BACKUP_BUCKET="$2"
            shift 2
            ;;
        --retention|-r)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --no-rds-snapshot)
            CREATE_RDS_SNAPSHOT="false"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -e, --environment    Environment name (default: production)"
            echo "  -b, --bucket         S3 bucket for backups"
            echo "  -r, --retention      Retention period in days (default: 30)"
            echo "  --no-rds-snapshot    Skip RDS snapshot creation"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$(dirname "${LOG_FILE}")"

# Run main
main "$@"
