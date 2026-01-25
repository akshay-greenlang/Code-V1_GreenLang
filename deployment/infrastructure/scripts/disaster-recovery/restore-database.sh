#!/bin/bash
# =============================================================================
# GreenLang Database Restore Script
# Production-grade database restoration with verification
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/greenlang/db-restore.log"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
BACKUP_PREFIX="${BACKUP_PREFIX:-database}"
ENVIRONMENT="${ENVIRONMENT:-production}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
PAGERDUTY_ROUTING_KEY="${PAGERDUTY_ROUTING_KEY:-}"

# Database configuration
DB_HOST="${DB_HOST:-greenlang-production.cluster-xxxxx.us-east-1.rds.amazonaws.com}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-greenlang}"
DB_USER="${DB_USER:-greenlang_admin}"
DB_PASSWORD="${DB_PASSWORD:-}"
DB_SSL_MODE="${DB_SSL_MODE:-verify-full}"

# Restore options
RESTORE_MODE="${RESTORE_MODE:-verify}"  # verify, dry-run, execute
BACKUP_ID=""
POINT_IN_TIME=""
TARGET_DB="${TARGET_DB:-}"

# Timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
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
log_step() { log "STEP" "${CYAN}>>> $*${NC}"; }

# Cleanup
cleanup() {
    local exit_code=$?
    log_info "Cleaning up temporary files..."
    rm -rf "${TEMP_DIR}" 2>/dev/null || true

    if [[ ${exit_code} -ne 0 ]]; then
        send_alert "FAILURE" "Database restore failed with exit code ${exit_code}"
    fi
    exit ${exit_code}
}

trap cleanup EXIT

TEMP_DIR=$(mktemp -d)
log_info "Created temporary directory: ${TEMP_DIR}"

# Alert notification
send_alert() {
    local status=$1
    local message=$2

    if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
        local color="danger"
        [[ "${status}" == "SUCCESS" ]] && color="good"
        [[ "${status}" == "WARNING" ]] && color="warning"

        curl -s -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"Database Restore ${status}\",
                    \"text\": \"${message}\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"${ENVIRONMENT}\", \"short\": true},
                        {\"title\": \"Database\", \"value\": \"${DB_NAME}\", \"short\": true},
                        {\"title\": \"Mode\", \"value\": \"${RESTORE_MODE}\", \"short\": true}
                    ]
                }]
            }" || true
    fi

    if [[ -n "${PAGERDUTY_ROUTING_KEY}" && "${status}" == "FAILURE" ]]; then
        curl -s -X POST "https://events.pagerduty.com/v2/enqueue" \
            -H 'Content-type: application/json' \
            -d "{
                \"routing_key\": \"${PAGERDUTY_ROUTING_KEY}\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"Database Restore Failed: ${message}\",
                    \"severity\": \"critical\",
                    \"source\": \"greenlang-restore-${ENVIRONMENT}\"
                }
            }" || true
    fi
}

# List available backups
list_backups() {
    log_step "Listing available backups..."

    echo -e "\n${CYAN}=== S3 Backups ===${NC}"
    aws s3 ls "s3://${BACKUP_BUCKET}/${BACKUP_PREFIX}/${ENVIRONMENT}/" \
        --human-readable | grep ".sql.gz$" | tail -20

    echo -e "\n${CYAN}=== RDS Snapshots ===${NC}"
    aws rds describe-db-snapshots \
        --db-instance-identifier "${RDS_INSTANCE_ID:-greenlang-production}" \
        --query 'DBSnapshots[*].[DBSnapshotIdentifier,SnapshotCreateTime,Status]' \
        --output table 2>/dev/null || echo "No RDS snapshots found"

    echo -e "\n${CYAN}=== Point-in-Time Recovery Window ===${NC}"
    aws rds describe-db-instances \
        --db-instance-identifier "${RDS_INSTANCE_ID:-greenlang-production}" \
        --query 'DBInstances[0].[LatestRestorableTime,EarliestRestorableTime]' \
        --output text 2>/dev/null || echo "PITR info not available"
}

# Download backup from S3
download_backup() {
    local backup_id=$1
    local backup_file="${TEMP_DIR}/${backup_id}.sql.gz"

    log_step "Downloading backup: ${backup_id}"

    local s3_path="s3://${BACKUP_BUCKET}/${BACKUP_PREFIX}/${ENVIRONMENT}/${backup_id}.sql.gz"

    # Download backup
    aws s3 cp "${s3_path}" "${backup_file}" --only-show-errors

    # Download and verify checksum
    aws s3 cp "${s3_path}.sha256" "${backup_file}.sha256" --only-show-errors

    local expected_checksum=$(cat "${backup_file}.sha256")
    local actual_checksum=$(sha256sum "${backup_file}" | awk '{print $1}')

    if [[ "${expected_checksum}" != "${actual_checksum}" ]]; then
        log_error "Checksum verification failed!"
        exit 1
    fi

    log_success "Backup downloaded and verified"
    echo "${backup_file}"
}

# Pre-restore checks
pre_restore_checks() {
    log_step "Running pre-restore checks..."

    # Check database connectivity
    export PGPASSWORD="${DB_PASSWORD}"
    if ! pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" &> /dev/null; then
        log_error "Cannot connect to database server"
        exit 1
    fi

    # Check target database
    local target_db="${TARGET_DB:-${DB_NAME}}"
    local db_exists=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
        -t -c "SELECT 1 FROM pg_database WHERE datname='${target_db}'" 2>/dev/null | tr -d ' ')

    if [[ "${db_exists}" == "1" && "${RESTORE_MODE}" == "execute" ]]; then
        log_warn "Target database '${target_db}' exists"

        if [[ "${FORCE_OVERWRITE:-false}" != "true" ]]; then
            read -p "Database exists. Overwrite? (yes/no): " confirm
            if [[ "${confirm}" != "yes" ]]; then
                log_error "Restore cancelled by user"
                exit 1
            fi
        fi
    fi

    # Check active connections
    local active_connections=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
        -t -c "SELECT count(*) FROM pg_stat_activity WHERE datname='${target_db}'" 2>/dev/null | tr -d ' ')

    if [[ "${active_connections:-0}" -gt 0 ]]; then
        log_warn "Database has ${active_connections} active connections"

        if [[ "${TERMINATE_CONNECTIONS:-false}" == "true" ]]; then
            log_info "Terminating existing connections..."
            psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
                -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='${target_db}' AND pid <> pg_backend_pid()"
        fi
    fi

    log_success "Pre-restore checks passed"
}

# Create pre-restore backup
create_pre_restore_backup() {
    log_step "Creating pre-restore backup..."

    local target_db="${TARGET_DB:-${DB_NAME}}"
    local backup_file="${TEMP_DIR}/pre-restore-${target_db}-${TIMESTAMP}.sql.gz"

    export PGPASSWORD="${DB_PASSWORD}"

    pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${target_db}" \
        --format=plain \
        --no-owner \
        --no-acl \
        2>/dev/null | gzip > "${backup_file}" || {
            log_warn "Could not create pre-restore backup (database may not exist)"
            return 0
        }

    # Upload to S3
    aws s3 cp "${backup_file}" \
        "s3://${BACKUP_BUCKET}/${BACKUP_PREFIX}/${ENVIRONMENT}/pre-restore/$(basename ${backup_file})" \
        --sse aws:kms \
        --only-show-errors

    log_success "Pre-restore backup created"
}

# Restore from S3 backup
restore_from_s3() {
    local backup_file=$1
    local target_db="${TARGET_DB:-${DB_NAME}}"

    log_step "Restoring database from S3 backup..."

    export PGPASSWORD="${DB_PASSWORD}"

    # Decompress backup
    local sql_file="${TEMP_DIR}/restore.sql"
    gunzip -c "${backup_file}" > "${sql_file}"

    if [[ "${RESTORE_MODE}" == "dry-run" ]]; then
        log_info "[DRY-RUN] Would restore to database: ${target_db}"
        log_info "[DRY-RUN] SQL file size: $(du -h ${sql_file} | cut -f1)"
        log_info "[DRY-RUN] First 50 lines:"
        head -50 "${sql_file}"
        return 0
    fi

    # Drop and recreate database
    if [[ "${DROP_DATABASE:-true}" == "true" ]]; then
        log_info "Dropping existing database..."
        psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
            -c "DROP DATABASE IF EXISTS ${target_db}"

        log_info "Creating new database..."
        psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d postgres \
            -c "CREATE DATABASE ${target_db} WITH OWNER ${DB_USER}"
    fi

    # Restore database
    log_info "Restoring data..."
    psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${target_db}" \
        --single-transaction \
        --set ON_ERROR_STOP=on \
        -f "${sql_file}" 2>&1 | tee -a "${LOG_FILE}"

    log_success "Database restored from S3 backup"
}

# Restore from RDS snapshot
restore_from_rds_snapshot() {
    local snapshot_id=$1
    local new_instance_id="${RDS_RESTORE_INSTANCE:-greenlang-restored-${TIMESTAMP}}"

    log_step "Restoring from RDS snapshot: ${snapshot_id}"

    if [[ "${RESTORE_MODE}" == "dry-run" ]]; then
        log_info "[DRY-RUN] Would restore RDS snapshot: ${snapshot_id}"
        log_info "[DRY-RUN] New instance: ${new_instance_id}"
        return 0
    fi

    # Get original instance configuration
    local source_instance="${RDS_INSTANCE_ID:-greenlang-production}"
    local instance_class=$(aws rds describe-db-instances \
        --db-instance-identifier "${source_instance}" \
        --query 'DBInstances[0].DBInstanceClass' \
        --output text 2>/dev/null || echo "db.r6g.large")

    local vpc_security_groups=$(aws rds describe-db-instances \
        --db-instance-identifier "${source_instance}" \
        --query 'DBInstances[0].VpcSecurityGroups[*].VpcSecurityGroupId' \
        --output text 2>/dev/null || echo "")

    local subnet_group=$(aws rds describe-db-instances \
        --db-instance-identifier "${source_instance}" \
        --query 'DBInstances[0].DBSubnetGroup.DBSubnetGroupName' \
        --output text 2>/dev/null || echo "greenlang-production")

    # Restore from snapshot
    aws rds restore-db-instance-from-db-snapshot \
        --db-instance-identifier "${new_instance_id}" \
        --db-snapshot-identifier "${snapshot_id}" \
        --db-instance-class "${instance_class}" \
        --db-subnet-group-name "${subnet_group}" \
        --vpc-security-group-ids ${vpc_security_groups} \
        --no-publicly-accessible \
        --tags "Key=Environment,Value=${ENVIRONMENT}" \
               "Key=RestoredFrom,Value=${snapshot_id}" \
               "Key=RestoredAt,Value=${TIMESTAMP}"

    log_info "Waiting for RDS instance to become available..."
    aws rds wait db-instance-available \
        --db-instance-identifier "${new_instance_id}"

    # Get new endpoint
    local new_endpoint=$(aws rds describe-db-instances \
        --db-instance-identifier "${new_instance_id}" \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)

    log_success "RDS instance restored: ${new_instance_id}"
    log_info "New endpoint: ${new_endpoint}"

    echo "${new_endpoint}"
}

# Point-in-time recovery
restore_point_in_time() {
    local restore_time=$1
    local new_instance_id="${RDS_RESTORE_INSTANCE:-greenlang-pitr-${TIMESTAMP}}"

    log_step "Restoring to point-in-time: ${restore_time}"

    if [[ "${RESTORE_MODE}" == "dry-run" ]]; then
        log_info "[DRY-RUN] Would restore to: ${restore_time}"
        log_info "[DRY-RUN] New instance: ${new_instance_id}"
        return 0
    fi

    local source_instance="${RDS_INSTANCE_ID:-greenlang-production}"

    aws rds restore-db-instance-to-point-in-time \
        --source-db-instance-identifier "${source_instance}" \
        --target-db-instance-identifier "${new_instance_id}" \
        --restore-time "${restore_time}" \
        --use-latest-restorable-time false \
        --no-publicly-accessible \
        --tags "Key=Environment,Value=${ENVIRONMENT}" \
               "Key=RestoreType,Value=point-in-time" \
               "Key=RestoreTime,Value=${restore_time}"

    log_info "Waiting for PITR restore to complete..."
    aws rds wait db-instance-available \
        --db-instance-identifier "${new_instance_id}"

    local new_endpoint=$(aws rds describe-db-instances \
        --db-instance-identifier "${new_instance_id}" \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)

    log_success "Point-in-time restore completed"
    log_info "New endpoint: ${new_endpoint}"

    echo "${new_endpoint}"
}

# Verify restored database
verify_restore() {
    local db_host="${1:-${DB_HOST}}"
    local target_db="${TARGET_DB:-${DB_NAME}}"

    log_step "Verifying restored database..."

    export PGPASSWORD="${DB_PASSWORD}"

    # Check connection
    if ! pg_isready -h "${db_host}" -p "${DB_PORT}" -U "${DB_USER}" &> /dev/null; then
        log_error "Cannot connect to restored database"
        return 1
    fi

    # Check tables
    local table_count=$(psql -h "${db_host}" -p "${DB_PORT}" -U "${DB_USER}" -d "${target_db}" \
        -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'" | tr -d ' ')

    log_info "Tables in database: ${table_count}"

    # Check critical tables
    local critical_tables=("users" "organizations" "carbon_calculations" "emissions_data")
    for table in "${critical_tables[@]}"; do
        local row_count=$(psql -h "${db_host}" -p "${DB_PORT}" -U "${DB_USER}" -d "${target_db}" \
            -t -c "SELECT count(*) FROM ${table}" 2>/dev/null | tr -d ' ' || echo "0")
        log_info "Table '${table}': ${row_count} rows"
    done

    # Run integrity checks
    log_info "Running database integrity checks..."
    psql -h "${db_host}" -p "${DB_PORT}" -U "${DB_USER}" -d "${target_db}" \
        -c "SELECT schemaname, tablename, n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC LIMIT 10"

    log_success "Database verification completed"
}

# Switch traffic to restored database
switch_traffic() {
    local new_endpoint=$1

    log_step "Switching traffic to restored database..."

    if [[ "${RESTORE_MODE}" == "dry-run" ]]; then
        log_info "[DRY-RUN] Would switch traffic to: ${new_endpoint}"
        return 0
    fi

    # Update Kubernetes secret
    kubectl create secret generic greenlang-db-connection \
        -n "${K8S_NAMESPACE:-greenlang-production}" \
        --from-literal=host="${new_endpoint}" \
        --from-literal=port="${DB_PORT}" \
        --from-literal=database="${TARGET_DB:-${DB_NAME}}" \
        --from-literal=username="${DB_USER}" \
        --from-literal=password="${DB_PASSWORD}" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Restart application pods to pick up new connection
    kubectl rollout restart deployment/greenlang-api-blue \
        -n "${K8S_NAMESPACE:-greenlang-production}"

    kubectl rollout status deployment/greenlang-api-blue \
        -n "${K8S_NAMESPACE:-greenlang-production}" \
        --timeout=300s

    log_success "Traffic switched to restored database"
}

# Main
main() {
    log_info "=========================================="
    log_info "GreenLang Database Restore - ${ENVIRONMENT}"
    log_info "=========================================="
    log_info "Mode: ${RESTORE_MODE}"
    log_info "Timestamp: ${TIMESTAMP}"

    case "${COMMAND:-restore}" in
        list)
            list_backups
            ;;

        restore)
            if [[ -z "${BACKUP_ID}" && -z "${POINT_IN_TIME}" && -z "${SNAPSHOT_ID}" ]]; then
                log_error "Must specify --backup-id, --snapshot-id, or --point-in-time"
                exit 1
            fi

            pre_restore_checks

            if [[ "${CREATE_PRE_BACKUP:-true}" == "true" ]]; then
                create_pre_restore_backup
            fi

            if [[ -n "${BACKUP_ID}" ]]; then
                # S3 backup restore
                local backup_file=$(download_backup "${BACKUP_ID}")
                restore_from_s3 "${backup_file}"
                verify_restore
            elif [[ -n "${SNAPSHOT_ID}" ]]; then
                # RDS snapshot restore
                local new_endpoint=$(restore_from_rds_snapshot "${SNAPSHOT_ID}")
                if [[ "${SWITCH_TRAFFIC:-false}" == "true" ]]; then
                    verify_restore "${new_endpoint}"
                    switch_traffic "${new_endpoint}"
                fi
            elif [[ -n "${POINT_IN_TIME}" ]]; then
                # Point-in-time recovery
                local new_endpoint=$(restore_point_in_time "${POINT_IN_TIME}")
                if [[ "${SWITCH_TRAFFIC:-false}" == "true" ]]; then
                    verify_restore "${new_endpoint}"
                    switch_traffic "${new_endpoint}"
                fi
            fi

            log_success "Restore completed successfully!"
            send_alert "SUCCESS" "Database restore completed successfully"
            ;;

        verify)
            verify_restore
            ;;

        *)
            log_error "Unknown command: ${COMMAND}"
            exit 1
            ;;
    esac
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        list|restore|verify)
            COMMAND="$1"
            shift
            ;;
        --backup-id|-b)
            BACKUP_ID="$2"
            shift 2
            ;;
        --snapshot-id|-s)
            SNAPSHOT_ID="$2"
            shift 2
            ;;
        --point-in-time|-p)
            POINT_IN_TIME="$2"
            shift 2
            ;;
        --target-db|-t)
            TARGET_DB="$2"
            shift 2
            ;;
        --mode|-m)
            RESTORE_MODE="$2"
            shift 2
            ;;
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --force)
            FORCE_OVERWRITE="true"
            shift
            ;;
        --switch-traffic)
            SWITCH_TRAFFIC="true"
            shift
            ;;
        --terminate-connections)
            TERMINATE_CONNECTIONS="true"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  list      List available backups"
            echo "  restore   Restore database"
            echo "  verify    Verify database integrity"
            echo ""
            echo "Options:"
            echo "  -b, --backup-id       S3 backup ID to restore"
            echo "  -s, --snapshot-id     RDS snapshot ID to restore"
            echo "  -p, --point-in-time   Point-in-time to restore (ISO 8601 format)"
            echo "  -t, --target-db       Target database name"
            echo "  -m, --mode            Restore mode: verify, dry-run, execute"
            echo "  -e, --environment     Environment name"
            echo "  --force               Force overwrite existing database"
            echo "  --switch-traffic      Switch application traffic after restore"
            echo "  --terminate-connections  Terminate existing connections"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$(dirname "${LOG_FILE}")"
main
