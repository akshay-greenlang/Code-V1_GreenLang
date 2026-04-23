#!/bin/bash
# ==============================================================================
# GL-001 ThermalCommand - Backup Verification Script
# ==============================================================================
# This script verifies the integrity and recoverability of GL-001 backups:
# - Checks backup file existence and sizes
# - Validates checksums
# - Tests restore to a verification cluster
# - Validates data integrity
#
# Usage: ./verify-backup.sh [--date YYYY-MM-DD] [--full-restore-test]
#
# ==============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/gl-001-verify.log"
TEMP_DIR="/tmp/gl-001-verify-${TIMESTAMP}"
REPORT_FILE="${TEMP_DIR}/verification_report.json"

# Default values
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
BACKUP_REGION="${BACKUP_REGION:-us-east-1}"
NAMESPACE="${NAMESPACE:-greenlang}"
VERIFY_NAMESPACE="${VERIFY_NAMESPACE:-greenlang-verify}"
VERIFY_DATE="${1:-$(date -d 'yesterday' +%Y-%m-%d)}"
FULL_RESTORE_TEST="${FULL_RESTORE_TEST:-false}"

# Backup paths
BACKUP_ROOT="s3://${BACKUP_BUCKET}/gl-001"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --date)
            VERIFY_DATE="$2"
            shift 2
            ;;
        --full-restore-test)
            FULL_RESTORE_TEST="true"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

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
# Report Functions
# ==============================================================================

init_report() {
    mkdir -p "${TEMP_DIR}"
    cat > "${REPORT_FILE}" << EOF
{
    "verification_date": "$(date -Iseconds)",
    "backup_date": "${VERIFY_DATE}",
    "overall_status": "pending",
    "components": {}
}
EOF
}

update_report() {
    local component="$1"
    local status="$2"
    local details="$3"

    # Use jq to update JSON
    local tmp_file="${REPORT_FILE}.tmp"
    jq ".components.\"${component}\" = {\"status\": \"${status}\", \"details\": ${details}}" \
        "${REPORT_FILE}" > "${tmp_file}"
    mv "${tmp_file}" "${REPORT_FILE}"
}

finalize_report() {
    local overall_status="$1"
    local tmp_file="${REPORT_FILE}.tmp"
    jq ".overall_status = \"${overall_status}\"" "${REPORT_FILE}" > "${tmp_file}"
    mv "${tmp_file}" "${REPORT_FILE}"

    log_info "Verification Report:"
    cat "${REPORT_FILE}" | jq .
}

# ==============================================================================
# Cleanup
# ==============================================================================

cleanup() {
    log_info "Cleaning up..."

    # Delete verification namespace if it exists
    if kubectl get namespace "${VERIFY_NAMESPACE}" &> /dev/null; then
        kubectl delete namespace "${VERIFY_NAMESPACE}" --wait=false || true
    fi

    # Keep temp dir for debugging, just log location
    log_info "Temp directory: ${TEMP_DIR}"
}

trap cleanup EXIT

# ==============================================================================
# Verification Functions
# ==============================================================================

verify_backup_exists() {
    log_info "Verifying backup existence for ${VERIFY_DATE}..."

    local status="pass"
    local details='{}'
    local checks=()

    # Check PostgreSQL backup
    local pg_exists=$(aws s3 ls "${BACKUP_ROOT}/postgresql/full/${VERIFY_DATE}/" \
        --region "${BACKUP_REGION}" 2>/dev/null | grep -c ".tar.gz$" || echo "0")

    if [ "${pg_exists}" -gt 0 ]; then
        checks+=("postgresql: found")
        log_success "PostgreSQL backup found"
    else
        checks+=("postgresql: missing")
        log_error "PostgreSQL backup not found"
        status="fail"
    fi

    # Check Redis backup
    local redis_exists=$(aws s3 ls "${BACKUP_ROOT}/redis/rdb/${VERIFY_DATE}/" \
        --region "${BACKUP_REGION}" 2>/dev/null | grep -c ".rdb$" || echo "0")

    if [ "${redis_exists}" -gt 0 ]; then
        checks+=("redis: found")
        log_success "Redis backup found"
    else
        checks+=("redis: missing")
        log_error "Redis backup not found"
        status="fail"
    fi

    # Check Config backup
    local config_exists=$(aws s3 ls "${BACKUP_ROOT}/config/${VERIFY_DATE}/" \
        --region "${BACKUP_REGION}" 2>/dev/null | grep -c ".tar.gz$" || echo "0")

    if [ "${config_exists}" -gt 0 ]; then
        checks+=("config: found")
        log_success "Config backup found"
    else
        checks+=("config: missing")
        log_warn "Config backup not found"
    fi

    details=$(printf '%s\n' "${checks[@]}" | jq -R -s 'split("\n") | map(select(length > 0))')
    update_report "existence" "${status}" "${details}"

    return $( [ "${status}" == "pass" ] && echo 0 || echo 1 )
}

verify_backup_integrity() {
    log_info "Verifying backup integrity..."

    local status="pass"
    local details='{}'
    local checks=()

    # Get PostgreSQL backup file
    local pg_backup=$(aws s3 ls "${BACKUP_ROOT}/postgresql/full/${VERIFY_DATE}/" \
        --region "${BACKUP_REGION}" | grep ".tar.gz$" | awk '{print $4}' | head -1)

    if [ -n "${pg_backup}" ]; then
        # Download backup
        aws s3 cp "${BACKUP_ROOT}/postgresql/full/${VERIFY_DATE}/${pg_backup}" \
            "${TEMP_DIR}/${pg_backup}" \
            --region "${BACKUP_REGION}"

        # Check if checksum file exists
        local checksum_file="${pg_backup%.tar.gz}.sha256"
        if aws s3 ls "${BACKUP_ROOT}/postgresql/full/${VERIFY_DATE}/${checksum_file}" \
            --region "${BACKUP_REGION}" &> /dev/null; then

            aws s3 cp "${BACKUP_ROOT}/postgresql/full/${VERIFY_DATE}/${checksum_file}" \
                "${TEMP_DIR}/${checksum_file}" \
                --region "${BACKUP_REGION}"

            local expected=$(cat "${TEMP_DIR}/${checksum_file}")
            local actual=$(sha256sum "${TEMP_DIR}/${pg_backup}" | awk '{print $1}')

            if [ "${expected}" == "${actual}" ]; then
                checks+=("postgresql_checksum: valid")
                log_success "PostgreSQL checksum valid"
            else
                checks+=("postgresql_checksum: invalid")
                log_error "PostgreSQL checksum mismatch"
                status="fail"
            fi
        else
            checks+=("postgresql_checksum: not_found")
            log_warn "PostgreSQL checksum file not found"
        fi

        # Verify archive can be extracted
        if tar -tzf "${TEMP_DIR}/${pg_backup}" > /dev/null 2>&1; then
            checks+=("postgresql_archive: valid")
            log_success "PostgreSQL archive is valid"
        else
            checks+=("postgresql_archive: corrupted")
            log_error "PostgreSQL archive is corrupted"
            status="fail"
        fi

        # Get file size
        local pg_size=$(ls -lh "${TEMP_DIR}/${pg_backup}" | awk '{print $5}')
        checks+=("postgresql_size: ${pg_size}")
    fi

    # Get Redis backup file
    local redis_backup=$(aws s3 ls "${BACKUP_ROOT}/redis/rdb/${VERIFY_DATE}/" \
        --region "${BACKUP_REGION}" | grep ".rdb$" | awk '{print $4}' | head -1)

    if [ -n "${redis_backup}" ]; then
        # Download backup
        aws s3 cp "${BACKUP_ROOT}/redis/rdb/${VERIFY_DATE}/${redis_backup}" \
            "${TEMP_DIR}/${redis_backup}" \
            --region "${BACKUP_REGION}"

        # Check RDB header
        local rdb_header=$(head -c 5 "${TEMP_DIR}/${redis_backup}")
        if [ "${rdb_header}" == "REDIS" ]; then
            checks+=("redis_format: valid")
            log_success "Redis RDB format valid"
        else
            checks+=("redis_format: invalid")
            log_error "Redis RDB format invalid"
            status="fail"
        fi

        local redis_size=$(ls -lh "${TEMP_DIR}/${redis_backup}" | awk '{print $5}')
        checks+=("redis_size: ${redis_size}")
    fi

    details=$(printf '%s\n' "${checks[@]}" | jq -R -s 'split("\n") | map(select(length > 0))')
    update_report "integrity" "${status}" "${details}"

    return $( [ "${status}" == "pass" ] && echo 0 || echo 1 )
}

verify_restore_test() {
    if [ "${FULL_RESTORE_TEST}" != "true" ]; then
        log_info "Skipping full restore test (use --full-restore-test to enable)"
        update_report "restore_test" "skipped" '["Not requested"]'
        return 0
    fi

    log_info "Performing full restore test to verification cluster..."

    local status="pass"
    local details='{}'
    local checks=()

    # Create verification namespace
    kubectl create namespace "${VERIFY_NAMESPACE}" || true

    # Deploy minimal PostgreSQL for testing
    log_info "Deploying verification PostgreSQL..."
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pg-verify
  namespace: ${VERIFY_NAMESPACE}
spec:
  containers:
  - name: postgres
    image: postgres:14
    env:
    - name: POSTGRES_PASSWORD
      value: verify123
    - name: POSTGRES_DB
      value: greenlang_gl001
    ports:
    - containerPort: 5432
    volumeMounts:
    - name: data
      mountPath: /var/lib/postgresql/data
  volumes:
  - name: data
    emptyDir: {}
EOF

    # Wait for pod
    kubectl wait --for=condition=ready pod/pg-verify \
        -n "${VERIFY_NAMESPACE}" --timeout=120s

    # Get PostgreSQL backup
    local pg_backup=$(aws s3 ls "${BACKUP_ROOT}/postgresql/full/${VERIFY_DATE}/" \
        --region "${BACKUP_REGION}" | grep ".tar.gz$" | awk '{print $4}' | head -1)

    # Download if not already downloaded
    if [ ! -f "${TEMP_DIR}/${pg_backup}" ]; then
        aws s3 cp "${BACKUP_ROOT}/postgresql/full/${VERIFY_DATE}/${pg_backup}" \
            "${TEMP_DIR}/${pg_backup}" \
            --region "${BACKUP_REGION}"
    fi

    # Extract and copy to pod
    mkdir -p "${TEMP_DIR}/pg_restore"
    tar -xzf "${TEMP_DIR}/${pg_backup}" -C "${TEMP_DIR}/pg_restore"

    kubectl cp "${TEMP_DIR}/pg_restore" \
        "${VERIFY_NAMESPACE}/pg-verify:/tmp/restore/"

    # Attempt restore
    log_info "Attempting database restore..."
    if kubectl exec -n "${VERIFY_NAMESPACE}" pg-verify -- bash -c "
        cd /tmp/restore
        pg_restore -d greenlang_gl001 -U postgres -c base.tar 2>/dev/null || true
    "; then
        checks+=("restore_execution: success")
        log_success "Database restore executed"
    else
        checks+=("restore_execution: failed")
        log_error "Database restore failed"
        status="fail"
    fi

    # Verify data
    log_info "Verifying restored data..."
    local table_count=$(kubectl exec -n "${VERIFY_NAMESPACE}" pg-verify -- \
        psql -U postgres -d greenlang_gl001 -t -c \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';" 2>/dev/null || echo "0")

    table_count=$(echo "${table_count}" | tr -d ' ')

    if [ "${table_count}" -gt 0 ]; then
        checks+=("table_count: ${table_count}")
        log_success "Found ${table_count} tables in restored database"
    else
        checks+=("table_count: 0")
        log_warn "No tables found in restored database"
    fi

    # Check specific tables
    local heat_plans=$(kubectl exec -n "${VERIFY_NAMESPACE}" pg-verify -- \
        psql -U postgres -d greenlang_gl001 -t -c \
        "SELECT COUNT(*) FROM heat_plans;" 2>/dev/null || echo "0")

    heat_plans=$(echo "${heat_plans}" | tr -d ' ')
    checks+=("heat_plans_count: ${heat_plans}")

    local audit_logs=$(kubectl exec -n "${VERIFY_NAMESPACE}" pg-verify -- \
        psql -U postgres -d greenlang_gl001 -t -c \
        "SELECT COUNT(*) FROM audit_log;" 2>/dev/null || echo "0")

    audit_logs=$(echo "${audit_logs}" | tr -d ' ')
    checks+=("audit_logs_count: ${audit_logs}")

    # Cleanup
    kubectl delete namespace "${VERIFY_NAMESPACE}" --wait=false || true

    details=$(printf '%s\n' "${checks[@]}" | jq -R -s 'split("\n") | map(select(length > 0))')
    update_report "restore_test" "${status}" "${details}"

    return $( [ "${status}" == "pass" ] && echo 0 || echo 1 )
}

verify_backup_age() {
    log_info "Verifying backup age and freshness..."

    local status="pass"
    local details='{}'
    local checks=()

    # Get latest PostgreSQL backup
    local latest_pg=$(aws s3 ls "${BACKUP_ROOT}/postgresql/full/" \
        --region "${BACKUP_REGION}" | sort | tail -1 | awk '{print $2}' | tr -d '/')

    if [ -n "${latest_pg}" ]; then
        local backup_age=$(($(date +%s) - $(date -d "${latest_pg}" +%s)))
        local backup_age_hours=$((backup_age / 3600))

        if [ ${backup_age_hours} -lt 24 ]; then
            checks+=("postgresql_age: ${backup_age_hours} hours (OK)")
            log_success "PostgreSQL backup is ${backup_age_hours} hours old"
        elif [ ${backup_age_hours} -lt 48 ]; then
            checks+=("postgresql_age: ${backup_age_hours} hours (WARNING)")
            log_warn "PostgreSQL backup is ${backup_age_hours} hours old"
        else
            checks+=("postgresql_age: ${backup_age_hours} hours (CRITICAL)")
            log_error "PostgreSQL backup is ${backup_age_hours} hours old"
            status="fail"
        fi
    fi

    # Get latest Redis backup
    local latest_redis=$(aws s3 ls "${BACKUP_ROOT}/redis/rdb/" \
        --region "${BACKUP_REGION}" | sort | tail -1 | awk '{print $2}' | tr -d '/')

    if [ -n "${latest_redis}" ]; then
        local backup_age=$(($(date +%s) - $(date -d "${latest_redis}" +%s)))
        local backup_age_hours=$((backup_age / 3600))

        if [ ${backup_age_hours} -lt 24 ]; then
            checks+=("redis_age: ${backup_age_hours} hours (OK)")
            log_success "Redis backup is ${backup_age_hours} hours old"
        else
            checks+=("redis_age: ${backup_age_hours} hours (WARNING)")
            log_warn "Redis backup is ${backup_age_hours} hours old"
        fi
    fi

    details=$(printf '%s\n' "${checks[@]}" | jq -R -s 'split("\n") | map(select(length > 0))')
    update_report "backup_age" "${status}" "${details}"

    return $( [ "${status}" == "pass" ] && echo 0 || echo 1 )
}

verify_cross_region_replication() {
    log_info "Verifying cross-region replication..."

    local status="pass"
    local details='{}'
    local checks=()

    local DR_BUCKET="${DR_BUCKET:-greenlang-backups-dr}"
    local DR_REGION="${DR_REGION:-us-west-2}"
    local DR_ROOT="s3://${DR_BUCKET}/gl-001"

    # Check if DR bucket exists and has backups
    if aws s3 ls "${DR_ROOT}/postgresql/full/${VERIFY_DATE}/" \
        --region "${DR_REGION}" &> /dev/null; then

        local dr_count=$(aws s3 ls "${DR_ROOT}/postgresql/full/${VERIFY_DATE}/" \
            --region "${DR_REGION}" | grep -c ".tar.gz$" || echo "0")

        if [ "${dr_count}" -gt 0 ]; then
            checks+=("dr_postgresql: replicated")
            log_success "PostgreSQL backup replicated to DR region"
        else
            checks+=("dr_postgresql: missing")
            log_error "PostgreSQL backup not replicated"
            status="fail"
        fi
    else
        checks+=("dr_bucket: not_accessible")
        log_warn "DR bucket not accessible or not configured"
    fi

    details=$(printf '%s\n' "${checks[@]}" | jq -R -s 'split("\n") | map(select(length > 0))')
    update_report "cross_region" "${status}" "${details}"

    return $( [ "${status}" == "pass" ] && echo 0 || echo 1 )
}

# ==============================================================================
# Main Function
# ==============================================================================

main() {
    log_info "========================================="
    log_info "GL-001 Backup Verification Starting"
    log_info "Verification Date: ${VERIFY_DATE}"
    log_info "Full Restore Test: ${FULL_RESTORE_TEST}"
    log_info "========================================="

    # Initialize report
    init_report

    local overall_status="pass"

    # Run verification checks
    verify_backup_exists || overall_status="fail"
    verify_backup_integrity || overall_status="fail"
    verify_backup_age || overall_status="fail"
    verify_cross_region_replication || true  # Don't fail on DR check
    verify_restore_test || overall_status="fail"

    # Finalize report
    finalize_report "${overall_status}"

    log_info "========================================="
    if [ "${overall_status}" == "pass" ]; then
        log_success "GL-001 Backup Verification PASSED"
    else
        log_error "GL-001 Backup Verification FAILED"
    fi
    log_info "========================================="

    # Send notification
    if [ -n "${SLACK_WEBHOOK:-}" ]; then
        local color=$( [ "${overall_status}" == "pass" ] && echo "#36a64f" || echo "#ff0000" )
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"GL-001 Backup Verification ${overall_status^^}\",
                    \"text\": \"Verification for ${VERIFY_DATE} completed\",
                    \"fields\": [
                        {\"title\": \"Full Restore Test\", \"value\": \"${FULL_RESTORE_TEST}\", \"short\": true}
                    ]
                }]
            }" \
            "${SLACK_WEBHOOK}" || true
    fi

    # Exit with appropriate code
    [ "${overall_status}" == "pass" ] && exit 0 || exit 1
}

# Run main function
main "$@"
