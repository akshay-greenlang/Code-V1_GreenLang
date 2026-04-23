#!/bin/bash
# GL-005 Combusense - Backup Verification Script
# Verifies backup integrity and completeness for combustion control system

set -euo pipefail

#######################################
# Configuration
#######################################
AGENT_ID="gl-005"
AGENT_NAME="combusense"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
VERIFY_DATE="${1:-$(date -d 'yesterday' +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)}"
VERIFY_DIR="/tmp/${AGENT_ID}-verify-$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/var/log/${AGENT_ID}/verify-$(date +%Y%m%d_%H%M%S).log"

# Verification settings
DOWNLOAD_SAMPLES="${DOWNLOAD_SAMPLES:-true}"
SAMPLE_RESTORE_TEST="${SAMPLE_RESTORE_TEST:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Results tracking
declare -A RESULTS

#######################################
# Logging
#######################################
log() {
    local level=$1
    shift
    local message=$*
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() { log "INFO" "$*"; }
log_warn() { log "${YELLOW}WARN${NC}" "$*"; }
log_error() { log "${RED}ERROR${NC}" "$*"; }
log_success() { log "${GREEN}SUCCESS${NC}" "$*"; }

#######################################
# Pre-flight
#######################################
preflight() {
    log_info "Starting backup verification for ${VERIFY_DATE}"

    # Create directories
    mkdir -p "${VERIFY_DIR}"
    mkdir -p "$(dirname ${LOG_FILE})"

    # Check S3 access
    if ! aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/" &> /dev/null; then
        log_error "Cannot access backup bucket"
        exit 1
    fi
}

#######################################
# Verify Component Backups
#######################################
verify_postgresql() {
    log_info "Verifying PostgreSQL backups..."
    local status="PASS"

    # Check full backup exists
    local full_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${VERIFY_DATE}/" 2>/dev/null | grep -c ".dump" || echo "0")
    if [[ "${full_backup}" -gt 0 ]]; then
        log_success "PostgreSQL full backup: FOUND (${full_backup} files)"
    else
        log_error "PostgreSQL full backup: MISSING"
        status="FAIL"
    fi

    # Check control schema backup
    local control_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/control/${VERIFY_DATE}/" 2>/dev/null | grep -c ".dump" || echo "0")
    if [[ "${control_backup}" -gt 0 ]]; then
        log_success "PostgreSQL control backup: FOUND"
    else
        log_warn "PostgreSQL control backup: MISSING"
    fi

    # Check sensor schema backup
    local sensor_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/sensors/${VERIFY_DATE}/" 2>/dev/null | grep -c ".dump" || echo "0")
    if [[ "${sensor_backup}" -gt 0 ]]; then
        log_success "PostgreSQL sensor backup: FOUND"
    else
        log_warn "PostgreSQL sensor backup: MISSING"
    fi

    # Check audit schema backup
    local audit_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/audit/${VERIFY_DATE}/" 2>/dev/null | grep -c ".dump" || echo "0")
    if [[ "${audit_backup}" -gt 0 ]]; then
        log_success "PostgreSQL audit backup: FOUND"
    else
        log_warn "PostgreSQL audit backup: MISSING"
    fi

    # Download and verify sample if enabled
    if [[ "${DOWNLOAD_SAMPLES}" == "true" && "${full_backup}" -gt 0 ]]; then
        log_info "Downloading PostgreSQL backup for verification..."
        local backup_file=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${VERIFY_DATE}/" | head -1 | awk '{print $4}')
        aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${VERIFY_DATE}/${backup_file}" \
            "${VERIFY_DIR}/postgresql_sample.dump" --quiet

        # Verify it's a valid PostgreSQL dump
        if file "${VERIFY_DIR}/postgresql_sample.dump" | grep -q "PostgreSQL\|data"; then
            log_success "PostgreSQL backup format: VALID"
        else
            log_error "PostgreSQL backup format: INVALID"
            status="FAIL"
        fi

        # Check file size (should be > 1KB for non-empty database)
        local size=$(stat -f%z "${VERIFY_DIR}/postgresql_sample.dump" 2>/dev/null || stat -c%s "${VERIFY_DIR}/postgresql_sample.dump")
        if [[ ${size} -gt 1024 ]]; then
            log_success "PostgreSQL backup size: ${size} bytes"
        else
            log_warn "PostgreSQL backup size: ${size} bytes (unusually small)"
        fi
    fi

    RESULTS["postgresql"]="${status}"
}

verify_redis() {
    log_info "Verifying Redis backups..."
    local status="PASS"

    # Check RDB backup exists
    local rdb_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${VERIFY_DATE}/" 2>/dev/null | grep -c ".rdb" || echo "0")
    if [[ "${rdb_backup}" -gt 0 ]]; then
        log_success "Redis RDB backup: FOUND"
    else
        log_error "Redis RDB backup: MISSING"
        status="FAIL"
    fi

    # Check AOF backup (optional)
    local aof_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/aof/${VERIFY_DATE}/" 2>/dev/null | grep -c ".aof" || echo "0")
    if [[ "${aof_backup}" -gt 0 ]]; then
        log_success "Redis AOF backup: FOUND"
    else
        log_info "Redis AOF backup: NOT FOUND (optional)"
    fi

    # Download and verify sample
    if [[ "${DOWNLOAD_SAMPLES}" == "true" && "${rdb_backup}" -gt 0 ]]; then
        log_info "Downloading Redis backup for verification..."
        local backup_file=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${VERIFY_DATE}/" | head -1 | awk '{print $4}')
        aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${VERIFY_DATE}/${backup_file}" \
            "${VERIFY_DIR}/redis_sample.rdb" --quiet

        # Verify RDB magic header
        if head -c 5 "${VERIFY_DIR}/redis_sample.rdb" | grep -q "REDIS"; then
            log_success "Redis RDB format: VALID"
        else
            log_error "Redis RDB format: INVALID"
            status="FAIL"
        fi
    fi

    RESULTS["redis"]="${status}"
}

verify_influxdb() {
    log_info "Verifying InfluxDB backups..."
    local status="PASS"

    # Check backup directory exists
    local influx_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/influxdb/${VERIFY_DATE}/" 2>/dev/null | wc -l || echo "0")
    if [[ "${influx_backup}" -gt 0 ]]; then
        log_success "InfluxDB backup: FOUND (${influx_backup} files)"
    else
        log_error "InfluxDB backup: MISSING"
        status="FAIL"
    fi

    # Check CQI history backup
    local cqi_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/cqi/${VERIFY_DATE}/" 2>/dev/null | grep -c ".csv" || echo "0")
    if [[ "${cqi_backup}" -gt 0 ]]; then
        log_success "CQI history backup: FOUND"
    else
        log_warn "CQI history backup: MISSING"
    fi

    RESULTS["influxdb"]="${status}"
}

verify_pid_parameters() {
    log_info "Verifying PID parameters backup..."
    local status="PASS"

    # Check PID parameters backup
    local pid_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/pid/${VERIFY_DATE}/" 2>/dev/null | grep -c ".json" || echo "0")
    if [[ "${pid_backup}" -gt 0 ]]; then
        log_success "PID parameters backup: FOUND"
    else
        log_error "PID parameters backup: MISSING"
        status="FAIL"
    fi

    # Check latest symlink
    local latest_pid=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/pid_parameters.json" 2>/dev/null | wc -l || echo "0")
    if [[ "${latest_pid}" -gt 0 ]]; then
        log_success "PID parameters latest: FOUND"
    else
        log_warn "PID parameters latest: MISSING"
    fi

    # Download and validate JSON
    if [[ "${DOWNLOAD_SAMPLES}" == "true" && "${pid_backup}" -gt 0 ]]; then
        log_info "Validating PID parameters JSON..."
        local backup_file=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/pid/${VERIFY_DATE}/" | head -1 | awk '{print $4}')
        aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/pid/${VERIFY_DATE}/${backup_file}" \
            "${VERIFY_DIR}/pid_params.json" --quiet

        if jq empty "${VERIFY_DIR}/pid_params.json" 2>/dev/null; then
            log_success "PID parameters JSON: VALID"

            # Check for required fields
            if jq -e '.controllers' "${VERIFY_DIR}/pid_params.json" > /dev/null 2>&1; then
                local controller_count=$(jq '.controllers | length' "${VERIFY_DIR}/pid_params.json")
                log_success "PID controllers found: ${controller_count}"
            else
                log_warn "PID parameters missing 'controllers' field"
            fi
        else
            log_error "PID parameters JSON: INVALID"
            status="FAIL"
        fi
    fi

    RESULTS["pid_parameters"]="${status}"
}

verify_sensor_calibration() {
    log_info "Verifying sensor calibration backup..."
    local status="PASS"

    # Check sensor calibration backup
    local sensor_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/sensors/${VERIFY_DATE}/" 2>/dev/null | grep -c ".json" || echo "0")
    if [[ "${sensor_backup}" -gt 0 ]]; then
        log_success "Sensor calibration backup: FOUND"
    else
        log_warn "Sensor calibration backup: MISSING"
        status="WARN"
    fi

    # Download and validate JSON
    if [[ "${DOWNLOAD_SAMPLES}" == "true" && "${sensor_backup}" -gt 0 ]]; then
        log_info "Validating sensor calibration JSON..."
        aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/sensor_calibration.json" \
            "${VERIFY_DIR}/sensor_cal.json" --quiet 2>/dev/null || true

        if [[ -f "${VERIFY_DIR}/sensor_cal.json" ]] && jq empty "${VERIFY_DIR}/sensor_cal.json" 2>/dev/null; then
            log_success "Sensor calibration JSON: VALID"
        else
            log_warn "Sensor calibration validation: SKIPPED"
        fi
    fi

    RESULTS["sensor_calibration"]="${status}"
}

verify_kafka() {
    log_info "Verifying Kafka backups..."
    local status="PASS"

    # Check consumer offsets backup
    local offsets_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/kafka/${VERIFY_DATE}/" 2>/dev/null | grep -c "consumer_offsets" || echo "0")
    if [[ "${offsets_backup}" -gt 0 ]]; then
        log_success "Kafka consumer offsets: FOUND"
    else
        log_warn "Kafka consumer offsets: MISSING"
    fi

    # Check topic configs backup
    local configs_backup=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/kafka/${VERIFY_DATE}/" 2>/dev/null | grep -c "topic_configs" || echo "0")
    if [[ "${configs_backup}" -gt 0 ]]; then
        log_success "Kafka topic configs: FOUND"
    else
        log_warn "Kafka topic configs: MISSING"
    fi

    RESULTS["kafka"]="${status}"
}

verify_manifest() {
    log_info "Verifying backup manifest..."
    local status="PASS"

    # Check manifest exists
    local manifest_count=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${VERIFY_DATE}/" 2>/dev/null | grep -c "manifest_" || echo "0")
    if [[ "${manifest_count}" -gt 0 ]]; then
        log_success "Backup manifest: FOUND"
    else
        log_error "Backup manifest: MISSING"
        status="FAIL"
    fi

    # Check checksums exist
    local checksum_count=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${VERIFY_DATE}/" 2>/dev/null | grep -c "checksums_" || echo "0")
    if [[ "${checksum_count}" -gt 0 ]]; then
        log_success "Backup checksums: FOUND"
    else
        log_warn "Backup checksums: MISSING"
    fi

    # Download and validate manifest
    if [[ "${DOWNLOAD_SAMPLES}" == "true" && "${manifest_count}" -gt 0 ]]; then
        log_info "Validating manifest JSON..."
        local manifest_file=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${VERIFY_DATE}/" | grep "manifest_" | head -1 | awk '{print $4}')
        aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${VERIFY_DATE}/${manifest_file}" \
            "${VERIFY_DIR}/manifest.json" --quiet

        if jq empty "${VERIFY_DIR}/manifest.json" 2>/dev/null; then
            log_success "Manifest JSON: VALID"

            # Extract and display backup info
            local backup_timestamp=$(jq -r '.backup_timestamp' "${VERIFY_DIR}/manifest.json")
            local backup_type=$(jq -r '.backup_type' "${VERIFY_DIR}/manifest.json")
            log_info "Backup timestamp: ${backup_timestamp}"
            log_info "Backup type: ${backup_type}"
        else
            log_error "Manifest JSON: INVALID"
            status="FAIL"
        fi
    fi

    RESULTS["manifest"]="${status}"
}

#######################################
# Control System Specific Verification
#######################################
verify_control_state_consistency() {
    log_info "Verifying control state consistency..."
    local status="PASS"

    # Check controller state backup exists
    local state_file=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/controller_state.json" 2>/dev/null | wc -l || echo "0")
    if [[ "${state_file}" -gt 0 ]]; then
        log_success "Controller state backup: FOUND"

        if [[ "${DOWNLOAD_SAMPLES}" == "true" ]]; then
            aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/controller_state.json" \
                "${VERIFY_DIR}/controller_state.json" --quiet

            if jq empty "${VERIFY_DIR}/controller_state.json" 2>/dev/null; then
                # Check for essential control fields
                local mode=$(jq -r '.mode // "unknown"' "${VERIFY_DIR}/controller_state.json")
                local output=$(jq -r '.output // "unknown"' "${VERIFY_DIR}/controller_state.json")
                log_info "Last controller mode: ${mode}"
                log_info "Last controller output: ${output}"
            fi
        fi
    else
        log_warn "Controller state backup: MISSING"
    fi

    RESULTS["control_state"]="${status}"
}

#######################################
# Generate Report
#######################################
generate_report() {
    log_info ""
    log_info "=========================================="
    log_info "BACKUP VERIFICATION REPORT"
    log_info "=========================================="
    log_info "Agent: ${AGENT_ID} (${AGENT_NAME})"
    log_info "Date: ${VERIFY_DATE}"
    log_info "=========================================="
    log_info ""

    local overall_status="PASS"
    local failed_count=0
    local warn_count=0

    for component in "${!RESULTS[@]}"; do
        local result="${RESULTS[$component]}"
        case "${result}" in
            PASS)
                log_success "${component}: ${result}"
                ;;
            WARN)
                log_warn "${component}: ${result}"
                ((warn_count++))
                ;;
            FAIL)
                log_error "${component}: ${result}"
                ((failed_count++))
                overall_status="FAIL"
                ;;
        esac
    done

    log_info ""
    log_info "=========================================="
    if [[ "${overall_status}" == "PASS" ]]; then
        if [[ ${warn_count} -gt 0 ]]; then
            log_warn "OVERALL: PASS WITH WARNINGS (${warn_count} warnings)"
        else
            log_success "OVERALL: PASS"
        fi
    else
        log_error "OVERALL: FAIL (${failed_count} failures)"
    fi
    log_info "=========================================="

    # Return appropriate exit code
    if [[ "${overall_status}" == "FAIL" ]]; then
        return 1
    fi
    return 0
}

#######################################
# Cleanup
#######################################
cleanup() {
    rm -rf "${VERIFY_DIR}"
}

#######################################
# Main
#######################################
main() {
    trap cleanup EXIT

    preflight

    verify_postgresql
    verify_redis
    verify_influxdb
    verify_pid_parameters
    verify_sensor_calibration
    verify_kafka
    verify_manifest
    verify_control_state_consistency

    generate_report
}

main "$@"
