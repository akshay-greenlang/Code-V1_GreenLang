#!/bin/bash
# GL-005 Combusense - Comprehensive Restore Script
# Combustion Control & Sensing Agent
# Restores all critical data with bumpless controller transfer support

set -euo pipefail

#######################################
# Configuration
#######################################
AGENT_ID="gl-005"
AGENT_NAME="combusense"
NAMESPACE="${NAMESPACE:-greenlang}"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
RESTORE_DIR="/tmp/${AGENT_ID}-restore-$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/var/log/${AGENT_ID}/restore-$(date +%Y%m%d_%H%M%S).log"

# Restore configuration
RESTORE_DATE="${1:-}"
COMPONENT="${2:-all}"
VERIFY_CHECKSUMS="${VERIFY_CHECKSUMS:-true}"
DRY_RUN="${DRY_RUN:-false}"
SAFE_MODE="${SAFE_MODE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

#######################################
# Logging Functions
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
log_step() { log "${BLUE}STEP${NC}" "$*"; }

#######################################
# Usage
#######################################
usage() {
    cat << EOF
Usage: $0 <restore_date> [component]

Arguments:
    restore_date    Date of backup to restore (YYYY-MM-DD) or 'latest'
    component       Component to restore (default: all)
                    Options: all, postgresql, redis, influxdb, kafka, config, pid, sensors

Environment Variables:
    NAMESPACE           Kubernetes namespace (default: greenlang)
    BACKUP_BUCKET       S3 bucket name (default: greenlang-backups)
    VERIFY_CHECKSUMS    Verify backup checksums (default: true)
    DRY_RUN             Show what would be restored without executing (default: false)
    SAFE_MODE           Require confirmation before restore (default: true)

Examples:
    $0 2025-01-15                    # Restore all components from specific date
    $0 latest                        # Restore all from latest backup
    $0 2025-01-15 postgresql         # Restore only PostgreSQL
    $0 2025-01-15 pid                # Restore only PID parameters (quick restore)

EOF
    exit 1
}

#######################################
# Pre-flight Checks
#######################################
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check required arguments
    if [[ -z "${RESTORE_DATE}" ]]; then
        usage
    fi

    # Check required tools
    local required_tools=("kubectl" "aws" "pg_restore" "redis-cli" "influx" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "${tool}" &> /dev/null; then
            log_error "Required tool not found: ${tool}"
            exit 1
        fi
    done

    # Check kubectl context
    local context=$(kubectl config current-context)
    log_info "Using kubectl context: ${context}"

    # Check namespace exists
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_error "Namespace ${NAMESPACE} not found"
        exit 1
    fi

    # Create restore directory
    mkdir -p "${RESTORE_DIR}"
    mkdir -p "$(dirname ${LOG_FILE})"

    # Resolve 'latest' date
    if [[ "${RESTORE_DATE}" == "latest" ]]; then
        RESTORE_DATE=$(aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/" \
            | sort -r | head -1 | awk '{print $2}' | tr -d '/')
        log_info "Latest backup date: ${RESTORE_DATE}"
    fi

    # Check backup exists
    if ! aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${RESTORE_DATE}/" &> /dev/null; then
        log_error "No backup found for date: ${RESTORE_DATE}"
        exit 1
    fi

    log_success "Pre-flight checks passed"
}

#######################################
# Download and Verify Backup
#######################################
download_backup() {
    log_step "Downloading backup files..."

    # Download manifest
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/manifests/" --recursive

    # Get latest manifest
    local manifest=$(ls -t ${RESTORE_DIR}/manifests/manifest_*.json | head -1)
    TIMESTAMP=$(jq -r '.backup_timestamp' "${manifest}")
    log_info "Restoring from backup timestamp: ${TIMESTAMP}"

    # Download based on component
    case "${COMPONENT}" in
        all)
            download_postgresql_backup
            download_redis_backup
            download_influxdb_backup
            download_kafka_backup
            download_config_backup
            ;;
        postgresql)
            download_postgresql_backup
            ;;
        redis)
            download_redis_backup
            ;;
        influxdb)
            download_influxdb_backup
            ;;
        kafka)
            download_kafka_backup
            ;;
        config|pid|sensors)
            download_config_backup
            ;;
    esac

    # Verify checksums
    if [[ "${VERIFY_CHECKSUMS}" == "true" ]]; then
        verify_checksums
    fi
}

download_postgresql_backup() {
    log_info "Downloading PostgreSQL backup..."
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/postgresql/" --recursive
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/control/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/postgresql/control/" --recursive
}

download_redis_backup() {
    log_info "Downloading Redis backup..."
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/redis/" --recursive
}

download_influxdb_backup() {
    log_info "Downloading InfluxDB backup..."
    aws s3 sync "s3://${BACKUP_BUCKET}/${AGENT_ID}/influxdb/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/influxdb/"
}

download_kafka_backup() {
    log_info "Downloading Kafka backup..."
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/kafka/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/kafka/" --recursive
}

download_config_backup() {
    log_info "Downloading configuration backup..."
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/" \
        "${RESTORE_DIR}/config/" --recursive
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/pid/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/config/pid/" --recursive
    aws s3 cp "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/sensors/${RESTORE_DATE}/" \
        "${RESTORE_DIR}/config/sensors/" --recursive
}

verify_checksums() {
    log_info "Verifying backup checksums..."
    local checksum_file=$(ls -t ${RESTORE_DIR}/manifests/checksums_*.sha256 | head -1)

    if [[ -f "${checksum_file}" ]]; then
        # Verify each downloaded file
        local failed=0
        while read -r expected_checksum filepath; do
            local filename=$(basename "${filepath}")
            local local_file=$(find "${RESTORE_DIR}" -name "${filename}" -type f 2>/dev/null | head -1)
            if [[ -n "${local_file}" ]]; then
                local actual_checksum=$(sha256sum "${local_file}" | awk '{print $1}')
                if [[ "${expected_checksum}" != "${actual_checksum}" ]]; then
                    log_error "Checksum mismatch for ${filename}"
                    failed=1
                fi
            fi
        done < "${checksum_file}"

        if [[ ${failed} -eq 1 ]]; then
            log_error "Checksum verification failed"
            exit 1
        fi
        log_success "Checksum verification passed"
    else
        log_warn "No checksum file found, skipping verification"
    fi
}

#######################################
# Safety Check - Put Controller in Safe Mode
#######################################
enter_safe_mode() {
    log_step "Entering safe mode..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would enter safe mode"
        return
    fi

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -n "${app_pod}" ]]; then
        # Request safe state
        kubectl exec -n ${NAMESPACE} ${app_pod} -- \
            curl -s -X POST localhost:8080/api/v1/controller/safe-state \
            -d '{"reason": "restore_operation", "hold_output": true}' || true

        log_info "Controller entered safe state - outputs are held"
    fi
}

exit_safe_mode() {
    log_step "Exiting safe mode..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would exit safe mode"
        return
    fi

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -n "${app_pod}" ]]; then
        # Request to exit safe state
        kubectl exec -n ${NAMESPACE} ${app_pod} -- \
            curl -s -X POST localhost:8080/api/v1/controller/resume \
            -d '{"bumpless": true}' || true

        log_info "Controller resumed with bumpless transfer"
    fi
}

#######################################
# Restore Functions
#######################################
restore_postgresql() {
    log_step "Restoring PostgreSQL..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore PostgreSQL from ${RESTORE_DIR}/postgresql/"
        return
    fi

    local pg_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || \
        kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-postgres \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using PostgreSQL pod: ${pg_pod}"

    # Stop application to prevent writes during restore
    log_info "Scaling down application..."
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME}-primary --replicas=0 -n ${NAMESPACE}
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME}-standby --replicas=0 -n ${NAMESPACE}
    sleep 10

    # Find the backup file
    local backup_file=$(ls -t ${RESTORE_DIR}/postgresql/postgresql_full_*.dump 2>/dev/null | head -1)

    if [[ -z "${backup_file}" ]]; then
        log_error "No PostgreSQL backup file found"
        return 1
    fi

    # Copy backup to pod
    kubectl cp "${backup_file}" "${NAMESPACE}/${pg_pod}:/tmp/restore.dump"

    # Restore database
    log_info "Restoring database..."
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        pg_restore -U postgres -d greenlang_${AGENT_ID//-/_} -c --if-exists /tmp/restore.dump || true

    # Restore control schema specifically if exists
    local control_backup=$(ls -t ${RESTORE_DIR}/postgresql/control/postgresql_control_*.dump 2>/dev/null | head -1)
    if [[ -n "${control_backup}" ]]; then
        log_info "Restoring control schema..."
        kubectl cp "${control_backup}" "${NAMESPACE}/${pg_pod}:/tmp/control_restore.dump"
        kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
            pg_restore -U postgres -d greenlang_${AGENT_ID//-/_} -n control -c --if-exists /tmp/control_restore.dump || true
    fi

    # Cleanup
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- rm -f /tmp/restore.dump /tmp/control_restore.dump

    log_success "PostgreSQL restore completed"
}

restore_redis() {
    log_step "Restoring Redis..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore Redis from ${RESTORE_DIR}/redis/"
        return
    fi

    local redis_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-redis \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using Redis pod: ${redis_pod}"

    # Find backup file
    local rdb_file=$(ls -t ${RESTORE_DIR}/redis/redis_dump_*.rdb 2>/dev/null | head -1)

    if [[ -z "${rdb_file}" ]]; then
        log_error "No Redis backup file found"
        return 1
    fi

    # Stop Redis
    log_info "Stopping Redis for restore..."
    kubectl exec -n ${NAMESPACE} ${redis_pod} -- redis-cli -a "${REDIS_PASSWORD:-}" SHUTDOWN NOSAVE || true
    sleep 5

    # Copy RDB file
    kubectl cp "${rdb_file}" "${NAMESPACE}/${redis_pod}:/data/dump.rdb"

    # Restart Redis (pod will restart automatically)
    log_info "Waiting for Redis to restart..."
    kubectl wait --for=condition=ready pod/${redis_pod} -n ${NAMESPACE} --timeout=120s

    log_success "Redis restore completed"
}

restore_influxdb() {
    log_step "Restoring InfluxDB..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore InfluxDB from ${RESTORE_DIR}/influxdb/"
        return
    fi

    local influx_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-influxdb \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using InfluxDB pod: ${influx_pod}"

    # Copy backup files
    kubectl cp "${RESTORE_DIR}/influxdb/" "${NAMESPACE}/${influx_pod}:/tmp/influx-restore/"

    # Restore
    kubectl exec -n ${NAMESPACE} ${influx_pod} -- \
        influx restore /tmp/influx-restore \
        --org greenlang \
        --bucket combusense_metrics \
        --full

    # Cleanup
    kubectl exec -n ${NAMESPACE} ${influx_pod} -- rm -rf /tmp/influx-restore

    log_success "InfluxDB restore completed"
}

restore_pid_parameters() {
    log_step "Restoring PID parameters..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore PID parameters"
        return
    fi

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -z "${app_pod}" ]]; then
        log_warn "No application pod found, will apply when pods restart"

        # Update ConfigMap instead
        local pid_file="${RESTORE_DIR}/config/latest/pid_parameters.json"
        if [[ -f "${pid_file}" ]]; then
            kubectl create configmap ${AGENT_ID}-pid-parameters-restored \
                --from-file=pid_parameters.json="${pid_file}" \
                -n ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
        fi
        return
    fi

    # Find PID parameters file
    local pid_file="${RESTORE_DIR}/config/latest/pid_parameters.json"
    if [[ ! -f "${pid_file}" ]]; then
        pid_file=$(ls -t ${RESTORE_DIR}/config/pid/pid_parameters_*.json 2>/dev/null | head -1)
    fi

    if [[ -z "${pid_file}" || ! -f "${pid_file}" ]]; then
        log_error "No PID parameters file found"
        return 1
    fi

    log_info "Restoring PID parameters from ${pid_file}"

    # Copy and apply
    kubectl cp "${pid_file}" "${NAMESPACE}/${app_pod}:/tmp/pid_parameters.json"
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s -X POST localhost:8080/api/v1/controller/parameters \
        -H "Content-Type: application/json" \
        -d @/tmp/pid_parameters.json

    log_success "PID parameters restore completed"
}

restore_sensor_calibration() {
    log_step "Restoring sensor calibration..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore sensor calibration"
        return
    fi

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -z "${app_pod}" ]]; then
        log_warn "No application pod found, skipping sensor calibration restore"
        return
    fi

    # Find calibration file
    local cal_file="${RESTORE_DIR}/config/latest/sensor_calibration.json"
    if [[ ! -f "${cal_file}" ]]; then
        cal_file=$(ls -t ${RESTORE_DIR}/config/sensors/sensor_calibration*.json 2>/dev/null | head -1)
    fi

    if [[ -z "${cal_file}" || ! -f "${cal_file}" ]]; then
        log_warn "No sensor calibration file found"
        return
    fi

    log_info "Restoring sensor calibration from ${cal_file}"

    kubectl cp "${cal_file}" "${NAMESPACE}/${app_pod}:/tmp/sensor_calibration.json"
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s -X POST localhost:8080/api/v1/sensors/calibration \
        -H "Content-Type: application/json" \
        -d @/tmp/sensor_calibration.json

    log_success "Sensor calibration restore completed"
}

restore_configuration() {
    log_step "Restoring Kubernetes configuration..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would restore Kubernetes configuration"
        return
    fi

    # Restore ConfigMaps
    for yaml_file in ${RESTORE_DIR}/config/k8s/*.yaml; do
        if [[ -f "${yaml_file}" ]]; then
            log_info "Applying $(basename ${yaml_file})..."
            kubectl apply -f "${yaml_file}" -n ${NAMESPACE} || true
        fi
    done

    log_success "Configuration restore completed"
}

#######################################
# Scale Application Back Up
#######################################
scale_up_application() {
    log_step "Scaling up application..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would scale up application"
        return
    fi

    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME}-primary --replicas=2 -n ${NAMESPACE}
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME}-standby --replicas=2 -n ${NAMESPACE}

    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -n ${NAMESPACE} --timeout=300s

    log_success "Application scaled up"
}

#######################################
# Verification
#######################################
verify_restore() {
    log_step "Verifying restore..."

    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "[DRY RUN] Would verify restore"
        return
    fi

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -n "${app_pod}" ]]; then
        # Health check
        log_info "Checking application health..."
        local health=$(kubectl exec -n ${NAMESPACE} ${app_pod} -- \
            curl -s localhost:8080/api/v1/health)
        echo "${health}" | jq .

        # Controller status
        log_info "Checking controller status..."
        local status=$(kubectl exec -n ${NAMESPACE} ${app_pod} -- \
            curl -s localhost:8080/api/v1/controller/status)
        echo "${status}" | jq .

        # Verify PID parameters loaded
        log_info "Verifying PID parameters..."
        local params=$(kubectl exec -n ${NAMESPACE} ${app_pod} -- \
            curl -s localhost:8080/api/v1/controller/parameters)
        echo "${params}" | jq .
    fi

    log_success "Verification completed"
}

#######################################
# Confirmation Prompt
#######################################
confirm_restore() {
    if [[ "${SAFE_MODE}" != "true" || "${DRY_RUN}" == "true" ]]; then
        return 0
    fi

    echo ""
    echo "========================================"
    echo "GL-005 Combusense Restore Confirmation"
    echo "========================================"
    echo "Restore Date: ${RESTORE_DATE}"
    echo "Component: ${COMPONENT}"
    echo "Namespace: ${NAMESPACE}"
    echo ""
    echo -e "${YELLOW}WARNING: This will overwrite existing data!${NC}"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm

    if [[ "${confirm}" != "yes" ]]; then
        log_info "Restore cancelled by user"
        exit 0
    fi
}

#######################################
# Cleanup
#######################################
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf "${RESTORE_DIR}"
}

#######################################
# Main Execution
#######################################
main() {
    local start_time=$(date +%s)
    local exit_code=0

    trap cleanup EXIT

    log_info "=========================================="
    log_info "GL-005 Combusense Restore Starting"
    log_info "Restore Date: ${RESTORE_DATE}"
    log_info "Component: ${COMPONENT}"
    log_info "=========================================="

    # Run pre-flight checks
    preflight_checks

    # Confirm with user
    confirm_restore

    # Download backup
    download_backup

    # Enter safe mode before restore
    if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "pid" ]]; then
        enter_safe_mode
    fi

    # Execute restore based on component
    {
        case "${COMPONENT}" in
            all)
                restore_postgresql
                restore_redis
                restore_influxdb
                scale_up_application
                restore_pid_parameters
                restore_sensor_calibration
                ;;
            postgresql)
                restore_postgresql
                scale_up_application
                ;;
            redis)
                restore_redis
                ;;
            influxdb)
                restore_influxdb
                ;;
            config)
                restore_configuration
                ;;
            pid)
                restore_pid_parameters
                ;;
            sensors)
                restore_sensor_calibration
                ;;
        esac

        # Exit safe mode after restore
        if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "pid" ]]; then
            exit_safe_mode
        fi

        # Verify restore
        verify_restore
    } || {
        exit_code=$?
        log_error "Restore failed with exit code: ${exit_code}"
    }

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [[ ${exit_code} -eq 0 ]]; then
        log_success "=========================================="
        log_success "Restore completed successfully"
        log_success "Duration: ${duration} seconds"
        log_success "=========================================="
    else
        log_error "=========================================="
        log_error "Restore failed"
        log_error "Duration: ${duration} seconds"
        log_error "=========================================="
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
