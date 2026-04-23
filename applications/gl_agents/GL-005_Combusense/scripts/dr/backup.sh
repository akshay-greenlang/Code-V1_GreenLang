#!/bin/bash
# GL-005 Combusense - Comprehensive Backup Script
# Combustion Control & Sensing Agent
# Backs up all critical data including PID parameters, sensor calibration, and control state

set -euo pipefail

#######################################
# Configuration
#######################################
AGENT_ID="gl-005"
AGENT_NAME="combusense"
NAMESPACE="${NAMESPACE:-greenlang}"
BACKUP_BUCKET="${BACKUP_BUCKET:-greenlang-backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y-%m-%d)
BACKUP_DIR="/tmp/${AGENT_ID}-backup-${TIMESTAMP}"
LOG_FILE="/var/log/${AGENT_ID}/backup-${TIMESTAMP}.log"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Control system specific
PID_BACKUP_ON_CHANGE="${PID_BACKUP_ON_CHANGE:-true}"
SENSOR_CALIBRATION_BACKUP="${SENSOR_CALIBRATION_BACKUP:-true}"
CQI_HISTORY_RETENTION_YEARS="${CQI_HISTORY_RETENTION_YEARS:-7}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

#######################################
# Pre-flight Checks
#######################################
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check required tools
    local required_tools=("kubectl" "aws" "pg_dump" "redis-cli" "influx")
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

    # Check S3 bucket access
    if ! aws s3 ls "s3://${BACKUP_BUCKET}" &> /dev/null; then
        log_error "Cannot access S3 bucket: ${BACKUP_BUCKET}"
        exit 1
    fi

    # Create backup directory
    mkdir -p "${BACKUP_DIR}"
    mkdir -p "$(dirname ${LOG_FILE})"

    log_success "Pre-flight checks passed"
}

#######################################
# PostgreSQL Backup
#######################################
backup_postgresql() {
    log_info "Starting PostgreSQL backup..."

    local pg_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-postgres,role=master \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -z "${pg_pod}" ]]; then
        pg_pod=$(kubectl get pods -n ${NAMESPACE} \
            -l app=${AGENT_ID}-postgres \
            -o jsonpath='{.items[0].metadata.name}')
    fi

    log_info "Using PostgreSQL pod: ${pg_pod}"

    # Full database backup
    log_info "Creating full database backup..."
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        pg_dump -U postgres -Fc greenlang_${AGENT_ID//-/_} \
        > "${BACKUP_DIR}/postgresql_full.dump"

    # Control system tables backup (separate for quick restore)
    log_info "Backing up control system tables..."
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        pg_dump -U postgres -Fc greenlang_${AGENT_ID//-/_} \
        -n control \
        > "${BACKUP_DIR}/postgresql_control.dump"

    # Sensor tables backup
    log_info "Backing up sensor tables..."
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        pg_dump -U postgres -Fc greenlang_${AGENT_ID//-/_} \
        -n sensors \
        > "${BACKUP_DIR}/postgresql_sensors.dump"

    # Audit tables backup
    log_info "Backing up audit tables..."
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        pg_dump -U postgres -Fc greenlang_${AGENT_ID//-/_} \
        -n audit \
        > "${BACKUP_DIR}/postgresql_audit.dump"

    # WAL archiving status
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        psql -U postgres -c "SELECT pg_current_wal_lsn();" \
        > "${BACKUP_DIR}/postgresql_wal_position.txt"

    # Upload to S3
    aws s3 cp "${BACKUP_DIR}/postgresql_full.dump" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${DATE}/postgresql_full_${TIMESTAMP}.dump" \
        --storage-class STANDARD_IA

    aws s3 cp "${BACKUP_DIR}/postgresql_control.dump" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/control/${DATE}/postgresql_control_${TIMESTAMP}.dump"

    aws s3 cp "${BACKUP_DIR}/postgresql_sensors.dump" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/sensors/${DATE}/postgresql_sensors_${TIMESTAMP}.dump"

    aws s3 cp "${BACKUP_DIR}/postgresql_audit.dump" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/audit/${DATE}/postgresql_audit_${TIMESTAMP}.dump"

    log_success "PostgreSQL backup completed"
}

#######################################
# Redis Backup
#######################################
backup_redis() {
    log_info "Starting Redis backup..."

    local redis_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-redis \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using Redis pod: ${redis_pod}"

    # Trigger BGSAVE
    log_info "Triggering Redis BGSAVE..."
    kubectl exec -n ${NAMESPACE} ${redis_pod} -- \
        redis-cli -a "${REDIS_PASSWORD:-}" BGSAVE

    # Wait for BGSAVE to complete
    log_info "Waiting for BGSAVE to complete..."
    while true; do
        local status=$(kubectl exec -n ${NAMESPACE} ${redis_pod} -- \
            redis-cli -a "${REDIS_PASSWORD:-}" LASTSAVE)
        sleep 2
        local new_status=$(kubectl exec -n ${NAMESPACE} ${redis_pod} -- \
            redis-cli -a "${REDIS_PASSWORD:-}" LASTSAVE)
        if [[ "${status}" != "${new_status}" ]]; then
            break
        fi
        sleep 1
    done

    # Copy RDB file
    kubectl cp "${NAMESPACE}/${redis_pod}:/data/dump.rdb" \
        "${BACKUP_DIR}/redis_dump.rdb"

    # Backup AOF if exists
    kubectl exec -n ${NAMESPACE} ${redis_pod} -- \
        test -f /data/appendonly.aof && \
    kubectl cp "${NAMESPACE}/${redis_pod}:/data/appendonly.aof" \
        "${BACKUP_DIR}/redis_appendonly.aof" || true

    # Backup control state keys
    log_info "Backing up control state keys..."
    kubectl exec -n ${NAMESPACE} ${redis_pod} -- \
        redis-cli -a "${REDIS_PASSWORD:-}" --scan --pattern "control:*" \
        > "${BACKUP_DIR}/redis_control_keys.txt"

    kubectl exec -n ${NAMESPACE} ${redis_pod} -- \
        redis-cli -a "${REDIS_PASSWORD:-}" --scan --pattern "pid:*" \
        > "${BACKUP_DIR}/redis_pid_keys.txt"

    # Upload to S3
    aws s3 cp "${BACKUP_DIR}/redis_dump.rdb" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/rdb/${DATE}/redis_dump_${TIMESTAMP}.rdb"

    if [[ -f "${BACKUP_DIR}/redis_appendonly.aof" ]]; then
        aws s3 cp "${BACKUP_DIR}/redis_appendonly.aof" \
            "s3://${BACKUP_BUCKET}/${AGENT_ID}/redis/aof/${DATE}/redis_aof_${TIMESTAMP}.aof"
    fi

    log_success "Redis backup completed"
}

#######################################
# InfluxDB Backup (Time-series metrics)
#######################################
backup_influxdb() {
    log_info "Starting InfluxDB backup..."

    local influx_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-influxdb \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using InfluxDB pod: ${influx_pod}"

    # Create backup
    kubectl exec -n ${NAMESPACE} ${influx_pod} -- \
        influx backup /tmp/influx-backup-${TIMESTAMP} \
        --org greenlang \
        --bucket combusense_metrics

    # Copy backup files
    kubectl cp "${NAMESPACE}/${influx_pod}:/tmp/influx-backup-${TIMESTAMP}" \
        "${BACKUP_DIR}/influxdb/"

    # Backup CQI history separately (long-term retention)
    log_info "Backing up CQI history..."
    kubectl exec -n ${NAMESPACE} ${influx_pod} -- \
        influx query 'from(bucket: "combusense_metrics") |> range(start: -24h) |> filter(fn: (r) => r._measurement == "cqi")' \
        --raw > "${BACKUP_DIR}/cqi_history_24h.csv"

    # Upload to S3
    aws s3 sync "${BACKUP_DIR}/influxdb/" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/influxdb/${DATE}/"

    aws s3 cp "${BACKUP_DIR}/cqi_history_24h.csv" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/cqi/${DATE}/cqi_history_${TIMESTAMP}.csv" \
        --storage-class GLACIER_IR

    log_success "InfluxDB backup completed"
}

#######################################
# PID Parameters Backup
#######################################
backup_pid_parameters() {
    log_info "Starting PID parameters backup..."

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using application pod: ${app_pod}"

    # Export current PID parameters
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/controller/parameters \
        > "${BACKUP_DIR}/pid_parameters.json"

    # Export controller state
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/controller/state \
        > "${BACKUP_DIR}/controller_state.json"

    # Export tuning history
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/controller/tuning-history \
        > "${BACKUP_DIR}/tuning_history.json"

    # Upload to S3
    aws s3 cp "${BACKUP_DIR}/pid_parameters.json" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/pid/${DATE}/pid_parameters_${TIMESTAMP}.json"

    # Also save as latest
    aws s3 cp "${BACKUP_DIR}/pid_parameters.json" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/pid_parameters.json"

    aws s3 cp "${BACKUP_DIR}/controller_state.json" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/controller_state.json"

    log_success "PID parameters backup completed"
}

#######################################
# Sensor Calibration Backup
#######################################
backup_sensor_calibration() {
    log_info "Starting sensor calibration backup..."

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}')

    # Export sensor calibration data
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/sensors/calibration \
        > "${BACKUP_DIR}/sensor_calibration.json"

    # Export temperature sensors
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/sensors/temperature/calibration \
        > "${BACKUP_DIR}/temperature_calibration.json"

    # Export analyzers (O2, CO)
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/sensors/analyzers/calibration \
        > "${BACKUP_DIR}/analyzer_calibration.json"

    # Export flame scanners
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/sensors/flame-scanners/calibration \
        > "${BACKUP_DIR}/flame_scanner_calibration.json"

    # Upload to S3
    aws s3 sync "${BACKUP_DIR}/" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/sensors/${DATE}/" \
        --exclude "*" \
        --include "*calibration*.json"

    # Also save as latest
    aws s3 cp "${BACKUP_DIR}/sensor_calibration.json" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/latest/sensor_calibration.json"

    log_success "Sensor calibration backup completed"
}

#######################################
# Kafka Backup (Consumer Offsets)
#######################################
backup_kafka() {
    log_info "Starting Kafka backup..."

    local kafka_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-kafka \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using Kafka pod: ${kafka_pod}"

    # Export consumer group offsets
    kubectl exec -n ${NAMESPACE} ${kafka_pod} -- \
        kafka-consumer-groups --bootstrap-server localhost:9092 \
        --all-groups --describe \
        > "${BACKUP_DIR}/kafka_consumer_offsets.txt"

    # Export topic configurations
    for topic in control-outputs sensor-data alarms control-state pid-parameters cqi-metrics safety-events; do
        kubectl exec -n ${NAMESPACE} ${kafka_pod} -- \
            kafka-configs --bootstrap-server localhost:9092 \
            --topic ${AGENT_ID}-${topic} --describe \
            >> "${BACKUP_DIR}/kafka_topic_configs.txt" 2>/dev/null || true
    done

    # Export compacted topics data (control state, PID parameters)
    log_info "Backing up compacted topic data..."
    kubectl exec -n ${NAMESPACE} ${kafka_pod} -- \
        kafka-console-consumer --bootstrap-server localhost:9092 \
        --topic ${AGENT_ID}-control-state \
        --from-beginning --timeout-ms 10000 \
        > "${BACKUP_DIR}/kafka_control_state.txt" 2>/dev/null || true

    kubectl exec -n ${NAMESPACE} ${kafka_pod} -- \
        kafka-console-consumer --bootstrap-server localhost:9092 \
        --topic ${AGENT_ID}-pid-parameters \
        --from-beginning --timeout-ms 10000 \
        > "${BACKUP_DIR}/kafka_pid_parameters.txt" 2>/dev/null || true

    # Upload to S3
    aws s3 cp "${BACKUP_DIR}/kafka_consumer_offsets.txt" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/kafka/${DATE}/consumer_offsets_${TIMESTAMP}.txt"

    aws s3 cp "${BACKUP_DIR}/kafka_topic_configs.txt" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/kafka/${DATE}/topic_configs_${TIMESTAMP}.txt"

    log_success "Kafka backup completed"
}

#######################################
# Configuration Backup
#######################################
backup_configuration() {
    log_info "Starting configuration backup..."

    # Export ConfigMaps
    kubectl get configmap -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -o yaml > "${BACKUP_DIR}/configmaps.yaml"

    kubectl get configmap ${AGENT_ID}-pid-parameters -n ${NAMESPACE} \
        -o yaml > "${BACKUP_DIR}/pid_configmap.yaml" 2>/dev/null || true

    kubectl get configmap ${AGENT_ID}-sensor-calibration -n ${NAMESPACE} \
        -o yaml > "${BACKUP_DIR}/sensor_configmap.yaml" 2>/dev/null || true

    # Export Secrets (encrypted)
    kubectl get secret -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -o yaml > "${BACKUP_DIR}/secrets_encrypted.yaml"

    # Export Deployment specs
    kubectl get deployment -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -o yaml > "${BACKUP_DIR}/deployments.yaml"

    # Export Services
    kubectl get service -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -o yaml > "${BACKUP_DIR}/services.yaml"

    # Export HPAs
    kubectl get hpa -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -o yaml > "${BACKUP_DIR}/hpas.yaml"

    # Export PDBs
    kubectl get pdb -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -o yaml > "${BACKUP_DIR}/pdbs.yaml"

    # Upload to S3
    aws s3 sync "${BACKUP_DIR}/" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/config/k8s/${DATE}/" \
        --exclude "*" \
        --include "*.yaml"

    log_success "Configuration backup completed"
}

#######################################
# Create Backup Manifest
#######################################
create_manifest() {
    log_info "Creating backup manifest..."

    local manifest="${BACKUP_DIR}/manifest.json"

    cat > "${manifest}" << EOF
{
    "agent_id": "${AGENT_ID}",
    "agent_name": "${AGENT_NAME}",
    "backup_timestamp": "${TIMESTAMP}",
    "backup_date": "${DATE}",
    "backup_type": "full",
    "components": {
        "postgresql": {
            "full": "postgresql_full_${TIMESTAMP}.dump",
            "control": "postgresql_control_${TIMESTAMP}.dump",
            "sensors": "postgresql_sensors_${TIMESTAMP}.dump",
            "audit": "postgresql_audit_${TIMESTAMP}.dump"
        },
        "redis": {
            "rdb": "redis_dump_${TIMESTAMP}.rdb",
            "aof": "redis_aof_${TIMESTAMP}.aof"
        },
        "influxdb": {
            "backup_dir": "influxdb/${DATE}/",
            "cqi_history": "cqi_history_${TIMESTAMP}.csv"
        },
        "kafka": {
            "consumer_offsets": "consumer_offsets_${TIMESTAMP}.txt",
            "topic_configs": "topic_configs_${TIMESTAMP}.txt"
        },
        "config": {
            "pid_parameters": "pid_parameters_${TIMESTAMP}.json",
            "controller_state": "controller_state.json",
            "sensor_calibration": "sensor_calibration.json"
        }
    },
    "retention": {
        "standard": "${RETENTION_DAYS} days",
        "cqi_history": "${CQI_HISTORY_RETENTION_YEARS} years",
        "audit_logs": "7 years"
    },
    "verification": {
        "checksum_algorithm": "sha256",
        "verified": false
    }
}
EOF

    # Calculate checksums
    log_info "Calculating checksums..."
    find "${BACKUP_DIR}" -type f ! -name "manifest.json" -exec sha256sum {} \; \
        > "${BACKUP_DIR}/checksums.sha256"

    # Upload manifest
    aws s3 cp "${manifest}" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${DATE}/manifest_${TIMESTAMP}.json"

    aws s3 cp "${BACKUP_DIR}/checksums.sha256" \
        "s3://${BACKUP_BUCKET}/${AGENT_ID}/manifests/${DATE}/checksums_${TIMESTAMP}.sha256"

    log_success "Backup manifest created"
}

#######################################
# Cleanup Old Backups
#######################################
cleanup_old_backups() {
    log_info "Cleaning up backups older than ${RETENTION_DAYS} days..."

    local cutoff_date=$(date -d "-${RETENTION_DAYS} days" +%Y-%m-%d)

    # List and delete old backups (except audit and CQI which have longer retention)
    aws s3 ls "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/" | while read -r line; do
        local dir_date=$(echo "$line" | awk '{print $2}' | tr -d '/')
        if [[ "${dir_date}" < "${cutoff_date}" ]]; then
            log_info "Deleting old backup: ${dir_date}"
            aws s3 rm "s3://${BACKUP_BUCKET}/${AGENT_ID}/postgresql/full/${dir_date}/" --recursive
        fi
    done

    # Cleanup local backup directory
    rm -rf "${BACKUP_DIR}"

    log_success "Cleanup completed"
}

#######################################
# Send Notification
#######################################
send_notification() {
    local status=$1
    local message=$2

    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color="good"
        [[ "${status}" != "success" ]] && color="danger"

        curl -s -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-type: application/json' \
            -d "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"GL-005 Combusense Backup ${status}\",
                    \"text\": \"${message}\",
                    \"fields\": [
                        {\"title\": \"Timestamp\", \"value\": \"${TIMESTAMP}\", \"short\": true},
                        {\"title\": \"Environment\", \"value\": \"${NAMESPACE}\", \"short\": true}
                    ]
                }]
            }"
    fi

    # PagerDuty notification for failures
    if [[ "${status}" == "failure" && -n "${PAGERDUTY_KEY:-}" ]]; then
        curl -s -X POST "https://events.pagerduty.com/v2/enqueue" \
            -H 'Content-type: application/json' \
            -d "{
                \"routing_key\": \"${PAGERDUTY_KEY}\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"GL-005 Combusense backup failed\",
                    \"severity\": \"warning\",
                    \"source\": \"backup-script\",
                    \"custom_details\": {
                        \"message\": \"${message}\",
                        \"timestamp\": \"${TIMESTAMP}\"
                    }
                }
            }"
    fi
}

#######################################
# Main Execution
#######################################
main() {
    local start_time=$(date +%s)
    local exit_code=0

    log_info "=========================================="
    log_info "GL-005 Combusense Backup Starting"
    log_info "Timestamp: ${TIMESTAMP}"
    log_info "=========================================="

    # Run pre-flight checks
    preflight_checks

    # Execute backups
    {
        backup_postgresql
        backup_redis
        backup_influxdb
        backup_pid_parameters
        backup_sensor_calibration
        backup_kafka
        backup_configuration
        create_manifest
        cleanup_old_backups
    } || {
        exit_code=$?
        log_error "Backup failed with exit code: ${exit_code}"
    }

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [[ ${exit_code} -eq 0 ]]; then
        log_success "=========================================="
        log_success "Backup completed successfully"
        log_success "Duration: ${duration} seconds"
        log_success "=========================================="
        send_notification "success" "Backup completed in ${duration}s"
    else
        log_error "=========================================="
        log_error "Backup failed"
        log_error "Duration: ${duration} seconds"
        log_error "=========================================="
        send_notification "failure" "Backup failed after ${duration}s"
    fi

    exit ${exit_code}
}

# Run main function
main "$@"
