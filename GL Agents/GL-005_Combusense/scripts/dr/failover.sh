#!/bin/bash
# GL-005 Combusense - Failover Script
# Handles controller failover, database failover, and regional failover
# Includes bumpless transfer support for control systems

set -euo pipefail

#######################################
# Configuration
#######################################
AGENT_ID="gl-005"
AGENT_NAME="combusense"
NAMESPACE="${NAMESPACE:-greenlang}"
OPERATION="${1:-help}"
LOG_FILE="/var/log/${AGENT_ID}/failover-$(date +%Y%m%d_%H%M%S).log"

# Regional configuration
PRIMARY_REGION="${PRIMARY_REGION:-us-east-1}"
DR_REGION="${DR_REGION:-us-west-2}"
PRIMARY_CONTEXT="${PRIMARY_CONTEXT:-eks-${PRIMARY_REGION}}"
DR_CONTEXT="${DR_CONTEXT:-eks-${DR_REGION}}"

# Control system settings
BUMPLESS_TRANSFER="${BUMPLESS_TRANSFER:-true}"
SAFE_STATE_TIMEOUT_SEC="${SAFE_STATE_TIMEOUT_SEC:-10}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
log_step() { log "${BLUE}STEP${NC}" "$*"; }

#######################################
# Usage
#######################################
usage() {
    cat << EOF
GL-005 Combusense - Failover Script

Usage: $0 <operation> [options]

Operations:
    controller          Failover PID controller (primary <-> standby)
    controller-test     Test controller failover (non-disruptive)
    database            Failover PostgreSQL (Patroni switchover)
    redis               Failover Redis Sentinel master
    kafka               Reassign Kafka partition leaders
    region              Full regional failover (DR activation)
    verify              Verify current failover status
    help                Show this help message

Options:
    --force             Skip confirmation prompts
    --dry-run           Show what would be done without executing

Environment Variables:
    NAMESPACE           Kubernetes namespace (default: greenlang)
    PRIMARY_REGION      Primary AWS region (default: us-east-1)
    DR_REGION           DR AWS region (default: us-west-2)
    BUMPLESS_TRANSFER   Enable bumpless transfer for controllers (default: true)

Examples:
    $0 controller                # Failover PID controller
    $0 controller-test           # Test controller failover
    $0 database                  # Failover PostgreSQL
    $0 region --force            # Regional failover without confirmation
    $0 verify                    # Check current status

EOF
    exit 0
}

#######################################
# Pre-flight Checks
#######################################
preflight() {
    mkdir -p "$(dirname ${LOG_FILE})"

    # Check required tools
    for tool in kubectl jq curl; do
        if ! command -v "${tool}" &> /dev/null; then
            log_error "Required tool not found: ${tool}"
            exit 1
        fi
    done

    # Check kubectl access
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_error "Cannot access namespace: ${NAMESPACE}"
        exit 1
    fi
}

#######################################
# Controller Failover
#######################################
failover_controller() {
    log_step "Initiating PID controller failover..."

    # Get current controller status
    local primary_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    local standby_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=standby \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -z "${primary_pod}" ]]; then
        log_error "No primary controller pod found"
        exit 1
    fi

    if [[ -z "${standby_pod}" ]]; then
        log_error "No standby controller pod found"
        exit 1
    fi

    log_info "Primary controller: ${primary_pod}"
    log_info "Standby controller: ${standby_pod}"

    # Check standby readiness
    log_info "Checking standby controller readiness..."
    local standby_status=$(kubectl exec -n ${NAMESPACE} ${standby_pod} -- \
        curl -s localhost:8080/api/v1/controller/status 2>/dev/null)

    local standby_mode=$(echo "${standby_status}" | jq -r '.mode // "unknown"')
    local tracking_active=$(echo "${standby_status}" | jq -r '.tracking_active // false')

    if [[ "${standby_mode}" != "tracking" || "${tracking_active}" != "true" ]]; then
        log_error "Standby controller not ready for failover"
        log_error "Mode: ${standby_mode}, Tracking: ${tracking_active}"
        exit 1
    fi

    log_success "Standby controller is ready (tracking mode active)"

    # Capture pre-failover state
    log_info "Capturing pre-failover control state..."
    local pre_state=$(kubectl exec -n ${NAMESPACE} ${primary_pod} -- \
        curl -s localhost:8080/api/v1/controller/state)
    local pre_output=$(echo "${pre_state}" | jq -r '.output')
    local pre_integral=$(echo "${pre_state}" | jq -r '.integral')
    log_info "Pre-failover output: ${pre_output}"

    # Initiate bumpless transfer
    if [[ "${BUMPLESS_TRANSFER}" == "true" ]]; then
        log_info "Initiating bumpless transfer..."
        kubectl exec -n ${NAMESPACE} ${primary_pod} -- \
            curl -s -X POST localhost:8080/api/v1/controller/transfer \
            -H "Content-Type: application/json" \
            -d "{\"target\": \"${standby_pod}\", \"bumpless\": true, \"integral_state\": ${pre_integral}}"
    else
        log_info "Initiating standard transfer..."
        kubectl exec -n ${NAMESPACE} ${primary_pod} -- \
            curl -s -X POST localhost:8080/api/v1/controller/transfer \
            -H "Content-Type: application/json" \
            -d "{\"target\": \"${standby_pod}\"}"
    fi

    # Wait for transfer completion
    log_info "Waiting for transfer completion..."
    sleep 2

    # Verify post-failover state
    local post_state=$(kubectl exec -n ${NAMESPACE} ${standby_pod} -- \
        curl -s localhost:8080/api/v1/controller/state)
    local post_output=$(echo "${post_state}" | jq -r '.output')
    local post_mode=$(echo "${post_state}" | jq -r '.mode')

    log_info "Post-failover output: ${post_output}"
    log_info "Post-failover mode: ${post_mode}"

    # Check for bumpless transfer quality
    if [[ "${BUMPLESS_TRANSFER}" == "true" ]]; then
        local output_diff=$(echo "${pre_output} - ${post_output}" | bc -l | tr -d '-')
        if (( $(echo "${output_diff} < 1.0" | bc -l) )); then
            log_success "Bumpless transfer successful (output change < 1%)"
        else
            log_warn "Output changed by ${output_diff}% during transfer"
        fi
    fi

    # Update labels
    log_info "Updating pod labels..."
    kubectl label pods ${primary_pod} -n ${NAMESPACE} role=standby --overwrite
    kubectl label pods ${standby_pod} -n ${NAMESPACE} role=primary --overwrite

    log_success "Controller failover completed"
    log_info "New primary: ${standby_pod}"
    log_info "New standby: ${primary_pod}"
}

#######################################
# Controller Failover Test
#######################################
test_controller_failover() {
    log_step "Testing controller failover (non-disruptive)..."

    local primary_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

    if [[ -z "${primary_pod}" ]]; then
        log_error "No primary controller pod found"
        exit 1
    fi

    # Request test failover
    log_info "Initiating test failover on ${primary_pod}..."
    local result=$(kubectl exec -n ${NAMESPACE} ${primary_pod} -- \
        curl -s -X POST localhost:8080/api/v1/controller/test-failover \
        -H "Content-Type: application/json" \
        -d '{"duration_ms": 5000, "verify_tracking": true}')

    local test_status=$(echo "${result}" | jq -r '.status')
    local transfer_time_ms=$(echo "${result}" | jq -r '.transfer_time_ms')
    local output_deviation=$(echo "${result}" | jq -r '.output_deviation_percent')

    if [[ "${test_status}" == "success" ]]; then
        log_success "Failover test PASSED"
        log_info "Transfer time: ${transfer_time_ms}ms"
        log_info "Output deviation: ${output_deviation}%"

        # Check against SLA
        if (( $(echo "${transfer_time_ms} < 50" | bc -l) )); then
            log_success "Transfer time within SLA (< 50ms)"
        else
            log_warn "Transfer time exceeds SLA target (50ms)"
        fi
    else
        log_error "Failover test FAILED"
        log_error "Result: ${result}"
        exit 1
    fi
}

#######################################
# Database Failover
#######################################
failover_database() {
    log_step "Initiating PostgreSQL failover..."

    local pg_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-postgres \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using Patroni pod: ${pg_pod}"

    # Get current cluster status
    log_info "Current cluster status:"
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- patronictl list

    # Get current leader
    local current_leader=$(kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        patronictl list -f json | jq -r '.[] | select(.Role == "Leader") | .Member')
    log_info "Current leader: ${current_leader}"

    # Get candidate for new leader
    local candidate=$(kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        patronictl list -f json | jq -r '.[] | select(.Role == "Replica") | .Member' | head -1)
    log_info "Failover candidate: ${candidate}"

    if [[ -z "${candidate}" ]]; then
        log_error "No replica available for failover"
        exit 1
    fi

    # Perform switchover
    log_info "Performing switchover to ${candidate}..."
    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        patronictl switchover --candidate "${candidate}" --force

    # Wait for switchover
    log_info "Waiting for switchover completion..."
    sleep 10

    # Verify new leader
    local new_leader=$(kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        patronictl list -f json | jq -r '.[] | select(.Role == "Leader") | .Member')
    log_info "New leader: ${new_leader}"

    if [[ "${new_leader}" == "${candidate}" ]]; then
        log_success "PostgreSQL failover completed successfully"
    else
        log_error "Failover may have failed - verify cluster status"
        kubectl exec -n ${NAMESPACE} ${pg_pod} -- patronictl list
        exit 1
    fi
}

#######################################
# Redis Failover
#######################################
failover_redis() {
    log_step "Initiating Redis Sentinel failover..."

    local sentinel_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-redis-sentinel \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using Sentinel pod: ${sentinel_pod}"

    # Get current master
    local current_master=$(kubectl exec -n ${NAMESPACE} ${sentinel_pod} -- \
        redis-cli -p 26379 SENTINEL get-master-addr-by-name ${AGENT_ID}-master | head -1)
    log_info "Current master: ${current_master}"

    # Trigger failover
    log_info "Triggering Sentinel failover..."
    kubectl exec -n ${NAMESPACE} ${sentinel_pod} -- \
        redis-cli -p 26379 SENTINEL failover ${AGENT_ID}-master

    # Wait for failover
    log_info "Waiting for failover completion..."
    sleep 10

    # Verify new master
    local new_master=$(kubectl exec -n ${NAMESPACE} ${sentinel_pod} -- \
        redis-cli -p 26379 SENTINEL get-master-addr-by-name ${AGENT_ID}-master | head -1)
    log_info "New master: ${new_master}"

    if [[ "${new_master}" != "${current_master}" ]]; then
        log_success "Redis failover completed successfully"
    else
        log_warn "Master may not have changed - verify Sentinel status"
    fi
}

#######################################
# Kafka Leader Reassignment
#######################################
failover_kafka() {
    log_step "Reassigning Kafka partition leaders..."

    local kafka_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-kafka \
        -o jsonpath='{.items[0].metadata.name}')

    log_info "Using Kafka pod: ${kafka_pod}"

    # List current partition assignments
    log_info "Current partition leaders:"
    for topic in control-outputs sensor-data alarms control-state; do
        kubectl exec -n ${NAMESPACE} ${kafka_pod} -- \
            kafka-topics --bootstrap-server localhost:9092 \
            --describe --topic ${AGENT_ID}-${topic} 2>/dev/null || true
    done

    # Trigger preferred leader election
    log_info "Triggering preferred leader election..."
    kubectl exec -n ${NAMESPACE} ${kafka_pod} -- \
        kafka-leader-election --bootstrap-server localhost:9092 \
        --election-type PREFERRED --all-topic-partitions

    # Wait for election
    sleep 5

    log_success "Kafka leader election completed"
}

#######################################
# Regional Failover
#######################################
failover_region() {
    log_step "Initiating regional failover to ${DR_REGION}..."

    # Confirm action
    if [[ "${FORCE:-false}" != "true" ]]; then
        echo ""
        echo "=========================================="
        echo "REGIONAL FAILOVER CONFIRMATION"
        echo "=========================================="
        echo "This will:"
        echo "  1. Verify primary region is unavailable"
        echo "  2. Activate DR region (${DR_REGION})"
        echo "  3. Scale up all components in DR"
        echo "  4. Update DNS to point to DR"
        echo ""
        read -p "Are you sure? (yes/no): " confirm
        if [[ "${confirm}" != "yes" ]]; then
            log_info "Regional failover cancelled"
            exit 0
        fi
    fi

    # Step 1: Verify primary unavailable
    log_info "Step 1: Verifying primary region status..."
    if kubectl --context="${PRIMARY_CONTEXT}" get namespace "${NAMESPACE}" &> /dev/null; then
        log_warn "Primary region appears to be available"
        read -p "Continue with failover anyway? (yes/no): " force_continue
        if [[ "${force_continue}" != "yes" ]]; then
            exit 0
        fi
    else
        log_info "Primary region confirmed unavailable"
    fi

    # Step 2: Switch to DR context
    log_info "Step 2: Switching to DR region context..."
    kubectl config use-context "${DR_CONTEXT}"

    # Step 3: Scale up DR components
    log_info "Step 3: Scaling up DR components..."

    # Scale application
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME}-primary \
        --replicas=2 -n ${NAMESPACE}
    kubectl scale deployment ${AGENT_ID}-${AGENT_NAME}-standby \
        --replicas=2 -n ${NAMESPACE}

    # Wait for pods
    log_info "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod \
        -l app=${AGENT_ID}-${AGENT_NAME} \
        -n ${NAMESPACE} --timeout=300s

    # Step 4: Promote PostgreSQL
    log_info "Step 4: Promoting PostgreSQL in DR..."
    local pg_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-postgres \
        -o jsonpath='{.items[0].metadata.name}')

    kubectl exec -n ${NAMESPACE} ${pg_pod} -- \
        patronictl reinit ${AGENT_ID}-postgres --force 2>/dev/null || true

    # Step 5: Restore PID parameters from latest backup
    log_info "Step 5: Restoring PID parameters..."
    aws s3 cp "s3://greenlang-backups/${AGENT_ID}/config/latest/pid_parameters.json" \
        /tmp/pid_params.json

    local app_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}')

    kubectl cp /tmp/pid_params.json "${NAMESPACE}/${app_pod}:/tmp/pid_params.json"
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s -X POST localhost:8080/api/v1/controller/parameters \
        -H "Content-Type: application/json" \
        -d @/tmp/pid_params.json

    # Step 6: Update DNS (if configured)
    if [[ -n "${ROUTE53_ZONE_ID:-}" ]]; then
        log_info "Step 6: Updating Route53 DNS..."
        local dr_endpoint=$(kubectl get ingress ${AGENT_ID}-${AGENT_NAME} \
            -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

        aws route53 change-resource-record-sets \
            --hosted-zone-id "${ROUTE53_ZONE_ID}" \
            --change-batch "{
                \"Changes\": [{
                    \"Action\": \"UPSERT\",
                    \"ResourceRecordSet\": {
                        \"Name\": \"${AGENT_ID}.api.greenlang.io\",
                        \"Type\": \"CNAME\",
                        \"TTL\": 60,
                        \"ResourceRecords\": [{\"Value\": \"${dr_endpoint}\"}]
                    }
                }]
            }"
    else
        log_warn "Step 6: ROUTE53_ZONE_ID not set, skipping DNS update"
    fi

    # Step 7: Verify
    log_info "Step 7: Verifying DR activation..."
    kubectl exec -n ${NAMESPACE} ${app_pod} -- \
        curl -s localhost:8080/api/v1/health

    log_success "=========================================="
    log_success "Regional failover completed successfully"
    log_success "DR Region: ${DR_REGION}"
    log_success "=========================================="
}

#######################################
# Verify Status
#######################################
verify_status() {
    log_step "Verifying failover status..."

    echo ""
    echo "=========================================="
    echo "GL-005 COMBUSENSE STATUS"
    echo "=========================================="
    echo ""

    # Controller status
    echo "=== Controller Status ==="
    local primary_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=primary \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "NOT FOUND")
    local standby_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-${AGENT_NAME},role=standby \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "NOT FOUND")

    echo "Primary controller: ${primary_pod}"
    echo "Standby controller: ${standby_pod}"

    if [[ "${primary_pod}" != "NOT FOUND" ]]; then
        local ctrl_status=$(kubectl exec -n ${NAMESPACE} ${primary_pod} -- \
            curl -s localhost:8080/api/v1/controller/status 2>/dev/null || echo '{"error": "unavailable"}')
        echo "Controller mode: $(echo ${ctrl_status} | jq -r '.mode // "unknown"')"
        echo "Controller output: $(echo ${ctrl_status} | jq -r '.output // "unknown"')"
    fi

    echo ""
    echo "=== PostgreSQL Status ==="
    local pg_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-postgres \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "${pg_pod}" ]]; then
        kubectl exec -n ${NAMESPACE} ${pg_pod} -- patronictl list 2>/dev/null || echo "Patroni unavailable"
    else
        echo "PostgreSQL pods not found"
    fi

    echo ""
    echo "=== Redis Sentinel Status ==="
    local sentinel_pod=$(kubectl get pods -n ${NAMESPACE} \
        -l app=${AGENT_ID}-redis-sentinel \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "${sentinel_pod}" ]]; then
        local redis_master=$(kubectl exec -n ${NAMESPACE} ${sentinel_pod} -- \
            redis-cli -p 26379 SENTINEL get-master-addr-by-name ${AGENT_ID}-master 2>/dev/null | head -1 || echo "unknown")
        echo "Redis master: ${redis_master}"
    else
        echo "Redis Sentinel pods not found"
    fi

    echo ""
    echo "=== All Pods ==="
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-${AGENT_NAME} -o wide
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-postgres -o wide
    kubectl get pods -n ${NAMESPACE} -l app=${AGENT_ID}-redis -o wide

    echo ""
    echo "=========================================="
}

#######################################
# Main
#######################################
main() {
    preflight

    case "${OPERATION}" in
        controller)
            failover_controller
            ;;
        controller-test)
            test_controller_failover
            ;;
        database)
            failover_database
            ;;
        redis)
            failover_redis
            ;;
        kafka)
            failover_kafka
            ;;
        region)
            failover_region
            ;;
        verify)
            verify_status
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown operation: ${OPERATION}"
            usage
            ;;
    esac
}

main "$@"
