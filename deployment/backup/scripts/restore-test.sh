#!/bin/bash
# =============================================================================
# Restore Test Script for GreenLang
# INFRA-001: Backup and Disaster Recovery
# Version: 1.0.0
# =============================================================================
#
# This script performs automated restore testing to validate backup integrity:
# - Creates test restore instances
# - Validates data integrity
# - Runs application connectivity tests
# - Cleans up test resources
#
# Usage: ./restore-test.sh [--type rds|velero|full] [--cleanup] [--dry-run]
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/greenlang/restore-test-$(date +%Y%m%d).log"
TEST_TIMESTAMP=$(date +%Y%m%d%H%M)

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
RDS_INSTANCE="${RDS_INSTANCE:-greenlang-postgres}"
TEST_RDS_INSTANCE="greenlang-restore-test-${TEST_TIMESTAMP}"
TEST_INSTANCE_CLASS="${TEST_INSTANCE_CLASS:-db.t3.small}"

# Kubernetes Configuration
KUBE_NAMESPACE="${KUBE_NAMESPACE:-greenlang}"
VELERO_NAMESPACE="${VELERO_NAMESPACE:-velero}"
TEST_NAMESPACE="greenlang-restore-test"

# Database Configuration
DB_USER="${DB_USER:-greenlang_admin}"
DB_PASSWORD="${DB_PASSWORD:-}"
DB_NAME="${DB_NAME:-greenlang}"

# Notification Configuration
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"

# Parse arguments
TEST_TYPE="full"
CLEANUP=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--type rds|velero|full] [--no-cleanup] [--dry-run]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Logging Functions
# =============================================================================

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# =============================================================================
# Test Results
# =============================================================================

declare -A TEST_RESULTS
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

record_test() {
    local test_name=$1
    local status=$2  # PASS, FAIL
    local details=$3
    local duration=$4

    TEST_RESULTS["$test_name"]="$status|$details|$duration"
    ((TOTAL_TESTS++))

    if [ "$status" = "PASS" ]; then
        ((PASSED_TESTS++))
        log_success "[PASS] $test_name: $details (${duration}s)"
    else
        ((FAILED_TESTS++))
        log_error "[FAIL] $test_name: $details (${duration}s)"
    fi
}

# =============================================================================
# Utility Functions
# =============================================================================

cleanup_resources() {
    log_info "=== Cleaning Up Test Resources ==="

    # Cleanup RDS test instance
    if aws rds describe-db-instances --db-instance-identifier "$TEST_RDS_INSTANCE" &>/dev/null; then
        log_info "Deleting test RDS instance: $TEST_RDS_INSTANCE"
        aws rds delete-db-instance \
            --db-instance-identifier "$TEST_RDS_INSTANCE" \
            --skip-final-snapshot \
            --delete-automated-backups 2>/dev/null || true
    fi

    # Cleanup Kubernetes test namespace
    if kubectl get namespace "$TEST_NAMESPACE" &>/dev/null; then
        log_info "Deleting test namespace: $TEST_NAMESPACE"
        kubectl delete namespace "$TEST_NAMESPACE" --wait=false 2>/dev/null || true
    fi

    # Cleanup Velero test restores
    kubectl delete restore -n "$VELERO_NAMESPACE" -l test=restore-test 2>/dev/null || true

    log_info "Cleanup completed"
}

wait_for_rds() {
    local instance=$1
    local timeout=${2:-1800}  # 30 minutes default
    local start_time=$(date +%s)

    log_info "Waiting for RDS instance $instance to be available..."

    while true; do
        local status=$(aws rds describe-db-instances \
            --db-instance-identifier "$instance" \
            --query 'DBInstances[0].DBInstanceStatus' \
            --output text 2>/dev/null || echo "pending")

        local elapsed=$(($(date +%s) - start_time))

        if [ "$status" = "available" ]; then
            log_info "RDS instance $instance is available (${elapsed}s)"
            return 0
        elif [ "$elapsed" -gt "$timeout" ]; then
            log_error "Timeout waiting for RDS instance $instance"
            return 1
        fi

        log_info "Status: $status (${elapsed}s elapsed)"
        sleep 30
    done
}

# =============================================================================
# RDS Restore Test
# =============================================================================

test_rds_restore() {
    log_info "=========================================="
    log_info "RDS POINT-IN-TIME RECOVERY TEST"
    log_info "=========================================="

    local start_time=$(date +%s)

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create PITR restore: $TEST_RDS_INSTANCE"
        record_test "rds_pitr_restore" "PASS" "Dry run - skipped" "0"
        return 0
    fi

    # Step 1: Get latest restorable time
    log_info "[Step 1/6] Getting latest restorable time..."
    local restore_time=$(aws rds describe-db-instances \
        --db-instance-identifier "$RDS_INSTANCE" \
        --query 'DBInstances[0].LatestRestorableTime' \
        --output text)

    if [ -z "$restore_time" ] || [ "$restore_time" = "None" ]; then
        record_test "rds_pitr_restore" "FAIL" "No restorable time available" "$(($(date +%s) - start_time))"
        return 1
    fi

    log_info "Restoring to: $restore_time"

    # Step 2: Create PITR restore
    log_info "[Step 2/6] Creating PITR restore..."
    local restore_start=$(date +%s)

    aws rds restore-db-instance-to-point-in-time \
        --source-db-instance-identifier "$RDS_INSTANCE" \
        --target-db-instance-identifier "$TEST_RDS_INSTANCE" \
        --restore-time "$restore_time" \
        --db-instance-class "$TEST_INSTANCE_CLASS" \
        --no-multi-az \
        --no-publicly-accessible \
        --tags Key=Purpose,Value=restore-test Key=AutoDelete,Value=true

    # Step 3: Wait for restore to complete
    log_info "[Step 3/6] Waiting for restore to complete..."
    if ! wait_for_rds "$TEST_RDS_INSTANCE" 2400; then
        record_test "rds_pitr_restore" "FAIL" "Restore timeout" "$(($(date +%s) - start_time))"
        return 1
    fi

    local restore_duration=$(($(date +%s) - restore_start))
    log_info "Restore completed in ${restore_duration}s"

    # Step 4: Get endpoint
    log_info "[Step 4/6] Getting test instance endpoint..."
    local test_endpoint=$(aws rds describe-db-instances \
        --db-instance-identifier "$TEST_RDS_INSTANCE" \
        --query 'DBInstances[0].Endpoint.Address' \
        --output text)

    log_info "Test endpoint: $test_endpoint"

    # Step 5: Validate database
    log_info "[Step 5/6] Validating restored database..."

    # Test connectivity
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$test_endpoint" -U "$DB_USER" -d "$DB_NAME" \
        -c "SELECT 1;" &>/dev/null; then
        record_test "rds_db_connectivity" "FAIL" "Cannot connect to restored database" "$(($(date +%s) - start_time))"
        return 1
    fi
    record_test "rds_db_connectivity" "PASS" "Database connectivity verified" "0"

    # Validate tables exist
    local table_count=$(PGPASSWORD="$DB_PASSWORD" psql -h "$test_endpoint" -U "$DB_USER" -d "$DB_NAME" \
        -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | tr -d ' ')

    if [ "$table_count" -gt 0 ]; then
        record_test "rds_table_count" "PASS" "$table_count tables found" "0"
    else
        record_test "rds_table_count" "FAIL" "No tables found in restored database" "0"
    fi

    # Validate data integrity
    local integrity_check=$(PGPASSWORD="$DB_PASSWORD" psql -h "$test_endpoint" -U "$DB_USER" -d "$DB_NAME" -t << EOF
SELECT
    CASE
        WHEN (SELECT COUNT(*) FROM organizations) > 0 THEN 'OK'
        ELSE 'EMPTY'
    END as organizations_check;
EOF
    )

    if [[ "$integrity_check" == *"OK"* ]]; then
        record_test "rds_data_integrity" "PASS" "Data integrity verified" "0"
    else
        record_test "rds_data_integrity" "WARN" "Organizations table is empty" "0"
    fi

    # Step 6: Test write operations
    log_info "[Step 6/6] Testing write operations..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$test_endpoint" -U "$DB_USER" -d "$DB_NAME" << EOF
CREATE TABLE IF NOT EXISTS _restore_test_verification (
    id SERIAL PRIMARY KEY,
    test_timestamp TIMESTAMP DEFAULT NOW(),
    test_value TEXT
);
INSERT INTO _restore_test_verification (test_value) VALUES ('Restore test at $(date)');
SELECT * FROM _restore_test_verification;
DROP TABLE _restore_test_verification;
EOF

    if [ $? -eq 0 ]; then
        record_test "rds_write_test" "PASS" "Write operations successful" "0"
    else
        record_test "rds_write_test" "FAIL" "Write operations failed" "0"
    fi

    local total_duration=$(($(date +%s) - start_time))
    record_test "rds_pitr_restore" "PASS" "PITR restore successful" "$total_duration"

    return 0
}

# =============================================================================
# Velero Restore Test
# =============================================================================

test_velero_restore() {
    log_info "=========================================="
    log_info "VELERO KUBERNETES RESTORE TEST"
    log_info "=========================================="

    local start_time=$(date +%s)

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would test Velero restore"
        record_test "velero_restore" "PASS" "Dry run - skipped" "0"
        return 0
    fi

    # Step 1: Find latest successful backup
    log_info "[Step 1/5] Finding latest successful backup..."
    local latest_backup=$(kubectl get backup -n "$VELERO_NAMESPACE" \
        --field-selector="status.phase=Completed" \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo "")

    if [ -z "$latest_backup" ]; then
        record_test "velero_backup_exists" "FAIL" "No completed backups found" "$(($(date +%s) - start_time))"
        return 1
    fi

    log_info "Latest backup: $latest_backup"
    record_test "velero_backup_exists" "PASS" "Found backup: $latest_backup" "0"

    # Step 2: Create test namespace
    log_info "[Step 2/5] Creating test namespace..."
    kubectl create namespace "$TEST_NAMESPACE" 2>/dev/null || true
    kubectl label namespace "$TEST_NAMESPACE" test=restore-test 2>/dev/null || true

    # Step 3: Create restore
    log_info "[Step 3/5] Creating Velero restore..."
    local restore_name="restore-test-${TEST_TIMESTAMP}"

    cat <<EOF | kubectl apply -f -
apiVersion: velero.io/v1
kind: Restore
metadata:
  name: ${restore_name}
  namespace: ${VELERO_NAMESPACE}
  labels:
    test: restore-test
spec:
  backupName: ${latest_backup}
  includedNamespaces:
    - ${KUBE_NAMESPACE}
  namespaceMapping:
    ${KUBE_NAMESPACE}: ${TEST_NAMESPACE}
  excludedResources:
    - persistentvolumeclaims
    - persistentvolumes
    - secrets
  restorePVs: false
EOF

    # Step 4: Wait for restore to complete
    log_info "[Step 4/5] Waiting for restore to complete..."
    local restore_start=$(date +%s)
    local timeout=600  # 10 minutes

    while true; do
        local status=$(kubectl get restore "$restore_name" -n "$VELERO_NAMESPACE" \
            -o jsonpath='{.status.phase}' 2>/dev/null || echo "InProgress")

        local elapsed=$(($(date +%s) - restore_start))

        if [ "$status" = "Completed" ]; then
            log_info "Restore completed in ${elapsed}s"
            break
        elif [ "$status" = "Failed" ]; then
            local errors=$(kubectl get restore "$restore_name" -n "$VELERO_NAMESPACE" \
                -o jsonpath='{.status.errors}' 2>/dev/null)
            record_test "velero_restore" "FAIL" "Restore failed: $errors" "$elapsed"
            return 1
        elif [ "$elapsed" -gt "$timeout" ]; then
            record_test "velero_restore" "FAIL" "Restore timeout after ${elapsed}s" "$elapsed"
            return 1
        fi

        log_info "Status: $status (${elapsed}s elapsed)"
        sleep 15
    done

    # Step 5: Validate restored resources
    log_info "[Step 5/5] Validating restored resources..."

    # Check for configmaps
    local configmap_count=$(kubectl get configmaps -n "$TEST_NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$configmap_count" -gt 0 ]; then
        record_test "velero_configmaps" "PASS" "$configmap_count ConfigMaps restored" "0"
    else
        record_test "velero_configmaps" "WARN" "No ConfigMaps in restore" "0"
    fi

    # Check for deployments
    local deployment_count=$(kubectl get deployments -n "$TEST_NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$deployment_count" -gt 0 ]; then
        record_test "velero_deployments" "PASS" "$deployment_count Deployments restored" "0"
    else
        record_test "velero_deployments" "WARN" "No Deployments in restore" "0"
    fi

    # Check for services
    local service_count=$(kubectl get services -n "$TEST_NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$service_count" -gt 0 ]; then
        record_test "velero_services" "PASS" "$service_count Services restored" "0"
    else
        record_test "velero_services" "WARN" "No Services in restore" "0"
    fi

    local total_duration=$(($(date +%s) - start_time))
    record_test "velero_restore" "PASS" "Velero restore test successful" "$total_duration"

    return 0
}

# =============================================================================
# Full Restore Test
# =============================================================================

test_full_restore() {
    log_info "=========================================="
    log_info "FULL RESTORE TEST"
    log_info "=========================================="

    test_rds_restore
    test_velero_restore
}

# =============================================================================
# Generate Report
# =============================================================================

generate_report() {
    log_info "=========================================="
    log_info "GENERATING TEST REPORT"
    log_info "=========================================="

    local report_file="/tmp/restore-test-report-${TEST_TIMESTAMP}.json"

    local report_json=$(cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "test_type": "$TEST_TYPE",
    "summary": {
        "total_tests": $TOTAL_TESTS,
        "passed": $PASSED_TESTS,
        "failed": $FAILED_TESTS,
        "status": "$([ $FAILED_TESTS -eq 0 ] && echo "PASSED" || echo "FAILED")"
    },
    "tests": {
EOF
)

    local first=true
    for test_name in "${!TEST_RESULTS[@]}"; do
        local result="${TEST_RESULTS[$test_name]}"
        local status=$(echo "$result" | cut -d'|' -f1)
        local details=$(echo "$result" | cut -d'|' -f2)
        local duration=$(echo "$result" | cut -d'|' -f3)

        if [ "$first" = true ]; then
            first=false
        else
            report_json+=","
        fi

        report_json+="
        \"$test_name\": {
            \"status\": \"$status\",
            \"details\": \"$details\",
            \"duration_seconds\": $duration
        }"
    done

    report_json+="
    }
}"

    echo "$report_json" > "$report_file"
    log_info "Report saved to: $report_file"

    # Send notifications
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        local color="good"
        [ $FAILED_TESTS -gt 0 ] && color="danger"

        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"GreenLang Restore Test Results\",
                    \"text\": \"Type: $TEST_TYPE\nStatus: $([ $FAILED_TESTS -eq 0 ] && echo 'PASSED' || echo 'FAILED')\",
                    \"fields\": [
                        {\"title\": \"Total Tests\", \"value\": \"$TOTAL_TESTS\", \"short\": true},
                        {\"title\": \"Passed\", \"value\": \"$PASSED_TESTS\", \"short\": true},
                        {\"title\": \"Failed\", \"value\": \"$FAILED_TESTS\", \"short\": true}
                    ]
                }]
            }" "$SLACK_WEBHOOK_URL" > /dev/null
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    mkdir -p "$(dirname "$LOG_FILE")"

    log_info "=========================================="
    log_info "GreenLang Restore Test"
    log_info "Test Type: $TEST_TYPE"
    log_info "Started: $(date)"
    log_info "=========================================="

    # Set up cleanup trap
    if [ "$CLEANUP" = true ]; then
        trap cleanup_resources EXIT
    fi

    # Run tests based on type
    case $TEST_TYPE in
        rds)
            test_rds_restore
            ;;
        velero)
            test_velero_restore
            ;;
        full)
            test_full_restore
            ;;
        *)
            log_error "Unknown test type: $TEST_TYPE"
            exit 1
            ;;
    esac

    # Generate report
    generate_report

    # Print summary
    echo ""
    echo "=========================================="
    echo "RESTORE TEST SUMMARY"
    echo "=========================================="
    echo "Test Type:    $TEST_TYPE"
    echo "Total Tests:  $TOTAL_TESTS"
    echo "Passed:       $PASSED_TESTS"
    echo "Failed:       $FAILED_TESTS"
    echo ""

    if [ $FAILED_TESTS -eq 0 ]; then
        echo "STATUS: ALL TESTS PASSED"
        exit 0
    else
        echo "STATUS: SOME TESTS FAILED"
        exit 1
    fi
}

main "$@"
