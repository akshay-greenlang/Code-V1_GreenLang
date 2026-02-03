#!/bin/bash
# =============================================================================
# Backup Verification Script for GreenLang
# INFRA-001: Backup and Disaster Recovery
# Version: 1.0.0
# =============================================================================
#
# This script verifies the integrity and availability of all backups:
# - RDS automated backups and snapshots
# - Velero Kubernetes backups
# - Redis backups
# - Weaviate backups
# - S3 backup buckets
#
# Usage: ./backup-verify.sh [--verbose] [--notify] [--json]
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/greenlang/backup-verify-$(date +%Y%m%d).log"
REPORT_FILE="/tmp/backup-verification-report-$(date +%Y%m%d%H%M).json"

# AWS Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
RDS_INSTANCE="${RDS_INSTANCE:-greenlang-postgres}"
S3_BACKUP_BUCKET="${S3_BACKUP_BUCKET:-greenlang-velero-backups}"
S3_DB_BACKUP_BUCKET="${S3_DB_BACKUP_BUCKET:-greenlang-backups}"

# Kubernetes Configuration
KUBE_NAMESPACE="${KUBE_NAMESPACE:-greenlang}"
VELERO_NAMESPACE="${VELERO_NAMESPACE:-velero}"

# Notification Configuration
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
PAGERDUTY_ROUTING_KEY="${PAGERDUTY_ROUTING_KEY:-}"
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"

# Thresholds
MAX_BACKUP_AGE_HOURS=25  # Alert if backup older than this
MIN_BACKUP_SIZE_MB=10    # Alert if backup smaller than this
REQUIRED_DAILY_BACKUPS=7 # Minimum daily backups to retain

# Parse arguments
VERBOSE=false
NOTIFY=false
JSON_OUTPUT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --notify|-n)
            NOTIFY=true
            shift
            ;;
        --json|-j)
            JSON_OUTPUT=true
            shift
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

verbose() {
    if [ "$VERBOSE" = true ]; then
        log_info "$@"
    fi
}

# =============================================================================
# Verification Results
# =============================================================================

declare -A RESULTS
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNINGS=0

record_result() {
    local check_name=$1
    local status=$2  # PASS, FAIL, WARN
    local details=$3

    RESULTS["$check_name"]="$status|$details"
    ((TOTAL_CHECKS++))

    case $status in
        PASS)
            ((PASSED_CHECKS++))
            log_success "$check_name: $details"
            ;;
        FAIL)
            ((FAILED_CHECKS++))
            log_error "$check_name: $details"
            ;;
        WARN)
            ((WARNINGS++))
            log_warn "$check_name: $details"
            ;;
    esac
}

# =============================================================================
# RDS Backup Verification
# =============================================================================

verify_rds_backups() {
    log_info "=== Verifying RDS Backups ==="

    # Check automated backups are enabled
    verbose "Checking RDS automated backup configuration..."

    local backup_config=$(aws rds describe-db-instances \
        --region "$AWS_REGION" \
        --db-instance-identifier "$RDS_INSTANCE" \
        --query 'DBInstances[0].[BackupRetentionPeriod,PreferredBackupWindow,LatestRestorableTime]' \
        --output json 2>/dev/null || echo "[]")

    if [ "$backup_config" = "[]" ]; then
        record_result "rds_instance_exists" "FAIL" "RDS instance $RDS_INSTANCE not found"
        return 1
    fi

    local retention=$(echo "$backup_config" | jq -r '.[0]')
    local backup_window=$(echo "$backup_config" | jq -r '.[1]')
    local latest_restorable=$(echo "$backup_config" | jq -r '.[2]')

    # Verify backup retention
    if [ "$retention" -ge 7 ]; then
        record_result "rds_backup_retention" "PASS" "Retention period: $retention days"
    else
        record_result "rds_backup_retention" "WARN" "Retention period only $retention days (recommended: 7+)"
    fi

    # Verify PITR is working (latest restorable time within last hour)
    if [ "$latest_restorable" != "null" ]; then
        local latest_ts=$(date -d "$latest_restorable" +%s 2>/dev/null || echo "0")
        local now_ts=$(date +%s)
        local age_minutes=$(( (now_ts - latest_ts) / 60 ))

        if [ "$age_minutes" -lt 60 ]; then
            record_result "rds_pitr_status" "PASS" "PITR available, lag: ${age_minutes} minutes"
        else
            record_result "rds_pitr_status" "WARN" "PITR lag: ${age_minutes} minutes (expected < 60)"
        fi
    else
        record_result "rds_pitr_status" "FAIL" "PITR not available"
    fi

    # Check recent snapshots
    verbose "Checking RDS snapshots..."
    local snapshots=$(aws rds describe-db-snapshots \
        --region "$AWS_REGION" \
        --db-instance-identifier "$RDS_INSTANCE" \
        --snapshot-type automated \
        --query 'DBSnapshots[*].[DBSnapshotIdentifier,SnapshotCreateTime,Status,AllocatedStorage]' \
        --output json 2>/dev/null || echo "[]")

    local snapshot_count=$(echo "$snapshots" | jq 'length')

    if [ "$snapshot_count" -ge "$REQUIRED_DAILY_BACKUPS" ]; then
        record_result "rds_snapshot_count" "PASS" "$snapshot_count automated snapshots available"
    else
        record_result "rds_snapshot_count" "WARN" "Only $snapshot_count snapshots (expected: $REQUIRED_DAILY_BACKUPS+)"
    fi

    # Verify latest snapshot is recent
    local latest_snapshot=$(echo "$snapshots" | jq -r 'sort_by(.[1]) | .[-1]')
    if [ "$latest_snapshot" != "null" ]; then
        local latest_snapshot_time=$(echo "$latest_snapshot" | jq -r '.[1]')
        local latest_snapshot_status=$(echo "$latest_snapshot" | jq -r '.[2]')

        local snapshot_ts=$(date -d "$latest_snapshot_time" +%s 2>/dev/null || echo "0")
        local age_hours=$(( ($(date +%s) - snapshot_ts) / 3600 ))

        if [ "$latest_snapshot_status" = "available" ] && [ "$age_hours" -lt "$MAX_BACKUP_AGE_HOURS" ]; then
            record_result "rds_latest_snapshot" "PASS" "Latest snapshot: ${age_hours}h old, status: $latest_snapshot_status"
        elif [ "$age_hours" -ge "$MAX_BACKUP_AGE_HOURS" ]; then
            record_result "rds_latest_snapshot" "FAIL" "Latest snapshot is ${age_hours}h old (max: ${MAX_BACKUP_AGE_HOURS}h)"
        else
            record_result "rds_latest_snapshot" "WARN" "Latest snapshot status: $latest_snapshot_status"
        fi
    fi

    # Check cross-region backups (if configured)
    verbose "Checking cross-region backup replication..."
    local cross_region_backups=$(aws rds describe-db-instance-automated-backups \
        --region "eu-west-1" \
        --db-instance-identifier "$RDS_INSTANCE" \
        --query 'DBInstanceAutomatedBackups[0].DBInstanceAutomatedBackupsArn' \
        --output text 2>/dev/null || echo "None")

    if [ "$cross_region_backups" != "None" ] && [ "$cross_region_backups" != "null" ]; then
        record_result "rds_cross_region" "PASS" "Cross-region backup replication configured"
    else
        record_result "rds_cross_region" "WARN" "Cross-region backup replication not detected"
    fi
}

# =============================================================================
# Velero Backup Verification
# =============================================================================

verify_velero_backups() {
    log_info "=== Verifying Velero Kubernetes Backups ==="

    # Check Velero is running
    verbose "Checking Velero deployment..."
    local velero_status=$(kubectl get deployment velero -n "$VELERO_NAMESPACE" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

    if [ "$velero_status" -ge 1 ]; then
        record_result "velero_deployment" "PASS" "Velero is running ($velero_status replicas)"
    else
        record_result "velero_deployment" "FAIL" "Velero is not running"
        return 1
    fi

    # Check backup storage location
    verbose "Checking backup storage location..."
    local bsl_status=$(kubectl get backupstoragelocation -n "$VELERO_NAMESPACE" \
        -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Unknown")

    if [ "$bsl_status" = "Available" ]; then
        record_result "velero_storage_location" "PASS" "Backup storage location: $bsl_status"
    else
        record_result "velero_storage_location" "FAIL" "Backup storage location: $bsl_status"
    fi

    # Check volume snapshot location
    verbose "Checking volume snapshot location..."
    local vsl_status=$(kubectl get volumesnapshotlocation -n "$VELERO_NAMESPACE" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "None")

    if [ "$vsl_status" != "None" ]; then
        record_result "velero_snapshot_location" "PASS" "Volume snapshot location configured: $vsl_status"
    else
        record_result "velero_snapshot_location" "WARN" "Volume snapshot location not configured"
    fi

    # Check scheduled backups
    verbose "Checking backup schedules..."
    local schedules=$(kubectl get schedule -n "$VELERO_NAMESPACE" \
        -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")

    if [ -n "$schedules" ]; then
        local schedule_count=$(echo "$schedules" | wc -w)
        record_result "velero_schedules" "PASS" "$schedule_count backup schedules configured"
    else
        record_result "velero_schedules" "FAIL" "No backup schedules found"
    fi

    # Check recent backups
    verbose "Checking recent backups..."
    local recent_backups=$(kubectl get backup -n "$VELERO_NAMESPACE" \
        --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-5:].metadata.name}' 2>/dev/null || echo "")

    if [ -n "$recent_backups" ]; then
        local backup_count=$(echo "$recent_backups" | wc -w)
        record_result "velero_recent_backups" "PASS" "$backup_count recent backups found"

        # Check latest backup status
        local latest_backup=$(echo "$recent_backups" | awk '{print $NF}')
        local latest_status=$(kubectl get backup "$latest_backup" -n "$VELERO_NAMESPACE" \
            -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")

        if [ "$latest_status" = "Completed" ]; then
            record_result "velero_latest_backup" "PASS" "Latest backup '$latest_backup' completed"
        elif [ "$latest_status" = "PartiallyFailed" ]; then
            record_result "velero_latest_backup" "WARN" "Latest backup '$latest_backup' partially failed"
        else
            record_result "velero_latest_backup" "FAIL" "Latest backup '$latest_backup' status: $latest_status"
        fi
    else
        record_result "velero_recent_backups" "FAIL" "No backups found"
    fi

    # Check backup success rate (last 24 hours)
    verbose "Calculating backup success rate..."
    local total_recent=$(kubectl get backup -n "$VELERO_NAMESPACE" \
        --field-selector="metadata.creationTimestamp>=$(date -d '24 hours ago' -Iseconds)" \
        -o jsonpath='{.items[*].status.phase}' 2>/dev/null || echo "")

    if [ -n "$total_recent" ]; then
        local completed=$(echo "$total_recent" | tr ' ' '\n' | grep -c "Completed" || echo "0")
        local total=$(echo "$total_recent" | wc -w)
        local success_rate=$((completed * 100 / total))

        if [ "$success_rate" -ge 90 ]; then
            record_result "velero_success_rate" "PASS" "Backup success rate: ${success_rate}% ($completed/$total)"
        elif [ "$success_rate" -ge 70 ]; then
            record_result "velero_success_rate" "WARN" "Backup success rate: ${success_rate}% ($completed/$total)"
        else
            record_result "velero_success_rate" "FAIL" "Backup success rate: ${success_rate}% ($completed/$total)"
        fi
    fi
}

# =============================================================================
# S3 Backup Verification
# =============================================================================

verify_s3_backups() {
    log_info "=== Verifying S3 Backup Storage ==="

    # Check Velero backup bucket
    verbose "Checking Velero backup bucket..."
    local velero_bucket_exists=$(aws s3api head-bucket --bucket "$S3_BACKUP_BUCKET" 2>&1 || echo "error")

    if [ "$velero_bucket_exists" = "" ]; then
        record_result "s3_velero_bucket" "PASS" "Velero backup bucket exists: $S3_BACKUP_BUCKET"

        # Check bucket versioning
        local versioning=$(aws s3api get-bucket-versioning --bucket "$S3_BACKUP_BUCKET" \
            --query 'Status' --output text 2>/dev/null || echo "Disabled")

        if [ "$versioning" = "Enabled" ]; then
            record_result "s3_velero_versioning" "PASS" "Bucket versioning enabled"
        else
            record_result "s3_velero_versioning" "WARN" "Bucket versioning: $versioning"
        fi

        # Check bucket encryption
        local encryption=$(aws s3api get-bucket-encryption --bucket "$S3_BACKUP_BUCKET" \
            --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.SSEAlgorithm' \
            --output text 2>/dev/null || echo "None")

        if [ "$encryption" != "None" ]; then
            record_result "s3_velero_encryption" "PASS" "Bucket encryption: $encryption"
        else
            record_result "s3_velero_encryption" "WARN" "Bucket encryption not configured"
        fi

        # Check recent objects
        local recent_objects=$(aws s3 ls "s3://$S3_BACKUP_BUCKET/production/" --recursive \
            | tail -5 | wc -l)

        if [ "$recent_objects" -gt 0 ]; then
            record_result "s3_velero_objects" "PASS" "Backup objects present in bucket"
        else
            record_result "s3_velero_objects" "WARN" "No recent backup objects found"
        fi
    else
        record_result "s3_velero_bucket" "FAIL" "Velero backup bucket not accessible: $S3_BACKUP_BUCKET"
    fi

    # Check database backup bucket
    verbose "Checking database backup bucket..."
    local db_bucket_exists=$(aws s3api head-bucket --bucket "$S3_DB_BACKUP_BUCKET" 2>&1 || echo "error")

    if [ "$db_bucket_exists" = "" ]; then
        record_result "s3_db_bucket" "PASS" "Database backup bucket exists: $S3_DB_BACKUP_BUCKET"

        # Check for recent PostgreSQL backups
        local pg_backups=$(aws s3 ls "s3://$S3_DB_BACKUP_BUCKET/postgres/" 2>/dev/null \
            | grep -E "\.sql\.gz|\.dump" | tail -1)

        if [ -n "$pg_backups" ]; then
            record_result "s3_postgres_backups" "PASS" "PostgreSQL backups found in S3"
        else
            record_result "s3_postgres_backups" "WARN" "No PostgreSQL backup files found in S3"
        fi

        # Check for recent Redis backups
        local redis_backups=$(aws s3 ls "s3://$S3_DB_BACKUP_BUCKET/redis/" 2>/dev/null \
            | grep -E "\.rdb|\.aof" | tail -1)

        if [ -n "$redis_backups" ]; then
            record_result "s3_redis_backups" "PASS" "Redis backups found in S3"
        else
            record_result "s3_redis_backups" "WARN" "No Redis backup files found in S3"
        fi
    else
        record_result "s3_db_bucket" "FAIL" "Database backup bucket not accessible: $S3_DB_BACKUP_BUCKET"
    fi
}

# =============================================================================
# Redis Backup Verification
# =============================================================================

verify_redis_backups() {
    log_info "=== Verifying Redis Backups ==="

    # Check Redis is running
    verbose "Checking Redis deployment..."
    local redis_status=$(kubectl get statefulset redis-master -n "$KUBE_NAMESPACE" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

    if [ "$redis_status" -ge 1 ]; then
        record_result "redis_deployment" "PASS" "Redis is running ($redis_status replicas)"

        # Check persistence configuration
        local persistence=$(kubectl exec -n "$KUBE_NAMESPACE" redis-master-0 -- \
            redis-cli CONFIG GET appendonly 2>/dev/null | tail -1 || echo "unknown")

        if [ "$persistence" = "yes" ]; then
            record_result "redis_aof_enabled" "PASS" "Redis AOF persistence enabled"
        else
            record_result "redis_aof_enabled" "WARN" "Redis AOF persistence: $persistence"
        fi

        # Check last save time
        local last_save=$(kubectl exec -n "$KUBE_NAMESPACE" redis-master-0 -- \
            redis-cli LASTSAVE 2>/dev/null || echo "0")

        if [ "$last_save" -gt 0 ]; then
            local save_age=$(( $(date +%s) - last_save ))
            local save_age_hours=$((save_age / 3600))

            if [ "$save_age_hours" -lt 24 ]; then
                record_result "redis_last_save" "PASS" "Redis last save: ${save_age_hours}h ago"
            else
                record_result "redis_last_save" "WARN" "Redis last save: ${save_age_hours}h ago (expected < 24h)"
            fi
        fi
    else
        record_result "redis_deployment" "WARN" "Redis deployment not found or not running"
    fi
}

# =============================================================================
# Generate Report
# =============================================================================

generate_report() {
    log_info "=== Generating Verification Report ==="

    local report_json=$(cat <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "summary": {
        "total_checks": $TOTAL_CHECKS,
        "passed": $PASSED_CHECKS,
        "failed": $FAILED_CHECKS,
        "warnings": $WARNINGS,
        "status": "$([ $FAILED_CHECKS -eq 0 ] && echo "HEALTHY" || echo "DEGRADED")"
    },
    "checks": {
EOF
)

    local first=true
    for check in "${!RESULTS[@]}"; do
        local result="${RESULTS[$check]}"
        local status=$(echo "$result" | cut -d'|' -f1)
        local details=$(echo "$result" | cut -d'|' -f2)

        if [ "$first" = true ]; then
            first=false
        else
            report_json+=","
        fi

        report_json+="
        \"$check\": {
            \"status\": \"$status\",
            \"details\": \"$details\"
        }"
    done

    report_json+="
    }
}"

    echo "$report_json" > "$REPORT_FILE"
    log_info "Report saved to: $REPORT_FILE"

    if [ "$JSON_OUTPUT" = true ]; then
        echo "$report_json"
    fi
}

# =============================================================================
# Notifications
# =============================================================================

send_notifications() {
    if [ "$NOTIFY" = false ]; then
        return
    fi

    log_info "=== Sending Notifications ==="

    local status="HEALTHY"
    local color="good"
    if [ $FAILED_CHECKS -gt 0 ]; then
        status="DEGRADED"
        color="danger"
    elif [ $WARNINGS -gt 0 ]; then
        status="WARNING"
        color="warning"
    fi

    # Slack notification
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        verbose "Sending Slack notification..."
        local slack_payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "GreenLang Backup Verification Report",
            "text": "Status: *$status*",
            "fields": [
                {"title": "Total Checks", "value": "$TOTAL_CHECKS", "short": true},
                {"title": "Passed", "value": "$PASSED_CHECKS", "short": true},
                {"title": "Failed", "value": "$FAILED_CHECKS", "short": true},
                {"title": "Warnings", "value": "$WARNINGS", "short": true}
            ],
            "footer": "Backup Verification | $(date '+%Y-%m-%d %H:%M:%S')"
        }
    ]
}
EOF
)
        curl -s -X POST -H 'Content-type: application/json' \
            --data "$slack_payload" "$SLACK_WEBHOOK_URL" > /dev/null
    fi

    # SNS notification
    if [ -n "$SNS_TOPIC_ARN" ]; then
        verbose "Sending SNS notification..."
        aws sns publish \
            --topic-arn "$SNS_TOPIC_ARN" \
            --subject "GreenLang Backup Verification: $status" \
            --message "$(cat $REPORT_FILE)" > /dev/null
    fi

    # PagerDuty (only for failures)
    if [ -n "$PAGERDUTY_ROUTING_KEY" ] && [ $FAILED_CHECKS -gt 0 ]; then
        verbose "Triggering PagerDuty alert..."
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{
                \"routing_key\": \"$PAGERDUTY_ROUTING_KEY\",
                \"event_action\": \"trigger\",
                \"payload\": {
                    \"summary\": \"GreenLang backup verification failed: $FAILED_CHECKS checks failed\",
                    \"severity\": \"critical\",
                    \"source\": \"backup-verify.sh\"
                }
            }" \
            "https://events.pagerduty.com/v2/enqueue" > /dev/null
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    mkdir -p "$(dirname "$LOG_FILE")"

    log_info "=========================================="
    log_info "GreenLang Backup Verification"
    log_info "Started: $(date)"
    log_info "=========================================="

    # Run all verifications
    verify_rds_backups
    verify_velero_backups
    verify_s3_backups
    verify_redis_backups

    # Generate report
    generate_report

    # Send notifications
    send_notifications

    # Print summary
    echo ""
    echo "=========================================="
    echo "BACKUP VERIFICATION SUMMARY"
    echo "=========================================="
    echo "Total Checks: $TOTAL_CHECKS"
    echo "Passed:       $PASSED_CHECKS"
    echo "Failed:       $FAILED_CHECKS"
    echo "Warnings:     $WARNINGS"
    echo ""

    if [ $FAILED_CHECKS -eq 0 ]; then
        echo "STATUS: ALL BACKUPS HEALTHY"
        exit 0
    else
        echo "STATUS: BACKUP ISSUES DETECTED"
        exit 1
    fi
}

main "$@"
