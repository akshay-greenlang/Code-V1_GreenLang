#!/usr/bin/env bash
# =============================================================================
# GreenLang PostgreSQL Backup and Restore Helper Script
# =============================================================================
# This script provides convenient commands for managing PostgreSQL backups
# and disaster recovery operations for the GreenLang platform.
#
# Usage:
#   ./backup-restore.sh <command> [options]
#
# Commands:
#   backup              - Trigger an immediate backup
#   restore             - Restore from a backup
#   list                - List available backups
#   status              - Check backup CronJob status
#   logs                - View backup/restore job logs
#   verify              - Verify a backup file
#   download            - Download a backup locally
#   cleanup             - Clean up old backups manually
#
# Examples:
#   ./backup-restore.sh backup
#   ./backup-restore.sh restore --latest
#   ./backup-restore.sh restore --timestamp 20241209_020000
#   ./backup-restore.sh list --days 7
#   ./backup-restore.sh download --file greenlang_greenlang_20241209_020000.sql.gz
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NAMESPACE="${GREENLANG_NAMESPACE:-greenlang-production}"
CRONJOB_NAME="${GREENLANG_BACKUP_CRONJOB:-postgres-backup}"
RESTORE_JOB_NAME="${GREENLANG_RESTORE_JOB:-postgres-restore}"
S3_BUCKET="${GREENLANG_BACKUP_BUCKET:-greenlang-backups-production}"
S3_PREFIX="${GREENLANG_BACKUP_PREFIX:-postgres/daily}"
S3_REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    local missing=()

    if ! command -v kubectl &> /dev/null; then
        missing+=("kubectl")
    fi

    if ! command -v aws &> /dev/null; then
        missing+=("aws-cli")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        echo "Please install the missing tools and try again."
        exit 1
    fi

    # Verify kubectl context
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi

    # Verify namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace '$NAMESPACE' does not exist."
        exit 1
    fi
}

print_header() {
    echo ""
    echo "=============================================="
    echo "  GreenLang PostgreSQL Backup & Restore"
    echo "=============================================="
    echo "  Namespace: $NAMESPACE"
    echo "  S3 Bucket: $S3_BUCKET"
    echo "=============================================="
    echo ""
}

# -----------------------------------------------------------------------------
# Backup Commands
# -----------------------------------------------------------------------------

cmd_backup() {
    log_info "Triggering immediate backup..."

    # Create a one-time job from the CronJob
    JOB_NAME="${CRONJOB_NAME}-manual-$(date +%Y%m%d%H%M%S)"

    kubectl create job "$JOB_NAME" \
        --from="cronjob/${CRONJOB_NAME}" \
        -n "$NAMESPACE"

    log_success "Backup job '$JOB_NAME' created."
    log_info "Waiting for job to start..."

    # Wait for the job to start
    kubectl wait --for=condition=Ready pod \
        -l job-name="$JOB_NAME" \
        -n "$NAMESPACE" \
        --timeout=120s 2>/dev/null || true

    log_info "Streaming logs from backup job..."
    kubectl logs -f "job/$JOB_NAME" -n "$NAMESPACE" --all-containers || true

    # Check job status
    JOB_STATUS=$(kubectl get job "$JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}')

    if [ "$JOB_STATUS" == "1" ]; then
        log_success "Backup completed successfully!"
    else
        log_error "Backup may have failed. Check logs for details."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Restore Commands
# -----------------------------------------------------------------------------

cmd_restore() {
    local RESTORE_MODE="latest"
    local RESTORE_TIMESTAMP=""
    local RESTORE_FILE=""
    local CONFIRM="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --latest)
                RESTORE_MODE="latest"
                shift
                ;;
            --timestamp)
                RESTORE_MODE="timestamp"
                RESTORE_TIMESTAMP="$2"
                shift 2
                ;;
            --file)
                RESTORE_MODE="specific_file"
                RESTORE_FILE="$2"
                shift 2
                ;;
            --confirm)
                CONFIRM="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                cmd_help
                exit 1
                ;;
        esac
    done

    # Safety confirmation
    if [ "$CONFIRM" != "true" ]; then
        echo ""
        log_warning "=============================================="
        log_warning "  WARNING: DATABASE RESTORE OPERATION"
        log_warning "=============================================="
        log_warning "This will:"
        log_warning "  1. Terminate all active database connections"
        log_warning "  2. DROP the existing database"
        log_warning "  3. Restore from backup"
        log_warning ""
        log_warning "Restore mode: $RESTORE_MODE"
        [ -n "$RESTORE_TIMESTAMP" ] && log_warning "Target timestamp: $RESTORE_TIMESTAMP"
        [ -n "$RESTORE_FILE" ] && log_warning "Target file: $RESTORE_FILE"
        log_warning "=============================================="
        echo ""
        read -p "Type 'yes' to confirm restore: " USER_CONFIRM
        if [ "$USER_CONFIRM" != "yes" ]; then
            log_info "Restore cancelled."
            exit 0
        fi
    fi

    log_info "Updating restore configuration..."

    # Update restore ConfigMap
    kubectl patch configmap postgres-restore-config \
        -n "$NAMESPACE" \
        --type merge \
        -p "{\"data\":{\"RESTORE_MODE\":\"$RESTORE_MODE\",\"RESTORE_TIMESTAMP\":\"$RESTORE_TIMESTAMP\",\"RESTORE_FILE\":\"$RESTORE_FILE\",\"CONFIRM_RESTORE\":\"true\"}}"

    log_info "Starting restore job..."

    # Delete existing restore job if exists
    kubectl delete job "$RESTORE_JOB_NAME" -n "$NAMESPACE" --ignore-not-found=true

    # Apply restore job
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    kubectl apply -f "$SCRIPT_DIR/../k8s/backup/postgres-restore-job.yaml"

    log_info "Waiting for restore job to start..."

    # Wait for the job to start
    sleep 5
    kubectl wait --for=condition=Ready pod \
        -l job-name="$RESTORE_JOB_NAME" \
        -n "$NAMESPACE" \
        --timeout=300s 2>/dev/null || true

    log_info "Streaming logs from restore job..."
    kubectl logs -f "job/$RESTORE_JOB_NAME" -n "$NAMESPACE" --all-containers || true

    # Check job status
    JOB_STATUS=$(kubectl get job "$RESTORE_JOB_NAME" -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' 2>/dev/null || echo "0")

    # Reset CONFIRM_RESTORE to prevent accidental re-runs
    kubectl patch configmap postgres-restore-config \
        -n "$NAMESPACE" \
        --type merge \
        -p '{"data":{"CONFIRM_RESTORE":"false"}}'

    if [ "$JOB_STATUS" == "1" ]; then
        log_success "Restore completed successfully!"
    else
        log_error "Restore may have failed. Check logs for details."
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# List Backups
# -----------------------------------------------------------------------------

cmd_list() {
    local DAYS=${1:-30}

    log_info "Listing backups from the last $DAYS days..."
    echo ""

    aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        --region "$S3_REGION" \
        | grep "greenlang_" \
        | sort -r \
        | head -n 50 \
        | while read -r line; do
            DATE=$(echo "$line" | awk '{print $1}')
            TIME=$(echo "$line" | awk '{print $2}')
            SIZE=$(echo "$line" | awk '{print $3}')
            FILE=$(echo "$line" | awk '{print $4}')

            # Convert size to human readable
            if [ "$SIZE" -gt 1073741824 ]; then
                SIZE_HR="$(echo "scale=2; $SIZE/1073741824" | bc) GB"
            elif [ "$SIZE" -gt 1048576 ]; then
                SIZE_HR="$(echo "scale=2; $SIZE/1048576" | bc) MB"
            elif [ "$SIZE" -gt 1024 ]; then
                SIZE_HR="$(echo "scale=2; $SIZE/1024" | bc) KB"
            else
                SIZE_HR="$SIZE B"
            fi

            printf "%-12s %-10s %12s   %s\n" "$DATE" "$TIME" "$SIZE_HR" "$FILE"
        done

    echo ""
    log_info "Total backup count:"
    aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        --region "$S3_REGION" \
        | grep -c "greenlang_" || echo "0"
}

# -----------------------------------------------------------------------------
# Status Command
# -----------------------------------------------------------------------------

cmd_status() {
    log_info "Checking backup CronJob status..."
    echo ""

    echo "=== CronJob Status ==="
    kubectl get cronjob "$CRONJOB_NAME" -n "$NAMESPACE" -o wide 2>/dev/null || \
        log_warning "CronJob '$CRONJOB_NAME' not found"

    echo ""
    echo "=== Recent Jobs ==="
    kubectl get jobs -n "$NAMESPACE" \
        -l app.kubernetes.io/component=backup \
        --sort-by='.metadata.creationTimestamp' \
        | tail -n 10

    echo ""
    echo "=== Recent Pods ==="
    kubectl get pods -n "$NAMESPACE" \
        -l app.kubernetes.io/component=backup \
        --sort-by='.metadata.creationTimestamp' \
        | tail -n 10

    echo ""
    echo "=== Last Successful Backup ==="
    LAST_SUCCESS=$(kubectl get cronjob "$CRONJOB_NAME" -n "$NAMESPACE" \
        -o jsonpath='{.status.lastSuccessfulTime}' 2>/dev/null || echo "N/A")
    echo "Last successful backup: $LAST_SUCCESS"

    echo ""
    echo "=== S3 Bucket Status ==="
    LATEST_BACKUP=$(aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        --region "$S3_REGION" \
        | grep "greenlang_" \
        | sort -r \
        | head -n 1)
    echo "Latest backup in S3: $LATEST_BACKUP"

    BACKUP_COUNT=$(aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        --region "$S3_REGION" \
        | grep -c "greenlang_" 2>/dev/null || echo "0")
    echo "Total backups: $BACKUP_COUNT"
}

# -----------------------------------------------------------------------------
# Logs Command
# -----------------------------------------------------------------------------

cmd_logs() {
    local JOB_TYPE=${1:-backup}
    local FOLLOW=${2:-false}

    if [ "$JOB_TYPE" == "backup" ]; then
        LABEL="app.kubernetes.io/component=backup"
    else
        LABEL="app.kubernetes.io/component=restore"
    fi

    log_info "Fetching logs for $JOB_TYPE jobs..."

    # Get the most recent pod
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" \
        -l "$LABEL" \
        --sort-by='.metadata.creationTimestamp' \
        -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo "")

    if [ -z "$POD_NAME" ]; then
        log_warning "No $JOB_TYPE pods found."
        return
    fi

    if [ "$FOLLOW" == "-f" ]; then
        kubectl logs -f "$POD_NAME" -n "$NAMESPACE" --all-containers
    else
        kubectl logs "$POD_NAME" -n "$NAMESPACE" --all-containers
    fi
}

# -----------------------------------------------------------------------------
# Verify Backup
# -----------------------------------------------------------------------------

cmd_verify() {
    local BACKUP_FILE=${1:-latest.sql.gz}

    log_info "Verifying backup: $BACKUP_FILE"

    # Download to temp location
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    log_info "Downloading backup..."
    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_FILE}" "$TEMP_DIR/$BACKUP_FILE" \
        --region "$S3_REGION"

    # Check if checksum file exists
    if aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_FILE}.sha256" --region "$S3_REGION" &>/dev/null; then
        log_info "Downloading checksum..."
        aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_FILE}.sha256" "$TEMP_DIR/${BACKUP_FILE}.sha256" \
            --region "$S3_REGION"

        log_info "Verifying checksum..."
        EXPECTED=$(cat "$TEMP_DIR/${BACKUP_FILE}.sha256" | cut -d' ' -f1)
        ACTUAL=$(sha256sum "$TEMP_DIR/$BACKUP_FILE" | cut -d' ' -f1)

        if [ "$EXPECTED" == "$ACTUAL" ]; then
            log_success "Checksum verified: $ACTUAL"
        else
            log_error "Checksum mismatch!"
            log_error "Expected: $EXPECTED"
            log_error "Actual: $ACTUAL"
            exit 1
        fi
    else
        log_warning "No checksum file found. Generating checksum..."
        sha256sum "$TEMP_DIR/$BACKUP_FILE"
    fi

    # Verify gzip integrity
    log_info "Verifying gzip integrity..."
    if gzip -t "$TEMP_DIR/$BACKUP_FILE" 2>/dev/null; then
        log_success "Gzip integrity verified."
    else
        log_error "Gzip integrity check failed!"
        exit 1
    fi

    # Check SQL structure
    log_info "Checking SQL structure..."
    TABLES=$(zcat "$TEMP_DIR/$BACKUP_FILE" | grep -c "CREATE TABLE" || echo "0")
    INSERTS=$(zcat "$TEMP_DIR/$BACKUP_FILE" | grep -c "INSERT INTO\|COPY" || echo "0")

    echo ""
    echo "=== Backup Summary ==="
    echo "File: $BACKUP_FILE"
    echo "Size: $(ls -lh "$TEMP_DIR/$BACKUP_FILE" | awk '{print $5}')"
    echo "CREATE TABLE statements: $TABLES"
    echo "INSERT/COPY statements: $INSERTS"
    echo ""

    log_success "Backup verification complete!"
}

# -----------------------------------------------------------------------------
# Download Backup
# -----------------------------------------------------------------------------

cmd_download() {
    local BACKUP_FILE=${1:-latest.sql.gz}
    local OUTPUT_DIR=${2:-.}

    log_info "Downloading backup: $BACKUP_FILE to $OUTPUT_DIR"

    aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_FILE}" "$OUTPUT_DIR/$BACKUP_FILE" \
        --region "$S3_REGION"

    # Also download checksum if available
    if aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_FILE}.sha256" --region "$S3_REGION" &>/dev/null; then
        aws s3 cp "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_FILE}.sha256" "$OUTPUT_DIR/${BACKUP_FILE}.sha256" \
            --region "$S3_REGION"
    fi

    log_success "Download complete: $OUTPUT_DIR/$BACKUP_FILE"
}

# -----------------------------------------------------------------------------
# Cleanup Old Backups
# -----------------------------------------------------------------------------

cmd_cleanup() {
    local RETENTION_DAYS=${1:-30}
    local DRY_RUN=${2:-true}

    log_info "Cleaning up backups older than $RETENTION_DAYS days..."

    CUTOFF_DATE=$(date -d "-${RETENTION_DAYS} days" +%Y-%m-%d 2>/dev/null || \
        date -v-${RETENTION_DAYS}d +%Y-%m-%d)

    log_info "Cutoff date: $CUTOFF_DATE"
    echo ""

    aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" \
        --region "$S3_REGION" \
        | while read -r line; do
            FILE_DATE=$(echo "$line" | awk '{print $1}')
            FILE_NAME=$(echo "$line" | awk '{print $4}')

            # Skip if not a backup file or is latest
            if [[ ! "$FILE_NAME" =~ ^greenlang_.*\.sql\.gz$ ]] || [ "$FILE_NAME" == "latest.sql.gz" ]; then
                continue
            fi

            if [[ "$FILE_DATE" < "$CUTOFF_DATE" ]]; then
                if [ "$DRY_RUN" == "true" ]; then
                    echo "[DRY-RUN] Would delete: $FILE_NAME (dated $FILE_DATE)"
                else
                    log_info "Deleting: $FILE_NAME"
                    aws s3 rm "s3://${S3_BUCKET}/${S3_PREFIX}/${FILE_NAME}" --region "$S3_REGION"
                    aws s3 rm "s3://${S3_BUCKET}/${S3_PREFIX}/${FILE_NAME}.sha256" --region "$S3_REGION" 2>/dev/null || true
                fi
            fi
        done

    if [ "$DRY_RUN" == "true" ]; then
        echo ""
        log_info "This was a dry run. Use '--execute' to actually delete files."
    else
        log_success "Cleanup complete!"
    fi
}

# -----------------------------------------------------------------------------
# Help Command
# -----------------------------------------------------------------------------

cmd_help() {
    cat << 'EOF'

GreenLang PostgreSQL Backup & Restore Helper

Usage: ./backup-restore.sh <command> [options]

Commands:
  backup                    Trigger an immediate backup

  restore [options]         Restore from a backup
    --latest                Restore from the most recent backup (default)
    --timestamp <TS>        Restore to specific timestamp (YYYYMMDD_HHMMSS)
    --file <filename>       Restore from specific backup file
    --confirm               Skip confirmation prompt (use with caution!)

  list [days]               List available backups (default: last 30 days)

  status                    Check backup CronJob status

  logs [type] [-f]          View job logs
    backup                  View backup job logs (default)
    restore                 View restore job logs
    -f                      Follow/stream logs

  verify [filename]         Verify a backup file integrity
                           (default: latest.sql.gz)

  download <file> [dir]     Download a backup locally

  cleanup [days] [--execute] Clean up backups older than N days
    --execute               Actually delete files (default: dry-run)

Environment Variables:
  GREENLANG_NAMESPACE       Kubernetes namespace (default: greenlang-production)
  GREENLANG_BACKUP_BUCKET   S3 bucket name
  GREENLANG_BACKUP_PREFIX   S3 prefix/folder
  AWS_REGION                AWS region for S3

Examples:
  # Trigger immediate backup
  ./backup-restore.sh backup

  # Restore from latest backup
  ./backup-restore.sh restore --latest

  # Restore to specific point in time
  ./backup-restore.sh restore --timestamp 20241209_020000

  # List recent backups
  ./backup-restore.sh list 7

  # Verify backup integrity
  ./backup-restore.sh verify greenlang_greenlang_20241209_020000.sql.gz

  # Download backup locally
  ./backup-restore.sh download latest.sql.gz ./backups/

  # Clean up old backups (dry-run)
  ./backup-restore.sh cleanup 30

  # Clean up old backups (execute)
  ./backup-restore.sh cleanup 30 --execute

EOF
}

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

main() {
    COMMAND=${1:-help}
    shift || true

    case $COMMAND in
        backup)
            check_prerequisites
            print_header
            cmd_backup "$@"
            ;;
        restore)
            check_prerequisites
            print_header
            cmd_restore "$@"
            ;;
        list)
            check_prerequisites
            print_header
            cmd_list "$@"
            ;;
        status)
            check_prerequisites
            print_header
            cmd_status
            ;;
        logs)
            check_prerequisites
            cmd_logs "$@"
            ;;
        verify)
            check_prerequisites
            print_header
            cmd_verify "$@"
            ;;
        download)
            check_prerequisites
            cmd_download "$@"
            ;;
        cleanup)
            check_prerequisites
            print_header
            cmd_cleanup "$@"
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            cmd_help
            exit 1
            ;;
    esac
}

main "$@"
