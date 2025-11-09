#!/bin/bash
# ==============================================================================
# GL-VCCI Production Backup Script
# ==============================================================================
# Creates a full backup of production database and configurations
#
# Version: 2.0.0
# Date: 2025-11-09

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/vcci}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PREFIX="vcci-backup-${TIMESTAMP}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-vcci}"
DB_USER="${DB_USER:-vcci_admin}"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==============================================================================
# Backup Process
# ==============================================================================

echo "======================================================================"
echo "GL-VCCI Production Backup"
echo "======================================================================"
echo ""

log_info "Starting backup at $(date)"
log_info "Backup directory: $BACKUP_DIR"
log_info "Timestamp: $TIMESTAMP"
echo ""

# 1. Create backup directory
log_info "Creating backup directory..."
mkdir -p "$BACKUP_DIR"

CURRENT_BACKUP_DIR="$BACKUP_DIR/$BACKUP_PREFIX"
mkdir -p "$CURRENT_BACKUP_DIR"

log_info "Backup will be stored in: $CURRENT_BACKUP_DIR"
echo ""

# 2. Backup PostgreSQL database
log_info "Backing up PostgreSQL database..."
PGPASSWORD="$DB_PASSWORD" pg_dump \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    -F c \
    -b \
    -v \
    -f "$CURRENT_BACKUP_DIR/database.dump"

if [ $? -eq 0 ]; then
    log_info "✓ Database backup completed successfully"
    DB_SIZE=$(du -sh "$CURRENT_BACKUP_DIR/database.dump" | cut -f1)
    log_info "  Database backup size: $DB_SIZE"
else
    log_error "✗ Database backup failed"
    exit 1
fi
echo ""

# 3. Backup Kubernetes configurations
log_info "Backing up Kubernetes configurations..."
mkdir -p "$CURRENT_BACKUP_DIR/k8s"

# Export all resources in the namespace
NAMESPACE="${KUBERNETES_NAMESPACE:-vcci-production}"

log_info "Exporting deployments..."
kubectl get deployments -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/k8s/deployments.yaml"

log_info "Exporting services..."
kubectl get services -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/k8s/services.yaml"

log_info "Exporting configmaps..."
kubectl get configmaps -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/k8s/configmaps.yaml"

log_info "Exporting ingress..."
kubectl get ingress -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/k8s/ingress.yaml"

log_info "Exporting persistent volume claims..."
kubectl get pvc -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/k8s/pvc.yaml"

log_info "✓ Kubernetes configurations backed up"
echo ""

# 4. Backup Kubernetes secrets (encrypted)
log_info "Backing up Kubernetes secrets..."
mkdir -p "$CURRENT_BACKUP_DIR/secrets"

kubectl get secrets -n "$NAMESPACE" -o yaml > "$CURRENT_BACKUP_DIR/secrets/secrets.yaml"

# Encrypt secrets backup
if command -v gpg &> /dev/null; then
    log_info "Encrypting secrets backup..."
    gpg --symmetric --cipher-algo AES256 "$CURRENT_BACKUP_DIR/secrets/secrets.yaml"
    rm "$CURRENT_BACKUP_DIR/secrets/secrets.yaml"
    log_info "✓ Secrets encrypted with GPG"
else
    log_warn "GPG not available, secrets not encrypted"
fi
echo ""

# 5. Backup Redis data
log_info "Backing up Redis data..."
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

if command -v redis-cli &> /dev/null; then
    # Trigger Redis BGSAVE
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE

    # Wait for BGSAVE to complete
    while [ "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)" = "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)" ]; do
        sleep 1
    done

    # Copy RDB file
    REDIS_DATA_DIR="${REDIS_DATA_DIR:-/var/lib/redis}"
    if [ -f "$REDIS_DATA_DIR/dump.rdb" ]; then
        cp "$REDIS_DATA_DIR/dump.rdb" "$CURRENT_BACKUP_DIR/redis.rdb"
        log_info "✓ Redis data backed up"
    else
        log_warn "Redis RDB file not found"
    fi
else
    log_warn "redis-cli not available, skipping Redis backup"
fi
echo ""

# 6. Backup application configurations
log_info "Backing up application configurations..."
mkdir -p "$CURRENT_BACKUP_DIR/config"

# Copy configuration files
if [ -d "$PROJECT_ROOT/config" ]; then
    cp -r "$PROJECT_ROOT/config"/* "$CURRENT_BACKUP_DIR/config/"
    log_info "✓ Application configurations backed up"
else
    log_warn "Config directory not found"
fi
echo ""

# 7. Backup monitoring configurations
log_info "Backing up monitoring configurations..."
mkdir -p "$CURRENT_BACKUP_DIR/monitoring"

if [ -d "$PROJECT_ROOT/monitoring" ]; then
    cp -r "$PROJECT_ROOT/monitoring"/* "$CURRENT_BACKUP_DIR/monitoring/"
    log_info "✓ Monitoring configurations backed up"
else
    log_warn "Monitoring directory not found"
fi
echo ""

# 8. Create backup manifest
log_info "Creating backup manifest..."
cat > "$CURRENT_BACKUP_DIR/MANIFEST.txt" <<EOF
GL-VCCI Production Backup Manifest
==================================

Backup Timestamp: $TIMESTAMP
Backup Date: $(date)
Environment: ${ENVIRONMENT:-production}
Version: ${VERSION:-unknown}

Backup Contents:
- Database dump (PostgreSQL)
- Kubernetes configurations
- Kubernetes secrets (encrypted)
- Redis data
- Application configurations
- Monitoring configurations

Database:
- Host: $DB_HOST
- Database: $DB_NAME
- Size: $DB_SIZE

Kubernetes:
- Namespace: $NAMESPACE

Notes:
- Secrets are encrypted with GPG (AES256)
- Use restore_production.sh to restore from this backup
- Retention period: $RETENTION_DAYS days

Generated by: backup_production.sh v2.0.0
EOF

log_info "✓ Backup manifest created"
echo ""

# 9. Compress backup
log_info "Compressing backup..."
cd "$BACKUP_DIR"
tar -czf "${BACKUP_PREFIX}.tar.gz" "$BACKUP_PREFIX"

if [ $? -eq 0 ]; then
    COMPRESSED_SIZE=$(du -sh "${BACKUP_PREFIX}.tar.gz" | cut -f1)
    log_info "✓ Backup compressed successfully"
    log_info "  Compressed size: $COMPRESSED_SIZE"

    # Remove uncompressed backup directory
    rm -rf "$BACKUP_PREFIX"
else
    log_error "✗ Backup compression failed"
    exit 1
fi
echo ""

# 10. Upload to cloud storage (optional)
if [ -n "$S3_BUCKET" ]; then
    log_info "Uploading backup to S3..."

    if command -v aws &> /dev/null; then
        aws s3 cp "${BACKUP_PREFIX}.tar.gz" "s3://$S3_BUCKET/backups/${BACKUP_PREFIX}.tar.gz"

        if [ $? -eq 0 ]; then
            log_info "✓ Backup uploaded to S3"
        else
            log_error "✗ S3 upload failed"
        fi
    else
        log_warn "AWS CLI not available, skipping S3 upload"
    fi
    echo ""
fi

# 11. Clean up old backups
log_info "Cleaning up old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "vcci-backup-*.tar.gz" -type f -mtime +"$RETENTION_DAYS" -delete

REMAINING_BACKUPS=$(find "$BACKUP_DIR" -name "vcci-backup-*.tar.gz" -type f | wc -l)
log_info "✓ Old backups cleaned up"
log_info "  Remaining backups: $REMAINING_BACKUPS"
echo ""

# 12. Verify backup integrity
log_info "Verifying backup integrity..."
if tar -tzf "${BACKUP_PREFIX}.tar.gz" > /dev/null 2>&1; then
    log_info "✓ Backup archive is valid"
else
    log_error "✗ Backup archive is corrupted"
    exit 1
fi
echo ""

# 13. Create latest symlink
log_info "Updating latest backup symlink..."
ln -sf "${BACKUP_PREFIX}.tar.gz" "$BACKUP_DIR/vcci-backup-latest.tar.gz"
log_info "✓ Latest backup symlink updated"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "======================================================================"
echo "Backup Summary"
echo "======================================================================"
echo ""
echo -e "${GREEN}Backup completed successfully${NC}"
echo ""
echo "Backup file: $BACKUP_DIR/${BACKUP_PREFIX}.tar.gz"
echo "Backup size: $COMPRESSED_SIZE"
echo "Timestamp: $TIMESTAMP"
echo ""
echo "To restore from this backup, use:"
echo "  ./restore_production.sh -f ${BACKUP_PREFIX}.tar.gz"
echo ""
echo "======================================================================"
