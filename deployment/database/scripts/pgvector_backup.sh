#!/usr/bin/env bash
# =============================================================================
# GreenLang Climate OS - pgvector Backup Script
# =============================================================================
# PRD: INFRA-005 Vector Database Infrastructure with pgvector
# Purpose: Logical backup of vector embedding tables to S3
# Usage: ./pgvector_backup.sh [daily|weekly|monthly]
# =============================================================================

set -euo pipefail

# Configuration
BACKUP_TYPE="${1:-daily}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%S")
DB_HOST="${PGVECTOR_HOST:-localhost}"
DB_PORT="${PGVECTOR_PORT:-5432}"
DB_NAME="${PGVECTOR_DATABASE:-greenlang}"
DB_USER="${PGVECTOR_USER:-greenlang_admin}"
S3_BUCKET="${BACKUP_S3_BUCKET:-greenlang-backups}"
S3_PREFIX="pgvector/${BACKUP_TYPE}/${TIMESTAMP}"
BACKUP_DIR="/tmp/pgvector-backup-${TIMESTAMP}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-90}"

# Tables to back up
VECTOR_TABLES=(
    "vector_embeddings"
    "embedding_collections"
    "embedding_jobs"
    "vector_search_logs"
    "vector_audit_log"
)

echo "============================================="
echo "pgvector Backup - ${BACKUP_TYPE}"
echo "Timestamp: ${TIMESTAMP}"
echo "Database: ${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo "S3 Target: s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "============================================="

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# 1. Dump vector tables (custom format for parallel restore)
echo "[1/5] Dumping vector tables..."
for table in "${VECTOR_TABLES[@]}"; do
    echo "  - Backing up ${table}..."
    pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -t "${table}" \
        -Fc \
        --no-owner \
        --no-privileges \
        --compress=9 \
        -f "${BACKUP_DIR}/${table}.dump" \
        2>&1 || {
            echo "ERROR: Failed to dump ${table}"
            exit 1
        }
done

# 2. Generate table row counts for verification
echo "[2/5] Recording row counts..."
psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
    -t -A -F',' \
    -c "SELECT tablename, n_live_tup FROM pg_stat_user_tables
        WHERE tablename IN ('vector_embeddings', 'embedding_collections',
                           'embedding_jobs', 'vector_search_logs', 'vector_audit_log')
        ORDER BY tablename;" \
    > "${BACKUP_DIR}/row_counts.csv"

# 3. Generate checksums
echo "[3/5] Generating checksums..."
cd "${BACKUP_DIR}"
sha256sum *.dump > checksums.sha256

# 4. Create metadata file
echo "[4/5] Creating metadata..."
cat > "${BACKUP_DIR}/metadata.json" << EOF
{
    "backup_type": "${BACKUP_TYPE}",
    "timestamp": "${TIMESTAMP}",
    "database": {
        "host": "${DB_HOST}",
        "port": ${DB_PORT},
        "name": "${DB_NAME}"
    },
    "tables": $(printf '%s\n' "${VECTOR_TABLES[@]}" | jq -R . | jq -s .),
    "pg_dump_version": "$(pg_dump --version | head -1)",
    "pgvector_version": "$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -A -c "SELECT extversion FROM pg_extension WHERE extname = 'vector'")"
}
EOF

# 5. Upload to S3
echo "[5/5] Uploading to S3..."
aws s3 sync "${BACKUP_DIR}/" "s3://${S3_BUCKET}/${S3_PREFIX}/" \
    --sse aws:kms \
    --storage-class STANDARD_IA \
    --no-progress

# Clean up local backup
rm -rf "${BACKUP_DIR}"

# Clean old backups based on retention
if [ "${BACKUP_TYPE}" = "daily" ]; then
    echo "Cleaning backups older than ${RETENTION_DAYS} days..."
    CUTOFF_DATE=$(date -u -d "${RETENTION_DAYS} days ago" +"%Y-%m-%d" 2>/dev/null || \
                  date -u -v-${RETENTION_DAYS}d +"%Y-%m-%d")
    aws s3 ls "s3://${S3_BUCKET}/pgvector/daily/" | while read -r line; do
        BACKUP_DATE=$(echo "${line}" | awk '{print $2}' | cut -d'T' -f1 | tr -d '/')
        if [[ "${BACKUP_DATE}" < "${CUTOFF_DATE}" ]]; then
            FOLDER=$(echo "${line}" | awk '{print $2}')
            echo "  Removing old backup: ${FOLDER}"
            aws s3 rm "s3://${S3_BUCKET}/pgvector/daily/${FOLDER}" --recursive --quiet
        fi
    done
fi

echo "============================================="
echo "Backup completed successfully!"
echo "Location: s3://${S3_BUCKET}/${S3_PREFIX}/"
echo "============================================="
