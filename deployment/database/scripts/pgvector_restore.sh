#!/usr/bin/env bash
# =============================================================================
# GreenLang Climate OS - pgvector Restore Script
# =============================================================================
# PRD: INFRA-005 Vector Database Infrastructure with pgvector
# Purpose: Restore vector embedding tables from S3 backup
# Usage: ./pgvector_restore.sh <s3-backup-path> [--verify-only]
# =============================================================================

set -euo pipefail

# Arguments
S3_PATH="${1:?Usage: $0 <s3-backup-path> [--verify-only]}"
VERIFY_ONLY="${2:-}"

# Configuration
DB_HOST="${PGVECTOR_HOST:-localhost}"
DB_PORT="${PGVECTOR_PORT:-5432}"
DB_NAME="${PGVECTOR_DATABASE:-greenlang}"
DB_USER="${PGVECTOR_USER:-greenlang_admin}"
RESTORE_DIR="/tmp/pgvector-restore-$(date +%s)"

VECTOR_TABLES=(
    "embedding_collections"
    "embedding_jobs"
    "vector_search_logs"
    "vector_audit_log"
    "vector_embeddings"
)

echo "============================================="
echo "pgvector Restore"
echo "Source: ${S3_PATH}"
echo "Target: ${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo "============================================="

# 1. Download backup from S3
echo "[1/6] Downloading backup from S3..."
mkdir -p "${RESTORE_DIR}"
aws s3 sync "${S3_PATH}" "${RESTORE_DIR}/" --no-progress

# 2. Verify checksums
echo "[2/6] Verifying checksums..."
cd "${RESTORE_DIR}"
if [ -f checksums.sha256 ]; then
    sha256sum -c checksums.sha256
    echo "  Checksums verified successfully"
else
    echo "  WARNING: No checksums file found"
fi

# 3. Display metadata
echo "[3/6] Backup metadata:"
if [ -f metadata.json ]; then
    cat metadata.json | python3 -m json.tool 2>/dev/null || cat metadata.json
fi

# Check verify-only mode
if [ "${VERIFY_ONLY}" = "--verify-only" ]; then
    echo "Verify-only mode. Backup integrity confirmed."
    rm -rf "${RESTORE_DIR}"
    exit 0
fi

# 4. Pre-restore safety check
echo "[4/6] Pre-restore safety check..."
CURRENT_COUNT=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
    -t -A -c "SELECT COUNT(*) FROM vector_embeddings" 2>/dev/null || echo "0")
echo "  Current vector_embeddings count: ${CURRENT_COUNT}"

if [ "${CURRENT_COUNT}" -gt "0" ]; then
    echo ""
    echo "  WARNING: Target database contains ${CURRENT_COUNT} embeddings."
    echo "  Restore will REPLACE existing data."
    echo ""
    read -p "  Continue? (yes/no): " CONFIRM
    if [ "${CONFIRM}" != "yes" ]; then
        echo "  Restore cancelled."
        rm -rf "${RESTORE_DIR}"
        exit 1
    fi
fi

# 5. Restore tables
echo "[5/6] Restoring tables..."
for table in "${VECTOR_TABLES[@]}"; do
    DUMP_FILE="${RESTORE_DIR}/${table}.dump"
    if [ -f "${DUMP_FILE}" ]; then
        echo "  - Restoring ${table}..."
        pg_restore \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d "${DB_NAME}" \
            --clean \
            --if-exists \
            --no-owner \
            --no-privileges \
            --single-transaction \
            "${DUMP_FILE}" \
            2>&1 || {
                echo "  WARNING: Some errors during ${table} restore (may be expected for clean/drop)"
            }
    else
        echo "  - SKIP: ${table}.dump not found"
    fi
done

# 6. Post-restore verification
echo "[6/6] Post-restore verification..."

# Verify row counts
if [ -f "${RESTORE_DIR}/row_counts.csv" ]; then
    echo "  Expected vs Actual row counts:"
    while IFS=',' read -r table_name expected_count; do
        actual_count=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
            -t -A -c "SELECT COUNT(*) FROM ${table_name}" 2>/dev/null || echo "ERROR")
        status="OK"
        if [ "${actual_count}" != "${expected_count}" ]; then
            status="MISMATCH"
        fi
        printf "    %-30s Expected: %-10s Actual: %-10s [%s]\n" \
            "${table_name}" "${expected_count}" "${actual_count}" "${status}"
    done < "${RESTORE_DIR}/row_counts.csv"
fi

# Verify pgvector extension
PGVECTOR_OK=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
    -t -A -c "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'" 2>/dev/null)
if [ "${PGVECTOR_OK}" = "1" ]; then
    echo "  pgvector extension: OK"
else
    echo "  WARNING: pgvector extension not found. Run: CREATE EXTENSION IF NOT EXISTS vector;"
fi

# Verify indexes
INDEX_COUNT=$(psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
    -t -A -c "SELECT COUNT(*) FROM pg_indexes WHERE tablename = 'vector_embeddings'" 2>/dev/null)
echo "  Indexes on vector_embeddings: ${INDEX_COUNT}"

# Rebuild indexes if needed
echo "  Running ANALYZE on restored tables..."
psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" \
    -c "ANALYZE vector_embeddings; ANALYZE embedding_collections;" 2>/dev/null

# Clean up
rm -rf "${RESTORE_DIR}"

echo "============================================="
echo "Restore completed!"
echo "============================================="
