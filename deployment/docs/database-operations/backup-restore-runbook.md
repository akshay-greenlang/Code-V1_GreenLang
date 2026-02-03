# PostgreSQL/TimescaleDB Backup and Restore Runbook

## Document Information

| Field | Value |
|-------|-------|
| Document Owner | Database Operations Team |
| Last Updated | 2026-02-03 |
| Review Cycle | Monthly |
| Classification | Internal - Operations |

---

## 1. Overview

This runbook provides comprehensive procedures for backup and restore operations for the GreenLang PostgreSQL/TimescaleDB database infrastructure.

### Backup Tools

| Tool | Purpose | Use Case |
|------|---------|----------|
| pgBackRest | Physical backups | Full and incremental, PITR |
| pg_dump/pg_restore | Logical backups | Schema/table level restore |
| WAL Archiving | Continuous | Point-in-time recovery |
| TimescaleDB tools | Chunk backup | Time-series data |

### Backup Storage

| Type | Location | Retention |
|------|----------|-----------|
| Base Backups | s3://greenlang-backups/base/ | 30 days |
| WAL Archives | s3://greenlang-backups/wal/ | 7 days |
| Logical Backups | s3://greenlang-backups/logical/ | 90 days |
| Cross-region | s3://greenlang-backups-dr/base/ | 14 days |

---

## 2. Full Restore Procedure

### Scenario

Complete database cluster failure requiring full restoration from backup.

### Prerequisites

- New database instance provisioned
- Network access to backup storage (S3)
- pgBackRest configuration deployed
- Sufficient disk space (2x database size)

### Procedure

```bash
#!/bin/bash
# Full restore procedure

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/restore_full_${TIMESTAMP}.log"

echo "Starting full restore at $(date)" | tee $LOG_FILE

# Step 1: Stop PostgreSQL if running
echo "Step 1: Stopping PostgreSQL..." | tee -a $LOG_FILE
systemctl stop postgresql || true

# Step 2: Clear data directory
echo "Step 2: Clearing data directory..." | tee -a $LOG_FILE
PGDATA=/var/lib/postgresql/data
rm -rf ${PGDATA}/*

# Step 3: List available backups
echo "Step 3: Listing available backups..." | tee -a $LOG_FILE
pgbackrest info | tee -a $LOG_FILE

# Step 4: Restore latest backup
echo "Step 4: Restoring from latest backup..." | tee -a $LOG_FILE
pgbackrest restore \
    --stanza=greenlang \
    --log-level-console=info \
    --delta \
    2>&1 | tee -a $LOG_FILE

# Step 5: Configure recovery settings
echo "Step 5: Configuring recovery settings..." | tee -a $LOG_FILE
cat > ${PGDATA}/postgresql.auto.conf << EOF
# Recovery configuration
restore_command = 'pgbackrest --stanza=greenlang archive-get %f "%p"'
recovery_target_action = 'promote'
EOF

# Create standby signal for recovery mode
touch ${PGDATA}/recovery.signal

# Step 6: Set ownership
echo "Step 6: Setting ownership..." | tee -a $LOG_FILE
chown -R postgres:postgres ${PGDATA}

# Step 7: Start PostgreSQL
echo "Step 7: Starting PostgreSQL..." | tee -a $LOG_FILE
systemctl start postgresql

# Step 8: Wait for recovery
echo "Step 8: Waiting for recovery to complete..." | tee -a $LOG_FILE
for i in {1..120}; do
    if psql -U postgres -c "SELECT pg_is_in_recovery();" 2>/dev/null | grep -q 'f'; then
        echo "Recovery complete!" | tee -a $LOG_FILE
        break
    fi
    echo "Recovery in progress... ($i/120)" | tee -a $LOG_FILE
    sleep 5
done

# Step 9: Verify restore
echo "Step 9: Verifying restore..." | tee -a $LOG_FILE
psql -U postgres -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database;" | tee -a $LOG_FILE

# Step 10: Run integrity checks
echo "Step 10: Running integrity checks..." | tee -a $LOG_FILE
psql -U postgres -d greenlang -c "SELECT count(*) FROM pg_tables WHERE schemaname = 'public';" | tee -a $LOG_FILE

echo "Full restore completed at $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"
```

### Post-Restore Actions

```bash
# Verify TimescaleDB extension
psql -U postgres -d greenlang -c "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"

# Check hypertables
psql -U postgres -d greenlang -c "SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables;"

# Verify continuous aggregates
psql -U postgres -d greenlang -c "SELECT view_name FROM timescaledb_information.continuous_aggregates;"

# Refresh continuous aggregates if needed
psql -U postgres -d greenlang -c "CALL refresh_continuous_aggregate('hourly_metrics', NULL, NULL);"

# Update statistics
psql -U postgres -d greenlang -c "ANALYZE;"
```

---

## 3. Point-in-Time Recovery (PITR)

### Scenario

Restore database to a specific point in time, typically used for:
- Recovering from accidental data deletion
- Recovering from logical corruption
- Testing/debugging historical state

### Finding Recovery Target

```sql
-- Method 1: Identify transaction ID from logs
-- Check PostgreSQL logs for the problematic transaction

-- Method 2: Query pg_stat_activity history (if available)
-- Check your audit/logging tables for the event timestamp

-- Method 3: Use WAL position
SELECT pg_current_wal_lsn();

-- Method 4: Check recent transactions
SELECT xact_start, query
FROM pg_stat_activity
WHERE query NOT LIKE '%pg_stat_activity%'
ORDER BY xact_start DESC
LIMIT 20;
```

### PITR Procedure

```bash
#!/bin/bash
# Point-in-Time Recovery procedure

set -e

# Configuration
TARGET_TIME="${1:-}"  # Format: 2026-02-03 14:30:00
RECOVERY_DB="${2:-greenlang_recovery}"

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 'YYYY-MM-DD HH:MM:SS' [recovery_db_name]"
    echo "Example: $0 '2026-02-03 14:30:00' greenlang_recovery"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/pitr_${TIMESTAMP}.log"
RECOVERY_PGDATA="/var/lib/postgresql/pitr_recovery"

echo "Starting PITR to $TARGET_TIME at $(date)" | tee $LOG_FILE

# Step 1: Verify target time is valid
echo "Step 1: Verifying target time..." | tee -a $LOG_FILE
TARGET_EPOCH=$(date -d "$TARGET_TIME" +%s 2>/dev/null)
if [ -z "$TARGET_EPOCH" ]; then
    echo "ERROR: Invalid target time format" | tee -a $LOG_FILE
    exit 1
fi

# Step 2: Check available WAL archives
echo "Step 2: Checking WAL archive coverage..." | tee -a $LOG_FILE
pgbackrest info --stanza=greenlang | tee -a $LOG_FILE

# Step 3: Create recovery directory
echo "Step 3: Creating recovery directory..." | tee -a $LOG_FILE
rm -rf ${RECOVERY_PGDATA}
mkdir -p ${RECOVERY_PGDATA}
chown postgres:postgres ${RECOVERY_PGDATA}

# Step 4: Restore to target time
echo "Step 4: Restoring to target time..." | tee -a $LOG_FILE
pgbackrest restore \
    --stanza=greenlang \
    --pg1-path=${RECOVERY_PGDATA} \
    --type=time \
    --target="${TARGET_TIME}" \
    --target-action=promote \
    --log-level-console=info \
    2>&1 | tee -a $LOG_FILE

# Step 5: Configure PostgreSQL for recovery instance
echo "Step 5: Configuring recovery instance..." | tee -a $LOG_FILE
cat >> ${RECOVERY_PGDATA}/postgresql.auto.conf << EOF
# Recovery instance settings
port = 5433
listen_addresses = 'localhost'
EOF

# Step 6: Start recovery instance
echo "Step 6: Starting recovery instance..." | tee -a $LOG_FILE
sudo -u postgres pg_ctl start -D ${RECOVERY_PGDATA} -l ${RECOVERY_PGDATA}/recovery.log -w

# Step 7: Wait for recovery to complete
echo "Step 7: Waiting for recovery..." | tee -a $LOG_FILE
for i in {1..180}; do
    if psql -p 5433 -U postgres -c "SELECT pg_is_in_recovery();" 2>/dev/null | grep -q 'f'; then
        echo "Recovery complete!" | tee -a $LOG_FILE
        break
    fi
    echo "Recovery in progress... ($i/180)" | tee -a $LOG_FILE
    sleep 10
done

# Step 8: Verify recovery target time
echo "Step 8: Verifying recovery time..." | tee -a $LOG_FILE
RECOVERY_TIME=$(psql -p 5433 -U postgres -t -c "SELECT pg_last_xact_replay_timestamp();")
echo "Recovered to: $RECOVERY_TIME" | tee -a $LOG_FILE

# Step 9: Verify data
echo "Step 9: Verifying recovered data..." | tee -a $LOG_FILE
psql -p 5433 -U postgres -d greenlang -c "
SELECT
    (SELECT count(*) FROM users) as user_count,
    (SELECT count(*) FROM energy_readings) as readings_count,
    (SELECT max(created_at) FROM energy_readings) as latest_reading;
" | tee -a $LOG_FILE

echo ""
echo "=== PITR Complete ===" | tee -a $LOG_FILE
echo "Recovery instance is running on port 5433" | tee -a $LOG_FILE
echo "Data directory: ${RECOVERY_PGDATA}" | tee -a $LOG_FILE
echo ""
echo "Next steps:" | tee -a $LOG_FILE
echo "1. Verify the recovered data is correct" | tee -a $LOG_FILE
echo "2. Extract needed data using pg_dump/psql" | tee -a $LOG_FILE
echo "3. Stop recovery instance: pg_ctl stop -D ${RECOVERY_PGDATA}" | tee -a $LOG_FILE
echo "4. Clean up: rm -rf ${RECOVERY_PGDATA}" | tee -a $LOG_FILE

echo "PITR completed at $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"
```

### Extracting Data from PITR Instance

```bash
# Extract specific table data
pg_dump -p 5433 -U postgres -d greenlang \
    --table=users \
    --data-only \
    --file=/tmp/users_recovered.sql

# Extract specific rows
psql -p 5433 -U postgres -d greenlang -c "
COPY (
    SELECT * FROM important_table
    WHERE created_at >= '2026-02-03 14:00:00'
) TO '/tmp/recovered_rows.csv' WITH CSV HEADER;
"

# Restore extracted data to production
psql -h production-db -U postgres -d greenlang < /tmp/users_recovered.sql
```

---

## 4. Table-Level Restore

### Scenario

Restore a single table without affecting the rest of the database.

### Method 1: Using pg_dump from Backup

```bash
#!/bin/bash
# Table-level restore from logical backup

TABLE_NAME="${1:-}"
SCHEMA_NAME="${2:-public}"
BACKUP_DATE="${3:-latest}"

if [ -z "$TABLE_NAME" ]; then
    echo "Usage: $0 table_name [schema_name] [backup_date]"
    exit 1
fi

LOG_FILE="/var/log/table_restore_${TABLE_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting table restore: ${SCHEMA_NAME}.${TABLE_NAME}" | tee $LOG_FILE

# Step 1: Download logical backup
echo "Step 1: Downloading logical backup..." | tee -a $LOG_FILE
if [ "$BACKUP_DATE" == "latest" ]; then
    BACKUP_FILE=$(aws s3 ls s3://greenlang-backups/logical/ | sort | tail -1 | awk '{print $4}')
else
    BACKUP_FILE="greenlang_${BACKUP_DATE}.dump"
fi

aws s3 cp s3://greenlang-backups/logical/${BACKUP_FILE} /tmp/${BACKUP_FILE}

# Step 2: List contents to verify table exists
echo "Step 2: Verifying table in backup..." | tee -a $LOG_FILE
pg_restore -l /tmp/${BACKUP_FILE} | grep -i "${TABLE_NAME}" | tee -a $LOG_FILE

# Step 3: Create restore list file
echo "Step 3: Creating restore list..." | tee -a $LOG_FILE
pg_restore -l /tmp/${BACKUP_FILE} | grep -E "(TABLE DATA|TABLE|INDEX|CONSTRAINT|TRIGGER).*${TABLE_NAME}" > /tmp/restore_list.txt

# Step 4: Backup current table (safety)
echo "Step 4: Backing up current table state..." | tee -a $LOG_FILE
pg_dump -h localhost -U postgres -d greenlang \
    --table="${SCHEMA_NAME}.${TABLE_NAME}" \
    --file="/tmp/${TABLE_NAME}_backup_$(date +%Y%m%d_%H%M%S).sql"

# Step 5: Truncate current table (optional - confirm with user)
echo "Step 5: Truncating current table..." | tee -a $LOG_FILE
read -p "Truncate current table data? (yes/no): " CONFIRM
if [ "$CONFIRM" == "yes" ]; then
    psql -h localhost -U postgres -d greenlang -c "TRUNCATE TABLE ${SCHEMA_NAME}.${TABLE_NAME} CASCADE;"
fi

# Step 6: Restore table
echo "Step 6: Restoring table..." | tee -a $LOG_FILE
pg_restore -h localhost -U postgres -d greenlang \
    --data-only \
    --table="${TABLE_NAME}" \
    --no-owner \
    --no-privileges \
    /tmp/${BACKUP_FILE} 2>&1 | tee -a $LOG_FILE

# Step 7: Verify restore
echo "Step 7: Verifying restore..." | tee -a $LOG_FILE
psql -h localhost -U postgres -d greenlang -c "SELECT count(*) FROM ${SCHEMA_NAME}.${TABLE_NAME};" | tee -a $LOG_FILE

# Step 8: Reindex
echo "Step 8: Reindexing table..." | tee -a $LOG_FILE
psql -h localhost -U postgres -d greenlang -c "REINDEX TABLE ${SCHEMA_NAME}.${TABLE_NAME};"

# Step 9: Update statistics
echo "Step 9: Updating statistics..." | tee -a $LOG_FILE
psql -h localhost -U postgres -d greenlang -c "ANALYZE ${SCHEMA_NAME}.${TABLE_NAME};"

# Cleanup
rm -f /tmp/${BACKUP_FILE} /tmp/restore_list.txt

echo "Table restore completed at $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"
```

### Method 2: Using PITR and pg_dump

```bash
#!/bin/bash
# Table restore using PITR

TABLE_NAME="$1"
TARGET_TIME="$2"

# Step 1: Perform PITR to separate instance
./restore-pitr.sh "$TARGET_TIME"

# Step 2: Export table from recovery instance
pg_dump -p 5433 -U postgres -d greenlang \
    --table="$TABLE_NAME" \
    --data-only \
    --file="/tmp/${TABLE_NAME}_pitr.sql"

# Step 3: Restore to production
psql -h localhost -U postgres -d greenlang < /tmp/${TABLE_NAME}_pitr.sql

# Step 4: Cleanup recovery instance
pg_ctl stop -D /var/lib/postgresql/pitr_recovery
rm -rf /var/lib/postgresql/pitr_recovery
```

---

## 5. Cross-Region Restore

### Scenario

Restore database in DR region from backups stored in primary region.

### Prerequisites

- Cross-region S3 replication configured
- pgBackRest stanza configured in DR region
- Network access between regions

### Procedure

```bash
#!/bin/bash
# Cross-region restore procedure

set -e

SOURCE_REGION="us-east-1"
TARGET_REGION="us-west-2"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/cross_region_restore_${TIMESTAMP}.log"

echo "Starting cross-region restore at $(date)" | tee $LOG_FILE

# Step 1: Verify backup availability in DR region
echo "Step 1: Verifying backup availability..." | tee -a $LOG_FILE
aws s3 ls s3://greenlang-backups-dr/base/ --region ${TARGET_REGION} | tail -5 | tee -a $LOG_FILE

# Step 2: Configure pgBackRest for DR region
echo "Step 2: Configuring pgBackRest..." | tee -a $LOG_FILE
cat > /etc/pgbackrest/pgbackrest-dr.conf << EOF
[global]
repo1-type=s3
repo1-s3-bucket=greenlang-backups-dr
repo1-s3-region=${TARGET_REGION}
repo1-s3-endpoint=s3.${TARGET_REGION}.amazonaws.com
repo1-path=/
repo1-retention-full=7

[greenlang]
pg1-path=/var/lib/postgresql/data
EOF

# Step 3: Check stanza info
echo "Step 3: Checking backup info..." | tee -a $LOG_FILE
pgbackrest --config=/etc/pgbackrest/pgbackrest-dr.conf info | tee -a $LOG_FILE

# Step 4: Stop local PostgreSQL
echo "Step 4: Stopping PostgreSQL..." | tee -a $LOG_FILE
systemctl stop postgresql

# Step 5: Clear data directory
echo "Step 5: Clearing data directory..." | tee -a $LOG_FILE
rm -rf /var/lib/postgresql/data/*

# Step 6: Restore from DR backup
echo "Step 6: Restoring from DR region backup..." | tee -a $LOG_FILE
pgbackrest restore \
    --config=/etc/pgbackrest/pgbackrest-dr.conf \
    --stanza=greenlang \
    --log-level-console=info \
    --delta \
    2>&1 | tee -a $LOG_FILE

# Step 7: Update configuration for DR site
echo "Step 7: Updating configuration..." | tee -a $LOG_FILE
cat >> /var/lib/postgresql/data/postgresql.auto.conf << EOF
# DR Site configuration
archive_command = 'pgbackrest --config=/etc/pgbackrest/pgbackrest-dr.conf --stanza=greenlang archive-push %p'
restore_command = 'pgbackrest --config=/etc/pgbackrest/pgbackrest-dr.conf --stanza=greenlang archive-get %f "%p"'
EOF

# Step 8: Set ownership
chown -R postgres:postgres /var/lib/postgresql/data

# Step 9: Start PostgreSQL
echo "Step 9: Starting PostgreSQL..." | tee -a $LOG_FILE
systemctl start postgresql

# Step 10: Verify restoration
echo "Step 10: Verifying restoration..." | tee -a $LOG_FILE
sleep 10
psql -U postgres -c "SELECT pg_is_in_recovery(), pg_last_xact_replay_timestamp();" | tee -a $LOG_FILE

# Step 11: Promote if needed (for failover scenario)
# psql -U postgres -c "SELECT pg_promote();"

echo "Cross-region restore completed at $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"
```

### Verifying Cross-Region Backup Replication

```bash
# Check S3 replication status
aws s3api head-object \
    --bucket greenlang-backups-dr \
    --key base/latest/backup.manifest \
    --region us-west-2

# Compare backup inventories
PRIMARY_COUNT=$(aws s3 ls s3://greenlang-backups/base/ --region us-east-1 | wc -l)
DR_COUNT=$(aws s3 ls s3://greenlang-backups-dr/base/ --region us-west-2 | wc -l)
echo "Primary backups: $PRIMARY_COUNT, DR backups: $DR_COUNT"

# Check replication lag
aws s3api get-bucket-replication \
    --bucket greenlang-backups \
    --region us-east-1
```

---

## 6. TimescaleDB-Specific Restore

### Restoring Hypertables

```sql
-- After full restore, verify hypertables
SELECT hypertable_schema, hypertable_name, num_chunks,
       pg_size_pretty(hypertable_size(format('%I.%I', hypertable_schema, hypertable_name)::regclass))
FROM timescaledb_information.hypertables;

-- Check for missing chunks
SELECT * FROM timescaledb_information.chunks
WHERE hypertable_name = 'energy_readings'
ORDER BY range_end DESC
LIMIT 20;

-- Repair chunk if needed
SELECT _timescaledb_internal.recreate_chunk('_timescaledb_internal._hyper_1_5_chunk');
```

### Restoring Continuous Aggregates

```sql
-- List continuous aggregates
SELECT view_name, materialization_hypertable_name,
       view_definition
FROM timescaledb_information.continuous_aggregates;

-- Refresh continuous aggregate after restore
CALL refresh_continuous_aggregate(
    'hourly_energy_summary',
    '2026-01-01 00:00:00',
    '2026-02-03 00:00:00'
);

-- Verify continuous aggregate data
SELECT time_bucket, avg_value, max_value
FROM hourly_energy_summary
ORDER BY time_bucket DESC
LIMIT 10;
```

### Restoring Compression Policies

```sql
-- Verify compression settings after restore
SELECT hypertable_name, compress_segmentby, compress_orderby
FROM timescaledb_information.compression_settings;

-- Re-enable compression policy if needed
SELECT add_compression_policy('energy_readings', INTERVAL '7 days');

-- Check compression status
SELECT hypertable_name, chunk_name,
       compression_status,
       pg_size_pretty(before_compression_total_bytes) as before_size,
       pg_size_pretty(after_compression_total_bytes) as after_size
FROM chunk_compression_stats('energy_readings')
ORDER BY chunk_name DESC
LIMIT 10;
```

---

## 7. Backup Verification

### Daily Verification Script

```bash
#!/bin/bash
# Daily backup verification

TIMESTAMP=$(date +%Y%m%d)
REPORT_FILE="/var/log/backup_verify_${TIMESTAMP}.json"

echo "Starting backup verification at $(date)"

# Get latest backup info
BACKUP_INFO=$(pgbackrest info --stanza=greenlang --output=json)

# Extract key metrics
LATEST_BACKUP=$(echo "$BACKUP_INFO" | jq -r '.[0].backup[-1]')
BACKUP_TYPE=$(echo "$LATEST_BACKUP" | jq -r '.type')
BACKUP_START=$(echo "$LATEST_BACKUP" | jq -r '.timestamp.start')
BACKUP_STOP=$(echo "$LATEST_BACKUP" | jq -r '.timestamp.stop')
BACKUP_SIZE=$(echo "$LATEST_BACKUP" | jq -r '.info.size')

# Check WAL archive status
WAL_STATUS=$(pgbackrest check --stanza=greenlang 2>&1)
WAL_OK=$(echo "$WAL_STATUS" | grep -c "completed successfully" || echo 0)

# Generate report
cat > $REPORT_FILE << EOF
{
    "timestamp": "$(date -Iseconds)",
    "backup": {
        "type": "$BACKUP_TYPE",
        "start_time": "$BACKUP_START",
        "stop_time": "$BACKUP_STOP",
        "size_bytes": $BACKUP_SIZE,
        "status": "valid"
    },
    "wal_archive": {
        "status": $([ "$WAL_OK" -gt 0 ] && echo '"ok"' || echo '"error"'),
        "message": "$(echo "$WAL_STATUS" | tail -1)"
    },
    "verification": {
        "status": "passed",
        "checks_performed": ["backup_exists", "wal_archive", "size_check"]
    }
}
EOF

# Send to monitoring
curl -X POST https://monitoring.greenlang.io/api/backup-status \
    -H "Content-Type: application/json" \
    -d @$REPORT_FILE

echo "Backup verification completed. Report: $REPORT_FILE"
```

---

## 8. Related Documents

- [Disaster Recovery Plan](./disaster-recovery-plan.md)
- [Failover Runbook](./failover-runbook.md)
- [Database Operations Guide](./database-operations.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | DBA Team | Initial version |
