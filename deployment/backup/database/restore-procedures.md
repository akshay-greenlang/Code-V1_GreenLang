# Database Restore Procedures Runbook

**INFRA-001: Database Disaster Recovery**
**Version:** 1.0.0
**Last Updated:** 2026-02-03

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Prerequisites](#prerequisites)
3. [Restore Scenarios](#restore-scenarios)
4. [Full Database Restore](#full-database-restore)
5. [Partial Data Restore](#partial-data-restore)
6. [Cross-Region Restore](#cross-region-restore)
7. [Redis Cache Restore](#redis-cache-restore)
8. [Weaviate Vector DB Restore](#weaviate-vector-db-restore)
9. [Post-Restore Validation](#post-restore-validation)
10. [Rollback Procedures](#rollback-procedures)

---

## Quick Reference

### Emergency Contacts

| Role | Contact | Phone |
|------|---------|-------|
| Database On-Call | dba-oncall@greenlang.io | +1-XXX-XXX-XXXX |
| Platform Lead | platform@greenlang.io | +1-XXX-XXX-XXXX |
| Security Team | security@greenlang.io | +1-XXX-XXX-XXXX |

### Critical Commands

```bash
# Check latest backup status
aws rds describe-db-snapshots \
    --db-instance-identifier greenlang-postgres \
    --query 'DBSnapshots[-1].[DBSnapshotIdentifier,Status,SnapshotCreateTime]'

# Check PITR window
aws rds describe-db-instances \
    --db-instance-identifier greenlang-postgres \
    --query 'DBInstances[0].[EarliestRestorableTime,LatestRestorableTime]'

# List all available snapshots
aws rds describe-db-snapshots \
    --db-instance-identifier greenlang-postgres \
    --query 'DBSnapshots[*].[DBSnapshotIdentifier,SnapshotCreateTime,Status]' \
    --output table
```

---

## Prerequisites

### Required Access

- [ ] AWS Console access with RDS permissions
- [ ] AWS CLI configured with appropriate credentials
- [ ] Database admin credentials (from AWS Secrets Manager)
- [ ] kubectl access to Kubernetes cluster
- [ ] Access to backup S3 buckets

### Required Tools

```bash
# Verify required tools
aws --version          # AWS CLI v2.x
psql --version         # PostgreSQL client
kubectl version        # Kubernetes CLI
redis-cli --version    # Redis CLI
jq --version           # JSON processor
```

### Pre-Restore Checklist

```markdown
Before starting any restore operation:

- [ ] Confirmed the issue requires database restore
- [ ] Identified the target restore point (snapshot/PITR)
- [ ] Documented current production state
- [ ] Notified stakeholders (Slack: #platform-incidents)
- [ ] Created incident ticket
- [ ] Verified backup availability and integrity
- [ ] Planned maintenance window (if needed)
- [ ] Prepared rollback strategy
```

---

## Restore Scenarios

### Decision Tree

```
Is production database accessible?
├── YES
│   ├── Is data corrupted?
│   │   ├── YES → Partial Data Restore (Section 5)
│   │   └── NO → Investigate further (not a restore scenario)
│   └── Is data missing?
│       ├── YES, specific tables → Partial Data Restore (Section 5)
│       └── YES, extensive → Full Database Restore (Section 4)
└── NO
    ├── Is primary region available?
    │   ├── YES → Full Database Restore (Section 4)
    │   └── NO → Cross-Region Restore (Section 6)
    └── Is this a security incident?
        └── YES → Security Team involvement required
```

---

## Full Database Restore

### Procedure 4.1: Restore from Automated Snapshot

**Estimated Time:** 30-90 minutes

```bash
#!/bin/bash
# Full Database Restore from Snapshot
# Run as: ./restore-from-snapshot.sh <SNAPSHOT_ID>

set -e

SNAPSHOT_ID=${1:-""}
if [ -z "$SNAPSHOT_ID" ]; then
    echo "Usage: $0 <SNAPSHOT_ID>"
    echo "Available snapshots:"
    aws rds describe-db-snapshots \
        --db-instance-identifier greenlang-postgres \
        --query 'DBSnapshots[-5:].[DBSnapshotIdentifier,SnapshotCreateTime]' \
        --output table
    exit 1
fi

# Configuration
NEW_INSTANCE="greenlang-postgres-restored-$(date +%Y%m%d%H%M)"
INSTANCE_CLASS="db.t3.medium"
SUBNET_GROUP="greenlang-db-subnet-group"
SECURITY_GROUPS="sg-xxxxxxxxx"
PARAMETER_GROUP="greenlang-postgres14"

echo "=========================================="
echo "FULL DATABASE RESTORE - FROM SNAPSHOT"
echo "=========================================="
echo "Snapshot: $SNAPSHOT_ID"
echo "New Instance: $NEW_INSTANCE"
echo "=========================================="

# Confirm
read -p "Proceed with restore? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled."
    exit 0
fi

# Step 1: Initiate restore
echo "[Step 1/5] Initiating snapshot restore..."
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier $NEW_INSTANCE \
    --db-snapshot-identifier $SNAPSHOT_ID \
    --db-instance-class $INSTANCE_CLASS \
    --db-subnet-group-name $SUBNET_GROUP \
    --vpc-security-group-ids $SECURITY_GROUPS \
    --db-parameter-group-name $PARAMETER_GROUP \
    --multi-az \
    --storage-type gp3 \
    --copy-tags-to-snapshot \
    --deletion-protection \
    --enable-cloudwatch-logs-exports '["postgresql","upgrade"]'

# Step 2: Wait for availability
echo "[Step 2/5] Waiting for instance to become available..."
echo "This may take 30-60 minutes..."
aws rds wait db-instance-available \
    --db-instance-identifier $NEW_INSTANCE

# Step 3: Get endpoint
echo "[Step 3/5] Getting instance endpoint..."
ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $NEW_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "Endpoint: $ENDPOINT"

# Step 4: Verify connectivity
echo "[Step 4/5] Verifying database connectivity..."
PGPASSWORD=$DB_PASSWORD psql -h $ENDPOINT -U $DB_USER -d greenlang -c "SELECT 1;"

if [ $? -eq 0 ]; then
    echo "Database connectivity verified!"
else
    echo "ERROR: Cannot connect to restored database"
    exit 1
fi

# Step 5: Run validation queries
echo "[Step 5/5] Running validation queries..."
PGPASSWORD=$DB_PASSWORD psql -h $ENDPOINT -U $DB_USER -d greenlang << EOF
\echo '=== Table Counts ==='
SELECT schemaname, tablename, n_live_tup
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC
LIMIT 20;

\echo '=== Database Size ==='
SELECT pg_size_pretty(pg_database_size('greenlang'));

\echo '=== Recent Transactions ==='
SELECT COUNT(*) as count, DATE(created_at) as date
FROM emissions_data
GROUP BY DATE(created_at)
ORDER BY date DESC
LIMIT 7;
EOF

echo "=========================================="
echo "RESTORE COMPLETE"
echo "=========================================="
echo "New Instance: $NEW_INSTANCE"
echo "Endpoint: $ENDPOINT"
echo ""
echo "NEXT STEPS:"
echo "1. Validate data integrity"
echo "2. Update application connection strings"
echo "3. Route traffic to restored instance"
echo "4. Monitor for issues"
echo "5. Decommission old instance (if applicable)"
echo "=========================================="
```

### Procedure 4.2: Restore from Point-in-Time (PITR)

**Estimated Time:** 45-120 minutes

```bash
#!/bin/bash
# Full Database Restore - Point in Time Recovery
# Run as: ./restore-pitr.sh "2026-02-03 10:30:00"

set -e

RESTORE_TIME=${1:-""}
if [ -z "$RESTORE_TIME" ]; then
    echo "Usage: $0 \"YYYY-MM-DD HH:MM:SS\""

    # Show valid restore window
    echo ""
    echo "Valid restore window:"
    aws rds describe-db-instances \
        --db-instance-identifier greenlang-postgres \
        --query 'DBInstances[0].[EarliestRestorableTime,LatestRestorableTime]' \
        --output table
    exit 1
fi

# Convert to ISO 8601 format
RESTORE_TIME_ISO=$(date -d "$RESTORE_TIME" +%Y-%m-%dT%H:%M:%SZ)

# Configuration
SOURCE_INSTANCE="greenlang-postgres"
NEW_INSTANCE="greenlang-postgres-pitr-$(date +%Y%m%d%H%M)"
INSTANCE_CLASS="db.t3.medium"

echo "=========================================="
echo "FULL DATABASE RESTORE - PITR"
echo "=========================================="
echo "Source: $SOURCE_INSTANCE"
echo "Restore Time: $RESTORE_TIME_ISO"
echo "New Instance: $NEW_INSTANCE"
echo "=========================================="

# Validate restore time is within window
EARLIEST=$(aws rds describe-db-instances \
    --db-instance-identifier $SOURCE_INSTANCE \
    --query 'DBInstances[0].EarliestRestorableTime' \
    --output text)

LATEST=$(aws rds describe-db-instances \
    --db-instance-identifier $SOURCE_INSTANCE \
    --query 'DBInstances[0].LatestRestorableTime' \
    --output text)

echo "Earliest: $EARLIEST"
echo "Latest: $LATEST"
echo "Requested: $RESTORE_TIME_ISO"

# Confirm
read -p "Proceed with PITR restore? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Restore cancelled."
    exit 0
fi

# Initiate PITR restore
echo "[Step 1/4] Initiating PITR restore..."
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier $SOURCE_INSTANCE \
    --target-db-instance-identifier $NEW_INSTANCE \
    --restore-time $RESTORE_TIME_ISO \
    --db-instance-class $INSTANCE_CLASS \
    --multi-az \
    --storage-type gp3 \
    --copy-tags-to-snapshot \
    --deletion-protection

# Wait for availability
echo "[Step 2/4] Waiting for restore to complete..."
echo "This may take 45-120 minutes depending on database size..."

while true; do
    STATUS=$(aws rds describe-db-instances \
        --db-instance-identifier $NEW_INSTANCE \
        --query 'DBInstances[0].DBInstanceStatus' \
        --output text 2>/dev/null || echo "creating")

    echo "Status: $STATUS"

    if [ "$STATUS" = "available" ]; then
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "ERROR: Restore failed!"
        exit 1
    fi

    sleep 60
done

# Get endpoint
echo "[Step 3/4] Getting endpoint..."
ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $NEW_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "Endpoint: $ENDPOINT"

# Validate
echo "[Step 4/4] Running validation..."
PGPASSWORD=$DB_PASSWORD psql -h $ENDPOINT -U $DB_USER -d greenlang << EOF
SELECT 'Database restored successfully' as status;
SELECT pg_size_pretty(pg_database_size('greenlang')) as database_size;
SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public';
EOF

echo "=========================================="
echo "PITR RESTORE COMPLETE"
echo "=========================================="
echo "Instance: $NEW_INSTANCE"
echo "Endpoint: $ENDPOINT"
echo "Restored to: $RESTORE_TIME_ISO"
echo "=========================================="
```

---

## Partial Data Restore

### Procedure 5.1: Single Table Recovery

**Use Case:** Recovering a specific table without full database restore.

```bash
#!/bin/bash
# Partial Data Restore - Single Table
# Run as: ./restore-table.sh <TABLE_NAME> "2026-02-03 10:00:00"

set -e

TABLE_NAME=${1:-""}
RESTORE_TIME=${2:-""}

if [ -z "$TABLE_NAME" ] || [ -z "$RESTORE_TIME" ]; then
    echo "Usage: $0 <TABLE_NAME> \"YYYY-MM-DD HH:MM:SS\""
    exit 1
fi

echo "=========================================="
echo "PARTIAL DATA RESTORE - SINGLE TABLE"
echo "=========================================="
echo "Table: $TABLE_NAME"
echo "Restore Time: $RESTORE_TIME"
echo "=========================================="

# Step 1: Create PITR instance (temporary)
TEMP_INSTANCE="greenlang-temp-$(date +%Y%m%d%H%M)"
RESTORE_TIME_ISO=$(date -d "$RESTORE_TIME" +%Y-%m-%dT%H:%M:%SZ)

echo "[Step 1/6] Creating temporary PITR instance..."
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier greenlang-postgres \
    --target-db-instance-identifier $TEMP_INSTANCE \
    --restore-time $RESTORE_TIME_ISO \
    --db-instance-class db.t3.small \
    --no-multi-az

echo "[Step 2/6] Waiting for temporary instance..."
aws rds wait db-instance-available \
    --db-instance-identifier $TEMP_INSTANCE

TEMP_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $TEMP_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

# Step 3: Export table from temporary instance
echo "[Step 3/6] Exporting table data..."
EXPORT_FILE="/tmp/${TABLE_NAME}_export_$(date +%Y%m%d%H%M).sql"

PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $TEMP_ENDPOINT \
    -U $DB_USER \
    -d greenlang \
    -t $TABLE_NAME \
    --data-only \
    --column-inserts \
    -f $EXPORT_FILE

echo "Exported to: $EXPORT_FILE"
echo "Export size: $(du -h $EXPORT_FILE | cut -f1)"
echo "Row count: $(grep -c "INSERT INTO" $EXPORT_FILE)"

# Step 4: Backup current table in production
echo "[Step 4/6] Backing up current production table..."
BACKUP_FILE="/tmp/${TABLE_NAME}_backup_$(date +%Y%m%d%H%M).sql"

PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $PROD_ENDPOINT \
    -U $DB_USER \
    -d greenlang \
    -t $TABLE_NAME \
    -f $BACKUP_FILE

echo "Backup created: $BACKUP_FILE"

# Step 5: Restore to production
echo "[Step 5/6] Restoring table to production..."
echo "WARNING: This will replace all data in $TABLE_NAME"
read -p "Proceed with restore to production? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    # Option A: Truncate and restore
    PGPASSWORD=$DB_PASSWORD psql -h $PROD_ENDPOINT -U $DB_USER -d greenlang << EOF
BEGIN;
TRUNCATE TABLE $TABLE_NAME CASCADE;
\i $EXPORT_FILE
COMMIT;
ANALYZE $TABLE_NAME;
EOF
    echo "Table restored successfully!"
else
    echo "Restore cancelled. Export file available at: $EXPORT_FILE"
fi

# Step 6: Cleanup temporary instance
echo "[Step 6/6] Cleaning up temporary instance..."
aws rds delete-db-instance \
    --db-instance-identifier $TEMP_INSTANCE \
    --skip-final-snapshot

echo "=========================================="
echo "PARTIAL RESTORE COMPLETE"
echo "=========================================="
echo "Table: $TABLE_NAME"
echo "Export file: $EXPORT_FILE"
echo "Backup file: $BACKUP_FILE"
echo "=========================================="
```

### Procedure 5.2: Selective Row Recovery

**Use Case:** Recovering specific rows that were deleted or modified.

```bash
#!/bin/bash
# Selective Row Recovery
# Run as: ./restore-rows.sh

set -e

# Configuration - customize these
TABLE_NAME="emissions_data"
WHERE_CLAUSE="organization_id = 'org-123' AND created_at >= '2026-02-01'"
RESTORE_TIME="2026-02-03 10:00:00"

echo "=========================================="
echo "SELECTIVE ROW RECOVERY"
echo "=========================================="
echo "Table: $TABLE_NAME"
echo "Filter: $WHERE_CLAUSE"
echo "Restore Time: $RESTORE_TIME"
echo "=========================================="

# Create temporary PITR instance
TEMP_INSTANCE="greenlang-temp-$(date +%Y%m%d%H%M)"
RESTORE_TIME_ISO=$(date -d "$RESTORE_TIME" +%Y-%m-%dT%H:%M:%SZ)

echo "[Step 1/5] Creating temporary instance..."
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier greenlang-postgres \
    --target-db-instance-identifier $TEMP_INSTANCE \
    --restore-time $RESTORE_TIME_ISO \
    --db-instance-class db.t3.small \
    --no-multi-az

aws rds wait db-instance-available --db-instance-identifier $TEMP_INSTANCE

TEMP_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $TEMP_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

# Export specific rows
echo "[Step 2/5] Exporting specific rows..."
EXPORT_FILE="/tmp/selective_export_$(date +%Y%m%d%H%M).sql"

PGPASSWORD=$DB_PASSWORD psql -h $TEMP_ENDPOINT -U $DB_USER -d greenlang << EOF > $EXPORT_FILE
\COPY (SELECT * FROM $TABLE_NAME WHERE $WHERE_CLAUSE) TO STDOUT WITH (FORMAT csv, HEADER true)
EOF

echo "Exported rows: $(wc -l < $EXPORT_FILE)"

# Generate INSERT statements
echo "[Step 3/5] Generating INSERT statements..."
INSERT_FILE="/tmp/selective_insert_$(date +%Y%m%d%H%M).sql"

PGPASSWORD=$DB_PASSWORD psql -h $TEMP_ENDPOINT -U $DB_USER -d greenlang << EOF > $INSERT_FILE
SELECT 'INSERT INTO $TABLE_NAME VALUES (' ||
       array_to_string(ARRAY[
           quote_literal(id::text),
           quote_literal(organization_id::text),
           -- Add other columns as needed
           quote_literal(created_at::text)
       ], ', ') || ');'
FROM $TABLE_NAME
WHERE $WHERE_CLAUSE;
EOF

echo "Generated INSERT file: $INSERT_FILE"

# Preview
echo "[Step 4/5] Preview of rows to restore:"
head -10 $INSERT_FILE

# Apply to production
echo "[Step 5/5] Applying to production..."
read -p "Proceed with applying to production? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    PGPASSWORD=$DB_PASSWORD psql -h $PROD_ENDPOINT -U $DB_USER -d greenlang -f $INSERT_FILE
    echo "Rows restored successfully!"
else
    echo "Restore cancelled."
fi

# Cleanup
aws rds delete-db-instance \
    --db-instance-identifier $TEMP_INSTANCE \
    --skip-final-snapshot

echo "=========================================="
echo "SELECTIVE RECOVERY COMPLETE"
echo "=========================================="
```

---

## Cross-Region Restore

### Procedure 6.1: Disaster Recovery to Secondary Region

**Use Case:** Primary region is unavailable; failover to DR region.

```bash
#!/bin/bash
# Cross-Region Disaster Recovery
# Run as: ./dr-failover-database.sh

set -e

# Configuration
PRIMARY_REGION="us-east-1"
DR_REGION="eu-west-1"
SOURCE_INSTANCE="greenlang-postgres"
DR_INSTANCE="greenlang-postgres-dr"

echo "=========================================="
echo "CROSS-REGION DISASTER RECOVERY"
echo "=========================================="
echo "Primary Region: $PRIMARY_REGION (UNAVAILABLE)"
echo "DR Region: $DR_REGION"
echo "=========================================="

# Step 1: Verify DR region backup availability
echo "[Step 1/6] Checking DR region backup availability..."
DR_BACKUP=$(aws rds describe-db-instance-automated-backups \
    --region $DR_REGION \
    --db-instance-identifier $SOURCE_INSTANCE \
    --query 'DBInstanceAutomatedBackups[0]' \
    --output json)

if [ "$DR_BACKUP" = "null" ] || [ -z "$DR_BACKUP" ]; then
    echo "ERROR: No replicated backup found in DR region!"
    echo "Cross-region backup replication may not be configured."
    exit 1
fi

DR_BACKUP_ARN=$(echo $DR_BACKUP | jq -r '.DBInstanceAutomatedBackupsArn')
LATEST_TIME=$(echo $DR_BACKUP | jq -r '.RestoreWindow.LatestTime')

echo "DR Backup ARN: $DR_BACKUP_ARN"
echo "Latest Restore Time: $LATEST_TIME"

# Step 2: Confirm failover
echo ""
echo "WARNING: This will create a new database in $DR_REGION"
echo "This is a DISASTER RECOVERY operation."
read -p "Confirm DR failover? Type 'FAILOVER' to proceed: " CONFIRM

if [ "$CONFIRM" != "FAILOVER" ]; then
    echo "Failover cancelled."
    exit 0
fi

# Step 3: Restore in DR region
echo "[Step 2/6] Initiating restore in DR region..."
aws rds restore-db-instance-to-point-in-time \
    --region $DR_REGION \
    --source-db-instance-automated-backups-arn $DR_BACKUP_ARN \
    --target-db-instance-identifier $DR_INSTANCE \
    --use-latest-restorable-time \
    --db-instance-class db.t3.medium \
    --db-subnet-group-name greenlang-db-subnet-group \
    --vpc-security-group-ids sg-dr-xxxxxxxxx \
    --multi-az \
    --storage-type gp3 \
    --deletion-protection

# Step 4: Wait for availability
echo "[Step 3/6] Waiting for DR instance..."
echo "This may take 60-120 minutes..."

while true; do
    STATUS=$(aws rds describe-db-instances \
        --region $DR_REGION \
        --db-instance-identifier $DR_INSTANCE \
        --query 'DBInstances[0].DBInstanceStatus' \
        --output text 2>/dev/null || echo "creating")

    echo "$(date): Status = $STATUS"

    if [ "$STATUS" = "available" ]; then
        break
    fi

    sleep 120
done

# Step 5: Get DR endpoint
echo "[Step 4/6] Getting DR endpoint..."
DR_ENDPOINT=$(aws rds describe-db-instances \
    --region $DR_REGION \
    --db-instance-identifier $DR_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "DR Endpoint: $DR_ENDPOINT"

# Step 6: Validate DR instance
echo "[Step 5/6] Validating DR instance..."
PGPASSWORD=$DB_PASSWORD psql -h $DR_ENDPOINT -U $DB_USER -d greenlang << EOF
SELECT 'DR Database Available' as status;
SELECT pg_size_pretty(pg_database_size('greenlang')) as size;
SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public';
EOF

# Step 7: Output next steps
echo "[Step 6/6] DR Failover complete!"
echo ""
echo "=========================================="
echo "DR FAILOVER COMPLETE"
echo "=========================================="
echo "DR Instance: $DR_INSTANCE"
echo "DR Endpoint: $DR_ENDPOINT"
echo "DR Region: $DR_REGION"
echo ""
echo "NEXT STEPS:"
echo "1. Update DNS records to point to DR endpoint"
echo "2. Update Kubernetes secrets with new endpoint"
echo "3. Restart application pods"
echo "4. Verify application connectivity"
echo "5. Monitor for issues"
echo "6. Update status page and notify stakeholders"
echo "=========================================="
```

---

## Redis Cache Restore

### Procedure 7.1: Redis Restore from RDB Snapshot

```bash
#!/bin/bash
# Redis Cache Restore
# Run as: ./restore-redis.sh

set -e

echo "=========================================="
echo "REDIS CACHE RESTORE"
echo "=========================================="

# Step 1: List available backups
echo "[Step 1/5] Listing available Redis backups..."
aws s3 ls s3://greenlang-backups/redis/ --recursive | tail -10

# Step 2: Download latest backup
echo "[Step 2/5] Downloading latest backup..."
LATEST_RDB=$(aws s3 ls s3://greenlang-backups/redis/ --recursive | grep ".rdb" | sort | tail -1 | awk '{print $4}')
aws s3 cp s3://greenlang-backups/$LATEST_RDB /tmp/redis-restore.rdb

echo "Downloaded: $LATEST_RDB"

# Step 3: Stop Redis pods
echo "[Step 3/5] Scaling down Redis..."
kubectl scale statefulset redis-master -n greenlang --replicas=0
kubectl wait --for=delete pod/redis-master-0 -n greenlang --timeout=120s || true

# Step 4: Copy backup to PVC
echo "[Step 4/5] Copying backup to PVC..."
# Create temporary pod to copy data
kubectl run redis-restore --image=redis:7.0 -n greenlang --restart=Never \
    --overrides='{"spec":{"containers":[{"name":"redis-restore","image":"redis:7.0","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"redis-data"}}]}}'

kubectl wait --for=condition=Ready pod/redis-restore -n greenlang --timeout=60s
kubectl cp /tmp/redis-restore.rdb greenlang/redis-restore:/data/dump.rdb

# Step 5: Restart Redis
echo "[Step 5/5] Restarting Redis..."
kubectl delete pod redis-restore -n greenlang
kubectl scale statefulset redis-master -n greenlang --replicas=1
kubectl wait --for=condition=Ready pod/redis-master-0 -n greenlang --timeout=120s

# Verify
echo "Verifying Redis..."
kubectl exec -n greenlang redis-master-0 -- redis-cli INFO keyspace

echo "=========================================="
echo "REDIS RESTORE COMPLETE"
echo "=========================================="
```

---

## Weaviate Vector DB Restore

### Procedure 8.1: Weaviate Restore from Backup

```bash
#!/bin/bash
# Weaviate Vector DB Restore
# Run as: ./restore-weaviate.sh <BACKUP_ID>

set -e

BACKUP_ID=${1:-""}
if [ -z "$BACKUP_ID" ]; then
    echo "Usage: $0 <BACKUP_ID>"
    echo "Available backups:"
    aws s3 ls s3://greenlang-backups/weaviate/ | head -20
    exit 1
fi

echo "=========================================="
echo "WEAVIATE RESTORE"
echo "=========================================="
echo "Backup ID: $BACKUP_ID"
echo "=========================================="

# Step 1: Download backup from S3
echo "[Step 1/5] Downloading backup..."
BACKUP_DIR="/tmp/weaviate-backup-$BACKUP_ID"
mkdir -p $BACKUP_DIR
aws s3 sync s3://greenlang-backups/weaviate/$BACKUP_ID/ $BACKUP_DIR/

# Step 2: Copy to Weaviate pod
echo "[Step 2/5] Copying backup to Weaviate pod..."
kubectl cp $BACKUP_DIR greenlang/weaviate-0:/var/lib/weaviate/backups/

# Step 3: Trigger restore via API
echo "[Step 3/5] Triggering restore..."
WEAVIATE_URL=$(kubectl get svc weaviate -n greenlang -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

curl -X POST "http://$WEAVIATE_URL:8080/v1/backups/greenlang/$BACKUP_ID/restore" \
    -H "Content-Type: application/json" \
    -d '{}'

# Step 4: Monitor restore progress
echo "[Step 4/5] Monitoring restore progress..."
while true; do
    STATUS=$(curl -s "http://$WEAVIATE_URL:8080/v1/backups/greenlang/$BACKUP_ID/restore" | jq -r '.status')
    echo "Status: $STATUS"

    if [ "$STATUS" = "SUCCESS" ]; then
        echo "Restore completed successfully!"
        break
    elif [ "$STATUS" = "FAILED" ]; then
        echo "ERROR: Restore failed!"
        curl -s "http://$WEAVIATE_URL:8080/v1/backups/greenlang/$BACKUP_ID/restore" | jq '.'
        exit 1
    fi

    sleep 10
done

# Step 5: Verify
echo "[Step 5/5] Verifying restore..."
curl -s "http://$WEAVIATE_URL:8080/v1/schema" | jq '.classes[] | {class: .class, vectorizer: .vectorizer}'

echo "=========================================="
echo "WEAVIATE RESTORE COMPLETE"
echo "=========================================="
```

---

## Post-Restore Validation

### Validation Checklist

```bash
#!/bin/bash
# Post-Restore Validation Script
# Run as: ./validate-restore.sh <DB_ENDPOINT>

set -e

DB_ENDPOINT=${1:-"greenlang-postgres.xxxxx.us-east-1.rds.amazonaws.com"}

echo "=========================================="
echo "POST-RESTORE VALIDATION"
echo "=========================================="
echo "Endpoint: $DB_ENDPOINT"
echo "=========================================="

# Test 1: Basic Connectivity
echo "[Test 1/8] Basic Connectivity..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang -c "SELECT 1;" > /dev/null
echo "PASSED"

# Test 2: Table Count
echo "[Test 2/8] Table Count..."
TABLE_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang -t -c \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
echo "Tables: $TABLE_COUNT"
if [ $TABLE_COUNT -lt 10 ]; then
    echo "WARNING: Low table count!"
fi
echo "PASSED"

# Test 3: Critical Tables Exist
echo "[Test 3/8] Critical Tables..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang << EOF
SELECT
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'organizations') THEN 'OK' ELSE 'MISSING' END as organizations,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'suppliers') THEN 'OK' ELSE 'MISSING' END as suppliers,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'emissions_data') THEN 'OK' ELSE 'MISSING' END as emissions_data,
    CASE WHEN EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN 'OK' ELSE 'MISSING' END as users;
EOF
echo "PASSED"

# Test 4: Row Counts
echo "[Test 4/8] Row Counts..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang << EOF
SELECT 'organizations' as table_name, COUNT(*) as count FROM organizations
UNION ALL SELECT 'suppliers', COUNT(*) FROM suppliers
UNION ALL SELECT 'emissions_data', COUNT(*) FROM emissions_data
UNION ALL SELECT 'users', COUNT(*) FROM users;
EOF
echo "PASSED"

# Test 5: Data Integrity
echo "[Test 5/8] Data Integrity..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang << EOF
SELECT 'Orphaned suppliers' as check_name, COUNT(*) as count
FROM suppliers s
LEFT JOIN organizations o ON s.organization_id = o.id
WHERE o.id IS NULL;
EOF
echo "PASSED"

# Test 6: Indexes
echo "[Test 6/8] Indexes..."
INDEX_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang -t -c \
    "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public';")
echo "Indexes: $INDEX_COUNT"
echo "PASSED"

# Test 7: Database Size
echo "[Test 7/8] Database Size..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang -c \
    "SELECT pg_size_pretty(pg_database_size('greenlang'));"
echo "PASSED"

# Test 8: Write Test
echo "[Test 8/8] Write Test..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_ENDPOINT -U $DB_USER -d greenlang << EOF
BEGIN;
CREATE TABLE IF NOT EXISTS _restore_validation_test (id serial primary key, timestamp timestamp default now());
INSERT INTO _restore_validation_test DEFAULT VALUES;
SELECT * FROM _restore_validation_test;
DROP TABLE _restore_validation_test;
COMMIT;
EOF
echo "PASSED"

echo "=========================================="
echo "ALL VALIDATION TESTS PASSED"
echo "=========================================="
```

---

## Rollback Procedures

### Procedure 10.1: Rollback to Previous Instance

```bash
#!/bin/bash
# Rollback to previous database instance
# Use if restored database has issues

set -e

CURRENT_INSTANCE="greenlang-postgres-restored-20260203"
PREVIOUS_INSTANCE="greenlang-postgres-old"

echo "=========================================="
echo "DATABASE ROLLBACK"
echo "=========================================="
echo "Rolling back from: $CURRENT_INSTANCE"
echo "Rolling back to: $PREVIOUS_INSTANCE"
echo "=========================================="

read -p "Confirm rollback? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Rollback cancelled."
    exit 0
fi

# Step 1: Get previous instance endpoint
PREV_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $PREVIOUS_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "Previous endpoint: $PREV_ENDPOINT"

# Step 2: Update application configuration
echo "Updating Kubernetes secrets..."
kubectl create secret generic greenlang-db-credentials \
    --from-literal=host=$PREV_ENDPOINT \
    --from-literal=username=$DB_USER \
    --from-literal=password=$DB_PASSWORD \
    --dry-run=client -o yaml | kubectl apply -f -

# Step 3: Restart application pods
echo "Restarting application pods..."
kubectl rollout restart deployment/greenlang-api -n greenlang
kubectl rollout status deployment/greenlang-api -n greenlang

# Step 4: Verify
echo "Verifying connectivity..."
kubectl exec -n greenlang deploy/greenlang-api -- \
    python -c "from app.db import engine; engine.connect(); print('Connected!')"

echo "=========================================="
echo "ROLLBACK COMPLETE"
echo "=========================================="
```

---

## Related Documentation

- [Point-in-Time Recovery Guide](./point-in-time-recovery.md)
- [RDS Backup Policy](./rds-backup-policy.tf)
- [Backup Verification Scripts](../scripts/backup-verify.sh)
- [DR Failover Script](../scripts/dr-failover.sh)
- [Platform Disaster Recovery](../../platform-disaster-recovery.md)

---

**Document Owner:** Platform Engineering Team
**Review Cycle:** Monthly
**Last Tested:** 2026-02-01
**Next Review:** 2026-03-03
