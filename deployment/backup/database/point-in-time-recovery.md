# Point-in-Time Recovery (PITR) Documentation

**INFRA-001: Database Disaster Recovery**
**Version:** 1.0.0
**Last Updated:** 2026-02-03

---

## Table of Contents

1. [Overview](#overview)
2. [PITR Capabilities](#pitr-capabilities)
3. [Recovery Scenarios](#recovery-scenarios)
4. [Step-by-Step Recovery Procedures](#step-by-step-recovery-procedures)
5. [Recovery Time Estimates](#recovery-time-estimates)
6. [Testing and Validation](#testing-and-validation)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Point-in-Time Recovery (PITR) allows restoration of the GreenLang PostgreSQL database to any point within the backup retention window. This capability is critical for recovering from:

- Accidental data deletion
- Data corruption
- Application bugs that modify data incorrectly
- Security incidents requiring data rollback

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Retention Period** | 35 days | Maximum recovery window |
| **RPO** | 5 minutes | Recovery Point Objective |
| **RTO** | 1-2 hours | Recovery Time Objective |
| **Backup Window** | 03:00-04:00 UTC | Daily automated backup |
| **WAL Archive Frequency** | 5 minutes | Transaction log archival |

---

## PITR Capabilities

### AWS RDS PostgreSQL PITR

AWS RDS automatically enables PITR when backup retention is configured. The system:

1. Takes daily automated snapshots during the backup window
2. Continuously archives transaction logs (WAL) to S3
3. Retains both snapshots and logs for the configured retention period
4. Allows restoration to any second within the retention window

### Recovery Points

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Recovery Window (35 Days)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Day 1         Day 7        Day 14        Day 21        Day 28   Day 35 │
│    │             │            │             │             │         │    │
│    ▼             ▼            ▼             ▼             ▼         ▼    │
│  [Snapshot]  [Snapshot]  [Snapshot]  [Snapshot]  [Snapshot]  [Snapshot] │
│    │             │            │             │             │         │    │
│    └─────────────┴────────────┴─────────────┴─────────────┴─────────┘   │
│                    Continuous WAL Archive (5-min intervals)              │
│                                                                          │
│  ◄───────────────────── Can restore to ANY point ─────────────────────► │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Recovery Scenarios

### Scenario 1: Accidental Table Deletion

**Situation:** A developer accidentally ran `DROP TABLE suppliers;` in production.

**Recovery Strategy:**
1. Identify the exact time before the deletion occurred
2. Create a new RDS instance restored to that point
3. Export the deleted table from the restored instance
4. Import the table into production

**Estimated Recovery Time:** 30-60 minutes

### Scenario 2: Data Corruption from Application Bug

**Situation:** A bug in the emissions calculation module corrupted 2 weeks of emissions data.

**Recovery Strategy:**
1. Identify when the bug was deployed
2. Create a restored instance to the point before deployment
3. Extract the corrupted data ranges
4. Apply selective data restoration

**Estimated Recovery Time:** 1-2 hours

### Scenario 3: Complete Database Failure

**Situation:** The primary RDS instance is unrecoverable.

**Recovery Strategy:**
1. Restore to the latest possible point (5 minutes ago)
2. Update application connection strings
3. Verify data integrity
4. Resume operations

**Estimated Recovery Time:** 1-2 hours

### Scenario 4: Ransomware/Security Incident

**Situation:** Malicious actors encrypted or deleted data.

**Recovery Strategy:**
1. Isolate the compromised environment
2. Identify the last known good state (before compromise)
3. Restore to an air-gapped/isolated environment first
4. Validate data integrity
5. Restore to production after security clearance

**Estimated Recovery Time:** 4-8 hours

---

## Step-by-Step Recovery Procedures

### Procedure A: AWS Console PITR Restore

```
Step 1: Navigate to RDS Console
─────────────────────────────────
1. Open AWS Console
2. Navigate to RDS > Databases
3. Select "greenlang-postgres" instance
4. Click "Actions" > "Restore to point in time"

Step 2: Configure Restoration
─────────────────────────────────
1. Select "Custom date and time"
2. Enter target date and time (UTC)
   Example: 2026-02-03 10:30:00 UTC
3. Specify new DB instance identifier:
   Example: greenlang-postgres-pitr-20260203

Step 3: Configure Instance Settings
─────────────────────────────────
1. DB instance class: Same as source (db.t3.medium)
2. Multi-AZ: Enable for production
3. Storage type: gp3
4. VPC: Same as source
5. Subnet group: Same as source
6. Security groups: Same as source

Step 4: Launch Restoration
─────────────────────────────────
1. Review all settings
2. Click "Restore to point in time"
3. Monitor progress in RDS console
4. Wait for status: "Available"
```

### Procedure B: AWS CLI PITR Restore

```bash
#!/bin/bash
# PITR Restore using AWS CLI
# INFRA-001: Database Recovery Script

# Configuration
SOURCE_INSTANCE="greenlang-postgres"
RESTORE_TIME="2026-02-03T10:30:00Z"  # ISO 8601 format (UTC)
TARGET_INSTANCE="greenlang-postgres-pitr-$(date +%Y%m%d%H%M)"
DB_SUBNET_GROUP="greenlang-db-subnet-group"
SECURITY_GROUP_IDS="sg-xxxxxxxxx"
INSTANCE_CLASS="db.t3.medium"

# Validate restore time is within retention window
echo "Validating restore point..."
EARLIEST_RESTORE=$(aws rds describe-db-instances \
    --db-instance-identifier $SOURCE_INSTANCE \
    --query 'DBInstances[0].LatestRestorableTime' \
    --output text)

echo "Latest restorable time: $EARLIEST_RESTORE"
echo "Requested restore time: $RESTORE_TIME"

# Perform PITR restore
echo "Starting Point-in-Time Recovery..."
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier $SOURCE_INSTANCE \
    --target-db-instance-identifier $TARGET_INSTANCE \
    --restore-time $RESTORE_TIME \
    --db-instance-class $INSTANCE_CLASS \
    --db-subnet-group-name $DB_SUBNET_GROUP \
    --vpc-security-group-ids $SECURITY_GROUP_IDS \
    --multi-az \
    --storage-type gp3 \
    --copy-tags-to-snapshot \
    --enable-cloudwatch-logs-exports '["postgresql","upgrade"]' \
    --deletion-protection \
    --tags Key=Environment,Value=recovery Key=Purpose,Value=pitr-restore

echo "Restore initiated. Monitoring progress..."

# Wait for instance to be available
aws rds wait db-instance-available \
    --db-instance-identifier $TARGET_INSTANCE

echo "PITR restore complete!"
echo "New instance: $TARGET_INSTANCE"

# Get endpoint
ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $TARGET_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "Endpoint: $ENDPOINT"
```

### Procedure C: Selective Data Recovery

For recovering specific tables or data ranges without full database restore:

```bash
#!/bin/bash
# Selective Data Recovery from PITR Instance
# INFRA-001: Granular Recovery Script

# Configuration
PITR_INSTANCE="greenlang-postgres-pitr-20260203"
PRODUCTION_INSTANCE="greenlang-postgres"
DATABASE="greenlang"
TABLE_TO_RECOVER="suppliers"
RECOVERY_FILE="/tmp/recovered_data.sql"

# Step 1: Get PITR instance endpoint
PITR_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $PITR_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "PITR Instance Endpoint: $PITR_ENDPOINT"

# Step 2: Export table from PITR instance
echo "Exporting table: $TABLE_TO_RECOVER"
PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $PITR_ENDPOINT \
    -U $DB_USER \
    -d $DATABASE \
    -t $TABLE_TO_RECOVER \
    --data-only \
    -f $RECOVERY_FILE

echo "Data exported to: $RECOVERY_FILE"

# Step 3: Verify exported data
echo "Verifying exported data..."
wc -l $RECOVERY_FILE
head -20 $RECOVERY_FILE

# Step 4: Import to production (with confirmation)
read -p "Proceed with import to production? (yes/no): " CONFIRM
if [ "$CONFIRM" = "yes" ]; then
    echo "Importing data to production..."

    # Option A: Direct import (if table is empty)
    # PGPASSWORD=$PROD_PASSWORD psql -h $PROD_ENDPOINT -U $DB_USER -d $DATABASE -f $RECOVERY_FILE

    # Option B: Merge with existing data
    # Create temp table, import, then merge

    echo "Import complete."
else
    echo "Import cancelled."
fi

# Step 5: Cleanup
echo "Cleaning up recovery artifacts..."
rm -f $RECOVERY_FILE
```

### Procedure D: Cross-Region PITR Recovery

```bash
#!/bin/bash
# Cross-Region PITR Recovery
# INFRA-001: Disaster Recovery to DR Region

# Configuration
SOURCE_REGION="us-east-1"
DR_REGION="eu-west-1"
SOURCE_INSTANCE="greenlang-postgres"
DR_INSTANCE="greenlang-postgres-dr-$(date +%Y%m%d)"
RESTORE_TIME="2026-02-03T10:30:00Z"

# Step 1: Get the replicated backup ARN in DR region
echo "Finding replicated backup in $DR_REGION..."
REPLICATED_ARN=$(aws rds describe-db-instance-automated-backups \
    --region $DR_REGION \
    --db-instance-identifier $SOURCE_INSTANCE \
    --query 'DBInstanceAutomatedBackups[0].DBInstanceAutomatedBackupsArn' \
    --output text)

echo "Replicated backup ARN: $REPLICATED_ARN"

# Step 2: Restore in DR region
echo "Restoring in $DR_REGION..."
aws rds restore-db-instance-to-point-in-time \
    --region $DR_REGION \
    --source-db-instance-automated-backups-arn $REPLICATED_ARN \
    --target-db-instance-identifier $DR_INSTANCE \
    --restore-time $RESTORE_TIME \
    --db-instance-class db.t3.medium \
    --db-subnet-group-name greenlang-db-subnet-group-dr \
    --vpc-security-group-ids sg-dr-xxxxxxxxx \
    --multi-az \
    --storage-type gp3

# Step 3: Monitor restoration
echo "Monitoring restoration progress..."
aws rds wait db-instance-available \
    --region $DR_REGION \
    --db-instance-identifier $DR_INSTANCE

echo "Cross-region PITR recovery complete!"
```

---

## Recovery Time Estimates

| Database Size | Restore Type | Estimated Time |
|---------------|--------------|----------------|
| < 10 GB | Full PITR | 15-30 minutes |
| 10-50 GB | Full PITR | 30-60 minutes |
| 50-200 GB | Full PITR | 1-2 hours |
| 200-500 GB | Full PITR | 2-4 hours |
| > 500 GB | Full PITR | 4-8 hours |

**Factors Affecting Recovery Time:**
- Database size
- Instance class selected
- I/O throughput
- Number of WAL files to replay
- Network conditions (cross-region)

---

## Testing and Validation

### Monthly PITR Test Procedure

```bash
#!/bin/bash
# Monthly PITR Test
# Schedule: First Saturday of each month

TEST_TIMESTAMP=$(date -d "1 hour ago" +%Y-%m-%dT%H:%M:%SZ)
TEST_INSTANCE="greenlang-postgres-test-$(date +%Y%m%d)"

echo "=== PITR TEST STARTED ==="
echo "Test Timestamp: $TEST_TIMESTAMP"
echo "Test Instance: $TEST_INSTANCE"

# 1. Create PITR restore
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier greenlang-postgres \
    --target-db-instance-identifier $TEST_INSTANCE \
    --restore-time $TEST_TIMESTAMP \
    --db-instance-class db.t3.small

# 2. Wait for availability
aws rds wait db-instance-available \
    --db-instance-identifier $TEST_INSTANCE

# 3. Run validation queries
ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $TEST_INSTANCE \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

echo "Running validation queries..."
PGPASSWORD=$TEST_PASSWORD psql -h $ENDPOINT -U $DB_USER -d greenlang << EOF
-- Validate table counts
SELECT 'organizations' as table_name, COUNT(*) as count FROM organizations;
SELECT 'suppliers' as table_name, COUNT(*) as count FROM suppliers;
SELECT 'emissions_data' as table_name, COUNT(*) as count FROM emissions_data;

-- Validate data integrity
SELECT 'orphaned records' as check_name, COUNT(*) as count
FROM suppliers s
LEFT JOIN organizations o ON s.organization_id = o.id
WHERE o.id IS NULL;

-- Validate recent data
SELECT 'recent_emissions' as check_name, COUNT(*) as count
FROM emissions_data
WHERE created_at > NOW() - INTERVAL '7 days';
EOF

# 4. Cleanup test instance
echo "Cleaning up test instance..."
aws rds delete-db-instance \
    --db-instance-identifier $TEST_INSTANCE \
    --skip-final-snapshot

echo "=== PITR TEST COMPLETED ==="
```

### Validation Checklist

```markdown
# PITR Validation Checklist

## Pre-Recovery
- [ ] Identified exact recovery point needed
- [ ] Verified recovery point is within retention window
- [ ] Documented current production state
- [ ] Notified stakeholders of recovery operation
- [ ] Prepared rollback plan

## During Recovery
- [ ] Restore initiated successfully
- [ ] Instance status progressing normally
- [ ] No errors in RDS events

## Post-Recovery Validation
- [ ] Instance is available
- [ ] Can connect to database
- [ ] All tables present
- [ ] Row counts match expectations
- [ ] No data corruption detected
- [ ] Application can connect
- [ ] Read/write operations working
- [ ] Performance is acceptable

## Cleanup
- [ ] Old instances terminated (if applicable)
- [ ] DNS/endpoints updated
- [ ] Documentation updated
- [ ] Incident report filed
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Cannot restore to specified time"

**Error:** `The specified restore time is not valid`

**Cause:** Requested time is outside the retention window or in the future.

**Solution:**
```bash
# Check valid restore window
aws rds describe-db-instances \
    --db-instance-identifier greenlang-postgres \
    --query 'DBInstances[0].[EarliestRestorableTime,LatestRestorableTime]'
```

#### Issue 2: "Insufficient capacity"

**Error:** `InsufficientDBInstanceCapacity`

**Cause:** No capacity available for the requested instance class in the AZ.

**Solution:**
```bash
# Try a different instance class or AZ
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier greenlang-postgres \
    --target-db-instance-identifier greenlang-postgres-pitr \
    --restore-time 2026-02-03T10:30:00Z \
    --db-instance-class db.t3.large \  # Different class
    --availability-zone us-east-1b      # Different AZ
```

#### Issue 3: "Restore taking too long"

**Cause:** Large database size or many WAL files to replay.

**Solution:**
1. Monitor progress in CloudWatch metrics
2. Check RDS events for any issues
3. Consider using a larger instance class for faster I/O

```bash
# Monitor restore progress
aws rds describe-events \
    --source-identifier greenlang-postgres-pitr \
    --source-type db-instance \
    --duration 60
```

#### Issue 4: "Data missing after restore"

**Cause:** Restored to wrong point in time or WAL replay incomplete.

**Solution:**
1. Verify the restore time was correct
2. Check if restore completed successfully
3. Attempt restore to a slightly earlier point

```bash
# Verify restoration details
aws rds describe-db-instances \
    --db-instance-identifier greenlang-postgres-pitr \
    --query 'DBInstances[0].InstanceCreateTime'
```

---

## Related Documentation

- [Restore Procedures Runbook](./restore-procedures.md)
- [Platform Disaster Recovery Strategy](../../platform-disaster-recovery.md)
- [Backup Scripts](../scripts/)

---

**Document Owner:** Platform Engineering Team
**Review Cycle:** Quarterly
**Next Review:** 2026-05-03
