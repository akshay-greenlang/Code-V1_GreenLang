# GreenLang Backup and Restore Procedures

**Version:** 1.0
**Last Updated:** 2025-11-07
**Document Classification:** CRITICAL - Operations
**Review Cycle:** Monthly
**Next Review:** 2025-12-07

---

## Executive Summary

This document provides comprehensive backup and restore procedures for the GreenLang production environment. It ensures business continuity through reliable data protection and recovery capabilities.

**Backup Philosophy:**
- **3-2-1 Rule:** 3 copies, 2 different media, 1 offsite
- **Automated:** All backups fully automated
- **Tested:** Regular restore testing validates backup integrity
- **Encrypted:** All backups encrypted at rest and in transit
- **Versioned:** Multiple restore points available

**Recovery Objectives:**
- **RPO (Recovery Point Objective):** 1 hour maximum data loss
- **RTO (Recovery Time Objective):** 4 hours to full recovery
- **Backup Retention:** 30 days online, 1 year archived

---

## Table of Contents

1. [What to Backup](#what-to-backup)
2. [Backup Schedules](#backup-schedules)
3. [Backup Procedures](#backup-procedures)
4. [Restore Procedures](#restore-procedures)
5. [Testing Procedures](#testing-procedures)
6. [Monitoring and Validation](#monitoring-and-validation)

---

## What to Backup

### 1. Database (PostgreSQL)

**Priority:** CRITICAL
**Frequency:** Continuous + Daily Full
**Retention:** 30 days

**Components:**
- Database schema
- All tables and data
- Users and permissions
- Stored procedures and functions
- Database configuration

**Backup Method:**
- Continuous WAL archiving
- Daily full backups
- Automated via pg_dump and WAL-E/WAL-G

---

### 2. Configuration Files

**Priority:** CRITICAL
**Frequency:** On change + Daily
**Retention:** 90 days

**Components:**
```
config/
├── production.yaml          # Application configuration
├── agents/                  # Agent configurations
├── secrets/                 # Encrypted secrets
├── kubernetes/             # K8s manifests
│   ├── deployments/
│   ├── services/
│   └── configmaps/
└── infrastructure/         # Terraform/CloudFormation
```

**Backup Method:**
- Git repository (primary)
- S3 versioned bucket (secondary)
- Config management system snapshots

---

### 3. Agent Pack Files

**Priority:** HIGH
**Frequency:** On change + Daily
**Retention:** 90 days

**Components:**
```
packs/
├── calculator.yml
├── data-processor.yml
├── carbon-analyzer.yml
└── ...
```

**Backup Method:**
- Git repository
- S3 bucket with versioning
- Artifact repository

---

### 4. Snapshots and Baselines

**Priority:** HIGH
**Frequency:** Weekly
**Retention:** 30 days

**Components:**
- System snapshots
- Performance baselines
- Configuration baselines
- Test data sets

**Backup Method:**
- EBS snapshots
- S3 bucket storage

---

### 5. Logs

**Priority:** MEDIUM
**Frequency:** Real-time
**Retention:** 90 days online, 1 year cold storage

**Components:**
- Application logs
- System logs
- Audit logs
- Access logs

**Backup Method:**
- Loki/Elasticsearch (online)
- S3 bucket (cold storage)
- Compressed and encrypted

---

### 6. Metrics and Monitoring Data

**Priority:** LOW
**Frequency:** Continuous
**Retention:** 30 days high-res, 1 year aggregated

**Components:**
- Prometheus metrics
- Grafana dashboards
- Alert history

**Backup Method:**
- Prometheus TSDB snapshots
- S3 bucket storage
- Grafana configuration in Git

---

## Backup Schedules

### Schedule Overview

| Backup Type | Frequency | Time (UTC) | Duration | Retention |
|-------------|-----------|------------|----------|-----------|
| **Database WAL** | Continuous | N/A | Real-time | 30 days |
| **Database Full** | Daily | 02:00 | ~30 min | 30 days |
| **Database Weekly** | Weekly | Sun 03:00 | ~45 min | 90 days |
| **Database Monthly** | Monthly | 1st Sun 04:00 | ~60 min | 1 year |
| **Configuration** | Daily | 01:00 | ~5 min | 90 days |
| **Agent Packs** | Daily | 01:30 | ~5 min | 90 days |
| **EBS Snapshots** | Weekly | Sun 00:00 | ~15 min | 30 days |
| **Logs Archive** | Daily | 05:00 | ~20 min | 1 year |

### Backup Windows

**Primary Window:** 01:00-05:00 UTC (low traffic period)
**Impact:** No service downtime, minimal performance impact (<5%)

---

## Backup Procedures

### BACKUP-001: Database Full Backup

**Objective:** Create complete database backup

**Frequency:** Daily at 02:00 UTC

**Automated Script:** `/opt/greenlang/scripts/backup-database-full.sh`

**Manual Procedure:**

```bash
#!/bin/bash
# Database Full Backup Procedure

# Configuration
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/backups/database"
S3_BUCKET="s3://greenlang-backups/database"
DB_HOST="db.greenlang.io"
DB_NAME="greenlang"
DB_USER="backup_user"

# Create backup directory
mkdir -p $BACKUP_DIR/$TIMESTAMP

# Step 1: Create database dump
echo "Starting database backup..."
pg_dump \
  -h $DB_HOST \
  -U $DB_USER \
  -d $DB_NAME \
  -F c \
  -b \
  -v \
  -f $BACKUP_DIR/$TIMESTAMP/greenlang.dump

# Step 2: Verify backup integrity
echo "Verifying backup..."
pg_restore --list $BACKUP_DIR/$TIMESTAMP/greenlang.dump > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Backup verification failed!"
    exit 1
fi

# Step 3: Create metadata
cat > $BACKUP_DIR/$TIMESTAMP/manifest.json << EOF
{
  "timestamp": "$TIMESTAMP",
  "database": "$DB_NAME",
  "size_bytes": $(stat -f%z $BACKUP_DIR/$TIMESTAMP/greenlang.dump),
  "backup_type": "full",
  "compression": "custom",
  "checksum": "$(sha256sum $BACKUP_DIR/$TIMESTAMP/greenlang.dump | awk '{print $1}')"
}
EOF

# Step 4: Compress backup
echo "Compressing backup..."
tar -czf $BACKUP_DIR/$TIMESTAMP/backup.tar.gz \
  -C $BACKUP_DIR/$TIMESTAMP \
  greenlang.dump manifest.json

# Step 5: Encrypt backup
echo "Encrypting backup..."
openssl enc -aes-256-cbc \
  -in $BACKUP_DIR/$TIMESTAMP/backup.tar.gz \
  -out $BACKUP_DIR/$TIMESTAMP/backup.tar.gz.enc \
  -pass file:/etc/greenlang/backup-key.txt

# Step 6: Upload to S3
echo "Uploading to S3..."
aws s3 cp $BACKUP_DIR/$TIMESTAMP/backup.tar.gz.enc \
  $S3_BUCKET/$TIMESTAMP/backup.tar.gz.enc \
  --storage-class STANDARD_IA

# Step 7: Upload metadata
aws s3 cp $BACKUP_DIR/$TIMESTAMP/manifest.json \
  $S3_BUCKET/$TIMESTAMP/manifest.json

# Step 8: Verify S3 upload
aws s3 ls $S3_BUCKET/$TIMESTAMP/

if [ $? -ne 0 ]; then
    echo "ERROR: S3 upload verification failed!"
    exit 1
fi

# Step 9: Clean up local files older than 7 days
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} +

# Step 10: Log success
echo "Backup completed successfully: $TIMESTAMP"
echo "Backup size: $(du -sh $BACKUP_DIR/$TIMESTAMP | cut -f1)"

# Send notification
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d "{\"text\":\"Database backup completed: $TIMESTAMP\"}"
```

**Validation:**
```bash
# Verify backup exists
aws s3 ls s3://greenlang-backups/database/$TIMESTAMP/

# Check backup size (should be >100MB)
aws s3 ls --summarize --human-readable \
  s3://greenlang-backups/database/$TIMESTAMP/
```

---

### BACKUP-002: Database WAL Archiving

**Objective:** Continuous backup of Write-Ahead Logs (WAL) for point-in-time recovery

**Frequency:** Continuous (as WAL segments complete)

**Configuration:**

```bash
# PostgreSQL configuration: /etc/postgresql/14/main/postgresql.conf

# Enable WAL archiving
wal_level = replica
archive_mode = on
archive_command = 'wal-g wal-push %p'
archive_timeout = 300  # Archive every 5 minutes

# WAL configuration
max_wal_size = 2GB
min_wal_size = 1GB
wal_keep_size = 1GB
```

**WAL-G Configuration:**

```bash
# /etc/wal-g.d/server.conf

AWS_REGION=us-east-1
WALG_S3_PREFIX=s3://greenlang-backups/wal
WALG_COMPRESSION_METHOD=lz4
WALG_DELTA_MAX_STEPS=5
AWS_ACCESS_KEY_ID=<from-secrets>
AWS_SECRET_ACCESS_KEY=<from-secrets>
```

**Monitoring:**
```bash
# Check WAL archiving status
psql -c "SELECT * FROM pg_stat_archiver;"

# Should show:
# - archived_count increasing
# - last_archived_time recent (<5 minutes ago)
# - failed_count = 0
```

---

### BACKUP-003: Configuration Backup

**Objective:** Backup all configuration files

**Frequency:** Daily at 01:00 UTC + On every change

**Automated Script:** `/opt/greenlang/scripts/backup-config.sh`

**Manual Procedure:**

```bash
#!/bin/bash
# Configuration Backup Procedure

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/backups/config"
S3_BUCKET="s3://greenlang-backups/config"

# Step 1: Create backup directory
mkdir -p $BACKUP_DIR/$TIMESTAMP

# Step 2: Backup Kubernetes configurations
kubectl get all --all-namespaces -o yaml > $BACKUP_DIR/$TIMESTAMP/k8s-all.yaml
kubectl get configmaps --all-namespaces -o yaml > $BACKUP_DIR/$TIMESTAMP/k8s-configmaps.yaml
kubectl get secrets --all-namespaces -o yaml > $BACKUP_DIR/$TIMESTAMP/k8s-secrets.yaml

# Step 3: Backup application configs
cp -r /opt/greenlang/config $BACKUP_DIR/$TIMESTAMP/

# Step 4: Backup agent packs
cp -r /opt/greenlang/packs $BACKUP_DIR/$TIMESTAMP/

# Step 5: Create tarball
tar -czf $BACKUP_DIR/$TIMESTAMP.tar.gz -C $BACKUP_DIR/$TIMESTAMP .

# Step 6: Upload to S3
aws s3 cp $BACKUP_DIR/$TIMESTAMP.tar.gz \
  $S3_BUCKET/$TIMESTAMP.tar.gz

# Step 7: Clean up
rm -rf $BACKUP_DIR/$TIMESTAMP

echo "Configuration backup completed: $TIMESTAMP"
```

---

### BACKUP-004: EBS Volume Snapshots

**Objective:** Create snapshots of EBS volumes

**Frequency:** Weekly on Sunday at 00:00 UTC

**Automated Script:** `/opt/greenlang/scripts/backup-ebs.sh`

**Manual Procedure:**

```bash
#!/bin/bash
# EBS Snapshot Procedure

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Get all EBS volumes tagged with Backup=true
VOLUMES=$(aws ec2 describe-volumes \
  --filters "Name=tag:Backup,Values=true" \
  --query 'Volumes[*].VolumeId' \
  --output text)

for VOLUME_ID in $VOLUMES; do
    echo "Creating snapshot for $VOLUME_ID..."

    # Create snapshot
    SNAPSHOT_ID=$(aws ec2 create-snapshot \
      --volume-id $VOLUME_ID \
      --description "Backup-$TIMESTAMP" \
      --tag-specifications "ResourceType=snapshot,Tags=[{Key=Name,Value=backup-$TIMESTAMP},{Key=Volume,Value=$VOLUME_ID}]" \
      --query 'SnapshotId' \
      --output text)

    echo "Snapshot created: $SNAPSHOT_ID"
done

# Delete snapshots older than 30 days
aws ec2 describe-snapshots --owner-ids self \
  --filters "Name=tag:Name,Values=backup-*" \
  --query "Snapshots[?StartTime<='$(date -d '30 days ago' --iso-8601)'].SnapshotId" \
  --output text | \
  xargs -I {} aws ec2 delete-snapshot --snapshot-id {}

echo "EBS snapshots completed: $TIMESTAMP"
```

---

## Restore Procedures

### RESTORE-001: Database Full Restore

**Objective:** Restore database from full backup

**Estimated Time:** 2-3 hours

**Prerequisites:**
- Backup file available
- Database server accessible
- Sufficient disk space

**Procedure:**

```bash
#!/bin/bash
# Database Full Restore Procedure

# Configuration
RESTORE_POINT="20251107-020000"  # Specify backup to restore
BACKUP_DIR="/restore/database"
S3_BUCKET="s3://greenlang-backups/database"
DB_HOST="db.greenlang.io"
DB_NAME="greenlang"

# Step 1: Stop application services
echo "Stopping application services..."
kubectl scale deployment greenlang-api --replicas=0
kubectl scale deployment greenlang-worker --replicas=0

# Wait for graceful shutdown
sleep 30

# Step 2: Verify no active connections
psql -h $DB_HOST -U postgres -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();
"

# Step 3: Download backup from S3
echo "Downloading backup..."
mkdir -p $BACKUP_DIR/$RESTORE_POINT
aws s3 cp $S3_BUCKET/$RESTORE_POINT/backup.tar.gz.enc \
  $BACKUP_DIR/$RESTORE_POINT/

# Step 4: Decrypt backup
echo "Decrypting backup..."
openssl enc -aes-256-cbc -d \
  -in $BACKUP_DIR/$RESTORE_POINT/backup.tar.gz.enc \
  -out $BACKUP_DIR/$RESTORE_POINT/backup.tar.gz \
  -pass file:/etc/greenlang/backup-key.txt

# Step 5: Extract backup
echo "Extracting backup..."
tar -xzf $BACKUP_DIR/$RESTORE_POINT/backup.tar.gz \
  -C $BACKUP_DIR/$RESTORE_POINT/

# Step 6: Verify backup integrity
echo "Verifying backup..."
pg_restore --list $BACKUP_DIR/$RESTORE_POINT/greenlang.dump

if [ $? -ne 0 ]; then
    echo "ERROR: Backup file corrupted!"
    exit 1
fi

# Step 7: Drop existing database (CAUTION!)
echo "WARNING: About to drop existing database!"
echo "Press Ctrl+C within 10 seconds to abort..."
sleep 10

psql -h $DB_HOST -U postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"
psql -h $DB_HOST -U postgres -c "CREATE DATABASE $DB_NAME;"

# Step 8: Restore database
echo "Restoring database (this may take 30-60 minutes)..."
pg_restore \
  -h $DB_HOST \
  -U postgres \
  -d $DB_NAME \
  -v \
  --no-owner \
  --no-privileges \
  $BACKUP_DIR/$RESTORE_POINT/greenlang.dump

if [ $? -ne 0 ]; then
    echo "ERROR: Database restore failed!"
    exit 1
fi

# Step 9: Verify database
echo "Verifying database..."
psql -h $DB_HOST -U postgres -d $DB_NAME -c "
  SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
  FROM pg_tables
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
  LIMIT 10;
"

# Step 10: Run VACUUM and ANALYZE
echo "Running VACUUM and ANALYZE..."
vacuumdb -h $DB_HOST -U postgres -d $DB_NAME --analyze

# Step 11: Restart application services
echo "Restarting application services..."
kubectl scale deployment greenlang-api --replicas=6
kubectl scale deployment greenlang-worker --replicas=4

# Step 12: Wait for services to be ready
kubectl wait --for=condition=ready pod \
  -l app=greenlang-api --timeout=300s

# Step 13: Validate application
echo "Validating application..."
curl https://api.greenlang.io/health

# Step 14: Run smoke tests
pytest tests/smoke/ --env=production

echo "Database restore completed successfully!"
echo "Restored from backup: $RESTORE_POINT"
```

**Validation Checklist:**
- [ ] Database accessible
- [ ] All tables present
- [ ] Row counts match expected
- [ ] Application services running
- [ ] Smoke tests passing
- [ ] No errors in logs

---

### RESTORE-002: Point-in-Time Recovery (PITR)

**Objective:** Restore database to specific point in time using WAL

**Estimated Time:** 3-4 hours

**Procedure:**

```bash
#!/bin/bash
# Point-in-Time Recovery Procedure

# Configuration
BASE_BACKUP="20251107-020000"
RECOVERY_TARGET="2025-11-07 14:30:00"  # Target time to restore to
RESTORE_DIR="/restore/pitr"

# Step 1: Stop services (same as RESTORE-001)

# Step 2: Download base backup
echo "Downloading base backup..."
mkdir -p $RESTORE_DIR
aws s3 cp s3://greenlang-backups/database/$BASE_BACKUP/backup.tar.gz.enc \
  $RESTORE_DIR/

# Decrypt and extract (same as RESTORE-001)

# Step 3: Clear PostgreSQL data directory
systemctl stop postgresql-14
rm -rf /var/lib/postgresql/14/main/*

# Step 4: Restore base backup
tar -xzf $RESTORE_DIR/greenlang.dump \
  -C /var/lib/postgresql/14/main/

# Step 5: Create recovery configuration
cat > /var/lib/postgresql/14/main/recovery.conf << EOF
# Recovery configuration
restore_command = 'wal-g wal-fetch %f %p'
recovery_target_time = '$RECOVERY_TARGET'
recovery_target_action = 'promote'
EOF

# Step 6: Start PostgreSQL in recovery mode
echo "Starting PostgreSQL in recovery mode..."
systemctl start postgresql-14

# Step 7: Monitor recovery progress
echo "Monitoring recovery (this may take 1-2 hours)..."
tail -f /var/log/postgresql/postgresql-14-main.log | \
  grep -E "recovery|restore|replay"

# Wait for "database system is ready to accept connections"

# Step 8: Verify recovery target
psql -U postgres -c "SELECT pg_last_xact_replay_timestamp();"

# Should be close to $RECOVERY_TARGET

# Step 9: Continue with validation (same as RESTORE-001)
```

---

### RESTORE-003: Configuration Restore

**Objective:** Restore configuration files

**Estimated Time:** 15-30 minutes

**Procedure:**

```bash
#!/bin/bash
# Configuration Restore Procedure

RESTORE_POINT="20251107-010000"
S3_BUCKET="s3://greenlang-backups/config"
RESTORE_DIR="/restore/config"

# Step 1: Download configuration backup
mkdir -p $RESTORE_DIR
aws s3 cp $S3_BUCKET/$RESTORE_POINT.tar.gz $RESTORE_DIR/

# Step 2: Extract
tar -xzf $RESTORE_DIR/$RESTORE_POINT.tar.gz -C $RESTORE_DIR/

# Step 3: Restore Kubernetes resources
kubectl apply -f $RESTORE_DIR/k8s-configmaps.yaml
kubectl apply -f $RESTORE_DIR/k8s-secrets.yaml

# Step 4: Restore application configs
cp -r $RESTORE_DIR/config/* /opt/greenlang/config/

# Step 5: Restore agent packs
cp -r $RESTORE_DIR/packs/* /opt/greenlang/packs/

# Step 6: Restart services to pick up new config
kubectl rollout restart deployment greenlang-api
kubectl rollout restart deployment greenlang-worker

echo "Configuration restored successfully!"
```

---

## Testing Procedures

### TEST-001: Monthly Backup Restore Test

**Objective:** Validate backup integrity and restore procedures

**Frequency:** Monthly (first Sunday)

**Procedure:**

```bash
#!/bin/bash
# Monthly Backup Restore Test

# Test environment: staging
ENVIRONMENT="staging"
LATEST_BACKUP=$(aws s3 ls s3://greenlang-backups/database/ | tail -1 | awk '{print $2}')

echo "Testing backup restore: $LATEST_BACKUP"

# Step 1: Restore to staging environment
./restore-database.sh --backup=$LATEST_BACKUP --target=staging

# Step 2: Run validation tests
pytest tests/backup-validation/ --env=staging -v

# Step 3: Compare data integrity
psql -h staging-db -c "SELECT count(*) FROM agents;"
psql -h production-db -c "SELECT count(*) FROM agents;"

# Counts should match (or be close for daily backup)

# Step 4: Test point-in-time recovery
TARGET_TIME=$(date -d '1 hour ago' --iso-8601=seconds)
./restore-pitr.sh --target="$TARGET_TIME" --destination=staging

# Step 5: Document results
cat > /reports/backup-test-$(date +%Y%m).txt << EOF
Backup Restore Test Report
Date: $(date)
Backup: $LATEST_BACKUP
Status: PASS/FAIL
Duration: X minutes
Issues: None/List
EOF

# Step 6: Clean up staging
# kubectl delete namespace greenlang-staging
```

**Success Criteria:**
- [ ] Backup downloaded successfully
- [ ] Backup decrypted successfully
- [ ] Database restored without errors
- [ ] Data integrity checks passed
- [ ] Application functional
- [ ] PITR successful
- [ ] Documentation updated

---

## Monitoring and Validation

### Backup Monitoring

**Metrics to Track:**
```promql
# Backup success rate
rate(gl_backup_success_total[24h]) / rate(gl_backup_attempts_total[24h])

# Backup duration
gl_backup_duration_seconds

# Backup size
gl_backup_size_bytes

# Time since last successful backup
time() - gl_backup_last_success_timestamp
```

**Alerts:**
```yaml
alerts:
  - name: backup_failed
    condition: gl_backup_failed_total > 0
    severity: critical
    notification: pagerduty

  - name: backup_missing
    condition: (time() - gl_backup_last_success_timestamp) > 86400
    severity: critical
    notification: pagerduty

  - name: backup_slow
    condition: gl_backup_duration_seconds > 7200
    severity: warning
    notification: slack
```

### Backup Validation Checklist

**Daily:**
- [ ] Check backup completion status
- [ ] Verify backup uploaded to S3
- [ ] Check backup size (should be consistent)
- [ ] Review backup logs for errors

**Weekly:**
- [ ] Test restore to staging (sample)
- [ ] Verify WAL archiving continuous
- [ ] Check S3 bucket lifecycle policies
- [ ] Review storage costs

**Monthly:**
- [ ] Full restore test to staging
- [ ] PITR test
- [ ] Review retention policies
- [ ] Update documentation

---

## Backup Best Practices

1. **Automate Everything:** No manual backups
2. **Test Restores Regularly:** Untested backups = no backups
3. **Monitor Continuously:** Alert on backup failures
4. **Encrypt All Backups:** Protect sensitive data
5. **Use Multiple Storage Locations:** S3 + different region
6. **Version Control Configs:** Git for all configurations
7. **Document Procedures:** Keep this guide updated
8. **Train Team:** Everyone should know restore procedures

---

## Appendix A: Backup Scripts Location

All backup scripts located in: `/opt/greenlang/scripts/backup/`

- `backup-database-full.sh` - Daily database backup
- `backup-database-wal.sh` - WAL archiving setup
- `backup-config.sh` - Configuration backup
- `backup-ebs.sh` - EBS snapshot creation
- `restore-database.sh` - Database restore
- `restore-pitr.sh` - Point-in-time recovery
- `restore-config.sh` - Configuration restore
- `test-backup.sh` - Backup test automation

---

## Appendix B: S3 Bucket Structure

```
s3://greenlang-backups/
├── database/
│   ├── 20251107-020000/
│   │   ├── backup.tar.gz.enc
│   │   └── manifest.json
│   └── ...
├── wal/
│   ├── 000000010000000000000001.lz4
│   └── ...
├── config/
│   ├── 20251107-010000.tar.gz
│   └── ...
├── ebs/
│   └── snapshots.log
└── logs/
    ├── 2025-11/
    └── ...
```

---

## Appendix C: Recovery Contacts

| Role | Name | Contact | Responsibility |
|------|------|---------|----------------|
| Backup Administrator | [Name] | [Phone] | Backup operations |
| Database Administrator | [Name] | [Phone] | Database restore |
| Systems Administrator | [Name] | [Phone] | Infrastructure restore |
| On-Call Engineer | [Rotation] | PagerDuty | Emergency restore |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | Operations Team | Initial comprehensive procedures |

**Next Review Date:** 2025-12-07
**Approved By:** [CTO], [Operations Lead], [DBA]

---

**Your data is only as safe as your last tested backup!**
