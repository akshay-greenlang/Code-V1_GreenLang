# GL-VCCI Backup and Disaster Recovery Plan
## Scope 3 Carbon Intelligence Platform v2.0

**Version:** 2.0.0
**Date:** November 8, 2025
**Classification:** CONFIDENTIAL - Internal Use Only
**Approval:** CTO, VP Engineering, CISO

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Recovery Objectives](#recovery-objectives)
3. [Backup Strategy](#backup-strategy)
4. [Backup Procedures](#backup-procedures)
5. [Recovery Procedures](#recovery-procedures)
6. [Disaster Scenarios](#disaster-scenarios)
7. [Testing & Validation](#testing--validation)
8. [Responsibilities](#responsibilities)

---

## Executive Summary

### Purpose

This document defines the backup and disaster recovery (DR) strategy for the GL-VCCI Scope 3 Carbon Intelligence Platform. It ensures business continuity and data protection in the event of system failures, data corruption, security incidents, or regional disasters.

### Scope

This plan covers:
- **Database:** PostgreSQL (primary data store)
- **Object Storage:** S3-compatible storage (uploaded files, reports)
- **Redis:** Cache and session data
- **Configuration:** Kubernetes manifests, secrets, ConfigMaps
- **Application Code:** Git repository

### Business Impact

| Data Loss Scenario | Business Impact | Recovery Priority |
|-------------------|----------------|-------------------|
| **Complete database loss** | CRITICAL - All tenant data lost | P0 - Immediate |
| **Recent data loss (<1 hour)** | HIGH - Recent calculations lost | P1 - < 1 hour |
| **File storage loss** | MEDIUM - Reports can be regenerated | P2 - < 4 hours |
| **Configuration loss** | MEDIUM - Redeployment required | P2 - < 4 hours |
| **Cache loss** | LOW - Performance degradation only | P3 - < 24 hours |

---

## Recovery Objectives

### RTO (Recovery Time Objective)

**Maximum acceptable downtime:**

| Severity | RTO | Description |
|----------|-----|-------------|
| **P0 - Critical** | 1 hour | Complete data loss or regional failure |
| **P1 - High** | 4 hours | Major component failure |
| **P2 - Medium** | 24 hours | Non-critical component failure |
| **P3 - Low** | 72 hours | Degraded performance |

### RPO (Recovery Point Objective)

**Maximum acceptable data loss:**

| Data Type | RPO | Backup Frequency | Retention |
|-----------|-----|------------------|-----------|
| **Database (Production)** | 15 minutes | Continuous (PITR) | 30 days |
| **Database (Snapshots)** | 24 hours | Daily | 90 days |
| **Object Storage** | 1 hour | Continuous replication | 90 days + versioning |
| **Redis** | 1 hour | AOF persistence | 7 days |
| **Configuration** | On commit | Git repository | Indefinite |
| **Application Code** | On commit | Git repository | Indefinite |

### SLA Commitments

**Production Environment:**
- **Uptime:** 99.9% (< 8.76 hours downtime/year)
- **Data Durability:** 99.999999999% (11 nines)
- **Data Loss:** < 15 minutes (RPO)
- **Recovery Time:** < 1 hour (RTO)

---

## Backup Strategy

### Multi-Layered Defense

```
┌─────────────────────────────────────────────────────────────┐
│                    DISASTER RECOVERY LAYERS                  │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Real-time Replication (Read Replicas)             │
│   - Cross-AZ replication (sync)                             │
│   - Immediate failover (<30s)                               │
│   - RPO: 0 seconds, RTO: 30 seconds                         │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Continuous Backups (PITR)                          │
│   - WAL archiving to S3                                      │
│   - Point-in-time recovery (15 min RPO)                     │
│   - RPO: 15 minutes, RTO: 1 hour                            │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Daily Snapshots                                     │
│   - Full database dumps                                      │
│   - 30-day retention                                         │
│   - RPO: 24 hours, RTO: 2 hours                             │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Cross-Region Replication                           │
│   - Async replication to DR region                          │
│   - Complete disaster recovery                               │
│   - RPO: 1 hour, RTO: 4 hours                               │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Offline Backups (Glacier)                          │
│   - Long-term archival (7 years)                            │
│   - Compliance (SOC 2, regulatory)                          │
│   - RPO: N/A, RTO: 24-48 hours                              │
└─────────────────────────────────────────────────────────────┘
```

### Backup Architecture

**Primary Region (us-east-1):**
- Production database with 2 read replicas (cross-AZ)
- Continuous WAL archiving to S3
- Daily snapshots to S3 (encrypted)
- Object storage with versioning

**DR Region (us-west-2):**
- Cross-region replication (async)
- Read-only database replica
- Object storage replica

**Offline Storage:**
- Glacier Deep Archive (compliance)
- Monthly snapshots (7-year retention)

---

## Backup Procedures

### 1. Database Backups

#### Continuous Backups (WAL Archiving)

**Automatic - No Manual Intervention Required**

PostgreSQL is configured for continuous archiving:

```sql
-- Verify WAL archiving is active
SELECT * FROM pg_stat_archiver;

-- Expected output:
-- archived_count | last_archived_wal | last_archived_time
-- > 0            | 00000...          | < 5 minutes ago
```

**Configuration:**
```ini
# postgresql.conf
wal_level = replica
archive_mode = on
archive_command = 'aws s3 cp %p s3://vcci-backups/wal/%f --region us-east-1'
archive_timeout = 300  # 5 minutes
```

**Monitoring:**
```bash
# Check WAL archive status
psql -c "SELECT * FROM pg_stat_archiver;"

# Check for failed archives
aws s3 ls s3://vcci-backups/wal/ | tail -20
```

#### Daily Snapshots

**Automatic via scripts/backup_database.sh**

**Schedule:** Daily at 2:00 AM UTC (low traffic)

```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="vcci_production_${BACKUP_DATE}.dump"
S3_BUCKET="s3://vcci-backups/daily"

# 1. Create compressed backup
pg_dump \
  -h vcci-prod.cluster-xyz.us-east-1.rds.amazonaws.com \
  -U vcci_admin \
  -d vcci_production \
  -F c \
  -Z 9 \
  -f /tmp/${BACKUP_FILE}

# 2. Verify backup
pg_restore --list /tmp/${BACKUP_FILE} | head -20

# 3. Upload to S3
aws s3 cp /tmp/${BACKUP_FILE} ${S3_BUCKET}/${BACKUP_FILE} \
  --storage-class STANDARD_IA \
  --server-side-encryption AES256

# 4. Verify upload
aws s3 ls ${S3_BUCKET}/${BACKUP_FILE}

# 5. Clean up local file
rm /tmp/${BACKUP_FILE}

# 6. Clean up old backups (keep 30 days)
aws s3 ls ${S3_BUCKET}/ | awk '{print $4}' | while read file; do
  file_date=$(echo $file | grep -oP '\d{8}')
  days_old=$(( ($(date +%s) - $(date -d $file_date +%s)) / 86400 ))

  if [ $days_old -gt 30 ]; then
    echo "Deleting old backup: $file (${days_old} days old)"
    aws s3 rm ${S3_BUCKET}/$file
  fi
done

# 7. Send notification
curl -X POST https://hooks.slack.com/services/XXX/YYY/ZZZ \
  -H 'Content-Type: application/json' \
  -d "{\"text\": \"✓ Database backup completed: ${BACKUP_FILE}\"}"
```

**Cron Schedule:**
```cron
# Crontab on backup server
0 2 * * * /opt/vcci/scripts/backup_database.sh >> /var/log/vcci/backup.log 2>&1
```

**Manual Backup (Emergency):**
```bash
# Run immediately
sudo -u postgres /opt/vcci/scripts/backup_database.sh
```

#### Weekly Full Backups

**Schedule:** Sundays at 1:00 AM UTC

```bash
#!/bin/bash
# scripts/backup_database_full.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="vcci_production_full_${BACKUP_DATE}.dump"
S3_BUCKET="s3://vcci-backups/weekly"

# Full backup with all data
pg_dump \
  -h vcci-prod.cluster-xyz.us-east-1.rds.amazonaws.com \
  -U vcci_admin \
  -d vcci_production \
  -F d \
  -j 4 \
  -f /tmp/${BACKUP_FILE}

# Upload to S3
aws s3 sync /tmp/${BACKUP_FILE} ${S3_BUCKET}/${BACKUP_FILE}/ \
  --storage-class STANDARD_IA

# Clean up (keep 90 days)
find /tmp -name "vcci_production_full_*" -mtime +90 -delete
```

#### Monthly Archive Backups

**Schedule:** 1st of month at 12:00 AM UTC

```bash
#!/bin/bash
# scripts/backup_database_archive.sh

BACKUP_DATE=$(date +%Y%m)
BACKUP_FILE="vcci_production_archive_${BACKUP_DATE}.dump"
S3_BUCKET="s3://vcci-backups/archive"

# Create backup
pg_dump -F c -Z 9 -f /tmp/${BACKUP_FILE} $DATABASE_URL

# Upload to Glacier Deep Archive
aws s3 cp /tmp/${BACKUP_FILE} ${S3_BUCKET}/${BACKUP_FILE} \
  --storage-class DEEP_ARCHIVE

# Keep forever (compliance requirement: 7 years)
```

### 2. Object Storage Backups

**S3 Versioning (Automatic):**

```bash
# Enable versioning on production bucket
aws s3api put-bucket-versioning \
  --bucket vcci-uploads \
  --versioning-configuration Status=Enabled

# Enable cross-region replication
aws s3api put-bucket-replication \
  --bucket vcci-uploads \
  --replication-configuration file://replication-config.json
```

**Replication Configuration:**
```json
{
  "Role": "arn:aws:iam::ACCOUNT:role/S3ReplicationRole",
  "Rules": [{
    "Status": "Enabled",
    "Priority": 1,
    "DeleteMarkerReplication": { "Status": "Enabled" },
    "Filter": {},
    "Destination": {
      "Bucket": "arn:aws:s3:::vcci-uploads-dr",
      "ReplicationTime": {
        "Status": "Enabled",
        "Time": { "Minutes": 15 }
      },
      "Metrics": {
        "Status": "Enabled",
        "EventThreshold": { "Minutes": 15 }
      }
    }
  }]
}
```

**Lifecycle Policy:**
```json
{
  "Rules": [{
    "Status": "Enabled",
    "Transitions": [
      {
        "Days": 30,
        "StorageClass": "STANDARD_IA"
      },
      {
        "Days": 90,
        "StorageClass": "GLACIER_IR"
      },
      {
        "Days": 365,
        "StorageClass": "DEEP_ARCHIVE"
      }
    ],
    "NoncurrentVersionExpiration": {
      "NoncurrentDays": 90
    }
  }]
}
```

### 3. Configuration Backups

**Git Repository (Automatic):**

All infrastructure as code is versioned in Git:
- Kubernetes manifests: `infrastructure/kubernetes/`
- Terraform configs: `infrastructure/terraform/`
- Helm charts: `infrastructure/helm/`

```bash
# Backup strategy
git push origin main
git push origin --tags
git push backup main  # Secondary remote (GitLab/Bitbucket)
```

**Kubernetes Resources (Daily Export):**

```bash
#!/bin/bash
# scripts/backup_k8s_resources.sh

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/k8s/${BACKUP_DATE}"
NAMESPACE="vcci-production"

mkdir -p ${BACKUP_DIR}

# Export all resources
kubectl get all -n ${NAMESPACE} -o yaml > ${BACKUP_DIR}/all-resources.yaml
kubectl get configmaps -n ${NAMESPACE} -o yaml > ${BACKUP_DIR}/configmaps.yaml
kubectl get secrets -n ${NAMESPACE} -o yaml > ${BACKUP_DIR}/secrets.yaml
kubectl get pvc -n ${NAMESPACE} -o yaml > ${BACKUP_DIR}/pvc.yaml
kubectl get ingress -n ${NAMESPACE} -o yaml > ${BACKUP_DIR}/ingress.yaml

# Upload to S3
tar -czf ${BACKUP_DIR}.tar.gz ${BACKUP_DIR}
aws s3 cp ${BACKUP_DIR}.tar.gz s3://vcci-backups/k8s/

# Clean up local
rm -rf ${BACKUP_DIR}*
```

### 4. Redis Backups

**AOF Persistence (Automatic):**

Redis is configured with Append-Only File persistence:

```conf
# redis.conf
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
```

**Daily RDB Snapshots:**

```bash
# Manual snapshot
redis-cli BGSAVE

# Automated via cron
0 3 * * * redis-cli BGSAVE && \
  aws s3 cp /var/lib/redis/dump.rdb s3://vcci-backups/redis/dump_$(date +\%Y\%m\%d).rdb
```

---

## Recovery Procedures

### 1. Database Recovery

#### Scenario A: Recent Data Loss (< 15 minutes)

**Use Point-in-Time Recovery (PITR)**

**Steps:**

```bash
# 1. Stop application
kubectl scale deployment/backend-api --replicas=0 -n vcci-production

# 2. Identify recovery point
TARGET_TIME="2025-11-08 14:30:00"  # Time before data loss

# 3. Create recovery directory
mkdir -p /recovery/pgdata

# 4. Restore base backup
aws s3 cp s3://vcci-backups/daily/vcci_production_20251108_020000.dump /recovery/

pg_restore -C -d postgres /recovery/vcci_production_20251108_020000.dump

# 5. Restore WAL files up to target time
# (AWS RDS does this automatically with restore-db-instance-to-point-in-time)

aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier vcci-prod \
  --target-db-instance-identifier vcci-prod-recovery \
  --restore-time "${TARGET_TIME}" \
  --no-multi-az

# 6. Wait for restoration (10-30 minutes)
aws rds wait db-instance-available \
  --db-instance-identifier vcci-prod-recovery

# 7. Verify data
psql -h vcci-prod-recovery... -U vcci_admin -d vcci_production -c "SELECT count(*) FROM suppliers;"

# 8. Promote recovery instance (or restore to main instance)
# Option A: Rename recovery instance to production (requires DNS update)
# Option B: Restore from recovery instance to main instance

# 9. Update application config
kubectl set env deployment/backend-api DATABASE_URL=$NEW_DATABASE_URL -n vcci-production

# 10. Scale up application
kubectl scale deployment/backend-api --replicas=3 -n vcci-production

# 11. Verify health
curl https://api.vcci.greenlang.io/health/ready
```

**Expected Recovery Time:** 1-2 hours
**Data Loss:** < 15 minutes

#### Scenario B: Complete Database Loss

**Use Latest Daily Snapshot**

```bash
# 1. Stop application
kubectl scale deployment/backend-api --replicas=0 -n vcci-production
kubectl scale deployment/worker --replicas=0 -n vcci-production

# 2. Identify latest backup
LATEST_BACKUP=$(aws s3 ls s3://vcci-backups/daily/ | sort | tail -1 | awk '{print $4}')

echo "Restoring from: ${LATEST_BACKUP}"

# 3. Download backup
aws s3 cp s3://vcci-backups/daily/${LATEST_BACKUP} /recovery/

# 4. Create new database
psql -h $NEW_DB_HOST -U postgres -c "CREATE DATABASE vcci_production;"

# 5. Restore backup
pg_restore -h $NEW_DB_HOST -U vcci_admin -d vcci_production -j 4 /recovery/${LATEST_BACKUP}

# 6. Verify restoration
psql -h $NEW_DB_HOST -U vcci_admin -d vcci_production -c "\dt"
psql -h $NEW_DB_HOST -U vcci_admin -d vcci_production -c "SELECT count(*) FROM suppliers;"

# 7. Update connection strings
kubectl set env deployment/backend-api DATABASE_URL=$NEW_DATABASE_URL -n vcci-production

# 8. Run smoke tests
pytest tests/integration/test_database.py

# 9. Scale up
kubectl scale deployment/backend-api --replicas=3 -n vcci-production
kubectl scale deployment/worker --replicas=3 -n vcci-production

# 10. Monitor
kubectl logs -f deployment/backend-api -n vcci-production
```

**Expected Recovery Time:** 2-4 hours
**Data Loss:** Up to 24 hours (last snapshot)

### 2. Object Storage Recovery

#### Restore Deleted Files

```bash
# List versions of a file
aws s3api list-object-versions \
  --bucket vcci-uploads \
  --prefix reports/2025/supplier_report.pdf

# Restore specific version
aws s3api copy-object \
  --copy-source vcci-uploads/reports/2025/supplier_report.pdf?versionId=VERSION_ID \
  --bucket vcci-uploads \
  --key reports/2025/supplier_report.pdf
```

#### Restore from DR Region

```bash
# Sync from DR region
aws s3 sync s3://vcci-uploads-dr/ s3://vcci-uploads/ \
  --source-region us-west-2 \
  --region us-east-1
```

### 3. Complete Regional Disaster

**Failover to DR Region (us-west-2)**

**Pre-Requisites:**
- DR region has async replica of database
- DR region has replicated object storage
- Kubernetes cluster pre-provisioned in DR region

**Steps:**

```bash
# 1. Promote DR database to primary
aws rds promote-read-replica \
  --db-instance-identifier vcci-prod-dr \
  --region us-west-2

# 2. Update DNS to point to DR region
aws route53 change-resource-record-sets \
  --hosted-zone-id Z123456 \
  --change-batch file://dns-failover.json

# 3. Deploy application to DR cluster
kubectl config use-context vcci-dr-cluster
kubectl apply -f infrastructure/kubernetes/ -n vcci-production

# 4. Scale up in DR region
kubectl scale deployment/backend-api --replicas=3 -n vcci-production
kubectl scale deployment/worker --replicas=3 -n vcci-production

# 5. Verify health
curl https://api.vcci.greenlang.io/health/ready

# 6. Monitor for 24 hours
# 7. Communicate status to customers
# 8. Plan recovery back to primary region
```

**Expected Recovery Time:** 4-6 hours
**Data Loss:** < 1 hour (replication lag)

---

## Disaster Scenarios

### Scenario Matrix

| Scenario | Likelihood | Impact | Recovery Procedure | RTO | RPO |
|----------|-----------|--------|-------------------|-----|-----|
| **Database corruption** | Medium | Critical | PITR or snapshot restore | 1-2h | 15m |
| **Accidental data deletion** | Medium | High | PITR | 1h | 0 |
| **Ransomware attack** | Low | Critical | Restore from offline backup | 4h | 24h |
| **Regional outage (AWS)** | Very Low | Critical | Failover to DR region | 4-6h | 1h |
| **Complete data center loss** | Very Low | Critical | DR region + Glacier restore | 24h | 24h |
| **Object storage deletion** | Low | Medium | Restore from versioning | 2h | 1h |
| **Configuration loss** | Low | Medium | Restore from Git | 1h | 0 |
| **Redis data loss** | Medium | Low | No recovery needed (cache only) | 0 | N/A |

### Response Team

| Role | Primary | Backup | Responsibilities |
|------|---------|--------|-----------------|
| **Incident Commander** | VP Engineering | CTO | Overall coordination, decision-making |
| **Database Lead** | Senior DBA | DevOps Lead | Database recovery, validation |
| **Infrastructure Lead** | DevOps Lead | Senior DevOps | K8s, networking, DNS |
| **Application Lead** | Senior Backend Eng | Lead Backend Eng | Application deployment, testing |
| **Communications** | Product Manager | Customer Success | Customer updates, status page |
| **Security Lead** | CISO | Security Engineer | If security incident |

---

## Testing & Validation

### Backup Testing Schedule

| Test Type | Frequency | Procedure | Pass Criteria |
|-----------|-----------|-----------|---------------|
| **Backup Verification** | Daily | Automated checksum validation | All backups have valid checksums |
| **Restore Test (Dev)** | Weekly | Restore to dev environment | Full restore in < 30 minutes |
| **Restore Test (Staging)** | Monthly | Restore to staging | Full restore in < 1 hour, all tests pass |
| **DR Failover Drill** | Quarterly | Full failover to DR region | Complete failover in < 6 hours |
| **Disaster Recovery Exercise** | Annually | Tabletop + full DR test | All procedures documented and validated |

### Backup Monitoring

**Automated Alerts:**

```yaml
# Prometheus alerts
- alert: BackupFailed
  expr: time() - vcci_last_backup_timestamp_seconds > 86400
  for: 1h
  labels:
    severity: critical
  annotations:
    summary: "Database backup has not completed in 24 hours"

- alert: WALArchivingBehind
  expr: pg_stat_archiver_archived_count{job="postgres"} == 0
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "WAL archiving has not occurred in 30 minutes"

- alert: ReplicationLag
  expr: pg_replication_lag_seconds > 300
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Replication lag is > 5 minutes"
```

**Monitoring Dashboard:**

- Last backup time
- Backup size trend
- WAL archiving rate
- Replication lag
- S3 replication status
- Backup success/failure rate

---

## Responsibilities

### Daily Operations

- **DevOps Team:**
  - Monitor backup completion
  - Respond to backup failures
  - Validate backup integrity

- **Database Team:**
  - Monitor replication lag
  - Perform weekly restore tests
  - Tune backup performance

### Incident Response

- **On-Call Engineer:**
  - Initiate recovery procedures
  - Escalate to Incident Commander if RTO at risk

- **Incident Commander:**
  - Declare incident severity
  - Coordinate recovery team
  - Approve recovery actions

### Continuous Improvement

- **Quarterly Review:**
  - Review backup/recovery metrics
  - Update procedures based on lessons learned
  - Validate RTO/RPO targets

---

## Appendix

### Quick Reference

**Emergency Contacts:**
- DevOps On-Call: oncall-devops@greenlang.io / +1-XXX-XXX-XXXX
- Database On-Call: oncall-dba@greenlang.io / +1-XXX-XXX-XXXX
- Incident Commander: VP-Engineering@greenlang.io / +1-XXX-XXX-XXXX

**Key Resources:**
- Backup Bucket: s3://vcci-backups
- DR Database: vcci-prod-dr.us-west-2.rds.amazonaws.com
- Runbook: docs/runbooks/DATA_RECOVERY.md
- Status Page: status.vcci.greenlang.io

### Backup Inventory

**As of November 8, 2025:**

| Backup Type | Count | Total Size | Oldest | Newest |
|-------------|-------|------------|--------|--------|
| Daily Snapshots | 30 | 1.2 TB | 30 days ago | Today |
| Weekly Full | 12 | 600 GB | 90 days ago | This week |
| Monthly Archive | 24 | 300 GB | 2 years ago | This month |
| WAL Archives | ~43,200 | 2.5 TB | 30 days ago | 5 min ago |

---

**Document Version:** 2.0.0
**Last Updated:** November 8, 2025
**Next Review:** February 8, 2026 (Quarterly)
**Document Owner:** VP Engineering
