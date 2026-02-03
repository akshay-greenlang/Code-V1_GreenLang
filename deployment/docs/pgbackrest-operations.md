# pgBackRest Operations Runbook

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Backup Operations](#backup-operations)
5. [Restore Operations](#restore-operations)
6. [Point-in-Time Recovery (PITR)](#point-in-time-recovery-pitr)
7. [Verification and Monitoring](#verification-and-monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Emergency Procedures](#emergency-procedures)
10. [Maintenance](#maintenance)

---

## Overview

pgBackRest is the backup and recovery solution for GreenLang's PostgreSQL databases. This runbook provides operational procedures for backup management, disaster recovery, and maintenance.

### Key Features

- **Backup Types**: Full, Differential, and Incremental
- **Storage**: AWS S3 with AES-256-CBC encryption
- **Compression**: LZ4 for optimal speed/compression balance
- **Retention**: 4 full (monthly), 7 differential (weekly), 14 incremental (daily)
- **PITR**: Point-in-Time Recovery via WAL archiving

### Contact Information

| Role | Contact |
|------|---------|
| DBA Team | dba-team@greenlang.io |
| On-Call | +1-555-DBA-HELP |
| Slack Channel | #database-ops |

---

## Architecture

### Backup Flow

```
PostgreSQL Primary
        |
        v
   pgBackRest
        |
        +---> WAL Archive (continuous)
        |
        +---> S3 Bucket (scheduled backups)
                  |
                  +---> Standard Storage (30 days)
                  +---> IA Storage (30-90 days)
                  +---> Glacier (90+ days)
```

### Component Diagram

```
+-------------------+     +-------------------+
|   Kubernetes      |     |   AWS Services    |
+-------------------+     +-------------------+
| - CronJob (Full)  |     | - S3 Bucket       |
| - CronJob (Diff)  |     | - KMS Key         |
| - CronJob (Incr)  |     | - Secrets Manager |
| - Job (Restore)   |     | - CloudWatch      |
| - ConfigMap       |     | - IAM Roles       |
| - Secrets (ESO)   |     +-------------------+
+-------------------+
```

### Backup Schedule

| Type | Schedule | Retention | Typical Size |
|------|----------|-----------|--------------|
| Full | 1st of month, 2 AM UTC | 4 backups | 100% of DB |
| Differential | Sunday, 2 AM UTC | 7 backups | 10-30% of DB |
| Incremental | Mon-Sat, 2 AM UTC | 14 backups | 1-5% of DB |

---

## Configuration

### pgBackRest Configuration File

Location: `/etc/pgbackrest/pgbackrest.conf`

```ini
[global]
repo1-type=s3
repo1-path=/pgbackrest/greenlang
repo1-s3-bucket=greenlang-pgbackrest-backups
repo1-s3-region=us-east-1
repo1-cipher-type=aes-256-cbc
compress-type=lz4
repo1-retention-full=4
repo1-retention-diff=7

[greenlang]
pg1-path=/var/lib/postgresql/data
pg1-port=5432
```

### Kubernetes Resources

```bash
# View configuration
kubectl -n greenlang-database get configmap pgbackrest-config -o yaml

# View secrets (external secrets operator)
kubectl -n greenlang-database get externalsecret pgbackrest-secrets -o yaml

# Check CronJob schedules
kubectl -n greenlang-database get cronjobs
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PGBACKREST_REPO1_CIPHER_PASS` | Encryption passphrase |
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |
| `PGHOST` | PostgreSQL host |
| `PGUSER` | PostgreSQL user |

---

## Backup Operations

### Manual Full Backup

Use this when you need an immediate full backup outside the schedule.

```bash
# Option 1: Trigger Kubernetes Job
kubectl -n greenlang-database create job --from=cronjob/pgbackrest-backup-full manual-full-$(date +%Y%m%d%H%M)

# Option 2: Direct pgBackRest command (from a pod)
kubectl -n greenlang-database exec -it deploy/greenlang-postgresql -- bash -c '
  pgbackrest --stanza=greenlang --type=full backup
'
```

### Manual Differential Backup

```bash
# Trigger Kubernetes Job
kubectl -n greenlang-database create job --from=cronjob/pgbackrest-backup-diff manual-diff-$(date +%Y%m%d%H%M)
```

### Manual Incremental Backup

```bash
# Trigger Kubernetes Job
kubectl -n greenlang-database create job --from=cronjob/pgbackrest-backup-incr manual-incr-$(date +%Y%m%d%H%M)
```

### Check Backup Status

```bash
# View all backups
pgbackrest --stanza=greenlang info

# View detailed backup info (JSON)
pgbackrest --stanza=greenlang info --output=json | jq .

# View specific backup
pgbackrest --stanza=greenlang info --set=20240115-020000F

# Check running backup jobs
kubectl -n greenlang-database get jobs -l backup-type
kubectl -n greenlang-database get pods -l backup-type --watch
```

### Verify Backup Integrity

```bash
# Verify latest backup
pgbackrest --stanza=greenlang --set=latest verify

# Verify specific backup
pgbackrest --stanza=greenlang --set=20240115-020000F verify

# Run verification script
/scripts/backup-verify.sh
```

---

## Restore Operations

### Pre-Restore Checklist

- [ ] Identify target recovery time or backup set
- [ ] Verify backup availability
- [ ] Prepare restore storage (PVC)
- [ ] Notify stakeholders
- [ ] Document reason for restore

### Full Restore to New Instance

```bash
# 1. Create restore PVC
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pgbackrest-restore-pvc
  namespace: greenlang-database
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp3
  resources:
    requests:
      storage: 500Gi
EOF

# 2. Create restore job
kubectl apply -f deployment/kubernetes/database/pgbackrest/job-restore.yaml

# 3. Monitor restore progress
kubectl -n greenlang-database logs -f job/pgbackrest-pitr-restore

# 4. Validate restored database
kubectl -n greenlang-database exec -it job/pgbackrest-pitr-restore -c restore-validator -- bash
```

### Restore to Existing Instance (IN-PLACE)

**WARNING**: This will overwrite the existing database!

```bash
# 1. Stop PostgreSQL
kubectl -n greenlang-database scale deployment greenlang-postgresql --replicas=0

# 2. Backup current data (optional safety measure)
kubectl -n greenlang-database exec -it $PG_POD -- tar -czf /tmp/pg-data-backup.tar.gz /var/lib/postgresql/data

# 3. Clear existing data
kubectl -n greenlang-database exec -it $PG_POD -- rm -rf /var/lib/postgresql/data/*

# 4. Run restore
pgbackrest --stanza=greenlang restore \
  --delta \
  --pg1-path=/var/lib/postgresql/data

# 5. Start PostgreSQL
kubectl -n greenlang-database scale deployment greenlang-postgresql --replicas=1
```

---

## Point-in-Time Recovery (PITR)

### Identify Recovery Target

```bash
# Find available recovery points
pgbackrest --stanza=greenlang info

# Sample output:
# stanza: greenlang
#     status: ok
#     cipher: aes-256-cbc
#     db (current)
#         wal archive min/max (14): 000000010000000000000001/00000001000000000000002F
#     full backup: 20240101-020000F
#         ...
#     diff backup: 20240107-020000D
#         ...
```

### Execute PITR

```bash
# Set target time
export RESTORE_TARGET_TIME="2024-01-15 14:30:00+00"

# Method 1: Using Kubernetes Job (Recommended)
cat > /tmp/pitr-job.yaml <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: pgbackrest-pitr-$(date +%Y%m%d%H%M)
  namespace: greenlang-database
spec:
  template:
    spec:
      containers:
      - name: restore
        image: pgbackrest/pgbackrest:2.49
        env:
        - name: RESTORE_TARGET_TIME
          value: "$RESTORE_TARGET_TIME"
        command: ["/scripts/restore-pitr.sh"]
      restartPolicy: Never
EOF

kubectl apply -f /tmp/pitr-job.yaml

# Method 2: Direct command
pgbackrest --stanza=greenlang restore \
  --target="$RESTORE_TARGET_TIME" \
  --target-action=promote \
  --type=time \
  --pg1-path=/var/lib/postgresql/data-restored \
  --delta
```

### PITR Recovery Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--target-action=promote` | Promote to primary | Full recovery |
| `--target-action=pause` | Pause at target | Verify before commit |
| `--target-action=shutdown` | Shutdown after recovery | Safe state |
| `--target-timeline=latest` | Follow latest timeline | After failover |

### Post-PITR Validation

```bash
# 1. Connect to restored database
psql -h localhost -p 5433 -U postgres

# 2. Verify recovery time
SELECT pg_last_xact_replay_timestamp();

# 3. Check data integrity
\dt
SELECT COUNT(*) FROM important_table;

# 4. Verify no data loss
SELECT MAX(created_at) FROM audit_log;
```

---

## Verification and Monitoring

### Daily Verification (Automated)

The verification script runs automatically via CronJob:

```bash
# Check verification job status
kubectl -n greenlang-database get cronjob pgbackrest-verify

# View recent verification logs
kubectl -n greenlang-database logs -l app.kubernetes.io/component=verify --tail=100
```

### Manual Verification

```bash
# Full verification
/scripts/backup-verify.sh

# Quick status check
pgbackrest --stanza=greenlang check
```

### Monitoring Dashboards

- **Grafana**: https://grafana.greenlang.io/d/pgbackrest
- **CloudWatch**: Console > CloudWatch > Dashboards > greenlang-pgbackrest

### Key Metrics

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `pgbackrest_last_backup_timestamp` | > 26 hours | No recent backup |
| `pgbackrest_backup_duration_seconds` | > 7200 | Backup taking too long |
| `pgbackrest_verify_status` | == 0 | Verification failed |
| `s3_bucket_size_bytes` | > 1TB | Storage growing |

### Alerting Channels

Alerts are sent to:
- Slack: #database-alerts
- PagerDuty: Database On-Call
- Email: dba-team@greenlang.io

---

## Troubleshooting

### Common Issues

#### Backup Failed: "unable to find primary"

```bash
# Check PostgreSQL is running
kubectl -n greenlang-database get pods -l app=postgresql

# Verify connection
pgbackrest --stanza=greenlang check

# Check pg_hba.conf allows connections
kubectl exec -it $PG_POD -- cat /var/lib/postgresql/data/pg_hba.conf
```

#### Backup Failed: S3 access denied

```bash
# Verify IAM role
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://greenlang-pgbackrest-backups/

# Check secrets
kubectl -n greenlang-database get secret pgbackrest-secrets -o yaml
```

#### Backup Failed: Encryption error

```bash
# Verify passphrase is set
echo $PGBACKREST_REPO1_CIPHER_PASS | wc -c

# Re-fetch secrets
kubectl -n greenlang-database rollout restart deployment/external-secrets
```

#### Restore Failed: WAL not found

```bash
# List available WAL
pgbackrest --stanza=greenlang archive-get --ls

# Check WAL archive status
pgbackrest --stanza=greenlang info --output=json | jq '.[0].archive'

# Verify S3 bucket contains WAL
aws s3 ls s3://greenlang-pgbackrest-backups/pgbackrest/greenlang/archive/
```

#### Slow Backup Performance

```bash
# Check network bandwidth
iperf3 -c s3.amazonaws.com -p 443

# Increase parallelism
pgbackrest --stanza=greenlang --type=full --process-max=8 backup

# Check S3 transfer acceleration
aws s3api get-bucket-accelerate-configuration --bucket greenlang-pgbackrest-backups
```

### Log Analysis

```bash
# pgBackRest logs
kubectl -n greenlang-database logs -l app.kubernetes.io/name=pgbackrest --tail=500

# PostgreSQL WAL archive logs
kubectl -n greenlang-database exec -it $PG_POD -- tail -f /var/log/postgresql/postgresql-*.log | grep -i archive

# CloudWatch Logs
aws logs get-log-events \
  --log-group-name /aws/pgbackrest/greenlang-prod \
  --log-stream-name backup
```

---

## Emergency Procedures

### Database Corruption Detected

**Severity**: Critical
**Response Time**: Immediate

```bash
# 1. Stop all write operations
kubectl -n greenlang exec -it $PG_POD -- psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state != 'idle';"

# 2. Create emergency backup (if possible)
pgbackrest --stanza=greenlang --type=full backup --force

# 3. Assess corruption
pg_dump -Fc greenlang > /tmp/emergency-dump.dump 2>&1 || echo "Dump failed"

# 4. If dump fails, proceed to PITR
export RESTORE_TARGET_TIME="$(date -d '1 hour ago' -Iseconds)"
/scripts/restore-pitr.sh

# 5. Notify stakeholders
curl -X POST $SLACK_WEBHOOK -d '{"text":"EMERGENCY: Database corruption detected, PITR in progress"}'
```

### S3 Bucket Unavailable

**Severity**: High
**Response Time**: 15 minutes

```bash
# 1. Check S3 status
aws s3 ls s3://greenlang-pgbackrest-backups/ || echo "S3 unavailable"

# 2. Check AWS status page
curl -s https://status.aws.amazon.com/

# 3. If regional outage, switch to secondary bucket
# Update pgbackrest.conf to use secondary bucket
sed -i 's/us-east-1/us-west-2/g' /etc/pgbackrest/pgbackrest.conf
sed -i 's/greenlang-pgbackrest-backups/greenlang-pgbackrest-backups-dr/g' /etc/pgbackrest/pgbackrest.conf

# 4. Run backup to secondary
pgbackrest --stanza=greenlang --type=full backup
```

### Complete Backup Chain Lost

**Severity**: Critical
**Response Time**: Immediate

```bash
# 1. Assess situation
pgbackrest --stanza=greenlang info

# 2. If no backups available, create new baseline
pgbackrest --stanza=greenlang stanza-create --force
pgbackrest --stanza=greenlang --type=full backup

# 3. Verify new backup
pgbackrest --stanza=greenlang --set=latest verify

# 4. Update retention and document incident
```

### Ransomware/Security Incident

**Severity**: Critical
**Response Time**: Immediate

```bash
# 1. Isolate affected systems
kubectl -n greenlang-database delete networkpolicy --all
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: emergency-isolate
  namespace: greenlang-database
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF

# 2. Identify last known good backup
pgbackrest --stanza=greenlang info --output=json | jq '.[] | .backup[] | select(.annotation.ransomware != "true")'

# 3. DO NOT delete any backups until security team approves
# 4. Contact security team immediately
# 5. Preserve evidence
```

---

## Maintenance

### Stanza Management

```bash
# Create stanza (initial setup)
pgbackrest --stanza=greenlang stanza-create

# Upgrade stanza (after PostgreSQL upgrade)
pgbackrest --stanza=greenlang stanza-upgrade

# Delete stanza (caution!)
pgbackrest --stanza=greenlang stanza-delete --force
```

### Backup Expiration

Automated via retention policies, but can be run manually:

```bash
# Expire old backups
pgbackrest --stanza=greenlang expire

# Dry run
pgbackrest --stanza=greenlang expire --dry-run
```

### Repository Maintenance

```bash
# Check repository info
pgbackrest --stanza=greenlang repo-ls

# Get repository size
aws s3 ls s3://greenlang-pgbackrest-backups/ --recursive --summarize

# Clean up incomplete multipart uploads
aws s3api list-multipart-uploads --bucket greenlang-pgbackrest-backups
```

### Configuration Updates

```bash
# Update ConfigMap
kubectl -n greenlang-database edit configmap pgbackrest-config

# Restart affected pods
kubectl -n greenlang-database rollout restart deployment/greenlang-postgresql

# Validate new configuration
pgbackrest --stanza=greenlang check
```

### Encryption Key Rotation

**Note**: Key rotation requires re-encrypting all backups. Plan carefully.

```bash
# 1. Create new encryption passphrase
NEW_PASSPHRASE=$(openssl rand -base64 32)

# 2. Update secret
kubectl -n greenlang-database patch secret pgbackrest-secrets -p "{\"stringData\":{\"encryption-passphrase-new\":\"$NEW_PASSPHRASE\"}}"

# 3. Run full backup with new key
PGBACKREST_REPO1_CIPHER_PASS=$NEW_PASSPHRASE pgbackrest --stanza=greenlang --type=full backup

# 4. Verify new backup
PGBACKREST_REPO1_CIPHER_PASS=$NEW_PASSPHRASE pgbackrest --stanza=greenlang --set=latest verify

# 5. Update primary passphrase
kubectl -n greenlang-database patch secret pgbackrest-secrets -p "{\"stringData\":{\"encryption-passphrase\":\"$NEW_PASSPHRASE\"}}"

# 6. Document old passphrase in secure storage (for old backup recovery)
```

---

## Appendix

### pgBackRest Command Reference

| Command | Description |
|---------|-------------|
| `pgbackrest backup` | Create backup |
| `pgbackrest restore` | Restore from backup |
| `pgbackrest info` | Display backup info |
| `pgbackrest verify` | Verify backup integrity |
| `pgbackrest check` | Validate configuration |
| `pgbackrest expire` | Remove expired backups |
| `pgbackrest stanza-create` | Create stanza |
| `pgbackrest stanza-upgrade` | Upgrade stanza |
| `pgbackrest archive-get` | Get WAL file |
| `pgbackrest archive-push` | Push WAL file |

### Recovery Time Objectives

| Scenario | RTO | RPO |
|----------|-----|-----|
| Single table recovery | 15 min | 0 (PITR) |
| Full database restore | 2 hours | 0 (PITR) |
| Complete DR failover | 4 hours | 5 min |

### File Locations

| File | Location |
|------|----------|
| Configuration | `/etc/pgbackrest/pgbackrest.conf` |
| Logs | `/var/log/pgbackrest/` |
| Spool | `/var/spool/pgbackrest/` |
| PostgreSQL Data | `/var/lib/postgresql/data/` |

### Related Documentation

- [PostgreSQL PITR Documentation](https://www.postgresql.org/docs/current/continuous-archiving.html)
- [pgBackRest User Guide](https://pgbackrest.org/user-guide.html)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)

---

**Document Version**: 1.0
**Last Updated**: 2024-01-15
**Owner**: GreenLang DBA Team
