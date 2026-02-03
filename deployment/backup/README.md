# GreenLang Backup and Disaster Recovery Strategy

**INFRA-001: Backup and Disaster Recovery Configuration**
**Version:** 1.0.0
**Last Updated:** 2026-02-03

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Backup Components](#backup-components)
4. [Recovery Objectives](#recovery-objectives)
5. [Quick Start](#quick-start)
6. [Directory Structure](#directory-structure)
7. [Backup Schedules](#backup-schedules)
8. [Restore Procedures](#restore-procedures)
9. [Disaster Recovery](#disaster-recovery)
10. [Testing and Validation](#testing-and-validation)
11. [Monitoring and Alerts](#monitoring-and-alerts)
12. [Compliance](#compliance)

---

## Overview

This directory contains the complete backup and disaster recovery (DR) configuration for the GreenLang Carbon Intelligence Platform. The backup strategy ensures data protection, business continuity, and compliance with regulatory requirements.

### Key Features

- **Multi-tier backup strategy**: Hourly, daily, weekly, and monthly backups
- **Point-in-Time Recovery (PITR)**: Restore to any second within 35 days
- **Cross-region replication**: Automated backup replication to DR region
- **Kubernetes backup**: Velero-based cluster state backup
- **Automated verification**: Daily backup integrity checks
- **DR automation**: One-command failover to DR region

---

## Architecture

```
                                PRIMARY REGION (us-east-1)
    +------------------------------------------------------------------+
    |                                                                  |
    |   +----------------+     +----------------+     +-------------+  |
    |   | Application    |     | PostgreSQL     |     | Redis       |  |
    |   | (Kubernetes)   |     | (RDS)          |     | (K8s)       |  |
    |   +-------+--------+     +-------+--------+     +------+------+  |
    |           |                      |                     |         |
    |           v                      v                     v         |
    |   +----------------+     +----------------+     +-------------+  |
    |   | Velero         |     | Automated      |     | RDB/AOF     |  |
    |   | Backups        |     | Snapshots      |     | Backups     |  |
    |   +-------+--------+     +-------+--------+     +------+------+  |
    |           |                      |                     |         |
    |           v                      v                     v         |
    |   +----------------------------------------------------------+   |
    |   |                     S3 Backup Buckets                    |   |
    |   |  - greenlang-velero-backups                              |   |
    |   |  - greenlang-backups (PostgreSQL, Redis, Weaviate)       |   |
    |   +---------------------------+------------------------------+   |
    |                               |                                  |
    +-------------------------------|----------------------------------+
                                    |
                    Cross-Region Replication
                                    |
                                    v
                              DR REGION (eu-west-1)
    +------------------------------------------------------------------+
    |   +----------------------------------------------------------+   |
    |   |                  Replicated Backups                      |   |
    |   |  - greenlang-velero-backups-dr                           |   |
    |   |  - greenlang-backups-dr                                  |   |
    |   |  - RDS Cross-Region Automated Backups                    |   |
    |   +----------------------------------------------------------+   |
    |                                                                  |
    |   (Standby infrastructure - activated during DR failover)        |
    |                                                                  |
    +------------------------------------------------------------------+
```

---

## Backup Components

### 1. Database Backups (PostgreSQL RDS)

| Type | Frequency | Retention | Storage |
|------|-----------|-----------|---------|
| Automated Snapshots | Daily | 35 days | RDS |
| WAL Archive (PITR) | Continuous (5 min) | 35 days | S3 |
| Weekly Snapshots | Weekly | 90 days | S3 |
| Monthly Snapshots | Monthly | 1 year | S3 Glacier |
| Cross-Region | Continuous | 35 days | eu-west-1 |

### 2. Kubernetes Backups (Velero)

| Type | Frequency | Retention | Scope |
|------|-----------|-----------|-------|
| Hourly | Every hour | 24 hours | Critical ConfigMaps/Secrets |
| Daily | 2:00 AM UTC | 7 days | Full namespace |
| Weekly | Sunday 3:00 AM | 30 days | All namespaces |
| Monthly | 1st of month | 1 year | Complete cluster |

### 3. Cache Backups (Redis)

| Type | Frequency | Retention |
|------|-----------|-----------|
| RDB Snapshots | Every 6 hours | 7 days |
| AOF Persistence | Continuous | 7 days |

### 4. Vector Database (Weaviate)

| Type | Frequency | Retention |
|------|-----------|-----------|
| Full Backup | Daily | 30 days |

---

## Recovery Objectives

| Metric | Target | Actual |
|--------|--------|--------|
| **RPO** (Recovery Point Objective) | < 1 hour | ~5 minutes (PITR) |
| **RTO** (Recovery Time Objective) | < 4 hours | 1-2 hours (typical) |
| **Availability** | 99.9% | Multi-AZ deployment |
| **Data Durability** | 99.999999999% | S3 + Cross-region |

---

## Quick Start

### Verify Backup Status

```bash
# Check all backup systems
./scripts/backup-verify.sh --verbose

# Check RDS backup status
aws rds describe-db-snapshots \
    --db-instance-identifier greenlang-postgres \
    --query 'DBSnapshots[-1].[DBSnapshotIdentifier,Status,SnapshotCreateTime]'

# Check Velero backups
kubectl get backups -n velero --sort-by=.metadata.creationTimestamp

# Check backup storage location
kubectl get backupstoragelocation -n velero
```

### Perform Manual Backup

```bash
# Create Velero backup
velero backup create manual-backup-$(date +%Y%m%d) \
    --include-namespaces greenlang \
    --wait

# Create RDS snapshot
aws rds create-db-snapshot \
    --db-instance-identifier greenlang-postgres \
    --db-snapshot-identifier manual-snapshot-$(date +%Y%m%d)
```

### Test Restore

```bash
# Run automated restore test (non-destructive)
./scripts/restore-test.sh --type full --dry-run

# Run actual restore test
./scripts/restore-test.sh --type rds
```

---

## Directory Structure

```
deployment/backup/
|
+-- README.md                    # This file
|
+-- velero/                      # Kubernetes backup configuration
|   +-- velero-install.yaml      # Velero deployment manifests
|   +-- backup-schedule.yaml     # Automated backup schedules
|   +-- backup-storage-location.yaml  # S3 backend configuration
|   +-- volume-snapshot-location.yaml # EBS snapshot configuration
|
+-- database/                    # Database backup configuration
|   +-- rds-backup-policy.tf     # Terraform for RDS backup automation
|   +-- point-in-time-recovery.md    # PITR documentation
|   +-- restore-procedures.md    # Database restore runbook
|
+-- scripts/                     # Operational scripts
|   +-- backup-verify.sh         # Verify backup integrity
|   +-- restore-test.sh          # Test restore procedures
|   +-- dr-failover.sh           # Disaster recovery failover
```

---

## Backup Schedules

### Automated Backup Schedule

```
+----------+------------+--------+------------+------------------+
| Backup   | Schedule   | TTL    | Scope      | Storage          |
+----------+------------+--------+------------+------------------+
| Hourly   | 0 * * * *  | 24h    | Critical   | S3 Standard      |
| Daily    | 0 2 * * *  | 7d     | Full NS    | S3 Standard      |
| Weekly   | 0 3 * * 0  | 30d    | All NS     | S3 Standard      |
| Monthly  | 0 4 1 * *  | 365d   | Complete   | S3 Glacier       |
+----------+------------+--------+------------+------------------+

RDS Automated:
- Daily snapshots: 03:00-04:00 UTC
- PITR: Continuous (5-minute intervals)
- Cross-region: Continuous replication
```

### Backup Windows (UTC)

```
00:00 |                                                          |
01:00 | [Config Backup]                                          |
02:00 | [Daily Velero Backup]                                    |
03:00 | [RDS Automated Snapshot] [Weekly Velero - Sundays]       |
04:00 | [Monthly Velero - 1st] [RDS Maintenance Window]          |
05:00 | [Cleanup Jobs]                                           |
...   |                                                          |
```

---

## Restore Procedures

### Quick Reference

| Scenario | Procedure | Estimated Time |
|----------|-----------|----------------|
| Accidental table deletion | [Partial Restore](#partial-restore) | 30-60 min |
| Data corruption | [PITR Restore](#pitr-restore) | 1-2 hours |
| Complete database failure | [Full DB Restore](#full-db-restore) | 1-2 hours |
| Kubernetes namespace loss | [Velero Restore](#velero-restore) | 15-30 min |
| Region failure | [DR Failover](#dr-failover) | 2-4 hours |

### PITR Restore

```bash
# Restore to specific point in time
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier greenlang-postgres \
    --target-db-instance-identifier greenlang-postgres-restored \
    --restore-time "2026-02-03T10:30:00Z"
```

### Velero Restore

```bash
# Restore from latest backup
velero restore create --from-backup greenlang-daily-backup-YYYYMMDD

# Restore specific namespace
velero restore create --from-backup greenlang-weekly-backup-YYYYMMDD \
    --include-namespaces greenlang
```

### Full Procedures

See detailed restore procedures in:
- [Point-in-Time Recovery Guide](./database/point-in-time-recovery.md)
- [Database Restore Runbook](./database/restore-procedures.md)

---

## Disaster Recovery

### DR Failover Process

```
1. Detection (0-5 min)
   - Automated monitoring alerts
   - Manual verification of outage

2. Decision (5-15 min)
   - Assess impact and scope
   - Confirm DR activation

3. Failover Execution (30-120 min)
   - Database: Create from cross-region backup
   - DNS: Update to DR region
   - Kubernetes: Deploy to DR cluster

4. Validation (15-30 min)
   - Verify data integrity
   - Test application functionality
   - Confirm traffic routing

5. Communication
   - Update status page
   - Notify stakeholders
```

### Execute DR Failover

```bash
# DR Drill (non-production)
./scripts/dr-failover.sh --mode drill

# Production Failover (requires confirmation)
./scripts/dr-failover.sh --mode failover --confirm
```

### Failback Process

After the primary region is restored:

1. Verify primary region health
2. Sync data from DR to primary
3. Update DNS to primary
4. Verify application in primary
5. Decommission DR resources

---

## Testing and Validation

### Backup Verification

```bash
# Daily automated verification
./scripts/backup-verify.sh --notify

# Manual verification with verbose output
./scripts/backup-verify.sh --verbose --json > report.json
```

### Restore Testing Schedule

| Test | Frequency | Duration | Owner |
|------|-----------|----------|-------|
| Backup integrity check | Daily | 5 min | Automated |
| RDS PITR test | Monthly | 2 hours | DBA |
| Velero restore test | Monthly | 1 hour | Platform |
| Full DR drill | Quarterly | 4 hours | All teams |

### DR Drill Checklist

```markdown
## Pre-Drill
- [ ] Schedule maintenance window
- [ ] Notify stakeholders
- [ ] Prepare test environment
- [ ] Review procedures

## During Drill
- [ ] Execute failover script
- [ ] Monitor progress
- [ ] Document timeline
- [ ] Test application

## Post-Drill
- [ ] Validate data integrity
- [ ] Measure RTO/RPO
- [ ] Clean up resources
- [ ] Document findings
- [ ] Update procedures
```

---

## Monitoring and Alerts

### Key Metrics

| Metric | Threshold | Alert |
|--------|-----------|-------|
| Backup age | > 25 hours | Critical |
| Backup success rate | < 90% | Warning |
| Storage location available | false | Critical |
| PITR lag | > 60 min | Warning |
| Cross-region replication lag | > 1 hour | Warning |

### Alert Channels

- **Slack**: #platform-alerts
- **PagerDuty**: backup-failure
- **Email**: platform-oncall@greenlang.io

### Prometheus Alerts

Backup alerts are configured in Velero and can be viewed:

```bash
# Check Prometheus rules
kubectl get prometheusrule -n velero

# View alert status
kubectl port-forward svc/prometheus -n monitoring 9090:9090
# Open http://localhost:9090/alerts
```

---

## Compliance

### Data Retention Requirements

| Regulation | Requirement | Implementation |
|------------|-------------|----------------|
| GDPR | Right to erasure | Data deletion procedures |
| SOC 2 | 90-day retention | Weekly backups (90 days) |
| ISO 27001 | Backup testing | Monthly restore tests |
| Internal | 1-year archive | Monthly backups to Glacier |

### Audit Trail

All backup operations are logged:
- CloudWatch Logs (backup operations)
- S3 access logs (backup retrieval)
- CloudTrail (API calls)

### Encryption

| Component | Encryption |
|-----------|------------|
| RDS snapshots | AWS KMS (AES-256) |
| S3 backups | SSE-KMS |
| In-transit | TLS 1.3 |
| Cross-region | KMS multi-region keys |

---

## Support

### Contact

- **Platform Team**: platform@greenlang.io
- **DBA On-Call**: dba-oncall@greenlang.io
- **Security Team**: security@greenlang.io

### Escalation

1. On-call engineer (PagerDuty)
2. Platform team lead
3. Engineering director
4. CTO (critical incidents)

### Related Documentation

- [Platform Disaster Recovery Strategy](../platform-disaster-recovery.md)
- [Deployment Guide](../DEPLOYMENT_GUIDE.md)
- [Security Audit](../security/README.md)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-03 | Initial release |

---

**Document Owner:** Platform Engineering Team
**Review Cycle:** Monthly
**Next Review:** 2026-03-03
