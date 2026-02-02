# GL-002 Flameguard - Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-002 |
| Agent Name | Flameguard (Boiler Efficiency Optimizer) |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Review Cycle | Quarterly |

---

## 1. Purpose and Scope

### 1.1 Purpose

This Disaster Recovery Plan provides procedures for recovering GL-002 Flameguard following a disaster event, ensuring continued boiler efficiency optimization and safety compliance (NFPA 85, ASME PTC 4.1).

### 1.2 Recovery Objectives

| Objective | Target | Maximum |
|-----------|--------|---------|
| RTO | 10 minutes | 30 minutes |
| RPO | 5 minutes | 15 minutes |
| MTD | 30 minutes | 2 hours |

---

## 2. Backup Procedures

### 2.1 Backup Schedule

| Data Type | Method | Frequency | Retention |
|-----------|--------|-----------|-----------|
| PostgreSQL | WAL Streaming | Continuous | 30 days |
| PostgreSQL Full | pg_basebackup | Daily | 30 days |
| Redis | RDB Snapshot | Every 5 min | 7 days |
| Configuration | GitOps | On change | Unlimited |
| Efficiency Calculations | Event Store | Continuous | 7 years |
| Audit Logs | Immutable Append | Continuous | 7 years |

### 2.2 Backup Script

```bash
#!/bin/bash
# GL-002 Flameguard Backup Script
# Runs daily at 02:00 UTC

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="s3://greenlang-backups/gl-002"
NAMESPACE="greenlang"

# PostgreSQL backup
kubectl exec -it postgresql-0 -n ${NAMESPACE} -- \
    pg_basebackup -D /tmp/backup -Ft -z -P

kubectl cp ${NAMESPACE}/postgresql-0:/tmp/backup \
    /tmp/postgresql_${TIMESTAMP}.tar.gz

aws s3 cp /tmp/postgresql_${TIMESTAMP}.tar.gz \
    ${BACKUP_BUCKET}/postgresql/full/

# Redis backup
kubectl exec -it redis-master-0 -n ${NAMESPACE} -- \
    redis-cli BGSAVE

sleep 10
kubectl cp ${NAMESPACE}/redis-master-0:/data/dump.rdb \
    /tmp/redis_${TIMESTAMP}.rdb

aws s3 cp /tmp/redis_${TIMESTAMP}.rdb \
    ${BACKUP_BUCKET}/redis/rdb/

echo "Backup completed: ${TIMESTAMP}"
```

---

## 3. Recovery Procedures

### 3.1 Disaster Classification

| Level | Description | Response |
|-------|-------------|----------|
| Level 1 | Single pod failure | Automatic restart |
| Level 2 | AZ failure | Automatic failover |
| Level 3 | Region failure | DR activation |
| Level 4 | Data corruption | Restore from backup |

### 3.2 Level 3: Region Failover

```bash
#!/bin/bash
# GL-002 Regional Failover

# 1. Verify primary unavailable
./verify-region.sh us-east-1
[ $? -eq 0 ] && echo "Primary healthy, aborting" && exit 1

# 2. Activate DR site
kubectl config use-context eks-us-west-2
kubectl scale deployment gl-002-flameguard --replicas=4 -n greenlang

# 3. Promote PostgreSQL
kubectl exec -it postgresql-0 -n greenlang -- patronictl failover

# 4. Update DNS
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch file://dns-failover.json

# 5. Verify
curl -f https://gl-002.greenlang.io/health
```

### 3.3 Restoration Script

```bash
#!/bin/bash
# GL-002 Restore from Backup

RESTORE_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}
BACKUP_BUCKET="s3://greenlang-backups/gl-002"

# Download backup
aws s3 cp ${BACKUP_BUCKET}/postgresql/full/${RESTORE_DATE}/ /tmp/restore/ --recursive

# Stop application
kubectl scale deployment gl-002-flameguard --replicas=0 -n greenlang

# Restore PostgreSQL
kubectl exec -it postgresql-0 -n greenlang -- \
    pg_restore -d greenlang_gl002 -c /tmp/restore/backup.tar

# Start application
kubectl scale deployment gl-002-flameguard --replicas=2 -n greenlang

# Verify
kubectl wait --for=condition=ready pod -l app=gl-002-flameguard -n greenlang
```

---

## 4. Communication Plan

### 4.1 Escalation Matrix

| Time | Action | Contacts |
|------|--------|----------|
| 0-5 min | Acknowledge | On-call SRE |
| 5-15 min | Escalate | SRE Lead, Safety Engineer |
| 15-30 min | Management | Director, Plant Ops |
| 30+ min | Executive | VP Engineering |

### 4.2 Notification Templates

**Internal Alert:**
```
[P1] GL-002 Flameguard - DR Activated
SEVERITY: Critical
IMPACT: Boiler efficiency optimization unavailable
STATUS: Failover in progress
ETA: 15 minutes
```

**Plant Operations:**
```
NOTICE: GL-002 Flameguard temporarily unavailable
Boilers operating in manual mode
Safety interlocks remain ACTIVE
Efficiency optimization will resume shortly
```

---

## 5. DR Testing

### 5.1 Test Schedule

| Test | Frequency | Duration |
|------|-----------|----------|
| Backup Verification | Daily | 15 min |
| Failover Test | Monthly | 1 hour |
| Full DR Test | Quarterly | 4 hours |

### 5.2 Test Scenarios

| Scenario | Steps | Success Criteria |
|----------|-------|------------------|
| Database Failover | Promote replica | RTO < 1 min |
| Application Recovery | Restore from backup | Data integrity verified |
| Regional Failover | Activate DR site | Service restored < 15 min |

---

## 6. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL DevOps | Initial release |
