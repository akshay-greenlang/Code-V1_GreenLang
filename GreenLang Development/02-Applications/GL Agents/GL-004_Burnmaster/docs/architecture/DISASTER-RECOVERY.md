# GL-004 Burnmaster - Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-004 |
| Agent Name | Burnmaster (Burner Optimization Agent) |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |

---

## 1. Recovery Objectives

| Objective | Target | Maximum |
|-----------|--------|---------|
| RTO | 10 minutes | 30 minutes |
| RPO | 5 minutes | 15 minutes |
| MTD | 30 minutes | 2 hours |

---

## 2. Backup Procedures

### 2.1 Backup Schedule

| Data Type | Frequency | Retention |
|-----------|-----------|-----------|
| PostgreSQL | Continuous WAL | 30 days |
| Redis | Every 5 min | 7 days |
| Emissions Data | Hourly | 7 years |
| Configuration | On change | Unlimited |

### 2.2 Backup Script

```bash
#!/bin/bash
# GL-004 Burnmaster Backup

AGENT_ID="gl-004"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="s3://greenlang-backups/${AGENT_ID}"

# PostgreSQL
kubectl exec -it gl-004-postgres-0 -n greenlang -- \
    pg_basebackup -D /tmp/backup -Ft -z -P

aws s3 cp /tmp/backup ${BACKUP_BUCKET}/postgresql/${TIMESTAMP}/

# Redis
kubectl exec -it gl-004-redis-0 -n greenlang -- redis-cli BGSAVE
kubectl cp greenlang/gl-004-redis-0:/data/dump.rdb /tmp/redis_${TIMESTAMP}.rdb
aws s3 cp /tmp/redis_${TIMESTAMP}.rdb ${BACKUP_BUCKET}/redis/${TIMESTAMP}/

# Emissions data archive
kubectl exec -it gl-004-burnmaster-0 -n greenlang -- \
    python -c "from climate.emissions import archive_reports; archive_reports('/tmp/emissions')"
aws s3 sync /tmp/emissions ${BACKUP_BUCKET}/emissions/${TIMESTAMP}/
```

---

## 3. Recovery Procedures

### 3.1 Regional Failover

```bash
#!/bin/bash
# GL-004 Regional Failover

# 1. Verify primary unavailable
./verify-region.sh us-east-1

# 2. Activate DR
kubectl config use-context eks-us-west-2
kubectl scale deployment gl-004-burnmaster --replicas=4 -n greenlang

# 3. Promote PostgreSQL
kubectl exec -it gl-004-postgres-0 -n greenlang -- patronictl failover

# 4. Update DNS
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch file://dns-failover.json

# 5. Verify
curl -f https://gl-004.greenlang.io/health
```

### 3.2 Data Restoration

```bash
#!/bin/bash
# GL-004 Restore from Backup

RESTORE_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}
BACKUP_BUCKET="s3://greenlang-backups/gl-004"

kubectl scale deployment gl-004-burnmaster --replicas=0 -n greenlang

aws s3 cp ${BACKUP_BUCKET}/postgresql/${RESTORE_DATE}/ /tmp/restore/ --recursive
kubectl exec -it gl-004-postgres-0 -n greenlang -- \
    pg_restore -d greenlang_gl004 -c /tmp/restore/*.tar.gz

kubectl scale deployment gl-004-burnmaster --replicas=2 -n greenlang
```

---

## 4. Communication Plan

| Time | Action | Contact |
|------|--------|---------|
| 0-5 min | Acknowledge | On-call SRE |
| 5-15 min | Escalate | SRE Lead |
| 15-30 min | Management | Director |

---

## 5. DR Testing

| Test | Frequency |
|------|-----------|
| Backup Verification | Daily |
| Failover Test | Monthly |
| Full DR Test | Quarterly |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-22 | Initial release |
