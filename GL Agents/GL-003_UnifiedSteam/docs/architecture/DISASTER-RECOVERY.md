# GL-003 UnifiedSteam - Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-003 |
| Agent Name | UnifiedSteam (Steam System Optimizer) |
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
| InfluxDB (Telemetry) | Hourly | 90 days |
| Trap Survey Data | Daily | 7 years |
| Configuration | On change | Unlimited |

### 2.2 Backup Script

```bash
#!/bin/bash
# GL-003 UnifiedSteam Backup

AGENT_ID="gl-003"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="s3://greenlang-backups/${AGENT_ID}"

# PostgreSQL
kubectl exec -it gl-003-postgres-0 -n greenlang -- \
    pg_basebackup -D /tmp/backup -Ft -z -P

aws s3 cp /tmp/backup ${BACKUP_BUCKET}/postgresql/${TIMESTAMP}/

# InfluxDB
kubectl exec -it gl-003-influxdb-0 -n greenlang -- \
    influx backup /tmp/influx-backup

aws s3 sync /tmp/influx-backup ${BACKUP_BUCKET}/influxdb/${TIMESTAMP}/

# Steam Trap Survey Data
kubectl exec -it gl-003-unifiedsteam-0 -n greenlang -- \
    python -c "from audit.evidence_pack import export_surveys; export_surveys('/tmp/surveys')"

aws s3 sync /tmp/surveys ${BACKUP_BUCKET}/surveys/${TIMESTAMP}/
```

---

## 3. Recovery Procedures

### 3.1 Regional Failover

```bash
#!/bin/bash
# GL-003 Regional Failover

# 1. Activate DR site
kubectl config use-context eks-us-west-2
kubectl scale deployment gl-003-unifiedsteam --replicas=4 -n greenlang

# 2. Promote PostgreSQL
kubectl exec -it gl-003-postgres-0 -n greenlang -- patronictl failover

# 3. Update DNS
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch file://dns-failover.json

# 4. Verify
curl -f https://gl-003.greenlang.io/api/v1/health
```

### 3.2 Data Restoration

```bash
#!/bin/bash
# GL-003 Restore from Backup

RESTORE_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}
BACKUP_BUCKET="s3://greenlang-backups/gl-003"

# Stop application
kubectl scale deployment gl-003-unifiedsteam --replicas=0 -n greenlang

# Restore PostgreSQL
aws s3 cp ${BACKUP_BUCKET}/postgresql/${RESTORE_DATE}/ /tmp/restore/ --recursive
kubectl cp /tmp/restore/ greenlang/gl-003-postgres-0:/tmp/restore/
kubectl exec -it gl-003-postgres-0 -n greenlang -- \
    pg_restore -d greenlang_gl003 -c /tmp/restore/*.tar.gz

# Restore InfluxDB
aws s3 sync ${BACKUP_BUCKET}/influxdb/${RESTORE_DATE}/ /tmp/influx-restore/
kubectl cp /tmp/influx-restore/ greenlang/gl-003-influxdb-0:/tmp/restore/
kubectl exec -it gl-003-influxdb-0 -n greenlang -- \
    influx restore /tmp/restore

# Start application
kubectl scale deployment gl-003-unifiedsteam --replicas=2 -n greenlang
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
