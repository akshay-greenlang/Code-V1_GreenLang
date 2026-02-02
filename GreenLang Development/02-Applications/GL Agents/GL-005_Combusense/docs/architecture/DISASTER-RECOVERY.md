# GL-005 Combusense - Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | GL-005 |
| Agent Name | Combusense (Combustion Control & Sensing Agent) |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |

---

## 1. Recovery Objectives

| Objective | Target | Maximum |
|-----------|--------|---------|
| RTO | 5 minutes | 15 minutes |
| RPO | 1 minute | 5 minutes |
| MTD | 15 minutes | 1 hour |
| Control Loop Recovery | 50ms | 1 second |

---

## 2. Backup Procedures

### 2.1 Backup Schedule

| Data Type | Frequency | Retention |
|-----------|-----------|-----------|
| PostgreSQL | Continuous WAL | 30 days |
| InfluxDB (Metrics) | Hourly | 90 days |
| PID Tuning Parameters | On change | Unlimited |
| Sensor Calibration | Daily | 1 year |
| CQI History | Continuous | 7 years |
| Configuration | On change | Unlimited |

### 2.2 Backup Script

```bash
#!/bin/bash
# GL-005 Combusense Backup

AGENT_ID="gl-005"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_BUCKET="s3://greenlang-backups/${AGENT_ID}"

# PostgreSQL
kubectl exec -it gl-005-postgres-0 -n greenlang -- \
    pg_basebackup -D /tmp/backup -Ft -z -P
aws s3 cp /tmp/backup ${BACKUP_BUCKET}/postgresql/${TIMESTAMP}/

# InfluxDB
kubectl exec -it gl-005-influxdb-0 -n greenlang -- \
    influx backup /tmp/influx-backup
aws s3 sync /tmp/influx-backup ${BACKUP_BUCKET}/influxdb/${TIMESTAMP}/

# PID Parameters
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl localhost:8080/api/v1/controller/parameters > /tmp/pid_params.json
aws s3 cp /tmp/pid_params.json ${BACKUP_BUCKET}/config/${TIMESTAMP}/

# Sensor Calibration
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl localhost:8080/api/v1/sensors/calibration > /tmp/calibration.json
aws s3 cp /tmp/calibration.json ${BACKUP_BUCKET}/config/${TIMESTAMP}/
```

---

## 3. Recovery Procedures

### 3.1 Controller Failover (Automatic)

The control system performs automatic bumpless transfer:

1. Watchdog detects primary controller failure (< 10ms)
2. Standby controller verifies ready state (< 5ms)
3. State transfer including integral term (< 10ms)
4. Standby becomes active (< 5ms)
5. Output tracking confirmed (< 20ms)

Total failover time: < 50ms

### 3.2 Regional Failover

```bash
#!/bin/bash
# GL-005 Regional Failover

# 1. Verify primary unavailable
./verify-region.sh us-east-1

# 2. Activate DR
kubectl config use-context eks-us-west-2
kubectl scale deployment gl-005-combusense --replicas=4 -n greenlang

# 3. Promote PostgreSQL
kubectl exec -it gl-005-postgres-0 -n greenlang -- patronictl failover

# 4. Restore PID parameters
aws s3 cp s3://greenlang-backups/gl-005/config/latest/pid_params.json /tmp/
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl -X POST -d @/tmp/pid_params.json localhost:8080/api/v1/controller/parameters

# 5. Update DNS
aws route53 change-resource-record-sets \
    --hosted-zone-id ${ZONE_ID} \
    --change-batch file://dns-failover.json

# 6. Verify control loop
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl localhost:8080/api/v1/controller/status
```

### 3.3 Data Restoration

```bash
#!/bin/bash
# GL-005 Restore from Backup

RESTORE_DATE=${1:-$(date -d "yesterday" +%Y-%m-%d)}
BACKUP_BUCKET="s3://greenlang-backups/gl-005"

# Stop application
kubectl scale deployment gl-005-combusense --replicas=0 -n greenlang

# Restore PostgreSQL
aws s3 cp ${BACKUP_BUCKET}/postgresql/${RESTORE_DATE}/ /tmp/restore/ --recursive
kubectl exec -it gl-005-postgres-0 -n greenlang -- \
    pg_restore -d greenlang_gl005 -c /tmp/restore/*.tar.gz

# Restore InfluxDB
aws s3 sync ${BACKUP_BUCKET}/influxdb/${RESTORE_DATE}/ /tmp/influx-restore/
kubectl exec -it gl-005-influxdb-0 -n greenlang -- \
    influx restore /tmp/influx-restore

# Start application
kubectl scale deployment gl-005-combusense --replicas=2 -n greenlang

# Restore PID parameters
aws s3 cp ${BACKUP_BUCKET}/config/${RESTORE_DATE}/pid_params.json /tmp/
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl -X POST -d @/tmp/pid_params.json localhost:8080/api/v1/controller/parameters

# Verify
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl localhost:8080/api/v1/health
```

---

## 4. Communication Plan

| Time | Action | Contact |
|------|--------|---------|
| 0-5 min | Acknowledge | On-call SRE |
| 5-15 min | Escalate | SRE Lead, Controls Engineer |
| 15-30 min | Management | Director |
| 30+ min | Executive | VP Engineering |

### 4.1 Safety Critical Communication

For any combustion control failure:

1. Immediately notify plant operations
2. Confirm safe state achieved
3. Document incident in safety log
4. Root cause analysis within 24 hours

---

## 5. DR Testing

| Test | Frequency | Duration |
|------|-----------|----------|
| Controller Failover | Weekly | 5 min |
| Backup Verification | Daily | 15 min |
| Database Failover | Monthly | 30 min |
| Regional Failover | Quarterly | 2 hours |
| Full DR Test | Semi-annually | 4 hours |

### 5.1 Controller Failover Test

```bash
#!/bin/bash
# Weekly controller failover test

# 1. Record current state
PRE_STATE=$(kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl -s localhost:8080/api/v1/controller/state)

# 2. Trigger failover
kubectl exec -it gl-005-combusense-0 -n greenlang -- \
    curl -X POST localhost:8080/api/v1/controller/test-failover

# 3. Verify bumpless transfer
POST_STATE=$(kubectl exec -it gl-005-combusense-1 -n greenlang -- \
    curl -s localhost:8080/api/v1/controller/state)

# 4. Compare outputs (should be continuous)
echo "Pre-failover output: $(echo $PRE_STATE | jq .output)"
echo "Post-failover output: $(echo $POST_STATE | jq .output)"

# 5. Failback
kubectl exec -it gl-005-combusense-1 -n greenlang -- \
    curl -X POST localhost:8080/api/v1/controller/transfer
```

---

## 6. Safety Considerations

### 6.1 Safe State Definition

When control system is unavailable:
- Fuel valve: Hold last position (with limits)
- Air damper: Hold last position
- Flame scanner: Continue monitoring via DCS
- Interlocks: Remain active (hardware-based)

### 6.2 Recovery Verification

Before returning to automatic control:
1. Verify all sensors responding
2. Confirm PID parameters loaded
3. Check interlock status
4. Verify communication with DCS
5. Manual bump test before auto mode

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-22 | Initial release |
