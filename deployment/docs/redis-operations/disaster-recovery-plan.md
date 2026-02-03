# Redis Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | DevOps Team |
| Review Frequency | Quarterly |

---

## 1. Executive Summary

This document outlines the disaster recovery (DR) plan for GreenLang's Redis infrastructure. It defines recovery objectives, scenarios, procedures, and communication protocols to ensure business continuity during Redis-related incidents.

### Recovery Objectives

| Objective | Target | Description |
|-----------|--------|-------------|
| **RTO (Recovery Time Objective)** | 5 minutes | Maximum acceptable downtime for Redis services |
| **RPO (Recovery Point Objective)** | 0 seconds | Zero data loss with synchronous replication |

---

## 2. Infrastructure Overview

### Redis Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │              Redis Sentinel Cluster             │
                    │                                                 │
    ┌───────────────┼───────────────┬───────────────┬────────────────┤
    │               │               │               │                │
    ▼               ▼               ▼               ▼                ▼
┌────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│Sentinel│    │ Sentinel │    │ Sentinel │    │ Sentinel │    │ Sentinel │
│  (1)   │    │   (2)    │    │   (3)    │    │   (4)    │    │   (5)    │
└────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
    │               │               │               │                │
    └───────────────┴───────────────┼───────────────┴────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │   Master    │ │   Replica   │ │   Replica   │
            │ (Primary)   │ │    (AZ-B)   │ │    (AZ-C)   │
            │   AZ-A      │ │             │ │             │
            └─────────────┘ └─────────────┘ └─────────────┘
                    │               ▲               ▲
                    │               │               │
                    └───────────────┴───────────────┘
                      Synchronous Replication
```

### Component Inventory

| Component | Count | Purpose | Location |
|-----------|-------|---------|----------|
| Redis Master | 1 | Primary read/write | AZ-A |
| Redis Replicas | 2 | Read replicas, failover candidates | AZ-B, AZ-C |
| Sentinel Nodes | 5 | Monitoring and automatic failover | Distributed across AZs |

### Network Configuration

| Component | DNS Endpoint | Port | Protocol |
|-----------|--------------|------|----------|
| Redis Master | redis-master.greenlang.internal | 6379 | TCP |
| Redis Replica 1 | redis-replica-1.greenlang.internal | 6379 | TCP |
| Redis Replica 2 | redis-replica-2.greenlang.internal | 6379 | TCP |
| Sentinel | redis-sentinel.greenlang.internal | 26379 | TCP |

---

## 3. Disaster Recovery Scenarios

### Scenario 1: Master Node Failure

**Description**: The Redis master node becomes unavailable due to hardware failure, software crash, or network isolation.

**Impact Level**: HIGH
**Expected RTO**: < 30 seconds (automatic)
**RPO**: 0 seconds (synchronous replication)

#### Automatic Recovery (Sentinel Failover)

1. Sentinel detects master is unreachable (down-after-milliseconds: 5000ms)
2. Sentinel initiates failover after quorum agreement
3. Best replica is promoted to master
4. Remaining replicas reconfigure to new master
5. Applications reconnect via Sentinel

#### Manual Intervention Triggers

- Sentinel failover does not complete within 60 seconds
- Multiple nodes report inconsistent states
- Network partition prevents quorum

#### Recovery Steps (if manual intervention required)

```bash
# 1. Assess situation
redis-cli -h sentinel-host -p 26379 SENTINEL master mymaster
redis-cli -h sentinel-host -p 26379 SENTINEL slaves mymaster

# 2. Force failover if needed
redis-cli -h sentinel-host -p 26379 SENTINEL FAILOVER mymaster

# 3. Verify new master
redis-cli -h sentinel-host -p 26379 SENTINEL get-master-addr-by-name mymaster

# 4. Check replication status
redis-cli -h new-master -p 6379 INFO replication
```

---

### Scenario 2: Availability Zone Failure

**Description**: An entire AWS Availability Zone becomes unavailable, affecting Redis nodes hosted there.

**Impact Level**: HIGH
**Expected RTO**: < 2 minutes
**RPO**: 0 seconds

#### Recovery Procedure

**Phase 1: Immediate Assessment (0-30 seconds)**

```bash
# Check which nodes are affected
for host in redis-master redis-replica-1 redis-replica-2; do
  echo "Checking $host..."
  redis-cli -h $host.greenlang.internal -p 6379 PING 2>/dev/null || echo "$host is DOWN"
done

# Check Sentinel quorum
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL CKQUORUM mymaster
```

**Phase 2: Failover Execution (30-60 seconds)**

If master is in failed AZ:
```bash
# Sentinel should auto-failover, but verify
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL FAILOVER mymaster

# Monitor failover progress
watch -n 1 'redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL master mymaster | grep -E "(ip|port|flags)"'
```

**Phase 3: Capacity Restoration (1-5 minutes)**

```bash
# Launch replacement nodes in surviving AZs
terraform apply -var="redis_replica_count=3" -target=module.redis

# Configure new replica
redis-cli -h new-replica -p 6379 REPLICAOF new-master 6379
```

**Phase 4: Validation**

```bash
# Verify cluster health
./redis-health-check.sh

# Test application connectivity
curl -f https://api.greenlang.io/api/v1/health
```

---

### Scenario 3: Region Failure

**Description**: Complete AWS region becomes unavailable, affecting all Redis infrastructure.

**Impact Level**: CRITICAL
**Expected RTO**: < 15 minutes
**RPO**: < 1 minute (cross-region async replication)

#### Prerequisites

- Cross-region replication configured to DR region
- DR region infrastructure pre-provisioned (warm standby)
- DNS failover mechanism in place

#### Recovery Procedure

**Phase 1: Decision Point (0-2 minutes)**

```
DR ACTIVATION DECISION TREE:
                    ┌─────────────────────────┐
                    │ Primary Region Failure  │
                    │      Detected?          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ AWS Status Confirms     │
                    │   Region Outage?        │
                    └───────────┬─────────────┘
                          Yes   │   No
                    ┌───────────▼───────────┐
                    │                       │
              ┌─────▼─────┐          ┌──────▼──────┐
              │ Activate  │          │ Wait & Re-  │
              │    DR     │          │   assess    │
              └───────────┘          └─────────────┘
```

**Phase 2: DR Region Activation (2-10 minutes)**

```bash
# 1. Verify DR region Redis cluster
export AWS_REGION=us-west-2
redis-cli -h redis-dr.greenlang.internal -p 6379 PING

# 2. Promote DR cluster to primary
redis-cli -h redis-dr.greenlang.internal -p 6379 REPLICAOF NO ONE

# 3. Update Route53 DNS
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch file://dns-failover.json

# 4. Update application configuration
kubectl set env deployment/greenlang-app \
  REDIS_HOST=redis-dr.greenlang.internal \
  -n greenlang

# 5. Restart applications to pick up new config
kubectl rollout restart deployment/greenlang-app -n greenlang
```

**Phase 3: Validation (10-15 minutes)**

```bash
# Verify application health
curl -f https://api.greenlang.io/api/v1/health

# Check Redis connectivity from applications
kubectl exec -it $(kubectl get pod -l app=greenlang-app -o jsonpath='{.items[0].metadata.name}' -n greenlang) \
  -n greenlang -- redis-cli -h $REDIS_HOST PING
```

---

## 4. Recovery Procedures

### 4.1 RDB Snapshot Restore

Use when you need to restore Redis to a specific point in time from RDB backup.

```bash
# 1. Stop Redis service
sudo systemctl stop redis

# 2. Backup current data
sudo mv /var/lib/redis/dump.rdb /var/lib/redis/dump.rdb.bak

# 3. Download RDB from S3
aws s3 cp s3://greenlang-backups/redis/dump-YYYYMMDD-HHMMSS.rdb /var/lib/redis/dump.rdb

# 4. Set correct permissions
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo chmod 640 /var/lib/redis/dump.rdb

# 5. Start Redis
sudo systemctl start redis

# 6. Verify data
redis-cli DBSIZE
redis-cli INFO persistence
```

### 4.2 AOF Recovery

Use when RDB is corrupted but AOF is available.

```bash
# 1. Check AOF validity
redis-check-aof --fix /var/lib/redis/appendonly.aof

# 2. If truncation needed, create backup first
cp /var/lib/redis/appendonly.aof /var/lib/redis/appendonly.aof.bak

# 3. Apply fix
redis-check-aof --fix /var/lib/redis/appendonly.aof

# 4. Start Redis with AOF only
redis-server --appendonly yes --dbfilename ""
```

### 4.3 Partial Data Recovery

Use when specific keys need to be recovered from backup.

```bash
# 1. Start temporary Redis instance with backup
docker run -d --name redis-recovery \
  -v /path/to/backup/dump.rdb:/data/dump.rdb \
  redis:7-alpine

# 2. Extract specific keys
redis-cli -h localhost -p 6380 --scan --pattern "user:*" | \
  xargs -I {} redis-cli -h localhost -p 6380 DUMP {} > keys_dump.txt

# 3. Restore to production
cat keys_dump.txt | while read key data; do
  redis-cli -h redis-master -p 6379 RESTORE "$key" 0 "$data"
done

# 4. Cleanup
docker stop redis-recovery && docker rm redis-recovery
```

---

## 5. Communication Plan

### Escalation Matrix

| Severity | Response Time | Primary Contact | Escalation |
|----------|---------------|-----------------|------------|
| P1 (Critical) | 5 minutes | On-call Engineer | VP Engineering |
| P2 (High) | 15 minutes | On-call Engineer | Engineering Manager |
| P3 (Medium) | 1 hour | On-call Engineer | Team Lead |
| P4 (Low) | 4 hours | On-call Engineer | - |

### Notification Templates

#### Initial Incident Notification

```
[INCIDENT] Redis Service Degradation

Severity: P1
Status: Investigating
Impact: [Description of user impact]
Started: [Timestamp]

Current Status:
- [Brief description of issue]
- [Actions being taken]

Next Update: [Time]

Incident Commander: [Name]
```

#### Failover Notification

```
[UPDATE] Redis Failover Executed

Previous Master: [IP/hostname]
New Master: [IP/hostname]
Failover Type: [Automatic/Manual]
Duration: [Time in seconds]

Impact:
- Brief connection interruption (~5 seconds)
- No data loss confirmed

Actions Required:
- Monitor application reconnection
- Verify data integrity

Next Update: [Time or "As needed"]
```

#### Resolution Notification

```
[RESOLVED] Redis Service Restored

Duration: [Total incident time]
Root Cause: [Brief description]
Resolution: [Actions taken]

Impact Summary:
- Affected Users: [Number/percentage]
- Data Loss: [None/Description]
- SLA Impact: [Yes/No]

Post-Incident:
- RCA scheduled for [Date]
- Follow-up ticket: [Link]
```

### Communication Channels

| Channel | Purpose | Audience |
|---------|---------|----------|
| #incident-redis | Real-time coordination | Engineering team |
| PagerDuty | On-call alerting | On-call engineers |
| Statuspage | Customer communication | External users |
| Email | Stakeholder updates | Management, affected teams |

---

## 6. Testing and Validation

### DR Test Schedule

| Test Type | Frequency | Duration | Scope |
|-----------|-----------|----------|-------|
| Failover Drill | Monthly | 30 minutes | Single node failover |
| AZ Failure Simulation | Quarterly | 2 hours | Multi-node failover |
| Region Failover | Annually | 4 hours | Full DR activation |

### Test Procedure Checklist

```
PRE-TEST:
[ ] Notify stakeholders
[ ] Confirm maintenance window
[ ] Verify backup integrity
[ ] Document current state

EXECUTION:
[ ] Execute failover scenario
[ ] Monitor automatic recovery
[ ] Verify data integrity
[ ] Test application connectivity
[ ] Measure actual RTO/RPO

POST-TEST:
[ ] Restore original configuration
[ ] Document lessons learned
[ ] Update runbooks as needed
[ ] File test results
```

---

## 7. Maintenance and Updates

### Document Review

- This DR plan must be reviewed quarterly
- Updates required after any significant infrastructure changes
- All changes must be version controlled and approved

### Related Documents

| Document | Location |
|----------|----------|
| Failover Runbook | [failover-runbook.md](failover-runbook.md) |
| Sentinel Operations | [sentinel-operations.md](sentinel-operations.md) |
| Data Recovery | [data-recovery.md](data-recovery.md) |
| Performance Tuning | [performance-tuning.md](performance-tuning.md) |

---

## Appendix A: Emergency Contacts

| Role | Name | Phone | Email |
|------|------|-------|-------|
| Primary On-Call | Rotation | PagerDuty | oncall@greenlang.io |
| Database Lead | [Name] | [Phone] | db-lead@greenlang.io |
| VP Engineering | [Name] | [Phone] | vp-eng@greenlang.io |
| AWS Support | - | - | Enterprise Support Portal |

## Appendix B: Recovery Verification Checklist

```
POST-RECOVERY VALIDATION:

[ ] Redis master responding to PING
[ ] All replicas connected and synced
[ ] Sentinel quorum established
[ ] Application health checks passing
[ ] No error spikes in logs
[ ] Metrics showing normal patterns
[ ] Data integrity spot checks passed
[ ] End-to-end tests successful
```
