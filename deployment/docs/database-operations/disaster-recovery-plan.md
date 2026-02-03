# PostgreSQL/TimescaleDB Disaster Recovery Plan

## Document Information

| Field | Value |
|-------|-------|
| Document Owner | Database Operations Team |
| Last Updated | 2026-02-03 |
| Review Cycle | Quarterly |
| Classification | Internal - Confidential |

---

## 1. Executive Summary

This document outlines the disaster recovery (DR) procedures for the GreenLang PostgreSQL/TimescaleDB database infrastructure. It defines recovery objectives, scenarios, and step-by-step procedures to restore database services during outages.

### Recovery Objectives

| Metric | Target | Description |
|--------|--------|-------------|
| **RTO (Recovery Time Objective)** | 30 minutes | Maximum acceptable downtime |
| **RPO (Recovery Point Objective)** | 0 (zero data loss) | Achieved via synchronous replication |

### Key Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| Primary DBA On-Call | dba-oncall@greenlang.io | PagerDuty |
| Secondary DBA | dba-team@greenlang.io | Slack #dba-alerts |
| Infrastructure Lead | infra-lead@greenlang.io | Phone |
| VP Engineering | vp-eng@greenlang.io | Emergency only |

---

## 2. Architecture Overview

### Database Topology

```
                    +-------------------+
                    |   Load Balancer   |
                    |   (HAProxy/PgPool)|
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+       +-----------v---------+
    |   Primary Node    |       |   Read Replicas     |
    |   (us-east-1a)    |       |   (us-east-1b/1c)   |
    |                   |       |                     |
    |  PostgreSQL 14    |       |  PostgreSQL 14      |
    |  TimescaleDB 2.x  |------>|  TimescaleDB 2.x    |
    |  Patroni          |  sync |  Patroni            |
    +--------+----------+       +---------------------+
             |
             | Streaming Replication
             v
    +-------------------+       +-------------------+
    |   DR Replica      |       |   Backup Storage  |
    |   (us-west-2a)    |       |   (S3/GCS)        |
    |                   |       |                   |
    |  Async Replica    |       |  - WAL Archives   |
    |  Hot Standby      |       |  - Base Backups   |
    +-------------------+       |  - pg_dump        |
                                +-------------------+
```

### Replication Configuration

| Type | Source | Target | Mode | Max Lag |
|------|--------|--------|------|---------|
| Synchronous | Primary | Sync Standby | synchronous_commit=on | 0 |
| Asynchronous | Primary | Async Replicas | streaming | < 1 minute |
| Cross-Region | Primary | DR Replica | streaming | < 5 minutes |
| WAL Archive | Primary | S3/GCS | archive_mode=on | < 1 minute |

---

## 3. Disaster Recovery Scenarios

### Scenario 1: Primary Instance Failure

**Description**: The primary PostgreSQL instance becomes unavailable due to hardware failure, process crash, or network isolation.

**Detection**:
- Patroni health check fails (3 consecutive failures)
- HAProxy marks primary as down
- PagerDuty alert triggered

**Impact**:
- Write operations fail
- Read operations continue on replicas
- Automatic failover initiates (if enabled)

**Recovery Procedure**:

```bash
# Step 1: Verify cluster status
patronictl -c /etc/patroni/patroni.yml list

# Step 2: If automatic failover did not occur, trigger manual failover
patronictl -c /etc/patroni/patroni.yml switchover --force

# Step 3: Verify new primary
patronictl -c /etc/patroni/patroni.yml list

# Step 4: Check replication status
SELECT client_addr, state, sent_lsn, write_lsn, flush_lsn, replay_lsn
FROM pg_stat_replication;

# Step 5: Verify application connectivity
psql -h ${DB_HOST} -U ${DB_USER} -c "SELECT pg_is_in_recovery();"
```

**RTO**: 2-5 minutes (automatic), 10-15 minutes (manual)
**RPO**: 0 (synchronous replica promoted)

---

### Scenario 2: Availability Zone Failure

**Description**: An entire AWS/GCP availability zone becomes unavailable, affecting all instances in that zone.

**Detection**:
- CloudWatch/Stackdriver AZ health alerts
- Multiple instance health checks fail simultaneously
- Network connectivity to AZ lost

**Impact**:
- If primary in failed AZ: write downtime until failover
- Reduced read capacity
- Potential impact on synchronous replication

**Recovery Procedure**:

```bash
# Step 1: Assess which nodes are affected
patronictl -c /etc/patroni/patroni.yml list
kubectl get pods -n database -o wide

# Step 2: If primary was in failed AZ, verify automatic failover
patronictl -c /etc/patroni/patroni.yml history

# Step 3: If no automatic failover, manually promote replica in healthy AZ
patronictl -c /etc/patroni/patroni.yml switchover \
  --candidate greenlang-db-1 \
  --force

# Step 4: Reconfigure synchronous replication
patronictl -c /etc/patroni/patroni.yml edit-config
# Set synchronous_mode_strict: false temporarily

# Step 5: Scale up new replicas in healthy AZs
kubectl scale statefulset greenlang-db --replicas=4 -n database

# Step 6: Once AZ recovers, rejoin old primary as replica
patronictl -c /etc/patroni/patroni.yml reinit greenlang-db-0
```

**RTO**: 5-15 minutes
**RPO**: 0-1 minute (depending on replication mode)

---

### Scenario 3: Region Failure

**Description**: An entire cloud region becomes unavailable, requiring activation of the DR site.

**Detection**:
- Region-wide service health dashboard shows outage
- All instances in region unreachable
- Cross-region replication lag increasing

**Impact**:
- Complete database unavailability in primary region
- Must activate DR site
- Potential data loss equal to async replication lag

**Recovery Procedure**:

```bash
# CRITICAL: This procedure activates the DR site
# Coordinate with Infrastructure and Application teams before proceeding

# Step 1: Confirm region failure (not just network partition)
# Check: https://status.aws.amazon.com or https://status.cloud.google.com

# Step 2: Connect to DR region infrastructure
export KUBECONFIG=/path/to/dr-region-kubeconfig
aws configure set region us-west-2

# Step 3: Check DR replica status
psql -h dr-replica.greenlang.internal -U admin -c "
SELECT pg_last_wal_receive_lsn(),
       pg_last_wal_replay_lsn(),
       pg_last_xact_replay_timestamp();
"

# Step 4: Promote DR replica to primary
# CAUTION: This breaks replication - coordinate with team
pg_ctl promote -D /var/lib/postgresql/data

# OR via Patroni if configured for cross-region
patronictl -c /etc/patroni/patroni.yml failover \
  --candidate greenlang-db-dr \
  --force

# Step 5: Update DNS to point to DR site
aws route53 change-resource-record-sets \
  --hosted-zone-id ${HOSTED_ZONE_ID} \
  --change-batch file://dns-failover-changeset.json

# Step 6: Verify DNS propagation
dig db.greenlang.io

# Step 7: Update application configuration
kubectl set env deployment/greenlang-api \
  DATABASE_URL=postgresql://user:pass@dr-db.greenlang.io:5432/greenlang \
  -n production

# Step 8: Restart application pods
kubectl rollout restart deployment/greenlang-api -n production

# Step 9: Verify application connectivity
curl -s https://api.greenlang.io/health | jq .
```

**Post-DR Recovery**:

```bash
# After primary region recovers:

# Step 1: Rebuild primary region as replica
pg_basebackup -h dr-primary.greenlang.internal \
  -D /var/lib/postgresql/data \
  -U replicator -v -P --wal-method=stream

# Step 2: Configure as standby
cat > /var/lib/postgresql/data/standby.signal << EOF
# Standby configuration
EOF

# Step 3: Plan switchback during maintenance window
# Follow switchover procedure in failover-runbook.md
```

**RTO**: 15-30 minutes
**RPO**: Up to 5 minutes (async replication lag)

---

### Scenario 4: Data Corruption

**Description**: Database corruption detected, possibly from software bug, hardware failure, or human error.

**Detection**:
- pg_amcheck reports corruption
- Checksum verification failures
- Application errors on specific data
- Inconsistent query results

**Types of Corruption**:

| Type | Scope | Recovery Method |
|------|-------|-----------------|
| Index corruption | Single index | REINDEX |
| Table corruption | Single table | Table restore from backup |
| Database corruption | Single database | Database restore |
| Cluster corruption | Full cluster | Full restore + PITR |
| Logical corruption | Data integrity | Point-in-time recovery |

**Recovery Procedure**:

```bash
# Step 1: Assess corruption scope
# Enable data checksums verification
pg_amcheck --all --install-missing

# Check for corruption
pg_amcheck -d greenlang --heapallindexed

# Step 2a: Index Corruption - REINDEX
REINDEX INDEX CONCURRENTLY affected_index_name;
# OR for entire table
REINDEX TABLE CONCURRENTLY affected_table_name;

# Step 2b: Table Corruption - Restore single table
# See backup-restore-runbook.md for table-level restore

# Step 2c: Logical Corruption - Point-in-Time Recovery
# Identify the time BEFORE corruption occurred
# Example: Accidental DELETE at 2026-02-03 14:30:00

# Stop writes immediately
kubectl scale deployment greenlang-api --replicas=0 -n production

# Perform PITR to time before corruption
# See restore-pitr.sh script

# Step 3: Verify data integrity after recovery
pg_amcheck -d greenlang --all

# Step 4: Document incident
# Create post-mortem report
```

**RTO**: 15-60 minutes (depending on corruption scope)
**RPO**: Varies (PITR allows recovery to any point)

---

## 4. Backup Infrastructure

### Backup Schedule

| Type | Frequency | Retention | Location |
|------|-----------|-----------|----------|
| Base Backup | Daily at 02:00 UTC | 30 days | S3: greenlang-backups/base/ |
| WAL Archives | Continuous | 7 days | S3: greenlang-backups/wal/ |
| pg_dump (logical) | Weekly | 90 days | S3: greenlang-backups/logical/ |
| TimescaleDB chunks | Daily | Per retention policy | S3: greenlang-backups/timescale/ |

### Backup Verification

```bash
# Daily automated verification
/opt/scripts/verify-backup.sh

# Manual verification
pgbackrest verify

# Test restore (weekly)
/opt/scripts/test-restore.sh
```

---

## 5. Communication Plan

### Incident Severity Levels

| Level | Description | Response Time | Communication |
|-------|-------------|---------------|---------------|
| SEV-1 | Complete outage | Immediate | All-hands, executives |
| SEV-2 | Degraded service | 15 minutes | Engineering + stakeholders |
| SEV-3 | Partial impact | 30 minutes | Engineering team |
| SEV-4 | Minor issue | 1 hour | Database team |

### Communication Channels

- **PagerDuty**: dba-oncall
- **Slack**: #incident-response, #dba-alerts
- **Status Page**: status.greenlang.io
- **Email**: incident@greenlang.io

### Communication Template

```
Subject: [SEV-X] Database Incident - [Brief Description]

Status: Investigating | Identified | Monitoring | Resolved

Impact: [Description of user impact]

Timeline:
- HH:MM UTC: Issue detected
- HH:MM UTC: [Action taken]
- HH:MM UTC: [Current status]

Next Update: [Time]

Incident Commander: [Name]
```

---

## 6. Testing Requirements

### Test Schedule

| Test Type | Frequency | Duration | Approval |
|-----------|-----------|----------|----------|
| Automated failover | Weekly | 30 min | Automated |
| Manual failover | Monthly | 2 hours | Team lead |
| AZ failover | Quarterly | 4 hours | Manager |
| Region failover | Semi-annual | 8 hours | VP Eng |
| Full DR test | Annual | 24 hours | Executive |

### Test Checklist

- [ ] Verify current backup status
- [ ] Confirm monitoring alerts are active
- [ ] Notify stakeholders of test window
- [ ] Execute failover procedure
- [ ] Measure actual RTO/RPO
- [ ] Verify application recovery
- [ ] Document results
- [ ] Update procedures if needed

---

## 7. Appendix

### A. Quick Reference Commands

```bash
# Check cluster status
patronictl -c /etc/patroni/patroni.yml list

# Force failover
patronictl -c /etc/patroni/patroni.yml failover --force

# Check replication lag
SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;

# Check backup status
pgbackrest info

# Restore to point-in-time
pgbackrest restore --target-time="2026-02-03 14:00:00" --type=time

# Check WAL shipping
SELECT * FROM pg_stat_archiver;
```

### B. Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| Patroni config | /etc/patroni/patroni.yml | Cluster management |
| PostgreSQL config | /var/lib/postgresql/data/postgresql.conf | DB settings |
| pgBackRest config | /etc/pgbackrest/pgbackrest.conf | Backup config |
| HAProxy config | /etc/haproxy/haproxy.cfg | Load balancer |

### C. Related Documents

- [Failover Runbook](./failover-runbook.md)
- [Backup and Restore Runbook](./backup-restore-runbook.md)
- [Database Operations Guide](./database-operations.md)
- [Monitoring and Alerting](../monitoring/alerting-rules.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | DBA Team | Initial version |
