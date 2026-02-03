# Patroni PostgreSQL/TimescaleDB HA Cluster Operations Runbook

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Cluster Status](#cluster-status)
5. [Manual Failover Procedures](#manual-failover-procedures)
6. [Adding/Removing Replicas](#addingremoving-replicas)
7. [Backup and Recovery](#backup-and-recovery)
8. [Configuration Changes](#configuration-changes)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Monitoring and Alerting](#monitoring-and-alerting)
11. [Disaster Recovery](#disaster-recovery)

---

## Overview

This runbook covers operational procedures for the GreenLang Patroni PostgreSQL/TimescaleDB high-availability cluster. The cluster consists of:

- **3 PostgreSQL/TimescaleDB nodes** in a StatefulSet
- **Patroni** for HA management and automatic failover
- **pgBackRest** for backup and point-in-time recovery
- **Kubernetes DCS** (Distributed Configuration Store) for leader election

### Key Components

| Component | Purpose |
|-----------|---------|
| Patroni | HA management, leader election, failover |
| PostgreSQL 15 | Database engine |
| TimescaleDB | Time-series extension |
| pgBackRest | Backup and recovery |
| postgres-exporter | Prometheus metrics |

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              Kubernetes Cluster                      │
                    │                                                      │
     ┌──────────────┼──────────────┬───────────────┬───────────────┐      │
     │              │              │               │               │      │
┌────▼────┐   ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐         │      │
│ Primary │   │ Replica │   │  Replica  │   │ pgBackRest│         │      │
│ Service │   │ Service │   │  (Sync)   │   │   Backup  │         │      │
└────┬────┘   └────┬────┘   └─────┬─────┘   └─────┬─────┘         │      │
     │              │              │               │               │      │
     │         ┌────┴──────────────┴───────────────┘               │      │
     │         │                                                    │      │
┌────▼────┬────▼────┬────────────┐                                 │      │
│patroni-0│patroni-1│ patroni-2  │◄──── StatefulSet                │      │
│ PRIMARY │ REPLICA │  REPLICA   │      (3 replicas)               │      │
└────┬────┴────┬────┴─────┬──────┘                                 │      │
     │         │          │                                         │      │
     └─────────┼──────────┘                                         │      │
               │                                                    │      │
        ┌──────▼──────┐                                             │      │
        │ K8s Endpoints│◄──── DCS (Leader Election)                 │      │
        │ (patroni-*)  │                                            │      │
        └─────────────┘                                             │      │
                    │                                               │      │
                    └───────────────────────────────────────────────┘      │
```

---

## Prerequisites

### Tools Required

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install patronictl wrapper (copy to pod)
kubectl cp scripts/patronictl-wrapper.sh greenlang-db/patroni-0:/scripts/

# Verify access
kubectl get pods -n greenlang-db
kubectl exec -it patroni-0 -n greenlang-db -- patronictl list greenlang-cluster
```

### Environment Variables

```bash
export NAMESPACE=greenlang-db
export CLUSTER_NAME=greenlang-cluster
export PRIMARY_POD=$(kubectl get pods -n $NAMESPACE -l role=master -o jsonpath='{.items[0].metadata.name}')
```

---

## Cluster Status

### View Cluster Status

```bash
# Using patronictl
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml list greenlang-cluster

# Expected output:
# + Cluster: greenlang-cluster (1234567890) ------+----+-----------+
# |   Member    |   Host    | Role    | State   | TL | Lag in MB |
# +-------------+-----------+---------+---------+----+-----------+
# | patroni-0   | 10.0.1.10 | Leader  | running |  2 |           |
# | patroni-1   | 10.0.1.11 | Replica | running |  2 |       0.0 |
# | patroni-2   | 10.0.1.12 | Replica | running |  2 |       0.0 |
# +-------------+-----------+---------+---------+----+-----------+
```

### Check Cluster Topology

```bash
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml topology greenlang-cluster
```

### View Cluster History

```bash
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml history greenlang-cluster
```

### Check Replication Lag

```bash
# From primary
kubectl exec -it $PRIMARY_POD -n greenlang-db -- psql -U postgres -c \
    "SELECT client_addr, state, sync_state,
            pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) AS lag_bytes
     FROM pg_stat_replication;"
```

### Check via Patroni REST API

```bash
# Check primary status
kubectl exec -it patroni-0 -n greenlang-db -- curl -s http://localhost:8008/primary

# Check replica status
kubectl exec -it patroni-1 -n greenlang-db -- curl -s http://localhost:8008/replica

# Get detailed node info
kubectl exec -it patroni-0 -n greenlang-db -- curl -s http://localhost:8008/patroni | jq .
```

---

## Manual Failover Procedures

### Planned Switchover (Zero Downtime)

Use switchover for planned maintenance. This is the preferred method.

```bash
# 1. Check current cluster status
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml list greenlang-cluster

# 2. Perform switchover to specific replica
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml switchover greenlang-cluster \
    --leader patroni-0 \
    --candidate patroni-1 \
    --force

# 3. Monitor switchover progress
watch -n 1 'kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml list greenlang-cluster'

# 4. Verify new primary
kubectl exec -it patroni-1 -n greenlang-db -- \
    curl -s http://localhost:8008/primary
```

### Scheduled Switchover

Schedule a switchover for a specific time.

```bash
# Schedule switchover for 2 AM
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml switchover greenlang-cluster \
    --candidate patroni-2 \
    --scheduled "2024-01-15T02:00:00+00:00" \
    --force
```

### Emergency Failover

Use only when primary is completely unavailable.

```bash
# WARNING: May result in data loss if primary has uncommitted transactions

# 1. Pause cluster (if possible)
kubectl exec -it patroni-1 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml pause greenlang-cluster --wait

# 2. Force failover to specific replica
kubectl exec -it patroni-1 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml failover greenlang-cluster \
    --candidate patroni-1 \
    --force

# 3. Resume cluster
kubectl exec -it patroni-1 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml resume greenlang-cluster --wait
```

### Rollback Failover

If a failover was unsuccessful.

```bash
# 1. Check which node has the latest data
for pod in patroni-0 patroni-1 patroni-2; do
    echo "=== $pod ==="
    kubectl exec -it $pod -n greenlang-db -- psql -U postgres -c \
        "SELECT pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn();" 2>/dev/null || echo "Unavailable"
done

# 2. Force leadership to the node with latest data
kubectl exec -it patroni-X -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml failover greenlang-cluster \
    --candidate patroni-X \
    --force
```

---

## Adding/Removing Replicas

### Adding a New Replica

```bash
# 1. Scale up the StatefulSet
kubectl scale statefulset patroni -n greenlang-db --replicas=4

# 2. Monitor new replica initialization
kubectl logs -f patroni-3 -n greenlang-db

# 3. Verify replica joined cluster
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml list greenlang-cluster
```

### Removing a Replica

```bash
# 1. Ensure you're not removing the primary
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml list greenlang-cluster

# 2. If removing the primary, switchover first
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml switchover greenlang-cluster \
    --leader patroni-3 \
    --candidate patroni-0 \
    --force

# 3. Scale down
kubectl scale statefulset patroni -n greenlang-db --replicas=3

# 4. Clean up PVC if needed
kubectl delete pvc pgdata-patroni-3 -n greenlang-db
```

### Reinitializing a Replica

Use when a replica has become corrupted or too far behind.

```bash
# 1. Reinitialize the replica
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml reinit greenlang-cluster patroni-2 --force

# 2. Monitor progress
kubectl logs -f patroni-2 -n greenlang-db

# 3. Verify replica is back in sync
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml list greenlang-cluster
```

---

## Backup and Recovery

### Trigger Manual Backup

```bash
# Full backup
kubectl exec -it $PRIMARY_POD -n greenlang-db -- \
    pgbackrest --stanza=greenlang backup --type=full

# Differential backup
kubectl exec -it $PRIMARY_POD -n greenlang-db -- \
    pgbackrest --stanza=greenlang backup --type=diff

# Incremental backup
kubectl exec -it $PRIMARY_POD -n greenlang-db -- \
    pgbackrest --stanza=greenlang backup --type=incr
```

### View Available Backups

```bash
kubectl exec -it $PRIMARY_POD -n greenlang-db -- \
    pgbackrest --stanza=greenlang info
```

### Point-in-Time Recovery

```bash
# 1. Pause the cluster
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml pause greenlang-cluster --wait

# 2. Stop all replicas
kubectl scale statefulset patroni -n greenlang-db --replicas=1

# 3. Perform PITR on primary
kubectl exec -it patroni-0 -n greenlang-db -- \
    pgbackrest --stanza=greenlang restore \
    --target="2024-01-15 10:30:00" \
    --target-action=promote \
    --type=time

# 4. Scale back up
kubectl scale statefulset patroni -n greenlang-db --replicas=3

# 5. Resume cluster
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml resume greenlang-cluster --wait
```

---

## Configuration Changes

### View Current Configuration

```bash
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml show-config greenlang-cluster
```

### Edit Cluster Configuration

```bash
# Interactive edit
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml edit-config greenlang-cluster

# Or apply specific changes
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml edit-config greenlang-cluster \
    --set "postgresql.parameters.max_connections=300"
```

### Reload Configuration

```bash
# Reload all nodes
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml reload greenlang-cluster --force

# Reload specific node
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml reload greenlang-cluster patroni-1 --force
```

### Restart PostgreSQL (Rolling)

```bash
# Restart all nodes (rolling)
for pod in patroni-2 patroni-1 patroni-0; do
    echo "Restarting $pod..."
    kubectl exec -it $pod -n greenlang-db -- \
        patronictl -c /etc/patroni/patroni.yml restart greenlang-cluster $pod --force
    sleep 30
done
```

---

## Troubleshooting Guide

### Common Issues

#### Issue: Split-Brain Detected

**Symptoms**: Multiple nodes claim to be primary

**Resolution**:
```bash
# 1. Pause cluster immediately
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml pause greenlang-cluster --wait

# 2. Check which node has latest timeline
for pod in patroni-0 patroni-1 patroni-2; do
    echo "=== $pod ==="
    kubectl exec -it $pod -n greenlang-db -- psql -U postgres -c \
        "SELECT pg_control_checkpoint();" 2>/dev/null
done

# 3. Demote incorrect primary
kubectl exec -it patroni-X -n greenlang-db -- pg_ctl stop -D $PGDATA -m fast

# 4. Resume cluster
kubectl exec -it patroni-Y -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml resume greenlang-cluster --wait
```

#### Issue: Replica Not Catching Up

**Symptoms**: High replication lag

**Resolution**:
```bash
# 1. Check replication status
kubectl exec -it $PRIMARY_POD -n greenlang-db -- psql -U postgres -c \
    "SELECT * FROM pg_stat_replication;"

# 2. Check for WAL files
kubectl exec -it $PRIMARY_POD -n greenlang-db -- ls -la /var/lib/postgresql/data/pgdata/pg_wal/

# 3. If too far behind, reinitialize
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml reinit greenlang-cluster patroni-X --force
```

#### Issue: Cannot Connect to Primary

**Symptoms**: Connection refused to primary service

**Resolution**:
```bash
# 1. Check service endpoints
kubectl get endpoints patroni-primary -n greenlang-db

# 2. Check pod labels
kubectl get pods -n greenlang-db --show-labels | grep role=master

# 3. Check Patroni API
for pod in patroni-0 patroni-1 patroni-2; do
    echo "=== $pod ==="
    kubectl exec -it $pod -n greenlang-db -- curl -s http://localhost:8008/patroni | jq '.role'
done

# 4. Force leader re-election if needed
kubectl exec -it patroni-0 -n greenlang-db -- \
    patronictl -c /etc/patroni/patroni.yml failover greenlang-cluster --force
```

#### Issue: DCS (Kubernetes) Unreachable

**Symptoms**: Patroni cannot reach Kubernetes API

**Resolution**:
```bash
# 1. Check RBAC permissions
kubectl auth can-i get endpoints -n greenlang-db --as=system:serviceaccount:greenlang-db:patroni

# 2. Check service account
kubectl get serviceaccount patroni -n greenlang-db

# 3. Verify network connectivity
kubectl exec -it patroni-0 -n greenlang-db -- curl -k https://kubernetes.default.svc/api/v1/namespaces/greenlang-db/endpoints

# 4. Restart pod if needed
kubectl delete pod patroni-0 -n greenlang-db
```

#### Issue: Disk Full

**Symptoms**: PostgreSQL refuses connections, errors in logs

**Resolution**:
```bash
# 1. Check disk usage
kubectl exec -it patroni-0 -n greenlang-db -- df -h /var/lib/postgresql/data

# 2. Clean up WAL files (if safe)
kubectl exec -it $PRIMARY_POD -n greenlang-db -- psql -U postgres -c \
    "SELECT pg_switch_wal();"

# 3. Force checkpoint and archive
kubectl exec -it $PRIMARY_POD -n greenlang-db -- psql -U postgres -c "CHECKPOINT;"

# 4. Clean up old backups
kubectl exec -it $PRIMARY_POD -n greenlang-db -- \
    pgbackrest --stanza=greenlang expire

# 5. Expand PVC if needed
kubectl patch pvc pgdata-patroni-0 -n greenlang-db -p '{"spec":{"resources":{"requests":{"storage":"1000Gi"}}}}'
```

### Log Analysis

```bash
# View Patroni logs
kubectl logs patroni-0 -n greenlang-db -c patroni --tail=100

# View PostgreSQL logs
kubectl exec -it patroni-0 -n greenlang-db -- tail -100 /var/lib/postgresql/data/pgdata/log/postgresql-*.log

# Search for specific errors
kubectl logs patroni-0 -n greenlang-db | grep -i error

# View events
kubectl get events -n greenlang-db --sort-by='.lastTimestamp'
```

---

## Monitoring and Alerting

### Key Metrics to Monitor

| Metric | Warning | Critical |
|--------|---------|----------|
| Replication Lag (seconds) | > 30s | > 60s |
| Replication Lag (bytes) | > 50MB | > 100MB |
| Connection Usage | > 80% | > 95% |
| Disk Usage | > 70% | > 85% |
| WAL Archive Failures | > 0 | > 5 |
| Failover Count (24h) | > 1 | > 3 |

### Prometheus Queries

```promql
# Replication lag in seconds
pg_replication_lag_seconds{cluster="greenlang-cluster"}

# Connection utilization
pg_stat_activity_count / pg_settings_max_connections * 100

# Database size
pg_database_size_bytes{datname="greenlang"}

# Transaction rate
rate(pg_stat_database_xact_commit[5m])

# TimescaleDB chunk count
timescaledb_chunks_total{cluster="greenlang-cluster"}
```

### Grafana Dashboard

Dashboard available at: `/d/patroni-overview/patroni-cluster-overview`

---

## Disaster Recovery

### Complete Cluster Recovery

```bash
# 1. Delete all pods
kubectl delete statefulset patroni -n greenlang-db

# 2. Clean up PVCs
kubectl delete pvc -l app=patroni -n greenlang-db

# 3. Clean up DCS endpoints
kubectl delete endpoints -l app=patroni -n greenlang-db

# 4. Redeploy via Helm
helm upgrade --install patroni ./helm/patroni -n greenlang-db \
    -f values-production.yaml

# 5. Restore from backup
kubectl exec -it patroni-0 -n greenlang-db -- \
    pgbackrest --stanza=greenlang restore --delta

# 6. Reinitialize replicas
kubectl scale statefulset patroni -n greenlang-db --replicas=3
```

### Cross-Region Recovery

```bash
# 1. Copy backup to target region
aws s3 sync s3://greenlang-db-backups-us-east-1 s3://greenlang-db-backups-us-west-2

# 2. Deploy cluster in target region
kubectl config use-context us-west-2
helm upgrade --install patroni ./helm/patroni -n greenlang-db \
    -f values-dr.yaml

# 3. Restore from backup
kubectl exec -it patroni-0 -n greenlang-db -- \
    pgbackrest --stanza=greenlang restore
```

---

## Maintenance Windows

### Recommended Maintenance Schedule

| Task | Frequency | Duration | Impact |
|------|-----------|----------|--------|
| Full Backup | Weekly (Sunday 2 AM) | 1-2 hours | None |
| VACUUM ANALYZE | Daily (3 AM) | 30 min | Minor |
| Minor Version Upgrade | Monthly | 15-30 min | Rolling restart |
| Major Version Upgrade | Yearly | 1-2 hours | Brief downtime |
| Storage Expansion | As needed | 5 min | None |

### Pre-Maintenance Checklist

- [ ] Verify all replicas are healthy
- [ ] Check replication lag < 10 seconds
- [ ] Verify recent backup completed successfully
- [ ] Notify stakeholders
- [ ] Prepare rollback plan
- [ ] Test connectivity to backup storage

### Post-Maintenance Checklist

- [ ] Verify cluster status
- [ ] Check all replicas rejoined
- [ ] Verify replication lag returned to normal
- [ ] Test application connectivity
- [ ] Verify backup functionality
- [ ] Update documentation

---

## Contact Information

| Role | Contact |
|------|---------|
| Primary On-Call | pagerduty:greenlang-db |
| Database Team | #greenlang-database (Slack) |
| Platform Team | #greenlang-platform (Slack) |
| Escalation | database-leads@greenlang.io |

---

**Document Version**: 1.0.0
**Last Updated**: 2024-01-15
**Authors**: GreenLang DevOps Team
