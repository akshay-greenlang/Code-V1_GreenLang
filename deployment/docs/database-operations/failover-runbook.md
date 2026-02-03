# PostgreSQL/TimescaleDB Failover Runbook

## Document Information

| Field | Value |
|-------|-------|
| Document Owner | Database Operations Team |
| Last Updated | 2026-02-03 |
| Review Cycle | Monthly |
| Classification | Internal - Operations |

---

## 1. Overview

This runbook provides step-by-step procedures for database failover operations including automatic failover via Patroni, manual failover, planned switchover, and post-failover validation.

### Prerequisites

- Access to Kubernetes cluster (`kubectl` configured)
- Access to database nodes (SSH keys)
- Patroni CLI (`patronictl`) available
- Appropriate permissions (DBA role)

### Environment Variables

```bash
# Set these before running procedures
export PATRONI_CONFIG=/etc/patroni/patroni.yml
export CLUSTER_NAME=greenlang-db
export NAMESPACE=database
export PRIMARY_HOST=greenlang-db-0.greenlang-db.database.svc.cluster.local
export DB_USER=postgres
export DB_NAME=greenlang
```

---

## 2. Automatic Failover (Patroni)

### How It Works

Patroni provides automatic failover capabilities using distributed consensus (etcd/ZooKeeper/Consul). When the primary becomes unavailable:

1. Patroni detects failure via health checks
2. Leader lock is released (or expires)
3. Replicas compete for leader lock
4. Winner is promoted to primary
5. Other replicas follow new primary
6. HAProxy/PgBouncer updates routing

### Failover Configuration

```yaml
# Patroni configuration for automatic failover
loop_wait: 10              # Seconds between health checks
ttl: 30                    # Leader lock TTL
retry_timeout: 10          # Retry timeout for DCS
maximum_lag_on_failover: 0 # Zero data loss (sync replica)

# Sync replication settings
synchronous_mode: true
synchronous_mode_strict: true
synchronous_node_count: 1
```

### Monitoring Automatic Failover

```bash
# Watch cluster status during failover
watch -n 1 'patronictl -c $PATRONI_CONFIG list'

# Check Patroni logs
kubectl logs -f greenlang-db-0 -n database -c patroni

# Check for recent failover events
patronictl -c $PATRONI_CONFIG history

# Verify new primary
psql -h $PRIMARY_HOST -U $DB_USER -c "SELECT pg_is_in_recovery();"
# Should return 'f' (false) for primary
```

### Troubleshooting Automatic Failover

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No failover occurs | All replicas have lag | Check `synchronous_mode_strict` setting |
| Multiple primaries | Split brain | Fence old primary, restart Patroni |
| Failover too slow | DCS latency | Reduce TTL and loop_wait |
| Replica not promoted | Missing sync replica | Check synchronous_standby_names |

---

## 3. Manual Failover Procedure

Use manual failover when automatic failover is disabled or has not triggered.

### Pre-Flight Checks

```bash
#!/bin/bash
# Pre-flight checks before manual failover

echo "=== Pre-Flight Checks ==="

# Check cluster status
echo "1. Cluster Status:"
patronictl -c $PATRONI_CONFIG list

# Check replication lag
echo "2. Replication Lag:"
psql -h $PRIMARY_HOST -U $DB_USER -d $DB_NAME -c "
SELECT
    client_addr,
    state,
    sync_state,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) / 1024 / 1024 AS lag_mb
FROM pg_stat_replication;
"

# Check for long-running transactions
echo "3. Long Running Transactions:"
psql -h $PRIMARY_HOST -U $DB_USER -d $DB_NAME -c "
SELECT pid, now() - xact_start AS duration, query
FROM pg_stat_activity
WHERE xact_start IS NOT NULL
  AND now() - xact_start > interval '1 minute'
ORDER BY duration DESC;
"

# Check for locks
echo "4. Blocking Locks:"
psql -h $PRIMARY_HOST -U $DB_USER -d $DB_NAME -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocking_locks.pid AS blocking_pid,
       blocked_activity.usename AS blocked_user,
       blocking_activity.usename AS blocking_user
FROM pg_locks blocked_locks
JOIN pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
JOIN pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
"

echo "=== Pre-Flight Checks Complete ==="
```

### Failover Procedure

```bash
#!/bin/bash
# Manual failover procedure

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/failover_${TIMESTAMP}.log"

echo "Starting manual failover at $(date)" | tee $LOG_FILE

# Step 1: Verify current primary is truly unavailable
echo "Step 1: Verifying primary status..." | tee -a $LOG_FILE
if psql -h $PRIMARY_HOST -U $DB_USER -c "SELECT 1" &>/dev/null; then
    echo "WARNING: Primary appears to be responding!" | tee -a $LOG_FILE
    echo "Consider using switchover instead of failover." | tee -a $LOG_FILE
    read -p "Continue with failover anyway? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Failover cancelled." | tee -a $LOG_FILE
        exit 1
    fi
fi

# Step 2: Check cluster status
echo "Step 2: Checking cluster status..." | tee -a $LOG_FILE
patronictl -c $PATRONI_CONFIG list | tee -a $LOG_FILE

# Step 3: Identify best candidate for promotion
echo "Step 3: Identifying failover candidate..." | tee -a $LOG_FILE
CANDIDATE=$(patronictl -c $PATRONI_CONFIG list -f json | \
    jq -r '.[] | select(.Role == "Sync Standby" or .Role == "Replica") |
    select(.Lag == 0 or .Lag == null) | .Member' | head -1)

if [ -z "$CANDIDATE" ]; then
    echo "ERROR: No suitable failover candidate found!" | tee -a $LOG_FILE
    exit 1
fi

echo "Selected candidate: $CANDIDATE" | tee -a $LOG_FILE

# Step 4: Execute failover
echo "Step 4: Executing failover to $CANDIDATE..." | tee -a $LOG_FILE
patronictl -c $PATRONI_CONFIG failover \
    --candidate $CANDIDATE \
    --force 2>&1 | tee -a $LOG_FILE

# Step 5: Wait for failover to complete
echo "Step 5: Waiting for failover to complete..." | tee -a $LOG_FILE
sleep 10

# Step 6: Verify new primary
echo "Step 6: Verifying new primary..." | tee -a $LOG_FILE
for i in {1..30}; do
    NEW_PRIMARY=$(patronictl -c $PATRONI_CONFIG list -f json | \
        jq -r '.[] | select(.Role == "Leader") | .Member')

    if [ "$NEW_PRIMARY" == "$CANDIDATE" ]; then
        echo "Failover successful! New primary: $NEW_PRIMARY" | tee -a $LOG_FILE
        break
    fi

    echo "Waiting for promotion... ($i/30)" | tee -a $LOG_FILE
    sleep 2
done

# Step 7: Final verification
echo "Step 7: Final cluster status..." | tee -a $LOG_FILE
patronictl -c $PATRONI_CONFIG list | tee -a $LOG_FILE

echo "Manual failover completed at $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"
```

---

## 4. Switchover Procedure (Planned)

Use switchover for planned maintenance with zero or minimal downtime.

### Pre-Switchover Checklist

- [ ] Notify stakeholders of maintenance window
- [ ] Verify all replicas are synchronized (lag = 0)
- [ ] No long-running transactions on primary
- [ ] Backup completed recently
- [ ] Monitoring alerts are acknowledged
- [ ] Rollback plan is documented

### Switchover Procedure

```bash
#!/bin/bash
# Planned switchover procedure

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/switchover_${TIMESTAMP}.log"

echo "Starting planned switchover at $(date)" | tee $LOG_FILE

# Step 1: Pre-switchover checks
echo "Step 1: Pre-switchover checks..." | tee -a $LOG_FILE
patronictl -c $PATRONI_CONFIG list | tee -a $LOG_FILE

# Verify no replication lag
LAG=$(patronictl -c $PATRONI_CONFIG list -f json | \
    jq -r '[.[] | select(.Role != "Leader") | .Lag // 0] | max')

if [ "$LAG" != "0" ] && [ "$LAG" != "null" ]; then
    echo "ERROR: Replication lag detected: ${LAG}MB" | tee -a $LOG_FILE
    echo "Wait for replicas to catch up before switchover." | tee -a $LOG_FILE
    exit 1
fi

# Step 2: Identify target for promotion
CURRENT_PRIMARY=$(patronictl -c $PATRONI_CONFIG list -f json | \
    jq -r '.[] | select(.Role == "Leader") | .Member')
TARGET=$(patronictl -c $PATRONI_CONFIG list -f json | \
    jq -r '.[] | select(.Role == "Sync Standby") | .Member' | head -1)

echo "Current primary: $CURRENT_PRIMARY" | tee -a $LOG_FILE
echo "Target for promotion: $TARGET" | tee -a $LOG_FILE

# Step 3: Pause application writes (optional, for zero data loss)
echo "Step 3: Pausing application traffic..." | tee -a $LOG_FILE
# kubectl scale deployment greenlang-api --replicas=0 -n production
# OR use connection pooler pause
# psql -h pgbouncer -U admin -p 6432 pgbouncer -c "PAUSE greenlang;"

# Step 4: Wait for final WAL replay
echo "Step 4: Waiting for final WAL replay..." | tee -a $LOG_FILE
sleep 5

# Step 5: Execute switchover
echo "Step 5: Executing switchover..." | tee -a $LOG_FILE
patronictl -c $PATRONI_CONFIG switchover \
    --master $CURRENT_PRIMARY \
    --candidate $TARGET \
    --scheduled now \
    --force 2>&1 | tee -a $LOG_FILE

# Step 6: Wait for switchover completion
echo "Step 6: Waiting for switchover to complete..." | tee -a $LOG_FILE
for i in {1..60}; do
    NEW_PRIMARY=$(patronictl -c $PATRONI_CONFIG list -f json | \
        jq -r '.[] | select(.Role == "Leader") | .Member')

    if [ "$NEW_PRIMARY" == "$TARGET" ]; then
        echo "Switchover successful!" | tee -a $LOG_FILE
        break
    fi

    echo "Waiting... ($i/60)" | tee -a $LOG_FILE
    sleep 2
done

# Step 7: Verify old primary becomes replica
echo "Step 7: Verifying old primary becomes replica..." | tee -a $LOG_FILE
sleep 10
patronictl -c $PATRONI_CONFIG list | tee -a $LOG_FILE

# Step 8: Resume application traffic
echo "Step 8: Resuming application traffic..." | tee -a $LOG_FILE
# kubectl scale deployment greenlang-api --replicas=3 -n production
# OR unpause connection pooler
# psql -h pgbouncer -U admin -p 6432 pgbouncer -c "RESUME greenlang;"

# Step 9: Verify application connectivity
echo "Step 9: Verifying application connectivity..." | tee -a $LOG_FILE
psql -h greenlang-db.database.svc.cluster.local -U $DB_USER -d $DB_NAME -c "
SELECT pg_is_in_recovery(), current_timestamp;
"

echo "Switchover completed at $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE"
```

---

## 5. Post-Failover Checklist

### Immediate Actions (0-5 minutes)

```bash
#!/bin/bash
# Post-failover immediate verification

echo "=== Post-Failover Immediate Checks ==="

# 1. Verify new primary is accepting writes
echo "1. Testing write capability..."
psql -h greenlang-db.database.svc.cluster.local -U $DB_USER -d $DB_NAME -c "
CREATE TEMP TABLE failover_test (id int);
INSERT INTO failover_test VALUES (1);
DROP TABLE failover_test;
SELECT 'Write test passed' AS result;
"

# 2. Verify cluster status
echo "2. Cluster status..."
patronictl -c $PATRONI_CONFIG list

# 3. Check replication is re-established
echo "3. Replication status..."
psql -h greenlang-db.database.svc.cluster.local -U $DB_USER -d $DB_NAME -c "
SELECT client_addr, state, sync_state, sent_lsn, replay_lsn
FROM pg_stat_replication;
"

# 4. Verify connection pooler connectivity
echo "4. Connection pooler status..."
psql -h pgbouncer -U admin -p 6432 pgbouncer -c "SHOW POOLS;"

# 5. Check application connectivity
echo "5. Application health check..."
curl -s http://greenlang-api.production.svc.cluster.local/health | jq .
```

### Short-term Actions (5-30 minutes)

- [ ] Verify all replicas have caught up
- [ ] Check application error rates in monitoring
- [ ] Verify TimescaleDB continuous aggregates are running
- [ ] Check background jobs are processing
- [ ] Review any failed transactions during failover
- [ ] Update monitoring dashboards if needed

### Medium-term Actions (30 minutes - 2 hours)

```bash
# Detailed post-failover validation

# 1. Check for any data inconsistencies
psql -h greenlang-db.database.svc.cluster.local -U $DB_USER -d $DB_NAME -c "
-- Check sequence values
SELECT schemaname, sequencename, last_value
FROM pg_sequences
ORDER BY schemaname, sequencename;
"

# 2. Verify TimescaleDB chunks
psql -h greenlang-db.database.svc.cluster.local -U $DB_USER -d $DB_NAME -c "
SELECT hypertable_name, chunk_name, range_start, range_end
FROM timescaledb_information.chunks
WHERE hypertable_name IN (SELECT table_name FROM _timescaledb_catalog.hypertable)
ORDER BY range_end DESC
LIMIT 20;
"

# 3. Check continuous aggregate status
psql -h greenlang-db.database.svc.cluster.local -U $DB_USER -d $DB_NAME -c "
SELECT view_name, materialization_hypertable_name
FROM timescaledb_information.continuous_aggregates;
"

# 4. Verify scheduled jobs
psql -h greenlang-db.database.svc.cluster.local -U $DB_USER -d $DB_NAME -c "
SELECT job_id, application_name, schedule_interval, last_run_status, next_start
FROM timescaledb_information.jobs
ORDER BY next_start;
"
```

### Long-term Actions (2-24 hours)

- [ ] Recover failed primary (if hardware okay)
- [ ] Rejoin recovered node as replica
- [ ] Verify full backup completed since failover
- [ ] Review and update runbooks based on experience
- [ ] Create incident post-mortem report
- [ ] Plan for any needed capacity adjustments

---

## 6. Application Reconnection

### Connection String Updates

After failover, applications may need to reconnect. If using proper high-availability setup, this should be automatic.

### HAProxy Configuration

```
# HAProxy configuration for automatic failover
backend postgresql_primary
    mode tcp
    option httpchk GET /primary
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server greenlang-db-0 greenlang-db-0:5432 check port 8008
    server greenlang-db-1 greenlang-db-1:5432 check port 8008
    server greenlang-db-2 greenlang-db-2:5432 check port 8008
```

### PgBouncer Configuration

```ini
[databases]
greenlang = host=greenlang-db.database.svc.cluster.local port=5432 dbname=greenlang

[pgbouncer]
listen_addr = *
listen_port = 6432
auth_type = scram-sha-256
pool_mode = transaction
server_reset_query = DISCARD ALL
server_check_query = SELECT 1
server_check_delay = 30
server_connect_timeout = 5
server_login_retry = 3
```

### Application Connection Handling

```python
# Python example with automatic retry
import psycopg2
from psycopg2 import OperationalError
import time

def get_connection(max_retries=5, retry_delay=2):
    """Get database connection with automatic retry."""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host="greenlang-db.database.svc.cluster.local",
                port=5432,
                database="greenlang",
                user="app_user",
                password=os.environ["DB_PASSWORD"],
                connect_timeout=10,
                # These settings help with failover
                target_session_attrs="read-write",
            )
            return conn
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
```

### Kubernetes Service Configuration

```yaml
# Service that automatically routes to primary via Patroni
apiVersion: v1
kind: Service
metadata:
  name: greenlang-db
  namespace: database
spec:
  type: ClusterIP
  selector:
    app: greenlang-db
    role: primary  # Label managed by Patroni
  ports:
    - port: 5432
      targetPort: 5432
```

---

## 7. Troubleshooting

### Common Issues

| Issue | Symptom | Resolution |
|-------|---------|------------|
| Split brain | Two primaries | Fence one primary, sync data, restart |
| Timeline divergence | Replica won't follow | Reinitialize replica from primary |
| Connection refused | Apps can't connect | Check HAProxy, DNS, firewall |
| Replication lag after failover | High lag | Check network, increase WAL senders |

### Diagnostic Commands

```bash
# Check Patroni state
patronictl -c $PATRONI_CONFIG show-config

# Check DCS (etcd) connectivity
etcdctl endpoint health

# Check PostgreSQL logs
kubectl logs greenlang-db-0 -n database -c postgres --tail=100

# Check network connectivity
kubectl exec greenlang-db-0 -n database -- pg_isready -h greenlang-db-1

# Check WAL position
psql -c "SELECT pg_current_wal_lsn();"  # On primary
psql -c "SELECT pg_last_wal_replay_lsn();"  # On replica
```

---

## 8. Related Documents

- [Disaster Recovery Plan](./disaster-recovery-plan.md)
- [Backup and Restore Runbook](./backup-restore-runbook.md)
- [Database Monitoring Guide](../monitoring/database-monitoring.md)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-03 | DBA Team | Initial version |
