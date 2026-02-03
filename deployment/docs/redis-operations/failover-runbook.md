# Redis Failover Runbook

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | DevOps Team |
| Purpose | Step-by-step failover procedures |

---

## 1. Overview

This runbook provides detailed procedures for Redis failover operations, both automatic (Sentinel-managed) and manual. Use this guide when responding to Redis master failures or performing planned failovers.

### When to Use This Runbook

- Redis master is unresponsive
- Planned maintenance requiring master failover
- Performance issues requiring master promotion
- Sentinel automatic failover has failed

---

## 2. Pre-Failover Checklist

Before initiating any failover, complete this checklist:

```
PRE-FAILOVER VERIFICATION:

[ ] Confirm the issue requires failover (not transient)
[ ] Check current replication lag on all replicas
[ ] Verify Sentinel cluster health and quorum
[ ] Identify best failover candidate
[ ] Notify relevant stakeholders
[ ] Prepare rollback plan
[ ] Have monitoring dashboards open
```

### Health Check Commands

```bash
# Check master status
redis-cli -h redis-master.greenlang.internal -p 6379 INFO replication

# Check replica lag
redis-cli -h redis-replica-1.greenlang.internal -p 6379 INFO replication | grep master_repl_offset
redis-cli -h redis-replica-2.greenlang.internal -p 6379 INFO replication | grep master_repl_offset

# Check Sentinel quorum
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL CKQUORUM mymaster
```

---

## 3. Automatic Failover (Sentinel)

### How Automatic Failover Works

```
SENTINEL AUTOMATIC FAILOVER TIMELINE:

T+0s     Master stops responding
         │
T+5s     Sentinel detects SDOWN (Subjectively Down)
         │
T+5-10s  Sentinels exchange SDOWN notifications
         │
T+10s    Quorum reached - ODOWN (Objectively Down)
         │
T+10-15s Sentinel leader elected
         │
T+15-20s Best replica selected for promotion
         │
T+20-25s Replica promoted to master (SLAVEOF NO ONE)
         │
T+25-30s Other replicas reconfigured
         │
T+30s    Failover complete
```

### Monitoring Automatic Failover

```bash
# Watch Sentinel logs (in real-time)
tail -f /var/log/redis/sentinel.log

# Monitor Sentinel state
watch -n 1 'redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL master mymaster'

# Check failover in progress
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL failover-status mymaster
```

### Automatic Failover Success Indicators

```bash
# Verify new master
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL get-master-addr-by-name mymaster

# Check new master role
NEW_MASTER=$(redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL get-master-addr-by-name mymaster | head -1)
redis-cli -h $NEW_MASTER -p 6379 INFO replication | grep role

# Verify replicas connected
redis-cli -h $NEW_MASTER -p 6379 INFO replication | grep connected_slaves
```

### Troubleshooting Failed Automatic Failover

**Problem: Failover not triggered after expected time**

```bash
# Check if ODOWN was reached
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL master mymaster | grep flags

# Verify quorum configuration
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL master mymaster | grep quorum

# Check for existing failover in progress
redis-cli -h sentinel.greenlang.internal -p 26379 SENTINEL failover-status mymaster
```

**Problem: Replica not promoted**

```bash
# Check replica priority (lower = preferred)
redis-cli -h redis-replica-1.greenlang.internal -p 6379 CONFIG GET replica-priority

# Check if replica is excluded from failover
# Priority 0 = never promote
redis-cli -h redis-replica-1.greenlang.internal -p 6379 CONFIG SET replica-priority 100
```

---

## 4. Manual Failover Procedure

### 4.1 Sentinel FAILOVER Command (Recommended)

Use this method when you need controlled failover with Sentinel coordination.

```bash
#!/bin/bash
# Manual failover using Sentinel FAILOVER command

SENTINEL_HOST="sentinel.greenlang.internal"
SENTINEL_PORT="26379"
MASTER_NAME="mymaster"

echo "=== Redis Manual Failover ==="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Step 1: Get current master
echo -e "\n[1/5] Current master:"
CURRENT_MASTER=$(redis-cli -h $SENTINEL_HOST -p $SENTINEL_PORT \
  SENTINEL get-master-addr-by-name $MASTER_NAME)
echo "Master: $CURRENT_MASTER"

# Step 2: Check replicas
echo -e "\n[2/5] Available replicas:"
redis-cli -h $SENTINEL_HOST -p $SENTINEL_PORT SENTINEL slaves $MASTER_NAME | \
  grep -E "^(name|flags|master-link-status)"

# Step 3: Trigger failover
echo -e "\n[3/5] Triggering failover..."
redis-cli -h $SENTINEL_HOST -p $SENTINEL_PORT SENTINEL FAILOVER $MASTER_NAME

# Step 4: Wait for failover
echo -e "\n[4/5] Waiting for failover to complete..."
for i in {1..30}; do
  sleep 1
  NEW_MASTER=$(redis-cli -h $SENTINEL_HOST -p $SENTINEL_PORT \
    SENTINEL get-master-addr-by-name $MASTER_NAME 2>/dev/null | head -1)

  if [ "$NEW_MASTER" != "$(echo $CURRENT_MASTER | head -1)" ]; then
    echo "Failover completed at iteration $i"
    break
  fi
  echo "Waiting... ($i/30)"
done

# Step 5: Verify new master
echo -e "\n[5/5] New master:"
redis-cli -h $SENTINEL_HOST -p $SENTINEL_PORT SENTINEL get-master-addr-by-name $MASTER_NAME

echo -e "\n=== Failover Complete ==="
```

### 4.2 Force Failover Without Sentinel

Use only when Sentinel is unavailable or not functioning.

**WARNING**: This is a disruptive operation and requires manual application reconfiguration.

```bash
#!/bin/bash
# Force failover without Sentinel (emergency only)

NEW_MASTER_HOST="redis-replica-1.greenlang.internal"
NEW_MASTER_PORT="6379"
OLD_MASTER_HOST="redis-master.greenlang.internal"
REMAINING_REPLICA="redis-replica-2.greenlang.internal"

echo "=== EMERGENCY MANUAL FAILOVER ==="
echo "WARNING: This bypasses Sentinel. Use only in emergencies."
read -p "Continue? (yes/no): " confirm
[ "$confirm" != "yes" ] && exit 1

# Step 1: Promote replica to master
echo -e "\n[1/4] Promoting $NEW_MASTER_HOST to master..."
redis-cli -h $NEW_MASTER_HOST -p $NEW_MASTER_PORT REPLICAOF NO ONE

# Step 2: Verify promotion
echo -e "\n[2/4] Verifying promotion..."
redis-cli -h $NEW_MASTER_HOST -p $NEW_MASTER_PORT INFO replication | grep role

# Step 3: Reconfigure remaining replica
echo -e "\n[3/4] Reconfiguring remaining replica..."
redis-cli -h $REMAINING_REPLICA -p 6379 REPLICAOF $NEW_MASTER_HOST $NEW_MASTER_PORT

# Step 4: Update Sentinel configuration manually
echo -e "\n[4/4] Sentinel reconfiguration required!"
echo "Run on each Sentinel node:"
echo "  redis-cli -p 26379 SENTINEL MONITOR mymaster $NEW_MASTER_HOST $NEW_MASTER_PORT 2"
echo "  redis-cli -p 26379 SENTINEL RESET mymaster"

echo -e "\n=== Manual steps required ==="
echo "1. Update application REDIS_HOST to: $NEW_MASTER_HOST"
echo "2. Reconfigure Sentinel on all nodes"
echo "3. Investigate old master: $OLD_MASTER_HOST"
```

### 4.3 Planned Maintenance Failover

Use for scheduled maintenance with minimal disruption.

```bash
#!/bin/bash
# Planned maintenance failover procedure

SENTINEL_HOST="sentinel.greenlang.internal"
MASTER_NAME="mymaster"

echo "=== Planned Maintenance Failover ==="

# Step 1: Pre-flight checks
echo "[1/6] Pre-flight checks..."
redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL CKQUORUM $MASTER_NAME
if [ $? -ne 0 ]; then
  echo "ERROR: Quorum check failed. Aborting."
  exit 1
fi

# Step 2: Check replication lag
echo -e "\n[2/6] Checking replication lag..."
MASTER_IP=$(redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL get-master-addr-by-name $MASTER_NAME | head -1)
MASTER_OFFSET=$(redis-cli -h $MASTER_IP -p 6379 INFO replication | grep master_repl_offset | cut -d: -f2 | tr -d '\r')
echo "Master offset: $MASTER_OFFSET"

# Check replica offsets
redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL slaves $MASTER_NAME | grep -E "(ip|slave_repl_offset)"

# Step 3: Disable writes temporarily (optional, for zero data loss)
echo -e "\n[3/6] Pausing writes to ensure sync..."
redis-cli -h $MASTER_IP -p 6379 CLIENT PAUSE 5000 WRITE
sleep 2

# Step 4: Verify all replicas synced
echo -e "\n[4/6] Verifying replica sync..."
SYNCED=true
for slave_ip in $(redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL slaves $MASTER_NAME | grep "^ip" | cut -d: -f2 | tr -d '\r'); do
  SLAVE_OFFSET=$(redis-cli -h $slave_ip -p 6379 INFO replication | grep slave_repl_offset | cut -d: -f2 | tr -d '\r')
  if [ "$SLAVE_OFFSET" != "$MASTER_OFFSET" ]; then
    echo "WARNING: Replica $slave_ip offset ($SLAVE_OFFSET) differs from master ($MASTER_OFFSET)"
    SYNCED=false
  fi
done

# Step 5: Perform failover
echo -e "\n[5/6] Initiating failover..."
redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL FAILOVER $MASTER_NAME

# Wait for completion
sleep 10

# Step 6: Verify and report
echo -e "\n[6/6] Verification..."
NEW_MASTER=$(redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL get-master-addr-by-name $MASTER_NAME | head -1)
echo "New master: $NEW_MASTER"

if [ "$NEW_MASTER" != "$MASTER_IP" ]; then
  echo "SUCCESS: Failover completed"
else
  echo "ERROR: Failover may have failed. Investigate."
  exit 1
fi
```

---

## 5. Post-Failover Validation

### Immediate Validation (within 1 minute)

```bash
#!/bin/bash
# Post-failover validation script

SENTINEL_HOST="sentinel.greenlang.internal"
MASTER_NAME="mymaster"

echo "=== Post-Failover Validation ==="

# 1. Verify new master
echo -e "\n[1] New Master Status:"
NEW_MASTER=$(redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL get-master-addr-by-name $MASTER_NAME | head -1)
echo "Master IP: $NEW_MASTER"
redis-cli -h $NEW_MASTER -p 6379 INFO replication | grep -E "(role|connected_slaves)"

# 2. Check Sentinel agreement
echo -e "\n[2] Sentinel Agreement:"
for sentinel in sentinel-1 sentinel-2 sentinel-3; do
  SEEN_MASTER=$(redis-cli -h $sentinel.greenlang.internal -p 26379 \
    SENTINEL get-master-addr-by-name $MASTER_NAME | head -1)
  echo "$sentinel sees master: $SEEN_MASTER"
done

# 3. Verify all replicas connected
echo -e "\n[3] Replica Status:"
redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL slaves $MASTER_NAME | \
  grep -E "(name|flags|master-link-status)"

# 4. Test write capability
echo -e "\n[4] Write Test:"
TEST_KEY="failover_test_$(date +%s)"
redis-cli -h $NEW_MASTER -p 6379 SET $TEST_KEY "test_value"
RESULT=$(redis-cli -h $NEW_MASTER -p 6379 GET $TEST_KEY)
if [ "$RESULT" == "test_value" ]; then
  echo "Write test: PASSED"
  redis-cli -h $NEW_MASTER -p 6379 DEL $TEST_KEY
else
  echo "Write test: FAILED"
fi

# 5. Check for errors in logs
echo -e "\n[5] Recent Error Check:"
echo "Checking Sentinel logs for errors..."
grep -i error /var/log/redis/sentinel.log | tail -5

echo -e "\n=== Validation Complete ==="
```

### Extended Validation (within 5 minutes)

```bash
# Check application health
curl -f https://api.greenlang.io/api/v1/health

# Verify application can connect
kubectl exec -it $(kubectl get pod -l app=greenlang-app -o jsonpath='{.items[0].metadata.name}' -n greenlang) \
  -n greenlang -- redis-cli -h redis-master PING

# Check for connection errors in application logs
kubectl logs -l app=greenlang-app -n greenlang --since=5m | grep -i redis

# Verify metrics are being collected
curl -s http://prometheus:9090/api/v1/query?query=redis_up | jq '.data.result'
```

---

## 6. Application Reconnection

### Connection Strategy

Applications should use Sentinel-aware Redis clients that automatically discover the current master.

#### Python (redis-py with Sentinel)

```python
from redis.sentinel import Sentinel

sentinel = Sentinel([
    ('sentinel-1.greenlang.internal', 26379),
    ('sentinel-2.greenlang.internal', 26379),
    ('sentinel-3.greenlang.internal', 26379)
], socket_timeout=0.5)

# Get master connection (auto-failover aware)
master = sentinel.master_for('mymaster', socket_timeout=0.5)

# Get replica connection for reads
replica = sentinel.slave_for('mymaster', socket_timeout=0.5)
```

#### Node.js (ioredis)

```javascript
const Redis = require('ioredis');

const redis = new Redis({
  sentinels: [
    { host: 'sentinel-1.greenlang.internal', port: 26379 },
    { host: 'sentinel-2.greenlang.internal', port: 26379 },
    { host: 'sentinel-3.greenlang.internal', port: 26379 }
  ],
  name: 'mymaster',
  // Reconnection settings
  retryStrategy: (times) => Math.min(times * 50, 2000),
  maxRetriesPerRequest: 3
});
```

### Verifying Application Reconnection

```bash
# Check application connection to Redis
kubectl exec -it deployment/greenlang-app -n greenlang -- \
  python -c "from redis.sentinel import Sentinel; \
  s = Sentinel([('sentinel.greenlang.internal', 26379)]); \
  print('Master:', s.discover_master('mymaster')); \
  print('Slaves:', s.discover_slaves('mymaster'))"
```

---

## 7. Rollback Procedure

If the failover causes issues and you need to restore the original master:

```bash
#!/bin/bash
# Rollback to original master (use with caution)

ORIGINAL_MASTER="10.0.1.100"  # Original master IP
SENTINEL_HOST="sentinel.greenlang.internal"
MASTER_NAME="mymaster"

echo "=== Failover Rollback ==="
echo "WARNING: This will force the original master back to primary role"
read -p "Original master IP ($ORIGINAL_MASTER) is healthy? (yes/no): " confirm
[ "$confirm" != "yes" ] && exit 1

# Step 1: Verify original master is available
echo -e "\n[1/4] Checking original master..."
redis-cli -h $ORIGINAL_MASTER -p 6379 PING
if [ $? -ne 0 ]; then
  echo "ERROR: Original master is not responding"
  exit 1
fi

# Step 2: Set original master to highest priority
echo -e "\n[2/4] Setting failover priority..."
redis-cli -h $ORIGINAL_MASTER -p 6379 CONFIG SET replica-priority 1

# Step 3: Trigger failover
echo -e "\n[3/4] Triggering failover to original master..."
redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL FAILOVER $MASTER_NAME

# Step 4: Wait and verify
sleep 15
echo -e "\n[4/4] Verification..."
CURRENT_MASTER=$(redis-cli -h $SENTINEL_HOST -p 26379 SENTINEL get-master-addr-by-name $MASTER_NAME | head -1)

if [ "$CURRENT_MASTER" == "$ORIGINAL_MASTER" ]; then
  echo "SUCCESS: Rolled back to original master"
else
  echo "WARNING: Failover completed but master is $CURRENT_MASTER (expected $ORIGINAL_MASTER)"
fi
```

---

## 8. Troubleshooting Common Issues

### Issue: Failover Takes Too Long

**Symptoms**: Failover exceeds 30 seconds

**Resolution**:
```bash
# Check Sentinel configuration
redis-cli -h sentinel -p 26379 SENTINEL master mymaster | grep -E "(down-after|failover-timeout)"

# Reduce down-after-milliseconds if too high (default 30000)
redis-cli -h sentinel -p 26379 SENTINEL SET mymaster down-after-milliseconds 5000
```

### Issue: Split-Brain Scenario

**Symptoms**: Multiple nodes claim to be master

**Resolution**:
```bash
# Identify all claimed masters
for host in redis-1 redis-2 redis-3; do
  ROLE=$(redis-cli -h $host.greenlang.internal -p 6379 INFO replication | grep role | cut -d: -f2)
  echo "$host: $ROLE"
done

# Force consensus through Sentinel
redis-cli -h sentinel -p 26379 SENTINEL RESET mymaster
```

### Issue: Replica Won't Sync After Failover

**Symptoms**: Replica shows master_link_status:down

**Resolution**:
```bash
# Check connectivity
redis-cli -h replica -p 6379 DEBUG SLEEP 0  # Test if responsive

# Force resync
redis-cli -h replica -p 6379 REPLICAOF new-master-ip 6379

# If full resync needed
redis-cli -h replica -p 6379 DEBUG RELOAD NOSAVE
```

---

## Appendix: Quick Reference

### Essential Commands

| Action | Command |
|--------|---------|
| Get current master | `SENTINEL get-master-addr-by-name mymaster` |
| Trigger failover | `SENTINEL FAILOVER mymaster` |
| Check quorum | `SENTINEL CKQUORUM mymaster` |
| List replicas | `SENTINEL slaves mymaster` |
| Force replica sync | `REPLICAOF <master> 6379` |
| Promote to master | `REPLICAOF NO ONE` |

### Key Metrics to Monitor During Failover

- `redis_master_link_status` - Replica connection to master
- `redis_connected_slaves` - Number of connected replicas
- `redis_master_repl_offset` - Replication offset
- `redis_sentinel_masters` - Monitored masters count
- `redis_sentinel_tilt` - Sentinel tilt mode indicator
