# Redis Sentinel Operations Guide

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | DevOps Team |
| Purpose | Sentinel cluster management and troubleshooting |

---

## 1. Overview

Redis Sentinel provides high availability for Redis through automatic failover, monitoring, and configuration provider services. This guide covers day-to-day operations of the Sentinel cluster.

### Sentinel Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Sentinel Cluster            │
                    │                                     │
                    │  ┌─────────┐ ┌─────────┐ ┌───────┐ │
                    │  │Sentinel │ │Sentinel │ │Sentinel│ │
                    │  │   #1    │ │   #2    │ │   #3   │ │
                    │  │  AZ-A   │ │  AZ-B   │ │  AZ-C  │ │
                    │  └────┬────┘ └────┬────┘ └───┬────┘ │
                    │       │           │          │      │
                    └───────┼───────────┼──────────┼──────┘
                            │           │          │
                            └─────┬─────┴─────┬────┘
                                  │           │
                                  ▼           ▼
                            ┌──────────┐ ┌──────────┐
                            │  Master  │ │ Replicas │
                            └──────────┘ └──────────┘
```

### Sentinel Configuration (sentinel.conf)

```conf
# Sentinel port
port 26379

# Monitor master named "mymaster" at given IP/port
# Quorum of 2 = at least 2 Sentinels must agree for failover
sentinel monitor mymaster 10.0.1.100 6379 2

# Consider master down after 5 seconds of no response
sentinel down-after-milliseconds mymaster 5000

# Number of replicas to reconfigure simultaneously after failover
sentinel parallel-syncs mymaster 1

# Failover timeout (how long to wait before considering failover failed)
sentinel failover-timeout mymaster 60000

# Authentication
sentinel auth-pass mymaster your-redis-password

# Sentinel own password
requirepass your-sentinel-password

# Announce IP (for NAT/container environments)
sentinel announce-ip 10.0.1.50
sentinel announce-port 26379
```

---

## 2. Sentinel Cluster Management

### 2.1 Checking Cluster Health

```bash
#!/bin/bash
# sentinel-health-check.sh

SENTINELS=("sentinel-1.greenlang.internal" "sentinel-2.greenlang.internal" "sentinel-3.greenlang.internal")
MASTER_NAME="mymaster"

echo "=== Sentinel Cluster Health Check ==="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# Check each Sentinel
for sentinel in "${SENTINELS[@]}"; do
  echo -e "\n--- $sentinel ---"

  # Ping check
  PING=$(redis-cli -h $sentinel -p 26379 PING 2>/dev/null)
  if [ "$PING" == "PONG" ]; then
    echo "Status: UP"

    # Get master info
    MASTER=$(redis-cli -h $sentinel -p 26379 SENTINEL get-master-addr-by-name $MASTER_NAME 2>/dev/null | head -1)
    echo "Sees Master: $MASTER"

    # Get sentinel count
    SENTINEL_COUNT=$(redis-cli -h $sentinel -p 26379 SENTINEL master $MASTER_NAME 2>/dev/null | grep num-other-sentinels -A 1 | tail -1)
    echo "Other Sentinels: $SENTINEL_COUNT"

    # Get slave count
    SLAVE_COUNT=$(redis-cli -h $sentinel -p 26379 SENTINEL master $MASTER_NAME 2>/dev/null | grep num-slaves -A 1 | tail -1)
    echo "Slaves: $SLAVE_COUNT"
  else
    echo "Status: DOWN"
  fi
done

# Quorum check
echo -e "\n=== Quorum Check ==="
redis-cli -h ${SENTINELS[0]} -p 26379 SENTINEL CKQUORUM $MASTER_NAME
```

### 2.2 Viewing Sentinel Information

```bash
# Get all information about monitored master
redis-cli -h sentinel -p 26379 SENTINEL master mymaster

# Get list of all sentinels monitoring this master
redis-cli -h sentinel -p 26379 SENTINEL sentinels mymaster

# Get list of all slaves for this master
redis-cli -h sentinel -p 26379 SENTINEL slaves mymaster

# Get current master address
redis-cli -h sentinel -p 26379 SENTINEL get-master-addr-by-name mymaster

# Check if Sentinel can reach quorum
redis-cli -h sentinel -p 26379 SENTINEL CKQUORUM mymaster

# Get Sentinel's pending scripts
redis-cli -h sentinel -p 26379 SENTINEL pending-scripts
```

---

## 3. Adding Sentinels

### 3.1 Adding a New Sentinel Node

**Step 1: Prepare the new node**

```bash
# Install Redis on new node
apt-get update && apt-get install -y redis-sentinel

# Create sentinel configuration
cat > /etc/redis/sentinel.conf << 'EOF'
port 26379
daemonize yes
pidfile /var/run/redis/redis-sentinel.pid
logfile /var/log/redis/sentinel.log
dir /var/lib/redis

# Monitor configuration
sentinel monitor mymaster 10.0.1.100 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 60000

# Authentication
sentinel auth-pass mymaster your-redis-password
requirepass your-sentinel-password

# Announce settings (adjust for your network)
sentinel announce-ip NEW_SENTINEL_IP
sentinel announce-port 26379
EOF
```

**Step 2: Start the new Sentinel**

```bash
# Start sentinel service
systemctl start redis-sentinel

# Verify it's running
redis-cli -h localhost -p 26379 PING

# Check it discovered the master
redis-cli -h localhost -p 26379 SENTINEL master mymaster
```

**Step 3: Verify cluster sees new Sentinel**

```bash
# On existing Sentinel, check other sentinels count
redis-cli -h sentinel-1 -p 26379 SENTINEL sentinels mymaster

# Should show the new sentinel in the list
```

### 3.2 Updating Quorum After Adding Sentinels

When you add more Sentinels, you may want to update the quorum:

```bash
# Calculate new quorum (majority)
# For 3 Sentinels: quorum = 2
# For 5 Sentinels: quorum = 3
# For 7 Sentinels: quorum = 4

# Update quorum on all Sentinels
for sentinel in sentinel-1 sentinel-2 sentinel-3 sentinel-4 sentinel-5; do
  redis-cli -h $sentinel -p 26379 SENTINEL SET mymaster quorum 3
done

# Verify the change
redis-cli -h sentinel-1 -p 26379 SENTINEL master mymaster | grep -A 1 quorum
```

---

## 4. Removing Sentinels

### 4.1 Graceful Sentinel Removal

```bash
#!/bin/bash
# remove-sentinel.sh

SENTINEL_TO_REMOVE="sentinel-5.greenlang.internal"
REMAINING_SENTINELS=("sentinel-1" "sentinel-2" "sentinel-3" "sentinel-4")
MASTER_NAME="mymaster"

echo "=== Removing Sentinel: $SENTINEL_TO_REMOVE ==="

# Step 1: Stop the Sentinel service
echo "[1/4] Stopping Sentinel service..."
ssh $SENTINEL_TO_REMOVE "systemctl stop redis-sentinel"

# Step 2: Wait for other Sentinels to detect it's down
echo "[2/4] Waiting for detection (30 seconds)..."
sleep 30

# Step 3: Reset Sentinel state on remaining nodes
echo "[3/4] Resetting Sentinel state on remaining nodes..."
for sentinel in "${REMAINING_SENTINELS[@]}"; do
  echo "Resetting $sentinel..."
  redis-cli -h $sentinel.greenlang.internal -p 26379 SENTINEL RESET $MASTER_NAME
done

# Step 4: Update quorum if necessary
echo "[4/4] Updating quorum..."
NEW_QUORUM=2  # Adjust based on remaining Sentinels
for sentinel in "${REMAINING_SENTINELS[@]}"; do
  redis-cli -h $sentinel.greenlang.internal -p 26379 SENTINEL SET $MASTER_NAME quorum $NEW_QUORUM
done

# Verify
echo -e "\n=== Verification ==="
redis-cli -h ${REMAINING_SENTINELS[0]}.greenlang.internal -p 26379 SENTINEL sentinels $MASTER_NAME | grep -c "name"
```

### 4.2 Emergency Sentinel Removal

When a Sentinel is permanently unavailable:

```bash
# On each remaining Sentinel, run RESET to clear stale info
redis-cli -h sentinel -p 26379 SENTINEL RESET mymaster

# This will:
# - Clear info about failed Sentinels
# - Re-discover healthy Sentinels
# - Re-evaluate master and slaves
```

---

## 5. Monitoring Sentinel Health

### 5.1 Key Metrics to Monitor

```bash
# Sentinel INFO command
redis-cli -h sentinel -p 26379 INFO

# Key metrics:
# - sentinel_masters: Number of monitored masters
# - sentinel_running_scripts: Currently running scripts
# - sentinel_scripts_queue_length: Pending notification scripts
# - sentinel_tilt: If 1, Sentinel is in TILT mode (unreliable timing)
```

### 5.2 Prometheus Metrics

```yaml
# prometheus-sentinel-rules.yml
groups:
  - name: sentinel_alerts
    rules:
      - alert: SentinelDown
        expr: redis_sentinel_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis Sentinel {{ $labels.instance }} is down"

      - alert: SentinelTiltMode
        expr: redis_sentinel_tilt == 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Sentinel {{ $labels.instance }} is in TILT mode"

      - alert: SentinelNoQuorum
        expr: redis_sentinel_masters_up < redis_sentinel_quorum
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Sentinel cluster cannot reach quorum"

      - alert: SentinelMasterChanged
        expr: changes(redis_sentinel_master_address[5m]) > 0
        labels:
          severity: warning
        annotations:
          summary: "Redis master changed in the last 5 minutes"
```

### 5.3 Health Check Script

```bash
#!/bin/bash
# sentinel-monitor.sh - Run as cron job

SENTINELS=("sentinel-1" "sentinel-2" "sentinel-3")
MASTER_NAME="mymaster"
ALERT_WEBHOOK="https://hooks.slack.com/services/xxx/yyy/zzz"

check_sentinel_health() {
  local sentinel=$1
  local result=$(redis-cli -h $sentinel.greenlang.internal -p 26379 PING 2>/dev/null)

  if [ "$result" != "PONG" ]; then
    return 1
  fi

  # Check for TILT mode
  local tilt=$(redis-cli -h $sentinel.greenlang.internal -p 26379 INFO | grep sentinel_tilt | cut -d: -f2 | tr -d '\r')
  if [ "$tilt" == "1" ]; then
    return 2
  fi

  return 0
}

# Main check loop
HEALTHY=0
UNHEALTHY=0
TILT=0

for sentinel in "${SENTINELS[@]}"; do
  check_sentinel_health $sentinel
  case $? in
    0) ((HEALTHY++)) ;;
    1) ((UNHEALTHY++)); echo "ALERT: $sentinel is DOWN" ;;
    2) ((TILT++)); echo "WARNING: $sentinel is in TILT mode" ;;
  esac
done

# Check quorum
QUORUM=2
if [ $HEALTHY -lt $QUORUM ]; then
  echo "CRITICAL: Quorum cannot be reached ($HEALTHY/$QUORUM sentinels healthy)"

  # Send alert
  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"CRITICAL: Redis Sentinel quorum lost! Only $HEALTHY sentinels healthy.\"}" \
    $ALERT_WEBHOOK
fi

echo "Summary: Healthy=$HEALTHY, Unhealthy=$UNHEALTHY, Tilt=$TILT"
```

---

## 6. Troubleshooting Sentinel Issues

### 6.1 Sentinel Not Detecting Master

**Symptoms**: Sentinel shows master as down but master is actually running

**Diagnosis**:
```bash
# Check network connectivity from Sentinel to master
redis-cli -h master-ip -p 6379 PING

# Check Sentinel's view
redis-cli -h sentinel -p 26379 SENTINEL master mymaster | grep -E "(ip|port|flags)"

# Check for authentication issues
redis-cli -h master-ip -p 6379 AUTH your-password
redis-cli -h master-ip -p 6379 PING
```

**Resolution**:
```bash
# If auth issue, update Sentinel config
redis-cli -h sentinel -p 26379 SENTINEL SET mymaster auth-pass your-correct-password

# If network issue, check firewall rules
iptables -L -n | grep 6379

# Reset Sentinel state
redis-cli -h sentinel -p 26379 SENTINEL RESET mymaster
```

### 6.2 Sentinel TILT Mode

**Symptoms**: Sentinel enters TILT mode, becomes unreliable

**Causes**:
- System time changed
- System overload
- Long blocking operations

**Diagnosis**:
```bash
# Check TILT status
redis-cli -h sentinel -p 26379 INFO | grep tilt

# Check system load
uptime
vmstat 1 5
```

**Resolution**:
```bash
# TILT mode clears automatically after 30 seconds of stable operation
# If persistent, restart Sentinel
systemctl restart redis-sentinel

# Verify TILT cleared
redis-cli -h sentinel -p 26379 INFO | grep tilt
```

### 6.3 Sentinel State Desync

**Symptoms**: Sentinels have different views of master/slaves

**Diagnosis**:
```bash
# Compare master view across all Sentinels
for sentinel in sentinel-1 sentinel-2 sentinel-3; do
  echo "$sentinel sees:"
  redis-cli -h $sentinel.greenlang.internal -p 26379 SENTINEL get-master-addr-by-name mymaster
done
```

**Resolution**:
```bash
# Reset all Sentinels to force re-discovery
for sentinel in sentinel-1 sentinel-2 sentinel-3; do
  redis-cli -h $sentinel.greenlang.internal -p 26379 SENTINEL RESET mymaster
done

# Wait for re-discovery (30 seconds)
sleep 30

# Verify consistency
for sentinel in sentinel-1 sentinel-2 sentinel-3; do
  redis-cli -h $sentinel.greenlang.internal -p 26379 SENTINEL get-master-addr-by-name mymaster
done
```

### 6.4 Failover Not Triggering

**Symptoms**: Master is down but failover doesn't happen

**Diagnosis**:
```bash
# Check if ODOWN was reached
redis-cli -h sentinel -p 26379 SENTINEL master mymaster | grep flags
# Should show "o_down" for failover to trigger

# Check quorum
redis-cli -h sentinel -p 26379 SENTINEL CKQUORUM mymaster

# Check if another failover is in progress
redis-cli -h sentinel -p 26379 SENTINEL failover-status mymaster
```

**Resolution**:
```bash
# If quorum issue, check Sentinel connectivity
for sentinel in sentinel-1 sentinel-2 sentinel-3; do
  redis-cli -h $sentinel.greenlang.internal -p 26379 PING
done

# If failover stuck, check failover timeout
redis-cli -h sentinel -p 26379 SENTINEL master mymaster | grep failover-timeout

# Force failover manually if needed
redis-cli -h sentinel -p 26379 SENTINEL FAILOVER mymaster
```

### 6.5 Sentinel Reports Wrong Number of Replicas

**Symptoms**: `num-slaves` doesn't match actual replica count

**Resolution**:
```bash
# Reset Sentinel to re-discover replicas
redis-cli -h sentinel -p 26379 SENTINEL RESET mymaster

# Manually check replicas
redis-cli -h master -p 6379 INFO replication | grep slave

# If replica is missing, check its connectivity
redis-cli -h replica -p 6379 INFO replication
```

---

## 7. Reset Sentinel State

### 7.1 Soft Reset (SENTINEL RESET)

Clears state for a specific master and re-discovers:

```bash
# Reset state for mymaster
redis-cli -h sentinel -p 26379 SENTINEL RESET mymaster

# This will:
# - Remove all slaves and sentinels for this master
# - Re-discover master, slaves, and other sentinels
# - Clear failover state
```

### 7.2 Hard Reset (Full Reconfiguration)

Use when Sentinel state is severely corrupted:

```bash
#!/bin/bash
# sentinel-hard-reset.sh

SENTINEL_HOST="sentinel-1.greenlang.internal"
MASTER_IP="10.0.1.100"
MASTER_PORT="6379"
MASTER_NAME="mymaster"
QUORUM="2"

echo "=== Sentinel Hard Reset ==="
echo "WARNING: This will completely reconfigure Sentinel"

# Step 1: Stop Sentinel
echo "[1/5] Stopping Sentinel..."
systemctl stop redis-sentinel

# Step 2: Remove old state
echo "[2/5] Removing old state..."
rm -f /var/lib/redis/sentinel.conf.bak
mv /etc/redis/sentinel.conf /etc/redis/sentinel.conf.bak

# Step 3: Create fresh configuration
echo "[3/5] Creating fresh configuration..."
cat > /etc/redis/sentinel.conf << EOF
port 26379
daemonize yes
pidfile /var/run/redis/redis-sentinel.pid
logfile /var/log/redis/sentinel.log
dir /var/lib/redis

sentinel monitor $MASTER_NAME $MASTER_IP $MASTER_PORT $QUORUM
sentinel down-after-milliseconds $MASTER_NAME 5000
sentinel parallel-syncs $MASTER_NAME 1
sentinel failover-timeout $MASTER_NAME 60000

sentinel auth-pass $MASTER_NAME your-redis-password
requirepass your-sentinel-password
EOF

# Step 4: Start Sentinel
echo "[4/5] Starting Sentinel..."
systemctl start redis-sentinel

# Step 5: Verify
echo "[5/5] Verification..."
sleep 5
redis-cli -h localhost -p 26379 SENTINEL master $MASTER_NAME | head -10
```

---

## 8. Sentinel Best Practices

### Configuration Best Practices

1. **Odd number of Sentinels**: Use 3, 5, or 7 Sentinels for proper quorum
2. **Distributed placement**: Place Sentinels across availability zones
3. **Quorum setting**: Set to majority (e.g., 2 for 3 Sentinels, 3 for 5)
4. **Timeouts**: Balance between quick detection and false positives
   - `down-after-milliseconds`: 5000-30000ms
   - `failover-timeout`: 60000-180000ms

### Operational Best Practices

1. **Monitor all Sentinels**: Don't just rely on one Sentinel's view
2. **Test failovers regularly**: Monthly failover drills in non-prod
3. **Keep Sentinels updated**: Match Redis version with Sentinel version
4. **Use authentication**: Always configure `auth-pass` and `requirepass`
5. **Network security**: Firewall Sentinel ports, use internal networks

### Alerting Best Practices

| Alert | Threshold | Severity |
|-------|-----------|----------|
| Sentinel down | 1 minute | Critical |
| Quorum at risk | Healthy < Quorum | Critical |
| TILT mode | 5 minutes | Warning |
| Failover occurred | Any | Info |
| Configuration drift | Hourly check | Warning |

---

## Appendix: Sentinel Commands Reference

| Command | Description |
|---------|-------------|
| `SENTINEL masters` | List all monitored masters |
| `SENTINEL master <name>` | Show master details |
| `SENTINEL slaves <name>` | List slaves for master |
| `SENTINEL sentinels <name>` | List other Sentinels |
| `SENTINEL get-master-addr-by-name <name>` | Get master IP:port |
| `SENTINEL FAILOVER <name>` | Force failover |
| `SENTINEL CKQUORUM <name>` | Check quorum |
| `SENTINEL RESET <name>` | Reset master state |
| `SENTINEL SET <name> <option> <value>` | Change configuration |
| `SENTINEL MONITOR <name> <ip> <port> <quorum>` | Add master to monitor |
| `SENTINEL REMOVE <name>` | Stop monitoring master |
| `SENTINEL FLUSHCONFIG` | Force config rewrite |
