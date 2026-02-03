# Redis Performance Tuning Guide

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Last Updated | 2026-02-03 |
| Owner | DevOps Team |
| Purpose | Performance optimization for Redis |

---

## 1. Overview

This guide covers performance tuning strategies for Redis deployments. Proper tuning can significantly improve throughput, reduce latency, and optimize resource utilization.

### Performance Tuning Areas

1. Memory Optimization
2. Connection Tuning
3. Persistence Tuning (AOF vs RDB)
4. Eviction Policy Selection
5. Pipeline Optimization
6. Cluster Mode Considerations
7. Operating System Tuning

---

## 2. Memory Optimization

### 2.1 Understanding Redis Memory Usage

```bash
# Get detailed memory information
redis-cli INFO memory

# Key metrics:
# - used_memory: Total bytes allocated by Redis
# - used_memory_rss: Resident Set Size (actual memory used)
# - mem_fragmentation_ratio: RSS / used_memory
# - used_memory_peak: Maximum memory ever used
```

### 2.2 Memory Configuration

```conf
# redis.conf memory settings

# Maximum memory limit (set based on available RAM)
# Leave ~25% for OS and other processes
maxmemory 12gb

# Memory policy when limit is reached
maxmemory-policy allkeys-lru

# Sample size for eviction algorithms
maxmemory-samples 10

# Enable active defragmentation (Redis 4.0+)
activedefrag yes
active-defrag-ignore-bytes 100mb
active-defrag-threshold-lower 10
active-defrag-threshold-upper 100
active-defrag-cycle-min 1
active-defrag-cycle-max 25
```

### 2.3 Memory Optimization Techniques

**Use Appropriate Data Structures**:

| Data Type | Use Case | Memory Efficiency |
|-----------|----------|-------------------|
| String | Simple values | Good for small values |
| Hash | Objects with fields | Better for objects (use hashes for small objects) |
| List | Queues, recent items | Good for sequential data |
| Set | Unique collections | Moderate |
| Sorted Set | Ranked data | Higher overhead (scores stored) |

**Hash Optimization (Small Hashes)**:

```conf
# Use ziplist encoding for small hashes (memory efficient)
hash-max-ziplist-entries 512
hash-max-ziplist-value 64

# Example: Store user data in hash vs separate keys
# Bad: user:1:name, user:1:email, user:1:age (3 keys)
# Good: HSET user:1 name "Alice" email "alice@example.com" age 30
```

**List Optimization**:

```conf
# Use quicklist (combination of ziplist and linked list)
list-max-ziplist-size -2  # -2 = 8KB per node
list-compress-depth 0     # 0 = no compression
```

**Set Optimization**:

```conf
# Use intset for integer-only sets
set-max-intset-entries 512
```

**Sorted Set Optimization**:

```conf
# Use ziplist for small sorted sets
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
```

### 2.4 Memory Analysis Tools

```bash
#!/bin/bash
# memory-analysis.sh - Analyze Redis memory usage

echo "=== Redis Memory Analysis ==="

# Overall memory
echo -e "\n--- Overall Memory ---"
redis-cli INFO memory | grep -E "(used_memory_human|used_memory_rss_human|mem_fragmentation_ratio|used_memory_peak_human)"

# Memory by data type
echo -e "\n--- Memory by Key Pattern ---"
redis-cli --scan --pattern "*" | head -10000 | while read key; do
  echo "$key $(redis-cli DEBUG OBJECT "$key" 2>/dev/null | grep -o 'serializedlength:[0-9]*')"
done | sort -t: -k2 -n -r | head -20

# Big keys detection
echo -e "\n--- Largest Keys ---"
redis-cli --bigkeys --no-auth-warning 2>/dev/null | grep -A 3 "Biggest"

# Memory fragmentation
echo -e "\n--- Fragmentation Analysis ---"
FRAG=$(redis-cli INFO memory | grep mem_fragmentation_ratio | cut -d: -f2 | tr -d '\r')
echo "Fragmentation ratio: $FRAG"
if (( $(echo "$FRAG > 1.5" | bc -l) )); then
  echo "WARNING: High fragmentation detected"
  echo "Consider: redis-cli MEMORY PURGE or restart"
fi
```

### 2.5 Handling Memory Fragmentation

```bash
# Check fragmentation
redis-cli INFO memory | grep mem_fragmentation

# Manual defragmentation (Redis 4.0+)
redis-cli MEMORY PURGE

# Enable active defragmentation
redis-cli CONFIG SET activedefrag yes

# For severe fragmentation, restart Redis
# This will reload data from RDB/AOF, defragmenting in the process
```

---

## 3. Connection Tuning

### 3.1 Connection Configuration

```conf
# Maximum number of clients
maxclients 10000

# TCP backlog (for high connection rates)
tcp-backlog 511

# TCP keepalive (seconds)
tcp-keepalive 300

# Client timeout (0 = disabled)
timeout 0

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
```

### 3.2 Connection Pooling Best Practices

**Python (redis-py)**:

```python
import redis

# Create connection pool
pool = redis.ConnectionPool(
    host='redis-master',
    port=6379,
    max_connections=50,          # Pool size
    socket_timeout=5,            # Operation timeout
    socket_connect_timeout=5,    # Connection timeout
    retry_on_timeout=True,       # Retry on timeout
    health_check_interval=30     # Health check frequency
)

# Use pool
r = redis.Redis(connection_pool=pool)
```

**Node.js (ioredis)**:

```javascript
const Redis = require('ioredis');

const redis = new Redis({
  host: 'redis-master',
  port: 6379,
  // Connection pool settings
  lazyConnect: true,
  maxRetriesPerRequest: 3,
  retryStrategy: (times) => Math.min(times * 50, 2000),
  // Socket options
  connectTimeout: 5000,
  commandTimeout: 5000,
  keepAlive: 30000
});

// For cluster mode
const cluster = new Redis.Cluster([
  { host: 'redis-1', port: 6379 },
  { host: 'redis-2', port: 6379 }
], {
  scaleReads: 'slave',
  redisOptions: {
    maxRetriesPerRequest: 3
  }
});
```

### 3.3 Connection Monitoring

```bash
#!/bin/bash
# monitor-connections.sh

echo "=== Connection Monitoring ==="

# Current connections
echo -e "\n--- Connection Stats ---"
redis-cli INFO clients

# Connected clients list
echo -e "\n--- Client List (top 10 by age) ---"
redis-cli CLIENT LIST | sort -t= -k6 -n -r | head -10

# Blocked clients
BLOCKED=$(redis-cli INFO clients | grep blocked_clients | cut -d: -f2 | tr -d '\r')
if [ "$BLOCKED" -gt 0 ]; then
  echo -e "\nWARNING: $BLOCKED blocked clients"
  redis-cli CLIENT LIST | grep "flags=b"
fi

# Connection rate
echo -e "\n--- Connection Rate ---"
redis-cli INFO stats | grep -E "(total_connections_received|rejected_connections)"
```

---

## 4. Persistence Tuning (AOF vs RDB)

### 4.1 Persistence Comparison

| Feature | RDB | AOF |
|---------|-----|-----|
| Data safety | Point-in-time | Near real-time |
| Recovery speed | Fast | Slower (replay) |
| File size | Smaller (compressed) | Larger |
| Performance impact | Periodic spikes | Continuous small impact |
| Recommended for | Backups, disaster recovery | Durability |

### 4.2 RDB Optimization

```conf
# Disable automatic RDB if using AOF for durability
# save ""

# Or optimize save intervals for your workload
save 900 1      # Save after 900s if 1+ keys changed
save 300 10     # Save after 300s if 10+ keys changed
save 60 10000   # Save after 60s if 10000+ keys changed

# Enable compression (CPU vs disk tradeoff)
rdbcompression yes

# RDB checksum (slight CPU overhead)
rdbchecksum yes

# Don't stop writes if RDB fails
stop-writes-on-bgsave-error no
```

### 4.3 AOF Optimization

```conf
# Enable AOF
appendonly yes

# Sync policy (choose based on durability needs)
# "always": Safest, slowest (fsync every write)
# "everysec": Good balance (fsync every second) - RECOMMENDED
# "no": Fastest, risky (OS handles fsync)
appendfsync everysec

# Don't fsync during rewrite (better performance)
no-appendfsync-on-rewrite yes

# Auto rewrite thresholds
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Load truncated AOF on startup (vs refusing to start)
aof-load-truncated yes

# Use RDB preamble in AOF (faster loading)
aof-use-rdb-preamble yes
```

### 4.4 Hybrid Persistence (Recommended)

```conf
# Use both RDB and AOF
# RDB for fast restarts, AOF for durability

appendonly yes
appendfsync everysec
aof-use-rdb-preamble yes

# Also create RDB snapshots for backup
save 900 1
save 300 10
save 60 10000
```

### 4.5 Monitor Persistence Performance

```bash
#!/bin/bash
# monitor-persistence.sh

echo "=== Persistence Performance ==="

# RDB status
echo -e "\n--- RDB Status ---"
redis-cli INFO persistence | grep -E "(rdb_|loading)"

# AOF status
echo -e "\n--- AOF Status ---"
redis-cli INFO persistence | grep -E "(aof_)"

# Last save times
echo -e "\n--- Save Times ---"
LAST_SAVE=$(redis-cli LASTSAVE)
LAST_SAVE_DATE=$(date -d @$LAST_SAVE)
echo "Last RDB save: $LAST_SAVE_DATE"

# Background save in progress?
BG_SAVE=$(redis-cli INFO persistence | grep rdb_bgsave_in_progress | cut -d: -f2 | tr -d '\r')
if [ "$BG_SAVE" == "1" ]; then
  echo "WARNING: Background save in progress"
fi

# AOF rewrite in progress?
AOF_REWRITE=$(redis-cli INFO persistence | grep aof_rewrite_in_progress | cut -d: -f2 | tr -d '\r')
if [ "$AOF_REWRITE" == "1" ]; then
  echo "WARNING: AOF rewrite in progress"
fi
```

---

## 5. Eviction Policy Selection

### 5.1 Available Policies

| Policy | Description | Best For |
|--------|-------------|----------|
| `noeviction` | Return errors when memory limit reached | When data loss is unacceptable |
| `allkeys-lru` | Evict least recently used keys | Cache with mixed access patterns |
| `allkeys-lfu` | Evict least frequently used keys | Cache with skewed access (hot keys) |
| `volatile-lru` | LRU among keys with TTL | Cache with expiring keys |
| `volatile-lfu` | LFU among keys with TTL | Cache with expiring keys, hot keys |
| `allkeys-random` | Random eviction | When access patterns unpredictable |
| `volatile-random` | Random among keys with TTL | Simple cache scenarios |
| `volatile-ttl` | Evict keys with shortest TTL | When TTL indicates importance |

### 5.2 Choosing the Right Policy

```
EVICTION POLICY DECISION TREE:

                    ┌───────────────────────────┐
                    │ Can you lose data?        │
                    └───────────────┬───────────┘
                            Yes     │     No
                    ┌───────────────┴───────────┐
                    │                           │
                    ▼                           ▼
        ┌───────────────────┐       ┌───────────────────┐
        │ Do all keys have  │       │   noeviction      │
        │ TTL (expire)?     │       │                   │
        └─────────┬─────────┘       └───────────────────┘
           Yes    │    No
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ Hot keys?     │   │ Hot keys?     │
└───────┬───────┘   └───────┬───────┘
  Yes   │   No        Yes   │   No
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│volatile-lfu   │   │ allkeys-lfu   │
│or volatile-lru│   │ or allkeys-lru│
└───────────────┘   └───────────────┘
```

### 5.3 Eviction Configuration

```conf
# Set eviction policy
maxmemory-policy allkeys-lfu

# Number of samples for eviction (higher = more accurate, slower)
maxmemory-samples 10

# For LFU: decay time for frequency counter
lfu-decay-time 1

# For LFU: initial counter value for new keys
lfu-log-factor 10
```

### 5.4 Monitor Eviction

```bash
# Check eviction stats
redis-cli INFO stats | grep evicted_keys

# Monitor eviction in real-time
redis-cli MONITOR | grep -i del  # Watch for evictions

# Alert on high eviction rate
EVICTED=$(redis-cli INFO stats | grep evicted_keys | cut -d: -f2 | tr -d '\r')
echo "Total evicted keys: $EVICTED"
```

---

## 6. Pipeline Optimization

### 6.1 Understanding Pipelines

```
WITHOUT PIPELINE:
Client         Redis
  │───SET k1───────►│
  │◄──────OK────────│
  │───SET k2───────►│
  │◄──────OK────────│
  │───SET k3───────►│
  │◄──────OK────────│

Total: 6 network round trips

WITH PIPELINE:
Client         Redis
  │───SET k1───────►│
  │───SET k2───────►│
  │───SET k3───────►│
  │◄──────OK────────│
  │◄──────OK────────│
  │◄──────OK────────│

Total: 1 network round trip
```

### 6.2 Pipeline Implementation

**Python**:

```python
import redis

r = redis.Redis()

# Without pipeline (slow)
for i in range(10000):
    r.set(f'key:{i}', f'value:{i}')

# With pipeline (fast)
pipe = r.pipeline(transaction=False)
for i in range(10000):
    pipe.set(f'key:{i}', f'value:{i}')
pipe.execute()

# With pipeline and batching (memory efficient for large datasets)
BATCH_SIZE = 1000
for batch_start in range(0, 100000, BATCH_SIZE):
    pipe = r.pipeline(transaction=False)
    for i in range(batch_start, min(batch_start + BATCH_SIZE, 100000)):
        pipe.set(f'key:{i}', f'value:{i}')
    pipe.execute()
```

**Node.js**:

```javascript
const Redis = require('ioredis');
const redis = new Redis();

// Pipeline
const pipeline = redis.pipeline();
for (let i = 0; i < 10000; i++) {
  pipeline.set(`key:${i}`, `value:${i}`);
}
await pipeline.exec();

// Multi-command pipeline with results
const results = await redis.pipeline()
  .set('foo', 'bar')
  .get('foo')
  .incr('counter')
  .exec();
```

### 6.3 Pipeline Best Practices

| Scenario | Recommendation |
|----------|---------------|
| Batch size | 100-1000 commands per pipeline |
| Memory concern | Batch large pipelines |
| Mixed read/write | Avoid transactions unless needed |
| Error handling | Check individual results |
| Timeout | Set appropriate pipeline timeout |

```python
# Error handling in pipeline
pipe = r.pipeline(transaction=False)
pipe.set('key1', 'value1')
pipe.get('nonexistent')
pipe.incr('counter')
results = pipe.execute(raise_on_error=False)

for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Command {i} failed: {result}")
```

---

## 7. Cluster Mode Considerations

### 7.1 Cluster vs Sentinel

| Feature | Sentinel | Cluster |
|---------|----------|---------|
| Scaling | Vertical (single master) | Horizontal (multiple masters) |
| Data size | Up to available RAM | Distributed across nodes |
| Complexity | Simple | More complex |
| Multi-key operations | Supported | Must be in same slot |
| Use case | HA for single instance | Large datasets, high throughput |

### 7.2 Cluster Configuration

```conf
# Enable cluster mode
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000

# Require full coverage (all slots must be available)
cluster-require-full-coverage yes

# Migration settings
cluster-migration-barrier 1

# Replica validity factor (for automatic failover)
cluster-replica-validity-factor 10
```

### 7.3 Cluster Performance Optimization

**Hash Tags for Multi-Key Operations**:

```python
# Without hash tags - may fail in cluster
r.mset({'user:1:name': 'Alice', 'user:1:email': 'alice@example.com'})  # May hit different slots!

# With hash tags - guaranteed same slot
r.mset({'{user:1}:name': 'Alice', '{user:1}:email': 'alice@example.com'})  # Same slot

# Hash tag syntax: {tag}key
# The tag part determines the slot
```

**Client-Side Routing Optimization**:

```python
from redis.cluster import RedisCluster

# Enable read from replicas for better throughput
rc = RedisCluster(
    host='redis-cluster',
    port=6379,
    read_from_replicas=True  # Distribute reads
)
```

### 7.4 Monitor Cluster Performance

```bash
#!/bin/bash
# cluster-performance.sh

echo "=== Cluster Performance ==="

# Cluster info
echo -e "\n--- Cluster State ---"
redis-cli -c CLUSTER INFO

# Slot distribution
echo -e "\n--- Slot Distribution ---"
redis-cli -c CLUSTER SLOTS | head -50

# Node memory usage
echo -e "\n--- Memory per Node ---"
for node in $(redis-cli -c CLUSTER NODES | awk '{print $2}' | cut -d: -f1); do
  MEM=$(redis-cli -h $node INFO memory | grep used_memory_human | cut -d: -f2)
  echo "$node: $MEM"
done

# Redirects (MOVED/ASK)
echo -e "\n--- Redirect Stats ---"
redis-cli INFO stats | grep -E "(keyspace_hits|keyspace_misses)"
```

---

## 8. Operating System Tuning

### 8.1 Linux Kernel Parameters

```bash
# /etc/sysctl.conf additions for Redis

# Increase max connections
net.core.somaxconn = 65535

# Increase backlog
net.ipv4.tcp_max_syn_backlog = 65535

# Disable transparent huge pages (important!)
# Add to /etc/rc.local or systemd service:
# echo never > /sys/kernel/mm/transparent_hugepage/enabled

# Memory overcommit (allow fork for RDB/AOF rewrite)
vm.overcommit_memory = 1

# Swappiness (minimize swapping)
vm.swappiness = 1

# Apply changes
sysctl -p
```

### 8.2 File Descriptors

```bash
# Increase file descriptor limits for Redis user
# /etc/security/limits.conf
redis soft nofile 65535
redis hard nofile 65535

# Or in systemd service file
[Service]
LimitNOFILE=65535
```

### 8.3 Disable Transparent Huge Pages

```bash
# Check current status
cat /sys/kernel/mm/transparent_hugepage/enabled

# Disable (temporary)
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Disable (permanent) - create systemd service
cat > /etc/systemd/system/disable-thp.service << 'EOF'
[Unit]
Description=Disable Transparent Huge Pages
After=sysinit.target local-fs.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo never > /sys/kernel/mm/transparent_hugepage/enabled && echo never > /sys/kernel/mm/transparent_hugepage/defrag'

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable disable-thp
systemctl start disable-thp
```

### 8.4 Network Optimization

```bash
# Increase network buffers
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Enable TCP window scaling
net.ipv4.tcp_window_scaling = 1

# Reuse TIME_WAIT connections
net.ipv4.tcp_tw_reuse = 1
```

---

## 9. Performance Monitoring

### 9.1 Key Metrics to Monitor

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `used_memory_rss` | Actual RAM used | > 90% of maxmemory |
| `mem_fragmentation_ratio` | Memory fragmentation | > 1.5 or < 1.0 |
| `connected_clients` | Client connections | > 80% of maxclients |
| `blocked_clients` | Clients waiting | > 0 for extended time |
| `instantaneous_ops_per_sec` | Current throughput | Baseline deviation |
| `keyspace_hits/misses` | Cache hit ratio | < 80% hit rate |
| `evicted_keys` | Keys evicted | > 0 (unexpected eviction) |
| `rejected_connections` | Connection rejections | > 0 |

### 9.2 Performance Dashboard Query

```bash
#!/bin/bash
# performance-dashboard.sh

echo "=== Redis Performance Dashboard ==="
echo "Time: $(date)"

# Get all INFO
INFO=$(redis-cli INFO all)

# Memory
echo -e "\n--- Memory ---"
echo "$INFO" | grep -E "^(used_memory_human|used_memory_rss_human|mem_fragmentation_ratio|maxmemory_human)"

# Connections
echo -e "\n--- Connections ---"
echo "$INFO" | grep -E "^(connected_clients|blocked_clients|rejected_connections)"

# Throughput
echo -e "\n--- Throughput ---"
echo "$INFO" | grep -E "^(instantaneous_ops_per_sec|total_commands_processed)"

# Cache performance
echo -e "\n--- Cache Performance ---"
HITS=$(echo "$INFO" | grep "^keyspace_hits:" | cut -d: -f2)
MISSES=$(echo "$INFO" | grep "^keyspace_misses:" | cut -d: -f2)
if [ -n "$HITS" ] && [ -n "$MISSES" ]; then
  TOTAL=$((HITS + MISSES))
  if [ "$TOTAL" -gt 0 ]; then
    HIT_RATE=$(echo "scale=2; $HITS * 100 / $TOTAL" | bc)
    echo "Hit rate: ${HIT_RATE}%"
  fi
fi

# Latency
echo -e "\n--- Latency Check ---"
redis-cli --latency-history -i 1 -c 5 2>/dev/null || echo "Latency test: $(redis-cli DEBUG SLEEP 0.001 2>/dev/null; echo 'OK')"

# Slow log
echo -e "\n--- Recent Slow Commands ---"
redis-cli SLOWLOG GET 5
```

### 9.3 Benchmark Testing

```bash
# Built-in benchmark tool
redis-benchmark -h localhost -p 6379 -c 50 -n 100000 -q

# Test specific commands
redis-benchmark -t set,get,lpush,lpop -c 50 -n 100000

# Pipeline benchmark
redis-benchmark -t set -c 50 -n 100000 -P 16  # 16 commands per pipeline

# Test with specific data size
redis-benchmark -t set -c 50 -n 100000 -d 1000  # 1KB values
```

---

## Appendix: Configuration Template

```conf
# Redis optimized configuration template
# Adjust values based on your hardware and workload

# Network
bind 0.0.0.0
port 6379
tcp-backlog 511
tcp-keepalive 300
timeout 0

# Memory
maxmemory 12gb
maxmemory-policy allkeys-lfu
maxmemory-samples 10

# Persistence (hybrid)
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite yes
aof-use-rdb-preamble yes
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Replication
replica-serve-stale-data yes
replica-read-only yes
repl-diskless-sync yes
repl-backlog-size 512mb

# Performance
activedefrag yes
active-defrag-ignore-bytes 100mb
active-defrag-threshold-lower 10
active-defrag-threshold-upper 100
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes

# Data structure optimization
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Clients
maxclients 10000
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
```
