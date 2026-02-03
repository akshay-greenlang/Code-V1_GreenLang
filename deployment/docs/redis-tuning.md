# Redis Performance Tuning Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Memory Configuration Guidelines](#memory-configuration-guidelines)
3. [Persistence Tradeoffs](#persistence-tradeoffs)
4. [Connection Pooling Best Practices](#connection-pooling-best-practices)
5. [Pipeline Optimization](#pipeline-optimization)
6. [Cluster Mode Considerations](#cluster-mode-considerations)
7. [Workload-Specific Configurations](#workload-specific-configurations)
8. [Monitoring and Alerting](#monitoring-and-alerting)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

This guide provides comprehensive Redis performance tuning recommendations for GreenLang's infrastructure. Redis serves multiple workloads including caching, session storage, message queuing, and real-time data processing.

### Key Principles

1. **Right-size for your workload** - Different workloads require different configurations
2. **Monitor first, tune second** - Use metrics to identify bottlenecks before making changes
3. **Test changes in staging** - Always validate configuration changes before production
4. **Document all changes** - Maintain a changelog of configuration modifications

---

## Memory Configuration Guidelines

### Understanding Memory Usage

Redis stores all data in memory. Understanding memory usage is critical for proper sizing.

```
Total Memory = Dataset Memory + Overhead + Fragmentation + Replication Buffers
```

#### Dataset Memory

The actual data stored in Redis:

| Data Type | Memory Overhead |
|-----------|-----------------|
| String (small) | 48 bytes + string length |
| String (large) | 48 bytes + string length |
| Hash (small) | ~100 bytes + ziplist data |
| Hash (large) | 200+ bytes + hashtable |
| List (small) | ~100 bytes + quicklist |
| Set (small) | ~100 bytes + intset |
| Sorted Set | 128+ bytes per element |

#### Memory Overhead

- **Per-key overhead**: ~48 bytes per key
- **Per-value overhead**: ~16 bytes per value
- **Expiration overhead**: ~16 bytes per key with TTL

### Memory Sizing Guidelines

```yaml
# Conservative sizing formula
required_memory = (average_key_size + average_value_size + 64) * estimated_key_count * 1.5

# Example calculation for 10 million keys
# Average key: 50 bytes, Average value: 200 bytes
# (50 + 200 + 64) * 10,000,000 * 1.5 = 4.71 GB
```

### maxmemory Configuration

Set `maxmemory` based on your workload:

| Workload | Recommended maxmemory |
|----------|----------------------|
| Caching | 70-80% of available RAM |
| Session Storage | 60-70% of available RAM |
| Message Queue | 50-60% of available RAM |
| Production (mixed) | 60-70% of available RAM |

```conf
# Leave 20-30% headroom for:
# - Memory fragmentation
# - Fork operations (RDB/AOF rewrite)
# - Client output buffers
# - Replication buffers

# Example: 8GB container with caching workload
maxmemory 6gb
```

### Eviction Policies

Choose the right eviction policy for your workload:

| Policy | Use Case | Description |
|--------|----------|-------------|
| `noeviction` | Queues/Streams | Return errors when memory limit is reached |
| `allkeys-lru` | General cache | Evict any key using approximated LRU |
| `allkeys-lfu` | Hot/cold cache | Evict any key using approximated LFU |
| `volatile-ttl` | Sessions | Evict keys with TTL, nearest expiration first |
| `volatile-lru` | Mixed workload | Evict keys with TTL using approximated LRU |
| `volatile-lfu` | Mixed workload | Evict keys with TTL using approximated LFU |

```conf
# Caching workload
maxmemory-policy allkeys-lru
maxmemory-samples 10  # Higher = more accurate but slower

# Session storage
maxmemory-policy volatile-ttl

# Message queue (never lose messages)
maxmemory-policy noeviction
```

### Memory Optimization Techniques

#### 1. Use Appropriate Data Structures

```python
# Bad: Storing JSON as strings
SET user:1000 '{"name":"John","email":"john@example.com","age":30}'

# Good: Using hashes (more memory efficient for small objects)
HSET user:1000 name "John" email "john@example.com" age 30
```

#### 2. Enable Compression for Lists

```conf
# Compress list nodes beyond depth 1
list-compress-depth 1
```

#### 3. Use Ziplist Encoding

```conf
# Optimize hash encoding for small hashes
hash-max-ziplist-entries 512
hash-max-ziplist-value 64

# Optimize sorted set encoding
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
```

#### 4. Active Defragmentation

```conf
# Enable active defragmentation
activedefrag yes
active-defrag-ignore-bytes 100mb
active-defrag-threshold-lower 10
active-defrag-threshold-upper 100
active-defrag-cycle-min 1
active-defrag-cycle-max 25
```

---

## Persistence Tradeoffs

### Overview

| Persistence | Durability | Performance | Recovery Speed |
|-------------|------------|-------------|----------------|
| None | None | Fastest | N/A |
| RDB only | Periodic | Fast | Fast |
| AOF only | Per-write | Slower | Slower |
| RDB + AOF | Best | Slowest | Fast |

### RDB (Snapshotting)

Best for: Backups, disaster recovery, fast restarts

```conf
# Snapshot configuration
save 900 1      # Save after 900s if at least 1 key changed
save 300 10     # Save after 300s if at least 10 keys changed
save 60 10000   # Save after 60s if at least 10000 keys changed

# RDB options
rdbcompression yes
rdbchecksum yes
stop-writes-on-bgsave-error yes
```

**Pros:**
- Compact single-file backup
- Fast restart times
- Good for disaster recovery
- Minimal performance impact during normal operation

**Cons:**
- Data loss between snapshots possible
- Fork operation can cause latency spikes
- Memory usage doubles during save (copy-on-write)

### AOF (Append-Only File)

Best for: Queues, streams, high-durability requirements

```conf
# AOF configuration
appendonly yes
appendfsync everysec  # Options: always, everysec, no

# AOF rewrite triggers
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Use RDB preamble for faster recovery
aof-use-rdb-preamble yes
```

**fsync Options:**

| Option | Durability | Performance |
|--------|------------|-------------|
| `always` | Every write | Slowest (100-1000 ops/sec) |
| `everysec` | 1 second max loss | Balanced (~100K ops/sec) |
| `no` | OS dependent | Fastest (~200K ops/sec) |

**Pros:**
- More durable than RDB
- Readable/editable format
- Can tune durability vs performance

**Cons:**
- Larger file sizes
- Slower recovery times
- Rewrite process uses CPU

### Hybrid Approach (Recommended for Production)

```conf
# Enable both RDB and AOF
save 900 1
save 300 10
save 60 10000

appendonly yes
appendfsync everysec
aof-use-rdb-preamble yes
```

### No Persistence (Caching Only)

```conf
# Disable all persistence
save ""
appendonly no
```

**Use when:**
- Data can be regenerated from source
- Pure caching workload
- Maximum performance required

---

## Connection Pooling Best Practices

### Why Connection Pooling?

- **Reduce latency**: Avoid TCP handshake overhead
- **Manage resources**: Control connection count
- **Improve throughput**: Reuse established connections

### Connection Pool Sizing

```
Optimal Pool Size = (Number of cores * 2) + Number of disks

# For network-bound workloads
Pool Size = Number of concurrent operations expected
```

#### Python (redis-py) Example

```python
import redis
from redis.connection import ConnectionPool

# Create connection pool
pool = ConnectionPool(
    host='redis-production-client.greenlang',
    port=6379,
    password='your_password',
    max_connections=50,          # Maximum connections
    socket_timeout=5,            # Read/write timeout
    socket_connect_timeout=5,    # Connection timeout
    retry_on_timeout=True,
    health_check_interval=30,    # Connection health check
    decode_responses=True
)

# Use the pool
client = redis.Redis(connection_pool=pool)
```

#### Node.js (ioredis) Example

```javascript
const Redis = require('ioredis');

const redis = new Redis.Cluster([
  { host: 'redis-production-client.greenlang', port: 6379 }
], {
  redisOptions: {
    password: 'your_password',
    connectTimeout: 5000,
    commandTimeout: 5000,
    maxRetriesPerRequest: 3
  },
  scaleReads: 'slave',           // Read from replicas
  enableReadyCheck: true,
  natMap: {},
  clusterRetryStrategy: (times) => Math.min(times * 100, 3000)
});

// Connection pool settings
redis.options.maxRetriesPerRequest = 3;
```

#### Go (go-redis) Example

```go
import (
    "github.com/redis/go-redis/v9"
    "time"
)

client := redis.NewClient(&redis.Options{
    Addr:         "redis-production-client.greenlang:6379",
    Password:     "your_password",
    DB:           0,
    PoolSize:     100,          // Maximum connections
    MinIdleConns: 10,           // Minimum idle connections
    MaxRetries:   3,
    DialTimeout:  5 * time.Second,
    ReadTimeout:  3 * time.Second,
    WriteTimeout: 3 * time.Second,
    PoolTimeout:  4 * time.Second,
    IdleTimeout:  5 * time.Minute,
})
```

### Redis Server Settings

```conf
# Maximum clients (connection limit)
maxclients 10000

# TCP keepalive (detect dead connections)
tcp-keepalive 300

# Client timeout (0 = disabled)
timeout 0

# TCP backlog (pending connections queue)
tcp-backlog 511
```

### Connection Pool Monitoring

Monitor these metrics:

- `redis_connected_clients` - Current connection count
- `redis_blocked_clients` - Blocked by BLPOP, etc.
- `redis_rejected_connections_total` - Connections rejected (maxclients hit)

---

## Pipeline Optimization

### What is Pipelining?

Pipelining allows sending multiple commands without waiting for individual responses, reducing network round-trips.

```
Without Pipeline: Request1 -> Response1 -> Request2 -> Response2 -> ...
With Pipeline:    Request1, Request2, ... -> Response1, Response2, ...
```

### Performance Impact

| Batch Size | Without Pipeline | With Pipeline | Improvement |
|------------|------------------|---------------|-------------|
| 1 | 1 RTT | 1 RTT | 0% |
| 10 | 10 RTT | 1 RTT | 90% |
| 100 | 100 RTT | 1 RTT | 99% |
| 1000 | 1000 RTT | 1 RTT | 99.9% |

### Python Pipelining Example

```python
import redis

client = redis.Redis(host='redis-cache', port=6379)

# Without pipeline (slow)
for i in range(1000):
    client.set(f'key:{i}', f'value:{i}')

# With pipeline (fast)
pipe = client.pipeline()
for i in range(1000):
    pipe.set(f'key:{i}', f'value:{i}')
results = pipe.execute()  # Single network round-trip
```

### Best Practices

#### 1. Optimal Batch Size

```python
# Too small = not enough benefit
# Too large = memory pressure, timeout risk

OPTIMAL_BATCH_SIZE = 100  # Start here and adjust based on monitoring

def batch_operations(client, operations, batch_size=100):
    results = []
    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        pipe = client.pipeline()
        for op in batch:
            op(pipe)
        results.extend(pipe.execute())
    return results
```

#### 2. Transaction Safety with MULTI/EXEC

```python
# Atomic transactions with pipeline
pipe = client.pipeline(transaction=True)
pipe.multi()
pipe.incr('counter')
pipe.expire('counter', 3600)
pipe.execute()
```

#### 3. Error Handling

```python
pipe = client.pipeline()
for i in range(100):
    pipe.set(f'key:{i}', f'value:{i}')

try:
    results = pipe.execute(raise_on_error=False)
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Command {i} failed: {result}")
except redis.RedisError as e:
    print(f"Pipeline failed: {e}")
```

### Lua Scripts for Complex Operations

For operations requiring atomicity:

```lua
-- increment_and_get.lua
local current = redis.call('GET', KEYS[1])
if current == false then
    current = 0
end
local new_value = tonumber(current) + tonumber(ARGV[1])
redis.call('SET', KEYS[1], new_value)
return new_value
```

```python
# Load and execute Lua script
script = client.register_script("""
local current = redis.call('GET', KEYS[1])
if current == false then current = 0 end
local new_value = tonumber(current) + tonumber(ARGV[1])
redis.call('SET', KEYS[1], new_value)
return new_value
""")

result = script(keys=['counter'], args=[5])
```

---

## Cluster Mode Considerations

### When to Use Cluster Mode

| Scenario | Recommendation |
|----------|----------------|
| < 16GB data | Single instance + Sentinel |
| 16GB - 100GB data | Consider cluster |
| > 100GB data | Use cluster |
| High availability critical | Cluster or Sentinel |
| Simple operations | Sentinel sufficient |
| Cross-slot operations needed | Avoid cluster |

### Cluster Architecture

```
+------------------+     +------------------+     +------------------+
|  Master 1        |     |  Master 2        |     |  Master 3        |
|  Slots: 0-5460   |     |  Slots: 5461-10922|    |  Slots: 10923-16383|
+------------------+     +------------------+     +------------------+
        |                        |                        |
+------------------+     +------------------+     +------------------+
|  Replica 1a      |     |  Replica 2a      |     |  Replica 3a      |
+------------------+     +------------------+     +------------------+
```

### Cluster Configuration

```conf
# Enable cluster mode
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 15000

# Don't require full coverage (allow partial failures)
cluster-require-full-coverage no

# Replica validity factor
cluster-replica-validity-factor 10

# Migration barrier
cluster-migration-barrier 1
```

### Hash Tags for Multi-Key Operations

```python
# Without hash tags - keys may be on different nodes (ERROR)
client.mget('user:1:name', 'user:1:email')  # May fail in cluster

# With hash tags - keys guaranteed on same node
client.mget('{user:1}:name', '{user:1}:email')  # Works in cluster
```

### Cluster-Aware Client Configuration

```python
from redis.cluster import RedisCluster

client = RedisCluster(
    host='redis-cluster',
    port=6379,
    password='your_password',
    skip_full_coverage_check=True,
    read_from_replicas=True  # Scale reads
)
```

### Cluster Resharding

```bash
# Add a new node
redis-cli --cluster add-node new_node:6379 existing_node:6379

# Reshard slots
redis-cli --cluster reshard existing_node:6379

# Remove a node
redis-cli --cluster del-node existing_node:6379 node_id
```

---

## Workload-Specific Configurations

### Caching Workload

**Characteristics:**
- High read throughput
- Data can be regenerated
- TTL on all keys
- Acceptable to lose data

```conf
# Memory
maxmemory 6gb
maxmemory-policy allkeys-lru

# No persistence (speed priority)
save ""
appendonly no

# Threading
io-threads 4
io-threads-do-reads yes

# Client limits
maxclients 10000
```

### Session Storage

**Characteristics:**
- Moderate read/write
- All keys have TTL
- Some durability needed
- Quick expiration important

```conf
# Memory
maxmemory 2gb
maxmemory-policy volatile-ttl

# RDB persistence (periodic snapshots)
save 900 1
save 300 10
save 60 10000
appendonly no

# Enable keyspace notifications for session expiry
notify-keyspace-events Ex

# Faster expiration processing
active-expire-effort 3
```

### Message Queue / Streams

**Characteristics:**
- Write-heavy
- Must never lose messages
- Order matters
- May accumulate backlog

```conf
# Memory
maxmemory 4gb
maxmemory-policy noeviction  # Never lose messages!

# AOF persistence (durability priority)
appendonly yes
appendfsync everysec
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Stream settings
stream-node-max-bytes 4096
stream-node-max-entries 100

# Require replicas for write confirmation
min-replicas-to-write 1
min-replicas-max-lag 10
```

### Real-Time Analytics

**Characteristics:**
- High write throughput
- Time-series data
- Aggregations needed
- Some data loss acceptable

```conf
# Memory
maxmemory 8gb
maxmemory-policy volatile-lfu

# Hybrid persistence
save 300 1
appendonly yes
appendfsync everysec

# Optimize for sorted sets (time-series)
zset-max-ziplist-entries 256
zset-max-ziplist-value 64

# Higher throughput
io-threads 8
io-threads-do-reads yes
```

---

## Monitoring and Alerting

### Key Metrics to Monitor

#### Memory Metrics

| Metric | Warning | Critical |
|--------|---------|----------|
| `used_memory_rss` | > 80% maxmemory | > 90% maxmemory |
| `mem_fragmentation_ratio` | > 1.5 | > 2.0 |
| `evicted_keys` | > 100/min | > 1000/min |

#### Performance Metrics

| Metric | Warning | Critical |
|--------|---------|----------|
| `instantaneous_ops_per_sec` | Baseline deviation > 30% | > 50% |
| `latency_percentile_99` | > 10ms | > 50ms |
| `blocked_clients` | > 10 | > 50 |

#### Replication Metrics

| Metric | Warning | Critical |
|--------|---------|----------|
| `master_link_status` | - | down |
| `master_repl_offset` lag | > 1MB | > 10MB |
| `connected_slaves` | < expected | 0 |

### Prometheus Alerting Rules

```yaml
groups:
  - name: redis-alerts
    rules:
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage high"

      - alert: RedisReplicationBroken
        expr: redis_connected_slaves < 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis replication broken"

      - alert: RedisHighLatency
        expr: redis_commands_duration_seconds_total / redis_commands_total > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis high command latency"
```

### Grafana Dashboard Panels

```json
{
  "panels": [
    {
      "title": "Memory Usage",
      "type": "gauge",
      "targets": [{
        "expr": "redis_memory_used_bytes / redis_memory_max_bytes * 100"
      }]
    },
    {
      "title": "Operations/sec",
      "type": "graph",
      "targets": [{
        "expr": "rate(redis_commands_total[1m])"
      }]
    },
    {
      "title": "Connected Clients",
      "type": "stat",
      "targets": [{
        "expr": "redis_connected_clients"
      }]
    }
  ]
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

**Symptoms:**
- `used_memory` approaching `maxmemory`
- Evictions increasing
- OOM kills

**Solutions:**
```bash
# Check memory usage by type
redis-cli INFO memory

# Find large keys
redis-cli --bigkeys

# Analyze memory
redis-cli MEMORY DOCTOR
```

#### 2. High Latency

**Symptoms:**
- Slow response times
- Client timeouts
- High CPU usage

**Solutions:**
```bash
# Check slow queries
redis-cli SLOWLOG GET 10

# Enable latency monitoring
redis-cli CONFIG SET latency-monitor-threshold 50
redis-cli LATENCY HISTORY command
redis-cli LATENCY DOCTOR
```

#### 3. Connection Issues

**Symptoms:**
- `rejected_connections` increasing
- Client connection errors
- `maxclients` reached

**Solutions:**
```bash
# Check current connections
redis-cli INFO clients

# Find idle connections
redis-cli CLIENT LIST

# Kill idle connections
redis-cli CLIENT KILL TYPE normal

# Increase maxclients
redis-cli CONFIG SET maxclients 20000
```

#### 4. Replication Lag

**Symptoms:**
- `master_repl_offset` increasing
- Stale reads from replicas
- `master_link_status` = down

**Solutions:**
```bash
# Check replication status
redis-cli INFO replication

# Check replication backlog
redis-cli INFO stats | grep sync

# Force full resync if needed
redis-cli REPLICAOF NO ONE
redis-cli REPLICAOF master_host master_port
```

### Debug Commands

```bash
# Overall server info
redis-cli INFO

# Memory analysis
redis-cli MEMORY STATS
redis-cli MEMORY DOCTOR

# Latency analysis
redis-cli LATENCY DOCTOR
redis-cli DEBUG SLEEP 0.1  # Test latency

# Keyspace analysis
redis-cli DBSIZE
redis-cli SCAN 0 COUNT 100

# Client analysis
redis-cli CLIENT LIST
redis-cli CLIENT INFO
```

### Performance Testing

```bash
# Built-in benchmark
redis-benchmark -h localhost -p 6379 -c 100 -n 100000

# Pipeline benchmark
redis-benchmark -h localhost -p 6379 -P 16 -c 100 -n 100000

# Specific command benchmark
redis-benchmark -t set,get -n 100000 -q
```

---

## Quick Reference

### Memory Sizing Calculator

```
Total Memory Needed =
  (Average Key Size + Average Value Size + 64)
  * Number of Keys
  * 1.5 (safety margin)
  + Replication Buffer (128MB recommended)
  + Client Output Buffers (varies)
```

### Configuration Checklist

- [ ] Set appropriate `maxmemory` (leave 20-30% headroom)
- [ ] Choose correct `maxmemory-policy` for workload
- [ ] Configure persistence based on durability needs
- [ ] Set `tcp-keepalive` for connection health
- [ ] Enable `activedefrag` if using Redis 4+
- [ ] Configure `io-threads` for multi-core systems
- [ ] Set up monitoring and alerting
- [ ] Document all custom configurations

### Workload Quick Settings

| Setting | Caching | Sessions | Queuing | Production |
|---------|---------|----------|---------|------------|
| maxmemory-policy | allkeys-lru | volatile-ttl | noeviction | volatile-lru |
| Persistence | None | RDB | AOF | RDB+AOF |
| maxmemory | 6gb | 2gb | 4gb | 8gb |
| io-threads | 4 | 4 | 4 | 4 |

---

## Additional Resources

- [Redis Documentation](https://redis.io/documentation)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)
- [Redis Administration](https://redis.io/docs/manual/admin/)
- [GreenLang Redis ConfigMaps](../kubernetes/database/redis/)
- [GreenLang Redis Configurations](../database/redis/)
