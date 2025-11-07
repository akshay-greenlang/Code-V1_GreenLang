# GreenLang Operations Performance Tuning Guide

**Version:** 1.0
**Last Updated:** 2025-11-07
**Document Classification:** Operations Reference
**Review Cycle:** Quarterly

---

## Executive Summary

This guide provides production-focused performance tuning procedures for the GreenLang platform. It covers system resource optimization, application tuning, infrastructure optimization, and capacity planning based on actual production benchmarks.

**Current Performance Baseline:**
- p50 Latency: 85ms
- p95 Latency: 208ms
- p99 Latency: 324ms
- Throughput: 222 RPS
- Error Rate: 0.0%
- CPU Usage: ~45% average
- Memory Usage: ~60% average

**Performance Targets:**
- p95 Latency: <500ms
- Throughput: >200 RPS
- Error Rate: <1%
- CPU Usage: <70% average
- Memory Usage: <75% average
- Uptime: 99.9%

---

## Table of Contents

1. [System Resource Optimization](#system-resource-optimization)
2. [Application Tuning](#application-tuning)
3. [Infrastructure Tuning](#infrastructure-tuning)
4. [Database Optimization](#database-optimization)
5. [Monitoring and Profiling](#monitoring-and-profiling)
6. [Capacity Planning](#capacity-planning)

---

## System Resource Optimization

### CPU Optimization

#### Current State Assessment
```bash
# Check CPU usage across pods
kubectl top pods -A --sort-by=cpu

# Check CPU usage over time
# Grafana: https://grafana.greenlang.io/d/cpu-usage

# Identify CPU-intensive processes
kubectl exec -it deploy/greenlang-api -- top -b -n 1 | head -20
```

#### Optimization Strategies

**1. Right-Size CPU Allocation**
```yaml
# Adjust CPU requests and limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-api
spec:
  template:
    spec:
      containers:
      - name: api
        resources:
          requests:
            cpu: "500m"      # Minimum needed
          limits:
            cpu: "2000m"     # Maximum allowed
```

**Guidelines:**
- Requests: Set to average CPU usage
- Limits: Set to 2x average (allows for bursts)
- Monitor throttling: `kubectl top pods`

**2. Enable CPU Affinity**
```yaml
# Pin pods to specific CPU cores
apiVersion: v1
kind: Pod
metadata:
  name: greenlang-api
spec:
  containers:
  - name: api
    resources:
      requests:
        cpu: "2"
      limits:
        cpu: "2"
  # This gives exclusive CPU access
```

**3. Optimize Python GIL**
```python
# For CPU-bound tasks, use ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=4)

# Run CPU-intensive tasks in separate processes
result = await loop.run_in_executor(executor, cpu_intensive_function, data)
```

**Validation:**
```bash
# CPU usage should be:
# - Stable (not spiking)
# - Below 70% average
# - No throttling (check cgroup CPU stats)

kubectl exec -it deploy/greenlang-api -- cat /sys/fs/cgroup/cpu/cpu.stat
```

---

### Memory Optimization

#### Current State Assessment
```bash
# Check memory usage
kubectl top pods -A --sort-by=memory

# Check memory usage over time
# Grafana: https://grafana.greenlang.io/d/memory-usage

# Check for memory leaks
kubectl exec -it deploy/greenlang-api -- python -m memory_profiler
```

#### Optimization Strategies

**1. Right-Size Memory Allocation**
```yaml
# Adjust memory requests and limits
apiVersion: apps/v1
kind: Deployment
metadata:
  name: greenlang-api
spec:
  template:
    spec:
      containers:
      - name: api
        resources:
          requests:
            memory: "1Gi"     # Minimum needed
          limits:
            memory: "2Gi"     # Maximum allowed (OOMKill threshold)
```

**Guidelines:**
- Requests: Set to average memory usage + 20%
- Limits: Set to peak memory usage + 30%
- Monitor OOMKills: `kubectl get pods -w`

**2. Optimize Python Memory Usage**
```python
# Use __slots__ to reduce memory per instance
class Agent:
    __slots__ = ['name', 'config', 'state']

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.state = {}

# Impact: 40-50% memory reduction per instance
```

**3. Implement Cache Size Limits**
```python
from functools import lru_cache

# Limit cache size
@lru_cache(maxsize=1000)
def expensive_operation(input_data):
    return result

# Clear cache periodically
expensive_operation.cache_clear()
```

**4. Enable Garbage Collection Tuning**
```python
import gc

# Tune garbage collection
gc.set_threshold(700, 10, 10)  # More aggressive GC

# Disable GC during critical operations
gc.disable()
# ... critical code ...
gc.enable()
gc.collect()
```

**Validation:**
```bash
# Memory usage should be:
# - Stable (not growing over time)
# - Below 75% of limit
# - No OOMKills

kubectl get events | grep OOMKilled
# Should return nothing
```

---

### Disk I/O Optimization

#### Current State Assessment
```bash
# Check disk I/O
kubectl exec -it deploy/greenlang-api -- iostat -x 1 5

# Check disk usage
kubectl exec -it deploy/greenlang-api -- df -h

# Check disk latency
# Grafana: https://grafana.greenlang.io/d/disk-io
```

#### Optimization Strategies

**1. Use SSD-Backed Storage**
```yaml
# StorageClass for SSD volumes
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
```

**2. Optimize Log Writing**
```python
# Buffer log writes
import logging
from logging.handlers import MemoryHandler

# Buffer 100 records before flush
memory_handler = MemoryHandler(
    capacity=100,
    flushLevel=logging.ERROR,
    target=file_handler
)
```

**3. Use Async I/O**
```python
import aiofiles

# Async file operations
async def write_log(message):
    async with aiofiles.open('log.txt', mode='a') as f:
        await f.write(message + '\n')
```

**Validation:**
```bash
# Disk I/O should be:
# - Write latency < 10ms
# - Read latency < 5ms
# - No I/O wait spikes
```

---

## Application Tuning

### Agent Concurrency Settings

**Current Benchmark:**
- 10 concurrent agents: 5.2x speedup
- 50 concurrent agents: 8.6x speedup
- 100 concurrent agents: 10.1x speedup

#### Optimal Concurrency Configuration

```python
# AsyncOrchestrator configuration
orchestrator_config = {
    "max_concurrent": 100,           # Maximum concurrent agents
    "batch_size": 10,                # Batch size for processing
    "timeout_seconds": 30.0,         # Per-agent timeout
    "retry_attempts": 3,             # Retry failed agents
    "circuit_breaker_threshold": 5,  # Failures before circuit opens
}

orchestrator = AsyncOrchestrator(orchestrator_config)
```

**Tuning Guidelines:**
- Start with `max_concurrent = 10`
- Increase by 10 until diminishing returns
- Monitor CPU/memory usage
- Optimal range: 50-100 for most workloads

**Validation:**
```bash
# Run load test with different concurrency levels
for concurrency in 10 20 50 100; do
    echo "Testing concurrency: $concurrency"
    locust -f tests/performance/locustfile.py \
        --users=$concurrency --spawn-rate=10 --run-time=5m \
        --html=report-$concurrency.html
done

# Compare p95 latency and throughput
```

---

### AsyncOrchestrator Optimization

**Current Performance:**
- 100 concurrent agents: 208ms p95 latency
- Throughput: 222 RPS
- Error rate: 0%

#### Optimization 1: Batching Strategy

```python
# Optimize batch processing
async def process_batch_optimized(agents, batch_size=20):
    results = []

    for i in range(0, len(agents), batch_size):
        batch = agents[i:i + batch_size]

        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[agent.execute() for agent in batch],
            return_exceptions=True
        )

        results.extend(batch_results)

    return results
```

**Impact:** 15-20% latency reduction

#### Optimization 2: Connection Pooling

```python
from aiohttp import ClientSession, TCPConnector

# Shared connection pool
connector = TCPConnector(
    limit=200,              # Total connections
    limit_per_host=50,      # Per-host limit
    ttl_dns_cache=300,      # DNS cache TTL
    keepalive_timeout=300   # Keep connections alive
)

session = ClientSession(connector=connector)
```

**Impact:** 30-40% reduction in connection overhead

#### Optimization 3: Circuit Breaker

```python
from circuitbreaker import circuit

class ResilientOrchestrator:
    @circuit(failure_threshold=5, recovery_timeout=60)
    async def execute_agent(self, agent):
        return await agent.execute()
```

**Impact:** Faster failure detection, prevents cascading failures

**Validation:**
```bash
# Performance should improve to:
# - p95 latency: <180ms
# - Throughput: >250 RPS
# - Error rate: <0.1%
```

---

### Cache Configuration

#### Redis Cache Optimization

```yaml
# Redis configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    # Memory
    maxmemory 2gb
    maxmemory-policy allkeys-lru

    # Performance
    save ""                    # Disable persistence for speed
    appendonly no

    # Connections
    maxclients 10000
    timeout 300

    # Pipelining
    tcp-backlog 511
```

**Application Cache Configuration:**
```python
from greenlang.cache import AsyncCache

cache = AsyncCache(
    redis_url="redis://redis:6379",
    ttl=300,                    # 5 minute TTL
    max_connections=50,
    socket_keepalive=True,
    decode_responses=False      # Store bytes for speed
)

# Cache hot paths
@cache.cached(ttl=600)  # 10 minute cache for agent packs
async def get_agent_pack(agent_name):
    return await load_agent_pack(agent_name)
```

**Cache Hit Rate Targets:**
- Agent packs: >95%
- Configuration: >90%
- LLM responses (where applicable): >60%

**Validation:**
```bash
# Check cache hit rate
redis-cli INFO stats | grep keyspace_hits
redis-cli INFO stats | grep keyspace_misses

# Calculate hit rate
# hit_rate = hits / (hits + misses)
# Target: >80%
```

---

### LLM API Rate Limiting

#### Optimize LLM Usage

```python
from greenlang.llm import LLMRateLimiter

rate_limiter = LLMRateLimiter(
    max_requests_per_minute=60,
    max_tokens_per_minute=90000,
    backoff_factor=1.5
)

# Apply rate limiting
async def call_llm_with_rate_limit(prompt):
    async with rate_limiter:
        return await llm_client.complete(prompt)
```

**Cost Optimization:**
```python
# Use cheaper models for simple tasks
model_router = {
    "simple": "gpt-3.5-turbo",      # $0.002 per 1K tokens
    "complex": "gpt-4",              # $0.03 per 1K tokens
    "reasoning": "o1-preview",       # $0.15 per 1K tokens
}

# Route based on complexity
def select_model(task_complexity):
    return model_router.get(task_complexity, "simple")
```

**Validation:**
```bash
# Monitor LLM API usage
# Dashboard: https://grafana.greenlang.io/d/llm-usage

# Cost should be:
# - Within budget
# - Trending down with optimization
# - No rate limit errors
```

---

## Infrastructure Tuning

### Node Configuration

#### EC2 Instance Sizing

**Current Setup:**
- Instance type: t3.xlarge (4 vCPU, 16GB RAM)
- Node count: 3
- Total capacity: 12 vCPU, 48GB RAM

**Optimization Recommendations:**

**For CPU-Intensive Workloads:**
```bash
# Switch to compute-optimized instances
# c5.2xlarge: 8 vCPU, 16GB RAM
# Better CPU performance, same cost

aws ec2 modify-instance-attribute \
    --instance-id i-xxxxx \
    --instance-type c5.2xlarge
```

**For Memory-Intensive Workloads:**
```bash
# Switch to memory-optimized instances
# r5.xlarge: 4 vCPU, 32GB RAM
# Double memory for same vCPU

aws ec2 modify-instance-attribute \
    --instance-id i-xxxxx \
    --instance-type r5.xlarge
```

**Auto-Scaling Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: greenlang-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: greenlang-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 75
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

---

### Network Optimization

#### Load Balancer Tuning

```yaml
# Application Load Balancer configuration
apiVersion: v1
kind: Service
metadata:
  name: greenlang-api
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "tcp"
    service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout: "300"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 8000
    protocol: TCP
```

#### Connection Keep-Alive

```python
# Enable HTTP keep-alive
from aiohttp import ClientSession, ClientTimeout

timeout = ClientTimeout(total=30, connect=5)

session = ClientSession(
    timeout=timeout,
    connector=TCPConnector(
        keepalive_timeout=300,
        force_close=False,
        enable_cleanup_closed=True
    )
)
```

**Impact:** Reduces connection overhead by 40-60%

---

## Database Optimization

### Connection Pool Tuning

**Current Configuration:**
```python
# Database connection pool
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,              # Connections per pod
    max_overflow=10,           # Additional connections on demand
    pool_timeout=30,           # Wait time for connection
    pool_recycle=3600,         # Recycle connections after 1 hour
    pool_pre_ping=True         # Verify connection before use
)
```

**Optimization:**
```python
# Increase pool size for high-concurrency workloads
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=50,              # Increased from 20
    max_overflow=20,           # Increased from 10
    pool_timeout=10,           # Reduced from 30 (fail faster)
    pool_recycle=1800,         # Recycle more often
    pool_pre_ping=True,
    echo_pool=True             # Log pool events for debugging
)
```

**Validation:**
```bash
# Check connection pool usage
psql -h db.greenlang.io -c "
  SELECT count(*), application_name, state
  FROM pg_stat_activity
  WHERE datname = 'greenlang'
  GROUP BY application_name, state;
"

# Pool should be:
# - Mostly active (not idle)
# - Below max_overflow threshold
# - No connection timeout errors
```

---

### Query Optimization

#### Index Strategy

```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_executions_created_at
  ON executions(created_at DESC);

CREATE INDEX CONCURRENTLY idx_executions_agent_name
  ON executions(agent_name);

CREATE INDEX CONCURRENTLY idx_executions_status
  ON executions(status)
  WHERE status IN ('running', 'pending');

-- Composite index for common query pattern
CREATE INDEX CONCURRENTLY idx_executions_agent_created
  ON executions(agent_name, created_at DESC);

-- Partial index for active records only
CREATE INDEX CONCURRENTLY idx_active_executions
  ON executions(id, created_at)
  WHERE status = 'running';
```

**Validation:**
```sql
-- Check if indexes are being used
EXPLAIN ANALYZE SELECT * FROM executions
  WHERE agent_name = 'calculator'
  ORDER BY created_at DESC
  LIMIT 10;

-- Should show "Index Scan using idx_executions_agent_created"
```

#### Query Tuning

```sql
-- Slow query analysis
SELECT
  query,
  calls,
  total_time,
  mean_time,
  max_time
FROM pg_stat_statements
WHERE mean_time > 100  -- Queries averaging >100ms
ORDER BY mean_time DESC
LIMIT 20;

-- Add EXPLAIN ANALYZE to slow queries
EXPLAIN (ANALYZE, BUFFERS)
SELECT ... ;
```

---

### Database Scaling

**Read Replicas:**
```python
# Route reads to replica
from sqlalchemy import create_engine

# Write to primary
write_engine = create_engine(PRIMARY_DB_URL)

# Read from replica
read_engine = create_engine(REPLICA_DB_URL)

# Use in application
def get_data(id):
    # Read operations use replica
    with read_engine.connect() as conn:
        return conn.execute(f"SELECT * FROM table WHERE id = {id}").fetchone()

def save_data(data):
    # Write operations use primary
    with write_engine.connect() as conn:
        conn.execute("INSERT INTO table VALUES (...)")
        conn.commit()
```

**Validation:**
```bash
# Check replication lag
psql -h replica.greenlang.io -c "
  SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;
"

# Lag should be:
# - < 10 seconds normally
# - < 60 seconds during peak load
```

---

## Monitoring and Profiling

### Key Performance Metrics

**Application Metrics:**
```promql
# Request rate
rate(gl_requests_total[5m])

# p95 latency
histogram_quantile(0.95, rate(gl_api_latency_seconds_bucket[5m]))

# Error rate
rate(gl_errors_total[5m]) / rate(gl_requests_total[5m])

# Throughput
sum(rate(gl_requests_total[5m]))

# Concurrency
gl_active_executions
```

**System Metrics:**
```promql
# CPU usage
100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100

# Disk I/O
rate(node_disk_io_time_seconds_total[5m])

# Network
rate(node_network_receive_bytes_total[5m])
```

### Performance Dashboard

Access: https://grafana.greenlang.io/d/performance-overview

**Panels:**
1. Request Rate (5m avg)
2. Latency Percentiles (p50, p95, p99)
3. Error Rate
4. CPU Usage by Pod
5. Memory Usage by Pod
6. Database Connection Pool
7. Cache Hit Rate
8. LLM API Latency

---

### Profiling Tools

**CPU Profiling:**
```bash
# Profile application CPU usage
kubectl exec -it deploy/greenlang-api -- py-spy top --pid 1

# Generate flame graph
kubectl exec -it deploy/greenlang-api -- py-spy record --pid 1 --duration 60 -o flamegraph.svg
```

**Memory Profiling:**
```bash
# Profile memory usage
kubectl exec -it deploy/greenlang-api -- python -m memory_profiler script.py

# Track memory allocations
kubectl exec -it deploy/greenlang-api -- python -m tracemalloc script.py
```

**Database Profiling:**
```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 100;  -- Log queries >100ms
SELECT pg_reload_conf();

-- Check logs
tail -f /var/log/postgresql/postgresql-14-main.log | grep "duration:"
```

---

## Capacity Planning

### Current Capacity

**Compute:**
- Total vCPU: 12 cores
- Total Memory: 48 GB
- Average CPU utilization: 45%
- Average Memory utilization: 60%
- Available headroom: 55% CPU, 40% memory

**Network:**
- Bandwidth: 10 Gbps
- Average utilization: 15%
- Peak utilization: 35%

**Database:**
- Instance type: db.r5.xlarge (4 vCPU, 32 GB)
- Storage: 500 GB gp3 SSD
- IOPS: 3000
- Throughput: 125 MB/s
- Current usage: 40% CPU, 55% memory

### Growth Projections

**Traffic Growth:**
- Current: 200 RPS average
- 3-month projection: 400 RPS (+100%)
- 6-month projection: 800 RPS (+300%)
- 12-month projection: 1600 RPS (+700%)

**Capacity Requirements:**

**3 Months (+100% traffic):**
- Compute: Add 3 more nodes (total 6)
- Database: Scale to db.r5.2xlarge
- Storage: Expand to 1 TB

**6 Months (+300% traffic):**
- Compute: Add 6 more nodes (total 12)
- Database: Add read replicas (1 primary + 2 replicas)
- Storage: Expand to 2 TB
- Consider database sharding

**12 Months (+700% traffic):**
- Compute: 20-25 nodes
- Database: 1 primary + 4 replicas, consider sharding
- Storage: 5 TB
- Multi-region deployment

### Cost Optimization

**Current Costs (monthly):**
- Compute (3 nodes): $450
- Database: $350
- Storage: $100
- Network: $50
- **Total: $950/month**

**Optimization Opportunities:**
1. Reserved Instances (40% savings): $570/month
2. Spot Instances for non-critical (30% savings): $665/month
3. Right-sized instances (20% savings): $760/month
4. Combined savings: **$450/month (47% reduction)**

**Implementation:**
```bash
# Purchase Reserved Instances
aws ec2 purchase-reserved-instances-offering \
  --reserved-instances-offering-id xxxxx \
  --instance-count 3

# Use Spot Instances for workers
kubectl apply -f k8s/workers-spot.yaml
```

---

## Performance Tuning Checklist

### Pre-Production
- [ ] Load testing completed at 2x expected traffic
- [ ] Stress testing completed at 5x expected traffic
- [ ] Soak testing completed (24 hours at expected load)
- [ ] All performance targets met
- [ ] Auto-scaling configured and tested
- [ ] Circuit breakers tested
- [ ] Cache warming procedures documented

### Production
- [ ] Monitor key metrics daily
- [ ] Review slow query log weekly
- [ ] Analyze performance trends weekly
- [ ] Capacity planning review monthly
- [ ] Load testing quarterly
- [ ] Performance optimization quarterly

### Continuous Optimization
- [ ] Identify top 10 slowest endpoints
- [ ] Profile slowest operations
- [ ] Optimize database queries
- [ ] Implement caching where beneficial
- [ ] Review and tune auto-scaling policies
- [ ] Monitor and reduce LLM API costs

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-07 | Operations Team | Initial comprehensive guide |

**Next Review Date:** 2026-02-07
**Approved By:** [CTO], [Operations Lead], [Engineering Lead]

---

**Performance is a feature - prioritize it!**
