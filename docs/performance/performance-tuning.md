# Performance Tuning Guide

## Overview

This guide provides strategies for optimizing GreenLang agent performance based on profiling results and production experience.

## Quick Wins

### 1. Enable Async Execution

**Before:**
```python
# Sync execution (slow)
for item in items:
    result = agent.run(item)
```

**After:**
```python
# Async execution (8.6x faster)
async with AsyncFuelAgentAI(config) as agent:
    tasks = [agent.run_async(item) for item in items]
    results = await asyncio.gather(*tasks)
```

**Impact:** 5-10x speedup for parallel workloads

### 2. Use Connection Pooling

```python
# Configure connection pool
config = {
    "http_client": {
        "max_connections": 100,
        "max_keepalive_connections": 20
    }
}
```

**Impact:** Reduces connection overhead by 70%

### 3. Enable Caching

```python
from greenlang.cache import AsyncCache

cache = AsyncCache(ttl=300)  # 5 minute TTL

@cache.cached
async def expensive_operation(input_data):
    # Expensive computation
    return result
```

**Impact:** 100x faster for cached results

## Optimization Strategies

### CPU Optimization

#### Bottleneck: JSON Serialization

**Problem:** JSON serialization taking >10% CPU time

**Solution:** Use msgpack or protobuf
```python
import msgpack

# Instead of json.dumps
data_bytes = msgpack.packb(data)
data = msgpack.unpackb(data_bytes)
```

**Impact:** 3-5x faster serialization

#### Bottleneck: String Operations

**Problem:** Excessive string concatenation

**Solution:** Use f-strings or join
```python
# Slow
result = ""
for item in items:
    result += str(item)

# Fast
result = "".join(str(item) for item in items)
```

### Memory Optimization

#### Issue: High Memory Usage

**Strategy 1: Streaming**
```python
# Instead of loading all data
async def process_large_dataset():
    async for chunk in dataset.stream():
        result = await process_chunk(chunk)
        yield result
```

**Strategy 2: Generator Pattern**
```python
def generate_items():
    for item in large_list:
        yield transform(item)
```

**Strategy 3: Clear Caches Periodically**
```python
# Clear cache after processing
cache.clear()
gc.collect()
```

#### Memory Leaks

**Detection:**
```python
import tracemalloc

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

# Run operations
await agent.run_async(input_data)

snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')
```

**Common Causes:**
1. Event handlers not removed
2. Circular references
3. Global caches growing unbounded

**Solutions:**
1. Use weak references
2. Implement cache size limits
3. Clean up resources in `__aexit__`

### I/O Optimization

#### Database Queries

**Problem:** N+1 query problem

**Solution:** Batch queries
```python
# Slow
for item in items:
    result = await db.query(item.id)

# Fast
ids = [item.id for item in items]
results = await db.query_batch(ids)
```

#### Network Requests

**Problem:** Sequential API calls

**Solution:** Concurrent requests
```python
# Slow
results = []
for url in urls:
    result = await http.get(url)
    results.append(result)

# Fast
tasks = [http.get(url) for url in urls]
results = await asyncio.gather(*tasks)
```

### Async Best Practices

#### 1. Avoid Blocking Operations

**Bad:**
```python
async def my_function():
    time.sleep(1)  # Blocks event loop!
    return result
```

**Good:**
```python
async def my_function():
    await asyncio.sleep(1)  # Non-blocking
    return result
```

#### 2. Use Semaphore for Rate Limiting

```python
semaphore = asyncio.Semaphore(10)  # Max 10 concurrent

async def limited_operation():
    async with semaphore:
        return await expensive_operation()
```

#### 3. Set Timeouts

```python
async def safe_operation():
    try:
        result = await asyncio.wait_for(
            operation(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        # Handle timeout
        return default_value
```

## Configuration Tuning

### Agent Configuration

```python
config = {
    # Execution
    "default_timeout": 30.0,  # 30 seconds
    "enable_validation": True,
    "enable_metrics": True,

    # Resource limits
    "max_concurrent_executions": 100,
    "memory_limit_mb": 512,

    # Caching
    "cache_ttl": 300,  # 5 minutes
    "cache_size": 1000,

    # Observability
    "enable_tracing": True,
    "enable_profiling": False,  # Only in dev
}
```

### Event Loop Tuning

```python
# For CPU-bound tasks
import concurrent.futures

executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
result = await loop.run_in_executor(executor, cpu_intensive_task)
```

## Performance Targets

### Service Level Objectives (SLOs)

| Metric | Target | Measurement |
|--------|--------|-------------|
| p95 Latency | < 500ms | Single agent execution |
| p99 Latency | < 1000ms | Single agent execution |
| Error Rate | < 1% | All executions |
| Throughput | > 10 RPS | 10 concurrent agents |
| Memory | < 500MB | Single agent |
| CPU | < 80% | Peak usage |

### Async Speedup Targets

| Concurrency | Target Speedup |
|-------------|----------------|
| 10 agents | 5x |
| 50 agents | 8x |
| 100 agents | 10x |

## Monitoring and Alerts

### Key Metrics to Monitor

```python
from greenlang.monitoring import MetricsCollector

metrics = MetricsCollector()

# Track execution time
with metrics.timer("agent_execution"):
    result = await agent.run_async(input_data)

# Track memory
metrics.gauge("memory_mb", process.memory_info().rss / 1024 / 1024)

# Track throughput
metrics.counter("requests_total")
```

### Alert Thresholds

```yaml
alerts:
  - name: high_latency
    condition: p95_latency_ms > 500
    severity: warning

  - name: error_rate_high
    condition: error_rate > 0.05
    severity: critical

  - name: memory_leak
    condition: memory_growth_mb_per_hour > 100
    severity: warning
```

## Production Optimizations

### 1. Connection Pooling

```python
from aiohttp import ClientSession, TCPConnector

connector = TCPConnector(
    limit=100,  # Total connections
    limit_per_host=30,
    ttl_dns_cache=300
)

session = ClientSession(connector=connector)
```

### 2. Request Batching

```python
class BatchProcessor:
    def __init__(self, batch_size=10, max_wait=1.0):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue = []

    async def process_batch(self):
        # Process batch
        results = await process_items(self.queue)
        self.queue.clear()
        return results
```

### 3. Circuit Breaker

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def call_external_api():
    return await http.get(url)
```

## Troubleshooting Performance Issues

### Issue: High Latency

**Diagnosis:**
1. Profile CPU usage
2. Check database query times
3. Review network latency

**Solutions:**
- Add indexes to database
- Implement caching
- Use CDN for static assets

### Issue: High Memory Usage

**Diagnosis:**
1. Take memory snapshots
2. Check for circular references
3. Review cache sizes

**Solutions:**
- Implement LRU cache
- Clear references explicitly
- Use weakref for callbacks

### Issue: Low Throughput

**Diagnosis:**
1. Check concurrent execution
2. Review blocking operations
3. Analyze queue depths

**Solutions:**
- Increase concurrency
- Convert sync to async
- Add connection pooling

## References

- [Load Testing Guide](load-testing-guide.md)
- [Profiling Guide](profiling-guide.md)
- [Async Programming Best Practices](https://docs.python.org/3/library/asyncio.html)
