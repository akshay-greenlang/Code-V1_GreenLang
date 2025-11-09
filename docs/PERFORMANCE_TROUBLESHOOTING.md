# Performance Troubleshooting Playbook

**Version:** 1.0.0
**Last Updated:** 2025-11-09
**Team:** Performance Engineering

This playbook provides step-by-step troubleshooting procedures for common performance issues in GreenLang infrastructure.

---

## Table of Contents

1. [High LLM Costs](#issue-1-high-llm-costs)
2. [Slow Agent Execution](#issue-2-slow-agent-execution)
3. [Memory Growth](#issue-3-memory-growth)
4. [Database Bottleneck](#issue-4-database-bottleneck)
5. [Low Cache Hit Rate](#issue-5-low-cache-hit-rate)
6. [High API Latency](#issue-6-high-api-latency)
7. [Connection Pool Exhaustion](#issue-7-connection-pool-exhaustion)
8. [Slow Query Performance](#issue-8-slow-query-performance)

---

## Issue 1: High LLM Costs

### Symptoms
- Monthly LLM costs > $1000
- Cost growing faster than usage
- Budget alerts triggering
- High token usage in metrics

### Diagnosis

**Step 1: Check semantic cache hit rate**
```python
from greenlang.intelligence import get_chat_metrics

metrics = get_chat_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']}")
print(f"Cost this month: ${metrics['total_cost']}")
```

**Expected:** Cache hit rate > 30%
**If lower:** Semantic caching not enabled or threshold too strict

**Step 2: Analyze cost by operation**
```bash
python tools/profiling/profile_llm_costs.py --report
```

Review the HTML report for:
- Which operations cost the most
- Model distribution (GPT-4 vs GPT-3.5)
- Token usage patterns

**Step 3: Check prompt lengths**
```python
from greenlang.intelligence import analyze_prompts

analysis = analyze_prompts()
print(f"Avg prompt length: {analysis['avg_tokens']}")
print(f"Max prompt length: {analysis['max_tokens']}")
```

**Expected:** Avg < 500 tokens
**If higher:** Prompts need compression

### Solution

**Quick Fix (5 minutes):**
```python
# Enable semantic caching
from greenlang.intelligence import ChatSession

session = ChatSession(
    cache_strategy="semantic",
    similarity_threshold=0.95  # Adjust based on needs
)
```

**Short-term (1 hour):**
1. **Enable prompt compression:**
   ```python
   from greenlang.intelligence import PromptCompressor

   compressor = PromptCompressor()
   compressed = compressor.compress(original_prompt)
   ```

2. **Switch to GPT-3.5 for simple tasks:**
   ```python
   # Simple extraction
   session = ChatSession(model="gpt-3.5-turbo")

   # Complex reasoning
   session = ChatSession(model="gpt-4")
   ```

3. **Batch similar requests:**
   ```python
   # Instead of 100 individual calls
   # Make 1 batched call
   batch_prompt = "Extract from:\n" + "\n".join(texts)
   ```

**Long-term (1 week):**
1. Implement request deduplication
2. Add operation-level cost tracking
3. Set up budget alerts per operation
4. Review and optimize all prompts
5. Consider fine-tuned models for repeated tasks

### Prevention

- **Monitor cost daily:** Set up Grafana dashboard
- **Budget alerts:** Trigger at 80% of monthly budget
- **Regular reviews:** Weekly prompt optimization reviews
- **Documentation:** Document why each prompt uses specific model

### Expected Impact
- **Immediate:** 30-50% cost reduction from caching
- **1 week:** 50-70% cost reduction from all optimizations
- **ROI:** $500-$700/month savings per $1000 baseline

---

## Issue 2: Slow Agent Execution

### Symptoms
- Agent processing time > 1 second per record
- Batch processing slower than expected
- User complaints about latency
- Timeout errors

### Diagnosis

**Step 1: Profile agent execution**
```bash
python tools/profiling/profile_cpu.py --function "myapp.agents.MyAgent.process" --report
```

Review flame graph for hotspots:
- Database queries
- API calls
- LLM calls
- Data transformations

**Step 2: Check for N+1 queries**
```python
from greenlang.db import get_query_log

log = get_query_log()
print(f"Query count: {len(log)}")
print(f"Unique queries: {len(set(log))}")
```

**Red flag:** Query count >> expected (e.g., 1000 queries for 10 records)

**Step 3: Monitor agent metrics**
```python
from greenlang.monitoring import get_agent_stats

stats = get_agent_stats("MyAgent")
print(stats)
# {
#   "avg_time_ms": 850,
#   "p95_time_ms": 1200,
#   "db_time_ms": 600,    # 70% of time!
#   "llm_time_ms": 150,
#   "cache_hit_rate": 0.2  # Low!
# }
```

### Solution

**Quick Fix (10 minutes):**
```python
# Enable agent-level caching
from greenlang.sdk.base import Agent
from greenlang.cache import cache_method

class MyAgent(Agent):

    @cache_method(ttl=3600)  # Cache results for 1 hour
    async def process(self, data: Dict) -> Dict:
        result = await self._expensive_operation(data)
        return result
```

**Short-term (2 hours):**

1. **Fix N+1 queries:**
   ```python
   # ❌ Bad: N+1 query
   for shipment in shipments:
       company = await db.get_company(shipment.company_id)

   # ✅ Good: Single JOIN query
   shipments_with_companies = await db.query("""
       SELECT s.*, c.name, c.country
       FROM shipments s
       JOIN companies c ON s.company_id = c.id
   """)
   ```

2. **Batch external API calls:**
   ```python
   # ❌ Bad: Individual calls
   for item in items:
       factor = await api.get_factor(item.code)

   # ✅ Good: Batch call
   codes = [item.code for item in items]
   factors = await api.get_factors_batch(codes)
   ```

3. **Parallelize independent operations:**
   ```python
   # ✅ Good: Parallel execution
   factor, company, config = await asyncio.gather(
       get_factor(code),
       get_company(id),
       get_config(type)
   )
   ```

**Long-term (1 week):**
1. Implement batch processing mode
2. Add connection pooling for external APIs
3. Pre-compute common calculations
4. Add indexes to frequently queried tables
5. Implement result streaming for real-time UX

### Prevention

- **Continuous profiling:** Profile on every deploy
- **Performance tests:** Run benchmarks in CI/CD
- **Regression detection:** Alert on 10% slowdown
- **Code reviews:** Check for N+1 queries, missing async

### Expected Impact
- **N+1 fix:** 10-100x faster
- **Caching:** 2-5x faster
- **Parallelization:** 2-3x faster
- **Overall:** 1s → 100ms typical improvement

---

## Issue 3: Memory Growth

### Symptoms
- Memory usage increases over time
- Out of memory errors
- Container/process restarts
- Degraded performance over time

### Diagnosis

**Step 1: Profile memory usage**
```bash
python tools/profiling/profile_memory.py --script main.py --leak-detection --report
```

Review report for:
- Memory growth over time
- Largest allocations
- Leak candidates

**Step 2: Check cache sizes**
```python
from greenlang.cache import get_cache_stats

stats = get_cache_stats()
print(f"L1 size: {stats['l1_size_mb']} MB")
print(f"L1 items: {stats['l1_item_count']}")
```

**Red flag:** Unbounded cache growth

**Step 3: Monitor object counts**
```python
import gc
import sys

# Get object counts by type
obj_counts = {}
for obj in gc.get_objects():
    type_name = type(obj).__name__
    obj_counts[type_name] = obj_counts.get(type_name, 0) + 1

# Top 10 object types
for obj_type, count in sorted(obj_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{obj_type}: {count}")
```

### Solution

**Quick Fix (5 minutes):**
```python
# Set cache size limits
from greenlang.cache import CacheManager

cache = CacheManager(
    max_size_mb=100,  # Limit L1 cache to 100 MB
    max_items=10000,  # Limit to 10K items
    eviction_policy="LRU"  # Evict least recently used
)
```

**Short-term (1 hour):**

1. **Use generators for large datasets:**
   ```python
   # ❌ Bad: Load all into memory
   all_shipments = list(get_all_shipments())

   # ✅ Good: Process in chunks
   for batch in batch_generator(get_all_shipments(), size=1000):
       process_batch(batch)
       # Memory freed after each batch
   ```

2. **Clean up references:**
   ```python
   # ✅ Good: Explicit cleanup
   large_dataset = load_data()
   process(large_dataset)
   del large_dataset  # Free memory
   gc.collect()       # Force garbage collection
   ```

3. **Use weak references for caches:**
   ```python
   import weakref

   cache = weakref.WeakValueDictionary()
   ```

**Long-term (1 week):**
1. Implement proper cache eviction policies
2. Add memory monitoring and alerts
3. Use memory-efficient data structures (numpy, pandas with chunks)
4. Profile and optimize top memory consumers
5. Add memory limits to containers

### Prevention

- **Memory profiling:** Profile before each release
- **Load testing:** Test with production-scale data
- **Monitoring:** Alert on memory growth > 10% per hour
- **Limits:** Set container memory limits (e.g., 2GB)

### Expected Impact
- **Cache limits:** Prevent unbounded growth
- **Generators:** 10-100x memory reduction for large datasets
- **Cleanup:** 20-50% memory reduction

---

## Issue 4: Database Bottleneck

### Symptoms
- Slow database queries (> 100ms)
- Connection pool exhaustion
- High database CPU/memory
- Query timeouts

### Diagnosis

**Step 1: Profile database queries**
```bash
python tools/profiling/profile_db.py --report
```

Review report for:
- Slow queries (> 100ms)
- Missing indexes
- N+1 query patterns

**Step 2: Check connection pool status**
```python
from greenlang.db import get_connection_pool

pool = get_connection_pool()
status = await pool.get_pool_status()

print(f"Active connections: {status['checked_out']}")
print(f"Pool size: {status['pool_size']}")
print(f"Utilization: {status['checked_out'] / status['pool_size'] * 100}%")
```

**Red flag:** Utilization > 90%

**Step 3: Analyze query execution plans**
```sql
EXPLAIN ANALYZE SELECT * FROM shipments WHERE country = 'US' AND year = 2024;
```

Look for:
- Sequential scans (should be index scans)
- High execution time
- High row counts

### Solution

**Quick Fix (10 minutes):**
```sql
-- Add missing indexes
CREATE INDEX idx_shipments_country_year ON shipments(country, year);
```

**Short-term (2 hours):**

1. **Increase connection pool size:**
   ```python
   pool = DatabaseConnectionPool(
       pool_size=50,  # Increased from 20
       max_overflow=20  # Increased from 10
   )
   ```

2. **Optimize slow queries:**
   ```sql
   -- ❌ Bad: SELECT *
   SELECT * FROM shipments WHERE country = 'US';

   -- ✅ Good: SELECT specific columns
   SELECT id, quantity, emissions FROM shipments WHERE country = 'US';
   ```

3. **Use read replicas:**
   ```python
   # Route reads to replica
   async with replica_pool.get_session() as session:
       result = await session.execute(read_query)

   # Route writes to primary
   async with primary_pool.get_session() as session:
       await session.execute(write_query)
   ```

**Long-term (1 week):**
1. Add comprehensive indexing strategy
2. Implement query result caching
3. Set up database monitoring (slow query log)
4. Optimize schema (denormalization where needed)
5. Consider database sharding for scale

### Prevention

- **Index all foreign keys:** Automatic in migrations
- **Monitor slow queries:** Alert on queries > 100ms
- **Query review:** Review query plans in code review
- **Load testing:** Test with production data volume

### Expected Impact
- **Indexes:** 10-1000x faster queries
- **Pool size:** Eliminate connection waits
- **Query optimization:** 2-10x faster
- **Read replicas:** 5x read throughput

---

## Issue 5: Low Cache Hit Rate

### Symptoms
- Cache hit rate < 30%
- High backend load
- Slow response times
- High costs (API, LLM)

### Diagnosis

**Step 1: Check cache statistics**
```python
from greenlang.cache import get_cache_stats

stats = get_cache_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Evictions: {stats['evictions']}")
```

**Step 2: Analyze cache key distribution**
```python
from greenlang.cache import analyze_cache_keys

analysis = analyze_cache_keys()
print(f"Unique keys: {analysis['unique_keys']}")
print(f"Key patterns: {analysis['patterns']}")
```

**Step 3: Check TTL settings**
```python
# Are items expiring too quickly?
print(f"Avg TTL: {stats['avg_ttl_seconds']}")
print(f"Items < 1h TTL: {stats['short_ttl_count']}")
```

### Solution

**Quick Fix (5 minutes):**
```python
# Increase TTL for stable data
@cache.memoize(ttl=86400)  # 24 hours instead of 1 hour
def get_emission_factor(code: str):
    return fetch_factor(code)
```

**Short-term (1 hour):**

1. **Normalize cache keys:**
   ```python
   # ❌ Bad: Non-normalized keys
   key1 = f"factor_{country}_{year}"  # "factor_US_2024"
   key2 = f"factor_{country}_{year}"  # "factor_us_2024" (different!)

   # ✅ Good: Normalized keys
   key = f"factor:{country.upper()}:{year}"
   ```

2. **Enable semantic caching for LLM:**
   ```python
   session = ChatSession(
       cache_strategy="semantic",
       similarity_threshold=0.95
   )
   ```

3. **Pre-warm cache:**
   ```python
   # Load common data into cache on startup
   async def warm_cache():
       common_codes = ["7208", "7209", "7210"]  # Most accessed
       for code in common_codes:
           factor = await get_emission_factor(code)
           cache.set(f"factor:{code}", factor, ttl=86400)
   ```

**Long-term (1 week):**
1. Implement cache analytics to identify patterns
2. Add cache warming for predictable access patterns
3. Use hierarchical caching (L1→L2→L3)
4. Implement cache stampede prevention
5. Review and adjust TTLs based on data change frequency

### Prevention

- **Monitor hit rate:** Alert if < 50%
- **Key normalization:** Enforce in code reviews
- **TTL review:** Quarterly review of TTL settings
- **Testing:** Include cache in integration tests

### Expected Impact
- **Normalized keys:** 20-40% hit rate improvement
- **Semantic caching:** 30-50% hit rate for LLM
- **Optimal TTLs:** 10-20% hit rate improvement
- **Pre-warming:** 40-60% hit rate for common data

---

## Issue 6: High API Latency

### Symptoms
- API response time > 500ms
- Timeouts on external calls
- Slow end-to-end processing

### Diagnosis

**Step 1: Identify slow endpoints**
```python
from greenlang.monitoring import get_api_metrics

metrics = get_api_metrics()
for endpoint, stats in metrics.items():
    if stats['p95_latency_ms'] > 500:
        print(f"{endpoint}: {stats['p95_latency_ms']}ms")
```

**Step 2: Check external API performance**
```python
from greenlang.services import analyze_external_calls

analysis = analyze_external_calls()
print(f"Avg external call time: {analysis['avg_time_ms']}ms")
print(f"Slowest API: {analysis['slowest_api']}")
```

**Step 3: Review timeout settings**
```python
# Are timeouts too aggressive?
config = get_api_config()
print(f"Timeout: {config['timeout']}s")
```

### Solution

**Quick Fix:**
```python
# Increase timeout for slow external APIs
api_client = APIClient(timeout=30)  # Increased from 10s
```

**Short-term:**
1. **Add request caching**
2. **Implement circuit breaker**
3. **Use async/parallel requests**
4. **Add request retries with backoff**

**Long-term:**
1. Negotiate SLAs with external providers
2. Implement request batching
3. Add fallback data sources
4. Build local factor database

### Expected Impact
- Caching: 80-90% latency reduction
- Parallel requests: 50-70% reduction
- Circuit breaker: Prevent cascading failures

---

## Quick Reference

### Emergency Response

| Symptom | First Action | Timeline |
|---------|--------------|----------|
| OOM Error | Restart + add memory limits | 5 minutes |
| DB Connection Exhaustion | Increase pool size | 10 minutes |
| High LLM Costs | Enable semantic cache | 5 minutes |
| Slow Queries | Add indexes | 10 minutes |
| API Timeouts | Increase timeout + cache | 10 minutes |

### Monitoring Commands

```bash
# Quick health check
python tools/health_check.py

# Performance snapshot
python tools/performance_snapshot.py

# Cost analysis
python tools/profiling/profile_llm_costs.py --report

# Memory check
python tools/profiling/profile_memory.py --snapshot

# Database status
python tools/profiling/profile_db.py --summary
```

### Contact

- **Critical Issues:** performance-oncall@greenlang.ai
- **Slack:** #performance-engineering
- **Escalation:** CTO (for production outages)

