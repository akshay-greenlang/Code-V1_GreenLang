# Performance Optimization Implementation Guide
## GL-VCCI Scope 3 Platform

**Version**: 1.0.0
**Date**: 2025-11-09
**Team**: Performance Optimization Team

This guide provides step-by-step instructions for implementing all performance optimizations in the GL-VCCI Scope 3 Platform.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Database Optimizations](#database-optimizations)
3. [Async/Await Conversion](#asyncawait-conversion)
4. [Multi-Level Caching](#multi-level-caching)
5. [Connection Pooling](#connection-pooling)
6. [Batch Processing](#batch-processing)
7. [API Pagination](#api-pagination)
8. [Response Caching](#response-caching)
9. [Performance Monitoring](#performance-monitoring)
10. [Load Testing](#load-testing)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Quick Start

### Prerequisites

```bash
# Install performance optimization dependencies
pip install -r requirements.txt

# Additional dependencies for performance optimizations
pip install \
    redis>=5.0.0 \
    hiredis>=2.3.0 \
    httpx>=0.25.0 \
    locust>=2.15.0 \
    prometheus-client>=0.19.0
```

### 5-Minute Setup

```python
# 1. Import performance modules
from database.optimizations import IndexManager, ALL_INDEXES
from cache.caching_strategy import MultiLevelCache, LRUCache, RedisCache
from monitoring.performance_monitoring import get_performance_monitor

# 2. Create database indexes
engine = get_engine()
index_manager = IndexManager(engine)
await index_manager.create_indexes(ALL_INDEXES)

# 3. Initialize caching
from redis.asyncio import Redis
redis = Redis.from_url("redis://localhost:6379/0")
cache = MultiLevelCache(
    l1_cache=LRUCache(max_size=1000),
    l2_cache=RedisCache(redis)
)

# 4. Start monitoring
monitor = get_performance_monitor()

# You're ready! 🚀
```

---

## Database Optimizations

### Step 1: Create Indexes

```python
from database.optimizations import IndexManager, ALL_INDEXES
from database.connection_pool import DatabaseEngineFactory

# Create async engine
engine = DatabaseEngineFactory.create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/vcci_scope3"
)

# Initialize index manager
index_manager = IndexManager(engine)

# Create all indexes
results = await index_manager.create_indexes(ALL_INDEXES)

# Verify results
for index_name, success in results.items():
    status = "✅" if success else "❌"
    print(f"{status} {index_name}")
```

### Step 2: Analyze Query Performance

```python
from database.optimizations import QueryAnalyzer
from sqlalchemy import select

# Initialize analyzer
analyzer = QueryAnalyzer(slow_query_threshold_ms=1000)

# Analyze a query
query = select(Emission).where(
    Emission.supplier_id == "SUP-001"
).where(
    Emission.transaction_date >= "2024-01-01"
)

analysis = await analyzer.analyze_query(session, query, include_execution=True)

# Review recommendations
if analysis.recommendations:
    print("⚠️  Query Optimization Recommendations:")
    for rec in analysis.recommendations:
        print(f"   - {rec}")

print(f"Query executed in {analysis.execution_time_ms:.2f}ms")
print(f"Uses index: {analysis.uses_index}")
print(f"Full table scan: {analysis.full_table_scan}")
```

### Step 3: Optimize N+1 Queries

```python
from database.optimizations import NPlusOneOptimizer
from sqlalchemy.orm import joinedload

# BEFORE: N+1 problem (1 query + N supplier queries)
suppliers = await session.execute(select(Supplier))
for supplier in suppliers.scalars():
    emissions = await session.execute(
        select(Emission).where(Emission.supplier_id == supplier.id)
    )
    # Process emissions...

# AFTER: Single query with JOIN
query = select(Supplier).options(joinedload(Supplier.emissions))
suppliers = await session.execute(query)

for supplier in suppliers.scalars():
    # Emissions already loaded!
    for emission in supplier.emissions:
        # Process emissions...
```

### Step 4: Optimize Batch Inserts

```python
from database.optimizations import batch_insert_optimized

# Prepare records
records = [
    {
        "supplier_id": "SUP-001",
        "emissions_tco2e": 1234.56,
        "transaction_date": "2024-01-01",
        # ... more fields
    }
    for i in range(10000)
]

# Batch insert (10-100x faster than individual inserts)
inserted_count = await batch_insert_optimized(
    session=session,
    table_name="emissions",
    records=records,
    batch_size=1000
)

print(f"Inserted {inserted_count} records")
```

---

## Async/Await Conversion

### Step 1: Convert Database Operations

```python
from services.async_conversions import AsyncDatabaseOperations

db_ops = AsyncDatabaseOperations(engine)

# Async query execution
async def get_emissions(supplier_id: str):
    async with db_ops.session() as session:
        result = await session.execute(
            select(Emission).where(Emission.supplier_id == supplier_id)
        )
        return result.scalars().all()

# Async bulk insert
async def import_emissions(records: List[Dict]):
    return await db_ops.bulk_insert(
        table_class=Emission,
        records=records,
        batch_size=1000
    )
```

### Step 2: Convert HTTP API Calls

```python
from services.async_conversions import AsyncHTTPClient

# Initialize client with connection pooling
async with AsyncHTTPClient(
    base_url="https://api.ecoinvent.org",
    timeout=30.0,
    max_retries=3,
    max_connections=100
) as client:
    # Single request
    factor = await client.get("/factors/electricity", params={"region": "US"})

    # Concurrent requests
    factors = await asyncio.gather(
        client.get("/factors/electricity"),
        client.get("/factors/natural_gas"),
        client.get("/factors/diesel")
    )
```

### Step 3: Convert Redis Operations

```python
from services.async_conversions import AsyncRedisOperations
from redis.asyncio import Redis

redis = Redis.from_url("redis://localhost:6379/0")
redis_ops = AsyncRedisOperations(redis)

# Async get/set
value = await redis_ops.get("emission_factor:electricity:US")
await redis_ops.set("emission_factor:electricity:US", factor_data, ttl=3600)

# Async multi-get
values = await redis_ops.mget(
    "factor:electricity:US",
    "factor:natural_gas:US",
    "factor:diesel:US"
)

# Async pipeline for batch operations
async with redis_ops.pipeline() as pipe:
    await pipe.set("key1", "value1")
    await pipe.set("key2", "value2")
    await pipe.set("key3", "value3")
    results = await pipe.execute()
```

### Step 4: Convert Factor Broker

```python
from services.async_conversions import AsyncFactorBroker

broker = AsyncFactorBroker(
    http_client=http_client,
    cache=redis_ops,
    cache_ttl=3600
)

# Single factor lookup (with automatic caching)
factor = await broker.get_factor(
    source="ecoinvent",
    activity="electricity_grid_mix",
    region="US"
)

# Batch factor lookup (concurrent)
requests = [
    {"source": "ecoinvent", "activity": "electricity", "region": "US"},
    {"source": "ecoinvent", "activity": "natural_gas", "region": "US"},
    {"source": "desnz", "activity": "diesel", "region": "UK"}
]

factors = await broker.get_factors_batch(requests)
```

---

## Multi-Level Caching

### Step 1: Initialize Cache Layers

```python
from cache.caching_strategy import (
    LRUCache,
    RedisCache,
    MultiLevelCache,
    CACHE_CONFIG
)
from redis.asyncio import Redis

# Initialize Redis
redis = Redis.from_url("redis://localhost:6379/0", decode_responses=False)

# Initialize cache layers
l1_cache = LRUCache(max_size=1000)  # In-memory, ~1ms
l2_cache = RedisCache(redis)         # Distributed, ~5-10ms

# Create multi-level cache
cache = MultiLevelCache(l1_cache, l2_cache, CACHE_CONFIG)
```

### Step 2: Use Cache for Emission Factors

```python
# Cache emission factor
async def get_cached_factor(source: str, activity: str, region: str):
    return await cache.get_or_set(
        cache_type="emission_factors",
        key=f"{source}:{activity}:{region}",
        fetcher=lambda: fetch_factor_from_api(source, activity, region),
        source=source,
        activity=activity,
        region=region
    )

# Usage
factor = await get_cached_factor("ecoinvent", "electricity", "US")
```

### Step 3: Cache Entity Resolution Results

```python
# Cache entity resolution
async def resolve_supplier_cached(supplier_name: str):
    import hashlib

    # Hash supplier name for cache key
    name_hash = hashlib.md5(supplier_name.encode()).hexdigest()

    return await cache.get_or_set(
        cache_type="entity_resolution",
        key=name_hash,
        fetcher=lambda: resolve_supplier(supplier_name),
        supplier_name_hash=name_hash
    )
```

### Step 4: Cache LLM Responses

```python
# Cache LLM responses (with compression)
async def get_llm_completion_cached(prompt: str):
    import hashlib

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

    return await cache.get_or_set(
        cache_type="llm_responses",
        key=prompt_hash,
        fetcher=lambda: call_llm_api(prompt),
        prompt_hash=prompt_hash
    )
```

### Step 5: Warm Cache on Startup

```python
from cache.caching_strategy import CacheWarmer

async def startup_event():
    """Warm cache during application startup"""
    warmer = CacheWarmer(cache)

    results = await warmer.warm_all()

    print("Cache warming complete:")
    print(f"  - Emission factors: {results['emission_factors']} cached")
    print(f"  - Suppliers: {results['suppliers']} cached")

# Add to FastAPI
@app.on_event("startup")
async def on_startup():
    await startup_event()
```

### Step 6: Monitor Cache Performance

```python
# Get cache statistics
stats = cache.get_stats()

print("Cache Performance:")
print(f"  L1 Hit Rate: {stats['l1']['hit_rate']:.2f}%")
print(f"  L2 Hit Rate: {stats['l2']['hit_rate']:.2f}%")
print(f"  Combined Hit Rate: {stats['combined']['hit_rate']:.2f}%")
print(f"  Total Requests: {stats['combined']['total_requests']}")
```

---

## Connection Pooling

### Step 1: Configure Production Pool

```python
from database.connection_pool import (
    DatabaseEngineFactory,
    PRODUCTION_POOL_CONFIG,
    OptimizedSessionFactory
)

# Create engine with optimized pool
engine = DatabaseEngineFactory.create_async_engine(
    database_url="postgresql+asyncpg://user:pass@localhost/vcci_scope3",
    pool_config=PRODUCTION_POOL_CONFIG,
    echo=False
)

# Create session factory
session_factory = OptimizedSessionFactory(engine)
```

### Step 2: Use Optimized Sessions

```python
# Use session with automatic transaction management
async def process_emissions(supplier_id: str):
    async with session_factory.session() as session:
        # Query
        result = await session.execute(
            select(Emission).where(Emission.supplier_id == supplier_id)
        )
        emissions = result.scalars().all()

        # Modify
        for emission in emissions:
            emission.processed = True

        # Auto-commit on context exit
        # Auto-rollback on exception

    return emissions
```

### Step 3: Monitor Pool Health

```python
from database.connection_pool import ConnectionPoolMonitor

monitor = ConnectionPoolMonitor(engine)

# Get pool statistics
stats = monitor.get_pool_stats()

print("Connection Pool Stats:")
print(f"  Size: {stats['pool_size']}")
print(f"  Checked Out: {stats['checked_out']}")
print(f"  Overflow: {stats['overflow']}")
print(f"  Utilization: {stats['utilization_pct']:.1f}%")
print(f"  Avg Checkout Time: {stats['avg_checkout_time_ms']:.2f}ms")

# Log stats periodically
import asyncio
from database.connection_pool import monitor_pool_health

asyncio.create_task(monitor_pool_health(monitor, interval_seconds=60))
```

### Step 4: Deploy PgBouncer (Production)

```python
from database.connection_pool import generate_pgbouncer_config

# Generate PgBouncer configuration
config = generate_pgbouncer_config(
    db_host="postgres.example.com",
    db_port=5432,
    db_name="vcci_scope3",
    db_user="vcci_user"
)

# Save to file
with open("/etc/pgbouncer/pgbouncer.ini", "w") as f:
    f.write(config)

# Update application DATABASE_URL to use PgBouncer
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost:6432/vcci_scope3"
```

---

## Batch Processing

### Step 1: Async Batch Processing

```python
from processing.batch_optimizer import (
    AsyncBatchProcessor,
    LARGE_BATCH_CONFIG
)

# Initialize processor
processor = AsyncBatchProcessor(config=LARGE_BATCH_CONFIG)

# Process batch
async def process_emissions_batch(records: List[Dict]):
    results, stats = await processor.process_batch(
        records=records,
        processor=calculate_emissions_async,
        batch_id="BATCH-001"
    )

    print(f"Processed: {stats.processed_records}/{stats.total_records}")
    print(f"Time: {stats.processing_time_seconds:.2f}s")
    print(f"Throughput: {stats.records_per_second:.1f} records/sec")
    print(f"Errors: {stats.failed_records}")

    return results, stats
```

### Step 2: Parallel Batch Processing (CPU-Bound)

```python
from processing.batch_optimizer import ParallelBatchProcessor

# For CPU-intensive tasks
processor = ParallelBatchProcessor(config=LARGE_BATCH_CONFIG)

def cpu_intensive_calculation(batch: List[Dict]) -> List[float]:
    """CPU-bound calculation (synchronous)"""
    return [complex_calculation(record) for record in batch]

# Process with multiprocessing
results, stats = processor.process_batch(
    records=large_dataset,
    processor=cpu_intensive_calculation,
    batch_id="CPU-BATCH-001"
)
```

### Step 3: Streaming Batch Processing

```python
from processing.batch_optimizer import StreamingBatchProcessor

# For very large datasets that don't fit in memory
processor = StreamingBatchProcessor()

async def process_large_file(file_path: str):
    async def data_iterator():
        """Iterator that yields records"""
        async with aiofiles.open(file_path, mode='r') as f:
            async for line in f:
                yield json.loads(line)

    async for batch_result in processor.process_stream(
        data_iterator=data_iterator(),
        processor=process_batch_async,
        batch_size=1000
    ):
        print(f"Processed batch: {len(batch_result)} results")
```

### Step 4: Dynamic Batch Size Optimization

```python
from processing.batch_optimizer import BatchSizeOptimizer

optimizer = BatchSizeOptimizer(
    initial_batch_size=1000,
    min_batch_size=100,
    max_batch_size=10000
)

current_batch_size = 1000

for iteration in range(10):
    # Process batch
    start_time = time.time()
    results = await process_batch(records, batch_size=current_batch_size)
    duration = time.time() - start_time

    records_per_second = len(results) / duration

    # Optimize batch size based on performance
    current_batch_size = optimizer.optimize(current_batch_size, records_per_second)

    print(f"Iteration {iteration}: batch_size={current_batch_size}, "
          f"throughput={records_per_second:.1f} rec/s")
```

---

## API Pagination

### Step 1: Add Cursor-Based Pagination

```python
from api.pagination import (
    CursorPaginator,
    PaginationParams,
    create_paginated_response
)
from fastapi import FastAPI, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

# Initialize paginator
paginator = CursorPaginator(
    cursor_fields=["created_at", "id"],  # Must be unique combination
    default_limit=100,
    max_limit=1000
)

@app.get("/api/v1/emissions")
async def get_emissions(
    cursor: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    db: AsyncSession = Depends(get_db)
):
    # Create pagination params
    params = PaginationParams(
        cursor=cursor,
        limit=limit,
        sort_by=sort_by,
        sort_order=sort_order
    )

    # Base query
    query = select(Emission)

    # Paginate
    result = await paginator.paginate(
        session=db,
        query=query,
        model_class=Emission,
        params=params
    )

    # Create response with links
    return create_paginated_response(
        data=[emission.to_dict() for emission in result.data],
        page_info=result.page_info,
        base_url="/api/v1/emissions",
        params={"cursor": cursor, "limit": limit}
    )
```

### Step 2: Client Usage

```python
import httpx

async def fetch_all_emissions():
    """Fetch all emissions using pagination"""
    all_emissions = []
    cursor = None

    async with httpx.AsyncClient() as client:
        while True:
            # Request page
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor

            response = await client.get(
                "http://localhost:8000/api/v1/emissions",
                params=params
            )
            data = response.json()

            # Add to results
            all_emissions.extend(data["data"])

            # Check for next page
            if not data["page_info"]["has_next_page"]:
                break

            cursor = data["page_info"]["end_cursor"]

    print(f"Fetched {len(all_emissions)} emissions total")
    return all_emissions
```

---

## Response Caching

### Step 1: Add Middleware

```python
from middleware.response_cache import ResponseCacheMiddleware
from redis.asyncio import Redis
from fastapi import FastAPI

app = FastAPI()

# Initialize Redis
redis = Redis.from_url("redis://localhost:6379/1", decode_responses=False)

# Add response cache middleware
app.add_middleware(
    ResponseCacheMiddleware,
    redis_client=redis,
    default_strategy="short",  # 5 minutes default
    cache_prefix="vcci_api_cache"
)
```

### Step 2: Configure Per-Endpoint Caching

```python
from fastapi import Response

@app.get("/api/v1/factors/{category}")
async def get_factor(category: str, response: Response):
    # Set custom cache headers
    response.headers["Cache-Control"] = "public, max-age=3600"  # 1 hour
    response.headers["Vary"] = "Accept, Accept-Language"

    factor = await factor_service.get(category)
    return factor
```

### Step 3: Invalidate Cache on Updates

```python
from middleware.response_cache import CacheInvalidator

invalidator = CacheInvalidator(redis, cache_prefix="vcci_api_cache")

@app.post("/api/v1/emissions")
async def create_emission(emission: EmissionCreate):
    # Create emission
    db_emission = await db.create(emission)

    # Invalidate related cache entries
    await invalidator.invalidate_by_path("/api/v1/emissions")
    await invalidator.invalidate_by_path("/api/v1/reports")

    return db_emission

@app.delete("/api/v1/emissions/{emission_id}")
async def delete_emission(emission_id: str):
    await db.delete(emission_id)

    # Invalidate cache
    await invalidator.invalidate_by_pattern("*emissions*")

    return {"status": "deleted"}
```

---

## Performance Monitoring

### Step 1: Initialize Monitoring

```python
from monitoring.performance_monitoring import get_performance_monitor

# Get global monitor instance
monitor = get_performance_monitor()
```

### Step 2: Track Database Queries

```python
# Track query performance
async def get_emissions_with_tracking(supplier_id: str):
    with monitor.metrics.track_db_query("select_emissions"):
        result = await db.execute(
            select(Emission).where(Emission.supplier_id == supplier_id)
        )
        return result.scalars().all()

# Log slow queries
with monitor.slow_query_logger.log_query(
    query_text="SELECT * FROM emissions WHERE ...",
    duration_seconds=2.5,
    query_type="select_emissions"
):
    pass
```

### Step 3: Track API Requests

```python
from fastapi import Request

@app.get("/api/v1/emissions/{emission_id}")
async def get_emission(emission_id: str, request: Request):
    async with monitor.metrics.track_http_request(
        method=request.method,
        endpoint=request.url.path
    ):
        emission = await db.get(emission_id)
        return emission
```

### Step 4: Track Cache Operations

```python
# Track cache hit/miss
result = await cache.get("key")

if result:
    monitor.metrics.track_cache_operation("l1", "get", "hit")
else:
    monitor.metrics.track_cache_operation("l1", "get", "miss")

# Update cache hit rate
stats = cache.get_stats()
monitor.metrics.update_cache_hit_rate("l1", stats["l1"]["hit_rate"])
```

### Step 5: Export Metrics

```python
from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    content = monitor.export_prometheus_metrics()
    return Response(content=content, media_type=CONTENT_TYPE_LATEST)
```

### Step 6: Get Performance Report

```python
@app.get("/admin/performance/report")
async def performance_report():
    """Get comprehensive performance report"""
    return monitor.get_performance_report()
```

---

## Load Testing

### Run Basic Load Test

```bash
# Start API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# In another terminal, run load test
locust -f tests/performance/load_test.py \
    --host=http://localhost:8000 \
    --users=100 \
    --spawn-rate=10 \
    --run-time=5m \
    --headless \
    --html=reports/load_test.html
```

### Validate Performance Targets

```bash
# Run comprehensive test with performance validation
locust -f tests/performance/load_test.py \
    MixedWorkloadUser \
    --host=http://localhost:8000 \
    --users=500 \
    --spawn-rate=50 \
    --run-time=10m \
    --headless \
    --html=reports/performance_validation.html

# Check results
echo "Performance Targets:"
echo "  ✅ Throughput: >= 5000 req/s"
echo "  ✅ P95 Latency: < 500ms"
echo "  ✅ P99 Latency: < 1000ms"
echo "  ✅ Error Rate: < 0.1%"
echo ""
echo "See report: reports/performance_validation.html"
```

---

## Troubleshooting

### High Database Latency

```python
# Check for slow queries
monitor = get_performance_monitor()
slow_queries = monitor.slow_query_logger.get_slowest_queries(n=10)

for query in slow_queries:
    print(f"Query: {query.query_text[:100]}")
    print(f"Duration: {query.duration_seconds:.2f}s")
    print(f"Type: {query.query_type}")
    print()

# Check for missing indexes
from database.optimizations import IndexManager

index_manager = IndexManager(engine)
missing = await index_manager.find_missing_indexes()

for rec in missing:
    if rec["priority"] in ("CRITICAL", "HIGH"):
        print(f"⚠️  {rec['recommendation']}")
```

### Low Cache Hit Rate

```python
# Check cache statistics
stats = cache.get_stats()

print("Cache Analysis:")
print(f"L1 Hit Rate: {stats['l1']['hit_rate']:.2f}%")
print(f"L2 Hit Rate: {stats['l2']['hit_rate']:.2f}%")

if stats['l1']['hit_rate'] < 70:
    print("⚠️  L1 cache hit rate low - consider:")
    print("   - Increasing L1 cache size")
    print("   - Adjusting cache TTL")
    print("   - Warming cache on startup")

# Check cache evictions
if stats['l1']['evictions'] > stats['l1']['hits']:
    print("⚠️  High eviction rate - increase cache size")
```

### High Connection Pool Utilization

```python
# Check pool stats
from database.connection_pool import ConnectionPoolMonitor

monitor = ConnectionPoolMonitor(engine)
stats = monitor.get_pool_stats()

if stats['utilization_pct'] > 80:
    print("⚠️  High pool utilization!")
    print(f"   Current: {stats['pool_size']}")
    print(f"   Checked out: {stats['checked_out']}")
    print(f"   Overflow: {stats['overflow']}")
    print()
    print("Recommendations:")
    print("   1. Increase pool_size")
    print("   2. Deploy PgBouncer")
    print("   3. Optimize query performance")
```

---

## Best Practices

### 1. Always Use Async for I/O

```python
# ❌ BAD: Synchronous I/O blocks event loop
def get_data():
    result = requests.get("https://api.example.com/data")
    return result.json()

# ✅ GOOD: Async I/O is non-blocking
async def get_data():
    async with httpx.AsyncClient() as client:
        result = await client.get("https://api.example.com/data")
        return result.json()
```

### 2. Cache Aggressively

```python
# ❌ BAD: Repeated database queries
for emission in emissions:
    factor = await db.query(Factor).get(emission.factor_id)

# ✅ GOOD: Cache factor lookups
factor_cache = {}
for emission in emissions:
    if emission.factor_id not in factor_cache:
        factor_cache[emission.factor_id] = await db.query(Factor).get(emission.factor_id)
    factor = factor_cache[emission.factor_id]
```

### 3. Use Cursor Pagination for Large Datasets

```python
# ❌ BAD: Offset pagination (slow for large offsets)
page = 1000
emissions = await db.query(Emission).offset(page * 100).limit(100).all()

# ✅ GOOD: Cursor pagination (fast at any position)
emissions = await paginator.paginate(query, cursor=last_cursor, limit=100)
```

### 4. Monitor Everything

```python
# ❌ BAD: No monitoring
result = await expensive_operation()

# ✅ GOOD: Track performance
with monitor.metrics.track_db_query("expensive_operation"):
    result = await expensive_operation()

monitor.api_latency_tracker.record(duration)
```

### 5. Batch Operations When Possible

```python
# ❌ BAD: Individual inserts
for record in records:
    await db.insert(record)

# ✅ GOOD: Batch insert
await db.bulk_insert(records, batch_size=1000)
```

---

## Production Checklist

Before deploying to production:

- [ ] All database indexes created
- [ ] PgBouncer deployed and configured
- [ ] Redis cluster deployed for L2 cache
- [ ] All I/O operations converted to async
- [ ] API response caching enabled
- [ ] Prometheus metrics exported
- [ ] Load tests passed (>5000 req/s, P95 <500ms)
- [ ] Monitoring dashboards configured
- [ ] Alert rules configured
- [ ] Rollback plan documented
- [ ] Performance baselines established

---

**End of Performance Optimization Guide**

For questions or issues, contact the Performance Optimization Team.
