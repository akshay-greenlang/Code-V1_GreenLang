# GreenLang Phase 5 Infrastructure - Quick Start Guide

## Installation

### Dependencies

Add to your `requirements.txt` or install directly:

```bash
# Cache dependencies
pip install redis[hiredis]>=4.5.0
pip install msgpack>=1.0.0
pip install cachetools>=5.3.0

# Database dependencies
pip install sqlalchemy[asyncio]>=2.0.0
pip install asyncpg>=0.28.0
pip install psycopg2-binary>=2.9.0

# Optional: For testing
pip install pytest>=7.4.0
pip install pytest-asyncio>=0.21.0
```

## Usage Examples

### 1. Multi-Layer Cache System

#### Basic Setup

```python
import asyncio
from greenlang.cache import CacheManager, initialize_cache_manager

async def main():
    # Initialize with default configuration
    manager = await initialize_cache_manager()

    # Set a value
    await manager.set("workflow:123", {"name": "My Workflow", "status": "active"}, ttl=3600)

    # Get a value (tries L1 → L2 → L3)
    workflow = await manager.get("workflow:123")
    print(workflow)  # {"name": "My Workflow", "status": "active"}

    # Invalidate a key
    await manager.invalidate("workflow:123")

    # Invalidate all workflows
    await manager.invalidate_pattern("workflow:*")

    # Get analytics
    stats = await manager.get_analytics()
    print(f"Cache hit rate: {stats['overall_hit_rate']:.2%}")
    print(f"L1 hit rate: {stats['l1_hit_rate']:.2%}")
    print(f"L2 hit rate: {stats['l2_hit_rate']:.2%}")

    await manager.stop()

asyncio.run(main())
```

#### Custom Configuration

```python
from greenlang.cache import (
    CacheArchitecture,
    CacheLayerConfig,
    CacheLayer,
    initialize_cache_manager
)

async def main():
    # Create high-performance configuration
    arch = CacheArchitecture.create_high_performance()

    # Or customize
    arch = CacheArchitecture.create_default()
    arch.l1_config.max_size_bytes = 500 * 1024 * 1024  # 500MB
    arch.l2_config.redis_host = "redis.production.example.com"
    arch.warming_config.enabled = True

    manager = await initialize_cache_manager(architecture=arch)

    # Use cache
    await manager.set("key", "value")

    await manager.stop()

asyncio.run(main())
```

#### Using Cache Decorators

```python
from greenlang.cache import L1MemoryCache, cache_result

cache = L1MemoryCache(max_size_mb=100)
await cache.start()

@cache_result(cache, ttl=300)
async def expensive_calculation(x, y):
    # Simulate expensive operation
    await asyncio.sleep(2)
    return x * y

# First call: takes 2 seconds
result1 = await expensive_calculation(5, 10)  # 50

# Second call: instant (cached)
result2 = await expensive_calculation(5, 10)  # 50

await cache.stop()
```

### 2. Database Connection Pooling

#### Initialize Connection Pool

```python
from greenlang.db import initialize_connection_pool

async def main():
    pool = await initialize_connection_pool(
        database_url="postgresql+asyncpg://user:pass@localhost/greenlang",
        pool_size=20,
        max_overflow=10,
        pool_timeout=30
    )

    # Use connection
    async with pool.get_session() as session:
        result = await session.execute(text("SELECT * FROM workflows"))
        workflows = result.fetchall()

    # Check health
    health = await pool.health_check()
    print(f"Database healthy: {health['healthy']}")

    # Get metrics
    metrics = await pool.get_metrics()
    print(f"Active connections: {metrics['active_connections']}")
    print(f"Pool utilization: {metrics['pool_utilization']:.1%}")

    await pool.close()

asyncio.run(main())
```

### 3. Query Optimizer

#### Track Slow Queries

```python
from greenlang.db import initialize_query_optimizer
from sqlalchemy import text

async def main():
    # Initialize optimizer
    optimizer = initialize_query_optimizer(
        slow_query_threshold_ms=100,
        enable_query_cache=True
    )

    # Track a query
    async with optimizer.track_query("SELECT * FROM workflows WHERE user_id = 123") as tracker:
        result = await session.execute(query)
        tracker.result = result

    # Get slow queries
    slow_queries = optimizer.get_slow_queries(limit=10)
    for query in slow_queries:
        print(f"Query: {query['query'][:100]}...")
        print(f"Duration: {query['duration_ms']}ms")

    # Get query metrics
    metrics = optimizer.get_query_metrics(order_by='avg_time_ms')
    for metric in metrics[:10]:
        print(f"{metric['query_text'][:80]}: {metric['avg_time_ms']}ms")

    # Get recommendations
    recommendations = optimizer.get_recommendations()
    for rec in recommendations:
        print(f"- {rec}")

    optimizer.stop()

asyncio.run(main())
```

#### Cacheable Queries

```python
async with optimizer.track_query(
    query_text="SELECT * FROM workflows WHERE id = ?",
    cacheable=True,
    cache_key="workflow:123"
) as tracker:
    # First call: hits database
    result = await session.execute(query)
    tracker.result = result

# Second call: served from cache
async with optimizer.track_query(
    query_text="SELECT * FROM workflows WHERE id = ?",
    cacheable=True,
    cache_key="workflow:123"
) as tracker:
    # Returns cached result instantly
    pass
```

### 4. Cache Invalidation Strategies

#### Event-Based Invalidation

```python
from greenlang.cache import (
    UnifiedInvalidationManager,
    InvalidationEvent,
    InvalidationRule
)

# Create invalidation manager
invalidation_mgr = UnifiedInvalidationManager()
await invalidation_mgr.start()

# Set callback to invalidate cache
async def invalidate_cache(keys):
    for key in keys:
        await cache_manager.invalidate(key)

invalidation_mgr.set_invalidation_callback(invalidate_cache)

# Add rules
workflow_rule = InvalidationRule(
    event=InvalidationEvent.WORKFLOW_UPDATED,
    key_pattern="workflow:{workflow_id}:*"
)
invalidation_mgr.add_event_rule(workflow_rule)

# Trigger event when workflow is updated
await invalidation_mgr.trigger_event(
    InvalidationEvent.WORKFLOW_UPDATED,
    context={"workflow_id": 123}
)
# This will invalidate all keys matching "workflow:123:*"

await invalidation_mgr.stop()
```

#### TTL-Based Invalidation

```python
# Register keys with TTL
invalidation_mgr.register_ttl("workflow:123", ttl_seconds=3600)
invalidation_mgr.register_ttl("user:456", ttl_seconds=1800)

# Background cleanup will automatically remove expired keys
```

#### Pattern-Based Invalidation

```python
# Find and invalidate all matching keys
count = await invalidation_mgr.invalidate_pattern("workflow:*")
print(f"Invalidated {count} workflow cache entries")

# Clear entire namespace
keys = invalidation_mgr.find_namespace_keys("emissions")
print(f"Found {len(keys)} keys in 'emissions' namespace")
```

### 5. Database Migrations

#### Apply Performance Indexes

```bash
# Using psql
psql -U postgres -d greenlang -f migrations/add_performance_indexes.sql

# Using Alembic (if integrated)
alembic upgrade head
```

#### Verify Indexes

```sql
-- Check index usage
SELECT * FROM index_usage_stats
ORDER BY index_scans DESC
LIMIT 20;

-- Check for tables needing indexes
SELECT * FROM tables_needing_indexes;

-- Check index bloat
SELECT * FROM index_bloat_stats
WHERE index_size > '100 MB';
```

## Production Configuration

### Environment Variables

```bash
# Cache Configuration
export GREENLANG_CACHE_L1_SIZE_MB=500
export GREENLANG_CACHE_L2_REDIS_HOST=redis.production.com
export GREENLANG_CACHE_L2_REDIS_PORT=6379
export GREENLANG_CACHE_L2_REDIS_PASSWORD=secret
export GREENLANG_CACHE_L3_DIR=/var/cache/greenlang

# Database Configuration
export GREENLANG_DB_URL=postgresql+asyncpg://user:pass@db.production.com/greenlang
export GREENLANG_DB_POOL_SIZE=50
export GREENLANG_DB_MAX_OVERFLOW=20
export GREENLANG_DB_POOL_TIMEOUT=30

# Monitoring
export GREENLANG_ENABLE_QUERY_CACHE=true
export GREENLANG_SLOW_QUERY_THRESHOLD_MS=100
```

### Full Production Setup

```python
import os
from greenlang.cache import CacheArchitecture, initialize_cache_manager
from greenlang.db import initialize_connection_pool, initialize_query_optimizer

async def initialize_infrastructure():
    # Configure cache
    arch = CacheArchitecture.create_default()

    # L1 Configuration
    arch.l1_config.max_size_bytes = int(os.getenv('GREENLANG_CACHE_L1_SIZE_MB', 100)) * 1024 * 1024

    # L2 Configuration
    arch.l2_config.redis_host = os.getenv('GREENLANG_CACHE_L2_REDIS_HOST', 'localhost')
    arch.l2_config.redis_port = int(os.getenv('GREENLANG_CACHE_L2_REDIS_PORT', 6379))
    arch.l2_config.redis_password = os.getenv('GREENLANG_CACHE_L2_REDIS_PASSWORD')
    arch.l2_config.redis_sentinel_enabled = os.getenv('GREENLANG_CACHE_L2_SENTINEL_ENABLED', 'false') == 'true'

    # L3 Configuration
    arch.l3_config.disk_cache_dir = os.getenv('GREENLANG_CACHE_L3_DIR', '~/.greenlang/cache')

    # Initialize cache
    cache_manager = await initialize_cache_manager(architecture=arch)

    # Initialize database pool
    db_pool = await initialize_connection_pool(
        database_url=os.getenv('GREENLANG_DB_URL'),
        pool_size=int(os.getenv('GREENLANG_DB_POOL_SIZE', 20)),
        max_overflow=int(os.getenv('GREENLANG_DB_MAX_OVERFLOW', 10)),
        pool_timeout=int(os.getenv('GREENLANG_DB_POOL_TIMEOUT', 30))
    )

    # Initialize query optimizer
    query_optimizer = initialize_query_optimizer(
        slow_query_threshold_ms=float(os.getenv('GREENLANG_SLOW_QUERY_THRESHOLD_MS', 100)),
        enable_query_cache=os.getenv('GREENLANG_ENABLE_QUERY_CACHE', 'true') == 'true'
    )

    return cache_manager, db_pool, query_optimizer

# In your application startup
async def startup():
    cache_manager, db_pool, query_optimizer = await initialize_infrastructure()

    # Health check
    cache_health = await cache_manager.health_check()
    db_health = await db_pool.health_check()

    print(f"Cache healthy: {cache_health['healthy']}")
    print(f"Database healthy: {db_health['healthy']}")

    return cache_manager, db_pool, query_optimizer
```

## Monitoring & Observability

### Cache Metrics

```python
# Get comprehensive analytics
stats = await cache_manager.get_analytics()

print("=== Cache Performance ===")
print(f"Overall hit rate: {stats['overall_hit_rate']:.2%}")
print(f"L1 hit rate: {stats['l1_hit_rate']:.2%}")
print(f"L2 hit rate: {stats['l2_hit_rate']:.2%}")
print(f"L3 hit rate: {stats['l3_hit_rate']:.2%}")

print("\n=== Cache Latency ===")
print(f"L1 p99: {stats['l1_p99_ms']:.2f}ms")
print(f"L2 p99: {stats['l2_p99_ms']:.2f}ms")
print(f"L3 p99: {stats['l3_p99_ms']:.2f}ms")

print("\n=== Cache Operations ===")
print(f"Total gets: {stats['total_gets']}")
print(f"Total sets: {stats['total_sets']}")
print(f"Total invalidations: {stats['total_invalidations']}")
print(f"Warming operations: {stats['warming_operations']}")
```

### Database Metrics

```python
# Query optimizer stats
stats = query_optimizer.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Average query time: {stats['avg_query_time_ms']:.2f}ms")
print(f"Slow queries: {stats['slow_queries']}")
print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")

# Connection pool metrics
metrics = await db_pool.get_metrics()
print(f"Active connections: {metrics['active_connections']}")
print(f"Idle connections: {metrics['idle_connections']}")
print(f"Pool utilization: {metrics['pool_utilization']:.1%}")
print(f"Connection errors: {metrics['connection_errors']}")
```

## Troubleshooting

### Cache Issues

**Problem:** Low cache hit rate

```python
# Check cache stats per layer
stats = await cache_manager.get_analytics()

# If L1 is low, increase size
# If L2 is low, check Redis connectivity
# If L3 is low, check disk space

# Enable cache warming
arch.warming_config.enabled = True
arch.warming_config.warm_top_n_items = 5000
```

**Problem:** High cache latency

```python
# Check which layer is slow
stats = await cache_manager.get_analytics()
print(f"L1 p99: {stats['l1_p99_ms']}ms")  # Should be < 10ms
print(f"L2 p99: {stats['l2_p99_ms']}ms")  # Should be < 50ms

# If L2 is slow, check Redis network latency
# If L3 is slow, check disk I/O
```

### Database Issues

**Problem:** Slow queries

```python
# Get slow queries
slow_queries = query_optimizer.get_slow_queries(min_duration_ms=1000)

for query in slow_queries:
    # Analyze query plan
    plan = await query_optimizer.analyze_query(query['query'], session)
    print(f"Cost: {plan.cost}")
    print(f"Uses index: {plan.uses_index}")
    print(f"Recommendations: {plan.recommendations}")
```

**Problem:** Connection pool exhaustion

```python
metrics = await db_pool.get_metrics()
if metrics['pool_utilization'] > 0.9:
    # Increase pool size
    # Or check for connection leaks
    print("Warning: Pool utilization > 90%")
```

## Performance Best Practices

1. **Use appropriate cache layers:**
   - L1 for hot data (< 1s access)
   - L2 for warm data (1s - 1h access)
   - L3 for cold data (> 1h access)

2. **Set appropriate TTLs:**
   - Frequently changing data: 60s
   - Moderately changing data: 3600s (1h)
   - Rarely changing data: 86400s (24h)

3. **Use cache warming for critical data:**
   - Pre-populate cache on startup
   - Use background refresh for frequently accessed data

4. **Monitor query performance:**
   - Set slow query threshold appropriately
   - Review slow queries weekly
   - Add indexes based on query patterns

5. **Tune connection pool:**
   - Monitor pool utilization
   - Adjust pool_size based on concurrent requests
   - Use connection pre-ping to avoid stale connections

---

For more information, see `PHASE_5_INFRASTRUCTURE_SUMMARY.md`.
