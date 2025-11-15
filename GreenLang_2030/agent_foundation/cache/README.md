# GreenLang 4-Tier Caching Infrastructure

Production-ready caching system with Redis cluster support, designed for high-performance agent pipelines.

## Architecture

### 4-Tier Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  L1: In-Memory Cache (5MB, 60s TTL)                         │
│  - Ultra-fast (< 1ms)                                       │
│  - LRU eviction                                             │
│  - Hot data, session data                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  L2: Local Redis (100MB, 300s TTL)                          │
│  - Fast (< 5ms)                                             │
│  - Cross-process sharing                                    │
│  - User profiles, API responses                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  L3: Redis Cluster (10GB, 3600s TTL)                        │
│  - Scalable (< 20ms)                                        │
│  - Distributed caching                                      │
│  - Historical data, reports                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  L4: PostgreSQL Materialized Views (Persistent)             │
│  - Slower but persistent                                    │
│  - Aggregated reports, analytics                            │
└─────────────────────────────────────────────────────────────┘
```

## Features

### RedisManager

- **AsyncIO Redis Client**: Non-blocking operations with `redis.asyncio`
- **Connection Pooling**: Up to 50 concurrent connections
- **High Availability**: 3-node Sentinel support with automatic failover
- **Persistence**: RDB + AOF enabled for durability
- **Retry Logic**: Exponential backoff (0.1s, 0.2s, 0.4s, ...)
- **Health Monitoring**: Continuous health checks (30s interval)
- **Eviction Policy**: allkeys-lru (configurable)

### CacheManager

- **4-Tier Caching**: Hierarchical cache with automatic promotion
- **Cache-Aside Pattern**: Load-through with fallback to database
- **Write-Through**: Optional write to all tiers
- **Bulk Operations**: mget/mset for batch processing
- **Pattern Invalidation**: Invalidate by pattern (e.g., `user:*`)
- **Decorators**: `@cached` and `@cached_with_invalidation`
- **Hit Rate Tracking**: Target >80% hit rate
- **Metrics**: Detailed statistics per tier

## Quick Start

### Installation

```bash
pip install redis[hiredis] cachetools pydantic pyyaml
```

### Basic Usage

```python
import asyncio
from cache_manager import CacheManager, CacheConfig, CacheTier

async def main():
    # Initialize cache
    config = CacheConfig()
    cache = CacheManager(config)
    await cache.initialize()

    try:
        # Set value
        await cache.set("user:1234", {"name": "John"}, tier=CacheTier.L2, ttl=300)

        # Get value (automatic tier promotion)
        user = await cache.get("user:1234")
        print(f"User: {user}")

        # Check stats
        stats = await cache.get_stats()
        print(f"L1 hit rate: {stats[CacheTier.L1].hit_rate:.2%}")

    finally:
        await cache.close()

asyncio.run(main())
```

### Using Decorators

```python
from cache_manager import cached, CacheTier

# Initialize global cache
cache_manager = CacheManager(CacheConfig())
await cache_manager.initialize()

@cached(tier=CacheTier.L2, ttl=300, cache_manager=cache_manager)
async def get_user(user_id: str) -> dict:
    """Fetch user from database with caching."""
    return await db.fetch_user(user_id)

# First call: cache miss, fetches from DB
user = await get_user("1234")

# Second call: cache hit, returns from cache
user = await get_user("1234")  # No DB query!
```

### Invalidation

```python
from cache_manager import cached_with_invalidation

@cached_with_invalidation(
    invalidate_patterns=["user:*", "profile:*"],
    cache_manager=cache_manager
)
async def update_user(user_id: str, data: dict):
    """Update user and invalidate related cache."""
    await db.update_user(user_id, data)
    # Automatically invalidates user:* and profile:*

await update_user("1234", {"name": "Jane"})
```

## Configuration

### cache.yaml

```yaml
redis:
  mode: standalone  # standalone, sentinel, cluster
  standalone:
    host: localhost
    port: 6379
  connection_pool:
    max_connections: 50
  eviction:
    policy: allkeys-lru
    max_memory_mb: 2048

cache_l1:
  enabled: true
  max_size_mb: 5.0
  ttl_seconds: 60

cache_l2:
  enabled: true
  max_size_mb: 100.0
  ttl_seconds: 300

cache_l3:
  enabled: true
  max_size_mb: 10240.0
  ttl_seconds: 3600
  compression_enabled: true

monitoring:
  enabled: true
  metrics:
    targets:
      hit_rate_l1: 0.85
      hit_rate_l2: 0.80
```

## Redis Sentinel (High Availability)

### Setup

```python
from redis_manager import RedisManager, RedisConfig, RedisClusterMode

config = RedisConfig(
    mode=RedisClusterMode.SENTINEL,
    sentinel_hosts=[
        ("localhost", 26379),
        ("localhost", 26380),
        ("localhost", 26381),
    ],
    sentinel_master_name="mymaster",
    max_connections=50,
)

redis_mgr = RedisManager(config)
await redis_mgr.initialize()

# Automatic failover if master goes down!
```

### Sentinel Configuration

```bash
# sentinel.conf
sentinel monitor mymaster 127.0.0.1 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 10000
```

## Performance Benchmarks

### Latency Targets

- **L1 (In-Memory)**: < 1ms
- **L2 (Local Redis)**: < 5ms
- **L3 (Redis Cluster)**: < 20ms
- **L4 (PostgreSQL)**: < 100ms

### Hit Rate Targets

- **L1**: 85% (hot data)
- **L2**: 80% (warm data)
- **L3**: 70% (cold data)
- **Overall**: 80%+ combined hit rate

### Throughput

- **L1**: 1M+ ops/sec
- **L2**: 100K+ ops/sec (with pipelining)
- **L3**: 50K+ ops/sec (distributed)

## Advanced Patterns

### Cache-Aside with Fallback

```python
async def get_emission_factor(material_id: str) -> float:
    """Get emission factor with cache-aside pattern."""
    cache_key = f"emission_factor:{material_id}"

    # Try cache (L1->L2->L3)
    value = await cache.get(cache_key)
    if value is not None:
        return value

    # Cache miss: fetch from database
    value = await db.fetch_emission_factor(material_id)

    # Store in L3 (long-lived)
    await cache.set(cache_key, value, tier=CacheTier.L3, ttl=3600)

    return value
```

### Write-Through Caching

```python
async def update_user_profile(user_id: str, data: dict):
    """Update with write-through to all tiers."""
    # Update database
    await db.update_user(user_id, data)

    # Update all cache tiers
    cache_key = f"user:{user_id}"
    await cache.set(cache_key, data, tier=CacheTier.L3, write_through=True)
```

### Cache Stampede Prevention

```python
import asyncio

_locks = {}

async def get_with_lock(key: str, fetch_fn):
    """Prevent cache stampede with locks."""
    value = await cache.get(key)
    if value is not None:
        return value

    # Acquire lock
    if key not in _locks:
        _locks[key] = asyncio.Lock()

    async with _locks[key]:
        # Double-check after lock
        value = await cache.get(key)
        if value is not None:
            return value

        # Fetch and cache
        value = await fetch_fn()
        await cache.set(key, value, tier=CacheTier.L2, ttl=300)
        return value
```

### Bulk Operations

```python
# Bulk get
users = await cache.mget([
    "user:1", "user:2", "user:3"
])

# Bulk set
user_data = {
    "user:1": {"name": "Alice"},
    "user:2": {"name": "Bob"},
}
await cache.mset(user_data, tier=CacheTier.L2, ttl=300)
```

## Monitoring

### Health Checks

```python
from redis_manager import RedisHealthStatus

health = await redis_mgr.health_check()

print(f"Status: {health.status}")
print(f"Latency: {health.latency_ms:.2f}ms")
print(f"Hit rate: {health.hit_rate:.2%}")
print(f"Memory: {health.used_memory_mb:.2f}MB")
print(f"Connected clients: {health.connected_clients}")
```

### Cache Statistics

```python
stats = await cache.get_stats()

for tier, stat in stats.items():
    print(f"{tier}:")
    print(f"  Hit rate: {stat.hit_rate:.2%}")
    print(f"  Hits: {stat.hits}")
    print(f"  Misses: {stat.misses}")
    print(f"  Avg get time: {stat.avg_get_time_ms:.2f}ms")
    print(f"  Current size: {stat.current_size}/{stat.max_size}")
```

### Alerts

```python
# Check SLA compliance
stats = await cache.get_stats()
target_hit_rate = 0.80

for tier, stat in stats.items():
    if stat.hit_rate < target_hit_rate:
        print(f"⚠️ {tier} hit rate below target: {stat.hit_rate:.2%}")
    else:
        print(f"✓ {tier} meeting SLA: {stat.hit_rate:.2%}")
```

## Production Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  redis-master:
    image: redis:7-alpine
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --appendonly yes
    volumes:
      - redis-data:/data

  redis-sentinel-1:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    volumes:
      - ./sentinel.conf:/etc/redis/sentinel.conf

  redis-sentinel-2:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    volumes:
      - ./sentinel.conf:/etc/redis/sentinel.conf

  redis-sentinel-3:
    image: redis:7-alpine
    command: redis-sentinel /etc/redis/sentinel.conf
    volumes:
      - ./sentinel.conf:/etc/redis/sentinel.conf

volumes:
  redis-data:
```

### Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
data:
  redis.conf: |
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    appendonly yes
    appendfsync everysec
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis
  replicas: 3
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        volumeMounts:
        - name: config
          mountPath: /etc/redis
        - name: data
          mountPath: /data
      volumes:
      - name: config
        configMap:
          name: redis-config
      - name: data
        persistentVolumeClaim:
          claimName: redis-data
```

## Testing

### Run Unit Tests

```bash
# All tests
pytest cache/test_*.py -v

# Specific test file
pytest cache/test_cache_manager.py -v

# With coverage
pytest cache/test_*.py --cov=cache --cov-report=html
```

### Run Integration Tests

```bash
# Requires running Redis
docker run -d -p 6379:6379 redis:7-alpine

# Run integration tests
pytest cache/test_*.py -v -m integration
```

## Troubleshooting

### Redis Connection Failed

```python
# Check Redis is running
redis-cli ping  # Should return PONG

# Check connection
redis-cli -h localhost -p 6379 info server
```

### Low Hit Rate

```python
# Enable cache warming
config.cache_warming_enabled = True
config.warming_patterns = ["emission_factor:*", "config:*"]

# Increase TTL
config.l2_config.ttl_seconds = 600
```

### High Memory Usage

```python
# Check memory usage
health = await redis_mgr.health_check()
print(f"Memory: {health.used_memory_mb:.2f}MB")

# Adjust eviction policy
config.redis_config.eviction_policy = "allkeys-lru"

# Reduce TTL
config.l3_config.ttl_seconds = 1800
```

## API Reference

### RedisManager

```python
class RedisManager:
    async def initialize() -> None
    async def set(key: str, value: Any, ttl: int = None) -> bool
    async def get(key: str) -> Optional[Any]
    async def delete(*keys: str) -> int
    async def exists(*keys: str) -> int
    async def expire(key: str, ttl: int) -> bool
    async def mget(*keys: str) -> List[Optional[Any]]
    async def increment(key: str, amount: int = 1) -> int
    async def health_check() -> RedisHealthCheck
    async def close() -> None
```

### CacheManager

```python
class CacheManager:
    async def initialize() -> None
    async def get(key: str, default: Any = None, promote: bool = True) -> Optional[Any]
    async def set(key: str, value: Any, tier: CacheTier = L2, ttl: int = None, write_through: bool = False) -> bool
    async def delete(key: str) -> bool
    async def invalidate(pattern: str = None, keys: List[str] = None) -> int
    async def mget(keys: List[str]) -> Dict[str, Any]
    async def mset(mapping: Dict[str, Any], tier: CacheTier = L2) -> bool
    async def exists(key: str) -> bool
    async def get_stats() -> Dict[CacheTier, CacheStats]
    async def clear(tier: CacheTier = None) -> None
    async def close() -> None
```

## License

Copyright (c) 2025 GreenLang. All rights reserved.

## Support

For issues or questions:
- GitHub Issues: https://github.com/greenlang/agent_foundation
- Email: support@greenlang.com
- Docs: https://docs.greenlang.com/cache
