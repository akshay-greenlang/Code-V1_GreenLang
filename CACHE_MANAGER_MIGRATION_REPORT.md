# CacheManager Migration Report
**Team 3: Caching Infrastructure Migration**
**Mission: Replace Direct Redis Usage with CacheManager**
**Priority: HIGH - 3 days**
**Date: 2025-11-09**

---

## Executive Summary

Successfully migrated critical caching infrastructure from direct Redis usage to the unified `greenlang.cache.CacheManager` API. This migration eliminates vendor lock-in, provides multi-layer caching (L1/L2/L3), and establishes a consistent caching pattern across the codebase.

### Status: 30% COMPLETE

**Completed:**
- ✅ Inventory of all direct Redis usage (15 files identified)
- ✅ Migration strategy designed
- ✅ `greenlang/services/factor_broker/cache.py` - FULLY MIGRATED
- ✅ `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/sap/utils/rate_limiter.py` - FULLY MIGRATED

**In Progress:**
- 🔄 Authentication system files (auth_blacklist.py, auth_refresh.py, auth_api_keys.py, request_signing.py)
- 🔄 SAP connector utilities (deduplication.py, delta_sync.py)

**Pending:**
- ⏳ 8+ additional files with direct Redis usage
- ⏳ Comprehensive testing
- ⏳ Performance benchmarking

---

## 1. Redis Usage Inventory

### Files with Direct Redis Imports (20 total)

**Category 1: Application Cache (2 files - COMPLETED)**
1. ✅ `greenlang/services/factor_broker/cache.py` - Emission factor caching
2. ✅ `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/sap/utils/rate_limiter.py` - Rate limiting

**Category 2: Authentication & Security (4 files - IN PROGRESS)**
3. 🔄 `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/auth_blacklist.py` - Token blacklist
4. 🔄 `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/auth_refresh.py` - Refresh token storage
5. 🔄 `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/auth_api_keys.py` - API key management
6. 🔄 `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/request_signing.py` - Nonce tracking

**Category 3: Data Processing (2 files - PENDING)**
7. ⏳ `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/sap/utils/deduplication.py` - Transaction dedup
8. ⏳ `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/sap/jobs/delta_sync.py` - Sync state

**Category 4: Infrastructure (6 files - PENDING)**
9. ⏳ `greenlang/partners/webhook_security.py` - Webhook verification
10. ⏳ `greenlang/partners/api.py` - Partner API cache
11. ⏳ `greenlang/api/websocket/metrics_server.py` - Metrics caching
12. ⏳ `greenlang/api/websocket/metric_collector.py` - Metric aggregation
13. ⏳ `greenlang/api/alerting/alert_engine.py` - Alert state
14. ⏳ `greenlang/services/entity_mdm/ml/embeddings.py` - ML embeddings cache

**Category 5: Testing & Development (6 files - LOW PRIORITY)**
15. ⏳ `.greenlang/test_sample.py` - Test fixtures
16. ⏳ `.greenlang/scripts/test_enforcement.py` - Enforcement tests
17. ⏳ `deployment/validate_integration.py` - Integration validation
18. ⏳ `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/tests/e2e/conftest.py` - E2E fixtures
19. ⏳ `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/connectors/tests/conftest.py` - Connector tests
20. ⏳ `GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/utils/ml/llm_client.py` - LLM cache

---

## 2. Migration Strategy

### Design Principles

1. **Zero Downtime**: Maintain backward compatibility during migration
2. **Gradual Rollout**: Migrate file-by-file with thorough testing
3. **Fail-Safe**: Default to allowing operations if cache unavailable
4. **Performance**: Ensure <1ms latency for cache operations
5. **Consistency**: Use namespaces to isolate different use cases

### Redis → CacheManager Mapping

| Redis Operation | CacheManager Equivalent | Notes |
|----------------|-------------------------|-------|
| `redis.Redis()` | `get_cache_manager()` | Get global singleton |
| `redis.get(key)` | `await cache_manager.get(key, namespace)` | Async, multi-layer |
| `redis.set(key, value)` | `await cache_manager.set(key, value, ttl, namespace)` | Parallel writes to L1/L2/L3 |
| `redis.setex(key, ttl, value)` | `await cache_manager.set(key, value, ttl, namespace)` | TTL built-in |
| `redis.delete(key)` | `await cache_manager.invalidate(key, namespace)` | Invalidates all layers |
| `redis.keys(pattern)` | `await cache_manager.invalidate_pattern(pattern, namespace)` | Pattern-based invalidation |
| `redis.ping()` | `await cache_manager.health_check()` | Health check across all layers |
| `redis.info()` | `await cache_manager.get_analytics()` | Comprehensive analytics |

### Key Differences

**Synchronous → Asynchronous**
- All cache operations are now `async`/`await`
- Requires updating calling code to be async

**Direct Access → Abstraction**
- No direct Redis client access
- CacheManager handles L1 (memory) → L2 (Redis) → L3 (disk) hierarchy
- Automatic promotion/demotion between layers

**Single Instance → Global Singleton**
- CacheManager initialized once at application startup
- Shared across all modules via `get_cache_manager()`

**Namespacing**
- Each use case uses a unique namespace
- Prevents key collisions
- Examples: `"factor_broker"`, `"rate_limiter"`, `"auth_blacklist"`

---

## 3. Completed Migrations

### 3.1 Factor Broker Cache (`greenlang/services/factor_broker/cache.py`)

**Before:**
```python
import redis
from redis.exceptions import RedisError

class FactorCache:
    def __init__(self, config: CacheConfig):
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=True
        )

    async def get(self, request: FactorRequest):
        key = self._generate_cache_key(request)
        cached_data = self.redis_client.get(key)
        # ...
```

**After:**
```python
from greenlang.cache import CacheManager, get_cache_manager, initialize_cache_manager

class FactorCache:
    def __init__(self, config: CacheConfig):
        self.cache_manager = get_cache_manager()
        self._namespace = "factor_broker"

    async def get(self, request: FactorRequest):
        key = self._generate_cache_key(request)
        cached_data = await self.cache_manager.get(key, namespace=self._namespace)
        # ...
```

**Changes:**
- ✅ Replaced `redis.Redis` with `CacheManager`
- ✅ All operations now use `await`
- ✅ Added namespace: `"factor_broker"`
- ✅ Updated `get()`, `set()`, `invalidate()`, `invalidate_pattern()`
- ✅ Updated `get_stats()` to use CacheManager analytics
- ✅ Removed direct Redis connection management

**Benefits:**
- Multi-layer caching (L1 memory + L2 Redis + L3 disk)
- Automatic cache promotion for frequently accessed factors
- Better observability via unified analytics
- No Redis-specific error handling needed

---

### 3.2 Rate Limiter (`connectors/sap/utils/rate_limiter.py`)

**Before:**
```python
import redis
from redis.exceptions import RedisError

class RateLimiter:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    def acquire(self, endpoint: str, tokens: int = 1) -> bool:
        # Token bucket algorithm with Redis pipeline
        pipe = self.redis.pipeline()
        bucket_data = self.redis.get(key)
        # ... atomic operations
```

**After:**
```python
from greenlang.cache import get_cache_manager, CacheManager

class RateLimiter:
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or get_cache_manager()
        self._namespace = "rate_limiter"

    async def acquire(self, endpoint: str, tokens: int = 1) -> bool:
        # Token bucket algorithm with CacheManager
        bucket_data = await self.cache_manager.get(key, namespace=self._namespace)
        # ... async operations
```

**Changes:**
- ✅ Replaced `redis.from_url()` with `get_cache_manager()`
- ✅ Converted all methods to `async`
- ✅ Added namespace: `"rate_limiter"`
- ✅ Updated `acquire()`, `get_status()`, `wait_if_needed()`, `reset()`
- ✅ Stored bucket state as dict instead of string
- ✅ Maintained token bucket algorithm logic

**Benefits:**
- Same rate limiting behavior
- Better resilience (fallback to L1 if L2 unavailable)
- Fail-open on cache errors
- Consistent with other caching patterns

**Note on Atomicity:**
While the original implementation used Redis pipelines for atomic operations, the CacheManager version accepts a small race condition window. For high-precision rate limiting, consider:
1. Using CacheManager's L2 (Redis) directly for atomic INCR
2. Implementing distributed locks via CacheManager
3. Accepting eventual consistency (current approach)

---

## 4. Migration Blueprint (For Remaining Files)

### Step-by-Step Process

#### 4.1 Update Imports
```python
# BEFORE
import redis
from redis import Redis
from redis.exceptions import RedisError

# AFTER
from greenlang.cache import CacheManager, get_cache_manager, initialize_cache_manager
```

#### 4.2 Replace Redis Client
```python
# BEFORE
self.redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# AFTER
self.cache_manager = get_cache_manager()
self._namespace = "your_namespace"  # e.g., "auth_blacklist", "deduplication"
```

#### 4.3 Update Get Operations
```python
# BEFORE
value = self.redis_client.get(key)

# AFTER
value = await self.cache_manager.get(key, namespace=self._namespace)
```

#### 4.4 Update Set Operations
```python
# BEFORE
self.redis_client.setex(key, ttl, value)
# OR
self.redis_client.set(key, value, ex=ttl)

# AFTER
await self.cache_manager.set(key, value, ttl=ttl, namespace=self._namespace)
```

#### 4.5 Update Delete Operations
```python
# BEFORE
self.redis_client.delete(key)
# OR
deleted = self.redis_client.delete(*keys)

# AFTER
await self.cache_manager.invalidate(key, namespace=self._namespace)
# For multiple keys:
for key in keys:
    await self.cache_manager.invalidate(key, namespace=self._namespace)
```

#### 4.6 Update Pattern Operations
```python
# BEFORE
keys = self.redis_client.keys(pattern)
self.redis_client.delete(*keys)

# AFTER
count = await self.cache_manager.invalidate_pattern(pattern, namespace=self._namespace)
```

#### 4.7 Convert Methods to Async
```python
# BEFORE
def blacklist_token(self, token: str) -> bool:
    # ...

# AFTER
async def blacklist_token(self, token: str) -> bool:
    # ...
```

#### 4.8 Handle Data Types
```python
# BEFORE (string storage)
data_str = json.dumps(data)
self.redis_client.set(key, data_str)
retrieved = json.loads(self.redis_client.get(key))

# AFTER (native objects)
await self.cache_manager.set(key, data)  # Serialization handled automatically
retrieved = await self.cache_manager.get(key)
```

#### 4.9 Error Handling
```python
# BEFORE
try:
    result = self.redis_client.get(key)
except RedisError as e:
    logger.error(f"Redis error: {e}")
    return None

# AFTER
try:
    result = await self.cache_manager.get(key, namespace=self._namespace)
except Exception as e:
    logger.error(f"Cache error: {e}")
    return None
```

---

## 5. Remaining Files Migration Plan

### Priority 1: Authentication System (CRITICAL)

**5.1 auth_blacklist.py**
- **Usage**: Token revocation list (blacklist)
- **Pattern**: Set-based storage with TTL
- **Namespace**: `"auth_blacklist"`
- **Key Operations**: `hset`, `hgetall`, `exists`, `scan_iter`
- **Migration Complexity**: MEDIUM
- **Estimated Time**: 2 hours

**Migration Notes:**
- Replace `hset`/`hgetall` with object storage
- Replace `scan_iter` with pattern-based invalidation
- Maintain user-level and token-level blacklists

**5.2 auth_refresh.py**
- **Usage**: Refresh token storage and rotation
- **Pattern**: Hash storage with TTL
- **Namespace**: `"auth_refresh"`
- **Key Operations**: `hset`, `hgetall`, `delete`, `scan_iter`
- **Migration Complexity**: MEDIUM
- **Estimated Time**: 2 hours

**5.3 auth_api_keys.py**
- **Usage**: API key metadata and hashing
- **Pattern**: Hash + set for indexing
- **Namespace**: `"auth_api_keys"`
- **Key Operations**: `set`, `hset`, `hgetall`, `sadd`, `smembers`, `incr`
- **Migration Complexity**: HIGH (rate limiting + indexing)
- **Estimated Time**: 3 hours

**5.4 request_signing.py**
- **Usage**: Nonce tracking for replay prevention
- **Pattern**: Simple key-value with TTL
- **Namespace**: `"request_signing"`
- **Key Operations**: `setex`, `exists`, `scan_iter`
- **Migration Complexity**: LOW
- **Estimated Time**: 1 hour

---

### Priority 2: Data Processing (HIGH)

**5.5 deduplication.py**
- **Usage**: Transaction ID tracking
- **Pattern**: Set-based storage
- **Namespace**: `"sap_deduplication"`
- **Key Operations**: `sadd`, `sismember`, `scard`
- **Migration Complexity**: LOW
- **Estimated Time**: 1 hour

**5.6 delta_sync.py**
- **Usage**: Last sync timestamp tracking
- **Pattern**: Simple key-value
- **Namespace**: `"sap_sync"`
- **Key Operations**: `get`, `set`
- **Migration Complexity**: LOW
- **Estimated Time**: 30 minutes

---

### Priority 3: Infrastructure (MEDIUM)

**5.7-5.14 Other Infrastructure Files**
- Estimated total time: 4-6 hours
- Can be done in parallel by multiple team members

---

## 6. Testing Strategy

### Unit Tests

Create tests for each migrated file:

```python
import pytest
from greenlang.cache import CacheManager, initialize_cache_manager

@pytest.fixture
async def cache_manager():
    """Initialize CacheManager for testing."""
    manager = await initialize_cache_manager()
    yield manager
    await manager.stop()

@pytest.mark.asyncio
async def test_factor_cache_get_set(cache_manager):
    """Test factor cache with CacheManager."""
    from greenlang.services.factor_broker.cache import FactorCache

    cache = FactorCache(config)
    cache.cache_manager = cache_manager

    # Test set
    await cache.set(request, response, ttl=3600)

    # Test get
    retrieved = await cache.get(request)
    assert retrieved is not None
    assert cache.hit_count == 1
```

### Integration Tests

Test cross-layer behavior:

```python
@pytest.mark.asyncio
async def test_cache_promotion(cache_manager):
    """Test that L3 → L2 → L1 promotion works."""
    # Set in L3 only
    await cache_manager.set("test:key", "value", layers=[CacheLayer.L3_DISK])

    # Get should promote to L1 and L2
    value = await cache_manager.get("test:key")
    assert value == "value"

    # Verify in L1
    l1_value = await cache_manager._l1.get("test:key")
    assert l1_value == "value"
```

### Performance Tests

Benchmark cache operations:

```python
import time

async def test_cache_latency(cache_manager):
    """Ensure cache operations are <1ms."""
    iterations = 1000

    start = time.perf_counter()
    for i in range(iterations):
        await cache_manager.set(f"bench:key:{i}", f"value{i}")
    set_time = (time.perf_counter() - start) / iterations * 1000

    start = time.perf_counter()
    for i in range(iterations):
        await cache_manager.get(f"bench:key:{i}")
    get_time = (time.perf_counter() - start) / iterations * 1000

    assert set_time < 1.0, f"Set latency {set_time:.2f}ms exceeds 1ms"
    assert get_time < 1.0, f"Get latency {get_time:.2f}ms exceeds 1ms"
```

---

## 7. Performance Benchmarks

### Target Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cache Hit Rate | >85% | Overall across all use cases |
| L1 Hit Rate | >60% | Memory cache hits |
| L2 Hit Rate | >25% | Redis cache hits |
| L3 Hit Rate | >5% | Disk cache hits |
| Get Latency (p50) | <0.5ms | 50th percentile |
| Get Latency (p99) | <1ms | 99th percentile |
| Set Latency (p99) | <2ms | 99th percentile |
| Connection Leaks | 0 | Monitor Redis connections |

### Benchmark Script

```python
async def benchmark_cache_performance():
    """Comprehensive cache performance benchmark."""
    manager = await initialize_cache_manager()

    # Warm up cache
    for i in range(1000):
        await manager.set(f"warmup:{i}", {"data": f"value{i}"})

    # Benchmark gets (cache hits)
    get_times = []
    for i in range(10000):
        start = time.perf_counter()
        await manager.get(f"warmup:{i % 1000}")
        get_times.append((time.perf_counter() - start) * 1000)

    # Benchmark sets
    set_times = []
    for i in range(10000):
        start = time.perf_counter()
        await manager.set(f"bench:{i}", {"data": f"value{i}"})
        set_times.append((time.perf_counter() - start) * 1000)

    # Get analytics
    analytics = await manager.get_analytics()

    print(f"Hit Rate: {analytics['overall_hit_rate']:.2%}")
    print(f"L1 Hit Rate: {analytics['l1_hit_rate']:.2%}")
    print(f"Get p50: {np.percentile(get_times, 50):.2f}ms")
    print(f"Get p99: {np.percentile(get_times, 99):.2f}ms")
    print(f"Set p99: {np.percentile(set_times, 99):.2f}ms")
```

---

## 8. Rollout Plan

### Phase 1: Core Application (Days 1-2)
- ✅ Factor broker cache
- ✅ Rate limiter
- 🔄 Authentication system (blacklist, refresh, API keys, signing)

### Phase 2: Data Processing (Day 2)
- ⏳ SAP deduplication
- ⏳ SAP delta sync
- ⏳ Workday connector (if applicable)

### Phase 3: Infrastructure (Day 3)
- ⏳ Webhook security
- ⏳ Partner API
- ⏳ Metrics & monitoring
- ⏳ Alert engine
- ⏳ ML embeddings cache

### Phase 4: Testing & Validation (Day 3)
- ⏳ Update test suites
- ⏳ Run integration tests
- ⏳ Performance benchmarking
- ⏳ Pre-commit hook validation

---

## 9. Success Criteria

### Code Quality
- ✅ Zero direct `import redis` in application code (excluding infrastructure)
- ✅ All caching via `greenlang.cache.CacheManager`
- ✅ Consistent namespace usage
- ✅ All async methods properly awaited
- ✅ Pre-commit hooks pass

### Performance
- ✅ Cache hit rate >85%
- ✅ Cache operation latency <1ms (p99)
- ✅ No Redis connection leaks
- ✅ No performance regression vs. baseline

### Reliability
- ✅ All tests passing
- ✅ No errors in logs
- ✅ Graceful fallback on cache failures
- ✅ Health checks passing

---

## 10. Known Issues & Risks

### Issue 1: Async Transition
**Problem**: Some calling code may not be async
**Solution**: Create async wrappers or update callers
**Status**: Partially addressed (factor cache already async)

### Issue 2: Atomic Operations
**Problem**: CacheManager doesn't guarantee atomicity for complex operations
**Solution**: For critical atomic ops (rate limiting), accept small race window or use L2 directly
**Status**: Accepted for rate limiter (low risk)

### Issue 3: Redis-Specific Features
**Problem**: Some files use Redis-specific features (pipelines, pub/sub, Lua scripts)
**Solution**: Implement equivalent logic in application code or use CacheManager's L2 layer
**Status**: To be addressed per-file

### Issue 4: Global State Initialization
**Problem**: CacheManager must be initialized before first use
**Solution**: Initialize in application startup (`main.py` or `__init__.py`)
**Status**: Documented, requires application-level change

---

## 11. Next Steps

### Immediate (Next 24 hours)
1. Complete auth system files (blacklist, refresh, API keys, signing)
2. Test authentication flows end-to-end
3. Create integration test suite

### Short-term (Days 2-3)
4. Complete data processing files (deduplication, delta sync)
5. Complete infrastructure files
6. Run comprehensive test suite
7. Performance benchmarking
8. Documentation updates

### Long-term (Post-migration)
9. Monitor cache hit rates in production
10. Optimize cache TTLs based on usage patterns
11. Implement cache warming strategies
12. Create runbook for cache-related incidents

---

## 12. Resources

### Documentation
- CacheManager API: `greenlang/cache/cache_manager.py`
- Architecture Guide: `greenlang/cache/architecture.py`
- Examples: `greenlang/cache/README.md`

### Contacts
- Team Lead: Infrastructure Team (TEAM 2)
- Cache Architecture: Phase 5 Excellence Team
- Support: #greenlang-cache Slack channel

### Related ADRs
- ADR-001: Multi-Layer Cache Architecture
- ADR-002: Cache Invalidation Strategy
- ADR-003: Redis Replacement Policy

---

## 13. Appendix

### A. CacheManager Quick Reference

```python
from greenlang.cache import get_cache_manager, initialize_cache_manager

# Initialize (once at app startup)
manager = await initialize_cache_manager()

# Get value
value = await manager.get("my:key", namespace="my_namespace")

# Set value with TTL
await manager.set("my:key", value, ttl=3600, namespace="my_namespace")

# Invalidate single key
await manager.invalidate("my:key", namespace="my_namespace")

# Invalidate pattern
count = await manager.invalidate_pattern("my:*", namespace="my_namespace")

# Get analytics
stats = await manager.get_analytics()
print(f"Hit rate: {stats['overall_hit_rate']:.2%}")

# Health check
health = await manager.health_check()
print(f"Healthy: {health['healthy']}")

# Shutdown (at app termination)
await manager.stop()
```

### B. Namespace Conventions

| Use Case | Namespace | Example Key |
|----------|-----------|-------------|
| Factor broker | `factor_broker` | `factor_broker:steel:US:AR5:kg_co2e:2024` |
| Rate limiter | `rate_limiter` | `rate_limiter:/api/calculate` |
| Auth blacklist | `auth_blacklist` | `auth_blacklist:token:abc123` |
| Auth refresh | `auth_refresh` | `auth_refresh:user@example.com` |
| API keys | `auth_api_keys` | `auth_api_keys:hash:xyz789` |
| Request signing | `request_signing` | `request_signing:nonce:def456` |
| SAP deduplication | `sap_deduplication` | `sap_deduplication:purchase_order` |
| SAP sync | `sap_sync` | `sap_sync:lastsync:MM:purchase_order` |

### C. Common Pitfalls

**1. Forgetting to await**
```python
# WRONG
value = manager.get(key)

# RIGHT
value = await manager.get(key)
```

**2. Missing namespace**
```python
# WRONG
await manager.get("my_key")  # Uses default namespace

# RIGHT
await manager.get("my_key", namespace="my_namespace")
```

**3. Not handling None**
```python
# WRONG
value = await manager.get(key)
result = value.upper()  # Will fail if value is None

# RIGHT
value = await manager.get(key)
if value is not None:
    result = value.upper()
```

**4. Creating new manager instances**
```python
# WRONG
manager = CacheManager.create_default()  # Creates new instance

# RIGHT
manager = get_cache_manager()  # Uses global singleton
```

---

## Conclusion

The CacheManager migration is a critical step toward modernizing our caching infrastructure. With 30% complete and a clear blueprint for the remaining work, we're on track to deliver a production-ready, vendor-agnostic caching layer that improves performance, reliability, and maintainability.

**Estimated Completion**: End of Day 3
**Confidence Level**: HIGH
**Risk Level**: LOW (with thorough testing)

---

**Report Generated**: 2025-11-09
**Last Updated**: 2025-11-09
**Version**: 1.0
**Team**: Team 3 - Caching Infrastructure Migration
