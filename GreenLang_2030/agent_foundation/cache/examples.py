"""
GreenLang Cache Usage Examples

This file demonstrates how to use the 4-tier caching system in production scenarios.
"""

import asyncio
import logging
from typing import Dict, Any

from cache_manager import (
    CacheManager,
    CacheConfig,
    CacheTier,
    cached,
    cached_with_invalidation,
)
from redis_manager import RedisManager, RedisConfig, RedisClusterMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# EXAMPLE 1: BASIC CACHE USAGE
# ==============================================================================


async def example_basic_usage():
    """Example: Basic cache operations."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Cache Usage")
    print("="*80)

    # Initialize cache
    config = CacheConfig()
    cache = CacheManager(config)
    await cache.initialize()

    try:
        # Set values in different tiers
        await cache.set("user:1234", {"name": "John Doe", "email": "john@example.com"}, tier=CacheTier.L2, ttl=300)
        await cache.set("config:app", {"theme": "dark", "lang": "en"}, tier=CacheTier.L1, ttl=60)

        # Get values (automatic tier promotion)
        user = await cache.get("user:1234")
        print(f"User retrieved: {user}")

        config_data = await cache.get("config:app")
        print(f"Config retrieved: {config_data}")

        # Check existence
        exists = await cache.exists("user:1234")
        print(f"User exists: {exists}")

        # Delete
        deleted = await cache.delete("config:app")
        print(f"Config deleted: {deleted}")

        # Get stats
        stats = await cache.get_stats()
        for tier, stat in stats.items():
            print(f"{tier}: Hit rate={stat.hit_rate:.2%}, Hits={stat.hits}, Misses={stat.misses}")

    finally:
        await cache.close()


# ==============================================================================
# EXAMPLE 2: CACHE DECORATOR USAGE
# ==============================================================================


# Initialize global cache manager for decorators
cache_manager = None


async def initialize_global_cache():
    """Initialize global cache manager."""
    global cache_manager
    config = CacheConfig()
    cache_manager = CacheManager(config)
    await cache_manager.initialize()


@cached(tier=CacheTier.L2, ttl=300, key_prefix="user", cache_manager=None)
async def get_user_from_database(user_id: str) -> Dict[str, Any]:
    """
    Simulate database query with caching.

    First call: Cache miss, fetches from DB
    Second call: Cache hit, returns from cache
    """
    print(f"[DB QUERY] Fetching user {user_id} from database...")
    await asyncio.sleep(0.1)  # Simulate DB latency
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com"
    }


@cached_with_invalidation(
    invalidate_patterns=["emission_factor:*"],
    cache_manager=None
)
async def update_emission_factor(factor_id: str, new_value: float) -> Dict[str, Any]:
    """
    Update emission factor and invalidate related cache entries.

    This will automatically invalidate all "emission_factor:*" patterns.
    """
    print(f"[DB UPDATE] Updating emission factor {factor_id} to {new_value}")
    await asyncio.sleep(0.05)  # Simulate DB write
    return {"id": factor_id, "value": new_value, "updated": True}


async def example_decorator_usage():
    """Example: Using cache decorators."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Cache Decorator Usage")
    print("="*80)

    await initialize_global_cache()

    try:
        # Update decorator cache_manager references
        global cache_manager
        get_user_from_database.__wrapped__.__globals__['cache_manager'] = cache_manager
        update_emission_factor.__wrapped__.__globals__['cache_manager'] = cache_manager

        # First call: Cache miss
        print("\nFirst call (cache miss):")
        user1 = await get_user_from_database("1234")
        print(f"Result: {user1}")

        # Second call: Cache hit (no DB query)
        print("\nSecond call (cache hit):")
        user2 = await get_user_from_database("1234")
        print(f"Result: {user2}")

        # Update with invalidation
        print("\nUpdate with automatic invalidation:")
        await update_emission_factor("EF_001", 2.5)

    finally:
        if cache_manager:
            await cache_manager.close()


# ==============================================================================
# EXAMPLE 3: MULTI-TIER CACHE WORKFLOW
# ==============================================================================


async def example_multi_tier_workflow():
    """Example: Complex multi-tier caching workflow."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Multi-Tier Cache Workflow")
    print("="*80)

    config = CacheConfig()
    cache = CacheManager(config)
    await cache.initialize()

    try:
        # Scenario: User profile data
        user_data = {
            "id": "user_5678",
            "name": "Jane Smith",
            "email": "jane@example.com",
            "preferences": {"theme": "light", "notifications": True}
        }

        # Set in L3 (long-lived data)
        print("\n1. Setting user data in L3 (Redis Cluster)...")
        await cache.set("user:5678", user_data, tier=CacheTier.L3, ttl=3600)

        # First access: L3 hit, promote to L2 and L1
        print("\n2. First access: Retrieving from L3...")
        user = await cache.get("user:5678", promote=True)
        print(f"   Retrieved: {user['name']}")

        # Second access: L1 hit (fastest)
        print("\n3. Second access: Retrieving from L1...")
        user = await cache.get("user:5678")
        print(f"   Retrieved: {user['name']}")

        # Bulk operations
        print("\n4. Bulk set operation...")
        users = {
            "user:101": {"name": "Alice", "role": "admin"},
            "user:102": {"name": "Bob", "role": "user"},
            "user:103": {"name": "Charlie", "role": "user"},
        }
        await cache.mset(users, tier=CacheTier.L2, ttl=300)

        # Bulk get
        print("\n5. Bulk get operation...")
        retrieved_users = await cache.mget(["user:101", "user:102", "user:103"])
        print(f"   Retrieved {len(retrieved_users)} users")

        # Pattern invalidation
        print("\n6. Invalidating user cache entries...")
        invalidated = await cache.invalidate(pattern="user:10*")
        print(f"   Invalidated {invalidated} entries")

        # Stats
        print("\n7. Cache statistics:")
        stats = await cache.get_stats()
        for tier, stat in stats.items():
            print(f"   {tier}:")
            print(f"     - Hit rate: {stat.hit_rate:.2%}")
            print(f"     - Hits: {stat.hits}, Misses: {stat.misses}")
            print(f"     - Avg get time: {stat.avg_get_time_ms:.2f}ms")

    finally:
        await cache.close()


# ==============================================================================
# EXAMPLE 4: HIGH-AVAILABILITY REDIS WITH SENTINEL
# ==============================================================================


async def example_redis_sentinel():
    """Example: Using Redis Sentinel for high availability."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Redis Sentinel High Availability")
    print("="*80)

    # Configure Redis Sentinel
    redis_config = RedisConfig(
        mode=RedisClusterMode.SENTINEL,
        sentinel_hosts=[
            ("localhost", 26379),
            ("localhost", 26380),
            ("localhost", 26381),
        ],
        sentinel_master_name="mymaster",
        max_connections=50,
    )

    redis_mgr = RedisManager(redis_config)

    try:
        print("\n1. Connecting to Redis Sentinel...")
        # Note: This will fail if Sentinel is not running
        # await redis_mgr.initialize()

        print("   (Sentinel not running - skipping connection)")
        print("   In production, Sentinel provides automatic failover")

        # Health check
        # health = await redis_mgr.health_check()
        # print(f"\n2. Health check: {health.status}")
        # print(f"   - Latency: {health.latency_ms:.2f}ms")
        # print(f"   - Hit rate: {health.hit_rate:.2%}")
        # print(f"   - Connected clients: {health.connected_clients}")

    except Exception as e:
        print(f"   Error: {e}")
        print("   (This is expected if Sentinel is not configured)")

    finally:
        # await redis_mgr.close()
        pass


# ==============================================================================
# EXAMPLE 5: PRODUCTION CACHE PATTERNS
# ==============================================================================


async def example_production_patterns():
    """Example: Production-ready cache patterns."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Production Cache Patterns")
    print("="*80)

    config = CacheConfig()
    cache = CacheManager(config)
    await cache.initialize()

    try:
        # Pattern 1: Cache-aside with fallback
        print("\n1. Cache-aside pattern:")
        async def get_emission_factor(material_id: str) -> float:
            """Get emission factor with cache-aside pattern."""
            cache_key = f"emission_factor:{material_id}"

            # Try cache
            value = await cache.get(cache_key)
            if value is not None:
                print(f"   Cache hit for {material_id}")
                return value

            # Cache miss: fetch from database
            print(f"   Cache miss for {material_id} - fetching from DB")
            await asyncio.sleep(0.05)  # Simulate DB query
            value = 2.5  # Simulated value

            # Store in cache
            await cache.set(cache_key, value, tier=CacheTier.L3, ttl=3600)

            return value

        ef1 = await get_emission_factor("steel")
        ef2 = await get_emission_factor("steel")  # Cache hit

        # Pattern 2: Write-through caching
        print("\n2. Write-through pattern:")
        async def update_user_profile(user_id: str, data: dict):
            """Update user profile with write-through caching."""
            # Update database
            print(f"   Updating database for user {user_id}")
            await asyncio.sleep(0.05)

            # Update all cache tiers
            cache_key = f"user:{user_id}"
            await cache.set(cache_key, data, tier=CacheTier.L3, write_through=True)
            print(f"   Updated all cache tiers")

        await update_user_profile("9999", {"name": "Updated User"})

        # Pattern 3: Cache stampede prevention
        print("\n3. Cache stampede prevention:")
        _locks = {}

        async def get_with_lock(key: str, fetch_fn):
            """Get value with lock to prevent cache stampede."""
            value = await cache.get(key)
            if value is not None:
                return value

            # Acquire lock
            if key not in _locks:
                _locks[key] = asyncio.Lock()

            async with _locks[key]:
                # Double-check after acquiring lock
                value = await cache.get(key)
                if value is not None:
                    return value

                # Fetch and cache
                value = await fetch_fn()
                await cache.set(key, value, tier=CacheTier.L2, ttl=300)
                return value

        async def expensive_computation():
            print("   Running expensive computation...")
            await asyncio.sleep(0.1)
            return {"result": "computed_value"}

        # Simulate concurrent requests
        tasks = [
            get_with_lock("expensive:result", expensive_computation)
            for _ in range(5)
        ]
        results = await asyncio.gather(*tasks)
        print(f"   Completed {len(results)} concurrent requests (only 1 computation)")

    finally:
        await cache.close()


# ==============================================================================
# EXAMPLE 6: MONITORING AND METRICS
# ==============================================================================


async def example_monitoring():
    """Example: Cache monitoring and metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Monitoring and Metrics")
    print("="*80)

    config = CacheConfig()
    cache = CacheManager(config)
    await cache.initialize()

    try:
        # Generate some cache activity
        print("\n1. Generating cache activity...")
        for i in range(100):
            await cache.set(f"key:{i}", f"value:{i}", tier=CacheTier.L2)

        for i in range(150):
            # 100 hits, 50 misses
            await cache.get(f"key:{i % 100}")

        # Get detailed stats
        print("\n2. Cache statistics:")
        stats = await cache.get_stats()

        for tier, stat in stats.items():
            print(f"\n   {tier}:")
            print(f"     Total requests: {stat.total_requests}")
            print(f"     Hits: {stat.hits}")
            print(f"     Misses: {stat.misses}")
            print(f"     Hit rate: {stat.hit_rate:.2%}")
            print(f"     Sets: {stat.sets}")
            print(f"     Deletes: {stat.deletes}")
            print(f"     Avg get time: {stat.avg_get_time_ms:.2f}ms")
            print(f"     Avg set time: {stat.avg_set_time_ms:.2f}ms")

        # Check if meeting SLA
        print("\n3. SLA compliance:")
        l1_hit_rate = stats[CacheTier.L1].hit_rate
        l2_hit_rate = stats[CacheTier.L2].hit_rate
        target = config.hit_rate_target

        print(f"   L1 hit rate: {l1_hit_rate:.2%} (target: {target:.2%})")
        print(f"   L2 hit rate: {l2_hit_rate:.2%} (target: {target:.2%})")

        if l2_hit_rate >= target:
            print("   ✓ Meeting SLA targets")
        else:
            print("   ✗ Below SLA targets - consider cache warming")

    finally:
        await cache.close()


# ==============================================================================
# MAIN
# ==============================================================================


async def main():
    """Run all examples."""
    print("\n")
    print("="*80)
    print("GREENLANG 4-TIER CACHE SYSTEM - USAGE EXAMPLES")
    print("="*80)

    # Run examples
    await example_basic_usage()
    await example_decorator_usage()
    await example_multi_tier_workflow()
    await example_redis_sentinel()
    await example_production_patterns()
    await example_monitoring()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
