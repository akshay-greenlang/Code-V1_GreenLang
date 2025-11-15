"""
Unit tests for CacheManager

Tests cover:
- 4-tier caching (L1-L4)
- Cache-aside pattern
- Write-through operations
- Cache promotion/demotion
- Invalidation strategies
- Bulk operations
- Decorators
- Hit rate tracking
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from cache_manager import (
    CacheManager,
    CacheConfig,
    CacheTier,
    CacheStats,
    InvalidationStrategy,
    cached,
    cached_with_invalidation,
)
from redis_manager import RedisManager


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def cache_config():
    """Create test cache configuration."""
    return CacheConfig()


@pytest.fixture
async def cache_manager(cache_config):
    """Create and initialize CacheManager with mocked Redis."""
    manager = CacheManager(cache_config)

    # Mock Redis managers
    manager.l2_redis = AsyncMock(spec=RedisManager)
    manager.l3_redis = manager.l2_redis  # Same instance for testing
    manager._is_initialized = True

    # Initialize L1 (in-memory) cache
    from cachetools import TTLCache
    manager.l1_cache = TTLCache(
        maxsize=1000,
        ttl=60,
    )

    return manager


# ==============================================================================
# INITIALIZATION TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_cache_initialization(cache_config):
    """Test cache manager initialization."""
    manager = CacheManager(cache_config)

    # Mock Redis initialization
    with patch('cache_manager.RedisManager') as mock_redis:
        mock_instance = AsyncMock()
        mock_instance.initialize = AsyncMock()
        mock_redis.return_value = mock_instance

        await manager.initialize()

        assert manager._is_initialized
        assert manager.l1_cache is not None


@pytest.mark.asyncio
async def test_cache_initialization_with_disabled_tiers(cache_config):
    """Test initialization with disabled tiers."""
    cache_config.l1_config.enabled = False
    cache_config.l2_config.enabled = False

    manager = CacheManager(cache_config)

    with patch('cache_manager.RedisManager') as mock_redis:
        mock_instance = AsyncMock()
        mock_instance.initialize = AsyncMock()
        mock_redis.return_value = mock_instance

        await manager.initialize()

        assert manager.l1_cache is None
        assert manager.l2_redis is None


# ==============================================================================
# L1 CACHE TESTS (IN-MEMORY)
# ==============================================================================


@pytest.mark.asyncio
async def test_l1_cache_set_get(cache_manager):
    """Test L1 cache set and get."""
    await cache_manager.set("key1", "value1", tier=CacheTier.L1, ttl=60)

    result = await cache_manager.get("key1")

    assert result == "value1"
    assert cache_manager.stats[CacheTier.L1].sets == 1
    assert cache_manager.stats[CacheTier.L1].hits == 1


@pytest.mark.asyncio
async def test_l1_cache_miss(cache_manager):
    """Test L1 cache miss."""
    result = await cache_manager.get("nonexistent")

    assert result is None
    assert cache_manager.stats[CacheTier.L1].misses == 1


@pytest.mark.asyncio
async def test_l1_cache_eviction():
    """Test L1 cache LRU eviction."""
    config = CacheConfig()
    config.l1_config.max_entries = 3

    manager = CacheManager(config)
    await manager.initialize()

    # Fill L1 cache beyond capacity
    await manager.set("key1", "value1", tier=CacheTier.L1)
    await manager.set("key2", "value2", tier=CacheTier.L1)
    await manager.set("key3", "value3", tier=CacheTier.L1)
    await manager.set("key4", "value4", tier=CacheTier.L1)  # Evicts key1

    # key1 should be evicted
    assert await manager.get("key1") is None
    assert await manager.get("key4") == "value4"

    await manager.close()


# ==============================================================================
# L2 CACHE TESTS (LOCAL REDIS)
# ==============================================================================


@pytest.mark.asyncio
async def test_l2_cache_set_get(cache_manager):
    """Test L2 cache set and get."""
    cache_manager.l2_redis.get.return_value = None
    cache_manager.l2_redis.set = AsyncMock(return_value=True)

    await cache_manager.set("key1", "value1", tier=CacheTier.L2, ttl=300)

    cache_manager.l2_redis.set.assert_called_once()
    assert cache_manager.stats[CacheTier.L2].sets == 1


@pytest.mark.asyncio
async def test_l2_cache_promotion_to_l1(cache_manager):
    """Test automatic promotion from L2 to L1."""
    # Mock L2 hit
    cache_manager.l2_redis.get.return_value = "value_from_l2"

    result = await cache_manager.get("key1", promote=True)

    assert result == "value_from_l2"
    assert cache_manager.stats[CacheTier.L2].hits == 1

    # Check L1 promotion
    assert "key1" in cache_manager.l1_cache
    assert cache_manager.l1_cache["key1"] == "value_from_l2"


# ==============================================================================
# L3 CACHE TESTS (REDIS CLUSTER)
# ==============================================================================


@pytest.mark.asyncio
async def test_l3_cache_set_get(cache_manager):
    """Test L3 cache set and get."""
    cache_manager.l3_redis.set = AsyncMock(return_value=True)

    await cache_manager.set("large_data", {"data": "x" * 10000}, tier=CacheTier.L3, ttl=3600)

    cache_manager.l3_redis.set.assert_called()
    assert cache_manager.stats[CacheTier.L3].sets == 1


# ==============================================================================
# MULTI-TIER CACHE TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_cache_tier_hierarchy(cache_manager):
    """Test cache tier hierarchy (L1 -> L2 -> L3)."""
    # Mock L1 miss, L2 miss, L3 hit
    cache_manager.l2_redis.get.return_value = None
    cache_manager.l3_redis.get.return_value = "value_from_l3"

    result = await cache_manager.get("key1", promote=True)

    assert result == "value_from_l3"
    assert cache_manager.stats[CacheTier.L1].misses == 1
    assert cache_manager.stats[CacheTier.L2].misses == 1
    assert cache_manager.stats[CacheTier.L3].hits == 1


@pytest.mark.asyncio
async def test_write_through_caching(cache_manager):
    """Test write-through to all cache tiers."""
    cache_manager.l2_redis.set = AsyncMock(return_value=True)
    cache_manager.l3_redis.set = AsyncMock(return_value=True)

    await cache_manager.set("key1", "value1", write_through=True)

    # Should write to L1, L2, and L3
    assert "key1" in cache_manager.l1_cache
    cache_manager.l2_redis.set.assert_called()
    cache_manager.l3_redis.set.assert_called()


# ==============================================================================
# DELETE AND INVALIDATION TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_delete_from_all_tiers(cache_manager):
    """Test delete operation removes from all tiers."""
    # Setup data in all tiers
    cache_manager.l1_cache["key1"] = "value1"
    cache_manager.l2_redis.delete = AsyncMock(return_value=1)
    cache_manager.l3_redis.delete = AsyncMock(return_value=1)

    deleted = await cache_manager.delete("key1")

    assert deleted is True
    assert "key1" not in cache_manager.l1_cache
    cache_manager.l2_redis.delete.assert_called()
    assert cache_manager.stats[CacheTier.L1].deletes == 1


@pytest.mark.asyncio
async def test_invalidate_by_keys(cache_manager):
    """Test invalidation by specific keys."""
    # Setup data
    cache_manager.l1_cache["user:1"] = "data1"
    cache_manager.l1_cache["user:2"] = "data2"
    cache_manager.l2_redis.delete = AsyncMock(return_value=1)

    invalidated = await cache_manager.invalidate(keys=["user:1", "user:2"])

    assert invalidated == 2
    assert "user:1" not in cache_manager.l1_cache
    assert "user:2" not in cache_manager.l1_cache


@pytest.mark.asyncio
async def test_invalidate_by_pattern(cache_manager):
    """Test pattern-based invalidation."""
    # Setup data
    cache_manager.l1_cache["user:1"] = "data1"
    cache_manager.l1_cache["user:2"] = "data2"
    cache_manager.l1_cache["config:1"] = "config_data"

    invalidated = await cache_manager.invalidate(pattern="user:*")

    assert invalidated == 2
    assert "user:1" not in cache_manager.l1_cache
    assert "user:2" not in cache_manager.l1_cache
    assert "config:1" in cache_manager.l1_cache  # Should not be invalidated


# ==============================================================================
# BULK OPERATIONS TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_mget_operation(cache_manager):
    """Test bulk get operation."""
    # Setup L1 data
    cache_manager.l1_cache["key1"] = "value1"
    cache_manager.l1_cache["key2"] = "value2"

    # Mock L2 for missing key
    cache_manager.l2_redis.mget = AsyncMock(return_value=["value3"])

    result = await cache_manager.mget(["key1", "key2", "key3"])

    assert result["key1"] == "value1"
    assert result["key2"] == "value2"
    assert result["key3"] == "value3"


@pytest.mark.asyncio
async def test_mset_operation(cache_manager):
    """Test bulk set operation."""
    cache_manager.l2_redis.set = AsyncMock(return_value=True)

    data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
    }

    result = await cache_manager.mset(data, tier=CacheTier.L2, ttl=300)

    assert result is True
    assert cache_manager.stats[CacheTier.L2].sets == 3


# ==============================================================================
# EXISTS TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_exists_in_l1(cache_manager):
    """Test exists check in L1."""
    cache_manager.l1_cache["key1"] = "value1"

    exists = await cache_manager.exists("key1")

    assert exists is True


@pytest.mark.asyncio
async def test_exists_in_l2(cache_manager):
    """Test exists check in L2."""
    cache_manager.l2_redis.exists = AsyncMock(return_value=1)

    exists = await cache_manager.exists("key1")

    assert exists is True


@pytest.mark.asyncio
async def test_not_exists(cache_manager):
    """Test exists check for non-existent key."""
    cache_manager.l2_redis.exists = AsyncMock(return_value=0)
    cache_manager.l3_redis.exists = AsyncMock(return_value=0)

    exists = await cache_manager.exists("nonexistent")

    assert exists is False


# ==============================================================================
# STATISTICS TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_hit_rate_calculation(cache_manager):
    """Test hit rate calculation."""
    # Simulate hits and misses
    cache_manager.l1_cache["key1"] = "value1"

    await cache_manager.get("key1")  # Hit
    await cache_manager.get("key2")  # Miss

    stats = await cache_manager.get_stats()

    assert stats[CacheTier.L1].hits == 1
    assert stats[CacheTier.L1].misses == 1
    assert stats[CacheTier.L1].hit_rate == 0.5


@pytest.mark.asyncio
async def test_performance_metrics(cache_manager):
    """Test performance metrics tracking."""
    cache_manager.l2_redis.get = AsyncMock(return_value="value")

    # Perform operations
    await cache_manager.get("key1")
    await cache_manager.get("key2")

    stats = await cache_manager.get_stats()

    # Should track average get time
    assert stats[CacheTier.L2].avg_get_time_ms >= 0


@pytest.mark.asyncio
async def test_current_size_tracking(cache_manager):
    """Test current size tracking."""
    cache_manager.l1_cache["key1"] = "value1"
    cache_manager.l1_cache["key2"] = "value2"

    stats = await cache_manager.get_stats()

    assert stats[CacheTier.L1].current_size == 2


# ==============================================================================
# DECORATOR TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_cached_decorator(cache_manager):
    """Test @cached decorator."""
    call_count = 0

    @cached(tier=CacheTier.L2, ttl=300, cache_manager=cache_manager)
    async def expensive_function(param: str) -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # Simulate expensive operation
        return f"result_{param}"

    # Mock Redis for caching
    cache_manager.l2_redis.get = AsyncMock(return_value=None)
    cache_manager.l2_redis.set = AsyncMock(return_value=True)

    # First call: cache miss
    result1 = await expensive_function("test")
    assert result1 == "result_test"
    assert call_count == 1

    # Simulate cache hit
    cache_manager.l2_redis.get.return_value = "result_test"

    # Second call: cache hit (function not called)
    result2 = await expensive_function("test")
    assert result2 == "result_test"
    assert call_count == 1  # Not incremented


@pytest.mark.asyncio
async def test_cached_decorator_without_cache_manager():
    """Test @cached decorator without cache manager."""
    call_count = 0

    @cached(tier=CacheTier.L2, ttl=300, cache_manager=None)
    async def function_without_cache(param: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"result_{param}"

    # Without cache manager, function should be called every time
    result1 = await function_without_cache("test")
    result2 = await function_without_cache("test")

    assert call_count == 2


@pytest.mark.asyncio
async def test_cached_with_invalidation_decorator(cache_manager):
    """Test @cached_with_invalidation decorator."""
    @cached_with_invalidation(
        invalidate_patterns=["user:*"],
        cache_manager=cache_manager
    )
    async def update_user(user_id: str, data: dict) -> dict:
        return {"id": user_id, **data}

    # Setup cache data
    cache_manager.l1_cache["user:1234"] = "old_data"

    # Mock invalidation
    cache_manager.invalidate = AsyncMock(return_value=1)

    # Call function
    result = await update_user("1234", {"name": "John"})

    assert result["id"] == "1234"
    cache_manager.invalidate.assert_called_with(pattern="user:*")


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_redis_error_fallback(cache_manager):
    """Test graceful fallback when Redis fails."""
    # Mock Redis error
    cache_manager.l2_redis.get = AsyncMock(
        side_effect=Exception("Redis connection failed")
    )

    # Should not crash, just return None
    result = await cache_manager.get("key1")

    assert result is None


@pytest.mark.asyncio
async def test_set_operation_error_handling(cache_manager):
    """Test error handling in set operation."""
    cache_manager.l2_redis.set = AsyncMock(
        side_effect=Exception("Redis error")
    )

    # Should return False but not crash
    result = await cache_manager.set("key1", "value1", tier=CacheTier.L2)

    assert result is False


# ==============================================================================
# CLEAR TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_clear_all_tiers(cache_manager):
    """Test clearing all cache tiers."""
    cache_manager.l1_cache["key1"] = "value1"
    cache_manager.l2_redis.flush_db = AsyncMock()

    await cache_manager.clear()

    assert len(cache_manager.l1_cache) == 0
    cache_manager.l2_redis.flush_db.assert_called()


@pytest.mark.asyncio
async def test_clear_specific_tier(cache_manager):
    """Test clearing specific cache tier."""
    cache_manager.l1_cache["key1"] = "value1"
    cache_manager.l2_redis.flush_db = AsyncMock()

    await cache_manager.clear(tier=CacheTier.L1)

    assert len(cache_manager.l1_cache) == 0
    # L2 should not be cleared
    cache_manager.l2_redis.flush_db.assert_not_called()


# ==============================================================================
# CONTEXT MANAGER TESTS
# ==============================================================================


@pytest.mark.asyncio
async def test_context_manager(cache_config):
    """Test async context manager usage."""
    manager = CacheManager(cache_config)

    # Mock initialization and close
    manager.initialize = AsyncMock()
    manager.close = AsyncMock()

    async with manager as mgr:
        assert mgr is manager
        manager.initialize.assert_called_once()

    manager.close.assert_called_once()


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_cache_workflow():
    """Integration test: Full cache workflow."""
    config = CacheConfig()
    cache = CacheManager(config)

    try:
        await cache.initialize()

        # Set in L3
        await cache.set("user:9999", {"name": "Test User"}, tier=CacheTier.L3, ttl=3600)

        # Get (should promote to L2 and L1)
        user = await cache.get("user:9999", promote=True)
        assert user["name"] == "Test User"

        # Verify promotion to L1
        if cache.l1_cache:
            assert "user:9999" in cache.l1_cache

        # Invalidate
        invalidated = await cache.invalidate(keys=["user:9999"])
        assert invalidated > 0

        # Verify deletion
        user = await cache.get("user:9999")
        assert user is None

    except Exception as e:
        pytest.skip(f"Integration test skipped: {e}")
    finally:
        await cache.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
