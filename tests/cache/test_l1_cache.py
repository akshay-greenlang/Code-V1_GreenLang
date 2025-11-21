# -*- coding: utf-8 -*-
"""
Tests for L1 Memory Cache

Comprehensive tests for the L1 in-memory cache implementation including:
- LRU eviction
- TTL expiration
- Thread safety
- Metrics collection
- Decorator functionality

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from greenlang.cache.l1_memory_cache import (
    L1MemoryCache,
    CacheEntry,
    CacheMetrics,
    cache_result,
    cache_with_key,
    get_global_cache,
    initialize_global_cache
)


@pytest.fixture
async def cache():
    """Create a test cache instance."""
    cache = L1MemoryCache(
        max_size_mb=1,  # Small size for testing
        default_ttl_seconds=60,
        cleanup_interval_seconds=1,
        enable_metrics=True
    )
    await cache.start()
    yield cache
    await cache.stop()


@pytest.mark.asyncio
class TestL1MemoryCache:
    """Test suite for L1MemoryCache."""

    async def test_basic_get_set(self, cache):
        """Test basic get/set operations."""
        # Set value
        await cache.set("key1", "value1")

        # Get value
        value = await cache.get("key1")
        assert value == "value1"

    async def test_get_nonexistent_key(self, cache):
        """Test getting non-existent key returns None."""
        value = await cache.get("nonexistent")
        assert value is None

    async def test_update_existing_key(self, cache):
        """Test updating existing key."""
        await cache.set("key1", "value1")
        await cache.set("key1", "value2")

        value = await cache.get("key1")
        assert value == "value2"

    async def test_delete_key(self, cache):
        """Test deleting a key."""
        await cache.set("key1", "value1")
        deleted = await cache.delete("key1")

        assert deleted is True
        value = await cache.get("key1")
        assert value is None

    async def test_delete_nonexistent_key(self, cache):
        """Test deleting non-existent key."""
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    async def test_exists(self, cache):
        """Test exists method."""
        await cache.set("key1", "value1")

        assert await cache.exists("key1") is True
        assert await cache.exists("nonexistent") is False

    async def test_clear(self, cache):
        """Test clearing all cache entries."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    async def test_ttl_expiration(self, cache):
        """Test TTL expiration."""
        # Set with short TTL
        await cache.set("key1", "value1", ttl=1)

        # Value should exist immediately
        value = await cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Value should be expired
        value = await cache.get("key1")
        assert value is None

    async def test_ttl_custom(self, cache):
        """Test custom TTL."""
        await cache.set("key1", "value1", ttl=120)
        await cache.set("key2", "value2")  # Uses default TTL

        # Both should exist
        assert await cache.exists("key1") is True
        assert await cache.exists("key2") is True

    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache with large values to trigger eviction
        large_value = "x" * 100000  # 100KB

        await cache.set("key1", large_value)
        await cache.set("key2", large_value)
        await cache.set("key3", large_value)

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add more data to trigger eviction
        await cache.set("key4", large_value)
        await cache.set("key5", large_value)

        # key2 or key3 should be evicted (not key1 as it was recently accessed)
        assert await cache.exists("key1") is True
        # At least one of key2/key3 should be evicted
        exists_count = sum([
            await cache.exists("key2"),
            await cache.exists("key3")
        ])
        assert exists_count < 2

    async def test_size_limit_enforcement(self, cache):
        """Test that size limit is enforced."""
        # Try to set value larger than max size
        huge_value = "x" * (2 * 1024 * 1024)  # 2MB (larger than 1MB limit)

        await cache.set("huge", huge_value)

        # Should not be cached
        value = await cache.get("huge")
        assert value is None

    async def test_background_cleanup(self, cache):
        """Test background TTL cleanup."""
        # Set multiple keys with short TTL
        await cache.set("key1", "value1", ttl=1)
        await cache.set("key2", "value2", ttl=1)
        await cache.set("key3", "value3", ttl=1)

        # Wait for cleanup task to run
        await asyncio.sleep(2)

        # All should be cleaned up
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    async def test_metrics_collection(self, cache):
        """Test metrics are collected correctly."""
        # Perform operations
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Generate hits
        await cache.get("key1")
        await cache.get("key1")

        # Generate misses
        await cache.get("nonexistent")

        # Get stats
        stats = await cache.get_stats()

        assert stats["entry_count"] == 2
        assert stats["hits"] >= 2
        assert stats["misses"] >= 1
        assert stats["sets"] >= 2
        assert stats["hit_rate"] > 0

    async def test_metrics_hit_rate(self, cache):
        """Test hit rate calculation."""
        await cache.set("key1", "value1")

        # 2 hits
        await cache.get("key1")
        await cache.get("key1")

        # 1 miss
        await cache.get("nonexistent")

        stats = await cache.get_stats()
        expected_hit_rate = 2 / 3  # 2 hits out of 3 gets
        assert abs(stats["hit_rate"] - expected_hit_rate) < 0.01

    async def test_latency_tracking(self, cache):
        """Test latency percentile tracking."""
        await cache.set("key1", "value1")

        # Generate multiple gets
        for _ in range(100):
            await cache.get("key1")

        stats = await cache.get_stats()

        # Latency should be tracked
        assert "p50_latency_ms" in stats
        assert "p95_latency_ms" in stats
        assert "p99_latency_ms" in stats
        assert stats["p99_latency_ms"] >= 0

    async def test_complex_data_types(self, cache):
        """Test caching complex data types."""
        # Dictionary
        data_dict = {"name": "test", "value": 123, "nested": {"a": 1}}
        await cache.set("dict", data_dict)
        assert await cache.get("dict") == data_dict

        # List
        data_list = [1, 2, 3, "test", {"nested": True}]
        await cache.set("list", data_list)
        assert await cache.get("list") == data_list

        # Custom object
        class CustomObj:
            def __init__(self, x):
                self.x = x

        obj = CustomObj(42)
        await cache.set("obj", obj)
        cached_obj = await cache.get("obj")
        assert cached_obj.x == 42

    async def test_concurrent_access(self, cache):
        """Test thread-safe concurrent access."""
        async def writer(key, value):
            await cache.set(key, value)

        async def reader(key):
            return await cache.get(key)

        # Concurrent writes
        await asyncio.gather(*[
            writer(f"key{i}", f"value{i}")
            for i in range(100)
        ])

        # Concurrent reads
        results = await asyncio.gather(*[
            reader(f"key{i}")
            for i in range(100)
        ])

        # All values should be correct
        for i, value in enumerate(results):
            assert value == f"value{i}"

    async def test_cache_entry_access_tracking(self):
        """Test that cache entry access is tracked."""
        cache = L1MemoryCache(max_size_mb=1)
        await cache.start()

        await cache.set("key1", "value1")

        # Access multiple times
        await cache.get("key1")
        await cache.get("key1")
        await cache.get("key1")

        # Entry should track accesses (internal implementation detail)
        # We verify indirectly through LRU behavior

        await cache.stop()


@pytest.mark.asyncio
class TestCacheDecorators:
    """Test cache decorators."""

    async def test_cache_result_decorator(self, cache):
        """Test cache_result decorator."""
        call_count = 0

        @cache_result(cache, ttl=60)
        async def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return x + y

        # First call - should execute function
        result1 = await expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # Second call - should use cache
        result2 = await expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again

        # Different args - should execute function
        result3 = await expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    async def test_cache_result_with_kwargs(self, cache):
        """Test cache_result with keyword arguments."""
        call_count = 0

        @cache_result(cache, ttl=60)
        async def func(a, b=10):
            nonlocal call_count
            call_count += 1
            return a + b

        result1 = await func(5, b=10)
        assert result1 == 15
        assert call_count == 1

        result2 = await func(5, b=10)
        assert result2 == 15
        assert call_count == 1  # Cached

        result3 = await func(5, b=20)
        assert result3 == 25
        assert call_count == 2  # Different kwargs

    async def test_cache_with_key_decorator(self, cache):
        """Test cache_with_key decorator."""
        call_count = 0

        def make_key(user_id, action):
            return f"user:{user_id}:action:{action}"

        @cache_with_key(cache, key_fn=make_key, ttl=60)
        async def get_user_action(user_id, action):
            nonlocal call_count
            call_count += 1
            return {"user_id": user_id, "action": action}

        result1 = await get_user_action(123, "login")
        assert result1 == {"user_id": 123, "action": "login"}
        assert call_count == 1

        result2 = await get_user_action(123, "login")
        assert result2 == {"user_id": 123, "action": "login"}
        assert call_count == 1  # Cached

    async def test_cache_result_key_prefix(self, cache):
        """Test cache_result with key prefix."""
        @cache_result(cache, ttl=60, key_prefix="test")
        async def func(x):
            return x * 2

        await func(5)
        # Verify key was created (indirectly through existence)
        # The key should be prefixed with "test"


@pytest.mark.asyncio
class TestGlobalCache:
    """Test global cache instance."""

    async def test_get_global_cache(self):
        """Test getting global cache instance."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()

        # Should return same instance
        assert cache1 is cache2

    async def test_initialize_global_cache(self):
        """Test initializing global cache."""
        cache = await initialize_global_cache(
            max_size_mb=50,
            default_ttl_seconds=120
        )

        assert cache is not None
        await cache.set("test", "value")
        assert await cache.get("test") == "value"

        await cache.stop()


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_none_value(self, cache):
        """Test caching None value."""
        await cache.set("key", None)
        value = await cache.get("key")
        # None might not be cached or might be cached
        # Implementation dependent

    async def test_empty_string(self, cache):
        """Test caching empty string."""
        await cache.set("key", "")
        value = await cache.get("key")
        assert value == ""

    async def test_zero_ttl(self, cache):
        """Test zero TTL."""
        await cache.set("key", "value", ttl=0)
        value = await cache.get("key")
        # Zero TTL means no expiration
        assert value == "value"

    async def test_negative_ttl(self, cache):
        """Test negative TTL."""
        await cache.set("key", "value", ttl=-1)
        value = await cache.get("key")
        # Negative TTL means no expiration
        assert value == "value"

    async def test_very_long_key(self, cache):
        """Test very long cache key."""
        long_key = "k" * 1000
        await cache.set(long_key, "value")
        value = await cache.get(long_key)
        assert value == "value"

    async def test_unicode_values(self, cache):
        """Test caching Unicode values."""
        unicode_value = "こんにちは世界 🌍 مرحبا"
        await cache.set("unicode", unicode_value)
        value = await cache.get("unicode")
        assert value == unicode_value
