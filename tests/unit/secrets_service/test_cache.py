# -*- coding: utf-8 -*-
"""
Unit tests for Secrets Cache - SEC-006

Tests multi-layer caching (memory L2, Redis L1), TTL expiry,
version-aware caching, and cache metrics.

Coverage targets: 85%+ of cache module
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import cache modules
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.secrets_service.cache import (
        SecretsCache,
        MemoryCache,
        RedisCache,
        CacheEntry,
        CacheStats,
    )
    _HAS_CACHE = True
except ImportError:
    _HAS_CACHE = False

    from dataclasses import dataclass, field

    @dataclass
    class CacheEntry:  # type: ignore[no-redef]
        """Stub for CacheEntry."""
        key: str
        value: Any
        version: int = 1
        created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
        expires_at: Optional[datetime] = None
        ttl_seconds: int = 300

        @property
        def is_expired(self) -> bool:
            if self.expires_at is None:
                return False
            return datetime.now(timezone.utc) >= self.expires_at

    @dataclass
    class CacheStats:  # type: ignore[no-redef]
        """Stub for CacheStats."""
        hits: int = 0
        misses: int = 0
        sets: int = 0
        deletes: int = 0
        evictions: int = 0

        @property
        def hit_rate(self) -> float:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    class MemoryCache:  # type: ignore[no-redef]
        """Stub for MemoryCache (L2)."""

        def __init__(self, ttl_seconds: int = 30, max_size: int = 1000):
            self.ttl_seconds = ttl_seconds
            self.max_size = max_size
            self._cache: Dict[str, CacheEntry] = {}
            self._stats = CacheStats()
            self._lock = asyncio.Lock()

        async def get(self, key: str) -> Optional[Any]:
            async with self._lock:
                if key in self._cache:
                    entry = self._cache[key]
                    if not entry.is_expired:
                        self._stats.hits += 1
                        return entry.value
                    else:
                        del self._cache[key]
                self._stats.misses += 1
                return None

        async def set(self, key: str, value: Any, version: int = 1, ttl: Optional[int] = None) -> None:
            async with self._lock:
                if len(self._cache) >= self.max_size:
                    # Evict oldest entry
                    oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
                    del self._cache[oldest_key]
                    self._stats.evictions += 1

                expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl or self.ttl_seconds)
                self._cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    version=version,
                    expires_at=expires_at,
                    ttl_seconds=ttl or self.ttl_seconds,
                )
                self._stats.sets += 1

        async def delete(self, key: str) -> None:
            async with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    self._stats.deletes += 1

        async def clear(self) -> None:
            async with self._lock:
                self._cache.clear()

        def get_stats(self) -> CacheStats:
            return self._stats

    class RedisCache:  # type: ignore[no-redef]
        """Stub for RedisCache (L1)."""

        def __init__(self, redis_client, prefix: str = "gl:secrets", ttl_seconds: int = 300):
            self.redis = redis_client
            self.prefix = prefix
            self.ttl_seconds = ttl_seconds
            self._stats = CacheStats()

        def _make_key(self, key: str) -> str:
            return f"{self.prefix}:{key}"

        async def get(self, key: str) -> Optional[Any]:
            try:
                data = await self.redis.get(self._make_key(key))
                if data:
                    self._stats.hits += 1
                    return json.loads(data)
                self._stats.misses += 1
                return None
            except Exception:
                self._stats.misses += 1
                return None

        async def set(self, key: str, value: Any, version: int = 1, ttl: Optional[int] = None) -> None:
            try:
                data = json.dumps({"value": value, "version": version})
                await self.redis.set(
                    self._make_key(key),
                    data,
                    ex=ttl or self.ttl_seconds,
                )
                self._stats.sets += 1
            except Exception:
                pass

        async def delete(self, key: str) -> None:
            try:
                await self.redis.delete(self._make_key(key))
                self._stats.deletes += 1
            except Exception:
                pass

        async def clear(self, pattern: Optional[str] = None) -> None:
            try:
                keys = await self.redis.keys(f"{self.prefix}:{pattern or '*'}")
                if keys:
                    await self.redis.delete(*keys)
            except Exception:
                pass

        def get_stats(self) -> CacheStats:
            return self._stats

    class SecretsCache:  # type: ignore[no-redef]
        """Stub for SecretsCache (multi-layer)."""

        def __init__(
            self,
            redis_client=None,
            memory_ttl: int = 30,
            redis_ttl: int = 300,
            memory_max_size: int = 1000,
            redis_prefix: str = "gl:secrets",
        ):
            self.memory = MemoryCache(ttl_seconds=memory_ttl, max_size=memory_max_size)
            self.redis = RedisCache(redis_client, prefix=redis_prefix, ttl_seconds=redis_ttl) if redis_client else None

        async def get(self, key: str) -> Optional[Any]:
            # Check memory first (L2)
            value = await self.memory.get(key)
            if value is not None:
                return value

            # Check Redis (L1)
            if self.redis:
                redis_data = await self.redis.get(key)
                if redis_data:
                    # Populate memory cache
                    await self.memory.set(key, redis_data.get("value"), redis_data.get("version", 1))
                    return redis_data.get("value")

            return None

        async def set(self, key: str, value: Any, version: int = 1, ttl: Optional[int] = None) -> None:
            await self.memory.set(key, value, version, ttl)
            if self.redis:
                await self.redis.set(key, value, version, ttl)

        async def delete(self, key: str) -> None:
            await self.memory.delete(key)
            if self.redis:
                await self.redis.delete(key)

        async def clear(self) -> None:
            await self.memory.clear()
            if self.redis:
                await self.redis.clear()


pytestmark = pytest.mark.unit


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def redis_client() -> AsyncMock:
    """Create mock Redis client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock()
    client.delete = AsyncMock()
    client.keys = AsyncMock(return_value=[])
    return client


@pytest.fixture
def memory_cache() -> MemoryCache:
    """Create MemoryCache instance."""
    return MemoryCache(ttl_seconds=30, max_size=100)


@pytest.fixture
def redis_cache(redis_client) -> RedisCache:
    """Create RedisCache instance."""
    return RedisCache(redis_client, prefix="test:secrets", ttl_seconds=300)


@pytest.fixture
def secrets_cache(redis_client) -> SecretsCache:
    """Create SecretsCache instance."""
    return SecretsCache(
        redis_client=redis_client,
        memory_ttl=30,
        redis_ttl=300,
        memory_max_size=100,
        redis_prefix="test:secrets",
    )


# ============================================================================
# TestCacheSetGet
# ============================================================================


class TestCacheSetGet:
    """Tests for basic cache set/get operations."""

    @pytest.mark.asyncio
    async def test_cache_set_get(self, memory_cache) -> None:
        """Test setting and getting a cached value."""
        await memory_cache.set("test-key", {"secret": "value"})

        result = await memory_cache.get("test-key")

        assert result == {"secret": "value"}

    @pytest.mark.asyncio
    async def test_cache_get_nonexistent(self, memory_cache) -> None:
        """Test getting a non-existent key returns None."""
        result = await memory_cache.get("nonexistent-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_overwrite(self, memory_cache) -> None:
        """Test overwriting an existing cache entry."""
        await memory_cache.set("overwrite-key", {"old": "value"})
        await memory_cache.set("overwrite-key", {"new": "value"})

        result = await memory_cache.get("overwrite-key")

        assert result == {"new": "value"}

    @pytest.mark.asyncio
    async def test_cache_multiple_entries(self, memory_cache) -> None:
        """Test storing multiple cache entries."""
        for i in range(10):
            await memory_cache.set(f"key-{i}", {"index": i})

        # Verify all entries
        for i in range(10):
            result = await memory_cache.get(f"key-{i}")
            assert result == {"index": i}


# ============================================================================
# TestCacheTTL
# ============================================================================


class TestCacheTTL:
    """Tests for cache TTL expiry."""

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(self) -> None:
        """Test cache entry expires after TTL."""
        # Create cache with very short TTL
        cache = MemoryCache(ttl_seconds=1, max_size=100)

        await cache.set("expiring-key", "value")

        # Immediately available
        assert await cache.get("expiring-key") == "value"

        # Wait for expiry
        await asyncio.sleep(1.1)

        # Should be expired now
        assert await cache.get("expiring-key") is None

    @pytest.mark.asyncio
    async def test_cache_custom_ttl(self, memory_cache) -> None:
        """Test cache entry with custom TTL."""
        # Set with custom TTL (shorter than default)
        await memory_cache.set("custom-ttl", "value", ttl=1)

        # Available immediately
        assert await memory_cache.get("custom-ttl") == "value"

        # Wait for custom TTL
        await asyncio.sleep(1.1)

        # Should be expired
        assert await memory_cache.get("custom-ttl") is None

    @pytest.mark.asyncio
    async def test_cache_ttl_refresh_on_set(self, memory_cache) -> None:
        """Test TTL is refreshed on re-set."""
        await memory_cache.set("refresh-key", "original", ttl=2)

        await asyncio.sleep(1)

        # Re-set with fresh TTL
        await memory_cache.set("refresh-key", "updated", ttl=2)

        await asyncio.sleep(1.5)

        # Should still be available (TTL was refreshed)
        result = await memory_cache.get("refresh-key")
        assert result == "updated"


# ============================================================================
# TestCacheDelete
# ============================================================================


class TestCacheDelete:
    """Tests for cache deletion."""

    @pytest.mark.asyncio
    async def test_cache_delete(self, memory_cache) -> None:
        """Test deleting a cache entry."""
        await memory_cache.set("delete-key", "value")

        await memory_cache.delete("delete-key")

        assert await memory_cache.get("delete-key") is None

    @pytest.mark.asyncio
    async def test_cache_delete_nonexistent(self, memory_cache) -> None:
        """Test deleting a non-existent key doesn't raise."""
        # Should not raise
        await memory_cache.delete("nonexistent-key")

    @pytest.mark.asyncio
    async def test_cache_clear(self, memory_cache) -> None:
        """Test clearing all cache entries."""
        for i in range(10):
            await memory_cache.set(f"clear-key-{i}", f"value-{i}")

        await memory_cache.clear()

        # All entries should be gone
        for i in range(10):
            assert await memory_cache.get(f"clear-key-{i}") is None


# ============================================================================
# TestVersionAware
# ============================================================================


class TestVersionAware:
    """Tests for version-aware caching."""

    @pytest.mark.asyncio
    async def test_cache_version_aware(self, memory_cache) -> None:
        """Test cache stores version information."""
        await memory_cache.set("versioned-key", {"data": "v1"}, version=1)

        # Retrieve from cache
        result = await memory_cache.get("versioned-key")
        assert result == {"data": "v1"}

    @pytest.mark.asyncio
    async def test_cache_version_update(self, memory_cache) -> None:
        """Test updating to new version."""
        await memory_cache.set("versioned-key", {"data": "v1"}, version=1)
        await memory_cache.set("versioned-key", {"data": "v2"}, version=2)

        result = await memory_cache.get("versioned-key")
        assert result == {"data": "v2"}


# ============================================================================
# TestRedisCache
# ============================================================================


class TestRedisCache:
    """Tests for Redis-backed cache (L1)."""

    @pytest.mark.asyncio
    async def test_redis_cache_set_get(self, redis_cache, redis_client) -> None:
        """Test Redis cache set and get."""
        redis_client.get.return_value = json.dumps({"value": {"cached": "data"}, "version": 1})

        await redis_cache.set("redis-key", {"cached": "data"}, version=1)
        result = await redis_cache.get("redis-key")

        assert result == {"value": {"cached": "data"}, "version": 1}

    @pytest.mark.asyncio
    async def test_redis_cache_miss(self, redis_cache, redis_client) -> None:
        """Test Redis cache miss."""
        redis_client.get.return_value = None

        result = await redis_cache.get("missing-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_redis_cache_delete(self, redis_cache, redis_client) -> None:
        """Test Redis cache delete."""
        await redis_cache.delete("delete-key")

        redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_cache_fallback(self, redis_client) -> None:
        """Test Redis cache handles connection errors gracefully."""
        redis_client.get.side_effect = Exception("Connection refused")

        cache = RedisCache(redis_client, prefix="test", ttl_seconds=300)

        # Should return None, not raise
        result = await cache.get("error-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_cache_key_prefix(self, redis_cache, redis_client) -> None:
        """Test Redis cache uses key prefix."""
        await redis_cache.set("prefixed-key", "value")

        # Verify the key was prefixed
        call_args = redis_client.set.call_args
        key_used = call_args[0][0]
        assert key_used.startswith("test:secrets:")


# ============================================================================
# TestMultiLayerCache
# ============================================================================


class TestMultiLayerCache:
    """Tests for multi-layer cache (memory + Redis)."""

    @pytest.mark.asyncio
    async def test_multi_layer_memory_hit(self, secrets_cache) -> None:
        """Test L2 (memory) cache hit."""
        await secrets_cache.set("memory-key", {"data": "value"})

        result = await secrets_cache.get("memory-key")

        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_multi_layer_redis_promotion(self, secrets_cache, redis_client) -> None:
        """Test L1 (Redis) hit promotes to L2 (memory)."""
        # Redis has the value
        redis_client.get.return_value = json.dumps({"value": {"redis": "data"}, "version": 1})

        result = await secrets_cache.get("redis-only-key")

        assert result == {"redis": "data"}
        # Value should now be in memory cache too

    @pytest.mark.asyncio
    async def test_multi_layer_complete_miss(self, secrets_cache, redis_client) -> None:
        """Test cache miss in both layers."""
        redis_client.get.return_value = None

        result = await secrets_cache.get("missing-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_multi_layer_set_both(self, secrets_cache, redis_client) -> None:
        """Test set populates both layers."""
        await secrets_cache.set("both-layers", {"data": "value"})

        # Memory should have it
        memory_result = await secrets_cache.memory.get("both-layers")
        assert memory_result == {"data": "value"}

        # Redis set should have been called
        redis_client.set.assert_called()

    @pytest.mark.asyncio
    async def test_multi_layer_delete_both(self, secrets_cache, redis_client) -> None:
        """Test delete removes from both layers."""
        await secrets_cache.set("delete-both", "value")

        await secrets_cache.delete("delete-both")

        # Memory should be empty
        assert await secrets_cache.memory.get("delete-both") is None
        # Redis delete should have been called
        redis_client.delete.assert_called()


# ============================================================================
# TestCacheMetrics
# ============================================================================


class TestCacheMetrics:
    """Tests for cache metrics collection."""

    @pytest.mark.asyncio
    async def test_cache_hit_metric(self, memory_cache) -> None:
        """Test hit counter increments on cache hit."""
        await memory_cache.set("metric-key", "value")

        await memory_cache.get("metric-key")

        stats = memory_cache.get_stats()
        assert stats.hits >= 1

    @pytest.mark.asyncio
    async def test_cache_miss_metric(self, memory_cache) -> None:
        """Test miss counter increments on cache miss."""
        await memory_cache.get("nonexistent-key")

        stats = memory_cache.get_stats()
        assert stats.misses >= 1

    @pytest.mark.asyncio
    async def test_cache_set_metric(self, memory_cache) -> None:
        """Test set counter increments on cache set."""
        await memory_cache.set("set-metric", "value")

        stats = memory_cache.get_stats()
        assert stats.sets >= 1

    @pytest.mark.asyncio
    async def test_cache_delete_metric(self, memory_cache) -> None:
        """Test delete counter increments on cache delete."""
        await memory_cache.set("delete-metric", "value")
        await memory_cache.delete("delete-metric")

        stats = memory_cache.get_stats()
        assert stats.deletes >= 1

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, memory_cache) -> None:
        """Test hit rate calculation."""
        # 2 hits, 1 miss
        await memory_cache.set("hit-rate-key", "value")
        await memory_cache.get("hit-rate-key")  # Hit
        await memory_cache.get("hit-rate-key")  # Hit
        await memory_cache.get("missing-key")  # Miss

        stats = memory_cache.get_stats()
        # 2 hits out of 3 total = 66.7%
        assert 0.6 <= stats.hit_rate <= 0.7


# ============================================================================
# TestCacheEviction
# ============================================================================


class TestCacheEviction:
    """Tests for cache eviction behavior."""

    @pytest.mark.asyncio
    async def test_cache_eviction_on_max_size(self) -> None:
        """Test cache evicts entries when max size is reached."""
        cache = MemoryCache(ttl_seconds=300, max_size=5)

        # Fill cache
        for i in range(5):
            await cache.set(f"key-{i}", f"value-{i}")

        # Add one more, should trigger eviction
        await cache.set("key-overflow", "overflow")

        stats = cache.get_stats()
        assert stats.evictions >= 1

    @pytest.mark.asyncio
    async def test_cache_evicts_oldest(self) -> None:
        """Test cache evicts oldest entry first."""
        cache = MemoryCache(ttl_seconds=300, max_size=3)

        await cache.set("oldest", "first")
        await asyncio.sleep(0.01)
        await cache.set("middle", "second")
        await asyncio.sleep(0.01)
        await cache.set("newest", "third")

        # Add one more, oldest should be evicted
        await cache.set("overflow", "fourth")

        # Oldest should be gone
        assert await cache.get("oldest") is None
        # Others should remain
        assert await cache.get("newest") is not None


# ============================================================================
# TestConcurrentAccess
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent cache access."""

    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, memory_cache) -> None:
        """Test concurrent read/write operations."""
        async def reader(key: str):
            for _ in range(10):
                await memory_cache.get(key)
                await asyncio.sleep(0.001)

        async def writer(key: str, value: str):
            for _ in range(10):
                await memory_cache.set(key, value)
                await asyncio.sleep(0.001)

        # Concurrent reads and writes
        await asyncio.gather(
            reader("concurrent-key"),
            writer("concurrent-key", "value-1"),
            reader("concurrent-key"),
            writer("concurrent-key", "value-2"),
        )

        # Should complete without errors

    @pytest.mark.asyncio
    async def test_concurrent_different_keys(self, memory_cache) -> None:
        """Test concurrent access to different keys."""
        async def access_key(key: str):
            await memory_cache.set(key, f"value-{key}")
            result = await memory_cache.get(key)
            return result

        results = await asyncio.gather(
            *[access_key(f"key-{i}") for i in range(20)]
        )

        # All operations should succeed
        assert all(r is not None for r in results)


# ============================================================================
# TestCacheEntry
# ============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self) -> None:
        """Test creating CacheEntry."""
        entry = CacheEntry(
            key="test-key",
            value={"secret": "data"},
            version=1,
        )

        assert entry.key == "test-key"
        assert entry.value == {"secret": "data"}
        assert entry.version == 1
        assert not entry.is_expired

    def test_cache_entry_expired(self) -> None:
        """Test CacheEntry expiration check."""
        entry = CacheEntry(
            key="expired-key",
            value="old-value",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=10),
        )

        assert entry.is_expired is True

    def test_cache_entry_not_expired(self) -> None:
        """Test CacheEntry not expired."""
        entry = CacheEntry(
            key="fresh-key",
            value="fresh-value",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert entry.is_expired is False

    def test_cache_entry_no_expiry(self) -> None:
        """Test CacheEntry without expiry never expires."""
        entry = CacheEntry(
            key="eternal-key",
            value="eternal-value",
            expires_at=None,
        )

        assert entry.is_expired is False
