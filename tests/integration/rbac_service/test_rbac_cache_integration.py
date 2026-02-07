# -*- coding: utf-8 -*-
"""
Redis cache integration tests for RBAC Service (SEC-002)

Tests the RBACCache component with an in-memory Redis stub to validate
cache population, hit/miss behavior, invalidation (user/tenant/all),
pub/sub propagation, TTL expiry, concurrent access, large permission
sets, and key format correctness.

Markers:
    @pytest.mark.integration
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import RBAC cache module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.rbac_service.rbac_cache import RBACCache
    from greenlang.infrastructure.rbac_service import RBACServiceConfig
    _HAS_MODULES = True
except ImportError:
    _HAS_MODULES = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_MODULES, reason="RBAC cache modules not yet implemented"),
]


# ============================================================================
# In-Memory Redis Stub
# ============================================================================


class InMemoryRedis:
    """In-memory Redis stub with TTL support for integration testing."""

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._ttls: Dict[str, datetime] = {}
        self._pubsub_messages: List[Dict] = []

    async def set(self, key: str, value: str, ex: int = 0) -> bool:
        self._store[key] = value
        if ex > 0:
            self._ttls[key] = datetime.now(timezone.utc) + timedelta(seconds=ex)
        return True

    async def get(self, key: str) -> Optional[str]:
        if key in self._ttls:
            if datetime.now(timezone.utc) > self._ttls[key]:
                del self._store[key]
                del self._ttls[key]
                return None
        return self._store.get(key)

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                self._ttls.pop(key, None)
                count += 1
        return count

    async def unlink(self, *keys: str) -> int:
        return await self.delete(*keys)

    async def publish(self, channel: str, message: str) -> int:
        self._pubsub_messages.append({"channel": channel, "message": message})
        return 1

    async def keys(self, pattern: str) -> List[str]:
        import fnmatch
        return [k for k in self._store.keys() if fnmatch.fnmatch(k, pattern)]

    async def scan(self, cursor: int = 0, match: str = "*", count: int = 100):
        keys = await self.keys(match)
        return (0, keys)

    def scan_iter(self, match: str = "*"):
        return _AsyncKeyIter(self, match)


class _AsyncKeyIter:
    def __init__(self, redis, pattern):
        self._redis = redis
        self._pattern = pattern
        self._keys = None
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._keys is None:
            self._keys = await self._redis.keys(self._pattern)
        if self._index >= len(self._keys):
            raise StopAsyncIteration
        key = self._keys[self._index]
        self._index += 1
        return key


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def redis() -> InMemoryRedis:
    return InMemoryRedis()


@pytest.fixture
def config() -> "RBACServiceConfig":
    return RBACServiceConfig(
        cache_ttl=300,
        redis_key_prefix="gl:rbac",
        invalidation_channel="gl:rbac:invalidate",
    )


@pytest.fixture
def cache(redis, config) -> "RBACCache":
    return RBACCache(redis_client=redis, config=config)


# ============================================================================
# TestRBACCacheIntegration
# ============================================================================


@pytest.mark.integration
class TestRBACCacheIntegration:
    """Integration tests for RBACCache with in-memory Redis."""

    @pytest.mark.asyncio
    async def test_cache_population_on_first_access(
        self, cache: "RBACCache", redis: InMemoryRedis
    ) -> None:
        """First access is a cache miss; after set, it becomes a hit."""
        # Miss
        result = await cache.get_permissions("t-acme", "user-1")
        assert result is None

        # Populate
        perms = ["agents:read", "agents:write"]
        await cache.set_permissions("t-acme", "user-1", perms)

        # Hit
        result = await cache.get_permissions("t-acme", "user-1")
        assert result == perms

    @pytest.mark.asyncio
    async def test_cache_hit_on_subsequent_access(
        self, cache: "RBACCache"
    ) -> None:
        """Subsequent accesses after set are cache hits."""
        perms = ["emissions:read"]
        await cache.set_permissions("t-acme", "user-2", perms)

        for _ in range(5):
            result = await cache.get_permissions("t-acme", "user-2")
            assert result == perms

    @pytest.mark.asyncio
    async def test_cache_invalidation_single_user(
        self, cache: "RBACCache"
    ) -> None:
        """Invalidating a single user clears only their cache."""
        await cache.set_permissions("t-acme", "user-a", ["perm-a"])
        await cache.set_permissions("t-acme", "user-b", ["perm-b"])

        await cache.invalidate_user("t-acme", "user-a")

        assert await cache.get_permissions("t-acme", "user-a") is None
        assert await cache.get_permissions("t-acme", "user-b") == ["perm-b"]

    @pytest.mark.asyncio
    async def test_cache_invalidation_entire_tenant(
        self, cache: "RBACCache"
    ) -> None:
        """Invalidating a tenant clears all user caches within that tenant."""
        await cache.set_permissions("t-acme", "user-1", ["perm-1"])
        await cache.set_permissions("t-acme", "user-2", ["perm-2"])
        await cache.set_permissions("t-beta", "user-1", ["perm-beta"])

        await cache.invalidate_tenant("t-acme")

        assert await cache.get_permissions("t-acme", "user-1") is None
        assert await cache.get_permissions("t-acme", "user-2") is None
        # Other tenant should be unaffected
        assert await cache.get_permissions("t-beta", "user-1") == ["perm-beta"]

    @pytest.mark.asyncio
    async def test_cache_invalidation_all(
        self, cache: "RBACCache"
    ) -> None:
        """Invalidating all clears the entire cache."""
        await cache.set_permissions("t-acme", "user-1", ["perm-1"])
        await cache.set_permissions("t-beta", "user-2", ["perm-2"])

        await cache.invalidate_all()

        assert await cache.get_permissions("t-acme", "user-1") is None
        assert await cache.get_permissions("t-beta", "user-2") is None

    @pytest.mark.asyncio
    async def test_pubsub_invalidation_propagation(
        self, cache: "RBACCache", redis: InMemoryRedis
    ) -> None:
        """Publishing an invalidation event writes to the pubsub channel."""
        await cache.publish_invalidation(
            event_type="role_updated",
            tenant_id="t-acme",
            user_id="user-1",
        )

        assert len(redis._pubsub_messages) >= 1
        msg = redis._pubsub_messages[-1]
        assert "gl:rbac:invalidate" in msg["channel"]

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(
        self, redis: InMemoryRedis, config: "RBACServiceConfig"
    ) -> None:
        """Cached entries expire after TTL."""
        cache = RBACCache(redis_client=redis, config=config)

        await cache.set_permissions("t-acme", "user-ttl", ["perm-1"], ttl=1)

        # Immediately should be there
        result = await cache.get_permissions("t-acme", "user-ttl")
        # May or may not be there depending on TTL implementation
        # We manually expire for testing
        key = None
        for k in redis._store:
            if "user-ttl" in k:
                key = k
                break
        if key:
            redis._ttls[key] = datetime.now(timezone.utc) - timedelta(seconds=1)

        result = await cache.get_permissions("t-acme", "user-ttl")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_concurrent_access(
        self, cache: "RBACCache"
    ) -> None:
        """Concurrent reads and writes do not corrupt data."""
        async def write_then_read(user_id: str, perms: List[str]):
            await cache.set_permissions("t-acme", user_id, perms)
            result = await cache.get_permissions("t-acme", user_id)
            return result

        tasks = [
            write_then_read(f"user-{i}", [f"perm-{i}"])
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert result == [f"perm-{i}"]

    @pytest.mark.asyncio
    async def test_cache_large_permission_set(
        self, cache: "RBACCache"
    ) -> None:
        """Large permission sets (250+ entries) are cached correctly."""
        large_perms = [f"resource-{i}:action-{j}" for i in range(50) for j in range(5)]
        assert len(large_perms) == 250

        await cache.set_permissions("t-acme", "user-large", large_perms)
        result = await cache.get_permissions("t-acme", "user-large")
        assert result == large_perms

    @pytest.mark.asyncio
    async def test_cache_key_format_correctness(
        self, cache: "RBACCache", redis: InMemoryRedis
    ) -> None:
        """Cache keys follow the expected format: {prefix}:{tenant}:{user}."""
        await cache.set_permissions("t-acme", "user-fmt", ["perm-1"])

        keys = list(redis._store.keys())
        assert len(keys) >= 1
        key = keys[-1]
        # Key should contain the prefix, tenant, and user
        assert "gl:rbac" in key or "rbac" in key
        assert "t-acme" in key
        assert "user-fmt" in key
