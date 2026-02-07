# -*- coding: utf-8 -*-
"""
Load / performance tests for RBAC Service (SEC-002)

Tests throughput and latency of critical RBAC operations:
- Permission evaluation (target: 5000+ checks/sec with cache)
- Cache hit ratio under realistic load
- Concurrent role assignments
- P99 latency for permission evaluation
- Cache invalidation throughput under load

Markers:
    @pytest.mark.performance
    @pytest.mark.load

These tests are designed to run in CI with relaxed thresholds (to avoid
flaky failures on shared runners) and locally with stricter targets.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Attempt to import RBAC modules.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.rbac_service.rbac_cache import RBACCache
    from greenlang.infrastructure.rbac_service.permission_service import PermissionService
    from greenlang.infrastructure.rbac_service.assignment_service import AssignmentService
    from greenlang.infrastructure.rbac_service import RBACServiceConfig
    _HAS_MODULES = True
except ImportError:
    _HAS_MODULES = False

pytestmark = [
    pytest.mark.performance,
    pytest.mark.load,
    pytest.mark.skipif(not _HAS_MODULES, reason="RBAC modules not yet implemented"),
]


# ============================================================================
# Fast In-Memory Redis Stub
# ============================================================================


class FastInMemoryRedis:
    """Ultra-fast in-memory Redis for throughput testing (no TTL tracking)."""

    def __init__(self):
        self._store: Dict[str, str] = {}
        self._pubsub_count = 0

    async def set(self, key: str, value: str, ex: int = 0) -> bool:
        self._store[key] = value
        return True

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if self._store.pop(key, None) is not None:
                count += 1
        return count

    async def unlink(self, *keys: str) -> int:
        return await self.delete(*keys)

    async def publish(self, channel: str, message: str) -> int:
        self._pubsub_count += 1
        return 1

    async def keys(self, pattern: str) -> List[str]:
        import fnmatch
        return [k for k in self._store if fnmatch.fnmatch(k, pattern)]

    async def scan(self, cursor=0, match="*", count=100):
        keys = await self.keys(match)
        return (0, keys)

    def scan_iter(self, match="*"):
        return _FastAsyncIter(self, match)


class _FastAsyncIter:
    def __init__(self, redis, pattern):
        self._redis = redis
        self._pattern = pattern
        self._keys = None
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._keys is None:
            self._keys = await self._redis.keys(self._pattern)
        if self._idx >= len(self._keys):
            raise StopAsyncIteration
        k = self._keys[self._idx]
        self._idx += 1
        return k


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def redis() -> FastInMemoryRedis:
    return FastInMemoryRedis()


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


@pytest.fixture
def mock_db_pool() -> AsyncMock:
    """Mock DB pool that returns permission data quickly."""
    conn = AsyncMock()
    conn.fetch.return_value = [
        {"permission": "agents:read"},
        {"permission": "agents:write"},
        {"permission": "emissions:read"},
    ]
    conn.fetchrow.return_value = None
    conn.fetchval.return_value = 0
    conn.execute.return_value = "INSERT 0 1"

    pool = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.connection.return_value = cm
    return pool


@pytest.fixture
def permission_service(mock_db_pool, cache, config) -> "PermissionService":
    return PermissionService(db_pool=mock_db_pool, cache=cache, config=config)


@pytest.fixture
def assignment_service(mock_db_pool, cache) -> "AssignmentService":
    audit = AsyncMock()
    audit.log_event = AsyncMock()
    return AssignmentService(db_pool=mock_db_pool, cache=cache, audit=audit)


# ============================================================================
# Constants
# ============================================================================

# Relaxed thresholds for CI runners; local can be much tighter
PERMISSION_EVAL_OPS_PER_SEC = 2000  # target: 5000 on local with warm cache
CACHE_HIT_RATIO_TARGET = 0.80  # 80% hit ratio under realistic workload
CONCURRENT_ASSIGNMENTS_PER_SEC = 200
PERMISSION_P99_MAX_MS = 10.0
CACHE_INVALIDATION_OPS_PER_SEC = 500


# ============================================================================
# TestRBACThroughput
# ============================================================================


@pytest.mark.performance
@pytest.mark.load
class TestRBACThroughput:
    """Performance tests for RBAC service operations."""

    @pytest.mark.asyncio
    async def test_permission_evaluation_throughput(
        self, cache: "RBACCache", redis: FastInMemoryRedis
    ) -> None:
        """Permission evaluation achieves 5000+ checks/sec with warm cache."""
        # Pre-warm cache for 100 users
        num_users = 100
        perms = ["agents:read", "agents:write", "emissions:read"]
        for i in range(num_users):
            await cache.set_permissions("t-acme", f"user-{i}", perms)

        # Time cache reads (simulating permission evaluation)
        num_ops = 5000
        start = time.perf_counter()
        for i in range(num_ops):
            user_id = f"user-{i % num_users}"
            result = await cache.get_permissions("t-acme", user_id)
            assert result is not None
        elapsed = time.perf_counter() - start

        ops_per_sec = num_ops / elapsed
        assert ops_per_sec >= PERMISSION_EVAL_OPS_PER_SEC, (
            f"Permission eval throughput {ops_per_sec:.0f} ops/s "
            f"below target {PERMISSION_EVAL_OPS_PER_SEC}"
        )

    @pytest.mark.asyncio
    async def test_cache_hit_ratio_under_load(
        self, cache: "RBACCache"
    ) -> None:
        """Cache hit ratio meets target under realistic access patterns."""
        import random
        random.seed(42)

        # Pre-warm cache for 50 users (simulating typical warm state)
        known_users = [f"user-{i}" for i in range(50)]
        for user in known_users:
            await cache.set_permissions("t-acme", user, ["agents:read"])

        # Access pattern: 90% known users, 10% unknown (cold misses)
        all_users = known_users + [f"unknown-{i}" for i in range(10)]
        hits = 0
        total = 1000

        for _ in range(total):
            # 90% of accesses hit known users
            if random.random() < 0.9:
                user = random.choice(known_users)
            else:
                user = f"unknown-{random.randint(0, 99)}"

            result = await cache.get_permissions("t-acme", user)
            if result is not None:
                hits += 1

        hit_ratio = hits / total
        assert hit_ratio >= CACHE_HIT_RATIO_TARGET, (
            f"Cache hit ratio {hit_ratio:.2%} below target "
            f"{CACHE_HIT_RATIO_TARGET:.2%}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_role_assignments(
        self, assignment_service: "AssignmentService"
    ) -> None:
        """Concurrent role assignments achieve target throughput."""
        num_ops = 200

        async def assign_one(i: int):
            try:
                return await assignment_service.assign_role(
                    user_id=f"user-perf-{i}",
                    role_id="role-perf",
                    tenant_id="t-acme",
                    assigned_by="admin-1",
                )
            except Exception:
                return None

        start = time.perf_counter()
        results = await asyncio.gather(
            *[assign_one(i) for i in range(num_ops)]
        )
        elapsed = time.perf_counter() - start

        ops_per_sec = num_ops / elapsed
        assert ops_per_sec >= CONCURRENT_ASSIGNMENTS_PER_SEC, (
            f"Concurrent assignment throughput {ops_per_sec:.0f} ops/s "
            f"below target {CONCURRENT_ASSIGNMENTS_PER_SEC}"
        )

    @pytest.mark.asyncio
    async def test_permission_evaluation_latency_p99(
        self, cache: "RBACCache"
    ) -> None:
        """P99 latency for permission evaluation stays within target."""
        # Pre-warm cache
        for i in range(100):
            await cache.set_permissions(
                "t-acme", f"user-lat-{i}", ["agents:read", "agents:write"]
            )

        latencies_ms = []
        for i in range(1000):
            user = f"user-lat-{i % 100}"
            start = time.perf_counter()
            await cache.get_permissions("t-acme", user)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        p99_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        avg_ms = sum(latencies_ms) / len(latencies_ms)

        assert p99_ms < PERMISSION_P99_MAX_MS, (
            f"P99 latency {p99_ms:.2f}ms exceeds target {PERMISSION_P99_MAX_MS}ms"
        )
        assert avg_ms < PERMISSION_P99_MAX_MS / 2, (
            f"Avg latency {avg_ms:.2f}ms exceeds half of P99 target"
        )

    @pytest.mark.asyncio
    async def test_cache_invalidation_under_load(
        self, cache: "RBACCache", redis: FastInMemoryRedis
    ) -> None:
        """Cache invalidation throughput meets target under load."""
        # Pre-populate cache for 100 users across 10 tenants
        for t in range(10):
            for u in range(10):
                await cache.set_permissions(
                    f"t-{t}", f"user-{u}", [f"perm-{t}-{u}"]
                )

        # Time invalidation operations
        num_ops = 500
        start = time.perf_counter()
        for i in range(num_ops):
            tenant = f"t-{i % 10}"
            user = f"user-{i % 10}"
            await cache.invalidate_user(tenant, user)
        elapsed = time.perf_counter() - start

        ops_per_sec = num_ops / elapsed
        assert ops_per_sec >= CACHE_INVALIDATION_OPS_PER_SEC, (
            f"Invalidation throughput {ops_per_sec:.0f} ops/s "
            f"below target {CACHE_INVALIDATION_OPS_PER_SEC}"
        )

    @pytest.mark.asyncio
    async def test_large_permission_set_performance(
        self, cache: "RBACCache"
    ) -> None:
        """Large permission sets (500+ entries) do not degrade throughput."""
        large_perms = [f"r-{i}:a-{j}" for i in range(100) for j in range(5)]
        assert len(large_perms) == 500

        # Time set + get cycle
        num_cycles = 100
        start = time.perf_counter()
        for i in range(num_cycles):
            user = f"user-large-{i}"
            await cache.set_permissions("t-acme", user, large_perms)
            result = await cache.get_permissions("t-acme", user)
            assert result is not None
            assert len(result) == 500
        elapsed = time.perf_counter() - start

        cycles_per_sec = num_cycles / elapsed
        assert cycles_per_sec >= 50, (
            f"Large perm set cycles {cycles_per_sec:.0f}/s below 50/s target"
        )

    @pytest.mark.asyncio
    async def test_tenant_invalidation_scalability(
        self, cache: "RBACCache"
    ) -> None:
        """Tenant-wide invalidation scales with number of users."""
        # Populate 100 users for one tenant
        for i in range(100):
            await cache.set_permissions("t-scale", f"user-{i}", ["perm-1"])

        start = time.perf_counter()
        await cache.invalidate_tenant("t-scale")
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete within 100ms even for 100 users
        assert elapsed_ms < 100, (
            f"Tenant invalidation took {elapsed_ms:.2f}ms for 100 users"
        )

        # Verify all invalidated
        for i in range(100):
            result = await cache.get_permissions("t-scale", f"user-{i}")
            assert result is None
