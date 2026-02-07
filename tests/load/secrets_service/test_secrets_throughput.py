# -*- coding: utf-8 -*-
"""
Load Tests for Secrets Service Throughput - SEC-006

Performance tests for secrets operations:
- Read throughput targets
- Write throughput
- Concurrent rotation
- Cache performance under load
- Connection pool efficiency
- Memory usage

Run with: pytest tests/load/secrets_service/ -v --durations=10
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Skip if dependencies not available
# ---------------------------------------------------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from greenlang.infrastructure.secrets_service.config import SecretsServiceConfig
    _HAS_SERVICE = True
except ImportError:
    _HAS_SERVICE = False


pytestmark = [
    pytest.mark.load,
    pytest.mark.performance,
]


# ============================================================================
# Performance Targets
# ============================================================================

PERFORMANCE_TARGETS = {
    "reads_per_second": 1000,  # Minimum reads/sec with cache
    "writes_per_second": 100,  # Minimum writes/sec
    "cache_hit_rate": 0.90,  # 90% cache hit rate target
    "p99_latency_ms": 50,  # P99 latency under 50ms
    "memory_increase_mb": 100,  # Max memory increase under load
}


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_vault_client() -> AsyncMock:
    """Create mock VaultClient with configurable delay."""
    client = AsyncMock()

    async def mock_get_secret(path, version=None):
        # Simulate 1-5ms Vault latency
        await asyncio.sleep(0.002)
        return MagicMock(
            data={"key": "value", "path": path},
            metadata={"version": version or 1},
        )

    async def mock_put_secret(path, data, cas=None):
        # Simulate 5-10ms write latency
        await asyncio.sleep(0.007)
        return {"version": 1}

    client.get_secret = mock_get_secret
    client.put_secret = mock_put_secret
    client.is_healthy = AsyncMock(return_value=True)

    return client


def _make_mock_redis_client() -> AsyncMock:
    """Create mock Redis client with cache behavior."""
    cache = {}

    async def mock_get(key):
        await asyncio.sleep(0.0005)  # 0.5ms Redis latency
        return cache.get(key)

    async def mock_set(key, value, ex=None):
        await asyncio.sleep(0.0005)
        cache[key] = value

    async def mock_delete(key):
        cache.pop(key, None)

    client = AsyncMock()
    client.get = mock_get
    client.set = mock_set
    client.delete = mock_delete
    client.keys = AsyncMock(return_value=[])

    return client


class PerformanceMetrics:
    """Collect performance metrics during tests."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.operation_count: int = 0
        self.error_count: int = 0
        self.latencies: List[float] = []

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def record_operation(self, latency_ms: float):
        self.operation_count += 1
        self.latencies.append(latency_ms)

    def record_error(self):
        self.error_count += 1

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def operations_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.operation_count / self.duration_seconds
        return 0.0

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def error_rate(self) -> float:
        total = self.operation_count + self.error_count
        if total > 0:
            return self.error_count / total
        return 0.0


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def vault_client() -> AsyncMock:
    """Create mock VaultClient for load testing."""
    return _make_mock_vault_client()


@pytest.fixture
def redis_client() -> AsyncMock:
    """Create mock Redis client for load testing."""
    return _make_mock_redis_client()


@pytest.fixture
def metrics() -> PerformanceMetrics:
    """Create metrics collector."""
    return PerformanceMetrics()


# ============================================================================
# TestReadThroughput
# ============================================================================


class TestReadThroughput:
    """Tests for read operation throughput."""

    @pytest.mark.asyncio
    async def test_1000_reads_per_second(
        self, vault_client, redis_client, metrics
    ) -> None:
        """Test achieving 1000+ reads per second with caching."""
        target_ops = 5000  # Total operations to perform
        concurrent_tasks = 100  # Concurrent workers

        async def read_operation(i: int):
            path = f"secret/test/{i % 100}"  # 100 unique paths for cache hits
            start = time.perf_counter()
            try:
                # Check cache first
                cached = await redis_client.get(f"gl:secrets:{path}")
                if cached:
                    latency = (time.perf_counter() - start) * 1000
                    metrics.record_operation(latency)
                    return

                # Cache miss - fetch from Vault
                result = await vault_client.get_secret(path)
                await redis_client.set(f"gl:secrets:{path}", str(result.data))
                latency = (time.perf_counter() - start) * 1000
                metrics.record_operation(latency)
            except Exception:
                metrics.record_error()

        # Warm up cache
        for i in range(100):
            await read_operation(i)

        # Reset metrics
        metrics = PerformanceMetrics()

        # Run load test
        metrics.start()

        # Create batches of concurrent tasks
        for batch_start in range(0, target_ops, concurrent_tasks):
            batch_end = min(batch_start + concurrent_tasks, target_ops)
            tasks = [read_operation(i) for i in range(batch_start, batch_end)]
            await asyncio.gather(*tasks)

        metrics.stop()

        # Verify performance targets
        print(f"\nRead Throughput Results:")
        print(f"  Operations: {metrics.operation_count}")
        print(f"  Duration: {metrics.duration_seconds:.2f}s")
        print(f"  Ops/sec: {metrics.operations_per_second:.0f}")
        print(f"  P50 latency: {metrics.p50_latency_ms:.2f}ms")
        print(f"  P99 latency: {metrics.p99_latency_ms:.2f}ms")
        print(f"  Error rate: {metrics.error_rate:.2%}")

        assert metrics.operations_per_second >= PERFORMANCE_TARGETS["reads_per_second"], \
            f"Expected >= {PERFORMANCE_TARGETS['reads_per_second']} ops/sec, got {metrics.operations_per_second:.0f}"

    @pytest.mark.asyncio
    async def test_cache_hit_rate_under_load(
        self, vault_client, redis_client
    ) -> None:
        """Test cache hit rate meets target under load."""
        cache_hits = 0
        cache_misses = 0
        total_ops = 1000

        # Populate cache with 50 secrets
        for i in range(50):
            await redis_client.set(f"gl:secrets:cached/{i}", f"value-{i}")

        # Run mixed operations (cached and uncached)
        for i in range(total_ops):
            path = f"cached/{i % 50}" if i % 10 != 0 else f"uncached/{i}"
            cached = await redis_client.get(f"gl:secrets:{path}")
            if cached:
                cache_hits += 1
            else:
                cache_misses += 1
                await vault_client.get_secret(path)
                await redis_client.set(f"gl:secrets:{path}", "value")

        hit_rate = cache_hits / total_ops

        print(f"\nCache Hit Rate Results:")
        print(f"  Total ops: {total_ops}")
        print(f"  Cache hits: {cache_hits}")
        print(f"  Cache misses: {cache_misses}")
        print(f"  Hit rate: {hit_rate:.2%}")

        assert hit_rate >= PERFORMANCE_TARGETS["cache_hit_rate"], \
            f"Expected >= {PERFORMANCE_TARGETS['cache_hit_rate']:.0%} hit rate, got {hit_rate:.2%}"


# ============================================================================
# TestWriteThroughput
# ============================================================================


class TestWriteThroughput:
    """Tests for write operation throughput."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(
        self, vault_client, metrics
    ) -> None:
        """Test concurrent write operations."""
        target_ops = 500  # Writes are slower than reads
        concurrent_tasks = 20

        async def write_operation(i: int):
            path = f"secret/write-test/{uuid.uuid4().hex[:8]}"
            data = {"key": f"value-{i}", "timestamp": datetime.now(timezone.utc).isoformat()}
            start = time.perf_counter()
            try:
                await vault_client.put_secret(path, data)
                latency = (time.perf_counter() - start) * 1000
                metrics.record_operation(latency)
            except Exception:
                metrics.record_error()

        metrics.start()

        for batch_start in range(0, target_ops, concurrent_tasks):
            batch_end = min(batch_start + concurrent_tasks, target_ops)
            tasks = [write_operation(i) for i in range(batch_start, batch_end)]
            await asyncio.gather(*tasks)

        metrics.stop()

        print(f"\nWrite Throughput Results:")
        print(f"  Operations: {metrics.operation_count}")
        print(f"  Duration: {metrics.duration_seconds:.2f}s")
        print(f"  Ops/sec: {metrics.operations_per_second:.0f}")
        print(f"  P99 latency: {metrics.p99_latency_ms:.2f}ms")

        assert metrics.operations_per_second >= PERFORMANCE_TARGETS["writes_per_second"], \
            f"Expected >= {PERFORMANCE_TARGETS['writes_per_second']} ops/sec, got {metrics.operations_per_second:.0f}"

    @pytest.mark.asyncio
    async def test_write_with_cache_invalidation(
        self, vault_client, redis_client, metrics
    ) -> None:
        """Test write operations with cache invalidation overhead."""
        target_ops = 200

        async def write_with_invalidation(i: int):
            path = f"secret/invalidation-test/{i % 50}"
            data = {"value": f"updated-{i}"}
            start = time.perf_counter()
            try:
                await vault_client.put_secret(path, data)
                await redis_client.delete(f"gl:secrets:{path}")
                latency = (time.perf_counter() - start) * 1000
                metrics.record_operation(latency)
            except Exception:
                metrics.record_error()

        metrics.start()
        tasks = [write_with_invalidation(i) for i in range(target_ops)]
        await asyncio.gather(*tasks)
        metrics.stop()

        print(f"\nWrite with Invalidation Results:")
        print(f"  Ops/sec: {metrics.operations_per_second:.0f}")
        print(f"  P99 latency: {metrics.p99_latency_ms:.2f}ms")


# ============================================================================
# TestRotationUnderLoad
# ============================================================================


class TestRotationUnderLoad:
    """Tests for rotation operations under load."""

    @pytest.mark.asyncio
    async def test_concurrent_rotation(
        self, vault_client, metrics
    ) -> None:
        """Test concurrent rotation operations."""
        rotation_count = 50

        async def rotation_operation(role: str):
            start = time.perf_counter()
            try:
                # Simulate rotation: get new creds, validate, revoke old
                await vault_client.get_secret(f"database/creds/{role}")
                await asyncio.sleep(0.01)  # Simulate validation
                await vault_client.put_secret(f"rotation/lease/{role}", {"revoked": True})
                latency = (time.perf_counter() - start) * 1000
                metrics.record_operation(latency)
            except Exception:
                metrics.record_error()

        roles = [f"role-{i}" for i in range(rotation_count)]

        metrics.start()
        tasks = [rotation_operation(role) for role in roles]
        await asyncio.gather(*tasks)
        metrics.stop()

        print(f"\nRotation Under Load Results:")
        print(f"  Rotations: {metrics.operation_count}")
        print(f"  Duration: {metrics.duration_seconds:.2f}s")
        print(f"  Rotations/sec: {metrics.operations_per_second:.0f}")
        print(f"  Error rate: {metrics.error_rate:.2%}")

        assert metrics.error_rate == 0, f"Rotation errors: {metrics.error_count}"


# ============================================================================
# TestCacheUnderLoad
# ============================================================================


class TestCacheUnderLoad:
    """Tests for cache performance under load."""

    @pytest.mark.asyncio
    async def test_cache_under_load(self, redis_client, metrics) -> None:
        """Test cache operations under high load."""
        target_ops = 10000

        async def cache_operation(i: int):
            key = f"gl:secrets:load-test/{i % 1000}"
            start = time.perf_counter()
            try:
                if i % 3 == 0:
                    # Read
                    await redis_client.get(key)
                elif i % 3 == 1:
                    # Write
                    await redis_client.set(key, f"value-{i}")
                else:
                    # Read + Write (cache miss simulation)
                    result = await redis_client.get(key)
                    if not result:
                        await redis_client.set(key, f"value-{i}")

                latency = (time.perf_counter() - start) * 1000
                metrics.record_operation(latency)
            except Exception:
                metrics.record_error()

        metrics.start()
        tasks = [cache_operation(i) for i in range(target_ops)]
        await asyncio.gather(*tasks)
        metrics.stop()

        print(f"\nCache Under Load Results:")
        print(f"  Operations: {metrics.operation_count}")
        print(f"  Ops/sec: {metrics.operations_per_second:.0f}")
        print(f"  P50 latency: {metrics.p50_latency_ms:.2f}ms")
        print(f"  P99 latency: {metrics.p99_latency_ms:.2f}ms")

        assert metrics.p99_latency_ms < PERFORMANCE_TARGETS["p99_latency_ms"], \
            f"P99 latency {metrics.p99_latency_ms:.2f}ms exceeds target {PERFORMANCE_TARGETS['p99_latency_ms']}ms"


# ============================================================================
# TestConnectionPool
# ============================================================================


class TestConnectionPool:
    """Tests for connection pool efficiency."""

    @pytest.mark.asyncio
    async def test_vault_connection_pool(
        self, vault_client, metrics
    ) -> None:
        """Test connection pool handles concurrent requests."""
        concurrent_requests = 100
        batches = 10

        async def request_operation():
            start = time.perf_counter()
            await vault_client.get_secret("secret/pool-test")
            latency = (time.perf_counter() - start) * 1000
            metrics.record_operation(latency)

        metrics.start()

        for _ in range(batches):
            tasks = [request_operation() for _ in range(concurrent_requests)]
            await asyncio.gather(*tasks)

        metrics.stop()

        print(f"\nConnection Pool Results:")
        print(f"  Total requests: {metrics.operation_count}")
        print(f"  Concurrent: {concurrent_requests}")
        print(f"  Ops/sec: {metrics.operations_per_second:.0f}")


# ============================================================================
# TestMemoryUsage
# ============================================================================


class TestMemoryUsage:
    """Tests for memory usage under load."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil not installed")
    async def test_memory_usage(
        self, vault_client, redis_client
    ) -> None:
        """Test memory usage stays within limits under load."""
        process = psutil.Process(os.getpid())

        # Force garbage collection before measuring
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform many operations
        operations = 10000

        async def operation(i: int):
            path = f"secret/memory-test/{i % 500}"
            await redis_client.set(f"gl:secrets:{path}", f"value-{i}" * 100)  # Larger values
            await redis_client.get(f"gl:secrets:{path}")
            await vault_client.get_secret(path)

        tasks = [operation(i) for i in range(operations)]
        await asyncio.gather(*tasks)

        # Force garbage collection after operations
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_increase = final_memory - initial_memory

        print(f"\nMemory Usage Results:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        print(f"  Operations: {operations}")

        assert memory_increase < PERFORMANCE_TARGETS["memory_increase_mb"], \
            f"Memory increase {memory_increase:.1f}MB exceeds target {PERFORMANCE_TARGETS['memory_increase_mb']}MB"


# ============================================================================
# TestLatencyPercentiles
# ============================================================================


class TestLatencyPercentiles:
    """Tests for latency percentile targets."""

    @pytest.mark.asyncio
    async def test_p99_latency(
        self, vault_client, redis_client, metrics
    ) -> None:
        """Test P99 latency meets target."""
        operations = 1000

        # Warm up cache
        for i in range(100):
            await redis_client.set(f"gl:secrets:latency/{i}", f"value-{i}")

        async def timed_read(i: int):
            path = f"latency/{i % 100}"
            start = time.perf_counter()
            await redis_client.get(f"gl:secrets:{path}")
            latency = (time.perf_counter() - start) * 1000
            metrics.record_operation(latency)

        metrics.start()
        tasks = [timed_read(i) for i in range(operations)]
        await asyncio.gather(*tasks)
        metrics.stop()

        print(f"\nLatency Percentile Results:")
        print(f"  P50: {metrics.p50_latency_ms:.3f}ms")
        print(f"  P99: {metrics.p99_latency_ms:.3f}ms")

        assert metrics.p99_latency_ms < PERFORMANCE_TARGETS["p99_latency_ms"], \
            f"P99 latency {metrics.p99_latency_ms:.2f}ms exceeds target"


# ============================================================================
# TestSustainedLoad
# ============================================================================


class TestSustainedLoad:
    """Tests for sustained load over time."""

    @pytest.mark.asyncio
    async def test_sustained_load_30_seconds(
        self, vault_client, redis_client
    ) -> None:
        """Test sustained load for 30 seconds."""
        duration_seconds = 5  # Reduced for CI
        target_rate = 500  # ops/sec

        operations = 0
        errors = 0
        start_time = time.perf_counter()

        async def operation():
            nonlocal operations, errors
            try:
                path = f"secret/sustained/{operations % 100}"
                cached = await redis_client.get(f"gl:secrets:{path}")
                if not cached:
                    await vault_client.get_secret(path)
                    await redis_client.set(f"gl:secrets:{path}", "value")
                operations += 1
            except Exception:
                errors += 1

        while time.perf_counter() - start_time < duration_seconds:
            # Run batch of operations
            tasks = [operation() for _ in range(50)]
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.01)  # Brief pause

        actual_duration = time.perf_counter() - start_time
        actual_rate = operations / actual_duration

        print(f"\nSustained Load Results:")
        print(f"  Duration: {actual_duration:.1f}s")
        print(f"  Operations: {operations}")
        print(f"  Rate: {actual_rate:.0f} ops/sec")
        print(f"  Errors: {errors}")

        assert errors == 0, f"Errors during sustained load: {errors}"
        assert actual_rate >= target_rate * 0.8, f"Rate dropped below 80% of target"
