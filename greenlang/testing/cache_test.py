"""
Cache Testing Framework
======================

Test cases and utilities for testing caching mechanisms.

This module provides specialized test cases for testing cache operations,
performance, TTL, invalidation, and hit/miss rates.
"""

import unittest
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, patch
from contextlib import contextmanager
import time
import hashlib

from .mocks import MockCacheManager
from .assertions import assert_cache_hit_rate


class CacheTestCase(unittest.TestCase):
    """
    Base test case for testing caching mechanisms.

    Provides mock cache, assertions for cache hits/misses, performance testing,
    TTL testing, and invalidation testing.

    Example:
    --------
    ```python
    class TestCaching(CacheTestCase):
        def test_cache_hit(self):
            self.cache.set("key", "value")
            result = self.cache.get("key")
            self.assertEqual(result, "value")
            self.assert_cache_hit("key")

        def test_ttl_expiration(self):
            self.cache.set("key", "value", ttl=1)
            time.sleep(2)
            result = self.cache.get("key")
            self.assertIsNone(result)
            self.assert_cache_miss("key")
    ```
    """

    def setUp(self):
        """Set up test fixtures and mock cache."""
        self.cache = MockCacheManager()

        # Track cache operations
        self.cache_operations = []
        self.hits = 0
        self.misses = 0

        # Performance tracking
        self.operation_times = []

    def tearDown(self):
        """Clean up after tests."""
        self.cache.clear()
        self.cache_operations.clear()

    def set_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> float:
        """
        Set a cache value and track operation time.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            Operation time in seconds
        """
        start = time.time()
        self.cache.set(key, value, ttl=ttl)
        op_time = time.time() - start

        self.cache_operations.append({
            'type': 'set',
            'key': key,
            'value': value,
            'ttl': ttl,
            'time': op_time,
        })

        self.operation_times.append(op_time)
        return op_time

    def get_cache(self, key: str) -> tuple[Any, float, bool]:
        """
        Get a cache value and track operation time and hit/miss.

        Args:
            key: Cache key

        Returns:
            Tuple of (value, operation_time, is_hit)
        """
        start = time.time()
        value = self.cache.get(key)
        op_time = time.time() - start

        is_hit = value is not None

        if is_hit:
            self.hits += 1
        else:
            self.misses += 1

        self.cache_operations.append({
            'type': 'get',
            'key': key,
            'value': value,
            'hit': is_hit,
            'time': op_time,
        })

        self.operation_times.append(op_time)
        return value, op_time, is_hit

    def delete_cache(self, key: str) -> float:
        """
        Delete a cache value and track operation time.

        Args:
            key: Cache key

        Returns:
            Operation time in seconds
        """
        start = time.time()
        self.cache.delete(key)
        op_time = time.time() - start

        self.cache_operations.append({
            'type': 'delete',
            'key': key,
            'time': op_time,
        })

        self.operation_times.append(op_time)
        return op_time

    def assert_cache_hit(self, key: str):
        """Assert that a cache key resulted in a hit."""
        value = self.cache.get(key)
        self.assertIsNotNone(
            value,
            f"Expected cache hit for key '{key}' but got miss"
        )

    def assert_cache_miss(self, key: str):
        """Assert that a cache key resulted in a miss."""
        value = self.cache.get(key)
        self.assertIsNone(
            value,
            f"Expected cache miss for key '{key}' but got hit"
        )

    def assert_cache_contains(self, key: str, expected_value: Any):
        """Assert that cache contains a specific key-value pair."""
        value = self.cache.get(key)
        self.assertEqual(
            value,
            expected_value,
            f"Cache value for '{key}' does not match expected"
        )

    def assert_hit_rate(self, min_rate: float, max_rate: float = 1.0):
        """
        Assert that cache hit rate is within expected range.

        Args:
            min_rate: Minimum acceptable hit rate (0.0 to 1.0)
            max_rate: Maximum acceptable hit rate (0.0 to 1.0)
        """
        total = self.hits + self.misses
        if total == 0:
            self.fail("No cache operations recorded")

        hit_rate = self.hits / total

        self.assertGreaterEqual(
            hit_rate,
            min_rate,
            f"Hit rate {hit_rate:.2%} below minimum {min_rate:.2%}"
        )

        self.assertLessEqual(
            hit_rate,
            max_rate,
            f"Hit rate {hit_rate:.2%} above maximum {max_rate:.2%}"
        )

    def assert_cache_performance(self, max_avg_time: float):
        """
        Assert that cache operations are fast enough.

        Args:
            max_avg_time: Maximum acceptable average operation time in seconds
        """
        if not self.operation_times:
            self.fail("No cache operations recorded")

        avg_time = sum(self.operation_times) / len(self.operation_times)

        self.assertLessEqual(
            avg_time,
            max_avg_time,
            f"Average cache operation time {avg_time:.4f}s exceeds max {max_avg_time:.4f}s"
        )

    def assert_ttl_respected(self, key: str, ttl: int):
        """
        Assert that TTL is respected for a cache key.

        Args:
            key: Cache key
            ttl: Expected TTL in seconds
        """
        # Set value with TTL
        self.cache.set(key, "value", ttl=ttl)

        # Should be available immediately
        self.assert_cache_hit(key)

        # Wait for TTL to expire
        time.sleep(ttl + 0.1)

        # Should be expired
        self.assert_cache_miss(key)

    def assert_cache_invalidated(self, key: str):
        """Assert that a cache key has been invalidated."""
        self.assert_cache_miss(key)

    def simulate_cache_load(
        self,
        num_operations: int,
        hit_probability: float = 0.7
    ):
        """
        Simulate cache load for performance testing.

        Args:
            num_operations: Number of operations to simulate
            hit_probability: Probability of cache hit (0.0 to 1.0)
        """
        import random

        keys = [f"key_{i}" for i in range(int(num_operations * (1 - hit_probability)))]

        for i in range(num_operations):
            if random.random() < hit_probability and i > 0:
                # Simulate cache hit
                key = random.choice(keys[:i])
                self.get_cache(key)
            else:
                # Simulate cache miss and set
                key = random.choice(keys)
                self.set_cache(key, f"value_{i}")
                self.get_cache(key)

    @contextmanager
    def mock_cache_backend(self, backend: str = "redis"):
        """
        Context manager for mocking specific cache backend.

        Args:
            backend: Backend type ('redis', 'memcached', 'memory')
        """
        original_backend = getattr(self.cache, '_backend', None)
        self.cache._backend = backend

        try:
            yield
        finally:
            if original_backend is not None:
                self.cache._backend = original_backend

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get aggregated cache statistics."""
        total = self.hits + self.misses

        return {
            'total_operations': len(self.cache_operations),
            'sets': len([op for op in self.cache_operations if op['type'] == 'set']),
            'gets': len([op for op in self.cache_operations if op['type'] == 'get']),
            'deletes': len([op for op in self.cache_operations if op['type'] == 'delete']),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(total, 1),
            'avg_operation_time': sum(self.operation_times) / max(len(self.operation_times), 1),
            'total_time': sum(self.operation_times),
        }
