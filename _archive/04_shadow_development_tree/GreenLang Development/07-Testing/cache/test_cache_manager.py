# -*- coding: utf-8 -*-
"""
Unit Tests for Cache Manager

Tests the 4-tier caching system (L1: Memory, L2: Redis, L3: Disk, L4: Database)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from decimal import Decimal
import time


class TestCacheManager:
    """Test CacheManager 4-tier caching"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test cache manager"""
        # Import here to avoid import errors if module doesn't exist yet
        try:
            from greenlang.cache.cache_manager import CacheManager
            self.CacheManager = CacheManager
        except ImportError:
            pytest.skip("CacheManager not available")

    def test_cache_initialization(self):
        """Test cache manager initializes all tiers"""
        cache = self.CacheManager()

        assert cache.l1_cache is not None  # Memory cache
        assert hasattr(cache, 'l2_cache')  # Redis cache
        assert hasattr(cache, 'l3_cache')  # Disk cache

    def test_l1_memory_cache_hit(self):
        """Test L1 memory cache hit"""
        cache = self.CacheManager()

        key = "test_key_001"
        value = {"result": 12.34, "unit": "kg_co2e"}

        # Set value
        cache.set(key, value)

        # Get value (should hit L1)
        result = cache.get(key)

        assert result == value
        assert cache.stats["l1_hits"] >= 1

    def test_l1_cache_miss_l2_hit(self):
        """Test L1 miss but L2 Redis hit"""
        cache = self.CacheManager()

        # Mock Redis cache
        with patch.object(cache, 'l2_cache') as mock_l2:
            mock_l2.get.return_value = {"result": 56.78}

            # Clear L1 to force L2 lookup
            cache.l1_cache.clear()

            result = cache.get("test_key_002")

            assert result == {"result": 56.78}
            mock_l2.get.assert_called_once()

    def test_cache_eviction_lru(self):
        """Test LRU eviction when cache is full"""
        cache = self.CacheManager(max_size=100)

        # Fill cache beyond capacity
        for i in range(150):
            cache.set(f"key_{i}", f"value_{i}")

        # First keys should be evicted
        assert cache.get("key_0") is None
        assert cache.get("key_149") is not None

    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL"""
        cache = self.CacheManager()

        key = "ttl_test"
        value = {"data": "expires"}

        # Set with 1 second TTL
        cache.set(key, value, ttl=1)

        # Should be available immediately
        assert cache.get(key) == value

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get(key) is None

    def test_cache_invalidation_pattern(self):
        """Test invalidating cache entries by pattern"""
        cache = self.CacheManager()

        # Set multiple keys
        cache.set("user:123:profile", {"name": "Alice"})
        cache.set("user:123:settings", {"theme": "dark"})
        cache.set("user:456:profile", {"name": "Bob"})

        # Invalidate all user:123 entries
        cache.invalidate_pattern("user:123:*")

        # user:123 entries should be gone
        assert cache.get("user:123:profile") is None
        assert cache.get("user:123:settings") is None

        # user:456 should still exist
        assert cache.get("user:456:profile") is not None

    def test_cache_performance_target(self):
        """Test cache operations meet performance targets (<1ms)"""
        cache = self.CacheManager()

        key = "perf_test"
        value = {"data": "performance"}

        # Measure set performance
        start = time.perf_counter()
        cache.set(key, value)
        set_time = (time.perf_counter() - start) * 1000

        # Measure get performance
        start = time.perf_counter()
        cache.get(key)
        get_time = (time.perf_counter() - start) * 1000

        # Should be < 1ms for L1 cache
        assert set_time < 1.0, f"Set took {set_time:.2f}ms"
        assert get_time < 1.0, f"Get took {get_time:.2f}ms"

    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        cache = self.CacheManager()

        # Perform operations
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()

        assert stats["total_requests"] >= 2
        assert stats["cache_hits"] >= 1
        assert stats["cache_misses"] >= 1
        assert "hit_rate" in stats

    def test_cache_serialization(self):
        """Test caching complex objects (Decimal, date, etc.)"""
        cache = self.CacheManager()

        complex_value = {
            "amount": Decimal("123.456"),
            "date": "2025-01-15",
            "nested": {
                "values": [1, 2, 3]
            }
        }

        cache.set("complex", complex_value)
        result = cache.get("complex")

        assert result == complex_value


class TestL1MemoryCache:
    """Test L1 in-memory cache"""

    def test_memory_cache_fast_access(self):
        """Test L1 cache provides fast access"""
        try:
            from greenlang.cache.l1_memory_cache import L1MemoryCache
        except ImportError:
            pytest.skip("L1MemoryCache not available")

        cache = L1MemoryCache(max_size=1000)

        # Set and get should be very fast
        start = time.perf_counter()

        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        for i in range(100):
            cache.get(f"key_{i}")

        elapsed = (time.perf_counter() - start) * 1000

        # 200 operations should take < 10ms
        assert elapsed < 10.0


class TestL2RedisCache:
    """Test L2 Redis cache"""

    @pytest.mark.integration
    def test_redis_cache_connection(self):
        """Test Redis cache connection"""
        try:
            from greenlang.cache.l2_redis_cache import L2RedisCache
        except ImportError:
            pytest.skip("L2RedisCache not available")

        # This would require actual Redis connection
        # Use mock for unit tests
        cache = L2RedisCache(host="localhost", port=6379)

        assert cache is not None


class TestCacheInvalidation:
    """Test cache invalidation strategies"""

    def test_time_based_invalidation(self):
        """Test time-based cache invalidation"""
        try:
            from greenlang.cache.invalidation import TimeBasedInvalidation
        except ImportError:
            pytest.skip("Cache invalidation not available")

        invalidation = TimeBasedInvalidation(ttl=60)

        # Entry should be valid immediately
        assert invalidation.is_valid("key1", set_time=time.time())

        # Entry should be invalid after TTL
        assert not invalidation.is_valid("key1", set_time=time.time() - 61)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
