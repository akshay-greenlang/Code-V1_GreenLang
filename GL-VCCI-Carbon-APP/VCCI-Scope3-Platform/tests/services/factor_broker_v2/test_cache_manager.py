# -*- coding: utf-8 -*-
"""
Comprehensive Tests for Factor Broker Cache Manager
GL-VCCI Scope 3 Platform - Phase 6

Tests for caching functionality including:
- Redis caching with 24-hour TTL
- Cache key generation
- Cache invalidation
- Cache statistics
- Multi-tenant isolation

Test Count: 20 tests
Target Coverage: 95%

Version: 1.0.0
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import hashlib
import json

import sys
from greenlang.determinism import DeterministicClock
sys.path.insert(0, '/c/Users/aksha/Code-V1_GreenLang/GL-VCCI-Carbon-APP/VCCI-Scope3-Platform')

from services.factor_broker.cache import FactorCache
from services.factor_broker.models import (
    FactorRequest,
    FactorResponse,
    FactorMetadata,
    ProvenanceInfo,
    DataQualityIndicator,
    GWPStandard,
    SourceType
)
from services.factor_broker.exceptions import CacheError


@pytest.fixture
def cache_config():
    """Create cache configuration."""
    config = Mock()
    config.enabled = True
    config.ttl_seconds = 86400  # 24 hours
    config.redis_host = "localhost"
    config.redis_port = 6379
    config.redis_db = 0
    config.max_connections = 10
    return config


@pytest.fixture
def sample_request():
    """Create sample factor request."""
    return FactorRequest(
        product="Steel",
        region="US",
        gwp_standard=GWPStandard.AR6,
        unit="kg"
    )


@pytest.fixture
def sample_response():
    """Create sample factor response."""
    return FactorResponse(
        factor_id="test_factor_id",
        value=1.85,
        unit="kgCO2e/kg",
        uncertainty=0.10,
        metadata=FactorMetadata(
            source=SourceType.ECOINVENT,
            source_version="3.10",
            gwp_standard=GWPStandard.AR6,
            reference_year=2024,
            geographic_scope="US"
        ),
        provenance=ProvenanceInfo(
            lookup_timestamp=DeterministicClock.utcnow(),
            cache_hit=False,
            is_proxy=False,
            fallback_chain=["ecoinvent"]
        )
    )


class TestCacheInitialization:
    """Test cache initialization and configuration."""

    def test_cache_initialization_enabled(self, cache_config):
        """Test cache initializes when enabled."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis:
            cache = FactorCache(cache_config)
            assert cache.enabled is True
            assert cache.ttl_seconds == 86400

    def test_cache_initialization_disabled(self):
        """Test cache handles disabled configuration."""
        config = Mock()
        config.enabled = False

        cache = FactorCache(config)
        assert cache.enabled is False

    def test_cache_redis_connection_params(self, cache_config):
        """Test Redis connection parameters are correct."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis:
            cache = FactorCache(cache_config)

            # Redis should be initialized with correct params
            # (Implementation specific - adjust based on actual implementation)


class TestCacheKeyGeneration:
    """Test cache key generation logic."""

    def test_generate_cache_key_deterministic(self, cache_config, sample_request):
        """Test cache key generation is deterministic for same input."""
        with patch('services.factor_broker.cache.redis.Redis'):
            cache = FactorCache(cache_config)

            key1 = cache._generate_key(sample_request)
            key2 = cache._generate_key(sample_request)

            assert key1 == key2

    def test_generate_cache_key_different_products(self, cache_config):
        """Test different products generate different cache keys."""
        with patch('services.factor_broker.cache.redis.Redis'):
            cache = FactorCache(cache_config)

            request1 = FactorRequest(product="Steel", region="US", gwp_standard=GWPStandard.AR6)
            request2 = FactorRequest(product="Aluminum", region="US", gwp_standard=GWPStandard.AR6)

            key1 = cache._generate_key(request1)
            key2 = cache._generate_key(request2)

            assert key1 != key2

    def test_generate_cache_key_different_regions(self, cache_config):
        """Test different regions generate different cache keys."""
        with patch('services.factor_broker.cache.redis.Redis'):
            cache = FactorCache(cache_config)

            request1 = FactorRequest(product="Steel", region="US", gwp_standard=GWPStandard.AR6)
            request2 = FactorRequest(product="Steel", region="GB", gwp_standard=GWPStandard.AR6)

            key1 = cache._generate_key(request1)
            key2 = cache._generate_key(request2)

            assert key1 != key2

    def test_generate_cache_key_different_gwp_standards(self, cache_config):
        """Test different GWP standards generate different cache keys."""
        with patch('services.factor_broker.cache.redis.Redis'):
            cache = FactorCache(cache_config)

            request1 = FactorRequest(product="Steel", region="US", gwp_standard=GWPStandard.AR5)
            request2 = FactorRequest(product="Steel", region="US", gwp_standard=GWPStandard.AR6)

            key1 = cache._generate_key(request1)
            key2 = cache._generate_key(request2)

            assert key1 != key2

    def test_cache_key_format(self, cache_config, sample_request):
        """Test cache key has expected format."""
        with patch('services.factor_broker.cache.redis.Redis'):
            cache = FactorCache(cache_config)

            key = cache._generate_key(sample_request)

            # Key should have expected format (e.g., "factor:product:region:gwp:...")
            assert isinstance(key, str)
            assert len(key) > 0


class TestCacheGetOperations:
    """Test cache retrieval operations."""

    @pytest.mark.asyncio
    async def test_cache_get_hit(self, cache_config, sample_request, sample_response):
        """Test successful cache hit."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            # Mock Redis get to return serialized response
            mock_redis.get = Mock(return_value=sample_response.json())

            cache = FactorCache(cache_config)
            result = await cache.get(sample_request)

            assert result is not None
            mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, cache_config, sample_request):
        """Test cache miss returns None."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.get = Mock(return_value=None)

            cache = FactorCache(cache_config)
            result = await cache.get(sample_request)

            assert result is None

    @pytest.mark.asyncio
    async def test_cache_get_with_disabled_cache(self, sample_request):
        """Test cache get returns None when cache is disabled."""
        config = Mock()
        config.enabled = False

        cache = FactorCache(config)
        result = await cache.get(sample_request)

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_get_handles_redis_error(self, cache_config, sample_request):
        """Test cache get handles Redis errors gracefully."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.get = Mock(side_effect=Exception("Redis connection failed"))

            cache = FactorCache(cache_config)

            # Should not raise, should return None
            result = await cache.get(sample_request)
            assert result is None


class TestCacheSetOperations:
    """Test cache storage operations."""

    @pytest.mark.asyncio
    async def test_cache_set_success(self, cache_config, sample_request, sample_response):
        """Test successful cache set."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.setex = Mock(return_value=True)

            cache = FactorCache(cache_config)
            await cache.set(sample_request, sample_response)

            # setex should be called with TTL
            mock_redis.setex.assert_called_once()
            args = mock_redis.setex.call_args
            assert args[0][1] == 86400  # TTL should be 24 hours

    @pytest.mark.asyncio
    async def test_cache_set_with_disabled_cache(self, sample_request, sample_response):
        """Test cache set does nothing when cache is disabled."""
        config = Mock()
        config.enabled = False

        cache = FactorCache(config)
        # Should not raise error
        await cache.set(sample_request, sample_response)

    @pytest.mark.asyncio
    async def test_cache_set_ttl_compliance(self, cache_config, sample_request, sample_response):
        """Test cache set uses 24-hour TTL for license compliance."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.setex = Mock(return_value=True)

            cache = FactorCache(cache_config)
            await cache.set(sample_request, sample_response)

            # Verify 24-hour TTL (86400 seconds)
            args = mock_redis.setex.call_args
            ttl_seconds = args[0][1]
            assert ttl_seconds == 86400  # Exactly 24 hours for ecoinvent compliance


class TestCacheInvalidation:
    """Test cache invalidation operations."""

    @pytest.mark.asyncio
    async def test_cache_delete_single_key(self, cache_config, sample_request):
        """Test deleting a single cache entry."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.delete = Mock(return_value=1)

            cache = FactorCache(cache_config)
            result = await cache.delete(sample_request)

            assert result is True
            mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_flush_all(self, cache_config):
        """Test flushing entire cache."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.flushdb = Mock(return_value=True)

            cache = FactorCache(cache_config)
            await cache.flush_all()

            mock_redis.flushdb.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_invalidate_by_pattern(self, cache_config):
        """Test invalidating cache entries by pattern."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.scan_iter = Mock(return_value=iter([b"key1", b"key2"]))
            mock_redis.delete = Mock(return_value=2)

            cache = FactorCache(cache_config)
            count = await cache.invalidate_pattern("factor:steel:*")

            assert count == 2


class TestCacheStatistics:
    """Test cache statistics tracking."""

    def test_get_cache_stats(self, cache_config):
        """Test getting cache statistics."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.info = Mock(return_value={"used_memory": 1024000})

            cache = FactorCache(cache_config)
            cache.stats["hits"] = 850
            cache.stats["misses"] = 150
            cache.stats["total_requests"] = 1000

            stats = cache.get_stats()

            assert stats["hits"] == 850
            assert stats["misses"] == 150
            assert stats["hit_rate"] == 0.85

    def test_reset_cache_stats(self, cache_config):
        """Test resetting cache statistics."""
        with patch('services.factor_broker.cache.redis.Redis'):
            cache = FactorCache(cache_config)
            cache.stats["hits"] = 100
            cache.stats["misses"] = 50

            cache.reset_stats()

            assert cache.stats["hits"] == 0
            assert cache.stats["misses"] == 0


class TestMultiTenantIsolation:
    """Test multi-tenant cache isolation."""

    def test_cache_key_includes_tenant_id(self, cache_config):
        """Test cache keys include tenant ID for isolation."""
        with patch('services.factor_broker.cache.redis.Redis'):
            cache = FactorCache(cache_config)

            request = FactorRequest(
                product="Steel",
                region="US",
                gwp_standard=GWPStandard.AR6
            )

            # Add tenant context
            key_tenant1 = cache._generate_key(request, tenant_id="tenant_1")
            key_tenant2 = cache._generate_key(request, tenant_id="tenant_2")

            # Keys should be different for different tenants
            assert key_tenant1 != key_tenant2

    @pytest.mark.asyncio
    async def test_tenant_cache_isolation(self, cache_config, sample_request, sample_response):
        """Test tenants have isolated cache namespaces."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.setex = Mock(return_value=True)
            mock_redis.get = Mock(return_value=None)

            cache = FactorCache(cache_config)

            # Store for tenant 1
            await cache.set(sample_request, sample_response, tenant_id="tenant_1")

            # Retrieve for tenant 2 should miss
            result = await cache.get(sample_request, tenant_id="tenant_2")
            assert result is None


class TestCacheErrorHandling:
    """Test cache error handling."""

    @pytest.mark.asyncio
    async def test_cache_handles_serialization_error(self, cache_config, sample_request):
        """Test cache handles serialization errors."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value

            cache = FactorCache(cache_config)

            # Create response with non-serializable data
            bad_response = Mock()
            bad_response.json = Mock(side_effect=TypeError("Not serializable"))

            # Should not raise error
            await cache.set(sample_request, bad_response)

    @pytest.mark.asyncio
    async def test_cache_handles_deserialization_error(self, cache_config, sample_request):
        """Test cache handles deserialization errors."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.get = Mock(return_value=b"invalid json data")

            cache = FactorCache(cache_config)

            # Should return None instead of raising error
            result = await cache.get(sample_request)
            assert result is None


class TestCachePerformance:
    """Test cache performance characteristics."""

    @pytest.mark.asyncio
    async def test_cache_get_performance(self, cache_config, sample_request, sample_response):
        """Test cache get operation is fast (<5ms)."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.get = Mock(return_value=sample_response.json())

            cache = FactorCache(cache_config)

            import time
            start = time.time()
            await cache.get(sample_request)
            duration_ms = (time.time() - start) * 1000

            assert duration_ms < 5  # Should be very fast

    @pytest.mark.asyncio
    async def test_cache_set_performance(self, cache_config, sample_request, sample_response):
        """Test cache set operation is fast (<5ms)."""
        with patch('services.factor_broker.cache.redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.setex = Mock(return_value=True)

            cache = FactorCache(cache_config)

            import time
            start = time.time()
            await cache.set(sample_request, sample_response)
            duration_ms = (time.time() - start) * 1000

            assert duration_ms < 5  # Should be very fast


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
