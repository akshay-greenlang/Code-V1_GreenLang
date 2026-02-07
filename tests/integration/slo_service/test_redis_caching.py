# -*- coding: utf-8 -*-
"""
Redis Caching Integration Tests for SLO Service (OBS-005)

Tests budget caching behavior including set/get, TTL expiry,
invalidation, miss fallback, connection failure handling,
and key format verification.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from greenlang.infrastructure.slo_service.error_budget import BudgetCache
from greenlang.infrastructure.slo_service.models import (
    BudgetStatus,
    ErrorBudget,
)


@pytest.fixture
def sample_budget():
    """Create a test error budget."""
    return ErrorBudget(
        slo_id="redis-test-slo",
        total_minutes=43.2,
        consumed_minutes=10.8,
        remaining_minutes=32.4,
        remaining_percent=75.0,
        consumed_percent=25.0,
        status=BudgetStatus.WARNING,
        sli_value=99.975,
        window="30d",
    )


@pytest.mark.integration
class TestRedisCaching:
    """Redis caching integration tests (mocked Redis)."""

    def test_budget_cache_set_and_get(self, sample_budget):
        """Set and get a budget from cache."""
        mock_redis = MagicMock()
        stored = {}

        def mock_setex(key, ttl, value):
            stored[key] = value

        def mock_get(key):
            return stored.get(key)

        mock_redis.setex = mock_setex
        mock_redis.get = mock_get

        cache = BudgetCache(mock_redis, ttl_seconds=120)
        cache.set(sample_budget)
        retrieved = cache.get("redis-test-slo")

        assert retrieved is not None
        assert retrieved.slo_id == "redis-test-slo"
        assert retrieved.consumed_percent == pytest.approx(25.0)

    def test_cache_ttl_expiry(self, sample_budget):
        """Verify setex is called with correct TTL."""
        mock_redis = MagicMock()
        cache = BudgetCache(mock_redis, ttl_seconds=120)
        cache.set(sample_budget)

        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 120  # TTL

    def test_cache_invalidation_on_update(self, sample_budget):
        """Invalidation deletes the cached key."""
        mock_redis = MagicMock()
        mock_redis.delete.return_value = 1
        cache = BudgetCache(mock_redis)

        result = cache.invalidate("redis-test-slo")
        assert result is True
        mock_redis.delete.assert_called_once()

    def test_cache_miss_fallback(self):
        """Cache miss returns None."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        cache = BudgetCache(mock_redis)

        result = cache.get("nonexistent-slo")
        assert result is None

    def test_redis_connection_failure_graceful(self, sample_budget):
        """Redis failures degrade gracefully."""
        bad_redis = MagicMock()
        bad_redis.get.side_effect = ConnectionError("Redis down")
        bad_redis.setex.side_effect = ConnectionError("Redis down")
        bad_redis.delete.side_effect = ConnectionError("Redis down")

        cache = BudgetCache(bad_redis)

        assert cache.get("slo-1") is None
        assert cache.set(sample_budget) is False
        assert cache.invalidate("slo-1") is False

    def test_cache_key_format(self):
        """Cache keys use the correct prefix format."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        cache = BudgetCache(mock_redis, prefix="gl:slo:budget:")

        cache.get("my-slo-id")
        mock_redis.get.assert_called_with("gl:slo:budget:my-slo-id")

    def test_concurrent_cache_access(self, sample_budget):
        """Concurrent cache access does not cause errors."""
        import threading

        mock_redis = MagicMock()
        storage = {}

        def mock_setex(key, ttl, value):
            storage[key] = value

        def mock_get(key):
            return storage.get(key)

        mock_redis.setex = mock_setex
        mock_redis.get = mock_get

        cache = BudgetCache(mock_redis)
        errors = []

        def cache_op(idx):
            try:
                budget = ErrorBudget(
                    slo_id=f"concurrent-{idx}",
                    total_minutes=43.2,
                    consumed_minutes=idx,
                    remaining_minutes=43.2 - idx,
                    remaining_percent=100.0 - (idx / 43.2 * 100),
                    consumed_percent=idx / 43.2 * 100,
                    status=BudgetStatus.HEALTHY,
                )
                cache.set(budget)
                cache.get(f"concurrent-{idx}")
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=cache_op, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_cache_serialization(self, sample_budget):
        """Budget is correctly serialized as JSON in cache."""
        mock_redis = MagicMock()
        stored_value = None

        def capture_setex(key, ttl, value):
            nonlocal stored_value
            stored_value = value

        mock_redis.setex = capture_setex
        cache = BudgetCache(mock_redis)
        cache.set(sample_budget)

        assert stored_value is not None
        parsed = json.loads(stored_value)
        assert parsed["slo_id"] == "redis-test-slo"
        assert parsed["status"] == "warning"
