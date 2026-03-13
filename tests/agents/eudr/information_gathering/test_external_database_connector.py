# -*- coding: utf-8 -*-
"""
Unit tests for ExternalDatabaseConnectorEngine - AGENT-EUDR-027

Tests engine initialization, single-source queries, batch queries,
circuit breaker state transitions, rate limiter token acquisition,
query cache operations, source availability, and source status reporting.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 (Engine 1: External Database Connector)
"""
from __future__ import annotations

import time

import pytest

from greenlang.agents.eudr.information_gathering.config import InformationGatheringConfig
from greenlang.agents.eudr.information_gathering.external_database_connector import (
    CircuitBreaker,
    CircuitState,
    ExternalDatabaseConnectorEngine,
    QueryCache,
    TokenBucketRateLimiter,
)
from greenlang.agents.eudr.information_gathering.models import (
    ExternalDatabaseSource,
    QueryStatus,
)


# ---------------------------------------------------------------------------
# Engine Tests
# ---------------------------------------------------------------------------


class TestExternalDatabaseConnectorEngineInit:
    """Test engine initialization."""

    def test_engine_initialization(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        sources = engine.get_available_sources()
        # 9 enabled adapters (national_customs and national_land_registry disabled)
        assert len(sources) == 9

    def test_engine_with_default_config(self):
        engine = ExternalDatabaseConnectorEngine()
        assert len(engine.get_available_sources()) == 9


class TestQuerySource:
    """Test single-source query operations."""

    @pytest.mark.asyncio
    async def test_query_source_success(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        result = await engine.query_source(
            ExternalDatabaseSource.EU_TRACES,
            {"certificate_number": "TRACES-2026-001"},
        )
        assert result.status == QueryStatus.SUCCESS
        assert result.record_count >= 1
        assert result.source == ExternalDatabaseSource.EU_TRACES
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_query_source_unknown_source(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        with pytest.raises(ValueError, match="No adapter registered"):
            await engine.query_source(
                ExternalDatabaseSource.NATIONAL_CUSTOMS,
                {"country_code": "DE"},
            )

    @pytest.mark.asyncio
    async def test_query_source_cites(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        result = await engine.query_source(
            ExternalDatabaseSource.CITES,
            {"taxon": "Swietenia macrophylla", "country_origin": "BR"},
        )
        assert result.status == QueryStatus.SUCCESS
        assert result.record_count >= 1

    @pytest.mark.asyncio
    async def test_query_source_comtrade(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        result = await engine.query_source(
            ExternalDatabaseSource.UN_COMTRADE,
            {"hs_code": "1801", "reporter_country": "DE"},
        )
        assert result.status == QueryStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_query_source_returns_cached(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        params = {"certificate_number": "TRACES-2026-002"}
        result1 = await engine.query_source(ExternalDatabaseSource.EU_TRACES, params)
        result2 = await engine.query_source(ExternalDatabaseSource.EU_TRACES, params)
        assert result2.cached is True

    @pytest.mark.asyncio
    async def test_query_source_provenance_hash_populated(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        result = await engine.query_source(
            ExternalDatabaseSource.FAO_STAT,
            {"country_code": "BRA", "item_code": "656"},
        )
        assert len(result.provenance_hash) == 64


class TestBatchQuery:
    """Test batch query across multiple sources."""

    @pytest.mark.asyncio
    async def test_batch_query(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        sources = [
            ExternalDatabaseSource.EU_TRACES,
            ExternalDatabaseSource.CITES,
            ExternalDatabaseSource.UN_COMTRADE,
        ]
        results = await engine.batch_query(
            sources,
            {"country_origin": "BR"},
        )
        assert len(results) == 3
        for result in results:
            assert result.source in sources

    @pytest.mark.asyncio
    async def test_batch_query_empty_list(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        results = await engine.batch_query([], {"country_origin": "DE"})
        assert results == []


# ---------------------------------------------------------------------------
# Circuit Breaker Tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Test circuit breaker state machine."""

    def test_circuit_breaker_closed_state(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_circuit_breaker_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_circuit_breaker_resets_after_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout_seconds=0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # With reset_timeout=0, it should transition to HALF_OPEN immediately
        time.sleep(0.01)
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_half_open_success_closes(self):
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout_seconds=0, half_open_max_calls=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.01)
        cb.allow_request()  # Transitions to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=2, reset_timeout_seconds=0)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.01)
        cb.allow_request()  # Transitions to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_closed_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2
        cb.record_success()
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# Rate Limiter Tests
# ---------------------------------------------------------------------------


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter."""

    def test_rate_limiter_allows_within_limit(self):
        rl = TokenBucketRateLimiter(rate=10.0, capacity=10)
        # Should allow up to capacity tokens
        for _ in range(10):
            assert rl.acquire() is True

    def test_rate_limiter_denies_over_capacity(self):
        rl = TokenBucketRateLimiter(rate=0.0, capacity=2)
        assert rl.acquire() is True
        assert rl.acquire() is True
        assert rl.acquire() is False

    def test_rate_limiter_refills(self):
        rl = TokenBucketRateLimiter(rate=1000.0, capacity=10)
        for _ in range(10):
            rl.acquire()
        time.sleep(0.02)  # Allow refill
        assert rl.acquire() is True


# ---------------------------------------------------------------------------
# Query Cache Tests
# ---------------------------------------------------------------------------


class TestQueryCache:
    """Test in-memory query cache."""

    def test_query_cache_set_get(self):
        cache = QueryCache(default_ttl_seconds=3600)
        cache.put("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_query_cache_miss(self):
        cache = QueryCache(default_ttl_seconds=3600)
        assert cache.get("nonexistent") is None

    def test_query_cache_expiry(self):
        cache = QueryCache(default_ttl_seconds=3600)
        # Use a very short TTL and wait for it to expire
        cache.put("key1", "value", ttl_seconds=0)
        # On Windows time.sleep may not be precise; use a longer wait
        time.sleep(0.05)
        assert cache.get("key1") is None

    def test_cache_hit_ratio(self):
        cache = QueryCache()
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.hit_ratio == pytest.approx(0.5)

    def test_cache_hit_ratio_empty(self):
        cache = QueryCache()
        assert cache.hit_ratio == 0.0

    def test_cache_clear(self):
        cache = QueryCache()
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.hit_ratio == 0.0


# ---------------------------------------------------------------------------
# Engine Status Tests
# ---------------------------------------------------------------------------


class TestEngineStatus:
    """Test engine status and cache reporting."""

    def test_get_available_sources(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        sources = engine.get_available_sources()
        assert ExternalDatabaseSource.EU_TRACES in sources
        assert ExternalDatabaseSource.NATIONAL_CUSTOMS not in sources

    def test_get_source_status(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        status = engine.get_source_status(ExternalDatabaseSource.EU_TRACES)
        assert status["adapter_registered"] is True
        assert status["enabled"] is True
        assert status["circuit_state"] == "closed"
        assert status["failure_count"] == 0
        assert status["rate_limiter_available"] is True

    def test_get_source_status_disabled(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        status = engine.get_source_status(ExternalDatabaseSource.NATIONAL_CUSTOMS)
        assert status["adapter_registered"] is False

    def test_clear_cache(self, config):
        engine = ExternalDatabaseConnectorEngine(config)
        engine.clear_cache()
        stats = engine.get_cache_stats()
        assert stats["total_entries"] == 0
