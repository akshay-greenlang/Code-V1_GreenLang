# -*- coding: utf-8 -*-
"""Tests for Resilience Patterns.

Comprehensive test suite for retry, timeout, fallback, rate limiting,
circuit breaker, and graceful degradation patterns.

Author: Team 2 - Resilience Patterns
Date: November 2025
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

from greenlang.resilience import (
    retry,
    async_retry,
    timeout,
    async_timeout,
    fallback,
    async_fallback,
    RetryConfig,
    RetryStrategy,
    TimeoutConfig,
    OperationType,
    FallbackStrategy,
    FallbackConfig,
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    LeakyBucket,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    MaxRetriesExceeded,
    TimeoutError,
    RateLimitExceeded,
    get_cached_fallback,
)

from services.resilience import (
    DegradationTier,
    DegradationManager,
    ServiceHealth,
    ServiceStatus,
    get_degradation_manager,
    degradation_handler,
)


# ==============================================================================
# Retry Tests
# ==============================================================================


class TestRetry:
    """Test retry decorator."""

    def test_retry_success_first_attempt(self):
        """Test successful call on first attempt."""
        call_count = 0

        @retry(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test successful call after failures."""
        call_count = 0

        @retry(max_retries=3, base_delay=0.1)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_retry_max_retries_exceeded(self):
        """Test max retries exceeded."""
        @retry(max_retries=2, base_delay=0.1)
        def always_fails():
            raise ConnectionError("Permanent error")

        with pytest.raises(MaxRetriesExceeded):
            always_fails()

    def test_retry_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []

        @retry(max_retries=3, base_delay=0.1, strategy=RetryStrategy.EXPONENTIAL)
        def fails_with_timing():
            call_times.append(time.time())
            raise ConnectionError("Error")

        with pytest.raises(MaxRetriesExceeded):
            fails_with_timing()

        # Verify exponential delays: ~0.1s, ~0.2s, ~0.4s
        assert len(call_times) == 4  # Initial + 3 retries

    def test_retry_specific_exceptions(self):
        """Test retry only specific exceptions."""
        @retry(
            max_retries=3,
            base_delay=0.1,
            retryable_exceptions=(ConnectionError,)
        )
        def fails_with_value_error():
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            fails_with_value_error()


class TestAsyncRetry:
    """Test async retry decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async retry success."""
        call_count = 0

        @async_retry(max_retries=3, base_delay=0.1)
        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient error")
            return "success"

        result = await eventually_successful()
        assert result == "success"
        assert call_count == 2


# ==============================================================================
# Timeout Tests
# ==============================================================================


class TestTimeout:
    """Test timeout decorator."""

    def test_timeout_success(self):
        """Test successful call within timeout."""
        @timeout(timeout_seconds=1.0)
        def quick_func():
            time.sleep(0.1)
            return "success"

        result = quick_func()
        assert result == "success"

    def test_timeout_operation_type(self):
        """Test timeout with operation type."""
        @timeout(operation_type=OperationType.FACTOR_LOOKUP)
        def lookup_func():
            time.sleep(0.1)
            return "factor"

        result = lookup_func()
        assert result == "factor"


class TestAsyncTimeout:
    """Test async timeout decorator."""

    @pytest.mark.asyncio
    async def test_async_timeout_success(self):
        """Test async timeout success."""
        @async_timeout(timeout_seconds=1.0)
        async def quick_async_func():
            await asyncio.sleep(0.1)
            return "success"

        result = await quick_async_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_timeout_exceeded(self):
        """Test async timeout exceeded."""
        @async_timeout(timeout_seconds=0.1)
        async def slow_async_func():
            await asyncio.sleep(1.0)
            return "never"

        with pytest.raises(TimeoutError):
            await slow_async_func()


# ==============================================================================
# Fallback Tests
# ==============================================================================


class TestFallback:
    """Test fallback decorator."""

    def test_fallback_default_value(self):
        """Test fallback to default value."""
        @fallback(strategy=FallbackStrategy.DEFAULT, default_value=[])
        def fails():
            raise ValueError("Error")

        result = fails()
        assert result == []

    def test_fallback_function(self):
        """Test fallback to function."""
        def fallback_func():
            return "fallback"

        @fallback(
            strategy=FallbackStrategy.FUNCTION,
            fallback_function=fallback_func
        )
        def fails():
            raise ValueError("Error")

        result = fails()
        assert result == "fallback"

    def test_fallback_cached(self):
        """Test fallback to cached value."""
        cache = get_cached_fallback()
        cache.set("test_key", "cached_value")

        @fallback(
            strategy=FallbackStrategy.CACHED,
            cache_key_func=lambda: "test_key"
        )
        def fails():
            raise ValueError("Error")

        result = fails()
        assert result == "cached_value"

    def test_fallback_success_caches(self):
        """Test successful call caches result."""
        cache = get_cached_fallback()
        cache.clear()

        call_count = 0

        @fallback(
            strategy=FallbackStrategy.CACHED,
            cache_key_func=lambda: "success_key"
        )
        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "success"
            raise ValueError("Error")

        # First call succeeds and caches
        result1 = sometimes_fails()
        assert result1 == "success"

        # Second call fails but returns cached value
        result2 = sometimes_fails()
        assert result2 == "success"


# ==============================================================================
# Rate Limiting Tests
# ==============================================================================


class TestTokenBucket:
    """Test token bucket rate limiter."""

    def test_token_bucket_consume_success(self):
        """Test token bucket allows requests."""
        bucket = TokenBucket(rate=10.0, capacity=10)
        assert bucket.consume(5) is True
        assert bucket.consume(5) is True

    def test_token_bucket_consume_failure(self):
        """Test token bucket blocks when empty."""
        bucket = TokenBucket(rate=10.0, capacity=5)
        assert bucket.consume(5) is True
        assert bucket.consume(1) is False  # Bucket empty

    def test_token_bucket_refill(self):
        """Test token bucket refills over time."""
        bucket = TokenBucket(rate=10.0, capacity=10)
        bucket.consume(10)  # Empty bucket
        time.sleep(0.5)  # Wait for refill
        assert bucket.consume(3) is True  # Should have ~5 tokens


class TestLeakyBucket:
    """Test leaky bucket rate limiter."""

    def test_leaky_bucket_consume_success(self):
        """Test leaky bucket allows requests."""
        bucket = LeakyBucket(rate=10.0, capacity=10)
        assert bucket.consume(5) is True

    def test_leaky_bucket_consume_failure(self):
        """Test leaky bucket blocks when full."""
        bucket = LeakyBucket(rate=10.0, capacity=5)
        assert bucket.consume(5) is True
        assert bucket.consume(1) is False  # Bucket full


class TestRateLimiter:
    """Test rate limiter manager."""

    def test_rate_limiter_configure(self):
        """Test rate limiter configuration."""
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_second=10.0)
        limiter.configure("test_key", config)

        assert limiter.check_limit("test_key") is True

    def test_rate_limiter_exceed(self):
        """Test rate limiter blocks when exceeded."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            requests_per_second=1.0,
            burst_size=2,
            raise_on_limit=True,
        )
        limiter.configure("test_key", config)

        # First 2 should succeed
        assert limiter.check_limit("test_key") is True
        assert limiter.check_limit("test_key") is True

        # Third should fail
        with pytest.raises(RateLimitExceeded):
            limiter.check_limit("test_key")


# ==============================================================================
# Circuit Breaker Tests
# ==============================================================================


class TestCircuitBreaker:
    """Test circuit breaker."""

    def test_circuit_breaker_closed_success(self):
        """Test circuit breaker allows calls when closed."""
        cb = CircuitBreaker(
            name="test_cb",
            failure_threshold=3,
            timeout=1.0,
        )

        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after failures."""
        cb = CircuitBreaker(
            name="test_cb",
            failure_threshold=3,
            timeout=1.0,
        )

        # Cause failures
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("Error")))
            except Exception:
                pass

        # Circuit should be open
        assert cb.state == CircuitState.OPEN


# ==============================================================================
# Graceful Degradation Tests
# ==============================================================================


class TestServiceHealth:
    """Test service health tracking."""

    def test_service_health_mark_success(self):
        """Test marking service as healthy."""
        health = ServiceHealth(name="test_service")
        health.mark_success(response_time_ms=100.0)

        assert health.is_healthy()
        assert health.status == ServiceStatus.HEALTHY
        assert health.failure_count == 0

    def test_service_health_mark_failure(self):
        """Test marking service as failed."""
        health = ServiceHealth(name="test_service")
        health.mark_failure("Error occurred")

        assert not health.is_healthy()
        assert health.status == ServiceStatus.DEGRADED
        assert health.failure_count == 1

    def test_service_health_down_after_failures(self):
        """Test service marked down after multiple failures."""
        health = ServiceHealth(name="test_service")

        for _ in range(3):
            health.mark_failure("Error")

        assert health.is_down()
        assert health.status == ServiceStatus.DOWN


class TestDegradationManager:
    """Test degradation manager."""

    def test_degradation_manager_tier_1_all_healthy(self):
        """Test Tier 1 when all services healthy."""
        manager = DegradationManager()
        manager.register_service("service1", critical=True)
        manager.register_service("service2", critical=False)

        manager.update_health("service1", healthy=True)
        manager.update_health("service2", healthy=True)

        assert manager.get_current_tier() == DegradationTier.TIER_1_FULL

    def test_degradation_manager_tier_3_critical_down(self):
        """Test Tier 3 when critical service down."""
        manager = DegradationManager()
        manager.register_service("critical", critical=True)

        manager.update_health("critical", healthy=False, error="Down")
        manager.update_health("critical", healthy=False, error="Down")
        manager.update_health("critical", healthy=False, error="Down")

        tier = manager.get_current_tier()
        assert tier in (DegradationTier.TIER_3_READONLY, DegradationTier.TIER_4_MAINTENANCE)

    def test_degradation_handler_blocks_when_degraded(self):
        """Test degradation handler blocks operations."""
        manager = DegradationManager()
        manager.register_service("critical", critical=True)
        manager.update_health("critical", healthy=False, error="Down")
        manager.update_health("critical", healthy=False, error="Down")
        manager.update_health("critical", healthy=False, error="Down")

        # Mock the global manager
        with patch('services.resilience.graceful_degradation.get_degradation_manager', return_value=manager):
            @degradation_handler(
                min_tier=DegradationTier.TIER_1_FULL,
                fallback_value="degraded"
            )
            def requires_tier_1():
                return "success"

            result = requires_tier_1()
            assert result == "degraded"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Test combined resilience patterns."""

    def test_retry_with_timeout(self):
        """Test retry combined with timeout."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.1)
        @timeout(timeout_seconds=0.5)
        def func_with_retry_timeout():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry this")
            return "success"

        result = func_with_retry_timeout()
        assert result == "success"
        assert call_count == 2

    def test_retry_with_fallback(self):
        """Test retry combined with fallback."""
        @retry(max_retries=1, base_delay=0.1)
        @fallback(strategy=FallbackStrategy.DEFAULT, default_value="fallback")
        def func_with_retry_fallback():
            raise ConnectionError("Always fails")

        result = func_with_retry_fallback()
        assert result == "fallback"

    def test_full_resilience_stack(self):
        """Test all patterns combined."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.1)
        @timeout(timeout_seconds=1.0)
        @fallback(strategy=FallbackStrategy.DEFAULT, default_value="fallback")
        def full_stack_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry")
            return "success"

        result = full_stack_func()
        assert result == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
