# -*- coding: utf-8 -*-
"""
Integration Tests for LLM Failover and Resilience.

Tests automatic failover scenarios, circuit breaker behavior, retry logic,
and error handling for the LLM system.

Test Coverage:
- Primary provider fails → automatic failover to secondary
- All providers fail → proper error handling
- Circuit breaker opens after N failures
- Circuit breaker recovers after timeout (half-open → closed)
- Retry logic with exponential backoff (1s, 2s, 4s, 8s)
- Rate limit handling and retry_after
- Network timeout handling
- Authentication and invalid request errors
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime

from llm.llm_router import LLMRouter, RoutingStrategy
from llm.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpenError
from llm.providers.base_provider import (
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ServiceUnavailableError,
    TokenUsage,
)


# ============================================================================
# Failover Scenario Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.failover
class TestFailoverScenarios:
    """Test automatic failover scenarios."""

    @pytest.mark.asyncio
    async def test_primary_provider_fails_failover_to_secondary(
        self, mock_anthropic_provider, mock_openai_provider
    ):
        """Test that when primary fails, router fails over to secondary."""
        # Create router
        router = LLMRouter(
            strategy=RoutingStrategy.PRIORITY,
            enable_circuit_breaker=False,  # Disable for simpler test
            max_retries=2
        )

        # Make anthropic fail
        async def failing_generate(*args, **kwargs):
            raise ProviderError("Provider unavailable", "anthropic", retryable=True)

        mock_anthropic_provider.generate.side_effect = failing_generate

        # Register providers (anthropic primary, openai secondary)
        router.register_provider("anthropic", mock_anthropic_provider, priority=1)
        router.register_provider("openai", mock_openai_provider, priority=2)

        # Make request
        request = GenerationRequest(prompt="Test", max_tokens=10)
        response = await router.generate(request)

        # Should failover to openai
        assert response.provider == "openai"

        # Check metrics
        metrics = router.get_metrics()
        assert metrics["global"]["failover_count"] > 0

        await router.close()

        print(f"\n[Failover] Primary failed → Successfully failed over to {response.provider}")
        print(f"[Failover] Failover count: {metrics['global']['failover_count']}")

    @pytest.mark.asyncio
    async def test_all_providers_fail_error_raised(
        self, mock_anthropic_provider, mock_openai_provider
    ):
        """Test that when all providers fail, proper error is raised."""
        # Create router
        router = LLMRouter(
            strategy=RoutingStrategy.PRIORITY,
            enable_circuit_breaker=False,
            max_retries=2
        )

        # Make both providers fail
        async def failing_generate(*args, **kwargs):
            raise ProviderError("Provider unavailable", "provider", retryable=True)

        mock_anthropic_provider.generate.side_effect = failing_generate
        mock_openai_provider.generate.side_effect = failing_generate

        # Register providers
        router.register_provider("anthropic", mock_anthropic_provider, priority=1)
        router.register_provider("openai", mock_openai_provider, priority=2)

        # Make request
        request = GenerationRequest(prompt="Test", max_tokens=10)

        with pytest.raises(ProviderError) as exc_info:
            await router.generate(request)

        # Error should mention all providers failed
        assert "all providers failed" in str(exc_info.value).lower()

        await router.close()

        print(f"\n[All Providers Failed] Error: {exc_info.value}")

    @pytest.mark.asyncio
    async def test_non_retryable_error_stops_retry(
        self, mock_anthropic_provider, mock_openai_provider
    ):
        """Test that non-retryable errors stop retry attempts."""
        # Create router
        router = LLMRouter(
            strategy=RoutingStrategy.PRIORITY,
            enable_circuit_breaker=False,
            max_retries=3
        )

        # Make anthropic return non-retryable error
        async def auth_error_generate(*args, **kwargs):
            raise AuthenticationError("Invalid API key", "anthropic")

        mock_anthropic_provider.generate.side_effect = auth_error_generate

        # Register providers
        router.register_provider("anthropic", mock_anthropic_provider, priority=1)
        router.register_provider("openai", mock_openai_provider, priority=2)

        # Make request
        request = GenerationRequest(prompt="Test", max_tokens=10)
        response = await router.generate(request)

        # Should immediately failover to openai (no retries on auth error)
        assert response.provider == "openai"

        await router.close()

        print(f"\n[Non-Retryable Error] Authentication error → Immediate failover to {response.provider}")


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.failover
class TestCircuitBreaker:
    """Test circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""
        # Threshold is 5 failures
        async def failing_call():
            raise Exception("Simulated failure")

        # Cause 5 failures
        for i in range(5):
            try:
                await circuit_breaker.call(failing_call)
            except Exception:
                pass

        # Circuit should be open
        assert circuit_breaker.is_open()
        assert circuit_breaker.state == CircuitState.OPEN

        print(f"\n[Circuit Breaker] Opened after 5 failures - State: {circuit_breaker.state}")

    @pytest.mark.asyncio
    async def test_circuit_rejects_requests_when_open(self, circuit_breaker):
        """Test circuit breaker rejects requests when open."""
        # Open circuit by causing failures
        async def failing_call():
            raise Exception("Simulated failure")

        for i in range(5):
            try:
                await circuit_breaker.call(failing_call)
            except Exception:
                pass

        # Try to make request when open
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await circuit_breaker.call(failing_call)

        assert circuit_breaker.is_open()
        assert exc_info.value.retry_after > 0

        print(f"\n[Circuit Breaker Open] Rejected request - Retry after: {exc_info.value.retry_after:.0f}s")

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open_after_timeout(self, circuit_breaker):
        """Test circuit breaker transitions to half-open after recovery timeout."""
        # Create circuit breaker with short timeout
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=2.0,  # 2 second timeout
            name="test-recovery"
        )

        # Open circuit
        async def failing_call():
            raise Exception("Simulated failure")

        for i in range(3):
            try:
                await breaker.call(failing_call)
            except Exception:
                pass

        assert breaker.is_open()
        print(f"\n[Circuit Recovery] Circuit opened")

        # Wait for recovery timeout
        await asyncio.sleep(2.5)

        # Next call should transition to half-open
        async def success_call():
            return "success"

        result = await breaker.call(success_call)

        # Circuit should be closed after successful call in half-open
        assert breaker.is_closed()
        assert result == "success"

        print(f"[Circuit Recovery] Circuit recovered after timeout - State: {breaker.state}")

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successful_half_open_call(self):
        """Test circuit closes after successful call in half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            name="test-close"
        )

        # Open circuit
        async def failing_call():
            raise Exception("Fail")

        for _ in range(3):
            try:
                await breaker.call(failing_call)
            except Exception:
                pass

        assert breaker.is_open()

        # Wait for recovery
        await asyncio.sleep(1.5)

        # Successful call should close circuit
        async def success_call():
            return "success"

        await breaker.call(success_call)

        assert breaker.is_closed()
        print(f"\n[Circuit Close] Circuit closed after successful recovery call")

    @pytest.mark.asyncio
    async def test_circuit_reopens_if_half_open_call_fails(self):
        """Test circuit reopens if call fails during half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            name="test-reopen"
        )

        # Open circuit
        async def failing_call():
            raise Exception("Fail")

        for _ in range(3):
            try:
                await breaker.call(failing_call)
            except Exception:
                pass

        # Wait for recovery
        await asyncio.sleep(1.5)

        # Failed call in half-open should reopen circuit
        try:
            await breaker.call(failing_call)
        except Exception:
            pass

        assert breaker.is_open()
        print(f"\n[Circuit Reopen] Circuit reopened after failed recovery attempt")


# ============================================================================
# Retry Logic Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.failover
class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Test that retries use exponential backoff (1s, 2s, 4s, 8s)."""
        from llm.providers.anthropic_provider import AnthropicProvider

        # Create provider with known backoff parameters
        provider = AnthropicProvider(
            model_id="claude-3-haiku-20240307",
            api_key="invalid-key",
            max_retries=4,
            base_delay=1.0,
            max_delay=8.0
        )

        # Mock the API call to always fail with retryable error
        with patch.object(provider.client.messages, 'create', side_effect=Exception("Server error")):
            request = GenerationRequest(prompt="Test", max_tokens=10)

            start_time = asyncio.get_event_loop().time()

            try:
                await provider.generate(request)
            except ProviderError:
                pass

            elapsed = asyncio.get_event_loop().time() - start_time

            # Total expected delay: 1 + 2 + 4 + 8 = 15 seconds (approximately)
            # Allow some tolerance for processing time
            assert elapsed >= 10, f"Expected ~15s delay, got {elapsed:.1f}s"

        await provider.close()

        print(f"\n[Exponential Backoff] Total retry time: {elapsed:.1f}s (expected ~15s)")

    @pytest.mark.asyncio
    async def test_max_retries_respected(self):
        """Test that max_retries parameter is respected."""
        from llm.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            model_id="claude-3-haiku-20240307",
            api_key="invalid-key",
            max_retries=2,  # Only 2 retries
            base_delay=0.1  # Fast for testing
        )

        call_count = 0

        async def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Server error")

        with patch.object(provider.client.messages, 'create', side_effect=count_calls):
            request = GenerationRequest(prompt="Test", max_tokens=10)

            try:
                await provider.generate(request)
            except ProviderError:
                pass

            # Should have called initial + 2 retries = 3 total
            assert call_count == 3, f"Expected 3 calls (1 initial + 2 retries), got {call_count}"

        await provider.close()

        print(f"\n[Max Retries] Made {call_count} attempts (1 initial + 2 retries)")


# ============================================================================
# Rate Limit Handling Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.failover
class TestRateLimitHandling:
    """Test rate limit error handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_respects_retry_after(self):
        """Test that rate limit errors respect retry_after header."""
        from llm.providers.anthropic_provider import AnthropicProvider
        from anthropic import RateLimitError as AnthropicRateLimitError

        provider = AnthropicProvider(
            model_id="claude-3-haiku-20240307",
            api_key="test-key",
            max_retries=2,
            base_delay=0.5
        )

        # Mock rate limit error with retry_after
        rate_limit_error = AnthropicRateLimitError("Rate limit exceeded")
        rate_limit_error.retry_after = 2.0  # 2 seconds

        call_count = 0

        async def rate_limited_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise rate_limit_error
            # Success on 3rd try
            from unittest.mock import Mock
            mock_response = Mock()
            mock_response.content = [Mock(text="Success")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_response.id = "test-id"
            mock_response.model = "claude-3-haiku-20240307"
            mock_response.stop_reason = "end_turn"
            return mock_response

        with patch.object(provider.client.messages, 'create', side_effect=rate_limited_call):
            request = GenerationRequest(prompt="Test", max_tokens=10)

            start_time = asyncio.get_event_loop().time()
            response = await provider.generate(request)
            elapsed = asyncio.get_event_loop().time() - start_time

            # Should have waited approximately 2s * 2 retries = 4s
            assert elapsed >= 3.0, f"Expected ~4s delay for rate limiting, got {elapsed:.1f}s"
            assert response.text == "Success"

        await provider.close()

        print(f"\n[Rate Limit] Waited {elapsed:.1f}s for rate limit recovery (2 retries × 2s)")

    @pytest.mark.asyncio
    async def test_rate_limit_exhausts_retries(self):
        """Test that persistent rate limiting eventually fails."""
        from llm.providers.anthropic_provider import AnthropicProvider
        from anthropic import RateLimitError as AnthropicRateLimitError

        provider = AnthropicProvider(
            model_id="claude-3-haiku-20240307",
            api_key="test-key",
            max_retries=2,
            base_delay=0.1
        )

        # Always rate limited
        rate_limit_error = AnthropicRateLimitError("Rate limit exceeded")

        with patch.object(provider.client.messages, 'create', side_effect=rate_limit_error):
            request = GenerationRequest(prompt="Test", max_tokens=10)

            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate(request)

            assert "rate limit exceeded" in str(exc_info.value).lower()

        await provider.close()

        print(f"\n[Rate Limit Exhausted] Failed after max retries: {exc_info.value}")


# ============================================================================
# Network Error Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.failover
class TestNetworkErrors:
    """Test network error handling."""

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self):
        """Test that timeout errors trigger retries."""
        from llm.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            model_id="claude-3-haiku-20240307",
            api_key="test-key",
            max_retries=2,
            base_delay=0.1,
            timeout=5.0
        )

        call_count = 0

        async def timeout_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError("Request timeout")
            # Success on 3rd try
            from unittest.mock import Mock
            mock_response = Mock()
            mock_response.content = [Mock(text="Success")]
            mock_response.usage = Mock(input_tokens=10, output_tokens=20)
            mock_response.id = "test-id"
            mock_response.model = "claude-3-haiku-20240307"
            mock_response.stop_reason = "end_turn"
            return mock_response

        with patch.object(provider.client.messages, 'create', side_effect=timeout_call):
            request = GenerationRequest(prompt="Test", max_tokens=10)
            response = await provider.generate(request)

            assert call_count == 3
            assert response.text == "Success"

        await provider.close()

        print(f"\n[Timeout Retry] Succeeded after {call_count} attempts (2 timeouts, then success)")


# ============================================================================
# Test Summary
# ============================================================================

def test_summary():
    """Print test summary information."""
    print("\n" + "=" * 80)
    print("LLM Failover and Resilience Integration Tests")
    print("=" * 80)
    print("\nTest Coverage:")
    print("  - Primary provider fails → failover to secondary")
    print("  - All providers fail → proper error handling")
    print("  - Circuit breaker opens after N failures")
    print("  - Circuit breaker recovery (half-open → closed)")
    print("  - Retry logic with exponential backoff (1s, 2s, 4s, 8s)")
    print("  - Rate limit handling and retry_after")
    print("  - Network timeout handling")
    print("  - Authentication error handling")
    print("=" * 80)
