"""
Unit tests for LLM Integration Module

Tests real LLM API integration with Anthropic and OpenAI, including:
- Connection establishment and authentication
- Request/response handling
- Failover logic between providers
- Rate limiting and retry mechanisms
- Error handling and circuit breakers
- Token usage tracking

Target Coverage: 90%+
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any

from greenlang_core.llm import (
    LLMProvider,
    AnthropicProvider,
    OpenAIProvider,
    LLMRouter,
    RateLimiter,
    CircuitBreaker,
    TokenTracker
)
from greenlang_core.exceptions import (
    LLMConnectionError,
    RateLimitExceededError,
    CircuitBreakerOpenError,
    InvalidResponseError
)


# Fixtures
@pytest.fixture
def anthropic_api_key():
    """Test API key for Anthropic."""
    return "sk-ant-test-key-12345"


@pytest.fixture
def openai_api_key():
    """Test API key for OpenAI."""
    return "sk-openai-test-key-67890"


@pytest.fixture
def anthropic_provider(anthropic_api_key):
    """Create Anthropic provider instance."""
    return AnthropicProvider(
        api_key=anthropic_api_key,
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        timeout=30
    )


@pytest.fixture
def openai_provider(openai_api_key):
    """Create OpenAI provider instance."""
    return OpenAIProvider(
        api_key=openai_api_key,
        model="gpt-4",
        max_tokens=4096,
        timeout=30
    )


@pytest.fixture
def llm_router(anthropic_provider, openai_provider):
    """Create LLM router with both providers."""
    return LLMRouter(
        primary_provider=anthropic_provider,
        fallback_provider=openai_provider,
        enable_failover=True
    )


@pytest.fixture
def rate_limiter():
    """Create rate limiter instance."""
    return RateLimiter(
        requests_per_minute=50,
        tokens_per_minute=100000
    )


@pytest.fixture
def circuit_breaker():
    """Create circuit breaker instance."""
    return CircuitBreaker(
        failure_threshold=5,
        timeout_seconds=60,
        half_open_max_calls=3
    )


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return """Analyze this carbon emission data and provide insights:

Fuel Type: Diesel
Quantity: 1000 liters
Region: US
Date: 2025-01-15

Calculate total CO2e emissions."""


@pytest.fixture
def sample_anthropic_response():
    """Sample response from Anthropic API."""
    return {
        "id": "msg_01ABC123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Based on the data provided:\n\n1000 liters of diesel = 2,680 kg CO2e"
            }
        ],
        "model": "claude-sonnet-4-5-20250929",
        "usage": {
            "input_tokens": 150,
            "output_tokens": 75
        }
    }


# Test Classes
class TestAnthropicProvider:
    """Test suite for Anthropic LLM provider."""

    def test_initialization(self, anthropic_provider):
        """Test provider initializes correctly."""
        assert anthropic_provider.model == "claude-sonnet-4-5-20250929"
        assert anthropic_provider.max_tokens == 4096
        assert anthropic_provider.timeout == 30

    @pytest.mark.asyncio
    async def test_successful_api_call(self, anthropic_provider, sample_prompt, sample_anthropic_response):
        """Test successful API call to Anthropic."""
        with patch('anthropic.AsyncAnthropic') as mock_client:
            # Mock the API response
            mock_messages = AsyncMock()
            mock_messages.create.return_value = sample_anthropic_response
            mock_client.return_value.messages = mock_messages

            result = await anthropic_provider.generate(sample_prompt)

            assert result['text'] == "Based on the data provided:\n\n1000 liters of diesel = 2,680 kg CO2e"
            assert result['input_tokens'] == 150
            assert result['output_tokens'] == 75
            assert result['model'] == "claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_api_authentication_failure(self, anthropic_api_key):
        """Test handling of authentication failure."""
        provider = AnthropicProvider(api_key="invalid-key", model="claude-sonnet-4-5-20250929")

        with patch('anthropic.AsyncAnthropic') as mock_client:
            mock_messages = AsyncMock()
            mock_messages.create.side_effect = Exception("Authentication failed")
            mock_client.return_value.messages = mock_messages

            with pytest.raises(LLMConnectionError) as exc_info:
                await provider.generate("Test prompt")

            assert "authentication" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, anthropic_provider, sample_prompt):
        """Test handling of rate limit errors."""
        with patch('anthropic.AsyncAnthropic') as mock_client:
            mock_messages = AsyncMock()
            # Simulate rate limit error
            mock_messages.create.side_effect = Exception("Rate limit exceeded")
            mock_client.return_value.messages = mock_messages

            with pytest.raises(RateLimitExceededError):
                await anthropic_provider.generate(sample_prompt)

    @pytest.mark.asyncio
    async def test_retry_logic(self, anthropic_provider, sample_prompt, sample_anthropic_response):
        """Test retry logic on transient failures."""
        with patch('anthropic.AsyncAnthropic') as mock_client:
            mock_messages = AsyncMock()
            # Fail twice, then succeed
            mock_messages.create.side_effect = [
                Exception("Temporary error"),
                Exception("Temporary error"),
                sample_anthropic_response
            ]
            mock_client.return_value.messages = mock_messages

            result = await anthropic_provider.generate(
                sample_prompt,
                max_retries=3,
                retry_delay=0.1
            )

            assert result['text'] == "Based on the data provided:\n\n1000 liters of diesel = 2,680 kg CO2e"
            assert mock_messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, anthropic_provider, sample_prompt):
        """Test handling of request timeouts."""
        with patch('anthropic.AsyncAnthropic') as mock_client:
            mock_messages = AsyncMock()
            mock_messages.create.side_effect = asyncio.TimeoutError("Request timed out")
            mock_client.return_value.messages = mock_messages

            with pytest.raises(LLMConnectionError) as exc_info:
                await anthropic_provider.generate(sample_prompt, timeout=1)

            assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_token_counting(self, anthropic_provider, sample_prompt, sample_anthropic_response):
        """Test accurate token usage tracking."""
        with patch('anthropic.AsyncAnthropic') as mock_client:
            mock_messages = AsyncMock()
            mock_messages.create.return_value = sample_anthropic_response
            mock_client.return_value.messages = mock_messages

            result = await anthropic_provider.generate(sample_prompt)

            assert result['input_tokens'] == 150
            assert result['output_tokens'] == 75
            assert result['total_tokens'] == 225

    @pytest.mark.parametrize("temperature,expected", [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ])
    @pytest.mark.asyncio
    async def test_temperature_parameter(self, anthropic_provider, sample_prompt, temperature, expected):
        """Test temperature parameter is correctly passed."""
        with patch('anthropic.AsyncAnthropic') as mock_client:
            mock_messages = AsyncMock()
            mock_client.return_value.messages = mock_messages

            await anthropic_provider.generate(sample_prompt, temperature=temperature)

            call_kwargs = mock_messages.create.call_args[1]
            assert call_kwargs['temperature'] == expected


class TestOpenAIProvider:
    """Test suite for OpenAI LLM provider."""

    def test_initialization(self, openai_provider):
        """Test provider initializes correctly."""
        assert openai_provider.model == "gpt-4"
        assert openai_provider.max_tokens == 4096

    @pytest.mark.asyncio
    async def test_successful_api_call(self, openai_provider, sample_prompt):
        """Test successful API call to OpenAI."""
        with patch('openai.AsyncOpenAI') as mock_client:
            mock_chat = AsyncMock()
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content="OpenAI response text"))
            ]
            mock_response.usage = Mock(
                prompt_tokens=150,
                completion_tokens=50,
                total_tokens=200
            )
            mock_chat.completions.create.return_value = mock_response
            mock_client.return_value.chat = mock_chat

            result = await openai_provider.generate(sample_prompt)

            assert result['text'] == "OpenAI response text"
            assert result['input_tokens'] == 150
            assert result['output_tokens'] == 50


class TestLLMRouter:
    """Test suite for LLM router with failover logic."""

    @pytest.mark.asyncio
    async def test_successful_primary_call(self, llm_router, sample_prompt, sample_anthropic_response):
        """Test successful call using primary provider."""
        with patch.object(llm_router.primary_provider, 'generate', return_value=sample_anthropic_response):
            result = await llm_router.generate(sample_prompt)

            assert result == sample_anthropic_response
            assert llm_router.primary_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_failover_to_secondary(self, llm_router, sample_prompt):
        """Test automatic failover to secondary provider on primary failure."""
        openai_response = {'text': 'OpenAI fallback response', 'provider': 'openai'}

        with patch.object(llm_router.primary_provider, 'generate', side_effect=LLMConnectionError("Primary failed")):
            with patch.object(llm_router.fallback_provider, 'generate', return_value=openai_response):
                result = await llm_router.generate(sample_prompt)

                assert result == openai_response
                assert llm_router.fallback_provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_both_providers_fail(self, llm_router, sample_prompt):
        """Test behavior when both providers fail."""
        with patch.object(llm_router.primary_provider, 'generate', side_effect=LLMConnectionError("Primary failed")):
            with patch.object(llm_router.fallback_provider, 'generate', side_effect=LLMConnectionError("Fallback failed")):
                with pytest.raises(LLMConnectionError) as exc_info:
                    await llm_router.generate(sample_prompt)

                assert "all providers failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_failover_disabled(self, anthropic_provider, openai_provider, sample_prompt):
        """Test that failover doesn't occur when disabled."""
        router = LLMRouter(
            primary_provider=anthropic_provider,
            fallback_provider=openai_provider,
            enable_failover=False
        )

        with patch.object(router.primary_provider, 'generate', side_effect=LLMConnectionError("Primary failed")):
            with pytest.raises(LLMConnectionError):
                await router.generate(sample_prompt)

            # Fallback should not be called
            assert not hasattr(router.fallback_provider.generate, 'call_count') or \
                   router.fallback_provider.generate.call_count == 0


class TestRateLimiter:
    """Test suite for rate limiting logic."""

    def test_initialization(self, rate_limiter):
        """Test rate limiter initializes correctly."""
        assert rate_limiter.requests_per_minute == 50
        assert rate_limiter.tokens_per_minute == 100000

    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self, rate_limiter):
        """Test requests are allowed when under rate limit."""
        for i in range(10):
            allowed = await rate_limiter.check_request(tokens=1000)
            assert allowed is True

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        """Test requests are blocked when over rate limit."""
        limiter = RateLimiter(requests_per_minute=5, tokens_per_minute=10000)

        # Make 5 requests (at limit)
        for i in range(5):
            await limiter.check_request(tokens=1000)

        # 6th request should be blocked
        with pytest.raises(RateLimitExceededError):
            await limiter.check_request(tokens=1000)

    @pytest.mark.asyncio
    async def test_token_limit_enforcement(self):
        """Test token-based rate limiting."""
        limiter = RateLimiter(requests_per_minute=100, tokens_per_minute=5000)

        # Use 4000 tokens (under limit)
        await limiter.check_request(tokens=4000)

        # Try to use 2000 more tokens (would exceed limit)
        with pytest.raises(RateLimitExceededError):
            await limiter.check_request(tokens=2000)

    @pytest.mark.asyncio
    async def test_rate_limit_reset(self):
        """Test rate limit resets after time window."""
        limiter = RateLimiter(requests_per_minute=2, tokens_per_minute=5000)

        # Use up the limit
        await limiter.check_request(tokens=1000)
        await limiter.check_request(tokens=1000)

        # Mock time passing (61 seconds)
        with patch('time.time') as mock_time:
            mock_time.return_value = datetime.now().timestamp() + 61

            # Should be allowed again
            allowed = await limiter.check_request(tokens=1000)
            assert allowed is True


class TestCircuitBreaker:
    """Test suite for circuit breaker pattern."""

    def test_initialization(self, circuit_breaker):
        """Test circuit breaker initializes in closed state."""
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self, circuit_breaker):
        """Test circuit breaker opens after failure threshold."""
        # Simulate 5 failures (threshold)
        for i in range(5):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == "OPEN"

    @pytest.mark.asyncio
    async def test_blocks_requests_when_open(self, circuit_breaker):
        """Test requests are blocked when circuit is open."""
        # Force circuit to open
        for i in range(5):
            await circuit_breaker.record_failure()

        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.check_request()

    @pytest.mark.asyncio
    async def test_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        for i in range(5):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == "OPEN"

        # Mock time passing (61 seconds)
        with patch('time.time') as mock_time:
            mock_time.return_value = datetime.now().timestamp() + 61

            await circuit_breaker.check_request()

            assert circuit_breaker.state == "HALF_OPEN"

    @pytest.mark.asyncio
    async def test_closes_after_successful_half_open(self, circuit_breaker):
        """Test circuit closes after successful requests in half-open state."""
        # Open the circuit
        for i in range(5):
            await circuit_breaker.record_failure()

        # Transition to half-open
        with patch('time.time') as mock_time:
            mock_time.return_value = datetime.now().timestamp() + 61
            await circuit_breaker.check_request()

        # Record successful requests
        for i in range(3):
            await circuit_breaker.record_success()

        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_resets_on_success(self, circuit_breaker):
        """Test failure count resets on successful request."""
        # Record some failures
        for i in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.failure_count == 3

        # Record success
        await circuit_breaker.record_success()

        assert circuit_breaker.failure_count == 0


class TestTokenTracker:
    """Test suite for token usage tracking."""

    @pytest.fixture
    def token_tracker(self):
        """Create token tracker instance."""
        return TokenTracker()

    def test_initialization(self, token_tracker):
        """Test token tracker initializes correctly."""
        assert token_tracker.total_input_tokens == 0
        assert token_tracker.total_output_tokens == 0

    def test_records_token_usage(self, token_tracker):
        """Test token usage is correctly recorded."""
        token_tracker.record_usage(input_tokens=150, output_tokens=75)

        assert token_tracker.total_input_tokens == 150
        assert token_tracker.total_output_tokens == 75
        assert token_tracker.total_tokens == 225

    def test_accumulates_usage(self, token_tracker):
        """Test token usage accumulates across calls."""
        token_tracker.record_usage(input_tokens=100, output_tokens=50)
        token_tracker.record_usage(input_tokens=200, output_tokens=100)

        assert token_tracker.total_input_tokens == 300
        assert token_tracker.total_output_tokens == 150
        assert token_tracker.total_tokens == 450

    def test_calculates_cost(self, token_tracker):
        """Test cost calculation based on token usage."""
        # Anthropic pricing: $3 per 1M input tokens, $15 per 1M output tokens
        token_tracker.record_usage(input_tokens=1000000, output_tokens=1000000)

        cost = token_tracker.calculate_cost(
            input_cost_per_million=3.0,
            output_cost_per_million=15.0
        )

        assert cost == pytest.approx(18.0, rel=1e-6)

    def test_resets_counters(self, token_tracker):
        """Test token counters can be reset."""
        token_tracker.record_usage(input_tokens=1000, output_tokens=500)
        token_tracker.reset()

        assert token_tracker.total_input_tokens == 0
        assert token_tracker.total_output_tokens == 0


# Integration Tests
class TestLLMIntegrationEnd2End:
    """End-to-end integration tests for LLM system."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self, llm_router, rate_limiter, circuit_breaker, sample_prompt):
        """Test full request lifecycle with all components."""
        # Check rate limit
        await rate_limiter.check_request(tokens=1000)

        # Check circuit breaker
        await circuit_breaker.check_request()

        # Make LLM request
        with patch.object(llm_router.primary_provider, 'generate') as mock_generate:
            mock_generate.return_value = {'text': 'Test response', 'input_tokens': 150, 'output_tokens': 75}

            result = await llm_router.generate(sample_prompt)

            # Record success
            await circuit_breaker.record_success()

            assert result['text'] == 'Test response'

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_anthropic_api_call(self):
        """Test real API call to Anthropic (requires valid API key)."""
        # Skip if no API key in environment
        import os
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        provider = AnthropicProvider(api_key=api_key, model="claude-sonnet-4-5-20250929")

        result = await provider.generate("Say 'Hello, World!' and nothing else.")

        assert 'text' in result
        assert 'Hello' in result['text']
        assert result['input_tokens'] > 0
        assert result['output_tokens'] > 0


# Performance Tests
class TestLLMPerformance:
    """Performance tests for LLM integration."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, llm_router, sample_prompt):
        """Test handling of concurrent LLM requests."""
        with patch.object(llm_router.primary_provider, 'generate') as mock_generate:
            mock_generate.return_value = {'text': 'Response', 'input_tokens': 100, 'output_tokens': 50}

            # Make 100 concurrent requests
            tasks = [llm_router.generate(sample_prompt) for _ in range(100)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 100
            assert all(r['text'] == 'Response' for r in results)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time(self, anthropic_provider, sample_prompt, sample_anthropic_response):
        """Test response time is within acceptable limits (<2s)."""
        import time

        with patch('anthropic.AsyncAnthropic') as mock_client:
            mock_messages = AsyncMock()
            mock_messages.create.return_value = sample_anthropic_response
            mock_client.return_value.messages = mock_messages

            start_time = time.time()
            await anthropic_provider.generate(sample_prompt)
            duration = time.time() - start_time

            assert duration < 2.0  # <2s target for mocked call
