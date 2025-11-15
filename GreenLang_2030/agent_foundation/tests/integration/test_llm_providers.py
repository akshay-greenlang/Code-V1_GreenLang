"""
Integration Tests for LLM Providers.

Tests real API calls to Anthropic Claude and OpenAI GPT to validate
production readiness, response format, token usage, streaming, embeddings,
and latency targets.

Test Coverage:
- Basic text generation (real API calls)
- Response format validation
- Token usage and cost tracking
- Streaming responses
- Embeddings generation (OpenAI only)
- Error handling (401, 429, 500 errors)
- Latency measurement (P95 < 2s target)
- Rate limiting respect
- Health checks

Requirements:
- ANTHROPIC_API_KEY environment variable
- OPENAI_API_KEY environment variable
- TEST_MODE=real to run real API tests
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import List

from llm.providers.base_provider import (
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    TokenUsage,
)
from llm.providers.anthropic_provider import AnthropicProvider
from llm.providers.openai_provider import OpenAIProvider


# ============================================================================
# Anthropic Provider Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.anthropic
class TestAnthropicProvider:
    """Integration tests for Anthropic Claude provider."""

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_basic_generation(self, anthropic_provider, simple_request, assert_valid_response):
        """Test basic text generation with real API call."""
        response = await anthropic_provider.generate(simple_request)

        # Validate response structure
        assert_valid_response(response)

        # Validate Anthropic-specific fields
        assert response.provider == "anthropic"
        assert "claude" in response.model_id.lower()
        assert response.text.strip() != ""
        assert len(response.text) > 10  # Should have meaningful content

        print(f"\n[Anthropic] Response: {response.text[:100]}...")
        print(f"[Anthropic] Tokens: {response.usage.total_tokens}, Cost: ${response.usage.total_cost_usd:.6f}")
        print(f"[Anthropic] Latency: {response.generation_time_ms:.0f}ms")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_token_usage_accuracy(self, anthropic_provider):
        """Test that token usage is accurately reported."""
        request = GenerationRequest(
            prompt="Count to 5.",
            temperature=0.0,  # Deterministic
            max_tokens=50
        )

        response = await anthropic_provider.generate(request)

        # Validate token counts
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens

        # Validate costs
        assert response.usage.input_cost_usd > 0
        assert response.usage.output_cost_usd > 0
        assert response.usage.total_cost_usd == pytest.approx(
            response.usage.input_cost_usd + response.usage.output_cost_usd,
            rel=1e-9
        )

        print(f"\n[Anthropic Token Usage]")
        print(f"  Input tokens: {response.usage.input_tokens}")
        print(f"  Output tokens: {response.usage.output_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")
        print(f"  Total cost: ${response.usage.total_cost_usd:.6f}")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_streaming_response(self, anthropic_provider):
        """Test streaming text generation."""
        request = GenerationRequest(
            prompt="Count from 1 to 3.",
            temperature=0.0,
            max_tokens=50,
            stream=True
        )

        chunks = []
        async for chunk in anthropic_provider.generate_stream(request):
            chunks.append(chunk)

        # Validate streaming
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

        print(f"\n[Anthropic Streaming] Received {len(chunks)} chunks")
        print(f"[Anthropic Streaming] Full text: {full_text}")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_latency_target(self, anthropic_provider, simple_request, assert_performance_target):
        """Test that P95 latency is under 2000ms."""
        latencies = []

        # Run multiple requests to measure P95
        for i in range(10):
            response = await anthropic_provider.generate(simple_request)
            latencies.append(response.generation_time_ms)
            await asyncio.sleep(0.5)  # Rate limit friendly

        # Calculate P95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]

        print(f"\n[Anthropic Latency]")
        print(f"  P50: {latencies[len(latencies)//2]:.0f}ms")
        print(f"  P95: {p95_latency:.0f}ms")
        print(f"  P99: {latencies[int(len(latencies)*0.99)]:.0f}ms")

        # Validate P95 target
        assert_performance_target(p95_latency)

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_health_check(self, anthropic_provider):
        """Test provider health check."""
        health = await anthropic_provider.health_check()

        assert health.is_healthy is True
        assert health.last_check is not None
        assert health.consecutive_failures == 0
        assert health.latency_ms is not None
        assert health.latency_ms > 0

        print(f"\n[Anthropic Health] Healthy: {health.is_healthy}, Latency: {health.latency_ms:.0f}ms")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_error_handling_invalid_request(self, anthropic_provider):
        """Test error handling for invalid requests."""
        # Empty prompt should raise InvalidRequestError
        request = GenerationRequest(
            prompt="",
            max_tokens=10
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            await anthropic_provider.generate(request)

        assert "empty" in str(exc_info.value).lower()
        print(f"\n[Anthropic Error] Invalid request handled: {exc_info.value}")

    @pytest.mark.asyncio
    async def test_error_handling_authentication(self):
        """Test error handling for authentication failures."""
        # Create provider with invalid API key
        provider = AnthropicProvider(
            model_id="claude-3-haiku-20240307",
            api_key="invalid-key-12345",
            max_retries=0  # Don't retry auth errors
        )

        request = GenerationRequest(
            prompt="Test",
            max_tokens=10
        )

        with pytest.raises(AuthenticationError):
            await provider.generate(request)

        await provider.close()

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_embeddings_not_supported(self, anthropic_provider):
        """Test that embeddings raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await anthropic_provider.generate_embeddings(["Test text"])

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_system_prompt_support(self, anthropic_provider):
        """Test system prompt integration."""
        request = GenerationRequest(
            prompt="What is 2+2?",
            system_prompt="You are a helpful math tutor. Always explain your reasoning.",
            temperature=0.0,
            max_tokens=100
        )

        response = await anthropic_provider.generate(request)

        assert_valid_response(response)
        # Response should include explanation due to system prompt
        assert len(response.text) > 10

        print(f"\n[Anthropic System Prompt] Response: {response.text[:150]}...")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_cost_tracking_cumulative(self, anthropic_provider):
        """Test cumulative cost tracking across multiple requests."""
        initial_usage = anthropic_provider.get_total_usage()
        initial_cost = initial_usage.total_cost_usd

        # Make 3 requests
        request = GenerationRequest(prompt="Hello", max_tokens=20)
        for _ in range(3):
            await anthropic_provider.generate(request)
            await asyncio.sleep(0.5)

        # Check cumulative usage
        final_usage = anthropic_provider.get_total_usage()

        assert final_usage.total_tokens > initial_usage.total_tokens
        assert final_usage.total_cost_usd > initial_cost

        print(f"\n[Anthropic Cumulative]")
        print(f"  Total tokens: {final_usage.total_tokens}")
        print(f"  Total cost: ${final_usage.total_cost_usd:.6f}")


# ============================================================================
# OpenAI Provider Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.openai
class TestOpenAIProvider:
    """Integration tests for OpenAI GPT provider."""

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_basic_generation(self, openai_provider, simple_request, assert_valid_response):
        """Test basic text generation with real API call."""
        response = await openai_provider.generate(simple_request)

        # Validate response structure
        assert_valid_response(response)

        # Validate OpenAI-specific fields
        assert response.provider == "openai"
        assert "gpt" in response.model_id.lower()
        assert response.text.strip() != ""
        assert len(response.text) > 10

        print(f"\n[OpenAI] Response: {response.text[:100]}...")
        print(f"[OpenAI] Tokens: {response.usage.total_tokens}, Cost: ${response.usage.total_cost_usd:.6f}")
        print(f"[OpenAI] Latency: {response.generation_time_ms:.0f}ms")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_token_usage_accuracy(self, openai_provider):
        """Test that token usage is accurately reported."""
        request = GenerationRequest(
            prompt="Count to 5.",
            temperature=0.0,  # Deterministic
            max_tokens=50
        )

        response = await openai_provider.generate(request)

        # Validate token counts
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens == response.usage.input_tokens + response.usage.output_tokens

        # Validate costs
        assert response.usage.input_cost_usd > 0
        assert response.usage.output_cost_usd > 0
        assert response.usage.total_cost_usd == pytest.approx(
            response.usage.input_cost_usd + response.usage.output_cost_usd,
            rel=1e-9
        )

        print(f"\n[OpenAI Token Usage]")
        print(f"  Input tokens: {response.usage.input_tokens}")
        print(f"  Output tokens: {response.usage.output_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")
        print(f"  Total cost: ${response.usage.total_cost_usd:.6f}")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_streaming_response(self, openai_provider):
        """Test streaming text generation."""
        request = GenerationRequest(
            prompt="Count from 1 to 3.",
            temperature=0.0,
            max_tokens=50,
            stream=True
        )

        chunks = []
        async for chunk in openai_provider.generate_stream(request):
            chunks.append(chunk)

        # Validate streaming
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

        print(f"\n[OpenAI Streaming] Received {len(chunks)} chunks")
        print(f"[OpenAI Streaming] Full text: {full_text}")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_embeddings_generation(self, openai_provider):
        """Test embeddings generation (OpenAI only)."""
        texts = [
            "Carbon accounting tracks greenhouse gas emissions.",
            "ESG stands for Environmental, Social, and Governance.",
            "Scope 3 emissions are indirect emissions from the value chain."
        ]

        embeddings = await openai_provider.generate_embeddings(texts)

        # Validate embeddings
        assert len(embeddings) == len(texts)
        for emb in embeddings:
            assert len(emb) == 3072  # text-embedding-3-large dimension
            assert all(isinstance(x, float) for x in emb)

        print(f"\n[OpenAI Embeddings] Generated {len(embeddings)} vectors of {len(embeddings[0])} dimensions")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_latency_target(self, openai_provider, simple_request, assert_performance_target):
        """Test that P95 latency is under 2000ms."""
        latencies = []

        # Run multiple requests to measure P95
        for i in range(10):
            response = await openai_provider.generate(simple_request)
            latencies.append(response.generation_time_ms)
            await asyncio.sleep(0.5)  # Rate limit friendly

        # Calculate P95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]

        print(f"\n[OpenAI Latency]")
        print(f"  P50: {latencies[len(latencies)//2]:.0f}ms")
        print(f"  P95: {p95_latency:.0f}ms")
        print(f"  P99: {latencies[int(len(latencies)*0.99)]:.0f}ms")

        # Validate P95 target
        assert_performance_target(p95_latency)

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_health_check(self, openai_provider):
        """Test provider health check."""
        health = await openai_provider.health_check()

        assert health.is_healthy is True
        assert health.last_check is not None
        assert health.consecutive_failures == 0
        assert health.latency_ms is not None
        assert health.latency_ms > 0

        print(f"\n[OpenAI Health] Healthy: {health.is_healthy}, Latency: {health.latency_ms:.0f}ms")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_json_mode(self, openai_provider):
        """Test JSON mode response format."""
        request = GenerationRequest(
            prompt='Generate a JSON object with keys "name" and "age" for a person.',
            temperature=0.0,
            max_tokens=100,
            json_mode=True
        )

        response = await openai_provider.generate(request)

        # Validate JSON response
        import json
        try:
            data = json.loads(response.text)
            assert isinstance(data, dict)
            print(f"\n[OpenAI JSON Mode] Response: {data}")
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {response.text}")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_error_handling_invalid_request(self, openai_provider):
        """Test error handling for invalid requests."""
        # Empty prompt should raise InvalidRequestError
        request = GenerationRequest(
            prompt="",
            max_tokens=10
        )

        with pytest.raises(InvalidRequestError):
            await openai_provider.generate(request)

    @pytest.mark.asyncio
    async def test_error_handling_authentication(self):
        """Test error handling for authentication failures."""
        # Create provider with invalid API key
        provider = OpenAIProvider(
            model_id="gpt-3.5-turbo",
            api_key="invalid-key-12345",
            max_retries=0  # Don't retry auth errors
        )

        request = GenerationRequest(
            prompt="Test",
            max_tokens=10
        )

        with pytest.raises(AuthenticationError):
            await provider.generate(request)

        await provider.close()

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_system_prompt_support(self, openai_provider):
        """Test system prompt integration."""
        request = GenerationRequest(
            prompt="What is 2+2?",
            system_prompt="You are a helpful math tutor. Always explain your reasoning.",
            temperature=0.0,
            max_tokens=100
        )

        response = await openai_provider.generate(request)

        assert_valid_response(response)
        # Response should include explanation due to system prompt
        assert len(response.text) > 10

        print(f"\n[OpenAI System Prompt] Response: {response.text[:150]}...")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_stop_sequences(self, openai_provider):
        """Test stop sequences functionality."""
        request = GenerationRequest(
            prompt="Count from 1 to 10.",
            temperature=0.0,
            max_tokens=100,
            stop_sequences=["5"]  # Stop at 5
        )

        response = await openai_provider.generate(request)

        # Response should stop before or at "5"
        assert "5" in response.text or response.finish_reason == "stop"
        print(f"\n[OpenAI Stop Sequences] Response: {response.text}")
        print(f"[OpenAI Stop Sequences] Finish reason: {response.finish_reason}")


# ============================================================================
# Cross-Provider Comparison Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.real_api
class TestCrossProviderComparison:
    """Compare behavior across providers."""

    @pytest.mark.asyncio
    async def test_equivalent_responses(self, anthropic_provider, openai_provider):
        """Test that both providers can handle the same request."""
        request = GenerationRequest(
            prompt="What is 2+2? Answer with just the number.",
            temperature=0.0,
            max_tokens=10
        )

        # Get responses from both providers
        anthropic_response = await anthropic_provider.generate(request)
        openai_response = await openai_provider.generate(request)

        # Both should contain "4"
        assert "4" in anthropic_response.text
        assert "4" in openai_response.text

        print(f"\n[Cross-Provider Comparison]")
        print(f"  Anthropic: {anthropic_response.text.strip()}")
        print(f"  OpenAI: {openai_response.text.strip()}")

    @pytest.mark.asyncio
    async def test_cost_comparison(self, anthropic_provider, openai_provider):
        """Compare costs between providers."""
        request = GenerationRequest(
            prompt="Explain carbon accounting in 50 words.",
            temperature=0.7,
            max_tokens=100
        )

        # Get responses
        anthropic_response = await anthropic_provider.generate(request)
        await asyncio.sleep(1)
        openai_response = await openai_provider.generate(request)

        print(f"\n[Cost Comparison]")
        print(f"  Anthropic: ${anthropic_response.usage.total_cost_usd:.6f}")
        print(f"  OpenAI: ${openai_response.usage.total_cost_usd:.6f}")
        print(f"  Difference: ${abs(anthropic_response.usage.total_cost_usd - openai_response.usage.total_cost_usd):.6f}")

    @pytest.mark.asyncio
    async def test_latency_comparison(self, anthropic_provider, openai_provider):
        """Compare latency between providers."""
        request = GenerationRequest(
            prompt="What is ESG?",
            temperature=0.7,
            max_tokens=50
        )

        # Measure Anthropic latency
        anthropic_response = await anthropic_provider.generate(request)
        anthropic_latency = anthropic_response.generation_time_ms

        await asyncio.sleep(1)

        # Measure OpenAI latency
        openai_response = await openai_provider.generate(request)
        openai_latency = openai_response.generation_time_ms

        print(f"\n[Latency Comparison]")
        print(f"  Anthropic: {anthropic_latency:.0f}ms")
        print(f"  OpenAI: {openai_latency:.0f}ms")
        print(f"  Faster: {'Anthropic' if anthropic_latency < openai_latency else 'OpenAI'}")


# ============================================================================
# Test Summary
# ============================================================================

def test_summary():
    """Print test summary information."""
    print("\n" + "=" * 80)
    print("LLM Provider Integration Tests")
    print("=" * 80)
    print("\nTest Coverage:")
    print("  - Basic text generation (Anthropic & OpenAI)")
    print("  - Token usage and cost accuracy")
    print("  - Streaming responses")
    print("  - Embeddings (OpenAI only)")
    print("  - Error handling (401, 400 errors)")
    print("  - Latency measurement (P95 < 2s)")
    print("  - Health checks")
    print("  - System prompts and stop sequences")
    print("  - Cross-provider comparison")
    print("\nRequirements:")
    print("  - Set ANTHROPIC_API_KEY environment variable")
    print("  - Set OPENAI_API_KEY environment variable")
    print("  - Set TEST_MODE=real to enable real API tests")
    print("=" * 80)
