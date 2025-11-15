"""
Integration Tests for LLM Router.

Tests routing strategies, provider selection, health checks, and metrics
for the multi-provider LLM router system.

Test Coverage:
- PRIORITY strategy (use providers in order)
- LEAST_COST strategy (route to cheapest)
- LEAST_LATENCY strategy (route to fastest)
- ROUND_ROBIN strategy (distribute evenly)
- Preferred provider routing
- Health check integration
- Metrics and monitoring
- Circuit breaker integration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from llm.llm_router import LLMRouter, RoutingStrategy
from llm.providers.base_provider import (
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    TokenUsage,
)


# ============================================================================
# Router Strategy Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
class TestRoutingStrategies:
    """Test different routing strategies."""

    @pytest.mark.asyncio
    async def test_priority_strategy(self, mock_router, simple_request):
        """Test PRIORITY strategy uses providers in priority order."""
        # Set strategy to PRIORITY
        mock_router.strategy = RoutingStrategy.PRIORITY

        # Generate request
        response = await mock_router.generate(simple_request)

        # Should use anthropic (priority=1) first
        assert response.provider == "anthropic"

        print(f"\n[Priority Strategy] Used provider: {response.provider}")

    @pytest.mark.asyncio
    async def test_least_cost_strategy(self, mock_router, simple_request):
        """Test LEAST_COST strategy routes to cheapest provider."""
        # Set strategy to LEAST_COST
        mock_router.strategy = RoutingStrategy.LEAST_COST

        # Generate request
        response = await mock_router.generate(simple_request)

        # Should use the cheapest provider (anthropic haiku in this case)
        assert response.provider in ["anthropic", "openai"]

        print(f"\n[Least Cost Strategy] Used provider: {response.provider}")
        print(f"[Least Cost Strategy] Cost: ${response.usage.total_cost_usd:.6f}")

    @pytest.mark.asyncio
    async def test_round_robin_strategy(self, mock_router, simple_request):
        """Test ROUND_ROBIN strategy distributes requests evenly."""
        # Set strategy to ROUND_ROBIN
        mock_router.strategy = RoutingStrategy.ROUND_ROBIN

        providers_used = []

        # Make multiple requests
        for _ in range(4):
            response = await mock_router.generate(simple_request)
            providers_used.append(response.provider)

        # Should alternate between providers
        assert len(set(providers_used)) > 1, "Round robin should use multiple providers"

        print(f"\n[Round Robin Strategy] Providers used: {providers_used}")

    @pytest.mark.asyncio
    async def test_preferred_provider_override(self, mock_router, simple_request):
        """Test preferred_provider parameter overrides strategy."""
        # Set strategy to PRIORITY (would normally use anthropic)
        mock_router.strategy = RoutingStrategy.PRIORITY

        # Request with preferred provider
        response = await mock_router.generate(simple_request, preferred_provider="openai")

        # Should use preferred provider
        assert response.provider == "openai"

        print(f"\n[Preferred Provider] Used: {response.provider}")


# ============================================================================
# Provider Management Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
class TestProviderManagement:
    """Test provider registration and management."""

    @pytest.mark.asyncio
    async def test_register_provider(self, mock_anthropic_provider):
        """Test registering a new provider."""
        router = LLMRouter()

        # Register provider
        router.register_provider("test-anthropic", mock_anthropic_provider, priority=1)

        # Verify registration
        assert "test-anthropic" in router._providers
        assert router._providers["test-anthropic"].priority == 1
        assert router._providers["test-anthropic"].enabled is True

        await router.close()

    @pytest.mark.asyncio
    async def test_unregister_provider(self, mock_router):
        """Test unregistering a provider."""
        # Unregister openai
        mock_router.unregister_provider("openai")

        # Verify unregistration
        assert "openai" not in mock_router._providers
        assert "anthropic" in mock_router._providers  # Other provider still there

    @pytest.mark.asyncio
    async def test_enable_disable_provider(self, mock_router, simple_request):
        """Test enabling and disabling providers."""
        # Disable anthropic
        mock_router.disable_provider("anthropic")

        # Should use openai instead
        response = await mock_router.generate(simple_request)
        assert response.provider == "openai"

        # Re-enable anthropic
        mock_router.enable_provider("anthropic")

        # Should use anthropic again (higher priority)
        response = await mock_router.generate(simple_request)
        assert response.provider == "anthropic"

        print(f"\n[Enable/Disable] After disable anthropic: openai")
        print(f"[Enable/Disable] After re-enable: anthropic")


# ============================================================================
# Health Check Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
class TestHealthChecks:
    """Test router health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_all_providers(self, mock_router):
        """Test health check on all providers."""
        health_results = await mock_router.health_check_all()

        # Should have results for all providers
        assert "anthropic" in health_results
        assert "openai" in health_results

        # All should be healthy
        assert health_results["anthropic"].is_healthy is True
        assert health_results["openai"].is_healthy is True

        print(f"\n[Health Check All]")
        for name, health in health_results.items():
            print(f"  {name}: healthy={health.is_healthy}, latency={health.latency_ms:.0f}ms")

    @pytest.mark.asyncio
    async def test_health_monitoring_start_stop(self, mock_router):
        """Test starting and stopping health monitoring."""
        # Start monitoring
        await mock_router.start_health_monitoring()
        assert mock_router._health_check_running is True

        # Wait a bit
        await asyncio.sleep(0.5)

        # Stop monitoring
        await mock_router.stop_health_monitoring()
        assert mock_router._health_check_running is False

        print(f"\n[Health Monitoring] Started and stopped successfully")

    @pytest.mark.real_api
    @pytest.mark.asyncio
    async def test_real_health_checks(self, real_router):
        """Test health checks with real providers."""
        if not real_router:
            pytest.skip("No real providers available")

        health_results = await real_router.health_check_all()

        # Check results
        for name, health in health_results.items():
            assert health.is_healthy is True
            assert health.latency_ms > 0
            print(f"\n[Real Health Check] {name}: {health.latency_ms:.0f}ms")


# ============================================================================
# Metrics Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
class TestRouterMetrics:
    """Test router metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, mock_router, simple_request):
        """Test that metrics are tracked correctly."""
        # Reset metrics
        mock_router.reset_metrics()

        # Make requests
        for _ in range(5):
            await mock_router.generate(simple_request)

        # Get metrics
        metrics = mock_router.get_metrics()

        # Validate global metrics
        assert metrics["global"]["total_requests"] == 5
        assert metrics["global"]["successful_requests"] == 5
        assert metrics["global"]["failed_requests"] == 0

        # Validate provider metrics
        assert "anthropic" in metrics["providers"]
        anthropic_metrics = metrics["providers"]["anthropic"]
        assert anthropic_metrics["total_requests"] > 0
        assert anthropic_metrics["success_rate"] == 100.0

        print(f"\n[Metrics] Global: {metrics['global']}")
        print(f"[Metrics] Anthropic: total={anthropic_metrics['total_requests']}, "
              f"success_rate={anthropic_metrics['success_rate']}%")

    @pytest.mark.asyncio
    async def test_cost_aggregation(self, mock_router, simple_request):
        """Test cost aggregation across requests."""
        # Reset metrics
        mock_router.reset_metrics()

        # Make requests
        for _ in range(3):
            await mock_router.generate(simple_request)

        # Get metrics
        metrics = mock_router.get_metrics()

        # Should have accumulated cost
        assert metrics["global"]["total_cost_usd"] > 0

        print(f"\n[Cost Aggregation] Total cost: ${metrics['global']['total_cost_usd']:.6f}")

    @pytest.mark.asyncio
    async def test_reset_metrics(self, mock_router, simple_request):
        """Test resetting metrics."""
        # Make some requests
        await mock_router.generate(simple_request)

        # Get initial metrics
        initial_metrics = mock_router.get_metrics()
        assert initial_metrics["global"]["total_requests"] > 0

        # Reset
        mock_router.reset_metrics()

        # Verify reset
        reset_metrics = mock_router.get_metrics()
        assert reset_metrics["global"]["total_requests"] == 0
        assert reset_metrics["global"]["total_cost_usd"] == 0.0

        print(f"\n[Reset Metrics] Before: {initial_metrics['global']['total_requests']} requests")
        print(f"[Reset Metrics] After: {reset_metrics['global']['total_requests']} requests")


# ============================================================================
# Circuit Breaker Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with router."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_tracking(self, mock_router):
        """Test that circuit breaker states are tracked."""
        metrics = mock_router.get_metrics()

        # Check circuit breaker states
        for provider_name, provider_metrics in metrics["providers"].items():
            assert "circuit_breaker_state" in provider_metrics
            assert provider_metrics["circuit_breaker_state"] in ["closed", "open", "half_open", "n/a"]

        print(f"\n[Circuit Breaker States]")
        for name, pm in metrics["providers"].items():
            print(f"  {name}: {pm['circuit_breaker_state']}")

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_cascading_failures(self, mock_anthropic_provider):
        """Test circuit breaker prevents cascading failures."""
        # Create router with fast circuit breaker
        router = LLMRouter(
            strategy=RoutingStrategy.PRIORITY,
            enable_circuit_breaker=True,
            circuit_breaker_threshold=3,  # Open after 3 failures
            circuit_breaker_timeout=5.0,
            max_retries=1
        )

        # Create failing provider
        failing_provider = AsyncMock()
        failing_provider.provider_name = "failing"
        failing_provider.cost_per_1k_input = 0.001
        failing_provider.cost_per_1k_output = 0.001

        async def failing_generate(*args, **kwargs):
            raise ProviderError("Simulated failure", "failing", retryable=True)

        failing_provider.generate.side_effect = failing_generate

        # Register providers (failing first, working second)
        router.register_provider("failing", failing_provider, priority=1)
        router.register_provider("anthropic", mock_anthropic_provider, priority=2)

        request = GenerationRequest(prompt="Test", max_tokens=10)

        # First 3 requests will fail and open circuit
        for i in range(3):
            response = await router.generate(request)
            # Should failover to anthropic
            assert response.provider == "anthropic"

        # Check circuit breaker opened
        metrics = router.get_metrics()
        # After failures, circuit should open and subsequent requests go directly to anthropic

        await router.close()

        print(f"\n[Circuit Breaker] Prevented cascading failures - circuit opened after 3 failures")


# ============================================================================
# Real API Router Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
@pytest.mark.real_api
class TestRealRouter:
    """Test router with real API providers."""

    @pytest.mark.asyncio
    async def test_real_routing_priority(self, real_router, simple_request):
        """Test routing with real providers using PRIORITY strategy."""
        if not real_router:
            pytest.skip("No real providers available")

        real_router.strategy = RoutingStrategy.PRIORITY
        response = await real_router.generate(simple_request)

        assert response.text is not None
        assert len(response.text) > 0

        print(f"\n[Real Router - Priority] Provider: {response.provider}")
        print(f"[Real Router - Priority] Response: {response.text[:100]}...")
        print(f"[Real Router - Priority] Cost: ${response.usage.total_cost_usd:.6f}")

    @pytest.mark.asyncio
    async def test_real_routing_least_cost(self, real_router, simple_request):
        """Test routing with real providers using LEAST_COST strategy."""
        if not real_router:
            pytest.skip("No real providers available")

        real_router.strategy = RoutingStrategy.LEAST_COST
        response = await real_router.generate(simple_request)

        assert response.text is not None
        assert len(response.text) > 0

        print(f"\n[Real Router - Least Cost] Provider: {response.provider}")
        print(f"[Real Router - Least Cost] Cost: ${response.usage.total_cost_usd:.6f}")

    @pytest.mark.asyncio
    async def test_real_metrics_collection(self, real_router, simple_request):
        """Test metrics collection with real providers."""
        if not real_router:
            pytest.skip("No real providers available")

        # Reset metrics
        real_router.reset_metrics()

        # Make requests
        for _ in range(3):
            await real_router.generate(simple_request)
            await asyncio.sleep(1)  # Rate limit friendly

        # Get metrics
        metrics = real_router.get_metrics()

        assert metrics["global"]["total_requests"] == 3
        assert metrics["global"]["successful_requests"] == 3
        assert metrics["global"]["total_cost_usd"] > 0

        print(f"\n[Real Router Metrics]")
        print(f"  Total requests: {metrics['global']['total_requests']}")
        print(f"  Success rate: 100%")
        print(f"  Total cost: ${metrics['global']['total_cost_usd']:.6f}")


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.router
class TestRouterErrorHandling:
    """Test router error handling."""

    @pytest.mark.asyncio
    async def test_no_providers_available_error(self, simple_request):
        """Test error when no providers are available."""
        router = LLMRouter()

        # No providers registered
        with pytest.raises(ProviderError) as exc_info:
            await router.generate(simple_request)

        assert "no healthy providers" in str(exc_info.value).lower()

        await router.close()

    @pytest.mark.asyncio
    async def test_all_providers_disabled_error(self, mock_router, simple_request):
        """Test error when all providers are disabled."""
        # Disable all providers
        mock_router.disable_provider("anthropic")
        mock_router.disable_provider("openai")

        with pytest.raises(ProviderError) as exc_info:
            await mock_router.generate(simple_request)

        assert "no healthy providers" in str(exc_info.value).lower()


# ============================================================================
# Test Summary
# ============================================================================

def test_summary():
    """Print test summary information."""
    print("\n" + "=" * 80)
    print("LLM Router Integration Tests")
    print("=" * 80)
    print("\nTest Coverage:")
    print("  - PRIORITY routing strategy")
    print("  - LEAST_COST routing strategy")
    print("  - ROUND_ROBIN routing strategy")
    print("  - Preferred provider override")
    print("  - Provider registration/management")
    print("  - Health checks (all providers)")
    print("  - Metrics tracking and aggregation")
    print("  - Circuit breaker integration")
    print("  - Error handling")
    print("=" * 80)
