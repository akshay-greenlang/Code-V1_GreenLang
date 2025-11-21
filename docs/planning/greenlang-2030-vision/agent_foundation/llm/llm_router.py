# -*- coding: utf-8 -*-
"""
LLMRouter - Multi-provider routing with automatic failover and load balancing.

This module implements intelligent routing across multiple LLM providers with
automatic failover, health monitoring, circuit breaker integration, and cost optimization.

Key Features:
- Multi-provider registration (Anthropic, OpenAI, custom providers)
- Automatic failover on provider failures (tries primary, falls back to secondary)
- Health check monitoring (every 30 seconds, configurable)
- Load balancing strategies:
  * ROUND_ROBIN: Distribute load evenly across providers
  * LEAST_COST: Route to cheapest provider for request
  * LEAST_LATENCY: Route to fastest provider
  * PRIORITY: Use providers in priority order with fallback
- Circuit breaker integration (prevents cascading failures)
- Provider metrics and monitoring
- Thread-safe implementation with async support

Architecture:
    User Request
         |
    LLMRouter (selects provider based on strategy)
         |
    +----+----+----+
    |    |    |    |
   P1   P2   P3  P4  (Circuit breakers protect each provider)
    |    |    |    |
    Claude GPT-4 etc

Example:
    >>> router = LLMRouter(strategy="least_cost")
    >>> router.register_provider("anthropic", anthropic_provider, priority=1)
    >>> router.register_provider("openai", openai_provider, priority=2)
    >>>
    >>> request = GenerationRequest(prompt="Analyze emissions data...")
    >>> response = await router.generate(request)
    >>> print(f"Used: {response.provider}, Cost: ${response.usage.total_cost_usd:.4f}")
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from .providers.base_provider import (
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_random
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderError,
    ProviderHealth,
    TokenUsage,
)

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Load balancing and routing strategies."""

    ROUND_ROBIN = "round_robin"  # Distribute evenly across healthy providers
    LEAST_COST = "least_cost"  # Route to cheapest provider
    LEAST_LATENCY = "least_latency"  # Route to fastest provider
    PRIORITY = "priority"  # Use priority order with fallback
    RANDOM = "random"  # Random selection from healthy providers


@dataclass
class ProviderMetrics:
    """Metrics for a registered provider."""

    provider_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    last_used: Optional[datetime] = None
    health_check_count: int = 0
    last_health_check: Optional[datetime] = None

    def update_request(
        self, success: bool, cost: float, tokens: int, latency_ms: float
    ) -> None:
        """Update metrics after a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_cost_usd += cost
        self.total_tokens += tokens
        self.last_used = DeterministicClock.utcnow()

        # Exponential moving average for latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = 0.8 * self.avg_latency_ms + 0.2 * latency_ms

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


@dataclass
class RegisteredProvider:
    """Wrapper for a registered provider with metadata."""

    name: str
    provider: BaseLLMProvider
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True
    circuit_breaker: Optional[CircuitBreaker] = None
    metrics: ProviderMetrics = field(default_factory=lambda: ProviderMetrics(""))

    def __post_init__(self):
        """Initialize metrics with provider name."""
        if not self.metrics.provider_name:
            self.metrics.provider_name = self.name


class LLMRouter:
    """
    Multi-provider LLM router with automatic failover and load balancing.

    The router manages multiple LLM providers and intelligently routes requests
    based on the configured strategy, provider health, and circuit breaker state.

    Attributes:
        strategy: Routing strategy to use
        health_check_interval: Seconds between health checks
        enable_circuit_breaker: Whether to use circuit breakers
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Seconds before attempting recovery
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.PRIORITY,
        health_check_interval: float = 30.0,
        enable_circuit_breaker: bool = True,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize LLM router.

        Args:
            strategy: Routing strategy (default: PRIORITY)
            health_check_interval: Seconds between health checks (default: 30)
            enable_circuit_breaker: Enable circuit breaker protection (default: True)
            circuit_breaker_threshold: Failures before opening (default: 5)
            circuit_breaker_timeout: Recovery timeout in seconds (default: 60)
            max_retries: Maximum retry attempts across providers (default: 3)
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.max_retries = max_retries

        # Provider registry
        self._providers: Dict[str, RegisteredProvider] = {}
        self._lock = Lock()
        self._round_robin_index = 0

        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_running = False

        # Global metrics
        self._global_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_cost_usd": 0.0,
            "failover_count": 0,
        }

        self._logger = logging.getLogger(f"{__name__}.router")
        self._logger.info(
            f"Initialized LLMRouter: strategy={strategy.value}, "
            f"circuit_breaker={enable_circuit_breaker}"
        )

    def register_provider(
        self,
        name: str,
        provider: BaseLLMProvider,
        priority: int = 100,
        enabled: bool = True,
    ) -> None:
        """
        Register a new LLM provider.

        Args:
            name: Unique provider name
            provider: Provider instance
            priority: Priority level (lower = higher priority, default: 100)
            enabled: Whether provider is enabled (default: True)

        Raises:
            ValueError: If provider name already registered
        """
        with self._lock:
            if name in self._providers:
                raise ValueError(f"Provider '{name}' is already registered")

            # Create circuit breaker if enabled
            circuit_breaker = None
            if self.enable_circuit_breaker:
                circuit_breaker = CircuitBreaker(
                    failure_threshold=self.circuit_breaker_threshold,
                    recovery_timeout=self.circuit_breaker_timeout,
                    name=f"{name}-breaker",
                )

            # Create registered provider
            registered = RegisteredProvider(
                name=name,
                provider=provider,
                priority=priority,
                enabled=enabled,
                circuit_breaker=circuit_breaker,
                metrics=ProviderMetrics(provider_name=name),
            )

            self._providers[name] = registered
            self._logger.info(
                f"Registered provider: {name} (priority={priority}, enabled={enabled})"
            )

    def unregister_provider(self, name: str) -> None:
        """
        Unregister a provider.

        Args:
            name: Provider name to unregister
        """
        with self._lock:
            if name in self._providers:
                del self._providers[name]
                self._logger.info(f"Unregistered provider: {name}")

    def enable_provider(self, name: str) -> None:
        """Enable a provider."""
        with self._lock:
            if name in self._providers:
                self._providers[name].enabled = True
                self._logger.info(f"Enabled provider: {name}")

    def disable_provider(self, name: str) -> None:
        """Disable a provider."""
        with self._lock:
            if name in self._providers:
                self._providers[name].enabled = False
                self._logger.info(f"Disabled provider: {name}")

    async def generate(
        self,
        request: GenerationRequest,
        preferred_provider: Optional[str] = None,
    ) -> GenerationResponse:
        """
        Generate text using the router's strategy with automatic failover.

        Args:
            request: Generation request
            preferred_provider: Preferred provider name (overrides strategy)

        Returns:
            Generation response

        Raises:
            ProviderError: If all providers fail
        """
        self._global_metrics["total_requests"] += 1
        start_time = time.time()

        # Get provider selection order
        providers = self._select_providers(preferred_provider)

        if not providers:
            raise ProviderError(
                "No healthy providers available",
                "router",
                retryable=False,
            )

        # Try providers in order with failover
        last_error: Optional[Exception] = None
        attempts = []

        for attempt, registered in enumerate(providers[: self.max_retries]):
            try:
                self._logger.info(
                    f"Attempting provider: {registered.name} "
                    f"(attempt {attempt + 1}/{min(len(providers), self.max_retries)})"
                )

                # Use circuit breaker if enabled
                if registered.circuit_breaker:
                    response = await registered.circuit_breaker.call(
                        self._generate_with_provider, registered, request
                    )
                else:
                    response = await self._generate_with_provider(registered, request)

                # Update metrics
                latency_ms = (time.time() - start_time) * 1000
                registered.metrics.update_request(
                    success=True,
                    cost=response.usage.total_cost_usd,
                    tokens=response.usage.total_tokens,
                    latency_ms=latency_ms,
                )

                self._global_metrics["successful_requests"] += 1
                self._global_metrics["total_cost_usd"] += response.usage.total_cost_usd

                if attempt > 0:
                    self._global_metrics["failover_count"] += 1
                    self._logger.info(
                        f"Failover successful to {registered.name} after {attempt} attempts"
                    )

                return response

            except CircuitBreakerOpenError as e:
                last_error = e
                self._logger.warning(
                    f"Circuit breaker OPEN for {registered.name}: {str(e)}"
                )
                attempts.append(
                    f"{registered.name}: circuit breaker open (retry after {e.retry_after:.0f}s)"
                )
                continue

            except ProviderError as e:
                last_error = e
                latency_ms = (time.time() - start_time) * 1000
                registered.metrics.update_request(
                    success=False,
                    cost=0.0,
                    tokens=0,
                    latency_ms=latency_ms,
                )

                self._logger.warning(
                    f"Provider {registered.name} failed (retryable={e.retryable}): {str(e)}"
                )
                attempts.append(f"{registered.name}: {str(e)}")

                # Don't retry if error is not retryable
                if not e.retryable:
                    break

                continue

            except Exception as e:
                last_error = e
                self._logger.error(
                    f"Unexpected error with {registered.name}: {str(e)}",
                    exc_info=True,
                )
                attempts.append(f"{registered.name}: unexpected error")
                continue

        # All providers failed
        self._global_metrics["failed_requests"] += 1

        failure_summary = "\n".join([f"  - {att}" for att in attempts])
        error_msg = (
            f"All providers failed after {len(attempts)} attempts:\n{failure_summary}"
        )

        self._logger.error(error_msg)
        raise ProviderError(
            error_msg,
            "router",
            retryable=False,
            original_error=last_error,
        )

    async def _generate_with_provider(
        self, registered: RegisteredProvider, request: GenerationRequest
    ) -> GenerationResponse:
        """
        Generate using a specific provider.

        Args:
            registered: Registered provider
            request: Generation request

        Returns:
            Generation response
        """
        return await registered.provider.generate(request)

    def _select_providers(
        self, preferred_provider: Optional[str] = None
    ) -> List[RegisteredProvider]:
        """
        Select providers based on strategy.

        Args:
            preferred_provider: Preferred provider name (overrides strategy)

        Returns:
            Ordered list of providers to try
        """
        with self._lock:
            # Filter to enabled providers with healthy circuit breakers
            available = [
                p
                for p in self._providers.values()
                if p.enabled
                and (
                    not p.circuit_breaker
                    or not p.circuit_breaker.is_open()
                )
            ]

            if not available:
                return []

            # Use preferred provider if specified
            if preferred_provider and preferred_provider in self._providers:
                pref = self._providers[preferred_provider]
                if pref in available:
                    # Put preferred first, then others
                    others = [p for p in available if p.name != preferred_provider]
                    return [pref] + others

            # Apply routing strategy
            if self.strategy == RoutingStrategy.PRIORITY:
                return sorted(available, key=lambda p: p.priority)

            elif self.strategy == RoutingStrategy.LEAST_COST:
                return sorted(
                    available,
                    key=lambda p: (
                        p.provider.cost_per_1k_input + p.provider.cost_per_1k_output
                    ),
                )

            elif self.strategy == RoutingStrategy.LEAST_LATENCY:
                # Sort by average latency (providers with no history go last)
                return sorted(
                    available,
                    key=lambda p: (
                        p.metrics.avg_latency_ms if p.metrics.avg_latency_ms > 0 else float("inf")
                    ),
                )

            elif self.strategy == RoutingStrategy.ROUND_ROBIN:
                # Rotate through available providers
                if not available:
                    return []
                index = self._round_robin_index % len(available)
                self._round_robin_index += 1
                # Start from index and wrap around
                return available[index:] + available[:index]

            elif self.strategy == RoutingStrategy.RANDOM:
                import random
                shuffled = available.copy()
                deterministic_random().shuffle(shuffled)
                return shuffled

            else:
                # Default to priority
                return sorted(available, key=lambda p: p.priority)

    async def health_check_all(self) -> Dict[str, ProviderHealth]:
        """
        Run health checks on all providers.

        Returns:
            Dict mapping provider names to health status
        """
        results = {}

        with self._lock:
            providers = list(self._providers.values())

        # Run health checks in parallel
        tasks = []
        for registered in providers:
            task = self._check_provider_health(registered)
            tasks.append((registered.name, task))

        for name, task in tasks:
            try:
                health = await task
                results[name] = health
            except Exception as e:
                self._logger.error(f"Health check failed for {name}: {str(e)}")
                results[name] = ProviderHealth(
                    is_healthy=False,
                    last_check=DeterministicClock.utcnow(),
                    last_error=str(e),
                )

        return results

    async def _check_provider_health(
        self, registered: RegisteredProvider
    ) -> ProviderHealth:
        """Check health of a single provider."""
        try:
            health = await registered.provider.health_check()
            registered.metrics.health_check_count += 1
            registered.metrics.last_health_check = DeterministicClock.utcnow()
            return health
        except Exception as e:
            self._logger.warning(f"Health check failed for {registered.name}: {str(e)}")
            return ProviderHealth(
                is_healthy=False,
                last_check=DeterministicClock.utcnow(),
                last_error=str(e),
            )

    async def start_health_monitoring(self) -> None:
        """Start background health check monitoring."""
        if self._health_check_running:
            self._logger.warning("Health monitoring already running")
            return

        self._health_check_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._logger.info(
            f"Started health monitoring (interval={self.health_check_interval}s)"
        )

    async def stop_health_monitoring(self) -> None:
        """Stop background health check monitoring."""
        self._health_check_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Stopped health monitoring")

    async def _health_check_loop(self) -> None:
        """Background loop for health checks."""
        while self._health_check_running:
            try:
                results = await self.health_check_all()

                # Log unhealthy providers
                for name, health in results.items():
                    if not health.is_healthy:
                        self._logger.warning(
                            f"Provider {name} is unhealthy: {health.last_error}"
                        )

            except Exception as e:
                self._logger.error(f"Health check loop error: {str(e)}", exc_info=True)

            # Wait for next interval
            await asyncio.sleep(self.health_check_interval)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get routing metrics.

        Returns:
            Dict with global and per-provider metrics
        """
        with self._lock:
            provider_metrics = {
                name: {
                    "total_requests": p.metrics.total_requests,
                    "successful_requests": p.metrics.successful_requests,
                    "failed_requests": p.metrics.failed_requests,
                    "success_rate": p.metrics.success_rate,
                    "total_cost_usd": p.metrics.total_cost_usd,
                    "total_tokens": p.metrics.total_tokens,
                    "avg_latency_ms": p.metrics.avg_latency_ms,
                    "last_used": p.metrics.last_used.isoformat()
                    if p.metrics.last_used
                    else None,
                    "enabled": p.enabled,
                    "circuit_breaker_state": p.circuit_breaker.state.value
                    if p.circuit_breaker
                    else "n/a",
                }
                for name, p in self._providers.items()
            }

            return {
                "global": self._global_metrics.copy(),
                "providers": provider_metrics,
                "strategy": self.strategy.value,
            }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._global_metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_cost_usd": 0.0,
                "failover_count": 0,
            }
            for provider in self._providers.values():
                provider.metrics = ProviderMetrics(provider_name=provider.name)
        self._logger.info("Reset all metrics")

    async def close(self) -> None:
        """Close router and cleanup resources."""
        await self.stop_health_monitoring()

        with self._lock:
            # Close all providers
            for registered in self._providers.values():
                try:
                    if hasattr(registered.provider, "close"):
                        await registered.provider.close()
                except Exception as e:
                    self._logger.error(
                        f"Error closing provider {registered.name}: {str(e)}"
                    )

        self._logger.info("LLMRouter closed")


# Example usage
if __name__ == "__main__":
    import asyncio
    import os

    from .providers.anthropic_provider import AnthropicProvider
    from .providers.openai_provider import OpenAIProvider

    async def main():
        """Test the LLM router with multiple providers."""
        # Initialize router with least-cost strategy
        router = LLMRouter(
            strategy=RoutingStrategy.LEAST_COST,
            health_check_interval=30.0,
            enable_circuit_breaker=True,
        )

        # Register providers
        try:
            anthropic = AnthropicProvider(
                model_id="claude-3-sonnet-20240229",  # Cheaper model
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            router.register_provider("anthropic", anthropic, priority=1)
            print("Registered Anthropic provider")
        except Exception as e:
            print(f"Could not register Anthropic: {e}")

        try:
            openai = OpenAIProvider(
                model_id="gpt-3.5-turbo",  # Cheaper model
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            router.register_provider("openai", openai, priority=2)
            print("Registered OpenAI provider")
        except Exception as e:
            print(f"Could not register OpenAI: {e}")

        # Start health monitoring
        await router.start_health_monitoring()

        # Test generation with automatic provider selection
        request = GenerationRequest(
            prompt="What is carbon accounting? Answer in 2 sentences.",
            temperature=0.7,
            max_tokens=100,
        )

        try:
            print("\n=== Test 1: Automatic provider selection (least-cost) ===")
            response = await router.generate(request)
            print(f"Provider used: {response.provider}")
            print(f"Response: {response.text}")
            print(f"Cost: ${response.usage.total_cost_usd:.4f}")
            print(f"Latency: {response.generation_time_ms:.0f}ms")
        except Exception as e:
            print(f"Generation failed: {e}")

        # Test with preferred provider
        try:
            print("\n=== Test 2: Preferred provider (anthropic) ===")
            response = await router.generate(request, preferred_provider="anthropic")
            print(f"Provider used: {response.provider}")
            print(f"Cost: ${response.usage.total_cost_usd:.4f}")
        except Exception as e:
            print(f"Generation failed: {e}")

        # Get metrics
        print("\n=== Router Metrics ===")
        metrics = router.get_metrics()
        print(f"Global metrics: {metrics['global']}")
        print(f"\nProvider metrics:")
        for name, pmetrics in metrics["providers"].items():
            print(f"  {name}:")
            print(f"    Requests: {pmetrics['total_requests']}")
            print(f"    Success rate: {pmetrics['success_rate']:.1f}%")
            print(f"    Total cost: ${pmetrics['total_cost_usd']:.4f}")
            print(f"    Avg latency: {pmetrics['avg_latency_ms']:.0f}ms")

        # Cleanup
        await router.close()

    asyncio.run(main())
