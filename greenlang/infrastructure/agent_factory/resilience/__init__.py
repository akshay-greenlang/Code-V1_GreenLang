"""
Agent Factory Resilience Module - INFRA-010

Provides production-grade resilience patterns for agent execution in the
GreenLang Climate OS platform. Implements Circuit Breaker, Fallback Chain,
Bulkhead Isolation, Retry Policy, and Timeout Guard to protect against
cascading failures and ensure system stability under load.

Public API:
    - CircuitBreaker: Circuit breaker pattern with sliding window failure tracking.
    - CircuitBreakerState: CLOSED / OPEN / HALF_OPEN state enumeration.
    - CircuitBreakerConfig: Configuration for circuit breaker behaviour.
    - CircuitOpenError: Raised when the circuit is open.
    - FallbackChain: Ordered fallback handler chain.
    - FallbackHandler: Protocol for fallback handler implementations.
    - FallbackResult: Result of a fallback chain execution.
    - BulkheadIsolation: Semaphore-based concurrency limiter per agent.
    - BulkheadConfig: Bulkhead configuration.
    - BulkheadFullError: Raised when the bulkhead is full.
    - RetryPolicy: Retry with exponential backoff and jitter.
    - RetryConfig: Retry configuration.
    - RetryExhaustedError: Raised when all retries are exhausted.
    - TimeoutGuard: Timeout enforcement for agent execution.
    - TimeoutConfig: Timeout configuration.
    - AgentTimeoutError: Raised when an agent exceeds its timeout.

Example:
    >>> from greenlang.infrastructure.agent_factory.resilience import (
    ...     CircuitBreaker, RetryPolicy, TimeoutGuard, BulkheadIsolation,
    ... )
    >>> cb = CircuitBreaker.get_or_create("intake-agent")
    >>> async with cb:
    ...     result = await agent.process(data)
"""

from __future__ import annotations

from greenlang.infrastructure.agent_factory.resilience.bulkhead import (
    BulkheadConfig,
    BulkheadFullError,
    BulkheadIsolation,
)
from greenlang.infrastructure.agent_factory.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitOpenError,
)
from greenlang.infrastructure.agent_factory.resilience.fallback import (
    FallbackChain,
    FallbackHandler,
    FallbackResult,
)
from greenlang.infrastructure.agent_factory.resilience.retry import (
    RetryConfig,
    RetryExhaustedError,
    RetryPolicy,
)
from greenlang.infrastructure.agent_factory.resilience.timeout import (
    AgentTimeoutError,
    TimeoutConfig,
    TimeoutGuard,
)

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitOpenError",
    # Fallback
    "FallbackChain",
    "FallbackHandler",
    "FallbackResult",
    # Bulkhead
    "BulkheadConfig",
    "BulkheadFullError",
    "BulkheadIsolation",
    # Retry
    "RetryConfig",
    "RetryExhaustedError",
    "RetryPolicy",
    # Timeout
    "AgentTimeoutError",
    "TimeoutConfig",
    "TimeoutGuard",
]
