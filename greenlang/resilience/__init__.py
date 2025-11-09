"""GreenLang Resilience Patterns.

Production-grade resilience infrastructure for distributed systems including:
- Retry logic with exponential backoff and jitter
- Timeout management for different operation types
- Fallback mechanisms and graceful degradation
- Rate limiting and throttling
- Circuit breaker patterns

Inspired by Netflix Hystrix, Polly, and resilience4j.

Author: GreenLang Resilience Team
Date: November 2025
Status: Production Ready
"""

from .retry import (
    retry,
    async_retry,
    RetryConfig,
    RetryStrategy,
    RetryableError,
    MaxRetriesExceeded,
)
from .timeout import (
    timeout,
    async_timeout,
    TimeoutConfig,
    OperationType,
    TimeoutError,
    get_timeout_for_operation,
)
from .fallback import (
    fallback,
    async_fallback,
    FallbackStrategy,
    FallbackConfig,
    CachedFallback,
    DefaultFallback,
    get_cached_fallback,
)
from .rate_limit_handler import (
    RateLimiter,
    TokenBucket,
    LeakyBucket,
    RateLimitConfig,
    RateLimitExceeded,
    get_rate_limiter,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitOpenError,
    create_circuit_breaker,
    get_circuit_breaker_stats,
)

__all__ = [
    # Retry
    "retry",
    "async_retry",
    "RetryConfig",
    "RetryStrategy",
    "RetryableError",
    "MaxRetriesExceeded",
    # Timeout
    "timeout",
    "async_timeout",
    "TimeoutConfig",
    "OperationType",
    "TimeoutError",
    "get_timeout_for_operation",
    # Fallback
    "fallback",
    "async_fallback",
    "FallbackStrategy",
    "FallbackConfig",
    "CachedFallback",
    "DefaultFallback",
    "get_cached_fallback",
    # Rate Limiting
    "RateLimiter",
    "TokenBucket",
    "LeakyBucket",
    "RateLimitConfig",
    "RateLimitExceeded",
    "get_rate_limiter",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitOpenError",
    "create_circuit_breaker",
    "get_circuit_breaker_stats",
]

__version__ = "1.0.0"
