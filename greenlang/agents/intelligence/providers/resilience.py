# -*- coding: utf-8 -*-
"""
Resilient HTTP Client with Circuit Breaker

Provides production-grade resilience for LLM provider calls:
- Circuit Breaker pattern (closed, open, half-open states)
- Exponential backoff for rate limits
- Automatic recovery after cooldown
- Graceful degradation under load

Architecture:
    Request → Circuit Check → [if closed] → HTTP Call → Track Success/Failure
                           → [if open] → Fast Fail (circuit_open error)
                           → [if half-open] → Test Call → Close or Re-Open

Example:
    client = ResilientHTTPClient(
        failure_threshold=5,      # Open after 5 failures
        recovery_timeout=60.0,    # Try recovery after 60s
        expected_exception=ProviderError
    )

    response = await client.call(
        func=provider._call_api,
        *args, **kwargs
    )
"""

from __future__ import annotations
import time
import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, fast-fail all requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    last_state_change: float
    total_calls: int
    failed_calls: int


class CircuitBreakerError(Exception):
    """Raised when circuit is open"""

    def __init__(self, stats: CircuitBreakerStats):
        self.stats = stats
        super().__init__(
            f"Circuit breaker is OPEN. "
            f"Failures: {stats.failure_count}, "
            f"Last failure: {stats.last_failure_time}s ago"
        )


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, fast-fail all requests
    - HALF_OPEN: Testing recovery, allow 1 request

    Transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN → CLOSED: If test request succeeds
    - HALF_OPEN → OPEN: If test request fails

    Usage:
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )

        async def make_request():
            if not breaker.can_proceed():
                raise CircuitBreakerError(breaker.get_stats())

            try:
                result = await api_call()
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before trying recovery
            success_threshold: Consecutive successes in half-open to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change = time.time()
        self._total_calls = 0
        self._failed_calls = 0

        logger.info(
            f"CircuitBreaker initialized: "
            f"failure_threshold={failure_threshold}, "
            f"recovery_timeout={recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state"""
        return self._state

    def can_proceed(self) -> bool:
        """
        Check if request can proceed

        Returns:
            True if request allowed, False if circuit open
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if should transition to half-open
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False

        # HALF_OPEN: allow one request to test
        return True

    def record_success(self) -> None:
        """Record successful request"""
        self._total_calls += 1
        self._success_count += 1
        self._failure_count = 0  # Reset failure counter

        if self._state == CircuitState.HALF_OPEN:
            if self._success_count >= self.success_threshold:
                # Recovered! Close circuit
                self._transition_to(CircuitState.CLOSED)
                logger.info("Circuit breaker recovered: HALF_OPEN → CLOSED")

    def record_failure(self) -> None:
        """Record failed request"""
        self._total_calls += 1
        self._failed_calls += 1
        self._failure_count += 1
        self._success_count = 0  # Reset success counter
        self._last_failure_time = time.time()

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                # Too many failures! Open circuit
                self._transition_to(CircuitState.OPEN)
                logger.error(
                    f"Circuit breaker opened after {self._failure_count} failures"
                )

        elif self._state == CircuitState.HALF_OPEN:
            # Recovery failed, back to open
            self._transition_to(CircuitState.OPEN)
            logger.warning("Circuit breaker recovery failed: HALF_OPEN → OPEN")

    def _should_attempt_reset(self) -> bool:
        """Check if should attempt recovery"""
        if self._last_failure_time is None:
            return False

        time_since_failure = time.time() - self._last_failure_time
        return time_since_failure >= self.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state"""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        logger.info(f"Circuit breaker: {old_state.value} → {new_state.value}")

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        return CircuitBreakerStats(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_state_change=self._last_state_change,
            total_calls=self._total_calls,
            failed_calls=self._failed_calls,
        )

    def reset(self) -> None:
        """Manually reset circuit breaker"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        logger.info("Circuit breaker manually reset")


class ResilientHTTPClient:
    """
    HTTP client with circuit breaker and exponential backoff

    Provides:
    - Circuit breaker for fast-fail during outages
    - Exponential backoff for rate limits
    - Automatic recovery
    - Statistics tracking

    Usage:
        client = ResilientHTTPClient()

        # Wrap provider calls
        response = await client.call(
            func=provider._call_api,
            url="https://api.openai.com/v1/chat/completions",
            payload={"model": "gpt-4", "messages": [...]}
        )
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        """
        Initialize resilient HTTP client

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before recovery attempt
            max_retries: Max retry attempts per request
            base_delay: Base delay for exponential backoff
        """
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.max_retries = max_retries
        self.base_delay = base_delay

        logger.info(
            f"ResilientHTTPClient initialized: "
            f"max_retries={max_retries}, base_delay={base_delay}s"
        )

    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute HTTP call with circuit breaker and retry

        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: If all retries exhausted
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            raise CircuitBreakerError(self.circuit_breaker.get_stats())

        # Retry with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                self.circuit_breaker.record_success()
                return result

            except Exception as e:
                self.circuit_breaker.record_failure()

                # Check if retryable
                is_retryable = self._is_retryable_error(e)

                if attempt >= self.max_retries or not is_retryable:
                    raise

                # Calculate backoff delay
                delay = self.base_delay * (2 ** attempt)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying after {delay}s: {e}"
                )

                await asyncio.sleep(delay)

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if error is retryable

        Args:
            error: Exception to check

        Returns:
            True if should retry, False otherwise
        """
        # Import here to avoid circular dependency
        from greenlang.agents.intelligence.providers.errors import (
            ProviderRateLimit,
            ProviderTimeout,
            ProviderServerError,
        )

        # Retry on rate limits, timeouts, server errors
        return isinstance(error, (
            ProviderRateLimit,
            ProviderTimeout,
            ProviderServerError,
        ))

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics"""
        return self.circuit_breaker.get_stats()

    def reset(self) -> None:
        """Reset circuit breaker"""
        self.circuit_breaker.reset()


# Global resilient client cache
_resilient_clients = {}


def get_resilient_client(
    provider_name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
) -> ResilientHTTPClient:
    """
    Get resilient HTTP client for provider (cached)

    Args:
        provider_name: Provider name (openai, anthropic, etc.)
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before recovery

    Returns:
        ResilientHTTPClient instance (cached per provider)

    Example:
        client = get_resilient_client("openai")
        response = await client.call(provider._call_api, ...)
    """
    key = f"{provider_name}_{failure_threshold}_{recovery_timeout}"
    if key not in _resilient_clients:
        _resilient_clients[key] = ResilientHTTPClient(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
    return _resilient_clients[key]
