"""
Retry Policy - Agent Factory Resilience (INFRA-010)

Provides configurable retry logic with exponential backoff and jitter
for transient failures in agent execution. Distinguishes between
retryable and non-retryable exceptions to avoid wasting resources
on permanent failures.

Classes:
    - RetryConfig: Configuration for retry behaviour.
    - RetryExhaustedError: Raised when all retry attempts are exhausted.
    - RetryPolicy: Core retry implementation with decorator support.

Example:
    >>> policy = RetryPolicy(RetryConfig(max_attempts=3))
    >>> result = await policy.execute(agent.process, data)

    >>> @RetryPolicy.as_decorator(RetryConfig(max_attempts=5))
    ... async def flaky_operation():
    ...     return await external_api.call()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry policy behaviour.

    Attributes:
        max_attempts: Maximum number of attempts (including the initial call).
        base_delay_s: Base delay between retries in seconds.
        max_delay_s: Maximum delay cap in seconds.
        backoff_multiplier: Multiplier applied to delay after each retry.
        jitter_range_s: Maximum random jitter added to delay (0.0 to this value).
        retryable_exceptions: Exception types that are eligible for retry.
            Empty set means all exceptions are retryable (except non-retryable).
        non_retryable_exceptions: Exception types that must never be retried.
    """

    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_range_s: float = 0.5
    retryable_exceptions: frozenset[Type[Exception]] = frozenset()
    non_retryable_exceptions: frozenset[Type[Exception]] = frozenset({
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        PermissionError,
    })


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        agent_key: The agent key (if available).
        attempts: Number of attempts made.
        total_duration_s: Total time spent across all attempts.
        last_exception: The final exception that caused failure.
        attempt_errors: List of (attempt_number, exception) pairs.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        total_duration_s: float,
        last_exception: Exception,
        attempt_errors: Optional[List[Tuple[int, Exception]]] = None,
    ) -> None:
        self.attempts = attempts
        self.total_duration_s = total_duration_s
        self.last_exception = last_exception
        self.attempt_errors = attempt_errors or []
        super().__init__(message)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class RetryMetrics:
    """Metrics for retry policy usage.

    Attributes:
        total_executions: Total calls to execute().
        total_retries: Total retry attempts across all executions.
        total_exhausted: Times all retries were exhausted.
        total_succeeded: Times execution succeeded (including after retries).
        total_duration_s: Cumulative time spent in retry logic.
    """

    total_executions: int = 0
    total_retries: int = 0
    total_exhausted: int = 0
    total_succeeded: int = 0
    total_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# On-Retry Callback
# ---------------------------------------------------------------------------

OnRetryCallback = Callable[[int, Exception, float], Awaitable[None]]
"""Async callback invoked before each retry.
Args: attempt_number, exception, next_delay_s.
"""


# ---------------------------------------------------------------------------
# Retry Policy
# ---------------------------------------------------------------------------


class RetryPolicy:
    """Retry logic with exponential backoff and jitter.

    Wraps an async callable and retries on transient failures according
    to the configured policy. Distinguishes between retryable and
    non-retryable exceptions.

    Attributes:
        config: Retry configuration.
        metrics: Observable metrics.
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[OnRetryCallback] = None,
    ) -> None:
        """Initialize a retry policy.

        Args:
            config: Retry configuration. Uses defaults if None.
            on_retry: Optional async callback invoked before each retry.
        """
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()
        self._on_retry = on_retry

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a callable with retry logic.

        Args:
            fn: Async callable to execute.
            *args: Positional arguments passed to fn.
            **kwargs: Keyword arguments passed to fn.

        Returns:
            The result of fn.

        Raises:
            RetryExhaustedError: If all attempts fail.
            Exception: If a non-retryable exception is raised.
        """
        self.metrics.total_executions += 1
        start_time = time.perf_counter()
        attempt_errors: List[Tuple[int, Exception]] = []
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await fn(*args, **kwargs)
                duration = time.perf_counter() - start_time
                self.metrics.total_succeeded += 1
                self.metrics.total_duration_s += duration

                if attempt > 1:
                    logger.info(
                        "RetryPolicy: succeeded on attempt %d/%d (%.2fs total)",
                        attempt, self.config.max_attempts, duration,
                    )
                return result

            except Exception as exc:
                last_exc = exc
                attempt_errors.append((attempt, exc))

                # Check if this exception is non-retryable
                if self._is_non_retryable(exc):
                    duration = time.perf_counter() - start_time
                    self.metrics.total_duration_s += duration
                    logger.warning(
                        "RetryPolicy: non-retryable exception on attempt %d: %s",
                        attempt, type(exc).__name__,
                    )
                    raise

                # Check if this exception is retryable
                if not self._is_retryable(exc):
                    duration = time.perf_counter() - start_time
                    self.metrics.total_duration_s += duration
                    logger.warning(
                        "RetryPolicy: non-retryable exception type %s on attempt %d",
                        type(exc).__name__, attempt,
                    )
                    raise

                # If this was the last attempt, do not sleep
                if attempt >= self.config.max_attempts:
                    break

                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_delay(attempt)
                self.metrics.total_retries += 1

                logger.warning(
                    "RetryPolicy: attempt %d/%d failed (%s: %s), "
                    "retrying in %.2fs",
                    attempt, self.config.max_attempts,
                    type(exc).__name__, str(exc)[:200], delay,
                )

                # Invoke on-retry callback
                if self._on_retry is not None:
                    try:
                        await self._on_retry(attempt, exc, delay)
                    except Exception as cb_exc:
                        logger.error(
                            "RetryPolicy: on_retry callback failed: %s", cb_exc,
                        )

                await asyncio.sleep(delay)

        # All attempts exhausted
        duration = time.perf_counter() - start_time
        self.metrics.total_exhausted += 1
        self.metrics.total_duration_s += duration

        assert last_exc is not None
        raise RetryExhaustedError(
            message=(
                f"RetryPolicy: all {self.config.max_attempts} attempts exhausted "
                f"after {duration:.2f}s. Last error: {last_exc}"
            ),
            attempts=self.config.max_attempts,
            total_duration_s=duration,
            last_exception=last_exc,
            attempt_errors=attempt_errors,
        )

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    @staticmethod
    def as_decorator(
        config: Optional[RetryConfig] = None,
        on_retry: Optional[OnRetryCallback] = None,
    ) -> Callable:
        """Create a decorator that wraps an async function with retry logic.

        Args:
            config: Retry configuration.
            on_retry: Optional callback invoked before each retry.

        Returns:
            Decorator function.

        Example:
            >>> @RetryPolicy.as_decorator(RetryConfig(max_attempts=5))
            ... async def fetch_data():
            ...     return await api.get("/data")
        """
        policy = RetryPolicy(config, on_retry)

        def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @functools.wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                return await policy.execute(fn, *args, **kwargs)
            wrapper._retry_policy = policy  # type: ignore[attr-defined]
            return wrapper

        return decorator

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _is_retryable(self, exc: Exception) -> bool:
        """Determine if an exception is eligible for retry.

        If retryable_exceptions is empty, all exceptions are retryable
        (except those in non_retryable_exceptions). Otherwise, only
        exceptions in the retryable set are retried.
        """
        if self._is_non_retryable(exc):
            return False
        if not self.config.retryable_exceptions:
            return True
        return isinstance(exc, tuple(self.config.retryable_exceptions))

    def _is_non_retryable(self, exc: Exception) -> bool:
        """Check if an exception is explicitly non-retryable."""
        if not self.config.non_retryable_exceptions:
            return False
        return isinstance(exc, tuple(self.config.non_retryable_exceptions))

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter.

        Args:
            attempt: The current attempt number (1-based).

        Returns:
            Delay in seconds before the next retry.
        """
        # Exponential backoff
        delay = self.config.base_delay_s * (
            self.config.backoff_multiplier ** (attempt - 1)
        )

        # Add jitter
        if self.config.jitter_range_s > 0:
            jitter = random.uniform(0, self.config.jitter_range_s)
            delay += jitter

        # Cap at max delay
        delay = min(delay, self.config.max_delay_s)

        return delay

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a diagnostic snapshot of the retry policy metrics.

        Returns:
            Dictionary with configuration and metrics.
        """
        return {
            "config": {
                "max_attempts": self.config.max_attempts,
                "base_delay_s": self.config.base_delay_s,
                "max_delay_s": self.config.max_delay_s,
                "backoff_multiplier": self.config.backoff_multiplier,
                "jitter_range_s": self.config.jitter_range_s,
            },
            "metrics": {
                "total_executions": self.metrics.total_executions,
                "total_retries": self.metrics.total_retries,
                "total_exhausted": self.metrics.total_exhausted,
                "total_succeeded": self.metrics.total_succeeded,
                "total_duration_s": round(self.metrics.total_duration_s, 3),
            },
        }


__all__ = [
    "OnRetryCallback",
    "RetryConfig",
    "RetryExhaustedError",
    "RetryMetrics",
    "RetryPolicy",
]
