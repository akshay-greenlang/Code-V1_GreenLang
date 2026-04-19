"""
Retry Policy Implementation for GreenLang

This module provides Tenacity-style retry policies
for handling transient failures.

Features:
- Exponential backoff
- Jitter support
- Custom retry conditions
- Max attempts limiting
- Timeout support
- Callback hooks

Example:
    >>> policy = RetryPolicy(config)
    >>> @policy
    ... async def fetch_data():
    ...     return await http_client.get("/data")
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Optional, Set, Type, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BackoffStrategy(str, Enum):
    """Backoff strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


class JitterType(str, Enum):
    """Jitter types."""
    NONE = "none"
    FULL = "full"
    EQUAL = "equal"
    DECORRELATED = "decorrelated"


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_attempts: int = 3
    max_delay_ms: int = 60000  # 60 seconds
    initial_delay_ms: int = 1000  # 1 second
    multiplier: float = 2.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter_type: JitterType = JitterType.FULL
    jitter_factor: float = 0.5
    retry_exceptions: List[Type[Exception]] = field(default_factory=list)
    ignore_exceptions: List[Type[Exception]] = field(default_factory=list)
    retry_on_result: Optional[Callable[[Any], bool]] = None
    timeout_ms: Optional[int] = None


class RetryAttempt(BaseModel):
    """Information about a retry attempt."""
    attempt_number: int = Field(..., description="Attempt number")
    delay_ms: float = Field(..., description="Delay before this attempt")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exception: Optional[str] = Field(default=None, description="Exception if failed")
    result: Optional[Any] = Field(default=None, description="Result if succeeded")


class RetryMetrics(BaseModel):
    """Metrics for retry policy."""
    total_calls: int = Field(default=0, description="Total calls")
    successful_calls: int = Field(default=0, description="Successful calls")
    failed_calls: int = Field(default=0, description="Failed calls after all retries")
    total_retries: int = Field(default=0, description="Total retry attempts")
    last_call_attempts: int = Field(default=0, description="Attempts in last call")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None
    ):
        """Initialize error."""
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(message)


class ExponentialBackoff:
    """
    Exponential backoff calculator.

    Calculates delay with exponential growth, optional jitter,
    and max delay capping.
    """

    def __init__(
        self,
        initial_delay_ms: int = 1000,
        multiplier: float = 2.0,
        max_delay_ms: int = 60000,
        jitter_type: JitterType = JitterType.FULL,
        jitter_factor: float = 0.5
    ):
        """
        Initialize exponential backoff.

        Args:
            initial_delay_ms: Initial delay in milliseconds
            multiplier: Multiplier for each attempt
            max_delay_ms: Maximum delay cap
            jitter_type: Type of jitter to apply
            jitter_factor: Jitter factor (0-1)
        """
        self.initial_delay = initial_delay_ms
        self.multiplier = multiplier
        self.max_delay = max_delay_ms
        self.jitter_type = jitter_type
        self.jitter_factor = jitter_factor
        self._last_delay = initial_delay_ms

    def calculate(self, attempt: int) -> float:
        """
        Calculate delay for an attempt.

        Args:
            attempt: Attempt number (1-based)

        Returns:
            Delay in milliseconds
        """
        # Calculate base delay
        base_delay = self.initial_delay * (self.multiplier ** (attempt - 1))
        base_delay = min(base_delay, self.max_delay)

        # Apply jitter
        delay = self._apply_jitter(base_delay)
        self._last_delay = delay

        return delay

    def _apply_jitter(self, delay: float) -> float:
        """Apply jitter to delay."""
        if self.jitter_type == JitterType.NONE:
            return delay

        elif self.jitter_type == JitterType.FULL:
            # Random between 0 and delay
            return random.uniform(0, delay)

        elif self.jitter_type == JitterType.EQUAL:
            # Random between delay/2 and delay
            half = delay * self.jitter_factor
            return delay - half + random.uniform(0, half * 2)

        elif self.jitter_type == JitterType.DECORRELATED:
            # Decorrelated jitter
            return random.uniform(
                self.initial_delay,
                self._last_delay * 3
            )

        return delay


class LinearBackoff:
    """Linear backoff calculator."""

    def __init__(
        self,
        initial_delay_ms: int = 1000,
        increment_ms: int = 1000,
        max_delay_ms: int = 60000
    ):
        """Initialize linear backoff."""
        self.initial_delay = initial_delay_ms
        self.increment = increment_ms
        self.max_delay = max_delay_ms

    def calculate(self, attempt: int) -> float:
        """Calculate delay for an attempt."""
        delay = self.initial_delay + (self.increment * (attempt - 1))
        return min(delay, self.max_delay)


class FibonacciBackoff:
    """Fibonacci backoff calculator."""

    def __init__(
        self,
        initial_delay_ms: int = 1000,
        max_delay_ms: int = 60000
    ):
        """Initialize Fibonacci backoff."""
        self.initial_delay = initial_delay_ms
        self.max_delay = max_delay_ms
        self._fib_cache = [1, 1]

    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        while len(self._fib_cache) <= n:
            self._fib_cache.append(
                self._fib_cache[-1] + self._fib_cache[-2]
            )
        return self._fib_cache[n]

    def calculate(self, attempt: int) -> float:
        """Calculate delay for an attempt."""
        fib = self._fibonacci(attempt)
        delay = self.initial_delay * fib
        return min(delay, self.max_delay)


class RetryPolicy:
    """
    Production-ready retry policy.

    Provides configurable retry behavior with multiple
    backoff strategies and conditions.

    Attributes:
        config: Retry configuration
        metrics: Retry metrics

    Example:
        >>> config = RetryConfig(
        ...     max_attempts=5,
        ...     backoff_strategy=BackoffStrategy.EXPONENTIAL,
        ...     retry_exceptions=[ConnectionError, TimeoutError]
        ... )
        >>> policy = RetryPolicy(config)
        >>> result = await policy.execute(fetch_data)
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry policy.

        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self._backoff = self._create_backoff()
        self._metrics = RetryMetrics()
        self._before_retry_callbacks: List[Callable] = []
        self._after_retry_callbacks: List[Callable] = []

        logger.info(
            f"RetryPolicy initialized: max_attempts={self.config.max_attempts}, "
            f"strategy={self.config.backoff_strategy.value}"
        )

    def _create_backoff(self) -> Any:
        """Create backoff calculator based on strategy."""
        if self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            return ExponentialBackoff(
                initial_delay_ms=self.config.initial_delay_ms,
                multiplier=self.config.multiplier,
                max_delay_ms=self.config.max_delay_ms,
                jitter_type=self.config.jitter_type,
                jitter_factor=self.config.jitter_factor
            )
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            return LinearBackoff(
                initial_delay_ms=self.config.initial_delay_ms,
                increment_ms=self.config.initial_delay_ms,
                max_delay_ms=self.config.max_delay_ms
            )
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            return FibonacciBackoff(
                initial_delay_ms=self.config.initial_delay_ms,
                max_delay_ms=self.config.max_delay_ms
            )
        else:
            # Fixed delay
            return lambda attempt: self.config.initial_delay_ms

    def should_retry(
        self,
        exception: Optional[Exception] = None,
        result: Any = None
    ) -> bool:
        """
        Determine if a call should be retried.

        Args:
            exception: Exception that occurred
            result: Result of the call

        Returns:
            True if should retry
        """
        # Check result condition
        if self.config.retry_on_result and result is not None:
            if self.config.retry_on_result(result):
                return True

        if exception is None:
            return False

        # Check ignore list
        if self.config.ignore_exceptions:
            for exc_type in self.config.ignore_exceptions:
                if isinstance(exception, exc_type):
                    return False

        # Check retry list
        if self.config.retry_exceptions:
            for exc_type in self.config.retry_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False

        # Default: retry all exceptions
        return True

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            RetryExhaustedError: If all retries exhausted
        """
        self._metrics.total_calls += 1
        attempts = []
        last_exception = None

        start_time = time.monotonic()
        timeout_seconds = self.config.timeout_ms / 1000 if self.config.timeout_ms else None

        for attempt in range(1, self.config.max_attempts + 1):
            # Check timeout
            if timeout_seconds:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout_seconds:
                    raise RetryExhaustedError(
                        f"Timeout after {elapsed:.2f}s",
                        attempt - 1,
                        last_exception
                    )

            # Calculate delay for this attempt
            if attempt > 1:
                delay_ms = self._backoff.calculate(attempt)
                attempts.append(RetryAttempt(
                    attempt_number=attempt,
                    delay_ms=delay_ms
                ))

                # Call before retry callbacks
                for callback in self._before_retry_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(attempt, delay_ms, last_exception)
                        else:
                            callback(attempt, delay_ms, last_exception)
                    except Exception as e:
                        logger.error(f"Before retry callback error: {e}")

                logger.debug(
                    f"Retry attempt {attempt}/{self.config.max_attempts} "
                    f"after {delay_ms:.0f}ms delay"
                )
                await asyncio.sleep(delay_ms / 1000)
                self._metrics.total_retries += 1

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Check if result should trigger retry
                if self.should_retry(result=result):
                    last_exception = ValueError(f"Retry condition on result: {result}")
                    continue

                self._metrics.successful_calls += 1
                self._metrics.last_call_attempts = attempt

                # Call after retry callbacks
                for callback in self._after_retry_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(attempt, None, result)
                        else:
                            callback(attempt, None, result)
                    except Exception as e:
                        logger.error(f"After retry callback error: {e}")

                return result

            except Exception as e:
                last_exception = e

                if not self.should_retry(exception=e):
                    self._metrics.failed_calls += 1
                    raise

                if attempt < self.config.max_attempts:
                    logger.warning(
                        f"Attempt {attempt} failed: {type(e).__name__}: {e}"
                    )
                else:
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed"
                    )

        # All retries exhausted
        self._metrics.failed_calls += 1
        self._metrics.last_call_attempts = self.config.max_attempts

        raise RetryExhaustedError(
            f"Failed after {self.config.max_attempts} attempts",
            self.config.max_attempts,
            last_exception
        )

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to wrap a function with retry logic.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.execute(func, *args, **kwargs)

        return wrapper

    def before_retry(
        self,
        callback: Callable[[int, float, Optional[Exception]], None]
    ) -> None:
        """
        Register a before-retry callback.

        Args:
            callback: Function called before each retry
        """
        self._before_retry_callbacks.append(callback)

    def after_retry(
        self,
        callback: Callable[[int, Optional[Exception], Any], None]
    ) -> None:
        """
        Register an after-retry callback.

        Args:
            callback: Function called after each attempt
        """
        self._after_retry_callbacks.append(callback)

    def get_metrics(self) -> RetryMetrics:
        """Get retry metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._metrics = RetryMetrics()


def retry(
    max_attempts: int = 3,
    initial_delay_ms: int = 1000,
    multiplier: float = 2.0,
    retry_exceptions: Optional[List[Type[Exception]]] = None
) -> Callable:
    """
    Simple retry decorator.

    Args:
        max_attempts: Maximum retry attempts
        initial_delay_ms: Initial delay
        multiplier: Backoff multiplier
        retry_exceptions: Exceptions to retry on

    Returns:
        Decorator function
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        initial_delay_ms=initial_delay_ms,
        multiplier=multiplier,
        retry_exceptions=retry_exceptions or []
    )
    policy = RetryPolicy(config)

    return policy
