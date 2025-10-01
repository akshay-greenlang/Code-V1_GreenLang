"""
Retry Logic with Exponential Backoff

Handles transient errors from LLM providers:
- Rate limits (429)
- Timeouts (408, 504)
- Server errors (500, 502, 503)

Features:
- Exponential backoff with jitter
- Configurable max retries
- Transient error classification
- Budget-aware retries (don't retry if budget exhausted)
"""

from __future__ import annotations
import asyncio
import random
import time
from typing import TypeVar, Callable, Any, Optional
from functools import wraps

T = TypeVar("T")


class RetryConfig:
    """
    Retry configuration

    Attributes:
        max_retries: Maximum retry attempts
        initial_delay_s: Initial delay before first retry
        max_delay_s: Maximum delay between retries
        backoff_factor: Exponential backoff multiplier
        jitter: Add randomness to prevent thundering herd
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_s: float = 1.0,
        max_delay_s: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_delay_s = initial_delay_s
        self.max_delay_s = max_delay_s
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt

        Uses exponential backoff: delay = initial * (factor ^ attempt)
        Capped at max_delay_s.
        Optional jitter adds randomness.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds

        Example:
            config = RetryConfig(initial_delay_s=1.0, backoff_factor=2.0)

            config.calculate_delay(0)  # ~1s
            config.calculate_delay(1)  # ~2s
            config.calculate_delay(2)  # ~4s
            config.calculate_delay(3)  # ~8s
        """
        delay = self.initial_delay_s * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay_s)

        if self.jitter:
            # Add ±25% jitter
            jitter_factor = 1.0 + (random.random() - 0.5) * 0.5
            delay *= jitter_factor

        return delay


def is_transient_error(error: Exception) -> bool:
    """
    Classify whether an error is transient (worth retrying)

    Transient errors:
    - Rate limits (provider-specific)
    - Timeouts
    - Server errors (5xx)
    - Network errors

    Non-transient errors (don't retry):
    - Authentication failures
    - Bad requests (invalid parameters)
    - Budget exceeded
    - Validation errors

    Args:
        error: Exception to classify

    Returns:
        True if error is transient and worth retrying

    Example:
        from greenlang.intelligence.providers.errors import ProviderRateLimit

        try:
            provider.chat(...)
        except Exception as e:
            if is_transient_error(e):
                # Retry
                ...
            else:
                # Fail immediately
                raise
    """
    # Import provider errors here to avoid circular dependency
    try:
        from greenlang.intelligence.providers.errors import (
            ProviderRateLimit,
            ProviderTimeout,
            ProviderServerError,
        )

        if isinstance(error, (ProviderRateLimit, ProviderTimeout, ProviderServerError)):
            return True
    except ImportError:
        pass

    # Budget errors are never transient
    from greenlang.intelligence.runtime.budget import BudgetExceeded

    if isinstance(error, BudgetExceeded):
        return False

    # Check error message for common transient patterns
    error_msg = str(error).lower()
    transient_patterns = [
        "rate limit",
        "429",
        "timeout",
        "timed out",
        "504",
        "503",
        "502",
        "500",
        "server error",
        "temporarily unavailable",
        "try again",
    ]
    return any(pattern in error_msg for pattern in transient_patterns)


async def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Execute function with retry and exponential backoff

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Function result

    Raises:
        Exception: If all retries exhausted or non-transient error

    Example:
        async def flaky_api_call():
            # Sometimes fails with rate limit
            ...

        result = await retry_with_backoff(
            flaky_api_call,
            config=RetryConfig(max_retries=3)
        )
    """
    config = config or RetryConfig()
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            # Don't retry non-transient errors
            if not is_transient_error(e):
                raise

            # Don't retry if max attempts reached
            if attempt >= config.max_retries:
                raise

            # Calculate delay and wait
            delay = config.calculate_delay(attempt)
            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Retry logic error: no exception but no result")


def retry_decorator(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry logic to async functions

    Args:
        config: Retry configuration

    Returns:
        Decorator function

    Example:
        @retry_decorator(config=RetryConfig(max_retries=3))
        async def call_openai_api(...):
            # API call that might fail transiently
            ...

        # Automatic retry on transient errors
        result = await call_openai_api(...)
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(func, *args, config=config, **kwargs)

        return wrapper

    return decorator


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted"""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Retry exhausted after {attempts} attempts. Last error: {last_error}"
        )
