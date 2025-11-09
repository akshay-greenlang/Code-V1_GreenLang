"""Retry Logic with Exponential Backoff and Jitter.

Production-grade retry decorator with configurable policies:
- Exponential backoff: 1s, 2s, 4s, 8s...
- Jitter to prevent thundering herd
- Specific exceptions to retry vs fail fast
- Configurable max retries
- Dead letter queue for permanent failures

Patterns inspired by AWS SDK, Google Cloud, and resilience4j.

Author: GreenLang Resilience Team
Date: November 2025
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# ==============================================================================
# Exceptions
# ==============================================================================


class RetryableError(Exception):
    """Base exception for retryable errors."""
    pass


class MaxRetriesExceeded(Exception):
    """Raised when max retries exceeded."""

    def __init__(self, attempts: int, last_error: Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Max retries exceeded after {attempts} attempts. "
            f"Last error: {last_error}"
        )


# ==============================================================================
# Configuration
# ==============================================================================


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 60.0)
        strategy: Backoff strategy (default: exponential)
        jitter: Add randomness to prevent thundering herd (default: True)
        jitter_range: Jitter range as fraction of delay (default: 0.1 = 10%)
        retryable_exceptions: Exceptions to retry (default: all)
        non_retryable_exceptions: Exceptions to fail fast (default: none)
        on_retry: Callback function on retry (optional)
        on_failure: Callback function on permanent failure (optional)
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_range: float = 0.1
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    non_retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (KeyboardInterrupt, SystemExit)
    )
    on_retry: Optional[Callable[[int, Exception], None]] = None
    on_failure: Optional[Callable[[int, Exception], None]] = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.CONSTANT:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.FIBONACCI:
            # Fibonacci: 1, 1, 2, 3, 5, 8, 13...
            fib = [1, 1]
            for i in range(2, attempt + 2):
                fib.append(fib[-1] + fib[-2])
            delay = self.base_delay * fib[attempt]
        else:
            delay = self.base_delay

        # Apply max delay cap
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)  # Ensure non-negative

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if exception should be retried.

        Args:
            exception: Exception that occurred
            attempt: Current attempt number

        Returns:
            True if should retry, False otherwise
        """
        # Check max retries
        if attempt >= self.max_retries:
            return False

        # Check non-retryable exceptions
        if isinstance(exception, self.non_retryable_exceptions):
            return False

        # Check retryable exceptions (if specified)
        if self.retryable_exceptions is not None:
            return isinstance(exception, self.retryable_exceptions)

        # Default: retry all exceptions
        return True


# ==============================================================================
# Retry Decorator (Synchronous)
# ==============================================================================


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    config: Optional[RetryConfig] = None,
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        strategy: Backoff strategy
        jitter: Add randomness to delays
        retryable_exceptions: Specific exceptions to retry
        non_retryable_exceptions: Exceptions to fail fast
        config: RetryConfig object (overrides individual params)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry(max_retries=3, base_delay=1.0)
        ... def fetch_data():
        ...     # May raise transient errors
        ...     return api_client.get("/data")

        >>> # With specific exceptions
        >>> @retry(
        ...     max_retries=5,
        ...     retryable_exceptions=(ConnectionError, TimeoutError)
        ... )
        ... def connect_to_db():
        ...     return db.connect()
    """
    # Build config
    if config is None:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=strategy,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
            non_retryable_exceptions=non_retryable_exceptions or (
                KeyboardInterrupt, SystemExit
            ),
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            last_error = None

            while attempt <= config.max_retries:
                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Success - log if this was a retry
                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded on attempt "
                            f"{attempt + 1} after {attempt} retries"
                        )

                    return result

                except Exception as e:
                    last_error = e

                    # Check if should retry
                    if not config.should_retry(e, attempt):
                        if attempt >= config.max_retries:
                            logger.error(
                                f"Max retries ({config.max_retries}) exceeded "
                                f"for {func.__name__}"
                            )
                            if config.on_failure:
                                config.on_failure(attempt + 1, e)
                            raise MaxRetriesExceeded(attempt + 1, e)
                        else:
                            logger.error(
                                f"Non-retryable error in {func.__name__}: {e}"
                            )
                            raise

                    # Calculate delay
                    delay = config.calculate_delay(attempt)

                    # Log retry
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{config.max_retries} "
                        f"for {func.__name__} after error: {e}. "
                        f"Waiting {delay:.2f}s..."
                    )

                    # Callback
                    if config.on_retry:
                        config.on_retry(attempt + 1, e)

                    # Wait before retry
                    time.sleep(delay)
                    attempt += 1

            # Should never reach here, but just in case
            logger.error(f"Unexpected retry loop exit for {func.__name__}")
            if config.on_failure and last_error:
                config.on_failure(attempt, last_error)
            raise MaxRetriesExceeded(attempt, last_error or Exception("Unknown error"))

        return wrapper  # type: ignore

    return decorator


# ==============================================================================
# Async Retry Decorator
# ==============================================================================


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    jitter: bool = True,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    non_retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    config: Optional[RetryConfig] = None,
) -> Callable[[F], F]:
    """Async retry decorator with exponential backoff and jitter.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        strategy: Backoff strategy
        jitter: Add randomness to delays
        retryable_exceptions: Specific exceptions to retry
        non_retryable_exceptions: Exceptions to fail fast
        config: RetryConfig object (overrides individual params)

    Returns:
        Decorated async function with retry logic

    Example:
        >>> @async_retry(max_retries=3, base_delay=1.0)
        ... async def fetch_data():
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as response:
        ...             return await response.json()
    """
    # Build config
    if config is None:
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=strategy,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions,
            non_retryable_exceptions=non_retryable_exceptions or (
                KeyboardInterrupt, SystemExit
            ),
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            last_error = None

            while attempt <= config.max_retries:
                try:
                    # Execute async function
                    result = await func(*args, **kwargs)

                    # Success - log if this was a retry
                    if attempt > 0:
                        logger.info(
                            f"Async function {func.__name__} succeeded on attempt "
                            f"{attempt + 1} after {attempt} retries"
                        )

                    return result

                except Exception as e:
                    last_error = e

                    # Check if should retry
                    if not config.should_retry(e, attempt):
                        if attempt >= config.max_retries:
                            logger.error(
                                f"Max retries ({config.max_retries}) exceeded "
                                f"for {func.__name__}"
                            )
                            if config.on_failure:
                                config.on_failure(attempt + 1, e)
                            raise MaxRetriesExceeded(attempt + 1, e)
                        else:
                            logger.error(
                                f"Non-retryable error in {func.__name__}: {e}"
                            )
                            raise

                    # Calculate delay
                    delay = config.calculate_delay(attempt)

                    # Log retry
                    logger.warning(
                        f"Async retry attempt {attempt + 1}/{config.max_retries} "
                        f"for {func.__name__} after error: {e}. "
                        f"Waiting {delay:.2f}s..."
                    )

                    # Callback
                    if config.on_retry:
                        config.on_retry(attempt + 1, e)

                    # Wait before retry (async)
                    await asyncio.sleep(delay)
                    attempt += 1

            # Should never reach here, but just in case
            logger.error(f"Unexpected retry loop exit for {func.__name__}")
            if config.on_failure and last_error:
                config.on_failure(attempt, last_error)
            raise MaxRetriesExceeded(attempt, last_error or Exception("Unknown error"))

        return wrapper  # type: ignore

    return decorator


# ==============================================================================
# Common Retry Configurations
# ==============================================================================


# Quick retries for fast operations
QUICK_RETRY = RetryConfig(
    max_retries=3,
    base_delay=0.5,
    max_delay=5.0,
    strategy=RetryStrategy.EXPONENTIAL,
)

# Standard retries for most operations
STANDARD_RETRY = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL,
)

# Aggressive retries for critical operations
AGGRESSIVE_RETRY = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL,
)

# Network retries with common network exceptions
NETWORK_RETRY = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        OSError,
    ),
)

# Database retries
DATABASE_RETRY = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL,
)


__all__ = [
    "retry",
    "async_retry",
    "RetryConfig",
    "RetryStrategy",
    "RetryableError",
    "MaxRetriesExceeded",
    "QUICK_RETRY",
    "STANDARD_RETRY",
    "AGGRESSIVE_RETRY",
    "NETWORK_RETRY",
    "DATABASE_RETRY",
]
