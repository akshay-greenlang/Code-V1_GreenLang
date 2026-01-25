# -*- coding: utf-8 -*-
# SAP Retry Logic
# Exponential backoff retry decorator for SAP API calls

"""
Retry Logic with Exponential Backoff
=====================================

Provides retry functionality with exponential backoff for handling transient failures
in SAP API calls.

Features:
---------
- Exponential backoff: 1s, 2s, 4s, 8s
- Configurable max retries (default: 4)
- Jitter to prevent thundering herd
- Retry on specific exception types
- Comprehensive retry attempt logging
- Async support

Usage:
------
```python
from connectors.sap.utils.retry_logic import retry_with_backoff
from greenlang.determinism import deterministic_random

@retry_with_backoff(max_retries=4, base_delay=1.0)
def call_sap_api():
    # Will retry on connection errors, timeouts, rate limits
    response = requests.get("https://sap.example.com/api/data")
    response.raise_for_status()
    return response.json()

# Custom retry conditions
@retry_with_backoff(
    max_retries=3,
    base_delay=2.0,
    retry_on=(ConnectionError, TimeoutError)
)
def custom_call():
    # Will only retry on ConnectionError and TimeoutError
    pass
```
"""

import functools
import logging
import random
import time
from typing import Callable, Optional, Tuple, Type, Union

import requests
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    Timeout,
    RequestException,
)

# Configure logger
logger = logging.getLogger(__name__)

# Default exceptions to retry on
DEFAULT_RETRY_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    Timeout,
    HTTPError,  # Will check for 429 and 5xx
    RequestException,
)


def retry_with_backoff(
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 4)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential calculation (default: 2.0)
        jitter: Add random jitter to prevent thundering herd (default: True)
        retry_on: Tuple of exception types to retry on (default: connection errors)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=2.0)
        >>> def api_call():
        >>>     return requests.get("https://api.example.com/data")
    """
    if retry_on is None:
        retry_on = DEFAULT_RETRY_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function with retry logic."""
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    # Attempt the function call
                    result = func(*args, **kwargs)

                    # If we're here, the call succeeded
                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded after {attempt} retries"
                        )
                    return result

                except retry_on as e:
                    last_exception = e

                    # Check if this is an HTTPError and should be retried
                    if isinstance(e, HTTPError):
                        status_code = getattr(e.response, 'status_code', None)
                        if status_code and not _should_retry_status(status_code):
                            # Don't retry on 4xx errors except 429
                            logger.warning(
                                f"Function {func.__name__} failed with non-retryable "
                                f"status code {status_code}"
                            )
                            raise

                    # If this was the last attempt, raise the exception
                    if attempt >= max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Last exception: {type(e).__name__}: {str(e)}"
                        )
                        raise

                    # Calculate backoff delay
                    delay = _calculate_backoff(
                        attempt=attempt,
                        base_delay=base_delay,
                        exponential_base=exponential_base,
                        max_delay=max_delay,
                        jitter=jitter,
                    )

                    # Log retry attempt
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Exception: {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    # Wait before retrying
                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def _calculate_backoff(
    attempt: int,
    base_delay: float,
    exponential_base: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """
    Calculate the backoff delay for a retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        exponential_base: Base for exponential calculation
        max_delay: Maximum allowed delay
        jitter: Whether to add random jitter

    Returns:
        Calculated delay in seconds
    """
    # Calculate exponential delay: base_delay * (exponential_base ^ attempt)
    delay = base_delay * (exponential_base ** attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter: random value between 0 and delay
    if jitter:
        delay = delay * (0.5 + deterministic_random().random() * 0.5)  # 50-100% of calculated delay

    return delay


def _should_retry_status(status_code: int) -> bool:
    """
    Determine if an HTTP status code should trigger a retry.

    Args:
        status_code: HTTP status code

    Returns:
        True if the status code should be retried, False otherwise
    """
    # Retry on:
    # - 429 Too Many Requests
    # - 500 Internal Server Error
    # - 502 Bad Gateway
    # - 503 Service Unavailable
    # - 504 Gateway Timeout
    return status_code in (429, 500, 502, 503, 504)


# Async version for async functions
async def retry_with_backoff_async(
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable:
    """
    Async version of retry_with_backoff decorator.

    Similar to retry_with_backoff but for async functions.
    """
    import asyncio

    if retry_on is None:
        retry_on = DEFAULT_RETRY_EXCEPTIONS

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """Async wrapper function with retry logic."""
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(
                            f"Function {func.__name__} succeeded after {attempt} retries"
                        )
                    return result

                except retry_on as e:
                    last_exception = e

                    if attempt >= max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Last exception: {type(e).__name__}: {str(e)}"
                        )
                        raise

                    delay = _calculate_backoff(
                        attempt=attempt,
                        base_delay=base_delay,
                        exponential_base=exponential_base,
                        max_delay=max_delay,
                        jitter=jitter,
                    )

                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Exception: {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper

    return decorator
