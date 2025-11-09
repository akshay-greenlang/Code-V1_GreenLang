"""Timeout Management for Different Operation Types.

Production-grade timeout decorator with operation-specific configurations:
- Factor lookup: 5 seconds
- LLM inference: 30 seconds
- ERP API calls: 10 seconds
- Database queries: 10 seconds
- Report generation: 60 seconds

Supports both synchronous and asynchronous operations.

Author: GreenLang Resilience Team
Date: November 2025
"""

import asyncio
import functools
import logging
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ==============================================================================
# Exceptions
# ==============================================================================


class TimeoutError(Exception):
    """Raised when operation times out."""

    def __init__(self, operation: str, timeout_seconds: float):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds}s"
        )


# ==============================================================================
# Configuration
# ==============================================================================


class OperationType(str, Enum):
    """Standard operation types with predefined timeouts."""
    FACTOR_LOOKUP = "factor_lookup"
    LLM_INFERENCE = "llm_inference"
    ERP_API_CALL = "erp_api_call"
    DATABASE_QUERY = "database_query"
    REPORT_GENERATION = "report_generation"
    FILE_UPLOAD = "file_upload"
    EXTERNAL_API = "external_api"
    CACHE_OPERATION = "cache_operation"
    COMPUTATION = "computation"
    CUSTOM = "custom"


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior.

    Attributes:
        timeout_seconds: Timeout duration in seconds
        operation_type: Type of operation (optional)
        raise_on_timeout: Raise exception on timeout (default: True)
        timeout_callback: Callback function on timeout (optional)
    """
    timeout_seconds: float
    operation_type: Optional[OperationType] = None
    raise_on_timeout: bool = True
    timeout_callback: Optional[Callable[[str, float], None]] = None


# Default timeout values by operation type
DEFAULT_TIMEOUTS: Dict[OperationType, float] = {
    OperationType.FACTOR_LOOKUP: 5.0,
    OperationType.LLM_INFERENCE: 30.0,
    OperationType.ERP_API_CALL: 10.0,
    OperationType.DATABASE_QUERY: 10.0,
    OperationType.REPORT_GENERATION: 60.0,
    OperationType.FILE_UPLOAD: 30.0,
    OperationType.EXTERNAL_API: 15.0,
    OperationType.CACHE_OPERATION: 2.0,
    OperationType.COMPUTATION: 20.0,
    OperationType.CUSTOM: 10.0,
}


def get_timeout_for_operation(operation_type: OperationType) -> float:
    """Get default timeout for operation type.

    Args:
        operation_type: Type of operation

    Returns:
        Timeout in seconds
    """
    return DEFAULT_TIMEOUTS.get(operation_type, 10.0)


# ==============================================================================
# Timeout Context Manager (Synchronous)
# ==============================================================================


@contextmanager
def timeout_context(timeout_seconds: float, operation_name: str = "operation"):
    """Context manager for timeout.

    Args:
        timeout_seconds: Timeout duration
        operation_name: Name of operation for error messages

    Raises:
        TimeoutError: If operation times out

    Example:
        >>> with timeout_context(5.0, "database_query"):
        ...     result = db.execute(query)
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(operation_name, timeout_seconds)

    # Set up signal handler (UNIX only)
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            yield
        finally:
            # Cancel alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    except AttributeError:
        # Windows doesn't support SIGALRM, fall back to no timeout
        logger.warning(
            f"Timeout not supported on this platform for {operation_name}"
        )
        yield


# ==============================================================================
# Timeout Decorator (Synchronous)
# ==============================================================================


def timeout(
    timeout_seconds: Optional[float] = None,
    operation_type: Optional[OperationType] = None,
    raise_on_timeout: bool = True,
    config: Optional[TimeoutConfig] = None,
) -> Callable[[F], F]:
    """Timeout decorator for synchronous functions.

    Args:
        timeout_seconds: Timeout duration (required if operation_type not set)
        operation_type: Operation type for default timeout
        raise_on_timeout: Raise exception on timeout
        config: TimeoutConfig object (overrides individual params)

    Returns:
        Decorated function with timeout

    Example:
        >>> @timeout(timeout_seconds=5.0)
        ... def fetch_data():
        ...     return requests.get(url).json()

        >>> @timeout(operation_type=OperationType.DATABASE_QUERY)
        ... def query_database():
        ...     return db.execute(query)
    """
    # Build config
    if config is None:
        if timeout_seconds is None and operation_type is None:
            raise ValueError("Must specify timeout_seconds or operation_type")

        actual_timeout = timeout_seconds or get_timeout_for_operation(
            operation_type  # type: ignore
        )

        config = TimeoutConfig(
            timeout_seconds=actual_timeout,
            operation_type=operation_type,
            raise_on_timeout=raise_on_timeout,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            operation_name = f"{func.__name__}"

            try:
                with timeout_context(config.timeout_seconds, operation_name):
                    return func(*args, **kwargs)

            except TimeoutError as e:
                logger.error(
                    f"Timeout occurred for {operation_name} "
                    f"after {config.timeout_seconds}s"
                )

                if config.timeout_callback:
                    config.timeout_callback(operation_name, config.timeout_seconds)

                if config.raise_on_timeout:
                    raise
                else:
                    return None

        return wrapper  # type: ignore

    return decorator


# ==============================================================================
# Async Timeout Decorator
# ==============================================================================


def async_timeout(
    timeout_seconds: Optional[float] = None,
    operation_type: Optional[OperationType] = None,
    raise_on_timeout: bool = True,
    config: Optional[TimeoutConfig] = None,
) -> Callable[[F], F]:
    """Timeout decorator for async functions.

    Args:
        timeout_seconds: Timeout duration (required if operation_type not set)
        operation_type: Operation type for default timeout
        raise_on_timeout: Raise exception on timeout
        config: TimeoutConfig object (overrides individual params)

    Returns:
        Decorated async function with timeout

    Example:
        >>> @async_timeout(timeout_seconds=5.0)
        ... async def fetch_data():
        ...     async with aiohttp.ClientSession() as session:
        ...         async with session.get(url) as response:
        ...             return await response.json()

        >>> @async_timeout(operation_type=OperationType.LLM_INFERENCE)
        ... async def call_llm(prompt: str):
        ...     return await llm_client.generate(prompt)
    """
    # Build config
    if config is None:
        if timeout_seconds is None and operation_type is None:
            raise ValueError("Must specify timeout_seconds or operation_type")

        actual_timeout = timeout_seconds or get_timeout_for_operation(
            operation_type  # type: ignore
        )

        config = TimeoutConfig(
            timeout_seconds=actual_timeout,
            operation_type=operation_type,
            raise_on_timeout=raise_on_timeout,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            operation_name = f"{func.__name__}"

            try:
                # Use asyncio.wait_for for async timeout
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=config.timeout_seconds
                )

            except asyncio.TimeoutError:
                logger.error(
                    f"Timeout occurred for async {operation_name} "
                    f"after {config.timeout_seconds}s"
                )

                if config.timeout_callback:
                    config.timeout_callback(operation_name, config.timeout_seconds)

                if config.raise_on_timeout:
                    raise TimeoutError(operation_name, config.timeout_seconds)
                else:
                    return None

        return wrapper  # type: ignore

    return decorator


# ==============================================================================
# Common Timeout Configurations
# ==============================================================================


# Quick operations
QUICK_TIMEOUT = TimeoutConfig(
    timeout_seconds=2.0,
    operation_type=OperationType.CACHE_OPERATION,
)

# Standard operations
STANDARD_TIMEOUT = TimeoutConfig(
    timeout_seconds=10.0,
    operation_type=OperationType.EXTERNAL_API,
)

# Long operations
LONG_TIMEOUT = TimeoutConfig(
    timeout_seconds=60.0,
    operation_type=OperationType.REPORT_GENERATION,
)

# Factor lookup
FACTOR_LOOKUP_TIMEOUT = TimeoutConfig(
    timeout_seconds=5.0,
    operation_type=OperationType.FACTOR_LOOKUP,
)

# LLM inference
LLM_TIMEOUT = TimeoutConfig(
    timeout_seconds=30.0,
    operation_type=OperationType.LLM_INFERENCE,
)

# ERP API
ERP_TIMEOUT = TimeoutConfig(
    timeout_seconds=10.0,
    operation_type=OperationType.ERP_API_CALL,
)

# Database
DATABASE_TIMEOUT = TimeoutConfig(
    timeout_seconds=10.0,
    operation_type=OperationType.DATABASE_QUERY,
)


__all__ = [
    "timeout",
    "async_timeout",
    "timeout_context",
    "TimeoutConfig",
    "OperationType",
    "TimeoutError",
    "get_timeout_for_operation",
    "DEFAULT_TIMEOUTS",
    "QUICK_TIMEOUT",
    "STANDARD_TIMEOUT",
    "LONG_TIMEOUT",
    "FACTOR_LOOKUP_TIMEOUT",
    "LLM_TIMEOUT",
    "ERP_TIMEOUT",
    "DATABASE_TIMEOUT",
]
