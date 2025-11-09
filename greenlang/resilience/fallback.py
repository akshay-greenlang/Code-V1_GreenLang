"""Fallback Mechanisms and Graceful Degradation.

Production-grade fallback patterns for resilient systems:
- Fallback to cached data when API unavailable
- Fallback to lower-tier calculation when primary fails
- Default responses for non-critical failures
- Chained fallback strategies

Inspired by Netflix Hystrix fallback patterns.

Author: GreenLang Resilience Team
Date: November 2025
"""

import functools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# ==============================================================================
# Fallback Strategies
# ==============================================================================


class FallbackStrategy(str, Enum):
    """Fallback strategy types."""
    CACHED = "cached"  # Return cached value
    DEFAULT = "default"  # Return default value
    FUNCTION = "function"  # Call fallback function
    NONE = "none"  # Return None
    RAISE = "raise"  # Re-raise the exception


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior.

    Attributes:
        strategy: Fallback strategy to use
        default_value: Default value to return (for DEFAULT strategy)
        fallback_function: Function to call (for FUNCTION strategy)
        cache_key_func: Function to generate cache key (for CACHED strategy)
        log_fallback: Log when fallback is used (default: True)
        on_fallback: Callback when fallback is triggered
        fallback_exceptions: Exceptions that trigger fallback (default: all)
    """
    strategy: FallbackStrategy = FallbackStrategy.DEFAULT
    default_value: Any = None
    fallback_function: Optional[Callable[..., Any]] = None
    cache_key_func: Optional[Callable[..., str]] = None
    log_fallback: bool = True
    on_fallback: Optional[Callable[[Exception, Any], None]] = None
    fallback_exceptions: Optional[tuple] = None


# ==============================================================================
# Cached Fallback Handler
# ==============================================================================


class CachedFallback:
    """Fallback handler using cache.

    Maintains an in-memory cache of successful results and returns
    cached values when operations fail.
    """

    def __init__(self, max_cache_size: int = 1000):
        """Initialize cached fallback.

        Args:
            max_cache_size: Maximum number of cached items
        """
        self._cache: Dict[str, Any] = {}
        self._max_cache_size = max_cache_size

    def set(self, key: str, value: Any) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Simple LRU eviction
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        return self._cache.get(key, default)

    def has(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        return key in self._cache

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def size(self) -> int:
        """Get cache size.

        Returns:
            Number of cached items
        """
        return len(self._cache)


# Global cached fallback instance
_global_cached_fallback = CachedFallback()


def get_cached_fallback() -> CachedFallback:
    """Get global cached fallback instance.

    Returns:
        CachedFallback instance
    """
    return _global_cached_fallback


# ==============================================================================
# Default Fallback Handler
# ==============================================================================


class DefaultFallback:
    """Fallback handler using default values.

    Provides sensible defaults for different data types.
    """

    @staticmethod
    def for_type(value_type: type) -> Any:
        """Get default value for type.

        Args:
            value_type: Type to get default for

        Returns:
            Default value for type
        """
        defaults = {
            int: 0,
            float: 0.0,
            str: "",
            bool: False,
            list: [],
            dict: {},
            set: set(),
            tuple: (),
        }
        return defaults.get(value_type)

    @staticmethod
    def empty_result() -> Dict[str, Any]:
        """Get empty result dictionary.

        Returns:
            Empty result with metadata
        """
        return {
            "data": None,
            "status": "fallback",
            "error": None,
            "cached": False,
        }


# ==============================================================================
# Fallback Decorator (Synchronous)
# ==============================================================================


def fallback(
    strategy: FallbackStrategy = FallbackStrategy.DEFAULT,
    default_value: Any = None,
    fallback_function: Optional[Callable[..., Any]] = None,
    cache_key_func: Optional[Callable[..., str]] = None,
    log_fallback: bool = True,
    fallback_exceptions: Optional[tuple] = None,
    config: Optional[FallbackConfig] = None,
) -> Callable[[F], F]:
    """Fallback decorator for graceful degradation.

    Args:
        strategy: Fallback strategy to use
        default_value: Default value (for DEFAULT strategy)
        fallback_function: Fallback function (for FUNCTION strategy)
        cache_key_func: Cache key generator (for CACHED strategy)
        log_fallback: Log fallback usage
        fallback_exceptions: Specific exceptions that trigger fallback
        config: FallbackConfig object (overrides individual params)

    Returns:
        Decorated function with fallback logic

    Example:
        >>> @fallback(strategy=FallbackStrategy.DEFAULT, default_value=[])
        ... def get_suppliers():
        ...     return api.fetch_suppliers()

        >>> @fallback(
        ...     strategy=FallbackStrategy.FUNCTION,
        ...     fallback_function=lambda: get_suppliers_from_cache()
        ... )
        ... def get_live_suppliers():
        ...     return api.fetch_live_suppliers()

        >>> @fallback(strategy=FallbackStrategy.CACHED)
        ... def get_emission_factors(category: str):
        ...     return factor_api.get(category)
    """
    # Build config
    if config is None:
        config = FallbackConfig(
            strategy=strategy,
            default_value=default_value,
            fallback_function=fallback_function,
            cache_key_func=cache_key_func,
            log_fallback=log_fallback,
            fallback_exceptions=fallback_exceptions,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Execute function
                result = func(*args, **kwargs)

                # Cache successful result if using CACHED strategy
                if config.strategy == FallbackStrategy.CACHED:
                    cache_key = _generate_cache_key(
                        func, args, kwargs, config.cache_key_func
                    )
                    _global_cached_fallback.set(cache_key, result)

                return result

            except Exception as e:
                # Check if should apply fallback
                should_fallback = (
                    config.fallback_exceptions is None or
                    isinstance(e, config.fallback_exceptions)
                )

                if not should_fallback:
                    raise

                # Log fallback
                if config.log_fallback:
                    logger.warning(
                        f"Fallback triggered for {func.__name__} "
                        f"due to {type(e).__name__}: {e}. "
                        f"Using strategy: {config.strategy}"
                    )

                # Apply fallback strategy
                fallback_result = _apply_fallback_strategy(
                    func, args, kwargs, config, e
                )

                # Callback
                if config.on_fallback:
                    config.on_fallback(e, fallback_result)

                return fallback_result

        return wrapper  # type: ignore

    return decorator


# ==============================================================================
# Async Fallback Decorator
# ==============================================================================


def async_fallback(
    strategy: FallbackStrategy = FallbackStrategy.DEFAULT,
    default_value: Any = None,
    fallback_function: Optional[Callable[..., Any]] = None,
    cache_key_func: Optional[Callable[..., str]] = None,
    log_fallback: bool = True,
    fallback_exceptions: Optional[tuple] = None,
    config: Optional[FallbackConfig] = None,
) -> Callable[[F], F]:
    """Async fallback decorator for graceful degradation.

    Args:
        strategy: Fallback strategy to use
        default_value: Default value (for DEFAULT strategy)
        fallback_function: Fallback function (for FUNCTION strategy)
        cache_key_func: Cache key generator (for CACHED strategy)
        log_fallback: Log fallback usage
        fallback_exceptions: Specific exceptions that trigger fallback
        config: FallbackConfig object (overrides individual params)

    Returns:
        Decorated async function with fallback logic

    Example:
        >>> @async_fallback(strategy=FallbackStrategy.DEFAULT, default_value=[])
        ... async def get_suppliers():
        ...     return await api.fetch_suppliers()
    """
    # Build config
    if config is None:
        config = FallbackConfig(
            strategy=strategy,
            default_value=default_value,
            fallback_function=fallback_function,
            cache_key_func=cache_key_func,
            log_fallback=log_fallback,
            fallback_exceptions=fallback_exceptions,
        )

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Execute async function
                result = await func(*args, **kwargs)

                # Cache successful result if using CACHED strategy
                if config.strategy == FallbackStrategy.CACHED:
                    cache_key = _generate_cache_key(
                        func, args, kwargs, config.cache_key_func
                    )
                    _global_cached_fallback.set(cache_key, result)

                return result

            except Exception as e:
                # Check if should apply fallback
                should_fallback = (
                    config.fallback_exceptions is None or
                    isinstance(e, config.fallback_exceptions)
                )

                if not should_fallback:
                    raise

                # Log fallback
                if config.log_fallback:
                    logger.warning(
                        f"Fallback triggered for async {func.__name__} "
                        f"due to {type(e).__name__}: {e}. "
                        f"Using strategy: {config.strategy}"
                    )

                # Apply fallback strategy (handle async fallback functions)
                fallback_result = await _apply_async_fallback_strategy(
                    func, args, kwargs, config, e
                )

                # Callback
                if config.on_fallback:
                    config.on_fallback(e, fallback_result)

                return fallback_result

        return wrapper  # type: ignore

    return decorator


# ==============================================================================
# Helper Functions
# ==============================================================================


def _generate_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    cache_key_func: Optional[Callable[..., str]],
) -> str:
    """Generate cache key for function call.

    Args:
        func: Function being called
        args: Positional arguments
        kwargs: Keyword arguments
        cache_key_func: Custom cache key function

    Returns:
        Cache key string
    """
    if cache_key_func:
        return cache_key_func(*args, **kwargs)

    # Default: use function name and string representation of args
    func_name = func.__name__
    args_str = "_".join(str(arg) for arg in args)
    kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    return f"{func_name}_{args_str}_{kwargs_str}"


def _apply_fallback_strategy(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: FallbackConfig,
    error: Exception,
) -> Any:
    """Apply fallback strategy.

    Args:
        func: Original function
        args: Original args
        kwargs: Original kwargs
        config: Fallback config
        error: Exception that triggered fallback

    Returns:
        Fallback value
    """
    if config.strategy == FallbackStrategy.DEFAULT:
        return config.default_value

    elif config.strategy == FallbackStrategy.CACHED:
        cache_key = _generate_cache_key(func, args, kwargs, config.cache_key_func)
        if _global_cached_fallback.has(cache_key):
            logger.info(f"Using cached value for {func.__name__}")
            return _global_cached_fallback.get(cache_key)
        else:
            logger.warning(
                f"No cached value available for {func.__name__}, "
                f"returning default"
            )
            return config.default_value

    elif config.strategy == FallbackStrategy.FUNCTION:
        if config.fallback_function is None:
            raise ValueError("fallback_function required for FUNCTION strategy")
        try:
            return config.fallback_function(*args, **kwargs)
        except Exception as fb_error:
            logger.error(
                f"Fallback function failed for {func.__name__}: {fb_error}"
            )
            return config.default_value

    elif config.strategy == FallbackStrategy.NONE:
        return None

    elif config.strategy == FallbackStrategy.RAISE:
        raise error

    else:
        logger.error(f"Unknown fallback strategy: {config.strategy}")
        return config.default_value


async def _apply_async_fallback_strategy(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: FallbackConfig,
    error: Exception,
) -> Any:
    """Apply async fallback strategy.

    Args:
        func: Original function
        args: Original args
        kwargs: Original kwargs
        config: Fallback config
        error: Exception that triggered fallback

    Returns:
        Fallback value
    """
    if config.strategy == FallbackStrategy.DEFAULT:
        return config.default_value

    elif config.strategy == FallbackStrategy.CACHED:
        cache_key = _generate_cache_key(func, args, kwargs, config.cache_key_func)
        if _global_cached_fallback.has(cache_key):
            logger.info(f"Using cached value for {func.__name__}")
            return _global_cached_fallback.get(cache_key)
        else:
            logger.warning(
                f"No cached value available for {func.__name__}, "
                f"returning default"
            )
            return config.default_value

    elif config.strategy == FallbackStrategy.FUNCTION:
        if config.fallback_function is None:
            raise ValueError("fallback_function required for FUNCTION strategy")
        try:
            # Check if fallback function is async
            import inspect
            if inspect.iscoroutinefunction(config.fallback_function):
                return await config.fallback_function(*args, **kwargs)
            else:
                return config.fallback_function(*args, **kwargs)
        except Exception as fb_error:
            logger.error(
                f"Fallback function failed for {func.__name__}: {fb_error}"
            )
            return config.default_value

    elif config.strategy == FallbackStrategy.NONE:
        return None

    elif config.strategy == FallbackStrategy.RAISE:
        raise error

    else:
        logger.error(f"Unknown fallback strategy: {config.strategy}")
        return config.default_value


__all__ = [
    "fallback",
    "async_fallback",
    "FallbackStrategy",
    "FallbackConfig",
    "CachedFallback",
    "DefaultFallback",
    "get_cached_fallback",
]
