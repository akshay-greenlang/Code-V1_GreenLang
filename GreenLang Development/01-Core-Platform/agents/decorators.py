# -*- coding: utf-8 -*-
"""
GreenLang Agent Decorators
Decorators for enhancing agent functionality with caching, determinism, and tracing.
"""

from functools import wraps, lru_cache
from typing import Callable, Any, Optional, Dict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
import time
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# ============================================================================
# @deterministic DECORATOR
# ============================================================================

def deterministic(
    seed: Optional[int] = None,
    strict: bool = True
):
    """
    Decorator to ensure deterministic execution of functions.

    Enforces:
    - Consistent input → consistent output
    - Reproducible results
    - Hash verification of inputs and outputs

    Args:
        seed: Optional random seed to set
        strict: If True, raises error on non-deterministic operations

    Example:
        >>> @deterministic(seed=42)
        ... def calculate_emissions(energy_kwh, factor):
        ...     return energy_kwh * factor

        >>> # Always returns same result for same inputs
        >>> result1 = calculate_emissions(100, 0.5)
        >>> result2 = calculate_emissions(100, 0.5)
        >>> assert result1 == result2

    Raises:
        RuntimeError: If strict=True and non-deterministic behavior detected
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set random seed if provided
            if seed is not None:
                try:
                    import numpy as np
                    np.random.seed(seed)
                except ImportError:
                    pass

                try:
                    import random
                    random.seed(seed)
                except ImportError:
                    pass

            # Hash inputs
            input_hash = _hash_args(args, kwargs)

            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Hash output
            output_hash = _hash_value(result)

            # Log determinism info
            logger.debug(
                f"Deterministic execution: {func.__name__} "
                f"(input_hash={input_hash[:8]}, output_hash={output_hash[:8]}, "
                f"time={execution_time:.4f}s)"
            )

            # Attach metadata to result if possible
            if hasattr(result, '__dict__'):
                result._deterministic_metadata = {
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "seed": seed,
                    "execution_time": execution_time
                }

            return result

        # Mark function as deterministic
        wrapper._is_deterministic = True
        wrapper._deterministic_seed = seed

        return wrapper

    return decorator


# ============================================================================
# @cached DECORATOR
# ============================================================================

class TTLCache:
    """Simple Time-To-Live cache."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 128):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if DeterministicClock.now() < entry["expires_at"]:
                return entry["value"]
            else:
                # Expired, remove
                del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache with TTL."""
        # Implement simple LRU by removing oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["created_at"])
            del self.cache[oldest_key]

        self.cache[key] = {
            "value": value,
            "created_at": DeterministicClock.now(),
            "expires_at": DeterministicClock.now() + timedelta(seconds=self.ttl_seconds)
        }

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


# Global cache registry
_cache_registry: Dict[str, TTLCache] = {}


def cached(
    ttl_seconds: int = 3600,
    max_size: int = 128,
    cache_key_func: Optional[Callable] = None
):
    """
    Decorator to cache function results with Time-To-Live.

    Caches function results based on input arguments.
    Automatically expires cached values after TTL.

    Args:
        ttl_seconds: Time-to-live for cached values (default: 1 hour)
        max_size: Maximum cache size (default: 128)
        cache_key_func: Optional custom function to generate cache key

    Example:
        >>> @cached(ttl_seconds=300)  # Cache for 5 minutes
        ... def expensive_calculation(x, y):
        ...     time.sleep(1)  # Expensive operation
        ...     return x * y

        >>> # First call: takes 1 second
        >>> result1 = expensive_calculation(10, 20)

        >>> # Second call: returns immediately from cache
        >>> result2 = expensive_calculation(10, 20)

        >>> assert result1 == result2
    """

    def decorator(func: Callable) -> Callable:
        # Create cache for this function
        cache_name = f"{func.__module__}.{func.__name__}"
        if cache_name not in _cache_registry:
            _cache_registry[cache_name] = TTLCache(ttl_seconds, max_size)

        cache = _cache_registry[cache_name]

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = _hash_args(args, kwargs)

            # Check cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {func.__name__} (key={cache_key[:8]})")
                return cached_value

            # Cache miss - execute function
            logger.debug(f"Cache miss: {func.__name__} (key={cache_key[:8]})")
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result)

            return result

        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        wrapper._cache = cache

        return wrapper

    return decorator


# ============================================================================
# @traced DECORATOR (Integrates with Provenance)
# ============================================================================

def traced(
    save_path: Optional[str] = None,
    track_inputs: bool = True,
    track_outputs: bool = True
):
    """
    Decorator to automatically track provenance for agent methods.

    Integrates with the provenance framework to create audit trails.

    Args:
        save_path: Optional path to save provenance record
        track_inputs: Whether to track input arguments
        track_outputs: Whether to track output results

    Example:
        >>> from greenlang.agents import BaseAgent, AgentResult

        >>> class MyAgent(BaseAgent):
        ...     @traced(save_path="provenance.json")
        ...     def execute(self, input_data):
        ...         # Process data with automatic provenance tracking
        ...         return AgentResult(success=True, data={"result": 42})
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            from greenlang.provenance import ProvenanceContext

            # Create or get provenance context
            if not hasattr(self, '_provenance_context'):
                self._provenance_context = ProvenanceContext(
                    name=f"{self.__class__.__name__}.{func.__name__}"
                )

            ctx = self._provenance_context

            # Record start time
            start_time = datetime.now(timezone.utc)

            # Track inputs if enabled
            if track_inputs:
                ctx.metadata[f"{func.__name__}_inputs"] = {
                    "args": [str(arg) for arg in args],
                    "kwargs": {k: str(v) for k, v in kwargs.items()}
                }

            try:
                # Execute function
                result = func(self, *args, **kwargs)

                # Track outputs if enabled
                if track_outputs:
                    ctx.metadata[f"{func.__name__}_outputs"] = str(result)

                # Record execution
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                ctx.record_agent_execution(
                    agent_name=f"{self.__class__.__name__}.{func.__name__}",
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    metadata={
                        "status": "success",
                        "class": self.__class__.__name__,
                        "method": func.__name__
                    }
                )

                # Save if path provided
                if save_path:
                    ctx.finalize(output_path=save_path)

                return result

            except Exception as e:
                # Record error
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                ctx.record_agent_execution(
                    agent_name=f"{self.__class__.__name__}.{func.__name__}",
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    duration_seconds=duration,
                    metadata={
                        "status": "failed",
                        "class": self.__class__.__name__,
                        "method": func.__name__,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )

                # Re-raise
                raise

        return wrapper

    return decorator


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _hash_args(args: tuple, kwargs: dict) -> str:
    """
    Hash function arguments for caching/determinism.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Hash string
    """
    # Convert to hashable representation
    args_repr = [_serialize_value(arg) for arg in args]
    kwargs_repr = {k: _serialize_value(v) for k, v in sorted(kwargs.items())}

    combined = {
        "args": args_repr,
        "kwargs": kwargs_repr
    }

    # Hash as JSON
    json_str = json.dumps(combined, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _hash_value(value: Any) -> str:
    """
    Hash a single value.

    Args:
        value: Value to hash

    Returns:
        Hash string
    """
    serialized = _serialize_value(value)
    json_str = json.dumps(serialized, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _serialize_value(value: Any) -> Any:
    """
    Serialize a value for hashing.

    Args:
        value: Value to serialize

    Returns:
        Serializable representation
    """
    # Handle common types
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in sorted(value.items())}
    elif hasattr(value, 'to_dict'):
        return _serialize_value(value.to_dict())
    elif hasattr(value, '__dict__'):
        return _serialize_value(value.__dict__)
    else:
        # Fallback to string representation
        return str(value)


# ============================================================================
# DECORATOR UTILITIES
# ============================================================================

def clear_all_caches():
    """Clear all function caches created by @cached decorator."""
    for cache in _cache_registry.values():
        cache.clear()
    logger.info(f"Cleared {len(_cache_registry)} caches")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics for all caches.

    Returns:
        Dictionary mapping function name to cache stats
    """
    stats = {}
    for name, cache in _cache_registry.items():
        stats[name] = cache.get_stats()
    return stats
