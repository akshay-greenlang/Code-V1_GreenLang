"""
GreenLang Caching Infrastructure

This module provides a production-ready 4-tier caching system with Redis cluster support.

Components:
    - RedisManager: AsyncIO Redis client with connection pooling and failover
    - CacheManager: 4-tier caching system (L1-L4)
    - Cache decorators: Easy-to-use caching decorators

Example:
    >>> from greenlang_foundation.cache import CacheManager, cached
    >>> cache = CacheManager()
    >>> await cache.initialize()
    >>>
    >>> @cached(ttl=300)
    >>> async def expensive_operation(param: str) -> dict:
    >>>     return {"result": "value"}
"""

from .redis_manager import RedisManager, RedisConfig, RedisHealthStatus
from .cache_manager import (
    CacheManager,
    CacheConfig,
    CacheTier,
    CacheStats,
    cached,
    cached_with_invalidation,
)

__all__ = [
    "RedisManager",
    "RedisConfig",
    "RedisHealthStatus",
    "CacheManager",
    "CacheConfig",
    "CacheTier",
    "CacheStats",
    "cached",
    "cached_with_invalidation",
]

__version__ = "1.0.0"
