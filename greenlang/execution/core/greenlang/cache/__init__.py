# -*- coding: utf-8 -*-
"""
GreenLang Cache Module

Provides Redis-based caching for emission factors and other data.
Implements cache-aside pattern with TTL management and automatic fallback.
"""

from .redis_client import (
    RedisCache,
    RedisCacheConfig,
    CacheEntry,
    get_cache,
    cache_emission_factor,
    get_cached_emission_factor,
    invalidate_emission_factor,
)

__all__ = [
    "RedisCache",
    "RedisCacheConfig",
    "CacheEntry",
    "get_cache",
    "cache_emission_factor",
    "get_cached_emission_factor",
    "invalidate_emission_factor",
]

__version__ = "1.0.0"
