"""
greenlang/cache

Caching utilities for GreenLang Framework

Provides:
- EmissionFactorCache: LRU cache with TTL for emission factors
- get_global_cache: Global cache instance accessor
"""

from .emission_factor_cache import (
    EmissionFactorCache,
    CacheEntry,
    get_global_cache,
    reset_global_cache,
)

__all__ = [
    "EmissionFactorCache",
    "CacheEntry",
    "get_global_cache",
    "reset_global_cache",
]
