"""
greenlang/cache

Advanced Multi-Layer Caching System for GreenLang Framework

Phase 5 Excellence - Infrastructure Team (TEAM 2)

Provides:
- Multi-layer cache architecture (L1 Memory, L2 Redis, L3 Disk)
- Cache management and orchestration
- Cache invalidation strategies
- EmissionFactorCache: LRU cache with TTL for emission factors (legacy)

Author: GreenLang Infrastructure Team
Date: 2025-11-08
Version: 5.0.0
"""

# Legacy emission factor cache
from .emission_factor_cache import (
    EmissionFactorCache,
    CacheEntry as LegacyCacheEntry,
    get_global_cache as get_legacy_cache,
    reset_global_cache as reset_legacy_cache,
)

# Phase 5: Advanced caching infrastructure
from .architecture import (
    CacheArchitecture,
    CacheLayer,
    CacheLayerConfig,
    CacheStrategy,
    EvictionPolicy,
    InvalidationStrategy,
    CacheKeyStrategy,
    CacheCoherenceConfig,
    CacheWarmingConfig,
)

from .l1_memory_cache import (
    L1MemoryCache,
    CacheEntry,
    CacheMetrics,
    cache_result,
    cache_with_key,
    get_global_cache as get_l1_cache,
    initialize_global_cache as initialize_l1_cache,
)

from .l2_redis_cache import (
    L2RedisCache,
    CircuitState,
    CircuitBreaker,
    RedisMetrics,
)

from .l3_disk_cache import (
    L3DiskCache,
    DiskCacheEntry,
)

from .cache_manager import (
    CacheManager,
    CacheAnalytics,
    CacheOperationResult,
    get_cache_manager,
    initialize_cache_manager,
)

from .invalidation import (
    InvalidationEvent,
    InvalidationRule,
    TTLInvalidationManager,
    EventBasedInvalidationManager,
    VersionBasedInvalidationManager,
    PatternBasedInvalidationManager,
    UnifiedInvalidationManager,
)

__all__ = [
    # Legacy
    "EmissionFactorCache",
    "LegacyCacheEntry",
    "get_legacy_cache",
    "reset_legacy_cache",

    # Architecture
    "CacheArchitecture",
    "CacheLayer",
    "CacheLayerConfig",
    "CacheStrategy",
    "EvictionPolicy",
    "InvalidationStrategy",
    "CacheKeyStrategy",
    "CacheCoherenceConfig",
    "CacheWarmingConfig",

    # L1 Memory Cache
    "L1MemoryCache",
    "CacheEntry",
    "CacheMetrics",
    "cache_result",
    "cache_with_key",
    "get_l1_cache",
    "initialize_l1_cache",

    # L2 Redis Cache
    "L2RedisCache",
    "CircuitState",
    "CircuitBreaker",
    "RedisMetrics",

    # L3 Disk Cache
    "L3DiskCache",
    "DiskCacheEntry",

    # Cache Manager
    "CacheManager",
    "CacheAnalytics",
    "CacheOperationResult",
    "get_cache_manager",
    "initialize_cache_manager",

    # Invalidation
    "InvalidationEvent",
    "InvalidationRule",
    "TTLInvalidationManager",
    "EventBasedInvalidationManager",
    "VersionBasedInvalidationManager",
    "PatternBasedInvalidationManager",
    "UnifiedInvalidationManager",
]

# Version info
__version__ = "5.0.0"
__author__ = "GreenLang Infrastructure Team"
