"""
GreenLang Multi-Layer Cache Architecture

This module defines the comprehensive 3-layer caching architecture for GreenLang,
providing high-performance caching with content-addressable storage, cache coherence,
and intelligent cache warming strategies.

Architecture Overview:
    L1 Cache (Memory):  In-process LRU cache, 100MB limit, TTL 60s
    L2 Cache (Redis):   Distributed cache, 1GB limit, TTL 3600s
    L3 Cache (Disk):    Persistent cache for large artifacts, 10GB limit, TTL 86400s

Performance Targets:
    - Cache hit rate: >80% for L1+L2 combined
    - Cache latency: p99 < 10ms for L1, p99 < 50ms for L2
    - Cache coherence: <100ms propagation delay for invalidations
    - Cache warming: <5s startup time for critical data

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Protocol
from datetime import datetime, timedelta


class CacheLayer(Enum):
    """
    Cache layer enumeration.

    Defines the three cache layers in the hierarchy:
    - L1_MEMORY: Fastest, smallest, in-process
    - L2_REDIS: Fast, medium, distributed
    - L3_DISK: Slower, largest, persistent
    """
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


class CacheStrategy(Enum):
    """
    Cache population and eviction strategies.

    Strategies:
    - WRITE_THROUGH: Write to cache and storage simultaneously
    - WRITE_BACK: Write to cache first, then to storage asynchronously
    - WRITE_AROUND: Write to storage, bypass cache
    - READ_THROUGH: Read from cache, populate from storage on miss
    - CACHE_ASIDE: Application manages cache population
    """
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    CACHE_ASIDE = "cache_aside"


class EvictionPolicy(Enum):
    """
    Cache eviction policies.

    Policies:
    - LRU: Least Recently Used
    - LFU: Least Frequently Used
    - FIFO: First In First Out
    - TTL: Time To Live based
    - SIZE: Size-based eviction
    """
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    SIZE = "size"


class InvalidationStrategy(Enum):
    """
    Cache invalidation strategies.

    Strategies:
    - TTL_BASED: Automatic expiration based on TTL
    - EVENT_BASED: Invalidate on specific events (data updates)
    - VERSION_BASED: Invalidate on version mismatch
    - PATTERN_BASED: Bulk invalidation using key patterns
    - MANUAL: Explicit invalidation calls
    """
    TTL_BASED = "ttl_based"
    EVENT_BASED = "event_based"
    VERSION_BASED = "version_based"
    PATTERN_BASED = "pattern_based"
    MANUAL = "manual"


@dataclass
class CacheLayerConfig:
    """
    Configuration for a single cache layer.

    Attributes:
        layer: The cache layer (L1, L2, or L3)
        max_size_bytes: Maximum size in bytes
        default_ttl_seconds: Default time-to-live in seconds
        eviction_policy: Policy for evicting items
        compression_enabled: Whether to compress cached values
        encryption_enabled: Whether to encrypt cached values (for sensitive data)
        metrics_enabled: Whether to collect metrics
    """
    layer: CacheLayer
    max_size_bytes: int
    default_ttl_seconds: int
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    compression_enabled: bool = False
    encryption_enabled: bool = False
    metrics_enabled: bool = True

    # L1-specific settings
    cleanup_interval_seconds: int = 60  # Background cleanup frequency
    thread_safe: bool = True

    # L2-specific settings (Redis)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_pool_size: int = 50
    redis_sentinel_enabled: bool = False
    redis_sentinel_hosts: List[str] = field(default_factory=list)
    redis_sentinel_service: str = "mymaster"

    # L3-specific settings (Disk)
    disk_cache_dir: str = "~/.greenlang/cache"
    disk_checkpoint_interval: int = 300  # Checkpoint every 5 minutes
    disk_corruption_check: bool = True


@dataclass
class CacheKeyStrategy:
    """
    Strategy for generating cache keys.

    Uses content-addressable hashing to ensure:
    1. Deterministic key generation
    2. Collision resistance
    3. Efficient key comparison

    Attributes:
        hash_algorithm: Algorithm for hashing (default: sha256)
        include_version: Include version in key
        include_namespace: Include namespace in key
        key_prefix: Prefix for all keys (for namespacing)
        normalize_keys: Normalize keys before hashing
    """
    hash_algorithm: str = "sha256"
    include_version: bool = True
    include_namespace: bool = True
    key_prefix: str = "gl"
    normalize_keys: bool = True

    def generate_key(
        self,
        *args: Any,
        namespace: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate a content-addressable cache key.

        Args:
            *args: Positional arguments to include in key
            namespace: Optional namespace for the key
            version: Optional version for the key
            **kwargs: Keyword arguments to include in key

        Returns:
            A deterministic cache key string

        Example:
            >>> strategy = CacheKeyStrategy()
            >>> key = strategy.generate_key("workflow", 123, namespace="emissions")
            >>> # Returns: "gl:emissions:v1:abc123def456..."
        """
        # Build key components
        components = []

        # Add prefix
        if self.key_prefix:
            components.append(self.key_prefix)

        # Add namespace
        if self.include_namespace and namespace:
            components.append(namespace)

        # Add version
        if self.include_version:
            ver = version or "v1"
            components.append(ver)

        # Serialize arguments
        data = {
            "args": args,
            "kwargs": kwargs
        }

        # Normalize if needed
        if self.normalize_keys:
            serialized = json.dumps(data, sort_keys=True, default=str)
        else:
            serialized = json.dumps(data, default=str)

        # Hash the serialized data
        hasher = hashlib.new(self.hash_algorithm)
        hasher.update(serialized.encode('utf-8'))
        key_hash = hasher.hexdigest()[:16]  # Use first 16 chars for brevity

        # Combine components
        components.append(key_hash)
        return ":".join(components)


@dataclass
class CacheCoherenceConfig:
    """
    Configuration for cache coherence protocol.

    Ensures consistency across distributed cache instances using
    publish/subscribe pattern for invalidation events.

    Attributes:
        enabled: Whether coherence is enabled
        pubsub_channel: Redis pub/sub channel for invalidations
        propagation_timeout_ms: Max time for invalidation propagation
        retry_attempts: Number of retry attempts for failed invalidations
        batch_invalidations: Whether to batch invalidation events
        batch_size: Max size for batched invalidations
        batch_timeout_ms: Max wait time for batch
    """
    enabled: bool = True
    pubsub_channel: str = "gl:cache:invalidations"
    propagation_timeout_ms: int = 100
    retry_attempts: int = 3
    batch_invalidations: bool = True
    batch_size: int = 100
    batch_timeout_ms: int = 50


@dataclass
class CacheWarmingConfig:
    """
    Configuration for cache warming strategy.

    Pre-populates cache with frequently accessed data on startup
    or in the background to improve cache hit rates.

    Attributes:
        enabled: Whether cache warming is enabled
        on_startup: Warm cache on application startup
        background_refresh: Enable background cache refresh
        refresh_interval_seconds: Interval for background refresh
        max_startup_time_seconds: Max time for startup warming
        priority_items: List of high-priority items to warm first
        predictive_warming: Use ML to predict items to warm
    """
    enabled: bool = True
    on_startup: bool = True
    background_refresh: bool = True
    refresh_interval_seconds: int = 3600  # Refresh every hour
    max_startup_time_seconds: int = 5
    priority_items: List[str] = field(default_factory=list)
    predictive_warming: bool = False

    # Warming data sources
    warm_from_access_logs: bool = True
    warm_from_analytics: bool = True
    warm_top_n_items: int = 1000


class CacheMetricsCollector(Protocol):
    """
    Protocol for cache metrics collection.

    Implementations must provide methods to track:
    - Cache hits/misses
    - Latency
    - Size usage
    - Evictions
    """

    def record_hit(self, layer: CacheLayer, key: str, latency_ms: float) -> None:
        """Record a cache hit."""
        ...

    def record_miss(self, layer: CacheLayer, key: str) -> None:
        """Record a cache miss."""
        ...

    def record_set(self, layer: CacheLayer, key: str, size_bytes: int) -> None:
        """Record a cache set operation."""
        ...

    def record_eviction(self, layer: CacheLayer, key: str, reason: str) -> None:
        """Record a cache eviction."""
        ...

    def record_invalidation(self, layer: CacheLayer, keys: List[str]) -> None:
        """Record cache invalidation."""
        ...


@dataclass
class CacheArchitecture:
    """
    Complete cache architecture configuration.

    This is the main configuration class that brings together all
    cache layers, strategies, and policies.

    Attributes:
        l1_config: Configuration for L1 (memory) cache
        l2_config: Configuration for L2 (Redis) cache
        l3_config: Configuration for L3 (disk) cache
        key_strategy: Key generation strategy
        coherence_config: Cache coherence configuration
        warming_config: Cache warming configuration
        strategy: Overall caching strategy
        metrics_collector: Optional metrics collector
    """
    l1_config: CacheLayerConfig
    l2_config: CacheLayerConfig
    l3_config: CacheLayerConfig
    key_strategy: CacheKeyStrategy
    coherence_config: CacheCoherenceConfig
    warming_config: CacheWarmingConfig
    strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    metrics_collector: Optional[CacheMetricsCollector] = None

    # Global settings
    enable_all_layers: bool = True
    fallback_on_error: bool = True  # Fallback to next layer on error
    parallel_writes: bool = True  # Write to multiple layers in parallel

    @classmethod
    def create_default(cls) -> "CacheArchitecture":
        """
        Create default cache architecture with recommended settings.

        Returns:
            CacheArchitecture with production-ready defaults

        Example:
            >>> arch = CacheArchitecture.create_default()
            >>> print(arch.l1_config.max_size_bytes)
            104857600  # 100MB
        """
        # L1 Cache: 100MB, 60s TTL
        l1_config = CacheLayerConfig(
            layer=CacheLayer.L1_MEMORY,
            max_size_bytes=100 * 1024 * 1024,  # 100MB
            default_ttl_seconds=60,
            eviction_policy=EvictionPolicy.LRU,
            compression_enabled=False,  # No compression for speed
            encryption_enabled=False,
            metrics_enabled=True,
            cleanup_interval_seconds=60,
            thread_safe=True
        )

        # L2 Cache: 1GB, 1 hour TTL
        l2_config = CacheLayerConfig(
            layer=CacheLayer.L2_REDIS,
            max_size_bytes=1 * 1024 * 1024 * 1024,  # 1GB
            default_ttl_seconds=3600,  # 1 hour
            eviction_policy=EvictionPolicy.LRU,
            compression_enabled=True,  # Compress for network efficiency
            encryption_enabled=False,
            metrics_enabled=True,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_pool_size=50,
            redis_sentinel_enabled=False
        )

        # L3 Cache: 10GB, 24 hour TTL
        l3_config = CacheLayerConfig(
            layer=CacheLayer.L3_DISK,
            max_size_bytes=10 * 1024 * 1024 * 1024,  # 10GB
            default_ttl_seconds=86400,  # 24 hours
            eviction_policy=EvictionPolicy.LRU,
            compression_enabled=True,
            encryption_enabled=False,
            metrics_enabled=True,
            disk_cache_dir="~/.greenlang/cache",
            disk_checkpoint_interval=300,
            disk_corruption_check=True
        )

        # Key strategy
        key_strategy = CacheKeyStrategy(
            hash_algorithm="sha256",
            include_version=True,
            include_namespace=True,
            key_prefix="gl",
            normalize_keys=True
        )

        # Coherence config
        coherence_config = CacheCoherenceConfig(
            enabled=True,
            pubsub_channel="gl:cache:invalidations",
            propagation_timeout_ms=100,
            retry_attempts=3,
            batch_invalidations=True,
            batch_size=100,
            batch_timeout_ms=50
        )

        # Warming config
        warming_config = CacheWarmingConfig(
            enabled=True,
            on_startup=True,
            background_refresh=True,
            refresh_interval_seconds=3600,
            max_startup_time_seconds=5,
            priority_items=[],
            predictive_warming=False,
            warm_from_access_logs=True,
            warm_from_analytics=True,
            warm_top_n_items=1000
        )

        return cls(
            l1_config=l1_config,
            l2_config=l2_config,
            l3_config=l3_config,
            key_strategy=key_strategy,
            coherence_config=coherence_config,
            warming_config=warming_config,
            strategy=CacheStrategy.CACHE_ASIDE,
            enable_all_layers=True,
            fallback_on_error=True,
            parallel_writes=True
        )

    @classmethod
    def create_high_performance(cls) -> "CacheArchitecture":
        """
        Create high-performance cache architecture.

        Optimized for:
        - Maximum cache hit rate
        - Minimum latency
        - Aggressive caching

        Returns:
            CacheArchitecture optimized for performance
        """
        arch = cls.create_default()

        # Increase L1 cache size
        arch.l1_config.max_size_bytes = 500 * 1024 * 1024  # 500MB
        arch.l1_config.default_ttl_seconds = 300  # 5 minutes

        # Increase L2 cache size
        arch.l2_config.max_size_bytes = 5 * 1024 * 1024 * 1024  # 5GB
        arch.l2_config.default_ttl_seconds = 7200  # 2 hours

        # Enable predictive warming
        arch.warming_config.predictive_warming = True
        arch.warming_config.warm_top_n_items = 5000

        return arch

    @classmethod
    def create_memory_constrained(cls) -> "CacheArchitecture":
        """
        Create memory-constrained cache architecture.

        Optimized for:
        - Minimal memory footprint
        - Disk-heavy caching
        - Resource efficiency

        Returns:
            CacheArchitecture optimized for low memory
        """
        arch = cls.create_default()

        # Reduce L1 cache size
        arch.l1_config.max_size_bytes = 20 * 1024 * 1024  # 20MB
        arch.l1_config.default_ttl_seconds = 30

        # Reduce L2 cache size
        arch.l2_config.max_size_bytes = 100 * 1024 * 1024  # 100MB
        arch.l2_config.default_ttl_seconds = 1800  # 30 minutes

        # Increase L3 cache reliance
        arch.l3_config.max_size_bytes = 50 * 1024 * 1024 * 1024  # 50GB

        return arch

    def validate(self) -> List[str]:
        """
        Validate the cache architecture configuration.

        Returns:
            List of validation errors (empty if valid)

        Example:
            >>> arch = CacheArchitecture.create_default()
            >>> errors = arch.validate()
            >>> if not errors:
            ...     print("Configuration is valid")
        """
        errors = []

        # Validate layer sizes
        if self.l1_config.max_size_bytes <= 0:
            errors.append("L1 cache size must be positive")

        if self.l2_config.max_size_bytes <= 0:
            errors.append("L2 cache size must be positive")

        if self.l3_config.max_size_bytes <= 0:
            errors.append("L3 cache size must be positive")

        # Validate TTLs
        if self.l1_config.default_ttl_seconds <= 0:
            errors.append("L1 TTL must be positive")

        if self.l2_config.default_ttl_seconds <= 0:
            errors.append("L2 TTL must be positive")

        if self.l3_config.default_ttl_seconds <= 0:
            errors.append("L3 TTL must be positive")

        # Validate TTL hierarchy (L1 < L2 < L3)
        if self.l1_config.default_ttl_seconds >= self.l2_config.default_ttl_seconds:
            errors.append("L1 TTL should be less than L2 TTL")

        if self.l2_config.default_ttl_seconds >= self.l3_config.default_ttl_seconds:
            errors.append("L2 TTL should be less than L3 TTL")

        # Validate Redis config if L2 is enabled
        if self.enable_all_layers:
            if not self.l2_config.redis_host:
                errors.append("Redis host must be specified for L2 cache")

            if self.l2_config.redis_port <= 0:
                errors.append("Redis port must be positive")

        # Validate coherence config
        if self.coherence_config.enabled:
            if self.coherence_config.propagation_timeout_ms <= 0:
                errors.append("Coherence propagation timeout must be positive")

        # Validate warming config
        if self.warming_config.enabled:
            if self.warming_config.max_startup_time_seconds <= 0:
                errors.append("Max startup time must be positive")

        return errors

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the cache architecture configuration.

        Returns:
            Dictionary with configuration summary
        """
        return {
            "layers": {
                "l1": {
                    "size_mb": self.l1_config.max_size_bytes / (1024 * 1024),
                    "ttl_seconds": self.l1_config.default_ttl_seconds,
                    "policy": self.l1_config.eviction_policy.value
                },
                "l2": {
                    "size_mb": self.l2_config.max_size_bytes / (1024 * 1024),
                    "ttl_seconds": self.l2_config.default_ttl_seconds,
                    "policy": self.l2_config.eviction_policy.value,
                    "redis_host": self.l2_config.redis_host
                },
                "l3": {
                    "size_gb": self.l3_config.max_size_bytes / (1024 * 1024 * 1024),
                    "ttl_seconds": self.l3_config.default_ttl_seconds,
                    "policy": self.l3_config.eviction_policy.value,
                    "disk_dir": self.l3_config.disk_cache_dir
                }
            },
            "strategy": self.strategy.value,
            "coherence_enabled": self.coherence_config.enabled,
            "warming_enabled": self.warming_config.enabled,
            "all_layers_enabled": self.enable_all_layers
        }
