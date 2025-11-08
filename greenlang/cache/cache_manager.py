"""
GreenLang Unified Cache Manager

Orchestrates the 3-layer cache hierarchy with intelligent routing,
cache warming, and coherence management.

Features:
- Unified API for L1 (memory) + L2 (Redis) + L3 (disk)
- Automatic cache hierarchy (try L1 → L2 → L3)
- Cache warming on startup and background refresh
- Distributed cache coherence via pub/sub
- Comprehensive analytics and monitoring
- Parallel writes to multiple layers
- Automatic fallback on layer failures

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Callable
from collections import defaultdict

from .architecture import (
    CacheArchitecture,
    CacheLayer,
    CacheKeyStrategy,
    CacheStrategy
)
from .l1_memory_cache import L1MemoryCache
from .l2_redis_cache import L2RedisCache
from .l3_disk_cache import L3DiskCache

logger = logging.getLogger(__name__)


@dataclass
class CacheOperationResult:
    """
    Result of a cache operation.

    Attributes:
        success: Whether operation succeeded
        value: Retrieved value (for get operations)
        layer: Layer where operation succeeded
        latency_ms: Operation latency in milliseconds
        error: Error message if failed
    """
    success: bool
    value: Optional[Any] = None
    layer: Optional[CacheLayer] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class CacheAnalytics:
    """
    Analytics data for cache operations.

    Tracks performance across all layers.
    """
    total_gets: int = 0
    total_sets: int = 0
    total_invalidations: int = 0

    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    total_misses: int = 0

    l1_latency_ms: List[float] = None
    l2_latency_ms: List[float] = None
    l3_latency_ms: List[float] = None

    warming_operations: int = 0
    coherence_invalidations: int = 0

    def __post_init__(self):
        if self.l1_latency_ms is None:
            self.l1_latency_ms = []
        if self.l2_latency_ms is None:
            self.l2_latency_ms = []
        if self.l3_latency_ms is None:
            self.l3_latency_ms = []

    def get_overall_hit_rate(self) -> float:
        """Calculate overall hit rate."""
        total = self.total_gets
        if total == 0:
            return 0.0
        hits = self.l1_hits + self.l2_hits + self.l3_hits
        return hits / total

    def get_layer_hit_rate(self, layer: CacheLayer) -> float:
        """Calculate hit rate for specific layer."""
        total = self.total_gets
        if total == 0:
            return 0.0

        if layer == CacheLayer.L1_MEMORY:
            return self.l1_hits / total
        elif layer == CacheLayer.L2_REDIS:
            return self.l2_hits / total
        elif layer == CacheLayer.L3_DISK:
            return self.l3_hits / total
        return 0.0

    def get_percentile(
        self,
        layer: CacheLayer,
        percentile: int
    ) -> float:
        """Get latency percentile for layer."""
        if layer == CacheLayer.L1_MEMORY:
            latencies = self.l1_latency_ms
        elif layer == CacheLayer.L2_REDIS:
            latencies = self.l2_latency_ms
        elif layer == CacheLayer.L3_DISK:
            latencies = self.l3_latency_ms
        else:
            return 0.0

        if not latencies:
            return 0.0

        sorted_lat = sorted(latencies)
        idx = int(len(sorted_lat) * percentile / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_gets": self.total_gets,
            "total_sets": self.total_sets,
            "total_invalidations": self.total_invalidations,
            "overall_hit_rate": self.get_overall_hit_rate(),
            "l1_hit_rate": self.get_layer_hit_rate(CacheLayer.L1_MEMORY),
            "l2_hit_rate": self.get_layer_hit_rate(CacheLayer.L2_REDIS),
            "l3_hit_rate": self.get_layer_hit_rate(CacheLayer.L3_DISK),
            "total_misses": self.total_misses,
            "l1_p99_ms": self.get_percentile(CacheLayer.L1_MEMORY, 99),
            "l2_p99_ms": self.get_percentile(CacheLayer.L2_REDIS, 99),
            "l3_p99_ms": self.get_percentile(CacheLayer.L3_DISK, 99),
            "warming_operations": self.warming_operations,
            "coherence_invalidations": self.coherence_invalidations,
        }


class CacheManager:
    """
    Unified cache manager orchestrating all cache layers.

    Provides a single interface to interact with the multi-layer cache
    hierarchy, handling automatic promotion/demotion, warming, and coherence.

    Example:
        >>> # Create with default architecture
        >>> manager = CacheManager.create_default()
        >>> await manager.start()
        >>>
        >>> # Get/Set operations
        >>> await manager.set("workflow:123", workflow_data, ttl=3600)
        >>> data = await manager.get("workflow:123")
        >>>
        >>> # Invalidation
        >>> await manager.invalidate("workflow:123")
        >>> await manager.invalidate_pattern("workflow:*")
        >>>
        >>> # Analytics
        >>> stats = await manager.get_analytics()
        >>> print(f"Hit rate: {stats['overall_hit_rate']:.2%}")
        >>>
        >>> await manager.stop()
    """

    def __init__(self, architecture: CacheArchitecture):
        """
        Initialize cache manager.

        Args:
            architecture: Cache architecture configuration
        """
        self._arch = architecture
        self._key_strategy = architecture.key_strategy

        # Cache layers
        self._l1: Optional[L1MemoryCache] = None
        self._l2: Optional[L2RedisCache] = None
        self._l3: Optional[L3DiskCache] = None

        # Analytics
        self._analytics = CacheAnalytics()

        # Cache warming
        self._warming_task: Optional[asyncio.Task] = None
        self._running = False

        # Access tracking for warming
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_lock = asyncio.Lock()

        logger.info("Initialized CacheManager")

    @classmethod
    def create_default(cls) -> "CacheManager":
        """
        Create cache manager with default architecture.

        Returns:
            CacheManager with production-ready defaults
        """
        arch = CacheArchitecture.create_default()
        return cls(arch)

    @classmethod
    def create_high_performance(cls) -> "CacheManager":
        """
        Create cache manager optimized for performance.

        Returns:
            CacheManager optimized for max performance
        """
        arch = CacheArchitecture.create_high_performance()
        return cls(arch)

    async def start(self) -> None:
        """Initialize and start all cache layers."""
        logger.info("Starting CacheManager...")

        # Validate architecture
        errors = self._arch.validate()
        if errors:
            raise ValueError(f"Invalid architecture: {errors}")

        # Initialize L1 (memory cache)
        if self._arch.enable_all_layers or self._arch.l1_config:
            self._l1 = L1MemoryCache(
                max_size_mb=self._arch.l1_config.max_size_bytes // (1024 * 1024),
                default_ttl_seconds=self._arch.l1_config.default_ttl_seconds,
                cleanup_interval_seconds=self._arch.l1_config.cleanup_interval_seconds,
                enable_metrics=self._arch.l1_config.metrics_enabled
            )
            await self._l1.start()
            logger.info("L1 memory cache started")

        # Initialize L2 (Redis cache)
        if self._arch.enable_all_layers or self._arch.l2_config:
            try:
                self._l2 = L2RedisCache(
                    host=self._arch.l2_config.redis_host,
                    port=self._arch.l2_config.redis_port,
                    db=self._arch.l2_config.redis_db,
                    password=self._arch.l2_config.redis_password,
                    pool_size=self._arch.l2_config.redis_pool_size,
                    default_ttl_seconds=self._arch.l2_config.default_ttl_seconds,
                    pubsub_channel=self._arch.coherence_config.pubsub_channel,
                    sentinel_enabled=self._arch.l2_config.redis_sentinel_enabled,
                    sentinel_hosts=self._arch.l2_config.redis_sentinel_hosts,
                    sentinel_service=self._arch.l2_config.redis_sentinel_service,
                    enable_metrics=self._arch.l2_config.metrics_enabled
                )

                # Register invalidation callback for coherence
                if self._arch.coherence_config.enabled:
                    self._l2.register_invalidation_callback(
                        self._handle_coherence_invalidation
                    )

                await self._l2.start()
                logger.info("L2 Redis cache started")
            except Exception as e:
                if not self._arch.fallback_on_error:
                    raise
                logger.warning(f"Failed to start L2 cache: {e}")
                self._l2 = None

        # Initialize L3 (disk cache)
        if self._arch.enable_all_layers or self._arch.l3_config:
            self._l3 = L3DiskCache(
                cache_dir=self._arch.l3_config.disk_cache_dir,
                max_size_gb=self._arch.l3_config.max_size_bytes // (1024 ** 3),
                default_ttl_seconds=self._arch.l3_config.default_ttl_seconds,
                compression_enabled=self._arch.l3_config.compression_enabled,
                checkpoint_interval_seconds=self._arch.l3_config.disk_checkpoint_interval,
                corruption_check=self._arch.l3_config.disk_corruption_check
            )
            await self._l3.start()
            logger.info("L3 disk cache started")

        # Start cache warming
        if self._arch.warming_config.enabled:
            self._running = True
            if self._arch.warming_config.on_startup:
                await self._warm_cache()

            if self._arch.warming_config.background_refresh:
                self._warming_task = asyncio.create_task(self._warming_loop())

        logger.info("CacheManager started successfully")

    async def stop(self) -> None:
        """Stop all cache layers."""
        logger.info("Stopping CacheManager...")

        self._running = False

        if self._warming_task:
            self._warming_task.cancel()
            try:
                await self._warming_task
            except asyncio.CancelledError:
                pass

        # Stop layers
        for cache in [self._l1, self._l2, self._l3]:
            if cache:
                try:
                    await cache.stop()
                except Exception as e:
                    logger.error(f"Error stopping cache layer: {e}")

        logger.info("CacheManager stopped")

    async def get(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get value from cache hierarchy.

        Tries L1 → L2 → L3 in order, promoting found values to faster layers.

        Args:
            key: Cache key
            namespace: Optional namespace

        Returns:
            Cached value or None if not found

        Example:
            >>> value = await manager.get("workflow:123", namespace="emissions")
        """
        start_time = time.perf_counter()
        self._analytics.total_gets += 1

        # Generate full key
        full_key = self._key_strategy.generate_key(key, namespace=namespace)

        # Track access for warming
        async with self._access_lock:
            self._access_counts[full_key] += 1

        # Try L1
        if self._l1:
            value = await self._l1.get(full_key)
            if value is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self._analytics.l1_hits += 1
                self._analytics.l1_latency_ms.append(latency_ms)
                return value

        # Try L2
        if self._l2:
            try:
                value = await self._l2.get(full_key)
                if value is not None:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._analytics.l2_hits += 1
                    self._analytics.l2_latency_ms.append(latency_ms)

                    # Promote to L1
                    if self._l1:
                        asyncio.create_task(
                            self._l1.set(
                                full_key,
                                value,
                                ttl=self._arch.l1_config.default_ttl_seconds
                            )
                        )

                    return value
            except Exception as e:
                if not self._arch.fallback_on_error:
                    raise
                logger.warning(f"L2 get failed: {e}")

        # Try L3
        if self._l3:
            value = await self._l3.get(full_key)
            if value is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self._analytics.l3_hits += 1
                self._analytics.l3_latency_ms.append(latency_ms)

                # Promote to L1 and L2
                if self._arch.parallel_writes:
                    tasks = []
                    if self._l1:
                        tasks.append(
                            self._l1.set(
                                full_key,
                                value,
                                ttl=self._arch.l1_config.default_ttl_seconds
                            )
                        )
                    if self._l2:
                        tasks.append(
                            self._l2.set(
                                full_key,
                                value,
                                ttl=self._arch.l2_config.default_ttl_seconds
                            )
                        )
                    if tasks:
                        asyncio.gather(*tasks, return_exceptions=True)

                return value

        # Cache miss
        self._analytics.total_misses += 1
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
        layers: Optional[List[CacheLayer]] = None
    ) -> bool:
        """
        Set value in cache.

        Writes to all enabled layers (or specified layers) in parallel.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (uses layer defaults if None)
            namespace: Optional namespace
            layers: Optional list of specific layers to write to

        Returns:
            True if at least one write succeeded

        Example:
            >>> await manager.set("workflow:123", data, ttl=7200)
        """
        self._analytics.total_sets += 1

        # Generate full key
        full_key = self._key_strategy.generate_key(key, namespace=namespace)

        # Determine which layers to write to
        write_layers = layers if layers else [
            CacheLayer.L1_MEMORY,
            CacheLayer.L2_REDIS,
            CacheLayer.L3_DISK
        ]

        tasks = []
        success = False

        # L1
        if CacheLayer.L1_MEMORY in write_layers and self._l1:
            layer_ttl = ttl if ttl else self._arch.l1_config.default_ttl_seconds
            tasks.append(self._l1.set(full_key, value, ttl=layer_ttl))

        # L2
        if CacheLayer.L2_REDIS in write_layers and self._l2:
            layer_ttl = ttl if ttl else self._arch.l2_config.default_ttl_seconds
            try:
                tasks.append(self._l2.set(full_key, value, ttl=layer_ttl))
            except Exception as e:
                if not self._arch.fallback_on_error:
                    raise
                logger.warning(f"L2 set failed: {e}")

        # L3
        if CacheLayer.L3_DISK in write_layers and self._l3:
            layer_ttl = ttl if ttl else self._arch.l3_config.default_ttl_seconds
            tasks.append(self._l3.set(full_key, value, ttl=layer_ttl))

        # Execute writes
        if self._arch.parallel_writes and len(tasks) > 1:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success = any(r is True for r in results if not isinstance(r, Exception))
        else:
            for task in tasks:
                try:
                    if await task:
                        success = True
                except Exception as e:
                    logger.error(f"Error in cache set: {e}")

        return success

    async def invalidate(
        self,
        key: str,
        namespace: Optional[str] = None,
        publish: bool = True
    ) -> bool:
        """
        Invalidate (delete) a key from all cache layers.

        Args:
            key: Cache key
            namespace: Optional namespace
            publish: Whether to publish invalidation event

        Returns:
            True if at least one deletion succeeded

        Example:
            >>> await manager.invalidate("workflow:123")
        """
        self._analytics.total_invalidations += 1

        # Generate full key
        full_key = self._key_strategy.generate_key(key, namespace=namespace)

        tasks = []
        if self._l1:
            tasks.append(self._l1.delete(full_key))
        if self._l2:
            tasks.append(self._l2.delete(full_key))
        if self._l3:
            tasks.append(self._l3.delete(full_key))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        success = any(r is True for r in results if not isinstance(r, Exception))

        # Publish invalidation for coherence
        if publish and self._l2 and self._arch.coherence_config.enabled:
            try:
                await self._l2._publish_invalidation([full_key])
            except Exception as e:
                logger.error(f"Error publishing invalidation: {e}")

        return success

    async def invalidate_pattern(
        self,
        pattern: str,
        namespace: Optional[str] = None
    ) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "workflow:*")
            namespace: Optional namespace

        Returns:
            Number of keys invalidated

        Example:
            >>> count = await manager.invalidate_pattern("workflow:123:*")
        """
        # Generate pattern with namespace
        if namespace:
            full_pattern = f"{self._key_strategy.key_prefix}:{namespace}:*:{pattern}"
        else:
            full_pattern = f"{self._key_strategy.key_prefix}:*:{pattern}"

        total_count = 0

        # L2 supports pattern matching
        if self._l2:
            try:
                count = await self._l2.delete_pattern(full_pattern)
                total_count += count
            except Exception as e:
                logger.error(f"Error in L2 pattern delete: {e}")

        # For L1 and L3, we'd need to iterate (not implemented here for brevity)
        # In production, you'd want to track keys per namespace

        self._analytics.total_invalidations += total_count
        return total_count

    async def _handle_coherence_invalidation(self, keys: List[str]) -> None:
        """
        Handle invalidation event from pub/sub.

        Args:
            keys: Keys to invalidate locally
        """
        self._analytics.coherence_invalidations += len(keys)

        # Invalidate in L1 (already invalidated in L2 by publisher)
        for key in keys:
            if self._l1:
                await self._l1.delete(key)
            if self._l3:
                await self._l3.delete(key)

        logger.debug(f"Coherence invalidation: {len(keys)} keys")

    async def _warm_cache(self) -> None:
        """Warm cache with frequently accessed data."""
        logger.info("Starting cache warming...")
        start_time = time.time()

        # Get top accessed keys
        async with self._access_lock:
            top_keys = sorted(
                self._access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self._arch.warming_config.warm_top_n_items]

        # Warm top keys (load from L3 to L2 and L1)
        for key, count in top_keys:
            if self._l3:
                value = await self._l3.get(key)
                if value is not None:
                    # Promote to faster layers
                    if self._l2:
                        await self._l2.set(
                            key,
                            value,
                            ttl=self._arch.l2_config.default_ttl_seconds
                        )
                    if self._l1:
                        await self._l1.set(
                            key,
                            value,
                            ttl=self._arch.l1_config.default_ttl_seconds
                        )

            self._analytics.warming_operations += 1

            # Check timeout
            if (time.time() - start_time) > self._arch.warming_config.max_startup_time_seconds:
                break

        elapsed = time.time() - start_time
        logger.info(
            f"Cache warming completed: {self._analytics.warming_operations} "
            f"operations in {elapsed:.2f}s"
        )

    async def _warming_loop(self) -> None:
        """Background task for periodic cache warming."""
        while self._running:
            try:
                await asyncio.sleep(
                    self._arch.warming_config.refresh_interval_seconds
                )
                await self._warm_cache()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in warming loop: {e}", exc_info=True)

    async def get_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics across all layers.

        Returns:
            Dictionary with analytics data
        """
        analytics = self._analytics.to_dict()

        # Add per-layer stats
        if self._l1:
            analytics["l1"] = await self._l1.get_stats()

        if self._l2:
            analytics["l2"] = await self._l2.get_stats()

        if self._l3:
            analytics["l3"] = await self._l3.get_stats()

        # Add architecture summary
        analytics["architecture"] = self._arch.get_summary()

        return analytics

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all cache layers.

        Returns:
            Health status dictionary
        """
        health = {
            "healthy": True,
            "layers": {}
        }

        # Check L1
        if self._l1:
            try:
                await self._l1.set("__health__", "ok", ttl=60)
                val = await self._l1.get("__health__")
                health["layers"]["l1"] = {"healthy": val == "ok"}
            except Exception as e:
                health["layers"]["l1"] = {"healthy": False, "error": str(e)}
                health["healthy"] = False

        # Check L2
        if self._l2:
            try:
                await self._l2.set("__health__", "ok", ttl=60)
                val = await self._l2.get("__health__")
                health["layers"]["l2"] = {"healthy": val == "ok"}
            except Exception as e:
                health["layers"]["l2"] = {"healthy": False, "error": str(e)}
                if not self._arch.fallback_on_error:
                    health["healthy"] = False

        # Check L3
        if self._l3:
            try:
                await self._l3.set("__health__", "ok", ttl=60)
                val = await self._l3.get("__health__")
                health["layers"]["l3"] = {"healthy": val == "ok"}
            except Exception as e:
                health["layers"]["l3"] = {"healthy": False, "error": str(e)}
                health["healthy"] = False

        return health


# Global cache manager instance
_global_manager: Optional[CacheManager] = None


def get_cache_manager() -> Optional[CacheManager]:
    """Get global cache manager instance."""
    return _global_manager


async def initialize_cache_manager(
    architecture: Optional[CacheArchitecture] = None
) -> CacheManager:
    """
    Initialize global cache manager.

    Args:
        architecture: Optional custom architecture (uses default if None)

    Returns:
        Initialized CacheManager instance

    Example:
        >>> manager = await initialize_cache_manager()
        >>> await manager.set("key", "value")
    """
    global _global_manager

    if _global_manager is not None:
        await _global_manager.stop()

    if architecture is None:
        architecture = CacheArchitecture.create_default()

    _global_manager = CacheManager(architecture)
    await _global_manager.start()

    return _global_manager
