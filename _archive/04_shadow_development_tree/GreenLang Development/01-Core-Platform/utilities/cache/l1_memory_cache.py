# -*- coding: utf-8 -*-
"""
GreenLang L1 Memory Cache Implementation

High-performance in-process memory cache using LRU eviction with TTL support.
Designed for ultra-low latency (<10ms p99) with thread-safe operations.

Features:
- LRU eviction with configurable size limit (100MB default)
- TTL-based expiration with background cleanup
- Thread-safe operations with asyncio support
- Prometheus metrics integration
- Decorator-based caching
- Size-aware eviction

Author: GreenLang Infrastructure Team (TEAM 2)
Date: 2025-11-08
Version: 5.0.0
"""

import asyncio
import functools
import logging
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union, Tuple
import pickle

try:
    from cachetools import LRUCache
except ImportError:
    # Fallback implementation if cachetools not available
    class LRUCache(OrderedDict):
        def __init__(self, maxsize):
            super().__init__()
            self.maxsize = maxsize

        def __setitem__(self, key, value):
            if key in self:
                self.move_to_end(key)
            super().__setitem__(key, value)
            if len(self) > self.maxsize:
                oldest = next(iter(self))
                del self[oldest]

        def __getitem__(self, key):
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

from .architecture import CacheLayer, CacheLayerConfig

# Type variable for generic caching
T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """
    Cache entry with metadata.

    Attributes:
        value: The cached value
        created_at: When the entry was created
        accessed_at: When the entry was last accessed
        ttl_seconds: Time-to-live in seconds
        size_bytes: Size of the entry in bytes
        access_count: Number of times accessed
    """
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: int
    size_bytes: int
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        return time.time() > (self.created_at + self.ttl_seconds)

    def update_access(self) -> None:
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheMetrics:
    """
    Thread-safe metrics collector for L1 cache.

    Tracks:
    - Hit/miss rates
    - Latency percentiles
    - Size usage
    - Eviction events
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._size_bytes = 0
        self._latencies_ms = []
        self._max_latencies = 10000  # Keep last 10k measurements

    def record_hit(self, latency_ms: float) -> None:
        """Record a cache hit."""
        with self._lock:
            self._hits += 1
            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > self._max_latencies:
                self._latencies_ms = self._latencies_ms[-self._max_latencies:]

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._misses += 1

    def record_set(self, size_bytes: int) -> None:
        """Record a cache set operation."""
        with self._lock:
            self._sets += 1
            self._size_bytes += size_bytes

    def record_eviction(self, size_bytes: int) -> None:
        """Record a cache eviction."""
        with self._lock:
            self._evictions += 1
            self._size_bytes -= size_bytes

    def get_hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0)."""
        with self._lock:
            total = self._hits + self._misses
            if total == 0:
                return 0.0
            return self._hits / total

    def get_latency_percentile(self, percentile: int) -> float:
        """
        Get latency percentile.

        Args:
            percentile: Percentile to calculate (e.g., 50, 95, 99)

        Returns:
            Latency in milliseconds
        """
        with self._lock:
            if not self._latencies_ms:
                return 0.0
            sorted_latencies = sorted(self._latencies_ms)
            index = int(len(sorted_latencies) * percentile / 100)
            return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    def get_stats(self) -> Dict[str, Any]:
        """Get all cache statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "hit_rate": self.get_hit_rate(),
                "size_bytes": self._size_bytes,
                "p50_latency_ms": self.get_latency_percentile(50),
                "p95_latency_ms": self.get_latency_percentile(95),
                "p99_latency_ms": self.get_latency_percentile(99),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._size_bytes = 0
            self._latencies_ms = []


class L1MemoryCache:
    """
    High-performance in-process memory cache.

    Features:
    - LRU eviction with size-based limits
    - TTL-based expiration
    - Thread-safe operations
    - Async/await interface
    - Background cleanup
    - Comprehensive metrics

    Example:
        >>> cache = L1MemoryCache(max_size_mb=100, default_ttl_seconds=60)
        >>> await cache.start()
        >>> await cache.set("key", "value", ttl=120)
        >>> value = await cache.get("key")
        >>> await cache.stop()
    """

    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_seconds: int = 60,
        cleanup_interval_seconds: int = 60,
        enable_metrics: bool = True
    ):
        """
        Initialize L1 memory cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl_seconds: Default TTL for cache entries
            cleanup_interval_seconds: Interval for background cleanup
            enable_metrics: Whether to enable metrics collection
        """
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._enable_metrics = enable_metrics

        # Thread-safe cache storage
        self._lock = threading.RLock()
        self._cache: Dict[str, CacheEntry] = {}
        self._current_size_bytes = 0

        # Metrics
        self._metrics = CacheMetrics() if enable_metrics else None

        # Background cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"Initialized L1 memory cache: "
            f"max_size={max_size_mb}MB, ttl={default_ttl_seconds}s"
        )

    async def start(self) -> None:
        """Start the cache and background cleanup task."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("L1 memory cache started")

    async def stop(self) -> None:
        """Stop the cache and cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("L1 memory cache stopped")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired

        Example:
            >>> value = await cache.get("my_key")
        """
        start_time = time.perf_counter()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                if self._metrics:
                    self._metrics.record_miss()
                return None

            # Check expiration
            if entry.is_expired():
                self._evict_entry(key, entry, reason="ttl_expired")
                if self._metrics:
                    self._metrics.record_miss()
                return None

            # Update access metadata
            entry.update_access()

            # Record metrics
            if self._metrics:
                latency_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.record_hit(latency_ms)

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (uses default if None)

        Example:
            >>> await cache.set("my_key", {"data": "value"}, ttl=120)
        """
        ttl_seconds = ttl if ttl is not None else self._default_ttl

        # Calculate entry size
        try:
            size_bytes = sys.getsizeof(value)
            # For complex objects, try pickle size
            if hasattr(value, '__dict__'):
                size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 1024  # Default estimate

        # Check if value is too large
        if size_bytes > self._max_size_bytes:
            logger.warning(
                f"Value too large for cache: {size_bytes} bytes "
                f"(max: {self._max_size_bytes})"
            )
            return

        with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_size_bytes -= old_entry.size_bytes

            # Evict entries if needed to make space
            while (
                self._current_size_bytes + size_bytes > self._max_size_bytes
                and self._cache
            ):
                self._evict_lru()

            # Create new entry
            now = time.time()
            entry = CacheEntry(
                value=value,
                created_at=now,
                accessed_at=now,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes,
                access_count=0
            )

            self._cache[key] = entry
            self._current_size_bytes += size_bytes

            if self._metrics:
                self._metrics.record_set(size_bytes)

    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False if not found

        Example:
            >>> deleted = await cache.delete("my_key")
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            self._evict_entry(key, entry, reason="manual_delete")
            return True

    async def clear(self) -> None:
        """
        Clear all entries from the cache.

        Example:
            >>> await cache.clear()
        """
        with self._lock:
            self._cache.clear()
            self._current_size_bytes = 0
            if self._metrics:
                self._metrics.reset()
        logger.info("L1 cache cleared")

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if exists and valid, False otherwise

        Example:
            >>> if await cache.exists("my_key"):
            ...     print("Key exists")
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._evict_entry(key, entry, reason="ttl_expired")
                return False
            return True

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats

        Example:
            >>> stats = await cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        with self._lock:
            stats = {
                "entry_count": len(self._cache),
                "size_bytes": self._current_size_bytes,
                "size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "utilization": self._current_size_bytes / self._max_size_bytes
                if self._max_size_bytes > 0 else 0,
            }

            if self._metrics:
                stats.update(self._metrics.get_stats())

            return stats

    def _evict_entry(self, key: str, entry: CacheEntry, reason: str) -> None:
        """
        Evict a cache entry (must be called with lock held).

        Args:
            key: Cache key
            entry: Cache entry
            reason: Reason for eviction
        """
        del self._cache[key]
        self._current_size_bytes -= entry.size_bytes

        if self._metrics:
            self._metrics.record_eviction(entry.size_bytes)

        logger.debug(f"Evicted cache entry: {key} (reason: {reason})")

    def _evict_lru(self) -> None:
        """
        Evict the least recently used entry (must be called with lock held).
        """
        if not self._cache:
            return

        # Find LRU entry
        lru_key = None
        lru_accessed_at = float('inf')

        for key, entry in self._cache.items():
            if entry.accessed_at < lru_accessed_at:
                lru_accessed_at = entry.accessed_at
                lru_key = key

        if lru_key:
            entry = self._cache[lru_key]
            self._evict_entry(lru_key, entry, reason="lru_eviction")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        expired_keys = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                entry = self._cache[key]
                self._evict_entry(key, entry, reason="ttl_expired")

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")


# Decorator for caching function results
def cache_result(
    cache: L1MemoryCache,
    ttl: Optional[int] = None,
    key_prefix: str = ""
) -> Callable:
    """
    Decorator to cache function results.

    Args:
        cache: L1MemoryCache instance
        ttl: Optional TTL in seconds
        key_prefix: Optional prefix for cache keys

    Example:
        >>> cache = L1MemoryCache()
        >>> @cache_result(cache, ttl=300)
        ... async def expensive_function(arg1, arg2):
        ...     # Expensive computation
        ...     return result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(filter(None, key_parts))

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator


def cache_with_key(
    cache: L1MemoryCache,
    key_fn: Callable[..., str],
    ttl: Optional[int] = None
) -> Callable:
    """
    Decorator to cache function results with custom key function.

    Args:
        cache: L1MemoryCache instance
        key_fn: Function to generate cache key from args/kwargs
        ttl: Optional TTL in seconds

    Example:
        >>> cache = L1MemoryCache()
        >>> def make_key(workflow_id, user_id):
        ...     return f"workflow:{workflow_id}:user:{user_id}"
        >>> @cache_with_key(cache, key_fn=make_key, ttl=300)
        ... async def get_workflow(workflow_id, user_id):
        ...     return fetch_workflow(workflow_id, user_id)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key using custom function
            cache_key = key_fn(*args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator


# Global cache instance (singleton pattern)
_global_cache: Optional[L1MemoryCache] = None
_cache_lock = threading.Lock()


def get_global_cache() -> L1MemoryCache:
    """
    Get or create global L1 cache instance.

    Returns:
        Global L1MemoryCache instance

    Example:
        >>> cache = get_global_cache()
        >>> await cache.set("key", "value")
    """
    global _global_cache
    with _cache_lock:
        if _global_cache is None:
            _global_cache = L1MemoryCache()
        return _global_cache


async def initialize_global_cache(
    max_size_mb: int = 100,
    default_ttl_seconds: int = 60
) -> L1MemoryCache:
    """
    Initialize and start global cache.

    Args:
        max_size_mb: Maximum cache size in MB
        default_ttl_seconds: Default TTL in seconds

    Returns:
        Initialized L1MemoryCache instance

    Example:
        >>> cache = await initialize_global_cache(max_size_mb=200)
    """
    global _global_cache
    with _cache_lock:
        if _global_cache is not None:
            await _global_cache.stop()

        _global_cache = L1MemoryCache(
            max_size_mb=max_size_mb,
            default_ttl_seconds=default_ttl_seconds
        )
        await _global_cache.start()
        return _global_cache
