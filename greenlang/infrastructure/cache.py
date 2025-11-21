"""
Cache Manager
=============

Multi-tier caching system for GreenLang with TTL support.

Author: Infrastructure Team
Created: 2025-11-21
"""

import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import hashlib
import json
from functools import wraps

from greenlang.infrastructure.base import BaseInfrastructureComponent, InfrastructureConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    max_size: int = 1000
    default_ttl: int = 3600  # 1 hour
    eviction_policy: str = 'lru'  # 'lru', 'lfu', 'fifo'
    enable_stats: bool = True


class CacheManager(BaseInfrastructureComponent):
    """
    Multi-tier cache manager with TTL and eviction policies.

    Supports memory caching with LRU/LFU/FIFO eviction and TTL expiration.
    """

    def __init__(self, config: Optional[InfrastructureConfig] = None,
                 cache_config: Optional[CacheConfig] = None):
        """Initialize cache manager."""
        super().__init__(config or InfrastructureConfig(component_name="CacheManager"))
        self.cache_config = cache_config or CacheConfig()
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _initialize(self) -> None:
        """Initialize cache resources."""
        logger.info(f"CacheManager initialized with max_size={self.cache_config.max_size}")

    def start(self) -> None:
        """Start the cache manager."""
        self.status = self.status.RUNNING
        logger.info("CacheManager started")

    def stop(self) -> None:
        """Stop the cache manager."""
        self.clear()
        self.status = self.status.STOPPED
        logger.info("CacheManager stopped")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        self.update_activity()

        entry = self.cache.get(key)
        if entry is None:
            self.misses += 1
            logger.debug(f"Cache miss for key: {key}")
            return None

        if entry.is_expired():
            self.misses += 1
            del self.cache[key]
            logger.debug(f"Cache entry expired for key: {key}")
            return None

        entry.touch()
        self.hits += 1
        logger.debug(f"Cache hit for key: {key}")

        self._update_metrics()
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
        """
        self.update_activity()

        # Check if eviction needed
        if len(self.cache) >= self.cache_config.max_size:
            self._evict()

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            ttl_seconds=ttl or self.cache_config.default_ttl
        )

        self.cache[key] = entry
        logger.debug(f"Cached value for key: {key} with TTL: {entry.ttl_seconds}s")

        self._update_metrics()

    def get_or_compute(self, key: str, compute_func: Callable[[], Any],
                      ttl: Optional[int] = None) -> Any:
        """
        Get from cache or compute if not present.

        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time to live in seconds

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        value = compute_func()
        self.set(key, value, ttl)
        return value

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was removed, False if not found
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache key: {key}")
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {size} cache entries")

    def _evict(self) -> None:
        """Evict entries based on policy."""
        if not self.cache:
            return

        self.evictions += 1

        if self.cache_config.eviction_policy == 'lru':
            # Evict least recently used
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k].last_accessed)
            del self.cache[oldest_key]
            logger.debug(f"Evicted LRU key: {oldest_key}")

        elif self.cache_config.eviction_policy == 'lfu':
            # Evict least frequently used
            least_used_key = min(self.cache.keys(),
                               key=lambda k: self.cache[k].access_count)
            del self.cache[least_used_key]
            logger.debug(f"Evicted LFU key: {least_used_key}")

        elif self.cache_config.eviction_policy == 'fifo':
            # Evict oldest entry
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]
            logger.debug(f"Evicted FIFO key: {oldest_key}")

    def _update_metrics(self) -> None:
        """Update cache metrics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        self._metrics.update({
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "max_size": self.cache_config.max_size
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_config.enable_stats:
            return {}

        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "evictions": self.evictions,
            "entries": [
                {
                    "key": k,
                    "access_count": v.access_count,
                    "age_seconds": (datetime.now() - v.created_at).total_seconds(),
                    "ttl_remaining": v.ttl_seconds - (datetime.now() - v.created_at).total_seconds() if v.ttl_seconds else None
                }
                for k, v in list(self.cache.items())[:10]  # Top 10 entries
            ]
        }

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()


def cached(ttl: int = 3600):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds

    Example:
        @cached(ttl=600)
        def expensive_function(x):
            return x ** 2
    """
    def decorator(func):
        cache = CacheManager()

        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{CacheManager.make_key(*args, **kwargs)}"
            return cache.get_or_compute(
                cache_key,
                lambda: func(*args, **kwargs),
                ttl=ttl
            )

        wrapper.cache = cache
        return wrapper

    return decorator