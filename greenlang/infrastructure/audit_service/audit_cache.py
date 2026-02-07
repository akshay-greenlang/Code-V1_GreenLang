# -*- coding: utf-8 -*-
"""
Audit Cache - SEC-005: Centralized Audit Logging Service

Provides Redis-backed event deduplication with configurable TTL.
Prevents duplicate events from being processed when using at-least-once
delivery semantics.

**Design Principles:**
- Redis-backed for distributed deduplication
- Configurable TTL (default 5 minutes)
- Fallback to in-memory cache if Redis unavailable
- Non-blocking cache operations

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache Configuration
# ---------------------------------------------------------------------------


@dataclass
class AuditCacheConfig:
    """Configuration for the AuditCache.

    Attributes:
        ttl_seconds: Time-to-live for cache entries (default 300 = 5 minutes).
        key_prefix: Redis key prefix (default "gl:audit:dedup:").
        max_memory_entries: Max entries for in-memory fallback (default 10000).
        enable_redis: Whether to use Redis (default True).
    """

    ttl_seconds: int = 300
    key_prefix: str = "gl:audit:dedup:"
    max_memory_entries: int = 10000
    enable_redis: bool = True


# ---------------------------------------------------------------------------
# Cache Metrics
# ---------------------------------------------------------------------------


@dataclass
class CacheMetrics:
    """Metrics for monitoring the audit cache.

    Attributes:
        checks_performed: Total duplicate checks performed.
        duplicates_found: Number of duplicates detected.
        marks_performed: Total events marked as processed.
        redis_errors: Number of Redis errors encountered.
        memory_fallback_hits: Times in-memory cache was used as fallback.
    """

    checks_performed: int = 0
    duplicates_found: int = 0
    marks_performed: int = 0
    redis_errors: int = 0
    memory_fallback_hits: int = 0


# ---------------------------------------------------------------------------
# In-Memory LRU Cache (Fallback)
# ---------------------------------------------------------------------------


class LRUCache:
    """Simple LRU cache with TTL support for fallback.

    Used when Redis is unavailable or disabled.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300) -> None:
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries.
            ttl_seconds: Time-to-live for entries in seconds.
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._lock = asyncio.Lock()

    async def contains(self, key: str) -> bool:
        """Check if key exists and is not expired.

        Args:
            key: The key to check.

        Returns:
            True if key exists and is not expired.
        """
        async with self._lock:
            if key not in self._cache:
                return False

            expires_at = self._cache[key]
            if time.time() > expires_at:
                del self._cache[key]
                return False

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return True

    async def add(self, key: str) -> None:
        """Add a key to the cache.

        Args:
            key: The key to add.
        """
        async with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            expires_at = time.time() + self._ttl_seconds
            self._cache[key] = expires_at

    async def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            now = time.time()
            expired = [k for k, v in self._cache.items() if now > v]
            for key in expired:
                del self._cache[key]
            return len(expired)

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of entries in cache.
        """
        return len(self._cache)


# ---------------------------------------------------------------------------
# Audit Cache
# ---------------------------------------------------------------------------


class AuditCache:
    """Redis-backed event deduplication cache.

    Uses Redis SET with TTL for distributed deduplication.
    Falls back to in-memory LRU cache if Redis is unavailable.

    Example:
        >>> cache = AuditCache(redis_client=redis)
        >>> if not await cache.check_duplicate(event_id):
        ...     # Process event
        ...     await cache.mark_processed(event_id)
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        config: Optional[AuditCacheConfig] = None,
    ) -> None:
        """Initialize the audit cache.

        Args:
            redis_client: Async Redis client (redis.asyncio).
            config: Cache configuration.
        """
        self._redis = redis_client
        self._config = config or AuditCacheConfig()
        self._metrics = CacheMetrics()
        self._memory_cache = LRUCache(
            max_size=self._config.max_memory_entries,
            ttl_seconds=self._config.ttl_seconds,
        )
        self._use_redis = self._config.enable_redis and redis_client is not None

    @property
    def metrics(self) -> CacheMetrics:
        """Get current cache metrics.

        Returns:
            Current metrics snapshot.
        """
        return self._metrics

    async def check_duplicate(self, event_id: str) -> bool:
        """Check if an event has already been processed.

        Args:
            event_id: Unique event identifier.

        Returns:
            True if the event is a duplicate (already processed).
        """
        self._metrics.checks_performed += 1

        if self._use_redis:
            try:
                key = f"{self._config.key_prefix}{event_id}"
                exists = await self._redis.exists(key)
                if exists:
                    self._metrics.duplicates_found += 1
                    return True
                return False
            except Exception as e:
                self._metrics.redis_errors += 1
                logger.warning(
                    "Redis check_duplicate failed, falling back to memory: %s",
                    e,
                )
                self._metrics.memory_fallback_hits += 1
                # Fall through to memory cache

        # Memory cache fallback
        is_duplicate = await self._memory_cache.contains(event_id)
        if is_duplicate:
            self._metrics.duplicates_found += 1
        return is_duplicate

    async def mark_processed(self, event_id: str) -> bool:
        """Mark an event as processed.

        Args:
            event_id: Unique event identifier.

        Returns:
            True if successfully marked, False on error.
        """
        self._metrics.marks_performed += 1

        if self._use_redis:
            try:
                key = f"{self._config.key_prefix}{event_id}"
                await self._redis.set(
                    key, "1", ex=self._config.ttl_seconds
                )
                return True
            except Exception as e:
                self._metrics.redis_errors += 1
                logger.warning(
                    "Redis mark_processed failed, falling back to memory: %s",
                    e,
                )
                self._metrics.memory_fallback_hits += 1
                # Fall through to memory cache

        # Memory cache fallback
        await self._memory_cache.add(event_id)
        return True

    async def check_and_mark(self, event_id: str) -> bool:
        """Atomically check if duplicate and mark as processed.

        Uses Redis SETNX for atomic check-and-set.

        Args:
            event_id: Unique event identifier.

        Returns:
            True if event is new (not a duplicate) and was marked.
            False if event is a duplicate.
        """
        self._metrics.checks_performed += 1
        self._metrics.marks_performed += 1

        if self._use_redis:
            try:
                key = f"{self._config.key_prefix}{event_id}"
                # SETNX returns True if key was set (event is new)
                was_set = await self._redis.set(
                    key,
                    "1",
                    ex=self._config.ttl_seconds,
                    nx=True,  # Only set if not exists
                )
                if not was_set:
                    self._metrics.duplicates_found += 1
                return bool(was_set)
            except Exception as e:
                self._metrics.redis_errors += 1
                logger.warning(
                    "Redis check_and_mark failed, falling back to memory: %s",
                    e,
                )
                self._metrics.memory_fallback_hits += 1
                # Fall through to memory cache

        # Memory cache fallback (not atomic but best effort)
        if await self._memory_cache.contains(event_id):
            self._metrics.duplicates_found += 1
            return False

        await self._memory_cache.add(event_id)
        return True

    async def cleanup(self) -> int:
        """Clean up expired entries from memory cache.

        Redis handles TTL expiration automatically.

        Returns:
            Number of entries cleaned up.
        """
        return await self._memory_cache.cleanup_expired()

    def get_memory_cache_size(self) -> int:
        """Get current size of memory cache.

        Returns:
            Number of entries in memory cache.
        """
        return self._memory_cache.size()


__all__ = [
    "AuditCache",
    "AuditCacheConfig",
    "CacheMetrics",
]
