# -*- coding: utf-8 -*-
"""
Secrets Cache - SEC-006

Two-layer caching for secrets:
- L1: Redis (5 minute TTL, shared across instances)
- L2: Memory (30 second TTL, per-instance)

Supports version-aware caching to handle secret updates correctly.

Example:
    >>> from greenlang.infrastructure.secrets_service.cache import SecretsCache
    >>> cache = SecretsCache(redis_client=redis, config=config)
    >>> await cache.set("tenant/acme/db", {"password": "secret"}, version=3)
    >>> data = await cache.get("tenant/acme/db", version=3)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram

    CACHE_HITS = Counter(
        "secrets_cache_hits_total",
        "Total number of cache hits",
        ["layer", "secret_type"],
    )
    CACHE_MISSES = Counter(
        "secrets_cache_misses_total",
        "Total number of cache misses",
        ["layer", "secret_type"],
    )
    CACHE_OPERATIONS = Histogram(
        "secrets_cache_operation_duration_seconds",
        "Duration of cache operations",
        ["operation", "layer"],
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    CACHE_HITS = None
    CACHE_MISSES = None
    CACHE_OPERATIONS = None


def _record_hit(layer: str, secret_type: str = "generic") -> None:
    """Record a cache hit metric."""
    if METRICS_AVAILABLE and CACHE_HITS:
        CACHE_HITS.labels(layer=layer, secret_type=secret_type).inc()


def _record_miss(layer: str, secret_type: str = "generic") -> None:
    """Record a cache miss metric."""
    if METRICS_AVAILABLE and CACHE_MISSES:
        CACHE_MISSES.labels(layer=layer, secret_type=secret_type).inc()


# ---------------------------------------------------------------------------
# Memory Cache (L2)
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cache entry with value and expiry."""

    value: Dict[str, Any]
    expiry: float
    version: Optional[int] = None

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        return time.time() >= self.expiry


class MemoryCache:
    """In-memory cache with TTL and size limits.

    Thread-safe using asyncio.Lock for async contexts.

    Attributes:
        ttl_seconds: Default TTL for entries.
        max_size: Maximum number of entries.
    """

    def __init__(
        self,
        ttl_seconds: int = 30,
        max_size: int = 1000,
    ):
        """Initialize memory cache.

        Args:
            ttl_seconds: Default TTL in seconds.
            max_size: Maximum cache entries.
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(
        self,
        key: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a value from cache.

        Args:
            key: Cache key.
            version: Expected version (None matches any).

        Returns:
            Cached value or None if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                _record_miss("memory")
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                _record_miss("memory")
                return None

            # Version check
            if version is not None and entry.version != version:
                self._misses += 1
                _record_miss("memory")
                return None

            self._hits += 1
            _record_hit("memory")
            return entry.value

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
        version: Optional[int] = None,
    ) -> None:
        """Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Optional TTL override.
            version: Secret version for version-aware caching.
        """
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                await self._evict_expired_unlocked()

            # Still at capacity? Evict oldest
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            expiry = time.time() + (ttl or self._ttl_seconds)
            self._cache[key] = CacheEntry(
                value=value,
                expiry=expiry,
                version=version,
            )

    async def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: Cache key to delete.

        Returns:
            True if key was deleted.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Simple prefix-based matching (not full glob).

        Args:
            pattern: Key prefix to match.

        Returns:
            Number of keys deleted.
        """
        async with self._lock:
            # For simplicity, treat pattern as prefix
            prefix = pattern.rstrip("*")
            keys_to_delete = [
                k for k in self._cache.keys() if k.startswith(prefix)
            ]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    async def clear(self) -> None:
        """Clear all entries from cache."""
        async with self._lock:
            self._cache.clear()

    async def _evict_expired_unlocked(self) -> int:
        """Evict expired entries (must hold lock).

        Returns:
            Number of entries evicted.
        """
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items() if v.expiry <= now
        ]
        for key in expired_keys:
            del self._cache[key]
        return len(expired_keys)

    async def cleanup_expired(self) -> int:
        """Clean up expired entries.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            return await self._evict_expired_unlocked()

    @property
    def size(self) -> int:
        """Current number of entries in cache."""
        return len(self._cache)

    @property
    def stats(self) -> Dict[str, int]:
        """Cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Redis Cache (L1)
# ---------------------------------------------------------------------------


class RedisSecretCache:
    """Redis-based cache for secrets (L1 layer).

    Provides shared caching across application instances.

    Attributes:
        redis_client: Async Redis client.
        key_prefix: Prefix for all cache keys.
        ttl_seconds: Default TTL.
    """

    def __init__(
        self,
        redis_client: Any,
        key_prefix: str = "gl:secrets",
        ttl_seconds: int = 300,
    ):
        """Initialize Redis cache.

        Args:
            redis_client: Async Redis client (aioredis).
            key_prefix: Prefix for cache keys.
            ttl_seconds: Default TTL in seconds.
        """
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _build_key(self, path: str, version: Optional[int] = None) -> str:
        """Build a cache key.

        Args:
            path: Secret path.
            version: Optional version.

        Returns:
            Full cache key.
        """
        key = f"{self._key_prefix}:{path}"
        if version is not None:
            key = f"{key}:v{version}"
        return key

    async def get(
        self,
        path: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a secret from Redis cache.

        Args:
            path: Secret path.
            version: Expected version.

        Returns:
            Cached secret data or None.
        """
        if self._redis is None:
            return None

        try:
            key = self._build_key(path, version)
            data = await self._redis.get(key)

            if data is None:
                self._misses += 1
                _record_miss("redis")
                return None

            self._hits += 1
            _record_hit("redis")

            # Deserialize
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(data)

        except Exception as e:
            logger.warning(
                "Redis cache get failed: %s",
                str(e),
                extra={"event_category": "secrets", "path": path},
            )
            self._misses += 1
            return None

    async def set(
        self,
        path: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
        version: Optional[int] = None,
    ) -> bool:
        """Set a secret in Redis cache.

        Args:
            path: Secret path.
            value: Secret data.
            ttl: Optional TTL override.
            version: Secret version.

        Returns:
            True if successfully cached.
        """
        if self._redis is None:
            return False

        try:
            key = self._build_key(path, version)
            serialized = json.dumps(value)
            await self._redis.setex(
                key,
                ttl or self._ttl_seconds,
                serialized,
            )
            return True

        except Exception as e:
            logger.warning(
                "Redis cache set failed: %s",
                str(e),
                extra={"event_category": "secrets", "path": path},
            )
            return False

    async def delete(self, path: str, version: Optional[int] = None) -> bool:
        """Delete a secret from Redis cache.

        Args:
            path: Secret path.
            version: Specific version (None deletes latest key).

        Returns:
            True if deleted.
        """
        if self._redis is None:
            return False

        try:
            key = self._build_key(path, version)
            result = await self._redis.delete(key)
            return result > 0

        except Exception as e:
            logger.warning(
                "Redis cache delete failed: %s",
                str(e),
                extra={"event_category": "secrets", "path": path},
            )
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern.

        Args:
            pattern: Glob pattern (e.g., "tenants/acme/*").

        Returns:
            Number of keys deleted.
        """
        if self._redis is None:
            return 0

        try:
            full_pattern = f"{self._key_prefix}:{pattern}"
            keys = []

            # Use SCAN for production-safe iteration
            async for key in self._redis.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                await self._redis.delete(*keys)

            return len(keys)

        except Exception as e:
            logger.warning(
                "Redis cache pattern delete failed: %s",
                str(e),
                extra={"event_category": "secrets", "pattern": pattern},
            )
            return 0

    async def clear(self) -> int:
        """Clear all secrets cache entries.

        Returns:
            Number of keys deleted.
        """
        return await self.delete_pattern("*")

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Combined Secrets Cache (L1 + L2)
# ---------------------------------------------------------------------------


@dataclass
class SecretsCacheConfig:
    """Configuration for the secrets cache."""

    # L1 Redis
    redis_enabled: bool = True
    redis_ttl_seconds: int = 300  # 5 minutes
    redis_key_prefix: str = "gl:secrets"

    # L2 Memory
    memory_enabled: bool = True
    memory_ttl_seconds: int = 30  # 30 seconds
    memory_max_size: int = 1000


class SecretsCache:
    """Two-layer cache for secrets.

    L1: Redis (shared, longer TTL)
    L2: Memory (per-instance, shorter TTL)

    Read path: L2 -> L1 -> Vault
    Write path: Vault -> L1 -> L2

    Attributes:
        config: Cache configuration.
        memory_cache: In-memory L2 cache.
        redis_cache: Redis L1 cache.
    """

    def __init__(
        self,
        redis_client: Any = None,
        config: Optional[SecretsCacheConfig] = None,
    ):
        """Initialize the secrets cache.

        Args:
            redis_client: Async Redis client.
            config: Cache configuration.
        """
        self.config = config or SecretsCacheConfig()

        # L2: Memory cache (always available)
        self.memory_cache: Optional[MemoryCache] = None
        if self.config.memory_enabled:
            self.memory_cache = MemoryCache(
                ttl_seconds=self.config.memory_ttl_seconds,
                max_size=self.config.memory_max_size,
            )

        # L1: Redis cache
        self.redis_cache: Optional[RedisSecretCache] = None
        if self.config.redis_enabled and redis_client is not None:
            self.redis_cache = RedisSecretCache(
                redis_client=redis_client,
                key_prefix=self.config.redis_key_prefix,
                ttl_seconds=self.config.redis_ttl_seconds,
            )

    async def get(
        self,
        key: str,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get a secret from cache.

        Checks L2 (memory) first, then L1 (Redis).
        On L1 hit, populates L2.

        Args:
            key: Secret path.
            version: Expected version.

        Returns:
            Cached secret data or None.
        """
        # Check L2 (memory)
        if self.memory_cache:
            value = await self.memory_cache.get(key, version)
            if value is not None:
                logger.debug(
                    "Cache hit (memory): %s",
                    key,
                    extra={"event_category": "secrets"},
                )
                return value

        # Check L1 (Redis)
        if self.redis_cache:
            value = await self.redis_cache.get(key, version)
            if value is not None:
                logger.debug(
                    "Cache hit (redis): %s",
                    key,
                    extra={"event_category": "secrets"},
                )
                # Populate L2
                if self.memory_cache:
                    await self.memory_cache.set(key, value, version=version)
                return value

        logger.debug(
            "Cache miss: %s",
            key,
            extra={"event_category": "secrets"},
        )
        return None

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None,
        version: Optional[int] = None,
    ) -> None:
        """Set a secret in cache (both layers).

        Args:
            key: Secret path.
            value: Secret data.
            ttl: Optional TTL override (for Redis).
            version: Secret version.
        """
        # Set in L1 (Redis)
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl=ttl, version=version)

        # Set in L2 (memory)
        if self.memory_cache:
            await self.memory_cache.set(key, value, version=version)

    async def delete(self, key: str, version: Optional[int] = None) -> None:
        """Delete a secret from cache (both layers).

        Args:
            key: Secret path.
            version: Specific version.
        """
        if self.memory_cache:
            await self.memory_cache.delete(key)

        if self.redis_cache:
            await self.redis_cache.delete(key, version)

    async def delete_pattern(self, pattern: str) -> int:
        """Delete secrets matching a pattern.

        Args:
            pattern: Glob pattern.

        Returns:
            Total keys deleted across layers.
        """
        deleted = 0

        if self.memory_cache:
            deleted += await self.memory_cache.delete_pattern(pattern)

        if self.redis_cache:
            deleted += await self.redis_cache.delete_pattern(pattern)

        return deleted

    async def invalidate_tenant(self, tenant_id: str) -> int:
        """Invalidate all cached secrets for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            Number of entries invalidated.
        """
        pattern = f"*tenants/{tenant_id}/*"
        return await self.delete_pattern(pattern)

    async def clear(self) -> None:
        """Clear all cached secrets."""
        if self.memory_cache:
            await self.memory_cache.clear()

        if self.redis_cache:
            await self.redis_cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        """Combined cache statistics."""
        return {
            "memory": self.memory_cache.stats if self.memory_cache else None,
            "redis": self.redis_cache.stats if self.redis_cache else None,
        }

    async def cleanup(self) -> Dict[str, int]:
        """Clean up expired entries.

        Returns:
            Dictionary with cleanup counts per layer.
        """
        result = {"memory": 0, "redis": 0}

        if self.memory_cache:
            result["memory"] = await self.memory_cache.cleanup_expired()

        # Redis handles expiry automatically

        return result
