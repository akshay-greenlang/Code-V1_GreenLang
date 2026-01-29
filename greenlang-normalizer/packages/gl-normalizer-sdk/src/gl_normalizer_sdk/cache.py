"""
Caching backends for the GreenLang Normalizer SDK.

This module provides caching capabilities to reduce API calls
and improve performance.

Example:
    >>> from gl_normalizer_sdk import MemoryCache
    >>> cache = MemoryCache(max_size=1000, ttl=3600)
    >>> cache.set("key", {"value": 42})
    >>> result = cache.get("key")
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import json
import hashlib


@dataclass
class CacheConfig:
    """Configuration for cache backends."""

    max_size: int = 10000
    ttl_seconds: int = 3600
    prefix: str = "glnorm:"
    serialize: bool = True


@dataclass
class CacheEntry:
    """A cached entry with expiration."""

    value: Any
    expires_at: datetime
    hits: int = 0


class CacheBackend(ABC):
    """
    Abstract base class for cache backends.

    Cache backends store conversion results and vocabulary data
    to reduce API calls and improve performance.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override in seconds
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached entries."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

    def make_key(self, *parts: str) -> str:
        """Create a cache key from parts."""
        key_str = ":".join(str(p) for p in parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]


class MemoryCache(CacheBackend):
    """
    In-memory cache with LRU eviction.

    This cache stores entries in memory using an OrderedDict
    for efficient LRU eviction.

    Example:
        >>> cache = MemoryCache(max_size=1000, ttl=3600)
        >>> cache.set("conversion:kg:t", {"factor": 0.001})
        >>> result = cache.get("conversion:kg:t")
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl: int = 3600,
    ) -> None:
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Check expiration
        if datetime.now() > entry.expires_at:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.hits += 1
        self._hits += 1

        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in memory cache."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        # Remove oldest entry if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
        self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear memory cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0

        return {
            "type": "memory",
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
        }


class RedisCache(CacheBackend):
    """
    Redis-backed cache for distributed caching.

    This cache uses Redis for distributed caching across
    multiple service instances.

    Example:
        >>> cache = RedisCache(host="localhost", port=6379)
        >>> cache.set("conversion:kg:t", {"factor": 0.001})
        >>> result = cache.get("conversion:kg:t")

    Note:
        Requires redis package: pip install redis
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "glnorm:",
        ttl: int = 3600,
    ) -> None:
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Optional password
            prefix: Key prefix
            ttl: Default TTL in seconds
        """
        self.prefix = prefix
        self.default_ttl = ttl
        self._client: Optional[Any] = None
        self._connection_params = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
        }

    def _get_client(self) -> Any:
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(**self._connection_params)
            except ImportError:
                raise ImportError(
                    "Redis package required. Install with: pip install redis"
                )
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        client = self._get_client()
        prefixed_key = self._make_key(key)

        data = client.get(prefixed_key)
        if data is None:
            return None

        return json.loads(data)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis."""
        client = self._get_client()
        prefixed_key = self._make_key(key)
        ttl = ttl or self.default_ttl

        client.setex(
            prefixed_key,
            ttl,
            json.dumps(value, default=str),
        )

    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        client = self._get_client()
        prefixed_key = self._make_key(key)
        return bool(client.delete(prefixed_key))

    def clear(self) -> None:
        """Clear all keys with prefix."""
        client = self._get_client()
        pattern = f"{self.prefix}*"
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics from Redis."""
        client = self._get_client()
        info = client.info("stats")
        pattern = f"{self.prefix}*"
        key_count = len(client.keys(pattern))

        return {
            "type": "redis",
            "size": key_count,
            "total_connections": info.get("total_connections_received", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
        }


class NullCache(CacheBackend):
    """
    No-op cache for testing or disabled caching.

    This cache does nothing and always returns None,
    useful for testing or when caching is disabled.
    """

    def get(self, key: str) -> Optional[Any]:
        """Always returns None."""
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Does nothing."""
        pass

    def delete(self, key: str) -> bool:
        """Always returns False."""
        return False

    def clear(self) -> None:
        """Does nothing."""
        pass

    def stats(self) -> Dict[str, Any]:
        """Return empty stats."""
        return {"type": "null", "enabled": False}


__all__ = [
    "CacheBackend",
    "CacheConfig",
    "CacheEntry",
    "MemoryCache",
    "RedisCache",
    "NullCache",
]
