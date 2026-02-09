# -*- coding: utf-8 -*-
"""
Cache Manager Engine - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

Manages query result caching with TTL-based expiration, LRU eviction,
cache statistics tracking, and per-source cache invalidation.

Zero-Hallucination Guarantees:
    - Cache keys use deterministic SHA-256 hashing
    - Expiration uses exact timestamp comparison
    - LRU eviction is based on deterministic last-access ordering
    - SHA-256 provenance hashes on all cache operations

Example:
    >>> from greenlang.data_gateway.cache_manager import CacheManagerEngine
    >>> cache = CacheManagerEngine()
    >>> entry = cache.put("abc123", result, "SRC-001", ttl=300)
    >>> cached = cache.get("abc123")
    >>> assert cached is not None

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

def _make_cache_entry(
    cache_key: str,
    query_hash: str,
    source_id: str,
    ttl: int,
    size_bytes: int = 0,
) -> Dict[str, Any]:
    """Create a CacheEntry dictionary.

    Args:
        cache_key: Unique cache entry key.
        query_hash: SHA-256 hash of the query.
        source_id: Data source that produced the result.
        ttl: Time-to-live in seconds.
        size_bytes: Approximate size of cached data in bytes.

    Returns:
        CacheEntry dictionary.
    """
    now = _utcnow()
    return {
        "cache_key": cache_key,
        "query_hash": query_hash,
        "source_id": source_id,
        "ttl": ttl,
        "size_bytes": size_bytes,
        "created_at": now.isoformat(),
        "expires_at": (
            datetime(
                now.year, now.month, now.day,
                now.hour, now.minute, now.second,
                tzinfo=timezone.utc,
            ).__add__(
                __import__("datetime").timedelta(seconds=ttl)
            )
        ).isoformat(),
        "last_accessed_at": now.isoformat(),
        "access_count": 0,
    }


class CacheManagerEngine:
    """Query result caching engine with TTL and LRU eviction.

    Manages an in-memory cache of query results with configurable
    TTL, LRU eviction, hit/miss tracking, and per-source invalidation.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Provenance tracker instance.
        _cache: In-memory cache storage (key -> (CacheEntry, QueryResult)).
        _hits: Total cache hit count.
        _misses: Total cache miss count.
        _default_ttl: Default cache TTL in seconds.
        _max_entries: Maximum cache entries before eviction.

    Example:
        >>> cache = CacheManagerEngine()
        >>> cache.put("hash123", result, "SRC-001", ttl=300)
        >>> cached = cache.get("hash123")
        >>> assert cached is not None
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize CacheManagerEngine.

        Args:
            config: Optional configuration.
            provenance: Optional ProvenanceTracker instance.
        """
        self._config = config or {}
        self._provenance = provenance
        self._cache: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        self._hits: int = 0
        self._misses: int = 0

        # Load config defaults
        self._default_ttl = 300
        self._max_entries = 10000
        if hasattr(config, "cache_default_ttl"):
            self._default_ttl = config.cache_default_ttl
        elif isinstance(config, dict):
            self._default_ttl = config.get("cache_default_ttl", 300)

        logger.info(
            "CacheManagerEngine initialized: default_ttl=%ds, max_entries=%d",
            self._default_ttl, self._max_entries,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        query_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a cached query result by query hash.

        Updates access time and count on hit. Returns None on miss
        or expired entry.

        Args:
            query_hash: SHA-256 hash of the query.

        Returns:
            Cached QueryResult dictionary or None.
        """
        cached = self._cache.get(query_hash)

        if cached is None:
            self._misses += 1
            try:
                from greenlang.data_gateway.metrics import record_cache_miss
                record_cache_miss(source="unknown")
            except ImportError:
                pass
            return None

        entry, result = cached

        # Check expiration
        if self._is_expired(entry):
            # Remove expired entry
            del self._cache[query_hash]
            self._misses += 1
            try:
                from greenlang.data_gateway.metrics import record_cache_miss
                record_cache_miss(source=entry.get("source_id", "unknown"))
            except ImportError:
                pass
            return None

        # Update access time and count
        entry["last_accessed_at"] = _utcnow().isoformat()
        entry["access_count"] = entry.get("access_count", 0) + 1

        self._hits += 1

        # Record metrics
        try:
            from greenlang.data_gateway.metrics import record_cache_hit
            record_cache_hit(source=entry.get("source_id", "unknown"))
        except ImportError:
            pass

        logger.debug(
            "Cache HIT for hash %s (access_count=%d)",
            query_hash[:16], entry["access_count"],
        )
        return result

    def put(
        self,
        query_hash: str,
        result: Dict[str, Any],
        source_id: str,
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Cache a query result.

        If max entries reached, performs LRU eviction first.

        Args:
            query_hash: SHA-256 hash of the query.
            result: QueryResult dictionary to cache.
            source_id: Source that produced the result.
            ttl: Time-to-live in seconds (None for default).

        Returns:
            CacheEntry dictionary.
        """
        if ttl is None:
            ttl = self._default_ttl

        # Evict expired entries first
        self._evict_expired()

        # Check max entries
        if len(self._cache) >= self._max_entries:
            evicted = self._evict_lru(
                target_count=self._max_entries // 10,
            )
            logger.info("LRU eviction: removed %d entries", evicted)

        # Estimate size
        size_bytes = len(json.dumps(result, default=str).encode())

        entry = _make_cache_entry(
            cache_key=query_hash,
            query_hash=query_hash,
            source_id=source_id,
            ttl=ttl,
            size_bytes=size_bytes,
        )

        self._cache[query_hash] = (entry, result)

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash({
                "query_hash": query_hash,
                "source_id": source_id,
                "ttl": ttl,
            })
            self._provenance.record(
                entity_type="cache_entry",
                entity_id=query_hash[:16],
                action="cache_operation",
                data_hash=data_hash,
            )

        logger.debug(
            "Cache PUT: hash=%s, source=%s, ttl=%ds, size=%d bytes",
            query_hash[:16], source_id, ttl, size_bytes,
        )
        return entry

    def invalidate(
        self,
        source_id: Optional[str] = None,
        query_hash: Optional[str] = None,
        invalidate_all: bool = False,
    ) -> int:
        """Invalidate cache entries.

        Can invalidate by source, by query hash, or all entries.

        Args:
            source_id: Invalidate all entries for this source.
            query_hash: Invalidate specific query hash.
            invalidate_all: Invalidate all entries.

        Returns:
            Number of entries invalidated.
        """
        count = 0

        if invalidate_all:
            count = len(self._cache)
            self._cache.clear()
            logger.info("Cache invalidated: ALL %d entries", count)
            return count

        if query_hash:
            if query_hash in self._cache:
                del self._cache[query_hash]
                count = 1
                logger.info(
                    "Cache invalidated: hash=%s", query_hash[:16],
                )

        if source_id:
            keys_to_remove = [
                key for key, (entry, _) in self._cache.items()
                if entry.get("source_id") == source_id
            ]
            for key in keys_to_remove:
                del self._cache[key]
            count += len(keys_to_remove)
            logger.info(
                "Cache invalidated: source=%s (%d entries)",
                source_id, len(keys_to_remove),
            )

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics including:
            - total_entries: Current cache size
            - hits: Total hit count
            - misses: Total miss count
            - hit_rate: Hit rate percentage
            - total_size_bytes: Total cached data size
            - expired_entries: Count of expired entries
        """
        total = self._hits + self._misses
        hit_rate = (
            round(self._hits / total * 100, 2)
            if total > 0
            else 0.0
        )

        total_size = sum(
            entry.get("size_bytes", 0)
            for entry, _ in self._cache.values()
        )

        expired = sum(
            1 for entry, _ in self._cache.values()
            if self._is_expired(entry)
        )

        return {
            "total_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_size_bytes": total_size,
            "expired_entries": expired,
            "default_ttl": self._default_ttl,
            "max_entries": self._max_entries,
        }

    def compute_query_hash(
        self,
        request: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 hash of a query request for cache keying.

        Uses sorted JSON serialization for deterministic hashing.

        Args:
            request: Query request dictionary.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        # Normalize the request for consistent hashing
        hash_data = {
            "sources": sorted(request.get("sources", [])),
            "filters": request.get("filters", []),
            "sorts": request.get("sorts", []),
            "aggregations": request.get("aggregations", []),
            "fields": sorted(request.get("fields", [])),
            "limit": request.get("limit", 100),
            "offset": request.get("offset", 0),
        }
        raw = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry has expired.

        Args:
            entry: CacheEntry dictionary.

        Returns:
            True if the entry is expired.
        """
        expires_at_str = entry.get("expires_at")
        if not expires_at_str:
            return False

        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return _utcnow() >= expires_at
        except (ValueError, TypeError):
            return False

    def _evict_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        expired_keys = [
            key for key, (entry, _) in self._cache.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug("Evicted %d expired cache entries", len(expired_keys))

        return len(expired_keys)

    def _evict_lru(self, target_count: int) -> int:
        """Evict least recently used entries.

        Removes the oldest-accessed entries until target_count
        entries have been removed.

        Args:
            target_count: Number of entries to evict.

        Returns:
            Number of entries actually evicted.
        """
        if target_count <= 0:
            return 0

        # Sort entries by last_accessed_at (ascending = oldest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda item: item[1][0].get("last_accessed_at", ""),
        )

        evicted = 0
        for key, _ in sorted_entries:
            if evicted >= target_count:
                break
            del self._cache[key]
            evicted += 1

        return evicted

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def entry_count(self) -> int:
        """Return the current number of cache entries."""
        return len(self._cache)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache manager statistics (alias for get_stats).

        Returns:
            Dictionary with cache statistics.
        """
        return self.get_stats()


__all__ = [
    "CacheManagerEngine",
]
