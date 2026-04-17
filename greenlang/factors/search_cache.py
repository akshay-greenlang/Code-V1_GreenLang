# -*- coding: utf-8 -*-
"""
Multi-tier search result cache for Factors catalog (F082).

Provides an L1 in-memory LRU cache and an L2 Redis cache layer for
factor search results, with TTL management and cache invalidation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cached result with TTL tracking."""

    key: str
    data: Any
    created_at: float
    ttl_seconds: int
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


@dataclass
class CacheStats:
    """Aggregate cache statistics."""

    l1_size: int = 0
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    evictions: int = 0

    @property
    def l1_hit_ratio(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total else 0.0

    @property
    def total_hit_ratio(self) -> float:
        total = self.l1_hits + self.l1_misses
        return (self.l1_hits + self.l2_hits) / total if total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l1_size": self.l1_size,
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l1_hit_ratio": round(self.l1_hit_ratio, 4),
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "total_hit_ratio": round(self.total_hit_ratio, 4),
            "evictions": self.evictions,
        }


class SearchCache:
    """
    Two-tier cache for factor search results.

    L1: In-memory LRU (thread-safe, configurable max size)
    L2: Redis (optional, gracefully degrades if unavailable)

    Cache keys are derived from search parameters via SHA-256 hashing.
    """

    DEFAULT_L1_MAX = 2000
    DEFAULT_TTL = 3600  # 1 hour

    def __init__(
        self,
        l1_max: int = DEFAULT_L1_MAX,
        default_ttl: int = DEFAULT_TTL,
        redis_client: Any = None,
        redis_prefix: str = "gl:factors:cache:",
    ) -> None:
        self._l1: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l1_max = l1_max
        self._default_ttl = default_ttl
        self._redis = redis_client
        self._redis_prefix = redis_prefix
        self._lock = threading.Lock()
        self._stats = CacheStats()

    @staticmethod
    def _make_key(params: Dict[str, Any]) -> str:
        """Create deterministic cache key from search parameters."""
        canonical = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    def get(self, params: Dict[str, Any]) -> Optional[Any]:
        """
        Look up cached search results.

        Checks L1 first, then L2. Promotes L2 hits to L1.
        """
        key = self._make_key(params)

        # L1 check
        with self._lock:
            entry = self._l1.get(key)
            if entry and not entry.is_expired:
                self._l1.move_to_end(key)
                entry.hit_count += 1
                self._stats.l1_hits += 1
                return entry.data
            elif entry:
                del self._l1[key]

        self._stats.l1_misses += 1

        # L2 check
        if self._redis:
            try:
                raw = self._redis.get(f"{self._redis_prefix}{key}")
                if raw:
                    data = json.loads(raw)
                    self._stats.l2_hits += 1
                    # Promote to L1
                    self._put_l1(key, data, self._default_ttl)
                    return data
            except Exception:
                pass
            self._stats.l2_misses += 1

        return None

    def put(self, params: Dict[str, Any], data: Any, ttl: Optional[int] = None) -> str:
        """
        Store search results in both cache tiers.

        Returns the cache key.
        """
        key = self._make_key(params)
        ttl_sec = ttl or self._default_ttl

        self._put_l1(key, data, ttl_sec)

        # L2 write
        if self._redis:
            try:
                self._redis.setex(
                    f"{self._redis_prefix}{key}",
                    ttl_sec,
                    json.dumps(data, default=str),
                )
            except Exception:
                pass

        return key

    def _put_l1(self, key: str, data: Any, ttl_sec: int) -> None:
        """Insert into L1 with eviction."""
        with self._lock:
            if key in self._l1:
                self._l1.move_to_end(key)
                self._l1[key].data = data
                self._l1[key].created_at = time.time()
            else:
                self._l1[key] = CacheEntry(
                    key=key, data=data, created_at=time.time(), ttl_seconds=ttl_sec
                )
            # Evict LRU entries if over limit
            while len(self._l1) > self._l1_max:
                self._l1.popitem(last=False)
                self._stats.evictions += 1
            self._stats.l1_size = len(self._l1)

    def invalidate(self, params: Dict[str, Any]) -> bool:
        """Invalidate a specific cache entry."""
        key = self._make_key(params)
        removed = False
        with self._lock:
            if key in self._l1:
                del self._l1[key]
                removed = True
                self._stats.l1_size = len(self._l1)
        if self._redis:
            try:
                self._redis.delete(f"{self._redis_prefix}{key}")
                removed = True
            except Exception:
                pass
        return removed

    def invalidate_edition(self, edition_id: str) -> int:
        """Invalidate all entries for a given edition (bulk flush)."""
        count = 0
        with self._lock:
            keys_to_remove = list(self._l1.keys())
            for k in keys_to_remove:
                del self._l1[k]
                count += 1
            self._stats.l1_size = len(self._l1)

        # For Redis, use pattern-based deletion if available
        if self._redis:
            try:
                pattern = f"{self._redis_prefix}*"
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        self._redis.delete(*keys)
                        count += len(keys)
                    if cursor == 0:
                        break
            except Exception:
                pass

        logger.info("Invalidated %d cache entries for edition=%s", count, edition_id)
        return count

    def clear(self) -> None:
        """Flush all cache entries."""
        with self._lock:
            self._l1.clear()
            self._stats.l1_size = 0

    @property
    def stats(self) -> CacheStats:
        return self._stats
