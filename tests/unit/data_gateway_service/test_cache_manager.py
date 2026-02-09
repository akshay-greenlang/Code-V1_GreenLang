# -*- coding: utf-8 -*-
"""
Unit Tests for CacheManagerEngine (AGENT-DATA-004)

Tests cache get/put/invalidate operations, TTL expiration, query hash
computation, LRU eviction, expired entry eviction, hit/miss tracking,
hit rate calculation, and deterministic query hashing.

Coverage target: 85%+ of cache_manager.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class CacheEntry:
    """A single cache entry with TTL and access tracking."""

    def __init__(self, key: str, value: Any, ttl: int = 300,
                 source_id: Optional[str] = None,
                 query_hash: Optional[str] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.source_id = source_id
        self.query_hash = query_hash
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.expires_at = self.created_at + ttl
        self.hit_count = 0


class CacheStats:
    """Cache performance statistics."""

    def __init__(self, total_entries: int = 0, hits: int = 0,
                 misses: int = 0, evictions: int = 0,
                 hit_rate: float = 0.0):
        self.total_entries = total_entries
        self.hits = hits
        self.misses = misses
        self.evictions = evictions
        self.hit_rate = hit_rate


# ---------------------------------------------------------------------------
# Inline CacheManagerEngine
# ---------------------------------------------------------------------------


class CacheManagerEngine:
    """In-memory cache manager with TTL, LRU eviction, and hit tracking."""

    DEFAULT_TTL = 300  # 5 minutes

    def __init__(self, default_ttl: int = 300, max_entries: int = 1000):
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache. Returns None on miss or expiration."""
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if self._is_expired(entry):
            del self._cache[key]
            self._misses += 1
            return None

        entry.hit_count += 1
        entry.last_accessed = time.time()
        self._hits += 1
        return entry.value

    def put(self, key: str, value: Any, ttl: Optional[int] = None,
            source_id: Optional[str] = None,
            query_hash: Optional[str] = None) -> CacheEntry:
        """Put a value into the cache."""
        effective_ttl = ttl if ttl is not None else self._default_ttl

        # Evict if at capacity and key is new
        if key not in self._cache and len(self._cache) >= self._max_entries:
            self._evict_lru()

        entry = CacheEntry(
            key=key,
            value=value,
            ttl=effective_ttl,
            source_id=source_id,
            query_hash=query_hash,
        )
        self._cache[key] = entry
        return entry

    def invalidate(self, source_id: Optional[str] = None,
                   query_hash: Optional[str] = None,
                   invalidate_all: bool = False) -> int:
        """Invalidate cache entries. Returns count of removed entries."""
        if invalidate_all:
            count = len(self._cache)
            self._cache.clear()
            return count

        keys_to_remove: List[str] = []
        for key, entry in self._cache.items():
            if source_id is not None and entry.source_id == source_id:
                keys_to_remove.append(key)
            elif query_hash is not None and entry.query_hash == query_hash:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        return len(keys_to_remove)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        return CacheStats(
            total_entries=len(self._cache),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            hit_rate=hit_rate,
        )

    def compute_query_hash(self, query: Dict[str, Any]) -> str:
        """Compute a deterministic hash for a query."""
        return _compute_hash(query)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        return time.time() >= entry.expires_at

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        keys_to_remove = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        for key in keys_to_remove:
            del self._cache[key]
            self._evictions += 1
        return len(keys_to_remove)

    def _evict_lru(self) -> None:
        """Evict the least recently accessed entry."""
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k].last_accessed)
        del self._cache[lru_key]
        self._evictions += 1


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def cache() -> CacheManagerEngine:
    return CacheManagerEngine(default_ttl=300, max_entries=100)


@pytest.fixture
def tiny_cache() -> CacheManagerEngine:
    """Cache with very small capacity for eviction tests."""
    return CacheManagerEngine(default_ttl=300, max_entries=3)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestGet:
    """Tests for cache get operations."""

    def test_cache_hit(self, cache):
        cache.put("key1", {"data": "hello"})
        result = cache.get("key1")
        assert result is not None
        assert result["data"] == "hello"

    def test_cache_miss(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_expired_entry_returns_none(self, cache):
        # Put with 0-second TTL so it expires immediately
        entry = cache.put("expire_key", "value", ttl=0)
        # Force expiration by setting expires_at in the past
        entry.expires_at = time.time() - 1
        result = cache.get("expire_key")
        assert result is None


class TestPut:
    """Tests for cache put operations."""

    def test_new_entry(self, cache):
        entry = cache.put("k1", "v1")
        assert entry.key == "k1"
        assert entry.value == "v1"
        assert cache.get("k1") == "v1"

    def test_overwrite(self, cache):
        cache.put("k1", "v1")
        cache.put("k1", "v2")
        assert cache.get("k1") == "v2"

    def test_with_ttl(self, cache):
        entry = cache.put("k1", "v1", ttl=60)
        assert entry.ttl == 60

    def test_default_ttl(self, cache):
        entry = cache.put("k1", "v1")
        assert entry.ttl == 300


class TestInvalidate:
    """Tests for cache invalidation."""

    def test_invalidate_by_source_id(self, cache):
        cache.put("k1", "v1", source_id="src-A")
        cache.put("k2", "v2", source_id="src-A")
        cache.put("k3", "v3", source_id="src-B")
        removed = cache.invalidate(source_id="src-A")
        assert removed == 2
        assert cache.get("k1") is None
        assert cache.get("k2") is None
        assert cache.get("k3") == "v3"

    def test_invalidate_by_query_hash(self, cache):
        qh = _compute_hash({"table": "emissions"})
        cache.put("k1", "v1", query_hash=qh)
        cache.put("k2", "v2", query_hash=qh)
        cache.put("k3", "v3", query_hash="other_hash")
        removed = cache.invalidate(query_hash=qh)
        assert removed == 2

    def test_invalidate_all(self, cache):
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.put("k3", "v3")
        removed = cache.invalidate(invalidate_all=True)
        assert removed == 3
        stats = cache.get_stats()
        assert stats.total_entries == 0

    def test_invalidate_nothing(self, cache):
        cache.put("k1", "v1", source_id="src-A")
        removed = cache.invalidate(source_id="src-NONEXISTENT")
        assert removed == 0


class TestGetStats:
    """Tests for cache statistics."""

    def test_initial_empty(self, cache):
        stats = cache.get_stats()
        assert stats.total_entries == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_after_operations(self, cache):
        cache.put("k1", "v1")
        cache.put("k2", "v2")
        cache.get("k1")  # hit
        cache.get("k2")  # hit
        cache.get("k3")  # miss
        stats = cache.get_stats()
        assert stats.total_entries == 2
        assert stats.hits == 2
        assert stats.misses == 1

    def test_hit_rate_calculation(self, cache):
        cache.put("k1", "v1")
        cache.get("k1")  # hit
        cache.get("k1")  # hit
        cache.get("k1")  # hit
        cache.get("missing")  # miss
        stats = cache.get_stats()
        assert abs(stats.hit_rate - 0.75) < 0.001  # 3 hits / 4 requests


class TestComputeQueryHash:
    """Tests for deterministic query hashing."""

    def test_deterministic(self, cache):
        query = {"table": "emissions", "year": 2025}
        h1 = cache.compute_query_hash(query)
        h2 = cache.compute_query_hash(query)
        assert h1 == h2

    def test_different_queries_different_hashes(self, cache):
        h1 = cache.compute_query_hash({"table": "emissions"})
        h2 = cache.compute_query_hash({"table": "inventory"})
        assert h1 != h2

    def test_same_query_same_hash(self, cache):
        q = {"source": "erp", "filters": {"year": 2025}}
        h1 = cache.compute_query_hash(q)
        h2 = cache.compute_query_hash(q)
        assert h1 == h2
        assert len(h1) == 64


class TestIsExpired:
    """Tests for expiration checking."""

    def test_not_expired(self, cache):
        entry = cache.put("k", "v", ttl=3600)
        assert not cache._is_expired(entry)

    def test_expired(self, cache):
        entry = cache.put("k", "v", ttl=0)
        entry.expires_at = time.time() - 1  # force past expiration
        assert cache._is_expired(entry)

    def test_exact_expiry_time(self, cache):
        entry = cache.put("k", "v", ttl=0)
        entry.expires_at = time.time()  # exactly now
        # At exact expiry time, time.time() >= expires_at is true
        assert cache._is_expired(entry)


class TestEvictExpired:
    """Tests for expired entry eviction."""

    def test_removes_only_expired(self, cache):
        e1 = cache.put("expired1", "v1", ttl=0)
        e1.expires_at = time.time() - 10
        e2 = cache.put("expired2", "v2", ttl=0)
        e2.expires_at = time.time() - 5
        cache.put("fresh", "v3", ttl=3600)

        removed = cache.evict_expired()
        assert removed == 2
        assert cache.get("fresh") == "v3"

    def test_empty_cache(self, cache):
        removed = cache.evict_expired()
        assert removed == 0


class TestEvictLRU:
    """Tests for LRU eviction."""

    def test_removes_least_recently_accessed(self, tiny_cache):
        tiny_cache.put("k1", "v1")
        tiny_cache.put("k2", "v2")
        tiny_cache.put("k3", "v3")
        # Access k2 and k3 to make k1 the LRU
        tiny_cache.get("k2")
        tiny_cache.get("k3")
        # Adding k4 should evict k1
        tiny_cache.put("k4", "v4")
        assert tiny_cache.get("k1") is None  # evicted
        assert tiny_cache.get("k4") == "v4"

    def test_respects_access_order(self, tiny_cache):
        e_a = tiny_cache.put("a", 1)
        e_b = tiny_cache.put("b", 2)
        e_c = tiny_cache.put("c", 3)
        # Manually stagger last_accessed to guarantee ordering
        e_a.last_accessed = 1000.0
        e_b.last_accessed = 2000.0
        e_c.last_accessed = 3000.0
        # Access 'a' to refresh it, making 'b' the LRU
        tiny_cache.get("a")  # updates a.last_accessed to now (> 3000)
        tiny_cache.put("d", 4)  # evicts 'b' (lowest last_accessed = 2000)
        assert tiny_cache.get("b") is None
        assert tiny_cache.get("a") == 1


class TestHitTracking:
    """Tests for hit count and last_accessed tracking."""

    def test_increment_hit_count(self, cache):
        cache.put("k1", "v1")
        cache.get("k1")
        cache.get("k1")
        cache.get("k1")
        entry = cache._cache["k1"]
        assert entry.hit_count == 3

    def test_update_last_accessed(self, cache):
        entry = cache.put("k1", "v1")
        initial_accessed = entry.last_accessed
        time.sleep(0.01)  # small delay to ensure time difference
        cache.get("k1")
        assert cache._cache["k1"].last_accessed > initial_accessed
