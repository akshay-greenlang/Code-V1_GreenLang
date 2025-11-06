"""
greenlang/cache/emission_factor_cache.py

Emission Factor Cache for FuelAgentAI v2

OBJECTIVE:
Achieve 95% cache hit rate to reduce database lookups and improve performance

FEATURES:
- LRU (Least Recently Used) eviction policy
- TTL (Time To Live) for cache entries
- Thread-safe operations
- Cache statistics tracking (hits, misses, hit rate)
- Memory-efficient storage
- Cache warming for common factors
- Invalidation strategies

PERFORMANCE TARGETS:
- 95% hit rate for typical workloads
- <1ms lookup time (cache hit)
- <10MB memory footprint (1000 factors)
- Thread-safe for concurrent access

CACHE KEY FORMAT:
fuel_type:unit:geography:scope:boundary:gwp_set
Example: diesel:gallons:US:1:combustion:IPCC_AR6_100

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import RLock
import hashlib
import json


class CacheEntry:
    """
    Represents a single cache entry with TTL tracking.
    """

    def __init__(self, value: Any, ttl_seconds: int = 3600):
        """
        Initialize cache entry.

        Args:
            value: Cached value (EmissionFactorRecord)
            ttl_seconds: Time to live in seconds (default: 1 hour)
        """
        self.value = value
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
        self.last_accessed = datetime.now()

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def access(self) -> Any:
        """Access cache entry and update statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value


class EmissionFactorCache:
    """
    LRU cache with TTL for emission factor lookups.

    Features:
    - LRU eviction when max_size reached
    - TTL-based expiration
    - Thread-safe operations
    - Hit/miss statistics
    - Cache warming for common factors
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        enable_stats: bool = True,
    ):
        """
        Initialize emission factor cache.

        Args:
            max_size: Maximum number of entries (default: 1000)
            ttl_seconds: Time to live in seconds (default: 1 hour)
            enable_stats: Enable statistics tracking (default: True)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_stats = enable_stats

        # LRU cache storage (OrderedDict maintains insertion order)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Thread safety
        self._lock = RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _make_cache_key(
        self,
        fuel_type: str,
        unit: str,
        geography: str = "US",
        scope: str = "1",
        boundary: str = "combustion",
        gwp_set: str = "IPCC_AR6_100",
    ) -> str:
        """
        Generate cache key from lookup parameters.

        Format: fuel_type:unit:geography:scope:boundary:gwp_set
        Example: diesel:gallons:US:1:combustion:IPCC_AR6_100

        Args:
            fuel_type: Fuel type
            unit: Unit
            geography: Geography/country
            scope: GHG scope
            boundary: Emission boundary
            gwp_set: GWP reference set

        Returns:
            Cache key string
        """
        # Normalize parameters
        fuel_type = fuel_type.lower().strip()
        unit = unit.lower().strip()
        geography = geography.upper().strip()
        scope = scope.strip()
        boundary = boundary.lower().strip()
        gwp_set = gwp_set.upper().strip()

        # Build key
        key = f"{fuel_type}:{unit}:{geography}:{scope}:{boundary}:{gwp_set}"
        return key

    def get(
        self,
        fuel_type: str,
        unit: str,
        geography: str = "US",
        scope: str = "1",
        boundary: str = "combustion",
        gwp_set: str = "IPCC_AR6_100",
    ) -> Optional[Any]:
        """
        Get emission factor from cache.

        Args:
            fuel_type: Fuel type
            unit: Unit
            geography: Geography/country
            scope: GHG scope
            boundary: Emission boundary
            gwp_set: GWP reference set

        Returns:
            Cached EmissionFactorRecord or None if not found/expired
        """
        key = self._make_cache_key(fuel_type, unit, geography, scope, boundary, gwp_set)

        with self._lock:
            # Check if key exists
            if key not in self._cache:
                if self.enable_stats:
                    self._misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                if self.enable_stats:
                    self._expirations += 1
                    self._misses += 1
                return None

            # Move to end (LRU: most recently used)
            self._cache.move_to_end(key)

            # Record hit
            if self.enable_stats:
                self._hits += 1

            return entry.access()

    def put(
        self,
        fuel_type: str,
        unit: str,
        value: Any,
        geography: str = "US",
        scope: str = "1",
        boundary: str = "combustion",
        gwp_set: str = "IPCC_AR6_100",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Put emission factor into cache.

        Args:
            fuel_type: Fuel type
            unit: Unit
            value: EmissionFactorRecord to cache
            geography: Geography/country
            scope: GHG scope
            boundary: Emission boundary
            gwp_set: GWP reference set
            ttl_seconds: Override default TTL (optional)
        """
        key = self._make_cache_key(fuel_type, unit, geography, scope, boundary, gwp_set)
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

        with self._lock:
            # Check if we need to evict (LRU)
            if key not in self._cache and len(self._cache) >= self.max_size:
                # Remove oldest entry (FIFO for LRU)
                self._cache.popitem(last=False)
                if self.enable_stats:
                    self._evictions += 1

            # Add/update entry
            self._cache[key] = CacheEntry(value, ttl)

            # Move to end (most recently used)
            self._cache.move_to_end(key)

    def invalidate(
        self,
        fuel_type: Optional[str] = None,
        geography: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries matching criteria.

        Args:
            fuel_type: Invalidate all entries for this fuel type (optional)
            geography: Invalidate all entries for this geography (optional)

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if fuel_type is None and geography is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                return count

            # Selective invalidation
            keys_to_remove = []
            for key in self._cache.keys():
                parts = key.split(":")
                if len(parts) >= 6:
                    key_fuel = parts[0]
                    key_geo = parts[2]

                    match = True
                    if fuel_type and key_fuel != fuel_type.lower():
                        match = False
                    if geography and key_geo != geography.upper():
                        match = False

                    if match:
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]

            return len(keys_to_remove)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            # Reset statistics
            if self.enable_stats:
                self._hits = 0
                self._misses = 0
                self._evictions = 0
                self._expirations = 0

    def warm_cache(self, common_factors: Dict[Tuple, Any]) -> int:
        """
        Warm cache with common emission factors.

        Args:
            common_factors: Dict of (fuel_type, unit, geography, ...) -> EmissionFactorRecord

        Returns:
            Number of entries added
        """
        count = 0
        for params, factor in common_factors.items():
            if len(params) == 6:
                fuel_type, unit, geography, scope, boundary, gwp_set = params
                self.put(fuel_type, unit, factor, geography, scope, boundary, gwp_set)
                count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, size, etc.
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "hit_rate_pct": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "utilization_pct": (len(self._cache) / self.max_size * 100),
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def get_entry_stats(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get statistics about cache entries.

        Args:
            top_n: Number of top entries to return

        Returns:
            Dict with top accessed entries, age distribution, etc.
        """
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                age_seconds = (datetime.now() - entry.created_at).total_seconds()
                entries.append({
                    "key": key,
                    "access_count": entry.access_count,
                    "age_seconds": age_seconds,
                    "expired": entry.is_expired(),
                })

            # Sort by access count (descending)
            entries.sort(key=lambda x: x["access_count"], reverse=True)

            return {
                "total_entries": len(entries),
                "top_accessed": entries[:top_n],
                "avg_access_count": sum(e["access_count"] for e in entries) / len(entries) if entries else 0,
                "avg_age_seconds": sum(e["age_seconds"] for e in entries) / len(entries) if entries else 0,
            }

    def __len__(self) -> int:
        """Get cache size."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key is in cache (doesn't update LRU order)."""
        with self._lock:
            return key in self._cache and not self._cache[key].is_expired()


# ==================== GLOBAL CACHE INSTANCE ====================

# Global cache instance (singleton pattern)
_global_cache: Optional[EmissionFactorCache] = None


def get_global_cache(
    max_size: int = 1000,
    ttl_seconds: int = 3600,
    enable_stats: bool = True,
) -> EmissionFactorCache:
    """
    Get or create global cache instance.

    Args:
        max_size: Maximum cache size
        ttl_seconds: TTL in seconds
        enable_stats: Enable statistics

    Returns:
        Global EmissionFactorCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = EmissionFactorCache(max_size, ttl_seconds, enable_stats)
    return _global_cache


def reset_global_cache() -> None:
    """Reset global cache instance."""
    global _global_cache
    _global_cache = None
