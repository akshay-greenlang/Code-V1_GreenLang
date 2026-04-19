# -*- coding: utf-8 -*-
"""
Redis Cache Client for GreenLang Emission Factor Database

Implements cache-aside pattern for emission factor lookups with:
- TTL management (default 24 hours)
- Automatic fallback to direct database if Redis unavailable
- Cache invalidation for data updates
- Connection pooling for performance
- Serialization/deserialization of Pydantic models
"""

import json
import hashlib
import logging
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field
import asyncio

logger = logging.getLogger(__name__)

# Type variable for generic cache operations
T = TypeVar('T')


class CacheStatus(str, Enum):
    """Cache operation status."""
    HIT = "hit"
    MISS = "miss"
    ERROR = "error"
    STALE = "stale"
    BYPASS = "bypass"


class RedisCacheConfig(BaseModel):
    """Configuration for Redis cache connection."""

    host: str = Field(default="localhost", description="Redis server hostname")
    port: int = Field(default=6379, description="Redis server port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")

    # Connection pool settings
    max_connections: int = Field(default=10, description="Max pool connections")
    socket_timeout: float = Field(default=5.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(default=5.0, description="Connection timeout")

    # TTL settings
    default_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        description="Default cache TTL in seconds"
    )
    emission_factor_ttl_seconds: int = Field(
        default=86400,  # 24 hours
        description="TTL for emission factor cache entries"
    )
    grid_intensity_ttl_seconds: int = Field(
        default=3600,  # 1 hour (more volatile)
        description="TTL for grid intensity cache entries"
    )

    # Key prefixes
    key_prefix: str = Field(default="greenlang:", description="Cache key prefix")
    emission_factor_prefix: str = Field(default="ef:", description="Emission factor key prefix")
    grid_intensity_prefix: str = Field(default="grid:", description="Grid intensity key prefix")

    # Behavior settings
    enable_fallback: bool = Field(
        default=True,
        description="Fall back to direct DB if cache unavailable"
    )
    log_cache_hits: bool = Field(default=False, description="Log cache hits")
    log_cache_misses: bool = Field(default=True, description="Log cache misses")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    ttl_seconds: Optional[int] = None
    source: str = "cache"
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value if isinstance(self.value, (dict, list, str, int, float, bool, type(None))) else str(self.value),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "ttl_seconds": self.ttl_seconds,
            "source": self.source,
            "hit_count": self.hit_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            ttl_seconds=data.get("ttl_seconds"),
            source=data.get("source", "cache"),
            hit_count=data.get("hit_count", 0)
        )


class CacheMetrics:
    """Cache performance metrics."""

    def __init__(self):
        self.hits: int = 0
        self.misses: int = 0
        self.errors: int = 0
        self.bypasses: int = 0
        self.sets: int = 0
        self.deletes: int = 0
        self.start_time: datetime = datetime.utcnow()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def uptime_seconds(self) -> float:
        """Get cache uptime in seconds."""
        return (datetime.utcnow() - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "bypasses": self.bypasses,
            "sets": self.sets,
            "deletes": self.deletes,
            "hit_rate": round(self.hit_rate, 4),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "start_time": self.start_time.isoformat()
        }


class RedisCache:
    """
    Redis cache client with cache-aside pattern.

    Features:
    - Automatic fallback to direct database if Redis unavailable
    - TTL management (24 hours default)
    - Connection pooling
    - Metrics collection
    - Cache invalidation
    - Pydantic model serialization
    """

    def __init__(self, config: Optional[RedisCacheConfig] = None):
        """
        Initialize Redis cache client.

        Args:
            config: Cache configuration
        """
        self.config = config or RedisCacheConfig()
        self.metrics = CacheMetrics()
        self._redis: Optional[Any] = None
        self._connected: bool = False
        self._local_cache: Dict[str, CacheEntry] = {}  # Fallback cache

        # Try to import redis
        try:
            import redis
            self._redis_module = redis
        except ImportError:
            logger.warning("Redis package not installed. Using local fallback cache.")
            self._redis_module = None

    async def connect(self) -> bool:
        """
        Connect to Redis server.

        Returns:
            True if connected, False otherwise
        """
        if self._redis_module is None:
            logger.info("Redis not available, using local cache fallback")
            return False

        try:
            self._redis = self._redis_module.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=True,
                max_connections=self.config.max_connections
            )

            # Test connection
            self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using local fallback cache.")
            self._connected = False
            return False

    def connect_sync(self) -> bool:
        """Synchronous connection method."""
        return asyncio.get_event_loop().run_until_complete(self.connect())

    def _is_available(self) -> bool:
        """Check if Redis is available."""
        if not self._connected or self._redis is None:
            return False

        try:
            self._redis.ping()
            return True
        except Exception:
            self._connected = False
            return False

    def _build_key(self, key_type: str, *parts: str) -> str:
        """Build cache key from parts."""
        base = self.config.key_prefix
        if key_type == "emission_factor":
            base += self.config.emission_factor_prefix
        elif key_type == "grid_intensity":
            base += self.config.grid_intensity_prefix

        return base + ":".join(str(p) for p in parts)

    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        elif isinstance(value, dict):
            return json.dumps(value)
        elif isinstance(value, CacheEntry):
            return json.dumps(value.to_dict())
        else:
            return json.dumps({"value": value})

    def _deserialize(self, data: str, model_class: Optional[type] = None) -> Any:
        """Deserialize value from storage."""
        try:
            parsed = json.loads(data)
            if model_class and issubclass(model_class, BaseModel):
                return model_class.model_validate(parsed)
            elif isinstance(parsed, dict) and "value" in parsed and len(parsed) == 1:
                return parsed["value"]
            return parsed
        except Exception as e:
            logger.warning(f"Failed to deserialize cache value: {e}")
            return None

    def get(
        self,
        key: str,
        key_type: str = "emission_factor",
        model_class: Optional[type] = None
    ) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key
            key_type: Type of key (for prefix)
            model_class: Optional Pydantic model class for deserialization

        Returns:
            Cached value or None if not found
        """
        full_key = self._build_key(key_type, key)

        # Try Redis first
        if self._is_available():
            try:
                data = self._redis.get(full_key)
                if data:
                    self.metrics.hits += 1
                    if self.config.log_cache_hits:
                        logger.debug(f"Cache HIT: {full_key}")
                    return self._deserialize(data, model_class)
                else:
                    self.metrics.misses += 1
                    if self.config.log_cache_misses:
                        logger.debug(f"Cache MISS: {full_key}")
                    return None
            except Exception as e:
                logger.warning(f"Redis GET error for {full_key}: {e}")
                self.metrics.errors += 1

        # Fall back to local cache
        if self.config.enable_fallback:
            entry = self._local_cache.get(full_key)
            if entry and not entry.is_expired:
                self.metrics.hits += 1
                entry.hit_count += 1
                return entry.value
            elif entry and entry.is_expired:
                del self._local_cache[full_key]

        self.metrics.misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        key_type: str = "emission_factor",
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            key_type: Type of key (for prefix and TTL)
            ttl_seconds: Custom TTL or None for default

        Returns:
            True if set successfully
        """
        full_key = self._build_key(key_type, key)

        # Determine TTL
        if ttl_seconds is None:
            if key_type == "emission_factor":
                ttl_seconds = self.config.emission_factor_ttl_seconds
            elif key_type == "grid_intensity":
                ttl_seconds = self.config.grid_intensity_ttl_seconds
            else:
                ttl_seconds = self.config.default_ttl_seconds

        serialized = self._serialize(value)

        # Try Redis first
        if self._is_available():
            try:
                self._redis.setex(full_key, ttl_seconds, serialized)
                self.metrics.sets += 1
                logger.debug(f"Cache SET: {full_key} (TTL: {ttl_seconds}s)")
                return True
            except Exception as e:
                logger.warning(f"Redis SET error for {full_key}: {e}")
                self.metrics.errors += 1

        # Fall back to local cache
        if self.config.enable_fallback:
            entry = CacheEntry(
                key=full_key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds),
                ttl_seconds=ttl_seconds,
                source="local"
            )
            self._local_cache[full_key] = entry
            self.metrics.sets += 1
            return True

        return False

    def delete(self, key: str, key_type: str = "emission_factor") -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key
            key_type: Type of key (for prefix)

        Returns:
            True if deleted
        """
        full_key = self._build_key(key_type, key)

        deleted = False

        # Delete from Redis
        if self._is_available():
            try:
                result = self._redis.delete(full_key)
                deleted = result > 0
                if deleted:
                    self.metrics.deletes += 1
            except Exception as e:
                logger.warning(f"Redis DELETE error for {full_key}: {e}")

        # Delete from local cache
        if full_key in self._local_cache:
            del self._local_cache[full_key]
            deleted = True
            self.metrics.deletes += 1

        return deleted

    def invalidate_pattern(self, pattern: str, key_type: str = "emission_factor") -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "diesel/*" to invalidate all diesel factors)
            key_type: Type of key (for prefix)

        Returns:
            Number of keys invalidated
        """
        full_pattern = self._build_key(key_type, pattern)
        count = 0

        # Invalidate in Redis
        if self._is_available():
            try:
                # Replace * with Redis glob pattern
                redis_pattern = full_pattern.replace("*", "*")
                keys = self._redis.keys(redis_pattern)
                if keys:
                    count = self._redis.delete(*keys)
                    self.metrics.deletes += count
                    logger.info(f"Invalidated {count} keys matching {redis_pattern}")
            except Exception as e:
                logger.warning(f"Redis INVALIDATE error for {full_pattern}: {e}")

        # Invalidate in local cache
        local_keys = [k for k in self._local_cache.keys() if k.startswith(full_pattern.replace("*", ""))]
        for k in local_keys:
            del self._local_cache[k]
            count += 1

        return count

    def invalidate_by_version(self, version: str) -> int:
        """
        Invalidate all cache entries for a specific data version.

        Args:
            version: Data version (e.g., "defra_2024")

        Returns:
            Number of keys invalidated
        """
        return self.invalidate_pattern(f"*{version}*")

    def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if cleared successfully
        """
        cleared = False

        # Clear Redis (only our keys)
        if self._is_available():
            try:
                pattern = f"{self.config.key_prefix}*"
                keys = self._redis.keys(pattern)
                if keys:
                    self._redis.delete(*keys)
                cleared = True
                logger.info(f"Cleared {len(keys)} keys from Redis cache")
            except Exception as e:
                logger.warning(f"Redis CLEAR error: {e}")

        # Clear local cache
        self._local_cache.clear()
        cleared = True

        return cleared

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self.metrics.to_dict()

    def close(self):
        """Close Redis connection."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
        self._connected = False


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache(config: Optional[RedisCacheConfig] = None) -> RedisCache:
    """
    Get global cache instance.

    Args:
        config: Optional configuration

    Returns:
        RedisCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache(config)
        _cache_instance.connect_sync()
    return _cache_instance


def cache_emission_factor(
    fuel_type: str,
    region: str,
    year: int,
    version: str,
    value: Any,
    ttl_seconds: Optional[int] = None
) -> bool:
    """
    Cache an emission factor.

    Args:
        fuel_type: Fuel type
        region: Region code
        year: Reference year
        version: Data version
        value: Value to cache
        ttl_seconds: Optional custom TTL

    Returns:
        True if cached successfully
    """
    cache = get_cache()
    key = f"{version}:{fuel_type}:{region}:{year}"
    return cache.set(key, value, "emission_factor", ttl_seconds)


def get_cached_emission_factor(
    fuel_type: str,
    region: str,
    year: int,
    version: str,
    model_class: Optional[type] = None
) -> Optional[Any]:
    """
    Get cached emission factor.

    Args:
        fuel_type: Fuel type
        region: Region code
        year: Reference year
        version: Data version
        model_class: Optional Pydantic model class

    Returns:
        Cached value or None
    """
    cache = get_cache()
    key = f"{version}:{fuel_type}:{region}:{year}"
    return cache.get(key, "emission_factor", model_class)


def invalidate_emission_factor(
    fuel_type: Optional[str] = None,
    region: Optional[str] = None,
    version: Optional[str] = None
) -> int:
    """
    Invalidate emission factor cache entries.

    Args:
        fuel_type: Optional fuel type filter
        region: Optional region filter
        version: Optional version filter

    Returns:
        Number of entries invalidated
    """
    cache = get_cache()

    # Build pattern
    parts = []
    if version:
        parts.append(version)
    else:
        parts.append("*")

    if fuel_type:
        parts.append(fuel_type)
    else:
        parts.append("*")

    if region:
        parts.append(region)
    else:
        parts.append("*")

    pattern = ":".join(parts)
    return cache.invalidate_pattern(pattern, "emission_factor")


class CachedEmissionFactorDatabase:
    """
    Emission factor database wrapper with caching.

    Implements cache-aside pattern:
    1. Check cache first
    2. If miss, query database
    3. Cache the result
    4. Return value
    """

    def __init__(
        self,
        database: Any,
        cache: Optional[RedisCache] = None,
        cache_config: Optional[RedisCacheConfig] = None
    ):
        """
        Initialize cached database.

        Args:
            database: EmissionFactorDatabase instance
            cache: Optional Redis cache
            cache_config: Optional cache configuration
        """
        self.database = database
        self.cache = cache or get_cache(cache_config)

    def lookup(
        self,
        fuel_type: str,
        region: str,
        year: int = 2024,
        version: Optional[str] = None,
        bypass_cache: bool = False
    ) -> Optional[Any]:
        """
        Look up emission factor with caching.

        Args:
            fuel_type: Fuel type
            region: Region code
            year: Reference year
            version: Data version
            bypass_cache: Skip cache lookup

        Returns:
            EmissionFactorRecord or None
        """
        version_str = str(version) if version else "default"
        cache_key = f"{version_str}:{fuel_type}:{region}:{year}"

        # Check cache first (unless bypassed)
        if not bypass_cache:
            cached = self.cache.get(cache_key, "emission_factor")
            if cached:
                return cached

        # Query database
        result = self.database.lookup(fuel_type, region, year, version)

        # Cache result if found
        if result:
            self.cache.set(cache_key, result, "emission_factor")

        return result

    def lookup_grid_intensity(
        self,
        location: str,
        year: int = 2023,
        bypass_cache: bool = False
    ) -> Optional[Any]:
        """
        Look up grid intensity with caching.

        Args:
            location: Location code
            year: Reference year
            bypass_cache: Skip cache

        Returns:
            Grid intensity record or None
        """
        cache_key = f"{location}:{year}"

        if not bypass_cache:
            cached = self.cache.get(cache_key, "grid_intensity")
            if cached:
                return cached

        result = self.database.lookup_grid_intensity(location, year)

        if result:
            self.cache.set(cache_key, result, "grid_intensity")

        return result

    def invalidate_all(self) -> int:
        """Invalidate all cached entries."""
        count = 0
        count += self.cache.invalidate_pattern("*", "emission_factor")
        count += self.cache.invalidate_pattern("*", "grid_intensity")
        return count

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self.cache.get_metrics()
