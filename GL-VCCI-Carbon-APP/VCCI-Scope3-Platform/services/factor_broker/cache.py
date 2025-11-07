"""
Factor Cache Implementation
GL-VCCI Scope 3 Platform

Redis-based caching for emission factors with license compliance (24-hour TTL).
Implements cache key generation, invalidation, and performance monitoring.

Version: 1.0.0
"""

import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import hashlib
import redis
from redis.exceptions import RedisError

from .models import FactorRequest, FactorResponse
from .exceptions import CacheError, LicenseViolationError
from .config import CacheConfig


logger = logging.getLogger(__name__)


class FactorCache:
    """
    Redis-based cache for emission factors.

    Implements:
    - License-compliant caching (24-hour TTL for ecoinvent)
    - Efficient key generation
    - Pattern-based invalidation
    - Cache statistics tracking

    Attributes:
        config: Cache configuration
        redis_client: Redis client instance
        hit_count: Number of cache hits
        miss_count: Number of cache misses
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize factor cache.

        Args:
            config: Cache configuration

        Raises:
            CacheError: If Redis connection fails
        """
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.hit_count = 0
        self.miss_count = 0

        if config.enabled:
            self._connect()

    def _connect(self):
        """
        Connect to Redis server.

        Raises:
            CacheError: If connection fails
        """
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,  # Decode bytes to strings
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test connection
            self.redis_client.ping()

            logger.info(
                f"Connected to Redis at {self.config.redis_host}:"
                f"{self.config.redis_port}/{self.config.redis_db}"
            )

        except RedisError as e:
            raise CacheError(
                operation="connect",
                reason=f"Failed to connect to Redis: {e}",
                original_exception=e
            )

    def _generate_cache_key(
        self,
        request: FactorRequest
    ) -> str:
        """
        Generate cache key from factor request.

        Format: {prefix}:factor:{product}:{region}:{gwp}:{unit}:{year}

        Args:
            request: Factor request

        Returns:
            Cache key string
        """
        # Normalize components
        product = request.product.lower().strip().replace(" ", "_")
        region = request.region.upper()
        gwp = request.gwp_standard.value
        unit = (request.unit or "default").lower().replace("/", "_per_")
        year = request.year or "latest"

        # Generate key
        key = (
            f"{self.config.key_prefix}:factor:"
            f"{product}:{region}:{gwp}:{unit}:{year}"
        )

        return key

    def _generate_cache_key_hash(
        self,
        request: FactorRequest
    ) -> str:
        """
        Generate cache key using hash for very long product names.

        Args:
            request: Factor request

        Returns:
            Hashed cache key
        """
        key_data = (
            f"{request.product}:"
            f"{request.region}:"
            f"{request.gwp_standard.value}:"
            f"{request.unit or 'default'}:"
            f"{request.year or 'latest'}"
        )

        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]

        return f"{self.config.key_prefix}:factor:hash:{key_hash}"

    def _serialize_response(
        self,
        response: FactorResponse
    ) -> str:
        """
        Serialize factor response to JSON.

        Args:
            response: Factor response

        Returns:
            JSON string
        """
        # Convert to dict and handle datetime serialization
        data = response.dict()

        # Convert datetime objects to ISO format strings
        self._convert_datetimes_to_iso(data)

        return json.dumps(data)

    def _convert_datetimes_to_iso(self, obj: Any):
        """
        Recursively convert datetime objects to ISO format strings.

        Args:
            obj: Object to convert (dict, list, or value)
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, datetime):
                    obj[key] = value.isoformat()
                elif isinstance(value, (dict, list)):
                    self._convert_datetimes_to_iso(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, datetime):
                    obj[i] = item.isoformat()
                elif isinstance(item, (dict, list)):
                    self._convert_datetimes_to_iso(item)

    def _deserialize_response(
        self,
        data: str
    ) -> FactorResponse:
        """
        Deserialize JSON to factor response.

        Args:
            data: JSON string

        Returns:
            FactorResponse instance
        """
        response_dict = json.loads(data)
        return FactorResponse(**response_dict)

    def _check_ttl_compliance(self, ttl_seconds: Optional[int] = None):
        """
        Check if TTL complies with license terms.

        Args:
            ttl_seconds: TTL in seconds

        Raises:
            LicenseViolationError: If TTL exceeds 24 hours
        """
        ttl = ttl_seconds or self.config.ttl_seconds

        # ecoinvent license allows max 24 hours caching
        max_allowed_ttl = 86400  # 24 hours

        if ttl > max_allowed_ttl:
            raise LicenseViolationError(
                violation_type="cache_ttl_exceeded",
                license_source="ecoinvent",
                details_dict={
                    "requested_ttl": ttl,
                    "max_allowed_ttl": max_allowed_ttl,
                    "ttl_hours": ttl / 3600
                }
            )

    async def get(
        self,
        request: FactorRequest
    ) -> Optional[FactorResponse]:
        """
        Get cached factor response.

        Args:
            request: Factor request

        Returns:
            FactorResponse if found in cache, None otherwise

        Raises:
            CacheError: If cache operation fails
        """
        if not self.config.enabled or not self.redis_client:
            return None

        try:
            # Generate cache key
            key = self._generate_cache_key(request)

            # Get from Redis
            cached_data = self.redis_client.get(key)

            if cached_data:
                # Cache hit
                self.hit_count += 1
                response = self._deserialize_response(cached_data)

                # Mark as cache hit in provenance
                response.provenance.cache_hit = True

                logger.debug(f"Cache hit for key: {key}")
                return response
            else:
                # Cache miss
                self.miss_count += 1
                logger.debug(f"Cache miss for key: {key}")
                return None

        except RedisError as e:
            # Log error but don't fail - cache is not critical
            logger.error(f"Cache get error: {e}", exc_info=True)
            return None

        except Exception as e:
            logger.error(f"Cache deserialization error: {e}", exc_info=True)
            return None

    async def set(
        self,
        request: FactorRequest,
        response: FactorResponse,
        ttl_seconds: Optional[int] = None
    ):
        """
        Set factor response in cache.

        Args:
            request: Factor request
            response: Factor response
            ttl_seconds: Time-to-live in seconds (defaults to config value)

        Raises:
            CacheError: If cache operation fails
            LicenseViolationError: If TTL exceeds license terms
        """
        if not self.config.enabled or not self.redis_client:
            return

        try:
            # Check license compliance
            ttl = ttl_seconds or self.config.ttl_seconds
            self._check_ttl_compliance(ttl)

            # Generate cache key
            key = self._generate_cache_key(request)

            # Serialize response
            serialized = self._serialize_response(response)

            # Set in Redis with TTL
            self.redis_client.setex(
                name=key,
                time=ttl,
                value=serialized
            )

            logger.debug(f"Cached factor for key: {key} (TTL: {ttl}s)")

        except (RedisError, LicenseViolationError) as e:
            raise CacheError(
                operation="set",
                reason=str(e),
                original_exception=e
            )

        except Exception as e:
            logger.error(f"Cache set error: {e}", exc_info=True)
            raise CacheError(
                operation="set",
                reason="Serialization error",
                original_exception=e
            )

    async def invalidate(
        self,
        request: FactorRequest
    ):
        """
        Invalidate cached factor.

        Args:
            request: Factor request

        Raises:
            CacheError: If cache operation fails
        """
        if not self.config.enabled or not self.redis_client:
            return

        try:
            key = self._generate_cache_key(request)
            self.redis_client.delete(key)
            logger.info(f"Invalidated cache for key: {key}")

        except RedisError as e:
            raise CacheError(
                operation="invalidate",
                reason=str(e),
                original_exception=e
            )

    async def invalidate_pattern(
        self,
        pattern: str
    ) -> int:
        """
        Invalidate all cache entries matching pattern.

        Args:
            pattern: Redis key pattern (e.g., "factor:steel:*")

        Returns:
            Number of keys invalidated

        Raises:
            CacheError: If cache operation fails
        """
        if not self.config.enabled or not self.redis_client:
            return 0

        try:
            # Build full pattern
            full_pattern = f"{self.config.key_prefix}:{pattern}"

            # Find matching keys
            keys = self.redis_client.keys(full_pattern)

            if keys:
                # Delete all matching keys
                count = self.redis_client.delete(*keys)
                logger.info(
                    f"Invalidated {count} cache entries matching pattern: {full_pattern}"
                )
                return count
            else:
                return 0

        except RedisError as e:
            raise CacheError(
                operation="invalidate_pattern",
                reason=str(e),
                original_exception=e
            )

    async def invalidate_source(
        self,
        source: str
    ) -> int:
        """
        Invalidate all cache entries from a specific source.

        Args:
            source: Source name (e.g., "ecoinvent", "desnz_uk")

        Returns:
            Number of keys invalidated

        Raises:
            CacheError: If cache operation fails
        """
        # Pattern to match all factors from this source
        # Note: This is approximate - actual implementation would need
        # source tracking in cache keys
        pattern = f"factor:*"  # Would need refinement

        logger.warning(
            f"Source-specific invalidation not fully implemented. "
            f"Use invalidate_all() or specific patterns."
        )

        return 0

    async def invalidate_all(self) -> int:
        """
        Invalidate all cached factors.

        Returns:
            Number of keys invalidated

        Raises:
            CacheError: If cache operation fails
        """
        return await self.invalidate_pattern("factor:*")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = (
            self.hit_count / total_requests if total_requests > 0 else 0.0
        )

        stats = {
            "enabled": self.config.enabled,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "ttl_seconds": self.config.ttl_seconds,
            "ttl_hours": self.config.ttl_seconds / 3600
        }

        if self.redis_client:
            try:
                # Get Redis info
                info = self.redis_client.info()
                stats["redis_used_memory_mb"] = (
                    info.get("used_memory", 0) / 1024 / 1024
                )
                stats["redis_connected_clients"] = info.get("connected_clients", 0)

                # Count factor keys
                factor_keys = self.redis_client.keys(
                    f"{self.config.key_prefix}:factor:*"
                )
                stats["cached_factors_count"] = len(factor_keys)

            except RedisError as e:
                logger.error(f"Error getting Redis stats: {e}")

        return stats

    def reset_stats(self):
        """Reset cache statistics counters."""
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache statistics reset")

    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FactorCache(enabled={self.config.enabled}, "
            f"ttl={self.config.ttl_seconds}s, "
            f"hit_rate={self.hit_count}/{self.hit_count + self.miss_count})"
        )
