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

from greenlang.cache import CacheManager, get_cache_manager, initialize_cache_manager

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
            CacheError: If cache initialization fails
        """
        self.config = config
        self.cache_manager: Optional[CacheManager] = None
        self.hit_count = 0
        self.miss_count = 0
        self._namespace = "factor_broker"

        if config.enabled:
            self._connect()

    def _connect(self):
        """
        Initialize CacheManager.

        Raises:
            CacheError: If connection fails
        """
        try:
            # Get global cache manager or initialize if needed
            self.cache_manager = get_cache_manager()

            if self.cache_manager is None:
                # Initialize with default architecture
                # Note: This should ideally be done at application startup
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create task
                    logger.warning(
                        "Cache manager not initialized. Factor broker cache will initialize it. "
                        "Consider initializing CacheManager at application startup."
                    )
                    # For now, we'll use a lazy initialization approach
                    self.cache_manager = None
                else:
                    # Initialize synchronously
                    self.cache_manager = loop.run_until_complete(initialize_cache_manager())

            logger.info("Factor cache connected to CacheManager")

        except Exception as e:
            raise CacheError(
                operation="connect",
                reason=f"Failed to initialize cache: {e}",
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
        if not self.config.enabled or not self.cache_manager:
            return None

        try:
            # Generate cache key
            key = self._generate_cache_key(request)

            # Get from CacheManager
            cached_data = await self.cache_manager.get(key, namespace=self._namespace)

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

        except Exception as e:
            # Log error but don't fail - cache is not critical
            logger.error(f"Cache get error: {e}", exc_info=True)
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
        if not self.config.enabled or not self.cache_manager:
            return

        try:
            # Check license compliance
            ttl = ttl_seconds or self.config.ttl_seconds
            self._check_ttl_compliance(ttl)

            # Generate cache key
            key = self._generate_cache_key(request)

            # Serialize response
            serialized = self._serialize_response(response)

            # Set in CacheManager with TTL
            await self.cache_manager.set(
                key=key,
                value=serialized,
                ttl=ttl,
                namespace=self._namespace
            )

            logger.debug(f"Cached factor for key: {key} (TTL: {ttl}s)")

        except LicenseViolationError as e:
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
        if not self.config.enabled or not self.cache_manager:
            return

        try:
            key = self._generate_cache_key(request)
            await self.cache_manager.invalidate(key, namespace=self._namespace)
            logger.info(f"Invalidated cache for key: {key}")

        except Exception as e:
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
            pattern: Key pattern (e.g., "factor:steel:*")

        Returns:
            Number of keys invalidated

        Raises:
            CacheError: If cache operation fails
        """
        if not self.config.enabled or not self.cache_manager:
            return 0

        try:
            # Build full pattern (namespace is handled by CacheManager)
            full_pattern = f"{self.config.key_prefix}:{pattern}"

            # Invalidate pattern
            count = await self.cache_manager.invalidate_pattern(
                full_pattern,
                namespace=self._namespace
            )

            logger.info(
                f"Invalidated {count} cache entries matching pattern: {full_pattern}"
            )
            return count

        except Exception as e:
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

    async def get_stats(self) -> Dict[str, Any]:
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

        if self.cache_manager:
            try:
                # Get cache manager analytics
                analytics = await self.cache_manager.get_analytics()
                stats["cache_manager_analytics"] = analytics
                stats["overall_hit_rate"] = analytics.get("overall_hit_rate", 0.0)

            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")

        return stats

    def reset_stats(self):
        """Reset cache statistics counters."""
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache statistics reset")

    async def close(self):
        """Close cache connection."""
        # Cache manager is shared globally, so we don't stop it here
        # It should be stopped at application shutdown
        logger.info("Factor cache closed (CacheManager remains active)")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FactorCache(enabled={self.config.enabled}, "
            f"ttl={self.config.ttl_seconds}s, "
            f"hit_rate={self.hit_count}/{self.hit_count + self.miss_count})"
        )
