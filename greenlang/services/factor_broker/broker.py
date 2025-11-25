# -*- coding: utf-8 -*-
"""
Factor Broker Core Implementation
GL-VCCI Scope 3 Platform

Main FactorBroker class that orchestrates multi-source cascading,
caching, and provenance tracking for emission factor resolution.

Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio

from greenlang.determinism import DeterministicClock
from .models import (
    FactorRequest,
    FactorResponse,
    GWPComparisonRequest,
    GWPComparisonResponse,
    HealthCheckResponse,
    GWPStandard,
    SourceType
)
from .sources import (
    FactorSource,
    EcoinventSource,
    DESNZSource,
    EPASource,
    ProxySource
)
from .cache import FactorCache
from .config import FactorBrokerConfig, SourceType as ConfigSourceType
from .exceptions import (
    FactorNotFoundError,
    SourceUnavailableError,
    ValidationError
)


logger = logging.getLogger(__name__)


class FactorBroker:
    """
    Main Factor Broker service.

    Orchestrates emission factor resolution with:
    - Multi-source cascading (ecoinvent → DESNZ → EPA → proxy)
    - License-compliant caching (24-hour TTL)
    - Provenance tracking
    - Performance monitoring

    Attributes:
        config: Factor broker configuration
        cache: Factor cache instance
        sources: Dictionary of data source instances
        performance_stats: Performance statistics tracker
    """

    def __init__(
        self,
        config: Optional[FactorBrokerConfig] = None
    ):
        """
        Initialize Factor Broker.

        Args:
            config: Configuration (defaults to loading from environment)
        """
        self.config = config or FactorBrokerConfig.from_env()

        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            raise ValidationError(
                field="configuration",
                value=None,
                reason=f"Configuration errors: {'; '.join(config_errors)}"
            )

        # Initialize cache
        self.cache = FactorCache(self.config.cache)

        # Initialize sources
        self.sources: Dict[SourceType, FactorSource] = {}
        self._initialize_sources()

        # Performance statistics
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "source_usage": {
                "ecoinvent": 0,
                "desnz_uk": 0,
                "epa_us": 0,
                "proxy": 0
            },
            "total_latency_ms": 0.0,
            "min_latency_ms": float('inf'),
            "max_latency_ms": 0.0
        }

        logger.info("FactorBroker initialized successfully")

    def _initialize_sources(self):
        """Initialize all configured data sources."""
        # Ecoinvent
        if self.config.is_source_enabled(ConfigSourceType.ECOINVENT):
            ecoinvent_config = self.config.get_source_config(ConfigSourceType.ECOINVENT)
            if ecoinvent_config:
                try:
                    self.sources[SourceType.ECOINVENT] = EcoinventSource(ecoinvent_config)
                    logger.info("Initialized ecoinvent source")
                except Exception as e:
                    logger.error(f"Failed to initialize ecoinvent source: {e}")

        # DESNZ UK
        if self.config.is_source_enabled(ConfigSourceType.DESNZ_UK):
            desnz_config = self.config.get_source_config(ConfigSourceType.DESNZ_UK)
            if desnz_config:
                try:
                    self.sources[SourceType.DESNZ_UK] = DESNZSource(desnz_config)
                    logger.info("Initialized DESNZ UK source")
                except Exception as e:
                    logger.error(f"Failed to initialize DESNZ source: {e}")

        # EPA US
        if self.config.is_source_enabled(ConfigSourceType.EPA_US):
            epa_config = self.config.get_source_config(ConfigSourceType.EPA_US)
            if epa_config:
                try:
                    self.sources[SourceType.EPA_US] = EPASource(epa_config)
                    logger.info("Initialized EPA US source")
                except Exception as e:
                    logger.error(f"Failed to initialize EPA source: {e}")

        # Proxy (always enabled as fallback)
        proxy_config = self.config.get_source_config(ConfigSourceType.PROXY)
        if proxy_config:
            self.sources[SourceType.PROXY] = ProxySource(
                proxy_config,
                self.config.proxy
            )
            logger.info("Initialized proxy source")

    async def resolve(
        self,
        request: FactorRequest
    ) -> FactorResponse:
        """
        Resolve emission factor with cascading fallback.

        Flow:
        1. Check cache
        2. Try ecoinvent
        3. Try DESNZ (if UK/EU)
        4. Try EPA (if US)
        5. Generate proxy factor
        6. Cache result
        7. Return response

        Args:
            request: Factor request

        Returns:
            FactorResponse with factor data and provenance

        Raises:
            FactorNotFoundError: If factor cannot be resolved
            ValidationError: If request is invalid
        """
        start_time = DeterministicClock.utcnow()
        self.performance_stats["total_requests"] += 1

        try:
            # Step 1: Check cache
            cached_response = await self.cache.get(request)
            if cached_response:
                self.performance_stats["cache_hits"] += 1
                self._update_performance_stats(start_time, success=True)
                logger.info(f"Cache hit for {request.product} ({request.region})")
                return cached_response

            # Step 2: Cascade through sources
            fallback_chain = []
            response = None

            for source_type in self.config.get_cascade_order():
                if source_type not in self.sources:
                    continue

                source = self.sources[source_type]
                fallback_chain.append(source_type.value)

                try:
                    logger.debug(
                        f"Attempting to fetch factor from {source_type.value} "
                        f"for {request.product} ({request.region})"
                    )

                    response = await source.fetch_factor(request)

                    if response:
                        # Update source usage stats
                        self.performance_stats["source_usage"][source_type.value] += 1

                        # Update fallback chain in provenance
                        response.provenance.fallback_chain = fallback_chain

                        # Cache the response
                        await self.cache.set(request, response)

                        # Log successful resolution
                        logger.info(
                            f"Resolved factor for {request.product} ({request.region}) "
                            f"from {source_type.value}"
                        )

                        self._update_performance_stats(start_time, success=True)
                        self.performance_stats["successful_requests"] += 1

                        return response

                except SourceUnavailableError as e:
                    logger.warning(
                        f"Source {source_type.value} unavailable: {e.message}"
                    )
                    continue

                except Exception as e:
                    logger.error(
                        f"Error fetching from {source_type.value}: {e}",
                        exc_info=True
                    )
                    continue

            # If we get here, no source could provide the factor
            self._update_performance_stats(start_time, success=False)
            self.performance_stats["failed_requests"] += 1

            raise FactorNotFoundError(
                product=request.product,
                region=request.region,
                gwp_standard=request.gwp_standard.value,
                tried_sources=fallback_chain,
                suggestions=self._get_product_suggestions(request.product)
            )

        except FactorNotFoundError:
            raise

        except Exception as e:
            self._update_performance_stats(start_time, success=False)
            self.performance_stats["failed_requests"] += 1
            logger.error(f"Unexpected error in resolve: {e}", exc_info=True)
            raise

    async def compare_gwp_standards(
        self,
        request: GWPComparisonRequest
    ) -> GWPComparisonResponse:
        """
        Compare emission factors between AR5 and AR6 GWP standards.

        Args:
            request: GWP comparison request

        Returns:
            GWPComparisonResponse with both factors and difference

        Raises:
            FactorNotFoundError: If factor cannot be resolved for either standard
        """
        # Fetch AR5 factor
        ar5_request = FactorRequest(
            product=request.product,
            region=request.region,
            gwp_standard=GWPStandard.AR5,
            unit=request.unit
        )
        ar5_response = await self.resolve(ar5_request)

        # Fetch AR6 factor
        ar6_request = FactorRequest(
            product=request.product,
            region=request.region,
            gwp_standard=GWPStandard.AR6,
            unit=request.unit
        )
        ar6_response = await self.resolve(ar6_request)

        # Calculate differences
        difference_absolute = ar6_response.value - ar5_response.value
        difference_percent = (
            (difference_absolute / ar5_response.value) * 100
            if ar5_response.value > 0 else 0.0
        )

        return GWPComparisonResponse(
            product=request.product,
            region=request.region,
            ar5=ar5_response,
            ar6=ar6_response,
            difference_percent=difference_percent,
            difference_absolute=difference_absolute
        )

    async def health_check(self) -> HealthCheckResponse:
        """
        Perform health check on Factor Broker and all sources.

        Returns:
            HealthCheckResponse with service status
        """
        # Check each data source
        source_statuses = {}

        for source_type, source in self.sources.items():
            try:
                source_health = await source.health_check()
                source_statuses[source_type.value] = source_health
            except Exception as e:
                source_statuses[source_type.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }

        # Calculate overall status
        unhealthy_sources = [
            name for name, status in source_statuses.items()
            if status.get("status") == "unhealthy"
        ]

        if not unhealthy_sources:
            overall_status = "healthy"
        elif len(unhealthy_sources) < len(source_statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # Get cache stats
        cache_stats = self.cache.get_stats()
        cache_hit_rate = cache_stats.get("hit_rate", 0.0)

        # Calculate average latency
        avg_latency_ms = (
            self.performance_stats["total_latency_ms"] /
            self.performance_stats["total_requests"]
            if self.performance_stats["total_requests"] > 0 else 0.0
        )

        return HealthCheckResponse(
            status=overall_status,
            cache_hit_rate=cache_hit_rate,
            average_latency_ms=avg_latency_ms,
            data_sources=source_statuses,
            timestamp=DeterministicClock.utcnow()
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        stats = self.performance_stats.copy()

        # Add cache stats
        stats["cache"] = self.cache.get_stats()

        # Calculate success rate
        total = stats["total_requests"]
        stats["success_rate"] = (
            stats["successful_requests"] / total if total > 0 else 0.0
        )

        # Calculate average latency
        stats["average_latency_ms"] = (
            stats["total_latency_ms"] / total if total > 0 else 0.0
        )

        return stats

    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "source_usage": {
                "ecoinvent": 0,
                "desnz_uk": 0,
                "epa_us": 0,
                "proxy": 0
            },
            "total_latency_ms": 0.0,
            "min_latency_ms": float('inf'),
            "max_latency_ms": 0.0
        }
        self.cache.reset_stats()
        logger.info("Performance statistics reset")

    def _update_performance_stats(
        self,
        start_time: datetime,
        success: bool
    ):
        """
        Update performance statistics.

        Args:
            start_time: Request start time
            success: Whether request was successful
        """
        latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000

        self.performance_stats["total_latency_ms"] += latency_ms
        self.performance_stats["min_latency_ms"] = min(
            self.performance_stats["min_latency_ms"],
            latency_ms
        )
        self.performance_stats["max_latency_ms"] = max(
            self.performance_stats["max_latency_ms"],
            latency_ms
        )

    def _get_product_suggestions(
        self,
        product: str,
        max_suggestions: int = 3
    ) -> List[str]:
        """
        Get product name suggestions for fuzzy matching.

        Args:
            product: Product name that was not found
            max_suggestions: Maximum number of suggestions

        Returns:
            List of suggested product names
        """
        # TODO: Implement fuzzy matching against known products
        # For now, return empty list
        return []

    async def close(self):
        """
        Close all connections and cleanup resources.
        """
        # Close cache
        self.cache.close()

        # Close all source connections
        for source in self.sources.values():
            if hasattr(source, 'close'):
                await source.close()

        logger.info("FactorBroker closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FactorBroker("
            f"sources={list(self.sources.keys())}, "
            f"cache_enabled={self.config.cache.enabled}, "
            f"requests={self.performance_stats['total_requests']}"
            f")"
        )
