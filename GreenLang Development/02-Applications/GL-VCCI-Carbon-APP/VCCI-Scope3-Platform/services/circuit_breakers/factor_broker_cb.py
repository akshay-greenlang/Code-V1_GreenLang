# -*- coding: utf-8 -*-
"""
Circuit Breaker for Factor Broker API Calls

Protects against failures in emission factor data sources:
- ecoinvent API
- DESNZ (UK Government) API
- EPA (US Environmental Protection Agency) API

Features:
- Separate circuit breakers for each factor source
- Fallback to cached factors when circuit is open
- Configurable thresholds per source
- Prometheus metrics for monitoring

Author: GreenLang Platform Team
Version: 1.0.0
Date: 2025-11-09
"""

from typing import Dict, Any, Optional, List
import requests
from datetime import datetime

from greenlang.determinism import DeterministicClock
from greenlang.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    create_circuit_breaker,
)
from greenlang.telemetry import get_logger
from greenlang.cache import get_cache_manager


logger = get_logger(__name__)


# ============================================================================
# FACTOR BROKER CIRCUIT BREAKER
# ============================================================================

class FactorBrokerCircuitBreaker:
    """
    Circuit breaker wrapper for emission factor broker API calls.

    Manages circuit breakers for multiple factor data sources with
    intelligent fallback to cached values when external APIs are unavailable.

    Example:
        >>> fb_cb = FactorBrokerCircuitBreaker()
        >>> factor = fb_cb.get_emission_factor(
        ...     source="ecoinvent",
        ...     activity="electricity_production",
        ...     region="US"
        ... )
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache = get_cache_manager()

        # Circuit breaker for ecoinvent API
        self.ecoinvent_cb = create_circuit_breaker(
            CircuitBreakerConfig(
                name="factor_broker_ecoinvent",
                fail_max=5,
                timeout_duration=90,  # 90 seconds before retry
                reset_timeout=60,
                fallback_function=self._fallback_ecoinvent,
            )
        )

        # Circuit breaker for DESNZ API
        self.desnz_cb = create_circuit_breaker(
            CircuitBreakerConfig(
                name="factor_broker_desnz",
                fail_max=5,
                timeout_duration=90,
                reset_timeout=60,
                fallback_function=self._fallback_desnz,
            )
        )

        # Circuit breaker for EPA API
        self.epa_cb = create_circuit_breaker(
            CircuitBreakerConfig(
                name="factor_broker_epa",
                fail_max=5,
                timeout_duration=90,
                reset_timeout=60,
                fallback_function=self._fallback_epa,
            )
        )

        self.logger.info("Factor broker circuit breakers initialized")

    def get_emission_factor(
        self,
        source: str,
        activity: str,
        region: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get emission factor with circuit breaker protection.

        Args:
            source: Factor data source (ecoinvent, desnz, epa)
            activity: Activity identifier
            region: Geographic region
            **kwargs: Additional parameters for the API

        Returns:
            Emission factor data

        Raises:
            CircuitOpenError: If circuit is open and no cached data available
            ValueError: If source is not supported
        """
        cb = self._get_circuit_breaker(source)

        try:
            return cb.call(
                self._fetch_factor,
                source=source,
                activity=activity,
                region=region,
                **kwargs
            )
        except CircuitOpenError as e:
            # Try cache as last resort
            cached = self._get_cached_factor(source, activity, region)
            if cached:
                self.logger.warning(
                    f"Circuit open for {source} - using cached factor",
                    extra={
                        "source": source,
                        "activity": activity,
                        "region": region,
                    }
                )
                return cached
            raise

    def _fetch_factor(
        self,
        source: str,
        activity: str,
        region: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch emission factor from external API.

        This is a placeholder - actual implementation would call
        the real factor broker service.
        """
        # Placeholder for actual API call
        # In production, this would call:
        # from services.factor_broker import FactorBroker
        # broker = FactorBroker()
        # return broker.fetch_factor(source, activity, region, **kwargs)

        self.logger.debug(
            f"Fetching factor from {source}",
            extra={
                "source": source,
                "activity": activity,
                "region": region,
            }
        )

        # Simulate API call for now
        # Real implementation would go here
        factor_data = {
            "source": source,
            "activity": activity,
            "region": region,
            "value": 0.5,  # kg CO2e per unit
            "unit": "kg_co2e",
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "quality": "high",
        }

        # Cache the result
        cache_key = f"factor:{source}:{activity}:{region}"
        self.cache.set(cache_key, factor_data, ttl=86400)  # 24 hours

        return factor_data

    def _get_circuit_breaker(self, source: str) -> CircuitBreaker:
        """Get the appropriate circuit breaker for the source."""
        if source == "ecoinvent":
            return self.ecoinvent_cb
        elif source == "desnz":
            return self.desnz_cb
        elif source == "epa":
            return self.epa_cb
        else:
            raise ValueError(f"Unsupported factor source: {source}")

    def _get_cached_factor(
        self,
        source: str,
        activity: str,
        region: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached emission factor if available."""
        cache_key = f"factor:{source}:{activity}:{region}"
        return self.cache.get(cache_key)

    def _fallback_ecoinvent(self, **kwargs) -> Dict[str, Any]:
        """Fallback for ecoinvent API failures."""
        return self._fallback_generic("ecoinvent", **kwargs)

    def _fallback_desnz(self, **kwargs) -> Dict[str, Any]:
        """Fallback for DESNZ API failures."""
        return self._fallback_generic("desnz", **kwargs)

    def _fallback_epa(self, **kwargs) -> Dict[str, Any]:
        """Fallback for EPA API failures."""
        return self._fallback_generic("epa", **kwargs)

    def _fallback_generic(self, source: str, **kwargs) -> Dict[str, Any]:
        """Generic fallback - try cache or return default conservative value."""
        activity = kwargs.get("activity", "unknown")
        region = kwargs.get("region", "unknown")

        # Try cache first
        cached = self._get_cached_factor(source, activity, region)
        if cached:
            self.logger.info(
                f"Using cached factor for {source}",
                extra={
                    "source": source,
                    "activity": activity,
                    "region": region,
                }
            )
            return cached

        # Return conservative default
        self.logger.warning(
            f"No cached data - using conservative default for {source}",
            extra={
                "source": source,
                "activity": activity,
                "region": region,
            }
        )

        return {
            "source": source,
            "activity": activity,
            "region": region,
            "value": 1.0,  # Conservative default
            "unit": "kg_co2e",
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "quality": "fallback",
            "note": "Conservative default - external API unavailable",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics for all sources."""
        return {
            "ecoinvent": self.ecoinvent_cb.get_stats(),
            "desnz": self.desnz_cb.get_stats(),
            "epa": self.epa_cb.get_stats(),
        }

    def reset_all(self):
        """Reset all circuit breakers."""
        self.ecoinvent_cb.reset()
        self.desnz_cb.reset()
        self.epa_cb.reset()
        self.logger.info("All factor broker circuit breakers reset")


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_instance: Optional[FactorBrokerCircuitBreaker] = None


def get_factor_broker_cb() -> FactorBrokerCircuitBreaker:
    """Get singleton instance of factor broker circuit breaker."""
    global _instance
    if _instance is None:
        _instance = FactorBrokerCircuitBreaker()
    return _instance
