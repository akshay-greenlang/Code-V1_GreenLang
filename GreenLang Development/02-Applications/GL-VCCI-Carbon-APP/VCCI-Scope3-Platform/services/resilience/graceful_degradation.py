# -*- coding: utf-8 -*-
"""Graceful Degradation for GL-VCCI Scope 3 Platform.

Tier-based degradation strategy to maintain service availability:
- Tier 1: Full functionality (all APIs working)
- Tier 2: Core functionality only (some APIs down)
- Tier 3: Read-only mode (all external APIs down)
- Tier 4: Maintenance mode (database issues)

Author: Team 2 - Resilience Patterns
Date: November 2025
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Degradation Tiers
# ==============================================================================


class DegradationTier(int, Enum):
    """Service degradation tiers (lower is better)."""
    TIER_1_FULL = 1  # Full functionality
    TIER_2_CORE = 2  # Core functionality only
    TIER_3_READONLY = 3  # Read-only mode
    TIER_4_MAINTENANCE = 4  # Maintenance mode

    def __str__(self) -> str:
        """String representation."""
        return self.name.replace("_", " ").title()

    @property
    def allows_writes(self) -> bool:
        """Check if tier allows write operations.

        Returns:
            True if writes allowed
        """
        return self.value <= DegradationTier.TIER_2_CORE.value

    @property
    def allows_external_apis(self) -> bool:
        """Check if tier allows external API calls.

        Returns:
            True if external APIs allowed
        """
        return self.value <= DegradationTier.TIER_2_CORE.value

    @property
    def allows_llm_inference(self) -> bool:
        """Check if tier allows LLM inference.

        Returns:
            True if LLM inference allowed
        """
        return self.value == DegradationTier.TIER_1_FULL.value

    @property
    def allows_complex_calculations(self) -> bool:
        """Check if tier allows complex calculations.

        Returns:
            True if complex calculations allowed
        """
        return self.value <= DegradationTier.TIER_2_CORE.value


# ==============================================================================
# Service Health
# ==============================================================================


class ServiceStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Service health status.

    Attributes:
        name: Service name
        status: Current status
        last_check: Last health check timestamp
        failure_count: Number of consecutive failures
        error_message: Last error message (if any)
        response_time_ms: Last response time in milliseconds
    """
    name: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: Optional[datetime] = None
    failure_count: int = 0
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None

    def is_healthy(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy
        """
        return self.status == ServiceStatus.HEALTHY

    def is_down(self) -> bool:
        """Check if service is down.

        Returns:
            True if down
        """
        return self.status == ServiceStatus.DOWN

    def mark_success(self, response_time_ms: Optional[float] = None) -> None:
        """Mark health check as successful.

        Args:
            response_time_ms: Response time in milliseconds
        """
        self.status = ServiceStatus.HEALTHY
        self.failure_count = 0
        self.error_message = None
        self.response_time_ms = response_time_ms
        self.last_check = DeterministicClock.now()

    def mark_failure(self, error: str) -> None:
        """Mark health check as failed.

        Args:
            error: Error message
        """
        self.failure_count += 1
        self.error_message = error
        self.last_check = DeterministicClock.now()

        # Update status based on failure count
        if self.failure_count >= 3:
            self.status = ServiceStatus.DOWN
        else:
            self.status = ServiceStatus.DEGRADED


# ==============================================================================
# Degradation Manager
# ==============================================================================


class DegradationManager:
    """Manages graceful degradation based on service health.

    Monitors dependent services and adjusts functionality tier
    based on availability.

    Example:
        >>> manager = DegradationManager()
        >>> manager.register_service("factor_api", critical=True)
        >>> manager.register_service("llm_api", critical=False)
        >>>
        >>> # Check health
        >>> manager.update_health("factor_api", healthy=True)
        >>> manager.update_health("llm_api", healthy=False)
        >>>
        >>> # Get current tier
        >>> tier = manager.get_current_tier()
        >>> if tier == DegradationTier.TIER_2_CORE:
        ...     print("Core functionality only")
    """

    def __init__(self):
        """Initialize degradation manager."""
        self._services: Dict[str, ServiceHealth] = {}
        self._critical_services: Set[str] = set()
        self._current_tier = DegradationTier.TIER_1_FULL
        self._lock = Lock()
        self._tier_change_callbacks: List[Callable[[DegradationTier, DegradationTier], None]] = []

        logger.info("Degradation manager initialized")

    def register_service(
        self,
        name: str,
        critical: bool = False
    ) -> None:
        """Register a service for health monitoring.

        Args:
            name: Service name
            critical: Whether service is critical for core functionality
        """
        with self._lock:
            self._services[name] = ServiceHealth(name=name)
            if critical:
                self._critical_services.add(name)

        logger.info(
            f"Registered service '{name}' "
            f"(critical={critical})"
        )

    def update_health(
        self,
        service_name: str,
        healthy: bool,
        error: Optional[str] = None,
        response_time_ms: Optional[float] = None,
    ) -> None:
        """Update service health status.

        Args:
            service_name: Service name
            healthy: Whether service is healthy
            error: Error message if unhealthy
            response_time_ms: Response time in milliseconds
        """
        with self._lock:
            if service_name not in self._services:
                logger.warning(
                    f"Service '{service_name}' not registered, registering now"
                )
                self._services[service_name] = ServiceHealth(name=service_name)

            service = self._services[service_name]

            if healthy:
                service.mark_success(response_time_ms)
            else:
                service.mark_failure(error or "Unknown error")

        # Recalculate tier
        self._recalculate_tier()

    def get_current_tier(self) -> DegradationTier:
        """Get current degradation tier.

        Returns:
            Current degradation tier
        """
        with self._lock:
            return self._current_tier

    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status for specific service.

        Args:
            service_name: Service name

        Returns:
            ServiceHealth or None if not registered
        """
        with self._lock:
            return self._services.get(service_name)

    def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status for all services.

        Returns:
            Dictionary of service health
        """
        with self._lock:
            return dict(self._services)

    def register_tier_change_callback(
        self,
        callback: Callable[[DegradationTier, DegradationTier], None]
    ) -> None:
        """Register callback for tier changes.

        Args:
            callback: Function to call on tier change (old_tier, new_tier)
        """
        self._tier_change_callbacks.append(callback)

    def _recalculate_tier(self) -> None:
        """Recalculate current degradation tier based on service health."""
        with self._lock:
            old_tier = self._current_tier

            # Check critical services
            critical_services_down = sum(
                1 for name in self._critical_services
                if self._services.get(name, ServiceHealth("")).is_down()
            )

            critical_services_degraded = sum(
                1 for name in self._critical_services
                if self._services.get(name, ServiceHealth("")).status == ServiceStatus.DEGRADED
            )

            # Determine tier based on service health
            if critical_services_down > 0:
                # Critical service down -> Tier 3 or 4
                if critical_services_down >= len(self._critical_services):
                    new_tier = DegradationTier.TIER_4_MAINTENANCE
                else:
                    new_tier = DegradationTier.TIER_3_READONLY
            elif critical_services_degraded > 0:
                # Critical service degraded -> Tier 2
                new_tier = DegradationTier.TIER_2_CORE
            else:
                # All services healthy -> Tier 1
                new_tier = DegradationTier.TIER_1_FULL

            # Update tier if changed
            if new_tier != old_tier:
                self._current_tier = new_tier
                logger.warning(
                    f"Degradation tier changed: {old_tier} -> {new_tier}"
                )

                # Call callbacks
                for callback in self._tier_change_callbacks:
                    try:
                        callback(old_tier, new_tier)
                    except Exception as e:
                        logger.error(
                            f"Error in tier change callback: {e}"
                        )

    def get_stats(self) -> Dict[str, Any]:
        """Get degradation manager statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            healthy_services = sum(
                1 for s in self._services.values()
                if s.is_healthy()
            )

            down_services = sum(
                1 for s in self._services.values()
                if s.is_down()
            )

            return {
                "current_tier": self._current_tier.name,
                "total_services": len(self._services),
                "critical_services": len(self._critical_services),
                "healthy_services": healthy_services,
                "down_services": down_services,
                "services": {
                    name: {
                        "status": health.status.value,
                        "failure_count": health.failure_count,
                        "last_check": (
                            health.last_check.isoformat()
                            if health.last_check else None
                        ),
                        "response_time_ms": health.response_time_ms,
                    }
                    for name, health in self._services.items()
                },
            }


# ==============================================================================
# Global Degradation Manager
# ==============================================================================


_global_degradation_manager: Optional[DegradationManager] = None
_manager_lock = Lock()


def get_degradation_manager() -> DegradationManager:
    """Get global degradation manager instance.

    Returns:
        DegradationManager instance
    """
    global _global_degradation_manager

    if _global_degradation_manager is None:
        with _manager_lock:
            if _global_degradation_manager is None:
                _global_degradation_manager = DegradationManager()

                # Register GL-VCCI critical services
                _global_degradation_manager.register_service(
                    "factor_api", critical=True
                )
                _global_degradation_manager.register_service(
                    "database", critical=True
                )
                _global_degradation_manager.register_service(
                    "erp_api", critical=False
                )
                _global_degradation_manager.register_service(
                    "llm_api", critical=False
                )

    return _global_degradation_manager


# ==============================================================================
# Degradation Handler Decorator
# ==============================================================================


def degradation_handler(
    min_tier: DegradationTier = DegradationTier.TIER_1_FULL,
    fallback_value: Any = None,
    raise_on_degraded: bool = False,
) -> Callable:
    """Decorator to handle degraded functionality.

    Args:
        min_tier: Minimum tier required for operation
        fallback_value: Value to return if tier insufficient
        raise_on_degraded: Raise exception if degraded

    Returns:
        Decorated function

    Example:
        >>> @degradation_handler(
        ...     min_tier=DegradationTier.TIER_2_CORE,
        ...     fallback_value=[]
        ... )
        ... def get_suppliers():
        ...     return external_api.fetch_suppliers()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_degradation_manager()
            current_tier = manager.get_current_tier()

            # Check if operation allowed at current tier
            if current_tier.value > min_tier.value:
                logger.warning(
                    f"Operation '{func.__name__}' not available at {current_tier}. "
                    f"Requires minimum {min_tier}"
                )

                if raise_on_degraded:
                    raise RuntimeError(
                        f"Operation '{func.__name__}' not available in degraded mode. "
                        f"Current tier: {current_tier}, Required: {min_tier}"
                    )

                return fallback_value

            # Execute function
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ==============================================================================
# Tier-Specific Functionality
# ==============================================================================


class TierFeatures:
    """Feature flags for each degradation tier."""

    @staticmethod
    def get_features(tier: DegradationTier) -> Dict[str, bool]:
        """Get enabled features for tier.

        Args:
            tier: Degradation tier

        Returns:
            Dictionary of feature flags
        """
        features = {
            DegradationTier.TIER_1_FULL: {
                "llm_inference": True,
                "external_apis": True,
                "complex_calculations": True,
                "write_operations": True,
                "report_generation": True,
                "file_uploads": True,
                "real_time_updates": True,
            },
            DegradationTier.TIER_2_CORE: {
                "llm_inference": False,
                "external_apis": True,
                "complex_calculations": True,
                "write_operations": True,
                "report_generation": True,
                "file_uploads": False,
                "real_time_updates": False,
            },
            DegradationTier.TIER_3_READONLY: {
                "llm_inference": False,
                "external_apis": False,
                "complex_calculations": False,
                "write_operations": False,
                "report_generation": False,
                "file_uploads": False,
                "real_time_updates": False,
            },
            DegradationTier.TIER_4_MAINTENANCE: {
                "llm_inference": False,
                "external_apis": False,
                "complex_calculations": False,
                "write_operations": False,
                "report_generation": False,
                "file_uploads": False,
                "real_time_updates": False,
            },
        }

        return features.get(tier, features[DegradationTier.TIER_4_MAINTENANCE])


__all__ = [
    "DegradationTier",
    "ServiceStatus",
    "ServiceHealth",
    "DegradationManager",
    "get_degradation_manager",
    "degradation_handler",
    "TierFeatures",
]
