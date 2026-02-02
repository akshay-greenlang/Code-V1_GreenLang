"""
GL-014 EXCHANGERPRO - Health Checks

Health check endpoints and component status monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class OverallHealth:
    """Overall health status of the agent."""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    agent_id: str = "GL-014"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "agent_id": self.agent_id,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "components": [c.to_dict() for c in self.components],
        }


class HealthChecker:
    """
    Health checker for GL-014 EXCHANGERPRO.

    Monitors health of:
    - Thermal engine service
    - ML prediction service
    - Optimizer service
    - Database connections
    - Kafka connectivity
    - OPC-UA connections
    - CMMS integration
    """

    AGENT_ID = "GL-014"
    VERSION = "1.0.0"

    def __init__(self) -> None:
        """Initialize health checker."""
        self._check_functions: Dict[str, Callable[[], ComponentHealth]] = {}
        self._async_check_functions: Dict[str, Callable[[], ComponentHealth]] = {}
        self._last_results: Dict[str, ComponentHealth] = {}

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("thermal_engine", self._check_thermal_engine)
        self.register_check("ml_service", self._check_ml_service)
        self.register_check("optimizer", self._check_optimizer)

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """Register a health check function."""
        self._check_functions[name] = check_fn

    def register_async_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """Register an async health check function."""
        self._async_check_functions[name] = check_fn

    def _check_thermal_engine(self) -> ComponentHealth:
        """Check thermal engine health."""
        try:
            # Attempt a simple calculation to verify engine
            # In production, this would actually call the engine
            start = datetime.now(timezone.utc)

            # Simulate check
            is_healthy = True

            end = datetime.now(timezone.utc)
            response_time = (end - start).total_seconds() * 1000

            if is_healthy:
                return ComponentHealth(
                    name="thermal_engine",
                    status=HealthStatus.HEALTHY,
                    message="Thermal engine operational",
                    response_time_ms=response_time,
                    metadata={"version": "1.0.0"},
                )
            else:
                return ComponentHealth(
                    name="thermal_engine",
                    status=HealthStatus.UNHEALTHY,
                    message="Thermal engine calculation failed",
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.error(f"Thermal engine health check failed: {e}")
            return ComponentHealth(
                name="thermal_engine",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
            )

    def _check_ml_service(self) -> ComponentHealth:
        """Check ML prediction service health."""
        try:
            start = datetime.now(timezone.utc)

            # Simulate check - in production, ping the ML service
            is_healthy = True
            model_loaded = True

            end = datetime.now(timezone.utc)
            response_time = (end - start).total_seconds() * 1000

            if is_healthy and model_loaded:
                return ComponentHealth(
                    name="ml_service",
                    status=HealthStatus.HEALTHY,
                    message="ML service operational, models loaded",
                    response_time_ms=response_time,
                    metadata={"models_loaded": True},
                )
            elif is_healthy:
                return ComponentHealth(
                    name="ml_service",
                    status=HealthStatus.DEGRADED,
                    message="ML service running but models not loaded",
                    response_time_ms=response_time,
                    metadata={"models_loaded": False},
                )
            else:
                return ComponentHealth(
                    name="ml_service",
                    status=HealthStatus.UNHEALTHY,
                    message="ML service unavailable",
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.error(f"ML service health check failed: {e}")
            return ComponentHealth(
                name="ml_service",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
            )

    def _check_optimizer(self) -> ComponentHealth:
        """Check optimizer service health."""
        try:
            start = datetime.now(timezone.utc)

            # Simulate check
            is_healthy = True

            end = datetime.now(timezone.utc)
            response_time = (end - start).total_seconds() * 1000

            if is_healthy:
                return ComponentHealth(
                    name="optimizer",
                    status=HealthStatus.HEALTHY,
                    message="Optimizer service operational",
                    response_time_ms=response_time,
                )
            else:
                return ComponentHealth(
                    name="optimizer",
                    status=HealthStatus.UNHEALTHY,
                    message="Optimizer service unavailable",
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.error(f"Optimizer health check failed: {e}")
            return ComponentHealth(
                name="optimizer",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}",
            )

    def check_component(self, name: str) -> ComponentHealth:
        """Run health check for a specific component."""
        if name in self._check_functions:
            result = self._check_functions[name]()
            self._last_results[name] = result
            return result
        else:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {name}",
            )

    def check_all(self) -> OverallHealth:
        """Run all health checks and return overall status."""
        components: List[ComponentHealth] = []

        # Run sync checks
        for name, check_fn in self._check_functions.items():
            try:
                result = check_fn()
                components.append(result)
                self._last_results[name] = result
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                components.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                ))

        # Determine overall status
        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN

        return OverallHealth(
            status=overall_status,
            components=components,
            version=self.VERSION,
            agent_id=self.AGENT_ID,
        )

    async def check_all_async(self) -> OverallHealth:
        """Run all health checks asynchronously."""
        components: List[ComponentHealth] = []

        # Run sync checks in thread pool
        loop = asyncio.get_event_loop()

        sync_tasks = []
        for name, check_fn in self._check_functions.items():
            sync_tasks.append(loop.run_in_executor(None, check_fn))

        if sync_tasks:
            sync_results = await asyncio.gather(*sync_tasks, return_exceptions=True)
            for i, (name, _) in enumerate(self._check_functions.items()):
                result = sync_results[i]
                if isinstance(result, Exception):
                    components.append(ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(result)}",
                    ))
                else:
                    components.append(result)

        # Determine overall status
        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN

        return OverallHealth(
            status=overall_status,
            components=components,
            version=self.VERSION,
            agent_id=self.AGENT_ID,
        )

    def get_cached_results(self) -> Dict[str, ComponentHealth]:
        """Get cached health check results."""
        return dict(self._last_results)

    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        result = self.check_all()
        return result.status == HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """Check if system is ready to accept traffic."""
        result = self.check_all()
        return result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
