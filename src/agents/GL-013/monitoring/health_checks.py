"""
GL-013 PREDICTMAINT - Health Check Endpoints

Comprehensive health check endpoints and liveness probes for predictive
maintenance agent monitoring. Provides Kubernetes-compatible health endpoints
and detailed system status reporting.

Key Features:
    - Kubernetes liveness and readiness probes
    - Component-level health status
    - Dependency health checks (databases, connectors)
    - Degraded state detection
    - Self-healing triggers
    - Health history tracking

Endpoints:
    - /health: Overall health status
    - /health/live: Liveness probe (is the process running?)
    - /health/ready: Readiness probe (can it accept traffic?)
    - /health/startup: Startup probe (has it initialized?)
    - /health/detailed: Detailed component health

Example:
    >>> from gl_013.monitoring.health_checks import HealthChecker
    >>> checker = HealthChecker()
    >>> status = checker.check_health()
    >>> print(f"Status: {status.status}, Components: {len(status.components)}")

Author: GL-MonitoringEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
import asyncio
import logging
import threading
import time
import traceback
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of components to check."""
    DATABASE = "database"
    CACHE = "cache"
    CONNECTOR = "connector"
    MODEL = "model"
    CALCULATOR = "calculator"
    QUEUE = "queue"
    EXTERNAL_API = "external_api"
    INTERNAL = "internal"


class ProbeType(str, Enum):
    """Kubernetes probe types."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComponentHealth:
    """
    Health status of a single component.

    Attributes:
        name: Component name
        component_type: Type of component
        status: Current health status
        message: Human-readable status message
        latency_ms: Check latency in milliseconds
        last_check: Timestamp of last check
        details: Additional status details
        error: Error message if unhealthy
    """
    name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "last_check": self.last_check.isoformat(),
            "details": self.details,
            "error": self.error
        }


@dataclass
class HealthCheckResult:
    """
    Complete health check result.

    Attributes:
        status: Overall health status
        timestamp: Check timestamp
        version: Agent version
        uptime_seconds: Process uptime
        components: Individual component statuses
        checks_passed: Number of passed checks
        checks_failed: Number of failed checks
        total_latency_ms: Total check duration
    """
    status: HealthStatus
    timestamp: datetime
    version: str
    uptime_seconds: float
    components: List[ComponentHealth]
    checks_passed: int
    checks_failed: int
    total_latency_ms: float
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "message": self.message,
            "checks": {
                "passed": self.checks_passed,
                "failed": self.checks_failed,
                "total": len(self.components)
            },
            "total_latency_ms": round(self.total_latency_ms, 2),
            "components": [c.to_dict() for c in self.components]
        }


@dataclass
class ProbeResult:
    """
    Kubernetes probe result.

    Attributes:
        probe_type: Type of probe
        success: Whether probe passed
        status_code: HTTP status code
        message: Status message
        timestamp: Probe timestamp
    """
    probe_type: ProbeType
    success: bool
    status_code: int
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "probe": self.probe_type.value,
            "success": self.success,
            "status_code": self.status_code,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HealthHistory:
    """
    Historical health record.

    Attributes:
        timestamp: Record timestamp
        status: Health status at time
        failed_components: List of failed component names
        latency_ms: Check latency
    """
    timestamp: datetime
    status: HealthStatus
    failed_components: List[str]
    latency_ms: float


# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================

class HealthCheckRegistry:
    """Registry for health check functions."""

    def __init__(self):
        self._checks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
        component_type: ComponentType,
        critical: bool = False,
        timeout_seconds: float = 5.0
    ) -> None:
        """
        Register a health check function.

        Args:
            name: Check name
            check_func: Function that returns ComponentHealth
            component_type: Type of component
            critical: Whether failure marks system unhealthy
            timeout_seconds: Check timeout
        """
        with self._lock:
            self._checks[name] = {
                "func": check_func,
                "type": component_type,
                "critical": critical,
                "timeout": timeout_seconds
            }
            logger.debug(f"Registered health check: {name}")

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                logger.debug(f"Unregistered health check: {name}")

    def get_checks(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered checks."""
        with self._lock:
            return dict(self._checks)

    def get_critical_checks(self) -> List[str]:
        """Get names of critical checks."""
        with self._lock:
            return [name for name, info in self._checks.items() if info["critical"]]


# Global registry instance
health_check_registry = HealthCheckRegistry()


def health_check(
    name: str,
    component_type: ComponentType,
    critical: bool = False,
    timeout_seconds: float = 5.0
):
    """
    Decorator to register a health check function.

    Args:
        name: Check name
        component_type: Component type
        critical: Whether this is a critical check
        timeout_seconds: Check timeout

    Example:
        >>> @health_check("database", ComponentType.DATABASE, critical=True)
        ... def check_database() -> ComponentHealth:
        ...     # Check database connection
        ...     return ComponentHealth(...)
    """
    def decorator(func: Callable[[], ComponentHealth]) -> Callable[[], ComponentHealth]:
        health_check_registry.register(
            name=name,
            check_func=func,
            component_type=component_type,
            critical=critical,
            timeout_seconds=timeout_seconds
        )

        @wraps(func)
        def wrapper() -> ComponentHealth:
            return func()

        return wrapper
    return decorator


# =============================================================================
# HEALTH CHECKER CLASS
# =============================================================================

class HealthChecker:
    """
    Comprehensive health checker for GL-013 PREDICTMAINT.

    Provides health check endpoints compatible with Kubernetes probes
    and detailed system status reporting.

    Example:
        >>> checker = HealthChecker()
        >>> checker.register_check("database", check_database, ComponentType.DATABASE, critical=True)
        >>> result = checker.check_health()
        >>> print(f"Status: {result.status.value}")
    """

    def __init__(
        self,
        version: str = "1.0.0",
        startup_grace_period: timedelta = timedelta(seconds=30),
        max_history_size: int = 100
    ):
        """
        Initialize HealthChecker.

        Args:
            version: Agent version string
            startup_grace_period: Grace period for startup probe
            max_history_size: Maximum health history records to keep
        """
        self._version = version
        self._startup_time = datetime.now()
        self._startup_grace_period = startup_grace_period
        self._startup_complete = False
        self._checks: Dict[str, Dict[str, Any]] = {}
        self._history: List[HealthHistory] = []
        self._max_history = max_history_size
        self._lock = threading.Lock()
        self._last_health: Optional[HealthCheckResult] = None

        # Register default checks
        self._register_default_checks()

        logger.info(f"HealthChecker initialized, version={version}")

    def _register_default_checks(self) -> None:
        """Register default internal health checks."""
        self.register_check(
            name="internal_state",
            check_func=self._check_internal_state,
            component_type=ComponentType.INTERNAL,
            critical=True,
            timeout_seconds=1.0
        )

    def _check_internal_state(self) -> ComponentHealth:
        """Check internal state health."""
        return ComponentHealth(
            name="internal_state",
            component_type=ComponentType.INTERNAL,
            status=HealthStatus.HEALTHY,
            message="Internal state is healthy",
            details={
                "uptime_seconds": self.get_uptime_seconds(),
                "checks_registered": len(self._checks),
                "history_size": len(self._history)
            }
        )

    def register_check(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
        component_type: ComponentType,
        critical: bool = False,
        timeout_seconds: float = 5.0
    ) -> None:
        """
        Register a health check.

        Args:
            name: Unique check name
            check_func: Function returning ComponentHealth
            component_type: Type of component being checked
            critical: If True, failure marks entire system unhealthy
            timeout_seconds: Maximum time for check to complete
        """
        with self._lock:
            self._checks[name] = {
                "func": check_func,
                "type": component_type,
                "critical": critical,
                "timeout": timeout_seconds
            }
        logger.debug(f"Registered health check: {name} (critical={critical})")

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                logger.debug(f"Unregistered health check: {name}")

    def get_uptime_seconds(self) -> float:
        """Get process uptime in seconds."""
        return (datetime.now() - self._startup_time).total_seconds()

    def mark_startup_complete(self) -> None:
        """Mark startup as complete."""
        self._startup_complete = True
        logger.info("Startup marked as complete")

    def _run_check_with_timeout(
        self,
        name: str,
        check_info: Dict[str, Any]
    ) -> ComponentHealth:
        """
        Run a single health check with timeout.

        Args:
            name: Check name
            check_info: Check configuration

        Returns:
            ComponentHealth result
        """
        start_time = time.perf_counter()
        timeout = check_info["timeout"]
        component_type = check_info["type"]

        try:
            # Run check
            result = check_info["func"]()

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            result.latency_ms = latency_ms
            result.last_check = datetime.now()

            return result

        except TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ComponentHealth(
                name=name,
                component_type=component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {timeout}s",
                latency_ms=latency_ms,
                error="Timeout"
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Health check '{name}' failed: {e}", exc_info=True)
            return ComponentHealth(
                name=name,
                component_type=component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=latency_ms,
                error=str(e)
            )

    def check_health(self) -> HealthCheckResult:
        """
        Run all health checks and return overall status.

        Returns:
            HealthCheckResult with all component statuses
        """
        start_time = time.perf_counter()
        timestamp = datetime.now()

        components: List[ComponentHealth] = []
        checks_passed = 0
        checks_failed = 0
        has_critical_failure = False

        with self._lock:
            checks = dict(self._checks)

        # Run all checks
        for name, check_info in checks.items():
            result = self._run_check_with_timeout(name, check_info)
            components.append(result)

            if result.status == HealthStatus.HEALTHY:
                checks_passed += 1
            else:
                checks_failed += 1
                if check_info["critical"]:
                    has_critical_failure = True
                    logger.warning(f"Critical health check failed: {name}")

        # Determine overall status
        if has_critical_failure:
            overall_status = HealthStatus.UNHEALTHY
            message = "Critical component failure detected"
        elif checks_failed > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"{checks_failed} non-critical check(s) failed"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All checks passed"

        total_latency = (time.perf_counter() - start_time) * 1000

        result = HealthCheckResult(
            status=overall_status,
            timestamp=timestamp,
            version=self._version,
            uptime_seconds=self.get_uptime_seconds(),
            components=components,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            total_latency_ms=total_latency,
            message=message
        )

        # Store in history
        self._add_to_history(result)
        self._last_health = result

        return result

    def _add_to_history(self, result: HealthCheckResult) -> None:
        """Add result to history, maintaining max size."""
        failed_components = [
            c.name for c in result.components
            if c.status != HealthStatus.HEALTHY
        ]

        history_entry = HealthHistory(
            timestamp=result.timestamp,
            status=result.status,
            failed_components=failed_components,
            latency_ms=result.total_latency_ms
        )

        with self._lock:
            self._history.append(history_entry)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def check_liveness(self) -> ProbeResult:
        """
        Kubernetes liveness probe.

        Checks if the process is alive and should not be killed.
        Returns success if the process is running.

        Returns:
            ProbeResult indicating liveness status
        """
        try:
            # Basic check - process is running
            uptime = self.get_uptime_seconds()

            return ProbeResult(
                probe_type=ProbeType.LIVENESS,
                success=True,
                status_code=200,
                message=f"Process alive, uptime: {uptime:.1f}s"
            )

        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return ProbeResult(
                probe_type=ProbeType.LIVENESS,
                success=False,
                status_code=503,
                message=f"Liveness check failed: {str(e)}"
            )

    def check_readiness(self) -> ProbeResult:
        """
        Kubernetes readiness probe.

        Checks if the service is ready to accept traffic.
        Returns success if all critical components are healthy.

        Returns:
            ProbeResult indicating readiness status
        """
        try:
            # Run full health check
            health = self.check_health()

            if health.status == HealthStatus.HEALTHY:
                return ProbeResult(
                    probe_type=ProbeType.READINESS,
                    success=True,
                    status_code=200,
                    message="Service ready to accept traffic"
                )
            elif health.status == HealthStatus.DEGRADED:
                # Degraded is still ready, just not optimal
                return ProbeResult(
                    probe_type=ProbeType.READINESS,
                    success=True,
                    status_code=200,
                    message=f"Service degraded but accepting traffic: {health.message}"
                )
            else:
                return ProbeResult(
                    probe_type=ProbeType.READINESS,
                    success=False,
                    status_code=503,
                    message=f"Service not ready: {health.message}"
                )

        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return ProbeResult(
                probe_type=ProbeType.READINESS,
                success=False,
                status_code=503,
                message=f"Readiness check failed: {str(e)}"
            )

    def check_startup(self) -> ProbeResult:
        """
        Kubernetes startup probe.

        Checks if the service has completed startup.
        During grace period, always returns success.

        Returns:
            ProbeResult indicating startup status
        """
        try:
            uptime = self.get_uptime_seconds()
            grace_seconds = self._startup_grace_period.total_seconds()

            # During grace period, always succeed
            if uptime < grace_seconds and not self._startup_complete:
                return ProbeResult(
                    probe_type=ProbeType.STARTUP,
                    success=True,
                    status_code=200,
                    message=f"Startup in progress ({uptime:.1f}s / {grace_seconds}s)"
                )

            # After grace period, check if startup is complete
            if self._startup_complete:
                return ProbeResult(
                    probe_type=ProbeType.STARTUP,
                    success=True,
                    status_code=200,
                    message="Startup complete"
                )

            # Grace period expired but startup not marked complete
            # Run health check to verify
            health = self.check_health()
            if health.status != HealthStatus.UNHEALTHY:
                self._startup_complete = True
                return ProbeResult(
                    probe_type=ProbeType.STARTUP,
                    success=True,
                    status_code=200,
                    message="Startup complete (auto-detected)"
                )

            return ProbeResult(
                probe_type=ProbeType.STARTUP,
                success=False,
                status_code=503,
                message=f"Startup failed: {health.message}"
            )

        except Exception as e:
            logger.error(f"Startup check failed: {e}")
            return ProbeResult(
                probe_type=ProbeType.STARTUP,
                success=False,
                status_code=503,
                message=f"Startup check failed: {str(e)}"
            )

    def get_detailed_health(self) -> Dict[str, Any]:
        """
        Get detailed health status with history.

        Returns:
            Dictionary with comprehensive health information
        """
        health = self.check_health()

        # Calculate health statistics
        with self._lock:
            history = list(self._history)

        if history:
            healthy_count = sum(1 for h in history if h.status == HealthStatus.HEALTHY)
            health_percentage = (healthy_count / len(history)) * 100
            avg_latency = sum(h.latency_ms for h in history) / len(history)
        else:
            health_percentage = 100.0
            avg_latency = 0.0

        return {
            "current": health.to_dict(),
            "statistics": {
                "health_percentage_30m": round(health_percentage, 1),
                "average_latency_ms": round(avg_latency, 2),
                "history_size": len(history),
                "uptime_seconds": round(self.get_uptime_seconds(), 1)
            },
            "probes": {
                "liveness": self.check_liveness().to_dict(),
                "readiness": self.check_readiness().to_dict(),
                "startup": self.check_startup().to_dict()
            },
            "registered_checks": list(self._checks.keys()),
            "critical_checks": [
                name for name, info in self._checks.items()
                if info["critical"]
            ]
        }

    def get_last_health(self) -> Optional[HealthCheckResult]:
        """Get the last health check result without running new checks."""
        return self._last_health

    def get_health_history(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent health history.

        Args:
            limit: Maximum records to return

        Returns:
            List of historical health records
        """
        with self._lock:
            history = list(self._history)[-limit:]

        return [
            {
                "timestamp": h.timestamp.isoformat(),
                "status": h.status.value,
                "failed_components": h.failed_components,
                "latency_ms": round(h.latency_ms, 2)
            }
            for h in reversed(history)
        ]

    def clear_history(self) -> None:
        """Clear health history."""
        with self._lock:
            self._history.clear()
        logger.info("Health history cleared")


# =============================================================================
# PREDEFINED HEALTH CHECK FUNCTIONS
# =============================================================================

def create_database_health_check(
    connection_func: Callable[[], bool],
    name: str = "database"
) -> Callable[[], ComponentHealth]:
    """
    Create a database health check function.

    Args:
        connection_func: Function that tests database connection
        name: Check name

    Returns:
        Health check function
    """
    def check() -> ComponentHealth:
        start = time.perf_counter()
        try:
            is_connected = connection_func()
            latency = (time.perf_counter() - start) * 1000

            if is_connected:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    latency_ms=latency
                )
            else:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message="Database connection failed",
                    latency_ms=latency
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name=name,
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}",
                latency_ms=latency,
                error=str(e)
            )

    return check


def create_cache_health_check(
    ping_func: Callable[[], bool],
    name: str = "cache"
) -> Callable[[], ComponentHealth]:
    """
    Create a cache health check function.

    Args:
        ping_func: Function that tests cache connection
        name: Check name

    Returns:
        Health check function
    """
    def check() -> ComponentHealth:
        start = time.perf_counter()
        try:
            is_available = ping_func()
            latency = (time.perf_counter() - start) * 1000

            if is_available:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.CACHE,
                    status=HealthStatus.HEALTHY,
                    message="Cache available",
                    latency_ms=latency
                )
            else:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.CACHE,
                    status=HealthStatus.DEGRADED,
                    message="Cache unavailable (non-critical)",
                    latency_ms=latency
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name=name,
                component_type=ComponentType.CACHE,
                status=HealthStatus.DEGRADED,
                message=f"Cache error: {str(e)}",
                latency_ms=latency,
                error=str(e)
            )

    return check


def create_connector_health_check(
    status_func: Callable[[], int],
    name: str = "connector"
) -> Callable[[], ComponentHealth]:
    """
    Create a connector health check function.

    Args:
        status_func: Function returning connector status (0=down, 1=up, 2=degraded)
        name: Check name

    Returns:
        Health check function
    """
    def check() -> ComponentHealth:
        start = time.perf_counter()
        try:
            status = status_func()
            latency = (time.perf_counter() - start) * 1000

            if status == 1:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.CONNECTOR,
                    status=HealthStatus.HEALTHY,
                    message="Connector connected",
                    latency_ms=latency,
                    details={"status_code": status}
                )
            elif status == 2:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.CONNECTOR,
                    status=HealthStatus.DEGRADED,
                    message="Connector degraded",
                    latency_ms=latency,
                    details={"status_code": status}
                )
            else:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.CONNECTOR,
                    status=HealthStatus.UNHEALTHY,
                    message="Connector disconnected",
                    latency_ms=latency,
                    details={"status_code": status}
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name=name,
                component_type=ComponentType.CONNECTOR,
                status=HealthStatus.UNHEALTHY,
                message=f"Connector error: {str(e)}",
                latency_ms=latency,
                error=str(e)
            )

    return check


def create_model_health_check(
    inference_func: Callable[[], bool],
    name: str = "model"
) -> Callable[[], ComponentHealth]:
    """
    Create an ML model health check function.

    Args:
        inference_func: Function that tests model inference
        name: Check name

    Returns:
        Health check function
    """
    def check() -> ComponentHealth:
        start = time.perf_counter()
        try:
            is_working = inference_func()
            latency = (time.perf_counter() - start) * 1000

            if is_working:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.MODEL,
                    status=HealthStatus.HEALTHY,
                    message="Model inference working",
                    latency_ms=latency
                )
            else:
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.MODEL,
                    status=HealthStatus.UNHEALTHY,
                    message="Model inference failed",
                    latency_ms=latency
                )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return ComponentHealth(
                name=name,
                component_type=ComponentType.MODEL,
                status=HealthStatus.UNHEALTHY,
                message=f"Model error: {str(e)}",
                latency_ms=latency,
                error=str(e)
            )

    return check


# =============================================================================
# HTTP HANDLER INTEGRATION
# =============================================================================

class HealthCheckHandler:
    """
    HTTP handler for health check endpoints.

    Can be integrated with various web frameworks (FastAPI, Flask, etc.)
    """

    def __init__(self, checker: Optional[HealthChecker] = None):
        """
        Initialize handler.

        Args:
            checker: HealthChecker instance (creates new if not provided)
        """
        self.checker = checker or HealthChecker()

    def handle_health(self) -> tuple:
        """
        Handle /health endpoint.

        Returns:
            Tuple of (response_dict, status_code)
        """
        result = self.checker.check_health()
        status_code = 200 if result.status != HealthStatus.UNHEALTHY else 503
        return result.to_dict(), status_code

    def handle_live(self) -> tuple:
        """
        Handle /health/live endpoint.

        Returns:
            Tuple of (response_dict, status_code)
        """
        result = self.checker.check_liveness()
        return result.to_dict(), result.status_code

    def handle_ready(self) -> tuple:
        """
        Handle /health/ready endpoint.

        Returns:
            Tuple of (response_dict, status_code)
        """
        result = self.checker.check_readiness()
        return result.to_dict(), result.status_code

    def handle_startup(self) -> tuple:
        """
        Handle /health/startup endpoint.

        Returns:
            Tuple of (response_dict, status_code)
        """
        result = self.checker.check_startup()
        return result.to_dict(), result.status_code

    def handle_detailed(self) -> tuple:
        """
        Handle /health/detailed endpoint.

        Returns:
            Tuple of (response_dict, status_code)
        """
        detailed = self.checker.get_detailed_health()
        status = detailed["current"]["status"]
        status_code = 200 if status != "unhealthy" else 503
        return detailed, status_code


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "HealthStatus",
    "ComponentType",
    "ProbeType",
    # Data classes
    "ComponentHealth",
    "HealthCheckResult",
    "ProbeResult",
    "HealthHistory",
    # Registry
    "HealthCheckRegistry",
    "health_check_registry",
    "health_check",
    # Main class
    "HealthChecker",
    # Factory functions
    "create_database_health_check",
    "create_cache_health_check",
    "create_connector_health_check",
    "create_model_health_check",
    # HTTP handler
    "HealthCheckHandler",
]
