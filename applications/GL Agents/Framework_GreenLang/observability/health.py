"""
GreenLang Framework - Health Check Module
==========================================

Provides Kubernetes-compatible health check endpoints for GreenLang agents.
Implements liveness, readiness, and startup probes with dependency health
checks and aggregation.

Features:
- Liveness probe: Is the agent alive and not deadlocked?
- Readiness probe: Is the agent ready to receive traffic?
- Startup probe: Has the agent completed initialization?
- Dependency health checks (database, cache, external services)
- Health aggregation with configurable thresholds
- Prometheus metrics integration
- JSON and simple text response formats

Example:
    >>> from greenlang_observability.health import HealthCheckManager
    >>>
    >>> health = HealthCheckManager(service_name="gl-006-heatreclaim")
    >>>
    >>> # Add dependency checks
    >>> health.add_dependency("database", check_database_connection)
    >>> health.add_dependency("redis", check_redis_connection)
    >>>
    >>> # Check health
    >>> result = health.check_readiness()
    >>> print(result.status)  # HealthStatus.HEALTHY or UNHEALTHY

Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

    def to_http_status(self) -> int:
        """Convert to HTTP status code."""
        if self == HealthStatus.HEALTHY:
            return 200
        elif self == HealthStatus.DEGRADED:
            return 200  # Still operational
        else:
            return 503


class CheckType(Enum):
    """Types of health checks."""

    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    DEPENDENCY = "dependency"


@dataclass
class HealthCheckResult:
    """
    Result of a health check.

    Attributes:
        status: Overall health status
        check_type: Type of check performed
        name: Name of the check
        message: Human-readable status message
        duration_ms: Time taken to perform check
        timestamp: When the check was performed
        details: Additional check-specific details
        dependencies: Results of dependency checks
    """

    status: HealthStatus
    check_type: CheckType
    name: str = ""
    message: str = ""
    duration_ms: float = 0.0
    timestamp: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[HealthCheckResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "status": self.status.value,
            "check_type": self.check_type.value,
            "name": self.name,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 3),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

        if self.details:
            result["details"] = self.details

        if self.dependencies:
            result["dependencies"] = [d.to_dict() for d in self.dependencies]

        return result

    def to_simple_response(self) -> str:
        """Convert to simple text response for probes."""
        return self.status.value


class HealthCheck(ABC):
    """Base class for health checks."""

    def __init__(
        self,
        name: str,
        timeout_seconds: float = 5.0,
        critical: bool = True,
    ):
        """
        Initialize health check.

        Args:
            name: Check name
            timeout_seconds: Timeout for check execution
            critical: If True, failure makes overall status UNHEALTHY;
                      if False, failure makes status DEGRADED
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.critical = critical

    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass

    def run(self) -> HealthCheckResult:
        """Run the check with timing and error handling."""
        start_time = time.perf_counter()

        try:
            result = self.check()
            result.duration_ms = (time.perf_counter() - start_time) * 1000
            result.name = self.name
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Health check '{self.name}' failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                check_type=CheckType.DEPENDENCY,
                name=self.name,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
            )


class LivenessCheck(HealthCheck):
    """
    Liveness check - verifies the agent is alive and not deadlocked.

    If this check fails, Kubernetes will restart the container.
    Should be fast and lightweight.
    """

    def __init__(
        self,
        name: str = "liveness",
        custom_check: Optional[Callable[[], bool]] = None,
    ):
        """Initialize liveness check."""
        super().__init__(name, timeout_seconds=1.0, critical=True)
        self.custom_check = custom_check
        self._last_heartbeat: Optional[datetime] = None
        self._heartbeat_threshold = timedelta(seconds=30)

    def record_heartbeat(self) -> None:
        """Record a heartbeat to indicate the agent is alive."""
        self._last_heartbeat = datetime.now(timezone.utc)

    def check(self) -> HealthCheckResult:
        """Check if agent is alive."""
        details: Dict[str, Any] = {}

        # Check heartbeat if configured
        if self._last_heartbeat:
            age = datetime.now(timezone.utc) - self._last_heartbeat
            details["last_heartbeat_age_seconds"] = age.total_seconds()

            if age > self._heartbeat_threshold:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    check_type=CheckType.LIVENESS,
                    message=f"No heartbeat for {age.total_seconds():.1f}s",
                    details=details,
                )

        # Run custom check if provided
        if self.custom_check:
            try:
                if not self.custom_check():
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        check_type=CheckType.LIVENESS,
                        message="Custom liveness check failed",
                        details=details,
                    )
            except Exception as e:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    check_type=CheckType.LIVENESS,
                    message=f"Custom check error: {str(e)}",
                    details=details,
                )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.LIVENESS,
            message="Agent is alive",
            details=details,
        )


class ReadinessCheck(HealthCheck):
    """
    Readiness check - verifies the agent is ready to receive traffic.

    If this check fails, Kubernetes will stop routing traffic to the pod.
    Should verify all dependencies are available.
    """

    def __init__(
        self,
        name: str = "readiness",
        custom_check: Optional[Callable[[], bool]] = None,
    ):
        """Initialize readiness check."""
        super().__init__(name, timeout_seconds=5.0, critical=True)
        self.custom_check = custom_check
        self._is_ready = False

    def set_ready(self, ready: bool = True) -> None:
        """Set the ready state."""
        self._is_ready = ready

    def check(self) -> HealthCheckResult:
        """Check if agent is ready."""
        if not self._is_ready:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                check_type=CheckType.READINESS,
                message="Agent is not ready",
            )

        if self.custom_check:
            try:
                if not self.custom_check():
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        check_type=CheckType.READINESS,
                        message="Custom readiness check failed",
                    )
            except Exception as e:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    check_type=CheckType.READINESS,
                    message=f"Custom check error: {str(e)}",
                )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.READINESS,
            message="Agent is ready",
        )


class StartupCheck(HealthCheck):
    """
    Startup check - verifies the agent has completed initialization.

    Used during container startup. After passing, liveness/readiness
    probes take over. Useful for slow-starting containers.
    """

    def __init__(
        self,
        name: str = "startup",
        custom_check: Optional[Callable[[], bool]] = None,
    ):
        """Initialize startup check."""
        super().__init__(name, timeout_seconds=30.0, critical=True)
        self.custom_check = custom_check
        self._started = False
        self._start_time: Optional[datetime] = None

    def mark_started(self) -> None:
        """Mark the agent as started."""
        self._started = True
        self._start_time = datetime.now(timezone.utc)

    def check(self) -> HealthCheckResult:
        """Check if agent has started."""
        details: Dict[str, Any] = {}

        if self._start_time:
            details["started_at"] = self._start_time.isoformat()

        if not self._started:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                check_type=CheckType.STARTUP,
                message="Agent has not completed startup",
                details=details,
            )

        if self.custom_check:
            try:
                if not self.custom_check():
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        check_type=CheckType.STARTUP,
                        message="Custom startup check failed",
                        details=details,
                    )
            except Exception as e:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    check_type=CheckType.STARTUP,
                    message=f"Custom check error: {str(e)}",
                    details=details,
                )

        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.STARTUP,
            message="Agent has started",
            details=details,
        )


class DependencyCheck(HealthCheck):
    """
    Dependency check - verifies an external dependency is available.

    Used to check database connections, cache availability,
    external service connectivity, etc.
    """

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], bool],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        description: str = "",
    ):
        """
        Initialize dependency check.

        Args:
            name: Dependency name
            check_fn: Function that returns True if healthy
            timeout_seconds: Check timeout
            critical: Whether failure is critical
            description: Human-readable description
        """
        super().__init__(name, timeout_seconds, critical)
        self.check_fn = check_fn
        self.description = description

    def check(self) -> HealthCheckResult:
        """Check dependency health."""
        try:
            is_healthy = self.check_fn()

            if is_healthy:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    check_type=CheckType.DEPENDENCY,
                    message=f"{self.name} is available",
                    details={"description": self.description} if self.description else {},
                )
            else:
                status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
                return HealthCheckResult(
                    status=status,
                    check_type=CheckType.DEPENDENCY,
                    message=f"{self.name} check returned false",
                    details={"critical": self.critical},
                )
        except Exception as e:
            status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
            return HealthCheckResult(
                status=status,
                check_type=CheckType.DEPENDENCY,
                message=f"{self.name} check failed: {str(e)}",
                details={"critical": self.critical, "error": str(e)},
            )


class AsyncDependencyCheck(HealthCheck):
    """Async version of dependency check for async dependencies."""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[bool]],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        description: str = "",
    ):
        """Initialize async dependency check."""
        super().__init__(name, timeout_seconds, critical)
        self.check_fn = check_fn
        self.description = description

    def check(self) -> HealthCheckResult:
        """Run async check synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            is_healthy = loop.run_until_complete(
                asyncio.wait_for(self.check_fn(), timeout=self.timeout_seconds)
            )

            if is_healthy:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    check_type=CheckType.DEPENDENCY,
                    message=f"{self.name} is available",
                )
            else:
                status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
                return HealthCheckResult(
                    status=status,
                    check_type=CheckType.DEPENDENCY,
                    message=f"{self.name} check returned false",
                )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                check_type=CheckType.DEPENDENCY,
                message=f"{self.name} check timed out",
            )
        except Exception as e:
            status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
            return HealthCheckResult(
                status=status,
                check_type=CheckType.DEPENDENCY,
                message=f"{self.name} check failed: {str(e)}",
            )

    async def check_async(self) -> HealthCheckResult:
        """Perform async check."""
        start_time = time.perf_counter()

        try:
            is_healthy = await asyncio.wait_for(
                self.check_fn(), timeout=self.timeout_seconds
            )
            duration_ms = (time.perf_counter() - start_time) * 1000

            if is_healthy:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    check_type=CheckType.DEPENDENCY,
                    name=self.name,
                    message=f"{self.name} is available",
                    duration_ms=duration_ms,
                )
            else:
                status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
                return HealthCheckResult(
                    status=status,
                    check_type=CheckType.DEPENDENCY,
                    name=self.name,
                    message=f"{self.name} check returned false",
                    duration_ms=duration_ms,
                )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                check_type=CheckType.DEPENDENCY,
                name=self.name,
                message=f"{self.name} check timed out",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
            return HealthCheckResult(
                status=status,
                check_type=CheckType.DEPENDENCY,
                name=self.name,
                message=f"{self.name} check failed: {str(e)}",
                duration_ms=duration_ms,
            )


class HealthCheckManager:
    """
    Central manager for all health checks.

    Coordinates liveness, readiness, startup, and dependency checks.
    Provides aggregated health status and Kubernetes-compatible endpoints.

    Example:
        >>> health = HealthCheckManager(service_name="gl-006-heatreclaim")
        >>>
        >>> # Add dependency checks
        >>> health.add_dependency("database", lambda: db.ping())
        >>> health.add_dependency("redis", lambda: redis.ping(), critical=False)
        >>>
        >>> # Mark as started and ready
        >>> health.mark_started()
        >>> health.set_ready(True)
        >>>
        >>> # Check health
        >>> liveness = health.check_liveness()
        >>> readiness = health.check_readiness()
    """

    def __init__(
        self,
        service_name: str,
        service_version: str = "1.0.0",
    ):
        """
        Initialize health check manager.

        Args:
            service_name: Name of the service
            service_version: Version of the service
        """
        self.service_name = service_name
        self.service_version = service_version

        # Initialize built-in checks
        self._liveness = LivenessCheck()
        self._readiness = ReadinessCheck()
        self._startup = StartupCheck()

        # Dependency checks
        self._dependencies: Dict[str, HealthCheck] = {}
        self._lock = threading.Lock()

    def add_dependency(
        self,
        name: str,
        check_fn: Callable[[], bool],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        description: str = "",
    ) -> None:
        """
        Add a dependency health check.

        Args:
            name: Dependency name
            check_fn: Function that returns True if healthy
            timeout_seconds: Check timeout
            critical: If True, failure makes overall status UNHEALTHY
            description: Human-readable description
        """
        with self._lock:
            self._dependencies[name] = DependencyCheck(
                name=name,
                check_fn=check_fn,
                timeout_seconds=timeout_seconds,
                critical=critical,
                description=description,
            )

    def add_async_dependency(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[bool]],
        timeout_seconds: float = 5.0,
        critical: bool = True,
        description: str = "",
    ) -> None:
        """Add an async dependency health check."""
        with self._lock:
            self._dependencies[name] = AsyncDependencyCheck(
                name=name,
                check_fn=check_fn,
                timeout_seconds=timeout_seconds,
                critical=critical,
                description=description,
            )

    def remove_dependency(self, name: str) -> bool:
        """Remove a dependency check."""
        with self._lock:
            if name in self._dependencies:
                del self._dependencies[name]
                return True
            return False

    def set_liveness_check(
        self,
        check_fn: Callable[[], bool],
    ) -> None:
        """Set custom liveness check function."""
        self._liveness = LivenessCheck(custom_check=check_fn)

    def set_readiness_check(
        self,
        check_fn: Callable[[], bool],
    ) -> None:
        """Set custom readiness check function."""
        self._readiness = ReadinessCheck(custom_check=check_fn)

    def set_startup_check(
        self,
        check_fn: Callable[[], bool],
    ) -> None:
        """Set custom startup check function."""
        self._startup = StartupCheck(custom_check=check_fn)

    def record_heartbeat(self) -> None:
        """Record a heartbeat for liveness check."""
        self._liveness.record_heartbeat()

    def mark_started(self) -> None:
        """Mark the agent as started."""
        self._startup.mark_started()

    def set_ready(self, ready: bool = True) -> None:
        """Set the agent ready state."""
        self._readiness.set_ready(ready)

    def check_liveness(self) -> HealthCheckResult:
        """
        Perform liveness check.

        Returns:
            Health check result
        """
        return self._liveness.run()

    def check_readiness(self, include_dependencies: bool = True) -> HealthCheckResult:
        """
        Perform readiness check including dependencies.

        Args:
            include_dependencies: Whether to check dependencies

        Returns:
            Aggregated health check result
        """
        start_time = time.perf_counter()

        # Check base readiness
        base_result = self._readiness.run()
        if base_result.status == HealthStatus.UNHEALTHY:
            return base_result

        # Check dependencies if requested
        dependency_results: List[HealthCheckResult] = []
        if include_dependencies:
            with self._lock:
                for dep in self._dependencies.values():
                    dependency_results.append(dep.run())

        # Aggregate results
        duration_ms = (time.perf_counter() - start_time) * 1000
        return self._aggregate_results(
            check_type=CheckType.READINESS,
            base_result=base_result,
            dependency_results=dependency_results,
            duration_ms=duration_ms,
        )

    def check_startup(self) -> HealthCheckResult:
        """
        Perform startup check.

        Returns:
            Health check result
        """
        return self._startup.run()

    def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Perform all health checks.

        Returns:
            Dictionary of check name to result
        """
        return {
            "liveness": self.check_liveness(),
            "readiness": self.check_readiness(),
            "startup": self.check_startup(),
        }

    def check_dependencies(self) -> List[HealthCheckResult]:
        """
        Check all dependencies.

        Returns:
            List of dependency check results
        """
        results: List[HealthCheckResult] = []
        with self._lock:
            for dep in self._dependencies.values():
                results.append(dep.run())
        return results

    def _aggregate_results(
        self,
        check_type: CheckType,
        base_result: HealthCheckResult,
        dependency_results: List[HealthCheckResult],
        duration_ms: float,
    ) -> HealthCheckResult:
        """Aggregate multiple check results into one."""
        # Determine overall status
        has_critical_failure = any(
            r.status == HealthStatus.UNHEALTHY
            for r in dependency_results
        )
        has_degraded = any(
            r.status == HealthStatus.DEGRADED
            for r in dependency_results
        )

        if base_result.status == HealthStatus.UNHEALTHY or has_critical_failure:
            overall_status = HealthStatus.UNHEALTHY
            message = "One or more critical checks failed"
        elif has_degraded:
            overall_status = HealthStatus.DEGRADED
            message = "One or more non-critical checks failed"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All checks passed"

        return HealthCheckResult(
            status=overall_status,
            check_type=check_type,
            name=self.service_name,
            message=message,
            duration_ms=duration_ms,
            details={
                "service_name": self.service_name,
                "service_version": self.service_version,
                "checks_passed": sum(
                    1 for r in dependency_results if r.status == HealthStatus.HEALTHY
                ),
                "checks_failed": sum(
                    1 for r in dependency_results if r.status != HealthStatus.HEALTHY
                ),
                "total_checks": len(dependency_results),
            },
            dependencies=dependency_results,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get health metrics for Prometheus.

        Returns:
            Dictionary of health metrics
        """
        liveness = self.check_liveness()
        readiness = self.check_readiness()
        startup = self.check_startup()
        dependencies = self.check_dependencies()

        return {
            "health_check_liveness": 1 if liveness.status == HealthStatus.HEALTHY else 0,
            "health_check_readiness": 1 if readiness.status == HealthStatus.HEALTHY else 0,
            "health_check_startup": 1 if startup.status == HealthStatus.HEALTHY else 0,
            "health_check_dependencies_total": len(dependencies),
            "health_check_dependencies_healthy": sum(
                1 for d in dependencies if d.status == HealthStatus.HEALTHY
            ),
            "health_check_dependencies_unhealthy": sum(
                1 for d in dependencies if d.status == HealthStatus.UNHEALTHY
            ),
        }


# Convenience functions for common dependency checks


def create_http_check(
    url: str,
    timeout_seconds: float = 5.0,
    expected_status: int = 200,
) -> Callable[[], bool]:
    """
    Create an HTTP health check function.

    Args:
        url: URL to check
        timeout_seconds: Request timeout
        expected_status: Expected HTTP status code

    Returns:
        Check function that returns True if healthy
    """
    def check() -> bool:
        try:
            import urllib.request

            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                return response.status == expected_status
        except Exception:
            return False

    return check


def create_tcp_check(
    host: str,
    port: int,
    timeout_seconds: float = 5.0,
) -> Callable[[], bool]:
    """
    Create a TCP connectivity check function.

    Args:
        host: Host to connect to
        port: Port to connect to
        timeout_seconds: Connection timeout

    Returns:
        Check function that returns True if connectable
    """
    def check() -> bool:
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_seconds)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    return check


def create_file_check(
    path: str,
    max_age_seconds: Optional[float] = None,
) -> Callable[[], bool]:
    """
    Create a file existence/freshness check function.

    Args:
        path: Path to file
        max_age_seconds: Maximum age of file (None for no age check)

    Returns:
        Check function that returns True if file exists and is fresh
    """
    def check() -> bool:
        try:
            import os

            if not os.path.exists(path):
                return False

            if max_age_seconds is not None:
                mtime = os.path.getmtime(path)
                age = time.time() - mtime
                return age <= max_age_seconds

            return True
        except Exception:
            return False

    return check
