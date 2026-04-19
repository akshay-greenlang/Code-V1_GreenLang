"""
GL-008 TRAPCATCHER - Health Check Module

Kubernetes-compatible health checks for liveness, readiness, and startup probes.
Implements comprehensive component health verification.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProbeType(str, Enum):
    """Kubernetes probe types."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass(frozen=True)
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    last_check: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Overall health check result."""
    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: str
    uptime_seconds: float
    version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "version": self.version,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": round(c.latency_ms, 2),
                    "last_check": c.last_check,
                    "details": c.details,
                }
                for c in self.components
            ],
        }

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Check if service is ready to accept traffic."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class HealthChecker:
    """
    Comprehensive health checker for GL-008 TRAPCATCHER.

    Performs health checks on all components and aggregates results
    for Kubernetes probes and monitoring.

    Usage:
        checker = HealthChecker(version="1.0.0")
        checker.register_check("database", check_database)
        checker.register_check("redis", check_redis)

        # For liveness probe
        result = await checker.check_liveness()

        # For readiness probe
        result = await checker.check_readiness()
    """

    def __init__(
        self,
        version: str = "1.0.0",
        startup_grace_period_seconds: float = 30.0,
    ):
        """
        Initialize health checker.

        Args:
            version: Agent version string
            startup_grace_period_seconds: Grace period before startup probe fails
        """
        self.version = version
        self.startup_grace_period = startup_grace_period_seconds
        self._start_time = time.time()
        self._checks: Dict[str, Callable] = {}
        self._essential_checks: List[str] = []
        self._startup_complete = False

    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time

    def register_check(
        self,
        name: str,
        check_func: Callable,
        essential: bool = False,
    ) -> None:
        """
        Register a health check function.

        Args:
            name: Unique name for the check
            check_func: Async function that returns (status, message, details)
            essential: If True, failure marks service as unhealthy
        """
        self._checks[name] = check_func
        if essential:
            self._essential_checks.append(name)
        logger.info(f"Registered health check: {name} (essential={essential})")

    def mark_startup_complete(self) -> None:
        """Mark startup as complete."""
        self._startup_complete = True
        logger.info("Startup marked as complete")

    async def _run_check(self, name: str) -> ComponentHealth:
        """
        Run a single health check.

        Args:
            name: Name of the check to run

        Returns:
            ComponentHealth result
        """
        check_func = self._checks.get(name)
        if not check_func:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check not found",
                latency_ms=0.0,
                last_check=datetime.now(timezone.utc).isoformat(),
            )

        start_time = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(check_func):
                status, message, details = await check_func()
            else:
                status, message, details = check_func()

            latency_ms = (time.perf_counter() - start_time) * 1000

            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                latency_ms=latency_ms,
                last_check=datetime.now(timezone.utc).isoformat(),
                details=details or {},
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Health check '{name}' failed with error: {e}")

            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=latency_ms,
                last_check=datetime.now(timezone.utc).isoformat(),
                details={"error": str(e)},
            )

    async def check_all(self) -> HealthCheckResult:
        """
        Run all registered health checks.

        Returns:
            Aggregated HealthCheckResult
        """
        components = []

        # Run all checks concurrently
        if self._checks:
            tasks = [
                self._run_check(name)
                for name in self._checks
            ]
            components = await asyncio.gather(*tasks)

        # Determine overall status
        overall_status = self._aggregate_status(list(components))

        return HealthCheckResult(
            status=overall_status,
            components=list(components),
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=self.uptime_seconds,
            version=self.version,
        )

    def _aggregate_status(
        self, components: List[ComponentHealth]
    ) -> HealthStatus:
        """
        Aggregate component statuses into overall status.

        Args:
            components: List of component health results

        Returns:
            Aggregated HealthStatus
        """
        if not components:
            return HealthStatus.HEALTHY

        # Check essential components first
        for comp in components:
            if comp.name in self._essential_checks:
                if comp.status == HealthStatus.UNHEALTHY:
                    return HealthStatus.UNHEALTHY

        # Count statuses
        unhealthy_count = sum(
            1 for c in components if c.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for c in components if c.status == HealthStatus.DEGRADED
        )

        if unhealthy_count > len(components) / 2:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count > 0 or degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def check_liveness(self) -> HealthCheckResult:
        """
        Perform liveness probe check.

        Liveness checks if the application is running.
        A failing liveness probe causes container restart.

        Returns:
            HealthCheckResult for liveness
        """
        # Liveness is a simple "am I alive" check
        # Only essential components affect liveness
        essential_results = []

        for name in self._essential_checks:
            result = await self._run_check(name)
            essential_results.append(result)

        overall_status = self._aggregate_status(essential_results)

        return HealthCheckResult(
            status=overall_status,
            components=essential_results,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=self.uptime_seconds,
            version=self.version,
        )

    async def check_readiness(self) -> HealthCheckResult:
        """
        Perform readiness probe check.

        Readiness checks if the application can accept traffic.
        A failing readiness probe removes pod from service endpoints.

        Returns:
            HealthCheckResult for readiness
        """
        # Readiness requires all components to be healthy or degraded
        return await self.check_all()

    async def check_startup(self) -> HealthCheckResult:
        """
        Perform startup probe check.

        Startup checks if the application has finished initializing.
        Failing startup probe during grace period is OK.

        Returns:
            HealthCheckResult for startup
        """
        if self._startup_complete:
            return await self.check_readiness()

        # During grace period, return healthy if within grace period
        if self.uptime_seconds < self.startup_grace_period:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                components=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=self.uptime_seconds,
                version=self.version,
            )

        # Grace period exceeded, check actual readiness
        return await self.check_readiness()


# =============================================================================
# STANDARD HEALTH CHECKS
# =============================================================================

async def check_database_health(
    connection_pool: Any,
) -> tuple[HealthStatus, str, dict]:
    """
    Check database connection health.

    Args:
        connection_pool: Database connection pool

    Returns:
        Tuple of (status, message, details)
    """
    try:
        async with connection_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            if result == 1:
                return (
                    HealthStatus.HEALTHY,
                    "Database connection OK",
                    {"pool_size": connection_pool.get_size()},
                )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Database connection failed: {str(e)}",
            {"error": str(e)},
        )

    return (
        HealthStatus.UNKNOWN,
        "Unknown database state",
        {},
    )


async def check_redis_health(redis_client: Any) -> tuple[HealthStatus, str, dict]:
    """
    Check Redis connection health.

    Args:
        redis_client: Redis client instance

    Returns:
        Tuple of (status, message, details)
    """
    try:
        pong = await redis_client.ping()
        if pong:
            info = await redis_client.info()
            return (
                HealthStatus.HEALTHY,
                "Redis connection OK",
                {
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory_human"),
                },
            )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Redis connection failed: {str(e)}",
            {"error": str(e)},
        )

    return (
        HealthStatus.UNKNOWN,
        "Unknown Redis state",
        {},
    )


def check_memory_health(
    threshold_mb: float = 1024.0,
) -> tuple[HealthStatus, str, dict]:
    """
    Check memory usage.

    Args:
        threshold_mb: Memory threshold in MB

    Returns:
        Tuple of (status, message, details)
    """
    import psutil

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        if memory_mb > threshold_mb:
            return (
                HealthStatus.DEGRADED,
                f"High memory usage: {memory_mb:.1f}MB",
                {"memory_mb": memory_mb, "threshold_mb": threshold_mb},
            )
        else:
            return (
                HealthStatus.HEALTHY,
                f"Memory usage OK: {memory_mb:.1f}MB",
                {"memory_mb": memory_mb, "threshold_mb": threshold_mb},
            )
    except ImportError:
        return (
            HealthStatus.UNKNOWN,
            "psutil not available",
            {},
        )
    except Exception as e:
        return (
            HealthStatus.UNKNOWN,
            f"Memory check failed: {str(e)}",
            {"error": str(e)},
        )
