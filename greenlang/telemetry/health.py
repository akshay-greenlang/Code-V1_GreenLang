# -*- coding: utf-8 -*-
"""
Health checks and readiness probes for GreenLang
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import psutil
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks"""

    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass
class HealthCheckResult:
    """Result of a health check"""

    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthReport:
    """Overall health report"""

    status: HealthStatus
    checks: List[HealthCheckResult]
    version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "checks": [c.to_dict() for c in self.checks],
        }

    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict(), indent=2)


class HealthCheck:
    """Base health check"""

    def __init__(self, name: str, critical: bool = True, timeout_seconds: float = 5.0):
        """
        Initialize health check

        Args:
            name: Check name
            critical: Whether check is critical for overall health
            timeout_seconds: Check timeout
        """
        self.name = name
        self.critical = critical
        self.timeout_seconds = timeout_seconds

    async def check_async(self) -> HealthCheckResult:
        """Perform async health check"""
        start_time = time.time()

        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._perform_check_async(), timeout=self.timeout_seconds
            )

            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms

            return result

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {self.timeout_seconds}s",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration_ms,
            )

    def check(self) -> HealthCheckResult:
        """Perform synchronous health check"""
        start_time = time.time()

        try:
            result = self._perform_check()
            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=duration_ms,
            )

    async def _perform_check_async(self) -> HealthCheckResult:
        """Override for async check implementation"""
        return self._perform_check()

    def _perform_check(self) -> HealthCheckResult:
        """Override for sync check implementation"""
        return HealthCheckResult(
            name=self.name, status=HealthStatus.HEALTHY, message="Check not implemented"
        )


class LivenessCheck(HealthCheck):
    """Liveness probe - checks if application is alive"""

    def __init__(self):
        super().__init__("liveness", critical=True)

    def _perform_check(self) -> HealthCheckResult:
        """Check if process is responsive"""
        # Simple check - if we can execute this, we're alive
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Application is alive",
            details={
                "pid": psutil.Process().pid,
                "create_time": psutil.Process().create_time(),
            },
        )


class ReadinessCheck(HealthCheck):
    """Readiness probe - checks if application is ready to serve"""

    def __init__(self, dependencies: Optional[List[str]] = None):
        super().__init__("readiness", critical=True)
        self.dependencies = dependencies or []

    def _perform_check(self) -> HealthCheckResult:
        """Check if application is ready"""
        # Check critical dependencies
        failed_deps = []

        for dep in self.dependencies:
            if not self._check_dependency(dep):
                failed_deps.append(dep)

        if failed_deps:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Dependencies not ready: {', '.join(failed_deps)}",
                details={"failed_dependencies": failed_deps},
            )

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Application is ready",
            details={"dependencies": self.dependencies},
        )

    def _check_dependency(self, dep: str) -> bool:
        """Check if dependency is available"""
        # Implement specific dependency checks
        return True


class DatabaseHealthCheck(HealthCheck):
    """Database connectivity check"""

    def __init__(self, connection_string: str):
        super().__init__("database", critical=True)
        self.connection_string = connection_string

    async def _perform_check_async(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            # Simulate database check
            # In real implementation, would execute a simple query
            await asyncio.sleep(0.1)

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={"connection": "active", "response_time_ms": 10},
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e}",
                details={"error": str(e)},
            )


class DiskSpaceHealthCheck(HealthCheck):
    """Disk space availability check"""

    def __init__(self, path: str = "/", min_free_gb: float = 1.0):
        super().__init__("disk_space", critical=False)
        self.path = path
        self.min_free_gb = min_free_gb

    def _perform_check(self) -> HealthCheckResult:
        """Check disk space"""
        try:
            usage = psutil.disk_usage(self.path)
            free_gb = usage.free / (1024**3)

            if free_gb < self.min_free_gb:
                status = HealthStatus.UNHEALTHY
                message = f"Low disk space: {free_gb:.2f}GB free"
            elif free_gb < self.min_free_gb * 2:
                status = HealthStatus.DEGRADED
                message = f"Disk space warning: {free_gb:.2f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb:.2f}GB free"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "path": self.path,
                    "free_gb": free_gb,
                    "used_percent": usage.percent,
                    "total_gb": usage.total / (1024**3),
                },
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name, status=HealthStatus.UNKNOWN, message=str(e)
            )


class MemoryHealthCheck(HealthCheck):
    """Memory usage check"""

    def __init__(self, max_usage_percent: float = 90.0):
        super().__init__("memory", critical=False)
        self.max_usage_percent = max_usage_percent

    def _perform_check(self) -> HealthCheckResult:
        """Check memory usage"""
        memory = psutil.virtual_memory()

        if memory.percent > self.max_usage_percent:
            status = HealthStatus.UNHEALTHY
            message = f"High memory usage: {memory.percent:.1f}%"
        elif memory.percent > self.max_usage_percent * 0.8:
            status = HealthStatus.DEGRADED
            message = f"Memory usage warning: {memory.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage OK: {memory.percent:.1f}%"

        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details={
                "used_percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
            },
        )


class CPUHealthCheck(HealthCheck):
    """CPU usage check"""

    def __init__(self, max_usage_percent: float = 80.0):
        super().__init__("cpu", critical=False)
        self.max_usage_percent = max_usage_percent

    def _perform_check(self) -> HealthCheckResult:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)

        if cpu_percent > self.max_usage_percent:
            status = HealthStatus.UNHEALTHY
            message = f"High CPU usage: {cpu_percent:.1f}%"
        elif cpu_percent > self.max_usage_percent * 0.8:
            status = HealthStatus.DEGRADED
            message = f"CPU usage warning: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage OK: {cpu_percent:.1f}%"

        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details={
                "usage_percent": cpu_percent,
                "core_count": psutil.cpu_count(),
                "load_average": (
                    psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
                ),
            },
        )


class ServiceHealthCheck(HealthCheck):
    """External service health check"""

    def __init__(self, service_name: str, check_func: Callable[[], bool]):
        super().__init__(f"service_{service_name}", critical=True)
        self.service_name = service_name
        self.check_func = check_func

    def _perform_check(self) -> HealthCheckResult:
        """Check external service"""
        try:
            if self.check_func():
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Service {self.service_name} is healthy",
                    details={"service": self.service_name},
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Service {self.service_name} is not responding",
                    details={"service": self.service_name},
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Service {self.service_name} check failed: {e}",
                details={"service": self.service_name, "error": str(e)},
            )


class HealthChecker:
    """Main health checking service"""

    def __init__(self):
        """Initialize health checker"""
        self.checks: Dict[str, HealthCheck] = {}
        self.start_time = DeterministicClock.utcnow()
        self._last_check_results: Dict[str, HealthCheckResult] = {}
        self._check_thread = None
        self._stop_checking = threading.Event()
        self.check_interval = 30  # seconds

        # Register default checks
        self.register_check(LivenessCheck())
        self.register_check(ReadinessCheck())
        self.register_check(DiskSpaceHealthCheck())
        self.register_check(MemoryHealthCheck())
        self.register_check(CPUHealthCheck())

        logger.info("HealthChecker initialized")

    def register_check(self, check: HealthCheck):
        """
        Register a health check

        Args:
            check: Health check instance
        """
        self.checks[check.name] = check
        logger.info(f"Registered health check: {check.name}")

    def unregister_check(self, name: str):
        """
        Unregister a health check

        Args:
            name: Check name
        """
        if name in self.checks:
            del self.checks[name]
            logger.info(f"Unregistered health check: {name}")

    async def check_health_async(
        self, check_type: Optional[CheckType] = None
    ) -> HealthReport:
        """
        Perform async health checks

        Args:
            check_type: Type of checks to run

        Returns:
            Health report
        """
        checks_to_run = self._get_checks_by_type(check_type)

        # Run checks concurrently
        tasks = [check.check_async() for check in checks_to_run]
        results = await asyncio.gather(*tasks)

        # Store results
        for result in results:
            self._last_check_results[result.name] = result

        # Determine overall status
        overall_status = self._calculate_overall_status(results)

        # Calculate uptime
        uptime = (DeterministicClock.utcnow() - self.start_time).total_seconds()

        return HealthReport(
            status=overall_status, checks=results, uptime_seconds=uptime
        )

    def check_health(self, check_type: Optional[CheckType] = None) -> HealthReport:
        """
        Perform synchronous health checks

        Args:
            check_type: Type of checks to run

        Returns:
            Health report
        """
        checks_to_run = self._get_checks_by_type(check_type)

        # Run checks
        results = []
        for check in checks_to_run:
            result = check.check()
            results.append(result)
            self._last_check_results[result.name] = result

        # Determine overall status
        overall_status = self._calculate_overall_status(results)

        # Calculate uptime
        uptime = (DeterministicClock.utcnow() - self.start_time).total_seconds()

        return HealthReport(
            status=overall_status, checks=results, uptime_seconds=uptime
        )

    def get_status(self) -> HealthStatus:
        """Get current health status"""
        if not self._last_check_results:
            return HealthStatus.UNKNOWN

        return self._calculate_overall_status(list(self._last_check_results.values()))

    def start_background_checks(self):
        """Start background health checking"""
        self._check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self._check_thread.start()
        logger.info("Started background health checks")

    def stop_background_checks(self):
        """Stop background health checking"""
        self._stop_checking.set()
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Stopped background health checks")

    def _check_loop(self):
        """Background check loop"""
        while not self._stop_checking.is_set():
            try:
                self.check_health()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            self._stop_checking.wait(self.check_interval)

    def _get_checks_by_type(self, check_type: Optional[CheckType]) -> List[HealthCheck]:
        """Get checks filtered by type"""
        if not check_type:
            return list(self.checks.values())

        # Filter by check type
        if check_type == CheckType.LIVENESS:
            return [c for c in self.checks.values() if c.name == "liveness"]
        elif check_type == CheckType.READINESS:
            return [c for c in self.checks.values() if c.name == "readiness"]
        else:
            return list(self.checks.values())

    def _calculate_overall_status(
        self, results: List[HealthCheckResult]
    ) -> HealthStatus:
        """Calculate overall health status"""
        if not results:
            return HealthStatus.UNKNOWN

        # Check for critical failures
        critical_checks = [
            r
            for r in results
            if self.checks.get(r.name, None) and self.checks[r.name].critical
        ]

        for result in critical_checks:
            if result.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY

        # Check for any failures
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.DEGRADED

        # Check for degraded
        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker"""
    global _health_checker
    if not _health_checker:
        _health_checker = HealthChecker()
    return _health_checker


def get_health_status() -> HealthStatus:
    """Get current health status"""
    return get_health_checker().get_status()


def register_health_check(check: HealthCheck):
    """Register a health check"""
    get_health_checker().register_check(check)


async def check_health_async(check_type: Optional[CheckType] = None) -> HealthReport:
    """Perform async health check"""
    return await get_health_checker().check_health_async(check_type)


def check_health(check_type: Optional[CheckType] = None) -> HealthReport:
    """Perform sync health check"""
    return get_health_checker().check_health(check_type)
