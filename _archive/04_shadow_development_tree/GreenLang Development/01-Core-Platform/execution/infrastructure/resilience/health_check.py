"""
Health Check Implementation for GreenLang

This module provides health check endpoints and
component health monitoring.

Features:
- Liveness and readiness probes
- Component health aggregation
- Custom health indicators
- Kubernetes-compatible responses
- Async health checks

Example:
    >>> checker = HealthChecker()
    >>> checker.add_check("database", db_health_check)
    >>> status = await checker.check_health()
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status values."""
    UP = "UP"
    DOWN = "DOWN"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"


class ProbeType(str, Enum):
    """Kubernetes probe types."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass
class HealthCheckConfig:
    """Configuration for health checker."""
    include_details: bool = True
    timeout_seconds: float = 10.0
    cache_duration_seconds: float = 0.0  # 0 = no caching
    failure_threshold: int = 3  # Consecutive failures to mark DOWN
    success_threshold: int = 1  # Consecutive successes to mark UP


class ComponentHealth(BaseModel):
    """Health status of a component."""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)
    duration_ms: float = Field(default=0.0, description="Check duration")
    last_check: datetime = Field(default_factory=datetime.utcnow)
    consecutive_failures: int = Field(default=0)
    consecutive_successes: int = Field(default=0)


class HealthResponse(BaseModel):
    """Overall health response."""
    status: HealthStatus = Field(..., description="Overall status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: Optional[str] = Field(default=None, description="Service version")
    uptime_seconds: Optional[float] = Field(default=None)
    components: Dict[str, ComponentHealth] = Field(default_factory=dict)


class HealthIndicator:
    """
    Base class for health indicators.

    Implement custom health checks by extending this class.
    """

    async def check(self) -> ComponentHealth:
        """
        Perform health check.

        Returns:
            Component health status
        """
        raise NotImplementedError


class DatabaseHealthIndicator(HealthIndicator):
    """Health indicator for database connections."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool]
    ):
        """
        Initialize database health indicator.

        Args:
            name: Database name
            check_func: Function that returns True if healthy
        """
        self.name = name
        self.check_func = check_func

    async def check(self) -> ComponentHealth:
        """Check database health."""
        start_time = time.monotonic()

        try:
            if asyncio.iscoroutinefunction(self.check_func):
                is_healthy = await self.check_func()
            else:
                is_healthy = self.check_func()

            duration = (time.monotonic() - start_time) * 1000

            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UP if is_healthy else HealthStatus.DOWN,
                duration_ms=duration,
                details={"type": "database"},
            )

        except Exception as e:
            duration = (time.monotonic() - start_time) * 1000
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.DOWN,
                duration_ms=duration,
                error=str(e),
            )


class DiskSpaceHealthIndicator(HealthIndicator):
    """Health indicator for disk space."""

    def __init__(
        self,
        path: str = "/",
        threshold_percent: float = 90.0
    ):
        """
        Initialize disk space health indicator.

        Args:
            path: Path to check
            threshold_percent: Threshold percentage for warning
        """
        self.path = path
        self.threshold = threshold_percent

    async def check(self) -> ComponentHealth:
        """Check disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.path)
            percent_used = (used / total) * 100

            status = HealthStatus.UP
            if percent_used >= self.threshold:
                status = HealthStatus.DEGRADED

            return ComponentHealth(
                name="disk_space",
                status=status,
                details={
                    "path": self.path,
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "percent_used": round(percent_used, 2),
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                error=str(e),
            )


class MemoryHealthIndicator(HealthIndicator):
    """Health indicator for memory usage."""

    def __init__(self, threshold_percent: float = 90.0):
        """
        Initialize memory health indicator.

        Args:
            threshold_percent: Threshold percentage for warning
        """
        self.threshold = threshold_percent

    async def check(self) -> ComponentHealth:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()

            status = HealthStatus.UP
            if memory.percent >= self.threshold:
                status = HealthStatus.DEGRADED

            return ComponentHealth(
                name="memory",
                status=status,
                details={
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "percent_used": memory.percent,
                },
            )

        except ImportError:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                error="psutil not installed",
            )
        except Exception as e:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                error=str(e),
            )


class HealthChecker:
    """
    Health checker for service components.

    Aggregates health from multiple components and provides
    overall health status.

    Attributes:
        config: Health checker configuration
        components: Registered health checks

    Example:
        >>> checker = HealthChecker()
        >>> checker.add_check("database", db_health)
        >>> checker.add_check("cache", cache_health)
        >>> health = await checker.check_health()
        >>> print(health.status)
    """

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        """
        Initialize health checker.

        Args:
            config: Health checker configuration
        """
        self.config = config or HealthCheckConfig()
        self._components: Dict[str, Callable] = {}
        self._indicators: Dict[str, HealthIndicator] = {}
        self._cached_health: Optional[HealthResponse] = None
        self._cache_time: Optional[float] = None
        self._component_states: Dict[str, ComponentHealth] = {}
        self._start_time = time.monotonic()
        self._version: Optional[str] = None
        self._router: Optional[APIRouter] = None

        logger.info("HealthChecker initialized")

    def set_version(self, version: str) -> None:
        """Set service version."""
        self._version = version

    def add_check(
        self,
        name: str,
        check_func: Callable[[], bool]
    ) -> None:
        """
        Add a simple health check function.

        Args:
            name: Component name
            check_func: Function that returns True if healthy
        """
        self._components[name] = check_func
        self._component_states[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN
        )
        logger.debug(f"Added health check: {name}")

    def add_indicator(
        self,
        name: str,
        indicator: HealthIndicator
    ) -> None:
        """
        Add a health indicator.

        Args:
            name: Component name
            indicator: Health indicator instance
        """
        self._indicators[name] = indicator
        self._component_states[name] = ComponentHealth(
            name=name,
            status=HealthStatus.UNKNOWN
        )
        logger.debug(f"Added health indicator: {name}")

    def remove_check(self, name: str) -> None:
        """Remove a health check."""
        self._components.pop(name, None)
        self._indicators.pop(name, None)
        self._component_states.pop(name, None)

    async def check_component(self, name: str) -> ComponentHealth:
        """
        Check health of a single component.

        Args:
            name: Component name

        Returns:
            Component health status
        """
        start_time = time.monotonic()
        current_state = self._component_states.get(name)

        try:
            # Check indicator first
            if name in self._indicators:
                health = await asyncio.wait_for(
                    self._indicators[name].check(),
                    timeout=self.config.timeout_seconds
                )
            elif name in self._components:
                check_func = self._components[name]
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await asyncio.wait_for(
                        check_func(),
                        timeout=self.config.timeout_seconds
                    )
                else:
                    is_healthy = check_func()

                health = ComponentHealth(
                    name=name,
                    status=HealthStatus.UP if is_healthy else HealthStatus.DOWN,
                    duration_ms=(time.monotonic() - start_time) * 1000,
                )
            else:
                health = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    error="Check not found",
                )

            # Update consecutive counters
            if health.status == HealthStatus.UP:
                health.consecutive_successes = (
                    current_state.consecutive_successes + 1 if current_state else 1
                )
                health.consecutive_failures = 0
            else:
                health.consecutive_failures = (
                    current_state.consecutive_failures + 1 if current_state else 1
                )
                health.consecutive_successes = 0

            self._component_states[name] = health
            return health

        except asyncio.TimeoutError:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.DOWN,
                error="Health check timeout",
                duration_ms=self.config.timeout_seconds * 1000,
                consecutive_failures=(
                    current_state.consecutive_failures + 1 if current_state else 1
                ),
            )
            self._component_states[name] = health
            return health

        except Exception as e:
            health = ComponentHealth(
                name=name,
                status=HealthStatus.DOWN,
                error=str(e),
                duration_ms=(time.monotonic() - start_time) * 1000,
                consecutive_failures=(
                    current_state.consecutive_failures + 1 if current_state else 1
                ),
            )
            self._component_states[name] = health
            return health

    async def check_health(self) -> HealthResponse:
        """
        Check health of all components.

        Returns:
            Overall health response
        """
        # Check cache
        if self.config.cache_duration_seconds > 0:
            if (self._cached_health and
                    self._cache_time and
                    time.monotonic() - self._cache_time < self.config.cache_duration_seconds):
                return self._cached_health

        # Check all components concurrently
        component_names = list(set(
            list(self._components.keys()) +
            list(self._indicators.keys())
        ))

        if not component_names:
            return HealthResponse(
                status=HealthStatus.UP,
                version=self._version,
                uptime_seconds=time.monotonic() - self._start_time,
            )

        tasks = [
            self.check_component(name)
            for name in component_names
        ]

        component_healths = await asyncio.gather(*tasks)

        # Aggregate status
        components = {h.name: h for h in component_healths}
        overall_status = self._aggregate_status(component_healths)

        response = HealthResponse(
            status=overall_status,
            version=self._version,
            uptime_seconds=time.monotonic() - self._start_time,
            components=components if self.config.include_details else {},
        )

        # Update cache
        self._cached_health = response
        self._cache_time = time.monotonic()

        return response

    def _aggregate_status(
        self,
        component_healths: List[ComponentHealth]
    ) -> HealthStatus:
        """Aggregate component statuses."""
        if not component_healths:
            return HealthStatus.UP

        statuses = [h.status for h in component_healths]

        if all(s == HealthStatus.UP for s in statuses):
            return HealthStatus.UP
        elif any(s == HealthStatus.DOWN for s in statuses):
            return HealthStatus.DOWN
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    async def check_liveness(self) -> HealthResponse:
        """
        Liveness probe check.

        Simple check that the service is running.
        """
        return HealthResponse(
            status=HealthStatus.UP,
            version=self._version,
            uptime_seconds=time.monotonic() - self._start_time,
        )

    async def check_readiness(self) -> HealthResponse:
        """
        Readiness probe check.

        Full health check to determine if ready to receive traffic.
        """
        return await self.check_health()

    def create_router(self, prefix: str = "/health") -> APIRouter:
        """
        Create FastAPI router for health endpoints.

        Args:
            prefix: Route prefix

        Returns:
            FastAPI router
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for health router")

        router = APIRouter(prefix=prefix)

        @router.get("", response_model=HealthResponse)
        async def health():
            return await self.check_health()

        @router.get("/live", response_model=HealthResponse)
        async def liveness():
            return await self.check_liveness()

        @router.get("/ready", response_model=HealthResponse)
        async def readiness():
            return await self.check_readiness()

        @router.get("/components/{name}", response_model=ComponentHealth)
        async def component_health(name: str):
            return await self.check_component(name)

        self._router = router
        return router

    @property
    def router(self) -> Optional[APIRouter]:
        """Get the FastAPI router."""
        return self._router
