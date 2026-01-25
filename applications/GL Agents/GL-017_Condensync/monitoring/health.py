# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Health Check Module
========================================

Kubernetes-compatible health checks for liveness, readiness, and startup probes.
Implements comprehensive component health verification for condenser optimization.

Health Check Types:
- Liveness Probe: Is the application running?
- Readiness Probe: Can the application accept traffic?
- Startup Probe: Has the application finished initializing?

Dependency Checks:
- OPC-UA Server connectivity
- Kafka broker connectivity
- CMMS API connectivity
- Database connectivity
- Cache (Redis) connectivity
- PI Server/Historian connectivity

Data Freshness:
- Tag data freshness verification
- Calculation result freshness
- Recommendation freshness

Standards Compliance:
- Kubernetes health check best practices
- GreenLang Global AI Standards v2.0

Example:
    >>> from monitoring.health import HealthCheckManager, HealthStatus
    >>> health = HealthCheckManager(version="1.0.0")
    >>> health.register_dependency_check("opc_ua", check_opc_ua_health)
    >>> health.register_dependency_check("kafka", check_kafka_health)
    >>>
    >>> # For liveness probe
    >>> result = await health.check_liveness()
    >>> assert result.is_healthy
    >>>
    >>> # For readiness probe
    >>> result = await health.check_readiness()

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class HealthStatus(str, Enum):
    """Health status values for components and overall health."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProbeType(str, Enum):
    """Kubernetes probe types."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


class DependencyType(str, Enum):
    """Types of external dependencies."""
    OPC_UA = "opc_ua"
    KAFKA = "kafka"
    CMMS = "cmms"
    DATABASE = "database"
    CACHE = "cache"
    PI_SERVER = "pi_server"
    HISTORIAN = "historian"
    API = "api"


class CheckCategory(str, Enum):
    """Categories of health checks."""
    DEPENDENCY = "dependency"
    DATA_FRESHNESS = "data_freshness"
    RESOURCE = "resource"
    INTERNAL = "internal"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class ComponentHealth:
    """
    Health status of a single component.

    Attributes:
        name: Component name
        status: Health status
        message: Status message
        latency_ms: Check latency in milliseconds
        last_check: ISO timestamp of last check
        category: Check category
        details: Additional details
    """
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    last_check: str
    category: CheckCategory = CheckCategory.INTERNAL
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """
    Overall health check result.

    Attributes:
        status: Overall health status
        probe_type: Type of probe (liveness, readiness, startup)
        components: List of component health results
        timestamp: Check timestamp
        uptime_seconds: Application uptime
        version: Agent version
    """
    status: HealthStatus
    probe_type: ProbeType
    components: List[ComponentHealth]
    timestamp: str
    uptime_seconds: float
    version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        return {
            "status": self.status.value,
            "probe_type": self.probe_type.value,
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
                    "category": c.category.value,
                    "details": c.details,
                }
                for c in self.components
            ],
            "summary": {
                "total_checks": len(self.components),
                "healthy": sum(1 for c in self.components if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.components if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.components if c.status == HealthStatus.UNHEALTHY),
                "unknown": sum(1 for c in self.components if c.status == HealthStatus.UNKNOWN),
            },
        }

    @property
    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Check if service is ready to accept traffic."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    @property
    def http_status_code(self) -> int:
        """Get HTTP status code for health check response."""
        if self.status == HealthStatus.HEALTHY:
            return 200
        elif self.status == HealthStatus.DEGRADED:
            return 200  # Degraded is still serving
        else:
            return 503


@dataclass
class DataFreshnessConfig:
    """
    Configuration for data freshness checks.

    Attributes:
        tag_group: Name of the tag group
        max_age_seconds: Maximum acceptable age in seconds
        condenser_ids: List of condenser IDs to check
        is_critical: Whether this check is critical for readiness
    """
    tag_group: str
    max_age_seconds: float
    condenser_ids: List[str] = field(default_factory=list)
    is_critical: bool = True


@dataclass
class DependencyConfig:
    """
    Configuration for dependency health check.

    Attributes:
        name: Dependency name
        dependency_type: Type of dependency
        endpoint: Connection endpoint
        timeout_seconds: Check timeout
        is_essential: Whether failure marks service as unhealthy
        check_interval_seconds: How often to check
    """
    name: str
    dependency_type: DependencyType
    endpoint: str = ""
    timeout_seconds: float = 5.0
    is_essential: bool = True
    check_interval_seconds: float = 30.0


# =============================================================================
# HEALTH CHECK FUNCTIONS TYPE
# =============================================================================

# Type alias for health check functions
HealthCheckFunc = Callable[[], Union[
    Tuple[HealthStatus, str, Dict[str, Any]],
    "asyncio.coroutine"
]]


# =============================================================================
# HEALTH CHECK MANAGER
# =============================================================================

class HealthCheckManager:
    """
    Comprehensive health check manager for GL-017 CONDENSYNC.

    Manages health checks for all components and dependencies.
    Provides Kubernetes-compatible liveness, readiness, and startup probes.

    Example:
        >>> health = HealthCheckManager(version="1.0.0")
        >>> health.register_dependency_check("opc_ua", check_opc_ua)
        >>> health.register_dependency_check("kafka", check_kafka, essential=True)
        >>> health.register_data_freshness_check(
        ...     DataFreshnessConfig(tag_group="temperatures", max_age_seconds=300)
        ... )
        >>>
        >>> # For liveness probe endpoint
        >>> result = await health.check_liveness()
        >>> if result.is_healthy:
        ...     return {"status": "ok"}
        >>>
        >>> # For readiness probe endpoint
        >>> result = await health.check_readiness()
        >>> return result.to_dict(), result.http_status_code

    Attributes:
        version: Agent version string
        startup_grace_period: Seconds before startup probe fails
    """

    def __init__(
        self,
        version: str = "1.0.0",
        startup_grace_period_seconds: float = 60.0,
        instance_id: Optional[str] = None,
    ):
        """
        Initialize health check manager.

        Args:
            version: Agent version string
            startup_grace_period_seconds: Grace period for startup probe
            instance_id: Optional instance identifier
        """
        self.version = version
        self.startup_grace_period = startup_grace_period_seconds
        self.instance_id = instance_id or os.getenv("HOSTNAME", "default")

        self._start_time = time.time()
        self._startup_complete = False

        # Check registries
        self._dependency_checks: Dict[str, Tuple[HealthCheckFunc, DependencyConfig]] = {}
        self._data_freshness_checks: Dict[str, DataFreshnessConfig] = {}
        self._internal_checks: Dict[str, HealthCheckFunc] = {}
        self._resource_checks: Dict[str, HealthCheckFunc] = {}

        # Essential checks that affect liveness
        self._essential_checks: List[str] = []

        # Cached results for efficiency
        self._check_cache: Dict[str, Tuple[ComponentHealth, float]] = {}
        self._cache_ttl_seconds: float = 5.0

        # Data freshness tracking
        self._last_data_timestamps: Dict[str, datetime] = {}

        logger.info(
            f"HealthCheckManager initialized: version={version}, "
            f"instance={self.instance_id}, grace_period={startup_grace_period_seconds}s"
        )

    @property
    def uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self._start_time

    # =========================================================================
    # CHECK REGISTRATION
    # =========================================================================

    def register_dependency_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
        dependency_type: DependencyType = DependencyType.API,
        essential: bool = False,
        endpoint: str = "",
        timeout_seconds: float = 5.0,
    ) -> None:
        """
        Register a dependency health check.

        Args:
            name: Unique name for the check
            check_func: Function that returns (status, message, details)
            dependency_type: Type of dependency
            essential: If True, failure marks service as unhealthy
            endpoint: Connection endpoint
            timeout_seconds: Check timeout
        """
        config = DependencyConfig(
            name=name,
            dependency_type=dependency_type,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            is_essential=essential,
        )

        self._dependency_checks[name] = (check_func, config)

        if essential:
            self._essential_checks.append(name)

        logger.info(
            f"Registered dependency check: {name} "
            f"(type={dependency_type.value}, essential={essential})"
        )

    def register_data_freshness_check(
        self,
        config: DataFreshnessConfig,
        check_func: Optional[HealthCheckFunc] = None,
    ) -> None:
        """
        Register a data freshness check.

        Args:
            config: Data freshness configuration
            check_func: Optional custom check function
        """
        self._data_freshness_checks[config.tag_group] = config

        if config.is_critical:
            self._essential_checks.append(f"freshness:{config.tag_group}")

        logger.info(
            f"Registered data freshness check: {config.tag_group} "
            f"(max_age={config.max_age_seconds}s, critical={config.is_critical})"
        )

    def register_internal_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
        essential: bool = False,
    ) -> None:
        """
        Register an internal health check.

        Args:
            name: Unique name for the check
            check_func: Function that returns (status, message, details)
            essential: If True, failure marks service as unhealthy
        """
        self._internal_checks[name] = check_func

        if essential:
            self._essential_checks.append(f"internal:{name}")

        logger.info(f"Registered internal check: {name} (essential={essential})")

    def register_resource_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
    ) -> None:
        """
        Register a resource health check (CPU, memory, disk).

        Args:
            name: Unique name for the check
            check_func: Function that returns (status, message, details)
        """
        self._resource_checks[name] = check_func
        logger.info(f"Registered resource check: {name}")

    def mark_startup_complete(self) -> None:
        """Mark startup as complete."""
        self._startup_complete = True
        logger.info("Startup marked as complete")

    def update_data_timestamp(
        self,
        tag_group: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update the last data timestamp for a tag group.

        Args:
            tag_group: Name of the tag group
            timestamp: Data timestamp (defaults to now)
        """
        self._last_data_timestamps[tag_group] = timestamp or datetime.now(timezone.utc)

    # =========================================================================
    # CHECK EXECUTION
    # =========================================================================

    async def _run_check(
        self,
        name: str,
        check_func: HealthCheckFunc,
        category: CheckCategory,
        timeout: float = 5.0,
    ) -> ComponentHealth:
        """
        Run a single health check with timeout.

        Args:
            name: Check name
            check_func: Check function
            category: Check category
            timeout: Timeout in seconds

        Returns:
            ComponentHealth result
        """
        # Check cache first
        cache_key = f"{category.value}:{name}"
        if cache_key in self._check_cache:
            cached_result, cache_time = self._check_cache[cache_key]
            if time.time() - cache_time < self._cache_ttl_seconds:
                return cached_result

        start_time = time.perf_counter()

        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=timeout)
            else:
                result = check_func()

            status, message, details = result
            latency_ms = (time.perf_counter() - start_time) * 1000

            health = ComponentHealth(
                name=name,
                status=status,
                message=message,
                latency_ms=latency_ms,
                last_check=datetime.now(timezone.utc).isoformat(),
                category=category,
                details=details or {},
            )

        except asyncio.TimeoutError:
            latency_ms = timeout * 1000
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {timeout}s",
                latency_ms=latency_ms,
                last_check=datetime.now(timezone.utc).isoformat(),
                category=category,
                details={"error": "timeout"},
            )
            logger.warning(f"Health check '{name}' timed out after {timeout}s")

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            health = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=latency_ms,
                last_check=datetime.now(timezone.utc).isoformat(),
                category=category,
                details={"error": str(e), "error_type": type(e).__name__},
            )
            logger.error(f"Health check '{name}' failed with error: {e}")

        # Cache result
        self._check_cache[cache_key] = (health, time.time())

        return health

    async def _check_data_freshness(
        self,
        config: DataFreshnessConfig,
    ) -> ComponentHealth:
        """
        Check data freshness for a tag group.

        Args:
            config: Data freshness configuration

        Returns:
            ComponentHealth result
        """
        start_time = time.perf_counter()
        name = f"freshness:{config.tag_group}"

        last_timestamp = self._last_data_timestamps.get(config.tag_group)

        if last_timestamp is None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"No data received for tag group: {config.tag_group}",
                latency_ms=latency_ms,
                last_check=datetime.now(timezone.utc).isoformat(),
                category=CheckCategory.DATA_FRESHNESS,
                details={"tag_group": config.tag_group, "max_age_seconds": config.max_age_seconds},
            )

        age_seconds = (datetime.now(timezone.utc) - last_timestamp).total_seconds()
        latency_ms = (time.perf_counter() - start_time) * 1000

        if age_seconds <= config.max_age_seconds:
            status = HealthStatus.HEALTHY
            message = f"Data fresh: age={age_seconds:.1f}s (max={config.max_age_seconds}s)"
        elif age_seconds <= config.max_age_seconds * 2:
            status = HealthStatus.DEGRADED
            message = f"Data stale: age={age_seconds:.1f}s (max={config.max_age_seconds}s)"
        else:
            status = HealthStatus.UNHEALTHY
            message = f"Data too old: age={age_seconds:.1f}s (max={config.max_age_seconds}s)"

        return ComponentHealth(
            name=name,
            status=status,
            message=message,
            latency_ms=latency_ms,
            last_check=datetime.now(timezone.utc).isoformat(),
            category=CheckCategory.DATA_FRESHNESS,
            details={
                "tag_group": config.tag_group,
                "age_seconds": round(age_seconds, 2),
                "max_age_seconds": config.max_age_seconds,
                "last_timestamp": last_timestamp.isoformat(),
            },
        )

    def _aggregate_status(
        self,
        components: List[ComponentHealth],
        check_essential_only: bool = False,
    ) -> HealthStatus:
        """
        Aggregate component statuses into overall status.

        Args:
            components: List of component health results
            check_essential_only: Only consider essential checks

        Returns:
            Aggregated HealthStatus
        """
        if not components:
            return HealthStatus.HEALTHY

        # Filter to essential checks if requested
        if check_essential_only:
            essential_names = set(self._essential_checks)
            components = [c for c in components if c.name in essential_names]

            if not components:
                return HealthStatus.HEALTHY

        # Check for unhealthy essential components
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

        # Determine overall status
        if unhealthy_count > len(components) / 2:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count > 0 or degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    # =========================================================================
    # PROBE METHODS
    # =========================================================================

    async def check_all(self) -> HealthCheckResult:
        """
        Run all registered health checks.

        Returns:
            Comprehensive HealthCheckResult
        """
        components: List[ComponentHealth] = []
        tasks = []

        # Dependency checks
        for name, (check_func, config) in self._dependency_checks.items():
            task = self._run_check(
                name=name,
                check_func=check_func,
                category=CheckCategory.DEPENDENCY,
                timeout=config.timeout_seconds,
            )
            tasks.append(task)

        # Data freshness checks
        for config in self._data_freshness_checks.values():
            task = self._check_data_freshness(config)
            tasks.append(task)

        # Internal checks
        for name, check_func in self._internal_checks.items():
            task = self._run_check(
                name=f"internal:{name}",
                check_func=check_func,
                category=CheckCategory.INTERNAL,
            )
            tasks.append(task)

        # Resource checks
        for name, check_func in self._resource_checks.items():
            task = self._run_check(
                name=f"resource:{name}",
                check_func=check_func,
                category=CheckCategory.RESOURCE,
            )
            tasks.append(task)

        # Run all checks concurrently
        if tasks:
            components = await asyncio.gather(*tasks)

        # Determine overall status
        overall_status = self._aggregate_status(list(components))

        return HealthCheckResult(
            status=overall_status,
            probe_type=ProbeType.READINESS,
            components=list(components),
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=self.uptime_seconds,
            version=self.version,
        )

    async def check_liveness(self) -> HealthCheckResult:
        """
        Perform liveness probe check.

        Liveness checks if the application is running.
        A failing liveness probe causes container restart.

        Only checks essential components.

        Returns:
            HealthCheckResult for liveness probe
        """
        components: List[ComponentHealth] = []

        # Only check essential dependencies
        for name in self._essential_checks:
            if name in self._dependency_checks:
                check_func, config = self._dependency_checks[name]
                result = await self._run_check(
                    name=name,
                    check_func=check_func,
                    category=CheckCategory.DEPENDENCY,
                    timeout=config.timeout_seconds,
                )
                components.append(result)

            elif name.startswith("internal:"):
                internal_name = name.replace("internal:", "")
                if internal_name in self._internal_checks:
                    result = await self._run_check(
                        name=name,
                        check_func=self._internal_checks[internal_name],
                        category=CheckCategory.INTERNAL,
                    )
                    components.append(result)

        # Determine status based on essential components only
        overall_status = self._aggregate_status(components, check_essential_only=True)

        return HealthCheckResult(
            status=overall_status,
            probe_type=ProbeType.LIVENESS,
            components=components,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=self.uptime_seconds,
            version=self.version,
        )

    async def check_readiness(self) -> HealthCheckResult:
        """
        Perform readiness probe check.

        Readiness checks if the application can accept traffic.
        A failing readiness probe removes pod from service endpoints.

        Checks all registered checks.

        Returns:
            HealthCheckResult for readiness probe
        """
        result = await self.check_all()
        result.probe_type = ProbeType.READINESS
        return result

    async def check_startup(self) -> HealthCheckResult:
        """
        Perform startup probe check.

        Startup checks if the application has finished initializing.
        Failing startup probe during grace period is OK.

        Returns:
            HealthCheckResult for startup probe
        """
        if self._startup_complete:
            result = await self.check_readiness()
            result.probe_type = ProbeType.STARTUP
            return result

        # During grace period, return healthy
        if self.uptime_seconds < self.startup_grace_period:
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                probe_type=ProbeType.STARTUP,
                components=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=self.uptime_seconds,
                version=self.version,
            )

        # Grace period exceeded, check actual readiness
        result = await self.check_readiness()
        result.probe_type = ProbeType.STARTUP
        return result

    # =========================================================================
    # QUICK CHECK METHODS
    # =========================================================================

    async def is_alive(self) -> bool:
        """
        Quick liveness check.

        Returns:
            True if service is alive
        """
        result = await self.check_liveness()
        return result.is_healthy

    async def is_ready(self) -> bool:
        """
        Quick readiness check.

        Returns:
            True if service is ready
        """
        result = await self.check_readiness()
        return result.is_ready


# =============================================================================
# STANDARD DEPENDENCY CHECKS
# =============================================================================

async def check_opc_ua_health(
    client: Any,
    timeout: float = 5.0,
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
    """
    Check OPC-UA server connection health.

    Args:
        client: OPC-UA client instance
        timeout: Check timeout

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Check if client is connected
        if not hasattr(client, 'is_connected') or not client.is_connected():
            return (
                HealthStatus.UNHEALTHY,
                "OPC-UA client not connected",
                {"connected": False},
            )

        # Try to read server state
        server_state = await asyncio.wait_for(
            client.get_server_state(),
            timeout=timeout,
        )

        return (
            HealthStatus.HEALTHY,
            f"OPC-UA connected, server state: {server_state}",
            {
                "connected": True,
                "server_state": str(server_state),
            },
        )

    except asyncio.TimeoutError:
        return (
            HealthStatus.UNHEALTHY,
            f"OPC-UA health check timed out after {timeout}s",
            {"error": "timeout"},
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"OPC-UA health check failed: {str(e)}",
            {"error": str(e)},
        )


async def check_kafka_health(
    producer: Any,
    consumer: Any = None,
    bootstrap_servers: str = "",
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
    """
    Check Kafka broker connectivity.

    Args:
        producer: Kafka producer instance
        consumer: Optional Kafka consumer instance
        bootstrap_servers: Kafka bootstrap servers

    Returns:
        Tuple of (status, message, details)
    """
    try:
        details: Dict[str, Any] = {"bootstrap_servers": bootstrap_servers}

        # Check producer health
        if hasattr(producer, 'bootstrap_connected'):
            producer_connected = producer.bootstrap_connected()
            details["producer_connected"] = producer_connected
        else:
            producer_connected = True  # Assume connected if no method

        # Check consumer health if provided
        consumer_connected = True
        if consumer and hasattr(consumer, 'bootstrap_connected'):
            consumer_connected = consumer.bootstrap_connected()
            details["consumer_connected"] = consumer_connected

        if producer_connected and consumer_connected:
            return (
                HealthStatus.HEALTHY,
                "Kafka connection healthy",
                details,
            )
        elif producer_connected or consumer_connected:
            return (
                HealthStatus.DEGRADED,
                "Kafka connection partially healthy",
                details,
            )
        else:
            return (
                HealthStatus.UNHEALTHY,
                "Kafka connection unhealthy",
                details,
            )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Kafka health check failed: {str(e)}",
            {"error": str(e)},
        )


async def check_cmms_health(
    client: Any,
    base_url: str = "",
    timeout: float = 5.0,
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
    """
    Check CMMS API connectivity.

    Args:
        client: HTTP client instance
        base_url: CMMS API base URL
        timeout: Check timeout

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Try to hit health endpoint
        response = await asyncio.wait_for(
            client.get(f"{base_url}/health"),
            timeout=timeout,
        )

        if response.status_code == 200:
            return (
                HealthStatus.HEALTHY,
                "CMMS API healthy",
                {
                    "base_url": base_url,
                    "status_code": response.status_code,
                },
            )
        elif response.status_code < 500:
            return (
                HealthStatus.DEGRADED,
                f"CMMS API returned {response.status_code}",
                {
                    "base_url": base_url,
                    "status_code": response.status_code,
                },
            )
        else:
            return (
                HealthStatus.UNHEALTHY,
                f"CMMS API error: {response.status_code}",
                {
                    "base_url": base_url,
                    "status_code": response.status_code,
                },
            )

    except asyncio.TimeoutError:
        return (
            HealthStatus.UNHEALTHY,
            f"CMMS API timed out after {timeout}s",
            {"base_url": base_url, "error": "timeout"},
        )
    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"CMMS API check failed: {str(e)}",
            {"base_url": base_url, "error": str(e)},
        )


async def check_database_health(
    connection_pool: Any,
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
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
                pool_size = (
                    connection_pool.get_size()
                    if hasattr(connection_pool, 'get_size')
                    else "unknown"
                )
                return (
                    HealthStatus.HEALTHY,
                    "Database connection healthy",
                    {"pool_size": pool_size},
                )

        return (
            HealthStatus.UNKNOWN,
            "Database returned unexpected result",
            {},
        )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Database connection failed: {str(e)}",
            {"error": str(e)},
        )


async def check_redis_health(
    redis_client: Any,
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
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
                "Redis connection healthy",
                {
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory_human"),
                },
            )

        return (
            HealthStatus.UNHEALTHY,
            "Redis ping failed",
            {},
        )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"Redis connection failed: {str(e)}",
            {"error": str(e)},
        )


async def check_pi_server_health(
    pi_client: Any,
    server_name: str = "",
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
    """
    Check PI Server/Historian connectivity.

    Args:
        pi_client: PI client instance
        server_name: PI server name

    Returns:
        Tuple of (status, message, details)
    """
    try:
        # Check connection
        if hasattr(pi_client, 'is_connected') and pi_client.is_connected():
            return (
                HealthStatus.HEALTHY,
                f"PI Server '{server_name}' connected",
                {"server_name": server_name, "connected": True},
            )

        return (
            HealthStatus.UNHEALTHY,
            f"PI Server '{server_name}' not connected",
            {"server_name": server_name, "connected": False},
        )

    except Exception as e:
        return (
            HealthStatus.UNHEALTHY,
            f"PI Server check failed: {str(e)}",
            {"server_name": server_name, "error": str(e)},
        )


# =============================================================================
# RESOURCE CHECKS
# =============================================================================

def check_memory_health(
    threshold_mb: float = 1024.0,
    warning_threshold_mb: float = 768.0,
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
    """
    Check process memory usage.

    Args:
        threshold_mb: Unhealthy threshold in MB
        warning_threshold_mb: Warning threshold in MB

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)

        details = {
            "memory_mb": round(memory_mb, 2),
            "threshold_mb": threshold_mb,
            "warning_threshold_mb": warning_threshold_mb,
        }

        if memory_mb > threshold_mb:
            return (
                HealthStatus.UNHEALTHY,
                f"High memory usage: {memory_mb:.1f}MB (threshold: {threshold_mb}MB)",
                details,
            )
        elif memory_mb > warning_threshold_mb:
            return (
                HealthStatus.DEGRADED,
                f"Elevated memory usage: {memory_mb:.1f}MB",
                details,
            )
        else:
            return (
                HealthStatus.HEALTHY,
                f"Memory usage OK: {memory_mb:.1f}MB",
                details,
            )

    except ImportError:
        return (
            HealthStatus.UNKNOWN,
            "psutil not available for memory check",
            {},
        )
    except Exception as e:
        return (
            HealthStatus.UNKNOWN,
            f"Memory check failed: {str(e)}",
            {"error": str(e)},
        )


def check_cpu_health(
    threshold_percent: float = 90.0,
    warning_threshold_percent: float = 70.0,
    interval: float = 0.1,
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
    """
    Check process CPU usage.

    Args:
        threshold_percent: Unhealthy threshold percentage
        warning_threshold_percent: Warning threshold percentage
        interval: Measurement interval

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import psutil

        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=interval)

        details = {
            "cpu_percent": round(cpu_percent, 2),
            "threshold_percent": threshold_percent,
            "warning_threshold_percent": warning_threshold_percent,
        }

        if cpu_percent > threshold_percent:
            return (
                HealthStatus.UNHEALTHY,
                f"High CPU usage: {cpu_percent:.1f}%",
                details,
            )
        elif cpu_percent > warning_threshold_percent:
            return (
                HealthStatus.DEGRADED,
                f"Elevated CPU usage: {cpu_percent:.1f}%",
                details,
            )
        else:
            return (
                HealthStatus.HEALTHY,
                f"CPU usage OK: {cpu_percent:.1f}%",
                details,
            )

    except ImportError:
        return (
            HealthStatus.UNKNOWN,
            "psutil not available for CPU check",
            {},
        )
    except Exception as e:
        return (
            HealthStatus.UNKNOWN,
            f"CPU check failed: {str(e)}",
            {"error": str(e)},
        )


def check_disk_health(
    path: str = "/",
    threshold_percent: float = 90.0,
    warning_threshold_percent: float = 80.0,
) -> Tuple[HealthStatus, str, Dict[str, Any]]:
    """
    Check disk usage.

    Args:
        path: Path to check
        threshold_percent: Unhealthy threshold percentage
        warning_threshold_percent: Warning threshold percentage

    Returns:
        Tuple of (status, message, details)
    """
    try:
        import psutil

        disk = psutil.disk_usage(path)
        usage_percent = disk.percent

        details = {
            "path": path,
            "usage_percent": round(usage_percent, 2),
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "threshold_percent": threshold_percent,
        }

        if usage_percent > threshold_percent:
            return (
                HealthStatus.UNHEALTHY,
                f"High disk usage: {usage_percent:.1f}%",
                details,
            )
        elif usage_percent > warning_threshold_percent:
            return (
                HealthStatus.DEGRADED,
                f"Elevated disk usage: {usage_percent:.1f}%",
                details,
            )
        else:
            return (
                HealthStatus.HEALTHY,
                f"Disk usage OK: {usage_percent:.1f}%",
                details,
            )

    except ImportError:
        return (
            HealthStatus.UNKNOWN,
            "psutil not available for disk check",
            {},
        )
    except Exception as e:
        return (
            HealthStatus.UNKNOWN,
            f"Disk check failed: {str(e)}",
            {"error": str(e)},
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "HealthCheckManager",
    # Data classes
    "HealthCheckResult",
    "ComponentHealth",
    "DataFreshnessConfig",
    "DependencyConfig",
    # Enums
    "HealthStatus",
    "ProbeType",
    "DependencyType",
    "CheckCategory",
    # Standard dependency checks
    "check_opc_ua_health",
    "check_kafka_health",
    "check_cmms_health",
    "check_database_health",
    "check_redis_health",
    "check_pi_server_health",
    # Resource checks
    "check_memory_health",
    "check_cpu_health",
    "check_disk_health",
]
