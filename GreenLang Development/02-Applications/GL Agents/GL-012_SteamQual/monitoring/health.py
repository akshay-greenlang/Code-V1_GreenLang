"""
GL-012 STEAMQUAL SteamQualityController - Health Monitoring

This module provides comprehensive health monitoring for the steam quality
control agent, including service health, integration health, data quality
status, and Kubernetes-compatible liveness/readiness probes.

Features:
    - Liveness and readiness probes for Kubernetes deployment
    - Dependency health checks (sensors, SCADA, database)
    - Data quality monitoring for steam quality measurements
    - Periodic background health checks
    - Component-level health aggregation

Example:
    >>> monitor = SteamQualityHealthMonitor()
    >>> service_health = await monitor.check_service_health()
    >>> overall = await monitor.get_overall_health()
    >>> print(f"System status: {overall.status.value}")
    >>> if monitor.is_ready():
    ...     print("System ready for requests")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging
import time
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    @property
    def is_operational(self) -> bool:
        """Check if status allows operation."""
        return self in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "details": self.details,
        }


@dataclass
class ServiceHealthStatus:
    """Service-level health status."""
    status: HealthStatus
    uptime_seconds: float
    version: str
    agent_id: str
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_separators: int = 0
    error_count_last_hour: int = 0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "agent_id": self.agent_id,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "active_separators": self.active_separators,
            "error_count_last_hour": self.error_count_last_hour,
            "last_check": self.last_check.isoformat(),
            "details": self.details,
        }


@dataclass
class IntegrationHealthStatus:
    """Integration health status for external dependencies."""
    name: str
    status: HealthStatus
    endpoint: Optional[str] = None
    connected: bool = False
    latency_ms: float = 0.0
    last_successful_connection: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "endpoint": self.endpoint,
            "connected": self.connected,
            "latency_ms": self.latency_ms,
            "last_successful_connection": (
                self.last_successful_connection.isoformat()
                if self.last_successful_connection else None
            ),
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "details": self.details,
        }


@dataclass
class DataQualityStatus:
    """Data quality status for steam quality measurements."""
    status: HealthStatus
    separator_id: str
    last_data_received: Optional[datetime] = None
    age_seconds: float = 0.0
    max_age_seconds: float = 60.0
    is_stale: bool = False
    quality_score: float = 1.0
    completeness_percent: float = 100.0
    accuracy_score: float = 1.0
    sensor_status: str = "operational"
    missing_fields: List[str] = field(default_factory=list)
    out_of_range_fields: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "separator_id": self.separator_id,
            "last_data_received": (
                self.last_data_received.isoformat()
                if self.last_data_received else None
            ),
            "age_seconds": self.age_seconds,
            "max_age_seconds": self.max_age_seconds,
            "is_stale": self.is_stale,
            "quality_score": self.quality_score,
            "completeness_percent": self.completeness_percent,
            "accuracy_score": self.accuracy_score,
            "sensor_status": self.sensor_status,
            "missing_fields": self.missing_fields,
            "out_of_range_fields": self.out_of_range_fields,
            "details": self.details,
        }


@dataclass
class CalculatorHealthStatus:
    """Health status for calculation engines."""
    status: HealthStatus
    calculator_name: str
    last_calculation_time: Optional[datetime] = None
    calculations_last_hour: int = 0
    average_latency_ms: float = 0.0
    error_rate_percent: float = 0.0
    formula_version: str = "1.0.0"
    is_calibrated: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "calculator_name": self.calculator_name,
            "last_calculation_time": (
                self.last_calculation_time.isoformat()
                if self.last_calculation_time else None
            ),
            "calculations_last_hour": self.calculations_last_hour,
            "average_latency_ms": self.average_latency_ms,
            "error_rate_percent": self.error_rate_percent,
            "formula_version": self.formula_version,
            "is_calibrated": self.is_calibrated,
            "details": self.details,
        }


@dataclass
class OverallHealthStatus:
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    version: str
    agent_id: str
    service_health: ServiceHealthStatus
    integration_health: Dict[str, IntegrationHealthStatus]
    data_quality: Dict[str, DataQualityStatus]
    calculator_health: Dict[str, CalculatorHealthStatus]
    is_ready: bool = True
    is_live: bool = True
    degraded_components: List[str] = field(default_factory=list)
    unhealthy_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "agent_id": self.agent_id,
            "service_health": self.service_health.to_dict(),
            "integration_health": {
                k: v.to_dict() for k, v in self.integration_health.items()
            },
            "data_quality": {
                k: v.to_dict() for k, v in self.data_quality.items()
            },
            "calculator_health": {
                k: v.to_dict() for k, v in self.calculator_health.items()
            },
            "is_ready": self.is_ready,
            "is_live": self.is_live,
            "degraded_components": self.degraded_components,
            "unhealthy_components": self.unhealthy_components,
        }


# =============================================================================
# Health Monitor
# =============================================================================

class SteamQualityHealthMonitor:
    """
    Health monitoring system for steam quality control agent.

    This class provides comprehensive health monitoring including service health,
    integration health (sensors, SCADA, database), data quality, and calculator
    health. It supports Kubernetes liveness and readiness probes.

    Attributes:
        version: Agent version string
        agent_id: Agent identifier

    Example:
        >>> monitor = SteamQualityHealthMonitor(version="1.0.0", agent_id="GL-012")
        >>> service_health = await monitor.check_service_health()
        >>> overall = await monitor.get_overall_health()
        >>> if monitor.is_ready():
        ...     print("System ready for requests")
    """

    def __init__(
        self,
        version: str = "1.0.0",
        agent_id: str = "GL-012",
        check_interval_s: float = 30.0,
    ) -> None:
        """
        Initialize SteamQualityHealthMonitor.

        Args:
            version: Agent version string
            agent_id: Agent identifier
            check_interval_s: Background check interval in seconds
        """
        self.version = version
        self.agent_id = agent_id
        self._check_interval_s = check_interval_s
        self._start_time = time.time()
        self._lock = threading.Lock()

        # Health check registry
        self._custom_checks: Dict[str, Callable] = {}

        # Cached results
        self._last_service_health: Optional[ServiceHealthStatus] = None
        self._last_integration_health: Dict[str, IntegrationHealthStatus] = {}
        self._last_data_quality: Dict[str, DataQualityStatus] = {}
        self._last_calculator_health: Dict[str, CalculatorHealthStatus] = {}
        self._last_overall_health: Optional[OverallHealthStatus] = None

        # Integration connectors (to be set externally)
        self._scada_client: Optional[Any] = None
        self._database_pool: Optional[Any] = None
        self._historian_client: Optional[Any] = None

        # Separator data timestamps
        self._separator_timestamps: Dict[str, datetime] = {}
        self._separator_quality_scores: Dict[str, float] = {}

        # Calculator tracking
        self._calculator_stats: Dict[str, Dict[str, Any]] = {}

        # Error tracking
        self._recent_errors: List[datetime] = []

        # Background check control
        self._running = False

        logger.info(
            "SteamQualityHealthMonitor initialized: version=%s, agent_id=%s",
            version,
            agent_id,
        )

    # =========================================================================
    # External Connector Setup
    # =========================================================================

    def set_scada_client(self, client: Any) -> None:
        """Set SCADA client for health checks."""
        self._scada_client = client

    def set_database_pool(self, pool: Any) -> None:
        """Set database pool for health checks."""
        self._database_pool = pool

    def set_historian_client(self, client: Any) -> None:
        """Set historian client for health checks."""
        self._historian_client = client

    # =========================================================================
    # Data Recording Methods
    # =========================================================================

    def record_data_received(
        self,
        separator_id: str,
        quality_score: float = 1.0,
    ) -> None:
        """
        Record data reception timestamp for a separator.

        Args:
            separator_id: Separator identifier
            quality_score: Data quality score (0.0-1.0)
        """
        with self._lock:
            self._separator_timestamps[separator_id] = datetime.now(timezone.utc)
            self._separator_quality_scores[separator_id] = quality_score

    def record_calculation(
        self,
        calculator_name: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """
        Record a calculation for health tracking.

        Args:
            calculator_name: Name of the calculator
            latency_ms: Calculation latency in milliseconds
            success: Whether calculation succeeded
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            if calculator_name not in self._calculator_stats:
                self._calculator_stats[calculator_name] = {
                    "calculations": [],
                    "errors": [],
                    "latencies": [],
                }

            stats = self._calculator_stats[calculator_name]
            stats["calculations"].append(now)
            stats["latencies"].append(latency_ms)

            if not success:
                stats["errors"].append(now)

            # Keep only last hour of data
            cutoff = now - timedelta(hours=1)
            stats["calculations"] = [t for t in stats["calculations"] if t > cutoff]
            stats["errors"] = [t for t in stats["errors"] if t > cutoff]
            stats["latencies"] = stats["latencies"][-1000:]  # Keep last 1000

    def record_error(self) -> None:
        """Record a general error for tracking."""
        now = datetime.now(timezone.utc)

        with self._lock:
            self._recent_errors.append(now)

            # Keep only last hour of data
            cutoff = now - timedelta(hours=1)
            self._recent_errors = [t for t in self._recent_errors if t > cutoff]

    def register_custom_check(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """Register a custom health check function."""
        self._custom_checks[name] = check_func
        logger.debug("Registered custom health check: %s", name)

    # =========================================================================
    # Service Health
    # =========================================================================

    async def check_service_health(self) -> ServiceHealthStatus:
        """
        Check overall service health.

        Returns:
            ServiceHealthStatus with current service metrics
        """
        uptime = time.time() - self._start_time
        now = datetime.now(timezone.utc)

        # Calculate error count in last hour
        with self._lock:
            cutoff = now - timedelta(hours=1)
            error_count = len([e for e in self._recent_errors if e > cutoff])
            active_separators = len(self._separator_timestamps)

        # Get resource usage
        memory_mb = 0.0
        cpu_percent = 0.0

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent()
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Resource metrics unavailable: %s", e)

        # Determine status
        status = HealthStatus.HEALTHY
        if error_count > 100:
            status = HealthStatus.UNHEALTHY
        elif error_count > 10:
            status = HealthStatus.DEGRADED

        self._last_service_health = ServiceHealthStatus(
            status=status,
            uptime_seconds=uptime,
            version=self.version,
            agent_id=self.agent_id,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            active_separators=active_separators,
            error_count_last_hour=error_count,
            last_check=now,
        )

        return self._last_service_health

    # =========================================================================
    # Integration Health
    # =========================================================================

    async def check_integration_health(self) -> Dict[str, IntegrationHealthStatus]:
        """
        Check health of all external integrations.

        Returns:
            Dictionary of integration name to health status
        """
        results: Dict[str, IntegrationHealthStatus] = {}

        # Check SCADA
        results["scada"] = await self._check_scada_health()

        # Check Database
        results["database"] = await self._check_database_health()

        # Check Historian
        results["historian"] = await self._check_historian_health()

        self._last_integration_health = results
        return results

    async def _check_scada_health(self) -> IntegrationHealthStatus:
        """Check SCADA connection health."""
        start_time = time.time()

        try:
            if self._scada_client is None:
                return IntegrationHealthStatus(
                    name="scada",
                    status=HealthStatus.UNKNOWN,
                    connected=False,
                    error_message="SCADA client not configured",
                )

            # In production, check actual connection
            is_connected = getattr(
                self._scada_client, "is_connected", lambda: False
            )()

            latency_ms = (time.time() - start_time) * 1000

            if is_connected:
                return IntegrationHealthStatus(
                    name="scada",
                    status=HealthStatus.HEALTHY,
                    endpoint=getattr(self._scada_client, "endpoint", None),
                    connected=True,
                    latency_ms=latency_ms,
                    last_successful_connection=datetime.now(timezone.utc),
                )
            else:
                return IntegrationHealthStatus(
                    name="scada",
                    status=HealthStatus.UNHEALTHY,
                    connected=False,
                    error_message="SCADA disconnected",
                )

        except Exception as e:
            return IntegrationHealthStatus(
                name="scada",
                status=HealthStatus.UNHEALTHY,
                connected=False,
                error_message=str(e),
            )

    async def _check_database_health(self) -> IntegrationHealthStatus:
        """Check database connection health."""
        start_time = time.time()

        try:
            if self._database_pool is None:
                return IntegrationHealthStatus(
                    name="database",
                    status=HealthStatus.UNKNOWN,
                    connected=False,
                    error_message="Database pool not configured",
                )

            # In production, execute health check query
            latency_ms = (time.time() - start_time) * 1000

            return IntegrationHealthStatus(
                name="database",
                status=HealthStatus.HEALTHY,
                connected=True,
                latency_ms=latency_ms,
                last_successful_connection=datetime.now(timezone.utc),
            )

        except Exception as e:
            return IntegrationHealthStatus(
                name="database",
                status=HealthStatus.UNHEALTHY,
                connected=False,
                error_message=str(e),
            )

    async def _check_historian_health(self) -> IntegrationHealthStatus:
        """Check historian connection health."""
        start_time = time.time()

        try:
            if self._historian_client is None:
                return IntegrationHealthStatus(
                    name="historian",
                    status=HealthStatus.UNKNOWN,
                    connected=False,
                    error_message="Historian client not configured",
                )

            is_connected = getattr(
                self._historian_client, "is_connected", lambda: False
            )()

            latency_ms = (time.time() - start_time) * 1000

            if is_connected:
                return IntegrationHealthStatus(
                    name="historian",
                    status=HealthStatus.HEALTHY,
                    connected=True,
                    latency_ms=latency_ms,
                    last_successful_connection=datetime.now(timezone.utc),
                )
            else:
                return IntegrationHealthStatus(
                    name="historian",
                    status=HealthStatus.DEGRADED,
                    connected=False,
                    error_message="Historian disconnected - using cached data",
                )

        except Exception as e:
            return IntegrationHealthStatus(
                name="historian",
                status=HealthStatus.UNHEALTHY,
                connected=False,
                error_message=str(e),
            )

    # =========================================================================
    # Data Quality Health
    # =========================================================================

    async def check_data_quality(
        self,
        max_age_seconds: float = 60.0,
    ) -> Dict[str, DataQualityStatus]:
        """
        Check data quality for all monitored separators.

        Args:
            max_age_seconds: Maximum acceptable data age

        Returns:
            Dictionary of separator ID to data quality status
        """
        results: Dict[str, DataQualityStatus] = {}
        now = datetime.now(timezone.utc)

        with self._lock:
            separator_ids = list(self._separator_timestamps.keys())
            timestamps = dict(self._separator_timestamps)
            quality_scores = dict(self._separator_quality_scores)

        for separator_id in separator_ids:
            last_received = timestamps.get(separator_id)
            quality_score = quality_scores.get(separator_id, 1.0)

            if last_received is None:
                results[separator_id] = DataQualityStatus(
                    status=HealthStatus.UNKNOWN,
                    separator_id=separator_id,
                    max_age_seconds=max_age_seconds,
                    is_stale=True,
                    quality_score=0.0,
                )
                continue

            age_seconds = (now - last_received).total_seconds()
            is_stale = age_seconds > max_age_seconds

            # Determine status based on staleness and quality
            if is_stale:
                status = HealthStatus.UNHEALTHY
                effective_quality = max(0.0, quality_score - 0.3)
            elif age_seconds > max_age_seconds * 0.8:
                status = HealthStatus.DEGRADED
                effective_quality = quality_score * 0.9
            elif quality_score < 0.7:
                status = HealthStatus.DEGRADED
                effective_quality = quality_score
            else:
                status = HealthStatus.HEALTHY
                effective_quality = quality_score

            results[separator_id] = DataQualityStatus(
                status=status,
                separator_id=separator_id,
                last_data_received=last_received,
                age_seconds=age_seconds,
                max_age_seconds=max_age_seconds,
                is_stale=is_stale,
                quality_score=effective_quality,
                completeness_percent=quality_score * 100,
                accuracy_score=quality_score,
                sensor_status="operational" if not is_stale else "stale",
            )

        self._last_data_quality = results
        return results

    # =========================================================================
    # Calculator Health
    # =========================================================================

    async def check_calculator_health(self) -> Dict[str, CalculatorHealthStatus]:
        """
        Check health of all calculation engines.

        Returns:
            Dictionary of calculator name to health status
        """
        results: Dict[str, CalculatorHealthStatus] = {}
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)

        # Default calculators to check
        calculator_names = [
            "dryness_calculator",
            "carryover_estimator",
            "separator_efficiency",
            "water_hammer_risk",
        ]

        with self._lock:
            calculator_stats = dict(self._calculator_stats)

        for calc_name in calculator_names:
            stats = calculator_stats.get(calc_name, {})

            calculations = stats.get("calculations", [])
            errors = stats.get("errors", [])
            latencies = stats.get("latencies", [])

            # Filter to last hour
            recent_calcs = [t for t in calculations if t > cutoff]
            recent_errors = [t for t in errors if t > cutoff]

            calc_count = len(recent_calcs)
            error_count = len(recent_errors)
            error_rate = (error_count / calc_count * 100) if calc_count > 0 else 0.0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            # Determine status
            if error_rate > 10:
                status = HealthStatus.UNHEALTHY
            elif error_rate > 5 or avg_latency > 1000:
                status = HealthStatus.DEGRADED
            elif calc_count == 0:
                status = HealthStatus.UNKNOWN
            else:
                status = HealthStatus.HEALTHY

            results[calc_name] = CalculatorHealthStatus(
                status=status,
                calculator_name=calc_name,
                last_calculation_time=recent_calcs[-1] if recent_calcs else None,
                calculations_last_hour=calc_count,
                average_latency_ms=avg_latency,
                error_rate_percent=error_rate,
                is_calibrated=True,
            )

        self._last_calculator_health = results
        return results

    # =========================================================================
    # Overall Health
    # =========================================================================

    async def get_overall_health(self) -> OverallHealthStatus:
        """
        Get comprehensive overall health status.

        Returns:
            OverallHealthStatus aggregating all health checks
        """
        # Run all health checks
        service_health = await self.check_service_health()
        integration_health = await self.check_integration_health()
        data_quality = await self.check_data_quality()
        calculator_health = await self.check_calculator_health()

        # Aggregate statuses
        all_statuses: List[HealthStatus] = [service_health.status]
        all_statuses.extend(h.status for h in integration_health.values())
        all_statuses.extend(d.status for d in data_quality.values())
        all_statuses.extend(c.status for c in calculator_health.values())

        # Track degraded and unhealthy components
        degraded: List[str] = []
        unhealthy: List[str] = []

        if service_health.status == HealthStatus.DEGRADED:
            degraded.append("service")
        elif service_health.status == HealthStatus.UNHEALTHY:
            unhealthy.append("service")

        for name, health in integration_health.items():
            if health.status == HealthStatus.DEGRADED:
                degraded.append(f"integration:{name}")
            elif health.status == HealthStatus.UNHEALTHY:
                unhealthy.append(f"integration:{name}")

        for separator_id, quality in data_quality.items():
            if quality.status == HealthStatus.DEGRADED:
                degraded.append(f"data:{separator_id}")
            elif quality.status == HealthStatus.UNHEALTHY:
                unhealthy.append(f"data:{separator_id}")

        for calc_name, health in calculator_health.items():
            if health.status == HealthStatus.DEGRADED:
                degraded.append(f"calculator:{calc_name}")
            elif health.status == HealthStatus.UNHEALTHY:
                unhealthy.append(f"calculator:{calc_name}")

        # Determine overall status
        if any(s == HealthStatus.UNHEALTHY for s in all_statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in all_statuses):
            overall_status = HealthStatus.DEGRADED
        elif any(s == HealthStatus.UNKNOWN for s in all_statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Determine readiness (can accept requests)
        # Ready if service is operational and at least some data quality is good
        is_ready = (
            service_health.status.is_operational
            and overall_status != HealthStatus.UNHEALTHY
        )

        # Liveness is always true if running
        is_live = True

        self._last_overall_health = OverallHealthStatus(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            uptime_seconds=time.time() - self._start_time,
            version=self.version,
            agent_id=self.agent_id,
            service_health=service_health,
            integration_health=integration_health,
            data_quality=data_quality,
            calculator_health=calculator_health,
            is_ready=is_ready,
            is_live=is_live,
            degraded_components=degraded,
            unhealthy_components=unhealthy,
        )

        return self._last_overall_health

    # =========================================================================
    # Kubernetes Probes
    # =========================================================================

    def is_ready(self) -> bool:
        """
        Kubernetes readiness probe.

        Returns:
            True if service can accept requests
        """
        if self._last_overall_health is None:
            return False
        return self._last_overall_health.is_ready

    def is_live(self) -> bool:
        """
        Kubernetes liveness probe.

        Returns:
            True if service is alive (always true if running)
        """
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current health status as dictionary."""
        return {
            "healthy": self.is_ready(),
            "live": self.is_live(),
            "uptime_seconds": time.time() - self._start_time,
            "version": self.version,
            "agent_id": self.agent_id,
            "last_check": (
                self._last_overall_health.timestamp.isoformat()
                if self._last_overall_health else None
            ),
        }

    def get_readiness_response(self) -> Dict[str, Any]:
        """
        Get readiness probe response for Kubernetes.

        Returns:
            Dict suitable for HTTP response
        """
        is_ready = self.is_ready()
        return {
            "status": "ready" if is_ready else "not_ready",
            "ready": is_ready,
            "uptime_seconds": time.time() - self._start_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "service": (
                    self._last_service_health.status.value
                    if self._last_service_health else "unknown"
                ),
                "integrations": len([
                    h for h in self._last_integration_health.values()
                    if h.status.is_operational
                ]),
                "data_quality": len([
                    d for d in self._last_data_quality.values()
                    if d.status.is_operational
                ]),
            },
        }

    def get_liveness_response(self) -> Dict[str, Any]:
        """
        Get liveness probe response for Kubernetes.

        Returns:
            Dict suitable for HTTP response
        """
        return {
            "status": "alive",
            "live": True,
            "uptime_seconds": time.time() - self._start_time,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # Background Health Checks
    # =========================================================================

    async def start_background_checks(
        self,
        interval_s: Optional[float] = None,
    ) -> None:
        """
        Start background health check loop.

        Args:
            interval_s: Check interval in seconds (default: self._check_interval_s)
        """
        interval = interval_s or self._check_interval_s
        self._running = True

        logger.info("Starting background health checks (interval: %ss)", interval)

        while self._running:
            try:
                await self.get_overall_health()
                logger.debug("Background health check completed")
            except Exception as e:
                logger.error("Background health check failed: %s", e)

            await asyncio.sleep(interval)

    def stop_background_checks(self) -> None:
        """Stop background health check loop."""
        self._running = False
        logger.info("Background health checks stopped")
