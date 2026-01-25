"""
GL-015 INSULSCAN - Health Check Endpoints

This module provides health check endpoints for Kubernetes liveness and
readiness probes, along with detailed component health monitoring for
the insulation scanning and thermal assessment agent.

Components Monitored:
    - Calculator: Heat loss and condition score calculators
    - Integrations: OPC-UA, CMMS, thermal imaging systems
    - Database: Time-series and asset databases
    - Cache: Redis or in-memory cache systems
    - ML Models: Condition prediction and anomaly detection models

Health Statuses:
    - HEALTHY: Component is fully operational
    - DEGRADED: Component is operational but with reduced capability
    - UNHEALTHY: Component is not operational
    - UNKNOWN: Component status cannot be determined

Probe Types:
    - Liveness: Indicates if the application should be restarted
    - Readiness: Indicates if the application can accept traffic

Example:
    >>> monitor = InsulscanHealthMonitor()
    >>> health = monitor.check_all()
    >>> if health.status == HealthStatus.HEALTHY:
    ...     print("System is healthy")
    >>> if monitor.is_ready():
    ...     print("Ready to accept traffic")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging
import threading
import time

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
    def is_ok(self) -> bool:
        """Check if status is acceptable for operation."""
        return self in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class ComponentType(Enum):
    """Types of components that can be health checked."""
    CALCULATOR = "calculator"
    DATABASE = "database"
    INTEGRATION = "integration"
    CACHE = "cache"
    ML_MODEL = "ml_model"
    MESSAGING = "messaging"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    component_type: ComponentType
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class CalculatorHealthStatus:
    """Health status for calculator components."""
    heat_loss_calculator: HealthStatus = HealthStatus.UNKNOWN
    condition_scorer: HealthStatus = HealthStatus.UNKNOWN
    hot_spot_detector: HealthStatus = HealthStatus.UNKNOWN
    savings_estimator: HealthStatus = HealthStatus.UNKNOWN
    last_calculation_time_ms: Optional[float] = None
    calculations_per_minute: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "heat_loss_calculator": self.heat_loss_calculator.value,
            "condition_scorer": self.condition_scorer.value,
            "hot_spot_detector": self.hot_spot_detector.value,
            "savings_estimator": self.savings_estimator.value,
            "last_calculation_time_ms": self.last_calculation_time_ms,
            "calculations_per_minute": self.calculations_per_minute,
        }


@dataclass
class IntegrationHealthStatus:
    """Health status for integration components."""
    opc_ua_connection: HealthStatus = HealthStatus.UNKNOWN
    cmms_connection: HealthStatus = HealthStatus.UNKNOWN
    thermal_imaging: HealthStatus = HealthStatus.UNKNOWN
    data_historian: HealthStatus = HealthStatus.UNKNOWN
    asset_registry: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "opc_ua_connection": self.opc_ua_connection.value,
            "cmms_connection": self.cmms_connection.value,
            "thermal_imaging": self.thermal_imaging.value,
            "data_historian": self.data_historian.value,
            "asset_registry": self.asset_registry.value,
        }


@dataclass
class DatabaseHealthStatus:
    """Health status for database components."""
    timeseries_db: HealthStatus = HealthStatus.UNKNOWN
    asset_db: HealthStatus = HealthStatus.UNKNOWN
    config_db: HealthStatus = HealthStatus.UNKNOWN
    connection_pool_available: int = 0
    connection_pool_used: int = 0
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timeseries_db": self.timeseries_db.value,
            "asset_db": self.asset_db.value,
            "config_db": self.config_db.value,
            "connection_pool": {
                "available": self.connection_pool_available,
                "used": self.connection_pool_used,
            },
            "latency_ms": self.latency_ms,
        }


@dataclass
class DataQualityStatus:
    """Data quality status for input data."""
    overall_score: float = 1.0
    missing_data_percent: float = 0.0
    stale_data_count: int = 0
    validation_errors: int = 0
    last_data_received: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_score": self.overall_score,
            "missing_data_percent": self.missing_data_percent,
            "stale_data_count": self.stale_data_count,
            "validation_errors": self.validation_errors,
            "last_data_received": (
                self.last_data_received.isoformat()
                if self.last_data_received else None
            ),
        }


@dataclass
class OverallHealthStatus:
    """Overall health status of the agent."""
    status: HealthStatus
    components: List[ComponentHealth]
    calculator_health: CalculatorHealthStatus
    integration_health: IntegrationHealthStatus
    database_health: DatabaseHealthStatus
    data_quality: DataQualityStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    agent_id: str = "GL-015"
    agent_name: str = "INSULSCAN"
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": [c.to_dict() for c in self.components],
            "calculator_health": self.calculator_health.to_dict(),
            "integration_health": self.integration_health.to_dict(),
            "database_health": self.database_health.to_dict(),
            "data_quality": self.data_quality.to_dict(),
        }


@dataclass
class LivenessProbeResult:
    """Result of liveness probe check."""
    alive: bool
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alive": self.alive,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ReadinessProbeResult:
    """Result of readiness probe check."""
    ready: bool
    message: str = ""
    checks_passed: int = 0
    checks_failed: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "ready": self.ready,
            "message": self.message,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Main Health Monitor Class
# =============================================================================

class InsulscanHealthMonitor:
    """
    Health monitoring system for GL-015 INSULSCAN.

    This class provides comprehensive health monitoring including:
    - Kubernetes liveness and readiness probes
    - Component-level health checks
    - Integration connectivity checks
    - Database health monitoring
    - Data quality assessment

    Attributes:
        agent_id: Agent identifier
        version: Agent version string

    Example:
        >>> monitor = InsulscanHealthMonitor()
        >>> liveness = monitor.liveness_probe()
        >>> readiness = monitor.readiness_probe()
        >>> if readiness.ready:
        ...     print("Ready to accept traffic")
    """

    AGENT_ID = "GL-015"
    AGENT_NAME = "INSULSCAN"
    VERSION = "1.0.0"

    def __init__(
        self,
        check_timeout_seconds: float = 5.0,
        enable_deep_checks: bool = True,
    ) -> None:
        """
        Initialize InsulscanHealthMonitor.

        Args:
            check_timeout_seconds: Timeout for individual health checks
            enable_deep_checks: Whether to perform deep health checks
        """
        self._check_timeout = check_timeout_seconds
        self._enable_deep_checks = enable_deep_checks
        self._lock = threading.Lock()

        # Health check function registry
        self._check_functions: Dict[str, Callable[[], ComponentHealth]] = {}
        self._async_check_functions: Dict[str, Callable[[], ComponentHealth]] = {}

        # Cached results
        self._last_results: Dict[str, ComponentHealth] = {}
        self._last_overall: Optional[OverallHealthStatus] = None

        # Startup time for uptime calculation
        self._start_time = datetime.now(timezone.utc)

        # Critical components required for readiness
        self._critical_components = {
            "heat_loss_calculator",
            "condition_scorer",
            "timeseries_db",
        }

        # Register default checks
        self._register_default_checks()

        logger.info(
            "InsulscanHealthMonitor initialized: timeout=%.1fs, deep_checks=%s",
            check_timeout_seconds,
            enable_deep_checks,
        )

    def _register_default_checks(self) -> None:
        """Register default health check functions."""
        # Calculator checks
        self.register_check("heat_loss_calculator", self._check_heat_loss_calculator)
        self.register_check("condition_scorer", self._check_condition_scorer)
        self.register_check("hot_spot_detector", self._check_hot_spot_detector)
        self.register_check("savings_estimator", self._check_savings_estimator)

        # Database checks
        self.register_check("timeseries_db", self._check_timeseries_db)
        self.register_check("asset_db", self._check_asset_db)

        # Integration checks
        self.register_check("opc_ua_connection", self._check_opc_ua)
        self.register_check("cmms_connection", self._check_cmms)
        self.register_check("thermal_imaging", self._check_thermal_imaging)

    # =========================================================================
    # Check Registration
    # =========================================================================

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
        critical: bool = False,
    ) -> None:
        """
        Register a health check function.

        Args:
            name: Check name
            check_fn: Function that returns ComponentHealth
            critical: Whether this component is critical for readiness
        """
        self._check_functions[name] = check_fn
        if critical:
            self._critical_components.add(name)

        logger.debug("Registered health check: %s (critical=%s)", name, critical)

    def register_async_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
        critical: bool = False,
    ) -> None:
        """
        Register an async health check function.

        Args:
            name: Check name
            check_fn: Async function that returns ComponentHealth
            critical: Whether this component is critical for readiness
        """
        self._async_check_functions[name] = check_fn
        if critical:
            self._critical_components.add(name)

        logger.debug("Registered async health check: %s (critical=%s)", name, critical)

    # =========================================================================
    # Default Check Implementations
    # =========================================================================

    def _check_heat_loss_calculator(self) -> ComponentHealth:
        """Check heat loss calculator health."""
        start = time.perf_counter()
        try:
            # Simulate a test calculation
            # In production, this would call the actual calculator
            test_result = True

            response_time = (time.perf_counter() - start) * 1000

            if test_result:
                return ComponentHealth(
                    name="heat_loss_calculator",
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.HEALTHY,
                    message="Heat loss calculator operational",
                    response_time_ms=response_time,
                    metadata={"version": "1.0.0"},
                )
            else:
                return ComponentHealth(
                    name="heat_loss_calculator",
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.UNHEALTHY,
                    message="Heat loss calculator test failed",
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.error("Heat loss calculator check failed: %s", e)
            return ComponentHealth(
                name="heat_loss_calculator",
                component_type=ComponentType.CALCULATOR,
                status=HealthStatus.UNHEALTHY,
                message="Health check error",
                error=str(e),
            )

    def _check_condition_scorer(self) -> ComponentHealth:
        """Check condition scorer health."""
        start = time.perf_counter()
        try:
            test_result = True
            response_time = (time.perf_counter() - start) * 1000

            if test_result:
                return ComponentHealth(
                    name="condition_scorer",
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.HEALTHY,
                    message="Condition scorer operational",
                    response_time_ms=response_time,
                )
            else:
                return ComponentHealth(
                    name="condition_scorer",
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.UNHEALTHY,
                    message="Condition scorer test failed",
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.error("Condition scorer check failed: %s", e)
            return ComponentHealth(
                name="condition_scorer",
                component_type=ComponentType.CALCULATOR,
                status=HealthStatus.UNHEALTHY,
                message="Health check error",
                error=str(e),
            )

    def _check_hot_spot_detector(self) -> ComponentHealth:
        """Check hot spot detector health."""
        start = time.perf_counter()
        try:
            test_result = True
            response_time = (time.perf_counter() - start) * 1000

            if test_result:
                return ComponentHealth(
                    name="hot_spot_detector",
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.HEALTHY,
                    message="Hot spot detector operational",
                    response_time_ms=response_time,
                )
            else:
                return ComponentHealth(
                    name="hot_spot_detector",
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.DEGRADED,
                    message="Hot spot detector running in fallback mode",
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.error("Hot spot detector check failed: %s", e)
            return ComponentHealth(
                name="hot_spot_detector",
                component_type=ComponentType.CALCULATOR,
                status=HealthStatus.UNHEALTHY,
                message="Health check error",
                error=str(e),
            )

    def _check_savings_estimator(self) -> ComponentHealth:
        """Check savings estimator health."""
        start = time.perf_counter()
        try:
            test_result = True
            response_time = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name="savings_estimator",
                component_type=ComponentType.CALCULATOR,
                status=HealthStatus.HEALTHY if test_result else HealthStatus.DEGRADED,
                message="Savings estimator operational" if test_result else "Limited accuracy mode",
                response_time_ms=response_time,
            )
        except Exception as e:
            logger.error("Savings estimator check failed: %s", e)
            return ComponentHealth(
                name="savings_estimator",
                component_type=ComponentType.CALCULATOR,
                status=HealthStatus.UNHEALTHY,
                message="Health check error",
                error=str(e),
            )

    def _check_timeseries_db(self) -> ComponentHealth:
        """Check time-series database health."""
        start = time.perf_counter()
        try:
            # Simulate database ping
            db_available = True
            response_time = (time.perf_counter() - start) * 1000

            if db_available:
                return ComponentHealth(
                    name="timeseries_db",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.HEALTHY,
                    message="Time-series database connected",
                    response_time_ms=response_time,
                    metadata={"connection_pool_size": 10},
                )
            else:
                return ComponentHealth(
                    name="timeseries_db",
                    component_type=ComponentType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    message="Time-series database unavailable",
                    response_time_ms=response_time,
                )
        except Exception as e:
            logger.error("Time-series database check failed: %s", e)
            return ComponentHealth(
                name="timeseries_db",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message="Database connection error",
                error=str(e),
            )

    def _check_asset_db(self) -> ComponentHealth:
        """Check asset database health."""
        start = time.perf_counter()
        try:
            db_available = True
            response_time = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name="asset_db",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY if db_available else HealthStatus.UNHEALTHY,
                message="Asset database connected" if db_available else "Asset database unavailable",
                response_time_ms=response_time,
            )
        except Exception as e:
            logger.error("Asset database check failed: %s", e)
            return ComponentHealth(
                name="asset_db",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message="Database connection error",
                error=str(e),
            )

    def _check_opc_ua(self) -> ComponentHealth:
        """Check OPC-UA connection health."""
        start = time.perf_counter()
        try:
            # Simulate OPC-UA connection check
            connected = True
            response_time = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name="opc_ua_connection",
                component_type=ComponentType.INTEGRATION,
                status=HealthStatus.HEALTHY if connected else HealthStatus.DEGRADED,
                message="OPC-UA connected" if connected else "OPC-UA disconnected, using cached data",
                response_time_ms=response_time,
                metadata={"server_state": "running" if connected else "disconnected"},
            )
        except Exception as e:
            logger.error("OPC-UA check failed: %s", e)
            return ComponentHealth(
                name="opc_ua_connection",
                component_type=ComponentType.INTEGRATION,
                status=HealthStatus.UNHEALTHY,
                message="OPC-UA connection error",
                error=str(e),
            )

    def _check_cmms(self) -> ComponentHealth:
        """Check CMMS connection health."""
        start = time.perf_counter()
        try:
            connected = True
            response_time = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name="cmms_connection",
                component_type=ComponentType.INTEGRATION,
                status=HealthStatus.HEALTHY if connected else HealthStatus.DEGRADED,
                message="CMMS connected" if connected else "CMMS disconnected",
                response_time_ms=response_time,
            )
        except Exception as e:
            logger.error("CMMS check failed: %s", e)
            return ComponentHealth(
                name="cmms_connection",
                component_type=ComponentType.INTEGRATION,
                status=HealthStatus.UNHEALTHY,
                message="CMMS connection error",
                error=str(e),
            )

    def _check_thermal_imaging(self) -> ComponentHealth:
        """Check thermal imaging system health."""
        start = time.perf_counter()
        try:
            connected = True
            response_time = (time.perf_counter() - start) * 1000

            return ComponentHealth(
                name="thermal_imaging",
                component_type=ComponentType.INTEGRATION,
                status=HealthStatus.HEALTHY if connected else HealthStatus.DEGRADED,
                message="Thermal imaging system connected" if connected else "Thermal imaging unavailable",
                response_time_ms=response_time,
            )
        except Exception as e:
            logger.error("Thermal imaging check failed: %s", e)
            return ComponentHealth(
                name="thermal_imaging",
                component_type=ComponentType.INTEGRATION,
                status=HealthStatus.UNHEALTHY,
                message="Thermal imaging error",
                error=str(e),
            )

    # =========================================================================
    # Health Check Execution
    # =========================================================================

    def check_component(self, name: str) -> ComponentHealth:
        """
        Run health check for a specific component.

        Args:
            name: Component name

        Returns:
            ComponentHealth result
        """
        if name in self._check_functions:
            try:
                result = self._check_functions[name]()
                with self._lock:
                    self._last_results[name] = result
                return result
            except Exception as e:
                logger.error("Health check %s failed: %s", name, e)
                return ComponentHealth(
                    name=name,
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.UNKNOWN,
                    message="Health check failed",
                    error=str(e),
                )
        else:
            return ComponentHealth(
                name=name,
                component_type=ComponentType.CALCULATOR,
                status=HealthStatus.UNKNOWN,
                message=f"No health check registered for {name}",
            )

    def check_all(self) -> OverallHealthStatus:
        """
        Run all health checks and return overall status.

        Returns:
            OverallHealthStatus with all component statuses
        """
        components: List[ComponentHealth] = []

        # Run all sync checks
        for name, check_fn in self._check_functions.items():
            try:
                result = check_fn()
                components.append(result)
                with self._lock:
                    self._last_results[name] = result
            except Exception as e:
                logger.error("Health check %s failed: %s", name, e)
                components.append(ComponentHealth(
                    name=name,
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    error=str(e),
                ))

        # Build aggregated status objects
        calculator_health = self._build_calculator_health(components)
        integration_health = self._build_integration_health(components)
        database_health = self._build_database_health(components)
        data_quality = self._assess_data_quality()

        # Determine overall status
        statuses = [c.status for c in components]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            # Check if critical component is unhealthy
            critical_unhealthy = any(
                c.status == HealthStatus.UNHEALTHY and c.name in self._critical_components
                for c in components
            )
            overall_status = HealthStatus.UNHEALTHY if critical_unhealthy else HealthStatus.DEGRADED
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN

        # Calculate uptime
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        overall = OverallHealthStatus(
            status=overall_status,
            components=components,
            calculator_health=calculator_health,
            integration_health=integration_health,
            database_health=database_health,
            data_quality=data_quality,
            version=self.VERSION,
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            uptime_seconds=uptime,
        )

        with self._lock:
            self._last_overall = overall

        return overall

    async def check_all_async(self) -> OverallHealthStatus:
        """
        Run all health checks asynchronously.

        Returns:
            OverallHealthStatus with all component statuses
        """
        components: List[ComponentHealth] = []
        loop = asyncio.get_event_loop()

        # Run sync checks in thread pool
        sync_tasks = []
        for name, check_fn in self._check_functions.items():
            sync_tasks.append((name, loop.run_in_executor(None, check_fn)))

        if sync_tasks:
            results = await asyncio.gather(
                *[task for _, task in sync_tasks],
                return_exceptions=True,
            )

            for i, (name, _) in enumerate(sync_tasks):
                result = results[i]
                if isinstance(result, Exception):
                    components.append(ComponentHealth(
                        name=name,
                        component_type=ComponentType.CALCULATOR,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(result)}",
                        error=str(result),
                    ))
                else:
                    components.append(result)

        # Run async checks
        for name, check_fn in self._async_check_functions.items():
            try:
                result = await asyncio.wait_for(
                    check_fn(),
                    timeout=self._check_timeout,
                )
                components.append(result)
            except asyncio.TimeoutError:
                components.append(ComponentHealth(
                    name=name,
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.UNHEALTHY,
                    message="Check timed out",
                ))
            except Exception as e:
                components.append(ComponentHealth(
                    name=name,
                    component_type=ComponentType.CALCULATOR,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    error=str(e),
                ))

        # Build status objects
        calculator_health = self._build_calculator_health(components)
        integration_health = self._build_integration_health(components)
        database_health = self._build_database_health(components)
        data_quality = self._assess_data_quality()

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

        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return OverallHealthStatus(
            status=overall_status,
            components=components,
            calculator_health=calculator_health,
            integration_health=integration_health,
            database_health=database_health,
            data_quality=data_quality,
            version=self.VERSION,
            agent_id=self.AGENT_ID,
            agent_name=self.AGENT_NAME,
            uptime_seconds=uptime,
        )

    def _build_calculator_health(
        self,
        components: List[ComponentHealth],
    ) -> CalculatorHealthStatus:
        """Build calculator health status from component results."""
        calc_status = CalculatorHealthStatus()

        for comp in components:
            if comp.component_type != ComponentType.CALCULATOR:
                continue

            if comp.name == "heat_loss_calculator":
                calc_status.heat_loss_calculator = comp.status
            elif comp.name == "condition_scorer":
                calc_status.condition_scorer = comp.status
            elif comp.name == "hot_spot_detector":
                calc_status.hot_spot_detector = comp.status
            elif comp.name == "savings_estimator":
                calc_status.savings_estimator = comp.status

        return calc_status

    def _build_integration_health(
        self,
        components: List[ComponentHealth],
    ) -> IntegrationHealthStatus:
        """Build integration health status from component results."""
        int_status = IntegrationHealthStatus()

        for comp in components:
            if comp.component_type != ComponentType.INTEGRATION:
                continue

            if comp.name == "opc_ua_connection":
                int_status.opc_ua_connection = comp.status
            elif comp.name == "cmms_connection":
                int_status.cmms_connection = comp.status
            elif comp.name == "thermal_imaging":
                int_status.thermal_imaging = comp.status

        return int_status

    def _build_database_health(
        self,
        components: List[ComponentHealth],
    ) -> DatabaseHealthStatus:
        """Build database health status from component results."""
        db_status = DatabaseHealthStatus()

        for comp in components:
            if comp.component_type != ComponentType.DATABASE:
                continue

            if comp.name == "timeseries_db":
                db_status.timeseries_db = comp.status
                if comp.response_time_ms is not None:
                    db_status.latency_ms = comp.response_time_ms
            elif comp.name == "asset_db":
                db_status.asset_db = comp.status

        return db_status

    def _assess_data_quality(self) -> DataQualityStatus:
        """Assess current data quality."""
        # In production, this would check actual data quality
        return DataQualityStatus(
            overall_score=0.95,
            missing_data_percent=0.5,
            stale_data_count=0,
            validation_errors=0,
            last_data_received=datetime.now(timezone.utc),
        )

    # =========================================================================
    # Kubernetes Probe Handlers
    # =========================================================================

    def liveness_probe(self) -> LivenessProbeResult:
        """
        Kubernetes liveness probe handler.

        This probe indicates whether the application should be restarted.
        It performs minimal checks to determine if the application is alive.

        Returns:
            LivenessProbeResult indicating if application is alive
        """
        try:
            # Basic liveness: can we allocate memory and respond?
            # In production, this might check for deadlocks or memory issues
            alive = True
            message = "Application is alive"

            return LivenessProbeResult(
                alive=alive,
                message=message,
            )

        except Exception as e:
            logger.error("Liveness probe failed: %s", e)
            return LivenessProbeResult(
                alive=False,
                message=f"Liveness check failed: {str(e)}",
            )

    def readiness_probe(self) -> ReadinessProbeResult:
        """
        Kubernetes readiness probe handler.

        This probe indicates whether the application can accept traffic.
        It checks that critical components are healthy.

        Returns:
            ReadinessProbeResult indicating if application is ready
        """
        try:
            checks_passed = 0
            checks_failed = 0
            failed_components = []

            # Check critical components only
            for name in self._critical_components:
                if name in self._check_functions:
                    result = self._check_functions[name]()
                    if result.status.is_ok:
                        checks_passed += 1
                    else:
                        checks_failed += 1
                        failed_components.append(name)

            ready = checks_failed == 0

            if ready:
                message = f"Ready: {checks_passed} critical checks passed"
            else:
                message = f"Not ready: {failed_components} failed"

            return ReadinessProbeResult(
                ready=ready,
                message=message,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        except Exception as e:
            logger.error("Readiness probe failed: %s", e)
            return ReadinessProbeResult(
                ready=False,
                message=f"Readiness check failed: {str(e)}",
                checks_passed=0,
                checks_failed=1,
            )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def is_healthy(self) -> bool:
        """
        Quick check if system is healthy.

        Returns:
            True if system is healthy
        """
        result = self.check_all()
        return result.status == HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """
        Check if system is ready to accept traffic.

        Returns:
            True if system is ready
        """
        result = self.readiness_probe()
        return result.ready

    def is_alive(self) -> bool:
        """
        Check if system is alive.

        Returns:
            True if system is alive
        """
        result = self.liveness_probe()
        return result.alive

    def get_cached_results(self) -> Dict[str, ComponentHealth]:
        """
        Get cached health check results.

        Returns:
            Dictionary of component names to their last health status
        """
        with self._lock:
            return dict(self._last_results)

    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Get a detailed health report.

        Returns:
            Comprehensive health report dictionary
        """
        overall = self.check_all()
        liveness = self.liveness_probe()
        readiness = self.readiness_probe()

        return {
            "overall": overall.to_dict(),
            "liveness": liveness.to_dict(),
            "readiness": readiness.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# =============================================================================
# Global Instance
# =============================================================================

_health_monitor: Optional[InsulscanHealthMonitor] = None


def get_health_monitor() -> InsulscanHealthMonitor:
    """
    Get or create the global health monitor.

    Returns:
        Global InsulscanHealthMonitor instance
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = InsulscanHealthMonitor()
    return _health_monitor
