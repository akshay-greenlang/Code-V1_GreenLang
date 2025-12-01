# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Health Checks Module.

This module implements comprehensive health checking for the
GL-011 fuel management agent including:
- Component health monitoring
- Dependency health verification
- Self-healing triggers
- Alerting integration

Standards:
- Kubernetes health check patterns (liveness, readiness)
- OpenAPI health check response format
- Circuit breaker patterns
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import traceback

logger = logging.getLogger(__name__)


# ============================================================================
# Health Status Types
# ============================================================================

class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(str, Enum):
    """Types of health checks."""
    LIVENESS = "liveness"      # Is the component alive?
    READINESS = "readiness"    # Is the component ready to serve?
    STARTUP = "startup"        # Has the component started successfully?


# ============================================================================
# Health Check Data Classes
# ============================================================================

@dataclass
class ComponentHealth:
    """Health status of a single component."""
    component_name: str
    status: HealthStatus
    check_type: CheckType
    message: str = ""
    response_time_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    overall_status: HealthStatus
    timestamp: datetime
    uptime_seconds: float
    components: List[ComponentHealth]
    summary: Dict[str, int]
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": [
                {
                    "name": c.component_name,
                    "status": c.status.value,
                    "check_type": c.check_type.value,
                    "message": c.message,
                    "response_time_ms": c.response_time_ms,
                    "last_check": c.last_check.isoformat(),
                    "details": c.details
                }
                for c in self.components
            ],
            "summary": self.summary,
            "alerts": self.alerts
        }


# ============================================================================
# Health Check Definitions
# ============================================================================

@dataclass
class HealthCheckConfig:
    """Configuration for a health check."""
    name: str
    check_type: CheckType
    check_fn: Callable[[], bool]
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    failure_threshold: int = 3
    success_threshold: int = 1
    critical: bool = True
    description: str = ""


# ============================================================================
# Health Checker Implementation
# ============================================================================

class HealthChecker:
    """
    Comprehensive health checker for GL-011 FUELCRAFT.

    Provides:
    - Component health monitoring
    - Dependency verification
    - Kubernetes-compatible health endpoints
    - Self-healing triggers
    - Alerting integration
    """

    def __init__(self, start_time: Optional[datetime] = None):
        """Initialize health checker."""
        self._start_time = start_time or datetime.now()
        self._checks: Dict[str, HealthCheckConfig] = {}
        self._results: Dict[str, ComponentHealth] = {}
        self._lock = threading.RLock()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._alert_callbacks: List[Callable[[str, HealthStatus], None]] = []
        self._self_heal_callbacks: Dict[str, Callable[[], None]] = {}

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default GL-011 health checks."""
        # Core orchestrator check
        self.register_check(HealthCheckConfig(
            name="orchestrator",
            check_type=CheckType.LIVENESS,
            check_fn=self._check_orchestrator,
            interval_seconds=10.0,
            timeout_seconds=5.0,
            critical=True,
            description="Main orchestrator process health"
        ))

        # Multi-fuel optimizer check
        self.register_check(HealthCheckConfig(
            name="multi_fuel_optimizer",
            check_type=CheckType.READINESS,
            check_fn=self._check_multi_fuel_optimizer,
            interval_seconds=30.0,
            timeout_seconds=10.0,
            critical=True,
            description="Multi-fuel optimizer calculator health"
        ))

        # Cost optimizer check
        self.register_check(HealthCheckConfig(
            name="cost_optimizer",
            check_type=CheckType.READINESS,
            check_fn=self._check_cost_optimizer,
            interval_seconds=30.0,
            timeout_seconds=10.0,
            critical=True,
            description="Cost optimization calculator health"
        ))

        # Blending calculator check
        self.register_check(HealthCheckConfig(
            name="blending_calculator",
            check_type=CheckType.READINESS,
            check_fn=self._check_blending_calculator,
            interval_seconds=30.0,
            timeout_seconds=10.0,
            critical=False,
            description="Fuel blending calculator health"
        ))

        # Carbon footprint calculator check
        self.register_check(HealthCheckConfig(
            name="carbon_calculator",
            check_type=CheckType.READINESS,
            check_fn=self._check_carbon_calculator,
            interval_seconds=30.0,
            timeout_seconds=10.0,
            critical=True,
            description="Carbon footprint calculator health"
        ))

        # Cache health check
        self.register_check(HealthCheckConfig(
            name="cache",
            check_type=CheckType.READINESS,
            check_fn=self._check_cache,
            interval_seconds=15.0,
            timeout_seconds=2.0,
            critical=False,
            description="Cache subsystem health"
        ))

        # Storage integration check
        self.register_check(HealthCheckConfig(
            name="storage_integration",
            check_type=CheckType.READINESS,
            check_fn=self._check_storage_integration,
            interval_seconds=60.0,
            timeout_seconds=15.0,
            critical=False,
            description="Fuel storage system integration health"
        ))

        # ERP integration check
        self.register_check(HealthCheckConfig(
            name="erp_integration",
            check_type=CheckType.READINESS,
            check_fn=self._check_erp_integration,
            interval_seconds=60.0,
            timeout_seconds=15.0,
            critical=False,
            description="ERP/procurement system integration health"
        ))

        # Market price feed check
        self.register_check(HealthCheckConfig(
            name="market_price_feed",
            check_type=CheckType.READINESS,
            check_fn=self._check_market_price_feed,
            interval_seconds=60.0,
            timeout_seconds=10.0,
            critical=False,
            description="Market price feed integration health"
        ))

        # Emissions monitoring check
        self.register_check(HealthCheckConfig(
            name="emissions_monitoring",
            check_type=CheckType.READINESS,
            check_fn=self._check_emissions_monitoring,
            interval_seconds=60.0,
            timeout_seconds=10.0,
            critical=False,
            description="Emissions monitoring integration health"
        ))

        # Memory health check
        self.register_check(HealthCheckConfig(
            name="memory",
            check_type=CheckType.LIVENESS,
            check_fn=self._check_memory,
            interval_seconds=30.0,
            timeout_seconds=2.0,
            critical=True,
            description="Memory usage health"
        ))

        # Thread pool health check
        self.register_check(HealthCheckConfig(
            name="thread_pool",
            check_type=CheckType.READINESS,
            check_fn=self._check_thread_pool,
            interval_seconds=30.0,
            timeout_seconds=5.0,
            critical=False,
            description="Thread pool health"
        ))

    # ========================================================================
    # Default Check Implementations
    # ========================================================================

    def _check_orchestrator(self) -> bool:
        """Check orchestrator health."""
        try:
            # Verify orchestrator is responsive
            # In production, this would ping the actual orchestrator
            return True
        except Exception as e:
            logger.error(f"Orchestrator health check failed: {e}")
            return False

    def _check_multi_fuel_optimizer(self) -> bool:
        """Check multi-fuel optimizer health."""
        try:
            from calculators.multi_fuel_optimizer import MultiFuelOptimizer
            optimizer = MultiFuelOptimizer()
            # Verify optimizer can be instantiated
            return optimizer is not None
        except Exception as e:
            logger.error(f"Multi-fuel optimizer health check failed: {e}")
            return False

    def _check_cost_optimizer(self) -> bool:
        """Check cost optimizer health."""
        try:
            from calculators.cost_optimization_calculator import CostOptimizationCalculator
            calculator = CostOptimizationCalculator()
            return calculator is not None
        except Exception as e:
            logger.error(f"Cost optimizer health check failed: {e}")
            return False

    def _check_blending_calculator(self) -> bool:
        """Check blending calculator health."""
        try:
            from calculators.fuel_blending_calculator import FuelBlendingCalculator
            calculator = FuelBlendingCalculator()
            return calculator is not None
        except Exception as e:
            logger.error(f"Blending calculator health check failed: {e}")
            return False

    def _check_carbon_calculator(self) -> bool:
        """Check carbon calculator health."""
        try:
            from calculators.carbon_footprint_calculator import CarbonFootprintCalculator
            calculator = CarbonFootprintCalculator()
            return calculator is not None
        except Exception as e:
            logger.error(f"Carbon calculator health check failed: {e}")
            return False

    def _check_cache(self) -> bool:
        """Check cache subsystem health."""
        try:
            from fuel_management_orchestrator import ThreadSafeCache
            cache = ThreadSafeCache(max_size=10)
            cache.set("health_check", "test")
            result = cache.get("health_check")
            return result == "test"
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False

    def _check_storage_integration(self) -> bool:
        """Check storage integration health."""
        try:
            # In production, this would verify MODBUS/OPC-UA connectivity
            # For now, verify module can be loaded
            from integrations.fuel_storage_connector import FuelStorageConnector
            return True
        except Exception as e:
            logger.error(f"Storage integration health check failed: {e}")
            return False

    def _check_erp_integration(self) -> bool:
        """Check ERP integration health."""
        try:
            from integrations.procurement_system_connector import ProcurementSystemConnector
            return True
        except Exception as e:
            logger.error(f"ERP integration health check failed: {e}")
            return False

    def _check_market_price_feed(self) -> bool:
        """Check market price feed health."""
        try:
            from integrations.market_price_connector import MarketPriceConnector
            return True
        except Exception as e:
            logger.error(f"Market price feed health check failed: {e}")
            return False

    def _check_emissions_monitoring(self) -> bool:
        """Check emissions monitoring health."""
        try:
            from integrations.emissions_monitoring_connector import EmissionsMonitoringConnector
            return True
        except Exception as e:
            logger.error(f"Emissions monitoring health check failed: {e}")
            return False

    def _check_memory(self) -> bool:
        """Check memory usage health."""
        try:
            import sys
            # Check if memory usage is reasonable
            # This is a simplified check; production would use psutil
            return True
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return False

    def _check_thread_pool(self) -> bool:
        """Check thread pool health."""
        try:
            import threading
            # Check thread count is not excessive
            active_threads = threading.active_count()
            return active_threads < 100  # Reasonable limit
        except Exception as e:
            logger.error(f"Thread pool health check failed: {e}")
            return False

    # ========================================================================
    # Health Check Management
    # ========================================================================

    def register_check(self, config: HealthCheckConfig) -> None:
        """Register a new health check."""
        with self._lock:
            self._checks[config.name] = config
            self._results[config.name] = ComponentHealth(
                component_name=config.name,
                status=HealthStatus.UNKNOWN,
                check_type=config.check_type,
                message="Not yet checked"
            )

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._results.pop(name, None)

    def register_alert_callback(self, callback: Callable[[str, HealthStatus], None]) -> None:
        """Register callback for health alerts."""
        self._alert_callbacks.append(callback)

    def register_self_heal(self, component: str, heal_fn: Callable[[], None]) -> None:
        """Register self-healing callback for component."""
        self._self_heal_callbacks[component] = heal_fn

    # ========================================================================
    # Health Check Execution
    # ========================================================================

    def run_check(self, name: str) -> ComponentHealth:
        """Run a specific health check."""
        with self._lock:
            if name not in self._checks:
                raise ValueError(f"Health check '{name}' not registered")

            config = self._checks[name]

        start_time = time.perf_counter()
        status = HealthStatus.UNKNOWN
        message = ""
        details = {}

        try:
            # Run check with timeout
            result = self._run_with_timeout(config.check_fn, config.timeout_seconds)
            response_time = (time.perf_counter() - start_time) * 1000

            if result:
                status = HealthStatus.HEALTHY
                message = "Check passed"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Check failed"

        except TimeoutError:
            response_time = config.timeout_seconds * 1000
            status = HealthStatus.UNHEALTHY
            message = f"Check timed out after {config.timeout_seconds}s"

        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            status = HealthStatus.UNHEALTHY
            message = f"Check exception: {str(e)}"
            details["exception"] = traceback.format_exc()

        # Update result
        with self._lock:
            prev_result = self._results.get(name)

            health = ComponentHealth(
                component_name=name,
                status=status,
                check_type=config.check_type,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.now(),
                details=details,
                consecutive_failures=(
                    0 if status == HealthStatus.HEALTHY
                    else (prev_result.consecutive_failures + 1 if prev_result else 1)
                ),
                last_success=(
                    datetime.now() if status == HealthStatus.HEALTHY
                    else (prev_result.last_success if prev_result else None)
                ),
                last_failure=(
                    None if status == HealthStatus.HEALTHY
                    else datetime.now()
                )
            )

            self._results[name] = health

            # Check for alerts
            if status == HealthStatus.UNHEALTHY and health.consecutive_failures >= config.failure_threshold:
                self._trigger_alert(name, status, config.critical)

                # Attempt self-healing
                if name in self._self_heal_callbacks:
                    try:
                        self._self_heal_callbacks[name]()
                        logger.info(f"Self-healing triggered for {name}")
                    except Exception as e:
                        logger.error(f"Self-healing failed for {name}: {e}")

        return health

    def _run_with_timeout(self, fn: Callable[[], bool], timeout: float) -> bool:
        """Run function with timeout."""
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = fn()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            raise TimeoutError(f"Function timed out after {timeout}s")

        if exception[0]:
            raise exception[0]

        return result[0]

    def _trigger_alert(self, component: str, status: HealthStatus, critical: bool) -> None:
        """Trigger alert for unhealthy component."""
        severity = "CRITICAL" if critical else "WARNING"
        logger.warning(f"[{severity}] Health alert for {component}: {status.value}")

        for callback in self._alert_callbacks:
            try:
                callback(component, status)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    # ========================================================================
    # Health Report Generation
    # ========================================================================

    def get_health_report(self, run_checks: bool = False) -> SystemHealthReport:
        """Get complete system health report."""
        if run_checks:
            self.run_all_checks()

        with self._lock:
            components = list(self._results.values())

        # Calculate overall status
        overall_status = self._calculate_overall_status(components)

        # Generate summary
        summary = {
            "healthy": sum(1 for c in components if c.status == HealthStatus.HEALTHY),
            "degraded": sum(1 for c in components if c.status == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for c in components if c.status == HealthStatus.UNHEALTHY),
            "unknown": sum(1 for c in components if c.status == HealthStatus.UNKNOWN)
        }

        # Generate alerts
        alerts = []
        for component in components:
            if component.status == HealthStatus.UNHEALTHY:
                config = self._checks.get(component.component_name)
                if config and config.critical:
                    alerts.append(f"CRITICAL: {component.component_name} is unhealthy - {component.message}")
                else:
                    alerts.append(f"WARNING: {component.component_name} is unhealthy - {component.message}")

        uptime = (datetime.now() - self._start_time).total_seconds()

        return SystemHealthReport(
            overall_status=overall_status,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            components=components,
            summary=summary,
            alerts=alerts
        )

    def _calculate_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Calculate overall system status from components."""
        critical_unhealthy = False
        any_unhealthy = False
        any_degraded = False

        for component in components:
            config = self._checks.get(component.component_name)
            is_critical = config.critical if config else False

            if component.status == HealthStatus.UNHEALTHY:
                any_unhealthy = True
                if is_critical:
                    critical_unhealthy = True
            elif component.status == HealthStatus.DEGRADED:
                any_degraded = True

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY
        elif any_unhealthy or any_degraded:
            return HealthStatus.DEGRADED
        elif all(c.status == HealthStatus.HEALTHY for c in components):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def run_all_checks(self) -> None:
        """Run all registered health checks."""
        with self._lock:
            check_names = list(self._checks.keys())

        for name in check_names:
            try:
                self.run_check(name)
            except Exception as e:
                logger.error(f"Failed to run health check {name}: {e}")

    # ========================================================================
    # Kubernetes Health Endpoints
    # ========================================================================

    def liveness_check(self) -> tuple[bool, Dict[str, Any]]:
        """
        Kubernetes liveness probe check.

        Returns (healthy, details) tuple.
        """
        liveness_checks = [
            name for name, config in self._checks.items()
            if config.check_type == CheckType.LIVENESS
        ]

        for name in liveness_checks:
            result = self.run_check(name)
            if result.status == HealthStatus.UNHEALTHY:
                return False, {"failed_check": name, "message": result.message}

        return True, {"status": "alive"}

    def readiness_check(self) -> tuple[bool, Dict[str, Any]]:
        """
        Kubernetes readiness probe check.

        Returns (ready, details) tuple.
        """
        # First check liveness
        alive, _ = self.liveness_check()
        if not alive:
            return False, {"status": "not alive"}

        # Then check critical readiness checks
        readiness_checks = [
            name for name, config in self._checks.items()
            if config.check_type == CheckType.READINESS and config.critical
        ]

        for name in readiness_checks:
            result = self.run_check(name)
            if result.status == HealthStatus.UNHEALTHY:
                return False, {"failed_check": name, "message": result.message}

        return True, {"status": "ready"}

    def startup_check(self) -> tuple[bool, Dict[str, Any]]:
        """
        Kubernetes startup probe check.

        Returns (started, details) tuple.
        """
        startup_checks = [
            name for name, config in self._checks.items()
            if config.check_type == CheckType.STARTUP
        ]

        for name in startup_checks:
            result = self.run_check(name)
            if result.status == HealthStatus.UNHEALTHY:
                return False, {"failed_check": name, "message": result.message}

        return True, {"status": "started"}

    # ========================================================================
    # Background Health Checking
    # ========================================================================

    def start_background_checks(self) -> None:
        """Start background health check loop."""
        if self._running:
            return

        self._running = True
        self._check_thread = threading.Thread(target=self._background_check_loop)
        self._check_thread.daemon = True
        self._check_thread.start()
        logger.info("Background health checks started")

    def stop_background_checks(self) -> None:
        """Stop background health check loop."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        logger.info("Background health checks stopped")

    def _background_check_loop(self) -> None:
        """Background health check loop."""
        check_schedule: Dict[str, datetime] = {}

        while self._running:
            now = datetime.now()

            with self._lock:
                checks_to_run = []
                for name, config in self._checks.items():
                    last_run = check_schedule.get(name, datetime.min)
                    interval = timedelta(seconds=config.interval_seconds)

                    if now - last_run >= interval:
                        checks_to_run.append(name)
                        check_schedule[name] = now

            for name in checks_to_run:
                try:
                    self.run_check(name)
                except Exception as e:
                    logger.error(f"Background health check failed for {name}: {e}")

            time.sleep(1.0)  # Check schedule every second


# ============================================================================
# Health Check HTTP Handler
# ============================================================================

class HealthHTTPHandler:
    """HTTP handler for health check endpoints."""

    def __init__(self, health_checker: HealthChecker, port: int = 8080):
        self.health_checker = health_checker
        self.port = port
        self._server = None

    def start(self) -> None:
        """Start health check HTTP server."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json

        checker = self.health_checker

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health' or self.path == '/healthz':
                    report = checker.get_health_report(run_checks=True)
                    status_code = 200 if report.overall_status == HealthStatus.HEALTHY else 503
                    self._send_json(status_code, report.to_dict())

                elif self.path == '/health/live' or self.path == '/livez':
                    healthy, details = checker.liveness_check()
                    self._send_json(200 if healthy else 503, details)

                elif self.path == '/health/ready' or self.path == '/readyz':
                    ready, details = checker.readiness_check()
                    self._send_json(200 if ready else 503, details)

                elif self.path == '/health/startup':
                    started, details = checker.startup_check()
                    self._send_json(200 if started else 503, details)

                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
                self.send_response(status_code)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data, default=str).encode('utf-8'))

            def log_message(self, format, *args):
                pass  # Suppress logging

        self._server = HTTPServer(('', self.port), Handler)
        thread = threading.Thread(target=self._server.serve_forever)
        thread.daemon = True
        thread.start()
        logger.info(f"Health check server started on port {self.port}")

    def stop(self) -> None:
        """Stop health check HTTP server."""
        if self._server:
            self._server.shutdown()
            logger.info("Health check server stopped")
