"""
GL-002 FLAMEGUARD - Circuit Breaker Health Monitoring

This module provides health check endpoints and monitoring for circuit
breakers. Integrates with the existing health check system and provides
Prometheus-compatible metrics.

Features:
    - Health check endpoints for individual breakers
    - Aggregated health status for all breakers
    - Prometheus metrics exposition
    - Kubernetes probe compatibility
    - Alerting integration

Example:
    >>> monitor = CircuitBreakerHealthMonitor()
    >>> monitor.register_connector(protected_scada)
    >>>
    >>> # Get health status
    >>> status = await monitor.check_health()
    >>>
    >>> # Prometheus metrics
    >>> metrics = monitor.get_prometheus_metrics()
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio
import logging
import time

from ..core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)
from .health import (
    HealthStatus,
    ComponentHealth,
    HealthCheckResult,
    HealthChecker,
)

logger = logging.getLogger(__name__)


class CircuitBreakerHealthStatus(Enum):
    """Health status specific to circuit breakers."""
    HEALTHY = "healthy"           # All breakers closed
    DEGRADED = "degraded"         # Some breakers half-open
    CRITICAL = "critical"         # Some breakers open
    OFFLINE = "offline"           # All breakers open


@dataclass
class CircuitBreakerHealthConfig:
    """Configuration for circuit breaker health monitoring."""
    check_interval_s: float = 30.0
    degraded_threshold_percent: float = 25.0    # % of half-open breakers to be degraded
    critical_threshold_percent: float = 50.0    # % of open breakers to be critical
    alert_on_state_change: bool = True
    alert_cooldown_s: float = 300.0
    include_metrics_in_response: bool = True
    prometheus_prefix: str = "greenlang_flameguard_circuit_breaker"


@dataclass
class CircuitBreakerAlert:
    """Alert for circuit breaker events."""
    breaker_name: str
    severity: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    old_state: Optional[CircuitState] = None
    new_state: Optional[CircuitState] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "breaker_name": self.breaker_name,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "old_state": self.old_state.value if self.old_state else None,
            "new_state": self.new_state.value if self.new_state else None,
        }


class CircuitBreakerHealthMonitor:
    """
    Health monitor for circuit breakers.

    Provides comprehensive health monitoring including:
    - Individual breaker health checks
    - Aggregated health status
    - Prometheus metrics
    - Kubernetes probe endpoints
    - Alert generation

    Attributes:
        config: Health monitoring configuration
        registry: Circuit breaker registry

    Example:
        >>> monitor = CircuitBreakerHealthMonitor()
        >>> status = await monitor.check_health()
        >>> if status.overall_status != CircuitBreakerHealthStatus.HEALTHY:
        ...     await monitor.trigger_alerts()
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerHealthConfig] = None,
        on_alert: Optional[Callable[[CircuitBreakerAlert], None]] = None,
    ) -> None:
        """
        Initialize CircuitBreakerHealthMonitor.

        Args:
            config: Health monitoring configuration
            on_alert: Callback for alerts
        """
        self.config = config or CircuitBreakerHealthConfig()
        self.registry = CircuitBreakerRegistry()
        self._on_alert = on_alert

        # Monitoring state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_check_time: Optional[datetime] = None
        self._alerts: List[CircuitBreakerAlert] = []
        self._last_alert_times: Dict[str, datetime] = {}

        # Metrics
        self._check_count = 0
        self._alert_count = 0

        logger.info("CircuitBreakerHealthMonitor initialized")

    async def check_health(self) -> Dict[str, Any]:
        """
        Check health of all circuit breakers.

        Returns:
            Health check result with overall status and per-breaker details
        """
        self._check_count += 1
        self._last_check_time = datetime.now(timezone.utc)

        breaker_statuses = self.registry.get_all_status()
        total = len(breaker_statuses)

        if total == 0:
            return {
                "overall_status": CircuitBreakerHealthStatus.HEALTHY.value,
                "message": "No circuit breakers registered",
                "timestamp": self._last_check_time.isoformat(),
                "breakers": {},
            }

        # Count states
        open_count = sum(
            1 for s in breaker_statuses.values()
            if s["state"] == CircuitState.OPEN.value
        )
        half_open_count = sum(
            1 for s in breaker_statuses.values()
            if s["state"] == CircuitState.HALF_OPEN.value
        )
        closed_count = total - open_count - half_open_count

        # Determine overall status
        open_percent = (open_count / total) * 100
        half_open_percent = (half_open_count / total) * 100

        if open_count == total:
            overall = CircuitBreakerHealthStatus.OFFLINE
            message = "All circuit breakers are OPEN"
        elif open_percent >= self.config.critical_threshold_percent:
            overall = CircuitBreakerHealthStatus.CRITICAL
            message = f"{open_count} of {total} breakers are OPEN"
        elif half_open_percent >= self.config.degraded_threshold_percent:
            overall = CircuitBreakerHealthStatus.DEGRADED
            message = f"{half_open_count} of {total} breakers are HALF_OPEN"
        elif open_count > 0 or half_open_count > 0:
            overall = CircuitBreakerHealthStatus.DEGRADED
            message = f"Some breakers not fully operational"
        else:
            overall = CircuitBreakerHealthStatus.HEALTHY
            message = "All circuit breakers operational"

        result = {
            "overall_status": overall.value,
            "message": message,
            "timestamp": self._last_check_time.isoformat(),
            "summary": {
                "total": total,
                "closed": closed_count,
                "half_open": half_open_count,
                "open": open_count,
            },
            "breakers": breaker_statuses,
        }

        if self.config.include_metrics_in_response:
            result["metrics"] = self._get_aggregated_metrics(breaker_statuses)

        return result

    def _get_aggregated_metrics(
        self,
        breaker_statuses: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Aggregate metrics from all breakers."""
        total_calls = sum(
            s.get("metrics", {}).get("total_calls", 0)
            for s in breaker_statuses.values()
        )
        failed_calls = sum(
            s.get("metrics", {}).get("failed_calls", 0)
            for s in breaker_statuses.values()
        )
        rejected_calls = sum(
            s.get("metrics", {}).get("rejected_calls", 0)
            for s in breaker_statuses.values()
        )
        state_transitions = sum(
            s.get("metrics", {}).get("state_transitions", 0)
            for s in breaker_statuses.values()
        )

        return {
            "total_calls": total_calls,
            "failed_calls": failed_calls,
            "rejected_calls": rejected_calls,
            "state_transitions": state_transitions,
            "failure_rate": failed_calls / total_calls if total_calls > 0 else 0.0,
            "rejection_rate": rejected_calls / total_calls if total_calls > 0 else 0.0,
        }

    async def check_breaker(self, name: str) -> Optional[ComponentHealth]:
        """
        Check health of a specific circuit breaker.

        Args:
            name: Name of the circuit breaker

        Returns:
            ComponentHealth or None if breaker not found
        """
        breaker = self.registry.get(name)
        if not breaker:
            return None

        start_time = time.time()
        status = breaker.get_status()
        response_time = (time.time() - start_time) * 1000

        if breaker.is_closed:
            health_status = HealthStatus.HEALTHY
            message = "Circuit breaker closed (normal)"
        elif breaker.is_half_open:
            health_status = HealthStatus.DEGRADED
            message = "Circuit breaker half-open (testing recovery)"
        else:
            health_status = HealthStatus.UNHEALTHY
            message = f"Circuit breaker open (retry in {status['time_until_retry_s']:.1f}s)"

        return ComponentHealth(
            name=f"circuit_breaker_{name}",
            status=health_status,
            message=message,
            response_time_ms=response_time,
            details=status,
        )

    def get_prometheus_metrics(self) -> str:
        """
        Get Prometheus-compatible metrics.

        Returns:
            Metrics in Prometheus exposition format
        """
        prefix = self.config.prometheus_prefix
        lines = []

        # Add HELP and TYPE for each metric
        lines.append(f"# HELP {prefix}_state Current state (0=closed, 1=half_open, 2=open)")
        lines.append(f"# TYPE {prefix}_state gauge")

        lines.append(f"# HELP {prefix}_total_calls Total number of calls")
        lines.append(f"# TYPE {prefix}_total_calls counter")

        lines.append(f"# HELP {prefix}_failed_calls Total number of failed calls")
        lines.append(f"# TYPE {prefix}_failed_calls counter")

        lines.append(f"# HELP {prefix}_rejected_calls Calls rejected due to open circuit")
        lines.append(f"# TYPE {prefix}_rejected_calls counter")

        lines.append(f"# HELP {prefix}_state_transitions Total state transitions")
        lines.append(f"# TYPE {prefix}_state_transitions counter")

        lines.append(f"# HELP {prefix}_failure_rate Current failure rate")
        lines.append(f"# TYPE {prefix}_failure_rate gauge")

        lines.append(f"# HELP {prefix}_time_in_open_seconds Total time in open state")
        lines.append(f"# TYPE {prefix}_time_in_open_seconds counter")

        # Add metrics for each breaker
        for name, breaker in self.registry._breakers.items():
            labels = f'breaker="{name}"'
            status = breaker.get_status()
            metrics = status.get("metrics", {})

            # State as numeric value
            state_val = {
                CircuitState.CLOSED: 0,
                CircuitState.HALF_OPEN: 1,
                CircuitState.OPEN: 2,
            }.get(breaker.state, -1)

            lines.append(f'{prefix}_state{{{labels}}} {state_val}')
            lines.append(f'{prefix}_total_calls{{{labels}}} {metrics.get("total_calls", 0)}')
            lines.append(f'{prefix}_failed_calls{{{labels}}} {metrics.get("failed_calls", 0)}')
            lines.append(f'{prefix}_rejected_calls{{{labels}}} {metrics.get("rejected_calls", 0)}')
            lines.append(f'{prefix}_state_transitions{{{labels}}} {metrics.get("state_transitions", 0)}')
            lines.append(f'{prefix}_failure_rate{{{labels}}} {metrics.get("failure_rate", 0.0):.4f}')
            lines.append(f'{prefix}_time_in_open_seconds{{{labels}}} {metrics.get("time_in_open_s", 0.0):.2f}')

        return "\n".join(lines)

    def is_ready(self) -> bool:
        """
        Kubernetes readiness probe.

        Returns:
            True if system is ready to receive traffic
        """
        # Ready if not all breakers are open
        open_breakers = self.registry.get_open_breakers()
        total = len(self.registry._breakers)

        if total == 0:
            return True

        # Ready if less than half of breakers are open
        return len(open_breakers) < (total / 2)

    def is_live(self) -> bool:
        """
        Kubernetes liveness probe.

        Returns:
            True if system is alive
        """
        # Always live if we can respond
        return True

    def get_readiness_response(self) -> Dict[str, Any]:
        """Get readiness probe response."""
        ready = self.is_ready()
        return {
            "ready": ready,
            "status": "ready" if ready else "not_ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "open_breakers": self.registry.get_open_breakers(),
        }

    def get_liveness_response(self) -> Dict[str, Any]:
        """Get liveness probe response."""
        return {
            "live": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _generate_alert(
        self,
        breaker_name: str,
        old_state: CircuitState,
        new_state: CircuitState,
    ) -> None:
        """Generate alert for state change."""
        now = datetime.now(timezone.utc)

        # Check cooldown
        last_alert = self._last_alert_times.get(breaker_name)
        if last_alert:
            elapsed = (now - last_alert).total_seconds()
            if elapsed < self.config.alert_cooldown_s:
                return

        # Determine severity
        if new_state == CircuitState.OPEN:
            severity = "critical"
            message = f"Circuit breaker '{breaker_name}' has OPENED"
        elif new_state == CircuitState.HALF_OPEN:
            severity = "warning"
            message = f"Circuit breaker '{breaker_name}' is testing recovery"
        else:
            severity = "info"
            message = f"Circuit breaker '{breaker_name}' has recovered"

        alert = CircuitBreakerAlert(
            breaker_name=breaker_name,
            severity=severity,
            message=message,
            old_state=old_state,
            new_state=new_state,
        )

        self._alerts.append(alert)
        self._alert_count += 1
        self._last_alert_times[breaker_name] = now

        logger.warning(f"Circuit breaker alert: {message}")

        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    async def start_monitoring(self) -> None:
        """Start background monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Circuit breaker health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Circuit breaker health monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_health()
            except Exception as e:
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(self.config.check_interval_s)

    def get_alerts(
        self,
        since: Optional[datetime] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get recent alerts.

        Args:
            since: Only return alerts after this time
            severity: Filter by severity
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp > since]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Return most recent first
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)

        return [a.to_dict() for a in alerts[:limit]]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
        logger.info("Alerts cleared")

    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            "running": self._running,
            "check_count": self._check_count,
            "alert_count": self._alert_count,
            "last_check": (
                self._last_check_time.isoformat()
                if self._last_check_time else None
            ),
            "breaker_count": len(self.registry._breakers),
            "health_summary": self.registry.get_health_summary(),
        }


def register_circuit_breaker_health_checks(
    health_checker: HealthChecker,
    registry: Optional[CircuitBreakerRegistry] = None,
) -> None:
    """
    Register circuit breaker health checks with the main health checker.

    Args:
        health_checker: Main health checker instance
        registry: Circuit breaker registry (uses singleton if not provided)
    """
    if registry is None:
        registry = CircuitBreakerRegistry()

    async def check_circuit_breakers() -> ComponentHealth:
        """Check all circuit breaker health."""
        summary = registry.get_health_summary()

        if summary["open"] > 0:
            status = HealthStatus.UNHEALTHY
            message = f"{summary['open']} circuit breakers open"
        elif summary["half_open"] > 0:
            status = HealthStatus.DEGRADED
            message = f"{summary['half_open']} circuit breakers recovering"
        else:
            status = HealthStatus.HEALTHY
            message = "All circuit breakers operational"

        return ComponentHealth(
            name="circuit_breakers",
            status=status,
            message=message,
            details=summary,
        )

    health_checker.register_check("circuit_breakers", check_circuit_breakers)
    logger.info("Registered circuit breaker health checks")
