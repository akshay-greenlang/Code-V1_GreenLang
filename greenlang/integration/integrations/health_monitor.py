"""
Health Monitor - Connector Health and Performance Monitoring
=============================================================

Centralized health monitoring for all integration connectors.

Features:
- Real-time health status tracking
- Performance metrics collection
- Prometheus metrics export
- Alerting on health degradation
- Historical health data
- Aggregated connector statistics

Example:
    >>> monitor = HealthMonitor()
    >>> monitor.register_connector(connector)
    >>> await monitor.start_monitoring()
    >>>
    >>> # Get health status
    >>> status = monitor.get_health_status("scada-connector")
    >>> print(f"Health: {status.health_status}")

Author: GreenLang Backend Team
Date: 2025-12-01
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import logging
from collections import defaultdict

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from greenlang.integrations.base_connector import (
    BaseConnector,
    HealthStatus,
    ConnectorMetrics
)

logger = logging.getLogger(__name__)


class HealthCheckResult(BaseModel):
    """Result of a health check."""

    connector_id: str = Field(..., description="Connector identifier")
    health_status: HealthStatus = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = Field(..., description="Health check latency")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")
    metrics: Optional[ConnectorMetrics] = Field(default=None, description="Connector metrics")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AggregatedHealth(BaseModel):
    """Aggregated health statistics across all connectors."""

    total_connectors: int = Field(..., description="Total registered connectors")
    healthy_count: int = Field(default=0, description="Healthy connectors")
    degraded_count: int = Field(default=0, description="Degraded connectors")
    unhealthy_count: int = Field(default=0, description="Unhealthy connectors")
    disconnected_count: int = Field(default=0, description="Disconnected connectors")

    overall_health: HealthStatus = Field(..., description="Overall system health")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthMonitor:
    """
    Centralized health monitor for all connectors.

    Provides:
    - Periodic health checks
    - Metrics collection
    - Prometheus integration
    - Health history tracking
    - Alert generation

    Example:
        >>> monitor = HealthMonitor(check_interval=30)
        >>> monitor.register_connector(scada_connector)
        >>> monitor.register_connector(erp_connector)
        >>> await monitor.start_monitoring()
        >>>
        >>> # Get health report
        >>> health = monitor.get_aggregated_health()
        >>> print(f"Healthy: {health.healthy_count}/{health.total_connectors}")
    """

    def __init__(
        self,
        check_interval: int = 60,
        enable_prometheus: bool = True,
        history_retention_hours: int = 24
    ):
        """
        Initialize health monitor.

        Args:
            check_interval: Health check interval in seconds
            enable_prometheus: Enable Prometheus metrics export
            history_retention_hours: Hours to retain health history
        """
        self.check_interval = check_interval
        self.history_retention_hours = history_retention_hours

        # Registered connectors
        self._connectors: Dict[str, BaseConnector] = {}

        # Health check results
        self._health_results: Dict[str, HealthCheckResult] = {}

        # Health history (last 24 hours by default)
        self._health_history: Dict[str, List[HealthCheckResult]] = defaultdict(list)

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

        # Prometheus metrics
        self._prometheus_enabled = enable_prometheus and PROMETHEUS_AVAILABLE
        if self._prometheus_enabled:
            self._init_prometheus_metrics()

        logger.info(
            f"Initialized HealthMonitor "
            f"(interval={check_interval}s, prometheus={self._prometheus_enabled})"
        )

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        # Health status gauge
        self.health_gauge = Gauge(
            'connector_health_status',
            'Connector health status (0=unknown, 1=healthy, 2=degraded, 3=unhealthy)',
            ['connector_id', 'connector_type']
        )

        # Request counters
        self.request_counter = Counter(
            'connector_requests_total',
            'Total connector requests',
            ['connector_id', 'status']
        )

        # Response time histogram
        self.response_histogram = Histogram(
            'connector_response_time_seconds',
            'Connector response time',
            ['connector_id']
        )

        # Circuit breaker opens
        self.circuit_breaker_counter = Counter(
            'connector_circuit_breaker_opens_total',
            'Total circuit breaker opens',
            ['connector_id']
        )

        logger.info("Initialized Prometheus metrics for health monitoring")

    def register_connector(self, connector: BaseConnector):
        """
        Register a connector for health monitoring.

        Args:
            connector: Connector instance to monitor
        """
        connector_id = connector.config.connector_id

        if connector_id in self._connectors:
            logger.warning(f"Connector {connector_id} already registered, replacing")

        self._connectors[connector_id] = connector
        logger.info(f"Registered connector for monitoring: {connector_id}")

    def unregister_connector(self, connector_id: str):
        """
        Unregister a connector from monitoring.

        Args:
            connector_id: Connector identifier
        """
        if connector_id in self._connectors:
            del self._connectors[connector_id]
            if connector_id in self._health_results:
                del self._health_results[connector_id]
            logger.info(f"Unregistered connector: {connector_id}")

    async def check_connector_health(self, connector_id: str) -> HealthCheckResult:
        """
        Perform health check on a specific connector.

        Args:
            connector_id: Connector identifier

        Returns:
            Health check result
        """
        if connector_id not in self._connectors:
            raise ValueError(f"Connector not registered: {connector_id}")

        connector = self._connectors[connector_id]
        start_time = datetime.now(timezone.utc)

        try:
            # Perform health check
            is_healthy = await connector.health_check()

            # Calculate latency
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Get metrics
            metrics = connector.get_metrics()

            # Create result
            result = HealthCheckResult(
                connector_id=connector_id,
                health_status=metrics.health_status,
                latency_ms=latency_ms,
                metrics=metrics
            )

            # Update Prometheus metrics
            if self._prometheus_enabled:
                self._update_prometheus_metrics(connector, metrics)

            return result

        except Exception as e:
            logger.error(f"Health check failed for {connector_id}: {e}", exc_info=True)

            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return HealthCheckResult(
                connector_id=connector_id,
                health_status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error_message=str(e)
            )

    async def check_all_connectors(self) -> Dict[str, HealthCheckResult]:
        """
        Perform health check on all registered connectors.

        Returns:
            Dictionary of connector_id -> HealthCheckResult
        """
        results = {}

        # Run all health checks concurrently
        tasks = {
            connector_id: self.check_connector_health(connector_id)
            for connector_id in self._connectors.keys()
        }

        # Gather results
        for connector_id, task in tasks.items():
            try:
                result = await task
                results[connector_id] = result
                self._health_results[connector_id] = result

                # Add to history
                self._add_to_history(connector_id, result)

            except Exception as e:
                logger.error(f"Failed to check health for {connector_id}: {e}")

        return results

    def _add_to_history(self, connector_id: str, result: HealthCheckResult):
        """Add health check result to history."""
        history = self._health_history[connector_id]
        history.append(result)

        # Prune old entries
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.history_retention_hours)
        self._health_history[connector_id] = [
            r for r in history if r.timestamp >= cutoff
        ]

    def _update_prometheus_metrics(self, connector: BaseConnector, metrics: ConnectorMetrics):
        """Update Prometheus metrics for a connector."""
        if not self._prometheus_enabled:
            return

        connector_id = connector.config.connector_id
        connector_type = connector.config.connector_type

        # Health status (map to numeric value)
        health_value = {
            HealthStatus.UNKNOWN: 0,
            HealthStatus.HEALTHY: 1,
            HealthStatus.DEGRADED: 2,
            HealthStatus.UNHEALTHY: 3
        }.get(metrics.health_status, 0)

        self.health_gauge.labels(
            connector_id=connector_id,
            connector_type=connector_type
        ).set(health_value)

        # Request counters
        self.request_counter.labels(
            connector_id=connector_id,
            status='success'
        )._value.set(metrics.successful_requests)

        self.request_counter.labels(
            connector_id=connector_id,
            status='failure'
        )._value.set(metrics.failed_requests)

        # Response time
        if metrics.avg_response_time_ms > 0:
            self.response_histogram.labels(
                connector_id=connector_id
            ).observe(metrics.avg_response_time_ms / 1000)

        # Circuit breaker
        if metrics.circuit_breaker_opens > 0:
            self.circuit_breaker_counter.labels(
                connector_id=connector_id
            )._value.set(metrics.circuit_breaker_opens)

    async def start_monitoring(self):
        """Start periodic health monitoring."""
        if self._running:
            logger.warning("Health monitoring already running")
            return

        self._running = True

        async def _monitoring_loop():
            while self._running:
                try:
                    await self.check_all_connectors()
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}", exc_info=True)

                await asyncio.sleep(self.check_interval)

        self._monitoring_task = asyncio.create_task(_monitoring_loop())
        logger.info(f"Started health monitoring (interval={self.check_interval}s)")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._running:
            return

        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Stopped health monitoring")

    def get_health_status(self, connector_id: str) -> Optional[HealthCheckResult]:
        """
        Get latest health status for a connector.

        Args:
            connector_id: Connector identifier

        Returns:
            Latest health check result or None
        """
        return self._health_results.get(connector_id)

    def get_health_history(
        self,
        connector_id: str,
        hours: Optional[int] = None
    ) -> List[HealthCheckResult]:
        """
        Get health history for a connector.

        Args:
            connector_id: Connector identifier
            hours: Number of hours to retrieve (default: all)

        Returns:
            List of health check results
        """
        history = self._health_history.get(connector_id, [])

        if hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
            history = [r for r in history if r.timestamp >= cutoff]

        return sorted(history, key=lambda r: r.timestamp, reverse=True)

    def get_aggregated_health(self) -> AggregatedHealth:
        """
        Get aggregated health statistics.

        Returns:
            Aggregated health across all connectors
        """
        total = len(self._connectors)
        healthy = 0
        degraded = 0
        unhealthy = 0
        disconnected = 0

        for result in self._health_results.values():
            if result.health_status == HealthStatus.HEALTHY:
                healthy += 1
            elif result.health_status == HealthStatus.DEGRADED:
                degraded += 1
            elif result.health_status == HealthStatus.UNHEALTHY:
                unhealthy += 1
            elif result.health_status == HealthStatus.DISCONNECTED:
                disconnected += 1

        # Determine overall health
        if unhealthy > 0 or disconnected > 0:
            overall = HealthStatus.UNHEALTHY
        elif degraded > 0:
            overall = HealthStatus.DEGRADED
        elif healthy == total:
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        return AggregatedHealth(
            total_connectors=total,
            healthy_count=healthy,
            degraded_count=degraded,
            unhealthy_count=unhealthy,
            disconnected_count=disconnected,
            overall_health=overall
        )

    def get_all_metrics(self) -> Dict[str, ConnectorMetrics]:
        """
        Get metrics for all connectors.

        Returns:
            Dictionary of connector_id -> ConnectorMetrics
        """
        return {
            connector_id: connector.get_metrics()
            for connector_id, connector in self._connectors.items()
        }
