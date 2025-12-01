# -*- coding: utf-8 -*-
"""
Message Bus Monitoring - Metrics collection and observability for message bus.

This module provides monitoring capabilities for the GreenLang message bus,
including metrics collection, health checks, and performance tracking.

Example:
    >>> from greenlang.core.messaging import InMemoryMessageBus
    >>> from greenlang.core.messaging.monitoring import MessageBusMonitor
    >>>
    >>> bus = InMemoryMessageBus()
    >>> monitor = MessageBusMonitor(bus)
    >>> await bus.start()
    >>> await monitor.start()
    >>>
    >>> # Get metrics
    >>> metrics = monitor.get_metrics_summary()
    >>> print(f"Events/sec: {metrics['events_per_second']}")
    >>>
    >>> # Check health
    >>> health = monitor.check_health()
    >>> print(f"Status: {health['status']}")
    >>>
    >>> await monitor.stop()
    >>> await bus.close()

Author: GreenLang Framework Team
Date: December 2025
Status: Production Ready
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.core.messaging.message_bus import MessageBus, MessageBusMetrics

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status of the message bus."""

    status: str  # "healthy", "degraded", "unhealthy"
    queue_health: str
    delivery_health: str
    error_rate: float
    queue_utilization: float
    avg_delivery_time_ms: float
    issues: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary."""
        return {
            "status": self.status,
            "queue_health": self.queue_health,
            "delivery_health": self.delivery_health,
            "error_rate": round(self.error_rate, 4),
            "queue_utilization": round(self.queue_utilization, 4),
            "avg_delivery_time_ms": round(self.avg_delivery_time_ms, 2),
            "issues": self.issues,
            "timestamp": self.timestamp,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for message bus."""

    events_per_second: float
    deliveries_per_second: float
    avg_delivery_time_ms: float
    p95_delivery_time_ms: float
    p99_delivery_time_ms: float
    error_rate: float
    timeout_rate: float
    dead_letter_rate: float
    queue_depth: int
    active_subscriptions: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary."""
        return {
            "events_per_second": round(self.events_per_second, 2),
            "deliveries_per_second": round(self.deliveries_per_second, 2),
            "avg_delivery_time_ms": round(self.avg_delivery_time_ms, 2),
            "p95_delivery_time_ms": round(self.p95_delivery_time_ms, 2),
            "p99_delivery_time_ms": round(self.p99_delivery_time_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "timeout_rate": round(self.timeout_rate, 4),
            "dead_letter_rate": round(self.dead_letter_rate, 4),
            "queue_depth": self.queue_depth,
            "active_subscriptions": self.active_subscriptions,
            "timestamp": self.timestamp,
        }


class MessageBusMonitor:
    """
    Monitor for message bus health and performance.

    Tracks metrics, detects performance issues, and provides health checks.

    Example:
        >>> monitor = MessageBusMonitor(bus, check_interval=10.0)
        >>> await monitor.start()
        >>> health = monitor.check_health()
        >>> if health.status != "healthy":
        ...     logger.warning(f"Issues: {health.issues}")
    """

    def __init__(
        self,
        message_bus: MessageBus,
        check_interval: float = 10.0,
        max_queue_utilization: float = 0.8,
        max_error_rate: float = 0.05,
        max_delivery_time_ms: float = 1000.0,
    ):
        """
        Initialize MessageBusMonitor.

        Args:
            message_bus: MessageBus instance to monitor
            check_interval: Interval between health checks (seconds)
            max_queue_utilization: Maximum acceptable queue utilization (0-1)
            max_error_rate: Maximum acceptable error rate (0-1)
            max_delivery_time_ms: Maximum acceptable avg delivery time (ms)
        """
        self.message_bus = message_bus
        self.check_interval = check_interval
        self.max_queue_utilization = max_queue_utilization
        self.max_error_rate = max_error_rate
        self.max_delivery_time_ms = max_delivery_time_ms

        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_metrics: Optional[MessageBusMetrics] = None
        self._last_check_time: float = 0
        self._delivery_time_history: List[float] = []
        self._events_history: List[int] = []
        self._deliveries_history: List[int] = []

        logger.info("MessageBusMonitor initialized")

    async def start(self) -> None:
        """Start monitoring."""
        if self._running:
            logger.warning("Monitor is already running")
            return

        self._running = True
        self._last_check_time = time.time()
        self._last_metrics = self.message_bus.get_metrics()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("MessageBusMonitor started")

    async def stop(self) -> None:
        """Stop monitoring."""
        logger.info("Stopping MessageBusMonitor...")
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("MessageBusMonitor stopped")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)

                # Collect metrics
                current_metrics = self.message_bus.get_metrics()
                current_time = time.time()
                time_delta = current_time - self._last_check_time

                # Calculate rates
                if self._last_metrics and time_delta > 0:
                    events_delta = (
                        current_metrics.events_published
                        - self._last_metrics.events_published
                    )
                    deliveries_delta = (
                        current_metrics.events_delivered
                        - self._last_metrics.events_delivered
                    )

                    events_per_sec = events_delta / time_delta
                    deliveries_per_sec = deliveries_delta / time_delta

                    # Store history (keep last 60 samples = 10 minutes at 10s interval)
                    self._events_history.append(int(events_per_sec))
                    self._deliveries_history.append(int(deliveries_per_sec))
                    self._delivery_time_history.append(
                        current_metrics.avg_delivery_time_ms
                    )

                    if len(self._events_history) > 60:
                        self._events_history.pop(0)
                    if len(self._deliveries_history) > 60:
                        self._deliveries_history.pop(0)
                    if len(self._delivery_time_history) > 60:
                        self._delivery_time_history.pop(0)

                # Check health
                health = self.check_health()
                if health.status != "healthy":
                    logger.warning(
                        f"Message bus health: {health.status}, issues: {health.issues}"
                    )

                self._last_metrics = current_metrics
                self._last_check_time = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)

    def check_health(self) -> HealthStatus:
        """
        Check message bus health.

        Returns:
            HealthStatus with overall status and details
        """
        metrics = self.message_bus.get_metrics()
        issues = []

        # Check queue utilization
        queue_health = "healthy"
        queue_utilization = 0.0

        if hasattr(self.message_bus, "config") and self.message_bus.config.max_queue_size > 0:
            max_queue = self.message_bus.config.max_queue_size
            queue_utilization = metrics.queue_size / max_queue

            if queue_utilization > self.max_queue_utilization:
                queue_health = "degraded"
                issues.append(
                    f"Queue utilization high: {queue_utilization:.1%} "
                    f"({metrics.queue_size}/{max_queue})"
                )
            if queue_utilization > 0.95:
                queue_health = "unhealthy"

        # Check error rate
        delivery_health = "healthy"
        error_rate = 0.0

        total_events = (
            metrics.events_delivered + metrics.events_failed + metrics.events_expired
        )
        if total_events > 0:
            error_rate = (metrics.events_failed + metrics.events_expired) / total_events

            if error_rate > self.max_error_rate:
                delivery_health = "degraded"
                issues.append(f"Error rate high: {error_rate:.1%}")
            if error_rate > 0.2:
                delivery_health = "unhealthy"

        # Check delivery time
        if metrics.avg_delivery_time_ms > self.max_delivery_time_ms:
            delivery_health = "degraded"
            issues.append(
                f"Avg delivery time high: {metrics.avg_delivery_time_ms:.1f}ms"
            )

        # Check dead letter queue
        if hasattr(self.message_bus, "get_dead_letter_queue"):
            dlq_size = len(self.message_bus.get_dead_letter_queue())
            if dlq_size > 100:
                issues.append(f"Dead letter queue size: {dlq_size}")

        # Determine overall status
        if queue_health == "unhealthy" or delivery_health == "unhealthy":
            overall_status = "unhealthy"
        elif queue_health == "degraded" or delivery_health == "degraded":
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return HealthStatus(
            status=overall_status,
            queue_health=queue_health,
            delivery_health=delivery_health,
            error_rate=error_rate,
            queue_utilization=queue_utilization,
            avg_delivery_time_ms=metrics.avg_delivery_time_ms,
            issues=issues,
        )

    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Get performance metrics.

        Returns:
            PerformanceMetrics with current performance data
        """
        metrics = self.message_bus.get_metrics()

        # Calculate rates from history
        events_per_second = (
            sum(self._events_history) / len(self._events_history)
            if self._events_history
            else 0.0
        )
        deliveries_per_second = (
            sum(self._deliveries_history) / len(self._deliveries_history)
            if self._deliveries_history
            else 0.0
        )

        # Calculate percentiles from delivery time history
        p95_delivery_time_ms = 0.0
        p99_delivery_time_ms = 0.0
        if self._delivery_time_history:
            sorted_times = sorted(self._delivery_time_history)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95_delivery_time_ms = sorted_times[p95_idx]
            p99_delivery_time_ms = sorted_times[p99_idx]

        # Calculate rates
        total_events = (
            metrics.events_delivered + metrics.events_failed + metrics.events_expired
        )
        error_rate = (
            (metrics.events_failed + metrics.events_expired) / total_events
            if total_events > 0
            else 0.0
        )

        total_requests = metrics.requests_sent
        timeout_rate = (
            metrics.requests_timeout / total_requests if total_requests > 0 else 0.0
        )

        dead_letter_rate = (
            metrics.events_dead_lettered / total_events if total_events > 0 else 0.0
        )

        return PerformanceMetrics(
            events_per_second=events_per_second,
            deliveries_per_second=deliveries_per_second,
            avg_delivery_time_ms=metrics.avg_delivery_time_ms,
            p95_delivery_time_ms=p95_delivery_time_ms,
            p99_delivery_time_ms=p99_delivery_time_ms,
            error_rate=error_rate,
            timeout_rate=timeout_rate,
            dead_letter_rate=dead_letter_rate,
            queue_depth=metrics.queue_size,
            active_subscriptions=metrics.active_subscriptions,
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Dictionary with health, performance, and raw metrics
        """
        health = self.check_health()
        performance = self.get_performance_metrics()
        raw_metrics = self.message_bus.get_metrics()

        return {
            "health": health.to_dict(),
            "performance": performance.to_dict(),
            "raw_metrics": raw_metrics.to_dict(),
            "monitoring": {
                "running": self._running,
                "check_interval": self.check_interval,
                "history_size": len(self._events_history),
            },
        }

    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            String with Prometheus-formatted metrics
        """
        metrics = self.message_bus.get_metrics()
        performance = self.get_performance_metrics()

        lines = [
            "# HELP greenlang_message_bus_events_published_total Total events published",
            "# TYPE greenlang_message_bus_events_published_total counter",
            f"greenlang_message_bus_events_published_total {metrics.events_published}",
            "",
            "# HELP greenlang_message_bus_events_delivered_total Total events delivered",
            "# TYPE greenlang_message_bus_events_delivered_total counter",
            f"greenlang_message_bus_events_delivered_total {metrics.events_delivered}",
            "",
            "# HELP greenlang_message_bus_events_failed_total Total events failed",
            "# TYPE greenlang_message_bus_events_failed_total counter",
            f"greenlang_message_bus_events_failed_total {metrics.events_failed}",
            "",
            "# HELP greenlang_message_bus_queue_size Current queue size",
            "# TYPE greenlang_message_bus_queue_size gauge",
            f"greenlang_message_bus_queue_size {metrics.queue_size}",
            "",
            "# HELP greenlang_message_bus_active_subscriptions Active subscriptions",
            "# TYPE greenlang_message_bus_active_subscriptions gauge",
            f"greenlang_message_bus_active_subscriptions {metrics.active_subscriptions}",
            "",
            "# HELP greenlang_message_bus_delivery_time_ms Average delivery time in milliseconds",
            "# TYPE greenlang_message_bus_delivery_time_ms gauge",
            f"greenlang_message_bus_delivery_time_ms {metrics.avg_delivery_time_ms:.2f}",
            "",
            "# HELP greenlang_message_bus_events_per_second Events published per second",
            "# TYPE greenlang_message_bus_events_per_second gauge",
            f"greenlang_message_bus_events_per_second {performance.events_per_second:.2f}",
            "",
            "# HELP greenlang_message_bus_error_rate Event error rate",
            "# TYPE greenlang_message_bus_error_rate gauge",
            f"greenlang_message_bus_error_rate {performance.error_rate:.4f}",
            "",
        ]

        return "\n".join(lines)


__all__ = [
    "MessageBusMonitor",
    "HealthStatus",
    "PerformanceMetrics",
]
