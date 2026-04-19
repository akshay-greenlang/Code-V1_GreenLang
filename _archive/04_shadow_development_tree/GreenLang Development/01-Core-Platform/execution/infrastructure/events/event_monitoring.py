"""
Event Monitoring Dashboard for GreenLang
=========================================

TASK-129: Event Monitoring Implementation

This module provides comprehensive monitoring capabilities for the event system,
including metrics collection, latency tracking, and Prometheus integration.

Features:
- Event throughput metrics
- Event latency tracking
- Dead letter queue statistics
- Consumer lag monitoring
- Event type distribution
- FastAPI endpoints for dashboard data
- Prometheus metrics export

Example:
    >>> from greenlang.infrastructure.events import EventMonitor, MonitorConfig
    >>> monitor = EventMonitor(config)
    >>> await monitor.start()
    >>> metrics = await monitor.get_dashboard_metrics()

Author: GreenLang Infrastructure Team
Created: 2025-12-07
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MonitorConfig:
    """Configuration for event monitoring."""
    # Metrics collection
    collection_interval_seconds: float = 10.0
    retention_hours: int = 24
    histogram_buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    # Prometheus
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"

    # Alerting
    enable_alerting: bool = True
    dlq_alert_threshold: int = 100
    latency_alert_threshold_ms: float = 1000.0
    error_rate_alert_threshold: float = 0.05  # 5%
    consumer_lag_alert_threshold: int = 10000

    # Dashboard
    dashboard_update_interval_seconds: float = 5.0

    # Logging
    log_metrics: bool = True
    log_interval_seconds: float = 60.0


# =============================================================================
# Metric Models
# =============================================================================


class MetricValue(BaseModel):
    """A single metric value with timestamp."""
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = Field(default_factory=dict)


class TimeSeriesMetric(BaseModel):
    """Time series of metric values."""
    name: str = Field(..., description="Metric name")
    metric_type: MetricType = Field(..., description="Type of metric")
    description: str = Field(default="", description="Metric description")
    values: List[MetricValue] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)

    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new value to the time series."""
        self.values.append(MetricValue(
            value=value,
            labels=labels or self.labels
        ))

    def get_latest(self) -> Optional[MetricValue]:
        """Get the latest value."""
        return self.values[-1] if self.values else None

    def get_average(self, window_seconds: float = 60.0) -> Optional[float]:
        """Get average value over time window."""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent = [v.value for v in self.values if v.timestamp >= cutoff]
        return sum(recent) / len(recent) if recent else None


class HistogramMetric(BaseModel):
    """Histogram metric with buckets."""
    name: str = Field(..., description="Metric name")
    description: str = Field(default="")
    buckets: List[float] = Field(default_factory=list)
    bucket_counts: Dict[float, int] = Field(default_factory=dict)
    sum_value: float = Field(default=0.0)
    count: int = Field(default=0)
    labels: Dict[str, str] = Field(default_factory=dict)

    def observe(self, value: float) -> None:
        """Observe a value in the histogram."""
        self.sum_value += value
        self.count += 1

        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1

    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get approximate percentile from histogram."""
        if self.count == 0:
            return None

        target = self.count * percentile
        cumulative = 0

        for bucket in sorted(self.buckets):
            cumulative += self.bucket_counts.get(bucket, 0)
            if cumulative >= target:
                return bucket

        return self.buckets[-1] if self.buckets else None


# =============================================================================
# Dashboard Models
# =============================================================================


class ThroughputMetrics(BaseModel):
    """Event throughput metrics."""
    events_per_second: float = Field(default=0.0)
    events_per_minute: float = Field(default=0.0)
    total_events_24h: int = Field(default=0)
    peak_events_per_second: float = Field(default=0.0)
    peak_timestamp: Optional[datetime] = Field(default=None)
    by_event_type: Dict[str, float] = Field(default_factory=dict)
    by_stream: Dict[str, float] = Field(default_factory=dict)


class LatencyMetrics(BaseModel):
    """Event latency metrics."""
    avg_latency_ms: float = Field(default=0.0)
    p50_latency_ms: float = Field(default=0.0)
    p95_latency_ms: float = Field(default=0.0)
    p99_latency_ms: float = Field(default=0.0)
    max_latency_ms: float = Field(default=0.0)
    by_event_type: Dict[str, float] = Field(default_factory=dict)


class DLQMetrics(BaseModel):
    """Dead letter queue metrics."""
    pending_count: int = Field(default=0)
    resolved_count: int = Field(default=0)
    discarded_count: int = Field(default=0)
    escalated_count: int = Field(default=0)
    total_24h: int = Field(default=0)
    by_failure_reason: Dict[str, int] = Field(default_factory=dict)
    by_event_type: Dict[str, int] = Field(default_factory=dict)
    oldest_pending_age_hours: Optional[float] = Field(default=None)
    avg_resolution_time_hours: Optional[float] = Field(default=None)


class ConsumerLagMetrics(BaseModel):
    """Consumer lag metrics."""
    total_lag: int = Field(default=0)
    by_consumer_group: Dict[str, int] = Field(default_factory=dict)
    by_partition: Dict[str, int] = Field(default_factory=dict)
    oldest_unprocessed_age_seconds: Optional[float] = Field(default=None)


class EventTypeDistribution(BaseModel):
    """Event type distribution."""
    distribution: Dict[str, int] = Field(default_factory=dict)
    percentages: Dict[str, float] = Field(default_factory=dict)
    trending_up: List[str] = Field(default_factory=list)
    trending_down: List[str] = Field(default_factory=list)


class SystemHealth(BaseModel):
    """Overall system health."""
    status: HealthStatus = Field(default=HealthStatus.HEALTHY)
    score: float = Field(default=100.0)  # 0-100
    components: Dict[str, HealthStatus] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)
    last_check: datetime = Field(default_factory=datetime.utcnow)


class DashboardMetrics(BaseModel):
    """Complete dashboard metrics."""
    throughput: ThroughputMetrics = Field(default_factory=ThroughputMetrics)
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    dlq: DLQMetrics = Field(default_factory=DLQMetrics)
    consumer_lag: ConsumerLagMetrics = Field(default_factory=ConsumerLagMetrics)
    event_distribution: EventTypeDistribution = Field(default_factory=EventTypeDistribution)
    health: SystemHealth = Field(default_factory=SystemHealth)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")

    def calculate_provenance_hash(self) -> str:
        """Calculate provenance hash for audit."""
        data = f"{self.generated_at.isoformat()}:{self.throughput.total_events_24h}"
        return hashlib.sha256(data.encode()).hexdigest()


class Alert(BaseModel):
    """System alert."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    severity: AlertSeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    metric_name: str = Field(default="", description="Related metric")
    metric_value: Optional[float] = Field(default=None)
    threshold: Optional[float] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = Field(default=False)
    acknowledged_at: Optional[datetime] = Field(default=None)
    acknowledged_by: Optional[str] = Field(default=None)


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """
    Collects and aggregates event metrics.

    Tracks throughput, latency, errors, and other key metrics
    for the event system.
    """

    def __init__(self, config: MonitorConfig):
        """Initialize the collector."""
        self.config = config

        # Counters
        self._event_count: Dict[str, int] = defaultdict(int)
        self._error_count: Dict[str, int] = defaultdict(int)

        # Latency histograms
        self._latency_histograms: Dict[str, HistogramMetric] = {}

        # Time series
        self._throughput_series: List[Tuple[datetime, float]] = []
        self._latency_series: List[Tuple[datetime, float]] = []

        # Consumer lag
        self._consumer_lag: Dict[str, int] = {}

        # Event type tracking
        self._event_type_counts: Dict[str, int] = defaultdict(int)
        self._event_type_counts_previous: Dict[str, int] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Peak tracking
        self._peak_throughput = 0.0
        self._peak_timestamp: Optional[datetime] = None

    async def record_event(
        self,
        event_type: str,
        stream_id: str,
        latency_ms: Optional[float] = None,
        success: bool = True
    ) -> None:
        """
        Record an event being processed.

        Args:
            event_type: Type of event
            stream_id: Stream identifier
            latency_ms: Processing latency in milliseconds
            success: Whether processing succeeded
        """
        async with self._lock:
            # Increment counters
            self._event_count["total"] += 1
            self._event_count[f"type:{event_type}"] += 1
            self._event_count[f"stream:{stream_id}"] += 1
            self._event_type_counts[event_type] += 1

            if not success:
                self._error_count["total"] += 1
                self._error_count[f"type:{event_type}"] += 1

            # Record latency
            if latency_ms is not None:
                if event_type not in self._latency_histograms:
                    self._latency_histograms[event_type] = HistogramMetric(
                        name=f"event_latency_{event_type}",
                        description=f"Latency for {event_type} events",
                        buckets=self.config.histogram_buckets
                    )
                self._latency_histograms[event_type].observe(latency_ms)

    async def record_throughput(self, events_per_second: float) -> None:
        """Record current throughput."""
        async with self._lock:
            now = datetime.utcnow()
            self._throughput_series.append((now, events_per_second))

            # Update peak
            if events_per_second > self._peak_throughput:
                self._peak_throughput = events_per_second
                self._peak_timestamp = now

            # Trim old data
            cutoff = now - timedelta(hours=self.config.retention_hours)
            self._throughput_series = [
                (t, v) for t, v in self._throughput_series if t >= cutoff
            ]

    async def record_consumer_lag(self, consumer_group: str, lag: int) -> None:
        """Record consumer lag."""
        async with self._lock:
            self._consumer_lag[consumer_group] = lag

    async def get_throughput_metrics(self) -> ThroughputMetrics:
        """Get throughput metrics."""
        async with self._lock:
            now = datetime.utcnow()

            # Calculate events per second (last minute)
            minute_ago = now - timedelta(minutes=1)
            recent = [v for t, v in self._throughput_series if t >= minute_ago]
            eps = sum(recent) / len(recent) if recent else 0.0

            # Events per minute
            epm = eps * 60

            # Total 24h
            total_24h = self._event_count["total"]

            # By event type
            by_type = {
                k.replace("type:", ""): v
                for k, v in self._event_count.items()
                if k.startswith("type:")
            }

            # By stream
            by_stream = {
                k.replace("stream:", ""): v
                for k, v in self._event_count.items()
                if k.startswith("stream:")
            }

            return ThroughputMetrics(
                events_per_second=eps,
                events_per_minute=epm,
                total_events_24h=total_24h,
                peak_events_per_second=self._peak_throughput,
                peak_timestamp=self._peak_timestamp,
                by_event_type={k: float(v) for k, v in by_type.items()},
                by_stream={k: float(v) for k, v in by_stream.items()}
            )

    async def get_latency_metrics(self) -> LatencyMetrics:
        """Get latency metrics."""
        async with self._lock:
            all_histograms = list(self._latency_histograms.values())

            if not all_histograms:
                return LatencyMetrics()

            # Aggregate all histograms
            total_sum = sum(h.sum_value for h in all_histograms)
            total_count = sum(h.count for h in all_histograms)

            avg = total_sum / total_count if total_count > 0 else 0.0

            # Get percentiles from first histogram as approximation
            first = all_histograms[0] if all_histograms else None
            p50 = first.get_percentile(0.5) if first else 0.0
            p95 = first.get_percentile(0.95) if first else 0.0
            p99 = first.get_percentile(0.99) if first else 0.0

            # Max across all histograms
            max_latency = max(
                (h.buckets[-1] if h.buckets else 0.0) for h in all_histograms
            ) if all_histograms else 0.0

            # By event type
            by_type = {
                h.name.replace("event_latency_", ""): h.sum_value / h.count if h.count > 0 else 0.0
                for h in all_histograms
            }

            return LatencyMetrics(
                avg_latency_ms=avg,
                p50_latency_ms=p50 or 0.0,
                p95_latency_ms=p95 or 0.0,
                p99_latency_ms=p99 or 0.0,
                max_latency_ms=max_latency,
                by_event_type=by_type
            )

    async def get_consumer_lag_metrics(self) -> ConsumerLagMetrics:
        """Get consumer lag metrics."""
        async with self._lock:
            total_lag = sum(self._consumer_lag.values())

            return ConsumerLagMetrics(
                total_lag=total_lag,
                by_consumer_group=dict(self._consumer_lag)
            )

    async def get_event_distribution(self) -> EventTypeDistribution:
        """Get event type distribution."""
        async with self._lock:
            total = sum(self._event_type_counts.values())

            # Calculate percentages
            percentages = {}
            if total > 0:
                percentages = {
                    k: (v / total) * 100
                    for k, v in self._event_type_counts.items()
                }

            # Determine trending
            trending_up = []
            trending_down = []

            for event_type, count in self._event_type_counts.items():
                prev = self._event_type_counts_previous.get(event_type, 0)
                if count > prev * 1.1:  # 10% increase
                    trending_up.append(event_type)
                elif count < prev * 0.9:  # 10% decrease
                    trending_down.append(event_type)

            # Update previous for next comparison
            self._event_type_counts_previous = dict(self._event_type_counts)

            return EventTypeDistribution(
                distribution=dict(self._event_type_counts),
                percentages=percentages,
                trending_up=trending_up,
                trending_down=trending_down
            )

    async def reset_metrics(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            self._event_count.clear()
            self._error_count.clear()
            self._latency_histograms.clear()
            self._throughput_series.clear()
            self._consumer_lag.clear()
            self._event_type_counts.clear()
            self._peak_throughput = 0.0
            self._peak_timestamp = None


# =============================================================================
# Event Monitor
# =============================================================================


class EventMonitor:
    """
    Event Monitoring Dashboard for GreenLang.

    Provides comprehensive monitoring, metrics collection,
    and alerting for the event system.

    Attributes:
        config: Monitor configuration
        collector: Metrics collector

    Example:
        >>> monitor = EventMonitor(MonitorConfig())
        >>> await monitor.start()
        >>>
        >>> # Get dashboard metrics
        >>> metrics = await monitor.get_dashboard_metrics()
        >>>
        >>> # Get Prometheus metrics
        >>> prometheus_text = await monitor.get_prometheus_metrics()
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        """
        Initialize the event monitor.

        Args:
            config: Monitor configuration
        """
        self.config = config or MonitorConfig()
        self.collector = MetricsCollector(self.config)

        # Alert management
        self._alerts: List[Alert] = []
        self._alert_callbacks: List[Callable] = []

        # DLQ reference (set externally)
        self._dlq = None

        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        self._running = False

        # Dashboard cache
        self._dashboard_cache: Optional[DashboardMetrics] = None
        self._cache_timestamp: Optional[datetime] = None

        logger.info("EventMonitor initialized")

    def set_dlq(self, dlq) -> None:
        """Set the dead letter queue for monitoring."""
        self._dlq = dlq

    def add_alert_callback(self, callback: Callable[[Alert], Any]) -> None:
        """Add a callback for alerts."""
        self._alert_callbacks.append(callback)

    async def start(self) -> None:
        """Start the event monitor."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._collection_task = asyncio.create_task(self._collection_loop())

        if self.config.enable_alerting:
            self._alert_task = asyncio.create_task(self._alert_loop())

        logger.info("EventMonitor started")

    async def stop(self) -> None:
        """Stop the event monitor."""
        self._running = False

        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        if self._alert_task:
            self._alert_task.cancel()
            try:
                await self._alert_task
            except asyncio.CancelledError:
                pass

        logger.info("EventMonitor stopped")

    async def _collection_loop(self) -> None:
        """Background loop for metrics collection."""
        last_event_count = 0
        last_check = time.time()

        while self._running:
            try:
                await asyncio.sleep(self.config.collection_interval_seconds)

                # Calculate throughput
                current_count = self.collector._event_count.get("total", 0)
                current_time = time.time()
                elapsed = current_time - last_check

                if elapsed > 0:
                    eps = (current_count - last_event_count) / elapsed
                    await self.collector.record_throughput(eps)

                last_event_count = current_count
                last_check = current_time

                # Log metrics if enabled
                if self.config.log_metrics:
                    metrics = await self.get_dashboard_metrics()
                    logger.info(
                        f"Event metrics: throughput={metrics.throughput.events_per_second:.2f}/s "
                        f"latency_p95={metrics.latency.p95_latency_ms:.2f}ms "
                        f"dlq_pending={metrics.dlq.pending_count}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

    async def _alert_loop(self) -> None:
        """Background loop for alert checking."""
        while self._running:
            try:
                await asyncio.sleep(self.config.collection_interval_seconds)
                await self._check_alerts()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert loop error: {e}")

    async def _check_alerts(self) -> None:
        """Check metrics and raise alerts if thresholds exceeded."""
        metrics = await self.get_dashboard_metrics()

        # Check DLQ threshold
        if metrics.dlq.pending_count >= self.config.dlq_alert_threshold:
            await self._raise_alert(
                AlertSeverity.WARNING,
                "DLQ Alert",
                f"Dead letter queue has {metrics.dlq.pending_count} pending events",
                "dlq_pending_count",
                float(metrics.dlq.pending_count),
                float(self.config.dlq_alert_threshold)
            )

        # Check latency threshold
        if metrics.latency.p95_latency_ms >= self.config.latency_alert_threshold_ms:
            await self._raise_alert(
                AlertSeverity.WARNING,
                "High Latency Alert",
                f"P95 latency is {metrics.latency.p95_latency_ms:.2f}ms",
                "latency_p95",
                metrics.latency.p95_latency_ms,
                self.config.latency_alert_threshold_ms
            )

        # Check consumer lag
        if metrics.consumer_lag.total_lag >= self.config.consumer_lag_alert_threshold:
            await self._raise_alert(
                AlertSeverity.WARNING,
                "Consumer Lag Alert",
                f"Total consumer lag is {metrics.consumer_lag.total_lag}",
                "consumer_lag",
                float(metrics.consumer_lag.total_lag),
                float(self.config.consumer_lag_alert_threshold)
            )

    async def _raise_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float
    ) -> None:
        """Raise an alert."""
        # Check for duplicate active alerts
        for existing in self._alerts:
            if (
                existing.metric_name == metric_name
                and not existing.acknowledged
                and existing.created_at > datetime.utcnow() - timedelta(hours=1)
            ):
                return  # Don't duplicate

        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )

        self._alerts.append(alert)

        logger.warning(f"Alert raised: {title} - {message}")

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def record_event(
        self,
        event_type: str,
        stream_id: str,
        latency_ms: Optional[float] = None,
        success: bool = True
    ) -> None:
        """
        Record an event being processed.

        Args:
            event_type: Type of event
            stream_id: Stream identifier
            latency_ms: Processing latency in milliseconds
            success: Whether processing succeeded
        """
        await self.collector.record_event(event_type, stream_id, latency_ms, success)

    async def record_consumer_lag(self, consumer_group: str, lag: int) -> None:
        """
        Record consumer lag.

        Args:
            consumer_group: Consumer group name
            lag: Number of messages behind
        """
        await self.collector.record_consumer_lag(consumer_group, lag)

    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """
        Get complete dashboard metrics.

        Returns:
            DashboardMetrics with all monitoring data
        """
        # Check cache
        if (
            self._dashboard_cache
            and self._cache_timestamp
            and (datetime.utcnow() - self._cache_timestamp).total_seconds() < self.config.dashboard_update_interval_seconds
        ):
            return self._dashboard_cache

        # Collect all metrics
        throughput = await self.collector.get_throughput_metrics()
        latency = await self.collector.get_latency_metrics()
        consumer_lag = await self.collector.get_consumer_lag_metrics()
        event_dist = await self.collector.get_event_distribution()

        # Get DLQ metrics if available
        dlq_metrics = DLQMetrics()
        if self._dlq:
            try:
                stats = await self._dlq.get_stats()
                dlq_metrics = DLQMetrics(
                    pending_count=stats.pending_count,
                    resolved_count=stats.resolved_count,
                    discarded_count=stats.discarded_count,
                    escalated_count=stats.escalated_count,
                    total_24h=stats.total_entries,
                    by_failure_reason=stats.entries_by_reason,
                    by_event_type=stats.entries_by_topic,
                    oldest_pending_age_hours=stats.oldest_pending_age_hours
                )
            except Exception as e:
                logger.error(f"Failed to get DLQ stats: {e}")

        # Calculate health
        health = await self._calculate_health(throughput, latency, dlq_metrics, consumer_lag)

        metrics = DashboardMetrics(
            throughput=throughput,
            latency=latency,
            dlq=dlq_metrics,
            consumer_lag=consumer_lag,
            event_distribution=event_dist,
            health=health
        )
        metrics.provenance_hash = metrics.calculate_provenance_hash()

        # Update cache
        self._dashboard_cache = metrics
        self._cache_timestamp = datetime.utcnow()

        return metrics

    async def _calculate_health(
        self,
        throughput: ThroughputMetrics,
        latency: LatencyMetrics,
        dlq: DLQMetrics,
        consumer_lag: ConsumerLagMetrics
    ) -> SystemHealth:
        """Calculate overall system health."""
        issues = []
        components = {}
        score = 100.0

        # Check throughput
        if throughput.events_per_second == 0:
            issues.append("No events being processed")
            components["throughput"] = HealthStatus.UNHEALTHY
            score -= 20
        else:
            components["throughput"] = HealthStatus.HEALTHY

        # Check latency
        if latency.p95_latency_ms > self.config.latency_alert_threshold_ms:
            issues.append(f"High latency: P95 = {latency.p95_latency_ms:.2f}ms")
            components["latency"] = HealthStatus.DEGRADED
            score -= 15
        else:
            components["latency"] = HealthStatus.HEALTHY

        # Check DLQ
        if dlq.pending_count >= self.config.dlq_alert_threshold:
            issues.append(f"DLQ has {dlq.pending_count} pending events")
            components["dlq"] = HealthStatus.DEGRADED
            score -= 15
        else:
            components["dlq"] = HealthStatus.HEALTHY

        # Check consumer lag
        if consumer_lag.total_lag >= self.config.consumer_lag_alert_threshold:
            issues.append(f"High consumer lag: {consumer_lag.total_lag}")
            components["consumer_lag"] = HealthStatus.DEGRADED
            score -= 10
        else:
            components["consumer_lag"] = HealthStatus.HEALTHY

        # Determine overall status
        if score >= 80:
            status = HealthStatus.HEALTHY
        elif score >= 50:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.UNHEALTHY

        return SystemHealth(
            status=status,
            score=max(0, score),
            components=components,
            issues=issues
        )

    async def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus format.

        Returns:
            Prometheus exposition format text
        """
        lines = []
        metrics = await self.get_dashboard_metrics()

        # Throughput metrics
        lines.append("# HELP greenlang_events_total Total events processed")
        lines.append("# TYPE greenlang_events_total counter")
        lines.append(f"greenlang_events_total {metrics.throughput.total_events_24h}")

        lines.append("# HELP greenlang_events_per_second Events processed per second")
        lines.append("# TYPE greenlang_events_per_second gauge")
        lines.append(f"greenlang_events_per_second {metrics.throughput.events_per_second:.2f}")

        # Latency metrics
        lines.append("# HELP greenlang_event_latency_ms Event processing latency")
        lines.append("# TYPE greenlang_event_latency_ms gauge")
        lines.append(f'greenlang_event_latency_ms{{quantile="0.5"}} {metrics.latency.p50_latency_ms:.2f}')
        lines.append(f'greenlang_event_latency_ms{{quantile="0.95"}} {metrics.latency.p95_latency_ms:.2f}')
        lines.append(f'greenlang_event_latency_ms{{quantile="0.99"}} {metrics.latency.p99_latency_ms:.2f}')

        # DLQ metrics
        lines.append("# HELP greenlang_dlq_pending Dead letter queue pending count")
        lines.append("# TYPE greenlang_dlq_pending gauge")
        lines.append(f"greenlang_dlq_pending {metrics.dlq.pending_count}")

        lines.append("# HELP greenlang_dlq_total Dead letter queue total 24h")
        lines.append("# TYPE greenlang_dlq_total counter")
        lines.append(f"greenlang_dlq_total {metrics.dlq.total_24h}")

        # Consumer lag
        lines.append("# HELP greenlang_consumer_lag Total consumer lag")
        lines.append("# TYPE greenlang_consumer_lag gauge")
        lines.append(f"greenlang_consumer_lag {metrics.consumer_lag.total_lag}")

        # Health score
        lines.append("# HELP greenlang_health_score System health score")
        lines.append("# TYPE greenlang_health_score gauge")
        lines.append(f"greenlang_health_score {metrics.health.score}")

        # Event type distribution
        lines.append("# HELP greenlang_events_by_type Events by type")
        lines.append("# TYPE greenlang_events_by_type counter")
        for event_type, count in metrics.event_distribution.distribution.items():
            lines.append(f'greenlang_events_by_type{{type="{event_type}"}} {count}')

        return "\n".join(lines)

    async def get_alerts(
        self,
        acknowledged: Optional[bool] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        Get alerts with optional filtering.

        Args:
            acknowledged: Filter by acknowledgment status
            severity: Filter by severity

        Returns:
            List of matching alerts
        """
        alerts = self._alerts

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging

        Returns:
            True if acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Check monitor health.

        Returns:
            Health status
        """
        metrics = await self.get_dashboard_metrics()

        return {
            "status": metrics.health.status.value,
            "score": metrics.health.score,
            "running": self._running,
            "active_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "components": {k: v.value for k, v in metrics.health.components.items()}
        }


# =============================================================================
# FastAPI Router
# =============================================================================


def create_monitoring_router(monitor: EventMonitor):
    """
    Create FastAPI router for event monitoring.

    Args:
        monitor: EventMonitor instance

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, HTTPException, Query, status
        from fastapi.responses import PlainTextResponse
    except ImportError:
        logger.warning("FastAPI not available, skipping router creation")
        return None

    router = APIRouter(prefix="/api/v1/monitoring", tags=["Event Monitoring"])

    @router.get("/dashboard")
    async def get_dashboard():
        """Get dashboard metrics."""
        metrics = await monitor.get_dashboard_metrics()
        return metrics.dict()

    @router.get("/throughput")
    async def get_throughput():
        """Get throughput metrics."""
        metrics = await monitor.get_dashboard_metrics()
        return metrics.throughput.dict()

    @router.get("/latency")
    async def get_latency():
        """Get latency metrics."""
        metrics = await monitor.get_dashboard_metrics()
        return metrics.latency.dict()

    @router.get("/dlq")
    async def get_dlq():
        """Get DLQ metrics."""
        metrics = await monitor.get_dashboard_metrics()
        return metrics.dlq.dict()

    @router.get("/consumer-lag")
    async def get_consumer_lag():
        """Get consumer lag metrics."""
        metrics = await monitor.get_dashboard_metrics()
        return metrics.consumer_lag.dict()

    @router.get("/distribution")
    async def get_distribution():
        """Get event type distribution."""
        metrics = await monitor.get_dashboard_metrics()
        return metrics.event_distribution.dict()

    @router.get("/health")
    async def get_health():
        """Get system health."""
        return await monitor.health_check()

    @router.get("/metrics", response_class=PlainTextResponse)
    async def get_prometheus_metrics():
        """Get Prometheus metrics."""
        return await monitor.get_prometheus_metrics()

    @router.get("/alerts")
    async def get_alerts(
        acknowledged: Optional[bool] = Query(None),
        severity: Optional[str] = Query(None)
    ):
        """Get alerts."""
        severity_enum = AlertSeverity(severity) if severity else None
        alerts = await monitor.get_alerts(acknowledged, severity_enum)
        return {"alerts": [a.dict() for a in alerts]}

    @router.post("/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert(alert_id: str, user: str = Query(...)):
        """Acknowledge an alert."""
        success = await monitor.acknowledge_alert(alert_id, user)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"message": "Alert acknowledged", "alert_id": alert_id}

    return router
