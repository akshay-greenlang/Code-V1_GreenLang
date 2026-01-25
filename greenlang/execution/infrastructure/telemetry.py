"""
Telemetry Collector
===================

Metrics collection and aggregation for GreenLang.

Author: Infrastructure Team
Created: 2025-11-21
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque

from greenlang.infrastructure.base import BaseInfrastructureComponent, InfrastructureConfig

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricAggregation:
    """Aggregated metric statistics."""
    name: str
    count: int
    sum: float
    mean: float
    min: float
    max: float
    stddev: Optional[float] = None
    percentiles: Dict[int, float] = field(default_factory=dict)


class TelemetryCollector(BaseInfrastructureComponent):
    """
    Collects and aggregates metrics for monitoring and observability.

    Supports counters, gauges, histograms, and timers with tagging.
    """

    def __init__(self, config: Optional[InfrastructureConfig] = None,
                 buffer_size: int = 10000):
        """Initialize telemetry collector."""
        super().__init__(config or InfrastructureConfig(component_name="TelemetryCollector"))
        self.buffer_size = buffer_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)

    def _initialize(self) -> None:
        """Initialize telemetry resources."""
        logger.info(f"TelemetryCollector initialized with buffer_size={self.buffer_size}")

    def start(self) -> None:
        """Start the telemetry collector."""
        self.status = self.status.RUNNING
        logger.info("TelemetryCollector started")

    def stop(self) -> None:
        """Stop the telemetry collector."""
        self.status = self.status.STOPPED
        logger.info("TelemetryCollector stopped")

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        self.update_activity()

        metric = Metric(
            name=name,
            value=value,
            type=self._infer_metric_type(name),
            timestamp=datetime.now(),
            tags=tags or {}
        )

        # Store in buffer
        metric_key = self._make_metric_key(name, tags)
        self.metrics[metric_key].append(metric)

        # Update type-specific storage
        if metric.type == MetricType.COUNTER:
            self.counters[metric_key] += value
        elif metric.type == MetricType.GAUGE:
            self.gauges[metric_key] = value
        elif metric.type == MetricType.TIMER:
            self.timers[metric_key].append(value)

        logger.debug(f"Recorded metric: {name}={value} with tags={tags}")

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Amount to increment
            tags: Optional tags
        """
        metric_key = self._make_metric_key(name, tags)
        self.counters[metric_key] += value
        self.record(name, value, tags)

    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.

        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags
        """
        metric_key = self._make_metric_key(name, tags)
        self.gauges[metric_key] = value
        self.record(name, value, tags)

    def timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timer metric.

        Args:
            name: Timer name
            duration: Duration in seconds
            tags: Optional tags
        """
        metric_key = self._make_metric_key(name, tags)
        self.timers[metric_key].append(duration)
        self.record(name, duration, tags)

    def get_metrics(self) -> List[Metric]:
        """
        Get all collected metrics.

        Returns:
            List of all metrics in buffer
        """
        all_metrics = []
        for metric_list in self.metrics.values():
            all_metrics.extend(metric_list)
        return sorted(all_metrics, key=lambda m: m.timestamp)

    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        metric_key = self._make_metric_key(name, tags)
        return self.counters.get(metric_key, 0.0)

    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value."""
        metric_key = self._make_metric_key(name, tags)
        return self.gauges.get(metric_key)

    def get_aggregation(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[MetricAggregation]:
        """
        Get aggregated statistics for a metric.

        Args:
            name: Metric name
            tags: Optional tags

        Returns:
            Aggregated statistics or None if no data
        """
        metric_key = self._make_metric_key(name, tags)
        metrics = self.metrics.get(metric_key, [])

        if not metrics:
            return None

        values = [m.value for m in metrics]

        # Calculate percentiles
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            if len(values) > 0:
                sorted_values = sorted(values)
                idx = int(len(sorted_values) * (p / 100))
                percentiles[p] = sorted_values[min(idx, len(sorted_values) - 1)]

        return MetricAggregation(
            name=name,
            count=len(values),
            sum=sum(values),
            mean=statistics.mean(values) if values else 0,
            min=min(values) if values else 0,
            max=max(values) if values else 0,
            stddev=statistics.stdev(values) if len(values) > 1 else None,
            percentiles=percentiles
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.

        Returns:
            Summary statistics for all metrics
        """
        summary = {
            "total_metrics": sum(len(m) for m in self.metrics.values()),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "aggregations": {}
        }

        # Add aggregations for each metric
        for metric_key in self.metrics.keys():
            name = metric_key.split(':')[0]
            agg = self.get_aggregation(name)
            if agg:
                summary["aggregations"][metric_key] = {
                    "count": agg.count,
                    "mean": agg.mean,
                    "min": agg.min,
                    "max": agg.max,
                    "p50": agg.percentiles.get(50),
                    "p95": agg.percentiles.get(95),
                    "p99": agg.percentiles.get(99)
                }

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.timers.clear()
        logger.info("All metrics reset")

    def _infer_metric_type(self, name: str) -> MetricType:
        """Infer metric type from name."""
        if any(suffix in name for suffix in ['_count', '_total', '_requests']):
            return MetricType.COUNTER
        elif any(suffix in name for suffix in ['_gauge', '_current', '_size']):
            return MetricType.GAUGE
        elif any(suffix in name for suffix in ['_time', '_duration', '_latency']):
            return MetricType.TIMER
        else:
            return MetricType.HISTOGRAM

    def _make_metric_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create unique key for metric with tags."""
        if not tags:
            return name

        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"