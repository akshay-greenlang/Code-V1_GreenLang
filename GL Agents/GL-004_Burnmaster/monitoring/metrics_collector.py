"""
GL-004 BURNMASTER Metrics Collector Module

This module provides comprehensive metrics collection for combustion optimization
operations, supporting Prometheus and OpenTelemetry export formats for
integration with standard observability platforms.

Example:
    >>> collector = MetricsCollector()
    >>> collector.record_metric("optimization_cycle_time", 150.5, {"unit": "BNR-001"})
    >>> prometheus_output = collector.export_metrics_prometheus()
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import statistics
import threading
import time
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    HISTOGRAM = "HISTOGRAM"
    SUMMARY = "SUMMARY"


# =============================================================================
# DATA MODELS
# =============================================================================

class DateRange(BaseModel):
    """Date range for metric queries."""

    start: datetime = Field(..., description="Start of date range")
    end: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="End of date range"
    )

    @validator('end')
    def end_after_start(cls, v, values):
        """Validate end is after start."""
        if 'start' in values and v < values['start']:
            raise ValueError("End date must be after start date")
        return v

    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return (self.end - self.start).total_seconds()


class MetricPoint(BaseModel):
    """A single metric data point."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Point timestamp"
    )
    value: float = Field(..., description="Metric value")
    tags: Dict[str, str] = Field(default_factory=dict, description="Metric tags")


class MetricHistory(BaseModel):
    """Historical metric data."""

    metric_name: str = Field(..., description="Metric name")
    metric_type: MetricType = Field(..., description="Metric type")
    period: DateRange = Field(..., description="Time period covered")

    # Data points
    points: List[MetricPoint] = Field(default_factory=list, description="Data points")

    # Statistics
    count: int = Field(default=0, ge=0, description="Number of points")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    avg_value: Optional[float] = Field(None, description="Average value")
    sum_value: Optional[float] = Field(None, description="Sum of values")
    std_dev: Optional[float] = Field(None, description="Standard deviation")

    # Percentiles
    p50: Optional[float] = Field(None, description="50th percentile")
    p90: Optional[float] = Field(None, description="90th percentile")
    p99: Optional[float] = Field(None, description="99th percentile")

    # Metadata
    tags_filter: Optional[Dict[str, str]] = Field(
        None, description="Tags used to filter"
    )


class MetricDefinition(BaseModel):
    """Definition of a metric."""

    name: str = Field(..., description="Metric name")
    metric_type: MetricType = Field(..., description="Metric type")
    description: str = Field(default="", description="Metric description")
    unit: str = Field(default="", description="Unit of measurement")
    labels: List[str] = Field(default_factory=list, description="Label names")
    buckets: Optional[List[float]] = Field(
        None, description="Histogram buckets"
    )


# =============================================================================
# PROMETHEUS-STYLE METRIC CLASSES
# =============================================================================

class Counter:
    """Prometheus-style counter metric."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Initialize counter."""
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **label_values) -> None:
        """Increment counter."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def get(self, **label_values) -> float:
        """Get counter value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def export_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        for key, value in self._values.items():
            if self.labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Gauge:
    """Prometheus-style gauge metric."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Initialize gauge."""
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[Tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **label_values) -> None:
        """Set gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, **label_values) -> None:
        """Increment gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def dec(self, amount: float = 1.0, **label_values) -> None:
        """Decrement gauge."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) - amount

    def get(self, **label_values) -> float:
        """Get gauge value."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        return self._values.get(key, 0)

    def export_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        for key, value in self._values.items():
            if self.labels:
                label_str = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                )
                lines.append(f"{self.name}{{{label_str}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class Histogram:
    """Prometheus-style histogram metric."""

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize histogram."""
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self.labels = labels or []
        self._observations: Dict[Tuple[str, ...], List[float]] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **label_values) -> None:
        """Record an observation."""
        key = tuple(label_values.get(l, "") for l in self.labels)
        with self._lock:
            if key not in self._observations:
                self._observations[key] = []
            self._observations[key].append(value)

    def export_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]

        for key, observations in self._observations.items():
            label_prefix = ""
            if self.labels:
                label_prefix = ",".join(
                    f'{l}="{v}"' for l, v in zip(self.labels, key)
                ) + ","

            # Bucket counts
            for bucket in self.buckets:
                count = sum(1 for o in observations if o <= bucket)
                lines.append(
                    f'{self.name}_bucket{{{label_prefix}le="{bucket}"}} {count}'
                )
            lines.append(
                f'{self.name}_bucket{{{label_prefix}le="+Inf"}} {len(observations)}'
            )

            # Sum and count
            label_part = label_prefix.rstrip(",") if label_prefix else ""
            lines.append(
                f"{self.name}_sum{{{label_part}}} {sum(observations)}"
            )
            lines.append(
                f"{self.name}_count{{{label_part}}} {len(observations)}"
            )

        return "\n".join(lines)


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Comprehensive metrics collection for combustion optimization.

    Collects operational metrics including optimization cycle times,
    recommendation counts, acceptance rates, constraint violations,
    safety margins, and efficiency gains. Supports export to Prometheus
    and OpenTelemetry formats.

    Attributes:
        prefix: Metric name prefix

    Example:
        >>> collector = MetricsCollector(prefix="burnmaster")
        >>> collector.record_metric("optimization_cycle_time", 150.5)
        >>> print(collector.export_metrics_prometheus())
    """

    def __init__(self, prefix: str = "burnmaster"):
        """
        Initialize the MetricsCollector.

        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix
        self._metrics: Dict[str, Any] = {}
        self._metric_history: Dict[str, List[MetricPoint]] = {}
        self._lock = threading.Lock()

        # Initialize standard combustion optimization metrics
        self._initialize_standard_metrics()

        logger.info(f"MetricsCollector initialized with prefix: {prefix}")

    def _initialize_standard_metrics(self) -> None:
        """Initialize standard combustion optimization metrics."""
        # Counters
        self._metrics['optimization_cycles_total'] = Counter(
            f"{self.prefix}_optimization_cycles_total",
            "Total number of optimization cycles executed",
            labels=["unit_id", "status"]
        )
        self._metrics['recommendation_count'] = Counter(
            f"{self.prefix}_recommendations_total",
            "Total number of recommendations generated",
            labels=["unit_id", "type"]
        )
        self._metrics['constraint_violations_total'] = Counter(
            f"{self.prefix}_constraint_violations_total",
            "Total constraint violations detected",
            labels=["unit_id", "constraint_type"]
        )

        # Gauges
        self._metrics['acceptance_rate'] = Gauge(
            f"{self.prefix}_acceptance_rate",
            "Recommendation acceptance rate (0-1)",
            labels=["unit_id"]
        )
        self._metrics['safety_margin'] = Gauge(
            f"{self.prefix}_safety_margin",
            "Current safety margin percentage",
            labels=["unit_id", "parameter"]
        )
        self._metrics['efficiency_gain'] = Gauge(
            f"{self.prefix}_efficiency_gain_percent",
            "Efficiency gain from optimization",
            labels=["unit_id"]
        )
        self._metrics['air_fuel_ratio'] = Gauge(
            f"{self.prefix}_air_fuel_ratio",
            "Current air-fuel ratio",
            labels=["unit_id"]
        )
        self._metrics['nox_ppm'] = Gauge(
            f"{self.prefix}_nox_ppm",
            "Current NOx emissions in ppm",
            labels=["unit_id"]
        )
        self._metrics['co_ppm'] = Gauge(
            f"{self.prefix}_co_ppm",
            "Current CO emissions in ppm",
            labels=["unit_id"]
        )
        self._metrics['o2_percent'] = Gauge(
            f"{self.prefix}_o2_percent",
            "Current O2 percentage",
            labels=["unit_id"]
        )
        self._metrics['combustion_efficiency'] = Gauge(
            f"{self.prefix}_combustion_efficiency_percent",
            "Current combustion efficiency",
            labels=["unit_id"]
        )
        self._metrics['flame_stability'] = Gauge(
            f"{self.prefix}_flame_stability_score",
            "Flame stability score (0-1)",
            labels=["unit_id"]
        )

        # Histograms
        self._metrics['optimization_cycle_time'] = Histogram(
            f"{self.prefix}_optimization_cycle_time_seconds",
            "Optimization cycle execution time",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            labels=["unit_id"]
        )
        self._metrics['recommendation_latency'] = Histogram(
            f"{self.prefix}_recommendation_latency_seconds",
            "Time to generate recommendation",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            labels=["unit_id"]
        )

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags/labels
        """
        tags = tags or {}

        with self._lock:
            # Record to appropriate metric type
            if name in self._metrics:
                metric = self._metrics[name]
                if isinstance(metric, Counter):
                    metric.inc(value, **tags)
                elif isinstance(metric, Gauge):
                    metric.set(value, **tags)
                elif isinstance(metric, Histogram):
                    metric.observe(value, **tags)

            # Store in history
            if name not in self._metric_history:
                self._metric_history[name] = []

            self._metric_history[name].append(MetricPoint(
                timestamp=datetime.now(timezone.utc),
                value=value,
                tags=tags
            ))

            # Limit history size (keep last 10000 points per metric)
            if len(self._metric_history[name]) > 10000:
                self._metric_history[name] = self._metric_history[name][-10000:]

        logger.debug(f"Recorded metric {name}={value}, tags={tags}")

    def get_metric_history(
        self,
        name: str,
        period: DateRange,
        tags_filter: Optional[Dict[str, str]] = None
    ) -> MetricHistory:
        """
        Get historical data for a metric.

        Args:
            name: Metric name
            period: Date range to query
            tags_filter: Optional tags to filter by

        Returns:
            MetricHistory with data points and statistics
        """
        points = []

        if name in self._metric_history:
            for point in self._metric_history[name]:
                # Check time range
                if not (period.start <= point.timestamp <= period.end):
                    continue

                # Check tags filter
                if tags_filter:
                    if not all(
                        point.tags.get(k) == v
                        for k, v in tags_filter.items()
                    ):
                        continue

                points.append(point)

        # Calculate statistics
        values = [p.value for p in points]

        history = MetricHistory(
            metric_name=name,
            metric_type=self._get_metric_type(name),
            period=period,
            points=points,
            count=len(points),
            tags_filter=tags_filter,
        )

        if values:
            history.min_value = min(values)
            history.max_value = max(values)
            history.avg_value = statistics.mean(values)
            history.sum_value = sum(values)

            if len(values) > 1:
                history.std_dev = statistics.stdev(values)

            # Calculate percentiles
            sorted_values = sorted(values)
            n = len(sorted_values)
            history.p50 = sorted_values[int(n * 0.5)]
            history.p90 = sorted_values[int(n * 0.9)] if n >= 10 else None
            history.p99 = sorted_values[int(n * 0.99)] if n >= 100 else None

        logger.info(
            f"Retrieved {len(points)} points for metric {name} "
            f"in range {period.start} to {period.end}"
        )

        return history

    def _get_metric_type(self, name: str) -> MetricType:
        """Get the type of a metric."""
        if name in self._metrics:
            metric = self._metrics[name]
            if isinstance(metric, Counter):
                return MetricType.COUNTER
            elif isinstance(metric, Gauge):
                return MetricType.GAUGE
            elif isinstance(metric, Histogram):
                return MetricType.HISTOGRAM
        return MetricType.GAUGE  # Default

    def export_metrics_prometheus(self) -> str:
        """
        Export all metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        output_parts = []

        for metric in self._metrics.values():
            if hasattr(metric, 'export_prometheus'):
                exported = metric.export_prometheus()
                if exported.strip():
                    output_parts.append(exported)

        output = "\n\n".join(output_parts)

        logger.debug(f"Exported {len(output_parts)} metrics to Prometheus format")

        return output

    def export_metrics_opentelemetry(self) -> Dict[str, Any]:
        """
        Export all metrics in OpenTelemetry format.

        Returns:
            Dictionary with OpenTelemetry metric data
        """
        now = datetime.now(timezone.utc)
        resource = {
            "service.name": "burnmaster",
            "service.version": "1.0.0",
        }

        metrics_data = []

        for name, metric in self._metrics.items():
            metric_data = {
                "name": metric.name if hasattr(metric, 'name') else name,
                "description": getattr(metric, 'description', ''),
                "unit": "",  # Would be set from definition
            }

            if isinstance(metric, Counter):
                metric_data["type"] = "sum"
                metric_data["aggregation_temporality"] = "cumulative"
                metric_data["is_monotonic"] = True
                metric_data["data_points"] = [
                    {
                        "attributes": dict(zip(metric.labels, key)),
                        "value": value,
                        "time_unix_nano": int(now.timestamp() * 1e9),
                    }
                    for key, value in metric._values.items()
                ]

            elif isinstance(metric, Gauge):
                metric_data["type"] = "gauge"
                metric_data["data_points"] = [
                    {
                        "attributes": dict(zip(metric.labels, key)),
                        "value": value,
                        "time_unix_nano": int(now.timestamp() * 1e9),
                    }
                    for key, value in metric._values.items()
                ]

            elif isinstance(metric, Histogram):
                metric_data["type"] = "histogram"
                metric_data["aggregation_temporality"] = "cumulative"
                metric_data["data_points"] = []

                for key, observations in metric._observations.items():
                    bucket_counts = [
                        sum(1 for o in observations if o <= b)
                        for b in metric.buckets
                    ]
                    bucket_counts.append(len(observations))  # +Inf bucket

                    metric_data["data_points"].append({
                        "attributes": dict(zip(metric.labels, key)),
                        "count": len(observations),
                        "sum": sum(observations),
                        "bucket_counts": bucket_counts,
                        "explicit_bounds": metric.buckets,
                        "time_unix_nano": int(now.timestamp() * 1e9),
                    })

            metrics_data.append(metric_data)

        result = {
            "resource_metrics": [{
                "resource": {"attributes": resource},
                "scope_metrics": [{
                    "scope": {
                        "name": "burnmaster.monitoring",
                        "version": "1.0.0",
                    },
                    "metrics": metrics_data,
                }],
            }],
        }

        logger.debug(f"Exported {len(metrics_data)} metrics to OpenTelemetry format")

        return result

    # Convenience methods for standard combustion metrics

    def record_optimization_cycle(
        self,
        unit_id: str,
        duration_seconds: float,
        status: str = "success"
    ) -> None:
        """Record completion of an optimization cycle."""
        self._metrics['optimization_cycles_total'].inc(
            1, unit_id=unit_id, status=status
        )
        self._metrics['optimization_cycle_time'].observe(
            duration_seconds, unit_id=unit_id
        )
        self.record_metric(
            'optimization_cycle_time',
            duration_seconds,
            {'unit_id': unit_id, 'status': status}
        )

    def record_recommendation(
        self,
        unit_id: str,
        recommendation_type: str,
        latency_seconds: float
    ) -> None:
        """Record a generated recommendation."""
        self._metrics['recommendation_count'].inc(
            1, unit_id=unit_id, type=recommendation_type
        )
        self._metrics['recommendation_latency'].observe(
            latency_seconds, unit_id=unit_id
        )
        self.record_metric(
            'recommendation_count',
            1,
            {'unit_id': unit_id, 'type': recommendation_type}
        )

    def update_acceptance_rate(self, unit_id: str, rate: float) -> None:
        """Update recommendation acceptance rate."""
        self._metrics['acceptance_rate'].set(rate, unit_id=unit_id)
        self.record_metric('acceptance_rate', rate, {'unit_id': unit_id})

    def record_constraint_violation(
        self,
        unit_id: str,
        constraint_type: str
    ) -> None:
        """Record a constraint violation."""
        self._metrics['constraint_violations_total'].inc(
            1, unit_id=unit_id, constraint_type=constraint_type
        )
        self.record_metric(
            'constraint_violations',
            1,
            {'unit_id': unit_id, 'constraint_type': constraint_type}
        )

    def update_safety_margin(
        self,
        unit_id: str,
        parameter: str,
        margin_percent: float
    ) -> None:
        """Update safety margin for a parameter."""
        self._metrics['safety_margin'].set(
            margin_percent, unit_id=unit_id, parameter=parameter
        )
        self.record_metric(
            'safety_margin',
            margin_percent,
            {'unit_id': unit_id, 'parameter': parameter}
        )

    def update_efficiency_gain(self, unit_id: str, gain_percent: float) -> None:
        """Update efficiency gain from optimization."""
        self._metrics['efficiency_gain'].set(gain_percent, unit_id=unit_id)
        self.record_metric('efficiency_gain', gain_percent, {'unit_id': unit_id})

    def update_combustion_state(
        self,
        unit_id: str,
        air_fuel_ratio: float,
        o2_percent: float,
        co_ppm: float,
        nox_ppm: float,
        efficiency: float,
        stability: float
    ) -> None:
        """Update all combustion state metrics at once."""
        self._metrics['air_fuel_ratio'].set(air_fuel_ratio, unit_id=unit_id)
        self._metrics['o2_percent'].set(o2_percent, unit_id=unit_id)
        self._metrics['co_ppm'].set(co_ppm, unit_id=unit_id)
        self._metrics['nox_ppm'].set(nox_ppm, unit_id=unit_id)
        self._metrics['combustion_efficiency'].set(efficiency, unit_id=unit_id)
        self._metrics['flame_stability'].set(stability, unit_id=unit_id)

        # Record to history
        tags = {'unit_id': unit_id}
        self.record_metric('air_fuel_ratio', air_fuel_ratio, tags)
        self.record_metric('o2_percent', o2_percent, tags)
        self.record_metric('co_ppm', co_ppm, tags)
        self.record_metric('nox_ppm', nox_ppm, tags)
        self.record_metric('combustion_efficiency', efficiency, tags)
        self.record_metric('flame_stability', stability, tags)

    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        total_points = sum(len(h) for h in self._metric_history.values())
        return {
            'total_metrics': len(self._metrics),
            'total_history_points': total_points,
            'metrics_with_history': len(self._metric_history),
        }
