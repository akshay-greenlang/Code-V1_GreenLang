"""
GreenLang Framework - Prometheus Metrics Module
================================================

Provides Prometheus-compatible metrics for GreenLang agents including
counters, histograms, gauges, and summaries with custom registry support.

Features:
- Counter: Monotonically increasing values (e.g., calculations performed)
- Histogram: Distribution of values with configurable buckets (e.g., latency)
- Gauge: Current value that can go up or down (e.g., queue depth)
- Summary: Statistical distribution with quantiles (e.g., response sizes)
- Custom metrics registry with namespace support
- Prometheus exposition format export
- Pre-built GreenLang agent metrics

Example:
    >>> from greenlang_observability.metrics import MetricsRegistry, Counter, Histogram
    >>>
    >>> # Create registry
    >>> registry = MetricsRegistry(namespace="greenlang")
    >>>
    >>> # Create metrics
    >>> calc_counter = registry.counter(
    ...     "calculations_total",
    ...     "Total number of calculations performed",
    ...     labels=["agent_id", "calculation_type"],
    ... )
    >>>
    >>> # Record metric
    >>> calc_counter.inc(labels={"agent_id": "GL-006", "calculation_type": "pinch"})

Version: 1.0.0
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type variable for generic metric types
T = TypeVar("T", bound="Metric")


class MetricType(Enum):
    """Types of Prometheus metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricsConfig:
    """
    Configuration for metrics infrastructure.

    Attributes:
        namespace: Metric namespace prefix
        subsystem: Metric subsystem prefix
        default_labels: Labels added to all metrics
        histogram_buckets: Default histogram buckets
        summary_quantiles: Default summary quantiles
        enable_default_metrics: Whether to create default GreenLang metrics
        exposition_port: Port for Prometheus scraping (0 to disable)
    """

    namespace: str = "greenlang"
    subsystem: str = ""
    default_labels: Dict[str, str] = field(default_factory=dict)
    histogram_buckets: Tuple[float, ...] = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.5, 5.0, 7.5, 10.0, float("inf"),
    )
    summary_quantiles: Tuple[Tuple[float, float], ...] = (
        (0.5, 0.05),   # 50th percentile, 5% error
        (0.9, 0.01),   # 90th percentile, 1% error
        (0.99, 0.001), # 99th percentile, 0.1% error
    )
    enable_default_metrics: bool = True
    exposition_port: int = 9090

    @classmethod
    def from_env(cls) -> MetricsConfig:
        """Create configuration from environment variables."""
        return cls(
            namespace=os.getenv("METRICS_NAMESPACE", "greenlang"),
            subsystem=os.getenv("METRICS_SUBSYSTEM", ""),
            exposition_port=int(os.getenv("METRICS_PORT", "9090")),
            enable_default_metrics=os.getenv(
                "METRICS_ENABLE_DEFAULTS", "true"
            ).lower() == "true",
        )


@dataclass
class LabelSet:
    """Represents a set of label key-value pairs."""

    labels: Dict[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Make hashable for use as dict key."""
        return hash(tuple(sorted(self.labels.items())))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, LabelSet):
            return False
        return self.labels == other.labels

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus label format."""
        if not self.labels:
            return ""
        label_strs = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(label_strs) + "}"


class Metric(ABC):
    """Base class for all metric types."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        namespace: str = "",
        subsystem: str = "",
    ):
        """
        Initialize metric.

        Args:
            name: Metric name
            description: Human-readable description
            labels: List of label names
            namespace: Metric namespace
            subsystem: Metric subsystem
        """
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.namespace = namespace
        self.subsystem = subsystem
        self._values: Dict[LabelSet, Any] = {}
        self._lock = threading.Lock()

    @property
    def full_name(self) -> str:
        """Get fully qualified metric name."""
        parts = [p for p in [self.namespace, self.subsystem, self.name] if p]
        return "_".join(parts)

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Return the metric type."""
        pass

    def _validate_labels(self, labels: Optional[Dict[str, str]]) -> LabelSet:
        """Validate and create label set."""
        if labels is None:
            labels = {}

        # Check all required labels are provided
        missing = set(self.label_names) - set(labels.keys())
        if missing:
            raise ValueError(f"Missing required labels: {missing}")

        # Check no extra labels
        extra = set(labels.keys()) - set(self.label_names)
        if extra:
            raise ValueError(f"Unknown labels: {extra}")

        return LabelSet(labels)

    @abstractmethod
    def to_prometheus_format(self) -> str:
        """Convert metric to Prometheus exposition format."""
        pass

    def clear(self) -> None:
        """Clear all values."""
        with self._lock:
            self._values.clear()


class Counter(Metric):
    """
    Counter metric - monotonically increasing value.

    Counters are cumulative metrics that only increase (or reset to zero on restart).
    Use for counting events like calculations performed or errors.

    Example:
        >>> counter = Counter("calculations_total", "Total calculations")
        >>> counter.inc()  # Increment by 1
        >>> counter.inc(5)  # Increment by 5
    """

    @property
    def metric_type(self) -> MetricType:
        """Return counter type."""
        return MetricType.COUNTER

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment counter.

        Args:
            value: Amount to increment (must be positive)
            labels: Label values
        """
        if value < 0:
            raise ValueError("Counter can only be incremented")

        label_set = self._validate_labels(labels)

        with self._lock:
            current = self._values.get(label_set, 0.0)
            self._values[label_set] = current + value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        label_set = self._validate_labels(labels)
        with self._lock:
            return self._values.get(label_set, 0.0)

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus format."""
        lines = [
            f"# HELP {self.full_name} {self.description}",
            f"# TYPE {self.full_name} counter",
        ]

        with self._lock:
            for label_set, value in self._values.items():
                label_str = label_set.to_prometheus_format()
                lines.append(f"{self.full_name}{label_str} {value}")

        return "\n".join(lines)


class Gauge(Metric):
    """
    Gauge metric - value that can go up and down.

    Gauges represent a single numerical value that can increase or decrease.
    Use for values like queue depth, active connections, or temperature.

    Example:
        >>> gauge = Gauge("queue_depth", "Current queue depth")
        >>> gauge.set(42)
        >>> gauge.inc()
        >>> gauge.dec(5)
    """

    @property
    def metric_type(self) -> MetricType:
        """Return gauge type."""
        return MetricType.GAUGE

    def set(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge to a specific value."""
        label_set = self._validate_labels(labels)
        with self._lock:
            self._values[label_set] = value

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment gauge."""
        label_set = self._validate_labels(labels)
        with self._lock:
            current = self._values.get(label_set, 0.0)
            self._values[label_set] = current + value

    def dec(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Decrement gauge."""
        label_set = self._validate_labels(labels)
        with self._lock:
            current = self._values.get(label_set, 0.0)
            self._values[label_set] = current - value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        label_set = self._validate_labels(labels)
        with self._lock:
            return self._values.get(label_set, 0.0)

    def set_to_current_time(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge to current Unix timestamp."""
        self.set(time.time(), labels)

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus format."""
        lines = [
            f"# HELP {self.full_name} {self.description}",
            f"# TYPE {self.full_name} gauge",
        ]

        with self._lock:
            for label_set, value in self._values.items():
                label_str = label_set.to_prometheus_format()
                lines.append(f"{self.full_name}{label_str} {value}")

        return "\n".join(lines)


@dataclass
class HistogramBucket:
    """Single histogram bucket."""

    upper_bound: float
    count: int = 0


class Histogram(Metric):
    """
    Histogram metric - distribution of values.

    Histograms sample observations and count them in configurable buckets.
    Use for measuring latencies, request sizes, etc.

    Example:
        >>> histogram = Histogram(
        ...     "calculation_duration_seconds",
        ...     "Calculation duration",
        ...     buckets=[0.1, 0.5, 1.0, 2.5, 5.0],
        ... )
        >>> histogram.observe(0.35)  # Record a value
    """

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
        2.5, 5.0, 7.5, 10.0, float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
        namespace: str = "",
        subsystem: str = "",
    ):
        """Initialize histogram with bucket boundaries."""
        super().__init__(name, description, labels, namespace, subsystem)
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)

        # Ensure +Inf bucket exists
        if self.buckets[-1] != float("inf"):
            self.buckets = tuple(list(self.buckets) + [float("inf")])
        else:
            self.buckets = tuple(self.buckets)

    @property
    def metric_type(self) -> MetricType:
        """Return histogram type."""
        return MetricType.HISTOGRAM

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record an observation.

        Args:
            value: The value to observe
            labels: Label values
        """
        label_set = self._validate_labels(labels)

        with self._lock:
            if label_set not in self._values:
                self._values[label_set] = {
                    "buckets": {b: 0 for b in self.buckets},
                    "sum": 0.0,
                    "count": 0,
                }

            data = self._values[label_set]
            data["sum"] += value
            data["count"] += 1

            for bucket in self.buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def get_sample_count(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> int:
        """Get total count of observations."""
        label_set = self._validate_labels(labels)
        with self._lock:
            if label_set in self._values:
                return self._values[label_set]["count"]
            return 0

    def get_sample_sum(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get sum of all observations."""
        label_set = self._validate_labels(labels)
        with self._lock:
            if label_set in self._values:
                return self._values[label_set]["sum"]
            return 0.0

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus format."""
        lines = [
            f"# HELP {self.full_name} {self.description}",
            f"# TYPE {self.full_name} histogram",
        ]

        with self._lock:
            for label_set, data in self._values.items():
                base_labels = label_set.labels.copy()

                # Output buckets
                cumulative = 0
                for bucket in self.buckets:
                    cumulative += data["buckets"].get(bucket, 0)
                    bucket_labels = base_labels.copy()
                    bucket_labels["le"] = (
                        "+Inf" if math.isinf(bucket) else str(bucket)
                    )
                    label_str = LabelSet(bucket_labels).to_prometheus_format()
                    lines.append(f"{self.full_name}_bucket{label_str} {cumulative}")

                # Output sum and count
                label_str = label_set.to_prometheus_format()
                lines.append(f"{self.full_name}_sum{label_str} {data['sum']}")
                lines.append(f"{self.full_name}_count{label_str} {data['count']}")

        return "\n".join(lines)

    def time(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> "HistogramTimer":
        """Context manager for timing code blocks."""
        return HistogramTimer(self, labels)


class HistogramTimer:
    """Context manager for timing with histograms."""

    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize timer."""
        self.histogram = histogram
        self.labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "HistogramTimer":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record observation."""
        if self._start is not None:
            duration = time.perf_counter() - self._start
            self.histogram.observe(duration, self.labels)


class Summary(Metric):
    """
    Summary metric - statistical distribution with quantiles.

    Summaries calculate streaming quantiles over a sliding time window.
    Use for tracking response sizes, percentiles, etc.

    Note: Unlike histograms, quantiles are calculated client-side.

    Example:
        >>> summary = Summary(
        ...     "response_size_bytes",
        ...     "Response size distribution",
        ... )
        >>> summary.observe(1024)
    """

    DEFAULT_QUANTILES = (
        (0.5, 0.05),
        (0.9, 0.01),
        (0.99, 0.001),
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        quantiles: Optional[Tuple[Tuple[float, float], ...]] = None,
        max_age_seconds: int = 600,
        age_buckets: int = 5,
        namespace: str = "",
        subsystem: str = "",
    ):
        """Initialize summary."""
        super().__init__(name, description, labels, namespace, subsystem)
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.max_age_seconds = max_age_seconds
        self.age_buckets = age_buckets

    @property
    def metric_type(self) -> MetricType:
        """Return summary type."""
        return MetricType.SUMMARY

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an observation."""
        label_set = self._validate_labels(labels)

        with self._lock:
            if label_set not in self._values:
                self._values[label_set] = {
                    "observations": [],
                    "sum": 0.0,
                    "count": 0,
                }

            data = self._values[label_set]
            data["observations"].append(value)
            data["sum"] += value
            data["count"] += 1

            # Keep observations bounded (simplified sliding window)
            max_observations = 10000
            if len(data["observations"]) > max_observations:
                data["observations"] = data["observations"][-max_observations:]

    def _calculate_quantile(
        self,
        observations: List[float],
        quantile: float,
    ) -> float:
        """Calculate quantile from observations."""
        if not observations:
            return 0.0

        sorted_obs = sorted(observations)
        n = len(sorted_obs)
        idx = int(quantile * n)
        idx = max(0, min(idx, n - 1))
        return sorted_obs[idx]

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus format."""
        lines = [
            f"# HELP {self.full_name} {self.description}",
            f"# TYPE {self.full_name} summary",
        ]

        with self._lock:
            for label_set, data in self._values.items():
                base_labels = label_set.labels.copy()
                observations = data["observations"]

                # Output quantiles
                for quantile, _ in self.quantiles:
                    value = self._calculate_quantile(observations, quantile)
                    quantile_labels = base_labels.copy()
                    quantile_labels["quantile"] = str(quantile)
                    label_str = LabelSet(quantile_labels).to_prometheus_format()
                    lines.append(f"{self.full_name}{label_str} {value}")

                # Output sum and count
                label_str = label_set.to_prometheus_format()
                lines.append(f"{self.full_name}_sum{label_str} {data['sum']}")
                lines.append(f"{self.full_name}_count{label_str} {data['count']}")

        return "\n".join(lines)


class MetricsRegistry:
    """
    Registry for managing metrics.

    Provides centralized metric registration, collection, and export.
    Supports multiple metric types and Prometheus exposition format.

    Example:
        >>> registry = MetricsRegistry(namespace="greenlang")
        >>> counter = registry.counter("requests_total", "Total requests")
        >>> histogram = registry.histogram("latency_seconds", "Request latency")
        >>> print(registry.expose())  # Prometheus format
    """

    _default_registry: Optional[MetricsRegistry] = None

    def __init__(
        self,
        namespace: str = "greenlang",
        subsystem: str = "",
        config: Optional[MetricsConfig] = None,
    ):
        """
        Initialize registry.

        Args:
            namespace: Metric namespace
            subsystem: Metric subsystem
            config: Full configuration object
        """
        if config:
            self.config = config
        else:
            self.config = MetricsConfig(
                namespace=namespace,
                subsystem=subsystem,
            )

        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

        # Set as default registry if none exists
        if MetricsRegistry._default_registry is None:
            MetricsRegistry._default_registry = self

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """
        Create and register a counter.

        Args:
            name: Metric name
            description: Human-readable description
            labels: List of label names

        Returns:
            Counter metric
        """
        metric = Counter(
            name=name,
            description=description,
            labels=labels,
            namespace=self.config.namespace,
            subsystem=self.config.subsystem,
        )
        return self._register(metric)

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """
        Create and register a gauge.

        Args:
            name: Metric name
            description: Human-readable description
            labels: List of label names

        Returns:
            Gauge metric
        """
        metric = Gauge(
            name=name,
            description=description,
            labels=labels,
            namespace=self.config.namespace,
            subsystem=self.config.subsystem,
        )
        return self._register(metric)

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """
        Create and register a histogram.

        Args:
            name: Metric name
            description: Human-readable description
            labels: List of label names
            buckets: Histogram bucket boundaries

        Returns:
            Histogram metric
        """
        metric = Histogram(
            name=name,
            description=description,
            labels=labels,
            buckets=buckets or self.config.histogram_buckets,
            namespace=self.config.namespace,
            subsystem=self.config.subsystem,
        )
        return self._register(metric)

    def summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        quantiles: Optional[Tuple[Tuple[float, float], ...]] = None,
    ) -> Summary:
        """
        Create and register a summary.

        Args:
            name: Metric name
            description: Human-readable description
            labels: List of label names
            quantiles: Quantile definitions

        Returns:
            Summary metric
        """
        metric = Summary(
            name=name,
            description=description,
            labels=labels,
            quantiles=quantiles or self.config.summary_quantiles,
            namespace=self.config.namespace,
            subsystem=self.config.subsystem,
        )
        return self._register(metric)

    def _register(self, metric: T) -> T:
        """Register a metric."""
        with self._lock:
            if metric.full_name in self._metrics:
                logger.warning(
                    f"Metric {metric.full_name} already registered, returning existing"
                )
                return self._metrics[metric.full_name]  # type: ignore

            self._metrics[metric.full_name] = metric
            return metric

    def get(self, name: str) -> Optional[Metric]:
        """Get a registered metric by name."""
        full_name = "_".join(
            [p for p in [self.config.namespace, self.config.subsystem, name] if p]
        )
        with self._lock:
            return self._metrics.get(full_name)

    def expose(self) -> str:
        """
        Export all metrics in Prometheus exposition format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        with self._lock:
            for metric in self._metrics.values():
                lines.append(metric.to_prometheus_format())

        return "\n\n".join(lines)

    def clear(self) -> None:
        """Clear all metric values (but keep registrations)."""
        with self._lock:
            for metric in self._metrics.values():
                metric.clear()

    def unregister(self, name: str) -> bool:
        """Unregister a metric."""
        full_name = "_".join(
            [p for p in [self.config.namespace, self.config.subsystem, name] if p]
        )
        with self._lock:
            if full_name in self._metrics:
                del self._metrics[full_name]
                return True
            return False


def get_default_registry() -> MetricsRegistry:
    """Get or create the default metrics registry."""
    if MetricsRegistry._default_registry is None:
        MetricsRegistry._default_registry = MetricsRegistry()
    return MetricsRegistry._default_registry


# Pre-defined GreenLang metrics


def _create_default_metrics(registry: MetricsRegistry) -> None:
    """Create default GreenLang metrics."""
    # These are created lazily when accessed
    pass


# Lazy metric accessors for common GreenLang metrics


class _LazyCounter:
    """Lazy counter that creates metric on first use."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels
        self._counter: Optional[Counter] = None

    def _get_counter(self) -> Counter:
        if self._counter is None:
            registry = get_default_registry()
            self._counter = registry.counter(
                self.name, self.description, self.labels
            )
        return self._counter

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._get_counter().inc(value, labels)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        return self._get_counter().get(labels)


class _LazyHistogram:
    """Lazy histogram that creates metric on first use."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels
        self.buckets = buckets
        self._histogram: Optional[Histogram] = None

    def _get_histogram(self) -> Histogram:
        if self._histogram is None:
            registry = get_default_registry()
            self._histogram = registry.histogram(
                self.name, self.description, self.labels, self.buckets
            )
        return self._histogram

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._get_histogram().observe(value, labels)

    def time(
        self,
        labels: Optional[Dict[str, str]] = None,
    ) -> HistogramTimer:
        return self._get_histogram().time(labels)


class _LazyGauge:
    """Lazy gauge that creates metric on first use."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels
        self._gauge: Optional[Gauge] = None

    def _get_gauge(self) -> Gauge:
        if self._gauge is None:
            registry = get_default_registry()
            self._gauge = registry.gauge(self.name, self.description, self.labels)
        return self._gauge

    def set(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._get_gauge().set(value, labels)

    def inc(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._get_gauge().inc(value, labels)

    def dec(
        self,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._get_gauge().dec(value, labels)

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        return self._get_gauge().get(labels)


class _LazySummary:
    """Lazy summary that creates metric on first use."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.labels = labels
        self._summary: Optional[Summary] = None

    def _get_summary(self) -> Summary:
        if self._summary is None:
            registry = get_default_registry()
            self._summary = registry.summary(
                self.name, self.description, self.labels
            )
        return self._summary

    def observe(
        self,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self._get_summary().observe(value, labels)


# Pre-defined GreenLang agent metrics

# Counter for calculations performed
calculation_counter = _LazyCounter(
    name="calculations_total",
    description="Total number of calculations performed",
    labels=["agent_id", "calculation_type", "status"],
)

# Histogram for calculation latency
calculation_latency = _LazyHistogram(
    name="calculation_duration_seconds",
    description="Calculation duration in seconds",
    labels=["agent_id", "calculation_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
)

# Gauge for queue depth
queue_depth = _LazyGauge(
    name="queue_depth",
    description="Current number of items in processing queue",
    labels=["agent_id", "queue_name"],
)

# Gauge for active tasks
active_tasks = _LazyGauge(
    name="active_tasks",
    description="Number of currently active tasks",
    labels=["agent_id", "task_type"],
)

# Summary for response sizes
response_size = _LazySummary(
    name="response_size_bytes",
    description="Size of agent responses in bytes",
    labels=["agent_id", "endpoint"],
)
