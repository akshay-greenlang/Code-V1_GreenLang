# -*- coding: utf-8 -*-
"""
Prometheus Metrics Module for GL-014 EXCHANGER-PRO.

This module provides comprehensive Prometheus metrics collection for:
- Request metrics (latency, count, size)
- Calculator metrics (duration, cache, fouling, health)
- Integration metrics (connector latency, errors, throughput)
- Business metrics (exchangers, schedules, savings, alerts)

Implements production-grade metric registration, label management,
histogram bucket configuration, and metric export endpoints.

Example:
    >>> from monitoring.metrics import MetricsCollector, get_metrics_collector
    >>> collector = get_metrics_collector()
    >>> collector.record_request_latency("/api/v1/analyze", "POST", 0.125)
    >>> collector.increment_calculation_count("fouling", "success")

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
import threading
import time
import logging
import re

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC TYPE ENUMERATION
# =============================================================================

class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# =============================================================================
# HISTOGRAM BUCKET CONFIGURATIONS
# =============================================================================

# Request latency buckets (in seconds)
REQUEST_LATENCY_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
    2.5, 5.0, 7.5, 10.0, float("inf")
)

# Calculation duration buckets (in seconds)
CALCULATION_DURATION_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0, 60.0, float("inf")
)

# Connector latency buckets (in seconds)
CONNECTOR_LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    30.0, 60.0, 120.0, float("inf")
)

# Request/response size buckets (in bytes)
SIZE_BUCKETS = (
    100, 500, 1000, 5000, 10000, 50000, 100000, 500000,
    1000000, 5000000, 10000000, float("inf")
)


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================

@dataclass(frozen=True)
class MetricDefinition:
    """
    Definition of a Prometheus metric.

    Attributes:
        name: Metric name (must follow Prometheus naming conventions)
        metric_type: Type of metric (counter, gauge, histogram, summary)
        description: Human-readable description
        labels: List of label names
        buckets: Histogram bucket boundaries (for histogram type)
        unit: Unit of measurement (optional)
    """
    name: str
    metric_type: MetricType
    description: str
    labels: tuple = field(default_factory=tuple)
    buckets: Optional[tuple] = None
    unit: str = ""

    def validate(self) -> bool:
        """Validate metric definition."""
        # Validate name format (Prometheus naming convention)
        if not re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$', self.name):
            raise ValueError(f"Invalid metric name: {self.name}")

        # Validate labels
        for label in self.labels:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label):
                raise ValueError(f"Invalid label name: {label}")

        # Validate buckets for histogram
        if self.metric_type == MetricType.HISTOGRAM:
            if self.buckets is None:
                raise ValueError(f"Histogram {self.name} requires buckets")
            if not all(isinstance(b, (int, float)) for b in self.buckets):
                raise ValueError(f"Buckets must be numeric for {self.name}")

        return True


# -----------------------------------------------------------------------------
# REQUEST METRICS
# -----------------------------------------------------------------------------

REQUEST_LATENCY_HISTOGRAM = MetricDefinition(
    name="gl014_request_latency_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Request latency in seconds by endpoint and method",
    labels=("endpoint", "method"),
    buckets=REQUEST_LATENCY_BUCKETS,
    unit="seconds"
)

REQUEST_COUNT_TOTAL = MetricDefinition(
    name="gl014_request_count_total",
    metric_type=MetricType.COUNTER,
    description="Total number of requests by endpoint and status",
    labels=("endpoint", "status", "method"),
    unit="requests"
)

ACTIVE_REQUESTS_GAUGE = MetricDefinition(
    name="gl014_active_requests",
    metric_type=MetricType.GAUGE,
    description="Number of currently active requests",
    labels=("endpoint",),
    unit="requests"
)

REQUEST_SIZE_HISTOGRAM = MetricDefinition(
    name="gl014_request_size_bytes",
    metric_type=MetricType.HISTOGRAM,
    description="Request body size in bytes",
    labels=("endpoint", "method"),
    buckets=SIZE_BUCKETS,
    unit="bytes"
)

RESPONSE_SIZE_HISTOGRAM = MetricDefinition(
    name="gl014_response_size_bytes",
    metric_type=MetricType.HISTOGRAM,
    description="Response body size in bytes",
    labels=("endpoint", "method"),
    buckets=SIZE_BUCKETS,
    unit="bytes"
)


# -----------------------------------------------------------------------------
# CALCULATOR METRICS
# -----------------------------------------------------------------------------

CALCULATION_DURATION_HISTOGRAM = MetricDefinition(
    name="gl014_calculation_duration_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Calculation duration in seconds by calculator type",
    labels=("calculator_type", "exchanger_type"),
    buckets=CALCULATION_DURATION_BUCKETS,
    unit="seconds"
)

CALCULATION_COUNT_TOTAL = MetricDefinition(
    name="gl014_calculation_count_total",
    metric_type=MetricType.COUNTER,
    description="Total number of calculations by type and status",
    labels=("calculator_type", "status"),
    unit="calculations"
)

CACHE_HIT_RATIO_GAUGE = MetricDefinition(
    name="gl014_cache_hit_ratio",
    metric_type=MetricType.GAUGE,
    description="Cache hit ratio for calculation results (0-1)",
    labels=("cache_type",),
    unit="ratio"
)

FOULING_RESISTANCE_GAUGE = MetricDefinition(
    name="gl014_fouling_resistance_m2kw",
    metric_type=MetricType.GAUGE,
    description="Current fouling resistance in m2-K/W by exchanger",
    labels=("exchanger_id", "side", "location"),
    unit="m2-K/W"
)

HEALTH_INDEX_GAUGE = MetricDefinition(
    name="gl014_health_index",
    metric_type=MetricType.GAUGE,
    description="Heat exchanger health index (0-100) by exchanger",
    labels=("exchanger_id", "location"),
    unit="score"
)


# -----------------------------------------------------------------------------
# INTEGRATION METRICS
# -----------------------------------------------------------------------------

CONNECTOR_LATENCY_HISTOGRAM = MetricDefinition(
    name="gl014_connector_latency_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Connector operation latency in seconds by system",
    labels=("system", "operation"),
    buckets=CONNECTOR_LATENCY_BUCKETS,
    unit="seconds"
)

CONNECTOR_ERRORS_TOTAL = MetricDefinition(
    name="gl014_connector_errors_total",
    metric_type=MetricType.COUNTER,
    description="Total connector errors by system and error type",
    labels=("system", "error_type", "operation"),
    unit="errors"
)

DATA_POINTS_PROCESSED_TOTAL = MetricDefinition(
    name="gl014_data_points_processed_total",
    metric_type=MetricType.COUNTER,
    description="Total data points processed from external systems",
    labels=("source", "data_type"),
    unit="points"
)


# -----------------------------------------------------------------------------
# BUSINESS METRICS
# -----------------------------------------------------------------------------

EXCHANGERS_MONITORED_GAUGE = MetricDefinition(
    name="gl014_exchangers_monitored",
    metric_type=MetricType.GAUGE,
    description="Number of heat exchangers currently being monitored",
    labels=("location", "exchanger_type"),
    unit="exchangers"
)

CLEANING_SCHEDULES_GENERATED_TOTAL = MetricDefinition(
    name="gl014_cleaning_schedules_generated_total",
    metric_type=MetricType.COUNTER,
    description="Total cleaning schedules generated by urgency level",
    labels=("urgency", "cleaning_method"),
    unit="schedules"
)

ESTIMATED_SAVINGS_GAUGE = MetricDefinition(
    name="gl014_estimated_savings_usd",
    metric_type=MetricType.GAUGE,
    description="Estimated savings in USD if cleaning recommendations followed",
    labels=("exchanger_id", "savings_type"),
    unit="USD"
)

FOULING_ALERTS_TOTAL = MetricDefinition(
    name="gl014_fouling_alerts_total",
    metric_type=MetricType.COUNTER,
    description="Total fouling alerts generated by severity",
    labels=("severity", "exchanger_id", "alert_type"),
    unit="alerts"
)


# =============================================================================
# METRIC VALUE STORAGE
# =============================================================================

@dataclass
class MetricValue:
    """
    Storage for a single metric value with labels.

    Attributes:
        value: Current metric value
        labels: Dictionary of label values
        timestamp: Last update timestamp
    """
    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HistogramValue:
    """
    Storage for histogram metric values.

    Attributes:
        buckets: Dictionary mapping bucket boundary to count
        sum_value: Sum of all observed values
        count: Total number of observations
        labels: Dictionary of label values
        timestamp: Last update timestamp
    """
    buckets: Dict[float, int]
    sum_value: float
    count: int
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# METRICS COLLECTOR CLASS
# =============================================================================

class MetricsCollector:
    """
    Production-grade Prometheus metrics collector for GL-014.

    Provides thread-safe metric registration, recording, and export
    for all heat exchanger monitoring metrics.

    Attributes:
        namespace: Metric namespace prefix
        subsystem: Metric subsystem identifier
        registry: Dictionary of registered metrics

    Example:
        >>> collector = MetricsCollector()
        >>> collector.register_metric(REQUEST_LATENCY_HISTOGRAM)
        >>> collector.record_histogram("gl014_request_latency_seconds",
        ...                           {"endpoint": "/api/v1/analyze", "method": "POST"},
        ...                           0.125)
    """

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "MetricsCollector":
        """Singleton pattern for global metrics collector."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        namespace: str = "gl014",
        subsystem: str = "exchanger_pro",
        auto_register: bool = True
    ):
        """
        Initialize the MetricsCollector.

        Args:
            namespace: Metric namespace prefix
            subsystem: Metric subsystem identifier
            auto_register: Whether to auto-register default metrics
        """
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.namespace = namespace
        self.subsystem = subsystem

        # Metric registries (thread-safe)
        self._metrics_lock = threading.RLock()
        self._definitions: Dict[str, MetricDefinition] = {}
        self._counters: Dict[str, Dict[str, float]] = {}
        self._gauges: Dict[str, Dict[str, float]] = {}
        self._histograms: Dict[str, Dict[str, HistogramValue]] = {}

        # Cache statistics
        self._cache_hits: Dict[str, int] = {}
        self._cache_misses: Dict[str, int] = {}

        # Start time for uptime calculation
        self._start_time = datetime.now(timezone.utc)

        # Auto-register default metrics
        if auto_register:
            self._register_default_metrics()

        self._initialized = True
        logger.info(f"MetricsCollector initialized: namespace={namespace}")

    def _register_default_metrics(self) -> None:
        """Register all default GL-014 metrics."""
        default_metrics = [
            # Request metrics
            REQUEST_LATENCY_HISTOGRAM,
            REQUEST_COUNT_TOTAL,
            ACTIVE_REQUESTS_GAUGE,
            REQUEST_SIZE_HISTOGRAM,
            RESPONSE_SIZE_HISTOGRAM,
            # Calculator metrics
            CALCULATION_DURATION_HISTOGRAM,
            CALCULATION_COUNT_TOTAL,
            CACHE_HIT_RATIO_GAUGE,
            FOULING_RESISTANCE_GAUGE,
            HEALTH_INDEX_GAUGE,
            # Integration metrics
            CONNECTOR_LATENCY_HISTOGRAM,
            CONNECTOR_ERRORS_TOTAL,
            DATA_POINTS_PROCESSED_TOTAL,
            # Business metrics
            EXCHANGERS_MONITORED_GAUGE,
            CLEANING_SCHEDULES_GENERATED_TOTAL,
            ESTIMATED_SAVINGS_GAUGE,
            FOULING_ALERTS_TOTAL,
        ]

        for metric in default_metrics:
            self.register_metric(metric)

        logger.debug(f"Registered {len(default_metrics)} default metrics")

    def register_metric(self, definition: MetricDefinition) -> None:
        """
        Register a metric definition.

        Args:
            definition: MetricDefinition to register

        Raises:
            ValueError: If metric name already registered or invalid
        """
        definition.validate()

        with self._metrics_lock:
            if definition.name in self._definitions:
                logger.debug(f"Metric already registered: {definition.name}")
                return

            self._definitions[definition.name] = definition

            # Initialize storage based on type
            if definition.metric_type == MetricType.COUNTER:
                self._counters[definition.name] = {}
            elif definition.metric_type == MetricType.GAUGE:
                self._gauges[definition.name] = {}
            elif definition.metric_type == MetricType.HISTOGRAM:
                self._histograms[definition.name] = {}

            logger.debug(f"Registered metric: {definition.name}")

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dictionary to a hashable key."""
        return "|".join(f"{k}={v}" for k, v in sorted(labels.items()))

    # -------------------------------------------------------------------------
    # COUNTER OPERATIONS
    # -------------------------------------------------------------------------

    def increment_counter(
        self,
        name: str,
        labels: Dict[str, str],
        value: float = 1.0
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            labels: Label values as dictionary
            value: Value to increment by (default 1.0)
        """
        with self._metrics_lock:
            if name not in self._counters:
                logger.warning(f"Counter not registered: {name}")
                return

            key = self._labels_to_key(labels)
            if key not in self._counters[name]:
                self._counters[name][key] = 0.0

            self._counters[name][key] += value

    def get_counter(self, name: str, labels: Dict[str, str]) -> float:
        """
        Get current counter value.

        Args:
            name: Metric name
            labels: Label values

        Returns:
            Current counter value
        """
        with self._metrics_lock:
            if name not in self._counters:
                return 0.0

            key = self._labels_to_key(labels)
            return self._counters[name].get(key, 0.0)

    # -------------------------------------------------------------------------
    # GAUGE OPERATIONS
    # -------------------------------------------------------------------------

    def set_gauge(
        self,
        name: str,
        labels: Dict[str, str],
        value: float
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            labels: Label values as dictionary
            value: Value to set
        """
        with self._metrics_lock:
            if name not in self._gauges:
                logger.warning(f"Gauge not registered: {name}")
                return

            key = self._labels_to_key(labels)
            self._gauges[name][key] = value

    def increment_gauge(
        self,
        name: str,
        labels: Dict[str, str],
        value: float = 1.0
    ) -> None:
        """
        Increment a gauge metric.

        Args:
            name: Metric name
            labels: Label values
            value: Value to increment by
        """
        with self._metrics_lock:
            if name not in self._gauges:
                logger.warning(f"Gauge not registered: {name}")
                return

            key = self._labels_to_key(labels)
            current = self._gauges[name].get(key, 0.0)
            self._gauges[name][key] = current + value

    def decrement_gauge(
        self,
        name: str,
        labels: Dict[str, str],
        value: float = 1.0
    ) -> None:
        """
        Decrement a gauge metric.

        Args:
            name: Metric name
            labels: Label values
            value: Value to decrement by
        """
        self.increment_gauge(name, labels, -value)

    def get_gauge(self, name: str, labels: Dict[str, str]) -> float:
        """
        Get current gauge value.

        Args:
            name: Metric name
            labels: Label values

        Returns:
            Current gauge value
        """
        with self._metrics_lock:
            if name not in self._gauges:
                return 0.0

            key = self._labels_to_key(labels)
            return self._gauges[name].get(key, 0.0)

    # -------------------------------------------------------------------------
    # HISTOGRAM OPERATIONS
    # -------------------------------------------------------------------------

    def record_histogram(
        self,
        name: str,
        labels: Dict[str, str],
        value: float
    ) -> None:
        """
        Record a value in a histogram metric.

        Args:
            name: Metric name
            labels: Label values as dictionary
            value: Value to record
        """
        with self._metrics_lock:
            if name not in self._definitions:
                logger.warning(f"Histogram not registered: {name}")
                return

            definition = self._definitions[name]
            if definition.metric_type != MetricType.HISTOGRAM:
                logger.warning(f"Metric {name} is not a histogram")
                return

            key = self._labels_to_key(labels)

            if key not in self._histograms[name]:
                # Initialize histogram with empty buckets
                self._histograms[name][key] = HistogramValue(
                    buckets={b: 0 for b in definition.buckets},
                    sum_value=0.0,
                    count=0,
                    labels=labels
                )

            histogram = self._histograms[name][key]

            # Update buckets (cumulative)
            for bucket in definition.buckets:
                if value <= bucket:
                    histogram.buckets[bucket] += 1

            histogram.sum_value += value
            histogram.count += 1
            histogram.timestamp = datetime.now(timezone.utc)

    def get_histogram(
        self,
        name: str,
        labels: Dict[str, str]
    ) -> Optional[HistogramValue]:
        """
        Get histogram data.

        Args:
            name: Metric name
            labels: Label values

        Returns:
            HistogramValue or None if not found
        """
        with self._metrics_lock:
            if name not in self._histograms:
                return None

            key = self._labels_to_key(labels)
            return self._histograms[name].get(key)

    # -------------------------------------------------------------------------
    # HIGH-LEVEL RECORDING METHODS
    # -------------------------------------------------------------------------

    def record_request_latency(
        self,
        endpoint: str,
        method: str,
        latency_seconds: float
    ) -> None:
        """
        Record API request latency.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            latency_seconds: Request latency in seconds
        """
        self.record_histogram(
            REQUEST_LATENCY_HISTOGRAM.name,
            {"endpoint": endpoint, "method": method},
            latency_seconds
        )

    def increment_request_count(
        self,
        endpoint: str,
        status: str,
        method: str
    ) -> None:
        """
        Increment request counter.

        Args:
            endpoint: API endpoint path
            status: Response status (success, error, etc.)
            method: HTTP method
        """
        self.increment_counter(
            REQUEST_COUNT_TOTAL.name,
            {"endpoint": endpoint, "status": status, "method": method}
        )

    def set_active_requests(self, endpoint: str, count: int) -> None:
        """
        Set active request count for an endpoint.

        Args:
            endpoint: API endpoint path
            count: Number of active requests
        """
        self.set_gauge(
            ACTIVE_REQUESTS_GAUGE.name,
            {"endpoint": endpoint},
            float(count)
        )

    def record_request_size(
        self,
        endpoint: str,
        method: str,
        size_bytes: int
    ) -> None:
        """
        Record request body size.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            size_bytes: Request body size in bytes
        """
        self.record_histogram(
            REQUEST_SIZE_HISTOGRAM.name,
            {"endpoint": endpoint, "method": method},
            float(size_bytes)
        )

    def record_response_size(
        self,
        endpoint: str,
        method: str,
        size_bytes: int
    ) -> None:
        """
        Record response body size.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            size_bytes: Response body size in bytes
        """
        self.record_histogram(
            RESPONSE_SIZE_HISTOGRAM.name,
            {"endpoint": endpoint, "method": method},
            float(size_bytes)
        )

    def record_calculation_duration(
        self,
        calculator_type: str,
        exchanger_type: str,
        duration_seconds: float
    ) -> None:
        """
        Record calculation duration.

        Args:
            calculator_type: Type of calculator (fouling, thermal, etc.)
            exchanger_type: Type of heat exchanger
            duration_seconds: Calculation duration in seconds
        """
        self.record_histogram(
            CALCULATION_DURATION_HISTOGRAM.name,
            {"calculator_type": calculator_type, "exchanger_type": exchanger_type},
            duration_seconds
        )

    def increment_calculation_count(
        self,
        calculator_type: str,
        status: str
    ) -> None:
        """
        Increment calculation counter.

        Args:
            calculator_type: Type of calculator
            status: Calculation status (success, error)
        """
        self.increment_counter(
            CALCULATION_COUNT_TOTAL.name,
            {"calculator_type": calculator_type, "status": status}
        )

    def set_cache_hit_ratio(self, cache_type: str, ratio: float) -> None:
        """
        Set cache hit ratio.

        Args:
            cache_type: Type of cache (calculation, lookup, etc.)
            ratio: Hit ratio between 0 and 1
        """
        self.set_gauge(
            CACHE_HIT_RATIO_GAUGE.name,
            {"cache_type": cache_type},
            ratio
        )

    def update_cache_stats(
        self,
        cache_type: str,
        hit: bool
    ) -> None:
        """
        Update cache hit/miss statistics and recalculate ratio.

        Args:
            cache_type: Type of cache
            hit: True if cache hit, False if miss
        """
        with self._metrics_lock:
            if cache_type not in self._cache_hits:
                self._cache_hits[cache_type] = 0
                self._cache_misses[cache_type] = 0

            if hit:
                self._cache_hits[cache_type] += 1
            else:
                self._cache_misses[cache_type] += 1

            total = self._cache_hits[cache_type] + self._cache_misses[cache_type]
            if total > 0:
                ratio = self._cache_hits[cache_type] / total
                self.set_cache_hit_ratio(cache_type, ratio)

    def set_fouling_resistance(
        self,
        exchanger_id: str,
        side: str,
        location: str,
        fouling_m2kw: float
    ) -> None:
        """
        Set fouling resistance for an exchanger.

        Args:
            exchanger_id: Heat exchanger identifier
            side: Fouling side (shell, tube)
            location: Plant location
            fouling_m2kw: Fouling resistance in m2-K/W
        """
        self.set_gauge(
            FOULING_RESISTANCE_GAUGE.name,
            {"exchanger_id": exchanger_id, "side": side, "location": location},
            fouling_m2kw
        )

    def set_health_index(
        self,
        exchanger_id: str,
        location: str,
        health_index: float
    ) -> None:
        """
        Set health index for an exchanger.

        Args:
            exchanger_id: Heat exchanger identifier
            location: Plant location
            health_index: Health index score (0-100)
        """
        self.set_gauge(
            HEALTH_INDEX_GAUGE.name,
            {"exchanger_id": exchanger_id, "location": location},
            health_index
        )

    def record_connector_latency(
        self,
        system: str,
        operation: str,
        latency_seconds: float
    ) -> None:
        """
        Record external connector latency.

        Args:
            system: External system name (historian, cmms, etc.)
            operation: Operation type (read, write, query)
            latency_seconds: Operation latency in seconds
        """
        self.record_histogram(
            CONNECTOR_LATENCY_HISTOGRAM.name,
            {"system": system, "operation": operation},
            latency_seconds
        )

    def increment_connector_errors(
        self,
        system: str,
        error_type: str,
        operation: str
    ) -> None:
        """
        Increment connector error counter.

        Args:
            system: External system name
            error_type: Type of error
            operation: Operation that failed
        """
        self.increment_counter(
            CONNECTOR_ERRORS_TOTAL.name,
            {"system": system, "error_type": error_type, "operation": operation}
        )

    def increment_data_points_processed(
        self,
        source: str,
        data_type: str,
        count: int = 1
    ) -> None:
        """
        Increment data points processed counter.

        Args:
            source: Data source identifier
            data_type: Type of data processed
            count: Number of data points
        """
        self.increment_counter(
            DATA_POINTS_PROCESSED_TOTAL.name,
            {"source": source, "data_type": data_type},
            float(count)
        )

    def set_exchangers_monitored(
        self,
        location: str,
        exchanger_type: str,
        count: int
    ) -> None:
        """
        Set count of monitored exchangers.

        Args:
            location: Plant location
            exchanger_type: Type of heat exchanger
            count: Number of exchangers
        """
        self.set_gauge(
            EXCHANGERS_MONITORED_GAUGE.name,
            {"location": location, "exchanger_type": exchanger_type},
            float(count)
        )

    def increment_cleaning_schedules(
        self,
        urgency: str,
        cleaning_method: str
    ) -> None:
        """
        Increment cleaning schedules generated counter.

        Args:
            urgency: Schedule urgency level
            cleaning_method: Recommended cleaning method
        """
        self.increment_counter(
            CLEANING_SCHEDULES_GENERATED_TOTAL.name,
            {"urgency": urgency, "cleaning_method": cleaning_method}
        )

    def set_estimated_savings(
        self,
        exchanger_id: str,
        savings_type: str,
        amount_usd: float
    ) -> None:
        """
        Set estimated savings amount.

        Args:
            exchanger_id: Heat exchanger identifier
            savings_type: Type of savings (energy, maintenance)
            amount_usd: Savings amount in USD
        """
        self.set_gauge(
            ESTIMATED_SAVINGS_GAUGE.name,
            {"exchanger_id": exchanger_id, "savings_type": savings_type},
            amount_usd
        )

    def increment_fouling_alerts(
        self,
        severity: str,
        exchanger_id: str,
        alert_type: str
    ) -> None:
        """
        Increment fouling alerts counter.

        Args:
            severity: Alert severity (warning, critical)
            exchanger_id: Heat exchanger identifier
            alert_type: Type of alert
        """
        self.increment_counter(
            FOULING_ALERTS_TOTAL.name,
            {"severity": severity, "exchanger_id": exchanger_id, "alert_type": alert_type}
        )

    # -------------------------------------------------------------------------
    # METRIC EXPORT
    # -------------------------------------------------------------------------

    def export_prometheus_format(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns:
            String containing metrics in Prometheus exposition format
        """
        lines: List[str] = []

        with self._metrics_lock:
            for name, definition in self._definitions.items():
                # Add HELP and TYPE headers
                lines.append(f"# HELP {name} {definition.description}")
                lines.append(f"# TYPE {name} {definition.metric_type.value}")

                if definition.metric_type == MetricType.COUNTER:
                    self._export_counter(lines, name, definition)
                elif definition.metric_type == MetricType.GAUGE:
                    self._export_gauge(lines, name, definition)
                elif definition.metric_type == MetricType.HISTOGRAM:
                    self._export_histogram(lines, name, definition)

                lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _export_counter(
        self,
        lines: List[str],
        name: str,
        definition: MetricDefinition
    ) -> None:
        """Export counter metrics."""
        if name not in self._counters:
            return

        for key, value in self._counters[name].items():
            labels_str = self._key_to_labels_str(key)
            if labels_str:
                lines.append(f"{name}{{{labels_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

    def _export_gauge(
        self,
        lines: List[str],
        name: str,
        definition: MetricDefinition
    ) -> None:
        """Export gauge metrics."""
        if name not in self._gauges:
            return

        for key, value in self._gauges[name].items():
            labels_str = self._key_to_labels_str(key)
            if labels_str:
                lines.append(f"{name}{{{labels_str}}} {value}")
            else:
                lines.append(f"{name} {value}")

    def _export_histogram(
        self,
        lines: List[str],
        name: str,
        definition: MetricDefinition
    ) -> None:
        """Export histogram metrics."""
        if name not in self._histograms:
            return

        for key, histogram in self._histograms[name].items():
            labels_str = self._key_to_labels_str(key)

            # Export bucket values
            for bucket, count in sorted(histogram.buckets.items()):
                bucket_str = "+Inf" if bucket == float("inf") else str(bucket)
                if labels_str:
                    lines.append(
                        f'{name}_bucket{{{labels_str},le="{bucket_str}"}} {count}'
                    )
                else:
                    lines.append(f'{name}_bucket{{le="{bucket_str}"}} {count}')

            # Export sum and count
            if labels_str:
                lines.append(f"{name}_sum{{{labels_str}}} {histogram.sum_value}")
                lines.append(f"{name}_count{{{labels_str}}} {histogram.count}")
            else:
                lines.append(f"{name}_sum {histogram.sum_value}")
                lines.append(f"{name}_count {histogram.count}")

    def _key_to_labels_str(self, key: str) -> str:
        """Convert label key back to Prometheus format."""
        if not key:
            return ""

        parts = key.split("|")
        label_pairs = []
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                label_pairs.append(f'{k}="{v}"')

        return ",".join(label_pairs)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dictionary of all metrics organized by type
        """
        with self._metrics_lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        key: {
                            "buckets": dict(h.buckets),
                            "sum": h.sum_value,
                            "count": h.count,
                            "labels": h.labels
                        }
                        for key, h in histograms.items()
                    }
                    for name, histograms in self._histograms.items()
                },
                "uptime_seconds": (
                    datetime.now(timezone.utc) - self._start_time
                ).total_seconds()
            }

    def reset_all(self) -> None:
        """Reset all metric values (for testing purposes)."""
        with self._metrics_lock:
            for name in self._counters:
                self._counters[name] = {}
            for name in self._gauges:
                self._gauges[name] = {}
            for name in self._histograms:
                self._histograms[name] = {}
            self._cache_hits = {}
            self._cache_misses = {}

        logger.info("All metrics reset")


# =============================================================================
# DECORATORS
# =============================================================================

def timed_operation(
    metric_name: str,
    labels_func: Optional[Callable[..., Dict[str, str]]] = None
) -> Callable:
    """
    Decorator to time function execution and record to histogram.

    Args:
        metric_name: Histogram metric name to record to
        labels_func: Function to extract labels from arguments

    Returns:
        Decorated function

    Example:
        >>> @timed_operation("gl014_calculation_duration_seconds",
        ...                  lambda self, x: {"calculator_type": "fouling"})
        ... def calculate_fouling(self, data):
        ...     # calculation logic
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                labels = labels_func(*args, **kwargs) if labels_func else {}
                collector.record_histogram(metric_name, labels, duration)

        return wrapper
    return decorator


def count_calls(
    metric_name: str,
    labels_func: Optional[Callable[..., Dict[str, str]]] = None,
    include_status: bool = True
) -> Callable:
    """
    Decorator to count function calls.

    Args:
        metric_name: Counter metric name
        labels_func: Function to extract labels
        include_status: Whether to include success/error status

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            labels = labels_func(*args, **kwargs) if labels_func else {}

            try:
                result = func(*args, **kwargs)
                if include_status:
                    labels["status"] = "success"
                collector.increment_counter(metric_name, labels)
                return result
            except Exception as e:
                if include_status:
                    labels["status"] = "error"
                collector.increment_counter(metric_name, labels)
                raise

        return wrapper
    return decorator


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.

    Returns:
        MetricsCollector singleton instance
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def create_metrics_endpoint() -> Callable[[], str]:
    """
    Create a metrics endpoint handler.

    Returns:
        Callable that returns Prometheus metrics string

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> metrics_handler = create_metrics_endpoint()
        >>> @app.get("/metrics")
        ... def metrics():
        ...     return Response(metrics_handler(), media_type="text/plain")
    """
    def handler() -> str:
        collector = get_metrics_collector()
        return collector.export_prometheus_format()

    return handler


def reset_metrics_collector() -> None:
    """Reset the global metrics collector (for testing)."""
    global _global_collector
    _global_collector = None


__all__ = [
    # Main Classes
    "MetricsCollector",
    "MetricDefinition",
    "MetricValue",
    "HistogramValue",

    # Enumerations
    "MetricType",

    # Request Metrics
    "REQUEST_LATENCY_HISTOGRAM",
    "REQUEST_COUNT_TOTAL",
    "ACTIVE_REQUESTS_GAUGE",
    "REQUEST_SIZE_HISTOGRAM",
    "RESPONSE_SIZE_HISTOGRAM",

    # Calculator Metrics
    "CALCULATION_DURATION_HISTOGRAM",
    "CALCULATION_COUNT_TOTAL",
    "CACHE_HIT_RATIO_GAUGE",
    "FOULING_RESISTANCE_GAUGE",
    "HEALTH_INDEX_GAUGE",

    # Integration Metrics
    "CONNECTOR_LATENCY_HISTOGRAM",
    "CONNECTOR_ERRORS_TOTAL",
    "DATA_POINTS_PROCESSED_TOTAL",

    # Business Metrics
    "EXCHANGERS_MONITORED_GAUGE",
    "CLEANING_SCHEDULES_GENERATED_TOTAL",
    "ESTIMATED_SAVINGS_GAUGE",
    "FOULING_ALERTS_TOTAL",

    # Bucket Configurations
    "REQUEST_LATENCY_BUCKETS",
    "CALCULATION_DURATION_BUCKETS",
    "CONNECTOR_LATENCY_BUCKETS",
    "SIZE_BUCKETS",

    # Decorators
    "timed_operation",
    "count_calls",

    # Utility Functions
    "get_metrics_collector",
    "create_metrics_endpoint",
    "reset_metrics_collector",
]
