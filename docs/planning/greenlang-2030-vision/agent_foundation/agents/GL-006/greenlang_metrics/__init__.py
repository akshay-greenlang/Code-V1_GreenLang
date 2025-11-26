# -*- coding: utf-8 -*-
"""
GreenLang Metrics Framework for GL-006 HeatRecoveryMaximizer.

This module provides Prometheus metrics collection and reporting utilities
for monitoring heat recovery operations, performance, and business outcomes.
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
import functools
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import prometheus_client, provide stubs if not available
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Stub classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
        def time(self): return contextmanager(lambda: (yield))()

    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

    class CollectorRegistry:
        def __init__(self): pass
        def register(self, *args, **kwargs): pass

    def generate_latest(*args): return b""
    CONTENT_TYPE_LATEST = "text/plain"
    REGISTRY = CollectorRegistry()


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    unit: str = ""


class MetricsRegistry:
    """
    Registry for managing metrics.

    Provides a centralized location for defining and accessing metrics.
    """

    def __init__(self, prefix: str = "gl006"):
        """
        Initialize the metrics registry.

        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix
        self._metrics: Dict[str, Any] = {}
        self._definitions: Dict[str, MetricDefinition] = {}

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        return f"{self.prefix}_{name}"

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Create or get a counter metric."""
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            self._metrics[full_name] = Counter(
                full_name,
                description,
                labels or [],
            )
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                description=description,
                metric_type=MetricType.COUNTER,
                labels=labels or [],
            )
        return self._metrics[full_name]

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Create or get a gauge metric."""
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            self._metrics[full_name] = Gauge(
                full_name,
                description,
                labels or [],
            )
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                description=description,
                metric_type=MetricType.GAUGE,
                labels=labels or [],
            )
        return self._metrics[full_name]

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """Create or get a histogram metric."""
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            kwargs = {"labelnames": labels or []}
            if buckets:
                kwargs["buckets"] = buckets
            self._metrics[full_name] = Histogram(
                full_name,
                description,
                **kwargs,
            )
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                description=description,
                metric_type=MetricType.HISTOGRAM,
                labels=labels or [],
                buckets=buckets,
            )
        return self._metrics[full_name]

    def summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
    ) -> Summary:
        """Create or get a summary metric."""
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            self._metrics[full_name] = Summary(
                full_name,
                description,
                labels or [],
            )
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                description=description,
                metric_type=MetricType.SUMMARY,
                labels=labels or [],
            )
        return self._metrics[full_name]

    def info(
        self,
        name: str,
        description: str,
    ) -> Info:
        """Create or get an info metric."""
        full_name = self._full_name(name)
        if full_name not in self._metrics:
            self._metrics[full_name] = Info(
                full_name,
                description,
            )
            self._definitions[full_name] = MetricDefinition(
                name=full_name,
                description=description,
                metric_type=MetricType.INFO,
            )
        return self._metrics[full_name]

    def get_definitions(self) -> Dict[str, MetricDefinition]:
        """Get all metric definitions."""
        return self._definitions.copy()


# Global metrics registry
registry = MetricsRegistry()


def timed(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to time function execution.

    Args:
        metric_name: Name of the histogram metric
        labels: Optional labels for the metric
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            histogram = registry.histogram(
                f"{metric_name}_duration_seconds",
                f"Duration of {metric_name} in seconds",
                list(labels.keys()) if labels else None,
            )
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            histogram = registry.histogram(
                f"{metric_name}_duration_seconds",
                f"Duration of {metric_name} in seconds",
                list(labels.keys()) if labels else None,
            )
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def counted(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to count function calls.

    Args:
        metric_name: Name of the counter metric
        labels: Optional labels for the metric
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            counter = registry.counter(
                f"{metric_name}_total",
                f"Total count of {metric_name}",
                list(labels.keys()) if labels else None,
            )
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            counter = registry.counter(
                f"{metric_name}_total",
                f"Total count of {metric_name}",
                list(labels.keys()) if labels else None,
            )
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def track_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager to track operation metrics.

    Args:
        operation_name: Name of the operation
        labels: Optional labels

    Yields:
        None
    """
    counter = registry.counter(
        f"{operation_name}_total",
        f"Total {operation_name} operations",
        list(labels.keys()) if labels else None,
    )
    histogram = registry.histogram(
        f"{operation_name}_duration_seconds",
        f"Duration of {operation_name} in seconds",
        list(labels.keys()) if labels else None,
    )
    error_counter = registry.counter(
        f"{operation_name}_errors_total",
        f"Total {operation_name} errors",
        list(labels.keys()) if labels else None,
    )

    start = time.time()
    try:
        yield
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()
    except Exception:
        if labels:
            error_counter.labels(**labels).inc()
        else:
            error_counter.inc()
        raise
    finally:
        duration = time.time() - start
        if labels:
            histogram.labels(**labels).observe(duration)
        else:
            histogram.observe(duration)


class HeatRecoveryMetrics:
    """
    Specialized metrics for heat recovery operations.

    Provides pre-defined metrics for common heat recovery measurements.
    """

    def __init__(self, registry: MetricsRegistry = registry):
        """Initialize heat recovery metrics."""
        self.registry = registry

        # Stream metrics
        self.streams_analyzed = registry.counter(
            "streams_analyzed",
            "Total number of streams analyzed",
            ["stream_type"],
        )
        self.stream_temperature = registry.gauge(
            "stream_temperature_celsius",
            "Current stream temperature",
            ["stream_id", "position"],
        )
        self.stream_flow_rate = registry.gauge(
            "stream_flow_rate_kg_per_second",
            "Current stream flow rate",
            ["stream_id"],
        )

        # Heat duty metrics
        self.heat_duty = registry.gauge(
            "heat_duty_kw",
            "Heat duty",
            ["stream_id", "type"],
        )
        self.total_recoverable_heat = registry.gauge(
            "total_recoverable_heat_kw",
            "Total recoverable heat",
        )
        self.actual_recovered_heat = registry.gauge(
            "actual_recovered_heat_kw",
            "Actually recovered heat",
        )

        # Analysis metrics
        self.pinch_temperature = registry.gauge(
            "pinch_temperature_celsius",
            "Calculated pinch temperature",
        )
        self.min_hot_utility = registry.gauge(
            "min_hot_utility_kw",
            "Minimum hot utility requirement",
        )
        self.min_cold_utility = registry.gauge(
            "min_cold_utility_kw",
            "Minimum cold utility requirement",
        )

        # Economic metrics
        self.annual_savings = registry.gauge(
            "annual_savings_usd",
            "Projected annual savings",
            ["project_id"],
        )
        self.roi = registry.gauge(
            "roi_percentage",
            "Return on investment",
            ["project_id"],
        )
        self.payback_years = registry.gauge(
            "payback_years",
            "Payback period",
            ["project_id"],
        )
        self.npv = registry.gauge(
            "npv_usd",
            "Net present value",
            ["project_id"],
        )

        # Calculation metrics
        self.calculations_total = registry.counter(
            "calculations_total",
            "Total calculations performed",
            ["calculation_type"],
        )
        self.calculation_duration = registry.histogram(
            "calculation_duration_seconds",
            "Duration of calculations",
            ["calculation_type"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60],
        )
        self.calculation_errors = registry.counter(
            "calculation_errors_total",
            "Total calculation errors",
            ["calculation_type", "error_type"],
        )

        # Validation metrics
        self.validations_total = registry.counter(
            "validations_total",
            "Total validations performed",
            ["validation_type"],
        )
        self.validation_failures = registry.counter(
            "validation_failures_total",
            "Total validation failures",
            ["validation_type", "severity"],
        )

        # System metrics
        self.active_analyses = registry.gauge(
            "active_analyses",
            "Number of currently active analyses",
        )
        self.queue_size = registry.gauge(
            "queue_size",
            "Number of items in processing queue",
        )

    def record_stream_analysis(self, stream_type: str):
        """Record a stream analysis."""
        self.streams_analyzed.labels(stream_type=stream_type).inc()

    def record_calculation(
        self,
        calculation_type: str,
        duration_seconds: float,
        success: bool = True,
        error_type: Optional[str] = None,
    ):
        """Record a calculation."""
        self.calculations_total.labels(calculation_type=calculation_type).inc()
        self.calculation_duration.labels(calculation_type=calculation_type).observe(duration_seconds)
        if not success and error_type:
            self.calculation_errors.labels(
                calculation_type=calculation_type,
                error_type=error_type,
            ).inc()

    def update_pinch_results(
        self,
        pinch_temp: float,
        hot_utility: float,
        cold_utility: float,
    ):
        """Update pinch analysis results."""
        self.pinch_temperature.set(pinch_temp)
        self.min_hot_utility.set(hot_utility)
        self.min_cold_utility.set(cold_utility)

    def update_economic_results(
        self,
        project_id: str,
        annual_savings: float,
        roi: float,
        payback: float,
        npv: float,
    ):
        """Update economic calculation results."""
        self.annual_savings.labels(project_id=project_id).set(annual_savings)
        self.roi.labels(project_id=project_id).set(roi)
        self.payback_years.labels(project_id=project_id).set(payback)
        self.npv.labels(project_id=project_id).set(npv)


# Global heat recovery metrics instance
hr_metrics = HeatRecoveryMetrics()


__all__ = [
    'PROMETHEUS_AVAILABLE',
    'MetricType',
    'MetricDefinition',
    'MetricsRegistry',
    'registry',
    'timed',
    'counted',
    'track_operation',
    'HeatRecoveryMetrics',
    'hr_metrics',
    # Re-export prometheus types for convenience
    'Counter',
    'Gauge',
    'Histogram',
    'Summary',
    'Info',
    'generate_latest',
    'CONTENT_TYPE_LATEST',
]
