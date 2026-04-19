"""
GL-014 EXCHANGERPRO - Metrics Collection

Prometheus-compatible metrics for monitoring agent performance,
thermal calculations, predictions, and recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import threading
import time


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBucket:
    """Histogram bucket for latency metrics."""
    le: float  # less than or equal
    count: int = 0


class MetricsCollector:
    """
    Metrics collector for GL-014 EXCHANGERPRO.

    Collects and exposes metrics in Prometheus format:
    - Calculation latencies
    - Prediction counts and accuracies
    - Recommendation rates
    - Error counts
    - Data quality scores
    """

    AGENT_ID = "GL-014"
    AGENT_NAME = "EXCHANGERPRO"

    # Default histogram buckets for latency (in seconds)
    DEFAULT_LATENCY_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._lock = threading.Lock()
        self._counters: Dict[str, Dict[str, float]] = {}
        self._gauges: Dict[str, Dict[str, float]] = {}
        self._histograms: Dict[str, Dict[str, List[HistogramBucket]]] = {}
        self._histogram_sums: Dict[str, Dict[str, float]] = {}
        self._histogram_counts: Dict[str, Dict[str, int]] = {}

        # Initialize standard metrics
        self._init_standard_metrics()

    def _init_standard_metrics(self) -> None:
        """Initialize standard agent metrics."""
        # Counters
        self._counters = {
            "gl014_thermal_calculations_total": {},
            "gl014_fouling_predictions_total": {},
            "gl014_cleaning_recommendations_total": {},
            "gl014_errors_total": {},
            "gl014_data_quality_violations_total": {},
        }

        # Gauges
        self._gauges = {
            "gl014_active_exchangers": {},
            "gl014_model_version": {},
            "gl014_last_calculation_timestamp": {},
            "gl014_average_ua_degradation": {},
            "gl014_pending_recommendations": {},
        }

        # Histograms
        for metric in [
            "gl014_thermal_calculation_duration_seconds",
            "gl014_fouling_prediction_duration_seconds",
            "gl014_optimization_duration_seconds",
        ]:
            self._histograms[metric] = {}
            self._histogram_sums[metric] = {}
            self._histogram_counts[metric] = {}

    def _labels_key(self, labels: Dict[str, str]) -> str:
        """Create a unique key from labels."""
        sorted_items = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_items)

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._counters:
                self._counters[name] = {}

            if key not in self._counters[name]:
                self._counters[name][key] = 0.0

            self._counters[name][key] += value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value."""
        labels = labels or {}
        key = self._labels_key(labels)

        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = {}

            self._gauges[name][key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        """Observe a value for a histogram metric."""
        labels = labels or {}
        key = self._labels_key(labels)
        buckets = buckets or self.DEFAULT_LATENCY_BUCKETS

        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = {}
                self._histogram_sums[name] = {}
                self._histogram_counts[name] = {}

            if key not in self._histograms[name]:
                self._histograms[name][key] = [
                    HistogramBucket(le=b) for b in buckets
                ] + [HistogramBucket(le=float("inf"))]
                self._histogram_sums[name][key] = 0.0
                self._histogram_counts[name][key] = 0

            # Update buckets
            for bucket in self._histograms[name][key]:
                if value <= bucket.le:
                    bucket.count += 1

            self._histogram_sums[name][key] += value
            self._histogram_counts[name][key] += 1

    def time_calculation(self, calculation_type: str, exchanger_id: str):
        """Context manager to time a calculation."""
        return _CalculationTimer(self, calculation_type, exchanger_id)

    # Convenience methods for common metrics
    def record_thermal_calculation(
        self,
        exchanger_id: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record a thermal calculation."""
        labels = {"exchanger_id": exchanger_id}

        self.inc_counter("gl014_thermal_calculations_total", labels=labels)
        self.observe_histogram(
            "gl014_thermal_calculation_duration_seconds",
            duration_seconds,
            labels=labels,
        )

        if not success:
            self.inc_counter("gl014_errors_total", labels={**labels, "type": "thermal"})

    def record_fouling_prediction(
        self,
        exchanger_id: str,
        duration_seconds: float,
        horizon_days: int,
        success: bool = True,
    ) -> None:
        """Record a fouling prediction."""
        labels = {"exchanger_id": exchanger_id, "horizon_days": str(horizon_days)}

        self.inc_counter("gl014_fouling_predictions_total", labels=labels)
        self.observe_histogram(
            "gl014_fouling_prediction_duration_seconds",
            duration_seconds,
            labels=labels,
        )

        if not success:
            self.inc_counter("gl014_errors_total", labels={**labels, "type": "prediction"})

    def record_cleaning_recommendation(
        self,
        exchanger_id: str,
        recommended: bool,
        days_until_cleaning: Optional[int] = None,
    ) -> None:
        """Record a cleaning recommendation."""
        labels = {
            "exchanger_id": exchanger_id,
            "recommended": str(recommended).lower(),
        }
        if days_until_cleaning is not None:
            labels["days_until_cleaning"] = str(days_until_cleaning)

        self.inc_counter("gl014_cleaning_recommendations_total", labels=labels)

    def record_data_quality_violation(
        self,
        exchanger_id: str,
        violation_type: str,
    ) -> None:
        """Record a data quality violation."""
        self.inc_counter(
            "gl014_data_quality_violations_total",
            labels={"exchanger_id": exchanger_id, "type": violation_type},
        )

    def update_ua_degradation(
        self,
        exchanger_id: str,
        ua_degradation: float,
    ) -> None:
        """Update UA degradation gauge for an exchanger."""
        self.set_gauge(
            "gl014_average_ua_degradation",
            ua_degradation,
            labels={"exchanger_id": exchanger_id},
        )

    def get_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        # Counters
        for name, values in self._counters.items():
            lines.append(f"# HELP {name} Counter metric")
            lines.append(f"# TYPE {name} counter")
            for labels_key, value in values.items():
                if labels_key:
                    lines.append(f"{name}{{{labels_key}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        # Gauges
        for name, values in self._gauges.items():
            lines.append(f"# HELP {name} Gauge metric")
            lines.append(f"# TYPE {name} gauge")
            for labels_key, value in values.items():
                if labels_key:
                    lines.append(f"{name}{{{labels_key}}} {value}")
                else:
                    lines.append(f"{name} {value}")

        # Histograms
        for name, values in self._histograms.items():
            lines.append(f"# HELP {name} Histogram metric")
            lines.append(f"# TYPE {name} histogram")
            for labels_key, buckets in values.items():
                base_labels = labels_key
                for bucket in buckets:
                    le_label = f'le="{bucket.le}"'
                    if base_labels:
                        full_labels = f"{base_labels},{le_label}"
                    else:
                        full_labels = le_label
                    lines.append(f"{name}_bucket{{{full_labels}}} {bucket.count}")

                # Sum and count
                if base_labels:
                    lines.append(f"{name}_sum{{{base_labels}}} {self._histogram_sums[name][labels_key]}")
                    lines.append(f"{name}_count{{{base_labels}}} {self._histogram_counts[name][labels_key]}")
                else:
                    lines.append(f"{name}_sum {self._histogram_sums[name][labels_key]}")
                    lines.append(f"{name}_count {self._histogram_counts[name][labels_key]}")

        return "\n".join(lines)

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: {
                    key: {
                        "buckets": [(b.le, b.count) for b in buckets],
                        "sum": self._histogram_sums[name][key],
                        "count": self._histogram_counts[name][key],
                    }
                    for key, buckets in values.items()
                }
                for name, values in self._histograms.items()
            },
        }


class _CalculationTimer:
    """Context manager for timing calculations."""

    def __init__(
        self,
        collector: MetricsCollector,
        calculation_type: str,
        exchanger_id: str,
    ) -> None:
        self.collector = collector
        self.calculation_type = calculation_type
        self.exchanger_id = exchanger_id
        self.start_time: float = 0.0

    def __enter__(self) -> "_CalculationTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.perf_counter() - self.start_time
        success = exc_type is None

        if self.calculation_type == "thermal":
            self.collector.record_thermal_calculation(
                self.exchanger_id, duration, success
            )
        elif self.calculation_type == "prediction":
            self.collector.record_fouling_prediction(
                self.exchanger_id, duration, horizon_days=7, success=success
            )


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
