# -*- coding: utf-8 -*-
"""
Unified Metrics Collection Engine - AGENT-FOUND-010: Observability & Telemetry Agent

Provides unified Prometheus-compatible metric recording with support for
counters, gauges, histograms, and summaries. All metric recordings include
SHA-256 provenance hashes for complete audit trails.

Zero-Hallucination Guarantees:
    - All metric values are deterministic numeric recordings
    - Provenance hashes use SHA-256 for tamper-evident audit trails
    - Label serialization is deterministic (sorted keys)
    - Histogram bucket boundaries are fixed at registration time
    - No probabilistic or ML-based metric computation

Example:
    >>> from greenlang.observability_agent.metrics_collector import MetricsCollector
    >>> from greenlang.observability_agent.config import ObservabilityConfig
    >>> collector = MetricsCollector(ObservabilityConfig())
    >>> collector.register_metric("requests_total", "counter", "Total requests")
    >>> collector.increment("requests_total", {"method": "GET"})
    >>> print(collector.export_prometheus())

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

# Default histogram buckets following Prometheus convention
DEFAULT_HISTOGRAM_BUCKETS: Tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


@dataclass
class MetricDefinition:
    """Definition of a registered metric.

    Attributes:
        metric_id: Unique identifier for this metric definition.
        name: Metric name (Prometheus-compatible).
        metric_type: One of counter, gauge, histogram, summary.
        description: Human-readable description.
        labels: Allowed label names for this metric.
        buckets: Histogram bucket boundaries (only for histogram type).
        created_at: Timestamp when metric was registered.
    """

    metric_id: str = ""
    name: str = ""
    metric_type: str = "counter"
    description: str = ""
    labels: List[str] = field(default_factory=list)
    buckets: Tuple[float, ...] = DEFAULT_HISTOGRAM_BUCKETS
    created_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        """Generate metric_id if not provided."""
        if not self.metric_id:
            self.metric_id = str(uuid.uuid4())


@dataclass
class MetricRecording:
    """Record of a single metric observation.

    Attributes:
        recording_id: Unique identifier for this recording.
        metric_name: Name of the metric recorded.
        value: Numeric value recorded.
        labels: Labels attached to this recording.
        timestamp: UTC timestamp of the recording.
        provenance_hash: SHA-256 hash for audit trail.
    """

    recording_id: str = ""
    metric_name: str = ""
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate recording_id if not provided."""
        if not self.recording_id:
            self.recording_id = str(uuid.uuid4())


@dataclass
class MetricSeries:
    """Time series data for a metric with specific label combination.

    Attributes:
        labels_key: Deterministic string key from label values.
        labels: Label key-value pairs.
        value: Current value of the series.
        recordings: Total number of recordings for this series.
        last_updated: Last recording timestamp.
        histogram_counts: Bucket counts for histogram metrics.
        histogram_sum: Sum of observed values for histogram metrics.
        histogram_count: Count of observations for histogram metrics.
    """

    labels_key: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    recordings: int = 0
    last_updated: datetime = field(default_factory=_utcnow)
    histogram_counts: Dict[float, int] = field(default_factory=dict)
    histogram_sum: float = 0.0
    histogram_count: int = 0


# =============================================================================
# MetricsCollector
# =============================================================================


class MetricsCollector:
    """Unified Prometheus-compatible metric collection engine.

    Records counters, gauges, histograms, and summaries with deterministic
    label serialization and SHA-256 provenance hashing for every observation.

    Thread-safe via a reentrant lock on all mutating operations.

    Attributes:
        _config: Observability configuration.
        _metrics: Registered metric definitions keyed by name.
        _series: Metric time series keyed by (metric_name, labels_key).
        _total_recordings: Running count of all recordings.
        _lock: Thread lock for concurrent access.

    Example:
        >>> collector = MetricsCollector(config)
        >>> collector.register_metric("latency", "histogram", "Request latency")
        >>> collector.observe_histogram("latency", 0.35, {"endpoint": "/api"})
        >>> print(collector.export_prometheus())
    """

    # Valid metric types
    VALID_TYPES: Tuple[str, ...] = ("counter", "gauge", "histogram", "summary")

    def __init__(self, config: Any) -> None:
        """Initialize MetricsCollector.

        Args:
            config: Observability configuration instance. Must expose
                    ``metric_retention_seconds`` and ``max_series`` attributes
                    (or reasonable defaults are used).
        """
        self._config = config
        self._metrics: Dict[str, MetricDefinition] = {}
        self._series: Dict[str, MetricSeries] = {}
        self._total_recordings: int = 0
        self._lock = threading.RLock()

        self._max_series: int = getattr(config, "max_series", 50000)
        self._retention_seconds: int = getattr(config, "metric_retention_seconds", 86400)

        logger.info(
            "MetricsCollector initialized: max_series=%d, retention=%ds",
            self._max_series,
            self._retention_seconds,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_metric(
        self,
        name: str,
        metric_type: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> MetricDefinition:
        """Register a new metric definition.

        Args:
            name: Metric name (must be unique, Prometheus-compatible).
            metric_type: One of counter, gauge, histogram, summary.
            description: Human-readable description.
            labels: Allowed label names for this metric.
            buckets: Custom histogram bucket boundaries (histogram only).

        Returns:
            MetricDefinition for the registered metric.

        Raises:
            ValueError: If name is empty, type is invalid, or metric already exists.
        """
        if not name or not name.strip():
            raise ValueError("Metric name must be non-empty")

        if metric_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid metric_type '{metric_type}'; must be one of {self.VALID_TYPES}"
            )

        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric '{name}' is already registered")

            definition = MetricDefinition(
                name=name,
                metric_type=metric_type,
                description=description,
                labels=labels or [],
                buckets=buckets or DEFAULT_HISTOGRAM_BUCKETS,
            )
            self._metrics[name] = definition

        logger.info(
            "Registered metric: name=%s, type=%s, labels=%s",
            name, metric_type, labels or [],
        )
        return definition

    # ------------------------------------------------------------------
    # Recording operations
    # ------------------------------------------------------------------

    def record(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> MetricRecording:
        """Record a metric observation with provenance hash.

        This is the general-purpose recording method. For type-specific
        convenience, use increment(), set_gauge(), or observe_histogram().

        Args:
            name: Name of the registered metric.
            value: Numeric value to record.
            labels: Label key-value pairs.

        Returns:
            MetricRecording with provenance hash.

        Raises:
            ValueError: If the metric is not registered.
        """
        labels = labels or {}
        self._validate_metric_exists(name)

        definition = self._metrics[name]
        self._validate_labels(definition, labels)

        now = _utcnow()
        provenance_hash = self._compute_recording_hash(name, value, labels, now)

        labels_key = self._labels_to_key(labels)
        series_key = f"{name}:{labels_key}"

        with self._lock:
            series = self._get_or_create_series(series_key, labels)
            series.value = self._apply_value(definition.metric_type, series.value, value)
            series.recordings += 1
            series.last_updated = now

            if definition.metric_type == "histogram":
                self._update_histogram_buckets(series, value, definition.buckets)

            self._total_recordings += 1

        recording = MetricRecording(
            metric_name=name,
            value=value,
            labels=labels,
            timestamp=now,
            provenance_hash=provenance_hash,
        )

        logger.debug(
            "Recorded metric: name=%s, value=%s, labels=%s, hash=%s",
            name, value, labels, provenance_hash[:12],
        )
        return recording

    def increment(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        amount: float = 1.0,
    ) -> MetricRecording:
        """Increment a counter metric.

        Args:
            name: Name of the registered counter metric.
            labels: Label key-value pairs.
            amount: Amount to increment by (must be non-negative).

        Returns:
            MetricRecording with provenance hash.

        Raises:
            ValueError: If metric is not a counter or amount is negative.
        """
        self._validate_metric_exists(name)
        definition = self._metrics[name]

        if definition.metric_type != "counter":
            raise ValueError(
                f"increment() requires a counter metric; '{name}' is {definition.metric_type}"
            )
        if amount < 0:
            raise ValueError(f"Counter increment amount must be non-negative; got {amount}")

        return self.record(name, amount, labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> MetricRecording:
        """Set a gauge metric to an absolute value.

        Args:
            name: Name of the registered gauge metric.
            value: Absolute value to set.
            labels: Label key-value pairs.

        Returns:
            MetricRecording with provenance hash.

        Raises:
            ValueError: If metric is not a gauge.
        """
        self._validate_metric_exists(name)
        definition = self._metrics[name]

        if definition.metric_type != "gauge":
            raise ValueError(
                f"set_gauge() requires a gauge metric; '{name}' is {definition.metric_type}"
            )

        labels = labels or {}
        now = _utcnow()
        provenance_hash = self._compute_recording_hash(name, value, labels, now)
        labels_key = self._labels_to_key(labels)
        series_key = f"{name}:{labels_key}"

        with self._lock:
            series = self._get_or_create_series(series_key, labels)
            series.value = value
            series.recordings += 1
            series.last_updated = now
            self._total_recordings += 1

        recording = MetricRecording(
            metric_name=name,
            value=value,
            labels=labels,
            timestamp=now,
            provenance_hash=provenance_hash,
        )
        logger.debug("Set gauge: name=%s, value=%s", name, value)
        return recording

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> MetricRecording:
        """Record a histogram observation.

        Args:
            name: Name of the registered histogram metric.
            value: Observed value.
            labels: Label key-value pairs.

        Returns:
            MetricRecording with provenance hash.

        Raises:
            ValueError: If metric is not a histogram.
        """
        self._validate_metric_exists(name)
        definition = self._metrics[name]

        if definition.metric_type != "histogram":
            raise ValueError(
                f"observe_histogram() requires a histogram metric; "
                f"'{name}' is {definition.metric_type}"
            )

        return self.record(name, value, labels)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_metric(self, name: str) -> Optional[MetricDefinition]:
        """Get a metric definition with its current series values.

        Args:
            name: Metric name.

        Returns:
            MetricDefinition or None if not found.
        """
        return self._metrics.get(name)

    def get_metric_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get the current value of a metric series.

        Args:
            name: Metric name.
            labels: Label key-value pairs to identify the series.

        Returns:
            Current metric value or None if series does not exist.
        """
        labels = labels or {}
        labels_key = self._labels_to_key(labels)
        series_key = f"{name}:{labels_key}"

        with self._lock:
            series = self._series.get(series_key)
            if series is None:
                return None
            return series.value

    def get_metric_series(self, name: str) -> List[MetricSeries]:
        """Get all series for a metric across all label combinations.

        Args:
            name: Metric name.

        Returns:
            List of MetricSeries for all label combinations.
        """
        prefix = f"{name}:"
        with self._lock:
            return [
                series
                for key, series in self._series.items()
                if key.startswith(prefix)
            ]

    def list_metrics(
        self,
        prefix_filter: Optional[str] = None,
    ) -> List[MetricDefinition]:
        """List all registered metric definitions.

        Args:
            prefix_filter: If provided, only return metrics whose name
                           starts with this prefix.

        Returns:
            List of MetricDefinition objects, sorted by name.
        """
        with self._lock:
            metrics = list(self._metrics.values())

        if prefix_filter:
            metrics = [m for m in metrics if m.name.startswith(prefix_filter)]

        metrics.sort(key=lambda m: m.name)
        return metrics

    def get_histogram_data(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get histogram bucket data for a metric series.

        Args:
            name: Histogram metric name.
            labels: Label key-value pairs.

        Returns:
            Dictionary with buckets, sum, and count; or None if not found.
        """
        labels = labels or {}
        labels_key = self._labels_to_key(labels)
        series_key = f"{name}:{labels_key}"

        with self._lock:
            series = self._series.get(series_key)
            if series is None:
                return None

            return {
                "buckets": dict(sorted(series.histogram_counts.items())),
                "sum": series.histogram_sum,
                "count": series.histogram_count,
            }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text exposition format.

        Returns:
            String in Prometheus text format suitable for scraping.
        """
        lines: List[str] = []

        with self._lock:
            for name in sorted(self._metrics.keys()):
                definition = self._metrics[name]
                lines.append(f"# HELP {name} {definition.description}")
                lines.append(f"# TYPE {name} {definition.metric_type}")

                prefix = f"{name}:"
                relevant_series = [
                    (key, series)
                    for key, series in self._series.items()
                    if key.startswith(prefix)
                ]

                for _key, series in sorted(relevant_series, key=lambda x: x[0]):
                    label_str = self._format_prometheus_labels(series.labels)

                    if definition.metric_type == "histogram":
                        lines.extend(
                            self._format_histogram_lines(name, label_str, series, definition)
                        )
                    else:
                        metric_line = f"{name}{label_str} {self._format_value(series.value)}"
                        lines.append(metric_line)

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics.

        Returns:
            Dictionary with total_metrics, total_series, total_recordings,
            metrics_by_type, and oldest_series timestamp.
        """
        with self._lock:
            type_counts: Dict[str, int] = {}
            for m in self._metrics.values():
                type_counts[m.metric_type] = type_counts.get(m.metric_type, 0) + 1

            oldest: Optional[datetime] = None
            for s in self._series.values():
                if oldest is None or s.last_updated < oldest:
                    oldest = s.last_updated

            return {
                "total_metrics": len(self._metrics),
                "total_series": len(self._series),
                "total_recordings": self._total_recordings,
                "max_series": self._max_series,
                "retention_seconds": self._retention_seconds,
                "metrics_by_type": type_counts,
                "oldest_series": oldest.isoformat() if oldest else None,
            }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_stale_series(self) -> int:
        """Remove time series that have not been updated within the retention window.

        Returns:
            Number of series removed.
        """
        cutoff = _utcnow().timestamp() - self._retention_seconds
        removed = 0

        with self._lock:
            stale_keys = [
                key
                for key, series in self._series.items()
                if series.last_updated.timestamp() < cutoff
            ]
            for key in stale_keys:
                del self._series[key]
                removed += 1

        if removed > 0:
            logger.info("Cleaned up %d stale metric series", removed)
        return removed

    def reset_metric(self, name: str) -> int:
        """Reset all series for a metric to zero values.

        Args:
            name: Metric name to reset.

        Returns:
            Number of series reset.

        Raises:
            ValueError: If metric is not registered.
        """
        self._validate_metric_exists(name)
        prefix = f"{name}:"
        count = 0

        with self._lock:
            for key, series in self._series.items():
                if key.startswith(prefix):
                    series.value = 0.0
                    series.histogram_counts.clear()
                    series.histogram_sum = 0.0
                    series.histogram_count = 0
                    count += 1

        logger.info("Reset %d series for metric '%s'", count, name)
        return count

    def unregister_metric(self, name: str) -> bool:
        """Unregister a metric and remove all its series.

        Args:
            name: Metric name to unregister.

        Returns:
            True if the metric was found and removed, False otherwise.
        """
        with self._lock:
            if name not in self._metrics:
                return False

            del self._metrics[name]

            prefix = f"{name}:"
            stale_keys = [k for k in self._series if k.startswith(prefix)]
            for key in stale_keys:
                del self._series[key]

        logger.info("Unregistered metric '%s' and removed %d series", name, len(stale_keys))
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_metric_exists(self, name: str) -> None:
        """Raise ValueError if metric is not registered.

        Args:
            name: Metric name to validate.

        Raises:
            ValueError: If the metric is not registered.
        """
        if name not in self._metrics:
            raise ValueError(f"Metric '{name}' is not registered")

    def _validate_labels(
        self,
        definition: MetricDefinition,
        labels: Dict[str, str],
    ) -> None:
        """Validate that label keys match the metric definition.

        Args:
            definition: Metric definition with allowed labels.
            labels: Provided label key-value pairs.

        Raises:
            ValueError: If unexpected label keys are present.
        """
        if definition.labels:
            unexpected = set(labels.keys()) - set(definition.labels)
            if unexpected:
                raise ValueError(
                    f"Unexpected labels for metric '{definition.name}': {unexpected}. "
                    f"Allowed: {definition.labels}"
                )

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Serialize labels deterministically for use as a dictionary key.

        Labels are sorted by key and serialized as ``key=value`` pairs
        joined with commas.

        Args:
            labels: Label key-value pairs.

        Returns:
            Deterministic string representation.
        """
        if not labels:
            return ""
        sorted_pairs = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_pairs)

    def _compute_recording_hash(
        self,
        name: str,
        value: float,
        labels: Dict[str, str],
        timestamp: datetime,
    ) -> str:
        """Compute SHA-256 provenance hash for a metric recording.

        Args:
            name: Metric name.
            value: Recorded value.
            labels: Label key-value pairs.
            timestamp: Recording timestamp.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps(
            {
                "name": name,
                "value": value,
                "labels": dict(sorted(labels.items())),
                "timestamp": timestamp.isoformat(),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_or_create_series(
        self,
        series_key: str,
        labels: Dict[str, str],
    ) -> MetricSeries:
        """Get an existing series or create a new one.

        Args:
            series_key: Composite key (metric_name:labels_key).
            labels: Label key-value pairs.

        Returns:
            MetricSeries instance.

        Raises:
            ValueError: If max_series limit would be exceeded.
        """
        series = self._series.get(series_key)
        if series is not None:
            return series

        if len(self._series) >= self._max_series:
            raise ValueError(
                f"Maximum series limit ({self._max_series}) reached; "
                "clean up stale series or increase the limit"
            )

        series = MetricSeries(
            labels_key=self._labels_to_key(labels),
            labels=dict(labels),
        )
        self._series[series_key] = series
        return series

    def _apply_value(
        self,
        metric_type: str,
        current: float,
        new_value: float,
    ) -> float:
        """Apply a recorded value according to the metric type.

        Counters accumulate; gauges and others pass through.

        Args:
            metric_type: The metric type.
            current: Current series value.
            new_value: Incoming recorded value.

        Returns:
            Updated series value.
        """
        if metric_type == "counter":
            return current + new_value
        return new_value

    def _update_histogram_buckets(
        self,
        series: MetricSeries,
        value: float,
        buckets: Tuple[float, ...],
    ) -> None:
        """Update histogram bucket counts for an observation.

        Args:
            series: MetricSeries to update.
            value: Observed value.
            buckets: Bucket boundaries.
        """
        for boundary in buckets:
            if value <= boundary:
                series.histogram_counts[boundary] = (
                    series.histogram_counts.get(boundary, 0) + 1
                )
        # +Inf bucket
        series.histogram_counts[math.inf] = (
            series.histogram_counts.get(math.inf, 0) + 1
        )
        series.histogram_sum += value
        series.histogram_count += 1

    def _format_prometheus_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus text exposition.

        Args:
            labels: Label key-value pairs.

        Returns:
            String like ``{key1="val1",key2="val2"}`` or empty string.
        """
        if not labels:
            return ""
        sorted_pairs = sorted(labels.items())
        inner = ",".join(f'{k}="{v}"' for k, v in sorted_pairs)
        return "{" + inner + "}"

    def _format_histogram_lines(
        self,
        name: str,
        label_str: str,
        series: MetricSeries,
        definition: MetricDefinition,
    ) -> List[str]:
        """Format histogram series as Prometheus exposition lines.

        Args:
            name: Metric name.
            label_str: Formatted Prometheus label string.
            series: MetricSeries with histogram data.
            definition: MetricDefinition with bucket boundaries.

        Returns:
            List of exposition lines for this histogram series.
        """
        lines: List[str] = []
        cumulative = 0

        for boundary in sorted(definition.buckets):
            cumulative += series.histogram_counts.get(boundary, 0)
            le_label = self._merge_label_str(label_str, f'le="{boundary}"')
            lines.append(f"{name}_bucket{le_label} {cumulative}")

        # +Inf bucket
        cumulative += series.histogram_counts.get(math.inf, 0) - cumulative
        inf_total = series.histogram_count
        inf_label = self._merge_label_str(label_str, 'le="+Inf"')
        lines.append(f"{name}_bucket{inf_label} {inf_total}")

        sum_line = f"{name}_sum{label_str} {self._format_value(series.histogram_sum)}"
        count_line = f"{name}_count{label_str} {series.histogram_count}"
        lines.append(sum_line)
        lines.append(count_line)

        return lines

    def _merge_label_str(self, existing: str, extra: str) -> str:
        """Merge an additional label into an existing Prometheus label string.

        Args:
            existing: Current label string (may be empty or ``{...}``).
            extra: Additional label assignment (e.g. ``le="0.5"``).

        Returns:
            Combined label string.
        """
        if not existing:
            return "{" + extra + "}"
        # Strip trailing }, append, close
        return existing[:-1] + "," + extra + "}"

    @staticmethod
    def _format_value(value: float) -> str:
        """Format a float value for Prometheus exposition.

        Integers are rendered without decimal points; floats use repr-style.

        Args:
            value: Numeric value.

        Returns:
            Formatted string representation.
        """
        if value == int(value) and not math.isinf(value):
            return str(int(value))
        return repr(value)


__all__ = [
    "MetricsCollector",
    "MetricDefinition",
    "MetricRecording",
    "MetricSeries",
    "DEFAULT_HISTOGRAM_BUCKETS",
]
