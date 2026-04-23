"""
MetricsCollector - Prometheus Metrics for FurnacePulse Monitoring

This module implements the MetricsCollector for exposing operational metrics
via Prometheus format. It tracks alert volumes, false positives, acknowledgement
rates, response times, data pipeline health, and model performance metrics.

Metric Categories:
    - Alert Metrics: Volumes, false positives, acknowledgement rates, response times
    - Pipeline Metrics: Lag, drop rates, missingness, schema errors
    - Model Metrics: Drift, performance, explanation stability
    - KPI Metrics: Business-level performance indicators

Example:
    >>> config = MetricsCollectorConfig(namespace="furnacepulse")
    >>> collector = MetricsCollector(config)
    >>> collector.record_alert_created("A-001", "MEDIUM")
    >>> collector.record_alert_acknowledged("A-001", response_time_seconds=120.5)
    >>> metrics_output = collector.export_prometheus()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of Prometheus metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricValue:
    """Current value of a metric with labels."""

    value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HistogramValue:
    """Histogram metric value with bucket counts."""

    sum_value: float
    count: int
    buckets: Dict[float, int]  # bucket_le -> count
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AlertMetrics(BaseModel):
    """Aggregated alert metrics for reporting."""

    total_alerts_created: int = Field(default=0)
    alerts_by_code: Dict[str, int] = Field(default_factory=dict)
    alerts_by_severity: Dict[str, int] = Field(default_factory=dict)
    acknowledged_count: int = Field(default=0)
    unacknowledged_count: int = Field(default=0)
    false_positive_count: int = Field(default=0)
    false_positive_rate: float = Field(default=0.0)
    avg_acknowledgement_time_seconds: float = Field(default=0.0)
    avg_resolution_time_seconds: float = Field(default=0.0)
    escalation_count: int = Field(default=0)
    suppression_count: int = Field(default=0)


class PipelineMetrics(BaseModel):
    """Data pipeline health metrics."""

    messages_processed: int = Field(default=0)
    messages_dropped: int = Field(default=0)
    drop_rate: float = Field(default=0.0)
    current_lag_seconds: float = Field(default=0.0)
    max_lag_seconds: float = Field(default=0.0)
    avg_processing_time_ms: float = Field(default=0.0)
    schema_errors: int = Field(default=0)
    parse_errors: int = Field(default=0)
    missing_fields_count: int = Field(default=0)
    missingness_rate: float = Field(default=0.0)
    last_message_timestamp: Optional[datetime] = Field(None)
    backpressure_events: int = Field(default=0)


class ModelMetrics(BaseModel):
    """Model performance and drift metrics."""

    model_name: str = Field(default="")
    predictions_made: int = Field(default=0)
    avg_inference_time_ms: float = Field(default=0.0)
    drift_score: float = Field(default=0.0)
    drift_detected: bool = Field(default=False)
    feature_drift_scores: Dict[str, float] = Field(default_factory=dict)
    prediction_confidence_avg: float = Field(default=0.0)
    explanation_stability_score: float = Field(default=0.0)
    last_retrain_timestamp: Optional[datetime] = Field(None)
    accuracy_estimate: float = Field(default=0.0)
    precision_estimate: float = Field(default=0.0)
    recall_estimate: float = Field(default=0.0)


class MetricsCollectorConfig(BaseModel):
    """Configuration for MetricsCollector."""

    namespace: str = Field(default="furnacepulse", description="Prometheus namespace")
    subsystem: str = Field(default="", description="Prometheus subsystem")
    enable_histogram_buckets: bool = Field(default=True)
    default_histogram_buckets: List[float] = Field(
        default=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
    )
    response_time_buckets: List[float] = Field(
        default=[1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600]
    )
    enable_kpi_metrics: bool = Field(default=True)
    aggregation_window_seconds: int = Field(default=300, ge=60)
    retention_hours: int = Field(default=24, ge=1)


# Metric definitions for FurnacePulse
ALERT_METRIC_DEFINITIONS = [
    MetricDefinition(
        name="alerts_total",
        metric_type=MetricType.COUNTER,
        description="Total number of alerts created",
        labels=["alert_code", "severity"],
    ),
    MetricDefinition(
        name="alerts_acknowledged_total",
        metric_type=MetricType.COUNTER,
        description="Total number of alerts acknowledged",
        labels=["alert_code"],
    ),
    MetricDefinition(
        name="alerts_resolved_total",
        metric_type=MetricType.COUNTER,
        description="Total number of alerts resolved",
        labels=["alert_code"],
    ),
    MetricDefinition(
        name="alerts_escalated_total",
        metric_type=MetricType.COUNTER,
        description="Total number of alerts escalated",
        labels=["alert_code"],
    ),
    MetricDefinition(
        name="alerts_suppressed_total",
        metric_type=MetricType.COUNTER,
        description="Total number of alerts suppressed (deduplicated)",
        labels=["alert_code"],
    ),
    MetricDefinition(
        name="alerts_false_positive_total",
        metric_type=MetricType.COUNTER,
        description="Total number of alerts marked as false positive",
        labels=["alert_code"],
    ),
    MetricDefinition(
        name="alert_acknowledgement_time_seconds",
        metric_type=MetricType.HISTOGRAM,
        description="Time to acknowledge alerts in seconds",
        labels=["alert_code"],
    ),
    MetricDefinition(
        name="alert_resolution_time_seconds",
        metric_type=MetricType.HISTOGRAM,
        description="Time to resolve alerts in seconds",
        labels=["alert_code"],
    ),
    MetricDefinition(
        name="alerts_active",
        metric_type=MetricType.GAUGE,
        description="Number of currently active alerts",
        labels=["alert_code", "severity"],
    ),
]

PIPELINE_METRIC_DEFINITIONS = [
    MetricDefinition(
        name="pipeline_messages_processed_total",
        metric_type=MetricType.COUNTER,
        description="Total messages processed by the pipeline",
        labels=["source", "stage"],
    ),
    MetricDefinition(
        name="pipeline_messages_dropped_total",
        metric_type=MetricType.COUNTER,
        description="Total messages dropped by the pipeline",
        labels=["source", "reason"],
    ),
    MetricDefinition(
        name="pipeline_lag_seconds",
        metric_type=MetricType.GAUGE,
        description="Current pipeline lag in seconds",
        labels=["source"],
    ),
    MetricDefinition(
        name="pipeline_processing_time_seconds",
        metric_type=MetricType.HISTOGRAM,
        description="Pipeline processing time in seconds",
        labels=["stage"],
    ),
    MetricDefinition(
        name="pipeline_schema_errors_total",
        metric_type=MetricType.COUNTER,
        description="Total schema validation errors",
        labels=["source", "error_type"],
    ),
    MetricDefinition(
        name="pipeline_missing_fields_total",
        metric_type=MetricType.COUNTER,
        description="Total missing required fields",
        labels=["source", "field_name"],
    ),
    MetricDefinition(
        name="pipeline_backpressure_events_total",
        metric_type=MetricType.COUNTER,
        description="Total backpressure events",
        labels=["source"],
    ),
]

MODEL_METRIC_DEFINITIONS = [
    MetricDefinition(
        name="model_predictions_total",
        metric_type=MetricType.COUNTER,
        description="Total predictions made by the model",
        labels=["model_name", "model_version"],
    ),
    MetricDefinition(
        name="model_inference_time_seconds",
        metric_type=MetricType.HISTOGRAM,
        description="Model inference time in seconds",
        labels=["model_name"],
    ),
    MetricDefinition(
        name="model_drift_score",
        metric_type=MetricType.GAUGE,
        description="Current model drift score (0-1, higher = more drift)",
        labels=["model_name", "feature"],
    ),
    MetricDefinition(
        name="model_prediction_confidence",
        metric_type=MetricType.HISTOGRAM,
        description="Prediction confidence scores",
        labels=["model_name"],
    ),
    MetricDefinition(
        name="model_explanation_stability",
        metric_type=MetricType.GAUGE,
        description="Explanation stability score (0-1, higher = more stable)",
        labels=["model_name"],
    ),
    MetricDefinition(
        name="model_accuracy",
        metric_type=MetricType.GAUGE,
        description="Estimated model accuracy based on feedback",
        labels=["model_name"],
    ),
]

KPI_METRIC_DEFINITIONS = [
    MetricDefinition(
        name="kpi_hotspot_detection_rate",
        metric_type=MetricType.GAUGE,
        description="Hotspot detection rate (detected/actual)",
        labels=[],
    ),
    MetricDefinition(
        name="kpi_mean_time_to_detection_seconds",
        metric_type=MetricType.GAUGE,
        description="Mean time to detect anomalies",
        labels=["alert_type"],
    ),
    MetricDefinition(
        name="kpi_mean_time_to_response_seconds",
        metric_type=MetricType.GAUGE,
        description="Mean time to respond to alerts",
        labels=["alert_type"],
    ),
    MetricDefinition(
        name="kpi_sensor_health_score",
        metric_type=MetricType.GAUGE,
        description="Overall sensor health score (0-100)",
        labels=["zone"],
    ),
    MetricDefinition(
        name="kpi_data_quality_score",
        metric_type=MetricType.GAUGE,
        description="Overall data quality score (0-100)",
        labels=["source"],
    ),
]


class MetricsCollector:
    """
    Prometheus metrics collector for FurnacePulse monitoring.

    This class collects and exposes operational metrics for alert management,
    data pipeline health, model performance, and business KPIs.

    Attributes:
        config: Collector configuration
        counters: Counter metric values
        gauges: Gauge metric values
        histograms: Histogram metric values
        metric_definitions: Registered metric definitions

    Example:
        >>> config = MetricsCollectorConfig()
        >>> collector = MetricsCollector(config)
        >>> collector.record_alert_created("A-001", "MEDIUM")
        >>> output = collector.export_prometheus()
    """

    def __init__(self, config: MetricsCollectorConfig):
        """
        Initialize MetricsCollector.

        Args:
            config: Collector configuration
        """
        self.config = config
        self._lock = Lock()

        # Metric storage
        self.counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.histograms: Dict[str, Dict[str, HistogramValue]] = defaultdict(dict)

        # Metric definitions
        self.metric_definitions: Dict[str, MetricDefinition] = {}

        # Time series for aggregation
        self._acknowledgement_times: List[Tuple[datetime, str, float]] = []
        self._resolution_times: List[Tuple[datetime, str, float]] = []
        self._inference_times: List[Tuple[datetime, str, float]] = []

        # Register all metrics
        self._register_metrics()

        logger.info(
            "MetricsCollector initialized with namespace=%s, %d metrics registered",
            config.namespace, len(self.metric_definitions)
        )

    def _register_metrics(self) -> None:
        """Register all metric definitions."""
        all_definitions = (
            ALERT_METRIC_DEFINITIONS
            + PIPELINE_METRIC_DEFINITIONS
            + MODEL_METRIC_DEFINITIONS
        )

        if self.config.enable_kpi_metrics:
            all_definitions += KPI_METRIC_DEFINITIONS

        for definition in all_definitions:
            full_name = self._get_full_metric_name(definition.name)
            self.metric_definitions[full_name] = definition

            # Initialize histogram buckets
            if definition.metric_type == MetricType.HISTOGRAM:
                if definition.name.endswith("_time_seconds"):
                    definition.buckets = self.config.response_time_buckets
                else:
                    definition.buckets = self.config.default_histogram_buckets

    def _get_full_metric_name(self, name: str) -> str:
        """Get the full metric name with namespace and subsystem."""
        parts = [self.config.namespace]
        if self.config.subsystem:
            parts.append(self.config.subsystem)
        parts.append(name)
        return "_".join(parts)

    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """Generate a unique key for a set of labels."""
        if not labels:
            return ""
        sorted_items = sorted(labels.items())
        return ",".join(f'{k}="{v}"' for k, v in sorted_items)

    # ==================== Alert Metrics ====================

    def record_alert_created(
        self,
        alert_code: str,
        severity: str,
    ) -> None:
        """
        Record that an alert was created.

        Args:
            alert_code: Alert taxonomy code (e.g., "A-001")
            severity: Alert severity level
        """
        with self._lock:
            metric_name = self._get_full_metric_name("alerts_total")
            label_key = self._get_label_key({"alert_code": alert_code, "severity": severity})
            self.counters[metric_name][label_key] += 1

            # Update active gauge
            gauge_name = self._get_full_metric_name("alerts_active")
            self.gauges[gauge_name][label_key] += 1

        logger.debug("Recorded alert created: %s %s", alert_code, severity)

    def record_alert_acknowledged(
        self,
        alert_code: str,
        response_time_seconds: float,
    ) -> None:
        """
        Record that an alert was acknowledged.

        Args:
            alert_code: Alert taxonomy code
            response_time_seconds: Time from creation to acknowledgement
        """
        with self._lock:
            # Increment counter
            metric_name = self._get_full_metric_name("alerts_acknowledged_total")
            label_key = self._get_label_key({"alert_code": alert_code})
            self.counters[metric_name][label_key] += 1

            # Record histogram
            hist_name = self._get_full_metric_name("alert_acknowledgement_time_seconds")
            self._record_histogram(hist_name, {"alert_code": alert_code}, response_time_seconds)

            # Store for aggregation
            self._acknowledgement_times.append(
                (datetime.utcnow(), alert_code, response_time_seconds)
            )

        logger.debug("Recorded alert acknowledged: %s in %.2fs", alert_code, response_time_seconds)

    def record_alert_resolved(
        self,
        alert_code: str,
        resolution_time_seconds: float,
    ) -> None:
        """
        Record that an alert was resolved.

        Args:
            alert_code: Alert taxonomy code
            resolution_time_seconds: Time from creation to resolution
        """
        with self._lock:
            # Increment counter
            metric_name = self._get_full_metric_name("alerts_resolved_total")
            label_key = self._get_label_key({"alert_code": alert_code})
            self.counters[metric_name][label_key] += 1

            # Record histogram
            hist_name = self._get_full_metric_name("alert_resolution_time_seconds")
            self._record_histogram(hist_name, {"alert_code": alert_code}, resolution_time_seconds)

            # Store for aggregation
            self._resolution_times.append(
                (datetime.utcnow(), alert_code, resolution_time_seconds)
            )

            # Decrease active gauge (need to find the right label key)
            gauge_name = self._get_full_metric_name("alerts_active")
            for key in self.gauges[gauge_name]:
                if f'alert_code="{alert_code}"' in key:
                    self.gauges[gauge_name][key] = max(0, self.gauges[gauge_name][key] - 1)
                    break

        logger.debug("Recorded alert resolved: %s in %.2fs", alert_code, resolution_time_seconds)

    def record_alert_escalated(self, alert_code: str) -> None:
        """Record that an alert was escalated."""
        with self._lock:
            metric_name = self._get_full_metric_name("alerts_escalated_total")
            label_key = self._get_label_key({"alert_code": alert_code})
            self.counters[metric_name][label_key] += 1

    def record_alert_suppressed(self, alert_code: str) -> None:
        """Record that an alert was suppressed (deduplicated)."""
        with self._lock:
            metric_name = self._get_full_metric_name("alerts_suppressed_total")
            label_key = self._get_label_key({"alert_code": alert_code})
            self.counters[metric_name][label_key] += 1

    def record_false_positive(self, alert_code: str) -> None:
        """Record that an alert was marked as false positive."""
        with self._lock:
            metric_name = self._get_full_metric_name("alerts_false_positive_total")
            label_key = self._get_label_key({"alert_code": alert_code})
            self.counters[metric_name][label_key] += 1

    # ==================== Pipeline Metrics ====================

    def record_message_processed(
        self,
        source: str,
        stage: str,
        processing_time_seconds: float,
    ) -> None:
        """
        Record that a message was processed.

        Args:
            source: Data source identifier
            stage: Pipeline stage
            processing_time_seconds: Processing time
        """
        with self._lock:
            # Increment counter
            metric_name = self._get_full_metric_name("pipeline_messages_processed_total")
            label_key = self._get_label_key({"source": source, "stage": stage})
            self.counters[metric_name][label_key] += 1

            # Record processing time histogram
            hist_name = self._get_full_metric_name("pipeline_processing_time_seconds")
            self._record_histogram(hist_name, {"stage": stage}, processing_time_seconds)

    def record_message_dropped(
        self,
        source: str,
        reason: str,
    ) -> None:
        """
        Record that a message was dropped.

        Args:
            source: Data source identifier
            reason: Reason for dropping (e.g., "parse_error", "schema_invalid")
        """
        with self._lock:
            metric_name = self._get_full_metric_name("pipeline_messages_dropped_total")
            label_key = self._get_label_key({"source": source, "reason": reason})
            self.counters[metric_name][label_key] += 1

    def update_pipeline_lag(
        self,
        source: str,
        lag_seconds: float,
    ) -> None:
        """
        Update the current pipeline lag.

        Args:
            source: Data source identifier
            lag_seconds: Current lag in seconds
        """
        with self._lock:
            metric_name = self._get_full_metric_name("pipeline_lag_seconds")
            label_key = self._get_label_key({"source": source})
            self.gauges[metric_name][label_key] = lag_seconds

    def record_schema_error(
        self,
        source: str,
        error_type: str,
    ) -> None:
        """
        Record a schema validation error.

        Args:
            source: Data source identifier
            error_type: Type of schema error
        """
        with self._lock:
            metric_name = self._get_full_metric_name("pipeline_schema_errors_total")
            label_key = self._get_label_key({"source": source, "error_type": error_type})
            self.counters[metric_name][label_key] += 1

    def record_missing_field(
        self,
        source: str,
        field_name: str,
    ) -> None:
        """
        Record a missing required field.

        Args:
            source: Data source identifier
            field_name: Name of the missing field
        """
        with self._lock:
            metric_name = self._get_full_metric_name("pipeline_missing_fields_total")
            label_key = self._get_label_key({"source": source, "field_name": field_name})
            self.counters[metric_name][label_key] += 1

    def record_backpressure_event(self, source: str) -> None:
        """Record a backpressure event."""
        with self._lock:
            metric_name = self._get_full_metric_name("pipeline_backpressure_events_total")
            label_key = self._get_label_key({"source": source})
            self.counters[metric_name][label_key] += 1

    # ==================== Model Metrics ====================

    def record_prediction(
        self,
        model_name: str,
        model_version: str,
        inference_time_seconds: float,
        confidence: float,
    ) -> None:
        """
        Record a model prediction.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            inference_time_seconds: Inference time
            confidence: Prediction confidence score
        """
        with self._lock:
            # Increment counter
            metric_name = self._get_full_metric_name("model_predictions_total")
            label_key = self._get_label_key({
                "model_name": model_name,
                "model_version": model_version
            })
            self.counters[metric_name][label_key] += 1

            # Record inference time histogram
            hist_name = self._get_full_metric_name("model_inference_time_seconds")
            self._record_histogram(hist_name, {"model_name": model_name}, inference_time_seconds)

            # Record confidence histogram
            conf_hist_name = self._get_full_metric_name("model_prediction_confidence")
            self._record_histogram(conf_hist_name, {"model_name": model_name}, confidence)

            # Store for aggregation
            self._inference_times.append(
                (datetime.utcnow(), model_name, inference_time_seconds)
            )

    def update_drift_score(
        self,
        model_name: str,
        overall_score: float,
        feature_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Update model drift scores.

        Args:
            model_name: Name of the model
            overall_score: Overall drift score (0-1)
            feature_scores: Per-feature drift scores (optional)
        """
        with self._lock:
            metric_name = self._get_full_metric_name("model_drift_score")

            # Overall drift
            label_key = self._get_label_key({"model_name": model_name, "feature": "overall"})
            self.gauges[metric_name][label_key] = overall_score

            # Per-feature drift
            if feature_scores:
                for feature, score in feature_scores.items():
                    label_key = self._get_label_key({
                        "model_name": model_name,
                        "feature": feature
                    })
                    self.gauges[metric_name][label_key] = score

    def update_explanation_stability(
        self,
        model_name: str,
        stability_score: float,
    ) -> None:
        """
        Update model explanation stability score.

        Args:
            model_name: Name of the model
            stability_score: Explanation stability score (0-1)
        """
        with self._lock:
            metric_name = self._get_full_metric_name("model_explanation_stability")
            label_key = self._get_label_key({"model_name": model_name})
            self.gauges[metric_name][label_key] = stability_score

    def update_model_accuracy(
        self,
        model_name: str,
        accuracy: float,
    ) -> None:
        """
        Update estimated model accuracy.

        Args:
            model_name: Name of the model
            accuracy: Accuracy estimate (0-1)
        """
        with self._lock:
            metric_name = self._get_full_metric_name("model_accuracy")
            label_key = self._get_label_key({"model_name": model_name})
            self.gauges[metric_name][label_key] = accuracy

    # ==================== KPI Metrics ====================

    def update_hotspot_detection_rate(self, rate: float) -> None:
        """Update the hotspot detection rate KPI."""
        with self._lock:
            metric_name = self._get_full_metric_name("kpi_hotspot_detection_rate")
            self.gauges[metric_name][""] = rate

    def update_mean_time_to_detection(
        self,
        alert_type: str,
        mttd_seconds: float,
    ) -> None:
        """
        Update mean time to detection KPI.

        Args:
            alert_type: Type of alert
            mttd_seconds: Mean time to detection in seconds
        """
        with self._lock:
            metric_name = self._get_full_metric_name("kpi_mean_time_to_detection_seconds")
            label_key = self._get_label_key({"alert_type": alert_type})
            self.gauges[metric_name][label_key] = mttd_seconds

    def update_mean_time_to_response(
        self,
        alert_type: str,
        mttr_seconds: float,
    ) -> None:
        """
        Update mean time to response KPI.

        Args:
            alert_type: Type of alert
            mttr_seconds: Mean time to response in seconds
        """
        with self._lock:
            metric_name = self._get_full_metric_name("kpi_mean_time_to_response_seconds")
            label_key = self._get_label_key({"alert_type": alert_type})
            self.gauges[metric_name][label_key] = mttr_seconds

    def update_sensor_health_score(
        self,
        zone: str,
        score: float,
    ) -> None:
        """
        Update sensor health score for a zone.

        Args:
            zone: Furnace zone identifier
            score: Health score (0-100)
        """
        with self._lock:
            metric_name = self._get_full_metric_name("kpi_sensor_health_score")
            label_key = self._get_label_key({"zone": zone})
            self.gauges[metric_name][label_key] = score

    def update_data_quality_score(
        self,
        source: str,
        score: float,
    ) -> None:
        """
        Update data quality score for a source.

        Args:
            source: Data source identifier
            score: Quality score (0-100)
        """
        with self._lock:
            metric_name = self._get_full_metric_name("kpi_data_quality_score")
            label_key = self._get_label_key({"source": source})
            self.gauges[metric_name][label_key] = score

    # ==================== Histogram Helpers ====================

    def _record_histogram(
        self,
        metric_name: str,
        labels: Dict[str, str],
        value: float,
    ) -> None:
        """Record a value in a histogram."""
        label_key = self._get_label_key(labels)
        definition = self.metric_definitions.get(metric_name)

        if label_key not in self.histograms[metric_name]:
            buckets = definition.buckets if definition else self.config.default_histogram_buckets
            self.histograms[metric_name][label_key] = HistogramValue(
                sum_value=0.0,
                count=0,
                buckets={b: 0 for b in buckets},
                labels=labels,
            )

        hist = self.histograms[metric_name][label_key]
        hist.sum_value += value
        hist.count += 1
        hist.timestamp = datetime.utcnow()

        # Update bucket counts
        for bucket_le in sorted(hist.buckets.keys()):
            if value <= bucket_le:
                hist.buckets[bucket_le] += 1

    # ==================== Export Methods ====================

    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        with self._lock:
            # Export counters
            for metric_name, values in self.counters.items():
                definition = self.metric_definitions.get(metric_name)
                if definition:
                    lines.append(f"# HELP {metric_name} {definition.description}")
                    lines.append(f"# TYPE {metric_name} counter")

                for label_key, value in values.items():
                    if label_key:
                        lines.append(f"{metric_name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{metric_name} {value}")

            # Export gauges
            for metric_name, values in self.gauges.items():
                definition = self.metric_definitions.get(metric_name)
                if definition:
                    lines.append(f"# HELP {metric_name} {definition.description}")
                    lines.append(f"# TYPE {metric_name} gauge")

                for label_key, value in values.items():
                    if label_key:
                        lines.append(f"{metric_name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{metric_name} {value}")

            # Export histograms
            for metric_name, hist_values in self.histograms.items():
                definition = self.metric_definitions.get(metric_name)
                if definition:
                    lines.append(f"# HELP {metric_name} {definition.description}")
                    lines.append(f"# TYPE {metric_name} histogram")

                for label_key, hist in hist_values.items():
                    base_labels = label_key
                    for bucket_le, count in sorted(hist.buckets.items()):
                        bucket_label = f'le="{bucket_le}"'
                        if base_labels:
                            lines.append(
                                f"{metric_name}_bucket{{{base_labels},{bucket_label}}} {count}"
                            )
                        else:
                            lines.append(f"{metric_name}_bucket{{{bucket_label}}} {count}")

                    # +Inf bucket
                    if base_labels:
                        lines.append(f'{metric_name}_bucket{{{base_labels},le="+Inf"}} {hist.count}')
                        lines.append(f"{metric_name}_sum{{{base_labels}}} {hist.sum_value}")
                        lines.append(f"{metric_name}_count{{{base_labels}}} {hist.count}")
                    else:
                        lines.append(f'{metric_name}_bucket{{le="+Inf"}} {hist.count}')
                        lines.append(f"{metric_name}_sum {hist.sum_value}")
                        lines.append(f"{metric_name}_count {hist.count}")

        return "\n".join(lines)

    def get_alert_metrics(self) -> AlertMetrics:
        """
        Get aggregated alert metrics.

        Returns:
            AlertMetrics summary object
        """
        with self._lock:
            metrics = AlertMetrics()

            # Count totals
            created_name = self._get_full_metric_name("alerts_total")
            for label_key, count in self.counters[created_name].items():
                metrics.total_alerts_created += int(count)

                # Parse labels
                if "alert_code=" in label_key:
                    for part in label_key.split(","):
                        if part.startswith("alert_code="):
                            code = part.split("=")[1].strip('"')
                            metrics.alerts_by_code[code] = (
                                metrics.alerts_by_code.get(code, 0) + int(count)
                            )
                        elif part.startswith("severity="):
                            sev = part.split("=")[1].strip('"')
                            metrics.alerts_by_severity[sev] = (
                                metrics.alerts_by_severity.get(sev, 0) + int(count)
                            )

            # Count acknowledged
            ack_name = self._get_full_metric_name("alerts_acknowledged_total")
            metrics.acknowledged_count = int(sum(self.counters[ack_name].values()))

            # Count false positives
            fp_name = self._get_full_metric_name("alerts_false_positive_total")
            metrics.false_positive_count = int(sum(self.counters[fp_name].values()))

            # Calculate false positive rate
            if metrics.total_alerts_created > 0:
                metrics.false_positive_rate = (
                    metrics.false_positive_count / metrics.total_alerts_created
                )

            # Count escalations
            esc_name = self._get_full_metric_name("alerts_escalated_total")
            metrics.escalation_count = int(sum(self.counters[esc_name].values()))

            # Count suppressions
            sup_name = self._get_full_metric_name("alerts_suppressed_total")
            metrics.suppression_count = int(sum(self.counters[sup_name].values()))

            # Calculate average acknowledgement time
            if self._acknowledgement_times:
                recent_times = [
                    t[2] for t in self._acknowledgement_times
                    if t[0] > datetime.utcnow() - timedelta(hours=self.config.retention_hours)
                ]
                if recent_times:
                    metrics.avg_acknowledgement_time_seconds = sum(recent_times) / len(recent_times)

            # Calculate average resolution time
            if self._resolution_times:
                recent_times = [
                    t[2] for t in self._resolution_times
                    if t[0] > datetime.utcnow() - timedelta(hours=self.config.retention_hours)
                ]
                if recent_times:
                    metrics.avg_resolution_time_seconds = sum(recent_times) / len(recent_times)

            return metrics

    def get_pipeline_metrics(self) -> PipelineMetrics:
        """
        Get aggregated pipeline metrics.

        Returns:
            PipelineMetrics summary object
        """
        with self._lock:
            metrics = PipelineMetrics()

            # Count processed messages
            proc_name = self._get_full_metric_name("pipeline_messages_processed_total")
            metrics.messages_processed = int(sum(self.counters[proc_name].values()))

            # Count dropped messages
            drop_name = self._get_full_metric_name("pipeline_messages_dropped_total")
            metrics.messages_dropped = int(sum(self.counters[drop_name].values()))

            # Calculate drop rate
            total = metrics.messages_processed + metrics.messages_dropped
            if total > 0:
                metrics.drop_rate = metrics.messages_dropped / total

            # Get current lag
            lag_name = self._get_full_metric_name("pipeline_lag_seconds")
            if self.gauges[lag_name]:
                metrics.current_lag_seconds = max(self.gauges[lag_name].values())
                metrics.max_lag_seconds = metrics.current_lag_seconds

            # Count schema errors
            schema_name = self._get_full_metric_name("pipeline_schema_errors_total")
            metrics.schema_errors = int(sum(self.counters[schema_name].values()))

            # Count missing fields
            missing_name = self._get_full_metric_name("pipeline_missing_fields_total")
            metrics.missing_fields_count = int(sum(self.counters[missing_name].values()))

            # Calculate missingness rate
            if metrics.messages_processed > 0:
                metrics.missingness_rate = metrics.missing_fields_count / metrics.messages_processed

            # Count backpressure events
            bp_name = self._get_full_metric_name("pipeline_backpressure_events_total")
            metrics.backpressure_events = int(sum(self.counters[bp_name].values()))

            return metrics

    def get_model_metrics(self, model_name: str) -> ModelMetrics:
        """
        Get aggregated model metrics.

        Args:
            model_name: Name of the model

        Returns:
            ModelMetrics summary object
        """
        with self._lock:
            metrics = ModelMetrics(model_name=model_name)

            # Count predictions
            pred_name = self._get_full_metric_name("model_predictions_total")
            for label_key, count in self.counters[pred_name].items():
                if f'model_name="{model_name}"' in label_key:
                    metrics.predictions_made += int(count)

            # Get drift score
            drift_name = self._get_full_metric_name("model_drift_score")
            for label_key, score in self.gauges[drift_name].items():
                if f'model_name="{model_name}"' in label_key:
                    if 'feature="overall"' in label_key:
                        metrics.drift_score = score
                        metrics.drift_detected = score > 0.5
                    else:
                        # Extract feature name
                        for part in label_key.split(","):
                            if part.startswith("feature="):
                                feature = part.split("=")[1].strip('"')
                                if feature != "overall":
                                    metrics.feature_drift_scores[feature] = score

            # Get explanation stability
            stab_name = self._get_full_metric_name("model_explanation_stability")
            label_key = self._get_label_key({"model_name": model_name})
            metrics.explanation_stability_score = self.gauges[stab_name].get(label_key, 0.0)

            # Get accuracy
            acc_name = self._get_full_metric_name("model_accuracy")
            metrics.accuracy_estimate = self.gauges[acc_name].get(label_key, 0.0)

            # Calculate average inference time
            if self._inference_times:
                recent_times = [
                    t[2] for t in self._inference_times
                    if t[1] == model_name and
                    t[0] > datetime.utcnow() - timedelta(hours=self.config.retention_hours)
                ]
                if recent_times:
                    metrics.avg_inference_time_ms = (sum(recent_times) / len(recent_times)) * 1000

            return metrics

    def cleanup_old_data(self) -> None:
        """Clean up old time series data beyond retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.config.retention_hours)

        with self._lock:
            self._acknowledgement_times = [
                t for t in self._acknowledgement_times if t[0] > cutoff
            ]
            self._resolution_times = [
                t for t in self._resolution_times if t[0] > cutoff
            ]
            self._inference_times = [
                t for t in self._inference_times if t[0] > cutoff
            ]

        logger.debug("Cleaned up old metrics data")
