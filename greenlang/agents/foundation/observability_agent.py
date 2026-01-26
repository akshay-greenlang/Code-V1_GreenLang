# -*- coding: utf-8 -*-
"""
GL-FOUND-X-010: GreenLang Observability Agent
=============================================

The observability infrastructure agent for GreenLang Climate OS.
Provides comprehensive metrics, logging, tracing, and alerting capabilities.

Capabilities:
    - Prometheus-compatible metrics collection and export
    - OpenTelemetry distributed tracing integration
    - JSON structured logging with correlation IDs
    - Health check probes (liveness and readiness)
    - Threshold-based alert generation
    - Grafana dashboard data provisioning

Zero-Hallucination Guarantees:
    - All metrics are deterministically calculated from source data
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the calculation path
    - All timestamps are traceable and reproducible

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics supported by the observability agent."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Severity levels for generated alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Status of an alert."""
    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class TraceStatus(str, Enum):
    """Status of a distributed trace span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


# Pre-defined metric names for GreenLang platform
PLATFORM_METRICS = {
    "agent_execution_duration_seconds": {
        "type": MetricType.HISTOGRAM,
        "description": "Duration of agent execution in seconds",
        "buckets": [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    },
    "agent_execution_total": {
        "type": MetricType.COUNTER,
        "description": "Total number of agent executions",
    },
    "agent_errors_total": {
        "type": MetricType.COUNTER,
        "description": "Total number of agent errors",
    },
    "lineage_completeness_ratio": {
        "type": MetricType.GAUGE,
        "description": "Ratio of data points with complete lineage (0.0 to 1.0)",
    },
    "zero_hallucination_compliance": {
        "type": MetricType.GAUGE,
        "description": "Compliance score for zero-hallucination requirements (0.0 to 1.0)",
    },
}

# Standard label keys
STANDARD_LABELS = ["agent_id", "agent_type", "tenant_id", "status"]


# =============================================================================
# Pydantic Models - Input/Output Data Structures
# =============================================================================

class MetricLabel(BaseModel):
    """A label key-value pair for a metric."""
    key: str = Field(..., description="Label key")
    value: str = Field(..., description="Label value")


class MetricDefinition(BaseModel):
    """Definition of a metric to collect."""
    name: str = Field(..., description="Metric name (e.g., agent_execution_total)")
    type: MetricType = Field(..., description="Type of metric")
    description: str = Field(default="", description="Human-readable description")
    labels: List[str] = Field(default_factory=list, description="Label keys")
    buckets: Optional[List[float]] = Field(
        None, description="Histogram bucket boundaries"
    )

    @validator('name')
    def validate_metric_name(cls, v):
        """Validate metric name follows Prometheus naming conventions."""
        if not v or not v[0].isalpha():
            raise ValueError("Metric name must start with a letter")
        for char in v:
            if not (char.isalnum() or char == '_'):
                raise ValueError(f"Invalid character in metric name: {char}")
        return v


class MetricValue(BaseModel):
    """A metric observation with labels and value."""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    labels: Dict[str, str] = Field(default_factory=dict, description="Label values")
    timestamp: Optional[datetime] = Field(None, description="Observation timestamp")


class TraceContext(BaseModel):
    """OpenTelemetry-compatible trace context."""
    trace_id: str = Field(..., description="128-bit trace ID as hex string")
    span_id: str = Field(..., description="64-bit span ID as hex string")
    parent_span_id: Optional[str] = Field(None, description="Parent span ID")
    trace_flags: int = Field(default=1, description="Trace flags (1 = sampled)")
    trace_state: Optional[str] = Field(None, description="W3C trace state")


class SpanDefinition(BaseModel):
    """Definition of a trace span."""
    name: str = Field(..., description="Span name")
    context: TraceContext = Field(..., description="Trace context")
    kind: str = Field(default="internal", description="Span kind")
    start_time: datetime = Field(..., description="Span start time")
    end_time: Optional[datetime] = Field(None, description="Span end time")
    status: TraceStatus = Field(default=TraceStatus.UNSET, description="Span status")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Span attributes")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Span events")


class LogEntry(BaseModel):
    """Structured log entry."""
    timestamp: datetime = Field(..., description="Log timestamp")
    level: str = Field(..., description="Log level (DEBUG, INFO, WARNING, ERROR)")
    message: str = Field(..., description="Log message")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    trace_id: Optional[str] = Field(None, description="Associated trace ID")
    span_id: Optional[str] = Field(None, description="Associated span ID")
    agent_id: Optional[str] = Field(None, description="Agent that generated the log")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")


class AlertRule(BaseModel):
    """Definition of an alert rule."""
    name: str = Field(..., description="Alert rule name")
    metric_name: str = Field(..., description="Metric to monitor")
    condition: str = Field(..., description="Condition operator (gt, lt, eq, gte, lte)")
    threshold: float = Field(..., description="Threshold value")
    duration_seconds: int = Field(default=60, description="Duration before firing")
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING, description="Alert severity")
    labels: Dict[str, str] = Field(default_factory=dict, description="Filter labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")

    @validator('condition')
    def validate_condition(cls, v):
        """Validate condition operator."""
        valid_conditions = {'gt', 'lt', 'eq', 'gte', 'lte', 'ne'}
        if v not in valid_conditions:
            raise ValueError(f"Condition must be one of: {valid_conditions}")
        return v


class Alert(BaseModel):
    """A generated alert instance."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_name: str = Field(..., description="Source alert rule name")
    status: AlertStatus = Field(default=AlertStatus.FIRING, description="Alert status")
    severity: AlertSeverity = Field(..., description="Alert severity")
    metric_name: str = Field(..., description="Metric that triggered alert")
    metric_value: float = Field(..., description="Value that triggered alert")
    threshold: float = Field(..., description="Threshold that was crossed")
    labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")
    started_at: datetime = Field(..., description="When alert started firing")
    resolved_at: Optional[datetime] = Field(None, description="When alert was resolved")


class HealthCheck(BaseModel):
    """Health check result."""
    name: str = Field(..., description="Health check name")
    status: HealthStatus = Field(..., description="Health status")
    message: Optional[str] = Field(None, description="Status message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    last_check: datetime = Field(..., description="Last check timestamp")
    duration_ms: float = Field(default=0.0, description="Check duration in ms")


class DashboardPanel(BaseModel):
    """Data for a Grafana dashboard panel."""
    panel_id: str = Field(..., description="Panel identifier")
    title: str = Field(..., description="Panel title")
    panel_type: str = Field(default="graph", description="Panel type (graph, gauge, table)")
    metric_queries: List[str] = Field(default_factory=list, description="Prometheus queries")
    time_range: str = Field(default="1h", description="Time range for data")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Panel data")


class ObservabilityInput(BaseModel):
    """Input data for the Observability Agent."""
    operation: str = Field(..., description="Operation to perform")
    metric: Optional[MetricValue] = Field(None, description="Metric to record")
    span: Optional[SpanDefinition] = Field(None, description="Span to record")
    log_entry: Optional[LogEntry] = Field(None, description="Log entry to record")
    alert_rule: Optional[AlertRule] = Field(None, description="Alert rule to add")
    health_check_name: Optional[str] = Field(None, description="Health check to run")
    dashboard_query: Optional[Dict[str, Any]] = Field(None, description="Dashboard query")

    @validator('operation')
    def validate_operation(cls, v):
        """Validate operation is supported."""
        valid_ops = {
            'record_metric', 'increment_counter', 'set_gauge',
            'observe_histogram', 'start_span', 'end_span',
            'log', 'add_alert_rule', 'check_alerts',
            'liveness_probe', 'readiness_probe',
            'get_dashboard_data', 'export_metrics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class ObservabilityOutput(BaseModel):
    """Output from the Observability Agent."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation that was performed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    processing_time_ms: float = Field(default=0.0, description="Processing duration")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Internal Data Structures
# =============================================================================

@dataclass
class MetricSeries:
    """Internal storage for a metric time series."""
    name: str
    type: MetricType
    description: str
    label_keys: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None

    # Storage for metric values by label set
    values: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    sums: Dict[str, float] = field(default_factory=dict)
    bucket_counts: Dict[str, Dict[float, int]] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=DeterministicClock.now)
    updated_at: datetime = field(default_factory=DeterministicClock.now)


@dataclass
class ActiveSpan:
    """Represents an active trace span."""
    span: SpanDefinition
    children: List[str] = field(default_factory=list)


# =============================================================================
# Main Observability Agent Implementation
# =============================================================================

class ObservabilityAgent(BaseAgent):
    """
    GL-FOUND-X-010: GreenLang Observability Agent

    Provides comprehensive observability infrastructure for GreenLang Climate OS.
    Handles metrics collection, distributed tracing, structured logging,
    health checks, and alert generation.

    Zero-Hallucination Guarantees:
        - All metrics calculated deterministically from source data
        - Complete provenance tracking with SHA-256 hashes
        - No LLM calls in the metric calculation path
        - All data has complete lineage

    Usage:
        agent = ObservabilityAgent()

        # Record a metric
        result = agent.run({
            "operation": "record_metric",
            "metric": {
                "name": "agent_execution_total",
                "value": 1.0,
                "labels": {"agent_id": "GL-MRV-001", "status": "success"}
            }
        })

        # Start a trace span
        result = agent.run({
            "operation": "start_span",
            "span": {
                "name": "process_emissions",
                "context": {"trace_id": "abc123", "span_id": "def456"},
                "start_time": "2024-01-01T00:00:00Z"
            }
        })
    """

    AGENT_ID = "GL-FOUND-X-010"
    AGENT_NAME = "GreenLang Observability Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the Observability Agent.

        Args:
            config: Optional agent configuration
        """
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Observability infrastructure for GreenLang Climate OS",
                version=self.VERSION,
                parameters={
                    "metrics_retention_hours": 24,
                    "max_active_spans": 10000,
                    "alert_evaluation_interval_seconds": 60,
                    "enable_prometheus_export": True,
                    "enable_opentelemetry": True,
                }
            )
        super().__init__(config)

        # Initialize metrics storage
        self._metrics: Dict[str, MetricSeries] = {}
        self._init_platform_metrics()

        # Initialize tracing storage
        self._active_spans: Dict[str, ActiveSpan] = {}
        self._completed_spans: List[SpanDefinition] = []
        self._span_limit = config.parameters.get("max_active_spans", 10000)

        # Initialize logging storage
        self._log_buffer: List[LogEntry] = []
        self._log_buffer_size = 10000

        # Initialize alerting
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []

        # Initialize health checks
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._last_health_status: Dict[str, HealthCheck] = {}
        self._register_default_health_checks()

        # Statistics tracking
        self._total_metrics_recorded = 0
        self._total_spans_created = 0
        self._total_logs_recorded = 0
        self._total_alerts_fired = 0

        self.logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def _init_platform_metrics(self):
        """Initialize pre-defined platform metrics."""
        for name, definition in PLATFORM_METRICS.items():
            self._metrics[name] = MetricSeries(
                name=name,
                type=definition["type"],
                description=definition["description"],
                label_keys=STANDARD_LABELS.copy(),
                buckets=definition.get("buckets"),
            )
        self.logger.debug(f"Initialized {len(PLATFORM_METRICS)} platform metrics")

    def _register_default_health_checks(self):
        """Register default health checks."""
        self._health_checks["metrics_store"] = self._check_metrics_store_health
        self._health_checks["span_store"] = self._check_span_store_health
        self._health_checks["log_buffer"] = self._check_log_buffer_health

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute an observability operation.

        Args:
            input_data: Input containing operation and relevant data

        Returns:
            AgentResult with operation results
        """
        start_time = time.time()

        try:
            # Parse and validate input
            obs_input = ObservabilityInput(**input_data)
            operation = obs_input.operation

            # Route to appropriate handler
            result_data = self._route_operation(obs_input)

            # Calculate provenance hash
            provenance_hash = self._compute_provenance_hash(input_data, result_data)

            processing_time_ms = (time.time() - start_time) * 1000

            output = ObservabilityOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            self.logger.error(f"Observability operation failed: {e}", exc_info=True)
            processing_time_ms = (time.time() - start_time) * 1000

            return AgentResult(
                success=False,
                error=str(e),
                data={
                    "operation": input_data.get("operation", "unknown"),
                    "processing_time_ms": processing_time_ms,
                },
            )

    def _route_operation(self, obs_input: ObservabilityInput) -> Dict[str, Any]:
        """
        Route operation to appropriate handler.

        Args:
            obs_input: Validated input

        Returns:
            Operation result data
        """
        operation = obs_input.operation

        if operation == "record_metric":
            return self._handle_record_metric(obs_input.metric)
        elif operation == "increment_counter":
            return self._handle_increment_counter(obs_input.metric)
        elif operation == "set_gauge":
            return self._handle_set_gauge(obs_input.metric)
        elif operation == "observe_histogram":
            return self._handle_observe_histogram(obs_input.metric)
        elif operation == "start_span":
            return self._handle_start_span(obs_input.span)
        elif operation == "end_span":
            return self._handle_end_span(obs_input.span)
        elif operation == "log":
            return self._handle_log(obs_input.log_entry)
        elif operation == "add_alert_rule":
            return self._handle_add_alert_rule(obs_input.alert_rule)
        elif operation == "check_alerts":
            return self._handle_check_alerts()
        elif operation == "liveness_probe":
            return self._handle_liveness_probe()
        elif operation == "readiness_probe":
            return self._handle_readiness_probe()
        elif operation == "get_dashboard_data":
            return self._handle_get_dashboard_data(obs_input.dashboard_query)
        elif operation == "export_metrics":
            return self._handle_export_metrics()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    # =========================================================================
    # Metrics Operations
    # =========================================================================

    def _handle_record_metric(self, metric: Optional[MetricValue]) -> Dict[str, Any]:
        """Record a metric value."""
        if metric is None:
            raise ValueError("Metric data required for record_metric operation")

        # Get or create metric series
        if metric.name not in self._metrics:
            self._metrics[metric.name] = MetricSeries(
                name=metric.name,
                type=MetricType.GAUGE,
                description="User-defined metric",
                label_keys=list(metric.labels.keys()),
            )

        series = self._metrics[metric.name]
        label_key = self._labels_to_key(metric.labels)

        # Store value based on metric type
        if series.type == MetricType.COUNTER:
            series.values[label_key] = series.values.get(label_key, 0) + metric.value
        elif series.type == MetricType.GAUGE:
            series.values[label_key] = metric.value
        elif series.type == MetricType.HISTOGRAM:
            self._record_histogram_value(series, label_key, metric.value)
        else:
            series.values[label_key] = metric.value

        series.updated_at = DeterministicClock.now()
        self._total_metrics_recorded += 1

        return {
            "metric_name": metric.name,
            "label_key": label_key,
            "value": series.values.get(label_key, metric.value),
            "type": series.type.value,
        }

    def _handle_increment_counter(self, metric: Optional[MetricValue]) -> Dict[str, Any]:
        """Increment a counter metric."""
        if metric is None:
            raise ValueError("Metric data required for increment_counter operation")

        if metric.name not in self._metrics:
            self._metrics[metric.name] = MetricSeries(
                name=metric.name,
                type=MetricType.COUNTER,
                description="User-defined counter",
                label_keys=list(metric.labels.keys()),
            )

        series = self._metrics[metric.name]
        if series.type != MetricType.COUNTER:
            raise ValueError(f"Metric {metric.name} is not a counter")

        label_key = self._labels_to_key(metric.labels)
        increment = metric.value if metric.value else 1.0
        series.values[label_key] = series.values.get(label_key, 0) + increment
        series.updated_at = DeterministicClock.now()
        self._total_metrics_recorded += 1

        return {
            "metric_name": metric.name,
            "label_key": label_key,
            "new_value": series.values[label_key],
            "increment": increment,
        }

    def _handle_set_gauge(self, metric: Optional[MetricValue]) -> Dict[str, Any]:
        """Set a gauge metric value."""
        if metric is None:
            raise ValueError("Metric data required for set_gauge operation")

        if metric.name not in self._metrics:
            self._metrics[metric.name] = MetricSeries(
                name=metric.name,
                type=MetricType.GAUGE,
                description="User-defined gauge",
                label_keys=list(metric.labels.keys()),
            )

        series = self._metrics[metric.name]
        label_key = self._labels_to_key(metric.labels)
        series.values[label_key] = metric.value
        series.updated_at = DeterministicClock.now()
        self._total_metrics_recorded += 1

        return {
            "metric_name": metric.name,
            "label_key": label_key,
            "value": metric.value,
        }

    def _handle_observe_histogram(self, metric: Optional[MetricValue]) -> Dict[str, Any]:
        """Observe a value in a histogram."""
        if metric is None:
            raise ValueError("Metric data required for observe_histogram operation")

        if metric.name not in self._metrics:
            # Create histogram with default buckets
            self._metrics[metric.name] = MetricSeries(
                name=metric.name,
                type=MetricType.HISTOGRAM,
                description="User-defined histogram",
                label_keys=list(metric.labels.keys()),
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            )

        series = self._metrics[metric.name]
        if series.type != MetricType.HISTOGRAM:
            raise ValueError(f"Metric {metric.name} is not a histogram")

        label_key = self._labels_to_key(metric.labels)
        self._record_histogram_value(series, label_key, metric.value)
        series.updated_at = DeterministicClock.now()
        self._total_metrics_recorded += 1

        return {
            "metric_name": metric.name,
            "label_key": label_key,
            "observed_value": metric.value,
            "count": series.counts.get(label_key, 0),
            "sum": series.sums.get(label_key, 0),
        }

    def _record_histogram_value(
        self, series: MetricSeries, label_key: str, value: float
    ):
        """Record a value in histogram buckets."""
        # Update count and sum
        series.counts[label_key] = series.counts.get(label_key, 0) + 1
        series.sums[label_key] = series.sums.get(label_key, 0) + value

        # Update bucket counts
        if label_key not in series.bucket_counts:
            series.bucket_counts[label_key] = {}
            if series.buckets:
                for bucket in series.buckets:
                    series.bucket_counts[label_key][bucket] = 0
                series.bucket_counts[label_key][float('inf')] = 0

        if series.buckets:
            for bucket in series.buckets:
                if value <= bucket:
                    series.bucket_counts[label_key][bucket] += 1
            series.bucket_counts[label_key][float('inf')] += 1

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert label dict to a deterministic string key."""
        sorted_items = sorted(labels.items())
        return ",".join(f"{k}={v}" for k, v in sorted_items)

    def _handle_export_metrics(self) -> Dict[str, Any]:
        """Export all metrics in Prometheus format."""
        lines = []
        timestamp = DeterministicClock.now()

        for name, series in self._metrics.items():
            # Add HELP and TYPE comments
            lines.append(f"# HELP {name} {series.description}")
            lines.append(f"# TYPE {name} {series.type.value}")

            if series.type == MetricType.HISTOGRAM:
                # Export histogram in Prometheus format
                for label_key, bucket_counts in series.bucket_counts.items():
                    labels = self._key_to_labels_str(label_key)
                    for bucket, count in sorted(bucket_counts.items()):
                        le = "+Inf" if bucket == float('inf') else str(bucket)
                        lines.append(f'{name}_bucket{{{labels},le="{le}"}} {count}')
                    lines.append(f'{name}_count{{{labels}}} {series.counts.get(label_key, 0)}')
                    lines.append(f'{name}_sum{{{labels}}} {series.sums.get(label_key, 0)}')
            else:
                # Export counter/gauge
                for label_key, value in series.values.items():
                    labels = self._key_to_labels_str(label_key)
                    if labels:
                        lines.append(f'{name}{{{labels}}} {value}')
                    else:
                        lines.append(f'{name} {value}')

        return {
            "format": "prometheus",
            "timestamp": timestamp.isoformat(),
            "metrics_count": len(self._metrics),
            "content": "\n".join(lines),
        }

    def _key_to_labels_str(self, label_key: str) -> str:
        """Convert label key back to Prometheus label format."""
        if not label_key:
            return ""
        parts = label_key.split(",")
        return ",".join(parts)

    # =========================================================================
    # Distributed Tracing Operations
    # =========================================================================

    def _handle_start_span(self, span: Optional[SpanDefinition]) -> Dict[str, Any]:
        """Start a new trace span."""
        if span is None:
            raise ValueError("Span data required for start_span operation")

        span_key = f"{span.context.trace_id}:{span.context.span_id}"

        if len(self._active_spans) >= self._span_limit:
            self.logger.warning("Active span limit reached, removing oldest spans")
            self._cleanup_old_spans()

        self._active_spans[span_key] = ActiveSpan(span=span)

        # Link to parent if exists
        if span.context.parent_span_id:
            parent_key = f"{span.context.trace_id}:{span.context.parent_span_id}"
            if parent_key in self._active_spans:
                self._active_spans[parent_key].children.append(span_key)

        self._total_spans_created += 1

        return {
            "trace_id": span.context.trace_id,
            "span_id": span.context.span_id,
            "parent_span_id": span.context.parent_span_id,
            "name": span.name,
            "start_time": span.start_time.isoformat(),
        }

    def _handle_end_span(self, span: Optional[SpanDefinition]) -> Dict[str, Any]:
        """End an active trace span."""
        if span is None:
            raise ValueError("Span data required for end_span operation")

        span_key = f"{span.context.trace_id}:{span.context.span_id}"

        if span_key not in self._active_spans:
            raise ValueError(f"Span not found: {span_key}")

        active_span = self._active_spans[span_key]
        active_span.span.end_time = span.end_time or DeterministicClock.now()
        active_span.span.status = span.status
        active_span.span.attributes.update(span.attributes)

        # Move to completed spans
        self._completed_spans.append(active_span.span)
        del self._active_spans[span_key]

        # Calculate duration
        duration_ms = (
            active_span.span.end_time - active_span.span.start_time
        ).total_seconds() * 1000

        return {
            "trace_id": span.context.trace_id,
            "span_id": span.context.span_id,
            "name": active_span.span.name,
            "duration_ms": duration_ms,
            "status": active_span.span.status.value,
        }

    def _cleanup_old_spans(self):
        """Remove oldest 10% of active spans."""
        if not self._active_spans:
            return

        spans_to_remove = max(1, len(self._active_spans) // 10)
        sorted_spans = sorted(
            self._active_spans.items(),
            key=lambda x: x[1].span.start_time
        )

        for span_key, _ in sorted_spans[:spans_to_remove]:
            del self._active_spans[span_key]

    # =========================================================================
    # Structured Logging Operations
    # =========================================================================

    def _handle_log(self, log_entry: Optional[LogEntry]) -> Dict[str, Any]:
        """Record a structured log entry."""
        if log_entry is None:
            raise ValueError("Log entry required for log operation")

        # Ensure correlation ID
        if not log_entry.correlation_id:
            log_entry.correlation_id = str(uuid.uuid4())

        # Add to buffer
        self._log_buffer.append(log_entry)
        self._total_logs_recorded += 1

        # Trim buffer if needed
        if len(self._log_buffer) > self._log_buffer_size:
            self._log_buffer = self._log_buffer[-self._log_buffer_size:]

        # Convert to JSON for structured output
        log_json = {
            "timestamp": log_entry.timestamp.isoformat(),
            "level": log_entry.level,
            "message": log_entry.message,
            "correlation_id": log_entry.correlation_id,
            "trace_id": log_entry.trace_id,
            "span_id": log_entry.span_id,
            "agent_id": log_entry.agent_id,
            "tenant_id": log_entry.tenant_id,
            **log_entry.attributes,
        }

        return {
            "correlation_id": log_entry.correlation_id,
            "logged": True,
            "buffer_size": len(self._log_buffer),
            "json": json.dumps(log_json),
        }

    # =========================================================================
    # Alert Operations
    # =========================================================================

    def _handle_add_alert_rule(self, rule: Optional[AlertRule]) -> Dict[str, Any]:
        """Add or update an alert rule."""
        if rule is None:
            raise ValueError("Alert rule required for add_alert_rule operation")

        self._alert_rules[rule.name] = rule

        return {
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "severity": rule.severity.value,
            "total_rules": len(self._alert_rules),
        }

    def _handle_check_alerts(self) -> Dict[str, Any]:
        """Evaluate all alert rules and return firing alerts."""
        now = DeterministicClock.now()
        new_alerts = []
        resolved_alerts = []

        for rule_name, rule in self._alert_rules.items():
            metric = self._metrics.get(rule.metric_name)
            if metric is None:
                continue

            # Check each label combination
            for label_key, value in metric.values.items():
                # Check if labels match rule filters
                if not self._labels_match(label_key, rule.labels):
                    continue

                # Evaluate condition
                is_firing = self._evaluate_condition(value, rule.condition, rule.threshold)

                alert_key = f"{rule_name}:{label_key}"

                if is_firing:
                    if alert_key not in self._active_alerts:
                        # New alert
                        alert = Alert(
                            rule_name=rule_name,
                            severity=rule.severity,
                            metric_name=rule.metric_name,
                            metric_value=value,
                            threshold=rule.threshold,
                            labels=self._key_to_labels_dict(label_key),
                            annotations=rule.annotations,
                            started_at=now,
                        )
                        self._active_alerts[alert_key] = alert
                        new_alerts.append(alert)
                        self._total_alerts_fired += 1
                else:
                    if alert_key in self._active_alerts:
                        # Resolve alert
                        alert = self._active_alerts[alert_key]
                        alert.status = AlertStatus.RESOLVED
                        alert.resolved_at = now
                        resolved_alerts.append(alert)
                        self._alert_history.append(alert)
                        del self._active_alerts[alert_key]

        return {
            "evaluated_rules": len(self._alert_rules),
            "active_alerts": len(self._active_alerts),
            "new_alerts": [a.model_dump() for a in new_alerts],
            "resolved_alerts": [a.model_dump() for a in resolved_alerts],
            "timestamp": now.isoformat(),
        }

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate a condition against a threshold."""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "ne":
            return value != threshold
        return False

    def _labels_match(self, label_key: str, filter_labels: Dict[str, str]) -> bool:
        """Check if label key matches filter labels."""
        if not filter_labels:
            return True
        labels = self._key_to_labels_dict(label_key)
        for key, value in filter_labels.items():
            if labels.get(key) != value:
                return False
        return True

    def _key_to_labels_dict(self, label_key: str) -> Dict[str, str]:
        """Convert label key string to dictionary."""
        if not label_key:
            return {}
        labels = {}
        for part in label_key.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                labels[key] = value
        return labels

    # =========================================================================
    # Health Check Operations
    # =========================================================================

    def _handle_liveness_probe(self) -> Dict[str, Any]:
        """Check if the agent is alive."""
        now = DeterministicClock.now()
        start = time.time()

        # Simple liveness: can we respond?
        health = HealthCheck(
            name="liveness",
            status=HealthStatus.HEALTHY,
            message="Observability agent is alive",
            last_check=now,
            duration_ms=(time.time() - start) * 1000,
        )

        return {
            "status": health.status.value,
            "message": health.message,
            "timestamp": now.isoformat(),
        }

    def _handle_readiness_probe(self) -> Dict[str, Any]:
        """Check if the agent is ready to accept traffic."""
        now = DeterministicClock.now()
        start = time.time()

        checks = {}
        overall_status = HealthStatus.HEALTHY

        for check_name, check_func in self._health_checks.items():
            try:
                result = check_func()
                checks[check_name] = {
                    "status": result.status.value,
                    "message": result.message,
                }
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED
            except Exception as e:
                checks[check_name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": str(e),
                }
                overall_status = HealthStatus.UNHEALTHY

        return {
            "status": overall_status.value,
            "checks": checks,
            "timestamp": now.isoformat(),
            "duration_ms": (time.time() - start) * 1000,
        }

    def _check_metrics_store_health(self) -> HealthCheck:
        """Check health of metrics storage."""
        now = DeterministicClock.now()
        metrics_count = len(self._metrics)

        return HealthCheck(
            name="metrics_store",
            status=HealthStatus.HEALTHY,
            message=f"Metrics store has {metrics_count} series",
            details={"series_count": metrics_count},
            last_check=now,
        )

    def _check_span_store_health(self) -> HealthCheck:
        """Check health of span storage."""
        now = DeterministicClock.now()
        active_count = len(self._active_spans)
        utilization = active_count / self._span_limit if self._span_limit else 0

        if utilization > 0.9:
            status = HealthStatus.DEGRADED
            message = f"Span store at {utilization:.1%} capacity"
        else:
            status = HealthStatus.HEALTHY
            message = f"Span store at {utilization:.1%} capacity"

        return HealthCheck(
            name="span_store",
            status=status,
            message=message,
            details={
                "active_spans": active_count,
                "limit": self._span_limit,
                "utilization": utilization,
            },
            last_check=now,
        )

    def _check_log_buffer_health(self) -> HealthCheck:
        """Check health of log buffer."""
        now = DeterministicClock.now()
        buffer_size = len(self._log_buffer)
        utilization = buffer_size / self._log_buffer_size if self._log_buffer_size else 0

        if utilization > 0.9:
            status = HealthStatus.DEGRADED
            message = f"Log buffer at {utilization:.1%} capacity"
        else:
            status = HealthStatus.HEALTHY
            message = f"Log buffer at {utilization:.1%} capacity"

        return HealthCheck(
            name="log_buffer",
            status=status,
            message=message,
            details={
                "buffer_size": buffer_size,
                "limit": self._log_buffer_size,
                "utilization": utilization,
            },
            last_check=now,
        )

    # =========================================================================
    # Dashboard Data Operations
    # =========================================================================

    def _handle_get_dashboard_data(
        self, query: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get data for Grafana dashboards."""
        if query is None:
            query = {}

        metric_name = query.get("metric_name")
        time_range = query.get("time_range", "1h")
        labels_filter = query.get("labels", {})

        panels = []

        if metric_name:
            # Get specific metric
            if metric_name in self._metrics:
                series = self._metrics[metric_name]
                panels.append(self._build_panel_data(series, labels_filter))
        else:
            # Get all platform metrics
            for name, series in self._metrics.items():
                if name in PLATFORM_METRICS:
                    panels.append(self._build_panel_data(series, labels_filter))

        # Add summary statistics
        summary = {
            "total_metrics_recorded": self._total_metrics_recorded,
            "total_spans_created": self._total_spans_created,
            "total_logs_recorded": self._total_logs_recorded,
            "total_alerts_fired": self._total_alerts_fired,
            "active_alerts": len(self._active_alerts),
            "active_spans": len(self._active_spans),
        }

        return {
            "panels": panels,
            "summary": summary,
            "time_range": time_range,
            "timestamp": DeterministicClock.now().isoformat(),
        }

    def _build_panel_data(
        self, series: MetricSeries, labels_filter: Dict[str, str]
    ) -> Dict[str, Any]:
        """Build dashboard panel data for a metric series."""
        data_points = []

        for label_key, value in series.values.items():
            if self._labels_match(label_key, labels_filter):
                data_points.append({
                    "labels": self._key_to_labels_dict(label_key),
                    "value": value,
                })

        return {
            "metric_name": series.name,
            "type": series.type.value,
            "description": series.description,
            "data_points": data_points,
            "updated_at": series.updated_at.isoformat(),
        }

    # =========================================================================
    # Provenance and Utility Methods
    # =========================================================================

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        """
        Compute SHA-256 hash for complete audit trail.

        Args:
            input_data: Input data
            output_data: Output data

        Returns:
            SHA-256 hash string (first 16 characters)
        """
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

    # =========================================================================
    # Convenience Methods for Direct API Usage
    # =========================================================================

    def record_agent_execution(
        self,
        agent_id: str,
        agent_type: str,
        tenant_id: str,
        status: str,
        duration_seconds: float,
    ):
        """
        Record an agent execution with all standard metrics.

        Args:
            agent_id: Agent identifier
            agent_type: Type of agent
            tenant_id: Tenant identifier
            status: Execution status (success/failure)
            duration_seconds: Execution duration
        """
        labels = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "tenant_id": tenant_id,
            "status": status,
        }

        # Record execution total
        self.run({
            "operation": "increment_counter",
            "metric": {
                "name": "agent_execution_total",
                "value": 1,
                "labels": labels,
            },
        })

        # Record duration histogram
        self.run({
            "operation": "observe_histogram",
            "metric": {
                "name": "agent_execution_duration_seconds",
                "value": duration_seconds,
                "labels": labels,
            },
        })

        # Record error if failed
        if status != "success":
            self.run({
                "operation": "increment_counter",
                "metric": {
                    "name": "agent_errors_total",
                    "value": 1,
                    "labels": labels,
                },
            })

    def create_trace(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new trace and return trace context.

        Args:
            name: Trace name
            attributes: Optional trace attributes

        Returns:
            Trace ID
        """
        trace_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex[:16]

        self.run({
            "operation": "start_span",
            "span": {
                "name": name,
                "context": {
                    "trace_id": trace_id,
                    "span_id": span_id,
                },
                "start_time": DeterministicClock.now().isoformat(),
                "attributes": attributes or {},
            },
        })

        return trace_id

    def log_structured(
        self,
        level: str,
        message: str,
        agent_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **attributes,
    ):
        """
        Log a structured message.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message
            agent_id: Optional agent ID
            tenant_id: Optional tenant ID
            trace_id: Optional trace ID for correlation
            **attributes: Additional log attributes
        """
        self.run({
            "operation": "log",
            "log_entry": {
                "timestamp": DeterministicClock.now().isoformat(),
                "level": level.upper(),
                "message": message,
                "agent_id": agent_id,
                "tenant_id": tenant_id,
                "trace_id": trace_id,
                "attributes": attributes,
            },
        })

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Summary dictionary with metric statistics
        """
        return {
            "total_series": len(self._metrics),
            "total_recorded": self._total_metrics_recorded,
            "platform_metrics": list(PLATFORM_METRICS.keys()),
            "custom_metrics": [
                name for name in self._metrics.keys()
                if name not in PLATFORM_METRICS
            ],
        }

    def get_active_alerts_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all active alerts.

        Returns:
            List of active alert summaries
        """
        return [
            {
                "alert_id": alert.alert_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "started_at": alert.started_at.isoformat(),
            }
            for alert in self._active_alerts.values()
        ]
