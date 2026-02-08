# -*- coding: utf-8 -*-
"""
Observability Agent Service Data Models - AGENT-FOUND-010: Observability & Telemetry Agent

Pydantic v2 data models for the Observability & Telemetry Agent SDK. Re-exports
the Layer 1 enumerations and models from the foundation agent, and defines
additional SDK models for metric recordings, trace records, log records, alert
instances, health probes, dashboard configs, SLO definitions, SLO status,
statistics, and request/response wrappers.

Models:
    - Re-exported enums: MetricType, AlertSeverity, AlertStatus, HealthStatus,
        TraceStatus
    - Re-exported Layer 1: MetricLabel, MetricDefinition, MetricValue,
        TraceContext, SpanDefinition, LogEntry, AlertRule, Alert, HealthCheck,
        DashboardPanel, ObservabilityInput, ObservabilityOutput
    - New enums: LogLevel, SLOType, ProbeType
    - SDK models: MetricRecording, TraceRecord, LogRecord, AlertInstance,
        HealthProbeResult, DashboardConfig, SLODefinition, SLOStatus,
        ObservabilityStatistics
    - Request/Response: RecordMetricRequest, CreateSpanRequest,
        IngestLogRequest, CreateAlertRuleRequest, CreateSLORequest,
        HealthCheckRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 enumerations
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.observability_agent import (
    MetricType,
    AlertSeverity,
    AlertStatus,
    HealthStatus,
    TraceStatus,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 models
# ---------------------------------------------------------------------------

from greenlang.agents.foundation.observability_agent import (
    MetricLabel,
    MetricDefinition,
    MetricValue,
    TraceContext,
    SpanDefinition,
    LogEntry,
    AlertRule,
    Alert,
    HealthCheck,
    DashboardPanel,
    ObservabilityInput,
    ObservabilityOutput,
)


# ---------------------------------------------------------------------------
# New enumerations (SDK-level)
# ---------------------------------------------------------------------------


class LogLevel(str, Enum):
    """Log severity levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SLOType(str, Enum):
    """Types of Service Level Objectives."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SATURATION = "saturation"


class ProbeType(str, Enum):
    """Types of health probes for Kubernetes-style health checking."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# SDK Data Models
# =============================================================================


class MetricRecording(BaseModel):
    """Record of a single metric observation for persistent storage.

    Captures the full context of a metric data point including labels,
    tenant isolation, and provenance tracking.

    Attributes:
        recording_id: Unique identifier for this metric recording.
        metric_name: Prometheus-compatible metric name.
        value: Numeric metric value.
        labels: Key-value label pairs for metric dimensionality.
        timestamp: Timestamp when the metric was recorded.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 hash for audit trail.
    """

    recording_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this metric recording",
    )
    metric_name: str = Field(
        ..., description="Prometheus-compatible metric name",
    )
    value: float = Field(
        ..., description="Numeric metric value",
    )
    labels: Dict[str, str] = Field(
        default_factory=dict, description="Key-value label pairs",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the metric was recorded",
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v: str) -> str:
        """Validate metric_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("metric_name must be non-empty")
        return v


class TraceRecord(BaseModel):
    """Record of a distributed trace span for persistent storage.

    Captures the full lifecycle of a trace span including timing,
    parent-child relationships, and associated attributes.

    Attributes:
        record_id: Unique identifier for this trace record.
        trace_id: 128-bit trace ID as hex string.
        span_id: 64-bit span ID as hex string.
        parent_span_id: Parent span ID for parent-child linking.
        operation_name: Name of the operation this span represents.
        service_name: Name of the service that produced this span.
        status: Span completion status.
        start_time: Span start timestamp.
        end_time: Span end timestamp.
        duration_ms: Span duration in milliseconds.
        attributes: Key-value span attributes.
        events: List of span events (logs, exceptions, etc.).
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 hash for audit trail.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this trace record",
    )
    trace_id: str = Field(
        ..., description="128-bit trace ID as hex string",
    )
    span_id: str = Field(
        ..., description="64-bit span ID as hex string",
    )
    parent_span_id: Optional[str] = Field(
        None, description="Parent span ID for parent-child linking",
    )
    operation_name: str = Field(
        ..., description="Name of the operation this span represents",
    )
    service_name: str = Field(
        default="", description="Name of the service that produced this span",
    )
    status: TraceStatus = Field(
        default=TraceStatus.UNSET, description="Span completion status",
    )
    start_time: datetime = Field(
        default_factory=_utcnow, description="Span start timestamp",
    )
    end_time: Optional[datetime] = Field(
        None, description="Span end timestamp",
    )
    duration_ms: float = Field(
        default=0.0, description="Span duration in milliseconds",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Key-value span attributes",
    )
    events: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of span events",
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, v: str) -> str:
        """Validate trace_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("trace_id must be non-empty")
        return v

    @field_validator("span_id")
    @classmethod
    def validate_span_id(cls, v: str) -> str:
        """Validate span_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("span_id must be non-empty")
        return v

    @field_validator("operation_name")
    @classmethod
    def validate_operation_name(cls, v: str) -> str:
        """Validate operation_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("operation_name must be non-empty")
        return v


class LogRecord(BaseModel):
    """Record of a structured log entry for persistent storage.

    Captures a complete structured log entry with correlation IDs
    for trace linking and tenant isolation.

    Attributes:
        record_id: Unique identifier for this log record.
        timestamp: Timestamp when the log entry was created.
        level: Log severity level.
        message: Human-readable log message.
        correlation_id: Correlation ID for request tracing.
        trace_id: Associated distributed trace ID.
        span_id: Associated span ID within the trace.
        agent_id: ID of the agent that generated the log.
        tenant_id: Tenant identifier for multi-tenant isolation.
        attributes: Additional structured key-value attributes.
        provenance_hash: SHA-256 hash for audit trail.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this log record",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the log entry was created",
    )
    level: LogLevel = Field(
        default=LogLevel.INFO, description="Log severity level",
    )
    message: str = Field(
        ..., description="Human-readable log message",
    )
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for request tracing",
    )
    trace_id: Optional[str] = Field(
        None, description="Associated distributed trace ID",
    )
    span_id: Optional[str] = Field(
        None, description="Associated span ID within the trace",
    )
    agent_id: Optional[str] = Field(
        None, description="ID of the agent that generated the log",
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Additional structured key-value attributes",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message is non-empty."""
        if not v or not v.strip():
            raise ValueError("message must be non-empty")
        return v


class AlertInstance(BaseModel):
    """Record of an alert firing or resolution event.

    Captures the state of an alert at a point in time including the
    metric value that triggered it and severity classification.

    Attributes:
        instance_id: Unique identifier for this alert instance.
        rule_name: Name of the alert rule that generated this instance.
        status: Current alert status (firing, resolved, pending).
        severity: Alert severity level.
        metric_name: Metric that triggered the alert.
        metric_value: Metric value at the time of alert evaluation.
        threshold: Threshold value that was crossed.
        labels: Alert labels for routing and grouping.
        annotations: Human-readable alert annotations.
        started_at: Timestamp when the alert started firing.
        resolved_at: Timestamp when the alert was resolved.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    instance_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this alert instance",
    )
    rule_name: str = Field(
        ..., description="Name of the alert rule that generated this instance",
    )
    status: AlertStatus = Field(
        default=AlertStatus.FIRING, description="Current alert status",
    )
    severity: AlertSeverity = Field(
        default=AlertSeverity.WARNING, description="Alert severity level",
    )
    metric_name: str = Field(
        default="", description="Metric that triggered the alert",
    )
    metric_value: float = Field(
        default=0.0, description="Metric value at the time of evaluation",
    )
    threshold: float = Field(
        default=0.0, description="Threshold value that was crossed",
    )
    labels: Dict[str, str] = Field(
        default_factory=dict, description="Alert labels for routing and grouping",
    )
    annotations: Dict[str, str] = Field(
        default_factory=dict, description="Human-readable alert annotations",
    )
    started_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the alert started firing",
    )
    resolved_at: Optional[datetime] = Field(
        None, description="Timestamp when the alert was resolved",
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("rule_name")
    @classmethod
    def validate_rule_name(cls, v: str) -> str:
        """Validate rule_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("rule_name must be non-empty")
        return v


class HealthProbeResult(BaseModel):
    """Result of a health probe check.

    Captures the outcome of a liveness, readiness, or startup probe
    for a specific service component.

    Attributes:
        probe_id: Unique identifier for this probe result.
        probe_type: Type of health probe executed.
        service_name: Name of the service being checked.
        status: Health check status outcome.
        message: Human-readable status message.
        details: Additional structured details about the check.
        duration_ms: Probe execution duration in milliseconds.
        checked_at: Timestamp when the probe was executed.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    probe_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this probe result",
    )
    probe_type: ProbeType = Field(
        default=ProbeType.LIVENESS, description="Type of health probe executed",
    )
    service_name: str = Field(
        ..., description="Name of the service being checked",
    )
    status: HealthStatus = Field(
        default=HealthStatus.HEALTHY, description="Health check status outcome",
    )
    message: Optional[str] = Field(
        None, description="Human-readable status message",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional structured details",
    )
    duration_ms: float = Field(
        default=0.0, description="Probe execution duration in milliseconds",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the probe was executed",
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("service_name")
    @classmethod
    def validate_service_name(cls, v: str) -> str:
        """Validate service_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("service_name must be non-empty")
        return v


class DashboardConfig(BaseModel):
    """Configuration for a Grafana dashboard.

    Defines the layout, panels, time range, and refresh interval
    for a dashboard provisioned by the observability agent.

    Attributes:
        dashboard_id: Unique identifier for this dashboard.
        name: Human-readable dashboard name.
        description: Dashboard description text.
        panels: List of panel configuration dictionaries.
        time_range: Default time range for the dashboard (e.g. "1h", "24h").
        refresh_interval: Auto-refresh interval (e.g. "30s", "1m").
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    dashboard_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this dashboard",
    )
    name: str = Field(
        ..., description="Human-readable dashboard name",
    )
    description: str = Field(
        default="", description="Dashboard description text",
    )
    panels: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of panel configuration dictionaries",
    )
    time_range: str = Field(
        default="1h", description="Default time range for the dashboard",
    )
    refresh_interval: str = Field(
        default="30s", description="Auto-refresh interval",
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class SLODefinition(BaseModel):
    """Definition of a Service Level Objective.

    Defines the target, evaluation window, and burn rate thresholds
    for an SLO following Google SRE Book practices.

    Attributes:
        slo_id: Unique identifier for this SLO.
        name: Human-readable SLO name.
        description: SLO description text.
        service_name: Service this SLO applies to.
        slo_type: Type of SLO (availability, latency, etc.).
        target: Target ratio (e.g. 0.999 for 99.9%).
        window_days: Evaluation window in days (typically 30).
        burn_rate_thresholds: Burn rate thresholds by window.
        created_at: Timestamp when the SLO was created.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    slo_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this SLO",
    )
    name: str = Field(
        ..., description="Human-readable SLO name",
    )
    description: str = Field(
        default="", description="SLO description text",
    )
    service_name: str = Field(
        default="", description="Service this SLO applies to",
    )
    slo_type: SLOType = Field(
        default=SLOType.AVAILABILITY, description="Type of SLO",
    )
    target: float = Field(
        default=0.999, ge=0.0, le=1.0, description="Target ratio (e.g. 0.999)",
    )
    window_days: int = Field(
        default=30, ge=1, description="Evaluation window in days",
    )
    burn_rate_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "fast_burn": 14.4,
            "medium_burn": 6.0,
            "slow_burn": 1.0,
        },
        description="Burn rate thresholds by window",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Timestamp when the SLO was created",
    )
    tenant_id: str = Field(
        default="default", description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class SLOStatus(BaseModel):
    """Current status of an SLO including error budget calculations.

    Provides a point-in-time view of SLO compliance with burn rate
    analysis across multiple time windows.

    Attributes:
        slo_id: Reference to the SLO definition.
        name: SLO name for display.
        current_value: Current measured value of the SLI.
        target: SLO target ratio.
        compliance_ratio: Current compliance ratio (current / target).
        error_budget_total: Total error budget for the window.
        error_budget_consumed: Amount of error budget consumed.
        error_budget_remaining: Remaining error budget.
        burn_rate_1h: Error budget burn rate over the last 1 hour.
        burn_rate_6h: Error budget burn rate over the last 6 hours.
        burn_rate_24h: Error budget burn rate over the last 24 hours.
        is_burning: Whether the SLO is currently burning error budget.
        window_start: Start of the evaluation window.
        window_end: End of the evaluation window.
    """

    slo_id: str = Field(
        ..., description="Reference to the SLO definition",
    )
    name: str = Field(
        default="", description="SLO name for display",
    )
    current_value: float = Field(
        default=1.0, description="Current measured value of the SLI",
    )
    target: float = Field(
        default=0.999, description="SLO target ratio",
    )
    compliance_ratio: float = Field(
        default=1.0, description="Current compliance ratio",
    )
    error_budget_total: float = Field(
        default=0.0, description="Total error budget for the window",
    )
    error_budget_consumed: float = Field(
        default=0.0, description="Amount of error budget consumed",
    )
    error_budget_remaining: float = Field(
        default=0.0, description="Remaining error budget",
    )
    burn_rate_1h: float = Field(
        default=0.0, description="Error budget burn rate over the last 1 hour",
    )
    burn_rate_6h: float = Field(
        default=0.0, description="Error budget burn rate over the last 6 hours",
    )
    burn_rate_24h: float = Field(
        default=0.0, description="Error budget burn rate over the last 24 hours",
    )
    is_burning: bool = Field(
        default=False, description="Whether the SLO is currently burning error budget",
    )
    window_start: Optional[datetime] = Field(
        None, description="Start of the evaluation window",
    )
    window_end: Optional[datetime] = Field(
        None, description="End of the evaluation window",
    )

    model_config = {"extra": "forbid"}

    @field_validator("slo_id")
    @classmethod
    def validate_slo_id(cls, v: str) -> str:
        """Validate slo_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("slo_id must be non-empty")
        return v


class ObservabilityStatistics(BaseModel):
    """Aggregated statistics for the observability agent.

    Attributes:
        total_metrics: Total number of metric recordings processed.
        total_spans: Total number of trace spans created.
        total_logs: Total number of log entries ingested.
        total_alerts_fired: Total number of alerts fired.
        active_alerts: Number of currently active (firing) alerts.
        active_spans: Number of currently active trace spans.
        slo_count: Number of registered SLO definitions.
        health_checks_total: Total number of health checks executed.
        uptime_seconds: Service uptime in seconds.
    """

    total_metrics: int = Field(
        default=0, description="Total number of metric recordings processed",
    )
    total_spans: int = Field(
        default=0, description="Total number of trace spans created",
    )
    total_logs: int = Field(
        default=0, description="Total number of log entries ingested",
    )
    total_alerts_fired: int = Field(
        default=0, description="Total number of alerts fired",
    )
    active_alerts: int = Field(
        default=0, description="Number of currently active alerts",
    )
    active_spans: int = Field(
        default=0, description="Number of currently active trace spans",
    )
    slo_count: int = Field(
        default=0, description="Number of registered SLO definitions",
    )
    health_checks_total: int = Field(
        default=0, description="Total number of health checks executed",
    )
    uptime_seconds: float = Field(
        default=0.0, description="Service uptime in seconds",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request / Response Models
# =============================================================================


class RecordMetricRequest(BaseModel):
    """Request body for recording a metric observation.

    Attributes:
        metric_name: Name of the metric to record.
        value: Numeric value to record.
        labels: Optional label key-value pairs.
        tenant_id: Optional tenant identifier.
    """

    metric_name: str = Field(
        ..., description="Name of the metric to record",
    )
    value: float = Field(
        ..., description="Numeric value to record",
    )
    labels: Dict[str, str] = Field(
        default_factory=dict, description="Optional label key-value pairs",
    )
    tenant_id: str = Field(
        default="default", description="Optional tenant identifier",
    )

    model_config = {"extra": "forbid"}

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v: str) -> str:
        """Validate metric_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("metric_name must be non-empty")
        return v


class CreateSpanRequest(BaseModel):
    """Request body for creating a new trace span.

    Attributes:
        operation_name: Name of the operation.
        service_name: Service that owns this span.
        parent_span_id: Optional parent span for nesting.
        attributes: Optional span attributes.
        tenant_id: Optional tenant identifier.
    """

    operation_name: str = Field(
        ..., description="Name of the operation",
    )
    service_name: str = Field(
        default="", description="Service that owns this span",
    )
    parent_span_id: Optional[str] = Field(
        None, description="Optional parent span for nesting",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Optional span attributes",
    )
    tenant_id: str = Field(
        default="default", description="Optional tenant identifier",
    )

    model_config = {"extra": "forbid"}

    @field_validator("operation_name")
    @classmethod
    def validate_operation_name(cls, v: str) -> str:
        """Validate operation_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("operation_name must be non-empty")
        return v


class IngestLogRequest(BaseModel):
    """Request body for ingesting a structured log entry.

    Attributes:
        level: Log severity level.
        message: Log message text.
        agent_id: Optional ID of the originating agent.
        trace_id: Optional trace ID for correlation.
        span_id: Optional span ID for correlation.
        attributes: Optional additional attributes.
        tenant_id: Optional tenant identifier.
    """

    level: LogLevel = Field(
        default=LogLevel.INFO, description="Log severity level",
    )
    message: str = Field(
        ..., description="Log message text",
    )
    agent_id: Optional[str] = Field(
        None, description="Optional ID of the originating agent",
    )
    trace_id: Optional[str] = Field(
        None, description="Optional trace ID for correlation",
    )
    span_id: Optional[str] = Field(
        None, description="Optional span ID for correlation",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Optional additional attributes",
    )
    tenant_id: str = Field(
        default="default", description="Optional tenant identifier",
    )

    model_config = {"extra": "forbid"}

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message is non-empty."""
        if not v or not v.strip():
            raise ValueError("message must be non-empty")
        return v


class CreateAlertRuleRequest(BaseModel):
    """Request body for creating an alert rule.

    Attributes:
        name: Alert rule name.
        metric_name: Metric to monitor.
        condition: Condition operator (gt, lt, eq, gte, lte, ne).
        threshold: Threshold value.
        severity: Alert severity.
        duration_seconds: Duration before firing.
        labels: Optional filter labels.
        annotations: Optional alert annotations.
        tenant_id: Optional tenant identifier.
    """

    name: str = Field(
        ..., description="Alert rule name",
    )
    metric_name: str = Field(
        ..., description="Metric to monitor",
    )
    condition: str = Field(
        ..., description="Condition operator (gt, lt, eq, gte, lte, ne)",
    )
    threshold: float = Field(
        ..., description="Threshold value",
    )
    severity: AlertSeverity = Field(
        default=AlertSeverity.WARNING, description="Alert severity",
    )
    duration_seconds: int = Field(
        default=60, ge=0, description="Duration before firing",
    )
    labels: Dict[str, str] = Field(
        default_factory=dict, description="Optional filter labels",
    )
    annotations: Dict[str, str] = Field(
        default_factory=dict, description="Optional alert annotations",
    )
    tenant_id: str = Field(
        default="default", description="Optional tenant identifier",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v: str) -> str:
        """Validate condition is a supported operator."""
        allowed = {"gt", "lt", "eq", "gte", "lte", "ne"}
        if v not in allowed:
            raise ValueError(f"condition must be one of {allowed}")
        return v


class CreateSLORequest(BaseModel):
    """Request body for creating a Service Level Objective.

    Attributes:
        name: SLO name.
        description: SLO description.
        service_name: Service this SLO applies to.
        slo_type: Type of SLO.
        target: Target ratio (0.0 to 1.0).
        window_days: Evaluation window in days.
        tenant_id: Optional tenant identifier.
    """

    name: str = Field(
        ..., description="SLO name",
    )
    description: str = Field(
        default="", description="SLO description",
    )
    service_name: str = Field(
        default="", description="Service this SLO applies to",
    )
    slo_type: SLOType = Field(
        default=SLOType.AVAILABILITY, description="Type of SLO",
    )
    target: float = Field(
        default=0.999, ge=0.0, le=1.0, description="Target ratio",
    )
    window_days: int = Field(
        default=30, ge=1, description="Evaluation window in days",
    )
    tenant_id: str = Field(
        default="default", description="Optional tenant identifier",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class HealthCheckRequest(BaseModel):
    """Request body for executing a health check probe.

    Attributes:
        probe_type: Type of health probe to execute.
        service_name: Service to check.
        tenant_id: Optional tenant identifier.
    """

    probe_type: ProbeType = Field(
        default=ProbeType.LIVENESS, description="Type of health probe to execute",
    )
    service_name: str = Field(
        default="", description="Service to check",
    )
    tenant_id: str = Field(
        default="default", description="Optional tenant identifier",
    )

    model_config = {"extra": "forbid"}


__all__ = [
    # Re-exported enums
    "MetricType",
    "AlertSeverity",
    "AlertStatus",
    "HealthStatus",
    "TraceStatus",
    # Re-exported Layer 1 models
    "MetricLabel",
    "MetricDefinition",
    "MetricValue",
    "TraceContext",
    "SpanDefinition",
    "LogEntry",
    "AlertRule",
    "Alert",
    "HealthCheck",
    "DashboardPanel",
    "ObservabilityInput",
    "ObservabilityOutput",
    # New enums
    "LogLevel",
    "SLOType",
    "ProbeType",
    # SDK models
    "MetricRecording",
    "TraceRecord",
    "LogRecord",
    "AlertInstance",
    "HealthProbeResult",
    "DashboardConfig",
    "SLODefinition",
    "SLOStatus",
    "ObservabilityStatistics",
    # Request / Response
    "RecordMetricRequest",
    "CreateSpanRequest",
    "IngestLogRequest",
    "CreateAlertRuleRequest",
    "CreateSLORequest",
    "HealthCheckRequest",
]
