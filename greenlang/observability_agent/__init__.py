# -*- coding: utf-8 -*-
"""
GL-FOUND-X-010: GreenLang Observability & Telemetry Agent Service SDK
=====================================================================

This package provides the observability and telemetry infrastructure SDK for
the GreenLang framework. It supports:

- Prometheus-compatible metric collection with counters, gauges, and histograms
- OpenTelemetry distributed tracing with span lifecycle management
- JSON structured logging with correlation IDs and trace linking
- Threshold-based alert evaluation with severity classification
- Kubernetes-style health probing (liveness, readiness, startup)
- Grafana dashboard configuration provisioning
- SLO tracking with error budget burn rate analysis (Google SRE Book)
- SHA-256 provenance chain tracking for complete audit trails
- 12 Prometheus metrics for self-monitoring
- FastAPI REST API with 20 endpoints
- Thread-safe configuration with GL_OBSERVABILITY_AGENT_ env prefix

Key Components:
    - config: ObservabilityAgentConfig with GL_OBSERVABILITY_AGENT_ env prefix
    - models: Pydantic v2 models for all data structures
    - metrics_collector: Unified Prometheus-compatible metric collection engine
    - trace_manager: OTel-compatible distributed tracing engine
    - log_aggregator: Structured log aggregation with correlation tracking
    - alert_evaluator: Threshold-based alert rule evaluation engine
    - health_checker: Kubernetes-style health probe orchestration
    - dashboard_provider: Grafana dashboard configuration provisioning
    - slo_tracker: SLO compliance and error budget tracking
    - provenance: SHA-256 chain-hashed audit trails
    - metrics: 12 Prometheus self-monitoring metrics
    - api: FastAPI HTTP service
    - setup: ObservabilityAgentService facade

Example:
    >>> from greenlang.observability_agent import ObservabilityAgentService
    >>> service = ObservabilityAgentService()
    >>> service.startup()
    >>> stats = service.get_statistics()

Agent ID: GL-FOUND-X-010
Agent Name: Observability & Telemetry Agent
"""

__version__ = "1.0.0"
__agent_id__ = "GL-FOUND-X-010"
__agent_name__ = "Observability & Telemetry Agent"

# SDK availability flag
OBSERVABILITY_AGENT_SDK_AVAILABLE = True

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from greenlang.observability_agent.config import (
    ObservabilityAgentConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models (enums, Layer 1, SDK)
# ---------------------------------------------------------------------------
from greenlang.observability_agent.models import (
    # Enumerations (Layer 1)
    MetricType,
    AlertSeverity,
    AlertStatus,
    HealthStatus,
    TraceStatus,
    # Layer 1 models
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
    # New enumerations (SDK)
    LogLevel,
    SLOType,
    ProbeType,
    # SDK models
    MetricRecording,
    TraceRecord,
    LogRecord,
    AlertInstance,
    HealthProbeResult,
    DashboardConfig,
    SLODefinition,
    SLOStatus,
    ObservabilityStatistics,
    # Request / Response
    RecordMetricRequest,
    CreateSpanRequest,
    IngestLogRequest,
    CreateAlertRuleRequest,
    CreateSLORequest,
    HealthCheckRequest,
)

# ---------------------------------------------------------------------------
# Core engines
# ---------------------------------------------------------------------------
from greenlang.observability_agent.metrics_collector import MetricsCollector
from greenlang.observability_agent.trace_manager import TraceManager
from greenlang.observability_agent.log_aggregator import LogAggregator
from greenlang.observability_agent.alert_evaluator import AlertEvaluator
from greenlang.observability_agent.health_checker import HealthChecker
from greenlang.observability_agent.dashboard_provider import DashboardProvider
from greenlang.observability_agent.slo_tracker import SLOTracker
from greenlang.observability_agent.provenance import ProvenanceTracker

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from greenlang.observability_agent.metrics import (
    PROMETHEUS_AVAILABLE,
    # Metric objects
    obs_metrics_recorded_total,
    obs_operation_duration_seconds,
    obs_spans_created_total,
    obs_spans_active,
    obs_logs_ingested_total,
    obs_alerts_evaluated_total,
    obs_alerts_firing,
    obs_health_checks_total,
    obs_health_status,
    obs_slo_compliance_ratio,
    obs_error_budget_remaining,
    obs_dashboard_queries_total,
    # Helper functions
    record_metric_recorded,
    record_operation_duration,
    record_span_created,
    update_active_spans,
    record_log_ingested,
    record_alert_evaluated,
    update_firing_alerts,
    record_health_check,
    update_health_status,
    update_slo_compliance,
    update_error_budget,
    record_dashboard_query,
)

# ---------------------------------------------------------------------------
# Service setup facade
# ---------------------------------------------------------------------------
from greenlang.observability_agent.setup import (
    ObservabilityAgentService,
    configure_observability_agent,
    get_observability_agent,
    get_router,
)

__all__ = [
    # Version
    "__version__",
    "__agent_id__",
    "__agent_name__",
    # SDK flag
    "OBSERVABILITY_AGENT_SDK_AVAILABLE",
    # Configuration
    "ObservabilityAgentConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Enumerations (Layer 1)
    "MetricType",
    "AlertSeverity",
    "AlertStatus",
    "HealthStatus",
    "TraceStatus",
    # Layer 1 models
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
    # New enumerations (SDK)
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
    # Core engines
    "MetricsCollector",
    "TraceManager",
    "LogAggregator",
    "AlertEvaluator",
    "HealthChecker",
    "DashboardProvider",
    "SLOTracker",
    "ProvenanceTracker",
    # Metric objects
    "PROMETHEUS_AVAILABLE",
    "obs_metrics_recorded_total",
    "obs_operation_duration_seconds",
    "obs_spans_created_total",
    "obs_spans_active",
    "obs_logs_ingested_total",
    "obs_alerts_evaluated_total",
    "obs_alerts_firing",
    "obs_health_checks_total",
    "obs_health_status",
    "obs_slo_compliance_ratio",
    "obs_error_budget_remaining",
    "obs_dashboard_queries_total",
    # Metric helper functions
    "record_metric_recorded",
    "record_operation_duration",
    "record_span_created",
    "update_active_spans",
    "record_log_ingested",
    "record_alert_evaluated",
    "update_firing_alerts",
    "record_health_check",
    "update_health_status",
    "update_slo_compliance",
    "update_error_budget",
    "record_dashboard_query",
    # Service setup facade
    "ObservabilityAgentService",
    "configure_observability_agent",
    "get_observability_agent",
    "get_router",
]
