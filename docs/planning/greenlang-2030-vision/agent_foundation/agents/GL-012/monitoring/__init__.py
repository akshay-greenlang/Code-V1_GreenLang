# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Steam Quality Controller Monitoring Package
=============================================================

Comprehensive monitoring, observability, and alerting for the GL-012
STEAMQUAL SteamQualityController agent. This package provides production-ready
components for metrics collection, health checking, alerting, and
distributed tracing.

Package Components:
-------------------

1. **Metrics** (metrics.py)
   - MetricsCollector: Prometheus-style metrics collection
   - Steam quality gauges (dryness, pressure, temperature)
   - Control system metrics (valve, desuperheater)
   - Calculation performance histograms
   - Cache and error counters

2. **Health Checks** (health_checks.py)
   - HealthChecker: Component health monitoring
   - Steam meter connectivity checks
   - Control valve responsiveness validation
   - Kubernetes liveness/readiness probes

3. **Alerting** (alerting.py)
   - AlertManager: Alert lifecycle management
   - Steam quality alerts (low dryness, high moisture)
   - Equipment fault alerts (valve stuck, desuperheater fault)
   - Configurable thresholds and severity levels

4. **Tracing** (tracing.py)
   - TracingConfig: OpenTelemetry-compatible tracing
   - Span creation for quality calculations
   - Control action tracing
   - Integration call tracing

Quick Start:
------------

    >>> from monitoring import (
    ...     MetricsCollector,
    ...     HealthChecker,
    ...     AlertManager,
    ...     TracingConfig,
    ... )
    >>>
    >>> # Initialize all monitoring components
    >>> metrics = MetricsCollector()
    >>> health = HealthChecker()
    >>> alerts = AlertManager()
    >>> tracing = TracingConfig()
    >>> tracing.initialize()
    >>>
    >>> # Record steam quality calculation
    >>> with metrics.track_calculation("dryness"):
    ...     dryness = calculate_steam_dryness()
    >>>
    >>> # Check system health
    >>> status = await health.check_all()
    >>> print(f"System status: {status.status}")
    >>>
    >>> # Raise alert if quality is low
    >>> if dryness < 0.92:
    ...     alerts.raise_alert(
    ...         AlertType.STEAM_QUALITY_LOW,
    ...         Severity.WARNING,
    ...         {"dryness": dryness}
    ...     )

Integration with Agent:
-----------------------

The monitoring package is designed to integrate seamlessly with the
GL-012 STEAMQUAL agent. Each component can be used independently
or combined for full observability:

    >>> class SteamQualityAgent:
    ...     def __init__(self):
    ...         self.metrics = MetricsCollector()
    ...         self.health = HealthChecker()
    ...         self.alerts = AlertManager()
    ...         self.tracing = TracingConfig()
    ...         self.tracing.initialize()
    ...
    ...     async def calculate_quality(self, data):
    ...         with self.tracing.create_span("calculate_quality"):
    ...             with self.metrics.track_calculation("quality"):
    ...                 result = self._perform_calculation(data)
    ...
    ...         if result.dryness < 0.92:
    ...             self.alerts.raise_alert(
    ...                 AlertType.STEAM_QUALITY_LOW,
    ...                 Severity.WARNING
    ...             )
    ...
    ...         return result

Exports:
--------

Metrics:
    - PROMETHEUS_AVAILABLE
    - MetricType
    - MetricValue
    - OperationalState
    - MetricsBuffer
    - MetricsCollector
    - get_metrics_collector
    - init_metrics_collector

Health Checks:
    - HealthStatus
    - ComponentHealth
    - ReadinessStatus
    - LivenessStatus
    - HealthCheckResult
    - HealthChecker
    - get_health_checker
    - init_health_checker

Alerting:
    - Severity
    - AlertType
    - AlertState
    - Alert
    - AlertThreshold
    - AlertManager
    - get_alert_manager
    - init_alert_manager

Tracing:
    - OTEL_AVAILABLE
    - SpanType
    - ExporterType
    - SpanData
    - StubSpan
    - StubTracer
    - TracingConfig
    - get_tracer
    - init_tracing
    - create_span
    - record_exception
    - add_event

Author: GreenLang Team
License: Proprietary
Version: 1.0.0
"""

# Metrics exports
from .metrics import (
    PROMETHEUS_AVAILABLE,
    MetricType,
    MetricValue,
    OperationalState,
    MetricsBuffer,
    MetricsCollector,
    get_metrics_collector,
    init_metrics_collector,
)

# Health check exports
from .health_checks import (
    HealthStatus,
    ComponentHealth,
    ReadinessStatus,
    LivenessStatus,
    HealthCheckResult,
    HealthChecker,
    get_health_checker,
    init_health_checker,
)

# Alerting exports
from .alerting import (
    Severity,
    AlertType,
    AlertState,
    Alert,
    AlertThreshold,
    AlertManager,
    get_alert_manager,
    init_alert_manager,
)

# Tracing exports
from .tracing import (
    OTEL_AVAILABLE,
    SpanType,
    ExporterType,
    SpanData,
    StubSpan,
    StubTracer,
    TracingConfig,
    get_tracer,
    init_tracing,
    create_span,
    record_exception,
    add_event,
)


# Package metadata
__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-012"
__codename__ = "STEAMQUAL"


# Convenience function to initialize all monitoring components
def init_monitoring(
    metrics_config: dict = None,
    health_config: dict = None,
    alert_config: dict = None,
    tracing_config: dict = None,
) -> tuple:
    """
    Initialize all monitoring components with given configurations.

    Args:
        metrics_config: Configuration for MetricsCollector
        health_config: Configuration for HealthChecker
        alert_config: Configuration for AlertManager
        tracing_config: Configuration for TracingConfig

    Returns:
        Tuple of (MetricsCollector, HealthChecker, AlertManager, TracingConfig)

    Example:
        >>> metrics, health, alerts, tracing = init_monitoring(
        ...     tracing_config={"endpoint": "http://jaeger:14268"}
        ... )
    """
    metrics = init_metrics_collector(**(metrics_config or {}))
    health = init_health_checker(**(health_config or {}))
    alerts = init_alert_manager(**(alert_config or {}))
    tracing = init_tracing(**(tracing_config or {}))

    return metrics, health, alerts, tracing


__all__ = [
    # Package info
    "__version__",
    "__author__",
    "__agent_id__",
    "__codename__",
    # Convenience functions
    "init_monitoring",
    # Metrics
    "PROMETHEUS_AVAILABLE",
    "MetricType",
    "MetricValue",
    "OperationalState",
    "MetricsBuffer",
    "MetricsCollector",
    "get_metrics_collector",
    "init_metrics_collector",
    # Health checks
    "HealthStatus",
    "ComponentHealth",
    "ReadinessStatus",
    "LivenessStatus",
    "HealthCheckResult",
    "HealthChecker",
    "get_health_checker",
    "init_health_checker",
    # Alerting
    "Severity",
    "AlertType",
    "AlertState",
    "Alert",
    "AlertThreshold",
    "AlertManager",
    "get_alert_manager",
    "init_alert_manager",
    # Tracing
    "OTEL_AVAILABLE",
    "SpanType",
    "ExporterType",
    "SpanData",
    "StubSpan",
    "StubTracer",
    "TracingConfig",
    "get_tracer",
    "init_tracing",
    "create_span",
    "record_exception",
    "add_event",
]
