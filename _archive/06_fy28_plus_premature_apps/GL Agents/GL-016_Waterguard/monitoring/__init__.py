"""
GL-016 Waterguard Monitoring Package

This package provides comprehensive monitoring capabilities for the
Waterguard boiler water chemistry optimization agent including:
    - Prometheus metrics collection
    - Alert generation and routing
    - Health check endpoints

Modules:
    - metrics: Prometheus-style metrics for chemistry, compliance, and savings
    - alerting: Alert generation and management
    - health_checks: Liveness and readiness probes

Example:
    >>> from monitoring import WaterguardMetrics, AlertManager, HealthChecker
    >>> metrics = WaterguardMetrics()
    >>> metrics.record_chemistry_reading("boiler-001", "conductivity", 1250.5, "uS/cm")
    >>> alert_manager = AlertManager()
    >>> health_checker = HealthChecker()

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from .metrics import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    WaterguardMetrics,
    MetricsHTTPHandler,
    Timer,
)

from .alerting import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertState,
    AlertRule,
    AlertRoute,
    AlertSilence,
)

from .health_checks import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    HealthCheckResult,
    DatabaseHealthCheck,
    KafkaHealthCheck,
    OPCUAHealthCheck,
    AnalyzerHealthCheck,
)

__all__ = [
    # Metrics
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "WaterguardMetrics",
    "MetricsHTTPHandler",
    "Timer",
    # Alerting
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertState",
    "AlertRule",
    "AlertRoute",
    "AlertSilence",
    # Health Checks
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "HealthCheckResult",
    "DatabaseHealthCheck",
    "KafkaHealthCheck",
    "OPCUAHealthCheck",
    "AnalyzerHealthCheck",
]

__version__ = "1.0.0"
