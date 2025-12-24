# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Monitoring Module

Production-grade monitoring capabilities including:
- Health check endpoints for Kubernetes probes
- Alert management with multi-channel routing
- Prometheus-compatible metrics collection
- Safety interlocks and operational safety

Example:
    >>> from monitoring import HealthCheckServer, AlertManager, MetricsServer
    >>> # Start health check server
    >>> health_server = HealthCheckServer(port=8010)
    >>> health_server.start()
    >>> # Start metrics server
    >>> metrics_server = MetricsServer(port=9010)
    >>> metrics_server.start()

Author: GreenLang GL-010 EmissionsGuardian
Version: 1.0.0
"""

from .health import (
    # Enums
    HealthStatus,
    ComponentType,
    # Models
    ComponentHealth,
    HealthCheckResult,
    # Checkers
    HealthChecker,
    TCPHealthChecker,
    CallableHealthChecker,
    # Registry and Server
    HealthCheckRegistry,
    HealthCheckServer,
    # Factory functions
    create_dahs_health_checker,
    create_database_health_checker,
    create_cems_health_checker,
    # Convenience functions
    get_health_registry,
    start_health_server,
    stop_health_server,
)

from .alerts import (
    # Enums
    AlertSeverity,
    AlertType,
    AlertState,
    NotificationChannel,
    # Models
    Alert,
    NotificationTarget,
    EscalationPolicy,
    # Handlers
    NotificationHandler,
    EmailNotificationHandler,
    WebhookNotificationHandler,
    PagerDutyNotificationHandler,
    LogNotificationHandler,
    # Throttling and Deduplication
    AlertThrottler,
    AlertDeduplicator,
    # Manager
    AlertManager,
    # Convenience functions
    get_alert_manager,
    send_alert,
    send_compliance_alert,
    send_fugitive_alert,
)

from .metrics import (
    # Enums
    SeverityLevel,
    ConfidenceLevel,
    Pollutant,
    ProcessingStage,
    # Models
    MetricLabel,
    # Metric types
    BaseMetric,
    Counter,
    Gauge,
    Histogram,
    HistogramTimer,
    # Registry and Server
    MetricsRegistry,
    MetricsServer,
    # Decorators
    track_processing_time,
    count_calls,
    # Functions
    calculate_metrics_provenance,
    get_registry,
    start_metrics_server,
    stop_metrics_server,
)

__all__ = [
    # Health
    "HealthStatus",
    "ComponentType",
    "ComponentHealth",
    "HealthCheckResult",
    "HealthChecker",
    "TCPHealthChecker",
    "CallableHealthChecker",
    "HealthCheckRegistry",
    "HealthCheckServer",
    "create_dahs_health_checker",
    "create_database_health_checker",
    "create_cems_health_checker",
    "get_health_registry",
    "start_health_server",
    "stop_health_server",
    # Alerts
    "AlertSeverity",
    "AlertType",
    "AlertState",
    "NotificationChannel",
    "Alert",
    "NotificationTarget",
    "EscalationPolicy",
    "NotificationHandler",
    "EmailNotificationHandler",
    "WebhookNotificationHandler",
    "PagerDutyNotificationHandler",
    "LogNotificationHandler",
    "AlertThrottler",
    "AlertDeduplicator",
    "AlertManager",
    "get_alert_manager",
    "send_alert",
    "send_compliance_alert",
    "send_fugitive_alert",
    # Metrics
    "SeverityLevel",
    "ConfidenceLevel",
    "Pollutant",
    "ProcessingStage",
    "MetricLabel",
    "BaseMetric",
    "Counter",
    "Gauge",
    "Histogram",
    "HistogramTimer",
    "MetricsRegistry",
    "MetricsServer",
    "track_processing_time",
    "count_calls",
    "calculate_metrics_provenance",
    "get_registry",
    "start_metrics_server",
    "stop_metrics_server",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
