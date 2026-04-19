"""
GL-004 BURNMASTER Monitoring Module

This module provides comprehensive monitoring capabilities for combustion
optimization operations, including alerting, health monitoring, metrics
collection, KPI tracking, performance analysis, drift detection, and
observability configuration.

Example:
    >>> from monitoring import AlertManager, SystemHealthMonitor
    >>> alert_manager = AlertManager()
    >>> health_monitor = SystemHealthMonitor()
"""

from monitoring.alert_manager import (
    AlertLevel,
    AlertRule,
    Alert,
    SendResult,
    AckResult,
    EscalationResult,
    AlertManager,
)

from monitoring.health_monitor import (
    DataQualityReport,
    SensorHealthReport,
    CalibrationStatus,
    ModelHealthReport,
    LoopHealthReport,
    HealthDashboard,
    SystemHealthMonitor,
)

from monitoring.metrics_collector import (
    DateRange,
    MetricHistory,
    MetricsCollector,
)

from monitoring.kpi_tracker import (
    TrendAnalysis,
    KPIReport,
    KPITracker,
)

from monitoring.performance_tracker import (
    ValueMetrics,
    ComparisonResult,
    PerformanceReport,
    OptimizerPerformanceTracker,
)

from monitoring.drift_monitor import (
    DriftStatus,
    AlertResult,
    RecalibrationRecommendation,
    DriftMonitor,
)

from monitoring.observability import (
    LogConfig,
    TraceConfig,
    MetricsConfig,
    Span,
    setup_logging,
    setup_tracing,
    setup_metrics,
    create_span,
    log_structured,
)

__all__ = [
    # Alert Manager
    "AlertLevel",
    "AlertRule",
    "Alert",
    "SendResult",
    "AckResult",
    "EscalationResult",
    "AlertManager",
    # Health Monitor
    "DataQualityReport",
    "SensorHealthReport",
    "CalibrationStatus",
    "ModelHealthReport",
    "LoopHealthReport",
    "HealthDashboard",
    "SystemHealthMonitor",
    # Metrics Collector
    "DateRange",
    "MetricHistory",
    "MetricsCollector",
    # KPI Tracker
    "TrendAnalysis",
    "KPIReport",
    "KPITracker",
    # Performance Tracker
    "ValueMetrics",
    "ComparisonResult",
    "PerformanceReport",
    "OptimizerPerformanceTracker",
    # Drift Monitor
    "DriftStatus",
    "AlertResult",
    "RecalibrationRecommendation",
    "DriftMonitor",
    # Observability
    "LogConfig",
    "TraceConfig",
    "MetricsConfig",
    "Span",
    "setup_logging",
    "setup_tracing",
    "setup_metrics",
    "create_span",
    "log_structured",
]

__version__ = "1.0.0"
__author__ = "GreenLang Engineering"
