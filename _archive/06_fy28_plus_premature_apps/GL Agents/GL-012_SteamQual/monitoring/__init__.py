"""
GL-012 STEAMQUAL SteamQualityController - Monitoring Module

This module provides comprehensive monitoring capabilities for the STEAMQUAL
steam quality control agent, including alerting, health checks, metrics
collection, and dashboard data providers.

Components:
    - metrics: Prometheus-compatible metrics collection for steam quality
    - alerting: Alert management with hysteresis and rate limiting
    - health: Health monitoring with liveness/readiness probes
    - dashboards: Dashboard data aggregation for UI consumption

The monitoring module ensures operational visibility and provides real-time
insights into steam quality across all monitored separators.

Key Features:
    - Steam quality metrics (dryness fraction, carryover risk, efficiency)
    - Alarm hysteresis to prevent alarm chatter
    - Rate limiting to prevent alarm flooding
    - Kubernetes-compatible health probes
    - Dashboard data for KPI visualization

Example:
    >>> from gl_012_steamqual.monitoring import (
    ...     SteamQualityMetricsCollector,
    ...     SteamQualityAlertManager,
    ...     SteamQualityHealthMonitor,
    ...     SteamQualityDashboardProvider,
    ... )
    >>> metrics = SteamQualityMetricsCollector(namespace="steamqual")
    >>> alert_manager = SteamQualityAlertManager()
    >>> health_monitor = SteamQualityHealthMonitor()
    >>> dashboard_provider = SteamQualityDashboardProvider(metrics, alert_manager)
"""

from .metrics import (
    # Data Classes
    MetricValue,
    SteamQualityMetrics,
    CalculationMetrics,
    MetricsSummary,
    # Main Class
    SteamQualityMetricsCollector,
)

from .alerting import (
    # Enums
    QualityAlertType,
    AlertSeverity,
    AlertState,
    EscalationLevel,
    # Configuration Classes
    HysteresisConfig,
    ThresholdConfig,
    # Data Classes
    AlertContext,
    Alert,
    AlertFilter,
    AcknowledgmentResult,
    EscalationResult,
    # Notification Channels
    NotificationChannel,
    LogNotificationChannel,
    # Main Class
    SteamQualityAlertManager,
    # Default Thresholds
    DEFAULT_THRESHOLDS,
)

from .health import (
    # Enums
    HealthStatus,
    # Data Classes
    ComponentHealth,
    ServiceHealthStatus,
    IntegrationHealthStatus,
    DataQualityStatus,
    CalculatorHealthStatus,
    OverallHealthStatus,
    # Main Class
    SteamQualityHealthMonitor,
)

from .dashboards import (
    # Data Classes
    TimeSeriesData,
    DashboardWidget,
    SeparatorStatus,
    RealTimeQualityDashboard,
    SeparatorDashboard,
    QualityTrendDashboard,
    AlertSummaryDashboard,
    KPIDashboard,
    # Main Class
    SteamQualityDashboardProvider,
)

__all__ = [
    # Metrics
    "MetricValue",
    "SteamQualityMetrics",
    "CalculationMetrics",
    "MetricsSummary",
    "SteamQualityMetricsCollector",
    # Alerting - Enums
    "QualityAlertType",
    "AlertSeverity",
    "AlertState",
    "EscalationLevel",
    # Alerting - Configuration
    "HysteresisConfig",
    "ThresholdConfig",
    "DEFAULT_THRESHOLDS",
    # Alerting - Data Classes
    "AlertContext",
    "Alert",
    "AlertFilter",
    "AcknowledgmentResult",
    "EscalationResult",
    # Alerting - Channels
    "NotificationChannel",
    "LogNotificationChannel",
    # Alerting - Main Class
    "SteamQualityAlertManager",
    # Health - Enums
    "HealthStatus",
    # Health - Data Classes
    "ComponentHealth",
    "ServiceHealthStatus",
    "IntegrationHealthStatus",
    "DataQualityStatus",
    "CalculatorHealthStatus",
    "OverallHealthStatus",
    # Health - Main Class
    "SteamQualityHealthMonitor",
    # Dashboards - Data Classes
    "TimeSeriesData",
    "DashboardWidget",
    "SeparatorStatus",
    "RealTimeQualityDashboard",
    "SeparatorDashboard",
    "QualityTrendDashboard",
    "AlertSummaryDashboard",
    "KPIDashboard",
    # Dashboards - Main Class
    "SteamQualityDashboardProvider",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-012"
__codename__ = "STEAMQUAL"
