"""
GL-015 INSULSCAN - Monitoring Module

This module provides comprehensive monitoring capabilities for the INSULSCAN
insulation scanning and thermal assessment agent, including Prometheus metrics,
health checks, alerting, and Grafana dashboard definitions.

Components:
    - metrics: Prometheus-compatible metrics for heat loss, condition scores,
               hot spot detection, and repair recommendations
    - health: Health monitoring with Kubernetes liveness/readiness probes
    - alerting: Alert management with deduplication, suppression, and
                multi-channel notifications (PagerDuty, Slack, Email)
    - dashboards: Grafana dashboard definitions and data providers

Key Features:
    - Heat loss tracking by asset, surface type, and insulation type
    - Condition score monitoring with threshold alerts
    - Hot spot detection with severity classification
    - Repair recommendation tracking with ROI metrics
    - Circuit breaker state monitoring
    - Kubernetes-compatible health probes
    - Multi-channel alert notifications

Prometheus Metrics:
    - insulscan_analyses_total: Total analyses performed
    - insulscan_analysis_duration_seconds: Analysis latency histogram
    - insulscan_heat_loss_watts: Current heat loss by asset
    - insulscan_condition_score: Condition score by asset
    - insulscan_hot_spots_detected: Hot spots found by severity
    - insulscan_repair_recommendations_total: Repair recommendations
    - insulscan_energy_savings_usd: Projected annual savings
    - insulscan_circuit_breaker_state: Circuit breaker status

Example:
    >>> from gl_015_insulscan.monitoring import (
    ...     InsulscanMetricsCollector,
    ...     InsulscanAlertManager,
    ...     InsulscanHealthMonitor,
    ...     InsulscanDashboardProvider,
    ... )
    >>> metrics = InsulscanMetricsCollector()
    >>> alert_manager = InsulscanAlertManager()
    >>> health_monitor = InsulscanHealthMonitor()
    >>> dashboard_provider = InsulscanDashboardProvider(metrics, alert_manager)
"""

from .metrics import (
    # Enumerations
    MetricType,
    SurfaceType,
    InsulationType,
    HotSpotSeverity,
    CircuitBreakerState,
    # Data Classes
    MetricValue,
    HistogramBucket,
    AnalysisMetrics,
    MetricsSummary,
    # Metric Definitions
    METRICS_DEFINITIONS,
    # Main Class
    InsulscanMetricsCollector,
    # Global Instance
    get_metrics_collector,
)

from .health import (
    # Enumerations
    HealthStatus,
    ComponentType,
    # Data Classes
    ComponentHealth,
    CalculatorHealthStatus,
    IntegrationHealthStatus,
    DatabaseHealthStatus,
    DataQualityStatus,
    OverallHealthStatus,
    LivenessProbeResult,
    ReadinessProbeResult,
    # Main Class
    InsulscanHealthMonitor,
    # Global Instance
    get_health_monitor,
)

from .alerting import (
    # Enumerations
    InsulationAlertType,
    AlertSeverity,
    AlertState,
    EscalationLevel,
    # Configuration Classes
    AlertRule,
    DEFAULT_ALERT_RULES,
    # Data Classes
    AlertContext,
    Alert,
    AlertFilter,
    AcknowledgmentResult,
    EscalationResult,
    # Notification Channels
    NotificationChannel,
    LogNotificationChannel,
    PagerDutyNotificationChannel,
    SlackNotificationChannel,
    EmailNotificationChannel,
    # Main Class
    InsulscanAlertManager,
    # Global Instance
    get_alert_manager,
)

from .dashboards import (
    # Data Classes
    TimeSeriesData,
    DashboardWidget,
    AssetStatus,
    HeatLossOverviewDashboard,
    AssetConditionHeatmapDashboard,
    TrendAnalysisDashboard,
    ROITrackingDashboard,
    AlertOverviewDashboard,
    # Grafana Builder
    GrafanaDashboardBuilder,
    # Main Class
    InsulscanDashboardProvider,
    # Global Instance
    get_dashboard_provider,
)

__all__ = [
    # Metrics - Enumerations
    "MetricType",
    "SurfaceType",
    "InsulationType",
    "HotSpotSeverity",
    "CircuitBreakerState",
    # Metrics - Data Classes
    "MetricValue",
    "HistogramBucket",
    "AnalysisMetrics",
    "MetricsSummary",
    # Metrics - Definitions
    "METRICS_DEFINITIONS",
    # Metrics - Main Class
    "InsulscanMetricsCollector",
    "get_metrics_collector",
    # Health - Enumerations
    "HealthStatus",
    "ComponentType",
    # Health - Data Classes
    "ComponentHealth",
    "CalculatorHealthStatus",
    "IntegrationHealthStatus",
    "DatabaseHealthStatus",
    "DataQualityStatus",
    "OverallHealthStatus",
    "LivenessProbeResult",
    "ReadinessProbeResult",
    # Health - Main Class
    "InsulscanHealthMonitor",
    "get_health_monitor",
    # Alerting - Enumerations
    "InsulationAlertType",
    "AlertSeverity",
    "AlertState",
    "EscalationLevel",
    # Alerting - Configuration
    "AlertRule",
    "DEFAULT_ALERT_RULES",
    # Alerting - Data Classes
    "AlertContext",
    "Alert",
    "AlertFilter",
    "AcknowledgmentResult",
    "EscalationResult",
    # Alerting - Notification Channels
    "NotificationChannel",
    "LogNotificationChannel",
    "PagerDutyNotificationChannel",
    "SlackNotificationChannel",
    "EmailNotificationChannel",
    # Alerting - Main Class
    "InsulscanAlertManager",
    "get_alert_manager",
    # Dashboards - Data Classes
    "TimeSeriesData",
    "DashboardWidget",
    "AssetStatus",
    "HeatLossOverviewDashboard",
    "AssetConditionHeatmapDashboard",
    "TrendAnalysisDashboard",
    "ROITrackingDashboard",
    "AlertOverviewDashboard",
    # Dashboards - Grafana
    "GrafanaDashboardBuilder",
    # Dashboards - Main Class
    "InsulscanDashboardProvider",
    "get_dashboard_provider",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-015"
__codename__ = "INSULSCAN"
