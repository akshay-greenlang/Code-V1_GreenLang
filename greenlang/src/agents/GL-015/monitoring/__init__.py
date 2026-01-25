# -*- coding: utf-8 -*-
"""
Monitoring Package for GL-015 INSULSCAN.

This package provides comprehensive monitoring capabilities for the
insulation scanning and thermal imaging agent:

- Prometheus metrics collection and export
- Health check endpoints (liveness, readiness, detailed)
- Alert rules and notification management
- Grafana dashboard configurations

Modules:
    metrics: Prometheus metrics definitions and collector
    health_checks: Health check implementations
    alerts: Alert rules and notification management
    grafana: Dashboard configurations

Example:
    >>> from monitoring import get_metrics_collector, get_health_checker
    >>> collector = get_metrics_collector()
    >>> collector.record_inspection_completed("FACILITY-001", "thermal_scan")
    >>> checker = get_health_checker()
    >>> if checker.is_ready():
    ...     print("Service ready")

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from monitoring.metrics import (
    # Main Classes
    MetricsCollector,
    MetricDefinition,
    MetricValue,
    HistogramValue,
    # Enumerations
    MetricType,
    # Request Metrics
    REQUEST_LATENCY_HISTOGRAM,
    REQUEST_COUNT_TOTAL,
    ACTIVE_REQUESTS_GAUGE,
    # Inspection Metrics
    INSPECTIONS_COMPLETED_TOTAL,
    THERMAL_IMAGES_PROCESSED_TOTAL,
    HOTSPOTS_DETECTED_TOTAL,
    ANOMALIES_CLASSIFIED_TOTAL,
    # Heat Loss Metrics
    TOTAL_HEAT_LOSS_WATTS_GAUGE,
    ENERGY_COST_DOLLARS_GAUGE,
    CARBON_EMISSIONS_KG_GAUGE,
    # Degradation Metrics
    EQUIPMENT_BY_CONDITION_GAUGE,
    AVERAGE_DEGRADATION_RATE_GAUGE,
    REPAIRS_PRIORITIZED_TOTAL,
    # Integration Metrics
    CAMERA_CONNECTION_STATUS_GAUGE,
    CMMS_WORK_ORDERS_CREATED_TOTAL,
    # Bucket Configurations
    REQUEST_LATENCY_BUCKETS,
    IMAGE_PROCESSING_DURATION_BUCKETS,
    HEAT_LOSS_BUCKETS,
    # Decorators
    timed_operation,
    count_calls,
    # Utility Functions
    get_metrics_collector,
    create_metrics_endpoint,
    reset_metrics_collector,
)

from monitoring.health_checks import (
    # Main Classes
    HealthChecker,
    BaseHealthCheck,
    # Health Check Implementations
    DatabaseHealthCheck,
    CacheHealthCheck,
    CameraHealthCheck,
    CMMSHealthCheck,
    CalculatorHealthCheck,
    SystemResourceHealthCheck,
    # Status Enumerations
    HealthStatus,
    ComponentStatus,
    # Result Classes
    ComponentHealth,
    DetailedHealthReport,
    # Utility Functions
    get_health_checker,
    create_health_endpoint,
    reset_health_checker,
)

from monitoring.alerts import (
    # Main Classes
    AlertManager,
    # Alert Enumerations
    AlertSeverity,
    AlertState,
    NotificationChannel,
    ComparisonOperator,
    # Alert Data Classes
    AlertCondition,
    AlertRule,
    AlertInstance,
    SilenceRule,
    InhibitRule,
    # Built-in Alert Rules
    HIGH_HEAT_LOSS_ALERT,
    CRITICAL_DEGRADATION_ALERT,
    SAFETY_TEMPERATURE_EXCEEDED_ALERT,
    INSPECTION_OVERDUE_ALERT,
    MOISTURE_DETECTED_ALERT,
    RAPID_DEGRADATION_RATE_ALERT,
    INTEGRATION_FAILURE_ALERT,
    # Notification Classes
    NotificationConfig,
    BaseNotifier,
    EmailNotifier,
    SlackNotifier,
    PagerDutyNotifier,
    # Utility Functions
    create_alert_manager,
    get_default_alert_rules,
    evaluate_threshold,
    reset_alert_manager,
)

from monitoring.grafana import (
    # Dashboard Builder
    GrafanaDashboardBuilder,
    GrafanaPanel,
    # Panel Types
    PanelType,
    DataSourceType,
    # Pre-built Dashboards
    INSULATION_HEALTH_OVERVIEW_DASHBOARD,
    HEAT_LOSS_MONITORING_DASHBOARD,
    THERMAL_IMAGING_RESULTS_DASHBOARD,
    REPAIR_TRACKING_DASHBOARD,
    ENERGY_IMPACT_DASHBOARD,
    FACILITY_COMPARISON_DASHBOARD,
    # Utility Functions
    get_all_dashboards,
    export_dashboard_json,
    import_dashboard_json,
)


__version__ = "1.0.0"
__author__ = "GreenLang AI Agent Factory"
__agent_id__ = "GL-015"
__codename__ = "INSULSCAN"


__all__ = [
    # Metrics
    "MetricsCollector",
    "MetricDefinition",
    "MetricValue",
    "HistogramValue",
    "MetricType",
    "REQUEST_LATENCY_HISTOGRAM",
    "REQUEST_COUNT_TOTAL",
    "ACTIVE_REQUESTS_GAUGE",
    "INSPECTIONS_COMPLETED_TOTAL",
    "THERMAL_IMAGES_PROCESSED_TOTAL",
    "HOTSPOTS_DETECTED_TOTAL",
    "ANOMALIES_CLASSIFIED_TOTAL",
    "TOTAL_HEAT_LOSS_WATTS_GAUGE",
    "ENERGY_COST_DOLLARS_GAUGE",
    "CARBON_EMISSIONS_KG_GAUGE",
    "EQUIPMENT_BY_CONDITION_GAUGE",
    "AVERAGE_DEGRADATION_RATE_GAUGE",
    "REPAIRS_PRIORITIZED_TOTAL",
    "CAMERA_CONNECTION_STATUS_GAUGE",
    "CMMS_WORK_ORDERS_CREATED_TOTAL",
    "REQUEST_LATENCY_BUCKETS",
    "IMAGE_PROCESSING_DURATION_BUCKETS",
    "HEAT_LOSS_BUCKETS",
    "timed_operation",
    "count_calls",
    "get_metrics_collector",
    "create_metrics_endpoint",
    "reset_metrics_collector",
    # Health Checks
    "HealthChecker",
    "BaseHealthCheck",
    "DatabaseHealthCheck",
    "CacheHealthCheck",
    "CameraHealthCheck",
    "CMMSHealthCheck",
    "CalculatorHealthCheck",
    "SystemResourceHealthCheck",
    "HealthStatus",
    "ComponentStatus",
    "ComponentHealth",
    "DetailedHealthReport",
    "get_health_checker",
    "create_health_endpoint",
    "reset_health_checker",
    # Alerts
    "AlertManager",
    "AlertSeverity",
    "AlertState",
    "NotificationChannel",
    "ComparisonOperator",
    "AlertCondition",
    "AlertRule",
    "AlertInstance",
    "SilenceRule",
    "InhibitRule",
    "HIGH_HEAT_LOSS_ALERT",
    "CRITICAL_DEGRADATION_ALERT",
    "SAFETY_TEMPERATURE_EXCEEDED_ALERT",
    "INSPECTION_OVERDUE_ALERT",
    "MOISTURE_DETECTED_ALERT",
    "RAPID_DEGRADATION_RATE_ALERT",
    "INTEGRATION_FAILURE_ALERT",
    "NotificationConfig",
    "BaseNotifier",
    "EmailNotifier",
    "SlackNotifier",
    "PagerDutyNotifier",
    "create_alert_manager",
    "get_default_alert_rules",
    "evaluate_threshold",
    "reset_alert_manager",
    # Grafana
    "GrafanaDashboardBuilder",
    "GrafanaPanel",
    "PanelType",
    "DataSourceType",
    "INSULATION_HEALTH_OVERVIEW_DASHBOARD",
    "HEAT_LOSS_MONITORING_DASHBOARD",
    "THERMAL_IMAGING_RESULTS_DASHBOARD",
    "REPAIR_TRACKING_DASHBOARD",
    "ENERGY_IMPACT_DASHBOARD",
    "FACILITY_COMPARISON_DASHBOARD",
    "get_all_dashboards",
    "export_dashboard_json",
    "import_dashboard_json",
]
