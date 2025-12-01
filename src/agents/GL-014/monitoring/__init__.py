# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Monitoring Module.

Comprehensive observability layer for heat exchanger optimization including:
- Prometheus metrics collection and export
- Health check endpoints (liveness, readiness, detailed)
- Alert rule definitions and management
- Grafana dashboard configurations

This module provides production-grade monitoring capabilities for
tracking heat exchanger performance, system health, and business metrics.

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from .metrics import (
    # Main Metrics Collector
    MetricsCollector,

    # Metric Types
    MetricType,

    # Request Metrics
    REQUEST_LATENCY_HISTOGRAM,
    REQUEST_COUNT_TOTAL,
    ACTIVE_REQUESTS_GAUGE,
    REQUEST_SIZE_HISTOGRAM,
    RESPONSE_SIZE_HISTOGRAM,

    # Calculator Metrics
    CALCULATION_DURATION_HISTOGRAM,
    CALCULATION_COUNT_TOTAL,
    CACHE_HIT_RATIO_GAUGE,
    FOULING_RESISTANCE_GAUGE,
    HEALTH_INDEX_GAUGE,

    # Integration Metrics
    CONNECTOR_LATENCY_HISTOGRAM,
    CONNECTOR_ERRORS_TOTAL,
    DATA_POINTS_PROCESSED_TOTAL,

    # Business Metrics
    EXCHANGERS_MONITORED_GAUGE,
    CLEANING_SCHEDULES_GENERATED_TOTAL,
    ESTIMATED_SAVINGS_GAUGE,
    FOULING_ALERTS_TOTAL,

    # Utility Functions
    get_metrics_collector,
    create_metrics_endpoint,
)

from .health_checks import (
    # Main Health Checker
    HealthChecker,

    # Status Enumerations
    HealthStatus,
    ComponentStatus,

    # Health Check Components
    DatabaseHealthCheck,
    CacheHealthCheck,
    HistorianHealthCheck,
    CMMSHealthCheck,
    CalculatorHealthCheck,
    SystemResourceHealthCheck,

    # Result Classes
    ComponentHealth,
    DetailedHealthReport,

    # Utility Functions
    create_health_endpoint,
    get_health_checker,
)

from .alerts import (
    # Main Alert Manager
    AlertManager,

    # Alert Enumerations
    AlertSeverity,
    AlertState,
    NotificationChannel,

    # Alert Rule Classes
    AlertRule,
    AlertCondition,
    AlertInstance,

    # Built-in Alert Rules
    HIGH_FOULING_RESISTANCE_ALERT,
    LOW_THERMAL_EFFICIENCY_ALERT,
    HIGH_PRESSURE_DROP_ALERT,
    PERFORMANCE_DEGRADATION_ALERT,
    CLEANING_OVERDUE_ALERT,
    PREDICTION_ACCURACY_LOW_ALERT,
    INTEGRATION_FAILURE_ALERT,
    HIGH_API_LATENCY_ALERT,
    HIGH_ERROR_RATE_ALERT,

    # Notification Classes
    NotificationConfig,
    EmailNotifier,
    SlackNotifier,
    PagerDutyNotifier,

    # Utility Functions
    create_alert_manager,
    get_default_alert_rules,
)

from .grafana import (
    # Dashboard Definitions
    HEAT_EXCHANGER_OVERVIEW_DASHBOARD,
    FOULING_MONITORING_DASHBOARD,
    PERFORMANCE_TRENDS_DASHBOARD,
    CLEANING_SCHEDULE_DASHBOARD,
    ECONOMIC_IMPACT_DASHBOARD,
    FLEET_COMPARISON_DASHBOARD,

    # Dashboard Builder
    GrafanaDashboardBuilder,

    # Panel Types
    PanelType,

    # Utility Functions
    export_dashboard_json,
    get_all_dashboards,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "MetricType",
    "REQUEST_LATENCY_HISTOGRAM",
    "REQUEST_COUNT_TOTAL",
    "ACTIVE_REQUESTS_GAUGE",
    "REQUEST_SIZE_HISTOGRAM",
    "RESPONSE_SIZE_HISTOGRAM",
    "CALCULATION_DURATION_HISTOGRAM",
    "CALCULATION_COUNT_TOTAL",
    "CACHE_HIT_RATIO_GAUGE",
    "FOULING_RESISTANCE_GAUGE",
    "HEALTH_INDEX_GAUGE",
    "CONNECTOR_LATENCY_HISTOGRAM",
    "CONNECTOR_ERRORS_TOTAL",
    "DATA_POINTS_PROCESSED_TOTAL",
    "EXCHANGERS_MONITORED_GAUGE",
    "CLEANING_SCHEDULES_GENERATED_TOTAL",
    "ESTIMATED_SAVINGS_GAUGE",
    "FOULING_ALERTS_TOTAL",
    "get_metrics_collector",
    "create_metrics_endpoint",

    # Health Checks
    "HealthChecker",
    "HealthStatus",
    "ComponentStatus",
    "DatabaseHealthCheck",
    "CacheHealthCheck",
    "HistorianHealthCheck",
    "CMMSHealthCheck",
    "CalculatorHealthCheck",
    "SystemResourceHealthCheck",
    "ComponentHealth",
    "DetailedHealthReport",
    "create_health_endpoint",
    "get_health_checker",

    # Alerts
    "AlertManager",
    "AlertSeverity",
    "AlertState",
    "NotificationChannel",
    "AlertRule",
    "AlertCondition",
    "AlertInstance",
    "HIGH_FOULING_RESISTANCE_ALERT",
    "LOW_THERMAL_EFFICIENCY_ALERT",
    "HIGH_PRESSURE_DROP_ALERT",
    "PERFORMANCE_DEGRADATION_ALERT",
    "CLEANING_OVERDUE_ALERT",
    "PREDICTION_ACCURACY_LOW_ALERT",
    "INTEGRATION_FAILURE_ALERT",
    "HIGH_API_LATENCY_ALERT",
    "HIGH_ERROR_RATE_ALERT",
    "NotificationConfig",
    "EmailNotifier",
    "SlackNotifier",
    "PagerDutyNotifier",
    "create_alert_manager",
    "get_default_alert_rules",

    # Grafana
    "HEAT_EXCHANGER_OVERVIEW_DASHBOARD",
    "FOULING_MONITORING_DASHBOARD",
    "PERFORMANCE_TRENDS_DASHBOARD",
    "CLEANING_SCHEDULE_DASHBOARD",
    "ECONOMIC_IMPACT_DASHBOARD",
    "FLEET_COMPARISON_DASHBOARD",
    "GrafanaDashboardBuilder",
    "PanelType",
    "export_dashboard_json",
    "get_all_dashboards",
]
