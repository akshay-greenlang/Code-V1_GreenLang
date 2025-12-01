"""
GL-013 PREDICTMAINT - Monitoring Package

Comprehensive monitoring and observability for the GL-013 Predictive
Maintenance Agent. This package provides Prometheus metrics collection,
alert rule management, Grafana dashboard support, and health check endpoints.

Subpackages:
    - alerts: Alert rules and Prometheus alerting configuration
    - grafana: Grafana dashboard definitions and specifications

Modules:
    - metrics: Prometheus metrics collection and MetricsCollector class
    - health_checks: Health check endpoints and liveness probes

Example:
    >>> from gl_013.monitoring import MetricsCollector, HealthChecker
    >>>
    >>> # Initialize metrics collector
    >>> collector = MetricsCollector()
    >>> collector.record_equipment_health("PUMP-001", "pump_centrifugal", 85.5)
    >>> collector.record_rul("PUMP-001", "pump_centrifugal", 2400.0)
    >>>
    >>> # Initialize health checker
    >>> checker = HealthChecker(version="1.0.0")
    >>> health = checker.check_health()
    >>> print(f"Status: {health.status.value}")
    >>>
    >>> # Use alerts
    >>> from gl_013.monitoring.alerts import ALERT_RULES, AlertSeverity
    >>> critical_rules = [r for r in ALERT_RULES if r.severity == AlertSeverity.CRITICAL]

Author: GL-MonitoringEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from typing import TYPE_CHECKING

# Version information
__version__ = "1.0.0"
__package_name__ = "gl_013.monitoring"
__author__ = "GL-MonitoringEngineer"

# Package exports
__all__ = [
    # Version
    "__version__",

    # Metrics module
    "MetricsCollector",
    "MetricSnapshot",
    "EquipmentMetrics",
    "OperationType",
    "AnomalySeverity",
    "MaintenanceUrgency",
    "ConnectorStatus",

    # Prometheus metrics
    "equipment_health_index",
    "equipment_rul_hours",
    "equipment_rul_days",
    "equipment_reliability",
    "failure_probability",
    "failure_probability_30d",
    "failure_probability_90d",
    "vibration_velocity_mm_s",
    "vibration_zone",
    "temperature_celsius",
    "thermal_life_consumed_percent",
    "operation_latency_seconds",
    "operations_total",
    "anomalies_detected_total",
    "anomaly_score",
    "maintenance_tasks_scheduled",
    "maintenance_cost_savings_usd",
    "cache_hit_rate",
    "connector_status",

    # Health checks module
    "HealthChecker",
    "HealthCheckHandler",
    "HealthCheckResult",
    "ComponentHealth",
    "ProbeResult",
    "HealthStatus",
    "ComponentType",
    "ProbeType",
    "health_check_registry",
    "health_check",
    "create_database_health_check",
    "create_cache_health_check",
    "create_connector_health_check",
    "create_model_health_check",

    # Alert rules (from subpackage)
    "AlertRule",
    "AlertGroup",
    "AlertSeverity",
    "AlertCategory",
    "ALERT_RULES",
    "ALERT_GROUPS",
    "get_rules_by_severity",
    "get_rules_by_category",
    "export_prometheus_rules",
]

# Lazy imports for better startup performance
if TYPE_CHECKING:
    # Metrics
    from gl_013.monitoring.metrics import (
        MetricsCollector,
        MetricSnapshot,
        EquipmentMetrics,
        OperationType,
        AnomalySeverity,
        MaintenanceUrgency,
        ConnectorStatus,
        equipment_health_index,
        equipment_rul_hours,
        equipment_rul_days,
        equipment_reliability,
        failure_probability,
        failure_probability_30d,
        failure_probability_90d,
        vibration_velocity_mm_s,
        vibration_zone,
        temperature_celsius,
        thermal_life_consumed_percent,
        operation_latency_seconds,
        operations_total,
        anomalies_detected_total,
        anomaly_score,
        maintenance_tasks_scheduled,
        maintenance_cost_savings_usd,
        cache_hit_rate,
        connector_status,
    )

    # Health checks
    from gl_013.monitoring.health_checks import (
        HealthChecker,
        HealthCheckHandler,
        HealthCheckResult,
        ComponentHealth,
        ProbeResult,
        HealthStatus,
        ComponentType,
        ProbeType,
        health_check_registry,
        health_check,
        create_database_health_check,
        create_cache_health_check,
        create_connector_health_check,
        create_model_health_check,
    )

    # Alerts
    from gl_013.monitoring.alerts.alert_rules import (
        AlertRule,
        AlertGroup,
        AlertSeverity,
        AlertCategory,
        ALERT_RULES,
        ALERT_GROUPS,
        get_rules_by_severity,
        get_rules_by_category,
        export_prometheus_rules,
    )


def __getattr__(name: str):
    """
    Lazy import implementation for better startup performance.

    This allows the package to be imported quickly while deferring
    the loading of heavy dependencies until they are actually needed.
    """
    # Metrics module
    metrics_exports = {
        "MetricsCollector",
        "MetricSnapshot",
        "EquipmentMetrics",
        "OperationType",
        "AnomalySeverity",
        "MaintenanceUrgency",
        "ConnectorStatus",
        "equipment_health_index",
        "equipment_rul_hours",
        "equipment_rul_days",
        "equipment_reliability",
        "failure_probability",
        "failure_probability_30d",
        "failure_probability_90d",
        "vibration_velocity_mm_s",
        "vibration_zone",
        "temperature_celsius",
        "thermal_life_consumed_percent",
        "operation_latency_seconds",
        "operations_total",
        "anomalies_detected_total",
        "anomaly_score",
        "maintenance_tasks_scheduled",
        "maintenance_cost_savings_usd",
        "cache_hit_rate",
        "connector_status",
    }

    if name in metrics_exports:
        from gl_013.monitoring import metrics
        return getattr(metrics, name)

    # Health checks module
    health_exports = {
        "HealthChecker",
        "HealthCheckHandler",
        "HealthCheckResult",
        "ComponentHealth",
        "ProbeResult",
        "HealthStatus",
        "ComponentType",
        "ProbeType",
        "health_check_registry",
        "health_check",
        "create_database_health_check",
        "create_cache_health_check",
        "create_connector_health_check",
        "create_model_health_check",
    }

    if name in health_exports:
        from gl_013.monitoring import health_checks
        return getattr(health_checks, name)

    # Alert rules
    alert_exports = {
        "AlertRule",
        "AlertGroup",
        "AlertSeverity",
        "AlertCategory",
        "ALERT_RULES",
        "ALERT_GROUPS",
        "get_rules_by_severity",
        "get_rules_by_category",
        "export_prometheus_rules",
    }

    if name in alert_exports:
        from gl_013.monitoring.alerts import alert_rules
        return getattr(alert_rules, name)

    raise AttributeError(f"module 'gl_013.monitoring' has no attribute '{name}'")


def get_package_info() -> dict:
    """
    Return comprehensive package information.

    Returns:
        dict: Package metadata including version, modules, and capabilities.
    """
    return {
        "package": __package_name__,
        "version": __version__,
        "author": __author__,
        "description": "Monitoring and observability for GL-013 PREDICTMAINT",
        "modules": [
            "metrics",
            "health_checks",
        ],
        "subpackages": [
            "alerts",
            "grafana",
        ],
        "capabilities": [
            "prometheus_metrics",
            "alert_rules",
            "grafana_dashboards",
            "health_checks",
            "kubernetes_probes",
        ],
        "metrics_count": 50,  # Approximate number of defined metrics
        "alert_rules_count": 35,  # Number of predefined alert rules
    }
