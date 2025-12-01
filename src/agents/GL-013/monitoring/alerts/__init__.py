"""
GL-013 PREDICTMAINT - Alerts Subpackage

Alert rules and Prometheus alerting configuration for predictive maintenance.

This subpackage provides:
    - Predefined alert rules for equipment health, failure prediction,
      vibration analysis, temperature monitoring, and system performance
    - Prometheus-compatible alert rule export
    - Alert severity and category management

Files:
    - alert_rules.py: Python alert rule definitions
    - prometheus_rules.yaml: Prometheus alerting rules in YAML format

Example:
    >>> from gl_013.monitoring.alerts import ALERT_RULES, AlertSeverity
    >>> critical_rules = [r for r in ALERT_RULES if r.severity == AlertSeverity.CRITICAL]
    >>> print(f"Critical alerts: {len(critical_rules)}")

Author: GL-MonitoringEngineer
Version: 1.0.0
"""

from gl_013.monitoring.alerts.alert_rules import (
    # Enums
    AlertSeverity,
    AlertCategory,
    NotificationChannel,
    # Data classes
    AlertThreshold,
    AlertRule,
    AlertGroup,
    # Rule collections
    ALERT_RULES,
    ALERT_GROUPS,
    EQUIPMENT_HEALTH_RULES,
    FAILURE_PREDICTION_RULES,
    VIBRATION_RULES,
    TEMPERATURE_RULES,
    ANOMALY_RULES,
    MAINTENANCE_RULES,
    INTEGRATION_RULES,
    SYSTEM_RULES,
    # Helper functions
    get_rules_by_severity,
    get_rules_by_category,
    get_enabled_rules,
    export_prometheus_rules,
    get_rule_by_name,
)

__version__ = "1.0.0"

__all__ = [
    # Enums
    "AlertSeverity",
    "AlertCategory",
    "NotificationChannel",
    # Data classes
    "AlertThreshold",
    "AlertRule",
    "AlertGroup",
    # Rule collections
    "ALERT_RULES",
    "ALERT_GROUPS",
    "EQUIPMENT_HEALTH_RULES",
    "FAILURE_PREDICTION_RULES",
    "VIBRATION_RULES",
    "TEMPERATURE_RULES",
    "ANOMALY_RULES",
    "MAINTENANCE_RULES",
    "INTEGRATION_RULES",
    "SYSTEM_RULES",
    # Helper functions
    "get_rules_by_severity",
    "get_rules_by_category",
    "get_enabled_rules",
    "export_prometheus_rules",
    "get_rule_by_name",
]
