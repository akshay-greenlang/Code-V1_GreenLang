# -*- coding: utf-8 -*-
"""
Alerts Package for GL-015 INSULSCAN.

This package provides alert rule definitions and management for
insulation scanning and thermal imaging monitoring.

Modules:
    alert_rules: Alert rule definitions and management

Example:
    >>> from monitoring.alerts import AlertManager, HIGH_HEAT_LOSS_ALERT
    >>> manager = AlertManager()
    >>> manager.register_rule(HIGH_HEAT_LOSS_ALERT)
    >>> alerts = manager.evaluate_rules({"heat_loss_wm": 500.0})

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from monitoring.alerts.alert_rules import (
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
    LOW_INSULATION_EFFICIENCY_ALERT,
    HIGH_API_LATENCY_ALERT,
    HIGH_ERROR_RATE_ALERT,
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

__all__ = [
    # Main Classes
    "AlertManager",
    # Alert Enumerations
    "AlertSeverity",
    "AlertState",
    "NotificationChannel",
    "ComparisonOperator",
    # Alert Data Classes
    "AlertCondition",
    "AlertRule",
    "AlertInstance",
    "SilenceRule",
    "InhibitRule",
    # Built-in Alert Rules
    "HIGH_HEAT_LOSS_ALERT",
    "CRITICAL_DEGRADATION_ALERT",
    "SAFETY_TEMPERATURE_EXCEEDED_ALERT",
    "INSPECTION_OVERDUE_ALERT",
    "MOISTURE_DETECTED_ALERT",
    "RAPID_DEGRADATION_RATE_ALERT",
    "INTEGRATION_FAILURE_ALERT",
    "LOW_INSULATION_EFFICIENCY_ALERT",
    "HIGH_API_LATENCY_ALERT",
    "HIGH_ERROR_RATE_ALERT",
    # Notification Classes
    "NotificationConfig",
    "BaseNotifier",
    "EmailNotifier",
    "SlackNotifier",
    "PagerDutyNotifier",
    # Utility Functions
    "create_alert_manager",
    "get_default_alert_rules",
    "evaluate_threshold",
    "reset_alert_manager",
]
