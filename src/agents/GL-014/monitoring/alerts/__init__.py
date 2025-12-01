# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Alerts Module.

Alert rule definitions and management for heat exchanger monitoring including:
- Fouling and performance alerts
- Integration failure alerts
- System health alerts
- Notification channel integrations (Email, Slack, PagerDuty)

Author: GreenLang AI Agent Factory
Version: 1.0.0
"""

from .alert_rules import (
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
    SilenceRule,
    InhibitRule,

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
    evaluate_threshold,
)

__all__ = [
    # Main Alert Manager
    "AlertManager",

    # Alert Enumerations
    "AlertSeverity",
    "AlertState",
    "NotificationChannel",

    # Alert Rule Classes
    "AlertRule",
    "AlertCondition",
    "AlertInstance",
    "SilenceRule",
    "InhibitRule",

    # Built-in Alert Rules
    "HIGH_FOULING_RESISTANCE_ALERT",
    "LOW_THERMAL_EFFICIENCY_ALERT",
    "HIGH_PRESSURE_DROP_ALERT",
    "PERFORMANCE_DEGRADATION_ALERT",
    "CLEANING_OVERDUE_ALERT",
    "PREDICTION_ACCURACY_LOW_ALERT",
    "INTEGRATION_FAILURE_ALERT",
    "HIGH_API_LATENCY_ALERT",
    "HIGH_ERROR_RATE_ALERT",

    # Notification Classes
    "NotificationConfig",
    "EmailNotifier",
    "SlackNotifier",
    "PagerDutyNotifier",

    # Utility Functions
    "create_alert_manager",
    "get_default_alert_rules",
    "evaluate_threshold",
]
