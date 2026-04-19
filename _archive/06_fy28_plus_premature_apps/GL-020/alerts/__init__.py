# -*- coding: utf-8 -*-
"""
GL-020 ECONOPULSE Alert System.

This module provides alert management components for economizer performance
monitoring, including alert generation, prioritization, deduplication,
escalation, and notification routing.

Components:
    - AlertManager: Main alert management class
    - Alert: Alert data model
    - AlertHistory: Alert history tracking

Example:
    >>> from greenlang.GL_020.alerts import AlertManager
    >>> manager = AlertManager(config)
    >>> alerts = manager.process_metrics(metrics)

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

from greenlang.GL_020.alerts.alert_manager import (
    Alert,
    AlertHistory,
    AlertManager,
)

__all__ = [
    "Alert",
    "AlertHistory",
    "AlertManager",
]
