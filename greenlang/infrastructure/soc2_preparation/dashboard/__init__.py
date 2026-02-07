# -*- coding: utf-8 -*-
"""
Dashboard and Monitoring Module - SEC-009 Phase 9

This module provides real-time compliance dashboards, metrics collection,
trend analysis, and alerting for SOC 2 Type II audit preparation.

Submodules:
    - metrics_collector: Compliance metrics calculation
    - trends: Historical trend analysis and prediction
    - alerts: Compliance alert conditions and notifications

Public API:
    - ComplianceMetrics: Metrics calculation and dashboard summaries
    - TrendAnalyzer: Historical trend analysis and prediction
    - ComplianceAlerts: Alert condition checking and notification
    - DashboardSummary: Combined metrics for dashboard display

Example:
    >>> from greenlang.infrastructure.soc2_preparation.dashboard import (
    ...     ComplianceMetrics,
    ...     TrendAnalyzer,
    ...     ComplianceAlerts,
    ... )
    >>> metrics = ComplianceMetrics(config)
    >>> summary = await metrics.get_dashboard_summary()
    >>> trends = TrendAnalyzer()
    >>> readiness_trend = await trends.analyze_readiness_trend(days=90)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging

from greenlang.infrastructure.soc2_preparation.dashboard.metrics_collector import (
    ComplianceMetrics,
    DashboardSummary,
    EvidenceCoverage,
    TestMetrics,
    FindingMetrics,
    SLAMetrics,
    AttestationMetrics,
)
from greenlang.infrastructure.soc2_preparation.dashboard.trends import (
    TrendAnalyzer,
    TrendPoint,
    FindingTrend,
    EvidenceTrend,
)
from greenlang.infrastructure.soc2_preparation.dashboard.alerts import (
    ComplianceAlerts,
    Alert,
    AlertCondition,
    AlertSeverity,
    AlertConfig,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Metrics
    "ComplianceMetrics",
    "DashboardSummary",
    "EvidenceCoverage",
    "TestMetrics",
    "FindingMetrics",
    "SLAMetrics",
    "AttestationMetrics",
    # Trends
    "TrendAnalyzer",
    "TrendPoint",
    "FindingTrend",
    "EvidenceTrend",
    # Alerts
    "ComplianceAlerts",
    "Alert",
    "AlertCondition",
    "AlertSeverity",
    "AlertConfig",
]
