# -*- coding: utf-8 -*-
"""
GreenLang SLO/SLI Definitions & Error Budget Management Service - OBS-005

Production-grade SLO/SLI definition, error budget tracking, multi-window
burn rate alerting (Google SRE Book), Prometheus recording/alert rule
generation, Grafana dashboard generation, and compliance reporting.

Key Features:
    - **5 SLI Types**: Availability, latency, correctness, throughput,
      freshness with PromQL query building
    - **Error Budget**: Real-time budget calculation with Redis caching
      and TimescaleDB snapshot persistence
    - **Burn Rate Alerting**: Multi-window (fast/medium/slow) following
      Google SRE Book methodology with both long+short confirmation
    - **Recording Rules**: Auto-generated Prometheus recording rules
      for SLI ratios, error budgets, and burn rates
    - **Alert Rules**: Auto-generated Prometheus alert rules for burn
      rate and budget threshold alerts
    - **Dashboards**: Auto-generated Grafana JSON dashboards (overview
      and error budget detail)
    - **Compliance Reporting**: Weekly, monthly, quarterly reports with
      trend analysis
    - **OBS-004 Integration**: Alerting bridge to unified alerting
    - **Prometheus Metrics**: 10 self-monitoring metrics

Sub-modules:
    config              - SLOServiceConfig with env-var overrides
    models              - SLO, SLI, ErrorBudget, BurnRateAlert, SLOReport
    sli_calculator      - PromQL query building and Prometheus querying
    slo_manager         - SLO CRUD, YAML import/export, version history
    error_budget        - Error budget calculation and caching
    burn_rate           - Multi-window burn rate calculation and alerting
    recording_rules     - Prometheus recording rule generation
    alert_rules         - Prometheus alert rule generation
    dashboard_generator - Grafana dashboard JSON generation
    compliance_reporter - Periodic compliance report generation
    alerting_bridge     - OBS-004 unified alerting integration
    metrics             - Prometheus metric definitions
    api/                - FastAPI REST API (20 endpoints)
    setup               - configure_slo_service() + SLOService facade

Quick Start:
    >>> from greenlang.infrastructure.slo_service import (
    ...     configure_slo_service,
    ...     get_slo_service,
    ...     SLOServiceConfig,
    ... )
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> configure_slo_service(app)
    >>> svc = get_slo_service(app)
    >>> slos = svc.list_slos()

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-005 SLO/SLI Definitions & Error Budget Management
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from greenlang.infrastructure.slo_service.config import (
    SLOServiceConfig,
    get_config,
    set_config,
    reset_config,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from greenlang.infrastructure.slo_service.models import (
    SLI,
    SLIType,
    SLO,
    SLOWindow,
    BurnRateWindow,
    BudgetStatus,
    ErrorBudget,
    BurnRateAlert,
    SLOReport,
    SLOReportEntry,
)

# ---------------------------------------------------------------------------
# Core Components
# ---------------------------------------------------------------------------

from greenlang.infrastructure.slo_service.slo_manager import SLOManager
from greenlang.infrastructure.slo_service.alerting_bridge import AlertingBridge

# ---------------------------------------------------------------------------
# Calculators (functional modules)
# ---------------------------------------------------------------------------

from greenlang.infrastructure.slo_service.sli_calculator import (
    build_sli_ratio_query,
    build_error_rate_query,
    generate_recording_rule,
    query_prometheus,
    calculate_sli,
)
from greenlang.infrastructure.slo_service.error_budget import (
    calculate_error_budget,
    budget_consumption_rate,
    forecast_exhaustion,
    check_budget_policy,
    BudgetCache,
)
from greenlang.infrastructure.slo_service.burn_rate import (
    calculate_burn_rate,
    should_alert,
    evaluate_burn_rate_windows,
    time_to_exhaustion_hours,
    build_burn_rate_promql,
    generate_burn_rate_alert_rule,
    generate_all_burn_rate_rules,
)

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

from greenlang.infrastructure.slo_service.recording_rules import (
    generate_all_recording_rules,
    write_recording_rules_file,
)
from greenlang.infrastructure.slo_service.alert_rules import (
    generate_all_alert_rules,
    write_alert_rules_file,
)
from greenlang.infrastructure.slo_service.dashboard_generator import (
    generate_overview_dashboard,
    generate_error_budget_dashboard,
    write_dashboards,
)
from greenlang.infrastructure.slo_service.compliance_reporter import (
    generate_report,
    calculate_trend,
    store_report,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

from greenlang.infrastructure.slo_service.metrics import (
    PROMETHEUS_AVAILABLE,
    record_evaluation,
    record_budget_remaining,
    record_burn_rate as record_burn_rate_metric,
    update_definitions_count,
    update_compliance_percent,
    record_alert_fired,
    record_recording_rules_generated,
    record_budget_snapshot,
    record_report_generated,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

from greenlang.infrastructure.slo_service.setup import (
    SLOService,
    configure_slo_service,
    get_slo_service,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Configuration
    "SLOServiceConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Models
    "SLI",
    "SLIType",
    "SLO",
    "SLOWindow",
    "BurnRateWindow",
    "BudgetStatus",
    "ErrorBudget",
    "BurnRateAlert",
    "SLOReport",
    "SLOReportEntry",
    # Core Components
    "SLOManager",
    "AlertingBridge",
    # SLI Calculator
    "build_sli_ratio_query",
    "build_error_rate_query",
    "generate_recording_rule",
    "query_prometheus",
    "calculate_sli",
    # Error Budget
    "calculate_error_budget",
    "budget_consumption_rate",
    "forecast_exhaustion",
    "check_budget_policy",
    "BudgetCache",
    # Burn Rate
    "calculate_burn_rate",
    "should_alert",
    "evaluate_burn_rate_windows",
    "time_to_exhaustion_hours",
    "build_burn_rate_promql",
    "generate_burn_rate_alert_rule",
    "generate_all_burn_rate_rules",
    # Generators
    "generate_all_recording_rules",
    "write_recording_rules_file",
    "generate_all_alert_rules",
    "write_alert_rules_file",
    "generate_overview_dashboard",
    "generate_error_budget_dashboard",
    "write_dashboards",
    # Compliance Reporter
    "generate_report",
    "calculate_trend",
    "store_report",
    # Metrics
    "PROMETHEUS_AVAILABLE",
    "record_evaluation",
    "record_budget_remaining",
    "record_burn_rate_metric",
    "update_definitions_count",
    "update_compliance_percent",
    "record_alert_fired",
    "record_recording_rules_generated",
    "record_budget_snapshot",
    "record_report_generated",
    # Setup
    "SLOService",
    "configure_slo_service",
    "get_slo_service",
]

logger.debug("SLO Service module loaded: version=%s", __version__)
