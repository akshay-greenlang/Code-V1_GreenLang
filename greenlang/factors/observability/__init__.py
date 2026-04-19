# -*- coding: utf-8 -*-
"""Factors observability: Prometheus metrics, ASGI middleware, SLO/SLI, health checks (F070-F073)."""

from greenlang.factors.observability.prometheus_exporter import FactorsMetrics, get_factors_metrics
from greenlang.factors.observability.health import HealthStatus, get_health_status
from greenlang.factors.observability.prometheus import (
    FactorsMetricsMiddleware,
    metrics_endpoint,
    record_search_results,
    record_match_score,
    record_qa_failure,
    update_edition_gauge,
)
from greenlang.factors.observability.sla import (
    FACTORS_SLO,
    FACTORS_SLOS,
    SLIDefinition,
    SLODefinition,
    check_all_slis,
    check_sli,
    calculate_error_budget,
    compliance_summary,
)

__all__ = [
    # Metrics singleton
    "FactorsMetrics",
    "get_factors_metrics",
    # Health checks
    "HealthStatus",
    "get_health_status",
    # ASGI middleware + endpoint
    "FactorsMetricsMiddleware",
    "metrics_endpoint",
    # Convenience metric helpers
    "record_search_results",
    "record_match_score",
    "record_qa_failure",
    "update_edition_gauge",
    # SLO/SLI
    "FACTORS_SLO",
    "FACTORS_SLOS",
    "SLIDefinition",
    "SLODefinition",
    "check_all_slis",
    "check_sli",
    "calculate_error_budget",
    "compliance_summary",
]
