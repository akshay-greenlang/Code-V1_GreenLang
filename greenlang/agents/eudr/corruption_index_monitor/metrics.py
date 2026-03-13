# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-019: Corruption Index Monitor

16 Prometheus metrics for corruption index monitor agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_cim_`` prefix (GreenLang EUDR Corruption
Index Monitor) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics (16 per PRD Section 7.3):
    Counters (9):
        1.  gl_eudr_cim_cpi_queries_total              - CPI score queries performed
        2.  gl_eudr_cim_wgi_queries_total               - WGI indicator queries performed
        3.  gl_eudr_cim_bribery_assessments_total       - Bribery risk assessments completed
        4.  gl_eudr_cim_institutional_assessments_total  - Institutional quality evaluations
        5.  gl_eudr_cim_trend_analyses_total             - Trend analyses completed
        6.  gl_eudr_cim_correlation_analyses_total       - Correlation analyses completed
        7.  gl_eudr_cim_alerts_generated_total           - Alerts generated
        8.  gl_eudr_cim_compliance_impacts_total         - Compliance impacts assessed
        9.  gl_eudr_cim_api_errors_total                 - API errors by operation

    Histograms (3):
        10. gl_eudr_cim_query_duration_seconds           - CPI/WGI query latency
        11. gl_eudr_cim_analysis_duration_seconds         - Trend/institutional analysis latency
        12. gl_eudr_cim_correlation_duration_seconds      - Correlation analysis latency

    Gauges (4):
        13. gl_eudr_cim_monitored_countries               - Number of actively monitored countries
        14. gl_eudr_cim_high_risk_countries                - Number of high/critical risk countries
        15. gl_eudr_cim_active_alerts                      - Number of active (unacknowledged) alerts
        16. gl_eudr_cim_data_freshness_days                - Age of most recent index data in days

Label Values Reference:
    country_code:
        ISO 3166-1 alpha-2 codes (e.g. BR, ID, CD, CI, GH).
    dimension:
        voice_accountability, political_stability,
        government_effectiveness, regulatory_quality,
        rule_of_law, control_of_corruption.
    risk_level:
        low, moderate, high, critical.
    sector:
        forestry, customs, agriculture, mining, extraction, judiciary.
    alert_severity:
        low, medium, high, critical.
    index_type:
        cpi, wgi, bribery, institutional.
    operation:
        query_cpi, query_wgi, assess_bribery, evaluate_iq,
        analyze_trend, analyze_correlation, generate_alert,
        assess_compliance, build_profile.

Example:
    >>> from greenlang.agents.eudr.corruption_index_monitor.metrics import (
    ...     record_cpi_query,
    ...     record_wgi_query,
    ...     record_alert_generated,
    ...     observe_query_duration,
    ...     set_monitored_countries,
    ...     set_high_risk_countries,
    ... )
    >>> record_cpi_query("BR", "high")
    >>> record_wgi_query("ID", "control_of_corruption")
    >>> record_alert_generated("BR", "high")
    >>> observe_query_duration(0.045, "cpi")
    >>> set_monitored_countries(180)
    >>> set_high_risk_countries(42, "critical")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; "
        "corruption index monitor metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(name: str, doc: str, labels: list) -> Counter:
        """Safely create or retrieve a Counter metric."""
        try:
            return Counter(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

    def _safe_histogram(name: str, doc: str, labels: list, buckets: tuple) -> Histogram:
        """Safely create or retrieve a Histogram metric."""
        try:
            return Histogram(name, doc, labels, buckets=buckets, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

    def _safe_gauge(name: str, doc: str, labels: list) -> Gauge:
        """Safely create or retrieve a Gauge metric."""
        try:
            return Gauge(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors[name]

# ---------------------------------------------------------------------------
# Metric definitions (16 metrics)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Counters (9)
    _cpi_queries_total = _safe_counter(
        "gl_eudr_cim_cpi_queries_total",
        "Total number of CPI score queries performed",
        ["country_code", "risk_level"],
    )

    _wgi_queries_total = _safe_counter(
        "gl_eudr_cim_wgi_queries_total",
        "Total number of WGI indicator queries performed",
        ["country_code", "dimension"],
    )

    _bribery_assessments_total = _safe_counter(
        "gl_eudr_cim_bribery_assessments_total",
        "Total number of bribery risk assessments completed",
        ["country_code", "sector"],
    )

    _institutional_assessments_total = _safe_counter(
        "gl_eudr_cim_institutional_assessments_total",
        "Total number of institutional quality evaluations completed",
        ["country_code"],
    )

    _trend_analyses_total = _safe_counter(
        "gl_eudr_cim_trend_analyses_total",
        "Total number of trend analyses completed",
        ["country_code", "index_type"],
    )

    _correlation_analyses_total = _safe_counter(
        "gl_eudr_cim_correlation_analyses_total",
        "Total number of deforestation-corruption correlation analyses completed",
        ["country_code", "index_type"],
    )

    _alerts_generated_total = _safe_counter(
        "gl_eudr_cim_alerts_generated_total",
        "Total number of alerts generated",
        ["country_code", "alert_severity"],
    )

    _compliance_impacts_total = _safe_counter(
        "gl_eudr_cim_compliance_impacts_total",
        "Total number of compliance impact assessments completed",
        ["country_code", "risk_level"],
    )

    _api_errors_total = _safe_counter(
        "gl_eudr_cim_api_errors_total",
        "Total number of API errors by operation",
        ["operation"],
    )

    # Histograms (3)
    _query_duration_seconds = _safe_histogram(
        "gl_eudr_cim_query_duration_seconds",
        "CPI/WGI query duration in seconds",
        ["index_type"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    _analysis_duration_seconds = _safe_histogram(
        "gl_eudr_cim_analysis_duration_seconds",
        "Trend/institutional analysis duration in seconds",
        ["analysis_type"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _correlation_duration_seconds = _safe_histogram(
        "gl_eudr_cim_correlation_duration_seconds",
        "Deforestation-corruption correlation analysis duration in seconds",
        ["country_code"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    # Gauges (4)
    _monitored_countries = _safe_gauge(
        "gl_eudr_cim_monitored_countries",
        "Number of actively monitored countries",
        [],
    )

    _high_risk_countries = _safe_gauge(
        "gl_eudr_cim_high_risk_countries",
        "Number of countries classified as high or critical risk",
        ["risk_level"],
    )

    _active_alerts = _safe_gauge(
        "gl_eudr_cim_active_alerts",
        "Number of active (unacknowledged) alerts",
        ["alert_severity"],
    )

    _data_freshness_days = _safe_gauge(
        "gl_eudr_cim_data_freshness_days",
        "Age of most recent index data in days",
        ["index_type"],
    )


# ---------------------------------------------------------------------------
# Helper functions (16 functions matching 16 metrics)
# ---------------------------------------------------------------------------


def record_cpi_query(
    country_code: str = "unknown",
    risk_level: str = "unknown",
) -> None:
    """Record a CPI score query.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        risk_level: Risk level result (low, moderate, high, critical).
    """
    if PROMETHEUS_AVAILABLE:
        _cpi_queries_total.labels(
            country_code=country_code,
            risk_level=risk_level,
        ).inc()


def record_wgi_query(
    country_code: str = "unknown",
    dimension: str = "all",
) -> None:
    """Record a WGI indicator query.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        dimension: WGI dimension queried (or "all" for composite).
    """
    if PROMETHEUS_AVAILABLE:
        _wgi_queries_total.labels(
            country_code=country_code,
            dimension=dimension,
        ).inc()


def record_bribery_assessment(
    country_code: str = "unknown",
    sector: str = "unknown",
) -> None:
    """Record a bribery risk assessment completion.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        sector: Bribery sector assessed (forestry, customs, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _bribery_assessments_total.labels(
            country_code=country_code,
            sector=sector,
        ).inc()


def record_institutional_assessment(
    country_code: str = "unknown",
) -> None:
    """Record an institutional quality evaluation completion.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _institutional_assessments_total.labels(
            country_code=country_code,
        ).inc()


def record_trend_analysis(
    country_code: str = "unknown",
    index_type: str = "cpi",
) -> None:
    """Record a trend analysis completion.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Index type analyzed (cpi, wgi, bribery, institutional).
    """
    if PROMETHEUS_AVAILABLE:
        _trend_analyses_total.labels(
            country_code=country_code,
            index_type=index_type,
        ).inc()


def record_correlation_analysis(
    country_code: str = "unknown",
    index_type: str = "cpi",
) -> None:
    """Record a deforestation-corruption correlation analysis completion.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Corruption index used in correlation.
    """
    if PROMETHEUS_AVAILABLE:
        _correlation_analyses_total.labels(
            country_code=country_code,
            index_type=index_type,
        ).inc()


def record_alert_generated(
    country_code: str = "unknown",
    alert_severity: str = "medium",
) -> None:
    """Record an alert generation.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        alert_severity: Alert severity (low, medium, high, critical).
    """
    if PROMETHEUS_AVAILABLE:
        _alerts_generated_total.labels(
            country_code=country_code,
            alert_severity=alert_severity,
        ).inc()


def record_compliance_impact(
    country_code: str = "unknown",
    risk_level: str = "unknown",
) -> None:
    """Record a compliance impact assessment completion.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        risk_level: Assessed risk level.
    """
    if PROMETHEUS_AVAILABLE:
        _compliance_impacts_total.labels(
            country_code=country_code,
            risk_level=risk_level,
        ).inc()


def record_api_error(operation: str) -> None:
    """Record an API error.

    Args:
        operation: Operation name (query_cpi, query_wgi, assess_bribery, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(operation=operation).inc()


def observe_query_duration(
    duration_seconds: float,
    index_type: str = "cpi",
) -> None:
    """Observe CPI/WGI query duration.

    Args:
        duration_seconds: Duration in seconds.
        index_type: Index type queried (cpi, wgi).
    """
    if PROMETHEUS_AVAILABLE:
        _query_duration_seconds.labels(
            index_type=index_type,
        ).observe(duration_seconds)


def observe_analysis_duration(
    duration_seconds: float,
    analysis_type: str = "trend",
) -> None:
    """Observe trend/institutional analysis duration.

    Args:
        duration_seconds: Duration in seconds.
        analysis_type: Type of analysis (trend, institutional, bribery).
    """
    if PROMETHEUS_AVAILABLE:
        _analysis_duration_seconds.labels(
            analysis_type=analysis_type,
        ).observe(duration_seconds)


def observe_correlation_duration(
    duration_seconds: float,
    country_code: str = "unknown",
) -> None:
    """Observe deforestation-corruption correlation analysis duration.

    Args:
        duration_seconds: Duration in seconds.
        country_code: ISO 3166-1 alpha-2 country code.
    """
    if PROMETHEUS_AVAILABLE:
        _correlation_duration_seconds.labels(
            country_code=country_code,
        ).observe(duration_seconds)


def set_monitored_countries(count: int) -> None:
    """Set the number of actively monitored countries.

    Args:
        count: Number of monitored countries (0-250).
    """
    if PROMETHEUS_AVAILABLE:
        _monitored_countries.set(count)


def set_high_risk_countries(
    count: int,
    risk_level: str = "high",
) -> None:
    """Set the number of high/critical risk countries.

    Args:
        count: Number of high/critical risk countries.
        risk_level: Risk level (high or critical).
    """
    if PROMETHEUS_AVAILABLE:
        _high_risk_countries.labels(risk_level=risk_level).set(count)


def set_active_alerts(
    count: int,
    alert_severity: str = "all",
) -> None:
    """Set the number of active (unacknowledged) alerts.

    Args:
        count: Number of active alerts.
        alert_severity: Alert severity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_alerts.labels(alert_severity=alert_severity).set(count)


def set_data_freshness_days(
    days: float,
    index_type: str = "cpi",
) -> None:
    """Set the age of most recent index data in days.

    Args:
        days: Number of days since last data update.
        index_type: Index type (cpi, wgi).
    """
    if PROMETHEUS_AVAILABLE:
        _data_freshness_days.labels(index_type=index_type).set(days)
