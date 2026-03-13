# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-018: Commodity Risk Analyzer

18 Prometheus metrics for commodity risk analyzer agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_cra_`` prefix (GreenLang EUDR Commodity
Risk Analyzer) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (9):
        1.  gl_eudr_cra_profiles_created_total           - Commodity profiles created
        2.  gl_eudr_cra_derived_products_analyzed_total   - Derived products analyzed
        3.  gl_eudr_cra_price_queries_total               - Price volatility queries
        4.  gl_eudr_cra_forecasts_generated_total         - Production forecasts generated
        5.  gl_eudr_cra_substitutions_detected_total      - Substitution events detected
        6.  gl_eudr_cra_compliance_checks_total           - Compliance checks performed
        7.  gl_eudr_cra_dd_workflows_initiated_total      - DD workflows initiated
        8.  gl_eudr_cra_portfolio_analyses_total          - Portfolio analyses completed
        9.  gl_eudr_cra_api_errors_total                  - API errors by operation

    Histograms (4):
        10. gl_eudr_cra_profile_duration_seconds          - Commodity profiling latency
        11. gl_eudr_cra_analysis_duration_seconds          - Derived product analysis latency
        12. gl_eudr_cra_forecast_duration_seconds          - Forecast generation latency
        13. gl_eudr_cra_portfolio_duration_seconds         - Portfolio aggregation latency

    Gauges (5):
        14. gl_eudr_cra_active_workflows                   - Active DD workflows
        15. gl_eudr_cra_monitored_commodities              - Number of actively monitored commodities
        16. gl_eudr_cra_portfolio_risk_exposure             - Current portfolio risk exposure
        17. gl_eudr_cra_high_risk_commodities              - Number of high/critical risk commodities
        18. gl_eudr_cra_active_substitution_alerts          - Active substitution alerts

Label Values Reference:
    commodity_type:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    risk_level:
        low, medium, high, critical.
    processing_stage:
        raw, primary, secondary, tertiary, finished, packaged.
    derived_category:
        See DerivedProductCategory enum values.
    volatility_level:
        low, moderate, high, extreme.
    market_condition:
        stable, volatile, disrupted, crisis.
    dd_level:
        simplified, standard, enhanced.
    compliance_status:
        compliant, partially_compliant, non_compliant, under_review, not_assessed.
    operation:
        profile, analyze, query, forecast, detect, check,
        initiate, aggregate, compare, export.

Example:
    >>> from greenlang.agents.eudr.commodity_risk_analyzer.metrics import (
    ...     record_profile_created,
    ...     record_derived_product_analyzed,
    ...     record_price_query,
    ...     observe_profile_duration,
    ...     set_monitored_commodities,
    ... )
    >>> record_profile_created("cocoa", "high")
    >>> record_derived_product_analyzed("cocoa", "chocolate")
    >>> record_price_query("coffee", "volatile")
    >>> observe_profile_duration(0.125, "cocoa")
    >>> set_monitored_commodities(7)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
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
        "commodity risk analyzer metrics disabled"
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
# Metric definitions (18 metrics)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Counters (9)
    _profiles_created_total = _safe_counter(
        "gl_eudr_cra_profiles_created_total",
        "Total number of commodity profiles created",
        ["commodity_type", "risk_level"],
    )

    _derived_products_analyzed_total = _safe_counter(
        "gl_eudr_cra_derived_products_analyzed_total",
        "Total number of derived products analyzed",
        ["commodity_type", "derived_category"],
    )

    _price_queries_total = _safe_counter(
        "gl_eudr_cra_price_queries_total",
        "Total number of price volatility queries",
        ["commodity_type", "volatility_level"],
    )

    _forecasts_generated_total = _safe_counter(
        "gl_eudr_cra_forecasts_generated_total",
        "Total number of production forecasts generated",
        ["commodity_type", "region"],
    )

    _substitutions_detected_total = _safe_counter(
        "gl_eudr_cra_substitutions_detected_total",
        "Total number of substitution events detected",
        ["from_commodity", "to_commodity"],
    )

    _compliance_checks_total = _safe_counter(
        "gl_eudr_cra_compliance_checks_total",
        "Total number of compliance checks performed",
        ["commodity_type", "compliance_status"],
    )

    _dd_workflows_initiated_total = _safe_counter(
        "gl_eudr_cra_dd_workflows_initiated_total",
        "Total number of due diligence workflows initiated",
        ["commodity_type", "dd_level"],
    )

    _portfolio_analyses_total = _safe_counter(
        "gl_eudr_cra_portfolio_analyses_total",
        "Total number of portfolio analyses completed",
        ["strategy"],
    )

    _api_errors_total = _safe_counter(
        "gl_eudr_cra_api_errors_total",
        "Total number of API errors by operation",
        ["operation"],
    )

    # Histograms (4)
    _profile_duration_seconds = _safe_histogram(
        "gl_eudr_cra_profile_duration_seconds",
        "Commodity profiling duration in seconds",
        ["commodity_type"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _analysis_duration_seconds = _safe_histogram(
        "gl_eudr_cra_analysis_duration_seconds",
        "Derived product analysis duration in seconds",
        ["commodity_type", "processing_stage"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _forecast_duration_seconds = _safe_histogram(
        "gl_eudr_cra_forecast_duration_seconds",
        "Forecast generation duration in seconds",
        ["commodity_type"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    _portfolio_duration_seconds = _safe_histogram(
        "gl_eudr_cra_portfolio_duration_seconds",
        "Portfolio aggregation duration in seconds",
        ["strategy"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )

    # Gauges (5)
    _active_workflows = _safe_gauge(
        "gl_eudr_cra_active_workflows",
        "Number of active due diligence workflows",
        ["commodity_type", "dd_level"],
    )

    _monitored_commodities = _safe_gauge(
        "gl_eudr_cra_monitored_commodities",
        "Number of actively monitored commodities",
        [],
    )

    _portfolio_risk_exposure = _safe_gauge(
        "gl_eudr_cra_portfolio_risk_exposure",
        "Current portfolio risk exposure score",
        ["strategy"],
    )

    _high_risk_commodities = _safe_gauge(
        "gl_eudr_cra_high_risk_commodities",
        "Number of commodities classified as high or critical risk",
        ["risk_level"],
    )

    _active_substitution_alerts = _safe_gauge(
        "gl_eudr_cra_active_substitution_alerts",
        "Number of active substitution risk alerts",
        ["from_commodity", "to_commodity"],
    )


# ---------------------------------------------------------------------------
# Helper functions (18 functions)
# ---------------------------------------------------------------------------


def record_profile_created(
    commodity_type: str, risk_level: str = "unknown"
) -> None:
    """Record a commodity profile creation.

    Args:
        commodity_type: Commodity type (cattle, cocoa, coffee, etc.).
        risk_level: Risk level (low, medium, high, critical).
    """
    if PROMETHEUS_AVAILABLE:
        _profiles_created_total.labels(
            commodity_type=commodity_type,
            risk_level=risk_level,
        ).inc()


def record_derived_product_analyzed(
    commodity_type: str, derived_category: str = "unknown"
) -> None:
    """Record a derived product analysis.

    Args:
        commodity_type: Source commodity type.
        derived_category: Derived product category.
    """
    if PROMETHEUS_AVAILABLE:
        _derived_products_analyzed_total.labels(
            commodity_type=commodity_type,
            derived_category=derived_category,
        ).inc()


def record_price_query(
    commodity_type: str, volatility_level: str = "unknown"
) -> None:
    """Record a price volatility query.

    Args:
        commodity_type: Commodity type.
        volatility_level: Volatility level (low, moderate, high, extreme).
    """
    if PROMETHEUS_AVAILABLE:
        _price_queries_total.labels(
            commodity_type=commodity_type,
            volatility_level=volatility_level,
        ).inc()


def record_forecast_generated(
    commodity_type: str, region: str = "global"
) -> None:
    """Record a production forecast generation.

    Args:
        commodity_type: Commodity type.
        region: Region code (ISO 3166-1 alpha-2 or "global").
    """
    if PROMETHEUS_AVAILABLE:
        _forecasts_generated_total.labels(
            commodity_type=commodity_type,
            region=region,
        ).inc()


def record_substitution_detected(
    from_commodity: str, to_commodity: str
) -> None:
    """Record a substitution event detection.

    Args:
        from_commodity: Original commodity type.
        to_commodity: Substituted commodity type.
    """
    if PROMETHEUS_AVAILABLE:
        _substitutions_detected_total.labels(
            from_commodity=from_commodity,
            to_commodity=to_commodity,
        ).inc()


def record_compliance_check(
    commodity_type: str, compliance_status: str = "not_assessed"
) -> None:
    """Record a compliance check.

    Args:
        commodity_type: Commodity type.
        compliance_status: Compliance status result.
    """
    if PROMETHEUS_AVAILABLE:
        _compliance_checks_total.labels(
            commodity_type=commodity_type,
            compliance_status=compliance_status,
        ).inc()


def record_dd_workflow_initiated(
    commodity_type: str, dd_level: str = "standard"
) -> None:
    """Record a due diligence workflow initiation.

    Args:
        commodity_type: Commodity type.
        dd_level: Due diligence level (simplified, standard, enhanced).
    """
    if PROMETHEUS_AVAILABLE:
        _dd_workflows_initiated_total.labels(
            commodity_type=commodity_type,
            dd_level=dd_level,
        ).inc()


def record_portfolio_analysis(strategy: str = "balanced") -> None:
    """Record a portfolio analysis completion.

    Args:
        strategy: Portfolio strategy (conservative, balanced, diversified, concentrated).
    """
    if PROMETHEUS_AVAILABLE:
        _portfolio_analyses_total.labels(strategy=strategy).inc()


def record_api_error(operation: str) -> None:
    """Record an API error.

    Args:
        operation: Operation name (profile, analyze, query, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(operation=operation).inc()


def observe_profile_duration(
    duration_seconds: float, commodity_type: str = "unknown"
) -> None:
    """Observe commodity profiling duration.

    Args:
        duration_seconds: Duration in seconds.
        commodity_type: Commodity type.
    """
    if PROMETHEUS_AVAILABLE:
        _profile_duration_seconds.labels(
            commodity_type=commodity_type,
        ).observe(duration_seconds)


def observe_analysis_duration(
    duration_seconds: float,
    commodity_type: str = "unknown",
    processing_stage: str = "unknown",
) -> None:
    """Observe derived product analysis duration.

    Args:
        duration_seconds: Duration in seconds.
        commodity_type: Commodity type.
        processing_stage: Processing stage (raw, primary, etc.).
    """
    if PROMETHEUS_AVAILABLE:
        _analysis_duration_seconds.labels(
            commodity_type=commodity_type,
            processing_stage=processing_stage,
        ).observe(duration_seconds)


def observe_forecast_duration(
    duration_seconds: float, commodity_type: str = "unknown"
) -> None:
    """Observe forecast generation duration.

    Args:
        duration_seconds: Duration in seconds.
        commodity_type: Commodity type.
    """
    if PROMETHEUS_AVAILABLE:
        _forecast_duration_seconds.labels(
            commodity_type=commodity_type,
        ).observe(duration_seconds)


def observe_portfolio_duration(
    duration_seconds: float, strategy: str = "balanced"
) -> None:
    """Observe portfolio aggregation duration.

    Args:
        duration_seconds: Duration in seconds.
        strategy: Portfolio strategy.
    """
    if PROMETHEUS_AVAILABLE:
        _portfolio_duration_seconds.labels(
            strategy=strategy,
        ).observe(duration_seconds)


def set_active_workflows(
    count: int,
    commodity_type: str = "all",
    dd_level: str = "all",
) -> None:
    """Set the number of active DD workflows.

    Args:
        count: Number of active workflows.
        commodity_type: Commodity type filter (or "all").
        dd_level: DD level filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_workflows.labels(
            commodity_type=commodity_type,
            dd_level=dd_level,
        ).set(count)


def set_monitored_commodities(count: int) -> None:
    """Set the number of actively monitored commodities.

    Args:
        count: Number of monitored commodities (0-7).
    """
    if PROMETHEUS_AVAILABLE:
        _monitored_commodities.set(count)


def set_portfolio_risk_exposure(
    exposure: float, strategy: str = "balanced"
) -> None:
    """Set the current portfolio risk exposure.

    Args:
        exposure: Portfolio risk exposure score (0-100).
        strategy: Portfolio strategy.
    """
    if PROMETHEUS_AVAILABLE:
        _portfolio_risk_exposure.labels(strategy=strategy).set(exposure)


def set_high_risk_commodities(
    count: int, risk_level: str = "high"
) -> None:
    """Set the number of high/critical risk commodities.

    Args:
        count: Number of high/critical risk commodities.
        risk_level: Risk level (high or critical).
    """
    if PROMETHEUS_AVAILABLE:
        _high_risk_commodities.labels(risk_level=risk_level).set(count)


def set_active_substitution_alerts(
    count: int,
    from_commodity: str = "all",
    to_commodity: str = "all",
) -> None:
    """Set the number of active substitution risk alerts.

    Args:
        count: Number of active alerts.
        from_commodity: Source commodity filter (or "all").
        to_commodity: Target commodity filter (or "all").
    """
    if PROMETHEUS_AVAILABLE:
        _active_substitution_alerts.labels(
            from_commodity=from_commodity,
            to_commodity=to_commodity,
        ).set(count)
