# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-025: Risk Mitigation Advisor

18 Prometheus metrics for risk mitigation advisor agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_rma_`` prefix (GreenLang EUDR Risk
Mitigation Advisor) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD):
    Counters (8):
        1.  gl_eudr_rma_strategies_recommended_total     - Strategies recommended
        2.  gl_eudr_rma_plans_created_total               - Remediation plans created
        3.  gl_eudr_rma_milestones_completed_total        - Milestones completed
        4.  gl_eudr_rma_capacity_enrollments_total        - Capacity building enrollments
        5.  gl_eudr_rma_measures_searched_total            - Measure library searches
        6.  gl_eudr_rma_effectiveness_measured_total       - Effectiveness measurements
        7.  gl_eudr_rma_trigger_events_total               - Monitoring trigger events
        8.  gl_eudr_rma_api_errors_total                   - API errors by operation

    Histograms (4):
        9.  gl_eudr_rma_strategy_latency_seconds          - Strategy recommendation latency
        10. gl_eudr_rma_plan_generation_seconds            - Plan generation duration
        11. gl_eudr_rma_optimization_seconds               - Optimization solve duration
        12. gl_eudr_rma_effectiveness_calc_seconds          - Effectiveness calc duration

    Gauges (6):
        13. gl_eudr_rma_active_plans                       - Number of active plans
        14. gl_eudr_rma_active_enrollments                 - Active capacity enrollments
        15. gl_eudr_rma_library_measures                   - Measures in library
        16. gl_eudr_rma_pending_adjustments                - Pending adaptive adjustments
        17. gl_eudr_rma_total_risk_reduction               - Aggregate risk reduction pct
        18. gl_eudr_rma_optimization_backlog                - Optimization requests queued

Label Values Reference:
    risk_category:
        country, supplier, commodity, corruption, deforestation,
        indigenous_rights, protected_areas, legal_compliance.
    iso_31000_type:
        avoid, reduce, share, retain.
    plan_status:
        draft, active, on_track, at_risk, delayed, completed,
        suspended, abandoned.
    commodity:
        cattle, cocoa, coffee, palm_oil, rubber, soya, wood.
    trigger_type:
        country_reclassification, supplier_risk_spike, deforestation_alert,
        indigenous_violation, protected_encroachment, audit_nonconformance.
    adjustment_type:
        plan_acceleration, scope_expansion, strategy_replacement,
        emergency_response, plan_deescalation.
    operation:
        recommend_strategy, create_plan, track_effectiveness,
        monitor_triggers, optimize_budget, search_measures,
        generate_report, collaborate.

Example:
    >>> from greenlang.agents.eudr.risk_mitigation_advisor.metrics import (
    ...     record_strategy_recommended,
    ...     record_plan_created,
    ...     record_effectiveness_measured,
    ...     observe_strategy_latency,
    ...     set_active_plans,
    ...     set_library_measures,
    ... )
    >>> record_strategy_recommended("country", "reduce")
    >>> record_plan_created("active", "palm_oil")
    >>> record_effectiveness_measured("supplier")
    >>> observe_strategy_latency(0.85, "xgboost")
    >>> set_active_plans(42, "active")
    >>> set_library_measures(512)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025 Risk Mitigation Advisor (GL-EUDR-RMA-025)
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
        "risk mitigation advisor metrics disabled"
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
            return _REGISTRY._names_to_collectors.get(
                name, Counter(name, doc, labels)
            )

    def _safe_histogram(
        name: str, doc: str, labels: list, buckets: tuple
    ) -> Histogram:
        """Safely create or retrieve a Histogram metric."""
        try:
            return Histogram(
                name, doc, labels, buckets=buckets, registry=_REGISTRY
            )
        except ValueError:
            return _REGISTRY._names_to_collectors.get(
                name, Histogram(name, doc, labels, buckets=buckets)
            )

    def _safe_gauge(name: str, doc: str, labels: list) -> Gauge:
        """Safely create or retrieve a Gauge metric."""
        try:
            return Gauge(name, doc, labels, registry=_REGISTRY)
        except ValueError:
            return _REGISTRY._names_to_collectors.get(
                name, Gauge(name, doc, labels)
            )

# ---------------------------------------------------------------------------
# Metric definitions (18 metrics)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # Counters (8)
    _strategies_recommended_total = _safe_counter(
        "gl_eudr_rma_strategies_recommended_total",
        "Total number of mitigation strategies recommended",
        ["risk_category", "iso_31000_type"],
    )

    _plans_created_total = _safe_counter(
        "gl_eudr_rma_plans_created_total",
        "Total number of remediation plans created",
        ["plan_status", "commodity"],
    )

    _milestones_completed_total = _safe_counter(
        "gl_eudr_rma_milestones_completed_total",
        "Total number of plan milestones completed",
        ["plan_status"],
    )

    _capacity_enrollments_total = _safe_counter(
        "gl_eudr_rma_capacity_enrollments_total",
        "Total number of supplier capacity building enrollments",
        ["commodity", "tier"],
    )

    _measures_searched_total = _safe_counter(
        "gl_eudr_rma_measures_searched_total",
        "Total number of measure library searches performed",
        ["risk_category"],
    )

    _effectiveness_measured_total = _safe_counter(
        "gl_eudr_rma_effectiveness_measured_total",
        "Total number of effectiveness measurements completed",
        ["risk_category"],
    )

    _trigger_events_total = _safe_counter(
        "gl_eudr_rma_trigger_events_total",
        "Total monitoring trigger events detected",
        ["trigger_type", "severity"],
    )

    _api_errors_total = _safe_counter(
        "gl_eudr_rma_api_errors_total",
        "Total API errors by operation",
        ["operation", "error_type"],
    )

    # Histograms (4)
    _strategy_latency = _safe_histogram(
        "gl_eudr_rma_strategy_latency_seconds",
        "Strategy recommendation processing latency in seconds",
        ["model_type"],
        (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )

    _plan_generation_duration = _safe_histogram(
        "gl_eudr_rma_plan_generation_seconds",
        "Remediation plan generation duration in seconds",
        ["template_type"],
        (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )

    _optimization_duration = _safe_histogram(
        "gl_eudr_rma_optimization_seconds",
        "Cost-benefit optimization solve duration in seconds",
        ["solver_type"],
        (0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
    )

    _effectiveness_calc_duration = _safe_histogram(
        "gl_eudr_rma_effectiveness_calc_seconds",
        "Effectiveness calculation duration in seconds",
        ["calculation_type"],
        (0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0),
    )

    # Gauges (6)
    _active_plans = _safe_gauge(
        "gl_eudr_rma_active_plans",
        "Number of currently active remediation plans",
        ["plan_status"],
    )

    _active_enrollments = _safe_gauge(
        "gl_eudr_rma_active_enrollments",
        "Number of active capacity building enrollments",
        ["commodity"],
    )

    _library_measures = _safe_gauge(
        "gl_eudr_rma_library_measures",
        "Total number of measures in the mitigation library",
        [],
    )

    _pending_adjustments = _safe_gauge(
        "gl_eudr_rma_pending_adjustments",
        "Number of pending adaptive plan adjustments",
        ["adjustment_type"],
    )

    _total_risk_reduction = _safe_gauge(
        "gl_eudr_rma_total_risk_reduction",
        "Aggregate risk reduction percentage achieved",
        ["risk_category"],
    )

    _optimization_backlog = _safe_gauge(
        "gl_eudr_rma_optimization_backlog",
        "Number of optimization requests in queue",
        [],
    )


# ---------------------------------------------------------------------------
# Counter helper functions (8)
# ---------------------------------------------------------------------------


def record_strategy_recommended(
    risk_category: str = "unknown",
    iso_31000_type: str = "reduce",
) -> None:
    """Record a strategy recommendation event.

    Args:
        risk_category: Risk category of the recommendation.
        iso_31000_type: ISO 31000 treatment type.
    """
    if PROMETHEUS_AVAILABLE:
        _strategies_recommended_total.labels(
            risk_category=risk_category,
            iso_31000_type=iso_31000_type,
        ).inc()


def record_plan_created(
    plan_status: str = "draft",
    commodity: str = "unknown",
) -> None:
    """Record a remediation plan creation event.

    Args:
        plan_status: Initial plan status.
        commodity: Target commodity.
    """
    if PROMETHEUS_AVAILABLE:
        _plans_created_total.labels(
            plan_status=plan_status,
            commodity=commodity,
        ).inc()


def record_milestone_completed(
    plan_status: str = "active",
) -> None:
    """Record a milestone completion event.

    Args:
        plan_status: Current plan status.
    """
    if PROMETHEUS_AVAILABLE:
        _milestones_completed_total.labels(
            plan_status=plan_status,
        ).inc()


def record_capacity_enrollment(
    commodity: str = "unknown",
    tier: str = "1",
) -> None:
    """Record a capacity building enrollment event.

    Args:
        commodity: Commodity for the enrollment.
        tier: Capacity building tier (1-4).
    """
    if PROMETHEUS_AVAILABLE:
        _capacity_enrollments_total.labels(
            commodity=commodity,
            tier=tier,
        ).inc()


def record_measure_searched(
    risk_category: str = "unknown",
) -> None:
    """Record a measure library search event.

    Args:
        risk_category: Risk category searched.
    """
    if PROMETHEUS_AVAILABLE:
        _measures_searched_total.labels(
            risk_category=risk_category,
        ).inc()


def record_effectiveness_measured(
    risk_category: str = "unknown",
) -> None:
    """Record an effectiveness measurement event.

    Args:
        risk_category: Risk category measured.
    """
    if PROMETHEUS_AVAILABLE:
        _effectiveness_measured_total.labels(
            risk_category=risk_category,
        ).inc()


def record_trigger_event(
    trigger_type: str = "unknown",
    severity: str = "medium",
) -> None:
    """Record a monitoring trigger event.

    Args:
        trigger_type: Type of trigger event.
        severity: Event severity level.
    """
    if PROMETHEUS_AVAILABLE:
        _trigger_events_total.labels(
            trigger_type=trigger_type,
            severity=severity,
        ).inc()


def record_api_error(
    operation: str = "unknown",
    error_type: str = "unknown",
) -> None:
    """Record an API error event.

    Args:
        operation: Operation that failed.
        error_type: Type of error encountered.
    """
    if PROMETHEUS_AVAILABLE:
        _api_errors_total.labels(
            operation=operation,
            error_type=error_type,
        ).inc()


# ---------------------------------------------------------------------------
# Histogram helper functions (4)
# ---------------------------------------------------------------------------


def observe_strategy_latency(
    duration_seconds: float,
    model_type: str = "xgboost",
) -> None:
    """Observe strategy recommendation latency.

    Args:
        duration_seconds: Processing duration in seconds.
        model_type: ML model type used.
    """
    if PROMETHEUS_AVAILABLE:
        _strategy_latency.labels(
            model_type=model_type,
        ).observe(duration_seconds)


def observe_plan_generation_duration(
    duration_seconds: float,
    template_type: str = "standard",
) -> None:
    """Observe plan generation duration.

    Args:
        duration_seconds: Generation duration in seconds.
        template_type: Plan template type used.
    """
    if PROMETHEUS_AVAILABLE:
        _plan_generation_duration.labels(
            template_type=template_type,
        ).observe(duration_seconds)


def observe_optimization_duration(
    duration_seconds: float,
    solver_type: str = "linprog",
) -> None:
    """Observe optimization solve duration.

    Args:
        duration_seconds: Optimization duration in seconds.
        solver_type: Solver type used.
    """
    if PROMETHEUS_AVAILABLE:
        _optimization_duration.labels(
            solver_type=solver_type,
        ).observe(duration_seconds)


def observe_effectiveness_calc_duration(
    duration_seconds: float,
    calculation_type: str = "roi",
) -> None:
    """Observe effectiveness calculation duration.

    Args:
        duration_seconds: Calculation duration in seconds.
        calculation_type: Type of calculation performed.
    """
    if PROMETHEUS_AVAILABLE:
        _effectiveness_calc_duration.labels(
            calculation_type=calculation_type,
        ).observe(duration_seconds)


# ---------------------------------------------------------------------------
# Gauge helper functions (6)
# ---------------------------------------------------------------------------


def set_active_plans(
    count: int,
    plan_status: str = "active",
) -> None:
    """Set the number of active remediation plans.

    Args:
        count: Current count of active plans.
        plan_status: Plan status category.
    """
    if PROMETHEUS_AVAILABLE:
        _active_plans.labels(
            plan_status=plan_status,
        ).set(count)


def set_active_enrollments(
    count: int,
    commodity: str = "all",
) -> None:
    """Set the number of active capacity building enrollments.

    Args:
        count: Current count of active enrollments.
        commodity: Commodity filter.
    """
    if PROMETHEUS_AVAILABLE:
        _active_enrollments.labels(
            commodity=commodity,
        ).set(count)


def set_library_measures(count: int) -> None:
    """Set the total number of measures in the library.

    Args:
        count: Current count of measures.
    """
    if PROMETHEUS_AVAILABLE:
        _library_measures.labels().set(count)


def set_pending_adjustments(
    count: int,
    adjustment_type: str = "all",
) -> None:
    """Set the number of pending adaptive adjustments.

    Args:
        count: Current count of pending adjustments.
        adjustment_type: Adjustment type filter.
    """
    if PROMETHEUS_AVAILABLE:
        _pending_adjustments.labels(
            adjustment_type=adjustment_type,
        ).set(count)


def set_total_risk_reduction(
    reduction_pct: float,
    risk_category: str = "composite",
) -> None:
    """Set the aggregate risk reduction percentage.

    Args:
        reduction_pct: Current aggregate risk reduction.
        risk_category: Risk category for the reduction.
    """
    if PROMETHEUS_AVAILABLE:
        _total_risk_reduction.labels(
            risk_category=risk_category,
        ).set(reduction_pct)


def set_optimization_backlog(count: int) -> None:
    """Set the number of queued optimization requests.

    Args:
        count: Current backlog count.
    """
    if PROMETHEUS_AVAILABLE:
        _optimization_backlog.labels().set(count)
