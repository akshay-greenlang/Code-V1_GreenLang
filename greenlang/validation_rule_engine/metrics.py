# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-019: Validation Rule Engine

12 Prometheus metrics for validation rule engine service monitoring with
graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_vre_`` prefix (GreenLang Validation Rule
Engine) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_vre_rules_registered_total             (Counter,   labels: rule_type, severity)
    2.  gl_vre_rule_sets_created_total            (Counter,   labels: pack_type)
    3.  gl_vre_evaluations_total                  (Counter,   labels: result, rule_type)
    4.  gl_vre_evaluation_failures_total          (Counter,   labels: severity)
    5.  gl_vre_conflicts_detected_total           (Counter,   labels: conflict_type)
    6.  gl_vre_reports_generated_total            (Counter,   labels: report_type, format)
    7.  gl_vre_rules_per_set                      (Histogram, buckets: set-size-scale)
    8.  gl_vre_evaluation_duration_seconds        (Histogram, labels: operation, buckets: eval-scale)
    9.  gl_vre_processing_duration_seconds        (Histogram, labels: operation, buckets: sub-second)
    10. gl_vre_active_rules                       (Gauge)
    11. gl_vre_active_rule_sets                   (Gauge)
    12. gl_vre_pass_rate                          (Gauge)

Label Values Reference:
    rule_type:
        range_check, format_validation, cross_field, regex, lookup,
        threshold, conditional, temporal, referential, custom,
        regulatory, completeness, uniqueness, consistency.
    severity:
        critical, error, warning, info, debug.
    pack_type:
        ghg_protocol, csrd_esrs, tcfd, cdp, sbti, iso14064, eudr,
        custom, internal, regulatory, compliance, quality.
    result:
        pass, fail, warning, error, skipped, not_applicable.
    conflict_type:
        contradiction, overlap, subsumption, redundancy,
        circular_dependency, priority_ambiguity, scope_collision,
        temporal_conflict.
    report_type:
        evaluation_summary, compliance_report, conflict_analysis,
        rule_coverage, audit_trail, quality_scorecard,
        trend_analysis, exception_report.
    format:
        json, html, pdf, csv, markdown, xml.
    operation (evaluation):
        single_rule, rule_set, compound_rule, batch,
        cross_field, conditional, full_pipeline.
    operation (processing):
        rule_register, rule_update, rule_delete, set_create,
        set_update, conflict_detect, conflict_resolve,
        report_generate, pack_import, pack_export,
        rule_activate, rule_deactivate.

Example:
    >>> from greenlang.validation_rule_engine.metrics import (
    ...     record_rule_registered,
    ...     record_evaluation,
    ...     set_active_rules,
    ... )
    >>> record_rule_registered("range_check", "error")
    >>> record_evaluation("pass", "range_check")
    >>> set_active_rules(42)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
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
        "prometheus_client not installed; validation rule engine metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Rules registered by rule type and severity level
    vre_rules_registered_total = Counter(
        "gl_vre_rules_registered_total",
        "Total validation rules registered in the rule engine",
        labelnames=["rule_type", "severity"],
    )

    # 2. Rule sets created by pack type (regulatory framework association)
    vre_rule_sets_created_total = Counter(
        "gl_vre_rule_sets_created_total",
        "Total rule sets created in the validation rule engine",
        labelnames=["pack_type"],
    )

    # 3. Rule evaluations by result outcome and rule type
    vre_evaluations_total = Counter(
        "gl_vre_evaluations_total",
        "Total rule evaluations performed by the validation rule engine",
        labelnames=["result", "rule_type"],
    )

    # 4. Evaluation failures by severity of the failed rule
    vre_evaluation_failures_total = Counter(
        "gl_vre_evaluation_failures_total",
        "Total rule evaluation failures by severity level",
        labelnames=["severity"],
    )

    # 5. Rule conflicts detected by conflict type
    vre_conflicts_detected_total = Counter(
        "gl_vre_conflicts_detected_total",
        "Total rule conflicts detected during evaluation or registration",
        labelnames=["conflict_type"],
    )

    # 6. Reports generated by report type and output format
    vre_reports_generated_total = Counter(
        "gl_vre_reports_generated_total",
        "Total validation reports generated by the rule engine",
        labelnames=["report_type", "format"],
    )

    # 7. Distribution of rules per rule set (set composition size)
    vre_rules_per_set = Histogram(
        "gl_vre_rules_per_set",
        "Distribution of the number of rules per rule set",
        buckets=(1, 5, 10, 25, 50, 100, 250, 500),
    )

    # 8. Evaluation duration for rule evaluation operations
    vre_evaluation_duration_seconds = Histogram(
        "gl_vre_evaluation_duration_seconds",
        "Duration of rule evaluation operations in seconds",
        labelnames=["operation"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30),
    )

    # 9. Processing duration for individual engine operations
    vre_processing_duration_seconds = Histogram(
        "gl_vre_processing_duration_seconds",
        "Validation rule engine operation processing duration in seconds",
        labelnames=["operation"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1, 5, 10),
    )

    # 10. Current number of active validation rules
    vre_active_rules = Gauge(
        "gl_vre_active_rules",
        "Current number of active validation rules in the engine",
    )

    # 11. Current number of active rule sets
    vre_active_rule_sets = Gauge(
        "gl_vre_active_rule_sets",
        "Current number of active rule sets in the engine",
    )

    # 12. Current overall pass rate across evaluations (0.0 to 1.0)
    vre_pass_rate = Gauge(
        "gl_vre_pass_rate",
        "Current overall pass rate of rule evaluations (0.0 to 1.0)",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    vre_rules_registered_total = None           # type: ignore[assignment]
    vre_rule_sets_created_total = None          # type: ignore[assignment]
    vre_evaluations_total = None                # type: ignore[assignment]
    vre_evaluation_failures_total = None        # type: ignore[assignment]
    vre_conflicts_detected_total = None         # type: ignore[assignment]
    vre_reports_generated_total = None          # type: ignore[assignment]
    vre_rules_per_set = None                    # type: ignore[assignment]
    vre_evaluation_duration_seconds = None      # type: ignore[assignment]
    vre_processing_duration_seconds = None      # type: ignore[assignment]
    vre_active_rules = None                     # type: ignore[assignment]
    vre_active_rule_sets = None                 # type: ignore[assignment]
    vre_pass_rate = None                        # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_rule_registered(rule_type: str, severity: str) -> None:
    """Record a validation rule registration event.

    Args:
        rule_type: Type of validation rule registered
            (range_check, format_validation, cross_field, regex,
            lookup, threshold, conditional, temporal, referential,
            custom, regulatory, completeness, uniqueness, consistency).
        severity: Severity level of the registered rule
            (critical, error, warning, info, debug).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_rules_registered_total.labels(
        rule_type=rule_type,
        severity=severity,
    ).inc()


def record_rule_set_created(pack_type: str) -> None:
    """Record a rule set creation event.

    Args:
        pack_type: Type or framework association of the rule set
            (ghg_protocol, csrd_esrs, tcfd, cdp, sbti, iso14064,
            eudr, custom, internal, regulatory, compliance, quality).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_rule_sets_created_total.labels(
        pack_type=pack_type,
    ).inc()


def record_evaluation(result: str, rule_type: str) -> None:
    """Record a rule evaluation execution event.

    Args:
        result: Evaluation result outcome
            (pass, fail, warning, error, skipped, not_applicable).
        rule_type: Type of validation rule that was evaluated
            (range_check, format_validation, cross_field, regex,
            lookup, threshold, conditional, temporal, referential,
            custom, regulatory, completeness, uniqueness, consistency).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_evaluations_total.labels(
        result=result,
        rule_type=rule_type,
    ).inc()


def record_evaluation_failure(severity: str) -> None:
    """Record an evaluation failure event by severity.

    Args:
        severity: Severity of the failed validation rule
            (critical, error, warning, info).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_evaluation_failures_total.labels(
        severity=severity,
    ).inc()


def record_conflict_detected(conflict_type: str) -> None:
    """Record a rule conflict detection event.

    Args:
        conflict_type: Type of conflict detected between rules
            (contradiction, overlap, subsumption, redundancy,
            circular_dependency, priority_ambiguity, scope_collision,
            temporal_conflict).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_conflicts_detected_total.labels(
        conflict_type=conflict_type,
    ).inc()


def record_report_generated(report_type: str, format: str) -> None:
    """Record a validation report generation event.

    Args:
        report_type: Type of validation report generated
            (evaluation_summary, compliance_report, conflict_analysis,
            rule_coverage, audit_trail, quality_scorecard,
            trend_analysis, exception_report).
        format: Output format of the generated report
            (json, html, pdf, csv, markdown, xml).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_reports_generated_total.labels(
        report_type=report_type,
        format=format,
    ).inc()


def observe_rules_per_set(count: int) -> None:
    """Record the number of rules in a newly created or updated rule set.

    Args:
        count: Number of rules contained in the rule set.
            Buckets: 1, 5, 10, 25, 50, 100, 250, 500.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_rules_per_set.observe(count)


def observe_evaluation_duration(operation: str, seconds: float) -> None:
    """Record the duration of a rule evaluation operation.

    Args:
        operation: Evaluation operation type label
            (single_rule, rule_set, compound_rule, batch,
            cross_field, conditional, full_pipeline).
        seconds: Evaluation wall-clock time in seconds.
            Buckets: 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_evaluation_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def observe_processing_duration(operation: str, seconds: float) -> None:
    """Record processing duration for a single engine operation.

    Args:
        operation: Operation type label
            (rule_register, rule_update, rule_delete, set_create,
            set_update, conflict_detect, conflict_resolve,
            report_generate, pack_import, pack_export,
            rule_activate, rule_deactivate).
        seconds: Operation duration in seconds.
            Buckets: 0.01, 0.05, 0.1, 0.5, 1, 5, 10.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_processing_duration_seconds.labels(
        operation=operation,
    ).observe(seconds)


def set_active_rules(count: int) -> None:
    """Set the gauge for current number of active validation rules.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of active validation rules currently loaded
            in the engine. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_active_rules.set(count)


def set_active_rule_sets(count: int) -> None:
    """Set the gauge for current number of active rule sets.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of active rule sets currently loaded
            in the engine. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_active_rule_sets.set(count)


def set_pass_rate(rate: float) -> None:
    """Set the gauge for current overall evaluation pass rate.

    This is an absolute set representing the fraction of evaluations
    that returned a ``pass`` result. The caller is responsible for
    computing the correct value from cumulative counters.

    Args:
        rate: Pass rate as a float between 0.0 (all fail) and
            1.0 (all pass). Values outside this range will still
            be accepted by Prometheus but are semantically incorrect.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    vre_pass_rate.set(rate)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "vre_rules_registered_total",
    "vre_rule_sets_created_total",
    "vre_evaluations_total",
    "vre_evaluation_failures_total",
    "vre_conflicts_detected_total",
    "vre_reports_generated_total",
    "vre_rules_per_set",
    "vre_evaluation_duration_seconds",
    "vre_processing_duration_seconds",
    "vre_active_rules",
    "vre_active_rule_sets",
    "vre_pass_rate",
    # Helper functions
    "record_rule_registered",
    "record_rule_set_created",
    "record_evaluation",
    "record_evaluation_failure",
    "record_conflict_detected",
    "record_report_generated",
    "observe_rules_per_set",
    "observe_evaluation_duration",
    "observe_processing_duration",
    "set_active_rules",
    "set_active_rule_sets",
    "set_pass_rate",
]
