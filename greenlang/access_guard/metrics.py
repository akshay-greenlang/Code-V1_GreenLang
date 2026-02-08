# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-FOUND-006: Access & Policy Guard

12 Prometheus metrics for access guard monitoring with graceful
fallback when prometheus_client is not installed.

Metrics:
    1.  gl_access_guard_decisions_total (Counter)
    2.  gl_access_guard_decision_duration_seconds (Histogram)
    3.  gl_access_guard_denials_total (Counter)
    4.  gl_access_guard_rate_limits_total (Counter)
    5.  gl_access_guard_policy_evaluations_total (Counter)
    6.  gl_access_guard_tenant_violations_total (Counter)
    7.  gl_access_guard_classification_checks_total (Counter)
    8.  gl_access_guard_policies_total (Gauge)
    9.  gl_access_guard_rules_total (Gauge)
    10. gl_access_guard_cache_hits_total (Counter)
    11. gl_access_guard_cache_misses_total (Counter)
    12. gl_access_guard_audit_events_total (Gauge)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
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
        "prometheus_client not installed; access guard metrics disabled"
    )


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Decisions count
    access_guard_decisions_total = Counter(
        "gl_access_guard_decisions_total",
        "Total access guard decisions rendered",
        labelnames=["action", "result"],
    )

    # 2. Decision duration
    access_guard_decision_duration_seconds = Histogram(
        "gl_access_guard_decision_duration_seconds",
        "Access guard decision duration in seconds",
        labelnames=["action"],
        buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )

    # 3. Denials by reason
    access_guard_denials_total = Counter(
        "gl_access_guard_denials_total",
        "Total access guard denials by reason",
        labelnames=["reason"],
    )

    # 4. Rate limits by tenant
    access_guard_rate_limits_total = Counter(
        "gl_access_guard_rate_limits_total",
        "Total rate limit triggers by tenant",
        labelnames=["tenant"],
    )

    # 5. Policy evaluations
    access_guard_policy_evaluations_total = Counter(
        "gl_access_guard_policy_evaluations_total",
        "Total policy evaluations by policy ID",
        labelnames=["policy_id"],
    )

    # 6. Tenant violations
    access_guard_tenant_violations_total = Counter(
        "gl_access_guard_tenant_violations_total",
        "Total tenant boundary violations",
    )

    # 7. Classification checks
    access_guard_classification_checks_total = Counter(
        "gl_access_guard_classification_checks_total",
        "Total data classification checks by level",
        labelnames=["level"],
    )

    # 8. Policies gauge
    access_guard_policies_total = Gauge(
        "gl_access_guard_policies_total",
        "Current number of loaded policies",
    )

    # 9. Rules gauge
    access_guard_rules_total = Gauge(
        "gl_access_guard_rules_total",
        "Current number of active rules",
    )

    # 10. Cache hits
    access_guard_cache_hits_total = Counter(
        "gl_access_guard_cache_hits_total",
        "Total decision cache hits",
    )

    # 11. Cache misses
    access_guard_cache_misses_total = Counter(
        "gl_access_guard_cache_misses_total",
        "Total decision cache misses",
    )

    # 12. Audit events gauge
    access_guard_audit_events_total = Gauge(
        "gl_access_guard_audit_events_total",
        "Current number of audit events stored",
    )

else:
    # No-op placeholders
    access_guard_decisions_total = None  # type: ignore[assignment]
    access_guard_decision_duration_seconds = None  # type: ignore[assignment]
    access_guard_denials_total = None  # type: ignore[assignment]
    access_guard_rate_limits_total = None  # type: ignore[assignment]
    access_guard_policy_evaluations_total = None  # type: ignore[assignment]
    access_guard_tenant_violations_total = None  # type: ignore[assignment]
    access_guard_classification_checks_total = None  # type: ignore[assignment]
    access_guard_policies_total = None  # type: ignore[assignment]
    access_guard_rules_total = None  # type: ignore[assignment]
    access_guard_cache_hits_total = None  # type: ignore[assignment]
    access_guard_cache_misses_total = None  # type: ignore[assignment]
    access_guard_audit_events_total = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_decision(action: str, result: str, duration_seconds: float) -> None:
    """Record an access decision.

    Args:
        action: Action that was evaluated (read, write, etc.).
        result: Decision result ("allow", "deny", "conditional").
        duration_seconds: Decision duration in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_decisions_total.labels(action=action, result=result).inc()
    access_guard_decision_duration_seconds.labels(action=action).observe(
        duration_seconds,
    )


def record_denial(reason: str) -> None:
    """Record a denial by reason category.

    Args:
        reason: Denial reason category.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_denials_total.labels(reason=reason).inc()


def record_rate_limit(tenant: str) -> None:
    """Record a rate limit trigger for a tenant.

    Args:
        tenant: Tenant identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_rate_limits_total.labels(tenant=tenant).inc()


def record_policy_evaluation(policy_id: str) -> None:
    """Record a policy evaluation.

    Args:
        policy_id: Policy identifier.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_policy_evaluations_total.labels(policy_id=policy_id).inc()


def record_tenant_violation() -> None:
    """Record a tenant boundary violation."""
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_tenant_violations_total.inc()


def record_classification_check(level: str) -> None:
    """Record a classification check by level.

    Args:
        level: Classification level checked.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_classification_checks_total.labels(level=level).inc()


def update_policies_count(count: int) -> None:
    """Set the policies gauge.

    Args:
        count: Current number of policies.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_policies_total.set(count)


def update_rules_count(count: int) -> None:
    """Set the rules gauge.

    Args:
        count: Current number of rules.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_rules_total.set(count)


def record_cache_hit() -> None:
    """Record a decision cache hit."""
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_cache_hits_total.inc()


def record_cache_miss() -> None:
    """Record a decision cache miss."""
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_cache_misses_total.inc()


def update_audit_events_count(count: int) -> None:
    """Set the audit events gauge.

    Args:
        count: Current number of audit events.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    access_guard_audit_events_total.set(count)


__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "access_guard_decisions_total",
    "access_guard_decision_duration_seconds",
    "access_guard_denials_total",
    "access_guard_rate_limits_total",
    "access_guard_policy_evaluations_total",
    "access_guard_tenant_violations_total",
    "access_guard_classification_checks_total",
    "access_guard_policies_total",
    "access_guard_rules_total",
    "access_guard_cache_hits_total",
    "access_guard_cache_misses_total",
    "access_guard_audit_events_total",
    # Helper functions
    "record_decision",
    "record_denial",
    "record_rate_limit",
    "record_policy_evaluation",
    "record_tenant_violation",
    "record_classification_check",
    "update_policies_count",
    "update_rules_count",
    "record_cache_hit",
    "record_cache_miss",
    "update_audit_events_count",
]
