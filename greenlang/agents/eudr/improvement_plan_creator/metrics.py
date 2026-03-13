# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-035: Improvement Plan Creator

40+ Prometheus metrics for improvement plan creation service observability
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_ipc_`` prefix for consistent
identification in Prometheus queries, Grafana dashboards, and alerting
rules across the GreenLang platform.

Metrics (42 per PRD Section 7.6):
    Counters (16):
        1.  gl_eudr_ipc_plans_created_total                  - Plans created [status]
        2.  gl_eudr_ipc_findings_aggregated_total             - Findings aggregated [source]
        3.  gl_eudr_ipc_gaps_identified_total                 - Gaps identified [severity]
        4.  gl_eudr_ipc_actions_generated_total               - Actions generated [type]
        5.  gl_eudr_ipc_root_causes_mapped_total              - Root causes mapped [category]
        6.  gl_eudr_ipc_actions_prioritized_total             - Actions prioritized [quadrant]
        7.  gl_eudr_ipc_progress_snapshots_total              - Progress snapshots captured
        8.  gl_eudr_ipc_stakeholders_assigned_total           - Stakeholders assigned [role]
        9.  gl_eudr_ipc_notifications_sent_total              - Notifications sent [channel]
        10. gl_eudr_ipc_actions_completed_total               - Actions completed
        11. gl_eudr_ipc_actions_verified_total                - Actions verified
        12. gl_eudr_ipc_escalations_triggered_total           - Escalations triggered [level]
        13. gl_eudr_ipc_plans_approved_total                  - Plans approved
        14. gl_eudr_ipc_reports_generated_total               - Reports generated [format]
        15. gl_eudr_ipc_duplicates_removed_total              - Duplicate findings removed
        16. gl_eudr_ipc_smart_validations_total               - SMART validations [result]

    Histograms (14):
        17. gl_eudr_ipc_finding_aggregation_duration_seconds  - Finding aggregation latency
        18. gl_eudr_ipc_gap_analysis_duration_seconds         - Gap analysis latency
        19. gl_eudr_ipc_action_generation_duration_seconds    - Action generation latency
        20. gl_eudr_ipc_root_cause_mapping_duration_seconds   - Root cause mapping latency
        21. gl_eudr_ipc_prioritization_duration_seconds       - Prioritization latency
        22. gl_eudr_ipc_progress_tracking_duration_seconds    - Progress tracking latency
        23. gl_eudr_ipc_stakeholder_coord_duration_seconds    - Stakeholder coordination latency
        24. gl_eudr_ipc_plan_creation_duration_seconds        - End-to-end plan creation latency
        25. gl_eudr_ipc_report_generation_duration_seconds    - Report generation latency
        26. gl_eudr_ipc_notification_dispatch_duration_seconds - Notification dispatch latency
        27. gl_eudr_ipc_fishbone_analysis_duration_seconds    - Fishbone analysis latency
        28. gl_eudr_ipc_five_whys_duration_seconds            - 5-Whys analysis latency
        29. gl_eudr_ipc_effectiveness_review_duration_seconds - Effectiveness review latency
        30. gl_eudr_ipc_raci_validation_duration_seconds      - RACI validation latency

    Gauges (12):
        31. gl_eudr_ipc_active_plans                         - Active plan count
        32. gl_eudr_ipc_pending_actions                      - Pending action count
        33. gl_eudr_ipc_overdue_actions                      - Overdue action count
        34. gl_eudr_ipc_critical_gaps_open                   - Open critical gaps count
        35. gl_eudr_ipc_high_gaps_open                       - Open high-severity gaps
        36. gl_eudr_ipc_overall_progress_percent             - Overall progress gauge (0-100)
        37. gl_eudr_ipc_avg_effectiveness_score              - Average effectiveness score
        38. gl_eudr_ipc_stakeholders_pending_ack             - Stakeholders pending acknowledgment
        39. gl_eudr_ipc_on_track_plans                       - Number of plans on track
        40. gl_eudr_ipc_off_track_plans                      - Number of plans off track
        41. gl_eudr_ipc_actions_on_hold                      - Actions on hold count
        42. gl_eudr_ipc_systemic_root_causes                 - Systemic root causes count

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.debug("prometheus_client not available; metrics disabled")


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    # Counters (16)
    _PLANS_CREATED = Counter(
        "gl_eudr_ipc_plans_created_total",
        "Improvement plans created",
        ["status"],
    )
    _FINDINGS_AGGREGATED = Counter(
        "gl_eudr_ipc_findings_aggregated_total",
        "Findings aggregated from upstream agents",
        ["source"],
    )
    _GAPS_IDENTIFIED = Counter(
        "gl_eudr_ipc_gaps_identified_total",
        "Compliance gaps identified",
        ["severity"],
    )
    _ACTIONS_GENERATED = Counter(
        "gl_eudr_ipc_actions_generated_total",
        "Improvement actions generated",
        ["type"],
    )
    _ROOT_CAUSES_MAPPED = Counter(
        "gl_eudr_ipc_root_causes_mapped_total",
        "Root causes mapped via 5-Whys or fishbone",
        ["category"],
    )
    _ACTIONS_PRIORITIZED = Counter(
        "gl_eudr_ipc_actions_prioritized_total",
        "Actions prioritized via Eisenhower matrix",
        ["quadrant"],
    )
    _PROGRESS_SNAPSHOTS = Counter(
        "gl_eudr_ipc_progress_snapshots_total",
        "Progress snapshots captured",
    )
    _STAKEHOLDERS_ASSIGNED = Counter(
        "gl_eudr_ipc_stakeholders_assigned_total",
        "Stakeholders assigned via RACI",
        ["role"],
    )
    _NOTIFICATIONS_SENT = Counter(
        "gl_eudr_ipc_notifications_sent_total",
        "Notifications sent to stakeholders",
        ["channel"],
    )
    _ACTIONS_COMPLETED = Counter(
        "gl_eudr_ipc_actions_completed_total",
        "Actions marked as completed",
    )
    _ACTIONS_VERIFIED = Counter(
        "gl_eudr_ipc_actions_verified_total",
        "Actions verified for effectiveness",
    )
    _ESCALATIONS_TRIGGERED = Counter(
        "gl_eudr_ipc_escalations_triggered_total",
        "Escalations triggered for overdue actions",
        ["level"],
    )
    _PLANS_APPROVED = Counter(
        "gl_eudr_ipc_plans_approved_total",
        "Plans approved for execution",
    )
    _REPORTS_GENERATED = Counter(
        "gl_eudr_ipc_reports_generated_total",
        "Reports generated",
        ["format"],
    )
    _DUPLICATES_REMOVED = Counter(
        "gl_eudr_ipc_duplicates_removed_total",
        "Duplicate findings removed during aggregation",
    )
    _SMART_VALIDATIONS = Counter(
        "gl_eudr_ipc_smart_validations_total",
        "SMART criteria validations performed",
        ["result"],
    )

    # Histograms (14)
    _FINDING_AGGREGATION_DURATION = Histogram(
        "gl_eudr_ipc_finding_aggregation_duration_seconds",
        "Finding aggregation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _GAP_ANALYSIS_DURATION = Histogram(
        "gl_eudr_ipc_gap_analysis_duration_seconds",
        "Gap analysis latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _ACTION_GENERATION_DURATION = Histogram(
        "gl_eudr_ipc_action_generation_duration_seconds",
        "Action generation latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _ROOT_CAUSE_MAPPING_DURATION = Histogram(
        "gl_eudr_ipc_root_cause_mapping_duration_seconds",
        "Root cause mapping latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _PRIORITIZATION_DURATION = Histogram(
        "gl_eudr_ipc_prioritization_duration_seconds",
        "Prioritization engine latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _PROGRESS_TRACKING_DURATION = Histogram(
        "gl_eudr_ipc_progress_tracking_duration_seconds",
        "Progress tracking latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _STAKEHOLDER_COORD_DURATION = Histogram(
        "gl_eudr_ipc_stakeholder_coord_duration_seconds",
        "Stakeholder coordination latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _PLAN_CREATION_DURATION = Histogram(
        "gl_eudr_ipc_plan_creation_duration_seconds",
        "End-to-end plan creation latency",
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _REPORT_GENERATION_DURATION = Histogram(
        "gl_eudr_ipc_report_generation_duration_seconds",
        "Report generation latency",
        buckets=(0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    _NOTIFICATION_DISPATCH_DURATION = Histogram(
        "gl_eudr_ipc_notification_dispatch_duration_seconds",
        "Notification dispatch latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _FISHBONE_ANALYSIS_DURATION = Histogram(
        "gl_eudr_ipc_fishbone_analysis_duration_seconds",
        "Fishbone analysis latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _FIVE_WHYS_DURATION = Histogram(
        "gl_eudr_ipc_five_whys_duration_seconds",
        "5-Whys analysis latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    _EFFECTIVENESS_REVIEW_DURATION = Histogram(
        "gl_eudr_ipc_effectiveness_review_duration_seconds",
        "Effectiveness review latency",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _RACI_VALIDATION_DURATION = Histogram(
        "gl_eudr_ipc_raci_validation_duration_seconds",
        "RACI validation latency",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    # Gauges (12)
    _ACTIVE_PLANS = Gauge(
        "gl_eudr_ipc_active_plans",
        "Number of active improvement plans",
    )
    _PENDING_ACTIONS = Gauge(
        "gl_eudr_ipc_pending_actions",
        "Number of pending actions across all plans",
    )
    _OVERDUE_ACTIONS = Gauge(
        "gl_eudr_ipc_overdue_actions",
        "Number of overdue actions",
    )
    _CRITICAL_GAPS_OPEN = Gauge(
        "gl_eudr_ipc_critical_gaps_open",
        "Number of open critical-severity gaps",
    )
    _HIGH_GAPS_OPEN = Gauge(
        "gl_eudr_ipc_high_gaps_open",
        "Number of open high-severity gaps",
    )
    _OVERALL_PROGRESS = Gauge(
        "gl_eudr_ipc_overall_progress_percent",
        "Overall progress percentage (0-100)",
    )
    _AVG_EFFECTIVENESS = Gauge(
        "gl_eudr_ipc_avg_effectiveness_score",
        "Average effectiveness score across verified actions",
    )
    _STAKEHOLDERS_PENDING_ACK = Gauge(
        "gl_eudr_ipc_stakeholders_pending_ack",
        "Number of stakeholders pending acknowledgment",
    )
    _ON_TRACK_PLANS = Gauge(
        "gl_eudr_ipc_on_track_plans",
        "Number of plans currently on track",
    )
    _OFF_TRACK_PLANS = Gauge(
        "gl_eudr_ipc_off_track_plans",
        "Number of plans currently off track",
    )
    _ACTIONS_ON_HOLD = Gauge(
        "gl_eudr_ipc_actions_on_hold",
        "Number of actions currently on hold",
    )
    _SYSTEMIC_ROOT_CAUSES = Gauge(
        "gl_eudr_ipc_systemic_root_causes",
        "Number of systemic (cross-cutting) root causes",
    )


# ---------------------------------------------------------------------------
# Helper Functions - Counters
# ---------------------------------------------------------------------------


def record_plan_created(status: str = "draft") -> None:
    """Record a plan creation metric."""
    if _PROMETHEUS_AVAILABLE:
        _PLANS_CREATED.labels(status=status).inc()


def record_finding_aggregated(source: str) -> None:
    """Record a finding aggregation metric."""
    if _PROMETHEUS_AVAILABLE:
        _FINDINGS_AGGREGATED.labels(source=source).inc()


def record_gap_identified(severity: str) -> None:
    """Record a gap identification metric."""
    if _PROMETHEUS_AVAILABLE:
        _GAPS_IDENTIFIED.labels(severity=severity).inc()


def record_action_generated(action_type: str) -> None:
    """Record an action generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIONS_GENERATED.labels(type=action_type).inc()


def record_root_cause_mapped(category: str) -> None:
    """Record a root cause mapping metric."""
    if _PROMETHEUS_AVAILABLE:
        _ROOT_CAUSES_MAPPED.labels(category=category).inc()


def record_action_prioritized(quadrant: str) -> None:
    """Record an action prioritization metric."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIONS_PRIORITIZED.labels(quadrant=quadrant).inc()


def record_progress_snapshot() -> None:
    """Record a progress snapshot metric."""
    if _PROMETHEUS_AVAILABLE:
        _PROGRESS_SNAPSHOTS.inc()


def record_stakeholder_assigned(role: str) -> None:
    """Record a stakeholder assignment metric."""
    if _PROMETHEUS_AVAILABLE:
        _STAKEHOLDERS_ASSIGNED.labels(role=role).inc()


def record_notification_sent(channel: str) -> None:
    """Record a notification sent metric."""
    if _PROMETHEUS_AVAILABLE:
        _NOTIFICATIONS_SENT.labels(channel=channel).inc()


def record_action_completed() -> None:
    """Record an action completion metric."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIONS_COMPLETED.inc()


def record_action_verified() -> None:
    """Record an action verification metric."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIONS_VERIFIED.inc()


def record_escalation_triggered(level: str) -> None:
    """Record an escalation trigger metric."""
    if _PROMETHEUS_AVAILABLE:
        _ESCALATIONS_TRIGGERED.labels(level=level).inc()


def record_plan_approved() -> None:
    """Record a plan approval metric."""
    if _PROMETHEUS_AVAILABLE:
        _PLANS_APPROVED.inc()


def record_report_generated(report_format: str = "json") -> None:
    """Record a report generation metric."""
    if _PROMETHEUS_AVAILABLE:
        _REPORTS_GENERATED.labels(format=report_format).inc()


def record_duplicates_removed(count: int = 1) -> None:
    """Record duplicate findings removed."""
    if _PROMETHEUS_AVAILABLE:
        _DUPLICATES_REMOVED.inc(count)


def record_smart_validation(result: str) -> None:
    """Record a SMART validation metric."""
    if _PROMETHEUS_AVAILABLE:
        _SMART_VALIDATIONS.labels(result=result).inc()


# ---------------------------------------------------------------------------
# Helper Functions - Histograms
# ---------------------------------------------------------------------------


def observe_finding_aggregation_duration(duration: float) -> None:
    """Observe finding aggregation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _FINDING_AGGREGATION_DURATION.observe(duration)


def observe_gap_analysis_duration(duration: float) -> None:
    """Observe gap analysis duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _GAP_ANALYSIS_DURATION.observe(duration)


def observe_action_generation_duration(duration: float) -> None:
    """Observe action generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ACTION_GENERATION_DURATION.observe(duration)


def observe_root_cause_mapping_duration(duration: float) -> None:
    """Observe root cause mapping duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _ROOT_CAUSE_MAPPING_DURATION.observe(duration)


def observe_prioritization_duration(duration: float) -> None:
    """Observe prioritization engine duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _PRIORITIZATION_DURATION.observe(duration)


def observe_progress_tracking_duration(duration: float) -> None:
    """Observe progress tracking duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _PROGRESS_TRACKING_DURATION.observe(duration)


def observe_stakeholder_coord_duration(duration: float) -> None:
    """Observe stakeholder coordination duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _STAKEHOLDER_COORD_DURATION.observe(duration)


def observe_plan_creation_duration(duration: float) -> None:
    """Observe end-to-end plan creation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _PLAN_CREATION_DURATION.observe(duration)


def observe_report_generation_duration(duration: float) -> None:
    """Observe report generation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _REPORT_GENERATION_DURATION.observe(duration)


def observe_notification_dispatch_duration(duration: float) -> None:
    """Observe notification dispatch duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _NOTIFICATION_DISPATCH_DURATION.observe(duration)


def observe_fishbone_analysis_duration(duration: float) -> None:
    """Observe fishbone analysis duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _FISHBONE_ANALYSIS_DURATION.observe(duration)


def observe_five_whys_duration(duration: float) -> None:
    """Observe 5-Whys analysis duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _FIVE_WHYS_DURATION.observe(duration)


def observe_effectiveness_review_duration(duration: float) -> None:
    """Observe effectiveness review duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _EFFECTIVENESS_REVIEW_DURATION.observe(duration)


def observe_raci_validation_duration(duration: float) -> None:
    """Observe RACI validation duration in seconds."""
    if _PROMETHEUS_AVAILABLE:
        _RACI_VALIDATION_DURATION.observe(duration)


# ---------------------------------------------------------------------------
# Helper Functions - Gauges
# ---------------------------------------------------------------------------


def set_active_plans(count: int) -> None:
    """Set gauge of active improvement plans."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIVE_PLANS.set(count)


def set_pending_actions(count: int) -> None:
    """Set gauge of pending actions."""
    if _PROMETHEUS_AVAILABLE:
        _PENDING_ACTIONS.set(count)


def set_overdue_actions(count: int) -> None:
    """Set gauge of overdue actions."""
    if _PROMETHEUS_AVAILABLE:
        _OVERDUE_ACTIONS.set(count)


def set_critical_gaps_open(count: int) -> None:
    """Set gauge of open critical-severity gaps."""
    if _PROMETHEUS_AVAILABLE:
        _CRITICAL_GAPS_OPEN.set(count)


def set_high_gaps_open(count: int) -> None:
    """Set gauge of open high-severity gaps."""
    if _PROMETHEUS_AVAILABLE:
        _HIGH_GAPS_OPEN.set(count)


def set_overall_progress(percent: float) -> None:
    """Set gauge of overall progress percentage (0-100)."""
    if _PROMETHEUS_AVAILABLE:
        _OVERALL_PROGRESS.set(percent)


def set_avg_effectiveness(score: float) -> None:
    """Set gauge of average effectiveness score."""
    if _PROMETHEUS_AVAILABLE:
        _AVG_EFFECTIVENESS.set(score)


def set_stakeholders_pending_ack(count: int) -> None:
    """Set gauge of stakeholders pending acknowledgment."""
    if _PROMETHEUS_AVAILABLE:
        _STAKEHOLDERS_PENDING_ACK.set(count)


def set_on_track_plans(count: int) -> None:
    """Set gauge of plans on track."""
    if _PROMETHEUS_AVAILABLE:
        _ON_TRACK_PLANS.set(count)


def set_off_track_plans(count: int) -> None:
    """Set gauge of plans off track."""
    if _PROMETHEUS_AVAILABLE:
        _OFF_TRACK_PLANS.set(count)


def set_actions_on_hold(count: int) -> None:
    """Set gauge of actions on hold."""
    if _PROMETHEUS_AVAILABLE:
        _ACTIONS_ON_HOLD.set(count)


def set_systemic_root_causes(count: int) -> None:
    """Set gauge of systemic root causes."""
    if _PROMETHEUS_AVAILABLE:
        _SYSTEMIC_ROOT_CAUSES.set(count)
