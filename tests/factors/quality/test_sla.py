# -*- coding: utf-8 -*-
"""Tests for SLA timer + escalation logic (GAP-15)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.factors.quality.sla import (
    DEFAULT_SLA_POLICIES,
    SLA_SCHEMA_SQL,
    SLAExpiredError,
    SLAPolicy,
    SLAStage,
    SLATimer,
    SLATimerStatus,
    auto_reject_stale,
    check_timers,
    complete_timer,
    ensure_sla_schema,
    escalate_overdue,
    get_policy,
    get_timer_for_factor,
    sla_dashboard_metrics,
    start_sla_timer,
    warn_overdue,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    ensure_sla_schema(c)
    try:
        yield c
    finally:
        c.close()


T0 = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Policy lookups
# ---------------------------------------------------------------------------


def test_community_initial_is_72h():
    p = get_policy(SLAStage.INITIAL_REVIEW, "community")
    assert p.duration_hours == 72
    assert p.warning_at_pct == 0.75


def test_pro_detailed_is_120h():
    p = get_policy(SLAStage.DETAILED_REVIEW, "pro")
    assert p.duration_hours == 120


def test_enterprise_cbam_has_auto_reject():
    p = get_policy(SLAStage.FINAL_APPROVAL, "enterprise_cbam")
    assert p.auto_reject_after_hours == 168


def test_unknown_tier_falls_back_to_community():
    p = get_policy(SLAStage.INITIAL_REVIEW, "not-a-tier")
    community = get_policy(SLAStage.INITIAL_REVIEW, "community")
    assert p.duration_hours == community.duration_hours


def test_policies_exist_for_all_stages_and_tiers():
    tiers = ("community", "pro", "enterprise", "enterprise_cbam")
    for tier in tiers:
        for stage in SLAStage:
            policy = get_policy(stage, tier)
            assert isinstance(policy, SLAPolicy)
            assert policy.duration_hours > 0


# ---------------------------------------------------------------------------
# Timer lifecycle
# ---------------------------------------------------------------------------


def test_start_timer_computes_deadline_and_warning(conn):
    timer = start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    assert timer.deadline == T0 + timedelta(hours=120)
    # Warning at 75% of 120h = 90h.
    assert timer.warning_at == T0 + timedelta(hours=90)
    assert timer.status == SLATimerStatus.ACTIVE


def test_start_timer_persists(conn):
    start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    loaded = get_timer_for_factor(conn, "EF:1", SLAStage.DETAILED_REVIEW)
    assert loaded is not None
    assert loaded.factor_id == "EF:1"
    assert loaded.stage == SLAStage.DETAILED_REVIEW


def test_is_warning_between_warn_and_deadline(conn):
    timer = start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    # 95h in => in warning band (warn=90h, deadline=120h)
    now = T0 + timedelta(hours=95)
    assert timer.is_warning(now)
    assert not timer.is_overdue(now)


def test_is_overdue_past_deadline(conn):
    timer = start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    now = T0 + timedelta(hours=121)
    assert timer.is_overdue(now)


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


def test_warn_overdue_moves_to_warned(conn):
    timer = start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    warned = warn_overdue(conn, timer, now=T0 + timedelta(hours=95))
    assert warned.status == SLATimerStatus.WARNED
    assert warned.escalation_history
    assert warned.escalation_history[-1]["type"] == "warning"


def test_escalate_overdue_increments_level(conn):
    timer = start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    escalated1 = escalate_overdue(conn, timer, now=T0 + timedelta(hours=130))
    assert escalated1.status == SLATimerStatus.ESCALATED
    escalated2 = escalate_overdue(conn, escalated1, now=T0 + timedelta(hours=140))
    levels = [
        e.get("level") for e in escalated2.escalation_history if e.get("type") == "escalation"
    ]
    assert levels == [1, 2]


def test_escalation_capped_at_max_level(conn):
    timer = start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    current = timer
    for i in range(10):
        current = escalate_overdue(conn, current, now=T0 + timedelta(hours=150 + i))
    levels = [
        e["level"] for e in current.escalation_history if e["type"] == "escalation"
    ]
    assert max(levels) <= 4


def test_auto_reject_stale_after_hard_deadline(conn):
    # enterprise_cbam final_approval has auto_reject_after_hours=168.
    timer = start_sla_timer(
        conn, "EF:1", SLAStage.FINAL_APPROVAL, "enterprise_cbam", now=T0
    )
    assert timer.auto_reject_after_hours == 168
    assert timer.should_auto_reject(T0 + timedelta(hours=169))
    expired = auto_reject_stale(conn, timer, now=T0 + timedelta(hours=170))
    assert expired.status == SLATimerStatus.EXPIRED
    assert expired.completed_at is not None


def test_complete_timer_marks_completed(conn):
    start_sla_timer(conn, "EF:1", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    completed = complete_timer(
        conn, "EF:1", SLAStage.DETAILED_REVIEW, now=T0 + timedelta(hours=10)
    )
    assert completed is not None
    assert completed.status == SLATimerStatus.COMPLETED
    assert completed.completed_at is not None


def test_complete_timer_no_op_when_absent(conn):
    result = complete_timer(conn, "EF:missing", SLAStage.DETAILED_REVIEW)
    assert result is None


# ---------------------------------------------------------------------------
# check_timers
# ---------------------------------------------------------------------------


def test_check_timers_returns_warnings(conn):
    start_sla_timer(conn, "EF:warn", SLAStage.INITIAL_REVIEW, "pro", now=T0)
    # Within warning band (pro initial = 48h, warn at 36h).
    now = T0 + timedelta(hours=37)
    actions = check_timers(conn, now=now)
    assert len(actions) == 1
    assert getattr(actions[0], "_action") == "warn"


def test_check_timers_flags_overdue_for_escalation(conn):
    start_sla_timer(conn, "EF:over", SLAStage.INITIAL_REVIEW, "pro", now=T0)
    now = T0 + timedelta(hours=60)  # > 48h deadline
    actions = check_timers(conn, now=now)
    assert any(getattr(t, "_action") == "escalate" for t in actions)


def test_check_timers_flags_auto_reject(conn):
    start_sla_timer(
        conn, "EF:cbam", SLAStage.FINAL_APPROVAL, "enterprise_cbam", now=T0
    )
    now = T0 + timedelta(hours=200)  # > 168h hard cutoff
    actions = check_timers(conn, now=now)
    assert any(getattr(t, "_action") == "auto_reject" for t in actions)


def test_check_timers_skips_completed(conn):
    start_sla_timer(conn, "EF:done", SLAStage.DETAILED_REVIEW, "pro", now=T0)
    complete_timer(conn, "EF:done", SLAStage.DETAILED_REVIEW, now=T0)
    actions = check_timers(conn, now=T0 + timedelta(hours=500))
    assert actions == []


# ---------------------------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------------------------


def test_dashboard_metrics_snapshot(conn):
    start_sla_timer(conn, "EF:1", SLAStage.INITIAL_REVIEW, "pro", now=T0)
    start_sla_timer(conn, "EF:2", SLAStage.INITIAL_REVIEW, "pro", now=T0)
    # Complete one.
    complete_timer(conn, "EF:1", SLAStage.INITIAL_REVIEW, now=T0)
    metrics = sla_dashboard_metrics(
        conn, tenant_id="tenant-a", now=T0 + timedelta(hours=50)
    )
    assert metrics["total_timers"] == 2
    assert metrics["status_counts"]["COMPLETED"] == 1
    # EF:2 is overdue (>48h for pro initial)
    assert metrics["overdue_timers"] == 1
    assert metrics["compliance_pct"] == 100.0


def test_dashboard_metrics_with_no_timers(conn):
    metrics = sla_dashboard_metrics(conn, now=T0)
    assert metrics["total_timers"] == 0
    assert metrics["overdue_timers"] == 0


# ---------------------------------------------------------------------------
# SLAExpiredError
# ---------------------------------------------------------------------------


def test_sla_expired_error_captures_stage_and_deadline():
    err = SLAExpiredError("EF:1", SLAStage.DETAILED_REVIEW.value, T0)
    assert err.factor_id == "EF:1"
    assert err.stage == SLAStage.DETAILED_REVIEW.value
    assert err.expired_at == T0


def test_schema_sql_constant_exists():
    assert "factors_sla_timers" in SLA_SCHEMA_SQL
    assert "factors_sla_policies" in SLA_SCHEMA_SQL


def test_default_policies_dict_contains_entries_per_tier():
    assert len(DEFAULT_SLA_POLICIES) >= 16  # 4 stages * 4 tiers
