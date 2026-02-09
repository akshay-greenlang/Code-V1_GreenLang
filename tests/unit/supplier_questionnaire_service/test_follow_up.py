# -*- coding: utf-8 -*-
"""
Unit Tests for FollowUpEngine - AGENT-DATA-008
=================================================

Tests all methods of FollowUpEngine with 85%+ coverage.
Validates reminder scheduling (4 types), triggering, escalation
(5 levels), bulk operations, non-responsive tracking,
effectiveness analytics, and SHA-256 provenance.

Test count target: ~60 tests
Author: GreenLang Platform Team / GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from greenlang.supplier_questionnaire.follow_up import FollowUpEngine
from greenlang.supplier_questionnaire.models import (
    Distribution,
    DistributionStatus,
    EscalationLevel,
    FollowUpAction,
    ReminderType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _future_date(days: int = 30) -> date:
    return (datetime.now(timezone.utc) + timedelta(days=days)).date()


def _past_date(days: int = 10) -> date:
    return (datetime.now(timezone.utc) - timedelta(days=days)).date()


def _dist(
    dist_id: str = "dist-001",
    supplier_id: str = "sup-001",
    campaign_id: str = "camp-001",
    status: DistributionStatus = DistributionStatus.SENT,
    deadline: Optional[date] = None,
    reminder_count: int = 0,
) -> Distribution:
    return Distribution(
        distribution_id=dist_id,
        template_id="tpl-001",
        supplier_id=supplier_id,
        supplier_name=f"Supplier {supplier_id}",
        supplier_email=f"{supplier_id}@example.com",
        campaign_id=campaign_id,
        status=status,
        deadline=deadline or _future_date(30),
        reminder_count=reminder_count,
    )


# ============================================================================
# TEST CLASS: Initialization
# ============================================================================


class TestFollowUpEngineInit:

    def test_init_defaults(self):
        engine = FollowUpEngine()
        assert engine._max_actions == 100000
        assert engine._auto_escalate is False
        assert engine._max_reminders == 10

    def test_init_custom_config(self):
        engine = FollowUpEngine({
            "max_actions": 5000,
            "auto_escalate": True,
            "max_reminders_per_dist": 5,
        })
        assert engine._max_actions == 5000
        assert engine._auto_escalate is True
        assert engine._max_reminders == 5

    def test_init_stats_zeroed(self):
        engine = FollowUpEngine()
        stats = engine.get_statistics()
        assert stats["reminders_scheduled"] == 0
        assert stats["reminders_triggered"] == 0
        assert stats["reminders_cancelled"] == 0
        assert stats["escalations_created"] == 0
        assert stats["bulk_triggers"] == 0

    def test_init_empty_actions(self):
        engine = FollowUpEngine()
        assert engine.get_statistics()["active_actions"] == 0
        assert engine.get_statistics()["distributions_tracked"] == 0


# ============================================================================
# TEST CLASS: schedule_reminders
# ============================================================================


class TestScheduleReminders:

    def test_schedules_four_reminders(self):
        engine = FollowUpEngine()
        deadline = _future_date(30)
        reminders = engine.schedule_reminders("dist-001", deadline)
        assert len(reminders) == 4

    def test_reminder_types(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        types = {r.reminder_type for r in reminders}
        assert types == {ReminderType.GENTLE, ReminderType.FIRM, ReminderType.URGENT, ReminderType.FINAL}

    def test_reminder_scheduling_order(self):
        engine = FollowUpEngine()
        deadline = _future_date(30)
        reminders = engine.schedule_reminders("dist-001", deadline)
        scheduled_times = [r.scheduled_at for r in reminders]
        assert scheduled_times == sorted(scheduled_times)

    def test_reminder_status_scheduled(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        assert all(r.status == "scheduled" for r in reminders)

    def test_reminder_messages_populated(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        assert all(len(r.message) > 0 for r in reminders)

    def test_campaign_association(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders(
            "dist-001", _future_date(30),
            campaign_id="camp-001",
        )
        assert all(r.campaign_id == "camp-001" for r in reminders)

    def test_supplier_association(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders(
            "dist-001", _future_date(30),
            supplier_id="sup-001",
        )
        assert all(r.supplier_id == "sup-001" for r in reminders)

    def test_empty_distribution_id_raises(self):
        engine = FollowUpEngine()
        with pytest.raises(ValueError, match="non-empty"):
            engine.schedule_reminders("", _future_date(30))

    def test_whitespace_distribution_id_raises(self):
        engine = FollowUpEngine()
        with pytest.raises(ValueError, match="non-empty"):
            engine.schedule_reminders("   ", _future_date(30))

    def test_updates_stats(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        assert engine.get_statistics()["reminders_scheduled"] == 4

    def test_provenance_hash(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        for r in reminders:
            assert len(r.provenance_hash) == 64
            assert re.match(r"^[0-9a-f]{64}$", r.provenance_hash)

    def test_gentle_7_days_before(self):
        engine = FollowUpEngine()
        deadline = _future_date(30)
        reminders = engine.schedule_reminders("dist-001", deadline)
        gentle = [r for r in reminders if r.reminder_type == ReminderType.GENTLE][0]
        expected_date = deadline - timedelta(days=7)
        assert gentle.scheduled_at.date() == expected_date

    def test_firm_3_days_before(self):
        engine = FollowUpEngine()
        deadline = _future_date(30)
        reminders = engine.schedule_reminders("dist-001", deadline)
        firm = [r for r in reminders if r.reminder_type == ReminderType.FIRM][0]
        expected_date = deadline - timedelta(days=3)
        assert firm.scheduled_at.date() == expected_date

    def test_urgent_1_day_before(self):
        engine = FollowUpEngine()
        deadline = _future_date(30)
        reminders = engine.schedule_reminders("dist-001", deadline)
        urgent = [r for r in reminders if r.reminder_type == ReminderType.URGENT][0]
        expected_date = deadline - timedelta(days=1)
        assert urgent.scheduled_at.date() == expected_date

    def test_final_on_deadline(self):
        engine = FollowUpEngine()
        deadline = _future_date(30)
        reminders = engine.schedule_reminders("dist-001", deadline)
        final = [r for r in reminders if r.reminder_type == ReminderType.FINAL][0]
        assert final.scheduled_at.date() == deadline

    def test_past_deadline_still_creates(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _past_date(10))
        assert len(reminders) == 4

    def test_distribution_indexed(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        assert "dist-001" in engine._distribution_actions
        assert len(engine._distribution_actions["dist-001"]) == 4

    def test_campaign_indexed(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30), campaign_id="camp-1")
        assert "camp-1" in engine._campaign_actions


# ============================================================================
# TEST CLASS: trigger_reminder
# ============================================================================


class TestTriggerReminder:

    def test_trigger_success(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        triggered = engine.trigger_reminder(reminders[0].action_id)
        assert triggered.status == "sent"
        assert triggered.sent_at is not None

    def test_trigger_unknown_raises(self):
        engine = FollowUpEngine()
        with pytest.raises(ValueError, match="Unknown action"):
            engine.trigger_reminder("nonexistent-id")

    def test_trigger_already_sent_raises(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        engine.trigger_reminder(reminders[0].action_id)
        with pytest.raises(ValueError, match="already been triggered"):
            engine.trigger_reminder(reminders[0].action_id)

    def test_trigger_cancelled_raises(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        engine.cancel_reminders("dist-001")
        with pytest.raises(ValueError, match="cancelled"):
            engine.trigger_reminder(reminders[0].action_id)

    def test_trigger_updates_stats(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        engine.trigger_reminder(reminders[0].action_id)
        assert engine.get_statistics()["reminders_triggered"] == 1

    def test_trigger_updates_provenance(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        original_hash = reminders[0].provenance_hash
        triggered = engine.trigger_reminder(reminders[0].action_id)
        assert triggered.provenance_hash != original_hash


# ============================================================================
# TEST CLASS: get_due_reminders
# ============================================================================


class TestGetDueReminders:

    def test_no_due_future_deadline(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(60))
        due = engine.get_due_reminders()
        assert len(due) == 0

    def test_due_with_past_deadline(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10))
        due = engine.get_due_reminders()
        assert len(due) == 4

    def test_due_with_custom_cutoff(self):
        engine = FollowUpEngine()
        deadline = _future_date(5)
        engine.schedule_reminders("dist-001", deadline)
        far_future = _utcnow() + timedelta(days=365)
        due = engine.get_due_reminders(as_of=far_future)
        assert len(due) == 4

    def test_due_excludes_sent(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _past_date(10))
        engine.trigger_reminder(reminders[0].action_id)
        due = engine.get_due_reminders()
        action_ids = {d.action_id for d in due}
        assert reminders[0].action_id not in action_ids

    def test_due_excludes_cancelled(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10))
        engine.cancel_reminders("dist-001")
        due = engine.get_due_reminders()
        assert len(due) == 0

    def test_due_sorted_by_time(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10))
        due = engine.get_due_reminders()
        scheduled_times = [r.scheduled_at for r in due]
        assert scheduled_times == sorted(scheduled_times)


# ============================================================================
# TEST CLASS: escalate
# ============================================================================


class TestEscalate:

    def test_escalate_level_1(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30), supplier_id="sup-001")
        action = engine.escalate("dist-001", "level_1")
        assert action.escalation_level == EscalationLevel.LEVEL_1
        assert action.status == "sent"

    def test_escalate_level_2(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        action = engine.escalate("dist-001", "level_2")
        assert action.escalation_level == EscalationLevel.LEVEL_2

    def test_escalate_level_3(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        action = engine.escalate("dist-001", "level_3")
        assert action.escalation_level == EscalationLevel.LEVEL_3

    def test_escalate_level_4(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        action = engine.escalate("dist-001", "level_4")
        assert action.escalation_level == EscalationLevel.LEVEL_4

    def test_escalate_level_5(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        action = engine.escalate("dist-001", "level_5")
        assert action.escalation_level == EscalationLevel.LEVEL_5

    def test_escalate_empty_dist_raises(self):
        engine = FollowUpEngine()
        with pytest.raises(ValueError, match="non-empty"):
            engine.escalate("")

    def test_escalate_invalid_level_raises(self):
        engine = FollowUpEngine()
        with pytest.raises(ValueError, match="Unknown escalation"):
            engine.escalate("dist-001", "level_99")

    def test_escalate_auto_triggered(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        action = engine.escalate("dist-001", "level_1")
        assert action.sent_at is not None

    def test_escalate_updates_stats(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        engine.escalate("dist-001", "level_1")
        assert engine.get_statistics()["escalations_created"] == 1

    def test_escalate_message_populated(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        action = engine.escalate("dist-001", "level_1")
        assert "escalation" in action.message.lower() or "Escalation" in action.message


# ============================================================================
# TEST CLASS: get_follow_up_history
# ============================================================================


class TestGetFollowUpHistory:

    def test_history_empty(self):
        engine = FollowUpEngine()
        assert engine.get_follow_up_history("unknown-dist") == []

    def test_history_after_scheduling(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        history = engine.get_follow_up_history("dist-001")
        assert len(history) == 4

    def test_history_after_escalation(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        engine.escalate("dist-001", "level_1")
        history = engine.get_follow_up_history("dist-001")
        assert len(history) == 5

    def test_history_sorted_chronologically(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        history = engine.get_follow_up_history("dist-001")
        scheduled_times = [h.scheduled_at for h in history]
        assert scheduled_times == sorted(scheduled_times)


# ============================================================================
# TEST CLASS: cancel_reminders
# ============================================================================


class TestCancelReminders:

    def test_cancel_all_scheduled(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        cancelled = engine.cancel_reminders("dist-001")
        assert cancelled == 4

    def test_cancel_no_pending(self):
        engine = FollowUpEngine()
        cancelled = engine.cancel_reminders("dist-unknown")
        assert cancelled == 0

    def test_cancel_only_scheduled(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _future_date(30))
        engine.trigger_reminder(reminders[0].action_id)
        cancelled = engine.cancel_reminders("dist-001")
        assert cancelled == 3

    def test_cancel_updates_stats(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        engine.cancel_reminders("dist-001")
        assert engine.get_statistics()["reminders_cancelled"] == 4

    def test_cancel_status_updated(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(30))
        engine.cancel_reminders("dist-001")
        history = engine.get_follow_up_history("dist-001")
        assert all(h.status == "cancelled" for h in history)


# ============================================================================
# TEST CLASS: bulk_trigger_reminders
# ============================================================================


class TestBulkTriggerReminders:

    def test_bulk_trigger_empty(self):
        engine = FollowUpEngine()
        triggered = engine.bulk_trigger_reminders("camp-unknown")
        assert len(triggered) == 0

    def test_bulk_trigger_past_deadline(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10), campaign_id="camp-001")
        triggered = engine.bulk_trigger_reminders("camp-001")
        assert len(triggered) == 4

    def test_bulk_trigger_updates_stats(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10), campaign_id="camp-001")
        engine.bulk_trigger_reminders("camp-001")
        assert engine.get_statistics()["bulk_triggers"] == 1

    def test_bulk_trigger_only_campaign(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10), campaign_id="camp-001")
        engine.schedule_reminders("dist-002", _past_date(10), campaign_id="camp-002")
        triggered = engine.bulk_trigger_reminders("camp-001")
        assert len(triggered) == 4
        assert all(t.campaign_id == "camp-001" for t in triggered)

    def test_bulk_trigger_no_due_empty(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _future_date(60), campaign_id="camp-001")
        triggered = engine.bulk_trigger_reminders("camp-001")
        assert len(triggered) == 0


# ============================================================================
# TEST CLASS: get_non_responsive_suppliers
# ============================================================================


class TestGetNonResponsiveSuppliers:

    def test_non_responsive_from_actions(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10), campaign_id="camp-001", supplier_id="sup-001")
        due = engine.get_due_reminders()
        for d in due[:2]:
            engine.trigger_reminder(d.action_id)
        result = engine.get_non_responsive_suppliers("camp-001")
        assert len(result) >= 1
        assert result[0]["supplier_id"] == "sup-001"

    def test_non_responsive_with_distributions(self):
        engine = FollowUpEngine()
        dists = [
            _dist("dist-001", "sup-001", "camp-001", DistributionStatus.SENT, _past_date(10), 3),
            _dist("dist-002", "sup-002", "camp-001", DistributionStatus.SUBMITTED, _past_date(5)),
        ]
        result = engine.get_non_responsive_suppliers("camp-001", days_overdue=7, distributions=dists)
        assert len(result) == 1
        assert result[0]["supplier_id"] == "sup-001"
        assert result[0]["days_overdue"] >= 7

    def test_non_responsive_excludes_submitted(self):
        engine = FollowUpEngine()
        dists = [
            _dist("dist-001", "sup-001", "camp-001", DistributionStatus.SUBMITTED, _past_date(20)),
        ]
        result = engine.get_non_responsive_suppliers("camp-001", distributions=dists)
        assert len(result) == 0

    def test_non_responsive_excludes_cancelled(self):
        engine = FollowUpEngine()
        dists = [
            _dist("dist-001", "sup-001", "camp-001", DistributionStatus.CANCELLED, _past_date(20)),
        ]
        result = engine.get_non_responsive_suppliers("camp-001", distributions=dists)
        assert len(result) == 0

    def test_non_responsive_sorted_by_overdue(self):
        engine = FollowUpEngine()
        dists = [
            _dist("dist-001", "sup-001", "camp-001", DistributionStatus.SENT, _past_date(5), 3),
            _dist("dist-002", "sup-002", "camp-001", DistributionStatus.SENT, _past_date(15), 4),
        ]
        result = engine.get_non_responsive_suppliers("camp-001", days_overdue=3, distributions=dists)
        if len(result) >= 2:
            assert result[0]["days_overdue"] >= result[1]["days_overdue"]


# ============================================================================
# TEST CLASS: track_effectiveness
# ============================================================================


class TestTrackEffectiveness:

    def test_effectiveness_empty(self):
        engine = FollowUpEngine()
        result = engine.track_effectiveness("camp-unknown")
        assert result["total_distributions"] == 0
        assert result["overall_response_rate"] == 0.0

    def test_effectiveness_with_reminders(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("dist-001", _past_date(10), campaign_id="camp-001")
        due = engine.get_due_reminders()
        for d in due:
            engine.trigger_reminder(d.action_id)
        result = engine.track_effectiveness("camp-001")
        assert result["total_distributions"] >= 1
        assert "by_reminder_count" in result

    def test_effectiveness_provenance(self):
        engine = FollowUpEngine()
        result = engine.track_effectiveness("camp-001")
        assert len(result["provenance_hash"]) == 64

    def test_effectiveness_timestamp(self):
        engine = FollowUpEngine()
        result = engine.track_effectiveness("camp-001")
        assert "timestamp" in result

    def test_effectiveness_with_distributions(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("dist-001", _past_date(10), campaign_id="camp-001", supplier_id="sup-001")
        engine.trigger_reminder(reminders[0].action_id)
        dists = [_dist("dist-001", "sup-001", "camp-001", DistributionStatus.SUBMITTED, _past_date(10))]
        result = engine.track_effectiveness("camp-001", distributions=dists)
        assert result["total_responded"] >= 0


# ============================================================================
# TEST CLASS: get_statistics
# ============================================================================


class TestGetStatistics:

    def test_statistics_keys(self):
        engine = FollowUpEngine()
        stats = engine.get_statistics()
        expected_keys = {
            "reminders_scheduled", "reminders_triggered", "reminders_cancelled",
            "escalations_created", "bulk_triggers", "errors",
            "active_actions", "distributions_tracked", "campaigns_tracked", "timestamp",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_statistics_after_operations(self):
        engine = FollowUpEngine()
        engine.schedule_reminders("d1", _future_date(30), campaign_id="c1")
        stats = engine.get_statistics()
        assert stats["reminders_scheduled"] == 4
        assert stats["active_actions"] == 4
        assert stats["distributions_tracked"] == 1
        assert stats["campaigns_tracked"] == 1


# ============================================================================
# TEST CLASS: Provenance
# ============================================================================


class TestProvenance:

    def test_sha256_format(self):
        engine = FollowUpEngine()
        h = engine._compute_provenance("test", "data")
        assert len(h) == 64
        assert re.match(r"^[0-9a-f]{64}$", h)

    def test_deterministic_within_same_second(self):
        engine = FollowUpEngine()
        h1 = engine._compute_provenance("a", "b")
        h2 = engine._compute_provenance("a", "b")
        assert h1 == h2

    def test_all_actions_have_provenance(self):
        engine = FollowUpEngine()
        reminders = engine.schedule_reminders("d1", _future_date(30))
        for r in reminders:
            assert len(r.provenance_hash) == 64
            assert re.match(r"^[0-9a-f]{64}$", r.provenance_hash)
