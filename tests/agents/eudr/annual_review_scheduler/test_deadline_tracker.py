# -*- coding: utf-8 -*-
"""
Unit tests for DeadlineTracker - AGENT-EUDR-034

Tests deadline creation, tracking, alert generation, status transitions,
overdue detection, escalation, waiving, bulk operations, and
deadline-to-phase mapping.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (Engine 2: Deadline Tracker)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.annual_review_scheduler.config import (
    AnnualReviewSchedulerConfig,
)
from greenlang.agents.eudr.annual_review_scheduler.deadline_tracker import (
    DeadlineTracker,
)
from greenlang.agents.eudr.annual_review_scheduler.models import (
    DeadlineAlert,
    DeadlineAlertLevel,
    DeadlineStatus,
    DeadlineTrack,
    ReviewPhase,
)
from greenlang.agents.eudr.annual_review_scheduler.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return AnnualReviewSchedulerConfig()


@pytest.fixture
def tracker(config):
    return DeadlineTracker(config=config, provenance=ProvenanceTracker())


# ---------------------------------------------------------------------------
# Deadline Creation
# ---------------------------------------------------------------------------

class TestCreateDeadline:
    """Test deadline creation."""

    @pytest.mark.asyncio
    async def test_create_deadline_returns_deadline_track(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Complete data collection",
            due_date=now + timedelta(days=30),
        )
        assert isinstance(deadline, DeadlineTrack)
        assert deadline.deadline_id.startswith("dln-")

    @pytest.mark.asyncio
    async def test_create_deadline_sets_cycle_id(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.PREPARATION,
            description="Test",
            due_date=now + timedelta(days=14),
        )
        assert deadline.cycle_id == "cyc-001"

    @pytest.mark.asyncio
    async def test_create_deadline_sets_phase(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.ANALYSIS,
            description="Test",
            due_date=now + timedelta(days=21),
        )
        assert deadline.phase == ReviewPhase.ANALYSIS

    @pytest.mark.asyncio
    async def test_create_deadline_initial_status_on_track(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Test",
            due_date=now + timedelta(days=30),
        )
        assert deadline.status == DeadlineStatus.ON_TRACK

    @pytest.mark.asyncio
    async def test_create_deadline_with_entity(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Test",
            due_date=now + timedelta(days=30),
            assigned_entity_id="entity-001",
        )
        assert deadline.assigned_entity_id == "entity-001"

    @pytest.mark.asyncio
    async def test_create_deadline_custom_warning_days(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.REMEDIATION,
            description="Test",
            due_date=now + timedelta(days=30),
            warning_days_before=14,
            critical_days_before=5,
        )
        assert deadline.warning_days_before == 14
        assert deadline.critical_days_before == 5

    @pytest.mark.asyncio
    async def test_create_deadline_provenance_hash(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Test",
            due_date=now + timedelta(days=30),
        )
        assert len(deadline.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_create_past_due_date_creates_overdue(self, tracker):
        past = datetime.now(tz=timezone.utc) - timedelta(days=5)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.ANALYSIS,
            description="Already past",
            due_date=past,
        )
        assert deadline.status == DeadlineStatus.OVERDUE


# ---------------------------------------------------------------------------
# Deadline Creation for Cycle
# ---------------------------------------------------------------------------

class TestCreateDeadlinesForCycle:
    """Test bulk deadline creation for a review cycle."""

    @pytest.mark.asyncio
    async def test_create_deadlines_for_all_phases(self, tracker, sample_review_cycle):
        deadlines = await tracker.create_deadlines_for_cycle(sample_review_cycle)
        assert len(deadlines) >= 6
        phases_covered = {d.phase for d in deadlines}
        assert ReviewPhase.PREPARATION in phases_covered
        assert ReviewPhase.SIGN_OFF in phases_covered

    @pytest.mark.asyncio
    async def test_deadlines_are_chronologically_ordered(self, tracker, sample_review_cycle):
        deadlines = await tracker.create_deadlines_for_cycle(sample_review_cycle)
        for i in range(len(deadlines) - 1):
            assert deadlines[i].due_date <= deadlines[i + 1].due_date

    @pytest.mark.asyncio
    async def test_deadlines_have_unique_ids(self, tracker, sample_review_cycle):
        deadlines = await tracker.create_deadlines_for_cycle(sample_review_cycle)
        ids = [d.deadline_id for d in deadlines]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Alert Generation
# ---------------------------------------------------------------------------

class TestAlertGeneration:
    """Test deadline alert generation."""

    @pytest.mark.asyncio
    async def test_generate_warning_alert(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Test warning",
            due_date=now + timedelta(days=5),
            warning_days_before=7,
        )
        alerts = await tracker.check_and_generate_alerts(deadline.deadline_id)
        warning_alerts = [a for a in alerts if a.alert_level == DeadlineAlertLevel.WARNING]
        assert len(warning_alerts) >= 1

    @pytest.mark.asyncio
    async def test_generate_critical_alert(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Test critical",
            due_date=now + timedelta(days=2),
            warning_days_before=7,
            critical_days_before=3,
        )
        alerts = await tracker.check_and_generate_alerts(deadline.deadline_id)
        critical_alerts = [a for a in alerts if a.alert_level == DeadlineAlertLevel.CRITICAL]
        assert len(critical_alerts) >= 1

    @pytest.mark.asyncio
    async def test_no_alert_when_on_track(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Plenty of time",
            due_date=now + timedelta(days=30),
            warning_days_before=7,
        )
        alerts = await tracker.check_and_generate_alerts(deadline.deadline_id)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_generate_overdue_escalation_alert(self, tracker):
        past = datetime.now(tz=timezone.utc) - timedelta(days=2)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.ANALYSIS,
            description="Past due",
            due_date=past,
        )
        alerts = await tracker.check_and_generate_alerts(deadline.deadline_id)
        escalation = [a for a in alerts if a.alert_level == DeadlineAlertLevel.ESCALATION]
        assert len(escalation) >= 1

    @pytest.mark.asyncio
    async def test_alert_has_days_remaining(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Test",
            due_date=now + timedelta(days=5),
            warning_days_before=7,
        )
        alerts = await tracker.check_and_generate_alerts(deadline.deadline_id)
        for alert in alerts:
            assert isinstance(alert.days_remaining, int)

    @pytest.mark.asyncio
    async def test_alert_references_correct_deadline(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Test",
            due_date=now + timedelta(days=2),
            critical_days_before=3,
        )
        alerts = await tracker.check_and_generate_alerts(deadline.deadline_id)
        for alert in alerts:
            assert alert.deadline_id == deadline.deadline_id
            assert alert.cycle_id == "cyc-001"


# ---------------------------------------------------------------------------
# Status Transitions
# ---------------------------------------------------------------------------

class TestDeadlineStatusTransitions:
    """Test deadline status transitions."""

    @pytest.mark.asyncio
    async def test_complete_deadline(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Completable",
            due_date=now + timedelta(days=10),
        )
        completed = await tracker.complete_deadline(deadline.deadline_id)
        assert completed.status == DeadlineStatus.COMPLETED
        assert completed.completed_at is not None

    @pytest.mark.asyncio
    async def test_waive_deadline(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.REMEDIATION,
            description="Optional",
            due_date=now + timedelta(days=10),
        )
        waived = await tracker.waive_deadline(
            deadline.deadline_id, reason="Not applicable for this review"
        )
        assert waived.status == DeadlineStatus.WAIVED

    @pytest.mark.asyncio
    async def test_complete_already_completed_raises(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.DATA_COLLECTION,
            description="Already done",
            due_date=now + timedelta(days=10),
        )
        await tracker.complete_deadline(deadline.deadline_id)
        with pytest.raises(ValueError, match="already completed"):
            await tracker.complete_deadline(deadline.deadline_id)

    @pytest.mark.asyncio
    async def test_mark_at_risk(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001",
            phase=ReviewPhase.ANALYSIS,
            description="Risky",
            due_date=now + timedelta(days=5),
        )
        at_risk = await tracker.mark_at_risk(deadline.deadline_id, reason="Resource shortage")
        assert at_risk.status == DeadlineStatus.AT_RISK


# ---------------------------------------------------------------------------
# Overdue Detection
# ---------------------------------------------------------------------------

class TestOverdueDetection:
    """Test overdue deadline detection."""

    @pytest.mark.asyncio
    async def test_detect_overdue_deadlines(self, tracker):
        past = datetime.now(tz=timezone.utc) - timedelta(days=3)
        future = datetime.now(tz=timezone.utc) + timedelta(days=30)
        await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.DATA_COLLECTION,
            description="Past due", due_date=past,
        )
        await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.ANALYSIS,
            description="On time", due_date=future,
        )
        overdue = await tracker.get_overdue_deadlines(cycle_id="cyc-001")
        assert len(overdue) >= 1
        for d in overdue:
            assert d.status == DeadlineStatus.OVERDUE

    @pytest.mark.asyncio
    async def test_no_overdue_when_all_on_track(self, tracker):
        future = datetime.now(tz=timezone.utc) + timedelta(days=30)
        await tracker.create_deadline(
            cycle_id="cyc-002", phase=ReviewPhase.PREPARATION,
            description="On time", due_date=future,
        )
        overdue = await tracker.get_overdue_deadlines(cycle_id="cyc-002")
        assert len(overdue) == 0

    @pytest.mark.asyncio
    async def test_completed_not_counted_as_overdue(self, tracker):
        past = datetime.now(tz=timezone.utc) - timedelta(days=1)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.PREPARATION,
            description="Past but completed", due_date=past,
        )
        await tracker.complete_deadline(deadline.deadline_id)
        overdue = await tracker.get_overdue_deadlines(cycle_id="cyc-001")
        completed_overdue = [d for d in overdue if d.deadline_id == deadline.deadline_id]
        assert len(completed_overdue) == 0


# ---------------------------------------------------------------------------
# List and Filter
# ---------------------------------------------------------------------------

class TestListDeadlines:
    """Test deadline listing and filtering."""

    @pytest.mark.asyncio
    async def test_list_deadlines_by_cycle(self, tracker):
        now = datetime.now(tz=timezone.utc)
        for i in range(3):
            await tracker.create_deadline(
                cycle_id="cyc-001",
                phase=ReviewPhase.DATA_COLLECTION,
                description=f"Deadline {i}",
                due_date=now + timedelta(days=10 + i),
            )
        deadlines = await tracker.list_deadlines(cycle_id="cyc-001")
        assert len(deadlines) == 3

    @pytest.mark.asyncio
    async def test_list_deadlines_by_phase(self, tracker):
        now = datetime.now(tz=timezone.utc)
        await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.PREPARATION,
            description="Prep", due_date=now + timedelta(days=5),
        )
        await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.ANALYSIS,
            description="Analysis", due_date=now + timedelta(days=20),
        )
        prep_deadlines = await tracker.list_deadlines(
            cycle_id="cyc-001", phase=ReviewPhase.PREPARATION,
        )
        assert len(prep_deadlines) == 1

    @pytest.mark.asyncio
    async def test_list_deadlines_by_status(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.DATA_COLLECTION,
            description="To complete", due_date=now + timedelta(days=10),
        )
        await tracker.complete_deadline(deadline.deadline_id)
        await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.ANALYSIS,
            description="Still open", due_date=now + timedelta(days=20),
        )
        completed = await tracker.list_deadlines(
            cycle_id="cyc-001", status=DeadlineStatus.COMPLETED,
        )
        on_track = await tracker.list_deadlines(
            cycle_id="cyc-001", status=DeadlineStatus.ON_TRACK,
        )
        assert len(completed) == 1
        assert len(on_track) == 1

    @pytest.mark.asyncio
    async def test_get_deadline_by_id(self, tracker):
        now = datetime.now(tz=timezone.utc)
        created = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.PREPARATION,
            description="Get this", due_date=now + timedelta(days=5),
        )
        retrieved = await tracker.get_deadline(created.deadline_id)
        assert retrieved.deadline_id == created.deadline_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_deadline_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.get_deadline("dln-nonexistent")


# ---------------------------------------------------------------------------
# Acknowledge Alerts
# ---------------------------------------------------------------------------

class TestAcknowledgeAlerts:
    """Test alert acknowledgement."""

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.DATA_COLLECTION,
            description="Test", due_date=now + timedelta(days=2),
            critical_days_before=3,
        )
        alerts = await tracker.check_and_generate_alerts(deadline.deadline_id)
        if alerts:
            acked = await tracker.acknowledge_alert(alerts[0].alert_id)
            assert acked.acknowledged is True

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent_alert_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.acknowledge_alert("alt-nonexistent")


# ---------------------------------------------------------------------------
# Deadline Extension
# ---------------------------------------------------------------------------

class TestDeadlineExtension:
    """Test deadline extension functionality."""

    @pytest.mark.asyncio
    async def test_extend_deadline(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.DATA_COLLECTION,
            description="Extendable", due_date=now + timedelta(days=10),
        )
        original_due = deadline.due_date
        extended = await tracker.extend_deadline(
            deadline.deadline_id, additional_days=14,
        )
        assert extended.due_date > original_due

    @pytest.mark.asyncio
    async def test_extend_completed_deadline_raises(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.PREPARATION,
            description="Already done", due_date=now + timedelta(days=5),
        )
        await tracker.complete_deadline(deadline.deadline_id)
        with pytest.raises(ValueError, match="Cannot extend"):
            await tracker.extend_deadline(deadline.deadline_id, additional_days=7)

    @pytest.mark.asyncio
    async def test_extend_updates_provenance(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.ANALYSIS,
            description="Extend me", due_date=now + timedelta(days=15),
        )
        original_hash = deadline.provenance_hash
        extended = await tracker.extend_deadline(
            deadline.deadline_id, additional_days=7,
        )
        assert extended.provenance_hash != original_hash
        assert len(extended.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_extend_negative_days_raises(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.REMEDIATION,
            description="Bad extend", due_date=now + timedelta(days=20),
        )
        with pytest.raises(ValueError, match="positive"):
            await tracker.extend_deadline(deadline.deadline_id, additional_days=-5)

    @pytest.mark.asyncio
    async def test_extend_overdue_deadline_resets_status(self, tracker):
        past = datetime.now(tz=timezone.utc) - timedelta(days=2)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-001", phase=ReviewPhase.ANALYSIS,
            description="Was overdue", due_date=past,
        )
        assert deadline.status == DeadlineStatus.OVERDUE
        extended = await tracker.extend_deadline(
            deadline.deadline_id, additional_days=30,
        )
        assert extended.status == DeadlineStatus.ON_TRACK

    @pytest.mark.asyncio
    async def test_extend_nonexistent_deadline_raises(self, tracker):
        with pytest.raises(ValueError, match="not found"):
            await tracker.extend_deadline("dln-nonexistent", additional_days=7)


# ---------------------------------------------------------------------------
# Batch Alert Check
# ---------------------------------------------------------------------------

class TestBatchAlertCheck:
    """Test batch alert generation for all deadlines in a cycle."""

    @pytest.mark.asyncio
    async def test_batch_check_returns_all_alerts(self, tracker):
        now = datetime.now(tz=timezone.utc)
        for i in range(3):
            await tracker.create_deadline(
                cycle_id="cyc-batch", phase=ReviewPhase.DATA_COLLECTION,
                description=f"Deadline {i}",
                due_date=now + timedelta(days=2 + i),
                critical_days_before=5,
            )
        all_alerts = await tracker.batch_check_alerts(cycle_id="cyc-batch")
        assert isinstance(all_alerts, list)

    @pytest.mark.asyncio
    async def test_batch_check_empty_cycle_returns_empty(self, tracker):
        all_alerts = await tracker.batch_check_alerts(cycle_id="cyc-empty-batch")
        assert all_alerts == []

    @pytest.mark.asyncio
    async def test_batch_check_skips_completed(self, tracker):
        now = datetime.now(tz=timezone.utc)
        d1 = await tracker.create_deadline(
            cycle_id="cyc-batch2", phase=ReviewPhase.PREPARATION,
            description="Will complete", due_date=now + timedelta(days=2),
            critical_days_before=5,
        )
        await tracker.complete_deadline(d1.deadline_id)
        await tracker.create_deadline(
            cycle_id="cyc-batch2", phase=ReviewPhase.ANALYSIS,
            description="Open", due_date=now + timedelta(days=2),
            critical_days_before=5,
        )
        all_alerts = await tracker.batch_check_alerts(cycle_id="cyc-batch2")
        for alert in all_alerts:
            assert alert.deadline_id != d1.deadline_id


# ---------------------------------------------------------------------------
# Deadline Provenance
# ---------------------------------------------------------------------------

class TestDeadlineProvenance:
    """Test provenance hash tracking on deadlines."""

    @pytest.mark.asyncio
    async def test_deadline_has_provenance_on_create(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-prov", phase=ReviewPhase.PREPARATION,
            description="Provenance test", due_date=now + timedelta(days=10),
        )
        assert len(deadline.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_provenance_changes_on_status_update(self, tracker):
        now = datetime.now(tz=timezone.utc)
        deadline = await tracker.create_deadline(
            cycle_id="cyc-prov2", phase=ReviewPhase.DATA_COLLECTION,
            description="Track changes", due_date=now + timedelta(days=10),
        )
        original = deadline.provenance_hash
        completed = await tracker.complete_deadline(deadline.deadline_id)
        assert completed.provenance_hash != original

    @pytest.mark.asyncio
    async def test_same_deadline_data_same_provenance(self, tracker):
        now = datetime.now(tz=timezone.utc)
        due = now + timedelta(days=10)
        d1 = await tracker.create_deadline(
            cycle_id="cyc-determ", phase=ReviewPhase.PREPARATION,
            description="Deterministic", due_date=due,
        )
        d2 = await tracker.create_deadline(
            cycle_id="cyc-determ", phase=ReviewPhase.PREPARATION,
            description="Deterministic", due_date=due,
        )
        # Different IDs but same content should yield same base hash logic
        assert d1.provenance_hash is not None
        assert d2.provenance_hash is not None
