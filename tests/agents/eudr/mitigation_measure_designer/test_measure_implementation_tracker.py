# -*- coding: utf-8 -*-
"""
Unit tests for MeasureImplementationTracker - AGENT-EUDR-029

Tests the full measure lifecycle: propose, approve, start, complete,
cancel. Also tests milestones, evidence, overdue detection, and
implementation progress.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.mitigation_measure_designer.config import (
    MitigationMeasureDesignerConfig,
)
from greenlang.agents.eudr.mitigation_measure_designer.measure_implementation_tracker import (
    MeasureImplementationTracker,
)
from greenlang.agents.eudr.mitigation_measure_designer.models import (
    Article11Category,
    EvidenceType,
    MeasurePriority,
    MeasureStatus,
    MeasureTemplate,
    MitigationMeasure,
    RiskDimension,
)
from greenlang.agents.eudr.mitigation_measure_designer.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return MitigationMeasureDesignerConfig()


@pytest.fixture
def tracker(config):
    return MeasureImplementationTracker(
        config=config, provenance=ProvenanceTracker(),
    )


@pytest.fixture
def base_template():
    return MeasureTemplate(
        template_id="TPL-TRACK-001",
        title="Tracking Test Template",
        description="Template for tracker tests.",
        article11_category=Article11Category.INDEPENDENT_AUDIT,
        applicable_dimensions=[RiskDimension.SUPPLIER],
        base_effectiveness=Decimal("25"),
    )


class TestProposeMeasure:
    """Test propose_measure lifecycle start."""

    def test_propose_creates_measure(self, tracker, base_template):
        measure = tracker.propose_measure(
            strategy_id="stg-001",
            template=base_template,
            dimension=RiskDimension.SUPPLIER,
        )
        assert measure.measure_id.startswith("msr-")
        assert measure.status == MeasureStatus.PROPOSED
        assert measure.strategy_id == "stg-001"
        assert measure.template_id == "TPL-TRACK-001"
        assert measure.title == "Tracking Test Template"

    def test_propose_sets_deadline(self, tracker, base_template):
        measure = tracker.propose_measure(
            strategy_id="stg-001",
            template=base_template,
            dimension=RiskDimension.SUPPLIER,
        )
        assert measure.deadline is not None
        assert measure.deadline > datetime.now(timezone.utc)

    def test_propose_with_assignee(self, tracker, base_template):
        measure = tracker.propose_measure(
            strategy_id="stg-001",
            template=base_template,
            dimension=RiskDimension.SUPPLIER,
            assigned_to="auditor@test.com",
        )
        assert measure.assigned_to == "auditor@test.com"

    def test_propose_with_priority(self, tracker, base_template):
        measure = tracker.propose_measure(
            strategy_id="stg-001",
            template=base_template,
            dimension=RiskDimension.SUPPLIER,
            priority=MeasurePriority.CRITICAL,
        )
        assert measure.priority == MeasurePriority.CRITICAL


class TestMeasureLifecycle:
    """Test full lifecycle: propose -> approve -> start -> complete."""

    def test_approve_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        approved = tracker.approve_measure(measure.measure_id, "admin")
        assert approved.status == MeasureStatus.APPROVED

    def test_start_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        tracker.approve_measure(measure.measure_id, "admin")
        started = tracker.start_measure(measure.measure_id)
        assert started.status == MeasureStatus.IN_PROGRESS
        assert started.started_at is not None

    def test_complete_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        tracker.approve_measure(measure.measure_id, "admin")
        tracker.start_measure(measure.measure_id)
        completed = tracker.complete_measure(measure.measure_id, Decimal("22"))
        assert completed.status == MeasureStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.actual_risk_reduction == Decimal("22")

    def test_cancel_proposed_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        cancelled = tracker.cancel_measure(measure.measure_id, "Budget cut")
        assert cancelled.status == MeasureStatus.CANCELLED

    def test_cancel_in_progress_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        tracker.approve_measure(measure.measure_id, "admin")
        tracker.start_measure(measure.measure_id)
        cancelled = tracker.cancel_measure(measure.measure_id, "Scope change")
        assert cancelled.status == MeasureStatus.CANCELLED


class TestInvalidTransitions:
    """Test that invalid status transitions raise ValueError."""

    def test_cannot_start_before_approve(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        with pytest.raises(ValueError, match="Invalid transition"):
            tracker.start_measure(measure.measure_id)

    def test_cannot_complete_before_start(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        tracker.approve_measure(measure.measure_id, "admin")
        with pytest.raises(ValueError, match="Invalid transition"):
            tracker.complete_measure(measure.measure_id)

    def test_cannot_cancel_completed_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        tracker.approve_measure(measure.measure_id, "admin")
        tracker.start_measure(measure.measure_id)
        tracker.complete_measure(measure.measure_id)
        with pytest.raises(ValueError, match="Invalid transition"):
            tracker.cancel_measure(measure.measure_id, "Too late")

    def test_nonexistent_measure_raises(self, tracker):
        with pytest.raises(ValueError, match="Measure not found"):
            tracker.approve_measure("nonexistent-id", "admin")


class TestMilestones:
    """Test milestone add and complete."""

    def test_add_milestone(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        due = datetime.now(timezone.utc) + timedelta(days=14)
        milestone = tracker.add_milestone(
            measure_id=measure.measure_id,
            title="Data Collection Complete",
            due_date=due,
            description="All supplier data collected and reviewed.",
        )
        assert milestone.milestone_id.startswith("mst-")
        assert milestone.title == "Data Collection Complete"
        assert milestone.status == MeasureStatus.PROPOSED

    def test_complete_milestone(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        due = datetime.now(timezone.utc) + timedelta(days=14)
        milestone = tracker.add_milestone(
            measure_id=measure.measure_id,
            title="Data Collection",
            due_date=due,
        )
        completed = tracker.complete_milestone(milestone.milestone_id)
        assert completed.status == MeasureStatus.COMPLETED
        assert completed.completed_at is not None

    def test_complete_nonexistent_milestone_raises(self, tracker):
        with pytest.raises(ValueError, match="Milestone not found"):
            tracker.complete_milestone("nonexistent-mst")


class TestEvidence:
    """Test evidence attachment."""

    def test_add_evidence_to_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        evidence = tracker.add_evidence(
            measure_id=measure.measure_id,
            evidence_type=EvidenceType.AUDIT_REPORT,
            title="Q1 Audit Report",
            file_ref="s3://evidence/q1_audit.pdf",
            uploaded_by="auditor@test.com",
        )
        assert evidence.evidence_id.startswith("evd-")
        assert evidence.evidence_type == EvidenceType.AUDIT_REPORT

    def test_evidence_added_to_measure_evidence_ids(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        evidence = tracker.add_evidence(
            measure_id=measure.measure_id,
            evidence_type=EvidenceType.DOCUMENT,
            title="Compliance Doc",
            file_ref="s3://docs/compliance.pdf",
            uploaded_by="user@test.com",
        )
        updated = tracker.get_measure(measure.measure_id)
        assert evidence.evidence_id in updated.evidence_ids

    def test_add_evidence_nonexistent_measure_raises(self, tracker):
        with pytest.raises(ValueError, match="Measure not found"):
            tracker.add_evidence(
                measure_id="nonexistent",
                evidence_type=EvidenceType.DOCUMENT,
                title="Doc",
                file_ref="s3://test",
                uploaded_by="user",
            )


class TestOverdueDetection:
    """Test overdue measure detection."""

    def test_no_overdue_initially(self, tracker):
        assert len(tracker.get_overdue_measures()) == 0

    def test_detects_overdue_measure(self, tracker, base_template):
        measure = tracker.propose_measure("stg-001", base_template, RiskDimension.SUPPLIER)
        # Manually set deadline to the past
        tracker._measures[measure.measure_id].deadline = (
            datetime.now(timezone.utc) - timedelta(days=1)
        )
        overdue = tracker.get_overdue_measures()
        assert len(overdue) == 1
        assert overdue[0].measure_id == measure.measure_id


class TestImplementationProgress:
    """Test get_implementation_progress."""

    def test_empty_strategy_progress(self, tracker):
        progress = tracker.get_implementation_progress("nonexistent-stg")
        assert progress["total_measures"] == 0
        assert progress["completion_pct"] == Decimal("0")

    def test_progress_with_mixed_statuses(self, tracker, base_template):
        m1 = tracker.propose_measure("stg-prog", base_template, RiskDimension.SUPPLIER)
        m2 = tracker.propose_measure("stg-prog", base_template, RiskDimension.SUPPLIER)
        tracker.approve_measure(m1.measure_id, "admin")
        tracker.start_measure(m1.measure_id)
        tracker.complete_measure(m1.measure_id)

        progress = tracker.get_implementation_progress("stg-prog")
        assert progress["total_measures"] == 2
        assert progress["completed_count"] == 1
        assert progress["completion_pct"] == Decimal("50.00")
