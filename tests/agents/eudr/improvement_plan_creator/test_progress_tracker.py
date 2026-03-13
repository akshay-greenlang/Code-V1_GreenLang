# -*- coding: utf-8 -*-
"""
Unit tests for ProgressTracker - AGENT-EUDR-035

Tests progress snapshot capture, milestone tracking, overdue detection,
effectiveness review, and plan-level progress metrics. Validates progress
calculation, risk trend analysis, and extension management.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (Engine 6: Progress Tracker)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
)
from greenlang.agents.eudr.improvement_plan_creator.progress_tracker import (
    ProgressTracker,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    ActionStatus,
    ActionType,
    EisenhowerQuadrant,
    EUDRCommodity,
    GapSeverity,
    ImprovementAction,
    ImprovementPlan,
    PlanStatus,
    ProgressMilestone,
    ProgressSnapshot,
    RiskLevel,
)


@pytest.fixture
def config():
    return ImprovementPlanCreatorConfig()


@pytest.fixture
def tracker(config):
    return ProgressTracker(config=config)


@pytest.fixture
def sample_plan(multiple_actions):
    """Create a sample improvement plan with actions."""
    now = datetime.now(tz=timezone.utc)
    return ImprovementPlan(
        plan_id="plan-001",
        operator_id="operator-001",
        title="Test Plan",
        description="Test improvement plan",
        status=PlanStatus.ACTIVE,
        commodity=EUDRCommodity.COCOA,
        risk_level=RiskLevel.HIGH,
        total_gaps=3,
        total_actions=len(multiple_actions),
        estimated_total_cost=Decimal("92000.00"),
        estimated_completion_days=120,
        target_completion=now + timedelta(days=120),
        created_at=now,
        actions=multiple_actions,
        provenance_hash="w" * 64,
    )


# ---------------------------------------------------------------------------
# Capture Snapshot
# ---------------------------------------------------------------------------

class TestCaptureSnapshot:
    """Test progress snapshot capture."""

    @pytest.mark.asyncio
    async def test_capture_returns_snapshot(self, tracker, sample_plan):
        snapshot = await tracker.capture_snapshot(sample_plan)
        assert isinstance(snapshot, ProgressSnapshot)

    @pytest.mark.asyncio
    async def test_snapshot_has_id(self, tracker, sample_plan):
        snapshot = await tracker.capture_snapshot(sample_plan)
        assert snapshot.snapshot_id.startswith("SNAP-")

    @pytest.mark.asyncio
    async def test_snapshot_links_to_plan(self, tracker, sample_plan):
        snapshot = await tracker.capture_snapshot(sample_plan)
        assert snapshot.plan_id == "plan-001"

    @pytest.mark.asyncio
    async def test_snapshot_has_progress(self, tracker, sample_plan):
        snapshot = await tracker.capture_snapshot(sample_plan)
        assert isinstance(snapshot.overall_progress, Decimal)
        assert Decimal("0") <= snapshot.overall_progress <= Decimal("100")

    @pytest.mark.asyncio
    async def test_snapshot_counts_actions(self, tracker, sample_plan):
        snapshot = await tracker.capture_snapshot(sample_plan)
        assert snapshot.actions_total == len(sample_plan.actions)

    @pytest.mark.asyncio
    async def test_snapshot_counts_completed(self, tracker, sample_plan):
        snapshot = await tracker.capture_snapshot(sample_plan)
        completed = sum(
            1 for a in sample_plan.actions
            if a.status in (ActionStatus.COMPLETED, ActionStatus.VERIFIED, ActionStatus.CLOSED)
        )
        assert snapshot.actions_completed == completed

    @pytest.mark.asyncio
    async def test_snapshot_has_provenance(self, tracker, sample_plan):
        snapshot = await tracker.capture_snapshot(sample_plan)
        assert len(snapshot.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Check Overdue
# ---------------------------------------------------------------------------

class TestCheckOverdue:
    """Test overdue action detection."""

    @pytest.mark.asyncio
    async def test_check_overdue_returns_list(self, tracker, sample_plan):
        overdue = await tracker.check_overdue(sample_plan)
        assert isinstance(overdue, list)

    @pytest.mark.asyncio
    async def test_check_overdue_empty_for_future(self, tracker):
        now = datetime.now(tz=timezone.utc)
        actions = [
            ImprovementAction(
                action_id="act-001",
                plan_id="plan-001",
                gap_id="gap-001",
                action_type=ActionType.CORRECTIVE,
                title="Test Action",
                status=ActionStatus.IN_PROGRESS,
                time_bound_deadline=now + timedelta(days=30),
                priority_score=Decimal("80.00"),
                eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
                urgency_score=Decimal("85.00"),
                importance_score=Decimal("75.00"),
                provenance_hash="a" * 64,
            )
        ]
        plan = ImprovementPlan(
            plan_id="plan-001",
            operator_id="operator-001",
            title="Test Plan",
            status=PlanStatus.ACTIVE,
            actions=actions,
            provenance_hash="p" * 64,
        )
        overdue = await tracker.check_overdue(plan)
        assert len(overdue) == 0

    @pytest.mark.asyncio
    async def test_check_overdue_detects_past(self, tracker):
        now = datetime.now(tz=timezone.utc)
        actions = [
            ImprovementAction(
                action_id="act-001",
                plan_id="plan-001",
                gap_id="gap-001",
                action_type=ActionType.CORRECTIVE,
                title="Test Action",
                status=ActionStatus.IN_PROGRESS,
                time_bound_deadline=now - timedelta(days=5),
                priority_score=Decimal("80.00"),
                eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
                urgency_score=Decimal("85.00"),
                importance_score=Decimal("75.00"),
                provenance_hash="a" * 64,
            )
        ]
        plan = ImprovementPlan(
            plan_id="plan-001",
            operator_id="operator-001",
            title="Test Plan",
            status=PlanStatus.ACTIVE,
            actions=actions,
            provenance_hash="p" * 64,
        )
        overdue = await tracker.check_overdue(plan)
        assert len(overdue) == 1

    @pytest.mark.asyncio
    async def test_check_overdue_ignores_completed(self, tracker):
        now = datetime.now(tz=timezone.utc)
        actions = [
            ImprovementAction(
                action_id="act-001",
                plan_id="plan-001",
                gap_id="gap-001",
                action_type=ActionType.CORRECTIVE,
                title="Test Action",
                status=ActionStatus.COMPLETED,
                time_bound_deadline=now - timedelta(days=5),
                priority_score=Decimal("80.00"),
                eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
                urgency_score=Decimal("85.00"),
                importance_score=Decimal("75.00"),
                provenance_hash="a" * 64,
            )
        ]
        plan = ImprovementPlan(
            plan_id="plan-001",
            operator_id="operator-001",
            title="Test Plan",
            status=PlanStatus.ACTIVE,
            actions=actions,
            provenance_hash="p" * 64,
        )
        overdue = await tracker.check_overdue(plan)
        assert len(overdue) == 0


# ---------------------------------------------------------------------------
# Milestone Management
# ---------------------------------------------------------------------------

class TestMilestoneManagement:
    """Test milestone creation and retrieval."""

    @pytest.mark.asyncio
    async def test_add_milestone_returns_record(self, tracker):
        milestone = await tracker.add_milestone(
            action_id="act-001",
            title="Design review complete",
            due_date=datetime.now(tz=timezone.utc) + timedelta(days=14),
        )
        assert isinstance(milestone, ProgressMilestone)

    @pytest.mark.asyncio
    async def test_add_milestone_has_id(self, tracker):
        milestone = await tracker.add_milestone(
            action_id="act-001",
            title="Test milestone",
            due_date=datetime.now(tz=timezone.utc) + timedelta(days=7),
        )
        assert milestone.milestone_id.startswith("MS-")

    @pytest.mark.asyncio
    async def test_add_milestone_links_to_action(self, tracker):
        milestone = await tracker.add_milestone(
            action_id="act-001",
            title="Test milestone",
            due_date=datetime.now(tz=timezone.utc) + timedelta(days=7),
        )
        assert milestone.action_id == "act-001"

    @pytest.mark.asyncio
    async def test_get_milestones_returns_list(self, tracker):
        await tracker.add_milestone(
            action_id="act-001",
            title="Milestone 1",
        )
        await tracker.add_milestone(
            action_id="act-001",
            title="Milestone 2",
        )
        milestones = await tracker.get_milestones("act-001")
        assert isinstance(milestones, list)
        assert len(milestones) == 2

    @pytest.mark.asyncio
    async def test_get_milestones_empty_for_unknown(self, tracker):
        milestones = await tracker.get_milestones("unknown-action")
        assert milestones == []


# ---------------------------------------------------------------------------
# Extension Management
# ---------------------------------------------------------------------------

class TestExtensionManagement:
    """Test deadline extension granting."""

    @pytest.mark.asyncio
    async def test_grant_extension_returns_bool(self, tracker):
        now = datetime.now(tz=timezone.utc)
        action = ImprovementAction(
            action_id="act-001",
            plan_id="plan-001",
            gap_id="gap-001",
            action_type=ActionType.CORRECTIVE,
            title="Test Action",
            status=ActionStatus.IN_PROGRESS,
            time_bound_deadline=now + timedelta(days=30),
            priority_score=Decimal("80.00"),
            eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
            urgency_score=Decimal("85.00"),
            importance_score=Decimal("75.00"),
            provenance_hash="a" * 64,
        )
        result = await tracker.grant_extension(action, extension_days=7)
        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_grant_extension_updates_deadline(self, tracker):
        now = datetime.now(tz=timezone.utc)
        original_deadline = now + timedelta(days=30)
        action = ImprovementAction(
            action_id="act-001",
            plan_id="plan-001",
            gap_id="gap-001",
            action_type=ActionType.CORRECTIVE,
            title="Test Action",
            status=ActionStatus.IN_PROGRESS,
            time_bound_deadline=original_deadline,
            priority_score=Decimal("80.00"),
            eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
            urgency_score=Decimal("85.00"),
            importance_score=Decimal("75.00"),
            provenance_hash="a" * 64,
        )
        await tracker.grant_extension(action, extension_days=7)
        assert action.time_bound_deadline == original_deadline + timedelta(days=7)

    @pytest.mark.asyncio
    async def test_grant_extension_increments_counter(self, tracker):
        now = datetime.now(tz=timezone.utc)
        action = ImprovementAction(
            action_id="act-001",
            plan_id="plan-001",
            gap_id="gap-001",
            action_type=ActionType.CORRECTIVE,
            title="Test Action",
            status=ActionStatus.IN_PROGRESS,
            time_bound_deadline=now + timedelta(days=30),
            priority_score=Decimal("80.00"),
            eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
            urgency_score=Decimal("85.00"),
            importance_score=Decimal("75.00"),
            provenance_hash="a" * 64,
        )
        await tracker.grant_extension(action, extension_days=7)
        assert action.extensions_used == 1

    @pytest.mark.asyncio
    async def test_grant_extension_respects_max_limit(self, tracker, config):
        now = datetime.now(tz=timezone.utc)
        action = ImprovementAction(
            action_id="act-001",
            plan_id="plan-001",
            gap_id="gap-001",
            action_type=ActionType.CORRECTIVE,
            title="Test Action",
            status=ActionStatus.IN_PROGRESS,
            time_bound_deadline=now + timedelta(days=30),
            extensions_used=config.max_extensions_per_action,
            priority_score=Decimal("80.00"),
            eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
            urgency_score=Decimal("85.00"),
            importance_score=Decimal("75.00"),
            provenance_hash="a" * 64,
        )
        result = await tracker.grant_extension(action, extension_days=7)
        assert result is False


# ---------------------------------------------------------------------------
# Snapshot Retrieval
# ---------------------------------------------------------------------------

class TestSnapshotRetrieval:
    """Test snapshot history retrieval."""

    @pytest.mark.asyncio
    async def test_get_snapshots_returns_list(self, tracker, sample_plan):
        await tracker.capture_snapshot(sample_plan)
        await tracker.capture_snapshot(sample_plan)
        snapshots = await tracker.get_snapshots("plan-001")
        assert isinstance(snapshots, list)
        assert len(snapshots) == 2

    @pytest.mark.asyncio
    async def test_get_snapshots_empty_for_unknown(self, tracker):
        snapshots = await tracker.get_snapshots("unknown-plan")
        assert snapshots == []

    @pytest.mark.asyncio
    async def test_get_snapshots_chronological(self, tracker, sample_plan):
        snap1 = await tracker.capture_snapshot(sample_plan)
        snap2 = await tracker.capture_snapshot(sample_plan)
        snapshots = await tracker.get_snapshots("plan-001")
        assert snapshots[0].snapshot_id == snap1.snapshot_id
        assert snapshots[1].snapshot_id == snap2.snapshot_id


# ---------------------------------------------------------------------------
# Effectiveness Review
# ---------------------------------------------------------------------------

class TestEffectivenessReview:
    """Test effectiveness review for completed actions."""

    @pytest.mark.asyncio
    async def test_review_effectiveness_returns_dict(self, tracker, sample_plan):
        result = await tracker.review_effectiveness(sample_plan)
        assert isinstance(result, dict)
        assert "plan_id" in result

    @pytest.mark.asyncio
    async def test_review_effectiveness_empty_plan(self, tracker):
        plan = ImprovementPlan(
            plan_id="plan-empty",
            operator_id="operator-001",
            title="Empty Plan",
            status=PlanStatus.DRAFT,
            actions=[],
            provenance_hash="p" * 64,
        )
        result = await tracker.review_effectiveness(plan)
        assert result["verified_count"] == 0

    @pytest.mark.asyncio
    async def test_review_effectiveness_counts_verified(self, tracker):
        now = datetime.now(tz=timezone.utc)
        actions = [
            ImprovementAction(
                action_id="act-001",
                plan_id="plan-001",
                gap_id="gap-001",
                action_type=ActionType.CORRECTIVE,
                title="Action 1",
                status=ActionStatus.VERIFIED,
                effectiveness_score=Decimal("85.00"),
                priority_score=Decimal("80.00"),
                eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
                urgency_score=Decimal("85.00"),
                importance_score=Decimal("75.00"),
                provenance_hash="a" * 64,
            ),
            ImprovementAction(
                action_id="act-002",
                plan_id="plan-001",
                gap_id="gap-001",
                action_type=ActionType.CORRECTIVE,
                title="Action 2",
                status=ActionStatus.CLOSED,
                effectiveness_score=Decimal("75.00"),
                priority_score=Decimal("80.00"),
                eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
                urgency_score=Decimal("85.00"),
                importance_score=Decimal("75.00"),
                provenance_hash="b" * 64,
            ),
        ]
        plan = ImprovementPlan(
            plan_id="plan-001",
            operator_id="operator-001",
            title="Test Plan",
            status=PlanStatus.ACTIVE,
            actions=actions,
            provenance_hash="p" * 64,
        )
        result = await tracker.review_effectiveness(plan)
        assert result["verified_count"] == 2

    @pytest.mark.asyncio
    async def test_review_effectiveness_calculates_avg(self, tracker):
        actions = [
            ImprovementAction(
                action_id="act-001",
                plan_id="plan-001",
                gap_id="gap-001",
                action_type=ActionType.CORRECTIVE,
                title="Action 1",
                status=ActionStatus.VERIFIED,
                effectiveness_score=Decimal("80.00"),
                priority_score=Decimal("80.00"),
                eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
                urgency_score=Decimal("85.00"),
                importance_score=Decimal("75.00"),
                provenance_hash="a" * 64,
            ),
            ImprovementAction(
                action_id="act-002",
                plan_id="plan-001",
                gap_id="gap-001",
                action_type=ActionType.CORRECTIVE,
                title="Action 2",
                status=ActionStatus.VERIFIED,
                effectiveness_score=Decimal("90.00"),
                priority_score=Decimal("80.00"),
                eisenhower_quadrant=EisenhowerQuadrant.DO_FIRST,
                urgency_score=Decimal("85.00"),
                importance_score=Decimal("75.00"),
                provenance_hash="b" * 64,
            ),
        ]
        plan = ImprovementPlan(
            plan_id="plan-001",
            operator_id="operator-001",
            title="Test Plan",
            status=PlanStatus.ACTIVE,
            actions=actions,
            provenance_hash="p" * 64,
        )
        result = await tracker.review_effectiveness(plan)
        # Average of 80 and 90 is 85
        assert result["avg_score"] == "85.00"


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test engine health check."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self, tracker):
        result = await tracker.health_check()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_health_check_has_status(self, tracker):
        result = await tracker.health_check()
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_has_engine_name(self, tracker):
        result = await tracker.health_check()
        assert result["engine"] == "ProgressTracker"
