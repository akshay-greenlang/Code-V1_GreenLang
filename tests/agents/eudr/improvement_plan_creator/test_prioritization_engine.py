# -*- coding: utf-8 -*-
"""
Unit tests for PrioritizationEngine - AGENT-EUDR-035

Tests action prioritization using Eisenhower matrix classification
and multi-factor risk-based scoring.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (Engine 4: Prioritization Engine)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.improvement_plan_creator.config import (
    ImprovementPlanCreatorConfig,
)
from greenlang.agents.eudr.improvement_plan_creator.prioritization_engine import (
    PrioritizationEngine,
)
from greenlang.agents.eudr.improvement_plan_creator.models import (
    ActionStatus,
    ActionType,
    EisenhowerQuadrant,
    EUDRCommodity,
    GapSeverity,
    ImprovementAction,
)


@pytest.fixture
def config():
    return ImprovementPlanCreatorConfig()


@pytest.fixture
def engine(config):
    return PrioritizationEngine(config=config)


# ---------------------------------------------------------------------------
# Prioritize Actions
# ---------------------------------------------------------------------------

class TestPrioritizeActions:
    """Test action prioritization and ranking."""

    @pytest.mark.asyncio
    async def test_prioritize_returns_sorted_list(self, engine, multiple_actions, multiple_gaps):
        ranked = await engine.prioritize_actions(
            actions=multiple_actions,
            gaps=multiple_gaps,
        )
        assert isinstance(ranked, list)
        assert len(ranked) == len(multiple_actions)

        # Verify sorted by priority_score descending
        for i in range(len(ranked) - 1):
            assert ranked[i].priority_score >= ranked[i + 1].priority_score

    @pytest.mark.asyncio
    async def test_prioritize_single_action(self, engine, sample_action, sample_gap):
        ranked = await engine.prioritize_actions(
            actions=[sample_action],
            gaps=[sample_gap],
        )
        assert len(ranked) == 1
        assert ranked[0].priority_score is not None

    @pytest.mark.asyncio
    async def test_prioritize_empty_actions(self, engine):
        ranked = await engine.prioritize_actions(
            actions=[],
            gaps=[],
        )
        assert ranked == []

    @pytest.mark.asyncio
    async def test_actions_have_priority_scores(self, engine, multiple_actions, multiple_gaps):
        ranked = await engine.prioritize_actions(
            actions=multiple_actions,
            gaps=multiple_gaps,
        )
        for action in ranked:
            assert action.priority_score is not None
            assert Decimal("0") <= action.priority_score <= Decimal("100")

    @pytest.mark.asyncio
    async def test_actions_have_urgency_scores(self, engine, multiple_actions, multiple_gaps):
        ranked = await engine.prioritize_actions(
            actions=multiple_actions,
            gaps=multiple_gaps,
        )
        for action in ranked:
            assert action.urgency_score is not None
            assert Decimal("0") <= action.urgency_score <= Decimal("100")

    @pytest.mark.asyncio
    async def test_actions_have_importance_scores(self, engine, multiple_actions, multiple_gaps):
        ranked = await engine.prioritize_actions(
            actions=multiple_actions,
            gaps=multiple_gaps,
        )
        for action in ranked:
            assert action.importance_score is not None
            assert Decimal("0") <= action.importance_score <= Decimal("100")


# ---------------------------------------------------------------------------
# Eisenhower Matrix Classification
# ---------------------------------------------------------------------------

class TestEisenhowerQuadrant:
    """Test Eisenhower matrix quadrant assignment."""

    @pytest.mark.asyncio
    async def test_actions_have_quadrant_assigned(self, engine, multiple_actions, multiple_gaps):
        ranked = await engine.prioritize_actions(
            actions=multiple_actions,
            gaps=multiple_gaps,
        )
        for action in ranked:
            assert action.eisenhower_quadrant is not None
            assert isinstance(action.eisenhower_quadrant, EisenhowerQuadrant)

    @pytest.mark.asyncio
    async def test_critical_gap_gets_do_first(self, engine, multiple_gaps):
        # Find the critical gap
        critical_gap = next(g for g in multiple_gaps if g.severity == GapSeverity.CRITICAL)

        # Create action for critical gap
        action = ImprovementAction(
            action_id="act-critical",
            plan_id="plan-001",
            gap_id=critical_gap.gap_id,
            action_type=ActionType.CORRECTIVE,
            title="Fix critical issue",
            description="Urgent and important",
            assigned_to="team@company.com",
            status=ActionStatus.PROPOSED,
            time_bound_deadline=datetime.now(tz=timezone.utc) + timedelta(days=7),
            estimated_effort_hours=Decimal("100"),
            estimated_cost=Decimal("10000"),
            provenance_hash="x" * 64,
        )

        ranked = await engine.prioritize_actions(
            actions=[action],
            gaps=[critical_gap],
        )

        # Critical severity + approaching deadline should be DO_FIRST
        assert ranked[0].eisenhower_quadrant == EisenhowerQuadrant.DO_FIRST



# ---------------------------------------------------------------------------
# Root Cause Impact on Prioritization
# ---------------------------------------------------------------------------

class TestRootCauseImpact:
    """Test root cause influence on priority scoring."""

    @pytest.mark.asyncio
    async def test_systemic_root_cause_boosts_importance(self, engine, sample_action, sample_gap, sample_root_cause):
        # Make root cause systemic
        sample_root_cause.systemic = True
        sample_root_cause.gap_id = sample_gap.gap_id

        ranked_without = await engine.prioritize_actions(
            actions=[sample_action],
            gaps=[sample_gap],
            root_causes=None,
        )

        importance_without = ranked_without[0].importance_score

        # Reset action for second test
        sample_action.importance_score = None
        sample_action.priority_score = None

        ranked_with = await engine.prioritize_actions(
            actions=[sample_action],
            gaps=[sample_gap],
            root_causes=[sample_root_cause],
        )

        importance_with = ranked_with[0].importance_score

        # Systemic root cause should boost importance
        assert importance_with >= importance_without


# ---------------------------------------------------------------------------
# Deadline Impact on Urgency
# ---------------------------------------------------------------------------

class TestDeadlineImpact:
    """Test deadline proximity impact on urgency scoring."""

    @pytest.mark.asyncio
    async def test_approaching_deadline_boosts_urgency(self, engine, sample_gap):
        # Action with near deadline
        action_near = ImprovementAction(
            action_id="act-near",
            plan_id="plan-001",
            gap_id=sample_gap.gap_id,
            action_type=ActionType.CORRECTIVE,
            title="Near deadline action",
            description="Due soon",
            assigned_to="team@company.com",
            status=ActionStatus.PROPOSED,
            time_bound_deadline=datetime.now(tz=timezone.utc) + timedelta(days=5),  # Within 7 days
            estimated_effort_hours=Decimal("100"),
            estimated_cost=Decimal("10000"),
            provenance_hash="x" * 64,
        )

        # Action with far deadline
        action_far = ImprovementAction(
            action_id="act-far",
            plan_id="plan-001",
            gap_id=sample_gap.gap_id,
            action_type=ActionType.CORRECTIVE,
            title="Far deadline action",
            description="Due later",
            assigned_to="team@company.com",
            status=ActionStatus.PROPOSED,
            time_bound_deadline=datetime.now(tz=timezone.utc) + timedelta(days=90),
            estimated_effort_hours=Decimal("100"),
            estimated_cost=Decimal("10000"),
            provenance_hash="y" * 64,
        )

        ranked = await engine.prioritize_actions(
            actions=[action_near, action_far],
            gaps=[sample_gap],
        )

        near_urgency = next(a for a in ranked if a.action_id == "act-near").urgency_score
        far_urgency = next(a for a in ranked if a.action_id == "act-far").urgency_score

        # Near deadline should have higher urgency
        assert near_urgency > far_urgency


# ---------------------------------------------------------------------------
# Resource Efficiency
# ---------------------------------------------------------------------------

class TestResourceEfficiency:
    """Test resource efficiency impact on scoring."""

    @pytest.mark.asyncio
    async def test_lower_cost_higher_efficiency(self, engine, sample_gap):
        # Low cost action
        action_cheap = ImprovementAction(
            action_id="act-cheap",
            plan_id="plan-001",
            gap_id=sample_gap.gap_id,
            action_type=ActionType.CORRECTIVE,
            title="Low cost action",
            description="Cheap fix",
            assigned_to="team@company.com",
            status=ActionStatus.PROPOSED,
            time_bound_deadline=datetime.now(tz=timezone.utc) + timedelta(days=30),
            estimated_effort_hours=Decimal("10"),
            estimated_cost=Decimal("1000"),
            provenance_hash="x" * 64,
        )

        # High cost action
        action_expensive = ImprovementAction(
            action_id="act-expensive",
            plan_id="plan-001",
            gap_id=sample_gap.gap_id,
            action_type=ActionType.CORRECTIVE,
            title="High cost action",
            description="Expensive fix",
            assigned_to="team@company.com",
            status=ActionStatus.PROPOSED,
            time_bound_deadline=datetime.now(tz=timezone.utc) + timedelta(days=30),
            estimated_effort_hours=Decimal("200"),
            estimated_cost=Decimal("50000"),
            provenance_hash="y" * 64,
        )

        ranked = await engine.prioritize_actions(
            actions=[action_cheap, action_expensive],
            gaps=[sample_gap],
        )

        # With same gap severity and deadline, cheaper should rank higher
        cheap_score = next(a for a in ranked if a.action_id == "act-cheap").priority_score
        expensive_score = next(a for a in ranked if a.action_id == "act-expensive").priority_score

        assert cheap_score > expensive_score


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """Test engine health check."""

    @pytest.mark.asyncio
    async def test_health_check_returns_dict(self, engine):
        health = await engine.health_check()
        assert isinstance(health, dict)
        assert "engine" in health
        assert health["engine"] == "PrioritizationEngine"

    @pytest.mark.asyncio
    async def test_health_check_has_status(self, engine):
        health = await engine.health_check()
        assert "status" in health
        assert health["status"] == "healthy"
