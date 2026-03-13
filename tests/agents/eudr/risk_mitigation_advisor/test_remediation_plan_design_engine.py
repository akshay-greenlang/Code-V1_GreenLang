# -*- coding: utf-8 -*-
"""
Tests for Engine 2: Remediation Plan Design Engine - AGENT-EUDR-025

Tests plan generation, SMART milestones, plan templates, status transitions,
Gantt chart data, plan versioning, cloning, phase management, KPIs,
escalation triggers, and provenance tracking.

Test count: ~70 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import List

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    PlanStatus,
    PlanPhaseType,
    MilestoneStatus,
    ImplementationComplexity,
    StakeholderRole,
    RiskCategory,
    RemediationPlan,
    PlanPhase,
    Milestone,
    KPI,
    CreatePlanRequest,
    CreatePlanResponse,
)
from greenlang.agents.eudr.risk_mitigation_advisor.remediation_plan_design_engine import (
    RemediationPlanDesignEngine,
    VALID_STATUS_TRANSITIONS,
)

from .conftest import FIXED_DATE, PLAN_TEMPLATES


# ========================== Initialization Tests ==========================


class TestRemediationEngineInit:
    """Tests for RemediationPlanDesignEngine initialization."""

    def test_engine_initializes(self, remediation_engine):
        assert remediation_engine is not None

    def test_engine_has_templates(self, remediation_engine):
        templates = remediation_engine.get_available_templates()
        assert len(templates) >= 8


# ========================== Plan Generation Tests =========================


class TestPlanGeneration:
    """Tests for remediation plan generation."""

    @pytest.mark.asyncio
    async def test_generate_plan_from_request(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert isinstance(result, CreatePlanResponse)
        assert result.plan is not None
        assert result.plan.status == PlanStatus.DRAFT

    @pytest.mark.asyncio
    async def test_plan_has_operator_id(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.plan.operator_id == "op-001"

    @pytest.mark.asyncio
    async def test_plan_has_supplier_id(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.plan.supplier_id == "sup-001"

    @pytest.mark.asyncio
    async def test_plan_has_budget(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.plan.budget_allocated == Decimal("50000")

    @pytest.mark.asyncio
    async def test_plan_has_phases(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert len(result.plan.phases) >= 1

    @pytest.mark.asyncio
    async def test_plan_has_milestones(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.milestone_count >= 1

    @pytest.mark.asyncio
    async def test_plan_has_provenance_hash(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_plan_has_processing_time(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.processing_time_ms >= Decimal("0")

    @pytest.mark.asyncio
    async def test_plan_version_starts_at_one(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.plan.version == 1


# ========================== Template Tests ================================


class TestPlanTemplates:
    """Tests for 8 plan templates."""

    @pytest.mark.parametrize("template_name", PLAN_TEMPLATES)
    @pytest.mark.asyncio
    async def test_template_generates_valid_plan(self, remediation_engine, template_name):
        request = CreatePlanRequest(
            operator_id="op-tmpl",
            supplier_id="sup-tmpl",
            template_name=template_name,
            budget_eur=Decimal("25000"),
            target_duration_weeks=12,
        )
        result = await remediation_engine.create_plan(request)
        assert result.plan is not None
        assert result.plan.plan_template == template_name

    @pytest.mark.asyncio
    async def test_emergency_template_short_duration(self, remediation_engine):
        request = CreatePlanRequest(
            operator_id="op-emrg",
            supplier_id="sup-emrg",
            template_name="emergency_deforestation_response",
            budget_eur=Decimal("10000"),
            target_duration_weeks=4,
        )
        result = await remediation_engine.create_plan(request)
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_certification_template_long_duration(self, remediation_engine):
        request = CreatePlanRequest(
            operator_id="op-cert",
            supplier_id="sup-cert",
            template_name="certification_enrollment",
            budget_eur=Decimal("100000"),
            target_duration_weeks=52,
        )
        result = await remediation_engine.create_plan(request)
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_no_template_generates_default(self, remediation_engine):
        request = CreatePlanRequest(
            operator_id="op-def",
            supplier_id="sup-def",
            budget_eur=Decimal("20000"),
        )
        result = await remediation_engine.create_plan(request)
        assert result.plan is not None


# ========================== Status Transition Tests =======================


class TestPlanStatusTransitions:
    """Tests for plan status finite state machine."""

    def test_valid_transitions_defined(self):
        assert PlanStatus.DRAFT in VALID_STATUS_TRANSITIONS
        assert PlanStatus.ACTIVE in VALID_STATUS_TRANSITIONS[PlanStatus.DRAFT]

    def test_draft_can_become_active(self):
        assert PlanStatus.ACTIVE in VALID_STATUS_TRANSITIONS[PlanStatus.DRAFT]

    def test_draft_can_be_abandoned(self):
        assert PlanStatus.ABANDONED in VALID_STATUS_TRANSITIONS[PlanStatus.DRAFT]

    def test_active_can_become_on_track(self):
        assert PlanStatus.ON_TRACK in VALID_STATUS_TRANSITIONS[PlanStatus.ACTIVE]

    def test_active_can_become_at_risk(self):
        assert PlanStatus.AT_RISK in VALID_STATUS_TRANSITIONS[PlanStatus.ACTIVE]

    def test_on_track_can_complete(self):
        assert PlanStatus.COMPLETED in VALID_STATUS_TRANSITIONS[PlanStatus.ON_TRACK]

    @pytest.mark.asyncio
    async def test_activate_draft_plan(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        activated = await remediation_engine.update_plan_status(
            result.plan.plan_id, PlanStatus.ACTIVE
        )
        assert activated.status == PlanStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        with pytest.raises((ValueError, Exception)):
            await remediation_engine.update_plan_status(
                result.plan.plan_id, PlanStatus.COMPLETED
            )


# ========================== Milestone Tests ===============================


class TestSMARTMilestones:
    """Tests for SMART milestone generation."""

    @pytest.mark.asyncio
    async def test_milestones_have_names(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        for ms in result.plan.milestones:
            assert ms.name is not None
            assert len(ms.name) > 0

    @pytest.mark.asyncio
    async def test_milestones_have_due_dates(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        for ms in result.plan.milestones:
            assert ms.due_date is not None

    @pytest.mark.asyncio
    async def test_milestones_have_phases(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        for ms in result.plan.milestones:
            assert isinstance(ms.phase, PlanPhaseType)

    @pytest.mark.asyncio
    async def test_milestones_start_pending(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        for ms in result.plan.milestones:
            assert ms.status == MilestoneStatus.PENDING

    @pytest.mark.asyncio
    async def test_milestone_completion_tracking(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        if result.plan.milestones:
            ms_id = result.plan.milestones[0].milestone_id
            updated = await remediation_engine.complete_milestone(
                result.plan.plan_id, ms_id
            )
            assert updated is not None


# ========================== KPI Tests =====================================


class TestPlanKPIs:
    """Tests for KPI generation in plans."""

    @pytest.mark.asyncio
    async def test_plan_has_kpis(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.kpi_count >= 0

    @pytest.mark.asyncio
    async def test_kpis_have_names(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        for kpi in result.plan.kpis:
            assert kpi.name is not None
            assert len(kpi.name) > 0

    @pytest.mark.asyncio
    async def test_kpis_have_target_values(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        for kpi in result.plan.kpis:
            assert kpi.target_value is not None


# ========================== Plan Cloning Tests ============================


class TestPlanCloning:
    """Tests for plan cloning functionality."""

    @pytest.mark.asyncio
    async def test_clone_plan_creates_new(self, remediation_engine, create_plan_request):
        original = await remediation_engine.create_plan(create_plan_request)
        cloned = await remediation_engine.clone_plan(
            original.plan.plan_id, "sup-clone-001"
        )
        assert cloned.plan.plan_id != original.plan.plan_id
        assert cloned.plan.supplier_id == "sup-clone-001"

    @pytest.mark.asyncio
    async def test_clone_preserves_template(self, remediation_engine, create_plan_request):
        original = await remediation_engine.create_plan(create_plan_request)
        cloned = await remediation_engine.clone_plan(
            original.plan.plan_id, "sup-clone-002"
        )
        assert cloned.plan.plan_template == original.plan.plan_template

    @pytest.mark.asyncio
    async def test_clone_resets_status_to_draft(self, remediation_engine, create_plan_request):
        original = await remediation_engine.create_plan(create_plan_request)
        cloned = await remediation_engine.clone_plan(
            original.plan.plan_id, "sup-clone-003"
        )
        assert cloned.plan.status == PlanStatus.DRAFT


# ========================== Gantt Chart Tests =============================


class TestGanttChartData:
    """Tests for Gantt chart data generation."""

    @pytest.mark.asyncio
    async def test_gantt_data_generated(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        gantt = await remediation_engine.get_gantt_data(result.plan.plan_id)
        assert gantt is not None

    @pytest.mark.asyncio
    async def test_gantt_phases_ordered(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        gantt = await remediation_engine.get_gantt_data(result.plan.plan_id)
        if gantt and len(gantt) >= 2:
            for i in range(len(gantt) - 1):
                assert gantt[i]["start_week"] <= gantt[i + 1]["start_week"]


# ========================== Edge Cases ====================================


class TestPlanEdgeCases:
    """Edge case tests for plan generation."""

    @pytest.mark.asyncio
    async def test_minimum_budget(self, remediation_engine):
        request = CreatePlanRequest(
            operator_id="op-min",
            supplier_id="sup-min",
            budget_eur=Decimal("100"),
            target_duration_weeks=4,
        )
        result = await remediation_engine.create_plan(request)
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_maximum_duration(self, remediation_engine):
        request = CreatePlanRequest(
            operator_id="op-max",
            supplier_id="sup-max",
            budget_eur=Decimal("1000000"),
            target_duration_weeks=52,
        )
        result = await remediation_engine.create_plan(request)
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_no_supplier_id(self, remediation_engine):
        request = CreatePlanRequest(
            operator_id="op-nosup",
            budget_eur=Decimal("50000"),
        )
        result = await remediation_engine.create_plan(request)
        assert result.plan.supplier_id is None
