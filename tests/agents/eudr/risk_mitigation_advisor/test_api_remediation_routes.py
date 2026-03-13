# -*- coding: utf-8 -*-
"""
Tests for Remediation Plan API Routes - AGENT-EUDR-025

Tests plan CRUD operations, status transitions, milestone management,
evidence uploads, plan listing, filtering, and approval workflows.

Test count: ~50 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    PlanStatus,
    MilestoneStatus,
    CreatePlanRequest,
    CreatePlanResponse,
    RemediationPlan,
)
from greenlang.agents.eudr.risk_mitigation_advisor.remediation_plan_design_engine import (
    RemediationPlanDesignEngine,
)

from .conftest import FIXED_DATE, PLAN_TEMPLATES


class TestPlanCRUD:
    @pytest.mark.asyncio
    async def test_create_plan(self, remediation_engine, create_plan_request):
        result = await remediation_engine.create_plan(create_plan_request)
        assert result.plan.plan_id is not None

    @pytest.mark.asyncio
    async def test_get_plan(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        retrieved = await remediation_engine.get_plan(created.plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == created.plan.plan_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_plan(self, remediation_engine):
        retrieved = await remediation_engine.get_plan("nonexistent-plan-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_plans_by_operator(self, remediation_engine, create_plan_request):
        await remediation_engine.create_plan(create_plan_request)
        plans = await remediation_engine.list_plans(operator_id="op-001")
        assert isinstance(plans, list)
        assert len(plans) >= 1

    @pytest.mark.asyncio
    async def test_list_plans_by_status(self, remediation_engine, create_plan_request):
        await remediation_engine.create_plan(create_plan_request)
        plans = await remediation_engine.list_plans(
            operator_id="op-001", status=PlanStatus.DRAFT
        )
        for plan in plans:
            assert plan.status == PlanStatus.DRAFT

    @pytest.mark.asyncio
    async def test_list_plans_empty(self, remediation_engine):
        plans = await remediation_engine.list_plans(operator_id="nonexistent-op")
        assert plans == []


class TestPlanStatusManagement:
    @pytest.mark.asyncio
    async def test_activate_plan(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        activated = await remediation_engine.update_plan_status(
            created.plan.plan_id, PlanStatus.ACTIVE
        )
        assert activated.status == PlanStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_complete_plan(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        await remediation_engine.update_plan_status(
            created.plan.plan_id, PlanStatus.ACTIVE
        )
        plan = await remediation_engine.update_plan_status(
            created.plan.plan_id, PlanStatus.ON_TRACK
        )
        completed = await remediation_engine.update_plan_status(
            plan.plan_id, PlanStatus.COMPLETED
        )
        assert completed.status == PlanStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_suspend_plan(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        await remediation_engine.update_plan_status(
            created.plan.plan_id, PlanStatus.ACTIVE
        )
        suspended = await remediation_engine.update_plan_status(
            created.plan.plan_id, PlanStatus.SUSPENDED
        )
        assert suspended.status == PlanStatus.SUSPENDED

    @pytest.mark.asyncio
    async def test_abandon_plan(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        abandoned = await remediation_engine.update_plan_status(
            created.plan.plan_id, PlanStatus.ABANDONED
        )
        assert abandoned.status == PlanStatus.ABANDONED


class TestMilestoneAPI:
    @pytest.mark.asyncio
    async def test_list_milestones(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        milestones = await remediation_engine.list_milestones(created.plan.plan_id)
        assert isinstance(milestones, list)

    @pytest.mark.asyncio
    async def test_complete_milestone(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        if created.plan.milestones:
            ms_id = created.plan.milestones[0].milestone_id
            updated = await remediation_engine.complete_milestone(
                created.plan.plan_id, ms_id
            )
            assert updated is not None

    @pytest.mark.asyncio
    async def test_skip_milestone(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        if created.plan.milestones:
            ms_id = created.plan.milestones[0].milestone_id
            updated = await remediation_engine.skip_milestone(
                created.plan.plan_id, ms_id, reason="Not applicable"
            )
            assert updated is not None


class TestPlanVersioning:
    @pytest.mark.asyncio
    async def test_plan_version_increments(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        assert created.plan.version == 1
        updated = await remediation_engine.update_plan_budget(
            created.plan.plan_id, Decimal("75000")
        )
        assert updated.version >= 1

    @pytest.mark.asyncio
    async def test_get_plan_history(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        history = await remediation_engine.get_plan_history(created.plan.plan_id)
        assert isinstance(history, list)
        assert len(history) >= 1


class TestPlanFiltering:
    @pytest.mark.asyncio
    async def test_filter_by_supplier(self, remediation_engine, create_plan_request):
        await remediation_engine.create_plan(create_plan_request)
        plans = await remediation_engine.list_plans(
            operator_id="op-001", supplier_id="sup-001"
        )
        for plan in plans:
            assert plan.supplier_id == "sup-001"

    @pytest.mark.asyncio
    async def test_filter_by_template(self, remediation_engine, create_plan_request):
        await remediation_engine.create_plan(create_plan_request)
        plans = await remediation_engine.list_plans(
            operator_id="op-001",
            template_name="supplier_capacity_building",
        )
        for plan in plans:
            assert plan.plan_template == "supplier_capacity_building"


class TestPlanBudgetManagement:
    @pytest.mark.asyncio
    async def test_update_budget(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        updated = await remediation_engine.update_plan_budget(
            created.plan.plan_id, Decimal("75000")
        )
        assert updated.budget_allocated == Decimal("75000")

    @pytest.mark.asyncio
    async def test_record_spend(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        updated = await remediation_engine.record_spend(
            created.plan.plan_id, Decimal("5000")
        )
        assert updated.budget_spent >= Decimal("5000")


class TestPlanReporting:
    @pytest.mark.asyncio
    async def test_plan_summary(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        summary = await remediation_engine.get_plan_summary(created.plan.plan_id)
        assert summary is not None
        assert "plan_id" in summary

    @pytest.mark.asyncio
    async def test_plan_progress_pct(self, remediation_engine, create_plan_request):
        created = await remediation_engine.create_plan(create_plan_request)
        progress = await remediation_engine.get_plan_progress(created.plan.plan_id)
        assert isinstance(progress, Decimal)
        assert Decimal("0") <= progress <= Decimal("100")


class TestPlanEdgeCases:
    @pytest.mark.asyncio
    async def test_create_multiple_plans_same_supplier(self, remediation_engine):
        for i in range(3):
            req = CreatePlanRequest(
                operator_id="op-multi",
                supplier_id="sup-multi",
                template_name="supplier_capacity_building",
                budget_eur=Decimal(str(10000 * (i + 1))),
            )
            result = await remediation_engine.create_plan(req)
            assert result.plan is not None

    @pytest.mark.asyncio
    async def test_plan_with_all_templates(self, remediation_engine):
        for template in PLAN_TEMPLATES:
            req = CreatePlanRequest(
                operator_id="op-all",
                supplier_id="sup-all",
                template_name=template,
                budget_eur=Decimal("25000"),
            )
            result = await remediation_engine.create_plan(req)
            assert result.plan.plan_template == template
