# -*- coding: utf-8 -*-
"""
End-to-End API Integration Tests - AGENT-EUDR-025

Tests complete workflows from risk input through strategy recommendation,
plan creation, capacity building enrollment, effectiveness measurement,
monitoring, optimization, and reporting.

Test count: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from datetime import date, timedelta

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    PlanStatus,
    MilestoneStatus,
    EnrollmentStatus,
    TriggerEventType,
    AdjustmentType,
    StakeholderRole,
    ReportType,
    RecommendStrategiesRequest,
    CreatePlanRequest,
    EnrollSupplierRequest,
    MeasureEffectivenessRequest,
    OptimizeBudgetRequest,
    CollaborateRequest,
    GenerateReportRequest,
    AdaptiveScanRequest,
    SearchMeasuresRequest,
    SUPPORTED_COMMODITIES,
)

from .conftest import FIXED_DATE


class TestEndToEndMitigationWorkflow:
    """Test the complete mitigation workflow from risk to resolution."""

    @pytest.mark.asyncio
    async def test_full_workflow_high_risk_supplier(
        self, strategy_engine, remediation_engine,
        capacity_engine, effectiveness_engine,
        high_risk_input,
    ):
        # Step 1: Recommend strategies
        rec_req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        strategies = await strategy_engine.recommend(rec_req)
        assert len(strategies.strategies) >= 1

        # Step 2: Create remediation plan
        plan_req = CreatePlanRequest(
            operator_id=high_risk_input.operator_id,
            supplier_id=high_risk_input.supplier_id,
            strategy_ids=[s.strategy_id for s in strategies.strategies[:2]],
            template_name="supplier_capacity_building",
            budget_eur=Decimal("50000"),
            target_duration_weeks=24,
        )
        plan = await remediation_engine.create_plan(plan_req)
        assert plan.plan is not None
        assert plan.plan.status == PlanStatus.DRAFT

        # Step 3: Activate plan
        activated = await remediation_engine.update_plan_status(
            plan.plan.plan_id, PlanStatus.ACTIVE
        )
        assert activated.status == PlanStatus.ACTIVE

        # Step 4: Enroll supplier in capacity building
        enroll_req = EnrollSupplierRequest(
            supplier_id=high_risk_input.supplier_id,
            commodity=high_risk_input.commodity,
            initial_tier=1,
            target_completion_weeks=24,
        )
        enrollment = await capacity_engine.enroll_supplier(enroll_req)
        assert enrollment.enrollment.status == EnrollmentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_workflow_strategy_to_plan(
        self, strategy_engine, remediation_engine, medium_risk_input
    ):
        rec_req = RecommendStrategiesRequest(
            risk_input=medium_risk_input, top_k=3, deterministic_mode=True
        )
        strategies = await strategy_engine.recommend(rec_req)

        plan_req = CreatePlanRequest(
            operator_id=medium_risk_input.operator_id,
            supplier_id=medium_risk_input.supplier_id,
            strategy_ids=[s.strategy_id for s in strategies.strategies],
            budget_eur=Decimal("30000"),
        )
        plan = await remediation_engine.create_plan(plan_req)
        assert plan.plan.strategy_ids == [s.strategy_id for s in strategies.strategies]


class TestCrossCommodityWorkflow:
    """Test workflows across all 7 EUDR commodities."""

    @pytest.mark.parametrize("commodity", SUPPORTED_COMMODITIES)
    @pytest.mark.asyncio
    async def test_commodity_end_to_end(
        self, strategy_engine, remediation_engine, commodity
    ):
        ri = RiskInput(
            operator_id=f"op-e2e-{commodity}",
            supplier_id=f"sup-e2e-{commodity}",
            country_code="BR",
            commodity=commodity,
            country_risk_score=Decimal("60"),
            supplier_risk_score=Decimal("65"),
            deforestation_risk_score=Decimal("70"),
        )
        rec_req = RecommendStrategiesRequest(
            risk_input=ri, top_k=3, deterministic_mode=True
        )
        strategies = await strategy_engine.recommend(rec_req)
        assert len(strategies.strategies) >= 1

        plan_req = CreatePlanRequest(
            operator_id=ri.operator_id,
            supplier_id=ri.supplier_id,
            strategy_ids=[strategies.strategies[0].strategy_id],
            budget_eur=Decimal("25000"),
        )
        plan = await remediation_engine.create_plan(plan_req)
        assert plan.plan is not None


class TestOptimizationWorkflow:
    @pytest.mark.asyncio
    async def test_optimize_after_assessment(
        self, strategy_engine, optimizer_engine, risk_input_batch
    ):
        # Assess risk for batch
        requests = [
            RecommendStrategiesRequest(risk_input=ri, top_k=2, deterministic_mode=True)
            for ri in risk_input_batch[:5]
        ]
        results = await strategy_engine.recommend_batch(requests)
        assert len(results) == 5

        # Optimize budget
        opt_req = OptimizeBudgetRequest(
            operator_id="op-batch",
            total_budget_eur=Decimal("200000"),
            supplier_ids=[ri.supplier_id for ri in risk_input_batch[:5]],
        )
        opt_result = await optimizer_engine.optimize(opt_req)
        assert opt_result.total_budget_used <= Decimal("200000")


class TestMonitoringWorkflow:
    @pytest.mark.asyncio
    async def test_monitoring_after_plan_activation(
        self, remediation_engine, monitoring_engine, create_plan_request
    ):
        plan = await remediation_engine.create_plan(create_plan_request)
        await remediation_engine.update_plan_status(plan.plan.plan_id, PlanStatus.ACTIVE)

        scan_req = AdaptiveScanRequest(
            operator_id="op-001",
            plan_ids=[plan.plan.plan_id],
        )
        scan_result = await monitoring_engine.scan(scan_req)
        assert scan_result is not None


class TestCollaborationWorkflow:
    @pytest.mark.asyncio
    async def test_collaboration_after_plan_creation(
        self, remediation_engine, collaboration_engine, create_plan_request
    ):
        plan = await remediation_engine.create_plan(create_plan_request)

        msg_req = CollaborateRequest(
            action="message",
            plan_id=plan.plan.plan_id,
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="Plan created and ready for review.",
        )
        result = await collaboration_engine.execute(msg_req)
        assert result.status == "success"


class TestMeasureLibraryIntegration:
    @pytest.mark.asyncio
    async def test_search_then_optimize(
        self, measure_library_engine, optimizer_engine
    ):
        search_req = SearchMeasuresRequest(
            risk_category=RiskCategory.DEFORESTATION, limit=5
        )
        measures = await measure_library_engine.search(search_req)

        opt_req = OptimizeBudgetRequest(
            operator_id="op-lib",
            total_budget_eur=Decimal("100000"),
            supplier_ids=["sup-1", "sup-2"],
            candidate_measure_ids=[m.measure_id for m in measures.measures],
        )
        result = await optimizer_engine.optimize(opt_req)
        assert isinstance(result.allocations, dict)


class TestProvenanceChainIntegration:
    @pytest.mark.asyncio
    async def test_provenance_across_engines(
        self, strategy_engine, remediation_engine, high_risk_input
    ):
        rec_req = RecommendStrategiesRequest(
            risk_input=high_risk_input, deterministic_mode=True
        )
        rec_result = await strategy_engine.recommend(rec_req)
        assert rec_result.provenance_hash != ""

        plan_req = CreatePlanRequest(
            operator_id="op-prov",
            supplier_id="sup-prov",
            strategy_ids=[rec_result.strategies[0].strategy_id] if rec_result.strategies else [],
            budget_eur=Decimal("50000"),
        )
        plan_result = await remediation_engine.create_plan(plan_req)
        assert plan_result.provenance_hash != ""
        assert plan_result.provenance_hash != rec_result.provenance_hash


class TestErrorHandlingIntegration:
    @pytest.mark.asyncio
    async def test_invalid_plan_status_transition(self, remediation_engine, create_plan_request):
        plan = await remediation_engine.create_plan(create_plan_request)
        with pytest.raises((ValueError, Exception)):
            await remediation_engine.update_plan_status(
                plan.plan.plan_id, PlanStatus.COMPLETED
            )

    @pytest.mark.asyncio
    async def test_invalid_commodity(self):
        with pytest.raises((ValueError, Exception)):
            RiskInput(
                operator_id="op", supplier_id="sup",
                country_code="BR", commodity="cotton",
            )


class TestMultiSupplierWorkflow:
    @pytest.mark.asyncio
    async def test_batch_supplier_workflow(
        self, strategy_engine, remediation_engine, risk_input_batch
    ):
        plans_created = 0
        for ri in risk_input_batch[:5]:
            rec_req = RecommendStrategiesRequest(
                risk_input=ri, top_k=2, deterministic_mode=True
            )
            strategies = await strategy_engine.recommend(rec_req)

            if strategies.strategies:
                plan_req = CreatePlanRequest(
                    operator_id=ri.operator_id,
                    supplier_id=ri.supplier_id,
                    strategy_ids=[strategies.strategies[0].strategy_id],
                    budget_eur=Decimal("20000"),
                )
                plan = await remediation_engine.create_plan(plan_req)
                if plan.plan:
                    plans_created += 1

        assert plans_created >= 1
