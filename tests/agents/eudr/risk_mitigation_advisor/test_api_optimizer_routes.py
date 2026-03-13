# -*- coding: utf-8 -*-
"""
Tests for Optimizer API Routes - AGENT-EUDR-025

Tests budget optimization endpoints, scenario analysis, Pareto generation,
RICE scoring, spend tracking, and input validation.

Test count: ~40 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    OptimizeBudgetRequest,
    OptimizeBudgetResponse,
)
from greenlang.agents.eudr.risk_mitigation_advisor.cost_benefit_optimizer_engine import (
    CostBenefitOptimizerEngine,
)


class TestOptimizationEndpoint:
    @pytest.mark.asyncio
    async def test_optimize_valid_request(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert isinstance(result, OptimizeBudgetResponse)

    @pytest.mark.asyncio
    async def test_optimize_budget_constraint(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert result.total_budget_used <= optimize_budget_request.total_budget_eur

    @pytest.mark.asyncio
    async def test_optimize_has_allocations(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert isinstance(result.allocations, dict)

    @pytest.mark.asyncio
    async def test_optimize_solver_status_valid(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert result.solver_status in ("optimal", "feasible", "infeasible", "greedy_fallback")

    @pytest.mark.asyncio
    async def test_optimize_provenance(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert len(result.provenance_hash) == 64


class TestScenarioEndpoint:
    @pytest.mark.asyncio
    async def test_run_scenarios(self, optimizer_engine, optimize_budget_request):
        scenarios = await optimizer_engine.run_scenarios(
            optimize_budget_request,
            budget_multipliers=[Decimal("0.5"), Decimal("1.0"), Decimal("1.5"), Decimal("2.0")],
        )
        assert len(scenarios) == 4

    @pytest.mark.asyncio
    async def test_scenario_increasing_budget(self, optimizer_engine, optimize_budget_request):
        scenarios = await optimizer_engine.run_scenarios(
            optimize_budget_request,
            budget_multipliers=[Decimal("1.0"), Decimal("2.0")],
        )
        assert scenarios[0]["expected_risk_reduction"] <= scenarios[1]["expected_risk_reduction"]

    @pytest.mark.asyncio
    async def test_scenario_empty_multipliers(self, optimizer_engine, optimize_budget_request):
        scenarios = await optimizer_engine.run_scenarios(
            optimize_budget_request, budget_multipliers=[]
        )
        assert scenarios == []


class TestParetoEndpoint:
    @pytest.mark.asyncio
    async def test_pareto_generated(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert isinstance(result.pareto_frontier, list)
        assert len(result.pareto_frontier) >= 1

    @pytest.mark.asyncio
    async def test_pareto_points_valid(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        for point in result.pareto_frontier:
            assert "budget" in point or "risk_reduction" in point


class TestRICEEndpoint:
    @pytest.mark.asyncio
    async def test_rice_scoring(self, optimizer_engine):
        score = await optimizer_engine.calculate_rice_score(
            reach=Decimal("200"),
            impact=Decimal("40"),
            confidence=Decimal("0.9"),
            effort=Decimal("25000"),
        )
        assert isinstance(score, Decimal)
        assert score > Decimal("0")

    @pytest.mark.asyncio
    async def test_rice_zero_effort(self, optimizer_engine):
        score = await optimizer_engine.calculate_rice_score(
            reach=Decimal("100"),
            impact=Decimal("30"),
            confidence=Decimal("0.8"),
            effort=Decimal("0"),
        )
        assert score is None or isinstance(score, Decimal)


class TestSpendTrackingEndpoint:
    @pytest.mark.asyncio
    async def test_variance_calculation(self, optimizer_engine):
        variance = await optimizer_engine.calculate_spend_variance(
            planned=Decimal("100000"),
            actual=Decimal("85000"),
        )
        assert variance == Decimal("-15000")

    @pytest.mark.asyncio
    async def test_variance_overspend(self, optimizer_engine):
        variance = await optimizer_engine.calculate_spend_variance(
            planned=Decimal("100000"),
            actual=Decimal("120000"),
        )
        assert variance == Decimal("20000")

    @pytest.mark.asyncio
    async def test_variance_exact(self, optimizer_engine):
        variance = await optimizer_engine.calculate_spend_variance(
            planned=Decimal("50000"),
            actual=Decimal("50000"),
        )
        assert variance == Decimal("0")


class TestOptimizerInputValidation:
    @pytest.mark.asyncio
    async def test_negative_budget_rejected(self):
        with pytest.raises((ValueError, Exception)):
            OptimizeBudgetRequest(
                operator_id="op",
                total_budget_eur=Decimal("-1000"),
            )

    @pytest.mark.asyncio
    async def test_empty_operator_rejected(self):
        with pytest.raises((ValueError, Exception)):
            OptimizeBudgetRequest(
                operator_id="",
                total_budget_eur=Decimal("50000"),
            )


class TestOptimizerDeterminism:
    @pytest.mark.asyncio
    async def test_same_input_same_output(self, optimizer_engine, optimize_budget_request):
        r1 = await optimizer_engine.optimize(optimize_budget_request)
        r2 = await optimizer_engine.optimize(optimize_budget_request)
        assert r1.total_budget_used == r2.total_budget_used
        assert r1.expected_risk_reduction == r2.expected_risk_reduction
        assert r1.provenance_hash == r2.provenance_hash


class TestOptimizerEdgeCases:
    @pytest.mark.asyncio
    async def test_very_small_budget(self, optimizer_engine):
        request = OptimizeBudgetRequest(
            operator_id="op-tiny",
            total_budget_eur=Decimal("10"),
            supplier_ids=["sup-1"],
            candidate_measure_ids=["meas-1"],
        )
        result = await optimizer_engine.optimize(request)
        assert isinstance(result, OptimizeBudgetResponse)

    @pytest.mark.asyncio
    async def test_many_suppliers(self, optimizer_engine):
        request = OptimizeBudgetRequest(
            operator_id="op-many",
            total_budget_eur=Decimal("5000000"),
            supplier_ids=[f"sup-{i}" for i in range(200)],
            candidate_measure_ids=[f"meas-{i}" for i in range(20)],
        )
        result = await optimizer_engine.optimize(request)
        assert isinstance(result, OptimizeBudgetResponse)
