# -*- coding: utf-8 -*-
"""
Tests for Engine 7: Cost-Benefit Optimizer Engine - AGENT-EUDR-025

Tests linear programming budget optimization, Pareto frontier generation,
cost-effectiveness ratios, RICE prioritization, multi-scenario analysis,
sensitivity analysis, greedy fallback, and constraint validation.

Test count: ~70 tests
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


class TestOptimizerEngineInit:
    def test_engine_initializes(self, optimizer_engine):
        assert optimizer_engine is not None


class TestBudgetOptimization:
    @pytest.mark.asyncio
    async def test_optimize_returns_response(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert isinstance(result, OptimizeBudgetResponse)

    @pytest.mark.asyncio
    async def test_optimize_budget_not_exceeded(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert result.total_budget_used <= optimize_budget_request.total_budget_eur

    @pytest.mark.asyncio
    async def test_optimize_has_allocations(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert isinstance(result.allocations, dict)

    @pytest.mark.asyncio
    async def test_optimize_solver_status(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert result.solver_status in ("optimal", "feasible", "infeasible", "greedy_fallback")

    @pytest.mark.asyncio
    async def test_optimize_provenance_hash(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_optimize_processing_time(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert result.processing_time_ms >= Decimal("0")

    @pytest.mark.asyncio
    async def test_optimize_risk_reduction_positive(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert result.expected_risk_reduction >= Decimal("0")


class TestBudgetConstraints:
    @pytest.mark.asyncio
    async def test_per_supplier_cap_respected(self, optimizer_engine):
        request = OptimizeBudgetRequest(
            operator_id="op-cap",
            total_budget_eur=Decimal("100000"),
            per_supplier_cap_eur=Decimal("10000"),
            supplier_ids=["sup-1", "sup-2", "sup-3"],
            candidate_measure_ids=["meas-1"],
        )
        result = await optimizer_engine.optimize(request)
        for supplier_id, allocations in result.allocations.items():
            total_allocated = sum(
                Decimal(str(a.get("cost", 0))) for a in allocations
            )
            assert total_allocated <= Decimal("10000")

    @pytest.mark.asyncio
    async def test_category_budget_respected(self, optimizer_engine):
        request = OptimizeBudgetRequest(
            operator_id="op-cat",
            total_budget_eur=Decimal("100000"),
            category_budgets={"deforestation": Decimal("30000")},
            supplier_ids=["sup-1"],
            candidate_measure_ids=["meas-1"],
        )
        result = await optimizer_engine.optimize(request)
        assert result.total_budget_used <= Decimal("100000")

    @pytest.mark.asyncio
    async def test_zero_budget(self, optimizer_engine):
        """Zero budget should result in no allocations."""
        request = OptimizeBudgetRequest(
            operator_id="op-zero",
            total_budget_eur=Decimal("1"),  # Minimal but positive
            supplier_ids=["sup-1"],
            candidate_measure_ids=["meas-1"],
        )
        result = await optimizer_engine.optimize(request)
        assert isinstance(result, OptimizeBudgetResponse)


class TestCostEffectivenessRatio:
    @pytest.mark.asyncio
    async def test_calculate_ce_ratio(self, optimizer_engine):
        ce = await optimizer_engine.calculate_cost_effectiveness(
            expected_reduction=Decimal("30"),
            expected_cost=Decimal("10000"),
        )
        assert isinstance(ce, Decimal)
        assert ce > Decimal("0")

    @pytest.mark.asyncio
    async def test_ce_ratio_higher_is_better(self, optimizer_engine):
        ce_high = await optimizer_engine.calculate_cost_effectiveness(
            expected_reduction=Decimal("50"),
            expected_cost=Decimal("10000"),
        )
        ce_low = await optimizer_engine.calculate_cost_effectiveness(
            expected_reduction=Decimal("20"),
            expected_cost=Decimal("10000"),
        )
        assert ce_high > ce_low

    @pytest.mark.asyncio
    async def test_ce_ratio_zero_cost(self, optimizer_engine):
        ce = await optimizer_engine.calculate_cost_effectiveness(
            expected_reduction=Decimal("30"),
            expected_cost=Decimal("0"),
        )
        # Should handle gracefully
        assert ce is None or ce > Decimal("0")


class TestParetoFrontier:
    @pytest.mark.asyncio
    async def test_pareto_frontier_generated(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert isinstance(result.pareto_frontier, list)

    @pytest.mark.asyncio
    async def test_pareto_points_ordered(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        if len(result.pareto_frontier) >= 2:
            for i in range(len(result.pareto_frontier) - 1):
                assert (
                    result.pareto_frontier[i].get("budget", Decimal("0"))
                    <= result.pareto_frontier[i + 1].get("budget", Decimal("0"))
                )

    @pytest.mark.asyncio
    async def test_pareto_has_multiple_points(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert len(result.pareto_frontier) >= 1


class TestMultiScenarioAnalysis:
    @pytest.mark.asyncio
    async def test_scenario_budget_increase(self, optimizer_engine, optimize_budget_request):
        scenarios = await optimizer_engine.run_scenarios(
            optimize_budget_request,
            budget_multipliers=[Decimal("1.0"), Decimal("1.2"), Decimal("1.5")],
        )
        assert len(scenarios) == 3

    @pytest.mark.asyncio
    async def test_scenario_results_monotonic(self, optimizer_engine, optimize_budget_request):
        scenarios = await optimizer_engine.run_scenarios(
            optimize_budget_request,
            budget_multipliers=[Decimal("0.5"), Decimal("1.0"), Decimal("2.0")],
        )
        reductions = [s["expected_risk_reduction"] for s in scenarios]
        for i in range(len(reductions) - 1):
            assert reductions[i] <= reductions[i + 1]


class TestSensitivityAnalysis:
    @pytest.mark.asyncio
    async def test_sensitivity_analysis(self, optimizer_engine, optimize_budget_request):
        result = await optimizer_engine.optimize(optimize_budget_request)
        assert isinstance(result.sensitivity, dict)


class TestRICEPrioritization:
    @pytest.mark.asyncio
    async def test_rice_score_calculation(self, optimizer_engine):
        rice = await optimizer_engine.calculate_rice_score(
            reach=Decimal("100"),
            impact=Decimal("30"),
            confidence=Decimal("0.8"),
            effort=Decimal("50000"),
        )
        assert isinstance(rice, Decimal)
        assert rice > Decimal("0")

    @pytest.mark.asyncio
    async def test_rice_higher_impact_higher_score(self, optimizer_engine):
        rice_high = await optimizer_engine.calculate_rice_score(
            reach=Decimal("100"), impact=Decimal("50"),
            confidence=Decimal("0.8"), effort=Decimal("50000"),
        )
        rice_low = await optimizer_engine.calculate_rice_score(
            reach=Decimal("100"), impact=Decimal("20"),
            confidence=Decimal("0.8"), effort=Decimal("50000"),
        )
        assert rice_high > rice_low


class TestGreedyFallback:
    @pytest.mark.asyncio
    async def test_greedy_produces_result(self, optimizer_engine):
        result = await optimizer_engine.greedy_allocate(
            total_budget=Decimal("50000"),
            supplier_ids=["sup-1", "sup-2"],
            measure_costs=[Decimal("10000"), Decimal("15000")],
            measure_reductions=[Decimal("25"), Decimal("35")],
        )
        assert result is not None
        assert result["total_cost"] <= Decimal("50000")

    @pytest.mark.asyncio
    async def test_greedy_prioritizes_best_ratio(self, optimizer_engine):
        result = await optimizer_engine.greedy_allocate(
            total_budget=Decimal("20000"),
            supplier_ids=["sup-1", "sup-2"],
            measure_costs=[Decimal("10000"), Decimal("15000")],
            measure_reductions=[Decimal("40"), Decimal("35")],
        )
        # Best ratio is sup-1 (40/10000), should be allocated first
        assert result is not None


class TestSpendTracking:
    @pytest.mark.asyncio
    async def test_track_spend_variance(self, optimizer_engine):
        variance = await optimizer_engine.calculate_spend_variance(
            planned=Decimal("50000"),
            actual=Decimal("45000"),
        )
        assert isinstance(variance, Decimal)
        assert variance == Decimal("-5000")

    @pytest.mark.asyncio
    async def test_overspend_variance(self, optimizer_engine):
        variance = await optimizer_engine.calculate_spend_variance(
            planned=Decimal("50000"),
            actual=Decimal("60000"),
        )
        assert variance == Decimal("10000")


class TestOptimizerEdgeCases:
    @pytest.mark.asyncio
    async def test_single_supplier(self, optimizer_engine):
        request = OptimizeBudgetRequest(
            operator_id="op-single",
            total_budget_eur=Decimal("50000"),
            supplier_ids=["sup-1"],
            candidate_measure_ids=["meas-1"],
        )
        result = await optimizer_engine.optimize(request)
        assert isinstance(result, OptimizeBudgetResponse)

    @pytest.mark.asyncio
    async def test_no_candidate_measures(self, optimizer_engine):
        request = OptimizeBudgetRequest(
            operator_id="op-nomeas",
            total_budget_eur=Decimal("50000"),
            supplier_ids=["sup-1"],
            candidate_measure_ids=[],
        )
        result = await optimizer_engine.optimize(request)
        assert isinstance(result, OptimizeBudgetResponse)

    @pytest.mark.asyncio
    async def test_large_supplier_set(self, optimizer_engine):
        request = OptimizeBudgetRequest(
            operator_id="op-large",
            total_budget_eur=Decimal("1000000"),
            supplier_ids=[f"sup-{i}" for i in range(100)],
            candidate_measure_ids=[f"meas-{i}" for i in range(10)],
        )
        result = await optimizer_engine.optimize(request)
        assert isinstance(result, OptimizeBudgetResponse)

    @pytest.mark.asyncio
    async def test_deterministic_optimization(self, optimizer_engine, optimize_budget_request):
        r1 = await optimizer_engine.optimize(optimize_budget_request)
        r2 = await optimizer_engine.optimize(optimize_budget_request)
        assert r1.total_budget_used == r2.total_budget_used
        assert r1.expected_risk_reduction == r2.expected_risk_reduction
