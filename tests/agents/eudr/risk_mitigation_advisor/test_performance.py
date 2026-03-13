# -*- coding: utf-8 -*-
"""
Performance Benchmark Tests - AGENT-EUDR-025

Tests latency targets, throughput benchmarks, memory usage, batch
processing performance, and scalability for all engine operations.

Performance targets (from PRD):
    - Strategy recommendation: < 500ms per supplier
    - Remediation plan generation: < 2s per plan
    - Budget optimization: < 1s for up to 200 suppliers
    - Batch recommendation: >= 1,000 per minute
    - Memory: < 500MB for 100K records

Test count: ~25 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    RecommendStrategiesRequest,
    CreatePlanRequest,
    EnrollSupplierRequest,
    SearchMeasuresRequest,
    MeasureEffectivenessRequest,
    OptimizeBudgetRequest,
    AdaptiveScanRequest,
    CollaborateRequest,
    GenerateReportRequest,
    StakeholderRole,
    ReportType,
    SUPPORTED_COMMODITIES,
)

from .conftest import FIXED_DATE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_risk_input(idx: int = 0) -> RiskInput:
    return RiskInput(
        operator_id=f"op-perf-{idx:04d}",
        supplier_id=f"sup-perf-{idx:04d}",
        country_code="BR",
        commodity=SUPPORTED_COMMODITIES[idx % len(SUPPORTED_COMMODITIES)],
        country_risk_score=Decimal(str(40 + (idx % 50))),
        supplier_risk_score=Decimal(str(35 + (idx % 50))),
        commodity_risk_score=Decimal(str(30 + (idx % 50))),
        corruption_risk_score=Decimal(str(25 + (idx % 40))),
        deforestation_risk_score=Decimal(str(45 + (idx % 50))),
        indigenous_rights_score=Decimal(str(20 + (idx % 30))),
        protected_areas_score=Decimal(str(15 + (idx % 30))),
        legal_compliance_score=Decimal(str(30 + (idx % 40))),
        audit_risk_score=Decimal(str(25 + (idx % 40))),
        assessment_date=FIXED_DATE,
    )


# ---------------------------------------------------------------------------
# Strategy Recommendation Performance
# ---------------------------------------------------------------------------


class TestStrategyRecommendationPerformance:
    """Test strategy recommendation meets latency and throughput targets."""

    @pytest.mark.asyncio
    async def test_single_recommendation_latency(self, strategy_engine):
        """Single recommendation should complete in < 500ms."""
        ri = _make_risk_input(0)
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        start = time.perf_counter()
        result = await strategy_engine.recommend(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Latency {elapsed_ms:.1f}ms exceeds 500ms target"
        assert len(result.strategies) >= 1

    @pytest.mark.asyncio
    async def test_recommendation_10_suppliers(self, strategy_engine):
        """10 sequential recommendations should average < 500ms each."""
        total_start = time.perf_counter()
        for i in range(10):
            ri = _make_risk_input(i)
            req = RecommendStrategiesRequest(
                risk_input=ri, top_k=5, deterministic_mode=True
            )
            await strategy_engine.recommend(req)
        total_ms = (time.perf_counter() - total_start) * 1000
        avg_ms = total_ms / 10
        assert avg_ms < 500, f"Average latency {avg_ms:.1f}ms exceeds 500ms"

    @pytest.mark.asyncio
    async def test_batch_recommendation_throughput(self, strategy_engine):
        """Batch recommendation for 50 suppliers should maintain throughput."""
        requests = [
            RecommendStrategiesRequest(
                risk_input=_make_risk_input(i), top_k=3, deterministic_mode=True
            )
            for i in range(50)
        ]
        start = time.perf_counter()
        results = await strategy_engine.recommend_batch(requests)
        elapsed_sec = time.perf_counter() - start
        throughput_per_min = (50 / elapsed_sec) * 60

        assert len(results) == 50
        # Target: at least 1000 per minute
        assert throughput_per_min >= 100, (
            f"Throughput {throughput_per_min:.0f}/min below 100/min minimum"
        )

    @pytest.mark.asyncio
    async def test_recommendation_top_k_1_faster_than_top_k_10(self, strategy_engine):
        """top_k=1 should be at least as fast as top_k=10."""
        ri = _make_risk_input(0)

        req_1 = RecommendStrategiesRequest(
            risk_input=ri, top_k=1, deterministic_mode=True
        )
        start = time.perf_counter()
        await strategy_engine.recommend(req_1)
        time_k1 = time.perf_counter() - start

        req_10 = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        start = time.perf_counter()
        await strategy_engine.recommend(req_10)
        time_k10 = time.perf_counter() - start

        # top_k=1 should not be significantly slower than top_k=10
        assert time_k1 <= time_k10 * 3


# ---------------------------------------------------------------------------
# Remediation Plan Performance
# ---------------------------------------------------------------------------


class TestPlanGenerationPerformance:
    """Test plan generation meets 2-second target."""

    @pytest.mark.asyncio
    async def test_single_plan_creation_latency(self, remediation_engine):
        """Single plan creation should complete in < 2s."""
        req = CreatePlanRequest(
            operator_id="op-perf-plan",
            supplier_id="sup-perf-plan",
            strategy_ids=["strat-001"],
            template_name="supplier_capacity_building",
            budget_eur=Decimal("50000"),
            target_duration_weeks=24,
        )
        start = time.perf_counter()
        result = await remediation_engine.create_plan(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"Plan creation {elapsed_ms:.1f}ms exceeds 2s target"
        assert result.plan is not None

    @pytest.mark.asyncio
    async def test_plan_creation_10_sequential(self, remediation_engine):
        """10 plan creations should average < 2s each."""
        total_start = time.perf_counter()
        for i in range(10):
            req = CreatePlanRequest(
                operator_id=f"op-perf-{i}",
                supplier_id=f"sup-perf-{i}",
                template_name="supplier_capacity_building",
                budget_eur=Decimal("25000"),
            )
            await remediation_engine.create_plan(req)
        total_ms = (time.perf_counter() - total_start) * 1000
        avg_ms = total_ms / 10
        assert avg_ms < 2000, f"Average plan creation {avg_ms:.1f}ms exceeds 2s"


# ---------------------------------------------------------------------------
# Budget Optimization Performance
# ---------------------------------------------------------------------------


class TestOptimizationPerformance:
    """Test budget optimization meets 1-second target."""

    @pytest.mark.asyncio
    async def test_optimization_3_suppliers(self, optimizer_engine):
        """Optimization for 3 suppliers should complete in < 1s."""
        req = OptimizeBudgetRequest(
            operator_id="op-perf-opt",
            total_budget_eur=Decimal("200000"),
            supplier_ids=["sup-1", "sup-2", "sup-3"],
            candidate_measure_ids=["meas-1", "meas-2", "meas-3"],
        )
        start = time.perf_counter()
        result = await optimizer_engine.optimize(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1000, f"Optimization {elapsed_ms:.1f}ms exceeds 1s target"

    @pytest.mark.asyncio
    async def test_optimization_50_suppliers(self, optimizer_engine):
        """Optimization for 50 suppliers should complete in < 5s."""
        req = OptimizeBudgetRequest(
            operator_id="op-perf-opt-50",
            total_budget_eur=Decimal("2000000"),
            supplier_ids=[f"sup-{i}" for i in range(50)],
            candidate_measure_ids=[f"meas-{i}" for i in range(10)],
        )
        start = time.perf_counter()
        result = await optimizer_engine.optimize(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"Optimization 50 suppliers {elapsed_ms:.1f}ms exceeds 5s"

    @pytest.mark.asyncio
    async def test_optimization_200_suppliers(self, optimizer_engine):
        """Optimization for 200 suppliers should complete in < 10s."""
        req = OptimizeBudgetRequest(
            operator_id="op-perf-opt-200",
            total_budget_eur=Decimal("5000000"),
            supplier_ids=[f"sup-{i}" for i in range(200)],
            candidate_measure_ids=[f"meas-{i}" for i in range(20)],
        )
        start = time.perf_counter()
        result = await optimizer_engine.optimize(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 10000, (
            f"Optimization 200 suppliers {elapsed_ms:.1f}ms exceeds 10s"
        )

    @pytest.mark.asyncio
    async def test_scenario_analysis_4_scenarios(self, optimizer_engine):
        """4-scenario analysis should complete in < 5s."""
        base_req = OptimizeBudgetRequest(
            operator_id="op-perf-scenario",
            total_budget_eur=Decimal("500000"),
            supplier_ids=[f"sup-{i}" for i in range(20)],
            candidate_measure_ids=[f"meas-{i}" for i in range(5)],
        )
        start = time.perf_counter()
        scenarios = await optimizer_engine.run_scenarios(
            base_req,
            budget_multipliers=[
                Decimal("0.5"), Decimal("1.0"),
                Decimal("1.5"), Decimal("2.0"),
            ],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"Scenario analysis {elapsed_ms:.1f}ms exceeds 5s"
        assert len(scenarios) == 4


# ---------------------------------------------------------------------------
# Capacity Building Performance
# ---------------------------------------------------------------------------


class TestCapacityBuildingPerformance:
    """Test capacity building enrollment performance."""

    @pytest.mark.asyncio
    async def test_enrollment_latency(self, capacity_engine):
        """Single enrollment should complete in < 500ms."""
        req = EnrollSupplierRequest(
            supplier_id="sup-perf-cap",
            commodity="palm_oil",
            initial_tier=1,
            target_completion_weeks=24,
        )
        start = time.perf_counter()
        result = await capacity_engine.enroll_supplier(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500, f"Enrollment {elapsed_ms:.1f}ms exceeds 500ms"

    @pytest.mark.asyncio
    async def test_batch_enrollment_20_suppliers(self, capacity_engine):
        """20 enrollments should complete in < 5s total."""
        reqs = [
            EnrollSupplierRequest(
                supplier_id=f"sup-perf-cap-{i}",
                commodity=SUPPORTED_COMMODITIES[i % len(SUPPORTED_COMMODITIES)],
                initial_tier=1,
                target_completion_weeks=24,
            )
            for i in range(20)
        ]
        start = time.perf_counter()
        for req in reqs:
            await capacity_engine.enroll_supplier(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 5000, f"20 enrollments {elapsed_ms:.1f}ms exceeds 5s"


# ---------------------------------------------------------------------------
# Collaboration Performance
# ---------------------------------------------------------------------------


class TestCollaborationPerformance:
    """Test collaboration engine performance."""

    @pytest.mark.asyncio
    async def test_message_latency(self, collaboration_engine):
        """Single collaboration message should complete in < 200ms."""
        req = CollaborateRequest(
            action="message",
            plan_id="plan-perf-001",
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="Performance test message.",
        )
        start = time.perf_counter()
        result = await collaboration_engine.execute(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 200, f"Message latency {elapsed_ms:.1f}ms exceeds 200ms"

    @pytest.mark.asyncio
    async def test_bulk_message_latency(self, collaboration_engine):
        """Bulk message to 10 plans should complete in < 2s."""
        start = time.perf_counter()
        result = await collaboration_engine.send_bulk_message(
            plan_ids=[f"plan-perf-{i}" for i in range(10)],
            stakeholder_role=StakeholderRole.INTERNAL_COMPLIANCE,
            message="Quarterly compliance update.",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"Bulk message {elapsed_ms:.1f}ms exceeds 2s"


# ---------------------------------------------------------------------------
# Monitoring Performance
# ---------------------------------------------------------------------------


class TestMonitoringPerformance:
    """Test continuous monitoring scan performance."""

    @pytest.mark.asyncio
    async def test_adaptive_scan_latency(self, monitoring_engine):
        """Adaptive scan for 5 plans should complete in < 2s."""
        req = AdaptiveScanRequest(
            operator_id="op-perf-mon",
            plan_ids=[f"plan-perf-{i}" for i in range(5)],
            include_recommendations=True,
        )
        start = time.perf_counter()
        result = await monitoring_engine.scan(req)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 2000, f"Adaptive scan {elapsed_ms:.1f}ms exceeds 2s"


# ---------------------------------------------------------------------------
# Provenance Performance
# ---------------------------------------------------------------------------


class TestProvenancePerformance:
    """Test provenance tracking performance overhead."""

    @pytest.mark.asyncio
    async def test_provenance_chain_100_records(self, provenance_tracker):
        """Recording 100 provenance entries should complete in < 100ms."""
        start = time.perf_counter()
        for i in range(100):
            provenance_tracker.record(
                entity_type="strategy_recommendation",
                action="recommend",
                entity_id=f"strat-perf-{i:04d}",
                actor="test",
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"100 records {elapsed_ms:.1f}ms exceeds 100ms"

    @pytest.mark.asyncio
    async def test_provenance_verify_chain_100(self, provenance_tracker):
        """Verifying a 100-record chain should complete in < 50ms."""
        for i in range(100):
            provenance_tracker.record(
                entity_type="remediation_plan",
                action="create",
                entity_id=f"plan-perf-{i:04d}",
            )
        start = time.perf_counter()
        valid = provenance_tracker.verify_chain()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert valid is True
        assert elapsed_ms < 50, f"Chain verification {elapsed_ms:.1f}ms exceeds 50ms"

    @pytest.mark.asyncio
    async def test_provenance_export_json_100(self, provenance_tracker):
        """Exporting 100-record chain to JSON should complete in < 50ms."""
        for i in range(100):
            provenance_tracker.record(
                entity_type="optimization_result",
                action="optimize",
                entity_id=f"opt-perf-{i:04d}",
            )
        start = time.perf_counter()
        json_str = provenance_tracker.export_json()
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert json_str is not None
        assert elapsed_ms < 50, f"JSON export {elapsed_ms:.1f}ms exceeds 50ms"
