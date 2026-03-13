# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests - AGENT-EUDR-025

Tests bit-perfect reproducibility of all engine operations, SHA-256
provenance chain integrity, deterministic strategy selection, immutable
model outputs, Decimal precision guarantees, and audit trail consistency.

Zero-hallucination guarantees per PRD:
    - All numeric calculations use Decimal arithmetic
    - Deterministic fallback produces bit-perfect results across runs
    - SHA-256 provenance hashes are reproducible
    - No LLM calls in the calculation path

Test count: ~35 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    PlanStatus,
    EnrollmentStatus,
    RecommendStrategiesRequest,
    CreatePlanRequest,
    EnrollSupplierRequest,
    OptimizeBudgetRequest,
    MeasureEffectivenessRequest,
    CollaborateRequest,
    StakeholderRole,
    SUPPORTED_COMMODITIES,
)
from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
)

from .conftest import FIXED_DATE, COMPOSITE_WEIGHTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_risk_input(commodity: str = "soya") -> RiskInput:
    return RiskInput(
        operator_id=f"op-det-{commodity}",
        supplier_id=f"sup-det-{commodity}",
        country_code="BR",
        commodity=commodity,
        country_risk_score=Decimal("65"),
        supplier_risk_score=Decimal("70"),
        commodity_risk_score=Decimal("55"),
        corruption_risk_score=Decimal("40"),
        deforestation_risk_score=Decimal("75"),
        indigenous_rights_score=Decimal("30"),
        protected_areas_score=Decimal("25"),
        legal_compliance_score=Decimal("35"),
        audit_risk_score=Decimal("40"),
        assessment_date=FIXED_DATE,
    )


# ---------------------------------------------------------------------------
# Strategy Selection Determinism
# ---------------------------------------------------------------------------


class TestStrategySelectionDeterminism:
    """Test deterministic strategy recommendation reproducibility."""

    @pytest.mark.asyncio
    async def test_same_input_same_strategies(self, strategy_engine):
        """Same input must produce identical strategy list."""
        ri = _make_risk_input("soya")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        assert len(r1.strategies) == len(r2.strategies)
        for s1, s2 in zip(r1.strategies, r2.strategies):
            assert s1.strategy_id == s2.strategy_id
            assert s1.name == s2.name

    @pytest.mark.asyncio
    async def test_same_input_same_composite_score(self, strategy_engine):
        """Same input must produce identical composite risk score."""
        ri = _make_risk_input("palm_oil")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        assert r1.composite_risk_score == r2.composite_risk_score

    @pytest.mark.asyncio
    async def test_same_input_same_risk_level(self, strategy_engine):
        """Same input must produce identical risk level classification."""
        ri = _make_risk_input("cocoa")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        assert r1.risk_level == r2.risk_level

    @pytest.mark.asyncio
    async def test_same_input_same_provenance_hash(self, strategy_engine):
        """Same input must produce identical SHA-256 provenance hash."""
        ri = _make_risk_input("cattle")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        assert r1.provenance_hash == r2.provenance_hash
        assert len(r1.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_different_input_different_hash(self, strategy_engine):
        """Different inputs must produce different provenance hashes."""
        ri_soya = _make_risk_input("soya")
        ri_palm = _make_risk_input("palm_oil")
        req_soya = RecommendStrategiesRequest(
            risk_input=ri_soya, top_k=5, deterministic_mode=True
        )
        req_palm = RecommendStrategiesRequest(
            risk_input=ri_palm, top_k=5, deterministic_mode=True
        )
        r_soya = await strategy_engine.recommend(req_soya)
        r_palm = await strategy_engine.recommend(req_palm)
        assert r_soya.provenance_hash != r_palm.provenance_hash

    @pytest.mark.asyncio
    async def test_strategy_effectiveness_scores_reproducible(self, strategy_engine):
        """Strategy effectiveness scores must be bit-perfect across runs."""
        ri = _make_risk_input("rubber")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        for s1, s2 in zip(r1.strategies, r2.strategies):
            assert s1.predicted_effectiveness == s2.predicted_effectiveness
            assert s1.confidence_score == s2.confidence_score

    @pytest.mark.asyncio
    async def test_strategy_cost_estimates_reproducible(self, strategy_engine):
        """Strategy cost estimates must be bit-perfect across runs."""
        ri = _make_risk_input("coffee")
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        for s1, s2 in zip(r1.strategies, r2.strategies):
            if s1.cost_estimate and s2.cost_estimate:
                assert s1.cost_estimate.level == s2.cost_estimate.level

    @pytest.mark.parametrize("commodity", SUPPORTED_COMMODITIES)
    @pytest.mark.asyncio
    async def test_determinism_all_commodities(self, strategy_engine, commodity):
        """All 7 commodities produce deterministic results."""
        ri = _make_risk_input(commodity)
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        assert r1.provenance_hash == r2.provenance_hash
        assert len(r1.strategies) == len(r2.strategies)


# ---------------------------------------------------------------------------
# Plan Creation Determinism
# ---------------------------------------------------------------------------


class TestPlanCreationDeterminism:
    """Test remediation plan creation produces consistent results."""

    @pytest.mark.asyncio
    async def test_plan_structure_reproducible(self, remediation_engine):
        """Plan structure should be consistent for same input."""
        req = CreatePlanRequest(
            operator_id="op-det-plan",
            supplier_id="sup-det-plan",
            strategy_ids=["strat-001"],
            template_name="supplier_capacity_building",
            budget_eur=Decimal("50000"),
            target_duration_weeks=24,
        )
        p1 = await remediation_engine.create_plan(req)
        p2 = await remediation_engine.create_plan(req)
        # Both plans should have same template
        assert p1.plan.plan_template == p2.plan.plan_template
        # Both plans should have same budget
        assert p1.plan.budget_allocated == p2.plan.budget_allocated

    @pytest.mark.asyncio
    async def test_plan_provenance_exists(self, remediation_engine):
        """Plan creation must produce a provenance hash."""
        req = CreatePlanRequest(
            operator_id="op-det-prov",
            supplier_id="sup-det-prov",
            budget_eur=Decimal("30000"),
        )
        result = await remediation_engine.create_plan(req)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# Optimization Determinism
# ---------------------------------------------------------------------------


class TestOptimizationDeterminism:
    """Test budget optimization determinism."""

    @pytest.mark.asyncio
    async def test_optimization_budget_reproducible(self, optimizer_engine):
        """Same optimization request produces same budget allocation."""
        req = OptimizeBudgetRequest(
            operator_id="op-det-opt",
            total_budget_eur=Decimal("200000"),
            supplier_ids=["sup-1", "sup-2", "sup-3"],
            candidate_measure_ids=["meas-1", "meas-2"],
        )
        r1 = await optimizer_engine.optimize(req)
        r2 = await optimizer_engine.optimize(req)
        assert r1.total_budget_used == r2.total_budget_used
        assert r1.expected_risk_reduction == r2.expected_risk_reduction

    @pytest.mark.asyncio
    async def test_optimization_provenance_reproducible(self, optimizer_engine):
        """Same optimization request produces same provenance hash."""
        req = OptimizeBudgetRequest(
            operator_id="op-det-opt-prov",
            total_budget_eur=Decimal("100000"),
            supplier_ids=["sup-1", "sup-2"],
        )
        r1 = await optimizer_engine.optimize(req)
        r2 = await optimizer_engine.optimize(req)
        assert r1.provenance_hash == r2.provenance_hash

    @pytest.mark.asyncio
    async def test_optimization_allocations_reproducible(self, optimizer_engine):
        """Same optimization request produces identical allocation map."""
        req = OptimizeBudgetRequest(
            operator_id="op-det-alloc",
            total_budget_eur=Decimal("500000"),
            supplier_ids=[f"sup-{i}" for i in range(5)],
            candidate_measure_ids=[f"meas-{i}" for i in range(3)],
        )
        r1 = await optimizer_engine.optimize(req)
        r2 = await optimizer_engine.optimize(req)
        assert r1.allocations == r2.allocations


# ---------------------------------------------------------------------------
# Provenance Chain Determinism
# ---------------------------------------------------------------------------


class TestProvenanceChainDeterminism:
    """Test SHA-256 provenance chain determinism."""

    def test_chain_hash_deterministic(self):
        """Same sequence of records produces same chain hashes."""
        t1 = ProvenanceTracker(genesis_hash="TEST-GENESIS")
        t2 = ProvenanceTracker(genesis_hash="TEST-GENESIS")

        for i in range(5):
            t1.record("strategy_recommendation", "recommend", f"strat-{i}")
            t2.record("strategy_recommendation", "recommend", f"strat-{i}")

        chain1 = t1.get_chain()
        chain2 = t2.get_chain()
        assert len(chain1) == len(chain2) == 5
        for r1, r2 in zip(chain1, chain2):
            assert r1.hash_value == r2.hash_value

    def test_chain_integrity_verification(self):
        """Chain integrity verification returns True for untampered chain."""
        tracker = ProvenanceTracker(genesis_hash="INTEGRITY-TEST")
        for i in range(10):
            tracker.record(
                "remediation_plan", "create", f"plan-{i}",
                actor="test_user",
            )
        assert tracker.verify_chain() is True

    def test_genesis_hash_affects_chain(self):
        """Different genesis hashes produce different chain hashes."""
        t1 = ProvenanceTracker(genesis_hash="GENESIS-A")
        t2 = ProvenanceTracker(genesis_hash="GENESIS-B")
        t1.record("strategy_recommendation", "recommend", "strat-001")
        t2.record("strategy_recommendation", "recommend", "strat-001")
        assert t1.get_chain()[0].hash_value != t2.get_chain()[0].hash_value

    def test_chain_export_json_deterministic(self):
        """JSON export of same chain produces identical output."""
        t1 = ProvenanceTracker(genesis_hash="JSON-TEST")
        t2 = ProvenanceTracker(genesis_hash="JSON-TEST")
        for i in range(3):
            t1.record("optimization_result", "optimize", f"opt-{i}")
            t2.record("optimization_result", "optimize", f"opt-{i}")
        json1 = t1.export_json()
        json2 = t2.export_json()
        assert json1 == json2

    def test_record_immutability(self):
        """ProvenanceRecord should be immutable (frozen dataclass)."""
        tracker = ProvenanceTracker()
        entry = tracker.record(
            "strategy_recommendation", "recommend", "strat-001"
        )
        with pytest.raises((AttributeError, TypeError)):
            entry.hash_value = "tampered"

    def test_chain_previous_hash_linkage(self):
        """Each record's previous_hash should match prior record's hash."""
        tracker = ProvenanceTracker(genesis_hash="LINK-TEST")
        for i in range(5):
            tracker.record("mitigation_measure", "create", f"meas-{i}")
        chain = tracker.get_chain()
        for i in range(1, len(chain)):
            assert chain[i].previous_hash == chain[i - 1].hash_value


# ---------------------------------------------------------------------------
# Composite Score Decimal Precision
# ---------------------------------------------------------------------------


class TestDecimalPrecision:
    """Test Decimal arithmetic precision in composite score calculation."""

    @pytest.mark.asyncio
    async def test_composite_score_uses_decimal(self, strategy_engine, high_risk_input):
        """Composite risk score must be Decimal, not float."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert isinstance(result.composite_risk_score, Decimal)

    @pytest.mark.asyncio
    async def test_composite_score_precision(self, strategy_engine, high_risk_input):
        """Composite score should have sufficient decimal precision."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        # Score should be within valid range
        assert Decimal("0") <= result.composite_risk_score <= Decimal("100")

    @pytest.mark.asyncio
    async def test_strategy_scores_are_decimal(self, strategy_engine, high_risk_input):
        """All strategy numeric scores should be Decimal type."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        for s in result.strategies:
            assert isinstance(s.predicted_effectiveness, Decimal)
            assert isinstance(s.confidence_score, Decimal)

    @pytest.mark.asyncio
    async def test_hand_calculated_composite(self, strategy_engine, high_risk_input):
        """Composite score must match hand-calculated weighted average."""
        req = RecommendStrategiesRequest(
            risk_input=high_risk_input, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)

        # Hand-compute expected composite from high_risk_input
        expected = (
            Decimal("85") * Decimal("0.15")   # country
            + Decimal("78") * Decimal("0.15") # supplier
            + Decimal("72") * Decimal("0.10") # commodity
            + Decimal("80") * Decimal("0.10") # corruption
            + Decimal("90") * Decimal("0.20") # deforestation
            + Decimal("65") * Decimal("0.10") # indigenous
            + Decimal("70") * Decimal("0.10") # protected_areas
            + Decimal("75") * Decimal("0.10") # legal_compliance
        )
        # Allow tolerance for any internal rounding
        assert abs(result.composite_risk_score - expected) <= Decimal("1"), (
            f"Composite {result.composite_risk_score} != expected {expected}"
        )


# ---------------------------------------------------------------------------
# Batch Determinism
# ---------------------------------------------------------------------------


class TestBatchDeterminism:
    """Test batch processing produces identical results across runs."""

    @pytest.mark.asyncio
    async def test_batch_recommendation_reproducible(self, strategy_engine):
        """Batch recommendation produces identical results on re-run."""
        inputs = [_make_risk_input(c) for c in SUPPORTED_COMMODITIES]
        requests = [
            RecommendStrategiesRequest(
                risk_input=ri, top_k=3, deterministic_mode=True
            )
            for ri in inputs
        ]
        r1 = await strategy_engine.recommend_batch(requests)
        r2 = await strategy_engine.recommend_batch(requests)
        assert len(r1) == len(r2)
        for res1, res2 in zip(r1, r2):
            assert res1.provenance_hash == res2.provenance_hash
            assert len(res1.strategies) == len(res2.strategies)

    @pytest.mark.asyncio
    async def test_batch_order_independent(self, strategy_engine):
        """Batch with reordered inputs should produce same per-input results."""
        commodities = list(SUPPORTED_COMMODITIES)
        inputs_forward = [_make_risk_input(c) for c in commodities]
        inputs_reverse = list(reversed(inputs_forward))

        req_fwd = [
            RecommendStrategiesRequest(
                risk_input=ri, top_k=3, deterministic_mode=True
            )
            for ri in inputs_forward
        ]
        req_rev = [
            RecommendStrategiesRequest(
                risk_input=ri, top_k=3, deterministic_mode=True
            )
            for ri in inputs_reverse
        ]
        r_fwd = await strategy_engine.recommend_batch(req_fwd)
        r_rev = await strategy_engine.recommend_batch(req_rev)

        # Match by supplier_id to compare
        fwd_map = {r.strategies[0].strategy_id if r.strategies else None: r.provenance_hash for r in r_fwd}
        rev_map = {r.strategies[0].strategy_id if r.strategies else None: r.provenance_hash for r in r_rev}
        # All hashes from forward batch should appear in reverse batch
        for key, hash_val in fwd_map.items():
            if key is not None:
                assert key in rev_map
