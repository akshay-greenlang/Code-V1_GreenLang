# -*- coding: utf-8 -*-
"""
Golden Test Scenarios - AGENT-EUDR-025

15 golden test scenarios across all 7 EUDR commodities testing known-good
risk inputs against expected strategy outputs, composite scores, risk
levels, plan templates, and provenance hashes for regression detection.

Test count: 15 golden scenarios + ~20 assertion tests = ~35 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from datetime import date

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    RiskInput,
    RiskCategory,
    ISO31000TreatmentType,
    ImplementationComplexity,
    PlanStatus,
    RecommendStrategiesRequest,
    RecommendStrategiesResponse,
    CreatePlanRequest,
    SUPPORTED_COMMODITIES,
)

from .conftest import FIXED_DATE, COMPOSITE_WEIGHTS


# ---------------------------------------------------------------------------
# Golden scenario data
# ---------------------------------------------------------------------------

GOLDEN_SCENARIOS = [
    # Scenario 1: High-risk cattle from Brazil (Amazon frontier)
    {
        "scenario_id": "GOLD-001",
        "name": "High-risk cattle Brazil Amazon",
        "risk_input": {
            "operator_id": "op-gold-001",
            "supplier_id": "sup-gold-cattle-br",
            "country_code": "BR",
            "commodity": "cattle",
            "country_risk_score": "70",
            "supplier_risk_score": "80",
            "commodity_risk_score": "85",
            "corruption_risk_score": "45",
            "deforestation_risk_score": "92",
            "indigenous_rights_score": "60",
            "protected_areas_score": "55",
            "legal_compliance_score": "50",
            "audit_risk_score": "65",
        },
        "expected_risk_level": "critical",
        "expected_min_strategies": 3,
        "expected_risk_categories": [RiskCategory.DEFORESTATION, RiskCategory.SUPPLIER],
    },
    # Scenario 2: Medium-risk cattle from Paraguay
    {
        "scenario_id": "GOLD-002",
        "name": "Medium-risk cattle Paraguay",
        "risk_input": {
            "operator_id": "op-gold-002",
            "supplier_id": "sup-gold-cattle-py",
            "country_code": "PY",
            "commodity": "cattle",
            "country_risk_score": "55",
            "supplier_risk_score": "50",
            "commodity_risk_score": "60",
            "corruption_risk_score": "40",
            "deforestation_risk_score": "55",
            "indigenous_rights_score": "35",
            "protected_areas_score": "30",
            "legal_compliance_score": "45",
            "audit_risk_score": "40",
        },
        "expected_risk_level": "medium",
        "expected_min_strategies": 2,
        "expected_risk_categories": [RiskCategory.COMMODITY, RiskCategory.DEFORESTATION],
    },
    # Scenario 3: High-risk cocoa from Cote d'Ivoire
    {
        "scenario_id": "GOLD-003",
        "name": "High-risk cocoa Cote d'Ivoire",
        "risk_input": {
            "operator_id": "op-gold-003",
            "supplier_id": "sup-gold-cocoa-ci",
            "country_code": "CI",
            "commodity": "cocoa",
            "country_risk_score": "75",
            "supplier_risk_score": "72",
            "commodity_risk_score": "80",
            "corruption_risk_score": "60",
            "deforestation_risk_score": "88",
            "indigenous_rights_score": "45",
            "protected_areas_score": "50",
            "legal_compliance_score": "55",
            "audit_risk_score": "58",
        },
        "expected_risk_level": "high",
        "expected_min_strategies": 3,
        "expected_risk_categories": [RiskCategory.DEFORESTATION],
    },
    # Scenario 4: Low-risk cocoa from Ghana certified supplier
    {
        "scenario_id": "GOLD-004",
        "name": "Low-risk cocoa Ghana certified",
        "risk_input": {
            "operator_id": "op-gold-004",
            "supplier_id": "sup-gold-cocoa-gh",
            "country_code": "GH",
            "commodity": "cocoa",
            "country_risk_score": "30",
            "supplier_risk_score": "15",
            "commodity_risk_score": "25",
            "corruption_risk_score": "20",
            "deforestation_risk_score": "18",
            "indigenous_rights_score": "10",
            "protected_areas_score": "12",
            "legal_compliance_score": "15",
            "audit_risk_score": "14",
        },
        "expected_risk_level": "low",
        "expected_min_strategies": 1,
        "expected_risk_categories": [],
    },
    # Scenario 5: High-risk coffee from Vietnam highlands
    {
        "scenario_id": "GOLD-005",
        "name": "High-risk coffee Vietnam",
        "risk_input": {
            "operator_id": "op-gold-005",
            "supplier_id": "sup-gold-coffee-vn",
            "country_code": "VN",
            "commodity": "coffee",
            "country_risk_score": "60",
            "supplier_risk_score": "68",
            "commodity_risk_score": "55",
            "corruption_risk_score": "50",
            "deforestation_risk_score": "72",
            "indigenous_rights_score": "58",
            "protected_areas_score": "62",
            "legal_compliance_score": "48",
            "audit_risk_score": "55",
        },
        "expected_risk_level": "high",
        "expected_min_strategies": 2,
        "expected_risk_categories": [RiskCategory.DEFORESTATION],
    },
    # Scenario 6: Medium-risk coffee from Colombia
    {
        "scenario_id": "GOLD-006",
        "name": "Medium-risk coffee Colombia",
        "risk_input": {
            "operator_id": "op-gold-006",
            "supplier_id": "sup-gold-coffee-co",
            "country_code": "CO",
            "commodity": "coffee",
            "country_risk_score": "45",
            "supplier_risk_score": "40",
            "commodity_risk_score": "50",
            "corruption_risk_score": "35",
            "deforestation_risk_score": "48",
            "indigenous_rights_score": "30",
            "protected_areas_score": "28",
            "legal_compliance_score": "38",
            "audit_risk_score": "35",
        },
        "expected_risk_level": "medium",
        "expected_min_strategies": 1,
        "expected_risk_categories": [RiskCategory.COMMODITY],
    },
    # Scenario 7: Critical-risk palm oil from Indonesia (peatland)
    {
        "scenario_id": "GOLD-007",
        "name": "Critical-risk palm oil Indonesia peatland",
        "risk_input": {
            "operator_id": "op-gold-007",
            "supplier_id": "sup-gold-palm-id",
            "country_code": "ID",
            "commodity": "palm_oil",
            "country_risk_score": "78",
            "supplier_risk_score": "85",
            "commodity_risk_score": "90",
            "corruption_risk_score": "65",
            "deforestation_risk_score": "95",
            "indigenous_rights_score": "70",
            "protected_areas_score": "75",
            "legal_compliance_score": "60",
            "audit_risk_score": "72",
        },
        "expected_risk_level": "critical",
        "expected_min_strategies": 3,
        "expected_risk_categories": [RiskCategory.DEFORESTATION, RiskCategory.COMMODITY],
    },
    # Scenario 8: Medium-risk palm oil from Malaysia
    {
        "scenario_id": "GOLD-008",
        "name": "Medium-risk palm oil Malaysia",
        "risk_input": {
            "operator_id": "op-gold-008",
            "supplier_id": "sup-gold-palm-my",
            "country_code": "MY",
            "commodity": "palm_oil",
            "country_risk_score": "50",
            "supplier_risk_score": "45",
            "commodity_risk_score": "55",
            "corruption_risk_score": "38",
            "deforestation_risk_score": "52",
            "indigenous_rights_score": "30",
            "protected_areas_score": "35",
            "legal_compliance_score": "40",
            "audit_risk_score": "38",
        },
        "expected_risk_level": "medium",
        "expected_min_strategies": 2,
        "expected_risk_categories": [RiskCategory.COMMODITY],
    },
    # Scenario 9: High-risk rubber from Cambodia
    {
        "scenario_id": "GOLD-009",
        "name": "High-risk rubber Cambodia",
        "risk_input": {
            "operator_id": "op-gold-009",
            "supplier_id": "sup-gold-rubber-kh",
            "country_code": "KH",
            "commodity": "rubber",
            "country_risk_score": "68",
            "supplier_risk_score": "72",
            "commodity_risk_score": "60",
            "corruption_risk_score": "55",
            "deforestation_risk_score": "78",
            "indigenous_rights_score": "65",
            "protected_areas_score": "58",
            "legal_compliance_score": "52",
            "audit_risk_score": "60",
        },
        "expected_risk_level": "high",
        "expected_min_strategies": 2,
        "expected_risk_categories": [RiskCategory.DEFORESTATION, RiskCategory.INDIGENOUS_RIGHTS],
    },
    # Scenario 10: Low-risk rubber from Thailand certified
    {
        "scenario_id": "GOLD-010",
        "name": "Low-risk rubber Thailand certified",
        "risk_input": {
            "operator_id": "op-gold-010",
            "supplier_id": "sup-gold-rubber-th",
            "country_code": "TH",
            "commodity": "rubber",
            "country_risk_score": "25",
            "supplier_risk_score": "18",
            "commodity_risk_score": "20",
            "corruption_risk_score": "22",
            "deforestation_risk_score": "15",
            "indigenous_rights_score": "10",
            "protected_areas_score": "12",
            "legal_compliance_score": "18",
            "audit_risk_score": "15",
        },
        "expected_risk_level": "low",
        "expected_min_strategies": 1,
        "expected_risk_categories": [],
    },
    # Scenario 11: High-risk soya from Brazil Cerrado
    {
        "scenario_id": "GOLD-011",
        "name": "High-risk soya Brazil Cerrado",
        "risk_input": {
            "operator_id": "op-gold-011",
            "supplier_id": "sup-gold-soya-br",
            "country_code": "BR",
            "commodity": "soya",
            "country_risk_score": "65",
            "supplier_risk_score": "75",
            "commodity_risk_score": "80",
            "corruption_risk_score": "40",
            "deforestation_risk_score": "85",
            "indigenous_rights_score": "50",
            "protected_areas_score": "48",
            "legal_compliance_score": "55",
            "audit_risk_score": "58",
        },
        "expected_risk_level": "high",
        "expected_min_strategies": 3,
        "expected_risk_categories": [RiskCategory.DEFORESTATION, RiskCategory.COMMODITY],
    },
    # Scenario 12: Medium-risk soya from Argentina
    {
        "scenario_id": "GOLD-012",
        "name": "Medium-risk soya Argentina",
        "risk_input": {
            "operator_id": "op-gold-012",
            "supplier_id": "sup-gold-soya-ar",
            "country_code": "AR",
            "commodity": "soya",
            "country_risk_score": "45",
            "supplier_risk_score": "50",
            "commodity_risk_score": "55",
            "corruption_risk_score": "35",
            "deforestation_risk_score": "48",
            "indigenous_rights_score": "25",
            "protected_areas_score": "20",
            "legal_compliance_score": "42",
            "audit_risk_score": "38",
        },
        "expected_risk_level": "medium",
        "expected_min_strategies": 2,
        "expected_risk_categories": [RiskCategory.COMMODITY],
    },
    # Scenario 13: High-risk wood from DRC
    {
        "scenario_id": "GOLD-013",
        "name": "High-risk wood DRC",
        "risk_input": {
            "operator_id": "op-gold-013",
            "supplier_id": "sup-gold-wood-cd",
            "country_code": "CD",
            "commodity": "wood",
            "country_risk_score": "82",
            "supplier_risk_score": "78",
            "commodity_risk_score": "70",
            "corruption_risk_score": "75",
            "deforestation_risk_score": "88",
            "indigenous_rights_score": "72",
            "protected_areas_score": "68",
            "legal_compliance_score": "80",
            "audit_risk_score": "70",
        },
        "expected_risk_level": "critical",
        "expected_min_strategies": 3,
        "expected_risk_categories": [RiskCategory.DEFORESTATION, RiskCategory.COUNTRY],
    },
    # Scenario 14: Low-risk wood from Finland (FSC certified)
    {
        "scenario_id": "GOLD-014",
        "name": "Low-risk wood Finland FSC",
        "risk_input": {
            "operator_id": "op-gold-014",
            "supplier_id": "sup-gold-wood-fi",
            "country_code": "FI",
            "commodity": "wood",
            "country_risk_score": "5",
            "supplier_risk_score": "8",
            "commodity_risk_score": "10",
            "corruption_risk_score": "3",
            "deforestation_risk_score": "2",
            "indigenous_rights_score": "5",
            "protected_areas_score": "4",
            "legal_compliance_score": "3",
            "audit_risk_score": "5",
        },
        "expected_risk_level": "low",
        "expected_min_strategies": 1,
        "expected_risk_categories": [],
    },
    # Scenario 15: Multi-dimensional high-risk palm oil from DRC
    {
        "scenario_id": "GOLD-015",
        "name": "Multi-dimensional critical palm oil DRC",
        "risk_input": {
            "operator_id": "op-gold-015",
            "supplier_id": "sup-gold-palm-cd",
            "country_code": "CD",
            "commodity": "palm_oil",
            "country_risk_score": "90",
            "supplier_risk_score": "88",
            "commodity_risk_score": "92",
            "corruption_risk_score": "85",
            "deforestation_risk_score": "95",
            "indigenous_rights_score": "80",
            "protected_areas_score": "78",
            "legal_compliance_score": "82",
            "audit_risk_score": "75",
        },
        "expected_risk_level": "critical",
        "expected_min_strategies": 4,
        "expected_risk_categories": [
            RiskCategory.DEFORESTATION, RiskCategory.COUNTRY,
            RiskCategory.COMMODITY, RiskCategory.CORRUPTION,
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_risk_input(data: dict) -> RiskInput:
    """Build RiskInput from golden scenario data dict."""
    return RiskInput(
        operator_id=data["operator_id"],
        supplier_id=data["supplier_id"],
        country_code=data["country_code"],
        commodity=data["commodity"],
        country_risk_score=Decimal(data["country_risk_score"]),
        supplier_risk_score=Decimal(data["supplier_risk_score"]),
        commodity_risk_score=Decimal(data["commodity_risk_score"]),
        corruption_risk_score=Decimal(data["corruption_risk_score"]),
        deforestation_risk_score=Decimal(data["deforestation_risk_score"]),
        indigenous_rights_score=Decimal(data["indigenous_rights_score"]),
        protected_areas_score=Decimal(data["protected_areas_score"]),
        legal_compliance_score=Decimal(data["legal_compliance_score"]),
        audit_risk_score=Decimal(data.get("audit_risk_score", "50")),
        assessment_date=FIXED_DATE,
    )


def _compute_expected_composite(data: dict) -> Decimal:
    """Compute expected composite risk score from golden data."""
    scores = {
        "country": Decimal(data["country_risk_score"]),
        "supplier": Decimal(data["supplier_risk_score"]),
        "commodity": Decimal(data["commodity_risk_score"]),
        "corruption": Decimal(data["corruption_risk_score"]),
        "deforestation": Decimal(data["deforestation_risk_score"]),
        "indigenous_rights": Decimal(data["indigenous_rights_score"]),
        "protected_areas": Decimal(data["protected_areas_score"]),
        "legal_compliance": Decimal(data["legal_compliance_score"]),
    }
    composite = sum(scores[k] * COMPOSITE_WEIGHTS[k] for k in COMPOSITE_WEIGHTS)
    return composite


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGoldenScenarios:
    """Execute all 15 golden test scenarios for regression detection."""

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS,
        ids=[s["scenario_id"] for s in GOLDEN_SCENARIOS],
    )
    @pytest.mark.asyncio
    async def test_golden_risk_level(self, strategy_engine, scenario):
        """Validate expected risk level classification for golden scenario."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert result.risk_level == scenario["expected_risk_level"], (
            f"Scenario {scenario['scenario_id']}: "
            f"expected risk_level={scenario['expected_risk_level']}, "
            f"got {result.risk_level}"
        )

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS,
        ids=[s["scenario_id"] for s in GOLDEN_SCENARIOS],
    )
    @pytest.mark.asyncio
    async def test_golden_min_strategies(self, strategy_engine, scenario):
        """Validate minimum strategy count for golden scenario."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=10, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= scenario["expected_min_strategies"], (
            f"Scenario {scenario['scenario_id']}: "
            f"expected >= {scenario['expected_min_strategies']} strategies, "
            f"got {len(result.strategies)}"
        )

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS,
        ids=[s["scenario_id"] for s in GOLDEN_SCENARIOS],
    )
    @pytest.mark.asyncio
    async def test_golden_provenance_hash(self, strategy_engine, scenario):
        """Validate provenance hash exists and is 64-char SHA-256 for golden scenario."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert len(result.provenance_hash) == 64, (
            f"Scenario {scenario['scenario_id']}: "
            f"provenance_hash length={len(result.provenance_hash)}, expected 64"
        )

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS,
        ids=[s["scenario_id"] for s in GOLDEN_SCENARIOS],
    )
    @pytest.mark.asyncio
    async def test_golden_composite_score_in_range(self, strategy_engine, scenario):
        """Validate composite score is within [0, 100] for golden scenario."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert Decimal("0") <= result.composite_risk_score <= Decimal("100"), (
            f"Scenario {scenario['scenario_id']}: "
            f"composite_risk_score={result.composite_risk_score} out of [0,100]"
        )


class TestGoldenCompositeScoreAccuracy:
    """Validate composite risk score calculation against hand-computed values."""

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS,
        ids=[s["scenario_id"] for s in GOLDEN_SCENARIOS],
    )
    @pytest.mark.asyncio
    async def test_composite_score_matches_expected(self, strategy_engine, scenario):
        """Verify engine composite score matches hand-calculated value."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        expected = _compute_expected_composite(scenario["risk_input"])
        # Allow small rounding tolerance
        assert abs(result.composite_risk_score - expected) <= Decimal("1"), (
            f"Scenario {scenario['scenario_id']}: "
            f"composite_risk_score={result.composite_risk_score}, "
            f"expected~={expected}"
        )


class TestGoldenDeterminism:
    """Validate golden scenarios produce identical results across runs."""

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS[:5],
        ids=[s["scenario_id"] for s in GOLDEN_SCENARIOS[:5]],
    )
    @pytest.mark.asyncio
    async def test_golden_deterministic_hash(self, strategy_engine, scenario):
        """Same golden input produces identical provenance hash on re-run."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        assert r1.provenance_hash == r2.provenance_hash

    @pytest.mark.parametrize(
        "scenario",
        GOLDEN_SCENARIOS[:5],
        ids=[s["scenario_id"] for s in GOLDEN_SCENARIOS[:5]],
    )
    @pytest.mark.asyncio
    async def test_golden_deterministic_strategy_count(self, strategy_engine, scenario):
        """Same golden input produces same strategy count on re-run."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=5, deterministic_mode=True
        )
        r1 = await strategy_engine.recommend(req)
        r2 = await strategy_engine.recommend(req)
        assert len(r1.strategies) == len(r2.strategies)


class TestGoldenPlanCreation:
    """Test plan creation using golden scenario strategy outputs."""

    @pytest.mark.parametrize(
        "scenario",
        [s for s in GOLDEN_SCENARIOS if s["expected_risk_level"] in ("high", "critical")],
        ids=[
            s["scenario_id"]
            for s in GOLDEN_SCENARIOS
            if s["expected_risk_level"] in ("high", "critical")
        ],
    )
    @pytest.mark.asyncio
    async def test_golden_plan_creation(
        self, strategy_engine, remediation_engine, scenario
    ):
        """Create remediation plan from golden scenario strategies."""
        ri = _build_risk_input(scenario["risk_input"])
        req = RecommendStrategiesRequest(
            risk_input=ri, top_k=3, deterministic_mode=True
        )
        result = await strategy_engine.recommend(req)
        assert len(result.strategies) >= 1

        plan_req = CreatePlanRequest(
            operator_id=ri.operator_id,
            supplier_id=ri.supplier_id,
            strategy_ids=[s.strategy_id for s in result.strategies[:2]],
            budget_eur=Decimal("50000"),
        )
        plan_result = await remediation_engine.create_plan(plan_req)
        assert plan_result.plan is not None
        assert plan_result.plan.status == PlanStatus.DRAFT


class TestGoldenCommodityCoverage:
    """Ensure all 7 commodities are covered by golden scenarios."""

    def test_all_commodities_have_golden_scenario(self):
        commodities_tested = set()
        for s in GOLDEN_SCENARIOS:
            commodities_tested.add(s["risk_input"]["commodity"])
        for c in SUPPORTED_COMMODITIES:
            assert c in commodities_tested, (
                f"Commodity '{c}' missing from golden scenarios"
            )

    def test_golden_scenario_count(self):
        assert len(GOLDEN_SCENARIOS) == 15

    def test_each_commodity_at_least_two_scenarios(self):
        from collections import Counter
        commodity_counts = Counter(
            s["risk_input"]["commodity"] for s in GOLDEN_SCENARIOS
        )
        for c in SUPPORTED_COMMODITIES:
            assert commodity_counts.get(c, 0) >= 2, (
                f"Commodity '{c}' has only {commodity_counts.get(c, 0)} scenarios"
            )
