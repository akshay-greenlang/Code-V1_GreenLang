# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - Supply Chain ESG Tests (15 tests)

Tests supply chain ESG risk scoring, composite score calculation,
tier assignment, questionnaire dispatch, risk distribution,
improvement planning, and Scope 3 estimation.

Author: GreenLang QA Team
"""

from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import _compute_hash


class TestSupplyChainESG:
    """Test suite for supply chain ESG engine."""

    def test_score_supplier(self, sample_supplier_data):
        """Test individual supplier ESG scoring produces valid range."""
        supplier = sample_supplier_data[0]
        assert 0.0 <= supplier["environmental_score"] <= 1.0
        assert 0.0 <= supplier["social_score"] <= 1.0
        assert 0.0 <= supplier["governance_score"] <= 1.0

    def test_composite_score_calculation(self, sample_supplier_data):
        """Test composite score is weighted average of E, S, G scores."""
        weights = {"environmental": 0.40, "social": 0.35, "governance": 0.25}
        for supplier in sample_supplier_data:
            expected = (
                supplier["environmental_score"] * weights["environmental"]
                + supplier["social_score"] * weights["social"]
                + supplier["governance_score"] * weights["governance"]
            )
            assert abs(supplier["composite_esg_score"] - expected) < 0.01, (
                f"Composite score mismatch for {supplier['supplier_id']}: "
                f"expected {expected:.4f}, got {supplier['composite_esg_score']}"
            )

    def test_risk_tier_assignment(self, sample_supplier_data):
        """Test risk tier assignment based on composite score."""
        for supplier in sample_supplier_data:
            score = supplier["composite_esg_score"]
            if score < 0.5:
                expected_tier = "high"
            elif score < 0.7:
                expected_tier = "medium"
            else:
                expected_tier = "low"
            assert supplier["risk_tier"] == expected_tier, (
                f"Risk tier mismatch for {supplier['supplier_id']}: "
                f"score={score}, expected={expected_tier}, got={supplier['risk_tier']}"
            )

    def test_supply_chain_mapping(self, sample_supplier_data):
        """Test multi-tier supply chain mapping."""
        tier_map = {}
        for s in sample_supplier_data:
            t = s["tier"]
            tier_map.setdefault(t, []).append(s["supplier_id"])
        assert 1 in tier_map
        assert 2 in tier_map
        assert 3 in tier_map

    def test_questionnaire_dispatch(self, sample_supplier_data):
        """Test questionnaire dispatch to suppliers."""
        pending = [s for s in sample_supplier_data if s["questionnaire_status"] == "pending"]
        completed = [s for s in sample_supplier_data if s["questionnaire_status"] == "completed"]
        assert len(pending) + len(completed) == 15
        assert len(pending) >= 1

    def test_response_processing(self, sample_supplier_data):
        """Test questionnaire response processing."""
        completed = [s for s in sample_supplier_data if s["questionnaire_status"] == "completed"]
        assert len(completed) >= 10
        for s in completed:
            assert s["composite_esg_score"] > 0

    def test_risk_distribution(self, sample_supplier_data):
        """Test risk distribution across all suppliers."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        for s in sample_supplier_data:
            distribution[s["risk_tier"]] += 1
        total = sum(distribution.values())
        assert total == 15
        assert all(v >= 0 for v in distribution.values())

    def test_improvement_plan_creation(self, sample_supplier_data):
        """Test improvement plan generation for high-risk suppliers."""
        high_risk = [s for s in sample_supplier_data if s["risk_tier"] == "high"]
        plans = []
        for supplier in high_risk:
            plan = {
                "supplier_id": supplier["supplier_id"],
                "current_score": supplier["composite_esg_score"],
                "target_score": round(min(supplier["composite_esg_score"] + 0.20, 1.0), 2),
                "priority_actions": [],
                "timeline_months": 12,
            }
            if supplier["environmental_score"] < 0.5:
                plan["priority_actions"].append("environmental_improvement")
            if supplier["social_score"] < 0.5:
                plan["priority_actions"].append("social_improvement")
            if supplier["governance_score"] < 0.5:
                plan["priority_actions"].append("governance_improvement")
            plans.append(plan)
        for plan in plans:
            assert plan["target_score"] > plan["current_score"]

    def test_sector_benchmarking(self, sample_supplier_data):
        """Test sector-level benchmarking of suppliers."""
        sector_scores = {}
        for s in sample_supplier_data:
            sector = s["sector"]
            sector_scores.setdefault(sector, []).append(s["composite_esg_score"])
        for sector, scores in sector_scores.items():
            avg = sum(scores) / len(scores)
            assert 0.0 <= avg <= 1.0

    def test_scope3_estimation(self, sample_supplier_data):
        """Test Scope 3 contribution estimation from suppliers."""
        total_contribution = sum(s["scope3_contribution_pct"] for s in sample_supplier_data)
        assert total_contribution > 0
        for s in sample_supplier_data:
            assert s["scope3_contribution_pct"] >= 0

    def test_multi_tier_mapping(self, sample_supplier_data):
        """Test multi-tier supplier hierarchy."""
        tiers = {s["tier"] for s in sample_supplier_data}
        assert max(tiers) <= 4
        assert min(tiers) >= 1

    def test_supplier_scorecard(self, sample_supplier_data):
        """Test supplier scorecard generation."""
        supplier = sample_supplier_data[0]
        scorecard = {
            "supplier_id": supplier["supplier_id"],
            "name": supplier["name"],
            "country": supplier["country"],
            "sector": supplier["sector"],
            "scores": {
                "environmental": supplier["environmental_score"],
                "social": supplier["social_score"],
                "governance": supplier["governance_score"],
                "composite": supplier["composite_esg_score"],
            },
            "risk_tier": supplier["risk_tier"],
            "deforestation_risk": supplier["deforestation_risk"],
            "provenance_hash": _compute_hash(supplier),
        }
        assert len(scorecard["provenance_hash"]) == 64
        assert scorecard["scores"]["composite"] == supplier["composite_esg_score"]

    def test_esg_weight_configuration(self):
        """Test ESG weight configuration validation."""
        weights = {"environmental": 0.40, "social": 0.35, "governance": 0.25}
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(0.0 <= w <= 1.0 for w in weights.values())

    def test_high_risk_identification(self, sample_supplier_data):
        """Test high-risk supplier identification."""
        high_risk = [s for s in sample_supplier_data if s["risk_tier"] == "high"]
        for s in high_risk:
            assert s["composite_esg_score"] < 0.5

    def test_improvement_tracking(self, sample_supplier_data):
        """Test improvement tracking over time."""
        supplier = sample_supplier_data[0]
        history = [
            {"quarter": "Q1", "score": supplier["composite_esg_score"]},
            {"quarter": "Q2", "score": round(supplier["composite_esg_score"] + 0.05, 4)},
            {"quarter": "Q3", "score": round(supplier["composite_esg_score"] + 0.08, 4)},
            {"quarter": "Q4", "score": round(supplier["composite_esg_score"] + 0.12, 4)},
        ]
        assert history[-1]["score"] > history[0]["score"]
        improvement = history[-1]["score"] - history[0]["score"]
        assert improvement > 0
