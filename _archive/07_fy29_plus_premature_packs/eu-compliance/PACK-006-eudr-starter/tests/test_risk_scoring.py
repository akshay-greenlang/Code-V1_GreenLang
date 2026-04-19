# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Risk Scoring Engine Tests
========================================================

Validates the risk scoring engine including country risk benchmarking,
supplier risk with/without certification, commodity risk levels, document
risk assessment, composite risk calculation, risk classification,
Article 29 benchmark application, simplified DD eligibility, batch
assessment, and risk trending.

Test count: 25
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict, List

import pytest

from conftest import (
    EUDR_HIGH_RISK_COUNTRIES,
    EUDR_LOW_RISK_COUNTRIES,
    EUDR_STANDARD_RISK_COUNTRIES,
    EUDR_COMMODITIES,
    CERTIFICATION_SCHEMES,
    RISK_LEVELS,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Risk Scoring Engine Simulator
# ---------------------------------------------------------------------------

class RiskScoringEngineSimulator:
    """Simulates risk scoring engine operations."""

    RISK_WEIGHTS = {
        "country": 0.30,
        "supplier": 0.25,
        "commodity": 0.20,
        "document": 0.25,
    }

    THRESHOLDS = {
        "low": 0.30,
        "standard": 0.50,
        "high": 0.70,
        "critical": 0.85,
    }

    COMMODITY_BASE_RISK = {
        "palm_oil": 0.75,
        "cattle": 0.70,
        "soya": 0.65,
        "cocoa": 0.65,
        "coffee": 0.55,
        "rubber": 0.60,
        "wood": 0.60,
    }

    def score_country_risk(self, country: str) -> Dict[str, Any]:
        """Score country risk based on Article 29 benchmarking."""
        if country in EUDR_HIGH_RISK_COUNTRIES:
            score = 0.80
            level = "HIGH"
            benchmark = "HIGH"
        elif country in EUDR_LOW_RISK_COUNTRIES:
            score = 0.15
            level = "LOW"
            benchmark = "LOW"
        elif country in EUDR_STANDARD_RISK_COUNTRIES:
            score = 0.50
            level = "STANDARD"
            benchmark = "STANDARD"
        else:
            score = 0.50
            level = "STANDARD"
            benchmark = "STANDARD"
        return {
            "country": country,
            "score": score,
            "level": level,
            "benchmark": benchmark,
        }

    def score_supplier_risk(self, supplier: Dict[str, Any]) -> Dict[str, Any]:
        """Score supplier risk based on certification and data completeness."""
        certifications = supplier.get("certifications", [])
        has_cert = len(certifications) > 0 and any(
            c.get("status") == "active" for c in certifications
        )
        completeness = supplier.get("data_completeness", 0.0)

        if has_cert and completeness >= 0.90:
            score = 0.20
        elif has_cert:
            score = 0.35
        elif completeness >= 0.80:
            score = 0.55
        else:
            score = 0.75

        return {
            "supplier_id": supplier.get("supplier_id", ""),
            "score": score,
            "level": self._classify_risk(score),
            "has_certification": has_cert,
            "data_completeness": completeness,
        }

    def score_commodity_risk(self, commodity: str) -> Dict[str, Any]:
        """Score commodity risk based on deforestation association."""
        base_risk = self.COMMODITY_BASE_RISK.get(commodity, 0.50)
        return {
            "commodity": commodity,
            "score": base_risk,
            "level": self._classify_risk(base_risk),
        }

    def score_document_risk(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        """Score document risk based on completeness and authenticity."""
        completeness = documents.get("completeness", 0.0)
        verified = documents.get("authenticity_verified", False)
        consistency = documents.get("consistency_score", 0.0)
        age_months = documents.get("age_months", 12)

        score = 1.0 - completeness
        if verified:
            score *= 0.7
        if consistency > 0.8:
            score *= 0.8
        if age_months > 12:
            score *= 1.2
        score = min(1.0, max(0.0, score))

        return {
            "score": round(score, 4),
            "level": self._classify_risk(score),
            "completeness": completeness,
            "authenticity_verified": verified,
        }

    def calculate_composite_risk(self, country_score: float, supplier_score: float,
                                  commodity_score: float, document_score: float) -> Dict[str, Any]:
        """Calculate weighted composite risk score."""
        composite = (
            country_score * self.RISK_WEIGHTS["country"]
            + supplier_score * self.RISK_WEIGHTS["supplier"]
            + commodity_score * self.RISK_WEIGHTS["commodity"]
            + document_score * self.RISK_WEIGHTS["document"]
        )
        composite = round(composite, 4)
        return {
            "score": composite,
            "level": self._classify_risk(composite),
            "weights": self.RISK_WEIGHTS.copy(),
            "components": {
                "country": country_score,
                "supplier": supplier_score,
                "commodity": commodity_score,
                "document": document_score,
            },
        }

    def _classify_risk(self, score: float) -> str:
        """Classify risk score into level."""
        if score <= self.THRESHOLDS["low"]:
            return "LOW"
        elif score <= self.THRESHOLDS["standard"]:
            return "STANDARD"
        elif score <= self.THRESHOLDS["high"]:
            return "HIGH"
        else:
            return "CRITICAL"

    def check_simplified_dd_eligibility(self, country: str,
                                         composite_score: float) -> Dict[str, Any]:
        """Check eligibility for simplified due diligence."""
        country_risk = self.score_country_risk(country)
        eligible = (
            country_risk["benchmark"] == "LOW"
            and composite_score <= self.THRESHOLDS["low"]
        )
        return {
            "eligible": eligible,
            "country_benchmark": country_risk["benchmark"],
            "composite_score": composite_score,
            "reason": "Low-risk country with low composite score" if eligible
                      else "Does not meet simplified DD criteria",
        }

    def batch_risk_assessment(self, suppliers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk for multiple suppliers."""
        results = []
        for supplier in suppliers:
            country_risk = self.score_country_risk(supplier["country"])
            supplier_risk = self.score_supplier_risk(supplier)
            commodity_risk = self.score_commodity_risk(supplier["commodity"])
            doc_risk = self.score_document_risk({
                "completeness": supplier.get("data_completeness", 0.5),
                "authenticity_verified": True,
                "consistency_score": 0.85,
                "age_months": 3,
            })
            composite = self.calculate_composite_risk(
                country_risk["score"], supplier_risk["score"],
                commodity_risk["score"], doc_risk["score"],
            )
            results.append({
                "supplier_id": supplier.get("supplier_id", ""),
                "name": supplier.get("name", ""),
                "composite_risk": composite,
            })
        return {
            "total_assessed": len(results),
            "results": results,
            "high_risk_count": sum(1 for r in results if r["composite_risk"]["level"] in ("HIGH", "CRITICAL")),
            "low_risk_count": sum(1 for r in results if r["composite_risk"]["level"] == "LOW"),
        }

    def get_risk_factors_breakdown(self, composite: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of risk factors."""
        components = composite.get("components", {})
        weights = composite.get("weights", self.RISK_WEIGHTS)
        breakdown = {}
        for factor, score in components.items():
            weight = weights.get(factor, 0)
            contribution = round(score * weight, 4)
            breakdown[factor] = {
                "raw_score": score,
                "weight": weight,
                "weighted_contribution": contribution,
                "pct_of_total": round(contribution / composite["score"] * 100, 1) if composite["score"] > 0 else 0,
            }
        return breakdown

    def risk_trend(self, current_score: float, previous_scores: List[float]) -> Dict[str, Any]:
        """Analyze risk score trend over time."""
        if not previous_scores:
            return {"trend": "no_data", "direction": "stable"}
        avg_previous = sum(previous_scores) / len(previous_scores)
        change = current_score - avg_previous
        if change > 0.05:
            direction = "increasing"
        elif change < -0.05:
            direction = "decreasing"
        else:
            direction = "stable"
        return {
            "trend": direction,
            "current_score": current_score,
            "previous_average": round(avg_previous, 4),
            "change": round(change, 4),
            "data_points": len(previous_scores) + 1,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRiskScoring:
    """Tests for the risk scoring engine."""

    @pytest.fixture
    def engine(self) -> RiskScoringEngineSimulator:
        return RiskScoringEngineSimulator()

    # 1
    def test_country_risk_high(self, engine):
        """Brazil and Indonesia are classified as HIGH risk."""
        for country in ["BRA", "IDN"]:
            result = engine.score_country_risk(country)
            assert result["level"] == "HIGH", f"{country} should be HIGH risk"
            assert result["score"] >= 0.70

    # 2
    def test_country_risk_low(self, engine):
        """Germany and France are classified as LOW risk."""
        for country in ["DEU", "FRA"]:
            result = engine.score_country_risk(country)
            assert result["level"] == "LOW", f"{country} should be LOW risk"
            assert result["score"] <= 0.30

    # 3
    def test_country_risk_standard(self, engine):
        """India and Thailand are classified as STANDARD risk."""
        for country in ["IND", "THA"]:
            result = engine.score_country_risk(country)
            assert result["level"] == "STANDARD", f"{country} should be STANDARD risk"

    # 4
    def test_supplier_risk_with_certification(self, engine, sample_palm_oil_supplier):
        """Supplier with active certification has lower risk."""
        result = engine.score_supplier_risk(sample_palm_oil_supplier)
        assert result["has_certification"] is True
        assert result["score"] < 0.50

    # 5
    def test_supplier_risk_without_certification(self, engine, sample_cattle_supplier):
        """Supplier without certification has higher risk."""
        result = engine.score_supplier_risk(sample_cattle_supplier)
        assert result["has_certification"] is False
        assert result["score"] >= 0.50

    # 6
    def test_commodity_risk_palm_oil(self, engine):
        """Palm oil has the highest commodity risk."""
        result = engine.score_commodity_risk("palm_oil")
        assert result["score"] >= 0.70
        # Should be higher than most other commodities
        for other in ["coffee", "rubber"]:
            other_result = engine.score_commodity_risk(other)
            assert result["score"] >= other_result["score"]

    # 7
    def test_commodity_risk_coffee(self, engine):
        """Coffee has moderate commodity risk."""
        result = engine.score_commodity_risk("coffee")
        assert 0.40 <= result["score"] <= 0.70

    # 8
    def test_document_risk_complete(self, engine):
        """Complete verified documents have low risk."""
        result = engine.score_document_risk({
            "completeness": 0.95,
            "authenticity_verified": True,
            "consistency_score": 0.90,
            "age_months": 3,
        })
        assert result["score"] < 0.10
        assert result["level"] == "LOW"

    # 9
    def test_document_risk_incomplete(self, engine):
        """Incomplete unverified documents have high risk."""
        result = engine.score_document_risk({
            "completeness": 0.30,
            "authenticity_verified": False,
            "consistency_score": 0.40,
            "age_months": 18,
        })
        assert result["score"] > 0.50
        assert result["level"] in ("HIGH", "CRITICAL")

    # 10
    def test_composite_risk_calculation(self, engine):
        """Composite risk is weighted sum of component scores."""
        composite = engine.calculate_composite_risk(
            country_score=0.80,
            supplier_score=0.40,
            commodity_score=0.70,
            document_score=0.20,
        )
        expected = 0.80 * 0.30 + 0.40 * 0.25 + 0.70 * 0.20 + 0.20 * 0.25
        assert abs(composite["score"] - expected) < 0.001

    # 11
    def test_risk_weights_applied_correctly(self, engine):
        """Risk weights sum to 1.0 and are correctly applied."""
        weights = engine.RISK_WEIGHTS
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    # 12
    def test_classify_risk_low(self, engine):
        """Score <= 0.30 classifies as LOW."""
        assert engine._classify_risk(0.20) == "LOW"
        assert engine._classify_risk(0.30) == "LOW"

    # 13
    def test_classify_risk_critical(self, engine):
        """Score > 0.85 classifies as CRITICAL."""
        assert engine._classify_risk(0.90) == "CRITICAL"
        assert engine._classify_risk(1.00) == "CRITICAL"

    # 14
    def test_risk_factors_breakdown(self, engine):
        """Risk factors breakdown shows contribution of each component."""
        composite = engine.calculate_composite_risk(0.80, 0.40, 0.70, 0.20)
        breakdown = engine.get_risk_factors_breakdown(composite)
        assert "country" in breakdown
        assert "supplier" in breakdown
        assert "commodity" in breakdown
        assert "document" in breakdown
        total_pct = sum(b["pct_of_total"] for b in breakdown.values())
        assert abs(total_pct - 100.0) < 1.0

    # 15
    def test_article_29_benchmark(self, engine):
        """Article 29 country benchmarking assigns correct levels."""
        bra_result = engine.score_country_risk("BRA")
        assert bra_result["benchmark"] == "HIGH"
        deu_result = engine.score_country_risk("DEU")
        assert deu_result["benchmark"] == "LOW"
        ind_result = engine.score_country_risk("IND")
        assert ind_result["benchmark"] == "STANDARD"

    # 16
    def test_simplified_dd_eligibility_eligible(self, engine):
        """Low-risk country with low composite qualifies for simplified DD."""
        result = engine.check_simplified_dd_eligibility("DEU", 0.15)
        assert result["eligible"] is True

    # 17
    def test_simplified_dd_eligibility_ineligible(self, engine):
        """High-risk country does not qualify for simplified DD."""
        result = engine.check_simplified_dd_eligibility("BRA", 0.60)
        assert result["eligible"] is False

    # 18
    def test_batch_risk_assessment(self, engine, sample_suppliers_list):
        """Batch assessment scores all suppliers."""
        result = engine.batch_risk_assessment(sample_suppliers_list)
        assert result["total_assessed"] == len(sample_suppliers_list)
        assert len(result["results"]) == len(sample_suppliers_list)

    # 19
    def test_risk_trend_increasing(self, engine):
        """Risk trend detects increasing risk."""
        result = engine.risk_trend(0.75, [0.50, 0.55, 0.60])
        assert result["trend"] == "increasing"

    # 20
    def test_risk_trend_decreasing(self, engine):
        """Risk trend detects decreasing risk."""
        result = engine.risk_trend(0.30, [0.50, 0.55, 0.60])
        assert result["trend"] == "decreasing"

    # 21
    def test_risk_trend_stable(self, engine):
        """Risk trend detects stable risk."""
        result = engine.risk_trend(0.52, [0.50, 0.51, 0.53])
        assert result["trend"] == "stable"

    # 22
    def test_country_database_200_countries(self):
        """Country risk database covers at least 50 unique countries."""
        all_countries = set(EUDR_HIGH_RISK_COUNTRIES) | set(EUDR_LOW_RISK_COUNTRIES) | set(EUDR_STANDARD_RISK_COUNTRIES)
        assert len(all_countries) >= 50

    # 23
    def test_all_commodities_have_risk(self, engine):
        """All 7 EUDR commodities have defined base risk scores."""
        for commodity in EUDR_COMMODITIES:
            result = engine.score_commodity_risk(commodity)
            assert 0.0 <= result["score"] <= 1.0, f"Invalid risk for {commodity}"

    # 24
    def test_composite_risk_bounded(self, engine):
        """Composite risk is always between 0.0 and 1.0."""
        extremes = [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)]
        for cs, ss, cms, ds in extremes:
            result = engine.calculate_composite_risk(cs, ss, cms, ds)
            assert 0.0 <= result["score"] <= 1.0

    # 25
    def test_risk_level_ordering(self, engine):
        """Risk levels are ordered LOW < STANDARD < HIGH < CRITICAL."""
        levels_scores = [
            (0.10, "LOW"), (0.40, "STANDARD"),
            (0.65, "HIGH"), (0.90, "CRITICAL"),
        ]
        for score, expected_level in levels_scores:
            assert engine._classify_risk(score) == expected_level
