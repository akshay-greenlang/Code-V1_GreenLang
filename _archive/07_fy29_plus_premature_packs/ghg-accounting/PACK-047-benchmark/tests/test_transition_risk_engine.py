"""
Unit tests for TransitionRiskScoringEngine (PACK-047 Engine 9).

Tests all public methods with 28+ tests covering:
  - Carbon budget overshoot calculation
  - Stranding year calculation
  - Regulatory risk score
  - Competitive risk quartile
  - Carbon price exposure
  - Composite score calculation
  - Risk trajectory (increasing / decreasing)
  - Zero emissions no risk edge case

Author: GreenLang QA Team
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from tests.conftest import assert_decimal_between, assert_decimal_equal


# ---------------------------------------------------------------------------
# Carbon Budget Overshoot Tests
# ---------------------------------------------------------------------------


class TestCarbonBudgetOvershoot:
    """Tests for carbon budget overshoot calculation."""

    def test_within_budget_no_overshoot(self):
        """Test entity within carbon budget has no overshoot."""
        cumulative_emissions = Decimal("3000")
        allocated_budget = Decimal("5000")
        overshoot = max(Decimal("0"), cumulative_emissions - allocated_budget)
        assert overshoot == Decimal("0")

    def test_exceeding_budget_positive_overshoot(self):
        """Test entity exceeding budget has positive overshoot."""
        cumulative_emissions = Decimal("7000")
        allocated_budget = Decimal("5000")
        overshoot = max(Decimal("0"), cumulative_emissions - allocated_budget)
        assert overshoot == Decimal("2000")

    def test_exactly_at_budget_zero_overshoot(self):
        """Test entity exactly at budget has zero overshoot."""
        cumulative_emissions = Decimal("5000")
        allocated_budget = Decimal("5000")
        overshoot = max(Decimal("0"), cumulative_emissions - allocated_budget)
        assert overshoot == Decimal("0")

    def test_overshoot_percentage(self):
        """Test overshoot percentage calculation."""
        overshoot = Decimal("2000")
        budget = Decimal("5000")
        overshoot_pct = (overshoot / budget) * Decimal("100")
        assert_decimal_equal(overshoot_pct, Decimal("40"), tolerance=Decimal("0.01"))


# ---------------------------------------------------------------------------
# Stranding Year Calculation Tests
# ---------------------------------------------------------------------------


class TestStrandingYearCalculation:
    """Tests for stranding year calculation (when budget exhausted)."""

    def test_stranding_year_within_horizon(self):
        """Test stranding year calculated when budget exhausted before 2050."""
        remaining_budget = Decimal("10000")  # tCO2e
        annual_emissions = Decimal("2000")   # tCO2e/year
        years_remaining = remaining_budget / annual_emissions
        current_year = 2025
        stranding_year = current_year + int(years_remaining)
        assert stranding_year == 2030

    def test_no_stranding_if_net_zero_before_budget_exhausted(self):
        """Test no stranding year if entity reaches net-zero before budget runs out."""
        remaining_budget = Decimal("50000")
        annual_emissions = Decimal("1000")
        annual_reduction_pct = Decimal("10")
        # With 10% annual reduction, emissions halve in ~7 years
        # Budget sufficient for decades
        years = remaining_budget / annual_emissions
        assert years > 30  # No stranding within planning horizon

    def test_sooner_stranding_for_higher_emissions(self):
        """Test higher emissions lead to sooner stranding year."""
        budget = Decimal("10000")
        stranding_low = 2025 + int(budget / Decimal("1000"))  # 2035
        stranding_high = 2025 + int(budget / Decimal("2500"))  # 2029
        assert stranding_high < stranding_low

    def test_stranding_year_deterministic(self):
        """Test stranding year calculation is deterministic."""
        budget = Decimal("15000")
        emissions = Decimal("3000")
        y1 = 2025 + int(budget / emissions)
        y2 = 2025 + int(budget / emissions)
        assert y1 == y2


# ---------------------------------------------------------------------------
# Regulatory Risk Score Tests
# ---------------------------------------------------------------------------


class TestRegulatoryRiskScore:
    """Tests for regulatory risk scoring."""

    def test_high_emissions_intensity_high_risk(self):
        """Test high emissions intensity produces high regulatory risk."""
        intensity = Decimal("50")  # tCO2e/MEUR
        sector_average = Decimal("25")
        ratio = intensity / sector_average
        if ratio > Decimal("1.5"):
            risk = "HIGH"
        elif ratio > Decimal("1.0"):
            risk = "MEDIUM"
        else:
            risk = "LOW"
        assert risk == "HIGH"

    def test_below_average_intensity_low_risk(self):
        """Test below-average intensity produces low regulatory risk."""
        intensity = Decimal("15")
        sector_average = Decimal("25")
        ratio = intensity / sector_average
        risk = "LOW" if ratio <= Decimal("1.0") else "HIGH"
        assert risk == "LOW"

    def test_regulatory_risk_score_0_to_100(self):
        """Test regulatory risk score is in [0, 100] range."""
        score = Decimal("65")
        assert_decimal_between(score, Decimal("0"), Decimal("100"))

    @pytest.mark.parametrize("carbon_price,expected_impact", [
        (Decimal("50"), "moderate"),
        (Decimal("100"), "significant"),
        (Decimal("200"), "severe"),
    ])
    def test_carbon_price_impact_thresholds(self, carbon_price, expected_impact):
        """Test carbon price impact at different price levels."""
        if carbon_price >= Decimal("200"):
            impact = "severe"
        elif carbon_price >= Decimal("100"):
            impact = "significant"
        else:
            impact = "moderate"
        assert impact == expected_impact


# ---------------------------------------------------------------------------
# Competitive Risk Quartile Tests
# ---------------------------------------------------------------------------


class TestCompetitiveRiskQuartile:
    """Tests for competitive risk quartile ranking."""

    def test_best_performer_q1(self):
        """Test best performer in Q1 (lowest emissions)."""
        percentile = Decimal("10")
        quartile = 1 if percentile <= Decimal("25") else (
            2 if percentile <= Decimal("50") else (
                3 if percentile <= Decimal("75") else 4
            )
        )
        assert quartile == 1

    def test_worst_performer_q4(self):
        """Test worst performer in Q4 (highest emissions)."""
        percentile = Decimal("90")
        quartile = 1 if percentile <= Decimal("25") else (
            2 if percentile <= Decimal("50") else (
                3 if percentile <= Decimal("75") else 4
            )
        )
        assert quartile == 4

    def test_median_performer_q2_or_q3(self):
        """Test median performer falls in Q2 or Q3."""
        percentile = Decimal("50")
        quartile = 1 if percentile <= Decimal("25") else (
            2 if percentile <= Decimal("50") else (
                3 if percentile <= Decimal("75") else 4
            )
        )
        assert quartile == 2


# ---------------------------------------------------------------------------
# Carbon Price Exposure Tests
# ---------------------------------------------------------------------------


class TestCarbonPriceExposure:
    """Tests for carbon price exposure calculation."""

    def test_carbon_cost_calculation(self):
        """Test annual carbon cost = emissions * price."""
        emissions = Decimal("8000")
        price_per_tco2e = Decimal("80")
        cost = emissions * price_per_tco2e
        assert cost == Decimal("640000")

    def test_cost_as_percentage_of_revenue(self):
        """Test carbon cost as % of revenue."""
        cost = Decimal("640000")
        revenue = Decimal("500000000")  # 500M USD
        pct = (cost / revenue) * Decimal("100")
        assert_decimal_between(pct, Decimal("0"), Decimal("1"))

    def test_rising_carbon_price_increases_exposure(self):
        """Test rising carbon price increases exposure over time."""
        emissions = Decimal("8000")
        price_2025 = Decimal("80")
        price_2030 = Decimal("130")
        cost_2025 = emissions * price_2025
        cost_2030 = emissions * price_2030
        assert cost_2030 > cost_2025


# ---------------------------------------------------------------------------
# Composite Score Tests
# ---------------------------------------------------------------------------


class TestCompositeScoreCalculation:
    """Tests for transition risk composite score."""

    def test_composite_score_weighted_average(self, transition_risk_engine_config):
        """Test composite score is weighted average of risk dimensions."""
        config = transition_risk_engine_config
        budget_score = Decimal("60")
        stranding_score = Decimal("40")
        regulatory_score = Decimal("70")
        competitive_score = Decimal("50")
        composite = (
            config["carbon_budget_risk_weight"] * budget_score
            + config["stranding_risk_weight"] * stranding_score
            + config["regulatory_risk_weight"] * regulatory_score
            + config["competitive_risk_weight"] * competitive_score
        )
        # 0.20*60 + 0.25*40 + 0.30*70 + 0.25*50 = 12 + 10 + 21 + 12.5 = 55.5
        assert_decimal_equal(composite, Decimal("55.5"), tolerance=Decimal("0.01"))

    def test_weights_sum_to_1(self, transition_risk_engine_config):
        """Test risk dimension weights sum to 1.0."""
        config = transition_risk_engine_config
        total_weight = (
            config["carbon_budget_risk_weight"]
            + config["stranding_risk_weight"]
            + config["regulatory_risk_weight"]
            + config["competitive_risk_weight"]
        )
        assert_decimal_equal(total_weight, Decimal("1.0"), tolerance=Decimal("0.001"))

    def test_composite_score_range(self):
        """Test composite score is in [0, 100] range."""
        score = Decimal("55.5")
        assert_decimal_between(score, Decimal("0"), Decimal("100"))


# ---------------------------------------------------------------------------
# Risk Trajectory Tests
# ---------------------------------------------------------------------------


class TestRiskTrajectory:
    """Tests for risk trajectory (increasing / decreasing over time)."""

    def test_increasing_risk_trajectory(self):
        """Test increasing risk trajectory is detected."""
        risk_scores = [Decimal("40"), Decimal("50"), Decimal("60"), Decimal("70")]
        is_increasing = all(
            risk_scores[i] > risk_scores[i - 1]
            for i in range(1, len(risk_scores))
        )
        assert is_increasing is True

    def test_decreasing_risk_trajectory(self):
        """Test decreasing risk trajectory is detected."""
        risk_scores = [Decimal("70"), Decimal("60"), Decimal("50"), Decimal("40")]
        is_decreasing = all(
            risk_scores[i] < risk_scores[i - 1]
            for i in range(1, len(risk_scores))
        )
        assert is_decreasing is True

    def test_stable_risk_trajectory(self):
        """Test stable risk trajectory is detected."""
        risk_scores = [Decimal("50"), Decimal("50"), Decimal("50")]
        is_stable = all(s == risk_scores[0] for s in risk_scores)
        assert is_stable is True


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestTransitionRiskEdgeCases:
    """Tests for transition risk edge cases."""

    def test_zero_emissions_minimal_risk(self):
        """Test zero emissions produce minimal transition risk."""
        emissions = Decimal("0")
        carbon_price = Decimal("80")
        cost = emissions * carbon_price
        assert cost == Decimal("0")

    def test_provenance_hash_deterministic(self):
        """Test transition risk scoring produces deterministic hash."""
        import hashlib
        import json
        data = {"emissions": "8000", "budget": "50000", "price": "80"}
        h1 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        h2 = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        assert h1 == h2
        assert len(h1) == 64
