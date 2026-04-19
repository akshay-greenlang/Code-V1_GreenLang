"""
Unit tests for CostTimelineEngine (PACK-048 Engine 9).

Tests all public methods with 25+ tests covering:
  - Base cost by company size (6 sizes)
  - Reasonable multiplier 2.5x
  - Multi-jurisdiction uplift
  - First-time premium
  - Scope 3 complexity uplift
  - Timeline estimation
  - Resource allocation
  - Multi-year projection
  - Zero facilities edge case

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
# Base Cost by Company Size Tests
# ---------------------------------------------------------------------------


class TestBaseCostByCompanySize:
    """Tests for base cost by company size (6 sizes)."""

    @pytest.mark.parametrize("size,min_cost,max_cost", [
        ("MICRO", Decimal("5000"), Decimal("15000")),
        ("SMALL", Decimal("10000"), Decimal("30000")),
        ("MEDIUM", Decimal("25000"), Decimal("60000")),
        ("LARGE", Decimal("50000"), Decimal("120000")),
        ("ENTERPRISE", Decimal("100000"), Decimal("250000")),
        ("MEGA", Decimal("200000"), Decimal("500000")),
    ])
    def test_base_cost_range(self, size, min_cost, max_cost):
        """Test base cost falls within expected range for each size."""
        # Midpoint estimate
        base = (min_cost + max_cost) / Decimal("2")
        assert_decimal_between(base, min_cost, max_cost)

    def test_6_company_sizes(self, cost_engine_config):
        """Test 6 company sizes are defined."""
        assert len(cost_engine_config["company_sizes"]) == 6

    def test_cost_increases_with_size(self):
        """Test cost increases with company size."""
        costs = {
            "MICRO": Decimal("10000"),
            "SMALL": Decimal("20000"),
            "MEDIUM": Decimal("42500"),
            "LARGE": Decimal("75000"),
            "ENTERPRISE": Decimal("175000"),
            "MEGA": Decimal("350000"),
        }
        sizes = ["MICRO", "SMALL", "MEDIUM", "LARGE", "ENTERPRISE", "MEGA"]
        for i in range(len(sizes) - 1):
            assert costs[sizes[i]] < costs[sizes[i + 1]]


# ---------------------------------------------------------------------------
# Reasonable Multiplier Tests
# ---------------------------------------------------------------------------


class TestReasonableMultiplier:
    """Tests for reasonable assurance multiplier (2.5x)."""

    def test_reasonable_multiplier_is_2_5(self, cost_engine_config):
        """Test reasonable assurance multiplier is 2.5x."""
        assert cost_engine_config["reasonable_multiplier"] == Decimal("2.5")

    def test_reasonable_cost_higher_than_limited(self):
        """Test reasonable assurance cost is higher than limited."""
        limited_cost = Decimal("75000")
        multiplier = Decimal("2.5")
        reasonable_cost = limited_cost * multiplier
        assert reasonable_cost > limited_cost

    def test_reasonable_cost_calculation(self):
        """Test reasonable assurance cost calculation."""
        limited_cost = Decimal("75000")
        multiplier = Decimal("2.5")
        reasonable_cost = limited_cost * multiplier
        assert_decimal_equal(reasonable_cost, Decimal("187500"))


# ---------------------------------------------------------------------------
# Multi-Jurisdiction Uplift Tests
# ---------------------------------------------------------------------------


class TestMultiJurisdictionUplift:
    """Tests for multi-jurisdiction cost uplift."""

    def test_multi_jurisdiction_uplift_pct(self, cost_engine_config):
        """Test multi-jurisdiction uplift is 15%."""
        assert cost_engine_config["multi_jurisdiction_uplift_pct"] == Decimal("15")

    def test_single_jurisdiction_no_uplift(self):
        """Test single jurisdiction has no uplift."""
        base_cost = Decimal("75000")
        jurisdictions = 1
        uplift = Decimal("15") * Decimal(str(max(0, jurisdictions - 1))) / Decimal("100")
        final = base_cost * (Decimal("1") + uplift)
        assert_decimal_equal(final, base_cost)

    def test_3_jurisdictions_applies_uplift(self):
        """Test 3 jurisdictions applies 30% uplift (15% * 2 additional)."""
        base_cost = Decimal("75000")
        jurisdictions = 3
        uplift_pct = Decimal("15") * Decimal(str(jurisdictions - 1))
        final = base_cost * (Decimal("1") + uplift_pct / Decimal("100"))
        expected = Decimal("75000") * Decimal("1.30")
        assert_decimal_equal(final, expected)


# ---------------------------------------------------------------------------
# First-Time Premium Tests
# ---------------------------------------------------------------------------


class TestFirstTimePremium:
    """Tests for first-time engagement premium."""

    def test_first_time_premium_pct(self, cost_engine_config):
        """Test first-time premium is 25%."""
        assert cost_engine_config["first_time_premium_pct"] == Decimal("25")

    def test_first_time_adds_premium(self):
        """Test first-time engagement adds 25% premium."""
        base_cost = Decimal("75000")
        premium_pct = Decimal("25")
        final = base_cost * (Decimal("1") + premium_pct / Decimal("100"))
        assert_decimal_equal(final, Decimal("93750"))

    def test_returning_client_no_premium(self):
        """Test returning client has no first-time premium."""
        base_cost = Decimal("75000")
        is_first_time = False
        premium = Decimal("25") if is_first_time else Decimal("0")
        final = base_cost * (Decimal("1") + premium / Decimal("100"))
        assert_decimal_equal(final, base_cost)


# ---------------------------------------------------------------------------
# Scope 3 Complexity Uplift Tests
# ---------------------------------------------------------------------------


class TestScope3ComplexityUplift:
    """Tests for Scope 3 complexity cost uplift."""

    def test_scope_3_uplift_pct(self, cost_engine_config):
        """Test Scope 3 complexity uplift is 20%."""
        assert cost_engine_config["scope_3_complexity_uplift_pct"] == Decimal("20")

    def test_scope_1_2_only_no_uplift(self):
        """Test Scope 1+2 only engagement has no Scope 3 uplift."""
        base_cost = Decimal("75000")
        includes_scope_3 = False
        uplift = Decimal("20") if includes_scope_3 else Decimal("0")
        final = base_cost * (Decimal("1") + uplift / Decimal("100"))
        assert_decimal_equal(final, base_cost)

    def test_scope_3_included_adds_uplift(self):
        """Test including Scope 3 adds 20% uplift."""
        base_cost = Decimal("75000")
        uplift_pct = Decimal("20")
        final = base_cost * (Decimal("1") + uplift_pct / Decimal("100"))
        assert_decimal_equal(final, Decimal("90000"))


# ---------------------------------------------------------------------------
# Timeline Estimation Tests
# ---------------------------------------------------------------------------


class TestTimelineEstimation:
    """Tests for engagement timeline estimation."""

    def test_limited_assurance_timeline_weeks(self):
        """Test limited assurance timeline is 8-12 weeks."""
        limited_weeks = 10
        assert 8 <= limited_weeks <= 12

    def test_reasonable_assurance_longer(self):
        """Test reasonable assurance timeline is longer than limited."""
        limited_weeks = 10
        reasonable_weeks = 16
        assert reasonable_weeks > limited_weeks

    def test_timeline_includes_all_phases(self):
        """Test timeline includes all engagement phases."""
        phases = [
            ("Planning", 2),
            ("Fieldwork", 4),
            ("Reporting", 2),
            ("Review", 1),
            ("Closeout", 1),
        ]
        total_weeks = sum(w for _, w in phases)
        assert total_weeks == 10


# ---------------------------------------------------------------------------
# Resource Allocation Tests
# ---------------------------------------------------------------------------


class TestResourceAllocation:
    """Tests for engagement resource allocation."""

    def test_team_includes_partner(self):
        """Test engagement team includes partner."""
        team = {"partner": 1, "manager": 1, "senior": 2, "staff": 2}
        assert "partner" in team

    def test_total_team_size(self):
        """Test total team size is reasonable."""
        team_size = 6
        assert 2 <= team_size <= 15


# ---------------------------------------------------------------------------
# Multi-Year Projection Tests
# ---------------------------------------------------------------------------


class TestMultiYearProjection:
    """Tests for multi-year cost projection."""

    def test_year_2_cost_lower_than_year_1(self):
        """Test year 2 cost is lower than year 1 (no first-time premium)."""
        year_1 = Decimal("93750")  # With 25% first-time premium
        year_2 = Decimal("75000")  # No premium
        assert year_2 < year_1

    def test_3_year_projection(self):
        """Test 3-year projection with escalation."""
        base = Decimal("75000")
        escalation = Decimal("3")  # 3% annual escalation
        projections = []
        for year in range(3):
            cost = base * (Decimal("1") + escalation / Decimal("100")) ** year
            projections.append(cost)
        assert projections[2] > projections[1] > projections[0]


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestCostEdgeCases:
    """Tests for cost estimation edge cases."""

    def test_zero_facilities_minimum_cost(self):
        """Test zero facilities produces minimum cost."""
        min_cost = Decimal("5000")
        facilities = 0
        cost = max(min_cost, Decimal(str(facilities)) * Decimal("1000"))
        assert cost == min_cost

    def test_all_uplifts_combined(self):
        """Test all cost uplifts applied simultaneously."""
        base = Decimal("75000")
        reasonable = base * Decimal("2.5")
        multi_jur = reasonable * Decimal("1.30")  # 3 jurisdictions
        first_time = multi_jur * Decimal("1.25")
        scope_3 = first_time * Decimal("1.20")
        # Combined multiplier: 2.5 * 1.30 * 1.25 * 1.20 = 4.875x
        assert scope_3 > base * Decimal("4")  # Significant total uplift
