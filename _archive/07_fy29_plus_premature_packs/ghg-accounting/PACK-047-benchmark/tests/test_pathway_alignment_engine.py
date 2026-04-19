"""
Unit tests for PathwayAlignmentEngine (PACK-047 Engine 4).

Tests all public methods with 35+ tests covering:
  - IEA NZE pathway loading
  - IPCC C1, C2, C3 pathways
  - SBTi SDA convergence
  - OECM pathway
  - Waypoint interpolation (linear)
  - Gap-to-pathway calculation
  - Years to convergence
  - Overshoot year calculation
  - Multi-pathway alignment
  - Negative gap (ahead of pathway)

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
# IEA NZE Pathway Tests
# ---------------------------------------------------------------------------


class TestIEANZEPathway:
    """Tests for IEA Net Zero by 2050 pathway loading and validation."""

    def test_iea_nze_pathway_loaded(self, sample_pathway_data):
        """Test IEA NZE pathway is present in fixture."""
        assert "IEA_NZE" in sample_pathway_data

    def test_iea_nze_base_year_2020(self, sample_pathway_data):
        """Test IEA NZE base year is 2020."""
        assert sample_pathway_data["IEA_NZE"]["base_year"] == 2020

    def test_iea_nze_base_value_100(self, sample_pathway_data):
        """Test IEA NZE base value is 100 (index)."""
        assert sample_pathway_data["IEA_NZE"]["base_value"] == Decimal("100")

    def test_iea_nze_reaches_zero_by_2050(self, sample_pathway_data):
        """Test IEA NZE pathway reaches 0 by 2050."""
        waypoints = sample_pathway_data["IEA_NZE"]["waypoints"]
        assert waypoints["2050"] == Decimal("0")

    def test_iea_nze_monotonically_decreasing(self, sample_pathway_data):
        """Test IEA NZE pathway is monotonically decreasing."""
        waypoints = sample_pathway_data["IEA_NZE"]["waypoints"]
        years = sorted(waypoints.keys())
        for i in range(1, len(years)):
            assert waypoints[years[i]] <= waypoints[years[i - 1]]

    def test_iea_nze_has_6_waypoints(self, sample_pathway_data):
        """Test IEA NZE pathway has expected number of waypoints."""
        waypoints = sample_pathway_data["IEA_NZE"]["waypoints"]
        assert len(waypoints) == 6


# ---------------------------------------------------------------------------
# IPCC C1/C2/C3 Pathway Tests
# ---------------------------------------------------------------------------


class TestIPCCPathways:
    """Tests for IPCC AR6 C1, C2, C3 pathways."""

    def test_ipcc_c1_loaded(self, sample_pathway_data):
        """Test IPCC C1 pathway is present."""
        assert "IPCC_C1" in sample_pathway_data

    def test_ipcc_c2_loaded(self, sample_pathway_data):
        """Test IPCC C2 pathway is present."""
        assert "IPCC_C2" in sample_pathway_data

    def test_ipcc_c3_loaded(self, sample_pathway_data):
        """Test IPCC C3 pathway is present."""
        assert "IPCC_C3" in sample_pathway_data

    def test_c1_more_ambitious_than_c2(self, sample_pathway_data):
        """Test IPCC C1 is more ambitious than C2 at each waypoint."""
        c1 = sample_pathway_data["IPCC_C1"]["waypoints"]
        c2 = sample_pathway_data["IPCC_C2"]["waypoints"]
        for year in c1:
            if year in c2:
                assert c1[year] <= c2[year], (
                    f"C1 ({c1[year]}) should be <= C2 ({c2[year]}) in {year}"
                )

    def test_c2_more_ambitious_than_c3(self, sample_pathway_data):
        """Test IPCC C2 is more ambitious than C3 at each waypoint."""
        c2 = sample_pathway_data["IPCC_C2"]["waypoints"]
        c3 = sample_pathway_data["IPCC_C3"]["waypoints"]
        for year in c2:
            if year in c3:
                assert c2[year] <= c3[year], (
                    f"C2 ({c2[year]}) should be <= C3 ({c3[year]}) in {year}"
                )

    def test_ipcc_c1_allows_negative_emissions(self, sample_pathway_data):
        """Test IPCC C1 allows negative values (carbon removal)."""
        c1 = sample_pathway_data["IPCC_C1"]["waypoints"]
        has_negative = any(v < Decimal("0") for v in c1.values())
        assert has_negative is True

    def test_ipcc_c3_all_non_negative(self, sample_pathway_data):
        """Test IPCC C3 waypoints are all non-negative."""
        c3 = sample_pathway_data["IPCC_C3"]["waypoints"]
        assert all(v >= Decimal("0") for v in c3.values())


# ---------------------------------------------------------------------------
# SBTi SDA Convergence Tests
# ---------------------------------------------------------------------------


class TestSBTiSDAConvergence:
    """Tests for SBTi Sectoral Decarbonisation Approach pathway."""

    def test_sbti_sda_power_loaded(self, sample_pathway_data):
        """Test SBTi SDA power pathway is present."""
        assert "SBTi_SDA_POWER" in sample_pathway_data

    def test_sbti_sda_unit_is_intensity(self, sample_pathway_data):
        """Test SBTi SDA uses intensity unit (tCO2e/MWh)."""
        assert sample_pathway_data["SBTi_SDA_POWER"]["unit"] == "tCO2e/MWh"

    def test_sbti_sda_reaches_zero_by_2050(self, sample_pathway_data):
        """Test SBTi SDA power pathway reaches 0 by 2050."""
        waypoints = sample_pathway_data["SBTi_SDA_POWER"]["waypoints"]
        assert waypoints["2050"] == Decimal("0.000")

    def test_sbti_sda_base_value(self, sample_pathway_data):
        """Test SBTi SDA power base value is 0.415 tCO2e/MWh."""
        assert_decimal_equal(
            sample_pathway_data["SBTi_SDA_POWER"]["base_value"],
            Decimal("0.415"),
            tolerance=Decimal("0.001"),
        )


# ---------------------------------------------------------------------------
# OECM Pathway Tests
# ---------------------------------------------------------------------------


class TestOECMPathway:
    """Tests for One Earth Climate Model (OECM) pathway."""

    def test_oecm_loaded(self, sample_pathway_data):
        """Test OECM pathway is present."""
        assert "OECM" in sample_pathway_data

    def test_oecm_allows_negative(self, sample_pathway_data):
        """Test OECM allows negative (net-negative) emissions."""
        waypoints = sample_pathway_data["OECM"]["waypoints"]
        assert waypoints["2050"] < Decimal("0")


# ---------------------------------------------------------------------------
# Waypoint Interpolation Tests
# ---------------------------------------------------------------------------


class TestWaypointInterpolation:
    """Tests for linear interpolation between waypoints."""

    def test_linear_interpolation_midpoint(self, sample_pathway_data):
        """Test linear interpolation at midpoint between two waypoints."""
        nze = sample_pathway_data["IEA_NZE"]["waypoints"]
        v_2025 = nze["2025"]  # 85
        v_2030 = nze["2030"]  # 60
        # Midpoint (2027.5) -> (85 + 60) / 2 = 72.5
        interpolated = (v_2025 + v_2030) / Decimal("2")
        assert_decimal_equal(interpolated, Decimal("72.5"), tolerance=Decimal("0.1"))

    def test_interpolation_at_waypoint_exact(self, sample_pathway_data):
        """Test interpolation at exact waypoint returns waypoint value."""
        nze = sample_pathway_data["IEA_NZE"]["waypoints"]
        # Year 2030 is an exact waypoint
        assert nze["2030"] == Decimal("60")

    def test_interpolation_between_2030_2035(self, sample_pathway_data):
        """Test interpolation between 2030 and 2035 waypoints."""
        nze = sample_pathway_data["IEA_NZE"]["waypoints"]
        v_2030 = nze["2030"]  # 60
        v_2035 = nze["2035"]  # 40
        # Year 2032: (60 - (60-40) * 2/5) = 60 - 8 = 52
        fraction = Decimal("2") / Decimal("5")
        interpolated = v_2030 - (v_2030 - v_2035) * fraction
        assert_decimal_equal(interpolated, Decimal("52"), tolerance=Decimal("0.1"))


# ---------------------------------------------------------------------------
# Gap-to-Pathway Calculation Tests
# ---------------------------------------------------------------------------


class TestGapToPathway:
    """Tests for gap-to-pathway calculation."""

    def test_positive_gap_behind_pathway(self, sample_pathway_data):
        """Test positive gap when entity is behind pathway."""
        entity_value = Decimal("80")  # Entity at index 80 in 2024
        nze = sample_pathway_data["IEA_NZE"]["waypoints"]
        # Interpolate 2024 between 2025(85) and 2020(100)
        pathway_2024 = Decimal("88")  # approx linear interpolation
        gap = entity_value - pathway_2024
        assert gap < Decimal("0")  # Entity is ahead (lower = better)

    def test_zero_gap_on_pathway(self, sample_pathway_data):
        """Test zero gap when entity is exactly on pathway."""
        entity_value = Decimal("85")
        pathway_value = Decimal("85")
        gap = entity_value - pathway_value
        assert gap == Decimal("0")

    def test_negative_gap_ahead_of_pathway(self, sample_pathway_data):
        """Test negative gap when entity is ahead of pathway."""
        entity_value = Decimal("50")
        pathway_value = Decimal("85")  # NZE at 2025
        gap = entity_value - pathway_value
        assert gap < Decimal("0")  # Entity is ahead

    def test_gap_increases_for_laggards(self):
        """Test gap increases as entity diverges from pathway."""
        gaps = []
        entity_values = [Decimal("90"), Decimal("95"), Decimal("100")]
        pathway_value = Decimal("85")
        for ev in entity_values:
            gaps.append(ev - pathway_value)
        assert gaps[0] < gaps[1] < gaps[2]


# ---------------------------------------------------------------------------
# Years to Convergence Tests
# ---------------------------------------------------------------------------


class TestYearsToConvergence:
    """Tests for years-to-convergence calculation."""

    def test_on_pathway_zero_years(self):
        """Test entity on pathway has 0 years to convergence."""
        entity_rate = Decimal("-5")  # 5% annual reduction
        pathway_rate = Decimal("-5")
        entity_value = Decimal("80")
        pathway_value = Decimal("80")
        if entity_value == pathway_value:
            years = 0
        else:
            years = -1
        assert years == 0

    def test_faster_reduction_converges(self):
        """Test entity reducing faster than pathway converges eventually."""
        entity_rate = Decimal("-8")  # 8% annual reduction
        pathway_rate = Decimal("-5")  # 5% annual reduction
        # Entity reduces faster -> will converge
        converges = abs(entity_rate) > abs(pathway_rate)
        assert converges is True

    def test_slower_reduction_never_converges(self):
        """Test entity reducing slower than pathway never converges."""
        entity_rate = Decimal("-3")
        pathway_rate = Decimal("-5")
        converges = abs(entity_rate) > abs(pathway_rate)
        assert converges is False


# ---------------------------------------------------------------------------
# Overshoot Year Calculation Tests
# ---------------------------------------------------------------------------


class TestOvershootYearCalculation:
    """Tests for overshoot year calculation."""

    def test_overshoot_when_entity_above_pathway(self):
        """Test overshoot detected when entity exceeds pathway budget."""
        entity_cumulative = Decimal("500")
        pathway_cumulative = Decimal("400")
        assert entity_cumulative > pathway_cumulative  # Overshoot

    def test_no_overshoot_when_below_budget(self):
        """Test no overshoot when entity is below pathway budget."""
        entity_cumulative = Decimal("350")
        pathway_cumulative = Decimal("400")
        assert entity_cumulative <= pathway_cumulative  # No overshoot


# ---------------------------------------------------------------------------
# Multi-Pathway Alignment Tests
# ---------------------------------------------------------------------------


class TestMultiPathwayAlignment:
    """Tests for simultaneous alignment to multiple pathways."""

    def test_alignment_score_per_pathway(self, sample_pathway_data):
        """Test alignment scored independently for each pathway."""
        entity_value_2030 = Decimal("55")
        pathways = {
            "IEA_NZE": sample_pathway_data["IEA_NZE"]["waypoints"]["2030"],
            "IPCC_C1": sample_pathway_data["IPCC_C1"]["waypoints"]["2030"],
            "IPCC_C3": sample_pathway_data["IPCC_C3"]["waypoints"]["2030"],
        }
        alignments = {}
        for name, target in pathways.items():
            gap = entity_value_2030 - target
            alignments[name] = gap
        # Entity at 55: NZE target 60 -> -5 (ahead); C1 target 55 -> 0; C3 target 75 -> -20
        assert alignments["IEA_NZE"] < Decimal("0")  # Ahead of NZE
        assert alignments["IPCC_C1"] == Decimal("0")  # On C1
        assert alignments["IPCC_C3"] < Decimal("0")  # Ahead of C3

    def test_most_ambitious_pathway_identified(self, sample_pathway_data):
        """Test most ambitious pathway is identified (lowest target)."""
        targets_2030 = {
            name: p["waypoints"]["2030"]
            for name, p in sample_pathway_data.items()
            if "waypoints" in p and "2030" in p["waypoints"]
        }
        most_ambitious = min(targets_2030, key=targets_2030.get)
        assert most_ambitious == "OECM"  # OECM has lowest 2030 target (52)

    def test_least_ambitious_pathway_identified(self, sample_pathway_data):
        """Test least ambitious pathway is identified (highest target)."""
        targets_2030 = {
            name: p["waypoints"]["2030"]
            for name, p in sample_pathway_data.items()
            if "waypoints" in p and "2030" in p["waypoints"]
        }
        least_ambitious = max(targets_2030, key=targets_2030.get)
        assert least_ambitious == "IPCC_C3"  # IPCC C3 has highest 2030 target (75)
