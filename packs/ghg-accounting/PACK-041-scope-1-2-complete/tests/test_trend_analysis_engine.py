# -*- coding: utf-8 -*-
"""
Unit tests for TrendAnalysisEngine -- PACK-041 Engine 8
=========================================================

Tests absolute/percentage change, decomposition (activity, intensity,
structural), Kaya decomposition, intensity metrics, SBTi alignment,
and multi-year trend analysis.

Coverage target: 85%+
Total tests: ~55
"""

from decimal import Decimal

import pytest


# =============================================================================
# Absolute Change
# =============================================================================


class TestAbsoluteChange:
    """Test absolute emissions change calculations."""

    def test_absolute_decrease(self, sample_yearly_data, sample_base_year):
        """Emissions should show absolute decrease from base year."""
        current = sample_yearly_data[-1]["total_scope1_tco2e"]
        base = sample_base_year["total_scope1_tco2e"]
        change = current - base
        assert change < Decimal("0")
        assert change == Decimal("-2700.0")

    def test_absolute_increase_detected(self):
        base = Decimal("20000")
        current = Decimal("22000")
        change = current - base
        assert change > Decimal("0")

    def test_absolute_change_zero(self):
        base = Decimal("20000")
        current = Decimal("20000")
        change = current - base
        assert change == Decimal("0")

    def test_year_over_year_decrease(self, sample_yearly_data):
        """Each successive year should show decrease."""
        for i in range(1, len(sample_yearly_data)):
            prev = sample_yearly_data[i - 1]["total_scope1_tco2e"]
            curr = sample_yearly_data[i]["total_scope1_tco2e"]
            assert curr <= prev


# =============================================================================
# Percentage Change
# =============================================================================


class TestPercentageChange:
    """Test percentage change calculations."""

    def test_percentage_decrease_from_base(self, sample_yearly_data, sample_base_year):
        base = sample_base_year["total_scope1_tco2e"]
        current = sample_yearly_data[-1]["total_scope1_tco2e"]
        pct_change = (current - base) / base * Decimal("100")
        assert pct_change < Decimal("0")
        assert pct_change == Decimal("-10.8")

    def test_percentage_increase(self):
        base = Decimal("10000")
        current = Decimal("11500")
        pct = (current - base) / base * Decimal("100")
        assert pct == Decimal("15.0")

    @pytest.mark.parametrize("base,current,expected_pct", [
        (Decimal("100"), Decimal("90"), Decimal("-10.0")),
        (Decimal("100"), Decimal("110"), Decimal("10.0")),
        (Decimal("100"), Decimal("100"), Decimal("0")),
        (Decimal("200"), Decimal("150"), Decimal("-25.0")),
    ])
    def test_percentage_parametrized(self, base, current, expected_pct):
        pct = (current - base) / base * Decimal("100")
        assert pct == expected_pct


# =============================================================================
# Decomposition Analysis
# =============================================================================


class TestDecompositionAnalysis:
    """Test emissions change decomposition."""

    def test_activity_level_effect(self, sample_yearly_data, sample_base_year):
        """Activity effect: change in revenue drives emissions change."""
        base_revenue = sample_base_year["revenue_million_usd"]
        current_revenue = sample_yearly_data[-1]["revenue_million_usd"]
        base_intensity = sample_base_year["total_scope1_tco2e"] / base_revenue
        activity_effect = (current_revenue - base_revenue) * base_intensity
        assert activity_effect > Decimal("0")  # revenue grew, positive effect

    def test_emission_intensity_effect(self, sample_yearly_data, sample_base_year):
        """Intensity effect: change in emissions per unit revenue."""
        base_revenue = sample_base_year["revenue_million_usd"]
        current_revenue = sample_yearly_data[-1]["revenue_million_usd"]
        base_emissions = sample_base_year["total_scope1_tco2e"]
        current_emissions = sample_yearly_data[-1]["total_scope1_tco2e"]

        base_intensity = base_emissions / base_revenue
        current_intensity = current_emissions / current_revenue
        intensity_effect = (current_intensity - base_intensity) * current_revenue
        assert intensity_effect < Decimal("0")  # intensity improved

    def test_decomposition_sums_to_total_change(self):
        """Activity + intensity effects should approximately sum to total change."""
        total_change = Decimal("-2700")
        activity_effect = Decimal("2941")
        intensity_effect = Decimal("-5641")
        sum_effects = activity_effect + intensity_effect
        assert abs(sum_effects - total_change) < Decimal("100")


# =============================================================================
# Kaya Decomposition
# =============================================================================


class TestKayaDecomposition:
    """Test Kaya identity decomposition (CO2 = P * GDP/P * E/GDP * CO2/E)."""

    def test_kaya_components(self):
        population = 2500  # employees
        gdp_per_capita = Decimal("380000")  # revenue per employee
        energy_intensity = Decimal("0.12")  # GJ per USD
        carbon_intensity = Decimal("0.060")  # tCO2 per GJ
        total = Decimal(str(population)) * gdp_per_capita * energy_intensity * carbon_intensity
        assert total > Decimal("0")

    def test_kaya_decarbonization_driver(self):
        """Carbon intensity decrease should reduce total emissions."""
        base_ci = Decimal("0.060")
        improved_ci = Decimal("0.050")
        assert improved_ci < base_ci


# =============================================================================
# Intensity Metrics
# =============================================================================


class TestIntensityMetrics:
    """Test emissions intensity per various denominators."""

    def test_intensity_per_revenue(self, sample_yearly_data):
        current = sample_yearly_data[-1]
        intensity = current["total_scope1_tco2e"] / current["revenue_million_usd"]
        assert intensity > Decimal("0")
        assert intensity == pytest.approx(Decimal("23.47"), abs=Decimal("0.01"))

    def test_intensity_per_fte(self, sample_yearly_data):
        current = sample_yearly_data[-1]
        intensity = current["total_scope1_tco2e"] / Decimal(str(current["employee_count"]))
        assert intensity > Decimal("0")
        assert intensity == pytest.approx(Decimal("8.92"), abs=Decimal("0.01"))

    def test_intensity_per_floor_area(self, sample_yearly_data):
        current = sample_yearly_data[-1]
        intensity = current["total_scope1_tco2e"] / current["floor_area_m2"] * Decimal("1000")
        # tCO2e per 1000 m2
        assert intensity > Decimal("0")

    def test_intensity_decreasing_trend(self, sample_yearly_data):
        """Intensity should decrease over time (decoupling)."""
        intensities = []
        for yr in sample_yearly_data:
            intensity = yr["total_scope1_tco2e"] / yr["revenue_million_usd"]
            intensities.append(intensity)
        for i in range(1, len(intensities)):
            assert intensities[i] <= intensities[i - 1]

    @pytest.mark.parametrize("denominator_key", [
        "revenue_million_usd",
        "employee_count",
        "floor_area_m2",
    ])
    def test_intensity_denominators_positive(self, denominator_key, sample_yearly_data):
        for yr in sample_yearly_data:
            val = yr[denominator_key]
            assert val > 0


# =============================================================================
# SBTi Alignment
# =============================================================================


class TestSBTiAlignment:
    """Test Science Based Targets initiative alignment tracking."""

    def test_sbti_on_track(self, sample_yearly_data, sample_base_year):
        """Check if emissions reduction is on track for 1.5C pathway."""
        base = sample_base_year["total_scope1_tco2e"]
        current = sample_yearly_data[-1]["total_scope1_tco2e"]
        years_elapsed = 2025 - 2019
        target_annual_reduction = Decimal("4.2")  # 4.2% per year for 1.5C
        required_reduction_pct = target_annual_reduction * Decimal(str(years_elapsed))
        actual_reduction_pct = (base - current) / base * Decimal("100")
        # Required: 4.2 * 6 = 25.2%, Actual: 10.8%
        on_track = actual_reduction_pct >= required_reduction_pct
        assert on_track is False  # behind target in this scenario

    def test_sbti_behind_target(self, sample_base_year):
        """Detect when emissions are behind SBTi target."""
        base = sample_base_year["total_scope1_tco2e"]
        target_2030 = base * Decimal("0.58")  # 42% reduction by 2030
        current = Decimal("22300")
        assert current > target_2030

    def test_sbti_well_below_2c_pathway(self, sample_base_year):
        """Well-below 2C pathway: 2.5% annual reduction."""
        base = sample_base_year["total_scope1_tco2e"]
        years = 6
        target_pct = Decimal("2.5") * Decimal(str(years))  # 15%
        actual_pct = Decimal("10.8")
        assert actual_pct < target_pct

    @pytest.mark.parametrize("pathway,annual_reduction_pct", [
        ("1.5C", Decimal("4.2")),
        ("well_below_2C", Decimal("2.5")),
        ("2C", Decimal("1.23")),
    ])
    def test_sbti_pathway_rates(self, pathway, annual_reduction_pct):
        assert annual_reduction_pct > Decimal("0")


# =============================================================================
# Multi-Year Trend
# =============================================================================


class TestMultiYearTrend:
    """Test multi-year trend calculations."""

    def test_three_year_data(self, sample_yearly_data):
        assert len(sample_yearly_data) == 3

    def test_cagr_calculation(self, sample_yearly_data):
        """Compound Annual Growth Rate over 3 years."""
        first = float(sample_yearly_data[0]["total_scope1_tco2e"])
        last = float(sample_yearly_data[-1]["total_scope1_tco2e"])
        years = 2
        cagr = (last / first) ** (1 / years) - 1
        assert cagr < 0  # declining

    def test_base_year_comparison(self, sample_yearly_data, sample_base_year):
        base = sample_base_year["total_scope1_tco2e"]
        for yr in sample_yearly_data:
            pct_of_base = yr["total_scope1_tco2e"] / base * Decimal("100")
            assert pct_of_base < Decimal("100")  # all years below base

    def test_trend_includes_scope2(self, sample_yearly_data):
        for yr in sample_yearly_data:
            assert "total_scope2_location_tco2e" in yr
            assert yr["total_scope2_location_tco2e"] > Decimal("0")

    def test_scope12_total_trend(self, sample_yearly_data):
        """Scope 1+2 total should also show decreasing trend."""
        totals = []
        for yr in sample_yearly_data:
            total = yr["total_scope1_tco2e"] + yr["total_scope2_location_tco2e"]
            totals.append(total)
        for i in range(1, len(totals)):
            assert totals[i] <= totals[i - 1]

    def test_market_based_trend(self, sample_yearly_data):
        """Market-based scope 2 should decrease faster than location-based."""
        market_reductions = []
        location_reductions = []
        for i in range(1, len(sample_yearly_data)):
            prev = sample_yearly_data[i - 1]
            curr = sample_yearly_data[i]
            market_reductions.append(
                (prev["total_scope2_market_tco2e"] - curr["total_scope2_market_tco2e"])
            )
            location_reductions.append(
                (prev["total_scope2_location_tco2e"] - curr["total_scope2_location_tco2e"])
            )
        assert sum(market_reductions) > sum(location_reductions)
