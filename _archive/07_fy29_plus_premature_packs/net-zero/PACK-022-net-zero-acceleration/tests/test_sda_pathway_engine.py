# -*- coding: utf-8 -*-
"""
Tests for SDAPathwayEngine - PACK-022 Engine 2

SBTi Sectoral Decarbonization Approach (SDA) for 12 sectors with
intensity convergence pathways, activity growth modeling, and
IEA Net Zero benchmark alignment checking.

Coverage targets: 85%+ of SDAPathwayEngine methods.
"""

import pytest
from decimal import Decimal

from engines.sda_pathway_engine import (
    SDAPathwayEngine,
    SDAInput,
    SDAResult,
    IntensityPoint,
    AbsolutePoint,
    ACAComparisonPoint,
    IEAAlignmentCheck,
    SDASector,
    IntensityUnit,
    ActivityMetric,
    PathwayStatus,
    SECTOR_BENCHMARKS,
    ACA_ANNUAL_RATE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create an SDAPathwayEngine instance."""
    return SDAPathwayEngine()


@pytest.fixture
def power_input():
    """Power generation sector input."""
    return SDAInput(
        entity_name="PowerGenCo",
        sector=SDASector.POWER_GENERATION,
        base_year=2020,
        target_year=2050,
        base_year_intensity=Decimal("0.450"),
        base_year_activity=Decimal("1000000"),  # 1M MWh
        base_year_emissions_tco2e=Decimal("450000"),
        activity_growth_rate_pct=Decimal("2.0"),
        projection_interval_years=5,
        near_term_target_year=2030,
        include_aca_comparison=True,
        include_iea_alignment=True,
    )


@pytest.fixture
def steel_input():
    """Steel sector input."""
    return SDAInput(
        entity_name="SteelCo",
        sector=SDASector.STEEL,
        base_year=2020,
        target_year=2050,
        base_year_intensity=Decimal("2.10"),
        base_year_activity=Decimal("500000"),  # 500k tonnes
        base_year_emissions_tco2e=Decimal("1050000"),
        activity_growth_rate_pct=Decimal("1.5"),
    )


@pytest.fixture
def cement_input():
    """Cement sector input."""
    return SDAInput(
        entity_name="CementCo",
        sector=SDASector.CEMENT,
        base_year=2020,
        target_year=2050,
        base_year_intensity=Decimal("0.700"),
        base_year_activity=Decimal("2000000"),  # 2M tonnes
        base_year_emissions_tco2e=Decimal("1400000"),
        activity_growth_rate_pct=Decimal("1.0"),
    )


@pytest.fixture
def buildings_input():
    """Commercial buildings sector input."""
    return SDAInput(
        entity_name="BuildCo",
        sector=SDASector.BUILDINGS_COMMERCIAL,
        base_year=2020,
        target_year=2050,
        base_year_intensity=Decimal("45.0"),
        base_year_activity=Decimal("500000"),  # 500k m2
        base_year_emissions_tco2e=Decimal("22500"),
        activity_growth_rate_pct=Decimal("2.5"),
    )


# ---------------------------------------------------------------------------
# TestSDABasic
# ---------------------------------------------------------------------------


class TestSDABasic:
    """Basic functionality tests for SDAPathwayEngine."""

    def test_engine_instantiation(self):
        """Engine can be instantiated."""
        engine = SDAPathwayEngine()
        assert engine.engine_version == "1.0.0"

    def test_calculate_returns_result(self, engine, power_input):
        """calculate() returns an SDAResult."""
        result = engine.calculate(power_input)
        assert isinstance(result, SDAResult)

    def test_result_entity_name(self, engine, power_input):
        """Result has correct entity name."""
        result = engine.calculate(power_input)
        assert result.entity_name == "PowerGenCo"

    def test_result_sector(self, engine, power_input):
        """Result has correct sector."""
        result = engine.calculate(power_input)
        assert result.sector == SDASector.POWER_GENERATION.value

    def test_result_sector_name(self, engine, power_input):
        """Result has human-readable sector name."""
        result = engine.calculate(power_input)
        assert result.sector_name == "Power Generation"

    def test_result_intensity_unit(self, engine, power_input):
        """Result has correct intensity unit."""
        result = engine.calculate(power_input)
        assert result.intensity_unit == IntensityUnit.TCO2E_PER_MWH.value

    def test_result_provenance_hash(self, engine, power_input):
        """Result has 64-char hex provenance hash."""
        result = engine.calculate(power_input)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_result_processing_time(self, engine, power_input):
        """Result has positive processing time."""
        result = engine.calculate(power_input)
        assert result.processing_time_ms > 0.0


# ---------------------------------------------------------------------------
# TestIntensityTrajectory
# ---------------------------------------------------------------------------


class TestIntensityTrajectory:
    """Tests for the intensity convergence trajectory."""

    def test_trajectory_has_entries(self, engine, power_input):
        """Intensity trajectory is populated."""
        result = engine.calculate(power_input)
        assert len(result.intensity_trajectory) > 0

    def test_trajectory_starts_at_base(self, engine, power_input):
        """First trajectory point is the base year."""
        result = engine.calculate(power_input)
        first = result.intensity_trajectory[0]
        assert first.year == 2020

    def test_trajectory_ends_at_target(self, engine, power_input):
        """Last trajectory point is the target year."""
        result = engine.calculate(power_input)
        last = result.intensity_trajectory[-1]
        assert last.year == 2050

    def test_base_intensity_matches_input(self, engine, power_input):
        """Base year company intensity matches the input."""
        result = engine.calculate(power_input)
        first = result.intensity_trajectory[0]
        assert float(first.company_intensity) == pytest.approx(0.45, rel=1e-3)

    def test_target_intensity_converges_to_benchmark(self, engine, power_input):
        """Target year intensity converges to sector benchmark."""
        result = engine.calculate(power_input)
        last = result.intensity_trajectory[-1]
        expected_target = float(
            SECTOR_BENCHMARKS[SDASector.POWER_GENERATION]["benchmarks"][2050]
        )
        assert float(last.company_intensity) == pytest.approx(expected_target, rel=1e-3)

    def test_intensity_decreases_over_time(self, engine, power_input):
        """Company intensity decreases from base to target."""
        result = engine.calculate(power_input)
        first_int = float(result.intensity_trajectory[0].company_intensity)
        last_int = float(result.intensity_trajectory[-1].company_intensity)
        assert last_int < first_int

    def test_convergence_pct_reaches_100(self, engine, power_input):
        """Convergence percentage reaches 100% at target year."""
        result = engine.calculate(power_input)
        last = result.intensity_trajectory[-1]
        assert float(last.convergence_pct) == pytest.approx(100.0, rel=1e-2)

    def test_convergence_pct_starts_at_zero(self, engine, power_input):
        """Convergence percentage starts at 0% at base year."""
        result = engine.calculate(power_input)
        first = result.intensity_trajectory[0]
        assert float(first.convergence_pct) == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# TestAbsoluteTrajectory
# ---------------------------------------------------------------------------


class TestAbsoluteTrajectory:
    """Tests for absolute emission projections."""

    def test_absolute_trajectory_populated(self, engine, power_input):
        """Absolute trajectory has entries."""
        result = engine.calculate(power_input)
        assert len(result.absolute_trajectory) > 0

    def test_absolute_trajectory_years_match_intensity(self, engine, power_input):
        """Absolute trajectory years match intensity trajectory years."""
        result = engine.calculate(power_input)
        int_years = [pt.year for pt in result.intensity_trajectory]
        abs_years = [pt.year for pt in result.absolute_trajectory]
        assert int_years == abs_years

    def test_activity_level_grows(self, engine, power_input):
        """Activity level grows over time with positive growth rate."""
        result = engine.calculate(power_input)
        first_activity = float(result.absolute_trajectory[0].activity_level)
        last_activity = float(result.absolute_trajectory[-1].activity_level)
        assert last_activity > first_activity

    def test_reduction_from_base_increases(self, engine, power_input):
        """Reduction from base increases toward end of trajectory."""
        result = engine.calculate(power_input)
        last = result.absolute_trajectory[-1]
        assert float(last.reduction_from_base_pct) > 0


# ---------------------------------------------------------------------------
# TestACAComparison
# ---------------------------------------------------------------------------


class TestACAComparison:
    """Tests for ACA vs SDA pathway comparison."""

    def test_aca_comparison_populated(self, engine, power_input):
        """ACA comparison is populated when requested."""
        result = engine.calculate(power_input)
        assert len(result.aca_comparison) > 0

    def test_aca_comparison_excluded_when_disabled(self, engine):
        """ACA comparison is empty when disabled."""
        inp = SDAInput(
            entity_name="NoACA",
            sector=SDASector.CEMENT,
            base_year=2020,
            target_year=2050,
            base_year_intensity=Decimal("0.65"),
            base_year_activity=Decimal("100000"),
            base_year_emissions_tco2e=Decimal("65000"),
            include_aca_comparison=False,
        )
        result = engine.calculate(inp)
        assert len(result.aca_comparison) == 0

    def test_aca_base_year_delta_zero(self, engine, power_input):
        """At base year, ACA emissions equal base emissions (delta ~0)."""
        result = engine.calculate(power_input)
        base_comp = [c for c in result.aca_comparison if c.year == 2020]
        if base_comp:
            assert float(base_comp[0].aca_emissions_tco2e) == pytest.approx(
                450000.0, rel=1e-3
            )

    def test_aca_emissions_decrease_linearly(self, engine, power_input):
        """ACA emissions decrease over time."""
        result = engine.calculate(power_input)
        if len(result.aca_comparison) >= 2:
            first = float(result.aca_comparison[0].aca_emissions_tco2e)
            last = float(result.aca_comparison[-1].aca_emissions_tco2e)
            assert last < first


# ---------------------------------------------------------------------------
# TestIEAAlignment
# ---------------------------------------------------------------------------


class TestIEAAlignment:
    """Tests for IEA NZE alignment checking."""

    def test_iea_checks_populated(self, engine, power_input):
        """IEA alignment checks are populated when requested."""
        result = engine.calculate(power_input)
        assert len(result.iea_alignment_checks) > 0

    def test_iea_checks_include_2030_2050(self, engine, power_input):
        """IEA checks cover at least 2030 and 2050."""
        result = engine.calculate(power_input)
        years = {c.year for c in result.iea_alignment_checks}
        assert 2030 in years
        assert 2050 in years

    def test_iea_alignment_at_target_year(self, engine, power_input):
        """At target year, company should be aligned (converged to benchmark)."""
        result = engine.calculate(power_input)
        check_2050 = [c for c in result.iea_alignment_checks if c.year == 2050]
        if check_2050:
            assert check_2050[0].aligned is True

    def test_iea_excluded_when_disabled(self, engine):
        """IEA checks are empty when disabled."""
        inp = SDAInput(
            entity_name="NoIEA",
            sector=SDASector.STEEL,
            base_year=2020,
            target_year=2050,
            base_year_intensity=Decimal("2.0"),
            base_year_activity=Decimal("100000"),
            base_year_emissions_tco2e=Decimal("200000"),
            include_iea_alignment=False,
        )
        result = engine.calculate(inp)
        assert len(result.iea_alignment_checks) == 0


# ---------------------------------------------------------------------------
# TestCumulativeAbatement
# ---------------------------------------------------------------------------


class TestCumulativeAbatement:
    """Tests for cumulative abatement calculation."""

    def test_cumulative_abatement_positive(self, engine, power_input):
        """Cumulative abatement is positive when intensity converges."""
        result = engine.calculate(power_input)
        assert float(result.total_cumulative_abatement_tco2e) > 0

    def test_cumulative_abatement_non_negative(self, engine, steel_input):
        """Cumulative abatement is never negative."""
        result = engine.calculate(steel_input)
        assert float(result.total_cumulative_abatement_tco2e) >= 0


# ---------------------------------------------------------------------------
# TestNearTermMetrics
# ---------------------------------------------------------------------------


class TestNearTermMetrics:
    """Tests for near-term intensity and reduction metrics."""

    def test_near_term_intensity_between_base_and_target(self, engine, power_input):
        """Near-term (2030) intensity is between base and target."""
        result = engine.calculate(power_input)
        base = float(result.base_year_intensity)
        target = float(result.target_year_intensity)
        near = float(result.near_term_intensity)
        assert target <= near <= base

    def test_near_term_reduction_positive(self, engine, power_input):
        """Near-term reduction percentage is positive."""
        result = engine.calculate(power_input)
        assert float(result.near_term_reduction_pct) > 0


# ---------------------------------------------------------------------------
# TestMultipleSectors
# ---------------------------------------------------------------------------


class TestMultipleSectors:
    """Tests across multiple SDA sectors."""

    @pytest.mark.parametrize("sector", [
        SDASector.POWER_GENERATION,
        SDASector.CEMENT,
        SDASector.STEEL,
        SDASector.ALUMINIUM,
        SDASector.PULP_PAPER,
        SDASector.TRANSPORT_ROAD,
        SDASector.BUILDINGS_COMMERCIAL,
        SDASector.BUILDINGS_RESIDENTIAL,
        SDASector.CHEMICALS,
        SDASector.AVIATION,
        SDASector.SHIPPING,
        SDASector.FOOD_BEVERAGE,
    ])
    def test_all_sectors_calculate(self, engine, sector):
        """Every supported sector can calculate successfully."""
        inp = SDAInput(
            entity_name=f"Test_{sector.value}",
            sector=sector,
            base_year=2020,
            target_year=2050,
            base_year_intensity=Decimal("1.0"),
            base_year_activity=Decimal("100000"),
            base_year_emissions_tco2e=Decimal("100000"),
            include_aca_comparison=True,
            include_iea_alignment=True,
        )
        result = engine.calculate(inp)
        assert isinstance(result, SDAResult)
        assert result.sector == sector.value
        assert len(result.provenance_hash) == 64

    def test_steel_target_intensity(self, engine, steel_input):
        """Steel converges to steel benchmark at 2050."""
        result = engine.calculate(steel_input)
        expected = float(SECTOR_BENCHMARKS[SDASector.STEEL]["benchmarks"][2050])
        assert float(result.target_year_intensity) == pytest.approx(expected, rel=1e-3)

    def test_cement_target_intensity(self, engine, cement_input):
        """Cement converges to cement benchmark at 2050."""
        result = engine.calculate(cement_input)
        expected = float(SECTOR_BENCHMARKS[SDASector.CEMENT]["benchmarks"][2050])
        assert float(result.target_year_intensity) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# TestUtilityMethods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    """Tests for engine utility methods."""

    def test_get_sector_info(self, engine):
        """get_sector_info returns sector data."""
        info = engine.get_sector_info(SDASector.POWER_GENERATION)
        assert info["name"] == "Power Generation"
        assert "benchmarks" in info

    def test_get_sector_info_unknown_raises(self, engine):
        """get_sector_info raises ValueError for invalid sector."""
        with pytest.raises((ValueError, KeyError)):
            engine.get_sector_info("nonexistent_sector")

    def test_get_supported_sectors(self, engine):
        """get_supported_sectors returns all 12 sectors."""
        sectors = engine.get_supported_sectors()
        assert len(sectors) == 12
        names = {s["sector"] for s in sectors}
        assert SDASector.POWER_GENERATION.value in names

    def test_get_summary(self, engine, power_input):
        """get_summary returns dict with expected keys."""
        result = engine.calculate(power_input)
        summary = engine.get_summary(result)
        assert summary["entity_name"] == "PowerGenCo"
        assert "convergence_rate_pct_yr" in summary
        assert "provenance_hash" in summary


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_high_activity_growth(self, engine):
        """High growth rate warns about absolute emission increase."""
        inp = SDAInput(
            entity_name="HighGrowth",
            sector=SDASector.CEMENT,
            base_year=2020,
            target_year=2050,
            base_year_intensity=Decimal("0.65"),
            base_year_activity=Decimal("100000"),
            base_year_emissions_tco2e=Decimal("65000"),
            activity_growth_rate_pct=Decimal("5.0"),
        )
        result = engine.calculate(inp)
        assert any("activity growth" in r.lower() for r in result.recommendations)

    def test_zero_activity_growth(self, engine):
        """Zero growth rate keeps activity constant."""
        inp = SDAInput(
            entity_name="Flat",
            sector=SDASector.STEEL,
            base_year=2020,
            target_year=2050,
            base_year_intensity=Decimal("2.0"),
            base_year_activity=Decimal("100000"),
            base_year_emissions_tco2e=Decimal("200000"),
            activity_growth_rate_pct=Decimal("0.0"),
        )
        result = engine.calculate(inp)
        first_act = float(result.absolute_trajectory[0].activity_level)
        last_act = float(result.absolute_trajectory[-1].activity_level)
        assert first_act == pytest.approx(last_act, rel=1e-3)

    def test_interval_1_year(self, engine):
        """Interval of 1 year produces annual trajectory."""
        inp = SDAInput(
            entity_name="Annual",
            sector=SDASector.CEMENT,
            base_year=2020,
            target_year=2030,
            base_year_intensity=Decimal("0.65"),
            base_year_activity=Decimal("100000"),
            base_year_emissions_tco2e=Decimal("65000"),
            projection_interval_years=1,
        )
        result = engine.calculate(inp)
        years = [pt.year for pt in result.intensity_trajectory]
        for y in range(2020, 2031):
            assert y in years


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for Pydantic input validation."""

    def test_target_must_be_after_base(self):
        """target_year <= base_year raises error."""
        with pytest.raises(Exception):
            SDAInput(
                entity_name="Bad",
                sector=SDASector.CEMENT,
                base_year=2025,
                target_year=2025,
                base_year_intensity=Decimal("0.5"),
                base_year_activity=Decimal("100000"),
                base_year_emissions_tco2e=Decimal("50000"),
            )

    def test_intensity_must_be_positive(self):
        """Zero intensity is rejected."""
        with pytest.raises(Exception):
            SDAInput(
                entity_name="Bad",
                sector=SDASector.CEMENT,
                base_year=2020,
                target_year=2050,
                base_year_intensity=Decimal("0"),
                base_year_activity=Decimal("100000"),
                base_year_emissions_tco2e=Decimal("50000"),
            )


# ---------------------------------------------------------------------------
# TestEnums
# ---------------------------------------------------------------------------


class TestEnums:
    """Tests for enum definitions."""

    def test_sda_sector_count(self):
        """SDASector has 12 members."""
        assert len(SDASector) == 12

    def test_pathway_status_values(self):
        """PathwayStatus has expected values."""
        assert PathwayStatus.ALIGNED.value == "aligned"
        assert PathwayStatus.ABOVE_PATHWAY.value == "above_pathway"
        assert PathwayStatus.BELOW_PATHWAY.value == "below_pathway"
        assert PathwayStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_intensity_unit_values(self):
        """IntensityUnit has expected values."""
        assert IntensityUnit.TCO2E_PER_MWH.value == "tCO2e/MWh"

    def test_sector_benchmarks_complete(self):
        """Every SDASector has an entry in SECTOR_BENCHMARKS."""
        for sector in SDASector:
            assert sector in SECTOR_BENCHMARKS
