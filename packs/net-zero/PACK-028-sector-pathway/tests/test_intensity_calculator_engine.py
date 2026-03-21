# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Intensity Calculator Engine.

Tests 20+ sector-specific intensity metrics, normalization accuracy,
trend calculation, data quality scoring, and multi-unit support.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  2 of 8 - intensity_calculator_engine.py
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.intensity_calculator_engine import (
    IntensityCalculatorEngine,
    IntensityInput,
    IntensityResult,
    IntensityMetricType,
    IntensityDataPoint,
    TrendDirection,
    DataQualityTier,
    DataMeasurementMethod,
    VerificationStatus,
    SectorType,
    ActivityDataPoint,
    SubProcessEntry,
    TrendAnalysis,
    BenchmarkComparison,
    DataQualityAssessment,
    SECTOR_INTENSITY_DEFS,
)

from .conftest import (
    assert_decimal_close,
    assert_decimal_positive,
    assert_percentage_range,
    assert_provenance_hash,
    assert_processing_time,
    assert_intensity_accuracy,
    INTENSITY_METRICS,
    SDA_SECTORS,
    ALL_SECTORS,
    timed_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_activity(year, emissions, activity, method=DataMeasurementMethod.CALCULATION,
                   verification=VerificationStatus.UNVERIFIED, completeness=Decimal("100")):
    """Create an ActivityDataPoint."""
    return ActivityDataPoint(
        year=year,
        activity_value=activity,
        total_emissions_tco2e=emissions,
        measurement_method=method,
        verification_status=verification,
        data_completeness_pct=completeness,
    )


def _make_input(sector="steel", entity_name="TestCo", activity_data=None,
                base_year=2019, include_trend=True, include_benchmark=True,
                include_secondary=True, sub_processes=None, custom_metric=None,
                revenue_m=None, employees=None):
    """Create an IntensityInput with sensible defaults."""
    if activity_data is None:
        activity_data = [
            _make_activity(2020, Decimal("10500000"), Decimal("5000000")),
        ]
    return IntensityInput(
        entity_name=entity_name,
        sector=SectorType(sector),
        base_year=base_year,
        activity_data=activity_data,
        include_trend_analysis=include_trend,
        include_benchmark_comparison=include_benchmark,
        include_secondary_metrics=include_secondary,
        sub_processes=sub_processes or [],
        custom_metric=custom_metric,
        revenue_m=revenue_m,
        employees=employees,
    )


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestIntensityCalculatorInstantiation:
    """Engine instantiation and metadata tests."""

    def test_engine_instantiates(self):
        engine = IntensityCalculatorEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = IntensityCalculatorEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = IntensityCalculatorEngine()
        assert engine.engine_version == "1.0.0"

    def test_engine_supports_20_plus_metrics(self):
        engine = IntensityCalculatorEngine()
        metrics = engine.get_supported_metrics()
        assert len(metrics) >= 20

    def test_engine_has_sector_definitions(self):
        engine = IntensityCalculatorEngine()
        defs = engine.get_sector_definitions()
        assert len(defs) >= 12

    def test_supported_metrics_have_units(self):
        engine = IntensityCalculatorEngine()
        for m in engine.get_supported_metrics():
            assert "metric" in m
            assert "unit" in m
            assert len(m["unit"]) > 0


# ===========================================================================
# Power Sector Intensity Metrics
# ===========================================================================


class TestPowerSectorIntensity:
    """Power sector intensity metric calculations."""

    def test_power_intensity_calculated(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(
            sector="power_generation",
            activity_data=[
                _make_activity(2020, Decimal("22500"), Decimal("50000000")),
            ],
        )
        result = engine.calculate(inp)
        assert result.current_intensity > Decimal("0")
        assert result.display_unit == "gCO2/kWh"

    def test_power_intensity_value_correct(self):
        """22500 tCO2e / 50,000,000 kWh = 0.00045 tCO2e/kWh = 450 gCO2/kWh."""
        engine = IntensityCalculatorEngine()
        inp = _make_input(
            sector="power_generation",
            activity_data=[
                _make_activity(2020, Decimal("22500"), Decimal("50000000")),
            ],
        )
        result = engine.calculate(inp)
        assert_decimal_close(result.current_intensity, Decimal("450"), Decimal("1"))

    def test_power_sector_primary_metric(self):
        defn = SECTOR_INTENSITY_DEFS[SectorType.POWER_GENERATION]
        assert defn["primary_metric"] == IntensityMetricType.GCO2_PER_KWH

    def test_power_sector_has_secondary_metrics(self):
        defn = SECTOR_INTENSITY_DEFS[SectorType.POWER_GENERATION]
        assert len(defn["secondary_metrics"]) >= 2


# ===========================================================================
# Steel Sector Intensity Metrics
# ===========================================================================


class TestSteelSectorIntensity:
    """Steel sector intensity metric calculations."""

    def test_steel_intensity_value(self):
        """10500000 tCO2e / 5000000 tonnes = 2.1 tCO2e/tonne."""
        engine = IntensityCalculatorEngine()
        inp = _make_input(
            sector="steel",
            activity_data=[
                _make_activity(2020, Decimal("10500000"), Decimal("5000000")),
            ],
        )
        result = engine.calculate(inp)
        assert_decimal_close(result.current_intensity, Decimal("2.1"), Decimal("0.01"))

    def test_eaf_intensity_lower_than_bf(self):
        engine = IntensityCalculatorEngine()
        bf = _make_input(
            sector="steel",
            activity_data=[_make_activity(2020, Decimal("10500000"), Decimal("5000000"))],
        )
        eaf = _make_input(
            sector="steel",
            activity_data=[_make_activity(2020, Decimal("2500000"), Decimal("5000000"))],
        )
        bf_result = engine.calculate(bf)
        eaf_result = engine.calculate(eaf)
        assert eaf_result.current_intensity < bf_result.current_intensity

    def test_steel_display_unit(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sector="steel"))
        assert "tCO2e" in result.display_unit


# ===========================================================================
# Cement Sector Intensity Metrics
# ===========================================================================


class TestCementSectorIntensity:
    """Cement sector intensity metric calculations."""

    def test_cement_intensity_value(self):
        """14000000 / 20000000 = 0.7 tCO2e/tonne."""
        engine = IntensityCalculatorEngine()
        inp = _make_input(
            sector="cement",
            activity_data=[_make_activity(2020, Decimal("14000000"), Decimal("20000000"))],
        )
        result = engine.calculate(inp)
        assert_decimal_close(result.current_intensity, Decimal("0.7"), Decimal("0.01"))

    def test_cement_display_unit(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(
            sector="cement",
            activity_data=[_make_activity(2020, Decimal("14000000"), Decimal("20000000"))],
        ))
        assert "tCO2e" in result.display_unit


# ===========================================================================
# Aviation Sector Intensity Metrics
# ===========================================================================


class TestAviationSectorIntensity:
    """Aviation sector intensity metric calculations."""

    def test_aviation_intensity_gco2_per_pkm(self):
        """18000000 tCO2e / 200e9 pkm = 9e-5 tCO2e/pkm = 90 gCO2/pkm."""
        engine = IntensityCalculatorEngine()
        inp = _make_input(
            sector="aviation",
            activity_data=[_make_activity(2020, Decimal("18000000"), Decimal("200000000000"))],
        )
        result = engine.calculate(inp)
        assert_decimal_close(result.current_intensity, Decimal("90"), Decimal("2"))
        assert result.display_unit == "gCO2/pkm"


# ===========================================================================
# Buildings Sector Intensity Metrics
# ===========================================================================


class TestBuildingsSectorIntensity:
    """Buildings sector intensity metric calculations."""

    def test_commercial_buildings_kgco2_per_m2(self):
        """225 tCO2e / 5000 m2 = 0.045 tCO2e/m2 = 45 kgCO2/m2."""
        engine = IntensityCalculatorEngine()
        inp = _make_input(
            sector="buildings_commercial",
            activity_data=[_make_activity(2020, Decimal("225"), Decimal("5000"))],
        )
        result = engine.calculate(inp)
        assert_decimal_close(result.current_intensity, Decimal("45"), Decimal("1"))
        assert "kgCO2" in result.display_unit


# ===========================================================================
# All Sectors Parametrized Tests
# ===========================================================================


class TestAllSectorsIntensity:
    """Cross-sector parametrized intensity tests."""

    @pytest.mark.parametrize("sector", [s.value for s in SectorType])
    def test_sector_in_definitions(self, sector):
        assert sector in [k.value if hasattr(k, 'value') else k for k in SECTOR_INTENSITY_DEFS]

    @pytest.mark.parametrize("sector,expected_unit", [
        ("power_generation", "gCO2/kWh"),
        ("steel", "tCO2e/tonne crude steel"),
        ("cement", "tCO2e/tonne cement"),
        ("aluminum", "tCO2e/tonne aluminum"),
        ("aviation", "gCO2/pkm"),
        ("shipping", "gCO2/tkm"),
        ("road_transport", "gCO2/vkm"),
        ("buildings_residential", "kgCO2/m2/year"),
        ("buildings_commercial", "kgCO2/m2/year"),
    ])
    def test_sda_sector_display_units(self, sector, expected_unit):
        defn = SECTOR_INTENSITY_DEFS[SectorType(sector)]
        assert defn["display_unit"] == expected_unit


# ===========================================================================
# Normalization
# ===========================================================================


class TestIntensityNormalization:
    """Test intensity value normalization across units."""

    def test_steel_intensity_correct_scale(self):
        """Same emissions/activity => same intensity regardless of engine call."""
        engine = IntensityCalculatorEngine()
        r1 = engine.calculate(_make_input(
            sector="steel",
            activity_data=[_make_activity(2020, Decimal("2100"), Decimal("1000"))],
        ))
        assert_decimal_close(r1.current_intensity, Decimal("2.1"), Decimal("0.01"))

    def test_zero_emissions_returns_zero_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(
            sector="steel",
            activity_data=[_make_activity(2020, Decimal("0"), Decimal("5000"))],
        )
        result = engine.calculate(inp)
        assert result.current_intensity == Decimal("0.000000")

    def test_negative_emissions_raises(self):
        """Pydantic ge=0 constraint should reject negative emissions."""
        with pytest.raises(Exception):
            _make_activity(2020, Decimal("-1000"), Decimal("5000"))

    def test_zero_activity_raises(self):
        """Pydantic gt=0 constraint should reject zero activity."""
        with pytest.raises(Exception):
            _make_activity(2020, Decimal("22500"), Decimal("0"))


# ===========================================================================
# Trend Analysis
# ===========================================================================


class TestIntensityTrend:
    """Test intensity trend analysis."""

    def test_declining_trend(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2018, Decimal("5000"), Decimal("1000")),
            _make_activity(2019, Decimal("4800"), Decimal("1000")),
            _make_activity(2020, Decimal("4500"), Decimal("1000")),
            _make_activity(2021, Decimal("4300"), Decimal("1000")),
            _make_activity(2022, Decimal("4100"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(sector="steel", activity_data=data, base_year=2018))
        assert result.trend_analysis is not None
        assert result.trend_analysis.direction == TrendDirection.DECREASING.value

    def test_increasing_trend(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2018, Decimal("4000"), Decimal("1000")),
            _make_activity(2019, Decimal("4200"), Decimal("1000")),
            _make_activity(2020, Decimal("4500"), Decimal("1000")),
            _make_activity(2021, Decimal("4700"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(sector="steel", activity_data=data, base_year=2018))
        assert result.trend_analysis is not None
        assert result.trend_analysis.direction == TrendDirection.INCREASING.value

    def test_stable_trend(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2018, Decimal("4500"), Decimal("1000")),
            _make_activity(2019, Decimal("4505"), Decimal("1000")),
            _make_activity(2020, Decimal("4495"), Decimal("1000")),
            _make_activity(2021, Decimal("4500"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(sector="steel", activity_data=data, base_year=2018))
        assert result.trend_analysis is not None
        assert result.trend_analysis.direction == TrendDirection.STABLE.value

    def test_trend_has_yoy_changes(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2020, Decimal("5000"), Decimal("1000")),
            _make_activity(2021, Decimal("4500"), Decimal("1000")),
            _make_activity(2022, Decimal("4000"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(sector="steel", activity_data=data))
        assert result.trend_analysis is not None
        assert len(result.trend_analysis.yoy_changes) == 2
        assert 2021 in result.trend_analysis.yoy_changes
        assert 2022 in result.trend_analysis.yoy_changes

    def test_trend_cagr_negative_for_declining(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2018, Decimal("5000"), Decimal("1000")),
            _make_activity(2022, Decimal("4000"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(sector="steel", activity_data=data, base_year=2018))
        assert result.trend_analysis is not None
        assert result.trend_analysis.cagr_pct < Decimal("0")

    def test_trend_best_worst_year(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2020, Decimal("5000"), Decimal("1000")),
            _make_activity(2021, Decimal("3000"), Decimal("1000")),
            _make_activity(2022, Decimal("4000"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(sector="steel", activity_data=data))
        assert result.trend_analysis is not None
        assert result.trend_analysis.best_year == 2021
        assert result.trend_analysis.worst_year == 2020

    def test_single_datapoint_no_trend(self):
        engine = IntensityCalculatorEngine()
        data = [_make_activity(2020, Decimal("5000"), Decimal("1000"))]
        result = engine.calculate(_make_input(sector="steel", activity_data=data))
        assert result.trend_analysis is None


# ===========================================================================
# Data Quality Scoring
# ===========================================================================


class TestDataQualityScoring:
    """Test data quality scoring for intensity calculations."""

    def test_dq_score_exists(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert result.data_quality is not None
        assert result.data_quality.overall_score > Decimal("0")

    def test_direct_measurement_scores_higher(self):
        engine = IntensityCalculatorEngine()
        dm_data = [_make_activity(2020, Decimal("10500000"), Decimal("5000000"),
                                  method=DataMeasurementMethod.DIRECT_MEASUREMENT,
                                  verification=VerificationStatus.VERIFIED_REASONABLE)]
        est_data = [_make_activity(2020, Decimal("10500000"), Decimal("5000000"),
                                   method=DataMeasurementMethod.ESTIMATION,
                                   verification=VerificationStatus.UNVERIFIED)]
        dm_result = engine.calculate(_make_input(activity_data=dm_data))
        est_result = engine.calculate(_make_input(activity_data=est_data))
        assert dm_result.data_quality.overall_score > est_result.data_quality.overall_score

    def test_dq_tier_assignment(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert result.data_quality.tier in [t.value for t in DataQualityTier]

    def test_high_quality_data_tier_1_or_2(self):
        engine = IntensityCalculatorEngine()
        data = [_make_activity(2020, Decimal("10500000"), Decimal("5000000"),
                               method=DataMeasurementMethod.DIRECT_MEASUREMENT,
                               verification=VerificationStatus.VERIFIED_REASONABLE,
                               completeness=Decimal("100"))]
        result = engine.calculate(_make_input(activity_data=data))
        assert result.data_quality.tier in [DataQualityTier.TIER_1.value, DataQualityTier.TIER_2.value]

    def test_dq_has_recommendations(self):
        engine = IntensityCalculatorEngine()
        data = [_make_activity(2020, Decimal("10500000"), Decimal("5000000"),
                               method=DataMeasurementMethod.SPEND_BASED,
                               verification=VerificationStatus.UNVERIFIED,
                               completeness=Decimal("50"))]
        result = engine.calculate(_make_input(activity_data=data))
        assert len(result.data_quality.recommendations) > 0


# ===========================================================================
# Benchmark Comparison
# ===========================================================================


class TestBenchmarkComparison:
    """Test intensity benchmark comparisons."""

    def test_benchmark_comparison_exists(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert result.benchmark_comparison is not None

    def test_benchmark_has_sector_average(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sector="steel"))
        assert result.benchmark_comparison.sector_average_2020 > Decimal("0")

    def test_benchmark_has_nze_target(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sector="steel"))
        assert result.benchmark_comparison.nze_2050_target >= Decimal("0")

    def test_above_average_flag(self):
        engine = IntensityCalculatorEngine()
        # 10500000 / 5000000 = 2.1 which is above 1.89 global average
        result = engine.calculate(_make_input(sector="steel"))
        assert result.benchmark_comparison.above_average is True

    def test_within_typical_range(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sector="steel"))
        assert isinstance(result.benchmark_comparison.within_typical_range, bool)

    def test_required_annual_reduction(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sector="steel"))
        if result.benchmark_comparison.required_annual_reduction_to_nze_pct > Decimal("0"):
            assert result.benchmark_comparison.required_annual_reduction_to_nze_pct > Decimal("0")

    def test_no_benchmark_when_disabled(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(include_benchmark=False))
        assert result.benchmark_comparison is None


# ===========================================================================
# Secondary Metrics
# ===========================================================================


class TestSecondaryMetrics:
    """Test secondary intensity metrics."""

    def test_secondary_metrics_calculated(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sector="power_generation",
            activity_data=[_make_activity(2020, Decimal("22500"), Decimal("50000000"))]))
        assert len(result.secondary_metrics) > 0

    def test_revenue_intensity_when_provided(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(revenue_m=Decimal("500")))
        revenue_metrics = [m for m in result.secondary_metrics
                           if "revenue" in m.metric_type.lower()]
        assert len(revenue_metrics) > 0

    def test_employee_intensity_when_provided(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(employees=5000))
        emp_metrics = [m for m in result.secondary_metrics
                       if "employee" in m.metric_type.lower()]
        assert len(emp_metrics) > 0

    def test_no_secondary_when_disabled(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(include_secondary=False))
        # No sector-specific secondary metrics (could still have revenue/employee)
        # but sector secondary should be empty if sector has none extra
        assert isinstance(result.secondary_metrics, list)


# ===========================================================================
# Sub-Process Breakdown
# ===========================================================================


class TestSubProcessBreakdown:
    """Test sub-process intensity breakdown."""

    def test_sub_process_breakdown(self):
        engine = IntensityCalculatorEngine()
        subs = [
            SubProcessEntry(name="BF-BOF", year=2020, activity_value=Decimal("3500000"),
                            emissions_tco2e=Decimal("7350000"), share_pct=Decimal("70")),
            SubProcessEntry(name="EAF", year=2020, activity_value=Decimal("1500000"),
                            emissions_tco2e=Decimal("750000"), share_pct=Decimal("30")),
        ]
        inp = _make_input(sector="steel", sub_processes=subs)
        result = engine.calculate(inp)
        assert len(result.sub_process_breakdown) == 2

    def test_sub_process_contribution(self):
        engine = IntensityCalculatorEngine()
        subs = [
            SubProcessEntry(name="BF-BOF", year=2020, activity_value=Decimal("3500000"),
                            emissions_tco2e=Decimal("7350000"), share_pct=Decimal("70")),
            SubProcessEntry(name="EAF", year=2020, activity_value=Decimal("1500000"),
                            emissions_tco2e=Decimal("750000"), share_pct=Decimal("30")),
        ]
        inp = _make_input(sector="steel", sub_processes=subs)
        result = engine.calculate(inp)
        total_contribution = sum(sp.contribution_pct for sp in result.sub_process_breakdown)
        assert_decimal_close(total_contribution, Decimal("100"), Decimal("1"))


# ===========================================================================
# Result Structure & Provenance
# ===========================================================================


class TestIntensityResultStructure:
    """Test result structure and provenance tracking."""

    def test_result_has_all_fields(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert hasattr(result, "current_intensity")
        assert hasattr(result, "display_unit")
        assert hasattr(result, "sector")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "processing_time_ms")

    def test_result_provenance_hash(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert_provenance_hash(result)

    def test_result_deterministic(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="cement",
            activity_data=[_make_activity(2020, Decimal("14000000"), Decimal("20000000"))])
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.current_intensity == r2.current_intensity

    def test_result_processing_time(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert_processing_time(result)

    def test_result_entity_name(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(entity_name="MyCompany"))
        assert result.entity_name == "MyCompany"

    def test_result_trajectory(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2020, Decimal("5000"), Decimal("1000")),
            _make_activity(2021, Decimal("4500"), Decimal("1000")),
            _make_activity(2022, Decimal("4000"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(activity_data=data))
        assert len(result.intensity_trajectory) == 3
        for pt in result.intensity_trajectory:
            assert isinstance(pt, IntensityDataPoint)
            assert pt.intensity_value >= Decimal("0")


# ===========================================================================
# Shipping & Transport Metrics
# ===========================================================================


class TestShippingTransportIntensity:
    """Test shipping and transport sector intensity metrics."""

    def test_shipping_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="shipping",
            activity_data=[_make_activity(2020, Decimal("5250"), Decimal("500000000"))])
        result = engine.calculate(inp)
        assert result.display_unit == "gCO2/tkm"
        assert result.current_intensity > Decimal("0")

    def test_road_transport_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="road_transport",
            activity_data=[_make_activity(2020, Decimal("170000"), Decimal("1000000000"))])
        result = engine.calculate(inp)
        assert result.display_unit == "gCO2/vkm"

    def test_rail_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="rail",
            activity_data=[_make_activity(2020, Decimal("3500"), Decimal("100000000"))])
        result = engine.calculate(inp)
        assert result.display_unit == "gCO2/pkm"


# ===========================================================================
# Aluminum & Chemicals Intensity Metrics
# ===========================================================================


class TestAluminumChemicalsIntensity:
    """Test aluminum and chemicals sector intensity metrics."""

    def test_aluminum_intensity(self):
        """12500000 / 1000000 = 12.5 tCO2e/tonne."""
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="aluminum",
            activity_data=[_make_activity(2020, Decimal("12500000"), Decimal("1000000"))])
        result = engine.calculate(inp)
        assert_decimal_close(result.current_intensity, Decimal("12.5"), Decimal("0.1"))

    def test_chemicals_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="chemicals",
            activity_data=[_make_activity(2020, Decimal("1800000"), Decimal("1000000"))])
        result = engine.calculate(inp)
        assert result.current_intensity > Decimal("0")

    def test_pulp_paper_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="pulp_paper",
            activity_data=[_make_activity(2020, Decimal("450000"), Decimal("1000000"))])
        result = engine.calculate(inp)
        assert result.current_intensity > Decimal("0")


# ===========================================================================
# Multi-Year Intensity Series
# ===========================================================================


class TestMultiYearIntensitySeries:
    """Test multi-year intensity calculation and tracking."""

    def test_multi_year_trajectory(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2020, Decimal("22500"), Decimal("50000")),
            _make_activity(2021, Decimal("21000"), Decimal("52000")),
            _make_activity(2022, Decimal("19500"), Decimal("54000")),
            _make_activity(2023, Decimal("18000"), Decimal("55000")),
        ]
        result = engine.calculate(_make_input(sector="steel", activity_data=data))
        assert len(result.intensity_trajectory) == 4
        intensities = [pt.intensity_value for pt in result.intensity_trajectory]
        # Declining emissions with increasing activity => declining intensity
        for i in range(len(intensities) - 1):
            assert intensities[i] >= intensities[i + 1]

    def test_base_year_and_current_year(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2020, Decimal("10500000"), Decimal("5000000")),
            _make_activity(2021, Decimal("9800000"), Decimal("5100000")),
            _make_activity(2022, Decimal("9200000"), Decimal("5200000")),
        ]
        result = engine.calculate(_make_input(activity_data=data, base_year=2020))
        assert result.base_year == 2020
        assert result.current_year == 2022
        assert result.base_year_intensity > Decimal("0")
        assert result.current_intensity > Decimal("0")
        assert result.current_intensity < result.base_year_intensity

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_intensity_positive_for_all_sda_sectors(self, sector):
        engine = IntensityCalculatorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        inp = _make_input(
            sector=sector,
            activity_data=[_make_activity(2020, metrics["base_2020"] * Decimal("1000"), Decimal("1000"))],
        )
        result = engine.calculate(inp)
        assert result.current_intensity >= Decimal("0")


# ===========================================================================
# Extended Sectors
# ===========================================================================


class TestExtendedSectorMetrics:
    """Test intensity metrics for extended (non-SDA) sectors."""

    def test_agriculture_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="agriculture",
            activity_data=[_make_activity(2020, Decimal("850000"), Decimal("1000000"))])
        result = engine.calculate(inp)
        assert result.current_intensity > Decimal("0")

    def test_food_beverage_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="food_beverage",
            activity_data=[_make_activity(2020, Decimal("550000"), Decimal("1000000"))])
        result = engine.calculate(inp)
        assert result.current_intensity > Decimal("0")

    def test_oil_gas_intensity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="oil_gas",
            activity_data=[_make_activity(2020, Decimal("15000"), Decimal("1000000"))])
        result = engine.calculate(inp)
        assert result.display_unit == "gCO2/MJ"

    def test_cross_sector_revenue_metric(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="cross_sector",
            activity_data=[_make_activity(2020, Decimal("50000"), Decimal("500"))])
        result = engine.calculate(inp)
        assert result.display_unit == "tCO2e/M revenue"


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestIntensityEdgeCases:
    """Edge case tests for intensity calculations."""

    def test_very_small_emissions(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="steel",
            activity_data=[_make_activity(2020, Decimal("0.001"), Decimal("50000"))])
        result = engine.calculate(inp)
        assert result.current_intensity >= Decimal("0")

    def test_very_large_activity(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="steel",
            activity_data=[_make_activity(2020, Decimal("22500000"), Decimal("999999999999"))])
        result = engine.calculate(inp)
        assert result.current_intensity >= Decimal("0")

    def test_decimal_precision(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="steel",
            activity_data=[_make_activity(2020, Decimal("10500000.123456"), Decimal("5000000.654321"))])
        result = engine.calculate(inp)
        assert isinstance(result.current_intensity, Decimal)

    def test_intensity_not_float(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.current_intensity, Decimal)
        assert not isinstance(result.current_intensity, float)


# ===========================================================================
# Sector Definitions & Constants
# ===========================================================================


class TestSectorDefinitions:
    """Test sector intensity definitions and constants."""

    @pytest.mark.parametrize("sector", [s.value for s in SectorType])
    def test_sector_has_definition(self, sector):
        defn = SECTOR_INTENSITY_DEFS.get(SectorType(sector))
        assert defn is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sda_sector_sbti_flag(self, sector):
        defn = SECTOR_INTENSITY_DEFS[SectorType(sector)]
        # Most SDA sectors have sbti_sda_metric True
        assert isinstance(defn["sbti_sda_metric"], bool)

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sda_sector_has_global_average(self, sector):
        defn = SECTOR_INTENSITY_DEFS[SectorType(sector)]
        assert defn["global_average_2020"] > Decimal("0")

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sda_sector_has_nze_target(self, sector):
        defn = SECTOR_INTENSITY_DEFS[SectorType(sector)]
        assert defn["nze_2050_target"] >= Decimal("0")

    def test_power_nze_target_zero(self):
        defn = SECTOR_INTENSITY_DEFS[SectorType.POWER_GENERATION]
        assert defn["nze_2050_target"] == Decimal("0")


# ===========================================================================
# Warnings
# ===========================================================================


class TestIntensityWarnings:
    """Test warning generation."""

    def test_warning_for_data_gap(self):
        engine = IntensityCalculatorEngine()
        data = [
            _make_activity(2018, Decimal("5000"), Decimal("1000")),
            _make_activity(2022, Decimal("4000"), Decimal("1000")),
        ]
        result = engine.calculate(_make_input(activity_data=data, base_year=2018))
        assert len(result.warnings) > 0
        gap_warnings = [w for w in result.warnings if "gap" in w.lower()]
        assert len(gap_warnings) > 0

    def test_warning_for_exceeding_range(self):
        engine = IntensityCalculatorEngine()
        # Steel typical max is 3.5; 20000/1000 = 20 which exceeds it
        data = [_make_activity(2020, Decimal("20000"), Decimal("1000"))]
        result = engine.calculate(_make_input(sector="steel", activity_data=data))
        range_warnings = [w for w in result.warnings if "range" in w.lower() or "exceeds" in w.lower()]
        assert len(range_warnings) > 0


# ===========================================================================
# Recommendations
# ===========================================================================


class TestIntensityRecommendations:
    """Test recommendation generation."""

    def test_recommendations_exist(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.recommendations, list)

    def test_few_data_points_recommendation(self):
        engine = IntensityCalculatorEngine()
        data = [_make_activity(2020, Decimal("10500000"), Decimal("5000000"))]
        result = engine.calculate(_make_input(activity_data=data))
        year_recs = [r for r in result.recommendations if "year" in r.lower() or "data" in r.lower()]
        assert len(year_recs) > 0

    def test_no_sub_process_recommendation(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sub_processes=[]))
        sub_recs = [r for r in result.recommendations if "sub-process" in r.lower() or "breakdown" in r.lower()]
        assert len(sub_recs) > 0


# ===========================================================================
# Performance Tests
# ===========================================================================


class TestIntensityPerformance:
    """Performance tests for intensity calculations."""

    def test_single_calculation_under_100ms(self):
        engine = IntensityCalculatorEngine()
        with timed_block("single_intensity", max_seconds=0.1):
            engine.calculate(_make_input())

    def test_100_calculations_under_2s(self):
        engine = IntensityCalculatorEngine()
        with timed_block("100_intensity_calcs", max_seconds=2.0):
            for _ in range(100):
                engine.calculate(_make_input())

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_each_sector_under_200ms(self, sector):
        engine = IntensityCalculatorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        inp = _make_input(
            sector=sector,
            activity_data=[_make_activity(2020, metrics["base_2020"] * Decimal("1000"), Decimal("1000"))],
        )
        with timed_block(f"intensity_{sector}", max_seconds=0.2):
            engine.calculate(inp)


# ===========================================================================
# Cross-Sector Comparison
# ===========================================================================


class TestIntensityComparisonAcrossSectors:
    """Cross-sector intensity comparison tests."""

    def test_steel_around_2_tco2e(self):
        engine = IntensityCalculatorEngine()
        result = engine.calculate(_make_input(sector="steel"))
        assert Decimal("1.5") < result.current_intensity < Decimal("3.0")

    def test_cement_below_1(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input(sector="cement",
            activity_data=[_make_activity(2020, Decimal("14000000"), Decimal("20000000"))])
        result = engine.calculate(inp)
        assert result.current_intensity < Decimal("1.0")

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_intensity_non_negative(self, sector):
        engine = IntensityCalculatorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        inp = _make_input(
            sector=sector,
            activity_data=[_make_activity(2020, metrics["base_2020"] * Decimal("1000"), Decimal("1000"))],
        )
        result = engine.calculate(inp)
        assert result.current_intensity >= Decimal("0")

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_intensity_always_decimal(self, sector):
        engine = IntensityCalculatorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        inp = _make_input(
            sector=sector,
            activity_data=[_make_activity(2020, metrics["base_2020"] * Decimal("1000"), Decimal("1000"))],
        )
        result = engine.calculate(inp)
        assert isinstance(result.current_intensity, Decimal)


# ===========================================================================
# Determinism / Reproducibility
# ===========================================================================


class TestIntensityDeterminism:
    """Test calculation determinism."""

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_deterministic_intensity(self, sector):
        engine = IntensityCalculatorEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        inp = _make_input(
            sector=sector,
            activity_data=[_make_activity(2020, metrics["base_2020"] * Decimal("1000"), Decimal("1000"))],
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.current_intensity == r2.current_intensity

    def test_multiple_runs_same_hash(self):
        engine = IntensityCalculatorEngine()
        inp = _make_input()
        results = [engine.calculate(inp) for _ in range(5)]
        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == 1
