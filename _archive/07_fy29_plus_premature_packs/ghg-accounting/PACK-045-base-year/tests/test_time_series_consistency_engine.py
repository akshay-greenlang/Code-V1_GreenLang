# -*- coding: utf-8 -*-
"""
Tests for TimeSeriesConsistencyEngine (Engine 7).

Covers consistency checks, normalization, trend analysis, and reporting.
Target: ~50 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.time_series_consistency_engine import (
    TimeSeriesConsistencyEngine,
    YearData,
    ConsistencyConfig,
    ConsistencyResult,
    ConsistencyStatus,
    InconsistencyFinding,
    NormalizationAdjustment,
    TrendPoint,
    TrendValidationResult,
    GWPVersion,
    ConsolidationApproach,
    ReportingFramework,
)

# Try to import optional types that may or may not exist
try:
    from engines.time_series_consistency_engine import InconsistencyType, NormalizationType
except ImportError:
    InconsistencyType = None
    NormalizationType = None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_time_series():
    """Sample time series data for 5 years with consistent methodology."""
    return [
        YearData(
            year=2019,
            total_tco2e=Decimal("10000"),
            scope1_tco2e=Decimal("5000"),
            scope2_location_tco2e=Decimal("3000"),
            scope2_market_tco2e=Decimal("2800"),
            scope3_tco2e=Decimal("2000"),
            gwp_version=GWPVersion.AR5,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
            is_base_year=True,
        ),
        YearData(
            year=2020,
            total_tco2e=Decimal("9500"),
            scope1_tco2e=Decimal("4800"),
            scope2_location_tco2e=Decimal("2800"),
            scope2_market_tco2e=Decimal("2600"),
            scope3_tco2e=Decimal("1900"),
            gwp_version=GWPVersion.AR5,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ),
        YearData(
            year=2021,
            total_tco2e=Decimal("9200"),
            scope1_tco2e=Decimal("4600"),
            scope2_location_tco2e=Decimal("2700"),
            scope2_market_tco2e=Decimal("2500"),
            scope3_tco2e=Decimal("1900"),
            gwp_version=GWPVersion.AR5,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ),
        YearData(
            year=2022,
            total_tco2e=Decimal("9000"),
            scope1_tco2e=Decimal("4500"),
            scope2_location_tco2e=Decimal("2600"),
            scope2_market_tco2e=Decimal("2400"),
            scope3_tco2e=Decimal("1900"),
            gwp_version=GWPVersion.AR5,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ),
        YearData(
            year=2023,
            total_tco2e=Decimal("8800"),
            scope1_tco2e=Decimal("4400"),
            scope2_location_tco2e=Decimal("2500"),
            scope2_market_tco2e=Decimal("2300"),
            scope3_tco2e=Decimal("1900"),
            gwp_version=GWPVersion.AR5,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ),
    ]


@pytest.fixture
def inconsistent_time_series():
    """Time series with inconsistent GWP versions."""
    return [
        YearData(
            year=2019,
            total_tco2e=Decimal("10000"),
            scope1_tco2e=Decimal("5000"),
            scope2_location_tco2e=Decimal("3000"),
            gwp_version=GWPVersion.AR4,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ),
        YearData(
            year=2020,
            total_tco2e=Decimal("9500"),
            scope1_tco2e=Decimal("4800"),
            scope2_location_tco2e=Decimal("2800"),
            gwp_version=GWPVersion.AR5,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ),
        YearData(
            year=2021,
            total_tco2e=Decimal("9200"),
            scope1_tco2e=Decimal("4600"),
            scope2_location_tco2e=Decimal("2700"),
            gwp_version=GWPVersion.AR6,
            consolidation_approach=ConsolidationApproach.OPERATIONAL_CONTROL,
        ),
    ]


# ============================================================================
# Engine Init
# ============================================================================

class TestTimeSeriesConsistencyEngineInit:
    def test_engine_creation(self, time_series_engine):
        assert time_series_engine is not None

    def test_engine_is_instance(self, time_series_engine):
        assert isinstance(time_series_engine, TimeSeriesConsistencyEngine)


# ============================================================================
# Assess Consistency (takes only year_data_series, no config param)
# ============================================================================

class TestAssessConsistency:
    def test_consistent_series(self, time_series_engine, sample_time_series):
        result = time_series_engine.assess_consistency(sample_time_series)
        assert isinstance(result, ConsistencyResult)
        assert result.status in list(ConsistencyStatus)

    def test_inconsistent_gwp(self, time_series_engine, inconsistent_time_series):
        result = time_series_engine.assess_consistency(inconsistent_time_series)
        assert isinstance(result, ConsistencyResult)
        # Should have findings
        assert result.total_findings >= 1

    def test_result_has_provenance_hash(self, time_series_engine, sample_time_series):
        result = time_series_engine.assess_consistency(sample_time_series)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_result_has_years_assessed(self, time_series_engine, sample_time_series):
        result = time_series_engine.assess_consistency(sample_time_series)
        assert len(result.years_assessed) >= 5

    def test_result_processing_time(self, time_series_engine, sample_time_series):
        result = time_series_engine.assess_consistency(sample_time_series)
        assert result.processing_time_ms >= 0


# ============================================================================
# Individual Consistency Checks
# ============================================================================

class TestCheckGWPConsistency:
    def test_consistent_gwp(self, time_series_engine, sample_time_series):
        findings = time_series_engine.check_gwp_consistency(sample_time_series)
        # All AR5 -> no findings
        assert len(findings) == 0

    def test_inconsistent_gwp(self, time_series_engine, inconsistent_time_series):
        findings = time_series_engine.check_gwp_consistency(inconsistent_time_series)
        assert len(findings) >= 1

    def test_findings_are_correct_type(self, time_series_engine, inconsistent_time_series):
        findings = time_series_engine.check_gwp_consistency(inconsistent_time_series)
        for f in findings:
            assert isinstance(f, InconsistencyFinding)


class TestCheckMethodologyConsistency:
    def test_consistent_methodology(self, time_series_engine, sample_time_series):
        findings = time_series_engine.check_methodology_consistency(sample_time_series)
        assert isinstance(findings, list)


class TestCheckBoundaryConsistency:
    def test_consistent_boundary(self, time_series_engine, sample_time_series):
        findings = time_series_engine.check_boundary_consistency(sample_time_series)
        assert isinstance(findings, list)


class TestCheckCoverageConsistency:
    def test_consistent_coverage(self, time_series_engine, sample_time_series):
        findings = time_series_engine.check_coverage_consistency(sample_time_series)
        assert isinstance(findings, list)


# ============================================================================
# Structural Changes
# ============================================================================

class TestDetectStructuralChanges:
    def test_no_structural_changes(self, time_series_engine, sample_time_series):
        changes = time_series_engine.detect_structural_changes(sample_time_series)
        assert isinstance(changes, list)

    def test_returns_list_of_dicts(self, time_series_engine, sample_time_series):
        changes = time_series_engine.detect_structural_changes(sample_time_series)
        for c in changes:
            assert isinstance(c, dict)


# ============================================================================
# Normalization
# ============================================================================

class TestNormalize:
    def test_normalize_for_structural_changes(self, time_series_engine, sample_time_series):
        adjustments = [
            NormalizationAdjustment(
                year=2019,
                normalization_type="structural_change",
                adjustment_tco2e=Decimal("500"),
                description="Add acquired entity emissions",
            ),
        ]
        result = time_series_engine.normalize_for_structural_changes(
            sample_time_series, adjustments
        )
        assert isinstance(result, list)
        assert len(result) >= len(sample_time_series)

    def test_normalize_returns_trend_points(self, time_series_engine, sample_time_series):
        adjustments = [
            NormalizationAdjustment(
                year=2020,
                normalization_type="structural_change",
                adjustment_tco2e=Decimal("200"),
            ),
        ]
        result = time_series_engine.normalize_for_structural_changes(
            sample_time_series, adjustments
        )
        for point in result:
            assert isinstance(point, TrendPoint)


# ============================================================================
# Trend Analysis
# ============================================================================

class TestCalculateTrend:
    def test_calculate_trend(self, time_series_engine, sample_time_series):
        trend = time_series_engine.calculate_trend(sample_time_series)
        assert isinstance(trend, list)
        assert len(trend) >= 1

    def test_trend_points_have_year(self, time_series_engine, sample_time_series):
        trend = time_series_engine.calculate_trend(sample_time_series)
        for point in trend:
            assert isinstance(point, TrendPoint)
            assert point.year >= 2019

    def test_trend_covers_all_years(self, time_series_engine, sample_time_series):
        trend = time_series_engine.calculate_trend(sample_time_series)
        years = [p.year for p in trend]
        assert 2019 in years
        assert 2023 in years


class TestSummarizeTrend:
    def test_summarize_trend(self, time_series_engine, sample_time_series):
        """summarize_trend takes List[TrendPoint], not List[YearData]."""
        trend_points = time_series_engine.calculate_trend(sample_time_series)
        summary = time_series_engine.summarize_trend(trend_points)
        assert isinstance(summary, dict)


# ============================================================================
# Compare Years
# ============================================================================

class TestCompareTwoYears:
    def test_compare_two_years(self, time_series_engine, sample_time_series):
        result = time_series_engine.compare_two_years(
            sample_time_series[0], sample_time_series[-1]
        )
        assert isinstance(result, dict)

    def test_compare_same_year(self, time_series_engine, sample_time_series):
        result = time_series_engine.compare_two_years(
            sample_time_series[0], sample_time_series[0]
        )
        assert isinstance(result, dict)


# ============================================================================
# Validate Trend for Reporting
# ============================================================================

class TestValidateTrend:
    def test_validate_trend_ghg_protocol(self, time_series_engine, sample_time_series):
        result = time_series_engine.validate_trend_for_reporting(
            sample_time_series, framework=ReportingFramework.GHG_PROTOCOL
        )
        assert isinstance(result, TrendValidationResult)

    def test_validate_trend_sbti(self, time_series_engine, sample_time_series):
        result = time_series_engine.validate_trend_for_reporting(
            sample_time_series, framework=ReportingFramework.SBTI
        )
        assert isinstance(result, TrendValidationResult)


# ============================================================================
# Generate Report
# ============================================================================

class TestGenerateConsistencyReport:
    def test_generate_report(self, time_series_engine, sample_time_series):
        consistency_result = time_series_engine.assess_consistency(sample_time_series)
        report = time_series_engine.generate_consistency_report(consistency_result)
        assert isinstance(report, str)
        assert len(report) > 0


# ============================================================================
# Get Year Data Helpers (take only series, no base_year param)
# ============================================================================

class TestGetYearData:
    def test_get_base_year_data(self, time_series_engine, sample_time_series):
        data = time_series_engine.get_base_year_data(sample_time_series)
        assert data is not None
        # First year with is_base_year=True, or earliest year
        assert data.year == 2019

    def test_get_latest_year_data(self, time_series_engine, sample_time_series):
        data = time_series_engine.get_latest_year_data(sample_time_series)
        assert data is not None
        assert data.year == 2023

    def test_get_base_year_empty(self, time_series_engine):
        data = time_series_engine.get_base_year_data([])
        assert data is None

    def test_get_latest_year_empty(self, time_series_engine):
        data = time_series_engine.get_latest_year_data([])
        assert data is None


# ============================================================================
# Model Tests
# ============================================================================

class TestYearDataModel:
    def test_create_year_data(self):
        yd = YearData(year=2022, total_tco2e=Decimal("10000"))
        assert yd.year == 2022

    def test_year_data_with_all_scopes(self):
        yd = YearData(
            year=2022,
            total_tco2e=Decimal("10000"),
            scope1_tco2e=Decimal("5000"),
            scope2_location_tco2e=Decimal("3000"),
            scope2_market_tco2e=Decimal("2800"),
            scope3_tco2e=Decimal("2000"),
        )
        assert yd.scope1_tco2e == Decimal("5000")
        assert yd.scope2_location_tco2e == Decimal("3000")

    def test_year_data_gwp_default(self):
        yd = YearData(year=2022, total_tco2e=Decimal("10000"))
        assert yd.gwp_version == GWPVersion.AR5

    def test_year_data_consolidation_default(self):
        yd = YearData(year=2022, total_tco2e=Decimal("10000"))
        assert yd.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL


# ============================================================================
# Enums
# ============================================================================

class TestEnums:
    def test_consistency_status(self):
        assert ConsistencyStatus.CONSISTENT is not None
        assert ConsistencyStatus.INCONSISTENT is not None
        assert ConsistencyStatus.NOT_ASSESSED is not None
        assert len(ConsistencyStatus) >= 3

    def test_gwp_version(self):
        assert GWPVersion.AR4 is not None
        assert GWPVersion.AR5 is not None
        assert GWPVersion.AR6 is not None
        assert len(GWPVersion) == 3

    def test_consolidation_approach(self):
        assert ConsolidationApproach.OPERATIONAL_CONTROL is not None
        assert ConsolidationApproach.FINANCIAL_CONTROL is not None
        assert ConsolidationApproach.EQUITY_SHARE is not None
        assert len(ConsolidationApproach) == 3

    def test_reporting_framework(self):
        assert ReportingFramework.GHG_PROTOCOL is not None
        assert ReportingFramework.SBTI is not None
        assert len(ReportingFramework) >= 4
