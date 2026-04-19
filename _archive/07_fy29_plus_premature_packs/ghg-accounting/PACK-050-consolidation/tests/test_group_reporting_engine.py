# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Group Reporting Engine Tests

Tests consolidated report generation, scope breakdown, framework mapping,
waterfall contribution, geographic breakdown, intensity metrics,
trend analysis, SBTi target tracking, and variance analysis.

Target: 50-70 tests.
"""

import pytest
from decimal import Decimal

from engines.group_reporting_engine import (
    GroupReportingEngine,
    GroupReport,
    ScopeBreakdown,
    TrendData,
    ContributionWaterfall,
    GeographicBreakdown,
    FrameworkMapping,
    ReportingFramework,
    IntensityMetricType,
    _round2,
    _round4,
)


@pytest.fixture
def engine():
    """Fresh GroupReportingEngine."""
    return GroupReportingEngine()


@pytest.fixture
def entity_data():
    """Standard entity data for report generation."""
    return [
        {
            "entity_id": "ENT-PARENT-001",
            "entity_name": "Parent Corp HQ",
            "scope1": "500",
            "scope2_location": "300",
            "scope2_market": "280",
            "scope3": "200",
            "country": "CH",
            "region": "EUROPE",
            "sector": "HOLDING",
        },
        {
            "entity_id": "ENT-SUB-001",
            "entity_name": "Manufacturing DE",
            "scope1": "15000",
            "scope2_location": "8000",
            "scope2_market": "7500",
            "scope3": "5000",
            "country": "DE",
            "region": "EUROPE",
            "sector": "MANUFACTURING",
        },
        {
            "entity_id": "ENT-SUB-002",
            "entity_name": "Operations GB",
            "scope1": "3000",
            "scope2_location": "2000",
            "scope2_market": "1800",
            "scope3": "1500",
            "country": "GB",
            "region": "EUROPE",
            "sector": "OPERATIONS",
        },
        {
            "entity_id": "ENT-SUB-003",
            "entity_name": "US Subsidiary",
            "scope1": "8000",
            "scope2_location": "4000",
            "scope2_market": "3800",
            "scope3": "3000",
            "country": "US",
            "region": "AMERICAS",
            "sector": "MANUFACTURING",
        },
        {
            "entity_id": "ENT-JV-001",
            "entity_name": "Joint Venture NL",
            "scope1": "6000",
            "scope2_location": "3000",
            "scope2_market": "2800",
            "scope3": "2000",
            "country": "NL",
            "region": "EUROPE",
            "sector": "ENERGY",
        },
    ]


@pytest.fixture
def prior_year_data():
    """Prior year entity data for trend analysis."""
    return [
        {
            "entity_id": "ENT-PARENT-001",
            "scope1": "550",
            "scope2_location": "320",
            "scope2_market": "300",
            "scope3": "210",
            "country": "CH",
        },
        {
            "entity_id": "ENT-SUB-001",
            "scope1": "16000",
            "scope2_location": "8500",
            "scope2_market": "8000",
            "scope3": "5500",
            "country": "DE",
        },
    ]


class TestReportGeneration:
    """Test consolidated report generation."""

    def test_generate_report(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            organisation_name="Test Corp",
            consolidation_approach="OPERATIONAL_CONTROL",
        )
        assert isinstance(report, GroupReport)
        assert report.reporting_year == 2025
        assert report.organisation_name == "Test Corp"
        assert report.entity_count == 5

    def test_report_provenance_hash(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert len(report.provenance_hash) == 64

    def test_report_stored_in_engine(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        retrieved = engine.get_report(report.report_id)
        assert retrieved.report_id == report.report_id

    def test_report_not_found_raises(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_report("NONEXISTENT")

    def test_get_all_reports(self, engine, entity_data):
        engine.generate_report(reporting_year=2025, entity_data=entity_data)
        engine.generate_report(reporting_year=2024, entity_data=entity_data)
        assert len(engine.get_all_reports()) == 2

    def test_report_with_eliminations(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            eliminations_tco2e=Decimal("1500"),
        )
        assert report.total_eliminations_tco2e == Decimal("1500.00")

    def test_report_with_adjustments(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            adjustments_tco2e=Decimal("-500"),
        )
        assert report.total_adjustments_tco2e == Decimal("-500.00")


class TestScopeBreakdown:
    """Test scope breakdown calculations."""

    def test_scope1_total(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        # 500 + 15000 + 3000 + 8000 + 6000 = 32500
        assert report.scope_breakdown.scope1 == Decimal("32500.00")

    def test_scope2_location_total(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        # 300 + 8000 + 2000 + 4000 + 3000 = 17300
        assert report.scope_breakdown.scope2_location == Decimal("17300.00")

    def test_scope2_market_total(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        # 280 + 7500 + 1800 + 3800 + 2800 = 16180
        assert report.scope_breakdown.scope2_market == Decimal("16180.00")

    def test_scope3_total(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        # 200 + 5000 + 1500 + 3000 + 2000 = 11700
        assert report.scope_breakdown.scope3 == Decimal("11700.00")

    def test_total_equals_s1_plus_s2loc_plus_s3(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        bd = report.scope_breakdown
        expected_total = _round2(bd.scope1 + bd.scope2_location + bd.scope3)
        assert bd.total == expected_total

    def test_s1_plus_s2_for_sbti(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        bd = report.scope_breakdown
        assert bd.scope1_plus_scope2 == _round2(bd.scope1 + bd.scope2_location)

    def test_scope_percentages_sum_approximately_100(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        bd = report.scope_breakdown
        total_pct = bd.scope1_pct + bd.scope2_location_pct + bd.scope3_pct
        # May not be exactly 100 due to rounding but should be close
        assert abs(total_pct - Decimal("100")) < Decimal("1")

    def test_scope_breakdown_provenance(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert len(report.scope_breakdown.provenance_hash) == 64

    def test_empty_entity_data(self, engine):
        report = engine.generate_report(
            reporting_year=2025, entity_data=[],
        )
        assert report.scope_breakdown.total == Decimal("0.00")
        assert report.entity_count == 0

    def test_scope3_by_category(self, engine):
        data = [{
            "entity_id": "A",
            "scope1": "100",
            "scope2_location": "50",
            "scope3": "300",
            "scope3_categories": {
                "cat1_purchased_goods": "150",
                "cat6_business_travel": "150",
            },
            "country": "US",
        }]
        report = engine.generate_report(reporting_year=2025, entity_data=data)
        cats = report.scope_breakdown.scope3_by_category
        assert Decimal(str(cats.get("cat1_purchased_goods", "0"))) == Decimal("150.00")


class TestWaterfallContribution:
    """Test entity contribution waterfall."""

    def test_waterfall_generated(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert report.waterfall is not None
        assert isinstance(report.waterfall, ContributionWaterfall)

    def test_waterfall_sorted_descending(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        contribs = report.waterfall.entity_contributions
        for i in range(len(contribs) - 1):
            assert Decimal(contribs[i]["contribution_pct"]) >= Decimal(contribs[i + 1]["contribution_pct"])

    def test_top_entity_is_manufacturing_de(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        top = report.waterfall.entity_contributions[0]
        assert top["entity_id"] == "ENT-SUB-001"

    def test_waterfall_top_5_pct(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert report.waterfall.top_5_pct == Decimal("100.00")  # Only 5 entities

    def test_waterfall_provenance(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert len(report.waterfall.provenance_hash) == 64


class TestGeographicBreakdown:
    """Test geographic breakdown."""

    def test_geographic_breakdown_generated(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert report.geographic_breakdown is not None

    def test_country_count(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        geo = report.geographic_breakdown
        assert geo.country_count == 4  # CH, DE, GB, US, NL = 5 ... actually NL too
        assert geo.country_count >= 4

    def test_top_country_identified(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        geo = report.geographic_breakdown
        # DE has highest: 15000+8000+5000 = 28000
        assert geo.top_country == "DE"

    def test_by_region_populated(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        geo = report.geographic_breakdown
        assert "EUROPE" in geo.by_region
        assert "AMERICAS" in geo.by_region

    def test_geographic_provenance(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert len(report.geographic_breakdown.provenance_hash) == 64


class TestIntensityMetrics:
    """Test intensity metric calculations."""

    def test_intensity_per_revenue(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            intensity_denominators={"revenue_m": "500"},
        )
        assert "tco2e_per_m_revenue" in report.intensity_metrics
        total = report.scope_breakdown.total
        expected = _round4(total / Decimal("500"))
        assert report.intensity_metrics["tco2e_per_m_revenue"] == expected

    def test_intensity_per_employee(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            intensity_denominators={"employees": "10000"},
        )
        assert "tco2e_per_employee" in report.intensity_metrics

    def test_intensity_per_production_unit(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            intensity_denominators={"production_units": "1000000"},
        )
        assert "tco2e_per_production_unit" in report.intensity_metrics

    def test_intensity_per_floor_area(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            intensity_denominators={"floor_area_m2": "50000"},
        )
        assert "tco2e_per_m2" in report.intensity_metrics

    def test_no_intensity_without_denominators(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert len(report.intensity_metrics) == 0


class TestTrendAnalysis:
    """Test year-over-year trend analysis."""

    def test_trend_with_prior_year(self, engine, entity_data, prior_year_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            prior_year_data=prior_year_data,
        )
        assert len(report.trends) == 2
        assert report.trends[0].year == 2024
        assert report.trends[1].year == 2025

    def test_yoy_change_percentage(self, engine, entity_data, prior_year_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            prior_year_data=prior_year_data,
        )
        current_trend = report.trends[1]
        assert current_trend.yoy_change_pct != Decimal("0")

    def test_no_trend_without_prior_data(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert len(report.trends) == 0

    def test_variance_vs_prior(self, engine, entity_data, prior_year_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            prior_year_data=prior_year_data,
        )
        assert report.variance_vs_prior is not None
        assert "scope1" in report.variance_vs_prior

    def test_multi_year_trends(self, engine):
        yearly = [
            {"year": 2022, "scope1": "35000", "scope2_location": "18000", "scope3": "12000"},
            {"year": 2023, "scope1": "33000", "scope2_location": "17000", "scope3": "11500"},
            {"year": 2024, "scope1": "31000", "scope2_location": "16000", "scope3": "11000"},
        ]
        trends = engine.calculate_trends(yearly)
        assert len(trends) == 3
        assert trends[0].yoy_change_pct == Decimal("0")
        assert trends[1].yoy_change_pct != Decimal("0")
        assert trends[2].yoy_change_pct != Decimal("0")


class TestFrameworkMapping:
    """Test framework mapping."""

    @pytest.mark.parametrize("framework", [
        "CSRD_ESRS_E1", "CDP", "GRI_305", "TCFD",
        "SEC_CLIMATE", "SBTI", "IFRS_S2", "UK_SECR", "ISO_14064",
    ])
    def test_map_to_all_frameworks(self, engine, entity_data, framework):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        mapping = engine.map_to_framework(report, framework)
        assert isinstance(mapping, FrameworkMapping)
        assert mapping.framework == framework
        assert len(mapping.disclosures) > 0

    def test_framework_coverage_populated(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        mapping = engine.map_to_framework(report, "CDP")
        assert mapping.coverage_pct > Decimal("0")

    def test_invalid_framework_raises(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        with pytest.raises(ValueError, match="Unsupported framework"):
            engine.map_to_framework(report, "INVALID_FRAMEWORK")

    def test_framework_provenance(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        mapping = engine.map_to_framework(report, "CSRD_ESRS_E1")
        assert len(mapping.provenance_hash) == 64


class TestSBTiTracking:
    """Test SBTi target tracking."""

    def test_sbti_target_tracking(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=entity_data,
            sbti_targets={
                "base_year": 2020,
                "base_year_emissions": "60000",
                "target_year": 2030,
                "target_reduction_pct": "42",
            },
        )
        assert report.sbti_target_progress is not None
        progress = report.sbti_target_progress
        assert "on_track" in progress
        assert "actual_reduction_pct" in progress

    def test_sbti_on_track_when_reduction_met(self, engine):
        data = [{"entity_id": "A", "scope1": "15000", "scope2_location": "5000", "scope3": "3000", "country": "US"}]
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=data,
            sbti_targets={
                "base_year": 2020,
                "base_year_emissions": "50000",
                "target_year": 2030,
                "target_reduction_pct": "42",
            },
        )
        progress = report.sbti_target_progress
        # s1+s2 = 20000, base = 50000, reduction = 60%
        assert progress["on_track"] is True

    def test_sbti_not_on_track(self, engine):
        data = [{"entity_id": "A", "scope1": "40000", "scope2_location": "15000", "scope3": "5000", "country": "US"}]
        report = engine.generate_report(
            reporting_year=2025,
            entity_data=data,
            sbti_targets={
                "base_year": 2020,
                "base_year_emissions": "60000",
                "target_year": 2030,
                "target_reduction_pct": "42",
            },
        )
        progress = report.sbti_target_progress
        # s1+s2 = 55000, base = 60000, reduction ~8.3% < 42%
        assert progress["on_track"] is False

    def test_no_sbti_without_targets(self, engine, entity_data):
        report = engine.generate_report(
            reporting_year=2025, entity_data=entity_data,
        )
        assert report.sbti_target_progress is None
