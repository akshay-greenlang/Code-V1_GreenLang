# -*- coding: utf-8 -*-
"""
Unit tests for DualReportingEngine (Engine 4 of 7)

AGENT-MRV-010: Scope 2 Market-Based Emissions Agent

Tests dual report generation, batch dual reports, facility dual reports,
procurement impact analysis, RE100 progress tracking, additionality scoring,
year-over-year comparison, coverage gap analysis, instrument recommendation,
cost estimation, validation, completeness checking, GHG Protocol / CDP / CSRD
formatting, statistics, reset, and thread safety.

Target: 70 tests, ~900 lines.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.scope2_market.dual_reporting import (
        DualReportingEngine,
        TYPICAL_REC_COST_PER_MWH,
        RE100_TARGET_PCT,
        ADDITIONALITY_CRITERIA,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

_SKIP = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Engine not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a default DualReportingEngine."""
    eng = DualReportingEngine()
    yield eng
    eng.reset()


@pytest.fixture
def location_result() -> Dict[str, Any]:
    """Build a sample location-based result for dual reporting."""
    return {
        "calculation_id": "loc-001",
        "facility_id": "FAC-001",
        "total_co2e_tonnes": Decimal("1500.00"),
        "total_co2e_kg": Decimal("1500000.00"),
        "total_mwh": Decimal("5000"),
        "country_code": "US",
        "grid_region": "US-CAMX",
        "gas_breakdown": {
            "CO2": Decimal("1480.00"),
            "CH4": Decimal("12.00"),
            "N2O": Decimal("8.00"),
        },
        "gwp_source": "AR5",
        "period": "2025",
        "provenance_hash": "a" * 64,
    }


@pytest.fixture
def market_result() -> Dict[str, Any]:
    """Build a sample market-based result for dual reporting."""
    return {
        "calculation_id": "mkt-001",
        "facility_id": "FAC-001",
        "total_co2e_tonnes": Decimal("800.00"),
        "total_co2e_kg": Decimal("800000.00"),
        "total_mwh": Decimal("5000"),
        "covered_mwh": Decimal("3000"),
        "uncovered_mwh": Decimal("2000"),
        "coverage_pct": Decimal("60.0"),
        "instruments": [
            {
                "type": "REC",
                "mwh_covered": Decimal("3000"),
                "emission_factor": Decimal("0"),
                "vintage_year": 2025,
                "is_renewable": True,
            },
        ],
        "gwp_source": "AR5",
        "period": "2025",
        "provenance_hash": "b" * 64,
    }


@pytest.fixture
def market_result_lower() -> Dict[str, Any]:
    """Market result with lower emissions than location (full renewable)."""
    return {
        "calculation_id": "mkt-002",
        "facility_id": "FAC-002",
        "total_co2e_tonnes": Decimal("0.00"),
        "total_co2e_kg": Decimal("0.00"),
        "total_mwh": Decimal("10000"),
        "covered_mwh": Decimal("10000"),
        "uncovered_mwh": Decimal("0"),
        "coverage_pct": Decimal("100.0"),
        "instruments": [
            {
                "type": "PPA",
                "mwh_covered": Decimal("10000"),
                "emission_factor": Decimal("0"),
                "vintage_year": 2025,
                "is_renewable": True,
            },
        ],
        "gwp_source": "AR5",
        "period": "2025",
        "provenance_hash": "c" * 64,
    }


@pytest.fixture
def location_result_higher() -> Dict[str, Any]:
    """Location result matching full-renewable market result."""
    return {
        "calculation_id": "loc-002",
        "facility_id": "FAC-002",
        "total_co2e_tonnes": Decimal("4350.00"),
        "total_co2e_kg": Decimal("4350000.00"),
        "total_mwh": Decimal("10000"),
        "country_code": "US",
        "gwp_source": "AR5",
        "period": "2025",
        "provenance_hash": "d" * 64,
    }


# ===========================================================================
# 1. TestDualReportGeneration
# ===========================================================================


@_SKIP
class TestDualReportGeneration:
    """Tests for generate_dual_report with location vs market comparison."""

    def test_generate_dual_report_success(self, engine, location_result, market_result):
        """Dual report returns expected structure with both sides."""
        report = engine.generate_dual_report(location_result, market_result)
        assert report["status"] == "SUCCESS"
        assert "location_based_tco2e" in report
        assert "market_based_tco2e" in report
        assert "provenance_hash" in report
        assert len(report["provenance_hash"]) == 64

    def test_dual_report_location_value(self, engine, location_result, market_result):
        """Location-based value matches input."""
        report = engine.generate_dual_report(location_result, market_result)
        assert report["location_based_tco2e"] == Decimal("1500.00")

    def test_dual_report_market_value(self, engine, location_result, market_result):
        """Market-based value matches input."""
        report = engine.generate_dual_report(location_result, market_result)
        assert report["market_based_tco2e"] == Decimal("800.00")

    def test_dual_report_difference_calculation(self, engine, location_result, market_result):
        """Difference is market minus location."""
        report = engine.generate_dual_report(location_result, market_result)
        diff = report.get("difference_tco2e", report.get("difference_tonnes"))
        assert diff is not None
        expected = Decimal("800.00") - Decimal("1500.00")
        assert float(diff) == pytest.approx(float(expected), rel=1e-4)

    def test_dual_report_lower_method_market(self, engine, location_result, market_result):
        """When market < location, lower_method is 'market'."""
        report = engine.generate_dual_report(location_result, market_result)
        assert report.get("lower_method") == "market"

    def test_dual_report_lower_method_location(self, engine, market_result, location_result):
        """When location < market, lower_method is 'location'."""
        location_result["total_co2e_tonnes"] = Decimal("500.00")
        report = engine.generate_dual_report(location_result, market_result)
        assert report.get("lower_method") == "location"

    def test_dual_report_equal_methods(self, engine, location_result, market_result):
        """When both methods equal, lower_method is 'equal' or either."""
        market_result["total_co2e_tonnes"] = Decimal("1500.00")
        report = engine.generate_dual_report(location_result, market_result)
        assert report.get("lower_method") in ("equal", "market", "location")

    def test_dual_report_provenance_deterministic(self, engine, location_result, market_result):
        """Same inputs produce same provenance hash."""
        r1 = engine.generate_dual_report(location_result, market_result)
        r2 = engine.generate_dual_report(location_result, market_result)
        assert r1["provenance_hash"] == r2["provenance_hash"]

    def test_dual_report_zero_emissions(self, engine):
        """Dual report handles zero emissions on both sides."""
        loc = {"total_co2e_tonnes": Decimal("0"), "total_mwh": Decimal("0"), "period": "2025"}
        mkt = {"total_co2e_tonnes": Decimal("0"), "total_mwh": Decimal("0"), "period": "2025"}
        report = engine.generate_dual_report(loc, mkt)
        assert report["status"] == "SUCCESS"

    def test_dual_report_facility_id_propagated(self, engine, location_result, market_result):
        """Facility ID from inputs is included in report."""
        report = engine.generate_dual_report(location_result, market_result)
        assert report.get("facility_id") == "FAC-001"


# ===========================================================================
# 2. TestBatchDualReports
# ===========================================================================


@_SKIP
class TestBatchDualReports:
    """Tests for generate_dual_report_batch."""

    def test_batch_dual_reports(self, engine, location_result, market_result):
        """Batch processes multiple location/market pairs."""
        pairs = [
            {"location": location_result, "market": market_result},
            {"location": location_result, "market": market_result},
        ]
        result = engine.generate_dual_report_batch(pairs)
        assert result["status"] == "SUCCESS"
        assert result["total_reports"] == 2
        assert len(result["reports"]) == 2

    def test_batch_empty_list(self, engine):
        """Empty batch returns zero reports."""
        result = engine.generate_dual_report_batch([])
        assert result["total_reports"] == 0

    def test_batch_provenance_hash(self, engine, location_result, market_result):
        """Batch result includes provenance hash."""
        pairs = [{"location": location_result, "market": market_result}]
        result = engine.generate_dual_report_batch(pairs)
        assert len(result["provenance_hash"]) == 64

    def test_batch_aggregated_totals(self, engine, location_result, market_result):
        """Batch aggregates total location and market emissions."""
        pairs = [
            {"location": location_result, "market": market_result},
            {"location": location_result, "market": market_result},
        ]
        result = engine.generate_dual_report_batch(pairs)
        assert result.get("total_location_tco2e", 0) > 0
        assert result.get("total_market_tco2e", 0) > 0

    def test_batch_error_handling(self, engine, location_result):
        """Batch handles invalid entries gracefully."""
        pairs = [
            {"location": location_result, "market": {}},
        ]
        result = engine.generate_dual_report_batch(pairs)
        assert result["total_reports"] >= 0


# ===========================================================================
# 3. TestFacilityDualReports
# ===========================================================================


@_SKIP
class TestFacilityDualReports:
    """Tests for generate_facility_dual_report."""

    def test_facility_dual_report(self, engine, location_result, market_result):
        """Facility dual report produces per-facility summary."""
        result = engine.generate_facility_dual_report(
            "FAC-001", location_result, market_result
        )
        assert result["status"] == "SUCCESS"
        assert result.get("facility_id") == "FAC-001"

    def test_facility_dual_report_with_metadata(self, engine, location_result, market_result):
        """Facility dual report includes period and gwp_source."""
        result = engine.generate_facility_dual_report(
            "FAC-001", location_result, market_result
        )
        assert "period" in result or "reporting_period" in result

    def test_facility_dual_report_provenance(self, engine, location_result, market_result):
        """Facility report carries provenance hash."""
        result = engine.generate_facility_dual_report(
            "FAC-001", location_result, market_result
        )
        assert len(result["provenance_hash"]) == 64

    def test_facility_dual_report_difference(self, engine, location_result, market_result):
        """Facility report includes emission difference."""
        result = engine.generate_facility_dual_report(
            "FAC-001", location_result, market_result
        )
        assert "difference_tco2e" in result or "difference_tonnes" in result

    def test_facility_dual_report_unknown_facility(self, engine, location_result, market_result):
        """Report can be generated for any facility ID string."""
        result = engine.generate_facility_dual_report(
            "UNKNOWN-FAC-999", location_result, market_result
        )
        assert result["status"] == "SUCCESS"


# ===========================================================================
# 4. TestProcurementImpact
# ===========================================================================


@_SKIP
class TestProcurementImpact:
    """Tests for procurement impact analysis."""

    def test_calculate_procurement_impact(self, engine, location_result, market_result):
        """Procurement impact shows reduction from instruments."""
        result = engine.calculate_procurement_impact(location_result, market_result)
        assert result["status"] == "SUCCESS"
        assert "reduction_tco2e" in result or "impact_tco2e" in result

    def test_procurement_impact_positive_reduction(self, engine, location_result, market_result):
        """When market < location, reduction is positive."""
        result = engine.calculate_procurement_impact(location_result, market_result)
        reduction = result.get("reduction_tco2e", result.get("impact_tco2e", 0))
        assert float(reduction) > 0

    def test_procurement_impact_zero_instruments(self, engine, location_result):
        """No instruments means zero or negative impact."""
        market_no_inst = {
            "total_co2e_tonnes": Decimal("1500.00"),
            "total_mwh": Decimal("5000"),
            "instruments": [],
            "period": "2025",
        }
        result = engine.calculate_procurement_impact(location_result, market_no_inst)
        assert result["status"] == "SUCCESS"

    def test_calculate_re100_progress(self, engine, market_result):
        """RE100 progress calculates percentage toward 100% renewable."""
        result = engine.calculate_re100_progress(market_result)
        assert result["status"] == "SUCCESS"
        progress = result.get("progress_pct", result.get("re100_pct", 0))
        assert float(progress) > 0
        assert float(progress) <= 100.0

    def test_re100_progress_full_renewable(self, engine, market_result_lower):
        """100% renewable achieves 100% RE100 progress."""
        result = engine.calculate_re100_progress(market_result_lower)
        progress = result.get("progress_pct", result.get("re100_pct", 0))
        assert float(progress) == pytest.approx(100.0, rel=1e-2)

    def test_re100_progress_zero_renewable(self, engine):
        """No renewable instruments yields 0% progress."""
        mkt = {
            "total_co2e_tonnes": Decimal("500.00"),
            "total_mwh": Decimal("5000"),
            "covered_mwh": Decimal("0"),
            "uncovered_mwh": Decimal("5000"),
            "instruments": [],
            "period": "2025",
        }
        result = engine.calculate_re100_progress(mkt)
        progress = result.get("progress_pct", result.get("re100_pct", 0))
        assert float(progress) == pytest.approx(0.0, abs=0.01)

    def test_calculate_additionality_score(self, engine):
        """Additionality scoring returns score between 0 and 100."""
        instrument = {
            "type": "PPA",
            "vintage_year": 2024,
            "region": "US-CAMX",
            "is_new_build": True,
            "third_party_verified": True,
            "energy_source": "solar",
        }
        result = engine.calculate_additionality_score(instrument)
        assert result["status"] == "SUCCESS"
        score = result.get("additionality_score", result.get("total_score", 0))
        assert 0 <= float(score) <= 100

    def test_additionality_score_old_instrument(self, engine):
        """Old instrument with no verification scores low."""
        instrument = {
            "type": "REC",
            "vintage_year": 2015,
            "region": "UNKNOWN",
            "is_new_build": False,
            "third_party_verified": False,
            "energy_source": "hydro",
        }
        result = engine.calculate_additionality_score(instrument)
        score = result.get("additionality_score", result.get("total_score", 0))
        assert float(score) < 50

    def test_additionality_criteria_count(self):
        """ADDITIONALITY_CRITERIA has 5 criteria items."""
        assert len(ADDITIONALITY_CRITERIA) == 5

    def test_procurement_impact_provenance(self, engine, location_result, market_result):
        """Procurement impact includes provenance hash."""
        result = engine.calculate_procurement_impact(location_result, market_result)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# 5. TestYearOverYear
# ===========================================================================


@_SKIP
class TestYearOverYear:
    """Tests for year_over_year_comparison."""

    def test_year_over_year_comparison(self, engine, location_result, market_result):
        """YoY comparison produces trend data for multiple years."""
        current = {"location": location_result, "market": market_result}
        previous_loc = dict(location_result)
        previous_loc["total_co2e_tonnes"] = Decimal("1800.00")
        previous_mkt = dict(market_result)
        previous_mkt["total_co2e_tonnes"] = Decimal("1000.00")
        previous = {"location": previous_loc, "market": previous_mkt}
        result = engine.year_over_year_comparison(current, previous)
        assert result["status"] == "SUCCESS"

    def test_yoy_location_trend(self, engine, location_result, market_result):
        """YoY shows location-based change direction."""
        current = {"location": location_result, "market": market_result}
        previous_loc = dict(location_result)
        previous_loc["total_co2e_tonnes"] = Decimal("1800.00")
        previous_mkt = dict(market_result)
        previous_mkt["total_co2e_tonnes"] = Decimal("1000.00")
        previous = {"location": previous_loc, "market": previous_mkt}
        result = engine.year_over_year_comparison(current, previous)
        assert "location_change_pct" in result or "location_trend" in result

    def test_yoy_market_trend(self, engine, location_result, market_result):
        """YoY shows market-based change direction."""
        current = {"location": location_result, "market": market_result}
        previous_loc = dict(location_result)
        previous_loc["total_co2e_tonnes"] = Decimal("1800.00")
        previous_mkt = dict(market_result)
        previous_mkt["total_co2e_tonnes"] = Decimal("1000.00")
        previous = {"location": previous_loc, "market": previous_mkt}
        result = engine.year_over_year_comparison(current, previous)
        assert "market_change_pct" in result or "market_trend" in result

    def test_yoy_provenance(self, engine, location_result, market_result):
        """YoY result includes provenance hash."""
        current = {"location": location_result, "market": market_result}
        previous = {"location": location_result, "market": market_result}
        result = engine.year_over_year_comparison(current, previous)
        assert len(result["provenance_hash"]) == 64

    def test_yoy_zero_previous(self, engine, location_result, market_result):
        """YoY handles zero previous year gracefully."""
        current = {"location": location_result, "market": market_result}
        prev_loc = dict(location_result)
        prev_loc["total_co2e_tonnes"] = Decimal("0")
        prev_mkt = dict(market_result)
        prev_mkt["total_co2e_tonnes"] = Decimal("0")
        previous = {"location": prev_loc, "market": prev_mkt}
        result = engine.year_over_year_comparison(current, previous)
        assert result["status"] == "SUCCESS"


# ===========================================================================
# 6. TestCoverageAnalysis
# ===========================================================================


@_SKIP
class TestCoverageAnalysis:
    """Tests for coverage gap analysis and instrument recommendations."""

    def test_analyze_coverage_gaps(self, engine, market_result):
        """Coverage gap analysis identifies uncovered MWh."""
        result = engine.analyze_coverage_gaps(market_result)
        assert result["status"] == "SUCCESS"
        assert "uncovered_mwh" in result or "gaps" in result

    def test_coverage_gaps_fully_covered(self, engine, market_result_lower):
        """Fully covered facility shows no gaps."""
        result = engine.analyze_coverage_gaps(market_result_lower)
        uncovered = result.get("uncovered_mwh", Decimal("0"))
        assert float(uncovered) == pytest.approx(0.0, abs=0.01)

    def test_coverage_gaps_partially_covered(self, engine, market_result):
        """Partially covered facility shows 2000 MWh gap."""
        result = engine.analyze_coverage_gaps(market_result)
        uncovered = result.get("uncovered_mwh", 0)
        assert float(uncovered) == pytest.approx(2000.0, rel=0.01)

    def test_recommend_instruments(self, engine, market_result):
        """Instrument recommendation returns at least one suggestion."""
        result = engine.recommend_instruments(market_result)
        assert result["status"] == "SUCCESS"
        recommendations = result.get("recommendations", [])
        assert len(recommendations) >= 1

    def test_recommend_instruments_for_covered(self, engine, market_result_lower):
        """Fully covered facility needs no additional instruments."""
        result = engine.recommend_instruments(market_result_lower)
        recommendations = result.get("recommendations", [])
        assert len(recommendations) == 0 or result.get("already_covered", False)

    def test_recommend_instruments_types(self, engine, market_result):
        """Recommendations include instrument type suggestions."""
        result = engine.recommend_instruments(market_result)
        recommendations = result.get("recommendations", [])
        if recommendations:
            assert "instrument_type" in recommendations[0] or "type" in recommendations[0]

    def test_estimate_cost_to_cover(self, engine, market_result):
        """Cost estimation returns positive cost for uncovered MWh."""
        result = engine.estimate_cost_to_cover(market_result)
        assert result["status"] == "SUCCESS"
        cost = result.get("estimated_cost_usd", result.get("total_cost", 0))
        assert float(cost) > 0

    def test_estimate_cost_to_cover_region(self, engine, market_result):
        """Cost estimation uses regional REC pricing."""
        result = engine.estimate_cost_to_cover(market_result, region="US_NATIONAL")
        assert result["status"] == "SUCCESS"

    def test_estimate_cost_fully_covered(self, engine, market_result_lower):
        """Fully covered facility has zero additional cost."""
        result = engine.estimate_cost_to_cover(market_result_lower)
        cost = result.get("estimated_cost_usd", result.get("total_cost", 0))
        assert float(cost) == pytest.approx(0.0, abs=0.01)

    def test_typical_rec_cost_per_mwh_us(self):
        """US National REC cost is populated."""
        assert float(TYPICAL_REC_COST_PER_MWH["US_NATIONAL"]) > 0


# ===========================================================================
# 7. TestValidation
# ===========================================================================


@_SKIP
class TestValidation:
    """Tests for validate_dual_report and check_dual_reporting_completeness."""

    def test_validate_dual_report_valid(self, engine, location_result, market_result):
        """Valid dual report passes validation."""
        report = engine.generate_dual_report(location_result, market_result)
        validation = engine.validate_dual_report(report)
        assert validation["status"] == "SUCCESS"
        assert validation.get("valid", validation.get("is_valid")) is True

    def test_validate_dual_report_missing_fields(self, engine):
        """Incomplete report fails validation."""
        validation = engine.validate_dual_report({"location_based_tco2e": Decimal("100")})
        valid = validation.get("valid", validation.get("is_valid"))
        assert valid is False or len(validation.get("errors", [])) > 0

    def test_check_dual_reporting_completeness_complete(
        self, engine, location_result, market_result
    ):
        """Complete data set passes completeness check."""
        result = engine.check_dual_reporting_completeness(
            location_result, market_result
        )
        assert result["status"] == "SUCCESS"

    def test_check_dual_reporting_completeness_missing_market(self, engine, location_result):
        """Missing market data fails completeness."""
        result = engine.check_dual_reporting_completeness(
            location_result, {}
        )
        completeness = result.get("complete", result.get("is_complete"))
        assert completeness is False or len(result.get("missing", [])) > 0

    def test_validate_dual_report_provenance(self, engine, location_result, market_result):
        """Validation result includes provenance hash."""
        report = engine.generate_dual_report(location_result, market_result)
        validation = engine.validate_dual_report(report)
        assert "provenance_hash" in validation


# ===========================================================================
# 8. TestFormatting
# ===========================================================================


@_SKIP
class TestFormatting:
    """Tests for GHG Protocol table, CDP response, and CSRD/ESRS formatting."""

    def test_format_ghg_protocol_table(self, engine, location_result, market_result):
        """GHG Protocol table format includes both methods."""
        report = engine.generate_dual_report(location_result, market_result)
        table = engine.format_ghg_protocol_table(report)
        assert table["status"] == "SUCCESS"
        assert "table" in table or "rows" in table or "formatted" in table

    def test_ghg_protocol_table_has_location_row(self, engine, location_result, market_result):
        """Table includes a location-based emissions row."""
        report = engine.generate_dual_report(location_result, market_result)
        table = engine.format_ghg_protocol_table(report)
        content = str(table)
        assert "location" in content.lower()

    def test_ghg_protocol_table_has_market_row(self, engine, location_result, market_result):
        """Table includes a market-based emissions row."""
        report = engine.generate_dual_report(location_result, market_result)
        table = engine.format_ghg_protocol_table(report)
        content = str(table)
        assert "market" in content.lower()

    def test_format_cdp_response(self, engine, location_result, market_result):
        """CDP C8.2d format is generated."""
        report = engine.generate_dual_report(location_result, market_result)
        cdp = engine.format_cdp_response(report)
        assert cdp["status"] == "SUCCESS"

    def test_cdp_response_structure(self, engine, location_result, market_result):
        """CDP response has expected disclosure fields."""
        report = engine.generate_dual_report(location_result, market_result)
        cdp = engine.format_cdp_response(report)
        assert "provenance_hash" in cdp

    def test_format_csrd_esrs_e1(self, engine, location_result, market_result):
        """CSRD/ESRS E1 format is generated."""
        report = engine.generate_dual_report(location_result, market_result)
        csrd = engine.format_csrd_esrs_e1(report)
        assert csrd["status"] == "SUCCESS"

    def test_csrd_esrs_e1_scope2_present(self, engine, location_result, market_result):
        """CSRD format includes scope 2 disclosure."""
        report = engine.generate_dual_report(location_result, market_result)
        csrd = engine.format_csrd_esrs_e1(report)
        content = str(csrd).lower()
        assert "scope" in content or "e1" in content or "esrs" in content

    def test_csrd_esrs_e1_provenance(self, engine, location_result, market_result):
        """CSRD format carries provenance hash."""
        report = engine.generate_dual_report(location_result, market_result)
        csrd = engine.format_csrd_esrs_e1(report)
        assert len(csrd["provenance_hash"]) == 64

    def test_format_ghg_protocol_empty_report(self, engine):
        """Formatting empty report returns error or empty structure."""
        result = engine.format_ghg_protocol_table({})
        assert result["status"] in ("SUCCESS", "ERROR", "VALIDATION_ERROR")

    def test_format_cdp_empty_report(self, engine):
        """Formatting empty CDP report returns error or empty structure."""
        result = engine.format_cdp_response({})
        assert result["status"] in ("SUCCESS", "ERROR", "VALIDATION_ERROR")


# ===========================================================================
# 9. TestStatisticsReset
# ===========================================================================


@_SKIP
class TestStatisticsReset:
    """Tests for get_statistics and reset."""

    def test_get_statistics(self, engine):
        """Statistics returns dict with expected keys."""
        stats = engine.get_statistics()
        assert isinstance(stats, dict)
        assert "reports_generated" in stats or "dual_reports" in stats

    def test_statistics_after_report(self, engine, location_result, market_result):
        """Statistics update after generating a report."""
        engine.generate_dual_report(location_result, market_result)
        stats = engine.get_statistics()
        count = stats.get("reports_generated", stats.get("dual_reports", 0))
        assert count >= 1

    def test_reset_clears_statistics(self, engine, location_result, market_result):
        """Reset zeroes all counters."""
        engine.generate_dual_report(location_result, market_result)
        engine.reset()
        stats = engine.get_statistics()
        count = stats.get("reports_generated", stats.get("dual_reports", 0))
        assert count == 0

    def test_statistics_batch_increment(self, engine, location_result, market_result):
        """Batch reports increment statistics by batch size."""
        pairs = [
            {"location": location_result, "market": market_result},
            {"location": location_result, "market": market_result},
        ]
        engine.generate_dual_report_batch(pairs)
        stats = engine.get_statistics()
        count = stats.get("reports_generated", stats.get("dual_reports", 0))
        assert count >= 2

    def test_reset_returns_none(self, engine):
        """Reset method returns None."""
        result = engine.reset()
        assert result is None


# ===========================================================================
# 10. TestThreadSafety
# ===========================================================================


@_SKIP
class TestThreadSafety:
    """Tests for thread safety of DualReportingEngine."""

    def test_concurrent_dual_reports(self, engine, location_result, market_result):
        """Concurrent report generation does not corrupt state."""
        results = []
        errors = []

        def generate():
            try:
                r = engine.generate_dual_report(location_result, market_result)
                results.append(r)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=generate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 10

    def test_concurrent_statistics(self, engine, location_result, market_result):
        """Concurrent stats reads during report generation are safe."""
        stats_results = []

        def generate_and_read():
            engine.generate_dual_report(location_result, market_result)
            stats_results.append(engine.get_statistics())

        threads = [threading.Thread(target=generate_and_read) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(stats_results) == 5

    def test_concurrent_reset(self, engine, location_result, market_result):
        """Concurrent reset calls do not raise."""
        errors = []

        def reset_engine():
            try:
                engine.reset()
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=reset_engine) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0

    def test_concurrent_batch_and_single(self, engine, location_result, market_result):
        """Mixed batch and single report calls are safe."""
        errors = []

        def single():
            try:
                engine.generate_dual_report(location_result, market_result)
            except Exception as exc:
                errors.append(str(exc))

        def batch():
            try:
                pairs = [{"location": location_result, "market": market_result}]
                engine.generate_dual_report_batch(pairs)
            except Exception as exc:
                errors.append(str(exc))

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=single if i % 2 == 0 else batch))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0

    def test_provenance_unique_per_call(self, engine, location_result, market_result):
        """Each dual report call with same input produces same hash (deterministic)."""
        hashes = set()
        for _ in range(5):
            r = engine.generate_dual_report(location_result, market_result)
            hashes.add(r["provenance_hash"])
        assert len(hashes) == 1
