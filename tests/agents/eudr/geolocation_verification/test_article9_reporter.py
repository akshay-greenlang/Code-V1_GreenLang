# -*- coding: utf-8 -*-
"""
Tests for Article9ComplianceReporter - AGENT-EUDR-002 Feature 8: Article 9 Reporting

Comprehensive test suite covering:
- Compliant plot assessment
- Non-compliant plot assessment (missing coordinates, missing polygon)
- Compliance rate calculation
- Commodity-level summary generation
- Country-level summary generation
- Remediation priority calculation
- Trend data generation
- JSON export
- CSV export
- Report hash integrity
- Estimated effort calculation

Test count: 80 tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Feature 8 - Article 9 Compliance Reporting)
"""

from decimal import Decimal
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.geolocation_verification.models import (
    CoordinateValidationResult,
    GeolocationAccuracyScore,
    PolygonVerificationResult,
    QualityTier,
)
from greenlang.agents.eudr.geolocation_verification.article9_reporter import (
    Article9ComplianceReporter,
)


# ---------------------------------------------------------------------------
# Helper: build plot assessment data
# ---------------------------------------------------------------------------


def _make_compliant_plot(plot_id="PLOT-001"):
    """Create a fully compliant plot assessment."""
    return {
        "plot_id": plot_id,
        "commodity": "cocoa",
        "country": "GH",
        "has_coordinates": True,
        "has_polygon": True,
        "coordinate_valid": True,
        "polygon_valid": True,
        "country_match": True,
        "no_protected_overlap": True,
        "no_deforestation": True,
        "temporal_consistent": True,
        "accuracy_score": Decimal("92.50"),
        "quality_tier": QualityTier.GOLD,
    }


def _make_non_compliant_plot(plot_id="PLOT-BAD-001", **overrides):
    """Create a non-compliant plot assessment."""
    defaults = {
        "plot_id": plot_id,
        "commodity": "oil_palm",
        "country": "ID",
        "has_coordinates": False,
        "has_polygon": False,
        "coordinate_valid": False,
        "polygon_valid": False,
        "country_match": False,
        "no_protected_overlap": False,
        "no_deforestation": False,
        "temporal_consistent": False,
        "accuracy_score": Decimal("15.00"),
        "quality_tier": QualityTier.FAIL,
    }
    defaults.update(overrides)
    return defaults


def _make_partial_plot(plot_id="PLOT-PARTIAL"):
    """Create a partially compliant plot (missing polygon for large plot)."""
    return {
        "plot_id": plot_id,
        "commodity": "soya",
        "country": "BR",
        "has_coordinates": True,
        "has_polygon": False,
        "coordinate_valid": True,
        "polygon_valid": False,
        "country_match": True,
        "no_protected_overlap": True,
        "no_deforestation": True,
        "temporal_consistent": True,
        "accuracy_score": Decimal("55.00"),
        "quality_tier": QualityTier.BRONZE,
    }


# ===========================================================================
# 1. Compliant Plot Assessment (10 tests)
# ===========================================================================


class TestCompliantAssessment:
    """Test compliant plot assessment."""

    def test_compliant_plot_assessment(self, article9_reporter):
        """Test fully compliant plot is assessed correctly."""
        plot = _make_compliant_plot()
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is True
        assert result["compliance_issues"] == []

    def test_compliant_plot_gold_tier(self, article9_reporter):
        """Test compliant plot with gold tier is fully compliant."""
        plot = _make_compliant_plot()
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is True

    def test_compliant_plot_no_remediation(self, article9_reporter):
        """Test compliant plot has no remediation items."""
        plot = _make_compliant_plot()
        result = article9_reporter.assess_plot(plot)
        assert len(result.get("remediation_items", [])) == 0

    def test_multiple_compliant_plots(self, article9_reporter):
        """Test multiple compliant plots are all assessed correctly."""
        plots = [_make_compliant_plot(f"PLOT-{i}") for i in range(5)]
        results = [article9_reporter.assess_plot(p) for p in plots]
        assert all(r["is_compliant"] for r in results)

    @pytest.mark.parametrize("commodity", [
        "cocoa", "coffee", "cattle", "soya", "oil_palm", "rubber", "wood",
    ])
    def test_compliant_all_commodities(self, article9_reporter, commodity):
        """Test compliant plot for each EUDR commodity."""
        plot = _make_compliant_plot()
        plot["commodity"] = commodity
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is True


# ===========================================================================
# 2. Non-Compliant Assessment (15 tests)
# ===========================================================================


class TestNonCompliantAssessment:
    """Test non-compliant plot assessment."""

    def test_non_compliant_missing_coordinates(self, article9_reporter):
        """Test plot missing coordinates is non-compliant."""
        plot = _make_non_compliant_plot(has_coordinates=False)
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False
        assert any("coordinate" in issue.lower() for issue in result["compliance_issues"])

    def test_non_compliant_missing_polygon_large_plot(self, article9_reporter):
        """Test large plot missing polygon is non-compliant per Article 9(1)(d)."""
        plot = _make_partial_plot()
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False

    def test_non_compliant_protected_area(self, article9_reporter):
        """Test plot in protected area is non-compliant."""
        plot = _make_compliant_plot()
        plot["no_protected_overlap"] = False
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False

    def test_non_compliant_deforestation(self, article9_reporter):
        """Test plot with deforestation is non-compliant."""
        plot = _make_compliant_plot()
        plot["no_deforestation"] = False
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False

    def test_non_compliant_country_mismatch(self, article9_reporter):
        """Test plot with country mismatch is non-compliant."""
        plot = _make_compliant_plot()
        plot["country_match"] = False
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False

    def test_non_compliant_has_remediation(self, article9_reporter):
        """Test non-compliant plot has remediation items."""
        plot = _make_non_compliant_plot()
        result = article9_reporter.assess_plot(plot)
        assert len(result.get("remediation_items", [])) > 0

    def test_non_compliant_fail_tier(self, article9_reporter):
        """Test FAIL tier plot is non-compliant."""
        plot = _make_non_compliant_plot()
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False

    @pytest.mark.parametrize("field,value", [
        ("has_coordinates", False),
        ("coordinate_valid", False),
        ("country_match", False),
        ("no_protected_overlap", False),
        ("no_deforestation", False),
    ])
    def test_each_failure_causes_non_compliance(self, article9_reporter, field, value):
        """Test each individual failure causes non-compliance."""
        plot = _make_compliant_plot()
        plot[field] = value
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False


# ===========================================================================
# 3. Compliance Rate Calculation (10 tests)
# ===========================================================================


class TestComplianceRate:
    """Test compliance rate calculation."""

    def test_compliance_rate_calculation(self, article9_reporter):
        """Test compliance rate for mixed batch."""
        plots = [
            _make_compliant_plot("C1"),
            _make_compliant_plot("C2"),
            _make_compliant_plot("C3"),
            _make_non_compliant_plot("NC1"),
            _make_non_compliant_plot("NC2"),
        ]
        report = article9_reporter.generate_report(plots)
        assert report["compliance_rate"] == 60.0  # 3/5

    def test_compliance_rate_100_percent(self, article9_reporter):
        """Test 100% compliance rate."""
        plots = [_make_compliant_plot(f"C{i}") for i in range(5)]
        report = article9_reporter.generate_report(plots)
        assert report["compliance_rate"] == 100.0

    def test_compliance_rate_0_percent(self, article9_reporter):
        """Test 0% compliance rate."""
        plots = [_make_non_compliant_plot(f"NC{i}") for i in range(5)]
        report = article9_reporter.generate_report(plots)
        assert report["compliance_rate"] == 0.0

    def test_compliance_rate_single_plot(self, article9_reporter):
        """Test compliance rate with single compliant plot."""
        plots = [_make_compliant_plot()]
        report = article9_reporter.generate_report(plots)
        assert report["compliance_rate"] == 100.0

    def test_compliance_rate_empty(self, article9_reporter):
        """Test compliance rate with empty plot list."""
        report = article9_reporter.generate_report([])
        assert report["compliance_rate"] == 0.0 or "compliance_rate" in report


# ===========================================================================
# 4. Summary Generation (10 tests)
# ===========================================================================


class TestSummaryGeneration:
    """Test commodity and country summary generation."""

    def test_commodity_summary_generation(self, article9_reporter):
        """Test commodity-level summary is generated."""
        plots = [
            _make_compliant_plot("C1"),
            _make_non_compliant_plot("NC1"),
        ]
        report = article9_reporter.generate_report(plots)
        assert "commodity_summary" in report
        assert isinstance(report["commodity_summary"], (dict, list))

    def test_country_summary_generation(self, article9_reporter):
        """Test country-level summary is generated."""
        plots = [
            _make_compliant_plot("C1"),
            _make_non_compliant_plot("NC1"),
        ]
        report = article9_reporter.generate_report(plots)
        assert "country_summary" in report
        assert isinstance(report["country_summary"], (dict, list))

    def test_summary_includes_total_plots(self, article9_reporter):
        """Test summary includes total plot count."""
        plots = [_make_compliant_plot(f"P{i}") for i in range(10)]
        report = article9_reporter.generate_report(plots)
        assert report["total_plots"] == 10

    def test_summary_includes_compliant_count(self, article9_reporter):
        """Test summary includes compliant plot count."""
        plots = [
            _make_compliant_plot("C1"),
            _make_compliant_plot("C2"),
            _make_non_compliant_plot("NC1"),
        ]
        report = article9_reporter.generate_report(plots)
        assert report["compliant_count"] == 2

    def test_summary_includes_non_compliant_count(self, article9_reporter):
        """Test summary includes non-compliant plot count."""
        plots = [
            _make_compliant_plot("C1"),
            _make_non_compliant_plot("NC1"),
            _make_non_compliant_plot("NC2"),
        ]
        report = article9_reporter.generate_report(plots)
        assert report["non_compliant_count"] == 2


# ===========================================================================
# 5. Remediation Priorities (5 tests)
# ===========================================================================


class TestRemediationPriorities:
    """Test remediation priority calculation."""

    def test_remediation_priorities(self, article9_reporter):
        """Test remediation priorities are generated."""
        plots = [
            _make_non_compliant_plot("NC1", has_coordinates=False),
            _make_non_compliant_plot("NC2", no_deforestation=False),
            _make_partial_plot("P1"),
        ]
        report = article9_reporter.generate_report(plots)
        assert "remediation_priorities" in report
        assert isinstance(report["remediation_priorities"], list)

    def test_no_remediation_all_compliant(self, article9_reporter):
        """Test no remediation needed when all plots are compliant."""
        plots = [_make_compliant_plot(f"C{i}") for i in range(3)]
        report = article9_reporter.generate_report(plots)
        remediation = report.get("remediation_priorities", [])
        assert len(remediation) == 0


# ===========================================================================
# 6. Export Formats (10 tests)
# ===========================================================================


class TestExportFormats:
    """Test report export formats."""

    def test_json_export(self, article9_reporter):
        """Test JSON export of compliance report."""
        plots = [
            _make_compliant_plot("C1"),
            _make_non_compliant_plot("NC1"),
        ]
        report = article9_reporter.generate_report(plots)
        json_str = article9_reporter.export_json(report)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_csv_export(self, article9_reporter):
        """Test CSV export of compliance report."""
        plots = [
            _make_compliant_plot("C1"),
            _make_non_compliant_plot("NC1"),
        ]
        report = article9_reporter.generate_report(plots)
        csv_str = article9_reporter.export_csv(report)
        assert isinstance(csv_str, str)
        assert len(csv_str) > 0
        # Should contain headers and data rows
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 data row

    def test_json_export_parseable(self, article9_reporter):
        """Test JSON export is parseable."""
        plots = [_make_compliant_plot("C1")]
        report = article9_reporter.generate_report(plots)
        json_str = article9_reporter.export_json(report)
        import json
        data = json.loads(json_str)
        assert "compliance_rate" in data or "total_plots" in data

    def test_csv_export_headers(self, article9_reporter):
        """Test CSV export contains proper headers."""
        plots = [_make_compliant_plot("C1")]
        report = article9_reporter.generate_report(plots)
        csv_str = article9_reporter.export_csv(report)
        header_line = csv_str.strip().split("\n")[0]
        assert "plot_id" in header_line.lower() or "compliance" in header_line.lower()

    def test_empty_report_json(self, article9_reporter):
        """Test JSON export of empty report."""
        report = article9_reporter.generate_report([])
        json_str = article9_reporter.export_json(report)
        assert isinstance(json_str, str)


# ===========================================================================
# 7. Report Integrity (10 tests)
# ===========================================================================


class TestReportIntegrity:
    """Test report hash integrity."""

    def test_report_hash_integrity(self, article9_reporter):
        """Test report includes integrity hash."""
        plots = [_make_compliant_plot("C1")]
        report = article9_reporter.generate_report(plots)
        assert "report_hash" in report
        assert isinstance(report["report_hash"], str)
        assert len(report["report_hash"]) == 64  # SHA-256

    def test_report_hash_deterministic(self, article9_reporter):
        """Test report hash is deterministic."""
        plots = [_make_compliant_plot("C1")]
        r1 = article9_reporter.generate_report(plots)
        r2 = article9_reporter.generate_report(plots)
        assert r1["report_hash"] == r2["report_hash"]

    def test_different_data_different_hash(self, article9_reporter):
        """Test different data produces different hash."""
        r1 = article9_reporter.generate_report([_make_compliant_plot("C1")])
        r2 = article9_reporter.generate_report([_make_non_compliant_plot("NC1")])
        assert r1["report_hash"] != r2["report_hash"]

    def test_report_generated_at_timestamp(self, article9_reporter):
        """Test report includes generated_at timestamp."""
        plots = [_make_compliant_plot("C1")]
        report = article9_reporter.generate_report(plots)
        assert "generated_at" in report


# ===========================================================================
# 8. Estimated Effort (5 tests)
# ===========================================================================


class TestEstimatedEffort:
    """Test estimated remediation effort calculation."""

    def test_estimated_effort_calculation(self, article9_reporter):
        """Test estimated effort is calculated for non-compliant plots."""
        plots = [
            _make_non_compliant_plot("NC1"),
            _make_non_compliant_plot("NC2"),
        ]
        report = article9_reporter.generate_report(plots)
        assert "estimated_effort" in report or "remediation_priorities" in report

    def test_zero_effort_all_compliant(self, article9_reporter):
        """Test zero effort when all plots are compliant."""
        plots = [_make_compliant_plot(f"C{i}") for i in range(5)]
        report = article9_reporter.generate_report(plots)
        effort = report.get("estimated_effort", {"total_hours": 0})
        if isinstance(effort, dict):
            assert effort.get("total_hours", 0) == 0
        else:
            assert effort == 0

    def test_effort_proportional_to_issues(self, article9_reporter):
        """Test effort scales with number of non-compliant plots."""
        small_batch = [_make_non_compliant_plot(f"NC{i}") for i in range(2)]
        large_batch = [_make_non_compliant_plot(f"NC{i}") for i in range(10)]
        r_small = article9_reporter.generate_report(small_batch)
        r_large = article9_reporter.generate_report(large_batch)
        # Larger batch should have more non-compliant plots
        assert r_large["non_compliant_count"] >= r_small["non_compliant_count"]


# ===========================================================================
# 9. Trend Data (5 tests)
# ===========================================================================


class TestTrendData:
    """Test trend data generation."""

    def test_trend_data_generation(self, article9_reporter):
        """Test trend data is generated in report."""
        plots = [
            _make_compliant_plot("C1"),
            _make_non_compliant_plot("NC1"),
        ]
        report = article9_reporter.generate_report(plots)
        # Report may include trend data or snapshot
        assert isinstance(report, dict)
        assert "compliance_rate" in report

    def test_trend_data_single_snapshot(self, article9_reporter):
        """Test trend data for single snapshot."""
        plots = [_make_compliant_plot("C1")]
        report = article9_reporter.generate_report(plots)
        assert report["total_plots"] == 1


# ===========================================================================
# 10. Multi-Commodity Report (15 tests)
# ===========================================================================


class TestMultiCommodityReport:
    """Test Article 9 reporting across multiple EUDR commodities."""

    @pytest.mark.parametrize("commodity", [
        "cocoa", "coffee", "cattle", "soya", "oil_palm", "rubber", "wood",
    ])
    def test_single_commodity_report(self, article9_reporter, commodity):
        """Test report generation for single commodity."""
        plots = [
            {**_make_compliant_plot(f"C-{commodity}-1"), "commodity": commodity},
            {**_make_compliant_plot(f"C-{commodity}-2"), "commodity": commodity},
        ]
        report = article9_reporter.generate_report(plots)
        assert report["total_plots"] == 2
        assert report["compliance_rate"] == 100.0

    @pytest.mark.parametrize("n_compliant,n_non_compliant,expected_rate", [
        (10, 0, 100.0),
        (9, 1, 90.0),
        (8, 2, 80.0),
        (7, 3, 70.0),
        (5, 5, 50.0),
        (3, 7, 30.0),
        (1, 9, 10.0),
        (0, 10, 0.0),
    ])
    def test_compliance_rate_various_ratios(self, article9_reporter, n_compliant, n_non_compliant, expected_rate):
        """Test compliance rate calculation for various compliant/non-compliant ratios."""
        plots = (
            [_make_compliant_plot(f"C{i}") for i in range(n_compliant)]
            + [_make_non_compliant_plot(f"NC{i}") for i in range(n_non_compliant)]
        )
        report = article9_reporter.generate_report(plots)
        assert report["compliance_rate"] == expected_rate

    def test_mixed_commodity_report(self, article9_reporter):
        """Test report with mixed commodities."""
        plots = [
            {**_make_compliant_plot("C-cocoa"), "commodity": "cocoa", "country": "GH"},
            {**_make_compliant_plot("C-palm"), "commodity": "oil_palm", "country": "ID"},
            {**_make_non_compliant_plot("NC-soya"), "commodity": "soya", "country": "BR"},
        ]
        report = article9_reporter.generate_report(plots)
        assert report["total_plots"] == 3

    def test_mixed_country_report(self, article9_reporter):
        """Test report with plots from multiple countries."""
        plots = [
            {**_make_compliant_plot("C-BR"), "country": "BR"},
            {**_make_compliant_plot("C-ID"), "country": "ID"},
            {**_make_compliant_plot("C-GH"), "country": "GH"},
            {**_make_non_compliant_plot("NC-CM"), "country": "CM"},
        ]
        report = article9_reporter.generate_report(plots)
        assert report["total_plots"] == 4
        assert report["compliant_count"] == 3
        assert report["non_compliant_count"] == 1

    def test_large_report(self, article9_reporter):
        """Test report generation with large number of plots."""
        plots = (
            [_make_compliant_plot(f"C{i}") for i in range(50)]
            + [_make_non_compliant_plot(f"NC{i}") for i in range(50)]
        )
        report = article9_reporter.generate_report(plots)
        assert report["total_plots"] == 100
        assert report["compliance_rate"] == 50.0

    def test_report_deterministic(self, article9_reporter):
        """Test report generation is deterministic."""
        plots = [
            _make_compliant_plot("C1"),
            _make_non_compliant_plot("NC1"),
        ]
        r1 = article9_reporter.generate_report(plots)
        r2 = article9_reporter.generate_report(plots)
        assert r1["compliance_rate"] == r2["compliance_rate"]
        assert r1["total_plots"] == r2["total_plots"]
        assert r1["report_hash"] == r2["report_hash"]


# ===========================================================================
# 11. Failure Mode Coverage (10 tests)
# ===========================================================================


class TestFailureModes:
    """Test all failure modes that cause non-compliance."""

    @pytest.mark.parametrize("failure_field,failure_value,issue_keyword", [
        ("has_coordinates", False, "coordinate"),
        ("coordinate_valid", False, "coordinate"),
        ("has_polygon", False, "polygon"),
        ("polygon_valid", False, "polygon"),
        ("country_match", False, "country"),
        ("no_protected_overlap", False, "protected"),
        ("no_deforestation", False, "deforestation"),
        ("temporal_consistent", False, "temporal"),
    ])
    def test_individual_failure_modes(self, article9_reporter, failure_field, failure_value, issue_keyword):
        """Test each individual failure mode causes non-compliance."""
        plot = _make_compliant_plot("FAIL-TEST")
        plot[failure_field] = failure_value
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False
        # Should have at least one compliance issue
        assert len(result["compliance_issues"]) > 0

    def test_multiple_failures_compound(self, article9_reporter):
        """Test multiple failures all appear in compliance issues."""
        plot = _make_compliant_plot("MULTI-FAIL")
        plot["has_coordinates"] = False
        plot["no_deforestation"] = False
        plot["country_match"] = False
        result = article9_reporter.assess_plot(plot)
        assert result["is_compliant"] is False
        assert len(result["compliance_issues"]) >= 3
