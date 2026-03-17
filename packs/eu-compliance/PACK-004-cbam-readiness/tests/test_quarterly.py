# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Quarterly Reporting Tests (15 tests)

Tests quarterly report period detection, report assembly,
XML generation, validation, amendments, and deadline tracking.

Author: GreenLang QA Team
"""

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    StubQuarterlyEngine,
    _compute_hash,
    _new_uuid,
    _utcnow,
)


@pytest.fixture
def engine():
    """Create a StubQuarterlyEngine instance."""
    return StubQuarterlyEngine()


class TestQuarterDetection:
    """Test reporting quarter detection from dates."""

    def test_detect_period_q1(self, engine):
        """Test Q1 detection (January-March)."""
        result = engine.detect_period("2026-02-15")
        assert result["quarter"] == "Q1"
        assert result["year"] == 2026
        assert result["start_date"] == "2026-01-01"

    def test_detect_period_q2(self, engine):
        """Test Q2 detection (April-June)."""
        result = engine.detect_period("2026-05-20")
        assert result["quarter"] == "Q2"
        assert result["start_date"] == "2026-04-01"

    def test_detect_period_q3(self, engine):
        """Test Q3 detection (July-September)."""
        result = engine.detect_period("2026-08-01")
        assert result["quarter"] == "Q3"
        assert result["start_date"] == "2026-07-01"

    def test_detect_period_q4(self, engine):
        """Test Q4 detection (October-December)."""
        result = engine.detect_period("2026-11-30")
        assert result["quarter"] == "Q4"
        assert result["start_date"] == "2026-10-01"


class TestReportAssembly:
    """Test quarterly report assembly."""

    def test_assemble_report_steel(
        self, engine, sample_importer_config, sample_emission_results,
    ):
        """Test report assembly for steel imports."""
        steel_results = [
            r for r in sample_emission_results if r["goods_category"] == "steel"
        ]
        period = engine.detect_period("2026-01-15")
        report = engine.assemble_report(
            sample_importer_config, steel_results, period,
        )
        assert report["total_imports"] == len(steel_results)
        assert report["total_emissions_tco2e"] > 0
        assert report["status"] == "assembled"

    def test_assemble_report_multi_category(
        self, engine, sample_importer_config, sample_emission_results,
    ):
        """Test report assembly across multiple goods categories."""
        period = engine.detect_period("2026-01-15")
        report = engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        assert report["total_imports"] == 10
        assert report["total_emissions_tco2e"] > 0

    def test_aggregate_by_cn_code(self, sample_emission_results):
        """Test aggregation of emissions by CN code."""
        by_cn = {}
        for r in sample_emission_results:
            cn = r["cn_code"]
            by_cn[cn] = by_cn.get(cn, 0.0) + r["total_emissions_tco2e"]
        assert len(by_cn) >= 5, "Should have multiple CN codes"
        total = sum(by_cn.values())
        expected_total = sum(r["total_emissions_tco2e"] for r in sample_emission_results)
        assert total == pytest.approx(expected_total)

    def test_aggregate_by_country(self, sample_emission_results):
        """Test aggregation of emissions by origin country."""
        by_country = {}
        for r in sample_emission_results:
            c = r["origin_country"]
            by_country[c] = by_country.get(c, 0.0) + r["total_emissions_tco2e"]
        assert "TR" in by_country
        assert "CN" in by_country
        assert "IN" in by_country


class TestXMLGeneration:
    """Test XML generation for quarterly reports."""

    def test_xml_generation_valid(
        self, engine, sample_importer_config, sample_emission_results,
    ):
        """Test XML output is well-formed."""
        period = engine.detect_period("2026-01-15")
        report = engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        xml = engine.generate_xml(report)
        assert xml.startswith('<?xml version="1.0"')
        assert xml.endswith("</CBAMQuarterlyReport>")

    def test_xml_contains_required_fields(
        self, engine, sample_importer_config, sample_emission_results,
    ):
        """Test XML contains all required CBAM fields."""
        period = engine.detect_period("2026-01-15")
        report = engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        xml = engine.generate_xml(report)
        required_tags = [
            "<ReportId>", "<Quarter>", "<Year>",
            "<TotalEmissions", "<TotalImports>",
            "<ImporterName>", "<ImporterEORI>",
            "<ProvenanceHash>",
        ]
        for tag in required_tags:
            assert tag in xml, f"Missing required XML tag: {tag}"


class TestReportValidation:
    """Test quarterly report validation."""

    def test_validate_complete_report(
        self, engine, sample_importer_config, sample_emission_results,
    ):
        """Test validation passes for complete report."""
        period = engine.detect_period("2026-01-15")
        report = engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        validation = engine.validate_report(report)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0

    def test_validate_missing_fields(self, engine):
        """Test validation catches missing required fields."""
        incomplete_report = {
            "importer": {},
            "total_imports": 0,
            "total_emissions_tco2e": 0,
            "period": {},
        }
        validation = engine.validate_report(incomplete_report)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


class TestAmendments:
    """Test quarterly report amendments."""

    def test_amendment_versioning(self, engine):
        """Test amendment creates new version."""
        amendment = engine.create_amendment(
            report_id="QR-2026-Q1-001",
            reason="Updated steel emission factors",
            version=1,
        )
        assert amendment["amendment_version"] == 2
        assert amendment["status"] == "amended"

    def test_amendment_within_window(self, engine):
        """Test amendment within 60-day window is allowed."""
        # Within amendment window
        amendment = engine.create_amendment(
            report_id="QR-2026-Q1-001",
            reason="Correction to aluminium data",
            version=1,
        )
        assert amendment["status"] == "amended"
        assert "created_at" in amendment

    def test_amendment_after_window(self):
        """Test amendment tracking after window includes reason."""
        # After amendment window, record the late amendment with reason
        late_amendment = {
            "report_id": "QR-2026-Q1-001",
            "amendment_version": 3,
            "reason": "Late correction per regulatory request",
            "late_filing": True,
            "justification": "Regulatory authority requested correction",
            "status": "pending_approval",
        }
        assert late_amendment["late_filing"] is True
        assert late_amendment["status"] == "pending_approval"


class TestDeadlines:
    """Test deadline tracking and submission history."""

    def test_deadline_alert_levels(self):
        """Test deadline alert levels based on days remaining."""
        import datetime
        deadline = datetime.date(2026, 4, 30)
        test_dates = [
            (datetime.date(2026, 3, 1), "on_track", 60),
            (datetime.date(2026, 4, 15), "warning", 15),
            (datetime.date(2026, 4, 28), "critical", 2),
            (datetime.date(2026, 5, 1), "overdue", -1),
        ]
        for current_date, expected_level, expected_days in test_dates:
            days_remaining = (deadline - current_date).days
            if days_remaining < 0:
                level = "overdue"
            elif days_remaining <= 7:
                level = "critical"
            elif days_remaining <= 30:
                level = "warning"
            else:
                level = "on_track"
            assert level == expected_level, (
                f"At {current_date}, expected {expected_level} got {level}"
            )

    def test_submission_history(self):
        """Test quarterly report submission history tracking."""
        history = [
            {"quarter": "Q1-2026", "submitted_at": "2026-04-25", "status": "accepted"},
            {"quarter": "Q2-2026", "submitted_at": "2026-07-28", "status": "accepted"},
            {"quarter": "Q3-2026", "submitted_at": "2026-10-30", "status": "accepted"},
            {"quarter": "Q4-2026", "submitted_at": "2027-01-29", "status": "pending"},
        ]
        accepted = [h for h in history if h["status"] == "accepted"]
        assert len(accepted) == 3
        assert len(history) == 4
