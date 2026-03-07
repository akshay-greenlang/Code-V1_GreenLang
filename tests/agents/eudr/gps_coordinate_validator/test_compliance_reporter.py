# -*- coding: utf-8 -*-
"""
Tests for ComplianceReporter - AGENT-EUDR-007 Engine 7: Compliance Reporting

Comprehensive test suite covering:
- Compliance certificate generation for COMPLIANT coordinates
- Compliance certificate generation for NON_COMPLIANT coordinates
- Compliance certificate for NEEDS_REVIEW (marginal) coordinates
- Batch summary statistics (correct totals)
- Batch summary error type breakdown
- Remediation guidance generation
- Remediation priority sorting
- Export to JSON format
- Export to CSV format
- Export to EUDR-compliant XML format
- Audit trail provenance chain
- Submission readiness percentage
- Batch multi-format report
- Parametrized tests for report formats

Test count: 45+ tests
Coverage target: >= 85% of ComplianceReporter module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import json

import pytest

from greenlang.agents.eudr.gps_coordinate_validator.models import (
    PrecisionLevel,
    PrecisionResult,
    SourceType,
    ValidationResult,
    ValidationError as VError,
    ValidationErrorType,
    ValidationSeverity,
)
from tests.agents.eudr.gps_coordinate_validator.conftest import (
    COCOA_FARM_GHANA,
    PALM_PLANTATION_INDONESIA,
    COFFEE_FARM_COLOMBIA,
    SOYA_FIELD_BRAZIL,
    RUBBER_FARM_THAILAND,
    CATTLE_RANCH_BRAZIL,
    TIMBER_FOREST_CONGO,
    OCEAN_POINT,
    LOW_PRECISION,
    NULL_ISLAND,
    HIGH_PRECISION,
    SHA256_HEX_LENGTH,
)


# ---------------------------------------------------------------------------
# Helper: build assessment result dicts for compliance reporting
# ---------------------------------------------------------------------------


def _compliant_assessment() -> dict:
    """Build a compliant assessment result for reporting tests."""
    return {
        "coordinate_id": "COORD-001",
        "latitude": COCOA_FARM_GHANA[0],
        "longitude": COCOA_FARM_GHANA[1],
        "total_score": 92.5,
        "tier": "gold",
        "is_on_land": True,
        "country_match": True,
        "eudr_adequate": True,
        "errors": [],
        "warnings": [],
        "commodity": "cocoa",
        "country": "GH",
    }


def _non_compliant_assessment() -> dict:
    """Build a non-compliant assessment result for reporting tests."""
    return {
        "coordinate_id": "COORD-002",
        "latitude": OCEAN_POINT[0],
        "longitude": OCEAN_POINT[1],
        "total_score": 25.0,
        "tier": "unverified",
        "is_on_land": False,
        "country_match": False,
        "eudr_adequate": False,
        "errors": [
            {"type": "on_water", "message": "Coordinate is in the ocean"},
            {"type": "low_precision", "message": "Insufficient decimal places"},
        ],
        "warnings": [],
        "commodity": "soya",
        "country": "BR",
    }


def _needs_review_assessment() -> dict:
    """Build a marginal assessment result requiring review."""
    return {
        "coordinate_id": "COORD-003",
        "latitude": LOW_PRECISION[0],
        "longitude": LOW_PRECISION[1],
        "total_score": 55.0,
        "tier": "bronze",
        "is_on_land": True,
        "country_match": True,
        "eudr_adequate": False,
        "errors": [],
        "warnings": [
            {"type": "low_precision", "message": "Only 1 decimal place"},
        ],
        "commodity": "cocoa",
        "country": "GH",
    }


def _batch_assessments() -> list:
    """Build a batch of mixed assessment results."""
    return [
        _compliant_assessment(),
        _non_compliant_assessment(),
        _needs_review_assessment(),
        {
            "coordinate_id": "COORD-004",
            "latitude": PALM_PLANTATION_INDONESIA[0],
            "longitude": PALM_PLANTATION_INDONESIA[1],
            "total_score": 88.0,
            "tier": "silver",
            "is_on_land": True,
            "country_match": True,
            "eudr_adequate": True,
            "errors": [],
            "warnings": [
                {"type": "near_protected", "message": "Near a protected area"},
            ],
            "commodity": "oil_palm",
            "country": "ID",
        },
        {
            "coordinate_id": "COORD-005",
            "latitude": SOYA_FIELD_BRAZIL[0],
            "longitude": SOYA_FIELD_BRAZIL[1],
            "total_score": 95.0,
            "tier": "gold",
            "is_on_land": True,
            "country_match": True,
            "eudr_adequate": True,
            "errors": [],
            "warnings": [],
            "commodity": "soya",
            "country": "BR",
        },
    ]


# ===========================================================================
# 1. Compliance Certificate Generation
# ===========================================================================


class TestComplianceCertificate:
    """Test compliance certificate generation for individual coordinates."""

    def test_compliance_cert_compliant(self, compliance_reporter):
        """Good coordinate generates COMPLIANT certificate."""
        cert = compliance_reporter.generate_certificate(_compliant_assessment())
        assert cert.status == "COMPLIANT"
        assert cert.coordinate_id == "COORD-001"
        assert cert.total_score >= 90.0

    def test_compliance_cert_non_compliant(self, compliance_reporter):
        """Bad coordinate generates NON_COMPLIANT certificate."""
        cert = compliance_reporter.generate_certificate(_non_compliant_assessment())
        assert cert.status == "NON_COMPLIANT"
        assert cert.coordinate_id == "COORD-002"
        assert cert.total_score < 50.0

    def test_compliance_cert_needs_review(self, compliance_reporter):
        """Marginal coordinate generates NEEDS_REVIEW certificate."""
        cert = compliance_reporter.generate_certificate(_needs_review_assessment())
        assert cert.status == "NEEDS_REVIEW"
        assert cert.coordinate_id == "COORD-003"

    def test_compliance_cert_has_id(self, compliance_reporter):
        """Certificate has a unique identifier."""
        cert = compliance_reporter.generate_certificate(_compliant_assessment())
        assert cert.certificate_id is not None
        assert len(cert.certificate_id) > 0

    def test_compliance_cert_has_timestamp(self, compliance_reporter):
        """Certificate has a generation timestamp."""
        cert = compliance_reporter.generate_certificate(_compliant_assessment())
        assert cert.generated_at is not None


# ===========================================================================
# 2. Batch Summary Statistics
# ===========================================================================


class TestBatchSummary:
    """Test batch summary statistics generation."""

    def test_batch_summary_stats(self, compliance_reporter):
        """Batch summary has correct total counts."""
        assessments = _batch_assessments()
        summary = compliance_reporter.generate_batch_summary(assessments)
        assert summary.total_count == 5
        assert summary.compliant_count >= 1
        assert summary.non_compliant_count >= 1
        assert summary.needs_review_count >= 0
        assert (
            summary.compliant_count
            + summary.non_compliant_count
            + summary.needs_review_count
        ) == summary.total_count

    def test_batch_summary_breakdown(self, compliance_reporter):
        """Batch summary includes error type breakdown."""
        assessments = _batch_assessments()
        summary = compliance_reporter.generate_batch_summary(assessments)
        assert hasattr(summary, "error_type_breakdown")
        assert isinstance(summary.error_type_breakdown, dict)

    def test_batch_summary_average_score(self, compliance_reporter):
        """Batch summary includes average accuracy score."""
        assessments = _batch_assessments()
        summary = compliance_reporter.generate_batch_summary(assessments)
        assert hasattr(summary, "average_score")
        assert 0.0 <= summary.average_score <= 100.0

    def test_batch_summary_empty(self, compliance_reporter):
        """Empty batch produces zero-count summary."""
        summary = compliance_reporter.generate_batch_summary([])
        assert summary.total_count == 0


# ===========================================================================
# 3. Remediation Guidance
# ===========================================================================


class TestRemediationGuidance:
    """Test remediation guidance generation."""

    def test_remediation_guidance_non_compliant(self, compliance_reporter):
        """Non-compliant coordinate generates remediation instructions."""
        guidance = compliance_reporter.generate_remediation(
            _non_compliant_assessment()
        )
        assert isinstance(guidance, list)
        assert len(guidance) >= 1
        for item in guidance:
            assert "instruction" in item or "action" in item

    def test_remediation_guidance_compliant(self, compliance_reporter):
        """Compliant coordinate has no remediation needed."""
        guidance = compliance_reporter.generate_remediation(
            _compliant_assessment()
        )
        assert isinstance(guidance, list)
        assert len(guidance) == 0

    def test_remediation_priority_sorted(self, compliance_reporter):
        """Remediation items are sorted by priority (highest first)."""
        assessment = _non_compliant_assessment()
        guidance = compliance_reporter.generate_remediation(assessment)
        if len(guidance) >= 2:
            priorities = [g.get("priority", 0) for g in guidance]
            assert priorities == sorted(priorities, reverse=True)


# ===========================================================================
# 4. Export Formats
# ===========================================================================


class TestExportJSON:
    """Test JSON export format."""

    def test_export_json_valid(self, compliance_reporter):
        """JSON export produces valid JSON."""
        assessments = _batch_assessments()
        json_str = compliance_reporter.export_json(assessments)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, (dict, list))

    def test_export_json_contains_coordinates(self, compliance_reporter):
        """JSON export contains coordinate data."""
        assessments = [_compliant_assessment()]
        json_str = compliance_reporter.export_json(assessments)
        parsed = json.loads(json_str)
        # Should contain the coordinate data somewhere in the structure
        json_text = json.dumps(parsed)
        assert "COORD-001" in json_text


class TestExportCSV:
    """Test CSV export format."""

    def test_export_csv_has_headers(self, compliance_reporter):
        """CSV export includes header row."""
        assessments = _batch_assessments()
        csv_str = compliance_reporter.export_csv(assessments)
        assert isinstance(csv_str, str)
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 data row
        header = lines[0]
        assert "coordinate_id" in header.lower() or "latitude" in header.lower()

    def test_export_csv_correct_row_count(self, compliance_reporter):
        """CSV export has correct number of data rows."""
        assessments = _batch_assessments()
        csv_str = compliance_reporter.export_csv(assessments)
        lines = csv_str.strip().split("\n")
        assert len(lines) == len(assessments) + 1  # +1 for header


class TestExportEUDRXML:
    """Test EUDR-compliant XML export format."""

    def test_export_eudr_xml_valid(self, compliance_reporter):
        """EUDR XML export produces valid XML string."""
        assessments = _batch_assessments()
        xml_str = compliance_reporter.export_eudr_xml(assessments)
        assert isinstance(xml_str, str)
        assert "<?xml" in xml_str or "<" in xml_str

    def test_export_eudr_xml_namespace(self, compliance_reporter):
        """EUDR XML includes EUDR namespace or declaration."""
        assessments = [_compliant_assessment()]
        xml_str = compliance_reporter.export_eudr_xml(assessments)
        # Should contain some EUDR-related tag or namespace
        assert "eudr" in xml_str.lower() or "coordinate" in xml_str.lower()


# ===========================================================================
# 5. Audit Trail
# ===========================================================================


class TestAuditTrail:
    """Test audit trail provenance chain."""

    def test_audit_trail_included(self, compliance_reporter):
        """Report includes provenance chain for audit."""
        assessments = _batch_assessments()
        report = compliance_reporter.generate_report(assessments)
        assert hasattr(report, "provenance_hash")
        assert report.provenance_hash != ""
        assert len(report.provenance_hash) == SHA256_HEX_LENGTH

    def test_audit_trail_deterministic(self, compliance_reporter):
        """Same input produces same audit trail hash."""
        assessments = _batch_assessments()
        r1 = compliance_reporter.generate_report(assessments)
        r2 = compliance_reporter.generate_report(assessments)
        assert r1.provenance_hash == r2.provenance_hash


# ===========================================================================
# 6. Submission Readiness
# ===========================================================================


class TestSubmissionReadiness:
    """Test submission readiness percentage calculation."""

    def test_submission_readiness_all_compliant(self, compliance_reporter):
        """All compliant coordinates = 100% readiness."""
        assessments = [_compliant_assessment(), _compliant_assessment()]
        readiness = compliance_reporter.calculate_readiness(assessments)
        assert readiness == 100.0

    def test_submission_readiness_mixed(self, compliance_reporter):
        """Mixed compliance = partial readiness."""
        assessments = _batch_assessments()
        readiness = compliance_reporter.calculate_readiness(assessments)
        assert 0.0 < readiness < 100.0

    def test_submission_readiness_none_compliant(self, compliance_reporter):
        """No compliant coordinates = 0% readiness."""
        assessments = [_non_compliant_assessment(), _non_compliant_assessment()]
        readiness = compliance_reporter.calculate_readiness(assessments)
        assert readiness == 0.0

    def test_submission_readiness_empty(self, compliance_reporter):
        """Empty batch = 0% readiness."""
        readiness = compliance_reporter.calculate_readiness([])
        assert readiness == 0.0


# ===========================================================================
# 7. Batch Multi-Format Report
# ===========================================================================


class TestBatchReport:
    """Test batch multi-format report generation."""

    def test_batch_report_generation(self, compliance_reporter):
        """Batch report is generated with all sections."""
        assessments = _batch_assessments()
        report = compliance_reporter.generate_report(assessments)
        assert report.total_count == 5
        assert hasattr(report, "summary")
        assert hasattr(report, "provenance_hash")

    def test_batch_report_includes_per_coordinate(self, compliance_reporter):
        """Batch report includes per-coordinate details."""
        assessments = _batch_assessments()
        report = compliance_reporter.generate_report(assessments)
        assert hasattr(report, "certificates") or hasattr(report, "details")


# ===========================================================================
# 8. Parametrized Report Format Tests
# ===========================================================================


@pytest.mark.parametrize(
    "fmt,method_name",
    [
        ("json", "export_json"),
        ("csv", "export_csv"),
        ("xml", "export_eudr_xml"),
    ],
    ids=["json", "csv", "xml"],
)
def test_parametrized_export_formats(compliance_reporter, fmt, method_name):
    """Parametrized: all export formats produce non-empty output."""
    assessments = _batch_assessments()
    method = getattr(compliance_reporter, method_name)
    output = method(assessments)
    assert isinstance(output, str)
    assert len(output) > 0


# ===========================================================================
# 9. Compliance Certificate - Extended
# ===========================================================================


class TestComplianceCertificateExtended:
    """Extended compliance certificate generation tests."""

    def test_compliance_cert_includes_commodity(self, compliance_reporter):
        """Certificate includes commodity information."""
        cert = compliance_reporter.generate_certificate(_compliant_assessment())
        assert hasattr(cert, "commodity") or hasattr(cert, "metadata")

    def test_compliance_cert_includes_country(self, compliance_reporter):
        """Certificate includes country information."""
        cert = compliance_reporter.generate_certificate(_compliant_assessment())
        assert hasattr(cert, "country") or hasattr(cert, "metadata")

    def test_compliance_cert_unique_ids(self, compliance_reporter):
        """Multiple certificates have unique IDs."""
        cert1 = compliance_reporter.generate_certificate(_compliant_assessment())
        cert2 = compliance_reporter.generate_certificate(_non_compliant_assessment())
        assert cert1.certificate_id != cert2.certificate_id

    def test_compliance_cert_score_included(self, compliance_reporter):
        """Certificate includes the total score."""
        cert = compliance_reporter.generate_certificate(_compliant_assessment())
        assert hasattr(cert, "total_score")
        assert cert.total_score == 92.5


# ===========================================================================
# 10. Batch Summary - Extended
# ===========================================================================


class TestBatchSummaryExtended:
    """Extended batch summary statistics tests."""

    def test_batch_summary_all_compliant(self, compliance_reporter):
        """Batch with all compliant coordinates."""
        assessments = [_compliant_assessment(), _compliant_assessment()]
        summary = compliance_reporter.generate_batch_summary(assessments)
        assert summary.total_count == 2
        assert summary.compliant_count == 2
        assert summary.non_compliant_count == 0

    def test_batch_summary_all_non_compliant(self, compliance_reporter):
        """Batch with all non-compliant coordinates."""
        assessments = [_non_compliant_assessment(), _non_compliant_assessment()]
        summary = compliance_reporter.generate_batch_summary(assessments)
        assert summary.total_count == 2
        assert summary.non_compliant_count == 2
        assert summary.compliant_count == 0

    def test_batch_summary_single_item(self, compliance_reporter):
        """Batch with a single assessment."""
        assessments = [_compliant_assessment()]
        summary = compliance_reporter.generate_batch_summary(assessments)
        assert summary.total_count == 1

    def test_batch_summary_average_score_value(self, compliance_reporter):
        """Verify average score is arithmetic mean."""
        assessments = _batch_assessments()
        summary = compliance_reporter.generate_batch_summary(assessments)
        expected_avg = sum(a["total_score"] for a in assessments) / len(assessments)
        assert abs(summary.average_score - expected_avg) < 0.1


# ===========================================================================
# 11. Remediation Guidance - Extended
# ===========================================================================


class TestRemediationGuidanceExtended:
    """Extended remediation guidance tests."""

    def test_remediation_needs_review(self, compliance_reporter):
        """Needs-review coordinate generates remediation guidance."""
        guidance = compliance_reporter.generate_remediation(
            _needs_review_assessment()
        )
        assert isinstance(guidance, list)
        # Should have at least one suggestion for improvement
        assert len(guidance) >= 1

    def test_remediation_items_have_action_or_instruction(self, compliance_reporter):
        """Each remediation item has an action or instruction field."""
        guidance = compliance_reporter.generate_remediation(
            _non_compliant_assessment()
        )
        for item in guidance:
            assert "action" in item or "instruction" in item

    def test_remediation_items_have_priority(self, compliance_reporter):
        """Each remediation item has a priority field."""
        guidance = compliance_reporter.generate_remediation(
            _non_compliant_assessment()
        )
        for item in guidance:
            assert "priority" in item
            assert isinstance(item["priority"], (int, float))


# ===========================================================================
# 12. Export JSON - Extended
# ===========================================================================


class TestExportJSONExtended:
    """Extended JSON export tests."""

    def test_export_json_single_assessment(self, compliance_reporter):
        """JSON export for a single assessment."""
        assessments = [_compliant_assessment()]
        json_str = compliance_reporter.export_json(assessments)
        parsed = json.loads(json_str)
        assert isinstance(parsed, (dict, list))

    def test_export_json_empty(self, compliance_reporter):
        """JSON export for empty assessments."""
        json_str = compliance_reporter.export_json([])
        parsed = json.loads(json_str)
        assert isinstance(parsed, (dict, list))

    def test_export_json_preserves_scores(self, compliance_reporter):
        """JSON export preserves score values."""
        assessments = [_compliant_assessment()]
        json_str = compliance_reporter.export_json(assessments)
        assert "92.5" in json_str or "92.50" in json_str


# ===========================================================================
# 13. Export CSV - Extended
# ===========================================================================


class TestExportCSVExtended:
    """Extended CSV export tests."""

    def test_export_csv_single_assessment(self, compliance_reporter):
        """CSV export for a single assessment."""
        assessments = [_compliant_assessment()]
        csv_str = compliance_reporter.export_csv(assessments)
        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row

    def test_export_csv_empty(self, compliance_reporter):
        """CSV export for empty assessments returns header only."""
        csv_str = compliance_reporter.export_csv([])
        lines = csv_str.strip().split("\n")
        assert len(lines) >= 1  # At least header

    def test_export_csv_no_injection(self, compliance_reporter):
        """CSV export does not contain formula injection characters."""
        assessments = _batch_assessments()
        csv_str = compliance_reporter.export_csv(assessments)
        # CSV injection characters should not appear at start of cells
        for line in csv_str.strip().split("\n")[1:]:
            for cell in line.split(","):
                cell = cell.strip().strip('"')
                if cell:
                    assert not cell.startswith("="), f"Potential CSV injection: {cell}"
                    assert not cell.startswith("+"), f"Potential CSV injection: {cell}"


# ===========================================================================
# 14. Audit Trail - Extended
# ===========================================================================


class TestAuditTrailExtended:
    """Extended audit trail provenance tests."""

    def test_audit_trail_different_inputs_different_hash(self, compliance_reporter):
        """Different assessment data produces different audit hashes."""
        r1 = compliance_reporter.generate_report([_compliant_assessment()])
        r2 = compliance_reporter.generate_report([_non_compliant_assessment()])
        assert r1.provenance_hash != r2.provenance_hash

    def test_audit_trail_hash_hex_format(self, compliance_reporter):
        """Audit trail hash is valid hexadecimal."""
        assessments = _batch_assessments()
        report = compliance_reporter.generate_report(assessments)
        import re
        assert re.match(r"^[0-9a-f]{64}$", report.provenance_hash)


# ===========================================================================
# 15. Submission Readiness - Extended
# ===========================================================================


class TestSubmissionReadinessExtended:
    """Extended submission readiness tests."""

    def test_readiness_percentage_range(self, compliance_reporter):
        """Readiness percentage is always 0-100."""
        assessments = _batch_assessments()
        readiness = compliance_reporter.calculate_readiness(assessments)
        assert 0.0 <= readiness <= 100.0

    def test_readiness_single_compliant(self, compliance_reporter):
        """Single compliant coordinate = 100% readiness."""
        assessments = [_compliant_assessment()]
        readiness = compliance_reporter.calculate_readiness(assessments)
        assert readiness == 100.0

    def test_readiness_single_non_compliant(self, compliance_reporter):
        """Single non-compliant coordinate = 0% readiness."""
        assessments = [_non_compliant_assessment()]
        readiness = compliance_reporter.calculate_readiness(assessments)
        assert readiness == 0.0


# ===========================================================================
# 16. Batch Report - Extended
# ===========================================================================


class TestBatchReportExtended:
    """Extended batch report generation tests."""

    def test_batch_report_empty(self, compliance_reporter):
        """Batch report for empty assessments."""
        report = compliance_reporter.generate_report([])
        assert report.total_count == 0
        assert hasattr(report, "provenance_hash")

    def test_batch_report_single_item(self, compliance_reporter):
        """Batch report for a single assessment."""
        assessments = [_compliant_assessment()]
        report = compliance_reporter.generate_report(assessments)
        assert report.total_count == 1

    def test_batch_report_large_batch(self, compliance_reporter):
        """Batch report for a larger batch of assessments."""
        # Create a batch of 20 assessments
        assessments = _batch_assessments() * 4  # 5 * 4 = 20
        report = compliance_reporter.generate_report(assessments)
        assert report.total_count == 20
        assert hasattr(report, "provenance_hash")
        assert len(report.provenance_hash) == SHA256_HEX_LENGTH
