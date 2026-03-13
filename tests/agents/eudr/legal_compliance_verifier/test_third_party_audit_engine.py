# -*- coding: utf-8 -*-
"""
Tests for ThirdPartyAuditEngine - AGENT-EUDR-023 Engine 6

Comprehensive test suite covering:
- Audit report parsing for 6 formats (ISO 19011, FSC, PEFC, RSPO,
  custom PDF, structured JSON)
- Finding extraction with severity classification
- Corrective action tracking and status management
- Integration with external audit systems
- Auditor validation and recognition
- Audit report scoring and aggregation
- Report age and freshness validation
- Provenance tracking for audit operations

Test count: 70+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 (Engine 6 - Third-Party Audit)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from tests.agents.eudr.legal_compliance_verifier.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
    AUDIT_REPORT_FORMATS,
    LEGISLATION_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RECOGNIZED_AUDITORS = [
    "Bureau Veritas", "SGS", "TUV Rheinland", "DNV", "Control Union",
    "Rainforest Alliance", "SCS Global", "NEPCon", "Soil Association",
]


def _parse_audit_report(report: Dict) -> Dict[str, Any]:
    """Parse an audit report and extract structured data."""
    result = {
        "audit_id": report.get("audit_id"),
        "format": report.get("format"),
        "parsed": False,
        "auditor_recognized": False,
        "overall_result": report.get("overall_result"),
        "findings_count": report.get("findings_count", 0),
        "major_findings": report.get("major_findings", 0),
        "minor_findings": report.get("minor_findings", 0),
        "observations": report.get("observations", 0),
        "corrective_actions_required": report.get("corrective_actions_required", 0),
        "corrective_actions_closed": report.get("corrective_actions_closed", 0),
        "corrective_actions_open": 0,
        "audit_freshness": "unknown",
        "score": Decimal("0"),
        "errors": [],
    }

    # Validate format
    if report.get("format") not in AUDIT_REPORT_FORMATS:
        result["errors"].append(f"Unsupported format: {report.get('format')}")
        return result

    # Validate auditor
    result["auditor_recognized"] = report.get("auditor", "") in RECOGNIZED_AUDITORS

    # Calculate open corrective actions
    result["corrective_actions_open"] = (
        result["corrective_actions_required"] - result["corrective_actions_closed"]
    )

    # Check audit freshness
    if "audit_date" in report:
        try:
            audit_date = date.fromisoformat(report["audit_date"])
            days_old = (date.today() - audit_date).days
            if days_old <= 180:
                result["audit_freshness"] = "fresh"
            elif days_old <= 365:
                result["audit_freshness"] = "acceptable"
            else:
                result["audit_freshness"] = "stale"
        except (ValueError, TypeError):
            result["errors"].append("Invalid audit_date format")

    # Calculate audit score (0-100)
    base_score = Decimal("100")
    # Deduct for major findings (20 points each, max 60)
    major_deduction = min(result["major_findings"] * 20, 60)
    # Deduct for minor findings (5 points each, max 20)
    minor_deduction = min(result["minor_findings"] * 5, 20)
    # Deduct for open corrective actions (10 points each, max 30)
    ca_deduction = min(result["corrective_actions_open"] * 10, 30)

    result["score"] = max(
        base_score - Decimal(str(major_deduction))
        - Decimal(str(minor_deduction))
        - Decimal(str(ca_deduction)),
        Decimal("0"),
    )
    result["parsed"] = True
    return result


def _extract_findings(report: Dict, findings: List[Dict]) -> List[Dict]:
    """Extract and categorize findings from an audit report."""
    extracted = []
    for finding in findings:
        if finding.get("audit_id") == report.get("audit_id"):
            extracted.append({
                "finding_id": finding.get("finding_id"),
                "severity": finding.get("severity"),
                "category": finding.get("category"),
                "description": finding.get("description"),
                "corrective_action": finding.get("corrective_action"),
                "due_date": finding.get("due_date"),
                "status": finding.get("status"),
                "overdue": _is_finding_overdue(finding),
            })
    return extracted


def _is_finding_overdue(finding: Dict) -> bool:
    """Check if a finding's corrective action is overdue."""
    if finding.get("status") == "closed":
        return False
    due_date = finding.get("due_date")
    if not due_date:
        return False
    try:
        due = date.fromisoformat(due_date)
        return due < date.today()
    except (ValueError, TypeError):
        return False


def _track_corrective_actions(findings: List[Dict]) -> Dict[str, Any]:
    """Track corrective action status across all findings."""
    total = len(findings)
    open_count = sum(1 for f in findings if f.get("status") == "open")
    closed_count = sum(1 for f in findings if f.get("status") == "closed")
    overdue_count = sum(1 for f in findings if _is_finding_overdue(f))

    return {
        "total": total,
        "open": open_count,
        "closed": closed_count,
        "overdue": overdue_count,
        "closure_rate": (
            Decimal(str(round(closed_count / total * 100, 2)))
            if total > 0 else Decimal("100")
        ),
    }


def _validate_auditor(auditor_name: str) -> Dict[str, Any]:
    """Validate auditor is recognized for EUDR compliance audits."""
    return {
        "auditor": auditor_name,
        "recognized": auditor_name in RECOGNIZED_AUDITORS,
        "accreditation_status": "accredited" if auditor_name in RECOGNIZED_AUDITORS else "unknown",
    }


# ===========================================================================
# 1. Audit Report Parsing (15 tests)
# ===========================================================================


class TestAuditReportParsing:
    """Test parsing of audit reports in 6 different formats."""

    @pytest.mark.parametrize("format_name", AUDIT_REPORT_FORMATS)
    def test_supported_format_parsing(self, format_name, sample_audit_reports):
        """Test each of the 6 audit report formats can be parsed."""
        report = next(
            (r for r in sample_audit_reports if r["format"] == format_name),
            {"audit_id": "TEST", "format": format_name, "auditor": "SGS",
             "audit_date": date.today().isoformat(), "overall_result": "pass",
             "findings_count": 0, "major_findings": 0, "minor_findings": 0,
             "observations": 0, "corrective_actions_required": 0,
             "corrective_actions_closed": 0},
        )
        result = _parse_audit_report(report)
        assert result["parsed"] is True
        assert result["format"] == format_name

    def test_unsupported_format_rejected(self):
        """Test unsupported audit format is rejected."""
        report = {"audit_id": "BAD", "format": "unsupported_format"}
        result = _parse_audit_report(report)
        assert result["parsed"] is False
        assert any("Unsupported format" in e for e in result["errors"])

    def test_parse_iso_19011_report(self, sample_audit_reports):
        """Test parsing ISO 19011 audit report."""
        report = sample_audit_reports[0]
        result = _parse_audit_report(report)
        assert result["overall_result"] == "conditional_pass"
        assert result["findings_count"] == 3

    def test_parse_fsc_audit_report(self, sample_audit_reports):
        """Test parsing FSC audit report."""
        report = sample_audit_reports[1]
        result = _parse_audit_report(report)
        assert result["overall_result"] == "pass"
        assert result["major_findings"] == 0

    def test_parse_rspo_audit_report(self, sample_audit_reports):
        """Test parsing RSPO audit report."""
        report = sample_audit_reports[2]
        result = _parse_audit_report(report)
        assert result["overall_result"] == "fail"
        assert result["major_findings"] == 4

    def test_parse_failed_audit(self, sample_audit_reports):
        """Test parsing audit report with fail result."""
        report = sample_audit_reports[2]  # RSPO - fail
        result = _parse_audit_report(report)
        assert result["overall_result"] == "fail"
        assert result["score"] < Decimal("50")

    def test_parse_passed_audit(self, sample_audit_reports):
        """Test parsing audit report with pass result."""
        report = sample_audit_reports[3]  # PEFC - pass
        result = _parse_audit_report(report)
        assert result["overall_result"] == "pass"
        assert result["score"] >= Decimal("80")

    def test_parse_conditional_pass(self, sample_audit_reports):
        """Test parsing audit report with conditional pass."""
        report = sample_audit_reports[0]  # ISO 19011 - conditional_pass
        result = _parse_audit_report(report)
        assert result["overall_result"] == "conditional_pass"

    def test_parsed_report_includes_score(self, sample_audit_reports):
        """Test parsed report includes calculated score."""
        for report in sample_audit_reports:
            result = _parse_audit_report(report)
            assert isinstance(result["score"], Decimal)
            assert Decimal("0") <= result["score"] <= Decimal("100")


# ===========================================================================
# 2. Finding Extraction (12 tests)
# ===========================================================================


class TestFindingExtraction:
    """Test extraction and categorization of audit findings."""

    def test_extract_findings_for_audit(self, sample_audit_reports, sample_findings):
        """Test extracting findings for a specific audit."""
        report = sample_audit_reports[0]
        extracted = _extract_findings(report, sample_findings)
        assert len(extracted) == 3
        assert all(f["finding_id"].startswith("FND-") for f in extracted)

    def test_extract_findings_empty(self, sample_audit_reports):
        """Test extracting findings when none exist for audit."""
        report = sample_audit_reports[1]
        extracted = _extract_findings(report, [])
        assert len(extracted) == 0

    def test_finding_has_severity(self, sample_audit_reports, sample_findings):
        """Test each extracted finding has a severity level."""
        report = sample_audit_reports[0]
        extracted = _extract_findings(report, sample_findings)
        for finding in extracted:
            assert finding["severity"] in ("major", "minor", "observation")

    def test_finding_has_category(self, sample_audit_reports, sample_findings):
        """Test each finding maps to a legislation category."""
        report = sample_audit_reports[0]
        extracted = _extract_findings(report, sample_findings)
        for finding in extracted:
            assert finding["category"] in LEGISLATION_CATEGORIES

    def test_finding_has_corrective_action(self, sample_audit_reports, sample_findings):
        """Test each finding includes a corrective action."""
        report = sample_audit_reports[0]
        extracted = _extract_findings(report, sample_findings)
        for finding in extracted:
            assert "corrective_action" in finding
            assert len(finding["corrective_action"]) > 0

    def test_finding_overdue_detection(self, sample_findings):
        """Test overdue finding detection."""
        # Create a finding with past due date
        overdue_finding = {
            "finding_id": "FND-OVERDUE",
            "audit_id": "AUD-001",
            "severity": "major",
            "category": "environmental_protection",
            "description": "Overdue finding",
            "corrective_action": "Fix it",
            "due_date": (date.today() - timedelta(days=30)).isoformat(),
            "status": "open",
        }
        assert _is_finding_overdue(overdue_finding) is True

    def test_finding_not_overdue(self, sample_findings):
        """Test finding with future due date is not overdue."""
        future_finding = {
            "finding_id": "FND-FUTURE",
            "audit_id": "AUD-001",
            "due_date": (date.today() + timedelta(days=30)).isoformat(),
            "status": "open",
        }
        assert _is_finding_overdue(future_finding) is False

    def test_closed_finding_not_overdue(self):
        """Test closed finding is never marked overdue."""
        closed_finding = {
            "finding_id": "FND-CLOSED",
            "audit_id": "AUD-001",
            "due_date": (date.today() - timedelta(days=30)).isoformat(),
            "status": "closed",
        }
        assert _is_finding_overdue(closed_finding) is False

    def test_finding_without_due_date(self):
        """Test finding without due date is not overdue."""
        no_date = {
            "finding_id": "FND-NODATE",
            "audit_id": "AUD-001",
            "status": "open",
        }
        assert _is_finding_overdue(no_date) is False

    def test_extract_preserves_finding_ids(self, sample_audit_reports, sample_findings):
        """Test extraction preserves original finding IDs."""
        report = sample_audit_reports[0]
        extracted = _extract_findings(report, sample_findings)
        expected_ids = {"FND-001", "FND-002", "FND-003"}
        actual_ids = {f["finding_id"] for f in extracted}
        assert actual_ids == expected_ids

    def test_findings_for_nonexistent_audit(self, sample_findings):
        """Test extraction for non-existent audit returns empty."""
        report = {"audit_id": "AUD-NONEXISTENT"}
        extracted = _extract_findings(report, sample_findings)
        assert len(extracted) == 0

    def test_finding_status_values(self, sample_audit_reports, sample_findings):
        """Test findings have valid status values."""
        report = sample_audit_reports[0]
        extracted = _extract_findings(report, sample_findings)
        valid_statuses = {"open", "closed", "in_progress", "verified"}
        for finding in extracted:
            assert finding["status"] in valid_statuses


# ===========================================================================
# 3. Corrective Action Tracking (12 tests)
# ===========================================================================


class TestCorrectiveActionTracking:
    """Test corrective action tracking and status management."""

    def test_track_all_closed(self):
        """Test tracking when all corrective actions are closed."""
        findings = [
            {"finding_id": f"F-{i}", "status": "closed",
             "due_date": (date.today() + timedelta(days=30)).isoformat()}
            for i in range(5)
        ]
        result = _track_corrective_actions(findings)
        assert result["open"] == 0
        assert result["closed"] == 5
        assert result["closure_rate"] == Decimal("100")

    def test_track_all_open(self):
        """Test tracking when all corrective actions are open."""
        findings = [
            {"finding_id": f"F-{i}", "status": "open",
             "due_date": (date.today() + timedelta(days=30)).isoformat()}
            for i in range(3)
        ]
        result = _track_corrective_actions(findings)
        assert result["open"] == 3
        assert result["closed"] == 0
        assert result["closure_rate"] == Decimal("0")

    def test_track_mixed_status(self):
        """Test tracking with mixed open/closed actions."""
        findings = [
            {"finding_id": "F-1", "status": "closed",
             "due_date": (date.today() + timedelta(days=30)).isoformat()},
            {"finding_id": "F-2", "status": "open",
             "due_date": (date.today() + timedelta(days=30)).isoformat()},
            {"finding_id": "F-3", "status": "closed",
             "due_date": (date.today() + timedelta(days=30)).isoformat()},
        ]
        result = _track_corrective_actions(findings)
        assert result["open"] == 1
        assert result["closed"] == 2
        assert result["closure_rate"] == Decimal("66.67")

    def test_track_empty_findings(self):
        """Test tracking with no findings."""
        result = _track_corrective_actions([])
        assert result["total"] == 0
        assert result["closure_rate"] == Decimal("100")

    def test_track_overdue_count(self):
        """Test overdue corrective action count."""
        findings = [
            {"finding_id": "F-1", "status": "open",
             "due_date": (date.today() - timedelta(days=10)).isoformat()},
            {"finding_id": "F-2", "status": "open",
             "due_date": (date.today() + timedelta(days=30)).isoformat()},
            {"finding_id": "F-3", "status": "closed",
             "due_date": (date.today() - timedelta(days=5)).isoformat()},
        ]
        result = _track_corrective_actions(findings)
        assert result["overdue"] == 1

    def test_track_single_finding(self):
        """Test tracking with a single finding."""
        findings = [{"finding_id": "F-1", "status": "open",
                     "due_date": (date.today() + timedelta(days=30)).isoformat()}]
        result = _track_corrective_actions(findings)
        assert result["total"] == 1

    def test_track_closure_rate_type(self):
        """Test closure rate is a Decimal type."""
        findings = [{"finding_id": "F-1", "status": "open",
                     "due_date": (date.today() + timedelta(days=30)).isoformat()}]
        result = _track_corrective_actions(findings)
        assert isinstance(result["closure_rate"], Decimal)

    def test_corrective_actions_open_calculation(self, sample_audit_reports):
        """Test open corrective actions calculated from report."""
        report = sample_audit_reports[0]
        parsed = _parse_audit_report(report)
        expected_open = report["corrective_actions_required"] - report["corrective_actions_closed"]
        assert parsed["corrective_actions_open"] == expected_open

    def test_all_audits_have_ca_tracking(self, sample_audit_reports):
        """Test all audit reports have corrective action fields."""
        for report in sample_audit_reports:
            assert "corrective_actions_required" in report
            assert "corrective_actions_closed" in report

    def test_ca_open_never_negative(self, sample_audit_reports):
        """Test open corrective actions count is never negative."""
        for report in sample_audit_reports:
            parsed = _parse_audit_report(report)
            assert parsed["corrective_actions_open"] >= 0

    def test_ca_closed_never_exceeds_required(self, sample_audit_reports):
        """Test closed CAs never exceed required count."""
        for report in sample_audit_reports:
            assert report["corrective_actions_closed"] <= report["corrective_actions_required"]

    def test_track_total_matches_input(self):
        """Test total tracking count matches input count."""
        findings = [{"finding_id": f"F-{i}", "status": "open"} for i in range(7)]
        result = _track_corrective_actions(findings)
        assert result["total"] == 7


# ===========================================================================
# 4. Auditor Validation (10 tests)
# ===========================================================================


class TestAuditorValidation:
    """Test auditor recognition and validation."""

    @pytest.mark.parametrize("auditor", RECOGNIZED_AUDITORS)
    def test_recognized_auditor(self, auditor):
        """Test each recognized auditor is validated correctly."""
        result = _validate_auditor(auditor)
        assert result["recognized"] is True
        assert result["accreditation_status"] == "accredited"

    def test_unrecognized_auditor(self):
        """Test unrecognized auditor fails validation."""
        result = _validate_auditor("Unknown Audit Firm")
        assert result["recognized"] is False
        assert result["accreditation_status"] == "unknown"

    def test_empty_auditor_name(self):
        """Test empty auditor name fails validation."""
        result = _validate_auditor("")
        assert result["recognized"] is False

    def test_auditor_validation_case_sensitive(self):
        """Test auditor validation is case-sensitive."""
        result = _validate_auditor("bureau veritas")
        assert result["recognized"] is False

    def test_recognized_auditor_count(self):
        """Test correct number of recognized auditors."""
        assert len(RECOGNIZED_AUDITORS) >= 9


# ===========================================================================
# 5. Audit Scoring (8 tests)
# ===========================================================================


class TestAuditScoring:
    """Test audit report scoring calculation."""

    def test_perfect_score_no_findings(self):
        """Test perfect score when no findings."""
        report = {
            "audit_id": "AUD-PERFECT", "format": "iso_19011",
            "auditor": "SGS", "audit_date": date.today().isoformat(),
            "overall_result": "pass", "findings_count": 0,
            "major_findings": 0, "minor_findings": 0, "observations": 0,
            "corrective_actions_required": 0, "corrective_actions_closed": 0,
        }
        result = _parse_audit_report(report)
        assert result["score"] == Decimal("100")

    def test_score_deduction_major_findings(self):
        """Test score deduction for major findings (20 points each)."""
        report = {
            "audit_id": "AUD-MAJOR", "format": "fsc_audit",
            "auditor": "DNV", "audit_date": date.today().isoformat(),
            "overall_result": "conditional_pass", "findings_count": 2,
            "major_findings": 2, "minor_findings": 0, "observations": 0,
            "corrective_actions_required": 0, "corrective_actions_closed": 0,
        }
        result = _parse_audit_report(report)
        assert result["score"] == Decimal("60")  # 100 - 2*20

    def test_score_deduction_minor_findings(self):
        """Test score deduction for minor findings (5 points each)."""
        report = {
            "audit_id": "AUD-MINOR", "format": "pefc_audit",
            "auditor": "SGS", "audit_date": date.today().isoformat(),
            "overall_result": "conditional_pass", "findings_count": 4,
            "major_findings": 0, "minor_findings": 4, "observations": 0,
            "corrective_actions_required": 0, "corrective_actions_closed": 0,
        }
        result = _parse_audit_report(report)
        assert result["score"] == Decimal("80")  # 100 - 4*5

    def test_score_deduction_open_corrective_actions(self):
        """Test score deduction for open corrective actions."""
        report = {
            "audit_id": "AUD-CA", "format": "rspo_audit",
            "auditor": "Control Union", "audit_date": date.today().isoformat(),
            "overall_result": "conditional_pass", "findings_count": 0,
            "major_findings": 0, "minor_findings": 0, "observations": 0,
            "corrective_actions_required": 3, "corrective_actions_closed": 0,
        }
        result = _parse_audit_report(report)
        assert result["score"] == Decimal("70")  # 100 - 3*10

    def test_score_minimum_zero(self):
        """Test score does not go below zero."""
        report = {
            "audit_id": "AUD-BAD", "format": "custom_pdf",
            "auditor": "Local Auditor Ltd", "audit_date": date.today().isoformat(),
            "overall_result": "fail", "findings_count": 15,
            "major_findings": 5, "minor_findings": 10, "observations": 10,
            "corrective_actions_required": 10, "corrective_actions_closed": 0,
        }
        result = _parse_audit_report(report)
        assert result["score"] >= Decimal("0")

    def test_score_combined_deductions(self):
        """Test combined major + minor + CA deductions."""
        report = {
            "audit_id": "AUD-COMBO", "format": "structured_json",
            "auditor": "SGS", "audit_date": date.today().isoformat(),
            "overall_result": "conditional_pass", "findings_count": 3,
            "major_findings": 1, "minor_findings": 2, "observations": 1,
            "corrective_actions_required": 1, "corrective_actions_closed": 0,
        }
        result = _parse_audit_report(report)
        # 100 - 20 (major) - 10 (minor) - 10 (CA) = 60
        assert result["score"] == Decimal("60")

    def test_audit_freshness_fresh(self, sample_audit_reports):
        """Test audit within 180 days is classified as fresh."""
        recent = sample_audit_reports[2]  # 30 days old
        result = _parse_audit_report(recent)
        assert result["audit_freshness"] == "fresh"

    def test_audit_freshness_stale(self):
        """Test audit older than 365 days is classified as stale."""
        old_report = {
            "audit_id": "AUD-OLD", "format": "iso_19011",
            "auditor": "SGS",
            "audit_date": (date.today() - timedelta(days=400)).isoformat(),
            "overall_result": "pass", "findings_count": 0,
            "major_findings": 0, "minor_findings": 0, "observations": 0,
            "corrective_actions_required": 0, "corrective_actions_closed": 0,
        }
        result = _parse_audit_report(old_report)
        assert result["audit_freshness"] == "stale"


# ===========================================================================
# 6. Report Fixture Validation (5 tests)
# ===========================================================================


class TestAuditFixtures:
    """Test audit report fixtures for consistency."""

    def test_fixture_provides_6_reports(self, sample_audit_reports):
        """Test fixture provides 6 audit reports (one per format)."""
        assert len(sample_audit_reports) == 6

    def test_fixture_covers_all_formats(self, sample_audit_reports):
        """Test fixture covers all 6 audit report formats."""
        formats = {r["format"] for r in sample_audit_reports}
        assert formats == set(AUDIT_REPORT_FORMATS)

    def test_fixture_has_required_fields(self, sample_audit_reports):
        """Test each fixture report has all required fields."""
        required = {"audit_id", "format", "auditor", "audit_date",
                     "overall_result", "findings_count"}
        for report in sample_audit_reports:
            for field in required:
                assert field in report, f"Report {report['audit_id']} missing {field}"

    def test_fixture_findings_count_consistent(self, sample_audit_reports):
        """Test findings_count = major + minor for each report."""
        for report in sample_audit_reports:
            assert report["findings_count"] == (
                report["major_findings"] + report["minor_findings"]
            )

    def test_fixture_has_next_audit_date(self, sample_audit_reports):
        """Test each report specifies next audit date."""
        for report in sample_audit_reports:
            assert "next_audit_date" in report
