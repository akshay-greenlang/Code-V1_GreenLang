# -*- coding: utf-8 -*-
"""
Unit tests for ReportGenerator -- CDP submission report generation.

Tests PDF generation, Excel export, XML/ORS format, executive summary,
submission checklist validation, and verification package assembly
with 27+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal
from datetime import datetime

import pytest

from services.config import SubmissionFormat
from services.models import (
    CDPSubmission,
    CDPQuestionnaire,
    CDPResponse,
    CDPScoringResult,
    _new_id,
)
from services.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

class TestPDFGeneration:
    """Test PDF report generation."""

    def test_generate_pdf_report(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_pdf(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            title="CDP Climate Change 2025 -- Acme Corp",
        )
        assert result["format"] == "pdf"
        assert result["file_path"] is not None
        assert result["file_path"].endswith(".pdf")

    def test_pdf_includes_all_modules(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_pdf(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            title="CDP Report",
        )
        assert result["modules_included"] >= 10

    def test_pdf_has_page_count(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_pdf(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            title="CDP Report",
        )
        assert result["page_count"] > 0

    def test_pdf_file_size_reasonable(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_pdf(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            title="CDP Report",
        )
        assert result["file_size_bytes"] > 0


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

class TestExcelExport:
    """Test Excel tabular export."""

    def test_generate_excel(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_excel(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert result["format"] == "excel"
        assert result["file_path"].endswith(".xlsx")

    def test_excel_has_sheets(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_excel(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert result["sheet_count"] > 0
        assert "sheets" in result

    def test_excel_includes_responses(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_excel(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert "responses" in result["sheets"] or "Responses" in result["sheets"]


# ---------------------------------------------------------------------------
# XML / ORS format
# ---------------------------------------------------------------------------

class TestXMLExport:
    """Test CDP Online Response System XML export."""

    def test_generate_xml(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_xml(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert result["format"] == "xml"
        assert result["file_path"].endswith(".xml")

    def test_xml_ors_compatible(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_xml(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert result["ors_compatible"] is True

    def test_xml_has_schema_version(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_xml(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert "schema_version" in result

    def test_xml_includes_metadata(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_xml(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert "organization" in result["metadata"]
        assert "reporting_year" in result["metadata"]


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

class TestExecutiveSummary:
    """Test executive summary generation."""

    def test_generate_executive_summary(self, report_generator, sample_questionnaire, sample_organization, sample_scoring_result):
        summary = report_generator.generate_executive_summary(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            scoring_result=sample_scoring_result,
        )
        assert "overall_score" in summary
        assert "score_band" in summary
        assert "key_highlights" in summary

    def test_summary_includes_score(self, report_generator, sample_questionnaire, sample_organization, sample_scoring_result):
        summary = report_generator.generate_executive_summary(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            scoring_result=sample_scoring_result,
        )
        assert summary["overall_score"] == Decimal("72.5")
        assert summary["score_band"] == "A-"

    def test_summary_has_recommendations(self, report_generator, sample_questionnaire, sample_organization, sample_scoring_result):
        summary = report_generator.generate_executive_summary(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            scoring_result=sample_scoring_result,
        )
        assert "recommendations" in summary
        assert len(summary["recommendations"]) > 0

    def test_board_report_format(self, report_generator, sample_questionnaire, sample_organization, sample_scoring_result):
        report = report_generator.generate_board_report(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            scoring_result=sample_scoring_result,
        )
        assert report["format"] == "pdf"
        assert "governance" in report.get("sections", []) or len(report.get("sections", [])) > 0


# ---------------------------------------------------------------------------
# Submission checklist
# ---------------------------------------------------------------------------

class TestSubmissionChecklist:
    """Test submission readiness checklist."""

    def test_validate_checklist_complete(self, report_generator, sample_questionnaire):
        checklist = report_generator.validate_submission_checklist(
            questionnaire_id=sample_questionnaire.id,
            all_required_answered=True,
            all_responses_approved=True,
            verification_attached=True,
            sign_off_completed=True,
        )
        assert checklist["is_ready"] is True
        assert checklist["missing_items"] == []

    def test_validate_checklist_incomplete(self, report_generator, sample_questionnaire):
        checklist = report_generator.validate_submission_checklist(
            questionnaire_id=sample_questionnaire.id,
            all_required_answered=True,
            all_responses_approved=False,
            verification_attached=False,
            sign_off_completed=False,
        )
        assert checklist["is_ready"] is False
        assert len(checklist["missing_items"]) >= 2

    def test_checklist_items_enumerated(self, report_generator, sample_questionnaire):
        checklist = report_generator.validate_submission_checklist(
            questionnaire_id=sample_questionnaire.id,
            all_required_answered=False,
            all_responses_approved=False,
            verification_attached=False,
            sign_off_completed=False,
        )
        assert len(checklist["missing_items"]) == 4

    def test_completion_percentage(self, report_generator, sample_questionnaire):
        checklist = report_generator.validate_submission_checklist(
            questionnaire_id=sample_questionnaire.id,
            all_required_answered=True,
            all_responses_approved=True,
            verification_attached=False,
            sign_off_completed=False,
        )
        assert checklist["completion_pct"] == Decimal("50.0")


# ---------------------------------------------------------------------------
# Verification package
# ---------------------------------------------------------------------------

class TestVerificationPackage:
    """Test verification package assembly."""

    def test_assemble_verification_package(self, report_generator, sample_questionnaire, sample_organization):
        package = report_generator.assemble_verification_package(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert "documents" in package
        assert "evidence_files" in package
        assert "summary" in package

    def test_package_includes_scope_breakdown(self, report_generator, sample_questionnaire, sample_organization):
        package = report_generator.assemble_verification_package(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
        )
        assert "scope_1" in package["summary"] or "scopes" in package["summary"]

    def test_create_submission_record(self, report_generator, sample_questionnaire, sample_organization):
        submission = report_generator.create_submission(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            format=SubmissionFormat.XML,
            file_path="/exports/cdp_2025.xml",
        )
        assert isinstance(submission, CDPSubmission)
        assert submission.submission_format == SubmissionFormat.XML
        assert submission.submitted_at is not None


# ---------------------------------------------------------------------------
# Multi-language support
# ---------------------------------------------------------------------------

class TestMultiLanguage:
    """Test multi-language report support."""

    def test_generate_english_report(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_pdf(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            title="CDP Climate Change 2025",
            language="en",
        )
        assert result["language"] == "en"

    def test_default_language_is_english(self, report_generator, sample_questionnaire, sample_organization):
        result = report_generator.generate_pdf(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            title="CDP Report",
        )
        assert result.get("language", "en") == "en"


# ---------------------------------------------------------------------------
# Submission lifecycle
# ---------------------------------------------------------------------------

class TestSubmissionLifecycle:
    """Test submission record management."""

    def test_submission_status_pending(self, report_generator, sample_questionnaire, sample_organization):
        submission = report_generator.create_submission(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            format=SubmissionFormat.XML,
            file_path="/exports/cdp.xml",
        )
        assert submission.status in ["pending", "submitted"]

    def test_submission_pdf_format(self, report_generator, sample_questionnaire, sample_organization):
        submission = report_generator.create_submission(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            format=SubmissionFormat.PDF,
            file_path="/exports/cdp.pdf",
        )
        assert submission.submission_format == SubmissionFormat.PDF

    def test_submission_has_reference(self, report_generator, sample_questionnaire, sample_organization):
        submission = report_generator.create_submission(
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            format=SubmissionFormat.XML,
            file_path="/exports/cdp.xml",
            submission_reference="CDP-2025-SUB-042",
        )
        assert submission.submission_reference == "CDP-2025-SUB-042"

    def test_get_submission_history(self, report_generator, sample_questionnaire, sample_organization):
        for i in range(3):
            report_generator.create_submission(
                questionnaire_id=sample_questionnaire.id,
                org_id=sample_organization.id,
                format=SubmissionFormat.XML,
                file_path=f"/exports/cdp_{i}.xml",
            )
        history = report_generator.get_submission_history(
            org_id=sample_organization.id,
        )
        assert len(history) >= 3
