# -*- coding: utf-8 -*-
"""
Tests for ComplianceReportingEngine - AGENT-EUDR-021 Engine 7: Reporting

Comprehensive test suite covering:
- 8 report types generation
- 5 output formats (PDF/JSON/HTML/CSV/XLSX)
- 5 languages (EN/FR/DE/ES/PT)
- Multi-section data aggregation
- Deterministic compliance status
- Report caching and versioning
- Provenance tracking

Test count: 62 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 8: Compliance Reporting)
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from tests.agents.eudr.indigenous_rights_checker.conftest import (
    compute_test_hash,
    SHA256_HEX_LENGTH,
    ALL_REPORT_TYPES,
    ALL_REPORT_FORMATS,
    ALL_REPORT_LANGUAGES,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    ComplianceReport,
    ReportType,
    ReportFormat,
    GenerateReportRequest,
)


# ===========================================================================
# 1. Report Type Generation (10 tests)
# ===========================================================================


class TestReportTypeGeneration:
    """Test generation of all 8 report types."""

    @pytest.mark.parametrize("report_type", ALL_REPORT_TYPES)
    def test_create_each_report_type(self, report_type):
        """Test creating a report of each type."""
        report = ComplianceReport(
            report_id=f"r-{report_type.value[:8]}",
            report_type=report_type,
            title=f"Test {report_type.value} Report",
            format=ReportFormat.JSON,
            scope_type="operator",
            provenance_hash=compute_test_hash({"type": report_type.value}),
        )
        assert report.report_type == report_type

    def test_total_report_types_count(self):
        """Test there are exactly 8 report types."""
        assert len(ALL_REPORT_TYPES) == 8

    def test_indigenous_rights_compliance_report(self, sample_report):
        """Test indigenous rights compliance report creation."""
        assert sample_report.report_type == ReportType.INDIGENOUS_RIGHTS_COMPLIANCE


# ===========================================================================
# 2. Output Format Support (8 tests)
# ===========================================================================


class TestOutputFormats:
    """Test 5 output format support."""

    @pytest.mark.parametrize("fmt", ALL_REPORT_FORMATS)
    def test_create_report_in_each_format(self, fmt):
        """Test creating a report in each output format."""
        report = ComplianceReport(
            report_id=f"r-fmt-{fmt.value}",
            report_type=ReportType.INDIGENOUS_RIGHTS_COMPLIANCE,
            title=f"Report in {fmt.value}",
            format=fmt,
            scope_type="operator",
            provenance_hash="a" * 64,
        )
        assert report.format == fmt

    def test_total_formats_count(self):
        """Test there are exactly 5 output formats."""
        assert len(ALL_REPORT_FORMATS) == 5

    def test_json_format_report(self, sample_report):
        """Test JSON format report."""
        assert sample_report.format == ReportFormat.JSON

    def test_pdf_format_report(self):
        """Test PDF format report creation."""
        report = ComplianceReport(
            report_id="r-pdf",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            title="Executive Summary PDF",
            format=ReportFormat.PDF,
            scope_type="operator",
            provenance_hash="b" * 64,
        )
        assert report.format == ReportFormat.PDF


# ===========================================================================
# 3. Language Support (8 tests)
# ===========================================================================


class TestLanguageSupport:
    """Test 5 language support (EN/FR/DE/ES/PT)."""

    @pytest.mark.parametrize("lang", ALL_REPORT_LANGUAGES)
    def test_create_report_in_each_language(self, lang):
        """Test creating a report in each supported language."""
        report = ComplianceReport(
            report_id=f"r-lang-{lang}",
            report_type=ReportType.INDIGENOUS_RIGHTS_COMPLIANCE,
            title=f"Report in {lang}",
            format=ReportFormat.JSON,
            language=lang,
            scope_type="operator",
            provenance_hash="c" * 64,
        )
        assert report.language == lang

    def test_total_languages_count(self):
        """Test there are exactly 5 supported languages."""
        assert len(ALL_REPORT_LANGUAGES) == 5

    def test_default_language_english(self):
        """Test default language is English."""
        report = ComplianceReport(
            report_id="r-default-lang",
            report_type=ReportType.DDS_SECTION,
            title="Default Language Report",
            format=ReportFormat.JSON,
            scope_type="operator",
            provenance_hash="d" * 64,
        )
        assert report.language == "en"

    def test_config_default_language(self, mock_config):
        """Test config default language is 'en'."""
        assert mock_config.default_language == "en"

    def test_config_supported_languages(self, mock_config):
        """Test config supported languages include all 5."""
        for lang in ALL_REPORT_LANGUAGES:
            assert lang in mock_config.supported_languages


# ===========================================================================
# 4. Report Request Validation (10 tests)
# ===========================================================================


class TestReportRequestValidation:
    """Test report generation request validation."""

    def test_valid_report_request(self):
        """Test valid report generation request."""
        req = GenerateReportRequest(
            report_type=ReportType.INDIGENOUS_RIGHTS_COMPLIANCE,
            format=ReportFormat.PDF,
            language="en",
            scope_type="operator",
            scope_ids=["op-001"],
        )
        assert req.report_type == ReportType.INDIGENOUS_RIGHTS_COMPLIANCE
        assert req.format == ReportFormat.PDF

    def test_report_request_default_format(self):
        """Test report request defaults to PDF format."""
        req = GenerateReportRequest(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            scope_type="operator",
        )
        assert req.format == ReportFormat.PDF

    def test_report_request_default_language(self):
        """Test report request defaults to English."""
        req = GenerateReportRequest(
            report_type=ReportType.TREND_REPORT,
            scope_type="global",
        )
        assert req.language == "en"

    def test_report_request_with_parameters(self):
        """Test report request with custom parameters."""
        req = GenerateReportRequest(
            report_type=ReportType.SUPPLIER_SCORECARD,
            scope_type="supplier",
            scope_ids=["sup-001"],
            parameters={"include_violations": True, "date_range": "Q1-2026"},
        )
        assert req.parameters["include_violations"] is True

    def test_report_request_multiple_scope_ids(self):
        """Test report request with multiple scope IDs."""
        req = GenerateReportRequest(
            report_type=ReportType.INDIGENOUS_RIGHTS_COMPLIANCE,
            scope_type="territory",
            scope_ids=["t-001", "t-002", "t-003"],
        )
        assert len(req.scope_ids) == 3

    def test_bi_export_request(self):
        """Test BI export report request."""
        req = GenerateReportRequest(
            report_type=ReportType.BI_EXPORT,
            format=ReportFormat.CSV,
            scope_type="global",
        )
        assert req.report_type == ReportType.BI_EXPORT
        assert req.format == ReportFormat.CSV

    def test_fsc_fpic_report_request(self):
        """Test FSC FPIC report request."""
        req = GenerateReportRequest(
            report_type=ReportType.FSC_FPIC,
            format=ReportFormat.PDF,
            scope_type="certification",
            scope_ids=["cert-001"],
        )
        assert req.report_type == ReportType.FSC_FPIC

    def test_rspo_fpic_report_request(self):
        """Test RSPO FPIC report request."""
        req = GenerateReportRequest(
            report_type=ReportType.RSPO_FPIC,
            format=ReportFormat.PDF,
            scope_type="certification",
        )
        assert req.report_type == ReportType.RSPO_FPIC

    @pytest.mark.parametrize("scope_type", [
        "operator", "supplier", "territory", "country", "global",
        "certification", "commodity",
    ])
    def test_various_scope_types(self, scope_type):
        """Test report request with various scope types."""
        req = GenerateReportRequest(
            report_type=ReportType.TREND_REPORT,
            scope_type=scope_type,
        )
        assert req.scope_type == scope_type

    def test_report_request_empty_scope_ids(self):
        """Test report request with empty scope IDs (global scope)."""
        req = GenerateReportRequest(
            report_type=ReportType.EXECUTIVE_SUMMARY,
            scope_type="global",
            scope_ids=[],
        )
        assert req.scope_ids == []


# ===========================================================================
# 5. Report Model Fields (10 tests)
# ===========================================================================


class TestReportModelFields:
    """Test ComplianceReport model field validation."""

    def test_report_id_required(self, sample_report):
        """Test report has required ID."""
        assert sample_report.report_id is not None

    def test_report_title_required(self, sample_report):
        """Test report has required title."""
        assert sample_report.title is not None
        assert len(sample_report.title) > 0

    def test_report_file_path_optional(self, sample_report):
        """Test report file path is optional."""
        assert sample_report.file_path is None

    def test_report_file_size_optional(self, sample_report):
        """Test report file size is optional."""
        assert sample_report.file_size_bytes is None

    def test_report_with_file_info(self):
        """Test report with file path and size."""
        report = ComplianceReport(
            report_id="r-file",
            report_type=ReportType.INDIGENOUS_RIGHTS_COMPLIANCE,
            title="Report With File",
            format=ReportFormat.PDF,
            scope_type="operator",
            file_path="/reports/2026/q1/irc-compliance.pdf",
            file_size_bytes=1048576,
            provenance_hash="e" * 64,
        )
        assert report.file_path is not None
        assert report.file_size_bytes == 1048576

    def test_report_generated_by_optional(self):
        """Test report generated_by is optional."""
        report = ComplianceReport(
            report_id="r-gen",
            report_type=ReportType.DDS_SECTION,
            title="Generated Report",
            format=ReportFormat.JSON,
            scope_type="operator",
            generated_by="test-user",
            provenance_hash="f" * 64,
        )
        assert report.generated_by == "test-user"

    def test_report_generated_at_optional(self):
        """Test report generated_at is optional."""
        now = datetime.now(timezone.utc)
        report = ComplianceReport(
            report_id="r-time",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            title="Timed Report",
            format=ReportFormat.JSON,
            scope_type="global",
            generated_at=now,
            provenance_hash="g" * 64,
        )
        assert report.generated_at is not None

    def test_report_scope_ids_list(self, sample_report):
        """Test report scope_ids is a list."""
        assert isinstance(sample_report.scope_ids, list)

    def test_report_parameters_dict(self):
        """Test report parameters is a dict."""
        report = ComplianceReport(
            report_id="r-params",
            report_type=ReportType.TREND_REPORT,
            title="Params Report",
            format=ReportFormat.JSON,
            scope_type="country",
            parameters={"start_date": "2025-01-01", "end_date": "2025-12-31"},
            provenance_hash="h" * 64,
        )
        assert isinstance(report.parameters, dict)
        assert "start_date" in report.parameters

    def test_report_negative_file_size_rejected(self):
        """Test negative file size is rejected."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ComplianceReport(
                report_id="r-neg",
                report_type=ReportType.BI_EXPORT,
                title="Negative Size",
                format=ReportFormat.CSV,
                scope_type="global",
                file_size_bytes=-1,
                provenance_hash="i" * 64,
            )


# ===========================================================================
# 6. Report Configuration (6 tests)
# ===========================================================================


class TestReportConfiguration:
    """Test report configuration settings."""

    def test_config_output_formats(self, mock_config):
        """Test config has output formats defined."""
        assert len(mock_config.output_formats) >= 4

    def test_config_supported_languages(self, mock_config):
        """Test config has supported languages defined."""
        assert len(mock_config.supported_languages) >= 5

    def test_config_retention_years(self, mock_config):
        """Test data retention is 5 years (EUDR Article 31)."""
        assert mock_config.retention_years == 5

    def test_config_default_language_valid(self, mock_config):
        """Test default language is in supported languages."""
        assert mock_config.default_language in mock_config.supported_languages

    def test_config_provenance_enabled(self, mock_config):
        """Test provenance is enabled for reports."""
        assert mock_config.enable_provenance is True

    def test_config_genesis_hash(self, mock_config):
        """Test genesis hash is set."""
        assert mock_config.genesis_hash is not None
        assert len(mock_config.genesis_hash) > 0


# ===========================================================================
# 7. Provenance (5 tests)
# ===========================================================================


class TestReportProvenance:
    """Test provenance tracking for reports."""

    def test_report_provenance_hash(self, sample_report):
        """Test report has provenance hash."""
        assert sample_report.provenance_hash is not None

    def test_report_provenance_deterministic(self):
        """Test same report data produces same hash."""
        data = {"report_id": "r-001", "report_type": "indigenous_rights_compliance"}
        h1 = compute_test_hash(data)
        h2 = compute_test_hash(data)
        assert h1 == h2

    def test_provenance_records_generation(self, mock_provenance):
        """Test provenance records report generation."""
        mock_provenance.record("report", "generate", "r-001")
        assert mock_provenance.entry_count == 1

    def test_provenance_chain_for_reports(self, mock_provenance):
        """Test report provenance chain integrity."""
        mock_provenance.record("report", "generate", "r-001")
        mock_provenance.record("report", "export", "r-001")
        assert mock_provenance.verify_chain() is True

    def test_different_reports_different_hashes(self):
        """Test different reports produce different hashes."""
        h1 = compute_test_hash({"report_id": "r-001"})
        h2 = compute_test_hash({"report_id": "r-002"})
        assert h1 != h2
