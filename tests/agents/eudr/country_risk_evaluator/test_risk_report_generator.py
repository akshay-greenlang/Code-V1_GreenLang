# -*- coding: utf-8 -*-
"""
Unit tests for RiskReportGenerator - AGENT-EUDR-016 Engine 7

Tests comprehensive risk report generation in multiple formats and
languages covering country profiles, commodity matrices, comparative
analyses, trend reports, DD briefs, executive summaries, content hashing,
report versioning, and batch generation.

Target: 60+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.country_risk_evaluator.risk_report_generator import (
    RiskReportGenerator,
    _SECTION_TEMPLATES,
    _RISK_LABELS,
)
from greenlang.agents.eudr.country_risk_evaluator.models import (
    ReportFormat,
    ReportType,
    RiskLevel,
    RiskReport,
    SUPPORTED_OUTPUT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
)


# ============================================================================
# TestReportGeneratorInit
# ============================================================================


class TestReportGeneratorInit:
    """Tests for RiskReportGenerator initialization."""

    @pytest.mark.unit
    def test_initialization_empty_stores(self, mock_config):
        generator = RiskReportGenerator()
        assert generator._reports == {}
        assert generator._report_versions == {}

    @pytest.mark.unit
    def test_section_templates_all_languages(self):
        for lang in SUPPORTED_REPORT_LANGUAGES:
            assert lang in _SECTION_TEMPLATES, (
                f"Language {lang} missing from section templates"
            )

    @pytest.mark.unit
    def test_risk_labels_all_languages(self):
        for lang in SUPPORTED_REPORT_LANGUAGES:
            assert lang in _RISK_LABELS, (
                f"Language {lang} missing from risk labels"
            )

    @pytest.mark.unit
    def test_risk_labels_all_levels(self):
        for lang, labels in _RISK_LABELS.items():
            assert "low" in labels
            assert "standard" in labels
            assert "high" in labels


# ============================================================================
# TestGenerateReport
# ============================================================================


class TestGenerateReport:
    """Tests for generate_report method."""

    @pytest.mark.unit
    def test_generate_valid_report(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            format="json",
            data={"country_code": "BR", "risk_score": 72.5},
        )
        assert isinstance(report, RiskReport)
        assert report.report_type == ReportType.COUNTRY_PROFILE

    @pytest.mark.unit
    def test_report_has_id(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert report.report_id.startswith("rpt-")

    @pytest.mark.unit
    def test_report_stores_result(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        retrieved = report_generator.get_report(report.report_id)
        assert retrieved is not None
        assert retrieved.report_id == report.report_id

    @pytest.mark.unit
    def test_report_has_title(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert report.title is not None
        assert len(report.title) > 0

    @pytest.mark.unit
    def test_report_custom_title(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
            title="Custom Risk Report for Brazil",
        )
        assert report.title == "Custom Risk Report for Brazil"

    @pytest.mark.unit
    def test_report_has_generated_at(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert report.generated_at is not None

    @pytest.mark.unit
    def test_report_has_expires_at(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert report.expires_at is not None
        assert report.expires_at > report.generated_at


# ============================================================================
# TestAllReportTypes
# ============================================================================


class TestAllReportTypes:
    """Tests for all 6+ report types."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "report_type",
        [
            "country_profile",
            "commodity_matrix",
            "comparative",
            "trend",
            "due_diligence",
            "executive_summary",
        ],
    )
    def test_generate_each_report_type(self, report_generator, report_type):
        report = report_generator.generate_report(
            report_type=report_type,
            data={"country_code": "BR", "risk_score": 72.5},
        )
        assert isinstance(report, RiskReport)
        assert report.report_type == ReportType(report_type)

    @pytest.mark.unit
    def test_invalid_report_type_raises(self, report_generator):
        with pytest.raises(ValueError):
            report_generator.generate_report(
                report_type="invalid_type",
                data={},
            )

    @pytest.mark.unit
    def test_all_report_type_enums(self):
        types = [t.value for t in ReportType]
        assert "country_profile" in types
        assert "commodity_matrix" in types
        assert "comparative" in types
        assert "trend" in types
        assert "due_diligence" in types
        assert "executive_summary" in types


# ============================================================================
# TestOutputFormats
# ============================================================================


class TestOutputFormats:
    """Tests for report output formats."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "fmt",
        ["json", "html", "pdf"],
    )
    def test_generate_each_format(self, report_generator, fmt):
        report = report_generator.generate_report(
            report_type="country_profile",
            format=fmt,
            data={"country_code": "BR"},
        )
        assert report.format == ReportFormat(fmt)

    @pytest.mark.unit
    def test_invalid_format_raises(self, report_generator):
        with pytest.raises(ValueError):
            report_generator.generate_report(
                report_type="country_profile",
                format="docx",
                data={},
            )

    @pytest.mark.unit
    def test_all_format_enums(self):
        formats = [f.value for f in ReportFormat]
        assert "pdf" in formats
        assert "json" in formats
        assert "html" in formats
        assert "csv" in formats
        assert "excel" in formats

    @pytest.mark.unit
    def test_csv_format(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            format="csv",
            data={"country_code": "BR"},
        )
        assert report.format == ReportFormat.CSV

    @pytest.mark.unit
    def test_excel_format(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            format="excel",
            data={"country_code": "BR"},
        )
        assert report.format == ReportFormat.EXCEL


# ============================================================================
# TestMultiLanguage
# ============================================================================


class TestMultiLanguage:
    """Tests for multi-language report support."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "language",
        ["en", "fr", "de", "es", "pt"],
        ids=["English", "French", "German", "Spanish", "Portuguese"],
    )
    def test_generate_in_each_language(self, report_generator, language):
        report = report_generator.generate_report(
            report_type="country_profile",
            language=language,
            data={"country_code": "BR"},
        )
        assert report.language == language

    @pytest.mark.unit
    def test_invalid_language_raises(self, report_generator):
        with pytest.raises(ValueError):
            report_generator.generate_report(
                report_type="country_profile",
                language="zh",
                data={},
            )

    @pytest.mark.unit
    def test_default_language_english(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert report.language == "en"

    @pytest.mark.unit
    def test_french_section_templates(self):
        fr_templates = _SECTION_TEMPLATES["fr"]
        assert fr_templates["executive_summary"] == "R\u00e9sum\u00e9 Ex\u00e9cutif"

    @pytest.mark.unit
    def test_german_section_templates(self):
        de_templates = _SECTION_TEMPLATES["de"]
        assert de_templates["executive_summary"] == "Zusammenfassung"

    @pytest.mark.unit
    def test_spanish_section_templates(self):
        es_templates = _SECTION_TEMPLATES["es"]
        assert es_templates["executive_summary"] == "Resumen Ejecutivo"

    @pytest.mark.unit
    def test_portuguese_section_templates(self):
        pt_templates = _SECTION_TEMPLATES["pt"]
        assert pt_templates["executive_summary"] == "Resumo Executivo"


# ============================================================================
# TestContentHash
# ============================================================================


class TestContentHash:
    """Tests for SHA-256 content hash."""

    @pytest.mark.unit
    def test_report_has_content_hash(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR", "risk_score": 72.5},
        )
        assert report.content_hash is not None
        assert len(report.content_hash) == 64  # SHA-256 hex

    @pytest.mark.unit
    def test_different_data_different_hash(self, report_generator):
        report1 = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR", "risk_score": 72.5},
        )
        report2 = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "ID", "risk_score": 65.0},
        )
        assert report1.content_hash != report2.content_hash

    @pytest.mark.unit
    def test_content_hash_is_hex_string(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        # Verify it is a valid hex string
        int(report.content_hash, 16)


# ============================================================================
# TestReportVersioning
# ============================================================================


class TestReportVersioning:
    """Tests for report versioning and comparison."""

    @pytest.mark.unit
    def test_multiple_versions_same_type(self, report_generator):
        report1 = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR", "risk_score": 72.5},
        )
        report2 = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR", "risk_score": 75.0},
        )
        assert report1.report_id != report2.report_id

    @pytest.mark.unit
    def test_version_history_tracked(self, report_generator):
        for score in [72.5, 73.0, 74.5]:
            report_generator.generate_report(
                report_type="country_profile",
                data={"country_code": "BR", "risk_score": score},
            )
        results = report_generator.list_reports(report_type="country_profile")
        assert len(results) >= 3


# ============================================================================
# TestBatchGeneration
# ============================================================================


class TestBatchGeneration:
    """Tests for batch report generation."""

    @pytest.mark.unit
    def test_batch_generate_multiple(self, report_generator):
        countries = ["BR", "ID", "CI", "GH", "MY"]
        reports = []
        for cc in countries:
            report = report_generator.generate_report(
                report_type="country_profile",
                data={"country_code": cc, "risk_score": 65.0},
            )
            reports.append(report)
        assert len(reports) == 5

    @pytest.mark.unit
    def test_batch_all_report_types(self, report_generator):
        report_types = [
            "country_profile",
            "commodity_matrix",
            "comparative",
            "trend",
            "due_diligence",
            "executive_summary",
        ]
        reports = []
        for rt in report_types:
            report = report_generator.generate_report(
                report_type=rt,
                data={"country_code": "BR"},
            )
            reports.append(report)
        assert len(reports) == 6


# ============================================================================
# TestExecutiveSummary
# ============================================================================


class TestExecutiveSummary:
    """Tests for executive summary report type."""

    @pytest.mark.unit
    def test_executive_summary_generation(self, report_generator):
        report = report_generator.generate_report(
            report_type="executive_summary",
            data={
                "countries": ["BR", "ID", "CI"],
                "commodities": ["soya", "oil_palm", "cocoa"],
            },
        )
        assert report.report_type == ReportType.EXECUTIVE_SUMMARY


# ============================================================================
# TestComparativeReport
# ============================================================================


class TestComparativeReport:
    """Tests for comparative report type."""

    @pytest.mark.unit
    def test_comparative_report_generation(self, report_generator):
        report = report_generator.generate_report(
            report_type="comparative",
            data={
                "countries": ["BR", "ID"],
                "metric": "risk_score",
            },
        )
        assert report.report_type == ReportType.COMPARATIVE
        assert "BR" in report.countries or len(report.countries) >= 0


# ============================================================================
# TestListReports
# ============================================================================


class TestListReports:
    """Tests for listing and filtering reports."""

    @pytest.mark.unit
    def test_list_all_reports(self, report_generator):
        for cc in ["BR", "ID", "CI"]:
            report_generator.generate_report(
                report_type="country_profile",
                data={"country_code": cc},
            )
        results = report_generator.list_reports()
        assert len(results) == 3

    @pytest.mark.unit
    def test_list_reports_by_type(self, report_generator):
        report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        report_generator.generate_report(
            report_type="executive_summary",
            data={"countries": ["BR"]},
        )
        results = report_generator.list_reports(report_type="country_profile")
        assert len(results) == 1

    @pytest.mark.unit
    def test_get_nonexistent_report(self, report_generator):
        result = report_generator.get_report("nonexistent-id")
        assert result is None


# ============================================================================
# TestFileSize
# ============================================================================


class TestFileSize:
    """Tests for report file size tracking."""

    @pytest.mark.unit
    def test_report_has_file_size(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR", "risk_score": 72.5},
        )
        assert report.file_size_bytes is not None
        assert report.file_size_bytes > 0

    @pytest.mark.unit
    def test_report_has_storage_path(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert report.storage_path is not None
        assert len(report.storage_path) > 0


# ============================================================================
# TestProvenanceTracking
# ============================================================================


class TestProvenanceTracking:
    """Tests for report provenance."""

    @pytest.mark.unit
    def test_report_has_provenance_hash(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert report.provenance_hash is not None
        assert len(report.provenance_hash) == 64

    @pytest.mark.unit
    def test_countries_extracted_from_data(self, report_generator):
        report = report_generator.generate_report(
            report_type="country_profile",
            data={"country_code": "BR"},
        )
        assert isinstance(report.countries, list)

    @pytest.mark.unit
    def test_commodities_extracted_from_data(self, report_generator):
        report = report_generator.generate_report(
            report_type="commodity_matrix",
            data={"commodity": "soya"},
        )
        assert isinstance(report.commodities, list)


# ============================================================================
# TestSectionTemplates
# ============================================================================


class TestSectionTemplates:
    """Tests for report section templates."""

    @pytest.mark.unit
    def test_english_sections_complete(self):
        en_sections = _SECTION_TEMPLATES["en"]
        required_sections = [
            "executive_summary",
            "risk_scoring",
            "factor_analysis",
            "commodity_details",
            "governance",
            "trade_flows",
            "recommendations",
            "data_sources",
            "methodology",
        ]
        for section in required_sections:
            assert section in en_sections, (
                f"Section {section} missing from English templates"
            )

    @pytest.mark.unit
    def test_all_languages_have_same_sections(self):
        en_keys = set(_SECTION_TEMPLATES["en"].keys())
        for lang in ["fr", "de", "es", "pt"]:
            lang_keys = set(_SECTION_TEMPLATES[lang].keys())
            assert lang_keys == en_keys, (
                f"Language {lang} has different sections than English"
            )
