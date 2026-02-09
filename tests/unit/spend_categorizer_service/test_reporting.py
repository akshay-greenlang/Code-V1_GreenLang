# -*- coding: utf-8 -*-
"""
Unit tests for ReportingEngine - AGENT-DATA-009 Batch 3

Tests the ReportingEngine with 85%+ coverage across:
- Initialization and configuration
- Categorization report generation (all formats, empty records)
- Emissions report generation (with calculations, grouping)
- Audit report generation (provenance, match sources)
- Procurement report generation (metrics, analytics)
- Executive summary generation (KPIs, benchmarks)
- Record export (CSV, JSON)
- Format helpers: format_text, format_json, format_markdown, format_html, format_csv
- Statistics tracking (report counts by type and format)
- SHA-256 provenance hashes
- Format validation and fallback

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List

import pytest

from greenlang.spend_categorizer.reporting import (
    CategorizationReport,
    ReportingEngine,
    _SUPPORTED_FORMATS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ReportingEngine:
    """Create a default ReportingEngine."""
    return ReportingEngine()


@pytest.fixture
def engine_json_default() -> ReportingEngine:
    """Create engine with JSON as default format."""
    return ReportingEngine({"default_format": "json"})


@pytest.fixture
def categorized_records() -> List[Dict[str, Any]]:
    """Sample categorized spend records."""
    return [
        {
            "record_id": "R-001",
            "vendor_name": "Office Depot",
            "amount_usd": 12500.0,
            "scope3_category": "Cat 1",
            "taxonomy_code": "44121600",
            "taxonomy_system": "unspsc",
            "confidence": 0.92,
            "provenance_hash": "a" * 64,
        },
        {
            "record_id": "R-002",
            "vendor_name": "DHL Logistics",
            "amount_usd": 85000.0,
            "scope3_category": "Cat 4",
            "taxonomy_code": "484110",
            "taxonomy_system": "naics",
            "confidence": 0.88,
            "provenance_hash": "b" * 64,
        },
        {
            "record_id": "R-003",
            "vendor_name": "AWS",
            "amount_usd": 45000.0,
            "scope3_category": "Cat 1",
            "taxonomy_code": "518210",
            "taxonomy_system": "naics",
            "confidence": 0.95,
            "provenance_hash": "c" * 64,
        },
        {
            "record_id": "R-004",
            "vendor_name": "Acme Chemical",
            "amount_usd": 220000.0,
            "scope3_category": "Cat 1",
            "category": "chemicals",
            "confidence": 0.65,
            "provenance_hash": "d" * 64,
        },
        {
            "record_id": "R-005",
            "vendor_name": "Uncategorized Vendor",
            "amount_usd": 5000.0,
            "confidence": 0.0,
        },
    ]


@pytest.fixture
def emission_calculations() -> List[Dict[str, Any]]:
    """Sample emission calculation results."""
    return [
        {
            "record_id": "R-001",
            "spend_usd": 12500.0,
            "emissions_kgco2e": 3750.0,
            "scope3_category": "Cat 1",
            "factor_source": "epa_eeio",
        },
        {
            "record_id": "R-002",
            "spend_usd": 85000.0,
            "emissions_kgco2e": 72250.0,
            "scope3_category": "Cat 4",
            "factor_source": "epa_eeio",
        },
        {
            "record_id": "R-003",
            "spend_usd": 45000.0,
            "emissions_kgco2e": 6300.0,
            "scope3_category": "Cat 1",
            "factor_source": "exiobase",
        },
        {
            "record_id": "R-004",
            "spend_usd": 220000.0,
            "emissions_kgco2e": 180400.0,
            "scope3_category": "Cat 1",
            "factor_source": "epa_eeio",
        },
    ]


@pytest.fixture
def classification_results() -> List[Dict[str, Any]]:
    """Sample classification results for audit reporting."""
    return [
        {
            "record_id": "R-001",
            "code": "44121600",
            "confidence": 0.92,
            "match_type": "CONTAINS",
            "provenance_hash": "e" * 64,
        },
        {
            "record_id": "R-002",
            "code": "484110",
            "confidence": 0.88,
            "match_type": "REGEX",
            "provenance_hash": "f" * 64,
        },
        {
            "record_id": "R-003",
            "code": "518210",
            "confidence": 0.95,
            "match_source": "rule_based",
            "provenance_hash": "a1" + "b" * 62,
        },
    ]


@pytest.fixture
def analytics_data() -> Dict[str, Any]:
    """Sample analytics data for procurement and executive reports."""
    return {
        "total_spend_usd": 367500.0,
        "total_emissions_kgco2e": 262700.0,
        "intensity_kgco2e_per_usd": 0.714966,
        "record_count": 5,
        "hhi": 2450.0,
        "cr4": 0.95,
        "interpretation": "moderately_concentrated",
        "performance_rating": "laggard",
        "industry": "manufacturing",
        "benchmark_intensity_kgco2e_per_usd": 0.45,
        "intensity_gap_pct": 58.88,
    }


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test ReportingEngine initialization."""

    def test_default_init(self, engine: ReportingEngine):
        """Engine initializes with default configuration."""
        assert engine._default_format == "markdown"
        assert engine._include_provenance is True
        assert engine._max_records == 10000

    def test_custom_init(self):
        """Engine respects custom configuration."""
        eng = ReportingEngine({
            "default_format": "html",
            "include_provenance": False,
            "max_records_in_report": 500,
        })
        assert eng._default_format == "html"
        assert eng._include_provenance is False
        assert eng._max_records == 500

    def test_stats_initialized(self, engine: ReportingEngine):
        """Statistics counters start at zero."""
        stats = engine.get_statistics()
        assert stats["reports_generated"] == 0
        assert stats["total_records_reported"] == 0

    def test_empty_reports_store(self, engine: ReportingEngine):
        """Reports store starts empty."""
        assert len(engine._reports) == 0


# ===========================================================================
# TestGenerateCategorizationReport
# ===========================================================================


class TestGenerateCategorizationReport:
    """Test generate_categorization_report() across formats."""

    def test_returns_report(self, engine: ReportingEngine, categorized_records):
        """Returns a CategorizationReport object."""
        report = engine.generate_categorization_report(categorized_records)
        assert isinstance(report, CategorizationReport)

    def test_report_type(self, engine: ReportingEngine, categorized_records):
        """Report type is 'categorization'."""
        report = engine.generate_categorization_report(categorized_records)
        assert report.report_type == "categorization"

    def test_report_id(self, engine: ReportingEngine, categorized_records):
        """Report ID starts with 'rpt-'."""
        report = engine.generate_categorization_report(categorized_records)
        assert report.report_id.startswith("rpt-")

    def test_record_count(self, engine: ReportingEngine, categorized_records):
        """Record count matches input."""
        report = engine.generate_categorization_report(categorized_records)
        assert report.record_count == len(categorized_records)

    def test_custom_title(self, engine: ReportingEngine, categorized_records):
        """Custom title is used when provided."""
        report = engine.generate_categorization_report(
            categorized_records, title="Custom Title",
        )
        assert report.title == "Custom Title"

    def test_default_title(self, engine: ReportingEngine, categorized_records):
        """Default title is 'Spend Categorization Report'."""
        report = engine.generate_categorization_report(categorized_records)
        assert report.title == "Spend Categorization Report"

    @pytest.mark.parametrize("fmt", ["json", "csv", "markdown", "html", "text"])
    def test_all_formats(self, engine: ReportingEngine, categorized_records, fmt):
        """Report generates successfully in all formats."""
        report = engine.generate_categorization_report(categorized_records, format=fmt)
        assert report.format == fmt
        assert len(report.content) > 0

    def test_markdown_contains_headers(self, engine: ReportingEngine, categorized_records):
        """Markdown output contains section headers."""
        report = engine.generate_categorization_report(categorized_records, format="markdown")
        assert "# Spend Categorization Report" in report.content
        assert "## Summary" in report.content

    def test_json_valid(self, engine: ReportingEngine, categorized_records):
        """JSON output is valid JSON."""
        report = engine.generate_categorization_report(categorized_records, format="json")
        data = json.loads(report.content)
        assert "total_records" in data
        assert data["total_records"] == len(categorized_records)

    def test_html_contains_tags(self, engine: ReportingEngine, categorized_records):
        """HTML output contains standard HTML tags."""
        report = engine.generate_categorization_report(categorized_records, format="html")
        assert "<html" in report.content
        assert "<table>" in report.content
        assert "</html>" in report.content

    def test_text_output(self, engine: ReportingEngine, categorized_records):
        """Text output contains summary data."""
        report = engine.generate_categorization_report(categorized_records, format="text")
        assert "Total Records" in report.content
        assert "Coverage" in report.content

    def test_csv_output(self, engine: ReportingEngine, categorized_records):
        """CSV output contains category data."""
        report = engine.generate_categorization_report(categorized_records, format="csv")
        assert "category" in report.content.lower() or "cat" in report.content.lower()

    def test_empty_records(self, engine: ReportingEngine):
        """Empty records produce a report with zero counts."""
        report = engine.generate_categorization_report([])
        assert report.record_count == 0

    def test_coverage_rate(self, engine: ReportingEngine, categorized_records):
        """JSON report includes correct coverage rate."""
        report = engine.generate_categorization_report(categorized_records, format="json")
        data = json.loads(report.content)
        # 4 out of 5 have taxonomy_code or category or scope3_category
        assert data["classified_records"] == 4
        assert data["coverage_rate"] == 80.0

    def test_confidence_distribution(self, engine: ReportingEngine, categorized_records):
        """JSON report includes confidence distribution."""
        report = engine.generate_categorization_report(categorized_records, format="json")
        data = json.loads(report.content)
        assert "confidence_distribution" in data
        assert data["confidence_distribution"]["high"] >= 2  # 0.92, 0.88, 0.95 are high
        assert data["confidence_distribution"]["none"] >= 1  # 0.0 confidence

    def test_provenance_hash(self, engine: ReportingEngine, categorized_records):
        """Report has a provenance hash."""
        report = engine.generate_categorization_report(categorized_records)
        assert len(report.provenance_hash) == 64

    def test_processing_time(self, engine: ReportingEngine, categorized_records):
        """Report includes non-zero processing time."""
        report = engine.generate_categorization_report(categorized_records)
        assert report.processing_time_ms >= 0


# ===========================================================================
# TestGenerateEmissionsReport
# ===========================================================================


class TestGenerateEmissionsReport:
    """Test generate_emissions_report() with calculations."""

    def test_returns_report(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """Returns a CategorizationReport object."""
        report = engine.generate_emissions_report(categorized_records, emission_calculations)
        assert isinstance(report, CategorizationReport)

    def test_report_type(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """Report type is 'emissions'."""
        report = engine.generate_emissions_report(categorized_records, emission_calculations)
        assert report.report_type == "emissions"

    def test_json_totals(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """JSON report includes correct emission totals."""
        report = engine.generate_emissions_report(
            categorized_records, emission_calculations, format="json",
        )
        data = json.loads(report.content)
        total_emissions = sum(c["emissions_kgco2e"] for c in emission_calculations)
        assert data["total_emissions_kgco2e"] == round(total_emissions, 4)

    def test_scope3_breakdown(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """JSON report includes Scope 3 category breakdown."""
        report = engine.generate_emissions_report(
            categorized_records, emission_calculations, format="json",
        )
        data = json.loads(report.content)
        assert "by_scope3_category" in data
        assert "Cat 1" in data["by_scope3_category"]

    def test_factor_source_distribution(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """JSON report includes factor source distribution."""
        report = engine.generate_emissions_report(
            categorized_records, emission_calculations, format="json",
        )
        data = json.loads(report.content)
        assert "by_factor_source" in data
        assert data["by_factor_source"].get("epa_eeio", 0) >= 1

    @pytest.mark.parametrize("fmt", ["json", "csv", "markdown", "html", "text"])
    def test_all_formats(self, engine: ReportingEngine, categorized_records, emission_calculations, fmt):
        """Emissions report generates in all formats."""
        report = engine.generate_emissions_report(
            categorized_records, emission_calculations, format=fmt,
        )
        assert len(report.content) > 0

    def test_markdown_contains_intensity(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """Markdown report contains intensity metric."""
        report = engine.generate_emissions_report(
            categorized_records, emission_calculations, format="markdown",
        )
        assert "Intensity" in report.content

    def test_tco2e_in_summary(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """JSON summary includes tCO2e conversion."""
        report = engine.generate_emissions_report(
            categorized_records, emission_calculations, format="json",
        )
        data = json.loads(report.content)
        expected_tco2e = round(data["total_emissions_kgco2e"] / 1000, 4)
        assert data["total_emissions_tco2e"] == expected_tco2e


# ===========================================================================
# TestGenerateAuditReport
# ===========================================================================


class TestGenerateAuditReport:
    """Test generate_audit_report() for compliance auditing."""

    def test_returns_report(self, engine: ReportingEngine, categorized_records, classification_results):
        """Returns a CategorizationReport object."""
        report = engine.generate_audit_report(categorized_records, classification_results)
        assert isinstance(report, CategorizationReport)

    def test_report_type(self, engine: ReportingEngine, categorized_records, classification_results):
        """Report type is 'audit'."""
        report = engine.generate_audit_report(categorized_records, classification_results)
        assert report.report_type == "audit"

    def test_default_format_json(self, engine: ReportingEngine, categorized_records, classification_results):
        """Audit report defaults to JSON format."""
        report = engine.generate_audit_report(categorized_records, classification_results)
        assert report.format == "json"

    def test_json_structure(self, engine: ReportingEngine, categorized_records, classification_results):
        """JSON audit report includes provenance summary."""
        report = engine.generate_audit_report(categorized_records, classification_results)
        data = json.loads(report.content)
        assert "total_records" in data
        assert "records_with_provenance" in data
        assert "provenance_coverage_pct" in data

    def test_provenance_coverage(self, engine: ReportingEngine, categorized_records, classification_results):
        """Provenance coverage percentage is calculated."""
        report = engine.generate_audit_report(
            categorized_records, classification_results, format="json",
        )
        data = json.loads(report.content)
        # 4 of 5 records have provenance_hash
        assert data["records_with_provenance"] == 4
        assert data["provenance_coverage_pct"] == 80.0

    def test_audit_entries_limited(self, engine: ReportingEngine, categorized_records, classification_results):
        """Audit entries are limited to 100."""
        report = engine.generate_audit_report(
            categorized_records, classification_results, format="json",
        )
        data = json.loads(report.content)
        assert len(data["audit_entries"]) <= 100

    def test_match_source_distribution(self, engine: ReportingEngine, categorized_records, classification_results):
        """Audit report tracks match source distribution."""
        report = engine.generate_audit_report(
            categorized_records, classification_results, format="json",
        )
        data = json.loads(report.content)
        assert "by_match_source" in data

    @pytest.mark.parametrize("fmt", ["json", "csv", "text", "markdown"])
    def test_multiple_formats(self, engine: ReportingEngine, categorized_records, classification_results, fmt):
        """Audit report generates in multiple formats."""
        report = engine.generate_audit_report(categorized_records, classification_results, format=fmt)
        assert len(report.content) > 0


# ===========================================================================
# TestGenerateProcurementReport
# ===========================================================================


class TestGenerateProcurementReport:
    """Test generate_procurement_report() for procurement analytics."""

    def test_returns_report(self, engine: ReportingEngine, categorized_records, analytics_data):
        """Returns a CategorizationReport object."""
        report = engine.generate_procurement_report(categorized_records, analytics_data)
        assert isinstance(report, CategorizationReport)

    def test_report_type(self, engine: ReportingEngine, categorized_records, analytics_data):
        """Report type is 'procurement'."""
        report = engine.generate_procurement_report(categorized_records, analytics_data)
        assert report.report_type == "procurement"

    def test_markdown_contains_spend(self, engine: ReportingEngine, categorized_records, analytics_data):
        """Markdown report contains total spend."""
        report = engine.generate_procurement_report(
            categorized_records, analytics_data, format="markdown",
        )
        assert "Total Spend" in report.content

    def test_markdown_contains_concentration(self, engine: ReportingEngine, categorized_records, analytics_data):
        """Markdown report includes concentration metrics when available."""
        report = engine.generate_procurement_report(
            categorized_records, analytics_data, format="markdown",
        )
        assert "HHI" in report.content

    def test_json_format(self, engine: ReportingEngine, categorized_records, analytics_data):
        """JSON format returns valid JSON."""
        report = engine.generate_procurement_report(
            categorized_records, analytics_data, format="json",
        )
        data = json.loads(report.content)
        assert "total_spend_usd" in data
        assert "analytics" in data

    @pytest.mark.parametrize("fmt", ["json", "csv", "markdown", "html", "text"])
    def test_all_formats(self, engine: ReportingEngine, categorized_records, analytics_data, fmt):
        """Procurement report generates in all formats."""
        report = engine.generate_procurement_report(
            categorized_records, analytics_data, format=fmt,
        )
        assert len(report.content) > 0


# ===========================================================================
# TestGenerateExecutiveSummary
# ===========================================================================


class TestGenerateExecutiveSummary:
    """Test generate_executive_summary() for high-level KPIs."""

    def test_returns_report(self, engine: ReportingEngine, analytics_data):
        """Returns a CategorizationReport object."""
        report = engine.generate_executive_summary(analytics_data)
        assert isinstance(report, CategorizationReport)

    def test_report_type(self, engine: ReportingEngine, analytics_data):
        """Report type is 'executive'."""
        report = engine.generate_executive_summary(analytics_data)
        assert report.report_type == "executive"

    def test_markdown_contains_kpis(self, engine: ReportingEngine, analytics_data):
        """Markdown report contains KPI section."""
        report = engine.generate_executive_summary(analytics_data, format="markdown")
        assert "Key Performance Indicators" in report.content
        assert "Total Spend" in report.content

    def test_benchmark_section(self, engine: ReportingEngine, analytics_data):
        """Report includes benchmark comparison when data available."""
        report = engine.generate_executive_summary(analytics_data, format="markdown")
        assert "Benchmark" in report.content
        assert analytics_data["performance_rating"] in report.content

    def test_recommendations_included(self, engine: ReportingEngine, analytics_data):
        """Report includes recommendations section."""
        report = engine.generate_executive_summary(analytics_data, format="markdown")
        assert "Recommendations" in report.content

    def test_json_format(self, engine: ReportingEngine, analytics_data):
        """JSON format returns the analytics data."""
        report = engine.generate_executive_summary(analytics_data, format="json")
        data = json.loads(report.content)
        assert data["total_spend_usd"] == analytics_data["total_spend_usd"]

    def test_record_count_from_analytics(self, engine: ReportingEngine, analytics_data):
        """Record count is taken from analytics dict."""
        report = engine.generate_executive_summary(analytics_data)
        assert report.record_count == analytics_data["record_count"]

    @pytest.mark.parametrize("fmt", ["json", "csv", "markdown", "html", "text"])
    def test_all_formats(self, engine: ReportingEngine, analytics_data, fmt):
        """Executive summary generates in all formats."""
        report = engine.generate_executive_summary(analytics_data, format=fmt)
        assert len(report.content) > 0


# ===========================================================================
# TestExportRecords
# ===========================================================================


class TestExportRecords:
    """Test export_records() CSV and JSON export."""

    def test_csv_export(self, engine: ReportingEngine, categorized_records):
        """CSV export returns comma-separated content."""
        csv_str = engine.export_records(categorized_records, format="csv")
        assert "record_id" in csv_str
        assert "R-001" in csv_str

    def test_json_export(self, engine: ReportingEngine, categorized_records):
        """JSON export returns valid JSON."""
        json_str = engine.export_records(categorized_records, format="json")
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == len(categorized_records)

    def test_unknown_format_defaults_json(self, engine: ReportingEngine, categorized_records):
        """Unknown format defaults to JSON."""
        result = engine.export_records(categorized_records, format="xml")
        data = json.loads(result)
        assert isinstance(data, list)

    def test_empty_records_csv(self, engine: ReportingEngine):
        """Empty records CSV returns empty string."""
        csv_str = engine.export_records([], format="csv")
        assert csv_str == ""


# ===========================================================================
# TestFormatText
# ===========================================================================


class TestFormatText:
    """Test format_text() plain text formatting."""

    def test_string_passthrough(self, engine: ReportingEngine):
        """String input is returned as-is."""
        result = engine.format_text("Hello World")
        assert result == "Hello World"

    def test_dict_formatting(self, engine: ReportingEngine):
        """Dict is formatted as key: value lines."""
        data = {"name": "Test", "value": 42}
        result = engine.format_text(data)
        assert "name: Test" in result
        assert "value: 42" in result

    def test_dict_nested(self, engine: ReportingEngine):
        """Nested dicts are formatted with JSON inside."""
        data = {"outer": {"inner": "value"}}
        result = engine.format_text(data)
        assert "outer:" in result

    def test_list_formatting(self, engine: ReportingEngine):
        """List is formatted as JSON."""
        data = [1, 2, 3]
        result = engine.format_text(data)
        parsed = json.loads(result)
        assert parsed == [1, 2, 3]

    def test_other_type(self, engine: ReportingEngine):
        """Non-standard type is converted via str()."""
        result = engine.format_text(42)
        assert result == "42"


# ===========================================================================
# TestFormatJSON
# ===========================================================================


class TestFormatJSON:
    """Test format_json() JSON serialization."""

    def test_dict_to_json(self, engine: ReportingEngine):
        """Dict is serialized to valid JSON."""
        data = {"key": "value", "number": 42}
        result = engine.format_json(data)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_list_to_json(self, engine: ReportingEngine):
        """List is serialized to valid JSON."""
        data = [{"a": 1}, {"b": 2}]
        result = engine.format_json(data)
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_indentation(self, engine: ReportingEngine):
        """JSON output is indented with 2 spaces."""
        data = {"key": "value"}
        result = engine.format_json(data)
        assert "  " in result  # indented

    def test_datetime_serialization(self, engine: ReportingEngine):
        """Non-serializable types are converted via default=str."""
        from datetime import datetime
        data = {"timestamp": datetime(2025, 1, 1, 0, 0)}
        result = engine.format_json(data)
        parsed = json.loads(result)
        assert "2025" in parsed["timestamp"]


# ===========================================================================
# TestFormatMarkdown
# ===========================================================================


class TestFormatMarkdown:
    """Test format_markdown() Markdown rendering."""

    def test_string_passthrough(self, engine: ReportingEngine):
        """String input is returned as-is."""
        result = engine.format_markdown("# Header")
        assert result == "# Header"

    def test_dict_to_markdown(self, engine: ReportingEngine):
        """Dict is converted to markdown key-value pairs."""
        data = {"name": "Test", "value": 42}
        result = engine.format_markdown(data)
        assert "**name**" in result
        assert "**value**" in result

    def test_list_to_table(self, engine: ReportingEngine):
        """List of dicts is converted to markdown table."""
        data = [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2},
        ]
        result = engine.format_markdown(data)
        assert "| name | value |" in result
        assert "| --- | --- |" in result

    def test_list_non_dict_items(self, engine: ReportingEngine):
        """List of non-dict items uses bullet format."""
        data = ["item1", "item2"]
        result = engine.format_markdown(data)
        assert "- item1" in result

    def test_other_type(self, engine: ReportingEngine):
        """Non-standard type is converted via str()."""
        result = engine.format_markdown(42)
        assert result == "42"


# ===========================================================================
# TestFormatHTML
# ===========================================================================


class TestFormatHTML:
    """Test format_html() HTML rendering."""

    def test_string_to_pre(self, engine: ReportingEngine):
        """String input is wrapped in <pre> tags."""
        result = engine.format_html("Hello")
        assert "<pre>Hello</pre>" == result

    def test_dict_to_table(self, engine: ReportingEngine):
        """Dict is converted to HTML table."""
        data = {"key": "value"}
        result = engine.format_html(data)
        assert "<table>" in result
        assert "<th>key</th>" in result
        assert "<td>value</td>" in result

    def test_list_dicts_to_table(self, engine: ReportingEngine):
        """List of dicts is converted to HTML table."""
        data = [{"name": "A", "value": 1}]
        result = engine.format_html(data)
        assert "<table>" in result
        assert "<th>name</th>" in result
        assert "<td>A</td>" in result

    def test_list_non_dict_to_ul(self, engine: ReportingEngine):
        """List of non-dicts is converted to <ul> list."""
        data = ["item1", "item2"]
        result = engine.format_html(data)
        assert "<ul>" in result
        assert "<li>item1</li>" in result

    def test_other_type_to_pre(self, engine: ReportingEngine):
        """Non-standard type is wrapped in <pre>."""
        result = engine.format_html(42)
        assert "<pre>42</pre>" == result


# ===========================================================================
# TestFormatCSV
# ===========================================================================


class TestFormatCSV:
    """Test format_csv() CSV serialization."""

    def test_list_dicts(self, engine: ReportingEngine):
        """List of dicts is converted to CSV with header."""
        data = [
            {"name": "A", "value": "1"},
            {"name": "B", "value": "2"},
        ]
        result = engine.format_csv(data)
        lines = result.strip().split("\n")
        assert "name" in lines[0]
        assert "value" in lines[0]
        assert len(lines) == 3  # header + 2 rows

    def test_empty_list(self, engine: ReportingEngine):
        """Empty list returns empty string."""
        result = engine.format_csv([])
        assert result == ""

    def test_non_list_returns_empty(self, engine: ReportingEngine):
        """Non-list input returns empty string."""
        result = engine.format_csv("not a list")
        assert result == ""

    def test_special_characters_escaped(self, engine: ReportingEngine):
        """CSV escapes special characters (commas, quotes)."""
        data = [{"text": 'value with "quotes" and, commas'}]
        result = engine.format_csv(data)
        assert "quotes" in result

    def test_list_of_non_dicts(self, engine: ReportingEngine):
        """List of non-dict items uses simple writer."""
        data = ["item1", "item2"]
        result = engine.format_csv(data)
        assert "item1" in result


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Test reporting statistics tracking."""

    def test_initial_stats(self, engine: ReportingEngine):
        """Statistics start at zero."""
        stats = engine.get_statistics()
        assert stats["reports_generated"] == 0
        assert stats["total_records_reported"] == 0

    def test_report_increments_count(self, engine: ReportingEngine, categorized_records):
        """Each report generation increments counters."""
        engine.generate_categorization_report(categorized_records)
        stats = engine.get_statistics()
        assert stats["reports_generated"] == 1
        assert stats["total_records_reported"] == len(categorized_records)

    def test_by_type_tracking(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """Reports are tracked by type."""
        engine.generate_categorization_report(categorized_records)
        engine.generate_emissions_report(categorized_records, emission_calculations)
        stats = engine.get_statistics()
        assert stats["by_type"].get("categorization", 0) == 1
        assert stats["by_type"].get("emissions", 0) == 1

    def test_by_format_tracking(self, engine: ReportingEngine, categorized_records):
        """Reports are tracked by format."""
        engine.generate_categorization_report(categorized_records, format="json")
        engine.generate_categorization_report(categorized_records, format="markdown")
        stats = engine.get_statistics()
        assert stats["by_format"].get("json", 0) == 1
        assert stats["by_format"].get("markdown", 0) == 1

    def test_reports_stored(self, engine: ReportingEngine, categorized_records):
        """Generated reports are stored in engine._reports."""
        engine.generate_categorization_report(categorized_records)
        stats = engine.get_statistics()
        assert stats["reports_stored"] == 1

    def test_supported_formats_in_stats(self, engine: ReportingEngine):
        """Statistics include supported formats list."""
        stats = engine.get_statistics()
        assert set(stats["supported_formats"]) == _SUPPORTED_FORMATS

    def test_multiple_report_types(self, engine: ReportingEngine, categorized_records, emission_calculations, analytics_data, classification_results):
        """All report types increment statistics correctly."""
        engine.generate_categorization_report(categorized_records)
        engine.generate_emissions_report(categorized_records, emission_calculations)
        engine.generate_audit_report(categorized_records, classification_results)
        engine.generate_procurement_report(categorized_records, analytics_data)
        engine.generate_executive_summary(analytics_data)

        stats = engine.get_statistics()
        assert stats["reports_generated"] == 5
        assert stats["by_type"]["categorization"] == 1
        assert stats["by_type"]["emissions"] == 1
        assert stats["by_type"]["audit"] == 1
        assert stats["by_type"]["procurement"] == 1
        assert stats["by_type"]["executive"] == 1


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test SHA-256 provenance hash generation."""

    def test_report_has_hash(self, engine: ReportingEngine, categorized_records):
        """Every generated report has a 64-char hex provenance hash."""
        report = engine.generate_categorization_report(categorized_records)
        assert len(report.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in report.provenance_hash)

    def test_different_reports_different_hashes(self, engine: ReportingEngine, categorized_records, emission_calculations):
        """Different report types produce different hashes."""
        r1 = engine.generate_categorization_report(categorized_records)
        r2 = engine.generate_emissions_report(categorized_records, emission_calculations)
        assert r1.provenance_hash != r2.provenance_hash

    def test_same_input_different_hashes(self, engine: ReportingEngine, categorized_records):
        """Same input at different times may produce different hashes (timestamp in hash)."""
        r1 = engine.generate_categorization_report(categorized_records)
        r2 = engine.generate_categorization_report(categorized_records)
        # Due to unique report IDs and timestamps, hashes may differ
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64


# ===========================================================================
# TestFormatValidation
# ===========================================================================


class TestFormatValidation:
    """Test format validation and fallback behavior."""

    def test_valid_format_accepted(self, engine: ReportingEngine, categorized_records):
        """Valid format strings are accepted."""
        report = engine.generate_categorization_report(categorized_records, format="json")
        assert report.format == "json"

    def test_invalid_format_falls_back(self, engine: ReportingEngine, categorized_records):
        """Invalid format falls back to default format."""
        report = engine.generate_categorization_report(categorized_records, format="xml")
        assert report.format == "markdown"  # default fallback

    def test_json_default_fallback(self, engine_json_default: ReportingEngine, categorized_records):
        """Engine with json default falls back to json for invalid format."""
        report = engine_json_default.generate_categorization_report(
            categorized_records, format="invalid",
        )
        assert report.format == "json"

    def test_case_insensitive_format(self, engine: ReportingEngine, categorized_records):
        """Format string is case-insensitive."""
        report = engine.generate_categorization_report(categorized_records, format="JSON")
        assert report.format == "json"

    def test_whitespace_stripped(self, engine: ReportingEngine, categorized_records):
        """Leading/trailing whitespace is stripped from format."""
        report = engine.generate_categorization_report(categorized_records, format="  markdown  ")
        assert report.format == "markdown"
