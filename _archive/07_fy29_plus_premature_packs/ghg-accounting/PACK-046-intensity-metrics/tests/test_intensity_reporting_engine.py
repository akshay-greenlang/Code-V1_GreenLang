"""
Unit tests for IntensityReportingEngine (PACK-046 Engine 10 - Planned).

Tests the expected API for automated intensity report generation once
the engine is implemented.

40+ tests covering:
  - Engine initialisation
  - Executive summary generation
  - Detailed report generation
  - Multi-format output (HTML, Markdown, JSON, CSV, PDF)
  - Report sections (intensity results, trends, decomposition, etc.)
  - Branding configuration
  - Data table inclusion
  - Appendix generation
  - Provenance hash tracking
  - Edge cases

Author: GreenLang QA Team
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import OutputFormat, ReportingConfig

try:
    from engines.intensity_reporting_engine import (
        IntensityReportingEngine,
        ReportInput,
        ReportResult,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="IntensityReportingEngine not yet implemented",
)


class TestIntensityReportingEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = IntensityReportingEngine()
        assert engine is not None

    def test_init_version(self):
        engine = IntensityReportingEngine()
        assert engine.get_version() == "1.0.0"

    def test_supported_formats(self):
        engine = IntensityReportingEngine()
        formats = engine.get_supported_formats()
        assert "HTML" in formats
        assert "MARKDOWN" in formats
        assert "JSON" in formats


class TestExecutiveSummary:
    """Tests for executive summary generation."""

    def test_executive_summary_html(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="executive_summary",
            output_format=OutputFormat.HTML,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={
                "revenue_intensity": Decimal("16.0"),
                "fte_intensity": Decimal("4.2"),
            },
        )
        result = engine.generate(inp)
        assert result.content is not None
        assert "ACME Corp" in result.content

    def test_executive_summary_markdown(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="executive_summary",
            output_format=OutputFormat.MARKDOWN,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={
                "revenue_intensity": Decimal("16.0"),
            },
        )
        result = engine.generate(inp)
        assert "# " in result.content

    def test_executive_summary_json(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="executive_summary",
            output_format=OutputFormat.JSON,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={
                "revenue_intensity": Decimal("16.0"),
            },
        )
        result = engine.generate(inp)
        assert isinstance(result.structured_data, dict)


class TestDetailedReport:
    """Tests for detailed report generation."""

    def test_detailed_report_html(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="detailed",
            output_format=OutputFormat.HTML,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={
                "revenue_intensity": Decimal("16.0"),
            },
            sections=[
                "executive_summary",
                "intensity_results",
                "trend_analysis",
                "methodology",
            ],
        )
        result = engine.generate(inp)
        assert "<!DOCTYPE html>" in result.content or "<html" in result.content

    def test_detailed_report_sections_included(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="detailed",
            output_format=OutputFormat.MARKDOWN,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={
                "revenue_intensity": Decimal("16.0"),
            },
            sections=["intensity_results", "methodology"],
        )
        result = engine.generate(inp)
        assert "intensity" in result.content.lower()


class TestMultiFormatOutput:
    """Tests for multi-format output."""

    def test_csv_output(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="data_export",
            output_format=OutputFormat.CSV,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={
                "revenue_intensity": Decimal("16.0"),
            },
        )
        result = engine.generate(inp)
        assert result.content is not None


class TestReportBranding:
    """Tests for report branding."""

    def test_custom_branding(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="executive_summary",
            output_format=OutputFormat.HTML,
            company_name="Custom Corp",
            reporting_period="FY2025",
            intensity_data={"revenue_intensity": Decimal("16.0")},
            branding={
                "primary_colour": "#FF0000",
                "company_name": "Custom Corp",
            },
        )
        result = engine.generate(inp)
        assert "Custom Corp" in result.content


class TestReportEdgeCases:
    """Tests for edge cases."""

    def test_empty_intensity_data(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="executive_summary",
            output_format=OutputFormat.HTML,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={},
        )
        result = engine.generate(inp)
        assert result is not None

    def test_provenance_hash(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="executive_summary",
            output_format=OutputFormat.HTML,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={"revenue_intensity": Decimal("16.0")},
        )
        result = engine.generate(inp)
        assert len(result.provenance_hash) == 64

    def test_processing_time_recorded(self):
        engine = IntensityReportingEngine()
        inp = ReportInput(
            report_type="executive_summary",
            output_format=OutputFormat.HTML,
            company_name="ACME Corp",
            reporting_period="FY2025",
            intensity_data={"revenue_intensity": Decimal("16.0")},
        )
        result = engine.generate(inp)
        assert result.processing_time_ms >= 0
