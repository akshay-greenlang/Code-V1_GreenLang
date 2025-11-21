# -*- coding: utf-8 -*-
"""
Comprehensive tests for GreenLang BaseReporter.

Tests cover:
- Reporter initialization and configuration
- Multi-format output (Markdown, HTML, JSON)
- Section management
- Data aggregation
- Summary generation
- Table rendering
- Report metadata
- Excel export (if available)
- Edge cases
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.reporter import (
    BaseReporter,
    ReporterConfig,
    ReportSection,
)


# Test Reporter Implementations

class SimpleReporter(BaseReporter):
    """Simple reporter that aggregates totals."""

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sum all values."""
        values = input_data.get("values", [])
        return {
            "total": sum(values),
            "count": len(values),
            "average": sum(values) / len(values) if values else 0
        }

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """Build simple sections."""
        return [
            ReportSection(
                title="Statistics",
                content=f"Total: {aggregated_data['total']}, Count: {aggregated_data['count']}",
                section_type="text"
            )
        ]


class TableReporter(BaseReporter):
    """Reporter with table output."""

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return data as-is."""
        return input_data

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """Build table section."""
        data = aggregated_data.get("data", [])
        return [
            ReportSection(
                title="Data Table",
                content=data,
                section_type="table"
            )
        ]


class MultiSectionReporter(BaseReporter):
    """Reporter with multiple sections."""

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate multiple metrics."""
        values = input_data.get("values", [])
        return {
            "sum": sum(values),
            "max": max(values) if values else 0,
            "min": min(values) if values else 0,
            "values": values
        }

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """Build multiple sections."""
        return [
            ReportSection(
                title="Summary",
                content=f"Sum: {aggregated_data['sum']}",
                level=2,
                section_type="text"
            ),
            ReportSection(
                title="Details",
                content=f"Max: {aggregated_data['max']}, Min: {aggregated_data['min']}",
                level=3,
                section_type="text"
            ),
            ReportSection(
                title="Values",
                content=aggregated_data['values'],
                level=2,
                section_type="list"
            )
        ]


class ListReporter(BaseReporter):
    """Reporter with list output."""

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return items."""
        return {"items": input_data.get("items", [])}

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """Build list section."""
        return [
            ReportSection(
                title="Items",
                content=aggregated_data["items"],
                section_type="list"
            )
        ]


# Test Classes

@pytest.mark.unit
class TestReporterConfig:
    """Test ReporterConfig model."""

    def test_config_defaults(self):
        """Test config with default values."""
        config = ReporterConfig(
            name="TestReporter",
            description="Test reporter"
        )

        assert config.output_format == "markdown"
        assert config.include_summary is True
        assert config.include_details is True
        assert config.include_charts is False
        assert config.template_path is None

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = ReporterConfig(
            name="CustomReporter",
            description="Custom reporter",
            output_format="html",
            include_summary=False,
            include_details=False,
            include_charts=True,
            template_path="/path/to/template"
        )

        assert config.output_format == "html"
        assert config.include_summary is False
        assert config.include_details is False
        assert config.include_charts is True
        assert config.template_path == "/path/to/template"


@pytest.mark.unit
class TestReportSection:
    """Test ReportSection model."""

    def test_section_creation(self):
        """Test creating a report section."""
        section = ReportSection(
            title="Test Section",
            content="Test content",
            level=2,
            section_type="text"
        )

        assert section.title == "Test Section"
        assert section.content == "Test content"
        assert section.level == 2
        assert section.section_type == "text"

    def test_section_with_table_content(self):
        """Test section with table content."""
        data = [
            {"name": "Item 1", "value": 100},
            {"name": "Item 2", "value": 200}
        ]
        section = ReportSection(
            title="Data",
            content=data,
            section_type="table"
        )

        assert section.section_type == "table"
        assert isinstance(section.content, list)

    def test_section_with_list_content(self):
        """Test section with list content."""
        items = ["Item 1", "Item 2", "Item 3"]
        section = ReportSection(
            title="List",
            content=items,
            section_type="list"
        )

        assert section.section_type == "list"
        assert isinstance(section.content, list)


@pytest.mark.unit
class TestReporterInitialization:
    """Test reporter initialization."""

    def test_initialization_defaults(self):
        """Test reporter initializes with defaults."""
        reporter = SimpleReporter()

        assert reporter.config is not None
        assert reporter.config.output_format == "markdown"
        assert reporter.sections == []
        assert reporter.metadata == {}

    def test_initialization_custom_config(self):
        """Test reporter with custom config."""
        config = ReporterConfig(
            name="CustomReporter",
            description="Custom",
            output_format="html"
        )
        reporter = SimpleReporter(config)

        assert reporter.config.output_format == "html"


@pytest.mark.unit
class TestDataAggregation:
    """Test data aggregation."""

    def test_simple_aggregation(self):
        """Test simple data aggregation."""
        reporter = SimpleReporter()
        input_data = {"values": [10, 20, 30, 40, 50]}

        aggregated = reporter.aggregate_data(input_data)

        assert aggregated["total"] == 150
        assert aggregated["count"] == 5
        assert aggregated["average"] == 30

    def test_aggregation_empty_data(self):
        """Test aggregation with empty data."""
        reporter = SimpleReporter()
        input_data = {"values": []}

        aggregated = reporter.aggregate_data(input_data)

        assert aggregated["total"] == 0
        assert aggregated["count"] == 0
        assert aggregated["average"] == 0

    def test_aggregation_preserves_data(self):
        """Test aggregation can preserve original data."""
        reporter = TableReporter()
        input_data = {
            "data": [
                {"id": 1, "value": 100},
                {"id": 2, "value": 200}
            ]
        }

        aggregated = reporter.aggregate_data(input_data)

        assert aggregated == input_data


@pytest.mark.unit
class TestSectionManagement:
    """Test section management."""

    def test_add_section(self):
        """Test adding sections manually."""
        reporter = SimpleReporter()

        reporter.add_section("Title 1", "Content 1")
        reporter.add_section("Title 2", "Content 2", level=3, section_type="text")

        assert len(reporter.sections) == 2
        assert reporter.sections[0].title == "Title 1"
        assert reporter.sections[1].level == 3

    def test_build_sections(self):
        """Test building sections from data."""
        reporter = MultiSectionReporter()
        aggregated = {"sum": 100, "max": 50, "min": 10, "values": [10, 20, 30]}

        sections = reporter.build_sections(aggregated)

        assert len(sections) == 3
        assert sections[0].title == "Summary"
        assert sections[1].title == "Details"
        assert sections[2].title == "Values"

    def test_sections_cleared_between_runs(self):
        """Test sections are cleared between executions."""
        reporter = SimpleReporter()

        # First run
        result1 = reporter.run({"values": [1, 2, 3]})
        assert len(reporter.sections) > 0

        # Second run
        result2 = reporter.run({"values": [4, 5, 6]})

        # Sections should be for second run only
        assert "15" in result2.data["report"]  # Sum of 4+5+6


@pytest.mark.unit
class TestSummaryGeneration:
    """Test summary generation."""

    def test_generate_summary_default(self):
        """Test default summary generation."""
        reporter = SimpleReporter()
        aggregated = {"total": 150, "count": 5, "average": 30.0}

        summary = reporter.generate_summary(aggregated)

        assert "Total" in summary
        assert "150.00" in summary
        assert "Count" in summary
        assert "5.00" in summary

    def test_summary_formatting(self):
        """Test summary formatting."""
        reporter = SimpleReporter()
        aggregated = {
            "total_sales": 1234.56,
            "item_count": 42
        }

        summary = reporter.generate_summary(aggregated)

        assert "Total Sales" in summary
        assert "1,234.56" in summary
        assert "Item Count" in summary

    def test_summary_included_in_report(self):
        """Test summary is included when enabled."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            include_summary=True
        )
        reporter = SimpleReporter(config)

        result = reporter.run({"values": [10, 20, 30]})

        assert "Summary" in result.data["report"]

    def test_summary_excluded_from_report(self):
        """Test summary is excluded when disabled."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            include_summary=False
        )
        reporter = SimpleReporter(config)

        result = reporter.run({"values": [10, 20, 30]})

        # Summary section should not be present
        assert len(reporter.sections) == 1  # Only the Statistics section


@pytest.mark.unit
class TestMarkdownRendering:
    """Test Markdown rendering."""

    def test_render_markdown_basic(self):
        """Test basic Markdown rendering."""
        reporter = SimpleReporter()
        result = reporter.run({"values": [10, 20, 30]})

        markdown = result.data["report"]

        assert "# SimpleReporter Report" in markdown
        assert "Generated:" in markdown
        assert "## Statistics" in markdown

    def test_render_markdown_with_text(self):
        """Test Markdown rendering with text content."""
        reporter = SimpleReporter()
        reporter.add_section("Test Section", "This is test content")

        markdown = reporter.render_markdown()

        assert "## Test Section" in markdown
        assert "This is test content" in markdown

    def test_render_markdown_with_table(self):
        """Test Markdown rendering with table."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            include_summary=False
        )
        reporter = TableReporter(config)

        data = [
            {"name": "Item 1", "value": 100},
            {"name": "Item 2", "value": 200}
        ]
        result = reporter.run({"data": data})

        markdown = result.data["report"]

        assert "| name |" in markdown or "| value |" in markdown
        assert "| ---" in markdown

    def test_render_markdown_with_list(self):
        """Test Markdown rendering with list."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            include_summary=False
        )
        reporter = ListReporter(config)

        result = reporter.run({"items": ["First", "Second", "Third"]})
        markdown = result.data["report"]

        assert "- First" in markdown
        assert "- Second" in markdown
        assert "- Third" in markdown

    def test_render_markdown_heading_levels(self):
        """Test Markdown heading levels."""
        reporter = SimpleReporter()
        reporter.add_section("Level 1", "Content", level=1)
        reporter.add_section("Level 2", "Content", level=2)
        reporter.add_section("Level 3", "Content", level=3)

        markdown = reporter.render_markdown()

        assert "# Level 1" in markdown
        assert "## Level 2" in markdown
        assert "### Level 3" in markdown


@pytest.mark.unit
class TestHTMLRendering:
    """Test HTML rendering."""

    def test_render_html_basic(self):
        """Test basic HTML rendering."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="html"
        )
        reporter = SimpleReporter(config)
        result = reporter.run({"values": [10, 20, 30]})

        html = result.data["report"]

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html
        assert "<h1>Test Report</h1>" in html

    def test_render_html_with_styles(self):
        """Test HTML includes CSS styles."""
        reporter = SimpleReporter()
        html = reporter.render_html()

        assert "<style>" in html
        assert "font-family" in html
        assert "</style>" in html

    def test_render_html_with_text(self):
        """Test HTML rendering with text content."""
        reporter = SimpleReporter()
        reporter.add_section("Test Section", "Test content")

        html = reporter.render_html()

        assert "<h2>Test Section</h2>" in html
        assert "<p>Test content</p>" in html

    def test_render_html_with_table(self):
        """Test HTML rendering with table."""
        reporter = TableReporter()
        data = [
            {"name": "Item 1", "value": 100},
            {"name": "Item 2", "value": 200}
        ]

        reporter.add_section("Data", data, section_type="table")
        html = reporter.render_html()

        assert "<table>" in html
        assert "<th>" in html
        assert "<td>" in html
        assert "</table>" in html

    def test_render_html_with_list(self):
        """Test HTML rendering with list."""
        reporter = ListReporter()
        reporter.add_section("Items", ["First", "Second"], section_type="list")

        html = reporter.render_html()

        assert "<ul>" in html
        assert "<li>First</li>" in html
        assert "<li>Second</li>" in html
        assert "</ul>" in html

    def test_render_html_heading_levels(self):
        """Test HTML heading levels."""
        reporter = SimpleReporter()
        reporter.add_section("Level 1", "Content", level=1)
        reporter.add_section("Level 2", "Content", level=2)
        reporter.add_section("Level 3", "Content", level=3)

        html = reporter.render_html()

        assert "<h1>Level 1</h1>" in html
        assert "<h2>Level 2</h2>" in html
        assert "<h3>Level 3</h3>" in html


@pytest.mark.unit
class TestJSONRendering:
    """Test JSON rendering."""

    def test_render_json_basic(self):
        """Test basic JSON rendering."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="json"
        )
        reporter = SimpleReporter(config)
        result = reporter.run({"values": [10, 20, 30]})

        json_str = result.data["report"]
        data = json.loads(json_str)

        assert data["report_name"] == "Test"
        assert "generated_at" in data
        assert "sections" in data

    def test_render_json_with_sections(self):
        """Test JSON includes all sections."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="json"
        )
        reporter = MultiSectionReporter(config)
        result = reporter.run({"values": [10, 20, 30]})

        json_str = result.data["report"]
        data = json.loads(json_str)

        # Should have summary + 3 detail sections
        assert len(data["sections"]) >= 3

    def test_render_json_with_metadata(self):
        """Test JSON includes metadata."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="json"
        )
        reporter = SimpleReporter(config)
        reporter.metadata = {"custom": "value"}

        json_str = reporter.render_json()
        data = json.loads(json_str)

        assert data["metadata"]["custom"] == "value"

    def test_render_json_valid_format(self):
        """Test rendered JSON is valid."""
        reporter = SimpleReporter()
        reporter.add_section("Test", "Content")

        json_str = reporter.render_json()

        # Should not raise
        data = json.loads(json_str)
        assert isinstance(data, dict)


@pytest.mark.unit
class TestTableRendering:
    """Test table rendering functionality."""

    def test_render_table_markdown(self):
        """Test Markdown table rendering."""
        reporter = SimpleReporter()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]

        table_md = reporter._render_table_markdown(data)

        assert "| name |" in table_md or "| age |" in table_md
        assert "| --- |" in table_md
        assert "Alice" in table_md
        assert "Bob" in table_md

    def test_render_table_markdown_empty(self):
        """Test Markdown table with empty data."""
        reporter = SimpleReporter()

        table_md = reporter._render_table_markdown([])
        assert table_md == "[]"

    def test_render_table_html(self):
        """Test HTML table rendering."""
        reporter = SimpleReporter()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]

        table_html = reporter._render_table_html(data)

        assert "<table>" in table_html
        assert "<th>name</th>" in table_html or "<th>age</th>" in table_html
        assert "<td>Alice</td>" in table_html
        assert "</table>" in table_html

    def test_render_table_html_empty(self):
        """Test HTML table with empty data."""
        reporter = SimpleReporter()

        table_html = reporter._render_table_html([])
        assert table_html == "[]"


@pytest.mark.unit
class TestExcelRendering:
    """Test Excel rendering functionality."""

    @pytest.mark.skipif(True, reason="Excel export requires openpyxl")
    def test_render_excel_basic(self):
        """Test basic Excel rendering."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="excel"
        )
        reporter = SimpleReporter(config)

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            temp_path = f.name

        try:
            result = reporter.run({
                "values": [10, 20, 30],
                "output_path": temp_path
            })

            assert result.success is True
            assert Path(temp_path).exists()
        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_render_excel_missing_library(self):
        """Test Excel rendering without openpyxl."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="excel"
        )
        reporter = SimpleReporter(config)

        with patch('greenlang.agents.reporter.openpyxl', None):
            with pytest.raises(ImportError, match="openpyxl required"):
                reporter.render_excel("test.xlsx")


@pytest.mark.unit
class TestReportExecution:
    """Test complete report execution."""

    def test_execute_markdown(self):
        """Test execution with Markdown output."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="markdown"
        )
        reporter = SimpleReporter(config)

        result = reporter.run({"values": [10, 20, 30]})

        assert result.success is True
        assert result.data["format"] == "markdown"
        assert result.data["sections_count"] >= 1
        assert "report" in result.data

    def test_execute_html(self):
        """Test execution with HTML output."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="html"
        )
        reporter = SimpleReporter(config)

        result = reporter.run({"values": [10, 20, 30]})

        assert result.success is True
        assert result.data["format"] == "html"
        assert "<!DOCTYPE html>" in result.data["report"]

    def test_execute_json(self):
        """Test execution with JSON output."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="json"
        )
        reporter = SimpleReporter(config)

        result = reporter.run({"values": [10, 20, 30]})

        assert result.success is True
        assert result.data["format"] == "json"
        json.loads(result.data["report"])  # Should not raise

    def test_execute_unsupported_format(self):
        """Test execution with unsupported format."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            output_format="pdf"
        )
        reporter = SimpleReporter(config)

        result = reporter.run({"values": [10, 20, 30]})

        assert result.success is False
        assert "unsupported" in result.error.lower()

    def test_execute_includes_metadata(self):
        """Test execution includes metadata in result."""
        reporter = SimpleReporter()
        result = reporter.run({"values": [10, 20, 30]})

        assert "aggregated_data" in result.metadata
        assert "generated_at" in result.metadata

    def test_execute_with_details_disabled(self):
        """Test execution with details disabled."""
        config = ReporterConfig(
            name="Test",
            description="Test",
            include_details=False,
            include_summary=True
        )
        reporter = SimpleReporter(config)

        result = reporter.run({"values": [10, 20, 30]})

        assert result.success is True
        # Should only have summary section
        assert result.data["sections_count"] == 1


@pytest.mark.unit
class TestInputValidation:
    """Test input validation."""

    def test_validate_input_success(self):
        """Test input validation with valid input."""
        reporter = SimpleReporter()
        assert reporter.validate_input({"values": [1, 2, 3]}) is True

    def test_validate_input_empty(self):
        """Test validation with empty input."""
        reporter = SimpleReporter()
        assert reporter.validate_input({}) is True  # Base validator allows empty

    def test_validate_input_none(self):
        """Test validation with None input."""
        reporter = SimpleReporter()
        assert reporter.validate_input(None) is False


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data(self):
        """Test reporting with empty data."""
        reporter = SimpleReporter()
        result = reporter.run({"values": []})

        assert result.success is True

    def test_single_value(self):
        """Test reporting with single value."""
        reporter = SimpleReporter()
        result = reporter.run({"values": [42]})

        assert result.success is True
        assert "42" in result.data["report"]

    def test_large_dataset(self):
        """Test reporting with large dataset."""
        reporter = SimpleReporter()
        large_values = list(range(10000))

        result = reporter.run({"values": large_values})

        assert result.success is True
        # Check aggregation worked
        assert "49995000" in result.data["report"]  # Sum of 0..9999

    def test_special_characters_in_content(self):
        """Test handling special characters."""
        reporter = SimpleReporter()
        reporter.add_section("Test & <Special>", "Content with 'quotes' and \"marks\"")

        markdown = reporter.render_markdown()
        html = reporter.render_html()

        # Should not crash
        assert "Test & <Special>" in markdown
        assert "Content" in html

    def test_unicode_characters(self):
        """Test handling Unicode characters."""
        reporter = SimpleReporter()
        reporter.add_section("Test Section", "Content with émojis 🚀 and ünîcödé")

        markdown = reporter.render_markdown()
        json_str = reporter.render_json()

        assert "émojis" in markdown or "Content" in markdown
        # JSON should handle unicode
        data = json.loads(json_str)

    def test_nested_data_structures(self):
        """Test reporting with nested data."""
        reporter = SimpleReporter()
        nested_data = {
            "level1": {
                "level2": {
                    "value": 42
                }
            }
        }

        result = reporter.run(nested_data)
        assert result.success is True

    def test_concurrent_reports(self):
        """Test multiple concurrent reporter executions."""
        import threading

        reporter = SimpleReporter()
        results = []
        errors = []

        def generate_report(values):
            try:
                result = reporter.run({"values": values})
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=generate_report, args=([i, i+1, i+2],))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0

    def test_multiple_format_renders(self):
        """Test rendering same report in multiple formats."""
        reporter = SimpleReporter()
        reporter.add_section("Test", "Content")

        markdown = reporter.render_markdown()
        html = reporter.render_html()
        json_str = reporter.render_json()

        assert "Test" in markdown
        assert "Test" in html
        data = json.loads(json_str)
        assert data["sections"][0]["title"] == "Test"

    def test_empty_section_title(self):
        """Test section with empty title."""
        reporter = SimpleReporter()
        reporter.add_section("", "Content")

        markdown = reporter.render_markdown()
        assert "##" in markdown

    def test_very_long_content(self):
        """Test section with very long content."""
        reporter = SimpleReporter()
        long_content = "x" * 100000

        reporter.add_section("Long Content", long_content)
        markdown = reporter.render_markdown()

        assert len(markdown) > 100000
