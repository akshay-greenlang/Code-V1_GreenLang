"""
Comprehensive tests for BaseReporter class.
Tests report generation, data aggregation, formatting, and section management.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List

from greenlang.agents.reporter import (
    BaseReporter, ReporterConfig, ReportSection
)
from greenlang.agents.base import AgentResult


class TestReporter(BaseReporter):
    """Simple test reporter for testing base functionality."""

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate test data."""
        if "values" in input_data:
            values = input_data["values"]
            return {
                "total": sum(values),
                "average": sum(values) / len(values) if values else 0,
                "count": len(values),
                "max": max(values) if values else 0,
                "min": min(values) if values else 0
            }
        return {"count": 0}

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        """Build test sections."""
        sections = []

        # Statistics section
        sections.append(ReportSection(
            title="Statistics",
            content=f"Processed {aggregated_data.get('count', 0)} items",
            level=2,
            section_type="text"
        ))

        # Details table
        if aggregated_data.get("count", 0) > 0:
            table_data = [
                {"Metric": "Total", "Value": aggregated_data["total"]},
                {"Metric": "Average", "Value": aggregated_data["average"]},
                {"Metric": "Maximum", "Value": aggregated_data["max"]},
                {"Metric": "Minimum", "Value": aggregated_data["min"]}
            ]
            sections.append(ReportSection(
                title="Details",
                content=table_data,
                level=2,
                section_type="table"
            ))

        return sections


class EmptyReporter(BaseReporter):
    """Reporter that returns empty aggregation."""

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        return []


class ListReporter(BaseReporter):
    """Reporter that uses list sections."""

    def aggregate_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"items": input_data.get("items", [])}

    def build_sections(self, aggregated_data: Dict[str, Any]) -> List[ReportSection]:
        return [
            ReportSection(
                title="Items",
                content=aggregated_data.get("items", []),
                level=2,
                section_type="list"
            )
        ]


class TestReporterConfig:
    """Test ReporterConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReporterConfig(
            name="TestReporter",
            description="Test reporter"
        )
        assert config.name == "TestReporter"
        assert config.output_format == "markdown"
        assert config.include_summary is True
        assert config.include_details is True
        assert config.include_charts is False
        assert config.template_path is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReporterConfig(
            name="CustomReporter",
            description="Custom test reporter",
            output_format="html",
            include_summary=False,
            include_details=True,
            include_charts=True,
            template_path="/path/to/template.html"
        )
        assert config.output_format == "html"
        assert config.include_summary is False
        assert config.include_charts is True
        assert config.template_path == "/path/to/template.html"

    def test_all_output_formats(self):
        """Test all supported output formats."""
        formats = ["markdown", "html", "json", "excel"]
        for fmt in formats:
            config = ReporterConfig(
                name="FormatTest",
                description="Format test",
                output_format=fmt
            )
            assert config.output_format == fmt


class TestReportSection:
    """Test ReportSection model."""

    def test_text_section(self):
        """Test text section creation."""
        section = ReportSection(
            title="Summary",
            content="This is a summary",
            level=2,
            section_type="text"
        )
        assert section.title == "Summary"
        assert section.content == "This is a summary"
        assert section.level == 2
        assert section.section_type == "text"

    def test_table_section(self):
        """Test table section creation."""
        table_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        section = ReportSection(
            title="Users",
            content=table_data,
            level=2,
            section_type="table"
        )
        assert section.section_type == "table"
        assert isinstance(section.content, list)
        assert len(section.content) == 2

    def test_list_section(self):
        """Test list section creation."""
        items = ["Item 1", "Item 2", "Item 3"]
        section = ReportSection(
            title="Items",
            content=items,
            level=3,
            section_type="list"
        )
        assert section.section_type == "list"
        assert section.level == 3
        assert len(section.content) == 3

    def test_heading_levels(self):
        """Test different heading levels."""
        for level in range(1, 7):
            section = ReportSection(
                title=f"Level {level}",
                content="Content",
                level=level,
                section_type="text"
            )
            assert section.level == level


class TestBaseReporter:
    """Test BaseReporter functionality."""

    def test_reporter_initialization(self):
        """Test reporter initialization."""
        reporter = TestReporter()

        assert reporter.config.name == "TestReporter"
        assert isinstance(reporter.sections, list)
        assert len(reporter.sections) == 0
        assert isinstance(reporter.metadata, dict)

    def test_reporter_with_custom_config(self):
        """Test reporter with custom configuration."""
        config = ReporterConfig(
            name="CustomReporter",
            description="Custom reporter",
            output_format="json"
        )
        reporter = TestReporter(config=config)

        assert reporter.config.name == "CustomReporter"
        assert reporter.config.output_format == "json"

    def test_add_section(self):
        """Test adding sections to report."""
        reporter = TestReporter()

        reporter.add_section("Section 1", "Content 1")
        reporter.add_section("Section 2", "Content 2", level=3)
        reporter.add_section("Section 3", ["item1", "item2"], section_type="list")

        assert len(reporter.sections) == 3
        assert reporter.sections[0].title == "Section 1"
        assert reporter.sections[1].level == 3
        assert reporter.sections[2].section_type == "list"

    def test_successful_execution(self):
        """Test successful report execution."""
        reporter = TestReporter()
        input_data = {"values": [10, 20, 30, 40, 50]}

        result = reporter.run(input_data)

        assert result.success is True
        assert "report" in result.data
        assert result.data["format"] == "markdown"
        assert result.data["sections_count"] > 0

    def test_aggregate_data(self):
        """Test data aggregation."""
        reporter = TestReporter()
        input_data = {"values": [100, 200, 300]}

        aggregated = reporter.aggregate_data(input_data)

        assert aggregated["total"] == 600
        assert aggregated["average"] == 200
        assert aggregated["count"] == 3
        assert aggregated["max"] == 300
        assert aggregated["min"] == 100

    def test_build_sections(self):
        """Test section building."""
        reporter = TestReporter()
        aggregated_data = {
            "total": 600,
            "average": 200,
            "count": 3,
            "max": 300,
            "min": 100
        }

        sections = reporter.build_sections(aggregated_data)

        assert len(sections) == 2
        assert sections[0].title == "Statistics"
        assert sections[1].title == "Details"
        assert sections[1].section_type == "table"


class TestSummaryGeneration:
    """Test summary generation functionality."""

    def test_generate_summary_with_numbers(self):
        """Test summary generation with numeric data."""
        reporter = TestReporter()
        aggregated_data = {
            "total_sales": 10000.50,
            "total_units": 100,
            "average_price": 100.01
        }

        summary = reporter.generate_summary(aggregated_data)

        assert "Total Sales" in summary
        assert "10,000.50" in summary
        assert "Total Units" in summary
        assert "Average Price" in summary

    def test_generate_summary_with_strings(self):
        """Test summary generation with string data."""
        reporter = TestReporter()
        aggregated_data = {
            "status": "completed",
            "category": "sales"
        }

        summary = reporter.generate_summary(aggregated_data)

        assert "Status" in summary
        assert "completed" in summary
        assert "Category" in summary
        assert "sales" in summary

    def test_generate_summary_mixed_types(self):
        """Test summary generation with mixed data types."""
        reporter = TestReporter()
        aggregated_data = {
            "count": 42,
            "percentage": 75.5,
            "label": "test",
            "active": True
        }

        summary = reporter.generate_summary(aggregated_data)

        assert "Count" in summary
        assert "42.00" in summary
        assert "Label" in summary
        assert "test" in summary


class TestMarkdownRendering:
    """Test Markdown report rendering."""

    def test_render_markdown_basic(self):
        """Test basic Markdown rendering."""
        reporter = TestReporter()
        reporter.add_section("Test Section", "Test content")

        markdown = reporter.render_markdown()

        assert "# TestReporter Report" in markdown
        assert "## Test Section" in markdown
        assert "Test content" in markdown
        assert "Generated:" in markdown

    def test_render_markdown_multiple_sections(self):
        """Test Markdown rendering with multiple sections."""
        reporter = TestReporter()
        reporter.add_section("Section 1", "Content 1", level=2)
        reporter.add_section("Section 2", "Content 2", level=3)

        markdown = reporter.render_markdown()

        assert "## Section 1" in markdown
        assert "### Section 2" in markdown
        assert "Content 1" in markdown
        assert "Content 2" in markdown

    def test_render_markdown_table(self):
        """Test Markdown table rendering."""
        reporter = TestReporter()
        table_data = [
            {"Name": "Alice", "Age": 30},
            {"Name": "Bob", "Age": 25}
        ]
        reporter.add_section("Users", table_data, section_type="table")

        markdown = reporter.render_markdown()

        assert "| Name | Age |" in markdown
        assert "| --- | --- |" in markdown
        assert "| Alice | 30 |" in markdown
        assert "| Bob | 25 |" in markdown

    def test_render_markdown_list(self):
        """Test Markdown list rendering."""
        reporter = TestReporter()
        items = ["Item 1", "Item 2", "Item 3"]
        reporter.add_section("Items", items, section_type="list")

        markdown = reporter.render_markdown()

        assert "- Item 1" in markdown
        assert "- Item 2" in markdown
        assert "- Item 3" in markdown

    def test_render_markdown_empty_table(self):
        """Test Markdown rendering with empty table."""
        reporter = TestReporter()
        reporter.add_section("Empty Table", [], section_type="table")

        markdown = reporter.render_markdown()

        assert "[]" in markdown or "Empty Table" in markdown


class TestHTMLRendering:
    """Test HTML report rendering."""

    def test_render_html_basic(self):
        """Test basic HTML rendering."""
        reporter = TestReporter()
        reporter.add_section("Test Section", "Test content")

        html = reporter.render_html()

        assert "<!DOCTYPE html>" in html
        assert "<h1>TestReporter Report</h1>" in html
        assert "<h2>Test Section</h2>" in html
        assert "<p>Test content</p>" in html

    def test_render_html_structure(self):
        """Test HTML document structure."""
        reporter = TestReporter()
        reporter.add_section("Section", "Content")

        html = reporter.render_html()

        assert "<html>" in html
        assert "<head>" in html
        assert "<title>TestReporter Report</title>" in html
        assert "<style>" in html
        assert "<body>" in html
        assert "</html>" in html

    def test_render_html_table(self):
        """Test HTML table rendering."""
        reporter = TestReporter()
        table_data = [
            {"Product": "Widget", "Price": 10.99},
            {"Product": "Gadget", "Price": 20.50}
        ]
        reporter.add_section("Products", table_data, section_type="table")

        html = reporter.render_html()

        assert "<table>" in html
        assert "<th>Product</th>" in html
        assert "<th>Price</th>" in html
        assert "<td>Widget</td>" in html
        assert "<td>Gadget</td>" in html

    def test_render_html_list(self):
        """Test HTML list rendering."""
        reporter = TestReporter()
        items = ["First", "Second", "Third"]
        reporter.add_section("List", items, section_type="list")

        html = reporter.render_html()

        assert "<ul>" in html
        assert "<li>First</li>" in html
        assert "<li>Second</li>" in html
        assert "<li>Third</li>" in html

    def test_render_html_heading_levels(self):
        """Test HTML heading levels."""
        reporter = TestReporter()
        reporter.add_section("Level 1", "Content", level=1)
        reporter.add_section("Level 2", "Content", level=2)
        reporter.add_section("Level 3", "Content", level=3)

        html = reporter.render_html()

        assert "<h1>Level 1</h1>" in html
        assert "<h2>Level 2</h2>" in html
        assert "<h3>Level 3</h3>" in html


class TestJSONRendering:
    """Test JSON report rendering."""

    def test_render_json_basic(self):
        """Test basic JSON rendering."""
        reporter = TestReporter()
        reporter.add_section("Test", "Content")

        json_str = reporter.render_json()
        data = json.loads(json_str)

        assert "report_name" in data
        assert data["report_name"] == "TestReporter"
        assert "generated_at" in data
        assert "sections" in data
        assert len(data["sections"]) == 1

    def test_render_json_structure(self):
        """Test JSON structure."""
        reporter = TestReporter()
        reporter.add_section("Section 1", "Content 1")
        reporter.add_section("Section 2", ["item1", "item2"], section_type="list")

        json_str = reporter.render_json()
        data = json.loads(json_str)

        assert len(data["sections"]) == 2
        assert data["sections"][0]["title"] == "Section 1"
        assert data["sections"][1]["section_type"] == "list"

    def test_render_json_metadata(self):
        """Test JSON with metadata."""
        reporter = TestReporter()
        reporter.metadata["custom_key"] = "custom_value"
        reporter.add_section("Test", "Content")

        json_str = reporter.render_json()
        data = json.loads(json_str)

        assert "metadata" in data
        assert data["metadata"]["custom_key"] == "custom_value"

    def test_render_json_valid(self):
        """Test that rendered JSON is valid."""
        reporter = TestReporter()
        table_data = [{"col1": "val1", "col2": 123}]
        reporter.add_section("Table", table_data, section_type="table")

        json_str = reporter.render_json()

        # Should not raise exception
        data = json.loads(json_str)
        assert isinstance(data, dict)


class TestExcelRendering:
    """Test Excel report rendering."""

    def test_render_excel_basic(self, tmp_path):
        """Test basic Excel rendering."""
        pytest.importorskip("openpyxl")

        reporter = TestReporter()
        reporter.add_section("Test Section", "Test content")

        output_path = tmp_path / "test_report.xlsx"
        reporter.render_excel(str(output_path))

        assert output_path.exists()

    def test_render_excel_with_table(self, tmp_path):
        """Test Excel rendering with table data."""
        pytest.importorskip("openpyxl")

        reporter = TestReporter()
        table_data = [
            {"Name": "Alice", "Score": 95},
            {"Name": "Bob", "Score": 87}
        ]
        reporter.add_section("Scores", table_data, section_type="table")

        output_path = tmp_path / "table_report.xlsx"
        reporter.render_excel(str(output_path))

        assert output_path.exists()

        # Verify content
        import openpyxl
        workbook = openpyxl.load_workbook(output_path)
        sheet = workbook.active

        # Check that table headers are written
        assert sheet.cell(row=1, column=1).value is not None

    def test_render_excel_multiple_sections(self, tmp_path):
        """Test Excel rendering with multiple sections."""
        pytest.importorskip("openpyxl")

        reporter = TestReporter()
        reporter.add_section("Summary", "This is a summary")
        reporter.add_section("Data", [{"Key": "Value"}], section_type="table")

        output_path = tmp_path / "multi_report.xlsx"
        reporter.render_excel(str(output_path))

        assert output_path.exists()

    def test_render_excel_missing_dependency(self, monkeypatch, tmp_path):
        """Test Excel rendering without openpyxl."""
        import sys

        # Mock openpyxl as not installed
        import_orig = __builtins__.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openpyxl":
                raise ImportError("No module named 'openpyxl'")
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(__builtins__, "__import__", mock_import)

        reporter = TestReporter()
        reporter.add_section("Test", "Content")

        with pytest.raises(ImportError, match="openpyxl required"):
            reporter.render_excel(str(tmp_path / "test.xlsx"))


class TestReportExecution:
    """Test full report execution."""

    def test_execute_markdown_format(self):
        """Test execution with Markdown format."""
        config = ReporterConfig(
            name="MarkdownReporter",
            description="Test",
            output_format="markdown"
        )
        reporter = TestReporter(config=config)

        result = reporter.run({"values": [1, 2, 3, 4, 5]})

        assert result.success is True
        assert "report" in result.data
        assert "# MarkdownReporter Report" in result.data["report"]

    def test_execute_html_format(self):
        """Test execution with HTML format."""
        config = ReporterConfig(
            name="HTMLReporter",
            description="Test",
            output_format="html"
        )
        reporter = TestReporter(config=config)

        result = reporter.run({"values": [10, 20, 30]})

        assert result.success is True
        assert "<!DOCTYPE html>" in result.data["report"]

    def test_execute_json_format(self):
        """Test execution with JSON format."""
        config = ReporterConfig(
            name="JSONReporter",
            description="Test",
            output_format="json"
        )
        reporter = TestReporter(config=config)

        result = reporter.run({"values": [5, 10, 15]})

        assert result.success is True
        json_data = json.loads(result.data["report"])
        assert "report_name" in json_data

    def test_execute_excel_format(self, tmp_path):
        """Test execution with Excel format."""
        pytest.importorskip("openpyxl")

        config = ReporterConfig(
            name="ExcelReporter",
            description="Test",
            output_format="excel"
        )
        reporter = TestReporter(config=config)

        output_path = tmp_path / "report.xlsx"
        result = reporter.run({
            "values": [1, 2, 3],
            "output_path": str(output_path)
        })

        assert result.success is True
        assert output_path.exists()

    def test_execute_with_summary_disabled(self):
        """Test execution with summary disabled."""
        config = ReporterConfig(
            name="NoSummary",
            description="Test",
            include_summary=False
        )
        reporter = TestReporter(config=config)

        result = reporter.run({"values": [1, 2, 3]})

        assert result.success is True
        # Should not have summary section
        summary_sections = [s for s in reporter.sections if s.title == "Summary"]
        assert len(summary_sections) == 0

    def test_execute_with_details_disabled(self):
        """Test execution with details disabled."""
        config = ReporterConfig(
            name="NoDetails",
            description="Test",
            include_details=False
        )
        reporter = TestReporter(config=config)

        result = reporter.run({"values": [1, 2, 3]})

        assert result.success is True
        # Should only have summary
        assert len(reporter.sections) <= 1

    def test_execute_unsupported_format(self):
        """Test execution with unsupported format."""
        config = ReporterConfig(
            name="BadFormat",
            description="Test",
            output_format="pdf"  # Unsupported
        )
        reporter = TestReporter(config=config)

        result = reporter.run({"values": [1, 2, 3]})

        assert result.success is False
        assert "Unsupported output format" in result.error


class TestTableRendering:
    """Test table rendering functionality."""

    def test_render_table_markdown_simple(self):
        """Test simple Markdown table."""
        reporter = TestReporter()
        data = [
            {"col1": "val1", "col2": "val2"},
            {"col1": "val3", "col2": "val4"}
        ]

        table = reporter._render_table_markdown(data)

        assert "| col1 | col2 |" in table
        assert "| --- | --- |" in table
        assert "| val1 | val2 |" in table
        assert "| val3 | val4 |" in table

    def test_render_table_markdown_empty(self):
        """Test empty Markdown table."""
        reporter = TestReporter()

        table = reporter._render_table_markdown([])

        assert table == "[]"

    def test_render_table_markdown_non_dict(self):
        """Test Markdown table with non-dict data."""
        reporter = TestReporter()

        table = reporter._render_table_markdown(["not", "a", "dict"])

        assert isinstance(table, str)

    def test_render_table_html_simple(self):
        """Test simple HTML table."""
        reporter = TestReporter()
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]

        table = reporter._render_table_html(data)

        assert "<table>" in table
        assert "<th>name</th>" in table
        assert "<th>age</th>" in table
        assert "<td>Alice</td>" in table
        assert "<td>30</td>" in table

    def test_render_table_html_missing_values(self):
        """Test HTML table with missing values."""
        reporter = TestReporter()
        data = [
            {"a": "1", "b": "2"},
            {"a": "3"}  # Missing 'b'
        ]

        table = reporter._render_table_html(data)

        assert "<table>" in table
        # Should handle missing values gracefully


class TestInputValidation:
    """Test input validation."""

    def test_validate_input_valid(self):
        """Test validation with valid input."""
        reporter = TestReporter()

        assert reporter.validate_input({"values": [1, 2, 3]}) is True

    def test_validate_input_empty(self):
        """Test validation with empty input."""
        reporter = TestReporter()

        assert reporter.validate_input({}) is False

    def test_validate_input_none(self):
        """Test validation with None input."""
        reporter = TestReporter()

        assert reporter.validate_input(None) is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_aggregation(self):
        """Test with empty aggregation results."""
        reporter = EmptyReporter()

        result = reporter.run({"test": "data"})

        assert result.success is True

    def test_no_sections(self):
        """Test with no sections."""
        reporter = EmptyReporter()

        result = reporter.run({"test": "data"})
        markdown = result.data["report"]

        assert "# EmptyReporter Report" in markdown

    def test_sections_cleared_between_runs(self):
        """Test that sections are cleared between executions."""
        reporter = TestReporter()

        reporter.run({"values": [1, 2, 3]})
        first_count = len(reporter.sections)

        reporter.run({"values": [4, 5, 6]})
        second_count = len(reporter.sections)

        # Should be same count (sections cleared)
        assert first_count == second_count

    def test_metadata_handling(self):
        """Test metadata handling in results."""
        reporter = TestReporter()

        result = reporter.run({"values": [10, 20, 30]})

        assert "metadata" in result.dict()
        assert "aggregated_data" in result.metadata
        assert "generated_at" in result.metadata

    def test_complex_nested_data(self):
        """Test with complex nested data structures."""
        reporter = TestReporter()
        reporter.add_section(
            "Complex",
            {"nested": {"data": {"structure": "value"}}},
            section_type="text"
        )

        markdown = reporter.render_markdown()
        assert "Complex" in markdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
