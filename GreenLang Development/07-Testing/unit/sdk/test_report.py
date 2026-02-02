# -*- coding: utf-8 -*-
"""
Comprehensive tests for SDK Report abstraction.

Tests cover:
- Report initialization
- Report generation in multiple formats
- Saving reports to files
- Template handling
- Format support
"""

import pytest
from pathlib import Path
from typing import Any, Optional, List
from greenlang.sdk.base import Report, Metadata


class MockReport(Report):
    """Mock report for testing."""

    def __init__(self, metadata: Optional[Metadata] = None):
        """Initialize mock report."""
        super().__init__(metadata)
        self.generated_format = None
        self.generated_data = None

    def generate(self, data: Any, format: str = "markdown") -> str:
        """Generate mock report."""
        self.generated_format = format
        self.generated_data = data

        if format == "markdown":
            return f"# Report\n\nData: {data}"
        elif format == "html":
            return f"<html><body><h1>Report</h1><p>{data}</p></body></html>"
        elif format == "json":
            import json

            return json.dumps({"report": "data", "content": str(data)})
        else:
            return f"Report: {data}"

    def save(self, content: str, path: Path) -> bool:
        """Save report to file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")
            return False


class AdvancedReport(Report):
    """Advanced report with templates."""

    def __init__(self, metadata: Optional[Metadata] = None):
        """Initialize advanced report."""
        super().__init__(metadata)
        self.templates = ["default", "detailed", "summary"]
        self.current_template = "default"

    def generate(self, data: Any, format: str = "markdown") -> str:
        """Generate report with templates."""
        if self.current_template == "detailed":
            content = f"Detailed Report\n\n{data}\n\nEnd of Report"
        elif self.current_template == "summary":
            content = f"Summary: {data}"
        else:
            content = f"Report: {data}"

        return content

    def save(self, content: str, path: Path) -> bool:
        """Save report to file."""
        try:
            path.write_text(content)
            return True
        except Exception:
            return False

    def get_templates(self) -> List[str]:
        """Get available templates."""
        return self.templates

    def get_formats(self) -> List[str]:
        """Get supported formats."""
        return ["markdown", "html", "json", "pdf"]


@pytest.mark.unit
class TestReportInitialization:
    """Test Report initialization."""

    def test_report_default_init(self):
        """Test creating report with defaults."""
        report = MockReport()

        assert report.metadata is not None
        assert report.metadata.id == "mockreport"
        assert report.metadata.name == "MockReport"
        assert report.logger is not None

    def test_report_with_custom_metadata(self):
        """Test creating report with custom metadata."""
        metadata = Metadata(
            id="custom-report",
            name="Custom Report",
            version="2.0.0",
            description="Test report",
        )
        report = MockReport(metadata=metadata)

        assert report.metadata.id == "custom-report"
        assert report.metadata.name == "Custom Report"
        assert report.metadata.version == "2.0.0"


@pytest.mark.unit
class TestReportGeneration:
    """Test report generation."""

    def test_generate_markdown(self):
        """Test generating markdown report."""
        report = MockReport()
        content = report.generate({"value": 42}, format="markdown")

        assert "# Report" in content
        assert "42" in content
        assert report.generated_format == "markdown"

    def test_generate_html(self):
        """Test generating HTML report."""
        report = MockReport()
        content = report.generate({"value": 42}, format="html")

        assert "<html>" in content
        assert "<h1>Report</h1>" in content
        assert report.generated_format == "html"

    def test_generate_json(self):
        """Test generating JSON report."""
        import json

        report = MockReport()
        content = report.generate({"value": 42}, format="json")

        data = json.loads(content)
        assert "report" in data
        assert report.generated_format == "json"

    def test_generate_default_format(self):
        """Test generating report with default format."""
        report = MockReport()
        content = report.generate({"test": "data"})

        # Default is markdown
        assert "# Report" in content

    def test_generate_unknown_format(self):
        """Test generating report with unknown format."""
        report = MockReport()
        content = report.generate("data", format="unknown")

        assert "Report:" in content

    def test_generate_with_different_data(self):
        """Test generating reports with different data."""
        report = MockReport()

        content1 = report.generate("text data")
        content2 = report.generate({"key": "value"})
        content3 = report.generate(12345)

        assert "text data" in content1
        assert "key" in content2
        assert "12345" in content3


@pytest.mark.unit
class TestReportSaving:
    """Test saving reports to files."""

    def test_save_markdown_report(self, tmp_path):
        """Test saving markdown report."""
        report = MockReport()
        content = report.generate({"value": 42}, "markdown")

        output_path = tmp_path / "report.md"
        result = report.save(content, output_path)

        assert result is True
        assert output_path.exists()
        assert "# Report" in output_path.read_text()

    def test_save_html_report(self, tmp_path):
        """Test saving HTML report."""
        report = MockReport()
        content = report.generate({"value": 42}, "html")

        output_path = tmp_path / "report.html"
        result = report.save(content, output_path)

        assert result is True
        assert output_path.exists()
        assert "<html>" in output_path.read_text()

    def test_save_creates_directory(self, tmp_path):
        """Test that save creates parent directories."""
        report = MockReport()
        content = report.generate("test")

        output_path = tmp_path / "nested" / "dir" / "report.md"
        result = report.save(content, output_path)

        assert result is True
        assert output_path.parent.exists()

    def test_save_multiple_reports(self, tmp_path):
        """Test saving multiple reports."""
        report = MockReport()

        for i in range(3):
            content = report.generate(f"Report {i}")
            path = tmp_path / f"report{i}.md"
            result = report.save(content, path)
            assert result is True

        assert len(list(tmp_path.glob("*.md"))) == 3


@pytest.mark.unit
class TestReportTemplates:
    """Test report template functionality."""

    def test_get_templates_default(self):
        """Test getting default templates."""
        report = MockReport()
        templates = report.get_templates()

        assert isinstance(templates, list)
        assert "default" in templates

    def test_get_templates_custom(self):
        """Test getting custom templates."""
        report = AdvancedReport()
        templates = report.get_templates()

        assert len(templates) == 3
        assert "default" in templates
        assert "detailed" in templates
        assert "summary" in templates

    def test_generate_with_template(self):
        """Test generating report with specific template."""
        report = AdvancedReport()

        # Detailed template
        report.current_template = "detailed"
        detailed_content = report.generate("test data")
        assert "Detailed Report" in detailed_content

        # Summary template
        report.current_template = "summary"
        summary_content = report.generate("test data")
        assert "Summary:" in summary_content


@pytest.mark.unit
class TestReportFormats:
    """Test report format support."""

    def test_get_formats_default(self):
        """Test getting default formats."""
        report = MockReport()
        formats = report.get_formats()

        assert isinstance(formats, list)
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_get_formats_custom(self):
        """Test getting custom formats."""
        report = AdvancedReport()
        formats = report.get_formats()

        assert len(formats) == 4
        assert "pdf" in formats

    def test_all_formats_generate(self):
        """Test that all supported formats can generate."""
        report = MockReport()
        formats = report.get_formats()
        data = {"test": "data"}

        for fmt in formats:
            content = report.generate(data, format=fmt)
            assert content is not None
            assert len(content) > 0


@pytest.mark.unit
class TestReportGenerateAndSave:
    """Test combined generate and save workflow."""

    def test_generate_and_save_workflow(self, tmp_path):
        """Test typical generate and save workflow."""
        report = MockReport()
        data = {"metric": 100, "status": "success"}

        # Generate
        content = report.generate(data, format="markdown")

        # Save
        output_path = tmp_path / "output.md"
        result = report.save(content, output_path)

        assert result is True
        assert output_path.exists()

    def test_multiple_format_generation(self, tmp_path):
        """Test generating same report in multiple formats."""
        report = MockReport()
        data = {"value": 42}

        formats = ["markdown", "html", "json"]
        extensions = {"markdown": ".md", "html": ".html", "json": ".json"}

        for fmt in formats:
            content = report.generate(data, format=fmt)
            path = tmp_path / f"report{extensions[fmt]}"
            report.save(content, path)

        # Verify all files created
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "report.html").exists()
        assert (tmp_path / "report.json").exists()


@pytest.mark.unit
class TestReportMetadata:
    """Test report metadata handling."""

    def test_metadata_in_report(self):
        """Test using metadata in reports."""
        metadata = Metadata(
            id="test-report",
            name="Test Report",
            version="1.0.0",
            author="Test Author",
        )
        report = MockReport(metadata=metadata)

        assert report.metadata.name == "Test Report"
        assert report.metadata.author == "Test Author"

    def test_metadata_timestamps(self):
        """Test metadata has timestamps."""
        report = MockReport()

        assert report.metadata.created_at is not None
        assert report.metadata.updated_at is not None


@pytest.mark.unit
class TestReportEdgeCases:
    """Test report edge cases."""

    def test_generate_with_empty_data(self):
        """Test generating report with empty data."""
        report = MockReport()
        content = report.generate({})

        assert content is not None
        assert "# Report" in content

    def test_generate_with_none_data(self):
        """Test generating report with None data."""
        report = MockReport()
        content = report.generate(None)

        assert content is not None

    def test_save_empty_content(self, tmp_path):
        """Test saving empty content."""
        report = MockReport()
        output_path = tmp_path / "empty.md"

        result = report.save("", output_path)

        assert result is True
        assert output_path.exists()
        assert output_path.read_text() == ""

    def test_generate_large_data(self):
        """Test generating report with large data."""
        report = MockReport()
        large_data = {"records": list(range(1000))}

        content = report.generate(large_data)

        assert content is not None
        assert len(content) > 0
