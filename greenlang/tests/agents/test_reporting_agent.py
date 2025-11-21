# -*- coding: utf-8 -*-
"""
Tests for ReportingAgent
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from greenlang.agents.templates import ReportingAgent, ReportFormat


class TestReportingAgent:
    """Test ReportingAgent."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ReportingAgent()
        assert agent is not None

    @pytest.mark.asyncio
    async def test_generate_json_report(self):
        """Test generating JSON report."""
        agent = ReportingAgent()

        data = pd.DataFrame({
            "category": ["A", "B"],
            "value": [100, 200]
        })

        result = await agent.generate_report(
            data=data,
            format=ReportFormat.JSON
        )

        assert result.success is True
        assert result.data is not None
        assert result.format == ReportFormat.JSON

    @pytest.mark.asyncio
    async def test_generate_csv_report(self):
        """Test generating CSV report."""
        agent = ReportingAgent()

        data = pd.DataFrame({
            "category": ["A", "B"],
            "value": [100, 200]
        })

        result = await agent.generate_report(
            data=data,
            format=ReportFormat.CSV
        )

        assert result.success is True
        assert "category,value" in result.data

    @pytest.mark.asyncio
    async def test_generate_excel_report(self):
        """Test generating Excel report."""
        agent = ReportingAgent()

        data = pd.DataFrame({
            "category": ["A", "B"],
            "value": [100, 200]
        })

        result = await agent.generate_report(
            data=data,
            format=ReportFormat.EXCEL
        )

        assert result.success is True
        assert isinstance(result.data, bytes)

    @pytest.mark.asyncio
    async def test_generate_html_report(self):
        """Test generating HTML report."""
        agent = ReportingAgent()

        data = pd.DataFrame({
            "category": ["A", "B"],
            "value": [100, 200]
        })

        result = await agent.generate_report(
            data=data,
            format=ReportFormat.HTML
        )

        assert result.success is True
        assert "<table" in result.data

    @pytest.mark.asyncio
    async def test_generate_markdown_report(self):
        """Test generating Markdown report."""
        agent = ReportingAgent()

        data = pd.DataFrame({
            "category": ["A", "B"],
            "value": [100, 200]
        })

        result = await agent.generate_report(
            data=data,
            format=ReportFormat.MARKDOWN
        )

        assert result.success is True
        assert "|" in result.data  # Markdown table format

    @pytest.mark.asyncio
    async def test_save_report_to_file(self):
        """Test saving report to file."""
        agent = ReportingAgent()

        data = pd.DataFrame({
            "category": ["A", "B"],
            "value": [100, 200]
        })

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            result = await agent.generate_report(
                data=data,
                format=ReportFormat.JSON,
                output_path=temp_path
            )

            assert result.success is True
            assert Path(temp_path).exists()

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_generate_empty_data(self):
        """Test generating report with empty data."""
        agent = ReportingAgent()

        data = pd.DataFrame()

        result = await agent.generate_report(
            data=data,
            format=ReportFormat.JSON
        )

        assert result.success is False
        assert len(result.errors) > 0

    def test_get_stats(self):
        """Test getting agent statistics."""
        agent = ReportingAgent()
        stats = agent.get_stats()

        assert "total_reports" in stats
        assert "reports_by_format" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
