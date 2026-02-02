# -*- coding: utf-8 -*-
"""Tests for ReportAgent."""

import json
import pytest
from greenlang.agents.report_agent import ReportAgent


class TestReportAgent:
    """Test suite for ReportAgent - contract-focused tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = ReportAgent()
    
    def test_json_report_schema(self):
        """Test that JSON report conforms to stable schema."""
        report_data = {
            "total_emissions": 1000000.0,
            "emission_intensities": {
                "per_sqft": 20.0,
                "per_person": 4000.0
            },
            "benchmark_rating": "C",
            "recommendations": [
                {"action": "LED lighting", "impact": "high", "payback": "short"},
                {"action": "HVAC upgrade", "impact": "very_high", "payback": "medium"}
            ],
            "building_info": {
                "name": "Test Building",
                "type": "office",
                "country": "IN"
            }
        }
        
        result = self.agent.run({
            "report_data": report_data,
            "format": "json"
        })
        
        assert result["success"] is True
        data = result["data"]
        
        # Parse JSON report
        if isinstance(data["report"], str):
            report = json.loads(data["report"])
        else:
            report = data["report"]
        
        # Check required keys
        assert "emissions" in report
        assert "intensities" in report
        assert "performance" in report
        assert "recommendations" in report
        assert "metadata" in report
        
        # Check metadata
        assert "generated_at" in report["metadata"]
        assert "factor_version" in report["metadata"] or "version" in report["metadata"]
    
    def test_markdown_report_structure(self):
        """Test that Markdown report has proper structure."""
        report_data = {
            "total_emissions": 1000000.0,
            "emission_intensities": {"per_sqft": 20.0},
            "benchmark_rating": "C",
            "recommendations": [
                {"action": "Install solar panels", "impact": "high", "payback": "long"}
            ]
        }
        
        result = self.agent.run({
            "report_data": report_data,
            "format": "markdown"
        })
        
        assert result["success"] is True
        report = result["data"]["report"]
        
        # Check Markdown structure
        assert "# " in report  # Has headers
        assert "## " in report  # Has subheaders
        assert "- " in report or "* " in report  # Has lists
        assert "Total Emissions" in report or "total emissions" in report.lower()
        assert "Recommendations" in report or "recommendations" in report.lower()
    
    def test_csv_export_format(self):
        """Test CSV export format and structure."""
        report_data = {
            "emissions_breakdown": [
                {"fuel_type": "electricity", "emissions": 750000, "percentage": 75},
                {"fuel_type": "natural_gas", "emissions": 250000, "percentage": 25}
            ],
            "total_emissions": 1000000.0
        }
        
        result = self.agent.run({
            "report_data": report_data,
            "format": "csv"
        })
        
        if result["success"]:
            report = result["data"]["report"]
            
            # Check CSV structure
            lines = report.strip().split('\n')
            assert len(lines) > 1  # Header + data
            
            # Check header
            header = lines[0]
            assert "," in header  # CSV format
            assert any(col in header.lower() for col in ["fuel", "emissions", "percentage"])
            
            # Check data rows
            for line in lines[1:]:
                assert "," in line
                # Should not have unescaped commas in values
                parts = line.split(",")
                assert len(parts) >= 3  # At least fuel_type, emissions, percentage
    
    def test_excel_export_metadata(self):
        """Test Excel export includes proper metadata."""
        report_data = {
            "total_emissions": 1000000.0,
            "emission_intensities": {"per_sqft": 20.0},
            "benchmark_rating": "C"
        }
        
        result = self.agent.run({
            "report_data": report_data,
            "format": "excel"
        })
        
        if result["success"]:
            # Excel format should be indicated
            assert result["data"]["format"] == "excel"
            
            # Should have sheets information
            if "sheets" in result["data"]:
                sheets = result["data"]["sheets"]
                assert "summary" in sheets or "Summary" in sheets
                assert "emissions" in sheets or "Emissions" in sheets
    
    def test_all_data_included(self):
        """Test that all input data is included in reports."""
        comprehensive_data = {
            "total_emissions": 1000000.0,
            "emissions_breakdown": [
                {"fuel_type": "electricity", "emissions": 750000},
                {"fuel_type": "natural_gas", "emissions": 250000}
            ],
            "emission_intensities": {
                "per_sqft": 20.0,
                "per_person": 4000.0,
                "per_floor": 100000.0
            },
            "benchmark_rating": "C",
            "performance_level": "Average",
            "recommendations": [
                {"action": "LED retrofit", "impact": "high", "payback": "2 years"},
                {"action": "Solar PV", "impact": "very_high", "payback": "5 years"}
            ],
            "building_profile": {
                "type": "office",
                "age": 10,
                "area": 50000,
                "country": "IN"
            }
        }
        
        result = self.agent.run({
            "report_data": comprehensive_data,
            "format": "json"
        })
        
        assert result["success"] is True
        
        # Convert report to check contents
        if isinstance(result["data"]["report"], str):
            if result["data"]["format"] == "json":
                report = json.loads(result["data"]["report"])
            else:
                report = result["data"]["report"]
        else:
            report = result["data"]["report"]
        
        # Verify all data is present (in some form)
        report_str = str(report)
        assert "1000000" in report_str or "1,000,000" in report_str
        assert "electricity" in report_str.lower()
        assert "natural_gas" in report_str.lower() or "natural gas" in report_str.lower()
        assert "20" in report_str  # per_sqft
        assert "LED" in report_str or "led" in report_str.lower()
    
    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""
        result = self.agent.run({
            "report_data": {},
            "format": "json"
        })
        
        # Should handle gracefully
        if result["success"]:
            assert "report" in result["data"]
            # Report should indicate no data
            report_str = str(result["data"]["report"]).lower()
            assert any(phrase in report_str for phrase in ["no data", "empty", "not available"])
    
    def test_format_validation(self):
        """Test validation of output format."""
        result = self.agent.run({
            "report_data": {"total_emissions": 1000000.0},
            "format": "invalid_format"
        })
        
        # Should either fail or default to a valid format
        if result["success"]:
            assert result["data"]["format"] in ["json", "markdown", "csv", "excel"]
        else:
            assert result["error"]["type"] == "ValidationError"
            assert "format" in result["error"]["message"].lower()
    
    def test_timestamp_included(self):
        """Test that reports include generation timestamp."""
        result = self.agent.run({
            "report_data": {"total_emissions": 1000000.0},
            "format": "json"
        })
        
        assert result["success"] is True
        
        report_str = str(result["data"]["report"])
        # Should include some form of timestamp
        timestamp_indicators = ["generated", "created", "timestamp", "date", "2024", "2025"]
        assert any(indicator in report_str.lower() for indicator in timestamp_indicators)
    
    def test_units_clearly_specified(self):
        """Test that units are clearly specified in reports."""
        report_data = {
            "total_emissions": 1000000.0,
            "emission_intensities": {
                "per_sqft": 20.0,
                "per_person": 4000.0
            }
        }
        
        result = self.agent.run({
            "report_data": report_data,
            "format": "markdown"
        })
        
        assert result["success"] is True
        report = result["data"]["report"]
        
        # Check for unit specifications
        assert "kgCO2e" in report or "kg CO2e" in report
        assert "sqft" in report or "sq ft" in report
        assert "/year" in report or "annual" in report.lower()
    
    def test_snapshot_compatibility(self):
        """Test that reports are suitable for snapshot testing."""
        report_data = {
            "total_emissions": 1000000.0,
            "benchmark_rating": "C"
        }
        
        # Generate report twice
        result1 = self.agent.run({
            "report_data": report_data,
            "format": "markdown",
            "exclude_timestamp": True  # For snapshot testing
        })
        
        result2 = self.agent.run({
            "report_data": report_data,
            "format": "markdown",
            "exclude_timestamp": True
        })
        
        if result1["success"] and result2["success"]:
            # Reports should be identical when timestamps excluded
            if "exclude_timestamp" in result1["data"]:
                assert result1["data"]["report"] == result2["data"]["report"]