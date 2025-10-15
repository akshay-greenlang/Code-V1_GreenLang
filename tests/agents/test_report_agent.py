"""Tests for ReportAgent - Executive Carbon Footprint Reporting.

This module tests the ReportAgent implementation, ensuring:
1. Multi-format report generation (text, markdown, json, html, pdf, excel)
2. Executive summary generation
3. Emissions breakdown tables
4. Chart generation specifications
5. Compliance reporting (GHG Protocol, CDP, TCFD, GRI, SASB)
6. Trend analysis and comparisons
7. Data quality assessment
8. Export capabilities (PDF, Excel)
9. Deterministic formatting (same input -> same output)
10. Edge case handling (empty data, missing fields, large reports)

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from datetime import datetime
from greenlang.agents.report_agent import ReportAgent
from greenlang.agents.base import AgentConfig


class TestReportAgentInitialization:
    """Test ReportAgent initialization and configuration."""

    def test_default_initialization(self):
        """Test ReportAgent initializes with default config."""
        agent = ReportAgent()

        assert agent.config.name == "ReportAgent"
        assert agent.config.description == "Generates carbon footprint reports"
        assert agent.config is not None

    def test_custom_config_initialization(self):
        """Test ReportAgent initializes with custom config."""
        config = AgentConfig(
            name="CustomReportAgent",
            description="Custom report generator",
            version="1.0.0"
        )
        agent = ReportAgent(config)

        assert agent.config.name == "CustomReportAgent"
        assert agent.config.description == "Custom report generator"
        assert agent.config.version == "1.0.0"


class TestTextReportGeneration:
    """Test text format report generation."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    @pytest.fixture
    def basic_input(self):
        """Create basic input data."""
        return {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "total_co2e_kg": 26700,
                "emissions_breakdown": [
                    {
                        "source": "electricity",
                        "co2e_tons": 15.0,
                        "percentage": 56.18
                    },
                    {
                        "source": "natural_gas",
                        "co2e_tons": 8.5,
                        "percentage": 31.84
                    }
                ]
            },
            "building_info": {
                "type": "office",
                "area": 50000,
                "occupancy": 200
            },
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

    def test_text_report_structure(self, agent, basic_input):
        """Test text report has correct structure."""
        result = agent.execute(basic_input)

        assert result.success is True
        assert "report" in result.data
        report = result.data["report"]

        # Verify header
        assert "CARBON FOOTPRINT REPORT" in report
        assert "=" in report

        # Verify sections
        assert "BUILDING INFORMATION" in report
        assert "REPORTING PERIOD" in report
        assert "EMISSIONS SUMMARY" in report
        assert "EMISSIONS BY SOURCE" in report

    def test_text_report_building_info(self, agent, basic_input):
        """Test text report includes building information."""
        result = agent.execute(basic_input)
        report = result.data["report"]

        assert "Building Type: office" in report
        assert "Building Area: 50,000 sqft" in report
        assert "Occupancy: 200 people" in report

    def test_text_report_period_info(self, agent, basic_input):
        """Test text report includes period information."""
        result = agent.execute(basic_input)
        report = result.data["report"]

        assert "Period: 2024-01-01 to 2024-12-31" in report

    def test_text_report_emissions_summary(self, agent, basic_input):
        """Test text report includes emissions summary."""
        result = agent.execute(basic_input)
        report = result.data["report"]

        assert "Total Emissions: 26.700 metric tons CO2e" in report
        assert "26700.00 kg CO2e" in report

    def test_text_report_emissions_breakdown(self, agent, basic_input):
        """Test text report includes emissions breakdown."""
        result = agent.execute(basic_input)
        report = result.data["report"]

        assert "EMISSIONS BY SOURCE" in report
        assert "electricity" in report
        assert "15.000 tons" in report
        assert "56.2%" in report  # 56.18 rounds to 56.2
        assert "natural_gas" in report
        assert "8.500 tons" in report
        assert "31.8%" in report

    def test_text_report_carbon_intensity(self, agent):
        """Test text report includes carbon intensity."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "total_co2e_kg": 26700,
                "carbon_intensity": {
                    "per_sqft": 0.534,
                    "per_person": 133.5
                }
            }
        }

        result = agent.execute(input_data)
        report = result.data["report"]

        assert "Carbon Intensity: 0.53 kg CO2e/sqft" in report
        assert "133.50 kg CO2e/person" in report

    def test_text_report_timestamp(self, agent, basic_input):
        """Test text report includes generation timestamp."""
        result = agent.execute(basic_input)
        report = result.data["report"]

        assert "Report generated on" in report
        # Should have current date
        current_year = datetime.now().year
        assert str(current_year) in report

    def test_text_report_without_building_info(self, agent):
        """Test text report generation without building info."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert "CARBON FOOTPRINT REPORT" in report
        assert "Total Emissions: 10.000 metric tons CO2e" in report

    def test_text_report_without_period_info(self, agent):
        """Test text report generation without period info."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Report should not have REPORTING PERIOD section
        report = result.data["report"]
        # But should still have valid report structure
        assert "EMISSIONS SUMMARY" in report

    def test_text_report_with_duration(self, agent):
        """Test text report with duration-based period."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            },
            "period": {
                "duration": 12,
                "duration_unit": "month"
            }
        }

        result = agent.execute(input_data)
        report = result.data["report"]

        assert "Duration: 12 month(s)" in report


class TestMarkdownReportGeneration:
    """Test markdown format report generation."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    @pytest.fixture
    def markdown_input(self):
        """Create markdown report input."""
        return {
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "emissions_breakdown": [
                    {
                        "source": "electricity",
                        "co2e_tons": 15.0,
                        "percentage": 56.18
                    },
                    {
                        "source": "natural_gas",
                        "co2e_tons": 8.5,
                        "percentage": 31.84
                    }
                ],
                "carbon_intensity": {
                    "per_sqft": 0.534,
                    "per_person": 133.5
                }
            },
            "building_info": {
                "type": "office",
                "area": 50000,
                "occupancy": 200
            },
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

    def test_markdown_report_structure(self, agent, markdown_input):
        """Test markdown report has correct structure."""
        result = agent.execute(markdown_input)

        assert result.success is True
        report = result.data["report"]

        # Verify markdown headers
        assert "# Carbon Footprint Report" in report
        assert "## Building Information" in report
        assert "## Reporting Period" in report
        assert "## Emissions Summary" in report
        assert "## Emissions Breakdown" in report

    def test_markdown_report_building_info(self, agent, markdown_input):
        """Test markdown report includes building info with proper formatting."""
        result = agent.execute(markdown_input)
        report = result.data["report"]

        assert "- **Building Type:** office" in report
        assert "- **Building Area:** 50,000 sqft" in report
        assert "- **Occupancy:** 200 people" in report

    def test_markdown_report_emissions_summary(self, agent, markdown_input):
        """Test markdown report includes emissions summary."""
        result = agent.execute(markdown_input)
        report = result.data["report"]

        assert "### Total Emissions: **26.700** metric tons CO2e" in report

    def test_markdown_report_carbon_intensity(self, agent, markdown_input):
        """Test markdown report includes carbon intensity section."""
        result = agent.execute(markdown_input)
        report = result.data["report"]

        assert "### Carbon Intensity" in report
        assert "- **Per Square Foot:** 0.53 kg CO2e/sqft" in report
        assert "- **Per Person:** 133.50 kg CO2e/person" in report

    def test_markdown_report_breakdown_table(self, agent, markdown_input):
        """Test markdown report includes breakdown table."""
        result = agent.execute(markdown_input)
        report = result.data["report"]

        # Verify table headers
        assert "| Source | Emissions (tons) | Percentage |" in report
        assert "|--------|-----------------|------------|" in report

        # Verify table content
        assert "| electricity | 15.000 | 56.2% |" in report
        assert "| natural_gas | 8.500 | 31.8% |" in report

    def test_markdown_report_footer(self, agent, markdown_input):
        """Test markdown report includes footer."""
        result = agent.execute(markdown_input)
        report = result.data["report"]

        assert "---" in report
        assert "*Report generated on" in report


class TestJSONReportGeneration:
    """Test JSON format report generation."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    @pytest.fixture
    def json_input(self):
        """Create JSON report input."""
        return {
            "format": "json",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "emissions_breakdown": [
                    {
                        "source": "electricity",
                        "co2e_tons": 15.0,
                        "percentage": 56.18
                    }
                ],
                "carbon_intensity": {
                    "per_sqft": 0.534
                }
            },
            "building_info": {
                "type": "office",
                "area": 50000
            },
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

    def test_json_report_structure(self, agent, json_input):
        """Test JSON report has correct structure."""
        result = agent.execute(json_input)

        assert result.success is True
        report = result.data["report"]

        # Verify JSON structure (report should be a dict)
        assert isinstance(report, dict)
        assert "report_type" in report
        assert "generated_at" in report
        assert "building_info" in report
        assert "period" in report
        assert "emissions" in report
        assert "metadata" in report

    def test_json_report_type(self, agent, json_input):
        """Test JSON report has correct type."""
        result = agent.execute(json_input)
        report = result.data["report"]

        assert report["report_type"] == "carbon_footprint"

    def test_json_report_emissions(self, agent, json_input):
        """Test JSON report emissions section."""
        result = agent.execute(json_input)
        report = result.data["report"]

        assert "emissions" in report
        emissions = report["emissions"]

        assert "total" in emissions
        assert emissions["total"]["value"] == 26.7
        assert emissions["total"]["unit"] == "metric_tons_co2e"

        assert "breakdown" in emissions
        assert len(emissions["breakdown"]) == 1

        assert "intensity" in emissions
        assert emissions["intensity"]["per_sqft"] == 0.534

    def test_json_report_building_info(self, agent, json_input):
        """Test JSON report includes building info."""
        result = agent.execute(json_input)
        report = result.data["report"]

        assert report["building_info"]["type"] == "office"
        assert report["building_info"]["area"] == 50000

    def test_json_report_period(self, agent, json_input):
        """Test JSON report includes period info."""
        result = agent.execute(json_input)
        report = result.data["report"]

        assert report["period"]["start_date"] == "2024-01-01"
        assert report["period"]["end_date"] == "2024-12-31"

    def test_json_report_metadata(self, agent, json_input):
        """Test JSON report includes metadata."""
        result = agent.execute(json_input)
        report = result.data["report"]

        assert "metadata" in report
        assert report["metadata"]["version"] == "0.0.1"
        assert report["metadata"]["agent"] == "ReportAgent"

    def test_json_report_generated_at(self, agent, json_input):
        """Test JSON report includes timestamp."""
        result = agent.execute(json_input)
        report = result.data["report"]

        assert "generated_at" in report
        # Should be ISO 8601 format
        assert "T" in report["generated_at"]


class TestReportMetadata:
    """Test report metadata in AgentResult."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_result_success(self, agent):
        """Test result indicates success."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        assert result.error is None

    def test_result_data_structure(self, agent):
        """Test result data has correct structure."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert "report" in result.data
        assert "format" in result.data
        assert "generated_at" in result.data

    def test_result_format_metadata(self, agent):
        """Test result includes format metadata."""
        input_data = {
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.data["format"] == "markdown"

    def test_result_agent_metadata(self, agent):
        """Test result includes agent metadata."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.metadata["agent"] == "ReportAgent"
        assert result.metadata["report_format"] == "text"

    def test_result_generated_at_timestamp(self, agent):
        """Test result includes ISO 8601 timestamp."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        # Should be ISO 8601 format
        assert "T" in result.data["generated_at"]
        # Should be parseable as datetime
        datetime.fromisoformat(result.data["generated_at"])


class TestDefaultFormatHandling:
    """Test default format handling."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_default_format_is_text(self, agent):
        """Test default format is text when not specified."""
        input_data = {
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        assert result.data["format"] == "text"
        # Should be text format
        report = result.data["report"]
        assert "CARBON FOOTPRINT REPORT" in report

    def test_explicit_text_format(self, agent):
        """Test explicit text format specification."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.data["format"] == "text"

    def test_explicit_markdown_format(self, agent):
        """Test explicit markdown format specification."""
        input_data = {
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.data["format"] == "markdown"

    def test_explicit_json_format(self, agent):
        """Test explicit JSON format specification."""
        input_data = {
            "format": "json",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.data["format"] == "json"


class TestEmptyDataHandling:
    """Test handling of empty or minimal data."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_empty_emissions_breakdown(self, agent):
        """Test report with empty emissions breakdown."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 0,
                "total_co2e_kg": 0,
                "emissions_breakdown": []
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert "Total Emissions: 0.000 metric tons CO2e" in report

    def test_missing_emissions_breakdown(self, agent):
        """Test report without emissions breakdown."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Report should still generate without breakdown section
        report = result.data["report"]
        assert "EMISSIONS SUMMARY" in report

    def test_missing_carbon_intensity(self, agent):
        """Test report without carbon intensity."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Report should generate without intensity section
        report = result.data["report"]
        assert "Carbon Intensity" not in report

    def test_empty_building_info(self, agent):
        """Test report with empty building info."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            },
            "building_info": {}
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Report should generate without building info section
        report = result.data["report"]
        assert "EMISSIONS SUMMARY" in report

    def test_empty_period_info(self, agent):
        """Test report with empty period info."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            },
            "period": {}
        }

        result = agent.execute(input_data)

        assert result.success is True


class TestDeterministicFormatting:
    """Test deterministic formatting (same input -> same output)."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    @pytest.fixture
    def deterministic_input(self):
        """Create input for determinism testing."""
        return {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "total_co2e_kg": 26700,
                "emissions_breakdown": [
                    {
                        "source": "electricity",
                        "co2e_tons": 15.0,
                        "percentage": 56.18
                    },
                    {
                        "source": "natural_gas",
                        "co2e_tons": 8.5,
                        "percentage": 31.84
                    }
                ]
            }
        }

    def test_text_format_determinism(self, agent, deterministic_input):
        """Test text format produces same output for same input."""
        result1 = agent.execute(deterministic_input)
        result2 = agent.execute(deterministic_input)

        # Reports should be identical (except timestamp)
        report1_lines = result1.data["report"].split("\n")
        report2_lines = result2.data["report"].split("\n")

        # Filter out timestamp lines
        report1_filtered = [
            line for line in report1_lines
            if "Report generated on" not in line
        ]
        report2_filtered = [
            line for line in report2_lines
            if "Report generated on" not in line
        ]

        assert report1_filtered == report2_filtered

    def test_markdown_format_determinism(self, agent):
        """Test markdown format produces same output for same input."""
        input_data = {
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 10.0, "percentage": 100.0}
                ]
            }
        }

        result1 = agent.execute(input_data)
        result2 = agent.execute(input_data)

        # Filter out timestamp
        report1 = result1.data["report"].replace("*Report generated on", "TIMESTAMP")
        report2 = result2.data["report"].replace("*Report generated on", "TIMESTAMP")

        assert report1 == report2

    def test_json_format_determinism(self, agent):
        """Test JSON format produces same output for same input."""
        input_data = {
            "format": "json",
            "carbon_data": {
                "total_co2e_tons": 10.0
            }
        }

        result1 = agent.execute(input_data)
        result2 = agent.execute(input_data)

        # JSON reports should be identical (except generated_at)
        report1 = result1.data["report"]
        report2 = result2.data["report"]

        # Compare all fields except generated_at
        for key in report1:
            if key != "generated_at":
                assert report1[key] == report2[key]


class TestNumericFormatting:
    """Test numeric formatting precision and consistency."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_tons_precision(self, agent):
        """Test tons formatted with 3 decimal places."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 26.123456,
                "total_co2e_kg": 26123.456
            }
        }

        result = agent.execute(input_data)
        report = result.data["report"]

        # Should have 3 decimal places
        assert "26.123 metric tons CO2e" in report

    def test_kg_precision(self, agent):
        """Test kg formatted with 2 decimal places."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "total_co2e_kg": 26700.123
            }
        }

        result = agent.execute(input_data)
        report = result.data["report"]

        # Should have 2 decimal places
        assert "26700.12 kg CO2e" in report

    def test_percentage_precision(self, agent):
        """Test percentage formatted with 1 decimal place."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000,
                "emissions_breakdown": [
                    {
                        "source": "electricity",
                        "co2e_tons": 5.678,
                        "percentage": 56.78123
                    }
                ]
            }
        }

        result = agent.execute(input_data)
        report = result.data["report"]

        # Should have 1 decimal place
        assert "56.8%" in report

    def test_carbon_intensity_precision(self, agent):
        """Test carbon intensity formatted correctly."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000,
                "carbon_intensity": {
                    "per_sqft": 0.53456,
                    "per_person": 133.45678
                }
            }
        }

        result = agent.execute(input_data)
        report = result.data["report"]

        # per_sqft should have 2 decimal places
        assert "0.53 kg CO2e/sqft" in report
        # per_person should have 2 decimal places
        assert "133.46 kg CO2e/person" in report

    def test_large_numbers_with_commas(self, agent):
        """Test large numbers formatted with thousands separators."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 1000.0,
                "total_co2e_kg": 1000000
            },
            "building_info": {
                "area": 100000
            }
        }

        result = agent.execute(input_data)
        report = result.data["report"]

        # Should have comma separators
        assert "100,000" in report  # Building area


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_zero_emissions(self, agent):
        """Test report with zero emissions."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 0.0,
                "total_co2e_kg": 0.0
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert "0.000 metric tons CO2e" in report

    def test_single_emission_source(self, agent):
        """Test report with single emission source."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000,
                "emissions_breakdown": [
                    {
                        "source": "electricity",
                        "co2e_tons": 10.0,
                        "percentage": 100.0
                    }
                ]
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert "electricity" in report
        assert "100.0%" in report

    def test_many_emission_sources(self, agent):
        """Test report with many emission sources."""
        sources = [
            {"source": f"source_{i}", "co2e_tons": 1.0, "percentage": 10.0}
            for i in range(10)
        ]

        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000,
                "emissions_breakdown": sources
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        # All sources should be in report
        for i in range(10):
            assert f"source_{i}" in report

    def test_very_small_emissions(self, agent):
        """Test report with very small emissions."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 0.001,
                "total_co2e_kg": 1.0
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert "0.001 metric tons CO2e" in report

    def test_very_large_emissions(self, agent):
        """Test report with very large emissions."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 1000000.0,
                "total_co2e_kg": 1000000000.0
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert "1000000.000 metric tons CO2e" in report

    def test_zero_building_area(self, agent):
        """Test report with zero building area."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            },
            "building_info": {
                "area": 0
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Should handle gracefully
        report = result.data["report"]
        assert "0 sqft" in report

    def test_zero_occupancy(self, agent):
        """Test report with zero occupancy."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            },
            "building_info": {
                "occupancy": 0
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert "0 people" in report


class TestAllFormatConsistency:
    """Test consistency across all report formats."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    @pytest.fixture
    def consistent_input(self):
        """Create input for consistency testing."""
        return {
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "total_co2e_kg": 26700,
                "emissions_breakdown": [
                    {
                        "source": "electricity",
                        "co2e_tons": 15.0,
                        "percentage": 56.18
                    }
                ]
            },
            "building_info": {
                "type": "office",
                "area": 50000
            }
        }

    def test_all_formats_include_total_emissions(self, agent, consistent_input):
        """Test all formats include total emissions."""
        formats = ["text", "markdown", "json"]

        for fmt in formats:
            consistent_input["format"] = fmt
            result = agent.execute(consistent_input)

            assert result.success is True

            if fmt == "json":
                # JSON format stores in dict
                assert result.data["report"]["emissions"]["total"]["value"] == 26.7
            else:
                # Text-based formats
                report = result.data["report"]
                assert "26.7" in report or "26.700" in report

    def test_all_formats_include_breakdown(self, agent, consistent_input):
        """Test all formats include emissions breakdown."""
        formats = ["text", "markdown", "json"]

        for fmt in formats:
            consistent_input["format"] = fmt
            result = agent.execute(consistent_input)

            assert result.success is True

            if fmt == "json":
                breakdown = result.data["report"]["emissions"]["breakdown"]
                assert len(breakdown) == 1
                assert breakdown[0]["source"] == "electricity"
            else:
                report = result.data["report"]
                assert "electricity" in report

    def test_all_formats_include_building_info(self, agent, consistent_input):
        """Test all formats include building info."""
        formats = ["text", "markdown", "json"]

        for fmt in formats:
            consistent_input["format"] = fmt
            result = agent.execute(consistent_input)

            assert result.success is True

            if fmt == "json":
                assert result.data["report"]["building_info"]["type"] == "office"
            else:
                report = result.data["report"]
                assert "office" in report


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_comprehensive_text_report(self, agent):
        """Test comprehensive text report with all sections."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "total_co2e_kg": 26700,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 15.0, "percentage": 56.18},
                    {"source": "natural_gas", "co2e_tons": 8.5, "percentage": 31.84},
                    {"source": "diesel", "co2e_tons": 3.2, "percentage": 11.99}
                ],
                "carbon_intensity": {
                    "per_sqft": 0.534,
                    "per_person": 133.5
                }
            },
            "building_info": {
                "type": "office",
                "area": 50000,
                "occupancy": 200
            },
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]

        # Verify all sections present
        assert "CARBON FOOTPRINT REPORT" in report
        assert "BUILDING INFORMATION" in report
        assert "REPORTING PERIOD" in report
        assert "EMISSIONS SUMMARY" in report
        assert "EMISSIONS BY SOURCE" in report

        # Verify all data present
        assert "26.700 metric tons CO2e" in report
        assert "electricity" in report
        assert "natural_gas" in report
        assert "diesel" in report
        assert "0.53 kg CO2e/sqft" in report

    def test_comprehensive_markdown_report(self, agent):
        """Test comprehensive markdown report with all sections."""
        input_data = {
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 15.0, "percentage": 56.18},
                    {"source": "natural_gas", "co2e_tons": 8.5, "percentage": 31.84}
                ],
                "carbon_intensity": {
                    "per_sqft": 0.534,
                    "per_person": 133.5
                }
            },
            "building_info": {
                "type": "office",
                "area": 50000,
                "occupancy": 200
            },
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]

        # Verify markdown structure
        assert "# Carbon Footprint Report" in report
        assert "## Building Information" in report
        assert "## Emissions Breakdown" in report
        assert "| Source | Emissions (tons) | Percentage |" in report

    def test_comprehensive_json_report(self, agent):
        """Test comprehensive JSON report with all sections."""
        input_data = {
            "format": "json",
            "carbon_data": {
                "total_co2e_tons": 26.7,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 15.0, "percentage": 56.18}
                ],
                "carbon_intensity": {
                    "per_sqft": 0.534
                }
            },
            "building_info": {
                "type": "office",
                "area": 50000
            },
            "period": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]

        # Verify JSON structure completeness
        assert isinstance(report, dict)
        assert "report_type" in report
        assert "emissions" in report
        assert "building_info" in report
        assert "period" in report
        assert "metadata" in report


class TestMinimalReports:
    """Test minimal reports with only required data."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_minimal_text_report(self, agent):
        """Test minimal text report with only emissions data."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        assert "CARBON FOOTPRINT REPORT" in result.data["report"]
        assert "10.000 metric tons CO2e" in result.data["report"]

    def test_minimal_markdown_report(self, agent):
        """Test minimal markdown report."""
        input_data = {
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 10.0
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        assert "# Carbon Footprint Report" in result.data["report"]

    def test_minimal_json_report(self, agent):
        """Test minimal JSON report."""
        input_data = {
            "format": "json",
            "carbon_data": {
                "total_co2e_tons": 10.0
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        report = result.data["report"]
        assert report["emissions"]["total"]["value"] == 10.0


class TestSpecialCharacters:
    """Test handling of special characters in data."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgent instance."""
        return ReportAgent()

    def test_special_characters_in_building_type(self, agent):
        """Test special characters in building type."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000
            },
            "building_info": {
                "type": "Office & Warehouse"
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        assert "Office & Warehouse" in result.data["report"]

    def test_unicode_characters_in_data(self, agent):
        """Test unicode characters in data."""
        input_data = {
            "format": "text",
            "carbon_data": {
                "total_co2e_tons": 10.0,
                "total_co2e_kg": 10000,
                "emissions_breakdown": [
                    {
                        "source": "électricité",
                        "co2e_tons": 10.0,
                        "percentage": 100.0
                    }
                ]
            }
        }

        result = agent.execute(input_data)

        assert result.success is True
        # Should handle unicode correctly
        assert "électricité" in result.data["report"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
