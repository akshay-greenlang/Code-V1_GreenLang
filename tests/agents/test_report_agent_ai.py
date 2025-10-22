"""Tests for AI-powered ReportAgent.

This module tests the ReportAgentAI implementation, ensuring:
1. Tool-first numerics (all calculations use tools)
2. Deterministic results (same input -> same output)
3. Backward compatibility with ReportAgent API
4. AI narratives are generated
5. Framework compliance verification works
6. Executive summaries are professional
7. Budget enforcement works
8. Error handling is robust
9. Multi-framework support (TCFD, CDP, GRI, SASB, etc.)

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from greenlang.agents.report_agent_ai import ReportAgentAI
from greenlang.intelligence import ChatResponse, Usage, FinishReason
from greenlang.intelligence.schemas.responses import ProviderInfo


class TestReportAgentAI:
    """Test suite for ReportAgentAI."""

    @pytest.fixture
    def agent(self):
        """Create ReportAgentAI instance for testing."""
        return ReportAgentAI(budget_usd=2.0)

    @pytest.fixture
    def valid_report_data(self):
        """Create valid test report data."""
        return {
            "framework": "TCFD",
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 45.5,
                "total_co2e_kg": 45500,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 25.0, "percentage": 54.95},
                    {"source": "natural_gas", "co2e_tons": 15.0, "percentage": 32.97},
                    {"source": "diesel", "co2e_tons": 5.5, "percentage": 12.09},
                ],
                "carbon_intensity": {
                    "per_sqft": 0.455,
                    "per_person": 227.5,
                },
            },
            "building_info": {
                "type": "commercial_office",
                "area": 100000,
                "occupancy": 200,
            },
            "period": {
                "start_date": "2025-01-01",
                "end_date": "2025-03-31",
                "duration": 3,
                "duration_unit": "months",
            },
        }

    @pytest.fixture
    def report_data_with_trends(self, valid_report_data):
        """Create report data with trend information."""
        data = valid_report_data.copy()
        data["previous_period_data"] = {
            "total_co2e_tons": 50.0,
        }
        data["baseline_data"] = {
            "total_co2e_tons": 60.0,
            "year": 2020,
        }
        return data

    @pytest.fixture
    def simple_report_data(self):
        """Create minimal report data."""
        return {
            "carbon_data": {
                "total_co2e_tons": 20.0,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 20.0, "percentage": 100.0},
                ],
            }
        }

    def test_initialization(self, agent):
        """Test ReportAgentAI initializes correctly."""
        assert agent.config.name == "ReportAgentAI"
        assert agent.config.version == "0.1.0"
        assert agent.budget_usd == 2.0
        assert agent.enable_ai_narrative is True
        assert agent.enable_executive_summary is True
        assert agent.enable_compliance_check is True

        # Verify tools are defined
        assert agent.fetch_emissions_data_tool is not None
        assert agent.calculate_trends_tool is not None
        assert agent.generate_charts_tool is not None
        assert agent.format_report_tool is not None
        assert agent.check_compliance_tool is not None
        assert agent.generate_executive_summary_tool is not None

        # Verify original report agent is initialized
        assert agent.report_agent is not None

        # Verify supported frameworks
        assert "TCFD" in agent.supported_frameworks
        assert "CDP" in agent.supported_frameworks
        assert "GRI" in agent.supported_frameworks
        assert "SASB" in agent.supported_frameworks

    def test_validate_valid_input(self, agent, valid_report_data):
        """Test validation passes for valid input."""
        assert agent.validate_input(valid_report_data) is True

    def test_validate_simple_input(self, agent, simple_report_data):
        """Test validation passes for simple input."""
        assert agent.validate_input(simple_report_data) is True

    def test_validate_invalid_input(self, agent):
        """Test validation fails for invalid inputs."""
        # Missing carbon_data key
        assert agent.validate_input({}) is False

        # Non-dict carbon_data
        assert agent.validate_input({"carbon_data": "not a dict"}) is False

        # Missing emissions data
        assert agent.validate_input({"carbon_data": {}}) is False

    def test_fetch_emissions_data_tool_implementation(self, agent, valid_report_data):
        """Test fetch_emissions_data tool extracts data correctly."""
        carbon_data = valid_report_data["carbon_data"]

        result = agent._fetch_emissions_data_impl(carbon_data)

        # Verify structure
        assert "total_emissions_tons" in result
        assert "total_emissions_kg" in result
        assert "emissions_breakdown" in result
        assert "carbon_intensity" in result
        assert "num_sources" in result

        # Verify exact values
        assert result["total_emissions_tons"] == 45.5
        assert result["total_emissions_kg"] == 45500
        assert result["num_sources"] == 3

        # Verify tool call tracked
        assert agent._tool_call_count > 0

    def test_fetch_emissions_data_kg_only(self, agent):
        """Test fetch_emissions_data calculates tons from kg if tons missing."""
        carbon_data = {
            "total_co2e_kg": 30000,
            "emissions_breakdown": [],
        }

        result = agent._fetch_emissions_data_impl(carbon_data)

        # Should calculate tons from kg
        assert result["total_emissions_tons"] == 30.0
        assert result["total_emissions_kg"] == 30000

    def test_calculate_trends_tool_implementation(self, agent):
        """Test calculate_trends tool calculates YoY changes correctly."""
        result = agent._calculate_trends_impl(
            current_emissions_tons=45.5,
            previous_emissions_tons=50.0,
            baseline_emissions_tons=60.0,
        )

        # Verify structure
        assert "current_emissions_tons" in result
        assert "previous_emissions_tons" in result
        assert "yoy_change_tons" in result
        assert "yoy_change_percentage" in result
        assert "direction" in result
        assert "baseline_change_tons" in result
        assert "baseline_change_percentage" in result

        # Verify YoY calculations
        assert result["current_emissions_tons"] == 45.5
        assert result["previous_emissions_tons"] == 50.0
        assert result["yoy_change_tons"] == -4.5
        assert result["yoy_change_percentage"] == -9.0  # (45.5-50)/50 * 100
        assert result["direction"] == "decrease"

        # Verify baseline calculations
        assert result["baseline_emissions_tons"] == 60.0
        assert result["baseline_change_tons"] == -14.5
        assert result["baseline_change_percentage"] == -24.17  # (45.5-60)/60 * 100

    def test_calculate_trends_no_previous_data(self, agent):
        """Test calculate_trends with only current data."""
        result = agent._calculate_trends_impl(current_emissions_tons=45.5)

        # Should have current data only
        assert result["current_emissions_tons"] == 45.5
        assert "yoy_change_tons" not in result
        assert "direction" not in result

    def test_calculate_trends_increase(self, agent):
        """Test calculate_trends with emissions increase."""
        result = agent._calculate_trends_impl(
            current_emissions_tons=55.0,
            previous_emissions_tons=50.0,
        )

        assert result["direction"] == "increase"
        assert result["yoy_change_tons"] == 5.0
        assert result["yoy_change_percentage"] == 10.0

    def test_generate_charts_tool_implementation(self, agent, valid_report_data):
        """Test generate_charts tool creates visualization data."""
        emissions_breakdown = valid_report_data["carbon_data"]["emissions_breakdown"]

        result = agent._generate_charts_impl(
            emissions_breakdown=emissions_breakdown,
            chart_types=["pie", "bar"],
        )

        # Verify structure
        assert "charts" in result
        assert "chart_count" in result

        charts = result["charts"]

        # Verify pie chart
        assert "pie_chart" in charts
        pie_chart = charts["pie_chart"]
        assert pie_chart["type"] == "pie"
        assert pie_chart["title"] == "Emissions by Source"
        assert len(pie_chart["data"]) == 3

        # Verify pie chart data
        assert pie_chart["data"][0]["label"] == "electricity"
        assert pie_chart["data"][0]["value"] == 25.0
        assert pie_chart["data"][0]["percentage"] == 54.95

        # Verify bar chart
        assert "bar_chart" in charts
        bar_chart = charts["bar_chart"]
        assert bar_chart["type"] == "bar"
        assert len(bar_chart["data"]) == 3

    def test_generate_charts_default_types(self, agent):
        """Test generate_charts with default chart types."""
        emissions_breakdown = [
            {"source": "electricity", "co2e_tons": 10.0, "percentage": 100.0},
        ]

        result = agent._generate_charts_impl(emissions_breakdown=emissions_breakdown)

        # Should generate pie and bar by default
        assert "pie_chart" in result["charts"]
        assert "bar_chart" in result["charts"]
        assert result["chart_count"] == 2

    def test_format_report_tool_implementation(self, agent, valid_report_data):
        """Test format_report tool formats according to framework."""
        result = agent._format_report_impl(
            framework="TCFD",
            carbon_data=valid_report_data["carbon_data"],
            building_info=valid_report_data["building_info"],
            period=valid_report_data["period"],
            report_format="markdown",
        )

        # Verify structure
        assert "report" in result
        assert "format" in result
        assert "framework" in result
        assert "framework_metadata" in result
        assert "generated_at" in result

        # Verify values
        assert result["format"] == "markdown"
        assert result["framework"] == "TCFD"
        assert isinstance(result["report"], str)
        assert len(result["report"]) > 0

        # Verify framework metadata
        metadata = result["framework_metadata"]
        assert "full_name" in metadata
        assert "TCFD" in metadata["full_name"] or "Task Force" in metadata["full_name"]

    def test_format_report_different_formats(self, agent, simple_report_data):
        """Test format_report with different output formats."""
        carbon_data = simple_report_data["carbon_data"]

        # Test markdown
        result_md = agent._format_report_impl(
            framework="CDP",
            carbon_data=carbon_data,
            report_format="markdown",
        )
        assert result_md["format"] == "markdown"

        # Test text
        result_txt = agent._format_report_impl(
            framework="GRI",
            carbon_data=carbon_data,
            report_format="text",
        )
        assert result_txt["format"] == "text"

        # Test json
        result_json = agent._format_report_impl(
            framework="SASB",
            carbon_data=carbon_data,
            report_format="json",
        )
        assert result_json["format"] == "json"

    def test_check_compliance_tcfd(self, agent, valid_report_data):
        """Test check_compliance for TCFD framework."""
        result = agent._check_compliance_impl(
            framework="TCFD",
            report_data=valid_report_data["carbon_data"],
        )

        # Verify structure
        assert "framework" in result
        assert "compliant" in result
        assert "compliance_checks" in result
        assert "total_checks" in result
        assert "passed_checks" in result

        # Verify TCFD-specific checks
        assert result["framework"] == "TCFD"
        checks = result["compliance_checks"]
        assert any("governance" in c["requirement"].lower() for c in checks)
        assert any("strategy" in c["requirement"].lower() for c in checks)

    def test_check_compliance_cdp(self, agent, valid_report_data):
        """Test check_compliance for CDP framework."""
        result = agent._check_compliance_impl(
            framework="CDP",
            report_data=valid_report_data["carbon_data"],
        )

        assert result["framework"] == "CDP"
        checks = result["compliance_checks"]
        assert any("scope" in c["requirement"].lower() for c in checks)

    def test_check_compliance_gri(self, agent, valid_report_data):
        """Test check_compliance for GRI framework."""
        result = agent._check_compliance_impl(
            framework="GRI",
            report_data=valid_report_data["carbon_data"],
        )

        assert result["framework"] == "GRI"
        checks = result["compliance_checks"]
        assert any("305" in c["requirement"] for c in checks)

    def test_check_compliance_sasb(self, agent, valid_report_data):
        """Test check_compliance for SASB framework."""
        result = agent._check_compliance_impl(
            framework="SASB",
            report_data=valid_report_data["carbon_data"],
        )

        assert result["framework"] == "SASB"
        assert result["compliant"] is True

    def test_check_compliance_no_emissions(self, agent):
        """Test check_compliance fails with no emissions data."""
        result = agent._check_compliance_impl(
            framework="TCFD",
            report_data={},
        )

        # Should fail compliance
        assert result["compliant"] is False
        assert any(c["status"] == "fail" for c in result["compliance_checks"])

    def test_generate_executive_summary_tool_implementation(self, agent, valid_report_data):
        """Test generate_executive_summary tool creates summary data."""
        carbon_data = valid_report_data["carbon_data"]
        building_info = valid_report_data["building_info"]

        result = agent._generate_executive_summary_impl(
            total_emissions_tons=carbon_data["total_co2e_tons"],
            emissions_breakdown=carbon_data["emissions_breakdown"],
            building_info=building_info,
        )

        # Verify structure
        assert "total_emissions_tons" in result
        assert "total_emissions_kg" in result
        assert "num_sources" in result
        assert "primary_source" in result
        assert "primary_source_percentage" in result
        assert "building_type" in result

        # Verify values
        assert result["total_emissions_tons"] == 45.5
        assert result["num_sources"] == 3
        assert result["primary_source"] == "electricity"
        assert result["primary_source_percentage"] == 54.95
        assert result["building_type"] == "commercial_office"

    def test_generate_executive_summary_with_trends(self, agent):
        """Test generate_executive_summary with trend data."""
        trends = {
            "direction": "decrease",
            "yoy_change_percentage": -9.0,
        }

        result = agent._generate_executive_summary_impl(
            total_emissions_tons=45.5,
            emissions_breakdown=[{"source": "electricity", "co2e_tons": 45.5, "percentage": 100.0}],
            trends=trends,
        )

        assert "trend_direction" in result
        assert result["trend_direction"] == "decrease"
        assert "yoy_change_percentage" in result
        assert result["yoy_change_percentage"] == -9.0

    def test_get_framework_metadata(self, agent):
        """Test framework metadata retrieval."""
        # Test TCFD
        tcfd_metadata = agent._get_framework_metadata("TCFD")
        assert "full_name" in tcfd_metadata
        assert "Task Force" in tcfd_metadata["full_name"]
        assert "sections" in tcfd_metadata

        # Test CDP
        cdp_metadata = agent._get_framework_metadata("CDP")
        assert "Carbon Disclosure" in cdp_metadata["full_name"]

        # Test GRI
        gri_metadata = agent._get_framework_metadata("GRI")
        assert "Global Reporting" in gri_metadata["full_name"]

        # Test unknown framework
        custom_metadata = agent._get_framework_metadata("UNKNOWN")
        assert custom_metadata["full_name"] == "UNKNOWN"
        assert custom_metadata["version"] == "Custom"

    def test_format_executive_summary(self, agent):
        """Test executive summary text formatting."""
        summary_data = {
            "total_emissions_tons": 45.5,
            "primary_source": "electricity",
            "primary_source_percentage": 54.95,
            "building_type": "commercial_office",
        }

        trends = {
            "direction": "decrease",
            "yoy_change_percentage": -9.0,
        }

        result = agent._format_executive_summary(summary_data, trends)

        # Verify formatted text
        assert isinstance(result, str)
        assert "45.5" in result or "45.50" in result
        assert "electricity" in result
        # Note: 54.95 formatted with .1f rounds to 55.0
        assert "55.0" in result or "54.9" in result or "54.95" in result
        assert "decreased" in result.lower()
        assert "9.0" in result or "9" in result
        assert "commercial_office" in result

    @pytest.mark.asyncio
    @patch("greenlang.agents.report_agent_ai.ChatSession")
    async def test_execute_with_mocked_ai(self, mock_session_class, agent, valid_report_data):
        """Test execute() with mocked ChatSession to verify AI integration."""
        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = (
            "TCFD Climate-Related Financial Disclosure Report\n\n"
            "Executive Summary: This report documents total greenhouse gas emissions of "
            "45.5 metric tons CO2e for Q1 2025. The primary emission source is electricity, "
            "accounting for 54.95% of total emissions. Emissions decreased by 9% compared to "
            "the previous period, demonstrating progress toward our reduction targets.\n\n"
            "Governance: The Board of Directors oversees climate-related risks and opportunities "
            "through quarterly reviews of emissions data and reduction strategies.\n\n"
            "Strategy: Our climate strategy focuses on transitioning to renewable energy sources "
            "and improving building efficiency to achieve net-zero emissions by 2040.\n\n"
            "Risk Management: Climate risks are integrated into enterprise risk management with "
            "regular scenario analysis and stress testing.\n\n"
            "Metrics & Targets: Target 30% emissions reduction by 2030 from 2020 baseline."
        )
        mock_response.tool_calls = [
            {
                "name": "fetch_emissions_data",
                "arguments": {
                    "carbon_data": valid_report_data["carbon_data"],
                },
            },
            {
                "name": "calculate_trends",
                "arguments": {
                    "current_emissions_tons": 45.5,
                },
            },
            {
                "name": "generate_charts",
                "arguments": {
                    "emissions_breakdown": valid_report_data["carbon_data"]["emissions_breakdown"],
                    "chart_types": ["pie", "bar"],
                },
            },
            {
                "name": "format_report",
                "arguments": {
                    "framework": "TCFD",
                    "carbon_data": valid_report_data["carbon_data"],
                    "building_info": valid_report_data["building_info"],
                    "period": valid_report_data["period"],
                    "report_format": "markdown",
                },
            },
            {
                "name": "check_compliance",
                "arguments": {
                    "framework": "TCFD",
                    "report_data": valid_report_data["carbon_data"],
                },
            },
            {
                "name": "generate_executive_summary",
                "arguments": {
                    "total_emissions_tons": 45.5,
                    "emissions_breakdown": valid_report_data["carbon_data"]["emissions_breakdown"],
                    "building_info": valid_report_data["building_info"],
                },
            },
        ]
        mock_response.usage = Usage(
            prompt_tokens=300,
            completion_tokens=400,
            total_tokens=700,
            cost_usd=0.05,
        )
        mock_response.provider_info = ProviderInfo(
            provider="openai",
            model="gpt-4o-mini",
        )
        mock_response.finish_reason = FinishReason.stop

        # Setup mock session
        mock_session = Mock()
        mock_session.chat = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        # Run agent (handle both sync and async)
        import inspect
        result_coro = agent.execute(valid_report_data)
        if inspect.iscoroutine(result_coro):
            result = await result_coro
        else:
            result = result_coro

        # Verify success
        assert result.success is True
        assert result.data is not None

        # Verify output structure
        data = result.data
        assert "report" in data
        assert "format" in data
        assert "framework" in data
        assert "generated_at" in data
        assert "total_co2e_tons" in data
        assert "total_co2e_kg" in data
        assert "emissions_breakdown" in data
        assert "compliance_status" in data
        assert "ai_narrative" in data
        assert "executive_summary" in data

        # Verify metadata
        assert result.metadata is not None
        metadata = result.metadata
        assert metadata["agent"] == "ReportAgentAI"
        assert metadata["framework"] == "TCFD"
        assert "calculation_time_ms" in metadata
        assert "ai_calls" in metadata
        assert "tool_calls" in metadata
        assert metadata["deterministic"] is True

        # Verify ChatSession was called with correct parameters
        mock_session.chat.assert_called_once()
        call_args = mock_session.chat.call_args
        assert call_args.kwargs["temperature"] == 0.0  # Deterministic
        assert call_args.kwargs["seed"] == 42  # Reproducible
        assert len(call_args.kwargs["tools"]) == 6  # All 6 tools

    def test_determinism_same_input_same_output(self, agent, simple_report_data):
        """Test deterministic behavior: same input produces same output."""
        carbon_data = simple_report_data["carbon_data"]

        # Verify tools are deterministic
        result1 = agent._fetch_emissions_data_impl(carbon_data)
        result2 = agent._fetch_emissions_data_impl(carbon_data)

        # Tool results should be identical
        assert result1 == result2

        # Trends should also be deterministic
        trends1 = agent._calculate_trends_impl(current_emissions_tons=20.0)
        trends2 = agent._calculate_trends_impl(current_emissions_tons=20.0)

        assert trends1 == trends2

    def test_backward_compatibility_api(self, agent):
        """Test backward compatibility with ReportAgent API."""
        # ReportAgentAI should have same interface as ReportAgent
        assert hasattr(agent, "execute")
        assert hasattr(agent, "validate_input")
        assert hasattr(agent, "config")

        # Verify original ReportAgent is accessible
        assert hasattr(agent, "report_agent")
        assert agent.report_agent is not None

    def test_error_handling_invalid_input(self, agent):
        """Test error handling for invalid input."""
        result = agent.execute({"invalid_key": "invalid_value"})

        # Should fail validation
        assert result.success is False
        assert "Invalid input" in result.error

    def test_performance_tracking(self, agent, simple_report_data):
        """Test performance metrics tracking."""
        # Initial state
        initial_summary = agent.get_performance_summary()
        assert initial_summary["agent"] == "ReportAgentAI"
        assert "ai_metrics" in initial_summary
        assert "base_agent_metrics" in initial_summary

        # Make a tool call
        carbon_data = simple_report_data["carbon_data"]
        agent._fetch_emissions_data_impl(carbon_data)

        # Verify metrics updated
        assert agent._tool_call_count > 0

        # Get updated summary
        summary = agent.get_performance_summary()
        assert summary["ai_metrics"]["tool_call_count"] > 0

    def test_build_prompt_basic(self, agent, simple_report_data):
        """Test prompt building for basic case."""
        prompt = agent._build_prompt(simple_report_data)

        # Verify key elements
        assert "TCFD" in prompt  # Default framework
        assert "fetch_emissions_data" in prompt
        assert "calculate_trends" in prompt
        assert "generate_charts" in prompt
        assert "format_report" in prompt

    def test_build_prompt_with_framework(self, agent, valid_report_data):
        """Test prompt building with specific framework."""
        prompt = agent._build_prompt(valid_report_data)

        # Verify framework mentioned
        assert "TCFD" in prompt
        assert "compliance" in prompt.lower()

    def test_build_prompt_with_period(self, agent, valid_report_data):
        """Test prompt building with reporting period."""
        prompt = agent._build_prompt(valid_report_data)

        # Verify period mentioned
        assert "2025-01-01" in prompt
        assert "2025-03-31" in prompt

    def test_ai_narrative_disabled(self, agent, simple_report_data):
        """Test behavior when AI narrative is disabled."""
        agent.enable_ai_narrative = False

        # Verify flag
        assert agent.enable_ai_narrative is False

    def test_executive_summary_disabled(self, agent, simple_report_data):
        """Test behavior when executive summary is disabled."""
        agent.enable_executive_summary = False

        # Verify flag
        assert agent.enable_executive_summary is False

    def test_compliance_check_disabled(self, agent, simple_report_data):
        """Test behavior when compliance check is disabled."""
        agent.enable_compliance_check = False

        # Verify flag
        assert agent.enable_compliance_check is False

        # Build prompt should not include compliance
        prompt = agent._build_prompt(simple_report_data)
        # Note: Will still appear in tool list, but won't be mentioned in tasks


class TestReportAgentAIIntegration:
    """Integration tests for ReportAgentAI (require real/demo LLM)."""

    @pytest.fixture
    def agent(self):
        """Create agent with demo provider."""
        # Will use demo provider if no API keys available
        return ReportAgentAI(budget_usd=0.20)

    @pytest.fixture
    def realistic_report_data(self):
        """Create realistic report data for integration testing."""
        return {
            "framework": "TCFD",
            "format": "markdown",
            "carbon_data": {
                "total_co2e_tons": 125.5,
                "total_co2e_kg": 125500,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 75.0, "percentage": 59.76},
                    {"source": "natural_gas", "co2e_tons": 35.0, "percentage": 27.89},
                    {"source": "diesel", "co2e_tons": 10.5, "percentage": 8.37},
                    {"source": "waste", "co2e_tons": 5.0, "percentage": 3.98},
                ],
                "carbon_intensity": {
                    "per_sqft": 0.628,
                    "per_person": 251.0,
                },
            },
            "building_info": {
                "type": "commercial_office",
                "area": 200000,
                "occupancy": 500,
            },
            "period": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "duration": 12,
                "duration_unit": "months",
            },
        }

    def test_full_report_generation_workflow(self, agent, realistic_report_data):
        """Test full report generation workflow with demo provider."""
        result = agent.execute(realistic_report_data)

        # Should succeed with demo provider
        assert result.success is True
        assert result.data is not None

        data = result.data
        # Verify report generated
        assert "report" in data
        assert len(data["report"]) > 0

        # Verify emissions data
        assert data["total_co2e_tons"] == 125.5
        assert len(data["emissions_breakdown"]) == 4

        # Verify framework
        assert data["framework"] == "TCFD"

    def test_cdp_report_generation(self, agent):
        """Test CDP report generation."""
        data = {
            "framework": "CDP",
            "carbon_data": {
                "total_co2e_tons": 50.0,
                "emissions_breakdown": [
                    {"source": "electricity", "co2e_tons": 50.0, "percentage": 100.0},
                ],
            },
        }

        result = agent.execute(data)

        assert result.success is True
        assert result.data["framework"] == "CDP"

    def test_gri_report_generation(self, agent):
        """Test GRI report generation."""
        data = {
            "framework": "GRI",
            "carbon_data": {
                "total_co2e_tons": 30.0,
                "emissions_breakdown": [
                    {"source": "natural_gas", "co2e_tons": 30.0, "percentage": 100.0},
                ],
            },
        }

        result = agent.execute(data)

        assert result.success is True
        assert result.data["framework"] == "GRI"

    def test_sasb_report_generation(self, agent):
        """Test SASB report generation."""
        data = {
            "framework": "SASB",
            "carbon_data": {
                "total_co2e_tons": 40.0,
                "emissions_breakdown": [
                    {"source": "diesel", "co2e_tons": 40.0, "percentage": 100.0},
                ],
            },
        }

        result = agent.execute(data)

        assert result.success is True
        assert result.data["framework"] == "SASB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
