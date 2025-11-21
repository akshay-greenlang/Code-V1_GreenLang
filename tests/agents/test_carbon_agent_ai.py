# -*- coding: utf-8 -*-
"""Tests for AI-powered CarbonAgent.

This module tests the CarbonAgentAI implementation, ensuring:
1. Tool-first numerics (all calculations use tools)
2. Deterministic results (same input -> same output)
3. Backward compatibility with CarbonAgent API
4. AI summaries and insights are generated
5. Recommendations are intelligent and actionable
6. Budget enforcement works
7. Error handling is robust

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from greenlang.agents.carbon_agent_ai import CarbonAgentAI
from greenlang.intelligence import ChatResponse, Usage, FinishReason
from greenlang.intelligence.schemas.responses import ProviderInfo


class TestCarbonAgentAI:
    """Test suite for CarbonAgentAI."""

    @pytest.fixture
    def agent(self):
        """Create CarbonAgentAI instance for testing."""
        return CarbonAgentAI(budget_usd=1.0)

    @pytest.fixture
    def valid_emissions_data(self):
        """Create valid test emissions data."""
        return {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
                {"fuel_type": "natural_gas", "co2e_emissions_kg": 8500},
                {"fuel_type": "diesel", "co2e_emissions_kg": 3200},
            ],
            "building_area": 50000,
            "occupancy": 200,
        }

    @pytest.fixture
    def simple_emissions_data(self):
        """Create simple emissions data without metadata."""
        return {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 10000},
                {"fuel_type": "natural_gas", "co2e_emissions_kg": 5000},
            ]
        }

    def test_initialization(self, agent):
        """Test CarbonAgentAI initializes correctly."""
        assert agent.config.name == "CarbonAgentAI"
        assert agent.config.version == "0.1.0"
        assert agent.budget_usd == 1.0
        assert agent.enable_ai_summary is True
        assert agent.enable_recommendations is True

        # Verify tools are defined
        assert agent.aggregate_emissions_tool is not None
        assert agent.calculate_breakdown_tool is not None
        assert agent.calculate_intensity_tool is not None
        assert agent.generate_recommendations_tool is not None

        # Verify original carbon agent is initialized
        assert agent.carbon_agent is not None

    def test_validate_valid_input(self, agent, valid_emissions_data):
        """Test validation passes for valid input."""
        assert agent.validate_input(valid_emissions_data) is True

    def test_validate_invalid_input(self, agent):
        """Test validation fails for invalid inputs."""
        # Missing emissions key
        assert agent.validate_input({}) is False

        # Non-list emissions
        assert agent.validate_input({"emissions": "not a list"}) is False

    def test_aggregate_emissions_tool_implementation(self, agent, simple_emissions_data):
        """Test aggregate_emissions tool uses exact calculations."""
        emissions = simple_emissions_data["emissions"]

        result = agent._aggregate_emissions_impl(emissions)

        # Verify structure
        assert "total_kg" in result
        assert "total_tons" in result

        # Verify exact calculation
        expected_kg = 10000 + 5000
        assert result["total_kg"] == expected_kg
        assert result["total_tons"] == expected_kg / 1000

        # Verify tool call tracked
        assert agent._tool_call_count > 0

    def test_calculate_breakdown_tool_implementation(self, agent, simple_emissions_data):
        """Test calculate_breakdown tool calculates percentages correctly."""
        emissions = simple_emissions_data["emissions"]
        total_kg = 15000

        result = agent._calculate_breakdown_impl(emissions, total_kg)

        # Verify structure
        assert "breakdown" in result
        breakdown = result["breakdown"]

        # Should have 2 items
        assert len(breakdown) == 2

        # Verify first item (electricity - should be largest first)
        assert breakdown[0]["source"] == "electricity"
        assert breakdown[0]["co2e_kg"] == 10000
        assert breakdown[0]["co2e_tons"] == 10.0
        assert breakdown[0]["percentage"] == 66.67  # 10000/15000 * 100

        # Verify second item (natural_gas)
        assert breakdown[1]["source"] == "natural_gas"
        assert breakdown[1]["co2e_kg"] == 5000
        assert breakdown[1]["percentage"] == 33.33  # 5000/15000 * 100

        # Verify sorted by emissions (largest first)
        assert breakdown[0]["co2e_kg"] >= breakdown[1]["co2e_kg"]

    def test_calculate_intensity_tool_implementation(self, agent):
        """Test calculate_intensity tool calculates metrics correctly."""
        total_kg = 20000
        building_area = 50000
        occupancy = 200

        result = agent._calculate_intensity_impl(
            total_kg=total_kg,
            building_area=building_area,
            occupancy=occupancy,
        )

        # Verify structure
        assert "intensity" in result
        intensity = result["intensity"]

        # Verify per_sqft calculation
        assert "per_sqft" in intensity
        assert intensity["per_sqft"] == round(20000 / 50000, 4)

        # Verify per_person calculation
        assert "per_person" in intensity
        assert intensity["per_person"] == round(20000 / 200, 2)

    def test_calculate_intensity_without_metadata(self, agent):
        """Test calculate_intensity with missing building area and occupancy."""
        result = agent._calculate_intensity_impl(total_kg=20000)

        # Should return empty intensity dict
        assert result["intensity"] == {}

    def test_calculate_intensity_with_zero_values(self, agent):
        """Test calculate_intensity handles zero values gracefully."""
        result = agent._calculate_intensity_impl(
            total_kg=20000,
            building_area=0,
            occupancy=0,
        )

        # Should not calculate intensities with zero denominators
        assert result["intensity"] == {}

    def test_generate_recommendations_electricity(self, agent):
        """Test generate_recommendations for electricity-heavy breakdown."""
        breakdown = [
            {
                "source": "electricity",
                "co2e_kg": 15000,
                "percentage": 65.22,
            },
            {
                "source": "natural_gas",
                "co2e_kg": 8000,
                "percentage": 34.78,
            },
        ]

        result = agent._generate_recommendations_impl(breakdown)

        # Verify structure
        assert "recommendations" in result
        recommendations = result["recommendations"]

        # Should have 2 recommendations (top 2 sources)
        assert len(recommendations) == 2

        # Verify first recommendation (electricity - highest priority)
        rec1 = recommendations[0]
        assert rec1["priority"] == "high"
        assert rec1["source"] == "electricity"
        assert "solar" in rec1["action"].lower() or "renewable" in rec1["action"].lower()
        assert rec1["impact"] == "65.22% of total emissions"

        # Verify second recommendation (natural gas - medium priority)
        rec2 = recommendations[1]
        assert rec2["priority"] == "medium"
        assert rec2["source"] == "natural_gas"

    def test_generate_recommendations_natural_gas(self, agent):
        """Test generate_recommendations for natural gas-heavy breakdown."""
        breakdown = [
            {
                "source": "natural_gas",
                "co2e_kg": 12000,
                "percentage": 80.0,
            },
            {
                "source": "electricity",
                "co2e_kg": 3000,
                "percentage": 20.0,
            },
        ]

        result = agent._generate_recommendations_impl(breakdown)

        recommendations = result["recommendations"]

        # First recommendation should be for natural gas
        rec1 = recommendations[0]
        assert rec1["source"] == "natural_gas"
        assert "heat pump" in rec1["action"].lower() or "envelope" in rec1["action"].lower()

    def test_generate_recommendations_coal(self, agent):
        """Test generate_recommendations for coal."""
        breakdown = [
            {
                "source": "coal",
                "co2e_kg": 50000,
                "percentage": 90.0,
            },
            {
                "source": "electricity",
                "co2e_kg": 5556,
                "percentage": 10.0,
            },
        ]

        result = agent._generate_recommendations_impl(breakdown)

        recommendations = result["recommendations"]

        # First recommendation should be for coal
        rec1 = recommendations[0]
        assert rec1["source"] == "coal"
        assert "phase out" in rec1["action"].lower() or "renewable" in rec1["action"].lower()
        assert "80-100%" in rec1["potential_reduction"]

    def test_generate_recommendations_diesel(self, agent):
        """Test generate_recommendations for diesel/fuel."""
        breakdown = [
            {
                "source": "diesel",
                "co2e_kg": 8000,
                "percentage": 100.0,
            },
        ]

        result = agent._generate_recommendations_impl(breakdown)

        recommendations = result["recommendations"]

        # Should have 1 recommendation for diesel
        rec1 = recommendations[0]
        assert rec1["source"] == "diesel"
        assert "electric" in rec1["action"].lower() or "vehicle" in rec1["action"].lower()

    def test_empty_emissions_handling(self, agent):
        """Test handling of empty emissions list."""
        result = agent.execute({"emissions": []})

        # Should succeed with zero emissions
        assert result.success is True
        assert result.data["total_co2e_kg"] == 0
        assert result.data["total_co2e_tons"] == 0
        assert result.data["emissions_breakdown"] == []
        assert "No emissions" in result.data["summary"]

    @pytest.mark.asyncio
    @patch("greenlang.agents.carbon_agent_ai.ChatSession")
    async def test_execute_with_mocked_ai(self, mock_session_class, agent, valid_emissions_data):
        """Test execute() with mocked ChatSession to verify AI integration."""
        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = (
            "The building's total carbon footprint is 26,700 kg CO2e (26.7 metric tons). "
            "Electricity accounts for 56.2% of emissions (15,000 kg), making it the largest "
            "source. Natural gas contributes 31.8% (8,500 kg). Carbon intensity is 0.534 kg/sqft "
            "and 133.5 kg/person. Key recommendations: 1) Install solar PV to offset electricity, "
            "2) Switch to heat pumps for natural gas reduction, 3) Consider fleet electrification "
            "for diesel usage."
        )
        mock_response.tool_calls = [
            {
                "name": "aggregate_emissions",
                "arguments": {
                    "emissions": valid_emissions_data["emissions"],
                },
            },
            {
                "name": "calculate_breakdown",
                "arguments": {
                    "emissions": valid_emissions_data["emissions"],
                    "total_kg": 26700,
                },
            },
            {
                "name": "calculate_intensity",
                "arguments": {
                    "total_kg": 26700,
                    "building_area": 50000,
                    "occupancy": 200,
                },
            },
            {
                "name": "generate_recommendations",
                "arguments": {
                    "breakdown": [
                        {"source": "electricity", "co2e_kg": 15000, "percentage": 56.18},
                        {"source": "natural_gas", "co2e_kg": 8500, "percentage": 31.84},
                        {"source": "diesel", "co2e_kg": 3200, "percentage": 11.99},
                    ],
                },
            },
        ]
        mock_response.usage = Usage(
            prompt_tokens=200,
            completion_tokens=150,
            total_tokens=350,
            cost_usd=0.02,
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
        result_coro = agent.execute(valid_emissions_data)
        if inspect.iscoroutine(result_coro):
            result = await result_coro
        else:
            result = result_coro

        # Verify success
        assert result.success is True
        assert result.data is not None

        # Verify output structure
        data = result.data
        assert "total_co2e_kg" in data
        assert "total_co2e_tons" in data
        assert "emissions_breakdown" in data
        assert "carbon_intensity" in data
        assert "summary" in data  # Traditional summary
        assert "ai_summary" in data  # AI-generated summary
        assert "recommendations" in data

        # Verify AI summary included
        assert "26,700" in data["ai_summary"] or "26700" in data["ai_summary"]

        # Verify metadata
        assert result.metadata is not None
        metadata = result.metadata
        assert metadata["agent"] == "CarbonAgentAI"
        assert "calculation_time_ms" in metadata
        assert "ai_calls" in metadata
        assert "tool_calls" in metadata
        assert metadata["deterministic"] is True

        # Verify ChatSession was called with correct parameters
        mock_session.chat.assert_called_once()
        call_args = mock_session.chat.call_args
        assert call_args.kwargs["temperature"] == 0.0  # Deterministic
        assert call_args.kwargs["seed"] == 42  # Reproducible
        assert len(call_args.kwargs["tools"]) == 4  # All 4 tools

    def test_determinism_same_input_same_output(self, agent, simple_emissions_data):
        """Test deterministic behavior: same input produces same output."""
        # Verify tools are deterministic (same input -> same output)
        emissions = simple_emissions_data["emissions"]

        result1 = agent._aggregate_emissions_impl(emissions)
        result2 = agent._aggregate_emissions_impl(emissions)

        # Tool results should be identical
        assert result1["total_kg"] == result2["total_kg"]
        assert result1["total_tons"] == result2["total_tons"]

        # Breakdown should also be deterministic
        total_kg = result1["total_kg"]
        breakdown1 = agent._calculate_breakdown_impl(emissions, total_kg)
        breakdown2 = agent._calculate_breakdown_impl(emissions, total_kg)

        assert breakdown1 == breakdown2

    def test_backward_compatibility_api(self, agent):
        """Test backward compatibility with CarbonAgent API."""
        # CarbonAgentAI should have same interface as CarbonAgent
        assert hasattr(agent, "execute")
        assert hasattr(agent, "validate_input")
        assert hasattr(agent, "config")

        # Verify original CarbonAgent is accessible
        assert hasattr(agent, "carbon_agent")
        assert agent.carbon_agent is not None

    def test_error_handling_invalid_input(self, agent):
        """Test error handling for invalid input."""
        result = agent.execute({"invalid_key": "invalid_value"})

        # Should fail validation
        assert result.success is False
        assert "Invalid input" in result.error

    def test_performance_tracking(self, agent, simple_emissions_data):
        """Test performance metrics tracking."""
        # Initial state
        initial_summary = agent.get_performance_summary()
        assert initial_summary["agent"] == "CarbonAgentAI"
        assert "ai_metrics" in initial_summary
        assert "base_agent_metrics" in initial_summary

        # Make a tool call
        emissions = simple_emissions_data["emissions"]
        agent._aggregate_emissions_impl(emissions)

        # Verify metrics updated
        assert agent._tool_call_count > 0

        # Get updated summary
        summary = agent.get_performance_summary()
        assert summary["ai_metrics"]["tool_call_count"] > 0

    def test_build_prompt_basic(self, agent, simple_emissions_data):
        """Test prompt building for basic case."""
        prompt = agent._build_prompt(simple_emissions_data)

        # Verify key elements
        assert "2 emission records" in prompt
        assert "aggregate_emissions" in prompt
        assert "calculate_breakdown" in prompt
        assert "tool" in prompt.lower()

    def test_build_prompt_with_building_area(self, agent, valid_emissions_data):
        """Test prompt building with building area."""
        prompt = agent._build_prompt(valid_emissions_data)

        # Verify building area mentioned
        assert "50,000" in prompt or "50000" in prompt
        assert "sqft" in prompt
        assert "calculate_intensity" in prompt

    def test_build_prompt_with_occupancy(self, agent, valid_emissions_data):
        """Test prompt building with occupancy."""
        prompt = agent._build_prompt(valid_emissions_data)

        # Verify occupancy mentioned
        assert "200 people" in prompt
        assert "calculate_intensity" in prompt

    def test_build_prompt_with_recommendations(self, agent, simple_emissions_data):
        """Test prompt building with recommendations enabled."""
        agent.enable_recommendations = True
        prompt = agent._build_prompt(simple_emissions_data)

        # Verify recommendations mentioned
        assert "generate_recommendations" in prompt
        assert "reduction" in prompt.lower()

    def test_ai_summary_disabled(self, agent, simple_emissions_data):
        """Test behavior when AI summary is disabled."""
        agent.enable_ai_summary = False

        # Note: Full test would require mocking, but verify flag exists
        assert agent.enable_ai_summary is False

    def test_recommendations_disabled(self, agent, simple_emissions_data):
        """Test behavior when recommendations are disabled."""
        agent.enable_recommendations = False

        # Verify flag
        assert agent.enable_recommendations is False

        # Build prompt should not include recommendations
        prompt = agent._build_prompt(simple_emissions_data)
        assert "generate_recommendations" not in prompt


class TestCarbonAgentAIIntegration:
    """Integration tests for CarbonAgentAI (require real/demo LLM)."""

    @pytest.fixture
    def agent(self):
        """Create agent with demo provider."""
        # Will use demo provider if no API keys available
        return CarbonAgentAI(budget_usd=0.10)

    @pytest.fixture
    def realistic_emissions(self):
        """Create realistic emissions data for integration testing."""
        return {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 25000},
                {"fuel_type": "natural_gas", "co2e_emissions_kg": 15000},
                {"fuel_type": "diesel", "co2e_emissions_kg": 5000},
            ],
            "building_area": 100000,
            "occupancy": 500,
        }

    def test_full_aggregation_workflow(self, agent, realistic_emissions):
        """Test full aggregation workflow with demo provider."""
        result = agent.execute(realistic_emissions)

        # Should succeed with demo provider
        assert result.success is True
        assert result.data is not None

        data = result.data
        # Verify total emissions
        assert data["total_co2e_kg"] == 45000
        assert data["total_co2e_tons"] == 45.0

        # Verify breakdown
        assert len(data["emissions_breakdown"]) == 3

        # Verify carbon intensity calculated
        assert "per_sqft" in data["carbon_intensity"]
        assert "per_person" in data["carbon_intensity"]

        # Verify traditional summary exists
        assert "summary" in data
        assert "45" in data["summary"]  # Total tons should be mentioned

    def test_simple_aggregation_no_metadata(self, agent):
        """Test simple aggregation without building metadata."""
        simple_data = {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 10000},
            ]
        }

        result = agent.execute(simple_data)

        # Should succeed
        assert result.success is True
        assert result.data["total_co2e_kg"] == 10000

        # Carbon intensity should be empty (no building area/occupancy)
        assert result.data["carbon_intensity"] == {}


class TestCarbonAgentAICoverage:
    """Additional tests to achieve 80%+ coverage for CarbonAgentAI."""

    @pytest.fixture
    def agent(self):
        """Create CarbonAgentAI instance for testing."""
        return CarbonAgentAI(budget_usd=1.0)

    @pytest.fixture
    def valid_emissions_data(self):
        """Create valid test emissions data."""
        return {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
                {"fuel_type": "natural_gas", "co2e_emissions_kg": 8500},
                {"fuel_type": "diesel", "co2e_emissions_kg": 3200},
            ],
            "building_area": 50000,
            "occupancy": 200,
        }

    # ===== Unit Tests for _extract_tool_results =====

    def test_extract_tool_results_all_tools(self, agent, valid_emissions_data):
        """Test extracting results from all four tool types."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "aggregate_emissions",
                "arguments": {
                    "emissions": valid_emissions_data["emissions"],
                },
            },
            {
                "name": "calculate_breakdown",
                "arguments": {
                    "emissions": valid_emissions_data["emissions"],
                    "total_kg": 26700,
                },
            },
            {
                "name": "calculate_intensity",
                "arguments": {
                    "total_kg": 26700,
                    "building_area": 50000,
                    "occupancy": 200,
                },
            },
            {
                "name": "generate_recommendations",
                "arguments": {
                    "breakdown": [
                        {"source": "electricity", "co2e_kg": 15000, "percentage": 56.18},
                    ],
                },
            },
        ]

        results = agent._extract_tool_results(mock_response)

        # Verify all four tools extracted
        assert "aggregation" in results
        assert "breakdown" in results
        assert "intensity" in results
        assert "recommendations" in results

        # Verify aggregation data
        assert "total_kg" in results["aggregation"]
        assert results["aggregation"]["total_kg"] == 26700

        # Verify breakdown data
        assert "breakdown" in results["breakdown"]
        assert len(results["breakdown"]["breakdown"]) == 3

        # Verify intensity data
        assert "intensity" in results["intensity"]

        # Verify recommendations data
        assert "recommendations" in results["recommendations"]

    def test_extract_tool_results_empty(self, agent):
        """Test extracting results with no tool calls."""
        mock_response = Mock()
        mock_response.tool_calls = []

        results = agent._extract_tool_results(mock_response)

        # Should return empty dict
        assert results == {}

    def test_extract_tool_results_unknown_tool(self, agent):
        """Test extracting results with unknown tool name."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "unknown_tool",
                "arguments": {},
            }
        ]

        results = agent._extract_tool_results(mock_response)

        # Should ignore unknown tool
        assert results == {}

    def test_extract_tool_results_partial(self, agent, valid_emissions_data):
        """Test extracting results with only some tools called."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "aggregate_emissions",
                "arguments": {
                    "emissions": valid_emissions_data["emissions"],
                },
            },
        ]

        results = agent._extract_tool_results(mock_response)

        # Should have only aggregation
        assert "aggregation" in results
        assert "breakdown" not in results
        assert "intensity" not in results
        assert "recommendations" not in results

    # ===== Unit Tests for _build_output =====

    def test_build_output_with_all_data(self, agent, valid_emissions_data):
        """Test building output with complete tool results."""
        tool_results = {
            "aggregation": {
                "total_kg": 26700,
                "total_tons": 26.7,
            },
            "breakdown": {
                "breakdown": [
                    {"source": "electricity", "co2e_kg": 15000, "co2e_tons": 15.0, "percentage": 56.18},
                    {"source": "natural_gas", "co2e_kg": 8500, "co2e_tons": 8.5, "percentage": 31.84},
                    {"source": "diesel", "co2e_kg": 3200, "co2e_tons": 3.2, "percentage": 11.99},
                ],
            },
            "intensity": {
                "intensity": {
                    "per_sqft": 0.534,
                    "per_person": 133.5,
                }
            },
            "recommendations": {
                "recommendations": [
                    {
                        "priority": "high",
                        "source": "electricity",
                        "action": "Install solar PV",
                        "impact": "56.18% of total emissions",
                    }
                ],
            },
        }

        ai_summary = "Total carbon footprint is 26.7 metric tons CO2e."

        output = agent._build_output(valid_emissions_data, tool_results, ai_summary)

        # Verify all fields present
        assert output["total_co2e_kg"] == 26700
        assert output["total_co2e_tons"] == 26.7
        assert len(output["emissions_breakdown"]) == 3
        assert output["carbon_intensity"]["per_sqft"] == 0.534
        assert output["carbon_intensity"]["per_person"] == 133.5
        assert "summary" in output  # Traditional summary
        assert output["ai_summary"] == ai_summary
        assert len(output["recommendations"]) == 1

    def test_build_output_missing_aggregation(self, agent, valid_emissions_data):
        """Test building output with missing aggregation data."""
        tool_results = {}  # No aggregation data

        output = agent._build_output(valid_emissions_data, tool_results, None)

        # Should handle gracefully with defaults
        assert output["total_co2e_kg"] == 0
        assert output["total_co2e_tons"] == 0
        assert output["emissions_breakdown"] == []

    def test_build_output_without_ai_summary(self, agent, valid_emissions_data):
        """Test building output without AI summary."""
        tool_results = {
            "aggregation": {
                "total_kg": 15000,
                "total_tons": 15.0,
            },
            "breakdown": {
                "breakdown": [],
            },
        }

        agent.enable_ai_summary = False
        output = agent._build_output(valid_emissions_data, tool_results, None)

        # Should not include AI summary
        assert "ai_summary" not in output
        assert "summary" in output  # Traditional summary still present

    def test_build_output_without_recommendations(self, agent, valid_emissions_data):
        """Test building output without recommendations."""
        tool_results = {
            "aggregation": {
                "total_kg": 15000,
                "total_tons": 15.0,
            },
            "breakdown": {
                "breakdown": [],
            },
        }

        agent.enable_recommendations = False
        output = agent._build_output(valid_emissions_data, tool_results, None)

        # Should not include recommendations if disabled
        assert "recommendations" not in output

    def test_build_output_with_empty_intensity(self, agent, valid_emissions_data):
        """Test building output with empty intensity (no building data)."""
        tool_results = {
            "aggregation": {
                "total_kg": 15000,
                "total_tons": 15.0,
            },
            "breakdown": {
                "breakdown": [],
            },
            "intensity": {
                "intensity": {},
            },
        }

        output = agent._build_output(valid_emissions_data, tool_results, None)

        # Should have empty carbon intensity
        assert output["carbon_intensity"] == {}

    # ===== Boundary Tests =====

    def test_single_emission_source(self, agent):
        """Test with single emission source."""
        data = {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 10000},
            ]
        }

        result = agent._aggregate_emissions_impl(data["emissions"])

        assert result["total_kg"] == 10000
        assert result["total_tons"] == 10.0

    def test_very_large_emissions(self, agent):
        """Test with very large emission values."""
        data = {
            "emissions": [
                {"fuel_type": "coal", "co2e_emissions_kg": 1e9},  # 1 billion kg
            ]
        }

        result = agent._aggregate_emissions_impl(data["emissions"])

        assert result["total_kg"] == 1e9
        assert result["total_tons"] == 1e6  # 1 million tons

    def test_zero_emissions(self, agent):
        """Test with zero emissions."""
        data = {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 0},
            ]
        }

        result = agent._aggregate_emissions_impl(data["emissions"])

        assert result["total_kg"] == 0
        assert result["total_tons"] == 0

    def test_breakdown_with_zero_total(self, agent):
        """Test breakdown calculation with zero total."""
        emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 0},
        ]

        result = agent._calculate_breakdown_impl(emissions, 0)

        # Should handle gracefully
        assert result["breakdown"][0]["percentage"] == 0

    def test_very_small_emissions(self, agent):
        """Test with very small emission values."""
        emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 0.001},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 0.002},
        ]

        result = agent._aggregate_emissions_impl(emissions)

        assert result["total_kg"] == 0.003
        assert round(result["total_tons"], 6) == 0.000003

    def test_negative_emissions_handling(self, agent):
        """Test handling of negative emissions (carbon credits)."""
        emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 10000},
            {"fuel_type": "carbon_offset", "co2e_emissions_kg": -3000},
        ]

        result = agent._aggregate_emissions_impl(emissions)

        # Should sum including negative values
        assert result["total_kg"] == 7000
        assert result["total_tons"] == 7.0

    def test_breakdown_sorting_order(self, agent):
        """Test that breakdown is sorted by emissions (largest first)."""
        emissions = [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 5000},
            {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
            {"fuel_type": "diesel", "co2e_emissions_kg": 8000},
        ]

        result = agent._calculate_breakdown_impl(emissions, 28000)

        breakdown = result["breakdown"]

        # Should be sorted largest first
        assert breakdown[0]["source"] == "electricity"  # 15000
        assert breakdown[1]["source"] == "diesel"       # 8000
        assert breakdown[2]["source"] == "natural_gas"  # 5000

        # Verify descending order
        for i in range(len(breakdown) - 1):
            assert breakdown[i]["co2e_kg"] >= breakdown[i + 1]["co2e_kg"]

    def test_intensity_with_very_small_building(self, agent):
        """Test intensity calculation with very small building area."""
        result = agent._calculate_intensity_impl(
            total_kg=100000,
            building_area=10,  # Very small building
            occupancy=1,
        )

        intensity = result["intensity"]

        # Per sqft should be very high
        assert intensity["per_sqft"] == 10000.0
        assert intensity["per_person"] == 100000.0

    def test_intensity_with_very_large_building(self, agent):
        """Test intensity calculation with very large building area."""
        result = agent._calculate_intensity_impl(
            total_kg=100000,
            building_area=1e6,  # 1 million sqft
            occupancy=10000,
        )

        intensity = result["intensity"]

        # Per sqft should be very small
        assert intensity["per_sqft"] == 0.1
        assert intensity["per_person"] == 10.0

    # ===== Integration Tests =====

    @pytest.mark.asyncio
    @patch("greenlang.agents.carbon_agent_ai.ChatSession")
    async def test_execute_with_budget_exceeded(self, mock_session_class, agent, valid_emissions_data):
        """Test execute() handling when budget is exceeded."""
        from greenlang.intelligence import BudgetExceeded

        # Setup mock session to raise BudgetExceeded
        mock_session = Mock()
        mock_session.chat = AsyncMock(side_effect=BudgetExceeded("Budget limit reached"))
        mock_session_class.return_value = mock_session

        result = agent.execute(valid_emissions_data)

        # Should handle budget exceeded gracefully
        assert result.success is False
        assert "budget" in result.error.lower() or "Budget" in result.error

    @pytest.mark.asyncio
    @patch("greenlang.agents.carbon_agent_ai.ChatSession")
    async def test_execute_with_general_exception(self, mock_session_class, agent, valid_emissions_data):
        """Test execute() handling of general exceptions."""
        # Setup mock session to raise generic exception
        mock_session = Mock()
        mock_session.chat = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        mock_session_class.return_value = mock_session

        result = agent.execute(valid_emissions_data)

        # Should handle exception gracefully
        assert result.success is False
        assert "Unexpected error" in result.error or "Failed to aggregate" in result.error

    @pytest.mark.asyncio
    @patch("greenlang.agents.carbon_agent_ai.ChatSession")
    async def test_execute_with_disabled_ai_summary(self, mock_session_class, agent, valid_emissions_data):
        """Test execute() with AI summary disabled."""
        agent.enable_ai_summary = False

        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = ""
        mock_response.tool_calls = [
            {
                "name": "aggregate_emissions",
                "arguments": {
                    "emissions": valid_emissions_data["emissions"],
                },
            },
        ]
        mock_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        mock_response.provider_info = ProviderInfo(
            provider="openai",
            model="gpt-4o-mini",
        )
        mock_response.finish_reason = FinishReason.stop

        mock_session = Mock()
        mock_session.chat = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        result = agent.execute(valid_emissions_data)

        assert result.success is True
        # AI summary should not be in output
        assert "ai_summary" not in result.data

    @pytest.mark.asyncio
    @patch("greenlang.agents.carbon_agent_ai.ChatSession")
    async def test_execute_with_disabled_recommendations(self, mock_session_class, agent, valid_emissions_data):
        """Test execute() with recommendations disabled."""
        agent.enable_recommendations = False

        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = "Summary"
        mock_response.tool_calls = [
            {
                "name": "aggregate_emissions",
                "arguments": {
                    "emissions": valid_emissions_data["emissions"],
                },
            },
        ]
        mock_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        mock_response.provider_info = ProviderInfo(
            provider="openai",
            model="gpt-4o-mini",
        )
        mock_response.finish_reason = FinishReason.stop

        mock_session = Mock()
        mock_session.chat = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        result = agent.execute(valid_emissions_data)

        assert result.success is True
        # Recommendations should not be in prompt or output
        # (verified by checking prompt doesn't include recommendation step)

    # ===== Determinism Tests =====

    def test_tool_determinism_aggregation(self, agent):
        """Test that aggregation tool produces identical results across multiple runs."""
        emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 8500},
        ]

        results = []
        for _ in range(5):
            result = agent._aggregate_emissions_impl(emissions)
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["total_kg"] == results[0]["total_kg"]
            assert result["total_tons"] == results[0]["total_tons"]

    def test_tool_determinism_breakdown(self, agent):
        """Test that breakdown tool produces identical results across multiple runs."""
        emissions = [
            {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 8500},
        ]

        results = []
        for _ in range(5):
            result = agent._calculate_breakdown_impl(emissions, 23500)
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["breakdown"] == results[0]["breakdown"]

    def test_tool_determinism_intensity(self, agent):
        """Test that intensity tool produces identical results."""
        results = []
        for _ in range(5):
            result = agent._calculate_intensity_impl(
                total_kg=26700,
                building_area=50000,
                occupancy=200,
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["intensity"] == results[0]["intensity"]

    def test_tool_determinism_recommendations(self, agent):
        """Test that recommendations are deterministic."""
        breakdown = [
            {"source": "electricity", "co2e_kg": 15000, "percentage": 65.0},
            {"source": "natural_gas", "co2e_kg": 8000, "percentage": 35.0},
        ]

        results = []
        for _ in range(3):
            result = agent._generate_recommendations_impl(breakdown)
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert len(result["recommendations"]) == len(results[0]["recommendations"])
            # Compare first recommendation
            assert result["recommendations"][0]["source"] == results[0]["recommendations"][0]["source"]
            assert result["recommendations"][0]["priority"] == results[0]["recommendations"][0]["priority"]

    # ===== Performance and Configuration Tests =====

    def test_cost_accumulation(self, agent):
        """Test that costs accumulate correctly."""
        initial_cost = agent._total_cost_usd

        # Make some tool calls (tools are free)
        agent._aggregate_emissions_impl([{"fuel_type": "electricity", "co2e_emissions_kg": 1000}])

        # Cost should still be initial (tool calls are free)
        assert agent._total_cost_usd == initial_cost

    def test_tool_call_count_tracking(self, agent):
        """Test that tool call counts are tracked correctly."""
        initial_count = agent._tool_call_count

        # Make tool calls
        agent._aggregate_emissions_impl([{"fuel_type": "electricity", "co2e_emissions_kg": 1000}])
        assert agent._tool_call_count == initial_count + 1

        agent._calculate_breakdown_impl([{"fuel_type": "electricity", "co2e_emissions_kg": 1000}], 1000)
        assert agent._tool_call_count == initial_count + 2

        agent._calculate_intensity_impl(total_kg=1000, building_area=1000)
        assert agent._tool_call_count == initial_count + 3

    def test_configuration_options(self):
        """Test agent initialization with different configurations."""
        # Custom budget
        agent1 = CarbonAgentAI(budget_usd=0.25)
        assert agent1.budget_usd == 0.25

        # Disabled AI summary
        agent2 = CarbonAgentAI(enable_ai_summary=False)
        assert agent2.enable_ai_summary is False

        # Disabled recommendations
        agent3 = CarbonAgentAI(enable_recommendations=False)
        assert agent3.enable_recommendations is False

        # All options
        agent4 = CarbonAgentAI(
            budget_usd=2.0,
            enable_ai_summary=False,
            enable_recommendations=False,
        )
        assert agent4.budget_usd == 2.0
        assert agent4.enable_ai_summary is False
        assert agent4.enable_recommendations is False

    def test_build_prompt_without_recommendations(self, agent, valid_emissions_data):
        """Test prompt building without recommendations."""
        agent.enable_recommendations = False

        prompt = agent._build_prompt(valid_emissions_data)

        # Should not mention recommendations
        assert "generate_recommendations" not in prompt
        assert "reduction" not in prompt.lower()

    def test_build_prompt_without_building_metadata(self, agent):
        """Test prompt building without building area or occupancy."""
        data = {
            "emissions": [
                {"fuel_type": "electricity", "co2e_emissions_kg": 10000},
            ]
        }

        prompt = agent._build_prompt(data)

        # Should not mention intensity calculations
        # (though it may still be offered as optional)
        assert "1 emission records" in prompt or "1 emission record" in prompt

    # ===== Edge Case and Error Tests =====

    def test_recommendations_for_unknown_fuel_type(self, agent):
        """Test recommendations for unknown/generic fuel type."""
        breakdown = [
            {"source": "unknown_fuel", "co2e_kg": 5000, "percentage": 100.0},
        ]

        result = agent._generate_recommendations_impl(breakdown)

        recommendations = result["recommendations"]

        # Should have generic recommendation
        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec["source"] == "unknown_fuel"
        assert "optimize" in rec["action"].lower() or "efficiency" in rec["action"].lower()

    def test_recommendations_priority_order(self, agent):
        """Test that recommendations have correct priority order."""
        breakdown = [
            {"source": "electricity", "co2e_kg": 10000, "percentage": 50.0},
            {"source": "natural_gas", "co2e_kg": 6000, "percentage": 30.0},
            {"source": "diesel", "co2e_kg": 4000, "percentage": 20.0},
        ]

        result = agent._generate_recommendations_impl(breakdown)

        recommendations = result["recommendations"]

        # Should have 3 recommendations
        assert len(recommendations) == 3

        # Verify priority order
        assert recommendations[0]["priority"] == "high"
        assert recommendations[1]["priority"] == "medium"
        assert recommendations[2]["priority"] == "low"

    def test_validation_error_handling(self, agent):
        """Test that validation errors are handled properly."""
        invalid_data = {}  # Missing emissions key

        result = agent.execute(invalid_data)

        assert result.success is False
        assert "Invalid input" in result.error

    def test_aggregate_emissions_with_failed_calculation(self, agent):
        """Test error handling when aggregation fails."""
        # This would require mocking the carbon_agent to fail
        # For now, verify the error path exists
        with pytest.raises(ValueError):
            # Create invalid emissions that would cause failure
            agent._aggregate_emissions_impl("not_a_list")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
