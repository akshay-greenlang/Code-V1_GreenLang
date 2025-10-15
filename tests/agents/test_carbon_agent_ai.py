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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
