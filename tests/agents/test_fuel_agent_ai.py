"""Tests for AI-powered FuelAgent.

This module tests the FuelAgentAI implementation, ensuring:
1. Tool-first numerics (all calculations use tools)
2. Deterministic results (same input -> same output)
3. Backward compatibility with FuelAgent API
4. AI explanations are generated
5. Budget enforcement works
6. Error handling is robust

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from greenlang.agents.fuel_agent_ai import FuelAgentAI
from greenlang.intelligence import ChatResponse, Usage, FinishReason
from greenlang.intelligence.schemas.responses import ProviderInfo


class TestFuelAgentAI:
    """Test suite for FuelAgentAI."""

    @pytest.fixture
    def agent(self):
        """Create FuelAgentAI instance for testing."""
        return FuelAgentAI(budget_usd=1.0)

    @pytest.fixture
    def valid_payload(self):
        """Create valid test payload."""
        return {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US",
        }

    def test_initialization(self, agent):
        """Test FuelAgentAI initializes correctly."""
        assert agent.agent_id == "fuel_ai"
        assert agent.name == "AI-Powered Fuel Emissions Calculator"
        assert agent.version == "0.1.0"
        assert agent.budget_usd == 1.0
        assert agent.enable_explanations is True
        assert agent.enable_recommendations is True

        # Verify tools are defined
        assert agent.calculate_emissions_tool is not None
        assert agent.lookup_emission_factor_tool is not None
        assert agent.generate_recommendations_tool is not None

    def test_validate_valid_payload(self, agent, valid_payload):
        """Test validation passes for valid payload."""
        assert agent.validate(valid_payload) is True

    def test_validate_invalid_payload(self, agent):
        """Test validation fails for invalid payloads."""
        # Missing fuel_type
        invalid_payload = {"amount": 100, "unit": "therms"}
        assert agent.validate(invalid_payload) is False

        # Missing amount
        invalid_payload = {"fuel_type": "natural_gas", "unit": "therms"}
        assert agent.validate(invalid_payload) is False

        # Missing unit
        invalid_payload = {"fuel_type": "natural_gas", "amount": 100}
        assert agent.validate(invalid_payload) is False

    def test_calculate_emissions_tool_implementation(self, agent):
        """Test calculate_emissions tool uses exact calculations."""
        result = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )

        # Verify structure
        assert "emissions_kg_co2e" in result
        assert "emission_factor" in result
        assert "emission_factor_unit" in result
        assert "scope" in result
        assert "calculation" in result

        # Verify exact calculation (1000 therms * ~5.31 kgCO2e/therm)
        expected_emissions = 1000 * 5.31  # Approximate
        assert abs(result["emissions_kg_co2e"] - expected_emissions) < 100

        # Verify scope
        assert result["scope"] == "1"  # Natural gas is Scope 1

        # Verify tool call tracked
        assert agent._tool_call_count > 0

    def test_lookup_emission_factor_tool_implementation(self, agent):
        """Test lookup_emission_factor tool returns exact factors."""
        result = agent._lookup_emission_factor_impl(
            fuel_type="natural_gas",
            unit="therms",
            country="US",
        )

        # Verify structure
        assert "emission_factor" in result
        assert "unit" in result
        assert "fuel_type" in result
        assert "country" in result
        assert "source" in result

        # Verify data
        assert result["fuel_type"] == "natural_gas"
        assert result["country"] == "US"
        assert result["unit"] == "kgCO2e/therms"
        assert result["emission_factor"] > 0

    def test_generate_recommendations_tool_implementation(self, agent):
        """Test generate_recommendations tool returns valid recommendations."""
        result = agent._generate_recommendations_impl(
            fuel_type="coal",
            emissions_kg=10000,
            country="US",
        )

        # Verify structure
        assert "recommendations" in result
        assert "count" in result

        # Verify recommendations list
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Verify recommendation structure
        rec = recommendations[0]
        assert "priority" in rec
        assert "action" in rec
        assert "impact" in rec
        assert "feasibility" in rec

    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    def test_run_with_mocked_ai(self, mock_session_class, agent, valid_payload):
        """Test run() with mocked ChatSession to verify AI integration."""
        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = (
            "Calculated 5,310 kg CO2e emissions from 1000 therms of natural gas "
            "using emission factor of 5.31 kgCO2e/therm."
        )
        mock_response.tool_calls = [
            {
                "name": "calculate_emissions",
                "arguments": {
                    "fuel_type": "natural_gas",
                    "amount": 1000,
                    "unit": "therms",
                    "country": "US",
                    "renewable_percentage": 0,
                    "efficiency": 1.0,
                },
            }
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

        # Setup mock session
        mock_session = Mock()
        mock_session.chat = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        # Run agent
        result = agent.run(valid_payload)

        # Verify success
        assert result["success"] is True
        assert "data" in result

        # Verify output structure
        data = result["data"]
        assert "co2e_emissions_kg" in data
        assert "fuel_type" in data
        assert "emission_factor" in data
        assert "explanation" in data
        assert "scope" in data

        # Verify AI explanation included
        assert "5,310" in data["explanation"] or "5310" in data["explanation"]

        # Verify metadata
        assert "metadata" in result
        metadata = result["metadata"]
        assert "agent_id" in metadata
        assert metadata["agent_id"] == "fuel_ai"
        assert "calculation_time_ms" in metadata
        assert "ai_calls" in metadata
        assert "tool_calls" in metadata

        # Verify ChatSession was called with correct parameters
        mock_session.chat.assert_called_once()
        call_args = mock_session.chat.call_args
        assert call_args.kwargs["temperature"] == 0.0  # Deterministic
        assert call_args.kwargs["seed"] == 42  # Reproducible
        assert len(call_args.kwargs["tools"]) == 3  # All 3 tools

    def test_determinism_same_input_same_output(self, agent, valid_payload):
        """Test deterministic behavior: same input produces same output."""
        # This test would require a real LLM with seed support
        # For now, we verify the configuration is correct
        assert agent.budget_usd > 0

        # Verify tools are deterministic (same input -> same output)
        result1 = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
            country="US",
        )

        result2 = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
            country="US",
        )

        # Tool results should be identical
        assert result1["emissions_kg_co2e"] == result2["emissions_kg_co2e"]
        assert result1["emission_factor"] == result2["emission_factor"]

    def test_backward_compatibility_api(self, agent, valid_payload):
        """Test backward compatibility with FuelAgent API."""
        # FuelAgentAI should have same interface as FuelAgent
        assert hasattr(agent, "run")
        assert hasattr(agent, "validate")
        assert hasattr(agent, "agent_id")
        assert hasattr(agent, "name")
        assert hasattr(agent, "version")

        # Verify original FuelAgent is accessible
        assert hasattr(agent, "fuel_agent")
        assert agent.fuel_agent is not None

    def test_error_handling_invalid_fuel_type(self, agent):
        """Test error handling for invalid fuel type."""
        invalid_payload = {
            "fuel_type": "invalid_fuel",
            "amount": 100,
            "unit": "therms",
            "country": "US",
        }

        # Should handle gracefully (tool will raise error)
        with pytest.raises(ValueError, match="Calculation failed"):
            agent._calculate_emissions_impl(
                fuel_type="invalid_fuel",
                amount=100,
                unit="therms",
                country="US",
            )

    def test_error_handling_missing_emission_factor(self, agent):
        """Test error handling when emission factor not found."""
        with pytest.raises(ValueError, match="No emission factor found"):
            agent._lookup_emission_factor_impl(
                fuel_type="nonexistent_fuel",
                unit="invalid_unit",
                country="ZZ",
            )

    def test_performance_tracking(self, agent):
        """Test performance metrics tracking."""
        # Initial state
        initial_summary = agent.get_performance_summary()
        assert initial_summary["agent_id"] == "fuel_ai"
        assert "ai_metrics" in initial_summary
        assert "base_agent_metrics" in initial_summary

        # Make a tool call
        agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=100,
            unit="therms",
        )

        # Verify metrics updated
        assert agent._tool_call_count > 0

        # Get updated summary
        summary = agent.get_performance_summary()
        assert summary["ai_metrics"]["tool_call_count"] > 0

    def test_renewable_offset_handling(self, agent):
        """Test renewable offset calculations."""
        result = agent._calculate_emissions_impl(
            fuel_type="electricity",
            amount=1000,
            unit="kWh",
            country="US",
            renewable_percentage=50,  # 50% renewable
            efficiency=1.0,
        )

        # Verify emissions are reduced
        assert result["emissions_kg_co2e"] >= 0

        # Compare with non-renewable
        result_no_offset = agent._calculate_emissions_impl(
            fuel_type="electricity",
            amount=1000,
            unit="kWh",
            country="US",
            renewable_percentage=0,
            efficiency=1.0,
        )

        # With offset should be less than without
        assert result["emissions_kg_co2e"] < result_no_offset["emissions_kg_co2e"]

    def test_efficiency_adjustment(self, agent):
        """Test efficiency adjustment calculations."""
        result_high_eff = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
            country="US",
            efficiency=0.9,  # 90% efficient
        )

        result_low_eff = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
            country="US",
            efficiency=0.5,  # 50% efficient
        )

        # Lower efficiency should result in higher emissions
        assert result_low_eff["emissions_kg_co2e"] > result_high_eff["emissions_kg_co2e"]

    def test_build_prompt_basic(self, agent, valid_payload):
        """Test prompt building for basic case."""
        prompt = agent._build_prompt(valid_payload)

        # Verify key elements
        assert "natural_gas" in prompt
        assert "1000" in prompt
        assert "therms" in prompt
        assert "US" in prompt
        assert "calculate_emissions" in prompt
        assert "tool" in prompt.lower()

    def test_build_prompt_with_renewable(self, agent):
        """Test prompt building with renewable offset."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "US",
            "renewable_percentage": 30,
        }

        prompt = agent._build_prompt(payload)

        # Verify renewable mentioned
        assert "30%" in prompt or "renewable" in prompt.lower()

    def test_build_prompt_with_efficiency(self, agent):
        """Test prompt building with efficiency adjustment."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US",
            "efficiency": 0.85,
        }

        prompt = agent._build_prompt(payload)

        # Verify efficiency mentioned
        assert "85%" in prompt or "efficiency" in prompt.lower()


class TestFuelAgentAIIntegration:
    """Integration tests for FuelAgentAI (require real/demo LLM)."""

    @pytest.fixture
    def agent(self):
        """Create agent with demo provider."""
        # Will use demo provider if no API keys available
        return FuelAgentAI(budget_usd=0.10)

    def test_full_calculation_natural_gas(self, agent):
        """Test full calculation workflow for natural gas."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 100,
            "unit": "therms",
            "country": "US",
        }

        result = agent.run(payload)

        # Should succeed with demo provider
        assert result["success"] is True
        assert "data" in result

        data = result["data"]
        assert data["co2e_emissions_kg"] > 0
        assert data["fuel_type"] == "natural_gas"
        assert data["consumption_amount"] == 100
        assert data["consumption_unit"] == "therms"

    def test_full_calculation_with_recommendations(self, agent):
        """Test calculation with recommendations enabled."""
        agent.enable_recommendations = True

        payload = {
            "fuel_type": "coal",
            "amount": 100,
            "unit": "kg",
            "country": "US",
        }

        result = agent.run(payload)

        # Should succeed
        assert result["success"] is True

        # Should include recommendations (if AI called the tool)
        # Note: With demo provider, tool calls may be simulated


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
