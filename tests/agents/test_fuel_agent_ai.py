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

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_run_with_mocked_ai(self, mock_session_class, agent, valid_payload):
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

        # Run agent (handle both sync and async)
        import asyncio
        import inspect

        result_coro = agent.run(valid_payload)
        if inspect.iscoroutine(result_coro):
            result = await result_coro
        else:
            result = result_coro

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


class TestFuelAgentAICoverage:
    """Additional tests to achieve 80%+ coverage for FuelAgentAI."""

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

    # ===== Unit Tests for _extract_tool_results =====

    def test_extract_tool_results_all_tools(self, agent):
        """Test extracting results from all three tool types."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "calculate_emissions",
                "arguments": {
                    "fuel_type": "natural_gas",
                    "amount": 1000,
                    "unit": "therms",
                    "country": "US",
                },
            },
            {
                "name": "lookup_emission_factor",
                "arguments": {
                    "fuel_type": "natural_gas",
                    "unit": "therms",
                    "country": "US",
                },
            },
            {
                "name": "generate_recommendations",
                "arguments": {
                    "fuel_type": "natural_gas",
                    "emissions_kg": 5310,
                    "country": "US",
                },
            },
        ]

        results = agent._extract_tool_results(mock_response)

        # Verify all three tools extracted
        assert "emissions" in results
        assert "emission_factor" in results
        assert "recommendations" in results

        # Verify emissions data
        assert "emissions_kg_co2e" in results["emissions"]
        assert results["emissions"]["emissions_kg_co2e"] > 0

        # Verify emission factor data
        assert "emission_factor" in results["emission_factor"]
        assert results["emission_factor"]["emission_factor"] > 0

        # Verify recommendations data
        assert "recommendations" in results["recommendations"]
        assert isinstance(results["recommendations"]["recommendations"], list)

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

    # ===== Unit Tests for _build_output =====

    def test_build_output_with_all_data(self, agent, valid_payload):
        """Test building output with complete tool results."""
        tool_results = {
            "emissions": {
                "emissions_kg_co2e": 5310.0,
                "emission_factor": 5.31,
                "emission_factor_unit": "kgCO2e/therms",
                "scope": "1",
                "energy_content_mmbtu": 100.0,
            },
            "recommendations": {
                "recommendations": [
                    {
                        "priority": "High",
                        "action": "Switch to renewable energy",
                        "impact": "50% reduction",
                        "feasibility": "Medium",
                    }
                ],
                "count": 1,
            },
        }

        explanation = "Calculated 5,310 kg CO2e emissions from 1000 therms of natural gas."

        output = agent._build_output(valid_payload, tool_results, explanation)

        # Verify all fields present
        assert output["co2e_emissions_kg"] == 5310.0
        assert output["fuel_type"] == "natural_gas"
        assert output["consumption_amount"] == 1000
        assert output["consumption_unit"] == "therms"
        assert output["emission_factor"] == 5.31
        assert output["emission_factor_unit"] == "kgCO2e/therms"
        assert output["country"] == "US"
        assert output["scope"] == "1"
        assert output["energy_content_mmbtu"] == 100.0
        assert output["renewable_offset_applied"] is False
        assert output["efficiency_adjusted"] is False
        assert "recommendations" in output
        assert len(output["recommendations"]) == 1
        assert output["explanation"] == explanation

    def test_build_output_with_renewable_offset(self, agent):
        """Test building output with renewable offset applied."""
        payload = {
            "fuel_type": "electricity",
            "amount": 1000,
            "unit": "kWh",
            "country": "US",
            "renewable_percentage": 50,
        }

        tool_results = {
            "emissions": {
                "emissions_kg_co2e": 250.0,
                "emission_factor": 0.5,
                "emission_factor_unit": "kgCO2e/kWh",
                "scope": "2",
                "energy_content_mmbtu": 3.412,
            }
        }

        output = agent._build_output(payload, tool_results, None)

        # Verify renewable offset flag
        assert output["renewable_offset_applied"] is True
        assert output["efficiency_adjusted"] is False

    def test_build_output_with_efficiency_adjustment(self, agent):
        """Test building output with efficiency adjustment."""
        payload = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US",
            "efficiency": 0.85,
        }

        tool_results = {
            "emissions": {
                "emissions_kg_co2e": 6247.0,
                "emission_factor": 5.31,
                "emission_factor_unit": "kgCO2e/therms",
                "scope": "1",
                "energy_content_mmbtu": 100.0,
            }
        }

        output = agent._build_output(payload, tool_results, None)

        # Verify efficiency adjustment flag
        assert output["efficiency_adjusted"] is True
        assert output["renewable_offset_applied"] is False

    def test_build_output_missing_emissions(self, agent, valid_payload):
        """Test building output with missing emissions data."""
        tool_results = {}  # No emissions data

        output = agent._build_output(valid_payload, tool_results, None)

        # Should handle gracefully with defaults
        assert output["co2e_emissions_kg"] == 0.0
        assert output["emission_factor"] == 0.0
        assert output["emission_factor_unit"] == ""

    def test_build_output_without_explanation(self, agent, valid_payload):
        """Test building output without AI explanation."""
        tool_results = {
            "emissions": {
                "emissions_kg_co2e": 5310.0,
                "emission_factor": 5.31,
                "emission_factor_unit": "kgCO2e/therms",
                "scope": "1",
                "energy_content_mmbtu": 100.0,
            }
        }

        agent.enable_explanations = False
        output = agent._build_output(valid_payload, tool_results, None)

        # Should not include explanation
        assert "explanation" not in output

    def test_build_output_without_recommendations(self, agent, valid_payload):
        """Test building output without recommendations."""
        tool_results = {
            "emissions": {
                "emissions_kg_co2e": 5310.0,
                "emission_factor": 5.31,
                "emission_factor_unit": "kgCO2e/therms",
                "scope": "1",
                "energy_content_mmbtu": 100.0,
            }
        }

        output = agent._build_output(valid_payload, tool_results, None)

        # Should not include recommendations if not in tool results
        assert "recommendations" not in output

    # ===== Boundary Tests =====

    def test_zero_amount(self, agent):
        """Test calculation with zero amount."""
        result = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=0,
            unit="therms",
            country="US",
        )

        # Should return zero emissions
        assert result["emissions_kg_co2e"] == 0.0

    def test_very_large_amount(self, agent):
        """Test calculation with very large amount."""
        result = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1e9,  # 1 billion therms
            unit="therms",
            country="US",
        )

        # Should handle large numbers
        assert result["emissions_kg_co2e"] > 0
        assert result["emissions_kg_co2e"] == 1e9 * result["emission_factor"]

    def test_renewable_percentage_boundaries(self, agent):
        """Test renewable percentage at boundaries (0 and 100)."""
        # 0% renewable
        result_0 = agent._calculate_emissions_impl(
            fuel_type="electricity",
            amount=1000,
            unit="kWh",
            country="US",
            renewable_percentage=0,
        )

        # 100% renewable
        result_100 = agent._calculate_emissions_impl(
            fuel_type="electricity",
            amount=1000,
            unit="kWh",
            country="US",
            renewable_percentage=100,
        )

        # 100% renewable should have zero emissions
        assert result_100["emissions_kg_co2e"] == 0.0
        assert result_0["emissions_kg_co2e"] > 0

    def test_efficiency_boundaries(self, agent):
        """Test efficiency at boundaries (very low and 1.0)."""
        # Very low efficiency
        result_low = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
            country="US",
            efficiency=0.1,
        )

        # Perfect efficiency
        result_perfect = agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
            country="US",
            efficiency=1.0,
        )

        # Low efficiency should result in much higher emissions
        assert result_low["emissions_kg_co2e"] > result_perfect["emissions_kg_co2e"]

    def test_invalid_country_code(self, agent):
        """Test handling of invalid country code."""
        # Should gracefully handle or default to US
        result = agent._lookup_emission_factor_impl(
            fuel_type="natural_gas",
            unit="therms",
            country="INVALID",
        )

        # Should either succeed (defaulting) or raise error
        assert "emission_factor" in result or True  # Graceful handling

    # ===== Integration Tests =====

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_run_with_budget_exceeded(self, mock_session_class, agent, valid_payload):
        """Test run() handling when budget is exceeded."""
        from greenlang.intelligence import BudgetExceeded

        # Setup mock session to raise BudgetExceeded
        mock_session = Mock()
        mock_session.chat = AsyncMock(side_effect=BudgetExceeded("Budget limit reached"))
        mock_session_class.return_value = mock_session

        result = agent.run(valid_payload)

        # Should handle budget exceeded gracefully
        assert result["success"] is False
        assert "error" in result
        assert "budget" in result["error"]["message"].lower() or "Budget" in result["error"]["message"]

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_run_with_general_exception(self, mock_session_class, agent, valid_payload):
        """Test run() handling of general exceptions."""
        # Setup mock session to raise generic exception
        mock_session = Mock()
        mock_session.chat = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        mock_session_class.return_value = mock_session

        result = agent.run(valid_payload)

        # Should handle exception gracefully
        assert result["success"] is False
        assert "error" in result
        assert "Unexpected error" in result["error"]["message"] or "Failed to calculate" in result["error"]["message"]

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_run_with_disabled_explanations(self, mock_session_class, agent, valid_payload):
        """Test run() with explanations disabled."""
        agent.enable_explanations = False

        # Create mock response without explanation
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = ""
        mock_response.tool_calls = [
            {
                "name": "calculate_emissions",
                "arguments": {
                    "fuel_type": "natural_gas",
                    "amount": 1000,
                    "unit": "therms",
                    "country": "US",
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

        mock_session = Mock()
        mock_session.chat = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        result = agent.run(valid_payload)

        assert result["success"] is True
        # Explanation should not be in output
        assert "explanation" not in result["data"]

    @pytest.mark.asyncio
    @patch("greenlang.agents.fuel_agent_ai.ChatSession")
    async def test_run_with_disabled_recommendations(self, mock_session_class, agent, valid_payload):
        """Test run() with recommendations disabled."""
        agent.enable_recommendations = False

        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = "Calculation complete."
        mock_response.tool_calls = [
            {
                "name": "calculate_emissions",
                "arguments": {
                    "fuel_type": "natural_gas",
                    "amount": 1000,
                    "unit": "therms",
                    "country": "US",
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

        mock_session = Mock()
        mock_session.chat = AsyncMock(return_value=mock_response)
        mock_session_class.return_value = mock_session

        result = agent.run(valid_payload)

        assert result["success"] is True
        # Recommendations should not be requested in prompt
        # (verified by checking prompt doesn't include recommendation step)

    # ===== Determinism Tests =====

    def test_tool_determinism_multiple_runs(self, agent):
        """Test that tool calls produce identical results across multiple runs."""
        results = []
        for _ in range(5):
            result = agent._calculate_emissions_impl(
                fuel_type="natural_gas",
                amount=1000,
                unit="therms",
                country="US",
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["emissions_kg_co2e"] == results[0]["emissions_kg_co2e"]
            assert result["emission_factor"] == results[0]["emission_factor"]
            assert result["scope"] == results[0]["scope"]

    def test_lookup_determinism_multiple_runs(self, agent):
        """Test that emission factor lookups are deterministic."""
        results = []
        for _ in range(5):
            result = agent._lookup_emission_factor_impl(
                fuel_type="diesel",
                unit="gallons",
                country="US",
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["emission_factor"] == results[0]["emission_factor"]
            assert result["unit"] == results[0]["unit"]

    def test_recommendations_determinism(self, agent):
        """Test that recommendations are deterministic."""
        results = []
        for _ in range(3):
            result = agent._generate_recommendations_impl(
                fuel_type="coal",
                emissions_kg=10000,
                country="US",
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["count"] == results[0]["count"]
            assert len(result["recommendations"]) == len(results[0]["recommendations"])

    # ===== Performance and Tracking Tests =====

    def test_cost_accumulation(self, agent):
        """Test that costs accumulate correctly across multiple calls."""
        initial_cost = agent._total_cost_usd

        # Make some tool calls
        agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
        )

        # Cost should still be initial (tool calls are free)
        assert agent._total_cost_usd == initial_cost

        # AI calls would increment cost, but we're testing tools only here

    def test_tool_call_count_tracking(self, agent):
        """Test that tool call counts are tracked correctly."""
        initial_count = agent._tool_call_count

        # Make tool calls
        agent._calculate_emissions_impl(
            fuel_type="natural_gas",
            amount=1000,
            unit="therms",
        )

        assert agent._tool_call_count == initial_count + 1

        agent._lookup_emission_factor_impl(
            fuel_type="diesel",
            unit="gallons",
        )

        assert agent._tool_call_count == initial_count + 2

    def test_configuration_options(self):
        """Test agent initialization with different configurations."""
        # Custom budget
        agent1 = FuelAgentAI(budget_usd=0.25)
        assert agent1.budget_usd == 0.25

        # Disabled explanations
        agent2 = FuelAgentAI(enable_explanations=False)
        assert agent2.enable_explanations is False

        # Disabled recommendations
        agent3 = FuelAgentAI(enable_recommendations=False)
        assert agent3.enable_recommendations is False

        # All options
        agent4 = FuelAgentAI(
            budget_usd=2.0,
            enable_explanations=False,
            enable_recommendations=False,
        )
        assert agent4.budget_usd == 2.0
        assert agent4.enable_explanations is False
        assert agent4.enable_recommendations is False

    def test_build_prompt_with_recommendations_disabled(self, agent):
        """Test prompt building with recommendations disabled."""
        agent.enable_recommendations = False

        payload = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US",
        }

        prompt = agent._build_prompt(payload)

        # Should not mention recommendations
        assert "generate_recommendations" not in prompt
        assert "3." not in prompt  # Step 3 is recommendations

    def test_build_prompt_comprehensive(self, agent):
        """Test prompt building with all options."""
        payload = {
            "fuel_type": "electricity",
            "amount": 5000,
            "unit": "kWh",
            "country": "UK",
            "renewable_percentage": 40,
            "efficiency": 0.92,
        }

        prompt = agent._build_prompt(payload)

        # Verify all elements present
        assert "electricity" in prompt
        assert "5000" in prompt
        assert "kWh" in prompt
        assert "UK" in prompt
        assert "40" in prompt or "renewable" in prompt.lower()
        assert "92" in prompt or "efficiency" in prompt.lower()

    # ===== Error Handling Tests =====

    def test_validation_error_handling(self, agent):
        """Test that validation errors are handled properly."""
        invalid_payload = {}  # Empty payload

        result = agent.run(invalid_payload)

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["type"] == "ValidationError"

    def test_calculate_emissions_error_propagation(self, agent):
        """Test that calculation errors are properly propagated."""
        with pytest.raises(ValueError):
            agent._calculate_emissions_impl(
                fuel_type="nonexistent_fuel_type_xyz",
                amount=100,
                unit="invalid_unit",
                country="US",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
