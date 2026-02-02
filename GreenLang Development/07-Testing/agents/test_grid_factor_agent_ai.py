# -*- coding: utf-8 -*-
"""Tests for AI-powered GridFactorAgent.

This module tests the GridFactorAgentAI implementation, ensuring:
1. Tool-first lookups (all data from database)
2. Deterministic results (same input -> same output)
3. Backward compatibility with GridFactorAgent API
4. AI explanations are generated
5. Budget enforcement works
6. Error handling is robust
7. Temporal interpolation works correctly
8. Weighted averages are calculated accurately
9. Recommendations are generated appropriately

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from greenlang.agents.grid_factor_agent_ai import GridFactorAgentAI
from greenlang.intelligence import ChatResponse, Usage, FinishReason
from greenlang.intelligence.schemas.responses import ProviderInfo


class TestGridFactorAgentAI:
    """Test suite for GridFactorAgentAI."""

    @pytest.fixture
    def agent(self):
        """Create GridFactorAgentAI instance for testing."""
        return GridFactorAgentAI(budget_usd=1.0)

    @pytest.fixture
    def valid_payload(self):
        """Create valid test payload."""
        return {
            "country": "US",
            "fuel_type": "electricity",
            "unit": "kWh",
        }

    def test_initialization(self, agent):
        """Test GridFactorAgentAI initializes correctly."""
        assert agent.agent_id == "grid_factor_ai"
        assert agent.name == "AI-Powered Grid Emission Factor Provider"
        assert agent.version == "0.1.0"
        assert agent.budget_usd == 1.0
        assert agent.enable_explanations is True
        assert agent.enable_recommendations is True

        # Verify tools are defined
        assert agent.lookup_grid_intensity_tool is not None
        assert agent.interpolate_hourly_data_tool is not None
        assert agent.calculate_weighted_average_tool is not None
        assert agent.generate_recommendations_tool is not None

    def test_validate_valid_payload(self, agent, valid_payload):
        """Test validation passes for valid payload."""
        assert agent.validate(valid_payload) is True

    def test_validate_invalid_payload(self, agent):
        """Test validation fails for invalid payloads."""
        # Missing country
        invalid_payload = {"fuel_type": "electricity", "unit": "kWh"}
        assert agent.validate(invalid_payload) is False

        # Missing fuel_type
        invalid_payload = {"country": "US", "unit": "kWh"}
        assert agent.validate(invalid_payload) is False

        # Missing unit
        invalid_payload = {"country": "US", "fuel_type": "electricity"}
        assert agent.validate(invalid_payload) is False

    def test_lookup_grid_intensity_tool_implementation(self, agent):
        """Test lookup_grid_intensity tool returns exact database values."""
        result = agent._lookup_grid_intensity_impl(
            country="US",
            fuel_type="electricity",
            unit="kWh",
            year=2025,
        )

        # Verify structure
        assert "emission_factor" in result
        assert "unit" in result
        assert "country" in result
        assert "fuel_type" in result
        assert "source" in result
        assert "version" in result
        assert "last_updated" in result

        # Verify data
        assert result["country"] == "US"
        assert result["fuel_type"] == "electricity"
        assert result["unit"] == "kgCO2e/kWh"
        assert result["emission_factor"] == 0.385  # US grid factor
        assert "grid_mix" in result

        # Verify tool call tracked
        assert agent._tool_call_count > 0

    def test_lookup_grid_intensity_different_countries(self, agent):
        """Test grid intensity lookup for different countries."""
        # Test US
        result_us = agent._lookup_grid_intensity_impl(
            country="US", fuel_type="electricity", unit="kWh"
        )
        assert result_us["emission_factor"] == 0.385

        # Test India (higher intensity - coal heavy)
        result_in = agent._lookup_grid_intensity_impl(
            country="IN", fuel_type="electricity", unit="kWh"
        )
        assert result_in["emission_factor"] == 0.71
        assert result_in["emission_factor"] > result_us["emission_factor"]

        # Test Brazil (lower intensity - hydro heavy)
        result_br = agent._lookup_grid_intensity_impl(
            country="BR", fuel_type="electricity", unit="kWh"
        )
        assert result_br["emission_factor"] == 0.12
        assert result_br["emission_factor"] < result_us["emission_factor"]

    def test_lookup_grid_intensity_different_units(self, agent):
        """Test grid intensity lookup with different units."""
        # kWh
        result_kwh = agent._lookup_grid_intensity_impl(
            country="US", fuel_type="electricity", unit="kWh"
        )
        assert result_kwh["emission_factor"] == 0.385

        # MWh (should be 1000x kWh)
        result_mwh = agent._lookup_grid_intensity_impl(
            country="US", fuel_type="electricity", unit="MWh"
        )
        assert result_mwh["emission_factor"] == 385.0

    def test_interpolate_hourly_data_tool_implementation(self, agent):
        """Test hourly interpolation for grid intensity."""
        base_intensity = 385.0  # US grid average

        # Morning peak (8 AM)
        result_morning = agent._interpolate_hourly_data_impl(
            base_intensity=base_intensity,
            hour=8,
            renewable_share=0.21,
        )
        assert "interpolated_intensity" in result_morning
        assert "period" in result_morning
        assert "peak_factor" in result_morning
        assert result_morning["period"] == "morning_peak"
        assert result_morning["interpolated_intensity"] > base_intensity

        # Evening peak (18:00)
        result_evening = agent._interpolate_hourly_data_impl(
            base_intensity=base_intensity,
            hour=18,
            renewable_share=0.21,
        )
        assert result_evening["period"] == "evening_peak"
        assert result_evening["interpolated_intensity"] > base_intensity
        assert result_evening["interpolated_intensity"] > result_morning["interpolated_intensity"]

        # Midday (solar generation, 13:00)
        result_midday = agent._interpolate_hourly_data_impl(
            base_intensity=base_intensity,
            hour=13,
            renewable_share=0.21,
        )
        assert result_midday["period"] == "midday"
        assert result_midday["interpolated_intensity"] < base_intensity

        # Off-peak (2 AM)
        result_offpeak = agent._interpolate_hourly_data_impl(
            base_intensity=base_intensity,
            hour=2,
            renewable_share=0.21,
        )
        assert result_offpeak["period"] == "off_peak"
        assert result_offpeak["interpolated_intensity"] < base_intensity

    def test_interpolate_hourly_data_renewable_impact(self, agent):
        """Test that higher renewable share reduces midday intensity."""
        base_intensity = 500.0

        # Low renewable share
        result_low = agent._interpolate_hourly_data_impl(
            base_intensity=base_intensity,
            hour=13,  # Midday
            renewable_share=0.1,
        )

        # High renewable share (more solar)
        result_high = agent._interpolate_hourly_data_impl(
            base_intensity=base_intensity,
            hour=13,  # Midday
            renewable_share=0.5,
        )

        # Higher renewable share should result in lower midday intensity
        assert result_high["interpolated_intensity"] < result_low["interpolated_intensity"]

    def test_calculate_weighted_average_tool_implementation(self, agent):
        """Test weighted average calculation."""
        intensities = [300.0, 400.0, 500.0]
        weights = [0.5, 0.3, 0.2]

        result = agent._calculate_weighted_average_impl(
            intensities=intensities,
            weights=weights,
        )

        # Verify structure
        assert "weighted_average" in result
        assert "min_intensity" in result
        assert "max_intensity" in result
        assert "range" in result

        # Verify calculation: 300*0.5 + 400*0.3 + 500*0.2 = 360
        expected = 300 * 0.5 + 400 * 0.3 + 500 * 0.2
        assert abs(result["weighted_average"] - expected) < 0.01

        # Verify min/max
        assert result["min_intensity"] == 300.0
        assert result["max_intensity"] == 500.0
        assert result["range"] == 200.0

    def test_calculate_weighted_average_normalization(self, agent):
        """Test weighted average normalizes weights."""
        intensities = [100.0, 200.0]
        weights = [2.0, 3.0]  # Sum to 5.0, not 1.0

        result = agent._calculate_weighted_average_impl(
            intensities=intensities,
            weights=weights,
        )

        # Should normalize: 2/5=0.4, 3/5=0.6
        # Result: 100*0.4 + 200*0.6 = 160
        assert abs(result["weighted_average"] - 160.0) < 0.01
        assert sum(result["normalized_weights"]) == pytest.approx(1.0, abs=0.001)

    def test_calculate_weighted_average_error_handling(self, agent):
        """Test weighted average error handling."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            agent._calculate_weighted_average_impl(
                intensities=[100.0, 200.0],
                weights=[0.5],  # Wrong length
            )

    def test_generate_recommendations_tool_implementation(self, agent):
        """Test recommendations generation."""
        result = agent._generate_recommendations_impl(
            country="US",
            current_intensity=385.0,
            renewable_share=0.21,
        )

        # Verify structure
        assert "recommendations" in result
        assert "count" in result
        assert "current_intensity" in result

        # Verify recommendations list
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 3

        # Verify recommendation structure
        rec = recommendations[0]
        assert "priority" in rec
        assert "action" in rec
        assert "impact" in rec
        assert "potential_reduction_gco2_kwh" in rec
        assert "estimated_payback" in rec

    def test_generate_recommendations_high_intensity_grid(self, agent):
        """Test recommendations for high-intensity coal-heavy grid."""
        result = agent._generate_recommendations_impl(
            country="IN",  # Coal-heavy grid
            current_intensity=710.0,
            renewable_share=0.23,
        )

        recommendations = result["recommendations"]

        # Should include critical priority recommendation for coal-heavy grid
        priorities = [r["priority"] for r in recommendations]
        assert "critical" in priorities

        # Should mention renewable energy prominently
        actions = " ".join([r["action"] for r in recommendations]).lower()
        assert "renewable" in actions or "solar" in actions

    def test_generate_recommendations_clean_grid(self, agent):
        """Test recommendations for clean hydro-heavy grid."""
        result = agent._generate_recommendations_impl(
            country="BR",  # Hydro-heavy grid
            current_intensity=120.0,
            renewable_share=0.83,
        )

        recommendations = result["recommendations"]

        # Should still provide recommendations but with different focus
        assert len(recommendations) >= 2

        # Should focus more on efficiency since grid is already clean
        actions = " ".join([r["action"] for r in recommendations]).lower()
        assert "efficiency" in actions or "reduce consumption" in actions

    @pytest.mark.asyncio
    @patch("greenlang.agents.grid_factor_agent_ai.ChatSession")
    async def test_run_with_mocked_ai(self, mock_session_class, agent, valid_payload):
        """Test run() with mocked ChatSession to verify AI integration."""
        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = (
            "The US grid has an average carbon intensity of 385 gCO2/kWh, "
            "which is slightly above the global average. This intensity reflects "
            "the US grid's mix of approximately 21% renewable energy and 79% fossil fuels."
        )
        mock_response.tool_calls = [
            {
                "name": "lookup_grid_intensity",
                "arguments": {
                    "country": "US",
                    "fuel_type": "electricity",
                    "unit": "kWh",
                    "year": 2025,
                },
            }
        ]
        mock_response.usage = Usage(
            prompt_tokens=120,
            completion_tokens=60,
            total_tokens=180,
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
        assert "emission_factor" in data
        assert "unit" in data
        assert "country" in data
        assert "fuel_type" in data
        assert "explanation" in data

        # Verify emission factor is exact
        assert data["emission_factor"] == 0.385

        # Verify AI explanation included
        assert "385" in data["explanation"]

        # Verify metadata
        assert "metadata" in result
        metadata = result["metadata"]
        assert metadata["agent_id"] == "grid_factor_ai"
        assert "lookup_time_ms" in metadata
        assert "ai_calls" in metadata
        assert "tool_calls" in metadata
        assert metadata["deterministic"] is True

        # Verify ChatSession was called with correct parameters
        mock_session.chat.assert_called_once()
        call_args = mock_session.chat.call_args
        assert call_args.kwargs["temperature"] == 0.0  # Deterministic
        assert call_args.kwargs["seed"] == 42  # Reproducible
        assert len(call_args.kwargs["tools"]) == 4  # All 4 tools

    def test_determinism_same_input_same_output(self, agent, valid_payload):
        """Test deterministic behavior: same input produces same output."""
        # Verify tools are deterministic
        result1 = agent._lookup_grid_intensity_impl(
            country="US",
            fuel_type="electricity",
            unit="kWh",
        )

        result2 = agent._lookup_grid_intensity_impl(
            country="US",
            fuel_type="electricity",
            unit="kWh",
        )

        # Tool results should be identical
        assert result1["emission_factor"] == result2["emission_factor"]
        assert result1["country"] == result2["country"]
        assert result1["fuel_type"] == result2["fuel_type"]

    def test_backward_compatibility_api(self, agent):
        """Test backward compatibility with GridFactorAgent API."""
        # GridFactorAgentAI should have same interface as GridFactorAgent
        assert hasattr(agent, "run")
        assert hasattr(agent, "validate")
        assert hasattr(agent, "agent_id")
        assert hasattr(agent, "name")
        assert hasattr(agent, "version")
        assert hasattr(agent, "get_available_countries")
        assert hasattr(agent, "get_available_fuel_types")

        # Verify original GridFactorAgent is accessible
        assert hasattr(agent, "grid_agent")
        assert agent.grid_agent is not None

    def test_get_available_countries(self, agent):
        """Test getting list of available countries."""
        countries = agent.get_available_countries()

        assert isinstance(countries, list)
        assert len(countries) > 0
        assert "US" in countries
        assert "IN" in countries
        assert "EU" in countries
        assert "metadata" not in countries  # Should exclude metadata

    def test_get_available_fuel_types(self, agent):
        """Test getting available fuel types for a country."""
        fuel_types = agent.get_available_fuel_types("US")

        assert isinstance(fuel_types, list)
        assert len(fuel_types) > 0
        assert "electricity" in fuel_types
        assert "natural_gas" in fuel_types
        assert "diesel" in fuel_types

    def test_error_handling_invalid_country(self, agent):
        """Test error handling for invalid country."""
        with pytest.raises(ValueError):
            agent._lookup_grid_intensity_impl(
                country="INVALID_COUNTRY",
                fuel_type="electricity",
                unit="kWh",
            )

    def test_error_handling_invalid_fuel_type(self, agent):
        """Test error handling for invalid fuel type."""
        with pytest.raises(ValueError):
            agent._lookup_grid_intensity_impl(
                country="US",
                fuel_type="invalid_fuel_type",
                unit="kWh",
            )

    def test_error_handling_invalid_unit(self, agent):
        """Test error handling for invalid unit."""
        with pytest.raises(ValueError):
            agent._lookup_grid_intensity_impl(
                country="US",
                fuel_type="electricity",
                unit="invalid_unit",
            )

    def test_performance_tracking(self, agent):
        """Test performance metrics tracking."""
        # Initial state
        initial_summary = agent.get_performance_summary()
        assert initial_summary["agent_id"] == "grid_factor_ai"
        assert "ai_metrics" in initial_summary
        assert "base_agent_metrics" in initial_summary

        # Make a tool call
        agent._lookup_grid_intensity_impl(
            country="US",
            fuel_type="electricity",
            unit="kWh",
        )

        # Verify metrics updated
        assert agent._tool_call_count > 0

        # Get updated summary
        summary = agent.get_performance_summary()
        assert summary["ai_metrics"]["tool_call_count"] > 0

    def test_build_prompt_basic(self, agent, valid_payload):
        """Test prompt building for basic case."""
        prompt = agent._build_prompt(valid_payload)

        # Verify key elements
        assert "US" in prompt
        assert "electricity" in prompt
        assert "kWh" in prompt
        assert "lookup_grid_intensity" in prompt
        assert "tool" in prompt.lower()

    def test_build_prompt_with_recommendations(self, agent):
        """Test prompt building with recommendations enabled."""
        agent.enable_recommendations = True

        payload = {
            "country": "IN",
            "fuel_type": "electricity",
            "unit": "kWh",
        }

        prompt = agent._build_prompt(payload)

        # Verify recommendations mentioned
        assert "generate_recommendations" in prompt or "recommendations" in prompt.lower()


class TestGridFactorAgentAIIntegration:
    """Integration tests for GridFactorAgentAI (require real/demo LLM)."""

    @pytest.fixture
    def agent(self):
        """Create agent with demo provider."""
        # Will use demo provider if no API keys available
        return GridFactorAgentAI(budget_usd=0.10)

    def test_full_lookup_us_grid(self, agent):
        """Test full lookup workflow for US electricity grid."""
        payload = {
            "country": "US",
            "fuel_type": "electricity",
            "unit": "kWh",
        }

        result = agent.run(payload)

        # Should succeed with demo provider
        assert result["success"] is True
        assert "data" in result

        data = result["data"]
        assert data["emission_factor"] == 0.385
        assert data["country"] == "US"
        assert data["fuel_type"] == "electricity"
        assert data["unit"] == "kgCO2e/kWh"

    def test_full_lookup_with_recommendations(self, agent):
        """Test lookup with recommendations enabled."""
        agent.enable_recommendations = True

        payload = {
            "country": "IN",  # High-intensity grid
            "fuel_type": "electricity",
            "unit": "kWh",
        }

        result = agent.run(payload)

        # Should succeed
        assert result["success"] is True

        # Should include recommendations (if AI called the tool)
        # Note: With demo provider, tool calls may be simulated

    def test_full_lookup_natural_gas(self, agent):
        """Test lookup for natural gas grid factor."""
        payload = {
            "country": "US",
            "fuel_type": "natural_gas",
            "unit": "therms",
        }

        result = agent.run(payload)

        assert result["success"] is True
        data = result["data"]
        assert data["emission_factor"] == 5.3
        assert data["fuel_type"] == "natural_gas"


class TestGridFactorAgentAICoverage:
    """Additional tests to achieve 80%+ coverage for GridFactorAgentAI."""

    @pytest.fixture
    def agent(self):
        """Create GridFactorAgentAI instance for testing."""
        return GridFactorAgentAI(budget_usd=1.0)

    @pytest.fixture
    def valid_payload(self):
        """Create valid test payload."""
        return {
            "country": "US",
            "fuel_type": "electricity",
            "unit": "kWh",
        }

    # ===== Unit Tests for _extract_tool_results =====

    def test_extract_tool_results_all_tools(self, agent, valid_payload):
        """Test extracting results from all four tool types."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "lookup_grid_intensity",
                "arguments": {
                    "country": "US",
                    "fuel_type": "electricity",
                    "unit": "kWh",
                    "year": 2025,
                },
            },
            {
                "name": "interpolate_hourly_data",
                "arguments": {
                    "base_intensity": 385.0,
                    "hour": 14,
                    "renewable_share": 0.21,
                },
            },
            {
                "name": "calculate_weighted_average",
                "arguments": {
                    "intensities": [300.0, 400.0, 500.0],
                    "weights": [0.5, 0.3, 0.2],
                },
            },
            {
                "name": "generate_recommendations",
                "arguments": {
                    "country": "US",
                    "current_intensity": 385.0,
                    "renewable_share": 0.21,
                },
            },
        ]

        results = agent._extract_tool_results(mock_response)

        # Verify all four tools extracted
        assert "lookup" in results
        assert "interpolation" in results
        assert "weighted_average" in results
        assert "recommendations" in results

        # Verify lookup data
        assert "emission_factor" in results["lookup"]
        assert results["lookup"]["emission_factor"] == 0.385

        # Verify interpolation data
        assert "interpolated_intensity" in results["interpolation"]

        # Verify weighted average data
        assert "weighted_average" in results["weighted_average"]

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

    def test_extract_tool_results_partial(self, agent):
        """Test extracting results with only some tools called."""
        mock_response = Mock()
        mock_response.tool_calls = [
            {
                "name": "lookup_grid_intensity",
                "arguments": {
                    "country": "US",
                    "fuel_type": "electricity",
                    "unit": "kWh",
                },
            },
        ]

        results = agent._extract_tool_results(mock_response)

        # Should have only lookup
        assert "lookup" in results
        assert "interpolation" not in results
        assert "weighted_average" not in results
        assert "recommendations" not in results

    # ===== Unit Tests for _build_output =====

    def test_build_output_with_all_data(self, agent, valid_payload):
        """Test building output with complete tool results."""
        tool_results = {
            "lookup": {
                "emission_factor": 0.385,
                "unit": "kgCO2e/kWh",
                "country": "US",
                "fuel_type": "electricity",
                "source": "EPA eGRID 2025",
                "version": "2025.1",
                "last_updated": "2025-01-15",
                "grid_mix": {
                    "coal": 0.21,
                    "natural_gas": 0.38,
                    "nuclear": 0.19,
                    "renewables": 0.21,
                    "other": 0.01,
                },
            },
            "interpolation": {
                "interpolated_intensity": 365.75,
                "base_intensity": 385.0,
                "hour": 13,
                "period": "midday",
                "peak_factor": 0.95,
            },
            "weighted_average": {
                "weighted_average": 360.0,
                "min_intensity": 300.0,
                "max_intensity": 500.0,
            },
            "recommendations": {
                "recommendations": [
                    {
                        "priority": "high",
                        "action": "Install solar PV",
                        "impact": "Offset grid emissions",
                    }
                ],
                "count": 1,
            },
        }

        explanation = "US grid has average intensity of 385 gCO2/kWh."

        output = agent._build_output(valid_payload, tool_results, explanation)

        # Verify all fields present
        assert output["emission_factor"] == 0.385
        assert output["unit"] == "kgCO2e/kWh"
        assert output["country"] == "US"
        assert output["fuel_type"] == "electricity"
        assert output["source"] == "EPA eGRID 2025"
        assert "grid_mix" in output
        assert output["explanation"] == explanation
        assert "recommendations" in output
        assert len(output["recommendations"]) == 1

    def test_build_output_missing_lookup(self, agent, valid_payload):
        """Test building output with missing lookup data."""
        tool_results = {}  # No lookup data

        output = agent._build_output(valid_payload, tool_results, None)

        # Should handle gracefully with defaults
        assert output["emission_factor"] == 0.0
        assert output["unit"] == ""

    def test_build_output_without_explanation(self, agent, valid_payload):
        """Test building output without AI explanation."""
        tool_results = {
            "lookup": {
                "emission_factor": 0.385,
                "unit": "kgCO2e/kWh",
                "country": "US",
                "fuel_type": "electricity",
                "source": "EPA eGRID 2025",
                "version": "2025.1",
                "last_updated": "2025-01-15",
                "grid_mix": {},
            }
        }

        agent.enable_explanations = False
        output = agent._build_output(valid_payload, tool_results, None)

        # Should not include explanation
        assert "explanation" not in output

    def test_build_output_without_recommendations(self, agent, valid_payload):
        """Test building output without recommendations."""
        tool_results = {
            "lookup": {
                "emission_factor": 0.385,
                "unit": "kgCO2e/kWh",
                "country": "US",
                "fuel_type": "electricity",
                "source": "EPA eGRID 2025",
                "version": "2025.1",
                "last_updated": "2025-01-15",
                "grid_mix": {},
            }
        }

        agent.enable_recommendations = False
        output = agent._build_output(valid_payload, tool_results, None)

        # Should not include recommendations if disabled
        assert "recommendations" not in output

    # ===== Boundary Tests =====

    def test_interpolate_all_hours(self, agent):
        """Test hourly interpolation for all 24 hours."""
        base_intensity = 400.0
        renewable_share = 0.3

        results = []
        for hour in range(24):
            result = agent._interpolate_hourly_data_impl(
                base_intensity=base_intensity,
                hour=hour,
                renewable_share=renewable_share,
            )
            results.append(result)

        # Verify all hours processed
        assert len(results) == 24

        # Verify proper categorization
        morning_peaks = [r for r in results if r["period"] == "morning_peak"]
        evening_peaks = [r for r in results if r["period"] == "evening_peak"]
        middays = [r for r in results if r["period"] == "midday"]
        off_peaks = [r for r in results if r["period"] == "off_peak"]

        # Verify we have all period types
        assert len(morning_peaks) > 0
        assert len(evening_peaks) > 0
        assert len(middays) > 0
        assert len(off_peaks) > 0

        # Evening peak should have highest intensity
        max_intensity = max(r["interpolated_intensity"] for r in results)
        max_period = [r for r in results if r["interpolated_intensity"] == max_intensity][0]
        assert max_period["period"] in ["evening_peak", "morning_peak"]

    def test_interpolate_zero_renewable_share(self, agent):
        """Test interpolation with zero renewable share."""
        result = agent._interpolate_hourly_data_impl(
            base_intensity=500.0,
            hour=13,  # Midday
            renewable_share=0.0,
        )

        # Should still work
        assert "interpolated_intensity" in result
        assert result["renewable_share"] == 0.0 or "renewable_share" not in result

    def test_interpolate_full_renewable_share(self, agent):
        """Test interpolation with 100% renewable share."""
        result = agent._interpolate_hourly_data_impl(
            base_intensity=500.0,
            hour=13,  # Midday
            renewable_share=1.0,
        )

        # Should reduce midday intensity significantly
        assert result["interpolated_intensity"] < 500.0

    def test_weighted_average_single_value(self, agent):
        """Test weighted average with single value."""
        result = agent._calculate_weighted_average_impl(
            intensities=[400.0],
            weights=[1.0],
        )

        # Should return the single value
        assert result["weighted_average"] == 400.0
        assert result["min_intensity"] == 400.0
        assert result["max_intensity"] == 400.0
        assert result["range"] == 0.0

    def test_weighted_average_equal_weights(self, agent):
        """Test weighted average with equal weights."""
        intensities = [100.0, 200.0, 300.0]
        weights = [1/3, 1/3, 1/3]

        result = agent._calculate_weighted_average_impl(
            intensities=intensities,
            weights=weights,
        )

        # Should be simple average: (100+200+300)/3 = 200
        assert abs(result["weighted_average"] - 200.0) < 0.01

    def test_weighted_average_extreme_weights(self, agent):
        """Test weighted average with one dominant weight."""
        intensities = [100.0, 200.0, 900.0]
        weights = [0.01, 0.01, 0.98]

        result = agent._calculate_weighted_average_impl(
            intensities=intensities,
            weights=weights,
        )

        # Should be close to 900: 100*0.01 + 200*0.01 + 900*0.98 = 885
        expected = 100*0.01 + 200*0.01 + 900*0.98
        assert abs(result["weighted_average"] - expected) < 0.01

    def test_recommendations_for_very_high_intensity(self, agent):
        """Test recommendations for extremely high grid intensity."""
        result = agent._generate_recommendations_impl(
            country="CN",  # Example high-intensity grid
            current_intensity=900.0,  # Very high
            renewable_share=0.1,
        )

        recommendations = result["recommendations"]

        # Should have critical priority recommendations
        priorities = [r["priority"] for r in recommendations]
        assert "critical" in priorities or "high" in priorities

        # Should emphasize renewable energy
        actions = " ".join([r["action"] for r in recommendations]).lower()
        assert "renewable" in actions or "solar" in actions or "clean" in actions

    def test_recommendations_for_very_low_intensity(self, agent):
        """Test recommendations for very clean grid."""
        result = agent._generate_recommendations_impl(
            country="NO",  # Example clean grid (hydro)
            current_intensity=50.0,  # Very low
            renewable_share=0.98,
        )

        recommendations = result["recommendations"]

        # Should still provide recommendations
        assert len(recommendations) >= 1

        # Should focus on efficiency/consumption reduction
        actions = " ".join([r["action"] for r in recommendations]).lower()
        assert "efficiency" in actions or "reduce" in actions or "consumption" in actions

    # ===== Integration Tests =====

    @pytest.mark.asyncio
    @patch("greenlang.agents.grid_factor_agent_ai.ChatSession")
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
        assert "budget" in result["error"]["message"].lower() or "Budget" in result["error"]["message"]

    @pytest.mark.asyncio
    @patch("greenlang.agents.grid_factor_agent_ai.ChatSession")
    async def test_run_with_general_exception(self, mock_session_class, agent, valid_payload):
        """Test run() handling of general exceptions."""
        # Setup mock session to raise generic exception
        mock_session = Mock()
        mock_session.chat = AsyncMock(side_effect=RuntimeError("Unexpected error"))
        mock_session_class.return_value = mock_session

        result = agent.run(valid_payload)

        # Should handle exception gracefully
        assert result["success"] is False
        assert "Unexpected error" in result["error"]["message"] or "Failed to lookup" in result["error"]["message"]

    @pytest.mark.asyncio
    @patch("greenlang.agents.grid_factor_agent_ai.ChatSession")
    async def test_run_with_disabled_explanations(self, mock_session_class, agent, valid_payload):
        """Test run() with explanations disabled."""
        agent.enable_explanations = False

        # Create mock response
        mock_response = Mock(spec=ChatResponse)
        mock_response.text = ""
        mock_response.tool_calls = [
            {
                "name": "lookup_grid_intensity",
                "arguments": {
                    "country": "US",
                    "fuel_type": "electricity",
                    "unit": "kWh",
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

        result = agent.run(valid_payload)

        assert result["success"] is True
        # Explanation should not be in output
        assert "explanation" not in result["data"]

    # ===== Determinism Tests =====

    def test_tool_determinism_lookup(self, agent):
        """Test that lookup tool produces identical results across multiple runs."""
        results = []
        for _ in range(5):
            result = agent._lookup_grid_intensity_impl(
                country="US",
                fuel_type="electricity",
                unit="kWh",
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["emission_factor"] == results[0]["emission_factor"]
            assert result["unit"] == results[0]["unit"]
            assert result["country"] == results[0]["country"]

    def test_tool_determinism_interpolation(self, agent):
        """Test that interpolation tool produces identical results."""
        results = []
        for _ in range(5):
            result = agent._interpolate_hourly_data_impl(
                base_intensity=385.0,
                hour=14,
                renewable_share=0.21,
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["interpolated_intensity"] == results[0]["interpolated_intensity"]
            assert result["period"] == results[0]["period"]

    def test_tool_determinism_weighted_average(self, agent):
        """Test that weighted average tool produces identical results."""
        results = []
        for _ in range(5):
            result = agent._calculate_weighted_average_impl(
                intensities=[300.0, 400.0, 500.0],
                weights=[0.5, 0.3, 0.2],
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert result["weighted_average"] == results[0]["weighted_average"]

    def test_tool_determinism_recommendations(self, agent):
        """Test that recommendations are deterministic."""
        results = []
        for _ in range(3):
            result = agent._generate_recommendations_impl(
                country="US",
                current_intensity=385.0,
                renewable_share=0.21,
            )
            results.append(result)

        # All results should be identical
        for result in results[1:]:
            assert len(result["recommendations"]) == len(results[0]["recommendations"])
            # Compare first recommendation
            if len(result["recommendations"]) > 0:
                assert result["recommendations"][0]["priority"] == results[0]["recommendations"][0]["priority"]

    # ===== Performance and Configuration Tests =====

    def test_cost_accumulation(self, agent):
        """Test that costs accumulate correctly."""
        initial_cost = agent._total_cost_usd

        # Make some tool calls (tools are free)
        agent._lookup_grid_intensity_impl(
            country="US",
            fuel_type="electricity",
            unit="kWh",
        )

        # Cost should still be initial (tool calls are free)
        assert agent._total_cost_usd == initial_cost

    def test_tool_call_count_tracking(self, agent):
        """Test that tool call counts are tracked correctly."""
        initial_count = agent._tool_call_count

        # Make tool calls
        agent._lookup_grid_intensity_impl(
            country="US",
            fuel_type="electricity",
            unit="kWh",
        )
        assert agent._tool_call_count == initial_count + 1

        agent._interpolate_hourly_data_impl(
            base_intensity=385.0,
            hour=14,
        )
        assert agent._tool_call_count == initial_count + 2

        agent._calculate_weighted_average_impl(
            intensities=[300.0, 400.0],
            weights=[0.5, 0.5],
        )
        assert agent._tool_call_count == initial_count + 3

    def test_configuration_options(self):
        """Test agent initialization with different configurations."""
        # Custom budget
        agent1 = GridFactorAgentAI(budget_usd=0.25)
        assert agent1.budget_usd == 0.25

        # Disabled explanations
        agent2 = GridFactorAgentAI(enable_explanations=False)
        assert agent2.enable_explanations is False

        # Disabled recommendations
        agent3 = GridFactorAgentAI(enable_recommendations=False)
        assert agent3.enable_recommendations is False

        # All options
        agent4 = GridFactorAgentAI(
            budget_usd=2.0,
            enable_explanations=False,
            enable_recommendations=False,
        )
        assert agent4.budget_usd == 2.0
        assert agent4.enable_explanations is False
        assert agent4.enable_recommendations is False

    def test_build_prompt_without_recommendations(self, agent, valid_payload):
        """Test prompt building without recommendations."""
        agent.enable_recommendations = False

        prompt = agent._build_prompt(valid_payload)

        # Should not mention recommendations
        assert "generate_recommendations" not in prompt or "recommendations" not in prompt.lower()

    # ===== Edge Case Tests =====

    def test_validation_error_handling(self, agent):
        """Test that validation errors are handled properly."""
        invalid_payload = {}  # Missing required fields

        result = agent.run(invalid_payload)

        assert result["success"] is False
        assert "error" in result

    def test_lookup_with_future_year(self, agent):
        """Test lookup with future year."""
        result = agent._lookup_grid_intensity_impl(
            country="US",
            fuel_type="electricity",
            unit="kWh",
            year=2030,
        )

        # Should work (may use projected or default values)
        assert "emission_factor" in result

    def test_interpolate_boundary_hours(self, agent):
        """Test interpolation at hour boundaries."""
        # Hour 0 (midnight)
        result_0 = agent._interpolate_hourly_data_impl(
            base_intensity=400.0,
            hour=0,
        )
        assert result_0["hour"] == 0
        assert result_0["period"] == "off_peak"

        # Hour 23 (11 PM)
        result_23 = agent._interpolate_hourly_data_impl(
            base_intensity=400.0,
            hour=23,
        )
        assert result_23["hour"] == 23
        assert result_23["period"] == "off_peak"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
