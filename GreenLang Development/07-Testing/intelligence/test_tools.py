# -*- coding: utf-8 -*-
"""
Tests for ToolRegistry and @tool decorator

Tests auto-discovery, invocation, validation, and agent integration.
Validates that @tool-decorated methods are properly registered and callable by LLMs.
"""

import pytest
from typing import Dict, Any
from greenlang.intelligence.runtime.agent_tools import (
    tool, ToolRegistry, ToolSpec, ToolInvocationError
)
from greenlang.agents.carbon_agent import CarbonAgent
from greenlang.agents.grid_factor_agent import GridFactorAgent
from greenlang.agents.energy_balance_agent import EnergyBalanceAgent


class TestToolDecorator:
    """Test @tool decorator functionality"""

    def test_tool_decorator_adds_spec_attribute(self):
        """@tool should add _tool_spec attribute to decorated function"""
        @tool(
            name="test_tool",
            description="Test tool",
            parameters_schema={"type": "object", "properties": {}},
            timeout_s=5.0
        )
        def sample_tool():
            return {"result": "success"}

        assert hasattr(sample_tool, "_tool_spec")
        spec = sample_tool._tool_spec
        assert isinstance(spec, ToolSpec)
        assert spec.name == "test_tool"
        assert spec.description == "Test tool"
        assert spec.timeout_s == 5.0

    def test_tool_decorator_preserves_function(self):
        """@tool should preserve original function behavior"""
        @tool(
            name="add",
            description="Add two numbers",
            parameters_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }
        )
        def add(a: float, b: float) -> float:
            return a + b

        # Function should still be callable directly
        result = add(2, 3)
        assert result == 5


class TestToolRegistry:
    """Test ToolRegistry auto-discovery and invocation"""

    def test_register_from_carbon_agent(self):
        """ToolRegistry should auto-discover @tool methods from CarbonAgent"""
        agent = CarbonAgent()
        registry = ToolRegistry()

        count = registry.register_from_agent(agent)

        assert count == 1  # CarbonAgent has 1 @tool method
        assert "calculate_carbon_footprint" in registry.get_tool_names()

    def test_register_from_grid_factor_agent(self):
        """ToolRegistry should auto-discover @tool methods from GridFactorAgent"""
        agent = GridFactorAgent()
        registry = ToolRegistry()

        count = registry.register_from_agent(agent)

        assert count == 1  # GridFactorAgent has 1 @tool method
        assert "get_emission_factor" in registry.get_tool_names()

    def test_register_from_energy_balance_agent(self):
        """ToolRegistry should auto-discover @tool methods from EnergyBalanceAgent"""
        agent = EnergyBalanceAgent()
        registry = ToolRegistry()

        count = registry.register_from_agent(agent)

        assert count == 1  # EnergyBalanceAgent has 1 @tool method
        assert "simulate_solar_energy_balance" in registry.get_tool_names()

    def test_register_multiple_agents(self):
        """ToolRegistry should register tools from multiple agents"""
        carbon_agent = CarbonAgent()
        grid_agent = GridFactorAgent()

        registry = ToolRegistry()
        registry.register_from_agent(carbon_agent)
        registry.register_from_agent(grid_agent)

        tool_names = registry.get_tool_names()
        assert "calculate_carbon_footprint" in tool_names
        assert "get_emission_factor" in tool_names
        assert len(tool_names) == 2

    def test_get_tool_defs_returns_valid_schemas(self):
        """get_tool_defs() should return valid OpenAI/Anthropic function schemas"""
        agent = CarbonAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        tool_defs = registry.get_tool_defs()

        assert len(tool_defs) == 1
        tool_def = tool_defs[0]

        # Check required fields for function calling
        assert tool_def.name == "calculate_carbon_footprint"
        assert "description" in tool_def.description
        assert tool_def.parameters_schema["type"] == "object"
        assert "properties" in tool_def.parameters_schema


class TestToolInvocation:
    """Test tool invocation with argument validation"""

    def test_invoke_carbon_agent_tool_valid_args(self):
        """Invoke calculate_carbon_footprint with valid arguments"""
        agent = CarbonAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        result = registry.invoke(
            "calculate_carbon_footprint",
            {
                "emissions": [
                    {"fuel_type": "diesel", "co2e_emissions_kg": 268.5},
                    {"fuel_type": "gasoline", "co2e_emissions_kg": 150.2}
                ]
            }
        )

        # Verify "No Naked Numbers" compliance
        assert "total_co2e" in result
        assert "value" in result["total_co2e"]
        assert "unit" in result["total_co2e"]
        assert "source" in result["total_co2e"]
        assert result["total_co2e"]["unit"] == "kg_CO2e"
        assert result["total_co2e"]["value"] == 418.7

    def test_invoke_grid_factor_agent_tool_valid_args(self):
        """Invoke get_emission_factor with valid arguments"""
        agent = GridFactorAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        result = registry.invoke(
            "get_emission_factor",
            {
                "country": "US",
                "fuel_type": "electricity",
                "unit": "kWh"
            }
        )

        # Verify "No Naked Numbers" compliance
        assert "emission_factor" in result
        assert "value" in result["emission_factor"]
        assert "unit" in result["emission_factor"]
        assert "source" in result["emission_factor"]
        assert "country" in result
        assert result["country"] == "US"

    def test_invoke_missing_required_argument(self):
        """Invoke should raise ToolInvocationError for missing required args"""
        agent = CarbonAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        with pytest.raises(ToolInvocationError) as exc_info:
            registry.invoke(
                "calculate_carbon_footprint",
                {}  # Missing required "emissions" field
            )

        assert "validation failed" in str(exc_info.value).lower()

    def test_invoke_invalid_argument_type(self):
        """Invoke should raise ToolInvocationError for invalid argument types"""
        agent = CarbonAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        with pytest.raises(ToolInvocationError) as exc_info:
            registry.invoke(
                "calculate_carbon_footprint",
                {
                    "emissions": "not_an_array"  # Should be array
                }
            )

        assert "validation failed" in str(exc_info.value).lower()

    def test_invoke_nonexistent_tool(self):
        """Invoke should raise ToolInvocationError for unknown tool"""
        registry = ToolRegistry()

        with pytest.raises(ToolInvocationError) as exc_info:
            registry.invoke("nonexistent_tool", {})

        assert "not found" in str(exc_info.value).lower()


class TestClimateValidatorIntegration:
    """Test that tools return climate-compliant data"""

    def test_carbon_agent_no_naked_numbers(self):
        """CarbonAgent tool should return all values with units and sources"""
        agent = CarbonAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        result = registry.invoke(
            "calculate_carbon_footprint",
            {
                "emissions": [
                    {"fuel_type": "diesel", "co2e_emissions_kg": 100}
                ]
            }
        )

        # Check total_co2e
        assert isinstance(result["total_co2e"], dict)
        assert "value" in result["total_co2e"]
        assert "unit" in result["total_co2e"]
        assert "source" in result["total_co2e"]

        # Check total_co2e_tons
        assert isinstance(result["total_co2e_tons"], dict)
        assert "value" in result["total_co2e_tons"]
        assert "unit" in result["total_co2e_tons"]
        assert "source" in result["total_co2e_tons"]

    def test_grid_factor_agent_no_naked_numbers(self):
        """GridFactorAgent tool should return all values with units and sources"""
        agent = GridFactorAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        result = registry.invoke(
            "get_emission_factor",
            {
                "country": "US",
                "fuel_type": "electricity",
                "unit": "kWh"
            }
        )

        # Check emission_factor
        assert isinstance(result["emission_factor"], dict)
        assert "value" in result["emission_factor"]
        assert "unit" in result["emission_factor"]
        assert "source" in result["emission_factor"]
        assert "version" in result["emission_factor"]
        assert "last_updated" in result["emission_factor"]

    def test_energy_balance_agent_no_naked_numbers(self):
        """EnergyBalanceAgent tool should return all values with units and sources"""
        import pandas as pd

        agent = EnergyBalanceAgent()
        registry = ToolRegistry()
        registry.register_from_agent(agent)

        # Create sample data
        solar_df = pd.DataFrame({
            "dni_w_per_m2": [800, 900, 1000],
            "temp_c": [25, 26, 27]
        })
        load_df = pd.DataFrame({
            "demand_kWh": [100, 150, 200]
        })

        result = registry.invoke(
            "simulate_solar_energy_balance",
            {
                "solar_resource_df_json": solar_df.to_json(orient="split"),
                "load_profile_df_json": load_df.to_json(orient="split"),
                "required_aperture_area_m2": 1000
            }
        )

        # Check solar_fraction
        assert isinstance(result["solar_fraction"], dict)
        assert "value" in result["solar_fraction"]
        assert "unit" in result["solar_fraction"]
        assert "source" in result["solar_fraction"]
        assert result["solar_fraction"]["unit"] == "fraction"

        # Check total_solar_yield
        assert isinstance(result["total_solar_yield"], dict)
        assert "value" in result["total_solar_yield"]
        assert "unit" in result["total_solar_yield"]
        assert "source" in result["total_solar_yield"]
        assert result["total_solar_yield"]["unit"] == "GWh"


class TestToolTimeout:
    """Test tool timeout enforcement"""

    def test_tool_with_short_timeout(self):
        """Tool should have configurable timeout"""
        @tool(
            name="fast_tool",
            description="Fast tool",
            parameters_schema={"type": "object", "properties": {}},
            timeout_s=0.1
        )
        def fast_tool():
            return {"result": "done"}

        spec = fast_tool._tool_spec
        assert spec.timeout_s == 0.1

    def test_tool_default_timeout(self):
        """Tool should have default 30s timeout"""
        @tool(
            name="default_tool",
            description="Default timeout tool",
            parameters_schema={"type": "object", "properties": {}}
        )
        def default_tool():
            return {"result": "done"}

        spec = default_tool._tool_spec
        assert spec.timeout_s == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
