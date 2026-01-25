"""
GreenLang Framework - MCP Calculator Tools Tests

Comprehensive test suite for MCP calculator tool definitions and implementations.
Tests cover:
- Tool definitions and parameters
- Input validation
- Calculation accuracy
- Provenance tracking
- Error handling

All tests follow GreenLang's zero-hallucination principle by verifying
deterministic calculations with known inputs and outputs.
"""

import pytest
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any

# Import modules under test
import sys
from pathlib import Path

# Add parent paths for imports
_framework_path = Path(__file__).parent.parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))

from advanced.mcp_protocol import (
    ToolCallRequest,
    ToolCallResponse,
    ToolCategory,
    SecurityLevel,
)
from tools.mcp_calculators import (
    # Tool definitions
    COMBUSTION_EFFICIENCY_DEFINITION,
    HEAT_BALANCE_DEFINITION,
    STEAM_PROPERTIES_DEFINITION,
    EMISSION_RATE_DEFINITION,
    HEAT_EXCHANGER_DEFINITION,
    # Tool classes
    CombustionEfficiencyTool,
    HeatBalanceTool,
    SteamPropertiesTool,
    EmissionRateTool,
    HeatExchangerTool,
    # Registry
    CALCULATOR_REGISTRY,
    create_calculator_registry,
    # Convenience functions
    get_calculator_tools,
    invoke_calculator,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def combustion_tool():
    """Create combustion efficiency tool instance."""
    return CombustionEfficiencyTool()


@pytest.fixture
def heat_balance_tool():
    """Create heat balance tool instance."""
    return HeatBalanceTool()


@pytest.fixture
def steam_tool():
    """Create steam properties tool instance."""
    return SteamPropertiesTool()


@pytest.fixture
def emission_tool():
    """Create emission rate tool instance."""
    return EmissionRateTool()


@pytest.fixture
def heat_exchanger_tool():
    """Create heat exchanger tool instance."""
    return HeatExchangerTool()


# =============================================================================
# TOOL DEFINITION TESTS
# =============================================================================

class TestToolDefinitions:
    """Test tool definitions are properly configured."""

    def test_combustion_efficiency_definition(self):
        """Test combustion efficiency tool definition."""
        defn = COMBUSTION_EFFICIENCY_DEFINITION
        assert defn.name == "calculate_combustion_efficiency"
        assert defn.category == ToolCategory.CALCULATOR
        assert defn.security_level == SecurityLevel.READ_ONLY
        assert len(defn.parameters) >= 4  # At least 4 required params

        # Check required parameters
        param_names = [p.name for p in defn.parameters]
        assert "fuel_higher_heating_value_mj_kg" in param_names
        assert "flue_gas_temperature_c" in param_names
        assert "ambient_temperature_c" in param_names
        assert "excess_air_percent" in param_names

    def test_heat_balance_definition(self):
        """Test heat balance tool definition."""
        defn = HEAT_BALANCE_DEFINITION
        assert defn.name == "calculate_heat_balance"
        assert defn.category == ToolCategory.CALCULATOR

        param_names = [p.name for p in defn.parameters]
        assert "heat_inputs" in param_names
        assert "heat_outputs" in param_names

    def test_steam_properties_definition(self):
        """Test steam properties tool definition."""
        defn = STEAM_PROPERTIES_DEFINITION
        assert defn.name == "calculate_steam_properties"
        assert "IAPWS-IF97" in defn.description

        param_names = [p.name for p in defn.parameters]
        assert "pressure_bar" in param_names

    def test_emission_rate_definition(self):
        """Test emission rate tool definition."""
        defn = EMISSION_RATE_DEFINITION
        assert defn.name == "calculate_emission_rate"
        assert "EPA" in defn.description

        # Check pollutant enum
        pollutant_param = next(p for p in defn.parameters if p.name == "pollutant")
        assert "NOx" in pollutant_param.enum
        assert "SO2" in pollutant_param.enum
        assert "CO2" in pollutant_param.enum

    def test_heat_exchanger_definition(self):
        """Test heat exchanger tool definition."""
        defn = HEAT_EXCHANGER_DEFINITION
        assert defn.name == "calculate_heat_exchanger"
        assert "LMTD" in defn.description or "NTU" in defn.description

        # Check configuration enum
        config_param = next(p for p in defn.parameters if p.name == "configuration")
        assert "counterflow" in config_param.enum
        assert "parallel" in config_param.enum


# =============================================================================
# COMBUSTION EFFICIENCY TESTS
# =============================================================================

class TestCombustionEfficiencyTool:
    """Test combustion efficiency calculations."""

    def test_basic_calculation(self, combustion_tool):
        """Test basic combustion efficiency calculation."""
        request = ToolCallRequest(
            tool_name="calculate_combustion_efficiency",
            arguments={
                "fuel_higher_heating_value_mj_kg": 50.0,
                "flue_gas_temperature_c": 200.0,
                "ambient_temperature_c": 25.0,
                "excess_air_percent": 20.0,
            }
        )

        response = combustion_tool.execute(request)

        assert response.success is True
        assert response.result is not None
        assert "efficiency_percent" in response.result
        assert 70 <= response.result["efficiency_percent"] <= 99
        assert response.result["provenance_hash"] != ""

    def test_high_efficiency_case(self, combustion_tool):
        """Test high efficiency scenario."""
        request = ToolCallRequest(
            tool_name="calculate_combustion_efficiency",
            arguments={
                "fuel_higher_heating_value_mj_kg": 55.0,
                "flue_gas_temperature_c": 150.0,  # Low stack temp
                "ambient_temperature_c": 20.0,
                "excess_air_percent": 10.0,  # Low excess air
                "radiation_loss_percent": 1.0,
            }
        )

        response = combustion_tool.execute(request)

        assert response.success is True
        # High efficiency expected
        assert response.result["efficiency_percent"] >= 85

    def test_low_efficiency_case(self, combustion_tool):
        """Test low efficiency scenario."""
        request = ToolCallRequest(
            tool_name="calculate_combustion_efficiency",
            arguments={
                "fuel_higher_heating_value_mj_kg": 30.0,
                "flue_gas_temperature_c": 400.0,  # High stack temp
                "ambient_temperature_c": 20.0,
                "excess_air_percent": 100.0,  # High excess air
                "fuel_moisture_percent": 30.0,  # High moisture
                "radiation_loss_percent": 3.0,
            }
        )

        response = combustion_tool.execute(request)

        assert response.success is True
        # Lower efficiency expected
        assert response.result["efficiency_percent"] < 85

    def test_loss_breakdown(self, combustion_tool):
        """Test that losses are properly broken down."""
        request = ToolCallRequest(
            tool_name="calculate_combustion_efficiency",
            arguments={
                "fuel_higher_heating_value_mj_kg": 50.0,
                "flue_gas_temperature_c": 200.0,
                "ambient_temperature_c": 25.0,
                "excess_air_percent": 20.0,
            }
        )

        response = combustion_tool.execute(request)
        result = response.result

        assert "stack_loss_percent" in result
        assert "radiation_loss_percent" in result
        assert "unburned_carbon_loss_percent" in result
        assert "moisture_loss_percent" in result
        assert "total_loss_percent" in result

        # Total loss should equal sum of components
        calculated_total = (
            result["stack_loss_percent"] +
            result["radiation_loss_percent"] +
            result["unburned_carbon_loss_percent"] +
            result["moisture_loss_percent"]
        )
        assert abs(calculated_total - result["total_loss_percent"]) < 0.01

    def test_provenance_determinism(self, combustion_tool):
        """Test that same inputs produce same provenance hash."""
        args = {
            "fuel_higher_heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 200.0,
            "ambient_temperature_c": 25.0,
            "excess_air_percent": 20.0,
        }

        request1 = ToolCallRequest(tool_name="calculate_combustion_efficiency", arguments=args)
        request2 = ToolCallRequest(tool_name="calculate_combustion_efficiency", arguments=args)

        response1 = combustion_tool.execute(request1)
        response2 = combustion_tool.execute(request2)

        assert response1.result["provenance_hash"] == response2.result["provenance_hash"]


# =============================================================================
# HEAT BALANCE TESTS
# =============================================================================

class TestHeatBalanceTool:
    """Test heat balance calculations."""

    def test_balanced_system(self, heat_balance_tool):
        """Test perfectly balanced system."""
        request = ToolCallRequest(
            tool_name="calculate_heat_balance",
            arguments={
                "heat_inputs": {"fuel": 1000},
                "heat_outputs": {"steam": 850},
                "heat_losses": {"stack": 100, "radiation": 50},
            }
        )

        response = heat_balance_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["heat_input_kw"] == 1000
        assert result["heat_output_kw"] == 850
        assert result["heat_loss_kw"] == 150
        assert result["is_balanced"] is True
        assert result["balance_error_percent"] < 1.0

    def test_unbalanced_system(self, heat_balance_tool):
        """Test unbalanced system detection."""
        request = ToolCallRequest(
            tool_name="calculate_heat_balance",
            arguments={
                "heat_inputs": {"fuel": 1000},
                "heat_outputs": {"steam": 600},  # Missing 400 kW
                "heat_losses": {"stack": 100},
            }
        )

        response = heat_balance_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["balance_error_percent"] > 1.0
        # With default 1% tolerance, this should be unbalanced
        assert result["is_balanced"] is False

    def test_efficiency_calculation(self, heat_balance_tool):
        """Test efficiency is correctly calculated."""
        request = ToolCallRequest(
            tool_name="calculate_heat_balance",
            arguments={
                "heat_inputs": {"fuel": 1000},
                "heat_outputs": {"steam": 880},
                "heat_losses": {"stack": 100, "radiation": 20},
            }
        )

        response = heat_balance_tool.execute(request)
        result = response.result

        expected_efficiency = 88.0  # 880/1000 * 100
        assert abs(result["efficiency_percent"] - expected_efficiency) < 0.1

    def test_multiple_streams(self, heat_balance_tool):
        """Test with multiple input/output streams."""
        request = ToolCallRequest(
            tool_name="calculate_heat_balance",
            arguments={
                "heat_inputs": {"fuel": 900, "preheat": 100},
                "heat_outputs": {"steam_hp": 500, "steam_lp": 300, "blowdown": 50},
                "heat_losses": {"stack": 100, "radiation": 30, "blowdown_flash": 20},
            }
        )

        response = heat_balance_tool.execute(request)
        result = response.result

        assert result["heat_input_kw"] == 1000
        assert result["heat_output_kw"] == 850
        assert result["heat_loss_kw"] == 150


# =============================================================================
# STEAM PROPERTIES TESTS
# =============================================================================

class TestSteamPropertiesTool:
    """Test steam property calculations."""

    def test_saturated_steam_at_1_bar(self, steam_tool):
        """Test saturated steam properties at 1 bar."""
        request = ToolCallRequest(
            tool_name="calculate_steam_properties",
            arguments={
                "pressure_bar": 1.01325,  # 1 atm
                "quality": 1.0,  # Saturated vapor
            }
        )

        response = steam_tool.execute(request)

        assert response.success is True
        result = response.result

        # At 1 atm, saturation temp should be ~100C
        assert 95 < result["temperature_c"] < 105
        assert result["phase"] in ["vapor", "two_phase"]

    def test_superheated_steam(self, steam_tool):
        """Test superheated steam properties."""
        request = ToolCallRequest(
            tool_name="calculate_steam_properties",
            arguments={
                "pressure_bar": 10.0,
                "temperature_c": 300.0,  # Well above saturation
            }
        )

        response = steam_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["phase"] == "vapor"
        assert result["quality"] is None  # No quality for superheated

    def test_subcooled_liquid(self, steam_tool):
        """Test subcooled liquid properties."""
        request = ToolCallRequest(
            tool_name="calculate_steam_properties",
            arguments={
                "pressure_bar": 10.0,
                "temperature_c": 50.0,  # Well below saturation
            }
        )

        response = steam_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["phase"] == "liquid"
        assert result["density_kg_m3"] > 900  # Liquid density

    def test_two_phase_region(self, steam_tool):
        """Test two-phase (wet steam) properties."""
        request = ToolCallRequest(
            tool_name="calculate_steam_properties",
            arguments={
                "pressure_bar": 5.0,
                "quality": 0.9,  # 90% vapor
            }
        )

        response = steam_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["phase"] == "two_phase"
        assert 0 <= result["quality"] <= 1

    def test_high_pressure_steam(self, steam_tool):
        """Test high pressure steam properties."""
        request = ToolCallRequest(
            tool_name="calculate_steam_properties",
            arguments={
                "pressure_bar": 100.0,
                "temperature_c": 500.0,
            }
        )

        response = steam_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["pressure_bar"] == 100.0
        assert result["temperature_c"] == 500.0


# =============================================================================
# EMISSION RATE TESTS
# =============================================================================

class TestEmissionRateTool:
    """Test emission rate calculations."""

    def test_nox_emission_rate(self, emission_tool):
        """Test NOx emission rate calculation."""
        request = ToolCallRequest(
            tool_name="calculate_emission_rate",
            arguments={
                "pollutant": "NOx",
                "concentration_ppm": 100.0,
                "stack_flow_rate_m3_hr": 50000.0,
                "stack_temperature_c": 200.0,
            }
        )

        response = emission_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["pollutant"] == "NOx"
        assert result["emission_rate_kg_hr"] > 0
        assert result["method"] == "EPA Method 19"

    def test_so2_from_mg_m3(self, emission_tool):
        """Test SO2 calculation from mg/m3 concentration."""
        request = ToolCallRequest(
            tool_name="calculate_emission_rate",
            arguments={
                "pollutant": "SO2",
                "concentration_mg_m3": 50.0,
                "stack_flow_rate_m3_hr": 100000.0,
                "stack_temperature_c": 180.0,
            }
        )

        response = emission_tool.execute(request)

        assert response.success is True
        result = response.result

        # 50 mg/m3 * 100000 m3/hr = 5000000 mg/hr = 5 kg/hr
        assert result["emission_rate_kg_hr"] > 0

    def test_o2_correction(self, emission_tool):
        """Test O2 correction is applied."""
        # Same concentration, different O2 levels
        args_base = {
            "pollutant": "CO",
            "concentration_ppm": 100.0,
            "stack_flow_rate_m3_hr": 50000.0,
            "stack_temperature_c": 150.0,
            "reference_oxygen_percent": 3.0,
        }

        # Low O2 (high combustion efficiency)
        request1 = ToolCallRequest(
            tool_name="calculate_emission_rate",
            arguments={**args_base, "measured_oxygen_percent": 2.0}
        )

        # High O2 (low combustion efficiency / more dilution)
        request2 = ToolCallRequest(
            tool_name="calculate_emission_rate",
            arguments={**args_base, "measured_oxygen_percent": 8.0}
        )

        response1 = emission_tool.execute(request1)
        response2 = emission_tool.execute(request2)

        # Corrected concentration at high O2 should be higher
        # (corrected to reference O2 basis)
        assert response2.result["concentration_mg_m3"] > response1.result["concentration_mg_m3"]

    def test_compliance_check(self, emission_tool):
        """Test compliance limit checking."""
        # Under limit
        request1 = ToolCallRequest(
            tool_name="calculate_emission_rate",
            arguments={
                "pollutant": "NOx",
                "concentration_mg_m3": 100.0,
                "stack_flow_rate_m3_hr": 50000.0,
                "stack_temperature_c": 200.0,
                "compliance_limit_mg_m3": 200.0,
            }
        )

        # Over limit
        request2 = ToolCallRequest(
            tool_name="calculate_emission_rate",
            arguments={
                "pollutant": "NOx",
                "concentration_mg_m3": 250.0,
                "stack_flow_rate_m3_hr": 50000.0,
                "stack_temperature_c": 200.0,
                "compliance_limit_mg_m3": 200.0,
            }
        )

        response1 = emission_tool.execute(request1)
        response2 = emission_tool.execute(request2)

        assert response1.result["is_compliant"] is True
        assert response2.result["is_compliant"] is False


# =============================================================================
# HEAT EXCHANGER TESTS
# =============================================================================

class TestHeatExchangerTool:
    """Test heat exchanger calculations."""

    def test_counterflow_exchanger(self, heat_exchanger_tool):
        """Test counterflow heat exchanger calculation."""
        request = ToolCallRequest(
            tool_name="calculate_heat_exchanger",
            arguments={
                "hot_inlet_temp_c": 200.0,
                "hot_outlet_temp_c": 100.0,
                "cold_inlet_temp_c": 30.0,
                "cold_outlet_temp_c": 80.0,
                "hot_mass_flow_kg_s": 2.0,
                "cold_mass_flow_kg_s": 4.0,
                "hot_specific_heat_kj_kg_k": 2.0,
                "cold_specific_heat_kj_kg_k": 4.186,
                "configuration": "counterflow",
            }
        )

        response = heat_exchanger_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["configuration"] == "counterflow"
        assert result["heat_duty_kw"] > 0
        assert result["lmtd_k"] > 0
        assert 0 < result["effectiveness"] < 1

    def test_parallel_flow_exchanger(self, heat_exchanger_tool):
        """Test parallel flow heat exchanger calculation."""
        request = ToolCallRequest(
            tool_name="calculate_heat_exchanger",
            arguments={
                "hot_inlet_temp_c": 150.0,
                "hot_outlet_temp_c": 80.0,
                "cold_inlet_temp_c": 20.0,
                "cold_outlet_temp_c": 60.0,
                "hot_mass_flow_kg_s": 1.5,
                "cold_mass_flow_kg_s": 2.0,
                "hot_specific_heat_kj_kg_k": 2.5,
                "cold_specific_heat_kj_kg_k": 4.186,
                "configuration": "parallel",
            }
        )

        response = heat_exchanger_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["configuration"] == "parallel"
        # Parallel flow typically has lower LMTD than counterflow
        assert result["lmtd_k"] > 0

    def test_ntu_method_with_ua(self, heat_exchanger_tool):
        """Test NTU method when UA is provided."""
        request = ToolCallRequest(
            tool_name="calculate_heat_exchanger",
            arguments={
                "hot_inlet_temp_c": 200.0,
                "cold_inlet_temp_c": 30.0,
                "hot_mass_flow_kg_s": 2.0,
                "cold_mass_flow_kg_s": 3.0,
                "hot_specific_heat_kj_kg_k": 2.0,
                "cold_specific_heat_kj_kg_k": 4.186,
                "overall_heat_transfer_coeff_w_m2_k": 500.0,
                "heat_transfer_area_m2": 10.0,
                "configuration": "counterflow",
                "method": "NTU",
            }
        )

        response = heat_exchanger_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["method"] == "NTU"
        assert result["ntu"] > 0
        assert result["outlet_temp_hot_c"] < 200.0  # Hot stream cools down
        assert result["outlet_temp_cold_c"] > 30.0  # Cold stream heats up

    def test_area_calculation(self, heat_exchanger_tool):
        """Test required area calculation."""
        request = ToolCallRequest(
            tool_name="calculate_heat_exchanger",
            arguments={
                "hot_inlet_temp_c": 150.0,
                "hot_outlet_temp_c": 80.0,
                "cold_inlet_temp_c": 20.0,
                "cold_outlet_temp_c": 60.0,
                "hot_mass_flow_kg_s": 1.0,
                "cold_mass_flow_kg_s": 1.75,
                "hot_specific_heat_kj_kg_k": 2.0,
                "cold_specific_heat_kj_kg_k": 4.0,
                "overall_heat_transfer_coeff_w_m2_k": 400.0,
            }
        )

        response = heat_exchanger_tool.execute(request)

        assert response.success is True
        result = response.result

        assert result["required_area_m2"] is not None
        assert result["required_area_m2"] > 0


# =============================================================================
# REGISTRY TESTS
# =============================================================================

class TestCalculatorRegistry:
    """Test the calculator tool registry."""

    def test_registry_created(self):
        """Test registry is properly created."""
        assert CALCULATOR_REGISTRY is not None
        tools = CALCULATOR_REGISTRY.list_tools()
        assert len(tools) == 5  # 5 calculator tools

    def test_registry_tools_registered(self):
        """Test all tools are registered."""
        tool_names = [t.name for t in CALCULATOR_REGISTRY.list_tools()]
        assert "calculate_combustion_efficiency" in tool_names
        assert "calculate_heat_balance" in tool_names
        assert "calculate_steam_properties" in tool_names
        assert "calculate_emission_rate" in tool_names
        assert "calculate_heat_exchanger" in tool_names

    def test_get_tool(self):
        """Test getting tool by name."""
        tool = CALCULATOR_REGISTRY.get_tool("calculate_combustion_efficiency")
        assert tool is not None
        assert tool.definition.name == "calculate_combustion_efficiency"

    def test_invoke_through_registry(self):
        """Test invoking tool through registry."""
        request = ToolCallRequest(
            tool_name="calculate_heat_balance",
            arguments={
                "heat_inputs": {"fuel": 1000},
                "heat_outputs": {"steam": 900},
                "heat_losses": {"stack": 100},
            }
        )

        response = CALCULATOR_REGISTRY.invoke(request)
        assert response.success is True


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_calculator_tools(self):
        """Test getting all calculator tools."""
        tools = get_calculator_tools()
        assert len(tools) == 5
        assert all(t.category == ToolCategory.CALCULATOR for t in tools)

    def test_invoke_calculator(self):
        """Test invoking calculator directly."""
        response = invoke_calculator(
            "calculate_heat_balance",
            {
                "heat_inputs": {"fuel": 500},
                "heat_outputs": {"steam": 450},
                "heat_losses": {"stack": 50},
            }
        )
        assert response.success is True
        assert response.result["is_balanced"] is True


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test input validation for calculator tools."""

    def test_missing_required_parameter(self, combustion_tool):
        """Test validation with missing required parameter."""
        request = ToolCallRequest(
            tool_name="calculate_combustion_efficiency",
            arguments={
                "fuel_higher_heating_value_mj_kg": 50.0,
                # Missing other required parameters
            }
        )

        # Validation should catch missing params
        errors = combustion_tool.validate_arguments(request.arguments)
        assert len(errors) > 0

    def test_parameter_out_of_range(self, combustion_tool):
        """Test validation with out-of-range parameter."""
        request = ToolCallRequest(
            tool_name="calculate_combustion_efficiency",
            arguments={
                "fuel_higher_heating_value_mj_kg": 100.0,  # Max is 60
                "flue_gas_temperature_c": 200.0,
                "ambient_temperature_c": 25.0,
                "excess_air_percent": 20.0,
            }
        )

        errors = combustion_tool.validate_arguments(request.arguments)
        assert len(errors) > 0
        assert any("fuel_higher_heating_value_mj_kg" in e for e in errors)


# =============================================================================
# GOLDEN MASTER TESTS (Determinism)
# =============================================================================

class TestDeterminism:
    """Test that calculations are deterministic (same input = same output)."""

    def test_combustion_efficiency_determinism(self, combustion_tool):
        """Test combustion efficiency is deterministic."""
        args = {
            "fuel_higher_heating_value_mj_kg": 50.0,
            "flue_gas_temperature_c": 200.0,
            "ambient_temperature_c": 25.0,
            "excess_air_percent": 20.0,
        }

        results = []
        for _ in range(3):
            request = ToolCallRequest(tool_name="test", arguments=args)
            response = combustion_tool.execute(request)
            results.append(response.result["efficiency_percent"])

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_heat_exchanger_determinism(self, heat_exchanger_tool):
        """Test heat exchanger is deterministic."""
        args = {
            "hot_inlet_temp_c": 150.0,
            "hot_outlet_temp_c": 80.0,
            "cold_inlet_temp_c": 20.0,
            "cold_outlet_temp_c": 60.0,
            "hot_mass_flow_kg_s": 1.0,
            "cold_mass_flow_kg_s": 1.5,
            "hot_specific_heat_kj_kg_k": 2.0,
            "cold_specific_heat_kj_kg_k": 4.186,
        }

        results = []
        for _ in range(3):
            request = ToolCallRequest(tool_name="test", arguments=args)
            response = heat_exchanger_tool.execute(request)
            results.append(response.result["heat_duty_kw"])

        assert all(r == results[0] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
