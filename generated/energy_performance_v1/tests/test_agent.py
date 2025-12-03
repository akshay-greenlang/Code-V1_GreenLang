"""
Test suite for Building Energy Performance Calculator.

This module provides unit tests for the BuildingEnergyPerformanceCalculatorAgent.
Generated from AgentSpec golden tests and property tests.

Run with: pytest tests/test_agent.py -v
"""

import pytest
from typing import Dict, Any

from energy_performance_v1.agent import BuildingEnergyPerformanceCalculatorAgent, BuildingEnergyPerformanceCalculatorAgentInput, BuildingEnergyPerformanceCalculatorAgentOutput


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent() -> BuildingEnergyPerformanceCalculatorAgent:
    """Create agent instance for testing."""
    return BuildingEnergyPerformanceCalculatorAgent(
        enable_provenance=True,
        enable_citations=True,
    )


@pytest.fixture
def sample_input() -> Dict[str, Any]:
    """Sample input data for testing."""
    return {
        "building_type": "sample_value",
        "floor_area_sqm": 1,
        "energy_consumption_kwh": 1,
        "climate_zone": "sample_value",
    }


# =============================================================================
# Golden Tests (from AgentSpec)
# =============================================================================

class TestGolden:
    """Golden test cases from AgentSpec."""

    @pytest.mark.asyncio
    async def test_office_building_compliant(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """
        office_building_compliant
        """
        input_data = BuildingEnergyPerformanceCalculatorAgentInput(
            building_type="office",
            floor_area_sqm=5000,
            energy_consumption_kwh=350000,
            climate_zone="4A",
        )

        result = await agent.run(input_data)

        assert result is not None
        assert result.output is not None
        assert abs(result.output.eui_kwh_per_sqm - 70) < 1, f"Expected eui_kwh_per_sqm=70, got {result.output.eui_kwh_per_sqm}"
        assert abs(result.output.bps_compliance_status - compliant) < 0.01, f"Expected bps_compliance_status=compliant, got {result.output.bps_compliance_status}"

    @pytest.mark.asyncio
    async def test_residential_building_noncompliant(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """
        residential_building_noncompliant
        """
        input_data = BuildingEnergyPerformanceCalculatorAgentInput(
            building_type="residential",
            floor_area_sqm=2000,
            energy_consumption_kwh=250000,
            climate_zone="5A",
        )

        result = await agent.run(input_data)

        assert result is not None
        assert result.output is not None
        assert abs(result.output.eui_kwh_per_sqm - 125) < 2, f"Expected eui_kwh_per_sqm=125, got {result.output.eui_kwh_per_sqm}"
        assert abs(result.output.bps_compliance_status - non_compliant) < 0.01, f"Expected bps_compliance_status=non_compliant, got {result.output.bps_compliance_status}"



# =============================================================================
# Property Tests
# =============================================================================

class TestProperties:
    """Property-based tests from AgentSpec."""

    def test_placeholder(self, agent):
        """Placeholder test - add property tests to AgentSpec."""
        assert agent is not None



# =============================================================================
# Unit Tests
# =============================================================================

class TestAgent:
    """Unit tests for BuildingEnergyPerformanceCalculatorAgent."""

    def test_agent_initialization(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_id == "buildings/energy_performance_v1"
        assert agent.agent_version == "1.0.0"

    def test_input_validation(self, agent: BuildingEnergyPerformanceCalculatorAgent, sample_input: Dict[str, Any]):
        """Test input validation."""
        input_data = BuildingEnergyPerformanceCalculatorAgentInput(**sample_input)
        assert input_data is not None

    @pytest.mark.asyncio
    async def test_execute_returns_output(self, agent: BuildingEnergyPerformanceCalculatorAgent, sample_input: Dict[str, Any]):
        """Test agent execution returns valid output."""
        input_data = BuildingEnergyPerformanceCalculatorAgentInput(**sample_input)
        result = await agent.run(input_data)

        assert result is not None
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_provenance_tracking(self, agent: BuildingEnergyPerformanceCalculatorAgent, sample_input: Dict[str, Any]):
        """Test provenance is tracked correctly."""
        input_data = BuildingEnergyPerformanceCalculatorAgentInput(**sample_input)
        result = await agent.run(input_data)

        assert result.provenance is not None
        assert result.provenance.input_hash is not None
        assert result.provenance.output_hash is not None
        assert result.provenance.provenance_chain is not None


# =============================================================================
# Tool Tests
# =============================================================================

class TestTools:
    """Tests for agent tools."""

    @pytest.mark.asyncio
    async def test_calculate_eui_exists(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """Test calculate_eui tool is registered."""
        assert "calculate_eui" in agent._tools

    @pytest.mark.asyncio
    async def test_calculate_eui_execution(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """Test calculate_eui tool executes."""
        result = await agent.call_calculate_eui()
        assert result is not None


    @pytest.mark.asyncio
    async def test_lookup_bps_threshold_exists(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """Test lookup_bps_threshold tool is registered."""
        assert "lookup_bps_threshold" in agent._tools

    @pytest.mark.asyncio
    async def test_lookup_bps_threshold_execution(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """Test lookup_bps_threshold tool executes."""
        result = await agent.call_lookup_bps_threshold()
        assert result is not None


    @pytest.mark.asyncio
    async def test_check_bps_compliance_exists(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """Test check_bps_compliance tool is registered."""
        assert "check_bps_compliance" in agent._tools

    @pytest.mark.asyncio
    async def test_check_bps_compliance_execution(self, agent: BuildingEnergyPerformanceCalculatorAgent):
        """Test check_bps_compliance tool executes."""
        result = await agent.call_check_bps_compliance()
        assert result is not None

