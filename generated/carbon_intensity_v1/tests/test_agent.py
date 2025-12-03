"""
Test suite for CBAM Carbon Intensity Calculator.

This module provides unit tests for the CbamCarbonIntensityCalculatorAgent.
Generated from AgentSpec golden tests and property tests.

Run with: pytest tests/test_agent.py -v
"""

import pytest
from typing import Dict, Any

from carbon_intensity_v1.agent import CbamCarbonIntensityCalculatorAgent, CbamCarbonIntensityCalculatorAgentInput, CbamCarbonIntensityCalculatorAgentOutput


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def agent() -> CbamCarbonIntensityCalculatorAgent:
    """Create agent instance for testing."""
    return CbamCarbonIntensityCalculatorAgent(
        enable_provenance=True,
        enable_citations=True,
    )


@pytest.fixture
def sample_input() -> Dict[str, Any]:
    """Sample input data for testing."""
    return {
        "product_type": "sample_value",
        "production_quantity": 1,
        "total_emissions": 1,
    }


# =============================================================================
# Golden Tests (from AgentSpec)
# =============================================================================

class TestGolden:
    """Golden test cases from AgentSpec."""

    @pytest.mark.asyncio
    async def test_steel_basic_oxygen(self, agent: CbamCarbonIntensityCalculatorAgent):
        """
        steel_basic_oxygen
        """
        input_data = CbamCarbonIntensityCalculatorAgentInput(
            product_type="steel_hot_rolled",
            production_quantity=1000,
            total_emissions=1850,
        )

        result = await agent.run(input_data)

        assert result is not None
        assert result.output is not None
        assert abs(result.output.carbon_intensity - 1.85) < 0.1, f"Expected carbon_intensity=1.85, got {result.output.carbon_intensity}"



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
    """Unit tests for CbamCarbonIntensityCalculatorAgent."""

    def test_agent_initialization(self, agent: CbamCarbonIntensityCalculatorAgent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.agent_id == "cbam/carbon_intensity_v1"
        assert agent.agent_version == "1.0.0"

    def test_input_validation(self, agent: CbamCarbonIntensityCalculatorAgent, sample_input: Dict[str, Any]):
        """Test input validation."""
        input_data = CbamCarbonIntensityCalculatorAgentInput(**sample_input)
        assert input_data is not None

    @pytest.mark.asyncio
    async def test_execute_returns_output(self, agent: CbamCarbonIntensityCalculatorAgent, sample_input: Dict[str, Any]):
        """Test agent execution returns valid output."""
        input_data = CbamCarbonIntensityCalculatorAgentInput(**sample_input)
        result = await agent.run(input_data)

        assert result is not None
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_provenance_tracking(self, agent: CbamCarbonIntensityCalculatorAgent, sample_input: Dict[str, Any]):
        """Test provenance is tracked correctly."""
        input_data = CbamCarbonIntensityCalculatorAgentInput(**sample_input)
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
    async def test_lookup_cbam_benchmark_exists(self, agent: CbamCarbonIntensityCalculatorAgent):
        """Test lookup_cbam_benchmark tool is registered."""
        assert "lookup_cbam_benchmark" in agent._tools

    @pytest.mark.asyncio
    async def test_lookup_cbam_benchmark_execution(self, agent: CbamCarbonIntensityCalculatorAgent):
        """Test lookup_cbam_benchmark tool executes."""
        result = await agent.call_lookup_cbam_benchmark()
        assert result is not None


    @pytest.mark.asyncio
    async def test_calculate_carbon_intensity_exists(self, agent: CbamCarbonIntensityCalculatorAgent):
        """Test calculate_carbon_intensity tool is registered."""
        assert "calculate_carbon_intensity" in agent._tools

    @pytest.mark.asyncio
    async def test_calculate_carbon_intensity_execution(self, agent: CbamCarbonIntensityCalculatorAgent):
        """Test calculate_carbon_intensity tool executes."""
        result = await agent.call_calculate_carbon_intensity()
        assert result is not None

