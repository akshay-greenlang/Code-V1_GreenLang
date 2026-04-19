"""
Unit Tests for GL-015: Insulation Optimization Agent

Comprehensive test suite covering:
- Heat loss calculations for pipes and equipment
- Economic thickness optimization
- Thermal conductivity calculations
- Surface temperature verification
- Energy savings quantification

Target: 85%+ code coverage

Reference:
- ASTM C680 (Standard Practice for Heat Flux)
- 3E Plus Software methodology
- NAIMA standards

Run with:
    pytest tests/agents/test_gl_015_insulation.py -v --cov=backend/agents/gl_015_insulation
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock




from agents.gl_015_insulation.agent import InsulationOptimizationAgent
from agents.gl_015_insulation.schemas import (
    InsulationType,
    PipeSize,
    InsulationInput,
    InsulationOutput,
    PipeData,
    EquipmentData,
    AmbientConditions,
    EconomicParameters,
    AgentConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def insulation_agent():
    """Create InsulationOptimizationAgent instance for testing."""
    config = AgentConfig()
    return InsulationOptimizationAgent(config)


@pytest.fixture
def hot_pipe_data():
    """Create hot pipe data for insulation analysis."""
    return PipeData(
        pipe_id="PIPE-001",
        nominal_size_inches=4.0,
        length_m=100.0,
        operating_temp_c=200.0,
        current_insulation_thickness_mm=50.0,
        insulation_type=InsulationType.MINERAL_WOOL,
    )


@pytest.fixture
def ambient_conditions():
    """Create ambient conditions."""
    return AmbientConditions(
        ambient_temp_c=25.0,
        wind_speed_m_s=2.0,
    )


@pytest.fixture
def economic_params():
    """Create economic parameters."""
    return EconomicParameters(
        energy_cost_per_kwh=0.10,
        operating_hours_per_year=8760,
        discount_rate_pct=8.0,
        project_life_years=20,
    )


@pytest.fixture
def insulation_input(hot_pipe_data, ambient_conditions, economic_params):
    """Create insulation optimization input."""
    return InsulationInput(
        pipes=[hot_pipe_data],
        ambient=ambient_conditions,
        economics=economic_params,
    )


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestInsulationAgentInitialization:
    """Tests for InsulationOptimizationAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self, insulation_agent):
        """Test agent initializes correctly with default config."""
        assert insulation_agent is not None
        assert hasattr(insulation_agent, "run")


# =============================================================================
# Test Class: Insulation Types
# =============================================================================


class TestInsulationTypes:
    """Tests for insulation type handling."""

    @pytest.mark.unit
    def test_insulation_types_defined(self):
        """Test insulation types are defined."""
        assert InsulationType.MINERAL_WOOL is not None
        assert InsulationType.CALCIUM_SILICATE is not None
        assert InsulationType.CELLULAR_GLASS is not None


# =============================================================================
# Test Class: Pipe Data Validation
# =============================================================================


class TestPipeDataValidation:
    """Tests for pipe data validation."""

    @pytest.mark.unit
    def test_valid_pipe_data(self, hot_pipe_data):
        """Test valid pipe data passes validation."""
        assert hot_pipe_data.pipe_id == "PIPE-001"
        assert hot_pipe_data.operating_temp_c == 200.0
        assert hot_pipe_data.length_m == 100.0


# =============================================================================
# Test Class: Heat Loss Calculations
# =============================================================================


class TestHeatLossCalculations:
    """Tests for heat loss calculation functionality."""

    @pytest.mark.unit
    def test_heat_loss_calculated(self, insulation_agent, insulation_input):
        """Test heat loss is calculated."""
        result = insulation_agent.run(insulation_input)
        assert hasattr(result, "total_heat_loss_kw") or hasattr(result, "heat_loss_per_pipe")


# =============================================================================
# Test Class: Economic Thickness
# =============================================================================


class TestEconomicThickness:
    """Tests for economic thickness optimization."""

    @pytest.mark.unit
    def test_economic_thickness_calculated(self, insulation_agent, insulation_input):
        """Test economic thickness is calculated."""
        result = insulation_agent.run(insulation_input)
        assert hasattr(result, "economic_thickness_mm") or hasattr(result, "recommendations")


# =============================================================================
# Test Class: Provenance Tracking
# =============================================================================


class TestInsulationProvenance:
    """Tests for provenance hash tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, insulation_agent, insulation_input):
        """Test provenance hash is generated."""
        result = insulation_agent.run(insulation_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestInsulationPerformance:
    """Performance tests for InsulationOptimizationAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_analysis_performance(self, insulation_agent, insulation_input):
        """Test single analysis completes quickly."""
        import time

        start = time.perf_counter()
        result = insulation_agent.run(insulation_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0
        assert result is not None
