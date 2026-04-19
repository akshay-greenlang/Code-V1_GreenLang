# -*- coding: utf-8 -*-
"""
Unit tests for GL-024 Air Preheater Agent Main Orchestration Module

Tests agent initialization, orchestration flow, and integration of all components.

Author: GreenLang Test Engineering Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from greenlang.agents.process_heat.gl_024_air_preheater.agent import (
    AirPreheaterAgent,
    AgentState,
    AgentMetrics,
    create_agent,
)
from greenlang.agents.process_heat.gl_024_air_preheater.config import (
    AirPreheaterConfig,
    PreheaterType,
    AirPreheaterType,
    create_test_config,
)
from greenlang.agents.process_heat.gl_024_air_preheater.schemas import (
    AirPreheaterInput,
    AirPreheaterOutput,
    HeatTransferAnalysis,
    LeakageAnalysis,
    ColdEndProtection,
    FoulingAnalysis,
    GasComposition,
    RegenerativeOperatingData,
    RotorStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def test_config():
    """Create test configuration."""
    return create_test_config("APH-TEST-001")


@pytest.fixture
def agent(test_config):
    """Create agent with test configuration."""
    return AirPreheaterAgent(config=test_config)


@pytest.fixture
def sample_input():
    """Create sample input for testing."""
    return {
        "preheater_id": "APH-TEST-001",
        "preheater_type": PreheaterType.REGENERATIVE,
        "boiler_load_pct": 85.0,
        "gas_inlet_temp_f": 650.0,
        "gas_outlet_temp_f": 300.0,
        "gas_flow_lb_hr": 500000.0,
        "gas_dp_in_wc": 3.5,
        "air_inlet_temp_f": 80.0,
        "air_outlet_temp_f": 550.0,
        "air_flow_lb_hr": 480000.0,
        "air_dp_in_wc": 4.5,
        "o2_inlet_pct": 3.0,
        "o2_outlet_pct": 4.5,
        "so3_ppm": 10.0,
        "so2_ppm": 1000.0,
    }


@pytest.fixture
def mock_input(sample_input):
    """Create mock input object."""
    mock = MagicMock(spec=AirPreheaterInput)
    for key, value in sample_input.items():
        setattr(mock, key, value)
    mock.equipment_tag = "APH-TEST-001"
    mock.operating_mode = "NORMAL"
    mock.gas_composition = {"H2O": 8.0, "CO2": 14.0}
    return mock


# =============================================================================
# AGENT INITIALIZATION TESTS
# =============================================================================

class TestAgentInitialization:
    """Test suite for agent initialization."""

    @pytest.mark.unit
    def test_agent_creation_with_default_config(self):
        """Test agent creation with default configuration."""
        agent = AirPreheaterAgent()
        assert agent is not None
        assert agent.config is not None
        assert agent.state == AgentState.IDLE

    @pytest.mark.unit
    def test_agent_creation_with_custom_config(self, test_config):
        """Test agent creation with custom configuration."""
        agent = AirPreheaterAgent(config=test_config)
        assert agent.config == test_config

    @pytest.mark.unit
    def test_agent_factory_function(self, test_config):
        """Test agent factory function."""
        agent = create_agent(config=test_config)
        assert isinstance(agent, AirPreheaterAgent)

    @pytest.mark.unit
    def test_agent_has_calculator(self, agent):
        """Test agent has calculator component."""
        assert hasattr(agent, 'calculator')
        assert agent.calculator is not None

    @pytest.mark.unit
    def test_agent_has_explainer(self, agent):
        """Test agent has explainer component."""
        assert hasattr(agent, 'explainer')
        assert agent.explainer is not None

    @pytest.mark.unit
    def test_agent_has_provenance_tracker(self, agent):
        """Test agent has provenance tracker."""
        assert hasattr(agent, 'provenance')
        assert agent.provenance is not None

    @pytest.mark.unit
    def test_agent_initial_metrics(self, agent):
        """Test agent initial metrics."""
        assert agent.metrics.total_analyses == 0
        assert agent.metrics.successful_analyses == 0
        assert agent.metrics.failed_analyses == 0


# =============================================================================
# AGENT STATE TESTS
# =============================================================================

class TestAgentState:
    """Test suite for agent state management."""

    @pytest.mark.unit
    def test_initial_state_idle(self, agent):
        """Test agent starts in IDLE state."""
        assert agent.state == AgentState.IDLE

    @pytest.mark.unit
    def test_agent_state_enum_values(self):
        """Test agent state enumeration values."""
        assert AgentState.IDLE.value == "idle"
        assert AgentState.ANALYZING.value == "analyzing"
        assert AgentState.OPTIMIZING.value == "optimizing"
        assert AgentState.ERROR.value == "error"


# =============================================================================
# AGENT STATUS TESTS
# =============================================================================

class TestAgentStatus:
    """Test suite for agent status reporting."""

    @pytest.mark.unit
    def test_get_status(self, agent):
        """Test agent status retrieval."""
        status = agent.get_status()

        assert "agent_id" in status
        assert status["agent_id"] == "GL-024"
        assert "agent_name" in status
        assert status["agent_name"] == "AIRPREHEATER"
        assert "version" in status
        assert "state" in status
        assert "metrics" in status

    @pytest.mark.unit
    def test_status_contains_metrics(self, agent):
        """Test status contains metrics."""
        status = agent.get_status()
        metrics = status["metrics"]

        assert "total_analyses" in metrics
        assert "successful_analyses" in metrics
        assert "failed_analyses" in metrics
        assert "success_rate_pct" in metrics


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test suite for input validation."""

    @pytest.mark.unit
    def test_validate_gas_temperature_order(self, agent):
        """Test validation fails when gas outlet > inlet."""
        mock_input = MagicMock()
        mock_input.gas_inlet_temp_f = 300.0
        mock_input.gas_outlet_temp_f = 650.0  # Higher than inlet - invalid
        mock_input.air_outlet_temp_f = 550.0
        mock_input.air_inlet_temp_f = 80.0
        mock_input.gas_flow_rate_lb_hr = 500000.0
        mock_input.air_flow_rate_lb_hr = 480000.0
        mock_input.o2_inlet_pct = 3.0

        with pytest.raises(ValueError):
            agent._validate_inputs(mock_input)

    @pytest.mark.unit
    def test_validate_air_temperature_order(self, agent):
        """Test validation fails when air outlet < inlet."""
        mock_input = MagicMock()
        mock_input.gas_inlet_temp_f = 650.0
        mock_input.gas_outlet_temp_f = 300.0
        mock_input.air_outlet_temp_f = 80.0  # Lower than inlet - invalid
        mock_input.air_inlet_temp_f = 550.0
        mock_input.gas_flow_rate_lb_hr = 500000.0
        mock_input.air_flow_rate_lb_hr = 480000.0
        mock_input.o2_inlet_pct = 3.0

        with pytest.raises(ValueError):
            agent._validate_inputs(mock_input)

    @pytest.mark.unit
    def test_validate_positive_flow_rates(self, agent):
        """Test validation fails for non-positive flow rates."""
        mock_input = MagicMock()
        mock_input.gas_inlet_temp_f = 650.0
        mock_input.gas_outlet_temp_f = 300.0
        mock_input.air_outlet_temp_f = 550.0
        mock_input.air_inlet_temp_f = 80.0
        mock_input.gas_flow_rate_lb_hr = 0.0  # Zero flow - invalid
        mock_input.air_flow_rate_lb_hr = 480000.0
        mock_input.o2_inlet_pct = 3.0

        with pytest.raises(ValueError):
            agent._validate_inputs(mock_input)


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestAgentMetrics:
    """Test suite for agent metrics."""

    @pytest.mark.unit
    def test_metrics_dataclass(self):
        """Test AgentMetrics dataclass."""
        metrics = AgentMetrics()
        assert metrics.total_analyses == 0
        assert metrics.successful_analyses == 0
        assert metrics.failed_analyses == 0
        assert metrics.average_processing_time_ms == 0.0

    @pytest.mark.unit
    def test_update_metrics_success(self, agent):
        """Test metrics update on success."""
        initial_total = agent.metrics.total_analyses
        agent._update_metrics(processing_time_ms=100.0, success=True)

        assert agent.metrics.total_analyses == initial_total + 1
        assert agent.metrics.successful_analyses == 1
        assert agent.metrics.last_analysis_time is not None

    @pytest.mark.unit
    def test_update_metrics_failure(self, agent):
        """Test metrics update on failure."""
        initial_total = agent.metrics.total_analyses
        agent._update_metrics(processing_time_ms=0.0, success=False)

        assert agent.metrics.total_analyses == initial_total + 1
        assert agent.metrics.failed_analyses == 1


# =============================================================================
# HELPER METHOD TESTS
# =============================================================================

class TestHelperMethods:
    """Test suite for agent helper methods."""

    @pytest.mark.unit
    def test_calculate_excess_air(self, agent):
        """Test excess air calculation from O2."""
        # At 3% O2, excess air ~ 15-20%
        excess_air = agent._calculate_excess_air(3.0)
        assert 10.0 < excess_air < 25.0

    @pytest.mark.unit
    def test_calculate_optimal_bisector_critical(self, agent):
        """Test optimal bisector for critical corrosion risk."""
        from greenlang.agents.process_heat.gl_024_air_preheater.schemas import CorrosionRiskLevel

        position = agent._calculate_optimal_bisector(
            CorrosionRiskLevel.CRITICAL,
            current_position=50.0,
        )
        # Should increase for cold-end protection
        assert position >= 60.0

    @pytest.mark.unit
    def test_calculate_scaph_steam_critical(self, agent):
        """Test SCAPH steam calculation for critical risk."""
        from greenlang.agents.process_heat.gl_024_air_preheater.schemas import CorrosionRiskLevel

        steam_flow = agent._calculate_scaph_steam(
            cold_end_margin=5.0,
            corrosion_risk=CorrosionRiskLevel.CRITICAL,
        )
        # Should be maximum steam flow
        assert steam_flow >= 3000.0


# =============================================================================
# AGENT CONSTANTS TESTS
# =============================================================================

class TestAgentConstants:
    """Test suite for agent constants."""

    @pytest.mark.unit
    def test_agent_id(self):
        """Test agent ID constant."""
        assert AirPreheaterAgent.AGENT_ID == "GL-024"

    @pytest.mark.unit
    def test_agent_name(self):
        """Test agent name constant."""
        assert AirPreheaterAgent.AGENT_NAME == "AIRPREHEATER"

    @pytest.mark.unit
    def test_agent_version(self):
        """Test agent version constant."""
        assert AirPreheaterAgent.VERSION == "1.0.0"
