# -*- coding: utf-8 -*-
"""
Unit tests for TrapcatcherAgent.

Tests main agent orchestration, diagnostic workflow, and fleet analysis.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent import (
    TrapcatcherAgent,
    AgentConfig,
    AgentMode,
    TrapDiagnosticInput,
    AlertLevel,
)


class TestTrapcatcherAgent:
    """Tests for TrapcatcherAgent class."""

    @pytest.fixture
    def agent(self):
        """Create default agent."""
        return TrapcatcherAgent()

    @pytest.fixture
    def healthy_input(self):
        """Create input for healthy trap."""
        return TrapDiagnosticInput(
            trap_id="ST-001",
            acoustic_amplitude_db=65.0,
            acoustic_frequency_khz=38.0,
            inlet_temp_c=185.0,
            outlet_temp_c=95.0,
            pressure_bar_g=10.0,
            trap_type="thermodynamic",
        )

    @pytest.fixture
    def failed_input(self):
        """Create input for failed trap."""
        return TrapDiagnosticInput(
            trap_id="ST-002",
            acoustic_amplitude_db=95.0,
            acoustic_frequency_khz=25.0,
            inlet_temp_c=185.0,
            outlet_temp_c=180.0,
            pressure_bar_g=10.0,
            trap_type="thermodynamic",
        )

    @pytest.fixture
    def leaking_input(self):
        """Create input for leaking trap."""
        return TrapDiagnosticInput(
            trap_id="ST-003",
            acoustic_amplitude_db=78.0,
            acoustic_frequency_khz=32.0,
            inlet_temp_c=185.0,
            outlet_temp_c=150.0,
            pressure_bar_g=10.0,
            trap_type="thermodynamic",
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.config is not None
        assert agent.config.agent_id == "GL-008"

    def test_agent_mode(self, agent):
        """Test agent mode is set."""
        assert agent.mode in (AgentMode.PRODUCTION, AgentMode.DEVELOPMENT, AgentMode.TESTING)

    def test_diagnose_healthy_trap(self, agent, healthy_input):
        """Test diagnosis of healthy trap."""
        result = agent.diagnose_trap(healthy_input)

        assert result is not None
        assert result.trap_id == "ST-001"
        assert result.condition == "healthy"
        assert result.severity == "none"
        assert result.energy_loss_kw >= 0
        assert result.alert_level == AlertLevel.NONE

    def test_diagnose_failed_trap(self, agent, failed_input):
        """Test diagnosis of failed trap."""
        result = agent.diagnose_trap(failed_input)

        assert result is not None
        assert result.trap_id == "ST-002"
        assert result.condition == "failed"
        assert result.severity in ("high", "critical")
        assert result.energy_loss_kw > 0
        assert result.alert_level in (AlertLevel.HIGH, AlertLevel.CRITICAL)

    def test_diagnose_leaking_trap(self, agent, leaking_input):
        """Test diagnosis of leaking trap."""
        result = agent.diagnose_trap(leaking_input)

        assert result is not None
        assert result.condition in ("leaking", "degraded")
        assert result.alert_level != AlertLevel.NONE

    def test_diagnosis_includes_timestamp(self, agent, healthy_input):
        """Test diagnosis includes timestamp."""
        result = agent.diagnose_trap(healthy_input)

        assert result.timestamp is not None
        assert isinstance(result.timestamp, datetime)

    def test_diagnosis_includes_provenance(self, agent, healthy_input):
        """Test diagnosis includes provenance hash."""
        result = agent.diagnose_trap(healthy_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) > 0

    def test_diagnosis_with_explanation(self, agent, failed_input):
        """Test diagnosis includes explanation."""
        result = agent.diagnose_trap(failed_input, include_explanation=True)

        assert result.explanation is not None

    def test_diagnosis_without_explanation(self, agent, failed_input):
        """Test diagnosis without explanation."""
        result = agent.diagnose_trap(failed_input, include_explanation=False)

        assert result.explanation is None

    def test_recommended_action(self, agent, failed_input):
        """Test recommended action is provided."""
        result = agent.diagnose_trap(failed_input)

        assert result.recommended_action is not None
        assert len(result.recommended_action) > 0

    def test_energy_loss_calculation(self, agent, failed_input):
        """Test energy loss is calculated."""
        result = agent.diagnose_trap(failed_input)

        assert result.energy_loss_kw >= 0
        assert result.annual_cost_usd >= 0
        assert result.annual_co2_kg >= 0

    def test_fleet_analysis(self, agent, healthy_input, failed_input, leaking_input):
        """Test fleet-wide analysis."""
        inputs = [healthy_input, failed_input, leaking_input]
        results, summary = agent.analyze_fleet(inputs)

        assert len(results) == 3
        assert summary is not None
        assert summary.total_traps == 3

    def test_fleet_summary_counts(self, agent, healthy_input, failed_input, leaking_input):
        """Test fleet summary counts."""
        inputs = [healthy_input, failed_input, leaking_input]
        _, summary = agent.analyze_fleet(inputs)

        assert summary.healthy_count >= 0
        assert summary.failed_count >= 0
        assert summary.leaking_count >= 0
        assert (summary.healthy_count + summary.failed_count +
                summary.leaking_count + summary.unknown_count) == summary.total_traps

    def test_fleet_summary_totals(self, agent, healthy_input, failed_input):
        """Test fleet summary totals."""
        inputs = [healthy_input, failed_input]
        _, summary = agent.analyze_fleet(inputs)

        assert summary.total_energy_loss_kw >= 0
        assert summary.total_annual_cost_usd >= 0
        assert summary.total_annual_co2_kg >= 0

    def test_fleet_health_score(self, agent, healthy_input, failed_input):
        """Test fleet health score calculation."""
        # All healthy
        healthy_inputs = [healthy_input] * 3
        _, healthy_summary = agent.analyze_fleet(healthy_inputs)

        # Mixed
        mixed_inputs = [healthy_input, failed_input]
        _, mixed_summary = agent.analyze_fleet(mixed_inputs)

        # Healthy fleet should have higher score
        assert healthy_summary.fleet_health_score >= mixed_summary.fleet_health_score

    def test_agent_status(self, agent):
        """Test agent status retrieval."""
        status = agent.get_status()

        assert "agent_id" in status
        assert "version" in status
        assert "mode" in status
        assert "status" in status
        assert "statistics" in status

    def test_deterministic_diagnosis(self, agent, healthy_input):
        """Test that same input produces same output."""
        result1 = agent.diagnose_trap(healthy_input)
        result2 = agent.diagnose_trap(healthy_input)

        assert result1.condition == result2.condition
        assert result1.severity == result2.severity
        assert result1.energy_loss_kw == result2.energy_loss_kw
        assert result1.provenance_hash == result2.provenance_hash

    def test_statistics_increment(self, agent, healthy_input, failed_input):
        """Test statistics are incremented."""
        initial_status = agent.get_status()
        initial_count = initial_status["statistics"]["total_diagnostics"]

        agent.diagnose_trap(healthy_input)
        agent.diagnose_trap(failed_input)

        final_status = agent.get_status()
        final_count = final_status["statistics"]["total_diagnostics"]

        assert final_count == initial_count + 2

    def test_custom_config(self):
        """Test agent with custom configuration."""
        config = AgentConfig(
            agent_id="GL-008-TEST",
            agent_name="TRAPCATCHER-TEST",
            version="1.0.0-test",
        )
        agent = TrapcatcherAgent(config)

        assert agent.config.agent_id == "GL-008-TEST"


class TestTrapDiagnosticInput:
    """Tests for TrapDiagnosticInput class."""

    def test_basic_input(self):
        """Test basic input creation."""
        input_data = TrapDiagnosticInput(
            trap_id="ST-001",
            pressure_bar_g=10.0,
        )

        assert input_data.trap_id == "ST-001"
        assert input_data.pressure_bar_g == 10.0

    def test_optional_fields(self):
        """Test optional fields default to None."""
        input_data = TrapDiagnosticInput(
            trap_id="ST-001",
            pressure_bar_g=10.0,
        )

        assert input_data.acoustic_amplitude_db is None
        assert input_data.inlet_temp_c is None

    def test_full_input(self):
        """Test input with all fields."""
        input_data = TrapDiagnosticInput(
            trap_id="ST-001",
            acoustic_amplitude_db=65.0,
            acoustic_frequency_khz=38.0,
            inlet_temp_c=185.0,
            outlet_temp_c=95.0,
            pressure_bar_g=10.0,
            trap_type="thermodynamic",
            orifice_diameter_mm=6.35,
            trap_age_years=3.0,
            last_maintenance_days=180,
            location="Building A",
            system="Steam Main",
        )

        assert input_data.trap_type == "thermodynamic"
        assert input_data.location == "Building A"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_levels_ordered(self):
        """Test alert levels have correct ordering."""
        levels = [AlertLevel.NONE, AlertLevel.LOW, AlertLevel.MEDIUM,
                  AlertLevel.HIGH, AlertLevel.CRITICAL]

        # Each level should be greater than previous (by value)
        for i in range(len(levels) - 1):
            assert levels[i].value < levels[i + 1].value
