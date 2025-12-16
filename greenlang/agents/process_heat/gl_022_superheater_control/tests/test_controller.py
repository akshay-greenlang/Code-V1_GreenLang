"""
GL-022 SuperheaterControlAgent - Controller Tests

This module provides comprehensive tests for:
- Main controller process() method
- Synchronous run() method
- Asynchronous arun() method
- Control action generation
- Temperature deviation handling
- Spray control recommendations
- Explainability (recommendations/warnings)
- Determinism verification (same input = same output)
- Output validation

Target: 85%+ coverage for agent.py

Focus Areas:
- Process logic correctness
- Control decision making
- Recommendation generation
- Warning generation
- Provenance tracking
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

import pytest

# Add agent paths
AGENT_BASE_PATH = Path(__file__).parent.parent.parent.parent.parent.parent
BACKEND_AGENT_PATH = AGENT_BASE_PATH / "GL-Agent-Factory" / "backend" / "agents"
sys.path.insert(0, str(AGENT_BASE_PATH))
sys.path.insert(0, str(BACKEND_AGENT_PATH))

try:
    from gl_022_superheater_control.agent import SuperheaterControlAgent
    from gl_022_superheater_control.models import (
        SuperheaterInput,
        SuperheaterOutput,
        SprayControlAction,
        ControlParameters,
    )
    from gl_022_superheater_control.formulas import (
        calculate_saturation_temperature,
        calculate_superheat,
        generate_calculation_hash,
    )
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    SuperheaterControlAgent = None

# Skip all tests if agent not available
pytestmark = pytest.mark.skipif(not AGENT_AVAILABLE, reason="Agent module not available")


# =============================================================================
# MAIN PROCESS METHOD TESTS
# =============================================================================

class TestControllerProcess:
    """Tests for the main _process() method."""

    @pytest.mark.unit
    def test_process_returns_output_model(self, agent, valid_input):
        """Test _process returns SuperheaterOutput model."""
        result = agent._process(valid_input)

        assert isinstance(result, SuperheaterOutput)

    @pytest.mark.unit
    def test_process_calculates_saturation_temp(self, agent, valid_input):
        """Test saturation temperature is calculated correctly."""
        result = agent._process(valid_input)

        expected_t_sat = calculate_saturation_temperature(valid_input.steam_pressure_bar)
        assert result.saturation_temp_c == pytest.approx(expected_t_sat, abs=0.1)

    @pytest.mark.unit
    def test_process_calculates_superheat(self, agent, valid_input):
        """Test superheat is calculated correctly."""
        result = agent._process(valid_input)

        expected_superheat = calculate_superheat(
            valid_input.outlet_steam_temp_c,
            valid_input.steam_pressure_bar
        )
        assert result.current_superheat_c == pytest.approx(expected_superheat, abs=0.1)

    @pytest.mark.unit
    def test_process_calculates_deviation(self, agent, valid_input):
        """Test temperature deviation is calculated correctly."""
        result = agent._process(valid_input)

        expected_deviation = valid_input.outlet_steam_temp_c - valid_input.target_steam_temp_c
        assert result.temperature_deviation_c == pytest.approx(expected_deviation, abs=0.1)

    @pytest.mark.unit
    def test_process_generates_provenance_hash(self, agent, valid_input):
        """Test provenance hash is generated."""
        result = agent._process(valid_input)

        assert result.calculation_hash is not None
        assert len(result.calculation_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.calculation_hash)

    @pytest.mark.unit
    def test_process_includes_timestamp(self, agent, valid_input):
        """Test calculation timestamp is included."""
        before = datetime.utcnow()
        result = agent._process(valid_input)
        after = datetime.utcnow()

        assert before <= result.calculation_timestamp <= after

    @pytest.mark.unit
    def test_process_includes_version(self, agent, valid_input):
        """Test agent version is included in output."""
        result = agent._process(valid_input)

        assert result.agent_version == agent.VERSION
        assert result.agent_version == "1.0.0"


# =============================================================================
# RUN METHOD TESTS
# =============================================================================

class TestRunMethod:
    """Tests for the synchronous run() method."""

    @pytest.mark.unit
    def test_run_accepts_dict(self, agent, valid_input_data):
        """Test run() accepts dictionary input."""
        result = agent.run(valid_input_data)

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_run_returns_dict(self, agent, valid_input_data):
        """Test run() returns dictionary output."""
        result = agent.run(valid_input_data)

        assert isinstance(result, dict)
        assert "spray_control" in result
        assert "safety_status" in result

    @pytest.mark.unit
    def test_run_validates_input(self, agent, valid_input_data):
        """Test run() validates input data."""
        # Remove required field
        invalid_data = valid_input_data.copy()
        del invalid_data["equipment_id"]

        with pytest.raises(Exception):  # Pydantic ValidationError
            agent.run(invalid_data)

    @pytest.mark.unit
    def test_run_serializes_output(self, agent, valid_input_data):
        """Test run() serializes output to dict correctly."""
        result = agent.run(valid_input_data)

        # Check nested structures are also dicts
        assert isinstance(result["spray_control"], dict)
        assert isinstance(result["control_parameters"], dict)

    @pytest.mark.unit
    def test_run_output_contains_all_fields(self, agent, valid_input_data):
        """Test run() output contains all expected fields."""
        result = agent.run(valid_input_data)

        expected_fields = [
            "spray_control",
            "control_parameters",
            "current_superheat_c",
            "saturation_temp_c",
            "temperature_deviation_c",
            "within_tolerance",
            "spray_energy_loss_kw",
            "tube_metal_margin_c",
            "safety_status",
            "thermal_efficiency_impact_pct",
            "calculation_hash",
            "calculation_timestamp",
            "agent_version",
            "recommendations",
            "warnings",
        ]

        for field in expected_fields:
            assert field in result, f"Missing field: {field}"


# =============================================================================
# ASYNC RUN METHOD TESTS
# =============================================================================

class TestAsyncRunMethod:
    """Tests for the asynchronous arun() method."""

    @pytest.mark.asyncio
    async def test_arun_returns_same_as_run(self, agent, valid_input_data):
        """Test arun() returns same result as run()."""
        sync_result = agent.run(valid_input_data)
        async_result = await agent.arun(valid_input_data)

        # Results should be equivalent (except timestamp)
        assert sync_result["spray_control"] == async_result["spray_control"]
        assert sync_result["safety_status"] == async_result["safety_status"]

    @pytest.mark.asyncio
    async def test_arun_is_awaitable(self, agent, valid_input_data):
        """Test arun() is properly awaitable."""
        result = await agent.arun(valid_input_data)

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_arun_validates_input(self, agent, valid_input_data):
        """Test arun() validates input data."""
        invalid_data = valid_input_data.copy()
        del invalid_data["equipment_id"]

        with pytest.raises(Exception):
            await agent.arun(invalid_data)


# =============================================================================
# CONTROL ACTION TESTS
# =============================================================================

class TestControlActions:
    """Tests for control action generation."""

    @pytest.mark.unit
    def test_action_increase_when_above_target(self, agent, valid_input_data):
        """Test INCREASE action when temperature above target."""
        # Temperature above target (450 > 400)
        valid_input_data["outlet_steam_temp_c"] = 450.0
        valid_input_data["target_steam_temp_c"] = 400.0

        result = agent.run(valid_input_data)

        assert result["spray_control"]["action_type"] == "INCREASE"

    @pytest.mark.unit
    def test_action_decrease_when_below_target(self, agent, input_below_target):
        """Test DECREASE action when temperature below target."""
        result = agent.run(input_below_target)

        assert result["spray_control"]["action_type"] == "DECREASE"

    @pytest.mark.unit
    def test_action_maintain_within_tolerance(self, agent, input_at_target):
        """Test MAINTAIN action when within tolerance."""
        result = agent.run(input_at_target)

        assert result["spray_control"]["action_type"] == "MAINTAIN"

    @pytest.mark.unit
    def test_spray_flow_capped_at_maximum(self, agent, input_spray_capacity_exceeded):
        """Test spray flow is capped at maximum capacity."""
        result = agent.run(input_spray_capacity_exceeded)

        max_spray = input_spray_capacity_exceeded["max_spray_flow_kg_s"]
        assert result["spray_control"]["target_spray_flow_kg_s"] <= max_spray

    @pytest.mark.unit
    def test_spray_flow_non_negative(self, agent, input_below_target):
        """Test spray flow is never negative."""
        result = agent.run(input_below_target)

        assert result["spray_control"]["target_spray_flow_kg_s"] >= 0

    @pytest.mark.unit
    def test_valve_position_in_range(self, agent, valid_input_data):
        """Test valve position is always 0-100%."""
        result = agent.run(valid_input_data)

        valve_pos = result["spray_control"]["valve_position_pct"]
        assert 0 <= valve_pos <= 100

    @pytest.mark.unit
    def test_rate_of_change_limited(self, agent, valid_input_data):
        """Test rate of change is limited to prevent thermal shock."""
        result = agent.run(valid_input_data)

        rate = result["spray_control"]["rate_of_change_pct_per_min"]
        # Max rate is 10% per minute
        assert abs(rate) <= 10.0


# =============================================================================
# TOLERANCE TESTS
# =============================================================================

class TestToleranceBehavior:
    """Tests for temperature tolerance behavior."""

    @pytest.mark.unit
    def test_within_tolerance_true(self, agent, input_at_target):
        """Test within_tolerance is True when within bounds."""
        result = agent.run(input_at_target)

        assert result["within_tolerance"] is True

    @pytest.mark.unit
    def test_within_tolerance_false_above(self, agent, valid_input_data):
        """Test within_tolerance is False when above bounds."""
        # 450C outlet, 400C target, 5C tolerance -> outside
        result = agent.run(valid_input_data)

        assert result["within_tolerance"] is False

    @pytest.mark.unit
    def test_within_tolerance_false_below(self, agent, input_below_target):
        """Test within_tolerance is False when below bounds."""
        result = agent.run(input_below_target)

        assert result["within_tolerance"] is False

    @pytest.mark.unit
    @pytest.mark.parametrize("deviation,expected_within", [
        (0.0, True),    # Exactly at target
        (2.0, True),    # Within 5C tolerance
        (4.9, True),    # Just inside tolerance
        (5.0, True),    # At boundary (inclusive)
        (5.1, False),   # Just outside tolerance
        (-2.0, True),   # Below target but within tolerance
        (-5.0, True),   # At boundary below
        (-5.1, False),  # Just outside below
    ])
    def test_tolerance_boundary_cases(self, agent, valid_input_data, deviation, expected_within):
        """Test tolerance boundary cases."""
        target = 400.0
        valid_input_data["target_steam_temp_c"] = target
        valid_input_data["outlet_steam_temp_c"] = target + deviation
        valid_input_data["process_temp_tolerance_c"] = 5.0

        result = agent.run(valid_input_data)

        assert result["within_tolerance"] == expected_within


# =============================================================================
# RECOMMENDATION GENERATION TESTS
# =============================================================================

class TestRecommendationGeneration:
    """Tests for recommendation and warning generation."""

    @pytest.mark.unit
    def test_recommendations_list_type(self, agent, valid_input_data):
        """Test recommendations is a list."""
        result = agent.run(valid_input_data)

        assert isinstance(result["recommendations"], list)

    @pytest.mark.unit
    def test_warnings_list_type(self, agent, valid_input_data):
        """Test warnings is a list."""
        result = agent.run(valid_input_data)

        assert isinstance(result["warnings"], list)

    @pytest.mark.unit
    def test_increase_spray_recommendation(self, agent, valid_input_data):
        """Test recommendation to increase spray when needed."""
        # Temperature significantly above target
        valid_input_data["outlet_steam_temp_c"] = 480.0
        valid_input_data["target_steam_temp_c"] = 400.0

        result = agent.run(valid_input_data)

        # Should have recommendation to increase spray
        recs = " ".join(result["recommendations"]).lower()
        assert "increase" in recs or "spray" in recs

    @pytest.mark.unit
    def test_spray_capacity_warning(self, agent, input_spray_capacity_exceeded):
        """Test warning when spray capacity exceeded."""
        result = agent.run(input_spray_capacity_exceeded)

        # Should have warning about capacity
        warnings = " ".join(result["warnings"]).lower()
        assert "capacity" in warnings or "exceeds" in warnings

    @pytest.mark.unit
    def test_low_superheat_warning(self, agent, input_low_superheat):
        """Test warning when superheat is low."""
        result = agent.run(input_low_superheat)

        # Should have warning about low superheat
        warnings = " ".join(result["warnings"]).lower()
        assert "superheat" in warnings or "below" in warnings

    @pytest.mark.unit
    def test_high_efficiency_impact_warning(self, agent, valid_input_data):
        """Test warning when efficiency impact is high."""
        # Create conditions that cause high spray usage
        valid_input_data["outlet_steam_temp_c"] = 550.0
        valid_input_data["target_steam_temp_c"] = 350.0
        valid_input_data["steam_flow_kg_s"] = 50.0

        result = agent.run(valid_input_data)

        # If efficiency impact > 1%, should have warning
        if result["thermal_efficiency_impact_pct"] > 1.0:
            warnings = " ".join(result["warnings"]).lower()
            assert "efficiency" in warnings or "impact" in warnings

    @pytest.mark.unit
    def test_no_recommendations_when_optimal(self, agent, input_at_target):
        """Test minimal recommendations when operating optimally."""
        result = agent.run(input_at_target)

        # At target, safe, good superheat - minimal recommendations expected
        # (may still have some depending on other conditions)
        # Just verify we don't have critical recommendations
        recs = " ".join(result["recommendations"]).lower()
        assert "immediately" not in recs


# =============================================================================
# PID PARAMETER OUTPUT TESTS
# =============================================================================

class TestPIDParameterOutput:
    """Tests for PID parameter inclusion in output."""

    @pytest.mark.unit
    def test_control_parameters_included(self, agent, valid_input_data):
        """Test control parameters are in output."""
        result = agent.run(valid_input_data)

        assert "control_parameters" in result
        params = result["control_parameters"]

        assert "kp" in params
        assert "ki" in params
        assert "kd" in params
        assert "deadband_c" in params
        assert "max_rate_c_per_min" in params

    @pytest.mark.unit
    def test_control_parameters_match_agent(self, agent, valid_input_data):
        """Test control parameters match agent configuration."""
        result = agent.run(valid_input_data)

        assert result["control_parameters"]["kp"] == agent.pid_params["kp"]
        assert result["control_parameters"]["ki"] == agent.pid_params["ki"]
        assert result["control_parameters"]["kd"] == agent.pid_params["kd"]

    @pytest.mark.unit
    def test_different_config_different_params(self, valid_input_data, aggressive_pid_config, conservative_pid_config):
        """Test different configs produce different PID parameters."""
        agent_aggressive = SuperheaterControlAgent(config=aggressive_pid_config)
        agent_conservative = SuperheaterControlAgent(config=conservative_pid_config)

        result_aggressive = agent_aggressive.run(valid_input_data)
        result_conservative = agent_conservative.run(valid_input_data)

        # PID parameters should differ
        assert result_aggressive["control_parameters"]["kp"] != result_conservative["control_parameters"]["kp"]


# =============================================================================
# ENERGY METRICS TESTS
# =============================================================================

class TestEnergyMetrics:
    """Tests for energy metrics in output."""

    @pytest.mark.unit
    def test_spray_energy_loss_calculated(self, agent, valid_input_data):
        """Test spray energy loss is calculated."""
        result = agent.run(valid_input_data)

        assert "spray_energy_loss_kw" in result
        assert result["spray_energy_loss_kw"] >= 0

    @pytest.mark.unit
    def test_spray_water_cost_calculated(self, agent, valid_input_data):
        """Test spray water cost is calculated."""
        result = agent.run(valid_input_data)

        assert "spray_water_cost_per_hour" in result
        assert result["spray_water_cost_per_hour"] >= 0

    @pytest.mark.unit
    def test_efficiency_impact_calculated(self, agent, valid_input_data):
        """Test thermal efficiency impact is calculated."""
        result = agent.run(valid_input_data)

        assert "thermal_efficiency_impact_pct" in result
        assert result["thermal_efficiency_impact_pct"] >= 0

    @pytest.mark.unit
    def test_energy_loss_increases_with_spray(self, agent, valid_input_data):
        """Test energy loss increases with higher spray flow."""
        # Low temperature difference -> low spray
        valid_input_data["outlet_steam_temp_c"] = 410.0
        result_low = agent.run(valid_input_data)

        # High temperature difference -> high spray
        valid_input_data["outlet_steam_temp_c"] = 500.0
        result_high = agent.run(valid_input_data)

        assert result_high["spray_energy_loss_kw"] >= result_low["spray_energy_loss_kw"]


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism (zero-hallucination)."""

    @pytest.mark.unit
    def test_same_input_same_hash(self, agent, valid_input_data):
        """Test same input produces same provenance hash."""
        result1 = agent.run(valid_input_data)
        result2 = agent.run(valid_input_data)

        assert result1["calculation_hash"] == result2["calculation_hash"]

    @pytest.mark.unit
    def test_same_input_same_spray_control(self, agent, valid_input_data):
        """Test same input produces same spray control."""
        result1 = agent.run(valid_input_data)
        result2 = agent.run(valid_input_data)

        assert result1["spray_control"] == result2["spray_control"]

    @pytest.mark.unit
    def test_same_input_same_metrics(self, agent, valid_input_data):
        """Test same input produces same metrics."""
        result1 = agent.run(valid_input_data)
        result2 = agent.run(valid_input_data)

        assert result1["current_superheat_c"] == result2["current_superheat_c"]
        assert result1["saturation_temp_c"] == result2["saturation_temp_c"]
        assert result1["spray_energy_loss_kw"] == result2["spray_energy_loss_kw"]

    @pytest.mark.unit
    def test_determinism_multiple_iterations(self, agent, valid_input_data):
        """Test determinism over multiple iterations."""
        results = [agent.run(valid_input_data) for _ in range(10)]

        first_hash = results[0]["calculation_hash"]
        first_spray = results[0]["spray_control"]

        for result in results[1:]:
            assert result["calculation_hash"] == first_hash
            assert result["spray_control"] == first_spray

    @pytest.mark.unit
    def test_different_input_different_hash(self, agent, valid_input_data):
        """Test different input produces different hash."""
        result1 = agent.run(valid_input_data)

        # Modify input
        valid_input_data["outlet_steam_temp_c"] += 1.0
        result2 = agent.run(valid_input_data)

        assert result1["calculation_hash"] != result2["calculation_hash"]

    @pytest.mark.unit
    def test_independent_agents_same_result(self, valid_input_data, default_config):
        """Test independent agent instances produce same result."""
        agent1 = SuperheaterControlAgent(config=default_config)
        agent2 = SuperheaterControlAgent(config=default_config)

        result1 = agent1.run(valid_input_data)
        result2 = agent2.run(valid_input_data)

        assert result1["calculation_hash"] == result2["calculation_hash"]


# =============================================================================
# EDGE CASE HANDLING TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge case handling."""

    @pytest.mark.unit
    def test_zero_steam_flow(self, agent, valid_input_data):
        """Test handling of zero steam flow."""
        valid_input_data["steam_flow_kg_s"] = 0.0

        result = agent.run(valid_input_data)

        # Should handle gracefully
        assert result["spray_control"]["target_spray_flow_kg_s"] == 0.0

    @pytest.mark.unit
    def test_minimum_pressure(self, agent, valid_input_data):
        """Test handling at minimum pressure (1 bar)."""
        valid_input_data["steam_pressure_bar"] = 1.0
        valid_input_data["outlet_steam_temp_c"] = 150.0
        valid_input_data["target_steam_temp_c"] = 200.0
        valid_input_data["inlet_steam_temp_c"] = 160.0

        result = agent.run(valid_input_data)

        assert result["saturation_temp_c"] == pytest.approx(100.0, abs=2.0)

    @pytest.mark.unit
    def test_maximum_pressure(self, agent, valid_input_data):
        """Test handling at maximum pressure (200 bar)."""
        valid_input_data["steam_pressure_bar"] = 200.0
        valid_input_data["outlet_steam_temp_c"] = 550.0
        valid_input_data["target_steam_temp_c"] = 500.0
        valid_input_data["inlet_steam_temp_c"] = 600.0

        result = agent.run(valid_input_data)

        assert result["saturation_temp_c"] > 350  # Should be > 365C

    @pytest.mark.unit
    def test_temperature_exactly_at_target(self, agent, valid_input_data):
        """Test handling when temperature exactly at target."""
        valid_input_data["outlet_steam_temp_c"] = 400.0
        valid_input_data["target_steam_temp_c"] = 400.0

        result = agent.run(valid_input_data)

        assert result["temperature_deviation_c"] == 0.0
        assert result["within_tolerance"] is True
        assert result["spray_control"]["action_type"] == "MAINTAIN"

    @pytest.mark.unit
    def test_no_tube_temp_sensor(self, agent, input_no_tube_temp_sensor):
        """Test handling when tube temperature sensor not available."""
        result = agent.run(input_no_tube_temp_sensor)

        # Should use max_tube_metal_temp_c as margin
        assert result["tube_metal_margin_c"] == input_no_tube_temp_sensor["max_tube_metal_temp_c"]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestControllerPerformance:
    """Performance tests for the controller."""

    @pytest.mark.performance
    @pytest.mark.slow
    def test_process_time_single(self, agent, valid_input_data):
        """Test single process completes in reasonable time."""
        import time

        start = time.perf_counter()
        agent.run(valid_input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in < 50ms (generous for test environments)
        assert elapsed_ms < 50

    @pytest.mark.performance
    @pytest.mark.slow
    def test_throughput(self, agent, large_batch_inputs, performance_thresholds):
        """Test batch processing throughput."""
        import time

        start = time.perf_counter()
        results = [agent.run(inp) for inp in large_batch_inputs]
        elapsed = time.perf_counter() - start

        throughput = len(large_batch_inputs) / elapsed

        assert throughput >= performance_thresholds["throughput_min"], \
            f"Throughput {throughput:.0f} below minimum {performance_thresholds['throughput_min']}"

    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_all_valid(self, agent, large_batch_inputs):
        """Test all batch results are valid."""
        results = [agent.run(inp) for inp in large_batch_inputs[:100]]  # First 100

        for result in results:
            assert result["calculation_hash"] is not None
            assert result["safety_status"] in ["SAFE", "WARNING", "CRITICAL"]
            assert 0 <= result["spray_control"]["valve_position_pct"] <= 100


# =============================================================================
# MULTIPLE SCENARIO TESTS
# =============================================================================

class TestMultipleScenarios:
    """Tests covering multiple operational scenarios."""

    @pytest.mark.unit
    def test_scenario_normal_operation(self, agent, valid_input_data):
        """Test normal operation scenario."""
        result = agent.run(valid_input_data)

        assert result["safety_status"] == "SAFE"
        assert result["current_superheat_c"] > 0

    @pytest.mark.unit
    def test_scenario_high_pressure(self, agent, input_high_pressure):
        """Test high pressure operation scenario."""
        result = agent.run(input_high_pressure)

        assert result["saturation_temp_c"] > 300  # High pressure = high T_sat

    @pytest.mark.unit
    def test_scenario_low_pressure(self, agent, input_low_pressure):
        """Test low pressure operation scenario."""
        result = agent.run(input_low_pressure)

        assert result["saturation_temp_c"] < 160  # Low pressure = low T_sat

    @pytest.mark.unit
    def test_scenario_startup(self, agent, valid_input_data):
        """Test startup scenario (low load)."""
        valid_input_data["burner_load_pct"] = 30.0
        valid_input_data["steam_flow_kg_s"] = 5.0
        valid_input_data["outlet_steam_temp_c"] = 380.0
        valid_input_data["target_steam_temp_c"] = 400.0

        result = agent.run(valid_input_data)

        # Below target, should reduce spray
        assert result["spray_control"]["action_type"] in ["DECREASE", "MAINTAIN"]

    @pytest.mark.unit
    def test_scenario_high_load(self, agent, valid_input_data):
        """Test high load scenario."""
        valid_input_data["burner_load_pct"] = 95.0
        valid_input_data["steam_flow_kg_s"] = 45.0
        valid_input_data["outlet_steam_temp_c"] = 520.0

        result = agent.run(valid_input_data)

        # High load with high temp -> needs spray
        assert result["spray_control"]["action_type"] == "INCREASE"

    @pytest.mark.unit
    def test_scenario_load_change(self, agent, valid_input_data):
        """Test load change scenario."""
        # Before load change
        valid_input_data["burner_load_pct"] = 70.0
        valid_input_data["outlet_steam_temp_c"] = 410.0
        result_before = agent.run(valid_input_data)

        # After load increase
        valid_input_data["burner_load_pct"] = 90.0
        valid_input_data["outlet_steam_temp_c"] = 450.0
        result_after = agent.run(valid_input_data)

        # After load increase, more spray needed
        assert result_after["spray_control"]["target_spray_flow_kg_s"] >= \
               result_before["spray_control"]["target_spray_flow_kg_s"]
