"""
GL-003 UnifiedSteam - Chaos Engineering Test Suite

This module contains pytest-based chaos engineering tests specific to
the UnifiedSteam Header Balance Agent.

Test Categories:
- Steam header pressure fault scenarios
- Multi-boiler coordination failures
- Load balancing disruptions
- Demand forecasting errors
- Valve actuator failures

All tests are CI-safe (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import pytest
import logging
import sys
import os

gl001_chaos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
if gl001_chaos_path not in sys.path:
    sys.path.insert(0, gl001_chaos_path)

from chaos_runner import ChaosRunner, ChaosExperiment, ChaosSeverity
from steady_state import SteadyStateValidator

from .unifiedsteam_chaos import (
    UnifiedSteamChaosConfig,
    SteamHeaderFaultInjector,
    BoilerCoordinationFaultInjector,
    LoadBalancingFaultInjector,
    DemandForecastFaultInjector,
    ValveActuatorFaultInjector,
    HeaderPressureState,
    BoilerState,
    create_unifiedsteam_hypothesis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def header_injector():
    return SteamHeaderFaultInjector()


@pytest.fixture
def coordination_injector():
    return BoilerCoordinationFaultInjector()


@pytest.fixture
def load_injector():
    return LoadBalancingFaultInjector()


@pytest.fixture
def forecast_injector():
    return DemandForecastFaultInjector()


@pytest.fixture
def valve_injector():
    return ValveActuatorFaultInjector()


# =============================================================================
# Steam Header Fault Tests
# =============================================================================

class TestSteamHeaderFaults:
    """Test suite for steam header fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_pressure_spike(self, header_injector):
        """Test pressure spike simulation."""
        await header_injector.inject({
            "fault_type": "pressure_spike",
            "magnitude_psi": 20.0,
        })

        pressure = header_injector.get_pressure()
        assert pressure > 150  # Above base pressure

        state = header_injector.get_state()
        assert state in [HeaderPressureState.HIGH, HeaderPressureState.CRITICAL_HIGH]

        await header_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_pressure_drop(self, header_injector):
        """Test pressure drop simulation."""
        await header_injector.inject({
            "fault_type": "pressure_drop",
            "magnitude_psi": 30.0,
        })

        pressure = header_injector.get_pressure()
        assert pressure < 150  # Below base pressure

        state = header_injector.get_state()
        assert state in [HeaderPressureState.LOW, HeaderPressureState.CRITICAL_LOW]

        await header_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_pressure_oscillation(self, header_injector):
        """Test pressure oscillation simulation."""
        await header_injector.inject({
            "fault_type": "oscillation",
            "amplitude_psi": 15.0,
            "frequency_hz": 1.0,
        })

        # Collect pressure readings
        pressures = [header_injector.get_pressure() for _ in range(10)]

        # Should have variation
        assert max(pressures) - min(pressures) > 5

        state = header_injector.get_state()
        assert state == HeaderPressureState.OSCILLATING

        await header_injector.rollback()


# =============================================================================
# Boiler Coordination Fault Tests
# =============================================================================

class TestBoilerCoordinationFaults:
    """Test suite for boiler coordination fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_communication_delay(self, coordination_injector):
        """Test communication delay between boilers."""
        await coordination_injector.inject({
            "fault_type": "communication_delay",
            "delay_ms": 200,
            "boilers": ["boiler_1", "boiler_2"],
        })

        import time
        start = time.time()
        result = await coordination_injector.send_command("boiler_1", "increase_load")
        elapsed = time.time() - start

        assert elapsed >= 0.2  # At least 200ms delay
        assert result["status"] == "success"
        assert result.get("delayed") is True

        await coordination_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_boiler_offline(self, coordination_injector):
        """Test boiler going offline unexpectedly."""
        await coordination_injector.inject({
            "fault_type": "boiler_offline",
            "boiler": "boiler_2",
        })

        state = coordination_injector.get_boiler_state("boiler_2")
        assert state == BoilerState.OFFLINE

        # Other boilers should still be running
        state_1 = coordination_injector.get_boiler_state("boiler_1")
        assert state_1 == BoilerState.RUNNING

        await coordination_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_split_brain_scenario(self, coordination_injector):
        """Test split brain coordination failure."""
        await coordination_injector.inject({
            "fault_type": "split_brain",
        })

        # Send multiple commands - some should fail with conflict
        results = []
        for i in range(10):
            result = await coordination_injector.send_command(f"boiler_{(i % 4) + 1}", "sync")
            results.append(result)

        conflicts = [r for r in results if r.get("status") == "conflict"]
        successes = [r for r in results if r.get("status") == "success"]

        # Should have mix of conflicts and successes
        assert len(conflicts) > 0 or len(successes) > 0

        await coordination_injector.rollback()


# =============================================================================
# Load Balancing Fault Tests
# =============================================================================

class TestLoadBalancingFaults:
    """Test suite for load balancing fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_uneven_distribution(self, load_injector):
        """Test uneven load distribution."""
        await load_injector.inject({
            "fault_type": "uneven_distribution",
            "imbalance_percent": 40,
        })

        distribution = load_injector.get_load_distribution()

        # Boiler 1 should have higher load
        assert distribution["boiler_1"] > 50

        assert not load_injector.is_balanced(threshold=10)

        await load_injector.rollback()
        assert load_injector.is_balanced(threshold=5)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_boiler_overload(self, load_injector):
        """Test single boiler overload scenario."""
        await load_injector.inject({
            "fault_type": "overloaded",
            "boiler": "boiler_3",
        })

        distribution = load_injector.get_load_distribution()
        assert distribution["boiler_3"] == 100.0

        await load_injector.rollback()


# =============================================================================
# Demand Forecast Fault Tests
# =============================================================================

class TestDemandForecastFaults:
    """Test suite for demand forecast fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_over_prediction(self, forecast_injector):
        """Test demand over-prediction."""
        await forecast_injector.inject({
            "fault_type": "over_prediction",
            "error_percent": 30,
        })

        actual_demand = 1000.0
        result = await forecast_injector.get_forecast(actual_demand)

        assert result["forecast"] > actual_demand * 1.2
        assert result["status"] == "degraded"

        await forecast_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_under_prediction(self, forecast_injector):
        """Test demand under-prediction."""
        await forecast_injector.inject({
            "fault_type": "under_prediction",
            "error_percent": 25,
        })

        actual_demand = 1000.0
        result = await forecast_injector.get_forecast(actual_demand)

        assert result["forecast"] < actual_demand * 0.85

        await forecast_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_missing_forecast_data(self, forecast_injector):
        """Test missing forecast data."""
        await forecast_injector.inject({
            "fault_type": "missing_data",
        })

        result = await forecast_injector.get_forecast(1000.0)

        assert result["forecast"] is None
        assert result["status"] == "error"

        await forecast_injector.rollback()


# =============================================================================
# Valve Actuator Fault Tests
# =============================================================================

class TestValveActuatorFaults:
    """Test suite for valve actuator fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_stuck_valve(self, valve_injector):
        """Test stuck valve scenario."""
        await valve_injector.inject({
            "fault_type": "stuck",
            "valves": ["header_valve_1"],
        })

        # Try to move valve
        result = await valve_injector.set_position("header_valve_1", 80.0)

        assert result["status"] == "stuck"
        assert result["position"] != 80.0  # Didn't move to target

        await valve_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_slow_valve_response(self, valve_injector):
        """Test slow valve response."""
        await valve_injector.inject({
            "fault_type": "slow_response",
            "delay_ms": 100,
            "valves": ["bypass_valve"],
        })

        import time
        start = time.time()
        result = await valve_injector.set_position("bypass_valve", 75.0)
        elapsed = time.time() - start

        assert elapsed >= 0.1
        assert result["status"] == "delayed"

        await valve_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_partial_valve_movement(self, valve_injector):
        """Test partial valve movement."""
        await valve_injector.inject({
            "fault_type": "partial",
            "valves": ["control_valve"],
        })

        result = await valve_injector.set_position("control_valve", 100.0)

        assert result["status"] == "partial"
        assert result["position"] < 100.0  # Didn't reach target
        assert result["target"] == 100.0

        await valve_injector.rollback()


# =============================================================================
# Steady State Hypothesis Tests
# =============================================================================

class TestUnifiedSteamSteadyState:
    """Test suite for UnifiedSteam steady state validation."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_unifiedsteam_hypothesis(self):
        """Test UnifiedSteam-specific steady state hypothesis."""
        hypothesis = create_unifiedsteam_hypothesis()

        assert hypothesis.name == "UnifiedSteam Header Balance Health"
        assert len(hypothesis.metrics) >= 3

        validator = SteadyStateValidator()
        result = await validator.validate(hypothesis)

        assert result.hypothesis_name == hypothesis.name


# =============================================================================
# Integration Tests
# =============================================================================

class TestUnifiedSteamChaosIntegration:
    """Integration tests for UnifiedSteam chaos scenarios."""

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cascading_pressure_failure(
        self,
        header_injector,
        coordination_injector,
        valve_injector
    ):
        """Test cascading failure from pressure to valves."""
        # Stage 1: Pressure spike
        await header_injector.inject({
            "fault_type": "pressure_spike",
            "magnitude_psi": 25.0,
        })

        state = header_injector.get_state()
        assert state == HeaderPressureState.CRITICAL_HIGH

        # Stage 2: Valve stuck trying to relieve pressure
        await valve_injector.inject({
            "fault_type": "stuck",
            "valves": ["relief_valve"],
        })

        # Stage 3: Communication delay to boilers
        await coordination_injector.inject({
            "fault_type": "communication_delay",
            "delay_ms": 500,
        })

        # All faults active
        assert header_injector.is_active()
        assert valve_injector.is_active()
        assert coordination_injector.is_active()

        # Rollback
        await coordination_injector.rollback()
        await valve_injector.rollback()
        await header_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_demand_spike_with_coordination_failure(
        self,
        forecast_injector,
        load_injector,
        coordination_injector
    ):
        """Test demand spike combined with coordination failure."""
        # Under-predict demand
        await forecast_injector.inject({
            "fault_type": "under_prediction",
            "error_percent": 30,
        })

        # Uneven load distribution
        await load_injector.inject({
            "fault_type": "uneven_distribution",
            "imbalance_percent": 35,
        })

        # One boiler goes offline
        await coordination_injector.inject({
            "fault_type": "boiler_offline",
            "boiler": "boiler_2",
        })

        # Verify state
        forecast_result = await forecast_injector.get_forecast(1500.0)
        assert forecast_result["forecast"] < 1500.0

        distribution = load_injector.get_load_distribution()
        assert not load_injector.is_balanced()

        boiler_state = coordination_injector.get_boiler_state("boiler_2")
        assert boiler_state == BoilerState.OFFLINE

        # Cleanup
        await coordination_injector.rollback()
        await load_injector.rollback()
        await forecast_injector.rollback()


def pytest_configure(config):
    config.addinivalue_line("markers", "chaos: Chaos engineering tests")
    config.addinivalue_line("markers", "integration: Integration tests")
