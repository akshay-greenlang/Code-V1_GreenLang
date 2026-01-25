"""
GL-004 Burnmaster - Chaos Engineering Test Suite

This module contains pytest-based chaos engineering tests specific to
the Burnmaster Combustion Optimizer Agent.

Test Categories:
- Combustion ratio fault scenarios
- Burner management failures
- Flame detection failures
- Emission monitoring disruptions
- Oxygen trim control failures

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

from .burnmaster_chaos import (
    BurnmasterChaosConfig,
    CombustionRatioFaultInjector,
    BurnerManagementFaultInjector,
    FlameDetectionFaultInjector,
    EmissionMonitoringFaultInjector,
    OxygenTrimFaultInjector,
    BurnerState,
    FlameStatus,
    EmissionLevel,
    create_burnmaster_hypothesis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def ratio_injector():
    return CombustionRatioFaultInjector()


@pytest.fixture
def burner_injector():
    return BurnerManagementFaultInjector()


@pytest.fixture
def flame_injector():
    return FlameDetectionFaultInjector()


@pytest.fixture
def emission_injector():
    return EmissionMonitoringFaultInjector()


@pytest.fixture
def o2_trim_injector():
    return OxygenTrimFaultInjector()


# =============================================================================
# Combustion Ratio Fault Tests
# =============================================================================

class TestCombustionRatioFaults:
    """Test suite for combustion ratio fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_lean_combustion(self, ratio_injector):
        """Test lean combustion (excess air) simulation."""
        await ratio_injector.inject({
            "fault_type": "lean",
            "deviation_percent": 15,
        })

        ratio = ratio_injector.get_ratio()
        assert ratio > 15.0  # Above stoichiometric

        quality = ratio_injector.get_combustion_quality()
        assert quality == "lean"

        await ratio_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_rich_combustion(self, ratio_injector):
        """Test rich combustion (excess fuel) simulation."""
        await ratio_injector.inject({
            "fault_type": "rich",
            "deviation_percent": 15,
        })

        ratio = ratio_injector.get_ratio()
        assert ratio < 15.0  # Below stoichiometric

        quality = ratio_injector.get_combustion_quality()
        assert quality == "rich"

        await ratio_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_ratio_oscillation(self, ratio_injector):
        """Test air/fuel ratio oscillation."""
        await ratio_injector.inject({
            "fault_type": "oscillation",
            "amplitude": 2.0,
            "frequency_hz": 1.0,
        })

        ratios = [ratio_injector.get_ratio() for _ in range(10)]
        variance = max(ratios) - min(ratios)

        assert variance > 1.0  # Should have significant variation

        await ratio_injector.rollback()


# =============================================================================
# Burner Management Fault Tests
# =============================================================================

class TestBurnerManagementFaults:
    """Test suite for burner management fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_ignition_failure(self, burner_injector):
        """Test burner ignition failure."""
        await burner_injector.inject({
            "fault_type": "ignition_failure",
            "failure_probability": 1.0,  # 100% failure for testing
        })

        result = await burner_injector.start_burner("burner_1")

        assert result["status"] == "ignition_failed"
        assert burner_injector.get_burner_state("burner_1") == BurnerState.LOCKOUT

        await burner_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_burner_lockout(self, burner_injector):
        """Test burner lockout condition."""
        await burner_injector.inject({
            "fault_type": "lockout",
            "burner": "burner_2",
        })

        state = burner_injector.get_burner_state("burner_2")
        assert state == BurnerState.LOCKOUT

        # Cannot start locked out burner
        result = await burner_injector.start_burner("burner_2")
        assert result["status"] == "lockout"

        await burner_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_unexpected_shutdown(self, burner_injector):
        """Test unexpected burner shutdown."""
        await burner_injector.inject({
            "fault_type": "unexpected_shutdown",
        })

        result = await burner_injector.start_burner("burner_1")

        assert result["status"] == "shutdown"

        await burner_injector.rollback()


# =============================================================================
# Flame Detection Fault Tests
# =============================================================================

class TestFlameDetectionFaults:
    """Test suite for flame detection fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_false_negative_flame_detection(self, flame_injector):
        """Test false negative (missed flame) detection."""
        await flame_injector.inject({
            "fault_type": "false_negative",
            "probability": 1.0,  # Always miss
        })

        result = flame_injector.detect_flame(actual_flame_present=True)

        assert result["detected"] is False
        assert result["status"] == FlameStatus.NOT_DETECTED
        assert "flame_not_detected" in result.get("error", "")

        await flame_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_false_positive_flame_detection(self, flame_injector):
        """Test false positive (ghost flame) detection."""
        await flame_injector.inject({
            "fault_type": "false_positive",
            "probability": 1.0,  # Always false positive
        })

        result = flame_injector.detect_flame(actual_flame_present=False)

        assert result["detected"] is True
        assert "ghost_flame" in result.get("warning", "")

        await flame_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_sensor_failure(self, flame_injector):
        """Test flame sensor failure."""
        await flame_injector.inject({
            "fault_type": "sensor_failure",
        })

        result = flame_injector.detect_flame(actual_flame_present=True)

        assert result["detected"] is None
        assert result["status"] == FlameStatus.SENSOR_FAULT

        await flame_injector.rollback()


# =============================================================================
# Emission Monitoring Fault Tests
# =============================================================================

class TestEmissionMonitoringFaults:
    """Test suite for emission monitoring fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_nox_sensor_drift(self, emission_injector):
        """Test NOx sensor drift."""
        await emission_injector.inject({
            "fault_type": "nox_drift",
            "drift_ppm": 25,
        })

        emissions = emission_injector.get_emissions()

        assert emissions["nox_ppm"] > 50  # Above typical + drift
        assert emissions["status"] == "nox_elevated"

        compliance = emission_injector.get_compliance_level()
        assert compliance == EmissionLevel.VIOLATION

        await emission_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_co_sensor_failure(self, emission_injector):
        """Test CO sensor failure."""
        await emission_injector.inject({
            "fault_type": "co_sensor_failure",
        })

        emissions = emission_injector.get_emissions()

        assert emissions["co_ppm"] is None
        assert emissions["status"] == "co_sensor_offline"

        compliance = emission_injector.get_compliance_level()
        assert compliance == EmissionLevel.SENSOR_OFFLINE

        await emission_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calibration_error(self, emission_injector):
        """Test emission sensor calibration error."""
        await emission_injector.inject({
            "fault_type": "calibration_error",
            "error_factor": 1.5,  # 50% over-reading
        })

        emissions = emission_injector.get_emissions()

        assert emissions["status"] == "calibration_suspect"
        # NOx should read higher due to calibration error
        assert emissions["nox_ppm"] > 40

        await emission_injector.rollback()


# =============================================================================
# Oxygen Trim Fault Tests
# =============================================================================

class TestOxygenTrimFaults:
    """Test suite for oxygen trim fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_control_loop_failure(self, o2_trim_injector):
        """Test O2 trim control loop failure."""
        await o2_trim_injector.inject({
            "fault_type": "control_failure",
        })

        # Get multiple readings - should drift
        readings = [o2_trim_injector.get_o2_level() for _ in range(5)]

        # Values should vary without control
        variance = max(readings) - min(readings)
        assert variance > 0  # Some drift expected

        result = await o2_trim_injector.adjust_trim(3.0)
        assert result["status"] == "control_fault"

        await o2_trim_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_o2_hunting(self, o2_trim_injector):
        """Test O2 level hunting (oscillation)."""
        await o2_trim_injector.inject({
            "fault_type": "hunting",
            "amplitude": 2.0,
            "frequency_hz": 1.0,
        })

        readings = [o2_trim_injector.get_o2_level() for _ in range(10)]

        # Should oscillate around setpoint
        assert max(readings) > 3.0  # Above setpoint
        assert min(readings) < 3.0  # Below setpoint

        await o2_trim_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_response_delay(self, o2_trim_injector):
        """Test O2 trim response delay."""
        await o2_trim_injector.inject({
            "fault_type": "response_delay",
            "delay_ms": 100,  # Short for testing
        })

        import time
        start = time.time()
        result = await o2_trim_injector.adjust_trim(4.0)
        elapsed = time.time() - start

        assert elapsed >= 0.1
        assert result["status"] == "delayed"

        await o2_trim_injector.rollback()


# =============================================================================
# Steady State Hypothesis Tests
# =============================================================================

class TestBurnmasterSteadyState:
    """Test suite for Burnmaster steady state validation."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_burnmaster_hypothesis(self):
        """Test Burnmaster-specific steady state hypothesis."""
        hypothesis = create_burnmaster_hypothesis()

        assert hypothesis.name == "Burnmaster Combustion Optimizer Health"
        assert len(hypothesis.metrics) >= 4

        validator = SteadyStateValidator()
        result = await validator.validate(hypothesis)

        assert result.hypothesis_name == hypothesis.name


# =============================================================================
# Integration Tests
# =============================================================================

class TestBurnmasterChaosIntegration:
    """Integration tests for Burnmaster chaos scenarios."""

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_combustion_cascade_failure(
        self,
        ratio_injector,
        flame_injector,
        emission_injector
    ):
        """Test cascading failure from combustion to emissions."""
        # Stage 1: Rich combustion
        await ratio_injector.inject({
            "fault_type": "rich",
            "deviation_percent": 20,
        })

        quality = ratio_injector.get_combustion_quality()
        assert quality == "rich"

        # Stage 2: Weak flame due to rich mixture
        await flame_injector.inject({
            "fault_type": "weak_signal",
        })

        flame_result = flame_injector.detect_flame(True)
        assert flame_result["status"] == FlameStatus.WEAK

        # Stage 3: Elevated CO emissions from rich combustion
        await emission_injector.inject({
            "fault_type": "calibration_error",
            "error_factor": 1.8,
        })

        emissions = emission_injector.get_emissions()
        assert emissions["co_ppm"] > 80

        # Rollback all
        await emission_injector.rollback()
        await flame_injector.rollback()
        await ratio_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_safety_shutdown_scenario(
        self,
        burner_injector,
        flame_injector,
        o2_trim_injector
    ):
        """Test safety shutdown scenario."""
        # Burner running
        await burner_injector.start_burner("burner_1")

        # Flame detection fails
        await flame_injector.inject({
            "fault_type": "false_negative",
            "probability": 1.0,
        })

        flame_result = flame_injector.detect_flame(True)
        assert flame_result["detected"] is False

        # This would trigger safety shutdown
        await burner_injector.inject({
            "fault_type": "unexpected_shutdown",
        })

        state = burner_injector.get_burner_state("burner_1")
        # After shutdown, burner is in shutdown state
        result = await burner_injector.start_burner("burner_1")
        assert result["status"] == "shutdown"

        # Cleanup
        await o2_trim_injector.rollback()
        await flame_injector.rollback()
        await burner_injector.rollback()


def pytest_configure(config):
    config.addinivalue_line("markers", "chaos: Chaos engineering tests")
    config.addinivalue_line("markers", "integration: Integration tests")
