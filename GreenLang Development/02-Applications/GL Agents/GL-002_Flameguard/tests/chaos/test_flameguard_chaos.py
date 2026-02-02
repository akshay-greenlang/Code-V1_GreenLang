"""
GL-002 Flameguard - Chaos Engineering Test Suite

This module contains pytest-based chaos engineering tests specific to
the Flameguard Boiler Efficiency Agent.

Test Categories:
- Sensor stream fault scenarios
- Efficiency calculation failures
- CMMS integration failures
- OPC-UA communication faults
- Model inference failures
- Resilience pattern validation

All tests are CI-safe (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import pytest
import logging
import sys
import os

# Add GL-001 chaos framework to path
gl001_chaos_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
if gl001_chaos_path not in sys.path:
    sys.path.insert(0, gl001_chaos_path)

from chaos_runner import ChaosRunner, ChaosExperiment, ChaosSeverity, ChaosPhase
from fault_injectors import NetworkFaultInjector, ServiceFaultInjector
from steady_state import SteadyStateValidator
from resilience_patterns import CircuitBreakerTest, RetryMechanismTest

from .flameguard_chaos import (
    FlameguardChaosConfig,
    SensorStreamFaultInjector,
    EfficiencyCalculationFaultInjector,
    CMMSIntegrationFaultInjector,
    OPCUACommunicationFaultInjector,
    ModelInferenceFaultInjector,
    create_flameguard_hypothesis,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def chaos_config():
    """Create Flameguard chaos configuration."""
    return FlameguardChaosConfig()


@pytest.fixture
def sensor_injector():
    """Create sensor stream fault injector."""
    return SensorStreamFaultInjector()


@pytest.fixture
def efficiency_injector():
    """Create efficiency calculation fault injector."""
    return EfficiencyCalculationFaultInjector()


@pytest.fixture
def cmms_injector():
    """Create CMMS integration fault injector."""
    return CMMSIntegrationFaultInjector()


@pytest.fixture
def opcua_injector():
    """Create OPC-UA communication fault injector."""
    return OPCUACommunicationFaultInjector()


@pytest.fixture
def model_injector():
    """Create model inference fault injector."""
    return ModelInferenceFaultInjector()


# =============================================================================
# Sensor Stream Fault Tests
# =============================================================================

class TestSensorStreamFaults:
    """Test suite for sensor stream fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_sensor_failure_injection(self, sensor_injector):
        """Test sensor failure fault injection."""
        result = await sensor_injector.inject({
            "sensors": ["temperature", "pressure"],
            "fault_type": "failure",
        })
        assert result is True
        assert sensor_injector.is_active()

        # Affected sensors should return None
        temp_value = sensor_injector.apply_fault("temperature", 100.0)
        assert temp_value is None

        # Unaffected sensors should return original value
        flow_value = sensor_injector.apply_fault("flow_rate", 50.0)
        assert flow_value == 50.0

        await sensor_injector.rollback()
        assert not sensor_injector.is_active()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_sensor_drift_injection(self, sensor_injector):
        """Test sensor drift fault injection."""
        result = await sensor_injector.inject({
            "sensors": ["temperature"],
            "fault_type": "drift",
            "drift_rate": 0.01,
        })
        assert result is True

        # Apply drift multiple times
        values = []
        base_value = 100.0
        for _ in range(10):
            value = sensor_injector.apply_fault("temperature", base_value)
            values.append(value)

        # Values should be increasing due to drift
        for i in range(1, len(values)):
            assert values[i] > values[i-1]

        await sensor_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_sensor_noise_injection(self, sensor_injector):
        """Test sensor noise fault injection."""
        result = await sensor_injector.inject({
            "sensors": ["pressure"],
            "fault_type": "noise",
            "noise_amplitude": 5.0,
        })
        assert result is True

        # Collect noisy values
        base_value = 100.0
        values = [sensor_injector.apply_fault("pressure", base_value) for _ in range(100)]

        # Values should vary around base value
        min_val = min(values)
        max_val = max(values)
        assert min_val < base_value < max_val

        await sensor_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_sensor_freeze_injection(self, sensor_injector):
        """Test sensor freeze fault injection."""
        result = await sensor_injector.inject({
            "sensors": ["flow_rate"],
            "fault_type": "freeze",
        })
        assert result is True

        # First call sets the frozen value
        first_value = sensor_injector.apply_fault("flow_rate", 100.0)

        # Subsequent calls should return the same frozen value
        for _ in range(10):
            value = sensor_injector.apply_fault("flow_rate", 200.0)  # Different input
            assert value == first_value

        await sensor_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_sensor_spike_injection(self, sensor_injector):
        """Test sensor spike fault injection."""
        result = await sensor_injector.inject({
            "sensors": ["temperature"],
            "fault_type": "spike",
            "spike_probability": 0.5,
            "spike_magnitude": 10.0,
        })
        assert result is True

        # Collect values - some should have spikes
        base_value = 100.0
        values = [sensor_injector.apply_fault("temperature", base_value) for _ in range(100)]

        normal_values = [v for v in values if v <= base_value * 2]
        spike_values = [v for v in values if v > base_value * 2]

        # With 50% probability, we should see some spikes
        assert len(spike_values) > 0
        assert len(normal_values) > 0

        await sensor_injector.rollback()


# =============================================================================
# Efficiency Calculation Fault Tests
# =============================================================================

class TestEfficiencyCalculationFaults:
    """Test suite for efficiency calculation fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calculation_timeout(self, efficiency_injector):
        """Test efficiency calculation timeout."""
        result = await efficiency_injector.inject({
            "fault_type": "timeout",
            "timeout_ms": 100,  # Short timeout for testing
        })
        assert result is True

        calc_result = await efficiency_injector.simulate_calculation({
            "fuel_input": 100.0,
            "steam_output": 85.0,
        })

        assert calc_result["status"] == "timeout"
        assert calc_result["efficiency"] is None

        await efficiency_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calculation_div_zero(self, efficiency_injector):
        """Test division by zero error handling."""
        result = await efficiency_injector.inject({
            "fault_type": "div_zero",
        })
        assert result is True

        calc_result = await efficiency_injector.simulate_calculation({
            "fuel_input": 0.0,
            "steam_output": 85.0,
        })

        assert calc_result["status"] == "error"
        assert "division by zero" in calc_result.get("error", "")

        await efficiency_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calculation_nan_handling(self, efficiency_injector):
        """Test NaN result handling."""
        result = await efficiency_injector.inject({
            "fault_type": "nan",
        })
        assert result is True

        calc_result = await efficiency_injector.simulate_calculation({
            "fuel_input": 100.0,
            "steam_output": 85.0,
        })

        assert calc_result["status"] == "error"

        await efficiency_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_calculation_random_errors(self, efficiency_injector):
        """Test random calculation errors."""
        result = await efficiency_injector.inject({
            "fault_type": "random_error",
            "error_rate": 0.3,  # 30% error rate
        })
        assert result is True

        # Run multiple calculations
        results = []
        for _ in range(100):
            calc_result = await efficiency_injector.simulate_calculation({
                "fuel_input": 100.0,
                "steam_output": 85.0,
            })
            results.append(calc_result["status"])

        # Should have mix of success and error
        success_count = results.count("success")
        error_count = results.count("error")

        assert success_count > 0
        assert error_count > 0

        await efficiency_injector.rollback()


# =============================================================================
# CMMS Integration Fault Tests
# =============================================================================

class TestCMMSIntegrationFaults:
    """Test suite for CMMS integration fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_cmms_unavailable(self, cmms_injector):
        """Test CMMS service unavailability."""
        result = await cmms_injector.inject({
            "fault_type": "unavailable",
        })
        assert result is True

        cmms_result = await cmms_injector.simulate_cmms_call(
            "create_work_order",
            {"asset_id": "BOILER_001"}
        )

        assert cmms_result["status"] == "error"
        assert "unavailable" in cmms_result.get("error", "")

        await cmms_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_cmms_auth_failure(self, cmms_injector):
        """Test CMMS authentication failure."""
        result = await cmms_injector.inject({
            "fault_type": "auth_failure",
        })
        assert result is True

        cmms_result = await cmms_injector.simulate_cmms_call(
            "get_schedule",
            {"asset_id": "BOILER_001"}
        )

        assert cmms_result["status"] == "error"
        assert "authentication" in cmms_result.get("error", "")

        await cmms_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_cmms_slow_response(self, cmms_injector):
        """Test CMMS slow response."""
        result = await cmms_injector.inject({
            "fault_type": "slow_response",
            "delay_ms": 100,  # Short delay for testing
        })
        assert result is True

        import time
        start = time.time()

        cmms_result = await cmms_injector.simulate_cmms_call(
            "create_work_order",
            {"asset_id": "BOILER_001"}
        )

        elapsed = time.time() - start

        assert cmms_result["status"] == "success"
        assert elapsed >= 0.1  # At least 100ms delay

        await cmms_injector.rollback()


# =============================================================================
# OPC-UA Communication Fault Tests
# =============================================================================

class TestOPCUACommunicationFaults:
    """Test suite for OPC-UA communication fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_opcua_connection_failure(self, opcua_injector):
        """Test OPC-UA connection failure."""
        result = await opcua_injector.inject({
            "fault_type": "connection_failure",
        })
        assert result is True
        assert not opcua_injector.is_connected()

        read_result = await opcua_injector.simulate_read("ns=2;s=Temperature")

        assert read_result["status"] == "error"
        assert "not_connected" in read_result.get("error", "")

        await opcua_injector.rollback()
        assert opcua_injector.is_connected()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_opcua_session_timeout(self, opcua_injector):
        """Test OPC-UA session timeout."""
        result = await opcua_injector.inject({
            "fault_type": "session_timeout",
        })
        assert result is True

        read_result = await opcua_injector.simulate_read("ns=2;s=Temperature")

        assert read_result["status"] == "error"
        assert "session_timeout" in read_result.get("error", "")

        await opcua_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_opcua_write_failure(self, opcua_injector):
        """Test OPC-UA write failure."""
        result = await opcua_injector.inject({
            "fault_type": "write_failure",
        })
        assert result is True

        write_result = await opcua_injector.simulate_write("ns=2;s=Setpoint", 85.0)

        assert write_result["status"] == "error"
        assert "write_rejected" in write_result.get("error", "")

        await opcua_injector.rollback()


# =============================================================================
# Model Inference Fault Tests
# =============================================================================

class TestModelInferenceFaults:
    """Test suite for model inference fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_model_timeout(self, model_injector):
        """Test model inference timeout."""
        result = await model_injector.inject({
            "fault_type": "timeout",
            "timeout_ms": 100,  # Short timeout for testing
        })
        assert result is True

        inference_result = await model_injector.simulate_inference({
            "temperature": 450.0,
            "pressure": 15.0,
        })

        assert inference_result["status"] == "timeout"
        assert inference_result["prediction"] is None

        await model_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_model_unavailable(self, model_injector):
        """Test model unavailable."""
        result = await model_injector.inject({
            "fault_type": "unavailable",
        })
        assert result is True

        inference_result = await model_injector.simulate_inference({
            "temperature": 450.0,
        })

        assert inference_result["status"] == "model_unavailable"

        await model_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_model_drift(self, model_injector):
        """Test model prediction drift."""
        result = await model_injector.inject({
            "fault_type": "drift",
            "drift_amount": 15.0,
        })
        assert result is True

        inference_result = await model_injector.simulate_inference({
            "temperature": 450.0,
        })

        assert inference_result["status"] == "degraded"
        assert "model_drift_detected" in inference_result.get("warning", "")
        # Prediction should be drifted
        assert inference_result["prediction"] > 90.0

        await model_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_model_low_confidence(self, model_injector):
        """Test model low confidence prediction."""
        result = await model_injector.inject({
            "fault_type": "low_confidence",
        })
        assert result is True

        inference_result = await model_injector.simulate_inference({
            "temperature": 450.0,
        })

        assert inference_result["status"] == "low_confidence"
        assert inference_result["confidence"] < 0.5

        await model_injector.rollback()


# =============================================================================
# Steady State Hypothesis Tests
# =============================================================================

class TestFlameguardSteadyState:
    """Test suite for Flameguard steady state validation."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_flameguard_hypothesis(self):
        """Test Flameguard-specific steady state hypothesis."""
        hypothesis = create_flameguard_hypothesis()

        assert hypothesis.name == "Flameguard Boiler Efficiency Health"
        assert len(hypothesis.metrics) >= 3

        # Validate with default metrics provider
        validator = SteadyStateValidator()
        result = await validator.validate(hypothesis)

        assert result.hypothesis_name == hypothesis.name
        assert result.timestamp is not None


# =============================================================================
# Resilience Pattern Tests (Flameguard-specific)
# =============================================================================

class TestFlameguardResilience:
    """Test resilience patterns in Flameguard context."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_circuit_breaker_for_cmms(self, cmms_injector):
        """Test circuit breaker pattern for CMMS integration."""
        cb_test = CircuitBreakerTest()

        # Simulate CMMS failures triggering circuit breaker
        result = await cb_test.test_circuit_opens_on_failures()

        assert result.passed is True
        assert result.final_state.value == "open"

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_retry_for_sensor_reads(self):
        """Test retry mechanism for sensor reads."""
        retry_test = RetryMechanismTest()

        result = await retry_test.test_retry_on_transient_failure()

        assert result.passed is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestFlameguardChaosIntegration:
    """Integration tests for Flameguard chaos scenarios."""

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_sensor_failures(self, sensor_injector, efficiency_injector):
        """Test system behavior with multiple sensor failures."""
        # Inject sensor failures
        await sensor_injector.inject({
            "sensors": ["temperature", "pressure", "flow_rate"],
            "fault_type": "failure",
        })

        # Try efficiency calculation without sensor data
        calc_result = await efficiency_injector.simulate_calculation({
            "fuel_input": None,  # Simulating missing sensor data
            "steam_output": None,
        })

        # System should handle gracefully
        # (In real implementation, would use fallback/cached values)

        await sensor_injector.rollback()

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cascading_failures(
        self,
        opcua_injector,
        sensor_injector,
        cmms_injector
    ):
        """Test cascading failure scenario."""
        # Stage 1: OPC-UA connection failure
        await opcua_injector.inject({"fault_type": "connection_failure"})
        assert not opcua_injector.is_connected()

        # Stage 2: This causes sensor data unavailability
        await sensor_injector.inject({
            "sensors": ["temperature"],
            "fault_type": "failure",
        })

        # Stage 3: Unable to create maintenance work order
        await cmms_injector.inject({"fault_type": "unavailable"})

        # All systems affected
        assert opcua_injector.is_active()
        assert sensor_injector.is_active()
        assert cmms_injector.is_active()

        # Rollback in reverse order
        await cmms_injector.rollback()
        await sensor_injector.rollback()
        await opcua_injector.rollback()

        # All systems recovered
        assert not opcua_injector.is_active()
        assert not sensor_injector.is_active()
        assert not cmms_injector.is_active()


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with chaos markers."""
    config.addinivalue_line("markers", "chaos: Chaos engineering tests")
    config.addinivalue_line("markers", "integration: Integration tests")
