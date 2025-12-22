"""
GL-002 Flameguard - Agent-Specific Chaos Engineering Components

This module provides chaos engineering components specific to the
Flameguard Boiler Efficiency Agent, including:

- Sensor stream fault injection
- Efficiency calculation faults
- CMMS integration failures
- Model inference delays
- Historical data corruption

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FlameguardChaosConfig:
    """Configuration for Flameguard-specific chaos tests."""

    # Sensor fault parameters
    sensor_failure_percent: float = 10.0
    sensor_drift_max_percent: float = 5.0
    sensor_noise_amplitude: float = 2.0

    # Calculation fault parameters
    calculation_timeout_ms: float = 5000.0
    calculation_error_rate: float = 5.0

    # Integration fault parameters
    cmms_unavailable_percent: float = 10.0
    opcua_latency_ms: float = 100.0

    # Model fault parameters
    model_inference_timeout_ms: float = 1000.0
    model_drift_threshold: float = 0.1


class SensorType(Enum):
    """Types of sensors in boiler systems."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    FUEL_RATE = "fuel_rate"
    FLUE_GAS_O2 = "flue_gas_o2"
    FLUE_GAS_CO2 = "flue_gas_co2"
    STACK_TEMPERATURE = "stack_temperature"


# =============================================================================
# Sensor Stream Fault Injector
# =============================================================================

class SensorStreamFaultInjector:
    """
    Inject faults into simulated sensor data streams.

    Fault types:
    - Sensor failure (no data)
    - Sensor drift (gradual deviation)
    - Sensor noise (random fluctuations)
    - Sensor freeze (stuck value)
    - Sensor spike (sudden extreme value)

    Example:
        >>> injector = SensorStreamFaultInjector()
        >>> await injector.inject({
        ...     "sensors": ["temperature", "pressure"],
        ...     "fault_type": "drift",
        ...     "drift_rate": 0.01
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._affected_sensors: Set[str] = set()
        self._params: Dict[str, Any] = {}
        self._original_values: Dict[str, float] = {}
        self._drift_accumulator: Dict[str, float] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject sensor stream faults."""
        try:
            self._fault_type = params.get("fault_type", "failure")
            self._affected_sensors = set(params.get("sensors", []))
            self._params = params

            # Initialize drift accumulators
            for sensor in self._affected_sensors:
                self._drift_accumulator[sensor] = 0.0

            logger.info(
                f"SensorStreamFaultInjector: Injecting {self._fault_type} fault "
                f"on sensors: {self._affected_sensors}"
            )

            self._active = True
            return True

        except Exception as e:
            logger.error(f"SensorStreamFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove sensor stream faults."""
        try:
            logger.info("SensorStreamFaultInjector: Rolling back sensor faults")
            self._active = False
            self._affected_sensors.clear()
            self._drift_accumulator.clear()
            return True

        except Exception as e:
            logger.error(f"SensorStreamFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        """Check if fault injection is active."""
        return self._active

    def apply_fault(self, sensor: str, value: float) -> Optional[float]:
        """
        Apply fault to sensor reading.

        Args:
            sensor: Sensor identifier
            value: Original sensor value

        Returns:
            Modified value or None if sensor failed
        """
        if not self._active or sensor not in self._affected_sensors:
            return value

        if self._fault_type == "failure":
            return None

        elif self._fault_type == "drift":
            drift_rate = self._params.get("drift_rate", 0.01)
            self._drift_accumulator[sensor] += drift_rate * value
            return value + self._drift_accumulator[sensor]

        elif self._fault_type == "noise":
            noise_amplitude = self._params.get("noise_amplitude", 2.0)
            noise = random.uniform(-noise_amplitude, noise_amplitude)
            return value + noise

        elif self._fault_type == "freeze":
            if sensor not in self._original_values:
                self._original_values[sensor] = value
            return self._original_values[sensor]

        elif self._fault_type == "spike":
            spike_probability = self._params.get("spike_probability", 0.1)
            spike_magnitude = self._params.get("spike_magnitude", 10.0)
            if random.random() < spike_probability:
                return value * spike_magnitude
            return value

        return value


# =============================================================================
# Efficiency Calculation Fault Injector
# =============================================================================

class EfficiencyCalculationFaultInjector:
    """
    Inject faults into efficiency calculation process.

    Fault types:
    - Calculation timeout
    - Division by zero scenarios
    - NaN/Inf propagation
    - Precision loss
    - Formula errors

    Example:
        >>> injector = EfficiencyCalculationFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "timeout",
        ...     "timeout_ms": 5000
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject efficiency calculation faults."""
        try:
            self._fault_type = params.get("fault_type", "timeout")
            self._params = params

            logger.info(
                f"EfficiencyCalculationFaultInjector: Injecting {self._fault_type} fault"
            )

            self._active = True
            return True

        except Exception as e:
            logger.error(f"EfficiencyCalculationFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove efficiency calculation faults."""
        try:
            logger.info("EfficiencyCalculationFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"EfficiencyCalculationFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        """Check if fault injection is active."""
        return self._active

    async def simulate_calculation(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate efficiency calculation with potential faults.

        Args:
            input_data: Input parameters for calculation

        Returns:
            Calculation result with fault effects
        """
        if not self._active:
            # Normal calculation
            efficiency = self._calculate_efficiency(input_data)
            return {"efficiency": efficiency, "status": "success"}

        if self._fault_type == "timeout":
            timeout_ms = self._params.get("timeout_ms", 5000)
            await asyncio.sleep(timeout_ms / 1000.0)
            return {"efficiency": None, "status": "timeout"}

        elif self._fault_type == "div_zero":
            return {"efficiency": float('inf'), "status": "error", "error": "division by zero"}

        elif self._fault_type == "nan":
            return {"efficiency": float('nan'), "status": "error", "error": "nan result"}

        elif self._fault_type == "precision_loss":
            efficiency = self._calculate_efficiency(input_data)
            # Simulate precision loss by rounding aggressively
            efficiency = round(efficiency, 1)
            return {"efficiency": efficiency, "status": "degraded"}

        elif self._fault_type == "random_error":
            if random.random() < self._params.get("error_rate", 0.1):
                return {"efficiency": None, "status": "error", "error": "random failure"}
            efficiency = self._calculate_efficiency(input_data)
            return {"efficiency": efficiency, "status": "success"}

        return {"efficiency": None, "status": "unknown_fault"}

    def _calculate_efficiency(self, input_data: Dict[str, float]) -> float:
        """Calculate boiler efficiency (simplified)."""
        # Simplified efficiency calculation for testing
        fuel_input = input_data.get("fuel_input", 100.0)
        steam_output = input_data.get("steam_output", 80.0)

        if fuel_input <= 0:
            return 0.0

        efficiency = (steam_output / fuel_input) * 100.0
        return min(max(efficiency, 0.0), 100.0)


# =============================================================================
# CMMS Integration Fault Injector
# =============================================================================

class CMMSIntegrationFaultInjector:
    """
    Inject faults into CMMS (Computerized Maintenance Management System) integration.

    Fault types:
    - Service unavailable
    - Authentication failure
    - Slow response
    - Data corruption
    - Partial data loss

    Example:
        >>> injector = CMMSIntegrationFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "unavailable",
        ...     "duration_seconds": 30
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject CMMS integration faults."""
        try:
            self._fault_type = params.get("fault_type", "unavailable")
            self._params = params

            logger.info(f"CMMSIntegrationFaultInjector: Injecting {self._fault_type} fault")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"CMMSIntegrationFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove CMMS integration faults."""
        try:
            logger.info("CMMSIntegrationFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"CMMSIntegrationFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        """Check if fault injection is active."""
        return self._active

    async def simulate_cmms_call(self, operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate CMMS API call with potential faults.

        Args:
            operation: CMMS operation (create_work_order, get_schedule, etc.)
            data: Operation data

        Returns:
            Simulated response
        """
        if not self._active:
            return {"status": "success", "data": {"work_order_id": "WO-12345"}}

        if self._fault_type == "unavailable":
            return {"status": "error", "error": "service_unavailable"}

        elif self._fault_type == "auth_failure":
            return {"status": "error", "error": "authentication_failed"}

        elif self._fault_type == "slow_response":
            delay_ms = self._params.get("delay_ms", 5000)
            await asyncio.sleep(delay_ms / 1000.0)
            return {"status": "success", "data": {"work_order_id": "WO-12345"}}

        elif self._fault_type == "data_corruption":
            return {"status": "success", "data": {"work_order_id": None, "corrupted": True}}

        elif self._fault_type == "partial_loss":
            if random.random() < 0.5:
                return {"status": "error", "error": "partial_data_loss"}
            return {"status": "success", "data": {"work_order_id": "WO-12345"}}

        return {"status": "error", "error": "unknown_fault"}


# =============================================================================
# OPC-UA Communication Fault Injector
# =============================================================================

class OPCUACommunicationFaultInjector:
    """
    Inject faults into OPC-UA communication.

    Fault types:
    - Connection failure
    - Session timeout
    - Subscription loss
    - Write failure
    - Security rejection

    Example:
        >>> injector = OPCUACommunicationFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "connection_failure",
        ...     "reconnect_delay_ms": 5000
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._connection_state = "connected"

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject OPC-UA communication faults."""
        try:
            self._fault_type = params.get("fault_type", "connection_failure")
            self._params = params

            if self._fault_type == "connection_failure":
                self._connection_state = "disconnected"

            logger.info(f"OPCUACommunicationFaultInjector: Injecting {self._fault_type} fault")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"OPCUACommunicationFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove OPC-UA communication faults."""
        try:
            logger.info("OPCUACommunicationFaultInjector: Rolling back")
            self._active = False
            self._connection_state = "connected"
            return True

        except Exception as e:
            logger.error(f"OPCUACommunicationFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        """Check if fault injection is active."""
        return self._active

    def is_connected(self) -> bool:
        """Check simulated connection state."""
        return self._connection_state == "connected"

    async def simulate_read(self, node_id: str) -> Dict[str, Any]:
        """Simulate OPC-UA read operation."""
        if not self._active:
            return {"status": "success", "value": random.uniform(0, 100), "quality": "Good"}

        if self._fault_type == "connection_failure":
            return {"status": "error", "error": "not_connected"}

        elif self._fault_type == "session_timeout":
            return {"status": "error", "error": "session_timeout"}

        elif self._fault_type == "subscription_loss":
            return {"status": "error", "error": "subscription_not_found"}

        return {"status": "error", "error": "unknown_fault"}

    async def simulate_write(self, node_id: str, value: Any) -> Dict[str, Any]:
        """Simulate OPC-UA write operation."""
        if not self._active:
            return {"status": "success"}

        if self._fault_type == "write_failure":
            return {"status": "error", "error": "write_rejected"}

        elif self._fault_type == "security_rejection":
            return {"status": "error", "error": "access_denied"}

        return {"status": "error", "error": "unknown_fault"}


# =============================================================================
# Model Inference Fault Injector
# =============================================================================

class ModelInferenceFaultInjector:
    """
    Inject faults into ML model inference.

    Fault types:
    - Inference timeout
    - Model unavailable
    - Prediction drift
    - Low confidence
    - Feature missing

    Example:
        >>> injector = ModelInferenceFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "timeout",
        ...     "timeout_ms": 1000
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject model inference faults."""
        try:
            self._fault_type = params.get("fault_type", "timeout")
            self._params = params

            logger.info(f"ModelInferenceFaultInjector: Injecting {self._fault_type} fault")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"ModelInferenceFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove model inference faults."""
        try:
            logger.info("ModelInferenceFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"ModelInferenceFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        """Check if fault injection is active."""
        return self._active

    async def simulate_inference(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Simulate model inference with potential faults."""
        if not self._active:
            return {
                "prediction": 85.0 + random.uniform(-5, 5),
                "confidence": 0.95,
                "status": "success"
            }

        if self._fault_type == "timeout":
            timeout_ms = self._params.get("timeout_ms", 1000)
            await asyncio.sleep(timeout_ms / 1000.0)
            return {"prediction": None, "status": "timeout"}

        elif self._fault_type == "unavailable":
            return {"prediction": None, "status": "model_unavailable"}

        elif self._fault_type == "drift":
            drift_amount = self._params.get("drift_amount", 10.0)
            return {
                "prediction": 85.0 + drift_amount,
                "confidence": 0.7,
                "status": "degraded",
                "warning": "model_drift_detected"
            }

        elif self._fault_type == "low_confidence":
            return {
                "prediction": 85.0,
                "confidence": 0.3,
                "status": "low_confidence"
            }

        elif self._fault_type == "feature_missing":
            return {"prediction": None, "status": "error", "error": "missing_features"}

        return {"prediction": None, "status": "unknown_fault"}


# =============================================================================
# Steady State Hypothesis for Flameguard
# =============================================================================

def create_flameguard_hypothesis():
    """Create steady state hypothesis specific to Flameguard agent."""
    import sys
    import os

    # Import from GL-001 chaos framework
    gl001_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
    if gl001_path not in sys.path:
        sys.path.insert(0, gl001_path)

    from steady_state import SteadyStateHypothesis, SteadyStateMetric, ComparisonOperator

    return SteadyStateHypothesis(
        name="Flameguard Boiler Efficiency Health",
        description="Validates Flameguard agent is operating normally",
        metrics=[
            SteadyStateMetric(
                name="response_time_ms",
                description="Efficiency calculation response time",
                threshold=100,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="error_rate_percent",
                description="Calculation error rate",
                threshold=1.0,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="sensor_availability_percent",
                description="Sensor data availability",
                threshold=99.0,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=True,
            ),
            SteadyStateMetric(
                name="efficiency_variance",
                description="Efficiency calculation variance",
                threshold=5.0,
                operator=ComparisonOperator.LESS_THAN,
                required=False,
            ),
            SteadyStateMetric(
                name="cmms_response_time_ms",
                description="CMMS integration response time",
                threshold=500,
                operator=ComparisonOperator.LESS_THAN,
                required=False,
            ),
        ],
        pass_threshold=0.8,
        aggregation="weighted",
    )
