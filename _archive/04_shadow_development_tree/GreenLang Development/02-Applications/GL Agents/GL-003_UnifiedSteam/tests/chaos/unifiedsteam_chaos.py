"""
GL-003 UnifiedSteam - Agent-Specific Chaos Engineering Components

This module provides chaos engineering components specific to the
UnifiedSteam Header Balance Agent, including:

- Steam header pressure imbalance simulation
- Multi-boiler coordination failures
- Load balancing disruptions
- Demand forecasting errors
- Valve actuator failures

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnifiedSteamChaosConfig:
    """Configuration for UnifiedSteam-specific chaos tests."""

    # Header pressure parameters
    pressure_deviation_max_psi: float = 10.0
    pressure_oscillation_frequency_hz: float = 0.1

    # Coordination parameters
    coordination_timeout_ms: float = 5000.0
    max_boiler_communication_delay_ms: float = 500.0

    # Load balancing parameters
    load_imbalance_max_percent: float = 20.0

    # Forecast parameters
    forecast_error_max_percent: float = 15.0

    # Actuator parameters
    valve_response_delay_ms: float = 200.0
    valve_failure_probability: float = 0.05


class HeaderPressureState(Enum):
    """Steam header pressure states."""
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    CRITICAL_HIGH = "critical_high"
    CRITICAL_LOW = "critical_low"
    OSCILLATING = "oscillating"


class BoilerState(Enum):
    """Boiler operational states."""
    RUNNING = "running"
    STANDBY = "standby"
    RAMPING_UP = "ramping_up"
    RAMPING_DOWN = "ramping_down"
    TRIPPED = "tripped"
    OFFLINE = "offline"


# =============================================================================
# Steam Header Fault Injector
# =============================================================================

class SteamHeaderFaultInjector:
    """
    Inject faults into steam header pressure management.

    Fault types:
    - Pressure spike (sudden increase)
    - Pressure drop (sudden decrease)
    - Pressure oscillation
    - Sensor failure
    - Relief valve malfunction

    Example:
        >>> injector = SteamHeaderFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "pressure_spike",
        ...     "magnitude_psi": 15.0
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._base_pressure = 150.0  # PSI
        self._current_pressure = 150.0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject steam header fault."""
        try:
            self._fault_type = params.get("fault_type", "pressure_spike")
            self._params = params

            logger.info(f"SteamHeaderFaultInjector: Injecting {self._fault_type} fault")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"SteamHeaderFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove steam header fault."""
        try:
            logger.info("SteamHeaderFaultInjector: Rolling back")
            self._active = False
            self._current_pressure = self._base_pressure
            return True

        except Exception as e:
            logger.error(f"SteamHeaderFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_pressure(self) -> float:
        """Get current simulated pressure with fault effects."""
        if not self._active:
            return self._base_pressure + random.uniform(-1, 1)

        if self._fault_type == "pressure_spike":
            magnitude = self._params.get("magnitude_psi", 15.0)
            return self._base_pressure + magnitude

        elif self._fault_type == "pressure_drop":
            magnitude = self._params.get("magnitude_psi", 20.0)
            return max(0, self._base_pressure - magnitude)

        elif self._fault_type == "oscillation":
            amplitude = self._params.get("amplitude_psi", 10.0)
            frequency = self._params.get("frequency_hz", 0.5)
            import math
            oscillation = amplitude * math.sin(2 * math.pi * frequency * time.time())
            return self._base_pressure + oscillation

        elif self._fault_type == "sensor_failure":
            return None  # No reading available

        elif self._fault_type == "stuck_valve":
            # Pressure builds up due to stuck relief valve
            return self._base_pressure * 1.3

        return self._base_pressure

    def get_state(self) -> HeaderPressureState:
        """Get current header pressure state."""
        pressure = self.get_pressure()

        if pressure is None:
            return HeaderPressureState.NORMAL  # Sensor failure - unknown state

        if pressure > 180:
            return HeaderPressureState.CRITICAL_HIGH
        elif pressure > 160:
            return HeaderPressureState.HIGH
        elif pressure < 100:
            return HeaderPressureState.CRITICAL_LOW
        elif pressure < 130:
            return HeaderPressureState.LOW
        elif self._fault_type == "oscillation":
            return HeaderPressureState.OSCILLATING
        else:
            return HeaderPressureState.NORMAL


# =============================================================================
# Boiler Coordination Fault Injector
# =============================================================================

class BoilerCoordinationFaultInjector:
    """
    Inject faults into multi-boiler coordination.

    Fault types:
    - Communication delay
    - Boiler offline (unexpected)
    - Coordination timeout
    - Split brain (conflicting commands)
    - Sequence error

    Example:
        >>> injector = BoilerCoordinationFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "communication_delay",
        ...     "delay_ms": 500
        ... })
    """

    def __init__(self, num_boilers: int = 4):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._boilers: Dict[str, BoilerState] = {
            f"boiler_{i+1}": BoilerState.RUNNING
            for i in range(num_boilers)
        }
        self._communication_delays: Dict[str, float] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject boiler coordination fault."""
        try:
            self._fault_type = params.get("fault_type", "communication_delay")
            self._params = params

            if self._fault_type == "boiler_offline":
                boiler = params.get("boiler", "boiler_1")
                self._boilers[boiler] = BoilerState.OFFLINE

            elif self._fault_type == "communication_delay":
                delay_ms = params.get("delay_ms", 500)
                for boiler in params.get("boilers", self._boilers.keys()):
                    self._communication_delays[boiler] = delay_ms

            logger.info(f"BoilerCoordinationFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"BoilerCoordinationFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove boiler coordination fault."""
        try:
            logger.info("BoilerCoordinationFaultInjector: Rolling back")
            self._active = False
            # Restore all boilers to running
            for boiler in self._boilers:
                self._boilers[boiler] = BoilerState.RUNNING
            self._communication_delays.clear()
            return True

        except Exception as e:
            logger.error(f"BoilerCoordinationFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_boiler_state(self, boiler: str) -> BoilerState:
        """Get state of a specific boiler."""
        return self._boilers.get(boiler, BoilerState.OFFLINE)

    async def send_command(self, boiler: str, command: str) -> Dict[str, Any]:
        """Simulate sending command to boiler with potential faults."""
        if not self._active:
            return {"status": "success", "boiler": boiler, "command": command}

        if self._fault_type == "communication_delay":
            delay = self._communication_delays.get(boiler, 0) / 1000.0
            await asyncio.sleep(delay)
            return {"status": "success", "boiler": boiler, "delayed": True}

        elif self._fault_type == "timeout":
            if random.random() < 0.5:
                return {"status": "timeout", "boiler": boiler}
            return {"status": "success", "boiler": boiler}

        elif self._fault_type == "split_brain":
            # Different boilers receive conflicting commands
            if random.random() < 0.3:
                return {"status": "conflict", "boiler": boiler, "error": "conflicting_command"}
            return {"status": "success", "boiler": boiler}

        return {"status": "error", "boiler": boiler}


# =============================================================================
# Load Balancing Fault Injector
# =============================================================================

class LoadBalancingFaultInjector:
    """
    Inject faults into load balancing logic.

    Fault types:
    - Uneven distribution
    - Overloaded boiler
    - Underutilized capacity
    - Algorithm failure
    - Demand spike

    Example:
        >>> injector = LoadBalancingFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "uneven_distribution",
        ...     "imbalance_percent": 30
        ... })
    """

    def __init__(self, num_boilers: int = 4):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._num_boilers = num_boilers
        self._load_distribution: Dict[str, float] = {
            f"boiler_{i+1}": 25.0 for i in range(num_boilers)  # Even 25% each
        }

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject load balancing fault."""
        try:
            self._fault_type = params.get("fault_type", "uneven_distribution")
            self._params = params

            if self._fault_type == "uneven_distribution":
                imbalance = params.get("imbalance_percent", 30)
                # Shift load to first boiler
                self._load_distribution["boiler_1"] = 25.0 + imbalance
                remaining = 75.0 - imbalance
                for i in range(1, self._num_boilers):
                    self._load_distribution[f"boiler_{i+1}"] = remaining / (self._num_boilers - 1)

            elif self._fault_type == "overloaded":
                boiler = params.get("boiler", "boiler_1")
                self._load_distribution[boiler] = 100.0  # 100% load on one

            logger.info(f"LoadBalancingFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"LoadBalancingFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove load balancing fault."""
        try:
            logger.info("LoadBalancingFaultInjector: Rolling back")
            self._active = False
            # Restore even distribution
            for i in range(self._num_boilers):
                self._load_distribution[f"boiler_{i+1}"] = 100.0 / self._num_boilers
            return True

        except Exception as e:
            logger.error(f"LoadBalancingFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution."""
        return self._load_distribution.copy()

    def is_balanced(self, threshold: float = 10.0) -> bool:
        """Check if load is balanced within threshold."""
        loads = list(self._load_distribution.values())
        return max(loads) - min(loads) <= threshold


# =============================================================================
# Demand Forecast Fault Injector
# =============================================================================

class DemandForecastFaultInjector:
    """
    Inject faults into demand forecasting.

    Fault types:
    - Over-prediction
    - Under-prediction
    - Delayed forecast
    - Missing data
    - Model drift

    Example:
        >>> injector = DemandForecastFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "over_prediction",
        ...     "error_percent": 25
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._base_demand = 1000.0  # Base demand in units

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject demand forecast fault."""
        try:
            self._fault_type = params.get("fault_type", "over_prediction")
            self._params = params

            logger.info(f"DemandForecastFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"DemandForecastFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove demand forecast fault."""
        try:
            logger.info("DemandForecastFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"DemandForecastFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    async def get_forecast(self, actual_demand: float) -> Dict[str, Any]:
        """Get demand forecast with potential faults."""
        if not self._active:
            # Normal forecast with small error
            error = random.uniform(-5, 5)
            forecast = actual_demand * (1 + error / 100)
            return {"forecast": forecast, "confidence": 0.95, "status": "success"}

        if self._fault_type == "over_prediction":
            error_percent = self._params.get("error_percent", 25)
            forecast = actual_demand * (1 + error_percent / 100)
            return {"forecast": forecast, "confidence": 0.7, "status": "degraded"}

        elif self._fault_type == "under_prediction":
            error_percent = self._params.get("error_percent", 25)
            forecast = actual_demand * (1 - error_percent / 100)
            return {"forecast": forecast, "confidence": 0.7, "status": "degraded"}

        elif self._fault_type == "delayed":
            delay_ms = self._params.get("delay_ms", 5000)
            await asyncio.sleep(delay_ms / 1000.0)
            return {"forecast": actual_demand, "confidence": 0.5, "status": "stale"}

        elif self._fault_type == "missing_data":
            return {"forecast": None, "confidence": 0, "status": "error"}

        elif self._fault_type == "model_drift":
            # Systematic error that grows over time
            drift = self._params.get("drift_amount", 10)
            forecast = actual_demand + drift
            return {"forecast": forecast, "confidence": 0.6, "status": "drift_detected"}

        return {"forecast": None, "status": "unknown_fault"}


# =============================================================================
# Valve Actuator Fault Injector
# =============================================================================

class ValveActuatorFaultInjector:
    """
    Inject faults into valve actuators.

    Fault types:
    - Stuck valve
    - Slow response
    - Partial movement
    - Reverse action
    - Position feedback error

    Example:
        >>> injector = ValveActuatorFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "stuck",
        ...     "valves": ["steam_header_1", "bypass_valve"]
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._valve_positions: Dict[str, float] = {}
        self._stuck_positions: Dict[str, float] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject valve actuator fault."""
        try:
            self._fault_type = params.get("fault_type", "stuck")
            self._params = params

            valves = params.get("valves", [])
            for valve in valves:
                if valve not in self._valve_positions:
                    self._valve_positions[valve] = 50.0  # 50% open

                if self._fault_type == "stuck":
                    self._stuck_positions[valve] = self._valve_positions[valve]

            logger.info(f"ValveActuatorFaultInjector: Injecting {self._fault_type} on {valves}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"ValveActuatorFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove valve actuator fault."""
        try:
            logger.info("ValveActuatorFaultInjector: Rolling back")
            self._active = False
            self._stuck_positions.clear()
            return True

        except Exception as e:
            logger.error(f"ValveActuatorFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    async def set_position(self, valve: str, target: float) -> Dict[str, Any]:
        """Set valve position with potential faults."""
        if valve not in self._valve_positions:
            self._valve_positions[valve] = 50.0

        if not self._active:
            self._valve_positions[valve] = target
            return {"valve": valve, "position": target, "status": "success"}

        if self._fault_type == "stuck" and valve in self._stuck_positions:
            return {
                "valve": valve,
                "position": self._stuck_positions[valve],
                "status": "stuck",
                "target": target
            }

        elif self._fault_type == "slow_response":
            delay_ms = self._params.get("delay_ms", 2000)
            await asyncio.sleep(delay_ms / 1000.0)
            self._valve_positions[valve] = target
            return {"valve": valve, "position": target, "status": "delayed"}

        elif self._fault_type == "partial":
            # Only moves partway
            current = self._valve_positions[valve]
            partial_move = current + (target - current) * 0.5
            self._valve_positions[valve] = partial_move
            return {"valve": valve, "position": partial_move, "status": "partial", "target": target}

        elif self._fault_type == "reverse":
            # Moves opposite direction
            current = self._valve_positions[valve]
            reverse_move = current - (target - current)
            reverse_move = max(0, min(100, reverse_move))
            self._valve_positions[valve] = reverse_move
            return {"valve": valve, "position": reverse_move, "status": "reverse"}

        return {"valve": valve, "status": "unknown_fault"}


# =============================================================================
# Steady State Hypothesis for UnifiedSteam
# =============================================================================

def create_unifiedsteam_hypothesis():
    """Create steady state hypothesis specific to UnifiedSteam agent."""
    import sys
    import os

    gl001_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
    if gl001_path not in sys.path:
        sys.path.insert(0, gl001_path)

    from steady_state import SteadyStateHypothesis, SteadyStateMetric, ComparisonOperator

    return SteadyStateHypothesis(
        name="UnifiedSteam Header Balance Health",
        description="Validates UnifiedSteam agent is operating normally",
        metrics=[
            SteadyStateMetric(
                name="header_pressure_psi",
                description="Steam header pressure",
                threshold=(140, 160),
                operator=ComparisonOperator.IN_RANGE,
                required=True,
            ),
            SteadyStateMetric(
                name="load_balance_variance",
                description="Load distribution variance across boilers",
                threshold=10.0,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="coordination_latency_ms",
                description="Multi-boiler coordination latency",
                threshold=100,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="forecast_accuracy_percent",
                description="Demand forecast accuracy",
                threshold=90.0,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=False,
            ),
            SteadyStateMetric(
                name="valve_response_time_ms",
                description="Average valve response time",
                threshold=500,
                operator=ComparisonOperator.LESS_THAN,
                required=False,
            ),
        ],
        pass_threshold=0.8,
        aggregation="weighted",
    )
