"""
GL-004 Burnmaster - Agent-Specific Chaos Engineering Components

This module provides chaos engineering components specific to the
Burnmaster Combustion Optimizer Agent, including:

- Combustion air/fuel ratio imbalance
- Burner management system failures
- Flame detection failures
- Emission monitoring disruptions
- Oxygen trim control failures

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
class BurnmasterChaosConfig:
    """Configuration for Burnmaster-specific chaos tests."""

    # Combustion parameters
    optimal_air_fuel_ratio: float = 15.0  # Stoichiometric ratio for natural gas
    air_fuel_deviation_max: float = 2.0

    # Burner parameters
    burner_response_time_ms: float = 500.0
    flame_detection_timeout_ms: float = 100.0

    # Emission parameters
    max_nox_ppm: float = 50.0
    max_co_ppm: float = 100.0
    o2_target_percent: float = 3.0


class BurnerState(Enum):
    """Burner operational states."""
    OFF = "off"
    PURGING = "purging"
    PILOT_IGNITION = "pilot_ignition"
    MAIN_FLAME = "main_flame"
    MODULATING = "modulating"
    SHUTDOWN = "shutdown"
    LOCKOUT = "lockout"


class FlameStatus(Enum):
    """Flame detection status."""
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    WEAK = "weak"
    UNSTABLE = "unstable"
    SENSOR_FAULT = "sensor_fault"


class EmissionLevel(Enum):
    """Emission compliance levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    SENSOR_OFFLINE = "sensor_offline"


# =============================================================================
# Combustion Ratio Fault Injector
# =============================================================================

class CombustionRatioFaultInjector:
    """
    Inject faults into combustion air/fuel ratio control.

    Fault types:
    - Lean combustion (excess air)
    - Rich combustion (excess fuel)
    - Ratio oscillation
    - Sensor drift
    - Control loop failure

    Example:
        >>> injector = CombustionRatioFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "lean",
        ...     "deviation_percent": 15
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._optimal_ratio = 15.0
        self._current_ratio = 15.0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject combustion ratio fault."""
        try:
            self._fault_type = params.get("fault_type", "lean")
            self._params = params

            logger.info(f"CombustionRatioFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"CombustionRatioFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove combustion ratio fault."""
        try:
            logger.info("CombustionRatioFaultInjector: Rolling back")
            self._active = False
            self._current_ratio = self._optimal_ratio
            return True

        except Exception as e:
            logger.error(f"CombustionRatioFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_ratio(self) -> float:
        """Get current air/fuel ratio with fault effects."""
        if not self._active:
            return self._optimal_ratio + random.uniform(-0.1, 0.1)

        deviation = self._params.get("deviation_percent", 10) / 100.0

        if self._fault_type == "lean":
            return self._optimal_ratio * (1 + deviation)

        elif self._fault_type == "rich":
            return self._optimal_ratio * (1 - deviation)

        elif self._fault_type == "oscillation":
            amplitude = self._params.get("amplitude", 2.0)
            frequency = self._params.get("frequency_hz", 0.5)
            import math
            oscillation = amplitude * math.sin(2 * math.pi * frequency * time.time())
            return self._optimal_ratio + oscillation

        elif self._fault_type == "drift":
            drift_rate = self._params.get("drift_rate", 0.01)
            self._current_ratio += drift_rate
            return self._current_ratio

        elif self._fault_type == "sensor_fault":
            return None  # No reading

        return self._optimal_ratio

    def get_combustion_quality(self) -> str:
        """Assess combustion quality based on ratio."""
        ratio = self.get_ratio()
        if ratio is None:
            return "unknown"

        if 14.5 <= ratio <= 15.5:
            return "optimal"
        elif 13.5 <= ratio <= 16.5:
            return "acceptable"
        elif ratio < 13.5:
            return "rich"
        else:
            return "lean"


# =============================================================================
# Burner Management Fault Injector
# =============================================================================

class BurnerManagementFaultInjector:
    """
    Inject faults into burner management system.

    Fault types:
    - Ignition failure
    - Unexpected shutdown
    - Lockout condition
    - Sequence error
    - Interlock bypass (simulated for testing)

    Example:
        >>> injector = BurnerManagementFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "ignition_failure",
        ...     "failure_probability": 0.3
        ... })
    """

    def __init__(self, num_burners: int = 2):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._burners: Dict[str, BurnerState] = {
            f"burner_{i+1}": BurnerState.OFF for i in range(num_burners)
        }

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject burner management fault."""
        try:
            self._fault_type = params.get("fault_type", "ignition_failure")
            self._params = params

            if self._fault_type == "lockout":
                burner = params.get("burner", "burner_1")
                self._burners[burner] = BurnerState.LOCKOUT

            logger.info(f"BurnerManagementFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"BurnerManagementFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove burner management fault."""
        try:
            logger.info("BurnerManagementFaultInjector: Rolling back")
            self._active = False
            for burner in self._burners:
                self._burners[burner] = BurnerState.OFF
            return True

        except Exception as e:
            logger.error(f"BurnerManagementFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_burner_state(self, burner: str) -> BurnerState:
        return self._burners.get(burner, BurnerState.OFF)

    async def start_burner(self, burner: str) -> Dict[str, Any]:
        """Attempt to start a burner with potential faults."""
        if burner not in self._burners:
            return {"status": "error", "error": "burner_not_found"}

        if self._burners[burner] == BurnerState.LOCKOUT:
            return {"status": "lockout", "burner": burner}

        if not self._active:
            self._burners[burner] = BurnerState.MAIN_FLAME
            return {"status": "success", "state": BurnerState.MAIN_FLAME.value}

        if self._fault_type == "ignition_failure":
            failure_prob = self._params.get("failure_probability", 0.3)
            if random.random() < failure_prob:
                self._burners[burner] = BurnerState.LOCKOUT
                return {"status": "ignition_failed", "burner": burner}

        elif self._fault_type == "unexpected_shutdown":
            self._burners[burner] = BurnerState.SHUTDOWN
            return {"status": "shutdown", "burner": burner}

        elif self._fault_type == "sequence_error":
            return {"status": "sequence_error", "burner": burner}

        self._burners[burner] = BurnerState.MAIN_FLAME
        return {"status": "success", "state": BurnerState.MAIN_FLAME.value}


# =============================================================================
# Flame Detection Fault Injector
# =============================================================================

class FlameDetectionFaultInjector:
    """
    Inject faults into flame detection system.

    Fault types:
    - False positive (ghost flame)
    - False negative (missed flame)
    - Weak signal
    - Sensor contamination
    - Scanner failure

    Example:
        >>> injector = FlameDetectionFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "false_negative",
        ...     "probability": 0.2
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject flame detection fault."""
        try:
            self._fault_type = params.get("fault_type", "false_negative")
            self._params = params

            logger.info(f"FlameDetectionFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"FlameDetectionFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove flame detection fault."""
        try:
            logger.info("FlameDetectionFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"FlameDetectionFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def detect_flame(self, actual_flame_present: bool) -> Dict[str, Any]:
        """Detect flame with potential faults."""
        if not self._active:
            return {
                "detected": actual_flame_present,
                "status": FlameStatus.DETECTED if actual_flame_present else FlameStatus.NOT_DETECTED,
                "signal_strength": 95 if actual_flame_present else 0,
            }

        probability = self._params.get("probability", 0.2)

        if self._fault_type == "false_negative":
            if actual_flame_present and random.random() < probability:
                return {
                    "detected": False,
                    "status": FlameStatus.NOT_DETECTED,
                    "signal_strength": 0,
                    "error": "flame_not_detected",
                }

        elif self._fault_type == "false_positive":
            if not actual_flame_present and random.random() < probability:
                return {
                    "detected": True,
                    "status": FlameStatus.DETECTED,
                    "signal_strength": 50,
                    "warning": "ghost_flame",
                }

        elif self._fault_type == "weak_signal":
            if actual_flame_present:
                return {
                    "detected": True,
                    "status": FlameStatus.WEAK,
                    "signal_strength": 30,
                }

        elif self._fault_type == "sensor_failure":
            return {
                "detected": None,
                "status": FlameStatus.SENSOR_FAULT,
                "signal_strength": None,
                "error": "scanner_offline",
            }

        return {
            "detected": actual_flame_present,
            "status": FlameStatus.DETECTED if actual_flame_present else FlameStatus.NOT_DETECTED,
            "signal_strength": 95 if actual_flame_present else 0,
        }


# =============================================================================
# Emission Monitoring Fault Injector
# =============================================================================

class EmissionMonitoringFaultInjector:
    """
    Inject faults into emission monitoring systems.

    Fault types:
    - NOx sensor drift
    - CO sensor failure
    - O2 analyzer fault
    - Data gap
    - Calibration error

    Example:
        >>> injector = EmissionMonitoringFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "nox_drift",
        ...     "drift_ppm": 20
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._base_nox = 30.0
        self._base_co = 50.0
        self._base_o2 = 3.0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject emission monitoring fault."""
        try:
            self._fault_type = params.get("fault_type", "nox_drift")
            self._params = params

            logger.info(f"EmissionMonitoringFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"EmissionMonitoringFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove emission monitoring fault."""
        try:
            logger.info("EmissionMonitoringFaultInjector: Rolling back")
            self._active = False
            return True

        except Exception as e:
            logger.error(f"EmissionMonitoringFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_emissions(self) -> Dict[str, Any]:
        """Get emission readings with potential faults."""
        if not self._active:
            return {
                "nox_ppm": self._base_nox + random.uniform(-2, 2),
                "co_ppm": self._base_co + random.uniform(-5, 5),
                "o2_percent": self._base_o2 + random.uniform(-0.2, 0.2),
                "status": "normal",
            }

        if self._fault_type == "nox_drift":
            drift = self._params.get("drift_ppm", 20)
            return {
                "nox_ppm": self._base_nox + drift,
                "co_ppm": self._base_co,
                "o2_percent": self._base_o2,
                "status": "nox_elevated",
            }

        elif self._fault_type == "co_sensor_failure":
            return {
                "nox_ppm": self._base_nox,
                "co_ppm": None,
                "o2_percent": self._base_o2,
                "status": "co_sensor_offline",
            }

        elif self._fault_type == "o2_analyzer_fault":
            return {
                "nox_ppm": self._base_nox,
                "co_ppm": self._base_co,
                "o2_percent": None,
                "status": "o2_analyzer_fault",
            }

        elif self._fault_type == "data_gap":
            if random.random() < 0.3:
                return {"status": "no_data"}
            return {
                "nox_ppm": self._base_nox,
                "co_ppm": self._base_co,
                "o2_percent": self._base_o2,
                "status": "normal",
            }

        elif self._fault_type == "calibration_error":
            error_factor = self._params.get("error_factor", 1.3)
            return {
                "nox_ppm": self._base_nox * error_factor,
                "co_ppm": self._base_co * error_factor,
                "o2_percent": self._base_o2 * error_factor,
                "status": "calibration_suspect",
            }

        return {"status": "unknown_fault"}

    def get_compliance_level(self) -> EmissionLevel:
        """Get emission compliance level."""
        emissions = self.get_emissions()

        if emissions.get("status") in ["co_sensor_offline", "o2_analyzer_fault", "no_data"]:
            return EmissionLevel.SENSOR_OFFLINE

        nox = emissions.get("nox_ppm", 0)
        co = emissions.get("co_ppm", 0)

        if nox is None or co is None:
            return EmissionLevel.SENSOR_OFFLINE

        if nox > 50 or co > 100:
            return EmissionLevel.VIOLATION
        elif nox > 40 or co > 80:
            return EmissionLevel.WARNING
        else:
            return EmissionLevel.COMPLIANT


# =============================================================================
# Oxygen Trim Fault Injector
# =============================================================================

class OxygenTrimFaultInjector:
    """
    Inject faults into oxygen trim control system.

    Fault types:
    - Control loop failure
    - Setpoint deviation
    - Response delay
    - Hunting (oscillation)
    - Sensor bypass

    Example:
        >>> injector = OxygenTrimFaultInjector()
        >>> await injector.inject({
        ...     "fault_type": "hunting",
        ...     "amplitude": 1.5
        ... })
    """

    def __init__(self):
        self._active = False
        self._fault_type = ""
        self._params: Dict[str, Any] = {}
        self._setpoint = 3.0  # Target O2%
        self._current_o2 = 3.0

    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject oxygen trim fault."""
        try:
            self._fault_type = params.get("fault_type", "control_failure")
            self._params = params

            logger.info(f"OxygenTrimFaultInjector: Injecting {self._fault_type}")
            self._active = True
            return True

        except Exception as e:
            logger.error(f"OxygenTrimFaultInjector: Injection failed: {e}")
            return False

    async def rollback(self) -> bool:
        """Remove oxygen trim fault."""
        try:
            logger.info("OxygenTrimFaultInjector: Rolling back")
            self._active = False
            self._current_o2 = self._setpoint
            return True

        except Exception as e:
            logger.error(f"OxygenTrimFaultInjector: Rollback failed: {e}")
            return False

    def is_active(self) -> bool:
        return self._active

    def get_o2_level(self) -> float:
        """Get current O2 level with fault effects."""
        if not self._active:
            return self._setpoint + random.uniform(-0.1, 0.1)

        if self._fault_type == "control_failure":
            # O2 drifts without control
            self._current_o2 += random.uniform(-0.2, 0.2)
            return max(0.5, min(10.0, self._current_o2))

        elif self._fault_type == "setpoint_deviation":
            deviation = self._params.get("deviation", 2.0)
            return self._setpoint + deviation

        elif self._fault_type == "hunting":
            amplitude = self._params.get("amplitude", 1.5)
            frequency = self._params.get("frequency_hz", 0.3)
            import math
            oscillation = amplitude * math.sin(2 * math.pi * frequency * time.time())
            return self._setpoint + oscillation

        elif self._fault_type == "response_delay":
            # Slowly approach setpoint
            target_diff = self._setpoint - self._current_o2
            self._current_o2 += target_diff * 0.1
            return self._current_o2

        return self._setpoint

    async def adjust_trim(self, target_o2: float) -> Dict[str, Any]:
        """Adjust O2 trim with potential faults."""
        if not self._active:
            self._setpoint = target_o2
            self._current_o2 = target_o2
            return {"status": "success", "o2_level": target_o2}

        if self._fault_type == "control_failure":
            return {"status": "control_fault", "o2_level": self._current_o2}

        elif self._fault_type == "response_delay":
            delay_ms = self._params.get("delay_ms", 5000)
            await asyncio.sleep(delay_ms / 1000.0)
            self._setpoint = target_o2
            return {"status": "delayed", "o2_level": self.get_o2_level()}

        return {"status": "success", "o2_level": target_o2}


# =============================================================================
# Steady State Hypothesis for Burnmaster
# =============================================================================

def create_burnmaster_hypothesis():
    """Create steady state hypothesis specific to Burnmaster agent."""
    import sys
    import os

    gl001_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "GL-001_Thermalcommand", "tests", "chaos"))
    if gl001_path not in sys.path:
        sys.path.insert(0, gl001_path)

    from steady_state import SteadyStateHypothesis, SteadyStateMetric, ComparisonOperator

    return SteadyStateHypothesis(
        name="Burnmaster Combustion Optimizer Health",
        description="Validates Burnmaster agent is operating normally",
        metrics=[
            SteadyStateMetric(
                name="air_fuel_ratio",
                description="Combustion air/fuel ratio",
                threshold=(14.5, 15.5),
                operator=ComparisonOperator.IN_RANGE,
                required=True,
            ),
            SteadyStateMetric(
                name="flame_signal_strength",
                description="Flame detector signal strength",
                threshold=80,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=True,
            ),
            SteadyStateMetric(
                name="nox_ppm",
                description="NOx emissions",
                threshold=50,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="co_ppm",
                description="CO emissions",
                threshold=100,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="o2_percent",
                description="Flue gas O2 percentage",
                threshold=(2.5, 4.0),
                operator=ComparisonOperator.IN_RANGE,
                required=False,
            ),
        ],
        pass_threshold=0.8,
        aggregation="weighted",
    )
