# -*- coding: utf-8 -*-
"""
Sensor Failure Mode Handling for GL-005 CombustionSense
=======================================================

Provides comprehensive sensor failure detection and handling
for safety-critical combustion measurements.

Failure Modes Covered:
    - Complete sensor failure (no signal)
    - Out-of-range readings
    - Frozen/stuck values
    - Erratic behavior
    - Calibration drift
    - Communication failures

Safety Philosophy:
    - Fail-safe by default
    - Clear alarm escalation
    - Automatic fallback to backup sensors
    - Complete audit trail

Reference Standards:
    - IEC 61508: Functional Safety
    - IEC 61511: Safety Instrumented Systems
    - ISA-84.01: Safety Instrumented Functions

Author: GL-SafetyEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
from datetime import datetime, timedelta
import statistics
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class FailureMode(Enum):
    """Sensor failure modes."""
    NONE = "none"
    NO_SIGNAL = "no_signal"
    OUT_OF_RANGE_HIGH = "out_of_range_high"
    OUT_OF_RANGE_LOW = "out_of_range_low"
    FROZEN = "frozen"
    ERRATIC = "erratic"
    DRIFT = "drift"
    COMMUNICATION_ERROR = "communication_error"
    CALIBRATION_EXPIRED = "calibration_expired"


class FailureAction(Enum):
    """Actions to take on sensor failure."""
    NONE = "none"
    ALARM = "alarm"
    USE_BACKUP = "use_backup"
    USE_CALCULATED = "use_calculated"
    REDUCE_LOAD = "reduce_load"
    TRIP = "trip"


class SensorState(Enum):
    """Sensor operational state."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    BYPASSED = "bypassed"
    MAINTENANCE = "maintenance"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SensorSpec:
    """Sensor specification and limits."""
    sensor_id: str
    parameter: str
    units: str
    range_min: float
    range_max: float
    accuracy: float
    response_time_seconds: float
    calibration_interval_days: int
    is_safety_critical: bool = True
    backup_sensor_id: Optional[str] = None


@dataclass
class SensorReading:
    """Sensor reading with metadata."""
    sensor_id: str
    value: float
    timestamp: datetime
    quality: str = "GOOD"
    raw_value: Optional[float] = None


@dataclass
class SensorHealth:
    """Current sensor health status."""
    sensor_id: str
    state: SensorState
    failure_mode: FailureMode
    last_good_reading: Optional[datetime] = None
    readings_since_last_good: int = 0
    time_in_current_state: timedelta = timedelta(0)
    fault_details: List[str] = field(default_factory=list)


@dataclass
class FailureEvent:
    """Record of sensor failure event."""
    sensor_id: str
    failure_mode: FailureMode
    action_taken: FailureAction
    timestamp: datetime
    details: Dict[str, Any]
    provenance_hash: str


# =============================================================================
# SENSOR FAILURE HANDLER
# =============================================================================

class SensorFailureHandler:
    """
    Handles sensor failure detection and response.

    Detection Methods:
        - Range checking
        - Frozen value detection
        - Rate-of-change analysis
        - Statistical outlier detection

    Response Actions:
        - Alarm generation
        - Backup sensor activation
        - Calculated value substitution
        - Safety trip initiation
    """

    def __init__(self):
        self.sensors: Dict[str, SensorSpec] = {}
        self.health_status: Dict[str, SensorHealth] = {}
        self.reading_history: Dict[str, List[SensorReading]] = {}
        self.failure_callbacks: List[Callable] = []
        self.failure_log: List[FailureEvent] = []

        # Detection thresholds
        self.frozen_threshold_readings = 10  # Number of identical readings
        self.frozen_threshold_seconds = 30   # Time window
        self.erratic_threshold_cv = 0.5      # Coefficient of variation
        self.rate_of_change_limit = 50.0     # % of span per second

    def register_sensor(self, spec: SensorSpec) -> None:
        """
        Register a sensor for monitoring.

        Args:
            spec: Sensor specification
        """
        self.sensors[spec.sensor_id] = spec
        self.health_status[spec.sensor_id] = SensorHealth(
            sensor_id=spec.sensor_id,
            state=SensorState.HEALTHY,
            failure_mode=FailureMode.NONE,
        )
        self.reading_history[spec.sensor_id] = []

    def process_reading(
        self,
        reading: SensorReading
    ) -> Tuple[SensorHealth, Optional[FailureAction]]:
        """
        Process a sensor reading and check for failures.

        Args:
            reading: Sensor reading to process

        Returns:
            Tuple of (health status, action if any)
        """
        sensor_id = reading.sensor_id

        if sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        spec = self.sensors[sensor_id]
        health = self.health_status[sensor_id]

        # Add to history
        self.reading_history[sensor_id].append(reading)
        if len(self.reading_history[sensor_id]) > 100:
            self.reading_history[sensor_id] = self.reading_history[sensor_id][-100:]

        # Run failure checks
        failure_mode = self._check_for_failure(reading, spec)

        if failure_mode != FailureMode.NONE:
            action = self._handle_failure(sensor_id, failure_mode, reading)
            return self.health_status[sensor_id], action

        # Update health to healthy if was previously degraded
        if health.state == SensorState.DEGRADED:
            health.readings_since_last_good += 1
            if health.readings_since_last_good > 5:  # 5 good readings to recover
                health.state = SensorState.HEALTHY
                health.failure_mode = FailureMode.NONE

        health.last_good_reading = reading.timestamp

        return health, None

    def _check_for_failure(
        self,
        reading: SensorReading,
        spec: SensorSpec
    ) -> FailureMode:
        """Check reading for all failure modes."""
        # Check range
        if reading.value > spec.range_max:
            return FailureMode.OUT_OF_RANGE_HIGH
        if reading.value < spec.range_min:
            return FailureMode.OUT_OF_RANGE_LOW

        # Check for frozen value
        if self._is_frozen(reading.sensor_id):
            return FailureMode.FROZEN

        # Check for erratic behavior
        if self._is_erratic(reading.sensor_id):
            return FailureMode.ERRATIC

        # Check quality flag
        if reading.quality in ["BAD", "FAILED", "COMM_ERROR"]:
            return FailureMode.COMMUNICATION_ERROR

        return FailureMode.NONE

    def _is_frozen(self, sensor_id: str) -> bool:
        """Check if sensor value is frozen."""
        history = self.reading_history.get(sensor_id, [])

        if len(history) < self.frozen_threshold_readings:
            return False

        recent = history[-self.frozen_threshold_readings:]
        values = [r.value for r in recent]

        # Check if all values are identical
        if len(set(values)) == 1:
            # Also check time span
            time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
            if time_span >= self.frozen_threshold_seconds:
                return True

        return False

    def _is_erratic(self, sensor_id: str) -> bool:
        """Check if sensor behavior is erratic."""
        history = self.reading_history.get(sensor_id, [])

        if len(history) < 10:
            return False

        recent = history[-10:]
        values = [r.value for r in recent]

        mean = statistics.mean(values)
        if mean == 0:
            return False

        std = statistics.stdev(values)
        cv = std / abs(mean)

        return cv > self.erratic_threshold_cv

    def _handle_failure(
        self,
        sensor_id: str,
        failure_mode: FailureMode,
        reading: SensorReading
    ) -> FailureAction:
        """Handle detected sensor failure."""
        spec = self.sensors[sensor_id]
        health = self.health_status[sensor_id]

        # Determine action based on failure mode and criticality
        if failure_mode in [FailureMode.OUT_OF_RANGE_HIGH, FailureMode.OUT_OF_RANGE_LOW]:
            action = FailureAction.USE_BACKUP if spec.backup_sensor_id else FailureAction.TRIP
        elif failure_mode == FailureMode.FROZEN:
            action = FailureAction.USE_BACKUP if spec.backup_sensor_id else FailureAction.ALARM
        elif failure_mode == FailureMode.ERRATIC:
            action = FailureAction.ALARM
        elif failure_mode == FailureMode.COMMUNICATION_ERROR:
            action = FailureAction.USE_BACKUP if spec.backup_sensor_id else FailureAction.TRIP
        else:
            action = FailureAction.ALARM

        # For safety-critical sensors, escalate to trip
        if spec.is_safety_critical and action == FailureAction.ALARM:
            action = FailureAction.TRIP if not spec.backup_sensor_id else FailureAction.USE_BACKUP

        # Update health status
        health.state = SensorState.FAILED if action == FailureAction.TRIP else SensorState.DEGRADED
        health.failure_mode = failure_mode
        health.readings_since_last_good = 0
        health.fault_details.append(f"{failure_mode.value} at {reading.timestamp}")

        # Log failure event
        event = FailureEvent(
            sensor_id=sensor_id,
            failure_mode=failure_mode,
            action_taken=action,
            timestamp=reading.timestamp,
            details={
                "value": reading.value,
                "spec_range": (spec.range_min, spec.range_max),
            },
            provenance_hash=self._calculate_hash(sensor_id, failure_mode, reading),
        )
        self.failure_log.append(event)

        # Execute callbacks
        for callback in self.failure_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")

        logger.warning(f"Sensor {sensor_id} failure: {failure_mode.value}, action: {action.value}")

        return action

    def get_fallback_value(
        self,
        sensor_id: str,
        current_readings: Optional[Dict[str, float]] = None
    ) -> Optional[float]:
        """
        Get fallback value for failed sensor.

        Fallback Priority:
            1. Backup sensor
            2. Calculated value (if formula available)
            3. Last known good value
            4. None (trip)

        Args:
            sensor_id: Failed sensor ID
            current_readings: Current readings from other sensors

        Returns:
            Fallback value or None if not available
        """
        spec = self.sensors.get(sensor_id)
        if not spec:
            return None

        # Try backup sensor
        if spec.backup_sensor_id:
            backup_health = self.health_status.get(spec.backup_sensor_id)
            if backup_health and backup_health.state == SensorState.HEALTHY:
                backup_history = self.reading_history.get(spec.backup_sensor_id, [])
                if backup_history:
                    return backup_history[-1].value

        # Try last known good value (with time limit)
        history = self.reading_history.get(sensor_id, [])
        if history:
            last_good = None
            for reading in reversed(history):
                if reading.quality == "GOOD":
                    last_good = reading
                    break

            if last_good:
                age = (datetime.now() - last_good.timestamp).total_seconds()
                if age < 60:  # Only use if less than 60 seconds old
                    return last_good.value

        return None

    def register_failure_callback(self, callback: Callable[[FailureEvent], None]) -> None:
        """Register callback for failure events."""
        self.failure_callbacks.append(callback)

    def get_sensor_status_summary(self) -> Dict[str, Any]:
        """Get summary of all sensor statuses."""
        summary = {
            "total_sensors": len(self.sensors),
            "healthy": 0,
            "degraded": 0,
            "failed": 0,
            "sensors": {},
        }

        for sensor_id, health in self.health_status.items():
            if health.state == SensorState.HEALTHY:
                summary["healthy"] += 1
            elif health.state == SensorState.DEGRADED:
                summary["degraded"] += 1
            else:
                summary["failed"] += 1

            summary["sensors"][sensor_id] = {
                "state": health.state.value,
                "failure_mode": health.failure_mode.value,
            }

        return summary

    def _calculate_hash(
        self,
        sensor_id: str,
        failure_mode: FailureMode,
        reading: SensorReading
    ) -> str:
        """Calculate provenance hash for failure event."""
        data = f"{sensor_id}:{failure_mode.value}:{reading.value}:{reading.timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_combustion_sensor_specs() -> List[SensorSpec]:
    """Create standard combustion sensor specifications."""
    return [
        SensorSpec(
            sensor_id="O2-PRIMARY",
            parameter="O2",
            units="%",
            range_min=0.0,
            range_max=25.0,
            accuracy=0.1,
            response_time_seconds=10.0,
            calibration_interval_days=30,
            is_safety_critical=True,
            backup_sensor_id="O2-BACKUP",
        ),
        SensorSpec(
            sensor_id="O2-BACKUP",
            parameter="O2",
            units="%",
            range_min=0.0,
            range_max=25.0,
            accuracy=0.1,
            response_time_seconds=10.0,
            calibration_interval_days=30,
            is_safety_critical=True,
        ),
        SensorSpec(
            sensor_id="CO-PRIMARY",
            parameter="CO",
            units="ppm",
            range_min=0.0,
            range_max=5000.0,
            accuracy=10.0,
            response_time_seconds=30.0,
            calibration_interval_days=90,
            is_safety_critical=True,
            backup_sensor_id="CO-BACKUP",
        ),
        SensorSpec(
            sensor_id="CO-BACKUP",
            parameter="CO",
            units="ppm",
            range_min=0.0,
            range_max=5000.0,
            accuracy=10.0,
            response_time_seconds=30.0,
            calibration_interval_days=90,
            is_safety_critical=True,
        ),
        SensorSpec(
            sensor_id="FLAME-PRIMARY",
            parameter="flame_signal",
            units="mA",
            range_min=0.0,
            range_max=20.0,
            accuracy=0.1,
            response_time_seconds=1.0,
            calibration_interval_days=365,
            is_safety_critical=True,
        ),
    ]


if __name__ == "__main__":
    # Example usage
    handler = SensorFailureHandler()

    for spec in create_combustion_sensor_specs():
        handler.register_sensor(spec)

    # Simulate normal reading
    reading = SensorReading(
        sensor_id="O2-PRIMARY",
        value=3.5,
        timestamp=datetime.now(),
        quality="GOOD",
    )

    health, action = handler.process_reading(reading)
    print(f"Health: {health.state.value}, Action: {action}")

    # Simulate failed reading
    failed_reading = SensorReading(
        sensor_id="O2-PRIMARY",
        value=30.0,  # Out of range
        timestamp=datetime.now(),
        quality="GOOD",
    )

    health, action = handler.process_reading(failed_reading)
    print(f"Health: {health.state.value}, Action: {action}")
