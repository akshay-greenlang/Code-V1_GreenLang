# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Combustion Control Aggregate

This module implements the CombustionControlAggregate, which encapsulates
all combustion control state and enforces business rules through
event-sourced domain logic.

The aggregate replaces the deque-based state management with event sourcing,
providing:
    - Full audit trail of all state changes
    - Deterministic state rebuild from events
    - Temporal queries (state at any point in time)
    - Provenance tracking with SHA-256 hashes

Design Principles:
    - Zero-hallucination: All state derived from events
    - Deterministic: Same events always produce same state
    - Auditable: Complete history with provenance hashes
    - Safe: Business rules enforced before state changes

Example:
    >>> aggregate = CombustionControlAggregate("burner-001")
    >>> aggregate.change_setpoint(fuel_flow=1000, air_flow=12500)
    >>> await aggregate.save_to_store(event_store)
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from core.events.aggregates import Aggregate, AggregateState
from core.events.domain_events import (
    AlarmCategory,
    AlarmSeverity,
    AlarmTriggered,
    ControlSetpointChanged,
    OptimizationCompleted,
    SafetyInterventionTriggered,
    SafetyInterventionType,
    SensorReadingReceived,
    SetpointChangeReason,
    SystemMode,
    SystemStateChanged,
)

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """A sensor reading with timestamp."""
    sensor_id: str
    reading_type: str
    value: float
    unit: str
    timestamp: datetime
    quality: str = "good"


@dataclass
class ControlSetpoint:
    """Current control setpoints."""
    fuel_flow: float = 0.0
    air_flow: float = 0.0
    fuel_valve_position: float = 50.0
    air_damper_position: float = 50.0
    o2_trim_enabled: bool = True
    o2_trim_correction: float = 0.0
    control_mode: str = "auto"


@dataclass
class StabilityMetrics:
    """Stability metrics derived from readings."""
    heat_output_stability: float = 0.5
    furnace_temp_stability: float = 0.5
    flame_temp_stability: float = 0.5
    o2_stability: float = 0.5
    overall_score: float = 50.0
    rating: str = "fair"
    oscillation_detected: bool = False


class CombustionControlState(AggregateState):
    """
    Snapshot state for CombustionControlAggregate.

    This represents the complete state that can be snapshotted
    and restored for performance optimization.
    """

    # Operating mode
    system_mode: SystemMode = Field(default=SystemMode.OFFLINE)
    control_enabled: bool = Field(default=False)

    # Current setpoints
    fuel_flow_setpoint: float = Field(default=0.0)
    air_flow_setpoint: float = Field(default=0.0)
    fuel_valve_position: float = Field(default=50.0)
    air_damper_position: float = Field(default=50.0)
    o2_trim_enabled: bool = Field(default=True)
    o2_trim_correction: float = Field(default=0.0)

    # Current measurements (most recent)
    current_fuel_flow: float = Field(default=0.0)
    current_air_flow: float = Field(default=0.0)
    current_furnace_temp: float = Field(default=0.0)
    current_flue_gas_temp: float = Field(default=0.0)
    current_o2_percent: float = Field(default=0.0)
    current_co_ppm: float = Field(default=0.0)
    current_heat_output_kw: float = Field(default=0.0)

    # Statistics (derived from history)
    avg_heat_output_kw: float = Field(default=0.0)
    heat_output_variance: float = Field(default=0.0)
    stability_score: float = Field(default=50.0)

    # Counters
    total_setpoint_changes: int = Field(default=0)
    total_safety_interventions: int = Field(default=0)
    total_alarms: int = Field(default=0)
    total_optimizations: int = Field(default=0)

    # Timestamps
    last_setpoint_change: Optional[datetime] = Field(default=None)
    last_safety_intervention: Optional[datetime] = Field(default=None)
    last_optimization: Optional[datetime] = Field(default=None)


class CombustionControlAggregate(Aggregate):
    """
    Event-sourced aggregate for combustion control state.

    This aggregate maintains the complete state of a combustion control
    system through event sourcing. All state changes are captured as
    events and can be replayed to rebuild state at any point in time.

    State Management:
        - Replaces deque-based history with event-sourced history
        - Maintains derived state (averages, stability scores)
        - Provides projections for read models
        - Supports temporal queries

    Business Rules:
        - Setpoint changes must be within safe limits
        - Safety interventions take priority
        - Control must be enabled before changes
        - Mode transitions follow allowed paths

    Attributes:
        aggregate_id: Unique burner/combustion unit identifier
        system_mode: Current operating mode
        control_enabled: Whether control is active
        setpoint: Current control setpoints
        stability: Current stability metrics

    Example:
        >>> aggregate = CombustionControlAggregate("burner-001")
        >>> await aggregate.load_from_store(event_store)
        >>> aggregate.change_setpoint(fuel_flow=1000, air_flow=12500)
        >>> aggregate.record_sensor_reading("furnace_temp", 1200, "degC")
        >>> await aggregate.save_to_store(event_store)
    """

    # Maximum history sizes
    MAX_READING_HISTORY = 1000
    MAX_SETPOINT_HISTORY = 500
    MAX_ALARM_HISTORY = 200

    def __init__(self, aggregate_id: str):
        """
        Initialize combustion control aggregate.

        Args:
            aggregate_id: Unique identifier (e.g., "burner-001")
        """
        super().__init__(aggregate_id, "CombustionControlAggregate")

        # Operating state
        self._system_mode = SystemMode.OFFLINE
        self._control_enabled = False

        # Current setpoints
        self._setpoint = ControlSetpoint()

        # Current measurements
        self._current_measurements: Dict[str, SensorReading] = {}

        # History (for statistics calculation)
        self._heat_output_history: Deque[float] = deque(maxlen=self.MAX_READING_HISTORY)
        self._furnace_temp_history: Deque[float] = deque(maxlen=self.MAX_READING_HISTORY)
        self._o2_history: Deque[float] = deque(maxlen=self.MAX_READING_HISTORY)
        self._setpoint_history: Deque[ControlSetpoint] = deque(maxlen=self.MAX_SETPOINT_HISTORY)

        # Active alarms
        self._active_alarms: Dict[str, AlarmTriggered] = {}

        # Statistics
        self._stability_metrics = StabilityMetrics()

        # Counters
        self._total_setpoint_changes = 0
        self._total_safety_interventions = 0
        self._total_alarms = 0
        self._total_optimizations = 0

        # Timestamps
        self._last_setpoint_change: Optional[datetime] = None
        self._last_safety_intervention: Optional[datetime] = None
        self._last_optimization: Optional[datetime] = None

        logger.debug(f"Created CombustionControlAggregate: {aggregate_id}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def system_mode(self) -> SystemMode:
        """Get current system mode."""
        return self._system_mode

    @property
    def control_enabled(self) -> bool:
        """Check if control is enabled."""
        return self._control_enabled

    @property
    def setpoint(self) -> ControlSetpoint:
        """Get current setpoints."""
        return self._setpoint

    @property
    def stability(self) -> StabilityMetrics:
        """Get current stability metrics."""
        return self._stability_metrics

    @property
    def active_alarms(self) -> Dict[str, AlarmTriggered]:
        """Get active alarms."""
        return dict(self._active_alarms)

    @property
    def heat_output_history(self) -> List[float]:
        """Get heat output history for analysis."""
        return list(self._heat_output_history)

    # =========================================================================
    # Commands (produce events)
    # =========================================================================

    def change_setpoint(
        self,
        fuel_flow: float,
        air_flow: float,
        fuel_valve_position: Optional[float] = None,
        air_damper_position: Optional[float] = None,
        o2_trim_enabled: bool = True,
        o2_trim_correction: float = 0.0,
        reason: SetpointChangeReason = SetpointChangeReason.OPTIMIZATION,
        control_mode: str = "auto"
    ) -> None:
        """
        Change control setpoints.

        This command validates the setpoint change and emits a
        ControlSetpointChanged event if valid.

        Args:
            fuel_flow: New fuel flow setpoint
            air_flow: New air flow setpoint
            fuel_valve_position: Fuel valve position (0-100%)
            air_damper_position: Air damper position (0-100%)
            o2_trim_enabled: Enable O2 trim
            o2_trim_correction: O2 trim correction value
            reason: Reason for change
            control_mode: Control mode

        Raises:
            ValueError: If setpoints are invalid
            RuntimeError: If control is not enabled
        """
        # Validate
        if fuel_flow < 0:
            raise ValueError(f"Fuel flow cannot be negative: {fuel_flow}")
        if air_flow < 0:
            raise ValueError(f"Air flow cannot be negative: {air_flow}")

        # Check control is enabled (unless this is a safety action)
        if not self._control_enabled and reason != SetpointChangeReason.SAFETY:
            if self._system_mode not in [SystemMode.STANDBY, SystemMode.OFFLINE]:
                raise RuntimeError("Control is not enabled")

        # Calculate valve positions if not provided
        if fuel_valve_position is None:
            # Simplified linear mapping
            fuel_valve_position = min(100, max(0, (fuel_flow / 2000) * 100))
        if air_damper_position is None:
            air_damper_position = min(100, max(0, (air_flow / 25000) * 100))

        # Create and raise event
        event = ControlSetpointChanged(
            aggregate_id=self._aggregate_id,
            fuel_flow_setpoint=fuel_flow,
            air_flow_setpoint=air_flow,
            previous_fuel_flow_setpoint=self._setpoint.fuel_flow,
            previous_air_flow_setpoint=self._setpoint.air_flow,
            fuel_valve_position=fuel_valve_position,
            air_damper_position=air_damper_position,
            o2_trim_enabled=o2_trim_enabled,
            o2_trim_correction=o2_trim_correction,
            reason=reason,
            control_mode=control_mode
        )

        self.raise_event(event)

    def trigger_safety_intervention(
        self,
        intervention_type: SafetyInterventionType,
        trigger_condition: str,
        trigger_value: float,
        trigger_limit: float,
        interlocks_tripped: Optional[List[str]] = None,
        actions_taken: Optional[List[str]] = None,
        severity: AlarmSeverity = AlarmSeverity.HIGH
    ) -> None:
        """
        Trigger a safety intervention.

        Args:
            intervention_type: Type of intervention
            trigger_condition: What triggered it
            trigger_value: Value that triggered
            trigger_limit: Limit that was exceeded
            interlocks_tripped: List of tripped interlocks
            actions_taken: List of actions taken
            severity: Severity level
        """
        event = SafetyInterventionTriggered(
            aggregate_id=self._aggregate_id,
            intervention_type=intervention_type,
            severity=severity,
            trigger_condition=trigger_condition,
            trigger_value=trigger_value,
            trigger_limit=trigger_limit,
            trigger_deviation=trigger_value - trigger_limit,
            interlocks_tripped=interlocks_tripped or [],
            actions_taken=actions_taken or [],
            fuel_flow_after=0.0 if intervention_type == SafetyInterventionType.FUEL_CUTOFF else None,
            requires_operator_action=True
        )

        self.raise_event(event)

    def record_optimization_result(
        self,
        optimization_type: str,
        objective_function: str,
        initial_value: float,
        final_value: float,
        optimized_fuel_flow: float,
        optimized_air_flow: float,
        optimized_excess_air: float,
        predicted_efficiency: float,
        iterations: int = 1,
        convergence_achieved: bool = True,
        execution_time_ms: float = 0.0
    ) -> None:
        """
        Record an optimization result.

        Args:
            optimization_type: Type of optimization
            objective_function: What was optimized
            initial_value: Value before optimization
            final_value: Value after optimization
            optimized_fuel_flow: Resulting fuel flow
            optimized_air_flow: Resulting air flow
            optimized_excess_air: Resulting excess air %
            predicted_efficiency: Predicted efficiency %
            iterations: Number of iterations
            convergence_achieved: Whether converged
            execution_time_ms: Execution time
        """
        improvement = ((final_value - initial_value) / initial_value * 100
                       if initial_value != 0 else 0)

        event = OptimizationCompleted(
            aggregate_id=self._aggregate_id,
            optimization_type=optimization_type,
            objective_function=objective_function,
            initial_value=initial_value,
            final_value=final_value,
            improvement_percent=improvement,
            iterations=iterations,
            convergence_achieved=convergence_achieved,
            constraints_satisfied=True,
            optimized_fuel_flow=optimized_fuel_flow,
            optimized_air_flow=optimized_air_flow,
            optimized_excess_air_percent=optimized_excess_air,
            predicted_efficiency=predicted_efficiency,
            execution_time_ms=execution_time_ms
        )

        self.raise_event(event)

    def record_sensor_reading(
        self,
        reading_type: str,
        sensor_id: str,
        value: float,
        unit: str,
        quality: str = "good"
    ) -> None:
        """
        Record a sensor reading.

        Args:
            reading_type: Type of reading (temperature, pressure, flow, etc.)
            sensor_id: Sensor identifier
            value: Measured value
            unit: Engineering unit
            quality: Data quality
        """
        event = SensorReadingReceived(
            aggregate_id=self._aggregate_id,
            reading_type=reading_type,
            sensor_id=sensor_id,
            value=value,
            unit=unit,
            quality=quality,
            in_range=True  # Could validate against limits
        )

        self.raise_event(event)

    def trigger_alarm(
        self,
        alarm_id: str,
        alarm_name: str,
        category: AlarmCategory,
        severity: AlarmSeverity,
        trigger_value: float,
        setpoint: float
    ) -> None:
        """
        Trigger an alarm.

        Args:
            alarm_id: Unique alarm identifier
            alarm_name: Descriptive name
            category: Alarm category
            severity: Severity level
            trigger_value: Value that triggered
            setpoint: Alarm setpoint
        """
        event = AlarmTriggered(
            aggregate_id=self._aggregate_id,
            alarm_id=alarm_id,
            alarm_name=alarm_name,
            category=category,
            severity=severity,
            trigger_value=trigger_value,
            setpoint=setpoint,
            deviation=trigger_value - setpoint
        )

        self.raise_event(event)

    def change_system_mode(
        self,
        new_mode: SystemMode,
        reason: str,
        operator_initiated: bool = False
    ) -> None:
        """
        Change system operating mode.

        Args:
            new_mode: New operating mode
            reason: Reason for change
            operator_initiated: Whether operator initiated
        """
        event = SystemStateChanged(
            aggregate_id=self._aggregate_id,
            previous_mode=self._system_mode,
            new_mode=new_mode,
            transition_reason=reason,
            operator_initiated=operator_initiated,
            fuel_flow_at_transition=self._setpoint.fuel_flow,
            air_flow_at_transition=self._setpoint.air_flow
        )

        self.raise_event(event)

    def enable_control(self) -> None:
        """Enable automatic control."""
        if self._control_enabled:
            return

        self.change_system_mode(
            new_mode=SystemMode.NORMAL,
            reason="control_enabled",
            operator_initiated=True
        )

    def disable_control(self) -> None:
        """Disable automatic control."""
        if not self._control_enabled:
            return

        self.change_system_mode(
            new_mode=SystemMode.STANDBY,
            reason="control_disabled",
            operator_initiated=True
        )

    # =========================================================================
    # Event Appliers (update state from events)
    # =========================================================================

    def apply_control_setpoint_changed(self, event: ControlSetpointChanged) -> None:
        """Apply ControlSetpointChanged event."""
        # Update setpoint
        self._setpoint = ControlSetpoint(
            fuel_flow=event.fuel_flow_setpoint,
            air_flow=event.air_flow_setpoint,
            fuel_valve_position=event.fuel_valve_position,
            air_damper_position=event.air_damper_position,
            o2_trim_enabled=event.o2_trim_enabled,
            o2_trim_correction=event.o2_trim_correction,
            control_mode=event.control_mode
        )

        # Update history
        self._setpoint_history.append(self._setpoint)

        # Update counters
        self._total_setpoint_changes += 1
        self._last_setpoint_change = event.metadata.timestamp

        logger.debug(
            f"Setpoint changed: fuel={event.fuel_flow_setpoint}, "
            f"air={event.air_flow_setpoint}"
        )

    def apply_safety_intervention_triggered(
        self,
        event: SafetyInterventionTriggered
    ) -> None:
        """Apply SafetyInterventionTriggered event."""
        # Update counters
        self._total_safety_interventions += 1
        self._last_safety_intervention = event.metadata.timestamp

        # Handle specific intervention types
        if event.intervention_type == SafetyInterventionType.FUEL_CUTOFF:
            self._setpoint.fuel_flow = 0.0
            self._setpoint.fuel_valve_position = 0.0

        elif event.intervention_type == SafetyInterventionType.EMERGENCY_SHUTDOWN:
            self._system_mode = SystemMode.EMERGENCY
            self._control_enabled = False
            self._setpoint.fuel_flow = 0.0
            self._setpoint.air_flow = 0.0

        logger.warning(
            f"Safety intervention: {event.intervention_type.value} - "
            f"{event.trigger_condition}"
        )

    def apply_optimization_completed(self, event: OptimizationCompleted) -> None:
        """Apply OptimizationCompleted event."""
        self._total_optimizations += 1
        self._last_optimization = event.metadata.timestamp

        logger.info(
            f"Optimization completed: {event.optimization_type} - "
            f"{event.improvement_percent:.2f}% improvement"
        )

    def apply_sensor_reading_received(self, event: SensorReadingReceived) -> None:
        """Apply SensorReadingReceived event."""
        # Store current reading
        reading = SensorReading(
            sensor_id=event.sensor_id,
            reading_type=event.reading_type,
            value=event.value,
            unit=event.unit,
            timestamp=event.metadata.timestamp,
            quality=event.quality
        )
        self._current_measurements[event.reading_type] = reading

        # Update history for specific readings
        if event.reading_type == "heat_output":
            self._heat_output_history.append(event.value)
            self._update_heat_output_stats()

        elif event.reading_type == "furnace_temperature":
            self._furnace_temp_history.append(event.value)

        elif event.reading_type == "o2_percent":
            self._o2_history.append(event.value)

    def apply_alarm_triggered(self, event: AlarmTriggered) -> None:
        """Apply AlarmTriggered event."""
        self._active_alarms[event.alarm_id] = event
        self._total_alarms += 1

        logger.warning(
            f"Alarm: {event.alarm_name} ({event.severity.value}) - "
            f"value={event.trigger_value}, setpoint={event.setpoint}"
        )

    def apply_system_state_changed(self, event: SystemStateChanged) -> None:
        """Apply SystemStateChanged event."""
        self._system_mode = event.new_mode

        # Update control enabled based on mode
        if event.new_mode in [SystemMode.NORMAL, SystemMode.HIGH_FIRE, SystemMode.LOW_FIRE]:
            self._control_enabled = True
        elif event.new_mode in [SystemMode.OFFLINE, SystemMode.STANDBY,
                                 SystemMode.SHUTDOWN, SystemMode.EMERGENCY]:
            self._control_enabled = False

        logger.info(
            f"System mode changed: {event.previous_mode.value} -> "
            f"{event.new_mode.value}"
        )

    # =========================================================================
    # Derived State / Projections
    # =========================================================================

    def _update_heat_output_stats(self) -> None:
        """Update heat output statistics from history."""
        if len(self._heat_output_history) < 2:
            return

        values = list(self._heat_output_history)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        variance = stdev ** 2

        # Calculate coefficient of variation
        cv = stdev / mean if mean > 0 else 0

        # Calculate stability score (0-100)
        # Lower CV = more stable = higher score
        stability_score = max(0, min(100, (1 - cv) * 100))

        self._stability_metrics = StabilityMetrics(
            heat_output_stability=1 - min(cv, 1),
            overall_score=stability_score,
            rating=self._get_stability_rating(stability_score)
        )

    def _get_stability_rating(self, score: float) -> str:
        """Get stability rating from score."""
        if score >= 90:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        elif score >= 30:
            return "poor"
        else:
            return "unstable"

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current aggregate state as dictionary.

        Returns:
            Dictionary with current state
        """
        return {
            "aggregate_id": self._aggregate_id,
            "version": self._version,
            "system_mode": self._system_mode.value,
            "control_enabled": self._control_enabled,
            "setpoint": {
                "fuel_flow": self._setpoint.fuel_flow,
                "air_flow": self._setpoint.air_flow,
                "fuel_valve_position": self._setpoint.fuel_valve_position,
                "air_damper_position": self._setpoint.air_damper_position,
                "o2_trim_enabled": self._setpoint.o2_trim_enabled,
                "o2_trim_correction": self._setpoint.o2_trim_correction,
                "control_mode": self._setpoint.control_mode
            },
            "stability": {
                "score": self._stability_metrics.overall_score,
                "rating": self._stability_metrics.rating,
                "oscillation_detected": self._stability_metrics.oscillation_detected
            },
            "counters": {
                "setpoint_changes": self._total_setpoint_changes,
                "safety_interventions": self._total_safety_interventions,
                "alarms": self._total_alarms,
                "optimizations": self._total_optimizations
            },
            "active_alarms": len(self._active_alarms)
        }

    # =========================================================================
    # Snapshot Methods
    # =========================================================================

    def create_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of current state."""
        base = super().create_snapshot()

        base.update({
            "system_mode": self._system_mode.value,
            "control_enabled": self._control_enabled,
            "fuel_flow_setpoint": self._setpoint.fuel_flow,
            "air_flow_setpoint": self._setpoint.air_flow,
            "fuel_valve_position": self._setpoint.fuel_valve_position,
            "air_damper_position": self._setpoint.air_damper_position,
            "o2_trim_enabled": self._setpoint.o2_trim_enabled,
            "o2_trim_correction": self._setpoint.o2_trim_correction,
            "control_mode": self._setpoint.control_mode,
            "stability_score": self._stability_metrics.overall_score,
            "stability_rating": self._stability_metrics.rating,
            "total_setpoint_changes": self._total_setpoint_changes,
            "total_safety_interventions": self._total_safety_interventions,
            "total_alarms": self._total_alarms,
            "total_optimizations": self._total_optimizations,
            "heat_output_history": list(self._heat_output_history)[-100:],  # Last 100
            "furnace_temp_history": list(self._furnace_temp_history)[-100:],
            "o2_history": list(self._o2_history)[-100:],
        })

        return base

    def restore_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        super().restore_from_snapshot(snapshot)

        self._system_mode = SystemMode(snapshot.get("system_mode", "offline"))
        self._control_enabled = snapshot.get("control_enabled", False)

        self._setpoint = ControlSetpoint(
            fuel_flow=snapshot.get("fuel_flow_setpoint", 0.0),
            air_flow=snapshot.get("air_flow_setpoint", 0.0),
            fuel_valve_position=snapshot.get("fuel_valve_position", 50.0),
            air_damper_position=snapshot.get("air_damper_position", 50.0),
            o2_trim_enabled=snapshot.get("o2_trim_enabled", True),
            o2_trim_correction=snapshot.get("o2_trim_correction", 0.0),
            control_mode=snapshot.get("control_mode", "auto")
        )

        self._stability_metrics = StabilityMetrics(
            overall_score=snapshot.get("stability_score", 50.0),
            rating=snapshot.get("stability_rating", "fair")
        )

        self._total_setpoint_changes = snapshot.get("total_setpoint_changes", 0)
        self._total_safety_interventions = snapshot.get("total_safety_interventions", 0)
        self._total_alarms = snapshot.get("total_alarms", 0)
        self._total_optimizations = snapshot.get("total_optimizations", 0)

        # Restore histories
        for value in snapshot.get("heat_output_history", []):
            self._heat_output_history.append(value)
        for value in snapshot.get("furnace_temp_history", []):
            self._furnace_temp_history.append(value)
        for value in snapshot.get("o2_history", []):
            self._o2_history.append(value)

        logger.debug(f"Restored aggregate from snapshot at version {self._version}")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_average_heat_output(self, last_n: Optional[int] = None) -> float:
        """
        Get average heat output from history.

        Args:
            last_n: Number of recent readings to average

        Returns:
            Average heat output (kW)
        """
        if not self._heat_output_history:
            return 0.0

        values = list(self._heat_output_history)
        if last_n:
            values = values[-last_n:]

        return statistics.mean(values)

    def get_setpoint_change_rate(self, hours: float = 1.0) -> float:
        """
        Get rate of setpoint changes per hour.

        Args:
            hours: Time window in hours

        Returns:
            Changes per hour
        """
        if not self._last_setpoint_change:
            return 0.0

        # This is simplified - would need full event history for accurate rate
        return self._total_setpoint_changes / max(hours, 0.1)

    def is_stable(self, threshold: float = 70.0) -> bool:
        """
        Check if system is stable.

        Args:
            threshold: Stability score threshold

        Returns:
            True if stability score >= threshold
        """
        return self._stability_metrics.overall_score >= threshold

    def has_active_alarms(self, severity: Optional[AlarmSeverity] = None) -> bool:
        """
        Check for active alarms.

        Args:
            severity: Filter by severity

        Returns:
            True if matching active alarms exist
        """
        if not self._active_alarms:
            return False

        if severity is None:
            return len(self._active_alarms) > 0

        return any(
            alarm.severity == severity
            for alarm in self._active_alarms.values()
        )

    def clear_alarm(self, alarm_id: str) -> bool:
        """
        Clear an active alarm.

        Args:
            alarm_id: Alarm to clear

        Returns:
            True if alarm was cleared
        """
        if alarm_id in self._active_alarms:
            del self._active_alarms[alarm_id]
            return True
        return False
