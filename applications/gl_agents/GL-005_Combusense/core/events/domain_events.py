# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Domain Events

This module defines all domain events for the combustion control system.
Each event represents a significant fact that occurred in the system and
can be used to rebuild system state through event replay.

Domain Events:
    - ControlSetpointChanged: Fuel/air setpoints were modified
    - SafetyInterventionTriggered: Safety system took action
    - OptimizationCompleted: Optimization cycle finished
    - SensorReadingReceived: New sensor data received
    - AlarmTriggered: Alarm condition detected
    - SystemStateChanged: System mode or status changed

Design Principles:
    - Events are past tense (something HAS happened)
    - Events are immutable facts
    - Events carry all necessary data for replay
    - Events support schema versioning

Example:
    >>> event = ControlSetpointChanged(
    ...     aggregate_id="combustion-001",
    ...     fuel_flow_setpoint=1000.0,
    ...     air_flow_setpoint=12500.0,
    ...     reason="optimization"
    ... )
    >>> print(event.event_type)  # "ControlSetpointChanged"
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, ConfigDict

from core.events.base_event import DomainEvent, EventMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class SetpointChangeReason(str, Enum):
    """Reason for setpoint change."""
    OPTIMIZATION = "optimization"
    MANUAL = "manual"
    SAFETY = "safety"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    LOAD_CHANGE = "load_change"
    FAULT_RECOVERY = "fault_recovery"


class SafetyInterventionType(str, Enum):
    """Type of safety intervention."""
    FUEL_CUTOFF = "fuel_cutoff"
    AIR_INCREASE = "air_increase"
    SETPOINT_REDUCTION = "setpoint_reduction"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    INTERLOCK_TRIP = "interlock_trip"
    ALARM_ESCALATION = "alarm_escalation"


class AlarmSeverity(str, Enum):
    """Alarm severity levels per ISA-18.2."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlarmCategory(str, Enum):
    """Alarm category classification."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    EMISSIONS = "emissions"
    FLAME = "flame"
    SAFETY = "safety"
    EQUIPMENT = "equipment"
    COMMUNICATION = "communication"


class SystemMode(str, Enum):
    """System operating mode."""
    OFFLINE = "offline"
    STANDBY = "standby"
    PURGE = "purge"
    IGNITION = "ignition"
    WARMUP = "warmup"
    NORMAL = "normal"
    HIGH_FIRE = "high_fire"
    LOW_FIRE = "low_fire"
    SHUTDOWN = "shutdown"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


# =============================================================================
# Domain Events
# =============================================================================


class ControlSetpointChanged(DomainEvent):
    """
    Event emitted when control setpoints are modified.

    This event captures changes to fuel flow, air flow, or other control
    setpoints. It includes both the new values and the reason for change.

    Attributes:
        fuel_flow_setpoint: New fuel flow setpoint (kg/hr or m3/hr)
        previous_fuel_flow_setpoint: Previous fuel flow setpoint
        air_flow_setpoint: New air flow setpoint (m3/hr)
        previous_air_flow_setpoint: Previous air flow setpoint
        fuel_valve_position: Fuel valve position (0-100%)
        air_damper_position: Air damper position (0-100%)
        o2_trim_enabled: Whether O2 trim is active
        o2_trim_correction: O2 trim correction applied
        reason: Reason for setpoint change
        control_mode: Current control mode (auto/manual)

    Example:
        >>> event = ControlSetpointChanged(
        ...     aggregate_id="combustion-001",
        ...     fuel_flow_setpoint=1000.0,
        ...     previous_fuel_flow_setpoint=950.0,
        ...     air_flow_setpoint=12500.0,
        ...     previous_air_flow_setpoint=12000.0,
        ...     reason=SetpointChangeReason.OPTIMIZATION
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # New setpoints
    fuel_flow_setpoint: float = Field(
        ...,
        ge=0,
        description="New fuel flow setpoint (kg/hr or m3/hr)"
    )
    air_flow_setpoint: float = Field(
        ...,
        ge=0,
        description="New air flow setpoint (m3/hr)"
    )

    # Previous setpoints for delta calculation
    previous_fuel_flow_setpoint: Optional[float] = Field(
        default=None,
        ge=0,
        description="Previous fuel flow setpoint"
    )
    previous_air_flow_setpoint: Optional[float] = Field(
        default=None,
        ge=0,
        description="Previous air flow setpoint"
    )

    # Valve/damper positions
    fuel_valve_position: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Fuel valve position (0-100%)"
    )
    air_damper_position: float = Field(
        default=50.0,
        ge=0,
        le=100,
        description="Air damper position (0-100%)"
    )

    # O2 trim
    o2_trim_enabled: bool = Field(
        default=True,
        description="O2 trim control enabled"
    )
    o2_trim_correction: float = Field(
        default=0.0,
        description="O2 trim correction applied (m3/hr)"
    )

    # Change metadata
    reason: SetpointChangeReason = Field(
        default=SetpointChangeReason.OPTIMIZATION,
        description="Reason for setpoint change"
    )
    control_mode: str = Field(
        default="auto",
        description="Control mode (auto/manual/cascade)"
    )

    # PID contributions (for diagnostics)
    pid_proportional: Optional[float] = Field(
        default=None,
        description="PID proportional term contribution"
    )
    pid_integral: Optional[float] = Field(
        default=None,
        description="PID integral term contribution"
    )
    pid_derivative: Optional[float] = Field(
        default=None,
        description="PID derivative term contribution"
    )

    @property
    def fuel_flow_delta(self) -> float:
        """Calculate fuel flow change."""
        if self.previous_fuel_flow_setpoint is None:
            return 0.0
        return self.fuel_flow_setpoint - self.previous_fuel_flow_setpoint

    @property
    def air_flow_delta(self) -> float:
        """Calculate air flow change."""
        if self.previous_air_flow_setpoint is None:
            return 0.0
        return self.air_flow_setpoint - self.previous_air_flow_setpoint


class SafetyInterventionTriggered(DomainEvent):
    """
    Event emitted when the safety system takes action.

    This event captures safety interventions including interlock trips,
    emergency shutdowns, and automatic protective actions.

    Attributes:
        intervention_type: Type of safety intervention
        severity: Severity level of the intervention
        trigger_condition: What triggered the intervention
        trigger_value: Measured value that triggered action
        trigger_limit: Limit that was exceeded
        interlocks_tripped: List of interlocks that tripped
        actions_taken: List of protective actions taken
        requires_operator_action: Whether operator action is needed
        auto_recovery_enabled: Whether system can auto-recover

    Example:
        >>> event = SafetyInterventionTriggered(
        ...     aggregate_id="combustion-001",
        ...     intervention_type=SafetyInterventionType.FUEL_CUTOFF,
        ...     severity=AlarmSeverity.CRITICAL,
        ...     trigger_condition="flame_loss",
        ...     trigger_value=0.0,
        ...     trigger_limit=1.0
        ... )
    """

    model_config = ConfigDict(frozen=True)

    intervention_type: SafetyInterventionType = Field(
        ...,
        description="Type of safety intervention"
    )
    severity: AlarmSeverity = Field(
        default=AlarmSeverity.HIGH,
        description="Severity of the intervention"
    )

    # Trigger information
    trigger_condition: str = Field(
        ...,
        description="Condition that triggered intervention"
    )
    trigger_value: float = Field(
        ...,
        description="Measured value that triggered action"
    )
    trigger_limit: float = Field(
        ...,
        description="Limit that was exceeded"
    )
    trigger_deviation: Optional[float] = Field(
        default=None,
        description="Deviation from limit"
    )

    # Interlock information
    interlocks_tripped: List[str] = Field(
        default_factory=list,
        description="List of interlock names that tripped"
    )
    interlocks_bypassed: List[str] = Field(
        default_factory=list,
        description="List of bypassed interlocks"
    )

    # Actions taken
    actions_taken: List[str] = Field(
        default_factory=list,
        description="List of protective actions taken"
    )
    fuel_flow_after: Optional[float] = Field(
        default=None,
        description="Fuel flow after intervention"
    )
    air_flow_after: Optional[float] = Field(
        default=None,
        description="Air flow after intervention"
    )

    # Recovery information
    requires_operator_action: bool = Field(
        default=True,
        description="Operator action required to clear"
    )
    auto_recovery_enabled: bool = Field(
        default=False,
        description="System can automatically recover"
    )
    estimated_recovery_time_seconds: Optional[int] = Field(
        default=None,
        description="Estimated time to recover"
    )

    # Safety system info
    safety_integrity_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="SIL level of the protection"
    )


class OptimizationCompleted(DomainEvent):
    """
    Event emitted when an optimization cycle completes.

    This event captures the results of fuel-air ratio optimization,
    efficiency optimization, or other control optimization activities.

    Attributes:
        optimization_type: Type of optimization performed
        objective_function: What was being optimized
        initial_value: Value before optimization
        final_value: Value after optimization
        improvement_percent: Percentage improvement achieved
        iterations: Number of optimization iterations
        constraints_satisfied: Whether all constraints were met
        new_setpoints: New setpoints from optimization

    Example:
        >>> event = OptimizationCompleted(
        ...     aggregate_id="combustion-001",
        ...     optimization_type="fuel_air_ratio",
        ...     objective_function="thermal_efficiency",
        ...     initial_value=85.5,
        ...     final_value=88.2,
        ...     improvement_percent=3.16
        ... )
    """

    model_config = ConfigDict(frozen=True)

    optimization_type: str = Field(
        ...,
        description="Type of optimization (fuel_air_ratio, efficiency, emissions)"
    )
    objective_function: str = Field(
        ...,
        description="Objective being optimized"
    )

    # Optimization results
    initial_value: float = Field(
        ...,
        description="Objective value before optimization"
    )
    final_value: float = Field(
        ...,
        description="Objective value after optimization"
    )
    improvement_percent: float = Field(
        ...,
        description="Percentage improvement"
    )

    # Optimization parameters
    iterations: int = Field(
        default=1,
        ge=1,
        description="Number of iterations"
    )
    convergence_achieved: bool = Field(
        default=True,
        description="Whether optimization converged"
    )
    constraints_satisfied: bool = Field(
        default=True,
        description="Whether all constraints met"
    )
    violated_constraints: List[str] = Field(
        default_factory=list,
        description="List of violated constraints"
    )

    # New setpoints from optimization
    optimized_fuel_flow: float = Field(
        ...,
        ge=0,
        description="Optimized fuel flow setpoint"
    )
    optimized_air_flow: float = Field(
        ...,
        ge=0,
        description="Optimized air flow setpoint"
    )
    optimized_excess_air_percent: float = Field(
        ...,
        ge=0,
        description="Optimized excess air percentage"
    )

    # Performance predictions
    predicted_efficiency: float = Field(
        ...,
        ge=0,
        le=100,
        description="Predicted efficiency (%)"
    )
    predicted_nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Predicted NOx emissions (ppm)"
    )
    predicted_co_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Predicted CO emissions (ppm)"
    )

    # Cost savings
    estimated_fuel_savings_percent: Optional[float] = Field(
        default=None,
        description="Estimated fuel savings (%)"
    )
    estimated_cost_savings_per_hour: Optional[float] = Field(
        default=None,
        description="Estimated cost savings ($/hr)"
    )

    # Execution metrics
    execution_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Optimization execution time (ms)"
    )


class SensorReadingReceived(DomainEvent):
    """
    Event emitted when new sensor data is received.

    This event captures sensor readings from the combustion system
    including flows, temperatures, pressures, and emissions data.

    Attributes:
        reading_type: Type of sensor reading
        sensor_id: Unique sensor identifier
        value: Measured value
        unit: Engineering unit
        quality: Data quality indicator
        raw_value: Raw sensor value before scaling
        timestamp_source: Timestamp from sensor

    Example:
        >>> event = SensorReadingReceived(
        ...     aggregate_id="combustion-001",
        ...     reading_type="temperature",
        ...     sensor_id="furnace_temp_001",
        ...     value=1200.5,
        ...     unit="degC",
        ...     quality="good"
        ... )
    """

    model_config = ConfigDict(frozen=True)

    reading_type: str = Field(
        ...,
        description="Type of reading (temperature, pressure, flow, emissions)"
    )
    sensor_id: str = Field(
        ...,
        description="Unique sensor identifier"
    )

    # Measurement
    value: float = Field(
        ...,
        description="Measured value in engineering units"
    )
    unit: str = Field(
        ...,
        description="Engineering unit (degC, kPa, kg/hr, ppm, etc.)"
    )
    raw_value: Optional[float] = Field(
        default=None,
        description="Raw sensor value before scaling"
    )

    # Data quality
    quality: str = Field(
        default="good",
        description="Data quality (good, uncertain, bad, stale)"
    )
    quality_code: Optional[int] = Field(
        default=None,
        description="OPC-style quality code"
    )

    # Timestamps
    timestamp_source: Optional[datetime] = Field(
        default=None,
        description="Timestamp from sensor/source"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Data latency (ms)"
    )

    # Validation
    in_range: bool = Field(
        default=True,
        description="Value within valid range"
    )
    range_min: Optional[float] = Field(
        default=None,
        description="Valid range minimum"
    )
    range_max: Optional[float] = Field(
        default=None,
        description="Valid range maximum"
    )

    # Derived data
    rate_of_change: Optional[float] = Field(
        default=None,
        description="Rate of change (per second)"
    )
    deviation_from_setpoint: Optional[float] = Field(
        default=None,
        description="Deviation from setpoint"
    )

    # Sensor health
    sensor_health: str = Field(
        default="healthy",
        description="Sensor health status"
    )
    maintenance_required: bool = Field(
        default=False,
        description="Sensor requires maintenance"
    )


class AlarmTriggered(DomainEvent):
    """
    Event emitted when an alarm condition is detected.

    This event captures alarm activations following ISA-18.2 alarm
    management principles including shelving, suppression, and priorities.

    Attributes:
        alarm_id: Unique alarm identifier
        alarm_name: Descriptive alarm name
        category: Alarm category
        severity: Alarm severity level
        trigger_value: Value that triggered alarm
        setpoint: Alarm setpoint
        deviation: Deviation from setpoint
        state: Alarm state (active, acknowledged, cleared)
        acknowledged_by: Who acknowledged the alarm

    Example:
        >>> event = AlarmTriggered(
        ...     aggregate_id="combustion-001",
        ...     alarm_id="HH_FURNACE_TEMP",
        ...     alarm_name="Furnace Temperature High-High",
        ...     category=AlarmCategory.TEMPERATURE,
        ...     severity=AlarmSeverity.CRITICAL,
        ...     trigger_value=1450.0,
        ...     setpoint=1400.0
        ... )
    """

    model_config = ConfigDict(frozen=True)

    alarm_id: str = Field(
        ...,
        description="Unique alarm identifier"
    )
    alarm_name: str = Field(
        ...,
        description="Descriptive alarm name"
    )

    # Classification
    category: AlarmCategory = Field(
        ...,
        description="Alarm category"
    )
    severity: AlarmSeverity = Field(
        ...,
        description="Alarm severity level"
    )

    # Trigger information
    trigger_value: float = Field(
        ...,
        description="Value that triggered alarm"
    )
    setpoint: float = Field(
        ...,
        description="Alarm setpoint"
    )
    deviation: float = Field(
        ...,
        description="Deviation from setpoint"
    )
    deviation_percent: Optional[float] = Field(
        default=None,
        description="Deviation as percentage"
    )

    # Alarm state (ISA-18.2)
    state: str = Field(
        default="active_unacknowledged",
        description="Alarm state (active_unacknowledged, active_acknowledged, cleared_unacknowledged, cleared)"
    )
    is_new: bool = Field(
        default=True,
        description="Is this a new alarm occurrence"
    )
    occurrence_count: int = Field(
        default=1,
        ge=1,
        description="Number of occurrences"
    )

    # Response information
    response_time_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time to acknowledge (seconds)"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="Who acknowledged the alarm"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="When alarm was acknowledged"
    )

    # Actions
    required_actions: List[str] = Field(
        default_factory=list,
        description="Required operator actions"
    )
    automated_actions_taken: List[str] = Field(
        default_factory=list,
        description="Automated actions taken"
    )

    # Shelving/suppression (ISA-18.2)
    is_shelved: bool = Field(
        default=False,
        description="Alarm is shelved"
    )
    is_suppressed: bool = Field(
        default=False,
        description="Alarm is suppressed"
    )
    suppression_reason: Optional[str] = Field(
        default=None,
        description="Reason for suppression"
    )


class SystemStateChanged(DomainEvent):
    """
    Event emitted when the system mode or operational state changes.

    This event captures transitions between operating modes, startup/shutdown
    sequences, and other system-level state changes.

    Attributes:
        previous_mode: Previous operating mode
        new_mode: New operating mode
        transition_reason: Reason for state change
        operator_initiated: Whether operator initiated change
        transition_successful: Whether transition completed
        sequence_step: Current step in sequence (if applicable)

    Example:
        >>> event = SystemStateChanged(
        ...     aggregate_id="combustion-001",
        ...     previous_mode=SystemMode.STANDBY,
        ...     new_mode=SystemMode.PURGE,
        ...     transition_reason="startup_initiated",
        ...     operator_initiated=True
        ... )
    """

    model_config = ConfigDict(frozen=True)

    previous_mode: SystemMode = Field(
        ...,
        description="Previous operating mode"
    )
    new_mode: SystemMode = Field(
        ...,
        description="New operating mode"
    )

    # Transition information
    transition_reason: str = Field(
        ...,
        description="Reason for state change"
    )
    operator_initiated: bool = Field(
        default=False,
        description="Whether operator initiated"
    )
    transition_successful: bool = Field(
        default=True,
        description="Whether transition completed"
    )

    # Sequence tracking
    sequence_name: Optional[str] = Field(
        default=None,
        description="Name of sequence (startup, shutdown, etc.)"
    )
    sequence_step: Optional[int] = Field(
        default=None,
        ge=0,
        description="Current step in sequence"
    )
    sequence_total_steps: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total steps in sequence"
    )

    # Timing
    transition_duration_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Duration of transition"
    )
    expected_duration_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Expected transition duration"
    )

    # System state at transition
    fuel_flow_at_transition: Optional[float] = Field(
        default=None,
        ge=0,
        description="Fuel flow at transition"
    )
    air_flow_at_transition: Optional[float] = Field(
        default=None,
        ge=0,
        description="Air flow at transition"
    )
    furnace_temperature_at_transition: Optional[float] = Field(
        default=None,
        description="Furnace temperature at transition"
    )

    # Interlocks
    interlocks_status: Dict[str, bool] = Field(
        default_factory=dict,
        description="Interlock status at transition"
    )
    permissives_satisfied: bool = Field(
        default=True,
        description="All permissives satisfied"
    )
    missing_permissives: List[str] = Field(
        default_factory=list,
        description="List of missing permissives"
    )

    # Control handoff
    control_transferred_from: Optional[str] = Field(
        default=None,
        description="Previous control owner"
    )
    control_transferred_to: Optional[str] = Field(
        default=None,
        description="New control owner"
    )


# =============================================================================
# Event Registry
# =============================================================================


EVENT_REGISTRY: Dict[str, type] = {
    "ControlSetpointChanged": ControlSetpointChanged,
    "SafetyInterventionTriggered": SafetyInterventionTriggered,
    "OptimizationCompleted": OptimizationCompleted,
    "SensorReadingReceived": SensorReadingReceived,
    "AlarmTriggered": AlarmTriggered,
    "SystemStateChanged": SystemStateChanged,
}


def get_event_class(event_type: str) -> type:
    """
    Get event class by type name.

    Args:
        event_type: Event type name

    Returns:
        Event class

    Raises:
        KeyError: If event type not found
    """
    if event_type not in EVENT_REGISTRY:
        raise KeyError(f"Unknown event type: {event_type}")
    return EVENT_REGISTRY[event_type]


def deserialize_event(event_data: Dict[str, Any]) -> DomainEvent:
    """
    Deserialize event from dictionary.

    Args:
        event_data: Dictionary with event data

    Returns:
        Deserialized domain event

    Raises:
        KeyError: If event type not found
        ValueError: If event data is invalid
    """
    event_type = event_data.get("event_type")
    if not event_type:
        raise ValueError("Event data missing 'event_type' field")

    event_class = get_event_class(event_type)
    return event_class(**event_data)
