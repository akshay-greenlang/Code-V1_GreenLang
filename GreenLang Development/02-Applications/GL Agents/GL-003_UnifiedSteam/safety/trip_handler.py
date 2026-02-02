"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Trip Handler

This module handles trip event detection, logging, and reporting for the
steam system. GL-003 has REPORTING ONLY authority - no control over SIS.

Safety Architecture:
    - Trip condition detection and monitoring
    - Comprehensive trip event logging
    - Safe state monitoring (no control authority)
    - Trip report generation for analysis

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - API 556 Instrumentation and Control Systems

IMPORTANT: GL-003 has NO control authority over Safety Instrumented Systems.
This module provides monitoring and reporting capabilities only.

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TripType(str, Enum):
    """Trip type enumeration."""
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"
    HIGH_FLOW = "high_flow"
    LOW_FLOW = "low_flow"
    EQUIPMENT_FAULT = "equipment_fault"
    LOSS_OF_UTILITY = "loss_of_utility"
    MANUAL = "manual"
    PROCESS_DEVIATION = "process_deviation"
    STEAM_QUALITY = "steam_quality"


class TripSeverity(str, Enum):
    """Trip severity enumeration."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TripState(str, Enum):
    """Trip state enumeration."""
    DETECTED = "detected"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESET = "reset"
    CLEARED = "cleared"


class SafeStateStatus(str, Enum):
    """Safe state status enumeration."""
    ACHIEVED = "achieved"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


# =============================================================================
# DATA MODELS
# =============================================================================

class TripDefinition(BaseModel):
    """Definition of a trip condition."""

    trip_id: str = Field(..., description="Trip identifier")
    name: str = Field(..., description="Trip name")
    trip_type: TripType = Field(..., description="Type of trip")
    parameter: str = Field(..., description="Parameter that triggers trip")
    setpoint: float = Field(..., description="Trip setpoint")
    operator: str = Field(
        default=">",
        description="Comparison operator (>, <, >=, <=)"
    )
    unit: str = Field(default="", description="Parameter unit")
    equipment_id: str = Field(default="", description="Associated equipment")
    severity: TripSeverity = Field(
        default=TripSeverity.HIGH,
        description="Trip severity"
    )
    safe_state_action: str = Field(
        default="",
        description="Safe state action description"
    )
    reset_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions required for reset"
    )
    is_latching: bool = Field(
        default=True,
        description="Trip latches until manual reset"
    )


class TripCondition(BaseModel):
    """Detected trip condition."""

    condition_id: str = Field(..., description="Condition ID")
    trip_definition_id: str = Field(..., description="Trip definition ID")
    trip_type: TripType = Field(..., description="Type of trip")
    parameter: str = Field(..., description="Parameter in trip")
    actual_value: float = Field(..., description="Value that caused trip")
    setpoint: float = Field(..., description="Trip setpoint")
    deviation: float = Field(..., description="Deviation from setpoint")
    unit: str = Field(default="", description="Parameter unit")
    equipment_id: str = Field(default="", description="Affected equipment")
    severity: TripSeverity = Field(..., description="Trip severity")
    detected_at: datetime = Field(
        default_factory=datetime.now,
        description="Detection timestamp"
    )
    message: str = Field(default="", description="Trip message")


class TripEvent(BaseModel):
    """Logged trip event."""

    event_id: str = Field(..., description="Event ID")
    trip_condition: TripCondition = Field(..., description="Trip condition")
    state: TripState = Field(..., description="Current trip state")
    event_time: datetime = Field(
        default_factory=datetime.now,
        description="Event timestamp"
    )
    acknowledged_by: Optional[str] = Field(
        None,
        description="User who acknowledged"
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="Acknowledgment timestamp"
    )
    reset_by: Optional[str] = Field(None, description="User who reset")
    reset_at: Optional[datetime] = Field(None, description="Reset timestamp")
    safe_state_status: SafeStateStatus = Field(
        default=SafeStateStatus.NOT_REQUIRED,
        description="Safe state status"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Event notes"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class SafeStateResult(BaseModel):
    """Result of safe state monitoring."""

    result_id: str = Field(..., description="Result ID")
    trip_event_id: str = Field(..., description="Associated trip event")
    trip_type: TripType = Field(..., description="Trip type")
    status: SafeStateStatus = Field(..., description="Safe state status")
    safe_state_achieved_at: Optional[datetime] = Field(
        None,
        description="Time safe state was achieved"
    )
    time_to_safe_state_s: Optional[float] = Field(
        None,
        description="Time to achieve safe state (seconds)"
    )
    equipment_states: Dict[str, str] = Field(
        default_factory=dict,
        description="Equipment states after trip"
    )
    verification_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Verification checks performed"
    )
    message: str = Field(default="", description="Status message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Result timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class TripReport(BaseModel):
    """Trip event report for analysis."""

    report_id: str = Field(..., description="Report ID")
    trip_event: TripEvent = Field(..., description="Trip event details")
    safe_state_result: Optional[SafeStateResult] = Field(
        None,
        description="Safe state result if applicable"
    )
    timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Event timeline"
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Contributing factors identified"
    )
    process_data_snapshot: Dict[str, float] = Field(
        default_factory=dict,
        description="Process data at time of trip"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for prevention"
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Report generation timestamp"
    )
    generated_by: str = Field(
        default="GL-003",
        description="Report generator"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class SystemState(BaseModel):
    """System state for trip detection."""

    parameters: Dict[str, float] = Field(
        ...,
        description="Current parameter values"
    )
    equipment_states: Dict[str, str] = Field(
        default_factory=dict,
        description="Equipment states"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="State timestamp"
    )


# =============================================================================
# TRIP HANDLER
# =============================================================================

class TripHandler:
    """
    Trip event handling for steam system safety.

    This handler provides trip detection, logging, and reporting for the
    steam system. IMPORTANT: GL-003 has REPORTING ONLY authority - it cannot
    initiate trips or control SIS equipment.

    Safety Philosophy:
        - Monitor trip conditions - NO initiation authority
        - Comprehensive trip event logging
        - Safe state monitoring (reporting only)
        - Trip analysis and reporting for continuous improvement

    Features:
        - Trip condition detection from process data
        - Complete trip event logging with timestamps
        - Safe state status monitoring
        - Detailed trip reports for analysis
        - Historical trip data access

    Attributes:
        _trip_definitions: Registered trip definitions
        _active_trips: Currently active trip events
        _trip_history: Historical trip events

    Example:
        >>> handler = TripHandler()
        >>> handler.register_trip_definition(definition)
        >>> condition = handler.detect_trip_condition(system_state, definitions)
        >>> if condition:
        ...     event = handler.log_trip_event(condition)
        ...     report = handler.generate_trip_report(event)
    """

    def __init__(self):
        """Initialize TripHandler."""
        self._trip_definitions: Dict[str, TripDefinition] = {}
        self._active_trips: Dict[str, TripEvent] = {}
        self._trip_history: List[TripEvent] = []
        self._safe_state_results: Dict[str, SafeStateResult] = {}
        self._max_history_size = 10000

        logger.info(
            "TripHandler initialized - GL-003 has REPORTING ONLY authority over trips"
        )

    def register_trip_definition(self, definition: TripDefinition) -> None:
        """
        Register a trip definition for monitoring.

        Note: This registers trips for MONITORING only. GL-003 cannot
        modify actual SIS trip logic.

        Args:
            definition: Trip definition to register
        """
        self._trip_definitions[definition.trip_id] = definition
        logger.info(
            f"Registered trip definition {definition.trip_id}: "
            f"{definition.name} ({definition.trip_type.value})"
        )

    def detect_trip_condition(
        self,
        system_state: SystemState,
        trip_definitions: Optional[List[TripDefinition]] = None
    ) -> Optional[TripCondition]:
        """
        Detect trip condition from current system state.

        This method monitors for trip conditions but does NOT initiate trips.
        GL-003 only detects and reports - the SIS handles actual trip initiation.

        Args:
            system_state: Current system state with parameter values
            trip_definitions: Trip definitions to check (uses registered if None)

        Returns:
            TripCondition if detected, None otherwise
        """
        definitions = trip_definitions or list(self._trip_definitions.values())

        for definition in definitions:
            if definition.parameter not in system_state.parameters:
                continue

            value = system_state.parameters[definition.parameter]

            # Evaluate trip condition
            if self._evaluate_trip_condition(definition, value):
                # Calculate deviation
                deviation = value - definition.setpoint

                # Generate condition ID
                condition_id = hashlib.sha256(
                    f"TRIP_{definition.trip_id}_{value}_{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16]

                condition = TripCondition(
                    condition_id=condition_id,
                    trip_definition_id=definition.trip_id,
                    trip_type=definition.trip_type,
                    parameter=definition.parameter,
                    actual_value=value,
                    setpoint=definition.setpoint,
                    deviation=deviation,
                    unit=definition.unit,
                    equipment_id=definition.equipment_id,
                    severity=definition.severity,
                    detected_at=datetime.now(),
                    message=(
                        f"Trip condition detected: {definition.name} - "
                        f"{definition.parameter}={value}{definition.unit} "
                        f"(setpoint: {definition.setpoint}{definition.unit})"
                    )
                )

                logger.warning(
                    f"TRIP CONDITION DETECTED: {definition.name} - "
                    f"{definition.parameter}={value} ({definition.operator}{definition.setpoint})"
                )

                return condition

        return None

    def log_trip_event(
        self,
        trip_condition: TripCondition
    ) -> TripEvent:
        """
        Log a trip event.

        This method creates a comprehensive log entry for a trip event.
        The actual trip response is handled by the SIS - GL-003 only logs.

        Args:
            trip_condition: Detected trip condition

        Returns:
            TripEvent: Logged trip event
        """
        start_time = datetime.now()

        # Generate event ID
        event_id = hashlib.sha256(
            f"EVENT_{trip_condition.condition_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        event = TripEvent(
            event_id=event_id,
            trip_condition=trip_condition,
            state=TripState.DETECTED,
            event_time=start_time,
            safe_state_status=SafeStateStatus.NOT_REQUIRED
        )

        # Calculate provenance hash
        event.provenance_hash = hashlib.sha256(
            f"{event_id}|{trip_condition.condition_id}|{trip_condition.trip_type.value}".encode()
        ).hexdigest()

        # Store in active trips
        self._active_trips[event_id] = event

        # Add to history
        self._trip_history.append(event)
        if len(self._trip_history) > self._max_history_size:
            self._trip_history = self._trip_history[-self._max_history_size:]

        logger.warning(
            f"Trip event logged: {event_id} - {trip_condition.trip_type.value} - "
            f"Severity: {trip_condition.severity.value}"
        )

        return event

    def initiate_safe_state(
        self,
        trip_type: TripType,
        trip_event_id: str,
        equipment_states: Optional[Dict[str, str]] = None
    ) -> SafeStateResult:
        """
        Monitor safe state achievement (REPORTING ONLY - no control authority).

        This method monitors and reports on safe state status. GL-003 does NOT
        control safe state actions - the SIS handles all safe state control.

        Args:
            trip_type: Type of trip
            trip_event_id: Associated trip event ID
            equipment_states: Current equipment states for monitoring

        Returns:
            SafeStateResult: Safe state monitoring result
        """
        start_time = datetime.now()

        # Get the trip event
        trip_event = self._active_trips.get(trip_event_id)
        if trip_event is None:
            logger.error(f"Trip event {trip_event_id} not found")
            raise ValueError(f"Trip event {trip_event_id} not found")

        # Generate result ID
        result_id = hashlib.sha256(
            f"SAFE_{trip_event_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Determine expected safe state based on trip type
        verification_checks = self._get_verification_checks(trip_type)

        # In actual implementation, this would monitor SIS outputs
        # For now, we report monitoring status
        status = SafeStateStatus.IN_PROGRESS
        message = (
            f"Monitoring safe state for {trip_type.value} trip - "
            "GL-003 has REPORTING ONLY authority"
        )

        result = SafeStateResult(
            result_id=result_id,
            trip_event_id=trip_event_id,
            trip_type=trip_type,
            status=status,
            equipment_states=equipment_states or {},
            verification_checks=verification_checks,
            message=message,
            timestamp=datetime.now()
        )

        # Calculate provenance hash
        result.provenance_hash = hashlib.sha256(
            f"{result_id}|{trip_event_id}|{status.value}".encode()
        ).hexdigest()

        # Store result
        self._safe_state_results[trip_event_id] = result

        # Update trip event
        trip_event.safe_state_status = status
        trip_event.notes.append(f"Safe state monitoring initiated: {result_id}")

        logger.info(
            f"Safe state monitoring initiated for trip {trip_event_id}: "
            f"status={status.value}"
        )

        return result

    def generate_trip_report(
        self,
        trip_event: TripEvent,
        process_data: Optional[Dict[str, float]] = None,
        include_analysis: bool = True
    ) -> TripReport:
        """
        Generate comprehensive trip report for analysis.

        This method creates a detailed report of a trip event including
        timeline, contributing factors, and recommendations.

        Args:
            trip_event: Trip event to report on
            process_data: Process data snapshot at time of trip
            include_analysis: Include analysis and recommendations

        Returns:
            TripReport: Comprehensive trip report
        """
        start_time = datetime.now()

        # Generate report ID
        report_id = hashlib.sha256(
            f"REPORT_{trip_event.event_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Build timeline
        timeline = self._build_timeline(trip_event)

        # Get safe state result if available
        safe_state_result = self._safe_state_results.get(trip_event.event_id)

        # Analyze contributing factors
        contributing_factors = []
        recommendations = []

        if include_analysis:
            contributing_factors = self._analyze_contributing_factors(
                trip_event, process_data
            )
            recommendations = self._generate_recommendations(
                trip_event, contributing_factors
            )

        report = TripReport(
            report_id=report_id,
            trip_event=trip_event,
            safe_state_result=safe_state_result,
            timeline=timeline,
            contributing_factors=contributing_factors,
            process_data_snapshot=process_data or {},
            recommendations=recommendations,
            generated_at=datetime.now(),
            generated_by="GL-003 UNIFIEDSTEAM"
        )

        # Calculate provenance hash
        report.provenance_hash = hashlib.sha256(
            f"{report_id}|{trip_event.event_id}|{len(timeline)}".encode()
        ).hexdigest()

        logger.info(
            f"Trip report generated: {report_id} for event {trip_event.event_id}"
        )

        return report

    def acknowledge_trip(
        self,
        event_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> TripEvent:
        """
        Record trip acknowledgment.

        Note: This records the acknowledgment in GL-003's logs. Actual
        trip acknowledgment in the SIS is done separately.

        Args:
            event_id: Trip event ID
            acknowledged_by: User acknowledging
            notes: Optional notes

        Returns:
            Updated TripEvent
        """
        if event_id not in self._active_trips:
            raise ValueError(f"Trip event {event_id} not found")

        event = self._active_trips[event_id]
        event.state = TripState.ACKNOWLEDGED
        event.acknowledged_by = acknowledged_by
        event.acknowledged_at = datetime.now()

        if notes:
            event.notes.append(f"Acknowledgment note: {notes}")

        logger.info(
            f"Trip {event_id} acknowledged by {acknowledged_by}"
        )

        return event

    def reset_trip(
        self,
        event_id: str,
        reset_by: str,
        notes: Optional[str] = None
    ) -> TripEvent:
        """
        Record trip reset.

        Note: This records the reset in GL-003's logs. Actual trip reset
        in the SIS is done separately.

        Args:
            event_id: Trip event ID
            reset_by: User resetting
            notes: Optional notes

        Returns:
            Updated TripEvent
        """
        if event_id not in self._active_trips:
            raise ValueError(f"Trip event {event_id} not found")

        event = self._active_trips[event_id]
        event.state = TripState.RESET
        event.reset_by = reset_by
        event.reset_at = datetime.now()

        if notes:
            event.notes.append(f"Reset note: {notes}")

        # Move to cleared state and remove from active
        event.state = TripState.CLEARED
        del self._active_trips[event_id]

        logger.info(
            f"Trip {event_id} reset by {reset_by}"
        )

        return event

    def get_active_trips(self) -> List[TripEvent]:
        """Get all currently active trips."""
        return list(self._active_trips.values())

    def get_trip_history(
        self,
        time_window_hours: int = 24,
        trip_type: Optional[TripType] = None
    ) -> List[TripEvent]:
        """
        Get trip history within time window.

        Args:
            time_window_hours: Time window in hours
            trip_type: Filter by trip type (optional)

        Returns:
            List of trip events
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)

        events = [
            e for e in self._trip_history
            if e.event_time >= cutoff_time
        ]

        if trip_type:
            events = [
                e for e in events
                if e.trip_condition.trip_type == trip_type
            ]

        return events

    def get_trip_statistics(
        self,
        time_window_hours: int = 720  # 30 days default
    ) -> Dict[str, Any]:
        """
        Get trip statistics for analysis.

        Args:
            time_window_hours: Time window in hours

        Returns:
            Dictionary with trip statistics
        """
        events = self.get_trip_history(time_window_hours)

        # Count by type
        type_counts = {}
        for event in events:
            trip_type = event.trip_condition.trip_type.value
            type_counts[trip_type] = type_counts.get(trip_type, 0) + 1

        # Count by severity
        severity_counts = {}
        for event in events:
            severity = event.trip_condition.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_trips": len(events),
            "active_trips": len(self._active_trips),
            "trips_by_type": type_counts,
            "trips_by_severity": severity_counts,
            "time_window_hours": time_window_hours,
            "generated_at": datetime.now().isoformat()
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _evaluate_trip_condition(
        self,
        definition: TripDefinition,
        value: float
    ) -> bool:
        """Evaluate if a trip condition is met."""
        operators = {
            ">": lambda v, s: v > s,
            "<": lambda v, s: v < s,
            ">=": lambda v, s: v >= s,
            "<=": lambda v, s: v <= s,
        }

        op_func = operators.get(definition.operator)
        if op_func is None:
            logger.error(f"Unknown operator: {definition.operator}")
            return False

        return op_func(value, definition.setpoint)

    def _get_verification_checks(
        self,
        trip_type: TripType
    ) -> Dict[str, bool]:
        """Get verification checks for trip type."""
        # Define standard verification checks by trip type
        checks = {
            TripType.HIGH_PRESSURE: {
                "pressure_relieved": False,
                "prv_opened": False,
                "isolation_complete": False
            },
            TripType.HIGH_TEMPERATURE: {
                "cooling_initiated": False,
                "desuperheater_active": False,
                "heat_input_reduced": False
            },
            TripType.LOW_FLOW: {
                "backup_supply_available": False,
                "isolation_complete": False,
                "equipment_protected": False
            }
        }

        return checks.get(trip_type, {"safe_state_verified": False})

    def _build_timeline(self, trip_event: TripEvent) -> List[Dict[str, Any]]:
        """Build event timeline."""
        timeline = [
            {
                "time": trip_event.trip_condition.detected_at.isoformat(),
                "event": "Trip condition detected",
                "details": trip_event.trip_condition.message
            },
            {
                "time": trip_event.event_time.isoformat(),
                "event": "Trip event logged",
                "details": f"Event ID: {trip_event.event_id}"
            }
        ]

        if trip_event.acknowledged_at:
            timeline.append({
                "time": trip_event.acknowledged_at.isoformat(),
                "event": "Trip acknowledged",
                "details": f"By: {trip_event.acknowledged_by}"
            })

        if trip_event.reset_at:
            timeline.append({
                "time": trip_event.reset_at.isoformat(),
                "event": "Trip reset",
                "details": f"By: {trip_event.reset_by}"
            })

        for note in trip_event.notes:
            timeline.append({
                "time": "N/A",
                "event": "Note",
                "details": note
            })

        return timeline

    def _analyze_contributing_factors(
        self,
        trip_event: TripEvent,
        process_data: Optional[Dict[str, float]]
    ) -> List[str]:
        """Analyze contributing factors for trip."""
        factors = []
        condition = trip_event.trip_condition

        # Analyze deviation
        if abs(condition.deviation) > condition.setpoint * 0.2:
            factors.append(
                f"Large deviation ({abs(condition.deviation):.1f}{condition.unit}) "
                f"from setpoint suggests rapid process upset"
            )

        # Analyze trip type
        if condition.trip_type in (TripType.HIGH_PRESSURE, TripType.HIGH_TEMPERATURE):
            factors.append(
                "High parameter trips often indicate inadequate heat removal "
                "or control system issues"
            )
        elif condition.trip_type in (TripType.LOW_PRESSURE, TripType.LOW_FLOW):
            factors.append(
                "Low parameter trips often indicate supply issues or "
                "excessive demand"
            )

        # Check process data for related anomalies
        if process_data:
            for param, value in process_data.items():
                if param != condition.parameter:
                    # Look for related parameters that may have contributed
                    if "pressure" in param.lower() or "temp" in param.lower():
                        factors.append(
                            f"Related parameter {param}={value} at time of trip"
                        )

        return factors

    def _generate_recommendations(
        self,
        trip_event: TripEvent,
        contributing_factors: List[str]
    ) -> List[str]:
        """Generate recommendations for trip prevention."""
        recommendations = []
        condition = trip_event.trip_condition

        # General recommendations based on trip type
        if condition.trip_type in (TripType.HIGH_PRESSURE, TripType.HIGH_TEMPERATURE):
            recommendations.append(
                "Review control loop tuning to improve response to process upsets"
            )
            recommendations.append(
                "Verify safety valve/relief device sizing is adequate"
            )
        elif condition.trip_type in (TripType.LOW_PRESSURE, TripType.LOW_FLOW):
            recommendations.append(
                "Review supply reliability and consider backup systems"
            )
            recommendations.append(
                "Verify demand management controls are functioning"
            )

        # Severity-based recommendations
        if condition.severity == TripSeverity.CRITICAL:
            recommendations.append(
                "Conduct root cause analysis for critical trip event"
            )
            recommendations.append(
                "Review alarm management to ensure early warning"
            )

        # Standard recommendations
        recommendations.append(
            "Review trip event with operations team for lessons learned"
        )

        return recommendations
