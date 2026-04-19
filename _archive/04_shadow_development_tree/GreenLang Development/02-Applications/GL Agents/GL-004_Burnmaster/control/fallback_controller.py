"""
GL-004 BURNMASTER - Fallback Controller

This module implements the fallback control system for the burner management
control system. It provides safe state operations during anomalies or failures
with automatic state preservation and controlled recovery.

Key Features:
    - Define and maintain safe states
    - Automatic trigger on anomaly detection
    - State preservation before fallback
    - Controlled recovery procedures
    - Complete incident logging

Operating Mode: FALLBACK
    - Safe state operation
    - Anomaly/failure response
    - Recovery procedures

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - NFPA 85/86 Boiler/Furnace Standards

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FallbackTrigger(str, Enum):
    """Triggers that can initiate fallback mode."""
    MANUAL = "manual"
    COMMUNICATION_LOSS = "communication_loss"
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    SAFETY_LIMIT = "safety_limit"
    BMS_INTERLOCK = "bms_interlock"
    ANOMALY_DETECTION = "anomaly_detection"
    MODE_TIMEOUT = "mode_timeout"
    PROCESS_INSTABILITY = "process_instability"
    WATCHDOG_TIMEOUT = "watchdog_timeout"


class FallbackSeverity(str, Enum):
    """Severity levels of fallback events."""
    WARNING = "warning"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafeStateType(str, Enum):
    """Types of safe states."""
    HOLD = "hold"
    LOW_FIRE = "low_fire"
    REDUCED_LOAD = "reduced_load"
    CONTROLLED_SHUTDOWN = "controlled_shutdown"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class RecoveryStatus(str, Enum):
    """Status of recovery from fallback."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    AWAITING_CLEARANCE = "awaiting_clearance"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class SafeState(BaseModel):
    """Definition of a safe operating state."""
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    state_type: SafeStateType
    name: str
    description: str = Field(default="")
    setpoints: Dict[str, float] = Field(default_factory=dict)
    output_limits: Dict[str, tuple] = Field(default_factory=dict)
    required_conditions: List[str] = Field(default_factory=list)
    recovery_procedure: str = Field(default="")
    minimum_hold_time_seconds: float = Field(default=60.0, ge=0)
    requires_operator_clearance: bool = Field(default=True)


class FallbackEvent(BaseModel):
    """Record of a fallback event."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger: FallbackTrigger
    severity: FallbackSeverity
    safe_state: SafeStateType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trigger_details: str = Field(default="")
    process_values_at_trigger: Dict[str, float] = Field(default_factory=dict)
    setpoints_at_trigger: Dict[str, float] = Field(default_factory=dict)
    initiated_by: str = Field(default="SYSTEM")
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.event_id}|{self.trigger.value}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class FallbackResult(BaseModel):
    """Result of entering fallback mode."""
    success: bool
    event: FallbackEvent
    safe_state_achieved: bool
    transition_time_seconds: float = Field(default=0.0, ge=0)
    outputs_applied: Dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None


class FallbackIncident(BaseModel):
    """Complete record of a fallback incident."""
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event: FallbackEvent
    duration_seconds: float = Field(default=0.0, ge=0)
    recovery_status: RecoveryStatus = Field(default=RecoveryStatus.NOT_STARTED)
    recovery_attempts: int = Field(default=0, ge=0)
    recovered_at: Optional[datetime] = None
    root_cause: str = Field(default="")
    corrective_actions: List[str] = Field(default_factory=list)
    resolved: bool = Field(default=False)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.incident_id}|{self.event.event_id}|{self.duration_seconds}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class RevertResult(BaseModel):
    """Result of reverting from fallback to normal operation."""
    success: bool
    incident_id: str
    recovery_status: RecoveryStatus
    previous_mode: str
    restored_setpoints: Dict[str, float] = Field(default_factory=dict)
    revert_time_seconds: float = Field(default=0.0, ge=0)
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


class NotificationResult(BaseModel):
    """Result of sending a fallback notification."""
    success: bool
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    recipients: List[str] = Field(default_factory=list)
    channels: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


class FallbackController:
    """
    Controls fallback operations for safe state management.

    This class manages fallback procedures including:
    - Safe state definitions and transitions
    - State preservation before fallback
    - Recovery procedures
    - Incident logging and tracking

    Example:
        >>> controller = FallbackController()
        >>> result = controller.enter_fallback(
        ...     trigger=FallbackTrigger.ANOMALY_DETECTION,
        ...     severity=FallbackSeverity.MAJOR,
        ...     details="Excessive O2 deviation detected"
        ... )
        >>> if result.success:
        ...     # Wait for conditions to clear
        ...     revert = controller.revert_to_normal(result.event.incident_id)
    """

    # Default safe states for different severity levels
    DEFAULT_SAFE_STATES: Dict[FallbackSeverity, SafeStateType] = {
        FallbackSeverity.WARNING: SafeStateType.HOLD,
        FallbackSeverity.MINOR: SafeStateType.REDUCED_LOAD,
        FallbackSeverity.MAJOR: SafeStateType.LOW_FIRE,
        FallbackSeverity.CRITICAL: SafeStateType.CONTROLLED_SHUTDOWN,
        FallbackSeverity.EMERGENCY: SafeStateType.EMERGENCY_SHUTDOWN,
    }

    def __init__(self) -> None:
        """Initialize the fallback controller."""
        self._is_in_fallback = False
        self._current_safe_state: Optional[SafeStateType] = None
        self._current_incident: Optional[FallbackIncident] = None
        self._safe_state_definitions: Dict[SafeStateType, SafeState] = {}
        self._incident_history: List[FallbackIncident] = []
        self._preserved_state: Dict[str, Any] = {}
        self._audit_log: List[Dict[str, Any]] = []

        # Callbacks
        self._notification_callback: Optional[Callable[[FallbackEvent], None]] = None
        self._pre_fallback_callback: Optional[Callable[[], Dict[str, Any]]] = None
        self._post_recovery_callback: Optional[Callable[[], None]] = None

        # Statistics
        self._total_fallbacks = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0

        # Initialize default safe state definitions
        self._initialize_safe_states()

        logger.info("FallbackController initialized")

    def _initialize_safe_states(self) -> None:
        """Initialize default safe state definitions."""
        self._safe_state_definitions = {
            SafeStateType.HOLD: SafeState(
                state_type=SafeStateType.HOLD,
                name="Hold Current Position",
                description="Maintain current setpoints, disable optimization",
                setpoints={},  # Keep current values
                minimum_hold_time_seconds=30.0,
                requires_operator_clearance=False,
                recovery_procedure="Verify process stability before resuming"
            ),
            SafeStateType.LOW_FIRE: SafeState(
                state_type=SafeStateType.LOW_FIRE,
                name="Low Fire Position",
                description="Reduce to minimum safe firing rate",
                setpoints={
                    "firing_rate": 0.25,
                    "damper_position": 30.0,
                    "fgr_damper": 5.0
                },
                minimum_hold_time_seconds=120.0,
                requires_operator_clearance=True,
                recovery_procedure="Verify all sensors, then gradually increase load"
            ),
            SafeStateType.REDUCED_LOAD: SafeState(
                state_type=SafeStateType.REDUCED_LOAD,
                name="Reduced Load",
                description="Reduce to 50% load",
                setpoints={
                    "firing_rate": 0.5,
                    "damper_position": 45.0
                },
                minimum_hold_time_seconds=60.0,
                requires_operator_clearance=True,
                recovery_procedure="Verify sensor readings before resuming full load"
            ),
            SafeStateType.CONTROLLED_SHUTDOWN: SafeState(
                state_type=SafeStateType.CONTROLLED_SHUTDOWN,
                name="Controlled Shutdown",
                description="Execute controlled shutdown sequence",
                setpoints={
                    "firing_rate": 0.0,
                    "damper_position": 100.0,  # Full open for purge
                    "fgr_damper": 0.0
                },
                minimum_hold_time_seconds=300.0,
                requires_operator_clearance=True,
                recovery_procedure="Complete restart procedure required"
            ),
            SafeStateType.EMERGENCY_SHUTDOWN: SafeState(
                state_type=SafeStateType.EMERGENCY_SHUTDOWN,
                name="Emergency Shutdown",
                description="Immediate fuel cutoff and purge",
                setpoints={
                    "firing_rate": 0.0,
                    "fuel_valve_position": 0.0,
                    "damper_position": 100.0
                },
                minimum_hold_time_seconds=600.0,
                requires_operator_clearance=True,
                recovery_procedure="Full system inspection and restart procedure required"
            ),
        }

    def enter_fallback(
        self,
        trigger: FallbackTrigger,
        severity: FallbackSeverity,
        details: str = "",
        initiated_by: str = "SYSTEM",
        safe_state_override: Optional[SafeStateType] = None
    ) -> FallbackResult:
        """
        Enter fallback mode and transition to safe state.

        Args:
            trigger: What triggered the fallback
            severity: Severity level of the trigger
            details: Additional details about the trigger
            initiated_by: Who/what initiated the fallback
            safe_state_override: Override the default safe state selection

        Returns:
            FallbackResult with transition outcome
        """
        start_time = datetime.now(timezone.utc)

        # Preserve current state before fallback
        self._preserve_current_state()

        # Determine safe state
        safe_state_type = safe_state_override or self.DEFAULT_SAFE_STATES.get(severity, SafeStateType.HOLD)
        safe_state_def = self._safe_state_definitions.get(safe_state_type)

        if not safe_state_def:
            return FallbackResult(
                success=False,
                event=FallbackEvent(
                    trigger=trigger,
                    severity=severity,
                    safe_state=safe_state_type,
                    trigger_details=details
                ),
                safe_state_achieved=False,
                error_message=f"Unknown safe state: {safe_state_type.value}"
            )

        # Create fallback event
        event = FallbackEvent(
            trigger=trigger,
            severity=severity,
            safe_state=safe_state_type,
            trigger_details=details,
            process_values_at_trigger=dict(self._preserved_state.get("process_values", {})),
            setpoints_at_trigger=dict(self._preserved_state.get("setpoints", {})),
            initiated_by=initiated_by
        )

        try:
            # Apply safe state setpoints
            outputs_applied = self._apply_safe_state(safe_state_def)

            # Update state
            self._is_in_fallback = True
            self._current_safe_state = safe_state_type

            # Create incident
            incident = FallbackIncident(
                event=event,
                recovery_status=RecoveryStatus.NOT_STARTED
            )
            self._current_incident = incident
            self._incident_history.append(incident)
            self._total_fallbacks += 1

            transition_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = FallbackResult(
                success=True,
                event=event,
                safe_state_achieved=True,
                transition_time_seconds=transition_time,
                outputs_applied=outputs_applied
            )

            # Send notification
            self._send_notification(event)

            self._log_event("FALLBACK_ENTERED", result)
            logger.warning(f"Fallback entered: {trigger.value} - {severity.value}")

            return result

        except Exception as e:
            logger.error(f"Fallback entry failed: {str(e)}", exc_info=True)
            return FallbackResult(
                success=False,
                event=event,
                safe_state_achieved=False,
                error_message=str(e)
            )

    def revert_to_normal(
        self,
        operator_clearance: bool = False,
        operator_id: Optional[str] = None
    ) -> RevertResult:
        """
        Revert from fallback to normal operation.

        Args:
            operator_clearance: Whether operator has cleared for recovery
            operator_id: ID of the operator providing clearance

        Returns:
            RevertResult with recovery outcome
        """
        if not self._is_in_fallback:
            return RevertResult(
                success=False,
                incident_id="",
                recovery_status=RecoveryStatus.NOT_STARTED,
                previous_mode="normal",
                error_message="Not in fallback mode"
            )

        if not self._current_incident:
            return RevertResult(
                success=False,
                incident_id="",
                recovery_status=RecoveryStatus.FAILED,
                previous_mode="fallback",
                error_message="No current incident"
            )

        incident = self._current_incident
        safe_state_def = self._safe_state_definitions.get(self._current_safe_state)

        # Check clearance requirements
        if safe_state_def and safe_state_def.requires_operator_clearance and not operator_clearance:
            incident.recovery_status = RecoveryStatus.AWAITING_CLEARANCE
            return RevertResult(
                success=False,
                incident_id=incident.incident_id,
                recovery_status=RecoveryStatus.AWAITING_CLEARANCE,
                previous_mode="fallback",
                error_message="Operator clearance required"
            )

        # Check minimum hold time
        if safe_state_def:
            hold_duration = (datetime.now(timezone.utc) - incident.event.timestamp).total_seconds()
            if hold_duration < safe_state_def.minimum_hold_time_seconds:
                remaining = safe_state_def.minimum_hold_time_seconds - hold_duration
                return RevertResult(
                    success=False,
                    incident_id=incident.incident_id,
                    recovery_status=RecoveryStatus.IN_PROGRESS,
                    previous_mode="fallback",
                    error_message=f"Minimum hold time not met. {remaining:.0f}s remaining"
                )

        start_time = datetime.now(timezone.utc)
        incident.recovery_status = RecoveryStatus.IN_PROGRESS
        incident.recovery_attempts += 1

        try:
            # Restore previous setpoints
            restored = self._restore_preserved_state()

            # Update state
            self._is_in_fallback = False
            self._current_safe_state = None
            incident.recovery_status = RecoveryStatus.COMPLETED
            incident.recovered_at = datetime.now(timezone.utc)
            incident.duration_seconds = (incident.recovered_at - incident.event.timestamp).total_seconds()
            incident.resolved = True

            self._successful_recoveries += 1

            revert_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = RevertResult(
                success=True,
                incident_id=incident.incident_id,
                recovery_status=RecoveryStatus.COMPLETED,
                previous_mode="fallback",
                restored_setpoints=restored,
                revert_time_seconds=revert_time
            )

            # Execute post-recovery callback
            if self._post_recovery_callback:
                try:
                    self._post_recovery_callback()
                except Exception as e:
                    result.warnings.append(f"Post-recovery callback failed: {str(e)}")

            self._current_incident = None
            self._log_event("FALLBACK_REVERTED", result)
            logger.info(f"Fallback reverted successfully after {incident.duration_seconds:.1f}s")

            return result

        except Exception as e:
            incident.recovery_status = RecoveryStatus.FAILED
            self._failed_recoveries += 1

            logger.error(f"Fallback revert failed: {str(e)}", exc_info=True)
            return RevertResult(
                success=False,
                incident_id=incident.incident_id,
                recovery_status=RecoveryStatus.FAILED,
                previous_mode="fallback",
                error_message=str(e)
            )

    def notify_operators(self, event: FallbackEvent) -> NotificationResult:
        """
        Send notifications about a fallback event.

        Args:
            event: The fallback event to notify about

        Returns:
            NotificationResult with notification status
        """
        return self._send_notification(event)

    def define_safe_state(self, safe_state: SafeState) -> None:
        """
        Define or update a safe state configuration.

        Args:
            safe_state: The safe state definition
        """
        self._safe_state_definitions[safe_state.state_type] = safe_state
        self._log_event("SAFE_STATE_DEFINED", safe_state)
        logger.info(f"Safe state defined: {safe_state.state_type.value}")

    def get_safe_state(self, state_type: SafeStateType) -> Optional[SafeState]:
        """Get a safe state definition."""
        return self._safe_state_definitions.get(state_type)

    def is_in_fallback(self) -> bool:
        """Check if currently in fallback mode."""
        return self._is_in_fallback

    def get_current_safe_state(self) -> Optional[SafeStateType]:
        """Get the current safe state type."""
        return self._current_safe_state

    def get_current_incident(self) -> Optional[FallbackIncident]:
        """Get the current fallback incident."""
        return self._current_incident

    def register_notification_callback(
        self,
        callback: Callable[[FallbackEvent], None]
    ) -> None:
        """Register a callback for fallback notifications."""
        self._notification_callback = callback

    def register_pre_fallback_callback(
        self,
        callback: Callable[[], Dict[str, Any]]
    ) -> None:
        """Register a callback to execute before entering fallback."""
        self._pre_fallback_callback = callback

    def register_post_recovery_callback(
        self,
        callback: Callable[[], None]
    ) -> None:
        """Register a callback to execute after recovery."""
        self._post_recovery_callback = callback

    def get_incident_history(self, limit: int = 100) -> List[FallbackIncident]:
        """Get recent incident history."""
        return list(reversed(self._incident_history[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        return {
            "total_fallbacks": self._total_fallbacks,
            "successful_recoveries": self._successful_recoveries,
            "failed_recoveries": self._failed_recoveries,
            "is_in_fallback": self._is_in_fallback,
            "current_safe_state": self._current_safe_state.value if self._current_safe_state else None,
            "recovery_success_rate": (
                self._successful_recoveries / (self._successful_recoveries + self._failed_recoveries)
                if (self._successful_recoveries + self._failed_recoveries) > 0 else 0.0
            )
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        return {
            "is_in_fallback": self._is_in_fallback,
            "current_safe_state": self._current_safe_state.value if self._current_safe_state else None,
            "current_incident": self._current_incident.model_dump() if self._current_incident else None,
            "statistics": self.get_statistics()
        }

    def _preserve_current_state(self) -> None:
        """Preserve current process state before fallback."""
        if self._pre_fallback_callback:
            try:
                self._preserved_state = self._pre_fallback_callback()
            except Exception as e:
                logger.error(f"Pre-fallback callback failed: {str(e)}")
                self._preserved_state = {}
        else:
            # Default preservation
            self._preserved_state = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "process_values": {},
                "setpoints": {}
            }

    def _apply_safe_state(self, safe_state: SafeState) -> Dict[str, float]:
        """Apply safe state setpoints."""
        applied = {}
        for tag, value in safe_state.setpoints.items():
            # In real implementation, this would write to DCS
            applied[tag] = value
            logger.info(f"Safe state applied: {tag} = {value}")
        return applied

    def _restore_preserved_state(self) -> Dict[str, float]:
        """Restore preserved setpoints."""
        restored = {}
        setpoints = self._preserved_state.get("setpoints", {})
        for tag, value in setpoints.items():
            # In real implementation, this would write to DCS
            restored[tag] = value
            logger.info(f"Restored setpoint: {tag} = {value}")
        return restored

    def _send_notification(self, event: FallbackEvent) -> NotificationResult:
        """Send notification about fallback event."""
        if self._notification_callback:
            try:
                self._notification_callback(event)
                return NotificationResult(
                    success=True,
                    recipients=["operators", "supervisors"],
                    channels=["HMI", "SMS", "EMAIL"]
                )
            except Exception as e:
                return NotificationResult(
                    success=False,
                    error_message=str(e)
                )
        return NotificationResult(
            success=True,
            recipients=[],
            channels=[]
        )

    def _log_event(self, event_type: str, data: Any) -> None:
        """Log an event to the audit trail."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data.model_dump() if hasattr(data, 'model_dump') else data
        }
        self._audit_log.append(audit_entry)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the audit log."""
        return list(reversed(self._audit_log[-limit:]))
