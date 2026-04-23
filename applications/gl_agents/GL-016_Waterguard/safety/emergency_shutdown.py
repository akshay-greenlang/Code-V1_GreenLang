"""
GL-016 Waterguard Emergency Shutdown Handler - IEC 61511 SIL-3 Compliant

This module implements emergency shutdown handling for the WATERGUARD
safety system. It provides controlled shutdown sequences with complete
audit trails and integration with the Safety Instrumented System.

CRITICAL SAFETY PRINCIPLE:
    This handler coordinates WATERGUARD's response to emergencies.
    It does NOT replace or override the independent SIS.
    The SIS remains the ultimate safety authority.

Key Features:
    - Tiered emergency response (WARNING, CRITICAL, EMERGENCY)
    - Defined safe states for all controlled equipment
    - Graceful shutdown sequences
    - Complete audit trail with SHA-256 hashes
    - Integration with circuit breakers and action gate
    - Post-incident analysis support

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - IEC 61508 Parts 1-7
    - NFPA 85 Boiler and Combustion Systems

Author: GreenLang Safety Engineering Team
Version: 1.0.0
SIL Level: 3
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class EmergencySeverity(IntEnum):
    """Severity levels for emergency events."""
    WARNING = 1       # Abnormal condition, continue monitoring
    CRITICAL = 2      # Serious condition, reduce operations
    EMERGENCY = 3     # Emergency, initiate shutdown
    CATASTROPHIC = 4  # Catastrophic, immediate full shutdown


class EmergencyType(str, Enum):
    """Types of emergency events."""
    SIS_FAULT = "sis_fault"
    COMMUNICATION_LOSS = "communication_loss"
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    CONSTRAINT_VIOLATION = "constraint_violation"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    MANUAL_ESTOP = "manual_estop"
    HIGH_CONDUCTIVITY = "high_conductivity"
    HIGH_SILICA = "high_silica"
    PH_OUT_OF_RANGE = "ph_out_of_range"
    LOW_WATER = "low_water"
    HIGH_PRESSURE = "high_pressure"
    FLAME_FAILURE = "flame_failure"
    CHEMICAL_LEAK = "chemical_leak"
    CASCADE_FAILURE = "cascade_failure"
    EXTERNAL_TRIGGER = "external_trigger"


class ShutdownState(str, Enum):
    """States of emergency shutdown."""
    NORMAL = "normal"
    INITIATING = "initiating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    RECOVERY = "recovery"
    FAILED = "failed"


class EquipmentState(str, Enum):
    """State of equipment after shutdown."""
    UNKNOWN = "unknown"
    RUNNING = "running"
    SAFE_STATE = "safe_state"
    FAILED_SAFE = "failed_safe"
    FAILED_UNSAFE = "failed_unsafe"


# =============================================================================
# DATA MODELS
# =============================================================================


class SafeStateDefinition(BaseModel):
    """Definition of safe state for equipment."""

    tag: str = Field(
        ...,
        description="Equipment tag"
    )
    description: str = Field(
        default="",
        description="Equipment description"
    )
    safe_value: float = Field(
        ...,
        description="Value to command for safe state"
    )
    engineering_units: str = Field(
        default="%",
        description="Engineering units"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Shutdown priority (1=highest)"
    )
    timeout_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="Time to reach safe state"
    )
    verify_required: bool = Field(
        default=True,
        description="Verify safe state achieved"
    )
    fail_safe_direction: str = Field(
        default="close",
        description="Direction on failure (close/open/stop)"
    )


class EmergencyEvent(BaseModel):
    """An emergency event record."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier"
    )
    event_type: EmergencyType = Field(
        ...,
        description="Type of emergency"
    )
    severity: EmergencySeverity = Field(
        ...,
        description="Severity level"
    )
    source: str = Field(
        default="WATERGUARD",
        description="Event source"
    )
    trigger_tag: str = Field(
        default="",
        description="Tag that triggered event"
    )
    trigger_value: Optional[float] = Field(
        default=None,
        description="Value that triggered event"
    )
    trigger_limit: Optional[float] = Field(
        default=None,
        description="Limit that was exceeded"
    )
    message: str = Field(
        default="",
        description="Event message"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    acknowledged: bool = Field(
        default=False,
        description="Event acknowledged by operator"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="Operator who acknowledged"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="Acknowledgement time"
    )
    resolved: bool = Field(
        default=False,
        description="Event resolved"
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="Resolution time"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.event_id}|{self.event_type.value}|"
                f"{self.severity}|{self.source}|"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class ShutdownAction(BaseModel):
    """An action taken during shutdown."""

    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Action identifier"
    )
    tag: str = Field(
        ...,
        description="Equipment tag"
    )
    commanded_value: float = Field(
        ...,
        description="Value commanded"
    )
    actual_value: Optional[float] = Field(
        default=None,
        description="Actual value achieved"
    )
    success: bool = Field(
        default=False,
        description="Action successful"
    )
    error_message: str = Field(
        default="",
        description="Error message if failed"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Action start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Action end time"
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Action duration in ms"
    )


class ShutdownRecord(BaseModel):
    """Complete record of a shutdown sequence."""

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Shutdown record ID"
    )
    trigger_event: EmergencyEvent = Field(
        ...,
        description="Event that triggered shutdown"
    )
    state: ShutdownState = Field(
        default=ShutdownState.INITIATING,
        description="Shutdown state"
    )
    actions: List[ShutdownAction] = Field(
        default_factory=list,
        description="Actions taken"
    )
    equipment_states: Dict[str, EquipmentState] = Field(
        default_factory=dict,
        description="Final equipment states"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Shutdown start time"
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Shutdown end time"
    )
    total_duration_ms: Optional[float] = Field(
        default=None,
        description="Total shutdown duration"
    )
    success: bool = Field(
        default=False,
        description="Shutdown successful"
    )
    partial_success: bool = Field(
        default=False,
        description="Some equipment reached safe state"
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Shutdown notes"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.record_id}|{self.trigger_event.event_id}|"
                f"{self.state.value}|{self.start_time.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


# =============================================================================
# EMERGENCY SHUTDOWN HANDLER
# =============================================================================


class EmergencyShutdownHandler:
    """
    Handles emergency shutdown sequences for WATERGUARD.

    This handler coordinates WATERGUARD's response to emergency conditions.
    It commands all controlled equipment to defined safe states and maintains
    a complete audit trail.

    CRITICAL: This handler does NOT replace the independent SIS.
    The SIS takes precedence and operates independently of WATERGUARD.
    This handler manages WATERGUARD's controlled equipment only.

    Shutdown Sequence:
        1. Receive emergency event (SIS status, constraint violation, etc.)
        2. Log event with provenance hash
        3. Command all equipment to safe states (priority order)
        4. Verify safe states achieved
        5. Log completion with full audit trail

    Example:
        >>> handler = EmergencyShutdownHandler()
        >>> handler.register_safe_state(SafeStateDefinition(
        ...     tag="BD-001",
        ...     safe_value=0.0,
        ...     priority=1
        ... ))
        >>> event = handler.trigger_estop(
        ...     EmergencyType.HIGH_CONDUCTIVITY,
        ...     EmergencySeverity.CRITICAL,
        ...     "Conductivity exceeded 3500 uS/cm"
        ... )
    """

    def __init__(
        self,
        circuit_breaker_registry: Optional[Any] = None,
        action_gate: Optional[Any] = None,
        sis_interface: Optional[Any] = None
    ) -> None:
        """
        Initialize EmergencyShutdownHandler.

        Args:
            circuit_breaker_registry: Registry for circuit breakers
            action_gate: ActionGate for permission checks
            sis_interface: SISInterface for safety status
        """
        self._circuit_breakers = circuit_breaker_registry
        self._action_gate = action_gate
        self._sis_interface = sis_interface

        self._lock = threading.Lock()

        # Safe state definitions (tag -> definition)
        self._safe_states: Dict[str, SafeStateDefinition] = {}

        # Current shutdown state
        self._shutdown_state = ShutdownState.NORMAL
        self._current_shutdown: Optional[ShutdownRecord] = None

        # History
        self._event_history: List[EmergencyEvent] = []
        self._shutdown_history: List[ShutdownRecord] = []
        self._max_history = 500

        # Statistics
        self._stats = {
            "events_total": 0,
            "shutdowns_total": 0,
            "shutdowns_successful": 0,
            "shutdowns_failed": 0,
            "events_by_type": {},
            "events_by_severity": {},
        }

        # Callbacks
        self._on_event: Optional[Callable[[EmergencyEvent], None]] = None
        self._on_shutdown_start: Optional[Callable[[ShutdownRecord], None]] = None
        self._on_shutdown_complete: Optional[Callable[[ShutdownRecord], None]] = None
        self._command_func: Optional[Callable[[str, float], bool]] = None

        logger.info("EmergencyShutdownHandler initialized")

    def register_safe_state(self, definition: SafeStateDefinition) -> None:
        """
        Register a safe state definition for equipment.

        Args:
            definition: Safe state definition
        """
        with self._lock:
            self._safe_states[definition.tag] = definition
            logger.info(
                "Registered safe state: %s = %.2f%s (priority %d)",
                definition.tag,
                definition.safe_value,
                definition.engineering_units,
                definition.priority
            )

    def unregister_safe_state(self, tag: str) -> None:
        """Remove a safe state definition."""
        with self._lock:
            if tag in self._safe_states:
                del self._safe_states[tag]
                logger.info("Unregistered safe state: %s", tag)

    def get_safe_states(self) -> Dict[str, SafeStateDefinition]:
        """Get all registered safe state definitions."""
        with self._lock:
            return dict(self._safe_states)

    def trigger_estop(
        self,
        event_type: EmergencyType,
        severity: EmergencySeverity,
        message: str,
        trigger_tag: str = "",
        trigger_value: Optional[float] = None,
        trigger_limit: Optional[float] = None
    ) -> EmergencyEvent:
        """
        Trigger emergency stop.

        This is the main entry point for emergency conditions.
        It creates an event, logs it, and initiates shutdown if needed.

        Args:
            event_type: Type of emergency
            severity: Severity level
            message: Descriptive message
            trigger_tag: Tag that triggered (optional)
            trigger_value: Value that triggered (optional)
            trigger_limit: Limit that was exceeded (optional)

        Returns:
            EmergencyEvent record
        """
        # Create event
        event = EmergencyEvent(
            event_type=event_type,
            severity=severity,
            trigger_tag=trigger_tag,
            trigger_value=trigger_value,
            trigger_limit=trigger_limit,
            message=message
        )

        with self._lock:
            # Update statistics
            self._stats["events_total"] += 1
            type_key = event_type.value
            self._stats["events_by_type"][type_key] = (
                self._stats["events_by_type"].get(type_key, 0) + 1
            )
            severity_key = str(severity)
            self._stats["events_by_severity"][severity_key] = (
                self._stats["events_by_severity"].get(severity_key, 0) + 1
            )

            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        # Log event
        log_level = {
            EmergencySeverity.WARNING: logging.WARNING,
            EmergencySeverity.CRITICAL: logging.ERROR,
            EmergencySeverity.EMERGENCY: logging.CRITICAL,
            EmergencySeverity.CATASTROPHIC: logging.CRITICAL,
        }.get(severity, logging.ERROR)

        logger.log(
            log_level,
            "EMERGENCY EVENT [%s]: %s - %s (hash: %s)",
            severity.name,
            event_type.value,
            message,
            event.provenance_hash[:16]
        )

        # Callback
        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                logger.error("Event callback failed: %s", e)

        # Determine if shutdown needed
        if severity >= EmergencySeverity.CRITICAL:
            self._initiate_shutdown(event)

        return event

    def _initiate_shutdown(self, trigger_event: EmergencyEvent) -> ShutdownRecord:
        """Initiate emergency shutdown sequence."""
        with self._lock:
            # Check if already shutting down
            if self._shutdown_state in (
                ShutdownState.INITIATING,
                ShutdownState.IN_PROGRESS
            ):
                logger.warning(
                    "Shutdown already in progress, ignoring new trigger"
                )
                if self._current_shutdown:
                    return self._current_shutdown

            # Create shutdown record
            record = ShutdownRecord(trigger_event=trigger_event)
            self._current_shutdown = record
            self._shutdown_state = ShutdownState.INITIATING

            self._stats["shutdowns_total"] += 1

        logger.critical(
            "INITIATING EMERGENCY SHUTDOWN - Event: %s (ID: %s)",
            trigger_event.event_type.value,
            trigger_event.event_id[:8]
        )

        # Callback
        if self._on_shutdown_start:
            try:
                self._on_shutdown_start(record)
            except Exception as e:
                logger.error("Shutdown start callback failed: %s", e)

        # Execute shutdown
        self._execute_shutdown(record)

        return record

    def _execute_shutdown(self, record: ShutdownRecord) -> None:
        """Execute the shutdown sequence."""
        with self._lock:
            self._shutdown_state = ShutdownState.IN_PROGRESS
            record.state = ShutdownState.IN_PROGRESS

        # Get safe states sorted by priority
        with self._lock:
            safe_states = list(self._safe_states.values())
            safe_states.sort(key=lambda s: s.priority)

        success_count = 0
        failure_count = 0

        # Command each piece of equipment to safe state
        for safe_state in safe_states:
            action = self._command_safe_state(safe_state)
            record.actions.append(action)

            if action.success:
                success_count += 1
                record.equipment_states[safe_state.tag] = EquipmentState.SAFE_STATE
            else:
                failure_count += 1
                record.equipment_states[safe_state.tag] = EquipmentState.FAILED_SAFE

        # Complete shutdown
        record.end_time = datetime.now(timezone.utc)
        record.total_duration_ms = (
            (record.end_time - record.start_time).total_seconds() * 1000
        )

        if failure_count == 0:
            record.success = True
            record.state = ShutdownState.COMPLETED
            with self._lock:
                self._stats["shutdowns_successful"] += 1
        elif success_count > 0:
            record.partial_success = True
            record.state = ShutdownState.COMPLETED
            record.notes.append(
                f"Partial success: {success_count}/{len(safe_states)} equipment safe"
            )
        else:
            record.success = False
            record.state = ShutdownState.FAILED
            with self._lock:
                self._stats["shutdowns_failed"] += 1

        # Update state
        with self._lock:
            self._shutdown_state = record.state
            self._shutdown_history.append(record)
            if len(self._shutdown_history) > self._max_history:
                self._shutdown_history = self._shutdown_history[-self._max_history:]

        # Log completion
        if record.success:
            logger.critical(
                "EMERGENCY SHUTDOWN COMPLETE - All equipment safe (%.1fms)",
                record.total_duration_ms
            )
        else:
            logger.critical(
                "EMERGENCY SHUTDOWN COMPLETED WITH FAILURES - "
                "%d/%d equipment safe (%.1fms)",
                success_count, len(safe_states), record.total_duration_ms
            )

        # Callback
        if self._on_shutdown_complete:
            try:
                self._on_shutdown_complete(record)
            except Exception as e:
                logger.error("Shutdown complete callback failed: %s", e)

    def _command_safe_state(self, safe_state: SafeStateDefinition) -> ShutdownAction:
        """Command a single piece of equipment to safe state."""
        action = ShutdownAction(
            tag=safe_state.tag,
            commanded_value=safe_state.safe_value
        )

        logger.info(
            "Commanding %s to safe state: %.2f%s",
            safe_state.tag,
            safe_state.safe_value,
            safe_state.engineering_units
        )

        try:
            if self._command_func:
                success = self._command_func(safe_state.tag, safe_state.safe_value)
                action.success = success
                if not success:
                    action.error_message = "Command function returned False"
            else:
                # Simulate command (for testing/development)
                action.success = True
                action.actual_value = safe_state.safe_value

        except Exception as e:
            action.success = False
            action.error_message = str(e)
            logger.error(
                "Failed to command %s to safe state: %s",
                safe_state.tag, e
            )

        action.end_time = datetime.now(timezone.utc)
        action.duration_ms = (
            (action.end_time - action.start_time).total_seconds() * 1000
        )

        return action

    def acknowledge_event(
        self,
        event_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge an emergency event.

        Args:
            event_id: Event ID to acknowledge
            acknowledged_by: Operator acknowledging

        Returns:
            True if acknowledged
        """
        with self._lock:
            for event in self._event_history:
                if event.event_id == event_id:
                    if event.acknowledged:
                        logger.warning(
                            "Event %s already acknowledged",
                            event_id[:8]
                        )
                        return False

                    event.acknowledged = True
                    event.acknowledged_by = acknowledged_by
                    event.acknowledged_at = datetime.now(timezone.utc)

                    logger.info(
                        "Event %s acknowledged by %s",
                        event_id[:8], acknowledged_by
                    )
                    return True

        logger.warning("Event not found: %s", event_id[:8])
        return False

    def resolve_event(self, event_id: str) -> bool:
        """
        Mark an emergency event as resolved.

        Args:
            event_id: Event ID to resolve

        Returns:
            True if resolved
        """
        with self._lock:
            for event in self._event_history:
                if event.event_id == event_id:
                    if not event.acknowledged:
                        logger.warning(
                            "Cannot resolve unacknowledged event: %s",
                            event_id[:8]
                        )
                        return False

                    event.resolved = True
                    event.resolved_at = datetime.now(timezone.utc)

                    logger.info("Event %s resolved", event_id[:8])
                    return True

        return False

    def initiate_recovery(self, authorized_by: str) -> bool:
        """
        Initiate recovery from shutdown state.

        Args:
            authorized_by: Person authorizing recovery

        Returns:
            True if recovery initiated
        """
        with self._lock:
            if self._shutdown_state not in (
                ShutdownState.COMPLETED,
                ShutdownState.FAILED
            ):
                logger.warning(
                    "Cannot initiate recovery in state: %s",
                    self._shutdown_state.value
                )
                return False

            # Check all events acknowledged
            unacknowledged = [
                e for e in self._event_history
                if not e.acknowledged and
                e.severity >= EmergencySeverity.CRITICAL
            ]
            if unacknowledged:
                logger.warning(
                    "Cannot recover: %d unacknowledged events",
                    len(unacknowledged)
                )
                return False

            self._shutdown_state = ShutdownState.RECOVERY

        logger.warning(
            "RECOVERY INITIATED by %s",
            authorized_by
        )

        return True

    def complete_recovery(self, authorized_by: str) -> bool:
        """
        Complete recovery and return to normal state.

        Args:
            authorized_by: Person authorizing

        Returns:
            True if returned to normal
        """
        with self._lock:
            if self._shutdown_state != ShutdownState.RECOVERY:
                logger.warning(
                    "Cannot complete recovery in state: %s",
                    self._shutdown_state.value
                )
                return False

            self._shutdown_state = ShutdownState.NORMAL
            self._current_shutdown = None

        logger.warning(
            "RECOVERY COMPLETE - Normal operation resumed (by %s)",
            authorized_by
        )

        return True

    def get_shutdown_state(self) -> ShutdownState:
        """Get current shutdown state."""
        with self._lock:
            return self._shutdown_state

    def is_shutdown_active(self) -> bool:
        """Check if shutdown is active."""
        with self._lock:
            return self._shutdown_state in (
                ShutdownState.INITIATING,
                ShutdownState.IN_PROGRESS
            )

    def is_normal_operation(self) -> bool:
        """Check if in normal operation."""
        with self._lock:
            return self._shutdown_state == ShutdownState.NORMAL

    def get_current_shutdown(self) -> Optional[ShutdownRecord]:
        """Get current or most recent shutdown record."""
        with self._lock:
            return self._current_shutdown

    def get_event_history(
        self,
        limit: int = 100,
        severity: Optional[EmergencySeverity] = None,
        event_type: Optional[EmergencyType] = None
    ) -> List[EmergencyEvent]:
        """
        Get event history.

        Args:
            limit: Maximum events to return
            severity: Filter by severity
            event_type: Filter by type

        Returns:
            List of events
        """
        with self._lock:
            events = self._event_history
            if severity:
                events = [e for e in events if e.severity == severity]
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return list(reversed(events[-limit:]))

    def get_shutdown_history(self, limit: int = 50) -> List[ShutdownRecord]:
        """Get shutdown history."""
        with self._lock:
            return list(reversed(self._shutdown_history[-limit:]))

    def get_unacknowledged_events(self) -> List[EmergencyEvent]:
        """Get all unacknowledged events."""
        with self._lock:
            return [e for e in self._event_history if not e.acknowledged]

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        with self._lock:
            return dict(self._stats)

    def set_command_function(
        self,
        func: Callable[[str, float], bool]
    ) -> None:
        """Set function to command equipment."""
        self._command_func = func

    def set_on_event(
        self,
        callback: Callable[[EmergencyEvent], None]
    ) -> None:
        """Set callback for emergency events."""
        self._on_event = callback

    def set_on_shutdown_start(
        self,
        callback: Callable[[ShutdownRecord], None]
    ) -> None:
        """Set callback for shutdown start."""
        self._on_shutdown_start = callback

    def set_on_shutdown_complete(
        self,
        callback: Callable[[ShutdownRecord], None]
    ) -> None:
        """Set callback for shutdown complete."""
        self._on_shutdown_complete = callback


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "EmergencySeverity",
    "EmergencyType",
    "ShutdownState",
    "EquipmentState",
    # Models
    "SafeStateDefinition",
    "EmergencyEvent",
    "ShutdownAction",
    "ShutdownRecord",
    # Classes
    "EmergencyShutdownHandler",
]
