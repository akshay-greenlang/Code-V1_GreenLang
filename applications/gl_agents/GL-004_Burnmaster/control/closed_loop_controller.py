"""
GL-004 BURNMASTER - Closed Loop Controller

This module implements the closed-loop control system for the burner management
control system. It provides automatic setpoint writing within bounded safety
envelopes with automatic fallback on anomaly detection.

Key Features:
    - Automatic control cycle execution
    - Bounded safety envelope enforcement
    - Write verification with read-back
    - Automatic fallback on anomalies
    - Complete audit trail with before/after states

Operating Mode: CLOSED_LOOP
    - Write setpoints within bounded envelope
    - Auto-fallback on anomalies
    - Continuous process optimization

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - NFPA 85/86 Boiler/Furnace Standards

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BurnerState(str, Enum):
    """Operational state of the burner."""
    OFF = "off"
    PURGE = "purge"
    IGNITION = "ignition"
    LOW_FIRE = "low_fire"
    MODULATING = "modulating"
    HIGH_FIRE = "high_fire"
    SHUTDOWN = "shutdown"
    FAULT = "fault"


class WriteStatus(str, Enum):
    """Status of a write operation."""
    PENDING = "pending"
    WRITING = "writing"
    VERIFYING = "verifying"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class AnomalyType(str, Enum):
    """Types of anomalies that trigger fallback."""
    COMMUNICATION_FAILURE = "communication_failure"
    VERIFICATION_FAILURE = "verification_failure"
    SAFETY_LIMIT_BREACH = "safety_limit_breach"
    SENSOR_FAULT = "sensor_fault"
    ACTUATOR_FAULT = "actuator_fault"
    PROCESS_INSTABILITY = "process_instability"
    BMS_INTERLOCK = "bms_interlock"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class RecoveryAction(str, Enum):
    """Actions for recovery from anomalies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    HOLD = "hold"
    ALERT = "alert"
    EMERGENCY_STOP = "emergency_stop"


class WriteResult(BaseModel):
    """Result of a write operation to DCS."""
    write_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tag_name: str
    value_before: float
    value_written: float
    value_verified: Optional[float] = None
    status: WriteStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    write_duration_ms: float = Field(default=0.0, ge=0)
    verification_passed: bool = Field(default=False)
    error_message: Optional[str] = None
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.write_id}|{self.tag_name}|{self.value_before}|{self.value_written}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class VerificationResult(BaseModel):
    """Result of verifying a write operation."""
    verification_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    write_id: str
    tag_name: str
    expected_value: float
    actual_value: float
    tolerance: float = Field(default=0.01)
    passed: bool
    deviation: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WriteFailure(BaseModel):
    """Record of a write failure."""
    failure_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    write_id: str
    tag_name: str
    failure_type: str
    error_message: str
    anomaly_type: Optional[AnomalyType] = None
    recovery_action: RecoveryAction
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = Field(default=False)
    resolution_timestamp: Optional[datetime] = None


class ControlCycleResult(BaseModel):
    """Result of a control cycle execution."""
    cycle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cycle_number: int
    started_at: datetime
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = Field(default=0.0, ge=0)
    writes_attempted: int = Field(default=0, ge=0)
    writes_successful: int = Field(default=0, ge=0)
    writes_failed: int = Field(default=0, ge=0)
    write_results: List[WriteResult] = Field(default_factory=list)
    anomalies_detected: List[AnomalyType] = Field(default_factory=list)
    fallback_triggered: bool = Field(default=False)
    burner_state: BurnerState
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.cycle_id}|{self.cycle_number}|{self.writes_successful}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class ControlOutput(BaseModel):
    """A control output to be written to DCS."""
    tag_name: str
    current_value: float
    target_value: float
    min_value: float
    max_value: float
    rate_limit: float = Field(default=1.0, gt=0)
    unit: str = Field(default="")
    priority: int = Field(default=1, ge=1)


class ClosedLoopController:
    """
    Controls closed-loop operation of the burner management system.

    This class manages automatic control including:
    - Control cycle execution
    - Setpoint writing with verification
    - Anomaly detection and fallback
    - Safety envelope enforcement

    Example:
        >>> controller = ClosedLoopController()
        >>> result = controller.execute_control_cycle()
        >>> if result.fallback_triggered:
        ...     handle_fallback()
    """

    # Default control parameters
    DEFAULT_CYCLE_INTERVAL_MS = 1000.0
    DEFAULT_WRITE_TIMEOUT_MS = 500.0
    DEFAULT_VERIFICATION_DELAY_MS = 100.0
    DEFAULT_VERIFICATION_TOLERANCE = 0.01

    def __init__(self) -> None:
        """Initialize the closed-loop controller."""
        self._is_active = False
        self._cycle_count = 0
        self._burner_state = BurnerState.OFF
        self._current_outputs: Dict[str, ControlOutput] = {}
        self._write_history: List[WriteResult] = []
        self._failure_history: List[WriteFailure] = []
        self._cycle_history: List[ControlCycleResult] = []
        self._audit_log: List[Dict[str, Any]] = []

        # Callbacks
        self._fallback_callback: Optional[Callable[[], None]] = None
        self._anomaly_handlers: Dict[AnomalyType, Callable[[AnomalyType], RecoveryAction]] = {}

        # Statistics
        self._total_writes = 0
        self._successful_writes = 0
        self._failed_writes = 0
        self._fallback_count = 0

        # Initialize default outputs
        self._initialize_default_outputs()

        logger.info("ClosedLoopController initialized")

    def _initialize_default_outputs(self) -> None:
        """Initialize default control outputs."""
        self._current_outputs = {
            "damper_position": ControlOutput(
                tag_name="damper_position",
                current_value=50.0,
                target_value=50.0,
                min_value=0.0,
                max_value=100.0,
                rate_limit=2.0,
                unit="%"
            ),
            "fuel_valve_position": ControlOutput(
                tag_name="fuel_valve_position",
                current_value=30.0,
                target_value=30.0,
                min_value=0.0,
                max_value=100.0,
                rate_limit=1.5,
                unit="%"
            ),
            "fgr_damper": ControlOutput(
                tag_name="fgr_damper",
                current_value=10.0,
                target_value=10.0,
                min_value=0.0,
                max_value=30.0,
                rate_limit=0.5,
                unit="%"
            ),
        }

    def activate(self) -> bool:
        """
        Activate closed-loop control.

        Returns:
            True if activation successful
        """
        if self._is_active:
            logger.warning("ClosedLoopController already active")
            return True

        # Verify safe to activate
        if self._burner_state == BurnerState.FAULT:
            logger.error("Cannot activate: burner in fault state")
            return False

        self._is_active = True
        self._log_event("CLOSED_LOOP_ACTIVATED", {"burner_state": self._burner_state.value})
        logger.info("Closed-loop control activated")

        return True

    def deactivate(self) -> bool:
        """
        Deactivate closed-loop control.

        Returns:
            True if deactivation successful
        """
        if not self._is_active:
            return True

        self._is_active = False
        self._log_event("CLOSED_LOOP_DEACTIVATED", {"cycle_count": self._cycle_count})
        logger.info("Closed-loop control deactivated")

        return True

    def execute_control_cycle(self) -> ControlCycleResult:
        """
        Execute one control cycle.

        Returns:
            ControlCycleResult with cycle outcome
        """
        start_time = datetime.now(timezone.utc)
        self._cycle_count += 1

        write_results = []
        anomalies = []
        fallback_triggered = False

        try:
            # Check preconditions
            if not self._is_active:
                return ControlCycleResult(
                    cycle_number=self._cycle_count,
                    started_at=start_time,
                    writes_attempted=0,
                    writes_successful=0,
                    writes_failed=0,
                    burner_state=self._burner_state
                )

            # Execute writes for each output
            for output in self._current_outputs.values():
                if output.current_value != output.target_value:
                    result = self._execute_write(output)
                    write_results.append(result)

                    if result.status == WriteStatus.FAILED:
                        # Check for anomalies
                        anomaly = self._detect_anomaly(result)
                        if anomaly:
                            anomalies.append(anomaly)
                            recovery = self._handle_anomaly(anomaly)
                            if recovery == RecoveryAction.FALLBACK:
                                fallback_triggered = True
                                break

            # Check for process anomalies
            process_anomaly = self._check_process_stability()
            if process_anomaly:
                anomalies.append(process_anomaly)
                if self._handle_anomaly(process_anomaly) == RecoveryAction.FALLBACK:
                    fallback_triggered = True

        except Exception as e:
            logger.error(f"Control cycle error: {str(e)}", exc_info=True)
            anomalies.append(AnomalyType.COMMUNICATION_FAILURE)
            fallback_triggered = True

        # Calculate statistics
        writes_successful = sum(1 for r in write_results if r.status == WriteStatus.SUCCESS)
        writes_failed = sum(1 for r in write_results if r.status == WriteStatus.FAILED)

        # Update totals
        self._total_writes += len(write_results)
        self._successful_writes += writes_successful
        self._failed_writes += writes_failed

        if fallback_triggered:
            self._fallback_count += 1
            self._trigger_fallback()

        duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        result = ControlCycleResult(
            cycle_number=self._cycle_count,
            started_at=start_time,
            duration_ms=duration,
            writes_attempted=len(write_results),
            writes_successful=writes_successful,
            writes_failed=writes_failed,
            write_results=write_results,
            anomalies_detected=anomalies,
            fallback_triggered=fallback_triggered,
            burner_state=self._burner_state
        )

        self._cycle_history.append(result)
        self._log_event("CONTROL_CYCLE", result)

        return result

    def write_setpoint(
        self,
        tag_name: str,
        value: float,
        verify: bool = True
    ) -> WriteResult:
        """
        Write a setpoint value to DCS.

        Args:
            tag_name: Name of the tag to write
            value: Value to write
            verify: Whether to verify the write

        Returns:
            WriteResult with write outcome
        """
        output = self._current_outputs.get(tag_name)
        if not output:
            return WriteResult(
                tag_name=tag_name,
                value_before=0.0,
                value_written=value,
                status=WriteStatus.REJECTED,
                error_message=f"Unknown tag: {tag_name}"
            )

        # Enforce safety envelope
        clamped_value = max(output.min_value, min(output.max_value, value))
        if clamped_value != value:
            logger.warning(f"Value {value} clamped to {clamped_value} for {tag_name}")

        # Update target
        output.target_value = clamped_value

        # Execute write
        return self._execute_write(output, verify)

    def verify_write(self, write_result: WriteResult) -> VerificationResult:
        """
        Verify a write operation by reading back the value.

        Args:
            write_result: The write result to verify

        Returns:
            VerificationResult with verification outcome
        """
        # Simulate read-back (in real implementation, this reads from DCS)
        output = self._current_outputs.get(write_result.tag_name)
        actual_value = output.current_value if output else write_result.value_written

        expected = write_result.value_written
        deviation = abs(actual_value - expected)
        tolerance = self.DEFAULT_VERIFICATION_TOLERANCE * abs(expected) if expected != 0 else 0.01
        passed = deviation <= tolerance

        verification = VerificationResult(
            write_id=write_result.write_id,
            tag_name=write_result.tag_name,
            expected_value=expected,
            actual_value=actual_value,
            tolerance=tolerance,
            passed=passed,
            deviation=deviation
        )

        write_result.value_verified = actual_value
        write_result.verification_passed = passed

        if not passed:
            logger.warning(f"Verification failed for {write_result.tag_name}: expected {expected}, got {actual_value}")

        return verification

    def handle_write_failure(self, failure: WriteFailure) -> RecoveryAction:
        """
        Handle a write failure and determine recovery action.

        Args:
            failure: The write failure to handle

        Returns:
            RecoveryAction to take
        """
        self._failure_history.append(failure)

        # Determine recovery action based on anomaly type
        if failure.anomaly_type == AnomalyType.SAFETY_LIMIT_BREACH:
            return RecoveryAction.FALLBACK
        elif failure.anomaly_type == AnomalyType.COMMUNICATION_FAILURE:
            # Check retry count
            recent_failures = self._count_recent_failures(failure.tag_name)
            if recent_failures < 3:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.FALLBACK
        elif failure.anomaly_type == AnomalyType.BMS_INTERLOCK:
            return RecoveryAction.EMERGENCY_STOP
        else:
            return RecoveryAction.HOLD

    def trigger_fallback(self, reason: str = "") -> bool:
        """
        Trigger fallback to safe state.

        Args:
            reason: Explanation for fallback

        Returns:
            True if fallback successful
        """
        logger.warning(f"Fallback triggered: {reason}")
        self._fallback_count += 1
        return self._trigger_fallback()

    def get_burner_state(self) -> BurnerState:
        """Get current burner operational state."""
        return self._burner_state

    def set_burner_state(self, state: BurnerState) -> None:
        """Set the burner operational state."""
        old_state = self._burner_state
        self._burner_state = state
        self._log_event("BURNER_STATE_CHANGED", {
            "old_state": old_state.value,
            "new_state": state.value
        })

    def register_fallback_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback for fallback events."""
        self._fallback_callback = callback

    def register_anomaly_handler(
        self,
        anomaly_type: AnomalyType,
        handler: Callable[[AnomalyType], RecoveryAction]
    ) -> None:
        """Register a handler for specific anomaly types."""
        self._anomaly_handlers[anomaly_type] = handler

    def get_output(self, tag_name: str) -> Optional[ControlOutput]:
        """Get a control output by tag name."""
        return self._current_outputs.get(tag_name)

    def set_output_target(self, tag_name: str, target_value: float) -> bool:
        """Set the target value for a control output."""
        output = self._current_outputs.get(tag_name)
        if not output:
            return False

        # Clamp to safety limits
        clamped = max(output.min_value, min(output.max_value, target_value))
        output.target_value = clamped

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get control statistics."""
        return {
            "total_cycles": self._cycle_count,
            "total_writes": self._total_writes,
            "successful_writes": self._successful_writes,
            "failed_writes": self._failed_writes,
            "fallback_count": self._fallback_count,
            "success_rate": self._successful_writes / self._total_writes if self._total_writes > 0 else 0.0,
            "is_active": self._is_active
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        return {
            "is_active": self._is_active,
            "cycle_count": self._cycle_count,
            "burner_state": self._burner_state.value,
            "statistics": self.get_statistics(),
            "outputs": {k: v.model_dump() for k, v in self._current_outputs.items()}
        }

    def get_write_history(self, limit: int = 100) -> List[WriteResult]:
        """Get recent write history."""
        return list(reversed(self._write_history[-limit:]))

    def get_failure_history(self, limit: int = 100) -> List[WriteFailure]:
        """Get recent failure history."""
        return list(reversed(self._failure_history[-limit:]))

    def get_cycle_history(self, limit: int = 100) -> List[ControlCycleResult]:
        """Get recent cycle history."""
        return list(reversed(self._cycle_history[-limit:]))

    def _execute_write(self, output: ControlOutput, verify: bool = True) -> WriteResult:
        """Execute a write operation for an output."""
        start_time = datetime.now(timezone.utc)
        value_before = output.current_value
        value_to_write = output.target_value

        try:
            # Simulate write operation
            # In real implementation, this would write to DCS
            output.current_value = value_to_write

            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            result = WriteResult(
                tag_name=output.tag_name,
                value_before=value_before,
                value_written=value_to_write,
                status=WriteStatus.SUCCESS,
                write_duration_ms=duration
            )

            # Verify if requested
            if verify:
                verification = self.verify_write(result)
                if not verification.passed:
                    result.status = WriteStatus.FAILED
                    result.error_message = "Verification failed"

            self._write_history.append(result)
            return result

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result = WriteResult(
                tag_name=output.tag_name,
                value_before=value_before,
                value_written=value_to_write,
                status=WriteStatus.FAILED,
                write_duration_ms=duration,
                error_message=str(e)
            )
            self._write_history.append(result)
            return result

    def _detect_anomaly(self, write_result: WriteResult) -> Optional[AnomalyType]:
        """Detect anomaly from write result."""
        if write_result.status == WriteStatus.FAILED:
            if "verification" in (write_result.error_message or "").lower():
                return AnomalyType.VERIFICATION_FAILURE
            elif "timeout" in (write_result.error_message or "").lower():
                return AnomalyType.COMMUNICATION_FAILURE
            else:
                return AnomalyType.ACTUATOR_FAULT
        return None

    def _handle_anomaly(self, anomaly: AnomalyType) -> RecoveryAction:
        """Handle an anomaly and return recovery action."""
        # Check for custom handler
        handler = self._anomaly_handlers.get(anomaly)
        if handler:
            return handler(anomaly)

        # Default handling
        if anomaly in [AnomalyType.SAFETY_LIMIT_BREACH, AnomalyType.BMS_INTERLOCK]:
            return RecoveryAction.FALLBACK
        elif anomaly == AnomalyType.COMMUNICATION_FAILURE:
            return RecoveryAction.RETRY
        else:
            return RecoveryAction.HOLD

    def _check_process_stability(self) -> Optional[AnomalyType]:
        """Check for process stability anomalies."""
        # In real implementation, this would analyze process trends
        return None

    def _trigger_fallback(self) -> bool:
        """Trigger fallback to safe state."""
        self._is_active = False
        self._log_event("FALLBACK_TRIGGERED", {"cycle_count": self._cycle_count})

        if self._fallback_callback:
            try:
                self._fallback_callback()
            except Exception as e:
                logger.error(f"Fallback callback failed: {str(e)}")

        return True

    def _count_recent_failures(self, tag_name: str, window_seconds: float = 60.0) -> int:
        """Count recent failures for a tag."""
        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
        return sum(
            1 for f in self._failure_history
            if f.tag_name == tag_name and f.timestamp.timestamp() > cutoff
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
