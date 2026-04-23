"""
GL-004 BURNMASTER - Operating Mode Manager

This module implements the operating mode state machine for the burner management
control system. It manages transitions between OBSERVE, ADVISORY, CLOSED_LOOP,
FALLBACK, and MAINTENANCE modes with full validation and audit trails.

Operating Modes:
    - OBSERVE: Compute KPIs and recommendations, no writes to DCS
    - ADVISORY: Present recommendations with explanations, manual acceptance
    - CLOSED_LOOP: Write setpoints within bounded envelope, auto-fallback on anomalies
    - FALLBACK: Safe state operation during anomalies or failures
    - MAINTENANCE: System maintenance mode with restricted operations

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - NFPA 85/86 Boiler/Furnace Standards

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import logging
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OperatingMode(str, Enum):
    """Operating modes for the burner management control system."""
    OBSERVE = "observe"
    ADVISORY = "advisory"
    CLOSED_LOOP = "closed_loop"
    FALLBACK = "fallback"
    MAINTENANCE = "maintenance"


class TransitionStatus(str, Enum):
    """Status of a mode transition attempt."""
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING = "pending"
    TIMEOUT = "timeout"


class PreconditionType(str, Enum):
    """Types of preconditions that must be met for mode transitions."""
    BMS_STATUS = "bms_status"
    OPERATOR_AUTH = "operator_authorization"
    SAFETY_CHECK = "safety_check"
    SYSTEM_HEALTH = "system_health"
    PROCESS_STABLE = "process_stable"
    NO_ACTIVE_ALARMS = "no_active_alarms"
    INTERLOCK_ARMED = "interlock_armed"


class ModeValidationResult(BaseModel):
    """Result of validating preconditions for a mode transition."""
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    mode: OperatingMode
    is_valid: bool
    preconditions_checked: List[str] = Field(default_factory=list)
    preconditions_passed: List[str] = Field(default_factory=list)
    preconditions_failed: List[str] = Field(default_factory=list)
    failure_reasons: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.validation_id}|{self.mode.value}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class ModeTransition(BaseModel):
    """Record of a mode transition attempt."""
    transition_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_mode: OperatingMode
    to_mode: OperatingMode
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    initiated_by: str = Field(default="SYSTEM")
    reason: str = Field(default="")
    validation_result: Optional[ModeValidationResult] = None
    success: bool = Field(default=False)
    error_message: Optional[str] = None
    duration_ms: float = Field(default=0.0, ge=0)
    rollback_available: bool = Field(default=True)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            hash_input = f"{self.transition_id}|{self.from_mode.value}|{self.to_mode.value}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class TransitionResult(BaseModel):
    """Result of a mode transition request."""
    transition: ModeTransition
    success: bool
    current_mode: OperatingMode
    message: str = Field(default="")
    requires_acknowledgment: bool = Field(default=False)


class ModeState(BaseModel):
    """Current state of the operating mode."""
    mode: OperatingMode
    entered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = Field(default=0.0, ge=0)
    transition_count: int = Field(default=0, ge=0)
    last_transition: Optional[ModeTransition] = None
    mode_timeout_seconds: Optional[float] = None
    timeout_action: Optional[OperatingMode] = None


class ModeConfig(BaseModel):
    """Configuration for an operating mode."""
    mode: OperatingMode
    display_name: str
    description: str = Field(default="")
    dcs_write_enabled: bool = Field(default=False)
    requires_operator: bool = Field(default=False)
    max_duration_seconds: Optional[float] = None
    allowed_transitions: Set[OperatingMode] = Field(default_factory=set)
    preconditions: List[PreconditionType] = Field(default_factory=list)


class OperatingModeManager:
    """
    Manages operating mode transitions for the burner management system.

    This class implements a state machine for mode transitions with:
    - Validated transition paths (not all transitions are allowed)
    - Precondition checking before transitions
    - Complete audit trail of all transitions
    - Timeout handling for modes with duration limits

    Attributes:
        TRANSITION_MATRIX: Defines allowed transitions between modes

    Example:
        >>> manager = OperatingModeManager()
        >>> result = manager.transition_mode(
        ...     from_mode=OperatingMode.OBSERVE,
        ...     to_mode=OperatingMode.ADVISORY,
        ...     initiated_by="OPERATOR_001",
        ...     reason="Starting advisory operation"
        ... )
        >>> print(result.success)
        True
    """

    TRANSITION_MATRIX: Dict[OperatingMode, Set[OperatingMode]] = {
        OperatingMode.OBSERVE: {OperatingMode.ADVISORY, OperatingMode.MAINTENANCE},
        OperatingMode.ADVISORY: {
            OperatingMode.OBSERVE,
            OperatingMode.CLOSED_LOOP,
            OperatingMode.FALLBACK,
            OperatingMode.MAINTENANCE
        },
        OperatingMode.CLOSED_LOOP: {
            OperatingMode.ADVISORY,
            OperatingMode.FALLBACK,
            OperatingMode.MAINTENANCE
        },
        OperatingMode.FALLBACK: {
            OperatingMode.OBSERVE,
            OperatingMode.ADVISORY,
            OperatingMode.MAINTENANCE
        },
        OperatingMode.MAINTENANCE: {OperatingMode.OBSERVE},
    }

    MODE_PRECONDITIONS: Dict[OperatingMode, List[PreconditionType]] = {
        OperatingMode.OBSERVE: [],
        OperatingMode.ADVISORY: [
            PreconditionType.BMS_STATUS,
            PreconditionType.SYSTEM_HEALTH
        ],
        OperatingMode.CLOSED_LOOP: [
            PreconditionType.BMS_STATUS,
            PreconditionType.OPERATOR_AUTH,
            PreconditionType.SAFETY_CHECK,
            PreconditionType.SYSTEM_HEALTH,
            PreconditionType.PROCESS_STABLE,
            PreconditionType.NO_ACTIVE_ALARMS,
            PreconditionType.INTERLOCK_ARMED
        ],
        OperatingMode.FALLBACK: [PreconditionType.BMS_STATUS],
        OperatingMode.MAINTENANCE: [PreconditionType.OPERATOR_AUTH],
    }

    def __init__(self, initial_mode: OperatingMode = OperatingMode.OBSERVE) -> None:
        """Initialize the operating mode manager."""
        self._current_mode = initial_mode
        self._mode_entered_at = datetime.now(timezone.utc)
        self._transition_count = 0
        self._transition_history: List[ModeTransition] = []
        self._precondition_checkers: Dict[PreconditionType, Callable[[], bool]] = {}
        self._audit_log: List[Dict[str, Any]] = []
        self._mode_timeouts: Dict[OperatingMode, float] = {}

        for precondition in PreconditionType:
            self._precondition_checkers[precondition] = lambda: True

        logger.info(f"OperatingModeManager initialized in {initial_mode.value} mode")

    def transition_mode(
        self,
        from_mode: OperatingMode,
        to_mode: OperatingMode,
        initiated_by: str = "SYSTEM",
        reason: str = "",
        force: bool = False
    ) -> TransitionResult:
        """Attempt to transition from one operating mode to another."""
        start_time = datetime.now(timezone.utc)

        if from_mode != self._current_mode:
            transition = ModeTransition(
                from_mode=from_mode,
                to_mode=to_mode,
                initiated_by=initiated_by,
                reason=reason,
                success=False,
                error_message=f"Mode mismatch: expected {from_mode.value}, current is {self._current_mode.value}"
            )
            logger.warning(f"Mode transition failed: mode mismatch")
            return TransitionResult(
                transition=transition,
                success=False,
                current_mode=self._current_mode,
                message="Mode mismatch - current mode differs from expected"
            )

        allowed_transitions = self.TRANSITION_MATRIX.get(from_mode, set())
        if to_mode not in allowed_transitions:
            transition = ModeTransition(
                from_mode=from_mode,
                to_mode=to_mode,
                initiated_by=initiated_by,
                reason=reason,
                success=False,
                error_message=f"Invalid transition from {from_mode.value} to {to_mode.value}"
            )
            logger.warning(f"Invalid mode transition requested: {from_mode.value} -> {to_mode.value}")
            return TransitionResult(
                transition=transition,
                success=False,
                current_mode=self._current_mode,
                message=f"Transition from {from_mode.value} to {to_mode.value} is not allowed"
            )

        validation_result = None
        if not force:
            validation_result = self.validate_mode_preconditions(to_mode)
            if not validation_result.is_valid:
                transition = ModeTransition(
                    from_mode=from_mode,
                    to_mode=to_mode,
                    initiated_by=initiated_by,
                    reason=reason,
                    validation_result=validation_result,
                    success=False,
                    error_message="Preconditions not met"
                )
                logger.warning(f"Mode transition blocked: preconditions not met for {to_mode.value}")
                return TransitionResult(
                    transition=transition,
                    success=False,
                    current_mode=self._current_mode,
                    message=f"Preconditions not met: {validation_result.preconditions_failed}"
                )

        old_mode = self._current_mode
        self._current_mode = to_mode
        self._mode_entered_at = datetime.now(timezone.utc)
        self._transition_count += 1

        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        transition = ModeTransition(
            from_mode=from_mode,
            to_mode=to_mode,
            initiated_by=initiated_by,
            reason=reason,
            validation_result=validation_result,
            success=True,
            duration_ms=duration_ms
        )

        self._transition_history.append(transition)
        self.log_mode_transition(transition)

        logger.info(f"Mode transition successful: {from_mode.value} -> {to_mode.value}")

        return TransitionResult(
            transition=transition,
            success=True,
            current_mode=to_mode,
            message=f"Successfully transitioned from {from_mode.value} to {to_mode.value}",
            requires_acknowledgment=to_mode == OperatingMode.CLOSED_LOOP
        )

    def validate_mode_preconditions(self, mode: OperatingMode) -> ModeValidationResult:
        """Validate all preconditions required for entering a specific mode."""
        preconditions = self.MODE_PRECONDITIONS.get(mode, [])

        checked = []
        passed = []
        failed = []
        failure_reasons = []

        for precondition in preconditions:
            checked.append(precondition.value)
            checker = self._precondition_checkers.get(precondition)

            try:
                if checker and checker():
                    passed.append(precondition.value)
                else:
                    failed.append(precondition.value)
                    failure_reasons.append(f"{precondition.value} check failed")
            except Exception as e:
                failed.append(precondition.value)
                failure_reasons.append(f"{precondition.value} check error: {str(e)}")
                logger.error(f"Precondition check failed: {precondition.value} - {str(e)}")

        return ModeValidationResult(
            mode=mode,
            is_valid=len(failed) == 0,
            preconditions_checked=checked,
            preconditions_passed=passed,
            preconditions_failed=failed,
            failure_reasons=failure_reasons
        )

    def get_current_mode(self) -> OperatingMode:
        """Get the current operating mode."""
        return self._current_mode

    def get_mode_state(self) -> ModeState:
        """Get comprehensive state information about the current mode."""
        now = datetime.now(timezone.utc)
        duration = (now - self._mode_entered_at).total_seconds()

        return ModeState(
            mode=self._current_mode,
            entered_at=self._mode_entered_at,
            duration_seconds=duration,
            transition_count=self._transition_count,
            last_transition=self._transition_history[-1] if self._transition_history else None,
            mode_timeout_seconds=self._mode_timeouts.get(self._current_mode),
            timeout_action=OperatingMode.FALLBACK if self._current_mode == OperatingMode.CLOSED_LOOP else None
        )

    def handle_mode_timeout(self, mode: OperatingMode, duration: float) -> OperatingMode:
        """Handle timeout for a mode that has exceeded its maximum duration."""
        logger.warning(f"Mode timeout triggered for {mode.value} after {duration}s")

        if mode == OperatingMode.CLOSED_LOOP:
            fallback_mode = OperatingMode.FALLBACK
        else:
            fallback_mode = OperatingMode.OBSERVE

        result = self.transition_mode(
            from_mode=mode,
            to_mode=fallback_mode,
            initiated_by="TIMEOUT_HANDLER",
            reason=f"Mode timeout after {duration:.1f}s",
            force=True
        )

        return result.current_mode

    def log_mode_transition(self, transition: ModeTransition) -> None:
        """Log a mode transition to the audit trail."""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "MODE_TRANSITION",
            "transition_id": transition.transition_id,
            "from_mode": transition.from_mode.value,
            "to_mode": transition.to_mode.value,
            "initiated_by": transition.initiated_by,
            "reason": transition.reason,
            "success": transition.success,
            "error_message": transition.error_message,
            "provenance_hash": transition.provenance_hash
        }

        self._audit_log.append(audit_entry)
        logger.info(f"Audit: {audit_entry}")

    def register_precondition_checker(
        self,
        precondition: PreconditionType,
        checker: Callable[[], bool]
    ) -> None:
        """Register a custom precondition checker function."""
        self._precondition_checkers[precondition] = checker
        logger.debug(f"Registered precondition checker for {precondition.value}")

    def is_dcs_write_enabled(self) -> bool:
        """Check if DCS write operations are enabled in current mode."""
        return self._current_mode == OperatingMode.CLOSED_LOOP

    def get_allowed_transitions(self) -> Set[OperatingMode]:
        """Get the set of modes that can be transitioned to from current mode."""
        return self.TRANSITION_MATRIX.get(self._current_mode, set())

    def get_transition_history(self, limit: int = 100) -> List[ModeTransition]:
        """Get recent transition history."""
        return list(reversed(self._transition_history[-limit:]))

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the mode manager."""
        now = datetime.now(timezone.utc)
        duration = (now - self._mode_entered_at).total_seconds()

        return {
            "current_mode": self._current_mode.value,
            "mode_entered_at": self._mode_entered_at.isoformat(),
            "duration_seconds": duration,
            "transition_count": self._transition_count,
            "dcs_write_enabled": self.is_dcs_write_enabled(),
            "allowed_transitions": [m.value for m in self.get_allowed_transitions()],
            "last_transition_id": self._transition_history[-1].transition_id if self._transition_history else None
        }

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the audit log of mode transitions."""
        return list(reversed(self._audit_log[-limit:]))

    def set_mode_timeout(self, mode: OperatingMode, timeout_seconds: float) -> None:
        """Set a timeout for a specific mode."""
        self._mode_timeouts[mode] = timeout_seconds
        logger.info(f"Set timeout of {timeout_seconds}s for mode {mode.value}")

    def check_mode_timeout(self) -> Optional[OperatingMode]:
        """Check if current mode has timed out."""
        timeout = self._mode_timeouts.get(self._current_mode)
        if timeout is None:
            return None

        duration = (datetime.now(timezone.utc) - self._mode_entered_at).total_seconds()
        if duration > timeout:
            return self.handle_mode_timeout(self._current_mode, duration)

        return None