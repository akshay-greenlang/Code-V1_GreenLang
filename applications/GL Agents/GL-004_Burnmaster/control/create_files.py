#!/usr/bin/env python3
"""Script to create all control module files."""
import os

control_dir = os.path.dirname(os.path.abspath(__file__))

# Mode manager content
mode_manager_content = '''"""
GL-004 BURNMASTER - Operating Mode Manager

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
    OBSERVE = "observe"
    ADVISORY = "advisory"
    CLOSED_LOOP = "closed_loop"
    FALLBACK = "fallback"
    MAINTENANCE = "maintenance"


class TransitionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING = "pending"
    TIMEOUT = "timeout"


class PreconditionType(str, Enum):
    BMS_STATUS = "bms_status"
    OPERATOR_AUTH = "operator_authorization"
    SAFETY_CHECK = "safety_check"
    SYSTEM_HEALTH = "system_health"
    PROCESS_STABLE = "process_stable"
    NO_ACTIVE_ALARMS = "no_active_alarms"
    INTERLOCK_ARMED = "interlock_armed"


class ModeValidationResult(BaseModel):
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
        if not self.provenance_hash:
            self.provenance_hash = hashlib.sha256(f"{self.validation_id}|{self.mode.value}".encode()).hexdigest()


class ModeTransition(BaseModel):
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


class TransitionResult(BaseModel):
    transition: ModeTransition
    success: bool
    current_mode: OperatingMode
    message: str = Field(default="")
    requires_acknowledgment: bool = Field(default=False)


class ModeState(BaseModel):
    mode: OperatingMode
    entered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_seconds: float = Field(default=0.0, ge=0)
    transition_count: int = Field(default=0, ge=0)
    last_transition: Optional[ModeTransition] = None
    mode_timeout_seconds: Optional[float] = None
    timeout_action: Optional[OperatingMode] = None


class ModeConfig(BaseModel):
    mode: OperatingMode
    display_name: str
    description: str = Field(default="")
    dcs_write_enabled: bool = Field(default=False)
    requires_operator: bool = Field(default=False)
    max_duration_seconds: Optional[float] = None
    allowed_transitions: Set[OperatingMode] = Field(default_factory=set)
    preconditions: List[PreconditionType] = Field(default_factory=list)


class OperatingModeManager:
    TRANSITION_MATRIX = {
        OperatingMode.OBSERVE: {OperatingMode.ADVISORY, OperatingMode.MAINTENANCE},
        OperatingMode.ADVISORY: {OperatingMode.OBSERVE, OperatingMode.CLOSED_LOOP, OperatingMode.FALLBACK, OperatingMode.MAINTENANCE},
        OperatingMode.CLOSED_LOOP: {OperatingMode.ADVISORY, OperatingMode.FALLBACK, OperatingMode.MAINTENANCE},
        OperatingMode.FALLBACK: {OperatingMode.OBSERVE, OperatingMode.ADVISORY, OperatingMode.MAINTENANCE},
        OperatingMode.MAINTENANCE: {OperatingMode.OBSERVE},
    }

    def __init__(self, initial_mode: OperatingMode = OperatingMode.OBSERVE) -> None:
        self._current_mode = initial_mode
        self._mode_entered_at = datetime.now(timezone.utc)
        self._transition_count = 0
        self._transition_history: List[ModeTransition] = []
        self._precondition_checkers: Dict[PreconditionType, Callable[[], bool]] = {}
        self._audit_log: List[Dict[str, Any]] = []
        for p in PreconditionType:
            self._precondition_checkers[p] = lambda: True

    def transition_mode(self, from_mode: OperatingMode, to_mode: OperatingMode, initiated_by: str = "SYSTEM", reason: str = "", force: bool = False) -> TransitionResult:
        if from_mode != self._current_mode:
            transition = ModeTransition(from_mode=from_mode, to_mode=to_mode, success=False, error_message="Mode mismatch")
            return TransitionResult(transition=transition, success=False, current_mode=self._current_mode, message="Mode mismatch")
        if to_mode not in self.TRANSITION_MATRIX.get(from_mode, set()):
            transition = ModeTransition(from_mode=from_mode, to_mode=to_mode, success=False, error_message="Invalid transition")
            return TransitionResult(transition=transition, success=False, current_mode=self._current_mode, message="Invalid transition")
        validation_result = None if force else self.validate_mode_preconditions(to_mode)
        if validation_result and not validation_result.is_valid:
            transition = ModeTransition(from_mode=from_mode, to_mode=to_mode, validation_result=validation_result, success=False)
            return TransitionResult(transition=transition, success=False, current_mode=self._current_mode, message="Preconditions not met")
        self._current_mode = to_mode
        self._mode_entered_at = datetime.now(timezone.utc)
        self._transition_count += 1
        transition = ModeTransition(from_mode=from_mode, to_mode=to_mode, initiated_by=initiated_by, reason=reason, validation_result=validation_result, success=True)
        self._transition_history.append(transition)
        self.log_mode_transition(transition)
        return TransitionResult(transition=transition, success=True, current_mode=to_mode, message=f"Transitioned to {to_mode.value}")

    def validate_mode_preconditions(self, mode: OperatingMode) -> ModeValidationResult:
        passed = [p.value for p in PreconditionType if self._precondition_checkers.get(p, lambda: True)()]
        failed = [p.value for p in PreconditionType if p.value not in passed]
        return ModeValidationResult(mode=mode, is_valid=len(failed) == 0, preconditions_checked=[p.value for p in PreconditionType], preconditions_passed=passed, preconditions_failed=failed)

    def get_current_mode(self) -> OperatingMode:
        return self._current_mode

    def get_mode_state(self) -> ModeState:
        return ModeState(mode=self._current_mode, entered_at=self._mode_entered_at, duration_seconds=(datetime.now(timezone.utc) - self._mode_entered_at).total_seconds(), transition_count=self._transition_count, last_transition=self._transition_history[-1] if self._transition_history else None)

    def handle_mode_timeout(self, mode: OperatingMode, duration: float) -> OperatingMode:
        result = self.transition_mode(from_mode=mode, to_mode=OperatingMode.OBSERVE, initiated_by="TIMEOUT", reason=f"Timeout after {duration}s", force=True)
        return result.current_mode

    def log_mode_transition(self, transition: ModeTransition) -> None:
        self._audit_log.append({"timestamp": datetime.now(timezone.utc).isoformat(), "event": "MODE_TRANSITION", "from": transition.from_mode.value, "to": transition.to_mode.value})

    def register_precondition_checker(self, precondition: PreconditionType, checker: Callable[[], bool]) -> None:
        self._precondition_checkers[precondition] = checker

    def is_dcs_write_enabled(self) -> bool:
        return self._current_mode == OperatingMode.CLOSED_LOOP

    def get_allowed_transitions(self) -> Set[OperatingMode]:
        return self.TRANSITION_MATRIX.get(self._current_mode, set())

    def get_transition_history(self, limit: int = 100) -> List[ModeTransition]:
        return list(reversed(self._transition_history[-limit:]))

    def get_status(self) -> Dict[str, Any]:
        return {"current_mode": self._current_mode.value, "duration": (datetime.now(timezone.utc) - self._mode_entered_at).total_seconds(), "transition_count": self._transition_count, "dcs_write_enabled": self.is_dcs_write_enabled()}
'''

with open(os.path.join(control_dir, 'mode_manager.py'), 'w', encoding='utf-8') as f:
    f.write(mode_manager_content)
print('Created mode_manager.py')

if __name__ == '__main__':
    print('Files created successfully')
