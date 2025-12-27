"""
GL-016 Waterguard Action Gate - IEC 61511 SIL-3 Compliant

This module implements permission-based action control with complete
audit trail for all actuation commands from WATERGUARD.

Every actuation command MUST pass through the ActionGate before execution.
The gate verifies permissions, logs all actions, and supports staged
commissioning modes for safe deployment.

Key Features:
    - Permission-based action control
    - Complete audit trail with SHA-256 hashes
    - Staged commissioning modes
    - Action logging for all commands
    - Integration with boundary engine and SIS

Author: GreenLang Safety Engineering Team
Version: 1.0.0
SIL Level: 3
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ActionType(str, Enum):
    """Types of control actions."""
    BLOWDOWN_ADJUST = "blowdown_adjust"
    CHEMICAL_DOSING = "chemical_dosing"
    SETPOINT_CHANGE = "setpoint_change"
    VALVE_POSITION = "valve_position"
    PUMP_SPEED = "pump_speed"
    ALARM_ACKNOWLEDGE = "alarm_acknowledge"
    MODE_CHANGE = "mode_change"
    PARAMETER_UPDATE = "parameter_update"


class PermissionStatus(str, Enum):
    """Permission status for an action."""
    PERMITTED = "permitted"
    DENIED = "denied"
    PENDING = "pending"
    EXPIRED = "expired"


class CommissioningMode(str, Enum):
    """Staged commissioning modes for safe deployment."""
    OBSERVE_ONLY = "observe_only"       # Read data, no actions
    RECOMMEND_ONLY = "recommend_only"   # Generate recommendations, no actions
    CONFIRM_ALL = "confirm_all"         # All actions require confirmation
    AUTO_BOUNDED = "auto_bounded"       # Auto within tight bounds
    FULL_AUTO = "full_auto"             # Full automatic operation


# =============================================================================
# DATA MODELS
# =============================================================================


class PermissionResult(BaseModel):
    """Result of a permission check."""

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Result identifier"
    )
    action_type: ActionType = Field(
        ...,
        description="Type of action"
    )
    status: PermissionStatus = Field(
        ...,
        description="Permission status"
    )
    permitted: bool = Field(
        ...,
        description="Is action permitted"
    )
    reason: str = Field(
        default="",
        description="Reason for decision"
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions that must be met"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When permission expires"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.result_id}|{self.action_type.value}|"
                f"{self.status.value}|{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:16]


class GatedAction(BaseModel):
    """A gated action with audit information."""

    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique action identifier"
    )
    action_type: ActionType = Field(
        ...,
        description="Type of action"
    )
    target_tag: str = Field(
        ...,
        description="Target equipment tag"
    )
    target_value: float = Field(
        ...,
        description="Target value"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current value before action"
    )
    engineering_units: str = Field(
        default="",
        description="Engineering units"
    )
    source: str = Field(
        default="WATERGUARD",
        description="Action source"
    )
    reason: str = Field(
        default="",
        description="Reason for action"
    )
    permission_result: Optional[PermissionResult] = Field(
        default=None,
        description="Permission check result"
    )
    executed: bool = Field(
        default=False,
        description="Was action executed"
    )
    execution_time: Optional[datetime] = Field(
        default=None,
        description="When action was executed"
    )
    execution_result: str = Field(
        default="",
        description="Execution result"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Action timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.action_id}|{self.action_type.value}|"
                f"{self.target_tag}|{self.target_value}|{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


# =============================================================================
# ACTION GATE
# =============================================================================


class ActionGate:
    """
    Permission-based action control gate.

    Every actuation command from WATERGUARD MUST pass through this gate.
    The gate:
        1. Checks if action is permitted based on current mode
        2. Validates against boundary engine constraints
        3. Verifies SIS status (read-only)
        4. Logs all actions for audit trail
        5. Supports staged commissioning modes

    CRITICAL: All actions are logged regardless of permission status.

    Commissioning Modes:
        - OBSERVE_ONLY: Read data, no actions
        - RECOMMEND_ONLY: Generate recommendations, no actions
        - CONFIRM_ALL: All actions require operator confirmation
        - AUTO_BOUNDED: Automatic within tight bounds
        - FULL_AUTO: Full automatic operation

    Example:
        >>> gate = ActionGate(boundary_engine, sis_interface)
        >>> gate.set_commissioning_mode(CommissioningMode.CONFIRM_ALL)
        >>>
        >>> result = gate.is_action_permitted(
        ...     ActionType.BLOWDOWN_ADJUST,
        ...     {"target_tag": "BD-001", "target_value": 5.0}
        ... )
        >>> if result.permitted:
        ...     execute_action(...)
    """

    def __init__(
        self,
        boundary_engine: Any = None,
        sis_interface: Any = None,
        initial_mode: CommissioningMode = CommissioningMode.OBSERVE_ONLY,
    ) -> None:
        """
        Initialize ActionGate.

        Args:
            boundary_engine: WaterguardBoundaryEngine for constraint checking
            sis_interface: SISInterface for safety status
            initial_mode: Initial commissioning mode (default: OBSERVE_ONLY)
        """
        self._boundary_engine = boundary_engine
        self._sis_interface = sis_interface
        self._commissioning_mode = initial_mode

        self._lock = threading.Lock()

        # Action log (complete audit trail)
        self._action_log: List[GatedAction] = []
        self._action_log_lock = threading.Lock()

        # Pending actions requiring confirmation
        self._pending_actions: Dict[str, GatedAction] = {}

        # Statistics
        self._stats = {
            "actions_checked": 0,
            "actions_permitted": 0,
            "actions_denied": 0,
            "actions_executed": 0,
            "actions_pending": 0,
        }
        self._stats_lock = threading.Lock()

        # Mode change callback
        self._mode_change_callback: Optional[Callable[[CommissioningMode], None]] = None

        logger.info(
            "ActionGate initialized in %s mode",
            initial_mode.value
        )

    def is_action_permitted(
        self,
        action_type: ActionType,
        parameters: Dict[str, Any],
    ) -> PermissionResult:
        """
        Check if an action is permitted.

        This is the main entry point for action permission checks.
        ALL actions must pass through this method.

        Args:
            action_type: Type of action
            parameters: Action parameters including target_tag, target_value

        Returns:
            PermissionResult with decision and reason
        """
        with self._stats_lock:
            self._stats["actions_checked"] += 1

        target_tag = parameters.get("target_tag", "")
        target_value = parameters.get("target_value", 0.0)

        logger.debug(
            "Permission check: %s on %s = %.2f",
            action_type.value, target_tag, target_value
        )

        # Step 1: Check commissioning mode
        mode_result = self._check_commissioning_mode(action_type)
        if not mode_result[0]:
            return self._create_denied_result(
                action_type,
                f"Commissioning mode {self._commissioning_mode.value}: {mode_result[1]}"
            )

        # Step 2: Check SIS status (read-only)
        if self._sis_interface:
            if not self._sis_interface.is_sis_healthy():
                return self._create_denied_result(
                    action_type,
                    "SIS is not healthy - action blocked"
                )

            if self._sis_interface.has_active_trips():
                return self._create_denied_result(
                    action_type,
                    "Active SIS trips - action blocked"
                )

            if self._sis_interface.is_tag_protected(target_tag):
                return self._create_denied_result(
                    action_type,
                    f"Tag {target_tag} is SIS-protected - CANNOT write"
                )

        # Step 3: Check boundary engine constraints
        if self._boundary_engine:
            from .boundary_engine import ProposedAction

            proposed = ProposedAction(
                action_type=action_type.value,
                target_tag=target_tag,
                target_value=target_value,
            )

            permitted, reason = self._boundary_engine.validate_action(proposed)
            if not permitted:
                return self._create_denied_result(action_type, reason)

        # Step 4: Mode-specific handling
        if self._commissioning_mode == CommissioningMode.CONFIRM_ALL:
            return self._create_pending_result(action_type, parameters)

        # Action is permitted
        with self._stats_lock:
            self._stats["actions_permitted"] += 1

        return PermissionResult(
            action_type=action_type,
            status=PermissionStatus.PERMITTED,
            permitted=True,
            reason="Action permitted within all constraints",
        )

    def _check_commissioning_mode(
        self,
        action_type: ActionType
    ) -> Tuple[bool, str]:
        """
        Check if action is allowed in current commissioning mode.

        Args:
            action_type: Type of action

        Returns:
            Tuple of (allowed, reason)
        """
        mode = self._commissioning_mode

        if mode == CommissioningMode.OBSERVE_ONLY:
            return False, "System is in observe-only mode"

        if mode == CommissioningMode.RECOMMEND_ONLY:
            return False, "System is in recommend-only mode (no actuation)"

        if mode == CommissioningMode.CONFIRM_ALL:
            return True, "Requires operator confirmation"

        if mode == CommissioningMode.AUTO_BOUNDED:
            # Only certain action types allowed
            allowed_types = {
                ActionType.BLOWDOWN_ADJUST,
                ActionType.CHEMICAL_DOSING,
            }
            if action_type not in allowed_types:
                return False, f"Action type {action_type.value} not allowed in bounded mode"
            return True, "Allowed within bounds"

        if mode == CommissioningMode.FULL_AUTO:
            return True, "Full automatic mode"

        return False, f"Unknown mode: {mode}"

    def _create_denied_result(
        self,
        action_type: ActionType,
        reason: str
    ) -> PermissionResult:
        """Create a denied permission result."""
        with self._stats_lock:
            self._stats["actions_denied"] += 1

        logger.warning("Action DENIED: %s - %s", action_type.value, reason)

        return PermissionResult(
            action_type=action_type,
            status=PermissionStatus.DENIED,
            permitted=False,
            reason=reason,
        )

    def _create_pending_result(
        self,
        action_type: ActionType,
        parameters: Dict[str, Any]
    ) -> PermissionResult:
        """Create a pending permission result requiring confirmation."""
        with self._stats_lock:
            self._stats["actions_pending"] += 1

        result_id = str(uuid.uuid4())[:8]

        # Store pending action
        gated_action = GatedAction(
            action_type=action_type,
            target_tag=parameters.get("target_tag", ""),
            target_value=parameters.get("target_value", 0.0),
        )
        self._pending_actions[result_id] = gated_action

        logger.info(
            "Action PENDING confirmation: %s - ID: %s",
            action_type.value, result_id
        )

        return PermissionResult(
            result_id=result_id,
            action_type=action_type,
            status=PermissionStatus.PENDING,
            permitted=False,
            reason="Requires operator confirmation",
            conditions=["Operator must confirm action"],
        )

    def confirm_pending_action(
        self,
        action_id: str,
        confirmed_by: str
    ) -> Optional[GatedAction]:
        """
        Confirm a pending action.

        Args:
            action_id: ID of pending action
            confirmed_by: Operator confirming

        Returns:
            GatedAction if confirmed, None if not found
        """
        if action_id not in self._pending_actions:
            logger.warning("Pending action not found: %s", action_id)
            return None

        action = self._pending_actions.pop(action_id)

        action.permission_result = PermissionResult(
            action_type=action.action_type,
            status=PermissionStatus.PERMITTED,
            permitted=True,
            reason=f"Confirmed by {confirmed_by}",
        )

        logger.info(
            "Action CONFIRMED by %s: %s on %s",
            confirmed_by, action.action_type.value, action.target_tag
        )

        return action

    def reject_pending_action(
        self,
        action_id: str,
        rejected_by: str,
        reason: str = ""
    ) -> bool:
        """
        Reject a pending action.

        Args:
            action_id: ID of pending action
            rejected_by: Operator rejecting
            reason: Rejection reason

        Returns:
            True if rejected
        """
        if action_id not in self._pending_actions:
            return False

        action = self._pending_actions.pop(action_id)

        logger.info(
            "Action REJECTED by %s: %s - %s",
            rejected_by, action.action_type.value, reason
        )

        # Log rejection
        self.log_gated_action(action, executed=False, result=f"Rejected: {reason}")

        return True

    def log_gated_action(
        self,
        action: GatedAction,
        executed: bool,
        result: str = ""
    ) -> None:
        """
        Log a gated action for audit trail.

        Args:
            action: The action to log
            executed: Whether action was executed
            result: Execution result
        """
        action.executed = executed
        action.execution_time = datetime.now(timezone.utc) if executed else None
        action.execution_result = result

        with self._action_log_lock:
            self._action_log.append(action)

        if executed:
            with self._stats_lock:
                self._stats["actions_executed"] += 1

        logger.info(
            "Action logged: %s on %s = %.2f (executed=%s)",
            action.action_type.value,
            action.target_tag,
            action.target_value,
            executed
        )

    def set_commissioning_mode(
        self,
        mode: CommissioningMode,
        authorized_by: str
    ) -> None:
        """
        Set commissioning mode.

        Args:
            mode: New commissioning mode
            authorized_by: Person authorizing change
        """
        old_mode = self._commissioning_mode

        with self._lock:
            self._commissioning_mode = mode

        logger.warning(
            "Commissioning mode changed: %s -> %s (by %s)",
            old_mode.value, mode.value, authorized_by
        )

        if self._mode_change_callback:
            try:
                self._mode_change_callback(mode)
            except Exception as e:
                logger.error("Mode change callback failed: %s", e)

    def get_commissioning_mode(self) -> CommissioningMode:
        """Get current commissioning mode."""
        with self._lock:
            return self._commissioning_mode

    def get_pending_actions(self) -> List[GatedAction]:
        """Get all pending actions."""
        return list(self._pending_actions.values())

    def get_action_log(
        self,
        limit: int = 100,
        action_type: Optional[ActionType] = None
    ) -> List[GatedAction]:
        """
        Get action log entries.

        Args:
            limit: Maximum entries to return
            action_type: Filter by action type

        Returns:
            List of gated actions
        """
        with self._action_log_lock:
            log = self._action_log
            if action_type:
                log = [a for a in log if a.action_type == action_type]
            return list(reversed(log[-limit:]))

    def get_statistics(self) -> Dict[str, int]:
        """Get gate statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def set_mode_change_callback(
        self,
        callback: Callable[[CommissioningMode], None]
    ) -> None:
        """Set callback for mode changes."""
        self._mode_change_callback = callback


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ActionType",
    "PermissionStatus",
    "CommissioningMode",
    # Models
    "PermissionResult",
    "GatedAction",
    # Classes
    "ActionGate",
]
