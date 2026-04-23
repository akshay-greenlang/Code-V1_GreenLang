"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Interlock Manager

This module manages interlocks for the steam system, ensuring GL-003
respects existing safety envelopes without modifying SIF logic.

Safety Architecture:
    - GL-003 respects existing safety envelopes (read-only access to SIS)
    - No control authority over Safety Instrumented Functions
    - Permissive checking before any control action
    - Complete interlock status monitoring

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - ANSI/ISA-18.2 Management of Alarm Systems

IMPORTANT: GL-003 has NO authority to modify SIF logic or bypass interlocks.
This module provides monitoring and permissive checking only.

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class InterlockState(str, Enum):
    """Interlock state enumeration."""
    NORMAL = "normal"
    ACTIVE = "active"
    BYPASSED = "bypassed"
    FAULT = "fault"
    UNKNOWN = "unknown"


class PermissiveStatus(str, Enum):
    """Permissive status enumeration."""
    GRANTED = "granted"
    DENIED = "denied"
    CONDITIONAL = "conditional"
    PENDING = "pending"


class InterlockType(str, Enum):
    """Interlock type enumeration."""
    SAFETY = "safety"  # SIF-related - GL-003 cannot modify
    PERMISSIVE = "permissive"  # Sequence permissive
    PROTECTIVE = "protective"  # Equipment protection
    OPERATIONAL = "operational"  # Operational interlock


class OperationType(str, Enum):
    """Operation type for permissive checking."""
    SETPOINT_CHANGE = "setpoint_change"
    VALVE_OPERATION = "valve_operation"
    MODE_CHANGE = "mode_change"
    START_EQUIPMENT = "start_equipment"
    STOP_EQUIPMENT = "stop_equipment"
    LOAD_CHANGE = "load_change"
    OPTIMIZATION = "optimization"


class ActionType(str, Enum):
    """Interlock action type."""
    TRIP = "trip"
    ALARM = "alarm"
    PERMISSIVE_BLOCK = "permissive_block"
    LIMIT = "limit"
    NOTIFICATION = "notification"


# =============================================================================
# DATA MODELS
# =============================================================================

class InterlockCondition(BaseModel):
    """Condition that triggers an interlock."""

    condition_id: str = Field(..., description="Condition identifier")
    parameter: str = Field(..., description="Parameter being monitored")
    operator: str = Field(
        ...,
        description="Comparison operator (>, <, >=, <=, ==, !=)"
    )
    threshold: float = Field(..., description="Threshold value")
    unit: str = Field(default="", description="Parameter unit")
    description: str = Field(default="", description="Condition description")


class InterlockDefinition(BaseModel):
    """Definition of an interlock."""

    interlock_id: str = Field(..., description="Interlock identifier")
    name: str = Field(..., description="Interlock name")
    interlock_type: InterlockType = Field(..., description="Interlock type")
    equipment_id: str = Field(..., description="Associated equipment ID")
    conditions: List[InterlockCondition] = Field(
        ...,
        description="Conditions that trigger this interlock"
    )
    action: ActionType = Field(..., description="Action when triggered")
    action_description: str = Field(
        default="",
        description="Description of action taken"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Priority (1=highest)"
    )
    is_safety_critical: bool = Field(
        default=False,
        description="Safety-critical interlock (SIF)"
    )
    can_bypass: bool = Field(
        default=False,
        description="Interlock can be bypassed"
    )
    bypass_authorization_required: Optional[str] = Field(
        None,
        description="Authorization level required for bypass"
    )


class ActiveInterlock(BaseModel):
    """Active interlock status."""

    interlock_id: str = Field(..., description="Interlock identifier")
    name: str = Field(..., description="Interlock name")
    state: InterlockState = Field(..., description="Current state")
    triggered_by: List[str] = Field(
        default_factory=list,
        description="Conditions that triggered the interlock"
    )
    triggered_at: Optional[datetime] = Field(
        None,
        description="Time interlock was triggered"
    )
    action_taken: str = Field(default="", description="Action taken")
    equipment_affected: str = Field(
        default="",
        description="Equipment affected by interlock"
    )
    is_safety_critical: bool = Field(
        default=False,
        description="Safety-critical interlock"
    )
    message: str = Field(default="", description="Status message")


class PermissiveResult(BaseModel):
    """Result of permissive check."""

    permissive_id: str = Field(..., description="Permissive check ID")
    operation_type: OperationType = Field(
        ...,
        description="Type of operation requested"
    )
    status: PermissiveStatus = Field(..., description="Permissive status")
    interlocks_checked: int = Field(
        default=0,
        description="Number of interlocks checked"
    )
    interlocks_blocking: List[str] = Field(
        default_factory=list,
        description="Interlocks blocking operation"
    )
    conditions_required: List[str] = Field(
        default_factory=list,
        description="Conditions required for operation"
    )
    message: str = Field(default="", description="Status message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Check timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class InterlockEvaluation(BaseModel):
    """Result of interlock evaluation."""

    evaluation_id: str = Field(..., description="Evaluation ID")
    interlocks_evaluated: int = Field(
        default=0,
        description="Number of interlocks evaluated"
    )
    active_interlocks: List[ActiveInterlock] = Field(
        default_factory=list,
        description="Currently active interlocks"
    )
    safety_interlocks_active: int = Field(
        default=0,
        description="Number of safety interlocks active"
    )
    system_safe: bool = Field(
        default=True,
        description="System is in safe state"
    )
    optimization_allowed: bool = Field(
        default=True,
        description="Optimization operations allowed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Evaluation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class SystemState(BaseModel):
    """Current system state for interlock evaluation."""

    parameters: Dict[str, float] = Field(
        ...,
        description="Current parameter values by tag"
    )
    equipment_states: Dict[str, str] = Field(
        default_factory=dict,
        description="Equipment states by ID"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="State timestamp"
    )


# =============================================================================
# INTERLOCK MANAGER
# =============================================================================

class InterlockManager:
    """
    Interlock management for steam system safety.

    This manager provides interlock monitoring and permissive checking for
    the steam system. IMPORTANT: GL-003 has NO authority to modify SIF
    logic or bypass safety interlocks.

    Safety Philosophy:
        - GL-003 respects ALL existing safety envelopes
        - No control authority over Safety Instrumented Systems
        - Permissive checking before any optimization action
        - Complete interlock status monitoring and logging

    Features:
        - Permissive checking for all operation types
        - Interlock registration and monitoring
        - System state evaluation against interlocks
        - Active interlock tracking
        - Audit trail for all checks

    Attributes:
        _interlocks: Registered interlock definitions
        _active_interlocks: Currently active interlocks
        _evaluation_history: History of evaluations

    Example:
        >>> manager = InterlockManager()
        >>> manager.register_interlock(
        ...     interlock_id="IL-001",
        ...     condition=condition,
        ...     action=ActionType.PERMISSIVE_BLOCK
        ... )
        >>> result = manager.check_permissives(OperationType.SETPOINT_CHANGE)
    """

    def __init__(self):
        """Initialize InterlockManager."""
        self._interlocks: Dict[str, InterlockDefinition] = {}
        self._active_interlocks: Dict[str, ActiveInterlock] = {}
        self._evaluation_history: List[InterlockEvaluation] = []
        self._permissive_history: List[PermissiveResult] = []
        self._max_history_size = 5000

        logger.info(
            "InterlockManager initialized - GL-003 has READ-ONLY access to SIS status"
        )

    def register_interlock(
        self,
        interlock_id: str,
        condition: InterlockCondition,
        action: ActionType,
        name: str = "",
        interlock_type: InterlockType = InterlockType.OPERATIONAL,
        equipment_id: str = "",
        priority: int = 2,
        is_safety_critical: bool = False
    ) -> None:
        """
        Register an interlock for monitoring.

        Note: For safety-critical interlocks (SIF), this only registers them
        for monitoring. GL-003 cannot modify their logic or behavior.

        Args:
            interlock_id: Unique interlock identifier
            condition: Interlock trigger condition
            action: Action type when triggered
            name: Human-readable name
            interlock_type: Type of interlock
            equipment_id: Associated equipment ID
            priority: Priority (1=highest)
            is_safety_critical: Is this a SIF (read-only for GL-003)
        """
        if is_safety_critical:
            logger.info(
                f"Registering SAFETY CRITICAL interlock {interlock_id} for "
                "monitoring only - GL-003 cannot modify SIF logic"
            )

        definition = InterlockDefinition(
            interlock_id=interlock_id,
            name=name or f"Interlock {interlock_id}",
            interlock_type=interlock_type,
            equipment_id=equipment_id,
            conditions=[condition],
            action=action,
            priority=priority,
            is_safety_critical=is_safety_critical,
            can_bypass=(not is_safety_critical and interlock_type != InterlockType.SAFETY)
        )

        self._interlocks[interlock_id] = definition

        logger.info(
            f"Registered interlock {interlock_id}: type={interlock_type.value}, "
            f"action={action.value}, safety_critical={is_safety_critical}"
        )

    def check_permissives(
        self,
        operation_type: OperationType,
        equipment_id: Optional[str] = None,
        parameters: Optional[Dict[str, float]] = None
    ) -> PermissiveResult:
        """
        Check permissives for an operation.

        This method checks all relevant interlocks to determine if an
        operation is permitted. Safety interlocks always block if active.

        Args:
            operation_type: Type of operation being requested
            equipment_id: Target equipment ID (optional)
            parameters: Current parameter values (optional)

        Returns:
            PermissiveResult: Permissive check result
        """
        start_time = datetime.now()
        interlocks_checked = 0
        interlocks_blocking = []
        conditions_required = []

        # Check all active interlocks
        for interlock_id, active in self._active_interlocks.items():
            if active.state != InterlockState.ACTIVE:
                continue

            definition = self._interlocks.get(interlock_id)
            if definition is None:
                continue

            interlocks_checked += 1

            # Check if this interlock affects the requested operation
            if self._interlock_affects_operation(definition, operation_type, equipment_id):
                if active.is_safety_critical:
                    # Safety interlocks ALWAYS block
                    interlocks_blocking.append(
                        f"{active.name} (SAFETY CRITICAL): {active.message}"
                    )
                elif definition.action in (ActionType.TRIP, ActionType.PERMISSIVE_BLOCK):
                    interlocks_blocking.append(
                        f"{active.name}: {active.message}"
                    )
                elif definition.action == ActionType.LIMIT:
                    conditions_required.append(
                        f"Limited by {active.name}: {active.message}"
                    )

        # Also check registered interlocks against current parameters
        if parameters:
            for interlock_id, definition in self._interlocks.items():
                if interlock_id in self._active_interlocks:
                    continue  # Already checked as active

                interlocks_checked += 1

                for condition in definition.conditions:
                    if condition.parameter in parameters:
                        value = parameters[condition.parameter]
                        if self._evaluate_condition(condition, value):
                            if definition.is_safety_critical:
                                interlocks_blocking.append(
                                    f"{definition.name} would trigger (SAFETY): "
                                    f"{condition.description}"
                                )
                            elif definition.action == ActionType.PERMISSIVE_BLOCK:
                                interlocks_blocking.append(
                                    f"{definition.name} would trigger: "
                                    f"{condition.description}"
                                )

        # Determine permissive status
        if interlocks_blocking:
            status = PermissiveStatus.DENIED
            message = f"Operation blocked by {len(interlocks_blocking)} interlock(s)"
        elif conditions_required:
            status = PermissiveStatus.CONDITIONAL
            message = f"Operation permitted with {len(conditions_required)} condition(s)"
        else:
            status = PermissiveStatus.GRANTED
            message = "Operation permitted - all permissives satisfied"

        # Generate permissive ID
        permissive_id = hashlib.sha256(
            f"PERM_{operation_type.value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = PermissiveResult(
            permissive_id=permissive_id,
            operation_type=operation_type,
            status=status,
            interlocks_checked=interlocks_checked,
            interlocks_blocking=interlocks_blocking,
            conditions_required=conditions_required,
            message=message,
            timestamp=datetime.now()
        )

        result.provenance_hash = hashlib.sha256(
            f"{permissive_id}|{status.value}|{len(interlocks_blocking)}".encode()
        ).hexdigest()

        # Store in history
        self._permissive_history.append(result)
        if len(self._permissive_history) > self._max_history_size:
            self._permissive_history = self._permissive_history[-self._max_history_size:]

        # Log result
        if status == PermissiveStatus.DENIED:
            logger.warning(
                f"Permissive DENIED for {operation_type.value}: "
                f"{len(interlocks_blocking)} blocking interlocks"
            )
        else:
            logger.info(
                f"Permissive {status.value} for {operation_type.value}"
            )

        return result

    def evaluate_interlocks(
        self,
        system_state: SystemState
    ) -> InterlockEvaluation:
        """
        Evaluate all interlocks against current system state.

        This method evaluates all registered interlocks against the current
        system state and updates the active interlock list.

        Args:
            system_state: Current system state with parameter values

        Returns:
            InterlockEvaluation: Evaluation result
        """
        start_time = datetime.now()
        active_list = []
        safety_count = 0
        system_safe = True
        optimization_allowed = True

        for interlock_id, definition in self._interlocks.items():
            triggered_conditions = []

            # Evaluate each condition
            for condition in definition.conditions:
                if condition.parameter in system_state.parameters:
                    value = system_state.parameters[condition.parameter]
                    if self._evaluate_condition(condition, value):
                        triggered_conditions.append(condition.condition_id)

            # Update active interlock status
            if triggered_conditions:
                state = InterlockState.ACTIVE
                triggered_at = start_time

                if interlock_id in self._active_interlocks:
                    # Keep original trigger time
                    triggered_at = self._active_interlocks[interlock_id].triggered_at or start_time

                active = ActiveInterlock(
                    interlock_id=interlock_id,
                    name=definition.name,
                    state=state,
                    triggered_by=triggered_conditions,
                    triggered_at=triggered_at,
                    action_taken=definition.action.value,
                    equipment_affected=definition.equipment_id,
                    is_safety_critical=definition.is_safety_critical,
                    message=f"Triggered by: {', '.join(triggered_conditions)}"
                )

                self._active_interlocks[interlock_id] = active
                active_list.append(active)

                if definition.is_safety_critical:
                    safety_count += 1
                    system_safe = False
                    optimization_allowed = False
                elif definition.action in (ActionType.TRIP, ActionType.PERMISSIVE_BLOCK):
                    optimization_allowed = False

            else:
                # Interlock not triggered - remove from active if present
                if interlock_id in self._active_interlocks:
                    del self._active_interlocks[interlock_id]
                    logger.info(f"Interlock {interlock_id} cleared")

        # Generate evaluation ID
        evaluation_id = hashlib.sha256(
            f"EVAL_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = InterlockEvaluation(
            evaluation_id=evaluation_id,
            interlocks_evaluated=len(self._interlocks),
            active_interlocks=active_list,
            safety_interlocks_active=safety_count,
            system_safe=system_safe,
            optimization_allowed=optimization_allowed,
            timestamp=datetime.now()
        )

        result.provenance_hash = hashlib.sha256(
            f"{evaluation_id}|{system_safe}|{len(active_list)}".encode()
        ).hexdigest()

        # Store in history
        self._evaluation_history.append(result)
        if len(self._evaluation_history) > self._max_history_size:
            self._evaluation_history = self._evaluation_history[-self._max_history_size:]

        if not system_safe:
            logger.warning(
                f"System NOT SAFE: {safety_count} safety interlocks active"
            )
        elif not optimization_allowed:
            logger.info(
                f"Optimization blocked: {len(active_list)} interlocks active"
            )

        return result

    def get_active_interlocks(self) -> List[ActiveInterlock]:
        """
        Get all currently active interlocks.

        Returns:
            List of active interlocks
        """
        return list(self._active_interlocks.values())

    def get_interlock_status(
        self,
        interlock_id: str
    ) -> Optional[ActiveInterlock]:
        """
        Get status of a specific interlock.

        Args:
            interlock_id: Interlock identifier

        Returns:
            ActiveInterlock status or None if not active
        """
        return self._active_interlocks.get(interlock_id)

    def get_interlocks_by_type(
        self,
        interlock_type: InterlockType
    ) -> List[InterlockDefinition]:
        """
        Get interlocks by type.

        Args:
            interlock_type: Type of interlocks to retrieve

        Returns:
            List of interlock definitions
        """
        return [
            defn for defn in self._interlocks.values()
            if defn.interlock_type == interlock_type
        ]

    def get_safety_interlocks(self) -> List[InterlockDefinition]:
        """
        Get all safety-critical interlocks.

        Returns:
            List of safety-critical interlock definitions
        """
        return [
            defn for defn in self._interlocks.values()
            if defn.is_safety_critical
        ]

    def is_optimization_allowed(self) -> bool:
        """
        Check if optimization operations are currently allowed.

        Returns:
            True if optimization is allowed
        """
        for active in self._active_interlocks.values():
            if active.state == InterlockState.ACTIVE:
                definition = self._interlocks.get(active.interlock_id)
                if definition and (
                    definition.is_safety_critical or
                    definition.action in (ActionType.TRIP, ActionType.PERMISSIVE_BLOCK)
                ):
                    return False
        return True

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _evaluate_condition(
        self,
        condition: InterlockCondition,
        value: float
    ) -> bool:
        """
        Evaluate a condition against a value.

        Args:
            condition: Condition to evaluate
            value: Value to check

        Returns:
            True if condition is triggered (interlock should activate)
        """
        operators = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: abs(v - t) < 0.001,
            "!=": lambda v, t: abs(v - t) >= 0.001,
        }

        op_func = operators.get(condition.operator)
        if op_func is None:
            logger.error(f"Unknown operator: {condition.operator}")
            return False

        return op_func(value, condition.threshold)

    def _interlock_affects_operation(
        self,
        definition: InterlockDefinition,
        operation_type: OperationType,
        equipment_id: Optional[str]
    ) -> bool:
        """
        Check if an interlock affects a specific operation.

        Args:
            definition: Interlock definition
            operation_type: Type of operation
            equipment_id: Target equipment (optional)

        Returns:
            True if interlock affects the operation
        """
        # Safety interlocks affect all operations
        if definition.is_safety_critical:
            return True

        # Check equipment match
        if equipment_id and definition.equipment_id:
            if definition.equipment_id != equipment_id:
                return False

        # All optimization operations affected by blocking interlocks
        if operation_type == OperationType.OPTIMIZATION:
            return definition.action in (ActionType.TRIP, ActionType.PERMISSIVE_BLOCK)

        # Setpoint changes affected by limits and blocks
        if operation_type == OperationType.SETPOINT_CHANGE:
            return definition.action in (
                ActionType.TRIP, ActionType.PERMISSIVE_BLOCK, ActionType.LIMIT
            )

        return True
