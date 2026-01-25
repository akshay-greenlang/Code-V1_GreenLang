"""
SafeStateManager - Safe State Definitions and Transitions

This module implements safe state management per IEC 61511-1 Clause 10.3.
Safe states are the fundamental requirement of any Safety Instrumented System -
the condition the process must reach to prevent or mitigate a hazardous event.

Key concepts:
- Safe State: A condition where the hazardous event cannot occur
- Safe State Transition: The sequence of actions to reach a safe state
- Process Safety Time (PST): Time available to reach safe state

Reference: IEC 61511-1 Clause 10.3.3, Clause 3.2.65

Example:
    >>> from greenlang.safety.srs.safe_state_manager import SafeStateManager
    >>> manager = SafeStateManager()
    >>> safe_state = manager.define_safe_state(
    ...     state_id="SS-001",
    ...     name="Burner Shutdown",
    ...     description="All burners extinguished, fuel valves closed"
    ... )
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class SafeStateType(str, Enum):
    """Types of safe states per IEC 61511."""

    SHUTDOWN = "shutdown"  # Complete process shutdown
    PARTIAL_SHUTDOWN = "partial_shutdown"  # Partial/sectional shutdown
    TRIP_TO_IDLE = "trip_to_idle"  # Trip to idle/standby state
    ISOLATION = "isolation"  # Isolate hazardous section
    DEPRESSURIZATION = "depressurization"  # Controlled depressurization
    VENTING = "venting"  # Controlled venting
    DIVERSION = "diversion"  # Divert to safe location
    CONTAINMENT = "containment"  # Contain hazardous material
    NEUTRALIZATION = "neutralization"  # Neutralize hazardous condition


class TransitionType(str, Enum):
    """Types of state transitions."""

    IMMEDIATE = "immediate"  # Immediate de-energize
    SEQUENCED = "sequenced"  # Sequenced shutdown
    TIMED = "timed"  # Time-delayed transition
    CONDITIONAL = "conditional"  # Conditional on process state


class SafeStateAction(BaseModel):
    """Individual action in a safe state transition."""

    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Action identifier"
    )
    sequence_number: int = Field(
        ...,
        ge=1,
        description="Action sequence number"
    )
    description: str = Field(
        ...,
        description="Action description"
    )
    equipment_tag: str = Field(
        ...,
        description="Equipment tag for action"
    )
    target_state: str = Field(
        ...,
        description="Target state for equipment (e.g., 'CLOSED', 'OFF')"
    )
    transition_type: TransitionType = Field(
        default=TransitionType.IMMEDIATE,
        description="Type of transition"
    )
    max_time_ms: float = Field(
        default=1000.0,
        gt=0,
        description="Maximum time to complete action (ms)"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Actions this depends on (action_ids)"
    )
    verify_state: bool = Field(
        default=True,
        description="Verify state after action"
    )
    failure_action: str = Field(
        default="alarm",
        description="Action if this step fails"
    )


class SafeState(BaseModel):
    """Safe state definition per IEC 61511."""

    state_id: str = Field(
        ...,
        description="Safe state identifier"
    )
    name: str = Field(
        ...,
        description="Safe state name"
    )
    description: str = Field(
        ...,
        description="Detailed description of safe state"
    )
    state_type: SafeStateType = Field(
        ...,
        description="Type of safe state"
    )
    hazard_mitigated: str = Field(
        ...,
        description="Hazard this safe state mitigates"
    )
    process_conditions: Dict[str, str] = Field(
        default_factory=dict,
        description="Process conditions defining safe state"
    )
    equipment_states: Dict[str, str] = Field(
        default_factory=dict,
        description="Equipment states in safe condition"
    )
    verification_points: List[str] = Field(
        default_factory=list,
        description="Points to verify safe state achieved"
    )
    manual_reset_required: bool = Field(
        default=True,
        description="Manual reset required to exit safe state"
    )
    restart_procedure_ref: str = Field(
        default="",
        description="Reference to restart procedure"
    )
    created_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation date"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafeStateTransition(BaseModel):
    """Transition sequence to reach a safe state."""

    transition_id: str = Field(
        default_factory=lambda: f"TR-{uuid.uuid4().hex[:8].upper()}",
        description="Transition identifier"
    )
    from_state: str = Field(
        default="OPERATING",
        description="Starting state"
    )
    to_state: str = Field(
        ...,
        description="Target safe state ID"
    )
    trigger_condition: str = Field(
        ...,
        description="Condition that triggers transition"
    )
    actions: List[SafeStateAction] = Field(
        default_factory=list,
        description="Ordered list of transition actions"
    )
    total_transition_time_ms: float = Field(
        default=0.0,
        description="Total time to complete transition"
    )
    is_fail_safe: bool = Field(
        default=True,
        description="Is transition fail-safe (de-energize-to-trip)"
    )
    partial_transition_allowed: bool = Field(
        default=False,
        description="Can transition stop partway?"
    )
    parallel_actions_allowed: bool = Field(
        default=False,
        description="Can actions execute in parallel?"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        pass


class SafeStateManager:
    """
    Safe State Manager for Safety Instrumented Systems.

    Manages safe state definitions and transitions per IEC 61511.
    Ensures:
    - Clear safe state definitions
    - Deterministic transition sequences
    - Verification requirements
    - Manual reset requirements

    The manager follows zero-hallucination principles:
    - All transitions are deterministic
    - No ambiguous states
    - Complete audit trail

    Attributes:
        safe_states: Dict of state_id to SafeState
        transitions: Dict of transition_id to SafeStateTransition

    Example:
        >>> manager = SafeStateManager()
        >>> state = manager.define_safe_state("SS-001", "Shutdown", ...)
        >>> transition = manager.create_transition("SS-001", actions)
    """

    def __init__(self):
        """Initialize SafeStateManager."""
        self.safe_states: Dict[str, SafeState] = {}
        self.transitions: Dict[str, SafeStateTransition] = {}
        logger.info("SafeStateManager initialized")

    def define_safe_state(
        self,
        state_id: str,
        name: str,
        description: str,
        state_type: SafeStateType,
        hazard_mitigated: str,
        process_conditions: Optional[Dict[str, str]] = None,
        equipment_states: Optional[Dict[str, str]] = None,
        verification_points: Optional[List[str]] = None,
        manual_reset_required: bool = True,
        restart_procedure_ref: str = ""
    ) -> SafeState:
        """
        Define a new safe state.

        Args:
            state_id: Unique identifier for safe state
            name: Safe state name
            description: Detailed description
            state_type: Type of safe state
            hazard_mitigated: Hazard this state mitigates
            process_conditions: Dict of process variable to condition
            equipment_states: Dict of equipment tag to state
            verification_points: List of verification requirements
            manual_reset_required: Require manual reset
            restart_procedure_ref: Reference to restart procedure

        Returns:
            SafeState object

        Raises:
            ValueError: If state_id already exists
        """
        logger.info(f"Defining safe state: {state_id} - {name}")

        if state_id in self.safe_states:
            raise ValueError(f"Safe state {state_id} already exists")

        safe_state = SafeState(
            state_id=state_id,
            name=name,
            description=description,
            state_type=state_type,
            hazard_mitigated=hazard_mitigated,
            process_conditions=process_conditions or {},
            equipment_states=equipment_states or {},
            verification_points=verification_points or [],
            manual_reset_required=manual_reset_required,
            restart_procedure_ref=restart_procedure_ref,
        )

        # Calculate provenance hash
        safe_state.provenance_hash = self._calculate_state_provenance(safe_state)

        self.safe_states[state_id] = safe_state

        logger.info(f"Safe state defined: {state_id}")

        return safe_state

    def create_transition(
        self,
        to_state_id: str,
        actions: List[SafeStateAction],
        from_state: str = "OPERATING",
        trigger_condition: str = "",
        is_fail_safe: bool = True,
        partial_transition_allowed: bool = False,
        parallel_actions_allowed: bool = False
    ) -> SafeStateTransition:
        """
        Create a transition to a safe state.

        Args:
            to_state_id: Target safe state ID
            actions: List of transition actions
            from_state: Starting state
            trigger_condition: Condition that triggers transition
            is_fail_safe: Is transition fail-safe
            partial_transition_allowed: Can transition stop partway
            parallel_actions_allowed: Can actions run in parallel

        Returns:
            SafeStateTransition object

        Raises:
            ValueError: If target state doesn't exist
        """
        logger.info(f"Creating transition to {to_state_id}")

        if to_state_id not in self.safe_states:
            raise ValueError(f"Safe state {to_state_id} not found")

        # Calculate total transition time
        if parallel_actions_allowed:
            # Max of all action times
            total_time = max(a.max_time_ms for a in actions) if actions else 0
        else:
            # Sum of all action times
            total_time = sum(a.max_time_ms for a in actions)

        transition = SafeStateTransition(
            from_state=from_state,
            to_state=to_state_id,
            trigger_condition=trigger_condition,
            actions=sorted(actions, key=lambda a: a.sequence_number),
            total_transition_time_ms=total_time,
            is_fail_safe=is_fail_safe,
            partial_transition_allowed=partial_transition_allowed,
            parallel_actions_allowed=parallel_actions_allowed,
        )

        # Calculate provenance hash
        transition.provenance_hash = self._calculate_transition_provenance(
            transition
        )

        self.transitions[transition.transition_id] = transition

        logger.info(
            f"Transition created: {transition.transition_id}, "
            f"total time: {total_time}ms"
        )

        return transition

    def validate_transition(
        self,
        transition: SafeStateTransition
    ) -> Dict[str, Any]:
        """
        Validate a transition sequence.

        Checks:
        - All dependencies are satisfied
        - Sequence is deterministic
        - Times are realistic
        - Fail-safe principles followed

        Args:
            transition: SafeStateTransition to validate

        Returns:
            Validation result dictionary
        """
        logger.info(f"Validating transition {transition.transition_id}")

        errors: List[str] = []
        warnings: List[str] = []

        # Check target state exists
        if transition.to_state not in self.safe_states:
            errors.append(f"Target state {transition.to_state} not defined")

        # Check action dependencies
        action_ids = {a.action_id for a in transition.actions}
        for action in transition.actions:
            for dep in action.dependencies:
                if dep not in action_ids:
                    errors.append(
                        f"Action {action.action_id} depends on undefined "
                        f"action {dep}"
                    )

        # Check sequence numbers
        seq_numbers = [a.sequence_number for a in transition.actions]
        if len(seq_numbers) != len(set(seq_numbers)):
            warnings.append("Duplicate sequence numbers detected")

        # Check fail-safe
        if not transition.is_fail_safe:
            warnings.append(
                "Transition is not fail-safe. Consider de-energize-to-trip."
            )

        # Check total time
        if transition.total_transition_time_ms > 5000:
            warnings.append(
                f"Total transition time {transition.total_transition_time_ms}ms "
                f"may be too long for some applications"
            )

        is_valid = len(errors) == 0

        return {
            "transition_id": transition.transition_id,
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "action_count": len(transition.actions),
            "total_time_ms": transition.total_transition_time_ms,
        }

    def get_safe_state(self, state_id: str) -> Optional[SafeState]:
        """Get safe state by ID."""
        return self.safe_states.get(state_id)

    def get_transitions_to_state(
        self,
        state_id: str
    ) -> List[SafeStateTransition]:
        """Get all transitions to a specific safe state."""
        return [
            t for t in self.transitions.values()
            if t.to_state == state_id
        ]

    def get_equipment_safe_states(
        self,
        equipment_tag: str
    ) -> Dict[str, str]:
        """
        Get safe states for specific equipment across all safe states.

        Args:
            equipment_tag: Equipment tag to look up

        Returns:
            Dict of safe_state_id to equipment state
        """
        result = {}
        for state_id, state in self.safe_states.items():
            if equipment_tag in state.equipment_states:
                result[state_id] = state.equipment_states[equipment_tag]
        return result

    def simulate_transition(
        self,
        transition: SafeStateTransition
    ) -> List[Dict[str, Any]]:
        """
        Simulate a transition execution.

        Returns step-by-step execution with timing.

        Args:
            transition: Transition to simulate

        Returns:
            List of execution steps with timing
        """
        logger.info(f"Simulating transition {transition.transition_id}")

        steps = []
        current_time_ms = 0.0
        completed_actions: Set[str] = set()

        # Sort actions by sequence
        sorted_actions = sorted(
            transition.actions,
            key=lambda a: a.sequence_number
        )

        for action in sorted_actions:
            # Check dependencies
            deps_satisfied = all(
                dep in completed_actions
                for dep in action.dependencies
            )

            if not deps_satisfied:
                steps.append({
                    "sequence": action.sequence_number,
                    "action_id": action.action_id,
                    "status": "BLOCKED",
                    "reason": "Dependencies not satisfied",
                    "start_time_ms": current_time_ms,
                })
                continue

            # Execute action
            step = {
                "sequence": action.sequence_number,
                "action_id": action.action_id,
                "equipment": action.equipment_tag,
                "target_state": action.target_state,
                "start_time_ms": current_time_ms,
                "duration_ms": action.max_time_ms,
                "end_time_ms": current_time_ms + action.max_time_ms,
                "status": "COMPLETED",
            }

            steps.append(step)
            completed_actions.add(action.action_id)

            if not transition.parallel_actions_allowed:
                current_time_ms += action.max_time_ms

        # Final step - verify safe state
        target_state = self.safe_states.get(transition.to_state)
        if target_state:
            steps.append({
                "sequence": len(sorted_actions) + 1,
                "action_id": "VERIFY",
                "description": f"Verify {target_state.name} achieved",
                "verification_points": target_state.verification_points,
                "start_time_ms": current_time_ms,
                "status": "VERIFY_REQUIRED",
            })

        return steps

    def export_state_diagram(self) -> str:
        """
        Export state diagram in Mermaid format.

        Returns:
            Mermaid diagram string
        """
        lines = ["stateDiagram-v2"]

        # Add states
        for state_id, state in self.safe_states.items():
            lines.append(f"    {state_id}: {state.name}")

        # Add transitions
        for transition in self.transitions.values():
            trigger = transition.trigger_condition or "trigger"
            lines.append(
                f"    {transition.from_state} --> {transition.to_state}: {trigger}"
            )

        return "\n".join(lines)

    def _calculate_state_provenance(self, state: SafeState) -> str:
        """Calculate SHA-256 provenance hash for safe state."""
        provenance_str = (
            f"{state.state_id}|{state.name}|{state.state_type}|"
            f"{len(state.equipment_states)}|{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_transition_provenance(
        self,
        transition: SafeStateTransition
    ) -> str:
        """Calculate SHA-256 provenance hash for transition."""
        provenance_str = (
            f"{transition.transition_id}|{transition.to_state}|"
            f"{len(transition.actions)}|{transition.total_transition_time_ms}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
