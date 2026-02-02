"""
Trip Logic Module - Safety Trip Logic Validation

This module provides validation and verification of safety trip logic
for process heat applications. It ensures trip logic is correctly
configured and will function as intended.

Standards:
    - IEC 61511: Safety Instrumented Systems for Process Industries
    - NFPA 86: Standard for Ovens and Furnaces
    - NFPA 85: Boiler and Combustion Systems Hazards Code

Key Capabilities:
    - Trip condition validation
    - Logic sequence verification
    - Response time analysis
    - Voting logic validation
    - Trip priority management
    - Cause-and-effect matrix verification

Example:
    >>> validator = TripLogicValidator()
    >>> condition = TripCondition(
    ...     name="High Temperature",
    ...     variable="furnace_temp",
    ...     operator=">=",
    ...     setpoint=1200.0,
    ...     unit="degC"
    ... )
    >>> result = validator.validate_condition(condition, current_value=850.0)
    >>> print(f"Trip required: {result.trip_required}")

CRITICAL: Trip logic validation is DETERMINISTIC. NO LLM calls permitted.
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class LogicType(str, Enum):
    """Types of trip logic."""
    SIMPLE = "simple"           # Single condition
    AND = "and"                 # All conditions must be true
    OR = "or"                   # Any condition triggers trip
    VOTING = "voting"           # M-of-N voting (e.g., 2oo3)
    SEQUENCE = "sequence"       # Sequential logic (timed)
    CONDITIONAL = "conditional"  # Conditional based on mode


class ComparisonOperator(str, Enum):
    """Comparison operators for trip conditions."""
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN_RANGE = "in_range"       # Between low and high
    OUT_OF_RANGE = "out_of_range"  # Outside low and high


class TripPriority(str, Enum):
    """Trip priority levels per NFPA 86."""
    EMERGENCY = "emergency"     # Immediate shutdown, no delay
    HIGH = "high"               # Safety critical, minimal delay
    MEDIUM = "medium"           # Operational safety
    LOW = "low"                 # Equipment protection
    ALARM_ONLY = "alarm_only"   # No trip, alarm only


class TripState(str, Enum):
    """Current state of trip logic."""
    NORMAL = "normal"           # No trip condition
    PRE_ALARM = "pre_alarm"     # Warning threshold exceeded
    ALARM = "alarm"             # Alarm threshold, trip pending
    TRIPPED = "tripped"         # Trip activated
    LATCHED = "latched"         # Trip latched, requires reset
    BYPASSED = "bypassed"       # Trip bypassed


class TripCondition(BaseModel):
    """
    Single trip condition definition.

    A trip condition specifies when a safety trip should occur
    based on a process variable exceeding a setpoint.

    Attributes:
        condition_id: Unique identifier
        name: Descriptive name
        variable: Process variable name
        operator: Comparison operator
        setpoint: Trip setpoint value
        unit: Engineering unit
        deadband: Deadband to prevent chatter
        time_delay_ms: Time delay before trip
        pre_alarm_setpoint: Warning threshold (optional)
    """
    condition_id: str = Field(default="", description="Unique identifier")
    name: str = Field(..., description="Condition name")
    variable: str = Field(..., description="Process variable name")
    operator: ComparisonOperator = Field(..., description="Comparison operator")
    setpoint: float = Field(..., description="Trip setpoint value")
    setpoint_low: Optional[float] = Field(None, description="Low setpoint for range")
    unit: str = Field("", description="Engineering unit")
    deadband: float = Field(0.0, ge=0, description="Deadband value")
    time_delay_ms: float = Field(0.0, ge=0, description="Trip delay (ms)")
    pre_alarm_setpoint: Optional[float] = Field(None, description="Pre-alarm threshold")
    description: Optional[str] = Field(None, description="Condition description")
    is_latching: bool = Field(True, description="Trip latches until reset")
    requires_acknowledgment: bool = Field(True, description="Requires operator ack")
    priority: TripPriority = Field(TripPriority.HIGH, description="Trip priority")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.condition_id:
            self.condition_id = f"COND-{hash(self.name) % 10000:04d}"

    @validator('setpoint_low')
    def validate_range_setpoints(cls, v, values):
        """Validate range operators have both setpoints."""
        operator = values.get('operator')
        if operator in (ComparisonOperator.IN_RANGE, ComparisonOperator.OUT_OF_RANGE):
            if v is None:
                raise ValueError(f"setpoint_low required for {operator.value} operator")
        return v


class TripAction(BaseModel):
    """
    Action to take when trip condition is met.

    Defines what happens when a trip is triggered, including
    equipment to shutdown and sequence of actions.

    Attributes:
        action_id: Unique identifier
        name: Action name
        action_type: Type of action (shutdown, close_valve, etc.)
        target: Equipment or device to act on
        sequence_order: Order in shutdown sequence
        response_time_ms: Required response time
    """
    action_id: str = Field(default="", description="Unique identifier")
    name: str = Field(..., description="Action name")
    action_type: str = Field(..., description="Action type")
    target: str = Field(..., description="Target equipment/device")
    target_state: str = Field(..., description="Target state (OFF, CLOSED, etc.)")
    sequence_order: int = Field(1, ge=1, description="Sequence order")
    response_time_ms: float = Field(1000.0, gt=0, description="Required response time")
    is_critical: bool = Field(True, description="Critical for safety")
    description: Optional[str] = Field(None, description="Action description")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.action_id:
            self.action_id = f"ACT-{hash(self.name) % 10000:04d}"


class TripLogicBlock(BaseModel):
    """
    Block of trip logic combining conditions.

    A logic block combines multiple conditions using AND, OR,
    or voting logic to determine trip state.

    Attributes:
        block_id: Unique identifier
        name: Block name
        logic_type: Type of logic (AND, OR, VOTING)
        conditions: List of trip conditions
        voting_m: M value for MooN voting
        voting_n: N value for MooN voting
        actions: Actions when block trips
    """
    block_id: str = Field(default="", description="Unique identifier")
    name: str = Field(..., description="Block name")
    logic_type: LogicType = Field(..., description="Logic type")
    conditions: List[TripCondition] = Field(..., description="Trip conditions")
    voting_m: Optional[int] = Field(None, ge=1, description="M for MooN voting")
    voting_n: Optional[int] = Field(None, ge=1, description="N for MooN voting")
    actions: List[TripAction] = Field(default_factory=list, description="Trip actions")
    priority: TripPriority = Field(TripPriority.HIGH, description="Block priority")
    description: Optional[str] = Field(None, description="Block description")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.block_id:
            self.block_id = f"BLK-{hash(self.name) % 10000:04d}"

    @validator('voting_m', 'voting_n')
    def validate_voting(cls, v, values, field):
        """Validate voting configuration."""
        logic_type = values.get('logic_type')
        if logic_type == LogicType.VOTING:
            if v is None:
                raise ValueError(f"{field.name} required for VOTING logic")
        return v


class TripEvaluationStep(BaseModel):
    """Individual evaluation step with provenance."""
    step_number: int = Field(..., description="Step sequence number")
    description: str = Field(..., description="Step description")
    condition_id: Optional[str] = Field(None, description="Condition evaluated")
    variable: Optional[str] = Field(None, description="Variable name")
    current_value: Optional[float] = Field(None, description="Current value")
    setpoint: Optional[float] = Field(None, description="Setpoint")
    operator: Optional[str] = Field(None, description="Operator")
    condition_met: Optional[bool] = Field(None, description="Condition result")
    logic_result: Optional[bool] = Field(None, description="Logic result")
    step_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.step_hash:
            self.step_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this step."""
        hash_data = {
            "step_number": self.step_number,
            "description": self.description,
            "condition_id": self.condition_id,
            "condition_met": self.condition_met,
            "logic_result": self.logic_result,
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True, default=str).encode()
        ).hexdigest()


class TripLogicResult(BaseModel):
    """
    Result of trip logic evaluation with provenance.

    Contains the evaluation result and complete audit trail
    for regulatory compliance.
    """
    # Evaluation result
    block_id: str = Field(..., description="Logic block evaluated")
    block_name: str = Field(..., description="Block name")
    trip_required: bool = Field(..., description="Whether trip is required")
    current_state: TripState = Field(..., description="Current trip state")
    priority: TripPriority = Field(..., description="Trip priority")

    # Condition details
    conditions_evaluated: int = Field(..., description="Number of conditions")
    conditions_met: int = Field(..., description="Conditions that triggered")
    conditions_not_met: int = Field(..., description="Conditions not triggered")

    # Voting result (if applicable)
    voting_result: Optional[str] = Field(None, description="e.g., '2 of 3 met'")

    # Actions to execute
    actions_required: List[TripAction] = Field(
        default_factory=list, description="Actions to execute"
    )

    # Provenance
    evaluation_steps: List[TripEvaluationStep] = Field(
        default_factory=list, description="Evaluation audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 hash")
    input_hash: str = Field(..., description="Input hash")

    # Metadata
    evaluation_time_ms: float = Field(..., description="Evaluation time")
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

    # Warnings
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            TripState: lambda v: v.value,
            TripPriority: lambda v: v.value,
        }

    def verify_provenance(self) -> bool:
        """Verify provenance hash matches evaluation steps."""
        step_data = [
            {
                "step_number": s.step_number,
                "description": s.description,
                "condition_met": s.condition_met,
                "logic_result": s.logic_result,
            }
            for s in self.evaluation_steps
        ]
        recalculated = hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()
        return recalculated == self.provenance_hash

    def to_audit_dict(self) -> Dict[str, Any]:
        """Export as audit-ready dictionary."""
        return {
            "block": {
                "id": self.block_id,
                "name": self.block_name,
            },
            "result": {
                "trip_required": self.trip_required,
                "current_state": self.current_state.value,
                "priority": self.priority.value,
                "conditions_met": self.conditions_met,
                "conditions_total": self.conditions_evaluated,
            },
            "provenance": {
                "hash": self.provenance_hash,
                "input_hash": self.input_hash,
                "step_count": len(self.evaluation_steps),
            },
            "metadata": {
                "evaluated_at": self.evaluated_at.isoformat(),
                "evaluation_time_ms": self.evaluation_time_ms,
            },
            "warnings": self.warnings,
        }


class TripLogicValidator:
    """
    Trip Logic Validator for process safety systems.

    This class validates safety trip logic configurations and
    evaluates current process values against trip conditions.
    All evaluations are deterministic with full provenance.

    Key Methods:
        validate_condition: Validate single trip condition
        evaluate_logic_block: Evaluate complete logic block
        verify_response_time: Verify response time requirements
        validate_voting_logic: Validate voting configuration

    Example:
        >>> validator = TripLogicValidator()
        >>> condition = TripCondition(
        ...     name="High Temp",
        ...     variable="furnace_temp",
        ...     operator=ComparisonOperator.GREATER_EQUAL,
        ...     setpoint=1200.0,
        ...     unit="degC"
        ... )
        >>> result = validator.validate_condition(condition, 1150.0)
        >>> print(f"Trip required: {result.trip_required}")

    CRITICAL: All evaluations are DETERMINISTIC. NO LLM calls permitted.
    """

    VERSION = "1.0.0"

    # Response time requirements per NFPA 86 (milliseconds)
    NFPA_RESPONSE_TIMES: Dict[TripPriority, float] = {
        TripPriority.EMERGENCY: 500.0,    # 0.5 seconds
        TripPriority.HIGH: 1000.0,        # 1 second
        TripPriority.MEDIUM: 3000.0,      # 3 seconds
        TripPriority.LOW: 5000.0,         # 5 seconds
        TripPriority.ALARM_ONLY: 10000.0,  # 10 seconds (alarm only)
    }

    def __init__(self):
        """Initialize Trip Logic Validator."""
        self._steps: List[TripEvaluationStep] = []
        self._step_counter = 0
        self._start_time: Optional[float] = None
        self._warnings: List[str] = []

    def _start_evaluation(self) -> None:
        """Reset evaluation state."""
        self._steps = []
        self._step_counter = 0
        self._start_time = time.perf_counter()
        self._warnings = []

    def _record_step(
        self,
        description: str,
        condition_id: Optional[str] = None,
        variable: Optional[str] = None,
        current_value: Optional[float] = None,
        setpoint: Optional[float] = None,
        operator: Optional[str] = None,
        condition_met: Optional[bool] = None,
        logic_result: Optional[bool] = None,
    ) -> TripEvaluationStep:
        """Record an evaluation step with provenance."""
        self._step_counter += 1
        step = TripEvaluationStep(
            step_number=self._step_counter,
            description=description,
            condition_id=condition_id,
            variable=variable,
            current_value=current_value,
            setpoint=setpoint,
            operator=operator,
            condition_met=condition_met,
            logic_result=logic_result,
        )
        self._steps.append(step)
        logger.debug(f"Trip eval step {self._step_counter}: {description}")
        return step

    def _compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash of all evaluation steps."""
        step_data = [
            {
                "step_number": s.step_number,
                "description": s.description,
                "condition_met": s.condition_met,
                "logic_result": s.logic_result,
            }
            for s in self._steps
        ]
        return hashlib.sha256(
            json.dumps(step_data, sort_keys=True).encode()
        ).hexdigest()

    def _compute_input_hash(self, inputs: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of inputs."""
        serializable = {k: str(v) for k, v in inputs.items()}
        return hashlib.sha256(
            json.dumps(serializable, sort_keys=True).encode()
        ).hexdigest()

    def _get_evaluation_time_ms(self) -> float:
        """Get evaluation time in milliseconds."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000

    def validate_condition(
        self,
        condition: TripCondition,
        current_value: float,
    ) -> Tuple[bool, TripState]:
        """
        Validate a single trip condition against current value.

        Args:
            condition: Trip condition to evaluate
            current_value: Current process variable value

        Returns:
            Tuple of (condition_met, trip_state)
        """
        # Apply comparison operator
        condition_met = self._evaluate_comparison(
            current_value,
            condition.operator,
            condition.setpoint,
            condition.setpoint_low,
            condition.deadband,
        )

        # Determine trip state
        if condition_met:
            trip_state = TripState.ALARM
        elif condition.pre_alarm_setpoint is not None:
            pre_alarm_met = self._evaluate_comparison(
                current_value,
                condition.operator,
                condition.pre_alarm_setpoint,
                None,
                condition.deadband,
            )
            trip_state = TripState.PRE_ALARM if pre_alarm_met else TripState.NORMAL
        else:
            trip_state = TripState.NORMAL

        return condition_met, trip_state

    def _evaluate_comparison(
        self,
        value: float,
        operator: ComparisonOperator,
        setpoint: float,
        setpoint_low: Optional[float],
        deadband: float,
    ) -> bool:
        """
        Evaluate comparison with deadband.

        DETERMINISTIC: Pure arithmetic comparison.
        """
        if operator == ComparisonOperator.GREATER_THAN:
            return value > (setpoint - deadband)
        elif operator == ComparisonOperator.GREATER_EQUAL:
            return value >= (setpoint - deadband)
        elif operator == ComparisonOperator.LESS_THAN:
            return value < (setpoint + deadband)
        elif operator == ComparisonOperator.LESS_EQUAL:
            return value <= (setpoint + deadband)
        elif operator == ComparisonOperator.EQUAL:
            return abs(value - setpoint) <= deadband
        elif operator == ComparisonOperator.NOT_EQUAL:
            return abs(value - setpoint) > deadband
        elif operator == ComparisonOperator.IN_RANGE:
            if setpoint_low is None:
                return False
            return setpoint_low <= value <= setpoint
        elif operator == ComparisonOperator.OUT_OF_RANGE:
            if setpoint_low is None:
                return False
            return value < setpoint_low or value > setpoint
        else:
            return False

    def evaluate_logic_block(
        self,
        block: TripLogicBlock,
        current_values: Dict[str, float],
    ) -> TripLogicResult:
        """
        Evaluate a complete trip logic block.

        Args:
            block: Logic block to evaluate
            current_values: Dict of variable name to current value

        Returns:
            TripLogicResult with complete evaluation and provenance
        """
        self._start_evaluation()
        logger.info(f"Evaluating trip logic block: {block.name}")

        inputs_summary = {
            "block_id": block.block_id,
            "logic_type": block.logic_type.value,
            "condition_count": len(block.conditions),
            "values": current_values,
        }

        # Step 1: Evaluate each condition
        condition_results: List[bool] = []
        conditions_met = 0

        for condition in block.conditions:
            value = current_values.get(condition.variable)

            if value is None:
                self._warnings.append(
                    f"Missing value for variable '{condition.variable}' "
                    f"in condition '{condition.name}'"
                )
                condition_results.append(False)
                self._record_step(
                    description=f"Condition '{condition.name}': VALUE MISSING",
                    condition_id=condition.condition_id,
                    variable=condition.variable,
                    condition_met=False,
                )
                continue

            met, state = self.validate_condition(condition, value)
            condition_results.append(met)

            if met:
                conditions_met += 1

            self._record_step(
                description=f"Evaluate condition: {condition.name}",
                condition_id=condition.condition_id,
                variable=condition.variable,
                current_value=value,
                setpoint=condition.setpoint,
                operator=condition.operator.value,
                condition_met=met,
            )

        # Step 2: Apply logic type
        trip_required = self._apply_logic(
            block.logic_type,
            condition_results,
            block.voting_m,
            block.voting_n,
        )

        self._record_step(
            description=f"Apply {block.logic_type.value} logic",
            logic_result=trip_required,
        )

        # Step 3: Determine current state
        if trip_required:
            current_state = TripState.ALARM
        elif conditions_met > 0:
            current_state = TripState.PRE_ALARM
        else:
            current_state = TripState.NORMAL

        # Step 4: Determine voting result string
        voting_result = None
        if block.logic_type == LogicType.VOTING:
            voting_result = f"{conditions_met} of {block.voting_n} conditions met"
            if block.voting_m:
                voting_result += f" (need {block.voting_m})"

        # Step 5: Get actions if trip required
        actions_required = block.actions if trip_required else []

        # Build result
        result = TripLogicResult(
            block_id=block.block_id,
            block_name=block.name,
            trip_required=trip_required,
            current_state=current_state,
            priority=block.priority,
            conditions_evaluated=len(block.conditions),
            conditions_met=conditions_met,
            conditions_not_met=len(block.conditions) - conditions_met,
            voting_result=voting_result,
            actions_required=actions_required,
            evaluation_steps=self._steps.copy(),
            provenance_hash=self._compute_provenance_hash(),
            input_hash=self._compute_input_hash(inputs_summary),
            evaluation_time_ms=self._get_evaluation_time_ms(),
            warnings=self._warnings.copy(),
        )

        logger.info(
            f"Trip logic evaluation complete: {block.name}, "
            f"trip_required={trip_required}, "
            f"conditions_met={conditions_met}/{len(block.conditions)}"
        )
        return result

    def _apply_logic(
        self,
        logic_type: LogicType,
        results: List[bool],
        voting_m: Optional[int],
        voting_n: Optional[int],
    ) -> bool:
        """
        Apply logic type to condition results.

        DETERMINISTIC: Pure boolean logic.
        """
        if not results:
            return False

        if logic_type == LogicType.SIMPLE:
            return results[0] if results else False

        elif logic_type == LogicType.AND:
            return all(results)

        elif logic_type == LogicType.OR:
            return any(results)

        elif logic_type == LogicType.VOTING:
            if voting_m is None or voting_n is None:
                return any(results)
            met_count = sum(1 for r in results if r)
            return met_count >= voting_m

        elif logic_type == LogicType.SEQUENCE:
            # For sequence logic, all must be true in order
            # This is simplified - real implementation would check timing
            return all(results)

        elif logic_type == LogicType.CONDITIONAL:
            # Conditional logic would depend on process mode
            # Simplified to OR logic here
            return any(results)

        return False

    def verify_response_time(
        self,
        block: TripLogicBlock,
        measured_response_ms: float,
    ) -> Tuple[bool, str]:
        """
        Verify response time meets NFPA requirements.

        Args:
            block: Logic block to verify
            measured_response_ms: Measured response time

        Returns:
            Tuple of (meets_requirement, message)
        """
        required_ms = self.NFPA_RESPONSE_TIMES.get(
            block.priority, 1000.0
        )

        meets_requirement = measured_response_ms <= required_ms

        if meets_requirement:
            message = (
                f"Response time {measured_response_ms:.0f}ms meets "
                f"requirement of {required_ms:.0f}ms for {block.priority.value} priority"
            )
        else:
            message = (
                f"FAILURE: Response time {measured_response_ms:.0f}ms exceeds "
                f"requirement of {required_ms:.0f}ms for {block.priority.value} priority"
            )

        return meets_requirement, message

    def validate_voting_logic(
        self,
        voting_m: int,
        voting_n: int,
        architecture: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate voting logic configuration.

        Args:
            voting_m: M value (required votes)
            voting_n: N value (total channels)
            architecture: Architecture string (e.g., "2oo3")

        Returns:
            Tuple of (is_valid, messages)
        """
        messages = []
        is_valid = True

        # Basic validation
        if voting_m < 1:
            messages.append("M must be >= 1")
            is_valid = False

        if voting_n < 1:
            messages.append("N must be >= 1")
            is_valid = False

        if voting_m > voting_n:
            messages.append("M cannot exceed N")
            is_valid = False

        # Check against architecture string
        expected_arch = f"{voting_m}oo{voting_n}"
        if architecture != expected_arch:
            messages.append(
                f"Architecture '{architecture}' does not match "
                f"M={voting_m}, N={voting_n} (expected '{expected_arch}')"
            )
            is_valid = False

        # Validate common architectures
        valid_architectures = ["1oo1", "1oo2", "2oo2", "1oo3", "2oo3", "2oo4"]
        if architecture not in valid_architectures:
            messages.append(
                f"Non-standard architecture '{architecture}'. "
                f"Standard architectures: {valid_architectures}"
            )
            # Not necessarily invalid, just unusual

        # Check for safety vs availability trade-off
        if voting_m == 1:
            messages.append(
                f"1oo{voting_n} prioritizes safety over availability "
                "(trips on any single failure)"
            )
        elif voting_m == voting_n:
            messages.append(
                f"{voting_m}oo{voting_n} prioritizes availability over safety "
                "(requires all channels to trip)"
            )

        return is_valid, messages

    def create_cause_effect_entry(
        self,
        cause: TripCondition,
        effects: List[TripAction],
    ) -> Dict[str, Any]:
        """
        Create a cause-and-effect matrix entry.

        Args:
            cause: Trip condition (cause)
            effects: List of actions (effects)

        Returns:
            Dictionary representing C&E matrix entry
        """
        return {
            "cause": {
                "id": cause.condition_id,
                "name": cause.name,
                "variable": cause.variable,
                "setpoint": f"{cause.operator.value} {cause.setpoint} {cause.unit}",
                "priority": cause.priority.value,
            },
            "effects": [
                {
                    "id": effect.action_id,
                    "name": effect.name,
                    "target": effect.target,
                    "action": effect.target_state,
                    "sequence": effect.sequence_order,
                }
                for effect in sorted(effects, key=lambda e: e.sequence_order)
            ],
            "response_time_required_ms": self.NFPA_RESPONSE_TIMES.get(
                cause.priority, 1000.0
            ),
        }

    def validate_shutdown_sequence(
        self,
        actions: List[TripAction],
    ) -> Tuple[bool, List[str]]:
        """
        Validate shutdown sequence ordering.

        Args:
            actions: List of shutdown actions

        Returns:
            Tuple of (is_valid, messages)
        """
        messages = []
        is_valid = True

        if not actions:
            messages.append("No actions defined in sequence")
            return True, messages

        # Check sequence numbers are unique
        sequence_nums = [a.sequence_order for a in actions]
        if len(sequence_nums) != len(set(sequence_nums)):
            messages.append("Duplicate sequence numbers found")
            is_valid = False

        # Sort by sequence
        sorted_actions = sorted(actions, key=lambda a: a.sequence_order)

        # Check critical actions are early in sequence
        for i, action in enumerate(sorted_actions):
            if action.is_critical and i > len(sorted_actions) // 2:
                messages.append(
                    f"Critical action '{action.name}' is late in sequence "
                    f"(position {i+1} of {len(sorted_actions)})"
                )

        # Check response time accumulation
        total_time = sum(a.response_time_ms for a in actions)
        if total_time > 5000:  # 5 seconds total
            messages.append(
                f"Total sequence time ({total_time:.0f}ms) exceeds 5 seconds"
            )

        return is_valid, messages
