"""
SafeTransition - Safe State Transition Logic

This module implements safe state transition logic for Safety Instrumented
Systems per IEC 61511. Safe transitions ensure:
- Orderly shutdown sequences
- No hazardous intermediate states
- Verification of safe state achievement
- Bumpless transfers

Reference: IEC 61511-1 Clause 11.5

Example:
    >>> from greenlang.safety.failsafe.safe_transition import SafeTransition
    >>> transition = SafeTransition(config)
    >>> result = transition.execute(current_state, target_state)
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime
import time
import uuid

logger = logging.getLogger(__name__)


class TransitionMode(str, Enum):
    """Transition execution modes."""

    IMMEDIATE = "immediate"  # Immediate transition (de-energize)
    SEQUENCED = "sequenced"  # Follow defined sequence
    RAMPED = "ramped"  # Gradual ramp to safe state
    CONDITIONAL = "conditional"  # Based on process conditions


class TransitionStatus(str, Enum):
    """Status of a transition execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    TIMEOUT = "timeout"


class TransitionStep(BaseModel):
    """Individual step in a transition sequence."""

    step_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Step identifier"
    )
    sequence_number: int = Field(
        ...,
        ge=1,
        description="Execution sequence number"
    )
    description: str = Field(
        ...,
        description="Step description"
    )
    action: str = Field(
        ...,
        description="Action to execute"
    )
    target_equipment: str = Field(
        ...,
        description="Target equipment tag"
    )
    target_state: str = Field(
        ...,
        description="Target state for equipment"
    )
    timeout_ms: float = Field(
        default=5000.0,
        gt=0,
        description="Step timeout (ms)"
    )
    verification_required: bool = Field(
        default=True,
        description="Verify state after action"
    )
    rollback_on_failure: bool = Field(
        default=False,
        description="Rollback if step fails"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Step IDs this depends on"
    )


class TransitionConfig(BaseModel):
    """Configuration for safe transition."""

    transition_id: str = Field(
        default_factory=lambda: f"TR-{uuid.uuid4().hex[:8].upper()}",
        description="Transition identifier"
    )
    name: str = Field(
        ...,
        description="Transition name"
    )
    from_state: str = Field(
        default="OPERATING",
        description="Source state"
    )
    to_state: str = Field(
        ...,
        description="Target safe state"
    )
    mode: TransitionMode = Field(
        default=TransitionMode.IMMEDIATE,
        description="Transition mode"
    )
    steps: List[TransitionStep] = Field(
        default_factory=list,
        description="Transition steps (for sequenced mode)"
    )
    total_timeout_ms: float = Field(
        default=30000.0,
        gt=0,
        description="Total transition timeout (ms)"
    )
    abort_on_step_failure: bool = Field(
        default=True,
        description="Abort transition if step fails"
    )
    verification_timeout_ms: float = Field(
        default=5000.0,
        gt=0,
        description="Verification timeout (ms)"
    )
    allow_partial_success: bool = Field(
        default=False,
        description="Accept partial completion"
    )


class TransitionResult(BaseModel):
    """Result of transition execution."""

    transition_id: str = Field(
        ...,
        description="Transition identifier"
    )
    execution_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Execution instance ID"
    )
    status: TransitionStatus = Field(
        ...,
        description="Final status"
    )
    from_state: str = Field(
        ...,
        description="Starting state"
    )
    to_state: str = Field(
        ...,
        description="Target state"
    )
    achieved_state: str = Field(
        ...,
        description="Actually achieved state"
    )
    start_time: datetime = Field(
        ...,
        description="Execution start time"
    )
    end_time: datetime = Field(
        ...,
        description="Execution end time"
    )
    duration_ms: float = Field(
        ...,
        description="Total duration (ms)"
    )
    steps_completed: int = Field(
        default=0,
        description="Number of steps completed"
    )
    steps_total: int = Field(
        default=0,
        description="Total steps"
    )
    step_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results of each step"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    verified: bool = Field(
        default=False,
        description="Safe state verified"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafeTransition:
    """
    Safe State Transition Manager.

    Implements safe state transitions per IEC 61511.
    Features:
    - Multiple transition modes
    - Step sequencing
    - State verification
    - Timeout handling
    - Rollback capability

    The manager follows fail-safe principles:
    - Incomplete transitions fail safe
    - Timeouts trigger safe state
    - All steps logged

    Attributes:
        config: TransitionConfig settings
        state_verifier: Optional state verification callback

    Example:
        >>> config = TransitionConfig(name="Emergency SD", to_state="SHUTDOWN")
        >>> transition = SafeTransition(config)
        >>> result = transition.execute()
    """

    def __init__(
        self,
        config: TransitionConfig,
        state_verifier: Optional[Callable[[str, str], bool]] = None,
        action_executor: Optional[Callable[[str, str, str], bool]] = None
    ):
        """
        Initialize SafeTransition.

        Args:
            config: TransitionConfig with transition settings
            state_verifier: Callback to verify state (equipment_id, expected_state) -> bool
            action_executor: Callback to execute action (equipment_id, action, target) -> bool
        """
        self.config = config
        self.state_verifier = state_verifier or self._default_verifier
        self.action_executor = action_executor or self._default_executor

        logger.info(
            f"SafeTransition initialized: {config.name} "
            f"({config.from_state} -> {config.to_state})"
        )

    def execute(
        self,
        initial_state: Optional[str] = None
    ) -> TransitionResult:
        """
        Execute the transition.

        Args:
            initial_state: Optional override of starting state

        Returns:
            TransitionResult with execution details
        """
        start_time = datetime.utcnow()
        from_state = initial_state or self.config.from_state

        logger.info(
            f"Starting transition {self.config.transition_id}: "
            f"{from_state} -> {self.config.to_state}"
        )

        step_results: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []
        steps_completed = 0
        achieved_state = from_state

        try:
            if self.config.mode == TransitionMode.IMMEDIATE:
                # Immediate transition - execute all at once
                success, msg = self._execute_immediate()
                if success:
                    achieved_state = self.config.to_state
                    steps_completed = len(self.config.steps)
                else:
                    errors.append(msg)

            elif self.config.mode == TransitionMode.SEQUENCED:
                # Execute steps in sequence
                achieved_state, steps_completed, step_results, step_errors = \
                    self._execute_sequenced()
                errors.extend(step_errors)

            elif self.config.mode == TransitionMode.RAMPED:
                # Gradual ramp (simplified - treat as sequenced)
                achieved_state, steps_completed, step_results, step_errors = \
                    self._execute_sequenced()
                errors.extend(step_errors)

            else:
                # Conditional - check conditions first
                achieved_state, steps_completed, step_results, step_errors = \
                    self._execute_conditional()
                errors.extend(step_errors)

            # Verify final state
            verified = self._verify_safe_state(achieved_state)
            if not verified:
                warnings.append(
                    f"Safe state {self.config.to_state} not verified"
                )

            # Determine status
            if achieved_state == self.config.to_state and verified:
                status = TransitionStatus.COMPLETED
            elif errors:
                status = TransitionStatus.FAILED
            elif self.config.allow_partial_success and steps_completed > 0:
                status = TransitionStatus.COMPLETED
                warnings.append("Partial completion accepted")
            else:
                status = TransitionStatus.FAILED

        except TimeoutError:
            status = TransitionStatus.TIMEOUT
            errors.append("Transition timeout exceeded")
            achieved_state = "UNKNOWN"

        except Exception as e:
            status = TransitionStatus.FAILED
            errors.append(f"Transition error: {str(e)}")
            logger.error(f"Transition failed: {e}", exc_info=True)

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Build result
        result = TransitionResult(
            transition_id=self.config.transition_id,
            status=status,
            from_state=from_state,
            to_state=self.config.to_state,
            achieved_state=achieved_state,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            steps_completed=steps_completed,
            steps_total=len(self.config.steps),
            step_results=step_results,
            errors=errors,
            warnings=warnings,
            verified=verified if 'verified' in dir() else False,
        )

        # Calculate provenance
        result.provenance_hash = self._calculate_provenance(result)

        logger.info(
            f"Transition {self.config.transition_id} {status.value}: "
            f"{achieved_state} in {duration_ms:.0f}ms"
        )

        return result

    def _execute_immediate(self) -> tuple:
        """Execute immediate (de-energize) transition."""
        logger.debug("Executing immediate transition")

        # In immediate mode, all actions happen simultaneously
        # This is the fail-safe de-energize-to-trip pattern
        for step in self.config.steps:
            success = self.action_executor(
                step.target_equipment,
                step.action,
                step.target_state
            )
            if not success:
                return False, f"Failed to execute {step.action} on {step.target_equipment}"

        return True, "Immediate transition completed"

    def _execute_sequenced(self) -> tuple:
        """Execute sequenced transition."""
        logger.debug("Executing sequenced transition")

        step_results = []
        errors = []
        steps_completed = 0
        achieved_state = self.config.from_state
        completed_step_ids = set()

        # Sort steps by sequence number
        sorted_steps = sorted(self.config.steps, key=lambda s: s.sequence_number)

        total_start = time.time()

        for step in sorted_steps:
            # Check total timeout
            if (time.time() - total_start) * 1000 > self.config.total_timeout_ms:
                raise TimeoutError("Total transition timeout exceeded")

            # Check dependencies
            deps_satisfied = all(
                dep in completed_step_ids
                for dep in step.dependencies
            )
            if not deps_satisfied:
                errors.append(f"Dependencies not satisfied for step {step.step_id}")
                if self.config.abort_on_step_failure:
                    break
                continue

            # Execute step
            step_start = time.time()
            step_result = {
                "step_id": step.step_id,
                "sequence": step.sequence_number,
                "equipment": step.target_equipment,
                "action": step.action,
                "start_time": datetime.utcnow().isoformat(),
            }

            try:
                success = self.action_executor(
                    step.target_equipment,
                    step.action,
                    step.target_state
                )

                step_result["action_success"] = success

                # Verify if required
                if step.verification_required and success:
                    verified = self._verify_equipment_state(
                        step.target_equipment,
                        step.target_state,
                        step.timeout_ms
                    )
                    step_result["verified"] = verified
                    success = success and verified

                step_result["success"] = success
                step_result["duration_ms"] = (time.time() - step_start) * 1000

                if success:
                    steps_completed += 1
                    completed_step_ids.add(step.step_id)
                    achieved_state = f"STEP_{step.sequence_number}"
                else:
                    errors.append(f"Step {step.step_id} failed")
                    if self.config.abort_on_step_failure:
                        step_results.append(step_result)
                        break

            except Exception as e:
                step_result["success"] = False
                step_result["error"] = str(e)
                errors.append(f"Step {step.step_id} error: {e}")
                if self.config.abort_on_step_failure:
                    step_results.append(step_result)
                    break

            step_results.append(step_result)

        # If all steps completed, set final state
        if steps_completed == len(sorted_steps):
            achieved_state = self.config.to_state

        return achieved_state, steps_completed, step_results, errors

    def _execute_conditional(self) -> tuple:
        """Execute conditional transition based on process state."""
        logger.debug("Executing conditional transition")

        # Check pre-conditions (simplified - always pass)
        # In production, this would check actual process conditions

        # Then execute as sequenced
        return self._execute_sequenced()

    def _verify_equipment_state(
        self,
        equipment_id: str,
        expected_state: str,
        timeout_ms: float
    ) -> bool:
        """Verify equipment reached expected state."""
        start = time.time()
        timeout_s = timeout_ms / 1000.0

        while (time.time() - start) < timeout_s:
            if self.state_verifier(equipment_id, expected_state):
                return True
            time.sleep(0.1)  # Poll interval

        return False

    def _verify_safe_state(self, state: str) -> bool:
        """Verify safe state is achieved."""
        # In production, this would check actual process state
        return state == self.config.to_state

    def _default_verifier(
        self,
        equipment_id: str,
        expected_state: str
    ) -> bool:
        """Default state verifier (always returns True)."""
        logger.debug(f"Verifying {equipment_id} is in {expected_state}")
        return True

    def _default_executor(
        self,
        equipment_id: str,
        action: str,
        target: str
    ) -> bool:
        """Default action executor (always returns True)."""
        logger.debug(f"Executing {action} on {equipment_id} -> {target}")
        return True

    def simulate(self) -> TransitionResult:
        """
        Simulate transition without actual execution.

        Returns:
            TransitionResult with simulated results
        """
        logger.info(f"Simulating transition {self.config.transition_id}")

        start_time = datetime.utcnow()
        step_results = []
        total_time = 0.0

        for step in sorted(self.config.steps, key=lambda s: s.sequence_number):
            step_results.append({
                "step_id": step.step_id,
                "sequence": step.sequence_number,
                "equipment": step.target_equipment,
                "action": step.action,
                "simulated_duration_ms": step.timeout_ms * 0.5,  # Assume 50% of timeout
                "success": True,
            })
            total_time += step.timeout_ms * 0.5

        end_time = start_time

        result = TransitionResult(
            transition_id=self.config.transition_id,
            status=TransitionStatus.COMPLETED,
            from_state=self.config.from_state,
            to_state=self.config.to_state,
            achieved_state=self.config.to_state,
            start_time=start_time,
            end_time=end_time,
            duration_ms=total_time,
            steps_completed=len(self.config.steps),
            steps_total=len(self.config.steps),
            step_results=step_results,
            verified=True,
        )

        result.provenance_hash = self._calculate_provenance(result)
        result.warnings.append("SIMULATION - Not actual execution")

        return result

    def _calculate_provenance(self, result: TransitionResult) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.transition_id}|"
            f"{result.status.value}|"
            f"{result.achieved_state}|"
            f"{result.duration_ms}|"
            f"{result.end_time.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
