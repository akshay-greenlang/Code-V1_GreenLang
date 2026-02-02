"""
Safety Action Gate for GL-001 ThermalCommand

This module implements the pre-write validation gate that is the final
safety checkpoint before any actuation command is sent to the PLC/DCS.

Key Functions:
- Pre-write validation
- Bounds checking
- Velocity limiting
- Interlock status verification
- Alarm state checking

CRITICAL: Every write command MUST pass through this gate.
No exceptions. No bypasses from GL-001.

Example:
    >>> from action_gate import SafetyActionGate
    >>> gate = SafetyActionGate(boundary_engine, sis_validator)
    >>> result = gate.evaluate(write_request)
    >>> if result.is_allowed:
    ...     execute_write(result.final_value)
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import threading

from pydantic import BaseModel, Field

from .safety_schemas import (
    BoundaryViolation,
    PolicyType,
    ViolationType,
    ViolationSeverity,
    GateDecision,
    TagWriteRequest,
    ActionGateResult,
    SafetyState,
    SISState,
    InterlockState,
    AlarmState,
    SafetyAuditRecord,
)
from .boundary_engine import SafetyBoundaryEngine
from .sis_validator import SISIndependenceValidator

logger = logging.getLogger(__name__)


class VelocityLimit(BaseModel):
    """Velocity limit specification for a tag type."""

    tag_pattern: str = Field(..., description="Glob pattern for matching tags")
    max_velocity: float = Field(..., description="Maximum change per second")
    engineering_units: str = Field(default="units/s", description="Units for velocity")


class GateConfiguration(BaseModel):
    """Configuration for the Safety Action Gate."""

    # Enable/disable checks
    enable_bounds_check: bool = Field(default=True, description="Enable bounds checking")
    enable_velocity_check: bool = Field(default=True, description="Enable velocity limiting")
    enable_interlock_check: bool = Field(default=True, description="Enable interlock checking")
    enable_alarm_check: bool = Field(default=True, description="Enable alarm checking")
    enable_sis_check: bool = Field(default=True, description="Enable SIS independence check")

    # Behavior
    allow_clamping: bool = Field(default=True, description="Allow clamping to bounds")
    block_on_critical_alarm: bool = Field(default=True, description="Block on critical alarms")
    block_on_any_alarm: bool = Field(default=False, description="Block on any alarm")

    # Timing
    max_evaluation_time_ms: float = Field(
        default=50.0,
        description="Maximum time for gate evaluation"
    )
    rate_limit_requests_per_second: int = Field(
        default=100,
        description="Maximum requests per second"
    )


# Default velocity limits by tag type
DEFAULT_VELOCITY_LIMITS: List[VelocityLimit] = [
    VelocityLimit(tag_pattern="TIC-*", max_velocity=2.0, engineering_units="degC/s"),
    VelocityLimit(tag_pattern="PIC-*", max_velocity=10.0, engineering_units="kPa/s"),
    VelocityLimit(tag_pattern="FIC-*", max_velocity=5.0, engineering_units="m3/h/s"),
    VelocityLimit(tag_pattern="LIC-*", max_velocity=1.0, engineering_units="%/s"),
    VelocityLimit(tag_pattern="XV-*", max_velocity=10.0, engineering_units="%/s"),
]


class SafetyActionGate:
    """
    Safety Action Gate - Final checkpoint before actuation.

    This gate performs comprehensive validation of every write request
    before allowing it to proceed to the PLC/DCS. It integrates:
    - Boundary Engine (bounds and rate checks)
    - SIS Validator (independence verification)
    - Interlock status
    - Alarm states
    - Velocity limits

    CRITICAL: This gate is MANDATORY. Every write MUST pass through.

    Attributes:
        boundary_engine: Safety Boundary Engine
        sis_validator: SIS Independence Validator
        config: Gate configuration
        velocity_limits: Velocity limits by tag type

    Example:
        >>> gate = SafetyActionGate(boundary_engine, sis_validator)
        >>> request = TagWriteRequest(tag_id="TIC-101", value=150.0)
        >>> result = gate.evaluate(request)
        >>> if result.decision == GateDecision.ALLOW:
        ...     write_to_plc(result.final_value)
        ... else:
        ...     handle_blocked_write(result.violations)
    """

    def __init__(
        self,
        boundary_engine: SafetyBoundaryEngine,
        sis_validator: SISIndependenceValidator,
        config: Optional[GateConfiguration] = None,
        velocity_limits: Optional[List[VelocityLimit]] = None,
        violation_handler: Optional[Callable[[BoundaryViolation], None]] = None,
    ) -> None:
        """
        Initialize Safety Action Gate.

        Args:
            boundary_engine: Safety Boundary Engine
            sis_validator: SIS Independence Validator
            config: Gate configuration
            velocity_limits: Custom velocity limits
            violation_handler: Callback for violations
        """
        self._boundary_engine = boundary_engine
        self._sis_validator = sis_validator
        self._config = config or GateConfiguration()
        self._velocity_limits = velocity_limits or DEFAULT_VELOCITY_LIMITS
        self._violation_handler = violation_handler

        # Current values for velocity calculation
        self._current_values: Dict[str, float] = {}
        self._last_write_times: Dict[str, datetime] = {}
        self._values_lock = threading.RLock()

        # Interlock and alarm states
        self._interlock_states: Dict[str, InterlockState] = {}
        self._alarm_states: Dict[str, AlarmState] = {}
        self._states_lock = threading.RLock()

        # Request rate limiting
        self._request_times: List[datetime] = []
        self._rate_limit_lock = threading.Lock()

        # Audit trail
        self._audit_records: List[SafetyAuditRecord] = []
        self._audit_lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_evaluations": 0,
            "allowed": 0,
            "blocked": 0,
            "clamped": 0,
            "rate_limited": 0,
            "sis_blocked": 0,
            "interlock_blocked": 0,
            "alarm_blocked": 0,
            "velocity_blocked": 0,
            "bounds_blocked": 0,
        }
        self._stats_lock = threading.Lock()

        logger.info("SafetyActionGate initialized")

    def evaluate(
        self,
        request: TagWriteRequest,
        override_clamping: Optional[bool] = None,
    ) -> ActionGateResult:
        """
        Evaluate a write request through all safety checks.

        This is the main entry point. EVERY write request MUST go through here.

        Args:
            request: Write request to evaluate
            override_clamping: Override default clamping behavior

        Returns:
            ActionGateResult with decision and details
        """
        start_time = datetime.utcnow()
        violations: List[BoundaryViolation] = []
        checks_passed: List[str] = []
        checks_failed: List[str] = []

        with self._stats_lock:
            self._stats["total_evaluations"] += 1

        logger.debug(
            f"Gate evaluation: tag={request.tag_id}, value={request.value}"
        )

        # Step 0: Rate limiting check
        if not self._check_rate_limit():
            with self._stats_lock:
                self._stats["rate_limited"] += 1
            return ActionGateResult(
                decision=GateDecision.BLOCK,
                original_request=request,
                violations=[BoundaryViolation(
                    policy_id="RATE_LIMIT_GATE",
                    policy_type=PolicyType.RATE_LIMIT,
                    tag_id=request.tag_id,
                    requested_value=request.value,
                    violation_type=ViolationType.RATE_EXCEEDED,
                    severity=ViolationSeverity.WARNING,
                    message="Gate rate limit exceeded - too many requests",
                )],
                checks_passed=[],
                checks_failed=["rate_limit_check"],
                gate_timestamp=start_time,
            )

        # Step 1: SIS Independence Check (CRITICAL - FIRST)
        if self._config.enable_sis_check:
            sis_result = self._sis_validator.validate_tag_access(request.tag_id, "write")
            if not sis_result.is_valid:
                violation = BoundaryViolation(
                    policy_id="SIS_INDEPENDENCE",
                    policy_type=PolicyType.WHITELIST,
                    tag_id=request.tag_id,
                    requested_value=request.value,
                    violation_type=ViolationType.SIS_VIOLATION,
                    severity=ViolationSeverity.EMERGENCY,
                    message=sis_result.message,
                )
                violations.append(violation)
                checks_failed.append("sis_independence")

                with self._stats_lock:
                    self._stats["sis_blocked"] += 1

                # SIS violations are ALWAYS blocking - return immediately
                return self._create_result(
                    GateDecision.BLOCK,
                    request,
                    None,
                    violations,
                    checks_passed,
                    checks_failed,
                    start_time,
                )
            else:
                checks_passed.append("sis_independence")

        # Step 2: Interlock Status Check
        if self._config.enable_interlock_check:
            interlock_violations = self._check_interlocks(request)
            if interlock_violations:
                violations.extend(interlock_violations)
                checks_failed.append("interlock_status")
                with self._stats_lock:
                    self._stats["interlock_blocked"] += 1
            else:
                checks_passed.append("interlock_status")

        # Step 3: Alarm State Check
        if self._config.enable_alarm_check:
            alarm_violations = self._check_alarms(request)
            if alarm_violations:
                violations.extend(alarm_violations)
                checks_failed.append("alarm_status")
                with self._stats_lock:
                    self._stats["alarm_blocked"] += 1
            else:
                checks_passed.append("alarm_status")

        # Step 4: Velocity Limit Check
        if self._config.enable_velocity_check:
            velocity_violations = self._check_velocity(request)
            if velocity_violations:
                violations.extend(velocity_violations)
                checks_failed.append("velocity_limit")
                with self._stats_lock:
                    self._stats["velocity_blocked"] += 1
            else:
                checks_passed.append("velocity_limit")

        # Step 5: Boundary Engine Check (includes bounds and rate limits)
        if self._config.enable_bounds_check:
            allow_clamp = override_clamping if override_clamping is not None else self._config.allow_clamping
            boundary_result = self._boundary_engine.validate_write_request(
                request, allow_clamping=allow_clamp
            )

            if boundary_result.violations:
                violations.extend(boundary_result.violations)
                checks_failed.extend(boundary_result.checks_failed)
                with self._stats_lock:
                    self._stats["bounds_blocked"] += 1
            else:
                checks_passed.extend(boundary_result.checks_passed)

            # Use boundary engine's decision if no other violations
            if not [v for v in violations if v not in boundary_result.violations]:
                decision = boundary_result.decision
                final_value = boundary_result.final_value
            else:
                # Other violations present - determine decision
                decision, final_value = self._determine_decision(request, violations)
        else:
            checks_passed.append("bounds_check_disabled")
            decision, final_value = self._determine_decision(request, violations)

        # Update statistics
        self._update_stats(decision)

        # Record current value if allowed
        if decision in (GateDecision.ALLOW, GateDecision.CLAMP) and final_value is not None:
            self._record_write(request.tag_id, final_value)

        # Create and return result
        result = self._create_result(
            decision, request, final_value, violations,
            checks_passed, checks_failed, start_time
        )

        # Handle violations if any
        if violations and self._violation_handler:
            for violation in violations:
                try:
                    self._violation_handler(violation)
                except Exception as e:
                    logger.error(f"Violation handler failed: {e}")

        return result

    def _check_rate_limit(self) -> bool:
        """
        Check if gate is within rate limits.

        Returns:
            True if within limits
        """
        with self._rate_limit_lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=1)

            # Clean old requests
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Check limit
            if len(self._request_times) >= self._config.rate_limit_requests_per_second:
                return False

            self._request_times.append(now)
            return True

    def _check_interlocks(
        self,
        request: TagWriteRequest
    ) -> List[BoundaryViolation]:
        """
        Check interlock states.

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []

        with self._states_lock:
            for interlock_id, state in self._interlock_states.items():
                if not state.is_permissive:
                    violations.append(BoundaryViolation(
                        policy_id=f"INTERLOCK_{interlock_id}",
                        policy_type=PolicyType.INTERLOCK,
                        tag_id=request.tag_id,
                        requested_value=request.value,
                        violation_type=ViolationType.INTERLOCK_ACTIVE,
                        severity=ViolationSeverity.EMERGENCY,
                        message=(
                            f"Interlock {interlock_id} is NOT PERMISSIVE: "
                            f"{state.cause or 'Unknown cause'}"
                        ),
                        context={
                            "interlock_id": interlock_id,
                            "cause": state.cause,
                        },
                    ))

        return violations

    def _check_alarms(
        self,
        request: TagWriteRequest
    ) -> List[BoundaryViolation]:
        """
        Check alarm states.

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []

        with self._states_lock:
            for alarm_id, state in self._alarm_states.items():
                if not state.is_active:
                    continue

                # Determine if we should block
                should_block = False
                if self._config.block_on_any_alarm:
                    should_block = True
                elif self._config.block_on_critical_alarm:
                    should_block = state.priority in ("CRITICAL", "EMERGENCY")

                # Check if alarm is related to the tag
                is_related = (
                    state.tag_id == request.tag_id or
                    state.tag_id.startswith(request.tag_id[:3])
                )

                if should_block and is_related:
                    violations.append(BoundaryViolation(
                        policy_id=f"ALARM_{alarm_id}",
                        policy_type=PolicyType.CONDITIONAL,
                        tag_id=request.tag_id,
                        requested_value=request.value,
                        violation_type=ViolationType.ALARM_ACTIVE,
                        severity=(
                            ViolationSeverity.EMERGENCY
                            if state.priority == "EMERGENCY"
                            else ViolationSeverity.CRITICAL
                        ),
                        message=(
                            f"Related alarm {alarm_id} is ACTIVE "
                            f"(priority: {state.priority})"
                        ),
                        context={
                            "alarm_id": alarm_id,
                            "alarm_tag": state.tag_id,
                            "priority": state.priority,
                            "acknowledged": state.acknowledged,
                        },
                    ))

        return violations

    def _check_velocity(
        self,
        request: TagWriteRequest
    ) -> List[BoundaryViolation]:
        """
        Check velocity limits.

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []

        # Find applicable velocity limit
        velocity_limit = None
        for limit in self._velocity_limits:
            from fnmatch import fnmatch
            if fnmatch(request.tag_id, limit.tag_pattern):
                velocity_limit = limit
                break

        if not velocity_limit:
            return violations

        # Get current value and time
        with self._values_lock:
            current_value = self._current_values.get(request.tag_id)
            last_time = self._last_write_times.get(request.tag_id)

        if current_value is None or last_time is None:
            # First write - no velocity check
            return violations

        # Calculate velocity
        elapsed = (datetime.utcnow() - last_time).total_seconds()
        if elapsed <= 0:
            elapsed = 0.001  # Minimum 1ms

        velocity = abs(request.value - current_value) / elapsed

        if velocity > velocity_limit.max_velocity:
            violations.append(BoundaryViolation(
                policy_id=f"VELOCITY_{velocity_limit.tag_pattern}",
                policy_type=PolicyType.RATE_LIMIT,
                tag_id=request.tag_id,
                requested_value=request.value,
                current_value=current_value,
                violation_type=ViolationType.RATE_EXCEEDED,
                severity=ViolationSeverity.CRITICAL,
                message=(
                    f"Velocity {velocity:.2f} {velocity_limit.engineering_units} "
                    f"exceeds limit {velocity_limit.max_velocity} "
                    f"{velocity_limit.engineering_units}"
                ),
                context={
                    "actual_velocity": velocity,
                    "max_velocity": velocity_limit.max_velocity,
                    "elapsed_seconds": elapsed,
                    "engineering_units": velocity_limit.engineering_units,
                },
            ))

        return violations

    def _determine_decision(
        self,
        request: TagWriteRequest,
        violations: List[BoundaryViolation],
    ) -> Tuple[GateDecision, Optional[float]]:
        """
        Determine final decision based on violations.

        Args:
            request: Original request
            violations: List of violations

        Returns:
            Tuple of (decision, final_value)
        """
        if not violations:
            return GateDecision.ALLOW, request.value

        # Check for blocking violations
        for v in violations:
            if v.severity == ViolationSeverity.EMERGENCY:
                return GateDecision.BLOCK, None

            if v.violation_type in (
                ViolationType.SIS_VIOLATION,
                ViolationType.INTERLOCK_ACTIVE,
                ViolationType.UNAUTHORIZED_TAG,
            ):
                return GateDecision.BLOCK, None

        # For other violations, block by default
        return GateDecision.BLOCK, None

    def _record_write(self, tag_id: str, value: float) -> None:
        """
        Record a successful write.

        Args:
            tag_id: Tag identifier
            value: Written value
        """
        with self._values_lock:
            self._current_values[tag_id] = value
            self._last_write_times[tag_id] = datetime.utcnow()

    def _update_stats(self, decision: GateDecision) -> None:
        """Update statistics."""
        with self._stats_lock:
            if decision == GateDecision.ALLOW:
                self._stats["allowed"] += 1
            elif decision == GateDecision.BLOCK:
                self._stats["blocked"] += 1
            elif decision == GateDecision.CLAMP:
                self._stats["clamped"] += 1

    def _create_result(
        self,
        decision: GateDecision,
        request: TagWriteRequest,
        final_value: Optional[float],
        violations: List[BoundaryViolation],
        checks_passed: List[str],
        checks_failed: List[str],
        start_time: datetime,
    ) -> ActionGateResult:
        """
        Create ActionGateResult.

        Args:
            decision: Gate decision
            request: Original request
            final_value: Final value to write
            violations: List of violations
            checks_passed: Passed checks
            checks_failed: Failed checks
            start_time: Evaluation start time

        Returns:
            ActionGateResult
        """
        evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = ActionGateResult(
            decision=decision,
            original_request=request,
            final_value=final_value,
            violations=violations,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            gate_timestamp=start_time,
            evaluation_time_ms=evaluation_time,
        )

        # Compute provenance
        result = ActionGateResult(
            **{**result.dict(), "provenance_hash": result.compute_provenance()}
        )

        # Check evaluation time
        if evaluation_time > self._config.max_evaluation_time_ms:
            logger.warning(
                f"Gate evaluation exceeded max time: "
                f"{evaluation_time:.2f}ms > {self._config.max_evaluation_time_ms}ms"
            )

        return result

    def update_interlock_state(self, interlock_state: InterlockState) -> None:
        """
        Update interlock state.

        Args:
            interlock_state: New interlock state
        """
        with self._states_lock:
            self._interlock_states[interlock_state.interlock_id] = interlock_state
            logger.debug(f"Updated interlock state: {interlock_state.interlock_id}")

        # Also update boundary engine
        self._boundary_engine.update_interlock_state(interlock_state)

    def update_alarm_state(self, alarm_state: AlarmState) -> None:
        """
        Update alarm state.

        Args:
            alarm_state: New alarm state
        """
        with self._states_lock:
            self._alarm_states[alarm_state.alarm_id] = alarm_state
            logger.debug(f"Updated alarm state: {alarm_state.alarm_id}")

        # Also update boundary engine
        self._boundary_engine.update_alarm_state(alarm_state)

    def update_current_value(self, tag_id: str, value: float) -> None:
        """
        Update current tag value for velocity calculation.

        Args:
            tag_id: Tag identifier
            value: Current value
        """
        with self._values_lock:
            self._current_values[tag_id] = value

        # Also update boundary engine
        self._boundary_engine.update_tag_value(tag_id, value)

    def update_current_values(self, values: Dict[str, float]) -> None:
        """
        Bulk update current tag values.

        Args:
            values: Dict of tag_id -> value
        """
        with self._values_lock:
            self._current_values.update(values)
            # Don't update timestamps - these are reads, not writes

        # Also update boundary engine
        self._boundary_engine.update_tag_values(values)

    def get_statistics(self) -> Dict[str, int]:
        """
        Get gate statistics.

        Returns:
            Dict of statistics
        """
        with self._stats_lock:
            return dict(self._stats)

    def get_config(self) -> GateConfiguration:
        """
        Get current configuration.

        Returns:
            Gate configuration
        """
        return self._config

    def update_config(self, config: GateConfiguration) -> None:
        """
        Update gate configuration.

        Args:
            config: New configuration

        Note:
            Cannot disable critical safety checks.
        """
        if not config.enable_sis_check:
            logger.warning("Cannot disable SIS check - ignoring")
            config = GateConfiguration(
                **{**config.dict(), "enable_sis_check": True}
            )

        self._config = config
        logger.info("Gate configuration updated")

    def evaluate_batch(
        self,
        requests: List[TagWriteRequest],
    ) -> List[ActionGateResult]:
        """
        Evaluate a batch of write requests.

        Args:
            requests: List of write requests

        Returns:
            List of results

        Note:
            Batch evaluation respects rate limits and may block requests.
        """
        results: List[ActionGateResult] = []

        for request in requests:
            result = self.evaluate(request)
            results.append(result)

            # Stop batch if emergency violation
            if any(v.severity == ViolationSeverity.EMERGENCY for v in result.violations):
                logger.critical("Batch evaluation stopped due to emergency violation")
                break

        return results

    def is_tag_writable(self, tag_id: str) -> bool:
        """
        Quick check if a tag is potentially writable.

        Args:
            tag_id: Tag identifier

        Returns:
            True if tag passes basic checks

        Note:
            This is a fast preliminary check. Full evaluation still required.
        """
        # Check SIS
        sis_result = self._sis_validator.validate_tag_access(tag_id, "write")
        if not sis_result.is_valid:
            return False

        # Check whitelist via boundary engine
        limits = self._boundary_engine._policy_manager.is_tag_allowed(tag_id)
        return limits

    def get_velocity_limits(self) -> List[VelocityLimit]:
        """
        Get configured velocity limits.

        Returns:
            List of velocity limits
        """
        return list(self._velocity_limits)

    def add_velocity_limit(self, limit: VelocityLimit) -> None:
        """
        Add a velocity limit.

        Args:
            limit: Velocity limit to add
        """
        self._velocity_limits.append(limit)
        logger.info(f"Added velocity limit for pattern: {limit.tag_pattern}")

    def clear_interlock_states(self) -> None:
        """Clear all interlock states."""
        with self._states_lock:
            self._interlock_states.clear()
        logger.info("Cleared all interlock states")

    def clear_alarm_states(self) -> None:
        """Clear all alarm states."""
        with self._states_lock:
            self._alarm_states.clear()
        logger.info("Cleared all alarm states")


class ActionGateFactory:
    """
    Factory for creating Safety Action Gate instances.

    Provides standardized gate creation with proper component wiring.

    Example:
        >>> factory = ActionGateFactory()
        >>> gate = factory.create_gate()
    """

    @staticmethod
    def create_gate(
        config: Optional[GateConfiguration] = None,
        violation_handler: Optional[Callable[[BoundaryViolation], None]] = None,
    ) -> SafetyActionGate:
        """
        Create a fully configured Safety Action Gate.

        Args:
            config: Optional configuration
            violation_handler: Optional violation handler

        Returns:
            Configured SafetyActionGate
        """
        # Create boundary engine
        boundary_engine = SafetyBoundaryEngine(
            violation_callback=violation_handler
        )

        # Create SIS validator
        sis_validator = SISIndependenceValidator(
            violation_callback=violation_handler
        )

        # Create gate
        gate = SafetyActionGate(
            boundary_engine=boundary_engine,
            sis_validator=sis_validator,
            config=config,
            violation_handler=violation_handler,
        )

        return gate

    @staticmethod
    def create_test_gate() -> SafetyActionGate:
        """
        Create a gate configured for testing.

        Returns:
            Test-configured SafetyActionGate
        """
        config = GateConfiguration(
            rate_limit_requests_per_second=1000,  # Higher for testing
            max_evaluation_time_ms=100.0,
        )

        return ActionGateFactory.create_gate(config=config)
