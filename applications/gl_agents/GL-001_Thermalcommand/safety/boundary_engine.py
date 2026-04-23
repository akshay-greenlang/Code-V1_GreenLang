"""
Safety Boundary Engine for GL-001 ThermalCommand

This module implements the core Safety Boundary Engine responsible for:
- Safety boundary policy definition and management
- Pre-optimization constraint enforcement
- Pre-actuation runtime gate
- Boundary violation detection and blocking
- Immutable audit record creation on violation

The engine follows zero-hallucination principles with deterministic
enforcement of all safety boundaries.

Example:
    >>> from boundary_engine import SafetyBoundaryEngine
    >>> engine = SafetyBoundaryEngine()
    >>> result = engine.validate_write_request(request)
    >>> if result.is_blocked:
    ...     handle_violation(result.violations)
"""

from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import threading
from collections import defaultdict

from .safety_schemas import (
    BoundaryPolicy,
    BoundaryViolation,
    PolicyType,
    ViolationType,
    ViolationSeverity,
    SafetyLevel,
    GateDecision,
    TagWriteRequest,
    ActionGateResult,
    SafetyState,
    SafetyAuditRecord,
    ConditionOperator,
    SISState,
    InterlockState,
    AlarmState,
)
from .boundary_policies import ThermalPolicyManager, get_policy_manager

logger = logging.getLogger(__name__)


class SafetyBoundaryEngine:
    """
    Core Safety Boundary Engine for GL-001 ThermalCommand.

    This engine is the primary enforcement point for all safety boundaries.
    It evaluates all write requests against defined policies before
    allowing actuation.

    Key Features:
    - Pre-optimization constraint enforcement
    - Pre-actuation runtime gate
    - Boundary violation detection and blocking
    - Rate limiting with sliding windows
    - Condition-based policy evaluation
    - Time-based restrictions
    - Immutable audit trail

    Attributes:
        policy_manager: Manager for boundary policies
        violation_handler: Handler for violations (injected)
        sis_validator: SIS independence validator (injected)
        audit_records: Immutable audit record chain

    Example:
        >>> engine = SafetyBoundaryEngine()
        >>> request = TagWriteRequest(tag_id="TIC-101", value=150.0)
        >>> result = engine.validate_write_request(request)
        >>> if result.decision == GateDecision.ALLOW:
        ...     write_to_plc(result.final_value)
    """

    def __init__(
        self,
        policy_manager: Optional[ThermalPolicyManager] = None,
        violation_callback: Optional[Callable[[BoundaryViolation], None]] = None,
        tag_value_provider: Optional[Callable[[str], Optional[float]]] = None,
    ) -> None:
        """
        Initialize the Safety Boundary Engine.

        Args:
            policy_manager: Policy manager (defaults to singleton)
            violation_callback: Callback for violation notifications
            tag_value_provider: Function to get current tag values
        """
        self._policy_manager = policy_manager or get_policy_manager()
        self._violation_callback = violation_callback
        self._tag_value_provider = tag_value_provider or self._default_tag_provider

        # Rate limiting state
        self._write_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._last_write_time: Dict[str, datetime] = {}
        self._last_write_value: Dict[str, float] = {}

        # Audit trail
        self._audit_records: List[SafetyAuditRecord] = []
        self._audit_lock = threading.Lock()

        # Current tag values cache
        self._tag_values: Dict[str, float] = {}
        self._tag_values_lock = threading.Lock()

        # System state
        self._sis_states: Dict[str, SISState] = {}
        self._interlock_states: Dict[str, InterlockState] = {}
        self._alarm_states: Dict[str, AlarmState] = {}

        # Statistics
        self._stats = {
            "total_requests": 0,
            "allowed": 0,
            "blocked": 0,
            "clamped": 0,
            "violations": 0,
        }
        self._stats_lock = threading.Lock()

        logger.info("SafetyBoundaryEngine initialized")

    def _default_tag_provider(self, tag_id: str) -> Optional[float]:
        """Default tag value provider using cache."""
        with self._tag_values_lock:
            return self._tag_values.get(tag_id)

    def update_tag_value(self, tag_id: str, value: float) -> None:
        """
        Update cached tag value.

        Args:
            tag_id: Tag identifier
            value: Current value
        """
        with self._tag_values_lock:
            self._tag_values[tag_id] = value

    def update_tag_values(self, values: Dict[str, float]) -> None:
        """
        Bulk update cached tag values.

        Args:
            values: Dict of tag_id -> value
        """
        with self._tag_values_lock:
            self._tag_values.update(values)

    def update_sis_state(self, sis_state: SISState) -> None:
        """
        Update SIS state.

        Args:
            sis_state: New SIS state
        """
        self._sis_states[sis_state.sis_id] = sis_state
        logger.debug(f"Updated SIS state: {sis_state.sis_id}")

    def update_interlock_state(self, interlock_state: InterlockState) -> None:
        """
        Update interlock state.

        Args:
            interlock_state: New interlock state
        """
        self._interlock_states[interlock_state.interlock_id] = interlock_state
        logger.debug(f"Updated interlock state: {interlock_state.interlock_id}")

    def update_alarm_state(self, alarm_state: AlarmState) -> None:
        """
        Update alarm state.

        Args:
            alarm_state: New alarm state
        """
        self._alarm_states[alarm_state.alarm_id] = alarm_state
        logger.debug(f"Updated alarm state: {alarm_state.alarm_id}")

    def validate_write_request(
        self,
        request: TagWriteRequest,
        allow_clamping: bool = True,
    ) -> ActionGateResult:
        """
        Validate a write request against all safety boundaries.

        This is the main entry point for pre-actuation validation.
        All write requests MUST pass through this gate.

        Args:
            request: Write request to validate
            allow_clamping: Whether to allow clamping to boundaries

        Returns:
            ActionGateResult with decision and details
        """
        start_time = datetime.utcnow()
        violations: List[BoundaryViolation] = []
        checks_passed: List[str] = []
        checks_failed: List[str] = []

        # Increment request counter
        with self._stats_lock:
            self._stats["total_requests"] += 1

        logger.debug(
            f"Validating write request: tag={request.tag_id}, value={request.value}"
        )

        # Step 1: Check whitelist
        whitelist_result = self._check_whitelist(request)
        if whitelist_result:
            violations.append(whitelist_result)
            checks_failed.append("whitelist_check")
        else:
            checks_passed.append("whitelist_check")

        # Step 2: Check absolute limits
        limit_violations = self._check_absolute_limits(request)
        violations.extend(limit_violations)
        if limit_violations:
            checks_failed.append("absolute_limits")
        else:
            checks_passed.append("absolute_limits")

        # Step 3: Check rate limits
        rate_violations = self._check_rate_limits(request)
        violations.extend(rate_violations)
        if rate_violations:
            checks_failed.append("rate_limits")
        else:
            checks_passed.append("rate_limits")

        # Step 4: Check conditional policies
        condition_violations = self._check_conditional_policies(request)
        violations.extend(condition_violations)
        if condition_violations:
            checks_failed.append("conditional_policies")
        else:
            checks_passed.append("conditional_policies")

        # Step 5: Check time-based restrictions
        time_violations = self._check_time_restrictions(request)
        violations.extend(time_violations)
        if time_violations:
            checks_failed.append("time_restrictions")
        else:
            checks_passed.append("time_restrictions")

        # Step 6: Check interlocks
        interlock_violations = self._check_interlocks(request)
        violations.extend(interlock_violations)
        if interlock_violations:
            checks_failed.append("interlock_check")
        else:
            checks_passed.append("interlock_check")

        # Step 7: Check alarms
        alarm_violations = self._check_alarms(request)
        violations.extend(alarm_violations)
        if alarm_violations:
            checks_failed.append("alarm_check")
        else:
            checks_passed.append("alarm_check")

        # Determine decision
        decision, final_value = self._determine_decision(
            request, violations, allow_clamping
        )

        # Calculate evaluation time
        evaluation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Create result
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

        # Update provenance
        result = ActionGateResult(
            **{**result.dict(), "provenance_hash": result.compute_provenance()}
        )

        # Update statistics
        self._update_stats(decision, len(violations))

        # Handle violations if any
        if violations:
            self._handle_violations(violations, result)

        # Record write if allowed
        if decision in (GateDecision.ALLOW, GateDecision.CLAMP) and final_value is not None:
            self._record_write(request.tag_id, final_value)

        logger.info(
            f"Gate result: tag={request.tag_id}, decision={decision}, "
            f"violations={len(violations)}, time_ms={evaluation_time:.2f}"
        )

        return result

    def _check_whitelist(self, request: TagWriteRequest) -> Optional[BoundaryViolation]:
        """
        Check if tag is in allowed whitelist.

        Args:
            request: Write request

        Returns:
            Violation if tag not allowed, None otherwise
        """
        if not self._policy_manager.is_tag_allowed(request.tag_id):
            # Check if it's blacklisted (more severe)
            if self._policy_manager.is_tag_blacklisted(request.tag_id):
                return BoundaryViolation(
                    policy_id="BLACKLIST",
                    policy_type=PolicyType.WHITELIST,
                    tag_id=request.tag_id,
                    requested_value=request.value,
                    violation_type=ViolationType.UNAUTHORIZED_TAG,
                    severity=ViolationSeverity.EMERGENCY,
                    message=f"Tag {request.tag_id} is BLACKLISTED - GL-001 MUST NEVER write to this tag",
                    context={"blacklisted": True},
                )

            return BoundaryViolation(
                policy_id="WHITELIST_001",
                policy_type=PolicyType.WHITELIST,
                tag_id=request.tag_id,
                requested_value=request.value,
                violation_type=ViolationType.UNAUTHORIZED_TAG,
                severity=ViolationSeverity.CRITICAL,
                message=f"Tag {request.tag_id} is not in allowed whitelist",
            )

        return None

    def _check_absolute_limits(
        self,
        request: TagWriteRequest
    ) -> List[BoundaryViolation]:
        """
        Check absolute min/max limits.

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []
        policies = self._policy_manager.get_policies_for_tag(request.tag_id)

        for policy in policies:
            if policy.policy_type != PolicyType.ABSOLUTE_LIMIT:
                continue

            # Check minimum
            if policy.min_value is not None and request.value < policy.min_value:
                violations.append(BoundaryViolation(
                    policy_id=policy.policy_id,
                    policy_type=PolicyType.ABSOLUTE_LIMIT,
                    tag_id=request.tag_id,
                    requested_value=request.value,
                    boundary_value=policy.min_value,
                    violation_type=ViolationType.UNDER_MIN,
                    severity=policy.severity,
                    message=(
                        f"Value {request.value} is below minimum {policy.min_value} "
                        f"({policy.engineering_units or 'units'})"
                    ),
                    context={
                        "policy_description": policy.description,
                        "engineering_units": policy.engineering_units,
                    },
                ))

            # Check maximum
            if policy.max_value is not None and request.value > policy.max_value:
                violations.append(BoundaryViolation(
                    policy_id=policy.policy_id,
                    policy_type=PolicyType.ABSOLUTE_LIMIT,
                    tag_id=request.tag_id,
                    requested_value=request.value,
                    boundary_value=policy.max_value,
                    violation_type=ViolationType.OVER_MAX,
                    severity=policy.severity,
                    message=(
                        f"Value {request.value} exceeds maximum {policy.max_value} "
                        f"({policy.engineering_units or 'units'})"
                    ),
                    context={
                        "policy_description": policy.description,
                        "engineering_units": policy.engineering_units,
                    },
                ))

        return violations

    def _check_rate_limits(
        self,
        request: TagWriteRequest
    ) -> List[BoundaryViolation]:
        """
        Check rate limits for tag.

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []
        rate_spec = self._policy_manager.get_rate_limits_for_tag(request.tag_id)

        if not rate_spec:
            return violations

        now = datetime.utcnow()
        tag_id = request.tag_id

        # Get last write info
        last_time = self._last_write_time.get(tag_id)
        last_value = self._last_write_value.get(tag_id)

        # Check cooldown
        if last_time and rate_spec.cooldown_seconds:
            elapsed = (now - last_time).total_seconds()
            if elapsed < rate_spec.cooldown_seconds:
                violations.append(BoundaryViolation(
                    policy_id="RATE_COOLDOWN",
                    policy_type=PolicyType.RATE_LIMIT,
                    tag_id=tag_id,
                    requested_value=request.value,
                    violation_type=ViolationType.RATE_EXCEEDED,
                    severity=ViolationSeverity.CRITICAL,
                    message=(
                        f"Cooldown period not elapsed. "
                        f"Elapsed: {elapsed:.1f}s, Required: {rate_spec.cooldown_seconds}s"
                    ),
                    context={
                        "elapsed_seconds": elapsed,
                        "required_seconds": rate_spec.cooldown_seconds,
                    },
                ))

        # Check rate of change per second
        if last_time and last_value is not None and rate_spec.max_change_per_second:
            elapsed = (now - last_time).total_seconds()
            if elapsed > 0:
                rate = abs(request.value - last_value) / elapsed
                if rate > rate_spec.max_change_per_second:
                    violations.append(BoundaryViolation(
                        policy_id="RATE_PER_SECOND",
                        policy_type=PolicyType.RATE_LIMIT,
                        tag_id=tag_id,
                        requested_value=request.value,
                        current_value=last_value,
                        violation_type=ViolationType.RATE_EXCEEDED,
                        severity=ViolationSeverity.CRITICAL,
                        message=(
                            f"Rate of change {rate:.2f}/s exceeds limit "
                            f"{rate_spec.max_change_per_second}/s"
                        ),
                        context={
                            "actual_rate": rate,
                            "max_rate": rate_spec.max_change_per_second,
                        },
                    ))

        # Check writes per minute
        if rate_spec.max_writes_per_minute:
            minute_ago = now - timedelta(minutes=1)
            recent_writes = [
                t for t, _ in self._write_history[tag_id]
                if t > minute_ago
            ]
            if len(recent_writes) >= rate_spec.max_writes_per_minute:
                violations.append(BoundaryViolation(
                    policy_id="RATE_WRITES_PER_MIN",
                    policy_type=PolicyType.RATE_LIMIT,
                    tag_id=tag_id,
                    requested_value=request.value,
                    violation_type=ViolationType.RATE_EXCEEDED,
                    severity=ViolationSeverity.CRITICAL,
                    message=(
                        f"Write count {len(recent_writes)}/min exceeds limit "
                        f"{rate_spec.max_writes_per_minute}/min"
                    ),
                    context={
                        "actual_writes": len(recent_writes),
                        "max_writes": rate_spec.max_writes_per_minute,
                    },
                ))

        # Check change per minute
        if rate_spec.max_change_per_minute and last_value is not None:
            minute_ago = now - timedelta(minutes=1)
            history = self._write_history.get(tag_id, [])
            if history:
                # Find value from ~1 minute ago
                old_values = [v for t, v in history if t <= minute_ago]
                if old_values:
                    old_value = old_values[-1]
                    change = abs(request.value - old_value)
                    if change > rate_spec.max_change_per_minute:
                        violations.append(BoundaryViolation(
                            policy_id="RATE_CHANGE_PER_MIN",
                            policy_type=PolicyType.RATE_LIMIT,
                            tag_id=tag_id,
                            requested_value=request.value,
                            violation_type=ViolationType.RATE_EXCEEDED,
                            severity=ViolationSeverity.CRITICAL,
                            message=(
                                f"Change {change:.2f}/min exceeds limit "
                                f"{rate_spec.max_change_per_minute}/min"
                            ),
                            context={
                                "actual_change": change,
                                "max_change": rate_spec.max_change_per_minute,
                            },
                        ))

        return violations

    def _check_conditional_policies(
        self,
        request: TagWriteRequest
    ) -> List[BoundaryViolation]:
        """
        Check conditional policies.

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []
        policies = self._policy_manager.get_policies_for_tag(request.tag_id)

        for policy in policies:
            if policy.policy_type not in (PolicyType.CONDITIONAL, PolicyType.INTERLOCK):
                continue

            if not policy.conditions:
                continue

            # Evaluate conditions
            condition_results = []
            for condition in policy.conditions:
                result = self._evaluate_condition(condition)
                condition_results.append(result)

            # Apply condition logic
            if policy.condition_logic == "AND":
                conditions_met = all(condition_results)
            else:  # OR
                conditions_met = any(condition_results)

            # If conditions are met, check if the policy applies limits
            if conditions_met:
                # Check if value violates conditional limits
                if policy.max_value is not None and request.value > policy.max_value:
                    violations.append(BoundaryViolation(
                        policy_id=policy.policy_id,
                        policy_type=policy.policy_type,
                        tag_id=request.tag_id,
                        requested_value=request.value,
                        boundary_value=policy.max_value,
                        violation_type=ViolationType.CONDITION_VIOLATED,
                        severity=policy.severity,
                        message=(
                            f"Conditional limit violated: {request.value} > "
                            f"{policy.max_value} when conditions met"
                        ),
                        context={
                            "policy_description": policy.description,
                            "conditions_met": True,
                        },
                    ))

                if policy.min_value is not None and request.value < policy.min_value:
                    violations.append(BoundaryViolation(
                        policy_id=policy.policy_id,
                        policy_type=policy.policy_type,
                        tag_id=request.tag_id,
                        requested_value=request.value,
                        boundary_value=policy.min_value,
                        violation_type=ViolationType.CONDITION_VIOLATED,
                        severity=policy.severity,
                        message=(
                            f"Conditional limit violated: {request.value} < "
                            f"{policy.min_value} when conditions met"
                        ),
                        context={
                            "policy_description": policy.description,
                            "conditions_met": True,
                        },
                    ))

        return violations

    def _evaluate_condition(self, condition: Any) -> bool:
        """
        Evaluate a single condition.

        Args:
            condition: Condition to evaluate

        Returns:
            True if condition is met
        """
        tag_value = self._tag_value_provider(condition.tag_id)

        if tag_value is None:
            logger.warning(f"Cannot evaluate condition: tag {condition.tag_id} not found")
            return False  # Fail safe

        operator = condition.operator
        target = condition.value

        if operator == ConditionOperator.EQUALS:
            return tag_value == target
        elif operator == ConditionOperator.NOT_EQUALS:
            return tag_value != target
        elif operator == ConditionOperator.GREATER_THAN:
            return tag_value > target
        elif operator == ConditionOperator.LESS_THAN:
            return tag_value < target
        elif operator == ConditionOperator.GREATER_EQUAL:
            return tag_value >= target
        elif operator == ConditionOperator.LESS_EQUAL:
            return tag_value <= target
        elif operator == ConditionOperator.IN_RANGE:
            return target <= tag_value <= condition.secondary_value
        elif operator == ConditionOperator.NOT_IN_RANGE:
            return not (target <= tag_value <= condition.secondary_value)

        return False

    def _check_time_restrictions(
        self,
        request: TagWriteRequest
    ) -> List[BoundaryViolation]:
        """
        Check time-based restrictions.

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []
        policies = self._policy_manager.get_policies_by_type(PolicyType.TIME_BASED)

        now = datetime.utcnow()
        current_time = now.time()
        current_day = now.weekday()

        for policy in policies:
            if not fnmatch(request.tag_id, policy.tag_pattern):
                continue

            if not policy.time_restrictions:
                continue

            for restriction in policy.time_restrictions:
                # Check if current day is in restriction days
                if current_day not in restriction.days_of_week:
                    continue

                # Check if current time is in restriction period
                in_restriction = False
                if restriction.start_time <= restriction.end_time:
                    # Normal case: e.g., 9 AM to 5 PM
                    in_restriction = restriction.start_time <= current_time <= restriction.end_time
                else:
                    # Overnight case: e.g., 10 PM to 6 AM
                    in_restriction = (
                        current_time >= restriction.start_time or
                        current_time <= restriction.end_time
                    )

                if in_restriction:
                    violations.append(BoundaryViolation(
                        policy_id=policy.policy_id,
                        policy_type=PolicyType.TIME_BASED,
                        tag_id=request.tag_id,
                        requested_value=request.value,
                        violation_type=ViolationType.TIME_RESTRICTED,
                        severity=policy.severity,
                        message=(
                            f"Write blocked during time restriction: "
                            f"{restriction.start_time} - {restriction.end_time}"
                        ),
                        context={
                            "restriction_start": str(restriction.start_time),
                            "restriction_end": str(restriction.end_time),
                            "current_time": str(current_time),
                        },
                    ))

        return violations

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

        for interlock_id, state in self._interlock_states.items():
            if not state.is_permissive:
                violations.append(BoundaryViolation(
                    policy_id=f"INTERLOCK_{interlock_id}",
                    policy_type=PolicyType.INTERLOCK,
                    tag_id=request.tag_id,
                    requested_value=request.value,
                    violation_type=ViolationType.INTERLOCK_ACTIVE,
                    severity=ViolationSeverity.EMERGENCY,
                    message=f"Interlock {interlock_id} is not permissive: {state.cause}",
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

        for alarm_id, state in self._alarm_states.items():
            # Only block on critical/emergency alarms
            if state.is_active and state.priority in ("CRITICAL", "EMERGENCY"):
                # Check if alarm is related to the tag being written
                if state.tag_id == request.tag_id or state.tag_id.startswith(request.tag_id[:3]):
                    violations.append(BoundaryViolation(
                        policy_id=f"ALARM_{alarm_id}",
                        policy_type=PolicyType.CONDITIONAL,
                        tag_id=request.tag_id,
                        requested_value=request.value,
                        violation_type=ViolationType.ALARM_ACTIVE,
                        severity=ViolationSeverity.CRITICAL,
                        message=f"Related alarm {alarm_id} is active (priority: {state.priority})",
                        context={
                            "alarm_id": alarm_id,
                            "alarm_tag": state.tag_id,
                            "priority": state.priority,
                        },
                    ))

        return violations

    def _determine_decision(
        self,
        request: TagWriteRequest,
        violations: List[BoundaryViolation],
        allow_clamping: bool,
    ) -> Tuple[GateDecision, Optional[float]]:
        """
        Determine final decision and value.

        Args:
            request: Original request
            violations: List of violations
            allow_clamping: Whether clamping is allowed

        Returns:
            Tuple of (decision, final_value)
        """
        if not violations:
            return GateDecision.ALLOW, request.value

        # Check for emergency/critical violations that must block
        for v in violations:
            if v.severity == ViolationSeverity.EMERGENCY:
                return GateDecision.BLOCK, None

            if v.violation_type in (
                ViolationType.UNAUTHORIZED_TAG,
                ViolationType.INTERLOCK_ACTIVE,
                ViolationType.SIS_VIOLATION,
                ViolationType.RATE_EXCEEDED,
                ViolationType.TIME_RESTRICTED,
            ):
                return GateDecision.BLOCK, None

        # For limit violations, try clamping if allowed
        if allow_clamping:
            limits = self._policy_manager.get_limits_for_tag(request.tag_id)
            clamped_value = request.value

            if limits["min"] is not None and clamped_value < limits["min"]:
                clamped_value = limits["min"]
            if limits["max"] is not None and clamped_value > limits["max"]:
                clamped_value = limits["max"]

            if clamped_value != request.value:
                logger.info(
                    f"Clamping value from {request.value} to {clamped_value} "
                    f"for tag {request.tag_id}"
                )
                return GateDecision.CLAMP, clamped_value

        return GateDecision.BLOCK, None

    def _record_write(self, tag_id: str, value: float) -> None:
        """
        Record a successful write for rate limiting.

        Args:
            tag_id: Tag identifier
            value: Written value
        """
        now = datetime.utcnow()
        self._last_write_time[tag_id] = now
        self._last_write_value[tag_id] = value
        self._write_history[tag_id].append((now, value))

        # Clean up old history (keep last 10 minutes)
        cutoff = now - timedelta(minutes=10)
        self._write_history[tag_id] = [
            (t, v) for t, v in self._write_history[tag_id]
            if t > cutoff
        ]

    def _update_stats(self, decision: GateDecision, violation_count: int) -> None:
        """Update statistics."""
        with self._stats_lock:
            if decision == GateDecision.ALLOW:
                self._stats["allowed"] += 1
            elif decision == GateDecision.BLOCK:
                self._stats["blocked"] += 1
            elif decision == GateDecision.CLAMP:
                self._stats["clamped"] += 1
            self._stats["violations"] += violation_count

    def _handle_violations(
        self,
        violations: List[BoundaryViolation],
        result: ActionGateResult
    ) -> None:
        """
        Handle violations by creating audit records and notifying.

        Args:
            violations: List of violations
            result: Gate result
        """
        for violation in violations:
            # Create audit record
            self._create_audit_record(violation, result)

            # Call violation callback if provided
            if self._violation_callback:
                try:
                    self._violation_callback(violation)
                except Exception as e:
                    logger.error(f"Violation callback failed: {e}")

    def _create_audit_record(
        self,
        violation: BoundaryViolation,
        result: ActionGateResult
    ) -> SafetyAuditRecord:
        """
        Create immutable audit record.

        Args:
            violation: The violation
            result: Gate result

        Returns:
            Created audit record
        """
        with self._audit_lock:
            # Get previous hash for chain
            previous_hash = ""
            if self._audit_records:
                previous_hash = self._audit_records[-1].provenance_hash

            record = SafetyAuditRecord(
                event_type="BOUNDARY_VIOLATION",
                violation=violation,
                gate_result=result,
                action_taken="BLOCKED" if result.is_blocked else "CLAMPED",
                operator_notified=violation.severity in (
                    ViolationSeverity.CRITICAL,
                    ViolationSeverity.EMERGENCY
                ),
                escalation_level=2 if violation.severity == ViolationSeverity.EMERGENCY else 1,
                previous_hash=previous_hash,
            )

            self._audit_records.append(record)
            logger.info(f"Created audit record: {record.record_id}")

            return record

    def get_audit_records(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SafetyAuditRecord]:
        """
        Get audit records.

        Args:
            since: Only records after this timestamp
            limit: Maximum records to return

        Returns:
            List of audit records
        """
        with self._audit_lock:
            records = self._audit_records
            if since:
                records = [r for r in records if r.timestamp > since]
            return records[-limit:]

    def verify_audit_chain(self) -> bool:
        """
        Verify integrity of audit record chain.

        Returns:
            True if chain is valid
        """
        with self._audit_lock:
            for i, record in enumerate(self._audit_records):
                if i == 0:
                    if record.previous_hash != "":
                        logger.error("First record should have empty previous_hash")
                        return False
                else:
                    expected_hash = self._audit_records[i - 1].provenance_hash
                    if record.previous_hash != expected_hash:
                        logger.error(f"Audit chain broken at record {i}")
                        return False

            return True

    def get_safety_state(self) -> SafetyState:
        """
        Get current safety state.

        Returns:
            Current SafetyState
        """
        with self._stats_lock:
            violations_count = self._stats["violations"]

        state = SafetyState(
            sis_states=list(self._sis_states.values()),
            interlock_states=list(self._interlock_states.values()),
            alarm_states=[a for a in self._alarm_states.values() if a.is_active],
            policies_enabled=len(self._policy_manager.get_enabled_policies()),
            policies_total=len(self._policy_manager.get_all_policies()),
            violations_last_hour=violations_count,  # Simplified for now
        )

        state = SafetyState(
            **{**state.dict(), "overall_safe": state.compute_overall_safe()}
        )

        return state

    def get_statistics(self) -> Dict[str, int]:
        """
        Get engine statistics.

        Returns:
            Dict of statistics
        """
        with self._stats_lock:
            return dict(self._stats)

    def enforce_pre_optimization_constraints(
        self,
        optimization_targets: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Enforce constraints before optimization.

        This is called BEFORE the optimization engine runs to ensure
        optimization targets are within safe bounds.

        Args:
            optimization_targets: Dict of tag_id -> target_value

        Returns:
            Dict of tag_id -> constrained_value
        """
        constrained = {}

        for tag_id, target in optimization_targets.items():
            # Get limits for this tag
            limits = self._policy_manager.get_limits_for_tag(tag_id)

            # Apply constraints
            value = target
            if limits["min"] is not None:
                value = max(value, limits["min"])
            if limits["max"] is not None:
                value = min(value, limits["max"])

            constrained[tag_id] = value

            if value != target:
                logger.info(
                    f"Pre-optimization constraint: {tag_id} "
                    f"{target} -> {value}"
                )

        return constrained

    def reset_rate_limit_state(self, tag_id: Optional[str] = None) -> None:
        """
        Reset rate limiting state.

        Args:
            tag_id: Specific tag to reset, or None for all

        Note:
            This should only be called during authorized resets.
        """
        if tag_id:
            self._write_history.pop(tag_id, None)
            self._last_write_time.pop(tag_id, None)
            self._last_write_value.pop(tag_id, None)
            logger.info(f"Reset rate limit state for tag: {tag_id}")
        else:
            self._write_history.clear()
            self._last_write_time.clear()
            self._last_write_value.clear()
            logger.info("Reset all rate limit state")
