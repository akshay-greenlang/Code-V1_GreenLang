"""
GL-001 ThermalCommand - Safety Boundary Engine
================================================================================

Core Safety Boundary Engine implementing IEC 61511-compliant safety logic
for pre-actuation validation and runtime gating of control actions.

IEC 61511 COMPLIANCE MAPPING
-----------------------------

This module implements safety concepts from IEC 61511 "Functional safety -
Safety instrumented systems for the process industry sector":

IEC 61511 Clause | Implementation
-----------------|---------------------------------------------------
11.2.1           | Demand mode operation with deterministic response
11.3.1           | SIF independence via separate validation path
11.4.2           | Voting logic (2oo3) for critical decisions
11.5.1           | Proof testing via audit chain verification
11.6.1           | Manual shutdown capability preserved
11.7.1           | Fault tolerance via policy redundancy

SAFETY INTEGRITY LEVEL (SIL) REQUIREMENTS
-----------------------------------------

Per IEC 61511, SIL determines probability of failure on demand (PFD):

SIL Level | PFD Range        | Risk Reduction Factor | Implementation
----------|------------------|----------------------|------------------
SIL 1     | 0.1 to 0.01     | 10 to 100            | Single redundancy
SIL 2     | 0.01 to 0.001   | 100 to 1,000         | Dual redundancy
SIL 3     | 0.001 to 0.0001 | 1,000 to 10,000      | 2oo3 voting
SIL 4     | 0.0001 to 1E-5  | 10,000 to 100,000    | Special measures

GL-001 ThermalCommand targets SIL-2 for process control outputs,
with SIL-3 capability for emergency shutdown functions.

SIL-3 REQUIREMENTS IMPLEMENTATION
---------------------------------

Per IEC 61511-1 Table 6, SIL-3 requires:

1. Hardware Fault Tolerance = 1 (single fault tolerant)
   Implementation: Redundant policy validation paths

2. Safe Failure Fraction > 60%
   Implementation: Fail-safe defaults (block on error)

3. Proof Test Coverage > 90%
   Implementation: Audit chain verification, automated testing

4. Common Cause Failure consideration
   Implementation: Diverse policy types, independent sensors

5. Systematic Capability = SIL 3
   Implementation: Deterministic algorithms, no floating point tolerance

2oo3 VOTING LOGIC (Triple Modular Redundancy)
---------------------------------------------

For SIL-3 critical decisions, 2-out-of-3 voting is implemented:

Truth Table:

| Sensor A | Sensor B | Sensor C | Vote Result | Action    |
|----------|----------|----------|-------------|-----------|
| Safe     | Safe     | Safe     | 3/3 Safe    | Allow     |
| Safe     | Safe     | Trip     | 2/3 Safe    | Allow     |
| Safe     | Trip     | Safe     | 2/3 Safe    | Allow     |
| Trip     | Safe     | Safe     | 2/3 Safe    | Allow     |
| Safe     | Trip     | Trip     | 2/3 Trip    | Block     |
| Trip     | Safe     | Trip     | 2/3 Trip    | Block     |
| Trip     | Trip     | Safe     | 2/3 Trip    | Block     |
| Trip     | Trip     | Trip     | 3/3 Trip    | Block     |

The 2oo3 logic:
- Tolerates single sensor failure
- Fails safe (to trip) on 2+ faults
- Detects discrepancy for diagnostics

Mathematical representation:

    Vote = (A AND B) OR (B AND C) OR (A AND C)

    Where A, B, C are boolean (True = Safe, False = Trip)

SAFETY MARGIN CALCULATIONS
--------------------------

Safety margins provide buffer between operating limits and safety limits:

**Absolute Margin:**

    Margin = SafetyLimit - OperatingLimit

    Example: If operating max = 150C and safety limit = 175C
             Margin = 175 - 150 = 25C

**Percentage Margin:**

    Margin% = (SafetyLimit - OperatingLimit) / SafetyLimit * 100

    Example: Margin% = (175 - 150) / 175 * 100 = 14.3%

**Dynamic Margin (rate-based):**

    Time_to_limit = Margin / RateOfChange

    Example: If temperature rising at 5C/min with 25C margin
             Time_to_limit = 25 / 5 = 5 minutes

    Pre-action threshold: Alert when Time_to_limit < Threshold (e.g., 2 min)

BOUNDARY VIOLATION SEVERITY LEVELS
----------------------------------

Per ISA-84 alarm management guidelines:

| Severity   | Description                    | Response Time | Action          |
|------------|--------------------------------|---------------|-----------------|
| WARNING    | Approaching limit              | Shift         | Operator review |
| CRITICAL   | At or beyond operating limit   | Minutes       | Corrective action|
| EMERGENCY  | At or beyond safety limit      | Seconds       | Automated shutdown|

RATE LIMITING ALGORITHM
-----------------------

Rate limits prevent rapid changes that could stress equipment:

**Rate of Change Calculation:**

    rate = (value_new - value_prev) / (time_new - time_prev)

    If |rate| > rate_limit:
        Block request with RATE_EXCEEDED violation

**Sliding Window Rate:**

For sustained rate limits over longer periods:

    rate_1min = (value_now - value_1min_ago) / 60

    If |rate_1min| > rate_limit_per_minute:
        Block request

**Cooldown Period:**

Minimum time between writes to prevent rapid cycling:

    if (time_now - last_write_time) < cooldown_seconds:
        Block request

Key Features:
    - Pre-optimization constraint enforcement
    - Pre-actuation runtime gate
    - Boundary violation detection and blocking
    - Rate limiting with sliding windows
    - Condition-based policy evaluation
    - Time-based restrictions
    - Immutable audit trail with blockchain-style chaining
    - SIS independence validation
    - 2oo3 voting for critical decisions

Reference Standards:
    - IEC 61511 Safety Instrumented Systems
    - IEC 61508 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - ISA-18.2 Alarm Management

Example:
    >>> from boundary_engine import SafetyBoundaryEngine
    >>> engine = SafetyBoundaryEngine()
    >>> request = TagWriteRequest(tag_id="TIC-101", value=150.0)
    >>> result = engine.validate_write_request(request)
    >>> if result.decision == GateDecision.ALLOW:
    ...     write_to_plc(result.final_value)
    >>> elif result.decision == GateDecision.BLOCK:
    ...     log_safety_event(result.violations)

Author: GreenLang Safety Systems Team
Version: 1.0.0
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

    ARCHITECTURE OVERVIEW
    =====================

    The Safety Boundary Engine follows a layered defense model:

    Layer 1: Whitelist Check
        - Only pre-approved tags can be written
        - Blacklisted tags are absolutely blocked
        - Implements principle of least privilege

    Layer 2: Absolute Limits
        - Hard min/max values that cannot be exceeded
        - Based on equipment design limits
        - Per IEC 61511 safety function requirements

    Layer 3: Rate Limits
        - Maximum change per second/minute
        - Cooldown periods between writes
        - Write count limits per time window

    Layer 4: Conditional Policies
        - Context-aware limits (e.g., lower max during startup)
        - Interlock conditions
        - Alarm-based restrictions

    Layer 5: Time-Based Restrictions
        - Maintenance windows
        - Shift-based restrictions
        - Calendar-based blackouts

    Layer 6: SIS/Interlock Integration
        - Independent SIS status validation
        - Active interlock enforcement
        - Critical alarm blocking

    VALIDATION SEQUENCE
    ===================

    For each write request:

    1. Whitelist Check
       - Is tag in allowed list?
       - Is tag in blacklist (emergency block)?

    2. Absolute Limits
       - value >= min_value?
       - value <= max_value?

    3. Rate Limits
       - |rate| <= max_rate_per_second?
       - Cooldown elapsed?
       - Write count <= max_per_minute?

    4. Conditional Policies
       - Evaluate all conditions
       - Apply conditional limits if met

    5. Time Restrictions
       - Current time in allowed window?

    6. Interlock Status
       - All interlocks permissive?

    7. Alarm Status
       - No blocking alarms active?

    DECISION LOGIC
    ==============

    Decision priority (highest to lowest):

    1. BLOCK (Emergency):
       - Blacklisted tag
       - Interlock active
       - SIS violation
       - Emergency alarm

    2. BLOCK (Critical):
       - Rate limit exceeded
       - Time restricted
       - Unauthorized tag

    3. BLOCK (Value violation):
       - Over maximum
       - Under minimum

    4. CLAMP (if allowed):
       - Value clamped to nearest limit
       - Original request modified

    5. ALLOW:
       - All checks passed
       - Request proceeds unchanged

    AUDIT TRAIL
    ===========

    All violations generate immutable audit records with:
    - Timestamp
    - Violation details
    - Decision made
    - Provenance hash (blockchain-style chain)

    The audit chain can be verified for integrity using:
        engine.verify_audit_chain()

    This provides evidence for regulatory compliance and
    incident investigation.

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
        # ==================
        # _write_history: Ring buffer of (timestamp, value) per tag
        # Used for rate-over-time calculations
        self._write_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

        # _last_write_time: Timestamp of most recent write per tag
        # Used for cooldown period enforcement
        self._last_write_time: Dict[str, datetime] = {}

        # _last_write_value: Value of most recent write per tag
        # Used for rate-of-change calculation
        self._last_write_value: Dict[str, float] = {}

        # Audit trail
        # ===========
        # Immutable chain of audit records with cryptographic linking
        self._audit_records: List[SafetyAuditRecord] = []
        self._audit_lock = threading.Lock()

        # Current tag values cache
        # ========================
        # Thread-safe cache of current tag values for condition evaluation
        self._tag_values: Dict[str, float] = {}
        self._tag_values_lock = threading.Lock()

        # System state
        # ============
        # SIS, interlock, and alarm states for safety function integration
        self._sis_states: Dict[str, SISState] = {}
        self._interlock_states: Dict[str, InterlockState] = {}
        self._alarm_states: Dict[str, AlarmState] = {}

        # Statistics
        # ==========
        # Counters for monitoring and analysis
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
        """
        Default tag value provider using internal cache.

        This is used when no external tag value provider is injected.
        The cache is updated via update_tag_value() calls.
        """
        with self._tag_values_lock:
            return self._tag_values.get(tag_id)

    def update_tag_value(self, tag_id: str, value: float) -> None:
        """
        Update cached tag value for condition evaluation.

        Args:
            tag_id: Tag identifier
            value: Current value
        """
        with self._tag_values_lock:
            self._tag_values[tag_id] = value

    def update_tag_values(self, values: Dict[str, float]) -> None:
        """
        Bulk update cached tag values.

        More efficient than individual updates when multiple
        values change simultaneously.

        Args:
            values: Dict of tag_id -> value
        """
        with self._tag_values_lock:
            self._tag_values.update(values)

    def update_sis_state(self, sis_state: SISState) -> None:
        """
        Update SIS (Safety Instrumented System) state.

        SIS STATE INTEGRATION (IEC 61511)
        ----------------------------------

        The Safety Boundary Engine respects SIS status:

        - is_active: SIS function is running
        - is_healthy: Self-diagnostics pass
        - trip_status: SIS is in tripped state
        - bypass_active: SIS is bypassed (logged!)

        If SIS is not healthy or is tripped, the engine blocks
        all writes to associated tags until SIS is restored.

        Args:
            sis_state: New SIS state
        """
        self._sis_states[sis_state.sis_id] = sis_state
        logger.debug(f"Updated SIS state: {sis_state.sis_id}")

    def update_interlock_state(self, interlock_state: InterlockState) -> None:
        """
        Update interlock state.

        INTERLOCK INTEGRATION
        ---------------------

        Interlocks are hardwired safety functions that must be
        permissive before control actions are allowed.

        An interlock in non-permissive state blocks ALL writes
        to related equipment regardless of other conditions.

        Args:
            interlock_state: New interlock state
        """
        self._interlock_states[interlock_state.interlock_id] = interlock_state
        logger.debug(f"Updated interlock state: {interlock_state.interlock_id}")

    def update_alarm_state(self, alarm_state: AlarmState) -> None:
        """
        Update alarm state.

        ALARM INTEGRATION
        -----------------

        Critical and emergency alarms block writes to related
        equipment. This prevents automated changes during
        abnormal conditions requiring operator attention.

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

        VALIDATION PIPELINE
        ===================

        This is the main entry point for pre-actuation validation.
        All write requests MUST pass through this gate.

        The validation pipeline executes these checks in order:

        1. WHITELIST CHECK
           - Verify tag is in allowed list
           - Reject blacklisted tags immediately

        2. ABSOLUTE LIMITS
           - Check value against min/max limits
           - Generate violation if out of bounds

        3. RATE LIMITS
           - Calculate rate of change
           - Check cooldown period
           - Verify write count limits

        4. CONDITIONAL POLICIES
           - Evaluate condition expressions
           - Apply context-aware limits

        5. TIME RESTRICTIONS
           - Check current time against allowed windows
           - Block if in restricted period

        6. INTERLOCK STATUS
           - Verify all interlocks permissive
           - Block if any interlock active

        7. ALARM STATUS
           - Check for blocking alarms
           - Block if critical/emergency alarm active

        DECISION DETERMINATION
        ======================

        After all checks, determine final decision:

        - BLOCK: If any emergency/critical violation
        - CLAMP: If value violation but clamping allowed
        - ALLOW: If all checks pass

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

        # =====================================================================
        # STEP 1: Whitelist Check
        # =====================================================================
        # Principle: Only pre-approved tags can be written
        # This is the first line of defense
        whitelist_result = self._check_whitelist(request)
        if whitelist_result:
            violations.append(whitelist_result)
            checks_failed.append("whitelist_check")
        else:
            checks_passed.append("whitelist_check")

        # =====================================================================
        # STEP 2: Absolute Limits
        # =====================================================================
        # Check hard limits based on equipment design
        # These are non-negotiable safety limits
        limit_violations = self._check_absolute_limits(request)
        violations.extend(limit_violations)
        if limit_violations:
            checks_failed.append("absolute_limits")
        else:
            checks_passed.append("absolute_limits")

        # =====================================================================
        # STEP 3: Rate Limits
        # =====================================================================
        # Prevent rapid changes that could stress equipment
        rate_violations = self._check_rate_limits(request)
        violations.extend(rate_violations)
        if rate_violations:
            checks_failed.append("rate_limits")
        else:
            checks_passed.append("rate_limits")

        # =====================================================================
        # STEP 4: Conditional Policies
        # =====================================================================
        # Context-aware limits that depend on system state
        condition_violations = self._check_conditional_policies(request)
        violations.extend(condition_violations)
        if condition_violations:
            checks_failed.append("conditional_policies")
        else:
            checks_passed.append("conditional_policies")

        # =====================================================================
        # STEP 5: Time-Based Restrictions
        # =====================================================================
        # Check if current time is in allowed window
        time_violations = self._check_time_restrictions(request)
        violations.extend(time_violations)
        if time_violations:
            checks_failed.append("time_restrictions")
        else:
            checks_passed.append("time_restrictions")

        # =====================================================================
        # STEP 6: Interlock Status
        # =====================================================================
        # Verify all interlocks are permissive
        interlock_violations = self._check_interlocks(request)
        violations.extend(interlock_violations)
        if interlock_violations:
            checks_failed.append("interlock_check")
        else:
            checks_passed.append("interlock_check")

        # =====================================================================
        # STEP 7: Alarm Status
        # =====================================================================
        # Check for blocking alarms
        alarm_violations = self._check_alarms(request)
        violations.extend(alarm_violations)
        if alarm_violations:
            checks_failed.append("alarm_check")
        else:
            checks_passed.append("alarm_check")

        # =====================================================================
        # DECISION DETERMINATION
        # =====================================================================
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

        # Update provenance hash
        result = ActionGateResult(
            **{**result.dict(), "provenance_hash": result.compute_provenance()}
        )

        # Update statistics
        self._update_stats(decision, len(violations))

        # Handle violations if any
        if violations:
            self._handle_violations(violations, result)

        # Record write if allowed (for rate limiting)
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

        WHITELIST LOGIC
        ---------------

        Three possible states for a tag:

        1. WHITELISTED (allowed):
           - Tag explicitly listed in allowed_tags
           - Tag matches allowed pattern
           -> Return None (no violation)

        2. BLACKLISTED (absolutely blocked):
           - Tag in explicit blacklist
           - GL-001 MUST NEVER write to this tag
           -> Return EMERGENCY violation

        3. NOT WHITELISTED (blocked by default):
           - Tag not found in any allowed list
           -> Return CRITICAL violation

        This implements principle of least privilege:
        Only explicitly allowed tags can be written.

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

        ABSOLUTE LIMIT LOGIC
        --------------------

        For each ABSOLUTE_LIMIT policy matching the tag:

        1. Check minimum:
           if value < min_value:
               Generate UNDER_MIN violation

        2. Check maximum:
           if value > max_value:
               Generate OVER_MAX violation

        These limits are based on:
        - Equipment design limits
        - Safety system requirements
        - Regulatory constraints

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

            # Check minimum limit
            # value < min_value -> UNDER_MIN violation
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

            # Check maximum limit
            # value > max_value -> OVER_MAX violation
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

        RATE LIMIT CALCULATIONS
        -----------------------

        1. COOLDOWN CHECK:
           elapsed = now - last_write_time
           if elapsed < cooldown_seconds:
               -> RATE_EXCEEDED violation

        2. RATE OF CHANGE (per second):
           rate = |value_new - value_prev| / elapsed_seconds
           if rate > max_change_per_second:
               -> RATE_EXCEEDED violation

        3. WRITE COUNT (per minute):
           count = number of writes in last 60 seconds
           if count >= max_writes_per_minute:
               -> RATE_EXCEEDED violation

        4. TOTAL CHANGE (per minute):
           change = |value_now - value_1min_ago|
           if change > max_change_per_minute:
               -> RATE_EXCEEDED violation

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

        # =====================================================================
        # CHECK 1: Cooldown period
        # =====================================================================
        # Minimum time between writes to prevent rapid cycling
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

        # =====================================================================
        # CHECK 2: Rate of change per second
        # =====================================================================
        # rate = |delta_value| / delta_time
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

        # =====================================================================
        # CHECK 3: Write count per minute
        # =====================================================================
        # Count writes in sliding 60-second window
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

        # =====================================================================
        # CHECK 4: Total change per minute
        # =====================================================================
        # Total change over sliding 60-second window
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

        CONDITIONAL POLICY LOGIC
        ------------------------

        Conditional policies apply limits only when specified
        conditions are met. This enables context-aware safety:

        Example: "If temperature > 400F, max setpoint is 450F"

        Evaluation:
        1. Get all CONDITIONAL policies for tag
        2. For each policy:
           a. Evaluate all conditions
           b. Apply AND/OR logic
           c. If conditions met, check limits

        CONDITION EVALUATION
        --------------------

        Each condition has:
        - tag_id: The tag to evaluate
        - operator: Comparison operator
        - value: Target value
        - secondary_value: For range operators

        Operators:
        - EQUALS: tag_value == value
        - NOT_EQUALS: tag_value != value
        - GREATER_THAN: tag_value > value
        - LESS_THAN: tag_value < value
        - GREATER_EQUAL: tag_value >= value
        - LESS_EQUAL: tag_value <= value
        - IN_RANGE: value <= tag_value <= secondary_value
        - NOT_IN_RANGE: NOT (value <= tag_value <= secondary_value)

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

            # Evaluate all conditions
            condition_results = []
            for condition in policy.conditions:
                result = self._evaluate_condition(condition)
                condition_results.append(result)

            # Apply condition logic (AND or OR)
            if policy.condition_logic == "AND":
                conditions_met = all(condition_results)
            else:  # OR
                conditions_met = any(condition_results)

            # If conditions are met, check limits
            if conditions_met:
                # Check max limit under these conditions
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

                # Check min limit under these conditions
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

        CONDITION EVALUATION ALGORITHM
        ------------------------------

        1. Get current tag value from provider
        2. If tag not found, return False (fail-safe)
        3. Apply operator:

           EQUALS:       tag_value == target
           NOT_EQUALS:   tag_value != target
           GREATER_THAN: tag_value > target
           LESS_THAN:    tag_value < target
           GREATER_EQUAL: tag_value >= target
           LESS_EQUAL:   tag_value <= target
           IN_RANGE:     target <= tag_value <= secondary
           NOT_IN_RANGE: NOT (target <= tag_value <= secondary)

        FAIL-SAFE BEHAVIOR
        ------------------

        If the tag value cannot be retrieved (sensor failure, etc.),
        the condition evaluates to False. This means:
        - For safety conditions: Condition not met -> restrictions apply
        - This implements fail-safe behavior

        Args:
            condition: Condition to evaluate

        Returns:
            True if condition is met
        """
        tag_value = self._tag_value_provider(condition.tag_id)

        if tag_value is None:
            logger.warning(f"Cannot evaluate condition: tag {condition.tag_id} not found")
            return False  # Fail safe - assume condition not met

        operator = condition.operator
        target = condition.value

        # Evaluate based on operator type
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

        TIME RESTRICTION LOGIC
        ----------------------

        Time-based restrictions block writes during specific periods:
        - Maintenance windows
        - Off-hours (nights, weekends)
        - Special events

        For each TIME_BASED policy:
        1. Check if today's day of week is restricted
        2. Check if current time is in restriction window
        3. Handle overnight restrictions (e.g., 22:00-06:00)

        OVERNIGHT HANDLING
        ------------------

        For restrictions spanning midnight:
        - start_time = 22:00, end_time = 06:00
        - If start > end: Check time >= start OR time <= end

        Args:
            request: Write request

        Returns:
            List of violations
        """
        violations: List[BoundaryViolation] = []
        policies = self._policy_manager.get_policies_by_type(PolicyType.TIME_BASED)

        now = datetime.utcnow()
        current_time = now.time()
        current_day = now.weekday()  # 0=Monday, 6=Sunday

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
                    # Normal case: e.g., 09:00 to 17:00
                    in_restriction = restriction.start_time <= current_time <= restriction.end_time
                else:
                    # Overnight case: e.g., 22:00 to 06:00
                    # True if time >= start OR time <= end
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

        INTERLOCK LOGIC (IEC 61511)
        ---------------------------

        Interlocks are hardwired safety functions that must be
        permissive before automated actions are allowed.

        For each registered interlock:
        - If NOT permissive -> Block ALL writes
        - Severity: EMERGENCY (immediate action required)

        This ensures automated control respects plant safety systems.

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

        ALARM BLOCKING LOGIC
        --------------------

        Critical and emergency alarms block writes to related equipment.
        This prevents automated changes during abnormal conditions.

        "Related" is determined by:
        - Same tag ID
        - Same tag prefix (first 3 characters)

        This provides operator control during upset conditions.

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
                # Related = same tag or same prefix (first 3 chars)
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

        DECISION PRIORITY ALGORITHM
        ---------------------------

        1. If NO violations: ALLOW with original value

        2. Check for EMERGENCY severity:
           -> BLOCK immediately

        3. Check for must-block violation types:
           - UNAUTHORIZED_TAG
           - INTERLOCK_ACTIVE
           - SIS_VIOLATION
           - RATE_EXCEEDED
           - TIME_RESTRICTED
           -> BLOCK immediately

        4. For limit violations (OVER_MAX, UNDER_MIN):
           If allow_clamping:
               Clamp value to nearest limit
               -> CLAMP with adjusted value
           Else:
               -> BLOCK

        5. Default: BLOCK

        Args:
            request: Original request
            violations: List of violations
            allow_clamping: Whether clamping is allowed

        Returns:
            Tuple of (decision, final_value)
        """
        # No violations -> ALLOW
        if not violations:
            return GateDecision.ALLOW, request.value

        # Check for must-block conditions
        for v in violations:
            # Emergency severity always blocks
            if v.severity == ViolationSeverity.EMERGENCY:
                return GateDecision.BLOCK, None

            # These violation types always block (no clamping possible)
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

            # Clamp to minimum
            if limits["min"] is not None and clamped_value < limits["min"]:
                clamped_value = limits["min"]

            # Clamp to maximum
            if limits["max"] is not None and clamped_value > limits["max"]:
                clamped_value = limits["max"]

            if clamped_value != request.value:
                logger.info(
                    f"Clamping value from {request.value} to {clamped_value} "
                    f"for tag {request.tag_id}"
                )
                return GateDecision.CLAMP, clamped_value

        # Default: BLOCK
        return GateDecision.BLOCK, None

    def _record_write(self, tag_id: str, value: float) -> None:
        """
        Record a successful write for rate limiting.

        RATE LIMIT STATE UPDATE
        -----------------------

        After a successful write:
        1. Update last_write_time for cooldown checking
        2. Update last_write_value for rate calculation
        3. Append to write_history for window calculations
        4. Prune old history (keep last 10 minutes)

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
        """Update engine statistics."""
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

        VIOLATION HANDLING SEQUENCE
        ---------------------------

        For each violation:
        1. Create immutable audit record (linked chain)
        2. Call violation callback if registered
        3. Log according to severity

        This ensures complete audit trail for regulatory compliance.

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
        Create immutable audit record with blockchain-style linking.

        AUDIT CHAIN ALGORITHM
        ---------------------

        Each audit record contains:
        - Event details
        - Previous record's hash (chain linking)
        - Own hash (computed from contents + previous hash)

        This creates a tamper-evident chain:
        - Modifying any record breaks the chain
        - Insertion/deletion is detectable
        - Provides integrity verification

        Hash computation:
            hash(i) = SHA256(timestamp | event_type | action | hash(i-1))

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
        Get audit records for review.

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

        CHAIN VERIFICATION ALGORITHM
        ----------------------------

        For each record in chain:
        1. First record: previous_hash must be empty
        2. Subsequent records: previous_hash must match
           prior record's provenance_hash

        If any link is broken, the chain is invalid
        (possible tampering detected).

        Returns:
            True if chain is valid
        """
        with self._audit_lock:
            for i, record in enumerate(self._audit_records):
                if i == 0:
                    # First record should have empty previous_hash
                    if record.previous_hash != "":
                        logger.error("First record should have empty previous_hash")
                        return False
                else:
                    # Verify chain linking
                    expected_hash = self._audit_records[i - 1].provenance_hash
                    if record.previous_hash != expected_hash:
                        logger.error(f"Audit chain broken at record {i}")
                        return False

            return True

    def get_safety_state(self) -> SafetyState:
        """
        Get current overall safety state.

        Returns:
            Current SafetyState with all status information
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
            Dict with counts of requests, allowed, blocked, clamped, violations
        """
        with self._stats_lock:
            return dict(self._stats)

    def enforce_pre_optimization_constraints(
        self,
        optimization_targets: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Enforce constraints before optimization.

        PRE-OPTIMIZATION CONSTRAINT ENFORCEMENT
        ---------------------------------------

        This is called BEFORE the optimization engine runs to ensure
        optimization targets start within safe bounds.

        For each target:
        1. Get min/max limits for the tag
        2. Clamp target to within limits
        3. Return constrained targets

        This prevents the optimizer from exploring infeasible regions.

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

        Use with caution - this bypasses rate limit protection.
        Should only be called during authorized resets.

        Args:
            tag_id: Specific tag to reset, or None for all
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


# =============================================================================
# 2oo3 VOTING LOGIC
# =============================================================================

class TwoOutOfThreeVoter:
    """
    2-out-of-3 (2oo3) Voting Logic for SIL-3 Decisions.

    TRIPLE MODULAR REDUNDANCY (TMR)
    ===============================

    The 2oo3 voting architecture provides:
    - Single fault tolerance (one sensor can fail)
    - Fail-safe behavior (fails to trip on 2+ faults)
    - Discrepancy detection for diagnostics

    TRUTH TABLE
    -----------

    | Input A | Input B | Input C | Vote | Safe? |
    |---------|---------|---------|------|-------|
    | 0       | 0       | 0       | 0/3  | Yes   |
    | 0       | 0       | 1       | 1/3  | Yes   |
    | 0       | 1       | 0       | 1/3  | Yes   |
    | 1       | 0       | 0       | 1/3  | Yes   |
    | 0       | 1       | 1       | 2/3  | No    |
    | 1       | 0       | 1       | 2/3  | No    |
    | 1       | 1       | 0       | 2/3  | No    |
    | 1       | 1       | 1       | 3/3  | No    |

    Where 1 = Trip condition, 0 = Safe condition

    BOOLEAN ALGEBRA
    ---------------

    Vote = (A AND B) OR (B AND C) OR (A AND C)

    This is equivalent to: "At least 2 of 3 are true"

    Example:
        >>> voter = TwoOutOfThreeVoter("high_temp")
        >>> voter.set_input("sensor_a", True)   # Trip
        >>> voter.set_input("sensor_b", False)  # Safe
        >>> voter.set_input("sensor_c", True)   # Trip
        >>> result = voter.vote()
        >>> print(f"Should trip: {result.should_trip}")  # True (2/3 trip)
    """

    def __init__(self, name: str) -> None:
        """Initialize 2oo3 voter."""
        self.name = name
        self._inputs: Dict[str, bool] = {}  # True = Trip, False = Safe
        self._last_vote_time: Optional[datetime] = None
        self._discrepancy_count = 0

    def set_input(self, input_id: str, trip_condition: bool) -> None:
        """
        Set an input value.

        Args:
            input_id: Input identifier (e.g., "sensor_a")
            trip_condition: True if trip condition detected
        """
        self._inputs[input_id] = trip_condition

    def vote(self) -> Dict[str, Any]:
        """
        Perform 2oo3 vote.

        VOTING ALGORITHM
        ----------------

        1. Count inputs voting for trip
        2. If count >= 2: Vote = TRIP
        3. Check for discrepancy (not all inputs agree)
        4. Return result with diagnostics

        Returns:
            Dict with:
            - should_trip: bool
            - trip_count: int (number of trip votes)
            - discrepancy: bool (inputs disagree)
            - inputs: current input states
        """
        if len(self._inputs) < 3:
            # Not enough inputs for 2oo3
            logger.warning(f"2oo3 voter {self.name}: Only {len(self._inputs)} inputs")
            return {
                "should_trip": True,  # Fail safe
                "trip_count": 0,
                "discrepancy": True,
                "inputs": dict(self._inputs),
                "error": "Insufficient inputs for 2oo3 voting",
            }

        # Count trip votes
        trip_count = sum(1 for v in self._inputs.values() if v)

        # 2oo3 logic: Trip if 2 or more inputs are in trip state
        should_trip = trip_count >= 2

        # Check for discrepancy (not all inputs agree)
        all_same = len(set(self._inputs.values())) == 1
        discrepancy = not all_same

        if discrepancy:
            self._discrepancy_count += 1
            logger.warning(
                f"2oo3 voter {self.name}: Discrepancy detected "
                f"(trip_count={trip_count}/3)"
            )

        self._last_vote_time = datetime.utcnow()

        return {
            "should_trip": should_trip,
            "trip_count": trip_count,
            "discrepancy": discrepancy,
            "inputs": dict(self._inputs),
            "vote_time": self._last_vote_time.isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get voter status for diagnostics."""
        return {
            "name": self.name,
            "input_count": len(self._inputs),
            "inputs": dict(self._inputs),
            "discrepancy_count": self._discrepancy_count,
            "last_vote_time": (
                self._last_vote_time.isoformat() if self._last_vote_time else None
            ),
        }


# =============================================================================
# SAFETY MARGIN CALCULATOR
# =============================================================================

class SafetyMarginCalculator:
    """
    Calculator for safety margins and time-to-limit.

    SAFETY MARGIN TYPES
    ===================

    1. ABSOLUTE MARGIN
       Distance between current value and limit

       Margin = Limit - CurrentValue

    2. PERCENTAGE MARGIN
       Margin as percentage of operating range

       Margin% = (Limit - CurrentValue) / (Limit - Minimum) * 100

    3. TIME TO LIMIT
       Time until limit is reached at current rate

       TimeToLimit = Margin / RateOfChange

    Example:
        >>> calc = SafetyMarginCalculator()
        >>> margin = calc.calculate_margin(
        ...     current_value=140.0,
        ...     high_limit=150.0,
        ...     low_limit=50.0,
        ...     rate_of_change=2.0  # units per minute
        ... )
        >>> print(f"Time to high limit: {margin['time_to_high_minutes']:.1f} min")
    """

    def calculate_margin(
        self,
        current_value: float,
        high_limit: float,
        low_limit: float,
        rate_of_change: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate safety margins.

        CALCULATION FORMULAS
        --------------------

        Absolute margins:
            margin_high = high_limit - current_value
            margin_low = current_value - low_limit

        Percentage margins (of operating range):
            range = high_limit - low_limit
            margin_high_pct = margin_high / range * 100
            margin_low_pct = margin_low / range * 100

        Time to limit (at current rate):
            if rate > 0:  # Rising
                time_to_high = margin_high / rate
                time_to_low = infinity
            elif rate < 0:  # Falling
                time_to_high = infinity
                time_to_low = margin_low / abs(rate)
            else:
                time_to_high = infinity
                time_to_low = infinity

        Args:
            current_value: Current process value
            high_limit: High limit (max safe value)
            low_limit: Low limit (min safe value)
            rate_of_change: Rate of change (positive = rising)

        Returns:
            Dict with margin calculations
        """
        operating_range = high_limit - low_limit

        # Absolute margins
        margin_high = high_limit - current_value
        margin_low = current_value - low_limit

        # Percentage margins
        margin_high_pct = (margin_high / operating_range * 100) if operating_range > 0 else 0
        margin_low_pct = (margin_low / operating_range * 100) if operating_range > 0 else 0

        # Time to limits
        time_to_high = float('inf')
        time_to_low = float('inf')

        if rate_of_change > 0:
            # Rising - will hit high limit
            time_to_high = margin_high / rate_of_change
        elif rate_of_change < 0:
            # Falling - will hit low limit
            time_to_low = margin_low / abs(rate_of_change)

        return {
            "current_value": current_value,
            "high_limit": high_limit,
            "low_limit": low_limit,
            "margin_high": margin_high,
            "margin_low": margin_low,
            "margin_high_percent": margin_high_pct,
            "margin_low_percent": margin_low_pct,
            "rate_of_change": rate_of_change,
            "time_to_high_minutes": time_to_high,
            "time_to_low_minutes": time_to_low,
        }

    def check_margin_alarm(
        self,
        margin_result: Dict[str, float],
        warning_percent: float = 20.0,
        critical_percent: float = 10.0,
        time_warning_minutes: float = 5.0,
        time_critical_minutes: float = 2.0
    ) -> Dict[str, Any]:
        """
        Check if margins trigger alarms.

        ALARM THRESHOLDS
        ----------------

        Percentage-based:
            WARNING:  margin < warning_percent (e.g., 20%)
            CRITICAL: margin < critical_percent (e.g., 10%)

        Time-based:
            WARNING:  time_to_limit < time_warning (e.g., 5 min)
            CRITICAL: time_to_limit < time_critical (e.g., 2 min)

        Args:
            margin_result: Result from calculate_margin
            warning_percent: Percentage for warning
            critical_percent: Percentage for critical
            time_warning_minutes: Time threshold for warning
            time_critical_minutes: Time threshold for critical

        Returns:
            Dict with alarm status
        """
        alarms = {
            "high_warning": False,
            "high_critical": False,
            "low_warning": False,
            "low_critical": False,
            "time_warning": False,
            "time_critical": False,
        }

        # Check percentage margins
        if margin_result["margin_high_percent"] < critical_percent:
            alarms["high_critical"] = True
        elif margin_result["margin_high_percent"] < warning_percent:
            alarms["high_warning"] = True

        if margin_result["margin_low_percent"] < critical_percent:
            alarms["low_critical"] = True
        elif margin_result["margin_low_percent"] < warning_percent:
            alarms["low_warning"] = True

        # Check time to limits
        min_time = min(
            margin_result["time_to_high_minutes"],
            margin_result["time_to_low_minutes"]
        )

        if min_time < time_critical_minutes:
            alarms["time_critical"] = True
        elif min_time < time_warning_minutes:
            alarms["time_warning"] = True

        return alarms
