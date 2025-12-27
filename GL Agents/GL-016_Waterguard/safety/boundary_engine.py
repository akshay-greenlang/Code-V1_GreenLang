"""
GL-016 Waterguard Safety Boundary Engine - IEC 61511 SIL-3 Compliant

This module implements the core Safety Boundary Engine for Waterguard water
treatment optimization. It enforces hard limits on all control actions to
prevent equipment damage, safety incidents, and water chemistry excursions.

Key Features:
    - Hard constraint enforcement for water chemistry parameters
    - OEM equipment limit protection
    - Watchdog timer for heartbeat monitoring
    - Fail-safe state management
    - Immutable audit trail with SHA-256 hashes

Water Treatment Constraints:
    - Conductivity: Prevents scaling and corrosion
    - Silica: Prevents turbine blade deposits
    - pH: Prevents acid/caustic damage
    - Dissolved Oxygen: Prevents pitting corrosion
    - Blowdown Rate: Protects water balance
    - Chemical Dosing: Prevents overdose incidents

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - ASME PTC 19.11 Steam and Water Sampling
    - EPRI Water Chemistry Guidelines

Example:
    >>> from boundary_engine import WaterguardBoundaryEngine
    >>> engine = WaterguardBoundaryEngine()
    >>> action = ProposedAction(action_type="BLOWDOWN", target_value=15.0)
    >>> permitted, reason = engine.validate_action(action)
    >>> if not permitted:
    ...     engine.fail_safe_state()

Author: GreenLang Safety Engineering Team
Version: 1.0.0
SIL Level: 3
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ConstraintType(str, Enum):
    """Types of safety constraints enforced by the boundary engine."""

    CONDUCTIVITY = "conductivity"
    SILICA = "silica"
    PH = "ph"
    DISSOLVED_O2 = "dissolved_o2"
    PHOSPHATE = "phosphate"
    HYDRAZINE = "hydrazine"
    BLOWDOWN_RATE = "blowdown_rate"
    DOSING_RATE = "dosing_rate"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    LEVEL = "level"
    OEM_LIMIT = "oem_limit"


class ViolationSeverity(str, Enum):
    """Severity levels for constraint violations."""

    WARNING = "warning"       # Log and alert, allow action with clamping
    CRITICAL = "critical"     # Block action, require operator acknowledgement
    EMERGENCY = "emergency"   # Block action, trigger fail-safe state


class HeartbeatStatus(str, Enum):
    """Status of the watchdog heartbeat."""

    HEALTHY = "healthy"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class FailSafeMode(str, Enum):
    """Fail-safe operating modes."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    FAIL_SAFE = "fail_safe"
    EMERGENCY_STOP = "emergency_stop"


# =============================================================================
# DATA MODELS
# =============================================================================


class WaterguardConstraints(BaseModel):
    """
    Hard constraints for water treatment parameters.

    These constraints represent absolute limits that MUST NOT be exceeded
    under any circumstances. Values are based on OEM specifications and
    industry best practices (EPRI guidelines).

    Attributes:
        max_conductivity_us_cm: Maximum feedwater conductivity
        max_silica_ppb: Maximum silica concentration
        ph_min: Minimum pH value
        ph_max: Maximum pH value
        max_dissolved_o2_ppb: Maximum dissolved oxygen
        max_blowdown_rate_pct: Maximum blowdown rate percentage
        max_dosing_rate_pct: Maximum chemical dosing rate
        oem_limits: Equipment-specific OEM limits
    """

    # Conductivity limits (microsiemens/cm)
    max_conductivity_us_cm: float = Field(
        default=3500.0,
        ge=0,
        le=10000,
        description="Maximum drum conductivity (uS/cm)"
    )
    feedwater_max_conductivity_us_cm: float = Field(
        default=0.5,
        ge=0,
        le=10,
        description="Maximum feedwater conductivity (uS/cm)"
    )

    # Silica limits (ppb)
    max_silica_ppb: float = Field(
        default=150.0,
        ge=0,
        le=500,
        description="Maximum silica in drum water (ppb)"
    )
    feedwater_max_silica_ppb: float = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Maximum feedwater silica (ppb)"
    )

    # pH limits (dimensionless)
    ph_min: float = Field(
        default=8.5,
        ge=6.0,
        le=10.0,
        description="Minimum drum water pH"
    )
    ph_max: float = Field(
        default=10.5,
        ge=8.0,
        le=12.0,
        description="Maximum drum water pH"
    )

    # Dissolved oxygen limits (ppb)
    max_dissolved_o2_ppb: float = Field(
        default=7.0,
        ge=0,
        le=100,
        description="Maximum dissolved oxygen (ppb)"
    )

    # Phosphate limits (ppm as PO4)
    phosphate_min_ppm: float = Field(
        default=2.0,
        ge=0,
        le=10,
        description="Minimum phosphate concentration (ppm)"
    )
    phosphate_max_ppm: float = Field(
        default=6.0,
        ge=1,
        le=20,
        description="Maximum phosphate concentration (ppm)"
    )

    # Blowdown rate limits (% of steam flow)
    max_blowdown_rate_pct: float = Field(
        default=5.0,
        ge=0,
        le=15,
        description="Maximum blowdown rate (% of steam)"
    )
    min_blowdown_rate_pct: float = Field(
        default=0.5,
        ge=0,
        le=5,
        description="Minimum blowdown rate (% of steam)"
    )

    # Rate of change limits
    max_blowdown_delta_per_min: float = Field(
        default=1.0,
        ge=0,
        le=5,
        description="Maximum blowdown rate change per minute (%)"
    )
    max_dosing_delta_per_min: float = Field(
        default=5.0,
        ge=0,
        le=20,
        description="Maximum dosing rate change per minute (%)"
    )

    # Chemical dosing limits (% of maximum pump capacity)
    max_dosing_rate_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Maximum chemical dosing rate (%)"
    )

    # Temperature limits (degrees F)
    max_drum_temp_f: float = Field(
        default=700.0,
        ge=200,
        le=800,
        description="Maximum drum temperature (F)"
    )

    # Pressure limits (psig)
    max_drum_pressure_psig: float = Field(
        default=2000.0,
        ge=0,
        le=3000,
        description="Maximum drum pressure (psig)"
    )

    # OEM-specific limits (equipment tag -> limit)
    oem_limits: Dict[str, float] = Field(
        default_factory=dict,
        description="Equipment-specific OEM limits"
    )

    @field_validator('ph_max')
    @classmethod
    def validate_ph_range(cls, v: float, info) -> float:
        """Ensure pH max is greater than pH min."""
        ph_min = info.data.get('ph_min', 8.5)
        if v <= ph_min:
            raise ValueError(f"pH max ({v}) must be greater than pH min ({ph_min})")
        return v


class ProposedAction(BaseModel):
    """
    A proposed control action to be validated by the boundary engine.

    Attributes:
        action_id: Unique action identifier
        action_type: Type of action (BLOWDOWN, DOSING, etc.)
        target_tag: Target equipment tag
        target_value: Proposed value
        current_value: Current measured value
        timestamp: Action timestamp
    """

    action_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique action identifier"
    )
    action_type: str = Field(
        ...,
        description="Type of control action"
    )
    target_tag: str = Field(
        ...,
        description="Target equipment tag"
    )
    target_value: float = Field(
        ...,
        description="Proposed value"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current measured value"
    )
    engineering_units: str = Field(
        default="",
        description="Engineering units"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Action timestamp"
    )
    source: str = Field(
        default="WATERGUARD",
        description="Source of the action"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Action priority (1=highest)"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )


class ConstraintViolation(BaseModel):
    """
    Record of a constraint violation.

    Attributes:
        violation_id: Unique violation identifier
        constraint_type: Type of constraint violated
        severity: Violation severity
        proposed_value: Value that would violate constraint
        limit_value: The constraint limit
        message: Human-readable violation message
        provenance_hash: SHA-256 hash for audit
    """

    violation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique violation identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Violation timestamp"
    )
    constraint_type: ConstraintType = Field(
        ...,
        description="Type of constraint violated"
    )
    severity: ViolationSeverity = Field(
        ...,
        description="Violation severity"
    )
    action: ProposedAction = Field(
        ...,
        description="The action that caused violation"
    )
    proposed_value: float = Field(
        ...,
        description="Value that would violate constraint"
    )
    limit_value: float = Field(
        ...,
        description="The constraint limit"
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current measured value"
    )
    message: str = Field(
        ...,
        description="Human-readable violation message"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{self.violation_id}|{self.timestamp.isoformat()}|"
            f"{self.constraint_type.value}|{self.severity.value}|"
            f"{self.proposed_value}|{self.limit_value}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()


class FailSafeState(BaseModel):
    """
    Definition of the fail-safe operating state.

    When triggered, WATERGUARD reverts to this baseline configuration
    that is known to be safe.
    """

    state_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique state identifier"
    )
    mode: FailSafeMode = Field(
        default=FailSafeMode.NORMAL,
        description="Current operating mode"
    )
    blowdown_rate_pct: float = Field(
        default=2.0,
        ge=0,
        le=10,
        description="Safe blowdown rate (%)"
    )
    chemical_dosing_enabled: bool = Field(
        default=False,
        description="Chemical dosing enabled in fail-safe"
    )
    optimization_enabled: bool = Field(
        default=False,
        description="Optimization enabled in fail-safe"
    )
    manual_control_only: bool = Field(
        default=True,
        description="Require manual control only"
    )
    reason: str = Field(
        default="",
        description="Reason for entering fail-safe"
    )
    entered_at: Optional[datetime] = Field(
        default=None,
        description="Time fail-safe was entered"
    )
    operator_notified: bool = Field(
        default=False,
        description="Operator has been notified"
    )

    def enter_fail_safe(self, reason: str) -> None:
        """Enter fail-safe mode."""
        self.mode = FailSafeMode.FAIL_SAFE
        self.reason = reason
        self.entered_at = datetime.now(timezone.utc)
        self.optimization_enabled = False
        self.chemical_dosing_enabled = False
        self.manual_control_only = True


# =============================================================================
# WATCHDOG TIMER
# =============================================================================


class WatchdogTimer:
    """
    Watchdog timer for heartbeat monitoring.

    Monitors the health of the WATERGUARD system and triggers
    fail-safe state if heartbeat expires.

    IEC 61511 SIL-3 Requirement: Watchdog timeout < 10 seconds

    Attributes:
        timeout_seconds: Heartbeat timeout
        warning_threshold_seconds: Warning threshold before timeout

    Example:
        >>> watchdog = WatchdogTimer(timeout_seconds=5.0)
        >>> watchdog.start()
        >>> while running:
        ...     do_work()
        ...     watchdog.pet()
        >>> watchdog.stop()
    """

    def __init__(
        self,
        timeout_seconds: float = 5.0,
        warning_threshold_seconds: float = 3.0,
        on_expire_callback: Optional[Callable[[], None]] = None,
        on_warning_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Initialize watchdog timer.

        Args:
            timeout_seconds: Time until heartbeat expires (SIL-3: < 10s)
            warning_threshold_seconds: Time until warning issued
            on_expire_callback: Called when watchdog expires
            on_warning_callback: Called when warning threshold reached
        """
        if timeout_seconds > 10.0:
            logger.warning(
                "Watchdog timeout %.1fs exceeds SIL-3 recommended 10s",
                timeout_seconds
            )

        self._timeout = timeout_seconds
        self._warning_threshold = warning_threshold_seconds
        self._on_expire = on_expire_callback
        self._on_warning = on_warning_callback

        self._last_pet: Optional[float] = None
        self._started = False
        self._expired = False
        self._warning_issued = False

        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info(
            "WatchdogTimer initialized: timeout=%.1fs, warning=%.1fs",
            timeout_seconds, warning_threshold_seconds
        )

    def start(self) -> None:
        """Start the watchdog timer."""
        with self._lock:
            if self._started:
                return

            self._started = True
            self._expired = False
            self._warning_issued = False
            self._last_pet = time.monotonic()
            self._stop_event.clear()

            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="WatchdogMonitor"
            )
            self._monitor_thread.start()

            logger.info("WatchdogTimer started")

    def stop(self) -> None:
        """Stop the watchdog timer."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        with self._lock:
            self._started = False

        logger.info("WatchdogTimer stopped")

    def pet(self) -> None:
        """
        Pet the watchdog (reset timer).

        This must be called regularly to prevent the watchdog from expiring.
        Call frequency should be < timeout_seconds / 2.
        """
        with self._lock:
            self._last_pet = time.monotonic()
            self._warning_issued = False

            if self._expired:
                logger.info("Watchdog recovered from expired state")
                self._expired = False

    def get_status(self) -> HeartbeatStatus:
        """
        Get current watchdog status.

        Returns:
            HeartbeatStatus indicating current state
        """
        with self._lock:
            if not self._started:
                return HeartbeatStatus.UNKNOWN

            if self._expired:
                return HeartbeatStatus.EXPIRED

            if self._last_pet is None:
                return HeartbeatStatus.UNKNOWN

            elapsed = time.monotonic() - self._last_pet

            if elapsed >= self._timeout:
                return HeartbeatStatus.EXPIRED
            elif elapsed >= self._warning_threshold:
                return HeartbeatStatus.STALE
            else:
                return HeartbeatStatus.HEALTHY

    def get_time_since_last_pet(self) -> Optional[float]:
        """Get seconds since last pet."""
        with self._lock:
            if self._last_pet is None:
                return None
            return time.monotonic() - self._last_pet

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            with self._lock:
                if self._last_pet is not None:
                    elapsed = time.monotonic() - self._last_pet

                    # Check for warning
                    if (elapsed >= self._warning_threshold and
                            not self._warning_issued and
                            not self._expired):
                        self._warning_issued = True
                        logger.warning(
                            "Watchdog warning: %.1fs since last heartbeat",
                            elapsed
                        )
                        if self._on_warning:
                            try:
                                self._on_warning()
                            except Exception as e:
                                logger.error("Warning callback failed: %s", e)

                    # Check for expiration
                    if elapsed >= self._timeout and not self._expired:
                        self._expired = True
                        logger.critical(
                            "WATCHDOG EXPIRED: %.1fs since last heartbeat",
                            elapsed
                        )
                        if self._on_expire:
                            try:
                                self._on_expire()
                            except Exception as e:
                                logger.error("Expire callback failed: %s", e)

            # Sleep briefly between checks
            self._stop_event.wait(timeout=0.1)


# =============================================================================
# BOUNDARY ENGINE
# =============================================================================


class WaterguardBoundaryEngine:
    """
    SIL-3 Safety Boundary Engine for Waterguard.

    Enforces hard limits on all control actions to prevent safety
    incidents and equipment damage. This is the primary safety gate
    that all proposed actions must pass through.

    Key Responsibilities:
        - Validate all proposed actions against hard constraints
        - Enforce rate-of-change limits
        - Monitor watchdog heartbeat
        - Trigger fail-safe state on violations
        - Maintain immutable audit trail

    CRITICAL: WATERGUARD is SUPERVISORY ONLY.
    It cannot override SIS, BMS, or low-water cutoff systems.

    Attributes:
        constraints: Water chemistry and equipment constraints
        fail_safe_state: Current fail-safe configuration
        watchdog: Heartbeat monitoring timer

    Example:
        >>> engine = WaterguardBoundaryEngine()
        >>> action = ProposedAction(
        ...     action_type="BLOWDOWN_ADJUST",
        ...     target_tag="BD-001",
        ...     target_value=15.0
        ... )
        >>> permitted, reason = engine.validate_action(action)
        >>> if not permitted:
        ...     engine.fail_safe_state()
    """

    def __init__(
        self,
        constraints: Optional[WaterguardConstraints] = None,
        violation_callback: Optional[Callable[[ConstraintViolation], None]] = None,
        watchdog_timeout_seconds: float = 5.0,
    ) -> None:
        """
        Initialize Boundary Engine.

        Args:
            constraints: Water chemistry constraints (defaults to standard)
            violation_callback: Callback for constraint violations
            watchdog_timeout_seconds: Watchdog timeout
        """
        self._constraints = constraints or WaterguardConstraints()
        self._violation_callback = violation_callback

        # Fail-safe state
        self._fail_safe = FailSafeState()
        self._fail_safe_lock = threading.Lock()

        # Watchdog timer
        self._watchdog = WatchdogTimer(
            timeout_seconds=watchdog_timeout_seconds,
            on_expire_callback=self._on_watchdog_expire,
        )

        # Rate tracking for rate-of-change limits
        self._last_values: Dict[str, Tuple[float, datetime]] = {}
        self._values_lock = threading.Lock()

        # Violation history
        self._violations: List[ConstraintViolation] = []
        self._violations_lock = threading.Lock()

        # Statistics
        self._stats = {
            "actions_validated": 0,
            "actions_permitted": 0,
            "actions_blocked": 0,
            "violations_total": 0,
            "fail_safe_triggered": 0,
        }
        self._stats_lock = threading.Lock()

        # Audit trail
        self._audit_records: List[Dict[str, Any]] = []
        self._audit_lock = threading.Lock()

        logger.info(
            "WaterguardBoundaryEngine initialized with SIL-3 constraints"
        )

    def start(self) -> None:
        """Start the boundary engine and watchdog."""
        self._watchdog.start()
        logger.info("WaterguardBoundaryEngine started")

    def stop(self) -> None:
        """Stop the boundary engine and watchdog."""
        self._watchdog.stop()
        logger.info("WaterguardBoundaryEngine stopped")

    def pet_watchdog(self) -> None:
        """Pet the watchdog to indicate healthy operation."""
        self._watchdog.pet()

    def validate_action(
        self,
        action: ProposedAction,
    ) -> Tuple[bool, str]:
        """
        Validate a proposed action against all safety constraints.

        This is the main entry point for action validation. ALL proposed
        actions MUST pass through this method before execution.

        Args:
            action: The proposed control action

        Returns:
            Tuple of (permitted: bool, reason: str)
            - permitted: True if action is allowed
            - reason: Explanation of decision

        Example:
            >>> action = ProposedAction(
            ...     action_type="BLOWDOWN_ADJUST",
            ...     target_tag="BD-001",
            ...     target_value=6.0
            ... )
            >>> permitted, reason = engine.validate_action(action)
            >>> if permitted:
            ...     execute_action(action)
        """
        with self._stats_lock:
            self._stats["actions_validated"] += 1

        start_time = datetime.now(timezone.utc)
        violations: List[ConstraintViolation] = []

        logger.debug(
            "Validating action: type=%s, tag=%s, value=%.2f",
            action.action_type, action.target_tag, action.target_value
        )

        # Step 1: Check if in fail-safe mode
        if self._fail_safe.mode != FailSafeMode.NORMAL:
            self._record_blocked_action(action, "Fail-safe mode active")
            return False, f"Action blocked: System in {self._fail_safe.mode.value} mode"

        # Step 2: Check watchdog status
        watchdog_status = self._watchdog.get_status()
        if watchdog_status == HeartbeatStatus.EXPIRED:
            self._record_blocked_action(action, "Watchdog expired")
            return False, "Action blocked: Watchdog heartbeat expired"

        # Step 3: Validate against constraint type
        action_type = action.action_type.upper()

        if "BLOWDOWN" in action_type:
            violations.extend(self._validate_blowdown(action))
        elif "DOSING" in action_type or "CHEMICAL" in action_type:
            violations.extend(self._validate_dosing(action))
        elif "CONDUCTIVITY" in action_type:
            violations.extend(self._validate_conductivity_setpoint(action))
        elif "SILICA" in action_type:
            violations.extend(self._validate_silica_setpoint(action))
        elif "PH" in action_type:
            violations.extend(self._validate_ph_setpoint(action))
        elif "PHOSPHATE" in action_type:
            violations.extend(self._validate_phosphate_setpoint(action))
        else:
            # Generic validation
            violations.extend(self._validate_generic(action))

        # Step 4: Check rate of change
        rate_violations = self._validate_rate_of_change(action)
        violations.extend(rate_violations)

        # Step 5: Check OEM limits
        oem_violations = self._validate_oem_limits(action)
        violations.extend(oem_violations)

        # Step 6: Process violations
        if violations:
            # Store violations
            with self._violations_lock:
                self._violations.extend(violations)

            # Call violation callback
            if self._violation_callback:
                for violation in violations:
                    try:
                        self._violation_callback(violation)
                    except Exception as e:
                        logger.error("Violation callback failed: %s", e)

            # Check for emergency severity
            emergency_violations = [
                v for v in violations
                if v.severity == ViolationSeverity.EMERGENCY
            ]
            if emergency_violations:
                self._trigger_fail_safe(
                    f"Emergency violation: {emergency_violations[0].message}"
                )

            with self._stats_lock:
                self._stats["actions_blocked"] += 1
                self._stats["violations_total"] += len(violations)

            self._record_blocked_action(action, violations[0].message)
            return False, violations[0].message

        # Action permitted
        with self._stats_lock:
            self._stats["actions_permitted"] += 1

        # Update rate tracking
        self._update_rate_tracking(action)

        # Record audit
        self._record_permitted_action(action)

        evaluation_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        logger.info(
            "Action PERMITTED: type=%s, tag=%s, value=%.2f (%.1fms)",
            action.action_type, action.target_tag, action.target_value,
            evaluation_time_ms
        )

        return True, "Action permitted within safety constraints"

    def _validate_blowdown(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate blowdown rate actions."""
        violations: List[ConstraintViolation] = []
        value = action.target_value

        # Check maximum blowdown rate
        if value > self._constraints.max_blowdown_rate_pct:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.BLOWDOWN_RATE,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=value,
                limit_value=self._constraints.max_blowdown_rate_pct,
                message=(
                    f"Blowdown rate {value:.1f}% exceeds maximum "
                    f"{self._constraints.max_blowdown_rate_pct:.1f}%"
                ),
            ))

        # Check minimum blowdown rate
        if value < self._constraints.min_blowdown_rate_pct:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.BLOWDOWN_RATE,
                severity=ViolationSeverity.WARNING,
                action=action,
                proposed_value=value,
                limit_value=self._constraints.min_blowdown_rate_pct,
                message=(
                    f"Blowdown rate {value:.1f}% below minimum "
                    f"{self._constraints.min_blowdown_rate_pct:.1f}%"
                ),
            ))

        return violations

    def _validate_dosing(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate chemical dosing actions."""
        violations: List[ConstraintViolation] = []
        value = action.target_value

        # Check maximum dosing rate
        if value > self._constraints.max_dosing_rate_pct:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.DOSING_RATE,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=value,
                limit_value=self._constraints.max_dosing_rate_pct,
                message=(
                    f"Dosing rate {value:.1f}% exceeds maximum "
                    f"{self._constraints.max_dosing_rate_pct:.1f}%"
                ),
            ))

        # Check for negative dosing
        if value < 0:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.DOSING_RATE,
                severity=ViolationSeverity.EMERGENCY,
                action=action,
                proposed_value=value,
                limit_value=0,
                message="Negative dosing rate not allowed",
            ))

        return violations

    def _validate_conductivity_setpoint(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate conductivity setpoint actions."""
        violations: List[ConstraintViolation] = []
        value = action.target_value

        # Check drum conductivity
        if "FEEDWATER" in action.target_tag.upper():
            limit = self._constraints.feedwater_max_conductivity_us_cm
        else:
            limit = self._constraints.max_conductivity_us_cm

        if value > limit:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.CONDUCTIVITY,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=value,
                limit_value=limit,
                message=(
                    f"Conductivity setpoint {value:.1f} uS/cm exceeds "
                    f"limit {limit:.1f} uS/cm"
                ),
            ))

        return violations

    def _validate_silica_setpoint(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate silica setpoint actions."""
        violations: List[ConstraintViolation] = []
        value = action.target_value

        # Check silica limit (critical for turbine blade deposits)
        if "FEEDWATER" in action.target_tag.upper():
            limit = self._constraints.feedwater_max_silica_ppb
        else:
            limit = self._constraints.max_silica_ppb

        if value > limit:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.SILICA,
                severity=ViolationSeverity.EMERGENCY,  # Turbine damage risk
                action=action,
                proposed_value=value,
                limit_value=limit,
                message=(
                    f"Silica setpoint {value:.1f} ppb exceeds "
                    f"limit {limit:.1f} ppb - TURBINE DAMAGE RISK"
                ),
            ))

        return violations

    def _validate_ph_setpoint(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate pH setpoint actions."""
        violations: List[ConstraintViolation] = []
        value = action.target_value

        # Check pH minimum
        if value < self._constraints.ph_min:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.PH,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=value,
                limit_value=self._constraints.ph_min,
                message=(
                    f"pH setpoint {value:.2f} below minimum "
                    f"{self._constraints.ph_min:.2f} - CORROSION RISK"
                ),
            ))

        # Check pH maximum
        if value > self._constraints.ph_max:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.PH,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=value,
                limit_value=self._constraints.ph_max,
                message=(
                    f"pH setpoint {value:.2f} exceeds maximum "
                    f"{self._constraints.ph_max:.2f} - CAUSTIC DAMAGE RISK"
                ),
            ))

        return violations

    def _validate_phosphate_setpoint(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate phosphate setpoint actions."""
        violations: List[ConstraintViolation] = []
        value = action.target_value

        if value < self._constraints.phosphate_min_ppm:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.PHOSPHATE,
                severity=ViolationSeverity.WARNING,
                action=action,
                proposed_value=value,
                limit_value=self._constraints.phosphate_min_ppm,
                message=(
                    f"Phosphate setpoint {value:.1f} ppm below minimum "
                    f"{self._constraints.phosphate_min_ppm:.1f} ppm"
                ),
            ))

        if value > self._constraints.phosphate_max_ppm:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.PHOSPHATE,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=value,
                limit_value=self._constraints.phosphate_max_ppm,
                message=(
                    f"Phosphate setpoint {value:.1f} ppm exceeds maximum "
                    f"{self._constraints.phosphate_max_ppm:.1f} ppm"
                ),
            ))

        return violations

    def _validate_generic(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate generic actions."""
        violations: List[ConstraintViolation] = []

        # Check for obviously invalid values
        if action.target_value < 0:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.OEM_LIMIT,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=action.target_value,
                limit_value=0,
                message=f"Negative value {action.target_value} not allowed",
            ))

        return violations

    def _validate_rate_of_change(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate rate of change limits."""
        violations: List[ConstraintViolation] = []

        with self._values_lock:
            last_entry = self._last_values.get(action.target_tag)

        if last_entry is None:
            return violations

        last_value, last_time = last_entry
        elapsed_minutes = (
            datetime.now(timezone.utc) - last_time
        ).total_seconds() / 60.0

        if elapsed_minutes <= 0:
            return violations

        rate = abs(action.target_value - last_value) / elapsed_minutes

        # Determine rate limit based on action type
        if "BLOWDOWN" in action.action_type.upper():
            max_rate = self._constraints.max_blowdown_delta_per_min
        elif "DOSING" in action.action_type.upper():
            max_rate = self._constraints.max_dosing_delta_per_min
        else:
            max_rate = 10.0  # Default rate limit

        if rate > max_rate:
            violations.append(ConstraintViolation(
                constraint_type=ConstraintType.BLOWDOWN_RATE,
                severity=ViolationSeverity.CRITICAL,
                action=action,
                proposed_value=rate,
                limit_value=max_rate,
                current_value=last_value,
                message=(
                    f"Rate of change {rate:.2f}/min exceeds maximum "
                    f"{max_rate:.2f}/min"
                ),
            ))

        return violations

    def _validate_oem_limits(
        self,
        action: ProposedAction
    ) -> List[ConstraintViolation]:
        """Validate OEM equipment limits."""
        violations: List[ConstraintViolation] = []

        oem_limit = self._constraints.oem_limits.get(action.target_tag)
        if oem_limit is not None:
            if action.target_value > oem_limit:
                violations.append(ConstraintViolation(
                    constraint_type=ConstraintType.OEM_LIMIT,
                    severity=ViolationSeverity.EMERGENCY,
                    action=action,
                    proposed_value=action.target_value,
                    limit_value=oem_limit,
                    message=(
                        f"Value {action.target_value} exceeds OEM limit "
                        f"{oem_limit} for {action.target_tag}"
                    ),
                ))

        return violations

    def _update_rate_tracking(self, action: ProposedAction) -> None:
        """Update rate tracking for permitted action."""
        with self._values_lock:
            self._last_values[action.target_tag] = (
                action.target_value,
                datetime.now(timezone.utc)
            )

    def fail_safe_state(self, reason: str = "Manual trigger") -> FailSafeState:
        """
        Revert to fail-safe baseline operation.

        This method immediately disables optimization and chemical dosing,
        sets blowdown to a safe baseline, and requires manual control.

        Args:
            reason: Reason for entering fail-safe

        Returns:
            Current FailSafeState
        """
        return self._trigger_fail_safe(reason)

    def _trigger_fail_safe(self, reason: str) -> FailSafeState:
        """Internal method to trigger fail-safe state."""
        with self._fail_safe_lock:
            self._fail_safe.enter_fail_safe(reason)

            with self._stats_lock:
                self._stats["fail_safe_triggered"] += 1

            # Record audit
            self._record_audit(
                "FAIL_SAFE_TRIGGERED",
                {
                    "reason": reason,
                    "state": self._fail_safe.model_dump(),
                }
            )

            logger.critical(
                "FAIL-SAFE TRIGGERED: %s - System reverting to baseline operation",
                reason
            )

            return self._fail_safe

    def exit_fail_safe(self, authorized_by: str) -> bool:
        """
        Exit fail-safe mode and return to normal operation.

        Args:
            authorized_by: Person authorizing exit from fail-safe

        Returns:
            True if successfully exited
        """
        with self._fail_safe_lock:
            if self._fail_safe.mode == FailSafeMode.NORMAL:
                return True

            self._fail_safe.mode = FailSafeMode.NORMAL
            self._fail_safe.optimization_enabled = True
            self._fail_safe.chemical_dosing_enabled = True
            self._fail_safe.manual_control_only = False

            self._record_audit(
                "FAIL_SAFE_EXITED",
                {
                    "authorized_by": authorized_by,
                    "previous_reason": self._fail_safe.reason,
                }
            )

            logger.info(
                "Fail-safe exited by %s - Normal operation resumed",
                authorized_by
            )

            return True

    def _on_watchdog_expire(self) -> None:
        """Handle watchdog expiration."""
        self._trigger_fail_safe("Watchdog heartbeat expired")

    def _record_audit(
        self,
        event_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Record an audit event."""
        with self._audit_lock:
            record = {
                "record_id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "details": details,
            }

            # Calculate provenance hash
            hash_str = f"{record['timestamp']}|{event_type}|{str(details)}"
            record["provenance_hash"] = hashlib.sha256(
                hash_str.encode()
            ).hexdigest()[:16]

            self._audit_records.append(record)

    def _record_permitted_action(self, action: ProposedAction) -> None:
        """Record a permitted action."""
        self._record_audit(
            "ACTION_PERMITTED",
            {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "target_tag": action.target_tag,
                "target_value": action.target_value,
            }
        )

    def _record_blocked_action(
        self,
        action: ProposedAction,
        reason: str
    ) -> None:
        """Record a blocked action."""
        self._record_audit(
            "ACTION_BLOCKED",
            {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "target_tag": action.target_tag,
                "target_value": action.target_value,
                "reason": reason,
            }
        )

    def get_constraints(self) -> WaterguardConstraints:
        """Get current constraints."""
        return self._constraints

    def update_constraints(
        self,
        constraints: WaterguardConstraints,
        authorized_by: str
    ) -> None:
        """
        Update constraints with authorization.

        Args:
            constraints: New constraints
            authorized_by: Person authorizing change
        """
        old_constraints = self._constraints
        self._constraints = constraints

        self._record_audit(
            "CONSTRAINTS_UPDATED",
            {
                "authorized_by": authorized_by,
                "changes": {
                    "old": old_constraints.model_dump(),
                    "new": constraints.model_dump(),
                }
            }
        )

        logger.info("Constraints updated by %s", authorized_by)

    def get_fail_safe_state(self) -> FailSafeState:
        """Get current fail-safe state."""
        with self._fail_safe_lock:
            return self._fail_safe

    def get_watchdog_status(self) -> HeartbeatStatus:
        """Get watchdog status."""
        return self._watchdog.get_status()

    def get_statistics(self) -> Dict[str, int]:
        """Get engine statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def get_violations(
        self,
        limit: int = 100
    ) -> List[ConstraintViolation]:
        """Get recent violations."""
        with self._violations_lock:
            return list(reversed(self._violations[-limit:]))

    def get_audit_records(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit records."""
        with self._audit_lock:
            return list(reversed(self._audit_records[-limit:]))


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ConstraintType",
    "ViolationSeverity",
    "HeartbeatStatus",
    "FailSafeMode",
    # Models
    "WaterguardConstraints",
    "ProposedAction",
    "ConstraintViolation",
    "FailSafeState",
    # Classes
    "WatchdogTimer",
    "WaterguardBoundaryEngine",
]
