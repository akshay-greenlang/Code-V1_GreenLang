"""
SIS Independence Validator for GL-001 ThermalCommand

This module ensures GL-001 maintains complete independence from
Safety Instrumented Systems (SIS). The validator enforces:
- GL-001 NEVER disables or modifies SIS
- SIS permissive read-only checks
- Independence verification
- SIS state monitoring

CRITICAL: GL-001 operates as an advisory/optimization system only.
It must NEVER interfere with safety-critical SIS operations.

Example:
    >>> from sis_validator import SISIndependenceValidator
    >>> validator = SISIndependenceValidator()
    >>> validator.verify_sis_independence(action)
    >>> if not validator.is_sis_tag(tag_id):
    ...     proceed_with_action()
"""

from datetime import datetime, timedelta
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import logging
import threading

from pydantic import BaseModel, Field

from .safety_schemas import (
    BoundaryViolation,
    ViolationType,
    ViolationSeverity,
    SafetyLevel,
    SISState,
    PolicyType,
    SafetyAuditRecord,
)

logger = logging.getLogger(__name__)


class SISViolationType(str, Enum):
    """Types of SIS independence violations."""

    WRITE_ATTEMPT = "write_attempt"
    DISABLE_ATTEMPT = "disable_attempt"
    BYPASS_ATTEMPT = "bypass_attempt"
    CONFIG_CHANGE_ATTEMPT = "config_change_attempt"
    TRIP_OVERRIDE_ATTEMPT = "trip_override_attempt"


class SISValidationResult(BaseModel):
    """Result of SIS validation check."""

    is_valid: bool = Field(..., description="Whether validation passed")
    violation_type: Optional[SISViolationType] = Field(
        None,
        description="Type of violation if invalid"
    )
    message: str = Field(default="", description="Validation message")
    sis_id: Optional[str] = Field(None, description="SIS ID if relevant")
    tag_id: Optional[str] = Field(None, description="Tag ID if relevant")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Validation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")

    def compute_hash(self) -> str:
        """Compute provenance hash."""
        hash_input = (
            f"{self.timestamp}|{self.is_valid}|"
            f"{self.violation_type}|{self.tag_id}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()


# =============================================================================
# SIS TAG PATTERNS - NEVER WRITABLE BY GL-001
# =============================================================================

# Safety Instrumented System tag patterns
SIS_TAG_PATTERNS: Set[str] = {
    "SIS-*",       # All SIS tags
    "ESD-*",       # Emergency Shutdown
    "PSV-*",       # Pressure Safety Valves
    "TSV-*",       # Temperature Safety Valves
    "FSV-*",       # Flow Safety Valves
    "LSV-*",       # Level Safety Valves
    "XV-ESD-*",    # ESD valves
    "XV-TRIP-*",   # Trip valves
    "XV-SIS-*",    # SIS valves
    "TRIP-*",      # Trip initiators
    "SHUTDOWN-*",  # Shutdown commands
    "SIF-*",       # Safety Instrumented Functions
    "IPL-*",       # Independent Protection Layers
    "HIPPS-*",     # High Integrity Pressure Protection
    "BPCS-SIF-*",  # BPCS Safety Interface
}

# SIS configuration tags - NEVER readable/writable
SIS_CONFIG_PATTERNS: Set[str] = {
    "SIS-CFG-*",
    "SIS-SETPOINT-*",
    "SIS-BYPASS-*",
    "SIS-INHIBIT-*",
    "SIS-OVERRIDE-*",
    "ESD-CFG-*",
    "TRIP-CFG-*",
}


class SISIndependenceValidator:
    """
    Validator ensuring GL-001 independence from SIS.

    CRITICAL SAFETY REQUIREMENT:
    GL-001 must NEVER:
    - Write to any SIS tag
    - Disable any SIS function
    - Modify SIS setpoints
    - Bypass SIS interlocks
    - Override trip conditions

    GL-001 MAY ONLY:
    - Read SIS status (permissive/trip state)
    - Adjust non-SIS setpoints to avoid SIS trips
    - Report SIS status for monitoring

    Attributes:
        sis_states: Current SIS states (read-only)
        violation_callback: Callback for violation notifications
        audit_records: Immutable audit trail

    Example:
        >>> validator = SISIndependenceValidator()
        >>> result = validator.validate_tag_access("SIS-101", "write")
        >>> assert not result.is_valid  # SIS tags are never writable
    """

    def __init__(
        self,
        violation_callback: Optional[Callable[[BoundaryViolation], None]] = None,
        sis_state_provider: Optional[Callable[[str], Optional[SISState]]] = None,
    ) -> None:
        """
        Initialize SIS Independence Validator.

        Args:
            violation_callback: Callback for violation notifications
            sis_state_provider: Function to get SIS states
        """
        self._violation_callback = violation_callback
        self._sis_state_provider = sis_state_provider or self._default_sis_provider

        # SIS state cache (read-only)
        self._sis_states: Dict[str, SISState] = {}
        self._sis_states_lock = threading.RLock()

        # Audit trail
        self._audit_records: List[SafetyAuditRecord] = []
        self._audit_lock = threading.Lock()

        # Statistics
        self._stats = {
            "validations_total": 0,
            "validations_passed": 0,
            "validations_failed": 0,
            "write_attempts_blocked": 0,
            "sis_states_read": 0,
        }
        self._stats_lock = threading.Lock()

        # Last validation timestamps for rate limiting
        self._last_validation: Dict[str, datetime] = {}

        logger.info("SISIndependenceValidator initialized")

    def _default_sis_provider(self, sis_id: str) -> Optional[SISState]:
        """Default SIS state provider using cache."""
        with self._sis_states_lock:
            return self._sis_states.get(sis_id)

    def update_sis_state(self, sis_state: SISState) -> None:
        """
        Update cached SIS state (READ-ONLY operation from external source).

        Args:
            sis_state: New SIS state from external monitoring

        Note:
            This only updates the read cache. GL-001 NEVER writes to SIS.
        """
        with self._sis_states_lock:
            self._sis_states[sis_state.sis_id] = sis_state
            logger.debug(f"Updated SIS state cache: {sis_state.sis_id}")

        with self._stats_lock:
            self._stats["sis_states_read"] += 1

    def is_sis_tag(self, tag_id: str) -> bool:
        """
        Check if a tag is a SIS tag.

        Args:
            tag_id: Tag identifier

        Returns:
            True if tag is a SIS tag (NEVER writable)
        """
        for pattern in SIS_TAG_PATTERNS:
            if fnmatch(tag_id, pattern):
                return True
        return False

    def is_sis_config_tag(self, tag_id: str) -> bool:
        """
        Check if a tag is a SIS configuration tag.

        Args:
            tag_id: Tag identifier

        Returns:
            True if tag is a SIS config tag (NEVER accessible)
        """
        for pattern in SIS_CONFIG_PATTERNS:
            if fnmatch(tag_id, pattern):
                return True
        return False

    def validate_tag_access(
        self,
        tag_id: str,
        access_type: str = "write",
    ) -> SISValidationResult:
        """
        Validate access to a tag from SIS independence perspective.

        Args:
            tag_id: Tag identifier
            access_type: Type of access ("read" or "write")

        Returns:
            SISValidationResult indicating if access is allowed
        """
        with self._stats_lock:
            self._stats["validations_total"] += 1

        # SIS config tags are NEVER accessible
        if self.is_sis_config_tag(tag_id):
            result = SISValidationResult(
                is_valid=False,
                violation_type=SISViolationType.CONFIG_CHANGE_ATTEMPT,
                message=f"SIS configuration tag {tag_id} is NEVER accessible by GL-001",
                tag_id=tag_id,
            )
            self._handle_violation(result)
            return result

        # SIS tags are NEVER writable
        if self.is_sis_tag(tag_id):
            if access_type == "write":
                result = SISValidationResult(
                    is_valid=False,
                    violation_type=SISViolationType.WRITE_ATTEMPT,
                    message=f"CRITICAL: GL-001 attempted to WRITE to SIS tag {tag_id}",
                    tag_id=tag_id,
                )
                self._handle_violation(result)
                return result
            else:
                # Read access to SIS status is allowed
                with self._stats_lock:
                    self._stats["validations_passed"] += 1
                return SISValidationResult(
                    is_valid=True,
                    message=f"Read access to SIS tag {tag_id} is permitted",
                    tag_id=tag_id,
                )

        # Non-SIS tag - access permitted
        with self._stats_lock:
            self._stats["validations_passed"] += 1

        return SISValidationResult(
            is_valid=True,
            message=f"Tag {tag_id} is not a SIS tag - access permitted",
            tag_id=tag_id,
        )

    def validate_action(
        self,
        action_type: str,
        target: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SISValidationResult:
        """
        Validate an action for SIS independence.

        Args:
            action_type: Type of action (write, disable, bypass, etc.)
            target: Target of action (tag_id, sis_id, etc.)
            parameters: Action parameters

        Returns:
            SISValidationResult indicating if action is allowed
        """
        with self._stats_lock:
            self._stats["validations_total"] += 1

        parameters = parameters or {}

        # Check for SIS disable attempt
        if action_type in ("disable", "stop", "shutdown"):
            if self.is_sis_tag(target) or target.startswith("SIS"):
                result = SISValidationResult(
                    is_valid=False,
                    violation_type=SISViolationType.DISABLE_ATTEMPT,
                    message=f"CRITICAL: GL-001 attempted to DISABLE SIS: {target}",
                    sis_id=target if target.startswith("SIS") else None,
                    tag_id=target,
                )
                self._handle_violation(result)
                return result

        # Check for SIS bypass attempt
        if action_type in ("bypass", "inhibit", "suppress"):
            result = SISValidationResult(
                is_valid=False,
                violation_type=SISViolationType.BYPASS_ATTEMPT,
                message=f"CRITICAL: GL-001 attempted to BYPASS SIS: {target}",
                sis_id=target if target.startswith("SIS") else None,
                tag_id=target,
            )
            self._handle_violation(result)
            return result

        # Check for trip override attempt
        if action_type in ("override", "reset_trip", "clear_trip"):
            if self.is_sis_tag(target):
                result = SISValidationResult(
                    is_valid=False,
                    violation_type=SISViolationType.TRIP_OVERRIDE_ATTEMPT,
                    message=f"CRITICAL: GL-001 attempted to OVERRIDE SIS trip: {target}",
                    sis_id=target if target.startswith("SIS") else None,
                    tag_id=target,
                )
                self._handle_violation(result)
                return result

        # Check for write to SIS tag
        if action_type == "write":
            return self.validate_tag_access(target, "write")

        # Non-SIS action - permitted
        with self._stats_lock:
            self._stats["validations_passed"] += 1

        return SISValidationResult(
            is_valid=True,
            message=f"Action {action_type} on {target} is permitted",
            tag_id=target,
        )

    def verify_sis_independence(
        self,
        proposed_actions: List[Dict[str, Any]]
    ) -> Tuple[bool, List[SISValidationResult]]:
        """
        Verify a batch of proposed actions maintains SIS independence.

        Args:
            proposed_actions: List of proposed actions with keys:
                - action_type: Type of action
                - target: Target tag/SIS
                - parameters: Optional parameters

        Returns:
            Tuple of (all_valid, list of validation results)
        """
        results: List[SISValidationResult] = []
        all_valid = True

        for action in proposed_actions:
            result = self.validate_action(
                action_type=action.get("action_type", "write"),
                target=action.get("target", ""),
                parameters=action.get("parameters"),
            )
            results.append(result)
            if not result.is_valid:
                all_valid = False

        return all_valid, results

    def get_sis_state(self, sis_id: str) -> Optional[SISState]:
        """
        Get current SIS state (READ-ONLY).

        Args:
            sis_id: SIS identifier

        Returns:
            SISState if available, None otherwise
        """
        state = self._sis_state_provider(sis_id)
        if state:
            with self._stats_lock:
                self._stats["sis_states_read"] += 1
        return state

    def get_all_sis_states(self) -> List[SISState]:
        """
        Get all cached SIS states (READ-ONLY).

        Returns:
            List of all SIS states
        """
        with self._sis_states_lock:
            return list(self._sis_states.values())

    def is_any_sis_tripped(self) -> bool:
        """
        Check if any SIS is in tripped state.

        Returns:
            True if any SIS is tripped
        """
        with self._sis_states_lock:
            return any(sis.trip_status for sis in self._sis_states.values())

    def is_any_sis_bypassed(self) -> bool:
        """
        Check if any SIS bypass is active.

        Returns:
            True if any bypass is active

        Note:
            Bypasses are set by authorized personnel, not GL-001.
            GL-001 should be more conservative when bypasses are active.
        """
        with self._sis_states_lock:
            return any(sis.bypass_active for sis in self._sis_states.values())

    def are_all_sis_healthy(self) -> bool:
        """
        Check if all SIS systems are healthy.

        Returns:
            True if all SIS systems are healthy
        """
        with self._sis_states_lock:
            if not self._sis_states:
                logger.warning("No SIS states available - assuming healthy")
                return True
            return all(sis.is_healthy for sis in self._sis_states.values())

    def check_sis_heartbeats(
        self,
        max_age_seconds: float = 60.0
    ) -> Dict[str, bool]:
        """
        Check SIS heartbeat freshness.

        Args:
            max_age_seconds: Maximum allowed age for heartbeat

        Returns:
            Dict of sis_id -> is_fresh
        """
        results: Dict[str, bool] = {}
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=max_age_seconds)

        with self._sis_states_lock:
            for sis_id, state in self._sis_states.items():
                results[sis_id] = state.last_heartbeat > cutoff
                if not results[sis_id]:
                    logger.warning(
                        f"SIS {sis_id} heartbeat stale: "
                        f"last={state.last_heartbeat}, cutoff={cutoff}"
                    )

        return results

    def _handle_violation(self, result: SISValidationResult) -> None:
        """
        Handle a SIS independence violation.

        Args:
            result: Validation result with violation
        """
        with self._stats_lock:
            self._stats["validations_failed"] += 1
            if result.violation_type == SISViolationType.WRITE_ATTEMPT:
                self._stats["write_attempts_blocked"] += 1

        logger.critical(
            f"SIS INDEPENDENCE VIOLATION: type={result.violation_type}, "
            f"tag={result.tag_id}, message={result.message}"
        )

        # Create audit record
        self._create_audit_record(result)

        # Create boundary violation for violation handler
        violation = BoundaryViolation(
            policy_id="SIS_INDEPENDENCE",
            policy_type=PolicyType.WHITELIST,
            tag_id=result.tag_id or "",
            violation_type=ViolationType.SIS_VIOLATION,
            severity=ViolationSeverity.EMERGENCY,
            message=result.message,
            context={
                "sis_violation_type": result.violation_type,
                "sis_id": result.sis_id,
            },
        )

        # Notify callback
        if self._violation_callback:
            try:
                self._violation_callback(violation)
            except Exception as e:
                logger.error(f"Violation callback failed: {e}")

    def _create_audit_record(self, result: SISValidationResult) -> None:
        """
        Create immutable audit record for SIS violation.

        Args:
            result: Validation result
        """
        with self._audit_lock:
            previous_hash = ""
            if self._audit_records:
                previous_hash = self._audit_records[-1].provenance_hash

            record = SafetyAuditRecord(
                event_type=f"SIS_VIOLATION_{result.violation_type}",
                action_taken="BLOCKED",
                operator_notified=True,
                escalation_level=3,  # Highest escalation for SIS violations
                previous_hash=previous_hash,
            )

            self._audit_records.append(record)
            logger.info(f"Created SIS audit record: {record.record_id}")

    def get_audit_records(
        self,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[SafetyAuditRecord]:
        """
        Get SIS audit records.

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

    def get_statistics(self) -> Dict[str, int]:
        """
        Get validation statistics.

        Returns:
            Dict of statistics
        """
        with self._stats_lock:
            return dict(self._stats)

    def get_sis_tag_patterns(self) -> Set[str]:
        """
        Get all SIS tag patterns.

        Returns:
            Set of SIS tag patterns

        Note:
            These patterns define tags that GL-001 must NEVER write to.
        """
        return SIS_TAG_PATTERNS.copy()

    def validate_optimization_targets(
        self,
        targets: Dict[str, float]
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Validate optimization targets don't include SIS tags.

        Args:
            targets: Dict of tag_id -> target_value

        Returns:
            Tuple of (valid_targets, removed_tags)

        Note:
            This is a pre-optimization check to ensure no SIS tags
            are included in optimization targets.
        """
        valid_targets: Dict[str, float] = {}
        removed_tags: List[str] = []

        for tag_id, value in targets.items():
            if self.is_sis_tag(tag_id):
                logger.warning(f"Removed SIS tag from optimization targets: {tag_id}")
                removed_tags.append(tag_id)
            else:
                valid_targets[tag_id] = value

        return valid_targets, removed_tags


# Type hint for Tuple
from typing import Tuple


class SISMonitor:
    """
    Monitor for SIS state changes.

    Provides continuous monitoring of SIS states and alerts
    when conditions change that may affect GL-001 operations.

    Example:
        >>> monitor = SISMonitor(validator)
        >>> monitor.register_callback(on_sis_trip)
        >>> monitor.start()
    """

    def __init__(
        self,
        validator: SISIndependenceValidator,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """
        Initialize SIS Monitor.

        Args:
            validator: SIS Independence Validator
            poll_interval_seconds: Polling interval
        """
        self._validator = validator
        self._poll_interval = poll_interval_seconds
        self._callbacks: List[Callable[[str, SISState], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._previous_states: Dict[str, SISState] = {}

        logger.info("SISMonitor initialized")

    def register_callback(
        self,
        callback: Callable[[str, SISState], None]
    ) -> None:
        """
        Register callback for SIS state changes.

        Args:
            callback: Function called with (event_type, sis_state)
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("SIS monitoring started")

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("SIS monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        import time

        while self._running:
            try:
                self._check_sis_states()
            except Exception as e:
                logger.error(f"SIS monitor error: {e}")

            time.sleep(self._poll_interval)

    def _check_sis_states(self) -> None:
        """Check for SIS state changes."""
        current_states = self._validator.get_all_sis_states()

        for state in current_states:
            previous = self._previous_states.get(state.sis_id)

            if previous:
                # Check for trip
                if state.trip_status and not previous.trip_status:
                    self._notify("SIS_TRIP", state)

                # Check for recovery
                if not state.trip_status and previous.trip_status:
                    self._notify("SIS_RECOVERY", state)

                # Check for bypass change
                if state.bypass_active != previous.bypass_active:
                    event = "SIS_BYPASS_ACTIVE" if state.bypass_active else "SIS_BYPASS_CLEARED"
                    self._notify(event, state)

                # Check for health change
                if not state.is_healthy and previous.is_healthy:
                    self._notify("SIS_UNHEALTHY", state)

            self._previous_states[state.sis_id] = state

    def _notify(self, event_type: str, state: SISState) -> None:
        """Notify callbacks of SIS event."""
        logger.info(f"SIS event: {event_type} for {state.sis_id}")

        for callback in self._callbacks:
            try:
                callback(event_type, state)
            except Exception as e:
                logger.error(f"SIS callback error: {e}")
