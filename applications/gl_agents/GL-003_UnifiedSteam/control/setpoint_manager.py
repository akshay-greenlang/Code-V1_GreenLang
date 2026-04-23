"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Setpoint Manager

This module implements centralized setpoint management for the steam system
with authorization, validation, rollback, and complete audit trail.

Control Architecture:
    - Centralized setpoint registration and tracking
    - Multi-level authorization enforcement
    - Comprehensive validation against constraints
    - Automatic rollback capability
    - Complete audit trail for regulatory compliance

Reference Standards:
    - ISA-18.2 Management of Alarm Systems
    - ISA-62443 Industrial Automation Security
    - IEC 61511 Functional Safety

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SetpointCategory(str, Enum):
    """Setpoint category enumeration."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    QUALITY = "quality"
    VALVE_POSITION = "valve_position"
    LEVEL = "level"


class SetpointSource(str, Enum):
    """Setpoint source enumeration."""
    OPERATOR = "operator"
    OPTIMIZER = "optimizer"
    CASCADE = "cascade"
    SAFETY_OVERRIDE = "safety_override"
    SCHEDULE = "schedule"
    EXTERNAL_SYSTEM = "external_system"


class AuthorizationLevel(str, Enum):
    """Authorization level enumeration."""
    OPERATOR = "operator"
    SUPERVISOR = "supervisor"
    ENGINEER = "engineer"
    SAFETY_SYSTEM = "safety_system"


class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING_REVIEW = "pending_review"


class ApplicationStatus(str, Enum):
    """Application status enumeration."""
    APPLIED = "applied"
    REJECTED = "rejected"
    PENDING = "pending"
    ROLLED_BACK = "rolled_back"


# =============================================================================
# DATA MODELS
# =============================================================================

class SetpointConstraint(BaseModel):
    """Constraints for a setpoint."""

    min_value: float = Field(..., description="Minimum allowed value")
    max_value: float = Field(..., description="Maximum allowed value")
    max_rate_per_min: float = Field(
        default=float('inf'),
        ge=0,
        description="Maximum rate of change per minute"
    )
    deadband: float = Field(
        default=0.0,
        ge=0,
        description="Deadband for change detection"
    )
    required_authorization: AuthorizationLevel = Field(
        default=AuthorizationLevel.OPERATOR,
        description="Required authorization level"
    )
    safety_critical: bool = Field(
        default=False,
        description="Is this a safety-critical setpoint"
    )


class SetpointDefinition(BaseModel):
    """Definition of a managed setpoint."""

    tag: str = Field(..., description="Setpoint tag identifier")
    description: str = Field(..., description="Human-readable description")
    category: SetpointCategory = Field(..., description="Setpoint category")
    unit: str = Field(..., description="Engineering unit")
    constraints: SetpointConstraint = Field(..., description="Setpoint constraints")
    equipment_id: str = Field(..., description="Associated equipment ID")
    default_value: float = Field(..., description="Default/safe value")
    current_value: Optional[float] = Field(
        None,
        description="Current setpoint value"
    )


class SetpointRegistration(BaseModel):
    """Result of setpoint registration."""

    registration_id: str = Field(..., description="Registration ID")
    tag: str = Field(..., description="Setpoint tag")
    value: float = Field(..., description="Registered value")
    source: SetpointSource = Field(..., description="Value source")
    registration_time: datetime = Field(
        default_factory=datetime.now,
        description="Registration timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Expiration time for temporary setpoints"
    )
    constraints_validated: bool = Field(
        default=False,
        description="Constraints were validated"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ValidationResult(BaseModel):
    """Result of setpoint validation."""

    validation_id: str = Field(..., description="Validation ID")
    tag: str = Field(..., description="Setpoint tag")
    proposed_value: float = Field(..., description="Proposed value")
    status: ValidationStatus = Field(..., description="Validation status")
    constraint_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Individual constraint check results"
    )
    violations: List[str] = Field(
        default_factory=list,
        description="Constraint violations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    requires_authorization: Optional[AuthorizationLevel] = Field(
        None,
        description="Authorization level required"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Validation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class Authorization(BaseModel):
    """Setpoint change authorization."""

    authorization_id: str = Field(..., description="Authorization ID")
    authorizer_id: str = Field(..., description="ID of authorizing entity")
    authorization_level: AuthorizationLevel = Field(
        ...,
        description="Authorization level"
    )
    tag: str = Field(..., description="Setpoint tag")
    proposed_value: float = Field(..., description="Authorized value")
    reason: str = Field(default="", description="Authorization reason")
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="Authorization start time"
    )
    valid_until: datetime = Field(
        default_factory=lambda: datetime.now() + timedelta(hours=8),
        description="Authorization expiration"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ApplicationResult(BaseModel):
    """Result of setpoint application."""

    application_id: str = Field(..., description="Application ID")
    tag: str = Field(..., description="Setpoint tag")
    previous_value: float = Field(..., description="Previous value")
    applied_value: float = Field(..., description="Applied value")
    status: ApplicationStatus = Field(..., description="Application status")
    authorization_id: Optional[str] = Field(
        None,
        description="Authorization ID used"
    )
    applied_at: datetime = Field(
        default_factory=datetime.now,
        description="Application timestamp"
    )
    message: str = Field(default="", description="Status message")
    rollback_available: bool = Field(
        default=True,
        description="Rollback is available"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class RollbackResult(BaseModel):
    """Result of setpoint rollback."""

    rollback_id: str = Field(..., description="Rollback ID")
    tag: str = Field(..., description="Setpoint tag")
    rolled_back_value: float = Field(
        ...,
        description="Value that was rolled back"
    )
    restored_value: float = Field(..., description="Restored value")
    reason: str = Field(..., description="Rollback reason")
    success: bool = Field(..., description="Rollback was successful")
    rolled_back_at: datetime = Field(
        default_factory=datetime.now,
        description="Rollback timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class SetpointChange(BaseModel):
    """Record of a setpoint change for history."""

    change_id: str = Field(..., description="Change ID")
    tag: str = Field(..., description="Setpoint tag")
    previous_value: float = Field(..., description="Previous value")
    new_value: float = Field(..., description="New value")
    source: SetpointSource = Field(..., description="Change source")
    authorization_id: Optional[str] = Field(
        None,
        description="Authorization ID"
    )
    reason: str = Field(default="", description="Change reason")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Change timestamp"
    )
    rolled_back: bool = Field(
        default=False,
        description="Change was rolled back"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class AuditRecord(BaseModel):
    """Audit record for setpoint operations."""

    audit_id: str = Field(..., description="Audit record ID")
    operation: str = Field(..., description="Operation type")
    tag: str = Field(..., description="Setpoint tag")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation details"
    )
    user_id: Optional[str] = Field(None, description="User ID")
    source: str = Field(..., description="Operation source")
    success: bool = Field(..., description="Operation succeeded")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Audit timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# SETPOINT MANAGER
# =============================================================================

class SetpointManager:
    """
    Centralized setpoint management for steam system optimization.

    This manager provides centralized control over all setpoints in the
    steam system with authorization, validation, and complete audit trail.

    Features:
        - Centralized setpoint registration and tracking
        - Multi-level authorization enforcement
        - Comprehensive constraint validation
        - Automatic rollback capability
        - Complete audit trail for compliance

    Safety Features:
        - All changes validated against constraints
        - Safety-critical setpoints require elevated authorization
        - Rate limiting prevents rapid changes
        - Complete audit trail for regulatory compliance

    Attributes:
        _definitions: Registered setpoint definitions
        _values: Current setpoint values
        _history: Setpoint change history
        _audit_log: Audit records

    Example:
        >>> manager = SetpointManager()
        >>> manager.register_setpoint(
        ...     tag="DS001.TEMP.SP",
        ...     value=400.0,
        ...     source=SetpointSource.OPERATOR,
        ...     constraints=SetpointConstraint(min_value=350, max_value=450)
        ... )
        >>> result = manager.apply_setpoint("DS001.TEMP.SP", 420.0, authorization)
    """

    def __init__(self):
        """Initialize SetpointManager."""
        self._definitions: Dict[str, SetpointDefinition] = {}
        self._values: Dict[str, float] = {}
        self._history: Dict[str, List[SetpointChange]] = defaultdict(list)
        self._audit_log: List[AuditRecord] = []
        self._pending_authorizations: Dict[str, Authorization] = {}
        self._max_history_per_tag = 100
        self._max_audit_records = 10000

        logger.info("SetpointManager initialized")

    def register_setpoint(
        self,
        tag: str,
        value: float,
        source: SetpointSource,
        constraints: SetpointConstraint,
        category: SetpointCategory = SetpointCategory.TEMPERATURE,
        unit: str = "",
        equipment_id: str = "",
        description: str = ""
    ) -> SetpointRegistration:
        """
        Register a new setpoint or update existing.

        This method registers a setpoint with its constraints and initial value.
        All setpoints must be registered before they can be managed.

        Args:
            tag: Unique setpoint tag identifier
            value: Initial setpoint value
            source: Source of the setpoint value
            constraints: Setpoint constraints
            category: Setpoint category
            unit: Engineering unit
            equipment_id: Associated equipment ID
            description: Human-readable description

        Returns:
            SetpointRegistration: Registration result

        Raises:
            ValueError: If value violates constraints
        """
        start_time = datetime.now()

        # Validate initial value against constraints
        if value < constraints.min_value or value > constraints.max_value:
            raise ValueError(
                f"Initial value {value} outside constraints "
                f"[{constraints.min_value}, {constraints.max_value}]"
            )

        # Create definition
        definition = SetpointDefinition(
            tag=tag,
            description=description or f"Setpoint {tag}",
            category=category,
            unit=unit,
            constraints=constraints,
            equipment_id=equipment_id,
            default_value=value,
            current_value=value
        )

        # Store definition and value
        self._definitions[tag] = definition
        self._values[tag] = value

        # Generate registration ID
        registration_id = hashlib.sha256(
            f"REG_{tag}_{value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        registration = SetpointRegistration(
            registration_id=registration_id,
            tag=tag,
            value=value,
            source=source,
            registration_time=start_time,
            constraints_validated=True
        )

        # Calculate provenance hash
        registration.provenance_hash = hashlib.sha256(
            f"{registration_id}|{tag}|{value}|{source.value}".encode()
        ).hexdigest()

        # Create audit record
        self._add_audit_record(
            operation="register",
            tag=tag,
            details={
                "value": value,
                "source": source.value,
                "constraints": constraints.dict()
            },
            source=source.value,
            success=True
        )

        logger.info(
            f"Registered setpoint {tag}: value={value}, "
            f"constraints=[{constraints.min_value}, {constraints.max_value}]"
        )

        return registration

    def validate_setpoint(
        self,
        tag: str,
        proposed_value: float
    ) -> ValidationResult:
        """
        Validate a proposed setpoint value against constraints.

        This method performs comprehensive validation including:
        - Range constraints (min/max)
        - Rate of change constraints
        - Safety-critical flags
        - Authorization requirements

        Args:
            tag: Setpoint tag
            proposed_value: Proposed new value

        Returns:
            ValidationResult: Validation result with details

        Raises:
            KeyError: If tag is not registered
        """
        start_time = datetime.now()

        if tag not in self._definitions:
            raise KeyError(f"Setpoint tag '{tag}' not registered")

        definition = self._definitions[tag]
        constraints = definition.constraints
        current_value = self._values.get(tag, definition.default_value)

        violations = []
        warnings = []
        constraint_checks = {}

        # Check 1: Range constraints
        if proposed_value < constraints.min_value:
            violations.append(
                f"Value {proposed_value} below minimum {constraints.min_value}"
            )
            constraint_checks["min_value"] = False
        else:
            constraint_checks["min_value"] = True

        if proposed_value > constraints.max_value:
            violations.append(
                f"Value {proposed_value} above maximum {constraints.max_value}"
            )
            constraint_checks["max_value"] = False
        else:
            constraint_checks["max_value"] = True

        # Check 2: Rate of change (if there's recent history)
        history = self._history.get(tag, [])
        if history and constraints.max_rate_per_min < float('inf'):
            last_change = history[-1]
            time_delta = (start_time - last_change.timestamp).total_seconds() / 60.0
            if time_delta > 0:
                rate = abs(proposed_value - current_value) / time_delta
                if rate > constraints.max_rate_per_min:
                    violations.append(
                        f"Rate of change {rate:.2f}/min exceeds max "
                        f"{constraints.max_rate_per_min}/min"
                    )
                    constraint_checks["rate_of_change"] = False
                else:
                    constraint_checks["rate_of_change"] = True

        # Check 3: Deadband
        if abs(proposed_value - current_value) < constraints.deadband:
            warnings.append(
                f"Change of {abs(proposed_value - current_value):.4f} is within "
                f"deadband of {constraints.deadband}"
            )

        # Check 4: Safety-critical flag
        requires_authorization = constraints.required_authorization
        if constraints.safety_critical:
            requires_authorization = AuthorizationLevel.ENGINEER
            warnings.append("Safety-critical setpoint - engineer authorization required")

        # Determine status
        if violations:
            status = ValidationStatus.INVALID
        elif constraints.safety_critical:
            status = ValidationStatus.PENDING_REVIEW
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        # Generate validation ID
        validation_id = hashlib.sha256(
            f"VAL_{tag}_{proposed_value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = ValidationResult(
            validation_id=validation_id,
            tag=tag,
            proposed_value=proposed_value,
            status=status,
            constraint_checks=constraint_checks,
            violations=violations,
            warnings=warnings,
            requires_authorization=requires_authorization,
            timestamp=start_time
        )

        # Calculate provenance hash
        result.provenance_hash = hashlib.sha256(
            f"{validation_id}|{status.value}|{len(violations)}".encode()
        ).hexdigest()

        logger.info(
            f"Validated setpoint {tag}: proposed={proposed_value}, "
            f"status={status.value}, violations={len(violations)}"
        )

        return result

    def apply_setpoint(
        self,
        tag: str,
        value: float,
        authorization: Authorization
    ) -> ApplicationResult:
        """
        Apply a new setpoint value with authorization.

        This method applies a setpoint change after validating authorization
        and constraints. All changes are logged for audit trail.

        Args:
            tag: Setpoint tag
            value: New setpoint value
            authorization: Authorization for the change

        Returns:
            ApplicationResult: Application result

        Raises:
            KeyError: If tag is not registered
            ValueError: If authorization is invalid or expired
        """
        start_time = datetime.now()

        if tag not in self._definitions:
            raise KeyError(f"Setpoint tag '{tag}' not registered")

        # Validate authorization
        auth_valid, auth_message = self._validate_authorization(
            authorization, tag, value
        )
        if not auth_valid:
            logger.warning(f"Authorization rejected for {tag}: {auth_message}")

            # Create rejection result
            application_id = hashlib.sha256(
                f"APP_{tag}_{value}_{start_time.isoformat()}".encode()
            ).hexdigest()[:16]

            result = ApplicationResult(
                application_id=application_id,
                tag=tag,
                previous_value=self._values.get(tag, 0),
                applied_value=value,
                status=ApplicationStatus.REJECTED,
                authorization_id=authorization.authorization_id,
                message=auth_message,
                rollback_available=False
            )

            self._add_audit_record(
                operation="apply_rejected",
                tag=tag,
                details={"value": value, "reason": auth_message},
                source=authorization.authorizer_id,
                success=False
            )

            return result

        # Validate the setpoint value
        validation = self.validate_setpoint(tag, value)
        if validation.status == ValidationStatus.INVALID:
            application_id = hashlib.sha256(
                f"APP_{tag}_{value}_{start_time.isoformat()}".encode()
            ).hexdigest()[:16]

            result = ApplicationResult(
                application_id=application_id,
                tag=tag,
                previous_value=self._values.get(tag, 0),
                applied_value=value,
                status=ApplicationStatus.REJECTED,
                authorization_id=authorization.authorization_id,
                message=f"Validation failed: {validation.violations}",
                rollback_available=False
            )

            self._add_audit_record(
                operation="apply_validation_failed",
                tag=tag,
                details={"value": value, "violations": validation.violations},
                source=authorization.authorizer_id,
                success=False
            )

            return result

        # Apply the change
        previous_value = self._values.get(tag, self._definitions[tag].default_value)
        self._values[tag] = value
        self._definitions[tag].current_value = value

        # Record change in history
        change_id = hashlib.sha256(
            f"CHG_{tag}_{value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        change = SetpointChange(
            change_id=change_id,
            tag=tag,
            previous_value=previous_value,
            new_value=value,
            source=SetpointSource.OPERATOR,  # Derived from authorization
            authorization_id=authorization.authorization_id,
            reason=authorization.reason,
            timestamp=start_time
        )
        change.provenance_hash = hashlib.sha256(
            f"{change_id}|{previous_value}|{value}".encode()
        ).hexdigest()

        self._add_to_history(tag, change)

        # Create application result
        application_id = hashlib.sha256(
            f"APP_{tag}_{value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = ApplicationResult(
            application_id=application_id,
            tag=tag,
            previous_value=previous_value,
            applied_value=value,
            status=ApplicationStatus.APPLIED,
            authorization_id=authorization.authorization_id,
            message=f"Setpoint changed from {previous_value} to {value}",
            rollback_available=True
        )

        result.provenance_hash = hashlib.sha256(
            f"{application_id}|{previous_value}|{value}|APPLIED".encode()
        ).hexdigest()

        # Audit record
        self._add_audit_record(
            operation="apply",
            tag=tag,
            details={
                "previous_value": previous_value,
                "new_value": value,
                "authorization_id": authorization.authorization_id
            },
            source=authorization.authorizer_id,
            success=True
        )

        logger.info(
            f"Applied setpoint {tag}: {previous_value} -> {value} "
            f"(auth: {authorization.authorization_id})"
        )

        return result

    def rollback_setpoint(
        self,
        tag: str,
        reason: str,
        user_id: str = "system"
    ) -> RollbackResult:
        """
        Rollback setpoint to previous value.

        This method rolls back the most recent setpoint change and restores
        the previous value. Rollback is logged for audit trail.

        Args:
            tag: Setpoint tag
            reason: Reason for rollback
            user_id: ID of user/system initiating rollback

        Returns:
            RollbackResult: Rollback result

        Raises:
            KeyError: If tag is not registered
            ValueError: If no previous value to rollback to
        """
        start_time = datetime.now()

        if tag not in self._definitions:
            raise KeyError(f"Setpoint tag '{tag}' not registered")

        history = self._history.get(tag, [])

        # Find most recent non-rolled-back change
        rollback_change = None
        for change in reversed(history):
            if not change.rolled_back:
                rollback_change = change
                break

        if rollback_change is None:
            raise ValueError(f"No changes to rollback for tag '{tag}'")

        # Perform rollback
        current_value = self._values[tag]
        restored_value = rollback_change.previous_value

        self._values[tag] = restored_value
        self._definitions[tag].current_value = restored_value

        # Mark the change as rolled back
        rollback_change.rolled_back = True

        # Generate rollback ID
        rollback_id = hashlib.sha256(
            f"RB_{tag}_{current_value}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = RollbackResult(
            rollback_id=rollback_id,
            tag=tag,
            rolled_back_value=current_value,
            restored_value=restored_value,
            reason=reason,
            success=True,
            rolled_back_at=start_time
        )

        result.provenance_hash = hashlib.sha256(
            f"{rollback_id}|{current_value}|{restored_value}".encode()
        ).hexdigest()

        # Audit record
        self._add_audit_record(
            operation="rollback",
            tag=tag,
            details={
                "rolled_back_value": current_value,
                "restored_value": restored_value,
                "reason": reason,
                "original_change_id": rollback_change.change_id
            },
            user_id=user_id,
            source="rollback",
            success=True
        )

        logger.info(
            f"Rolled back setpoint {tag}: {current_value} -> {restored_value} "
            f"(reason: {reason})"
        )

        return result

    def get_setpoint_history(
        self,
        tag: str,
        time_window_minutes: int = 60
    ) -> List[SetpointChange]:
        """
        Get setpoint change history within time window.

        Args:
            tag: Setpoint tag
            time_window_minutes: Time window in minutes

        Returns:
            List of setpoint changes within the time window

        Raises:
            KeyError: If tag is not registered
        """
        if tag not in self._definitions:
            raise KeyError(f"Setpoint tag '{tag}' not registered")

        history = self._history.get(tag, [])
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        return [
            change for change in history
            if change.timestamp >= cutoff_time
        ]

    def get_current_value(self, tag: str) -> float:
        """
        Get current setpoint value.

        Args:
            tag: Setpoint tag

        Returns:
            Current setpoint value

        Raises:
            KeyError: If tag is not registered
        """
        if tag not in self._definitions:
            raise KeyError(f"Setpoint tag '{tag}' not registered")

        return self._values.get(tag, self._definitions[tag].default_value)

    def get_definition(self, tag: str) -> SetpointDefinition:
        """
        Get setpoint definition.

        Args:
            tag: Setpoint tag

        Returns:
            Setpoint definition

        Raises:
            KeyError: If tag is not registered
        """
        if tag not in self._definitions:
            raise KeyError(f"Setpoint tag '{tag}' not registered")

        return self._definitions[tag]

    def get_all_tags(self) -> List[str]:
        """Get all registered setpoint tags."""
        return list(self._definitions.keys())

    def get_audit_log(
        self,
        tag: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> List[AuditRecord]:
        """
        Get audit log entries.

        Args:
            tag: Filter by tag (None for all)
            time_window_minutes: Time window in minutes

        Returns:
            List of audit records
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        records = [
            rec for rec in self._audit_log
            if rec.timestamp >= cutoff_time
        ]

        if tag:
            records = [rec for rec in records if rec.tag == tag]

        return records

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _validate_authorization(
        self,
        authorization: Authorization,
        tag: str,
        value: float
    ) -> Tuple[bool, str]:
        """
        Validate authorization for setpoint change.

        Args:
            authorization: Authorization to validate
            tag: Setpoint tag
            value: Proposed value

        Returns:
            Tuple of (is_valid, message)
        """
        # Check expiration
        if datetime.now() > authorization.valid_until:
            return False, "Authorization expired"

        if datetime.now() < authorization.valid_from:
            return False, "Authorization not yet valid"

        # Check tag matches
        if authorization.tag != tag:
            return False, f"Authorization tag mismatch: {authorization.tag} != {tag}"

        # Check value matches
        if abs(authorization.proposed_value - value) > 0.001:
            return False, (
                f"Authorization value mismatch: "
                f"{authorization.proposed_value} != {value}"
            )

        # Check authorization level
        definition = self._definitions.get(tag)
        if definition:
            required_level = definition.constraints.required_authorization
            auth_levels = [
                AuthorizationLevel.OPERATOR,
                AuthorizationLevel.SUPERVISOR,
                AuthorizationLevel.ENGINEER,
                AuthorizationLevel.SAFETY_SYSTEM
            ]
            required_idx = auth_levels.index(required_level)
            provided_idx = auth_levels.index(authorization.authorization_level)

            if provided_idx < required_idx:
                return False, (
                    f"Insufficient authorization level: "
                    f"{authorization.authorization_level.value} < {required_level.value}"
                )

        return True, "Authorization valid"

    def _add_to_history(self, tag: str, change: SetpointChange) -> None:
        """Add change to history with size limit."""
        self._history[tag].append(change)
        if len(self._history[tag]) > self._max_history_per_tag:
            self._history[tag] = self._history[tag][-self._max_history_per_tag:]

    def _add_audit_record(
        self,
        operation: str,
        tag: str,
        details: Dict[str, Any],
        source: str,
        success: bool,
        user_id: Optional[str] = None
    ) -> None:
        """Add audit record."""
        audit_id = hashlib.sha256(
            f"AUD_{operation}_{tag}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        record = AuditRecord(
            audit_id=audit_id,
            operation=operation,
            tag=tag,
            details=details,
            user_id=user_id,
            source=source,
            success=success,
            timestamp=datetime.now()
        )

        record.provenance_hash = hashlib.sha256(
            f"{audit_id}|{operation}|{tag}|{success}".encode()
        ).hexdigest()

        self._audit_log.append(record)

        # Trim audit log if too large
        if len(self._audit_log) > self._max_audit_records:
            self._audit_log = self._audit_log[-self._max_audit_records:]
