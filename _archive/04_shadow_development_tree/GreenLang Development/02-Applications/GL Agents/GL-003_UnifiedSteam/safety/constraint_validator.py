"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Constraint Validator

This module implements constraint validation for all recommendations,
setpoint changes, and actuator operations to ensure safety compliance.

Safety Architecture:
    - Never pass recommendations violating safety constraints
    - Multi-layer validation (recommendation, setpoint, actuator)
    - Rate of change enforcement
    - Complete validation audit trail

Reference Standards:
    - IEC 61511 Functional Safety
    - ISA-84 Safety Instrumented Systems
    - ASME B31.1 Power Piping

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ValidationStatus(str, Enum):
    """Validation status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    CONDITIONAL = "conditional"
    PENDING = "pending"


class ConstraintCategory(str, Enum):
    """Constraint category enumeration."""
    SAFETY = "safety"
    OPERATIONAL = "operational"
    EQUIPMENT = "equipment"
    REGULATORY = "regulatory"
    PROCESS = "process"


class ViolationType(str, Enum):
    """Violation type enumeration."""
    PRESSURE_LIMIT = "pressure_limit"
    TEMPERATURE_LIMIT = "temperature_limit"
    QUALITY_LIMIT = "quality_limit"
    RATE_LIMIT = "rate_limit"
    ACTUATOR_LIMIT = "actuator_limit"
    INTERLOCK = "interlock"
    PROCESS_CONSTRAINT = "process_constraint"


# =============================================================================
# DATA MODELS
# =============================================================================

class Constraint(BaseModel):
    """Definition of a safety or operational constraint."""

    constraint_id: str = Field(..., description="Constraint identifier")
    name: str = Field(..., description="Constraint name")
    category: ConstraintCategory = Field(..., description="Constraint category")
    parameter: str = Field(..., description="Parameter being constrained")
    min_value: Optional[float] = Field(None, description="Minimum allowed value")
    max_value: Optional[float] = Field(None, description="Maximum allowed value")
    max_rate: Optional[float] = Field(
        None,
        description="Maximum rate of change per minute"
    )
    unit: str = Field(default="", description="Parameter unit")
    description: str = Field(default="", description="Constraint description")
    is_hard_constraint: bool = Field(
        default=True,
        description="Hard constraint cannot be violated"
    )
    enabled: bool = Field(default=True, description="Constraint is enabled")


class Recommendation(BaseModel):
    """Optimization recommendation to be validated."""

    recommendation_id: str = Field(..., description="Recommendation ID")
    equipment_id: str = Field(..., description="Target equipment ID")
    parameter: str = Field(..., description="Parameter being changed")
    current_value: float = Field(..., description="Current value")
    proposed_value: float = Field(..., description="Proposed value")
    unit: str = Field(default="", description="Value unit")
    source: str = Field(default="optimizer", description="Recommendation source")
    reason: str = Field(default="", description="Reason for recommendation")
    expected_benefit: str = Field(default="", description="Expected benefit")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Recommendation timestamp"
    )


class SetpointChange(BaseModel):
    """Setpoint change request to be validated."""

    change_id: str = Field(..., description="Change ID")
    tag: str = Field(..., description="Setpoint tag")
    equipment_id: str = Field(..., description="Equipment ID")
    current_value: float = Field(..., description="Current setpoint")
    proposed_value: float = Field(..., description="Proposed setpoint")
    unit: str = Field(default="", description="Setpoint unit")
    source: str = Field(default="", description="Change source")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Change request timestamp"
    )


class Violation(BaseModel):
    """Constraint violation details."""

    violation_id: str = Field(..., description="Violation ID")
    constraint_id: str = Field(..., description="Violated constraint ID")
    violation_type: ViolationType = Field(..., description="Type of violation")
    parameter: str = Field(..., description="Parameter in violation")
    value: float = Field(..., description="Violating value")
    limit: float = Field(..., description="Constraint limit")
    severity: str = Field(default="high", description="Violation severity")
    message: str = Field(..., description="Violation message")
    remediation: str = Field(
        default="",
        description="Suggested remediation"
    )


class ValidationResult(BaseModel):
    """Result of constraint validation."""

    validation_id: str = Field(..., description="Validation ID")
    status: ValidationStatus = Field(..., description="Validation status")
    item_type: str = Field(
        ...,
        description="Type of item validated (recommendation, setpoint, actuator)"
    )
    item_id: str = Field(..., description="ID of validated item")
    constraints_checked: int = Field(
        default=0,
        description="Number of constraints checked"
    )
    constraints_passed: int = Field(
        default=0,
        description="Number of constraints passed"
    )
    violations: List[Violation] = Field(
        default_factory=list,
        description="List of violations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    conditions: List[str] = Field(
        default_factory=list,
        description="Conditions for conditional approval"
    )
    validated_value: Optional[float] = Field(
        None,
        description="Validated value (may be adjusted)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Validation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class ActuatorCheck(BaseModel):
    """Result of actuator limit check."""

    check_id: str = Field(..., description="Check ID")
    actuator_id: str = Field(..., description="Actuator ID")
    proposed_position: float = Field(
        ...,
        description="Proposed actuator position"
    )
    current_position: float = Field(
        ...,
        description="Current actuator position"
    )
    min_position: float = Field(
        default=0.0,
        description="Minimum allowed position"
    )
    max_position: float = Field(
        default=100.0,
        description="Maximum allowed position"
    )
    within_limits: bool = Field(
        ...,
        description="Position is within limits"
    )
    clamped_position: float = Field(
        ...,
        description="Position clamped to limits"
    )
    travel_required: float = Field(
        ...,
        description="Travel required for change"
    )
    message: str = Field(default="", description="Check message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Check timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class RateCheck(BaseModel):
    """Result of rate of change check."""

    check_id: str = Field(..., description="Check ID")
    parameter: str = Field(..., description="Parameter checked")
    current_value: float = Field(..., description="Current value")
    proposed_value: float = Field(..., description="Proposed value")
    time_delta_min: float = Field(
        ...,
        description="Time delta in minutes"
    )
    calculated_rate: float = Field(
        ...,
        description="Calculated rate of change"
    )
    max_rate: float = Field(..., description="Maximum allowed rate")
    unit: str = Field(default="", description="Rate unit")
    within_limit: bool = Field(
        ...,
        description="Rate is within limit"
    )
    limited_value: float = Field(
        ...,
        description="Value limited to max rate"
    )
    message: str = Field(default="", description="Check message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Check timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


# =============================================================================
# CONSTRAINT VALIDATOR
# =============================================================================

class ConstraintValidator:
    """
    Constraint validation for safety compliance.

    This validator ensures all recommendations, setpoint changes, and actuator
    operations comply with safety constraints before implementation.

    Validation Philosophy:
        - NEVER pass recommendations violating hard safety constraints
        - Soft constraints generate warnings but may be overridden
        - Rate limits prevent rapid changes that cause equipment damage
        - All validations logged for audit trail

    Features:
        - Multi-layer validation hierarchy
        - Constraint prioritization (safety > equipment > operational)
        - Rate of change enforcement
        - Actuator limit checking
        - Complete validation audit trail

    Attributes:
        _constraints: Registered constraints by ID
        _validation_history: History of validations
        _last_values: Last known values for rate checking

    Example:
        >>> validator = ConstraintValidator()
        >>> validator.register_constraint(Constraint(...))
        >>> result = validator.validate_recommendation(recommendation, safety_envelope)
    """

    def __init__(self):
        """Initialize ConstraintValidator."""
        self._constraints: Dict[str, Constraint] = {}
        self._validation_history: List[ValidationResult] = []
        self._last_values: Dict[str, Tuple[float, datetime]] = {}
        self._max_history_size = 5000

        logger.info("ConstraintValidator initialized")

    def register_constraint(self, constraint: Constraint) -> None:
        """
        Register a constraint for validation.

        Args:
            constraint: Constraint to register
        """
        self._constraints[constraint.constraint_id] = constraint
        logger.info(
            f"Registered constraint {constraint.constraint_id}: "
            f"{constraint.name} ({constraint.category.value})"
        )

    def validate_recommendation(
        self,
        recommendation: Recommendation,
        safety_envelope: Any
    ) -> ValidationResult:
        """
        Validate a recommendation against safety envelope and constraints.

        This method performs comprehensive validation of optimization
        recommendations to ensure they do not violate safety limits.

        Args:
            recommendation: Recommendation to validate
            safety_envelope: SafetyEnvelope instance for envelope checks

        Returns:
            ValidationResult: Validation result with any violations
        """
        start_time = datetime.now()
        violations = []
        warnings = []
        conditions = []
        constraints_checked = 0
        constraints_passed = 0

        # Step 1: Check against safety envelope
        if safety_envelope is not None:
            try:
                envelope_result = safety_envelope.check_within_envelope(
                    equipment_id=recommendation.equipment_id,
                    parameter=recommendation.parameter,
                    value=recommendation.proposed_value
                )

                constraints_checked += 1

                if envelope_result.status.value.startswith("trip"):
                    violations.append(Violation(
                        violation_id=f"VIO_ENV_{envelope_result.check_id}",
                        constraint_id="safety_envelope",
                        violation_type=self._get_violation_type(recommendation.parameter),
                        parameter=recommendation.parameter,
                        value=recommendation.proposed_value,
                        limit=envelope_result.min_limit if "low" in envelope_result.status.value else envelope_result.max_limit,
                        severity="critical",
                        message=envelope_result.message,
                        remediation=f"Reduce {recommendation.parameter} to within envelope limits"
                    ))
                elif envelope_result.status.value.startswith("alarm"):
                    warnings.append(envelope_result.message)
                    constraints_passed += 1
                else:
                    constraints_passed += 1

            except KeyError:
                warnings.append(
                    f"No envelope limits defined for {recommendation.equipment_id}.{recommendation.parameter}"
                )

        # Step 2: Check against registered constraints
        for constraint in self._constraints.values():
            if not constraint.enabled:
                continue

            if constraint.parameter != recommendation.parameter:
                continue

            constraints_checked += 1
            violation = self._check_constraint(
                constraint, recommendation.proposed_value
            )

            if violation:
                if constraint.is_hard_constraint:
                    violations.append(violation)
                else:
                    warnings.append(violation.message)
                    constraints_passed += 1
            else:
                constraints_passed += 1

        # Step 3: Check rate of change if we have previous value
        rate_check = self.check_rate_of_change(
            parameter=f"{recommendation.equipment_id}.{recommendation.parameter}",
            current=recommendation.current_value,
            proposed=recommendation.proposed_value
        )

        constraints_checked += 1
        if not rate_check.within_limit:
            violations.append(Violation(
                violation_id=f"VIO_RATE_{rate_check.check_id}",
                constraint_id="rate_limit",
                violation_type=ViolationType.RATE_LIMIT,
                parameter=recommendation.parameter,
                value=rate_check.calculated_rate,
                limit=rate_check.max_rate,
                severity="high",
                message=rate_check.message,
                remediation=f"Limit change to {rate_check.limited_value}"
            ))
            conditions.append(
                f"Apply change gradually over multiple steps to limit={rate_check.limited_value}"
            )
        else:
            constraints_passed += 1

        # Determine overall status
        if violations:
            # Check if all violations are rate-related (can be conditional)
            non_rate_violations = [
                v for v in violations
                if v.violation_type != ViolationType.RATE_LIMIT
            ]
            if non_rate_violations:
                status = ValidationStatus.INVALID
            else:
                status = ValidationStatus.CONDITIONAL
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        # Generate validation ID
        validation_id = hashlib.sha256(
            f"VAL_{recommendation.recommendation_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        # Determine validated value
        validated_value = recommendation.proposed_value
        if status == ValidationStatus.CONDITIONAL and rate_check:
            validated_value = rate_check.limited_value

        result = ValidationResult(
            validation_id=validation_id,
            status=status,
            item_type="recommendation",
            item_id=recommendation.recommendation_id,
            constraints_checked=constraints_checked,
            constraints_passed=constraints_passed,
            violations=violations,
            warnings=warnings,
            conditions=conditions,
            validated_value=validated_value if status != ValidationStatus.INVALID else None,
            timestamp=datetime.now()
        )

        result.provenance_hash = hashlib.sha256(
            f"{validation_id}|{status.value}|{len(violations)}".encode()
        ).hexdigest()

        # Store in history
        self._add_to_history(result)

        # Log result
        if status == ValidationStatus.INVALID:
            logger.warning(
                f"Recommendation {recommendation.recommendation_id} INVALID: "
                f"{len(violations)} violations"
            )
        else:
            logger.info(
                f"Recommendation {recommendation.recommendation_id} {status.value}: "
                f"checked={constraints_checked}, passed={constraints_passed}"
            )

        return result

    def validate_setpoint_change(
        self,
        change: SetpointChange,
        constraints: List[Constraint]
    ) -> ValidationResult:
        """
        Validate a setpoint change against constraints.

        Args:
            change: Setpoint change to validate
            constraints: List of applicable constraints

        Returns:
            ValidationResult: Validation result
        """
        start_time = datetime.now()
        violations = []
        warnings = []
        constraints_checked = 0
        constraints_passed = 0

        # Check each constraint
        for constraint in constraints:
            if not constraint.enabled:
                continue

            constraints_checked += 1
            violation = self._check_constraint(constraint, change.proposed_value)

            if violation:
                if constraint.is_hard_constraint:
                    violations.append(violation)
                else:
                    warnings.append(violation.message)
                    constraints_passed += 1
            else:
                constraints_passed += 1

        # Check rate of change
        rate_check = self.check_rate_of_change(
            parameter=change.tag,
            current=change.current_value,
            proposed=change.proposed_value
        )

        constraints_checked += 1
        if not rate_check.within_limit:
            violations.append(Violation(
                violation_id=f"VIO_RATE_{rate_check.check_id}",
                constraint_id="rate_limit",
                violation_type=ViolationType.RATE_LIMIT,
                parameter=change.tag,
                value=rate_check.calculated_rate,
                limit=rate_check.max_rate,
                severity="high",
                message=rate_check.message,
                remediation=f"Limit change to {rate_check.limited_value}"
            ))
        else:
            constraints_passed += 1

        # Determine status
        if violations:
            status = ValidationStatus.INVALID
        elif warnings:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.VALID

        # Generate validation ID
        validation_id = hashlib.sha256(
            f"VAL_SP_{change.change_id}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = ValidationResult(
            validation_id=validation_id,
            status=status,
            item_type="setpoint_change",
            item_id=change.change_id,
            constraints_checked=constraints_checked,
            constraints_passed=constraints_passed,
            violations=violations,
            warnings=warnings,
            validated_value=change.proposed_value if status != ValidationStatus.INVALID else None,
            timestamp=datetime.now()
        )

        result.provenance_hash = hashlib.sha256(
            f"{validation_id}|{status.value}|{len(violations)}".encode()
        ).hexdigest()

        self._add_to_history(result)

        logger.info(
            f"Setpoint change {change.change_id} {status.value}: "
            f"checked={constraints_checked}, passed={constraints_passed}"
        )

        return result

    def check_actuator_limits(
        self,
        actuator_id: str,
        proposed_position: float,
        current_position: Optional[float] = None,
        min_position: float = 0.0,
        max_position: float = 100.0
    ) -> ActuatorCheck:
        """
        Check actuator position against limits.

        Args:
            actuator_id: Actuator identifier
            proposed_position: Proposed position (%)
            current_position: Current position (optional)
            min_position: Minimum allowed position
            max_position: Maximum allowed position

        Returns:
            ActuatorCheck: Actuator limit check result
        """
        start_time = datetime.now()

        # Clamp to limits
        clamped = max(min_position, min(max_position, proposed_position))
        within_limits = (proposed_position >= min_position and
                        proposed_position <= max_position)

        # Calculate travel
        current = current_position if current_position is not None else 50.0
        travel = abs(proposed_position - current)

        # Build message
        if within_limits:
            message = f"Actuator {actuator_id} position {proposed_position}% within limits"
        else:
            message = (
                f"Actuator {actuator_id} position {proposed_position}% outside limits "
                f"[{min_position}, {max_position}], clamped to {clamped}%"
            )

        # Generate check ID
        check_id = hashlib.sha256(
            f"ACT_{actuator_id}_{proposed_position}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = ActuatorCheck(
            check_id=check_id,
            actuator_id=actuator_id,
            proposed_position=proposed_position,
            current_position=current,
            min_position=min_position,
            max_position=max_position,
            within_limits=within_limits,
            clamped_position=clamped,
            travel_required=travel,
            message=message,
            timestamp=datetime.now()
        )

        result.provenance_hash = hashlib.sha256(
            f"{check_id}|{within_limits}|{clamped}".encode()
        ).hexdigest()

        if not within_limits:
            logger.warning(message)
        else:
            logger.debug(message)

        return result

    def check_rate_of_change(
        self,
        parameter: str,
        current: float,
        proposed: float,
        max_rate_per_min: Optional[float] = None,
        unit: str = ""
    ) -> RateCheck:
        """
        Check rate of change against limits.

        Args:
            parameter: Parameter identifier
            current: Current value
            proposed: Proposed value
            max_rate_per_min: Maximum rate per minute (uses default if None)
            unit: Value unit

        Returns:
            RateCheck: Rate check result
        """
        start_time = datetime.now()

        # Get default max rate from constraints or use conservative default
        if max_rate_per_min is None:
            # Check if we have a constraint for this parameter
            for constraint in self._constraints.values():
                if constraint.parameter == parameter and constraint.max_rate:
                    max_rate_per_min = constraint.max_rate
                    break

            # Default conservative rate limit
            if max_rate_per_min is None:
                max_rate_per_min = float('inf')  # No limit if not specified

        # Calculate time delta from last value
        time_delta_min = 1.0  # Default to 1 minute
        if parameter in self._last_values:
            last_value, last_time = self._last_values[parameter]
            time_delta_min = max(
                0.001,
                (start_time - last_time).total_seconds() / 60.0
            )

        # Calculate rate
        change = abs(proposed - current)
        calculated_rate = change / time_delta_min if time_delta_min > 0 else float('inf')

        # Check against limit
        within_limit = calculated_rate <= max_rate_per_min

        # Calculate limited value
        if within_limit:
            limited_value = proposed
        else:
            max_change = max_rate_per_min * time_delta_min
            if proposed > current:
                limited_value = current + max_change
            else:
                limited_value = current - max_change

        # Build message
        if within_limit:
            message = (
                f"Rate {calculated_rate:.2f}/min within limit {max_rate_per_min}/min"
            )
        else:
            message = (
                f"Rate {calculated_rate:.2f}/min exceeds limit {max_rate_per_min}/min, "
                f"limited to {limited_value}"
            )

        # Update last value
        self._last_values[parameter] = (current, start_time)

        # Generate check ID
        check_id = hashlib.sha256(
            f"RATE_{parameter}_{proposed}_{start_time.isoformat()}".encode()
        ).hexdigest()[:16]

        result = RateCheck(
            check_id=check_id,
            parameter=parameter,
            current_value=current,
            proposed_value=proposed,
            time_delta_min=time_delta_min,
            calculated_rate=calculated_rate,
            max_rate=max_rate_per_min,
            unit=unit,
            within_limit=within_limit,
            limited_value=limited_value,
            message=message,
            timestamp=datetime.now()
        )

        result.provenance_hash = hashlib.sha256(
            f"{check_id}|{within_limit}|{calculated_rate}".encode()
        ).hexdigest()

        return result

    def get_validation_history(
        self,
        item_id: Optional[str] = None,
        time_window_minutes: int = 60
    ) -> List[ValidationResult]:
        """
        Get validation history.

        Args:
            item_id: Filter by item ID (optional)
            time_window_minutes: Time window in minutes

        Returns:
            List of validation results
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        results = [
            v for v in self._validation_history
            if v.timestamp >= cutoff_time
        ]

        if item_id:
            results = [v for v in results if v.item_id == item_id]

        return results

    def get_constraints(
        self,
        category: Optional[ConstraintCategory] = None
    ) -> List[Constraint]:
        """Get registered constraints."""
        if category:
            return [
                c for c in self._constraints.values()
                if c.category == category
            ]
        return list(self._constraints.values())

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _check_constraint(
        self,
        constraint: Constraint,
        value: float
    ) -> Optional[Violation]:
        """
        Check value against a single constraint.

        Args:
            constraint: Constraint to check
            value: Value to check

        Returns:
            Violation if constraint violated, None otherwise
        """
        violation_id = hashlib.sha256(
            f"VIO_{constraint.constraint_id}_{value}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Check minimum
        if constraint.min_value is not None and value < constraint.min_value:
            return Violation(
                violation_id=violation_id,
                constraint_id=constraint.constraint_id,
                violation_type=self._get_violation_type(constraint.parameter),
                parameter=constraint.parameter,
                value=value,
                limit=constraint.min_value,
                severity="high" if constraint.is_hard_constraint else "medium",
                message=(
                    f"{constraint.name}: {value} below minimum {constraint.min_value} "
                    f"{constraint.unit}"
                ),
                remediation=f"Increase {constraint.parameter} to at least {constraint.min_value}"
            )

        # Check maximum
        if constraint.max_value is not None and value > constraint.max_value:
            return Violation(
                violation_id=violation_id,
                constraint_id=constraint.constraint_id,
                violation_type=self._get_violation_type(constraint.parameter),
                parameter=constraint.parameter,
                value=value,
                limit=constraint.max_value,
                severity="high" if constraint.is_hard_constraint else "medium",
                message=(
                    f"{constraint.name}: {value} above maximum {constraint.max_value} "
                    f"{constraint.unit}"
                ),
                remediation=f"Decrease {constraint.parameter} to at most {constraint.max_value}"
            )

        return None

    def _get_violation_type(self, parameter: str) -> ViolationType:
        """Map parameter to violation type."""
        parameter_lower = parameter.lower()

        if "pressure" in parameter_lower:
            return ViolationType.PRESSURE_LIMIT
        elif "temp" in parameter_lower:
            return ViolationType.TEMPERATURE_LIMIT
        elif "quality" in parameter_lower or "dryness" in parameter_lower:
            return ViolationType.QUALITY_LIMIT
        elif "rate" in parameter_lower:
            return ViolationType.RATE_LIMIT
        elif "valve" in parameter_lower or "actuator" in parameter_lower:
            return ViolationType.ACTUATOR_LIMIT
        else:
            return ViolationType.PROCESS_CONSTRAINT

    def _add_to_history(self, result: ValidationResult) -> None:
        """Add validation result to history with size limit."""
        self._validation_history.append(result)
        if len(self._validation_history) > self._max_history_size:
            self._validation_history = self._validation_history[-self._max_history_size:]
