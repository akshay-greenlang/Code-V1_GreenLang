"""
SafetyConstraintValidator - Validates all combustion safety constraints.

This module validates O2, CO, NOx, draft and other combustion constraints.
CRITICAL: All constraints MUST pass before ANY setpoint write operation.

Example:
    >>> validator = SafetyConstraintValidator()
    >>> result = validator.validate_all_constraints(state, limits)
    >>> if not result.all_passed:
    ...     # BLOCK all setpoint writes
    ...     pass
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConstraintSeverity(str, Enum):
    """Severity level of constraint violation."""
    CRITICAL = "critical"  # Immediate shutdown required
    HIGH = "high"  # Block all optimization
    MEDIUM = "medium"  # Limit optimization scope
    LOW = "low"  # Advisory only
    INFO = "info"  # Informational


class ConstraintType(str, Enum):
    """Type of safety constraint."""
    EMISSIONS = "emissions"
    COMBUSTION = "combustion"
    EQUIPMENT = "equipment"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"


class SafetyLimits(BaseModel):
    """Safety limits for constraint validation."""
    o2_min: float = Field(..., ge=0.5, description="Minimum O2 percentage")
    o2_max: float = Field(..., le=15.0, description="Maximum O2 percentage")
    co_limit: float = Field(..., ge=0, description="CO limit in ppm")
    nox_limit: float = Field(..., ge=0, description="NOx limit in ppm")
    draft_min: float = Field(..., description="Minimum draft in inwc")
    draft_max: float = Field(..., description="Maximum draft in inwc")
    flame_signal_min: float = Field(default=20.0, description="Minimum flame signal")
    opacity_limit: float = Field(default=20.0, description="Stack opacity limit %")

    @validator('o2_max')
    def validate_o2_range(cls, v, values):
        """Validate O2 range is valid."""
        if 'o2_min' in values and v <= values['o2_min']:
            raise ValueError('o2_max must be greater than o2_min')
        return v


class BurnerState(BaseModel):
    """Current state of burner for constraint validation."""
    unit_id: str = Field(..., description="Unit identifier")
    o2_actual: float = Field(..., description="Actual O2 percentage")
    co_actual: float = Field(..., ge=0, description="Actual CO in ppm")
    nox_actual: float = Field(..., ge=0, description="Actual NOx in ppm")
    draft_actual: float = Field(..., description="Actual draft in inwc")
    flame_signal: float = Field(..., description="Flame scanner signal")
    firing_rate: float = Field(..., ge=0, le=100, description="Firing rate %")
    steam_flow: float = Field(..., ge=0, description="Steam flow rate")
    fuel_flow: float = Field(..., ge=0, description="Fuel flow rate")
    air_flow: float = Field(..., ge=0, description="Air flow rate")
    opacity: Optional[float] = Field(None, ge=0, le=100, description="Stack opacity %")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Result of a single constraint validation."""
    constraint_name: str = Field(..., description="Name of constraint")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    passed: bool = Field(..., description="Whether constraint passed")
    actual_value: float = Field(..., description="Actual measured value")
    limit_value: float = Field(..., description="Limit value")
    margin: float = Field(..., description="Margin to limit")
    margin_percentage: float = Field(..., description="Margin as percentage")
    severity: ConstraintSeverity = Field(..., description="Severity if violated")
    message: str = Field(..., description="Human readable message")
    recommendation: Optional[str] = Field(None, description="Recommendation if violated")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class ComprehensiveValidation(BaseModel):
    """Result of comprehensive constraint validation."""
    all_passed: bool = Field(..., description="Whether all constraints passed")
    critical_violations: int = Field(default=0, description="Count of critical violations")
    high_violations: int = Field(default=0, description="Count of high violations")
    medium_violations: int = Field(default=0, description="Count of medium violations")
    low_violations: int = Field(default=0, description="Count of low violations")
    results: List[ValidationResult] = Field(default_factory=list)
    blocking_constraints: List[str] = Field(default_factory=list)
    overall_health_score: float = Field(..., ge=0, le=100, description="Overall health 0-100")
    recommendation: str = Field(..., description="Overall recommendation")
    can_optimize: bool = Field(..., description="Whether optimization is allowed")
    observe_only: bool = Field(default=False, description="Fallback to observe-only mode")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class SafetyConstraintValidator:
    """
    SafetyConstraintValidator validates all combustion safety constraints.

    CRITICAL SAFETY INVARIANT:
    - All constraints MUST pass before ANY setpoint write
    - ANY critical violation triggers immediate observe-only mode
    - Validation results are immutable and auditable

    Attributes:
        validation_history: List of past validation results

    Example:
        >>> validator = SafetyConstraintValidator()
        >>> state = BurnerState(unit_id="BLR-001", o2_actual=3.5, ...)
        >>> limits = SafetyLimits(o2_min=2.0, o2_max=6.0, ...)
        >>> result = validator.validate_all_constraints(state, limits)
        >>> if not result.all_passed:
        ...     # BLOCK optimization, fallback to observe-only
        ...     pass
    """

    def __init__(self):
        """Initialize SafetyConstraintValidator."""
        self.validation_history: List[ComprehensiveValidation] = []
        self._creation_time = datetime.utcnow()
        logger.info("SafetyConstraintValidator initialized")

    def validate_o2_limits(
        self,
        o2: float,
        min_o2: float,
        max_o2: float
    ) -> ValidationResult:
        """
        Validate O2 is within safe limits.

        O2 too low: risk of incomplete combustion, CO formation
        O2 too high: energy waste, potential for NOx formation

        Args:
            o2: Actual O2 percentage
            min_o2: Minimum O2 limit
            max_o2: Maximum O2 limit

        Returns:
            ValidationResult with pass/fail status
        """
        passed = min_o2 <= o2 <= max_o2

        if o2 < min_o2:
            margin = o2 - min_o2  # Negative
            limit_value = min_o2
            severity = ConstraintSeverity.CRITICAL if o2 < (min_o2 - 0.5) else ConstraintSeverity.HIGH
            message = f"O2 below minimum: {o2:.2f}% < {min_o2:.2f}%"
            recommendation = "Increase excess air immediately"
        elif o2 > max_o2:
            margin = max_o2 - o2  # Negative
            limit_value = max_o2
            severity = ConstraintSeverity.MEDIUM
            message = f"O2 above maximum: {o2:.2f}% > {max_o2:.2f}%"
            recommendation = "Reduce excess air to improve efficiency"
        else:
            margin = min(o2 - min_o2, max_o2 - o2)
            limit_value = min_o2 if (o2 - min_o2) < (max_o2 - o2) else max_o2
            severity = ConstraintSeverity.INFO
            message = f"O2 within limits: {o2:.2f}% in [{min_o2:.2f}%, {max_o2:.2f}%]"
            recommendation = None

        range_size = max_o2 - min_o2
        margin_percentage = (margin / range_size * 100) if range_size > 0 else 0

        provenance_hash = hashlib.sha256(
            f"o2_{o2}_{min_o2}_{max_o2}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return ValidationResult(
            constraint_name="O2_limits",
            constraint_type=ConstraintType.COMBUSTION,
            passed=passed,
            actual_value=o2,
            limit_value=limit_value,
            margin=margin,
            margin_percentage=margin_percentage,
            severity=severity,
            message=message,
            recommendation=recommendation,
            provenance_hash=provenance_hash
        )

    def validate_co_limit(
        self,
        co: float,
        co_limit: float
    ) -> ValidationResult:
        """
        Validate CO is below limit.

        CO indicates incomplete combustion - safety and efficiency concern.

        Args:
            co: Actual CO in ppm
            co_limit: Maximum CO limit in ppm

        Returns:
            ValidationResult with pass/fail status
        """
        passed = co <= co_limit
        margin = co_limit - co

        if not passed:
            severity = ConstraintSeverity.CRITICAL if co > (co_limit * 1.5) else ConstraintSeverity.HIGH
            message = f"CO exceeds limit: {co:.1f} ppm > {co_limit:.1f} ppm"
            recommendation = "Increase excess air or check burner alignment"
        else:
            severity = ConstraintSeverity.INFO
            message = f"CO within limit: {co:.1f} ppm <= {co_limit:.1f} ppm"
            recommendation = None

        margin_percentage = (margin / co_limit * 100) if co_limit > 0 else 0

        provenance_hash = hashlib.sha256(
            f"co_{co}_{co_limit}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return ValidationResult(
            constraint_name="CO_limit",
            constraint_type=ConstraintType.EMISSIONS,
            passed=passed,
            actual_value=co,
            limit_value=co_limit,
            margin=margin,
            margin_percentage=margin_percentage,
            severity=severity,
            message=message,
            recommendation=recommendation,
            provenance_hash=provenance_hash
        )

    def validate_nox_limit(
        self,
        nox: float,
        nox_limit: float
    ) -> ValidationResult:
        """
        Validate NOx is below regulatory limit.

        NOx is a regulated pollutant - compliance is mandatory.

        Args:
            nox: Actual NOx in ppm
            nox_limit: Maximum NOx limit in ppm

        Returns:
            ValidationResult with pass/fail status
        """
        passed = nox <= nox_limit
        margin = nox_limit - nox

        if not passed:
            severity = ConstraintSeverity.CRITICAL  # Regulatory violation
            message = f"NOx exceeds regulatory limit: {nox:.1f} ppm > {nox_limit:.1f} ppm"
            recommendation = "Reduce firing rate or adjust combustion parameters"
        else:
            if margin < (nox_limit * 0.1):
                severity = ConstraintSeverity.MEDIUM
                message = f"NOx approaching limit: {nox:.1f} ppm (limit: {nox_limit:.1f} ppm)"
                recommendation = "Monitor closely, consider preemptive reduction"
            else:
                severity = ConstraintSeverity.INFO
                message = f"NOx within limit: {nox:.1f} ppm <= {nox_limit:.1f} ppm"
                recommendation = None

        margin_percentage = (margin / nox_limit * 100) if nox_limit > 0 else 0

        provenance_hash = hashlib.sha256(
            f"nox_{nox}_{nox_limit}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return ValidationResult(
            constraint_name="NOx_limit",
            constraint_type=ConstraintType.REGULATORY,
            passed=passed,
            actual_value=nox,
            limit_value=nox_limit,
            margin=margin,
            margin_percentage=margin_percentage,
            severity=severity,
            message=message,
            recommendation=recommendation,
            provenance_hash=provenance_hash
        )

    def validate_draft_limits(
        self,
        draft: float,
        min_draft: float,
        max_draft: float
    ) -> ValidationResult:
        """
        Validate furnace draft is within safe limits.

        Draft too low: risk of flame instability or puff-back
        Draft too high: excessive heat loss, potential tube damage

        Args:
            draft: Actual draft in inches water column
            min_draft: Minimum draft limit
            max_draft: Maximum draft limit

        Returns:
            ValidationResult with pass/fail status
        """
        passed = min_draft <= draft <= max_draft

        if draft < min_draft:
            margin = draft - min_draft  # Negative
            limit_value = min_draft
            severity = ConstraintSeverity.HIGH
            message = f"Draft below minimum: {draft:.3f} inwc < {min_draft:.3f} inwc"
            recommendation = "Increase induced draft fan speed"
        elif draft > max_draft:
            margin = max_draft - draft  # Negative
            limit_value = max_draft
            severity = ConstraintSeverity.MEDIUM
            message = f"Draft above maximum: {draft:.3f} inwc > {max_draft:.3f} inwc"
            recommendation = "Reduce induced draft fan speed"
        else:
            margin = min(draft - min_draft, max_draft - draft)
            limit_value = min_draft if (draft - min_draft) < (max_draft - draft) else max_draft
            severity = ConstraintSeverity.INFO
            message = f"Draft within limits: {draft:.3f} inwc"
            recommendation = None

        range_size = max_draft - min_draft
        margin_percentage = (margin / range_size * 100) if range_size > 0 else 0

        provenance_hash = hashlib.sha256(
            f"draft_{draft}_{min_draft}_{max_draft}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return ValidationResult(
            constraint_name="draft_limits",
            constraint_type=ConstraintType.COMBUSTION,
            passed=passed,
            actual_value=draft,
            limit_value=limit_value,
            margin=margin,
            margin_percentage=margin_percentage,
            severity=severity,
            message=message,
            recommendation=recommendation,
            provenance_hash=provenance_hash
        )

    def validate_all_constraints(
        self,
        state: BurnerState,
        limits: SafetyLimits
    ) -> ComprehensiveValidation:
        """
        Validate ALL safety constraints comprehensively.

        CRITICAL: This method MUST be called before ANY optimization action.

        Args:
            state: Current burner state
            limits: Safety limits to validate against

        Returns:
            ComprehensiveValidation with overall pass/fail and details
        """
        results: List[ValidationResult] = []

        # Validate O2
        o2_result = self.validate_o2_limits(
            state.o2_actual,
            limits.o2_min,
            limits.o2_max
        )
        results.append(o2_result)

        # Validate CO
        co_result = self.validate_co_limit(
            state.co_actual,
            limits.co_limit
        )
        results.append(co_result)

        # Validate NOx
        nox_result = self.validate_nox_limit(
            state.nox_actual,
            limits.nox_limit
        )
        results.append(nox_result)

        # Validate Draft
        draft_result = self.validate_draft_limits(
            state.draft_actual,
            limits.draft_min,
            limits.draft_max
        )
        results.append(draft_result)

        # Validate flame signal
        flame_result = self._validate_flame_signal(
            state.flame_signal,
            limits.flame_signal_min
        )
        results.append(flame_result)

        # Validate opacity if available
        if state.opacity is not None:
            opacity_result = self._validate_opacity(
                state.opacity,
                limits.opacity_limit
            )
            results.append(opacity_result)

        # Count violations by severity
        critical_violations = sum(
            1 for r in results
            if not r.passed and r.severity == ConstraintSeverity.CRITICAL
        )
        high_violations = sum(
            1 for r in results
            if not r.passed and r.severity == ConstraintSeverity.HIGH
        )
        medium_violations = sum(
            1 for r in results
            if not r.passed and r.severity == ConstraintSeverity.MEDIUM
        )
        low_violations = sum(
            1 for r in results
            if not r.passed and r.severity == ConstraintSeverity.LOW
        )

        # Determine blocking constraints
        blocking_constraints = [
            r.constraint_name for r in results
            if not r.passed and r.severity in [
                ConstraintSeverity.CRITICAL,
                ConstraintSeverity.HIGH
            ]
        ]

        # Calculate overall health score
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        base_score = (passed_count / total_count * 100) if total_count > 0 else 0

        # Penalize for severity
        health_score = base_score - (critical_violations * 30) - (high_violations * 15)
        health_score = max(0, min(100, health_score))

        # Determine if optimization is allowed
        all_passed = all(r.passed for r in results)
        can_optimize = critical_violations == 0 and high_violations == 0
        observe_only = critical_violations > 0

        # Generate recommendation
        if observe_only:
            recommendation = "CRITICAL: Switch to observe-only mode immediately"
        elif not can_optimize:
            recommendation = "WARNING: Optimization blocked, address high-priority constraints"
        elif medium_violations > 0:
            recommendation = "CAUTION: Optimization allowed with restrictions"
        else:
            recommendation = "OK: All constraints satisfied, full optimization allowed"

        # Create provenance hash
        provenance_str = f"{state.json()}{limits.json()}{datetime.utcnow().isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        comprehensive = ComprehensiveValidation(
            all_passed=all_passed,
            critical_violations=critical_violations,
            high_violations=high_violations,
            medium_violations=medium_violations,
            low_violations=low_violations,
            results=results,
            blocking_constraints=blocking_constraints,
            overall_health_score=health_score,
            recommendation=recommendation,
            can_optimize=can_optimize,
            observe_only=observe_only,
            provenance_hash=provenance_hash
        )

        # Store in history
        self.validation_history.append(comprehensive)

        # Log result
        if observe_only:
            logger.critical(
                f"CONSTRAINT VALIDATION CRITICAL: {critical_violations} critical violations - "
                f"OBSERVE ONLY MODE"
            )
        elif not can_optimize:
            logger.warning(
                f"Constraint validation: {high_violations} high violations - optimization blocked"
            )
        else:
            logger.info(f"Constraint validation passed: health score {health_score:.1f}")

        return comprehensive

    def _validate_flame_signal(
        self,
        flame_signal: float,
        min_signal: float
    ) -> ValidationResult:
        """Validate flame signal is above minimum."""
        passed = flame_signal >= min_signal
        margin = flame_signal - min_signal

        if not passed:
            severity = ConstraintSeverity.CRITICAL
            message = f"Flame signal below minimum: {flame_signal:.1f} < {min_signal:.1f}"
            recommendation = "CHECK FLAME IMMEDIATELY - potential flameout"
        else:
            if margin < (min_signal * 0.2):
                severity = ConstraintSeverity.HIGH
                message = f"Flame signal weak: {flame_signal:.1f}"
                recommendation = "Monitor flame stability closely"
            else:
                severity = ConstraintSeverity.INFO
                message = f"Flame signal OK: {flame_signal:.1f}"
                recommendation = None

        margin_percentage = (margin / min_signal * 100) if min_signal > 0 else 0

        provenance_hash = hashlib.sha256(
            f"flame_{flame_signal}_{min_signal}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return ValidationResult(
            constraint_name="flame_signal",
            constraint_type=ConstraintType.COMBUSTION,
            passed=passed,
            actual_value=flame_signal,
            limit_value=min_signal,
            margin=margin,
            margin_percentage=margin_percentage,
            severity=severity,
            message=message,
            recommendation=recommendation,
            provenance_hash=provenance_hash
        )

    def _validate_opacity(
        self,
        opacity: float,
        limit: float
    ) -> ValidationResult:
        """Validate stack opacity is below limit."""
        passed = opacity <= limit
        margin = limit - opacity

        if not passed:
            severity = ConstraintSeverity.HIGH
            message = f"Opacity exceeds limit: {opacity:.1f}% > {limit:.1f}%"
            recommendation = "Check combustion, possible soot buildup"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Opacity within limit: {opacity:.1f}%"
            recommendation = None

        margin_percentage = (margin / limit * 100) if limit > 0 else 0

        provenance_hash = hashlib.sha256(
            f"opacity_{opacity}_{limit}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return ValidationResult(
            constraint_name="opacity_limit",
            constraint_type=ConstraintType.REGULATORY,
            passed=passed,
            actual_value=opacity,
            limit_value=limit,
            margin=margin,
            margin_percentage=margin_percentage,
            severity=severity,
            message=message,
            recommendation=recommendation,
            provenance_hash=provenance_hash
        )
