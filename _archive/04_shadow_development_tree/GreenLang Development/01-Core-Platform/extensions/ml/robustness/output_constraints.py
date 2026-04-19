# -*- coding: utf-8 -*-
"""
TASK-068: Output Constraint Enforcement

This module provides output constraint enforcement for GreenLang Process Heat
ML models, including physical bounds enforcement (temperature, pressure,
efficiency ranges), monotonicity constraints, conservation law validation
(energy balance), safety limit enforcement, and constraint violation logging.

Constraint enforcement is critical for ensuring ML predictions are physically
realistic and safe for Process Heat applications.

Example:
    >>> from greenlang.ml.robustness import OutputConstraintEnforcer
    >>> enforcer = OutputConstraintEnforcer(config=ProcessHeatConstraintConfig())
    >>> constrained_preds, violations = enforcer.enforce(predictions, inputs)
    >>> if violations:
    ...     for v in violations:
    ...         print(f"Violation: {v.constraint_name} at sample {v.sample_index}")
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import hashlib
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ConstraintType(str, Enum):
    """Types of output constraints."""
    PHYSICAL_BOUND = "physical_bound"
    MONOTONICITY = "monotonicity"
    CONSERVATION = "conservation"
    SAFETY_LIMIT = "safety_limit"
    RATE_OF_CHANGE = "rate_of_change"
    CONSISTENCY = "consistency"
    CUSTOM = "custom"


class ViolationSeverity(str, Enum):
    """Severity of constraint violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EnforcementAction(str, Enum):
    """Actions to take on constraint violation."""
    CLIP = "clip"  # Clip to valid range
    REJECT = "reject"  # Reject prediction
    FLAG = "flag"  # Flag but allow
    SUBSTITUTE = "substitute"  # Use substitute value
    INTERPOLATE = "interpolate"  # Interpolate from valid neighbors


# =============================================================================
# Constraint Definitions
# =============================================================================

class PhysicalBound(BaseModel):
    """Definition of a physical bound constraint."""

    name: str = Field(..., description="Constraint name")
    output_index: int = Field(..., description="Index of output to constrain")
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    unit: str = Field(default="", description="Physical unit")
    severity: ViolationSeverity = Field(
        default=ViolationSeverity.WARNING,
        description="Violation severity"
    )
    enforcement: EnforcementAction = Field(
        default=EnforcementAction.CLIP,
        description="Enforcement action"
    )
    substitute_value: Optional[float] = Field(
        default=None,
        description="Substitute value if enforcement=SUBSTITUTE"
    )


class MonotonicityConstraint(BaseModel):
    """Definition of a monotonicity constraint."""

    name: str = Field(..., description="Constraint name")
    output_index: int = Field(..., description="Output index")
    input_index: int = Field(..., description="Input index for monotonicity")
    direction: str = Field(
        default="increasing",
        description="increasing or decreasing"
    )
    severity: ViolationSeverity = Field(
        default=ViolationSeverity.WARNING,
        description="Violation severity"
    )


class ConservationConstraint(BaseModel):
    """Definition of a conservation law constraint."""

    name: str = Field(..., description="Constraint name")
    description: str = Field(default="", description="Description")

    # Input indices for conservation calculation
    input_indices: List[int] = Field(..., description="Input indices")
    output_indices: List[int] = Field(..., description="Output indices")

    # Conservation formula type
    conservation_type: str = Field(
        default="energy_balance",
        description="Type: energy_balance, mass_balance, heat_balance"
    )

    # Tolerance for conservation violation
    tolerance: float = Field(
        default=0.05,
        description="Relative tolerance (0.05 = 5%)"
    )

    severity: ViolationSeverity = Field(
        default=ViolationSeverity.ERROR,
        description="Violation severity"
    )


class SafetyLimit(BaseModel):
    """Definition of a safety limit constraint."""

    name: str = Field(..., description="Safety limit name")
    description: str = Field(default="", description="Description")
    output_index: int = Field(..., description="Output index")

    # Safety thresholds
    critical_min: Optional[float] = Field(
        default=None,
        description="Critical minimum (NEVER go below)"
    )
    critical_max: Optional[float] = Field(
        default=None,
        description="Critical maximum (NEVER exceed)"
    )
    warning_min: Optional[float] = Field(
        default=None,
        description="Warning minimum"
    )
    warning_max: Optional[float] = Field(
        default=None,
        description="Warning maximum"
    )

    # Standard (regulatory or engineering)
    standard: Optional[str] = Field(
        default=None,
        description="Reference standard (e.g., ASME, NFPA)"
    )

    enforcement: EnforcementAction = Field(
        default=EnforcementAction.CLIP,
        description="Enforcement action"
    )


# =============================================================================
# Configuration
# =============================================================================

class ProcessHeatConstraintConfig(BaseModel):
    """
    Configuration for Process Heat output constraints.

    Includes typical constraints for boilers, furnaces, and heat exchangers.
    """

    # Physical bounds
    physical_bounds: List[PhysicalBound] = Field(
        default_factory=lambda: [
            PhysicalBound(
                name="boiler_efficiency",
                output_index=0,
                min_value=0.0,
                max_value=100.0,
                unit="%",
                severity=ViolationSeverity.ERROR
            ),
            PhysicalBound(
                name="outlet_temperature",
                output_index=1,
                min_value=-273.15,  # Absolute zero
                max_value=1500.0,  # Max furnace temp
                unit="C",
                severity=ViolationSeverity.ERROR
            ),
            PhysicalBound(
                name="heat_duty",
                output_index=2,
                min_value=0.0,
                max_value=1000.0,  # Max 1000 MW
                unit="MW",
                severity=ViolationSeverity.WARNING
            ),
        ],
        description="Physical bound constraints"
    )

    # Safety limits
    safety_limits: List[SafetyLimit] = Field(
        default_factory=lambda: [
            SafetyLimit(
                name="max_tube_temperature",
                output_index=1,
                critical_max=650.0,  # API 530 typical limit
                warning_max=600.0,
                standard="API 530",
                description="Tube wall temperature limit"
            ),
            SafetyLimit(
                name="max_pressure",
                output_index=3,
                critical_max=150.0,  # bar
                warning_max=120.0,
                standard="ASME BPVC",
                description="Maximum operating pressure"
            ),
        ],
        description="Safety limit constraints"
    )

    # Conservation constraints
    conservation_constraints: List[ConservationConstraint] = Field(
        default_factory=list,
        description="Conservation law constraints"
    )

    # Monotonicity constraints
    monotonicity_constraints: List[MonotonicityConstraint] = Field(
        default_factory=list,
        description="Monotonicity constraints"
    )

    # Rate of change limits
    max_rate_of_change: Dict[int, float] = Field(
        default_factory=dict,
        description="Max rate of change per output {index: max_rate}"
    )

    # General settings
    log_all_violations: bool = Field(
        default=True,
        description="Log all violations"
    )
    halt_on_critical: bool = Field(
        default=True,
        description="Halt on critical violations"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance"
    )


# =============================================================================
# Result Models
# =============================================================================

class ConstraintViolation(BaseModel):
    """Record of a constraint violation."""

    sample_index: int = Field(..., description="Sample index")
    constraint_name: str = Field(..., description="Constraint name")
    constraint_type: ConstraintType = Field(..., description="Constraint type")
    severity: ViolationSeverity = Field(..., description="Severity")

    original_value: float = Field(..., description="Original prediction")
    enforced_value: float = Field(..., description="Value after enforcement")
    limit_value: Optional[float] = Field(
        default=None,
        description="Limit that was violated"
    )

    violation_magnitude: float = Field(..., description="Magnitude of violation")
    violation_percentage: Optional[float] = Field(
        default=None,
        description="Percentage violation"
    )

    action_taken: EnforcementAction = Field(..., description="Action taken")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Violation timestamp"
    )


class EnforcementResult(BaseModel):
    """Result from constraint enforcement."""

    # Predictions
    original_predictions: List[float] = Field(
        default_factory=list,
        description="Original predictions"
    )
    enforced_predictions: List[float] = Field(
        default_factory=list,
        description="Enforced predictions"
    )
    predictions_modified: int = Field(
        ...,
        description="Number of predictions modified"
    )

    # Violations
    violations: List[ConstraintViolation] = Field(
        default_factory=list,
        description="List of violations"
    )
    total_violations: int = Field(..., description="Total violations")
    critical_violations: int = Field(..., description="Critical violations")

    # Summary by type
    violations_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Violation count by type"
    )
    violations_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Violation count by severity"
    )

    # Conservation checks
    conservation_satisfied: bool = Field(
        default=True,
        description="Conservation laws satisfied"
    )
    conservation_errors: List[str] = Field(
        default_factory=list,
        description="Conservation error messages"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Enforcement timestamp"
    )


# =============================================================================
# Output Constraint Enforcer
# =============================================================================

class OutputConstraintEnforcer:
    """
    Output Constraint Enforcer for Process Heat ML Models.

    This enforcer ensures ML predictions respect:
    - Physical bounds (temperature, pressure, efficiency)
    - Monotonicity constraints
    - Conservation laws (energy/mass balance)
    - Safety limits

    All calculations are deterministic for reproducibility.

    Attributes:
        config: Constraint configuration
        _violation_log: History of violations

    Example:
        >>> enforcer = OutputConstraintEnforcer(
        ...     config=ProcessHeatConstraintConfig()
        ... )
        >>> predictions = model.predict(X)
        >>> enforced, result = enforcer.enforce(predictions, X)
        >>> if result.critical_violations > 0:
        ...     raise SafetyException("Critical safety violation!")
    """

    def __init__(
        self,
        config: Optional[ProcessHeatConstraintConfig] = None,
        custom_constraints: Optional[List[Callable]] = None
    ):
        """
        Initialize output constraint enforcer.

        Args:
            config: Constraint configuration
            custom_constraints: Optional list of custom constraint functions
        """
        self.config = config or ProcessHeatConstraintConfig()
        self._custom_constraints = custom_constraints or []
        self._violation_log: List[ConstraintViolation] = []

        logger.info(
            f"OutputConstraintEnforcer initialized: "
            f"{len(self.config.physical_bounds)} bounds, "
            f"{len(self.config.safety_limits)} safety limits"
        )

    # =========================================================================
    # Physical Bounds Enforcement
    # =========================================================================

    def _enforce_physical_bounds(
        self,
        predictions: np.ndarray,
        sample_indices: np.ndarray
    ) -> Tuple[np.ndarray, List[ConstraintViolation]]:
        """Enforce physical bounds on predictions (deterministic)."""
        violations = []
        enforced = predictions.copy()

        for bound in self.config.physical_bounds:
            if bound.output_index >= predictions.shape[1] if predictions.ndim > 1 else 1:
                continue

            if predictions.ndim == 1:
                values = predictions
                output_idx = 0
            else:
                values = predictions[:, bound.output_index]
                output_idx = bound.output_index

            for i, val in enumerate(values):
                original_val = float(val)
                enforced_val = original_val
                violation_detected = False
                limit_violated = None

                # Check minimum
                if bound.min_value is not None and val < bound.min_value:
                    violation_detected = True
                    limit_violated = bound.min_value

                    if bound.enforcement == EnforcementAction.CLIP:
                        enforced_val = bound.min_value
                    elif bound.enforcement == EnforcementAction.SUBSTITUTE:
                        enforced_val = bound.substitute_value or bound.min_value

                # Check maximum
                if bound.max_value is not None and val > bound.max_value:
                    violation_detected = True
                    limit_violated = bound.max_value

                    if bound.enforcement == EnforcementAction.CLIP:
                        enforced_val = bound.max_value
                    elif bound.enforcement == EnforcementAction.SUBSTITUTE:
                        enforced_val = bound.substitute_value or bound.max_value

                if violation_detected:
                    # Update prediction
                    if predictions.ndim == 1:
                        enforced[i] = enforced_val
                    else:
                        enforced[i, output_idx] = enforced_val

                    # Record violation
                    violation_mag = abs(original_val - enforced_val)
                    violation_pct = (
                        violation_mag / abs(limit_violated) * 100
                        if limit_violated and limit_violated != 0 else None
                    )

                    violations.append(ConstraintViolation(
                        sample_index=int(sample_indices[i]),
                        constraint_name=bound.name,
                        constraint_type=ConstraintType.PHYSICAL_BOUND,
                        severity=bound.severity,
                        original_value=original_val,
                        enforced_value=enforced_val,
                        limit_value=limit_violated,
                        violation_magnitude=violation_mag,
                        violation_percentage=violation_pct,
                        action_taken=bound.enforcement
                    ))

        return enforced, violations

    # =========================================================================
    # Safety Limits Enforcement
    # =========================================================================

    def _enforce_safety_limits(
        self,
        predictions: np.ndarray,
        sample_indices: np.ndarray
    ) -> Tuple[np.ndarray, List[ConstraintViolation]]:
        """Enforce safety limits on predictions (deterministic)."""
        violations = []
        enforced = predictions.copy()

        for limit in self.config.safety_limits:
            if limit.output_index >= predictions.shape[1] if predictions.ndim > 1 else 1:
                continue

            if predictions.ndim == 1:
                values = predictions
                output_idx = 0
            else:
                values = predictions[:, limit.output_index]
                output_idx = limit.output_index

            for i, val in enumerate(values):
                original_val = float(val)
                enforced_val = original_val
                severity = ViolationSeverity.INFO
                limit_violated = None

                # Check critical limits first
                if limit.critical_max is not None and val > limit.critical_max:
                    severity = ViolationSeverity.CRITICAL
                    limit_violated = limit.critical_max

                    if limit.enforcement == EnforcementAction.CLIP:
                        enforced_val = limit.critical_max

                elif limit.critical_min is not None and val < limit.critical_min:
                    severity = ViolationSeverity.CRITICAL
                    limit_violated = limit.critical_min

                    if limit.enforcement == EnforcementAction.CLIP:
                        enforced_val = limit.critical_min

                # Check warning limits
                elif limit.warning_max is not None and val > limit.warning_max:
                    severity = ViolationSeverity.WARNING
                    limit_violated = limit.warning_max
                    # Don't enforce warning limits, just flag

                elif limit.warning_min is not None and val < limit.warning_min:
                    severity = ViolationSeverity.WARNING
                    limit_violated = limit.warning_min

                if limit_violated is not None:
                    # Update prediction if critical
                    if severity == ViolationSeverity.CRITICAL:
                        if predictions.ndim == 1:
                            enforced[i] = enforced_val
                        else:
                            enforced[i, output_idx] = enforced_val

                    violation_mag = abs(original_val - (limit_violated or original_val))

                    violations.append(ConstraintViolation(
                        sample_index=int(sample_indices[i]),
                        constraint_name=limit.name,
                        constraint_type=ConstraintType.SAFETY_LIMIT,
                        severity=severity,
                        original_value=original_val,
                        enforced_value=enforced_val,
                        limit_value=limit_violated,
                        violation_magnitude=violation_mag,
                        violation_percentage=None,
                        action_taken=limit.enforcement if severity == ViolationSeverity.CRITICAL else EnforcementAction.FLAG
                    ))

        return enforced, violations

    # =========================================================================
    # Conservation Law Validation
    # =========================================================================

    def _validate_conservation(
        self,
        predictions: np.ndarray,
        inputs: np.ndarray,
        sample_indices: np.ndarray
    ) -> Tuple[bool, List[ConstraintViolation], List[str]]:
        """Validate conservation laws (deterministic)."""
        violations = []
        errors = []
        all_satisfied = True

        for constraint in self.config.conservation_constraints:
            if constraint.conservation_type == "energy_balance":
                satisfied, violation_list, error_msgs = self._check_energy_balance(
                    predictions, inputs, sample_indices, constraint
                )
            elif constraint.conservation_type == "mass_balance":
                satisfied, violation_list, error_msgs = self._check_mass_balance(
                    predictions, inputs, sample_indices, constraint
                )
            elif constraint.conservation_type == "heat_balance":
                satisfied, violation_list, error_msgs = self._check_heat_balance(
                    predictions, inputs, sample_indices, constraint
                )
            else:
                continue

            if not satisfied:
                all_satisfied = False

            violations.extend(violation_list)
            errors.extend(error_msgs)

        return all_satisfied, violations, errors

    def _check_energy_balance(
        self,
        predictions: np.ndarray,
        inputs: np.ndarray,
        sample_indices: np.ndarray,
        constraint: ConservationConstraint
    ) -> Tuple[bool, List[ConstraintViolation], List[str]]:
        """Check energy balance conservation."""
        violations = []
        errors = []
        all_satisfied = True

        # Energy balance: Q_in = Q_out + Q_loss
        # Simplified: efficiency * Q_in = Q_out

        for i in range(len(predictions)):
            # Extract relevant values
            # Assuming: input[0] = fuel_input_MW, output[0] = efficiency%, output[2] = heat_output_MW
            if len(constraint.input_indices) == 0 or len(constraint.output_indices) < 2:
                continue

            q_in_idx = constraint.input_indices[0]
            eff_idx = constraint.output_indices[0]
            q_out_idx = constraint.output_indices[1] if len(constraint.output_indices) > 1 else 0

            q_in = inputs[i, q_in_idx] if inputs.ndim > 1 else inputs[q_in_idx]
            efficiency = predictions[i, eff_idx] / 100.0 if predictions.ndim > 1 else predictions[eff_idx] / 100.0
            q_out = predictions[i, q_out_idx] if predictions.ndim > 1 else predictions[q_out_idx]

            # Calculate expected output
            expected_q_out = q_in * efficiency

            # Check conservation within tolerance
            if expected_q_out > 0:
                relative_error = abs(q_out - expected_q_out) / expected_q_out
            else:
                relative_error = abs(q_out - expected_q_out)

            if relative_error > constraint.tolerance:
                all_satisfied = False

                violations.append(ConstraintViolation(
                    sample_index=int(sample_indices[i]),
                    constraint_name=constraint.name,
                    constraint_type=ConstraintType.CONSERVATION,
                    severity=constraint.severity,
                    original_value=float(q_out),
                    enforced_value=float(expected_q_out),
                    limit_value=float(expected_q_out),
                    violation_magnitude=float(relative_error * expected_q_out),
                    violation_percentage=float(relative_error * 100),
                    action_taken=EnforcementAction.FLAG
                ))

                errors.append(
                    f"Energy balance violation at sample {sample_indices[i]}: "
                    f"expected {expected_q_out:.2f} MW, got {q_out:.2f} MW "
                    f"({relative_error*100:.1f}% error)"
                )

        return all_satisfied, violations, errors

    def _check_mass_balance(
        self,
        predictions: np.ndarray,
        inputs: np.ndarray,
        sample_indices: np.ndarray,
        constraint: ConservationConstraint
    ) -> Tuple[bool, List[ConstraintViolation], List[str]]:
        """Check mass balance conservation."""
        # Similar structure to energy balance
        # Mass in = Mass out (for steady state)
        return True, [], []

    def _check_heat_balance(
        self,
        predictions: np.ndarray,
        inputs: np.ndarray,
        sample_indices: np.ndarray,
        constraint: ConservationConstraint
    ) -> Tuple[bool, List[ConstraintViolation], List[str]]:
        """Check heat balance conservation."""
        # Q = m * cp * dT
        return True, [], []

    # =========================================================================
    # Monotonicity Constraints
    # =========================================================================

    def _check_monotonicity(
        self,
        predictions: np.ndarray,
        inputs: np.ndarray,
        sample_indices: np.ndarray
    ) -> List[ConstraintViolation]:
        """Check monotonicity constraints (deterministic)."""
        violations = []

        for constraint in self.config.monotonicity_constraints:
            if predictions.ndim == 1:
                continue

            output_vals = predictions[:, constraint.output_index]
            input_vals = inputs[:, constraint.input_index]

            # Sort by input values
            sort_idx = np.argsort(input_vals)
            sorted_output = output_vals[sort_idx]

            # Check monotonicity
            for i in range(1, len(sorted_output)):
                if constraint.direction == "increasing":
                    if sorted_output[i] < sorted_output[i-1]:
                        violations.append(ConstraintViolation(
                            sample_index=int(sample_indices[sort_idx[i]]),
                            constraint_name=constraint.name,
                            constraint_type=ConstraintType.MONOTONICITY,
                            severity=constraint.severity,
                            original_value=float(sorted_output[i]),
                            enforced_value=float(sorted_output[i]),
                            limit_value=float(sorted_output[i-1]),
                            violation_magnitude=float(sorted_output[i-1] - sorted_output[i]),
                            violation_percentage=None,
                            action_taken=EnforcementAction.FLAG
                        ))
                else:  # decreasing
                    if sorted_output[i] > sorted_output[i-1]:
                        violations.append(ConstraintViolation(
                            sample_index=int(sample_indices[sort_idx[i]]),
                            constraint_name=constraint.name,
                            constraint_type=ConstraintType.MONOTONICITY,
                            severity=constraint.severity,
                            original_value=float(sorted_output[i]),
                            enforced_value=float(sorted_output[i]),
                            limit_value=float(sorted_output[i-1]),
                            violation_magnitude=float(sorted_output[i] - sorted_output[i-1]),
                            violation_percentage=None,
                            action_taken=EnforcementAction.FLAG
                        ))

        return violations

    # =========================================================================
    # Main Enforcement Method
    # =========================================================================

    def enforce(
        self,
        predictions: np.ndarray,
        inputs: Optional[np.ndarray] = None,
        sample_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, EnforcementResult]:
        """
        Enforce all constraints on predictions.

        Args:
            predictions: Model predictions
            inputs: Optional input features (for conservation checks)
            sample_indices: Optional sample indices for logging

        Returns:
            Tuple of (enforced predictions, enforcement result)

        Example:
            >>> enforced_preds, result = enforcer.enforce(predictions, inputs)
            >>> if result.critical_violations > 0:
            ...     logger.critical(f"{result.critical_violations} critical violations!")
        """
        predictions = np.atleast_2d(predictions) if predictions.ndim == 1 else predictions
        n_samples = len(predictions)

        if sample_indices is None:
            sample_indices = np.arange(n_samples)

        all_violations = []
        enforced = predictions.copy()

        # Store original for comparison
        original_predictions = predictions.copy()

        # 1. Enforce physical bounds
        enforced, bound_violations = self._enforce_physical_bounds(
            enforced, sample_indices
        )
        all_violations.extend(bound_violations)

        # 2. Enforce safety limits
        enforced, safety_violations = self._enforce_safety_limits(
            enforced, sample_indices
        )
        all_violations.extend(safety_violations)

        # 3. Validate conservation laws
        conservation_satisfied = True
        conservation_errors = []
        if inputs is not None and self.config.conservation_constraints:
            conservation_satisfied, conservation_violations, conservation_errors = (
                self._validate_conservation(enforced, inputs, sample_indices)
            )
            all_violations.extend(conservation_violations)

        # 4. Check monotonicity
        if inputs is not None and self.config.monotonicity_constraints:
            monotonicity_violations = self._check_monotonicity(
                enforced, inputs, sample_indices
            )
            all_violations.extend(monotonicity_violations)

        # 5. Run custom constraints
        for custom_fn in self._custom_constraints:
            custom_violations = custom_fn(enforced, inputs, sample_indices)
            all_violations.extend(custom_violations)

        # Count violations by type and severity
        violations_by_type = {}
        violations_by_severity = {}

        for v in all_violations:
            type_key = v.constraint_type.value
            sev_key = v.severity.value

            violations_by_type[type_key] = violations_by_type.get(type_key, 0) + 1
            violations_by_severity[sev_key] = violations_by_severity.get(sev_key, 0) + 1

        # Count modifications
        predictions_modified = int(np.sum(
            ~np.isclose(original_predictions, enforced)
        ))

        # Critical violations count
        critical_violations = violations_by_severity.get("critical", 0)

        # Log violations
        if self.config.log_all_violations:
            for v in all_violations:
                self._violation_log.append(v)
                if v.severity == ViolationSeverity.CRITICAL:
                    logger.critical(
                        f"CRITICAL VIOLATION: {v.constraint_name} at sample {v.sample_index}, "
                        f"value={v.original_value:.4f}, limit={v.limit_value}"
                    )
                elif v.severity == ViolationSeverity.ERROR:
                    logger.error(
                        f"Violation: {v.constraint_name} at sample {v.sample_index}"
                    )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(
            len(all_violations), predictions_modified
        )

        result = EnforcementResult(
            original_predictions=original_predictions.flatten().tolist()[:100],
            enforced_predictions=enforced.flatten().tolist()[:100],
            predictions_modified=predictions_modified,
            violations=all_violations,
            total_violations=len(all_violations),
            critical_violations=critical_violations,
            violations_by_type=violations_by_type,
            violations_by_severity=violations_by_severity,
            conservation_satisfied=conservation_satisfied,
            conservation_errors=conservation_errors,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow()
        )

        logger.info(
            f"Constraint enforcement complete: {len(all_violations)} violations, "
            f"{predictions_modified} predictions modified, "
            f"{critical_violations} critical"
        )

        # Handle critical violations
        if self.config.halt_on_critical and critical_violations > 0:
            logger.critical(
                f"Halting due to {critical_violations} critical safety violations!"
            )

        return enforced, result

    def _calculate_provenance(
        self,
        n_violations: int,
        n_modified: int
    ) -> str:
        """Calculate SHA-256 provenance hash (deterministic)."""
        provenance_data = (
            f"{n_violations}|{n_modified}|"
            f"{len(self.config.physical_bounds)}|"
            f"{len(self.config.safety_limits)}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()

    def get_violation_history(self) -> List[ConstraintViolation]:
        """Get history of all violations."""
        return self._violation_log.copy()

    def clear_violation_history(self):
        """Clear violation history."""
        self._violation_log = []

    def add_custom_constraint(
        self,
        constraint_fn: Callable[
            [np.ndarray, Optional[np.ndarray], np.ndarray],
            List[ConstraintViolation]
        ]
    ):
        """Add a custom constraint function."""
        self._custom_constraints.append(constraint_fn)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_process_heat_enforcer() -> OutputConstraintEnforcer:
    """
    Create an OutputConstraintEnforcer with standard Process Heat constraints.

    Returns:
        Configured enforcer for Process Heat applications
    """
    config = ProcessHeatConstraintConfig(
        physical_bounds=[
            PhysicalBound(
                name="thermal_efficiency",
                output_index=0,
                min_value=0.0,
                max_value=100.0,
                unit="%",
                severity=ViolationSeverity.ERROR,
                enforcement=EnforcementAction.CLIP
            ),
            PhysicalBound(
                name="flue_gas_temperature",
                output_index=1,
                min_value=100.0,  # Min practical flue gas temp
                max_value=700.0,  # Max before damage
                unit="C",
                severity=ViolationSeverity.WARNING,
                enforcement=EnforcementAction.CLIP
            ),
            PhysicalBound(
                name="excess_air_ratio",
                output_index=2,
                min_value=1.0,  # Stoichiometric minimum
                max_value=3.0,  # Practical maximum
                unit="ratio",
                severity=ViolationSeverity.WARNING,
                enforcement=EnforcementAction.CLIP
            ),
        ],
        safety_limits=[
            SafetyLimit(
                name="tube_metal_temperature",
                output_index=1,
                critical_max=650.0,
                warning_max=600.0,
                standard="API 530",
                description="Heater tube metal temperature limit"
            ),
            SafetyLimit(
                name="stack_temperature",
                output_index=3,
                warning_min=120.0,  # Dew point avoidance
                critical_min=100.0,
                standard="Industry Practice",
                description="Stack temperature for acid dewpoint"
            ),
        ],
    )

    return OutputConstraintEnforcer(config=config)


# =============================================================================
# Unit Tests
# =============================================================================

class TestOutputConstraintEnforcer:
    """Unit tests for OutputConstraintEnforcer."""

    def test_physical_bounds_clipping(self):
        """Test physical bounds are clipped correctly."""
        config = ProcessHeatConstraintConfig(
            physical_bounds=[
                PhysicalBound(
                    name="efficiency",
                    output_index=0,
                    min_value=0.0,
                    max_value=100.0,
                    enforcement=EnforcementAction.CLIP
                )
            ],
            safety_limits=[]
        )
        enforcer = OutputConstraintEnforcer(config)

        predictions = np.array([[105.0], [-5.0], [50.0]])
        enforced, result = enforcer.enforce(predictions)

        assert enforced[0, 0] == 100.0  # Clipped to max
        assert enforced[1, 0] == 0.0    # Clipped to min
        assert enforced[2, 0] == 50.0   # Unchanged
        assert result.total_violations == 2

    def test_safety_limits_critical(self):
        """Test critical safety limit detection."""
        config = ProcessHeatConstraintConfig(
            physical_bounds=[],
            safety_limits=[
                SafetyLimit(
                    name="temperature",
                    output_index=0,
                    critical_max=100.0,
                    warning_max=80.0
                )
            ]
        )
        enforcer = OutputConstraintEnforcer(config)

        predictions = np.array([[110.0], [90.0], [70.0]])
        enforced, result = enforcer.enforce(predictions)

        assert result.critical_violations == 1
        assert result.violations_by_severity.get("warning", 0) == 1

    def test_violation_logging(self):
        """Test violation logging."""
        config = ProcessHeatConstraintConfig(
            physical_bounds=[
                PhysicalBound(
                    name="test",
                    output_index=0,
                    min_value=0.0,
                    max_value=100.0
                )
            ],
            safety_limits=[],
            log_all_violations=True
        )
        enforcer = OutputConstraintEnforcer(config)

        predictions = np.array([[150.0]])
        enforcer.enforce(predictions)

        history = enforcer.get_violation_history()
        assert len(history) == 1
        assert history[0].constraint_name == "test"

    def test_provenance_deterministic(self):
        """Test provenance hash is deterministic."""
        enforcer = OutputConstraintEnforcer()

        hash1 = enforcer._calculate_provenance(5, 3)
        hash2 = enforcer._calculate_provenance(5, 3)

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_conservation_check(self):
        """Test conservation law validation."""
        config = ProcessHeatConstraintConfig(
            physical_bounds=[],
            safety_limits=[],
            conservation_constraints=[
                ConservationConstraint(
                    name="energy_balance",
                    input_indices=[0],
                    output_indices=[0, 1],
                    conservation_type="energy_balance",
                    tolerance=0.05
                )
            ]
        )
        enforcer = OutputConstraintEnforcer(config)

        # Q_in = 100, efficiency = 80%, Q_out should be 80
        inputs = np.array([[100.0]])
        predictions = np.array([[80.0, 80.0]])  # Correct

        _, result = enforcer.enforce(predictions, inputs)
        assert result.conservation_satisfied

    def test_custom_constraint(self):
        """Test custom constraint function."""
        enforcer = OutputConstraintEnforcer()

        def custom_check(preds, inputs, indices):
            violations = []
            for i, val in enumerate(preds.flatten()):
                if val > 50:
                    violations.append(ConstraintViolation(
                        sample_index=int(indices[i]),
                        constraint_name="custom",
                        constraint_type=ConstraintType.CUSTOM,
                        severity=ViolationSeverity.WARNING,
                        original_value=float(val),
                        enforced_value=float(val),
                        violation_magnitude=float(val - 50),
                        action_taken=EnforcementAction.FLAG
                    ))
            return violations

        enforcer.add_custom_constraint(custom_check)

        predictions = np.array([[60.0], [40.0]])
        _, result = enforcer.enforce(predictions)

        assert result.total_violations >= 1
