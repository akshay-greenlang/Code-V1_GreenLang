"""
GL-016 Waterguard Constraint Handler - Constraint Management

Constraint management for optimization with hard and soft constraints.
Handles chemistry limits, equipment limits, and constraint relaxation
for infeasibility detection.

Constraint Types:
    - Hard: Chemistry limits, equipment limits (must satisfy)
    - Soft: Target ranges (optimize within, with penalty)

Features:
    - Constraint relaxation for infeasibility detection
    - Constraint violation penalty functions
    - Constraint satisfaction monitoring

Reference Standards:
    - CTI STD-201 (Cooling Tower Water Treatment)
    - ASHRAE 188 (Legionella Risk Management)

Author: GreenLang Water Treatment Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ConstraintType(str, Enum):
    """Types of constraints."""
    HARD = "hard"
    SOFT = "soft"


class ConstraintCategory(str, Enum):
    """Categories of constraints."""
    CHEMISTRY = "chemistry"
    EQUIPMENT = "equipment"
    RAMP_RATE = "ramp_rate"
    SAFETY = "safety"
    OPERATIONAL = "operational"


class BoundType(str, Enum):
    """Types of bounds."""
    LOWER = "lower"
    UPPER = "upper"
    EQUALITY = "equality"
    RANGE = "range"


class ViolationSeverity(str, Enum):
    """Severity of constraint violations."""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# =============================================================================
# DATA MODELS
# =============================================================================

class ConstraintViolation(BaseModel):
    """Details of a constraint violation."""
    constraint_name: str
    constraint_type: ConstraintType
    category: ConstraintCategory
    severity: ViolationSeverity

    # Violation details
    current_value: float
    limit_value: float
    violation_amount: float
    violation_percent: float = 0.0

    # Penalty
    penalty_value: float = 0.0

    # Timestamp
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_critical(self) -> bool:
        """Check if violation is critical."""
        return self.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.EMERGENCY]


class ConstraintRelaxation(BaseModel):
    """Constraint relaxation for infeasibility handling."""
    constraint_name: str
    original_limit: float
    relaxed_limit: float
    relaxation_amount: float
    relaxation_cost: float = 0.0
    reason: str = ""


class ConstraintStatus(BaseModel):
    """Status of constraint evaluation."""
    constraint_name: str
    is_satisfied: bool
    margin: float  # Positive = satisfied, Negative = violated
    margin_percent: float = 0.0
    current_value: float
    limit_value: float


class ConstraintSummary(BaseModel):
    """Summary of all constraints."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_constraints: int
    satisfied_count: int
    violated_count: int
    hard_violations: int
    soft_violations: int

    statuses: List[ConstraintStatus] = Field(default_factory=list)
    violations: List[ConstraintViolation] = Field(default_factory=list)
    relaxations: List[ConstraintRelaxation] = Field(default_factory=list)

    total_penalty: float = 0.0
    is_feasible: bool = True

    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            provenance_str = (
                f"{self.timestamp.isoformat()}|{self.total_constraints}|"
                f"{self.satisfied_count}|{self.total_penalty:.4f}"
            )
            self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# CONSTRAINT CLASSES
# =============================================================================

class Constraint(ABC):
    """
    Abstract base class for constraints - ZERO HALLUCINATION.

    All constraints must be deterministic and produce
    reproducible results for the same inputs.
    """

    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType,
        category: ConstraintCategory,
        description: str = ""
    ):
        """
        Initialize constraint.

        Args:
            name: Constraint name
            constraint_type: HARD or SOFT
            category: Constraint category
            description: Human-readable description
        """
        self.name = name
        self.constraint_type = constraint_type
        self.category = category
        self.description = description
        self._is_active = True

    @abstractmethod
    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ConstraintStatus:
        """
        Evaluate constraint satisfaction - DETERMINISTIC.

        Args:
            decision_variables: Decision variable values
            context: Additional context

        Returns:
            ConstraintStatus with satisfaction details
        """
        pass

    @abstractmethod
    def get_penalty(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> float:
        """
        Get penalty for constraint violation - DETERMINISTIC.

        Args:
            decision_variables: Decision variable values
            context: Additional context

        Returns:
            Penalty value (0 if satisfied)
        """
        pass

    def activate(self) -> None:
        """Activate this constraint."""
        self._is_active = True

    def deactivate(self) -> None:
        """Deactivate this constraint."""
        self._is_active = False

    @property
    def is_active(self) -> bool:
        """Check if constraint is active."""
        return self._is_active


class HardConstraint(Constraint):
    """
    Hard constraint that must be satisfied - ZERO HALLUCINATION.

    Violation of hard constraints makes solution infeasible.
    """

    def __init__(
        self,
        name: str,
        category: ConstraintCategory,
        expression: Callable[[Dict[str, float], Dict[str, Any]], float],
        bound_type: BoundType,
        limit_value: float,
        variable_name: str = "",
        unit: str = "",
        description: str = ""
    ):
        """
        Initialize hard constraint.

        Args:
            name: Constraint name
            category: Constraint category
            expression: Function computing constraint value
            bound_type: Type of bound (LOWER, UPPER, etc.)
            limit_value: Limit value
            variable_name: Name of constrained variable
            unit: Unit of measurement
            description: Description
        """
        super().__init__(name, ConstraintType.HARD, category, description)
        self.expression = expression
        self.bound_type = bound_type
        self.limit_value = limit_value
        self.variable_name = variable_name
        self.unit = unit

    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ConstraintStatus:
        """Evaluate hard constraint - DETERMINISTIC."""
        if not self._is_active:
            return ConstraintStatus(
                constraint_name=self.name,
                is_satisfied=True,
                margin=float('inf'),
                current_value=0,
                limit_value=self.limit_value
            )

        # Compute current value
        current_value = self.expression(decision_variables, context)

        # Calculate margin based on bound type
        if self.bound_type == BoundType.LOWER:
            margin = current_value - self.limit_value
        elif self.bound_type == BoundType.UPPER:
            margin = self.limit_value - current_value
        elif self.bound_type == BoundType.EQUALITY:
            margin = -abs(current_value - self.limit_value)
        else:
            margin = 0.0

        is_satisfied = margin >= 0

        # Calculate margin percent
        if self.limit_value != 0:
            margin_percent = (margin / abs(self.limit_value)) * 100
        else:
            margin_percent = margin * 100

        return ConstraintStatus(
            constraint_name=self.name,
            is_satisfied=is_satisfied,
            margin=margin,
            margin_percent=margin_percent,
            current_value=current_value,
            limit_value=self.limit_value
        )

    def get_penalty(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> float:
        """
        Get penalty for hard constraint violation.

        Hard constraints return infinite penalty when violated.
        """
        status = self.evaluate(decision_variables, context)

        if status.is_satisfied:
            return 0.0
        else:
            return float('inf')  # Infinite penalty for hard constraint violation


class SoftConstraint(Constraint):
    """
    Soft constraint with penalty for violation - ZERO HALLUCINATION.

    Soft constraints can be violated with a penalty cost.
    Used for target ranges and preferences.
    """

    def __init__(
        self,
        name: str,
        category: ConstraintCategory,
        expression: Callable[[Dict[str, float], Dict[str, Any]], float],
        target_value: float,
        tolerance: float,
        penalty_weight: float = 1.0,
        penalty_type: str = "quadratic",
        variable_name: str = "",
        unit: str = "",
        description: str = ""
    ):
        """
        Initialize soft constraint.

        Args:
            name: Constraint name
            category: Constraint category
            expression: Function computing constraint value
            target_value: Target value
            tolerance: Acceptable deviation from target
            penalty_weight: Weight for penalty calculation
            penalty_type: Type of penalty (linear, quadratic)
            variable_name: Name of constrained variable
            unit: Unit of measurement
            description: Description
        """
        super().__init__(name, ConstraintType.SOFT, category, description)
        self.expression = expression
        self.target_value = target_value
        self.tolerance = tolerance
        self.penalty_weight = penalty_weight
        self.penalty_type = penalty_type
        self.variable_name = variable_name
        self.unit = unit

    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ConstraintStatus:
        """Evaluate soft constraint - DETERMINISTIC."""
        if not self._is_active:
            return ConstraintStatus(
                constraint_name=self.name,
                is_satisfied=True,
                margin=self.tolerance,
                current_value=self.target_value,
                limit_value=self.target_value
            )

        current_value = self.expression(decision_variables, context)
        deviation = abs(current_value - self.target_value)
        margin = self.tolerance - deviation
        is_satisfied = margin >= 0

        if self.target_value != 0:
            margin_percent = (margin / abs(self.target_value)) * 100
        else:
            margin_percent = margin * 100

        return ConstraintStatus(
            constraint_name=self.name,
            is_satisfied=is_satisfied,
            margin=margin,
            margin_percent=margin_percent,
            current_value=current_value,
            limit_value=self.target_value
        )

    def get_penalty(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> float:
        """
        Get penalty for soft constraint violation - DETERMINISTIC.

        Penalty is proportional to violation amount.
        """
        status = self.evaluate(decision_variables, context)

        if status.is_satisfied:
            return 0.0

        violation = abs(status.margin)  # Margin is negative when violated

        if self.penalty_type == "quadratic":
            penalty = self.penalty_weight * (violation ** 2)
        elif self.penalty_type == "linear":
            penalty = self.penalty_weight * violation
        else:
            penalty = self.penalty_weight * violation

        return penalty


# =============================================================================
# CONSTRAINT HANDLER
# =============================================================================

class ConstraintHandler:
    """
    Constraint handler for optimization - ZERO HALLUCINATION.

    Manages hard and soft constraints, evaluates satisfaction,
    and handles constraint relaxation for infeasibility.

    Example:
        >>> handler = ConstraintHandler()
        >>> handler.add_constraint(HardConstraint(
        ...     name="conductivity_max",
        ...     category=ConstraintCategory.CHEMISTRY,
        ...     expression=lambda d, c: c.get("conductivity", 1500),
        ...     bound_type=BoundType.UPPER,
        ...     limit_value=3000
        ... ))
        >>> summary = handler.evaluate_all(decision_vars, context)
        >>> if not summary.is_feasible:
        ...     relaxed = handler.relax_for_feasibility(decision_vars, context)
    """

    def __init__(self):
        """Initialize constraint handler."""
        self._constraints: Dict[str, Constraint] = {}
        self._relaxation_enabled = True
        self._max_relaxation_percent = 20.0  # Maximum 20% relaxation

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the handler."""
        self._constraints[constraint.name] = constraint
        logger.debug("Added constraint: %s (%s)", constraint.name, constraint.constraint_type.value)

    def remove_constraint(self, name: str) -> bool:
        """Remove a constraint by name."""
        if name in self._constraints:
            del self._constraints[name]
            return True
        return False

    def get_constraint(self, name: str) -> Optional[Constraint]:
        """Get constraint by name."""
        return self._constraints.get(name)

    def get_all_constraints(self) -> List[Constraint]:
        """Get all registered constraints."""
        return list(self._constraints.values())

    def get_hard_constraints(self) -> List[Constraint]:
        """Get all hard constraints."""
        return [c for c in self._constraints.values() if c.constraint_type == ConstraintType.HARD]

    def get_soft_constraints(self) -> List[Constraint]:
        """Get all soft constraints."""
        return [c for c in self._constraints.values() if c.constraint_type == ConstraintType.SOFT]

    def evaluate_all(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ConstraintSummary:
        """
        Evaluate all constraints - DETERMINISTIC.

        Args:
            decision_variables: Decision variable values
            context: Additional context

        Returns:
            ConstraintSummary with all constraint statuses
        """
        statuses = []
        violations = []
        total_penalty = 0.0
        satisfied_count = 0
        hard_violations = 0
        soft_violations = 0

        for constraint in self._constraints.values():
            if not constraint.is_active:
                continue

            status = constraint.evaluate(decision_variables, context)
            statuses.append(status)

            if status.is_satisfied:
                satisfied_count += 1
            else:
                # Create violation record
                violation = ConstraintViolation(
                    constraint_name=constraint.name,
                    constraint_type=constraint.constraint_type,
                    category=constraint.category,
                    severity=self._determine_severity(status),
                    current_value=status.current_value,
                    limit_value=status.limit_value,
                    violation_amount=abs(status.margin),
                    violation_percent=abs(status.margin_percent),
                    penalty_value=constraint.get_penalty(decision_variables, context)
                )
                violations.append(violation)

                if constraint.constraint_type == ConstraintType.HARD:
                    hard_violations += 1
                else:
                    soft_violations += 1

            # Accumulate penalty
            penalty = constraint.get_penalty(decision_variables, context)
            if penalty < float('inf'):
                total_penalty += penalty

        is_feasible = hard_violations == 0

        return ConstraintSummary(
            total_constraints=len([c for c in self._constraints.values() if c.is_active]),
            satisfied_count=satisfied_count,
            violated_count=len(violations),
            hard_violations=hard_violations,
            soft_violations=soft_violations,
            statuses=statuses,
            violations=violations,
            total_penalty=total_penalty,
            is_feasible=is_feasible
        )

    def _determine_severity(self, status: ConstraintStatus) -> ViolationSeverity:
        """Determine violation severity based on margin."""
        if status.is_satisfied:
            return ViolationSeverity.NONE

        violation_pct = abs(status.margin_percent)

        if violation_pct > 50:
            return ViolationSeverity.EMERGENCY
        elif violation_pct > 20:
            return ViolationSeverity.CRITICAL
        elif violation_pct > 5:
            return ViolationSeverity.WARNING
        else:
            return ViolationSeverity.WARNING

    def get_total_penalty(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> float:
        """Get total penalty from all constraints - DETERMINISTIC."""
        total = 0.0

        for constraint in self._constraints.values():
            if not constraint.is_active:
                continue

            penalty = constraint.get_penalty(decision_variables, context)
            if penalty < float('inf'):
                total += penalty
            else:
                return float('inf')  # Any hard constraint violation

        return total

    def is_feasible(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> bool:
        """Check if solution is feasible (all hard constraints satisfied)."""
        for constraint in self._constraints.values():
            if not constraint.is_active:
                continue

            if constraint.constraint_type == ConstraintType.HARD:
                status = constraint.evaluate(decision_variables, context)
                if not status.is_satisfied:
                    return False

        return True

    def find_binding_constraints(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any],
        margin_threshold: float = 0.05
    ) -> List[str]:
        """
        Find constraints that are binding (close to their limits).

        Args:
            decision_variables: Decision variable values
            context: Context
            margin_threshold: Threshold for considering constraint binding

        Returns:
            List of binding constraint names
        """
        binding = []

        for constraint in self._constraints.values():
            if not constraint.is_active:
                continue

            status = constraint.evaluate(decision_variables, context)

            # Binding if margin is small (satisfied but close to limit)
            if status.is_satisfied and abs(status.margin_percent) < margin_threshold * 100:
                binding.append(constraint.name)

        return binding

    def relax_for_feasibility(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any],
        relaxation_cost_weight: float = 1000.0
    ) -> Tuple[bool, List[ConstraintRelaxation]]:
        """
        Attempt to relax constraints to achieve feasibility - DETERMINISTIC.

        Only relaxes hard constraints that are violated.
        Returns relaxation information with associated costs.

        Args:
            decision_variables: Decision variable values
            context: Context
            relaxation_cost_weight: Cost per unit of relaxation

        Returns:
            Tuple of (is_now_feasible, list of relaxations)
        """
        if not self._relaxation_enabled:
            return False, []

        relaxations = []

        for constraint in self._constraints.values():
            if not constraint.is_active:
                continue

            if constraint.constraint_type != ConstraintType.HARD:
                continue

            if not isinstance(constraint, HardConstraint):
                continue

            status = constraint.evaluate(decision_variables, context)

            if status.is_satisfied:
                continue

            # Calculate required relaxation
            violation = abs(status.margin)
            original_limit = constraint.limit_value

            # Calculate relaxed limit
            if constraint.bound_type == BoundType.UPPER:
                relaxed_limit = original_limit + violation * 1.1  # 10% margin
            elif constraint.bound_type == BoundType.LOWER:
                relaxed_limit = original_limit - violation * 1.1
            else:
                continue

            # Check if relaxation is within allowed bounds
            max_relax = abs(original_limit) * (self._max_relaxation_percent / 100)
            relaxation_amount = abs(relaxed_limit - original_limit)

            if relaxation_amount > max_relax:
                logger.warning(
                    "Constraint %s requires %.2f relaxation, max allowed is %.2f",
                    constraint.name, relaxation_amount, max_relax
                )
                continue

            # Calculate relaxation cost
            relaxation_cost = relaxation_amount * relaxation_cost_weight

            relaxations.append(ConstraintRelaxation(
                constraint_name=constraint.name,
                original_limit=original_limit,
                relaxed_limit=relaxed_limit,
                relaxation_amount=relaxation_amount,
                relaxation_cost=relaxation_cost,
                reason=f"Violated by {violation:.2f}"
            ))

            # Apply relaxation temporarily
            constraint.limit_value = relaxed_limit

        # Check if now feasible
        is_feasible = self.is_feasible(decision_variables, context)

        # Restore original limits
        for relaxation in relaxations:
            constraint = self._constraints.get(relaxation.constraint_name)
            if constraint and isinstance(constraint, HardConstraint):
                constraint.limit_value = relaxation.original_limit

        return is_feasible, relaxations

    def set_relaxation_enabled(self, enabled: bool) -> None:
        """Enable or disable constraint relaxation."""
        self._relaxation_enabled = enabled

    def set_max_relaxation_percent(self, percent: float) -> None:
        """Set maximum allowed relaxation percentage."""
        self._max_relaxation_percent = max(0, min(100, percent))


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_chemistry_constraints() -> List[Constraint]:
    """
    Create standard chemistry constraints for water treatment.

    Returns list of hard constraints for chemistry limits.
    """
    constraints = []

    # Conductivity limits
    constraints.append(HardConstraint(
        name="conductivity_max",
        category=ConstraintCategory.CHEMISTRY,
        expression=lambda d, c: c.get("conductivity_us_cm", 1500),
        bound_type=BoundType.UPPER,
        limit_value=3000,
        variable_name="conductivity",
        unit="uS/cm",
        description="Maximum conductivity limit"
    ))

    constraints.append(HardConstraint(
        name="conductivity_min",
        category=ConstraintCategory.CHEMISTRY,
        expression=lambda d, c: c.get("conductivity_us_cm", 1500),
        bound_type=BoundType.LOWER,
        limit_value=500,
        variable_name="conductivity",
        unit="uS/cm",
        description="Minimum conductivity limit"
    ))

    # pH limits
    constraints.append(HardConstraint(
        name="ph_max",
        category=ConstraintCategory.CHEMISTRY,
        expression=lambda d, c: c.get("ph", 8.0),
        bound_type=BoundType.UPPER,
        limit_value=9.0,
        variable_name="pH",
        unit="pH",
        description="Maximum pH limit"
    ))

    constraints.append(HardConstraint(
        name="ph_min",
        category=ConstraintCategory.CHEMISTRY,
        expression=lambda d, c: c.get("ph", 8.0),
        bound_type=BoundType.LOWER,
        limit_value=7.0,
        variable_name="pH",
        unit="pH",
        description="Minimum pH limit"
    ))

    # LSI limits
    constraints.append(HardConstraint(
        name="lsi_max",
        category=ConstraintCategory.CHEMISTRY,
        expression=lambda d, c: c.get("lsi", 0.0),
        bound_type=BoundType.UPPER,
        limit_value=1.0,
        variable_name="LSI",
        unit="",
        description="Maximum LSI (scaling tendency)"
    ))

    constraints.append(HardConstraint(
        name="lsi_min",
        category=ConstraintCategory.CHEMISTRY,
        expression=lambda d, c: c.get("lsi", 0.0),
        bound_type=BoundType.LOWER,
        limit_value=-1.0,
        variable_name="LSI",
        unit="",
        description="Minimum LSI (corrosion tendency)"
    ))

    return constraints


def create_equipment_constraints() -> List[Constraint]:
    """
    Create standard equipment constraints for water treatment.

    Returns list of hard constraints for equipment limits.
    """
    constraints = []

    # Blowdown valve limits
    constraints.append(HardConstraint(
        name="blowdown_max",
        category=ConstraintCategory.EQUIPMENT,
        expression=lambda d, c: d.get("blowdown_pct", 30),
        bound_type=BoundType.UPPER,
        limit_value=100,
        variable_name="blowdown_valve",
        unit="%",
        description="Maximum blowdown valve position"
    ))

    constraints.append(HardConstraint(
        name="blowdown_min",
        category=ConstraintCategory.EQUIPMENT,
        expression=lambda d, c: d.get("blowdown_pct", 30),
        bound_type=BoundType.LOWER,
        limit_value=0,
        variable_name="blowdown_valve",
        unit="%",
        description="Minimum blowdown valve position"
    ))

    # Dosing pump limits
    constraints.append(HardConstraint(
        name="scale_pump_max",
        category=ConstraintCategory.EQUIPMENT,
        expression=lambda d, c: d.get("scale_inhibitor_pct", 0),
        bound_type=BoundType.UPPER,
        limit_value=100,
        variable_name="scale_pump",
        unit="%",
        description="Maximum scale inhibitor pump speed"
    ))

    constraints.append(HardConstraint(
        name="corrosion_pump_max",
        category=ConstraintCategory.EQUIPMENT,
        expression=lambda d, c: d.get("corrosion_inhibitor_pct", 0),
        bound_type=BoundType.UPPER,
        limit_value=100,
        variable_name="corrosion_pump",
        unit="%",
        description="Maximum corrosion inhibitor pump speed"
    ))

    return constraints


def create_ramp_rate_constraints(
    max_blowdown_ramp: float = 10.0,
    max_dosing_ramp: float = 5.0
) -> List[Constraint]:
    """
    Create ramp rate constraints.

    Args:
        max_blowdown_ramp: Maximum blowdown change per minute (%)
        max_dosing_ramp: Maximum dosing change per minute (%)

    Returns list of hard constraints for ramp rates.
    """
    constraints = []

    constraints.append(HardConstraint(
        name="blowdown_ramp_rate",
        category=ConstraintCategory.RAMP_RATE,
        expression=lambda d, c: abs(d.get("blowdown_pct", 30) - c.get("prev_blowdown_pct", 30)),
        bound_type=BoundType.UPPER,
        limit_value=max_blowdown_ramp * c.get("time_step_min", 1) if 'c' in dir() else max_blowdown_ramp,
        variable_name="blowdown_ramp",
        unit="%/step",
        description="Maximum blowdown ramp rate"
    ))

    return constraints


def create_default_constraint_handler() -> ConstraintHandler:
    """Create constraint handler with standard water treatment constraints."""
    handler = ConstraintHandler()

    # Add chemistry constraints
    for c in create_chemistry_constraints():
        handler.add_constraint(c)

    # Add equipment constraints
    for c in create_equipment_constraints():
        handler.add_constraint(c)

    logger.info("Created constraint handler with %d constraints",
               len(handler.get_all_constraints()))

    return handler
