"""
GL-004 Burnmaster - Constraint Management System

This module implements constraint handling for combustion control optimization.

Constraint Types:
    - HardConstraint: Safety limits, permit limits (must not violate)
    - SoftConstraint: Preferred operating windows (can trade off)
    - RateLimitConstraint: Actuator change rate limits
    - DeadbandConstraint: Prevent hunting/oscillations
    - ConstraintSet: Collection with violation detection and margin computation

Author: GreenLang Optimization Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import uuid

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of constraints."""
    HARD = "hard"
    SOFT = "soft"
    RATE_LIMIT = "rate_limit"
    DEADBAND = "deadband"
    EQUALITY = "equality"


class ConstraintStatus(str, Enum):
    """Status of constraint evaluation."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    AT_LIMIT = "at_limit"
    NEAR_LIMIT = "near_limit"


class ViolationSeverity(str, Enum):
    """Severity of constraint violation."""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    SAFETY = "safety"


class ConstraintEvaluation(BaseModel):
    """Result of constraint evaluation."""
    constraint_id: str = Field(..., description="Constraint identifier")
    constraint_name: str = Field(..., description="Constraint name")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    status: ConstraintStatus = Field(..., description="Constraint status")
    value: float = Field(..., description="Current value")
    limit_lower: Optional[float] = Field(default=None, description="Lower limit")
    limit_upper: Optional[float] = Field(default=None, description="Upper limit")
    margin: float = Field(default=0.0, description="Margin to nearest limit")
    margin_percent: float = Field(default=0.0, description="Margin as percent of range")
    violation_amount: float = Field(default=0.0, ge=0, description="Amount of violation")
    severity: ViolationSeverity = Field(default=ViolationSeverity.NONE)
    penalty: float = Field(default=0.0, ge=0, description="Penalty for violation")
    is_violated: bool = Field(default=False)
    evaluation_time_ms: float = Field(default=0.0, ge=0)


class ConstraintSetResult(BaseModel):
    """Result of evaluating a constraint set."""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluations: Dict[str, ConstraintEvaluation] = Field(default_factory=dict)
    total_violations: int = Field(default=0, ge=0)
    total_penalty: float = Field(default=0.0, ge=0)
    is_feasible: bool = Field(default=True)
    hard_violations: int = Field(default=0, ge=0)
    soft_violations: int = Field(default=0, ge=0)
    min_margin: float = Field(default=float("inf"))
    min_margin_constraint: Optional[str] = Field(default=None)
    max_violation: float = Field(default=0.0, ge=0)
    max_violation_constraint: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        if not self.provenance_hash:
            evals_str = "|".join([f"{k}:{v.status.value}" for k, v in self.evaluations.items()])
            hash_input = f"{self.result_id}|{self.total_violations}|{evals_str}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class BaseConstraint(ABC):
    """Abstract base class for constraints."""

    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType,
        variable_name: str,
        enabled: bool = True,
        description: str = ""
    ) -> None:
        self.constraint_id = str(uuid.uuid4())[:12]
        self.name = name
        self.constraint_type = constraint_type
        self.variable_name = variable_name
        self.enabled = enabled
        self.description = description

    @abstractmethod
    def evaluate(self, value: float, **kwargs) -> ConstraintEvaluation:
        """Evaluate constraint satisfaction."""
        pass

    @abstractmethod
    def get_margin(self, value: float) -> float:
        """Get margin to constraint boundary."""
        pass

    def _create_evaluation(
        self,
        value: float,
        status: ConstraintStatus,
        margin: float,
        violation_amount: float = 0.0,
        severity: ViolationSeverity = ViolationSeverity.NONE,
        penalty: float = 0.0,
        limit_lower: Optional[float] = None,
        limit_upper: Optional[float] = None,
        eval_time: float = 0.0
    ) -> ConstraintEvaluation:
        """Create a constraint evaluation result."""
        margin_percent = 0.0
        if limit_lower is not None and limit_upper is not None:
            range_val = limit_upper - limit_lower
            if range_val > 0:
                margin_percent = (margin / range_val) * 100

        return ConstraintEvaluation(
            constraint_id=self.constraint_id,
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            status=status,
            value=value,
            limit_lower=limit_lower,
            limit_upper=limit_upper,
            margin=margin,
            margin_percent=margin_percent,
            violation_amount=violation_amount,
            severity=severity,
            penalty=penalty,
            is_violated=status == ConstraintStatus.VIOLATED,
            evaluation_time_ms=eval_time
        )


class HardConstraint(BaseConstraint):
    """
    Hard constraint that must not be violated.

    Used for safety limits, permit limits, equipment limits.
    Any violation results in infeasibility.
    """

    def __init__(
        self,
        name: str,
        variable_name: str,
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        penalty_coefficient: float = 1000.0,
        severity: ViolationSeverity = ViolationSeverity.CRITICAL,
        enabled: bool = True,
        description: str = ""
    ) -> None:
        super().__init__(name, ConstraintType.HARD, variable_name, enabled, description)
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.penalty_coefficient = penalty_coefficient
        self.severity = severity

        if lower_limit is None and upper_limit is None:
            raise ValueError("At least one limit must be specified")

    def evaluate(self, value: float, **kwargs) -> ConstraintEvaluation:
        """Evaluate hard constraint."""
        start_time = datetime.now(timezone.utc)

        if not self.enabled:
            eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return self._create_evaluation(
                value=value, status=ConstraintStatus.SATISFIED,
                margin=float("inf"), limit_lower=self.lower_limit,
                limit_upper=self.upper_limit, eval_time=eval_time
            )

        # Check violations
        violation_amount = 0.0
        if self.lower_limit is not None and value < self.lower_limit:
            violation_amount = self.lower_limit - value
        elif self.upper_limit is not None and value > self.upper_limit:
            violation_amount = value - self.upper_limit

        margin = self.get_margin(value)
        is_violated = violation_amount > 0
        penalty = violation_amount * self.penalty_coefficient if is_violated else 0.0

        # Determine status
        if is_violated:
            status = ConstraintStatus.VIOLATED
        elif margin <= 0.01 * (abs(self.upper_limit or 0) + abs(self.lower_limit or 0) + 1):
            status = ConstraintStatus.AT_LIMIT
        elif margin <= 0.05 * (abs(self.upper_limit or 0) + abs(self.lower_limit or 0) + 1):
            status = ConstraintStatus.NEAR_LIMIT
        else:
            status = ConstraintStatus.SATISFIED

        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return self._create_evaluation(
            value=value, status=status, margin=margin,
            violation_amount=violation_amount,
            severity=self.severity if is_violated else ViolationSeverity.NONE,
            penalty=penalty, limit_lower=self.lower_limit,
            limit_upper=self.upper_limit, eval_time=eval_time
        )

    def get_margin(self, value: float) -> float:
        """Get margin to nearest limit."""
        margins = []
        if self.lower_limit is not None:
            margins.append(value - self.lower_limit)
        if self.upper_limit is not None:
            margins.append(self.upper_limit - value)
        return min(margins) if margins else float("inf")


class SoftConstraint(BaseConstraint):
    """
    Soft constraint representing preferred operating windows.

    Violations incur a penalty but do not cause infeasibility.
    Used for efficiency targets, comfort ranges, etc.
    """

    def __init__(
        self,
        name: str,
        variable_name: str,
        target_lower: Optional[float] = None,
        target_upper: Optional[float] = None,
        penalty_coefficient: float = 10.0,
        penalty_type: str = "linear",  # linear, quadratic
        enabled: bool = True,
        description: str = ""
    ) -> None:
        super().__init__(name, ConstraintType.SOFT, variable_name, enabled, description)
        self.target_lower = target_lower
        self.target_upper = target_upper
        self.penalty_coefficient = penalty_coefficient
        self.penalty_type = penalty_type

    def evaluate(self, value: float, **kwargs) -> ConstraintEvaluation:
        """Evaluate soft constraint."""
        start_time = datetime.now(timezone.utc)

        if not self.enabled:
            eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return self._create_evaluation(
                value=value, status=ConstraintStatus.SATISFIED,
                margin=float("inf"), limit_lower=self.target_lower,
                limit_upper=self.target_upper, eval_time=eval_time
            )

        # Calculate deviation from target range
        deviation = 0.0
        if self.target_lower is not None and value < self.target_lower:
            deviation = self.target_lower - value
        elif self.target_upper is not None and value > self.target_upper:
            deviation = value - self.target_upper

        # Calculate penalty
        if self.penalty_type == "quadratic":
            penalty = self.penalty_coefficient * deviation ** 2
        else:
            penalty = self.penalty_coefficient * deviation

        margin = self.get_margin(value)
        is_violated = deviation > 0

        status = ConstraintStatus.VIOLATED if is_violated else ConstraintStatus.SATISFIED

        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return self._create_evaluation(
            value=value, status=status, margin=margin,
            violation_amount=deviation,
            severity=ViolationSeverity.WARNING if is_violated else ViolationSeverity.NONE,
            penalty=penalty, limit_lower=self.target_lower,
            limit_upper=self.target_upper, eval_time=eval_time
        )

    def get_margin(self, value: float) -> float:
        """Get margin to target boundaries."""
        margins = []
        if self.target_lower is not None:
            margins.append(value - self.target_lower)
        if self.target_upper is not None:
            margins.append(self.target_upper - value)
        return min(margins) if margins else float("inf")


class RateLimitConstraint(BaseConstraint):
    """
    Rate limit constraint for actuator change rates.

    Limits how fast a variable can change per time unit.
    """

    def __init__(
        self,
        name: str,
        variable_name: str,
        max_rate_per_second: float,
        penalty_coefficient: float = 50.0,
        enabled: bool = True,
        description: str = ""
    ) -> None:
        super().__init__(name, ConstraintType.RATE_LIMIT, variable_name, enabled, description)
        self.max_rate_per_second = max_rate_per_second
        self.penalty_coefficient = penalty_coefficient
        self._previous_value: Optional[float] = None
        self._previous_time: Optional[datetime] = None

    def evaluate(
        self,
        value: float,
        previous_value: Optional[float] = None,
        dt_seconds: float = 1.0,
        **kwargs
    ) -> ConstraintEvaluation:
        """Evaluate rate limit constraint."""
        start_time = datetime.now(timezone.utc)

        if not self.enabled:
            eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return self._create_evaluation(
                value=value, status=ConstraintStatus.SATISFIED,
                margin=self.max_rate_per_second, eval_time=eval_time
            )

        # Use provided previous value or stored one
        prev = previous_value if previous_value is not None else self._previous_value

        if prev is None:
            # First evaluation, no rate to check
            self._previous_value = value
            self._previous_time = datetime.now(timezone.utc)
            eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return self._create_evaluation(
                value=value, status=ConstraintStatus.SATISFIED,
                margin=self.max_rate_per_second, eval_time=eval_time
            )

        # Calculate actual rate
        if dt_seconds <= 0:
            dt_seconds = 1.0
        actual_rate = abs(value - prev) / dt_seconds

        # Check violation
        rate_excess = max(0.0, actual_rate - self.max_rate_per_second)
        is_violated = rate_excess > 0
        penalty = rate_excess * self.penalty_coefficient if is_violated else 0.0
        margin = self.max_rate_per_second - actual_rate

        status = ConstraintStatus.VIOLATED if is_violated else ConstraintStatus.SATISFIED

        # Update stored values
        self._previous_value = value
        self._previous_time = datetime.now(timezone.utc)

        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return self._create_evaluation(
            value=actual_rate, status=status, margin=margin,
            violation_amount=rate_excess,
            severity=ViolationSeverity.WARNING if is_violated else ViolationSeverity.NONE,
            penalty=penalty, limit_upper=self.max_rate_per_second,
            eval_time=eval_time
        )

    def get_margin(self, value: float) -> float:
        """Get margin to rate limit."""
        return self.max_rate_per_second - abs(value)

    def reset(self) -> None:
        """Reset stored previous value."""
        self._previous_value = None
        self._previous_time = None


class DeadbandConstraint(BaseConstraint):
    """
    Deadband constraint to prevent hunting/oscillations.

    Penalizes changes smaller than a threshold to avoid
    excessive actuator movements.
    """

    def __init__(
        self,
        name: str,
        variable_name: str,
        deadband_size: float,
        penalty_coefficient: float = 5.0,
        enabled: bool = True,
        description: str = ""
    ) -> None:
        super().__init__(name, ConstraintType.DEADBAND, variable_name, enabled, description)
        self.deadband_size = deadband_size
        self.penalty_coefficient = penalty_coefficient
        self._reference_value: Optional[float] = None

    def evaluate(
        self,
        value: float,
        reference_value: Optional[float] = None,
        **kwargs
    ) -> ConstraintEvaluation:
        """Evaluate deadband constraint."""
        start_time = datetime.now(timezone.utc)

        if not self.enabled:
            eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return self._create_evaluation(
                value=value, status=ConstraintStatus.SATISFIED,
                margin=self.deadband_size, eval_time=eval_time
            )

        # Use provided reference or stored one
        ref = reference_value if reference_value is not None else self._reference_value

        if ref is None:
            self._reference_value = value
            eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return self._create_evaluation(
                value=value, status=ConstraintStatus.SATISFIED,
                margin=self.deadband_size, eval_time=eval_time
            )

        # Check if change is within deadband
        change = abs(value - ref)
        is_within_deadband = change < self.deadband_size

        # Penalize small changes that might cause hunting
        penalty = 0.0
        if is_within_deadband and change > 0:
            # Penalize changes that are too small but non-zero
            penalty = self.penalty_coefficient * (1.0 - change / self.deadband_size)

        status = ConstraintStatus.AT_LIMIT if is_within_deadband else ConstraintStatus.SATISFIED
        margin = self.deadband_size - change

        # Update reference if change exceeds deadband
        if not is_within_deadband:
            self._reference_value = value

        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return self._create_evaluation(
            value=change, status=status, margin=margin,
            penalty=penalty, limit_lower=0.0, limit_upper=self.deadband_size,
            eval_time=eval_time
        )

    def get_margin(self, value: float) -> float:
        """Get margin to deadband edge."""
        if self._reference_value is None:
            return self.deadband_size
        return self.deadband_size - abs(value - self._reference_value)

    def reset(self) -> None:
        """Reset reference value."""
        self._reference_value = None


class ConstraintSet:
    """
    Collection of constraints with unified evaluation.

    Provides:
    - Batch evaluation of all constraints
    - Violation detection and counting
    - Margin computation
    - Feasibility checking
    """

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.constraints: Dict[str, BaseConstraint] = {}
        self._constraint_order: List[str] = []

    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Add a constraint to the set."""
        self.constraints[constraint.constraint_id] = constraint
        self._constraint_order.append(constraint.constraint_id)
        logger.debug("Added constraint: %s (%s)", constraint.name, constraint.constraint_type.value)

    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint from the set."""
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]
            self._constraint_order.remove(constraint_id)
            return True
        return False

    def evaluate(self, values: Dict[str, float], **kwargs) -> ConstraintSetResult:
        """
        Evaluate all constraints.

        Args:
            values: Dictionary mapping variable names to values
            **kwargs: Additional arguments (dt_seconds, previous_values, etc.)

        Returns:
            ConstraintSetResult with all evaluations
        """
        evaluations: Dict[str, ConstraintEvaluation] = {}
        total_penalty = 0.0
        hard_violations = 0
        soft_violations = 0
        min_margin = float("inf")
        min_margin_constraint = None
        max_violation = 0.0
        max_violation_constraint = None

        previous_values = kwargs.get("previous_values", {})
        dt_seconds = kwargs.get("dt_seconds", 1.0)

        for cid in self._constraint_order:
            constraint = self.constraints[cid]
            if not constraint.enabled:
                continue

            value = values.get(constraint.variable_name)
            if value is None:
                logger.warning("No value for constraint variable: %s", constraint.variable_name)
                continue

            # Prepare kwargs for specific constraint types
            eval_kwargs = {}
            if isinstance(constraint, RateLimitConstraint):
                eval_kwargs["previous_value"] = previous_values.get(constraint.variable_name)
                eval_kwargs["dt_seconds"] = dt_seconds
            elif isinstance(constraint, DeadbandConstraint):
                eval_kwargs["reference_value"] = previous_values.get(constraint.variable_name)

            evaluation = constraint.evaluate(value, **eval_kwargs)
            evaluations[cid] = evaluation

            # Aggregate results
            total_penalty += evaluation.penalty

            if evaluation.is_violated:
                if constraint.constraint_type == ConstraintType.HARD:
                    hard_violations += 1
                else:
                    soft_violations += 1

            if evaluation.margin < min_margin:
                min_margin = evaluation.margin
                min_margin_constraint = constraint.name

            if evaluation.violation_amount > max_violation:
                max_violation = evaluation.violation_amount
                max_violation_constraint = constraint.name

        is_feasible = hard_violations == 0
        total_violations = hard_violations + soft_violations

        return ConstraintSetResult(
            evaluations=evaluations,
            total_violations=total_violations,
            total_penalty=total_penalty,
            is_feasible=is_feasible,
            hard_violations=hard_violations,
            soft_violations=soft_violations,
            min_margin=min_margin if min_margin != float("inf") else 0.0,
            min_margin_constraint=min_margin_constraint,
            max_violation=max_violation,
            max_violation_constraint=max_violation_constraint
        )

    def get_constraint(self, constraint_id: str) -> Optional[BaseConstraint]:
        """Get constraint by ID."""
        return self.constraints.get(constraint_id)

    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[BaseConstraint]:
        """Get all constraints of a specific type."""
        return [c for c in self.constraints.values() if c.constraint_type == constraint_type]

    def get_constraints_by_variable(self, variable_name: str) -> List[BaseConstraint]:
        """Get all constraints for a specific variable."""
        return [c for c in self.constraints.values() if c.variable_name == variable_name]

    def enable_all(self) -> None:
        """Enable all constraints."""
        for c in self.constraints.values():
            c.enabled = True

    def disable_all(self) -> None:
        """Disable all constraints."""
        for c in self.constraints.values():
            c.enabled = False

    def reset_all(self) -> None:
        """Reset all stateful constraints."""
        for c in self.constraints.values():
            if hasattr(c, "reset"):
                c.reset()


def create_combustion_constraint_set() -> ConstraintSet:
    """Create standard constraint set for combustion optimization."""
    cs = ConstraintSet("combustion_constraints")

    # Hard safety constraints
    cs.add_constraint(HardConstraint(
        name="O2_Lower_Safety", variable_name="o2_percent",
        lower_limit=0.5, severity=ViolationSeverity.SAFETY,
        description="Minimum O2 to prevent fuel-rich conditions"
    ))
    cs.add_constraint(HardConstraint(
        name="O2_Upper_Limit", variable_name="o2_percent",
        upper_limit=10.0, severity=ViolationSeverity.CRITICAL,
        description="Maximum O2 to prevent efficiency loss"
    ))
    cs.add_constraint(HardConstraint(
        name="CO_Permit_Limit", variable_name="co_ppm",
        upper_limit=200.0, severity=ViolationSeverity.CRITICAL,
        description="CO permit limit"
    ))
    cs.add_constraint(HardConstraint(
        name="NOx_Permit_Limit", variable_name="nox_ppm",
        upper_limit=100.0, severity=ViolationSeverity.CRITICAL,
        description="NOx permit limit"
    ))

    # Soft operating targets
    cs.add_constraint(SoftConstraint(
        name="O2_Optimal_Range", variable_name="o2_percent",
        target_lower=2.0, target_upper=4.0,
        description="Optimal O2 range for efficiency"
    ))
    cs.add_constraint(SoftConstraint(
        name="CO_Optimal", variable_name="co_ppm",
        target_upper=50.0,
        description="Target CO for good combustion"
    ))

    # Rate limits
    cs.add_constraint(RateLimitConstraint(
        name="O2_Rate_Limit", variable_name="o2_setpoint_percent",
        max_rate_per_second=0.5,
        description="O2 setpoint rate limit"
    ))
    cs.add_constraint(RateLimitConstraint(
        name="Damper_Rate_Limit", variable_name="air_damper_position",
        max_rate_per_second=5.0,
        description="Air damper rate limit"
    ))
    cs.add_constraint(RateLimitConstraint(
        name="Valve_Rate_Limit", variable_name="fuel_valve_position",
        max_rate_per_second=2.0,
        description="Fuel valve rate limit"
    ))

    # Deadbands
    cs.add_constraint(DeadbandConstraint(
        name="O2_Deadband", variable_name="o2_setpoint_percent",
        deadband_size=0.1,
        description="O2 setpoint deadband"
    ))
    cs.add_constraint(DeadbandConstraint(
        name="Damper_Deadband", variable_name="air_damper_position",
        deadband_size=0.5,
        description="Air damper deadband"
    ))

    return cs
