"""
Optimization Schemas - Data models for combustion optimization.

This module defines Pydantic models for optimization objectives, constraints,
results, and setpoint recommendations. These schemas support the optimization
engine that minimizes fuel cost and emissions while maintaining safety.

Example:
    >>> from optimization_schemas import OptimizationObjective, Constraint
    >>> objective = OptimizationObjective(
    ...     fuel_cost_weight=0.6,
    ...     emissions_cost_weight=0.3,
    ...     stability_penalty_weight=0.1
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import hashlib


class ObjectiveType(str, Enum):
    """Type of optimization objective."""
    MINIMIZE_FUEL_COST = "minimize_fuel_cost"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    MINIMIZE_TOTAL_COST = "minimize_total_cost"
    PARETO_OPTIMAL = "pareto_optimal"


class ConstraintType(str, Enum):
    """Type of optimization constraint."""
    HARD = "hard"  # Must be satisfied (violation causes infeasibility)
    SOFT = "soft"  # Preferred but can be violated with penalty


class ConstraintOperator(str, Enum):
    """Operator for constraint comparison."""
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    EQUAL = "eq"
    RANGE = "range"


class OptimizationStatus(str, Enum):
    """Status of optimization solution."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    ERROR = "error"


class OptimizationObjective(BaseModel):
    """
    Optimization objective function definition.

    Defines the weights for different cost components in the objective function:
    minimize(fuel_cost_weight * fuel_cost + emissions_cost_weight * emissions_cost
             + stability_penalty_weight * stability_penalty)

    Attributes:
        fuel_cost_weight: Weight for fuel cost component ($/hr)
        emissions_cost_weight: Weight for emissions cost component ($/ton CO2e)
        stability_penalty_weight: Weight for stability penalty
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    objective_id: str = Field(default="default", min_length=1, max_length=50, description="Unique identifier for this objective")
    objective_type: ObjectiveType = Field(default=ObjectiveType.MINIMIZE_TOTAL_COST, description="Type of optimization objective")
    fuel_cost_weight: float = Field(default=1.0, ge=0.0, le=100.0, description="Weight for fuel cost in objective function")
    emissions_cost_weight: float = Field(default=1.0, ge=0.0, le=100.0, description="Weight for emissions cost in objective function")
    stability_penalty_weight: float = Field(default=1.0, ge=0.0, le=100.0, description="Weight for stability penalty in objective function")
    efficiency_bonus_weight: float = Field(default=0.0, ge=0.0, le=100.0, description="Weight for efficiency bonus (negative cost)")

    # Cost parameters
    fuel_price_per_mj: float = Field(default=0.01, ge=0.0, le=1.0, description="Fuel price in $/MJ")
    co2_price_per_ton: float = Field(default=50.0, ge=0.0, le=1000.0, description="CO2 price in $/ton")
    nox_price_per_ton: float = Field(default=5000.0, ge=0.0, le=100000.0, description="NOx price in $/ton")

    # Normalization
    normalize_weights: bool = Field(default=True, description="Whether to normalize weights to sum to 1.0")

    @computed_field
    @property
    def total_weight(self) -> float:
        """Calculate total weight for normalization."""
        return (self.fuel_cost_weight + self.emissions_cost_weight +
                self.stability_penalty_weight + self.efficiency_bonus_weight)

    @computed_field
    @property
    def normalized_fuel_weight(self) -> float:
        """Get normalized fuel cost weight."""
        if self.normalize_weights and self.total_weight > 0:
            return self.fuel_cost_weight / self.total_weight
        return self.fuel_cost_weight

    @computed_field
    @property
    def normalized_emissions_weight(self) -> float:
        """Get normalized emissions cost weight."""
        if self.normalize_weights and self.total_weight > 0:
            return self.emissions_cost_weight / self.total_weight
        return self.emissions_cost_weight

    def calculate_total_cost(
        self,
        fuel_cost: float,
        emissions_cost: float,
        stability_penalty: float,
        efficiency_bonus: float = 0.0
    ) -> float:
        """
        Calculate total weighted cost.

        Args:
            fuel_cost: Fuel cost component
            emissions_cost: Emissions cost component
            stability_penalty: Stability penalty component
            efficiency_bonus: Efficiency bonus (subtracted from cost)

        Returns:
            Total weighted cost
        """
        if self.normalize_weights and self.total_weight > 0:
            return (
                self.normalized_fuel_weight * fuel_cost +
                self.normalized_emissions_weight * emissions_cost +
                (self.stability_penalty_weight / self.total_weight) * stability_penalty -
                (self.efficiency_bonus_weight / self.total_weight) * efficiency_bonus
            )
        return (
            self.fuel_cost_weight * fuel_cost +
            self.emissions_cost_weight * emissions_cost +
            self.stability_penalty_weight * stability_penalty -
            self.efficiency_bonus_weight * efficiency_bonus
        )


class Constraint(BaseModel):
    """
    Single optimization constraint.

    Represents a constraint on an optimization variable with hard/soft classification
    and configurable violation margins.

    Attributes:
        constraint_type: Hard (must satisfy) or Soft (prefer to satisfy)
        parameter: Name of the constrained parameter
        limit: Constraint limit value
        margin: Safety margin (for soft constraints, violation buffer)
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    constraint_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier for this constraint")
    constraint_type: ConstraintType = Field(default=ConstraintType.HARD, description="Type of constraint (hard/soft)")
    parameter: str = Field(..., min_length=1, max_length=100, description="Name of the constrained parameter")
    operator: ConstraintOperator = Field(..., description="Comparison operator for the constraint")
    limit: float = Field(..., description="Primary constraint limit value")
    limit_upper: Optional[float] = Field(default=None, description="Upper limit for range constraints")
    margin: float = Field(default=0.0, ge=0.0, description="Safety margin or buffer zone")
    unit: str = Field(default="", max_length=50, description="Engineering unit for the parameter")

    # Soft constraint parameters
    violation_penalty: float = Field(default=1000.0, ge=0.0, description="Penalty multiplier for soft constraint violation")
    max_violation: Optional[float] = Field(default=None, ge=0.0, description="Maximum allowable violation for soft constraints")

    # Metadata
    description: str = Field(default="", max_length=500, description="Human-readable description of the constraint")
    source: str = Field(default="", max_length=200, description="Source of this constraint (regulation, safety, etc.)")
    active: bool = Field(default=True, description="Whether this constraint is currently active")

    @model_validator(mode='after')
    def validate_range_constraint(self) -> 'Constraint':
        """Validate range constraint has both limits."""
        if self.operator == ConstraintOperator.RANGE and self.limit_upper is None:
            raise ValueError("Range constraint requires limit_upper")
        if self.operator == ConstraintOperator.RANGE and self.limit >= self.limit_upper:
            raise ValueError("For range constraint, limit must be less than limit_upper")
        return self

    def is_satisfied(self, value: float) -> bool:
        """
        Check if value satisfies the constraint.

        Args:
            value: Value to check against constraint

        Returns:
            True if constraint is satisfied
        """
        if self.operator == ConstraintOperator.LESS_THAN:
            return value < self.limit
        elif self.operator == ConstraintOperator.LESS_THAN_OR_EQUAL:
            return value <= self.limit
        elif self.operator == ConstraintOperator.GREATER_THAN:
            return value > self.limit
        elif self.operator == ConstraintOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.limit
        elif self.operator == ConstraintOperator.EQUAL:
            return abs(value - self.limit) <= self.margin
        elif self.operator == ConstraintOperator.RANGE:
            return self.limit <= value <= self.limit_upper
        return False

    def get_violation(self, value: float) -> float:
        """
        Calculate constraint violation amount.

        Args:
            value: Value to check

        Returns:
            Violation amount (0 if satisfied, positive if violated)
        """
        if self.operator == ConstraintOperator.LESS_THAN:
            return max(0.0, value - self.limit)
        elif self.operator == ConstraintOperator.LESS_THAN_OR_EQUAL:
            return max(0.0, value - self.limit)
        elif self.operator == ConstraintOperator.GREATER_THAN:
            return max(0.0, self.limit - value)
        elif self.operator == ConstraintOperator.GREATER_THAN_OR_EQUAL:
            return max(0.0, self.limit - value)
        elif self.operator == ConstraintOperator.EQUAL:
            return max(0.0, abs(value - self.limit) - self.margin)
        elif self.operator == ConstraintOperator.RANGE:
            if value < self.limit:
                return self.limit - value
            elif value > self.limit_upper:
                return value - self.limit_upper
        return 0.0

    def get_penalty(self, value: float) -> float:
        """Calculate penalty for constraint violation."""
        violation = self.get_violation(value)
        if self.constraint_type == ConstraintType.HARD:
            return float('inf') if violation > 0 else 0.0
        return violation * self.violation_penalty


class ConstraintSet(BaseModel):
    """
    Collection of optimization constraints.

    Provides methods to validate all constraints and aggregate violations.
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    constraint_set_id: str = Field(default="default", min_length=1, max_length=50, description="Unique identifier for this constraint set")
    name: str = Field(default="Default Constraints", max_length=200, description="Human-readable name for this constraint set")
    constraints: List[Constraint] = Field(default_factory=list, description="List of constraints in this set")
    version: str = Field(default="1.0", description="Version of this constraint set")
    effective_date: Optional[datetime] = Field(default=None, description="Date when this constraint set became effective")

    @field_validator('constraints')
    @classmethod
    def validate_unique_ids(cls, v: List[Constraint]) -> List[Constraint]:
        """Ensure no duplicate constraint IDs."""
        ids = [c.constraint_id for c in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate constraint IDs detected")
        return v

    def get_constraint(self, constraint_id: str) -> Optional[Constraint]:
        """Get a constraint by ID."""
        for constraint in self.constraints:
            if constraint.constraint_id == constraint_id:
                return constraint
        return None

    def get_constraints_for_parameter(self, parameter: str) -> List[Constraint]:
        """Get all constraints for a specific parameter."""
        return [c for c in self.constraints if c.parameter == parameter and c.active]

    def check_all(self, values: Dict[str, float]) -> Dict[str, bool]:
        """
        Check all constraints against provided values.

        Args:
            values: Dictionary mapping parameter names to values

        Returns:
            Dictionary mapping constraint IDs to satisfaction status
        """
        results = {}
        for constraint in self.constraints:
            if constraint.active and constraint.parameter in values:
                results[constraint.constraint_id] = constraint.is_satisfied(values[constraint.parameter])
        return results

    def is_feasible(self, values: Dict[str, float]) -> bool:
        """Check if all hard constraints are satisfied."""
        for constraint in self.constraints:
            if constraint.active and constraint.constraint_type == ConstraintType.HARD:
                if constraint.parameter in values:
                    if not constraint.is_satisfied(values[constraint.parameter]):
                        return False
        return True

    def get_total_penalty(self, values: Dict[str, float]) -> float:
        """Calculate total penalty for all constraint violations."""
        total_penalty = 0.0
        for constraint in self.constraints:
            if constraint.active and constraint.parameter in values:
                penalty = constraint.get_penalty(values[constraint.parameter])
                if penalty == float('inf'):
                    return float('inf')
                total_penalty += penalty
        return total_penalty

    def get_active_constraints(self) -> List[Constraint]:
        """Get all active constraints."""
        return [c for c in self.constraints if c.active]

    def get_hard_constraints(self) -> List[Constraint]:
        """Get all hard constraints."""
        return [c for c in self.constraints if c.constraint_type == ConstraintType.HARD and c.active]

    def get_soft_constraints(self) -> List[Constraint]:
        """Get all soft constraints."""
        return [c for c in self.constraints if c.constraint_type == ConstraintType.SOFT and c.active]


class SetpointValue(BaseModel):
    """Single setpoint value with metadata."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    tag: str = Field(..., min_length=1, max_length=100, description="Tag/variable name for this setpoint")
    value: float = Field(..., description="Optimal setpoint value")
    unit: str = Field(default="", max_length=50, description="Engineering unit")
    previous_value: Optional[float] = Field(default=None, description="Previous setpoint value before optimization")
    change_percent: Optional[float] = Field(default=None, description="Percentage change from previous value")
    min_bound: Optional[float] = Field(default=None, description="Minimum allowed value")
    max_bound: Optional[float] = Field(default=None, description="Maximum allowed value")


class BindingConstraint(BaseModel):
    """Information about a binding constraint at the optimum."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    constraint_id: str = Field(..., description="ID of the binding constraint")
    parameter: str = Field(..., description="Parameter name")
    limit: float = Field(..., description="Constraint limit value")
    optimal_value: float = Field(..., description="Value at optimal solution")
    shadow_price: Optional[float] = Field(default=None, description="Shadow price (dual value) of constraint")
    margin: float = Field(default=0.0, description="Distance from constraint limit")


class OptimizationResult(BaseModel):
    """
    Complete result from optimization solver.

    Contains optimal setpoints, objective value, binding constraints,
    and solution metadata.

    Attributes:
        optimal_setpoints: Dictionary of optimal setpoint values
        objective_value: Optimal objective function value
        binding_constraints: List of constraints that are binding at optimum
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    result_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S"), description="Unique identifier for this result")
    status: OptimizationStatus = Field(..., description="Status of optimization solution")
    optimal_setpoints: List[SetpointValue] = Field(default_factory=list, description="List of optimal setpoint values")
    objective_value: float = Field(..., description="Optimal objective function value")

    # Cost breakdown
    fuel_cost: Optional[float] = Field(default=None, ge=0.0, description="Fuel cost component at optimum")
    emissions_cost: Optional[float] = Field(default=None, ge=0.0, description="Emissions cost component at optimum")
    stability_penalty: Optional[float] = Field(default=None, ge=0.0, description="Stability penalty at optimum")
    total_cost_per_hour: Optional[float] = Field(default=None, description="Total operating cost in $/hr")

    # Constraint information
    binding_constraints: List[BindingConstraint] = Field(default_factory=list, description="Constraints that are binding at optimum")
    violated_constraints: List[str] = Field(default_factory=list, description="IDs of violated soft constraints")

    # Performance metrics at optimum
    predicted_efficiency: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Predicted efficiency at optimum")
    predicted_co_ppm: Optional[float] = Field(default=None, ge=0.0, description="Predicted CO emissions at optimum")
    predicted_nox_ppm: Optional[float] = Field(default=None, ge=0.0, description="Predicted NOx emissions at optimum")
    predicted_stability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Predicted stability at optimum")

    # Solver information
    solver_name: str = Field(default="", max_length=100, description="Name of optimization solver used")
    iterations: Optional[int] = Field(default=None, ge=0, description="Number of solver iterations")
    solve_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Solver time in milliseconds")

    # Timestamps and provenance
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When optimization was performed")
    provenance_hash: Optional[str] = Field(default=None, description="SHA-256 hash for audit trail")

    @computed_field
    @property
    def is_optimal(self) -> bool:
        """Check if solution is optimal."""
        return self.status == OptimizationStatus.OPTIMAL

    @computed_field
    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible."""
        return self.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]

    @computed_field
    @property
    def improvement_percent(self) -> Optional[float]:
        """Calculate improvement if previous values available."""
        if self.optimal_setpoints and self.fuel_cost is not None:
            # Calculate from setpoint changes if available
            total_change = sum(
                abs(sp.change_percent or 0) for sp in self.optimal_setpoints
            )
            return total_change / len(self.optimal_setpoints) if self.optimal_setpoints else None
        return None

    def get_setpoint(self, tag: str) -> Optional[SetpointValue]:
        """Get setpoint value for a specific tag."""
        for sp in self.optimal_setpoints:
            if sp.tag == tag:
                return sp
        return None

    def to_setpoint_dict(self) -> Dict[str, float]:
        """Convert optimal setpoints to simple dictionary."""
        return {sp.tag: sp.value for sp in self.optimal_setpoints}

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        data = {
            "result_id": self.result_id,
            "status": self.status.value,
            "objective_value": self.objective_value,
            "setpoints": [(sp.tag, sp.value) for sp in self.optimal_setpoints],
            "timestamp": self.timestamp.isoformat()
        }
        import json
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()


class ConfidenceLevel(str, Enum):
    """Confidence level for recommendations."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class RecommendationPriority(str, Enum):
    """Priority level for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class SetpointRecommendation(BaseModel):
    """
    Setpoint change recommendation with explanation.

    Provides a recommended setpoint change with confidence level,
    expected benefits, and human-readable explanation.

    Attributes:
        tag: Tag/variable name for the setpoint
        value: Recommended new value
        confidence: Confidence level of recommendation
        explanation: Human-readable explanation of recommendation
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    recommendation_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17], description="Unique identifier")
    tag: str = Field(..., min_length=1, max_length=100, description="Tag/variable name for the setpoint")
    current_value: Optional[float] = Field(default=None, description="Current setpoint value")
    recommended_value: float = Field(..., description="Recommended new setpoint value")
    unit: str = Field(default="", max_length=50, description="Engineering unit")

    # Confidence and priority
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence level of recommendation")
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Numeric confidence score (0-1)")
    priority: RecommendationPriority = Field(default=RecommendationPriority.MEDIUM, description="Priority level")

    # Expected benefits
    expected_fuel_savings_percent: Optional[float] = Field(default=None, description="Expected fuel savings as percentage")
    expected_emissions_reduction_percent: Optional[float] = Field(default=None, description="Expected emissions reduction as percentage")
    expected_efficiency_gain_percent: Optional[float] = Field(default=None, description="Expected efficiency gain as percentage")

    # Explanation
    explanation: str = Field(..., min_length=1, max_length=1000, description="Human-readable explanation of recommendation")
    rationale: List[str] = Field(default_factory=list, description="List of reasons supporting this recommendation")

    # Risk assessment
    risk_level: str = Field(default="low", description="Risk level of implementing recommendation")
    potential_issues: List[str] = Field(default_factory=list, description="Potential issues or side effects")

    # Validity
    valid_until: Optional[datetime] = Field(default=None, description="Recommendation validity expiration")
    requires_operator_approval: bool = Field(default=True, description="Whether operator approval is required")
    auto_implement: bool = Field(default=False, description="Whether to auto-implement if in closed-loop mode")

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When recommendation was generated")

    @computed_field
    @property
    def change_amount(self) -> Optional[float]:
        """Calculate absolute change amount."""
        if self.current_value is not None:
            return self.recommended_value - self.current_value
        return None

    @computed_field
    @property
    def change_percent(self) -> Optional[float]:
        """Calculate percentage change."""
        if self.current_value is not None and self.current_value != 0:
            return ((self.recommended_value - self.current_value) / self.current_value) * 100.0
        return None

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Check if recommendation is still valid."""
        if self.valid_until is None:
            return True
        return datetime.utcnow() < self.valid_until


class OptimizationScenario(BaseModel):
    """Defines an optimization scenario with specific parameters."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    scenario_id: str = Field(..., min_length=1, max_length=50, description="Unique scenario identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable scenario name")
    description: str = Field(default="", max_length=1000, description="Detailed scenario description")
    objective: OptimizationObjective = Field(..., description="Optimization objective for this scenario")
    constraints: ConstraintSet = Field(..., description="Constraint set for this scenario")

    # Scenario parameters
    load_percent: Optional[float] = Field(default=None, ge=0.0, le=120.0, description="Target load as percentage")
    ambient_temperature_celsius: Optional[float] = Field(default=None, description="Ambient temperature condition")
    fuel_type: Optional[str] = Field(default=None, description="Fuel type for this scenario")

    # Metadata
    active: bool = Field(default=True, description="Whether this scenario is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    created_by: Optional[str] = Field(default=None, description="Creator identifier")


__all__ = [
    "ObjectiveType",
    "ConstraintType",
    "ConstraintOperator",
    "OptimizationStatus",
    "OptimizationObjective",
    "Constraint",
    "ConstraintSet",
    "SetpointValue",
    "BindingConstraint",
    "OptimizationResult",
    "ConfidenceLevel",
    "RecommendationPriority",
    "SetpointRecommendation",
    "OptimizationScenario",
]
