"""
UQ Schemas - Pydantic models for Uncertainty Quantification Engine

This module defines all data models for the UQ Engine with complete
type safety and validation. All models support SHA-256 provenance
tracking for audit compliance.

Key Models:
    - PredictionInterval: Upper/lower bounds with confidence level
    - Scenario: Single realization of uncertain variables
    - ScenarioSet: Collection of scenarios with probabilities
    - CalibrationMetrics: PICP, interval width, reliability metrics
    - UncertaintyBand: Multi-quantile uncertainty representation
    - RobustSolution: Optimization solution with feasibility guarantees

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class UncertaintySourceType(str, Enum):
    """Types of uncertainty sources in process heat systems."""

    WEATHER = "weather"
    PRICE = "price"
    DEMAND = "demand"
    EQUIPMENT = "equipment"
    MEASUREMENT = "measurement"
    MODEL = "model"
    PARAMETER = "parameter"


class DistributionType(str, Enum):
    """Statistical distribution types for uncertainty modeling."""

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    EMPIRICAL = "empirical"
    BOOTSTRAP = "bootstrap"
    KERNEL_DENSITY = "kernel_density"


class OptimizationObjective(str, Enum):
    """Optimization objectives for robust planning."""

    MIN_EXPECTED_COST = "min_expected_cost"
    MIN_MAX_REGRET = "min_max_regret"
    MIN_CVaR = "min_cvar"  # Conditional Value at Risk
    MAX_RELIABILITY = "max_reliability"
    MULTI_OBJECTIVE = "multi_objective"


class ConstraintType(str, Enum):
    """Types of robust constraints."""

    HARD = "hard"  # Must be satisfied in all scenarios
    SOFT = "soft"  # Can be violated with penalty
    CHANCE = "chance"  # Probabilistic constraint


class CalibrationStatus(str, Enum):
    """Calibration status indicators."""

    WELL_CALIBRATED = "well_calibrated"
    UNDER_CONFIDENT = "under_confident"  # Intervals too wide
    OVER_CONFIDENT = "over_confident"  # Intervals too narrow
    DRIFT_DETECTED = "drift_detected"
    REQUIRES_RETRAINING = "requires_retraining"


class ProvenanceRecord(BaseModel):
    """
    Provenance record for audit trail - ZERO HALLUCINATION GUARANTEE.

    Every calculation includes a SHA-256 hash of inputs and outputs
    for complete reproducibility and regulatory compliance.
    """

    record_id: UUID = Field(default_factory=uuid4, description="Unique record identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Record creation time")
    calculation_type: str = Field(..., description="Type of calculation performed")
    input_hash: str = Field(..., description="SHA-256 hash of all inputs")
    output_hash: str = Field(..., description="SHA-256 hash of all outputs")
    combined_hash: str = Field(..., description="SHA-256 hash of entire calculation")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values (for audit)")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output values (for audit)")
    computation_time_ms: float = Field(default=0.0, description="Computation time in milliseconds")

    @staticmethod
    def compute_hash(data: Any) -> str:
        """Compute SHA-256 hash of data - DETERMINISTIC."""
        if isinstance(data, dict):
            # Sort keys for deterministic hashing
            sorted_data = json.dumps(data, sort_keys=True, default=str)
        else:
            sorted_data = str(data)
        return hashlib.sha256(sorted_data.encode('utf-8')).hexdigest()

    @classmethod
    def create(
        cls,
        calculation_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        computation_time_ms: float = 0.0
    ) -> ProvenanceRecord:
        """Create a provenance record with computed hashes."""
        input_hash = cls.compute_hash(inputs)
        output_hash = cls.compute_hash(outputs)
        combined_hash = cls.compute_hash({
            "inputs": inputs,
            "outputs": outputs,
            "calculation_type": calculation_type
        })

        return cls(
            calculation_type=calculation_type,
            input_hash=input_hash,
            output_hash=output_hash,
            combined_hash=combined_hash,
            inputs=inputs,
            outputs=outputs,
            computation_time_ms=computation_time_ms
        )


class QuantileValue(BaseModel):
    """Single quantile value with probability level."""

    probability: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Quantile probability (0-1)"
    )
    value: Decimal = Field(..., description="Value at this quantile")

    @field_validator('probability', mode='before')
    @classmethod
    def coerce_probability(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, float):
            return Decimal(str(v))
        return v

    @field_validator('value', mode='before')
    @classmethod
    def coerce_value(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v


class QuantileSet(BaseModel):
    """
    Set of quantiles representing a distribution - DETERMINISTIC.

    Used for prediction intervals and uncertainty bands.
    All values are stored as Decimal for bit-perfect reproducibility.
    """

    quantiles: List[QuantileValue] = Field(..., min_length=1, description="List of quantile values")
    variable_name: str = Field(..., description="Name of the variable")
    unit: str = Field(..., description="Unit of measurement")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @property
    def p10(self) -> Optional[Decimal]:
        """Get P10 (10th percentile) value."""
        return self._get_quantile(Decimal("0.10"))

    @property
    def p50(self) -> Optional[Decimal]:
        """Get P50 (median) value."""
        return self._get_quantile(Decimal("0.50"))

    @property
    def p90(self) -> Optional[Decimal]:
        """Get P90 (90th percentile) value."""
        return self._get_quantile(Decimal("0.90"))

    def _get_quantile(self, probability: Decimal) -> Optional[Decimal]:
        """Get value at specified quantile probability."""
        for q in self.quantiles:
            if q.probability == probability:
                return q.value
        return None

    def get_interval(self, lower_prob: Decimal, upper_prob: Decimal) -> Tuple[Decimal, Decimal]:
        """Get interval between two quantile probabilities."""
        lower = self._get_quantile(lower_prob)
        upper = self._get_quantile(upper_prob)
        if lower is None or upper is None:
            raise ValueError(f"Quantiles {lower_prob} and {upper_prob} not available")
        return (lower, upper)


class PredictionInterval(BaseModel):
    """
    Prediction interval with confidence level - DETERMINISTIC.

    Represents uncertainty around a point prediction with
    lower and upper bounds at a specified confidence level.
    """

    interval_id: UUID = Field(default_factory=uuid4, description="Unique interval identifier")
    point_estimate: Decimal = Field(..., description="Central point estimate")
    lower_bound: Decimal = Field(..., description="Lower bound of interval")
    upper_bound: Decimal = Field(..., description="Upper bound of interval")
    confidence_level: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence level (e.g., 0.90 for 90%)"
    )
    variable_name: str = Field(..., description="Name of predicted variable")
    unit: str = Field(..., description="Unit of measurement")
    horizon_minutes: int = Field(default=60, ge=1, description="Forecast horizon in minutes")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    source_model: str = Field(default="unknown", description="Model that generated prediction")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @field_validator('point_estimate', 'lower_bound', 'upper_bound', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric types to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v

    @model_validator(mode='after')
    def validate_bounds(self) -> PredictionInterval:
        """Ensure lower <= point <= upper."""
        if self.lower_bound > self.point_estimate:
            raise ValueError("Lower bound cannot exceed point estimate")
        if self.point_estimate > self.upper_bound:
            raise ValueError("Point estimate cannot exceed upper bound")
        return self

    @property
    def interval_width(self) -> Decimal:
        """Calculate interval width."""
        return self.upper_bound - self.lower_bound

    @property
    def relative_width(self) -> Decimal:
        """Calculate relative interval width (width / point_estimate)."""
        if self.point_estimate == 0:
            return Decimal("inf")
        return self.interval_width / abs(self.point_estimate)

    def contains(self, actual_value: Decimal) -> bool:
        """Check if actual value falls within interval."""
        return self.lower_bound <= actual_value <= self.upper_bound


class UncertaintySource(BaseModel):
    """
    Definition of an uncertainty source for scenario generation.

    Describes the statistical properties of uncertain variables
    in the process heat system.
    """

    source_id: UUID = Field(default_factory=uuid4, description="Unique source identifier")
    name: str = Field(..., description="Human-readable name")
    source_type: UncertaintySourceType = Field(..., description="Type of uncertainty")
    distribution: DistributionType = Field(..., description="Statistical distribution")
    parameters: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Distribution parameters (mean, std, min, max, etc.)"
    )
    unit: str = Field(..., description="Unit of measurement")
    correlation_group: Optional[str] = Field(
        default=None,
        description="Group ID for correlated uncertainties"
    )
    time_correlation: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Temporal autocorrelation coefficient"
    )

    @field_validator('parameters', mode='before')
    @classmethod
    def coerce_parameters(cls, v: Dict[str, Any]) -> Dict[str, Decimal]:
        """Convert parameter values to Decimal."""
        return {k: Decimal(str(val)) for k, val in v.items()}


class UncertaintyBand(BaseModel):
    """
    Multi-quantile uncertainty band for visualization.

    Represents multiple confidence levels for fan charts
    and uncertainty visualization.
    """

    band_id: UUID = Field(default_factory=uuid4, description="Unique band identifier")
    variable_name: str = Field(..., description="Name of the variable")
    unit: str = Field(..., description="Unit of measurement")
    timestamps: List[datetime] = Field(..., description="Time points")
    quantile_sets: List[QuantileSet] = Field(..., description="Quantiles at each time point")
    confidence_levels: List[Decimal] = Field(
        default_factory=lambda: [Decimal("0.50"), Decimal("0.80"), Decimal("0.90"), Decimal("0.95")],
        description="Confidence levels represented"
    )
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @model_validator(mode='after')
    def validate_alignment(self) -> UncertaintyBand:
        """Ensure timestamps and quantile_sets are aligned."""
        if len(self.timestamps) != len(self.quantile_sets):
            raise ValueError("Number of timestamps must match number of quantile sets")
        return self


class ScenarioVariable(BaseModel):
    """Single variable value within a scenario."""

    name: str = Field(..., description="Variable name")
    value: Decimal = Field(..., description="Variable value in this scenario")
    unit: str = Field(..., description="Unit of measurement")
    uncertainty_source: Optional[str] = Field(
        default=None,
        description="Reference to uncertainty source"
    )

    @field_validator('value', mode='before')
    @classmethod
    def coerce_value(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v


class Scenario(BaseModel):
    """
    Single scenario realization - DETERMINISTIC.

    Represents one possible future state of all uncertain
    variables for robust optimization.
    """

    scenario_id: UUID = Field(default_factory=uuid4, description="Unique scenario identifier")
    name: str = Field(..., description="Human-readable scenario name")
    probability: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Scenario probability weight"
    )
    variables: List[ScenarioVariable] = Field(..., description="Variable values in scenario")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Scenario generation time")
    horizon_start: datetime = Field(..., description="Start of scenario horizon")
    horizon_end: datetime = Field(..., description="End of scenario horizon")
    is_base_case: bool = Field(default=False, description="Whether this is the expected/base case")
    is_worst_case: bool = Field(default=False, description="Whether this is worst case for robust opt")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Generation provenance")

    @field_validator('probability', mode='before')
    @classmethod
    def coerce_probability(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, float):
            return Decimal(str(v))
        return v

    def get_variable(self, name: str) -> Optional[ScenarioVariable]:
        """Get variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def get_value(self, name: str) -> Optional[Decimal]:
        """Get variable value by name."""
        var = self.get_variable(name)
        return var.value if var else None


class ScenarioSet(BaseModel):
    """
    Collection of scenarios for stochastic optimization - DETERMINISTIC.

    Scenarios can be generated via:
    - Monte Carlo sampling
    - Moment matching
    - Historical scenarios
    - Expert-defined scenarios
    """

    set_id: UUID = Field(default_factory=uuid4, description="Unique set identifier")
    name: str = Field(..., description="Scenario set name")
    scenarios: List[Scenario] = Field(..., min_length=1, description="List of scenarios")
    generation_method: str = Field(
        default="monte_carlo",
        description="Method used to generate scenarios"
    )
    uncertainty_sources: List[UncertaintySource] = Field(
        default_factory=list,
        description="Sources used to generate scenarios"
    )
    correlation_matrix: Optional[List[List[Decimal]]] = Field(
        default=None,
        description="Correlation matrix between uncertainty sources"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Generation provenance")

    @model_validator(mode='after')
    def validate_probabilities(self) -> ScenarioSet:
        """Ensure scenario probabilities sum to 1 (within tolerance)."""
        total_prob = sum(s.probability for s in self.scenarios)
        tolerance = Decimal("0.001")
        if abs(total_prob - Decimal("1.0")) > tolerance:
            raise ValueError(f"Scenario probabilities must sum to 1.0, got {total_prob}")
        return self

    @property
    def num_scenarios(self) -> int:
        """Number of scenarios in set."""
        return len(self.scenarios)

    def get_base_case(self) -> Optional[Scenario]:
        """Get the base case scenario."""
        for scenario in self.scenarios:
            if scenario.is_base_case:
                return scenario
        return None

    def get_worst_cases(self) -> List[Scenario]:
        """Get all worst case scenarios."""
        return [s for s in self.scenarios if s.is_worst_case]


class RobustConstraint(BaseModel):
    """
    Robust constraint specification for optimization.

    Defines constraints that must be satisfied across
    scenarios with specified reliability.
    """

    constraint_id: UUID = Field(default_factory=uuid4, description="Unique constraint identifier")
    name: str = Field(..., description="Constraint name")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    expression: str = Field(..., description="Constraint expression (symbolic)")
    bound: Decimal = Field(..., description="Constraint bound value")
    bound_type: str = Field(
        default="<=",
        description="Bound type: <=, >=, =="
    )
    reliability: Decimal = Field(
        default=Decimal("0.95"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Required reliability (for chance constraints)"
    )
    penalty: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Penalty for soft constraint violation"
    )

    @field_validator('bound', 'reliability', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v


class RobustSolution(BaseModel):
    """
    Robust optimization solution with feasibility guarantees - DETERMINISTIC.

    Contains optimal decisions that satisfy constraints across
    all scenarios in the uncertainty set.
    """

    solution_id: UUID = Field(default_factory=uuid4, description="Unique solution identifier")
    objective_value: Decimal = Field(..., description="Optimal objective value")
    objective_type: OptimizationObjective = Field(..., description="Optimization objective used")
    decision_variables: Dict[str, Decimal] = Field(
        ...,
        description="Optimal decision variable values"
    )
    scenario_set_id: UUID = Field(..., description="Reference to scenario set used")
    feasibility_rate: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Fraction of scenarios where solution is feasible"
    )
    worst_case_objective: Optional[Decimal] = Field(
        default=None,
        description="Objective value in worst case scenario"
    )
    expected_objective: Optional[Decimal] = Field(
        default=None,
        description="Expected objective value across scenarios"
    )
    cvar: Optional[Decimal] = Field(
        default=None,
        description="Conditional Value at Risk"
    )
    binding_constraints: List[str] = Field(
        default_factory=list,
        description="List of constraints that are binding"
    )
    constraint_slacks: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Slack values for each constraint"
    )
    solve_time_ms: float = Field(default=0.0, ge=0, description="Solve time in milliseconds")
    solver_status: str = Field(default="optimal", description="Solver status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Solution timestamp")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @field_validator('objective_value', 'worst_case_objective', 'expected_objective', 'cvar', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Optional[Decimal]:
        """Convert to Decimal for precision."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v

    @field_validator('decision_variables', 'constraint_slacks', mode='before')
    @classmethod
    def coerce_dict_values(cls, v: Dict[str, Any]) -> Dict[str, Decimal]:
        """Convert dict values to Decimal."""
        return {k: Decimal(str(val)) for k, val in v.items()}


class CalibrationMetrics(BaseModel):
    """
    Calibration metrics for prediction interval evaluation - DETERMINISTIC.

    Tracks how well prediction intervals match actual outcomes
    for reliability assessment and drift detection.
    """

    metrics_id: UUID = Field(default_factory=uuid4, description="Unique metrics identifier")
    model_name: str = Field(..., description="Name of the forecasting model")
    variable_name: str = Field(..., description="Variable being predicted")
    evaluation_period_start: datetime = Field(..., description="Start of evaluation period")
    evaluation_period_end: datetime = Field(..., description="End of evaluation period")
    num_predictions: int = Field(ge=1, description="Number of predictions evaluated")

    # Primary metrics
    picp: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Prediction Interval Coverage Probability"
    )
    target_coverage: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Target coverage (confidence level)"
    )
    mpiw: Decimal = Field(
        ge=Decimal("0"),
        description="Mean Prediction Interval Width"
    )
    nmpiw: Decimal = Field(
        ge=Decimal("0"),
        description="Normalized Mean Prediction Interval Width"
    )

    # Additional metrics
    winkler_score: Optional[Decimal] = Field(
        default=None,
        description="Winkler score (interval accuracy)"
    )
    crps: Optional[Decimal] = Field(
        default=None,
        description="Continuous Ranked Probability Score"
    )
    calibration_error: Decimal = Field(
        default=Decimal("0"),
        description="Absolute difference between PICP and target"
    )

    # Status
    status: CalibrationStatus = Field(
        default=CalibrationStatus.WELL_CALIBRATED,
        description="Calibration status assessment"
    )
    drift_detected: bool = Field(
        default=False,
        description="Whether drift has been detected"
    )
    retraining_recommended: bool = Field(
        default=False,
        description="Whether model retraining is recommended"
    )

    # Provenance
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @field_validator('picp', 'target_coverage', 'mpiw', 'nmpiw', 'calibration_error', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v

    @model_validator(mode='after')
    def compute_derived_metrics(self) -> CalibrationMetrics:
        """Compute derived metrics and status."""
        # Compute calibration error
        self.calibration_error = abs(self.picp - self.target_coverage)

        # Determine calibration status
        tolerance = Decimal("0.05")  # 5% tolerance
        if self.calibration_error <= tolerance:
            self.status = CalibrationStatus.WELL_CALIBRATED
        elif self.picp > self.target_coverage + tolerance:
            self.status = CalibrationStatus.UNDER_CONFIDENT
        elif self.picp < self.target_coverage - tolerance:
            self.status = CalibrationStatus.OVER_CONFIDENT

        # Check if retraining is needed
        if self.calibration_error > Decimal("0.10"):  # >10% error
            self.retraining_recommended = True

        return self


class ReliabilityDiagramPoint(BaseModel):
    """Single point on a reliability diagram."""

    predicted_probability: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Predicted probability/confidence"
    )
    observed_frequency: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Observed frequency of coverage"
    )
    num_samples: int = Field(ge=1, description="Number of samples in this bin")

    @field_validator('predicted_probability', 'observed_frequency', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v


class ReliabilityDiagram(BaseModel):
    """
    Reliability diagram data for calibration visualization - DETERMINISTIC.

    Shows relationship between predicted probabilities and
    observed frequencies for interval calibration assessment.
    """

    diagram_id: UUID = Field(default_factory=uuid4, description="Unique diagram identifier")
    model_name: str = Field(..., description="Model being evaluated")
    variable_name: str = Field(..., description="Variable being predicted")
    points: List[ReliabilityDiagramPoint] = Field(..., description="Diagram points")
    num_bins: int = Field(default=10, ge=2, description="Number of bins used")
    total_samples: int = Field(ge=1, description="Total number of samples")
    expected_calibration_error: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Expected Calibration Error (ECE)"
    )
    maximum_calibration_error: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Maximum Calibration Error (MCE)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @field_validator('expected_calibration_error', 'maximum_calibration_error', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v


class FanChartData(BaseModel):
    """
    Fan chart data for uncertainty visualization - DETERMINISTIC.

    Provides data for rendering fan charts showing prediction
    uncertainty over time with multiple confidence bands.
    """

    chart_id: UUID = Field(default_factory=uuid4, description="Unique chart identifier")
    variable_name: str = Field(..., description="Variable being visualized")
    unit: str = Field(..., description="Unit of measurement")
    timestamps: List[datetime] = Field(..., description="Time points")
    central_values: List[Decimal] = Field(..., description="Central/expected values")
    bands: Dict[str, List[Tuple[Decimal, Decimal]]] = Field(
        ...,
        description="Confidence bands: {'50%': [(lower, upper), ...], '90%': [...]}"
    )
    historical_values: Optional[List[Decimal]] = Field(
        default=None,
        description="Historical actual values for comparison"
    )
    historical_timestamps: Optional[List[datetime]] = Field(
        default=None,
        description="Timestamps for historical values"
    )
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @field_validator('central_values', mode='before')
    @classmethod
    def coerce_central_values(cls, v: List[Any]) -> List[Decimal]:
        """Convert to Decimal list."""
        return [Decimal(str(x)) for x in v]

    @model_validator(mode='after')
    def validate_alignment(self) -> FanChartData:
        """Ensure all lists are aligned."""
        n = len(self.timestamps)
        if len(self.central_values) != n:
            raise ValueError("Central values must match timestamps length")
        for band_name, band_values in self.bands.items():
            if len(band_values) != n:
                raise ValueError(f"Band '{band_name}' must have {n} values")
        return self


class RiskAssessment(BaseModel):
    """
    Risk assessment for constraint binding probability - DETERMINISTIC.

    Evaluates the probability that constraints will become
    binding under uncertainty for proactive risk management.
    """

    assessment_id: UUID = Field(default_factory=uuid4, description="Unique assessment identifier")
    constraint_name: str = Field(..., description="Constraint being assessed")
    current_value: Decimal = Field(..., description="Current constraint value")
    constraint_bound: Decimal = Field(..., description="Constraint bound")
    headroom: Decimal = Field(..., description="Distance to constraint bound")
    headroom_percent: Decimal = Field(..., description="Headroom as percentage")
    probability_of_binding: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Probability constraint becomes binding"
    )
    probability_of_violation: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Probability constraint is violated"
    )
    time_to_binding_minutes: Optional[int] = Field(
        default=None,
        description="Expected time until constraint binds (if trending)"
    )
    risk_level: str = Field(
        default="low",
        description="Risk level: low, medium, high, critical"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Recommended action if risk is elevated"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @field_validator('current_value', 'constraint_bound', 'headroom', 'headroom_percent',
                     'probability_of_binding', 'probability_of_violation', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v

    @model_validator(mode='after')
    def compute_risk_level(self) -> RiskAssessment:
        """Compute risk level based on probabilities."""
        p_binding = self.probability_of_binding
        p_violation = self.probability_of_violation

        if p_violation > Decimal("0.10"):
            self.risk_level = "critical"
        elif p_binding > Decimal("0.50") or p_violation > Decimal("0.05"):
            self.risk_level = "high"
        elif p_binding > Decimal("0.20"):
            self.risk_level = "medium"
        else:
            self.risk_level = "low"

        return self


class ScenarioComparison(BaseModel):
    """
    Scenario comparison output for decision support - DETERMINISTIC.

    Compares outcomes across scenarios to support robust
    decision making under uncertainty.
    """

    comparison_id: UUID = Field(default_factory=uuid4, description="Unique comparison identifier")
    scenarios_compared: List[UUID] = Field(..., description="IDs of scenarios compared")
    metric_name: str = Field(..., description="Metric being compared")
    unit: str = Field(..., description="Unit of measurement")
    values_by_scenario: Dict[str, Decimal] = Field(
        ...,
        description="Metric value for each scenario"
    )
    expected_value: Decimal = Field(..., description="Probability-weighted expected value")
    min_value: Decimal = Field(..., description="Minimum across scenarios")
    max_value: Decimal = Field(..., description="Maximum across scenarios")
    range_value: Decimal = Field(..., description="Range (max - min)")
    std_dev: Decimal = Field(..., description="Standard deviation across scenarios")
    regret_by_scenario: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Regret values for min-max regret optimization"
    )
    max_regret: Optional[Decimal] = Field(
        default=None,
        description="Maximum regret value"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Comparison timestamp")
    provenance: Optional[ProvenanceRecord] = Field(default=None, description="Calculation provenance")

    @field_validator('expected_value', 'min_value', 'max_value', 'range_value', 'std_dev', mode='before')
    @classmethod
    def coerce_to_decimal(cls, v: Any) -> Decimal:
        """Convert to Decimal for precision."""
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        return v

    @field_validator('values_by_scenario', mode='before')
    @classmethod
    def coerce_dict_values(cls, v: Dict[str, Any]) -> Dict[str, Decimal]:
        """Convert dict values to Decimal."""
        return {k: Decimal(str(val)) for k, val in v.items()}
