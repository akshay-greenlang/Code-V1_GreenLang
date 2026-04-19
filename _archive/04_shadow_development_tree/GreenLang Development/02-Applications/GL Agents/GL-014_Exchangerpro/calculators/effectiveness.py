"""
GL-014 EXCHANGERPRO - Effectiveness Calculator

Deterministic calculation of thermal effectiveness metrics for heat exchanger
performance monitoring and optimization.

Thermal Effectiveness Metrics:
    epsilon = Q_actual / Q_max
           = (T_hot_in - T_hot_out) / (T_hot_in - T_cold_in)  [if C_h = C_min]
           = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)  [if C_c = C_min]

Approach Temperatures:
    Hot approach (pinch) = T_hot_out - T_cold_in
    Cold approach = T_hot_in - T_cold_out

Temperature Range:
    Range_hot = T_hot_in - T_hot_out
    Range_cold = T_cold_out - T_cold_in

Capacity Ratio:
    R = Range_hot / Range_cold (if cold has larger range)
    R = Range_cold / Range_hot (if hot has larger range)

TEMA Compliance:
    - Uses TEMA terminology (range, approach, effectiveness)
    - Calculates all metrics for exchanger performance reports

Reference:
    - TEMA Standards, 10th Edition
    - ASME PTC 12.5-2000
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math


# =============================================================================
# Constants
# =============================================================================

# Minimum temperature difference for meaningful calculation (K)
MIN_TEMP_DIFF = 0.01

# Warning threshold for low effectiveness
LOW_EFFECTIVENESS_THRESHOLD = 0.3

# Warning threshold for close approach temperature (K)
MIN_APPROACH_WARNING = 5.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TemperatureData:
    """
    Temperature data for effectiveness calculation.

    All temperatures in Kelvin internally for thermodynamic consistency.
    """
    T_hot_in_K: float         # Hot fluid inlet temperature [K]
    T_hot_out_K: float        # Hot fluid outlet temperature [K]
    T_cold_in_K: float        # Cold fluid inlet temperature [K]
    T_cold_out_K: float       # Cold fluid outlet temperature [K]

    @classmethod
    def from_celsius(
        cls,
        T_hot_in_C: float,
        T_hot_out_C: float,
        T_cold_in_C: float,
        T_cold_out_C: float,
    ) -> "TemperatureData":
        """Create from Celsius temperatures."""
        return cls(
            T_hot_in_K=T_hot_in_C + 273.15,
            T_hot_out_K=T_hot_out_C + 273.15,
            T_cold_in_K=T_cold_in_C + 273.15,
            T_cold_out_K=T_cold_out_C + 273.15,
        )

    @property
    def T_hot_in_C(self) -> float:
        """Hot inlet in Celsius."""
        return self.T_hot_in_K - 273.15

    @property
    def T_hot_out_C(self) -> float:
        """Hot outlet in Celsius."""
        return self.T_hot_out_K - 273.15

    @property
    def T_cold_in_C(self) -> float:
        """Cold inlet in Celsius."""
        return self.T_cold_in_K - 273.15

    @property
    def T_cold_out_C(self) -> float:
        """Cold outlet in Celsius."""
        return self.T_cold_out_K - 273.15

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "T_hot_in_K": self.T_hot_in_K,
            "T_hot_out_K": self.T_hot_out_K,
            "T_cold_in_K": self.T_cold_in_K,
            "T_cold_out_K": self.T_cold_out_K,
            "T_hot_in_C": round(self.T_hot_in_C, 2),
            "T_hot_out_C": round(self.T_hot_out_C, 2),
            "T_cold_in_C": round(self.T_cold_in_C, 2),
            "T_cold_out_C": round(self.T_cold_out_C, 2),
        }


@dataclass
class CapacityData:
    """Heat capacity rate data for full effectiveness calculation."""
    C_hot_W_K: Optional[float] = None    # Hot-side capacity rate [W/K]
    C_cold_W_K: Optional[float] = None   # Cold-side capacity rate [W/K]

    @property
    def C_min_W_K(self) -> Optional[float]:
        """Minimum heat capacity rate."""
        if self.C_hot_W_K is None or self.C_cold_W_K is None:
            return None
        return min(self.C_hot_W_K, self.C_cold_W_K)

    @property
    def C_max_W_K(self) -> Optional[float]:
        """Maximum heat capacity rate."""
        if self.C_hot_W_K is None or self.C_cold_W_K is None:
            return None
        return max(self.C_hot_W_K, self.C_cold_W_K)

    @property
    def C_ratio(self) -> Optional[float]:
        """Capacity ratio C_min/C_max."""
        if self.C_min_W_K is None or self.C_max_W_K is None or self.C_max_W_K == 0:
            return None
        return self.C_min_W_K / self.C_max_W_K

    @property
    def min_side(self) -> Optional[str]:
        """Which side has minimum capacity rate."""
        if self.C_hot_W_K is None or self.C_cold_W_K is None:
            return None
        return "hot" if self.C_hot_W_K <= self.C_cold_W_K else "cold"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "C_hot_W_K": self.C_hot_W_K,
            "C_cold_W_K": self.C_cold_W_K,
            "C_min_W_K": self.C_min_W_K,
            "C_max_W_K": self.C_max_W_K,
            "C_ratio": round(self.C_ratio, 6) if self.C_ratio else None,
            "min_side": self.min_side,
        }


@dataclass
class EffectivenessInputs:
    """
    Inputs for effectiveness calculation.

    Temperature data is required.
    Capacity data is optional (enables full effectiveness calculation).
    """
    temperatures: TemperatureData
    capacities: Optional[CapacityData] = None

    # Reference values for comparison
    epsilon_design: Optional[float] = None     # Design effectiveness
    epsilon_baseline: Optional[float] = None   # Baseline (clean) effectiveness

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for hashing."""
        return {
            "temperatures": self.temperatures.to_dict(),
            "capacities": self.capacities.to_dict() if self.capacities else None,
            "epsilon_design": self.epsilon_design,
            "epsilon_baseline": self.epsilon_baseline,
        }


@dataclass
class ApproachTemperatures:
    """
    Approach (pinch) temperatures.

    These are the closest temperature approaches in the exchanger.
    """
    hot_approach_K: float     # T_hot_out - T_cold_in (counter-current pinch)
    cold_approach_K: float    # T_hot_in - T_cold_out
    minimum_approach_K: float # min(hot_approach, cold_approach)
    approach_location: str    # "hot_end" or "cold_end"

    # Celsius versions
    hot_approach_C: float
    cold_approach_C: float
    minimum_approach_C: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hot_approach_K": round(self.hot_approach_K, 3),
            "cold_approach_K": round(self.cold_approach_K, 3),
            "minimum_approach_K": round(self.minimum_approach_K, 3),
            "approach_location": self.approach_location,
            "hot_approach_C": round(self.hot_approach_C, 3),
            "cold_approach_C": round(self.cold_approach_C, 3),
            "minimum_approach_C": round(self.minimum_approach_C, 3),
        }


@dataclass
class TemperatureRanges:
    """Temperature ranges (changes) for each stream."""
    range_hot_K: float        # T_hot_in - T_hot_out
    range_cold_K: float       # T_cold_out - T_cold_in
    range_max_K: float        # Maximum possible: T_hot_in - T_cold_in
    ratio_hot_to_max: float   # range_hot / range_max
    ratio_cold_to_max: float  # range_cold / range_max

    # Celsius versions (numerically equal for ranges)
    range_hot_C: float
    range_cold_C: float
    range_max_C: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "range_hot_K": round(self.range_hot_K, 3),
            "range_cold_K": round(self.range_cold_K, 3),
            "range_max_K": round(self.range_max_K, 3),
            "ratio_hot_to_max": round(self.ratio_hot_to_max, 6),
            "ratio_cold_to_max": round(self.ratio_cold_to_max, 6),
            "range_hot_C": round(self.range_hot_C, 3),
            "range_cold_C": round(self.range_cold_C, 3),
            "range_max_C": round(self.range_max_C, 3),
        }


@dataclass
class EffectivenessMetrics:
    """
    Core effectiveness metrics.
    """
    # Primary effectiveness (based on C_min side if known)
    epsilon: float            # Q_actual / Q_max

    # Side-specific effectiveness
    epsilon_hot: float        # (T_hot_in - T_hot_out) / (T_hot_in - T_cold_in)
    epsilon_cold: float       # (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)

    # Which side determines primary effectiveness
    primary_side: str         # "hot", "cold", or "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "epsilon": round(self.epsilon, 6),
            "epsilon_hot": round(self.epsilon_hot, 6),
            "epsilon_cold": round(self.epsilon_cold, 6),
            "primary_side": self.primary_side,
        }


@dataclass
class PerformanceComparison:
    """Comparison with design and baseline values."""
    epsilon_current: float
    epsilon_design: Optional[float]
    epsilon_baseline: Optional[float]

    # Derived metrics
    performance_vs_design: Optional[float]     # current / design
    performance_vs_baseline: Optional[float]   # current / baseline
    degradation_from_baseline: Optional[float] # (baseline - current) / baseline * 100

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "epsilon_current": round(self.epsilon_current, 6),
            "epsilon_design": round(self.epsilon_design, 6) if self.epsilon_design else None,
            "epsilon_baseline": round(self.epsilon_baseline, 6) if self.epsilon_baseline else None,
            "performance_vs_design": round(self.performance_vs_design, 4) if self.performance_vs_design else None,
            "performance_vs_baseline": round(self.performance_vs_baseline, 4) if self.performance_vs_baseline else None,
            "degradation_from_baseline": round(self.degradation_from_baseline, 2) if self.degradation_from_baseline else None,
        }


@dataclass
class EffectivenessResult:
    """
    Complete effectiveness calculation result.
    """
    # Core metrics
    effectiveness: EffectivenessMetrics
    approaches: ApproachTemperatures
    ranges: TemperatureRanges

    # Capacity data (if provided)
    capacity_data: Optional[CapacityData]

    # Performance comparison
    comparison: Optional[PerformanceComparison]

    # Heat transfer (if capacities known)
    Q_actual_W: Optional[float]
    Q_max_W: Optional[float]

    # Validation
    is_valid: bool
    has_temperature_cross: bool
    warnings: List[str] = field(default_factory=list)

    # Calculation trace
    calculation_steps: List[str] = field(default_factory=list)

    # Provenance
    inputs_hash: str = ""
    outputs_hash: str = ""
    computation_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0
    calculator_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "effectiveness": self.effectiveness.to_dict(),
            "approaches": self.approaches.to_dict(),
            "ranges": self.ranges.to_dict(),
            "capacity_data": self.capacity_data.to_dict() if self.capacity_data else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "Q_actual_W": round(self.Q_actual_W, 3) if self.Q_actual_W else None,
            "Q_max_W": round(self.Q_max_W, 3) if self.Q_max_W else None,
            "is_valid": self.is_valid,
            "has_temperature_cross": self.has_temperature_cross,
            "warnings": self.warnings,
            "calculation_steps": self.calculation_steps,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "computation_hash": self.computation_hash,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": round(self.execution_time_ms, 3),
            "calculator_version": self.calculator_version,
        }


# =============================================================================
# Effectiveness Calculator
# =============================================================================

class EffectivenessCalculator:
    """
    Deterministic Thermal Effectiveness Calculator.

    Calculates all thermal effectiveness metrics for heat exchanger
    performance monitoring:

    1. Thermal Effectiveness (epsilon)
        - Based on C_min side (true effectiveness)
        - Hot-side and cold-side effectiveness

    2. Approach Temperatures
        - Hot approach (T_hot_out - T_cold_in)
        - Cold approach (T_hot_in - T_cold_out)
        - Minimum approach (pinch)

    3. Temperature Ranges
        - Hot range (cooling)
        - Cold range (heating)
        - Maximum possible range

    4. Performance Comparison
        - vs. design effectiveness
        - vs. baseline (clean) effectiveness
        - Degradation percentage

    Zero-Hallucination Guarantee:
        All calculations are deterministic. Same inputs produce
        bit-perfect identical outputs. No LLM involvement.

    Example:
        >>> calc = EffectivenessCalculator()
        >>> temps = TemperatureData.from_celsius(
        ...     T_hot_in_C=150.0, T_hot_out_C=90.0,
        ...     T_cold_in_C=30.0, T_cold_out_C=80.0
        ... )
        >>> inputs = EffectivenessInputs(temperatures=temps)
        >>> result = calc.calculate(inputs)
        >>> print(f"Effectiveness = {result.effectiveness.epsilon:.4f}")
        >>> print(f"Minimum approach = {result.approaches.minimum_approach_C:.1f} C")
    """

    NAME = "EffectivenessCalculator"
    VERSION = "1.0.0"
    AGENT_ID = "GL-014"

    def __init__(
        self,
        min_temp_diff: float = MIN_TEMP_DIFF,
        approach_warning: float = MIN_APPROACH_WARNING,
    ):
        """
        Initialize Effectiveness Calculator.

        Args:
            min_temp_diff: Minimum temperature difference for validity [K]
            approach_warning: Threshold for approach temperature warning [K]
        """
        self.min_temp_diff = min_temp_diff
        self.approach_warning = approach_warning

    def calculate(self, inputs: EffectivenessInputs) -> EffectivenessResult:
        """
        Calculate all effectiveness metrics.

        Args:
            inputs: Temperature data and optional capacity data

        Returns:
            EffectivenessResult with all metrics and provenance
        """
        start_time = datetime.now(timezone.utc)
        warnings: List[str] = []
        calculation_steps: List[str] = []

        temps = inputs.temperatures

        # Step 1: Validate inputs
        validation_errors = self._validate_inputs(inputs)
        if validation_errors:
            warnings.extend(validation_errors)

        # Step 2: Check for temperature cross
        has_cross = self._check_temperature_cross(temps)
        if has_cross:
            warnings.append("Temperature cross detected - thermodynamically infeasible")

        # Step 3: Calculate temperature ranges
        ranges = self._calculate_ranges(temps)
        calculation_steps.append(
            f"Temperature ranges: hot={ranges.range_hot_C:.2f}C, cold={ranges.range_cold_C:.2f}C, max={ranges.range_max_C:.2f}C"
        )

        # Step 4: Calculate approach temperatures
        approaches = self._calculate_approaches(temps)
        calculation_steps.append(
            f"Approach temperatures: hot_end={approaches.hot_approach_C:.2f}C, cold_end={approaches.cold_approach_C:.2f}C"
        )

        if approaches.minimum_approach_K < self.approach_warning:
            warnings.append(
                f"Minimum approach ({approaches.minimum_approach_C:.1f}C) is below warning threshold "
                f"({self.approach_warning - 273.15:.1f}C). Risk of temperature cross under upset."
            )

        # Step 5: Calculate effectiveness metrics
        effectiveness = self._calculate_effectiveness(temps, inputs.capacities)
        calculation_steps.append(
            f"Effectiveness: epsilon={effectiveness.epsilon:.4f} ({effectiveness.primary_side} side)"
        )
        calculation_steps.append(
            f"epsilon_hot={effectiveness.epsilon_hot:.4f}, epsilon_cold={effectiveness.epsilon_cold:.4f}"
        )

        if effectiveness.epsilon < LOW_EFFECTIVENESS_THRESHOLD:
            warnings.append(
                f"Low effectiveness ({effectiveness.epsilon:.2%}). Check for fouling, bypassing, or underperformance."
            )

        # Step 6: Calculate heat transfer if capacities known
        Q_actual_W = None
        Q_max_W = None
        if inputs.capacities is not None and inputs.capacities.C_min_W_K is not None:
            Q_max_W = inputs.capacities.C_min_W_K * ranges.range_max_K
            Q_actual_W = effectiveness.epsilon * Q_max_W
            calculation_steps.append(
                f"Q_max = C_min * dT_max = {inputs.capacities.C_min_W_K:.1f} * {ranges.range_max_K:.2f} = {Q_max_W:.0f} W"
            )
            calculation_steps.append(f"Q_actual = epsilon * Q_max = {Q_actual_W:.0f} W")

        # Step 7: Performance comparison
        comparison = None
        if inputs.epsilon_design is not None or inputs.epsilon_baseline is not None:
            comparison = self._calculate_comparison(
                epsilon_current=effectiveness.epsilon,
                epsilon_design=inputs.epsilon_design,
                epsilon_baseline=inputs.epsilon_baseline,
            )

            if comparison.degradation_from_baseline is not None and comparison.degradation_from_baseline > 10:
                warnings.append(
                    f"Effectiveness degradation of {comparison.degradation_from_baseline:.1f}% from baseline. "
                    "Fouling or mechanical issue likely."
                )

        # Step 8: Compute provenance
        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        inputs_hash = self._compute_hash(inputs.to_dict())

        result_data = {
            "epsilon": effectiveness.epsilon,
            "minimum_approach_K": approaches.minimum_approach_K,
            "Q_actual_W": Q_actual_W,
        }
        outputs_hash = self._compute_hash(result_data)

        computation_hash = self._compute_hash({
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "calculator": self.NAME,
            "version": self.VERSION,
        })

        return EffectivenessResult(
            effectiveness=effectiveness,
            approaches=approaches,
            ranges=ranges,
            capacity_data=inputs.capacities,
            comparison=comparison,
            Q_actual_W=Q_actual_W,
            Q_max_W=Q_max_W,
            is_valid=len(validation_errors) == 0 and not has_cross,
            has_temperature_cross=has_cross,
            warnings=warnings,
            calculation_steps=calculation_steps,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            computation_hash=computation_hash,
            timestamp=start_time,
            execution_time_ms=execution_time_ms,
            calculator_version=self.VERSION,
        )

    def _validate_inputs(self, inputs: EffectivenessInputs) -> List[str]:
        """Validate inputs."""
        errors: List[str] = []
        temps = inputs.temperatures

        # Temperature validation
        if temps.T_hot_in_K <= 0:
            errors.append(f"Hot inlet temperature must be positive: {temps.T_hot_in_K}")
        if temps.T_hot_out_K <= 0:
            errors.append(f"Hot outlet temperature must be positive: {temps.T_hot_out_K}")
        if temps.T_cold_in_K <= 0:
            errors.append(f"Cold inlet temperature must be positive: {temps.T_cold_in_K}")
        if temps.T_cold_out_K <= 0:
            errors.append(f"Cold outlet temperature must be positive: {temps.T_cold_out_K}")

        # Hot stream should cool down
        if temps.T_hot_in_K < temps.T_hot_out_K:
            errors.append(
                f"Hot stream should cool: T_in ({temps.T_hot_in_C:.1f}C) < T_out ({temps.T_hot_out_C:.1f}C)"
            )

        # Cold stream should heat up
        if temps.T_cold_out_K < temps.T_cold_in_K:
            errors.append(
                f"Cold stream should heat: T_out ({temps.T_cold_out_C:.1f}C) < T_in ({temps.T_cold_in_C:.1f}C)"
            )

        # Hot inlet should exceed cold inlet
        if temps.T_hot_in_K <= temps.T_cold_in_K:
            errors.append(
                f"Hot inlet must exceed cold inlet: T_h_in ({temps.T_hot_in_C:.1f}C) <= T_c_in ({temps.T_cold_in_C:.1f}C)"
            )

        # Capacity validation
        if inputs.capacities is not None:
            if inputs.capacities.C_hot_W_K is not None and inputs.capacities.C_hot_W_K <= 0:
                errors.append(f"Hot capacity rate must be positive: {inputs.capacities.C_hot_W_K}")
            if inputs.capacities.C_cold_W_K is not None and inputs.capacities.C_cold_W_K <= 0:
                errors.append(f"Cold capacity rate must be positive: {inputs.capacities.C_cold_W_K}")

        return errors

    def _check_temperature_cross(self, temps: TemperatureData) -> bool:
        """Check for temperature cross."""
        # Cross occurs if cold outlet exceeds hot inlet
        if temps.T_cold_out_K >= temps.T_hot_in_K:
            return True

        # For counter-current, also check approach temperatures
        hot_approach = temps.T_hot_out_K - temps.T_cold_in_K
        cold_approach = temps.T_hot_in_K - temps.T_cold_out_K

        return hot_approach <= 0 or cold_approach <= 0

    def _calculate_ranges(self, temps: TemperatureData) -> TemperatureRanges:
        """Calculate temperature ranges."""
        range_hot_K = temps.T_hot_in_K - temps.T_hot_out_K
        range_cold_K = temps.T_cold_out_K - temps.T_cold_in_K
        range_max_K = temps.T_hot_in_K - temps.T_cold_in_K

        if range_max_K > self.min_temp_diff:
            ratio_hot = range_hot_K / range_max_K
            ratio_cold = range_cold_K / range_max_K
        else:
            ratio_hot = 0.0
            ratio_cold = 0.0

        return TemperatureRanges(
            range_hot_K=range_hot_K,
            range_cold_K=range_cold_K,
            range_max_K=range_max_K,
            ratio_hot_to_max=ratio_hot,
            ratio_cold_to_max=ratio_cold,
            range_hot_C=range_hot_K,  # Delta T is same in K and C
            range_cold_C=range_cold_K,
            range_max_C=range_max_K,
        )

    def _calculate_approaches(self, temps: TemperatureData) -> ApproachTemperatures:
        """Calculate approach temperatures."""
        # For counter-current flow
        hot_approach_K = temps.T_hot_out_K - temps.T_cold_in_K
        cold_approach_K = temps.T_hot_in_K - temps.T_cold_out_K

        minimum_approach_K = min(hot_approach_K, cold_approach_K)
        approach_location = "hot_end" if hot_approach_K <= cold_approach_K else "cold_end"

        return ApproachTemperatures(
            hot_approach_K=hot_approach_K,
            cold_approach_K=cold_approach_K,
            minimum_approach_K=minimum_approach_K,
            approach_location=approach_location,
            hot_approach_C=hot_approach_K,  # Delta T is same in K and C
            cold_approach_C=cold_approach_K,
            minimum_approach_C=minimum_approach_K,
        )

    def _calculate_effectiveness(
        self,
        temps: TemperatureData,
        capacities: Optional[CapacityData],
    ) -> EffectivenessMetrics:
        """Calculate effectiveness metrics."""
        range_max = temps.T_hot_in_K - temps.T_cold_in_K

        if range_max <= self.min_temp_diff:
            return EffectivenessMetrics(
                epsilon=0.0,
                epsilon_hot=0.0,
                epsilon_cold=0.0,
                primary_side="unknown",
            )

        # Side-specific effectiveness
        epsilon_hot = (temps.T_hot_in_K - temps.T_hot_out_K) / range_max
        epsilon_cold = (temps.T_cold_out_K - temps.T_cold_in_K) / range_max

        # Determine which side has C_min (controls effectiveness)
        if capacities is not None and capacities.min_side is not None:
            primary_side = capacities.min_side
            if primary_side == "hot":
                epsilon = epsilon_hot
            else:
                epsilon = epsilon_cold
        else:
            # Without capacity data, use the smaller effectiveness
            # (represents the limiting side)
            if epsilon_hot <= epsilon_cold:
                epsilon = epsilon_hot
                primary_side = "hot"
            else:
                epsilon = epsilon_cold
                primary_side = "cold"

        # Clamp to valid range
        epsilon = max(0.0, min(1.0, epsilon))
        epsilon_hot = max(0.0, min(1.0, epsilon_hot))
        epsilon_cold = max(0.0, min(1.0, epsilon_cold))

        return EffectivenessMetrics(
            epsilon=epsilon,
            epsilon_hot=epsilon_hot,
            epsilon_cold=epsilon_cold,
            primary_side=primary_side,
        )

    def _calculate_comparison(
        self,
        epsilon_current: float,
        epsilon_design: Optional[float],
        epsilon_baseline: Optional[float],
    ) -> PerformanceComparison:
        """Calculate performance comparison metrics."""
        performance_vs_design = None
        performance_vs_baseline = None
        degradation_from_baseline = None

        if epsilon_design is not None and epsilon_design > 0:
            performance_vs_design = epsilon_current / epsilon_design

        if epsilon_baseline is not None and epsilon_baseline > 0:
            performance_vs_baseline = epsilon_current / epsilon_baseline
            degradation_from_baseline = (epsilon_baseline - epsilon_current) / epsilon_baseline * 100.0

        return PerformanceComparison(
            epsilon_current=epsilon_current,
            epsilon_design=epsilon_design,
            epsilon_baseline=epsilon_baseline,
            performance_vs_design=performance_vs_design,
            performance_vs_baseline=performance_vs_baseline,
            degradation_from_baseline=degradation_from_baseline,
        )

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        normalized = self._normalize_for_hash(data)
        json_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _normalize_for_hash(self, obj: Any) -> Any:
        """Normalize for consistent hashing."""
        if obj is None:
            return None
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, float):
            return round(obj, 10)
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._normalize_for_hash(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._normalize_for_hash(v) for k, v in sorted(obj.items())}
        else:
            return str(obj)


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_effectiveness(
    T_hot_in_K: float,
    T_hot_out_K: float,
    T_cold_in_K: float,
    T_cold_out_K: float,
    C_hot_W_K: Optional[float] = None,
    C_cold_W_K: Optional[float] = None,
) -> float:
    """
    Quick effectiveness calculation (no provenance).

    If capacities are provided, uses C_min side.
    Otherwise, returns the smaller of hot/cold effectiveness.

    Args:
        T_hot_in_K, T_hot_out_K: Hot stream temperatures [K]
        T_cold_in_K, T_cold_out_K: Cold stream temperatures [K]
        C_hot_W_K, C_cold_W_K: Optional heat capacity rates [W/K]

    Returns:
        Thermal effectiveness (0 to 1)
    """
    range_max = T_hot_in_K - T_cold_in_K
    if range_max <= 0:
        return 0.0

    epsilon_hot = (T_hot_in_K - T_hot_out_K) / range_max
    epsilon_cold = (T_cold_out_K - T_cold_in_K) / range_max

    if C_hot_W_K is not None and C_cold_W_K is not None:
        if C_hot_W_K <= C_cold_W_K:
            return max(0.0, min(1.0, epsilon_hot))
        else:
            return max(0.0, min(1.0, epsilon_cold))

    return max(0.0, min(1.0, min(epsilon_hot, epsilon_cold)))


def calculate_approach_temperature(
    T_hot_out_K: float,
    T_cold_in_K: float,
) -> float:
    """
    Calculate hot-end approach (pinch) temperature.

    For counter-current flow, this is typically the minimum approach.

    Args:
        T_hot_out_K: Hot outlet temperature [K]
        T_cold_in_K: Cold inlet temperature [K]

    Returns:
        Approach temperature [K]
    """
    return T_hot_out_K - T_cold_in_K
