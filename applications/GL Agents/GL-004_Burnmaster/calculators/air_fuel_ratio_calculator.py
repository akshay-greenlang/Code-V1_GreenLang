"""
Air-Fuel Ratio Calculator for GL-004 BURNMASTER

Zero-hallucination calculation engine for air-fuel ratio optimization.
All calculations are deterministic, auditable, and bit-perfect reproducible.

This module implements:
- Current ratio computation from flow measurements
- Optimal ratio computation based on fuel type and load
- O2 trim bias calculations
- Ratio deviation analysis
- Stability tracking for ratio history

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
import statistics

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Pydantic Schemas for Input/Output
# =============================================================================

class RatioCalculationInput(BaseModel):
    """Input schema for air-fuel ratio calculation."""

    fuel_flow: float = Field(..., ge=0.0, description="Fuel flow rate in kg/h or Nm3/h")
    air_flow: float = Field(..., ge=0.0, description="Air flow rate in kg/h or Nm3/h")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of measurement")

    @field_validator('fuel_flow', 'air_flow')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Flow values must be non-negative")
        return v


class RatioCalculationResult(BaseModel):
    """Output schema for air-fuel ratio calculation with provenance."""

    current_ratio: Decimal = Field(..., description="Computed air-fuel ratio")
    is_valid: bool = Field(..., description="Whether the calculation is valid")
    calculation_steps: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uncertainty: Optional[float] = Field(default=None, description="Uncertainty estimate (percent)")


class OptimalRatioInput(BaseModel):
    """Input schema for optimal ratio calculation."""

    fuel_type: str = Field(..., description="Type of fuel (natural_gas, diesel, fuel_oil, etc.)")
    load_percent: float = Field(..., ge=0.0, le=100.0, description="Current load as percentage of max")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Operational constraints")


class OptimalRatioResult(BaseModel):
    """Output schema for optimal ratio calculation."""

    optimal_ratio: Decimal = Field(..., description="Computed optimal air-fuel ratio")
    stoichiometric_ratio: Decimal = Field(..., description="Stoichiometric ratio for fuel")
    excess_air_percent: Decimal = Field(..., description="Recommended excess air percentage")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class O2TrimInput(BaseModel):
    """Input schema for O2 trim bias calculation."""

    target_o2: float = Field(..., ge=0.0, le=21.0, description="Target O2 percentage in flue gas")
    current_o2: float = Field(..., ge=0.0, le=21.0, description="Current O2 percentage in flue gas")
    gain: float = Field(default=1.0, ge=0.1, le=10.0, description="Controller gain factor")


class O2TrimResult(BaseModel):
    """Output schema for O2 trim bias calculation."""

    bias_value: Decimal = Field(..., description="Computed O2 trim bias")
    direction: str = Field(..., description="Direction of adjustment (increase/decrease/hold)")
    magnitude_percent: Decimal = Field(..., description="Magnitude of adjustment as percentage")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class StabilityMetrics(BaseModel):
    """Stability metrics for air-fuel ratio history."""

    mean_ratio: Decimal = Field(..., description="Mean air-fuel ratio over period")
    std_deviation: Decimal = Field(..., description="Standard deviation of ratio")
    coefficient_of_variation: Decimal = Field(..., description="CV = std_dev / mean * 100")
    max_deviation: Decimal = Field(..., description="Maximum deviation from mean")
    min_ratio: Decimal = Field(..., description="Minimum ratio in period")
    max_ratio: Decimal = Field(..., description="Maximum ratio in period")
    is_stable: bool = Field(..., description="Whether ratio is considered stable")
    stability_score: float = Field(..., ge=0.0, le=100.0, description="Stability score 0-100")
    sample_count: int = Field(..., description="Number of samples analyzed")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# =============================================================================
# Stoichiometric Ratios Database (DETERMINISTIC LOOKUP)
# =============================================================================

# These are theoretical stoichiometric air-fuel ratios by mass
# Source: Engineering Toolbox, NIST, combustion engineering references
STOICHIOMETRIC_RATIOS: Dict[str, Decimal] = {
    "natural_gas": Decimal("17.2"),      # CH4 dominant
    "methane": Decimal("17.2"),
    "propane": Decimal("15.7"),
    "butane": Decimal("15.5"),
    "diesel": Decimal("14.5"),
    "fuel_oil_2": Decimal("14.4"),
    "fuel_oil_6": Decimal("13.8"),
    "kerosene": Decimal("14.7"),
    "gasoline": Decimal("14.7"),
    "hydrogen": Decimal("34.3"),
    "coal_bituminous": Decimal("11.5"),
    "coal_anthracite": Decimal("11.3"),
    "biomass_wood": Decimal("6.0"),
    "biomass_bagasse": Decimal("5.5"),
    "biogas": Decimal("10.0"),           # ~60% CH4, 40% CO2
    "landfill_gas": Decimal("8.0"),
    "coke_oven_gas": Decimal("4.5"),
    "blast_furnace_gas": Decimal("0.9"),
}

# Recommended excess air percentages by fuel type and load range
# Source: Industrial combustion optimization best practices
EXCESS_AIR_RECOMMENDATIONS: Dict[str, Dict[str, Decimal]] = {
    "natural_gas": {
        "low_load": Decimal("15.0"),      # <40% load
        "mid_load": Decimal("10.0"),      # 40-80% load
        "high_load": Decimal("5.0"),      # >80% load
    },
    "diesel": {
        "low_load": Decimal("25.0"),
        "mid_load": Decimal("20.0"),
        "high_load": Decimal("15.0"),
    },
    "fuel_oil_2": {
        "low_load": Decimal("25.0"),
        "mid_load": Decimal("20.0"),
        "high_load": Decimal("15.0"),
    },
    "fuel_oil_6": {
        "low_load": Decimal("30.0"),
        "mid_load": Decimal("25.0"),
        "high_load": Decimal("20.0"),
    },
    "coal_bituminous": {
        "low_load": Decimal("35.0"),
        "mid_load": Decimal("30.0"),
        "high_load": Decimal("25.0"),
    },
    "biomass_wood": {
        "low_load": Decimal("40.0"),
        "mid_load": Decimal("35.0"),
        "high_load": Decimal("30.0"),
    },
}


# =============================================================================
# Air-Fuel Ratio Calculator Class
# =============================================================================

class AirFuelRatioCalculator:
    """
    Zero-hallucination calculator for air-fuel ratio optimization.

    Guarantees:
    - Deterministic: Same input produces same output (bit-perfect)
    - Auditable: SHA-256 provenance hash for every calculation
    - Reproducible: Complete calculation step tracking
    - NO LLM: Pure arithmetic and lookup operations only

    Example:
        >>> calculator = AirFuelRatioCalculator()
        >>> result = calculator.compute_current_ratio(100.0, 1720.0)
        >>> print(result.current_ratio)
        17.200
    """

    def __init__(self, precision: int = 3):
        """
        Initialize calculator with precision settings.

        Args:
            precision: Decimal places for output values (default: 3)
        """
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding (ROUND_HALF_UP for regulatory compliance)."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Core Calculation Methods
    # -------------------------------------------------------------------------

    def compute_current_ratio(
        self,
        fuel_flow: float,
        air_flow: float
    ) -> RatioCalculationResult:
        """
        Compute current air-fuel ratio from flow measurements.

        DETERMINISTIC: air_flow / fuel_flow

        Args:
            fuel_flow: Fuel flow rate (kg/h or Nm3/h)
            air_flow: Air flow rate (kg/h or Nm3/h)

        Returns:
            RatioCalculationResult with computed ratio and provenance

        Raises:
            ValueError: If fuel_flow is zero (division by zero)
        """
        # Validate input
        input_data = RatioCalculationInput(fuel_flow=fuel_flow, air_flow=air_flow)

        calculation_steps = []

        # Step 1: Check for zero fuel flow
        if input_data.fuel_flow == 0.0:
            # Cannot compute ratio with zero fuel flow
            provenance = self._compute_provenance_hash({
                'fuel_flow': fuel_flow,
                'air_flow': air_flow,
                'error': 'zero_fuel_flow'
            })
            return RatioCalculationResult(
                current_ratio=Decimal('0'),
                is_valid=False,
                calculation_steps=[{
                    'step': 1,
                    'description': 'Validation',
                    'result': 'Invalid - fuel flow is zero'
                }],
                provenance_hash=provenance,
                uncertainty=None
            )

        # Step 2: Convert to Decimal for precision
        fuel_decimal = Decimal(str(fuel_flow))
        air_decimal = Decimal(str(air_flow))

        calculation_steps.append({
            'step': 1,
            'description': 'Convert to Decimal',
            'fuel_flow': str(fuel_decimal),
            'air_flow': str(air_decimal)
        })

        # Step 3: Compute ratio (DETERMINISTIC division)
        ratio = air_decimal / fuel_decimal

        calculation_steps.append({
            'step': 2,
            'description': 'Compute ratio = air_flow / fuel_flow',
            'operation': 'division',
            'result': str(ratio)
        })

        # Step 4: Apply precision rounding
        ratio_rounded = self._quantize(ratio)

        calculation_steps.append({
            'step': 3,
            'description': f'Apply precision (ROUND_HALF_UP to {self.precision} decimals)',
            'result': str(ratio_rounded)
        })

        # Step 5: Estimate uncertainty (based on typical flow meter accuracy)
        # Typical flow meter accuracy is +/- 1-2%
        uncertainty = 2.0  # Conservative 2% uncertainty

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'fuel_flow': fuel_flow,
            'air_flow': air_flow,
            'ratio': str(ratio_rounded),
            'precision': self.precision
        })

        return RatioCalculationResult(
            current_ratio=ratio_rounded,
            is_valid=True,
            calculation_steps=calculation_steps,
            provenance_hash=provenance,
            uncertainty=uncertainty
        )

    def compute_optimal_ratio(
        self,
        fuel_type: str,
        load: float,
        constraints: Dict[str, Any] = None
    ) -> OptimalRatioResult:
        """
        Compute optimal air-fuel ratio based on fuel type and load.

        DETERMINISTIC: Lookup stoichiometric ratio + excess air calculation

        Args:
            fuel_type: Type of fuel (from STOICHIOMETRIC_RATIOS keys)
            load: Current load as percentage of maximum (0-100)
            constraints: Optional operational constraints

        Returns:
            OptimalRatioResult with recommended ratio and provenance
        """
        if constraints is None:
            constraints = {}

        # Normalize fuel type
        fuel_type_normalized = fuel_type.lower().replace(' ', '_').replace('-', '_')

        # Step 1: Lookup stoichiometric ratio (DETERMINISTIC)
        if fuel_type_normalized not in STOICHIOMETRIC_RATIOS:
            # Fallback to natural gas if unknown
            stoich_ratio = STOICHIOMETRIC_RATIOS['natural_gas']
            confidence = 0.5  # Lower confidence for fallback
        else:
            stoich_ratio = STOICHIOMETRIC_RATIOS[fuel_type_normalized]
            confidence = 0.95

        # Step 2: Determine load category
        if load < 40.0:
            load_category = 'low_load'
        elif load < 80.0:
            load_category = 'mid_load'
        else:
            load_category = 'high_load'

        # Step 3: Lookup excess air recommendation (DETERMINISTIC)
        if fuel_type_normalized in EXCESS_AIR_RECOMMENDATIONS:
            excess_air = EXCESS_AIR_RECOMMENDATIONS[fuel_type_normalized][load_category]
        else:
            # Default excess air values
            default_excess = {'low_load': Decimal('20.0'), 'mid_load': Decimal('15.0'), 'high_load': Decimal('10.0')}
            excess_air = default_excess[load_category]
            confidence *= 0.8  # Reduce confidence for default values

        # Step 4: Apply constraints
        if 'min_excess_air' in constraints:
            min_ea = Decimal(str(constraints['min_excess_air']))
            excess_air = max(excess_air, min_ea)
        if 'max_excess_air' in constraints:
            max_ea = Decimal(str(constraints['max_excess_air']))
            excess_air = min(excess_air, max_ea)

        # Step 5: Calculate optimal ratio
        # optimal = stoichiometric * (1 + excess_air/100)
        excess_factor = Decimal('1') + (excess_air / Decimal('100'))
        optimal_ratio = self._quantize(stoich_ratio * excess_factor)

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'fuel_type': fuel_type_normalized,
            'load': load,
            'constraints': str(constraints),
            'stoich_ratio': str(stoich_ratio),
            'excess_air': str(excess_air),
            'optimal_ratio': str(optimal_ratio)
        })

        return OptimalRatioResult(
            optimal_ratio=optimal_ratio,
            stoichiometric_ratio=stoich_ratio,
            excess_air_percent=self._quantize(excess_air),
            confidence=confidence,
            provenance_hash=provenance
        )

    def compute_o2_trim_bias(
        self,
        target_o2: float,
        current_o2: float,
        gain: float = 1.0
    ) -> O2TrimResult:
        """
        Compute O2 trim bias for air-fuel ratio adjustment.

        DETERMINISTIC: (target - current) * gain

        Args:
            target_o2: Target O2 percentage in flue gas (0-21%)
            current_o2: Current O2 percentage in flue gas (0-21%)
            gain: Controller gain factor (default 1.0)

        Returns:
            O2TrimResult with bias value and direction
        """
        # Validate input
        input_data = O2TrimInput(target_o2=target_o2, current_o2=current_o2, gain=gain)

        # Convert to Decimal
        target = Decimal(str(input_data.target_o2))
        current = Decimal(str(input_data.current_o2))
        gain_dec = Decimal(str(input_data.gain))

        # Step 1: Compute error (DETERMINISTIC)
        error = target - current

        # Step 2: Apply gain (DETERMINISTIC)
        bias_raw = error * gain_dec
        bias_value = self._quantize(bias_raw)

        # Step 3: Determine direction
        if bias_value > Decimal('0.05'):
            direction = 'increase_air'  # Need more air to increase O2
        elif bias_value < Decimal('-0.05'):
            direction = 'decrease_air'  # Need less air to decrease O2
        else:
            direction = 'hold'  # Within deadband

        # Step 4: Calculate magnitude as percentage of typical adjustment
        # Typical O2 range is 2-8%, so we normalize by that range
        magnitude_percent = self._quantize(abs(bias_value) / Decimal('6') * Decimal('100'))

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'target_o2': target_o2,
            'current_o2': current_o2,
            'gain': gain,
            'bias_value': str(bias_value),
            'direction': direction
        })

        return O2TrimResult(
            bias_value=bias_value,
            direction=direction,
            magnitude_percent=magnitude_percent,
            provenance_hash=provenance
        )

    def compute_ratio_deviation(
        self,
        actual: float,
        target: float
    ) -> Tuple[Decimal, Decimal]:
        """
        Compute deviation between actual and target ratio.

        DETERMINISTIC: Returns absolute and percentage deviation

        Args:
            actual: Actual air-fuel ratio
            target: Target air-fuel ratio

        Returns:
            Tuple of (absolute_deviation, percentage_deviation)
        """
        actual_dec = Decimal(str(actual))
        target_dec = Decimal(str(target))

        # Absolute deviation
        abs_deviation = self._quantize(actual_dec - target_dec)

        # Percentage deviation (relative to target)
        if target_dec != Decimal('0'):
            pct_deviation = self._quantize((abs_deviation / target_dec) * Decimal('100'))
        else:
            pct_deviation = Decimal('0')

        return abs_deviation, pct_deviation

    def track_ratio_stability(
        self,
        history: List[float],
        stability_threshold: float = 5.0
    ) -> StabilityMetrics:
        """
        Track air-fuel ratio stability over historical data.

        DETERMINISTIC: Statistical analysis of ratio history

        Args:
            history: List of historical ratio values
            stability_threshold: CV threshold for stability (default 5%)

        Returns:
            StabilityMetrics with mean, std, CV, and stability assessment
        """
        if not history or len(history) < 2:
            # Not enough data for stability analysis
            provenance = self._compute_provenance_hash({'history': history, 'error': 'insufficient_data'})
            return StabilityMetrics(
                mean_ratio=Decimal('0'),
                std_deviation=Decimal('0'),
                coefficient_of_variation=Decimal('0'),
                max_deviation=Decimal('0'),
                min_ratio=Decimal('0'),
                max_ratio=Decimal('0'),
                is_stable=False,
                stability_score=0.0,
                sample_count=len(history) if history else 0,
                provenance_hash=provenance
            )

        # Convert to Decimal for precision
        history_decimal = [Decimal(str(v)) for v in history]

        # Step 1: Compute mean (DETERMINISTIC)
        mean_val = sum(history_decimal) / len(history_decimal)
        mean_rounded = self._quantize(mean_val)

        # Step 2: Compute standard deviation (DETERMINISTIC)
        variance = sum((x - mean_val) ** 2 for x in history_decimal) / len(history_decimal)
        std_dev = variance.sqrt()
        std_rounded = self._quantize(std_dev)

        # Step 3: Compute coefficient of variation (DETERMINISTIC)
        if mean_val != Decimal('0'):
            cv = (std_dev / mean_val) * Decimal('100')
        else:
            cv = Decimal('0')
        cv_rounded = self._quantize(cv)

        # Step 4: Compute min, max, and max deviation (DETERMINISTIC)
        min_ratio = min(history_decimal)
        max_ratio = max(history_decimal)
        deviations = [abs(x - mean_val) for x in history_decimal]
        max_deviation = max(deviations)

        # Step 5: Assess stability (DETERMINISTIC rule)
        is_stable = cv_rounded <= Decimal(str(stability_threshold))

        # Step 6: Calculate stability score (0-100)
        # Score is inversely proportional to CV
        # CV of 0 = score of 100, CV of 10+ = score of 0
        if cv_rounded >= Decimal('10'):
            stability_score = 0.0
        else:
            stability_score = float((Decimal('10') - cv_rounded) / Decimal('10') * Decimal('100'))

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'history_length': len(history),
            'mean': str(mean_rounded),
            'std': str(std_rounded),
            'cv': str(cv_rounded),
            'is_stable': is_stable
        })

        return StabilityMetrics(
            mean_ratio=mean_rounded,
            std_deviation=std_rounded,
            coefficient_of_variation=cv_rounded,
            max_deviation=self._quantize(max_deviation),
            min_ratio=self._quantize(min_ratio),
            max_ratio=self._quantize(max_ratio),
            is_stable=is_stable,
            stability_score=round(stability_score, 1),
            sample_count=len(history),
            provenance_hash=provenance
        )

    # -------------------------------------------------------------------------
    # Batch Processing Methods
    # -------------------------------------------------------------------------

    def compute_ratios_batch(
        self,
        measurements: List[Dict[str, float]]
    ) -> List[RatioCalculationResult]:
        """
        Compute air-fuel ratios for a batch of measurements.

        Args:
            measurements: List of dicts with 'fuel_flow' and 'air_flow' keys

        Returns:
            List of RatioCalculationResult for each measurement
        """
        results = []
        for m in measurements:
            result = self.compute_current_ratio(
                fuel_flow=m.get('fuel_flow', 0.0),
                air_flow=m.get('air_flow', 0.0)
            )
            results.append(result)
        return results
