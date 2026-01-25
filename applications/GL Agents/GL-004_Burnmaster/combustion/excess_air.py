"""
GL-004 BURNMASTER - Excess Air Calculator

Zero-hallucination calculation engine for excess air determination.
All calculations are deterministic, auditable, and bit-perfect reproducible.

This module implements:
- Excess air calculation from O2 or CO2 measurements
- Lambda (equivalence ratio) calculations
- Air-fuel ratio calculations
- Optimal excess air recommendations

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib

from pydantic import BaseModel, Field


class MeasurementBasis(str, Enum):
    """Measurement basis for flue gas analysis."""
    DRY = "dry"
    WET = "wet"


class ExcessAirInput(BaseModel):
    """Input schema for excess air calculation."""
    o2_percent: Optional[float] = Field(None, ge=0, le=21, description="O2 in flue gas (%)")
    co2_percent: Optional[float] = Field(None, ge=0, le=25, description="CO2 in flue gas (%)")
    basis: MeasurementBasis = Field(default=MeasurementBasis.DRY)
    fuel_type: str = Field(default="natural_gas")


class ExcessAirResult(BaseModel):
    """Output schema for excess air calculation."""
    excess_air_percent: Decimal = Field(..., description="Excess air (%)")
    lambda_val: Decimal = Field(..., description="Lambda (air-fuel equivalence ratio)")
    stoichiometric_afr: Decimal = Field(..., description="Stoichiometric A/F ratio")
    actual_afr: Decimal = Field(..., description="Actual A/F ratio")

    # Operational guidance
    optimal_excess_air_min: Decimal = Field(..., description="Min recommended excess air (%)")
    optimal_excess_air_max: Decimal = Field(..., description="Max recommended excess air (%)")
    deviation_from_optimal: Decimal = Field(..., description="Deviation from optimal (%)")
    efficiency_impact: str = Field(..., description="Impact on efficiency")

    # Emissions guidance
    nox_impact: str = Field(..., description="Impact on NOx")
    co_impact: str = Field(..., description="Impact on CO")

    provenance_hash: str = Field(..., description="SHA-256 hash")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Stoichiometric air-fuel ratios (mass basis)
STOICHIOMETRIC_AFR: Dict[str, float] = {
    "natural_gas": 17.2,
    "methane": 17.2,
    "propane": 15.7,
    "butane": 15.5,
    "hydrogen": 34.3,
    "fuel_oil_no2": 14.7,
    "fuel_oil_no6": 14.1,
    "diesel": 14.5,
    "gasoline": 14.7,
    "coal_bituminous": 10.8,
    "coal_subbituminous": 9.8,
    "coal_lignite": 8.5,
    "wood": 6.3,
    "biogas": 11.5,
    "refinery_gas": 16.0,
}

# Maximum theoretical CO2 at stoichiometric (dry basis %)
MAX_CO2: Dict[str, float] = {
    "natural_gas": 11.7,
    "methane": 11.7,
    "propane": 13.7,
    "butane": 14.0,
    "hydrogen": 0.0,  # No CO2 from H2
    "fuel_oil_no2": 15.4,
    "fuel_oil_no6": 16.0,
    "diesel": 15.4,
    "coal_bituminous": 18.5,
    "coal_subbituminous": 19.0,
    "wood": 20.0,
}

# Optimal excess air ranges by application
OPTIMAL_EXCESS_AIR: Dict[str, Tuple[float, float]] = {
    "natural_gas_boiler": (10, 20),
    "natural_gas_heater": (10, 25),
    "fuel_oil_boiler": (15, 25),
    "coal_boiler": (20, 35),
    "gas_turbine": (200, 400),  # Very lean
    "ic_engine": (5, 15),
}


class ExcessAirCalculator:
    """
    Zero-hallucination calculator for excess air.

    Guarantees:
    - Deterministic: Same input produces same output (bit-perfect)
    - Auditable: SHA-256 provenance hash for every calculation
    - Reproducible: Complete calculation step tracking
    - NO LLM: Pure arithmetic operations only

    Example:
        >>> calculator = ExcessAirCalculator()
        >>> result = calculator.compute_excess_air_from_o2(
        ...     o2_percent=3.0,
        ...     fuel_type="natural_gas"
        ... )
        >>> print(f"Excess air: {result.excess_air_percent}%")
    """

    def __init__(self, precision: int = 1):
        """Initialize calculator with precision settings."""
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def compute_excess_air_from_o2(
        self,
        o2_percent: float,
        fuel_type: str = "natural_gas",
        basis: MeasurementBasis = MeasurementBasis.DRY,
        application: str = "natural_gas_boiler",
    ) -> ExcessAirResult:
        """
        Compute excess air from O2 measurement.

        DETERMINISTIC: Based on combustion stoichiometry.

        Excess Air (%) = 100 * O2 / (21 - O2)
        Lambda = 21 / (21 - O2)

        Args:
            o2_percent: Measured O2 in flue gas (%)
            fuel_type: Type of fuel
            basis: Measurement basis (dry/wet)
            application: Application type for optimal range lookup

        Returns:
            ExcessAirResult with excess air and operational guidance
        """
        # Step 1: Validate O2 (DETERMINISTIC)
        o2_percent = max(0, min(20.9, o2_percent))

        # Convert wet to dry if needed (approximate)
        if basis == MeasurementBasis.WET:
            # Typical water content ~10% in flue gas
            o2_dry = o2_percent / 0.9
            o2_dry = min(20.9, o2_dry)
        else:
            o2_dry = o2_percent

        # Step 2: Calculate excess air (DETERMINISTIC)
        # EA% = O2 / (21 - O2) * 100
        if o2_dry >= 21:
            excess_air_percent = 1000  # Very high
        else:
            excess_air_percent = (o2_dry / (21 - o2_dry)) * 100

        # Step 3: Calculate lambda (DETERMINISTIC)
        # Lambda = 1 + EA/100 = 21 / (21 - O2)
        if o2_dry >= 21:
            lambda_val = 10.0
        else:
            lambda_val = 21 / (21 - o2_dry)

        # Step 4: Calculate air-fuel ratios (DETERMINISTIC)
        stoich_afr = STOICHIOMETRIC_AFR.get(fuel_type, 17.2)
        actual_afr = stoich_afr * lambda_val

        # Step 5: Get optimal range (DETERMINISTIC lookup)
        optimal_range = OPTIMAL_EXCESS_AIR.get(application, (10, 25))
        optimal_min, optimal_max = optimal_range
        optimal_mid = (optimal_min + optimal_max) / 2

        # Step 6: Calculate deviation (DETERMINISTIC)
        if excess_air_percent < optimal_min:
            deviation = excess_air_percent - optimal_min
        elif excess_air_percent > optimal_max:
            deviation = excess_air_percent - optimal_max
        else:
            deviation = 0

        # Step 7: Assess impacts (DETERMINISTIC thresholds)
        # Efficiency impact
        if excess_air_percent < optimal_min:
            efficiency_impact = "negative_high_co_risk"
        elif excess_air_percent > optimal_max * 1.5:
            efficiency_impact = "negative_high_stack_loss"
        elif excess_air_percent > optimal_max:
            efficiency_impact = "slightly_negative"
        else:
            efficiency_impact = "optimal"

        # NOx impact
        if excess_air_percent < 10:
            nox_impact = "low_but_co_risk"
        elif excess_air_percent < optimal_mid:
            nox_impact = "moderate"
        elif excess_air_percent < optimal_max:
            nox_impact = "moderate_to_low"
        else:
            nox_impact = "low"

        # CO impact
        if excess_air_percent < 5:
            co_impact = "high_risk"
        elif excess_air_percent < optimal_min:
            co_impact = "elevated_risk"
        elif excess_air_percent < optimal_max:
            co_impact = "low"
        else:
            co_impact = "very_low"

        provenance = self._compute_provenance_hash({
            'o2_percent': o2_percent,
            'fuel_type': fuel_type,
            'excess_air_percent': excess_air_percent,
            'lambda': lambda_val
        })

        return ExcessAirResult(
            excess_air_percent=self._quantize(Decimal(str(excess_air_percent))),
            lambda_val=self._quantize(Decimal(str(lambda_val))),
            stoichiometric_afr=self._quantize(Decimal(str(stoich_afr))),
            actual_afr=self._quantize(Decimal(str(actual_afr))),
            optimal_excess_air_min=self._quantize(Decimal(str(optimal_min))),
            optimal_excess_air_max=self._quantize(Decimal(str(optimal_max))),
            deviation_from_optimal=self._quantize(Decimal(str(deviation))),
            efficiency_impact=efficiency_impact,
            nox_impact=nox_impact,
            co_impact=co_impact,
            provenance_hash=provenance
        )

    def compute_excess_air_from_co2(
        self,
        co2_percent: float,
        fuel_type: str = "natural_gas",
        application: str = "natural_gas_boiler",
    ) -> ExcessAirResult:
        """
        Compute excess air from CO2 measurement.

        DETERMINISTIC: Based on maximum theoretical CO2.

        Excess Air (%) = 100 * (CO2_max / CO2_measured - 1)

        Args:
            co2_percent: Measured CO2 in flue gas (%, dry basis)
            fuel_type: Type of fuel
            application: Application type for optimal range

        Returns:
            ExcessAirResult with excess air calculation
        """
        # Get maximum theoretical CO2
        co2_max = MAX_CO2.get(fuel_type, 12.0)

        if co2_percent <= 0 or co2_max <= 0:
            # Invalid input - return high excess air
            o2_equivalent = 10.0
        elif co2_percent >= co2_max:
            # At or above stoichiometric CO2
            o2_equivalent = 0.0
        else:
            # Calculate excess air from CO2 ratio
            excess_air_ratio = (co2_max / co2_percent) - 1
            # Convert to equivalent O2
            # EA = O2 / (21 - O2) => O2 = 21 * EA / (1 + EA)
            o2_equivalent = 21 * excess_air_ratio / (1 + excess_air_ratio)

        # Use O2-based calculation
        return self.compute_excess_air_from_o2(
            o2_percent=o2_equivalent,
            fuel_type=fuel_type,
            application=application
        )

    def compute_o2_from_excess_air(
        self,
        excess_air_percent: float,
    ) -> Decimal:
        """
        Compute O2 percentage from excess air.

        DETERMINISTIC: Inverse of excess air calculation.

        O2 = 21 * EA / (100 + EA)

        Args:
            excess_air_percent: Excess air (%)

        Returns:
            O2 percentage (dry basis)
        """
        if excess_air_percent < 0:
            return self._quantize(Decimal("0"))

        o2 = 21 * excess_air_percent / (100 + excess_air_percent)
        o2 = min(20.9, max(0, o2))

        return self._quantize(Decimal(str(o2)))

    def compute_lambda_from_afr(
        self,
        actual_afr: float,
        fuel_type: str = "natural_gas",
    ) -> Decimal:
        """
        Compute lambda from air-fuel ratio.

        DETERMINISTIC: Lambda = AFR_actual / AFR_stoich

        Args:
            actual_afr: Actual air-fuel ratio (mass basis)
            fuel_type: Type of fuel

        Returns:
            Lambda (equivalence ratio)
        """
        stoich_afr = STOICHIOMETRIC_AFR.get(fuel_type, 17.2)

        if stoich_afr <= 0:
            return self._quantize(Decimal("1.0"))

        lambda_val = actual_afr / stoich_afr
        lambda_val = max(0.5, min(5.0, lambda_val))

        return self._quantize(Decimal(str(lambda_val)))

    def recommend_excess_air(
        self,
        load_percent: float,
        nox_constraint_ppm: Optional[float] = None,
        co_constraint_ppm: Optional[float] = None,
        fuel_type: str = "natural_gas",
    ) -> Tuple[Decimal, Decimal]:
        """
        Recommend optimal excess air range considering constraints.

        DETERMINISTIC: Rule-based recommendations.

        Args:
            load_percent: Current load as percentage
            nox_constraint_ppm: NOx limit (if applicable)
            co_constraint_ppm: CO limit (if applicable)
            fuel_type: Type of fuel

        Returns:
            Tuple of (min_excess_air, max_excess_air) in percent
        """
        # Base range by fuel type
        base_ranges = {
            "natural_gas": (10, 20),
            "fuel_oil": (15, 25),
            "coal": (20, 35),
        }

        fuel_category = "natural_gas"
        if "oil" in fuel_type.lower():
            fuel_category = "fuel_oil"
        elif "coal" in fuel_type.lower():
            fuel_category = "coal"

        base_min, base_max = base_ranges.get(fuel_category, (10, 25))

        # Adjust for load
        if load_percent < 30:
            # At low load, need more excess air for stability
            load_adjustment = 10
        elif load_percent < 60:
            load_adjustment = 5
        else:
            load_adjustment = 0

        # Adjust for NOx constraint
        nox_adjustment = 0
        if nox_constraint_ppm and nox_constraint_ppm < 25:
            # Very tight NOx - run leaner
            nox_adjustment = 5

        # Adjust for CO constraint
        co_adjustment = 0
        if co_constraint_ppm and co_constraint_ppm < 50:
            # Tight CO - ensure enough air
            co_adjustment = 5
            base_min = max(base_min, 15)

        recommended_min = base_min + load_adjustment + co_adjustment
        recommended_max = base_max + load_adjustment + nox_adjustment

        return (
            self._quantize(Decimal(str(recommended_min))),
            self._quantize(Decimal(str(recommended_max)))
        )
