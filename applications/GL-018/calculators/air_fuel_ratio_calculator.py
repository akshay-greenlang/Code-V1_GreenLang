"""
GL-018 FLUEFLOW - Air-Fuel Ratio Calculator

Zero-hallucination, deterministic calculations for air-fuel ratio analysis
and stoichiometric combustion calculations.

This module provides:
- Theoretical air requirements for fuels (natural gas, oil, coal)
- Actual air-fuel ratio from O2 measurements
- Stoichiometric calculations for different fuels
- Lambda (λ) calculation

Standards Reference:
- ASME PTC 4.1 - Fired Steam Generators Performance Test Code
- API 560 - Fired Heaters for General Refinery Service
- ISO 13790 - Energy Performance of Buildings

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Atomic masses (for stoichiometric calculations)
ATOMIC_MASS_C = 12.011
ATOMIC_MASS_H = 1.008
ATOMIC_MASS_O = 15.999
ATOMIC_MASS_S = 32.065
ATOMIC_MASS_N = 14.007

# Molecular masses
MW_O2 = 32.0
MW_N2 = 28.0
MW_AIR = 28.96

# Air composition (by volume, dry basis)
O2_IN_AIR_VOLUME_PCT = 21.0
N2_IN_AIR_VOLUME_PCT = 79.0

# Air composition (by mass)
O2_IN_AIR_MASS_PCT = 23.2
N2_IN_AIR_MASS_PCT = 76.8


# =============================================================================
# FUEL COMPOSITION DATABASE
# =============================================================================

# Typical fuel compositions for stoichiometric calculations
# Format: {fuel_type: ultimate_analysis}
FUEL_COMPOSITIONS = {
    "Natural Gas": {
        "C_pct": 75.0,
        "H_pct": 25.0,
        "O_pct": 0.0,
        "S_pct": 0.0,
        "N_pct": 0.0,
        "ash_pct": 0.0,
        "moisture_pct": 0.0,
        "LHV_MJ_kg": 50.0,
    },
    "Fuel Oil": {
        "C_pct": 87.0,
        "H_pct": 12.5,
        "O_pct": 0.0,
        "S_pct": 0.5,
        "N_pct": 0.0,
        "ash_pct": 0.0,
        "moisture_pct": 0.0,
        "LHV_MJ_kg": 42.0,
    },
    "Coal": {
        "C_pct": 75.0,
        "H_pct": 5.0,
        "O_pct": 10.0,
        "S_pct": 1.0,
        "N_pct": 1.5,
        "ash_pct": 7.5,
        "moisture_pct": 10.0,
        "LHV_MJ_kg": 25.0,
    },
    "Biomass": {
        "C_pct": 50.0,
        "H_pct": 6.0,
        "O_pct": 43.0,
        "S_pct": 0.1,
        "N_pct": 0.5,
        "ash_pct": 0.4,
        "moisture_pct": 20.0,
        "LHV_MJ_kg": 18.0,
    },
    "Diesel": {
        "C_pct": 86.0,
        "H_pct": 13.0,
        "O_pct": 0.0,
        "S_pct": 1.0,
        "N_pct": 0.0,
        "ash_pct": 0.0,
        "moisture_pct": 0.0,
        "LHV_MJ_kg": 43.0,
    },
    "Propane": {
        "C_pct": 81.8,
        "H_pct": 18.2,
        "O_pct": 0.0,
        "S_pct": 0.0,
        "N_pct": 0.0,
        "ash_pct": 0.0,
        "moisture_pct": 0.0,
        "LHV_MJ_kg": 46.0,
    },
}


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class AirFuelRatioInput:
    """
    Input parameters for air-fuel ratio calculations.

    Attributes:
        fuel_type: Type of fuel being burned
        C_pct: Carbon content (%, dry basis, optional if fuel_type provided)
        H_pct: Hydrogen content (%, dry basis, optional)
        O_pct: Oxygen content (%, dry basis, optional)
        S_pct: Sulfur content (%, dry basis, optional)
        N_pct: Nitrogen content (%, dry basis, optional)
        moisture_pct: Moisture content (%)
        O2_measured_pct: Measured O2 in flue gas (%, dry basis)
        excess_air_pct: Measured excess air (%, optional - calculated if not provided)
    """
    fuel_type: str
    O2_measured_pct: float
    C_pct: Optional[float] = None
    H_pct: Optional[float] = None
    O_pct: Optional[float] = None
    S_pct: Optional[float] = None
    N_pct: Optional[float] = None
    moisture_pct: float = 0.0
    excess_air_pct: Optional[float] = None


@dataclass(frozen=True)
class AirFuelRatioOutput:
    """
    Output results from air-fuel ratio calculations.

    Attributes:
        theoretical_air_kg_kg: Theoretical air required (kg air/kg fuel)
        theoretical_air_nm3_kg: Theoretical air required (Nm³ air/kg fuel)
        actual_air_kg_kg: Actual air supplied (kg air/kg fuel)
        actual_air_nm3_kg: Actual air supplied (Nm³ air/kg fuel)
        excess_air_kg_kg: Excess air (kg air/kg fuel)
        excess_air_pct: Excess air percentage (%)
        lambda_ratio: Stoichiometric ratio (λ = actual/theoretical)
        fuel_air_ratio: Fuel-air ratio (kg fuel/kg air)
        O2_theoretical_pct: Theoretical O2 at stoichiometric (%)
        O2_actual_pct: Actual O2 in flue gas (%)
        air_requirement_rating: Air requirement rating
    """
    theoretical_air_kg_kg: float
    theoretical_air_nm3_kg: float
    actual_air_kg_kg: float
    actual_air_nm3_kg: float
    excess_air_kg_kg: float
    excess_air_pct: float
    lambda_ratio: float
    fuel_air_ratio: float
    O2_theoretical_pct: float
    O2_actual_pct: float
    air_requirement_rating: str


# =============================================================================
# AIR-FUEL RATIO CALCULATOR CLASS
# =============================================================================

class AirFuelRatioCalculator:
    """
    Zero-hallucination air-fuel ratio calculator.

    Implements deterministic stoichiometric calculations following
    ASME PTC 4.1 and API 560. All calculations produce bit-perfect
    reproducible results with complete provenance.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = AirFuelRatioCalculator()
        >>> inputs = AirFuelRatioInput(
        ...     fuel_type="Natural Gas",
        ...     O2_measured_pct=3.5
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Lambda: {result.lambda_ratio:.3f}")
    """

    VERSION = "1.0.0"
    NAME = "AirFuelRatioCalculator"

    def __init__(self):
        """Initialize the air-fuel ratio calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: AirFuelRatioInput
    ) -> Tuple[AirFuelRatioOutput, ProvenanceRecord]:
        """
        Perform complete air-fuel ratio analysis.

        Args:
            inputs: AirFuelRatioInput with required parameters

        Returns:
            Tuple of (AirFuelRatioOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["ASME PTC 4.1", "API 560"],
                "domain": "Air-Fuel Ratio Analysis"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "fuel_type": inputs.fuel_type,
            "O2_measured_pct": inputs.O2_measured_pct,
            "C_pct": inputs.C_pct,
            "H_pct": inputs.H_pct,
            "O_pct": inputs.O_pct,
            "S_pct": inputs.S_pct,
            "N_pct": inputs.N_pct,
            "moisture_pct": inputs.moisture_pct,
            "excess_air_pct": inputs.excess_air_pct
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Get fuel composition
        fuel_comp = self._get_fuel_composition(inputs)

        # Step 1: Calculate theoretical oxygen requirement
        O2_theoretical = self._calculate_theoretical_oxygen(fuel_comp)

        # Step 2: Calculate theoretical air requirement (mass basis)
        theoretical_air_kg = self._calculate_theoretical_air_mass(O2_theoretical)

        # Step 3: Calculate theoretical air requirement (volume basis)
        theoretical_air_nm3 = self._calculate_theoretical_air_volume(O2_theoretical)

        # Step 4: Calculate excess air from O2 measurement
        excess_air_pct = self._calculate_excess_air_from_O2(inputs.O2_measured_pct)

        # Step 5: Calculate lambda (λ)
        lambda_ratio = self._calculate_lambda(excess_air_pct)

        # Step 6: Calculate actual air requirement
        actual_air_kg = self._calculate_actual_air_mass(
            theoretical_air_kg,
            excess_air_pct
        )

        actual_air_nm3 = self._calculate_actual_air_volume(
            theoretical_air_nm3,
            excess_air_pct
        )

        # Step 7: Calculate excess air (absolute)
        excess_air_kg = actual_air_kg - theoretical_air_kg

        # Step 8: Calculate fuel-air ratio
        fuel_air_ratio = 1.0 / actual_air_kg

        # Step 9: Determine rating
        air_rating = self._determine_air_requirement_rating(lambda_ratio)

        # Create output
        output = AirFuelRatioOutput(
            theoretical_air_kg_kg=round(theoretical_air_kg, 3),
            theoretical_air_nm3_kg=round(theoretical_air_nm3, 3),
            actual_air_kg_kg=round(actual_air_kg, 3),
            actual_air_nm3_kg=round(actual_air_nm3, 3),
            excess_air_kg_kg=round(excess_air_kg, 3),
            excess_air_pct=round(excess_air_pct, 2),
            lambda_ratio=round(lambda_ratio, 3),
            fuel_air_ratio=round(fuel_air_ratio, 4),
            O2_theoretical_pct=0.0,  # At stoichiometric
            O2_actual_pct=round(inputs.O2_measured_pct, 2),
            air_requirement_rating=air_rating
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "theoretical_air_kg_kg": output.theoretical_air_kg_kg,
            "theoretical_air_nm3_kg": output.theoretical_air_nm3_kg,
            "actual_air_kg_kg": output.actual_air_kg_kg,
            "actual_air_nm3_kg": output.actual_air_nm3_kg,
            "excess_air_kg_kg": output.excess_air_kg_kg,
            "excess_air_pct": output.excess_air_pct,
            "lambda_ratio": output.lambda_ratio,
            "fuel_air_ratio": output.fuel_air_ratio,
            "O2_theoretical_pct": output.O2_theoretical_pct,
            "O2_actual_pct": output.O2_actual_pct,
            "air_requirement_rating": output.air_requirement_rating
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: AirFuelRatioInput) -> None:
        """Validate input parameters."""
        if inputs.O2_measured_pct < 0 or inputs.O2_measured_pct > 21:
            raise ValueError(
                f"O2 measurement out of range: {inputs.O2_measured_pct}%"
            )

        if inputs.moisture_pct < 0 or inputs.moisture_pct > 100:
            raise ValueError(
                f"Moisture content out of range: {inputs.moisture_pct}%"
            )

    def _get_fuel_composition(self, inputs: AirFuelRatioInput) -> Dict[str, float]:
        """Get fuel composition from inputs or database."""
        if inputs.fuel_type in FUEL_COMPOSITIONS:
            # Use database composition
            comp = FUEL_COMPOSITIONS[inputs.fuel_type].copy()

            # Override with custom values if provided
            if inputs.C_pct is not None:
                comp["C_pct"] = inputs.C_pct
            if inputs.H_pct is not None:
                comp["H_pct"] = inputs.H_pct
            if inputs.O_pct is not None:
                comp["O_pct"] = inputs.O_pct
            if inputs.S_pct is not None:
                comp["S_pct"] = inputs.S_pct
            if inputs.N_pct is not None:
                comp["N_pct"] = inputs.N_pct

            comp["moisture_pct"] = inputs.moisture_pct

            return comp
        else:
            # Custom fuel - require full composition
            if inputs.C_pct is None or inputs.H_pct is None:
                raise ValueError(
                    f"Unknown fuel type '{inputs.fuel_type}' - "
                    "C_pct and H_pct must be provided"
                )

            return {
                "C_pct": inputs.C_pct,
                "H_pct": inputs.H_pct,
                "O_pct": inputs.O_pct or 0.0,
                "S_pct": inputs.S_pct or 0.0,
                "N_pct": inputs.N_pct or 0.0,
                "ash_pct": 0.0,
                "moisture_pct": inputs.moisture_pct
            }

    def _calculate_theoretical_oxygen(
        self,
        fuel_comp: Dict[str, float]
    ) -> float:
        """
        Calculate theoretical oxygen requirement.

        Formula (ASME PTC 4.1):
            O2_theor = (2.67×C + 8×H - O + S) / 100  [kg O2 / kg fuel]

        Where:
            2.67 = 32/12 (MW_O2/MW_C)
            8 = (32/2)/(2) (H requires O2, produces H2O)

        Args:
            fuel_comp: Fuel composition dictionary

        Returns:
            Theoretical oxygen requirement (kg O2/kg fuel)
        """
        C = fuel_comp["C_pct"]
        H = fuel_comp["H_pct"]
        O = fuel_comp["O_pct"]
        S = fuel_comp["S_pct"]

        # Stoichiometric oxygen calculation
        O2_theor = (2.67 * C + 8.0 * H - O + S) / 100.0

        self._tracker.add_step(
            step_number=1,
            description="Calculate theoretical oxygen requirement",
            operation="stoichiometric_oxygen",
            inputs={
                "C_pct": C,
                "H_pct": H,
                "O_pct": O,
                "S_pct": S
            },
            output_value=O2_theor,
            output_name="O2_theoretical_kg_kg",
            formula="O2 = (2.67×C + 8×H - O + S) / 100"
        )

        return O2_theor

    def _calculate_theoretical_air_mass(
        self,
        O2_theoretical_kg: float
    ) -> float:
        """
        Calculate theoretical air requirement (mass basis).

        Formula:
            Air_theor = O2_theor / 0.232  [kg air / kg fuel]

        Where 0.232 is the mass fraction of O2 in air.

        Args:
            O2_theoretical_kg: Theoretical O2 requirement (kg/kg fuel)

        Returns:
            Theoretical air requirement (kg air/kg fuel)
        """
        air_theor = O2_theoretical_kg / (O2_IN_AIR_MASS_PCT / 100.0)

        self._tracker.add_step(
            step_number=2,
            description="Calculate theoretical air (mass basis)",
            operation="divide",
            inputs={
                "O2_theoretical_kg_kg": O2_theoretical_kg,
                "O2_in_air_mass_fraction": O2_IN_AIR_MASS_PCT / 100.0
            },
            output_value=air_theor,
            output_name="theoretical_air_kg_kg",
            formula="Air = O2 / 0.232"
        )

        return air_theor

    def _calculate_theoretical_air_volume(
        self,
        O2_theoretical_kg: float
    ) -> float:
        """
        Calculate theoretical air requirement (volume basis at STP).

        Formula:
            Air_theor_vol = O2_theor / (MW_O2 × 0.21) × 22.4  [Nm³/kg fuel]

        Where:
            22.4 L/mol is molar volume at STP
            0.21 is volume fraction of O2 in air

        Args:
            O2_theoretical_kg: Theoretical O2 requirement (kg/kg fuel)

        Returns:
            Theoretical air requirement (Nm³/kg fuel)
        """
        # Convert kg O2 to kmol O2
        O2_kmol = O2_theoretical_kg / MW_O2

        # Convert to volume (m³ at STP)
        O2_vol = O2_kmol * 22.4

        # Air volume (O2 is 21% by volume)
        air_vol = O2_vol / (O2_IN_AIR_VOLUME_PCT / 100.0)

        self._tracker.add_step(
            step_number=3,
            description="Calculate theoretical air (volume basis)",
            operation="volume_conversion",
            inputs={
                "O2_theoretical_kg_kg": O2_theoretical_kg,
                "MW_O2": MW_O2,
                "O2_kmol": O2_kmol,
                "O2_vol_nm3": O2_vol,
                "O2_in_air_vol_fraction": O2_IN_AIR_VOLUME_PCT / 100.0
            },
            output_value=air_vol,
            output_name="theoretical_air_nm3_kg",
            formula="Air = (O2_kg / MW_O2) × 22.4 / 0.21"
        )

        return air_vol

    def _calculate_excess_air_from_O2(self, O2_pct: float) -> float:
        """
        Calculate excess air percentage from O2 measurement.

        Formula:
            Excess_Air% = (O2 / (21 - O2)) × 100

        Args:
            O2_pct: Measured O2 in flue gas (%, dry basis)

        Returns:
            Excess air percentage (%)
        """
        excess_air = (O2_pct / (21.0 - O2_pct)) * 100.0

        self._tracker.add_step(
            step_number=4,
            description="Calculate excess air from O2 measurement",
            operation="excess_air_calc",
            inputs={
                "O2_measured_pct": O2_pct,
                "O2_in_air_pct": 21.0
            },
            output_value=excess_air,
            output_name="excess_air_pct",
            formula="Excess_Air% = (O2 / (21 - O2)) × 100"
        )

        return excess_air

    def _calculate_lambda(self, excess_air_pct: float) -> float:
        """
        Calculate lambda (λ) - stoichiometric ratio.

        Formula:
            λ = Actual_Air / Theoretical_Air = 1 + (Excess_Air / 100)

        Args:
            excess_air_pct: Excess air percentage (%)

        Returns:
            Lambda ratio (dimensionless)
        """
        lambda_val = 1.0 + (excess_air_pct / 100.0)

        self._tracker.add_step(
            step_number=5,
            description="Calculate lambda (stoichiometric ratio)",
            operation="lambda_calc",
            inputs={
                "excess_air_pct": excess_air_pct
            },
            output_value=lambda_val,
            output_name="lambda_ratio",
            formula="λ = 1 + (Excess_Air / 100)"
        )

        return lambda_val

    def _calculate_actual_air_mass(
        self,
        theoretical_air_kg: float,
        excess_air_pct: float
    ) -> float:
        """
        Calculate actual air supplied (mass basis).

        Formula:
            Actual_Air = Theoretical_Air × (1 + Excess_Air/100)

        Args:
            theoretical_air_kg: Theoretical air (kg/kg fuel)
            excess_air_pct: Excess air percentage (%)

        Returns:
            Actual air supplied (kg/kg fuel)
        """
        actual_air = theoretical_air_kg * (1.0 + excess_air_pct / 100.0)

        self._tracker.add_step(
            step_number=6,
            description="Calculate actual air supplied (mass)",
            operation="multiply",
            inputs={
                "theoretical_air_kg_kg": theoretical_air_kg,
                "excess_air_factor": 1.0 + excess_air_pct / 100.0
            },
            output_value=actual_air,
            output_name="actual_air_kg_kg",
            formula="Actual = Theoretical × (1 + EA/100)"
        )

        return actual_air

    def _calculate_actual_air_volume(
        self,
        theoretical_air_nm3: float,
        excess_air_pct: float
    ) -> float:
        """
        Calculate actual air supplied (volume basis).

        Formula:
            Actual_Air = Theoretical_Air × (1 + Excess_Air/100)

        Args:
            theoretical_air_nm3: Theoretical air (Nm³/kg fuel)
            excess_air_pct: Excess air percentage (%)

        Returns:
            Actual air supplied (Nm³/kg fuel)
        """
        actual_air = theoretical_air_nm3 * (1.0 + excess_air_pct / 100.0)

        self._tracker.add_step(
            step_number=7,
            description="Calculate actual air supplied (volume)",
            operation="multiply",
            inputs={
                "theoretical_air_nm3_kg": theoretical_air_nm3,
                "excess_air_factor": 1.0 + excess_air_pct / 100.0
            },
            output_value=actual_air,
            output_name="actual_air_nm3_kg",
            formula="Actual = Theoretical × (1 + EA/100)"
        )

        return actual_air

    def _determine_air_requirement_rating(self, lambda_ratio: float) -> str:
        """
        Determine air requirement rating from lambda.

        Ratings:
        - Optimal: 1.1 <= λ <= 1.2
        - Good: 1.05 <= λ < 1.1 or 1.2 < λ <= 1.3
        - Fair: 1.0 < λ < 1.05 or 1.3 < λ <= 1.5
        - Rich: λ < 1.0 (insufficient air)
        - Lean: λ > 1.5 (excessive air)

        Args:
            lambda_ratio: Lambda value

        Returns:
            Rating string
        """
        if 1.1 <= lambda_ratio <= 1.2:
            rating = "Optimal"
        elif (1.05 <= lambda_ratio < 1.1) or (1.2 < lambda_ratio <= 1.3):
            rating = "Good"
        elif (1.0 < lambda_ratio < 1.05) or (1.3 < lambda_ratio <= 1.5):
            rating = "Fair"
        elif lambda_ratio < 1.0:
            rating = "Rich (Insufficient Air)"
        else:
            rating = "Lean (Excessive Air)"

        self._tracker.add_step(
            step_number=8,
            description="Determine air requirement rating",
            operation="threshold_classification",
            inputs={
                "lambda_ratio": lambda_ratio,
                "thresholds": {
                    "optimal_min": 1.1,
                    "optimal_max": 1.2,
                    "good_min": 1.05,
                    "good_max": 1.3
                }
            },
            output_value=rating,
            output_name="air_requirement_rating",
            formula="Rating based on lambda thresholds"
        )

        return rating


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_theoretical_air_from_composition(
    C_pct: float,
    H_pct: float,
    O_pct: float = 0.0,
    S_pct: float = 0.0
) -> float:
    """
    Calculate theoretical air requirement from fuel composition.

    Formula:
        O2_theor = (2.67×C + 8×H - O + S) / 100
        Air_theor = O2_theor / 0.232

    Args:
        C_pct: Carbon content (%)
        H_pct: Hydrogen content (%)
        O_pct: Oxygen content (%)
        S_pct: Sulfur content (%)

    Returns:
        Theoretical air requirement (kg air/kg fuel)

    Example:
        >>> air = calculate_theoretical_air_from_composition(75.0, 25.0)
        >>> print(f"Theoretical Air: {air:.2f} kg/kg")
    """
    O2_theor = (2.67 * C_pct + 8.0 * H_pct - O_pct + S_pct) / 100.0
    air_theor = O2_theor / 0.232
    return air_theor


def calculate_lambda_from_O2(O2_pct_dry: float) -> float:
    """
    Calculate lambda from O2 measurement (standalone).

    Formula:
        λ = 1 + (O2 / (21 - O2))

    Args:
        O2_pct_dry: O2 in flue gas (%, dry basis)

    Returns:
        Lambda ratio

    Example:
        >>> lambda_val = calculate_lambda_from_O2(3.5)
        >>> print(f"Lambda: {lambda_val:.3f}")  # 1.200
    """
    excess_air_pct = (O2_pct_dry / (21.0 - O2_pct_dry)) * 100.0
    lambda_val = 1.0 + (excess_air_pct / 100.0)
    return lambda_val
