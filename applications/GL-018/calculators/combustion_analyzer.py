"""
GL-018 FLUEFLOW - Combustion Analyzer

Zero-hallucination, deterministic calculations for flue gas analysis
and combustion characterization following ASME PTC 4.1 standards.

This module provides:
- O2, CO2, CO, NOx measurement validation
- Dry vs wet gas conversions
- Stoichiometric combustion calculations
- Excess air calculations
- Flue gas volume calculations

Standards Reference:
- ASME PTC 4.1 - Fired Steam Generators Performance Test Code
- EPA Method 19 - Determination of SO2 Removal Efficiency
- ISO 10396 - Stationary Source Emissions - Sampling
- EN 14181 - Quality Assurance of Automated Measuring Systems

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

# Gas constants
UNIVERSAL_GAS_CONSTANT = 8.314  # J/mol-K
STANDARD_TEMP_K = 273.15  # 0°C in Kelvin
STANDARD_PRESSURE_PA = 101325  # 1 atm in Pa

# Molecular weights (g/mol)
MW_O2 = 32.0
MW_N2 = 28.0
MW_CO2 = 44.0
MW_H2O = 18.0
MW_CO = 28.0
MW_SO2 = 64.0
MW_NOX = 46.0  # As NO2 equivalent
MW_AIR = 28.96

# Reference oxygen for excess air calculations
REFERENCE_O2_PERCENT = 21.0  # O2 in air (dry basis)

# Typical water vapor content in flue gas (for wet/dry conversions)
TYPICAL_H2O_PERCENT_WET = 10.0


class FuelType(Enum):
    """Supported fuel types for combustion analysis."""
    NATURAL_GAS = "Natural Gas"
    FUEL_OIL = "Fuel Oil"
    COAL = "Coal"
    BIOMASS = "Biomass"
    DIESEL = "Diesel"
    PROPANE = "Propane"


class GasBasis(Enum):
    """Gas measurement basis (wet or dry)."""
    WET = "Wet Basis"
    DRY = "Dry Basis"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class CombustionInput:
    """
    Input parameters for combustion analysis.

    Attributes:
        O2_pct: Oxygen concentration (%, dry basis)
        CO2_pct: Carbon dioxide concentration (%, dry basis)
        CO_ppm: Carbon monoxide concentration (ppm, dry basis)
        NOx_ppm: Nitrogen oxides concentration (ppm, dry basis)
        SO2_ppm: Sulfur dioxide concentration (ppm, dry basis, optional)
        flue_gas_temp_c: Flue gas temperature (°C)
        ambient_temp_c: Ambient air temperature (°C)
        fuel_type: Type of fuel being burned
        gas_basis: Measurement basis (wet or dry)
        h2o_pct_wet: Water vapor content if wet basis (%)
    """
    O2_pct: float
    CO2_pct: float
    CO_ppm: float
    NOx_ppm: float
    flue_gas_temp_c: float
    ambient_temp_c: float
    fuel_type: str = FuelType.NATURAL_GAS.value
    gas_basis: str = GasBasis.DRY.value
    SO2_ppm: float = 0.0
    h2o_pct_wet: Optional[float] = None


@dataclass(frozen=True)
class CombustionOutput:
    """
    Output results from combustion analysis.

    Attributes:
        excess_air_pct: Excess air percentage (%)
        excess_O2_pct: Excess oxygen percentage (%)
        stoichiometric_ratio: Actual/theoretical air ratio (lambda, λ)
        CO2_max_pct: Maximum theoretical CO2 (%)
        flue_gas_volume_nm3_kg: Flue gas volume (Nm³/kg fuel, dry)
        flue_gas_volume_nm3_kg_wet: Flue gas volume (Nm³/kg fuel, wet)
        O2_wet_pct: O2 concentration on wet basis (%)
        CO2_wet_pct: CO2 concentration on wet basis (%)
        combustion_quality_index: Combustion quality (0-100)
        combustion_quality_rating: Quality rating
        is_complete_combustion: True if CO < 400 ppm
        O2_dry_pct: O2 concentration on dry basis (%)
        CO2_dry_pct: CO2 concentration on dry basis (%)
    """
    excess_air_pct: float
    excess_O2_pct: float
    stoichiometric_ratio: float
    CO2_max_pct: float
    flue_gas_volume_nm3_kg: float
    flue_gas_volume_nm3_kg_wet: float
    O2_wet_pct: float
    CO2_wet_pct: float
    combustion_quality_index: float
    combustion_quality_rating: str
    is_complete_combustion: bool
    O2_dry_pct: float
    CO2_dry_pct: float


# =============================================================================
# FUEL PROPERTIES DATABASE
# =============================================================================

# Typical fuel properties for combustion calculations
# Format: {fuel_type: (C%, H%, O%, S%, N%, ash%, LHV_MJ/kg, CO2_max_pct)}
FUEL_PROPERTIES = {
    FuelType.NATURAL_GAS.value: {
        "C_pct": 75.0,  # Carbon content
        "H_pct": 25.0,  # Hydrogen content
        "O_pct": 0.0,   # Oxygen content
        "S_pct": 0.0,   # Sulfur content
        "N_pct": 0.0,   # Nitrogen content
        "ash_pct": 0.0, # Ash content
        "LHV_MJ_kg": 50.0,  # Lower heating value
        "CO2_max_pct": 11.8,  # Maximum theoretical CO2
        "stoich_air_kg_kg": 17.2,  # Stoichiometric air requirement
    },
    FuelType.FUEL_OIL.value: {
        "C_pct": 87.0,
        "H_pct": 12.5,
        "O_pct": 0.0,
        "S_pct": 0.5,
        "N_pct": 0.0,
        "ash_pct": 0.0,
        "LHV_MJ_kg": 42.0,
        "CO2_max_pct": 15.5,
        "stoich_air_kg_kg": 14.5,
    },
    FuelType.COAL.value: {
        "C_pct": 75.0,
        "H_pct": 5.0,
        "O_pct": 10.0,
        "S_pct": 1.0,
        "N_pct": 1.5,
        "ash_pct": 7.5,
        "LHV_MJ_kg": 25.0,
        "CO2_max_pct": 18.5,
        "stoich_air_kg_kg": 9.5,
    },
    FuelType.BIOMASS.value: {
        "C_pct": 50.0,
        "H_pct": 6.0,
        "O_pct": 43.0,
        "S_pct": 0.1,
        "N_pct": 0.5,
        "ash_pct": 0.4,
        "LHV_MJ_kg": 18.0,
        "CO2_max_pct": 20.2,
        "stoich_air_kg_kg": 6.0,
    },
    FuelType.DIESEL.value: {
        "C_pct": 86.0,
        "H_pct": 13.0,
        "O_pct": 0.0,
        "S_pct": 1.0,
        "N_pct": 0.0,
        "ash_pct": 0.0,
        "LHV_MJ_kg": 43.0,
        "CO2_max_pct": 15.3,
        "stoich_air_kg_kg": 14.3,
    },
    FuelType.PROPANE.value: {
        "C_pct": 81.8,
        "H_pct": 18.2,
        "O_pct": 0.0,
        "S_pct": 0.0,
        "N_pct": 0.0,
        "ash_pct": 0.0,
        "LHV_MJ_kg": 46.0,
        "CO2_max_pct": 13.7,
        "stoich_air_kg_kg": 15.7,
    },
}


# =============================================================================
# COMBUSTION ANALYZER CLASS
# =============================================================================

class CombustionAnalyzer:
    """
    Zero-hallucination combustion analyzer for flue gas analysis.

    Implements deterministic calculations following ASME PTC 4.1 and
    EPA Method 19 for combustion characterization. All calculations
    produce bit-perfect reproducible results with complete provenance.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> analyzer = CombustionAnalyzer()
        >>> inputs = CombustionInput(
        ...     O2_pct=3.5,
        ...     CO2_pct=12.0,
        ...     CO_ppm=50.0,
        ...     NOx_ppm=150.0,
        ...     flue_gas_temp_c=180.0,
        ...     ambient_temp_c=25.0,
        ...     fuel_type=FuelType.NATURAL_GAS.value
        ... )
        >>> result, provenance = analyzer.calculate(inputs)
        >>> print(f"Excess Air: {result.excess_air_pct:.1f}%")
    """

    VERSION = "1.0.0"
    NAME = "CombustionAnalyzer"

    def __init__(self):
        """Initialize the combustion analyzer."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: CombustionInput
    ) -> Tuple[CombustionOutput, ProvenanceRecord]:
        """
        Perform complete combustion analysis.

        Args:
            inputs: CombustionInput with all required parameters

        Returns:
            Tuple of (CombustionOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["ASME PTC 4.1", "EPA Method 19", "ISO 10396"],
                "domain": "Combustion Analysis"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "O2_pct": inputs.O2_pct,
            "CO2_pct": inputs.CO2_pct,
            "CO_ppm": inputs.CO_ppm,
            "NOx_ppm": inputs.NOx_ppm,
            "SO2_ppm": inputs.SO2_ppm,
            "flue_gas_temp_c": inputs.flue_gas_temp_c,
            "ambient_temp_c": inputs.ambient_temp_c,
            "fuel_type": inputs.fuel_type,
            "gas_basis": inputs.gas_basis,
            "h2o_pct_wet": inputs.h2o_pct_wet
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Get fuel properties
        fuel_props = FUEL_PROPERTIES.get(inputs.fuel_type)
        if fuel_props is None:
            raise ValueError(f"Unknown fuel type: {inputs.fuel_type}")

        # Step 1: Convert to dry basis if needed
        O2_dry, CO2_dry = self._convert_to_dry_basis(
            inputs.O2_pct,
            inputs.CO2_pct,
            inputs.gas_basis,
            inputs.h2o_pct_wet
        )

        # Step 2: Calculate excess air percentage
        excess_air_pct = self._calculate_excess_air(O2_dry)

        # Step 3: Calculate excess O2
        excess_O2 = self._calculate_excess_oxygen(O2_dry)

        # Step 4: Calculate stoichiometric ratio (lambda)
        stoich_ratio = self._calculate_stoichiometric_ratio(excess_air_pct)

        # Step 5: Get maximum theoretical CO2
        CO2_max = fuel_props["CO2_max_pct"]

        # Step 6: Calculate flue gas volumes
        flue_gas_vol_dry, flue_gas_vol_wet = self._calculate_flue_gas_volume(
            fuel_props,
            excess_air_pct,
            inputs.flue_gas_temp_c
        )

        # Step 7: Convert concentrations to wet basis
        h2o_pct = inputs.h2o_pct_wet if inputs.h2o_pct_wet else TYPICAL_H2O_PERCENT_WET
        O2_wet, CO2_wet = self._convert_to_wet_basis(O2_dry, CO2_dry, h2o_pct)

        # Step 8: Calculate combustion quality index
        quality_index = self._calculate_combustion_quality(
            O2_dry,
            CO2_dry,
            inputs.CO_ppm,
            fuel_props["CO2_max_pct"]
        )

        # Step 9: Determine quality rating
        quality_rating = self._determine_quality_rating(quality_index)

        # Step 10: Check for complete combustion
        is_complete = self._check_complete_combustion(inputs.CO_ppm)

        # Create output
        output = CombustionOutput(
            excess_air_pct=round(excess_air_pct, 2),
            excess_O2_pct=round(excess_O2, 2),
            stoichiometric_ratio=round(stoich_ratio, 3),
            CO2_max_pct=round(CO2_max, 2),
            flue_gas_volume_nm3_kg=round(flue_gas_vol_dry, 3),
            flue_gas_volume_nm3_kg_wet=round(flue_gas_vol_wet, 3),
            O2_wet_pct=round(O2_wet, 2),
            CO2_wet_pct=round(CO2_wet, 2),
            combustion_quality_index=round(quality_index, 1),
            combustion_quality_rating=quality_rating,
            is_complete_combustion=is_complete,
            O2_dry_pct=round(O2_dry, 2),
            CO2_dry_pct=round(CO2_dry, 2)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "excess_air_pct": output.excess_air_pct,
            "excess_O2_pct": output.excess_O2_pct,
            "stoichiometric_ratio": output.stoichiometric_ratio,
            "CO2_max_pct": output.CO2_max_pct,
            "flue_gas_volume_nm3_kg": output.flue_gas_volume_nm3_kg,
            "flue_gas_volume_nm3_kg_wet": output.flue_gas_volume_nm3_kg_wet,
            "O2_wet_pct": output.O2_wet_pct,
            "CO2_wet_pct": output.CO2_wet_pct,
            "combustion_quality_index": output.combustion_quality_index,
            "combustion_quality_rating": output.combustion_quality_rating,
            "is_complete_combustion": output.is_complete_combustion,
            "O2_dry_pct": output.O2_dry_pct,
            "CO2_dry_pct": output.CO2_dry_pct
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: CombustionInput) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If any input is invalid
        """
        if inputs.O2_pct < 0 or inputs.O2_pct > 21:
            raise ValueError(
                f"O2 concentration {inputs.O2_pct}% out of range (0-21%)"
            )

        if inputs.CO2_pct < 0 or inputs.CO2_pct > 20:
            raise ValueError(
                f"CO2 concentration {inputs.CO2_pct}% out of range (0-20%)"
            )

        if inputs.CO_ppm < 0:
            raise ValueError("CO concentration cannot be negative")

        if inputs.NOx_ppm < 0:
            raise ValueError("NOx concentration cannot be negative")

        if inputs.flue_gas_temp_c < 50 or inputs.flue_gas_temp_c > 1200:
            raise ValueError(
                f"Flue gas temperature {inputs.flue_gas_temp_c}°C out of range (50-1200°C)"
            )

        if inputs.ambient_temp_c < -20 or inputs.ambient_temp_c > 50:
            raise ValueError(
                f"Ambient temperature {inputs.ambient_temp_c}°C out of range (-20-50°C)"
            )

    def _convert_to_dry_basis(
        self,
        O2_measured: float,
        CO2_measured: float,
        basis: str,
        h2o_pct: Optional[float]
    ) -> Tuple[float, float]:
        """
        Convert gas concentrations to dry basis if needed.

        Formula (wet to dry):
            C_dry = C_wet / (1 - H2O/100)

        Args:
            O2_measured: Measured O2 concentration (%)
            CO2_measured: Measured CO2 concentration (%)
            basis: Measurement basis (wet or dry)
            h2o_pct: Water vapor content if wet basis (%)

        Returns:
            Tuple of (O2_dry, CO2_dry)
        """
        if basis == GasBasis.DRY.value:
            # Already dry basis
            O2_dry = O2_measured
            CO2_dry = CO2_measured

            self._tracker.add_step(
                step_number=1,
                description="Gas already on dry basis - no conversion needed",
                operation="passthrough",
                inputs={
                    "O2_measured_pct": O2_measured,
                    "CO2_measured_pct": CO2_measured,
                    "basis": basis
                },
                output_value=O2_dry,
                output_name="O2_dry_pct",
                formula="O2_dry = O2_measured (already dry)"
            )
        else:
            # Convert from wet to dry
            h2o = h2o_pct if h2o_pct else TYPICAL_H2O_PERCENT_WET
            dry_fraction = 1.0 - (h2o / 100.0)

            O2_dry = O2_measured / dry_fraction
            CO2_dry = CO2_measured / dry_fraction

            self._tracker.add_step(
                step_number=1,
                description="Convert wet basis to dry basis",
                operation="wet_to_dry_conversion",
                inputs={
                    "O2_wet_pct": O2_measured,
                    "CO2_wet_pct": CO2_measured,
                    "h2o_pct": h2o,
                    "dry_fraction": dry_fraction
                },
                output_value=O2_dry,
                output_name="O2_dry_pct",
                formula="C_dry = C_wet / (1 - H2O/100)"
            )

        return O2_dry, CO2_dry

    def _calculate_excess_air(self, O2_pct_dry: float) -> float:
        """
        Calculate excess air percentage from O2 measurement.

        Formula (ASME PTC 4.1):
            Excess_Air% = (O2_measured / (21 - O2_measured)) × 100

        This is a fundamental combustion equation relating measured
        oxygen to the amount of air supplied beyond stoichiometric.

        Args:
            O2_pct_dry: Oxygen concentration on dry basis (%)

        Returns:
            Excess air percentage (%)
        """
        excess_air = (O2_pct_dry / (21.0 - O2_pct_dry)) * 100.0

        self._tracker.add_step(
            step_number=2,
            description="Calculate excess air percentage",
            operation="excess_air_calc",
            inputs={
                "O2_pct_dry": O2_pct_dry,
                "reference_O2_pct": REFERENCE_O2_PERCENT
            },
            output_value=excess_air,
            output_name="excess_air_pct",
            formula="Excess_Air% = (O2 / (21 - O2)) × 100"
        )

        return excess_air

    def _calculate_excess_oxygen(self, O2_pct_dry: float) -> float:
        """
        Calculate excess oxygen percentage.

        This is simply the measured O2 minus O2 at stoichiometric (0%).

        Args:
            O2_pct_dry: Oxygen concentration on dry basis (%)

        Returns:
            Excess oxygen percentage (%)
        """
        excess_O2 = O2_pct_dry  # Since stoichiometric O2 = 0

        self._tracker.add_step(
            step_number=3,
            description="Calculate excess oxygen",
            operation="excess_oxygen",
            inputs={
                "O2_pct_dry": O2_pct_dry
            },
            output_value=excess_O2,
            output_name="excess_O2_pct",
            formula="Excess_O2 = O2_measured (stoich O2 = 0)"
        )

        return excess_O2

    def _calculate_stoichiometric_ratio(self, excess_air_pct: float) -> float:
        """
        Calculate stoichiometric ratio (lambda, λ).

        Formula:
            λ = Actual_Air / Theoretical_Air = 1 + (Excess_Air / 100)

        Args:
            excess_air_pct: Excess air percentage (%)

        Returns:
            Stoichiometric ratio (dimensionless)
        """
        lambda_ratio = 1.0 + (excess_air_pct / 100.0)

        self._tracker.add_step(
            step_number=4,
            description="Calculate stoichiometric ratio (lambda)",
            operation="lambda_calculation",
            inputs={
                "excess_air_pct": excess_air_pct
            },
            output_value=lambda_ratio,
            output_name="stoichiometric_ratio",
            formula="λ = 1 + (Excess_Air / 100)"
        )

        return lambda_ratio

    def _calculate_flue_gas_volume(
        self,
        fuel_props: Dict[str, float],
        excess_air_pct: float,
        flue_gas_temp_c: float
    ) -> Tuple[float, float]:
        """
        Calculate flue gas volume per kg of fuel.

        Uses simplified correlation for flue gas volume based on
        stoichiometric air and excess air.

        Formula:
            V_fg_dry = (stoich_air × (1 + EA/100) × 0.85) Nm³/kg fuel
            V_fg_wet = V_fg_dry × (1 + 0.1)  # Approximate 10% moisture

        Args:
            fuel_props: Fuel properties dictionary
            excess_air_pct: Excess air percentage (%)
            flue_gas_temp_c: Flue gas temperature (°C)

        Returns:
            Tuple of (volume_dry, volume_wet) in Nm³/kg fuel
        """
        stoich_air = fuel_props["stoich_air_kg_kg"]

        # Flue gas volume (dry basis) - simplified correlation
        # Approximately 0.85 Nm³ flue gas per kg air
        vol_dry = stoich_air * (1.0 + excess_air_pct / 100.0) * 0.85

        # Flue gas volume (wet basis) - add ~10% for moisture
        vol_wet = vol_dry * 1.10

        self._tracker.add_step(
            step_number=5,
            description="Calculate flue gas volume",
            operation="flue_gas_volume",
            inputs={
                "stoich_air_kg_kg": stoich_air,
                "excess_air_pct": excess_air_pct,
                "conversion_factor": 0.85
            },
            output_value=vol_dry,
            output_name="flue_gas_volume_nm3_kg",
            formula="V_fg = stoich_air × (1 + EA/100) × 0.85"
        )

        return vol_dry, vol_wet

    def _convert_to_wet_basis(
        self,
        O2_dry: float,
        CO2_dry: float,
        h2o_pct: float
    ) -> Tuple[float, float]:
        """
        Convert gas concentrations from dry to wet basis.

        Formula (dry to wet):
            C_wet = C_dry × (1 - H2O/100)

        Args:
            O2_dry: O2 concentration on dry basis (%)
            CO2_dry: CO2 concentration on dry basis (%)
            h2o_pct: Water vapor content (%)

        Returns:
            Tuple of (O2_wet, CO2_wet)
        """
        wet_fraction = 1.0 - (h2o_pct / 100.0)

        O2_wet = O2_dry * wet_fraction
        CO2_wet = CO2_dry * wet_fraction

        self._tracker.add_step(
            step_number=6,
            description="Convert dry basis to wet basis",
            operation="dry_to_wet_conversion",
            inputs={
                "O2_dry_pct": O2_dry,
                "CO2_dry_pct": CO2_dry,
                "h2o_pct": h2o_pct,
                "wet_fraction": wet_fraction
            },
            output_value=O2_wet,
            output_name="O2_wet_pct",
            formula="C_wet = C_dry × (1 - H2O/100)"
        )

        return O2_wet, CO2_wet

    def _calculate_combustion_quality(
        self,
        O2_pct: float,
        CO2_pct: float,
        CO_ppm: float,
        CO2_max_pct: float
    ) -> float:
        """
        Calculate combustion quality index (0-100).

        Quality index considers:
        - CO2/CO2_max ratio (50% weight) - higher is better
        - CO concentration (30% weight) - lower is better
        - O2 level (20% weight) - moderate is better

        Args:
            O2_pct: O2 concentration (%)
            CO2_pct: CO2 concentration (%)
            CO_ppm: CO concentration (ppm)
            CO2_max_pct: Maximum theoretical CO2 (%)

        Returns:
            Quality index (0-100)
        """
        # CO2 efficiency score (0-50)
        co2_score = (CO2_pct / CO2_max_pct) * 50.0
        co2_score = min(co2_score, 50.0)

        # CO score (0-30): penalize high CO
        # Good: CO < 50 ppm, Poor: CO > 400 ppm
        if CO_ppm < 50:
            co_score = 30.0
        elif CO_ppm < 400:
            co_score = 30.0 * (1.0 - (CO_ppm - 50) / 350.0)
        else:
            co_score = 0.0

        # O2 score (0-20): optimal range 2-4%
        if 2.0 <= O2_pct <= 4.0:
            o2_score = 20.0
        elif O2_pct < 2.0:
            # Too lean - incomplete combustion risk
            o2_score = (O2_pct / 2.0) * 20.0
        else:
            # Too rich - excess air losses
            o2_score = 20.0 * max(0, 1.0 - (O2_pct - 4.0) / 10.0)

        quality_index = co2_score + co_score + o2_score

        self._tracker.add_step(
            step_number=7,
            description="Calculate combustion quality index",
            operation="quality_index",
            inputs={
                "O2_pct": O2_pct,
                "CO2_pct": CO2_pct,
                "CO_ppm": CO_ppm,
                "CO2_max_pct": CO2_max_pct,
                "co2_score": co2_score,
                "co_score": co_score,
                "o2_score": o2_score
            },
            output_value=quality_index,
            output_name="combustion_quality_index",
            formula="Quality = CO2_score(50%) + CO_score(30%) + O2_score(20%)"
        )

        return quality_index

    def _determine_quality_rating(self, quality_index: float) -> str:
        """
        Determine quality rating from quality index.

        Ratings:
        - Excellent: >= 85
        - Good: 70-84
        - Fair: 55-69
        - Poor: 40-54
        - Critical: < 40

        Args:
            quality_index: Quality index (0-100)

        Returns:
            Quality rating string
        """
        if quality_index >= 85:
            rating = "Excellent"
        elif quality_index >= 70:
            rating = "Good"
        elif quality_index >= 55:
            rating = "Fair"
        elif quality_index >= 40:
            rating = "Poor"
        else:
            rating = "Critical"

        self._tracker.add_step(
            step_number=8,
            description="Determine combustion quality rating",
            operation="threshold_classification",
            inputs={
                "quality_index": quality_index,
                "thresholds": {
                    "excellent": 85,
                    "good": 70,
                    "fair": 55,
                    "poor": 40
                }
            },
            output_value=rating,
            output_name="combustion_quality_rating",
            formula="Rating based on quality index thresholds"
        )

        return rating

    def _check_complete_combustion(self, CO_ppm: float) -> bool:
        """
        Check if combustion is complete.

        Complete combustion criterion: CO < 400 ppm
        Per EPA and ASME guidelines.

        Args:
            CO_ppm: CO concentration (ppm)

        Returns:
            True if combustion is complete
        """
        is_complete = CO_ppm < 400.0

        self._tracker.add_step(
            step_number=9,
            description="Check for complete combustion",
            operation="threshold_check",
            inputs={
                "CO_ppm": CO_ppm,
                "threshold_ppm": 400.0
            },
            output_value=float(is_complete),
            output_name="is_complete_combustion",
            formula="Complete if CO < 400 ppm"
        )

        return is_complete


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_excess_air_from_O2(O2_pct_dry: float) -> float:
    """
    Calculate excess air from O2 measurement (standalone function).

    Formula:
        Excess_Air% = (O2 / (21 - O2)) × 100

    Args:
        O2_pct_dry: O2 concentration on dry basis (%)

    Returns:
        Excess air percentage (%)

    Example:
        >>> excess_air = calculate_excess_air_from_O2(3.5)
        >>> print(f"Excess Air: {excess_air:.1f}%")  # 20.0%
    """
    if O2_pct_dry < 0 or O2_pct_dry >= 21:
        raise ValueError(f"O2 must be in range 0-21%, got {O2_pct_dry}%")

    return (O2_pct_dry / (21.0 - O2_pct_dry)) * 100.0


def convert_wet_to_dry(
    concentration_wet: float,
    h2o_pct: float
) -> float:
    """
    Convert gas concentration from wet to dry basis.

    Formula:
        C_dry = C_wet / (1 - H2O/100)

    Args:
        concentration_wet: Concentration on wet basis (%)
        h2o_pct: Water vapor content (%)

    Returns:
        Concentration on dry basis (%)
    """
    if h2o_pct < 0 or h2o_pct >= 100:
        raise ValueError(f"H2O must be in range 0-100%, got {h2o_pct}%")

    dry_fraction = 1.0 - (h2o_pct / 100.0)
    return concentration_wet / dry_fraction


def convert_dry_to_wet(
    concentration_dry: float,
    h2o_pct: float
) -> float:
    """
    Convert gas concentration from dry to wet basis.

    Formula:
        C_wet = C_dry × (1 - H2O/100)

    Args:
        concentration_dry: Concentration on dry basis (%)
        h2o_pct: Water vapor content (%)

    Returns:
        Concentration on wet basis (%)
    """
    if h2o_pct < 0 or h2o_pct >= 100:
        raise ValueError(f"H2O must be in range 0-100%, got {h2o_pct}%")

    wet_fraction = 1.0 - (h2o_pct / 100.0)
    return concentration_dry * wet_fraction
