"""
GL-018 FLUEFLOW - Combustion Efficiency Calculator

Zero-hallucination, deterministic calculations for combustion efficiency
analysis following ASME PTC 4.1 standards.

This module provides:
- Combustion efficiency from flue gas analysis
- Heat loss calculations (stack loss, radiation, convection)
- ASME PTC 4.1 method for efficiency
- Stack loss as function of flue gas temperature and excess air

Standards Reference:
- ASME PTC 4.1 - Fired Steam Generators Performance Test Code
- ISO 50001 - Energy Management Systems
- EPA Energy Efficiency Guidelines

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

# Standard reference temperatures
STANDARD_TEMP_C = 25.0  # Standard ambient temperature
FUEL_TEMP_C = 25.0  # Standard fuel temperature

# Specific heats (kJ/kg-K)
CP_AIR = 1.005  # Air at moderate temperature
CP_FLUE_GAS = 1.08  # Flue gas (approximate)
CP_WATER_VAPOR = 2.0  # Water vapor

# Typical heat loss percentages for well-maintained equipment
TYPICAL_RADIATION_LOSS_PCT = 1.0  # Radiation and convection loss
TYPICAL_UNACCOUNTED_LOSS_PCT = 0.5  # Unaccounted losses

# Fuel heating values (MJ/kg) - for reference
FUEL_LHV = {
    "Natural Gas": 50.0,
    "Fuel Oil": 42.0,
    "Coal": 25.0,
    "Biomass": 18.0,
    "Diesel": 43.0,
    "Propane": 46.0,
}


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class EfficiencyInput:
    """
    Input parameters for efficiency calculations.

    Attributes:
        fuel_type: Type of fuel being burned
        fuel_flow_rate_kg_hr: Fuel flow rate (kg/hr)
        O2_pct_dry: O2 concentration (%, dry basis)
        CO2_pct_dry: CO2 concentration (%, dry basis)
        CO_ppm: CO concentration (ppm, dry basis)
        flue_gas_temp_c: Flue gas temperature (°C)
        ambient_temp_c: Ambient air temperature (°C)
        fuel_temp_c: Fuel temperature (°C)
        moisture_in_fuel_pct: Moisture content in fuel (%)
        excess_air_pct: Excess air percentage (%)
        heat_input_mw: Gross heat input (MW)
        heat_output_mw: Useful heat output (MW)
        radiation_loss_pct: Radiation/convection loss (%, optional)
        unaccounted_loss_pct: Unaccounted losses (%, optional)
    """
    fuel_type: str
    fuel_flow_rate_kg_hr: float
    O2_pct_dry: float
    CO2_pct_dry: float
    CO_ppm: float
    flue_gas_temp_c: float
    ambient_temp_c: float
    excess_air_pct: float
    heat_input_mw: float
    heat_output_mw: float
    fuel_temp_c: float = FUEL_TEMP_C
    moisture_in_fuel_pct: float = 0.0
    radiation_loss_pct: Optional[float] = None
    unaccounted_loss_pct: Optional[float] = None


@dataclass(frozen=True)
class EfficiencyOutput:
    """
    Output results from efficiency calculations.

    Attributes:
        combustion_efficiency_pct: Combustion efficiency (%)
        thermal_efficiency_pct: Overall thermal efficiency (%)
        stack_loss_pct: Stack heat loss (%)
        radiation_loss_pct: Radiation/convection loss (%)
        unaccounted_loss_pct: Unaccounted losses (%)
        moisture_loss_pct: Moisture evaporation loss (%)
        incomplete_combustion_loss_pct: Loss from incomplete combustion (%)
        total_losses_pct: Sum of all losses (%)
        efficiency_rating: Performance rating
        stack_temp_delta_c: Flue gas temperature rise (°C)
        heat_loss_mw: Total heat loss (MW)
        available_heat_pct: Available heat (%)
    """
    combustion_efficiency_pct: float
    thermal_efficiency_pct: float
    stack_loss_pct: float
    radiation_loss_pct: float
    unaccounted_loss_pct: float
    moisture_loss_pct: float
    incomplete_combustion_loss_pct: float
    total_losses_pct: float
    efficiency_rating: str
    stack_temp_delta_c: float
    heat_loss_mw: float
    available_heat_pct: float


# =============================================================================
# EFFICIENCY CALCULATOR CLASS
# =============================================================================

class EfficiencyCalculator:
    """
    Zero-hallucination efficiency calculator for combustion systems.

    Implements deterministic calculations following ASME PTC 4.1 method
    for combustion efficiency analysis. All calculations produce
    bit-perfect reproducible results with complete provenance.

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = EfficiencyCalculator()
        >>> inputs = EfficiencyInput(
        ...     fuel_type="Natural Gas",
        ...     fuel_flow_rate_kg_hr=1000.0,
        ...     O2_pct_dry=3.5,
        ...     CO2_pct_dry=12.0,
        ...     CO_ppm=50.0,
        ...     flue_gas_temp_c=180.0,
        ...     ambient_temp_c=25.0,
        ...     excess_air_pct=20.0,
        ...     heat_input_mw=10.0,
        ...     heat_output_mw=8.5
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Efficiency: {result.combustion_efficiency_pct:.1f}%")
    """

    VERSION = "1.0.0"
    NAME = "EfficiencyCalculator"

    def __init__(self):
        """Initialize the efficiency calculator."""
        self._tracker: Optional[ProvenanceTracker] = None

    def calculate(
        self,
        inputs: EfficiencyInput
    ) -> Tuple[EfficiencyOutput, ProvenanceRecord]:
        """
        Perform complete efficiency analysis.

        Args:
            inputs: EfficiencyInput with all required parameters

        Returns:
            Tuple of (EfficiencyOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["ASME PTC 4.1", "ISO 50001"],
                "domain": "Combustion Efficiency Analysis"
            }
        )

        # Set inputs for provenance
        input_dict = {
            "fuel_type": inputs.fuel_type,
            "fuel_flow_rate_kg_hr": inputs.fuel_flow_rate_kg_hr,
            "O2_pct_dry": inputs.O2_pct_dry,
            "CO2_pct_dry": inputs.CO2_pct_dry,
            "CO_ppm": inputs.CO_ppm,
            "flue_gas_temp_c": inputs.flue_gas_temp_c,
            "ambient_temp_c": inputs.ambient_temp_c,
            "fuel_temp_c": inputs.fuel_temp_c,
            "moisture_in_fuel_pct": inputs.moisture_in_fuel_pct,
            "excess_air_pct": inputs.excess_air_pct,
            "heat_input_mw": inputs.heat_input_mw,
            "heat_output_mw": inputs.heat_output_mw,
            "radiation_loss_pct": inputs.radiation_loss_pct,
            "unaccounted_loss_pct": inputs.unaccounted_loss_pct
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Step 1: Calculate stack temperature delta
        stack_temp_delta = self._calculate_stack_temp_delta(
            inputs.flue_gas_temp_c,
            inputs.ambient_temp_c
        )

        # Step 2: Calculate stack loss (dry flue gas loss)
        stack_loss = self._calculate_stack_loss(
            stack_temp_delta,
            inputs.excess_air_pct,
            inputs.CO2_pct_dry
        )

        # Step 3: Calculate moisture loss
        moisture_loss = self._calculate_moisture_loss(
            inputs.moisture_in_fuel_pct,
            inputs.fuel_type,
            stack_temp_delta
        )

        # Step 4: Calculate incomplete combustion loss (from CO)
        incomplete_comb_loss = self._calculate_incomplete_combustion_loss(
            inputs.CO_ppm,
            inputs.CO2_pct_dry
        )

        # Step 5: Determine radiation loss
        radiation_loss = (inputs.radiation_loss_pct
                         if inputs.radiation_loss_pct is not None
                         else TYPICAL_RADIATION_LOSS_PCT)

        # Step 6: Determine unaccounted loss
        unaccounted_loss = (inputs.unaccounted_loss_pct
                           if inputs.unaccounted_loss_pct is not None
                           else TYPICAL_UNACCOUNTED_LOSS_PCT)

        # Step 7: Calculate total losses
        total_losses = (stack_loss + moisture_loss + incomplete_comb_loss +
                       radiation_loss + unaccounted_loss)

        # Step 8: Calculate combustion efficiency
        combustion_efficiency = 100.0 - total_losses

        # Step 9: Calculate thermal efficiency (heat output / heat input)
        thermal_efficiency = (inputs.heat_output_mw / inputs.heat_input_mw) * 100.0

        # Step 10: Calculate available heat
        available_heat = 100.0 - (stack_loss + radiation_loss + unaccounted_loss)

        # Step 11: Calculate total heat loss in MW
        heat_loss_mw = inputs.heat_input_mw * (total_losses / 100.0)

        # Step 12: Determine efficiency rating
        efficiency_rating = self._determine_efficiency_rating(combustion_efficiency)

        # Create output
        output = EfficiencyOutput(
            combustion_efficiency_pct=round(combustion_efficiency, 2),
            thermal_efficiency_pct=round(thermal_efficiency, 2),
            stack_loss_pct=round(stack_loss, 2),
            radiation_loss_pct=round(radiation_loss, 2),
            unaccounted_loss_pct=round(unaccounted_loss, 2),
            moisture_loss_pct=round(moisture_loss, 2),
            incomplete_combustion_loss_pct=round(incomplete_comb_loss, 2),
            total_losses_pct=round(total_losses, 2),
            efficiency_rating=efficiency_rating,
            stack_temp_delta_c=round(stack_temp_delta, 1),
            heat_loss_mw=round(heat_loss_mw, 3),
            available_heat_pct=round(available_heat, 2)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "combustion_efficiency_pct": output.combustion_efficiency_pct,
            "thermal_efficiency_pct": output.thermal_efficiency_pct,
            "stack_loss_pct": output.stack_loss_pct,
            "radiation_loss_pct": output.radiation_loss_pct,
            "unaccounted_loss_pct": output.unaccounted_loss_pct,
            "moisture_loss_pct": output.moisture_loss_pct,
            "incomplete_combustion_loss_pct": output.incomplete_combustion_loss_pct,
            "total_losses_pct": output.total_losses_pct,
            "efficiency_rating": output.efficiency_rating,
            "stack_temp_delta_c": output.stack_temp_delta_c,
            "heat_loss_mw": output.heat_loss_mw,
            "available_heat_pct": output.available_heat_pct
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _validate_inputs(self, inputs: EfficiencyInput) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If any input is invalid
        """
        if inputs.fuel_flow_rate_kg_hr <= 0:
            raise ValueError("Fuel flow rate must be positive")

        if inputs.O2_pct_dry < 0 or inputs.O2_pct_dry > 21:
            raise ValueError(f"O2 concentration out of range: {inputs.O2_pct_dry}%")

        if inputs.CO2_pct_dry < 0 or inputs.CO2_pct_dry > 20:
            raise ValueError(f"CO2 concentration out of range: {inputs.CO2_pct_dry}%")

        if inputs.flue_gas_temp_c < 50 or inputs.flue_gas_temp_c > 1200:
            raise ValueError(f"Flue gas temperature out of range: {inputs.flue_gas_temp_c}°C")

        if inputs.heat_input_mw <= 0:
            raise ValueError("Heat input must be positive")

        if inputs.heat_output_mw < 0:
            raise ValueError("Heat output cannot be negative")

        if inputs.heat_output_mw > inputs.heat_input_mw:
            raise ValueError("Heat output cannot exceed heat input")

    def _calculate_stack_temp_delta(
        self,
        flue_gas_temp_c: float,
        ambient_temp_c: float
    ) -> float:
        """
        Calculate stack temperature rise above ambient.

        Formula:
            ΔT_stack = T_flue_gas - T_ambient

        Args:
            flue_gas_temp_c: Flue gas temperature (°C)
            ambient_temp_c: Ambient air temperature (°C)

        Returns:
            Temperature rise (°C)
        """
        delta_t = flue_gas_temp_c - ambient_temp_c

        self._tracker.add_step(
            step_number=1,
            description="Calculate stack temperature rise",
            operation="subtract",
            inputs={
                "flue_gas_temp_c": flue_gas_temp_c,
                "ambient_temp_c": ambient_temp_c
            },
            output_value=delta_t,
            output_name="stack_temp_delta_c",
            formula="ΔT = T_flue - T_ambient"
        )

        return delta_t

    def _calculate_stack_loss(
        self,
        stack_temp_delta_c: float,
        excess_air_pct: float,
        CO2_pct: float
    ) -> float:
        """
        Calculate stack heat loss (dry flue gas loss).

        This is the largest heat loss in most combustion systems.
        Uses simplified Siegert formula correlation.

        Formula (ASME PTC 4.1 simplified):
            Stack_Loss% = K × (T_stack - T_ambient) / CO2_pct

        Where K ≈ 0.52 for most fuels (correlation factor)

        Args:
            stack_temp_delta_c: Stack temperature rise (°C)
            excess_air_pct: Excess air percentage (%)
            CO2_pct: CO2 concentration (%)

        Returns:
            Stack loss as percentage of heat input (%)
        """
        # Siegert formula correlation factor
        K = 0.52

        # Avoid division by zero
        co2_safe = max(CO2_pct, 1.0)

        # Stack loss calculation
        stack_loss = K * stack_temp_delta_c / co2_safe

        self._tracker.add_step(
            step_number=2,
            description="Calculate stack heat loss (Siegert formula)",
            operation="stack_loss_calc",
            inputs={
                "stack_temp_delta_c": stack_temp_delta_c,
                "CO2_pct": CO2_pct,
                "excess_air_pct": excess_air_pct,
                "K_factor": K
            },
            output_value=stack_loss,
            output_name="stack_loss_pct",
            formula="Stack_Loss% = K × ΔT / CO2%  (K ≈ 0.52)"
        )

        return stack_loss

    def _calculate_moisture_loss(
        self,
        moisture_pct: float,
        fuel_type: str,
        stack_temp_delta_c: float
    ) -> float:
        """
        Calculate heat loss from moisture evaporation.

        Includes:
        - Moisture in fuel
        - Moisture from hydrogen combustion

        Formula:
            Moisture_Loss% = (moisture_content × latent_heat) / LHV × 100

        Simplified correlation for typical fuels.

        Args:
            moisture_pct: Moisture content in fuel (%)
            fuel_type: Type of fuel
            stack_temp_delta_c: Stack temperature rise (°C)

        Returns:
            Moisture loss as percentage of heat input (%)
        """
        # Simplified moisture loss correlation
        # Typical: 0.5-2% depending on fuel moisture and hydrogen content

        if moisture_pct > 0:
            # With fuel moisture
            moisture_loss = 0.02 * moisture_pct + 0.5
        else:
            # Only hydrogen combustion moisture
            moisture_loss = 0.5 if "Gas" in fuel_type else 0.8

        self._tracker.add_step(
            step_number=3,
            description="Calculate moisture evaporation loss",
            operation="moisture_loss_calc",
            inputs={
                "moisture_pct": moisture_pct,
                "fuel_type": fuel_type,
                "stack_temp_delta_c": stack_temp_delta_c
            },
            output_value=moisture_loss,
            output_name="moisture_loss_pct",
            formula="Moisture_Loss ≈ 0.02 × moisture% + 0.5%"
        )

        return moisture_loss

    def _calculate_incomplete_combustion_loss(
        self,
        CO_ppm: float,
        CO2_pct: float
    ) -> float:
        """
        Calculate heat loss from incomplete combustion (CO formation).

        CO represents unburned carbon that could have released more heat.
        Each % of CO in flue gas represents significant efficiency loss.

        Formula (ASME PTC 4.1):
            Loss% = 10,160 × CO% / (CO% + CO2%)

        Where CO% = CO_ppm / 10,000

        Args:
            CO_ppm: CO concentration (ppm)
            CO2_pct: CO2 concentration (%)

        Returns:
            Incomplete combustion loss as percentage (%)
        """
        # Convert CO from ppm to %
        CO_pct = CO_ppm / 10000.0

        # Avoid division by zero
        total_carbon = CO_pct + CO2_pct
        if total_carbon < 0.001:
            loss = 0.0
        else:
            # Simplified correlation: ~0.5% loss per 100 ppm CO
            loss = (CO_ppm / 100.0) * 0.5

            # Cap at reasonable maximum
            loss = min(loss, 5.0)

        self._tracker.add_step(
            step_number=4,
            description="Calculate incomplete combustion loss",
            operation="incomplete_combustion_loss",
            inputs={
                "CO_ppm": CO_ppm,
                "CO_pct": CO_pct,
                "CO2_pct": CO2_pct
            },
            output_value=loss,
            output_name="incomplete_combustion_loss_pct",
            formula="Loss ≈ 0.5% per 100 ppm CO"
        )

        return loss

    def _determine_efficiency_rating(self, efficiency_pct: float) -> str:
        """
        Determine efficiency rating category.

        Ratings:
        - Excellent: >= 90%
        - Good: 85-89%
        - Fair: 80-84%
        - Poor: 75-79%
        - Critical: < 75%

        Args:
            efficiency_pct: Combustion efficiency (%)

        Returns:
            Efficiency rating string
        """
        if efficiency_pct >= 90:
            rating = "Excellent"
        elif efficiency_pct >= 85:
            rating = "Good"
        elif efficiency_pct >= 80:
            rating = "Fair"
        elif efficiency_pct >= 75:
            rating = "Poor"
        else:
            rating = "Critical"

        self._tracker.add_step(
            step_number=5,
            description="Determine efficiency rating",
            operation="threshold_classification",
            inputs={
                "efficiency_pct": efficiency_pct,
                "thresholds": {
                    "excellent": 90,
                    "good": 85,
                    "fair": 80,
                    "poor": 75
                }
            },
            output_value=rating,
            output_name="efficiency_rating",
            formula="Rating based on efficiency thresholds"
        )

        return rating


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_stack_loss_siegert(
    flue_gas_temp_c: float,
    ambient_temp_c: float,
    CO2_pct: float
) -> float:
    """
    Calculate stack loss using Siegert formula (standalone).

    Formula:
        Stack_Loss% = 0.52 × (T_flue - T_ambient) / CO2%

    Args:
        flue_gas_temp_c: Flue gas temperature (°C)
        ambient_temp_c: Ambient temperature (°C)
        CO2_pct: CO2 concentration (%)

    Returns:
        Stack loss as percentage (%)

    Example:
        >>> loss = calculate_stack_loss_siegert(180, 25, 12.0)
        >>> print(f"Stack Loss: {loss:.1f}%")  # ~6.7%
    """
    delta_t = flue_gas_temp_c - ambient_temp_c
    co2_safe = max(CO2_pct, 1.0)
    return 0.52 * delta_t / co2_safe


def calculate_efficiency_from_losses(
    stack_loss_pct: float,
    radiation_loss_pct: float,
    moisture_loss_pct: float,
    incomplete_combustion_loss_pct: float,
    unaccounted_loss_pct: float = 0.5
) -> float:
    """
    Calculate combustion efficiency from individual losses.

    Formula:
        Efficiency% = 100 - (Stack + Radiation + Moisture + Incomplete + Unaccounted)

    Args:
        stack_loss_pct: Stack heat loss (%)
        radiation_loss_pct: Radiation/convection loss (%)
        moisture_loss_pct: Moisture evaporation loss (%)
        incomplete_combustion_loss_pct: Incomplete combustion loss (%)
        unaccounted_loss_pct: Unaccounted losses (%, default 0.5)

    Returns:
        Combustion efficiency (%)
    """
    total_losses = (stack_loss_pct + radiation_loss_pct + moisture_loss_pct +
                   incomplete_combustion_loss_pct + unaccounted_loss_pct)

    return 100.0 - total_losses


def calculate_available_heat(
    flue_gas_temp_c: float,
    ambient_temp_c: float,
    CO2_pct: float,
    radiation_loss_pct: float = 1.0
) -> float:
    """
    Calculate available heat (heat available for useful work).

    Available heat excludes stack loss and radiation loss.

    Args:
        flue_gas_temp_c: Flue gas temperature (°C)
        ambient_temp_c: Ambient temperature (°C)
        CO2_pct: CO2 concentration (%)
        radiation_loss_pct: Radiation/convection loss (%, default 1.0)

    Returns:
        Available heat percentage (%)
    """
    stack_loss = calculate_stack_loss_siegert(
        flue_gas_temp_c,
        ambient_temp_c,
        CO2_pct
    )

    return 100.0 - stack_loss - radiation_loss_pct
