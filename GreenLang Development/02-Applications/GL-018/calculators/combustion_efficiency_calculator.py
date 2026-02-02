# -*- coding: utf-8 -*-
"""
GL-018 FLUEFLOW - Combustion Efficiency Calculator

Zero-hallucination, deterministic calculations for combustion efficiency
analysis following ASME PTC 4.1 standards.

This module provides:
- Siegert formula implementation (ASME PTC 4.1)
- Heat loss method (indirect efficiency)
- Dry gas loss calculation
- Moisture loss (fuel moisture, combustion moisture)
- Unburned carbon loss
- Radiation and convection losses
- Sensible heat in ash

Standards Reference:
- ASME PTC 4.1 - Fired Steam Generators Performance Test Code
- ASME PTC 4 - Performance Test Code for Fired Steam Generators
- ISO 50001 - Energy Management Systems
- EN 12952-15 - Water-tube Boilers: Acceptance Tests
- EPA Energy Efficiency Guidelines

Formula Derivations:

    Siegert Formula (Dry Flue Gas Loss):
        L_dfg = K * (T_fg - T_a) / CO2%

        Where K is fuel-specific constant:
        - Natural gas: K = 0.38
        - Fuel oil: K = 0.44
        - Coal: K = 0.52

    Heat Loss Method (Indirect Efficiency):
        Efficiency = 100 - (L_dfg + L_mH2O + L_mf + L_CO + L_UC + L_rad + L_ash)

        Where:
        - L_dfg: Dry flue gas loss
        - L_mH2O: Loss from H2O in combustion products (hydrogen combustion)
        - L_mf: Loss from moisture in fuel
        - L_CO: Loss from incomplete combustion (CO formation)
        - L_UC: Loss from unburned carbon
        - L_rad: Radiation and convection loss
        - L_ash: Sensible heat in ash

    Dry Flue Gas Loss (detailed):
        L_dfg = (m_dfg * Cp_dfg * (T_fg - T_a)) / HHV * 100

    Moisture Loss from Hydrogen Combustion:
        L_mH2O = (9 * H% / 100) * (hfg + Cp_steam * (T_fg - T_ref)) / HHV * 100

    Moisture Loss from Fuel Moisture:
        L_mf = (M% / 100) * (hfg + Cp_steam * (T_fg - T_ref)) / HHV * 100

    CO Loss:
        L_CO = (CO / (CO + CO2)) * 5654 * C% / HHV * 100

    Unburned Carbon Loss:
        L_UC = 33820 * (ash% * UBC%) / HHV * 100

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Reference temperature (deg C)
REFERENCE_TEMP_C = 25.0

# Latent heat of water vaporization at 25 deg C (kJ/kg)
LATENT_HEAT_H2O_25C = 2442.0

# Specific heats (kJ/kg-K)
CP_AIR = 1.005
CP_FLUE_GAS = 1.08
CP_WATER_VAPOR = 2.01
CP_ASH = 0.84

# Heating value of CO vs complete combustion (kJ/kg C)
# CO: 10,110 kJ/kg C, CO2: 32,790 kJ/kg C
HEATING_VALUE_CO = 10110.0  # kJ/kg C
HEATING_VALUE_C_COMPLETE = 32790.0  # kJ/kg C

# Carbon loss per kg unburned carbon (kJ/kg)
HEATING_VALUE_UNBURNED_CARBON = 33820.0

# Typical radiation losses by equipment type (%)
RADIATION_LOSS_TYPICAL = {
    "watertube_boiler": 0.8,
    "firetube_boiler": 1.0,
    "packaged_boiler": 1.2,
    "industrial_furnace": 1.5,
    "process_heater": 1.8,
    "incinerator": 2.0,
}


# =============================================================================
# SIEGERT CONSTANTS
# =============================================================================

class SiegertConstants:
    """
    Siegert formula constants for different fuel types.

    The Siegert formula is an empirical correlation:
        L_dfg% = K * (T_fg - T_a) / CO2%

    Constants are derived from fuel composition and
    thermodynamic properties of combustion products.

    Reference: ASME PTC 4.1, EN 12952-15
    """

    # K factors for Siegert formula
    K_FACTORS = {
        # Gaseous fuels (lower K due to high H content)
        "natural_gas": 0.38,
        "propane": 0.40,
        "butane": 0.41,
        "hydrogen": 0.33,

        # Liquid fuels (medium K)
        "fuel_oil_no2": 0.44,
        "fuel_oil_no4": 0.46,
        "fuel_oil_no6": 0.48,
        "diesel": 0.45,
        "kerosene": 0.43,

        # Solid fuels (higher K)
        "bituminous_coal": 0.52,
        "subbituminous_coal": 0.56,
        "lignite": 0.60,
        "anthracite": 0.50,
        "wood": 0.54,
        "biomass": 0.55,
        "peat": 0.58,
    }

    # Alternative formulation with A1 and A2 factors
    # L_dfg = (A1 / CO2 + A2) * (T_fg - T_a)
    A_FACTORS = {
        "natural_gas": {"A1": 0.354, "A2": 0.00063},
        "fuel_oil_no2": {"A1": 0.406, "A2": 0.00082},
        "fuel_oil_no6": {"A1": 0.445, "A2": 0.00090},
        "bituminous_coal": {"A1": 0.495, "A2": 0.00100},
    }

    @classmethod
    def get_k_factor(cls, fuel_type: str) -> float:
        """Get Siegert K factor for fuel type."""
        key = fuel_type.lower().replace(" ", "_").replace("-", "_")

        if key in cls.K_FACTORS:
            return cls.K_FACTORS[key]

        # Default to coal value (conservative)
        return 0.52


# =============================================================================
# FUEL DATA
# =============================================================================

@dataclass(frozen=True)
class FuelProperties:
    """
    Fuel properties for efficiency calculations.

    Attributes:
        fuel_type: Fuel type identifier
        carbon_pct: Carbon content (mass %)
        hydrogen_pct: Hydrogen content (mass %)
        oxygen_pct: Oxygen content (mass %)
        nitrogen_pct: Nitrogen content (mass %)
        sulfur_pct: Sulfur content (mass %)
        moisture_pct: Moisture content (mass %)
        ash_pct: Ash content (mass %)
        hhv_kj_kg: Higher heating value (kJ/kg)
        lhv_kj_kg: Lower heating value (kJ/kg)
        theoretical_co2_max_pct: Maximum CO2 at stoichiometric (vol % dry)
    """
    fuel_type: str
    carbon_pct: float
    hydrogen_pct: float
    hhv_kj_kg: float
    lhv_kj_kg: float
    theoretical_co2_max_pct: float
    oxygen_pct: float = 0.0
    nitrogen_pct: float = 0.0
    sulfur_pct: float = 0.0
    moisture_pct: float = 0.0
    ash_pct: float = 0.0


# Standard fuel properties database
FUEL_PROPERTIES_DB = {
    "natural_gas": FuelProperties(
        fuel_type="natural_gas",
        carbon_pct=74.0,
        hydrogen_pct=24.0,
        oxygen_pct=0.5,
        nitrogen_pct=1.0,
        sulfur_pct=0.01,
        moisture_pct=0.0,
        ash_pct=0.0,
        hhv_kj_kg=55500.0,
        lhv_kj_kg=50000.0,
        theoretical_co2_max_pct=11.8
    ),
    "fuel_oil_no2": FuelProperties(
        fuel_type="fuel_oil_no2",
        carbon_pct=87.2,
        hydrogen_pct=12.5,
        oxygen_pct=0.1,
        nitrogen_pct=0.0,
        sulfur_pct=0.2,
        moisture_pct=0.0,
        ash_pct=0.0,
        hhv_kj_kg=45500.0,
        lhv_kj_kg=42700.0,
        theoretical_co2_max_pct=15.5
    ),
    "fuel_oil_no6": FuelProperties(
        fuel_type="fuel_oil_no6",
        carbon_pct=87.0,
        hydrogen_pct=10.5,
        oxygen_pct=0.5,
        nitrogen_pct=0.3,
        sulfur_pct=1.5,
        moisture_pct=0.0,
        ash_pct=0.1,
        hhv_kj_kg=43000.0,
        lhv_kj_kg=40500.0,
        theoretical_co2_max_pct=15.8
    ),
    "diesel": FuelProperties(
        fuel_type="diesel",
        carbon_pct=86.5,
        hydrogen_pct=13.2,
        oxygen_pct=0.0,
        nitrogen_pct=0.0,
        sulfur_pct=0.3,
        moisture_pct=0.0,
        ash_pct=0.0,
        hhv_kj_kg=45800.0,
        lhv_kj_kg=43100.0,
        theoretical_co2_max_pct=15.3
    ),
    "propane": FuelProperties(
        fuel_type="propane",
        carbon_pct=81.8,
        hydrogen_pct=18.2,
        oxygen_pct=0.0,
        nitrogen_pct=0.0,
        sulfur_pct=0.0,
        moisture_pct=0.0,
        ash_pct=0.0,
        hhv_kj_kg=50300.0,
        lhv_kj_kg=46400.0,
        theoretical_co2_max_pct=13.7
    ),
    "bituminous_coal": FuelProperties(
        fuel_type="bituminous_coal",
        carbon_pct=75.5,
        hydrogen_pct=5.0,
        oxygen_pct=6.5,
        nitrogen_pct=1.5,
        sulfur_pct=1.5,
        moisture_pct=3.5,
        ash_pct=6.5,
        hhv_kj_kg=31400.0,
        lhv_kj_kg=30200.0,
        theoretical_co2_max_pct=18.5
    ),
    "subbituminous_coal": FuelProperties(
        fuel_type="subbituminous_coal",
        carbon_pct=52.0,
        hydrogen_pct=3.5,
        oxygen_pct=11.0,
        nitrogen_pct=0.8,
        sulfur_pct=0.4,
        moisture_pct=25.0,
        ash_pct=7.3,
        hhv_kj_kg=21500.0,
        lhv_kj_kg=20300.0,
        theoretical_co2_max_pct=17.5
    ),
    "wood_biomass": FuelProperties(
        fuel_type="wood_biomass",
        carbon_pct=50.0,
        hydrogen_pct=6.0,
        oxygen_pct=42.0,
        nitrogen_pct=0.5,
        sulfur_pct=0.1,
        moisture_pct=20.0,
        ash_pct=1.4,
        hhv_kj_kg=18500.0,
        lhv_kj_kg=17000.0,
        theoretical_co2_max_pct=20.2
    ),
}


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class CombustionEfficiencyInput:
    """
    Input parameters for combustion efficiency calculations.

    Attributes:
        fuel_type: Fuel type identifier
        flue_gas_temp_c: Flue gas temperature (deg C)
        ambient_temp_c: Ambient air temperature (deg C)
        o2_pct_dry: Oxygen concentration (vol % dry)
        co2_pct_dry: CO2 concentration (vol % dry)
        co_ppm: CO concentration (ppm dry)
        excess_air_pct: Excess air percentage (%)
        fuel_moisture_pct: Additional fuel moisture if different from standard (%)
        unburned_carbon_pct: Unburned carbon in ash (% of ash)
        equipment_type: Equipment type for radiation loss estimation
        custom_fuel: Custom fuel properties (optional)
        include_radiation_loss: Include radiation/convection loss
        include_ash_loss: Include sensible heat in ash
        ash_temp_c: Ash temperature (deg C), default = flue gas temp
    """
    fuel_type: str
    flue_gas_temp_c: float
    ambient_temp_c: float
    o2_pct_dry: float
    co2_pct_dry: float
    excess_air_pct: float
    co_ppm: float = 0.0
    fuel_moisture_pct: Optional[float] = None
    unburned_carbon_pct: float = 0.0
    equipment_type: str = "watertube_boiler"
    custom_fuel: Optional[FuelProperties] = None
    include_radiation_loss: bool = True
    include_ash_loss: bool = True
    ash_temp_c: Optional[float] = None


@dataclass(frozen=True)
class HeatLossBreakdown:
    """
    Breakdown of heat losses in combustion system.

    All values are in percentage of fuel heat input (HHV basis).

    Attributes:
        dry_flue_gas_loss_pct: Loss from sensible heat in dry flue gas
        moisture_from_hydrogen_loss_pct: Loss from H2O from hydrogen combustion
        moisture_in_fuel_loss_pct: Loss from evaporating fuel moisture
        moisture_in_air_loss_pct: Loss from humidity in combustion air
        co_loss_pct: Loss from incomplete combustion (CO)
        unburned_carbon_loss_pct: Loss from unburned carbon in ash
        radiation_convection_loss_pct: Surface radiation and convection
        ash_sensible_heat_loss_pct: Sensible heat in ash/refuse
        unaccounted_loss_pct: Unaccounted losses
        total_losses_pct: Sum of all losses
    """
    dry_flue_gas_loss_pct: float
    moisture_from_hydrogen_loss_pct: float
    moisture_in_fuel_loss_pct: float
    moisture_in_air_loss_pct: float
    co_loss_pct: float
    unburned_carbon_loss_pct: float
    radiation_convection_loss_pct: float
    ash_sensible_heat_loss_pct: float
    unaccounted_loss_pct: float
    total_losses_pct: float


@dataclass(frozen=True)
class CombustionEfficiencyOutput:
    """
    Complete combustion efficiency analysis results.

    Attributes:
        combustion_efficiency_hhv_pct: Efficiency on HHV basis (%)
        combustion_efficiency_lhv_pct: Efficiency on LHV basis (%)
        thermal_efficiency_pct: Useful heat output efficiency (%)

        dry_flue_gas_loss_pct: Dry flue gas loss (%)
        total_moisture_loss_pct: Total moisture related losses (%)
        co_loss_pct: CO incomplete combustion loss (%)
        unburned_carbon_loss_pct: Unburned carbon loss (%)
        radiation_loss_pct: Radiation/convection loss (%)
        ash_loss_pct: Ash sensible heat loss (%)
        total_losses_pct: Sum of all losses (%)

        heat_loss_breakdown: Detailed breakdown of all losses
        available_heat_pct: Heat available for useful work (%)
        stack_loss_pct: Total stack loss (dry + moisture)

        efficiency_rating: Performance rating
        efficiency_improvement_potential_pct: Potential improvement (%)
        recommended_flue_gas_temp_c: Optimal flue gas temperature (deg C)
        recommended_o2_pct: Optimal O2 setpoint (%)

        siegert_k_factor: Siegert K factor used
        fuel_hhv_kj_kg: Fuel HHV used (kJ/kg)
        fuel_lhv_kj_kg: Fuel LHV used (kJ/kg)
    """
    # Efficiency values
    combustion_efficiency_hhv_pct: float
    combustion_efficiency_lhv_pct: float
    thermal_efficiency_pct: float

    # Individual losses
    dry_flue_gas_loss_pct: float
    total_moisture_loss_pct: float
    co_loss_pct: float
    unburned_carbon_loss_pct: float
    radiation_loss_pct: float
    ash_loss_pct: float
    total_losses_pct: float

    # Detailed breakdown
    heat_loss_breakdown: HeatLossBreakdown

    # Derived values
    available_heat_pct: float
    stack_loss_pct: float

    # Performance assessment
    efficiency_rating: str
    efficiency_improvement_potential_pct: float
    recommended_flue_gas_temp_c: float
    recommended_o2_pct: float

    # Calculation parameters
    siegert_k_factor: float
    fuel_hhv_kj_kg: float
    fuel_lhv_kj_kg: float


# =============================================================================
# COMBUSTION EFFICIENCY CALCULATOR
# =============================================================================

class CombustionEfficiencyCalculator:
    """
    Zero-hallucination calculator for combustion efficiency analysis.

    Implements ASME PTC 4.1 heat loss method for comprehensive
    combustion efficiency calculations with complete provenance.

    Calculation Methods:
    1. Siegert Formula (quick estimation)
    2. Heat Loss Method (detailed, ASME PTC 4.1)

    All calculations are:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = CombustionEfficiencyCalculator()
        >>> inputs = CombustionEfficiencyInput(
        ...     fuel_type='natural_gas',
        ...     flue_gas_temp_c=180.0,
        ...     ambient_temp_c=25.0,
        ...     o2_pct_dry=3.5,
        ...     co2_pct_dry=11.0,
        ...     excess_air_pct=20.0,
        ...     co_ppm=50.0
        ... )
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Efficiency: {result.combustion_efficiency_hhv_pct:.1f}%")
    """

    VERSION = "1.0.0"
    NAME = "CombustionEfficiencyCalculator"

    def __init__(self):
        """Initialize the combustion efficiency calculator."""
        self._tracker: Optional[ProvenanceTracker] = None
        self._step_counter: int = 0

    def calculate(
        self,
        inputs: CombustionEfficiencyInput
    ) -> Tuple[CombustionEfficiencyOutput, ProvenanceRecord]:
        """
        Perform complete combustion efficiency analysis.

        Args:
            inputs: CombustionEfficiencyInput with operating data

        Returns:
            Tuple of (CombustionEfficiencyOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid or out of range
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["ASME PTC 4.1", "ASME PTC 4", "EN 12952-15"],
                "domain": "Combustion Efficiency Analysis",
                "method": "Heat Loss Method"
            }
        )
        self._step_counter = 0

        # Set inputs for provenance
        input_dict = {
            "fuel_type": inputs.fuel_type,
            "flue_gas_temp_c": inputs.flue_gas_temp_c,
            "ambient_temp_c": inputs.ambient_temp_c,
            "o2_pct_dry": inputs.o2_pct_dry,
            "co2_pct_dry": inputs.co2_pct_dry,
            "co_ppm": inputs.co_ppm,
            "excess_air_pct": inputs.excess_air_pct,
            "fuel_moisture_pct": inputs.fuel_moisture_pct,
            "unburned_carbon_pct": inputs.unburned_carbon_pct,
            "equipment_type": inputs.equipment_type,
            "include_radiation_loss": inputs.include_radiation_loss,
            "include_ash_loss": inputs.include_ash_loss
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Get fuel properties
        fuel = self._get_fuel_properties(inputs)

        # Step 1: Calculate dry flue gas loss (Siegert formula)
        k_factor = SiegertConstants.get_k_factor(inputs.fuel_type)
        dfg_loss = self._calculate_dry_flue_gas_loss(
            inputs.flue_gas_temp_c,
            inputs.ambient_temp_c,
            inputs.co2_pct_dry,
            k_factor
        )

        # Step 2: Calculate moisture loss from hydrogen combustion
        moisture_h2_loss = self._calculate_moisture_from_hydrogen_loss(
            fuel.hydrogen_pct,
            inputs.flue_gas_temp_c,
            inputs.ambient_temp_c,
            fuel.hhv_kj_kg
        )

        # Step 3: Calculate moisture loss from fuel moisture
        fuel_moisture = inputs.fuel_moisture_pct if inputs.fuel_moisture_pct is not None else fuel.moisture_pct
        moisture_fuel_loss = self._calculate_moisture_in_fuel_loss(
            fuel_moisture,
            inputs.flue_gas_temp_c,
            inputs.ambient_temp_c,
            fuel.hhv_kj_kg
        )

        # Step 4: Calculate moisture loss from air humidity
        moisture_air_loss = self._calculate_moisture_in_air_loss(
            inputs.excess_air_pct,
            inputs.flue_gas_temp_c,
            inputs.ambient_temp_c,
            fuel.hhv_kj_kg
        )

        # Step 5: Calculate CO loss (incomplete combustion)
        co_loss = self._calculate_co_loss(
            inputs.co_ppm,
            inputs.co2_pct_dry,
            fuel.carbon_pct,
            fuel.hhv_kj_kg
        )

        # Step 6: Calculate unburned carbon loss
        uc_loss = self._calculate_unburned_carbon_loss(
            fuel.ash_pct,
            inputs.unburned_carbon_pct,
            fuel.hhv_kj_kg
        )

        # Step 7: Calculate radiation loss
        if inputs.include_radiation_loss:
            rad_loss = self._calculate_radiation_loss(inputs.equipment_type)
        else:
            rad_loss = 0.0

        # Step 8: Calculate ash sensible heat loss
        if inputs.include_ash_loss and fuel.ash_pct > 0:
            ash_temp = inputs.ash_temp_c if inputs.ash_temp_c is not None else inputs.flue_gas_temp_c
            ash_loss = self._calculate_ash_sensible_heat_loss(
                fuel.ash_pct,
                inputs.unburned_carbon_pct,
                ash_temp,
                inputs.ambient_temp_c,
                fuel.hhv_kj_kg
            )
        else:
            ash_loss = 0.0

        # Step 9: Calculate unaccounted loss
        unaccounted_loss = 0.5  # Typical default value

        # Step 10: Sum all losses
        total_moisture_loss = moisture_h2_loss + moisture_fuel_loss + moisture_air_loss
        total_losses = (dfg_loss + total_moisture_loss + co_loss +
                       uc_loss + rad_loss + ash_loss + unaccounted_loss)

        # Step 11: Calculate efficiency (HHV basis)
        efficiency_hhv = 100.0 - total_losses

        # Step 12: Calculate efficiency (LHV basis)
        # LHV efficiency is higher because it doesn't include latent heat
        hhv_to_lhv_ratio = fuel.hhv_kj_kg / fuel.lhv_kj_kg
        efficiency_lhv = efficiency_hhv * hhv_to_lhv_ratio

        # Cap LHV efficiency at reasonable maximum
        efficiency_lhv = min(efficiency_lhv, 99.5)

        # Step 13: Calculate stack loss and available heat
        stack_loss = dfg_loss + total_moisture_loss
        available_heat = 100.0 - (stack_loss + rad_loss)

        # Step 14: Determine efficiency rating
        efficiency_rating = self._determine_efficiency_rating(efficiency_hhv, inputs.fuel_type)

        # Step 15: Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(
            efficiency_hhv,
            inputs.flue_gas_temp_c,
            inputs.o2_pct_dry,
            inputs.fuel_type
        )

        # Step 16: Calculate recommendations
        recommended_fg_temp = self._calculate_optimal_flue_gas_temp(inputs.fuel_type)
        recommended_o2 = self._calculate_optimal_o2(inputs.fuel_type)

        # Create heat loss breakdown
        breakdown = HeatLossBreakdown(
            dry_flue_gas_loss_pct=round(dfg_loss, 3),
            moisture_from_hydrogen_loss_pct=round(moisture_h2_loss, 3),
            moisture_in_fuel_loss_pct=round(moisture_fuel_loss, 3),
            moisture_in_air_loss_pct=round(moisture_air_loss, 3),
            co_loss_pct=round(co_loss, 3),
            unburned_carbon_loss_pct=round(uc_loss, 3),
            radiation_convection_loss_pct=round(rad_loss, 3),
            ash_sensible_heat_loss_pct=round(ash_loss, 3),
            unaccounted_loss_pct=round(unaccounted_loss, 3),
            total_losses_pct=round(total_losses, 3)
        )

        # Create output
        output = CombustionEfficiencyOutput(
            combustion_efficiency_hhv_pct=round(efficiency_hhv, 2),
            combustion_efficiency_lhv_pct=round(efficiency_lhv, 2),
            thermal_efficiency_pct=round(efficiency_hhv, 2),  # Same as combustion for this calculation
            dry_flue_gas_loss_pct=round(dfg_loss, 2),
            total_moisture_loss_pct=round(total_moisture_loss, 2),
            co_loss_pct=round(co_loss, 2),
            unburned_carbon_loss_pct=round(uc_loss, 2),
            radiation_loss_pct=round(rad_loss, 2),
            ash_loss_pct=round(ash_loss, 2),
            total_losses_pct=round(total_losses, 2),
            heat_loss_breakdown=breakdown,
            available_heat_pct=round(available_heat, 2),
            stack_loss_pct=round(stack_loss, 2),
            efficiency_rating=efficiency_rating,
            efficiency_improvement_potential_pct=round(improvement_potential, 2),
            recommended_flue_gas_temp_c=round(recommended_fg_temp, 0),
            recommended_o2_pct=round(recommended_o2, 1),
            siegert_k_factor=k_factor,
            fuel_hhv_kj_kg=fuel.hhv_kj_kg,
            fuel_lhv_kj_kg=fuel.lhv_kj_kg
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "combustion_efficiency_hhv_pct": output.combustion_efficiency_hhv_pct,
            "combustion_efficiency_lhv_pct": output.combustion_efficiency_lhv_pct,
            "dry_flue_gas_loss_pct": output.dry_flue_gas_loss_pct,
            "total_moisture_loss_pct": output.total_moisture_loss_pct,
            "co_loss_pct": output.co_loss_pct,
            "unburned_carbon_loss_pct": output.unburned_carbon_loss_pct,
            "radiation_loss_pct": output.radiation_loss_pct,
            "total_losses_pct": output.total_losses_pct,
            "stack_loss_pct": output.stack_loss_pct,
            "efficiency_rating": output.efficiency_rating,
            "efficiency_improvement_potential_pct": output.efficiency_improvement_potential_pct
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _next_step(self) -> int:
        """Get next step number for provenance tracking."""
        self._step_counter += 1
        return self._step_counter

    def _validate_inputs(self, inputs: CombustionEfficiencyInput) -> None:
        """Validate input parameters."""
        if inputs.flue_gas_temp_c < 50 or inputs.flue_gas_temp_c > 1200:
            raise ValueError(
                f"Flue gas temperature {inputs.flue_gas_temp_c} deg C out of range (50-1200 deg C)"
            )

        if inputs.ambient_temp_c < -20 or inputs.ambient_temp_c > 50:
            raise ValueError(
                f"Ambient temperature {inputs.ambient_temp_c} deg C out of range (-20 to 50 deg C)"
            )

        if inputs.o2_pct_dry < 0 or inputs.o2_pct_dry > 21:
            raise ValueError(f"O2 concentration {inputs.o2_pct_dry}% out of range (0-21%)")

        if inputs.co2_pct_dry <= 0 or inputs.co2_pct_dry > 25:
            raise ValueError(f"CO2 concentration {inputs.co2_pct_dry}% out of range (0-25%)")

        if inputs.co_ppm < 0:
            raise ValueError("CO concentration cannot be negative")

        if inputs.excess_air_pct < 0:
            raise ValueError("Excess air cannot be negative")

    def _get_fuel_properties(self, inputs: CombustionEfficiencyInput) -> FuelProperties:
        """Get fuel properties from input or database."""
        if inputs.custom_fuel is not None:
            return inputs.custom_fuel

        fuel_key = inputs.fuel_type.lower().replace(" ", "_").replace("-", "_")
        if fuel_key not in FUEL_PROPERTIES_DB:
            raise ValueError(f"Unknown fuel type: {inputs.fuel_type}")

        return FUEL_PROPERTIES_DB[fuel_key]

    def _calculate_dry_flue_gas_loss(
        self,
        flue_gas_temp_c: float,
        ambient_temp_c: float,
        co2_pct: float,
        k_factor: float
    ) -> float:
        """
        Calculate dry flue gas loss using Siegert formula.

        Formula (ASME PTC 4.1):
            L_dfg% = K * (T_fg - T_a) / CO2%

        Where K is fuel-specific constant accounting for:
        - Flue gas specific heat
        - Flue gas mass per unit fuel
        - Fuel heating value

        Args:
            flue_gas_temp_c: Flue gas temperature (deg C)
            ambient_temp_c: Ambient temperature (deg C)
            co2_pct: CO2 concentration (vol % dry)
            k_factor: Siegert K factor

        Returns:
            Dry flue gas loss as percentage of HHV (%)
        """
        step = self._next_step()

        # Temperature difference
        delta_t = flue_gas_temp_c - ambient_temp_c

        # Siegert formula
        # Protect against division by zero
        co2_safe = max(co2_pct, 1.0)
        dfg_loss = k_factor * delta_t / co2_safe

        self._tracker.add_step(
            step_number=step,
            description="Calculate dry flue gas loss (Siegert formula)",
            operation="siegert_formula",
            inputs={
                "flue_gas_temp_c": flue_gas_temp_c,
                "ambient_temp_c": ambient_temp_c,
                "delta_t_c": delta_t,
                "co2_pct": co2_pct,
                "k_factor": k_factor
            },
            output_value=dfg_loss,
            output_name="dry_flue_gas_loss_pct",
            formula="L_dfg% = K * (T_fg - T_a) / CO2%"
        )

        return dfg_loss

    def _calculate_moisture_from_hydrogen_loss(
        self,
        hydrogen_pct: float,
        flue_gas_temp_c: float,
        ambient_temp_c: float,
        hhv_kj_kg: float
    ) -> float:
        """
        Calculate loss from moisture formed by hydrogen combustion.

        Formula (ASME PTC 4.1):
            L_mH2O% = (9 * H%/100) * (hfg + Cp_steam * (T_fg - T_ref)) / HHV * 100

        Where:
        - 9 kg H2O per kg H2 (stoichiometric)
        - hfg = latent heat of vaporization at reference temp
        - Cp_steam = specific heat of steam

        Args:
            hydrogen_pct: Hydrogen content in fuel (mass %)
            flue_gas_temp_c: Flue gas temperature (deg C)
            ambient_temp_c: Ambient/reference temperature (deg C)
            hhv_kj_kg: Fuel higher heating value (kJ/kg)

        Returns:
            Moisture loss from hydrogen as percentage of HHV (%)
        """
        step = self._next_step()

        # Mass of water formed per kg fuel
        m_h2o = 9.0 * (hydrogen_pct / 100.0)

        # Enthalpy to be removed from water vapor
        # hfg at 25 deg C + sensible heat to flue gas temp
        delta_t = flue_gas_temp_c - REFERENCE_TEMP_C
        h_h2o = LATENT_HEAT_H2O_25C + CP_WATER_VAPOR * delta_t

        # Loss percentage
        loss = (m_h2o * h_h2o) / hhv_kj_kg * 100.0

        self._tracker.add_step(
            step_number=step,
            description="Calculate moisture loss from hydrogen combustion",
            operation="moisture_h2_loss",
            inputs={
                "hydrogen_pct": hydrogen_pct,
                "mass_h2o_per_kg_fuel": m_h2o,
                "flue_gas_temp_c": flue_gas_temp_c,
                "reference_temp_c": REFERENCE_TEMP_C,
                "latent_heat_kj_kg": LATENT_HEAT_H2O_25C,
                "cp_steam_kj_kg_k": CP_WATER_VAPOR,
                "enthalpy_h2o_kj_kg": h_h2o,
                "hhv_kj_kg": hhv_kj_kg
            },
            output_value=loss,
            output_name="moisture_from_hydrogen_loss_pct",
            formula="L = (9 * H/100) * (hfg + Cp * dT) / HHV * 100"
        )

        return loss

    def _calculate_moisture_in_fuel_loss(
        self,
        moisture_pct: float,
        flue_gas_temp_c: float,
        ambient_temp_c: float,
        hhv_kj_kg: float
    ) -> float:
        """
        Calculate loss from evaporating fuel moisture.

        Formula (ASME PTC 4.1):
            L_mf% = (M%/100) * (hfg + Cp_steam * (T_fg - T_ref)) / HHV * 100

        Args:
            moisture_pct: Fuel moisture content (mass %)
            flue_gas_temp_c: Flue gas temperature (deg C)
            ambient_temp_c: Reference temperature (deg C)
            hhv_kj_kg: Fuel higher heating value (kJ/kg)

        Returns:
            Moisture in fuel loss as percentage of HHV (%)
        """
        step = self._next_step()

        if moisture_pct <= 0:
            self._tracker.add_step(
                step_number=step,
                description="Calculate moisture in fuel loss (zero moisture)",
                operation="moisture_fuel_loss",
                inputs={"moisture_pct": moisture_pct},
                output_value=0.0,
                output_name="moisture_in_fuel_loss_pct",
                formula="No fuel moisture"
            )
            return 0.0

        # Mass of moisture per kg fuel
        m_h2o = moisture_pct / 100.0

        # Enthalpy to be removed
        delta_t = flue_gas_temp_c - REFERENCE_TEMP_C
        h_h2o = LATENT_HEAT_H2O_25C + CP_WATER_VAPOR * delta_t

        # Loss percentage
        loss = (m_h2o * h_h2o) / hhv_kj_kg * 100.0

        self._tracker.add_step(
            step_number=step,
            description="Calculate moisture in fuel loss",
            operation="moisture_fuel_loss",
            inputs={
                "moisture_pct": moisture_pct,
                "mass_h2o_per_kg_fuel": m_h2o,
                "flue_gas_temp_c": flue_gas_temp_c,
                "enthalpy_h2o_kj_kg": h_h2o,
                "hhv_kj_kg": hhv_kj_kg
            },
            output_value=loss,
            output_name="moisture_in_fuel_loss_pct",
            formula="L = (M/100) * (hfg + Cp * dT) / HHV * 100"
        )

        return loss

    def _calculate_moisture_in_air_loss(
        self,
        excess_air_pct: float,
        flue_gas_temp_c: float,
        ambient_temp_c: float,
        hhv_kj_kg: float
    ) -> float:
        """
        Calculate loss from moisture in combustion air (humidity).

        This is typically small (<0.5%) but included for completeness.

        Assumes standard humidity of ~0.01 kg H2O/kg dry air.

        Args:
            excess_air_pct: Excess air percentage (%)
            flue_gas_temp_c: Flue gas temperature (deg C)
            ambient_temp_c: Ambient temperature (deg C)
            hhv_kj_kg: Fuel higher heating value (kJ/kg)

        Returns:
            Moisture in air loss as percentage of HHV (%)
        """
        step = self._next_step()

        # Typical humidity ratio (kg H2O/kg dry air) at 50% RH, 25 deg C
        humidity_ratio = 0.010

        # Estimate air per kg fuel (rough: ~15 kg air/kg fuel for most fuels)
        # Adjusted for excess air
        air_per_kg_fuel = 15.0 * (1.0 + excess_air_pct / 100.0)

        # Mass of moisture from air
        m_h2o_air = air_per_kg_fuel * humidity_ratio

        # Heat carried by this moisture
        delta_t = flue_gas_temp_c - ambient_temp_c
        h_h2o = CP_WATER_VAPOR * delta_t

        # Loss percentage
        loss = (m_h2o_air * h_h2o) / hhv_kj_kg * 100.0

        self._tracker.add_step(
            step_number=step,
            description="Calculate moisture in combustion air loss",
            operation="moisture_air_loss",
            inputs={
                "excess_air_pct": excess_air_pct,
                "humidity_ratio_kg_kg": humidity_ratio,
                "air_per_kg_fuel": air_per_kg_fuel,
                "mass_h2o_from_air": m_h2o_air,
                "flue_gas_temp_c": flue_gas_temp_c
            },
            output_value=loss,
            output_name="moisture_in_air_loss_pct",
            formula="L = m_air * humidity * Cp * dT / HHV * 100"
        )

        return loss

    def _calculate_co_loss(
        self,
        co_ppm: float,
        co2_pct: float,
        carbon_pct: float,
        hhv_kj_kg: float
    ) -> float:
        """
        Calculate loss from incomplete combustion (CO formation).

        Formula (ASME PTC 4.1):
            L_CO% = (CO / (CO + CO2)) * (5654 * C%) / HHV * 100

        Where 5654 kJ/kg C is the heating value difference between
        complete combustion (CO2) and partial combustion (CO).

        Alternatively:
            L_CO% = (CO / (CO + CO2)) * 23656 * C% / HHV

        Args:
            co_ppm: CO concentration (ppm dry)
            co2_pct: CO2 concentration (vol % dry)
            carbon_pct: Carbon content in fuel (mass %)
            hhv_kj_kg: Fuel higher heating value (kJ/kg)

        Returns:
            CO loss as percentage of HHV (%)
        """
        step = self._next_step()

        if co_ppm <= 0:
            self._tracker.add_step(
                step_number=step,
                description="Calculate CO loss (zero CO)",
                operation="co_loss",
                inputs={"co_ppm": co_ppm},
                output_value=0.0,
                output_name="co_loss_pct",
                formula="No CO detected"
            )
            return 0.0

        # Convert CO ppm to volume percent
        co_pct = co_ppm / 10000.0

        # Ratio of carbon burned to CO vs total carbon
        co_co2_ratio = co_pct / (co_pct + co2_pct)

        # Heat lost by forming CO instead of CO2
        # Difference: 32790 - 10110 = 22680 kJ/kg C (using exact values)
        # Common approximation: 5654 kJ/kg C when using specific formula
        heat_diff_kj_per_kg_c = 5654.0

        # Loss calculation
        loss = co_co2_ratio * heat_diff_kj_per_kg_c * (carbon_pct / 100.0) / hhv_kj_kg * 100.0

        self._tracker.add_step(
            step_number=step,
            description="Calculate CO incomplete combustion loss",
            operation="co_loss",
            inputs={
                "co_ppm": co_ppm,
                "co_pct": co_pct,
                "co2_pct": co2_pct,
                "co_co2_ratio": co_co2_ratio,
                "carbon_pct": carbon_pct,
                "heat_diff_kj_kg_c": heat_diff_kj_per_kg_c,
                "hhv_kj_kg": hhv_kj_kg
            },
            output_value=loss,
            output_name="co_loss_pct",
            formula="L = (CO/(CO+CO2)) * 5654 * C% / HHV * 100"
        )

        return loss

    def _calculate_unburned_carbon_loss(
        self,
        ash_pct: float,
        unburned_carbon_pct: float,
        hhv_kj_kg: float
    ) -> float:
        """
        Calculate loss from unburned carbon in ash/refuse.

        Formula (ASME PTC 4.1):
            L_UC% = 33820 * (ash% * UBC%) / 100 / HHV * 100

        Where:
        - 33820 kJ/kg is the heating value of carbon
        - UBC% is percentage of carbon in ash

        This is primarily relevant for solid fuels.

        Args:
            ash_pct: Ash content in fuel (mass %)
            unburned_carbon_pct: Unburned carbon in ash (% of ash mass)
            hhv_kj_kg: Fuel higher heating value (kJ/kg)

        Returns:
            Unburned carbon loss as percentage of HHV (%)
        """
        step = self._next_step()

        if ash_pct <= 0 or unburned_carbon_pct <= 0:
            self._tracker.add_step(
                step_number=step,
                description="Calculate unburned carbon loss (no ash/UBC)",
                operation="unburned_carbon_loss",
                inputs={
                    "ash_pct": ash_pct,
                    "unburned_carbon_pct": unburned_carbon_pct
                },
                output_value=0.0,
                output_name="unburned_carbon_loss_pct",
                formula="No ash or unburned carbon"
            )
            return 0.0

        # Mass of unburned carbon per kg fuel
        m_ubc = (ash_pct / 100.0) * (unburned_carbon_pct / 100.0)

        # Heat loss from unburned carbon
        loss = (HEATING_VALUE_UNBURNED_CARBON * m_ubc) / hhv_kj_kg * 100.0

        self._tracker.add_step(
            step_number=step,
            description="Calculate unburned carbon loss",
            operation="unburned_carbon_loss",
            inputs={
                "ash_pct": ash_pct,
                "unburned_carbon_pct": unburned_carbon_pct,
                "mass_ubc_kg_per_kg_fuel": m_ubc,
                "heating_value_c_kj_kg": HEATING_VALUE_UNBURNED_CARBON,
                "hhv_kj_kg": hhv_kj_kg
            },
            output_value=loss,
            output_name="unburned_carbon_loss_pct",
            formula="L = 33820 * (ash% * UBC% / 10000) / HHV * 100"
        )

        return loss

    def _calculate_radiation_loss(self, equipment_type: str) -> float:
        """
        Calculate radiation and convection loss from equipment surface.

        Uses empirical correlations based on equipment type.
        For detailed analysis, surface temperature measurements
        would be required.

        Args:
            equipment_type: Type of combustion equipment

        Returns:
            Radiation/convection loss as percentage of HHV (%)
        """
        step = self._next_step()

        # Get typical radiation loss for equipment type
        key = equipment_type.lower().replace(" ", "_").replace("-", "_")
        rad_loss = RADIATION_LOSS_TYPICAL.get(key, 1.0)

        self._tracker.add_step(
            step_number=step,
            description="Estimate radiation and convection loss",
            operation="radiation_loss",
            inputs={
                "equipment_type": equipment_type,
                "typical_loss_table": RADIATION_LOSS_TYPICAL
            },
            output_value=rad_loss,
            output_name="radiation_convection_loss_pct",
            formula="Empirical correlation based on equipment type"
        )

        return rad_loss

    def _calculate_ash_sensible_heat_loss(
        self,
        ash_pct: float,
        unburned_carbon_pct: float,
        ash_temp_c: float,
        ambient_temp_c: float,
        hhv_kj_kg: float
    ) -> float:
        """
        Calculate sensible heat loss in ash/refuse.

        Formula:
            L_ash% = m_ash * Cp_ash * (T_ash - T_ref) / HHV * 100

        Args:
            ash_pct: Ash content in fuel (mass %)
            unburned_carbon_pct: Unburned carbon in ash (%)
            ash_temp_c: Ash discharge temperature (deg C)
            ambient_temp_c: Reference temperature (deg C)
            hhv_kj_kg: Fuel higher heating value (kJ/kg)

        Returns:
            Ash sensible heat loss as percentage of HHV (%)
        """
        step = self._next_step()

        if ash_pct <= 0:
            self._tracker.add_step(
                step_number=step,
                description="Calculate ash sensible heat loss (no ash)",
                operation="ash_sensible_heat",
                inputs={"ash_pct": ash_pct},
                output_value=0.0,
                output_name="ash_sensible_heat_loss_pct",
                formula="No ash"
            )
            return 0.0

        # Mass of ash + unburned carbon per kg fuel
        m_ash = (ash_pct / 100.0) * (1.0 + unburned_carbon_pct / 100.0)

        # Sensible heat in ash
        delta_t = ash_temp_c - ambient_temp_c
        h_ash = CP_ASH * delta_t

        # Loss percentage
        loss = (m_ash * h_ash) / hhv_kj_kg * 100.0

        self._tracker.add_step(
            step_number=step,
            description="Calculate sensible heat in ash",
            operation="ash_sensible_heat",
            inputs={
                "ash_pct": ash_pct,
                "unburned_carbon_pct": unburned_carbon_pct,
                "mass_ash_kg_per_kg_fuel": m_ash,
                "ash_temp_c": ash_temp_c,
                "ambient_temp_c": ambient_temp_c,
                "cp_ash_kj_kg_k": CP_ASH,
                "hhv_kj_kg": hhv_kj_kg
            },
            output_value=loss,
            output_name="ash_sensible_heat_loss_pct",
            formula="L = m_ash * Cp * (T_ash - T_ref) / HHV * 100"
        )

        return loss

    def _determine_efficiency_rating(
        self,
        efficiency: float,
        fuel_type: str
    ) -> str:
        """
        Determine efficiency performance rating.

        Ratings vary by fuel type due to different theoretical maximums.

        Args:
            efficiency: Combustion efficiency (% HHV)
            fuel_type: Fuel type

        Returns:
            Performance rating string
        """
        step = self._next_step()

        # Define thresholds by fuel category
        fuel_lower = fuel_type.lower()

        if "gas" in fuel_lower or "propane" in fuel_lower:
            # Gas-fired: higher efficiency expected
            thresholds = {"excellent": 88, "good": 84, "fair": 80, "poor": 75}
        elif "oil" in fuel_lower or "diesel" in fuel_lower:
            # Oil-fired: medium efficiency expected
            thresholds = {"excellent": 86, "good": 82, "fair": 78, "poor": 72}
        else:
            # Solid fuels: lower efficiency expected
            thresholds = {"excellent": 82, "good": 78, "fair": 74, "poor": 68}

        if efficiency >= thresholds["excellent"]:
            rating = "Excellent"
        elif efficiency >= thresholds["good"]:
            rating = "Good"
        elif efficiency >= thresholds["fair"]:
            rating = "Fair"
        elif efficiency >= thresholds["poor"]:
            rating = "Poor"
        else:
            rating = "Critical"

        self._tracker.add_step(
            step_number=step,
            description="Determine efficiency rating",
            operation="threshold_classification",
            inputs={
                "efficiency_pct": efficiency,
                "fuel_type": fuel_type,
                "thresholds": thresholds
            },
            output_value=rating,
            output_name="efficiency_rating",
            formula="Rating based on fuel-specific thresholds"
        )

        return rating

    def _calculate_improvement_potential(
        self,
        current_efficiency: float,
        flue_gas_temp: float,
        o2_pct: float,
        fuel_type: str
    ) -> float:
        """
        Calculate potential efficiency improvement.

        Based on comparison to typical best-in-class performance.

        Args:
            current_efficiency: Current efficiency (%)
            flue_gas_temp: Current flue gas temp (deg C)
            o2_pct: Current O2 (%)
            fuel_type: Fuel type

        Returns:
            Potential improvement (percentage points)
        """
        step = self._next_step()

        # Best-in-class efficiency targets by fuel type
        fuel_lower = fuel_type.lower()

        if "gas" in fuel_lower:
            target_efficiency = 91.0
            optimal_fg_temp = 150.0
            optimal_o2 = 2.5
        elif "oil" in fuel_lower or "diesel" in fuel_lower:
            target_efficiency = 88.0
            optimal_fg_temp = 180.0
            optimal_o2 = 3.0
        else:
            target_efficiency = 84.0
            optimal_fg_temp = 200.0
            optimal_o2 = 4.0

        # Calculate improvement potential
        improvement = target_efficiency - current_efficiency
        improvement = max(0.0, improvement)  # Can't have negative improvement

        # Additional potential from operating parameters
        if flue_gas_temp > optimal_fg_temp:
            # ~1% per 20 deg C above optimal
            temp_improvement = (flue_gas_temp - optimal_fg_temp) / 20.0
            improvement = min(improvement + temp_improvement, 15.0)

        if o2_pct > optimal_o2 + 1.0:
            # ~0.5% per % excess O2 above optimal
            o2_improvement = (o2_pct - optimal_o2 - 1.0) * 0.5
            improvement = min(improvement + o2_improvement, 15.0)

        self._tracker.add_step(
            step_number=step,
            description="Calculate efficiency improvement potential",
            operation="improvement_potential",
            inputs={
                "current_efficiency_pct": current_efficiency,
                "target_efficiency_pct": target_efficiency,
                "flue_gas_temp_c": flue_gas_temp,
                "optimal_fg_temp_c": optimal_fg_temp,
                "o2_pct": o2_pct,
                "optimal_o2_pct": optimal_o2
            },
            output_value=improvement,
            output_name="improvement_potential_pct",
            formula="Improvement = Target - Current + Operational factors"
        )

        return improvement

    def _calculate_optimal_flue_gas_temp(self, fuel_type: str) -> float:
        """Calculate recommended flue gas temperature for fuel type."""
        fuel_lower = fuel_type.lower()

        if "gas" in fuel_lower:
            # Gas: Can go lower, but watch for condensation
            return 150.0
        elif "oil" in fuel_lower or "diesel" in fuel_lower:
            # Oil: Watch for sulfur acid dew point
            return 175.0
        elif "coal" in fuel_lower:
            # Coal: Higher to avoid acid dew point
            return 200.0
        else:
            # Default
            return 180.0

    def _calculate_optimal_o2(self, fuel_type: str) -> float:
        """Calculate recommended O2 setpoint for fuel type."""
        fuel_lower = fuel_type.lower()

        if "gas" in fuel_lower:
            # Gas: Can run leaner
            return 2.5
        elif "oil" in fuel_lower:
            # Oil: Moderate excess air
            return 3.0
        elif "coal" in fuel_lower:
            # Coal: More excess air for complete combustion
            return 4.5
        else:
            return 3.0


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_stack_loss_siegert(
    flue_gas_temp_c: float,
    ambient_temp_c: float,
    co2_pct: float,
    fuel_type: str = "natural_gas"
) -> float:
    """
    Calculate stack loss using Siegert formula (standalone function).

    Formula:
        Stack_Loss% = K * (T_fg - T_a) / CO2%

    Args:
        flue_gas_temp_c: Flue gas temperature (deg C)
        ambient_temp_c: Ambient temperature (deg C)
        co2_pct: CO2 concentration (vol % dry)
        fuel_type: Fuel type for K factor selection

    Returns:
        Stack loss as percentage of HHV (%)

    Example:
        >>> loss = calculate_stack_loss_siegert(180, 25, 11.5, "natural_gas")
        >>> print(f"Stack Loss: {loss:.1f}%")  # ~5.1%
    """
    if co2_pct <= 0:
        raise ValueError("CO2 concentration must be positive")

    k = SiegertConstants.get_k_factor(fuel_type)
    return k * (flue_gas_temp_c - ambient_temp_c) / co2_pct


def calculate_efficiency_from_losses(
    dry_flue_gas_loss: float,
    moisture_loss: float,
    co_loss: float = 0.0,
    radiation_loss: float = 1.0,
    unaccounted_loss: float = 0.5
) -> float:
    """
    Calculate combustion efficiency from individual losses.

    Formula:
        Efficiency% = 100 - Sum of all losses

    Args:
        dry_flue_gas_loss: Dry flue gas loss (%)
        moisture_loss: Total moisture losses (%)
        co_loss: CO incomplete combustion loss (%)
        radiation_loss: Radiation/convection loss (%)
        unaccounted_loss: Unaccounted losses (%)

    Returns:
        Combustion efficiency (% HHV)

    Example:
        >>> eff = calculate_efficiency_from_losses(5.0, 6.0, 0.2, 1.0, 0.5)
        >>> print(f"Efficiency: {eff:.1f}%")  # 87.3%
    """
    total_losses = (dry_flue_gas_loss + moisture_loss + co_loss +
                   radiation_loss + unaccounted_loss)
    return 100.0 - total_losses


def calculate_moisture_loss(
    hydrogen_pct: float,
    moisture_pct: float,
    flue_gas_temp_c: float,
    hhv_kj_kg: float
) -> float:
    """
    Calculate total moisture-related heat loss.

    Args:
        hydrogen_pct: Hydrogen content in fuel (mass %)
        moisture_pct: Moisture content in fuel (mass %)
        flue_gas_temp_c: Flue gas temperature (deg C)
        hhv_kj_kg: Fuel higher heating value (kJ/kg)

    Returns:
        Total moisture loss as percentage of HHV (%)
    """
    # H2O from hydrogen combustion: 9 kg H2O per kg H2
    m_h2o_h2 = 9.0 * (hydrogen_pct / 100.0)

    # H2O from fuel moisture
    m_h2o_fuel = moisture_pct / 100.0

    # Total H2O
    m_h2o_total = m_h2o_h2 + m_h2o_fuel

    # Enthalpy
    delta_t = flue_gas_temp_c - REFERENCE_TEMP_C
    h_h2o = LATENT_HEAT_H2O_25C + CP_WATER_VAPOR * delta_t

    # Loss
    return (m_h2o_total * h_h2o) / hhv_kj_kg * 100.0


def calculate_co_loss(
    co_ppm: float,
    co2_pct: float,
    carbon_pct: float,
    hhv_kj_kg: float
) -> float:
    """
    Calculate loss from incomplete combustion (CO formation).

    Args:
        co_ppm: CO concentration (ppm dry)
        co2_pct: CO2 concentration (vol % dry)
        carbon_pct: Carbon content in fuel (mass %)
        hhv_kj_kg: Fuel higher heating value (kJ/kg)

    Returns:
        CO loss as percentage of HHV (%)
    """
    if co_ppm <= 0:
        return 0.0

    co_pct = co_ppm / 10000.0
    co_co2_ratio = co_pct / (co_pct + co2_pct)

    return co_co2_ratio * 5654.0 * (carbon_pct / 100.0) / hhv_kj_kg * 100.0


def estimate_efficiency_quick(
    flue_gas_temp_c: float,
    o2_pct: float,
    fuel_type: str = "natural_gas"
) -> float:
    """
    Quick estimation of combustion efficiency.

    Uses simplified correlations for rapid assessment.

    Args:
        flue_gas_temp_c: Flue gas temperature (deg C)
        o2_pct: O2 concentration (vol % dry)
        fuel_type: Fuel type

    Returns:
        Estimated combustion efficiency (% HHV)

    Example:
        >>> eff = estimate_efficiency_quick(180, 3.5, "natural_gas")
        >>> print(f"Estimated Efficiency: {eff:.1f}%")
    """
    # Base efficiency at optimal conditions
    fuel_lower = fuel_type.lower()

    if "gas" in fuel_lower:
        base_eff = 92.0
        temp_coeff = 0.035  # % loss per deg C
    elif "oil" in fuel_lower:
        base_eff = 89.0
        temp_coeff = 0.040
    else:
        base_eff = 85.0
        temp_coeff = 0.045

    # Temperature penalty (reference: 150 deg C)
    temp_loss = max(0, (flue_gas_temp_c - 150)) * temp_coeff

    # Excess air penalty (reference: 2.5% O2)
    excess_air = (o2_pct / (20.95 - o2_pct)) * 100
    ea_loss = max(0, (excess_air - 15)) * 0.02

    return max(60.0, base_eff - temp_loss - ea_loss)


def get_siegert_k_factor(fuel_type: str) -> float:
    """
    Get Siegert K factor for a fuel type.

    Args:
        fuel_type: Fuel type identifier

    Returns:
        Siegert K factor

    Example:
        >>> k = get_siegert_k_factor("natural_gas")
        >>> print(f"K factor: {k}")  # 0.38
    """
    return SiegertConstants.get_k_factor(fuel_type)
