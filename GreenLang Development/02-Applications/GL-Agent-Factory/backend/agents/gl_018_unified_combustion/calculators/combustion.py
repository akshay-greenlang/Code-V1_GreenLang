"""
Core Combustion Calculations for GL-018 UNIFIEDCOMBUSTION Agent

This module implements fundamental combustion engineering calculations
following industry standards and zero-hallucination principles.

All formulas are from authoritative references:
- ASME PTC 4 Fired Steam Generators
- EPA Method 19 for combustion efficiency
- API 535 Burners for Fired Heaters
- Gas Engineers Handbook
- Industrial Heating Equipment Association

Zero-hallucination: All calculations are deterministic physics formulas.
No ML/LLM in the calculation path.

Example:
    >>> from combustion import calculate_combustion_efficiency
    >>> efficiency = calculate_combustion_efficiency(
    ...     o2_percent=3.5,
    ...     stack_temp_c=350,
    ...     fuel_type="natural_gas"
    ... )
    >>> print(f"Efficiency: {efficiency:.1f}%")
"""

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Fuel Properties Database - Deterministic Lookup
# =============================================================================

FUEL_PROPERTIES: Dict[str, Dict[str, float]] = {
    "natural_gas": {
        "stoich_air_ratio": 9.52,  # Stoichiometric air-fuel ratio (vol/vol)
        "stoich_o2_ratio": 2.0,    # O2 required per unit fuel (vol/vol)
        "hhv_mj_m3": 37.3,         # Higher heating value MJ/m3
        "lhv_mj_m3": 33.7,         # Lower heating value MJ/m3
        "co2_max_pct": 11.7,       # Max CO2 at stoichiometric (%)
        "adiabatic_temp_c": 1950,  # Adiabatic flame temperature at stoich
        "optimal_o2_min": 2.0,     # Optimal O2 range minimum
        "optimal_o2_max": 4.0,     # Optimal O2 range maximum
        "siegert_a2": 0.37,        # Siegert formula coefficient
        "carbon_content": 0.75,    # kg C per kg fuel
        "hydrogen_content": 0.25,  # kg H per kg fuel
        "density_kg_m3": 0.68,     # Density at STP
    },
    "propane": {
        "stoich_air_ratio": 23.8,
        "stoich_o2_ratio": 5.0,
        "hhv_mj_m3": 93.1,
        "lhv_mj_m3": 85.8,
        "co2_max_pct": 13.8,
        "adiabatic_temp_c": 1980,
        "optimal_o2_min": 2.5,
        "optimal_o2_max": 4.5,
        "siegert_a2": 0.38,
        "carbon_content": 0.82,
        "hydrogen_content": 0.18,
        "density_kg_m3": 1.88,
    },
    "butane": {
        "stoich_air_ratio": 30.9,
        "stoich_o2_ratio": 6.5,
        "hhv_mj_m3": 118.5,
        "lhv_mj_m3": 109.4,
        "co2_max_pct": 14.1,
        "adiabatic_temp_c": 1970,
        "optimal_o2_min": 2.5,
        "optimal_o2_max": 4.5,
        "siegert_a2": 0.39,
        "carbon_content": 0.83,
        "hydrogen_content": 0.17,
        "density_kg_m3": 2.50,
    },
    "fuel_oil_2": {
        "stoich_air_ratio": 14.1,  # kg air/kg fuel
        "stoich_o2_ratio": 3.3,
        "hhv_mj_kg": 45.5,
        "lhv_mj_kg": 42.6,
        "co2_max_pct": 15.4,
        "adiabatic_temp_c": 2030,
        "optimal_o2_min": 3.0,
        "optimal_o2_max": 5.0,
        "siegert_a2": 0.47,
        "carbon_content": 0.87,
        "hydrogen_content": 0.13,
        "density_kg_m3": 840,
    },
    "fuel_oil_6": {
        "stoich_air_ratio": 13.8,
        "stoich_o2_ratio": 3.5,
        "hhv_mj_kg": 42.5,
        "lhv_mj_kg": 40.2,
        "co2_max_pct": 16.0,
        "adiabatic_temp_c": 2080,
        "optimal_o2_min": 3.5,
        "optimal_o2_max": 6.0,
        "siegert_a2": 0.50,
        "carbon_content": 0.88,
        "hydrogen_content": 0.10,
        "density_kg_m3": 970,
    },
    "hydrogen": {
        "stoich_air_ratio": 2.38,
        "stoich_o2_ratio": 0.5,
        "hhv_mj_m3": 12.7,
        "lhv_mj_m3": 10.8,
        "co2_max_pct": 0.0,  # No CO2 from H2 combustion
        "adiabatic_temp_c": 2210,
        "optimal_o2_min": 1.0,
        "optimal_o2_max": 3.0,
        "siegert_a2": 0.32,
        "carbon_content": 0.0,
        "hydrogen_content": 1.0,
        "density_kg_m3": 0.082,
    },
    "biogas": {
        "stoich_air_ratio": 5.7,
        "stoich_o2_ratio": 1.2,
        "hhv_mj_m3": 22.0,
        "lhv_mj_m3": 20.0,
        "co2_max_pct": 10.5,
        "adiabatic_temp_c": 1750,
        "optimal_o2_min": 2.5,
        "optimal_o2_max": 5.0,
        "siegert_a2": 0.42,
        "carbon_content": 0.52,
        "hydrogen_content": 0.07,
        "density_kg_m3": 1.15,
    },
    "coal": {
        "stoich_air_ratio": 10.5,  # kg air/kg coal (approximate)
        "stoich_o2_ratio": 2.5,
        "hhv_mj_kg": 28.0,
        "lhv_mj_kg": 26.5,
        "co2_max_pct": 18.5,
        "adiabatic_temp_c": 2100,
        "optimal_o2_min": 4.0,
        "optimal_o2_max": 7.0,
        "siegert_a2": 0.55,
        "carbon_content": 0.75,
        "hydrogen_content": 0.05,
        "density_kg_m3": 1350,  # bulk density
    },
    "biomass": {
        "stoich_air_ratio": 5.8,
        "stoich_o2_ratio": 1.3,
        "hhv_mj_kg": 18.0,
        "lhv_mj_kg": 16.5,
        "co2_max_pct": 20.0,
        "adiabatic_temp_c": 1650,
        "optimal_o2_min": 5.0,
        "optimal_o2_max": 8.0,
        "siegert_a2": 0.58,
        "carbon_content": 0.50,
        "hydrogen_content": 0.06,
        "density_kg_m3": 500,  # typical bulk density
    },
}


# =============================================================================
# Core Combustion Calculations
# =============================================================================

def calculate_excess_air(o2_percent: float) -> float:
    """
    Calculate excess air percentage from flue gas O2 measurement.

    Formula (dry basis):
        Excess Air (%) = O2 / (21 - O2) * 100

    This is the fundamental equation for excess air calculation
    assuming complete combustion and dry flue gas measurement.

    Reference: ASME PTC 4, EPA Method 19

    Args:
        o2_percent: Oxygen percentage in flue gas (0-21%).

    Returns:
        Excess air percentage.

    Raises:
        ValueError: If O2 is outside valid range.

    Example:
        >>> excess = calculate_excess_air(3.0)
        >>> print(f"Excess air: {excess:.1f}%")
        Excess air: 16.7%
    """
    if o2_percent < 0:
        raise ValueError(f"O2 percent must be >= 0, got {o2_percent}")
    if o2_percent >= 21:
        raise ValueError(f"O2 percent must be < 21, got {o2_percent}")

    # Standard formula: EA = O2 / (21 - O2) * 100
    excess_air = (o2_percent / (21.0 - o2_percent)) * 100.0

    logger.debug(f"Excess air calculation: O2={o2_percent}% -> EA={excess_air:.1f}%")

    return excess_air


def calculate_lambda(o2_percent: float) -> float:
    """
    Calculate lambda (air-fuel equivalence ratio).

    Lambda = Actual Air / Stoichiometric Air
    Lambda = 1.0 at stoichiometric combustion
    Lambda > 1.0 is lean (excess air)
    Lambda < 1.0 is rich (excess fuel)

    Formula:
        Lambda = 1 + (Excess Air / 100)

    Reference: Automotive and combustion engineering standards

    Args:
        o2_percent: Oxygen percentage in flue gas.

    Returns:
        Lambda value (typically 1.0 to 1.5 for industrial burners).

    Example:
        >>> lambda_val = calculate_lambda(3.5)
        >>> print(f"Lambda: {lambda_val:.3f}")
    """
    excess_air = calculate_excess_air(o2_percent)
    lambda_val = 1.0 + (excess_air / 100.0)

    return lambda_val


def calculate_air_fuel_ratio(
    o2_percent: float,
    fuel_type: str
) -> Tuple[float, float]:
    """
    Calculate actual and stoichiometric air-fuel ratios.

    Formula:
        Actual A/F = Stoich A/F * Lambda
        Lambda = 1 + (Excess Air / 100)

    Args:
        o2_percent: Oxygen percentage in flue gas.
        fuel_type: Type of fuel from FUEL_PROPERTIES.

    Returns:
        Tuple of (actual_ratio, stoichiometric_ratio).

    Example:
        >>> actual, stoich = calculate_air_fuel_ratio(3.5, "natural_gas")
        >>> print(f"Actual A/F: {actual:.2f}, Stoich: {stoich:.2f}")
    """
    if fuel_type not in FUEL_PROPERTIES:
        logger.warning(f"Unknown fuel type {fuel_type}, using natural_gas")
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]
    stoich_ratio = fuel["stoich_air_ratio"]

    excess_air = calculate_excess_air(o2_percent)
    lambda_val = 1.0 + (excess_air / 100.0)
    actual_ratio = stoich_ratio * lambda_val

    return (actual_ratio, stoich_ratio)


def calculate_combustion_efficiency(
    o2_percent: float,
    stack_temp_c: float,
    fuel_type: str,
    ambient_temp_c: float = 25.0,
    co_ppm: float = 0.0
) -> Dict[str, float]:
    """
    Calculate combustion efficiency using the Siegert formula.

    The Siegert formula is the industry standard for boiler/burner
    efficiency assessment based on stack loss.

    Formula (Siegert):
        Stack Loss (%) = (A2 / CO2%) * (T_flue - T_ambient)
        Combustion Efficiency (%) = 100 - Stack Loss - Unburned Loss - Other Losses

    Reference: ASME PTC 4, VDI 2067, EPA Method 19

    Args:
        o2_percent: Oxygen percentage in flue gas (dry basis).
        stack_temp_c: Stack/flue gas temperature in Celsius.
        fuel_type: Type of fuel from FUEL_PROPERTIES.
        ambient_temp_c: Ambient air temperature in Celsius (default 25C).
        co_ppm: Carbon monoxide in ppm for unburned loss calculation.

    Returns:
        Dictionary with:
        - combustion_efficiency: Overall combustion efficiency (%)
        - stack_loss: Heat loss in stack gas (%)
        - unburned_loss: Loss from incomplete combustion (%)
        - excess_air: Excess air percentage
        - co2_actual: Actual CO2 percentage in flue gas

    Example:
        >>> result = calculate_combustion_efficiency(
        ...     o2_percent=3.5,
        ...     stack_temp_c=350,
        ...     fuel_type="natural_gas"
        ... )
        >>> print(f"Efficiency: {result['combustion_efficiency']:.1f}%")
    """
    if fuel_type not in FUEL_PROPERTIES:
        logger.warning(f"Unknown fuel type {fuel_type}, using natural_gas")
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]

    # Calculate excess air from O2
    excess_air = calculate_excess_air(o2_percent)

    # Calculate actual CO2 from stoichiometric CO2 and excess air
    # CO2_actual = CO2_max / (1 + EA/100)
    co2_max = fuel["co2_max_pct"]
    if co2_max > 0:
        co2_actual = co2_max / (1.0 + excess_air / 100.0)
        co2_actual = max(co2_actual, 1.0)  # Avoid division issues
    else:
        # For hydrogen - no CO2 produced
        co2_actual = 1.0  # Use equivalent factor for calculation

    # Siegert coefficient for the fuel
    a2 = fuel["siegert_a2"]

    # Stack loss using Siegert formula
    # Stack Loss = (A2 / CO2) * (T_stack - T_ambient)
    stack_loss = (a2 / co2_actual) * (stack_temp_c - ambient_temp_c)

    # Unburned fuel loss from CO
    # Approximately 0.001% loss per ppm CO
    if co_ppm > 0:
        unburned_loss = co_ppm * 0.001
    else:
        unburned_loss = 0.5  # Default small loss

    # Radiation and other minor losses (typically 1-2%)
    radiation_loss = 1.5

    # Total combustion efficiency
    combustion_efficiency = 100.0 - stack_loss - unburned_loss - radiation_loss

    # Clamp to reasonable range (65-99%)
    combustion_efficiency = max(65.0, min(99.0, combustion_efficiency))

    logger.debug(
        f"Combustion efficiency: stack_loss={stack_loss:.2f}%, "
        f"unburned={unburned_loss:.2f}%, total={combustion_efficiency:.1f}%"
    )

    return {
        "combustion_efficiency": combustion_efficiency,
        "stack_loss": stack_loss,
        "unburned_loss": unburned_loss,
        "radiation_loss": radiation_loss,
        "excess_air": excess_air,
        "co2_actual": co2_actual,
    }


def calculate_adiabatic_flame_temperature(
    fuel_type: str,
    excess_air_percent: float,
    preheat_air_temp_c: float = 25.0
) -> float:
    """
    Calculate adiabatic flame temperature.

    The adiabatic flame temperature is the maximum theoretical
    temperature achievable with no heat loss.

    Formula (approximation):
        T_ad = T_stoich - (EA * cooling_factor) + (preheat_bonus)

    Reference: Combustion Engineering texts, NFPA 86

    Args:
        fuel_type: Type of fuel from FUEL_PROPERTIES.
        excess_air_percent: Excess air percentage.
        preheat_air_temp_c: Combustion air preheat temperature (C).

    Returns:
        Adiabatic flame temperature in Celsius.

    Example:
        >>> temp = calculate_adiabatic_flame_temperature("natural_gas", 20, 200)
        >>> print(f"Adiabatic flame temp: {temp:.0f}C")
    """
    if fuel_type not in FUEL_PROPERTIES:
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]

    # Stoichiometric adiabatic temperature
    t_stoich = fuel["adiabatic_temp_c"]

    # Excess air cooling effect (~17C drop per 10% excess air)
    cooling_rate = 17.0  # C per 10% excess air
    excess_air_cooling = (excess_air_percent / 10.0) * cooling_rate

    # Preheat bonus (~0.45C increase per degree of preheat above ambient)
    preheat_bonus = (preheat_air_temp_c - 25.0) * 0.45

    # Calculate adiabatic temperature
    t_adiabatic = t_stoich - excess_air_cooling + preheat_bonus

    # Clamp to reasonable range
    t_adiabatic = max(800.0, min(2500.0, t_adiabatic))

    return t_adiabatic


def calculate_heat_input(
    fuel_flow_rate: float,
    fuel_type: str,
    fuel_heating_value: Optional[float] = None,
    flow_unit: str = "m3/h"
) -> float:
    """
    Calculate heat input rate in MW.

    Formula:
        Heat Input (MW) = Fuel Flow * Heating Value / 3600

    Args:
        fuel_flow_rate: Fuel flow rate in specified units.
        fuel_type: Type of fuel from FUEL_PROPERTIES.
        fuel_heating_value: Override heating value (MJ/unit), uses default if None.
        flow_unit: Unit of fuel flow (m3/h, kg/h).

    Returns:
        Heat input in MW.

    Example:
        >>> heat_mw = calculate_heat_input(1000, "natural_gas")
        >>> print(f"Heat input: {heat_mw:.2f} MW")
    """
    if fuel_type not in FUEL_PROPERTIES:
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]

    # Determine heating value
    if fuel_heating_value is not None:
        hv = fuel_heating_value
    elif "hhv_mj_m3" in fuel:
        hv = fuel["hhv_mj_m3"]
    elif "hhv_mj_kg" in fuel:
        hv = fuel["hhv_mj_kg"]
    else:
        hv = 37.3  # Default natural gas

    # Calculate heat input
    # Heat (MW) = Flow (unit/h) * HV (MJ/unit) / 3600 (s/h) * 1 (MW/MJ/s)
    heat_input_mw = fuel_flow_rate * hv / 3600.0

    return heat_input_mw


def calculate_co2_emission_rate(
    fuel_flow_rate: float,
    fuel_type: str,
    flow_unit: str = "m3/h"
) -> float:
    """
    Calculate CO2 emission rate in kg/h.

    Uses stoichiometric combustion calculation based on
    carbon content of fuel.

    Formula:
        CO2 (kg/h) = Fuel Flow * Density * Carbon Content * (44/12)

    The factor 44/12 is the molecular weight ratio of CO2 to C.

    Reference: GHG Protocol, EPA emission factors

    Args:
        fuel_flow_rate: Fuel flow rate.
        fuel_type: Type of fuel.
        flow_unit: Unit of fuel flow.

    Returns:
        CO2 emission rate in kg/h.
    """
    if fuel_type not in FUEL_PROPERTIES:
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]

    # Get fuel properties
    carbon_content = fuel["carbon_content"]
    density = fuel["density_kg_m3"]

    # Calculate mass flow rate
    if flow_unit == "m3/h":
        mass_flow_kgh = fuel_flow_rate * density
    else:
        mass_flow_kgh = fuel_flow_rate

    # CO2 emission rate
    # CO2 = Mass Flow * Carbon Content * (MW_CO2 / MW_C)
    co2_rate_kgh = mass_flow_kgh * carbon_content * (44.0 / 12.0)

    return co2_rate_kgh


def calculate_nox_emission_rate(
    nox_ppm: float,
    flue_gas_flow_m3h: float,
    o2_percent: float = 3.0
) -> float:
    """
    Calculate NOx emission rate in kg/h.

    Converts NOx concentration to mass emission rate.

    Formula:
        NOx (kg/h) = NOx(ppm) * FlueGasFlow(m3/h) * MW_NOx / (22.4 * 1e6)

    Reference: EPA Method 19, 40 CFR Part 60

    Args:
        nox_ppm: NOx concentration in ppm.
        flue_gas_flow_m3h: Flue gas flow rate in m3/h at STP.
        o2_percent: Reference O2 for correction (default 3%).

    Returns:
        NOx emission rate in kg/h.
    """
    # Molecular weight of NO2 (using NO2 as NOx equivalent)
    mw_no2 = 46.0

    # Convert ppm to mass rate
    # ppm = (volume NOx / volume total) * 1e6
    # kg/h = ppm * m3/h * kg/m3
    # At STP: 1 mole = 22.4 L, so density = MW/22.4 kg/m3
    nox_rate_kgh = nox_ppm * flue_gas_flow_m3h * mw_no2 / (22.4 * 1e6)

    return nox_rate_kgh


def correct_emissions_to_reference_o2(
    measured_value: float,
    measured_o2: float,
    reference_o2: float = 3.0
) -> float:
    """
    Correct emission concentration to reference O2 basis.

    Standard reference O2 is 3% for most regulations.

    Formula:
        Corrected = Measured * (21 - Reference_O2) / (21 - Measured_O2)

    Reference: EPA 40 CFR Part 60, EU Industrial Emissions Directive

    Args:
        measured_value: Measured concentration (ppm or mg/m3).
        measured_o2: Measured O2 percentage.
        reference_o2: Reference O2 percentage (default 3%).

    Returns:
        Corrected concentration at reference O2.
    """
    if measured_o2 >= 21:
        raise ValueError(f"Measured O2 must be < 21%, got {measured_o2}")
    if reference_o2 >= 21:
        raise ValueError(f"Reference O2 must be < 21%, got {reference_o2}")

    # O2 correction formula
    corrected = measured_value * (21.0 - reference_o2) / (21.0 - measured_o2)

    return corrected


def calculate_emission_index(
    emission_rate_kgh: float,
    heat_input_mw: float
) -> float:
    """
    Calculate emission index in g/GJ or kg/GJ.

    Emission index normalizes emissions to heat input for
    comparison across different equipment sizes.

    Formula:
        Emission Index = Emission Rate / Heat Input
        (kg/h) / (MW) = kg/MWh = kg/GJ * 3.6

    Args:
        emission_rate_kgh: Emission rate in kg/h.
        heat_input_mw: Heat input in MW.

    Returns:
        Emission index in kg/GJ.
    """
    if heat_input_mw <= 0:
        raise ValueError(f"Heat input must be > 0, got {heat_input_mw}")

    # Convert MW to GJ/h: 1 MW = 3.6 GJ/h
    heat_input_gjh = heat_input_mw * 3.6

    # Emission index
    emission_index = emission_rate_kgh / heat_input_gjh

    return emission_index


def estimate_flue_gas_flow(
    fuel_flow_rate: float,
    fuel_type: str,
    excess_air_percent: float,
    flow_unit: str = "m3/h"
) -> float:
    """
    Estimate flue gas volumetric flow rate.

    Formula:
        Flue Gas = (Stoich Air + Excess Air + Products) * Fuel Flow
        Simplified: Flue Gas ~ Fuel Flow * (1 + Stoich_Air_Ratio * Lambda)

    Args:
        fuel_flow_rate: Fuel flow rate.
        fuel_type: Type of fuel.
        excess_air_percent: Excess air percentage.
        flow_unit: Unit of fuel flow.

    Returns:
        Estimated flue gas flow in m3/h at STP.
    """
    if fuel_type not in FUEL_PROPERTIES:
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]
    stoich_air = fuel["stoich_air_ratio"]

    # Lambda from excess air
    lambda_val = 1.0 + excess_air_percent / 100.0

    # Approximate flue gas flow
    # Products + excess air
    # For natural gas: ~1 m3 fuel -> ~10-11 m3 flue gas at stoich
    flue_gas_factor = 1.0 + stoich_air * lambda_val

    flue_gas_flow = fuel_flow_rate * flue_gas_factor

    return flue_gas_flow


def calculate_fuel_composition_heating_value(
    methane_pct: float = 0,
    ethane_pct: float = 0,
    propane_pct: float = 0,
    butane_pct: float = 0,
    hydrogen_pct: float = 0,
    co2_pct: float = 0,
    nitrogen_pct: float = 0
) -> Dict[str, float]:
    """
    Calculate heating value from detailed gas composition.

    Uses component heating values and molar fractions for
    accurate HHV/LHV calculation.

    Reference: Gas Engineers Handbook, ASTM D3588

    Args:
        methane_pct: Methane (CH4) percentage.
        ethane_pct: Ethane (C2H6) percentage.
        propane_pct: Propane (C3H8) percentage.
        butane_pct: Butane (C4H10) percentage.
        hydrogen_pct: Hydrogen (H2) percentage.
        co2_pct: CO2 percentage (inert).
        nitrogen_pct: N2 percentage (inert).

    Returns:
        Dictionary with HHV and LHV in MJ/m3.
    """
    # Component heating values (MJ/m3 at STP)
    HHV_COMPONENTS = {
        "methane": 37.7,
        "ethane": 66.0,
        "propane": 93.9,
        "butane": 121.8,
        "hydrogen": 12.7,
        "co2": 0.0,
        "nitrogen": 0.0,
    }

    LHV_COMPONENTS = {
        "methane": 33.9,
        "ethane": 60.4,
        "propane": 86.4,
        "butane": 112.4,
        "hydrogen": 10.8,
        "co2": 0.0,
        "nitrogen": 0.0,
    }

    # Calculate weighted average
    total_pct = (methane_pct + ethane_pct + propane_pct +
                 butane_pct + hydrogen_pct + co2_pct + nitrogen_pct)

    if total_pct <= 0:
        # Return default natural gas values
        return {"hhv_mj_m3": 37.3, "lhv_mj_m3": 33.7}

    # Normalize to 100%
    factor = 100.0 / total_pct if total_pct != 100 else 1.0

    hhv = (
        methane_pct * HHV_COMPONENTS["methane"] +
        ethane_pct * HHV_COMPONENTS["ethane"] +
        propane_pct * HHV_COMPONENTS["propane"] +
        butane_pct * HHV_COMPONENTS["butane"] +
        hydrogen_pct * HHV_COMPONENTS["hydrogen"]
    ) * factor / 100.0

    lhv = (
        methane_pct * LHV_COMPONENTS["methane"] +
        ethane_pct * LHV_COMPONENTS["ethane"] +
        propane_pct * LHV_COMPONENTS["propane"] +
        butane_pct * LHV_COMPONENTS["butane"] +
        hydrogen_pct * LHV_COMPONENTS["hydrogen"]
    ) * factor / 100.0

    return {
        "hhv_mj_m3": hhv,
        "lhv_mj_m3": lhv,
        "wobbe_index": hhv / math.sqrt(total_pct / 100.0) if total_pct > 0 else 0,
    }


def calculate_stoichiometric_o2_requirement(
    fuel_type: str,
    fuel_flow_rate: float,
    flow_unit: str = "m3/h"
) -> float:
    """
    Calculate stoichiometric O2 requirement for complete combustion.

    Formula for hydrocarbons (CxHy):
        O2 required = x + y/4 moles per mole of fuel

    Args:
        fuel_type: Type of fuel.
        fuel_flow_rate: Fuel flow rate.
        flow_unit: Unit of fuel flow.

    Returns:
        O2 requirement in m3/h at STP.
    """
    if fuel_type not in FUEL_PROPERTIES:
        fuel_type = "natural_gas"

    fuel = FUEL_PROPERTIES[fuel_type]
    stoich_o2_ratio = fuel["stoich_o2_ratio"]

    o2_required = fuel_flow_rate * stoich_o2_ratio

    return o2_required


def apply_decimal_precision(
    value: float,
    precision: int = 3
) -> Decimal:
    """
    Apply regulatory-compliant decimal precision.

    Uses ROUND_HALF_UP (banker's rounding) for regulatory consistency.

    Args:
        value: Value to round.
        precision: Number of decimal places.

    Returns:
        Decimal with specified precision.
    """
    quantize_str = '0.' + '0' * precision
    decimal_value = Decimal(str(value))
    return decimal_value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
