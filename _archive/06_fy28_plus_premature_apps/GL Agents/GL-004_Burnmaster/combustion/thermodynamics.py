"""
GL-004 BURNMASTER Thermodynamics Module

Thermodynamic property calculations for combustion gases,
heat balances, efficiency calculations, and energy analysis.

Reference Standards:
- NIST-JANAF Thermochemical Tables
- ASME PTC 4.1 (Steam Generating Units)
- ISO 13443 (Natural gas - Standard reference conditions)

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np


class EfficiencyMethod(str, Enum):
    """Methods for efficiency calculation."""
    INPUT_OUTPUT = "input_output"  # Direct method
    HEAT_LOSS = "heat_loss"  # Indirect method (ASME PTC 4)


@dataclass(frozen=True)
class GasProperties:
    """Thermodynamic properties of a gas."""
    cp_j_mol_k: float  # Molar heat capacity at constant pressure
    cp_j_kg_k: float  # Mass heat capacity at constant pressure
    cv_j_mol_k: float  # Molar heat capacity at constant volume
    cv_j_kg_k: float  # Mass heat capacity at constant volume
    gamma: float  # Heat capacity ratio (cp/cv)
    molecular_weight: float  # kg/kmol
    enthalpy_j_mol: float  # Molar enthalpy
    entropy_j_mol_k: float  # Molar entropy


class HeatBalanceResult(NamedTuple):
    """Result of heat balance calculation."""
    heat_input_mw: float
    useful_output_mw: float
    stack_loss_mw: float
    radiation_loss_mw: float
    unburned_loss_mw: float
    moisture_loss_mw: float
    other_losses_mw: float
    efficiency_pct: float


class EfficiencyResult(NamedTuple):
    """Detailed efficiency breakdown."""
    gross_efficiency_pct: float
    net_efficiency_pct: float
    combustion_efficiency_pct: float
    heat_transfer_efficiency_pct: float

    # Losses (% of heat input)
    dry_flue_gas_loss_pct: float
    moisture_in_fuel_loss_pct: float
    moisture_from_h2_loss_pct: float
    moisture_in_air_loss_pct: float
    unburned_carbon_loss_pct: float
    radiation_loss_pct: float
    manufacturer_margin_pct: float


# Shomate equation coefficients for common gases
# Cp = A + B*t + C*t^2 + D*t^3 + E/t^2 (J/mol/K)
# where t = T(K) / 1000
SHOMATE_COEFFICIENTS = {
    "N2": {
        "A": 28.98641,
        "B": 1.853978,
        "C": -9.647459,
        "D": 16.63537,
        "E": 0.000117,
        "F": -8.671914,
        "G": 226.4168,
        "H": 0.0,
        "T_min": 298,
        "T_max": 6000,
    },
    "O2": {
        "A": 30.03235,
        "B": 8.772972,
        "C": -3.988133,
        "D": 0.788313,
        "E": -0.741599,
        "F": -11.32468,
        "G": 236.1663,
        "H": 0.0,
        "T_min": 298,
        "T_max": 6000,
    },
    "CO2": {
        "A": 24.99735,
        "B": 55.18696,
        "C": -33.69137,
        "D": 7.948387,
        "E": -0.136638,
        "F": -403.6075,
        "G": 228.2431,
        "H": -393.5224,
        "T_min": 298,
        "T_max": 1200,
    },
    "H2O": {
        "A": 30.09200,
        "B": 6.832514,
        "C": 6.793435,
        "D": -2.534480,
        "E": 0.082139,
        "F": -250.8810,
        "G": 223.3967,
        "H": -241.8264,
        "T_min": 500,
        "T_max": 1700,
    },
    "CO": {
        "A": 25.56759,
        "B": 6.096130,
        "C": 4.054656,
        "D": -2.671301,
        "E": 0.131021,
        "F": -118.0089,
        "G": 227.3665,
        "H": -110.5271,
        "T_min": 298,
        "T_max": 1300,
    },
    "CH4": {
        "A": -0.703029,
        "B": 108.4773,
        "C": -42.52157,
        "D": 5.862788,
        "E": 0.678565,
        "F": -76.84376,
        "G": 158.7163,
        "H": -74.87310,
        "T_min": 298,
        "T_max": 1300,
    },
}

# Molecular weights (kg/kmol)
MOLECULAR_WEIGHTS = {
    "N2": 28.014,
    "O2": 31.998,
    "CO2": 44.01,
    "H2O": 18.015,
    "CO": 28.01,
    "CH4": 16.043,
    "SO2": 64.066,
    "Air": 28.97,
}

# Higher and Lower heating values (MJ/kg)
HEATING_VALUES = {
    "natural_gas": {"HHV": 55.5, "LHV": 50.0},
    "propane": {"HHV": 50.4, "LHV": 46.4},
    "fuel_oil_2": {"HHV": 45.5, "LHV": 42.8},
    "fuel_oil_6": {"HHV": 43.0, "LHV": 40.5},
    "coal_bituminous": {"HHV": 32.5, "LHV": 31.0},
    "hydrogen": {"HHV": 142.0, "LHV": 120.0},
}

# Gas constant
R_UNIVERSAL = 8.314462  # J/mol/K


def compute_cp_shomate(species: str, temp_k: float) -> float:
    """
    Compute heat capacity using Shomate equation.

    Args:
        species: Chemical species name
        temp_k: Temperature (K)

    Returns:
        Molar heat capacity Cp (J/mol/K)
    """
    if species not in SHOMATE_COEFFICIENTS:
        raise ValueError(f"Unknown species: {species}")

    coef = SHOMATE_COEFFICIENTS[species]
    t = temp_k / 1000.0  # Convert to kK

    cp = (
        coef["A"] +
        coef["B"] * t +
        coef["C"] * t**2 +
        coef["D"] * t**3 +
        coef["E"] / t**2
    )

    return cp


def compute_enthalpy_shomate(species: str, temp_k: float) -> float:
    """
    Compute molar enthalpy using Shomate equation.

    Args:
        species: Chemical species name
        temp_k: Temperature (K)

    Returns:
        Molar enthalpy (kJ/mol), relative to 298.15 K
    """
    if species not in SHOMATE_COEFFICIENTS:
        raise ValueError(f"Unknown species: {species}")

    coef = SHOMATE_COEFFICIENTS[species]
    t = temp_k / 1000.0  # Convert to kK

    h = (
        coef["A"] * t +
        coef["B"] * t**2 / 2 +
        coef["C"] * t**3 / 3 +
        coef["D"] * t**4 / 4 -
        coef["E"] / t +
        coef["F"]
    )

    return h  # kJ/mol


def compute_entropy_shomate(species: str, temp_k: float) -> float:
    """
    Compute molar entropy using Shomate equation.

    Args:
        species: Chemical species name
        temp_k: Temperature (K)

    Returns:
        Molar entropy (J/mol/K)
    """
    if species not in SHOMATE_COEFFICIENTS:
        raise ValueError(f"Unknown species: {species}")

    coef = SHOMATE_COEFFICIENTS[species]
    t = temp_k / 1000.0  # Convert to kK

    s = (
        coef["A"] * math.log(t) +
        coef["B"] * t +
        coef["C"] * t**2 / 2 +
        coef["D"] * t**3 / 3 -
        coef["E"] / (2 * t**2) +
        coef["G"]
    )

    return s  # J/mol/K


def compute_flue_gas_enthalpy(
    temp_k: float,
    co2_mol_frac: float,
    h2o_mol_frac: float,
    n2_mol_frac: float,
    o2_mol_frac: float,
    reference_temp_k: float = 298.15,
) -> float:
    """
    Compute enthalpy of flue gas mixture.

    Args:
        temp_k: Gas temperature (K)
        co2_mol_frac: CO2 mole fraction
        h2o_mol_frac: H2O mole fraction
        n2_mol_frac: N2 mole fraction
        o2_mol_frac: O2 mole fraction
        reference_temp_k: Reference temperature (K)

    Returns:
        Mixture enthalpy (kJ/mol)
    """
    # Compute individual enthalpies
    h_co2 = compute_enthalpy_shomate("CO2", temp_k) - compute_enthalpy_shomate("CO2", reference_temp_k)
    h_h2o = compute_enthalpy_shomate("H2O", temp_k) - compute_enthalpy_shomate("H2O", reference_temp_k)
    h_n2 = compute_enthalpy_shomate("N2", temp_k) - compute_enthalpy_shomate("N2", reference_temp_k)
    h_o2 = compute_enthalpy_shomate("O2", temp_k) - compute_enthalpy_shomate("O2", reference_temp_k)

    # Mixture enthalpy
    h_mix = (
        co2_mol_frac * h_co2 +
        h2o_mol_frac * h_h2o +
        n2_mol_frac * h_n2 +
        o2_mol_frac * h_o2
    )

    return h_mix


def compute_flue_gas_cp(
    temp_k: float,
    co2_mol_frac: float,
    h2o_mol_frac: float,
    n2_mol_frac: float,
    o2_mol_frac: float,
) -> float:
    """
    Compute heat capacity of flue gas mixture.

    Args:
        temp_k: Gas temperature (K)
        co2_mol_frac: CO2 mole fraction
        h2o_mol_frac: H2O mole fraction
        n2_mol_frac: N2 mole fraction
        o2_mol_frac: O2 mole fraction

    Returns:
        Mixture Cp (J/mol/K)
    """
    cp_co2 = compute_cp_shomate("CO2", temp_k)
    cp_h2o = compute_cp_shomate("H2O", temp_k)
    cp_n2 = compute_cp_shomate("N2", temp_k)
    cp_o2 = compute_cp_shomate("O2", temp_k)

    cp_mix = (
        co2_mol_frac * cp_co2 +
        h2o_mol_frac * cp_h2o +
        n2_mol_frac * cp_n2 +
        o2_mol_frac * cp_o2
    )

    return cp_mix


def compute_stack_loss(
    stack_temp_c: float,
    ambient_temp_c: float,
    excess_o2_pct: float,
    fuel_type: str = "natural_gas",
    moisture_in_fuel_pct: float = 0.0,
) -> float:
    """
    Compute dry flue gas heat loss (stack loss).

    Based on simplified Siegert formula.

    Args:
        stack_temp_c: Stack temperature (°C)
        ambient_temp_c: Ambient temperature (°C)
        excess_o2_pct: Excess O2 in flue gas (%)
        fuel_type: Type of fuel
        moisture_in_fuel_pct: Moisture content in fuel (%)

    Returns:
        Stack loss as percentage of heat input
    """
    # Temperature difference
    delta_t = stack_temp_c - ambient_temp_c

    # Siegert constants for different fuels
    siegert_constants = {
        "natural_gas": {"A1": 0.37, "A2": 0.009},
        "propane": {"A1": 0.42, "A2": 0.010},
        "fuel_oil_2": {"A1": 0.47, "A2": 0.007},
        "fuel_oil_6": {"A1": 0.50, "A2": 0.006},
        "coal_bituminous": {"A1": 0.65, "A2": 0.005},
    }

    constants = siegert_constants.get(fuel_type, {"A1": 0.45, "A2": 0.008})

    # Compute CO2_max for fuel (approximate)
    co2_max = {
        "natural_gas": 11.8,
        "propane": 13.8,
        "fuel_oil_2": 15.5,
        "fuel_oil_6": 16.5,
        "coal_bituminous": 18.5,
    }.get(fuel_type, 12.0)

    # Actual CO2 (approximate from O2)
    co2_actual = co2_max * (21.0 - excess_o2_pct) / 21.0

    if co2_actual <= 0:
        return 100.0  # Invalid - treat as complete loss

    # Stack loss (Siegert formula)
    stack_loss = (constants["A1"] + constants["A2"] * delta_t) * delta_t / co2_actual

    # Add moisture loss if applicable
    moisture_loss = moisture_in_fuel_pct * 0.1  # Approximate

    return stack_loss + moisture_loss


def compute_radiation_loss(
    furnace_rating_mw: float,
    load_fraction: float,
) -> float:
    """
    Estimate radiation and convection loss from furnace casing.

    Based on ASME PTC 4 guidelines.

    Args:
        furnace_rating_mw: Furnace design rating (MW thermal)
        load_fraction: Current load as fraction of design (0-1)

    Returns:
        Radiation loss as percentage of heat input
    """
    if load_fraction <= 0:
        return 0.0

    # Base radiation loss at full load (larger units have lower %)
    if furnace_rating_mw <= 5:
        base_loss = 1.5
    elif furnace_rating_mw <= 20:
        base_loss = 1.0
    elif furnace_rating_mw <= 100:
        base_loss = 0.6
    else:
        base_loss = 0.4

    # Radiation loss increases at lower loads (approximately proportional to 1/load)
    # but capped at a maximum
    radiation_loss = min(base_loss / load_fraction, 3.0)

    return radiation_loss


def compute_unburned_loss(
    co_ppm: float,
    combustible_in_ash_pct: float = 0.0,
    fuel_type: str = "natural_gas",
) -> float:
    """
    Compute loss due to unburned combustibles.

    Args:
        co_ppm: CO in flue gas (ppm)
        combustible_in_ash_pct: Combustible content in ash (for solid fuels)
        fuel_type: Type of fuel

    Returns:
        Unburned loss as percentage of heat input
    """
    # CO loss (approximate)
    # CO has ~28% of the heating value of the carbon that didn't burn to CO2
    # Typical: 0.001% loss per ppm CO for natural gas
    co_loss = co_ppm * 0.001

    # Combustible in ash loss (for solid fuels)
    # Approximate: each 1% combustible = 0.5% heat loss
    ash_loss = combustible_in_ash_pct * 0.5

    return co_loss + ash_loss


def compute_efficiency_indirect(
    stack_temp_c: float,
    ambient_temp_c: float,
    excess_o2_pct: float,
    co_ppm: float,
    furnace_rating_mw: float,
    load_fraction: float,
    fuel_type: str = "natural_gas",
    moisture_in_fuel_pct: float = 0.0,
    combustible_in_ash_pct: float = 0.0,
    manufacturer_margin_pct: float = 0.5,
) -> EfficiencyResult:
    """
    Compute thermal efficiency using indirect (heat loss) method.

    Per ASME PTC 4 methodology.

    Args:
        stack_temp_c: Stack temperature (°C)
        ambient_temp_c: Ambient temperature (°C)
        excess_o2_pct: Excess O2 in flue gas (%)
        co_ppm: CO in flue gas (ppm)
        furnace_rating_mw: Furnace design rating (MW thermal)
        load_fraction: Current load fraction (0-1)
        fuel_type: Type of fuel
        moisture_in_fuel_pct: Moisture content in fuel (%)
        combustible_in_ash_pct: Combustible in ash (for solid fuels)
        manufacturer_margin_pct: Manufacturer's margin allowance

    Returns:
        Detailed efficiency breakdown
    """
    # Compute individual losses
    stack_loss = compute_stack_loss(
        stack_temp_c, ambient_temp_c, excess_o2_pct,
        fuel_type, moisture_in_fuel_pct
    )

    radiation_loss = compute_radiation_loss(furnace_rating_mw, load_fraction)

    unburned_loss = compute_unburned_loss(co_ppm, combustible_in_ash_pct, fuel_type)

    # Moisture losses (simplified)
    moisture_fuel_loss = moisture_in_fuel_pct * 0.1
    moisture_h2_loss = 5.0 if fuel_type == "natural_gas" else 3.0  # Approx
    moisture_air_loss = 0.2  # Small for most conditions

    # Total losses
    total_losses = (
        stack_loss +
        radiation_loss +
        unburned_loss +
        moisture_fuel_loss +
        moisture_h2_loss +
        moisture_air_loss +
        manufacturer_margin_pct
    )

    # Gross efficiency (based on HHV)
    gross_efficiency = 100.0 - total_losses

    # Net efficiency (approximate, based on LHV)
    # For natural gas, LHV is about 90% of HHV
    lhv_hhv_ratio = 0.90 if fuel_type == "natural_gas" else 0.95
    net_efficiency = gross_efficiency / lhv_hhv_ratio

    # Combustion efficiency (fuel burnt completely)
    combustion_efficiency = 100.0 - unburned_loss

    # Heat transfer efficiency
    heat_transfer_efficiency = gross_efficiency / combustion_efficiency * 100

    return EfficiencyResult(
        gross_efficiency_pct=max(0, gross_efficiency),
        net_efficiency_pct=min(100, net_efficiency),
        combustion_efficiency_pct=combustion_efficiency,
        heat_transfer_efficiency_pct=heat_transfer_efficiency,
        dry_flue_gas_loss_pct=stack_loss - moisture_in_fuel_pct * 0.1,
        moisture_in_fuel_loss_pct=moisture_fuel_loss,
        moisture_from_h2_loss_pct=moisture_h2_loss,
        moisture_in_air_loss_pct=moisture_air_loss,
        unburned_carbon_loss_pct=unburned_loss,
        radiation_loss_pct=radiation_loss,
        manufacturer_margin_pct=manufacturer_margin_pct,
    )


def compute_efficiency_direct(
    useful_output_mw: float,
    fuel_flow_kg_s: float,
    fuel_type: str = "natural_gas",
    use_lhv: bool = False,
) -> float:
    """
    Compute thermal efficiency using direct (input-output) method.

    Args:
        useful_output_mw: Useful heat output (MW)
        fuel_flow_kg_s: Fuel mass flow rate (kg/s)
        fuel_type: Type of fuel
        use_lhv: Use lower heating value (default: HHV)

    Returns:
        Thermal efficiency (%)
    """
    hv_key = "LHV" if use_lhv else "HHV"
    heating_value = HEATING_VALUES.get(fuel_type, {}).get(hv_key, 50.0)

    heat_input_mw = fuel_flow_kg_s * heating_value

    if heat_input_mw <= 0:
        return 0.0

    efficiency = (useful_output_mw / heat_input_mw) * 100

    return min(100, max(0, efficiency))


def compute_heat_rate(
    power_output_mw: float,
    fuel_flow_kg_s: float,
    fuel_type: str = "natural_gas",
    use_lhv: bool = False,
) -> float:
    """
    Compute heat rate (fuel energy per unit output).

    Args:
        power_output_mw: Power output (MW)
        fuel_flow_kg_s: Fuel mass flow rate (kg/s)
        fuel_type: Type of fuel
        use_lhv: Use lower heating value

    Returns:
        Heat rate (kJ/kWh or BTU/kWh)
    """
    if power_output_mw <= 0:
        return float("inf")

    hv_key = "LHV" if use_lhv else "HHV"
    heating_value_mj_kg = HEATING_VALUES.get(fuel_type, {}).get(hv_key, 50.0)

    heat_input_mw = fuel_flow_kg_s * heating_value_mj_kg

    # Heat rate in MJ/MWh = kJ/kWh
    heat_rate = heat_input_mw / power_output_mw * 1000  # kJ/kWh

    return heat_rate


def compute_fuel_intensity(
    fuel_flow_kg_s: float,
    production_rate: float,
    production_unit: str = "tonne",
) -> float:
    """
    Compute fuel intensity (fuel per unit production).

    Args:
        fuel_flow_kg_s: Fuel mass flow rate (kg/s)
        production_rate: Production rate
        production_unit: Unit of production

    Returns:
        Fuel intensity (kg fuel per unit production per hour)
    """
    if production_rate <= 0:
        return float("inf")

    fuel_kg_h = fuel_flow_kg_s * 3600
    fuel_intensity = fuel_kg_h / production_rate

    return fuel_intensity


def compute_heat_balance(
    fuel_flow_kg_s: float,
    useful_output_mw: float,
    stack_temp_c: float,
    ambient_temp_c: float,
    excess_o2_pct: float,
    furnace_rating_mw: float,
    co_ppm: float = 100.0,
    fuel_type: str = "natural_gas",
) -> HeatBalanceResult:
    """
    Compute complete heat balance for the combustion system.

    Args:
        fuel_flow_kg_s: Fuel flow rate (kg/s)
        useful_output_mw: Useful heat output (MW)
        stack_temp_c: Stack temperature (°C)
        ambient_temp_c: Ambient temperature (°C)
        excess_o2_pct: Excess O2 (%)
        furnace_rating_mw: Furnace rating (MW)
        co_ppm: CO in flue gas (ppm)
        fuel_type: Fuel type

    Returns:
        Heat balance breakdown
    """
    # Heat input
    heating_value = HEATING_VALUES.get(fuel_type, {}).get("HHV", 50.0)
    heat_input_mw = fuel_flow_kg_s * heating_value

    if heat_input_mw <= 0:
        return HeatBalanceResult(
            heat_input_mw=0,
            useful_output_mw=0,
            stack_loss_mw=0,
            radiation_loss_mw=0,
            unburned_loss_mw=0,
            moisture_loss_mw=0,
            other_losses_mw=0,
            efficiency_pct=0,
        )

    load_fraction = min(1.0, heat_input_mw / furnace_rating_mw) if furnace_rating_mw > 0 else 1.0

    # Compute losses as percentages
    stack_loss_pct = compute_stack_loss(stack_temp_c, ambient_temp_c, excess_o2_pct, fuel_type)
    radiation_loss_pct = compute_radiation_loss(furnace_rating_mw, load_fraction)
    unburned_loss_pct = compute_unburned_loss(co_ppm, 0, fuel_type)
    moisture_loss_pct = 5.0  # Approximate for natural gas

    # Convert to MW
    stack_loss_mw = heat_input_mw * stack_loss_pct / 100
    radiation_loss_mw = heat_input_mw * radiation_loss_pct / 100
    unburned_loss_mw = heat_input_mw * unburned_loss_pct / 100
    moisture_loss_mw = heat_input_mw * moisture_loss_pct / 100

    # Other losses (to balance)
    accounted_losses = (stack_loss_mw + radiation_loss_mw +
                        unburned_loss_mw + moisture_loss_mw)
    expected_losses = heat_input_mw - useful_output_mw
    other_losses_mw = max(0, expected_losses - accounted_losses)

    # Efficiency
    efficiency_pct = (useful_output_mw / heat_input_mw) * 100 if heat_input_mw > 0 else 0

    return HeatBalanceResult(
        heat_input_mw=heat_input_mw,
        useful_output_mw=useful_output_mw,
        stack_loss_mw=stack_loss_mw,
        radiation_loss_mw=radiation_loss_mw,
        unburned_loss_mw=unburned_loss_mw,
        moisture_loss_mw=moisture_loss_mw,
        other_losses_mw=other_losses_mw,
        efficiency_pct=efficiency_pct,
    )
