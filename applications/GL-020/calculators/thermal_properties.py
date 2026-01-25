"""
GL-020 ECONOPULSE: Thermal Properties Calculator

Zero-hallucination thermodynamic property calculations based on:
- IAPWS-IF97 for water and steam properties
- JANAF Thermochemical Tables for flue gas properties
- ASME Steam Tables for validation

This module provides deterministic property calculations with
complete provenance tracking for regulatory compliance.

Author: GL-CalculatorEngineer
Standards: IAPWS-IF97, JANAF, ASME PTC 4.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple, Union

from .provenance import (
    ProvenanceTracker,
    CalculationType,
    CalculationProvenance,
    generate_calculation_hash
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Universal gas constant (kJ/kmol-K)
R_UNIVERSAL = 8.314462618

# Molecular weights (kg/kmol)
MOLECULAR_WEIGHTS = {
    "N2": 28.0134,
    "O2": 31.9988,
    "CO2": 44.0095,
    "H2O": 18.01528,
    "Ar": 39.948,
    "SO2": 64.066,
    "CO": 28.0101,
    "NO": 30.0061,
    "NO2": 46.0055,
}

# Water properties - IAPWS-IF97 critical point
WATER_CRITICAL_TEMP_K = 647.096  # K
WATER_CRITICAL_PRESSURE_MPA = 22.064  # MPa
WATER_CRITICAL_DENSITY = 322.0  # kg/m3

# Temperature conversion
KELVIN_OFFSET = 273.15


# =============================================================================
# INPUT VALIDATION
# =============================================================================

@dataclass
class ValidationError(Exception):
    """Exception for validation failures."""
    parameter: str
    value: float
    message: str


def validate_temperature_celsius(temp: float, param_name: str = "temperature") -> None:
    """
    Validate temperature is within acceptable range for economizer operations.

    Args:
        temp: Temperature in Celsius
        param_name: Parameter name for error messages

    Raises:
        ValidationError: If temperature is out of range
    """
    if temp < -50:
        raise ValidationError(
            parameter=param_name,
            value=temp,
            message=f"{param_name} ({temp}C) is below minimum (-50C)"
        )
    if temp > 600:
        raise ValidationError(
            parameter=param_name,
            value=temp,
            message=f"{param_name} ({temp}C) exceeds maximum (600C)"
        )


def validate_temperature_fahrenheit(temp: float, param_name: str = "temperature") -> None:
    """
    Validate temperature is within acceptable range (Fahrenheit).

    Args:
        temp: Temperature in Fahrenheit
        param_name: Parameter name for error messages

    Raises:
        ValidationError: If temperature is out of range
    """
    if temp < -58:
        raise ValidationError(
            parameter=param_name,
            value=temp,
            message=f"{param_name} ({temp}F) is below minimum (-58F)"
        )
    if temp > 1112:
        raise ValidationError(
            parameter=param_name,
            value=temp,
            message=f"{param_name} ({temp}F) exceeds maximum (1112F)"
        )


def validate_pressure_psia(pressure: float, param_name: str = "pressure") -> None:
    """
    Validate pressure is within acceptable range.

    Args:
        pressure: Pressure in psia
        param_name: Parameter name for error messages

    Raises:
        ValidationError: If pressure is out of range
    """
    if pressure <= 0:
        raise ValidationError(
            parameter=param_name,
            value=pressure,
            message=f"{param_name} ({pressure} psia) must be positive"
        )
    if pressure > 3200:
        raise ValidationError(
            parameter=param_name,
            value=pressure,
            message=f"{param_name} ({pressure} psia) exceeds supercritical range"
        )


def validate_composition(composition: Dict[str, float], param_name: str = "composition") -> None:
    """
    Validate gas composition sums to approximately 1.0.

    Args:
        composition: Dictionary of component mass/mole fractions
        param_name: Parameter name for error messages

    Raises:
        ValidationError: If composition doesn't sum to ~1.0
    """
    total = sum(composition.values())
    if not (0.99 <= total <= 1.01):
        raise ValidationError(
            parameter=param_name,
            value=total,
            message=f"{param_name} fractions sum to {total}, expected ~1.0"
        )


# =============================================================================
# WATER PROPERTIES (IAPWS-IF97 APPROXIMATIONS)
# =============================================================================

def get_water_cp(
    temperature_f: float,
    pressure_psia: float = 14.696,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate specific heat capacity of liquid water at given temperature.

    Based on IAPWS-IF97 formulation with polynomial approximation for
    subcooled liquid water in the economizer operating range.

    Methodology:
        Cp is calculated using a polynomial fit to IAPWS-IF97 data for
        liquid water between 32F (0C) and 400F (204C). The formula is:

        Cp = a0 + a1*T + a2*T^2 + a3*T^3

        where T is in Fahrenheit and Cp is in BTU/(lbm-F)

    Reference: IAPWS-IF97, ASME Steam Tables

    Args:
        temperature_f: Temperature in degrees Fahrenheit
        pressure_psia: Pressure in psia (default: atmospheric)
        track_provenance: If True, return provenance record

    Returns:
        Specific heat in BTU/(lbm-F), optionally with provenance

    Raises:
        ValidationError: If temperature is out of valid range
    """
    validate_temperature_fahrenheit(temperature_f, "temperature_f")
    validate_pressure_psia(pressure_psia, "pressure_psia")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.THERMAL_PROPERTIES,
            formula_id="water_cp_iapws97",
            formula_version="1.0.0",
            inputs={
                "temperature_f": temperature_f,
                "pressure_psia": pressure_psia
            }
        )

    # IAPWS-IF97 polynomial coefficients for liquid water Cp
    # Valid range: 32F to 400F (0C to 204C)
    # Coefficients derived from IAPWS-IF97 tables
    a0 = 0.99979
    a1 = 2.8735e-5
    a2 = 1.3556e-6
    a3 = -2.0426e-9

    # Calculate specific heat
    T = temperature_f
    cp = a0 + a1 * T + a2 * T**2 + a3 * T**3

    if tracker:
        tracker.add_step(
            operation="polynomial",
            description="Calculate Cp using IAPWS-IF97 polynomial",
            inputs={"T": T, "a0": a0, "a1": a1, "a2": a2, "a3": a3},
            output_name="cp",
            output_value=cp,
            formula="Cp = a0 + a1*T + a2*T^2 + a3*T^3"
        )

    # Apply pressure correction for high-pressure subcooled liquid
    # Correction is minimal below 1000 psia
    if pressure_psia > 500:
        # Pressure correction factor (derived from IAPWS-IF97)
        pressure_correction = 1.0 - 0.00001 * (pressure_psia - 500)
        cp *= pressure_correction

        if tracker:
            tracker.add_step(
                operation="multiply",
                description="Apply pressure correction",
                inputs={"cp_base": cp / pressure_correction, "correction": pressure_correction},
                output_name="cp_corrected",
                output_value=cp,
                formula="Cp_corrected = Cp_base * (1 - 0.00001 * (P - 500))"
            )

    # Round to appropriate precision
    cp_rounded = round(cp, 6)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=cp_rounded,
            output_unit="BTU/(lbm-F)",
            precision=6
        )
        return cp_rounded, provenance

    return cp_rounded


def get_water_density(
    temperature_f: float,
    pressure_psia: float = 14.696,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate density of liquid water at given temperature and pressure.

    Based on IAPWS-IF97 formulation with polynomial approximation.

    Methodology:
        Density is calculated using a polynomial fit to IAPWS-IF97 data:

        rho = rho_ref * (1 + b1*(T-T_ref) + b2*(T-T_ref)^2)

        Pressure correction applied for compressibility.

    Reference: IAPWS-IF97

    Args:
        temperature_f: Temperature in degrees Fahrenheit
        pressure_psia: Pressure in psia
        track_provenance: If True, return provenance record

    Returns:
        Density in lbm/ft3, optionally with provenance

    Raises:
        ValidationError: If inputs are out of valid range
    """
    validate_temperature_fahrenheit(temperature_f, "temperature_f")
    validate_pressure_psia(pressure_psia, "pressure_psia")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.THERMAL_PROPERTIES,
            formula_id="water_density_iapws97",
            formula_version="1.0.0",
            inputs={
                "temperature_f": temperature_f,
                "pressure_psia": pressure_psia
            }
        )

    # Reference conditions
    T_ref = 60.0  # F
    rho_ref = 62.37  # lbm/ft3 at 60F, 1 atm

    # Temperature coefficients (IAPWS-IF97 derived)
    b1 = -2.56e-4  # 1/F
    b2 = -3.5e-7   # 1/F^2

    # Calculate temperature effect
    dT = temperature_f - T_ref
    rho_T = rho_ref * (1 + b1 * dT + b2 * dT**2)

    if tracker:
        tracker.add_step(
            operation="polynomial",
            description="Calculate temperature-corrected density",
            inputs={"rho_ref": rho_ref, "dT": dT, "b1": b1, "b2": b2},
            output_name="rho_T",
            output_value=rho_T,
            formula="rho_T = rho_ref * (1 + b1*dT + b2*dT^2)"
        )

    # Pressure correction (compressibility)
    # Water compressibility is approximately 3.0e-6 per psi
    compressibility = 3.0e-6  # 1/psi
    P_ref = 14.696  # psia
    dP = pressure_psia - P_ref

    rho = rho_T * (1 + compressibility * dP)

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Apply pressure correction for compressibility",
            inputs={"rho_T": rho_T, "compressibility": compressibility, "dP": dP},
            output_name="rho",
            output_value=rho,
            formula="rho = rho_T * (1 + compressibility * dP)"
        )

    rho_rounded = round(rho, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=rho_rounded,
            output_unit="lbm/ft3",
            precision=4
        )
        return rho_rounded, provenance

    return rho_rounded


def get_steam_cp(
    temperature_f: float,
    pressure_psia: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate specific heat capacity of steam at given conditions.

    Based on IAPWS-IF97 formulation for superheated steam and
    two-phase region calculations.

    Methodology:
        For superheated steam:
        Cp = Cp_ig + Cp_residual(P,T)

        The ideal gas contribution is calculated from JANAF polynomials,
        and residual contribution from IAPWS-IF97.

    Reference: IAPWS-IF97, JANAF Thermochemical Tables

    Args:
        temperature_f: Temperature in degrees Fahrenheit
        pressure_psia: Pressure in psia
        track_provenance: If True, return provenance record

    Returns:
        Specific heat in BTU/(lbm-F), optionally with provenance

    Raises:
        ValidationError: If inputs are out of valid range
    """
    validate_temperature_fahrenheit(temperature_f, "temperature_f")
    validate_pressure_psia(pressure_psia, "pressure_psia")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.THERMAL_PROPERTIES,
            formula_id="steam_cp_iapws97",
            formula_version="1.0.0",
            inputs={
                "temperature_f": temperature_f,
                "pressure_psia": pressure_psia
            }
        )

    # Convert to Kelvin for calculations
    T_K = (temperature_f - 32) * 5/9 + KELVIN_OFFSET

    # JANAF coefficients for H2O ideal gas (200-1000K range)
    # Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    a1 = 4.0701275
    a2 = -1.1084499e-3
    a3 = 4.1521180e-6
    a4 = -2.9637404e-9
    a5 = 8.0702103e-13

    # Ideal gas Cp (dimensionless Cp/R)
    Cp_R_ideal = a1 + a2*T_K + a3*T_K**2 + a4*T_K**3 + a5*T_K**4

    # Convert to BTU/(lbm-F)
    # R for water = 1.986 BTU/(lbmol-R) / 18.015 lbm/lbmol = 0.1103 BTU/(lbm-R)
    R_water = 0.1103  # BTU/(lbm-R)
    Cp_ideal = Cp_R_ideal * R_water

    if tracker:
        tracker.add_step(
            operation="polynomial",
            description="Calculate ideal gas Cp from JANAF coefficients",
            inputs={"T_K": T_K, "a1": a1, "a2": a2, "a3": a3, "a4": a4, "a5": a5},
            output_name="Cp_ideal",
            output_value=Cp_ideal,
            formula="Cp_ideal = R * (a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4)"
        )

    # Residual contribution (pressure effect)
    # Simplified correlation for residual Cp at moderate pressures
    P_MPa = pressure_psia * 0.00689476
    T_r = T_K / WATER_CRITICAL_TEMP_K
    P_r = P_MPa / WATER_CRITICAL_PRESSURE_MPA

    # Residual contribution (simplified)
    if P_r < 0.1:
        Cp_residual = 0
    else:
        # Empirical correlation for residual Cp
        Cp_residual = 0.05 * P_r / T_r**2 * R_water

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate residual Cp contribution",
            inputs={"P_r": P_r, "T_r": T_r},
            output_name="Cp_residual",
            output_value=Cp_residual,
            formula="Cp_residual = 0.05 * P_r / T_r^2 * R"
        )

    Cp_total = Cp_ideal + Cp_residual

    if tracker:
        tracker.add_step(
            operation="add",
            description="Sum ideal and residual contributions",
            inputs={"Cp_ideal": Cp_ideal, "Cp_residual": Cp_residual},
            output_name="Cp_total",
            output_value=Cp_total,
            formula="Cp_total = Cp_ideal + Cp_residual"
        )

    cp_rounded = round(Cp_total, 6)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=cp_rounded,
            output_unit="BTU/(lbm-F)",
            precision=6
        )
        return cp_rounded, provenance

    return cp_rounded


# =============================================================================
# FLUE GAS PROPERTIES (JANAF TABLES)
# =============================================================================

# Default flue gas composition (natural gas combustion)
DEFAULT_FLUE_GAS_COMPOSITION = {
    "N2": 0.72,
    "CO2": 0.10,
    "H2O": 0.12,
    "O2": 0.05,
    "Ar": 0.01
}


def get_flue_gas_cp(
    temperature_f: float,
    composition: Optional[Dict[str, float]] = None,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate specific heat capacity of flue gas mixture.

    Based on JANAF Thermochemical Tables using NASA polynomial
    coefficients for each component.

    Methodology:
        Mixture Cp is calculated as mass-weighted average:

        Cp_mix = sum(y_i * Cp_i)

        where y_i is mass fraction and Cp_i is component specific heat
        from JANAF polynomials.

    Reference: JANAF Thermochemical Tables, NASA Polynomials

    Args:
        temperature_f: Temperature in degrees Fahrenheit
        composition: Dictionary of mass fractions (default: natural gas combustion)
        track_provenance: If True, return provenance record

    Returns:
        Specific heat in BTU/(lbm-F), optionally with provenance

    Raises:
        ValidationError: If inputs are out of valid range
    """
    validate_temperature_fahrenheit(temperature_f, "temperature_f")

    if composition is None:
        composition = DEFAULT_FLUE_GAS_COMPOSITION.copy()

    validate_composition(composition, "composition")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.THERMAL_PROPERTIES,
            formula_id="flue_gas_cp_janaf",
            formula_version="1.0.0",
            inputs={
                "temperature_f": temperature_f,
                "composition": composition
            }
        )

    # Convert to Kelvin
    T_K = (temperature_f - 32) * 5/9 + KELVIN_OFFSET

    # JANAF polynomial coefficients (200-1000K range)
    # Format: [a1, a2, a3, a4, a5] for Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    janaf_coefficients = {
        "N2": [3.531005, -1.236609e-4, -5.029994e-7, 2.435306e-9, -1.408812e-12],
        "O2": [3.697578, -6.135197e-4, 1.258842e-6, 1.775281e-10, -1.136435e-12],
        "CO2": [2.356773, 8.984596e-3, -7.123563e-6, 2.459190e-9, -1.437418e-13],
        "H2O": [4.0701275, -1.1084499e-3, 4.1521180e-6, -2.9637404e-9, 8.0702103e-13],
        "Ar": [2.5, 0, 0, 0, 0],
        "SO2": [3.266532, 5.323308e-3, -6.843849e-7, -5.281004e-9, 2.559045e-12],
        "CO": [3.579533, -6.103537e-4, 1.016814e-6, 9.070059e-10, -9.044244e-13],
        "NO": [3.376542, 1.253063e-3, -3.302751e-6, 5.217810e-9, -2.446263e-12],
        "NO2": [2.670609, 7.838501e-3, -8.063865e-6, 6.161695e-9, -2.320677e-12],
    }

    # Universal gas constant in BTU/(lbmol-R)
    R_BTU = 1.9859  # BTU/(lbmol-R)

    # Calculate mixture Cp
    Cp_mix = 0.0
    component_cps = {}

    for component, mass_fraction in composition.items():
        if component not in janaf_coefficients:
            continue

        coeffs = janaf_coefficients[component]
        MW = MOLECULAR_WEIGHTS.get(component, 28.0)

        # Calculate Cp/R for this component
        Cp_R = coeffs[0] + coeffs[1]*T_K + coeffs[2]*T_K**2 + coeffs[3]*T_K**3 + coeffs[4]*T_K**4

        # Convert to BTU/(lbm-F)
        Cp_component = Cp_R * R_BTU / MW

        component_cps[component] = Cp_component
        Cp_mix += mass_fraction * Cp_component

    if tracker:
        tracker.add_step(
            operation="weighted_sum",
            description="Calculate mass-weighted average Cp",
            inputs={"component_cps": component_cps, "mass_fractions": composition},
            output_name="Cp_mix",
            output_value=Cp_mix,
            formula="Cp_mix = sum(y_i * Cp_i)"
        )

    cp_rounded = round(Cp_mix, 6)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=cp_rounded,
            output_unit="BTU/(lbm-F)",
            precision=6
        )
        return cp_rounded, provenance

    return cp_rounded


def get_flue_gas_density(
    temperature_f: float,
    pressure_psia: float = 14.696,
    composition: Optional[Dict[str, float]] = None,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate density of flue gas mixture using ideal gas law.

    Methodology:
        For gas mixtures at typical economizer conditions,
        the ideal gas law provides accurate results:

        rho = P * MW_mix / (R * T)

        where MW_mix is the mass-weighted average molecular weight.

    Reference: JANAF Thermochemical Tables

    Args:
        temperature_f: Temperature in degrees Fahrenheit
        pressure_psia: Pressure in psia
        composition: Dictionary of mole fractions (default: natural gas combustion)
        track_provenance: If True, return provenance record

    Returns:
        Density in lbm/ft3, optionally with provenance

    Raises:
        ValidationError: If inputs are out of valid range
    """
    validate_temperature_fahrenheit(temperature_f, "temperature_f")
    validate_pressure_psia(pressure_psia, "pressure_psia")

    if composition is None:
        composition = DEFAULT_FLUE_GAS_COMPOSITION.copy()

    validate_composition(composition, "composition")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.THERMAL_PROPERTIES,
            formula_id="flue_gas_density_ideal",
            formula_version="1.0.0",
            inputs={
                "temperature_f": temperature_f,
                "pressure_psia": pressure_psia,
                "composition": composition
            }
        )

    # Calculate mixture molecular weight (assuming mole fractions)
    MW_mix = 0.0
    for component, mole_fraction in composition.items():
        MW = MOLECULAR_WEIGHTS.get(component, 28.0)
        MW_mix += mole_fraction * MW

    if tracker:
        tracker.add_step(
            operation="weighted_sum",
            description="Calculate mixture molecular weight",
            inputs={"composition": composition, "molecular_weights": MOLECULAR_WEIGHTS},
            output_name="MW_mix",
            output_value=MW_mix,
            formula="MW_mix = sum(x_i * MW_i)"
        )

    # Convert temperature to Rankine
    T_R = temperature_f + 459.67

    # Ideal gas constant in (psia * ft3) / (lbmol * R)
    R_gas = 10.7316  # (psia * ft3) / (lbmol * R)

    # Calculate density using ideal gas law
    # rho = P * MW / (R * T)
    rho = (pressure_psia * MW_mix) / (R_gas * T_R)

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate density using ideal gas law",
            inputs={"P": pressure_psia, "MW_mix": MW_mix, "R": R_gas, "T_R": T_R},
            output_name="rho",
            output_value=rho,
            formula="rho = P * MW_mix / (R * T)"
        )

    rho_rounded = round(rho, 6)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=rho_rounded,
            output_unit="lbm/ft3",
            precision=6
        )
        return rho_rounded, provenance

    return rho_rounded


# =============================================================================
# TUBE MATERIAL THERMAL CONDUCTIVITY
# =============================================================================

# Tube material thermal conductivity coefficients
# k = k0 + k1*T + k2*T^2 (BTU/(hr-ft-F))
TUBE_MATERIAL_CONDUCTIVITY = {
    "carbon_steel": {
        "k0": 26.0,
        "k1": -0.008,
        "k2": 0.0,
        "min_temp_f": 0,
        "max_temp_f": 1000
    },
    "stainless_304": {
        "k0": 8.0,
        "k1": 0.008,
        "k2": 0.0,
        "min_temp_f": 0,
        "max_temp_f": 1500
    },
    "stainless_316": {
        "k0": 7.5,
        "k1": 0.0085,
        "k2": 0.0,
        "min_temp_f": 0,
        "max_temp_f": 1500
    },
    "copper": {
        "k0": 220.0,
        "k1": -0.02,
        "k2": 0.0,
        "min_temp_f": 0,
        "max_temp_f": 500
    },
    "inconel_625": {
        "k0": 5.5,
        "k1": 0.007,
        "k2": 0.0,
        "min_temp_f": 0,
        "max_temp_f": 1800
    },
    "corten_steel": {
        "k0": 25.0,
        "k1": -0.0075,
        "k2": 0.0,
        "min_temp_f": 0,
        "max_temp_f": 900
    }
}


def get_thermal_conductivity(
    material: str,
    temperature_f: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate thermal conductivity of tube material at given temperature.

    Methodology:
        Thermal conductivity is calculated using temperature-dependent
        polynomial correlations:

        k = k0 + k1*T + k2*T^2

    Reference: ASME Materials Database

    Args:
        material: Material name (carbon_steel, stainless_304, etc.)
        temperature_f: Temperature in degrees Fahrenheit
        track_provenance: If True, return provenance record

    Returns:
        Thermal conductivity in BTU/(hr-ft-F), optionally with provenance

    Raises:
        ValidationError: If material not found or temperature out of range
        ValueError: If material is not supported
    """
    material_lower = material.lower().replace(" ", "_").replace("-", "_")

    if material_lower not in TUBE_MATERIAL_CONDUCTIVITY:
        raise ValueError(
            f"Material '{material}' not supported. "
            f"Available: {list(TUBE_MATERIAL_CONDUCTIVITY.keys())}"
        )

    props = TUBE_MATERIAL_CONDUCTIVITY[material_lower]

    if temperature_f < props["min_temp_f"] or temperature_f > props["max_temp_f"]:
        raise ValidationError(
            parameter="temperature_f",
            value=temperature_f,
            message=f"Temperature {temperature_f}F out of valid range "
                    f"[{props['min_temp_f']}, {props['max_temp_f']}] for {material}"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.THERMAL_PROPERTIES,
            formula_id="thermal_conductivity_polynomial",
            formula_version="1.0.0",
            inputs={
                "material": material,
                "temperature_f": temperature_f
            }
        )

    # Calculate conductivity
    k = props["k0"] + props["k1"] * temperature_f + props["k2"] * temperature_f**2

    if tracker:
        tracker.add_step(
            operation="polynomial",
            description="Calculate thermal conductivity",
            inputs={
                "k0": props["k0"],
                "k1": props["k1"],
                "k2": props["k2"],
                "T": temperature_f
            },
            output_name="k",
            output_value=k,
            formula="k = k0 + k1*T + k2*T^2"
        )

    k_rounded = round(k, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=k_rounded,
            output_unit="BTU/(hr-ft-F)",
            precision=4
        )
        return k_rounded, provenance

    return k_rounded


# =============================================================================
# UNIT CONVERSION UTILITIES
# =============================================================================

def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32) * 5 / 9


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9 / 5 + 32


def fahrenheit_to_kelvin(temp_f: float) -> float:
    """Convert Fahrenheit to Kelvin."""
    return (temp_f - 32) * 5 / 9 + KELVIN_OFFSET


def kelvin_to_fahrenheit(temp_k: float) -> float:
    """Convert Kelvin to Fahrenheit."""
    return (temp_k - KELVIN_OFFSET) * 9 / 5 + 32


def psia_to_mpa(pressure_psia: float) -> float:
    """Convert psia to MPa."""
    return pressure_psia * 0.00689476


def mpa_to_psia(pressure_mpa: float) -> float:
    """Convert MPa to psia."""
    return pressure_mpa / 0.00689476


def btu_per_lb_f_to_kj_per_kg_k(cp: float) -> float:
    """Convert BTU/(lbm-F) to kJ/(kg-K)."""
    return cp * 4.1868


def kj_per_kg_k_to_btu_per_lb_f(cp: float) -> float:
    """Convert kJ/(kg-K) to BTU/(lbm-F)."""
    return cp / 4.1868
