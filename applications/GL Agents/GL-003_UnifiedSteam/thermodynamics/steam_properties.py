"""
Steam Properties Calculator - High-Level Property Interface

This module provides a high-level interface for computing steam properties
using the IAPWS-IF97 standard. All calculations are deterministic with
complete provenance tracking for regulatory compliance.

Supported Input Modes:
- (P, T): Pressure and Temperature
- (P, h): Pressure and Enthalpy
- (P, x): Pressure and Quality (dryness fraction)
- (T, x): Temperature and Quality (dryness fraction)

Author: GL-CalculatorEngineer
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import hashlib
import json

from .iapws_if97 import (
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    detect_region,
    get_saturation_pressure,
    get_saturation_temperature,
    region1_specific_volume,
    region1_specific_enthalpy,
    region1_specific_entropy,
    region1_specific_internal_energy,
    region1_specific_isobaric_heat_capacity,
    region1_speed_of_sound,
    region2_specific_volume,
    region2_specific_enthalpy,
    region2_specific_entropy,
    region2_specific_internal_energy,
    region2_specific_isobaric_heat_capacity,
    region2_speed_of_sound,
    region4_saturation_properties,
    region4_mixture_enthalpy,
    region4_mixture_entropy,
    region4_mixture_specific_volume,
    compute_property_derivatives,
    compute_calculation_provenance,
    celsius_to_kelvin,
    kelvin_to_celsius,
    kpa_to_mpa,
    mpa_to_kpa,
    compute_density,
)


class SteamState(Enum):
    """Steam state classification."""
    COMPRESSED_LIQUID = "compressed_liquid"  # Subcooled water (T < Tsat)
    SATURATED_LIQUID = "saturated_liquid"    # x = 0
    WET_STEAM = "wet_steam"                  # 0 < x < 1
    SATURATED_VAPOR = "saturated_vapor"      # x = 1
    SUPERHEATED_VAPOR = "superheated_vapor"  # T > Tsat


@dataclass
class SteamProperties:
    """
    Complete steam properties at a given state point.

    All properties include units in the attribute name for clarity.
    """
    # State point identification
    pressure_kpa: float
    temperature_c: float
    region: int
    state: SteamState

    # Thermodynamic properties
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kgk: float
    specific_volume_m3_kg: float
    density_kg_m3: float
    specific_internal_energy_kj_kg: float

    # Additional properties
    specific_heat_cp_kj_kgk: Optional[float] = None
    speed_of_sound_m_s: Optional[float] = None

    # Two-phase properties (only for wet steam)
    quality_x: Optional[float] = None
    superheat_degree_c: Optional[float] = None

    # Saturation reference (for superheated steam)
    saturation_temperature_c: Optional[float] = None

    # Provenance tracking
    provenance_hash: str = ""
    calculation_timestamp: str = ""

    # Derivatives for optimization (optional)
    derivatives: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class SaturationProperties:
    """
    Saturation properties at a given pressure or temperature.
    """
    # State point
    pressure_kpa: float
    temperature_c: float

    # Saturated liquid properties (subscript f)
    hf_kj_kg: float              # Enthalpy of saturated liquid
    sf_kj_kgk: float             # Entropy of saturated liquid
    vf_m3_kg: float              # Specific volume of saturated liquid
    rho_f_kg_m3: float           # Density of saturated liquid

    # Saturated vapor properties (subscript g)
    hg_kj_kg: float              # Enthalpy of saturated vapor
    sg_kj_kgk: float             # Entropy of saturated vapor
    vg_m3_kg: float              # Specific volume of saturated vapor
    rho_g_kg_m3: float           # Density of saturated vapor

    # Latent heat properties (subscript fg)
    hfg_kj_kg: float             # Latent heat of vaporization
    sfg_kj_kgk: float            # Entropy of vaporization

    # Provenance
    provenance_hash: str = ""


@dataclass
class InputValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_input_ranges(
    pressure_kpa: Optional[float] = None,
    temperature_c: Optional[float] = None,
    quality_x: Optional[float] = None,
    enthalpy_kj_kg: Optional[float] = None,
) -> InputValidationResult:
    """
    Validate input parameters against IAPWS-IF97 valid ranges.

    DETERMINISTIC: Same inputs always produce same validation result.

    Args:
        pressure_kpa: Pressure in kPa (optional)
        temperature_c: Temperature in Celsius (optional)
        quality_x: Steam quality / dryness fraction (optional)
        enthalpy_kj_kg: Specific enthalpy in kJ/kg (optional)

    Returns:
        InputValidationResult with validation status and any errors/warnings
    """
    errors = []
    warnings = []

    # Pressure validation
    if pressure_kpa is not None:
        P_mpa = kpa_to_mpa(pressure_kpa)
        P_MIN = REGION_BOUNDARIES["P_MIN"]
        P_MAX = REGION_BOUNDARIES["P_MAX_1_2"]

        if P_mpa < P_MIN:
            errors.append(
                f"Pressure {pressure_kpa:.2f} kPa is below minimum "
                f"{P_MIN * 1000:.4f} kPa (triple point pressure)"
            )
        elif P_mpa > P_MAX:
            errors.append(
                f"Pressure {pressure_kpa:.2f} kPa exceeds maximum "
                f"{P_MAX * 1000:.0f} kPa"
            )

        # Warning for near-critical pressure
        P_CRIT = IF97_CONSTANTS["P_CRIT"]
        if abs(P_mpa - P_CRIT) < 0.1:
            warnings.append(
                f"Pressure {pressure_kpa:.2f} kPa is near critical pressure. "
                "Calculations may have reduced accuracy."
            )

    # Temperature validation
    if temperature_c is not None:
        T_k = celsius_to_kelvin(temperature_c)
        T_MIN = REGION_BOUNDARIES["T_MIN"]
        T_MAX = REGION_BOUNDARIES["T_MAX_2"]

        if T_k < T_MIN:
            errors.append(
                f"Temperature {temperature_c:.2f} C is below minimum "
                f"{T_MIN - 273.15:.2f} C (0 C / freezing point)"
            )
        elif T_k > T_MAX:
            errors.append(
                f"Temperature {temperature_c:.2f} C exceeds maximum "
                f"{T_MAX - 273.15:.0f} C for standard regions"
            )

        # Warning for near-critical temperature
        T_CRIT = IF97_CONSTANTS["T_CRIT"]
        if abs(T_k - T_CRIT) < 5:
            warnings.append(
                f"Temperature {temperature_c:.2f} C is near critical temperature. "
                "Calculations may have reduced accuracy."
            )

    # Quality validation
    if quality_x is not None:
        if quality_x < 0:
            errors.append(
                f"Quality {quality_x:.4f} is negative. "
                "Quality must be between 0 and 1."
            )
        elif quality_x > 1:
            errors.append(
                f"Quality {quality_x:.4f} exceeds 1. "
                "Quality must be between 0 and 1."
            )
        elif quality_x < 0.05:
            warnings.append(
                f"Quality {quality_x:.4f} is very low (nearly all liquid). "
                "Consider checking for subcooled liquid state."
            )
        elif quality_x > 0.95:
            warnings.append(
                f"Quality {quality_x:.4f} is very high (nearly all vapor). "
                "Consider checking for superheated vapor state."
            )

    # Enthalpy validation (basic range check)
    if enthalpy_kj_kg is not None:
        if enthalpy_kj_kg < 0:
            warnings.append(
                f"Enthalpy {enthalpy_kj_kg:.2f} kJ/kg is negative. "
                "This is unusual for steam systems."
            )
        elif enthalpy_kj_kg > 4500:
            warnings.append(
                f"Enthalpy {enthalpy_kj_kg:.2f} kJ/kg is very high. "
                "This indicates very high temperature steam."
            )

    return InputValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# =============================================================================
# MAIN PROPERTY CALCULATION FUNCTIONS
# =============================================================================

def compute_properties(
    pressure_kpa: float,
    temperature_c: Optional[float] = None,
    quality_x: Optional[float] = None,
    enthalpy_kj_kg: Optional[float] = None,
    compute_derivatives: bool = False,
) -> SteamProperties:
    """
    Compute steam properties at the specified state point.

    DETERMINISTIC: Same inputs always produce same output with identical
    provenance hash.

    Supported input combinations:
    - (P, T): pressure_kpa + temperature_c
    - (P, x): pressure_kpa + quality_x (two-phase region)
    - (P, h): pressure_kpa + enthalpy_kj_kg (iterative solution)

    Args:
        pressure_kpa: Pressure in kPa (required)
        temperature_c: Temperature in Celsius (optional)
        quality_x: Steam quality / dryness fraction (optional)
        enthalpy_kj_kg: Specific enthalpy in kJ/kg (optional)
        compute_derivatives: Whether to compute property derivatives

    Returns:
        SteamProperties with all calculated properties

    Raises:
        ValueError: If inputs are invalid or outside valid range
        ValueError: If insufficient inputs provided
    """
    from datetime import datetime

    # Validate inputs
    validation = validate_input_ranges(
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        quality_x=quality_x,
        enthalpy_kj_kg=enthalpy_kj_kg,
    )

    if not validation.is_valid:
        raise ValueError(f"Invalid inputs: {'; '.join(validation.errors)}")

    # Convert units
    P_mpa = kpa_to_mpa(pressure_kpa)

    # Determine calculation mode
    if temperature_c is not None:
        # Mode: (P, T) - Direct calculation
        T_k = celsius_to_kelvin(temperature_c)
        return _compute_from_pt(
            P_mpa, T_k, pressure_kpa, temperature_c, compute_derivatives
        )

    elif quality_x is not None:
        # Mode: (P, x) - Two-phase calculation
        return _compute_from_px(P_mpa, quality_x, pressure_kpa, compute_derivatives)

    elif enthalpy_kj_kg is not None:
        # Mode: (P, h) - Iterative solution
        return _compute_from_ph(
            P_mpa, enthalpy_kj_kg, pressure_kpa, compute_derivatives
        )

    else:
        raise ValueError(
            "Insufficient inputs. Provide one of: "
            "temperature_c, quality_x, or enthalpy_kj_kg"
        )


def _compute_from_pt(
    P_mpa: float,
    T_k: float,
    pressure_kpa: float,
    temperature_c: float,
    compute_derivatives: bool,
) -> SteamProperties:
    """
    Compute properties from pressure and temperature.

    DETERMINISTIC: Same inputs always produce same output.
    """
    from datetime import datetime

    # Detect region
    region = detect_region(P_mpa, T_k)

    # Get saturation temperature
    try:
        T_sat_k = get_saturation_temperature(P_mpa)
        T_sat_c = kelvin_to_celsius(T_sat_k)
    except ValueError:
        # Above critical pressure
        T_sat_k = None
        T_sat_c = None

    # Determine state
    if T_sat_k is not None:
        if abs(T_k - T_sat_k) < 0.001:
            # At saturation - need quality to fully define state
            state = SteamState.SATURATED_VAPOR  # Assume saturated vapor
        elif T_k < T_sat_k:
            state = SteamState.COMPRESSED_LIQUID
        else:
            state = SteamState.SUPERHEATED_VAPOR
    else:
        state = SteamState.SUPERHEATED_VAPOR  # Supercritical

    # Calculate properties based on region
    if region == 1:
        # Compressed liquid
        h = region1_specific_enthalpy(P_mpa, T_k)
        s = region1_specific_entropy(P_mpa, T_k)
        v = region1_specific_volume(P_mpa, T_k)
        u = region1_specific_internal_energy(P_mpa, T_k)
        cp = region1_specific_isobaric_heat_capacity(P_mpa, T_k)
        w = region1_speed_of_sound(P_mpa, T_k)

    elif region == 2:
        # Superheated vapor
        h = region2_specific_enthalpy(P_mpa, T_k)
        s = region2_specific_entropy(P_mpa, T_k)
        v = region2_specific_volume(P_mpa, T_k)
        u = region2_specific_internal_energy(P_mpa, T_k)
        cp = region2_specific_isobaric_heat_capacity(P_mpa, T_k)
        w = region2_speed_of_sound(P_mpa, T_k)

    else:
        raise ValueError(f"Region {region} not supported for (P,T) calculation")

    # Calculate density
    rho = compute_density(v)

    # Calculate superheat degree (only for superheated vapor)
    superheat = None
    if state == SteamState.SUPERHEATED_VAPOR and T_sat_c is not None:
        superheat = temperature_c - T_sat_c

    # Compute derivatives if requested
    derivatives = None
    if compute_derivatives and region in [1, 2]:
        derivatives = {
            "h": compute_property_derivatives(P_mpa, T_k, "h"),
            "s": compute_property_derivatives(P_mpa, T_k, "s"),
            "v": compute_property_derivatives(P_mpa, T_k, "v"),
        }

    # Create provenance hash
    inputs = {
        "pressure_kpa": pressure_kpa,
        "temperature_c": temperature_c,
        "region": region,
    }
    outputs = {
        "h": h,
        "s": s,
        "v": v,
        "u": u,
        "rho": rho,
    }
    provenance_hash = compute_calculation_provenance(inputs, outputs)

    return SteamProperties(
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        region=region,
        state=state,
        specific_enthalpy_kj_kg=h,
        specific_entropy_kj_kgk=s,
        specific_volume_m3_kg=v,
        density_kg_m3=rho,
        specific_internal_energy_kj_kg=u,
        specific_heat_cp_kj_kgk=cp,
        speed_of_sound_m_s=w,
        quality_x=None,
        superheat_degree_c=superheat,
        saturation_temperature_c=T_sat_c,
        provenance_hash=provenance_hash,
        calculation_timestamp=datetime.utcnow().isoformat(),
        derivatives=derivatives,
    )


def _compute_from_px(
    P_mpa: float,
    quality_x: float,
    pressure_kpa: float,
    compute_derivatives: bool,
) -> SteamProperties:
    """
    Compute properties from pressure and quality (two-phase region).

    DETERMINISTIC: Same inputs always produce same output.
    """
    from datetime import datetime

    # Validate quality
    if quality_x < 0 or quality_x > 1:
        raise ValueError(f"Quality must be between 0 and 1, got {quality_x}")

    # Get saturation properties
    sat = region4_saturation_properties(P_mpa)
    T_sat_c = kelvin_to_celsius(sat.temperature_k)

    # Determine state
    if quality_x == 0:
        state = SteamState.SATURATED_LIQUID
    elif quality_x == 1:
        state = SteamState.SATURATED_VAPOR
    else:
        state = SteamState.WET_STEAM

    # Calculate mixture properties
    h = region4_mixture_enthalpy(P_mpa, quality_x)
    s = region4_mixture_entropy(P_mpa, quality_x)
    v = region4_mixture_specific_volume(P_mpa, quality_x)
    rho = compute_density(v)

    # Internal energy: u = h - P*v
    u = h - P_mpa * 1000 * v  # P in kPa, v in m3/kg, u in kJ/kg

    # Create provenance hash
    inputs = {
        "pressure_kpa": pressure_kpa,
        "quality_x": quality_x,
        "region": 4,
    }
    outputs = {
        "h": h,
        "s": s,
        "v": v,
        "u": u,
        "rho": rho,
    }
    provenance_hash = compute_calculation_provenance(inputs, outputs)

    return SteamProperties(
        pressure_kpa=pressure_kpa,
        temperature_c=T_sat_c,
        region=4,
        state=state,
        specific_enthalpy_kj_kg=h,
        specific_entropy_kj_kgk=s,
        specific_volume_m3_kg=v,
        density_kg_m3=rho,
        specific_internal_energy_kj_kg=u,
        specific_heat_cp_kj_kgk=None,  # Not defined for two-phase
        speed_of_sound_m_s=None,       # Not defined for two-phase
        quality_x=quality_x,
        superheat_degree_c=0.0,
        saturation_temperature_c=T_sat_c,
        provenance_hash=provenance_hash,
        calculation_timestamp=datetime.utcnow().isoformat(),
        derivatives=None,
    )


def _compute_from_ph(
    P_mpa: float,
    enthalpy_kj_kg: float,
    pressure_kpa: float,
    compute_derivatives: bool,
) -> SteamProperties:
    """
    Compute properties from pressure and enthalpy (iterative solution).

    DETERMINISTIC: Same inputs always produce same output.
    """
    from datetime import datetime

    # Get saturation properties at this pressure
    try:
        sat = region4_saturation_properties(P_mpa)
        hf = sat.hf
        hg = sat.hg
        T_sat_k = sat.temperature_k
        T_sat_c = kelvin_to_celsius(T_sat_k)

        # Determine region based on enthalpy
        if enthalpy_kj_kg < hf - 0.01:
            # Compressed liquid region
            region = 1
            state = SteamState.COMPRESSED_LIQUID
        elif enthalpy_kj_kg <= hg + 0.01:
            # Two-phase region
            region = 4
            quality_x = (enthalpy_kj_kg - hf) / (hg - hf)
            quality_x = max(0, min(1, quality_x))  # Clamp to [0, 1]
            return _compute_from_px(P_mpa, quality_x, pressure_kpa, compute_derivatives)
        else:
            # Superheated region
            region = 2
            state = SteamState.SUPERHEATED_VAPOR

    except ValueError:
        # Above critical pressure - only superheated region
        region = 2
        state = SteamState.SUPERHEATED_VAPOR
        T_sat_c = None

    # Iterative solution for temperature
    if region == 1:
        T_k = _solve_temperature_from_enthalpy_region1(P_mpa, enthalpy_kj_kg)
    else:  # region == 2
        T_k = _solve_temperature_from_enthalpy_region2(P_mpa, enthalpy_kj_kg)

    temperature_c = kelvin_to_celsius(T_k)

    # Now compute all properties at (P, T)
    return _compute_from_pt(
        P_mpa, T_k, pressure_kpa, temperature_c, compute_derivatives
    )


def _solve_temperature_from_enthalpy_region1(
    P_mpa: float,
    target_h: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """
    Solve for temperature given pressure and enthalpy in Region 1.

    Uses Newton-Raphson iteration. DETERMINISTIC.
    """
    # Initial guess: saturation temperature
    try:
        T_k = get_saturation_temperature(P_mpa)
    except ValueError:
        T_k = 400.0  # Default initial guess

    for _ in range(max_iter):
        h = region1_specific_enthalpy(P_mpa, T_k)
        error = h - target_h

        if abs(error) < tol:
            return T_k

        # Numerical derivative
        dT = 0.01
        h_plus = region1_specific_enthalpy(P_mpa, T_k + dT)
        dh_dT = (h_plus - h) / dT

        if abs(dh_dT) < 1e-10:
            raise ValueError("Derivative too small in Newton-Raphson iteration")

        T_k = T_k - error / dh_dT

    raise ValueError(
        f"Failed to converge in Region 1 enthalpy solution after {max_iter} iterations"
    )


def _solve_temperature_from_enthalpy_region2(
    P_mpa: float,
    target_h: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """
    Solve for temperature given pressure and enthalpy in Region 2.

    Uses Newton-Raphson iteration. DETERMINISTIC.
    """
    # Initial guess: slightly above saturation temperature
    try:
        T_sat_k = get_saturation_temperature(P_mpa)
        T_k = T_sat_k + 50  # Start 50 K above saturation
    except ValueError:
        T_k = 600.0  # Default initial guess

    for _ in range(max_iter):
        try:
            h = region2_specific_enthalpy(P_mpa, T_k)
        except ValueError:
            # Out of range - adjust guess
            T_k = min(T_k, 800 + 273.15)
            continue

        error = h - target_h

        if abs(error) < tol:
            return T_k

        # Numerical derivative
        dT = 0.1
        try:
            h_plus = region2_specific_enthalpy(P_mpa, T_k + dT)
            dh_dT = (h_plus - h) / dT
        except ValueError:
            dh_dT = 2.0  # Approximate Cp for steam

        if abs(dh_dT) < 1e-10:
            raise ValueError("Derivative too small in Newton-Raphson iteration")

        T_k = T_k - error / dh_dT

        # Ensure temperature stays in valid range
        T_k = max(T_k, REGION_BOUNDARIES["T_MIN"])
        T_k = min(T_k, REGION_BOUNDARIES["T_MAX_2"])

    raise ValueError(
        f"Failed to converge in Region 2 enthalpy solution after {max_iter} iterations"
    )


# =============================================================================
# SATURATION PROPERTIES
# =============================================================================

def get_saturation_properties(
    pressure_kpa: Optional[float] = None,
    temperature_c: Optional[float] = None,
) -> SaturationProperties:
    """
    Get saturation properties at given pressure or temperature.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_kpa: Saturation pressure in kPa (optional)
        temperature_c: Saturation temperature in Celsius (optional)

    Returns:
        SaturationProperties with all saturation state data

    Raises:
        ValueError: If neither or both arguments provided
        ValueError: If values are outside saturation range
    """
    if (pressure_kpa is None) == (temperature_c is None):
        raise ValueError("Provide exactly one of pressure_kpa or temperature_c")

    if pressure_kpa is not None:
        P_mpa = kpa_to_mpa(pressure_kpa)
        T_sat_k = get_saturation_temperature(P_mpa)
        T_sat_c = kelvin_to_celsius(T_sat_k)
    else:
        T_sat_k = celsius_to_kelvin(temperature_c)
        P_mpa = get_saturation_pressure(T_sat_k)
        pressure_kpa = mpa_to_kpa(P_mpa)
        T_sat_c = temperature_c

    # Get saturation properties from IF97
    sat = region4_saturation_properties(P_mpa)

    # Create provenance hash
    inputs = {
        "pressure_kpa": pressure_kpa,
        "temperature_c": T_sat_c,
    }
    outputs = {
        "hf": sat.hf,
        "hg": sat.hg,
        "sf": sat.sf,
        "sg": sat.sg,
        "vf": sat.vf,
        "vg": sat.vg,
    }
    provenance_hash = compute_calculation_provenance(inputs, outputs)

    return SaturationProperties(
        pressure_kpa=pressure_kpa,
        temperature_c=T_sat_c,
        hf_kj_kg=sat.hf,
        sf_kj_kgk=sat.sf,
        vf_m3_kg=sat.vf,
        rho_f_kg_m3=compute_density(sat.vf),
        hg_kj_kg=sat.hg,
        sg_kj_kgk=sat.sg,
        vg_m3_kg=sat.vg,
        rho_g_kg_m3=compute_density(sat.vg),
        hfg_kj_kg=sat.hfg,
        sfg_kj_kgk=sat.sfg,
        provenance_hash=provenance_hash,
    )


# =============================================================================
# STATE DETECTION AND UTILITY FUNCTIONS
# =============================================================================

def detect_steam_state(
    pressure_kpa: float,
    temperature_c: float,
    tolerance_c: float = 0.5,
) -> SteamState:
    """
    Detect the steam state based on pressure and temperature.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius
        tolerance_c: Temperature tolerance for saturation detection

    Returns:
        SteamState enumeration value

    Raises:
        ValueError: If inputs are outside valid range
    """
    # Validate inputs
    validation = validate_input_ranges(
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
    )
    if not validation.is_valid:
        raise ValueError(f"Invalid inputs: {'; '.join(validation.errors)}")

    P_mpa = kpa_to_mpa(pressure_kpa)

    try:
        T_sat_k = get_saturation_temperature(P_mpa)
        T_sat_c = kelvin_to_celsius(T_sat_k)

        if temperature_c < T_sat_c - tolerance_c:
            return SteamState.COMPRESSED_LIQUID
        elif abs(temperature_c - T_sat_c) <= tolerance_c:
            return SteamState.WET_STEAM  # At saturation - could be two-phase
        else:
            return SteamState.SUPERHEATED_VAPOR

    except ValueError:
        # Above critical pressure - supercritical fluid
        return SteamState.SUPERHEATED_VAPOR


def compute_superheat_degree(
    pressure_kpa: float,
    temperature_c: float,
) -> float:
    """
    Compute the degree of superheat above saturation temperature.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Actual temperature in Celsius

    Returns:
        Superheat degree in Celsius (0 or negative if not superheated)

    Raises:
        ValueError: If inputs are outside valid range
    """
    P_mpa = kpa_to_mpa(pressure_kpa)

    try:
        T_sat_k = get_saturation_temperature(P_mpa)
        T_sat_c = kelvin_to_celsius(T_sat_k)
        return temperature_c - T_sat_c
    except ValueError:
        # Above critical pressure - no saturation point
        return float('nan')


def compute_dryness_fraction(
    pressure_kpa: float,
    enthalpy_kj_kg: float,
) -> float:
    """
    Compute the dryness fraction (quality) from pressure and enthalpy.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_kpa: Pressure in kPa
        enthalpy_kj_kg: Specific enthalpy in kJ/kg

    Returns:
        Dryness fraction x (0 = saturated liquid, 1 = saturated vapor)
        Returns < 0 if subcooled, > 1 if superheated

    Raises:
        ValueError: If pressure is outside valid range
    """
    P_mpa = kpa_to_mpa(pressure_kpa)

    try:
        sat = region4_saturation_properties(P_mpa)
        hf = sat.hf
        hg = sat.hg
        hfg = sat.hfg

        if abs(hfg) < 1e-10:
            raise ValueError("Near critical point - latent heat approaches zero")

        # x = (h - hf) / hfg
        x = (enthalpy_kj_kg - hf) / hfg

        return x

    except ValueError as e:
        raise ValueError(f"Cannot compute dryness fraction: {e}")
