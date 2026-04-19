"""
Steam Properties Module for GL-012_SteamQual

This module provides steam property wrapper functions for the SteamQual agent,
implementing IAPWS-IF97 calculations with a focus on steam quality monitoring
and control applications.

Key Functions:
- get_saturation_temperature(P): Saturation temperature at pressure
- get_saturation_pressure(T): Saturation pressure at temperature
- get_saturation_properties(P): Complete saturation properties tuple
- compute_superheat_margin(P, T): Degrees above saturation
- determine_steam_state(P, T): Classify steam state

All functions are DETERMINISTIC with complete provenance tracking for
zero-hallucination compliance.

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum
import hashlib
import json
import logging
from datetime import datetime

from .iapws_wrapper import (
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    celsius_to_kelvin,
    kelvin_to_celsius,
    kpa_to_mpa,
    mpa_to_kpa,
    compute_density,
    get_saturation_pressure as iapws_get_saturation_pressure,
    get_saturation_temperature as iapws_get_saturation_temperature,
    get_saturation_properties as iapws_get_saturation_properties,
    region1_specific_enthalpy,
    region1_specific_entropy,
    region1_specific_volume,
    region1_specific_isobaric_heat_capacity,
    region2_specific_enthalpy,
    region2_specific_entropy,
    region2_specific_volume,
    region2_specific_isobaric_heat_capacity,
    region4_mixture_enthalpy,
    region4_mixture_entropy,
    region4_mixture_specific_volume,
    detect_region,
    compute_provenance_hash,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SteamState(Enum):
    """Steam state classification for process control."""
    SUBCOOLED = "subcooled"           # Compressed liquid (T < Tsat)
    SATURATED = "saturated"           # At saturation line (T = Tsat)
    WET_STEAM = "wet_steam"           # Two-phase mixture (0 < x < 1)
    SUPERHEATED = "superheated"       # Superheated vapor (T > Tsat)
    SUPERCRITICAL = "supercritical"   # Above critical point


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SaturationPropertiesTuple:
    """
    Complete saturation properties at a given pressure.

    Attributes:
        pressure_kpa: Saturation pressure in kPa
        temperature_c: Saturation temperature in Celsius
        hf: Saturated liquid specific enthalpy [kJ/kg]
        hg: Saturated vapor specific enthalpy [kJ/kg]
        hfg: Latent heat of vaporization [kJ/kg]
        sf: Saturated liquid specific entropy [kJ/(kg*K)]
        sg: Saturated vapor specific entropy [kJ/(kg*K)]
        sfg: Entropy of vaporization [kJ/(kg*K)]
        vf: Saturated liquid specific volume [m^3/kg]
        vg: Saturated vapor specific volume [m^3/kg]
        provenance_hash: SHA-256 hash for audit trail
    """
    pressure_kpa: float
    temperature_c: float
    hf: float
    hg: float
    hfg: float
    sf: float
    sg: float
    sfg: float
    vf: float
    vg: float
    provenance_hash: str = ""

    def as_tuple(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """Return properties as tuple (hf, hg, hfg, sf, sg, sfg, vf, vg)."""
        return (self.hf, self.hg, self.hfg, self.sf, self.sg, self.sfg, self.vf, self.vg)


@dataclass
class SteamStateResult:
    """
    Result of steam state determination.

    Attributes:
        state: The determined steam state
        pressure_kpa: Input pressure
        temperature_c: Input temperature
        saturation_temperature_c: Saturation temperature at pressure
        superheat_margin_c: Degrees above saturation (negative if subcooled)
        confidence: Confidence level in determination (0-1)
        provenance_hash: SHA-256 hash for audit trail
    """
    state: SteamState
    pressure_kpa: float
    temperature_c: float
    saturation_temperature_c: float
    superheat_margin_c: float
    confidence: float
    provenance_hash: str = ""


@dataclass
class SuperheatMarginResult:
    """
    Result of superheat margin calculation.

    Attributes:
        margin_c: Superheat margin in Celsius (T - Tsat)
        pressure_kpa: Input pressure
        temperature_c: Input temperature
        saturation_temperature_c: Calculated saturation temperature
        is_superheated: True if temperature > saturation temperature
        is_critical: True if near critical conditions
        provenance_hash: SHA-256 hash for audit trail
    """
    margin_c: float
    pressure_kpa: float
    temperature_c: float
    saturation_temperature_c: float
    is_superheated: bool
    is_critical: bool
    provenance_hash: str = ""


@dataclass
class SteamPropertiesResult:
    """
    Complete steam properties at a given state point.
    """
    # Input conditions
    pressure_kpa: float
    temperature_c: float
    region: int
    state: SteamState

    # Thermodynamic properties
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kgk: float
    specific_volume_m3_kg: float
    density_kg_m3: float

    # Optional properties
    specific_heat_cp_kj_kgk: Optional[float] = None
    quality_x: Optional[float] = None
    superheat_margin_c: Optional[float] = None

    # Provenance
    provenance_hash: str = ""
    calculation_timestamp: str = ""


# =============================================================================
# PRIMARY FUNCTIONS
# =============================================================================

def get_saturation_temperature(pressure_kpa: float) -> float:
    """
    Get saturation temperature at given pressure.

    DETERMINISTIC: Same input always produces same output.

    Uses IAPWS-IF97 backward equation for Region 4 to compute
    the saturation temperature corresponding to a given pressure.

    Args:
        pressure_kpa: Pressure in kPa (0.6117 to 22064 kPa)

    Returns:
        Saturation temperature in Celsius

    Raises:
        ValueError: If pressure is outside valid saturation range

    Example:
        >>> T_sat = get_saturation_temperature(100.0)  # 1 bar
        >>> print(f"Tsat at 1 bar: {T_sat:.2f} C")
        Tsat at 1 bar: 99.61 C
    """
    P_mpa = kpa_to_mpa(pressure_kpa)

    try:
        T_sat_k = iapws_get_saturation_temperature(P_mpa)
        T_sat_c = kelvin_to_celsius(T_sat_k)

        logger.debug(
            f"Saturation temperature at P={pressure_kpa:.2f} kPa: "
            f"T_sat={T_sat_c:.4f} C"
        )

        return T_sat_c

    except ValueError as e:
        logger.error(f"Failed to compute saturation temperature: {e}")
        raise ValueError(
            f"Pressure {pressure_kpa:.2f} kPa is outside valid saturation range "
            f"[{REGION_BOUNDARIES['P_MIN'] * 1000:.4f}, "
            f"{IF97_CONSTANTS['P_CRIT'] * 1000:.2f}] kPa"
        ) from e


def get_saturation_pressure(temperature_c: float) -> float:
    """
    Get saturation pressure at given temperature.

    DETERMINISTIC: Same input always produces same output.

    Uses IAPWS-IF97 saturation pressure equation for Region 4 to compute
    the saturation pressure corresponding to a given temperature.

    Args:
        temperature_c: Temperature in Celsius (0 to 373.946 C)

    Returns:
        Saturation pressure in kPa

    Raises:
        ValueError: If temperature is outside valid saturation range

    Example:
        >>> P_sat = get_saturation_pressure(100.0)  # 100 C
        >>> print(f"Psat at 100 C: {P_sat:.2f} kPa")
        Psat at 100 C: 101.33 kPa
    """
    T_k = celsius_to_kelvin(temperature_c)

    try:
        P_mpa = iapws_get_saturation_pressure(T_k)
        P_kpa = mpa_to_kpa(P_mpa)

        logger.debug(
            f"Saturation pressure at T={temperature_c:.2f} C: "
            f"P_sat={P_kpa:.4f} kPa"
        )

        return P_kpa

    except ValueError as e:
        logger.error(f"Failed to compute saturation pressure: {e}")
        raise ValueError(
            f"Temperature {temperature_c:.2f} C is outside valid saturation range "
            f"[0, {IF97_CONSTANTS['T_CRIT'] - 273.15:.2f}] C"
        ) from e


def get_saturation_properties(
    pressure_kpa: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Get complete saturation properties at given pressure.

    DETERMINISTIC: Same input always produces same output.

    Returns all saturation properties needed for steam quality calculations
    including enthalpies, entropies, and specific volumes for both saturated
    liquid and saturated vapor states.

    Args:
        pressure_kpa: Pressure in kPa (0.6117 to 22064 kPa)

    Returns:
        Tuple of (hf, hg, hfg, sf, sg, sfg, vf, vg) where:
        - hf: Saturated liquid enthalpy [kJ/kg]
        - hg: Saturated vapor enthalpy [kJ/kg]
        - hfg: Latent heat of vaporization [kJ/kg]
        - sf: Saturated liquid entropy [kJ/(kg*K)]
        - sg: Saturated vapor entropy [kJ/(kg*K)]
        - sfg: Entropy of vaporization [kJ/(kg*K)]
        - vf: Saturated liquid specific volume [m^3/kg]
        - vg: Saturated vapor specific volume [m^3/kg]

    Raises:
        ValueError: If pressure is outside valid saturation range

    Example:
        >>> hf, hg, hfg, sf, sg, sfg, vf, vg = get_saturation_properties(1000.0)
        >>> print(f"At 10 bar: hfg = {hfg:.2f} kJ/kg")
        At 10 bar: hfg = 2015.29 kJ/kg
    """
    P_mpa = kpa_to_mpa(pressure_kpa)

    try:
        sat = iapws_get_saturation_properties(P_mpa)

        logger.debug(
            f"Saturation properties at P={pressure_kpa:.2f} kPa: "
            f"hf={sat.hf:.2f}, hg={sat.hg:.2f}, hfg={sat.hfg:.2f} kJ/kg"
        )

        return (sat.hf, sat.hg, sat.hfg, sat.sf, sat.sg, sat.sfg, sat.vf, sat.vg)

    except ValueError as e:
        logger.error(f"Failed to compute saturation properties: {e}")
        raise


def get_saturation_properties_full(pressure_kpa: float) -> SaturationPropertiesTuple:
    """
    Get complete saturation properties with metadata.

    DETERMINISTIC: Same input always produces same output.

    Similar to get_saturation_properties but returns a dataclass with
    additional metadata including provenance hash for audit trails.

    Args:
        pressure_kpa: Pressure in kPa

    Returns:
        SaturationPropertiesTuple with all properties and provenance

    Raises:
        ValueError: If pressure is outside valid range
    """
    P_mpa = kpa_to_mpa(pressure_kpa)
    sat = iapws_get_saturation_properties(P_mpa)
    T_sat_c = kelvin_to_celsius(sat.temperature_k)

    # Compute provenance hash
    inputs = {"pressure_kpa": pressure_kpa}
    outputs = {
        "hf": sat.hf, "hg": sat.hg, "hfg": sat.hfg,
        "sf": sat.sf, "sg": sat.sg, "sfg": sat.sfg,
        "vf": sat.vf, "vg": sat.vg,
        "temperature_c": T_sat_c,
    }
    provenance = compute_provenance_hash(inputs, outputs, "IAPWS-IF97-Region4")

    return SaturationPropertiesTuple(
        pressure_kpa=pressure_kpa,
        temperature_c=T_sat_c,
        hf=sat.hf,
        hg=sat.hg,
        hfg=sat.hfg,
        sf=sat.sf,
        sg=sat.sg,
        sfg=sat.sfg,
        vf=sat.vf,
        vg=sat.vg,
        provenance_hash=provenance,
    )


def compute_superheat_margin(pressure_kpa: float, temperature_c: float) -> SuperheatMarginResult:
    """
    Compute the superheat margin (degrees above saturation temperature).

    DETERMINISTIC: Same inputs always produce same output.

    The superheat margin is a critical parameter for steam quality control:
    - Positive margin: Steam is superheated (desirable for many applications)
    - Zero margin: Steam is at saturation (wet steam risk)
    - Negative margin: Subcooled liquid (not steam)

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Actual temperature in Celsius

    Returns:
        SuperheatMarginResult with margin and state information

    Example:
        >>> result = compute_superheat_margin(1000.0, 190.0)  # 10 bar, 190 C
        >>> print(f"Superheat: {result.margin_c:.1f} C, Superheated: {result.is_superheated}")
        Superheat: 10.1 C, Superheated: True
    """
    P_mpa = kpa_to_mpa(pressure_kpa)
    P_CRIT = IF97_CONSTANTS["P_CRIT"]

    # Check if near or above critical pressure
    is_critical = P_mpa > P_CRIT * 0.95

    try:
        T_sat_c = get_saturation_temperature(pressure_kpa)
        margin_c = temperature_c - T_sat_c
        is_superheated = margin_c > 0

    except ValueError:
        # Above critical pressure - no saturation point
        T_sat_c = float('nan')
        margin_c = float('nan')
        is_superheated = False
        is_critical = True

    # Compute provenance
    inputs = {"pressure_kpa": pressure_kpa, "temperature_c": temperature_c}
    outputs = {
        "margin_c": margin_c,
        "saturation_temperature_c": T_sat_c,
        "is_superheated": is_superheated,
    }
    provenance = compute_provenance_hash(inputs, outputs, "superheat_margin")

    logger.info(
        f"Superheat margin at P={pressure_kpa:.1f} kPa, T={temperature_c:.1f} C: "
        f"margin={margin_c:.2f} C"
    )

    return SuperheatMarginResult(
        margin_c=margin_c,
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        saturation_temperature_c=T_sat_c,
        is_superheated=is_superheated,
        is_critical=is_critical,
        provenance_hash=provenance,
    )


def determine_steam_state(
    pressure_kpa: float,
    temperature_c: float,
    tolerance_c: float = 0.5,
) -> SteamStateResult:
    """
    Determine the steam state based on pressure and temperature.

    DETERMINISTIC: Same inputs always produce same output.

    Classifies steam into one of five states based on comparison
    with saturation conditions:
    - SUBCOOLED: Temperature below saturation (compressed liquid)
    - SATURATED: Temperature at saturation (within tolerance)
    - WET_STEAM: At saturation, potentially two-phase
    - SUPERHEATED: Temperature above saturation
    - SUPERCRITICAL: Above critical pressure

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius
        tolerance_c: Temperature tolerance for saturation detection
                    (default 0.5 C for industrial applications)

    Returns:
        SteamStateResult with state classification and details

    Example:
        >>> result = determine_steam_state(1000.0, 200.0)
        >>> print(f"State: {result.state.value}")
        State: superheated
    """
    P_mpa = kpa_to_mpa(pressure_kpa)
    P_CRIT = IF97_CONSTANTS["P_CRIT"]

    # Check for supercritical conditions
    if P_mpa > P_CRIT:
        state = SteamState.SUPERCRITICAL
        T_sat_c = float('nan')
        margin_c = float('nan')
        confidence = 0.95  # High confidence for supercritical detection

    else:
        try:
            T_sat_c = get_saturation_temperature(pressure_kpa)
            margin_c = temperature_c - T_sat_c

            if margin_c < -tolerance_c:
                state = SteamState.SUBCOOLED
                confidence = min(0.99, 0.8 + abs(margin_c) / 10.0)

            elif abs(margin_c) <= tolerance_c:
                # At saturation - could be wet steam
                state = SteamState.SATURATED
                # Lower confidence near saturation due to measurement uncertainty
                confidence = max(0.5, 0.9 - abs(margin_c) / tolerance_c * 0.4)

            else:
                state = SteamState.SUPERHEATED
                confidence = min(0.99, 0.8 + margin_c / 20.0)

        except ValueError:
            # Unusual case - pressure at boundary
            state = SteamState.SUPERCRITICAL
            T_sat_c = float('nan')
            margin_c = float('nan')
            confidence = 0.7

    # Compute provenance
    inputs = {
        "pressure_kpa": pressure_kpa,
        "temperature_c": temperature_c,
        "tolerance_c": tolerance_c,
    }
    outputs = {
        "state": state.value,
        "saturation_temperature_c": T_sat_c if not (T_sat_c != T_sat_c) else None,
        "margin_c": margin_c if not (margin_c != margin_c) else None,
        "confidence": confidence,
    }
    provenance = compute_provenance_hash(inputs, outputs, "steam_state_determination")

    logger.info(
        f"Steam state at P={pressure_kpa:.1f} kPa, T={temperature_c:.1f} C: "
        f"{state.value} (confidence={confidence:.2f})"
    )

    return SteamStateResult(
        state=state,
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        saturation_temperature_c=T_sat_c if not (T_sat_c != T_sat_c) else 0.0,
        superheat_margin_c=margin_c if not (margin_c != margin_c) else 0.0,
        confidence=confidence,
        provenance_hash=provenance,
    )


def compute_steam_properties(
    pressure_kpa: float,
    temperature_c: Optional[float] = None,
    quality_x: Optional[float] = None,
) -> SteamPropertiesResult:
    """
    Compute steam properties at given conditions.

    DETERMINISTIC: Same inputs always produce same output.

    Supports two input modes:
    - (P, T): Pressure and temperature for single-phase regions
    - (P, x): Pressure and quality for two-phase region

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius (optional)
        quality_x: Steam quality for two-phase (optional)

    Returns:
        SteamPropertiesResult with all calculated properties

    Raises:
        ValueError: If inputs are invalid or insufficient
    """
    if temperature_c is None and quality_x is None:
        raise ValueError("Must provide either temperature_c or quality_x")

    P_mpa = kpa_to_mpa(pressure_kpa)

    if quality_x is not None:
        # Two-phase calculation
        if quality_x < 0 or quality_x > 1:
            raise ValueError(f"Quality must be in [0, 1], got {quality_x}")

        sat = iapws_get_saturation_properties(P_mpa)
        T_sat_c = kelvin_to_celsius(sat.temperature_k)

        h = region4_mixture_enthalpy(P_mpa, quality_x)
        s = region4_mixture_entropy(P_mpa, quality_x)
        v = region4_mixture_specific_volume(P_mpa, quality_x)
        rho = compute_density(v)

        region = 4
        state = SteamState.WET_STEAM
        cp = None
        superheat = 0.0
        temperature_c = T_sat_c

    else:
        # Single-phase calculation
        T_k = celsius_to_kelvin(temperature_c)
        region = detect_region(P_mpa, T_k)

        # Determine state
        state_result = determine_steam_state(pressure_kpa, temperature_c)
        state = state_result.state
        superheat = state_result.superheat_margin_c

        if region == 1:
            h = region1_specific_enthalpy(P_mpa, T_k)
            s = region1_specific_entropy(P_mpa, T_k)
            v = region1_specific_volume(P_mpa, T_k)
            cp = region1_specific_isobaric_heat_capacity(P_mpa, T_k)

        elif region == 2:
            h = region2_specific_enthalpy(P_mpa, T_k)
            s = region2_specific_entropy(P_mpa, T_k)
            v = region2_specific_volume(P_mpa, T_k)
            cp = region2_specific_isobaric_heat_capacity(P_mpa, T_k)

        else:
            raise ValueError(f"Unsupported region {region} for property calculation")

        rho = compute_density(v)
        quality_x = None

    # Compute provenance
    inputs = {
        "pressure_kpa": pressure_kpa,
        "temperature_c": temperature_c,
        "quality_x": quality_x,
    }
    outputs = {"h": h, "s": s, "v": v, "rho": rho, "region": region}
    provenance = compute_provenance_hash(inputs, outputs, "steam_properties")

    return SteamPropertiesResult(
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        region=region,
        state=state,
        specific_enthalpy_kj_kg=h,
        specific_entropy_kj_kgk=s,
        specific_volume_m3_kg=v,
        density_kg_m3=rho,
        specific_heat_cp_kj_kgk=cp,
        quality_x=quality_x,
        superheat_margin_c=superheat,
        provenance_hash=provenance,
        calculation_timestamp=datetime.utcnow().isoformat(),
    )


def compute_quality_from_enthalpy(pressure_kpa: float, enthalpy_kj_kg: float) -> float:
    """
    Compute steam quality (dryness fraction) from enthalpy.

    DETERMINISTIC: Same inputs always produce same output.

    Uses the relationship: x = (h - hf) / hfg

    Args:
        pressure_kpa: Pressure in kPa
        enthalpy_kj_kg: Specific enthalpy in kJ/kg

    Returns:
        Steam quality x (clamped to [0, 1])
        Returns < 0 if subcooled, > 1 if superheated (unclamped)

    Example:
        >>> x = compute_quality_from_enthalpy(1000.0, 2500.0)
        >>> print(f"Quality: {x:.3f}")
    """
    hf, hg, hfg, _, _, _, _, _ = get_saturation_properties(pressure_kpa)

    if abs(hfg) < 1e-10:
        raise ValueError("Near critical point - latent heat approaches zero")

    x = (enthalpy_kj_kg - hf) / hfg

    logger.debug(
        f"Quality from enthalpy at P={pressure_kpa:.1f} kPa: "
        f"h={enthalpy_kj_kg:.1f} kJ/kg, x={x:.4f}"
    )

    return x


def compute_quality_from_entropy(pressure_kpa: float, entropy_kj_kgk: float) -> float:
    """
    Compute steam quality (dryness fraction) from entropy.

    DETERMINISTIC: Same inputs always produce same output.

    Uses the relationship: x = (s - sf) / sfg

    Args:
        pressure_kpa: Pressure in kPa
        entropy_kj_kgk: Specific entropy in kJ/(kg*K)

    Returns:
        Steam quality x (may be < 0 or > 1 if outside two-phase region)
    """
    _, _, _, sf, sg, sfg, _, _ = get_saturation_properties(pressure_kpa)

    if abs(sfg) < 1e-10:
        raise ValueError("Near critical point - entropy of vaporization approaches zero")

    x = (entropy_kj_kgk - sf) / sfg

    return x


def validate_pressure_temperature(
    pressure_kpa: float,
    temperature_c: float,
) -> Tuple[bool, List[str]]:
    """
    Validate pressure and temperature inputs for IAPWS-IF97.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    P_mpa = kpa_to_mpa(pressure_kpa)
    T_k = celsius_to_kelvin(temperature_c)

    # Check pressure bounds
    P_MIN = REGION_BOUNDARIES["P_MIN"]
    P_MAX = REGION_BOUNDARIES["P_MAX_1_2"]

    if P_mpa < P_MIN:
        errors.append(
            f"Pressure {pressure_kpa:.2f} kPa below minimum "
            f"{P_MIN * 1000:.4f} kPa"
        )
    elif P_mpa > P_MAX:
        errors.append(
            f"Pressure {pressure_kpa:.2f} kPa exceeds maximum "
            f"{P_MAX * 1000:.0f} kPa"
        )

    # Check temperature bounds
    T_MIN = REGION_BOUNDARIES["T_MIN"]
    T_MAX = REGION_BOUNDARIES["T_MAX_2"]

    if T_k < T_MIN:
        errors.append(
            f"Temperature {temperature_c:.2f} C below minimum 0 C"
        )
    elif T_k > T_MAX:
        errors.append(
            f"Temperature {temperature_c:.2f} C exceeds maximum "
            f"{T_MAX - 273.15:.0f} C"
        )

    return len(errors) == 0, errors
