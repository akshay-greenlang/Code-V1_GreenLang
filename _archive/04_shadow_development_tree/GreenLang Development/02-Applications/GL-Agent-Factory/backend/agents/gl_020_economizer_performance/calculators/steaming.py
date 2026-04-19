"""
Steaming Risk Calculator

Implements IAPWS-IF97 saturation temperature calculation and steaming risk
detection for economizer water-side analysis.

Reference:
    IAPWS-IF97: "Revised Release on the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam"
    International Association for the Properties of Water and Steam, 2007.

Steaming occurs when the economizer outlet water temperature approaches
or exceeds the saturation temperature corresponding to the drum pressure.
This causes vapor formation in the economizer tubes, leading to:
- Flow instability (two-phase flow)
- Water hammer
- Tube overheating
- Accelerated corrosion

ZERO-HALLUCINATION: All calculations use exact IAPWS-IF97 Equation 31
coefficients for saturation temperature.
"""

import math
import logging
from typing import NamedTuple
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level classifications for steaming."""
    NONE = "NONE"           # No risk - large margin to saturation
    LOW = "LOW"             # Low risk - adequate margin
    MODERATE = "MODERATE"   # Moderate risk - reduced margin
    HIGH = "HIGH"           # High risk - approaching saturation
    CRITICAL = "CRITICAL"   # Critical - at or above saturation


class SteamingAnalysis(NamedTuple):
    """Complete steaming risk analysis results."""
    saturation_temperature_celsius: float  # Saturation temp at drum pressure
    water_outlet_temperature_celsius: float  # Actual water outlet temperature
    approach_to_saturation_celsius: float   # Margin below saturation
    subcooling_margin_celsius: float        # Same as approach (alias)
    risk_level: RiskLevel                   # Risk classification
    risk_score: float                       # Numeric risk score (0-100)
    recommended_max_outlet_celsius: float   # Recommended maximum outlet temp
    description: str                        # Human-readable risk description


# =============================================================================
# IAPWS-IF97 REGION 4 SATURATION COEFFICIENTS (EQUATION 31)
# =============================================================================

# Exact coefficients from IAPWS-IF97 Table 34
# These define the saturation-pressure curve (backward equation)
IAPWS_IF97_SATURATION_COEFFICIENTS = {
    "n1": 0.11670521452767E+04,
    "n2": -0.72421316703206E+06,
    "n3": -0.17073846940092E+02,
    "n4": 0.12020824702470E+05,
    "n5": -0.32325550322333E+07,
    "n6": 0.14915108613530E+02,
    "n7": -0.48232657361591E+04,
    "n8": 0.40511340542057E+06,
    "n9": -0.23855557567849E+00,
    "n10": 0.65017534844798E+03,
}


def saturation_temperature_IF97(P_MPa: float) -> float:
    """
    Calculate saturation temperature using IAPWS-IF97 Equation 31.

    ZERO-HALLUCINATION FORMULA (IAPWS-IF97 Equation 31):

    T_sat = (n10 + D - sqrt((n10 + D)^2 - 4*(n9 + n10*D))) / 2

    Where:
        D = 2*G / (-F - sqrt(F^2 - 4*E*G))
        E = beta^2 + n3*beta + n6
        F = n1*beta^2 + n4*beta + n7
        G = n2*beta^2 + n5*beta + n8
        beta = P^0.25

    This is the backward equation T_s(P) for Region 4 of IAPWS-IF97.

    Valid range: 611.213 Pa <= P <= 22.064 MPa (0.000611213 to 22.064 MPa)
                 (from triple point to critical point)

    Args:
        P_MPa: Saturation pressure in MPa (absolute)

    Returns:
        Saturation temperature in Kelvin

    Raises:
        ValueError: If pressure is outside valid range

    Example:
        >>> T_sat_K = saturation_temperature_IF97(4.0)  # 4 MPa drum pressure
        >>> T_sat_C = T_sat_K - 273.15
        >>> print(f"Saturation temperature: {T_sat_C:.2f} deg C")
        Saturation temperature: 250.33 deg C

    Notes:
        - For typical boiler drum pressures (2-20 MPa), accuracy is +/- 0.0025%
        - The critical point is at 22.064 MPa, 647.096 K
        - The triple point is at 611.657 Pa, 273.16 K
    """
    # Extract coefficients
    n = IAPWS_IF97_SATURATION_COEFFICIENTS
    n1, n2, n3, n4, n5 = n["n1"], n["n2"], n["n3"], n["n4"], n["n5"]
    n6, n7, n8, n9, n10 = n["n6"], n["n7"], n["n8"], n["n9"], n["n10"]

    # Valid pressure range (IAPWS-IF97)
    P_TRIPLE_POINT_MPA = 611.657e-6  # 611.657 Pa = 0.000611657 MPa
    P_CRITICAL_POINT_MPA = 22.064    # Critical pressure

    # Input validation
    if P_MPa < P_TRIPLE_POINT_MPA:
        raise ValueError(
            f"Pressure {P_MPa} MPa is below triple point pressure "
            f"({P_TRIPLE_POINT_MPA:.6f} MPa)"
        )
    if P_MPa > P_CRITICAL_POINT_MPA:
        raise ValueError(
            f"Pressure {P_MPa} MPa exceeds critical point pressure "
            f"({P_CRITICAL_POINT_MPA} MPa)"
        )

    # ZERO-HALLUCINATION: IAPWS-IF97 Equation 31
    # Step 1: Calculate beta = P^0.25
    beta = P_MPa ** 0.25

    # Step 2: Calculate E, F, G
    E = beta * beta + n3 * beta + n6
    F = n1 * beta * beta + n4 * beta + n7
    G = n2 * beta * beta + n5 * beta + n8

    # Step 3: Calculate D
    # D = 2*G / (-F - sqrt(F^2 - 4*E*G))
    discriminant = F * F - 4.0 * E * G
    if discriminant < 0:
        raise ValueError(
            f"Negative discriminant in IF97 calculation at P={P_MPa} MPa. "
            f"This should not occur for valid pressures."
        )

    D = 2.0 * G / (-F - math.sqrt(discriminant))

    # Step 4: Calculate T_sat
    # T = (n10 + D - sqrt((n10 + D)^2 - 4*(n9 + n10*D))) / 2
    term = n10 + D
    inner_discriminant = term * term - 4.0 * (n9 + n10 * D)

    if inner_discriminant < 0:
        raise ValueError(
            f"Negative inner discriminant in IF97 calculation at P={P_MPa} MPa"
        )

    T_kelvin = (term - math.sqrt(inner_discriminant)) / 2.0

    logger.debug(
        f"IAPWS-IF97 saturation temperature: {T_kelvin:.4f} K "
        f"({T_kelvin - 273.15:.2f} deg C) at {P_MPa} MPa"
    )

    return T_kelvin


def saturation_temperature_celsius(P_MPa: float) -> float:
    """
    Calculate saturation temperature in Celsius.

    Convenience wrapper around saturation_temperature_IF97.

    Args:
        P_MPa: Saturation pressure in MPa (absolute)

    Returns:
        Saturation temperature in degrees Celsius
    """
    T_kelvin = saturation_temperature_IF97(P_MPa)
    return T_kelvin - 273.15


def detect_steaming_risk(
    T_water_out_celsius: float,
    drum_pressure_MPa: float,
    safety_margin_celsius: float = 15.0
) -> SteamingAnalysis:
    """
    Detect steaming risk in economizer based on outlet water temperature.

    Steaming risk is assessed by comparing the water outlet temperature
    to the saturation temperature at the drum pressure.

    Risk Level Classification:
    - NONE:     Approach >= 30 deg C (large safety margin)
    - LOW:      20 <= Approach < 30 deg C (adequate margin)
    - MODERATE: 10 <= Approach < 20 deg C (monitor closely)
    - HIGH:     0 < Approach < 10 deg C (take corrective action)
    - CRITICAL: Approach <= 0 deg C (steaming occurring)

    Args:
        T_water_out_celsius: Economizer water outlet temperature (deg C)
        drum_pressure_MPa: Steam drum pressure (MPa absolute)
        safety_margin_celsius: Minimum recommended margin (default 15 deg C)

    Returns:
        SteamingAnalysis with complete risk assessment

    Example:
        >>> analysis = detect_steaming_risk(
        ...     T_water_out_celsius=235.0,
        ...     drum_pressure_MPa=4.0,
        ... )
        >>> print(f"Risk level: {analysis.risk_level}")
        >>> print(f"Approach to saturation: {analysis.approach_to_saturation_celsius:.1f} deg C")
    """
    # Calculate saturation temperature using IAPWS-IF97
    T_sat_kelvin = saturation_temperature_IF97(drum_pressure_MPa)
    T_sat_celsius = T_sat_kelvin - 273.15

    # Calculate approach to saturation (subcooling margin)
    approach_celsius = T_sat_celsius - T_water_out_celsius
    subcooling_margin = approach_celsius  # Alias

    # Recommended maximum outlet temperature
    recommended_max = T_sat_celsius - safety_margin_celsius

    # Classify risk level
    if approach_celsius >= 30.0:
        risk_level = RiskLevel.NONE
        risk_score = 0.0
        description = (
            f"No steaming risk. Water outlet ({T_water_out_celsius:.1f} deg C) is "
            f"{approach_celsius:.1f} deg C below saturation ({T_sat_celsius:.1f} deg C)."
        )
    elif approach_celsius >= 20.0:
        risk_level = RiskLevel.LOW
        risk_score = 25.0
        description = (
            f"Low steaming risk. Water outlet ({T_water_out_celsius:.1f} deg C) has "
            f"{approach_celsius:.1f} deg C margin to saturation."
        )
    elif approach_celsius >= 10.0:
        risk_level = RiskLevel.MODERATE
        risk_score = 50.0
        description = (
            f"Moderate steaming risk. Water outlet ({T_water_out_celsius:.1f} deg C) is "
            f"within {approach_celsius:.1f} deg C of saturation. Monitor closely."
        )
    elif approach_celsius > 0:
        risk_level = RiskLevel.HIGH
        risk_score = 75.0
        description = (
            f"HIGH steaming risk! Water outlet ({T_water_out_celsius:.1f} deg C) is only "
            f"{approach_celsius:.1f} deg C below saturation. Reduce heat input immediately."
        )
    else:
        risk_level = RiskLevel.CRITICAL
        risk_score = 100.0
        description = (
            f"CRITICAL: Steaming is occurring! Water outlet ({T_water_out_celsius:.1f} deg C) "
            f"exceeds saturation temperature ({T_sat_celsius:.1f} deg C) by "
            f"{abs(approach_celsius):.1f} deg C. Emergency action required."
        )

    logger.info(
        f"Steaming risk analysis: {risk_level.value} "
        f"(T_out={T_water_out_celsius:.1f} C, T_sat={T_sat_celsius:.1f} C, "
        f"approach={approach_celsius:.1f} C)"
    )

    return SteamingAnalysis(
        saturation_temperature_celsius=round(T_sat_celsius, 2),
        water_outlet_temperature_celsius=round(T_water_out_celsius, 2),
        approach_to_saturation_celsius=round(approach_celsius, 2),
        subcooling_margin_celsius=round(subcooling_margin, 2),
        risk_level=risk_level,
        risk_score=risk_score,
        recommended_max_outlet_celsius=round(recommended_max, 2),
        description=description,
    )


def calculate_safe_outlet_temperature(
    drum_pressure_MPa: float,
    margin_celsius: float = 15.0
) -> float:
    """
    Calculate the safe maximum water outlet temperature.

    Args:
        drum_pressure_MPa: Steam drum pressure (MPa)
        margin_celsius: Safety margin below saturation (default 15 deg C)

    Returns:
        Maximum safe water outlet temperature in Celsius
    """
    T_sat_celsius = saturation_temperature_celsius(drum_pressure_MPa)
    return T_sat_celsius - margin_celsius


def estimate_steaming_onset_load(
    T_water_out_current_celsius: float,
    T_water_out_design_celsius: float,
    drum_pressure_MPa: float,
    current_load_percent: float
) -> float:
    """
    Estimate the load percentage at which steaming might begin.

    This uses linear interpolation assuming outlet temperature
    increases linearly with load. For more accurate predictions,
    use detailed heat transfer modeling.

    Args:
        T_water_out_current_celsius: Current water outlet temperature
        T_water_out_design_celsius: Design water outlet temperature
        drum_pressure_MPa: Steam drum pressure
        current_load_percent: Current boiler load (%)

    Returns:
        Estimated load percentage at steaming onset
    """
    T_sat_celsius = saturation_temperature_celsius(drum_pressure_MPa)

    # Assume linear relationship between load and outlet temperature
    # T_out = a * Load + b
    # Using current conditions and ambient assumption at 0% load

    if current_load_percent <= 0:
        return 100.0  # Cannot estimate

    # Rate of temperature rise per load percent
    T_ambient_approx = 25.0  # Assumed ambient feed temperature
    dT_dLoad = (T_water_out_current_celsius - T_ambient_approx) / current_load_percent

    if dT_dLoad <= 0:
        return 100.0  # Invalid - no risk

    # Load at which T_out = T_sat
    Load_steaming = (T_sat_celsius - T_ambient_approx) / dT_dLoad

    return min(Load_steaming, 200.0)  # Cap at 200% for sanity
