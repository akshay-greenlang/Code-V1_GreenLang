"""
Vacuum Optimization Calculator

This module implements vacuum system calculations and optimization for
steam surface condensers, following industry standards and thermodynamic
principles.

All formulas are from standard references:
- HEI Standards for Steam Surface Condensers
- ASME PTC 12.2 Steam Surface Condensers
- Steam Tables (IAPWS-IF97)
- Perry's Chemical Engineers' Handbook

Zero-hallucination: All calculations are deterministic physics formulas.
No ML/LLM in the calculation path.

Example:
    >>> pressure = calculate_saturation_pressure(40.0)
    >>> print(f"Saturation pressure at 40C: {pressure:.2f} kPa abs")
"""

import math
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


# Steam table data (simplified correlation coefficients)
# Reference: IAPWS-IF97 simplified
ANTOINE_COEFFICIENTS = {
    "A": 8.07131,
    "B": 1730.63,
    "C": 233.426,
}

# Air in-leakage standards
# Reference: HEI Standards, Section 5
AIR_LEAKAGE_LIMITS = {
    "new_condenser": 0.0007,  # kg/hr per kW heat load
    "old_condenser": 0.0014,  # kg/hr per kW heat load
    "degraded": 0.0028,  # kg/hr per kW heat load
}

# Vacuum pump sizing factors
# Reference: HEI Standards
VACUUM_PUMP_FACTORS = {
    "hogging": 5.0,  # Factor for initial evacuation
    "holding": 1.0,  # Factor for steady-state
    "safety_margin": 1.25,  # Design safety factor
}


def calculate_saturation_pressure(
    temperature_c: float
) -> float:
    """
    Calculate steam saturation pressure from temperature.

    Uses Antoine equation correlation:
        log10(P_mmHg) = A - B/(C + T)

    Then converts to kPa.

    Reference: Steam tables, Antoine equation

    Args:
        temperature_c: Temperature in Celsius.

    Returns:
        Saturation pressure in kPa absolute.

    Raises:
        ValueError: If temperature is out of valid range.

    Example:
        >>> p = calculate_saturation_pressure(40.0)
        >>> print(f"P_sat: {p:.3f} kPa")
    """
    if temperature_c < 0 or temperature_c > 100:
        raise ValueError(
            f"Temperature must be 0-100C for this correlation, got {temperature_c}"
        )

    a = ANTOINE_COEFFICIENTS["A"]
    b = ANTOINE_COEFFICIENTS["B"]
    c = ANTOINE_COEFFICIENTS["C"]

    # Antoine equation (pressure in mmHg)
    log_p_mmhg = a - b / (c + temperature_c)
    p_mmhg = 10 ** log_p_mmhg

    # Convert mmHg to kPa (1 mmHg = 0.133322 kPa)
    p_kpa = p_mmhg * 0.133322

    return p_kpa


def calculate_saturation_temperature(
    pressure_kpa: float
) -> float:
    """
    Calculate steam saturation temperature from pressure.

    Inverts Antoine equation:
        T = B / (A - log10(P_mmHg)) - C

    Reference: Steam tables, Antoine equation inverse

    Args:
        pressure_kpa: Absolute pressure in kPa.

    Returns:
        Saturation temperature in Celsius.

    Raises:
        ValueError: If pressure is out of valid range.

    Example:
        >>> t = calculate_saturation_temperature(7.38)
        >>> print(f"T_sat: {t:.1f} C")
    """
    if pressure_kpa <= 0 or pressure_kpa > 101.325:
        raise ValueError(
            f"Pressure must be 0-101.325 kPa for this correlation, got {pressure_kpa}"
        )

    a = ANTOINE_COEFFICIENTS["A"]
    b = ANTOINE_COEFFICIENTS["B"]
    c = ANTOINE_COEFFICIENTS["C"]

    # Convert kPa to mmHg
    p_mmhg = pressure_kpa / 0.133322

    # Inverse Antoine equation
    t_c = b / (a - math.log10(p_mmhg)) - c

    return t_c


def calculate_vacuum_inches_hg(
    absolute_pressure_kpa: float,
    barometric_pressure_kpa: float = 101.325
) -> float:
    """
    Calculate vacuum in inches of mercury.

    Formula:
        Vacuum (in Hg) = (P_baro - P_abs) * 0.2953

    Reference: Standard pressure conversion

    Args:
        absolute_pressure_kpa: Absolute pressure in condenser (kPa).
        barometric_pressure_kpa: Local barometric pressure (kPa).

    Returns:
        Vacuum in inches of mercury.

    Example:
        >>> vacuum = calculate_vacuum_inches_hg(7.0, 101.325)
        >>> print(f"Vacuum: {vacuum:.1f} in Hg")
    """
    # 1 kPa = 0.2953 in Hg
    vacuum_kpa = barometric_pressure_kpa - absolute_pressure_kpa
    vacuum_in_hg = vacuum_kpa * 0.2953

    return vacuum_in_hg


def calculate_absolute_pressure_from_vacuum(
    vacuum_in_hg: float,
    barometric_pressure_kpa: float = 101.325
) -> float:
    """
    Calculate absolute pressure from vacuum reading.

    Formula:
        P_abs = P_baro - (Vacuum_in_Hg / 0.2953)

    Reference: Standard pressure conversion

    Args:
        vacuum_in_hg: Vacuum in inches of mercury.
        barometric_pressure_kpa: Local barometric pressure (kPa).

    Returns:
        Absolute pressure in kPa.

    Example:
        >>> p_abs = calculate_absolute_pressure_from_vacuum(28.0)
        >>> print(f"P_abs: {p_abs:.2f} kPa")
    """
    vacuum_kpa = vacuum_in_hg / 0.2953
    p_abs = barometric_pressure_kpa - vacuum_kpa

    return max(0.1, p_abs)  # Clamp to prevent negative


def calculate_vacuum_efficiency(
    actual_pressure_kpa: float,
    theoretical_pressure_kpa: float
) -> float:
    """
    Calculate vacuum system efficiency.

    Formula:
        Efficiency = P_theoretical / P_actual * 100

    The theoretical pressure is based on cooling water temperature.
    Lower actual pressure compared to theoretical indicates higher efficiency.

    Note: This formula gives efficiency > 100% when actual < theoretical,
    which is thermodynamically impossible without subcooling. The calculation
    is bounded at 100%.

    Reference: HEI Standards

    Args:
        actual_pressure_kpa: Actual condenser pressure (kPa abs).
        theoretical_pressure_kpa: Theoretical minimum pressure (kPa abs).

    Returns:
        Vacuum efficiency percentage (0-100).

    Example:
        >>> eff = calculate_vacuum_efficiency(8.0, 7.0)
        >>> print(f"Vacuum efficiency: {eff:.1f}%")
    """
    if actual_pressure_kpa <= 0:
        raise ValueError(f"Actual pressure must be > 0, got {actual_pressure_kpa}")
    if theoretical_pressure_kpa <= 0:
        raise ValueError(
            f"Theoretical pressure must be > 0, got {theoretical_pressure_kpa}"
        )

    # Efficiency ratio
    efficiency = (theoretical_pressure_kpa / actual_pressure_kpa) * 100

    # Cap at 100% (100% means achieving theoretical minimum)
    return min(100.0, max(0.0, efficiency))


def calculate_air_inleakage_rate(
    condenser_volume_m3: float,
    vacuum_kpa: float,
    leakage_area_mm2: float
) -> float:
    """
    Calculate air in-leakage rate through openings.

    Uses orifice flow equation for sonic conditions (choked flow):
        m_air = C_d * A * sqrt(2 * rho_air * P_atm) * (1 - P_vac/P_atm)

    Simplified correlation for typical conditions:
        m_air = 0.00785 * A * sqrt(P_diff)

    Reference: HEI Standards, Fluid dynamics

    Args:
        condenser_volume_m3: Condenser shell volume in m3.
        vacuum_kpa: Vacuum pressure in kPa abs.
        leakage_area_mm2: Estimated total leakage area in mm2.

    Returns:
        Air in-leakage rate in kg/hr.

    Example:
        >>> leakage = calculate_air_inleakage_rate(100, 7.0, 10)
        >>> print(f"Air leakage: {leakage:.3f} kg/hr")
    """
    # Atmospheric pressure
    p_atm = 101.325  # kPa

    # Pressure difference
    p_diff = p_atm - vacuum_kpa  # kPa

    if p_diff <= 0:
        return 0.0

    # Convert area to m2
    area_m2 = leakage_area_mm2 * 1e-6

    # Simplified leakage correlation
    # Assumes choked flow for typical vacuum conditions
    # Reference: Engineering Toolbox, vacuum leak calculations
    c_discharge = 0.65
    air_density = 1.2  # kg/m3 at ambient

    # Mass flow rate (kg/s)
    m_air_kg_s = c_discharge * area_m2 * math.sqrt(2 * air_density * p_diff * 1000)

    # Convert to kg/hr
    m_air_kg_hr = m_air_kg_s * 3600

    return m_air_kg_hr


def assess_air_leakage_severity(
    measured_leakage_kg_hr: float,
    heat_load_kw: float
) -> Tuple[str, float]:
    """
    Assess air in-leakage severity against HEI standards.

    Reference: HEI Standards, Section 5.3

    Args:
        measured_leakage_kg_hr: Measured air in-leakage in kg/hr.
        heat_load_kw: Condenser heat load in kW.

    Returns:
        Tuple of (severity_level, normalized_leakage).
        severity_level: "EXCELLENT", "ACCEPTABLE", "HIGH", "CRITICAL"
        normalized_leakage: Leakage per kW of heat load

    Example:
        >>> severity, norm = assess_air_leakage_severity(15.0, 24000)
        >>> print(f"Severity: {severity}")
    """
    if heat_load_kw <= 0:
        raise ValueError(f"Heat load must be > 0, got {heat_load_kw}")

    # Normalized leakage (kg/hr per kW)
    normalized = measured_leakage_kg_hr / heat_load_kw

    # Compare against standards
    if normalized <= AIR_LEAKAGE_LIMITS["new_condenser"]:
        severity = "EXCELLENT"
    elif normalized <= AIR_LEAKAGE_LIMITS["old_condenser"]:
        severity = "ACCEPTABLE"
    elif normalized <= AIR_LEAKAGE_LIMITS["degraded"]:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    logger.info(
        f"Air leakage assessment: {measured_leakage_kg_hr:.2f} kg/hr, "
        f"normalized={normalized:.5f} kg/hr/kW, severity={severity}"
    )

    return (severity, normalized)


def calculate_vacuum_pump_capacity(
    air_leakage_kg_hr: float,
    vapor_leakage_kg_hr: float,
    suction_pressure_kpa: float,
    suction_temp_c: float,
    mode: str = "holding"
) -> float:
    """
    Calculate required vacuum pump capacity.

    Formula:
        Capacity = (m_air + m_vapor) * v_specific * safety_factor

    Where v_specific is the specific volume at suction conditions.

    Reference: HEI Standards, Vacuum pump sizing

    Args:
        air_leakage_kg_hr: Air in-leakage rate in kg/hr.
        vapor_leakage_kg_hr: Steam/vapor in air removal system (kg/hr).
        suction_pressure_kpa: Vacuum pump suction pressure (kPa abs).
        suction_temp_c: Suction temperature in Celsius.
        mode: "hogging" for initial evacuation, "holding" for steady-state.

    Returns:
        Required vacuum pump capacity in m3/min.

    Example:
        >>> capacity = calculate_vacuum_pump_capacity(
        ...     15.0, 50.0, 5.0, 35.0, "holding"
        ... )
        >>> print(f"Pump capacity: {capacity:.1f} m3/min")
    """
    # Mode factor
    if mode.lower() == "hogging":
        mode_factor = VACUUM_PUMP_FACTORS["hogging"]
    else:
        mode_factor = VACUUM_PUMP_FACTORS["holding"]

    safety_factor = VACUUM_PUMP_FACTORS["safety_margin"]

    # Total mass flow to remove
    total_mass_kg_hr = air_leakage_kg_hr + vapor_leakage_kg_hr

    # Specific volume of air at suction conditions
    # Using ideal gas law: v = R*T/P
    r_air = 0.287  # kJ/kg-K
    t_kelvin = suction_temp_c + 273.15

    v_specific = r_air * t_kelvin / suction_pressure_kpa  # m3/kg

    # Volumetric flow rate (m3/hr)
    vol_flow_m3_hr = total_mass_kg_hr * v_specific * mode_factor * safety_factor

    # Convert to m3/min
    vol_flow_m3_min = vol_flow_m3_hr / 60

    logger.debug(
        f"Vacuum pump sizing: m_total={total_mass_kg_hr:.1f} kg/hr, "
        f"v_specific={v_specific:.3f} m3/kg, capacity={vol_flow_m3_min:.1f} m3/min"
    )

    return vol_flow_m3_min


def calculate_theoretical_vacuum(
    cooling_water_inlet_temp_c: float,
    cooling_water_rise_c: float,
    ttd_c: float
) -> float:
    """
    Calculate theoretical best achievable vacuum.

    The theoretical vacuum is based on the saturation pressure at
    the condenser operating temperature, which depends on cooling
    water conditions.

    Formula:
        T_sat = T_cw_inlet + Rise + TTD
        P_theoretical = P_sat(T_sat)

    Reference: Thermodynamic principles

    Args:
        cooling_water_inlet_temp_c: Cooling water inlet temperature (C).
        cooling_water_rise_c: Expected cooling water temperature rise (C).
        ttd_c: Target terminal temperature difference (C).

    Returns:
        Theoretical minimum condenser pressure in kPa abs.

    Example:
        >>> p_theo = calculate_theoretical_vacuum(25.0, 10.0, 3.0)
        >>> print(f"Theoretical pressure: {p_theo:.3f} kPa")
    """
    # Calculate saturation temperature
    t_sat = cooling_water_inlet_temp_c + cooling_water_rise_c + ttd_c

    # Get saturation pressure
    p_theoretical = calculate_saturation_pressure(t_sat)

    return p_theoretical


def calculate_vacuum_deviation(
    actual_pressure_kpa: float,
    theoretical_pressure_kpa: float
) -> Tuple[float, str]:
    """
    Calculate vacuum deviation from theoretical.

    Formula:
        Deviation = P_actual - P_theoretical
        Status based on magnitude

    Reference: HEI Standards, performance assessment

    Args:
        actual_pressure_kpa: Actual condenser pressure (kPa abs).
        theoretical_pressure_kpa: Theoretical minimum pressure (kPa abs).

    Returns:
        Tuple of (deviation_kpa, status).
        deviation_kpa: Pressure deviation in kPa
        status: "EXCELLENT", "GOOD", "MARGINAL", "POOR"

    Example:
        >>> dev, status = calculate_vacuum_deviation(8.0, 7.0)
        >>> print(f"Deviation: {dev:.2f} kPa, Status: {status}")
    """
    deviation = actual_pressure_kpa - theoretical_pressure_kpa

    # Status thresholds (kPa deviation)
    if deviation <= 0.5:
        status = "EXCELLENT"
    elif deviation <= 1.0:
        status = "GOOD"
    elif deviation <= 2.0:
        status = "MARGINAL"
    else:
        status = "POOR"

    return (deviation, status)


def calculate_power_loss_from_vacuum_degradation(
    actual_pressure_kpa: float,
    design_pressure_kpa: float,
    turbine_power_mw: float,
    exhaust_enthalpy_sensitivity: float = 0.012
) -> float:
    """
    Calculate turbine power loss from vacuum degradation.

    Higher backpressure reduces turbine output. The sensitivity
    depends on turbine design but typically ~1-1.5% per kPa.

    Formula:
        Power_loss = P_turbine * sensitivity * (P_actual - P_design)

    Reference: Turbine performance curves, HEI Standards

    Args:
        actual_pressure_kpa: Actual condenser pressure (kPa abs).
        design_pressure_kpa: Design condenser pressure (kPa abs).
        turbine_power_mw: Turbine rated power in MW.
        exhaust_enthalpy_sensitivity: Power sensitivity factor (MW/kPa per MW).

    Returns:
        Power loss in MW.

    Example:
        >>> loss = calculate_power_loss_from_vacuum_degradation(
        ...     9.0, 7.0, 500.0
        ... )
        >>> print(f"Power loss: {loss:.1f} MW")
    """
    if actual_pressure_kpa <= design_pressure_kpa:
        return 0.0  # No loss if at or better than design

    pressure_deviation = actual_pressure_kpa - design_pressure_kpa

    # Power loss calculation
    power_loss = turbine_power_mw * exhaust_enthalpy_sensitivity * pressure_deviation

    logger.info(
        f"Vacuum degradation power loss: dP={pressure_deviation:.2f} kPa, "
        f"loss={power_loss:.2f} MW ({power_loss/turbine_power_mw*100:.2f}%)"
    )

    return power_loss


def calculate_optimal_vacuum(
    cooling_water_inlet_temp_c: float,
    cooling_water_flow_kg_s: float,
    heat_duty_kw: float,
    surface_area_m2: float,
    u_clean: float,
    cleanliness_factor: float = 0.85
) -> Dict[str, float]:
    """
    Calculate optimal condenser vacuum for given conditions.

    Performs iterative heat balance to find operating point.

    Reference: Heat balance, thermodynamic principles

    Args:
        cooling_water_inlet_temp_c: CW inlet temperature (C).
        cooling_water_flow_kg_s: CW flow rate (kg/s).
        heat_duty_kw: Condenser heat duty (kW).
        surface_area_m2: Heat transfer surface area (m2).
        u_clean: Clean heat transfer coefficient (W/m2K).
        cleanliness_factor: HEI cleanliness factor (0-1).

    Returns:
        Dictionary with optimal operating parameters:
        - saturation_temp_c
        - saturation_pressure_kpa
        - cooling_water_outlet_temp_c
        - ttd_c
        - lmtd_c

    Example:
        >>> result = calculate_optimal_vacuum(
        ...     25.0, 1000.0, 24000, 5000, 3500, 0.85
        ... )
    """
    # Apply cleanliness factor
    u_actual = u_clean * cleanliness_factor

    # Water specific heat (kJ/kg-K)
    cp_water = 4.18

    # Calculate CW outlet temperature from energy balance
    # Q = m_cw * Cp * (T_out - T_in)
    t_cw_out = cooling_water_inlet_temp_c + heat_duty_kw / (cooling_water_flow_kg_s * cp_water)

    # Iterative solution for saturation temperature
    # Q = U * A * LMTD
    # Start with initial guess
    t_sat_guess = t_cw_out + 5.0  # Initial TTD guess of 5C

    for _ in range(20):  # Max iterations
        # Calculate LMTD
        dt1 = t_sat_guess - cooling_water_inlet_temp_c
        dt2 = t_sat_guess - t_cw_out

        if dt1 <= 0 or dt2 <= 0:
            t_sat_guess += 2.0
            continue

        if dt1 == dt2:
            lmtd = dt1
        else:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        # Calculate heat duty from U*A*LMTD
        q_calc = u_actual * surface_area_m2 * lmtd / 1000  # kW

        # Adjust saturation temperature
        error = (q_calc - heat_duty_kw) / heat_duty_kw

        if abs(error) < 0.001:
            break

        # Newton-Raphson style adjustment
        t_sat_guess = t_sat_guess * (1 - error * 0.3)

    # Final calculations
    ttd = t_sat_guess - t_cw_out
    p_sat = calculate_saturation_pressure(t_sat_guess)

    return {
        "saturation_temp_c": t_sat_guess,
        "saturation_pressure_kpa": p_sat,
        "cooling_water_outlet_temp_c": t_cw_out,
        "ttd_c": ttd,
        "lmtd_c": lmtd,
    }


def generate_vacuum_recommendations(
    actual_pressure_kpa: float,
    design_pressure_kpa: float,
    air_leakage_severity: str,
    cleanliness_factor: float,
    ttd_c: float
) -> List[Dict[str, str]]:
    """
    Generate recommendations for vacuum optimization.

    Reference: HEI Standards, Best practices

    Args:
        actual_pressure_kpa: Actual condenser pressure (kPa abs).
        design_pressure_kpa: Design condenser pressure (kPa abs).
        air_leakage_severity: Air leakage severity level.
        cleanliness_factor: Current cleanliness factor.
        ttd_c: Current terminal temperature difference.

    Returns:
        List of recommendation dictionaries with action and priority.

    Example:
        >>> recs = generate_vacuum_recommendations(
        ...     9.0, 7.0, "HIGH", 0.75, 8.0
        ... )
    """
    recommendations = []

    # Check pressure deviation
    pressure_deviation = actual_pressure_kpa - design_pressure_kpa

    if pressure_deviation > 2.0:
        recommendations.append({
            "action": "Investigate root cause of severe vacuum degradation",
            "priority": "CRITICAL",
            "reason": f"Pressure {pressure_deviation:.1f} kPa above design",
        })
    elif pressure_deviation > 1.0:
        recommendations.append({
            "action": "Schedule condenser performance review",
            "priority": "HIGH",
            "reason": f"Pressure {pressure_deviation:.1f} kPa above design",
        })

    # Check air leakage
    if air_leakage_severity == "CRITICAL":
        recommendations.append({
            "action": "Perform immediate air leak survey and repairs",
            "priority": "CRITICAL",
            "reason": "Critical air in-leakage detected",
        })
    elif air_leakage_severity == "HIGH":
        recommendations.append({
            "action": "Schedule air leak detection and repair",
            "priority": "HIGH",
            "reason": "Elevated air in-leakage",
        })

    # Check cleanliness
    if cleanliness_factor < 0.70:
        recommendations.append({
            "action": "Schedule condenser tube cleaning",
            "priority": "HIGH",
            "reason": f"Cleanliness factor {cleanliness_factor:.2f} below threshold",
        })
    elif cleanliness_factor < 0.80:
        recommendations.append({
            "action": "Plan condenser tube cleaning at next opportunity",
            "priority": "MEDIUM",
            "reason": f"Cleanliness factor {cleanliness_factor:.2f} declining",
        })

    # Check TTD
    if ttd_c > 8.0:
        recommendations.append({
            "action": "Investigate high TTD - check for air blanketing or fouling",
            "priority": "HIGH",
            "reason": f"TTD of {ttd_c:.1f}C is excessive",
        })
    elif ttd_c > 5.0:
        recommendations.append({
            "action": "Monitor TTD trend for continued degradation",
            "priority": "MEDIUM",
            "reason": f"TTD of {ttd_c:.1f}C is above optimal",
        })

    return recommendations
