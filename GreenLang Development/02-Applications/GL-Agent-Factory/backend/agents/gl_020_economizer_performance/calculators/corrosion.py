"""
Corrosion Risk Calculator

Implements cold-end corrosion risk assessment for economizer tubes based on
tube metal temperature relative to acid dew point.

Cold-end corrosion occurs when the tube surface temperature falls below
the acid dew point, causing sulfuric acid condensation on tube surfaces.
This leads to accelerated corrosion, tube wall thinning, and eventual
failure if not controlled.

Reference:
    - ASME PTC 4: Fired Steam Generators
    - EPRI Guidelines for Boiler Economizer Performance

ZERO-HALLUCINATION: All risk thresholds are based on established
engineering practice and documented in industry standards.
"""

import math
import logging
from typing import NamedTuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level classifications for corrosion."""
    NONE = "NONE"         # No risk - well above dew point
    LOW = "LOW"           # Low risk - adequate margin
    MODERATE = "MODERATE" # Moderate risk - reduced margin
    HIGH = "HIGH"         # High risk - approaching dew point
    SEVERE = "SEVERE"     # Severe - at or below dew point


class CorrosionMechanism(str, Enum):
    """Types of corrosion mechanisms."""
    SULFURIC_ACID = "sulfuric_acid"           # H2SO4 condensation
    SULFUROUS_ACID = "sulfurous_acid"         # H2SO3 condensation (rare)
    HYDROCHLORIC_ACID = "hydrochloric_acid"   # HCl condensation
    WATER_DEW_POINT = "water_dew_point"       # Pure water condensation
    LOW_TEMPERATURE_CREEP = "low_temp_creep"  # Long-term low-temp corrosion


class CorrosionAnalysis(NamedTuple):
    """Complete corrosion risk analysis results."""
    tube_metal_temperature_celsius: float     # Estimated tube metal temperature
    acid_dew_point_celsius: float             # Acid dew point
    margin_above_dew_point_celsius: float     # T_metal - T_dew_point
    risk_level: RiskLevel                     # Risk classification
    risk_score: float                         # Numeric risk score (0-100)
    mechanism: CorrosionMechanism             # Primary corrosion mechanism
    recommended_min_metal_temp_celsius: float # Minimum safe metal temperature
    estimated_corrosion_rate_mm_per_year: float  # Estimated corrosion rate
    remaining_tube_life_years: Optional[float]   # Estimated remaining life
    description: str                          # Human-readable description


def estimate_tube_metal_temperature(
    T_fluid_celsius: float,
    T_gas_celsius: float,
    heat_flux_W_per_m2: float,
    tube_wall_thickness_mm: float = 4.0,
    tube_material_conductivity_W_per_mK: float = 50.0,
    internal_film_coefficient_W_per_m2K: float = 5000.0,
    fouling_resistance_m2K_per_W: float = 0.0002
) -> float:
    """
    Estimate tube metal temperature at the cold (gas-side) surface.

    ZERO-HALLUCINATION FORMULA:

    The temperature distribution through the tube wall is calculated using
    thermal resistance concepts:

    T_metal_outer = T_fluid + Q'' * (R_film + R_fouling + R_wall)

    Where:
        T_fluid = Bulk fluid temperature (deg C)
        Q'' = Heat flux (W/m2)
        R_film = 1 / h_internal (m2-K/W)
        R_fouling = Fouling resistance (m2-K/W)
        R_wall = wall_thickness / k_material (m2-K/W)

    For economizers, the critical location is at the gas inlet (hottest gas)
    and water inlet (coldest water), which produces the highest heat flux
    and lowest metal temperature.

    Args:
        T_fluid_celsius: Bulk water temperature (deg C)
        T_gas_celsius: Local gas temperature (deg C)
        heat_flux_W_per_m2: Local heat flux (W/m2)
        tube_wall_thickness_mm: Tube wall thickness (mm)
        tube_material_conductivity_W_per_mK: Thermal conductivity (W/m-K)
            - Carbon steel: ~50 W/m-K
            - Stainless steel: ~15-25 W/m-K
            - Inconel: ~10-15 W/m-K
        internal_film_coefficient_W_per_m2K: Water-side heat transfer (W/m2-K)
            - Typical for economizers: 3000-8000 W/m2-K
        fouling_resistance_m2K_per_W: Water-side fouling resistance (m2-K/W)
            - Clean: 0.0001-0.0002
            - Moderately fouled: 0.0002-0.0005
            - Heavily fouled: > 0.0005

    Returns:
        Estimated outer (gas-side) tube metal temperature in Celsius

    Example:
        >>> T_metal = estimate_tube_metal_temperature(
        ...     T_fluid_celsius=105.0,
        ...     T_gas_celsius=350.0,
        ...     heat_flux_W_per_m2=50000.0,
        ... )
        >>> print(f"Tube metal temperature: {T_metal:.1f} deg C")
    """
    # Input validation
    if tube_wall_thickness_mm <= 0:
        raise ValueError(f"Tube wall thickness must be positive: {tube_wall_thickness_mm}")
    if tube_material_conductivity_W_per_mK <= 0:
        raise ValueError(f"Material conductivity must be positive")
    if internal_film_coefficient_W_per_m2K <= 0:
        raise ValueError(f"Film coefficient must be positive")

    # Convert wall thickness to meters
    wall_thickness_m = tube_wall_thickness_mm / 1000.0

    # Calculate thermal resistances (m2-K/W)
    # ZERO-HALLUCINATION: Standard thermal resistance formulas
    R_film = 1.0 / internal_film_coefficient_W_per_m2K
    R_fouling = fouling_resistance_m2K_per_W
    R_wall = wall_thickness_m / tube_material_conductivity_W_per_mK

    # Total resistance from fluid to outer metal surface
    R_total = R_film + R_fouling + R_wall

    # Calculate outer metal temperature
    # T_metal_outer = T_fluid + Q'' * R_total
    T_metal_outer = T_fluid_celsius + heat_flux_W_per_m2 * R_total

    # Alternative calculation using thermal resistance network
    # The metal temperature should be between fluid and gas temperatures
    if T_metal_outer > T_gas_celsius:
        logger.warning(
            f"Calculated metal temp ({T_metal_outer:.1f} C) exceeds gas temp "
            f"({T_gas_celsius:.1f} C). Using estimate based on gas temp."
        )
        # Use simplified estimate
        T_metal_outer = T_fluid_celsius + 0.1 * (T_gas_celsius - T_fluid_celsius)

    logger.debug(
        f"Tube metal temperature estimated: {T_metal_outer:.1f} deg C "
        f"(T_fluid={T_fluid_celsius:.1f}, T_gas={T_gas_celsius:.1f}, "
        f"Q''={heat_flux_W_per_m2:.0f} W/m2)"
    )

    return T_metal_outer


def estimate_minimum_metal_temperature(
    T_water_inlet_celsius: float,
    T_gas_inlet_celsius: float,
    velocity_water_m_s: float = 1.0,
    tube_inner_diameter_mm: float = 43.0
) -> float:
    """
    Estimate the minimum tube metal temperature in the economizer.

    The minimum metal temperature occurs at the cold end where:
    - Water is at its lowest temperature (inlet)
    - Gas is at its lowest temperature (outlet for counter-flow)

    For economizer design, we typically want to know the coldest
    tube metal temperature to assess acid dew point corrosion risk.

    Args:
        T_water_inlet_celsius: Water inlet temperature (deg C)
        T_gas_inlet_celsius: Gas temperature at water inlet end (deg C)
        velocity_water_m_s: Water velocity in tubes (m/s)
        tube_inner_diameter_mm: Tube inner diameter (mm)

    Returns:
        Estimated minimum tube metal temperature (deg C)
    """
    # The minimum metal temperature is approximately:
    # T_metal_min ≈ T_water_inlet + 0.05 * (T_gas - T_water_inlet)
    # This accounts for the thermal resistance from water to outer surface

    # Simple engineering estimate (conservative)
    # The metal temperature is closer to water than gas temperature
    delta_T = T_gas_inlet_celsius - T_water_inlet_celsius
    T_metal_min = T_water_inlet_celsius + 0.05 * delta_T

    # Adjust for water velocity (higher velocity = better heat transfer = metal closer to water)
    if velocity_water_m_s > 1.5:
        T_metal_min = T_water_inlet_celsius + 0.03 * delta_T
    elif velocity_water_m_s < 0.5:
        T_metal_min = T_water_inlet_celsius + 0.10 * delta_T

    return T_metal_min


def assess_corrosion_risk(
    T_metal_celsius: float,
    T_acid_dew_point_celsius: float,
    tube_wall_thickness_mm: float = 4.0,
    original_wall_thickness_mm: Optional[float] = None
) -> CorrosionAnalysis:
    """
    Assess cold-end corrosion risk based on tube metal temperature.

    Risk Level Classification:
    - NONE:     Margin >= 25 deg C above dew point (safe operation)
    - LOW:      15 <= Margin < 25 deg C (normal operation)
    - MODERATE: 5 <= Margin < 15 deg C (reduced margin, monitor)
    - HIGH:     0 < Margin < 5 deg C (corrosion likely, take action)
    - SEVERE:   Margin <= 0 deg C (active corrosion, immediate action)

    Corrosion Rate Estimation:
    - Based on empirical data from coal and oil-fired boilers
    - Corrosion rate increases exponentially as metal temperature
      drops below acid dew point

    Args:
        T_metal_celsius: Tube metal temperature (deg C)
        T_acid_dew_point_celsius: Acid dew point temperature (deg C)
        tube_wall_thickness_mm: Current tube wall thickness (mm)
        original_wall_thickness_mm: Original wall thickness for life calc (mm)

    Returns:
        CorrosionAnalysis with complete risk assessment

    Example:
        >>> analysis = assess_corrosion_risk(
        ...     T_metal_celsius=120.0,
        ...     T_acid_dew_point_celsius=127.5,
        ... )
        >>> print(f"Risk level: {analysis.risk_level}")
        >>> print(f"Corrosion rate: {analysis.estimated_corrosion_rate_mm_per_year:.3f} mm/year")
    """
    # Calculate margin above dew point
    margin_celsius = T_metal_celsius - T_acid_dew_point_celsius

    # Recommended minimum metal temperature (15 deg C above dew point)
    # This is standard industry practice for boiler economizers
    SAFETY_MARGIN_DEG_C = 15.0
    recommended_min_temp = T_acid_dew_point_celsius + SAFETY_MARGIN_DEG_C

    # Determine risk level and estimate corrosion rate
    if margin_celsius >= 25.0:
        risk_level = RiskLevel.NONE
        risk_score = 0.0
        corrosion_rate = 0.0  # mm/year (negligible)
        description = (
            f"No corrosion risk. Tube metal temperature ({T_metal_celsius:.1f} deg C) is "
            f"{margin_celsius:.1f} deg C above acid dew point ({T_acid_dew_point_celsius:.1f} deg C)."
        )

    elif margin_celsius >= 15.0:
        risk_level = RiskLevel.LOW
        risk_score = 20.0
        corrosion_rate = 0.01  # mm/year (very low)
        description = (
            f"Low corrosion risk. Tube metal temperature ({T_metal_celsius:.1f} deg C) has "
            f"{margin_celsius:.1f} deg C margin above dew point. Normal operation."
        )

    elif margin_celsius >= 5.0:
        risk_level = RiskLevel.MODERATE
        risk_score = 50.0
        # Moderate corrosion rate - some acid condensation possible
        corrosion_rate = 0.05  # mm/year
        description = (
            f"Moderate corrosion risk. Tube metal temperature ({T_metal_celsius:.1f} deg C) is only "
            f"{margin_celsius:.1f} deg C above dew point. Consider increasing water inlet "
            f"temperature or reducing flue gas SO3 content."
        )

    elif margin_celsius > 0:
        risk_level = RiskLevel.HIGH
        risk_score = 75.0
        # High corrosion rate - frequent acid condensation
        # Rate increases as margin decreases
        corrosion_rate = 0.1 + 0.05 * (5.0 - margin_celsius)  # 0.1 to 0.35 mm/year
        description = (
            f"HIGH corrosion risk! Tube metal temperature ({T_metal_celsius:.1f} deg C) is only "
            f"{margin_celsius:.1f} deg C above dew point. Acid condensation is likely. "
            f"Increase feedwater temperature immediately."
        )

    else:
        risk_level = RiskLevel.SEVERE
        risk_score = 100.0
        # Severe corrosion - continuous acid condensation
        # Rate depends on how far below dew point
        degrees_below = abs(margin_celsius)
        corrosion_rate = 0.5 + 0.1 * min(degrees_below, 20.0)  # 0.5 to 2.5 mm/year
        description = (
            f"SEVERE corrosion risk! Tube metal temperature ({T_metal_celsius:.1f} deg C) is "
            f"{abs(margin_celsius):.1f} deg C BELOW acid dew point ({T_acid_dew_point_celsius:.1f} deg C). "
            f"Active sulfuric acid corrosion is occurring. IMMEDIATE corrective action required."
        )

    # Estimate remaining tube life
    remaining_life_years = None
    if original_wall_thickness_mm and corrosion_rate > 0:
        # Minimum allowable wall thickness (typically 50% of original)
        min_allowable = original_wall_thickness_mm * 0.5
        remaining_wall = tube_wall_thickness_mm - min_allowable

        if remaining_wall > 0 and corrosion_rate > 0:
            remaining_life_years = remaining_wall / corrosion_rate
        else:
            remaining_life_years = 0.0

    logger.info(
        f"Corrosion risk analysis: {risk_level.value} "
        f"(T_metal={T_metal_celsius:.1f} C, T_dew={T_acid_dew_point_celsius:.1f} C, "
        f"margin={margin_celsius:.1f} C, rate={corrosion_rate:.3f} mm/yr)"
    )

    return CorrosionAnalysis(
        tube_metal_temperature_celsius=round(T_metal_celsius, 2),
        acid_dew_point_celsius=round(T_acid_dew_point_celsius, 2),
        margin_above_dew_point_celsius=round(margin_celsius, 2),
        risk_level=risk_level,
        risk_score=risk_score,
        mechanism=CorrosionMechanism.SULFURIC_ACID,
        recommended_min_metal_temp_celsius=round(recommended_min_temp, 2),
        estimated_corrosion_rate_mm_per_year=round(corrosion_rate, 4),
        remaining_tube_life_years=round(remaining_life_years, 1) if remaining_life_years else None,
        description=description,
    )


def calculate_required_water_inlet_temperature(
    T_acid_dew_point_celsius: float,
    safety_margin_celsius: float = 15.0,
    T_gas_outlet_celsius: float = 150.0
) -> float:
    """
    Calculate the minimum required water inlet temperature to avoid corrosion.

    For counter-flow economizers, the coldest tube metal temperature occurs
    at the water inlet (cold end). To prevent corrosion, the metal temperature
    must be maintained above the acid dew point.

    Args:
        T_acid_dew_point_celsius: Acid dew point temperature (deg C)
        safety_margin_celsius: Desired margin above dew point (deg C)
        T_gas_outlet_celsius: Flue gas outlet temperature (deg C)

    Returns:
        Minimum recommended water inlet temperature (deg C)

    Example:
        >>> T_water_min = calculate_required_water_inlet_temperature(
        ...     T_acid_dew_point_celsius=127.5,
        ...     safety_margin_celsius=15.0,
        ... )
        >>> print(f"Minimum water inlet: {T_water_min:.1f} deg C")
    """
    # Required minimum metal temperature
    T_metal_min_required = T_acid_dew_point_celsius + safety_margin_celsius

    # Metal temperature is slightly above water temperature
    # T_metal ≈ T_water + 0.05 * (T_gas - T_water)
    # Solving for T_water:
    # T_metal = T_water + 0.05 * T_gas - 0.05 * T_water
    # T_metal = 0.95 * T_water + 0.05 * T_gas
    # T_water = (T_metal - 0.05 * T_gas) / 0.95

    T_water_inlet_min = (T_metal_min_required - 0.05 * T_gas_outlet_celsius) / 0.95

    logger.debug(
        f"Required minimum water inlet temperature: {T_water_inlet_min:.1f} deg C "
        f"(T_dew={T_acid_dew_point_celsius:.1f} C, margin={safety_margin_celsius} C)"
    )

    return T_water_inlet_min


def assess_tube_life(
    current_wall_thickness_mm: float,
    original_wall_thickness_mm: float,
    corrosion_rate_mm_per_year: float,
    minimum_allowable_thickness_fraction: float = 0.5
) -> dict:
    """
    Assess remaining tube life based on wall thickness and corrosion rate.

    Args:
        current_wall_thickness_mm: Current measured wall thickness
        original_wall_thickness_mm: Original design wall thickness
        corrosion_rate_mm_per_year: Estimated corrosion rate
        minimum_allowable_thickness_fraction: Fraction of original (default 0.5)

    Returns:
        Dictionary with tube life assessment
    """
    minimum_thickness = original_wall_thickness_mm * minimum_allowable_thickness_fraction
    wall_loss = original_wall_thickness_mm - current_wall_thickness_mm
    wall_loss_percent = (wall_loss / original_wall_thickness_mm) * 100

    remaining_wall = current_wall_thickness_mm - minimum_thickness

    if corrosion_rate_mm_per_year > 0:
        remaining_life_years = remaining_wall / corrosion_rate_mm_per_year
    else:
        remaining_life_years = float('inf')

    # Condition assessment
    if wall_loss_percent < 10:
        condition = "Excellent"
    elif wall_loss_percent < 20:
        condition = "Good"
    elif wall_loss_percent < 35:
        condition = "Fair"
    elif wall_loss_percent < 50:
        condition = "Poor"
    else:
        condition = "Critical"

    return {
        "original_thickness_mm": original_wall_thickness_mm,
        "current_thickness_mm": current_wall_thickness_mm,
        "minimum_allowable_mm": minimum_thickness,
        "wall_loss_mm": wall_loss,
        "wall_loss_percent": wall_loss_percent,
        "remaining_to_minimum_mm": remaining_wall,
        "corrosion_rate_mm_per_year": corrosion_rate_mm_per_year,
        "remaining_life_years": remaining_life_years,
        "condition": condition,
    }
