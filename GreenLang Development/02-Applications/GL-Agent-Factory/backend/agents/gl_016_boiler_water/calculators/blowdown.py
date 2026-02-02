"""
Blowdown Optimization Calculator for Boiler Water Treatment

This module implements blowdown rate optimization calculations
for boiler water management, balancing water quality against
water and energy savings.

All formulas are deterministic physics-based calculations
following zero-hallucination principles. No ML/LLM in calculation path.

Standards Reference:
    - ASME PTC 4 Fired Steam Generators
    - DOE Steam System Best Practices
    - EPRI Heat Rate Improvement Guidelines

Example:
    >>> blowdown_rate = calculate_blowdown_rate(cycles=10)
    >>> savings = calculate_blowdown_savings(current_rate=10, optimal_rate=5)
"""

import math
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


# Constants for energy calculations
WATER_SPECIFIC_HEAT_BTU_LB_F = 1.0  # BTU/lb-F
LATENT_HEAT_BTU_LB = 970  # Approximate latent heat of vaporization at typical pressures
BOILER_EFFICIENCY_DEFAULT = 0.82  # 82% typical boiler efficiency
HOURS_PER_YEAR = 8760


def calculate_blowdown_rate_from_cycles(
    cycles_of_concentration: float
) -> float:
    """
    Calculate required blowdown rate from cycles of concentration.

    The fundamental mass balance equation:
        Blowdown Rate (%) = 100 / (COC - 1)

    This assumes makeup water equals steam production + blowdown.

    Args:
        cycles_of_concentration: Target cycles of concentration (must be > 1).

    Returns:
        Required blowdown rate as percentage of steam production.

    Raises:
        ValueError: If cycles <= 1.

    Example:
        >>> rate = calculate_blowdown_rate_from_cycles(10)
        >>> print(f"Blowdown rate: {rate:.2f}%")
        Blowdown rate: 11.11%
    """
    if cycles_of_concentration <= 1:
        raise ValueError(
            f"Cycles of concentration must be > 1, got {cycles_of_concentration}"
        )

    blowdown_rate = 100.0 / (cycles_of_concentration - 1)

    logger.debug(f"Blowdown rate for COC={cycles_of_concentration}: {blowdown_rate:.2f}%")

    return blowdown_rate


def calculate_cycles_from_blowdown(
    blowdown_rate_percent: float
) -> float:
    """
    Calculate cycles of concentration from blowdown rate.

    Inverse of the mass balance equation:
        COC = 1 + (100 / Blowdown Rate)

    Args:
        blowdown_rate_percent: Blowdown rate as percentage of steam.

    Returns:
        Cycles of concentration.

    Raises:
        ValueError: If blowdown rate <= 0.

    Example:
        >>> coc = calculate_cycles_from_blowdown(10)
        >>> print(f"COC: {coc:.1f}")
        COC: 11.0
    """
    if blowdown_rate_percent <= 0:
        raise ValueError(
            f"Blowdown rate must be > 0, got {blowdown_rate_percent}"
        )

    cycles = 1 + (100.0 / blowdown_rate_percent)

    return cycles


def calculate_makeup_water_flow(
    steam_rate_lb_hr: float,
    blowdown_rate_percent: float,
    condensate_return_percent: float = 0
) -> float:
    """
    Calculate makeup water flow rate.

    Formula:
        Makeup = (Steam * (1 - Condensate Return)) + Blowdown

    where Blowdown = Steam * (Blowdown Rate / 100)

    Args:
        steam_rate_lb_hr: Steam production rate in lb/hr.
        blowdown_rate_percent: Blowdown rate as percentage.
        condensate_return_percent: Percentage of condensate returned.

    Returns:
        Makeup water flow rate in lb/hr.

    Example:
        >>> makeup = calculate_makeup_water_flow(50000, 5, 80)
        >>> print(f"Makeup: {makeup:.0f} lb/hr")
    """
    # Steam lost (not returned as condensate)
    steam_loss = steam_rate_lb_hr * (1 - condensate_return_percent / 100)

    # Blowdown flow
    blowdown_flow = steam_rate_lb_hr * (blowdown_rate_percent / 100)

    # Total makeup
    makeup = steam_loss + blowdown_flow

    logger.debug(
        f"Makeup calculation: steam_loss={steam_loss:.0f}, "
        f"blowdown={blowdown_flow:.0f}, total={makeup:.0f} lb/hr"
    )

    return makeup


def calculate_blowdown_heat_loss(
    blowdown_rate_lb_hr: float,
    blowdown_temperature_f: float,
    makeup_temperature_f: float = 60,
    has_heat_recovery: bool = False,
    heat_recovery_efficiency: float = 0.6
) -> Dict[str, float]:
    """
    Calculate heat loss from boiler blowdown.

    Formula:
        Heat Loss = Blowdown Flow * (T_blowdown - T_makeup) * Cp

    If heat recovery is installed, recovered heat is subtracted.

    Args:
        blowdown_rate_lb_hr: Blowdown flow rate in lb/hr.
        blowdown_temperature_f: Blowdown water temperature in F.
        makeup_temperature_f: Makeup water temperature in F.
        has_heat_recovery: Whether heat recovery equipment is installed.
        heat_recovery_efficiency: Efficiency of heat recovery (0-1).

    Returns:
        Dictionary with heat loss values in BTU/hr and MMBtu/year.

    Example:
        >>> heat = calculate_blowdown_heat_loss(5000, 350, 60)
        >>> print(f"Heat loss: {heat['mmbtu_per_year']:.0f} MMBtu/year")
    """
    # Sensible heat in blowdown water
    temp_diff = blowdown_temperature_f - makeup_temperature_f
    sensible_heat = blowdown_rate_lb_hr * temp_diff * WATER_SPECIFIC_HEAT_BTU_LB_F

    # Flash steam (partial latent heat loss) - approximate 10% at typical pressures
    flash_fraction = 0.10
    flash_heat = blowdown_rate_lb_hr * flash_fraction * LATENT_HEAT_BTU_LB

    # Total heat loss
    total_heat_btu_hr = sensible_heat + flash_heat

    # Apply heat recovery if present
    if has_heat_recovery:
        recovered = total_heat_btu_hr * heat_recovery_efficiency
        net_heat_loss = total_heat_btu_hr - recovered
    else:
        recovered = 0
        net_heat_loss = total_heat_btu_hr

    # Annual values
    annual_mmbtu = (net_heat_loss * HOURS_PER_YEAR) / 1_000_000

    return {
        'heat_loss_btu_hr': round(net_heat_loss, 0),
        'gross_loss_btu_hr': round(total_heat_btu_hr, 0),
        'recovered_btu_hr': round(recovered, 0),
        'mmbtu_per_year': round(annual_mmbtu, 1),
    }


def calculate_blowdown_water_loss(
    blowdown_rate_lb_hr: float,
    operating_hours_per_year: float = 8760
) -> Dict[str, float]:
    """
    Calculate water lost through blowdown.

    Args:
        blowdown_rate_lb_hr: Blowdown flow rate in lb/hr.
        operating_hours_per_year: Annual operating hours.

    Returns:
        Dictionary with water loss in gallons per hour/day/year.

    Example:
        >>> water = calculate_blowdown_water_loss(5000)
        >>> print(f"Water loss: {water['gallons_per_year']:,.0f} gal/year")
    """
    # Convert lb/hr to gal/hr (water density ~ 8.34 lb/gal)
    WATER_DENSITY_LB_GAL = 8.34

    gal_per_hr = blowdown_rate_lb_hr / WATER_DENSITY_LB_GAL
    gal_per_day = gal_per_hr * 24
    gal_per_year = gal_per_hr * operating_hours_per_year

    return {
        'gallons_per_hour': round(gal_per_hr, 1),
        'gallons_per_day': round(gal_per_day, 0),
        'gallons_per_year': round(gal_per_year, 0),
    }


def calculate_blowdown_savings(
    steam_rate_lb_hr: float,
    current_blowdown_percent: float,
    optimal_blowdown_percent: float,
    blowdown_temperature_f: float,
    makeup_temperature_f: float,
    water_cost_per_1000_gal: float,
    fuel_cost_per_mmbtu: float,
    boiler_efficiency: float = 0.82,
    operating_hours_per_year: float = 8760
) -> Dict[str, float]:
    """
    Calculate annual savings from blowdown optimization.

    Compares current blowdown operation with optimal operation
    to determine water, energy, and cost savings.

    Args:
        steam_rate_lb_hr: Steam production rate in lb/hr.
        current_blowdown_percent: Current blowdown rate percentage.
        optimal_blowdown_percent: Recommended optimal rate percentage.
        blowdown_temperature_f: Blowdown water temperature in F.
        makeup_temperature_f: Makeup water temperature in F.
        water_cost_per_1000_gal: Water cost per 1000 gallons.
        fuel_cost_per_mmbtu: Fuel cost per MMBtu.
        boiler_efficiency: Boiler thermal efficiency (0-1).
        operating_hours_per_year: Annual operating hours.

    Returns:
        Dictionary with water, energy, and cost savings.

    Example:
        >>> savings = calculate_blowdown_savings(
        ...     50000, 10, 5, 350, 60, 5.0, 10.0
        ... )
        >>> print(f"Annual savings: ${savings['total_cost_savings']:,.0f}")
    """
    # Current operation
    current_blowdown_lb_hr = steam_rate_lb_hr * (current_blowdown_percent / 100)
    current_water = calculate_blowdown_water_loss(
        current_blowdown_lb_hr, operating_hours_per_year
    )
    current_heat = calculate_blowdown_heat_loss(
        current_blowdown_lb_hr, blowdown_temperature_f, makeup_temperature_f
    )

    # Optimal operation
    optimal_blowdown_lb_hr = steam_rate_lb_hr * (optimal_blowdown_percent / 100)
    optimal_water = calculate_blowdown_water_loss(
        optimal_blowdown_lb_hr, operating_hours_per_year
    )
    optimal_heat = calculate_blowdown_heat_loss(
        optimal_blowdown_lb_hr, blowdown_temperature_f, makeup_temperature_f
    )

    # Water savings
    water_savings_gpy = current_water['gallons_per_year'] - optimal_water['gallons_per_year']
    water_cost_savings = (water_savings_gpy / 1000) * water_cost_per_1000_gal

    # Energy savings (fuel required to heat makeup water)
    energy_savings_mmbtu = current_heat['mmbtu_per_year'] - optimal_heat['mmbtu_per_year']
    # Fuel required = energy / boiler efficiency
    fuel_savings_mmbtu = energy_savings_mmbtu / boiler_efficiency if boiler_efficiency > 0 else 0
    energy_cost_savings = fuel_savings_mmbtu * fuel_cost_per_mmbtu

    # Total savings
    total_savings = water_cost_savings + energy_cost_savings

    # Calculate percentage reduction
    blowdown_reduction_percent = (
        (current_blowdown_percent - optimal_blowdown_percent) / current_blowdown_percent * 100
        if current_blowdown_percent > 0 else 0
    )

    logger.info(
        f"Blowdown optimization savings: water=${water_cost_savings:,.0f}, "
        f"energy=${energy_cost_savings:,.0f}, total=${total_savings:,.0f}/year"
    )

    return {
        'water_savings_gallons_per_year': round(water_savings_gpy, 0),
        'water_cost_savings': round(water_cost_savings, 2),
        'energy_savings_mmbtu_per_year': round(energy_savings_mmbtu, 1),
        'fuel_savings_mmbtu_per_year': round(fuel_savings_mmbtu, 1),
        'energy_cost_savings': round(energy_cost_savings, 2),
        'total_cost_savings': round(total_savings, 2),
        'blowdown_reduction_percent': round(blowdown_reduction_percent, 1),
    }


def determine_optimal_blowdown(
    current_cycles: float,
    optimal_cycles: float,
    current_blowdown_percent: float,
    max_blowdown_percent: float = 10.0,
    min_blowdown_percent: float = 1.0
) -> Dict[str, Any]:
    """
    Determine optimal blowdown rate and adjustment action.

    Compares current operation with optimal and provides
    specific adjustment recommendations.

    Args:
        current_cycles: Current cycles of concentration.
        optimal_cycles: Target optimal cycles.
        current_blowdown_percent: Current blowdown rate.
        max_blowdown_percent: Maximum allowable blowdown rate.
        min_blowdown_percent: Minimum practical blowdown rate.

    Returns:
        Dictionary with optimal rate and adjustment recommendation.

    Example:
        >>> result = determine_optimal_blowdown(5, 10, 20)
        >>> print(result['adjustment_action'])
    """
    # Calculate optimal blowdown rate from cycles
    optimal_blowdown = calculate_blowdown_rate_from_cycles(optimal_cycles)

    # Apply constraints
    optimal_blowdown = max(min_blowdown_percent, min(max_blowdown_percent, optimal_blowdown))

    # Determine adjustment needed
    rate_diff = optimal_blowdown - current_blowdown_percent
    cycles_diff = optimal_cycles - current_cycles

    if abs(rate_diff) < 0.5:
        action = "Maintain current blowdown rate - operating near optimal"
        priority = "LOW"
    elif rate_diff < -2:
        action = f"Reduce blowdown rate from {current_blowdown_percent:.1f}% to {optimal_blowdown:.1f}% to increase COC"
        priority = "HIGH" if rate_diff < -5 else "MEDIUM"
    elif rate_diff < 0:
        action = f"Slightly reduce blowdown rate to {optimal_blowdown:.1f}% for water savings"
        priority = "LOW"
    elif rate_diff > 2:
        action = f"Increase blowdown rate from {current_blowdown_percent:.1f}% to {optimal_blowdown:.1f}% to maintain water quality"
        priority = "HIGH"
    else:
        action = f"Slightly increase blowdown rate to {optimal_blowdown:.1f}% for better water quality control"
        priority = "LOW"

    return {
        'current_rate_percent': round(current_blowdown_percent, 2),
        'optimal_rate_percent': round(optimal_blowdown, 2),
        'rate_change_percent': round(rate_diff, 2),
        'current_cycles': round(current_cycles, 2),
        'optimal_cycles': round(optimal_cycles, 2),
        'adjustment_action': action,
        'adjustment_priority': priority,
    }


def calculate_flash_tank_recovery(
    blowdown_rate_lb_hr: float,
    boiler_pressure_psig: float,
    flash_tank_pressure_psig: float = 15
) -> Dict[str, float]:
    """
    Calculate potential recovery from flash tank on blowdown.

    Flash tanks can recover 10-15% of blowdown energy as
    low-pressure steam.

    Reference: DOE Steam System Best Practices

    Args:
        blowdown_rate_lb_hr: Blowdown flow rate in lb/hr.
        boiler_pressure_psig: Boiler operating pressure in PSIG.
        flash_tank_pressure_psig: Flash tank operating pressure.

    Returns:
        Dictionary with flash steam recovery potential.

    Example:
        >>> recovery = calculate_flash_tank_recovery(5000, 600, 15)
        >>> print(f"Flash steam: {recovery['flash_steam_lb_hr']:.0f} lb/hr")
    """
    # Approximate enthalpy values (simplified correlation)
    # More accurate values would use steam tables

    # Saturation temperature approximation
    def sat_temp_f(p_psig):
        return 212 + (p_psig * 0.55)  # Simplified correlation

    t_boiler = sat_temp_f(boiler_pressure_psig)
    t_flash = sat_temp_f(flash_tank_pressure_psig)

    # Flash fraction (Raoult's law approximation)
    # Flash % = (T_high - T_low) / (T_high - 212) * base_factor
    if t_boiler > 212:
        flash_fraction = min(0.20, (t_boiler - t_flash) / (t_boiler - 212) * 0.15)
    else:
        flash_fraction = 0

    flash_steam_lb_hr = blowdown_rate_lb_hr * flash_fraction
    remaining_water_lb_hr = blowdown_rate_lb_hr - flash_steam_lb_hr

    # Energy recovered (flash steam enthalpy)
    energy_recovered_btu_hr = flash_steam_lb_hr * LATENT_HEAT_BTU_LB
    annual_mmbtu = (energy_recovered_btu_hr * HOURS_PER_YEAR) / 1_000_000

    return {
        'flash_fraction': round(flash_fraction, 3),
        'flash_steam_lb_hr': round(flash_steam_lb_hr, 1),
        'remaining_water_lb_hr': round(remaining_water_lb_hr, 1),
        'energy_recovered_btu_hr': round(energy_recovered_btu_hr, 0),
        'annual_recovery_mmbtu': round(annual_mmbtu, 1),
    }


def calculate_heat_exchanger_recovery(
    blowdown_rate_lb_hr: float,
    blowdown_temperature_f: float,
    makeup_temperature_f: float,
    heat_exchanger_efficiency: float = 0.6
) -> Dict[str, float]:
    """
    Calculate potential recovery from blowdown heat exchanger.

    A heat exchanger can preheat makeup water using blowdown heat.

    Args:
        blowdown_rate_lb_hr: Blowdown flow rate in lb/hr.
        blowdown_temperature_f: Blowdown temperature in F.
        makeup_temperature_f: Incoming makeup temperature in F.
        heat_exchanger_efficiency: Heat exchanger effectiveness (0-1).

    Returns:
        Dictionary with heat recovery potential.

    Example:
        >>> recovery = calculate_heat_exchanger_recovery(5000, 350, 60, 0.6)
        >>> print(f"Recovery: {recovery['annual_mmbtu']:.0f} MMBtu/year")
    """
    # Available heat in blowdown
    available_heat = blowdown_rate_lb_hr * (blowdown_temperature_f - makeup_temperature_f) * WATER_SPECIFIC_HEAT_BTU_LB_F

    # Recovered heat
    recovered_heat = available_heat * heat_exchanger_efficiency

    # Makeup water temperature rise
    # Assuming makeup flow ~ blowdown flow for approximation
    temp_rise = (blowdown_temperature_f - makeup_temperature_f) * heat_exchanger_efficiency

    # Annual recovery
    annual_mmbtu = (recovered_heat * HOURS_PER_YEAR) / 1_000_000

    return {
        'available_heat_btu_hr': round(available_heat, 0),
        'recovered_heat_btu_hr': round(recovered_heat, 0),
        'makeup_temp_rise_f': round(temp_rise, 1),
        'preheated_temp_f': round(makeup_temperature_f + temp_rise, 1),
        'annual_mmbtu': round(annual_mmbtu, 1),
    }


def estimate_blowdown_temperature(
    operating_pressure_psig: float
) -> float:
    """
    Estimate blowdown water temperature from operating pressure.

    Uses saturation temperature correlation.

    Args:
        operating_pressure_psig: Boiler operating pressure in PSIG.

    Returns:
        Estimated saturation temperature in Fahrenheit.

    Example:
        >>> temp = estimate_blowdown_temperature(600)
        >>> print(f"Blowdown temp: {temp:.0f} F")
    """
    # Simplified saturation temperature correlation
    # More accurate: use steam tables or IAPWS-IF97

    if operating_pressure_psig <= 0:
        return 212.0  # Atmospheric

    # Approximate correlation (valid 0-3000 psig)
    # T_sat = 212 + 0.55 * P for low pressures
    # More complex for high pressures

    if operating_pressure_psig < 100:
        temp = 212 + operating_pressure_psig * 0.65
    elif operating_pressure_psig < 500:
        temp = 277 + (operating_pressure_psig - 100) * 0.45
    elif operating_pressure_psig < 1500:
        temp = 457 + (operating_pressure_psig - 500) * 0.25
    else:
        temp = 707 + (operating_pressure_psig - 1500) * 0.08

    return round(temp, 1)


def generate_blowdown_recommendation(
    steam_rate_lb_hr: float,
    current_blowdown_percent: float,
    current_cycles: float,
    optimal_cycles: float,
    operating_pressure_psig: float,
    water_cost: float,
    fuel_cost: float,
    operating_hours: float = 8760
) -> Dict[str, Any]:
    """
    Generate complete blowdown optimization recommendation.

    Combines all calculations into a comprehensive recommendation.

    Args:
        steam_rate_lb_hr: Steam production rate in lb/hr.
        current_blowdown_percent: Current blowdown rate.
        current_cycles: Current cycles of concentration.
        optimal_cycles: Optimal cycles of concentration.
        operating_pressure_psig: Boiler operating pressure.
        water_cost: Water cost per 1000 gallons.
        fuel_cost: Fuel cost per MMBtu.
        operating_hours: Annual operating hours.

    Returns:
        Comprehensive recommendation dictionary.
    """
    # Get blowdown optimization details
    optimization = determine_optimal_blowdown(
        current_cycles,
        optimal_cycles,
        current_blowdown_percent
    )

    optimal_blowdown = optimization['optimal_rate_percent']

    # Estimate blowdown temperature
    blowdown_temp = estimate_blowdown_temperature(operating_pressure_psig)
    makeup_temp = 60  # Standard assumption

    # Calculate savings
    savings = calculate_blowdown_savings(
        steam_rate_lb_hr,
        current_blowdown_percent,
        optimal_blowdown,
        blowdown_temp,
        makeup_temp,
        water_cost,
        fuel_cost,
        operating_hours_per_year=operating_hours
    )

    # Calculate water losses
    current_blowdown_lb_hr = steam_rate_lb_hr * (current_blowdown_percent / 100)
    optimal_blowdown_lb_hr = steam_rate_lb_hr * (optimal_blowdown / 100)

    current_water = calculate_blowdown_water_loss(current_blowdown_lb_hr, operating_hours)
    optimal_water = calculate_blowdown_water_loss(optimal_blowdown_lb_hr, operating_hours)

    return {
        'current_rate_percent': current_blowdown_percent,
        'optimal_rate_percent': optimal_blowdown,
        'cycles_of_concentration': current_cycles,
        'optimal_cycles': optimal_cycles,
        'water_savings_gpy': savings['water_savings_gallons_per_year'],
        'energy_savings_mmbtu_year': savings['energy_savings_mmbtu_per_year'],
        'cost_savings_per_year': savings['total_cost_savings'],
        'adjustment_action': optimization['adjustment_action'],
        'blowdown_temp_f': blowdown_temp,
        'current_water_loss_gpy': current_water['gallons_per_year'],
        'optimal_water_loss_gpy': optimal_water['gallons_per_year'],
    }
