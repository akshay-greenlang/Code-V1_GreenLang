"""
Economic Thickness Calculator

This module implements economic thickness optimization for industrial insulation
using deterministic engineering economics formulas. Finds the optimal insulation
thickness that minimizes total cost (insulation cost + energy cost).

Economic thickness methodology follows industry standards:
- ASTM C680 Standard Practice for Estimate of Heat Gain/Loss
- NAIMA 3E Plus methodology
- DOE Industrial Insulation Program guidelines

All calculations are deterministic - no ML/LLM in the calculation path.
This ensures zero-hallucination compliance for regulatory calculations.

Example:
    >>> from economic_thickness import calculate_economic_thickness
    >>> result = calculate_economic_thickness(
    ...     t_hot_c=300, t_ambient_c=25,
    ...     k_insulation=0.04, energy_cost_per_kwh=0.10,
    ...     insulation_cost_per_m3=300, pipe_diameter_m=0.1
    ... )
    >>> print(f"Economic thickness: {result['economic_thickness_mm']:.0f} mm")
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .heat_loss import (
    calculate_flat_surface_heat_loss,
    calculate_cylindrical_heat_loss,
    calculate_bare_surface_heat_loss,
    calculate_annual_energy_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class EconomicThicknessResult:
    """
    Results from economic thickness calculation.

    Attributes:
        economic_thickness_m: Optimal insulation thickness in meters.
        economic_thickness_mm: Optimal thickness in millimeters.
        economic_thickness_inches: Optimal thickness in inches.
        minimum_total_cost: Minimum annual total cost at optimal thickness.
        annual_energy_cost: Annual energy cost at optimal thickness.
        annual_insulation_cost: Annualized insulation cost at optimal thickness.
        heat_loss_at_optimum_w: Heat loss at optimal thickness.
        roi_years: Simple payback period in years.
        npv: Net present value of insulation investment.
        irr_percent: Internal rate of return.
        energy_savings_percent: Energy savings vs bare surface.
        thickness_analysis: Analysis at various thicknesses.
    """
    economic_thickness_m: float
    economic_thickness_mm: float
    economic_thickness_inches: float
    minimum_total_cost: float
    annual_energy_cost: float
    annual_insulation_cost: float
    heat_loss_at_optimum_w: float
    roi_years: float
    npv: float
    irr_percent: float
    energy_savings_percent: float
    thickness_analysis: List[Dict[str, float]]


# Economic calculation parameters
DEFAULT_PARAMS = {
    "discount_rate": 0.08,  # 8% discount rate
    "project_life_years": 20,  # 20 year project life
    "maintenance_factor": 0.02,  # 2% annual maintenance
    "installation_factor": 1.5,  # Installation cost factor
    "inflation_rate": 0.03,  # 3% annual energy price inflation
    "tax_rate": 0.25,  # 25% corporate tax rate
}


def calculate_annualized_cost(
    capital_cost: float,
    discount_rate: float,
    project_life_years: int
) -> float:
    """
    Convert capital cost to annualized cost using capital recovery factor.

    Formula:
        CRF = r * (1 + r)^n / ((1 + r)^n - 1)
        Annual_Cost = Capital * CRF

    Args:
        capital_cost: Initial capital investment.
        discount_rate: Annual discount rate (0-1).
        project_life_years: Project life in years.

    Returns:
        Annualized capital cost.
    """
    if discount_rate <= 0:
        return capital_cost / project_life_years

    r = discount_rate
    n = project_life_years

    # Capital recovery factor
    crf = r * ((1 + r) ** n) / (((1 + r) ** n) - 1)

    return capital_cost * crf


def calculate_npv(
    annual_savings: float,
    capital_cost: float,
    discount_rate: float,
    project_life_years: int
) -> float:
    """
    Calculate Net Present Value of insulation investment.

    Formula:
        NPV = sum(Annual_Savings / (1+r)^t) - Capital_Cost

    Args:
        annual_savings: Annual energy savings.
        capital_cost: Initial investment.
        discount_rate: Discount rate.
        project_life_years: Project life.

    Returns:
        Net present value.
    """
    r = discount_rate
    n = project_life_years

    if r <= 0:
        pv_savings = annual_savings * n
    else:
        # Present value of annuity
        pv_savings = annual_savings * (1 - (1 + r) ** (-n)) / r

    return pv_savings - capital_cost


def calculate_simple_payback(
    capital_cost: float,
    annual_savings: float
) -> float:
    """
    Calculate simple payback period.

    Args:
        capital_cost: Initial investment.
        annual_savings: Annual savings.

    Returns:
        Payback period in years.
    """
    if annual_savings <= 0:
        return float('inf')

    return capital_cost / annual_savings


def estimate_irr(
    capital_cost: float,
    annual_savings: float,
    project_life_years: int
) -> float:
    """
    Estimate Internal Rate of Return using iterative method.

    Args:
        capital_cost: Initial investment.
        annual_savings: Annual savings.
        project_life_years: Project life.

    Returns:
        IRR as percentage.
    """
    if capital_cost <= 0 or annual_savings <= 0:
        return 0.0

    # Use bisection method to find IRR
    low = 0.0
    high = 1.0  # 100% IRR as upper bound

    for _ in range(50):  # 50 iterations for convergence
        mid = (low + high) / 2
        npv = calculate_npv(annual_savings, capital_cost, mid, project_life_years)

        if abs(npv) < 0.01:
            return mid * 100  # Convert to percentage

        if npv > 0:
            low = mid
        else:
            high = mid

    return ((low + high) / 2) * 100


def calculate_insulation_cost(
    thickness_m: float,
    area_m2: float,
    insulation_cost_per_m3: float,
    installation_factor: float = 1.5
) -> float:
    """
    Calculate total installed insulation cost.

    Args:
        thickness_m: Insulation thickness in meters.
        area_m2: Surface area in square meters.
        insulation_cost_per_m3: Material cost per cubic meter.
        installation_factor: Installation labor factor (1.5 = 50% adder).

    Returns:
        Total installed cost.
    """
    volume_m3 = thickness_m * area_m2
    material_cost = volume_m3 * insulation_cost_per_m3
    installed_cost = material_cost * installation_factor

    return installed_cost


def calculate_annual_energy_cost(
    heat_loss_w: float,
    energy_cost_per_kwh: float,
    operating_hours_per_year: float = 8760,
    boiler_efficiency: float = 0.85
) -> float:
    """
    Calculate annual energy cost from heat loss.

    Args:
        heat_loss_w: Heat loss in Watts.
        energy_cost_per_kwh: Energy cost per kWh.
        operating_hours_per_year: Annual operating hours.
        boiler_efficiency: Boiler/heater efficiency (0-1).

    Returns:
        Annual energy cost.
    """
    # Energy consumed (accounting for boiler efficiency)
    energy_kwh = (heat_loss_w / 1000) * operating_hours_per_year / boiler_efficiency

    return energy_kwh * energy_cost_per_kwh


def calculate_economic_thickness_flat(
    t_hot_c: float,
    t_ambient_c: float,
    k_insulation: float,
    area_m2: float,
    energy_cost_per_kwh: float,
    insulation_cost_per_m3: float,
    operating_hours_per_year: float = 8760,
    discount_rate: float = 0.08,
    project_life_years: int = 20,
    boiler_efficiency: float = 0.85,
    installation_factor: float = 1.5,
    wind_speed_m_s: float = 0.0,
    emissivity: float = 0.9,
    min_thickness_m: float = 0.025,
    max_thickness_m: float = 0.300,
    thickness_increment_m: float = 0.0125
) -> EconomicThicknessResult:
    """
    Calculate economic insulation thickness for flat surface.

    Uses iterative optimization to find thickness that minimizes total cost.

    Args:
        t_hot_c: Hot surface temperature in Celsius.
        t_ambient_c: Ambient temperature in Celsius.
        k_insulation: Thermal conductivity in W/m-K.
        area_m2: Surface area in square meters.
        energy_cost_per_kwh: Energy cost per kWh.
        insulation_cost_per_m3: Material cost per cubic meter.
        operating_hours_per_year: Annual operating hours.
        discount_rate: Discount rate for economic analysis.
        project_life_years: Project life in years.
        boiler_efficiency: Heat source efficiency.
        installation_factor: Installation cost multiplier.
        wind_speed_m_s: Wind speed.
        emissivity: Surface emissivity.
        min_thickness_m: Minimum thickness to consider.
        max_thickness_m: Maximum thickness to consider.
        thickness_increment_m: Increment for analysis.

    Returns:
        EconomicThicknessResult with optimal thickness and economics.
    """
    # Calculate bare surface heat loss (baseline)
    bare_result = calculate_bare_surface_heat_loss(
        t_hot_c=t_hot_c,
        t_ambient_c=t_ambient_c,
        area_m2=area_m2,
        wind_speed_m_s=wind_speed_m_s,
        emissivity=emissivity
    )
    bare_heat_loss = bare_result.heat_loss_w
    bare_energy_cost = calculate_annual_energy_cost(
        bare_heat_loss, energy_cost_per_kwh, operating_hours_per_year, boiler_efficiency
    )

    # Analyze various thicknesses
    thickness_analysis = []
    best_thickness = min_thickness_m
    min_total_cost = float('inf')

    thickness = min_thickness_m
    while thickness <= max_thickness_m + 0.0001:
        # Calculate heat loss at this thickness
        result = calculate_flat_surface_heat_loss(
            t_hot_c=t_hot_c,
            t_ambient_c=t_ambient_c,
            insulation_thickness_m=thickness,
            k_insulation=k_insulation,
            area_m2=area_m2,
            wind_speed_m_s=wind_speed_m_s,
            emissivity=emissivity
        )

        heat_loss = result.heat_loss_w

        # Annual energy cost
        energy_cost = calculate_annual_energy_cost(
            heat_loss, energy_cost_per_kwh, operating_hours_per_year, boiler_efficiency
        )

        # Insulation capital cost
        capital_cost = calculate_insulation_cost(
            thickness, area_m2, insulation_cost_per_m3, installation_factor
        )

        # Annualized insulation cost
        annual_insulation_cost = calculate_annualized_cost(
            capital_cost, discount_rate, project_life_years
        )

        # Total annual cost
        total_cost = energy_cost + annual_insulation_cost

        # Annual savings vs bare
        annual_savings = bare_energy_cost - energy_cost

        # Economic metrics
        npv = calculate_npv(annual_savings, capital_cost, discount_rate, project_life_years)
        payback = calculate_simple_payback(capital_cost, annual_savings)

        # Store analysis
        thickness_analysis.append({
            "thickness_mm": thickness * 1000,
            "heat_loss_w": heat_loss,
            "annual_energy_cost": energy_cost,
            "annual_insulation_cost": annual_insulation_cost,
            "total_annual_cost": total_cost,
            "annual_savings": annual_savings,
            "capital_cost": capital_cost,
            "npv": npv,
            "payback_years": payback,
            "surface_temp_c": result.surface_temp_c,
        })

        # Track optimum
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_thickness = thickness

        thickness += thickness_increment_m

    # Get results at optimal thickness
    optimal_analysis = None
    for analysis in thickness_analysis:
        if abs(analysis["thickness_mm"] - best_thickness * 1000) < 1:
            optimal_analysis = analysis
            break

    if optimal_analysis is None:
        optimal_analysis = thickness_analysis[0]

    # Calculate final metrics
    energy_savings_percent = (
        (bare_energy_cost - optimal_analysis["annual_energy_cost"]) / bare_energy_cost * 100
        if bare_energy_cost > 0 else 0
    )

    irr = estimate_irr(
        optimal_analysis["capital_cost"],
        optimal_analysis["annual_savings"],
        project_life_years
    )

    logger.info(
        f"Economic thickness: {best_thickness * 1000:.0f}mm, "
        f"Total cost: ${min_total_cost:.0f}/yr, "
        f"Energy savings: {energy_savings_percent:.1f}%"
    )

    return EconomicThicknessResult(
        economic_thickness_m=best_thickness,
        economic_thickness_mm=best_thickness * 1000,
        economic_thickness_inches=best_thickness * 39.37,
        minimum_total_cost=min_total_cost,
        annual_energy_cost=optimal_analysis["annual_energy_cost"],
        annual_insulation_cost=optimal_analysis["annual_insulation_cost"],
        heat_loss_at_optimum_w=optimal_analysis["heat_loss_w"],
        roi_years=optimal_analysis["payback_years"],
        npv=optimal_analysis["npv"],
        irr_percent=irr,
        energy_savings_percent=energy_savings_percent,
        thickness_analysis=thickness_analysis,
    )


def calculate_economic_thickness_pipe(
    t_hot_c: float,
    t_ambient_c: float,
    k_insulation: float,
    pipe_outer_diameter_m: float,
    pipe_length_m: float,
    energy_cost_per_kwh: float,
    insulation_cost_per_m3: float,
    operating_hours_per_year: float = 8760,
    discount_rate: float = 0.08,
    project_life_years: int = 20,
    boiler_efficiency: float = 0.85,
    installation_factor: float = 1.5,
    wind_speed_m_s: float = 0.0,
    emissivity: float = 0.9,
    min_thickness_m: float = 0.025,
    max_thickness_m: float = 0.150,
    thickness_increment_m: float = 0.0125
) -> EconomicThicknessResult:
    """
    Calculate economic insulation thickness for pipe.

    Uses iterative optimization for cylindrical geometry.

    Args:
        t_hot_c: Hot surface temperature in Celsius.
        t_ambient_c: Ambient temperature in Celsius.
        k_insulation: Thermal conductivity in W/m-K.
        pipe_outer_diameter_m: Pipe outer diameter in meters.
        pipe_length_m: Pipe length in meters.
        energy_cost_per_kwh: Energy cost per kWh.
        insulation_cost_per_m3: Material cost per cubic meter.
        operating_hours_per_year: Annual operating hours.
        discount_rate: Discount rate.
        project_life_years: Project life.
        boiler_efficiency: Heat source efficiency.
        installation_factor: Installation cost multiplier.
        wind_speed_m_s: Wind speed.
        emissivity: Surface emissivity.
        min_thickness_m: Minimum thickness.
        max_thickness_m: Maximum thickness.
        thickness_increment_m: Analysis increment.

    Returns:
        EconomicThicknessResult with optimal thickness and economics.
    """
    pipe_radius = pipe_outer_diameter_m / 2

    # Bare pipe surface area (for baseline)
    bare_area = 2 * math.pi * pipe_radius * pipe_length_m

    # Calculate bare surface heat loss
    bare_result = calculate_bare_surface_heat_loss(
        t_hot_c=t_hot_c,
        t_ambient_c=t_ambient_c,
        area_m2=bare_area,
        wind_speed_m_s=wind_speed_m_s,
        emissivity=emissivity
    )
    bare_heat_loss = bare_result.heat_loss_w
    bare_energy_cost = calculate_annual_energy_cost(
        bare_heat_loss, energy_cost_per_kwh, operating_hours_per_year, boiler_efficiency
    )

    # Analyze various thicknesses
    thickness_analysis = []
    best_thickness = min_thickness_m
    min_total_cost = float('inf')

    thickness = min_thickness_m
    while thickness <= max_thickness_m + 0.0001:
        # Calculate heat loss for pipe with insulation
        result = calculate_cylindrical_heat_loss(
            t_hot_c=t_hot_c,
            t_ambient_c=t_ambient_c,
            pipe_outer_radius_m=pipe_radius,
            insulation_thickness_m=thickness,
            k_insulation=k_insulation,
            pipe_length_m=pipe_length_m,
            wind_speed_m_s=wind_speed_m_s,
            emissivity=emissivity
        )

        heat_loss = result.heat_loss_w

        # Annual energy cost
        energy_cost = calculate_annual_energy_cost(
            heat_loss, energy_cost_per_kwh, operating_hours_per_year, boiler_efficiency
        )

        # Insulation volume (cylindrical shell)
        r_inner = pipe_radius
        r_outer = pipe_radius + thickness
        volume_m3 = math.pi * (r_outer ** 2 - r_inner ** 2) * pipe_length_m

        # Capital cost
        material_cost = volume_m3 * insulation_cost_per_m3
        capital_cost = material_cost * installation_factor

        # Annualized insulation cost
        annual_insulation_cost = calculate_annualized_cost(
            capital_cost, discount_rate, project_life_years
        )

        # Total annual cost
        total_cost = energy_cost + annual_insulation_cost

        # Annual savings vs bare
        annual_savings = bare_energy_cost - energy_cost

        # Economic metrics
        npv = calculate_npv(annual_savings, capital_cost, discount_rate, project_life_years)
        payback = calculate_simple_payback(capital_cost, annual_savings)

        # Store analysis
        thickness_analysis.append({
            "thickness_mm": thickness * 1000,
            "heat_loss_w": heat_loss,
            "annual_energy_cost": energy_cost,
            "annual_insulation_cost": annual_insulation_cost,
            "total_annual_cost": total_cost,
            "annual_savings": annual_savings,
            "capital_cost": capital_cost,
            "npv": npv,
            "payback_years": payback,
            "surface_temp_c": result.surface_temp_c,
        })

        # Track optimum
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_thickness = thickness

        thickness += thickness_increment_m

    # Get results at optimal thickness
    optimal_analysis = None
    for analysis in thickness_analysis:
        if abs(analysis["thickness_mm"] - best_thickness * 1000) < 1:
            optimal_analysis = analysis
            break

    if optimal_analysis is None:
        optimal_analysis = thickness_analysis[0]

    # Final metrics
    energy_savings_percent = (
        (bare_energy_cost - optimal_analysis["annual_energy_cost"]) / bare_energy_cost * 100
        if bare_energy_cost > 0 else 0
    )

    irr = estimate_irr(
        optimal_analysis["capital_cost"],
        optimal_analysis["annual_savings"],
        project_life_years
    )

    logger.info(
        f"Pipe economic thickness: {best_thickness * 1000:.0f}mm, "
        f"Total cost: ${min_total_cost:.0f}/yr, "
        f"Energy savings: {energy_savings_percent:.1f}%"
    )

    return EconomicThicknessResult(
        economic_thickness_m=best_thickness,
        economic_thickness_mm=best_thickness * 1000,
        economic_thickness_inches=best_thickness * 39.37,
        minimum_total_cost=min_total_cost,
        annual_energy_cost=optimal_analysis["annual_energy_cost"],
        annual_insulation_cost=optimal_analysis["annual_insulation_cost"],
        heat_loss_at_optimum_w=optimal_analysis["heat_loss_w"],
        roi_years=optimal_analysis["payback_years"],
        npv=optimal_analysis["npv"],
        irr_percent=irr,
        energy_savings_percent=energy_savings_percent,
        thickness_analysis=thickness_analysis,
    )


def calculate_economic_thickness(
    t_hot_c: float,
    t_ambient_c: float,
    k_insulation: float,
    energy_cost_per_kwh: float,
    insulation_cost_per_m3: float,
    surface_type: str = "flat",
    area_m2: Optional[float] = None,
    pipe_diameter_m: Optional[float] = None,
    pipe_length_m: Optional[float] = None,
    **kwargs
) -> EconomicThicknessResult:
    """
    Calculate economic insulation thickness (unified interface).

    Automatically selects flat or cylindrical calculation based on inputs.

    Args:
        t_hot_c: Hot surface temperature in Celsius.
        t_ambient_c: Ambient temperature in Celsius.
        k_insulation: Thermal conductivity in W/m-K.
        energy_cost_per_kwh: Energy cost per kWh.
        insulation_cost_per_m3: Material cost per cubic meter.
        surface_type: 'flat', 'pipe', or 'cylindrical'.
        area_m2: Surface area for flat surfaces.
        pipe_diameter_m: Pipe diameter for cylindrical.
        pipe_length_m: Pipe length for cylindrical.
        **kwargs: Additional parameters passed to calculation functions.

    Returns:
        EconomicThicknessResult.

    Example:
        >>> result = calculate_economic_thickness(
        ...     t_hot_c=300, t_ambient_c=25, k_insulation=0.04,
        ...     energy_cost_per_kwh=0.10, insulation_cost_per_m3=300,
        ...     surface_type="pipe", pipe_diameter_m=0.1, pipe_length_m=100
        ... )
    """
    if surface_type.lower() in ["pipe", "cylindrical"]:
        if pipe_diameter_m is None or pipe_length_m is None:
            raise ValueError("pipe_diameter_m and pipe_length_m required for pipe calculation")

        return calculate_economic_thickness_pipe(
            t_hot_c=t_hot_c,
            t_ambient_c=t_ambient_c,
            k_insulation=k_insulation,
            pipe_outer_diameter_m=pipe_diameter_m,
            pipe_length_m=pipe_length_m,
            energy_cost_per_kwh=energy_cost_per_kwh,
            insulation_cost_per_m3=insulation_cost_per_m3,
            **kwargs
        )
    else:
        if area_m2 is None:
            raise ValueError("area_m2 required for flat surface calculation")

        return calculate_economic_thickness_flat(
            t_hot_c=t_hot_c,
            t_ambient_c=t_ambient_c,
            k_insulation=k_insulation,
            area_m2=area_m2,
            energy_cost_per_kwh=energy_cost_per_kwh,
            insulation_cost_per_m3=insulation_cost_per_m3,
            **kwargs
        )


def compare_materials_economically(
    t_hot_c: float,
    t_ambient_c: float,
    materials: List[Tuple[str, float, float]],  # (name, k_value, cost_per_m3)
    surface_type: str,
    area_m2: Optional[float] = None,
    pipe_diameter_m: Optional[float] = None,
    pipe_length_m: Optional[float] = None,
    energy_cost_per_kwh: float = 0.10,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Compare multiple insulation materials economically.

    Args:
        t_hot_c: Hot surface temperature.
        t_ambient_c: Ambient temperature.
        materials: List of (name, k_value, cost_per_m3) tuples.
        surface_type: 'flat' or 'pipe'.
        area_m2: Surface area for flat.
        pipe_diameter_m: Pipe diameter.
        pipe_length_m: Pipe length.
        energy_cost_per_kwh: Energy cost.
        **kwargs: Additional calculation parameters.

    Returns:
        List of comparison results sorted by total cost.
    """
    comparisons = []

    for name, k_value, cost_per_m3 in materials:
        try:
            result = calculate_economic_thickness(
                t_hot_c=t_hot_c,
                t_ambient_c=t_ambient_c,
                k_insulation=k_value,
                energy_cost_per_kwh=energy_cost_per_kwh,
                insulation_cost_per_m3=cost_per_m3,
                surface_type=surface_type,
                area_m2=area_m2,
                pipe_diameter_m=pipe_diameter_m,
                pipe_length_m=pipe_length_m,
                **kwargs
            )

            comparisons.append({
                "material_name": name,
                "k_value": k_value,
                "cost_per_m3": cost_per_m3,
                "economic_thickness_mm": result.economic_thickness_mm,
                "minimum_total_cost": result.minimum_total_cost,
                "annual_energy_cost": result.annual_energy_cost,
                "npv": result.npv,
                "roi_years": result.roi_years,
                "energy_savings_percent": result.energy_savings_percent,
            })

        except Exception as e:
            logger.warning(f"Failed to analyze material {name}: {e}")
            continue

    # Sort by minimum total cost
    comparisons.sort(key=lambda x: x["minimum_total_cost"])

    return comparisons


def calculate_roi_analysis(
    capital_cost: float,
    annual_savings: float,
    project_life_years: int = 20,
    discount_rate: float = 0.08,
    energy_escalation_rate: float = 0.03
) -> Dict[str, Any]:
    """
    Comprehensive ROI analysis for insulation investment.

    Args:
        capital_cost: Initial investment.
        annual_savings: First year annual savings.
        project_life_years: Analysis period.
        discount_rate: Discount rate.
        energy_escalation_rate: Annual energy price increase.

    Returns:
        Dictionary with comprehensive ROI metrics.
    """
    # Simple payback
    simple_payback = calculate_simple_payback(capital_cost, annual_savings)

    # NPV with escalating savings
    npv = 0
    cumulative_savings = 0
    discounted_payback = float('inf')

    for year in range(1, project_life_years + 1):
        # Savings escalate with energy prices
        year_savings = annual_savings * ((1 + energy_escalation_rate) ** (year - 1))

        # Discounted savings
        discounted_savings = year_savings / ((1 + discount_rate) ** year)
        npv += discounted_savings
        cumulative_savings += discounted_savings

        # Track when we recover investment
        if cumulative_savings >= capital_cost and discounted_payback == float('inf'):
            # Linear interpolation for fractional year
            prev_cumulative = cumulative_savings - discounted_savings
            fraction = (capital_cost - prev_cumulative) / discounted_savings
            discounted_payback = year - 1 + fraction

    # Final NPV
    npv -= capital_cost

    # IRR
    irr = estimate_irr(capital_cost, annual_savings, project_life_years)

    # Total lifetime savings (nominal)
    total_nominal_savings = sum(
        annual_savings * ((1 + energy_escalation_rate) ** (y - 1))
        for y in range(1, project_life_years + 1)
    )

    return {
        "simple_payback_years": simple_payback,
        "discounted_payback_years": discounted_payback,
        "npv": npv,
        "irr_percent": irr,
        "capital_cost": capital_cost,
        "first_year_savings": annual_savings,
        "total_nominal_savings": total_nominal_savings,
        "roi_percent": ((total_nominal_savings - capital_cost) / capital_cost * 100)
            if capital_cost > 0 else 0,
        "profitability_index": (npv + capital_cost) / capital_cost if capital_cost > 0 else 0,
    }
