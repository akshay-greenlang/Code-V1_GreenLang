"""
GL-080: OPEX Optimizer Formulas Module

Deterministic calculation formulas for operational cost analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class OpexSummary:
    """Summary of operating expenditure."""
    energy_cost: float
    maintenance_cost: float
    labor_cost: float
    other_cost: float
    total_opex: float


@dataclass
class MaintenanceSchedule:
    """Optimized maintenance schedule."""
    equipment_name: str
    current_frequency: str
    optimized_frequency: str
    current_annual_cost: float
    optimized_annual_cost: float
    annual_savings: float


def calculate_annual_opex(
    energy_costs: Dict[str, float],
    maintenance_costs: Dict[str, float],
    labor_costs: Dict[str, float],
    other_costs: Dict[str, float],
) -> OpexSummary:
    """
    Calculate total annual operating expenditure.

    ZERO-HALLUCINATION: Simple summation

    Args:
        energy_costs: Energy cost components
        maintenance_costs: Maintenance cost components
        labor_costs: Labor cost components
        other_costs: Other operating costs

    Returns:
        OpexSummary with all cost components
    """
    energy_total = sum(energy_costs.values())
    maintenance_total = sum(maintenance_costs.values())
    labor_total = sum(labor_costs.values())
    other_total = sum(other_costs.values())

    return OpexSummary(
        energy_cost=round(energy_total, 2),
        maintenance_cost=round(maintenance_total, 2),
        labor_cost=round(labor_total, 2),
        other_cost=round(other_total, 2),
        total_opex=round(energy_total + maintenance_total + labor_total + other_total, 2),
    )


def calculate_maintenance_schedule(
    equipment_name: str,
    current_frequency_hours: int,
    failure_rate: float,
    repair_cost: float,
    pm_cost: float,
) -> MaintenanceSchedule:
    """
    Calculate optimal maintenance schedule.

    ZERO-HALLUCINATION: Standard PM optimization formula

    Args:
        equipment_name: Equipment identifier
        current_frequency_hours: Current PM interval
        failure_rate: Failures per operating hour
        repair_cost: Cost of unplanned repair
        pm_cost: Cost of preventive maintenance

    Returns:
        MaintenanceSchedule with optimization recommendations

    Formula:
        Optimal_Interval = sqrt(2 * PM_Cost / (Failure_Rate * Repair_Cost))
    """
    import math

    if failure_rate <= 0 or repair_cost <= 0:
        return MaintenanceSchedule(
            equipment_name=equipment_name,
            current_frequency="Current",
            optimized_frequency="No change",
            current_annual_cost=pm_cost,
            optimized_annual_cost=pm_cost,
            annual_savings=0,
        )

    # Calculate optimal PM interval
    optimal_hours = math.sqrt(2 * pm_cost / (failure_rate * repair_cost))

    # Annual operating hours
    annual_hours = 8760

    # Current cost (PM + expected failures)
    current_pm_events = annual_hours / current_frequency_hours
    current_failures = failure_rate * annual_hours
    current_cost = (current_pm_events * pm_cost) + (current_failures * repair_cost)

    # Optimized cost
    optimized_pm_events = annual_hours / optimal_hours
    optimized_cost = optimized_pm_events * pm_cost + (failure_rate * annual_hours * 0.5 * repair_cost)

    return MaintenanceSchedule(
        equipment_name=equipment_name,
        current_frequency=f"{current_frequency_hours} hours",
        optimized_frequency=f"{round(optimal_hours)} hours",
        current_annual_cost=round(current_cost, 2),
        optimized_annual_cost=round(optimized_cost, 2),
        annual_savings=round(max(0, current_cost - optimized_cost), 2),
    )


def calculate_energy_cost(
    consumption_kwh: float,
    energy_rate: float,
    demand_kw: float = 0,
    demand_rate: float = 0,
    power_factor: float = 1.0,
    pf_penalty_threshold: float = 0.90,
) -> Dict[str, float]:
    """
    Calculate energy cost with demand and power factor.

    ZERO-HALLUCINATION: Standard utility billing calculation

    Args:
        consumption_kwh: Energy consumption (kWh)
        energy_rate: Energy rate ($/kWh)
        demand_kw: Peak demand (kW)
        demand_rate: Demand rate ($/kW)
        power_factor: Power factor
        pf_penalty_threshold: PF threshold for penalty

    Returns:
        Dictionary with cost breakdown
    """
    energy_cost = consumption_kwh * energy_rate
    demand_cost = demand_kw * demand_rate

    # Power factor penalty
    pf_penalty = 0
    if power_factor < pf_penalty_threshold and power_factor > 0:
        pf_multiplier = pf_penalty_threshold / power_factor
        pf_penalty = demand_cost * (pf_multiplier - 1)

    return {
        "energy_cost": round(energy_cost, 2),
        "demand_cost": round(demand_cost, 2),
        "pf_penalty": round(pf_penalty, 2),
        "total_cost": round(energy_cost + demand_cost + pf_penalty, 2),
    }


def calculate_labor_optimization(
    current_fte: float,
    annual_salary: float,
    benefits_percent: float,
    automation_potential_percent: float,
    automation_cost: float,
) -> Dict[str, float]:
    """
    Calculate labor cost optimization through automation.

    ZERO-HALLUCINATION: Standard ROI calculation

    Args:
        current_fte: Current full-time equivalents
        annual_salary: Annual salary per FTE
        benefits_percent: Benefits as % of salary
        automation_potential_percent: % of work automatable
        automation_cost: One-time automation investment

    Returns:
        Dictionary with labor optimization analysis
    """
    total_labor_cost = current_fte * annual_salary * (1 + benefits_percent/100)
    savings_potential = total_labor_cost * (automation_potential_percent/100)
    payback_years = automation_cost / savings_potential if savings_potential > 0 else float('inf')

    return {
        "current_labor_cost": round(total_labor_cost, 2),
        "annual_savings": round(savings_potential, 2),
        "automation_cost": round(automation_cost, 2),
        "payback_years": round(payback_years, 2),
        "optimized_fte": round(current_fte * (1 - automation_potential_percent/100), 1),
    }


def project_opex_savings(
    baseline_opex: float,
    annual_savings: float,
    years: int,
    escalation_rate: float = 2.5,
    discount_rate: float = 8.0,
) -> Dict[str, float]:
    """
    Project OPEX savings over time.

    ZERO-HALLUCINATION: Standard NPV calculation

    Args:
        baseline_opex: Current annual OPEX
        annual_savings: Annual savings from optimization
        years: Analysis period
        escalation_rate: Annual cost escalation (%)
        discount_rate: Discount rate for NPV (%)

    Returns:
        Dictionary with savings projections
    """
    total_savings = 0
    npv_savings = 0

    for year in range(1, years + 1):
        # Escalated savings (savings grow with inflation)
        escalated_savings = annual_savings * ((1 + escalation_rate/100) ** (year - 1))
        total_savings += escalated_savings

        # Discounted savings
        npv_savings += escalated_savings / ((1 + discount_rate/100) ** year)

    return {
        "nominal_savings": round(total_savings, 2),
        "npv_savings": round(npv_savings, 2),
        "average_annual_savings": round(total_savings / years, 2),
        "lifetime_opex_reduction_percent": round(annual_savings / baseline_opex * 100, 1) if baseline_opex > 0 else 0,
    }
