"""
VFD Energy Savings Calculator

Physics-based calculations for VFD energy savings, payback analysis,
and comparison of control methods.

The key insight is that for centrifugal loads (fans, pumps):
    Power ~ Speed^3

So reducing speed by 20% reduces power by ~50%!

References:
    - DOE Motor Challenge Program
    - IEC 61800-9: Energy Efficiency of VFD Systems
    - ASHRAE 90.1: Energy Standard for Buildings
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def calculate_vfd_energy_savings(
    rated_power_kw: float,
    current_flow_pct: float,
    current_method: str = "throttling",
    rated_speed_rpm: float = 1800,
    operating_hours_year: int = 8000,
    motor_efficiency_pct: float = 93.0,
    vfd_efficiency_pct: float = 97.0,
    load_type: str = "variable_torque"
) -> Dict[str, float]:
    """
    Calculate energy savings from VFD vs current control method.

    Control methods compared:
    - Throttling: Valve/damper restricts flow (very inefficient)
    - Bypass: Recirculation (moderately inefficient)
    - Outlet damper: Fan outlet damper
    - Inlet vane: Fan inlet guide vanes
    - VFD: Variable speed control (most efficient for variable loads)

    Args:
        rated_power_kw: Motor rated power.
        current_flow_pct: Current operating flow as % of rated.
        current_method: Current flow control method.
        rated_speed_rpm: Motor rated speed.
        operating_hours_year: Annual operating hours.
        motor_efficiency_pct: Motor efficiency.
        vfd_efficiency_pct: VFD efficiency.
        load_type: "variable_torque" or "constant_torque".

    Returns:
        Dictionary with energy analysis results.

    Example:
        >>> savings = calculate_vfd_energy_savings(
        ...     75, 70, "throttling", 1800, 8000, 93, 97
        ... )
        >>> print(f"Annual savings: {savings['annual_savings_kwh']:,.0f} kWh")
    """
    # Calculate power for current method
    flow_ratio = current_flow_pct / 100

    # Power consumption by control method
    method_factors = {
        "throttling": _throttling_power_factor(flow_ratio),
        "bypass": _bypass_power_factor(flow_ratio),
        "outlet_damper": _damper_power_factor(flow_ratio),
        "inlet_vane": _inlet_vane_power_factor(flow_ratio),
        "vfd": flow_ratio ** 3 if load_type == "variable_torque" else flow_ratio,
    }

    current_factor = method_factors.get(current_method, method_factors["throttling"])
    vfd_factor = method_factors["vfd"]

    # Current power consumption
    current_power_kw = rated_power_kw * current_factor

    # VFD power consumption (including VFD and motor efficiency)
    # VFD power = (shaft power) / motor_eff / vfd_eff
    shaft_power_vfd = rated_power_kw * vfd_factor
    vfd_input_power_kw = shaft_power_vfd / (motor_efficiency_pct / 100) / (vfd_efficiency_pct / 100)

    # For fair comparison, current method also needs motor efficiency
    current_input_power_kw = current_power_kw / (motor_efficiency_pct / 100)

    # Power savings
    power_savings_kw = current_input_power_kw - vfd_input_power_kw
    savings_pct = (power_savings_kw / current_input_power_kw * 100) if current_input_power_kw > 0 else 0

    # Annual energy
    current_annual_kwh = current_input_power_kw * operating_hours_year
    vfd_annual_kwh = vfd_input_power_kw * operating_hours_year
    annual_savings_kwh = current_annual_kwh - vfd_annual_kwh

    return {
        "current_method": current_method,
        "flow_pct": current_flow_pct,
        "current_power_factor": round(current_factor, 4),
        "vfd_power_factor": round(vfd_factor, 4),
        "current_input_power_kw": round(current_input_power_kw, 2),
        "vfd_input_power_kw": round(vfd_input_power_kw, 2),
        "power_savings_kw": round(power_savings_kw, 2),
        "savings_percentage": round(savings_pct, 1),
        "current_annual_kwh": round(current_annual_kwh, 0),
        "vfd_annual_kwh": round(vfd_annual_kwh, 0),
        "annual_savings_kwh": round(annual_savings_kwh, 0),
    }


def _throttling_power_factor(flow_ratio: float) -> float:
    """Power factor for throttling control (valve/damper restriction)."""
    # Throttling wastes energy as pressure drop across valve
    # Power stays nearly constant (~95% of rated at 50% flow)
    return 0.95 - 0.45 * (1 - flow_ratio)


def _bypass_power_factor(flow_ratio: float) -> float:
    """Power factor for bypass/recirculation control."""
    # Part of flow is recirculated, pump/fan runs at full power
    return 1.0 - 0.2 * (1 - flow_ratio)


def _damper_power_factor(flow_ratio: float) -> float:
    """Power factor for outlet damper control."""
    # Better than throttling, follows approximate curve
    return 0.8 + 0.2 * flow_ratio ** 2


def _inlet_vane_power_factor(flow_ratio: float) -> float:
    """Power factor for inlet guide vane control (fans)."""
    # Pre-rotational swirl reduces power requirement
    return 0.4 + 0.6 * flow_ratio ** 1.5


def calculate_payback_period(
    vfd_cost_usd: float,
    annual_savings_kwh: float,
    electricity_price_usd_kwh: float,
    installation_cost_usd: float = 0,
    maintenance_savings_usd_year: float = 0
) -> Dict[str, float]:
    """
    Calculate VFD investment payback period.

    Args:
        vfd_cost_usd: VFD equipment cost.
        annual_savings_kwh: Annual energy savings in kWh.
        electricity_price_usd_kwh: Electricity price per kWh.
        installation_cost_usd: Installation/commissioning cost.
        maintenance_savings_usd_year: Annual maintenance cost reduction (soft starts, etc.).

    Returns:
        Dictionary with payback analysis.

    Example:
        >>> payback = calculate_payback_period(8000, 50000, 0.10, 2000, 500)
        >>> print(f"Payback period: {payback['simple_payback_years']:.1f} years")
    """
    total_investment = vfd_cost_usd + installation_cost_usd

    annual_energy_savings_usd = annual_savings_kwh * electricity_price_usd_kwh
    total_annual_savings = annual_energy_savings_usd + maintenance_savings_usd_year

    if total_annual_savings <= 0:
        return {
            "total_investment": total_investment,
            "annual_energy_savings_usd": 0,
            "total_annual_savings_usd": 0,
            "simple_payback_years": float('inf'),
            "note": "No savings calculated"
        }

    simple_payback = total_investment / total_annual_savings

    # Calculate NPV and IRR (simplified 10-year analysis at 8% discount)
    discount_rate = 0.08
    years = 10
    npv = -total_investment
    for year in range(1, years + 1):
        npv += total_annual_savings / ((1 + discount_rate) ** year)

    # Approximate IRR using trial method
    irr = _calculate_irr(total_investment, total_annual_savings, years)

    return {
        "total_investment_usd": round(total_investment, 2),
        "annual_energy_savings_usd": round(annual_energy_savings_usd, 2),
        "annual_maintenance_savings_usd": round(maintenance_savings_usd_year, 2),
        "total_annual_savings_usd": round(total_annual_savings, 2),
        "simple_payback_years": round(simple_payback, 2),
        "npv_10_year_usd": round(npv, 2),
        "irr_pct": round(irr * 100, 1),
        "recommended": simple_payback < 3,
    }


def _calculate_irr(investment: float, annual_cashflow: float, years: int) -> float:
    """Calculate IRR using iterative method."""
    # Simple bisection method
    low, high = -0.5, 1.0

    for _ in range(50):
        mid = (low + high) / 2
        npv = -investment
        for year in range(1, years + 1):
            npv += annual_cashflow / ((1 + mid) ** year)

        if abs(npv) < 0.01:
            return mid
        elif npv > 0:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def calculate_annual_operating_cost(
    power_kw: float,
    operating_hours_year: int,
    electricity_price_usd_kwh: float,
    demand_charge_usd_kw_month: float = 0,
    power_factor_penalty: float = 0
) -> Dict[str, float]:
    """
    Calculate total annual operating cost including demand charges.

    Args:
        power_kw: Operating power in kW.
        operating_hours_year: Annual operating hours.
        electricity_price_usd_kwh: Energy price per kWh.
        demand_charge_usd_kw_month: Monthly demand charge per kW.
        power_factor_penalty: Annual PF penalty cost.

    Returns:
        Dictionary with cost breakdown.
    """
    energy_cost = power_kw * operating_hours_year * electricity_price_usd_kwh
    demand_cost = power_kw * demand_charge_usd_kw_month * 12

    total_cost = energy_cost + demand_cost + power_factor_penalty

    return {
        "average_power_kw": round(power_kw, 2),
        "annual_energy_kwh": round(power_kw * operating_hours_year, 0),
        "energy_cost_usd": round(energy_cost, 2),
        "demand_cost_usd": round(demand_cost, 2),
        "power_factor_penalty_usd": round(power_factor_penalty, 2),
        "total_annual_cost_usd": round(total_cost, 2),
    }


def compare_control_methods(
    rated_power_kw: float,
    load_profile: List[Tuple[float, float]],
    electricity_price_usd_kwh: float = 0.10,
    operating_hours_year: int = 8000
) -> Dict[str, Dict]:
    """
    Compare different flow control methods for a given load profile.

    Load profile is list of (flow_pct, hours_at_flow) tuples.

    Args:
        rated_power_kw: Motor rated power.
        load_profile: List of (flow%, hours) tuples.
        electricity_price_usd_kwh: Electricity price.
        operating_hours_year: Total operating hours (for verification).

    Returns:
        Dictionary with comparison for each control method.

    Example:
        >>> profile = [(100, 2000), (80, 3000), (60, 2000), (40, 1000)]
        >>> comparison = compare_control_methods(75, profile, 0.10, 8000)
    """
    methods = ["throttling", "bypass", "outlet_damper", "inlet_vane", "vfd"]
    results = {}

    for method in methods:
        total_kwh = 0
        weighted_power = 0
        total_hours = 0

        for flow_pct, hours in load_profile:
            flow_ratio = flow_pct / 100

            # Calculate power factor for this method
            if method == "throttling":
                factor = _throttling_power_factor(flow_ratio)
            elif method == "bypass":
                factor = _bypass_power_factor(flow_ratio)
            elif method == "outlet_damper":
                factor = _damper_power_factor(flow_ratio)
            elif method == "inlet_vane":
                factor = _inlet_vane_power_factor(flow_ratio)
            else:  # VFD
                factor = flow_ratio ** 3

            power_kw = rated_power_kw * factor
            energy_kwh = power_kw * hours

            total_kwh += energy_kwh
            weighted_power += power_kw * hours
            total_hours += hours

        avg_power = weighted_power / total_hours if total_hours > 0 else 0
        annual_cost = total_kwh * electricity_price_usd_kwh

        results[method] = {
            "total_energy_kwh": round(total_kwh, 0),
            "average_power_kw": round(avg_power, 2),
            "annual_cost_usd": round(annual_cost, 2),
        }

    # Calculate savings vs throttling
    baseline_cost = results["throttling"]["annual_cost_usd"]
    for method, data in results.items():
        savings = baseline_cost - data["annual_cost_usd"]
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        results[method]["savings_vs_throttling_usd"] = round(savings, 2)
        results[method]["savings_pct"] = round(savings_pct, 1)

    return results
