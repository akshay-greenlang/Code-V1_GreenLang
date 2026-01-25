"""
GL-082: ROI Calculator Formulas Module

Deterministic calculation formulas for ROI analysis.
"""

from typing import Dict, List, Optional, Tuple


def calculate_npv(
    cash_flows: List[float],
    discount_rate: float,
) -> float:
    """
    Calculate Net Present Value.

    ZERO-HALLUCINATION: Standard NPV formula

    Args:
        cash_flows: List of cash flows (year 0 to N)
        discount_rate: Annual discount rate (%)

    Returns:
        NPV value

    Formula:
        NPV = SUM(CF_t / (1 + r)^t) for t = 0 to N
    """
    rate = discount_rate / 100
    npv = sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))
    return round(npv, 2)


def calculate_irr(
    cash_flows: List[float],
    max_iterations: int = 100,
    tolerance: float = 0.0001,
) -> Optional[float]:
    """
    Calculate Internal Rate of Return using Newton-Raphson.

    ZERO-HALLUCINATION: Standard IRR iterative calculation

    Args:
        cash_flows: List of cash flows
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        IRR as percentage, or None if not calculable

    Formula:
        Solves: SUM(CF_t / (1 + IRR)^t) = 0
    """
    rate = 0.10  # Initial guess

    for _ in range(max_iterations):
        npv = sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cash_flows))
        npv_derivative = sum(
            -t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cash_flows)
        )

        if abs(npv_derivative) < 1e-10:
            return None

        new_rate = rate - npv / npv_derivative

        if abs(new_rate - rate) < tolerance:
            return round(new_rate * 100, 2)

        rate = new_rate

    return None


def calculate_payback_period(
    initial_investment: float,
    annual_cash_flows: List[float],
    discounted: bool = False,
    discount_rate: float = 0,
) -> Optional[float]:
    """
    Calculate simple or discounted payback period.

    ZERO-HALLUCINATION: Standard payback calculation

    Args:
        initial_investment: Initial investment (positive value)
        annual_cash_flows: Annual cash inflows (positive values)
        discounted: Use discounted payback
        discount_rate: Discount rate (%) for discounted payback

    Returns:
        Payback period in years, or None if never payback
    """
    cumulative = -initial_investment
    rate = discount_rate / 100

    for year, cf in enumerate(annual_cash_flows, 1):
        if discounted:
            cf = cf / ((1 + rate) ** year)

        cumulative += cf

        if cumulative >= 0:
            # Interpolate
            prev_cumulative = cumulative - cf
            if cf != 0:
                fraction = abs(prev_cumulative) / cf
                return round(year - 1 + fraction, 2)
            return float(year)

    return None


def calculate_roi(
    total_gains: float,
    total_costs: float,
) -> float:
    """
    Calculate Return on Investment percentage.

    ZERO-HALLUCINATION: Standard ROI formula

    Args:
        total_gains: Total returns/gains
        total_costs: Total investment costs

    Returns:
        ROI as percentage

    Formula:
        ROI = (Gains - Costs) / Costs * 100
    """
    if total_costs == 0:
        return 0.0

    roi = (total_gains - total_costs) / total_costs * 100
    return round(roi, 2)


def calculate_mirr(
    cash_flows: List[float],
    finance_rate: float,
    reinvestment_rate: float,
) -> Optional[float]:
    """
    Calculate Modified Internal Rate of Return.

    ZERO-HALLUCINATION: Standard MIRR formula

    Args:
        cash_flows: List of cash flows
        finance_rate: Finance rate for negative flows (%)
        reinvestment_rate: Reinvestment rate for positive flows (%)

    Returns:
        MIRR as percentage, or None if not calculable

    Formula:
        MIRR = (FV_positives / PV_negatives)^(1/n) - 1
    """
    n = len(cash_flows) - 1
    if n <= 0:
        return None

    f_rate = finance_rate / 100
    r_rate = reinvestment_rate / 100

    # PV of negative cash flows at finance rate
    pv_negatives = sum(
        cf / ((1 + f_rate) ** t)
        for t, cf in enumerate(cash_flows) if cf < 0
    )

    # FV of positive cash flows at reinvestment rate
    fv_positives = sum(
        cf * ((1 + r_rate) ** (n - t))
        for t, cf in enumerate(cash_flows) if cf > 0
    )

    if pv_negatives >= 0 or fv_positives <= 0:
        return None

    mirr = ((fv_positives / abs(pv_negatives)) ** (1/n) - 1) * 100
    return round(mirr, 2)


def run_sensitivity_analysis(
    base_npv: float,
    base_parameters: Dict[str, float],
    sensitivity_range: float,
    npv_calculator,  # Callable to recalculate NPV
) -> List[Dict[str, float]]:
    """
    Run sensitivity analysis on parameters.

    ZERO-HALLUCINATION: Standard one-at-a-time sensitivity

    Args:
        base_npv: Base case NPV
        base_parameters: Dictionary of parameter values
        sensitivity_range: +/- range percentage
        npv_calculator: Function to calculate NPV with modified parameters

    Returns:
        List of sensitivity results
    """
    results = []

    for param, base_value in base_parameters.items():
        low_value = base_value * (1 - sensitivity_range/100)
        high_value = base_value * (1 + sensitivity_range/100)

        low_npv = npv_calculator(**{param: low_value})
        high_npv = npv_calculator(**{param: high_value})

        spread = abs(high_npv - low_npv)
        sensitivity_index = spread / abs(base_npv) if base_npv != 0 else 0

        results.append({
            "parameter": param,
            "base_value": base_value,
            "low_value": low_value,
            "high_value": high_value,
            "low_npv": low_npv,
            "high_npv": high_npv,
            "spread": spread,
            "sensitivity_index": round(sensitivity_index, 4),
        })

    # Sort by sensitivity index
    results.sort(key=lambda x: x["sensitivity_index"], reverse=True)
    return results


def calculate_profitability_index(
    pv_cash_inflows: float,
    initial_investment: float,
) -> float:
    """
    Calculate Profitability Index.

    ZERO-HALLUCINATION: Standard PI formula

    Args:
        pv_cash_inflows: Present value of cash inflows
        initial_investment: Initial investment

    Returns:
        Profitability Index

    Formula:
        PI = PV of Cash Inflows / Initial Investment
    """
    if initial_investment == 0:
        return 0.0

    return round(pv_cash_inflows / initial_investment, 2)


def calculate_lcoe(
    total_lifecycle_cost: float,
    total_energy_produced_kwh: float,
    discount_rate: float,
    project_life_years: int,
) -> float:
    """
    Calculate Levelized Cost of Energy.

    ZERO-HALLUCINATION: Standard LCOE formula

    Args:
        total_lifecycle_cost: Total project cost (NPV)
        total_energy_produced_kwh: Total energy production (kWh)
        discount_rate: Discount rate (%)
        project_life_years: Project lifetime

    Returns:
        LCOE in $/kWh

    Formula:
        LCOE = Total_Lifecycle_Cost / Total_Energy_Production
    """
    if total_energy_produced_kwh == 0:
        return 0.0

    lcoe = total_lifecycle_cost / total_energy_produced_kwh
    return round(lcoe, 4)
