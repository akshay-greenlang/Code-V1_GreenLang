"""
GL-079: CAPEX Analyzer Formulas Module

This module contains all deterministic calculation formulas for capital
expenditure analysis and cost estimation.

Formula Sources:
- RSMeans construction cost methodologies
- NREL cost benchmarking studies
- Standard project cost estimation practices

Example:
    >>> from formulas import calculate_total_capex
    >>> total = calculate_total_capex(equipment=100000, installation=50000)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CapexSummary:
    """Summary of capital expenditure calculation."""
    equipment_cost: float
    installation_cost: float
    soft_cost: float
    contingency: float
    total_capex: float
    cost_per_unit: Optional[float]


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a parameter."""
    parameter_name: str
    base_case: float
    low_case: float
    high_case: float
    tornado_spread: float
    sensitivity_rank: int


@dataclass
class CostBreakdown:
    """Detailed cost breakdown result."""
    category: str
    amount: float
    percentage: float
    subcategories: List[Dict[str, float]]


# =============================================================================
# TOTAL CAPEX CALCULATION
# =============================================================================

def calculate_total_capex(
    equipment_cost: float,
    installation_cost: float,
    soft_cost: float = 0.0,
    contingency_percent: float = 10.0,
    escalation_percent: float = 0.0,
) -> CapexSummary:
    """
    Calculate total capital expenditure.

    ZERO-HALLUCINATION: Standard CAPEX formula

    Args:
        equipment_cost: Total equipment/material cost
        installation_cost: Total installation/labor cost
        soft_cost: Soft costs (engineering, permitting, etc.)
        contingency_percent: Contingency percentage (0-50)
        escalation_percent: Cost escalation for future projects

    Returns:
        CapexSummary with all cost components

    Formula:
        Subtotal = Equipment + Installation + Soft
        Contingency = Subtotal * Contingency%
        Escalation = (Subtotal + Contingency) * Escalation%
        Total = Subtotal + Contingency + Escalation
    """
    subtotal = equipment_cost + installation_cost + soft_cost
    contingency = subtotal * (contingency_percent / 100)

    pre_escalation = subtotal + contingency
    escalation = pre_escalation * (escalation_percent / 100)

    total = pre_escalation + escalation

    return CapexSummary(
        equipment_cost=round(equipment_cost, 2),
        installation_cost=round(installation_cost, 2),
        soft_cost=round(soft_cost, 2),
        contingency=round(contingency, 2),
        total_capex=round(total, 2),
        cost_per_unit=None,
    )


def calculate_cost_per_unit(
    total_capex: float,
    capacity: float,
    unit: str = "kW",
) -> Tuple[float, str]:
    """
    Calculate cost per unit of capacity.

    ZERO-HALLUCINATION: Simple division

    Args:
        total_capex: Total capital expenditure
        capacity: Project capacity
        unit: Capacity unit (kW, MW, tons, sqft)

    Returns:
        Tuple of (cost_per_unit, unit_string)

    Formula:
        Cost/Unit = Total_CAPEX / Capacity
    """
    if capacity <= 0:
        return 0.0, f"$/[{unit}]"

    cost_per_unit = total_capex / capacity
    return round(cost_per_unit, 2), f"$/{unit}"


# =============================================================================
# CONTINGENCY CALCULATIONS
# =============================================================================

def calculate_contingency(
    base_cost: float,
    project_complexity: str = "MEDIUM",
    technology_maturity: str = "MATURE",
    custom_percent: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate appropriate contingency based on project factors.

    ZERO-HALLUCINATION: Industry-standard contingency guidelines

    Args:
        base_cost: Base project cost before contingency
        project_complexity: LOW, MEDIUM, HIGH, VERY_HIGH
        technology_maturity: EMERGING, DEVELOPING, MATURE, PROVEN
        custom_percent: Override with custom percentage

    Returns:
        Tuple of (contingency_amount, contingency_percent)

    Reference:
        AACE International Recommended Practice 18R-97
        - Conceptual: 25-50%
        - Budget: 15-30%
        - Definitive: 5-15%
    """
    if custom_percent is not None:
        percent = custom_percent
    else:
        # Base contingency by complexity
        complexity_base = {
            "LOW": 5.0,
            "MEDIUM": 10.0,
            "HIGH": 15.0,
            "VERY_HIGH": 25.0,
        }

        # Technology maturity adjustment
        maturity_adj = {
            "EMERGING": 10.0,
            "DEVELOPING": 5.0,
            "MATURE": 0.0,
            "PROVEN": -2.0,
        }

        base_pct = complexity_base.get(project_complexity.upper(), 10.0)
        adj = maturity_adj.get(technology_maturity.upper(), 0.0)
        percent = max(5.0, base_pct + adj)  # Minimum 5%

    contingency_amount = base_cost * (percent / 100)
    return round(contingency_amount, 2), round(percent, 1)


# =============================================================================
# INSTALLED COST CALCULATIONS
# =============================================================================

def calculate_installed_cost(
    equipment_cost: float,
    installation_factor: float = 1.5,
) -> float:
    """
    Calculate total installed cost using installation factor.

    ZERO-HALLUCINATION: Standard installation factor method

    Args:
        equipment_cost: Base equipment cost
        installation_factor: Multiplier for installation (typically 1.3-2.5)

    Returns:
        Total installed cost

    Formula:
        Installed_Cost = Equipment_Cost * Installation_Factor

    Typical Factors (RSMeans):
        - Simple equipment: 1.3
        - Standard equipment: 1.5
        - Complex equipment: 2.0
        - Process equipment: 2.5
    """
    return round(equipment_cost * installation_factor, 2)


def calculate_installation_labor(
    equipment_cost: float,
    labor_percent: float = 35.0,
    labor_rate_per_hour: float = 75.0,
) -> Tuple[float, float]:
    """
    Calculate installation labor cost and hours.

    ZERO-HALLUCINATION: Standard labor estimation

    Args:
        equipment_cost: Base equipment cost
        labor_percent: Labor as % of equipment (typically 25-50%)
        labor_rate_per_hour: Blended labor rate

    Returns:
        Tuple of (labor_cost, estimated_hours)

    Formula:
        Labor_Cost = Equipment_Cost * Labor_Percent
        Hours = Labor_Cost / Labor_Rate
    """
    labor_cost = equipment_cost * (labor_percent / 100)
    hours = labor_cost / labor_rate_per_hour if labor_rate_per_hour > 0 else 0

    return round(labor_cost, 2), round(hours, 1)


# =============================================================================
# SOFT COST CALCULATIONS
# =============================================================================

def calculate_soft_costs(
    hard_costs: float,
    engineering_percent: float = 8.0,
    permitting_percent: float = 3.0,
    project_management_percent: float = 5.0,
    financing_costs: float = 0.0,
    other_soft_costs: float = 0.0,
) -> Dict[str, float]:
    """
    Calculate soft costs as percentages of hard costs.

    ZERO-HALLUCINATION: Industry-standard soft cost ratios

    Args:
        hard_costs: Total hard costs (equipment + installation)
        engineering_percent: Engineering as % of hard costs
        permitting_percent: Permitting as % of hard costs
        project_management_percent: PM as % of hard costs
        financing_costs: Fixed financing costs
        other_soft_costs: Other fixed soft costs

    Returns:
        Dictionary of soft cost components

    Typical Ranges:
        - Engineering: 5-12%
        - Permitting: 2-5%
        - Project Management: 3-8%
    """
    engineering = hard_costs * (engineering_percent / 100)
    permitting = hard_costs * (permitting_percent / 100)
    project_mgmt = hard_costs * (project_management_percent / 100)

    total = engineering + permitting + project_mgmt + financing_costs + other_soft_costs

    return {
        "engineering": round(engineering, 2),
        "permitting": round(permitting, 2),
        "project_management": round(project_mgmt, 2),
        "financing": round(financing_costs, 2),
        "other": round(other_soft_costs, 2),
        "total_soft_costs": round(total, 2),
        "soft_cost_percent": round(total / hard_costs * 100, 1) if hard_costs > 0 else 0,
    }


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity_analysis(
    base_costs: Dict[str, float],
    variation_percent: float = 20.0,
    parameters: Optional[List[str]] = None,
) -> List[SensitivityResult]:
    """
    Run sensitivity analysis on cost parameters.

    ZERO-HALLUCINATION: Standard sensitivity calculation

    Args:
        base_costs: Dictionary of cost components
        variation_percent: +/- variation for analysis
        parameters: Specific parameters to analyze (None = all)

    Returns:
        List of SensitivityResult sorted by impact

    Formula:
        Low_Case = Base * (1 - Variation%)
        High_Case = Base * (1 + Variation%)
        Spread = High_Case - Low_Case
    """
    base_total = sum(base_costs.values())
    params_to_analyze = parameters or list(base_costs.keys())

    results = []

    for param in params_to_analyze:
        if param not in base_costs:
            continue

        base_value = base_costs[param]
        low_value = base_value * (1 - variation_percent / 100)
        high_value = base_value * (1 + variation_percent / 100)

        # Calculate total with low and high values
        low_total = base_total - base_value + low_value
        high_total = base_total - base_value + high_value

        spread = high_total - low_total

        results.append(SensitivityResult(
            parameter_name=param,
            base_case=round(base_value, 2),
            low_case=round(low_value, 2),
            high_case=round(high_value, 2),
            tornado_spread=round(spread, 2),
            sensitivity_rank=0,  # Will be set after sorting
        ))

    # Sort by spread (descending) and assign ranks
    results.sort(key=lambda x: x.tornado_spread, reverse=True)
    for i, result in enumerate(results):
        result.sensitivity_rank = i + 1

    return results


# =============================================================================
# COST ESCALATION
# =============================================================================

def calculate_escalation(
    current_cost: float,
    years: int,
    annual_escalation_rate: float = 3.0,
) -> Tuple[float, float]:
    """
    Calculate cost escalation over time.

    ZERO-HALLUCINATION: Standard compound escalation

    Args:
        current_cost: Current cost
        years: Number of years to escalate
        annual_escalation_rate: Annual escalation percentage

    Returns:
        Tuple of (escalated_cost, escalation_amount)

    Formula:
        Escalated_Cost = Current_Cost * (1 + Rate)^Years
    """
    escalation_factor = (1 + annual_escalation_rate / 100) ** years
    escalated_cost = current_cost * escalation_factor
    escalation_amount = escalated_cost - current_cost

    return round(escalated_cost, 2), round(escalation_amount, 2)


def calculate_regional_adjustment(
    base_cost: float,
    location_factor: float = 1.0,
) -> float:
    """
    Adjust costs for regional differences.

    ZERO-HALLUCINATION: RSMeans location factor methodology

    Args:
        base_cost: National average cost
        location_factor: Regional adjustment factor

    Returns:
        Regionally adjusted cost

    Typical Location Factors:
        - National Average: 1.00
        - San Francisco: 1.25
        - New York: 1.20
        - Houston: 0.90
        - Phoenix: 0.95

    Formula:
        Adjusted_Cost = Base_Cost * Location_Factor
    """
    return round(base_cost * location_factor, 2)


# =============================================================================
# BENCHMARKING CALCULATIONS
# =============================================================================

def calculate_benchmark_percentile(
    project_cost_per_unit: float,
    benchmark_low: float,
    benchmark_median: float,
    benchmark_high: float,
) -> Tuple[float, str]:
    """
    Calculate project cost percentile against benchmarks.

    ZERO-HALLUCINATION: Standard percentile calculation

    Args:
        project_cost_per_unit: Project's cost per unit
        benchmark_low: 25th percentile benchmark
        benchmark_median: 50th percentile benchmark
        benchmark_high: 75th percentile benchmark

    Returns:
        Tuple of (percentile, status)

    Status:
        - BELOW: Project cost below 25th percentile
        - LOW: Between 25th and 50th percentile
        - MEDIAN: Near 50th percentile
        - HIGH: Between 50th and 75th percentile
        - ABOVE: Above 75th percentile
    """
    if project_cost_per_unit <= benchmark_low:
        percentile = (project_cost_per_unit / benchmark_low) * 25
        status = "BELOW"
    elif project_cost_per_unit <= benchmark_median:
        range_size = benchmark_median - benchmark_low
        position = project_cost_per_unit - benchmark_low
        percentile = 25 + (position / range_size) * 25
        status = "LOW"
    elif project_cost_per_unit <= benchmark_high:
        range_size = benchmark_high - benchmark_median
        position = project_cost_per_unit - benchmark_median
        percentile = 50 + (position / range_size) * 25
        status = "HIGH"
    else:
        overage = project_cost_per_unit - benchmark_high
        percentile = min(75 + (overage / benchmark_high) * 25, 100)
        status = "ABOVE"

    return round(percentile, 1), status


# =============================================================================
# FINANCING CALCULATIONS
# =============================================================================

def calculate_financing_costs(
    principal: float,
    interest_rate: float,
    term_years: int,
    loan_fees_percent: float = 2.0,
) -> Dict[str, float]:
    """
    Calculate project financing costs.

    ZERO-HALLUCINATION: Standard loan calculations

    Args:
        principal: Loan principal amount
        interest_rate: Annual interest rate (%)
        term_years: Loan term in years
        loan_fees_percent: Origination fees (%)

    Returns:
        Dictionary with financing cost components

    Formula:
        Monthly_Payment = P * [r(1+r)^n] / [(1+r)^n - 1]
        Total_Interest = (Monthly_Payment * 12 * Years) - Principal
    """
    if interest_rate <= 0 or term_years <= 0:
        return {
            "loan_fees": round(principal * loan_fees_percent / 100, 2),
            "total_interest": 0.0,
            "monthly_payment": 0.0,
            "total_payments": principal,
        }

    # Monthly rate and number of payments
    monthly_rate = interest_rate / 100 / 12
    num_payments = term_years * 12

    # Monthly payment formula
    if monthly_rate > 0:
        monthly_payment = principal * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / (
            (1 + monthly_rate) ** num_payments - 1
        )
    else:
        monthly_payment = principal / num_payments

    total_payments = monthly_payment * num_payments
    total_interest = total_payments - principal
    loan_fees = principal * (loan_fees_percent / 100)

    return {
        "loan_fees": round(loan_fees, 2),
        "total_interest": round(total_interest, 2),
        "monthly_payment": round(monthly_payment, 2),
        "total_payments": round(total_payments, 2),
        "total_financing_cost": round(loan_fees + total_interest, 2),
    }
