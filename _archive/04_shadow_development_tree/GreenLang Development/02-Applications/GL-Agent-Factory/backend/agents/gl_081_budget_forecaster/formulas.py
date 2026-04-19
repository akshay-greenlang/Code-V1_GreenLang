"""
GL-081: Budget Forecaster Formulas Module

Deterministic calculation formulas for budget forecasting.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ForecastResult:
    """Budget forecast result."""
    year: int
    base_forecast: float
    low_bound: float
    high_bound: float
    confidence_level: float


def calculate_budget_forecast(
    base_amount: float,
    growth_rate: float,
    years: int,
    inflation_rate: float = 2.5,
) -> List[ForecastResult]:
    """
    Calculate budget forecast with confidence intervals.

    ZERO-HALLUCINATION: Compound growth formula

    Args:
        base_amount: Starting budget amount
        growth_rate: Annual growth rate (%)
        years: Number of years to forecast
        inflation_rate: Annual inflation rate (%)

    Returns:
        List of ForecastResult for each year

    Formula:
        Forecast_Y = Base * (1 + Growth + Inflation)^Y
    """
    results = []
    combined_rate = (growth_rate + inflation_rate) / 100

    for year in range(1, years + 1):
        forecast = base_amount * ((1 + combined_rate) ** year)

        # Confidence interval widens with time
        confidence = 0.95 - (0.02 * year)  # Decreasing confidence
        variance = forecast * (0.05 + 0.02 * year)  # Increasing variance

        results.append(ForecastResult(
            year=year,
            base_forecast=round(forecast, 2),
            low_bound=round(forecast - variance, 2),
            high_bound=round(forecast + variance, 2),
            confidence_level=round(confidence, 2),
        ))

    return results


def calculate_variance(
    budgeted: float,
    actual: float,
) -> Tuple[float, float, str]:
    """
    Calculate budget variance.

    ZERO-HALLUCINATION: Standard variance calculation

    Args:
        budgeted: Budgeted amount
        actual: Actual amount

    Returns:
        Tuple of (variance_amount, variance_percent, status)

    Formula:
        Variance = Actual - Budgeted
        Variance% = (Actual - Budgeted) / Budgeted * 100
    """
    variance = actual - budgeted
    variance_pct = (variance / budgeted * 100) if budgeted != 0 else 0

    if variance_pct < -5:
        status = "FAVORABLE"  # Under budget
    elif variance_pct > 5:
        status = "UNFAVORABLE"  # Over budget
    else:
        status = "ON_TRACK"

    return round(variance, 2), round(variance_pct, 1), status


def run_monte_carlo_forecast(
    base_amount: float,
    mean_growth: float,
    std_growth: float,
    years: int,
    iterations: int = 1000,
    seed: int = None,
) -> Dict[str, float]:
    """
    Run Monte Carlo simulation for budget forecast.

    ZERO-HALLUCINATION: Standard Monte Carlo simulation

    Args:
        base_amount: Starting budget amount
        mean_growth: Mean annual growth rate (%)
        std_growth: Standard deviation of growth (%)
        years: Number of years
        iterations: Number of Monte Carlo iterations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with simulation statistics
    """
    if seed is not None:
        random.seed(seed)

    final_values = []

    for _ in range(iterations):
        value = base_amount
        for _ in range(years):
            growth = random.gauss(mean_growth, std_growth) / 100
            value = value * (1 + growth)
        final_values.append(value)

    final_values.sort()

    return {
        "mean": round(sum(final_values) / iterations, 2),
        "median": round(final_values[iterations // 2], 2),
        "p10": round(final_values[int(iterations * 0.10)], 2),
        "p25": round(final_values[int(iterations * 0.25)], 2),
        "p75": round(final_values[int(iterations * 0.75)], 2),
        "p90": round(final_values[int(iterations * 0.90)], 2),
        "std_dev": round((sum((x - sum(final_values)/iterations)**2 for x in final_values) / iterations)**0.5, 2),
    }


def calculate_trend_projection(
    historical_values: List[float],
    years_to_project: int,
) -> List[float]:
    """
    Calculate trend-based projection using linear regression.

    ZERO-HALLUCINATION: Simple linear regression

    Args:
        historical_values: Historical budget values
        years_to_project: Number of years to project

    Returns:
        List of projected values

    Formula:
        y = a + b*x (linear regression)
        b = SUM((x-x_mean)(y-y_mean)) / SUM((x-x_mean)^2)
        a = y_mean - b * x_mean
    """
    n = len(historical_values)
    if n < 2:
        return [historical_values[-1] if historical_values else 0] * years_to_project

    # Calculate means
    x_values = list(range(n))
    x_mean = sum(x_values) / n
    y_mean = sum(historical_values) / n

    # Calculate slope and intercept
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, historical_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    slope = numerator / denominator if denominator != 0 else 0
    intercept = y_mean - slope * x_mean

    # Project future values
    projections = []
    for i in range(years_to_project):
        future_x = n + i
        projected_value = intercept + slope * future_x
        projections.append(round(max(0, projected_value), 2))  # No negative budgets

    return projections


def calculate_cagr(
    start_value: float,
    end_value: float,
    years: int,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    ZERO-HALLUCINATION: Standard CAGR formula

    Args:
        start_value: Starting value
        end_value: Ending value
        years: Number of years

    Returns:
        CAGR as percentage

    Formula:
        CAGR = (End_Value / Start_Value)^(1/Years) - 1
    """
    if start_value <= 0 or years <= 0:
        return 0.0

    cagr = ((end_value / start_value) ** (1 / years) - 1) * 100
    return round(cagr, 2)
