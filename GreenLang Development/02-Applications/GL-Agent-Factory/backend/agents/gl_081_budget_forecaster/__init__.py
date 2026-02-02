"""GL-081: Budget Forecaster Agent (BUDGETFORECASTER)"""

from .agent import (
    BudgetForecasterAgent,
    BudgetForecasterInput,
    BudgetForecasterOutput,
    BudgetItem,
    ForecastScenario,
    BudgetForecast,
    VarianceAnalysis,
    BudgetCategory,
    ForecastMethod,
    PACK_SPEC,
)

from .formulas import (
    calculate_budget_forecast,
    calculate_variance,
    run_monte_carlo_forecast,
    calculate_trend_projection,
)

__all__ = [
    "BudgetForecasterAgent",
    "BudgetForecasterInput",
    "BudgetForecasterOutput",
    "BudgetItem",
    "ForecastScenario",
    "BudgetForecast",
    "VarianceAnalysis",
    "BudgetCategory",
    "ForecastMethod",
    "PACK_SPEC",
    "calculate_budget_forecast",
    "calculate_variance",
    "run_monte_carlo_forecast",
    "calculate_trend_projection",
]

__version__ = "1.0.0"
__agent_id__ = "GL-081"
__agent_name__ = "BUDGETFORECASTER"
