# GL-081: Budget Forecaster Agent (BUDGETFORECASTER)

## Overview

The BudgetForecasterAgent provides multi-year budget forecasting with scenario analysis, variance tracking, and Monte Carlo simulation for energy and sustainability budgets.

## Features

- Multi-year budget forecasting
- Scenario analysis (conservative/moderate/aggressive)
- Historical variance analysis
- Monte Carlo simulation
- Trend-based projections
- Risk assessment

## Quick Start

```python
from backend.agents.gl_081_budget_forecaster import (
    BudgetForecasterAgent,
    BudgetForecasterInput,
    BudgetItem,
    BudgetCategory,
)

agent = BudgetForecasterAgent()
input_data = BudgetForecasterInput(
    budget_name="FY2025 Sustainability Budget",
    fiscal_year_start=2025,
    budget_items=[
        BudgetItem(
            category=BudgetCategory.ENERGY,
            description="Electricity",
            base_amount_usd=500000,
            growth_rate_percent=3,
        ),
    ],
    forecast_years=5,
)
result = agent.run(input_data)
print(f"Expected Budget: ${result.expected_value_forecast.total_forecast_usd:,.2f}")
```

## License

Proprietary - GreenLang Platform
