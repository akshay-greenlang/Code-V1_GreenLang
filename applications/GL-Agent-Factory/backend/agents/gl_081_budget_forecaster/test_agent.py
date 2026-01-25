"""GL-081: Budget Forecaster Agent - Test Suite"""

import pytest
from .agent import (
    BudgetForecasterAgent,
    BudgetForecasterInput,
    BudgetItem,
    BudgetCategory,
)
from .formulas import (
    calculate_budget_forecast,
    calculate_variance,
    run_monte_carlo_forecast,
    calculate_trend_projection,
    calculate_cagr,
)


@pytest.fixture
def agent():
    return BudgetForecasterAgent()


@pytest.fixture
def basic_input():
    return BudgetForecasterInput(
        budget_name="FY2025 Energy Budget",
        fiscal_year_start=2025,
        budget_items=[
            BudgetItem(
                category=BudgetCategory.ENERGY,
                description="Electricity",
                base_amount_usd=500000,
                growth_rate_percent=3,
            ),
            BudgetItem(
                category=BudgetCategory.MAINTENANCE,
                description="Equipment Maintenance",
                base_amount_usd=100000,
            ),
        ],
        forecast_years=5,
    )


class TestAgentInitialization:
    def test_agent_creates(self, agent):
        assert agent.AGENT_ID == "GL-081"

    def test_run_returns_output(self, agent, basic_input):
        result = agent.run(basic_input)
        assert len(result.forecasts) > 0


class TestFormulas:
    def test_budget_forecast(self):
        results = calculate_budget_forecast(100000, 5, 5)
        assert len(results) == 5
        assert results[0].base_forecast > 100000

    def test_variance_favorable(self):
        variance, pct, status = calculate_variance(100000, 90000)
        assert status == "FAVORABLE"
        assert pct < 0

    def test_variance_unfavorable(self):
        variance, pct, status = calculate_variance(100000, 115000)
        assert status == "UNFAVORABLE"

    def test_monte_carlo(self):
        result = run_monte_carlo_forecast(100000, 5, 2, 5, 100, seed=42)
        assert result["mean"] > 0
        assert result["p10"] < result["p90"]

    def test_trend_projection(self):
        historical = [100, 110, 120, 130]
        projections = calculate_trend_projection(historical, 3)
        assert len(projections) == 3
        assert projections[0] > 130

    def test_cagr(self):
        cagr = calculate_cagr(100, 161.05, 5)
        assert abs(cagr - 10) < 0.5  # ~10% CAGR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
