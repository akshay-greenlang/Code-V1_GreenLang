"""GL-082: ROI Calculator Agent - Test Suite"""

import pytest
from .agent import (
    ROICalculatorAgent,
    ROICalculatorInput,
    CashFlow,
    InvestmentType,
    CashFlowType,
)
from .formulas import (
    calculate_npv,
    calculate_irr,
    calculate_payback_period,
    calculate_roi,
    calculate_mirr,
    calculate_profitability_index,
)


@pytest.fixture
def agent():
    return ROICalculatorAgent()


@pytest.fixture
def basic_input():
    return ROICalculatorInput(
        project_name="LED Retrofit Project",
        investment_type=InvestmentType.EFFICIENCY,
        initial_investment_usd=100000,
        cash_flows=[
            CashFlow(
                year=1,
                amount_usd=30000,
                flow_type=CashFlowType.ENERGY_SAVINGS,
                is_recurring=True,
            ),
        ],
        analysis_period_years=5,
        discount_rate_percent=8,
    )


class TestAgentInitialization:
    def test_agent_creates(self, agent):
        assert agent.AGENT_ID == "GL-082"

    def test_run_returns_output(self, agent, basic_input):
        result = agent.run(basic_input)
        assert result.roi_metrics.npv_usd is not None


class TestFormulas:
    def test_npv_positive(self):
        # Investment of 100, returns of 50 each year for 3 years
        cash_flows = [-100, 50, 50, 50]
        npv = calculate_npv(cash_flows, 10)
        assert npv > 0

    def test_npv_at_zero_rate(self):
        cash_flows = [-100, 50, 50, 50]
        npv = calculate_npv(cash_flows, 0)
        assert npv == 50  # -100 + 150 = 50

    def test_irr_calculation(self):
        cash_flows = [-100, 50, 50, 50]
        irr = calculate_irr(cash_flows)
        assert irr is not None
        assert 20 < irr < 30  # Should be around 23%

    def test_simple_payback(self):
        payback = calculate_payback_period(100, [30, 30, 30, 30])
        assert payback is not None
        assert 3 < payback < 4  # Between 3-4 years

    def test_discounted_payback(self):
        payback = calculate_payback_period(100, [30, 30, 30, 30, 30], discounted=True, discount_rate=10)
        assert payback is not None
        assert payback > 3  # Longer than simple payback

    def test_roi_calculation(self):
        roi = calculate_roi(150, 100)
        assert roi == 50  # (150-100)/100 * 100

    def test_mirr_calculation(self):
        cash_flows = [-100, 30, 30, 30, 30, 30]
        mirr = calculate_mirr(cash_flows, 8, 6)
        assert mirr is not None
        assert mirr > 0

    def test_profitability_index(self):
        pi = calculate_profitability_index(120, 100)
        assert pi == 1.2


class TestDecisionMaking:
    def test_accept_decision(self, agent, basic_input):
        result = agent.run(basic_input)
        # With positive NPV, should accept
        if result.roi_metrics.npv_usd > 0:
            assert result.investment_decision in ["ACCEPT", "MARGINAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
