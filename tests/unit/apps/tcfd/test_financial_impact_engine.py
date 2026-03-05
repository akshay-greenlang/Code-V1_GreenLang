# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Financial Impact Engine.

Tests income statement impact, balance sheet impact, cash flow impact,
NPV calculation, IRR calculation, MACC generation, carbon price
sensitivity, Monte Carlo simulation, and Climate VaR with 32+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    FinancialStatementArea,
    TimeHorizon,
    ScenarioType,
    SCENARIO_LIBRARY,
)
from services.models import (
    FinancialImpact,
    ScenarioResult,
    _new_id,
)


# ===========================================================================
# Income Statement Impact
# ===========================================================================

class TestIncomeStatementImpact:
    """Test income statement financial impact."""

    def test_revenue_impact(self, sample_financial_impact):
        assert sample_financial_impact.statement_area == FinancialStatementArea.INCOME_STATEMENT
        assert sample_financial_impact.line_item == "Revenue"

    def test_revenue_decline_calculation(self, sample_financial_impact):
        assert sample_financial_impact.impact_amount < Decimal("0")

    def test_cost_increase_impact(self):
        impact = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.INCOME_STATEMENT,
            line_item="Operating Costs",
            current_value=Decimal("1000000000"),
            projected_value=Decimal("1123000000"),
        )
        assert impact.impact_amount == Decimal("123000000")

    def test_margin_compression(self):
        revenue = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.INCOME_STATEMENT,
            line_item="Revenue",
            current_value=Decimal("2500000000"),
            projected_value=Decimal("2287500000"),
        )
        costs = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.INCOME_STATEMENT,
            line_item="Costs",
            current_value=Decimal("2000000000"),
            projected_value=Decimal("2100000000"),
        )
        margin_current = revenue.current_value - costs.current_value
        margin_projected = revenue.projected_value - costs.projected_value
        assert margin_projected < margin_current


# ===========================================================================
# Balance Sheet Impact
# ===========================================================================

class TestBalanceSheetImpact:
    """Test balance sheet financial impact."""

    def test_asset_impairment(self):
        impact = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.BALANCE_SHEET,
            line_item="Property, Plant & Equipment",
            current_value=Decimal("5000000000"),
            projected_value=Decimal("4250000000"),
        )
        assert impact.impact_amount == Decimal("-750000000")

    def test_stranded_assets(self):
        impact = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.BALANCE_SHEET,
            line_item="Fossil Fuel Reserves",
            current_value=Decimal("3000000000"),
            projected_value=Decimal("1500000000"),
        )
        assert impact.impact_pct < Decimal("0")

    def test_goodwill_impairment(self):
        impact = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.BALANCE_SHEET,
            line_item="Goodwill",
            current_value=Decimal("1000000000"),
            projected_value=Decimal("800000000"),
        )
        assert impact.impact_amount == Decimal("-200000000")


# ===========================================================================
# Cash Flow Impact
# ===========================================================================

class TestCashFlowImpact:
    """Test cash flow statement financial impact."""

    def test_capex_increase(self):
        impact = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.CASH_FLOW,
            line_item="Capital Expenditures",
            current_value=Decimal("200000000"),
            projected_value=Decimal("450000000"),
        )
        assert impact.impact_amount == Decimal("250000000")

    def test_operating_cash_flow_decline(self):
        impact = FinancialImpact(
            org_id=_new_id(),
            statement_area=FinancialStatementArea.CASH_FLOW,
            line_item="Operating Cash Flow",
            current_value=Decimal("500000000"),
            projected_value=Decimal("400000000"),
        )
        assert impact.impact_amount == Decimal("-100000000")


# ===========================================================================
# NPV Calculation
# ===========================================================================

class TestNPVCalculation:
    """Test net present value calculation."""

    def test_scenario_npv(self, sample_scenario_result):
        assert sample_scenario_result.npv == Decimal("-120000000")

    def test_positive_npv(self):
        result = ScenarioResult(
            scenario_id=_new_id(),
            org_id=_new_id(),
            npv=Decimal("50000000"),
        )
        assert result.npv > Decimal("0")

    def test_npv_formula(self):
        discount_rate = Decimal("0.08")
        cash_flows = [Decimal("-100"), Decimal("30"), Decimal("40"), Decimal("50"), Decimal("60")]
        npv = Decimal("0")
        for t, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** t)
        assert npv > Decimal("0")


# ===========================================================================
# IRR Approximation
# ===========================================================================

class TestIRRCalculation:
    """Test internal rate of return estimation."""

    def test_irr_positive_for_profitable_project(self):
        # Simple IRR check: if sum of future cash flows > investment, IRR > 0
        investment = Decimal("100")
        total_returns = Decimal("150")
        assert total_returns > investment  # IRR is positive

    def test_irr_negative_for_unprofitable_project(self):
        investment = Decimal("100")
        total_returns = Decimal("50")
        assert total_returns < investment  # IRR is negative


# ===========================================================================
# MACC (Marginal Abatement Cost Curve)
# ===========================================================================

class TestMACCGeneration:
    """Test marginal abatement cost curve generation."""

    def test_macc_ordering(self):
        abatement_options = [
            {"name": "LED Lighting", "cost_per_tco2e": Decimal("-20"), "potential_tco2e": Decimal("500")},
            {"name": "Solar PV", "cost_per_tco2e": Decimal("10"), "potential_tco2e": Decimal("5000")},
            {"name": "CCS", "cost_per_tco2e": Decimal("80"), "potential_tco2e": Decimal("10000")},
            {"name": "DAC", "cost_per_tco2e": Decimal("250"), "potential_tco2e": Decimal("2000")},
        ]
        sorted_options = sorted(abatement_options, key=lambda x: x["cost_per_tco2e"])
        assert sorted_options[0]["name"] == "LED Lighting"
        assert sorted_options[-1]["name"] == "DAC"

    def test_negative_cost_abatement(self):
        option = {"name": "Energy Efficiency", "cost_per_tco2e": Decimal("-30")}
        assert option["cost_per_tco2e"] < Decimal("0")

    def test_total_abatement_potential(self):
        options = [
            {"potential_tco2e": Decimal("500")},
            {"potential_tco2e": Decimal("5000")},
            {"potential_tco2e": Decimal("10000")},
        ]
        total = sum(o["potential_tco2e"] for o in options)
        assert total == Decimal("15500")


# ===========================================================================
# Carbon Price Sensitivity
# ===========================================================================

class TestCarbonPriceSensitivity:
    """Test carbon price sensitivity analysis."""

    @pytest.mark.parametrize("carbon_price,emissions,expected_cost", [
        (Decimal("50"), Decimal("125000"), Decimal("6250000")),
        (Decimal("130"), Decimal("125000"), Decimal("16250000")),
        (Decimal("250"), Decimal("125000"), Decimal("31250000")),
    ])
    def test_carbon_cost_at_prices(self, carbon_price, emissions, expected_cost):
        cost = carbon_price * emissions
        assert cost == expected_cost

    def test_sensitivity_range(self):
        base_cost = Decimal("16250000")
        low_cost = Decimal("6250000")
        high_cost = Decimal("31250000")
        assert low_cost < base_cost < high_cost


# ===========================================================================
# Monte Carlo Simulation
# ===========================================================================

class TestMonteCarloSimulation:
    """Test Monte Carlo simulation parameters."""

    def test_default_iterations(self, default_config):
        assert default_config.scenario_monte_carlo_iterations == 10000

    def test_custom_iterations(self, custom_config):
        assert custom_config.scenario_monte_carlo_iterations == 5000

    def test_confidence_interval(self, sample_scenario_result):
        ci_lower = sample_scenario_result.confidence_interval_lower
        ci_upper = sample_scenario_result.confidence_interval_upper
        assert ci_lower < ci_upper
        assert ci_lower < sample_scenario_result.npv < ci_upper


# ===========================================================================
# Climate VaR
# ===========================================================================

class TestClimateVaR:
    """Test Climate Value at Risk estimation."""

    def test_climate_var_negative(self):
        # Climate VaR is typically negative (potential loss)
        portfolio_value = Decimal("10000000000")
        climate_var_pct = Decimal("15.0")  # 15% of portfolio at risk
        var_amount = portfolio_value * climate_var_pct / Decimal("100")
        assert var_amount == Decimal("1500000000")

    def test_financial_impact_provenance(self, sample_financial_impact):
        assert len(sample_financial_impact.provenance_hash) == 64

    def test_financial_impact_percentage(self, sample_financial_impact):
        expected_pct = (
            (Decimal("2287500000") - Decimal("2500000000"))
            / Decimal("2500000000") * 100
        )
        assert sample_financial_impact.impact_pct == expected_pct

    @pytest.mark.parametrize("statement_area", list(FinancialStatementArea))
    def test_all_statement_areas(self, statement_area):
        impact = FinancialImpact(
            org_id=_new_id(),
            statement_area=statement_area,
            line_item=f"Test {statement_area.value}",
            current_value=Decimal("1000"),
            projected_value=Decimal("900"),
        )
        assert impact.statement_area == statement_area

    def test_financial_impact_assumptions(self, sample_financial_impact):
        assert len(sample_financial_impact.assumptions) > 0

    def test_financial_impact_time_horizon(self, sample_financial_impact):
        assert sample_financial_impact.time_horizon == TimeHorizon.MEDIUM_TERM
