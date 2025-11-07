"""
Tests for Financial Metrics Tool
=================================

Comprehensive test suite for FinancialMetricsTool including:
- NPV calculations
- IRR calculations
- Payback period calculations
- Incentive handling
- Edge cases and error handling

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.agents.tools.financial import FinancialMetricsTool


class TestFinancialMetricsTool:
    """Test suite for FinancialMetricsTool."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return FinancialMetricsTool()

    def test_basic_npv_calculation(self, tool):
        """Test basic NPV calculation."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert "npv" in result.data
        assert result.data["npv"] > 0  # Should be positive NPV
        assert "irr" in result.data
        assert result.data["simple_payback_years"] is not None

    def test_npv_with_escalation(self, tool):
        """Test NPV calculation with energy cost escalation."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            energy_cost_escalation=0.03,  # 3% annual escalation
        )

        assert result.success
        assert result.data["npv"] > 0

        # NPV with escalation should be higher than without
        result_no_escalation = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            energy_cost_escalation=0.0,
        )

        assert result.data["npv"] > result_no_escalation.data["npv"]

    def test_npv_with_om_costs(self, tool):
        """Test NPV calculation with O&M costs."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            annual_om_cost=1000,
        )

        assert result.success

        # NPV with O&M should be lower than without
        result_no_om = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            annual_om_cost=0,
        )

        assert result.data["npv"] < result_no_om.data["npv"]

    def test_irr_calculation(self, tool):
        """Test IRR calculation."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=10000,
            lifetime_years=10,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["irr"] is not None
        assert 0 < result.data["irr"] < 1  # IRR should be reasonable

    def test_simple_payback(self, tool):
        """Test simple payback period calculation."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=10000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["simple_payback_years"] is not None
        # Should pay back in approximately 5 years (50000 / 10000)
        assert 4.5 <= result.data["simple_payback_years"] <= 5.5

    def test_discounted_payback(self, tool):
        """Test discounted payback period calculation."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=10000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["discounted_payback_years"] is not None

        # Discounted payback should be longer than simple payback
        assert result.data["discounted_payback_years"] > result.data["simple_payback_years"]

    def test_incentives_year_0(self, tool):
        """Test incentives applied in year 0."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            incentives=[
                {"name": "ITC", "amount": 15000, "year": 0},
            ],
        )

        assert result.success
        assert result.data["total_incentives"] == 15000
        assert result.data["year_0_incentives"] == 15000
        assert result.data["net_capital_cost"] == 35000  # 50000 - 15000

        # NPV should be higher with incentive
        result_no_incentive = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.data["npv"] > result_no_incentive.data["npv"]

    def test_incentives_multiple_years(self, tool):
        """Test incentives applied over multiple years."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            incentives=[
                {"name": "ITC", "amount": 15000, "year": 0},
                {"name": "PTC Year 1", "amount": 5000, "year": 1},
                {"name": "PTC Year 2", "amount": 5000, "year": 2},
            ],
        )

        assert result.success
        assert result.data["total_incentives"] == 25000
        assert result.data["year_0_incentives"] == 15000
        assert len(result.data["incentive_details"]) == 3

    def test_depreciation_benefits(self, tool):
        """Test MACRS depreciation tax benefits."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            include_depreciation=True,
            tax_rate=0.21,
        )

        assert result.success

        # NPV with depreciation should be higher
        result_no_depreciation = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            include_depreciation=False,
        )

        assert result.data["npv"] > result_no_depreciation.data["npv"]

    def test_salvage_value(self, tool):
        """Test salvage value at end of life."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            salvage_value=5000,
        )

        assert result.success

        # NPV with salvage value should be higher
        result_no_salvage = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            salvage_value=0,
        )

        assert result.data["npv"] > result_no_salvage.data["npv"]

    def test_lifecycle_cost(self, tool):
        """Test lifecycle cost calculation."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            annual_om_cost=1000,
        )

        assert result.success
        assert "lifecycle_cost" in result.data
        assert result.data["lifecycle_cost"] > result.data["capital_cost"]

    def test_benefit_cost_ratio(self, tool):
        """Test benefit-cost ratio calculation."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=10000,
            lifetime_years=10,
            discount_rate=0.05,
        )

        assert result.success
        assert "benefit_cost_ratio" in result.data
        assert result.data["benefit_cost_ratio"] > 1  # Should be positive

    def test_negative_npv(self, tool):
        """Test project with negative NPV."""
        result = tool.execute(
            capital_cost=100000,
            annual_savings=2000,  # Very low savings
            lifetime_years=10,
            discount_rate=0.10,  # High discount rate
        )

        assert result.success
        assert result.data["npv"] < 0  # Should be negative NPV

    def test_zero_savings(self, tool):
        """Test edge case with zero savings."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=0,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["npv"] < 0  # Should be negative
        assert result.data["simple_payback_years"] is None  # Never pays back

    def test_high_discount_rate(self, tool):
        """Test with high discount rate."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.15,  # 15% discount rate
        )

        assert result.success

        # NPV with high discount rate should be lower
        result_low_rate = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.data["npv"] < result_low_rate.data["npv"]

    def test_short_lifetime(self, tool):
        """Test project with short lifetime."""
        result = tool.execute(
            capital_cost=10000,
            annual_savings=3000,
            lifetime_years=5,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["simple_payback_years"] < 5

    def test_long_lifetime(self, tool):
        """Test project with long lifetime."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=5000,
            lifetime_years=50,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["npv"] > 0

    def test_cash_flow_array(self, tool):
        """Test that cash flow arrays are returned."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=10,
            discount_rate=0.05,
        )

        assert result.success
        assert "annual_cash_flows" in result.data
        assert len(result.data["annual_cash_flows"]) == 11  # 0 to 10 years
        assert "cumulative_cash_flows" in result.data
        assert len(result.data["cumulative_cash_flows"]) == 11

    def test_invalid_capital_cost(self, tool):
        """Test input validation for negative capital cost."""
        result = tool.execute(
            capital_cost=-50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert not result.success
        assert "capital_cost must be non-negative" in result.error

    def test_invalid_lifetime(self, tool):
        """Test input validation for invalid lifetime."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=0,
            discount_rate=0.05,
        )

        assert not result.success
        assert "lifetime_years must be positive" in result.error

    def test_invalid_discount_rate(self, tool):
        """Test input validation for invalid discount rate."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=1.5,  # > 1.0
        )

        assert not result.success
        assert "discount_rate must be between 0 and 1" in result.error

    def test_citations_included(self, tool):
        """Test that calculation citations are included."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert len(result.citations) > 0
        assert any(c.step_name == "calculate_npv" for c in result.citations)
        assert any(c.step_name == "calculate_irr" for c in result.citations)

    def test_metadata_included(self, tool):
        """Test that metadata is included in result."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert "calculation_inputs" in result.metadata
        assert "summary" in result.metadata

    def test_tool_definition(self, tool):
        """Test that tool definition is valid."""
        tool_def = tool.get_tool_def()

        assert tool_def.name == "calculate_financial_metrics"
        assert tool_def.safety.value == "deterministic"
        assert "capital_cost" in tool_def.parameters["properties"]
        assert "annual_savings" in tool_def.parameters["properties"]
        assert "lifetime_years" in tool_def.parameters["properties"]

    def test_realistic_solar_project(self, tool):
        """Test realistic solar installation project."""
        result = tool.execute(
            capital_cost=100000,
            annual_savings=12000,
            lifetime_years=25,
            discount_rate=0.06,
            annual_om_cost=500,
            energy_cost_escalation=0.025,
            incentives=[
                {"name": "IRA 2022 ITC", "amount": 30000, "year": 0},
            ],
            include_depreciation=True,
            tax_rate=0.21,
            salvage_value=5000,
        )

        assert result.success
        assert result.data["npv"] > 0
        assert result.data["irr"] > 0
        assert result.data["simple_payback_years"] < 10
        assert result.data["benefit_cost_ratio"] > 1

    def test_realistic_hvac_upgrade(self, tool):
        """Test realistic HVAC upgrade project."""
        result = tool.execute(
            capital_cost=25000,
            annual_savings=4000,
            lifetime_years=15,
            discount_rate=0.05,
            annual_om_cost=200,
            energy_cost_escalation=0.02,
        )

        assert result.success
        assert result.data["simple_payback_years"] < 8
        assert result.data["npv"] > 0

    def test_realistic_led_retrofit(self, tool):
        """Test realistic LED lighting retrofit."""
        result = tool.execute(
            capital_cost=15000,
            annual_savings=3500,
            lifetime_years=10,
            discount_rate=0.05,
            annual_om_cost=100,
        )

        assert result.success
        assert result.data["simple_payback_years"] < 5
        assert result.data["irr"] > 0.15  # Should have good IRR

    def test_tool_execution_metrics(self, tool):
        """Test that execution metrics are tracked."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert result.execution_time_ms > 0

        # Check tool stats
        stats = tool.get_stats()
        assert stats["executions"] >= 1
        assert stats["total_time_ms"] > 0


class TestFinancialMetricsEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return FinancialMetricsTool()

    def test_minimum_lifetime(self, tool):
        """Test minimum lifetime (1 year)."""
        result = tool.execute(
            capital_cost=1000,
            annual_savings=1500,
            lifetime_years=1,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["simple_payback_years"] < 1

    def test_maximum_incentive(self, tool):
        """Test incentive larger than capital cost."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            incentives=[
                {"name": "Large Grant", "amount": 60000, "year": 0},
            ],
        )

        assert result.success
        assert result.data["net_capital_cost"] < 0  # Negative net cost (profitable from start)

    def test_zero_discount_rate(self, tool):
        """Test zero discount rate."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=10,
            discount_rate=0.0,
        )

        assert result.success
        # With zero discount rate, NPV = sum of all cash flows
        expected_npv = (8000 * 10) - 50000
        assert abs(result.data["npv"] - expected_npv) < 100

    def test_very_small_savings(self, tool):
        """Test very small annual savings."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=10,
            lifetime_years=25,
            discount_rate=0.05,
        )

        assert result.success
        assert result.data["npv"] < 0
        assert result.data["simple_payback_years"] is None

    def test_multiple_incentives_same_year(self, tool):
        """Test multiple incentives in the same year."""
        result = tool.execute(
            capital_cost=50000,
            annual_savings=8000,
            lifetime_years=25,
            discount_rate=0.05,
            incentives=[
                {"name": "ITC", "amount": 10000, "year": 0},
                {"name": "State Rebate", "amount": 5000, "year": 0},
                {"name": "Utility Incentive", "amount": 3000, "year": 0},
            ],
        )

        assert result.success
        assert result.data["year_0_incentives"] == 18000
        assert result.data["total_incentives"] == 18000
