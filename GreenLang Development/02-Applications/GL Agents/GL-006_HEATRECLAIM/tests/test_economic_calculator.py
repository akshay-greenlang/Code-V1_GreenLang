"""
GL-006 HEATRECLAIM - Economic Calculator Tests

Comprehensive test suite for economic analysis calculations including
capital cost estimation, NPV, IRR, payback period, and TAC.
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from ..core.schemas import HeatExchanger
from ..core.config import ExchangerType, EconomicParameters
from ..calculators.economic_calculator import (
    EconomicCalculator,
    CapitalCostBreakdown,
    OperatingCostBreakdown,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def economic_calculator():
    """Create economic calculator with default parameters."""
    return EconomicCalculator()


@pytest.fixture
def custom_params():
    """Create custom economic parameters."""
    return EconomicParameters(
        discount_rate=0.10,
        project_lifetime_years=20,
        operating_hours_per_year=8000,
        steam_cost_usd_gj=15.0,
        cooling_water_cost_usd_gj=2.0,
        electricity_cost_usd_kwh=0.12,
        installation_factor=1.5,
        piping_factor=0.15,
        instrumentation_factor=0.10,
        maintenance_cost_fraction=0.03,
    )


@pytest.fixture
def shell_and_tube_exchanger():
    """Create a shell and tube heat exchanger."""
    return HeatExchanger(
        exchanger_id="HX-001",
        exchanger_name="Process HX 1",
        exchanger_type=ExchangerType.SHELL_AND_TUBE,
        hot_stream_id="H1",
        cold_stream_id="C1",
        duty_kW=500.0,
        hot_inlet_T_C=150.0,
        hot_outlet_T_C=90.0,
        cold_inlet_T_C=30.0,
        cold_outlet_T_C=70.0,
        area_m2=50.0,
    )


@pytest.fixture
def plate_exchanger():
    """Create a plate heat exchanger."""
    return HeatExchanger(
        exchanger_id="HX-002",
        exchanger_name="Process HX 2",
        exchanger_type=ExchangerType.PLATE,
        hot_stream_id="H2",
        cold_stream_id="C2",
        duty_kW=300.0,
        hot_inlet_T_C=120.0,
        hot_outlet_T_C=80.0,
        cold_inlet_T_C=40.0,
        cold_outlet_T_C=65.0,
        area_m2=25.0,
    )


@pytest.fixture
def exchanger_list(shell_and_tube_exchanger, plate_exchanger):
    """Create list of exchangers."""
    return [shell_and_tube_exchanger, plate_exchanger]


# =============================================================================
# TEST: INITIALIZATION
# =============================================================================

class TestEconomicCalculatorInit:
    """Test economic calculator initialization."""

    def test_default_initialization(self):
        """Test default parameters."""
        calc = EconomicCalculator()
        assert calc.cost_year == 2024
        assert calc.params is not None

    def test_custom_parameters(self, custom_params):
        """Test custom parameter initialization."""
        calc = EconomicCalculator(params=custom_params)
        assert calc.params.discount_rate == 0.10
        assert calc.params.project_lifetime_years == 20

    def test_custom_cost_year(self):
        """Test custom cost year."""
        calc = EconomicCalculator(cost_year=2023)
        assert calc.cost_year == 2023


# =============================================================================
# TEST: EXCHANGER CAPITAL COST ESTIMATION
# =============================================================================

class TestExchangerCapitalCost:
    """Test heat exchanger capital cost estimation."""

    def test_shell_and_tube_cost(self, economic_calculator, shell_and_tube_exchanger):
        """Test S&T exchanger cost estimation."""
        cost = economic_calculator.estimate_exchanger_capital_cost(shell_and_tube_exchanger)

        assert cost > 0
        # With area=50 m2, cost should be in reasonable range
        assert 50000 < cost < 500000

    def test_plate_exchanger_cost(self, economic_calculator, plate_exchanger):
        """Test plate exchanger cost estimation."""
        cost = economic_calculator.estimate_exchanger_capital_cost(plate_exchanger)

        assert cost > 0
        # Plate exchangers have higher material factor
        assert 30000 < cost < 300000

    def test_cost_increases_with_area(self, economic_calculator):
        """Capital cost should increase with area."""
        hx_small = HeatExchanger(
            exchanger_id="HX-S",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=100.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=80.0,
            cold_inlet_T_C=30.0,
            cold_outlet_T_C=50.0,
            area_m2=10.0,
        )

        hx_large = HeatExchanger(
            exchanger_id="HX-L",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=1000.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=80.0,
            cold_inlet_T_C=30.0,
            cold_outlet_T_C=50.0,
            area_m2=100.0,
        )

        cost_small = economic_calculator.estimate_exchanger_capital_cost(hx_small)
        cost_large = economic_calculator.estimate_exchanger_capital_cost(hx_large)

        assert cost_large > cost_small

    def test_zero_area_returns_zero_cost(self, economic_calculator):
        """Zero area exchanger should have zero cost."""
        hx = HeatExchanger(
            exchanger_id="HX-ZERO",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=0.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=100.0,
            cold_inlet_T_C=30.0,
            cold_outlet_T_C=30.0,
            area_m2=0.0,
        )

        cost = economic_calculator.estimate_exchanger_capital_cost(hx)
        assert cost == 0.0

    def test_override_area(self, economic_calculator, shell_and_tube_exchanger):
        """Test cost with overridden area."""
        cost_default = economic_calculator.estimate_exchanger_capital_cost(
            shell_and_tube_exchanger
        )
        cost_override = economic_calculator.estimate_exchanger_capital_cost(
            shell_and_tube_exchanger, area_m2=100.0
        )

        assert cost_override > cost_default


# =============================================================================
# TEST: CAPITAL COST BREAKDOWN
# =============================================================================

class TestCapitalCostBreakdown:
    """Test total capital cost calculation."""

    def test_capital_costs_itemized(self, economic_calculator, exchanger_list):
        """Test itemized capital cost breakdown."""
        breakdown = economic_calculator.calculate_capital_costs(exchanger_list)

        assert isinstance(breakdown, CapitalCostBreakdown)
        assert breakdown.equipment_cost_usd > 0
        assert breakdown.installation_cost_usd > 0
        assert breakdown.piping_cost_usd > 0
        assert breakdown.instrumentation_cost_usd > 0
        assert breakdown.engineering_cost_usd > 0
        assert breakdown.contingency_usd > 0

    def test_total_equals_sum(self, economic_calculator, exchanger_list):
        """Total should equal sum of components."""
        breakdown = economic_calculator.calculate_capital_costs(exchanger_list)

        calculated_total = (
            breakdown.equipment_cost_usd +
            breakdown.installation_cost_usd +
            breakdown.piping_cost_usd +
            breakdown.instrumentation_cost_usd +
            breakdown.engineering_cost_usd +
            breakdown.contingency_usd
        )

        assert breakdown.total_capital_usd == pytest.approx(calculated_total, rel=0.01)

    def test_no_contingency(self, economic_calculator, exchanger_list):
        """Test calculation without contingency."""
        breakdown = economic_calculator.calculate_capital_costs(
            exchanger_list, include_contingency=False
        )

        assert breakdown.contingency_usd == 0.0

    def test_custom_contingency(self, economic_calculator, exchanger_list):
        """Test custom contingency percentage."""
        breakdown_15 = economic_calculator.calculate_capital_costs(
            exchanger_list, contingency_percent=15.0
        )
        breakdown_25 = economic_calculator.calculate_capital_costs(
            exchanger_list, contingency_percent=25.0
        )

        assert breakdown_25.contingency_usd > breakdown_15.contingency_usd

    def test_empty_exchanger_list(self, economic_calculator):
        """Empty list should return zeros."""
        breakdown = economic_calculator.calculate_capital_costs([])

        assert breakdown.equipment_cost_usd == 0
        assert breakdown.total_capital_usd == 0


# =============================================================================
# TEST: UTILITY SAVINGS
# =============================================================================

class TestUtilitySavings:
    """Test utility cost savings calculation."""

    def test_hot_utility_savings(self, economic_calculator):
        """Test savings from hot utility reduction."""
        savings = economic_calculator.calculate_utility_savings(
            hot_utility_reduction_kW=1000.0,
            cold_utility_reduction_kW=0.0,
        )

        assert savings > 0
        # 1000 kW * 8000 hr * 0.0036 GJ/kWh * $10/GJ ≈ $288,000
        assert 200000 < savings < 400000

    def test_cold_utility_savings(self, economic_calculator):
        """Test savings from cold utility reduction."""
        savings = economic_calculator.calculate_utility_savings(
            hot_utility_reduction_kW=0.0,
            cold_utility_reduction_kW=500.0,
        )

        assert savings > 0
        # Cold utility is cheaper than hot

    def test_combined_savings(self, economic_calculator):
        """Test combined hot and cold utility savings."""
        hot_only = economic_calculator.calculate_utility_savings(
            hot_utility_reduction_kW=500.0,
            cold_utility_reduction_kW=0.0,
        )
        cold_only = economic_calculator.calculate_utility_savings(
            hot_utility_reduction_kW=0.0,
            cold_utility_reduction_kW=300.0,
        )
        combined = economic_calculator.calculate_utility_savings(
            hot_utility_reduction_kW=500.0,
            cold_utility_reduction_kW=300.0,
        )

        assert combined == pytest.approx(hot_only + cold_only, rel=0.01)

    def test_custom_utility_costs(self, economic_calculator):
        """Test with custom utility costs."""
        savings_default = economic_calculator.calculate_utility_savings(
            hot_utility_reduction_kW=1000.0,
            cold_utility_reduction_kW=0.0,
        )
        savings_expensive = economic_calculator.calculate_utility_savings(
            hot_utility_reduction_kW=1000.0,
            cold_utility_reduction_kW=0.0,
            hot_utility_cost_usd_gj=20.0,  # More expensive
        )

        assert savings_expensive > savings_default


# =============================================================================
# TEST: OPERATING COSTS
# =============================================================================

class TestOperatingCosts:
    """Test operating cost calculations."""

    def test_operating_costs_itemized(self, economic_calculator):
        """Test itemized operating cost breakdown."""
        breakdown = economic_calculator.calculate_operating_costs(
            capital_cost_usd=1000000.0,
            utility_cost_usd_yr=50000.0,
        )

        assert isinstance(breakdown, OperatingCostBreakdown)
        assert breakdown.utility_cost_usd_yr == 50000.0
        assert breakdown.maintenance_cost_usd_yr > 0
        assert breakdown.labor_cost_usd_yr > 0
        assert breakdown.insurance_cost_usd_yr > 0

    def test_total_operating_cost(self, economic_calculator):
        """Total should equal sum of components."""
        breakdown = economic_calculator.calculate_operating_costs(
            capital_cost_usd=1000000.0,
            utility_cost_usd_yr=50000.0,
        )

        calculated = (
            breakdown.utility_cost_usd_yr +
            breakdown.maintenance_cost_usd_yr +
            breakdown.labor_cost_usd_yr +
            breakdown.insurance_cost_usd_yr
        )

        assert breakdown.total_operating_usd_yr == pytest.approx(calculated, rel=0.01)

    def test_maintenance_scales_with_capital(self, economic_calculator):
        """Maintenance should scale with capital cost."""
        breakdown_low = economic_calculator.calculate_operating_costs(
            capital_cost_usd=500000.0
        )
        breakdown_high = economic_calculator.calculate_operating_costs(
            capital_cost_usd=1000000.0
        )

        assert breakdown_high.maintenance_cost_usd_yr > breakdown_low.maintenance_cost_usd_yr


# =============================================================================
# TEST: NPV CALCULATION
# =============================================================================

class TestNPVCalculation:
    """Test Net Present Value calculations."""

    def test_positive_npv(self, economic_calculator):
        """Good project should have positive NPV."""
        npv = economic_calculator.calculate_npv(
            capital_cost_usd=500000.0,
            annual_savings_usd=100000.0,
            annual_operating_cost_usd=10000.0,
        )

        # With $90k net annual over 15 years at 8%, NPV should be positive
        assert npv > 0

    def test_negative_npv(self, economic_calculator):
        """Poor project should have negative NPV."""
        npv = economic_calculator.calculate_npv(
            capital_cost_usd=1000000.0,
            annual_savings_usd=20000.0,
            annual_operating_cost_usd=30000.0,  # Negative net cash flow
        )

        assert npv < 0

    def test_npv_increases_with_savings(self, economic_calculator):
        """NPV should increase with higher savings."""
        npv_low = economic_calculator.calculate_npv(
            capital_cost_usd=500000.0,
            annual_savings_usd=50000.0,
        )
        npv_high = economic_calculator.calculate_npv(
            capital_cost_usd=500000.0,
            annual_savings_usd=100000.0,
        )

        assert npv_high > npv_low

    def test_npv_decreases_with_discount_rate(self, economic_calculator):
        """NPV should decrease with higher discount rate."""
        npv_low_r = economic_calculator.calculate_npv(
            capital_cost_usd=500000.0,
            annual_savings_usd=80000.0,
            discount_rate=0.05,
        )
        npv_high_r = economic_calculator.calculate_npv(
            capital_cost_usd=500000.0,
            annual_savings_usd=80000.0,
            discount_rate=0.15,
        )

        assert npv_high_r < npv_low_r

    def test_npv_with_zero_discount(self, economic_calculator):
        """Zero discount rate should give simple sum."""
        npv = economic_calculator.calculate_npv(
            capital_cost_usd=100000.0,
            annual_savings_usd=20000.0,
            discount_rate=0.0,
            project_lifetime_years=10,
        )

        # Simple sum: -100000 + 20000 * 10 = 100000
        assert npv == pytest.approx(100000.0, rel=0.01)


# =============================================================================
# TEST: IRR CALCULATION
# =============================================================================

class TestIRRCalculation:
    """Test Internal Rate of Return calculations."""

    def test_irr_exists(self, economic_calculator):
        """Valid project should have calculable IRR."""
        irr = economic_calculator.calculate_irr(
            capital_cost_usd=500000.0,
            annual_savings_usd=100000.0,
        )

        assert irr is not None
        assert irr > 0

    def test_irr_negative_net_returns_none(self, economic_calculator):
        """Negative net cash flow should return None."""
        irr = economic_calculator.calculate_irr(
            capital_cost_usd=500000.0,
            annual_savings_usd=10000.0,
            annual_operating_cost_usd=20000.0,
        )

        assert irr is None

    def test_irr_reasonable_range(self, economic_calculator):
        """IRR should be in reasonable range for typical projects."""
        irr = economic_calculator.calculate_irr(
            capital_cost_usd=500000.0,
            annual_savings_usd=80000.0,
            project_lifetime_years=15,
        )

        assert irr is not None
        assert 0.05 < irr < 0.30  # 5% to 30%

    def test_irr_increases_with_savings(self, economic_calculator):
        """IRR should increase with higher savings."""
        irr_low = economic_calculator.calculate_irr(
            capital_cost_usd=500000.0,
            annual_savings_usd=60000.0,
        )
        irr_high = economic_calculator.calculate_irr(
            capital_cost_usd=500000.0,
            annual_savings_usd=100000.0,
        )

        assert irr_high > irr_low


# =============================================================================
# TEST: PAYBACK CALCULATION
# =============================================================================

class TestPaybackCalculation:
    """Test simple payback period calculations."""

    def test_simple_payback(self, economic_calculator):
        """Test basic payback calculation."""
        payback = economic_calculator.calculate_payback(
            capital_cost_usd=500000.0,
            annual_savings_usd=100000.0,
        )

        # $500k / $100k/yr = 5 years
        assert payback == pytest.approx(5.0, rel=0.01)

    def test_payback_with_operating_costs(self, economic_calculator):
        """Payback should account for operating costs."""
        payback = economic_calculator.calculate_payback(
            capital_cost_usd=500000.0,
            annual_savings_usd=100000.0,
            annual_operating_cost_usd=20000.0,
        )

        # $500k / ($100k - $20k)/yr = 6.25 years
        assert payback == pytest.approx(6.25, rel=0.01)

    def test_payback_zero_net_returns_infinity(self, economic_calculator):
        """Zero or negative net should return infinity."""
        payback = economic_calculator.calculate_payback(
            capital_cost_usd=500000.0,
            annual_savings_usd=50000.0,
            annual_operating_cost_usd=60000.0,
        )

        assert payback == float('inf')

    def test_payback_decreases_with_higher_savings(self, economic_calculator):
        """Payback should decrease with higher savings."""
        payback_low = economic_calculator.calculate_payback(
            capital_cost_usd=500000.0,
            annual_savings_usd=50000.0,
        )
        payback_high = economic_calculator.calculate_payback(
            capital_cost_usd=500000.0,
            annual_savings_usd=100000.0,
        )

        assert payback_high < payback_low


# =============================================================================
# TEST: TOTAL ANNUAL COST (TAC)
# =============================================================================

class TestTotalAnnualCost:
    """Test Total Annual Cost calculations."""

    def test_tac_basic(self, economic_calculator):
        """Test basic TAC calculation."""
        tac = economic_calculator.calculate_total_annual_cost(
            capital_cost_usd=1000000.0,
            annual_operating_cost_usd=50000.0,
        )

        assert tac > 50000  # Should be operating + annualized capital

    def test_tac_includes_annualized_capital(self, economic_calculator):
        """TAC should include annualized capital."""
        tac_low_cap = economic_calculator.calculate_total_annual_cost(
            capital_cost_usd=500000.0,
            annual_operating_cost_usd=50000.0,
        )
        tac_high_cap = economic_calculator.calculate_total_annual_cost(
            capital_cost_usd=1000000.0,
            annual_operating_cost_usd=50000.0,
        )

        assert tac_high_cap > tac_low_cap

    def test_tac_increases_with_discount_rate(self, economic_calculator):
        """Higher discount rate should increase annualized capital cost."""
        tac_low = economic_calculator.calculate_total_annual_cost(
            capital_cost_usd=1000000.0,
            annual_operating_cost_usd=50000.0,
            discount_rate=0.05,
        )
        tac_high = economic_calculator.calculate_total_annual_cost(
            capital_cost_usd=1000000.0,
            annual_operating_cost_usd=50000.0,
            discount_rate=0.15,
        )

        assert tac_high > tac_low


# =============================================================================
# TEST: FULL ECONOMIC ANALYSIS
# =============================================================================

class TestFullEconomicAnalysis:
    """Test complete economic analysis."""

    def test_full_analysis_structure(self, economic_calculator, exchanger_list):
        """Test full analysis returns complete structure."""
        result = economic_calculator.calculate_full_analysis(
            exchangers=exchanger_list,
            hot_utility_reduction_kW=500.0,
            cold_utility_reduction_kW=200.0,
        )

        assert result is not None
        assert result.total_capital_cost_usd > 0
        assert result.annual_utility_savings_usd > 0
        assert result.payback_period_years > 0
        assert result.npv_usd is not None

    def test_full_analysis_co2_reduction(self, economic_calculator, exchanger_list):
        """Analysis should include CO2 reduction."""
        result = economic_calculator.calculate_full_analysis(
            exchangers=exchanger_list,
            hot_utility_reduction_kW=1000.0,
            cold_utility_reduction_kW=0.0,
        )

        assert result.co2_reduction_tonnes_yr > 0

    def test_full_analysis_provenance(self, economic_calculator, exchanger_list):
        """Analysis should include provenance hashes."""
        result = economic_calculator.calculate_full_analysis(
            exchangers=exchanger_list,
            hot_utility_reduction_kW=500.0,
            cold_utility_reduction_kW=200.0,
        )

        assert result.input_hash is not None
        assert result.output_hash is not None

    def test_full_analysis_reproducibility(self, economic_calculator, exchanger_list):
        """Same inputs should produce identical results."""
        result1 = economic_calculator.calculate_full_analysis(
            exchangers=exchanger_list,
            hot_utility_reduction_kW=500.0,
            cold_utility_reduction_kW=200.0,
        )
        result2 = economic_calculator.calculate_full_analysis(
            exchangers=exchanger_list,
            hot_utility_reduction_kW=500.0,
            cold_utility_reduction_kW=200.0,
        )

        assert result1.total_capital_cost_usd == result2.total_capital_cost_usd
        assert result1.npv_usd == result2.npv_usd
        assert result1.input_hash == result2.input_hash


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_project(self, economic_calculator):
        """Test with very small capital cost."""
        hx = HeatExchanger(
            exchanger_id="HX-TINY",
            exchanger_type=ExchangerType.DOUBLE_PIPE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=10.0,
            hot_inlet_T_C=60.0,
            hot_outlet_T_C=50.0,
            cold_inlet_T_C=30.0,
            cold_outlet_T_C=35.0,
            area_m2=1.0,
        )

        cost = economic_calculator.estimate_exchanger_capital_cost(hx)
        assert cost > 0

    def test_very_large_project(self, economic_calculator):
        """Test with very large capital cost."""
        hx = HeatExchanger(
            exchanger_id="HX-HUGE",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=50000.0,
            hot_inlet_T_C=300.0,
            hot_outlet_T_C=100.0,
            cold_inlet_T_C=30.0,
            cold_outlet_T_C=200.0,
            area_m2=1000.0,
        )

        cost = economic_calculator.estimate_exchanger_capital_cost(hx)
        assert cost > 0

    def test_different_exchanger_types(self, economic_calculator):
        """Test cost estimation for all exchanger types."""
        exchanger_types = [
            ExchangerType.SHELL_AND_TUBE,
            ExchangerType.PLATE,
            ExchangerType.PLATE_FIN,
            ExchangerType.SPIRAL,
            ExchangerType.AIR_COOLED,
            ExchangerType.DOUBLE_PIPE,
            ExchangerType.ECONOMIZER,
            ExchangerType.RECUPERATOR,
        ]

        for ex_type in exchanger_types:
            hx = HeatExchanger(
                exchanger_id=f"HX-{ex_type.value}",
                exchanger_type=ex_type,
                hot_stream_id="H1",
                cold_stream_id="C1",
                duty_kW=200.0,
                hot_inlet_T_C=100.0,
                hot_outlet_T_C=80.0,
                cold_inlet_T_C=30.0,
                cold_outlet_T_C=50.0,
                area_m2=20.0,
            )

            cost = economic_calculator.estimate_exchanger_capital_cost(hx)
            assert cost > 0, f"Cost should be positive for {ex_type}"
