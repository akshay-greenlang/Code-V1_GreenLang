# -*- coding: utf-8 -*-
"""
ROI Calculator tests for GL-006 HeatRecoveryMaximizer.

This module validates financial analysis calculations including:
- Capital cost estimation accuracy
- Operating cost calculations
- Energy savings calculations
- Simple payback period
- Net Present Value (NPV)
- Internal Rate of Return (IRR)
- Profitability index
- Discounted payback
- Sensitivity analysis
- CO2 emission reduction

Target: 25+ ROI tests
"""

import pytest
import numpy as np
import math
from typing import Dict, List, Any
from decimal import Decimal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from calculators.roi_calculator import (
    ROICalculator,
    CapitalCostInput,
    OperatingCostInput,
    EnergySavingsInput,
    FinancialParameters,
    ROIResult,
    SensitivityResult,
    EquipmentType,
    DepreciationMethod,
    EquipmentCostFactors
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calculator():
    """Create ROI calculator instance."""
    return ROICalculator()


@pytest.fixture
def standard_capital_input():
    """Create standard capital cost input."""
    return CapitalCostInput(
        equipment_type=EquipmentType.SHELL_TUBE_HX,
        heat_capacity_kw=500.0,
        material="carbon_steel",
        pressure_rating_bar=10.0,
        include_installation=True,
        include_auxiliary=True,
        location_factor=1.0
    )


@pytest.fixture
def standard_operating_input():
    """Create standard operating cost input."""
    return OperatingCostInput(
        maintenance_percent_of_capital=2.0,
        operating_hours_per_year=8000,
        pumping_power_kw=5.0,
        electricity_cost_usd_per_kwh=0.10,
        insurance_percent=0.5
    )


@pytest.fixture
def standard_savings_input():
    """Create standard energy savings input."""
    return EnergySavingsInput(
        heat_recovery_kw=500.0,
        operating_hours_per_year=8000,
        energy_cost_usd_per_kwh=0.08,
        energy_cost_escalation_percent=3.0,
        avoided_fuel="natural_gas",
        system_efficiency=0.85
    )


@pytest.fixture
def standard_financial_params():
    """Create standard financial parameters."""
    return FinancialParameters(
        discount_rate_percent=10.0,
        analysis_period_years=15,
        inflation_rate_percent=2.5,
        tax_rate_percent=25.0,
        residual_value_percent=10.0
    )


# ============================================================================
# CAPITAL COST ESTIMATION TESTS
# ============================================================================

@pytest.mark.roi
class TestCapitalCostEstimation:
    """Test capital cost estimation accuracy."""

    def test_bare_equipment_cost_calculation(self, calculator, standard_capital_input):
        """Test bare equipment cost calculation."""
        costs = calculator._calculate_capital_costs(standard_capital_input)

        assert costs['equipment_cost'] > 0
        # 500 kW * ~$600/kW = ~$300,000 ballpark
        assert 100000 < costs['equipment_cost'] < 1000000

    def test_installation_cost_calculation(self, calculator, standard_capital_input):
        """Test installation cost is calculated correctly."""
        costs = calculator._calculate_capital_costs(standard_capital_input)

        assert costs['installation_cost'] > 0
        # Installation typically 25% of equipment
        assert costs['installation_cost'] <= costs['equipment_cost']

    def test_total_capital_cost(self, calculator, standard_capital_input):
        """Test total capital cost is sum of components."""
        costs = calculator._calculate_capital_costs(standard_capital_input)

        expected_total = costs['equipment_cost'] + costs['installation_cost']
        assert costs['total_capital_cost'] == pytest.approx(expected_total, rel=0.01)

    def test_material_cost_multiplier(self, calculator):
        """Test material cost multiplier is applied correctly."""
        carbon_steel_input = CapitalCostInput(
            equipment_type=EquipmentType.SHELL_TUBE_HX,
            heat_capacity_kw=100.0,
            material="carbon_steel"
        )
        stainless_input = CapitalCostInput(
            equipment_type=EquipmentType.SHELL_TUBE_HX,
            heat_capacity_kw=100.0,
            material="stainless_steel_304"
        )

        costs_cs = calculator._calculate_capital_costs(carbon_steel_input)
        costs_ss = calculator._calculate_capital_costs(stainless_input)

        # Stainless should be ~2x carbon steel
        assert costs_ss['equipment_cost'] > costs_cs['equipment_cost']
        ratio = costs_ss['equipment_cost'] / costs_cs['equipment_cost']
        assert 1.5 < ratio < 3.0

    def test_location_factor_applied(self, calculator):
        """Test location cost factor is applied."""
        base_input = CapitalCostInput(
            equipment_type=EquipmentType.SHELL_TUBE_HX,
            heat_capacity_kw=100.0,
            material="carbon_steel",
            location_factor=1.0
        )
        high_cost_input = CapitalCostInput(
            equipment_type=EquipmentType.SHELL_TUBE_HX,
            heat_capacity_kw=100.0,
            material="carbon_steel",
            location_factor=1.5
        )

        costs_base = calculator._calculate_capital_costs(base_input)
        costs_high = calculator._calculate_capital_costs(high_cost_input)

        assert costs_high['equipment_cost'] == pytest.approx(
            costs_base['equipment_cost'] * 1.5, rel=0.01
        )

    def test_different_equipment_types(self, calculator):
        """Test cost estimation for different equipment types."""
        equipment_types = [
            EquipmentType.SHELL_TUBE_HX,
            EquipmentType.PLATE_HX,
            EquipmentType.ECONOMIZER,
            EquipmentType.AIR_PREHEATER
        ]

        costs = {}
        for eq_type in equipment_types:
            input_data = CapitalCostInput(
                equipment_type=eq_type,
                heat_capacity_kw=100.0,
                material="carbon_steel"
            )
            result = calculator._calculate_capital_costs(input_data)
            costs[eq_type] = result['equipment_cost']

        # All should have positive costs
        assert all(c > 0 for c in costs.values())

        # Plate HX typically cheaper than shell & tube for same capacity
        assert costs[EquipmentType.PLATE_HX] <= costs[EquipmentType.SHELL_TUBE_HX]


# ============================================================================
# OPERATING COST TESTS
# ============================================================================

@pytest.mark.roi
class TestOperatingCostCalculation:
    """Test operating cost calculations."""

    def test_maintenance_cost_calculation(self, calculator, standard_operating_input):
        """Test maintenance cost is calculated correctly."""
        capital_cost = 300000.0

        costs = calculator._calculate_annual_operating_costs(
            standard_operating_input, capital_cost
        )

        # 2% of capital = $6,000
        expected_maintenance = capital_cost * 0.02
        assert costs['maintenance_cost'] == pytest.approx(expected_maintenance, rel=0.01)

    def test_pumping_energy_cost(self, calculator, standard_operating_input):
        """Test pumping/auxiliary energy cost calculation."""
        costs = calculator._calculate_annual_operating_costs(
            standard_operating_input, 300000.0
        )

        # 5 kW * 8000 hours * $0.10/kWh = $4,000
        expected_energy = 5.0 * 8000 * 0.10
        assert costs['energy_cost'] == pytest.approx(expected_energy, rel=0.01)

    def test_insurance_cost(self, calculator, standard_operating_input):
        """Test insurance cost calculation."""
        capital_cost = 300000.0

        costs = calculator._calculate_annual_operating_costs(
            standard_operating_input, capital_cost
        )

        # 0.5% of capital = $1,500
        expected_insurance = capital_cost * 0.005
        assert costs['insurance_cost'] == pytest.approx(expected_insurance, rel=0.01)

    def test_total_operating_cost(self, calculator, standard_operating_input):
        """Test total operating cost is sum of components."""
        capital_cost = 300000.0

        costs = calculator._calculate_annual_operating_costs(
            standard_operating_input, capital_cost
        )

        expected_total = (
            costs['maintenance_cost'] +
            costs['energy_cost'] +
            costs['water_treatment_cost'] +
            costs['insurance_cost']
        )

        assert costs['total_annual_cost'] == pytest.approx(expected_total, rel=0.01)


# ============================================================================
# ENERGY SAVINGS TESTS
# ============================================================================

@pytest.mark.roi
class TestEnergySavingsCalculation:
    """Test energy savings calculations."""

    def test_annual_kwh_savings(self, calculator, standard_savings_input):
        """Test annual energy savings in kWh."""
        savings = calculator._calculate_annual_savings(standard_savings_input)

        # 500 kW * 8000 hours * 0.85 efficiency = 3,400,000 kWh
        expected_kwh = 500.0 * 8000 * 0.85
        assert savings['annual_kwh_savings'] == pytest.approx(expected_kwh, rel=0.01)

    def test_annual_cost_savings(self, calculator, standard_savings_input):
        """Test annual cost savings in USD."""
        savings = calculator._calculate_annual_savings(standard_savings_input)

        # 3,400,000 kWh * $0.08/kWh = $272,000
        expected_cost = 500.0 * 8000 * 0.85 * 0.08
        assert savings['annual_cost_savings'] == pytest.approx(expected_cost, rel=0.01)

    def test_system_efficiency_effect(self, calculator):
        """Test system efficiency affects savings."""
        savings_high = EnergySavingsInput(
            heat_recovery_kw=500.0,
            operating_hours_per_year=8000,
            energy_cost_usd_per_kwh=0.08,
            system_efficiency=0.95  # High efficiency
        )
        savings_low = EnergySavingsInput(
            heat_recovery_kw=500.0,
            operating_hours_per_year=8000,
            energy_cost_usd_per_kwh=0.08,
            system_efficiency=0.70  # Low efficiency
        )

        result_high = calculator._calculate_annual_savings(savings_high)
        result_low = calculator._calculate_annual_savings(savings_low)

        assert result_high['annual_kwh_savings'] > result_low['annual_kwh_savings']
        ratio = result_high['annual_kwh_savings'] / result_low['annual_kwh_savings']
        assert ratio == pytest.approx(0.95 / 0.70, rel=0.01)


# ============================================================================
# SIMPLE PAYBACK TESTS
# ============================================================================

@pytest.mark.roi
class TestSimplePayback:
    """Test simple payback period calculation."""

    def test_simple_payback_calculation(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test simple payback period calculation."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # Simple payback = Capital / Net Annual Savings
        expected_payback = result.total_capital_cost_usd / result.net_annual_savings_usd

        assert result.simple_payback_years == pytest.approx(expected_payback, rel=0.01)

    def test_simple_payback_reasonable_range(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test simple payback is in reasonable range."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # For typical heat recovery project: 1-5 years
        assert 0.5 < result.simple_payback_years < 10


# ============================================================================
# NPV TESTS
# ============================================================================

@pytest.mark.roi
class TestNPVCalculation:
    """Test Net Present Value calculation."""

    def test_npv_calculation_formula(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test NPV calculation follows correct formula."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # NPV should be calculated
        assert result.npv_usd is not None

    def test_npv_positive_for_good_project(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test NPV is positive for economically viable project."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # For typical heat recovery project, NPV should be positive
        assert result.npv_usd > 0

    def test_npv_affected_by_discount_rate(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input
    ):
        """Test NPV decreases with higher discount rate."""
        low_discount = FinancialParameters(
            discount_rate_percent=5.0,
            analysis_period_years=15
        )
        high_discount = FinancialParameters(
            discount_rate_percent=20.0,
            analysis_period_years=15
        )

        result_low = calculator.calculate_roi(
            standard_capital_input, standard_operating_input,
            standard_savings_input, low_discount
        )
        result_high = calculator.calculate_roi(
            standard_capital_input, standard_operating_input,
            standard_savings_input, high_discount
        )

        # Higher discount rate should give lower NPV
        assert result_low.npv_usd > result_high.npv_usd

    def test_npv_increases_with_analysis_period(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input
    ):
        """Test NPV increases with longer analysis period."""
        short_period = FinancialParameters(
            discount_rate_percent=10.0,
            analysis_period_years=5
        )
        long_period = FinancialParameters(
            discount_rate_percent=10.0,
            analysis_period_years=20
        )

        result_short = calculator.calculate_roi(
            standard_capital_input, standard_operating_input,
            standard_savings_input, short_period
        )
        result_long = calculator.calculate_roi(
            standard_capital_input, standard_operating_input,
            standard_savings_input, long_period
        )

        # Longer period should give higher NPV
        assert result_long.npv_usd > result_short.npv_usd


# ============================================================================
# IRR TESTS
# ============================================================================

@pytest.mark.roi
class TestIRRCalculation:
    """Test Internal Rate of Return calculation."""

    def test_irr_calculation(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test IRR is calculated."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        assert result.irr_percent is not None

    def test_irr_greater_than_discount_rate_for_positive_npv(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test IRR > discount rate when NPV is positive."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        if result.npv_usd > 0:
            assert result.irr_percent > standard_financial_params.discount_rate_percent

    def test_irr_reasonable_range(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test IRR is in reasonable range."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # IRR should be positive and less than 100% for typical projects
        assert 0 < result.irr_percent < 100


# ============================================================================
# PROFITABILITY INDEX TESTS
# ============================================================================

@pytest.mark.roi
class TestProfitabilityIndex:
    """Test profitability index calculation."""

    def test_profitability_index_greater_than_one(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test PI > 1 for positive NPV project."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        if result.npv_usd > 0:
            assert result.profitability_index > 1.0

    def test_profitability_index_formula(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test PI = (NPV + Capital) / Capital."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        expected_pi = (result.npv_usd + result.total_capital_cost_usd) / result.total_capital_cost_usd

        assert result.profitability_index == pytest.approx(expected_pi, rel=0.01)


# ============================================================================
# DISCOUNTED PAYBACK TESTS
# ============================================================================

@pytest.mark.roi
class TestDiscountedPayback:
    """Test discounted payback period calculation."""

    def test_discounted_payback_greater_than_simple(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test discounted payback >= simple payback."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # Discounted payback should be longer (present value of future savings is less)
        assert result.discounted_payback_years >= result.simple_payback_years

    def test_discounted_payback_reasonable_range(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test discounted payback is in reasonable range."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # Should be less than analysis period for viable project
        assert result.discounted_payback_years < standard_financial_params.analysis_period_years


# ============================================================================
# CO2 EMISSION REDUCTION TESTS
# ============================================================================

@pytest.mark.roi
class TestCO2Reduction:
    """Test CO2 emission reduction calculations."""

    def test_lifetime_co2_reduction(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test lifetime CO2 reduction is calculated."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        assert result.lifetime_co2_reduction_tonnes > 0

    def test_co2_reduction_calculation(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test CO2 reduction calculation formula."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        # CO2 = kWh * emission_factor * years / 1000
        co2_factor = calculator.CO2_FACTORS['natural_gas']  # 0.18 kg CO2/kWh
        expected_co2 = (
            result.annual_energy_savings_kwh *
            standard_financial_params.analysis_period_years *
            co2_factor / 1000
        )

        assert result.lifetime_co2_reduction_tonnes == pytest.approx(expected_co2, rel=0.1)

    def test_cost_per_tonne_co2(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test cost per tonne CO2 avoided."""
        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        expected_cost_per_tonne = (
            result.total_capital_cost_usd / result.lifetime_co2_reduction_tonnes
        )

        assert result.cost_per_tonne_co2_avoided_usd == pytest.approx(expected_cost_per_tonne, rel=0.01)


# ============================================================================
# SENSITIVITY ANALYSIS TESTS
# ============================================================================

@pytest.mark.roi
class TestSensitivityAnalysis:
    """Test sensitivity analysis functionality."""

    def test_sensitivity_analysis_returns_results(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test sensitivity analysis returns results."""
        base_result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        sensitivity_results = calculator.perform_sensitivity_analysis(
            base_result,
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        assert len(sensitivity_results) > 0

    def test_sensitivity_analysis_parameters(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_savings_input, standard_financial_params
    ):
        """Test sensitivity analysis covers key parameters."""
        base_result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        sensitivity_results = calculator.perform_sensitivity_analysis(
            base_result,
            standard_capital_input,
            standard_operating_input,
            standard_savings_input,
            standard_financial_params
        )

        param_names = [r.parameter_name for r in sensitivity_results]

        # Should analyze key parameters
        assert 'energy cost' in param_names
        assert 'capital cost' in param_names
        assert 'discount rate' in param_names


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

@pytest.mark.roi
class TestROIEdgeCases:
    """Test ROI calculator edge cases."""

    def test_zero_operating_hours(self, calculator, standard_capital_input, standard_financial_params):
        """Test with zero operating hours."""
        operating = OperatingCostInput(
            maintenance_percent_of_capital=2.0,
            operating_hours_per_year=0
        )
        savings = EnergySavingsInput(
            heat_recovery_kw=500.0,
            operating_hours_per_year=0,
            energy_cost_usd_per_kwh=0.08
        )

        result = calculator.calculate_roi(
            standard_capital_input, operating, savings, standard_financial_params
        )

        # Zero operating hours = zero savings
        assert result.annual_energy_savings_kwh == 0
        assert result.simple_payback_years == 999  # No payback

    def test_very_high_energy_cost(
        self, calculator, standard_capital_input, standard_operating_input,
        standard_financial_params
    ):
        """Test with very high energy cost."""
        savings = EnergySavingsInput(
            heat_recovery_kw=500.0,
            operating_hours_per_year=8000,
            energy_cost_usd_per_kwh=0.50,  # Very high
            system_efficiency=0.85
        )

        result = calculator.calculate_roi(
            standard_capital_input,
            standard_operating_input,
            savings,
            standard_financial_params
        )

        # High energy cost = higher savings, shorter payback
        assert result.simple_payback_years < 3

    def test_very_small_heat_recovery(
        self, calculator, standard_operating_input,
        standard_financial_params
    ):
        """Test with very small heat recovery capacity."""
        capital = CapitalCostInput(
            equipment_type=EquipmentType.PLATE_HX,
            heat_capacity_kw=10.0,  # Very small
            material="carbon_steel"
        )
        savings = EnergySavingsInput(
            heat_recovery_kw=10.0,
            operating_hours_per_year=8000,
            energy_cost_usd_per_kwh=0.08
        )

        result = calculator.calculate_roi(
            capital,
            standard_operating_input,
            savings,
            standard_financial_params
        )

        # Small project should still calculate
        assert result.npv_usd is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "roi"])
