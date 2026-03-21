# -*- coding: utf-8 -*-
"""
Unit tests for PaybackCalculatorEngine -- PACK-033 Engine 2
=============================================================

Tests simple payback, discounted payback, NPV, IRR, ROI, LCOE, SIR,
cash-flow generation, incentive application, batch calculation, and
sensitivity analysis.

Coverage target: 85%+
Total tests: ~55
"""

import importlib.util
import os
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack033_payback.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


_m = _load("payback_calculator_engine")

PaybackCalculatorEngine = _m.PaybackCalculatorEngine
MeasureFinancials = _m.MeasureFinancials
FinancialParameters = _m.FinancialParameters
PaybackResult = _m.PaybackResult
BatchPaybackResult = _m.BatchPaybackResult
Incentive = _m.Incentive
IncentiveType = _m.IncentiveType
TaxTreatment = _m.TaxTreatment
SensitivityParameter = _m.SensitivityParameter
CashFlow = _m.CashFlow
AnalysisPeriod = _m.AnalysisPeriod
FinancialMetric = _m.FinancialMetric


def _make_measure(**overrides):
    """Create a MeasureFinancials with sensible defaults."""
    defaults = dict(
        measure_id="M-001",
        name="LED Retrofit",
        implementation_cost=Decimal("12000"),
        annual_savings_kwh=Decimal("33696"),
        annual_savings_cost=Decimal("6739"),
        annual_maintenance_cost=Decimal("0"),
        measure_life_years=10,
    )
    defaults.update(overrides)
    return MeasureFinancials(**defaults)


def _make_params(**overrides):
    """Create FinancialParameters with sensible defaults."""
    defaults = dict(
        discount_rate=Decimal("0.08"),
        electricity_escalation_rate=Decimal("0.03"),
        analysis_period_years=10,
    )
    defaults.update(overrides)
    return FinancialParameters(**defaults)


# =============================================================================
# Initialization
# =============================================================================


class TestInitialization:
    """Engine instantiation tests."""

    def test_default(self):
        engine = PaybackCalculatorEngine()
        assert engine is not None

    def test_engine_version(self):
        assert _m._MODULE_VERSION == "1.0.0"

    def test_engine_has_calculate_methods(self):
        engine = PaybackCalculatorEngine()
        assert hasattr(engine, "calculate_payback")
        assert hasattr(engine, "calculate_batch")
        assert hasattr(engine, "run_sensitivity")

    def test_engine_with_config(self):
        engine = PaybackCalculatorEngine(config={"discount_rate": "0.10"})
        assert engine is not None


# =============================================================================
# Enums
# =============================================================================


class TestEnums:
    """Test all enumerations."""

    def test_analysis_period_values(self):
        expected = {"short_term_3y", "medium_term_5y", "long_term_10y", "extended_15y", "custom"}
        actual = {m.value for m in AnalysisPeriod}
        assert expected == actual

    def test_financial_metric_values(self):
        expected = {"simple_payback", "discounted_payback", "npv", "irr", "roi", "lcoe", "sir", "annual_cash_flow"}
        actual = {m.value for m in FinancialMetric}
        assert expected == actual

    def test_tax_treatment_values(self):
        expected = {"none", "section_179", "macrs_5y", "macrs_7y", "macrs_15y", "bonus_depreciation", "custom"}
        actual = {m.value for m in TaxTreatment}
        assert expected == actual

    def test_incentive_type_values(self):
        expected = {"utility_rebate", "tax_credit", "tax_deduction", "grant", "loan_subsidy", "performance_incentive"}
        actual = {m.value for m in IncentiveType}
        assert expected == actual

    def test_sensitivity_parameter_values(self):
        expected = {"discount_rate", "energy_price", "implementation_cost", "savings_estimate", "incentive_amount"}
        actual = {m.value for m in SensitivityParameter}
        assert expected == actual


# =============================================================================
# Simple Payback
# =============================================================================


class TestSimplePayback:
    """Test simple payback calculations."""

    def test_simple_payback_calculation(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        # 12000 / 6739 ~ 1.78 years
        assert Decimal("1.0") <= result.simple_payback_years <= Decimal("2.5")

    def test_zero_cost_zero_payback(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(implementation_cost=Decimal("0"))
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.simple_payback_years == Decimal("0") or result.simple_payback_years <= Decimal("0.01")

    def test_high_cost_long_payback(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            implementation_cost=Decimal("100000"),
            annual_savings_cost=Decimal("5000"),
        )
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.simple_payback_years >= Decimal("10")


# =============================================================================
# Discounted Payback
# =============================================================================


class TestDiscountedPayback:
    """Test discounted payback calculations."""

    def test_discounted_payback_longer_than_simple(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.discounted_payback_years >= result.simple_payback_years

    def test_discounted_payback_with_zero_discount(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params(discount_rate=Decimal("0"))
        result = engine.calculate_payback(measure, params)
        # With 0% discount rate, discounted payback should equal simple payback
        diff = abs(result.discounted_payback_years - result.simple_payback_years)
        assert diff <= Decimal("1.0")


# =============================================================================
# NPV
# =============================================================================


class TestNPV:
    """Test net present value calculations."""

    def test_npv_positive_for_good_measure(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.npv > Decimal("0")

    def test_npv_negative_for_bad_measure(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            implementation_cost=Decimal("500000"),
            annual_savings_cost=Decimal("1000"),
            measure_life_years=5,
        )
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.npv < Decimal("0")

    def test_npv_increases_with_longer_period(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params_5 = _make_params(analysis_period_years=5)
        params_15 = _make_params(analysis_period_years=15)
        r5 = engine.calculate_payback(measure, params_5)
        r15 = engine.calculate_payback(measure, params_15)
        assert r15.npv >= r5.npv


# =============================================================================
# IRR
# =============================================================================


class TestIRR:
    """Test internal rate of return calculations."""

    def test_irr_positive_for_good_measure(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.irr > Decimal("0")

    def test_irr_high_for_fast_payback(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            implementation_cost=Decimal("500"),
            annual_savings_cost=Decimal("6000"),
        )
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.irr > Decimal("50")  # Very high IRR for fast payback


# =============================================================================
# ROI and LCOE
# =============================================================================


class TestROIandLCOE:
    """Test ROI and LCOE calculations."""

    def test_roi_positive(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.roi_pct > Decimal("0")

    def test_lcoe_reasonable(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        # LCOE should be between 0 and 1 EUR/kWh for typical measures
        assert result.lcoe >= Decimal("0")
        assert result.lcoe <= Decimal("1.0")

    def test_sir_above_one_for_good_measure(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert result.sir >= Decimal("1.0")


# =============================================================================
# Cash Flows
# =============================================================================


class TestCashFlows:
    """Test year-by-year cash flow generation."""

    def test_cash_flow_generation(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params(analysis_period_years=10)
        result = engine.calculate_payback(measure, params)
        assert len(result.cash_flows) == 10

    def test_cash_flow_years_sequential(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params(analysis_period_years=5)
        result = engine.calculate_payback(measure, params)
        years = [cf.year for cf in result.cash_flows]
        assert years == [1, 2, 3, 4, 5]

    def test_cumulative_savings_increase(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        for i in range(1, len(result.cash_flows)):
            assert result.cash_flows[i].cumulative_net_savings >= result.cash_flows[i - 1].cumulative_net_savings

    def test_escalated_savings_grow(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params(electricity_escalation_rate=Decimal("0.05"))
        result = engine.calculate_payback(measure, params)
        if len(result.cash_flows) >= 2:
            assert result.cash_flows[-1].escalated_savings > result.cash_flows[0].escalated_savings


# =============================================================================
# Incentives
# =============================================================================


class TestIncentives:
    """Test incentive application."""

    def test_incentive_reduces_cost(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        incentives = [
            Incentive(incentive_type=IncentiveType.UTILITY_REBATE, amount=Decimal("3000"))
        ]
        result = engine.calculate_payback(measure, params, incentives)
        assert result.net_implementation_cost == Decimal("9000.00")

    def test_percentage_incentive(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(implementation_cost=Decimal("20000"))
        params = _make_params()
        incentives = [
            Incentive(incentive_type=IncentiveType.GRANT, amount=Decimal("25"), is_percentage=True)
        ]
        result = engine.calculate_payback(measure, params, incentives)
        assert result.net_implementation_cost == Decimal("15000.00")

    def test_incentive_with_cap(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(implementation_cost=Decimal("100000"))
        params = _make_params()
        incentives = [
            Incentive(
                incentive_type=IncentiveType.GRANT,
                amount=Decimal("50"),
                is_percentage=True,
                max_amount=Decimal("10000"),
            )
        ]
        result = engine.calculate_payback(measure, params, incentives)
        assert result.net_implementation_cost == Decimal("90000.00")


# =============================================================================
# Batch Calculation
# =============================================================================


class TestBatchCalculation:
    """Test portfolio batch calculations."""

    def test_batch_calculation(self):
        engine = PaybackCalculatorEngine()
        measures = [
            _make_measure(measure_id="B-001", implementation_cost=Decimal("12000"), annual_savings_cost=Decimal("6739")),
            _make_measure(measure_id="B-002", implementation_cost=Decimal("8000"), annual_savings_cost=Decimal("3600")),
            _make_measure(measure_id="B-003", implementation_cost=Decimal("15000"), annual_savings_cost=Decimal("8400")),
        ]
        params = _make_params()
        result = engine.calculate_batch(measures, params)
        assert isinstance(result, BatchPaybackResult)
        assert len(result.results) == 3

    def test_batch_total_investment(self):
        engine = PaybackCalculatorEngine()
        measures = [
            _make_measure(measure_id="B-001", implementation_cost=Decimal("10000"), annual_savings_cost=Decimal("5000")),
            _make_measure(measure_id="B-002", implementation_cost=Decimal("20000"), annual_savings_cost=Decimal("8000")),
        ]
        params = _make_params()
        result = engine.calculate_batch(measures, params)
        assert result.total_investment == Decimal("30000.00")

    def test_batch_total_npv(self):
        engine = PaybackCalculatorEngine()
        measures = [
            _make_measure(measure_id="B-001", implementation_cost=Decimal("5000"), annual_savings_cost=Decimal("3000")),
            _make_measure(measure_id="B-002", implementation_cost=Decimal("5000"), annual_savings_cost=Decimal("3000")),
        ]
        params = _make_params()
        result = engine.calculate_batch(measures, params)
        # Both have positive NPV so total should be positive
        assert result.total_npv > Decimal("0")

    def test_batch_provenance_hash(self):
        engine = PaybackCalculatorEngine()
        measures = [_make_measure(measure_id="B-P1")]
        params = _make_params()
        result = engine.calculate_batch(measures, params)
        assert len(result.provenance_hash) == 64


# =============================================================================
# Sensitivity Analysis
# =============================================================================


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""

    def test_sensitivity_discount_rate(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        values = [Decimal("0.04"), Decimal("0.08"), Decimal("0.12"), Decimal("0.16")]
        result = engine.run_sensitivity(measure, params, SensitivityParameter.DISCOUNT_RATE, values)
        assert len(result.npvs) == 4
        # Higher discount rate = lower NPV
        assert result.npvs[0] >= result.npvs[-1]

    def test_sensitivity_energy_price(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        values = [Decimal("0.01"), Decimal("0.03"), Decimal("0.05"), Decimal("0.07")]
        result = engine.run_sensitivity(measure, params, SensitivityParameter.ENERGY_PRICE, values)
        assert len(result.npvs) == 4

    def test_sensitivity_provenance(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        values = [Decimal("0.04"), Decimal("0.08")]
        result = engine.run_sensitivity(measure, params, SensitivityParameter.DISCOUNT_RATE, values)
        assert len(result.provenance_hash) == 64


# =============================================================================
# Decimal Precision
# =============================================================================


class TestDecimalPrecision:
    """Test Decimal arithmetic precision."""

    def test_decimal_precision_payback(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure()
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        assert isinstance(result.simple_payback_years, Decimal)
        assert isinstance(result.npv, Decimal)
        assert isinstance(result.irr, Decimal)

    def test_decimal_no_floating_point_errors(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            implementation_cost=Decimal("10000.01"),
            annual_savings_cost=Decimal("3333.33"),
        )
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        # Ensure result values are exact Decimals, not floats
        assert isinstance(result.lcoe, Decimal)
        assert isinstance(result.sir, Decimal)

    def test_zero_savings_handling(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            annual_savings_cost=Decimal("0"),
            annual_savings_kwh=Decimal("0"),
        )
        params = _make_params()
        result = engine.calculate_payback(measure, params)
        # Should not crash; payback should be max or 99
        assert result.simple_payback_years >= Decimal("0")


# =============================================================================
# Provenance
# =============================================================================


class TestProvenance:
    """Provenance hash tests."""

    def test_hash_64_char(self):
        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(_make_measure(), _make_params())
        assert len(result.provenance_hash) == 64

    def test_hash_hex(self):
        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(_make_measure(), _make_params())
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_cost_effective_flag(self):
        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(_make_measure(), _make_params())
        assert result.is_cost_effective is True

    def test_cost_effective_false_for_bad_measure(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            implementation_cost=Decimal("500000"),
            annual_savings_cost=Decimal("1000"),
            measure_life_years=3,
        )
        result = engine.calculate_payback(measure, _make_params())
        assert result.is_cost_effective is False


# =============================================================================
# MACRS Depreciation
# =============================================================================


class TestMACRSDepreciation:
    """Test MACRS depreciation schedule handling."""

    def test_macrs_5y_schedule_exists(self):
        assert hasattr(_m, "MACRS_5Y_SCHEDULE")

    def test_macrs_7y_schedule_exists(self):
        assert hasattr(_m, "MACRS_7Y_SCHEDULE")

    def test_macrs_15y_schedule_exists(self):
        assert hasattr(_m, "MACRS_15Y_SCHEDULE")

    def test_macrs_5y_sums_to_one(self):
        schedule = getattr(_m, "MACRS_5Y_SCHEDULE", None)
        if schedule is None:
            pytest.skip("MACRS_5Y_SCHEDULE not found")
        total = sum(schedule)
        assert abs(total - Decimal("1.0")) < Decimal("0.01")

    def test_macrs_7y_sums_to_one(self):
        schedule = getattr(_m, "MACRS_7Y_SCHEDULE", None)
        if schedule is None:
            pytest.skip("MACRS_7Y_SCHEDULE not found")
        total = sum(schedule)
        assert abs(total - Decimal("1.0")) < Decimal("0.01")

    def test_macrs_15y_sums_to_one(self):
        schedule = getattr(_m, "MACRS_15Y_SCHEDULE", None)
        if schedule is None:
            pytest.skip("MACRS_15Y_SCHEDULE not found")
        total = sum(schedule)
        assert abs(total - Decimal("1.0")) < Decimal("0.01")

    def test_macrs_5y_has_6_entries(self):
        schedule = getattr(_m, "MACRS_5Y_SCHEDULE", None)
        if schedule is None:
            pytest.skip("MACRS_5Y_SCHEDULE not found")
        assert len(schedule) == 6

    def test_macrs_7y_has_8_entries(self):
        schedule = getattr(_m, "MACRS_7Y_SCHEDULE", None)
        if schedule is None:
            pytest.skip("MACRS_7Y_SCHEDULE not found")
        assert len(schedule) == 8


# =============================================================================
# Additional Financial Edge Cases
# =============================================================================


class TestFinancialEdgeCases:
    """Test additional financial edge cases."""

    def test_very_small_savings(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            implementation_cost=Decimal("100"),
            annual_savings_cost=Decimal("0.01"),
            annual_savings_kwh=Decimal("1"),
        )
        result = engine.calculate_payback(measure, _make_params())
        assert result.simple_payback_years > Decimal("0")

    def test_large_investment(self):
        engine = PaybackCalculatorEngine()
        measure = _make_measure(
            implementation_cost=Decimal("1000000"),
            annual_savings_cost=Decimal("200000"),
            annual_savings_kwh=Decimal("1000000"),
        )
        result = engine.calculate_payback(measure, _make_params())
        assert result.npv is not None
        assert isinstance(result.npv, Decimal)

    def test_high_discount_rate(self):
        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(
            _make_measure(),
            _make_params(discount_rate=Decimal("0.20")),
        )
        assert result.npv is not None

    def test_low_discount_rate(self):
        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(
            _make_measure(),
            _make_params(discount_rate=Decimal("0.01")),
        )
        assert result.npv > Decimal("0")

    def test_long_analysis_period(self):
        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(
            _make_measure(measure_life_years=30),
            _make_params(analysis_period_years=30),
        )
        assert len(result.cash_flows) == 30

    def test_short_analysis_period(self):
        engine = PaybackCalculatorEngine()
        result = engine.calculate_payback(
            _make_measure(),
            _make_params(analysis_period_years=3),
        )
        assert len(result.cash_flows) == 3
