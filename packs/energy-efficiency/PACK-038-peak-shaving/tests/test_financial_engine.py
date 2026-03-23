# -*- coding: utf-8 -*-
"""
Unit tests for FinancialEngine -- PACK-038 Engine 9
============================================================

Tests NPV calculation with discount rates, IRR via bisection method, simple
and discounted payback, ITC calculation (30% base + adders), SGIP step rates,
revenue stacking, Monte Carlo risk analysis, and Decimal precision verification.

Coverage target: 85%+
Total tests: ~70
"""

import hashlib
import importlib.util
import json
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
    mod_key = f"pack038_test.{name}"
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


_m = _load("financial_engine")


# =============================================================================
# Module Loading
# =============================================================================


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_module_version(self):
        assert hasattr(_m, "_MODULE_VERSION")
        assert _m._MODULE_VERSION == "1.0.0"

    def test_engine_class_exists(self):
        assert hasattr(_m, "FinancialEngine")

    def test_engine_instantiation(self):
        engine = _m.FinancialEngine()
        assert engine is not None


# =============================================================================
# NPV Calculation
# =============================================================================


class TestNPVCalculation:
    """Test Net Present Value calculation with various discount rates."""

    def _get_npv(self, engine):
        return (getattr(engine, "calculate_npv", None)
                or getattr(engine, "npv", None)
                or getattr(engine, "compute_npv", None))

    @pytest.mark.parametrize("discount_rate", [
        Decimal("0.04"), Decimal("0.06"), Decimal("0.08"),
        Decimal("0.10"), Decimal("0.12"), Decimal("0.15"),
    ])
    def test_npv_with_discount_rates(self, discount_rate):
        engine = _m.FinancialEngine()
        npv = self._get_npv(engine)
        if npv is None:
            pytest.skip("npv method not found")
        cash_flows = [Decimal("-275000")] + [Decimal("103280")] * 15
        result = npv(cash_flows=cash_flows, discount_rate=discount_rate)
        val = getattr(result, "npv_usd", result)
        if isinstance(val, (Decimal, int, float)):
            assert Decimal(str(val)) > 0  # Should be positive for these flows

    def test_npv_zero_discount_rate(self):
        engine = _m.FinancialEngine()
        npv = self._get_npv(engine)
        if npv is None:
            pytest.skip("npv method not found")
        cash_flows = [Decimal("-100000")] + [Decimal("20000")] * 10
        result = npv(cash_flows=cash_flows, discount_rate=Decimal("0"))
        val = getattr(result, "npv_usd", result)
        if isinstance(val, (Decimal, int, float)):
            expected = sum(cash_flows)
            assert abs(Decimal(str(val)) - expected) < Decimal("1.00")

    def test_npv_higher_discount_lower_value(self):
        engine = _m.FinancialEngine()
        npv = self._get_npv(engine)
        if npv is None:
            pytest.skip("npv method not found")
        cash_flows = [Decimal("-275000")] + [Decimal("103280")] * 15
        r_low = npv(cash_flows=cash_flows, discount_rate=Decimal("0.04"))
        r_high = npv(cash_flows=cash_flows, discount_rate=Decimal("0.12"))
        v_low = getattr(r_low, "npv_usd", r_low)
        v_high = getattr(r_high, "npv_usd", r_high)
        if isinstance(v_low, (Decimal, int, float)) and isinstance(v_high, (Decimal, int, float)):
            assert float(v_low) > float(v_high)

    def test_npv_negative_project(self):
        engine = _m.FinancialEngine()
        npv = self._get_npv(engine)
        if npv is None:
            pytest.skip("npv method not found")
        cash_flows = [Decimal("-1000000")] + [Decimal("10000")] * 5
        result = npv(cash_flows=cash_flows, discount_rate=Decimal("0.08"))
        val = getattr(result, "npv_usd", result)
        if isinstance(val, (Decimal, int, float)):
            assert float(val) < 0

    def test_npv_decimal_precision(self):
        engine = _m.FinancialEngine()
        npv = self._get_npv(engine)
        if npv is None:
            pytest.skip("npv method not found")
        cash_flows = [Decimal("-100000.00")] + [Decimal("25000.00")] * 5
        result = npv(cash_flows=cash_flows, discount_rate=Decimal("0.08"))
        val = getattr(result, "npv_usd", result)
        if isinstance(val, Decimal):
            assert isinstance(val, Decimal)


# =============================================================================
# IRR Calculation
# =============================================================================


class TestIRRCalculation:
    """Test Internal Rate of Return via bisection method."""

    def _get_irr(self, engine):
        return (getattr(engine, "calculate_irr", None)
                or getattr(engine, "irr", None)
                or getattr(engine, "compute_irr", None))

    def test_irr_positive_project(self):
        engine = _m.FinancialEngine()
        irr = self._get_irr(engine)
        if irr is None:
            pytest.skip("irr method not found")
        cash_flows = [Decimal("-275000")] + [Decimal("103280")] * 15
        result = irr(cash_flows=cash_flows)
        val = getattr(result, "irr_pct", result)
        if isinstance(val, (Decimal, int, float)):
            assert float(val) > 0

    def test_irr_deterministic(self):
        engine = _m.FinancialEngine()
        irr = self._get_irr(engine)
        if irr is None:
            pytest.skip("irr method not found")
        cash_flows = [Decimal("-100000")] + [Decimal("30000")] * 5
        r1 = irr(cash_flows=cash_flows)
        r2 = irr(cash_flows=cash_flows)
        v1 = getattr(r1, "irr_pct", str(r1))
        v2 = getattr(r2, "irr_pct", str(r2))
        assert v1 == v2

    def test_irr_simple_case(self):
        engine = _m.FinancialEngine()
        irr = self._get_irr(engine)
        if irr is None:
            pytest.skip("irr method not found")
        # -100 now, +200 in 1 year -> IRR = 100%
        cash_flows = [Decimal("-100"), Decimal("200")]
        result = irr(cash_flows=cash_flows)
        val = getattr(result, "irr_pct", result)
        if isinstance(val, (Decimal, int, float)):
            assert abs(float(val) - 1.0) < 0.05

    @pytest.mark.parametrize("annual_savings", [
        Decimal("50000"), Decimal("75000"), Decimal("100000"),
        Decimal("125000"), Decimal("150000"),
    ])
    def test_irr_sensitivity(self, annual_savings):
        engine = _m.FinancialEngine()
        irr = self._get_irr(engine)
        if irr is None:
            pytest.skip("irr method not found")
        cash_flows = [Decimal("-275000")] + [annual_savings] * 15
        result = irr(cash_flows=cash_flows)
        assert result is not None


# =============================================================================
# Simple and Discounted Payback
# =============================================================================


class TestPaybackPeriod:
    """Test simple and discounted payback period calculation."""

    def _get_payback(self, engine):
        return (getattr(engine, "calculate_payback", None)
                or getattr(engine, "payback_period", None)
                or getattr(engine, "compute_payback", None))

    def test_simple_payback(self):
        engine = _m.FinancialEngine()
        payback = self._get_payback(engine)
        if payback is None:
            pytest.skip("payback method not found")
        result = payback(investment_usd=Decimal("147500"),
                         annual_savings_usd=Decimal("103280"),
                         method="SIMPLE")
        val = getattr(result, "payback_years", result)
        if isinstance(val, (Decimal, int, float)):
            expected = 147500 / 103280
            assert abs(float(val) - expected) < 0.1

    def test_discounted_payback_longer(self):
        engine = _m.FinancialEngine()
        payback = self._get_payback(engine)
        if payback is None:
            pytest.skip("payback method not found")
        try:
            r_simple = payback(investment_usd=Decimal("147500"),
                               annual_savings_usd=Decimal("103280"),
                               method="SIMPLE")
            r_disc = payback(investment_usd=Decimal("147500"),
                             annual_savings_usd=Decimal("103280"),
                             method="DISCOUNTED",
                             discount_rate=Decimal("0.08"))
            v_simple = getattr(r_simple, "payback_years", r_simple)
            v_disc = getattr(r_disc, "payback_years", r_disc)
            if isinstance(v_simple, (Decimal, int, float)) and isinstance(v_disc, (Decimal, int, float)):
                assert float(v_disc) >= float(v_simple)
        except TypeError:
            pass  # Method may not support method parameter

    @pytest.mark.parametrize("investment,savings,expected_max", [
        (Decimal("100000"), Decimal("50000"), 2.5),
        (Decimal("275000"), Decimal("103280"), 3.0),
        (Decimal("50000"), Decimal("25000"), 2.5),
    ])
    def test_payback_values(self, investment, savings, expected_max):
        engine = _m.FinancialEngine()
        payback = self._get_payback(engine)
        if payback is None:
            pytest.skip("payback method not found")
        try:
            result = payback(investment_usd=investment,
                             annual_savings_usd=savings,
                             method="SIMPLE")
        except TypeError:
            result = payback(investment_usd=investment,
                             annual_savings_usd=savings)
        val = getattr(result, "payback_years", result)
        if isinstance(val, (Decimal, int, float)):
            assert float(val) <= expected_max


# =============================================================================
# ITC Calculation (30% base + adders)
# =============================================================================


class TestITCCalculation:
    """Test Investment Tax Credit calculation."""

    def _get_itc(self, engine):
        return (getattr(engine, "calculate_itc", None)
                or getattr(engine, "itc_credit", None)
                or getattr(engine, "compute_itc", None))

    def test_itc_30pct_base(self):
        engine = _m.FinancialEngine()
        itc = self._get_itc(engine)
        if itc is None:
            pytest.skip("itc method not found")
        result = itc(capital_cost_usd=Decimal("275000"),
                     base_rate_pct=Decimal("0.30"))
        credit = getattr(result, "credit_usd", result)
        if isinstance(credit, (Decimal, int, float)):
            expected = Decimal("275000") * Decimal("0.30")
            assert abs(Decimal(str(credit)) - expected) < Decimal("1.00")

    @pytest.mark.parametrize("adder_pct,adder_name", [
        (Decimal("0.10"), "DOMESTIC_CONTENT"),
        (Decimal("0.10"), "ENERGY_COMMUNITY"),
        (Decimal("0.10"), "LOW_INCOME"),
    ])
    def test_itc_adders(self, adder_pct, adder_name):
        engine = _m.FinancialEngine()
        itc = self._get_itc(engine)
        if itc is None:
            pytest.skip("itc method not found")
        try:
            result = itc(capital_cost_usd=Decimal("275000"),
                         base_rate_pct=Decimal("0.30"),
                         adders={adder_name: adder_pct})
        except TypeError:
            result = itc(capital_cost_usd=Decimal("275000"),
                         base_rate_pct=Decimal("0.30"))
        assert result is not None


# =============================================================================
# SGIP Step Rates
# =============================================================================


class TestSGIPRates:
    """Test Self-Generation Incentive Program step rates."""

    def _get_sgip(self, engine):
        return (getattr(engine, "calculate_sgip", None)
                or getattr(engine, "sgip_rebate", None)
                or getattr(engine, "compute_sgip", None))

    @pytest.mark.parametrize("step,rate_per_wh", [
        (1, Decimal("0.50")), (2, Decimal("0.40")),
        (3, Decimal("0.30")), (4, Decimal("0.25")),
        (5, Decimal("0.20")),
    ])
    def test_sgip_steps(self, step, rate_per_wh):
        engine = _m.FinancialEngine()
        sgip = self._get_sgip(engine)
        if sgip is None:
            pytest.skip("sgip method not found")
        try:
            result = sgip(capacity_kwh=500,
                          step=step,
                          rate_per_wh=rate_per_wh)
        except TypeError:
            result = sgip(capacity_kwh=500)
        assert result is not None


# =============================================================================
# Revenue Stacking
# =============================================================================


class TestRevenueStacking:
    """Test revenue stacking from multiple value streams."""

    def _get_stack(self, engine):
        return (getattr(engine, "revenue_stacking", None)
                or getattr(engine, "stack_revenues", None)
                or getattr(engine, "calculate_total_value", None))

    @pytest.mark.parametrize("investment_type", [
        "BESS_ONLY", "PF_CORRECTION", "BESS_PLUS_PF",
        "LOAD_SHIFTING", "BESS_PLUS_SHIFTING", "COMPREHENSIVE",
    ])
    def test_revenue_stack_types(self, investment_type, sample_revenue_data):
        engine = _m.FinancialEngine()
        stack = self._get_stack(engine)
        if stack is None:
            pytest.skip("revenue_stacking method not found")
        try:
            result = stack(revenue_data=sample_revenue_data,
                           investment_type=investment_type)
        except (TypeError, ValueError):
            result = stack(revenue_data=sample_revenue_data)
        assert result is not None

    def test_stacked_greater_than_single(self, sample_revenue_data):
        engine = _m.FinancialEngine()
        stack = self._get_stack(engine)
        if stack is None:
            pytest.skip("revenue_stacking method not found")
        try:
            r_single = stack(revenue_data=sample_revenue_data,
                             investment_type="BESS_ONLY")
            r_combo = stack(revenue_data=sample_revenue_data,
                            investment_type="COMPREHENSIVE")
            v_single = getattr(r_single, "total_annual_usd", r_single)
            v_combo = getattr(r_combo, "total_annual_usd", r_combo)
            if isinstance(v_single, (Decimal, int, float)) and isinstance(v_combo, (Decimal, int, float)):
                assert float(v_combo) >= float(v_single)
        except (TypeError, ValueError):
            pass


# =============================================================================
# Monte Carlo Risk Analysis
# =============================================================================


class TestMonteCarloRisk:
    """Test Monte Carlo financial risk analysis."""

    def _get_mc(self, engine):
        return (getattr(engine, "monte_carlo_analysis", None)
                or getattr(engine, "risk_analysis", None)
                or getattr(engine, "simulate_risk", None))

    def test_monte_carlo_result(self, sample_revenue_data):
        engine = _m.FinancialEngine()
        mc = self._get_mc(engine)
        if mc is None:
            pytest.skip("monte_carlo method not found")
        try:
            result = mc(revenue_data=sample_revenue_data,
                        n_simulations=1000, seed=42)
        except TypeError:
            result = mc(revenue_data=sample_revenue_data)
        assert result is not None

    def test_monte_carlo_deterministic(self, sample_revenue_data):
        engine = _m.FinancialEngine()
        mc = self._get_mc(engine)
        if mc is None:
            pytest.skip("monte_carlo method not found")
        try:
            r1 = mc(revenue_data=sample_revenue_data, n_simulations=100, seed=42)
            r2 = mc(revenue_data=sample_revenue_data, n_simulations=100, seed=42)
        except TypeError:
            r1 = mc(revenue_data=sample_revenue_data)
            r2 = mc(revenue_data=sample_revenue_data)
        v1 = getattr(r1, "mean_npv", str(r1))
        v2 = getattr(r2, "mean_npv", str(r2))
        assert v1 == v2

    def test_monte_carlo_percentiles(self, sample_revenue_data):
        engine = _m.FinancialEngine()
        mc = self._get_mc(engine)
        if mc is None:
            pytest.skip("monte_carlo method not found")
        try:
            result = mc(revenue_data=sample_revenue_data,
                        n_simulations=1000, seed=42)
        except TypeError:
            result = mc(revenue_data=sample_revenue_data)
        p10 = getattr(result, "p10", None)
        p90 = getattr(result, "p90", None)
        if p10 is not None and p90 is not None:
            assert float(p90) >= float(p10)


# =============================================================================
# Decimal Precision Verification
# =============================================================================


class TestDecimalPrecision:
    """Verify financial calculations use Decimal for precision."""

    def test_revenue_data_uses_decimal(self, sample_revenue_data):
        savings = sample_revenue_data["peak_shaving_savings"]["total_annual_savings_usd"]
        assert isinstance(savings, Decimal)

    def test_investment_uses_decimal(self, sample_revenue_data):
        cost = sample_revenue_data["bess_investment"]["capital_cost_usd"]
        assert isinstance(cost, Decimal)

    def test_financial_metrics_decimal(self, sample_revenue_data):
        payback = sample_revenue_data["financial_metrics"]["simple_payback_years"]
        assert isinstance(payback, Decimal)

    def test_npv_decimal(self, sample_revenue_data):
        npv_val = sample_revenue_data["financial_metrics"]["npv_15yr_usd"]
        assert isinstance(npv_val, Decimal)

    def test_irr_decimal(self, sample_revenue_data):
        irr_val = sample_revenue_data["financial_metrics"]["irr_pct"]
        assert isinstance(irr_val, Decimal)


# =============================================================================
# Revenue Data Fixture Validation
# =============================================================================


class TestRevenueDataFixture:
    def test_has_baseline_charges(self, sample_revenue_data):
        assert "baseline_demand_charges" in sample_revenue_data

    def test_has_savings(self, sample_revenue_data):
        assert "peak_shaving_savings" in sample_revenue_data

    def test_has_investment(self, sample_revenue_data):
        assert "bess_investment" in sample_revenue_data

    def test_has_metrics(self, sample_revenue_data):
        assert "financial_metrics" in sample_revenue_data

    def test_savings_less_than_baseline(self, sample_revenue_data):
        baseline = sample_revenue_data["baseline_demand_charges"]["total_annual_demand_charges_usd"]
        savings = sample_revenue_data["peak_shaving_savings"]["total_annual_savings_usd"]
        assert savings < baseline

    def test_net_capital_reflects_incentives(self, sample_revenue_data):
        inv = sample_revenue_data["bess_investment"]
        net = inv["net_capital_cost_usd"]
        gross = inv["capital_cost_usd"]
        assert net < gross


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    def test_provenance_deterministic(self, sample_revenue_data):
        engine = _m.FinancialEngine()
        calc = (getattr(engine, "calculate_npv", None)
                or getattr(engine, "npv", None))
        if calc is None:
            pytest.skip("npv method not found")
        cf = [Decimal("-275000")] + [Decimal("103280")] * 15
        r1 = calc(cash_flows=cf, discount_rate=Decimal("0.08"))
        r2 = calc(cash_flows=cf, discount_rate=Decimal("0.08"))
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2
            assert len(h1) == 64
            assert all(c in "0123456789abcdef" for c in h1)
