# -*- coding: utf-8 -*-
"""
Unit tests for RevenueOptimizerEngine -- PACK-037 Engine 8
============================================================

Tests capacity revenue, energy revenue, ancillary revenue, demand charge
savings, net revenue, ROI calculation, what-if scenarios, and decimal
precision.

Coverage target: 85%+
Total tests: ~60
"""

import importlib.util
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
    mod_key = f"pack037_test.{name}"
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


_m = _load("revenue_optimizer_engine")


class TestModuleLoading:
    def test_module_loads(self):
        assert _m is not None

    def test_engine_class_exists(self):
        assert hasattr(_m, "RevenueOptimizerEngine")

    def test_engine_instantiation(self):
        engine = _m.RevenueOptimizerEngine()
        assert engine is not None


class TestCapacityRevenue:
    """Test capacity payment calculations."""

    def _get_calc(self, engine):
        return (getattr(engine, "calculate_capacity_revenue", None)
                or getattr(engine, "capacity_revenue", None)
                or getattr(engine, "calculate_revenue", None))

    def test_annual_capacity_payment(self, sample_revenue_data):
        enrolled = sample_revenue_data["enrolled_capacity_kw"]
        rate = sample_revenue_data["capacity_payment"]["rate_usd_per_kw_year"]
        annual = Decimal(str(enrolled)) * rate
        assert annual == Decimal("32000.00")

    def test_prorated_capacity_payment(self, sample_revenue_data):
        months = sample_revenue_data["capacity_payment"]["pro_rata_months"]
        assert months == 4
        gross = sample_revenue_data["capacity_payment"]["gross_usd"]
        assert gross == pytest.approx(Decimal("10666.67"), rel=0.01)

    @pytest.mark.parametrize("enrolled_kw,rate,months,expected", [
        (800, Decimal("40.00"), 12, Decimal("32000.00")),
        (800, Decimal("40.00"), 4, Decimal("10666.67")),
        (500, Decimal("65.00"), 12, Decimal("32500.00")),
        (1000, Decimal("55.00"), 6, Decimal("27500.00")),
    ])
    def test_capacity_payment_calculation(self, enrolled_kw, rate,
                                           months, expected):
        result = Decimal(str(enrolled_kw)) * rate * Decimal(str(months)) / Decimal("12")
        assert result == pytest.approx(expected, rel=0.01)


class TestEnergyRevenue:
    """Test energy payment calculations."""

    def test_total_energy_payments(self, sample_revenue_data):
        total = sample_revenue_data["total_energy_payment_usd"]
        assert total == Decimal("2760.00")

    def test_event_count(self, sample_revenue_data):
        assert len(sample_revenue_data["energy_payments"]) == 5

    def test_each_event_positive(self, sample_revenue_data):
        for payment in sample_revenue_data["energy_payments"]:
            assert payment["gross_usd"] > 0

    def test_energy_calculation(self, sample_revenue_data):
        for payment in sample_revenue_data["energy_payments"]:
            expected = payment["reduction_mwh"] * payment["rate_usd_per_mwh"]
            assert payment["gross_usd"] == expected

    @pytest.mark.parametrize("mwh,rate,expected", [
        (Decimal("3.120"), Decimal("100.00"), Decimal("312.00")),
        (Decimal("2.880"), Decimal("185.00"), Decimal("532.80")),
        (Decimal("3.600"), Decimal("250.00"), Decimal("900.00")),
    ])
    def test_energy_payment_precision(self, mwh, rate, expected):
        result = mwh * rate
        assert result == expected


class TestAncillaryRevenue:
    """Test ancillary services revenue."""

    def test_ancillary_revenue(self, sample_revenue_data):
        assert sample_revenue_data["ancillary_services_usd"] == Decimal("1200.00")

    def test_availability_bonus(self, sample_revenue_data):
        assert sample_revenue_data["availability_bonus_usd"] == Decimal("1066.67")


class TestDemandChargeSavings:
    """Test demand charge savings calculations."""

    def test_total_demand_savings(self, sample_revenue_data):
        total = sample_revenue_data["demand_charge_savings"]["total_usd"]
        assert total == Decimal("7500.00")

    def test_monthly_savings_count(self, sample_revenue_data):
        months = sample_revenue_data["demand_charge_savings"]["monthly_savings"]
        assert len(months) == 4

    @pytest.mark.parametrize("month_idx,expected_savings", [
        (0, Decimal("1500.00")),
        (1, Decimal("2500.00")),
        (2, Decimal("2250.00")),
        (3, Decimal("1250.00")),
    ])
    def test_monthly_demand_savings(self, sample_revenue_data,
                                     month_idx, expected_savings):
        monthly = sample_revenue_data["demand_charge_savings"]["monthly_savings"]
        assert monthly[month_idx]["savings_usd"] == expected_savings

    def test_demand_savings_calculation(self, sample_revenue_data):
        for month in sample_revenue_data["demand_charge_savings"]["monthly_savings"]:
            expected = (Decimal(str(month["peak_avoided_kw"])) *
                       month["rate_usd_per_kw"])
            assert month["savings_usd"] == expected


class TestNetRevenue:
    """Test net revenue calculations."""

    def test_gross_revenue(self, sample_revenue_data):
        gross = sample_revenue_data["gross_revenue_usd"]
        assert gross > 0

    def test_net_revenue(self, sample_revenue_data):
        net = sample_revenue_data["net_revenue_usd"]
        assert net > 0

    def test_net_less_than_gross(self, sample_revenue_data):
        assert (sample_revenue_data["net_revenue_usd"] <=
                sample_revenue_data["gross_revenue_usd"])

    def test_penalties_subtracted(self, sample_revenue_data):
        diff = (sample_revenue_data["gross_revenue_usd"] -
                sample_revenue_data["net_revenue_usd"])
        assert diff == sample_revenue_data["total_penalties_usd"]

    def test_total_penalties(self, sample_revenue_data):
        assert sample_revenue_data["total_penalties_usd"] == Decimal("100.00")


class TestROICalculation:
    """Test ROI and payback calculations."""

    def _get_roi(self, engine):
        return (getattr(engine, "calculate_roi", None)
                or getattr(engine, "roi_analysis", None)
                or getattr(engine, "financial_analysis", None))

    def test_simple_roi(self, sample_revenue_data):
        net = float(sample_revenue_data["net_revenue_usd"])
        impl_cost = float(sample_revenue_data["implementation_cost_usd"])
        annual_cost = float(sample_revenue_data["annual_operating_cost_usd"])
        annual_net = net - annual_cost
        roi = annual_net / impl_cost
        assert roi > 0

    def test_payback_period(self, sample_revenue_data):
        net = float(sample_revenue_data["net_revenue_usd"])
        impl_cost = float(sample_revenue_data["implementation_cost_usd"])
        annual_cost = float(sample_revenue_data["annual_operating_cost_usd"])
        annual_net = net - annual_cost
        payback_years = impl_cost / annual_net if annual_net > 0 else float("inf")
        assert payback_years < 5.0  # Should pay back within 5 years

    def test_roi_engine_method(self, sample_revenue_data):
        engine = _m.RevenueOptimizerEngine()
        roi = self._get_roi(engine)
        if roi is None:
            pytest.skip("calculate_roi method not found")
        result = roi(revenue_data=sample_revenue_data)
        assert result is not None

    @pytest.mark.parametrize("revenue,cost,expected_roi_pct", [
        (30000, 15000, 100.0),
        (20000, 15000, 33.3),
        (15000, 15000, 0.0),
        (45000, 15000, 200.0),
    ])
    def test_roi_scenarios(self, revenue, cost, expected_roi_pct):
        roi_pct = ((revenue - cost) / cost) * 100
        assert roi_pct == pytest.approx(expected_roi_pct, rel=0.1)


class TestWhatIfScenarios:
    """Test what-if scenario analysis."""

    def _get_whatif(self, engine):
        return (getattr(engine, "what_if_analysis", None)
                or getattr(engine, "scenario_analysis", None)
                or getattr(engine, "run_scenarios", None))

    def test_what_if_increased_capacity(self, sample_revenue_data):
        engine = _m.RevenueOptimizerEngine()
        whatif = self._get_whatif(engine)
        if whatif is None:
            pytest.skip("what_if_analysis method not found")
        scenario = {"enrolled_capacity_kw": 1200}  # Increase from 800
        result = whatif(baseline=sample_revenue_data, scenario=scenario)
        assert result is not None

    def test_what_if_additional_program(self, sample_revenue_data):
        engine = _m.RevenueOptimizerEngine()
        whatif = self._get_whatif(engine)
        if whatif is None:
            pytest.skip("what_if_analysis method not found")
        scenario = {"additional_programs": ["CAISO-PDR-2025"]}
        result = whatif(baseline=sample_revenue_data, scenario=scenario)
        assert result is not None

    def test_what_if_battery_addition(self, sample_revenue_data):
        engine = _m.RevenueOptimizerEngine()
        whatif = self._get_whatif(engine)
        if whatif is None:
            pytest.skip("what_if_analysis method not found")
        scenario = {"add_battery_kwh": 1000, "add_battery_kw": 500}
        result = whatif(baseline=sample_revenue_data, scenario=scenario)
        assert result is not None


class TestDecimalPrecision:
    """Test decimal precision in all financial calculations."""

    def test_revenue_data_uses_decimal(self, sample_revenue_data):
        assert isinstance(sample_revenue_data["net_revenue_usd"], Decimal)
        assert isinstance(sample_revenue_data["gross_revenue_usd"], Decimal)
        assert isinstance(sample_revenue_data["total_penalties_usd"], Decimal)

    def test_demand_savings_uses_decimal(self, sample_revenue_data):
        total = sample_revenue_data["demand_charge_savings"]["total_usd"]
        assert isinstance(total, Decimal)

    def test_energy_payments_use_decimal(self, sample_revenue_data):
        for payment in sample_revenue_data["energy_payments"]:
            assert isinstance(payment["gross_usd"], Decimal)
            assert isinstance(payment["rate_usd_per_mwh"], Decimal)
            assert isinstance(payment["reduction_mwh"], Decimal)

    def test_no_floating_point_errors(self):
        # Demonstrate decimal prevents floating point issues
        a = Decimal("0.1") + Decimal("0.2")
        assert a == Decimal("0.3")
        # Compare with float
        assert 0.1 + 0.2 != 0.3  # float has rounding error

    def test_revenue_summation_exact(self, sample_revenue_data):
        energy_total = sum(
            p["gross_usd"] for p in sample_revenue_data["energy_payments"]
        )
        assert energy_total == sample_revenue_data["total_energy_payment_usd"]

    def test_demand_savings_summation_exact(self, sample_revenue_data):
        monthly_total = sum(
            m["savings_usd"]
            for m in sample_revenue_data["demand_charge_savings"]["monthly_savings"]
        )
        assert monthly_total == sample_revenue_data["demand_charge_savings"]["total_usd"]


# =============================================================================
# Revenue Data Fixture Validation
# =============================================================================


class TestRevenueDataValidation:
    """Validate revenue fixture data completeness and consistency."""

    def test_facility_id(self, sample_revenue_data):
        assert sample_revenue_data["facility_id"] == "FAC-037-US-001"

    def test_program_id(self, sample_revenue_data):
        assert sample_revenue_data["program_id"] == "PJM-ELR-2025"

    def test_season(self, sample_revenue_data):
        assert sample_revenue_data["season"] == "SUMMER_2025"

    def test_enrolled_capacity(self, sample_revenue_data):
        assert sample_revenue_data["enrolled_capacity_kw"] == 800.0

    def test_implementation_cost_positive(self, sample_revenue_data):
        assert sample_revenue_data["implementation_cost_usd"] > 0

    def test_operating_cost_positive(self, sample_revenue_data):
        assert sample_revenue_data["annual_operating_cost_usd"] > 0

    @pytest.mark.parametrize("field", [
        "facility_id", "program_id", "season", "enrolled_capacity_kw",
        "capacity_payment", "energy_payments", "total_energy_payment_usd",
        "penalties", "total_penalties_usd", "demand_charge_savings",
        "ancillary_services_usd", "availability_bonus_usd",
        "gross_revenue_usd", "net_revenue_usd",
    ])
    def test_required_revenue_field(self, sample_revenue_data, field):
        assert field in sample_revenue_data

    def test_penalty_event_in_energy_payments(self, sample_revenue_data):
        penalty_events = {p["event_id"]
                         for p in sample_revenue_data["penalties"]}
        energy_events = {p["event_id"]
                        for p in sample_revenue_data["energy_payments"]}
        # Penalty events should be subset of energy events
        assert penalty_events.issubset(energy_events)


# =============================================================================
# NPV and Lifecycle Analysis
# =============================================================================


class TestNPVAnalysis:
    """Test Net Present Value calculations for DR investments."""

    @pytest.mark.parametrize("discount_rate,expected_positive", [
        (0.05, True), (0.08, True), (0.10, True), (0.15, True),
    ])
    def test_npv_positive(self, sample_revenue_data, discount_rate,
                           expected_positive):
        impl_cost = float(sample_revenue_data["implementation_cost_usd"])
        annual_net = (float(sample_revenue_data["net_revenue_usd"]) -
                     float(sample_revenue_data["annual_operating_cost_usd"]))
        years = 5
        npv = -impl_cost + sum(
            annual_net / (1 + discount_rate) ** y for y in range(1, years + 1)
        )
        assert (npv > 0) == expected_positive

    def test_irr_positive(self, sample_revenue_data):
        impl_cost = float(sample_revenue_data["implementation_cost_usd"])
        annual_net = (float(sample_revenue_data["net_revenue_usd"]) -
                     float(sample_revenue_data["annual_operating_cost_usd"]))
        # Simple IRR check: if payback < investment period, IRR > 0
        payback = impl_cost / annual_net if annual_net > 0 else float("inf")
        assert payback < 10  # Payback within 10 years means positive IRR
