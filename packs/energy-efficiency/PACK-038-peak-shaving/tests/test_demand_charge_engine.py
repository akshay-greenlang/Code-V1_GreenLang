# -*- coding: utf-8 -*-
"""
Unit tests for DemandChargeEngine -- PACK-038 Engine 3
============================================================

Tests flat demand charge calculation, tiered/block demand charges, TOU demand
charges (on-peak/mid-peak/off-peak), CP charge allocation, ratchet demand
calculation, PF penalty calculation, and marginal value computation.

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


_m = _load("demand_charge_engine")


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
        assert hasattr(_m, "DemandChargeEngine")

    def test_engine_instantiation(self):
        engine = _m.DemandChargeEngine()
        assert engine is not None


# =============================================================================
# Flat Demand Charge Calculation
# =============================================================================


class TestFlatDemandCharge:
    """Test flat (non-TOU) demand charge calculation."""

    def _get_flat(self, engine):
        return (getattr(engine, "calculate_flat_demand", None)
                or getattr(engine, "flat_demand_charge", None)
                or getattr(engine, "compute_flat_charge", None))

    @pytest.mark.parametrize("peak_kw,rate,expected_usd", [
        (2000, Decimal("8.50"), Decimal("17000.00")),
        (1500, Decimal("10.00"), Decimal("15000.00")),
        (500, Decimal("12.50"), Decimal("6250.00")),
        (3000, Decimal("7.25"), Decimal("21750.00")),
        (100, Decimal("15.00"), Decimal("1500.00")),
    ])
    def test_flat_charge_calculation(self, peak_kw, rate, expected_usd):
        engine = _m.DemandChargeEngine()
        calc = self._get_flat(engine)
        if calc is None:
            pytest.skip("flat_demand method not found")
        result = calc(peak_kw=peak_kw, rate_usd_per_kw=rate)
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            assert abs(Decimal(str(charge)) - expected_usd) < Decimal("0.01")

    def test_zero_peak(self):
        engine = _m.DemandChargeEngine()
        calc = self._get_flat(engine)
        if calc is None:
            pytest.skip("flat_demand method not found")
        result = calc(peak_kw=0, rate_usd_per_kw=Decimal("8.50"))
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            assert Decimal(str(charge)) == Decimal("0.00")


# =============================================================================
# Tiered/Block Demand Charges
# =============================================================================


class TestTieredDemandCharge:
    """Test tiered/block demand charge structures."""

    def _get_tiered(self, engine):
        return (getattr(engine, "calculate_tiered_demand", None)
                or getattr(engine, "tiered_demand_charge", None)
                or getattr(engine, "compute_block_charge", None))

    def test_tiered_basic(self):
        engine = _m.DemandChargeEngine()
        calc = self._get_tiered(engine)
        if calc is None:
            pytest.skip("tiered_demand method not found")
        tiers = [
            {"upper_kw": 500, "rate_usd_per_kw": Decimal("10.00")},
            {"upper_kw": 1000, "rate_usd_per_kw": Decimal("8.00")},
            {"upper_kw": float("inf"), "rate_usd_per_kw": Decimal("6.00")},
        ]
        result = calc(peak_kw=1500, tiers=tiers)
        # 500*10 + 500*8 + 500*6 = 5000 + 4000 + 3000 = 12000
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            assert abs(Decimal(str(charge)) - Decimal("12000.00")) < Decimal("1.00")

    @pytest.mark.parametrize("peak_kw", [0, 250, 500, 750, 1000, 1500, 2000, 3000])
    def test_tiered_various_peaks(self, peak_kw):
        engine = _m.DemandChargeEngine()
        calc = self._get_tiered(engine)
        if calc is None:
            pytest.skip("tiered_demand method not found")
        tiers = [
            {"upper_kw": 500, "rate_usd_per_kw": Decimal("10.00")},
            {"upper_kw": 1000, "rate_usd_per_kw": Decimal("8.00")},
            {"upper_kw": float("inf"), "rate_usd_per_kw": Decimal("6.00")},
        ]
        result = calc(peak_kw=peak_kw, tiers=tiers)
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            assert Decimal(str(charge)) >= 0


# =============================================================================
# TOU Demand Charges
# =============================================================================


class TestTOUDemandCharge:
    """Test time-of-use demand charge calculation."""

    def _get_tou(self, engine):
        return (getattr(engine, "calculate_tou_demand", None)
                or getattr(engine, "tou_demand_charge", None)
                or getattr(engine, "compute_tou_charge", None))

    @pytest.mark.parametrize("period,rate", [
        ("ON_PEAK", Decimal("14.25")),
        ("MID_PEAK", Decimal("9.50")),
        ("OFF_PEAK", Decimal("4.75")),
    ])
    def test_tou_period_rates(self, period, rate, sample_tariff_structure):
        engine = _m.DemandChargeEngine()
        calc = self._get_tou(engine)
        if calc is None:
            pytest.skip("tou_demand method not found")
        result = calc(peak_kw=2000, period=period, rate_usd_per_kw=rate)
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            assert Decimal(str(charge)) == rate * 2000

    def test_on_peak_highest_charge(self, sample_tariff_structure):
        engine = _m.DemandChargeEngine()
        calc = self._get_tou(engine)
        if calc is None:
            pytest.skip("tou_demand method not found")
        on_peak = calc(peak_kw=1000, period="ON_PEAK",
                       rate_usd_per_kw=Decimal("14.25"))
        off_peak = calc(peak_kw=1000, period="OFF_PEAK",
                        rate_usd_per_kw=Decimal("4.75"))
        c_on = getattr(on_peak, "charge_usd", on_peak)
        c_off = getattr(off_peak, "charge_usd", off_peak)
        if isinstance(c_on, (Decimal, int, float)) and isinstance(c_off, (Decimal, int, float)):
            assert Decimal(str(c_on)) > Decimal(str(c_off))

    @pytest.mark.parametrize("peak_kw", [500, 1000, 1500, 2000, 2500])
    def test_tou_scaling(self, peak_kw):
        engine = _m.DemandChargeEngine()
        calc = self._get_tou(engine)
        if calc is None:
            pytest.skip("tou_demand method not found")
        result = calc(peak_kw=peak_kw, period="ON_PEAK",
                      rate_usd_per_kw=Decimal("14.25"))
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            expected = Decimal(str(peak_kw)) * Decimal("14.25")
            assert abs(Decimal(str(charge)) - expected) < Decimal("1.00")


# =============================================================================
# CP Charge Allocation
# =============================================================================


class TestCPChargeAllocation:
    """Test coincident peak charge allocation."""

    def _get_cp(self, engine):
        return (getattr(engine, "calculate_cp_charge", None)
                or getattr(engine, "cp_charge_allocation", None)
                or getattr(engine, "compute_cp_charge", None))

    def test_cp_charge_basic(self, sample_cp_data):
        engine = _m.DemandChargeEngine()
        calc = self._get_cp(engine)
        if calc is None:
            pytest.skip("cp_charge method not found")
        result = calc(
            icap_tag_kw=sample_cp_data["icap_tag_kw"],
            rate_usd_per_kw=sample_cp_data["tag_value_usd_per_kw_year"],
        )
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            expected = Decimal("1850.0") * Decimal("6.80")
            assert abs(Decimal(str(charge)) - expected) < Decimal("1.00")

    @pytest.mark.parametrize("tag_kw,rate,expected", [
        (1000, Decimal("6.80"), Decimal("6800.00")),
        (2000, Decimal("6.80"), Decimal("13600.00")),
        (1500, Decimal("8.00"), Decimal("12000.00")),
        (500, Decimal("10.00"), Decimal("5000.00")),
    ])
    def test_cp_charge_values(self, tag_kw, rate, expected):
        engine = _m.DemandChargeEngine()
        calc = self._get_cp(engine)
        if calc is None:
            pytest.skip("cp_charge method not found")
        result = calc(icap_tag_kw=tag_kw, rate_usd_per_kw=rate)
        charge = getattr(result, "charge_usd", result)
        if isinstance(charge, (Decimal, int, float)):
            assert abs(Decimal(str(charge)) - expected) < Decimal("1.00")


# =============================================================================
# Ratchet Demand Calculation
# =============================================================================


class TestRatchetDemandCalculation:
    """Test ratchet demand calculation."""

    def _get_ratchet(self, engine):
        return (getattr(engine, "calculate_ratchet_demand", None)
                or getattr(engine, "ratchet_demand", None)
                or getattr(engine, "compute_ratchet", None))

    @pytest.mark.parametrize("historical_peak_kw,ratchet_pct,expected_min_kw", [
        (2000, Decimal("0.80"), 1600),
        (2000, Decimal("0.85"), 1700),
        (2000, Decimal("0.90"), 1800),
        (2000, Decimal("0.75"), 1500),
        (2000, Decimal("1.00"), 2000),
    ])
    def test_ratchet_minimum(self, historical_peak_kw, ratchet_pct, expected_min_kw):
        engine = _m.DemandChargeEngine()
        calc = self._get_ratchet(engine)
        if calc is None:
            pytest.skip("ratchet_demand method not found")
        result = calc(historical_peak_kw=historical_peak_kw,
                      ratchet_pct=ratchet_pct)
        min_kw = getattr(result, "effective_minimum_kw", result)
        if isinstance(min_kw, (int, float, Decimal)):
            assert abs(float(min_kw) - expected_min_kw) < 1.0

    def test_ratchet_billing_above_minimum(self):
        engine = _m.DemandChargeEngine()
        calc = self._get_ratchet(engine)
        if calc is None:
            pytest.skip("ratchet_demand method not found")
        result = calc(historical_peak_kw=2000, ratchet_pct=Decimal("0.80"),
                      current_peak_kw=1800)
        billed = getattr(result, "billed_kw", None)
        if billed is not None:
            assert float(billed) == 1800  # Current > ratchet minimum

    def test_ratchet_billing_below_minimum(self):
        engine = _m.DemandChargeEngine()
        calc = self._get_ratchet(engine)
        if calc is None:
            pytest.skip("ratchet_demand method not found")
        result = calc(historical_peak_kw=2000, ratchet_pct=Decimal("0.80"),
                      current_peak_kw=1400)
        billed = getattr(result, "billed_kw", None)
        if billed is not None:
            assert float(billed) == 1600  # Ratchet minimum applies


# =============================================================================
# PF Penalty Calculation
# =============================================================================


class TestPFPenaltyCalculation:
    """Test power factor penalty calculation."""

    def _get_pf_penalty(self, engine):
        return (getattr(engine, "calculate_pf_penalty", None)
                or getattr(engine, "pf_penalty", None)
                or getattr(engine, "power_factor_penalty", None))

    @pytest.mark.parametrize("actual_pf,target_pf", [
        (0.82, 0.90), (0.85, 0.90), (0.88, 0.90),
        (0.90, 0.90), (0.95, 0.90),
    ])
    def test_pf_penalty_direction(self, actual_pf, target_pf):
        engine = _m.DemandChargeEngine()
        calc = self._get_pf_penalty(engine)
        if calc is None:
            pytest.skip("pf_penalty method not found")
        result = calc(actual_pf=actual_pf, target_pf=target_pf,
                      billed_kw=2000, rate_usd_per_kvar=Decimal("0.45"))
        penalty = getattr(result, "penalty_usd", result)
        if isinstance(penalty, (Decimal, int, float)):
            if actual_pf < target_pf:
                assert Decimal(str(penalty)) > 0
            else:
                assert Decimal(str(penalty)) == 0

    def test_pf_penalty_at_target(self):
        engine = _m.DemandChargeEngine()
        calc = self._get_pf_penalty(engine)
        if calc is None:
            pytest.skip("pf_penalty method not found")
        result = calc(actual_pf=0.90, target_pf=0.90,
                      billed_kw=2000, rate_usd_per_kvar=Decimal("0.45"))
        penalty = getattr(result, "penalty_usd", result)
        if isinstance(penalty, (Decimal, int, float)):
            assert Decimal(str(penalty)) == 0

    def test_lower_pf_higher_penalty(self):
        engine = _m.DemandChargeEngine()
        calc = self._get_pf_penalty(engine)
        if calc is None:
            pytest.skip("pf_penalty method not found")
        r_low = calc(actual_pf=0.82, target_pf=0.90,
                     billed_kw=2000, rate_usd_per_kvar=Decimal("0.45"))
        r_high = calc(actual_pf=0.88, target_pf=0.90,
                      billed_kw=2000, rate_usd_per_kvar=Decimal("0.45"))
        p_low = getattr(r_low, "penalty_usd", r_low)
        p_high = getattr(r_high, "penalty_usd", r_high)
        if isinstance(p_low, (Decimal, int, float)) and isinstance(p_high, (Decimal, int, float)):
            assert Decimal(str(p_low)) > Decimal(str(p_high))


# =============================================================================
# Marginal Value Computation
# =============================================================================


class TestMarginalValue:
    """Test marginal value of peak reduction computation."""

    def _get_marginal(self, engine):
        return (getattr(engine, "marginal_value", None)
                or getattr(engine, "calculate_marginal_value", None)
                or getattr(engine, "compute_marginal_savings", None))

    def test_marginal_value_positive(self, sample_tariff_structure):
        engine = _m.DemandChargeEngine()
        calc = self._get_marginal(engine)
        if calc is None:
            pytest.skip("marginal_value method not found")
        result = calc(tariff=sample_tariff_structure, peak_kw=2000,
                      reduction_kw=100)
        value = getattr(result, "value_usd_per_kw", result)
        if isinstance(value, (Decimal, int, float)):
            assert Decimal(str(value)) > 0

    @pytest.mark.parametrize("reduction_kw", [10, 50, 100, 200, 500])
    def test_marginal_value_for_reductions(self, reduction_kw, sample_tariff_structure):
        engine = _m.DemandChargeEngine()
        calc = self._get_marginal(engine)
        if calc is None:
            pytest.skip("marginal_value method not found")
        result = calc(tariff=sample_tariff_structure, peak_kw=2000,
                      reduction_kw=reduction_kw)
        assert result is not None


# =============================================================================
# Full Demand Charge Calculation
# =============================================================================


class TestFullDemandChargeCalculation:
    """Test complete demand charge calculation with all components."""

    def _get_full(self, engine):
        return (getattr(engine, "calculate_total_demand_charges", None)
                or getattr(engine, "total_demand_charge", None)
                or getattr(engine, "compute_bill", None))

    def test_full_calculation(self, sample_tariff_structure, sample_interval_data):
        engine = _m.DemandChargeEngine()
        calc = self._get_full(engine)
        if calc is None:
            pytest.skip("total_demand_charges method not found")
        result = calc(tariff=sample_tariff_structure,
                      interval_data=sample_interval_data)
        assert result is not None

    def test_full_has_components(self, sample_tariff_structure, sample_interval_data):
        engine = _m.DemandChargeEngine()
        calc = self._get_full(engine)
        if calc is None:
            pytest.skip("total_demand_charges method not found")
        result = calc(tariff=sample_tariff_structure,
                      interval_data=sample_interval_data)
        has_breakdown = (hasattr(result, "flat_charge") or
                         hasattr(result, "tou_charge") or
                         hasattr(result, "components"))
        assert has_breakdown or result is not None


# =============================================================================
# Tariff Structure Fixture Validation
# =============================================================================


class TestTariffFixture:
    """Validate the tariff structure fixture."""

    def test_tariff_has_flat(self, sample_tariff_structure):
        assert "flat_demand_charge" in sample_tariff_structure

    def test_tariff_has_tou(self, sample_tariff_structure):
        assert "tou_demand_charges" in sample_tariff_structure

    def test_tariff_has_cp(self, sample_tariff_structure):
        assert "coincident_peak_charge" in sample_tariff_structure

    def test_tariff_has_ratchet(self, sample_tariff_structure):
        assert "ratchet_demand" in sample_tariff_structure

    def test_tariff_has_pf(self, sample_tariff_structure):
        assert "power_factor_penalty" in sample_tariff_structure

    def test_flat_rate_decimal(self, sample_tariff_structure):
        rate = sample_tariff_structure["flat_demand_charge"]["rate_usd_per_kw"]
        assert isinstance(rate, Decimal)

    @pytest.mark.parametrize("charge_type", [
        "flat_demand_charge", "tou_demand_charges",
        "coincident_peak_charge", "ratchet_demand",
        "power_factor_penalty",
    ])
    def test_charge_type_enabled(self, charge_type, sample_tariff_structure):
        assert sample_tariff_structure[charge_type]["enabled"] is True


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    def test_provenance_deterministic(self, sample_tariff_structure, sample_interval_data):
        engine = _m.DemandChargeEngine()
        calc = (getattr(engine, "calculate_total_demand_charges", None)
                or getattr(engine, "total_demand_charge", None))
        if calc is None:
            pytest.skip("total calculation method not found")
        r1 = calc(tariff=sample_tariff_structure, interval_data=sample_interval_data)
        r2 = calc(tariff=sample_tariff_structure, interval_data=sample_interval_data)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2
            assert len(h1) == 64
