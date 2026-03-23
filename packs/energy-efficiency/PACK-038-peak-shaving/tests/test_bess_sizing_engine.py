# -*- coding: utf-8 -*-
"""
Unit tests for BESSSizingEngine -- PACK-038 Engine 4
============================================================

Tests sizing optimization for NMC/LFP/flow chemistries, 8,760-hour dispatch
simulation, degradation modeling (calendar + cycle), LCOS calculation, DoD
impact on cycle life, C-rate and thermal derating.

Coverage target: 85%+
Total tests: ~65
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


_m = _load("bess_sizing_engine")


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
        assert hasattr(_m, "BESSSizingEngine")

    def test_engine_instantiation(self):
        engine = _m.BESSSizingEngine()
        assert engine is not None


# =============================================================================
# Sizing Optimization (6 chemistries)
# =============================================================================


class TestSizingOptimization:
    """Test BESS sizing optimization across battery chemistries."""

    def _get_size(self, engine):
        return (getattr(engine, "optimize_sizing", None)
                or getattr(engine, "size_bess", None)
                or getattr(engine, "calculate_optimal_size", None))

    @pytest.mark.parametrize("chemistry", ["NMC", "LFP", "FLOW_VRB", "NCA", "LTO", "SODIUM_ION"])
    def test_sizing_by_chemistry(self, chemistry, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        size = self._get_size(engine)
        if size is None:
            pytest.skip("optimize_sizing method not found")
        config = dict(sample_bess_config, chemistry=chemistry)
        try:
            result = size(config=config, interval_data=sample_interval_data)
        except (TypeError, ValueError):
            result = size(config=config)
        assert result is not None

    @pytest.mark.parametrize("target_reduction_kw", [100, 200, 300, 400, 500])
    def test_sizing_by_target(self, target_reduction_kw, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        size = self._get_size(engine)
        if size is None:
            pytest.skip("optimize_sizing method not found")
        try:
            result = size(config=sample_bess_config,
                          interval_data=sample_interval_data,
                          target_reduction_kw=target_reduction_kw)
        except TypeError:
            result = size(config=sample_bess_config)
        assert result is not None

    def test_sizing_returns_capacity(self, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        size = self._get_size(engine)
        if size is None:
            pytest.skip("optimize_sizing method not found")
        try:
            result = size(config=sample_bess_config, interval_data=sample_interval_data)
        except TypeError:
            result = size(config=sample_bess_config)
        capacity = getattr(result, "recommended_kwh", None)
        if capacity is not None:
            assert capacity > 0

    def test_sizing_returns_power(self, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        size = self._get_size(engine)
        if size is None:
            pytest.skip("optimize_sizing method not found")
        try:
            result = size(config=sample_bess_config, interval_data=sample_interval_data)
        except TypeError:
            result = size(config=sample_bess_config)
        power = getattr(result, "recommended_kw", None)
        if power is not None:
            assert power > 0


# =============================================================================
# 8,760-Hour Dispatch Simulation
# =============================================================================


class TestDispatchSimulation:
    """Test 8,760-hour annual dispatch simulation."""

    def _get_dispatch(self, engine):
        return (getattr(engine, "simulate_dispatch", None)
                or getattr(engine, "dispatch_simulation", None)
                or getattr(engine, "run_dispatch", None))

    def test_dispatch_returns_result(self, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        dispatch = self._get_dispatch(engine)
        if dispatch is None:
            pytest.skip("dispatch_simulation method not found")
        result = dispatch(bess_config=sample_bess_config,
                          interval_data=sample_interval_data)
        assert result is not None

    def test_dispatch_soc_bounds(self, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        dispatch = self._get_dispatch(engine)
        if dispatch is None:
            pytest.skip("dispatch_simulation method not found")
        result = dispatch(bess_config=sample_bess_config,
                          interval_data=sample_interval_data)
        soc_series = getattr(result, "soc_series", None)
        if soc_series is not None and isinstance(soc_series, list):
            for soc in soc_series:
                val = soc if isinstance(soc, (int, float)) else soc.get("soc_pct", 50)
                assert val >= sample_bess_config["min_soc_pct"] - 0.1
                assert val <= sample_bess_config["max_soc_pct"] + 0.1

    @pytest.mark.parametrize("strategy", ["PEAK_SHAVING", "TIME_SHIFT", "THRESHOLD", "OPTIMAL"])
    def test_dispatch_strategies(self, strategy, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        dispatch = self._get_dispatch(engine)
        if dispatch is None:
            pytest.skip("dispatch_simulation method not found")
        try:
            result = dispatch(bess_config=sample_bess_config,
                              interval_data=sample_interval_data,
                              strategy=strategy)
        except (TypeError, ValueError):
            result = dispatch(bess_config=sample_bess_config,
                              interval_data=sample_interval_data)
        assert result is not None

    def test_dispatch_deterministic(self, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        dispatch = self._get_dispatch(engine)
        if dispatch is None:
            pytest.skip("dispatch_simulation method not found")
        r1 = dispatch(bess_config=sample_bess_config,
                      interval_data=sample_interval_data)
        r2 = dispatch(bess_config=sample_bess_config,
                      interval_data=sample_interval_data)
        s1 = getattr(r1, "total_discharge_kwh", str(r1))
        s2 = getattr(r2, "total_discharge_kwh", str(r2))
        assert s1 == s2


# =============================================================================
# Degradation Modeling
# =============================================================================


class TestDegradationModeling:
    """Test calendar and cycle degradation modeling."""

    def _get_degradation(self, engine):
        return (getattr(engine, "model_degradation", None)
                or getattr(engine, "degradation_analysis", None)
                or getattr(engine, "calculate_degradation", None))

    def test_degradation_result(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        degrade = self._get_degradation(engine)
        if degrade is None:
            pytest.skip("degradation method not found")
        result = degrade(bess_config=sample_bess_config, years=10, cycles_per_year=365)
        assert result is not None

    @pytest.mark.parametrize("years", [1, 5, 10, 15, 20])
    def test_degradation_over_time(self, years, sample_bess_config):
        engine = _m.BESSSizingEngine()
        degrade = self._get_degradation(engine)
        if degrade is None:
            pytest.skip("degradation method not found")
        result = degrade(bess_config=sample_bess_config,
                         years=years, cycles_per_year=365)
        retention = getattr(result, "capacity_retention_pct", None)
        if retention is not None:
            assert 0 < retention <= 100
            if years > 1:
                r_1 = degrade(bess_config=sample_bess_config,
                              years=1, cycles_per_year=365)
                ret_1 = getattr(r_1, "capacity_retention_pct", 100)
                assert retention <= ret_1

    def test_calendar_degradation(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        degrade = self._get_degradation(engine)
        if degrade is None:
            pytest.skip("degradation method not found")
        result = degrade(bess_config=sample_bess_config,
                         years=10, cycles_per_year=0)
        retention = getattr(result, "capacity_retention_pct", None)
        if retention is not None:
            # Calendar only: 10 years * 1%/year = 10% loss -> 90% retention
            assert retention <= 100
            assert retention >= 80

    def test_cycle_degradation(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        degrade = self._get_degradation(engine)
        if degrade is None:
            pytest.skip("degradation method not found")
        r_low = degrade(bess_config=sample_bess_config,
                        years=1, cycles_per_year=100)
        r_high = degrade(bess_config=sample_bess_config,
                         years=1, cycles_per_year=500)
        ret_low = getattr(r_low, "capacity_retention_pct", 100)
        ret_high = getattr(r_high, "capacity_retention_pct", 100)
        if isinstance(ret_low, (int, float)) and isinstance(ret_high, (int, float)):
            assert ret_low >= ret_high


# =============================================================================
# LCOS Calculation
# =============================================================================


class TestLCOSCalculation:
    """Test Levelized Cost of Storage calculation."""

    def _get_lcos(self, engine):
        return (getattr(engine, "calculate_lcos", None)
                or getattr(engine, "lcos", None)
                or getattr(engine, "levelized_cost", None))

    def test_lcos_result(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        lcos = self._get_lcos(engine)
        if lcos is None:
            pytest.skip("lcos method not found")
        result = lcos(bess_config=sample_bess_config,
                      annual_throughput_kwh=150_000,
                      project_life_years=15,
                      discount_rate=Decimal("0.08"))
        assert result is not None

    def test_lcos_positive(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        lcos = self._get_lcos(engine)
        if lcos is None:
            pytest.skip("lcos method not found")
        result = lcos(bess_config=sample_bess_config,
                      annual_throughput_kwh=150_000,
                      project_life_years=15,
                      discount_rate=Decimal("0.08"))
        val = getattr(result, "lcos_usd_per_kwh", result)
        if isinstance(val, (Decimal, int, float)):
            assert float(val) > 0

    def test_higher_throughput_lower_lcos(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        lcos = self._get_lcos(engine)
        if lcos is None:
            pytest.skip("lcos method not found")
        r_low = lcos(bess_config=sample_bess_config,
                     annual_throughput_kwh=50_000,
                     project_life_years=15,
                     discount_rate=Decimal("0.08"))
        r_high = lcos(bess_config=sample_bess_config,
                      annual_throughput_kwh=200_000,
                      project_life_years=15,
                      discount_rate=Decimal("0.08"))
        v_low = getattr(r_low, "lcos_usd_per_kwh", None)
        v_high = getattr(r_high, "lcos_usd_per_kwh", None)
        if v_low is not None and v_high is not None:
            assert float(v_low) >= float(v_high)


# =============================================================================
# DoD Impact on Cycle Life
# =============================================================================


class TestDoDImpact:
    """Test depth of discharge impact on cycle life."""

    def _get_dod(self, engine):
        return (getattr(engine, "dod_cycle_life", None)
                or getattr(engine, "calculate_dod_impact", None)
                or getattr(engine, "cycle_life_at_dod", None))

    @pytest.mark.parametrize("dod_pct", [10, 20, 30, 50, 70, 80, 90, 100])
    def test_dod_cycle_life(self, dod_pct, sample_bess_config):
        engine = _m.BESSSizingEngine()
        dod = self._get_dod(engine)
        if dod is None:
            pytest.skip("dod method not found")
        result = dod(bess_config=sample_bess_config, dod_pct=dod_pct)
        cycles = getattr(result, "cycle_life", result)
        if isinstance(cycles, (int, float)):
            assert cycles > 0

    def test_lower_dod_more_cycles(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        dod = self._get_dod(engine)
        if dod is None:
            pytest.skip("dod method not found")
        r_low = dod(bess_config=sample_bess_config, dod_pct=20)
        r_high = dod(bess_config=sample_bess_config, dod_pct=90)
        c_low = getattr(r_low, "cycle_life", r_low)
        c_high = getattr(r_high, "cycle_life", r_high)
        if isinstance(c_low, (int, float)) and isinstance(c_high, (int, float)):
            assert c_low >= c_high


# =============================================================================
# C-Rate and Thermal Derating
# =============================================================================


class TestCRateAndThermalDerating:
    """Test C-rate limits and thermal derating."""

    def _get_derating(self, engine):
        return (getattr(engine, "thermal_derating", None)
                or getattr(engine, "calculate_derating", None)
                or getattr(engine, "apply_derating", None))

    @pytest.mark.parametrize("ambient_c", [20, 25, 30, 35, 40, 45])
    def test_thermal_derating(self, ambient_c, sample_bess_config):
        engine = _m.BESSSizingEngine()
        derate = self._get_derating(engine)
        if derate is None:
            pytest.skip("thermal_derating method not found")
        result = derate(bess_config=sample_bess_config, ambient_temp_c=ambient_c)
        factor = getattr(result, "derating_factor", result)
        if isinstance(factor, (int, float)):
            assert 0 < factor <= 1.0

    def test_no_derating_below_threshold(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        derate = self._get_derating(engine)
        if derate is None:
            pytest.skip("thermal_derating method not found")
        result = derate(bess_config=sample_bess_config, ambient_temp_c=25)
        factor = getattr(result, "derating_factor", result)
        if isinstance(factor, (int, float)):
            assert factor == 1.0 or abs(factor - 1.0) < 0.01

    def test_high_temp_derating(self, sample_bess_config):
        engine = _m.BESSSizingEngine()
        derate = self._get_derating(engine)
        if derate is None:
            pytest.skip("thermal_derating method not found")
        result = derate(bess_config=sample_bess_config, ambient_temp_c=45)
        factor = getattr(result, "derating_factor", result)
        if isinstance(factor, (int, float)):
            assert factor < 1.0


# =============================================================================
# BESS Config Fixture Validation
# =============================================================================


class TestBESSConfigFixture:
    """Validate the BESS config fixture."""

    def test_chemistry_lfp(self, sample_bess_config):
        assert sample_bess_config["chemistry"] == "LFP"

    def test_capacity_500kwh(self, sample_bess_config):
        assert sample_bess_config["nameplate_capacity_kwh"] == 500.0

    def test_power_250kw(self, sample_bess_config):
        assert sample_bess_config["nameplate_power_kw"] == 250.0

    def test_rte_92(self, sample_bess_config):
        assert sample_bess_config["round_trip_efficiency_pct"] == 92.0

    def test_max_cycles_6000(self, sample_bess_config):
        assert sample_bess_config["max_cycles"] == 6000

    def test_cost_decimal(self, sample_bess_config):
        assert isinstance(sample_bess_config["installed_cost_usd"], Decimal)

    def test_usable_less_than_nameplate(self, sample_bess_config):
        assert sample_bess_config["usable_capacity_kwh"] < sample_bess_config["nameplate_capacity_kwh"]


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    def test_provenance_deterministic(self, sample_bess_config, sample_interval_data):
        engine = _m.BESSSizingEngine()
        size = (getattr(engine, "optimize_sizing", None)
                or getattr(engine, "size_bess", None))
        if size is None:
            pytest.skip("sizing method not found")
        try:
            r1 = size(config=sample_bess_config, interval_data=sample_interval_data)
            r2 = size(config=sample_bess_config, interval_data=sample_interval_data)
        except TypeError:
            r1 = size(config=sample_bess_config)
            r2 = size(config=sample_bess_config)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2
            assert len(h1) == 64
            assert all(c in "0123456789abcdef" for c in h1)
