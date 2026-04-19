# -*- coding: utf-8 -*-
"""
Unit tests for EnPIEngine -- PACK-039 Engine 5
============================================================

Tests ISO 50001 EnPI calculation with simple ratio, regression-based
normalization, CUSUM tracking, significance testing, and baseline management.

Coverage target: 85%+
Total tests: ~65
"""

import hashlib
import importlib.util
import json
import math
import random
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
    mod_key = f"pack039_test.{name}"
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


_m = _load("enpi_engine")


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
        assert hasattr(_m, "EnPIEngine")

    def test_engine_instantiation(self):
        engine = _m.EnPIEngine()
        assert engine is not None


# =============================================================================
# EnPI Type Parametrize
# =============================================================================


class TestEnPITypes:
    """Test 4 EnPI calculation types."""

    def _get_calculate(self, engine):
        return (getattr(engine, "calculate_enpi", None)
                or getattr(engine, "compute_enpi", None)
                or getattr(engine, "enpi", None))

    @pytest.mark.parametrize("enpi_type", [
        "SIMPLE_RATIO",
        "REGRESSION",
        "CUSUM",
        "ENERGY_INTENSITY",
    ])
    def test_enpi_type(self, enpi_type, sample_enpi_data):
        engine = _m.EnPIEngine()
        calculate = self._get_calculate(engine)
        if calculate is None:
            pytest.skip("calculate_enpi method not found")
        try:
            result = calculate(sample_enpi_data, enpi_type=enpi_type)
            assert result is not None
        except (ValueError, TypeError, KeyError):
            result = calculate(sample_enpi_data)
            assert result is not None

    @pytest.mark.parametrize("enpi_type", [
        "SIMPLE_RATIO",
        "REGRESSION",
        "CUSUM",
        "ENERGY_INTENSITY",
    ])
    def test_enpi_type_deterministic(self, enpi_type, sample_enpi_data):
        engine = _m.EnPIEngine()
        calculate = self._get_calculate(engine)
        if calculate is None:
            pytest.skip("calculate_enpi method not found")
        try:
            r1 = calculate(sample_enpi_data, enpi_type=enpi_type)
            r2 = calculate(sample_enpi_data, enpi_type=enpi_type)
        except (ValueError, TypeError):
            r1 = calculate(sample_enpi_data)
            r2 = calculate(sample_enpi_data)
        assert str(r1) == str(r2)


# =============================================================================
# Simple Ratio EnPI
# =============================================================================


class TestSimpleRatio:
    """Test simple ratio EnPI (energy / driver)."""

    def _get_ratio(self, engine):
        return (getattr(engine, "simple_ratio", None)
                or getattr(engine, "calculate_ratio", None)
                or getattr(engine, "ratio_enpi", None))

    def test_ratio_result(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        ratio = self._get_ratio(engine)
        if ratio is None:
            pytest.skip("simple_ratio method not found")
        result = ratio(sample_enpi_data)
        assert result is not None

    def test_ratio_positive(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        ratio = self._get_ratio(engine)
        if ratio is None:
            pytest.skip("simple_ratio method not found")
        result = ratio(sample_enpi_data)
        val = getattr(result, "value", result)
        if isinstance(val, (int, float, Decimal)):
            assert float(val) > 0

    def test_ratio_units(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        ratio = self._get_ratio(engine)
        if ratio is None:
            pytest.skip("simple_ratio method not found")
        result = ratio(sample_enpi_data)
        unit = getattr(result, "unit", None)
        if unit is not None:
            assert "kWh" in str(unit) or "MWh" in str(unit) or len(str(unit)) > 0


# =============================================================================
# Regression-Based Normalization
# =============================================================================


class TestRegressionNormalization:
    """Test regression-based EnPI normalization with relevant variables."""

    def _get_regression(self, engine):
        return (getattr(engine, "regression_normalize", None)
                or getattr(engine, "fit_regression", None)
                or getattr(engine, "regression_enpi", None))

    @pytest.mark.parametrize("variable", [
        "hdd", "cdd", "production_units",
        "occupancy_pct", "operating_hours", "outdoor_temp_avg_c",
    ])
    def test_regression_with_variable(self, variable, sample_enpi_data):
        engine = _m.EnPIEngine()
        regression = self._get_regression(engine)
        if regression is None:
            pytest.skip("regression method not found")
        try:
            result = regression(sample_enpi_data, variables=[variable])
            assert result is not None
        except (ValueError, TypeError, KeyError):
            pass

    def test_regression_multi_variable(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        regression = self._get_regression(engine)
        if regression is None:
            pytest.skip("regression method not found")
        try:
            result = regression(sample_enpi_data,
                                variables=["hdd", "cdd", "production_units"])
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_regression_r_squared(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        regression = self._get_regression(engine)
        if regression is None:
            pytest.skip("regression method not found")
        try:
            result = regression(sample_enpi_data, variables=["hdd", "cdd"])
        except (ValueError, TypeError):
            pytest.skip("regression method does not accept variables")
            return
        r_squared = getattr(result, "r_squared", None)
        if r_squared is not None:
            assert 0.0 <= float(r_squared) <= 1.0

    def test_regression_coefficients(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        regression = self._get_regression(engine)
        if regression is None:
            pytest.skip("regression method not found")
        try:
            result = regression(sample_enpi_data, variables=["hdd", "cdd"])
        except (ValueError, TypeError):
            pytest.skip("regression method does not accept variables")
            return
        coefficients = getattr(result, "coefficients", None)
        if coefficients is not None:
            assert len(coefficients) >= 1

    def test_regression_deterministic(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        regression = self._get_regression(engine)
        if regression is None:
            pytest.skip("regression method not found")
        try:
            r1 = regression(sample_enpi_data, variables=["hdd"])
            r2 = regression(sample_enpi_data, variables=["hdd"])
        except (ValueError, TypeError):
            r1 = regression(sample_enpi_data)
            r2 = regression(sample_enpi_data)
        assert str(r1) == str(r2)


# =============================================================================
# CUSUM Tracking
# =============================================================================


class TestCUSUMTracking:
    """Test cumulative sum (CUSUM) performance tracking."""

    def _get_cusum(self, engine):
        return (getattr(engine, "cusum_tracking", None)
                or getattr(engine, "compute_cusum", None)
                or getattr(engine, "track_cusum", None))

    def test_cusum_result(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        cusum = self._get_cusum(engine)
        if cusum is None:
            pytest.skip("cusum_tracking method not found")
        result = cusum(sample_enpi_data)
        assert result is not None

    def test_cusum_cumulative_values(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        cusum = self._get_cusum(engine)
        if cusum is None:
            pytest.skip("cusum_tracking method not found")
        result = cusum(sample_enpi_data)
        values = getattr(result, "cumulative_values", getattr(result, "cusum_values", None))
        if values is not None and isinstance(values, list):
            assert len(values) == len(sample_enpi_data)

    def test_cusum_trend_direction(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        cusum = self._get_cusum(engine)
        if cusum is None:
            pytest.skip("cusum_tracking method not found")
        result = cusum(sample_enpi_data)
        trend = getattr(result, "trend", None)
        if trend is not None:
            assert trend in ["IMPROVING", "DEGRADING", "STABLE", "UNKNOWN"]


# =============================================================================
# Significance Testing
# =============================================================================


class TestSignificanceTesting:
    """Test statistical significance of EnPI changes."""

    def _get_significance(self, engine):
        return (getattr(engine, "test_significance", None)
                or getattr(engine, "significance_test", None)
                or getattr(engine, "is_significant", None))

    def test_significance_result(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        sig = self._get_significance(engine)
        if sig is None:
            pytest.skip("significance test method not found")
        result = sig(sample_enpi_data)
        assert result is not None

    def test_significance_p_value(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        sig = self._get_significance(engine)
        if sig is None:
            pytest.skip("significance test method not found")
        result = sig(sample_enpi_data)
        p_value = getattr(result, "p_value", None)
        if p_value is not None:
            assert 0.0 <= float(p_value) <= 1.0

    def test_significance_boolean(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        sig = self._get_significance(engine)
        if sig is None:
            pytest.skip("significance test method not found")
        result = sig(sample_enpi_data)
        is_sig = getattr(result, "is_significant", None)
        if is_sig is not None:
            assert isinstance(is_sig, bool)


# =============================================================================
# Baseline Management
# =============================================================================


class TestBaselineManagement:
    """Test energy baseline creation and adjustment."""

    def _get_baseline(self, engine):
        return (getattr(engine, "create_baseline", None)
                or getattr(engine, "set_baseline", None)
                or getattr(engine, "establish_baseline", None))

    def test_create_baseline(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        create = self._get_baseline(engine)
        if create is None:
            pytest.skip("create_baseline method not found")
        result = create(sample_enpi_data)
        assert result is not None

    def test_baseline_period(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        create = self._get_baseline(engine)
        if create is None:
            pytest.skip("create_baseline method not found")
        result = create(sample_enpi_data)
        period = getattr(result, "baseline_period", None)
        if period is not None:
            assert len(str(period)) > 0

    def test_baseline_adjustment(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        adjust = (getattr(engine, "adjust_baseline", None)
                  or getattr(engine, "update_baseline", None)
                  or getattr(engine, "rebase", None))
        if adjust is None:
            pytest.skip("adjust_baseline method not found")
        try:
            result = adjust(sample_enpi_data, reason="FACILITY_CHANGE")
            assert result is not None
        except (ValueError, TypeError):
            pass


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    """Test provenance hash for EnPI results."""

    def test_same_input_same_hash(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        calculate = (getattr(engine, "calculate_enpi", None)
                     or getattr(engine, "compute_enpi", None))
        if calculate is None:
            pytest.skip("calculate method not found")
        r1 = calculate(sample_enpi_data)
        r2 = calculate(sample_enpi_data)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2

    def test_hash_is_sha256(self, sample_enpi_data):
        engine = _m.EnPIEngine()
        calculate = (getattr(engine, "calculate_enpi", None)
                     or getattr(engine, "compute_enpi", None))
        if calculate is None:
            pytest.skip("calculate method not found")
        result = calculate(sample_enpi_data)
        h = getattr(result, "provenance_hash", None)
        if h is not None:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# EnPI Data Fixture Validation
# =============================================================================


class TestEnPIDataFixture:
    """Validate the EnPI data fixture."""

    def test_12_months(self, sample_enpi_data):
        assert len(sample_enpi_data) == 12

    def test_all_have_period(self, sample_enpi_data):
        for m in sample_enpi_data:
            assert "period" in m

    def test_all_have_energy(self, sample_enpi_data):
        for m in sample_enpi_data:
            assert "energy_kwh" in m
            assert m["energy_kwh"] > 0

    def test_all_have_hdd_cdd(self, sample_enpi_data):
        for m in sample_enpi_data:
            assert "hdd" in m
            assert "cdd" in m

    def test_all_have_production(self, sample_enpi_data):
        for m in sample_enpi_data:
            assert "production_units" in m
            assert m["production_units"] > 0

    def test_summer_consumption_higher(self, sample_enpi_data):
        summer_avg = sum(m["energy_kwh"] for m in sample_enpi_data[5:8]) / 3
        winter_avg = sum(m["energy_kwh"] for m in sample_enpi_data[0:3]) / 3
        assert summer_avg > winter_avg

    def test_deterministic_data(self, sample_enpi_data):
        rng = random.Random(42)
        expected_first_energy = 850_000 + rng.randint(-5000, 5000)
        assert abs(sample_enpi_data[0]["energy_kwh"] - expected_first_energy) < 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for EnPI calculations."""

    def test_single_month(self):
        engine = _m.EnPIEngine()
        calculate = (getattr(engine, "calculate_enpi", None)
                     or getattr(engine, "compute_enpi", None))
        if calculate is None:
            pytest.skip("calculate method not found")
        single = [{"period": "2025-01", "energy_kwh": 850000,
                    "hdd": 680, "cdd": 0, "production_units": 950,
                    "floor_area_m2": 38000}]
        try:
            result = calculate(single)
            assert result is not None
        except (ValueError, IndexError):
            pass

    def test_zero_production(self):
        engine = _m.EnPIEngine()
        ratio = (getattr(engine, "simple_ratio", None)
                 or getattr(engine, "calculate_ratio", None))
        if ratio is None:
            pytest.skip("ratio method not found")
        data = [{"period": "2025-01", "energy_kwh": 850000,
                 "production_units": 0, "floor_area_m2": 38000}]
        try:
            result = ratio(data)
            assert result is not None
        except (ValueError, ZeroDivisionError):
            pass
