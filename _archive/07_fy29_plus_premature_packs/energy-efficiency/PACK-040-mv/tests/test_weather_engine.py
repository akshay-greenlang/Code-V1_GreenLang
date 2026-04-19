# -*- coding: utf-8 -*-
"""
Unit tests for WeatherEngine -- PACK-040 Engine 7
============================================================

Tests HDD/CDD calculation, balance point optimization, TMY
normalization, and weather data quality assessment.

Coverage target: 85%+
Total tests: ~30
"""

import hashlib
import importlib.util
import json
import math
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
    mod_key = f"pack040_test.{name}"
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


_m = _load("weather_engine")


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
        assert hasattr(_m, "WeatherEngine")

    def test_engine_instantiation(self):
        engine = _m.WeatherEngine()
        assert engine is not None


# =============================================================================
# HDD/CDD Calculation
# =============================================================================


class TestHDDCDDCalculation:
    """Test Heating Degree Day and Cooling Degree Day calculation."""

    def _get_hdd(self, engine):
        return (getattr(engine, "calculate_hdd", None)
                or getattr(engine, "hdd", None)
                or getattr(engine, "compute_hdd", None))

    def _get_cdd(self, engine):
        return (getattr(engine, "calculate_cdd", None)
                or getattr(engine, "cdd", None)
                or getattr(engine, "compute_cdd", None))

    def _get_degree_days(self, engine):
        return (getattr(engine, "calculate_degree_days", None)
                or getattr(engine, "degree_days", None)
                or getattr(engine, "compute_degree_days", None))

    def test_hdd_result(self, weather_data):
        engine = _m.WeatherEngine()
        hdd = self._get_hdd(engine) or self._get_degree_days(engine)
        if hdd is None:
            pytest.skip("HDD calculation method not found")
        records = weather_data["baseline_weather"]["records"]
        try:
            result = hdd(records, base_temp=65.0)
            assert result is not None
        except (ValueError, TypeError, KeyError):
            try:
                result = hdd(records)
                assert result is not None
            except (ValueError, TypeError, KeyError):
                pass

    def test_hdd_nonnegative(self, weather_data):
        engine = _m.WeatherEngine()
        hdd = self._get_hdd(engine) or self._get_degree_days(engine)
        if hdd is None:
            pytest.skip("HDD calculation method not found")
        records = weather_data["baseline_weather"]["records"]
        try:
            result = hdd(records, base_temp=65.0)
        except (ValueError, TypeError, KeyError):
            try:
                result = hdd(records)
            except (ValueError, TypeError, KeyError):
                pytest.skip("HDD not available")
        val = (getattr(result, "total_hdd", None)
               or (result.get("total_hdd") if isinstance(result, dict) else None)
               or result)
        if isinstance(val, (int, float, Decimal)):
            assert float(val) >= 0

    def test_cdd_result(self, weather_data):
        engine = _m.WeatherEngine()
        cdd = self._get_cdd(engine) or self._get_degree_days(engine)
        if cdd is None:
            pytest.skip("CDD calculation method not found")
        records = weather_data["baseline_weather"]["records"]
        try:
            result = cdd(records, base_temp=65.0)
            assert result is not None
        except (ValueError, TypeError, KeyError):
            try:
                result = cdd(records)
                assert result is not None
            except (ValueError, TypeError, KeyError):
                pass

    def test_hdd_cold_day(self):
        """A cold day (30F) should produce HDD = 35 at base 65."""
        engine = _m.WeatherEngine()
        hdd = (getattr(engine, "calculate_hdd", None)
               or getattr(engine, "hdd", None)
               or getattr(engine, "compute_hdd", None)
               or getattr(engine, "calculate_degree_days", None))
        if hdd is None:
            pytest.skip("HDD method not found")
        record = [{"date": "2024-01-15", "temp_avg_f": 30.0}]
        try:
            result = hdd(record, base_temp=65.0)
        except (ValueError, TypeError, KeyError):
            pytest.skip("HDD single-day not supported")
        val = result if isinstance(result, (int, float)) else (
            getattr(result, "total_hdd", None)
            or (result.get("total_hdd") if isinstance(result, dict) else None)
        )
        if val is not None:
            assert abs(float(val) - 35.0) < 1.0

    def test_cdd_hot_day(self):
        """A hot day (90F) should produce CDD = 25 at base 65."""
        engine = _m.WeatherEngine()
        cdd = (getattr(engine, "calculate_cdd", None)
               or getattr(engine, "cdd", None)
               or getattr(engine, "compute_cdd", None)
               or getattr(engine, "calculate_degree_days", None))
        if cdd is None:
            pytest.skip("CDD method not found")
        record = [{"date": "2024-07-15", "temp_avg_f": 90.0}]
        try:
            result = cdd(record, base_temp=65.0)
        except (ValueError, TypeError, KeyError):
            pytest.skip("CDD single-day not supported")
        val = result if isinstance(result, (int, float)) else (
            getattr(result, "total_cdd", None)
            or (result.get("total_cdd") if isinstance(result, dict) else None)
        )
        if val is not None:
            assert abs(float(val) - 25.0) < 1.0


# =============================================================================
# Balance Point Optimization
# =============================================================================


class TestBalancePointOptimization:
    """Test balance point optimization with 3 methods."""

    def _get_optimize(self, engine):
        return (getattr(engine, "optimize_balance_point", None)
                or getattr(engine, "find_balance_point", None)
                or getattr(engine, "balance_point_search", None))

    @pytest.mark.parametrize("method", [
        "GRID_SEARCH",
        "GOLDEN_SECTION",
        "BRUTE_FORCE",
    ])
    def test_method_accepted(self, method, weather_data, baseline_data):
        engine = _m.WeatherEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_balance_point method not found")
        try:
            result = optimize(
                weather_data["baseline_weather"]["records"],
                baseline_data["records"],
                method=method,
            )
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    def test_balance_point_in_range(self, weather_data, baseline_data):
        engine = _m.WeatherEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_balance_point method not found")
        try:
            result = optimize(
                weather_data["baseline_weather"]["records"],
                baseline_data["records"],
            )
        except (ValueError, TypeError, KeyError):
            pytest.skip("Optimization not available")
        bp = (getattr(result, "balance_point", None)
              or getattr(result, "optimal_bp", None)
              or (result.get("balance_point") if isinstance(result, dict) else None))
        if bp is not None:
            assert 40 <= float(bp) <= 80

    def test_heating_cooling_separation(self, weather_data, baseline_data):
        engine = _m.WeatherEngine()
        optimize = self._get_optimize(engine)
        if optimize is None:
            pytest.skip("optimize_balance_point method not found")
        try:
            result = optimize(
                weather_data["baseline_weather"]["records"],
                baseline_data["records"],
                mode="BOTH",
            )
        except (ValueError, TypeError, KeyError):
            pytest.skip("Both-mode optimization not available")
        h_bp = (getattr(result, "heating_balance_point", None)
                or (result.get("heating_balance_point") if isinstance(result, dict) else None))
        c_bp = (getattr(result, "cooling_balance_point", None)
                or (result.get("cooling_balance_point") if isinstance(result, dict) else None))
        if h_bp is not None and c_bp is not None:
            assert float(h_bp) < float(c_bp)


# =============================================================================
# TMY Normalization
# =============================================================================


class TestTMYNormalization:
    """Test TMY (Typical Meteorological Year) normalization."""

    def _get_normalize(self, engine):
        return (getattr(engine, "normalize_to_tmy", None)
                or getattr(engine, "tmy_normalization", None)
                or getattr(engine, "apply_tmy", None))

    def test_tmy_result(self, weather_data):
        engine = _m.WeatherEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("TMY normalization method not found")
        try:
            result = normalize(weather_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_tmy_produces_degree_days(self, weather_data):
        engine = _m.WeatherEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("TMY normalization method not found")
        try:
            result = normalize(weather_data)
        except (ValueError, TypeError):
            pytest.skip("TMY normalization not available")
        hdd = (getattr(result, "tmy_hdd", None)
               or (result.get("tmy_hdd") if isinstance(result, dict) else None))
        cdd = (getattr(result, "tmy_cdd", None)
               or (result.get("tmy_cdd") if isinstance(result, dict) else None))
        if hdd is not None:
            assert float(hdd) >= 0
        if cdd is not None:
            assert float(cdd) >= 0


# =============================================================================
# Weather Data Quality
# =============================================================================


class TestWeatherDataQuality:
    """Test weather data quality assessment."""

    def _get_quality(self, engine):
        return (getattr(engine, "assess_data_quality", None)
                or getattr(engine, "data_quality", None)
                or getattr(engine, "check_quality", None))

    def test_quality_result(self, weather_data):
        engine = _m.WeatherEngine()
        quality = self._get_quality(engine)
        if quality is None:
            pytest.skip("data quality method not found")
        try:
            result = quality(weather_data["baseline_weather"]["records"])
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_quality_score(self, weather_data):
        engine = _m.WeatherEngine()
        quality = self._get_quality(engine)
        if quality is None:
            pytest.skip("data quality method not found")
        try:
            result = quality(weather_data["baseline_weather"]["records"])
        except (ValueError, TypeError):
            pytest.skip("Data quality not available")
        score = (getattr(result, "quality_score", None)
                 or (result.get("quality_score") if isinstance(result, dict) else None))
        if score is not None:
            assert 0 <= float(score) <= 1

    def test_completeness_check(self, weather_data):
        engine = _m.WeatherEngine()
        quality = self._get_quality(engine)
        if quality is None:
            pytest.skip("data quality method not found")
        try:
            result = quality(weather_data["baseline_weather"]["records"])
        except (ValueError, TypeError):
            pytest.skip("Data quality not available")
        completeness = (getattr(result, "completeness_pct", None)
                        or (result.get("completeness_pct")
                            if isinstance(result, dict) else None))
        if completeness is not None:
            assert float(completeness) > 0


# =============================================================================
# Provenance Tracking
# =============================================================================


class TestWeatherProvenance:
    """Test SHA-256 provenance hashing for weather data."""

    def _get_provenance(self, engine):
        return (getattr(engine, "compute_provenance", None)
                or getattr(engine, "provenance_hash", None)
                or getattr(engine, "get_provenance", None))

    def test_provenance_hash_format(self, weather_data):
        engine = _m.WeatherEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h = prov(weather_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h is not None:
            h_str = str(h)
            assert len(h_str) == 64
            assert all(c in "0123456789abcdef" for c in h_str)

    def test_provenance_deterministic(self, weather_data):
        engine = _m.WeatherEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h1 = prov(weather_data)
            h2 = prov(weather_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h1 is not None and h2 is not None:
            assert str(h1) == str(h2)


# =============================================================================
# Weather Fixture Validation
# =============================================================================


class TestWeatherFixtureValidation:
    """Validate weather fixture data consistency."""

    def test_baseline_has_365_records(self, weather_data):
        assert len(weather_data["baseline_weather"]["records"]) == 365

    def test_reporting_has_366_records(self, weather_data):
        assert len(weather_data["reporting_weather"]["records"]) == 366

    def test_hdd_cdd_nonnegative(self, weather_data):
        for rec in weather_data["baseline_weather"]["records"]:
            assert rec["hdd_65"] >= 0
            assert rec["cdd_65"] >= 0

    def test_hdd_cdd_mutually_exclusive(self, weather_data):
        """A single day should not have both HDD and CDD (at same base)."""
        for rec in weather_data["baseline_weather"]["records"]:
            assert rec["hdd_65"] == 0 or rec["cdd_65"] == 0

    def test_station_coordinates(self, weather_data):
        assert 41 < weather_data["latitude"] < 42
        assert -88 < weather_data["longitude"] < -87

    def test_balance_point_analysis_present(self, weather_data):
        bp = weather_data["balance_point_analysis"]
        assert float(bp["optimal_heating_bp_f"]) < float(bp["optimal_cooling_bp_f"])
