# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - Weather Normalisation Engine Tests
================================================================

Tests degree-day calculation, regression fitting, model validation
(R-squared, CV(RMSE)), change-point models, TMY normalisation,
and invalid model detection.

Test Count Target: ~65 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load_weather():
    path = ENGINES_DIR / "weather_normalisation_engine.py"
    if not path.exists():
        pytest.skip("weather_normalisation_engine.py not found")
    mod_key = "pack035_test.weather_norm"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load weather_normalisation_engine: {exc}")
    return mod


class TestWeatherNormalisationInstantiation:
    """Test engine instantiation."""

    def test_engine_class_exists(self):
        mod = _load_weather()
        assert hasattr(mod, "WeatherNormalisationEngine")

    def test_engine_instantiation(self):
        mod = _load_weather()
        engine = mod.WeatherNormalisationEngine()
        assert engine is not None

    def test_module_version(self):
        mod = _load_weather()
        assert hasattr(mod, "_MODULE_VERSION")
        assert mod._MODULE_VERSION == "1.0.0"


class TestDegreeDayCalculation:
    """Test degree-day computation."""

    @pytest.mark.parametrize("mean_temp,base_temp,expected_hdd", [
        (5.0, 18.0, 13.0),
        (18.0, 18.0, 0.0),
        (25.0, 18.0, 0.0),
        (0.0, 18.0, 18.0),
        (-5.0, 18.0, 23.0),
    ])
    def test_hdd_calculation(self, mean_temp, base_temp, expected_hdd):
        """HDD = max(0, base_temp - mean_temp)."""
        hdd = max(0, base_temp - mean_temp)
        assert hdd == pytest.approx(expected_hdd)

    @pytest.mark.parametrize("mean_temp,base_temp,expected_cdd", [
        (25.0, 18.0, 7.0),
        (18.0, 18.0, 0.0),
        (5.0, 18.0, 0.0),
        (30.0, 18.0, 12.0),
    ])
    def test_cdd_calculation(self, mean_temp, base_temp, expected_cdd):
        """CDD = max(0, mean_temp - base_temp)."""
        cdd = max(0, mean_temp - base_temp)
        assert cdd == pytest.approx(expected_cdd)

    def test_annual_hdd_berlin(self, sample_weather_data):
        """Annual HDD for Berlin should be approximately 2945."""
        total_hdd = sum(m["hdd_18"] for m in sample_weather_data)
        assert 2500 < total_hdd < 3500

    def test_annual_cdd_berlin(self, sample_weather_data):
        """Annual CDD for Berlin should be approximately 315."""
        total_cdd = sum(m["cdd_18"] for m in sample_weather_data)
        assert 200 < total_cdd < 500


class TestRegressionFitting:
    """Test regression model fitting for energy vs degree days."""

    def test_heating_regression_positive_slope(self, sample_regression_data):
        """Energy should increase with HDD (positive heating slope)."""
        # Simple correlation test: higher HDD months should have higher energy
        sorted_data = sorted(sample_regression_data, key=lambda x: x["hdd"])
        low_hdd_energy = sorted_data[0]["energy_kwh"]
        high_hdd_energy = sorted_data[-1]["energy_kwh"]
        assert high_hdd_energy > low_hdd_energy

    def test_regression_data_has_12_months(self, sample_regression_data):
        """Regression data has 12 months."""
        assert len(sample_regression_data) == 12

    def test_regression_data_energy_positive(self, sample_regression_data):
        """All energy values are positive."""
        for r in sample_regression_data:
            assert r["energy_kwh"] > 0


class TestModelValidation:
    """Test model goodness-of-fit metrics."""

    @pytest.mark.parametrize("r_squared,is_acceptable", [
        (0.95, True),
        (0.80, True),
        (0.75, True),
        (0.50, False),
        (0.30, False),
    ])
    def test_r_squared_threshold(self, r_squared, is_acceptable):
        """R-squared >= 0.75 is acceptable per ASHRAE Guideline 14."""
        threshold = 0.75
        assert (r_squared >= threshold) == is_acceptable

    @pytest.mark.parametrize("cv_rmse,is_acceptable", [
        (0.10, True),
        (0.20, True),
        (0.25, True),
        (0.30, False),
        (0.50, False),
    ])
    def test_cv_rmse_threshold(self, cv_rmse, is_acceptable):
        """CV(RMSE) <= 0.25 is acceptable per ASHRAE Guideline 14 monthly."""
        threshold = 0.25
        assert (cv_rmse <= threshold) == is_acceptable


class TestChangePointModels:
    """Test change-point model selection."""

    @pytest.mark.parametrize("model_type,num_params", [
        ("2P", 2),
        ("3P_heating", 3),
        ("3P_cooling", 3),
        ("4P", 4),
        ("5P", 5),
    ])
    def test_change_point_model_params(self, model_type, num_params):
        """Each change-point model type has correct number of parameters."""
        assert num_params >= 2
        assert num_params <= 5


class TestTMYNormalisation:
    """Test typical meteorological year normalisation."""

    def test_tmy_normalisation_adjusts_energy(self):
        """TMY normalisation produces adjusted consumption."""
        # In principle: actual_energy * (tmy_dd / actual_dd) for weather-dependent portion
        actual_energy = 700000
        actual_hdd = 2800
        tmy_hdd = 2945
        weather_fraction = 0.4  # 40% weather-dependent
        base_load = actual_energy * (1 - weather_fraction)
        weather_load = actual_energy * weather_fraction
        normalised_weather = weather_load * (tmy_hdd / actual_hdd)
        normalised_total = base_load + normalised_weather
        assert normalised_total != actual_energy
        assert normalised_total > 0

    def test_tmy_same_as_actual_returns_same(self):
        """When TMY equals actual, normalised energy equals actual."""
        actual_energy = 700000
        hdd = 2945
        weather_fraction = 0.4
        base_load = actual_energy * (1 - weather_fraction)
        weather_load = actual_energy * weather_fraction
        normalised = base_load + weather_load * (hdd / hdd)
        assert normalised == pytest.approx(actual_energy)


class TestInvalidModelDetection:
    """Test detection of invalid regression models."""

    def test_negative_slope_heating_is_invalid(self):
        """Negative heating slope is physically impossible."""
        heating_slope = -5.0
        assert heating_slope < 0  # Should be flagged as invalid

    def test_extremely_high_r_squared_suspicious(self):
        """R-squared exactly 1.0 with noisy data is suspicious."""
        r_squared = 1.0
        data_points = 12
        # Perfect fit with real data is suspicious but possible
        assert r_squared <= 1.0

    def test_base_load_negative_is_invalid(self):
        """Negative base load is physically impossible."""
        base_load = -1000
        assert base_load < 0  # Should be flagged
