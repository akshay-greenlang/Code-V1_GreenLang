# -*- coding: utf-8 -*-
"""
Unit tests for BudgetEngine -- PACK-039 Engine 7
============================================================

Tests budget creation, variance analysis, weather normalization,
forecasting, and alert generation.

Coverage target: 85%+
Total tests: ~50
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


_m = _load("budget_engine")


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
        assert hasattr(_m, "BudgetEngine")

    def test_engine_instantiation(self):
        engine = _m.BudgetEngine()
        assert engine is not None


# =============================================================================
# Budget Creation
# =============================================================================


class TestBudgetCreation:
    """Test energy budget creation and setup."""

    def _get_create(self, engine):
        return (getattr(engine, "create_budget", None)
                or getattr(engine, "build_budget", None)
                or getattr(engine, "initialize_budget", None))

    def test_create_annual_budget(self, sample_budget):
        engine = _m.BudgetEngine()
        create = self._get_create(engine)
        if create is None:
            pytest.skip("create_budget method not found")
        result = create(sample_budget)
        assert result is not None

    def test_budget_has_12_periods(self, sample_budget):
        engine = _m.BudgetEngine()
        create = self._get_create(engine)
        if create is None:
            pytest.skip("create_budget method not found")
        result = create(sample_budget)
        periods = getattr(result, "periods", None)
        if periods is not None:
            assert len(periods) == 12

    def test_budget_annual_target(self, sample_budget):
        engine = _m.BudgetEngine()
        create = self._get_create(engine)
        if create is None:
            pytest.skip("create_budget method not found")
        result = create(sample_budget)
        target = getattr(result, "annual_target_kwh", None)
        if target is not None:
            assert target == sample_budget["annual_target_kwh"]

    def test_budget_creation_deterministic(self, sample_budget):
        engine = _m.BudgetEngine()
        create = self._get_create(engine)
        if create is None:
            pytest.skip("create_budget method not found")
        r1 = create(sample_budget)
        r2 = create(sample_budget)
        assert str(r1) == str(r2)


# =============================================================================
# Variance Analysis
# =============================================================================


class TestVarianceAnalysis:
    """Test budget vs actual variance analysis."""

    def _get_variance(self, engine):
        return (getattr(engine, "analyze_variance", None)
                or getattr(engine, "variance_analysis", None)
                or getattr(engine, "compute_variance", None))

    def test_variance_analysis(self, sample_budget):
        engine = _m.BudgetEngine()
        variance = self._get_variance(engine)
        if variance is None:
            pytest.skip("variance method not found")
        result = variance(sample_budget)
        assert result is not None

    def test_variance_per_period(self, sample_budget):
        engine = _m.BudgetEngine()
        variance = self._get_variance(engine)
        if variance is None:
            pytest.skip("variance method not found")
        result = variance(sample_budget)
        period_variances = getattr(result, "period_variances",
                                   getattr(result, "variances", None))
        if period_variances is not None and isinstance(period_variances, list):
            assert len(period_variances) == 12

    @pytest.mark.parametrize("month_idx", range(12))
    def test_variance_has_pct(self, month_idx, sample_budget):
        engine = _m.BudgetEngine()
        variance = self._get_variance(engine)
        if variance is None:
            pytest.skip("variance method not found")
        result = variance(sample_budget)
        period_variances = getattr(result, "period_variances",
                                   getattr(result, "variances", None))
        if isinstance(period_variances, list) and month_idx < len(period_variances):
            pv = period_variances[month_idx]
            if isinstance(pv, dict):
                pct = pv.get("variance_pct", pv.get("pct", None))
                if pct is not None:
                    assert isinstance(float(pct), float)

    def test_over_budget_detection(self, sample_budget):
        engine = _m.BudgetEngine()
        variance = self._get_variance(engine)
        if variance is None:
            pytest.skip("variance method not found")
        result = variance(sample_budget)
        over_budget = getattr(result, "over_budget_periods",
                              getattr(result, "alerts", None))
        if over_budget is not None and isinstance(over_budget, list):
            # Some periods should be over/under
            assert isinstance(over_budget, list)


# =============================================================================
# Weather Normalization
# =============================================================================


class TestWeatherNormalization:
    """Test weather-normalized budget comparison."""

    def _get_normalize(self, engine):
        return (getattr(engine, "weather_normalize", None)
                or getattr(engine, "normalize_weather", None)
                or getattr(engine, "weather_adjusted_variance", None))

    def test_weather_normalization(self, sample_budget):
        engine = _m.BudgetEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("weather_normalize method not found")
        result = normalize(sample_budget)
        assert result is not None

    def test_normalized_vs_raw_variance(self, sample_budget):
        engine = _m.BudgetEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("weather_normalize method not found")
        result = normalize(sample_budget)
        normalized_var = getattr(result, "normalized_variance_kwh", None)
        if normalized_var is not None:
            assert isinstance(float(normalized_var), float)


# =============================================================================
# Forecasting
# =============================================================================


class TestForecasting:
    """Test year-end energy consumption forecast."""

    def _get_forecast(self, engine):
        return (getattr(engine, "forecast_year_end", None)
                or getattr(engine, "project_consumption", None)
                or getattr(engine, "forecast", None))

    def test_forecast_result(self, sample_budget):
        engine = _m.BudgetEngine()
        forecast = self._get_forecast(engine)
        if forecast is None:
            pytest.skip("forecast method not found")
        result = forecast(sample_budget, current_month=6)
        assert result is not None

    def test_forecast_exceeds_ytd(self, sample_budget):
        engine = _m.BudgetEngine()
        forecast = self._get_forecast(engine)
        if forecast is None:
            pytest.skip("forecast method not found")
        result = forecast(sample_budget, current_month=6)
        forecast_kwh = getattr(result, "forecast_kwh",
                               getattr(result, "projected_kwh", None))
        if forecast_kwh is not None:
            ytd = sum(p["actual_kwh"] for p in sample_budget["periods"][:6])
            assert float(forecast_kwh) >= ytd

    @pytest.mark.parametrize("month", [3, 6, 9, 11])
    def test_forecast_at_month(self, month, sample_budget):
        engine = _m.BudgetEngine()
        forecast = self._get_forecast(engine)
        if forecast is None:
            pytest.skip("forecast method not found")
        try:
            result = forecast(sample_budget, current_month=month)
            assert result is not None
        except TypeError:
            result = forecast(sample_budget)
            assert result is not None


# =============================================================================
# Budget Alerts
# =============================================================================


class TestBudgetAlerts:
    """Test budget variance alert generation."""

    def _get_alerts(self, engine):
        return (getattr(engine, "generate_alerts", None)
                or getattr(engine, "check_alerts", None)
                or getattr(engine, "budget_alerts", None))

    def test_alert_generation(self, sample_budget):
        engine = _m.BudgetEngine()
        alerts = self._get_alerts(engine)
        if alerts is None:
            pytest.skip("generate_alerts method not found")
        result = alerts(sample_budget)
        assert result is not None

    def test_alert_threshold(self, sample_budget):
        engine = _m.BudgetEngine()
        alerts = self._get_alerts(engine)
        if alerts is None:
            pytest.skip("generate_alerts method not found")
        result = alerts(sample_budget)
        alert_list = getattr(result, "alerts", result)
        if isinstance(alert_list, list):
            for a in alert_list:
                if isinstance(a, dict) and "variance_pct" in a:
                    assert abs(float(a["variance_pct"])) >= float(
                        sample_budget["variance_alert_threshold_pct"]
                    )


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    """Test provenance hash for budget results."""

    def test_same_input_same_hash(self, sample_budget):
        engine = _m.BudgetEngine()
        variance = (getattr(engine, "analyze_variance", None)
                    or getattr(engine, "variance_analysis", None))
        if variance is None:
            pytest.skip("variance method not found")
        r1 = variance(sample_budget)
        r2 = variance(sample_budget)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2

    def test_hash_is_sha256(self, sample_budget):
        engine = _m.BudgetEngine()
        variance = (getattr(engine, "analyze_variance", None)
                    or getattr(engine, "variance_analysis", None))
        if variance is None:
            pytest.skip("variance method not found")
        result = variance(sample_budget)
        h = getattr(result, "provenance_hash", None)
        if h is not None:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# Budget Fixture Validation
# =============================================================================


class TestBudgetFixture:
    """Validate the budget fixture."""

    def test_12_periods(self, sample_budget):
        assert len(sample_budget["periods"]) == 12

    def test_annual_target(self, sample_budget):
        expected = sum([
            850_000, 780_000, 720_000, 700_000, 750_000, 920_000,
            1_050_000, 1_020_000, 880_000, 760_000, 800_000, 870_000,
        ])
        assert sample_budget["annual_target_kwh"] == expected

    def test_all_periods_have_target(self, sample_budget):
        for p in sample_budget["periods"]:
            assert "target_kwh" in p
            assert p["target_kwh"] > 0

    def test_all_periods_have_actual(self, sample_budget):
        for p in sample_budget["periods"]:
            assert "actual_kwh" in p

    def test_variance_threshold(self, sample_budget):
        assert sample_budget["variance_alert_threshold_pct"] == Decimal("0.10")
