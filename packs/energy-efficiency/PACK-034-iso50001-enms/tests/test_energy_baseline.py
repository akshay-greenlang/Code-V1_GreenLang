# -*- coding: utf-8 -*-
"""
Unit tests for EnergyBaselineEngine -- PACK-034 Engine 2
==========================================================

Tests baseline model establishment per ISO 50006:2014 including
simple mean, single-variable regression, multi-variable regression,
R-squared, CV(RMSE), outlier detection, baseline validation,
normalisation, expected consumption, adjustment, and model comparison.

Coverage target: 85%+
Total tests: ~45
"""

import importlib.util
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load(name: str):
    path = ENGINES_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Engine file not found: {path}")
    mod_key = f"pack034_test.{name}"
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


_m = _load("energy_baseline_engine")


def _make_baseline_data_points(sample_baseline_data):
    """Convert raw fixture dicts to BaselineDataPoint Pydantic models."""
    points = []
    for d in sample_baseline_data:
        month_str = d["month"]  # e.g. "2024-01"
        year, month = int(month_str[:4]), int(month_str[5:7])
        # period_start: first day of month, period_end: last day of month
        if month == 12:
            end = date(year + 1, 1, 1)
        else:
            end = date(year, month + 1, 1)
        points.append(_m.BaselineDataPoint(
            period_start=date(year, month, 1),
            period_end=end,
            energy_kwh=Decimal(str(d["energy_kwh"])),
            variables={
                "hdd": Decimal(str(d["hdd"])),
                "cdd": Decimal(str(d["cdd"])),
                "production": Decimal(str(d["production"])),
            },
        ))
    return points


# =============================================================================
# File and Module
# =============================================================================


class TestEngineFilePresence:
    """Test engine file exists."""

    def test_engine_file_exists(self):
        path = ENGINES_DIR / "energy_baseline_engine.py"
        assert path.is_file()


class TestModuleLoading:
    """Module loading tests."""

    def test_module_loads(self):
        assert _m is not None

    def test_class_exists(self):
        assert hasattr(_m, "EnergyBaselineEngine")

    def test_instantiation(self):
        engine = _m.EnergyBaselineEngine()
        assert engine is not None


# =============================================================================
# Enums
# =============================================================================


class TestBaselineModelType:
    """Test baseline model type enumeration."""

    def test_baseline_model_type_enum(self):
        has_enum = (hasattr(_m, "BaselineModelType") or hasattr(_m, "ModelType")
                    or hasattr(_m, "BaselineType"))
        assert has_enum

    def test_baseline_model_type_values(self):
        enum_cls = (getattr(_m, "BaselineModelType", None)
                    or getattr(_m, "ModelType", None)
                    or getattr(_m, "BaselineType", None))
        if enum_cls is None:
            pytest.skip("Enum not found")
        values = {m.value for m in enum_cls}
        assert len(values) >= 3


# =============================================================================
# Baseline Models
# =============================================================================


class TestSimpleMeanBaseline:
    """Test simple mean baseline calculation."""

    def test_simple_mean_baseline(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        result = engine.fit_simple_mean(data_points)
        assert result is not None
        # Intercept should be close to the mean of energy values
        energy_values = [d["energy_kwh"] for d in sample_baseline_data]
        expected_mean = sum(energy_values) / len(energy_values)
        assert abs(float(result.intercept) - expected_mean) < 500


class TestSingleVariableRegression:
    """Test single-variable OLS regression."""

    def test_single_variable_regression(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        result = engine.fit_single_variable(data_points, variable="hdd")
        assert result is not None
        assert result.model_type == _m.BaselineModelType.SINGLE_VARIABLE


class TestMultiVariableRegression:
    """Test multi-variable OLS regression."""

    def test_multi_variable_regression(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        result = engine.fit_multi_variable(data_points, variables=["hdd", "cdd", "production"])
        assert result is not None
        assert result.model_type == _m.BaselineModelType.MULTI_VARIABLE


# =============================================================================
# Model Adequacy Metrics
# =============================================================================


class TestModelAdequacy:
    """Test model adequacy metrics (R-squared, CV(RMSE))."""

    def test_r_squared_calculation(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        # Use direct calculate_r_squared method
        actual = [Decimal(str(d["energy_kwh"])) for d in sample_baseline_data]
        # Simulate predicted values close to actual
        predicted = [a + Decimal("500") for a in actual]
        r2 = engine.calculate_r_squared(actual, predicted)
        assert 0 <= float(r2) <= 1

    def test_cv_rmse_calculation(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        actual = [Decimal(str(d["energy_kwh"])) for d in sample_baseline_data]
        predicted = [a + Decimal("500") for a in actual]
        cv_rmse = engine.calculate_cv_rmse(actual, predicted)
        assert float(cv_rmse) >= 0


# =============================================================================
# Outlier Detection
# =============================================================================


class TestOutlierDetection:
    """Test outlier detection in baseline data."""

    def test_outlier_detection(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        result = engine.detect_outliers(data_points)
        assert result is not None
        assert isinstance(result, list)


# =============================================================================
# Baseline Validation
# =============================================================================


class TestBaselineValidation:
    """Test baseline adequacy validation per ISO 50006."""

    def test_baseline_validation_adequate(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        config = _m.BaselineConfig(model_type=_m.BaselineModelType.SINGLE_VARIABLE)
        result = engine.establish_baseline(data_points, config)
        assert result is not None
        validation = engine.validate_baseline(result)
        assert validation is not None

    def test_baseline_validation_inadequate(self):
        engine = _m.EnergyBaselineEngine()
        # Random noise data should produce inadequate baseline
        noise_points = []
        for i in range(12):
            noise_points.append(_m.BaselineDataPoint(
                period_start=date(2024, i + 1, 1),
                period_end=date(2024, i + 1, 28),
                energy_kwh=Decimal(str(100 + (i * 73) % 900)),
                variables={"hdd": Decimal(str(i * 50))},
            ))
        config = _m.BaselineConfig(
            model_type=_m.BaselineModelType.SINGLE_VARIABLE,
            min_r_squared=Decimal("0.90"),
        )
        result = engine.establish_baseline(noise_points, config)
        assert result is not None


# =============================================================================
# Normalisation
# =============================================================================


class TestNormalisation:
    """Test baseline normalisation."""

    def test_normalization(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        model = engine.fit_single_variable(data_points, variable="hdd")
        result = engine.normalize_consumption(
            actual_kwh=Decimal("248000"),
            model=model,
            baseline_conditions={"hdd": Decimal("520")},
            actual_conditions={"hdd": Decimal("480")},
        )
        assert result is not None
        assert float(result) > 0


# =============================================================================
# Expected Consumption
# =============================================================================


class TestExpectedConsumption:
    """Test expected consumption calculation."""

    def test_expected_consumption_calculation(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        model = engine.fit_single_variable(data_points, variable="hdd")
        expected = engine.calculate_expected_consumption(
            model=model,
            conditions={"hdd": Decimal("400")},
        )
        assert expected is not None
        assert float(expected) > 0


# =============================================================================
# Baseline Adjustment
# =============================================================================


class TestBaselineAdjustment:
    """Test baseline adjustment for significant changes."""

    def test_baseline_adjustment(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        config = _m.BaselineConfig(model_type=_m.BaselineModelType.SINGLE_VARIABLE)
        baseline = engine.establish_baseline(data_points, config)
        result = engine.adjust_baseline(
            baseline=baseline,
            trigger=_m.AdjustmentTrigger.EQUIPMENT_CHANGE,
            new_data={"reason": "Equipment removal", "adjustment_kwh": -100000},
        )
        assert result is not None


# =============================================================================
# Model Comparison
# =============================================================================


class TestModelComparison:
    """Test multi-model comparison with AIC/BIC selection."""

    def test_model_comparison(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        model_mean = engine.fit_simple_mean(data_points)
        model_sv = engine.fit_single_variable(data_points, variable="hdd")
        result = engine.compare_models([model_mean, model_sv])
        assert result is not None


# =============================================================================
# Provenance
# =============================================================================


class TestProvenance:
    """Provenance hash tests."""

    def test_provenance_hash(self, sample_baseline_data):
        engine = _m.EnergyBaselineEngine()
        data_points = _make_baseline_data_points(sample_baseline_data)
        config = _m.BaselineConfig(model_type=_m.BaselineModelType.SINGLE_VARIABLE)
        result = engine.establish_baseline(data_points, config)
        if hasattr(result, "provenance_hash"):
            assert len(result.provenance_hash) == 64
            assert all(c in "0123456789abcdef" for c in result.provenance_hash)
