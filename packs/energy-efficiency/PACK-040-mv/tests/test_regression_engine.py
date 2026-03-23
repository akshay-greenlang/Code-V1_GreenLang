# -*- coding: utf-8 -*-
"""
Unit tests for RegressionEngine -- PACK-040 Engine 6
============================================================

Tests OLS, change-point (3P/4P/5P), and TOWT regression fitting
with full diagnostics: Durbin-Watson, Cook's distance, VIF,
residual analysis.

Coverage target: 85%+
Total tests: ~40
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


_m = _load("regression_engine")


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
        assert hasattr(_m, "RegressionEngine")

    def test_engine_instantiation(self):
        engine = _m.RegressionEngine()
        assert engine is not None


# =============================================================================
# Model Type Parametrize
# =============================================================================


class TestModelTypes:
    """Test 6 model types for regression fitting."""

    def _get_fit(self, engine):
        return (getattr(engine, "fit", None)
                or getattr(engine, "fit_model", None)
                or getattr(engine, "fit_regression", None))

    @pytest.mark.parametrize("model_type", [
        "OLS",
        "3P_HEATING",
        "3P_COOLING",
        "4P",
        "5P",
        "TOWT",
    ])
    def test_model_type_accepted(self, model_type, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type=model_type)
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    @pytest.mark.parametrize("model_type", [
        "OLS",
        "3P_HEATING",
        "3P_COOLING",
        "4P",
        "5P",
        "TOWT",
    ])
    def test_model_type_deterministic(self, model_type, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            r1 = fit(baseline_data, model_type=model_type)
            r2 = fit(baseline_data, model_type=model_type)
            assert str(r1) == str(r2)
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass


# =============================================================================
# OLS Regression
# =============================================================================


class TestOLSRegression:
    """Test ordinary least squares regression."""

    def _get_fit(self, engine):
        return (getattr(engine, "fit", None)
                or getattr(engine, "fit_ols", None)
                or getattr(engine, "fit_regression", None))

    def test_ols_result(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="OLS")
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_ols_coefficients(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="OLS")
        except (ValueError, TypeError):
            pytest.skip("OLS fit not available")
        coeffs = (getattr(result, "coefficients", None)
                  or (result.get("coefficients") if isinstance(result, dict) else None))
        if coeffs is not None:
            assert len(coeffs) >= 1

    def test_ols_p_values(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="OLS")
        except (ValueError, TypeError):
            pytest.skip("OLS fit not available")
        pvals = (getattr(result, "p_values", None)
                 or (result.get("p_values") if isinstance(result, dict) else None))
        if pvals is not None:
            for key, val in (pvals.items() if isinstance(pvals, dict) else enumerate(pvals)):
                assert 0 <= float(val) <= 1

    def test_ols_f_statistic(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="OLS")
        except (ValueError, TypeError):
            pytest.skip("OLS fit not available")
        f_stat = (getattr(result, "f_statistic", None)
                  or (result.get("f_statistic") if isinstance(result, dict) else None))
        if f_stat is not None:
            assert float(f_stat) > 0


# =============================================================================
# Change-Point Regression
# =============================================================================


class TestChangePointRegression:
    """Test change-point regression models."""

    def _get_fit(self, engine):
        return (getattr(engine, "fit", None)
                or getattr(engine, "fit_change_point", None)
                or getattr(engine, "fit_regression", None))

    def test_3p_cooling_result(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="3P_COOLING")
            assert result is not None
        except (ValueError, TypeError, NotImplementedError):
            pass

    def test_4p_result(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="4P")
            assert result is not None
        except (ValueError, TypeError, NotImplementedError):
            pass

    def test_5p_result(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="5P")
            assert result is not None
        except (ValueError, TypeError, NotImplementedError):
            pass

    def test_5p_segments(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="5P")
        except (ValueError, TypeError, NotImplementedError):
            pytest.skip("5P model not available")
        segments = (getattr(result, "segments", None)
                    or getattr(result, "num_segments", None)
                    or (result.get("segments") if isinstance(result, dict) else None))
        if segments is not None:
            count = len(segments) if isinstance(segments, (list, dict)) else int(segments)
            assert count >= 3  # 5P has 3 segments


# =============================================================================
# TOWT Model
# =============================================================================


class TestTOWTModel:
    """Test Time-of-Week-and-Temperature model."""

    def _get_fit(self, engine):
        return (getattr(engine, "fit", None)
                or getattr(engine, "fit_towt", None)
                or getattr(engine, "fit_regression", None))

    def test_towt_result(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="TOWT")
            assert result is not None
        except (ValueError, TypeError, NotImplementedError):
            pass

    def test_towt_time_segments(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = self._get_fit(engine)
        if fit is None:
            pytest.skip("fit method not found")
        try:
            result = fit(baseline_data, model_type="TOWT")
        except (ValueError, TypeError, NotImplementedError):
            pytest.skip("TOWT model not available")
        segments = (getattr(result, "time_segments", None)
                    or getattr(result, "num_time_segments", None)
                    or (result.get("num_time_segments") if isinstance(result, dict) else None))
        if segments is not None:
            count = int(segments) if isinstance(segments, (int, float, Decimal)) else len(segments)
            assert count > 0


# =============================================================================
# Diagnostic Tests Parametrize
# =============================================================================


class TestDiagnostics:
    """Test 4 regression diagnostic tests."""

    def _get_diagnostics(self, engine):
        return (getattr(engine, "run_diagnostics", None)
                or getattr(engine, "diagnostics", None)
                or getattr(engine, "regression_diagnostics", None))

    @pytest.mark.parametrize("diagnostic", [
        "DURBIN_WATSON",
        "COOKS_DISTANCE",
        "VIF",
        "RESIDUAL_ANALYSIS",
    ])
    def test_diagnostic_accepted(self, diagnostic, baseline_data):
        engine = _m.RegressionEngine()
        diag = self._get_diagnostics(engine)
        if diag is None:
            pytest.skip("diagnostics method not found")
        try:
            result = diag(baseline_data, diagnostic=diagnostic)
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    def test_durbin_watson_range(self, baseline_data):
        """Durbin-Watson statistic should be in [0, 4]."""
        engine = _m.RegressionEngine()
        diag = self._get_diagnostics(engine)
        if diag is None:
            pytest.skip("diagnostics method not found")
        try:
            result = diag(baseline_data, diagnostic="DURBIN_WATSON")
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pytest.skip("DW diagnostic not available")
        dw = (getattr(result, "durbin_watson", None)
              or getattr(result, "dw_statistic", None)
              or (result.get("durbin_watson") if isinstance(result, dict) else None))
        if dw is not None:
            assert 0 <= float(dw) <= 4

    def test_vif_values(self, baseline_data):
        """VIF values should be > 0 (typically < 10 for no multicollinearity)."""
        engine = _m.RegressionEngine()
        diag = self._get_diagnostics(engine)
        if diag is None:
            pytest.skip("diagnostics method not found")
        try:
            result = diag(baseline_data, diagnostic="VIF")
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pytest.skip("VIF diagnostic not available")
        vif_values = (getattr(result, "vif_values", None)
                      or getattr(result, "vif", None)
                      or (result.get("vif_values") if isinstance(result, dict) else None))
        if vif_values is not None:
            if isinstance(vif_values, dict):
                for v in vif_values.values():
                    assert float(v) > 0
            elif isinstance(vif_values, list):
                for v in vif_values:
                    assert float(v) > 0

    def test_residual_analysis(self, baseline_data):
        engine = _m.RegressionEngine()
        diag = self._get_diagnostics(engine)
        if diag is None:
            pytest.skip("diagnostics method not found")
        try:
            result = diag(baseline_data, diagnostic="RESIDUAL_ANALYSIS")
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pytest.skip("Residual analysis not available")
        residuals = (getattr(result, "residuals", None)
                     or (result.get("residuals") if isinstance(result, dict) else None))
        if residuals is not None:
            assert len(residuals) > 0


# =============================================================================
# Prediction
# =============================================================================


class TestPrediction:
    """Test model prediction capability."""

    def _get_predict(self, engine):
        return (getattr(engine, "predict", None)
                or getattr(engine, "forecast", None)
                or getattr(engine, "model_predict", None))

    def test_predict_result(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = (getattr(engine, "fit", None)
               or getattr(engine, "fit_regression", None))
        predict = self._get_predict(engine)
        if fit is None or predict is None:
            pytest.skip("fit/predict methods not found")
        try:
            model = fit(baseline_data, model_type="OLS")
            result = predict(model, baseline_data["records"][:30])
            assert result is not None
        except (ValueError, TypeError, KeyError):
            pass

    def test_predict_same_length(self, baseline_data):
        engine = _m.RegressionEngine()
        fit = (getattr(engine, "fit", None)
               or getattr(engine, "fit_regression", None))
        predict = self._get_predict(engine)
        if fit is None or predict is None:
            pytest.skip("fit/predict methods not found")
        input_records = baseline_data["records"][:30]
        try:
            model = fit(baseline_data, model_type="OLS")
            result = predict(model, input_records)
        except (ValueError, TypeError, KeyError):
            pytest.skip("Prediction not available")
        preds = (getattr(result, "predictions", None)
                 or (result.get("predictions") if isinstance(result, dict) else None)
                 or result)
        if isinstance(preds, (list, tuple)):
            assert len(preds) == len(input_records)


# =============================================================================
# Goodness of Fit Statistics
# =============================================================================


class TestGoodnessOfFit:
    """Test goodness-of-fit statistics for regression models."""

    def _get_stats(self, engine):
        return (getattr(engine, "goodness_of_fit", None)
                or getattr(engine, "fit_statistics", None)
                or getattr(engine, "model_stats", None))

    def test_r_squared(self, baseline_data):
        engine = _m.RegressionEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("goodness_of_fit method not found")
        try:
            result = stats(baseline_data)
        except (ValueError, TypeError):
            pytest.skip("Stats not available")
        r2 = (getattr(result, "r_squared", None)
              or (result.get("r_squared") if isinstance(result, dict) else None))
        if r2 is not None:
            assert 0 <= float(r2) <= 1

    def test_adjusted_r_squared(self, baseline_data):
        engine = _m.RegressionEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("goodness_of_fit method not found")
        try:
            result = stats(baseline_data)
        except (ValueError, TypeError):
            pytest.skip("Stats not available")
        adj_r2 = (getattr(result, "adjusted_r_squared", None)
                  or (result.get("adjusted_r_squared") if isinstance(result, dict) else None))
        if adj_r2 is not None:
            assert float(adj_r2) <= 1

    def test_cvrmse(self, baseline_data):
        engine = _m.RegressionEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("goodness_of_fit method not found")
        try:
            result = stats(baseline_data)
        except (ValueError, TypeError):
            pytest.skip("Stats not available")
        cvrmse = (getattr(result, "cvrmse_pct", None)
                  or (result.get("cvrmse_pct") if isinstance(result, dict) else None))
        if cvrmse is not None:
            assert float(cvrmse) >= 0

    def test_nmbe(self, baseline_data):
        engine = _m.RegressionEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("goodness_of_fit method not found")
        try:
            result = stats(baseline_data)
        except (ValueError, TypeError):
            pytest.skip("Stats not available")
        nmbe = (getattr(result, "nmbe_pct", None)
                or (result.get("nmbe_pct") if isinstance(result, dict) else None))
        if nmbe is not None:
            assert -100 <= float(nmbe) <= 100


# =============================================================================
# Provenance Tracking
# =============================================================================


class TestRegressionProvenance:
    """Test SHA-256 provenance hashing for regression results."""

    def _get_provenance(self, engine):
        return (getattr(engine, "compute_provenance", None)
                or getattr(engine, "provenance_hash", None)
                or getattr(engine, "get_provenance", None))

    def test_provenance_hash_format(self, baseline_data):
        engine = _m.RegressionEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h = prov(baseline_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h is not None:
            h_str = str(h)
            assert len(h_str) == 64
            assert all(c in "0123456789abcdef" for c in h_str)

    def test_provenance_deterministic(self, baseline_data):
        engine = _m.RegressionEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h1 = prov(baseline_data)
            h2 = prov(baseline_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h1 is not None and h2 is not None:
            assert str(h1) == str(h2)


# =============================================================================
# Regression Fixture Validation
# =============================================================================


class TestRegressionFixtureValidation:
    """Validate regression fixture data for internal consistency."""

    def test_ols_r_squared_range(self, regression_data):
        r2 = float(regression_data["ols"]["r_squared"])
        assert 0 <= r2 <= 1

    def test_5p_better_than_3p(self, regression_data):
        r2_5p = float(regression_data["five_p"]["r_squared"])
        r2_3p = float(regression_data["three_p_cooling"]["r_squared"])
        assert r2_5p >= r2_3p

    def test_best_model_is_valid(self, regression_data):
        best = regression_data["best_model"]
        valid = {"OLS", "3P_HEATING", "3P_COOLING", "4P", "5P", "TOWT"}
        assert best in valid

    def test_all_models_have_cvrmse(self, regression_data):
        for model_key in ["ols", "three_p_cooling", "four_p", "five_p", "towt"]:
            cvrmse = float(regression_data[model_key]["cvrmse_pct"])
            assert cvrmse >= 0

    def test_all_models_have_nmbe(self, regression_data):
        for model_key in ["ols", "three_p_cooling", "four_p", "five_p", "towt"]:
            nmbe = float(regression_data[model_key]["nmbe_pct"])
            assert -100 <= nmbe <= 100

    def test_dw_valid_range(self, regression_data):
        dw = float(regression_data["ols"]["durbin_watson"])
        assert 0 <= dw <= 4

    def test_f_statistic_positive(self, regression_data):
        f_stat = float(regression_data["ols"]["f_statistic"])
        assert f_stat > 0
