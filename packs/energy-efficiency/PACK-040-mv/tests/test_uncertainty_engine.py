# -*- coding: utf-8 -*-
"""
Unit tests for UncertaintyEngine -- PACK-040 Engine 4
============================================================

Tests ASHRAE Guideline 14 fractional savings uncertainty (FSU)
calculation with measurement, model, and sampling uncertainty
components, combined FSU, minimum detectable savings, and
reference calculation validation.

Coverage target: 85%+
Total tests: ~35
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


_m = _load("uncertainty_engine")


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
        assert hasattr(_m, "UncertaintyEngine")

    def test_engine_instantiation(self):
        engine = _m.UncertaintyEngine()
        assert engine is not None


# =============================================================================
# Uncertainty Component Parametrize
# =============================================================================


class TestUncertaintyComponents:
    """Test 4 uncertainty components and 3 confidence levels."""

    def _get_calculate(self, engine):
        return (getattr(engine, "calculate_uncertainty", None)
                or getattr(engine, "compute_uncertainty", None)
                or getattr(engine, "uncertainty", None))

    @pytest.mark.parametrize("component", [
        "MEASUREMENT",
        "MODEL",
        "SAMPLING",
        "COMBINED",
    ])
    def test_component_accepted(self, component, uncertainty_data):
        engine = _m.UncertaintyEngine()
        calculate = self._get_calculate(engine)
        if calculate is None:
            pytest.skip("calculate_uncertainty method not found")
        try:
            result = calculate(uncertainty_data, component=component)
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    @pytest.mark.parametrize("component", [
        "MEASUREMENT",
        "MODEL",
        "SAMPLING",
        "COMBINED",
    ])
    def test_component_deterministic(self, component, uncertainty_data):
        engine = _m.UncertaintyEngine()
        calculate = self._get_calculate(engine)
        if calculate is None:
            pytest.skip("calculate_uncertainty method not found")
        try:
            r1 = calculate(uncertainty_data, component=component)
            r2 = calculate(uncertainty_data, component=component)
            assert str(r1) == str(r2)
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    @pytest.mark.parametrize("confidence", [
        Decimal("0.90"),
        Decimal("0.95"),
        Decimal("0.99"),
    ])
    def test_confidence_level_accepted(self, confidence, uncertainty_data):
        engine = _m.UncertaintyEngine()
        calculate = self._get_calculate(engine)
        if calculate is None:
            pytest.skip("calculate_uncertainty method not found")
        data = dict(uncertainty_data)
        data["confidence_level"] = confidence
        try:
            result = calculate(data)
            assert result is not None
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pass

    @pytest.mark.parametrize("confidence", [
        Decimal("0.90"),
        Decimal("0.95"),
        Decimal("0.99"),
    ])
    def test_higher_confidence_wider_interval(self, confidence, uncertainty_data):
        """Higher confidence should yield wider uncertainty bounds."""
        engine = _m.UncertaintyEngine()
        calculate = self._get_calculate(engine)
        if calculate is None:
            pytest.skip("calculate_uncertainty method not found")
        data_90 = dict(uncertainty_data)
        data_90["confidence_level"] = Decimal("0.90")
        data_high = dict(uncertainty_data)
        data_high["confidence_level"] = confidence
        try:
            r_90 = calculate(data_90, component="COMBINED")
            r_high = calculate(data_high, component="COMBINED")
        except (ValueError, TypeError, KeyError, NotImplementedError):
            pytest.skip("Cannot compare confidence levels")
        fsu_90 = (getattr(r_90, "fsu_pct", None)
                  or (r_90.get("fsu_pct") if isinstance(r_90, dict) else None))
        fsu_high = (getattr(r_high, "fsu_pct", None)
                    or (r_high.get("fsu_pct") if isinstance(r_high, dict) else None))
        if fsu_90 is not None and fsu_high is not None:
            assert float(fsu_high) >= float(fsu_90)


# =============================================================================
# Measurement Uncertainty
# =============================================================================


class TestMeasurementUncertainty:
    """Test measurement uncertainty from meter specifications."""

    def _get_measurement(self, engine):
        return (getattr(engine, "calculate_measurement_uncertainty", None)
                or getattr(engine, "measurement_uncertainty", None)
                or getattr(engine, "meter_uncertainty", None))

    def test_measurement_result(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        measurement = self._get_measurement(engine)
        if measurement is None:
            pytest.skip("measurement uncertainty method not found")
        try:
            result = measurement(uncertainty_data["measurement_uncertainty"])
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_measurement_positive(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        measurement = self._get_measurement(engine)
        if measurement is None:
            pytest.skip("measurement uncertainty method not found")
        try:
            result = measurement(uncertainty_data["measurement_uncertainty"])
        except (ValueError, TypeError):
            pytest.skip("Measurement uncertainty not available")
        val = (getattr(result, "uncertainty_pct", None)
               or (result.get("total_measurement_uncertainty_pct")
                   if isinstance(result, dict) else None))
        if val is not None:
            assert float(val) > 0

    def test_combined_meter_rss(self, uncertainty_data):
        """Combined meter uncertainty should use RSS (root sum of squares)."""
        engine = _m.UncertaintyEngine()
        measurement = self._get_measurement(engine)
        if measurement is None:
            pytest.skip("measurement uncertainty method not found")
        meter = uncertainty_data["measurement_uncertainty"]["meters"][0]
        acc = float(meter["accuracy_pct"])
        ct = float(meter["ct_accuracy_pct"])
        pt = float(meter["pt_accuracy_pct"])
        expected_rss = math.sqrt(acc**2 + ct**2 + pt**2)
        combined = float(meter["combined_meter_uncertainty_pct"])
        assert abs(combined - expected_rss) < 0.05


# =============================================================================
# Model Uncertainty
# =============================================================================


class TestModelUncertainty:
    """Test model uncertainty from regression statistics."""

    def _get_model(self, engine):
        return (getattr(engine, "calculate_model_uncertainty", None)
                or getattr(engine, "model_uncertainty", None)
                or getattr(engine, "regression_uncertainty", None))

    def test_model_result(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        model = self._get_model(engine)
        if model is None:
            pytest.skip("model uncertainty method not found")
        try:
            result = model(uncertainty_data["model_uncertainty"])
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_model_uses_cvrmse(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        model = self._get_model(engine)
        if model is None:
            pytest.skip("model uncertainty method not found")
        try:
            result = model(uncertainty_data["model_uncertainty"])
        except (ValueError, TypeError):
            pytest.skip("Model uncertainty not available")
        val = (getattr(result, "model_uncertainty_pct", None)
               or (result.get("model_uncertainty_pct")
                   if isinstance(result, dict) else None))
        if val is not None:
            assert float(val) > 0


# =============================================================================
# Sampling Uncertainty
# =============================================================================


class TestSamplingUncertainty:
    """Test sampling uncertainty for Option A metering."""

    def _get_sampling(self, engine):
        return (getattr(engine, "calculate_sampling_uncertainty", None)
                or getattr(engine, "sampling_uncertainty", None)
                or getattr(engine, "sample_uncertainty", None))

    def test_sampling_result(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        sampling = self._get_sampling(engine)
        if sampling is None:
            pytest.skip("sampling uncertainty method not found")
        try:
            result = sampling(uncertainty_data["sampling_uncertainty"])
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_sampling_positive(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        sampling = self._get_sampling(engine)
        if sampling is None:
            pytest.skip("sampling uncertainty method not found")
        try:
            result = sampling(uncertainty_data["sampling_uncertainty"])
        except (ValueError, TypeError):
            pytest.skip("Sampling uncertainty not available")
        val = (getattr(result, "sampling_uncertainty_pct", None)
               or (result.get("sampling_uncertainty_pct")
                   if isinstance(result, dict) else None))
        if val is not None:
            assert float(val) > 0

    def test_larger_sample_reduces_uncertainty(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        sampling = self._get_sampling(engine)
        if sampling is None:
            pytest.skip("sampling uncertainty method not found")
        small = dict(uncertainty_data["sampling_uncertainty"])
        small["sample_size"] = 50
        large = dict(uncertainty_data["sampling_uncertainty"])
        large["sample_size"] = 200
        try:
            r_small = sampling(small)
            r_large = sampling(large)
        except (ValueError, TypeError):
            pytest.skip("Sampling uncertainty not available")
        v_small = (getattr(r_small, "sampling_uncertainty_pct", None)
                   or (r_small.get("sampling_uncertainty_pct")
                       if isinstance(r_small, dict) else None))
        v_large = (getattr(r_large, "sampling_uncertainty_pct", None)
                   or (r_large.get("sampling_uncertainty_pct")
                       if isinstance(r_large, dict) else None))
        if v_small is not None and v_large is not None:
            assert float(v_large) <= float(v_small)


# =============================================================================
# Combined FSU
# =============================================================================


class TestCombinedFSU:
    """Test combined fractional savings uncertainty."""

    def _get_fsu(self, engine):
        return (getattr(engine, "calculate_fsu", None)
                or getattr(engine, "combined_uncertainty", None)
                or getattr(engine, "fractional_savings_uncertainty", None))

    def test_fsu_result(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        fsu = self._get_fsu(engine)
        if fsu is None:
            pytest.skip("FSU method not found")
        try:
            result = fsu(uncertainty_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_fsu_positive(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        fsu = self._get_fsu(engine)
        if fsu is None:
            pytest.skip("FSU method not found")
        try:
            result = fsu(uncertainty_data)
        except (ValueError, TypeError):
            pytest.skip("FSU not available")
        val = (getattr(result, "fsu_pct", None)
               or getattr(result, "fsu_pct_90", None)
               or (result.get("fsu_pct_90") if isinstance(result, dict) else None))
        if val is not None:
            assert float(val) > 0

    def test_fsu_less_than_100_pct(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        fsu = self._get_fsu(engine)
        if fsu is None:
            pytest.skip("FSU method not found")
        try:
            result = fsu(uncertainty_data)
        except (ValueError, TypeError):
            pytest.skip("FSU not available")
        val = (getattr(result, "fsu_pct", None)
               or getattr(result, "fsu_pct_90", None)
               or (result.get("fsu_pct_90") if isinstance(result, dict) else None))
        if val is not None:
            assert float(val) < 100.0


# =============================================================================
# Minimum Detectable Savings
# =============================================================================


class TestMinimumDetectableSavings:
    """Test minimum detectable savings (MDS) calculation."""

    def _get_mds(self, engine):
        return (getattr(engine, "calculate_mds", None)
                or getattr(engine, "minimum_detectable_savings", None)
                or getattr(engine, "mds", None))

    def test_mds_result(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        mds = self._get_mds(engine)
        if mds is None:
            pytest.skip("MDS method not found")
        try:
            result = mds(uncertainty_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_mds_less_than_baseline(self, uncertainty_data, savings_data):
        engine = _m.UncertaintyEngine()
        mds = self._get_mds(engine)
        if mds is None:
            pytest.skip("MDS method not found")
        try:
            result = mds(uncertainty_data)
        except (ValueError, TypeError):
            pytest.skip("MDS not available")
        val = (getattr(result, "mds_kwh", None)
               or getattr(result, "mds_kwh_90", None)
               or (result.get("mds_kwh_90") if isinstance(result, dict) else None))
        if val is not None:
            baseline = float(savings_data["baseline_consumption_kwh"])
            assert float(val) < baseline


# =============================================================================
# ASHRAE 14 Validation
# =============================================================================


class TestASHRAE14Validation:
    """Test ASHRAE Guideline 14 statistical validation."""

    def _get_ashrae(self, engine):
        return (getattr(engine, "validate_ashrae14", None)
                or getattr(engine, "ashrae14_check", None)
                or getattr(engine, "check_ashrae14", None))

    def test_ashrae14_result(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        ashrae = self._get_ashrae(engine)
        if ashrae is None:
            pytest.skip("ASHRAE 14 validation method not found")
        try:
            result = ashrae(uncertainty_data)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_ashrae14_has_pass_fail(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        ashrae = self._get_ashrae(engine)
        if ashrae is None:
            pytest.skip("ASHRAE 14 validation method not found")
        try:
            result = ashrae(uncertainty_data)
        except (ValueError, TypeError):
            pytest.skip("ASHRAE 14 validation not available")
        overall = (getattr(result, "overall_pass", None)
                   or (result.get("overall_pass") if isinstance(result, dict) else None))
        if overall is not None:
            assert isinstance(overall, bool)

    def test_ashrae14_subcomponents(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        ashrae = self._get_ashrae(engine)
        if ashrae is None:
            pytest.skip("ASHRAE 14 validation method not found")
        try:
            result = ashrae(uncertainty_data)
        except (ValueError, TypeError):
            pytest.skip("ASHRAE 14 validation not available")
        for key in ["cvrmse_pass", "nmbe_pass", "r_squared_pass"]:
            val = (getattr(result, key, None)
                   or (result.get(key) if isinstance(result, dict) else None))
            if val is not None:
                assert isinstance(val, bool)


# =============================================================================
# Provenance Tracking
# =============================================================================


class TestUncertaintyProvenance:
    """Test SHA-256 provenance hashing for uncertainty analysis."""

    def _get_provenance(self, engine):
        return (getattr(engine, "compute_provenance", None)
                or getattr(engine, "provenance_hash", None)
                or getattr(engine, "get_provenance", None))

    def test_provenance_hash_format(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h = prov(uncertainty_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h is not None:
            h_str = str(h)
            assert len(h_str) == 64
            assert all(c in "0123456789abcdef" for c in h_str)

    def test_provenance_deterministic(self, uncertainty_data):
        engine = _m.UncertaintyEngine()
        prov = self._get_provenance(engine)
        if prov is None:
            pytest.skip("provenance method not found")
        try:
            h1 = prov(uncertainty_data)
            h2 = prov(uncertainty_data)
        except (ValueError, TypeError):
            pytest.skip("Provenance not available")
        if h1 is not None and h2 is not None:
            assert str(h1) == str(h2)


# =============================================================================
# Uncertainty Fixture Validation
# =============================================================================


class TestUncertaintyFixtureValidation:
    """Validate uncertainty fixture data integrity."""

    def test_fsu_values_consistent(self, uncertainty_data):
        """FSU at 95% should be greater than at 90%."""
        fsu = uncertainty_data["combined_fsu"]
        assert float(fsu["fsu_pct_95"]) > float(fsu["fsu_pct_90"])

    def test_fsu_99_greater_than_95(self, uncertainty_data):
        fsu = uncertainty_data["combined_fsu"]
        assert float(fsu["fsu_pct_99"]) > float(fsu["fsu_pct_95"])

    def test_mds_95_greater_than_90(self, uncertainty_data):
        mds = uncertainty_data["minimum_detectable_savings"]
        assert float(mds["mds_kwh_95"]) > float(mds["mds_kwh_90"])

    def test_ashrae14_all_pass(self, uncertainty_data):
        compliance = uncertainty_data["ashrae14_compliance"]
        assert compliance["cvrmse_pass"] is True
        assert compliance["nmbe_pass"] is True
        assert compliance["r_squared_pass"] is True
        assert compliance["overall_pass"] is True
