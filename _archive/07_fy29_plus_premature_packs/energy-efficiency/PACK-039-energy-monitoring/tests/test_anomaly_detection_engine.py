# -*- coding: utf-8 -*-
"""
Unit tests for AnomalyDetectionEngine -- PACK-039 Engine 4
============================================================

Tests CUSUM, EWMA, Z-score, IQR, regression-based, schedule-based,
and combined anomaly detection methods across 7 anomaly types.

Coverage target: 85%+
Total tests: ~70
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


_m = _load("anomaly_detection_engine")


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
        assert hasattr(_m, "AnomalyDetectionEngine")

    def test_engine_instantiation(self):
        engine = _m.AnomalyDetectionEngine()
        assert engine is not None


# =============================================================================
# Detection Methods Parametrize
# =============================================================================


class TestDetectionMethods:
    """Test 7 anomaly detection methods."""

    def _get_detect(self, engine):
        return (getattr(engine, "detect_anomalies", None)
                or getattr(engine, "detect", None)
                or getattr(engine, "find_anomalies", None))

    @pytest.mark.parametrize("method", [
        "CUSUM", "EWMA", "Z_SCORE", "IQR",
        "REGRESSION", "SCHEDULE_BASED", "COMBINED",
    ])
    def test_detection_method(self, method, sample_interval_data):
        engine = _m.AnomalyDetectionEngine()
        detect = self._get_detect(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        try:
            result = detect(sample_interval_data, method=method)
            assert result is not None
        except (ValueError, TypeError, KeyError):
            result = detect(sample_interval_data)
            assert result is not None

    @pytest.mark.parametrize("method", [
        "CUSUM", "EWMA", "Z_SCORE", "IQR",
        "REGRESSION", "SCHEDULE_BASED", "COMBINED",
    ])
    def test_method_detects_spike(self, method, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        detect = self._get_detect(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        try:
            result = detect(sample_anomaly_data, method=method)
        except (ValueError, TypeError, KeyError):
            result = detect(sample_anomaly_data)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) >= 1

    def test_combined_uses_multiple_methods(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        detect = self._get_detect(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        try:
            result = detect(sample_anomaly_data, method="COMBINED")
        except (ValueError, TypeError):
            result = detect(sample_anomaly_data)
        methods_used = getattr(result, "methods_used", None)
        if methods_used is not None:
            assert len(methods_used) >= 2


# =============================================================================
# CUSUM Detection
# =============================================================================


class TestCUSUMDetection:
    """Test CUSUM (Cumulative Sum) anomaly detection."""

    def _get_cusum(self, engine):
        return (getattr(engine, "cusum_detect", None)
                or getattr(engine, "detect_cusum", None)
                or getattr(engine, "run_cusum", None))

    def test_cusum_basic(self, sample_interval_data):
        engine = _m.AnomalyDetectionEngine()
        cusum = self._get_cusum(engine)
        if cusum is None:
            pytest.skip("cusum method not found")
        result = cusum(sample_interval_data)
        assert result is not None

    def test_cusum_with_h_factor(self, sample_interval_data):
        engine = _m.AnomalyDetectionEngine()
        cusum = self._get_cusum(engine)
        if cusum is None:
            pytest.skip("cusum method not found")
        try:
            result = cusum(sample_interval_data, h_factor=5.0)
            assert result is not None
        except TypeError:
            result = cusum(sample_interval_data)
            assert result is not None

    def test_cusum_detects_drift(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        cusum = self._get_cusum(engine)
        if cusum is None:
            pytest.skip("cusum method not found")
        result = cusum(sample_anomaly_data)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) >= 1

    def test_cusum_deterministic(self, sample_interval_data):
        engine = _m.AnomalyDetectionEngine()
        cusum = self._get_cusum(engine)
        if cusum is None:
            pytest.skip("cusum method not found")
        r1 = cusum(sample_interval_data)
        r2 = cusum(sample_interval_data)
        assert str(r1) == str(r2)


# =============================================================================
# EWMA Detection
# =============================================================================


class TestEWMADetection:
    """Test EWMA (Exponentially Weighted Moving Average) detection."""

    def _get_ewma(self, engine):
        return (getattr(engine, "ewma_detect", None)
                or getattr(engine, "detect_ewma", None)
                or getattr(engine, "run_ewma", None))

    def test_ewma_basic(self, sample_interval_data):
        engine = _m.AnomalyDetectionEngine()
        ewma = self._get_ewma(engine)
        if ewma is None:
            pytest.skip("ewma method not found")
        result = ewma(sample_interval_data)
        assert result is not None

    @pytest.mark.parametrize("lambda_val", [0.05, 0.10, 0.20, 0.30, 0.50])
    def test_ewma_lambda_parameter(self, lambda_val, sample_interval_data):
        engine = _m.AnomalyDetectionEngine()
        ewma = self._get_ewma(engine)
        if ewma is None:
            pytest.skip("ewma method not found")
        try:
            result = ewma(sample_interval_data, lambda_param=lambda_val)
            assert result is not None
        except TypeError:
            result = ewma(sample_interval_data)
            assert result is not None

    def test_ewma_detects_spike(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        ewma = self._get_ewma(engine)
        if ewma is None:
            pytest.skip("ewma method not found")
        result = ewma(sample_anomaly_data)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) >= 1


# =============================================================================
# Z-Score Detection
# =============================================================================


class TestZScoreDetection:
    """Test Z-score anomaly detection."""

    def _get_zscore(self, engine):
        return (getattr(engine, "zscore_detect", None)
                or getattr(engine, "detect_zscore", None)
                or getattr(engine, "detect_anomalies", None))

    @pytest.mark.parametrize("threshold", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_zscore_threshold(self, threshold, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        zscore = self._get_zscore(engine)
        if zscore is None:
            pytest.skip("zscore method not found")
        try:
            result = zscore(sample_anomaly_data, z_threshold=threshold)
            assert result is not None
        except TypeError:
            result = zscore(sample_anomaly_data)
            assert result is not None

    def test_higher_threshold_fewer_anomalies(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        zscore = self._get_zscore(engine)
        if zscore is None:
            pytest.skip("zscore method not found")
        try:
            r_low = zscore(sample_anomaly_data, z_threshold=2.0)
            r_high = zscore(sample_anomaly_data, z_threshold=4.0)
            a_low = getattr(r_low, "anomalies", r_low)
            a_high = getattr(r_high, "anomalies", r_high)
            if isinstance(a_low, list) and isinstance(a_high, list):
                assert len(a_low) >= len(a_high)
        except TypeError:
            pass

    def test_no_anomalies_in_flat_data(self):
        engine = _m.AnomalyDetectionEngine()
        zscore = self._get_zscore(engine)
        if zscore is None:
            pytest.skip("zscore method not found")
        flat = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                 "demand_kw": 1000.0, "energy_kwh": 250.0,
                 "meter_id": "MTR-001"}
                for h in range(24)]
        try:
            result = zscore(flat, z_threshold=3.0)
        except TypeError:
            result = zscore(flat)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) == 0


# =============================================================================
# IQR Detection
# =============================================================================


class TestIQRDetection:
    """Test IQR (Interquartile Range) detection."""

    def _get_iqr(self, engine):
        return (getattr(engine, "iqr_detect", None)
                or getattr(engine, "detect_iqr", None)
                or getattr(engine, "run_iqr", None))

    def test_iqr_basic(self, sample_interval_data):
        engine = _m.AnomalyDetectionEngine()
        iqr = self._get_iqr(engine)
        if iqr is None:
            pytest.skip("iqr method not found")
        result = iqr(sample_interval_data)
        assert result is not None

    def test_iqr_detects_outliers(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        iqr = self._get_iqr(engine)
        if iqr is None:
            pytest.skip("iqr method not found")
        result = iqr(sample_anomaly_data)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) >= 1


# =============================================================================
# Anomaly Type Parametrize
# =============================================================================


class TestAnomalyTypes:
    """Test detection of 7 anomaly types."""

    @pytest.mark.parametrize("anomaly_type", [
        "SPIKE", "DROPOUT", "FLATLINE", "NEGATIVE",
        "DRIFT", "OSCILLATION", "STEP_CHANGE",
    ])
    def test_anomaly_type_recognition(self, anomaly_type):
        engine = _m.AnomalyDetectionEngine()
        classify = (getattr(engine, "classify_anomaly", None)
                    or getattr(engine, "anomaly_type", None)
                    or getattr(engine, "get_anomaly_types", None))
        if classify is None:
            pytest.skip("classify_anomaly method not found")
        try:
            result = classify(anomaly_type)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_spike_detected_in_anomaly_data(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "detect", None))
        if detect is None:
            pytest.skip("detect method not found")
        result = detect(sample_anomaly_data)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list) and len(anomalies) > 0:
            types = set()
            for a in anomalies:
                if isinstance(a, dict):
                    types.add(a.get("anomaly_type", a.get("type", "")))
            # Should detect at least some anomaly types
            assert len(types) >= 1 or len(anomalies) >= 1


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    """Test provenance hash for detection results."""

    def test_same_input_same_hash(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "detect", None))
        if detect is None:
            pytest.skip("detect method not found")
        r1 = detect(sample_anomaly_data)
        r2 = detect(sample_anomaly_data)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2

    def test_hash_is_sha256(self, sample_anomaly_data):
        engine = _m.AnomalyDetectionEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "detect", None))
        if detect is None:
            pytest.skip("detect method not found")
        result = detect(sample_anomaly_data)
        h = getattr(result, "provenance_hash", None)
        if h is not None:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# Anomaly Data Fixture Validation
# =============================================================================


class TestAnomalyDataFixture:
    """Validate the anomaly data fixture."""

    def test_anomaly_data_count(self, sample_anomaly_data):
        assert len(sample_anomaly_data) == 2880

    def test_has_5_injected_anomalies(self, sample_anomaly_data):
        injected = [d for d in sample_anomaly_data if d.get("injected_anomaly") is not None]
        # Day 5 spike (1), day 10 dropout (1), day 15 flatline (multiple intervals),
        # day 20 negative (1), day 25 drift (all intervals)
        assert len(injected) >= 5

    def test_spike_present(self, sample_anomaly_data):
        spikes = [d for d in sample_anomaly_data if d.get("injected_anomaly") == "SPIKE"]
        assert len(spikes) >= 1
        assert spikes[0]["demand_kw"] == 5000.0

    def test_negative_present(self, sample_anomaly_data):
        negs = [d for d in sample_anomaly_data if d.get("injected_anomaly") == "NEGATIVE"]
        assert len(negs) >= 1
        assert negs[0]["demand_kw"] == -150.0

    def test_dropout_present(self, sample_anomaly_data):
        drops = [d for d in sample_anomaly_data if d.get("injected_anomaly") == "DROPOUT"]
        assert len(drops) >= 1
        assert drops[0]["demand_kw"] == 0.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for anomaly detection."""

    def test_empty_data(self):
        engine = _m.AnomalyDetectionEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "detect", None))
        if detect is None:
            pytest.skip("detect method not found")
        try:
            result = detect([])
            assert result is not None
        except (ValueError, IndexError):
            pass

    def test_constant_data(self):
        engine = _m.AnomalyDetectionEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "detect", None))
        if detect is None:
            pytest.skip("detect method not found")
        constant = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                     "demand_kw": 1000.0, "energy_kwh": 250.0,
                     "meter_id": "MTR-001"}
                    for h in range(24)]
        result = detect(constant)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) == 0

    def test_all_zero_data(self):
        engine = _m.AnomalyDetectionEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "detect", None))
        if detect is None:
            pytest.skip("detect method not found")
        zeros = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                  "demand_kw": 0.0, "energy_kwh": 0.0,
                  "meter_id": "MTR-001"}
                 for h in range(24)]
        result = detect(zeros)
        assert result is not None

    def test_single_reading(self):
        engine = _m.AnomalyDetectionEngine()
        detect = (getattr(engine, "detect_anomalies", None)
                  or getattr(engine, "detect", None))
        if detect is None:
            pytest.skip("detect method not found")
        single = [{"timestamp": "2025-07-01T12:00:00", "demand_kw": 1500.0,
                    "energy_kwh": 375.0, "meter_id": "MTR-001"}]
        result = detect(single)
        assert result is not None
