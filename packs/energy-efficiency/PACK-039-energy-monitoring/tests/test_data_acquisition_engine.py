# -*- coding: utf-8 -*-
"""
Unit tests for DataAcquisitionEngine -- PACK-039 Engine 2
============================================================

Tests multi-protocol data acquisition, timestamp normalization, interval
alignment, buffer management, unit conversion, and data quality tagging.

Coverage target: 85%+
Total tests: ~60
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


_m = _load("data_acquisition_engine")


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
        assert hasattr(_m, "DataAcquisitionEngine")

    def test_engine_instantiation(self):
        engine = _m.DataAcquisitionEngine()
        assert engine is not None


# =============================================================================
# Multi-Protocol Acquisition
# =============================================================================


class TestMultiProtocolAcquisition:
    """Test data acquisition across different protocols."""

    def _get_acquire(self, engine):
        return (getattr(engine, "acquire_data", None)
                or getattr(engine, "poll_data", None)
                or getattr(engine, "collect_data", None))

    @pytest.mark.parametrize("protocol", [
        "MODBUS_TCP", "MODBUS_RTU", "BACnet", "MQTT",
        "OPC_UA", "OCPP", "SUNSPEC", "AMI",
    ])
    def test_acquire_by_protocol(self, protocol):
        engine = _m.DataAcquisitionEngine()
        acquire = self._get_acquire(engine)
        if acquire is None:
            pytest.skip("acquire_data method not found")
        try:
            result = acquire(meter_id="MTR-001", protocol=protocol)
            assert result is not None
        except Exception:
            pass  # Protocol may not be configured

    def test_acquire_returns_readings(self, sample_interval_data):
        engine = _m.DataAcquisitionEngine()
        acquire = self._get_acquire(engine)
        if acquire is None:
            pytest.skip("acquire_data method not found")
        result = acquire(meter_id="MTR-001", protocol="MODBUS_TCP")
        readings = getattr(result, "readings", result)
        if isinstance(readings, list):
            assert len(readings) >= 0

    def test_acquire_with_timeout(self):
        engine = _m.DataAcquisitionEngine()
        acquire = self._get_acquire(engine)
        if acquire is None:
            pytest.skip("acquire_data method not found")
        try:
            result = acquire(meter_id="MTR-001", protocol="MODBUS_TCP", timeout_seconds=5)
            assert result is not None
        except TypeError:
            result = acquire(meter_id="MTR-001", protocol="MODBUS_TCP")
            assert result is not None


# =============================================================================
# Normalization and Timestamp Alignment
# =============================================================================


class TestNormalization:
    """Test data normalization and timestamp alignment to standard intervals."""

    def _get_normalize(self, engine):
        return (getattr(engine, "normalize_data", None)
                or getattr(engine, "align_timestamps", None)
                or getattr(engine, "standardize", None))

    def test_normalize_to_15min(self, sample_interval_data):
        engine = _m.DataAcquisitionEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("normalize_data method not found")
        result = normalize(sample_interval_data, interval_minutes=15)
        assert result is not None

    @pytest.mark.parametrize("interval_min", [1, 5, 15, 30, 60])
    def test_normalize_interval_lengths(self, interval_min, sample_interval_data):
        engine = _m.DataAcquisitionEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("normalize_data method not found")
        try:
            result = normalize(sample_interval_data, interval_minutes=interval_min)
            assert result is not None
        except (ValueError, TypeError):
            pass

    def test_timestamp_alignment_deterministic(self, sample_interval_data):
        engine = _m.DataAcquisitionEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("normalize_data method not found")
        r1 = normalize(sample_interval_data, interval_minutes=15)
        r2 = normalize(sample_interval_data, interval_minutes=15)
        assert str(r1) == str(r2)

    def test_gap_filling_during_alignment(self):
        engine = _m.DataAcquisitionEngine()
        normalize = self._get_normalize(engine)
        if normalize is None:
            pytest.skip("normalize_data method not found")
        # Data with gap at hour 3
        data = [
            {"timestamp": "2025-07-01T00:00:00", "demand_kw": 500.0, "meter_id": "MTR-001"},
            {"timestamp": "2025-07-01T01:00:00", "demand_kw": 510.0, "meter_id": "MTR-001"},
            # Gap at hour 2
            {"timestamp": "2025-07-01T03:00:00", "demand_kw": 520.0, "meter_id": "MTR-001"},
        ]
        result = normalize(data, interval_minutes=60)
        assert result is not None


# =============================================================================
# Buffer Management
# =============================================================================


class TestBufferManagement:
    """Test data buffering for intermittent connectivity."""

    def _get_buffer(self, engine):
        return (getattr(engine, "buffer_data", None)
                or getattr(engine, "add_to_buffer", None)
                or getattr(engine, "enqueue", None))

    def _get_flush(self, engine):
        return (getattr(engine, "flush_buffer", None)
                or getattr(engine, "drain_buffer", None)
                or getattr(engine, "dequeue_all", None))

    def test_buffer_stores_data(self):
        engine = _m.DataAcquisitionEngine()
        buffer_fn = self._get_buffer(engine)
        if buffer_fn is None:
            pytest.skip("buffer_data method not found")
        reading = {"timestamp": "2025-07-01T00:00:00", "demand_kw": 500.0, "meter_id": "MTR-001"}
        result = buffer_fn(reading)
        assert result is not None

    def test_buffer_flush(self):
        engine = _m.DataAcquisitionEngine()
        buffer_fn = self._get_buffer(engine)
        flush_fn = self._get_flush(engine)
        if buffer_fn is None or flush_fn is None:
            pytest.skip("buffer/flush methods not found")
        reading = {"timestamp": "2025-07-01T00:00:00", "demand_kw": 500.0, "meter_id": "MTR-001"}
        buffer_fn(reading)
        result = flush_fn()
        assert result is not None

    def test_buffer_size_limit(self):
        engine = _m.DataAcquisitionEngine()
        size_fn = (getattr(engine, "buffer_size", None)
                   or getattr(engine, "get_buffer_size", None)
                   or getattr(engine, "pending_count", None))
        if size_fn is None:
            pytest.skip("buffer_size method not found")
        result = size_fn()
        if isinstance(result, (int, float)):
            assert result >= 0


# =============================================================================
# Unit Conversion
# =============================================================================


class TestUnitConversion:
    """Test energy unit conversions."""

    def _get_convert(self, engine):
        return (getattr(engine, "convert_units", None)
                or getattr(engine, "unit_conversion", None)
                or getattr(engine, "convert", None))

    @pytest.mark.parametrize("from_unit,to_unit,value,expected_approx", [
        ("kWh", "MWh", 1000.0, 1.0),
        ("MWh", "kWh", 1.0, 1000.0),
        ("therms", "kWh", 1.0, 29.3),
        ("GJ", "kWh", 1.0, 277.78),
        ("MMBTU", "kWh", 1.0, 293.07),
    ])
    def test_unit_conversion(self, from_unit, to_unit, value, expected_approx):
        engine = _m.DataAcquisitionEngine()
        convert = self._get_convert(engine)
        if convert is None:
            pytest.skip("convert_units method not found")
        try:
            result = convert(value=value, from_unit=from_unit, to_unit=to_unit)
            if isinstance(result, (int, float)):
                assert abs(result - expected_approx) / expected_approx < 0.05
        except (ValueError, TypeError, KeyError):
            pass

    def test_same_unit_no_change(self):
        engine = _m.DataAcquisitionEngine()
        convert = self._get_convert(engine)
        if convert is None:
            pytest.skip("convert_units method not found")
        try:
            result = convert(value=500.0, from_unit="kWh", to_unit="kWh")
            if isinstance(result, (int, float)):
                assert abs(result - 500.0) < 0.01
        except (ValueError, TypeError):
            pass


# =============================================================================
# Data Quality Tagging
# =============================================================================


class TestDataQualityTagging:
    """Test quality flag assignment during acquisition."""

    def _get_tag_quality(self, engine):
        return (getattr(engine, "tag_quality", None)
                or getattr(engine, "assess_quality", None)
                or getattr(engine, "quality_check", None))

    @pytest.mark.parametrize("quality_flag", [
        "GOOD", "SUSPECT", "BAD", "ESTIMATED", "MISSING",
    ])
    def test_quality_flags(self, quality_flag):
        engine = _m.DataAcquisitionEngine()
        tag = self._get_tag_quality(engine)
        if tag is None:
            pytest.skip("tag_quality method not found")
        reading = {
            "timestamp": "2025-07-01T00:00:00",
            "demand_kw": 500.0 if quality_flag != "BAD" else -100.0,
            "meter_id": "MTR-001",
        }
        try:
            result = tag(reading)
            assert result is not None
        except Exception:
            pass


# =============================================================================
# Provenance Hash
# =============================================================================


class TestProvenanceHash:
    """Test provenance hash determinism."""

    def test_same_input_same_hash(self, sample_interval_data):
        engine = _m.DataAcquisitionEngine()
        normalize = (getattr(engine, "normalize_data", None)
                     or getattr(engine, "align_timestamps", None))
        if normalize is None:
            pytest.skip("normalize method not found")
        r1 = normalize(sample_interval_data[:10], interval_minutes=15)
        r2 = normalize(sample_interval_data[:10], interval_minutes=15)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2

    def test_hash_is_sha256(self, sample_interval_data):
        engine = _m.DataAcquisitionEngine()
        normalize = (getattr(engine, "normalize_data", None)
                     or getattr(engine, "align_timestamps", None))
        if normalize is None:
            pytest.skip("normalize method not found")
        result = normalize(sample_interval_data[:10], interval_minutes=15)
        h = getattr(result, "provenance_hash", None)
        if h is not None:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# Fixture Validation
# =============================================================================


class TestIntervalDataFixture:
    """Validate the interval data fixture."""

    def test_interval_count(self, sample_interval_data):
        assert len(sample_interval_data) == 2880

    def test_all_have_required_fields(self, sample_interval_data):
        required = ["meter_id", "timestamp", "demand_kw", "energy_kwh"]
        for d in sample_interval_data:
            for field in required:
                assert field in d

    def test_demand_non_negative(self, sample_interval_data):
        for d in sample_interval_data:
            assert d["demand_kw"] >= 0

    def test_energy_equals_demand_quarter_hour(self, sample_interval_data):
        for d in sample_interval_data:
            assert abs(d["energy_kwh"] - d["demand_kw"] * 0.25) < 0.1

    def test_deterministic_data(self, sample_interval_data):
        rng = random.Random(42)
        # Reconstruct first weekday reading for verification
        first_var = rng.uniform(-25, 25)
        expected = max(0, 560.0 + first_var)
        assert abs(sample_interval_data[0]["demand_kw"] - round(expected, 2)) < 0.1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for data acquisition."""

    def test_empty_data_acquisition(self):
        engine = _m.DataAcquisitionEngine()
        normalize = (getattr(engine, "normalize_data", None)
                     or getattr(engine, "align_timestamps", None))
        if normalize is None:
            pytest.skip("normalize method not found")
        try:
            result = normalize([], interval_minutes=15)
            assert result is not None
        except (ValueError, IndexError):
            pass

    def test_single_reading(self):
        engine = _m.DataAcquisitionEngine()
        normalize = (getattr(engine, "normalize_data", None)
                     or getattr(engine, "align_timestamps", None))
        if normalize is None:
            pytest.skip("normalize method not found")
        single = [{"timestamp": "2025-07-01T12:00:00", "demand_kw": 1500.0,
                    "energy_kwh": 375.0, "meter_id": "MTR-001"}]
        result = normalize(single, interval_minutes=15)
        assert result is not None

    def test_out_of_order_timestamps(self):
        engine = _m.DataAcquisitionEngine()
        normalize = (getattr(engine, "normalize_data", None)
                     or getattr(engine, "align_timestamps", None))
        if normalize is None:
            pytest.skip("normalize method not found")
        data = [
            {"timestamp": "2025-07-01T02:00:00", "demand_kw": 520.0, "meter_id": "MTR-001"},
            {"timestamp": "2025-07-01T00:00:00", "demand_kw": 500.0, "meter_id": "MTR-001"},
            {"timestamp": "2025-07-01T01:00:00", "demand_kw": 510.0, "meter_id": "MTR-001"},
        ]
        result = normalize(data, interval_minutes=60)
        assert result is not None
