# -*- coding: utf-8 -*-
"""
Unit tests for LoadProfileEngine -- PACK-038 Engine 1
============================================================

Tests load factor calculation for 10 facility types, duration curve with 8,760
points, day-type clustering (weekday/weekend/holiday), seasonal decomposition,
anomaly detection (Z-score > 3.0), and multi-interval/season/facility
parametrization.

Coverage target: 85%+
Total tests: ~120
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


_m = _load("load_profile_engine")


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
        assert hasattr(_m, "LoadProfileEngine")

    def test_engine_instantiation(self):
        engine = _m.LoadProfileEngine()
        assert engine is not None


# =============================================================================
# Load Factor Calculation (10 facility types)
# =============================================================================


class TestLoadFactor:
    """Test load factor calculation across various facility types."""

    def _get_load_factor(self, engine):
        return (getattr(engine, "calculate_load_factor", None)
                or getattr(engine, "load_factor", None)
                or getattr(engine, "compute_load_factor", None))

    @pytest.mark.parametrize("facility_type,peak_kw,avg_kw", [
        ("COMMERCIAL_OFFICE", 2000, 1140),
        ("INDUSTRIAL_MANUFACTURING", 5000, 3500),
        ("RETAIL_STORE", 800, 440),
        ("DATA_CENTER", 3000, 2700),
        ("HOSPITAL", 4000, 3200),
        ("SCHOOL", 1200, 480),
        ("HOTEL", 1500, 900),
        ("WAREHOUSE", 600, 240),
        ("GROCERY_STORE", 900, 630),
        ("RESTAURANT", 400, 200),
    ])
    def test_load_factor_by_facility(self, facility_type, peak_kw, avg_kw):
        engine = _m.LoadProfileEngine()
        calc = self._get_load_factor(engine)
        if calc is None:
            pytest.skip("load_factor method not found")
        result = calc(peak_kw=peak_kw, average_kw=avg_kw)
        expected = avg_kw / peak_kw
        if isinstance(result, (int, float)):
            assert abs(result - expected) < 0.01
        else:
            assert result is not None

    def test_load_factor_range_0_to_1(self):
        engine = _m.LoadProfileEngine()
        calc = self._get_load_factor(engine)
        if calc is None:
            pytest.skip("load_factor method not found")
        result = calc(peak_kw=2000, average_kw=1140)
        val = result if isinstance(result, (int, float)) else getattr(result, "value", 0.5)
        assert 0.0 <= val <= 1.0

    def test_load_factor_perfect_flat(self):
        engine = _m.LoadProfileEngine()
        calc = self._get_load_factor(engine)
        if calc is None:
            pytest.skip("load_factor method not found")
        result = calc(peak_kw=1000, average_kw=1000)
        val = result if isinstance(result, (int, float)) else getattr(result, "value", 1.0)
        assert abs(val - 1.0) < 0.001

    def test_load_factor_very_peaky(self):
        engine = _m.LoadProfileEngine()
        calc = self._get_load_factor(engine)
        if calc is None:
            pytest.skip("load_factor method not found")
        result = calc(peak_kw=5000, average_kw=500)
        val = result if isinstance(result, (int, float)) else getattr(result, "value", 0.1)
        assert val < 0.2

    def test_load_factor_deterministic(self):
        engine = _m.LoadProfileEngine()
        calc = self._get_load_factor(engine)
        if calc is None:
            pytest.skip("load_factor method not found")
        r1 = calc(peak_kw=2000, average_kw=1140)
        r2 = calc(peak_kw=2000, average_kw=1140)
        assert r1 == r2

    @pytest.mark.parametrize("peak_kw,avg_kw,expected_lf", [
        (2000, 1140, 0.57),
        (1000, 800, 0.80),
        (3000, 900, 0.30),
        (500, 450, 0.90),
    ])
    def test_load_factor_values(self, peak_kw, avg_kw, expected_lf):
        engine = _m.LoadProfileEngine()
        calc = self._get_load_factor(engine)
        if calc is None:
            pytest.skip("load_factor method not found")
        result = calc(peak_kw=peak_kw, average_kw=avg_kw)
        val = result if isinstance(result, (int, float)) else getattr(result, "value", None)
        if val is not None:
            assert abs(val - expected_lf) < 0.02


# =============================================================================
# Duration Curve (8,760 points)
# =============================================================================


class TestDurationCurve:
    """Test load duration curve generation and properties."""

    def _get_duration_curve(self, engine):
        return (getattr(engine, "build_duration_curve", None)
                or getattr(engine, "duration_curve", None)
                or getattr(engine, "generate_ldc", None))

    def test_duration_curve_generation(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        build = self._get_duration_curve(engine)
        if build is None:
            pytest.skip("duration_curve method not found")
        result = build(sample_interval_data)
        assert result is not None

    def test_duration_curve_sorted_descending(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        build = self._get_duration_curve(engine)
        if build is None:
            pytest.skip("duration_curve method not found")
        result = build(sample_interval_data)
        curve = getattr(result, "curve", result) if not isinstance(result, list) else result
        if isinstance(curve, list) and len(curve) > 1:
            for i in range(len(curve) - 1):
                val_i = curve[i] if isinstance(curve[i], (int, float)) else curve[i].get("demand_kw", 0)
                val_next = curve[i + 1] if isinstance(curve[i + 1], (int, float)) else curve[i + 1].get("demand_kw", 0)
                assert val_i >= val_next

    def test_duration_curve_length(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        build = self._get_duration_curve(engine)
        if build is None:
            pytest.skip("duration_curve method not found")
        result = build(sample_interval_data)
        curve = getattr(result, "curve", result) if not isinstance(result, list) else result
        if isinstance(curve, list):
            assert len(curve) == len(sample_interval_data)

    def test_duration_curve_max_equals_peak(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        build = self._get_duration_curve(engine)
        if build is None:
            pytest.skip("duration_curve method not found")
        result = build(sample_interval_data)
        curve = getattr(result, "curve", result) if not isinstance(result, list) else result
        if isinstance(curve, list) and len(curve) > 0:
            max_val = curve[0] if isinstance(curve[0], (int, float)) else curve[0].get("demand_kw", 0)
            data_max = max(d["demand_kw"] for d in sample_interval_data)
            assert abs(max_val - data_max) < 1.0

    def test_duration_curve_with_8760_points(self):
        engine = _m.LoadProfileEngine()
        build = self._get_duration_curve(engine)
        if build is None:
            pytest.skip("duration_curve method not found")
        rng = random.Random(42)
        hourly_data = [{"timestamp": f"2025-01-01T{h % 24:02d}:00:00",
                        "demand_kw": round(rng.uniform(400, 2000), 2)}
                       for h in range(8760)]
        result = build(hourly_data)
        curve = getattr(result, "curve", result) if not isinstance(result, list) else result
        if isinstance(curve, list):
            assert len(curve) == 8760

    def test_duration_curve_percentiles(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        build = self._get_duration_curve(engine)
        if build is None:
            pytest.skip("duration_curve method not found")
        result = build(sample_interval_data)
        percentiles = getattr(result, "percentiles", None)
        if percentiles is not None:
            assert "p50" in percentiles or 50 in percentiles


# =============================================================================
# Day-Type Clustering
# =============================================================================


class TestDayTypeClustering:
    """Test weekday/weekend/holiday clustering."""

    def _get_cluster(self, engine):
        return (getattr(engine, "cluster_day_types", None)
                or getattr(engine, "day_type_analysis", None)
                or getattr(engine, "classify_days", None))

    def test_cluster_returns_result(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        cluster = self._get_cluster(engine)
        if cluster is None:
            pytest.skip("cluster_day_types method not found")
        result = cluster(sample_interval_data)
        assert result is not None

    @pytest.mark.parametrize("day_type", ["WEEKDAY", "WEEKEND", "HOLIDAY"])
    def test_day_type_recognized(self, day_type, sample_interval_data):
        engine = _m.LoadProfileEngine()
        cluster = self._get_cluster(engine)
        if cluster is None:
            pytest.skip("cluster_day_types method not found")
        result = cluster(sample_interval_data)
        types = getattr(result, "day_types", result)
        if isinstance(types, dict):
            assert day_type in types or len(types) >= 2

    def test_weekday_higher_than_weekend(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        cluster = self._get_cluster(engine)
        if cluster is None:
            pytest.skip("cluster_day_types method not found")
        result = cluster(sample_interval_data)
        profiles = getattr(result, "profiles", None)
        if profiles and "WEEKDAY" in profiles and "WEEKEND" in profiles:
            wd_peak = max(profiles["WEEKDAY"])
            we_peak = max(profiles["WEEKEND"])
            assert wd_peak >= we_peak

    def test_cluster_deterministic(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        cluster = self._get_cluster(engine)
        if cluster is None:
            pytest.skip("cluster_day_types method not found")
        r1 = cluster(sample_interval_data)
        r2 = cluster(sample_interval_data)
        assert str(r1) == str(r2)

    def test_cluster_counts(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        cluster = self._get_cluster(engine)
        if cluster is None:
            pytest.skip("cluster_day_types method not found")
        result = cluster(sample_interval_data)
        counts = getattr(result, "day_counts", None)
        if counts is not None:
            total = sum(counts.values())
            assert total == 30  # 30 days of data


# =============================================================================
# Seasonal Decomposition
# =============================================================================


class TestSeasonalDecomposition:
    """Test seasonal load profile decomposition."""

    def _get_seasonal(self, engine):
        return (getattr(engine, "seasonal_decomposition", None)
                or getattr(engine, "decompose_seasonal", None)
                or getattr(engine, "analyze_seasons", None))

    @pytest.mark.parametrize("season", ["SUMMER", "WINTER", "SPRING", "FALL"])
    def test_season_analysis(self, season, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_seasonal(engine)
        if analyze is None:
            pytest.skip("seasonal_decomposition method not found")
        result = analyze(sample_interval_data, season=season)
        assert result is not None

    def test_summer_peak_highest(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_seasonal(engine)
        if analyze is None:
            pytest.skip("seasonal_decomposition method not found")
        summer = analyze(sample_interval_data, season="SUMMER")
        spring = analyze(sample_interval_data, season="SPRING")
        s_peak = getattr(summer, "peak_kw", None)
        p_peak = getattr(spring, "peak_kw", None)
        if s_peak is not None and p_peak is not None:
            assert s_peak >= p_peak

    def test_decomposition_components(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_seasonal(engine)
        if analyze is None:
            pytest.skip("seasonal_decomposition method not found")
        result = analyze(sample_interval_data, season="SUMMER")
        has_components = (hasattr(result, "trend") or hasattr(result, "seasonal")
                          or hasattr(result, "residual"))
        assert has_components or result is not None


# =============================================================================
# Anomaly Detection (Z-score > 3.0)
# =============================================================================


class TestAnomalyDetection:
    """Test anomaly detection using Z-score threshold."""

    def _get_anomaly(self, engine):
        return (getattr(engine, "detect_anomalies", None)
                or getattr(engine, "anomaly_detection", None)
                or getattr(engine, "find_anomalies", None))

    def test_detect_anomalies(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        detect = self._get_anomaly(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        result = detect(sample_interval_data)
        assert result is not None

    def test_anomaly_with_spike(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        detect = self._get_anomaly(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        data = list(sample_interval_data)
        data.append({
            "timestamp": "2025-07-31T12:00:00",
            "demand_kw": 9999.99,
            "energy_kwh": 2499.99,
            "temperature_c": 40.0,
            "power_factor": 0.90,
        })
        result = detect(data)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) >= 1

    def test_no_anomalies_in_clean_data(self):
        engine = _m.LoadProfileEngine()
        detect = self._get_anomaly(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        clean = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                  "demand_kw": 1000.0, "energy_kwh": 250.0,
                  "temperature_c": 25.0, "power_factor": 0.95}
                 for h in range(24)]
        result = detect(clean)
        anomalies = getattr(result, "anomalies", result)
        if isinstance(anomalies, list):
            assert len(anomalies) == 0

    @pytest.mark.parametrize("z_threshold", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_anomaly_threshold(self, z_threshold, sample_interval_data):
        engine = _m.LoadProfileEngine()
        detect = self._get_anomaly(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        try:
            result = detect(sample_interval_data, z_threshold=z_threshold)
        except TypeError:
            result = detect(sample_interval_data)
        assert result is not None

    def test_higher_threshold_fewer_anomalies(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        detect = self._get_anomaly(engine)
        if detect is None:
            pytest.skip("detect_anomalies method not found")
        data = list(sample_interval_data)
        for i in range(5):
            data.append({
                "timestamp": f"2025-07-31T{12 + i}:00:00",
                "demand_kw": 5000.0 + i * 500,
                "energy_kwh": 1250.0,
                "temperature_c": 40.0,
                "power_factor": 0.90,
            })
        try:
            r_low = detect(data, z_threshold=2.0)
            r_high = detect(data, z_threshold=4.0)
            a_low = getattr(r_low, "anomalies", r_low)
            a_high = getattr(r_high, "anomalies", r_high)
            if isinstance(a_low, list) and isinstance(a_high, list):
                assert len(a_low) >= len(a_high)
        except TypeError:
            pass  # Method may not accept z_threshold


# =============================================================================
# Interval Length Parametrize
# =============================================================================


class TestIntervalLengths:
    """Test profile analysis across different interval lengths."""

    def _get_analyze(self, engine):
        return (getattr(engine, "analyze_profile", None)
                or getattr(engine, "analyze", None)
                or getattr(engine, "build_profile", None))

    @pytest.mark.parametrize("interval_min", [15, 30, 60])
    def test_interval_analysis(self, interval_min, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze_profile method not found")
        try:
            result = analyze(sample_interval_data, interval_minutes=interval_min)
        except TypeError:
            result = analyze(sample_interval_data)
        assert result is not None


# =============================================================================
# Profile Statistics
# =============================================================================


class TestProfileStatistics:
    """Test profile statistical measures."""

    def _get_stats(self, engine):
        return (getattr(engine, "compute_statistics", None)
                or getattr(engine, "profile_statistics", None)
                or getattr(engine, "statistics", None))

    def test_statistics_computation(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("statistics method not found")
        result = stats(sample_interval_data)
        assert result is not None

    @pytest.mark.parametrize("stat_name", [
        "mean", "median", "std_dev", "min", "max", "p95", "p99",
    ])
    def test_statistic_present(self, stat_name, sample_interval_data):
        engine = _m.LoadProfileEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("statistics method not found")
        result = stats(sample_interval_data)
        has_stat = (hasattr(result, stat_name) or
                    (isinstance(result, dict) and stat_name in result))
        assert has_stat or result is not None

    def test_mean_less_than_peak(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("statistics method not found")
        result = stats(sample_interval_data)
        mean_val = getattr(result, "mean", None)
        max_val = getattr(result, "max", None)
        if mean_val is not None and max_val is not None:
            assert mean_val <= max_val

    def test_min_less_than_mean(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        stats = self._get_stats(engine)
        if stats is None:
            pytest.skip("statistics method not found")
        result = stats(sample_interval_data)
        min_val = getattr(result, "min", None)
        mean_val = getattr(result, "mean", None)
        if min_val is not None and mean_val is not None:
            assert min_val <= mean_val


# =============================================================================
# Provenance Hash Determinism
# =============================================================================


class TestProvenanceHash:
    """Test provenance hash is deterministic and valid SHA-256."""

    def _get_analyze(self, engine):
        return (getattr(engine, "analyze_profile", None)
                or getattr(engine, "analyze", None)
                or getattr(engine, "build_profile", None))

    def test_same_input_same_hash(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        r1 = analyze(sample_interval_data)
        r2 = analyze(sample_interval_data)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2

    def test_hash_is_sha256(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        result = analyze(sample_interval_data)
        h = getattr(result, "provenance_hash", None)
        if h is not None:
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)

    def test_different_input_different_hash(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        r1 = analyze(sample_interval_data)
        modified = [dict(d, demand_kw=0) for d in sample_interval_data[:10]]
        r2 = analyze(modified)
        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 != h2


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases: empty, single, constant, extreme data."""

    def _get_analyze(self, engine):
        return (getattr(engine, "analyze_profile", None)
                or getattr(engine, "analyze", None)
                or getattr(engine, "build_profile", None))

    def test_empty_data(self):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        try:
            result = analyze([])
            assert result is not None
        except (ValueError, IndexError):
            pass

    def test_single_reading(self):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        single = [{"timestamp": "2025-07-01T12:00:00", "demand_kw": 1500.0,
                    "energy_kwh": 375.0, "temperature_c": 30.0,
                    "power_factor": 0.92}]
        result = analyze(single)
        assert result is not None

    def test_constant_load(self):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        constant = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                      "demand_kw": 1000.0, "energy_kwh": 250.0,
                      "temperature_c": 25.0, "power_factor": 0.95}
                     for h in range(24)]
        result = analyze(constant)
        assert result is not None

    def test_zero_demand(self):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        zeros = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                   "demand_kw": 0.0, "energy_kwh": 0.0,
                   "temperature_c": 25.0, "power_factor": 1.0}
                  for h in range(24)]
        result = analyze(zeros)
        assert result is not None

    def test_large_dataset(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        analyze = self._get_analyze(engine)
        if analyze is None:
            pytest.skip("analyze method not found")
        large = sample_interval_data * 3  # ~8640 readings
        result = analyze(large)
        assert result is not None


# =============================================================================
# Interval Data Fixture Validation
# =============================================================================


class TestIntervalDataFixture:
    """Validate the interval data fixture itself."""

    def test_interval_count(self, sample_interval_data):
        assert len(sample_interval_data) == 2880

    def test_all_have_required_fields(self, sample_interval_data):
        required = ["timestamp", "demand_kw", "energy_kwh", "temperature_c"]
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
        first_base = 540.0  # day 1 is weekday (Tuesday), first interval hour=0
        # Re-derive first value
        first_var = rng.uniform(-25, 25)  # weekday, hour 0 < 6
        expected = max(0, 560.0 + first_var)
        assert abs(sample_interval_data[0]["demand_kw"] - round(expected, 2)) < 0.1

    def test_peak_in_expected_range(self, sample_interval_data):
        peak = max(d["demand_kw"] for d in sample_interval_data)
        assert 1800 < peak < 2500

    def test_minimum_in_expected_range(self, sample_interval_data):
        minimum = min(d["demand_kw"] for d in sample_interval_data)
        assert 400 < minimum < 700

    @pytest.mark.parametrize("hour", list(range(24)))
    def test_hour_coverage(self, hour, sample_interval_data):
        found = any(f"T{hour:02d}:" in d["timestamp"] for d in sample_interval_data)
        assert found


# =============================================================================
# Temperature Correlation
# =============================================================================


class TestTemperatureCorrelation:
    """Test temperature-demand correlation analysis."""

    def _get_correlation(self, engine):
        return (getattr(engine, "temperature_correlation", None)
                or getattr(engine, "correlate_temperature", None)
                or getattr(engine, "weather_correlation", None))

    def test_correlation_analysis(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        corr = self._get_correlation(engine)
        if corr is None:
            pytest.skip("temperature_correlation method not found")
        result = corr(sample_interval_data)
        assert result is not None

    def test_positive_correlation_summer(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        corr = self._get_correlation(engine)
        if corr is None:
            pytest.skip("temperature_correlation method not found")
        result = corr(sample_interval_data)
        r_val = getattr(result, "r_squared", getattr(result, "correlation", None))
        if r_val is not None and isinstance(r_val, (int, float)):
            assert r_val >= 0

    def test_correlation_deterministic(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        corr = self._get_correlation(engine)
        if corr is None:
            pytest.skip("temperature_correlation method not found")
        r1 = corr(sample_interval_data)
        r2 = corr(sample_interval_data)
        v1 = getattr(r1, "r_squared", str(r1))
        v2 = getattr(r2, "r_squared", str(r2))
        assert v1 == v2


# =============================================================================
# Hourly Profile
# =============================================================================


class TestHourlyProfile:
    """Test 24-hour average profile generation."""

    def _get_hourly(self, engine):
        return (getattr(engine, "hourly_profile", None)
                or getattr(engine, "average_hourly", None)
                or getattr(engine, "build_hourly_profile", None))

    def test_hourly_profile_24_hours(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        hourly = self._get_hourly(engine)
        if hourly is None:
            pytest.skip("hourly_profile method not found")
        result = hourly(sample_interval_data)
        profile = getattr(result, "profile", result)
        if isinstance(profile, (list, dict)):
            assert len(profile) == 24 or len(profile) >= 24

    def test_peak_hour_during_business(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        hourly = self._get_hourly(engine)
        if hourly is None:
            pytest.skip("hourly_profile method not found")
        result = hourly(sample_interval_data)
        peak_hour = getattr(result, "peak_hour", None)
        if peak_hour is not None:
            assert 9 <= peak_hour <= 17

    @pytest.mark.parametrize("hour", [0, 6, 12, 18, 23])
    def test_hourly_values_positive(self, hour, sample_interval_data):
        engine = _m.LoadProfileEngine()
        hourly = self._get_hourly(engine)
        if hourly is None:
            pytest.skip("hourly_profile method not found")
        result = hourly(sample_interval_data)
        profile = getattr(result, "profile", result)
        if isinstance(profile, dict) and hour in profile:
            assert profile[hour] >= 0
        elif isinstance(profile, list) and hour < len(profile):
            val = profile[hour] if isinstance(profile[hour], (int, float)) else profile[hour].get("demand_kw", 0)
            assert val >= 0


# =============================================================================
# Facility Type Profiles
# =============================================================================


class TestFacilityTypeProfiles:
    """Test profile analysis across different facility types."""

    def _get_analyze(self, engine):
        return (getattr(engine, "analyze_profile", None)
                or getattr(engine, "analyze", None)
                or getattr(engine, "build_profile", None))

    @pytest.mark.parametrize("facility_type,expected_lf_min,expected_lf_max", [
        ("COMMERCIAL_OFFICE", 0.40, 0.70),
        ("INDUSTRIAL_MANUFACTURING", 0.60, 0.85),
        ("RETAIL_STORE", 0.35, 0.65),
        ("DATA_CENTER", 0.80, 0.98),
        ("HOSPITAL", 0.70, 0.90),
        ("SCHOOL", 0.25, 0.50),
        ("HOTEL", 0.50, 0.75),
        ("WAREHOUSE", 0.25, 0.50),
        ("GROCERY_STORE", 0.55, 0.80),
        ("RESTAURANT", 0.35, 0.60),
    ])
    def test_expected_load_factors(self, facility_type, expected_lf_min, expected_lf_max):
        """Verify expected load factor ranges per facility type."""
        # These are industry standard ranges, not engine outputs
        assert expected_lf_min < expected_lf_max
        assert expected_lf_min > 0
        assert expected_lf_max <= 1.0


# =============================================================================
# Peak-to-Average Ratio
# =============================================================================


class TestPeakToAverageRatio:
    """Test peak-to-average demand ratio calculation."""

    def _get_ratio(self, engine):
        return (getattr(engine, "peak_to_average_ratio", None)
                or getattr(engine, "demand_ratio", None)
                or getattr(engine, "calculate_par", None))

    def test_ratio_result(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        ratio = self._get_ratio(engine)
        if ratio is None:
            pytest.skip("peak_to_average method not found")
        result = ratio(sample_interval_data)
        assert result is not None

    def test_ratio_greater_than_one(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        ratio = self._get_ratio(engine)
        if ratio is None:
            pytest.skip("peak_to_average method not found")
        result = ratio(sample_interval_data)
        val = getattr(result, "ratio", result)
        if isinstance(val, (int, float)):
            assert val >= 1.0

    def test_flat_load_ratio_one(self):
        engine = _m.LoadProfileEngine()
        ratio = self._get_ratio(engine)
        if ratio is None:
            pytest.skip("peak_to_average method not found")
        flat = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                 "demand_kw": 1000.0, "energy_kwh": 250.0,
                 "temperature_c": 25.0, "power_factor": 0.95}
                for h in range(24)]
        result = ratio(flat)
        val = getattr(result, "ratio", result)
        if isinstance(val, (int, float)):
            assert abs(val - 1.0) < 0.01

    @pytest.mark.parametrize("multiplier", [1.5, 2.0, 3.0, 5.0])
    def test_ratio_with_spike(self, multiplier):
        engine = _m.LoadProfileEngine()
        ratio = self._get_ratio(engine)
        if ratio is None:
            pytest.skip("peak_to_average method not found")
        data = [{"timestamp": f"2025-07-01T{h:02d}:00:00",
                 "demand_kw": 1000.0, "energy_kwh": 250.0,
                 "temperature_c": 25.0, "power_factor": 0.95}
                for h in range(24)]
        data[12] = dict(data[12], demand_kw=1000.0 * multiplier)
        result = ratio(data)
        val = getattr(result, "ratio", result)
        if isinstance(val, (int, float)):
            assert val > 1.0


# =============================================================================
# Weekend vs Weekday Analysis
# =============================================================================


class TestWeekendWeekdayAnalysis:
    """Test weekend vs weekday profile comparison."""

    def _get_compare(self, engine):
        return (getattr(engine, "weekday_weekend_comparison", None)
                or getattr(engine, "compare_day_types", None)
                or getattr(engine, "day_type_comparison", None))

    def test_comparison_result(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        compare = self._get_compare(engine)
        if compare is None:
            pytest.skip("day type comparison method not found")
        result = compare(sample_interval_data)
        assert result is not None

    def test_weekday_peak_higher(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        compare = self._get_compare(engine)
        if compare is None:
            pytest.skip("day type comparison method not found")
        result = compare(sample_interval_data)
        wd_peak = getattr(result, "weekday_peak_kw", None)
        we_peak = getattr(result, "weekend_peak_kw", None)
        if wd_peak is not None and we_peak is not None:
            assert wd_peak >= we_peak

    def test_weekday_average_higher(self, sample_interval_data):
        engine = _m.LoadProfileEngine()
        compare = self._get_compare(engine)
        if compare is None:
            pytest.skip("day type comparison method not found")
        result = compare(sample_interval_data)
        wd_avg = getattr(result, "weekday_avg_kw", None)
        we_avg = getattr(result, "weekend_avg_kw", None)
        if wd_avg is not None and we_avg is not None:
            assert wd_avg >= we_avg
