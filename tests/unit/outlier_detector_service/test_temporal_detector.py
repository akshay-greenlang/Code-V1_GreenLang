# -*- coding: utf-8 -*-
"""
Unit tests for TemporalDetectorEngine - AGENT-DATA-013

Tests CUSUM, trend break, seasonal residual, moving window, EWMA,
change point detection, and edge cases.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

import math
from typing import List

import pytest

from greenlang.outlier_detector.temporal_detector import (
    TemporalDetectorEngine,
    _safe_mean,
    _safe_std,
    _safe_median,
    _severity_from_score,
)
from greenlang.outlier_detector.models import (
    DetectionMethod,
    OutlierScore,
    SeverityLevel,
    TemporalMethod,
    TemporalResult,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def engine(config):
    return TemporalDetectorEngine(config)


@pytest.fixture
def stable_series() -> List[float]:
    """50-point stable time series with small oscillations."""
    return [10.0 + math.sin(i * 0.3) * 2.0 for i in range(50)]


@pytest.fixture
def series_with_step() -> List[float]:
    """50-point series with regime change at index 25."""
    return [10.0] * 25 + [100.0] * 25


@pytest.fixture
def series_with_spikes() -> List[float]:
    """30-point series with 3 spike anomalies."""
    values = [10.0 + math.sin(i * 0.5) * 3.0 for i in range(30)]
    values[10] = 500.0
    values[20] = -300.0
    values[25] = 800.0
    return values


@pytest.fixture
def seasonal_series() -> List[float]:
    """48-point series with period-12 seasonality and 2 anomalies."""
    values = []
    for i in range(48):
        seasonal = 10.0 * math.sin(2 * math.pi * (i % 12) / 12.0)
        values.append(50.0 + seasonal + i * 0.1)
    values[15] = 500.0  # anomaly
    values[35] = -200.0  # anomaly
    return values


@pytest.fixture
def short_series() -> List[float]:
    return [1.0, 2.0, 3.0]


@pytest.fixture
def constant_series() -> List[float]:
    return [5.0] * 30


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_std_single(self):
        assert _safe_std([5.0]) == 0.0

    def test_safe_median_even(self):
        assert _safe_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_severity_boundaries(self):
        assert _severity_from_score(0.95) == SeverityLevel.CRITICAL
        assert _severity_from_score(0.80) == SeverityLevel.HIGH
        assert _severity_from_score(0.60) == SeverityLevel.MEDIUM
        assert _severity_from_score(0.40) == SeverityLevel.LOW
        assert _severity_from_score(0.10) == SeverityLevel.INFO


# =========================================================================
# CUSUM detection
# =========================================================================


class TestDetectCusum:
    """Tests for detect_cusum method."""

    def test_returns_temporal_result(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], TemporalResult)

    def test_method_is_cusum(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        assert results[0].method == TemporalMethod.CUSUM

    def test_detects_step_change(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        assert results[0].anomalies_found > 0
        assert len(results[0].change_points) > 0

    def test_stable_series_fewer_changes(self, engine, stable_series):
        results = engine.detect_cusum(stable_series)
        assert results[0].anomalies_found <= 2

    def test_series_length_correct(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        assert results[0].series_length == len(series_with_step)

    def test_scores_length_matches(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        assert len(results[0].scores) == len(series_with_step)

    def test_custom_threshold(self, engine, series_with_step):
        strict = engine.detect_cusum(series_with_step, threshold=1.0)
        lenient = engine.detect_cusum(series_with_step, threshold=1000.0)
        assert strict[0].anomalies_found >= lenient[0].anomalies_found

    def test_custom_drift(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step, drift=0.1)
        assert isinstance(results, list)

    def test_provenance_hash_present(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        assert len(results[0].provenance_hash) == 64

    def test_short_series_empty_result(self, engine, short_series):
        results = engine.detect_cusum(short_series)
        assert results[0].anomalies_found == 0

    def test_column_name_propagated(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step, column_name="temp")
        assert results[0].column_name == "temp"


# =========================================================================
# Trend break detection
# =========================================================================


class TestDetectTrendBreaks:
    """Tests for detect_trend_breaks method."""

    def test_returns_temporal_result(self, engine, series_with_step):
        results = engine.detect_trend_breaks(series_with_step)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_method_is_trend_break(self, engine, series_with_step):
        results = engine.detect_trend_breaks(series_with_step)
        assert results[0].method == TemporalMethod.TREND_BREAK

    def test_detects_step_change(self, engine, series_with_step):
        results = engine.detect_trend_breaks(series_with_step)
        assert results[0].anomalies_found > 0

    def test_short_series(self, engine, short_series):
        results = engine.detect_trend_breaks(short_series)
        assert results[0].anomalies_found == 0

    def test_custom_window(self, engine, series_with_step):
        results = engine.detect_trend_breaks(series_with_step, window=5)
        assert isinstance(results, list)

    def test_scores_length_matches(self, engine, series_with_step):
        results = engine.detect_trend_breaks(series_with_step)
        assert len(results[0].scores) == len(series_with_step)

    def test_provenance_hash_present(self, engine, series_with_step):
        results = engine.detect_trend_breaks(series_with_step)
        assert len(results[0].provenance_hash) == 64


# =========================================================================
# Seasonal residual detection
# =========================================================================


class TestDetectSeasonalResiduals:
    """Tests for detect_seasonal_residuals method."""

    def test_returns_temporal_result(self, engine, seasonal_series):
        results = engine.detect_seasonal_residuals(seasonal_series, period=12)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_method_is_seasonal_residual(self, engine, seasonal_series):
        results = engine.detect_seasonal_residuals(seasonal_series, period=12)
        assert results[0].method == TemporalMethod.SEASONAL_RESIDUAL

    def test_detects_anomalies(self, engine, seasonal_series):
        results = engine.detect_seasonal_residuals(seasonal_series, period=12)
        assert results[0].anomalies_found > 0

    def test_short_series_insufficient(self, engine, short_series):
        results = engine.detect_seasonal_residuals(short_series, period=12)
        assert results[0].anomalies_found == 0

    def test_series_length_correct(self, engine, seasonal_series):
        results = engine.detect_seasonal_residuals(seasonal_series, period=12)
        assert results[0].series_length == len(seasonal_series)

    def test_default_period(self, engine, seasonal_series):
        results = engine.detect_seasonal_residuals(seasonal_series)
        assert isinstance(results, list)

    def test_provenance_hash_present(self, engine, seasonal_series):
        results = engine.detect_seasonal_residuals(seasonal_series, period=12)
        assert len(results[0].provenance_hash) == 64


# =========================================================================
# Moving window detection
# =========================================================================


class TestDetectMovingWindow:
    """Tests for detect_moving_window method."""

    def test_returns_temporal_result(self, engine, series_with_spikes):
        results = engine.detect_moving_window(series_with_spikes)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_method_is_moving_window(self, engine, series_with_spikes):
        results = engine.detect_moving_window(series_with_spikes)
        assert results[0].method == TemporalMethod.MOVING_WINDOW

    def test_detects_spikes(self, engine, series_with_spikes):
        results = engine.detect_moving_window(series_with_spikes)
        assert results[0].anomalies_found > 0

    def test_custom_window(self, engine, series_with_spikes):
        results = engine.detect_moving_window(series_with_spikes, window=5)
        assert isinstance(results, list)

    def test_custom_threshold(self, engine, series_with_spikes):
        strict = engine.detect_moving_window(series_with_spikes, threshold=1.0)
        lenient = engine.detect_moving_window(series_with_spikes, threshold=10.0)
        assert strict[0].anomalies_found >= lenient[0].anomalies_found

    def test_scores_length_matches(self, engine, series_with_spikes):
        results = engine.detect_moving_window(series_with_spikes)
        assert len(results[0].scores) == len(series_with_spikes)

    def test_provenance_hash_present(self, engine, series_with_spikes):
        results = engine.detect_moving_window(series_with_spikes)
        assert len(results[0].provenance_hash) == 64


# =========================================================================
# EWMA detection
# =========================================================================


class TestDetectEWMA:
    """Tests for detect_ewma method."""

    def test_returns_temporal_result(self, engine, series_with_spikes):
        results = engine.detect_ewma(series_with_spikes)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_method_is_ewma(self, engine, series_with_spikes):
        results = engine.detect_ewma(series_with_spikes)
        assert results[0].method == TemporalMethod.EWMA

    def test_detects_spikes(self, engine, series_with_spikes):
        results = engine.detect_ewma(series_with_spikes)
        assert results[0].anomalies_found > 0

    def test_custom_alpha(self, engine, series_with_spikes):
        r1 = engine.detect_ewma(series_with_spikes, alpha=0.1)
        r2 = engine.detect_ewma(series_with_spikes, alpha=0.9)
        assert isinstance(r1, list) and isinstance(r2, list)

    def test_short_series_empty(self, engine, short_series):
        results = engine.detect_ewma(short_series)
        assert results[0].anomalies_found == 0

    def test_provenance_hash_present(self, engine, series_with_spikes):
        results = engine.detect_ewma(series_with_spikes)
        assert len(results[0].provenance_hash) == 64

    def test_scores_length_matches(self, engine, series_with_spikes):
        results = engine.detect_ewma(series_with_spikes)
        assert len(results[0].scores) == len(series_with_spikes)


# =========================================================================
# Change point detection (multi-method summary)
# =========================================================================


class TestDetectChangePoints:
    """Tests for detect_change_points method."""

    def test_returns_list_of_dicts(self, engine, series_with_step):
        results = engine.detect_change_points(series_with_step)
        assert isinstance(results, list)
        for cp in results:
            assert isinstance(cp, dict)

    def test_dict_has_required_keys(self, engine, series_with_step):
        results = engine.detect_change_points(series_with_step)
        for cp in results:
            assert "index" in cp
            assert "value" in cp
            assert "method" in cp
            assert "score" in cp

    def test_detects_step_change(self, engine, series_with_step):
        results = engine.detect_change_points(series_with_step)
        assert len(results) > 0

    def test_deduplicates_by_index(self, engine, series_with_step):
        results = engine.detect_change_points(series_with_step)
        indices = [cp["index"] for cp in results]
        assert len(indices) == len(set(indices))

    def test_sorted_by_index(self, engine, series_with_step):
        results = engine.detect_change_points(series_with_step)
        indices = [cp["index"] for cp in results]
        assert indices == sorted(indices)


# =========================================================================
# Edge cases and determinism
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and reproducibility."""

    def test_constant_series(self, engine, constant_series):
        cusum = engine.detect_cusum(constant_series)
        assert cusum[0].anomalies_found == 0

    def test_deterministic_cusum(self, engine, series_with_step):
        r1 = engine.detect_cusum(series_with_step)
        r2 = engine.detect_cusum(series_with_step)
        assert r1[0].anomalies_found == r2[0].anomalies_found
        assert r1[0].change_points == r2[0].change_points

    def test_deterministic_ewma(self, engine, series_with_spikes):
        r1 = engine.detect_ewma(series_with_spikes)
        r2 = engine.detect_ewma(series_with_spikes)
        for s1, s2 in zip(r1[0].scores, r2[0].scores):
            assert s1.score == s2.score

    def test_score_bounds_cusum(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        for s in results[0].scores:
            assert 0.0 <= s.score <= 1.0

    def test_score_bounds_ewma(self, engine, series_with_spikes):
        results = engine.detect_ewma(series_with_spikes)
        for s in results[0].scores:
            assert 0.0 <= s.score <= 1.0

    def test_compute_slope_flat(self):
        slope = TemporalDetectorEngine._compute_slope([5.0, 5.0, 5.0])
        assert slope == 0.0

    def test_compute_slope_positive(self):
        slope = TemporalDetectorEngine._compute_slope([1.0, 2.0, 3.0, 4.0])
        assert slope > 0.0

    def test_compute_slope_single(self):
        slope = TemporalDetectorEngine._compute_slope([5.0])
        assert slope == 0.0

    def test_baseline_stats_present(self, engine, series_with_step):
        results = engine.detect_cusum(series_with_step)
        assert results[0].baseline_mean is not None
        assert results[0].baseline_std is not None
