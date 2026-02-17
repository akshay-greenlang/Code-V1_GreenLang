# -*- coding: utf-8 -*-
"""
Unit tests for StatisticalDetectorEngine - AGENT-DATA-013

Tests all 8 detection methods (IQR, z-score, modified z-score, MAD,
Grubbs, Tukey, percentile, ensemble) plus helpers and edge cases.
Target: 60+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

import math
from typing import List

import pytest

from greenlang.outlier_detector.statistical_detector import (
    StatisticalDetectorEngine,
    _safe_mean,
    _safe_std,
    _safe_median,
    _percentile,
    _severity_from_score,
)
from greenlang.outlier_detector.models import (
    DetectionMethod,
    EnsembleResult,
    OutlierScore,
    SeverityLevel,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def engine(config):
    return StatisticalDetectorEngine(config)


@pytest.fixture
def normal_values() -> List[float]:
    """100 values in range 1..100 with no outliers."""
    return [float(i) for i in range(1, 101)]


@pytest.fixture
def data_with_outliers() -> List[float]:
    """95 normal + 5 extreme values."""
    normal = [float(i) for i in range(1, 96)]
    outliers = [500.0, 600.0, 700.0, -200.0, -300.0]
    return normal + outliers


@pytest.fixture
def small_data() -> List[float]:
    return [1.0, 2.0, 3.0]


@pytest.fixture
def constant_data() -> List[float]:
    return [5.0] * 20


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_single(self):
        assert _safe_mean([42.0]) == 42.0

    def test_safe_mean_normal(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_safe_std_empty(self):
        assert _safe_std([]) == 0.0

    def test_safe_std_single(self):
        assert _safe_std([5.0]) == 0.0

    def test_safe_std_uniform(self):
        assert _safe_std([5.0, 5.0, 5.0]) == 0.0

    def test_safe_std_normal(self):
        result = _safe_std([1.0, 2.0, 3.0])
        assert result > 0.0

    def test_safe_std_with_precomputed_mean(self):
        vals = [10.0, 20.0, 30.0]
        mean = 20.0
        result = _safe_std(vals, mean)
        assert result > 0.0

    def test_safe_median_empty(self):
        assert _safe_median([]) == 0.0

    def test_safe_median_odd(self):
        assert _safe_median([3.0, 1.0, 2.0]) == 2.0

    def test_safe_median_even(self):
        assert _safe_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_percentile_empty(self):
        assert _percentile([], 0.5) == 0.0

    def test_percentile_single(self):
        assert _percentile([10.0], 0.5) == 10.0

    def test_percentile_50(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 0.5) == 3.0

    def test_percentile_0(self):
        vals = [1.0, 2.0, 3.0]
        assert _percentile(vals, 0.0) == 1.0

    def test_percentile_100(self):
        vals = [1.0, 2.0, 3.0]
        assert _percentile(vals, 1.0) == 3.0

    def test_severity_critical(self):
        assert _severity_from_score(0.95) == SeverityLevel.CRITICAL

    def test_severity_high(self):
        assert _severity_from_score(0.80) == SeverityLevel.HIGH

    def test_severity_medium(self):
        assert _severity_from_score(0.60) == SeverityLevel.MEDIUM

    def test_severity_low(self):
        assert _severity_from_score(0.40) == SeverityLevel.LOW

    def test_severity_info(self):
        assert _severity_from_score(0.1) == SeverityLevel.INFO


# =========================================================================
# IQR detection
# =========================================================================


class TestDetectIQR:
    """Tests for detect_iqr method."""

    def test_returns_list_of_outlier_scores(self, engine, data_with_outliers):
        result = engine.detect_iqr(data_with_outliers)
        assert isinstance(result, list)
        assert all(isinstance(s, OutlierScore) for s in result)

    def test_length_matches_input(self, engine, data_with_outliers):
        result = engine.detect_iqr(data_with_outliers)
        assert len(result) == len(data_with_outliers)

    def test_detects_known_outliers(self, engine, data_with_outliers):
        result = engine.detect_iqr(data_with_outliers)
        outlier_values = {s.value for s in result if s.is_outlier}
        assert 500.0 in outlier_values
        assert -300.0 in outlier_values

    def test_no_outliers_in_normal_data(self, engine, normal_values):
        result = engine.detect_iqr(normal_values)
        outliers = [s for s in result if s.is_outlier]
        assert len(outliers) == 0

    def test_custom_multiplier(self, engine, data_with_outliers):
        strict = engine.detect_iqr(data_with_outliers, multiplier=0.5)
        lenient = engine.detect_iqr(data_with_outliers, multiplier=5.0)
        strict_count = sum(1 for s in strict if s.is_outlier)
        lenient_count = sum(1 for s in lenient if s.is_outlier)
        assert strict_count >= lenient_count

    def test_column_name_propagated(self, engine, data_with_outliers):
        result = engine.detect_iqr(data_with_outliers, column_name="emissions")
        assert all(s.column_name == "emissions" for s in result)

    def test_method_is_iqr(self, engine, data_with_outliers):
        result = engine.detect_iqr(data_with_outliers)
        assert all(s.method == DetectionMethod.IQR for s in result)

    def test_provenance_hash_present(self, engine, data_with_outliers):
        result = engine.detect_iqr(data_with_outliers)
        assert all(len(s.provenance_hash) == 64 for s in result)

    def test_empty_input(self, engine):
        result = engine.detect_iqr([])
        assert result == []

    def test_constant_data_no_outliers(self, engine, constant_data):
        result = engine.detect_iqr(constant_data)
        assert all(not s.is_outlier for s in result)

    def test_small_dataset(self, engine, small_data):
        result = engine.detect_iqr(small_data)
        assert isinstance(result, list)


# =========================================================================
# Z-score detection
# =========================================================================


class TestDetectZscore:
    """Tests for detect_zscore method."""

    def test_returns_outlier_scores(self, engine, data_with_outliers):
        result = engine.detect_zscore(data_with_outliers)
        assert isinstance(result, list)
        assert all(isinstance(s, OutlierScore) for s in result)

    def test_detects_extreme_values(self, engine, data_with_outliers):
        result = engine.detect_zscore(data_with_outliers)
        outlier_values = {s.value for s in result if s.is_outlier}
        assert 700.0 in outlier_values or 600.0 in outlier_values

    def test_no_outliers_in_uniform(self, engine, normal_values):
        result = engine.detect_zscore(normal_values)
        outliers = [s for s in result if s.is_outlier]
        assert len(outliers) == 0

    def test_custom_threshold(self, engine, data_with_outliers):
        strict = engine.detect_zscore(data_with_outliers, threshold=1.0)
        lenient = engine.detect_zscore(data_with_outliers, threshold=5.0)
        assert sum(1 for s in strict if s.is_outlier) >= sum(
            1 for s in lenient if s.is_outlier
        )

    def test_method_is_zscore(self, engine, data_with_outliers):
        result = engine.detect_zscore(data_with_outliers)
        assert all(s.method == DetectionMethod.ZSCORE for s in result)

    def test_constant_data(self, engine, constant_data):
        result = engine.detect_zscore(constant_data)
        assert all(not s.is_outlier for s in result)


# =========================================================================
# Modified Z-score detection
# =========================================================================


class TestDetectModifiedZscore:
    """Tests for detect_modified_zscore method."""

    def test_returns_outlier_scores(self, engine, data_with_outliers):
        result = engine.detect_modified_zscore(data_with_outliers)
        assert isinstance(result, list)
        assert len(result) == len(data_with_outliers)

    def test_detects_outliers(self, engine, data_with_outliers):
        result = engine.detect_modified_zscore(data_with_outliers)
        n_outliers = sum(1 for s in result if s.is_outlier)
        assert n_outliers > 0

    def test_method_correct(self, engine, data_with_outliers):
        result = engine.detect_modified_zscore(data_with_outliers)
        assert all(s.method == DetectionMethod.MODIFIED_ZSCORE for s in result)

    def test_custom_threshold(self, engine, data_with_outliers):
        strict = engine.detect_modified_zscore(data_with_outliers, threshold=1.0)
        lenient = engine.detect_modified_zscore(data_with_outliers, threshold=10.0)
        assert sum(1 for s in strict if s.is_outlier) >= sum(
            1 for s in lenient if s.is_outlier
        )


# =========================================================================
# MAD detection
# =========================================================================


class TestDetectMAD:
    """Tests for detect_mad method."""

    def test_returns_outlier_scores(self, engine, data_with_outliers):
        result = engine.detect_mad(data_with_outliers)
        assert isinstance(result, list)
        assert len(result) == len(data_with_outliers)

    def test_detects_outliers(self, engine, data_with_outliers):
        result = engine.detect_mad(data_with_outliers)
        n_outliers = sum(1 for s in result if s.is_outlier)
        assert n_outliers > 0

    def test_method_correct(self, engine, data_with_outliers):
        result = engine.detect_mad(data_with_outliers)
        assert all(s.method == DetectionMethod.MAD for s in result)

    def test_constant_data(self, engine, constant_data):
        result = engine.detect_mad(constant_data)
        assert all(not s.is_outlier for s in result)


# =========================================================================
# Grubbs detection
# =========================================================================


class TestDetectGrubbs:
    """Tests for detect_grubbs method."""

    def test_returns_outlier_scores(self, engine, data_with_outliers):
        result = engine.detect_grubbs(data_with_outliers)
        assert isinstance(result, list)
        assert len(result) == len(data_with_outliers)

    def test_detects_outliers(self, engine, data_with_outliers):
        result = engine.detect_grubbs(data_with_outliers)
        n_outliers = sum(1 for s in result if s.is_outlier)
        assert n_outliers > 0

    def test_method_correct(self, engine, data_with_outliers):
        result = engine.detect_grubbs(data_with_outliers)
        assert all(s.method == DetectionMethod.GRUBBS for s in result)

    def test_custom_alpha(self, engine, data_with_outliers):
        strict = engine.detect_grubbs(data_with_outliers, alpha=0.01)
        lenient = engine.detect_grubbs(data_with_outliers, alpha=0.10)
        assert sum(1 for s in strict if s.is_outlier) <= sum(
            1 for s in lenient if s.is_outlier
        )


# =========================================================================
# Tukey detection
# =========================================================================


class TestDetectTukey:
    """Tests for detect_tukey method."""

    def test_returns_outlier_scores(self, engine, data_with_outliers):
        result = engine.detect_tukey(data_with_outliers)
        assert isinstance(result, list)
        assert len(result) == len(data_with_outliers)

    def test_detects_outliers(self, engine, data_with_outliers):
        result = engine.detect_tukey(data_with_outliers)
        n_outliers = sum(1 for s in result if s.is_outlier)
        assert n_outliers > 0

    def test_method_correct(self, engine, data_with_outliers):
        result = engine.detect_tukey(data_with_outliers)
        assert all(s.method == DetectionMethod.TUKEY for s in result)


# =========================================================================
# Percentile detection
# =========================================================================


class TestDetectPercentile:
    """Tests for detect_percentile method."""

    def test_returns_outlier_scores(self, engine, data_with_outliers):
        result = engine.detect_percentile(data_with_outliers)
        assert isinstance(result, list)
        assert len(result) == len(data_with_outliers)

    def test_detects_outliers(self, engine, data_with_outliers):
        result = engine.detect_percentile(data_with_outliers)
        n_outliers = sum(1 for s in result if s.is_outlier)
        assert n_outliers > 0

    def test_method_correct(self, engine, data_with_outliers):
        result = engine.detect_percentile(data_with_outliers)
        assert all(s.method == DetectionMethod.PERCENTILE for s in result)

    def test_custom_bounds(self, engine, data_with_outliers):
        strict = engine.detect_percentile(data_with_outliers, lower=0.10, upper=0.90)
        lenient = engine.detect_percentile(data_with_outliers, lower=0.01, upper=0.99)
        assert sum(1 for s in strict if s.is_outlier) >= sum(
            1 for s in lenient if s.is_outlier
        )


# =========================================================================
# Ensemble detection
# =========================================================================


class TestDetectEnsemble:
    """Tests for detect_ensemble method."""

    def test_returns_ensemble_results(self, engine, data_with_outliers):
        result = engine.detect_ensemble(data_with_outliers)
        assert isinstance(result, list)
        assert all(isinstance(r, EnsembleResult) for r in result)

    def test_length_matches_input(self, engine, data_with_outliers):
        result = engine.detect_ensemble(data_with_outliers)
        assert len(result) == len(data_with_outliers)

    def test_detects_outliers(self, engine, data_with_outliers):
        result = engine.detect_ensemble(data_with_outliers)
        n_outliers = sum(1 for r in result if r.is_outlier)
        assert n_outliers > 0

    def test_has_method_scores(self, engine, data_with_outliers):
        result = engine.detect_ensemble(data_with_outliers)
        for r in result:
            assert isinstance(r.method_scores, dict)
            assert len(r.method_scores) > 0

    def test_column_name_propagated(self, engine, data_with_outliers):
        result = engine.detect_ensemble(data_with_outliers, column_name="val")
        assert all(r.column_name == "val" for r in result)

    def test_custom_methods(self, engine, data_with_outliers):
        result = engine.detect_ensemble(
            data_with_outliers,
            methods=[DetectionMethod.IQR, DetectionMethod.ZSCORE],
        )
        assert isinstance(result, list)

    def test_empty_input(self, engine):
        result = engine.detect_ensemble([])
        assert result == []

    def test_provenance_hash_present(self, engine, data_with_outliers):
        result = engine.detect_ensemble(data_with_outliers)
        assert all(len(r.provenance_hash) == 64 for r in result)

    def test_ensemble_score_range(self, engine, data_with_outliers):
        result = engine.detect_ensemble(data_with_outliers)
        for r in result:
            assert 0.0 <= r.ensemble_score <= 1.0


# =========================================================================
# Determinism and provenance
# =========================================================================


class TestDeterminism:
    """Test that results are reproducible."""

    def test_iqr_deterministic(self, engine, data_with_outliers):
        r1 = engine.detect_iqr(data_with_outliers)
        r2 = engine.detect_iqr(data_with_outliers)
        for s1, s2 in zip(r1, r2):
            assert s1.score == s2.score
            assert s1.is_outlier == s2.is_outlier

    def test_ensemble_deterministic(self, engine, data_with_outliers):
        r1 = engine.detect_ensemble(data_with_outliers)
        r2 = engine.detect_ensemble(data_with_outliers)
        for s1, s2 in zip(r1, r2):
            assert s1.ensemble_score == s2.ensemble_score

    def test_score_bounds(self, engine, data_with_outliers):
        """All scores must be in [0, 1]."""
        for method_fn in [
            engine.detect_iqr,
            engine.detect_zscore,
            engine.detect_mad,
            engine.detect_grubbs,
            engine.detect_tukey,
            engine.detect_percentile,
        ]:
            result = method_fn(data_with_outliers)
            for s in result:
                assert 0.0 <= s.score <= 1.0, f"{method_fn.__name__}: score={s.score}"
