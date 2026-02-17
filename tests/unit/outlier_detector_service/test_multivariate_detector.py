# -*- coding: utf-8 -*-
"""
Unit tests for MultivariateDetectorEngine - AGENT-DATA-013

Tests Mahalanobis distance, Isolation Forest, Local Outlier Factor (LOF),
DBSCAN, matrix utilities, and edge cases.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from greenlang.outlier_detector.multivariate_detector import (
    MultivariateDetectorEngine,
    _severity_from_score,
)
from greenlang.outlier_detector.models import (
    DetectionMethod,
    MultivariateResult,
    OutlierScore,
    SeverityLevel,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def engine(config):
    return MultivariateDetectorEngine(config)


@pytest.fixture
def normal_records() -> List[Dict[str, Any]]:
    """20 records forming a tight cluster with 2 multivariate outliers."""
    records = []
    for i in range(18):
        records.append({"x": float(i), "y": float(i * 2), "z": float(i * 3)})
    # Multivariate outliers
    records.append({"x": 100.0, "y": 200.0, "z": 300.0})
    records.append({"x": -50.0, "y": -100.0, "z": -150.0})
    return records


@pytest.fixture
def small_records() -> List[Dict[str, Any]]:
    """Too few records for multivariate detection."""
    return [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]


@pytest.fixture
def clustered_records() -> List[Dict[str, Any]]:
    """Records forming two clusters with noise."""
    records = []
    # Cluster 1 around (10, 10)
    for i in range(10):
        records.append({"a": 10.0 + i * 0.1, "b": 10.0 + i * 0.2})
    # Cluster 2 around (50, 50)
    for i in range(10):
        records.append({"a": 50.0 + i * 0.1, "b": 50.0 + i * 0.2})
    # Noise points
    records.append({"a": 200.0, "b": 200.0})
    records.append({"a": -100.0, "b": -100.0})
    return records


@pytest.fixture
def constant_records() -> List[Dict[str, Any]]:
    """All identical records (singular covariance)."""
    return [{"x": 5.0, "y": 10.0} for _ in range(10)]


@pytest.fixture
def missing_records() -> List[Dict[str, Any]]:
    """Records with missing/non-numeric values."""
    return [
        {"x": 1.0, "y": 2.0},
        {"x": None, "y": 3.0},
        {"x": 4.0, "y": "bad"},
        {"x": 5.0, "y": 6.0},
    ]


# =========================================================================
# Severity helper
# =========================================================================


class TestSeverityHelper:
    """Test _severity_from_score function."""

    def test_critical(self):
        assert _severity_from_score(0.95) == SeverityLevel.CRITICAL

    def test_info(self):
        assert _severity_from_score(0.1) == SeverityLevel.INFO


# =========================================================================
# Mahalanobis distance detection
# =========================================================================


class TestDetectMahalanobis:
    """Tests for detect_mahalanobis method."""

    def test_returns_multivariate_result(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert isinstance(result, MultivariateResult)

    def test_method_is_mahalanobis(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert result.method == DetectionMethod.MAHALANOBIS

    def test_total_points_correct(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert result.total_points == len(normal_records)

    def test_detects_outliers(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert result.outliers_found > 0

    def test_scores_length_matches(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert len(result.scores) == len(normal_records)

    def test_columns_stored(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert result.columns == ["x", "y", "z"]

    def test_custom_threshold(self, engine, normal_records):
        strict = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"], threshold=1.0,
        )
        lenient = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"], threshold=10.0,
        )
        assert strict.outliers_found >= lenient.outliers_found

    def test_provenance_hash_present(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert len(result.provenance_hash) == 64

    def test_small_dataset(self, engine, small_records):
        result = engine.detect_mahalanobis(
            small_records, ["x", "y"],
        )
        assert result.outliers_found == 0

    def test_singular_covariance_fallback(self, engine, constant_records):
        result = engine.detect_mahalanobis(
            constant_records, ["x", "y"],
        )
        assert isinstance(result, MultivariateResult)

    def test_missing_values_skipped(self, engine, missing_records):
        result = engine.detect_mahalanobis(
            missing_records, ["x", "y"],
        )
        # Only 2 valid records -> insufficient
        assert isinstance(result, MultivariateResult)

    def test_score_bounds(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        for s in result.scores:
            assert 0.0 <= s.score <= 1.0


# =========================================================================
# Isolation Forest detection
# =========================================================================


class TestDetectIsolationForest:
    """Tests for detect_isolation_forest method."""

    def test_returns_multivariate_result(self, engine, normal_records):
        result = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        assert isinstance(result, MultivariateResult)

    def test_method_is_isolation_forest(self, engine, normal_records):
        result = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        assert result.method == DetectionMethod.ISOLATION_FOREST

    def test_detects_outliers(self, engine, normal_records):
        result = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=20,
        )
        assert result.outliers_found > 0

    def test_total_points_correct(self, engine, normal_records):
        result = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        assert result.total_points == len(normal_records)

    def test_scores_length_matches(self, engine, normal_records):
        result = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        assert len(result.scores) == len(normal_records)

    def test_score_range(self, engine, normal_records):
        result = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        for s in result.scores:
            assert 0.0 <= s.score <= 1.0

    def test_provenance_hash_present(self, engine, normal_records):
        result = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        assert len(result.provenance_hash) == 64

    def test_small_dataset(self, engine, small_records):
        result = engine.detect_isolation_forest(
            small_records, ["x", "y"], n_trees=10,
        )
        assert result.outliers_found == 0

    def test_deterministic(self, engine, normal_records):
        r1 = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        r2 = engine.detect_isolation_forest(
            normal_records, ["x", "y", "z"], n_trees=10,
        )
        for s1, s2 in zip(r1.scores, r2.scores):
            assert s1.score == s2.score


# =========================================================================
# Local Outlier Factor (LOF) detection
# =========================================================================


class TestDetectLOF:
    """Tests for detect_lof method."""

    def test_returns_multivariate_result(self, engine, clustered_records):
        result = engine.detect_lof(clustered_records, ["a", "b"], k=5)
        assert isinstance(result, MultivariateResult)

    def test_method_is_lof(self, engine, clustered_records):
        result = engine.detect_lof(clustered_records, ["a", "b"], k=5)
        assert result.method == DetectionMethod.LOF

    def test_detects_noise_points(self, engine, clustered_records):
        result = engine.detect_lof(clustered_records, ["a", "b"], k=5)
        assert result.outliers_found > 0

    def test_total_points_correct(self, engine, clustered_records):
        result = engine.detect_lof(clustered_records, ["a", "b"], k=5)
        assert result.total_points == len(clustered_records)

    def test_scores_length_matches(self, engine, clustered_records):
        result = engine.detect_lof(clustered_records, ["a", "b"], k=5)
        assert len(result.scores) == len(clustered_records)

    def test_score_range(self, engine, clustered_records):
        result = engine.detect_lof(clustered_records, ["a", "b"], k=5)
        for s in result.scores:
            assert 0.0 <= s.score <= 1.0

    def test_provenance_hash_present(self, engine, clustered_records):
        result = engine.detect_lof(clustered_records, ["a", "b"], k=5)
        assert len(result.provenance_hash) == 64

    def test_insufficient_data(self, engine, small_records):
        result = engine.detect_lof(small_records, ["x", "y"], k=5)
        assert result.outliers_found == 0


# =========================================================================
# DBSCAN detection
# =========================================================================


class TestDetectDBSCAN:
    """Tests for detect_dbscan method."""

    def test_returns_multivariate_result(self, engine, clustered_records):
        result = engine.detect_dbscan(
            clustered_records, ["a", "b"], eps=5.0, min_samples=3,
        )
        assert isinstance(result, MultivariateResult)

    def test_method_is_dbscan(self, engine, clustered_records):
        result = engine.detect_dbscan(
            clustered_records, ["a", "b"], eps=5.0, min_samples=3,
        )
        assert result.method == DetectionMethod.DBSCAN

    def test_detects_noise_as_outliers(self, engine, clustered_records):
        result = engine.detect_dbscan(
            clustered_records, ["a", "b"], eps=5.0, min_samples=3,
        )
        assert result.outliers_found > 0

    def test_scores_length_matches(self, engine, clustered_records):
        result = engine.detect_dbscan(
            clustered_records, ["a", "b"], eps=5.0, min_samples=3,
        )
        assert len(result.scores) == len(clustered_records)

    def test_cluster_labels_in_details(self, engine, clustered_records):
        result = engine.detect_dbscan(
            clustered_records, ["a", "b"], eps=5.0, min_samples=3,
        )
        for s in result.scores:
            assert "cluster_label" in s.details

    def test_auto_estimate_eps(self, engine, clustered_records):
        result = engine.detect_dbscan(
            clustered_records, ["a", "b"], min_samples=3,
        )
        assert isinstance(result, MultivariateResult)

    def test_provenance_hash_present(self, engine, clustered_records):
        result = engine.detect_dbscan(
            clustered_records, ["a", "b"], eps=5.0, min_samples=3,
        )
        assert len(result.provenance_hash) == 64

    def test_insufficient_data(self, engine, small_records):
        result = engine.detect_dbscan(
            small_records, ["x", "y"], min_samples=5,
        )
        assert result.outliers_found == 0


# =========================================================================
# Matrix utilities
# =========================================================================


class TestMatrixUtilities:
    """Tests for covariance and matrix inversion helpers."""

    def test_compute_covariance_matrix(self, engine):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        cov = engine._compute_covariance_matrix(data)
        assert len(cov) == 2
        assert len(cov[0]) == 2

    def test_covariance_symmetric(self, engine):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        cov = engine._compute_covariance_matrix(data)
        assert cov[0][1] == pytest.approx(cov[1][0])

    def test_invert_identity(self, engine):
        identity = [[1.0, 0.0], [0.0, 1.0]]
        inv = engine._invert_matrix(identity)
        assert inv is not None
        assert inv[0][0] == pytest.approx(1.0)
        assert inv[1][1] == pytest.approx(1.0)
        assert abs(inv[0][1]) < 1e-10
        assert abs(inv[1][0]) < 1e-10

    def test_invert_singular_returns_none(self, engine):
        singular = [[1.0, 2.0], [2.0, 4.0]]
        inv = engine._invert_matrix(singular)
        assert inv is None

    def test_diagonal_inverse(self, engine):
        cov = [[4.0, 0.0], [0.0, 9.0]]
        inv = engine._diagonal_inverse(cov, 2)
        assert inv[0][0] == pytest.approx(0.25)
        assert inv[1][1] == pytest.approx(1.0 / 9.0)

    def test_pairwise_distances_symmetric(self, engine):
        data = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        dist = engine._pairwise_distances(data)
        assert dist[0][1] == pytest.approx(dist[1][0])
        assert dist[0][0] == 0.0

    def test_euclidean_distance(self, engine):
        d = engine._euclidean_distance([0.0, 0.0], [3.0, 4.0])
        assert d == pytest.approx(5.0)

    def test_average_path_length_n1(self):
        assert MultivariateDetectorEngine._average_path_length(1) == 0.0

    def test_average_path_length_n2(self):
        assert MultivariateDetectorEngine._average_path_length(2) == 1.0

    def test_average_path_length_large(self):
        result = MultivariateDetectorEngine._average_path_length(100)
        assert result > 0.0


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and data quality."""

    def test_empty_records(self, engine):
        result = engine.detect_mahalanobis([], ["x", "y"])
        assert result.outliers_found == 0

    def test_single_column(self, engine, normal_records):
        result = engine.detect_mahalanobis(normal_records, ["x"])
        assert isinstance(result, MultivariateResult)

    def test_confidence_bounds(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_mean_distance_nonnegative(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert result.mean_distance >= 0.0

    def test_max_distance_nonnegative(self, engine, normal_records):
        result = engine.detect_mahalanobis(
            normal_records, ["x", "y", "z"],
        )
        assert result.max_distance >= 0.0
