# -*- coding: utf-8 -*-
"""
Unit tests for OutlierClassifierEngine - AGENT-DATA-013

Tests 5-class classification (error, genuine_extreme, data_entry,
regime_change, sensor_fault), confidence computation, treatment
recommendations, and edge cases.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.outlier_detector.outlier_classifier import (
    OutlierClassifierEngine,
    _CLASS_TREATMENTS,
    _safe_mean,
    _safe_std,
    _severity_from_score,
)
from greenlang.outlier_detector.models import (
    DetectionMethod,
    OutlierClass,
    OutlierClassification,
    OutlierScore,
    SeverityLevel,
    TreatmentStrategy,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def engine(config):
    return OutlierClassifierEngine(config)


def _make_detection(
    value: Any = 100.0,
    score: float = 0.8,
    is_outlier: bool = True,
    method: DetectionMethod = DetectionMethod.IQR,
    record_index: int = 0,
    details: Dict[str, Any] = None,
    column_name: str = "val",
    confidence: float = 0.7,
) -> OutlierScore:
    """Create a test OutlierScore."""
    return OutlierScore(
        record_index=record_index,
        column_name=column_name,
        value=value,
        method=method,
        score=score,
        is_outlier=is_outlier,
        threshold=1.5,
        severity=_severity_from_score(score),
        details=details or {"mean": 10.0, "std": 5.0},
        confidence=confidence,
        provenance_hash="a" * 64,
    )


@pytest.fixture
def error_detection():
    """Detection that resembles a processing error."""
    return _make_detection(
        value=1e16,
        score=0.95,
        details={"mean": 10.0, "std": 5.0},
    )


@pytest.fixture
def data_entry_detection():
    """Detection that resembles a data entry mistake (10x the mean)."""
    return _make_detection(
        value=100.0,
        score=0.7,
        details={"mean": 10.0, "std": 5.0},
    )


@pytest.fixture
def sensor_detection():
    """Detection that resembles sensor fault (sensor bound value)."""
    return _make_detection(
        value=9999.0,
        score=0.95,
        details={"mean": 50.0, "std": 20.0},
    )


@pytest.fixture
def regime_detection():
    """Detection that may be a regime change."""
    return _make_detection(
        value=100.0,
        score=0.7,
        method=DetectionMethod.TEMPORAL,
        details={"mean": 10.0, "std": 5.0},
    )


@pytest.fixture
def genuine_detection():
    """Detection that is a genuine extreme."""
    return _make_detection(
        value=35.7,
        score=0.65,
        details={"mean": 10.0, "std": 5.0},
    )


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    return [{"val": 10.0}, {"val": 12.0}, {"val": 100.0}]


# =========================================================================
# Treatment mapping constant
# =========================================================================


class TestClassTreatments:
    """Test _CLASS_TREATMENTS mapping."""

    def test_error_maps_to_remove(self):
        assert _CLASS_TREATMENTS[OutlierClass.ERROR] == TreatmentStrategy.REMOVE

    def test_genuine_maps_to_flag(self):
        assert _CLASS_TREATMENTS[OutlierClass.GENUINE_EXTREME] == TreatmentStrategy.FLAG

    def test_data_entry_maps_to_replace(self):
        assert _CLASS_TREATMENTS[OutlierClass.DATA_ENTRY] == TreatmentStrategy.REPLACE

    def test_regime_change_maps_to_investigate(self):
        assert _CLASS_TREATMENTS[OutlierClass.REGIME_CHANGE] == TreatmentStrategy.INVESTIGATE

    def test_sensor_fault_maps_to_remove(self):
        assert _CLASS_TREATMENTS[OutlierClass.SENSOR_FAULT] == TreatmentStrategy.REMOVE

    def test_all_classes_covered(self):
        for oc in OutlierClass:
            assert oc in _CLASS_TREATMENTS


# =========================================================================
# classify_single
# =========================================================================


class TestClassifySingle:
    """Tests for classify_single method."""

    def test_returns_outlier_classification(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert isinstance(result, OutlierClassification)

    def test_record_index_matches(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert result.record_index == error_detection.record_index

    def test_column_name_matches(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert result.column_name == error_detection.column_name

    def test_has_class_scores(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert isinstance(result.class_scores, dict)
        assert len(result.class_scores) == 5

    def test_has_evidence(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert isinstance(result.evidence, list)

    def test_has_recommended_treatment(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert isinstance(result.recommended_treatment, TreatmentStrategy)

    def test_provenance_hash_present(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert len(result.provenance_hash) == 64

    def test_confidence_range(self, engine, error_detection):
        result = engine.classify_single(error_detection, {"val": 1e16})
        assert 0.0 <= result.confidence <= 1.0


# =========================================================================
# Error classification
# =========================================================================


class TestClassifyError:
    """Tests for error classification heuristics."""

    def test_extreme_magnitude(self, engine):
        det = _make_detection(value=1e16, score=0.9, details={"mean": 10.0, "std": 5.0})
        score, evidence = engine._classify_error(det, {"val": 1e16})
        assert score > 0.0
        assert any("magnitude" in e.lower() for e in evidence)

    def test_exact_zero_in_nonzero_context(self, engine):
        det = _make_detection(value=0.0, score=0.6, details={"mean": 10.0})
        score, evidence = engine._classify_error(det, {"val": 0.0})
        assert score > 0.0

    def test_negative_in_positive_column(self, engine):
        det = _make_detection(value=-50.0, score=0.7, details={"mean": 10.0})
        score, evidence = engine._classify_error(det, {"val": -50.0})
        assert score > 0.0

    def test_power_of_2(self, engine):
        det = _make_detection(value=256.0, score=0.5, details={"mean": 10.0})
        score, evidence = engine._classify_error(det, {"val": 256.0})
        assert score > 0.0

    def test_none_value_returns_zero(self, engine):
        det = _make_detection(value=None, score=0.5, details={"mean": 10.0})
        score, evidence = engine._classify_error(det, {})
        assert score == 0.0

    def test_score_capped_at_1(self, engine):
        det = _make_detection(value=-1e16, score=0.99, details={"mean": 10.0})
        score, evidence = engine._classify_error(det, {})
        assert score <= 1.0


# =========================================================================
# Data entry classification
# =========================================================================


class TestClassifyDataEntry:
    """Tests for data entry classification heuristics."""

    def test_10x_mean(self, engine):
        det = _make_detection(value=100.0, score=0.7, details={"mean": 10.0, "std": 5.0})
        score, evidence = engine._classify_data_entry(det, {"val": 100.0})
        assert score > 0.0

    def test_100x_mean(self, engine):
        det = _make_detection(value=1000.0, score=0.9, details={"mean": 10.0, "std": 5.0})
        score, evidence = engine._classify_data_entry(det, {"val": 1000.0})
        assert score > 0.0

    def test_round_number(self, engine):
        det = _make_detection(value=10000.0, score=0.7, details={"mean": 50.0, "std": 20.0})
        score, evidence = engine._classify_data_entry(det, {"val": 10000.0})
        assert score > 0.0

    def test_digit_transposition(self, engine):
        det = _make_detection(value=1324.0, score=0.7, details={"mean": 1234.0, "std": 50.0})
        score, evidence = engine._classify_data_entry(det, {"val": 1324.0})
        assert score > 0.0

    def test_none_value_returns_zero(self, engine):
        det = _make_detection(value=None, score=0.5, details={"mean": 10.0})
        score, evidence = engine._classify_data_entry(det, {})
        assert score == 0.0


# =========================================================================
# Regime change classification
# =========================================================================


class TestClassifyRegimeChange:
    """Tests for regime change classification heuristics."""

    def test_subsequent_values_at_new_level(self, engine):
        det = _make_detection(
            value=100.0, score=0.7, record_index=5,
            details={"mean": 10.0, "std": 5.0},
        )
        context = {"time_series": [10] * 5 + [100, 98, 102, 99, 100]}
        score, evidence = engine._classify_regime_change(det, {}, context)
        assert score > 0.0

    def test_step_change_pattern(self, engine):
        det = _make_detection(
            value=100.0, score=0.7, record_index=5,
            details={"mean": 10.0, "std": 5.0},
        )
        context = {"time_series": [10, 11, 12, 10, 11, 100]}
        score, evidence = engine._classify_regime_change(det, {}, context)
        assert score > 0.0

    def test_known_events(self, engine):
        det = _make_detection(value=100.0, score=0.7, details={"mean": 10.0})
        context = {"events": ["factory_upgrade"]}
        score, evidence = engine._classify_regime_change(det, {}, context)
        assert score > 0.0

    def test_temporal_method_bonus(self, engine):
        det = _make_detection(
            value=100.0, score=0.7,
            method=DetectionMethod.TEMPORAL,
            details={"mean": 10.0},
        )
        score, evidence = engine._classify_regime_change(det, {}, {})
        assert score > 0.0

    def test_no_context_low_score(self, engine):
        det = _make_detection(value=100.0, score=0.7, details={"mean": 10.0})
        score, evidence = engine._classify_regime_change(det, {}, {})
        # Without context, only temporal method bonus might contribute
        assert isinstance(score, float)


# =========================================================================
# Sensor fault classification
# =========================================================================


class TestClassifySensorFault:
    """Tests for sensor fault classification heuristics."""

    def test_sensor_bound_value(self, engine, sensor_detection):
        score, evidence = engine._classify_sensor_fault(sensor_detection, {})
        assert score > 0.0

    def test_common_sensor_bounds(self, engine):
        for bound in [0.0, 9999.0, 65535.0, -32768.0, 32767.0]:
            det = _make_detection(
                value=bound, score=0.95,
                details={"std": 20.0},
            )
            score, evidence = engine._classify_sensor_fault(det, {})
            assert score > 0.0

    def test_very_high_score_bonus(self, engine):
        det = _make_detection(value=1234.0, score=0.95, details={"std": 5.0})
        score, evidence = engine._classify_sensor_fault(det, {})
        assert score > 0.0

    def test_stuck_at_zero(self, engine):
        det = _make_detection(
            value=0.0, score=0.95,
            details={"std": 20.0},
        )
        score, evidence = engine._classify_sensor_fault(det, {})
        assert score > 0.0

    def test_none_value_returns_zero(self, engine):
        det = _make_detection(value=None, score=0.5, details={})
        score, evidence = engine._classify_sensor_fault(det, {})
        assert score == 0.0


# =========================================================================
# Genuine extreme classification
# =========================================================================


class TestClassifyGenuineExtreme:
    """Tests for genuine extreme classification heuristics."""

    def test_base_score(self, engine, genuine_detection):
        score, evidence = engine._classify_genuine_extreme(
            genuine_detection, {},
        )
        assert score >= 0.3  # base score

    def test_moderate_outlier_bonus(self, engine):
        det = _make_detection(
            value=35.7, score=0.65,
            details={"mean": 10.0, "std": 5.0},
        )
        score, evidence = engine._classify_genuine_extreme(det, {})
        assert score > 0.3

    def test_non_round_number_bonus(self, engine):
        det = _make_detection(
            value=37.82, score=0.65,
            details={"mean": 10.0, "std": 5.0},
        )
        score, evidence = engine._classify_genuine_extreme(det, {})
        assert any("non-round" in e.lower() for e in evidence)

    def test_positive_direction_bonus(self, engine):
        det = _make_detection(
            value=50.0, score=0.65,
            details={"mean": 10.0, "std": 5.0},
        )
        score, evidence = engine._classify_genuine_extreme(det, {})
        assert score > 0.3

    def test_none_value(self, engine):
        det = _make_detection(value=None, score=0.5, details={"mean": 10.0})
        score, evidence = engine._classify_genuine_extreme(det, {})
        assert score == 0.3  # just base score


# =========================================================================
# Batch classification
# =========================================================================


class TestClassifyOutliers:
    """Tests for classify_outliers batch method."""

    def test_only_outliers_classified(self, engine, sample_records):
        detections = [
            _make_detection(value=10.0, score=0.1, is_outlier=False, record_index=0),
            _make_detection(value=12.0, score=0.1, is_outlier=False, record_index=1),
            _make_detection(value=100.0, score=0.9, is_outlier=True, record_index=2),
        ]
        results = engine.classify_outliers(detections, sample_records)
        assert len(results) == 1
        assert results[0].record_index == 2

    def test_multiple_outliers(self, engine, sample_records):
        detections = [
            _make_detection(value=10.0, score=0.9, is_outlier=True, record_index=0),
            _make_detection(value=100.0, score=0.9, is_outlier=True, record_index=2),
        ]
        results = engine.classify_outliers(detections, sample_records)
        assert len(results) == 2

    def test_no_outliers_returns_empty(self, engine, sample_records):
        detections = [
            _make_detection(value=10.0, score=0.1, is_outlier=False, record_index=0),
        ]
        results = engine.classify_outliers(detections, sample_records)
        assert results == []

    def test_empty_detections(self, engine, sample_records):
        results = engine.classify_outliers([], sample_records)
        assert results == []


# =========================================================================
# Confidence computation
# =========================================================================


class TestComputeClassificationConfidence:
    """Tests for compute_classification_confidence method."""

    def test_clear_winner(self, engine):
        scores = {"error": 0.9, "data_entry": 0.1, "genuine_extreme": 0.1,
                  "regime_change": 0.0, "sensor_fault": 0.0}
        conf = engine.compute_classification_confidence(scores)
        assert conf > 0.5

    def test_ambiguous_scores(self, engine):
        scores = {"error": 0.5, "data_entry": 0.5, "genuine_extreme": 0.5,
                  "regime_change": 0.5, "sensor_fault": 0.5}
        conf = engine.compute_classification_confidence(scores)
        assert conf <= 0.3

    def test_empty_scores(self, engine):
        assert engine.compute_classification_confidence({}) == 0.0

    def test_single_class(self, engine):
        scores = {"error": 0.8}
        conf = engine.compute_classification_confidence(scores)
        assert conf > 0.0

    def test_confidence_range(self, engine):
        scores = {"error": 0.7, "data_entry": 0.3, "genuine_extreme": 0.2,
                  "regime_change": 0.1, "sensor_fault": 0.0}
        conf = engine.compute_classification_confidence(scores)
        assert 0.0 <= conf <= 1.0
