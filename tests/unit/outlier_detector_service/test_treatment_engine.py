# -*- coding: utf-8 -*-
"""
Unit tests for TreatmentEngine - AGENT-DATA-013

Tests cap_values, winsorize, flag_outliers, remove_outliers,
replace_with_imputed, mark_for_investigation, undo_treatment,
compute_impact, and edge cases.
Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.outlier_detector.treatment_engine import (
    TreatmentEngine,
    _safe_mean,
    _safe_median,
    _safe_std,
    _percentile,
)
from greenlang.outlier_detector.models import (
    DetectionMethod,
    ImpactAnalysis,
    OutlierScore,
    SeverityLevel,
    TreatmentResult,
    TreatmentStrategy,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def engine(config):
    return TreatmentEngine(config)


def _make_detection(
    value: Any = 500.0,
    score: float = 0.9,
    is_outlier: bool = True,
    record_index: int = 0,
    column_name: str = "val",
    confidence: float = 0.8,
) -> OutlierScore:
    """Create a test OutlierScore."""
    return OutlierScore(
        record_index=record_index,
        column_name=column_name,
        value=value,
        method=DetectionMethod.IQR,
        score=score,
        is_outlier=is_outlier,
        threshold=1.5,
        severity=SeverityLevel.HIGH,
        details={},
        confidence=confidence,
        provenance_hash="a" * 64,
    )


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """10 records with 2 outliers at indices 3 and 7."""
    return [
        {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
        {"val": 500.0}, {"val": 9.0}, {"val": 13.0},
        {"val": 11.5}, {"val": -200.0}, {"val": 10.0},
        {"val": 12.0},
    ]


@pytest.fixture
def sample_detections() -> List[OutlierScore]:
    """Detections for sample_records: 2 outliers at indices 3 and 7."""
    return [
        _make_detection(value=10.0, score=0.1, is_outlier=False, record_index=0),
        _make_detection(value=12.0, score=0.1, is_outlier=False, record_index=1),
        _make_detection(value=11.0, score=0.1, is_outlier=False, record_index=2),
        _make_detection(value=500.0, score=0.95, is_outlier=True, record_index=3),
        _make_detection(value=9.0, score=0.1, is_outlier=False, record_index=4),
        _make_detection(value=13.0, score=0.1, is_outlier=False, record_index=5),
        _make_detection(value=11.5, score=0.1, is_outlier=False, record_index=6),
        _make_detection(value=-200.0, score=0.9, is_outlier=True, record_index=7),
        _make_detection(value=10.0, score=0.1, is_outlier=False, record_index=8),
        _make_detection(value=12.0, score=0.1, is_outlier=False, record_index=9),
    ]


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_normal(self):
        assert _safe_mean([2.0, 4.0, 6.0]) == pytest.approx(4.0)

    def test_safe_std_single(self):
        assert _safe_std([5.0]) == 0.0

    def test_safe_median_even(self):
        assert _safe_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_safe_median_odd(self):
        assert _safe_median([1.0, 2.0, 3.0]) == 2.0

    def test_percentile_empty(self):
        assert _percentile([], 0.5) == 0.0

    def test_percentile_single(self):
        assert _percentile([42.0], 0.5) == 42.0

    def test_percentile_median(self):
        result = _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5)
        assert result == pytest.approx(3.0)


# =========================================================================
# Cap values
# =========================================================================


class TestCapValues:
    """Tests for cap_values method."""

    def test_returns_treatment_results(self, engine, sample_records):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, TreatmentResult) for r in results)

    def test_caps_high_value(self, engine, sample_records):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
        )
        capped_indices = {r.record_index for r in results}
        assert 3 in capped_indices  # 500.0 > 100.0

    def test_caps_low_value(self, engine, sample_records):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
        )
        capped_indices = {r.record_index for r in results}
        assert 7 in capped_indices  # -200.0 < 0.0

    def test_treated_value_within_bounds(self, engine, sample_records):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
        )
        for r in results:
            assert 0.0 <= r.treated_value <= 100.0

    def test_strategy_is_cap(self, engine, sample_records):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
        )
        for r in results:
            assert r.strategy == TreatmentStrategy.CAP

    def test_reversible(self, engine, sample_records):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
        )
        for r in results:
            assert r.reversible is True

    def test_provenance_hash_present(self, engine, sample_records):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
        )
        for r in results:
            assert len(r.provenance_hash) == 64

    def test_auto_detect_bounds(self, engine, sample_records):
        results = engine.cap_values(sample_records, "val")
        assert isinstance(results, list)

    def test_with_detections_filter(self, engine, sample_records, sample_detections):
        results = engine.cap_values(
            sample_records, "val", lower=0.0, upper=100.0,
            detections=sample_detections,
        )
        # Only outlier indices should be capped
        for r in results:
            assert r.record_index in {3, 7}


# =========================================================================
# Winsorize
# =========================================================================


class TestWinsorize:
    """Tests for winsorize method."""

    def test_returns_treatment_results(self, engine, sample_records):
        results = engine.winsorize(sample_records, "val")
        assert isinstance(results, list)

    def test_strategy_is_winsorize(self, engine, sample_records):
        results = engine.winsorize(sample_records, "val")
        for r in results:
            assert r.strategy == TreatmentStrategy.WINSORIZE

    def test_custom_percentile(self, engine, sample_records):
        results = engine.winsorize(sample_records, "val", pct=0.1)
        assert isinstance(results, list)

    def test_reversible(self, engine, sample_records):
        results = engine.winsorize(sample_records, "val")
        for r in results:
            assert r.reversible is True

    def test_provenance_hash_present(self, engine, sample_records):
        results = engine.winsorize(sample_records, "val")
        for r in results:
            assert len(r.provenance_hash) == 64


# =========================================================================
# Flag outliers
# =========================================================================


class TestFlagOutliers:
    """Tests for flag_outliers method."""

    def test_returns_treatment_results(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        assert isinstance(results, list)
        assert all(isinstance(r, TreatmentResult) for r in results)

    def test_only_outliers_flagged(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        assert len(results) == 2  # indices 3 and 7

    def test_strategy_is_flag(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        for r in results:
            assert r.strategy == TreatmentStrategy.FLAG

    def test_value_unchanged(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        for r in results:
            assert r.treated_value == r.original_value

    def test_reversible(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        for r in results:
            assert r.reversible is True

    def test_provenance_hash_present(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        for r in results:
            assert len(r.provenance_hash) == 64


# =========================================================================
# Remove outliers
# =========================================================================


class TestRemoveOutliers:
    """Tests for remove_outliers method."""

    def test_returns_treatment_results(self, engine, sample_records, sample_detections):
        results = engine.remove_outliers(sample_records, sample_detections)
        assert isinstance(results, list)

    def test_only_outliers_removed(self, engine, sample_records, sample_detections):
        results = engine.remove_outliers(sample_records, sample_detections)
        assert len(results) == 2

    def test_treated_value_is_none(self, engine, sample_records, sample_detections):
        results = engine.remove_outliers(sample_records, sample_detections)
        for r in results:
            assert r.treated_value is None

    def test_strategy_is_remove(self, engine, sample_records, sample_detections):
        results = engine.remove_outliers(sample_records, sample_detections)
        for r in results:
            assert r.strategy == TreatmentStrategy.REMOVE

    def test_provenance_hash_present(self, engine, sample_records, sample_detections):
        results = engine.remove_outliers(sample_records, sample_detections)
        for r in results:
            assert len(r.provenance_hash) == 64


# =========================================================================
# Replace with imputed
# =========================================================================


class TestReplaceWithImputed:
    """Tests for replace_with_imputed method."""

    def test_returns_treatment_results(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(sample_records, sample_detections)
        assert isinstance(results, list)

    def test_only_outliers_replaced(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(sample_records, sample_detections)
        assert len(results) == 2

    def test_strategy_is_replace(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(sample_records, sample_detections)
        for r in results:
            assert r.strategy == TreatmentStrategy.REPLACE

    def test_default_method_is_median(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(sample_records, sample_detections)
        # Replaced with median of non-outlier values
        non_outlier_vals = [10.0, 12.0, 11.0, 9.0, 13.0, 11.5, 10.0, 12.0]
        expected_median = _safe_median(non_outlier_vals)
        for r in results:
            assert r.treated_value == pytest.approx(expected_median)

    def test_mean_method(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(
            sample_records, sample_detections, method="mean",
        )
        non_outlier_vals = [10.0, 12.0, 11.0, 9.0, 13.0, 11.5, 10.0, 12.0]
        expected_mean = _safe_mean(non_outlier_vals)
        for r in results:
            assert r.treated_value == pytest.approx(expected_mean)

    def test_mode_method(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(
            sample_records, sample_detections, method="mode",
        )
        # Mode of non-outlier values; 10.0 and 12.0 each appear twice
        for r in results:
            assert r.treated_value is not None

    def test_empty_detections_returns_empty(self, engine, sample_records):
        results = engine.replace_with_imputed(sample_records, [])
        assert results == []

    def test_provenance_hash_present(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(sample_records, sample_detections)
        for r in results:
            assert len(r.provenance_hash) == 64

    def test_reversible(self, engine, sample_records, sample_detections):
        results = engine.replace_with_imputed(sample_records, sample_detections)
        for r in results:
            assert r.reversible is True


# =========================================================================
# Mark for investigation
# =========================================================================


class TestMarkForInvestigation:
    """Tests for mark_for_investigation method."""

    def test_returns_treatment_results(self, engine, sample_records, sample_detections):
        results = engine.mark_for_investigation(sample_records, sample_detections)
        assert isinstance(results, list)

    def test_only_outliers_marked(self, engine, sample_records, sample_detections):
        results = engine.mark_for_investigation(sample_records, sample_detections)
        assert len(results) == 2

    def test_strategy_is_investigate(self, engine, sample_records, sample_detections):
        results = engine.mark_for_investigation(sample_records, sample_detections)
        for r in results:
            assert r.strategy == TreatmentStrategy.INVESTIGATE

    def test_value_preserved(self, engine, sample_records, sample_detections):
        results = engine.mark_for_investigation(sample_records, sample_detections)
        for r in results:
            assert r.treated_value == r.original_value

    def test_provenance_hash_present(self, engine, sample_records, sample_detections):
        results = engine.mark_for_investigation(sample_records, sample_detections)
        for r in results:
            assert len(r.provenance_hash) == 64


# =========================================================================
# Apply treatment (dispatcher)
# =========================================================================


class TestApplyTreatment:
    """Tests for apply_treatment dispatcher method."""

    def test_flag_strategy(self, engine, sample_records, sample_detections):
        results = engine.apply_treatment(
            sample_records, sample_detections, TreatmentStrategy.FLAG,
        )
        for r in results:
            assert r.strategy == TreatmentStrategy.FLAG

    def test_remove_strategy(self, engine, sample_records, sample_detections):
        results = engine.apply_treatment(
            sample_records, sample_detections, TreatmentStrategy.REMOVE,
        )
        for r in results:
            assert r.strategy == TreatmentStrategy.REMOVE

    def test_investigate_strategy(self, engine, sample_records, sample_detections):
        results = engine.apply_treatment(
            sample_records, sample_detections, TreatmentStrategy.INVESTIGATE,
        )
        for r in results:
            assert r.strategy == TreatmentStrategy.INVESTIGATE

    def test_replace_strategy(self, engine, sample_records, sample_detections):
        results = engine.apply_treatment(
            sample_records, sample_detections, TreatmentStrategy.REPLACE,
        )
        for r in results:
            assert r.strategy == TreatmentStrategy.REPLACE


# =========================================================================
# Undo treatment
# =========================================================================


class TestUndoTreatment:
    """Tests for undo_treatment method."""

    def test_undo_returns_treatment_result(self, engine, sample_records, sample_detections):
        # First apply a treatment
        results = engine.flag_outliers(sample_records, sample_detections)
        treatment_id = results[0].treatment_id
        undo_result = engine.undo_treatment(treatment_id)
        assert isinstance(undo_result, TreatmentResult)

    def test_undo_restores_original_value(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        original_val = results[0].original_value
        treatment_id = results[0].treatment_id
        undo_result = engine.undo_treatment(treatment_id)
        assert undo_result.treated_value == original_val

    def test_undo_not_found_raises(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.undo_treatment("nonexistent-id")

    def test_undo_twice_raises(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        treatment_id = results[0].treatment_id
        engine.undo_treatment(treatment_id)
        with pytest.raises(ValueError, match="already undone"):
            engine.undo_treatment(treatment_id)

    def test_undo_not_reversible(self, engine, sample_records, sample_detections):
        results = engine.flag_outliers(sample_records, sample_detections)
        treatment_id = results[0].treatment_id
        undo_result = engine.undo_treatment(treatment_id)
        assert undo_result.reversible is False


# =========================================================================
# Compute impact
# =========================================================================


class TestComputeImpact:
    """Tests for compute_impact method."""

    def test_returns_impact_analysis(self, engine):
        original = [{"val": 10.0}, {"val": 500.0}, {"val": 12.0}]
        treated = [{"val": 10.0}, {"val": 50.0}, {"val": 12.0}]
        result = engine.compute_impact(original, treated, "val")
        assert isinstance(result, ImpactAnalysis)

    def test_column_name_stored(self, engine):
        original = [{"val": 10.0}, {"val": 500.0}]
        treated = [{"val": 10.0}, {"val": 50.0}]
        result = engine.compute_impact(original, treated, "val")
        assert result.column_name == "val"

    def test_records_affected(self, engine):
        original = [{"val": 10.0}, {"val": 500.0}, {"val": 12.0}]
        treated = [{"val": 10.0}, {"val": 50.0}, {"val": 12.0}]
        result = engine.compute_impact(original, treated, "val")
        assert result.records_affected == 1

    def test_mean_change_pct(self, engine):
        original = [{"val": 10.0}, {"val": 500.0}]
        treated = [{"val": 10.0}, {"val": 10.0}]
        result = engine.compute_impact(original, treated, "val")
        assert result.mean_change_pct > 0.0

    def test_std_change_pct(self, engine):
        original = [{"val": 10.0}, {"val": 500.0}]
        treated = [{"val": 10.0}, {"val": 10.0}]
        result = engine.compute_impact(original, treated, "val")
        assert result.std_change_pct > 0.0

    def test_no_change(self, engine):
        original = [{"val": 10.0}, {"val": 12.0}]
        treated = [{"val": 10.0}, {"val": 12.0}]
        result = engine.compute_impact(original, treated, "val")
        assert result.records_affected == 0
        assert result.mean_change_pct == 0.0

    def test_provenance_hash_present(self, engine):
        original = [{"val": 10.0}]
        treated = [{"val": 20.0}]
        result = engine.compute_impact(original, treated, "val")
        assert len(result.provenance_hash) == 64

    def test_distribution_shift(self, engine):
        original = [{"val": float(i)} for i in range(100)]
        treated = [{"val": float(i + 50)} for i in range(100)]
        result = engine.compute_impact(original, treated, "val")
        assert result.distribution_shift > 0.0


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_records_cap(self, engine):
        results = engine.cap_values([], "val", lower=0.0, upper=100.0)
        assert results == []

    def test_missing_column_cap(self, engine):
        records = [{"other": 10.0}]
        results = engine.cap_values(records, "val", lower=0.0, upper=100.0)
        assert results == []

    def test_non_numeric_values_skipped(self, engine):
        records = [{"val": "bad"}, {"val": None}, {"val": 500.0}]
        results = engine.cap_values(records, "val", lower=0.0, upper=100.0)
        assert isinstance(results, list)

    def test_all_non_outlier_detections(self, engine, sample_records):
        detections = [
            _make_detection(value=10.0, score=0.1, is_outlier=False, record_index=i)
            for i in range(len(sample_records))
        ]
        flag_results = engine.flag_outliers(sample_records, detections)
        assert flag_results == []

    def test_deterministic_flag(self, engine, sample_records, sample_detections):
        r1 = engine.flag_outliers(sample_records, sample_detections)
        r2 = engine.flag_outliers(sample_records, sample_detections)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.record_index == b.record_index
            assert a.original_value == b.original_value
