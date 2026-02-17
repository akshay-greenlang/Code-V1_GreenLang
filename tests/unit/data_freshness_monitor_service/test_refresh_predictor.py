# -*- coding: utf-8 -*-
"""
Unit tests for RefreshPredictorEngine - AGENT-DATA-016 Engine 5.

Tests all public methods of RefreshPredictorEngine including prediction
strategies, interval statistics, estimation algorithms, confidence scoring,
prediction evaluation, anomaly detection, batch predictions, accuracy
tracking, and edge cases.

Target: 70+ tests, 85%+ coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from greenlang.data_freshness_monitor.refresh_predictor import (
    PredictionStatus,
    RefreshPrediction,
    RefreshPredictorEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _make_history(
    count: int,
    interval_hours: float = 24.0,
    end: datetime | None = None,
) -> List[datetime]:
    """Build a list of evenly-spaced refresh timestamps.

    Args:
        count: Number of timestamps.
        interval_hours: Hours between consecutive timestamps.
        end: Most-recent timestamp. Defaults to now.

    Returns:
        List of UTC datetimes from oldest to newest.
    """
    if end is None:
        end = _utcnow()
    return [
        end - timedelta(hours=interval_hours * (count - 1 - i))
        for i in range(count)
    ]


def _make_irregular_history(
    intervals_hours: List[float],
    start: datetime | None = None,
) -> List[datetime]:
    """Build refresh history from a list of interval durations.

    Args:
        intervals_hours: List of intervals between consecutive timestamps.
        start: First timestamp. Defaults to 30 days ago.

    Returns:
        List of datetimes with length len(intervals_hours) + 1.
    """
    if start is None:
        start = _utcnow() - timedelta(days=30)
    result = [start]
    for interval in intervals_hours:
        result.append(result[-1] + timedelta(hours=interval))
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> RefreshPredictorEngine:
    """Create a fresh RefreshPredictorEngine with default config."""
    return RefreshPredictorEngine()


@pytest.fixture
def custom_engine() -> RefreshPredictorEngine:
    """Create a RefreshPredictorEngine with custom configuration."""
    return RefreshPredictorEngine(config={
        "prediction_min_samples": 3,
        "default_decay_factor": 0.9,
        "max_confidence": 0.99,
        "anomaly_delay_factor": 2.0,
    })


@pytest.fixture
def short_history() -> List[datetime]:
    """2 data points -- below default min_samples (5)."""
    return _make_history(2, interval_hours=24.0)


@pytest.fixture
def medium_history() -> List[datetime]:
    """7 data points -- between 5 and 10."""
    return _make_history(7, interval_hours=24.0)


@pytest.fixture
def rich_history() -> List[datetime]:
    """15 data points -- above 10."""
    return _make_history(15, interval_hours=24.0)


# ============================================================================
# Test: predict_next_refresh
# ============================================================================


class TestPredictNextRefresh:
    """Tests for the predict_next_refresh method."""

    def test_predict_with_empty_history(self, engine: RefreshPredictorEngine):
        """Empty history should fall back to cadence-based prediction."""
        pred = engine.predict_next_refresh("ds-001", [], cadence_hours=24.0)
        assert isinstance(pred, RefreshPrediction)
        assert pred.algorithm == "cadence"
        assert pred.confidence == 0.3
        assert pred.sample_count == 0
        assert pred.dataset_id == "ds-001"

    def test_predict_with_single_entry(self, engine: RefreshPredictorEngine):
        """Single-entry history should use cadence-based prediction."""
        now = _utcnow()
        pred = engine.predict_next_refresh("ds-002", [now], cadence_hours=12.0)
        assert pred.algorithm == "cadence"
        assert pred.confidence == 0.3
        assert pred.sample_count == 1

    def test_predict_with_short_history(
        self, engine: RefreshPredictorEngine, short_history: List[datetime],
    ):
        """2-sample history (below min_samples=5) uses cadence-based."""
        pred = engine.predict_next_refresh(
            "ds-003", short_history, cadence_hours=24.0,
        )
        assert pred.algorithm == "cadence"
        assert pred.sample_count == 2

    def test_predict_with_three_samples(self, engine: RefreshPredictorEngine):
        """3-sample history (below default min_samples=5) uses cadence."""
        history = _make_history(3, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-004", history, cadence_hours=24.0)
        assert pred.algorithm == "cadence"
        assert pred.sample_count == 3

    def test_predict_with_four_samples(self, engine: RefreshPredictorEngine):
        """4-sample history (below default min_samples=5) uses cadence."""
        history = _make_history(4, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-005", history, cadence_hours=24.0)
        assert pred.algorithm == "cadence"

    def test_predict_with_five_samples_uses_mean(
        self, engine: RefreshPredictorEngine,
    ):
        """5-sample history (exactly min_samples) uses mean_interval."""
        history = _make_history(5, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-006", history, cadence_hours=24.0)
        assert pred.algorithm == "mean_interval"
        assert pred.sample_count == 5

    def test_predict_with_medium_history_uses_mean(
        self, engine: RefreshPredictorEngine, medium_history: List[datetime],
    ):
        """7-sample history uses mean_interval."""
        pred = engine.predict_next_refresh(
            "ds-007", medium_history, cadence_hours=24.0,
        )
        assert pred.algorithm == "mean_interval"
        assert pred.sample_count == 7

    def test_predict_with_ten_samples_uses_mean(
        self, engine: RefreshPredictorEngine,
    ):
        """10-sample history (boundary) uses mean_interval."""
        history = _make_history(10, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-008", history, cadence_hours=24.0)
        assert pred.algorithm == "mean_interval"
        assert pred.sample_count == 10

    def test_predict_with_eleven_samples_uses_weighted(
        self, engine: RefreshPredictorEngine,
    ):
        """11-sample history uses weighted_recent."""
        history = _make_history(11, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-009", history, cadence_hours=24.0)
        assert pred.algorithm == "weighted_recent"
        assert pred.sample_count == 11

    def test_predict_with_rich_history_uses_weighted(
        self, engine: RefreshPredictorEngine, rich_history: List[datetime],
    ):
        """15-sample history uses weighted_recent."""
        pred = engine.predict_next_refresh(
            "ds-010", rich_history, cadence_hours=24.0,
        )
        assert pred.algorithm == "weighted_recent"
        assert pred.sample_count == 15

    def test_predict_provenance_hash_is_set(
        self, engine: RefreshPredictorEngine, medium_history: List[datetime],
    ):
        """Every prediction must have a non-empty provenance hash."""
        pred = engine.predict_next_refresh(
            "ds-011", medium_history, cadence_hours=24.0,
        )
        assert isinstance(pred.provenance_hash, str)
        assert len(pred.provenance_hash) == 64  # SHA-256

    def test_predict_status_is_pending(
        self, engine: RefreshPredictorEngine, medium_history: List[datetime],
    ):
        """New predictions should have PENDING status."""
        pred = engine.predict_next_refresh(
            "ds-012", medium_history, cadence_hours=24.0,
        )
        assert pred.status == PredictionStatus.PENDING

    def test_predict_stores_prediction(
        self, engine: RefreshPredictorEngine, medium_history: List[datetime],
    ):
        """Predictions are persisted in internal storage."""
        pred = engine.predict_next_refresh(
            "ds-013", medium_history, cadence_hours=24.0,
        )
        preds = engine.get_predictions("ds-013")
        assert len(preds) == 1
        assert preds[0].prediction_id == pred.prediction_id

    def test_predict_increments_stats(
        self, engine: RefreshPredictorEngine, medium_history: List[datetime],
    ):
        """Prediction increments the created counter."""
        engine.predict_next_refresh("ds-014", medium_history, cadence_hours=24.0)
        stats = engine.get_statistics()
        assert stats["predictions_created"] == 1

    def test_predict_invalid_empty_dataset_id(
        self, engine: RefreshPredictorEngine,
    ):
        """Empty dataset_id raises ValueError."""
        with pytest.raises(ValueError, match="dataset_id"):
            engine.predict_next_refresh("", [], cadence_hours=24.0)

    def test_predict_invalid_whitespace_dataset_id(
        self, engine: RefreshPredictorEngine,
    ):
        """Whitespace-only dataset_id raises ValueError."""
        with pytest.raises(ValueError, match="dataset_id"):
            engine.predict_next_refresh("   ", [], cadence_hours=24.0)

    def test_predict_invalid_zero_cadence(
        self, engine: RefreshPredictorEngine,
    ):
        """cadence_hours <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="cadence_hours"):
            engine.predict_next_refresh("ds-015", [], cadence_hours=0)

    def test_predict_invalid_negative_cadence(
        self, engine: RefreshPredictorEngine,
    ):
        """Negative cadence_hours raises ValueError."""
        with pytest.raises(ValueError, match="cadence_hours"):
            engine.predict_next_refresh("ds-016", [], cadence_hours=-1.0)

    def test_predict_unsorted_history_is_sorted(
        self, engine: RefreshPredictorEngine,
    ):
        """Engine sorts unsorted history internally."""
        now = _utcnow()
        unsorted = [
            now - timedelta(hours=24),
            now - timedelta(hours=72),
            now - timedelta(hours=48),
            now,
            now - timedelta(hours=96),
        ]
        pred = engine.predict_next_refresh("ds-017", unsorted, cadence_hours=24.0)
        assert pred.sample_count == 5
        # The predicted time should be after the most recent timestamp
        assert pred.predicted_at > now - timedelta(hours=1)

    def test_predict_with_custom_min_samples(
        self, custom_engine: RefreshPredictorEngine,
    ):
        """Custom min_samples=3 uses mean at 3 samples."""
        history = _make_history(3, interval_hours=24.0)
        pred = custom_engine.predict_next_refresh("ds-018", history, cadence_hours=24.0)
        assert pred.algorithm == "mean_interval"

    def test_predict_cadence_hours_stored(
        self, engine: RefreshPredictorEngine,
    ):
        """cadence_hours value is stored in the prediction."""
        pred = engine.predict_next_refresh("ds-019", [], cadence_hours=48.0)
        assert pred.cadence_hours == 48.0


# ============================================================================
# Test: compute_mean_interval / compute_median_interval
# ============================================================================


class TestIntervalComputation:
    """Tests for compute_mean_interval and compute_median_interval."""

    def test_mean_interval_empty(self, engine: RefreshPredictorEngine):
        """Empty history returns 0.0."""
        assert engine.compute_mean_interval([]) == 0.0

    def test_mean_interval_single(self, engine: RefreshPredictorEngine):
        """Single entry returns 0.0."""
        assert engine.compute_mean_interval([_utcnow()]) == 0.0

    def test_mean_interval_regular(self, engine: RefreshPredictorEngine):
        """Regular 24h intervals produce a mean of 24.0."""
        history = _make_history(5, interval_hours=24.0)
        mean = engine.compute_mean_interval(history)
        assert mean == pytest.approx(24.0, abs=0.01)

    def test_mean_interval_irregular(self, engine: RefreshPredictorEngine):
        """Irregular intervals produce the correct arithmetic mean."""
        history = _make_irregular_history([12.0, 24.0, 36.0, 48.0])
        mean = engine.compute_mean_interval(history)
        expected = statistics.mean([12.0, 24.0, 36.0, 48.0])
        assert mean == pytest.approx(expected, abs=0.01)

    def test_median_interval_empty(self, engine: RefreshPredictorEngine):
        """Empty history returns 0.0."""
        assert engine.compute_median_interval([]) == 0.0

    def test_median_interval_single(self, engine: RefreshPredictorEngine):
        """Single entry returns 0.0."""
        assert engine.compute_median_interval([_utcnow()]) == 0.0

    def test_median_interval_regular(self, engine: RefreshPredictorEngine):
        """Regular 24h intervals produce a median of 24.0."""
        history = _make_history(5, interval_hours=24.0)
        median = engine.compute_median_interval(history)
        assert median == pytest.approx(24.0, abs=0.01)

    def test_median_interval_odd_count(self, engine: RefreshPredictorEngine):
        """Odd number of intervals returns the middle value."""
        history = _make_irregular_history([10.0, 20.0, 30.0])
        median = engine.compute_median_interval(history)
        assert median == pytest.approx(20.0, abs=0.01)

    def test_median_interval_even_count(self, engine: RefreshPredictorEngine):
        """Even number of intervals returns average of two middle values."""
        history = _make_irregular_history([10.0, 20.0, 30.0, 40.0])
        median = engine.compute_median_interval(history)
        expected = statistics.median([10.0, 20.0, 30.0, 40.0])
        assert median == pytest.approx(expected, abs=0.01)


# ============================================================================
# Test: compute_interval_statistics
# ============================================================================


class TestIntervalStatistics:
    """Tests for compute_interval_statistics."""

    def test_stats_empty(self, engine: RefreshPredictorEngine):
        """Empty history produces all-zero stats with count=0."""
        stats = engine.compute_interval_statistics([])
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["stddev"] == 0.0

    def test_stats_single(self, engine: RefreshPredictorEngine):
        """Single entry produces all-zero stats."""
        stats = engine.compute_interval_statistics([_utcnow()])
        assert stats["count"] == 0

    def test_stats_regular_intervals(self, engine: RefreshPredictorEngine):
        """Regular 24h intervals: mean=24, stddev~0, cv~0."""
        history = _make_history(6, interval_hours=24.0)
        stats = engine.compute_interval_statistics(history)
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(24.0, abs=0.01)
        assert stats["median"] == pytest.approx(24.0, abs=0.01)
        assert stats["stddev"] == pytest.approx(0.0, abs=0.01)
        assert stats["cv"] == pytest.approx(0.0, abs=0.01)
        assert stats["min"] == pytest.approx(24.0, abs=0.01)
        assert stats["max"] == pytest.approx(24.0, abs=0.01)

    def test_stats_irregular_intervals(self, engine: RefreshPredictorEngine):
        """Irregular intervals produce correct min/max/cv."""
        history = _make_irregular_history([6.0, 12.0, 24.0, 48.0])
        stats = engine.compute_interval_statistics(history)
        assert stats["count"] == 4
        assert stats["min"] == pytest.approx(6.0, abs=0.01)
        assert stats["max"] == pytest.approx(48.0, abs=0.01)
        assert stats["cv"] > 0.0  # Non-zero coefficient of variation

    def test_stats_two_entries(self, engine: RefreshPredictorEngine):
        """Two entries produce one interval, stddev=0."""
        now = _utcnow()
        history = [now - timedelta(hours=10), now]
        stats = engine.compute_interval_statistics(history)
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(10.0, abs=0.01)
        assert stats["stddev"] == 0.0


# ============================================================================
# Test: estimate_next_by_mean / median / weighted_recent / cadence
# ============================================================================


class TestEstimationMethods:
    """Tests for the four estimation methods."""

    def test_estimate_by_mean_regular(self, engine: RefreshPredictorEngine):
        """Mean estimation adds mean interval to last timestamp."""
        history = _make_history(5, interval_hours=24.0)
        estimated = engine.estimate_next_by_mean(history)
        expected = history[-1] + timedelta(hours=24.0)
        assert abs((estimated - expected).total_seconds()) < 1.0

    def test_estimate_by_mean_requires_two_samples(
        self, engine: RefreshPredictorEngine,
    ):
        """Fewer than 2 samples raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 samples"):
            engine.estimate_next_by_mean([_utcnow()])

    def test_estimate_by_mean_empty_raises(
        self, engine: RefreshPredictorEngine,
    ):
        """Empty history raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 samples"):
            engine.estimate_next_by_mean([])

    def test_estimate_by_median_regular(self, engine: RefreshPredictorEngine):
        """Median estimation adds median interval to last timestamp."""
        history = _make_history(5, interval_hours=24.0)
        estimated = engine.estimate_next_by_median(history)
        expected = history[-1] + timedelta(hours=24.0)
        assert abs((estimated - expected).total_seconds()) < 1.0

    def test_estimate_by_median_requires_two_samples(
        self, engine: RefreshPredictorEngine,
    ):
        """Fewer than 2 samples raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 samples"):
            engine.estimate_next_by_median([_utcnow()])

    def test_estimate_by_weighted_recent_regular(
        self, engine: RefreshPredictorEngine,
    ):
        """Weighted-recent on regular intervals is close to mean."""
        history = _make_history(10, interval_hours=24.0)
        estimated = engine.estimate_next_by_weighted_recent(history)
        expected = history[-1] + timedelta(hours=24.0)
        # With regular intervals, weighted average should be very close to 24h
        assert abs((estimated - expected).total_seconds()) < 60.0

    def test_estimate_by_weighted_recent_requires_two(
        self, engine: RefreshPredictorEngine,
    ):
        """Fewer than 2 samples raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 samples"):
            engine.estimate_next_by_weighted_recent([_utcnow()])

    def test_estimate_by_weighted_recent_invalid_decay(
        self, engine: RefreshPredictorEngine,
    ):
        """Decay factor outside (0,1) raises ValueError."""
        history = _make_history(5, interval_hours=24.0)
        with pytest.raises(ValueError, match="decay_factor"):
            engine.estimate_next_by_weighted_recent(history, decay_factor=0.0)
        with pytest.raises(ValueError, match="decay_factor"):
            engine.estimate_next_by_weighted_recent(history, decay_factor=1.0)
        with pytest.raises(ValueError, match="decay_factor"):
            engine.estimate_next_by_weighted_recent(history, decay_factor=1.5)

    def test_estimate_by_weighted_recent_custom_decay(
        self, engine: RefreshPredictorEngine,
    ):
        """Custom decay factor produces a valid prediction."""
        history = _make_history(10, interval_hours=24.0)
        estimated = engine.estimate_next_by_weighted_recent(
            history, decay_factor=0.5,
        )
        # Should still be roughly 24h from last
        delta = (estimated - history[-1]).total_seconds() / 3600.0
        assert 20.0 < delta < 28.0

    def test_estimate_by_weighted_recent_recency_bias(
        self, engine: RefreshPredictorEngine,
    ):
        """Weighted-recent biases toward the most recent intervals."""
        # Old intervals are 48h, recent intervals are 12h
        history = _make_irregular_history([48, 48, 48, 48, 12, 12, 12, 12])
        estimated = engine.estimate_next_by_weighted_recent(history)
        delta = (estimated - history[-1]).total_seconds() / 3600.0
        # With 0.85 decay, should be closer to 12h than to 30h (mean)
        assert delta < 30.0

    def test_estimate_by_cadence(self, engine: RefreshPredictorEngine):
        """Cadence estimation adds cadence hours to last refresh."""
        now = _utcnow()
        estimated = engine.estimate_next_by_cadence(now, 48.0)
        expected = now + timedelta(hours=48.0)
        assert estimated == expected


# ============================================================================
# Test: compute_prediction_confidence
# ============================================================================


class TestConfidence:
    """Tests for compute_prediction_confidence."""

    def test_confidence_empty_history(self, engine: RefreshPredictorEngine):
        """Empty history returns 0.0 confidence."""
        assert engine.compute_prediction_confidence([], 24.0) == 0.0

    def test_confidence_single_entry(self, engine: RefreshPredictorEngine):
        """Single entry returns 0.0 confidence (no intervals)."""
        assert engine.compute_prediction_confidence([_utcnow()], 24.0) == 0.0

    def test_confidence_increases_with_samples(
        self, engine: RefreshPredictorEngine,
    ):
        """More samples should produce higher confidence."""
        c3 = engine.compute_prediction_confidence(
            _make_history(3, 24.0), 24.0,
        )
        c10 = engine.compute_prediction_confidence(
            _make_history(10, 24.0), 24.0,
        )
        c20 = engine.compute_prediction_confidence(
            _make_history(20, 24.0), 24.0,
        )
        assert c3 < c10 < c20

    def test_confidence_regular_higher_than_irregular(
        self, engine: RefreshPredictorEngine,
    ):
        """Regular intervals yield higher confidence than irregular."""
        regular = _make_history(10, interval_hours=24.0)
        irregular = _make_irregular_history([6, 48, 12, 36, 3, 72, 24, 18, 9])
        c_regular = engine.compute_prediction_confidence(regular, 24.0)
        c_irregular = engine.compute_prediction_confidence(irregular, 24.0)
        assert c_regular > c_irregular

    def test_confidence_capped_at_max(self, engine: RefreshPredictorEngine):
        """Confidence is capped at max_confidence (default 0.95)."""
        huge_history = _make_history(100, interval_hours=24.0)
        confidence = engine.compute_prediction_confidence(huge_history, 24.0)
        assert confidence <= 0.95

    def test_confidence_capped_at_custom_max(
        self, custom_engine: RefreshPredictorEngine,
    ):
        """Custom max_confidence (0.99) is respected."""
        huge_history = _make_history(100, interval_hours=24.0)
        confidence = custom_engine.compute_prediction_confidence(
            huge_history, 24.0,
        )
        assert confidence <= 0.99

    def test_confidence_recency_bonus_present(
        self, engine: RefreshPredictorEngine,
    ):
        """With 5+ consistent intervals within 1.5x cadence, get recency bonus."""
        # 6 intervals all at 24h with cadence=24h -> tolerance=36h -> all pass
        history = _make_history(7, interval_hours=24.0)
        confidence = engine.compute_prediction_confidence(history, 24.0)
        # base = min(1, 6/20)*0.5 = 0.15
        # regularity = (1-0)*0.3 = 0.3
        # recency = 0.2
        # total = 0.65
        assert confidence >= 0.6

    def test_confidence_no_recency_bonus_insufficient_window(
        self, engine: RefreshPredictorEngine,
    ):
        """Fewer than 5 intervals means no recency bonus (window < 5)."""
        history = _make_history(4, interval_hours=24.0)
        confidence = engine.compute_prediction_confidence(history, 24.0)
        # 3 intervals < 5 window size -> recency bonus = 0
        # base = min(1, 3/20)*0.5 = 0.075
        # regularity = ~0.3
        # total ~= 0.375
        assert confidence < 0.6

    def test_confidence_no_recency_bonus_exceeded_tolerance(
        self, engine: RefreshPredictorEngine,
    ):
        """If any of the last 5 intervals exceeds 1.5x cadence, no bonus."""
        # 6 intervals: five are 24h, one recent spike at 48h
        # cadence=24, tolerance=36; 48 > 36 -> no recency bonus
        history = _make_irregular_history([24, 24, 24, 24, 24, 48])
        confidence = engine.compute_prediction_confidence(history, 24.0)
        # Without recency bonus, should be < full
        assert confidence < 0.8

    def test_confidence_nonnegative(self, engine: RefreshPredictorEngine):
        """Confidence is always >= 0.0."""
        # Very irregular history
        history = _make_irregular_history([0.1, 1000, 0.5, 500, 1, 999])
        confidence = engine.compute_prediction_confidence(history, 1.0)
        assert confidence >= 0.0


# ============================================================================
# Test: evaluate_prediction
# ============================================================================


class TestEvaluatePrediction:
    """Tests for evaluate_prediction."""

    def test_evaluate_on_time(self, engine: RefreshPredictorEngine):
        """Actual within 25% of cadence from predicted -> ON_TIME."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-020", history, cadence_hours=24.0)
        # Actual arrives exactly at predicted time
        result = engine.evaluate_prediction(pred.prediction_id, pred.predicted_at)
        assert result["status"] == PredictionStatus.ON_TIME.value
        assert result["error_hours"] == pytest.approx(0.0, abs=0.01)

    def test_evaluate_on_time_within_threshold(
        self, engine: RefreshPredictorEngine,
    ):
        """Actual within 25% of cadence threshold -> ON_TIME."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-021", history, cadence_hours=24.0)
        # Cadence=24, 25% = 6h; actual arrives 5h late -> ON_TIME
        actual = pred.predicted_at + timedelta(hours=5)
        result = engine.evaluate_prediction(pred.prediction_id, actual)
        assert result["status"] == PredictionStatus.ON_TIME.value

    def test_evaluate_late(self, engine: RefreshPredictorEngine):
        """Actual > predicted, error <= cadence -> LATE."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-022", history, cadence_hours=24.0)
        # Cadence=24, 25%=6h; actual arrives 10h late -> LATE (10 > 6, 10 <= 24)
        actual = pred.predicted_at + timedelta(hours=10)
        result = engine.evaluate_prediction(pred.prediction_id, actual)
        assert result["status"] == PredictionStatus.LATE.value

    def test_evaluate_very_late(self, engine: RefreshPredictorEngine):
        """Actual > predicted, error > cadence -> VERY_LATE."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-023", history, cadence_hours=24.0)
        # Cadence=24; actual arrives 30h late -> VERY_LATE (30 > 24)
        actual = pred.predicted_at + timedelta(hours=30)
        result = engine.evaluate_prediction(pred.prediction_id, actual)
        assert result["status"] == PredictionStatus.VERY_LATE.value

    def test_evaluate_error_hours_correct(
        self, engine: RefreshPredictorEngine,
    ):
        """error_hours matches absolute time difference."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-024", history, cadence_hours=24.0)
        actual = pred.predicted_at + timedelta(hours=15)
        result = engine.evaluate_prediction(pred.prediction_id, actual)
        assert result["error_hours"] == pytest.approx(15.0, abs=0.01)

    def test_evaluate_updates_stats(self, engine: RefreshPredictorEngine):
        """Evaluation increments statistics counters."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-025", history, cadence_hours=24.0)
        engine.evaluate_prediction(pred.prediction_id, pred.predicted_at)
        stats = engine.get_statistics()
        assert stats["predictions_evaluated"] == 1

    def test_evaluate_unknown_id_raises(self, engine: RefreshPredictorEngine):
        """Non-existent prediction_id raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            engine.evaluate_prediction("nonexistent-id", _utcnow())

    def test_evaluate_returns_provenance_hash(
        self, engine: RefreshPredictorEngine,
    ):
        """Evaluation record contains a provenance_hash."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-026", history, cadence_hours=24.0)
        result = engine.evaluate_prediction(pred.prediction_id, pred.predicted_at)
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_evaluate_early_arrival_on_time(
        self, engine: RefreshPredictorEngine,
    ):
        """Actual arrives before predicted but within cadence -> ON_TIME."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-027", history, cadence_hours=24.0)
        # Actual arrives 10h early: error=10, within cadence=24 and actual < predicted
        actual = pred.predicted_at - timedelta(hours=10)
        result = engine.evaluate_prediction(pred.prediction_id, actual)
        assert result["status"] == PredictionStatus.ON_TIME.value

    def test_evaluate_early_arrival_large_error_late(
        self, engine: RefreshPredictorEngine,
    ):
        """Actual arrives way before predicted, error > cadence -> LATE."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-028", history, cadence_hours=24.0)
        # Actual arrives 30h early: error=30 > cadence=24, actual < predicted
        actual = pred.predicted_at - timedelta(hours=30)
        result = engine.evaluate_prediction(pred.prediction_id, actual)
        assert result["status"] == PredictionStatus.LATE.value


# ============================================================================
# Test: detect_anomalous_delay
# ============================================================================


class TestAnomalousDelay:
    """Tests for detect_anomalous_delay."""

    def test_no_anomaly_when_on_time(self, engine: RefreshPredictorEngine):
        """No anomaly when current time is before expected refresh."""
        now = _utcnow()
        last_refresh = now - timedelta(hours=12)
        result = engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0, current_time=now,
        )
        assert result["is_anomalous"] is False
        assert result["delay_hours"] == 0.0

    def test_no_anomaly_small_delay(self, engine: RefreshPredictorEngine):
        """Delay below 1.5x threshold is not anomalous."""
        now = _utcnow()
        last_refresh = now - timedelta(hours=30)  # 6h past 24h cadence
        result = engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0, current_time=now,
        )
        # delay=6h, factor=6/24=0.25 < 1.5 -> not anomalous
        assert result["is_anomalous"] is False

    def test_anomaly_detected(self, engine: RefreshPredictorEngine):
        """Delay > 1.5x cadence is anomalous."""
        now = _utcnow()
        last_refresh = now - timedelta(hours=72)  # 48h past 24h cadence
        result = engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0, current_time=now,
        )
        # delay=48h, factor=48/24=2.0 > 1.5 -> anomalous
        assert result["is_anomalous"] is True
        assert result["delay_hours"] == pytest.approx(48.0, abs=0.01)
        assert result["delay_factor"] == pytest.approx(2.0, abs=0.01)

    def test_anomaly_exactly_at_threshold(
        self, engine: RefreshPredictorEngine,
    ):
        """Delay at exactly 1.5x threshold is NOT anomalous (> not >=)."""
        now = _utcnow()
        # 24h cadence -> expected at +24h -> delay=36h from last means 36-24=12h delay
        # factor = 12/24 = 0.5 < 1.5 -> not anomalous
        # To get factor=1.5 exactly: delay = 1.5 * 24 = 36h past expected
        # total = 24 + 36 = 60h from last refresh
        last_refresh = now - timedelta(hours=60)
        result = engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0, current_time=now,
        )
        # delay=36h, factor=36/24=1.5, NOT > 1.5 -> not anomalous
        assert result["is_anomalous"] is False

    def test_anomaly_just_above_threshold(
        self, engine: RefreshPredictorEngine,
    ):
        """Delay just above 1.5x threshold is anomalous."""
        now = _utcnow()
        last_refresh = now - timedelta(hours=61)
        result = engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0, current_time=now,
        )
        # delay=37h, factor=37/24=1.541 > 1.5 -> anomalous
        assert result["is_anomalous"] is True

    def test_anomaly_increments_stats(self, engine: RefreshPredictorEngine):
        """Anomalous detection increments the anomaly counter."""
        now = _utcnow()
        last_refresh = now - timedelta(hours=100)
        engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0, current_time=now,
        )
        stats = engine.get_statistics()
        assert stats["anomalies_detected"] == 1

    def test_anomaly_provenance_hash(self, engine: RefreshPredictorEngine):
        """Result contains a provenance hash."""
        now = _utcnow()
        result = engine.detect_anomalous_delay(
            now - timedelta(hours=10), cadence_hours=24.0, current_time=now,
        )
        assert len(result["provenance_hash"]) == 64

    def test_anomaly_custom_threshold(
        self, custom_engine: RefreshPredictorEngine,
    ):
        """Custom anomaly_delay_factor=2.0 changes the threshold."""
        now = _utcnow()
        last_refresh = now - timedelta(hours=60)
        # delay=36h, factor=36/24=1.5 < 2.0 -> not anomalous
        result = custom_engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0, current_time=now,
        )
        assert result["is_anomalous"] is False

    def test_anomaly_default_current_time(
        self, engine: RefreshPredictorEngine,
    ):
        """When current_time is None, uses utcnow."""
        last_refresh = _utcnow() - timedelta(hours=100)
        result = engine.detect_anomalous_delay(
            last_refresh, cadence_hours=24.0,
        )
        # Should compute without error
        assert "is_anomalous" in result


# ============================================================================
# Test: predict_batch
# ============================================================================


class TestPredictBatch:
    """Tests for predict_batch."""

    def test_batch_single(self, engine: RefreshPredictorEngine):
        """Single-entry batch returns one prediction."""
        history = _make_history(7, interval_hours=24.0)
        results = engine.predict_batch([
            {"dataset_id": "ds-b1", "refresh_history": history, "cadence_hours": 24.0},
        ])
        assert len(results) == 1
        assert results[0].dataset_id == "ds-b1"

    def test_batch_multiple(self, engine: RefreshPredictorEngine):
        """Multi-entry batch returns one prediction per entry."""
        h1 = _make_history(3, interval_hours=12.0)
        h2 = _make_history(8, interval_hours=24.0)
        h3 = _make_history(15, interval_hours=48.0)
        results = engine.predict_batch([
            {"dataset_id": "ds-b2", "refresh_history": h1, "cadence_hours": 12.0},
            {"dataset_id": "ds-b3", "refresh_history": h2, "cadence_hours": 24.0},
            {"dataset_id": "ds-b4", "refresh_history": h3, "cadence_hours": 48.0},
        ])
        assert len(results) == 3
        assert results[0].dataset_id == "ds-b2"
        assert results[1].dataset_id == "ds-b3"
        assert results[2].dataset_id == "ds-b4"

    def test_batch_empty(self, engine: RefreshPredictorEngine):
        """Empty batch returns empty list."""
        results = engine.predict_batch([])
        assert results == []

    def test_batch_invalid_entry_raises(
        self, engine: RefreshPredictorEngine,
    ):
        """Batch with invalid entry raises ValueError."""
        with pytest.raises(ValueError):
            engine.predict_batch([
                {"dataset_id": "", "refresh_history": [], "cadence_hours": 24.0},
            ])


# ============================================================================
# Test: get_prediction_accuracy
# ============================================================================


class TestPredictionAccuracy:
    """Tests for get_prediction_accuracy."""

    def test_accuracy_no_evaluations(self, engine: RefreshPredictorEngine):
        """No evaluations returns zeroed accuracy dict."""
        acc = engine.get_prediction_accuracy()
        assert acc["evaluated"] == 0
        assert acc["mean_error_hours"] == 0.0
        assert acc["median_error_hours"] == 0.0
        assert acc["by_status"] == {}

    def test_accuracy_after_one_evaluation(
        self, engine: RefreshPredictorEngine,
    ):
        """After one evaluation, accuracy reflects that result."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-acc", history, cadence_hours=24.0)
        actual = pred.predicted_at + timedelta(hours=3)
        engine.evaluate_prediction(pred.prediction_id, actual)
        acc = engine.get_prediction_accuracy()
        assert acc["evaluated"] == 1
        assert acc["total"] >= 1
        assert acc["mean_error_hours"] == pytest.approx(3.0, abs=0.1)

    def test_accuracy_multiple_evaluations(
        self, engine: RefreshPredictorEngine,
    ):
        """Multiple evaluations produce correct aggregate stats."""
        errors = [2.0, 4.0, 6.0]
        for i, err in enumerate(errors):
            history = _make_history(7, interval_hours=24.0)
            pred = engine.predict_next_refresh(
                f"ds-macc-{i}", history, cadence_hours=24.0,
            )
            actual = pred.predicted_at + timedelta(hours=err)
            engine.evaluate_prediction(pred.prediction_id, actual)

        acc = engine.get_prediction_accuracy()
        assert acc["evaluated"] == 3
        assert acc["mean_error_hours"] == pytest.approx(4.0, abs=0.1)
        assert acc["median_error_hours"] == pytest.approx(4.0, abs=0.1)

    def test_accuracy_by_status_counts(
        self, engine: RefreshPredictorEngine,
    ):
        """by_status shows correct counts per status."""
        history = _make_history(7, interval_hours=24.0)
        # ON_TIME: error < 6h (25% of 24h)
        p1 = engine.predict_next_refresh("ds-s1", history, cadence_hours=24.0)
        engine.evaluate_prediction(p1.prediction_id, p1.predicted_at)
        # LATE: actual > predicted, 6 < error <= 24
        p2 = engine.predict_next_refresh("ds-s2", history, cadence_hours=24.0)
        engine.evaluate_prediction(
            p2.prediction_id, p2.predicted_at + timedelta(hours=12),
        )
        acc = engine.get_prediction_accuracy()
        assert PredictionStatus.ON_TIME.value in acc["by_status"]


# ============================================================================
# Test: get_predictions / get_statistics / reset
# ============================================================================


class TestRetrievalAndLifecycle:
    """Tests for retrieval, statistics, and reset."""

    def test_get_predictions_unknown_dataset(
        self, engine: RefreshPredictorEngine,
    ):
        """Unknown dataset_id returns empty list."""
        assert engine.get_predictions("nonexistent") == []

    def test_get_predictions_newest_first(
        self, engine: RefreshPredictorEngine,
    ):
        """Predictions are returned newest-first."""
        for i in range(3):
            engine.predict_next_refresh(
                "ds-order", _make_history(7, interval_hours=24.0),
                cadence_hours=24.0,
            )
        preds = engine.get_predictions("ds-order")
        assert len(preds) == 3
        # Newest first: created_at should be non-increasing
        for i in range(1, len(preds)):
            assert preds[i - 1].created_at >= preds[i].created_at

    def test_get_statistics_initial(self, engine: RefreshPredictorEngine):
        """Initial statistics are all zero."""
        stats = engine.get_statistics()
        assert stats["predictions_created"] == 0
        assert stats["predictions_evaluated"] == 0
        assert stats["anomalies_detected"] == 0
        assert stats["datasets_tracked"] == 0

    def test_get_statistics_after_operations(
        self, engine: RefreshPredictorEngine,
    ):
        """Statistics reflect all operations."""
        history = _make_history(7, interval_hours=24.0)
        engine.predict_next_refresh("ds-st1", history, cadence_hours=24.0)
        engine.predict_next_refresh("ds-st2", history, cadence_hours=24.0)
        stats = engine.get_statistics()
        assert stats["predictions_created"] == 2
        assert stats["datasets_tracked"] == 2
        assert stats["total_predictions_stored"] == 2
        assert "timestamp" in stats

    def test_reset_clears_everything(self, engine: RefreshPredictorEngine):
        """Reset clears all predictions, evaluations, and stats."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh("ds-rst", history, cadence_hours=24.0)
        engine.evaluate_prediction(pred.prediction_id, pred.predicted_at)
        engine.reset()
        assert engine.get_predictions("ds-rst") == []
        stats = engine.get_statistics()
        assert stats["predictions_created"] == 0
        assert stats["predictions_evaluated"] == 0

    def test_prediction_id_is_unique(self, engine: RefreshPredictorEngine):
        """Each prediction has a unique ID."""
        history = _make_history(7, interval_hours=24.0)
        ids = set()
        for _ in range(10):
            pred = engine.predict_next_refresh(
                "ds-uniq", history, cadence_hours=24.0,
            )
            ids.add(pred.prediction_id)
        assert len(ids) == 10

    def test_prediction_id_has_prefix(self, engine: RefreshPredictorEngine):
        """Prediction IDs have the RPR prefix."""
        history = _make_history(7, interval_hours=24.0)
        pred = engine.predict_next_refresh(
            "ds-pfx", history, cadence_hours=24.0,
        )
        assert pred.prediction_id.startswith("RPR-")

    def test_multiple_datasets_tracked_separately(
        self, engine: RefreshPredictorEngine,
    ):
        """Predictions for different datasets are stored separately."""
        history = _make_history(7, interval_hours=24.0)
        engine.predict_next_refresh("ds-A", history, cadence_hours=24.0)
        engine.predict_next_refresh("ds-B", history, cadence_hours=24.0)
        engine.predict_next_refresh("ds-A", history, cadence_hours=24.0)
        assert len(engine.get_predictions("ds-A")) == 2
        assert len(engine.get_predictions("ds-B")) == 1
