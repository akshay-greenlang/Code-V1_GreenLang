# -*- coding: utf-8 -*-
"""
Unit tests for FreshnessCheckerEngine - AGENT-DATA-016 Engine 3.

Tests all 17 public methods of FreshnessCheckerEngine with 85%+ coverage.
Validates freshness scoring algorithm (piecewise-linear 5-tier), SLA
evaluation logic, batch/group operations, check history tracking, stale
dataset identification, SLA compliance summaries, heatmap generation,
statistics aggregation, provenance hashing, and edge cases.

Target: 80+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from greenlang.data_freshness_monitor.freshness_checker import (
    FreshnessCheckerEngine,
    FreshnessCheck,
    FreshnessSummary,
    FreshnessLevel,
    SLAStatus,
)
from greenlang.data_freshness_monitor.config import DataFreshnessMonitorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _val(v) -> str:
    """Extract lowercase string value from enum or plain string."""
    if hasattr(v, "value"):
        return str(v.value).lower()
    return str(v).lower()


def _now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _ago(hours: float) -> datetime:
    """Return a UTC datetime that is ``hours`` ago from now."""
    return _now() - timedelta(hours=hours)


def _make_check_entry(
    dataset_id: str,
    hours_ago: float,
    sla_warn: float = 24.0,
    sla_crit: float = 72.0,
    weight: float = 1.0,
) -> Dict[str, Any]:
    """Build a dictionary suitable for batch_check or check_dataset_group."""
    return {
        "dataset_id": dataset_id,
        "last_refreshed_at": _ago(hours_ago),
        "sla_warning_hours": sla_warn,
        "sla_critical_hours": sla_crit,
        "weight": weight,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> DataFreshnessMonitorConfig:
    """Create a default DataFreshnessMonitorConfig for tests."""
    return DataFreshnessMonitorConfig()


@pytest.fixture
def engine(config: DataFreshnessMonitorConfig) -> FreshnessCheckerEngine:
    """Create a fresh FreshnessCheckerEngine instance for each test."""
    return FreshnessCheckerEngine(config=config)


@pytest.fixture
def populated_engine(engine: FreshnessCheckerEngine) -> FreshnessCheckerEngine:
    """Engine pre-populated with checks across all 5 freshness tiers."""
    # EXCELLENT: 0.5h ago
    engine.check_freshness("ds-excellent", _ago(0.5), 24.0, 72.0)
    # GOOD: 3h ago
    engine.check_freshness("ds-good", _ago(3.0), 24.0, 72.0)
    # FAIR: 12h ago
    engine.check_freshness("ds-fair", _ago(12.0), 24.0, 72.0)
    # POOR: 48h ago (warning SLA)
    engine.check_freshness("ds-poor", _ago(48.0), 24.0, 72.0)
    # STALE: 100h ago (breached SLA)
    engine.check_freshness("ds-stale", _ago(100.0), 24.0, 72.0)
    return engine


# ===========================================================================
# 1. __init__
# ===========================================================================


class TestInit:
    """Tests for FreshnessCheckerEngine.__init__."""

    def test_init_with_config(self, config: DataFreshnessMonitorConfig):
        """Engine initializes correctly with an explicit config."""
        eng = FreshnessCheckerEngine(config=config)
        assert eng._config is config
        assert eng.get_check_count() == 0
        assert eng._check_history == {}

    def test_init_default_config(self):
        """Engine initializes with get_config() when config is None."""
        eng = FreshnessCheckerEngine()
        assert eng._config is not None
        assert eng.get_check_count() == 0

    def test_init_lock_exists(self, engine: FreshnessCheckerEngine):
        """Engine has a threading lock for concurrency safety."""
        assert engine._lock is not None


# ===========================================================================
# 2. check_freshness
# ===========================================================================


class TestCheckFreshness:
    """Tests for FreshnessCheckerEngine.check_freshness."""

    def test_basic_check_returns_freshness_check(self, engine: FreshnessCheckerEngine):
        """A basic check returns a FreshnessCheck dataclass instance."""
        result = engine.check_freshness("ds-001", _ago(0.5), 24.0, 72.0)
        assert isinstance(result, FreshnessCheck)

    def test_check_id_prefix(self, engine: FreshnessCheckerEngine):
        """Check ID starts with FC- prefix."""
        result = engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert result.check_id.startswith("FC-")
        assert len(result.check_id) == 15  # FC- + 12 hex chars

    def test_dataset_id_preserved(self, engine: FreshnessCheckerEngine):
        """Dataset ID is stored in the result."""
        result = engine.check_freshness("my-dataset", _ago(2.0), 24.0, 72.0)
        assert result.dataset_id == "my-dataset"

    def test_provenance_hash_is_64_chars(self, engine: FreshnessCheckerEngine):
        """Provenance hash is a 64-character SHA-256 hex digest."""
        result = engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_sla_thresholds_stored(self, engine: FreshnessCheckerEngine):
        """SLA warning and critical hours are preserved in the result."""
        result = engine.check_freshness("ds-001", _ago(5.0), 12.0, 36.0)
        assert result.sla_warning_hours == 12.0
        assert result.sla_critical_hours == 36.0

    def test_age_hours_nonnegative(self, engine: FreshnessCheckerEngine):
        """Age in hours is always non-negative."""
        result = engine.check_freshness("ds-001", _ago(10.0), 24.0, 72.0)
        assert result.age_hours >= 0.0

    def test_freshness_score_in_range(self, engine: FreshnessCheckerEngine):
        """Freshness score is between 0.0 and 1.0."""
        result = engine.check_freshness("ds-001", _ago(50.0), 24.0, 72.0)
        assert 0.0 <= result.freshness_score <= 1.0

    def test_check_increments_count(self, engine: FreshnessCheckerEngine):
        """Each check increments the total check count."""
        assert engine.get_check_count() == 0
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert engine.get_check_count() == 1
        engine.check_freshness("ds-002", _ago(2.0), 24.0, 72.0)
        assert engine.get_check_count() == 2

    def test_check_stores_in_history(self, engine: FreshnessCheckerEngine):
        """Check results are stored in the per-dataset history."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        history = engine.get_check_history("ds-001")
        assert len(history) == 1

    def test_checked_at_is_iso_string(self, engine: FreshnessCheckerEngine):
        """The checked_at field is an ISO 8601 string."""
        result = engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert isinstance(result.checked_at, str)
        # Should parse without error
        datetime.fromisoformat(result.checked_at)

    def test_last_refreshed_at_is_iso_string(self, engine: FreshnessCheckerEngine):
        """The last_refreshed_at field is an ISO 8601 string."""
        result = engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert isinstance(result.last_refreshed_at, str)
        datetime.fromisoformat(result.last_refreshed_at)

    # --- Validation errors ---

    def test_empty_dataset_id_raises(self, engine: FreshnessCheckerEngine):
        """Empty dataset_id raises ValueError."""
        with pytest.raises(ValueError, match="dataset_id must not be empty"):
            engine.check_freshness("", _ago(1.0), 24.0, 72.0)

    def test_whitespace_dataset_id_raises(self, engine: FreshnessCheckerEngine):
        """Whitespace-only dataset_id raises ValueError."""
        with pytest.raises(ValueError, match="dataset_id must not be empty"):
            engine.check_freshness("   ", _ago(1.0), 24.0, 72.0)

    def test_zero_warning_hours_raises(self, engine: FreshnessCheckerEngine):
        """sla_warning_hours=0 raises ValueError."""
        with pytest.raises(ValueError, match="sla_warning_hours must be > 0"):
            engine.check_freshness("ds-001", _ago(1.0), 0.0, 72.0)

    def test_negative_warning_hours_raises(self, engine: FreshnessCheckerEngine):
        """Negative sla_warning_hours raises ValueError."""
        with pytest.raises(ValueError, match="sla_warning_hours must be > 0"):
            engine.check_freshness("ds-001", _ago(1.0), -5.0, 72.0)

    def test_zero_critical_hours_raises(self, engine: FreshnessCheckerEngine):
        """sla_critical_hours=0 raises ValueError."""
        with pytest.raises(ValueError, match="sla_critical_hours must be > 0"):
            engine.check_freshness("ds-001", _ago(1.0), 24.0, 0.0)

    def test_negative_critical_hours_raises(self, engine: FreshnessCheckerEngine):
        """Negative sla_critical_hours raises ValueError."""
        with pytest.raises(ValueError, match="sla_critical_hours must be > 0"):
            engine.check_freshness("ds-001", _ago(1.0), 24.0, -10.0)

    def test_warning_equals_critical_raises(self, engine: FreshnessCheckerEngine):
        """sla_warning_hours == sla_critical_hours raises ValueError."""
        with pytest.raises(ValueError, match="sla_warning_hours must be < sla_critical_hours"):
            engine.check_freshness("ds-001", _ago(1.0), 48.0, 48.0)

    def test_warning_greater_than_critical_raises(self, engine: FreshnessCheckerEngine):
        """sla_warning_hours > sla_critical_hours raises ValueError."""
        with pytest.raises(ValueError, match="sla_warning_hours must be < sla_critical_hours"):
            engine.check_freshness("ds-001", _ago(1.0), 72.0, 24.0)

    def test_to_dict_serialization(self, engine: FreshnessCheckerEngine):
        """FreshnessCheck.to_dict() returns a valid dictionary."""
        result = engine.check_freshness("ds-001", _ago(3.0), 24.0, 72.0)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["dataset_id"] == "ds-001"
        assert "check_id" in d
        assert "provenance_hash" in d


# ===========================================================================
# 3. batch_check
# ===========================================================================


class TestBatchCheck:
    """Tests for FreshnessCheckerEngine.batch_check."""

    def test_batch_check_empty_list(self, engine: FreshnessCheckerEngine):
        """Batch check with empty list returns empty results."""
        results = engine.batch_check([])
        assert results == []
        assert engine.get_check_count() == 0

    def test_batch_check_single_entry(self, engine: FreshnessCheckerEngine):
        """Batch check with one entry returns one result."""
        entries = [_make_check_entry("ds-001", 5.0)]
        results = engine.batch_check(entries)
        assert len(results) == 1
        assert results[0].dataset_id == "ds-001"

    def test_batch_check_multiple_entries(self, engine: FreshnessCheckerEngine):
        """Batch check with multiple entries preserves order."""
        entries = [
            _make_check_entry("ds-a", 1.0),
            _make_check_entry("ds-b", 10.0),
            _make_check_entry("ds-c", 50.0),
        ]
        results = engine.batch_check(entries)
        assert len(results) == 3
        assert results[0].dataset_id == "ds-a"
        assert results[1].dataset_id == "ds-b"
        assert results[2].dataset_id == "ds-c"

    def test_batch_check_increments_count(self, engine: FreshnessCheckerEngine):
        """Batch check increments the total count by number of entries."""
        entries = [_make_check_entry(f"ds-{i}", float(i)) for i in range(5)]
        engine.batch_check(entries)
        assert engine.get_check_count() == 5

    def test_batch_check_invalid_entry_raises(self, engine: FreshnessCheckerEngine):
        """Batch check propagates ValueError from invalid entries."""
        entries = [
            _make_check_entry("ds-a", 1.0),
            {"dataset_id": "", "last_refreshed_at": _ago(1.0),
             "sla_warning_hours": 24.0, "sla_critical_hours": 72.0},
        ]
        with pytest.raises(ValueError, match="dataset_id must not be empty"):
            engine.batch_check(entries)


# ===========================================================================
# 4. compute_age_hours
# ===========================================================================


class TestComputeAgeHours:
    """Tests for FreshnessCheckerEngine.compute_age_hours."""

    def test_age_zero(self, engine: FreshnessCheckerEngine):
        """Age of a just-refreshed dataset is approximately zero."""
        now = _now()
        age = engine.compute_age_hours(now, now)
        assert age == pytest.approx(0.0, abs=0.01)

    def test_age_positive(self, engine: FreshnessCheckerEngine):
        """Age is correctly computed for a dataset refreshed hours ago."""
        now = _now()
        age = engine.compute_age_hours(now - timedelta(hours=5.0), now)
        assert age == pytest.approx(5.0, abs=0.01)

    def test_age_fractional(self, engine: FreshnessCheckerEngine):
        """Age computation handles fractional hours."""
        now = _now()
        age = engine.compute_age_hours(now - timedelta(minutes=30), now)
        assert age == pytest.approx(0.5, abs=0.01)

    def test_future_timestamp_clamps_to_zero(self, engine: FreshnessCheckerEngine):
        """Future last_refreshed_at clamps age to 0.0."""
        now = _now()
        future = now + timedelta(hours=2.0)
        age = engine.compute_age_hours(future, now)
        assert age == 0.0

    def test_naive_datetime_assumed_utc(self, engine: FreshnessCheckerEngine):
        """Timezone-naive datetimes are treated as UTC."""
        now_utc = _now()
        naive_past = now_utc.replace(tzinfo=None) - timedelta(hours=3.0)
        age = engine.compute_age_hours(naive_past, now_utc)
        assert age == pytest.approx(3.0, abs=0.05)

    def test_both_naive_datetimes(self, engine: FreshnessCheckerEngine):
        """Two naive datetimes are both treated as UTC."""
        now_naive = datetime.utcnow()
        past_naive = now_naive - timedelta(hours=10.0)
        age = engine.compute_age_hours(past_naive, now_naive)
        assert age == pytest.approx(10.0, abs=0.01)

    def test_large_age(self, engine: FreshnessCheckerEngine):
        """Extremely old datasets produce large age values."""
        now = _now()
        ancient = now - timedelta(days=365)
        age = engine.compute_age_hours(ancient, now)
        assert age == pytest.approx(365 * 24, abs=1.0)

    def test_default_current_time_uses_utcnow(self, engine: FreshnessCheckerEngine):
        """When current_time is None, compute_age_hours uses UTC now."""
        past = _now() - timedelta(hours=2.0)
        age = engine.compute_age_hours(past)
        assert age == pytest.approx(2.0, abs=0.1)


# ===========================================================================
# 5. compute_freshness_score
# ===========================================================================


class TestComputeFreshnessScore:
    """Tests for the piecewise-linear 5-tier freshness scoring algorithm.

    Default config boundaries:
      excellent: 1.0h, good: 6.0h, fair: 24.0h, poor: 72.0h
    """

    # --- Tier 1: EXCELLENT (age <= 1.0h) -> score = 1.0 ---

    def test_score_at_zero_hours(self, engine: FreshnessCheckerEngine):
        """Age 0h yields a perfect score of 1.0."""
        assert engine.compute_freshness_score(0.0) == 1.0

    def test_score_at_half_hour(self, engine: FreshnessCheckerEngine):
        """Age 0.5h is in EXCELLENT tier, score = 1.0."""
        assert engine.compute_freshness_score(0.5) == 1.0

    def test_score_at_excellent_boundary(self, engine: FreshnessCheckerEngine):
        """Age exactly at excellent boundary (1.0h) is EXCELLENT, score = 1.0."""
        assert engine.compute_freshness_score(1.0) == 1.0

    # --- Tier 2: GOOD (1.0 < age <= 6.0h) -> linear 1.0 -> 0.85 ---

    def test_score_just_above_excellent(self, engine: FreshnessCheckerEngine):
        """Age just above excellent boundary starts GOOD tier interpolation."""
        score = engine.compute_freshness_score(1.001)
        assert score < 1.0
        assert score > 0.85

    def test_score_at_good_midpoint(self, engine: FreshnessCheckerEngine):
        """Age at midpoint of GOOD tier (3.5h) is linearly interpolated."""
        # Midpoint between 1.0h and 6.0h = 3.5h
        # fraction = (3.5-1)/(6-1) = 2.5/5 = 0.5
        # score = 1.0 + (0.85-1.0)*0.5 = 1.0 - 0.075 = 0.925
        score = engine.compute_freshness_score(3.5)
        assert score == pytest.approx(0.925, abs=1e-6)

    def test_score_at_good_boundary(self, engine: FreshnessCheckerEngine):
        """Age at good boundary (6.0h) yields score = 0.85."""
        score = engine.compute_freshness_score(6.0)
        assert score == pytest.approx(0.85, abs=1e-6)

    # --- Tier 3: FAIR (6.0 < age <= 24.0h) -> linear 0.85 -> 0.70 ---

    def test_score_just_above_good(self, engine: FreshnessCheckerEngine):
        """Age just above good boundary starts FAIR tier interpolation."""
        score = engine.compute_freshness_score(6.001)
        assert score < 0.85
        assert score > 0.70

    def test_score_at_fair_midpoint(self, engine: FreshnessCheckerEngine):
        """Age at midpoint of FAIR tier (15.0h) is linearly interpolated."""
        # fraction = (15-6)/(24-6) = 9/18 = 0.5
        # score = 0.85 + (0.70-0.85)*0.5 = 0.85 - 0.075 = 0.775
        score = engine.compute_freshness_score(15.0)
        assert score == pytest.approx(0.775, abs=1e-6)

    def test_score_at_fair_boundary(self, engine: FreshnessCheckerEngine):
        """Age at fair boundary (24.0h) yields score = 0.70."""
        score = engine.compute_freshness_score(24.0)
        assert score == pytest.approx(0.70, abs=1e-6)

    # --- Tier 4: POOR (24.0 < age <= 72.0h) -> linear 0.70 -> 0.50 ---

    def test_score_just_above_fair(self, engine: FreshnessCheckerEngine):
        """Age just above fair boundary starts POOR tier interpolation."""
        score = engine.compute_freshness_score(24.001)
        assert score < 0.70
        assert score > 0.50

    def test_score_at_poor_midpoint(self, engine: FreshnessCheckerEngine):
        """Age at midpoint of POOR tier (48.0h) is linearly interpolated."""
        # fraction = (48-24)/(72-24) = 24/48 = 0.5
        # score = 0.70 + (0.50-0.70)*0.5 = 0.70 - 0.10 = 0.60
        score = engine.compute_freshness_score(48.0)
        assert score == pytest.approx(0.60, abs=1e-6)

    def test_score_at_poor_boundary(self, engine: FreshnessCheckerEngine):
        """Age at poor boundary (72.0h) yields score = 0.50."""
        score = engine.compute_freshness_score(72.0)
        assert score == pytest.approx(0.50, abs=1e-6)

    # --- Tier 5: STALE (age > 72.0h) -> decays toward 0.0 ---

    def test_score_just_above_poor(self, engine: FreshnessCheckerEngine):
        """Age just above poor boundary enters STALE decay."""
        score = engine.compute_freshness_score(72.001)
        assert score < 0.50
        assert score > 0.0

    def test_score_at_stale_decay(self, engine: FreshnessCheckerEngine):
        """Age 144h (poor*2 above poor) decays score to 0.25.

        Formula: 0.50 - (144-72)/(72*2) * 0.50 = 0.50 - 72/144*0.50 = 0.50 - 0.25 = 0.25
        """
        score = engine.compute_freshness_score(144.0)
        assert score == pytest.approx(0.25, abs=1e-6)

    def test_score_at_full_decay(self, engine: FreshnessCheckerEngine):
        """Age at full decay point (72 + 72*2 = 216h) yields score = 0.0.

        Formula: 0.50 - (216-72)/(72*2)*0.50 = 0.50 - 144/144*0.50 = 0.0
        """
        score = engine.compute_freshness_score(216.0)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_score_beyond_full_decay_clamped(self, engine: FreshnessCheckerEngine):
        """Age far beyond full decay is clamped to 0.0."""
        score = engine.compute_freshness_score(1000.0)
        assert score == 0.0

    def test_negative_age_clamped_to_zero(self, engine: FreshnessCheckerEngine):
        """Negative age_hours is clamped to 0 and scores 1.0."""
        score = engine.compute_freshness_score(-5.0)
        assert score == 1.0


# ===========================================================================
# 6. classify_freshness_level
# ===========================================================================


class TestClassifyFreshnessLevel:
    """Tests for FreshnessCheckerEngine.classify_freshness_level."""

    @pytest.mark.parametrize("age,expected_level", [
        (0.0, FreshnessLevel.EXCELLENT),
        (0.5, FreshnessLevel.EXCELLENT),
        (1.0, FreshnessLevel.EXCELLENT),
    ])
    def test_excellent_tier(self, engine: FreshnessCheckerEngine, age, expected_level):
        """Ages within [0, 1.0h] are classified as EXCELLENT."""
        assert engine.classify_freshness_level(age) == expected_level

    @pytest.mark.parametrize("age,expected_level", [
        (1.001, FreshnessLevel.GOOD),
        (3.0, FreshnessLevel.GOOD),
        (6.0, FreshnessLevel.GOOD),
    ])
    def test_good_tier(self, engine: FreshnessCheckerEngine, age, expected_level):
        """Ages within (1.0h, 6.0h] are classified as GOOD."""
        assert engine.classify_freshness_level(age) == expected_level

    @pytest.mark.parametrize("age,expected_level", [
        (6.001, FreshnessLevel.FAIR),
        (15.0, FreshnessLevel.FAIR),
        (24.0, FreshnessLevel.FAIR),
    ])
    def test_fair_tier(self, engine: FreshnessCheckerEngine, age, expected_level):
        """Ages within (6.0h, 24.0h] are classified as FAIR."""
        assert engine.classify_freshness_level(age) == expected_level

    @pytest.mark.parametrize("age,expected_level", [
        (24.001, FreshnessLevel.POOR),
        (48.0, FreshnessLevel.POOR),
        (72.0, FreshnessLevel.POOR),
    ])
    def test_poor_tier(self, engine: FreshnessCheckerEngine, age, expected_level):
        """Ages within (24.0h, 72.0h] are classified as POOR."""
        assert engine.classify_freshness_level(age) == expected_level

    @pytest.mark.parametrize("age,expected_level", [
        (72.001, FreshnessLevel.STALE),
        (100.0, FreshnessLevel.STALE),
        (1000.0, FreshnessLevel.STALE),
    ])
    def test_stale_tier(self, engine: FreshnessCheckerEngine, age, expected_level):
        """Ages beyond 72.0h are classified as STALE."""
        assert engine.classify_freshness_level(age) == expected_level

    def test_negative_age_classified_as_excellent(self, engine: FreshnessCheckerEngine):
        """Negative age_hours is clamped to 0 and classified as EXCELLENT."""
        assert engine.classify_freshness_level(-10.0) == FreshnessLevel.EXCELLENT


# ===========================================================================
# 7. evaluate_sla_status
# ===========================================================================


class TestEvaluateSLAStatus:
    """Tests for FreshnessCheckerEngine.evaluate_sla_status."""

    def test_compliant_well_below_warning(self, engine: FreshnessCheckerEngine):
        """Age well below warning threshold is COMPLIANT."""
        assert engine.evaluate_sla_status(5.0, 24.0, 72.0) == SLAStatus.COMPLIANT

    def test_compliant_just_below_warning(self, engine: FreshnessCheckerEngine):
        """Age just below warning threshold is COMPLIANT."""
        assert engine.evaluate_sla_status(23.999, 24.0, 72.0) == SLAStatus.COMPLIANT

    def test_warning_at_exact_warning_threshold(self, engine: FreshnessCheckerEngine):
        """Age exactly at warning threshold is WARNING."""
        assert engine.evaluate_sla_status(24.0, 24.0, 72.0) == SLAStatus.WARNING

    def test_warning_between_thresholds(self, engine: FreshnessCheckerEngine):
        """Age between warning and critical thresholds is WARNING."""
        assert engine.evaluate_sla_status(50.0, 24.0, 72.0) == SLAStatus.WARNING

    def test_warning_just_below_critical(self, engine: FreshnessCheckerEngine):
        """Age just below critical threshold is WARNING."""
        assert engine.evaluate_sla_status(71.999, 24.0, 72.0) == SLAStatus.WARNING

    def test_breached_at_exact_critical_threshold(self, engine: FreshnessCheckerEngine):
        """Age exactly at critical threshold is BREACHED."""
        assert engine.evaluate_sla_status(72.0, 24.0, 72.0) == SLAStatus.BREACHED

    def test_breached_above_critical(self, engine: FreshnessCheckerEngine):
        """Age above critical threshold is BREACHED."""
        assert engine.evaluate_sla_status(100.0, 24.0, 72.0) == SLAStatus.BREACHED

    def test_negative_age_is_compliant(self, engine: FreshnessCheckerEngine):
        """Negative age clamps to 0 and is COMPLIANT."""
        assert engine.evaluate_sla_status(-5.0, 24.0, 72.0) == SLAStatus.COMPLIANT

    def test_zero_age_is_compliant(self, engine: FreshnessCheckerEngine):
        """Zero age is always COMPLIANT."""
        assert engine.evaluate_sla_status(0.0, 1.0, 2.0) == SLAStatus.COMPLIANT


# ===========================================================================
# 8. check_dataset_group
# ===========================================================================


class TestCheckDatasetGroup:
    """Tests for FreshnessCheckerEngine.check_dataset_group."""

    def test_group_check_returns_expected_keys(self, engine: FreshnessCheckerEngine):
        """Group check result contains all required keys."""
        group = [_make_check_entry("ds-001", 1.0)]
        result = engine.check_dataset_group(group)
        assert "checks" in result
        assert "group_freshness_score" in result
        assert "worst_sla_status" in result
        assert "total_datasets" in result
        assert "stale_count" in result
        assert "provenance_hash" in result

    def test_group_total_datasets(self, engine: FreshnessCheckerEngine):
        """total_datasets matches the number of input entries."""
        group = [
            _make_check_entry("ds-a", 1.0),
            _make_check_entry("ds-b", 10.0),
            _make_check_entry("ds-c", 50.0),
        ]
        result = engine.check_dataset_group(group)
        assert result["total_datasets"] == 3

    def test_group_stale_count(self, engine: FreshnessCheckerEngine):
        """stale_count correctly counts datasets with STALE freshness level."""
        group = [
            _make_check_entry("ds-fresh", 0.5),    # EXCELLENT
            _make_check_entry("ds-stale1", 100.0),  # STALE
            _make_check_entry("ds-stale2", 200.0),  # STALE
        ]
        result = engine.check_dataset_group(group)
        assert result["stale_count"] == 2

    def test_group_worst_sla_breached(self, engine: FreshnessCheckerEngine):
        """worst_sla_status is BREACHED when any dataset is breached."""
        group = [
            _make_check_entry("ds-ok", 1.0),
            _make_check_entry("ds-breached", 100.0),
        ]
        result = engine.check_dataset_group(group)
        assert result["worst_sla_status"] == SLAStatus.BREACHED.value

    def test_group_worst_sla_warning(self, engine: FreshnessCheckerEngine):
        """worst_sla_status is WARNING when the worst is warning."""
        group = [
            _make_check_entry("ds-ok", 1.0, sla_warn=24.0, sla_crit=72.0),
            _make_check_entry("ds-warn", 30.0, sla_warn=24.0, sla_crit=72.0),
        ]
        result = engine.check_dataset_group(group)
        assert result["worst_sla_status"] == SLAStatus.WARNING.value

    def test_group_worst_sla_all_compliant(self, engine: FreshnessCheckerEngine):
        """worst_sla_status is COMPLIANT when all datasets are compliant."""
        group = [
            _make_check_entry("ds-a", 1.0),
            _make_check_entry("ds-b", 2.0),
        ]
        result = engine.check_dataset_group(group)
        assert result["worst_sla_status"] == SLAStatus.COMPLIANT.value

    def test_group_provenance_hash_is_64_chars(self, engine: FreshnessCheckerEngine):
        """Group provenance hash is a valid 64-character SHA-256 hex digest."""
        group = [_make_check_entry("ds-001", 5.0)]
        result = engine.check_dataset_group(group)
        assert len(result["provenance_hash"]) == 64

    def test_group_with_weights(self, engine: FreshnessCheckerEngine):
        """Group check applies custom weights to freshness score."""
        group = [
            _make_check_entry("ds-a", 0.5, weight=3.0),  # EXCELLENT, score=1.0
            _make_check_entry("ds-b", 100.0, weight=1.0), # STALE, low score
        ]
        result = engine.check_dataset_group(group)
        # Weighted toward the excellent dataset
        assert result["group_freshness_score"] > 0.5

    def test_group_empty_list(self, engine: FreshnessCheckerEngine):
        """Group check with empty list returns zero total and score 0.0."""
        result = engine.check_dataset_group([])
        assert result["total_datasets"] == 0
        assert result["group_freshness_score"] == 0.0
        assert result["stale_count"] == 0


# ===========================================================================
# 9. compute_group_freshness_score
# ===========================================================================


class TestComputeGroupFreshnessScore:
    """Tests for FreshnessCheckerEngine.compute_group_freshness_score."""

    def test_empty_checks_returns_zero(self, engine: FreshnessCheckerEngine):
        """Empty check list returns 0.0."""
        assert engine.compute_group_freshness_score([]) == 0.0

    def test_equal_weight_average(self, engine: FreshnessCheckerEngine):
        """Without explicit weights, all checks are weighted equally."""
        c1 = FreshnessCheck(dataset_id="ds-a", freshness_score=1.0)
        c2 = FreshnessCheck(dataset_id="ds-b", freshness_score=0.5)
        score = engine.compute_group_freshness_score([c1, c2])
        assert score == pytest.approx(0.75, abs=1e-6)

    def test_weighted_average(self, engine: FreshnessCheckerEngine):
        """Explicit weights produce a correct weighted average."""
        c1 = FreshnessCheck(dataset_id="ds-a", freshness_score=1.0)
        c2 = FreshnessCheck(dataset_id="ds-b", freshness_score=0.0)
        weights = {"ds-a": 3.0, "ds-b": 1.0}
        score = engine.compute_group_freshness_score([c1, c2], weights)
        # (1.0*3 + 0.0*1) / (3+1) = 0.75
        assert score == pytest.approx(0.75, abs=1e-6)

    def test_zero_weight_excluded(self, engine: FreshnessCheckerEngine):
        """Datasets with weight <= 0 are excluded from the average."""
        c1 = FreshnessCheck(dataset_id="ds-a", freshness_score=1.0)
        c2 = FreshnessCheck(dataset_id="ds-b", freshness_score=0.0)
        weights = {"ds-a": 1.0, "ds-b": 0.0}
        score = engine.compute_group_freshness_score([c1, c2], weights)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_negative_weight_excluded(self, engine: FreshnessCheckerEngine):
        """Datasets with negative weight are excluded from the average."""
        c1 = FreshnessCheck(dataset_id="ds-a", freshness_score=0.8)
        c2 = FreshnessCheck(dataset_id="ds-b", freshness_score=0.2)
        weights = {"ds-a": 2.0, "ds-b": -1.0}
        score = engine.compute_group_freshness_score([c1, c2], weights)
        assert score == pytest.approx(0.8, abs=1e-6)

    def test_all_zero_weights_returns_zero(self, engine: FreshnessCheckerEngine):
        """When all weights are zero, the group score is 0.0."""
        c1 = FreshnessCheck(dataset_id="ds-a", freshness_score=1.0)
        weights = {"ds-a": 0.0}
        score = engine.compute_group_freshness_score([c1], weights)
        assert score == 0.0

    def test_single_check_returns_its_score(self, engine: FreshnessCheckerEngine):
        """A single check returns its own freshness score."""
        c = FreshnessCheck(dataset_id="ds-x", freshness_score=0.42)
        score = engine.compute_group_freshness_score([c])
        assert score == pytest.approx(0.42, abs=1e-6)


# ===========================================================================
# 10. get_check_history
# ===========================================================================


class TestGetCheckHistory:
    """Tests for FreshnessCheckerEngine.get_check_history."""

    def test_no_history_returns_empty(self, engine: FreshnessCheckerEngine):
        """Non-existent dataset returns an empty list."""
        assert engine.get_check_history("nonexistent") == []

    def test_history_ordering_most_recent_first(self, engine: FreshnessCheckerEngine):
        """History is returned most recent first."""
        engine.check_freshness("ds-001", _ago(10.0), 24.0, 72.0)
        engine.check_freshness("ds-001", _ago(5.0), 24.0, 72.0)
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        history = engine.get_check_history("ds-001")
        assert len(history) == 3
        # Most recent (age ~1h) should be first
        assert history[0].age_hours < history[1].age_hours
        assert history[1].age_hours < history[2].age_hours

    def test_history_limit(self, engine: FreshnessCheckerEngine):
        """History respects the limit parameter."""
        for i in range(10):
            engine.check_freshness("ds-001", _ago(float(i + 1)), 24.0, 72.0)
        history = engine.get_check_history("ds-001", limit=3)
        assert len(history) == 3

    def test_history_default_limit_100(self, engine: FreshnessCheckerEngine):
        """Default limit is 100."""
        for i in range(5):
            engine.check_freshness("ds-001", _ago(float(i + 1)), 24.0, 72.0)
        history = engine.get_check_history("ds-001")
        assert len(history) == 5  # less than limit, so returns all

    def test_history_isolation_between_datasets(self, engine: FreshnessCheckerEngine):
        """History for one dataset does not leak into another."""
        engine.check_freshness("ds-a", _ago(1.0), 24.0, 72.0)
        engine.check_freshness("ds-b", _ago(2.0), 24.0, 72.0)
        assert len(engine.get_check_history("ds-a")) == 1
        assert len(engine.get_check_history("ds-b")) == 1


# ===========================================================================
# 11. get_latest_check
# ===========================================================================


class TestGetLatestCheck:
    """Tests for FreshnessCheckerEngine.get_latest_check."""

    def test_no_checks_returns_none(self, engine: FreshnessCheckerEngine):
        """Returns None for a dataset with no checks."""
        assert engine.get_latest_check("nonexistent") is None

    def test_returns_most_recent_check(self, engine: FreshnessCheckerEngine):
        """Returns the most recent check for a dataset."""
        engine.check_freshness("ds-001", _ago(10.0), 24.0, 72.0)
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        latest = engine.get_latest_check("ds-001")
        assert latest is not None
        # The second check (age ~1h) should be returned as latest
        assert latest.age_hours < 2.0

    def test_returns_freshness_check_instance(self, engine: FreshnessCheckerEngine):
        """Returned object is a FreshnessCheck instance."""
        engine.check_freshness("ds-001", _ago(3.0), 24.0, 72.0)
        latest = engine.get_latest_check("ds-001")
        assert isinstance(latest, FreshnessCheck)


# ===========================================================================
# 12. get_stale_datasets
# ===========================================================================


class TestGetStaleDatasets:
    """Tests for FreshnessCheckerEngine.get_stale_datasets."""

    def test_no_stale_returns_empty(self, engine: FreshnessCheckerEngine):
        """Returns empty list when no datasets exceed the threshold."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        stale = engine.get_stale_datasets(48.0)
        assert stale == []

    def test_identifies_stale_datasets(self, populated_engine: FreshnessCheckerEngine):
        """Returns datasets whose latest age exceeds the threshold."""
        stale = populated_engine.get_stale_datasets(48.0)
        stale_ids = [s.dataset_id for s in stale]
        assert "ds-stale" in stale_ids
        assert "ds-poor" not in stale_ids  # 48h is not > 48h
        assert "ds-excellent" not in stale_ids

    def test_sorted_by_age_descending(self, engine: FreshnessCheckerEngine):
        """Results are sorted by age_hours descending (stalest first)."""
        engine.check_freshness("ds-a", _ago(100.0), 24.0, 72.0)
        engine.check_freshness("ds-b", _ago(200.0), 24.0, 72.0)
        engine.check_freshness("ds-c", _ago(150.0), 24.0, 72.0)
        stale = engine.get_stale_datasets(50.0)
        assert len(stale) == 3
        assert stale[0].age_hours >= stale[1].age_hours >= stale[2].age_hours

    def test_returns_freshness_summary_instances(self, engine: FreshnessCheckerEngine):
        """Results are FreshnessSummary instances."""
        engine.check_freshness("ds-001", _ago(100.0), 24.0, 72.0)
        stale = engine.get_stale_datasets(50.0)
        assert len(stale) == 1
        assert isinstance(stale[0], FreshnessSummary)

    def test_summary_fields_populated(self, engine: FreshnessCheckerEngine):
        """FreshnessSummary has all expected fields populated."""
        engine.check_freshness("ds-001", _ago(100.0), 24.0, 72.0)
        stale = engine.get_stale_datasets(50.0)
        s = stale[0]
        assert s.dataset_id == "ds-001"
        assert s.age_hours > 50.0
        assert 0.0 <= s.freshness_score <= 1.0
        assert s.freshness_level is not None
        assert s.sla_status is not None
        assert s.last_checked_at is not None

    def test_threshold_zero_returns_all(self, engine: FreshnessCheckerEngine):
        """Threshold of 0 returns all datasets with age > 0."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        stale = engine.get_stale_datasets(0.0)
        assert len(stale) == 1

    def test_freshness_summary_to_dict(self, engine: FreshnessCheckerEngine):
        """FreshnessSummary.to_dict() returns a valid dictionary."""
        engine.check_freshness("ds-001", _ago(100.0), 24.0, 72.0)
        stale = engine.get_stale_datasets(50.0)
        d = stale[0].to_dict()
        assert isinstance(d, dict)
        assert "dataset_id" in d


# ===========================================================================
# 13. get_sla_compliance_summary
# ===========================================================================


class TestGetSLAComplianceSummary:
    """Tests for FreshnessCheckerEngine.get_sla_compliance_summary."""

    def test_empty_engine_returns_zeros(self, engine: FreshnessCheckerEngine):
        """Empty engine returns total=0 and all counts zero."""
        summary = engine.get_sla_compliance_summary()
        assert summary["total"] == 0
        assert summary["compliant"] == 0
        assert summary["warning"] == 0
        assert summary["breached"] == 0

    def test_all_compliant(self, engine: FreshnessCheckerEngine):
        """All datasets compliant yields 100% compliant."""
        engine.check_freshness("ds-a", _ago(1.0), 24.0, 72.0)
        engine.check_freshness("ds-b", _ago(2.0), 24.0, 72.0)
        summary = engine.get_sla_compliance_summary()
        assert summary["total"] == 2
        assert summary["compliant"] == 2
        assert summary["compliant_pct"] == 100.0

    def test_mixed_sla_statuses(self, populated_engine: FreshnessCheckerEngine):
        """Mixed statuses produce correct counts and percentages."""
        summary = populated_engine.get_sla_compliance_summary()
        assert summary["total"] == 5
        # ds-excellent(compliant), ds-good(compliant), ds-fair(compliant),
        # ds-poor(warning), ds-stale(breached)
        assert summary["compliant"] == 3
        assert summary["warning"] == 1
        assert summary["breached"] == 1
        assert summary["compliant_pct"] == pytest.approx(60.0, abs=0.01)
        assert summary["warning_pct"] == pytest.approx(20.0, abs=0.01)
        assert summary["breached_pct"] == pytest.approx(20.0, abs=0.01)

    def test_percentages_sum_to_100(self, populated_engine: FreshnessCheckerEngine):
        """All percentages sum to 100.0."""
        summary = populated_engine.get_sla_compliance_summary()
        total_pct = summary["compliant_pct"] + summary["warning_pct"] + summary["breached_pct"]
        assert total_pct == pytest.approx(100.0, abs=0.1)

    def test_provenance_hash_present(self, engine: FreshnessCheckerEngine):
        """Summary includes a provenance hash."""
        summary = engine.get_sla_compliance_summary()
        assert "provenance_hash" in summary
        assert len(summary["provenance_hash"]) == 64

    def test_latest_check_used_for_status(self, engine: FreshnessCheckerEngine):
        """Summary uses only the latest check per dataset."""
        # First check: breached
        engine.check_freshness("ds-001", _ago(100.0), 24.0, 72.0)
        # Second check: compliant (more recent data)
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        summary = engine.get_sla_compliance_summary()
        assert summary["total"] == 1
        assert summary["compliant"] == 1
        assert summary["breached"] == 0


# ===========================================================================
# 14. compute_freshness_heatmap
# ===========================================================================


class TestComputeFreshnessHeatmap:
    """Tests for FreshnessCheckerEngine.compute_freshness_heatmap."""

    def test_heatmap_no_history_returns_none_values(self, engine: FreshnessCheckerEngine):
        """Datasets without check history get None values."""
        heatmap = engine.compute_freshness_heatmap(["ds-unknown"])
        assert "ds-unknown" in heatmap
        entry = heatmap["ds-unknown"]
        assert entry["score"] is None
        assert entry["level"] is None
        assert entry["age_hours"] is None

    def test_heatmap_with_history(self, engine: FreshnessCheckerEngine):
        """Datasets with history get their latest score, level, and age."""
        engine.check_freshness("ds-001", _ago(3.0), 24.0, 72.0)
        heatmap = engine.compute_freshness_heatmap(["ds-001"])
        entry = heatmap["ds-001"]
        assert entry["score"] is not None
        assert 0.0 <= entry["score"] <= 1.0
        assert entry["level"] is not None
        assert entry["age_hours"] is not None
        assert entry["age_hours"] > 0.0

    def test_heatmap_mixed_known_unknown(self, engine: FreshnessCheckerEngine):
        """Heatmap handles a mix of known and unknown datasets."""
        engine.check_freshness("ds-known", _ago(1.0), 24.0, 72.0)
        heatmap = engine.compute_freshness_heatmap(["ds-known", "ds-unknown"])
        assert heatmap["ds-known"]["score"] is not None
        assert heatmap["ds-unknown"]["score"] is None

    def test_heatmap_empty_list(self, engine: FreshnessCheckerEngine):
        """Empty dataset list returns empty heatmap."""
        heatmap = engine.compute_freshness_heatmap([])
        assert heatmap == {}

    def test_heatmap_multiple_datasets(self, populated_engine: FreshnessCheckerEngine):
        """Heatmap covers multiple datasets with correct data."""
        ids = ["ds-excellent", "ds-good", "ds-stale"]
        heatmap = populated_engine.compute_freshness_heatmap(ids)
        assert len(heatmap) == 3
        assert heatmap["ds-excellent"]["level"] == FreshnessLevel.EXCELLENT.value
        assert heatmap["ds-stale"]["level"] == FreshnessLevel.STALE.value


# ===========================================================================
# 15. get_check_count
# ===========================================================================


class TestGetCheckCount:
    """Tests for FreshnessCheckerEngine.get_check_count."""

    def test_initial_count_is_zero(self, engine: FreshnessCheckerEngine):
        """Fresh engine has check count of zero."""
        assert engine.get_check_count() == 0

    def test_count_increments(self, engine: FreshnessCheckerEngine):
        """Count increments with each check."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert engine.get_check_count() == 1
        engine.check_freshness("ds-002", _ago(2.0), 24.0, 72.0)
        assert engine.get_check_count() == 2

    def test_count_after_batch(self, engine: FreshnessCheckerEngine):
        """Count reflects all checks from a batch operation."""
        entries = [_make_check_entry(f"ds-{i}", float(i + 1)) for i in range(7)]
        engine.batch_check(entries)
        assert engine.get_check_count() == 7


# ===========================================================================
# 16. get_statistics
# ===========================================================================


class TestGetStatistics:
    """Tests for FreshnessCheckerEngine.get_statistics."""

    def test_empty_engine_statistics(self, engine: FreshnessCheckerEngine):
        """Fresh engine returns zero totals and empty distributions."""
        stats = engine.get_statistics()
        assert stats["total_checks"] == 0
        assert stats["tracked_datasets"] == 0
        assert stats["average_freshness_score"] == 0.0
        # All distribution counts should be zero
        for level in FreshnessLevel:
            assert stats["freshness_level_distribution"][level.value] == 0
        for status in SLAStatus:
            assert stats["sla_status_distribution"][status.value] == 0

    def test_statistics_after_checks(self, populated_engine: FreshnessCheckerEngine):
        """Statistics reflect all performed checks."""
        stats = populated_engine.get_statistics()
        assert stats["total_checks"] == 5
        assert stats["tracked_datasets"] == 5
        assert 0.0 < stats["average_freshness_score"] <= 1.0

    def test_freshness_level_distribution(self, populated_engine: FreshnessCheckerEngine):
        """Freshness level distribution counts match the datasets."""
        stats = populated_engine.get_statistics()
        dist = stats["freshness_level_distribution"]
        assert dist[FreshnessLevel.EXCELLENT.value] == 1
        assert dist[FreshnessLevel.GOOD.value] == 1
        assert dist[FreshnessLevel.FAIR.value] == 1
        assert dist[FreshnessLevel.POOR.value] == 1
        assert dist[FreshnessLevel.STALE.value] == 1

    def test_sla_status_distribution(self, populated_engine: FreshnessCheckerEngine):
        """SLA status distribution counts match the datasets."""
        stats = populated_engine.get_statistics()
        dist = stats["sla_status_distribution"]
        assert dist[SLAStatus.COMPLIANT.value] == 3
        assert dist[SLAStatus.WARNING.value] == 1
        assert dist[SLAStatus.BREACHED.value] == 1

    def test_average_score_computation(self, engine: FreshnessCheckerEngine):
        """Average score is the mean of latest freshness scores."""
        engine.check_freshness("ds-a", _ago(0.5), 24.0, 72.0)  # score ~1.0
        engine.check_freshness("ds-b", _ago(0.5), 24.0, 72.0)  # score ~1.0
        stats = engine.get_statistics()
        assert stats["average_freshness_score"] == pytest.approx(1.0, abs=0.01)

    def test_statistics_uses_latest_check_only(self, engine: FreshnessCheckerEngine):
        """Statistics use only the latest check for each dataset."""
        engine.check_freshness("ds-001", _ago(100.0), 24.0, 72.0)  # STALE
        engine.check_freshness("ds-001", _ago(0.5), 24.0, 72.0)   # EXCELLENT
        stats = engine.get_statistics()
        assert stats["tracked_datasets"] == 1
        assert stats["freshness_level_distribution"][FreshnessLevel.EXCELLENT.value] == 1
        assert stats["freshness_level_distribution"][FreshnessLevel.STALE.value] == 0


# ===========================================================================
# 17. reset
# ===========================================================================


class TestReset:
    """Tests for FreshnessCheckerEngine.reset."""

    def test_reset_clears_check_count(self, engine: FreshnessCheckerEngine):
        """Reset sets check count back to zero."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert engine.get_check_count() == 1
        engine.reset()
        assert engine.get_check_count() == 0

    def test_reset_clears_history(self, engine: FreshnessCheckerEngine):
        """Reset clears all check history."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        assert len(engine.get_check_history("ds-001")) == 1
        engine.reset()
        assert engine.get_check_history("ds-001") == []

    def test_reset_clears_latest_check(self, engine: FreshnessCheckerEngine):
        """After reset, get_latest_check returns None."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        engine.reset()
        assert engine.get_latest_check("ds-001") is None

    def test_reset_allows_reuse(self, engine: FreshnessCheckerEngine):
        """Engine is fully functional after a reset."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        engine.reset()
        result = engine.check_freshness("ds-002", _ago(2.0), 24.0, 72.0)
        assert result.dataset_id == "ds-002"
        assert engine.get_check_count() == 1

    def test_reset_clears_statistics(self, engine: FreshnessCheckerEngine):
        """Statistics are reset to zero after reset."""
        engine.check_freshness("ds-001", _ago(1.0), 24.0, 72.0)
        engine.reset()
        stats = engine.get_statistics()
        assert stats["total_checks"] == 0
        assert stats["tracked_datasets"] == 0


# ===========================================================================
# Integration / Cross-method tests
# ===========================================================================


class TestCrossMethodIntegration:
    """Tests that validate interactions between multiple methods."""

    def test_check_freshness_consistency_with_compute_methods(
        self, engine: FreshnessCheckerEngine,
    ):
        """check_freshness result is consistent with individual compute methods."""
        last_refresh = _ago(15.0)
        result = engine.check_freshness("ds-001", last_refresh, 24.0, 72.0)
        # Verify score matches standalone computation
        standalone_score = engine.compute_freshness_score(result.age_hours)
        assert result.freshness_score == pytest.approx(standalone_score, abs=1e-4)
        # Verify level matches standalone classification
        standalone_level = engine.classify_freshness_level(result.age_hours)
        assert result.freshness_level == standalone_level.value
        # Verify SLA status matches standalone evaluation
        standalone_sla = engine.evaluate_sla_status(result.age_hours, 24.0, 72.0)
        assert result.sla_status == standalone_sla.value

    def test_multiple_checks_same_dataset_updates_latest(
        self, engine: FreshnessCheckerEngine,
    ):
        """Multiple checks on the same dataset correctly update the latest."""
        engine.check_freshness("ds-001", _ago(50.0), 24.0, 72.0)
        engine.check_freshness("ds-001", _ago(0.5), 24.0, 72.0)
        latest = engine.get_latest_check("ds-001")
        assert latest is not None
        assert latest.age_hours < 2.0  # most recent check
        assert latest.freshness_level == FreshnessLevel.EXCELLENT.value

    def test_stale_detection_uses_latest_check(self, engine: FreshnessCheckerEngine):
        """Stale detection uses the latest check, not historical ones."""
        # Old check: stale
        engine.check_freshness("ds-001", _ago(200.0), 24.0, 72.0)
        # New check: fresh
        engine.check_freshness("ds-001", _ago(0.5), 24.0, 72.0)
        stale = engine.get_stale_datasets(48.0)
        stale_ids = [s.dataset_id for s in stale]
        assert "ds-001" not in stale_ids

    def test_heatmap_reflects_latest_check(self, engine: FreshnessCheckerEngine):
        """Heatmap uses the latest check for each dataset."""
        engine.check_freshness("ds-001", _ago(200.0), 24.0, 72.0)
        engine.check_freshness("ds-001", _ago(0.5), 24.0, 72.0)
        heatmap = engine.compute_freshness_heatmap(["ds-001"])
        assert heatmap["ds-001"]["level"] == FreshnessLevel.EXCELLENT.value

    def test_full_lifecycle(self, engine: FreshnessCheckerEngine):
        """Full lifecycle: check, query, batch, group, stats, reset."""
        # Single check
        c1 = engine.check_freshness("ds-001", _ago(3.0), 24.0, 72.0)
        assert c1.freshness_level == FreshnessLevel.GOOD.value

        # Batch check
        batch_results = engine.batch_check([
            _make_check_entry("ds-002", 0.5),
            _make_check_entry("ds-003", 50.0),
        ])
        assert len(batch_results) == 2
        assert engine.get_check_count() == 3

        # Group check
        group_result = engine.check_dataset_group([
            _make_check_entry("ds-004", 1.0),
            _make_check_entry("ds-005", 100.0),
        ])
        assert group_result["total_datasets"] == 2
        assert engine.get_check_count() == 5

        # Statistics
        stats = engine.get_statistics()
        assert stats["tracked_datasets"] == 5
        assert stats["total_checks"] == 5

        # SLA compliance
        summary = engine.get_sla_compliance_summary()
        assert summary["total"] == 5

        # Reset
        engine.reset()
        assert engine.get_check_count() == 0
        assert engine.get_statistics()["tracked_datasets"] == 0


# ===========================================================================
# Enum tests
# ===========================================================================


class TestEnums:
    """Tests for FreshnessLevel and SLAStatus enums."""

    def test_freshness_level_values(self):
        """FreshnessLevel enum has correct string values."""
        assert FreshnessLevel.EXCELLENT.value == "excellent"
        assert FreshnessLevel.GOOD.value == "good"
        assert FreshnessLevel.FAIR.value == "fair"
        assert FreshnessLevel.POOR.value == "poor"
        assert FreshnessLevel.STALE.value == "stale"

    def test_sla_status_values(self):
        """SLAStatus enum has correct string values."""
        assert SLAStatus.COMPLIANT.value == "compliant"
        assert SLAStatus.WARNING.value == "warning"
        assert SLAStatus.BREACHED.value == "breached"

    def test_freshness_level_is_str_enum(self):
        """FreshnessLevel inherits from str."""
        assert isinstance(FreshnessLevel.EXCELLENT, str)

    def test_sla_status_is_str_enum(self):
        """SLAStatus inherits from str."""
        assert isinstance(SLAStatus.COMPLIANT, str)
