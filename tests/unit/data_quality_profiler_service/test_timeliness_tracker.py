# -*- coding: utf-8 -*-
"""
Unit tests for TimelinessTracker engine.

AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)
Tests freshness scoring, SLA compliance, stale record detection,
update frequency analysis, timeliness scoring, and issue generation.

Target: 90+ tests, 85%+ coverage.
"""

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from greenlang.data_quality_profiler.timeliness_tracker import (
    TimelinessTracker,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
    _parse_timestamp,
    _compute_provenance,
    _safe_stdev,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow_fixed() -> datetime:
    """Return a deterministic now() for tests that need time control."""
    return datetime(2026, 2, 9, 12, 0, 0, tzinfo=timezone.utc)


def _ts_ago(hours: float) -> str:
    """Return ISO timestamp `hours` before the fixed now."""
    dt = _utcnow_fixed() - timedelta(hours=hours)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> TimelinessTracker:
    """Create a default TimelinessTracker."""
    return TimelinessTracker()


@pytest.fixture
def custom_tracker() -> TimelinessTracker:
    """Create a tracker with custom thresholds."""
    return TimelinessTracker(config={
        "excellent_hours": 2.0,
        "good_hours": 12.0,
        "fair_hours": 48.0,
        "poor_hours": 168.0,
        "default_sla_hours": 48.0,
    })


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Test TimelinessTracker initialization."""

    def test_default_config(self):
        """Default thresholds are applied."""
        t = TimelinessTracker()
        assert t._excellent_hours == 1.0
        assert t._good_hours == 6.0
        assert t._fair_hours == 24.0
        assert t._poor_hours == 72.0
        assert t._default_sla == 24.0

    def test_custom_config(self):
        """Custom config overrides defaults."""
        t = TimelinessTracker(config={
            "excellent_hours": 0.5,
            "good_hours": 3.0,
            "fair_hours": 12.0,
            "poor_hours": 48.0,
            "default_sla_hours": 12.0,
        })
        assert t._excellent_hours == 0.5
        assert t._good_hours == 3.0
        assert t._fair_hours == 12.0
        assert t._poor_hours == 48.0
        assert t._default_sla == 12.0

    def test_initial_stats(self):
        """Stats start at zero."""
        t = TimelinessTracker()
        stats = t.get_statistics()
        assert stats["checks_completed"] == 0
        assert stats["total_fresh"] == 0
        assert stats["total_stale"] == 0

    def test_none_config_uses_defaults(self):
        """Passing None uses defaults."""
        t = TimelinessTracker(config=None)
        assert t._excellent_hours == 1.0


# ---------------------------------------------------------------------------
# TestCheckFreshness
# ---------------------------------------------------------------------------


class TestCheckFreshness:
    """Test check_freshness() method."""

    def test_very_recent_data(self, tracker):
        """Data updated just now -> excellent freshness."""
        now = datetime.now(timezone.utc)
        result = tracker.check_freshness("test_ds", now.isoformat())
        assert result["freshness_score"] >= 0.99
        assert result["freshness_level"] == "excellent"
        assert result["is_fresh"] is True

    def test_one_hour_ago(self, tracker):
        """1 hour ago -> still excellent or good."""
        ts = datetime.now(timezone.utc) - timedelta(hours=1)
        result = tracker.check_freshness("ds", ts.isoformat())
        assert result["freshness_score"] >= 0.85
        assert result["freshness_level"] in ("excellent", "good")

    def test_twelve_hours_ago(self, tracker):
        """12 hours ago -> fair range."""
        ts = datetime.now(timezone.utc) - timedelta(hours=12)
        result = tracker.check_freshness("ds", ts.isoformat())
        assert result["freshness_level"] == "fair"
        assert 0.70 <= result["freshness_score"] <= 0.90

    def test_fifty_hours_ago(self, tracker):
        """50 hours ago -> poor range."""
        ts = datetime.now(timezone.utc) - timedelta(hours=50)
        result = tracker.check_freshness("ds", ts.isoformat())
        assert result["freshness_level"] == "poor"
        assert 0.45 <= result["freshness_score"] <= 0.75

    def test_very_stale(self, tracker):
        """100 hours ago -> stale, score 0.0."""
        ts = datetime.now(timezone.utc) - timedelta(hours=100)
        result = tracker.check_freshness("ds", ts.isoformat())
        assert result["freshness_level"] == "stale"
        assert result["freshness_score"] == 0.0

    def test_sla_compliant(self, tracker):
        """Data within SLA window."""
        ts = datetime.now(timezone.utc) - timedelta(hours=12)
        result = tracker.check_freshness("ds", ts, sla_hours=24)
        assert result["sla_compliant"] is True

    def test_sla_non_compliant(self, tracker):
        """Data outside SLA window."""
        ts = datetime.now(timezone.utc) - timedelta(hours=30)
        result = tracker.check_freshness("ds", ts, sla_hours=24)
        assert result["sla_compliant"] is False

    def test_custom_sla(self, tracker):
        """Custom SLA overrides default."""
        ts = datetime.now(timezone.utc) - timedelta(hours=5)
        result = tracker.check_freshness("ds", ts, sla_hours=4)
        assert result["sla_hours"] == 4
        assert result["is_fresh"] is False

    def test_dataset_name_in_result(self, tracker):
        """dataset_name is echoed in result."""
        ts = datetime.now(timezone.utc)
        result = tracker.check_freshness("my_dataset", ts)
        assert result["dataset_name"] == "my_dataset"

    def test_provenance_hash(self, tracker):
        """provenance_hash is 64-char hex."""
        ts = datetime.now(timezone.utc)
        result = tracker.check_freshness("ds", ts)
        assert len(result["provenance_hash"]) == 64

    def test_freshness_level_key(self, tracker):
        """freshness_level key is present."""
        ts = datetime.now(timezone.utc)
        result = tracker.check_freshness("ds", ts)
        assert result["freshness_level"] in (
            "excellent", "good", "fair", "poor", "stale"
        )

    def test_age_hours_calculation(self, tracker):
        """age_hours is non-negative."""
        ts = datetime.now(timezone.utc) - timedelta(hours=5)
        result = tracker.check_freshness("ds", ts)
        assert result["age_hours"] >= 4.5

    def test_iso_datetime_input(self, tracker):
        """ISO string input accepted."""
        ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        result = tracker.check_freshness("ds", ts)
        assert result["freshness_score"] > 0.0

    def test_epoch_timestamp_input(self, tracker):
        """Unix timestamp input accepted."""
        epoch = time.time() - 3600  # 1 hour ago
        result = tracker.check_freshness("ds", epoch)
        assert result["freshness_score"] > 0.0

    def test_unparseable_raises(self, tracker):
        """Unparseable timestamp raises ValueError."""
        with pytest.raises(ValueError, match="Cannot parse"):
            tracker.check_freshness("ds", "not-a-date")


# ---------------------------------------------------------------------------
# TestComputeFreshnessScore
# ---------------------------------------------------------------------------


class TestComputeFreshnessScore:
    """Test compute_freshness_score()."""

    def test_age_zero(self, tracker):
        """age=0 -> 1.0."""
        assert tracker.compute_freshness_score(0) == 1.0

    def test_age_half_hour(self, tracker):
        """age=0.5 (within excellent) -> 1.0."""
        assert tracker.compute_freshness_score(0.5) == 1.0

    def test_age_one_hour_boundary(self, tracker):
        """age=1.0 (excellent boundary) -> 1.0."""
        assert tracker.compute_freshness_score(1.0) == 1.0

    def test_age_three_hours(self, tracker):
        """age=3 -> between 0.85 and 1.0 (good region)."""
        score = tracker.compute_freshness_score(3.0)
        assert 0.85 <= score <= 1.0

    def test_age_six_hours(self, tracker):
        """age=6 (good boundary) -> 0.85."""
        score = tracker.compute_freshness_score(6.0)
        assert abs(score - 0.85) < 0.01

    def test_age_twelve_hours(self, tracker):
        """age=12 -> in fair region."""
        score = tracker.compute_freshness_score(12.0)
        assert 0.70 <= score <= 0.85

    def test_age_24_hours(self, tracker):
        """age=24 (fair boundary) -> 0.70."""
        score = tracker.compute_freshness_score(24.0)
        assert abs(score - 0.70) < 0.01

    def test_age_48_hours(self, tracker):
        """age=48 -> in poor region."""
        score = tracker.compute_freshness_score(48.0)
        assert 0.50 <= score <= 0.70

    def test_age_72_hours(self, tracker):
        """age=72 (poor boundary) -> 0.50."""
        score = tracker.compute_freshness_score(72.0)
        assert abs(score - 0.50) < 0.01

    def test_age_100_hours(self, tracker):
        """age=100 (beyond poor) -> 0.0."""
        assert tracker.compute_freshness_score(100.0) == 0.0

    def test_negative_age(self, tracker):
        """Negative age -> 1.0."""
        assert tracker.compute_freshness_score(-5.0) == 1.0


# ---------------------------------------------------------------------------
# TestCheckFieldFreshness
# ---------------------------------------------------------------------------


class TestCheckFieldFreshness:
    """Test check_field_freshness()."""

    def test_per_record_analysis(self, tracker):
        """Each record gets a freshness score."""
        now = datetime.now(timezone.utc)
        data = [
            {"ts": (now - timedelta(hours=1)).isoformat()},
            {"ts": (now - timedelta(hours=50)).isoformat()},
        ]
        result = tracker.check_field_freshness(data, "ts")
        assert result["total_records"] == 2
        assert len(result["record_scores"]) == 2

    def test_mixed_timestamps(self, tracker):
        """Mix of recent and stale records."""
        now = datetime.now(timezone.utc)
        data = [
            {"ts": now.isoformat()},
            {"ts": (now - timedelta(hours=100)).isoformat()},
        ]
        result = tracker.check_field_freshness(data, "ts")
        scores = [r["freshness_score"] for r in result["record_scores"]]
        assert max(scores) > 0.5
        assert min(scores) == 0.0

    def test_all_recent(self, tracker):
        """All records are fresh."""
        now = datetime.now(timezone.utc)
        data = [{"ts": now.isoformat()} for _ in range(5)]
        result = tracker.check_field_freshness(data, "ts")
        assert result["summary"]["mean_score"] > 0.9

    def test_all_stale(self, tracker):
        """All records are very stale."""
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        data = [{"ts": old.isoformat()} for _ in range(5)]
        result = tracker.check_field_freshness(data, "ts")
        assert result["summary"]["mean_score"] == 0.0

    def test_null_timestamps(self, tracker):
        """Null timestamps counted as parse failures."""
        data = [{"ts": None}, {"ts": None}]
        result = tracker.check_field_freshness(data, "ts")
        assert result["parse_failures"] == 2

    def test_invalid_timestamps(self, tracker):
        """Invalid strings counted as parse failures."""
        data = [{"ts": "not-a-date"}, {"ts": "garbage"}]
        result = tracker.check_field_freshness(data, "ts")
        assert result["parse_failures"] == 2

    def test_empty_data_raises(self, tracker):
        """Empty data raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            tracker.check_field_freshness([], "ts")

    def test_freshness_distribution(self, tracker):
        """freshness_distribution dict present."""
        now = datetime.now(timezone.utc)
        data = [{"ts": now.isoformat()}]
        result = tracker.check_field_freshness(data, "ts")
        assert isinstance(result["freshness_distribution"], dict)


# ---------------------------------------------------------------------------
# TestComputeTimelinessScore
# ---------------------------------------------------------------------------


class TestComputeTimelinessScore:
    """Test compute_timeliness_score()."""

    def test_all_recent(self, tracker):
        """All recent timestamps -> high score."""
        now = datetime.now(timezone.utc)
        data = [{"ts": now.isoformat()} for _ in range(5)]
        score = tracker.compute_timeliness_score(data, ["ts"])
        assert score >= 0.9

    def test_all_stale(self, tracker):
        """All very stale timestamps -> score 0.0."""
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        data = [{"ts": old.isoformat()} for _ in range(5)]
        score = tracker.compute_timeliness_score(data, ["ts"])
        assert score == 0.0

    def test_mixed(self, tracker):
        """Mixed fresh and stale -> moderate score."""
        now = datetime.now(timezone.utc)
        data = [
            {"ts": now.isoformat()},
            {"ts": (now - timedelta(hours=200)).isoformat()},
        ]
        score = tracker.compute_timeliness_score(data, ["ts"])
        assert 0.0 < score < 1.0

    def test_no_timestamp_columns(self, tracker):
        """No timestamp columns -> 1.0."""
        data = [{"val": 1}]
        score = tracker.compute_timeliness_score(data, [])
        assert score == 1.0

    def test_single_record(self, tracker):
        """Single record."""
        now = datetime.now(timezone.utc)
        data = [{"ts": now.isoformat()}]
        score = tracker.compute_timeliness_score(data, ["ts"])
        assert score > 0.5

    def test_empty_data(self, tracker):
        """Empty data -> 1.0."""
        score = tracker.compute_timeliness_score([], ["ts"])
        assert score == 1.0

    def test_multiple_timestamp_columns(self, tracker):
        """Multiple timestamp columns are averaged."""
        now = datetime.now(timezone.utc)
        data = [{"ts1": now.isoformat(), "ts2": now.isoformat()}]
        score = tracker.compute_timeliness_score(data, ["ts1", "ts2"])
        assert score > 0.5

    def test_non_parseable_ignored(self, tracker):
        """Non-parseable timestamps are ignored."""
        now = datetime.now(timezone.utc)
        data = [
            {"ts": now.isoformat()},
            {"ts": "not-a-date"},
        ]
        score = tracker.compute_timeliness_score(data, ["ts"])
        assert score > 0.0


# ---------------------------------------------------------------------------
# TestDetectStaleRecords
# ---------------------------------------------------------------------------


class TestDetectStaleRecords:
    """Test detect_stale_records()."""

    def test_no_stale_records(self, tracker):
        """Recent data has no stale records."""
        now = datetime.now(timezone.utc)
        data = [{"ts": now.isoformat()} for _ in range(5)]
        stale = tracker.detect_stale_records(data, "ts")
        assert len(stale) == 0

    def test_all_stale(self, tracker):
        """Old data is all stale."""
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        data = [{"ts": old.isoformat()} for _ in range(3)]
        stale = tracker.detect_stale_records(data, "ts")
        assert len(stale) == 3

    def test_mixed_stale(self, tracker):
        """Mix of stale and fresh."""
        now = datetime.now(timezone.utc)
        data = [
            {"ts": now.isoformat()},
            {"ts": (now - timedelta(hours=200)).isoformat()},
        ]
        stale = tracker.detect_stale_records(data, "ts")
        assert len(stale) == 1

    def test_custom_threshold(self, tracker):
        """Custom threshold changes detection sensitivity."""
        now = datetime.now(timezone.utc)
        data = [{"ts": (now - timedelta(hours=5)).isoformat()}]
        # Default threshold is 72h, so 5h is not stale
        stale_default = tracker.detect_stale_records(data, "ts")
        assert len(stale_default) == 0
        # With 4h threshold, 5h is stale
        stale_custom = tracker.detect_stale_records(data, "ts", threshold_hours=4)
        assert len(stale_custom) == 1

    def test_null_timestamps(self, tracker):
        """Null timestamps are marked stale."""
        data = [{"ts": None}]
        stale = tracker.detect_stale_records(data, "ts")
        assert len(stale) == 1
        assert stale[0]["reason"] == "unparseable_timestamp"

    def test_empty_data(self, tracker):
        """Empty data returns empty list."""
        stale = tracker.detect_stale_records([], "ts")
        assert stale == []

    def test_stale_record_fields(self, tracker):
        """Stale record dict has expected fields."""
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        data = [{"ts": old.isoformat()}]
        stale = tracker.detect_stale_records(data, "ts")
        assert stale[0]["row_index"] == 0
        assert stale[0]["age_hours"] > 100
        assert stale[0]["freshness_score"] == 0.0

    def test_edge_threshold(self, tracker):
        """Record at exactly the threshold is not stale (> not >=)."""
        now = datetime.now(timezone.utc)
        # Set threshold to exactly 10 hours, make data exactly 10h old
        ts = (now - timedelta(hours=10)).isoformat()
        data = [{"ts": ts}]
        stale = tracker.detect_stale_records(data, "ts", threshold_hours=10.5)
        # age ~10h which is not > 10.5
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# TestComputeUpdateFrequency
# ---------------------------------------------------------------------------


class TestComputeUpdateFrequency:
    """Test compute_update_frequency()."""

    def test_regular_intervals(self, tracker):
        """Regular 1-hour intervals -> high regularity."""
        base = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
        timestamps = [(base + timedelta(hours=i)).isoformat() for i in range(10)]
        result = tracker.compute_update_frequency(timestamps)
        assert abs(result["mean_interval_hours"] - 1.0) < 0.01
        assert result["regularity_score"] > 0.9

    def test_irregular_intervals(self, tracker):
        """Irregular intervals -> low regularity."""
        base = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
        offsets = [0, 1, 2, 10, 11, 50]
        timestamps = [(base + timedelta(hours=h)).isoformat() for h in offsets]
        result = tracker.compute_update_frequency(timestamps)
        assert result["regularity_score"] < 0.8

    def test_single_timestamp(self, tracker):
        """Single timestamp -> no intervals."""
        result = tracker.compute_update_frequency(["2026-01-01T00:00:00Z"])
        assert result["total_updates"] == 1
        assert result["mean_interval_hours"] == 0.0
        assert result["regularity_score"] == 1.0

    def test_empty(self, tracker):
        """Empty list."""
        result = tracker.compute_update_frequency([])
        assert result["total_updates"] == 0
        assert result["regularity_score"] == 1.0

    def test_mean_median_stddev(self, tracker):
        """Mean, median, stddev are computed."""
        base = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
        timestamps = [(base + timedelta(hours=i * 2)).isoformat() for i in range(5)]
        result = tracker.compute_update_frequency(timestamps)
        assert abs(result["mean_interval_hours"] - 2.0) < 0.01
        assert abs(result["median_interval_hours"] - 2.0) < 0.01
        assert result["stddev_interval_hours"] == 0.0

    def test_two_timestamps(self, tracker):
        """Two timestamps produce one interval."""
        base = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
        ts = [base.isoformat(), (base + timedelta(hours=6)).isoformat()]
        result = tracker.compute_update_frequency(ts)
        assert result["total_intervals"] == 1
        assert abs(result["mean_interval_hours"] - 6.0) < 0.01

    def test_min_max_interval(self, tracker):
        """Min and max intervals are correct."""
        base = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
        offsets = [0, 1, 5, 6]
        timestamps = [(base + timedelta(hours=h)).isoformat() for h in offsets]
        result = tracker.compute_update_frequency(timestamps)
        assert result["min_interval_hours"] == 1.0
        assert result["max_interval_hours"] == 4.0

    def test_unsorted_timestamps(self, tracker):
        """Timestamps are sorted internally."""
        base = datetime(2026, 2, 9, 0, 0, 0, tzinfo=timezone.utc)
        timestamps = [
            (base + timedelta(hours=10)).isoformat(),
            base.isoformat(),
            (base + timedelta(hours=5)).isoformat(),
        ]
        result = tracker.compute_update_frequency(timestamps)
        assert result["total_intervals"] == 2


# ---------------------------------------------------------------------------
# TestCheckSLACompliance
# ---------------------------------------------------------------------------


class TestCheckSLACompliance:
    """Test check_sla_compliance()."""

    def test_all_compliant(self, tracker):
        """All datasets within SLA."""
        results = [
            {"age_hours": 5, "dataset_name": "ds1"},
            {"age_hours": 10, "dataset_name": "ds2"},
        ]
        compliance = tracker.check_sla_compliance(results, sla_hours=24)
        assert compliance["compliance_rate"] == 1.0
        assert compliance["non_compliant_count"] == 0

    def test_mixed_compliance(self, tracker):
        """Some within SLA, some not."""
        results = [
            {"age_hours": 5, "dataset_name": "ds1"},
            {"age_hours": 30, "dataset_name": "ds2"},
        ]
        compliance = tracker.check_sla_compliance(results, sla_hours=24)
        assert compliance["compliance_rate"] == 0.5
        assert compliance["non_compliant_count"] == 1

    def test_none_compliant(self, tracker):
        """All outside SLA."""
        results = [
            {"age_hours": 50, "dataset_name": "ds1"},
            {"age_hours": 100, "dataset_name": "ds2"},
        ]
        compliance = tracker.check_sla_compliance(results, sla_hours=24)
        assert compliance["compliance_rate"] == 0.0

    def test_custom_sla(self, tracker):
        """Custom SLA overrides default."""
        results = [{"age_hours": 5, "dataset_name": "ds1"}]
        compliance = tracker.check_sla_compliance(results, sla_hours=4)
        assert compliance["non_compliant_count"] == 1

    def test_empty_list(self, tracker):
        """Empty results -> compliance 1.0."""
        compliance = tracker.check_sla_compliance([], sla_hours=24)
        assert compliance["compliance_rate"] == 1.0

    def test_provenance_hash(self, tracker):
        """Provenance hash is present."""
        results = [{"age_hours": 5, "dataset_name": "ds1"}]
        compliance = tracker.check_sla_compliance(results, sla_hours=24)
        assert len(compliance["provenance_hash"]) == 64


# ---------------------------------------------------------------------------
# TestGenerateTimelinessIssues
# ---------------------------------------------------------------------------


class TestGenerateTimelinessIssues:
    """Test generate_timeliness_issues()."""

    def test_stale_data_issue(self, tracker):
        """Very old data generates stale_data issue."""
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        data = [{"ts": old.isoformat()} for _ in range(5)]
        issues = tracker.generate_timeliness_issues(data, ["ts"])
        types = [i["type"] for i in issues]
        assert "stale_data" in types

    def test_no_issues_for_fresh(self, tracker):
        """Fresh data generates no stale_data issues."""
        now = datetime.now(timezone.utc)
        data = [{"ts": now.isoformat()} for _ in range(5)]
        issues = tracker.generate_timeliness_issues(data, ["ts"])
        stale_issues = [i for i in issues if i["type"] == "stale_data"]
        assert len(stale_issues) == 0

    def test_sla_breach_issue(self, tracker):
        """Staleness warning generated for old data."""
        old = datetime.now(timezone.utc) - timedelta(hours=100)
        data = [{"ts": old.isoformat()}]
        issues = tracker.generate_timeliness_issues(data, ["ts"])
        types = [i["type"] for i in issues]
        assert "data_staleness_warning" in types

    def test_severity_levels(self, tracker):
        """Severity reflects staleness magnitude."""
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        data = [{"ts": old.isoformat()} for _ in range(10)]
        issues = tracker.generate_timeliness_issues(data, ["ts"])
        stale_issues = [i for i in issues if i["type"] == "stale_data"]
        if stale_issues:
            assert stale_issues[0]["severity"] in (
                SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM
            )

    def test_empty_data(self, tracker):
        """Empty data returns no issues."""
        issues = tracker.generate_timeliness_issues([], ["ts"])
        assert issues == []

    def test_empty_columns(self, tracker):
        """No timestamp columns returns no issues."""
        data = [{"val": 1}]
        issues = tracker.generate_timeliness_issues(data, [])
        assert issues == []

    def test_unparseable_timestamps_issue(self, tracker):
        """Unparseable timestamps generate issue."""
        data = [{"ts": "garbage"} for _ in range(5)]
        issues = tracker.generate_timeliness_issues(data, ["ts"])
        types = [i["type"] for i in issues]
        assert "unparseable_timestamps" in types


# ---------------------------------------------------------------------------
# TestStatistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test get_statistics()."""

    def test_initial_statistics(self, tracker):
        """Initial stats all zero."""
        stats = tracker.get_statistics()
        assert stats["checks_completed"] == 0
        assert stats["total_fresh"] == 0
        assert stats["total_stale"] == 0
        assert stats["stored_checks"] == 0

    def test_post_check_statistics(self, tracker):
        """Stats update after check."""
        now = datetime.now(timezone.utc)
        tracker.check_freshness("ds", now)
        stats = tracker.get_statistics()
        assert stats["checks_completed"] == 1
        assert stats["total_fresh"] == 1
        assert stats["stored_checks"] == 1

    def test_stale_counted(self, tracker):
        """Stale checks increment total_stale."""
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        tracker.check_freshness("ds", old)
        stats = tracker.get_statistics()
        assert stats["total_stale"] == 1

    def test_freshness_rate(self, tracker):
        """freshness_rate is computed correctly."""
        now = datetime.now(timezone.utc)
        tracker.check_freshness("ds1", now)
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        tracker.check_freshness("ds2", old)
        stats = tracker.get_statistics()
        assert stats["freshness_rate"] == 0.5


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_sha256_format(self, tracker):
        """Provenance hash is 64-char hex (SHA-256)."""
        now = datetime.now(timezone.utc)
        result = tracker.check_freshness("ds", now)
        h = result["provenance_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_helper_function(self):
        """_compute_provenance returns 64-char hex."""
        h = _compute_provenance("test_op", "test_data")
        assert len(h) == 64

    def test_check_id_format(self, tracker):
        """check_id starts with TML-."""
        now = datetime.now(timezone.utc)
        result = tracker.check_freshness("ds", now)
        assert result["check_id"].startswith("TML-")


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread safety of TimelinessTracker."""

    def test_concurrent_checks(self, tracker):
        """Multiple threads can run check_freshness concurrently."""
        now = datetime.now(timezone.utc)
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(5):
                    tracker.check_freshness("ds", now)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = tracker.get_statistics()
        assert stats["checks_completed"] == 20

    def test_concurrent_stats_access(self, tracker):
        """Stats can be read concurrently with checks."""
        now = datetime.now(timezone.utc)
        results: List[Dict] = []

        def check_worker():
            for _ in range(10):
                tracker.check_freshness("ds", now)

        def stats_worker():
            for _ in range(10):
                results.append(tracker.get_statistics())

        t1 = threading.Thread(target=check_worker)
        t2 = threading.Thread(target=stats_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 10


# ---------------------------------------------------------------------------
# TestStorageAndRetrieval
# ---------------------------------------------------------------------------


class TestStorageAndRetrieval:
    """Test check storage, listing, and deletion."""

    def test_get_check(self, tracker):
        """Retrieve a stored check by ID."""
        now = datetime.now(timezone.utc)
        result = tracker.check_freshness("ds", now)
        stored = tracker.get_check(result["check_id"])
        assert stored is not None
        assert stored["check_id"] == result["check_id"]

    def test_get_nonexistent(self, tracker):
        """Get nonexistent ID returns None."""
        assert tracker.get_check("TML-nonexistent") is None

    def test_list_checks(self, tracker):
        """list_checks returns stored results."""
        now = datetime.now(timezone.utc)
        tracker.check_freshness("ds1", now)
        tracker.check_freshness("ds2", now)
        checks = tracker.list_checks()
        assert len(checks) == 2

    def test_list_pagination(self, tracker):
        """list_checks supports limit and offset."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            tracker.check_freshness(f"ds{i}", now)
        page = tracker.list_checks(limit=2, offset=1)
        assert len(page) == 2

    def test_delete_check(self, tracker):
        """Delete removes check from storage."""
        now = datetime.now(timezone.utc)
        result = tracker.check_freshness("ds", now)
        assert tracker.delete_check(result["check_id"]) is True
        assert tracker.get_check(result["check_id"]) is None

    def test_delete_nonexistent(self, tracker):
        """Delete nonexistent returns False."""
        assert tracker.delete_check("TML-nonexistent") is False


# ---------------------------------------------------------------------------
# TestParseTimestamp
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    """Test _parse_timestamp helper."""

    def test_none(self):
        """None -> None."""
        assert _parse_timestamp(None) is None

    def test_datetime_with_tz(self):
        """datetime with tz is returned as-is."""
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _parse_timestamp(dt) == dt

    def test_datetime_naive(self):
        """Naive datetime gets UTC attached."""
        dt = datetime(2026, 1, 1)
        result = _parse_timestamp(dt)
        assert result.tzinfo == timezone.utc

    def test_iso_string(self):
        """ISO format string is parsed."""
        result = _parse_timestamp("2026-01-15T10:30:00+00:00")
        assert result is not None
        assert result.year == 2026

    def test_iso_z_suffix(self):
        """ISO format with Z suffix is parsed."""
        result = _parse_timestamp("2026-01-15T10:30:00Z")
        assert result is not None

    def test_unix_timestamp_int(self):
        """Integer unix timestamp is parsed."""
        result = _parse_timestamp(1700000000)
        assert result is not None

    def test_unix_timestamp_float(self):
        """Float unix timestamp is parsed."""
        result = _parse_timestamp(1700000000.5)
        assert result is not None

    def test_empty_string(self):
        """Empty string -> None."""
        assert _parse_timestamp("") is None

    def test_date_only(self):
        """Date-only string is parsed."""
        result = _parse_timestamp("2026-01-15")
        assert result is not None
        assert result.year == 2026

    def test_unparseable(self):
        """Garbage string -> None."""
        assert _parse_timestamp("not-a-date-at-all") is None
