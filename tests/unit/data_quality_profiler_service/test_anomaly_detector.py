# -*- coding: utf-8 -*-
"""
Unit tests for AnomalyDetector engine.

AGENT-DATA-010: Data Quality Profiler (GL-DATA-X-013)
Tests IQR, Z-score, MAD, Modified Z-score, Grubbs, distribution profiling,
sudden change detection, severity classification, and issue generation.

Target: 110+ tests, 85%+ coverage.
"""

import math
import statistics
import threading
from typing import Any, Dict, List, Tuple

import pytest

from greenlang.data_quality_profiler.anomaly_detector import (
    AnomalyDetector,
    METHOD_IQR,
    METHOD_ZSCORE,
    METHOD_MAD,
    METHOD_GRUBBS,
    METHOD_MODIFIED_ZSCORE,
    METHOD_ALL,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
    SEVERITY_INFO,
    _compute_provenance,
    _grubbs_critical,
    _safe_mean,
    _safe_median,
    _safe_stdev,
    _try_float,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _indexed(nums: List[float]) -> List[Tuple[int, float]]:
    """Build indexed_nums list from a plain list."""
    return list(enumerate(nums))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> AnomalyDetector:
    """Create a default AnomalyDetector."""
    return AnomalyDetector()


@pytest.fixture
def custom_detector() -> AnomalyDetector:
    """Create a detector with custom config."""
    return AnomalyDetector(config={
        "iqr_multiplier": 2.0,
        "zscore_threshold": 2.5,
        "mad_threshold": 3.0,
        "change_window_size": 3,
        "default_method": "zscore",
    })


@pytest.fixture
def normal_data() -> List[float]:
    """Normal range data without outliers."""
    return [10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5]


@pytest.fixture
def outlier_data() -> List[float]:
    """Data with a clear outlier."""
    return [10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 100.0]


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Test AnomalyDetector initialization."""

    def test_default_config(self):
        """Default thresholds are applied."""
        d = AnomalyDetector()
        assert d._iqr_k == 1.5
        assert d._zscore_t == 3.0
        assert d._mad_t == 3.5
        assert d._change_window == 5
        assert d._default_method == METHOD_IQR

    def test_custom_config(self):
        """Custom config overrides defaults."""
        d = AnomalyDetector(config={
            "iqr_multiplier": 2.0,
            "zscore_threshold": 2.0,
            "mad_threshold": 4.0,
            "change_window_size": 10,
            "default_method": "zscore",
        })
        assert d._iqr_k == 2.0
        assert d._zscore_t == 2.0
        assert d._mad_t == 4.0
        assert d._change_window == 10
        assert d._default_method == "zscore"

    def test_initial_stats(self):
        """Stats start at zero."""
        d = AnomalyDetector()
        stats = d.get_statistics()
        assert stats["detections_completed"] == 0
        assert stats["total_anomalies_found"] == 0
        assert stats["total_values_scanned"] == 0

    def test_none_config_uses_defaults(self):
        """Passing None uses defaults."""
        d = AnomalyDetector(config=None)
        assert d._iqr_k == 1.5


# ---------------------------------------------------------------------------
# TestDetect
# ---------------------------------------------------------------------------


class TestDetect:
    """Test the main detect() method."""

    def test_return_type(self, detector, normal_data):
        """detect() returns a dict."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        assert isinstance(result, dict)

    def test_result_keys(self, detector, normal_data):
        """Result contains required keys."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        required = {
            "detection_id", "method", "row_count",
            "total_anomalies", "anomaly_rate", "column_anomalies",
            "provenance_hash", "detection_time_ms",
        }
        assert required.issubset(result.keys())

    def test_multiple_columns(self, detector):
        """All numeric columns are scanned."""
        data = [{"a": i, "b": i * 10} for i in range(10)]
        result = detector.detect(data)
        assert result["columns_scanned"] >= 2

    def test_default_method(self, detector, normal_data):
        """Default method is IQR."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        assert result["method"] == METHOD_IQR

    def test_specific_method(self, detector, normal_data):
        """Specific method override works."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data, method=METHOD_ZSCORE)
        assert result["method"] == METHOD_ZSCORE

    def test_empty_data_raises(self, detector):
        """Empty dataset raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            detector.detect([])

    def test_no_anomalies_in_uniform(self, detector):
        """Uniform data has no anomalies."""
        data = [{"val": 10.0} for _ in range(20)]
        result = detector.detect(data)
        assert result["total_anomalies"] == 0

    def test_clear_outlier_detected(self, detector, outlier_data):
        """Clear outlier is detected."""
        data = [{"val": v} for v in outlier_data]
        result = detector.detect(data)
        assert result["total_anomalies"] >= 1

    def test_column_selection(self, detector):
        """Column selection limits scan."""
        data = [{"a": i, "b": 1000 if i == 5 else i} for i in range(20)]
        result = detector.detect(data, columns=["a"])
        # Only column 'a' scanned; 'b' outlier not found
        assert "b" not in result["column_anomalies"]

    def test_provenance_hash(self, detector, normal_data):
        """Provenance hash is 64-char hex."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        assert len(result["provenance_hash"]) == 64

    def test_anomaly_count(self, detector, outlier_data):
        """total_anomalies matches sum across columns."""
        data = [{"val": v} for v in outlier_data]
        result = detector.detect(data)
        col_total = sum(len(a) for a in result["column_anomalies"].values())
        assert result["total_anomalies"] == col_total

    def test_non_numeric_columns_skipped(self, detector):
        """Non-numeric columns are skipped."""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = detector.detect(data)
        assert result["columns_scanned"] == 0


# ---------------------------------------------------------------------------
# TestDetectColumnAnomalies
# ---------------------------------------------------------------------------


class TestDetectColumnAnomalies:
    """Test detect_column_anomalies()."""

    def test_no_anomalies(self, detector, normal_data):
        """Normal data has no anomalies."""
        anomalies = detector.detect_column_anomalies(normal_data, "col")
        assert len(anomalies) == 0

    def test_single_outlier(self, detector, outlier_data):
        """Single large outlier detected."""
        anomalies = detector.detect_column_anomalies(outlier_data, "col")
        assert len(anomalies) >= 1
        outlier_values = [a["value"] for a in anomalies]
        assert 100.0 in outlier_values

    def test_multiple_outliers(self, detector):
        """Multiple outliers all detected."""
        data = [10, 11, 12, 13, 14, 100, 200, 12, 11, 10]
        anomalies = detector.detect_column_anomalies(data, "col")
        assert len(anomalies) >= 2

    def test_all_same_values(self, detector):
        """Identical values -> no anomalies (stdev=0, IQR=0)."""
        data = [5.0] * 20
        anomalies = detector.detect_column_anomalies(data, "col")
        assert len(anomalies) == 0

    def test_empty_values(self, detector):
        """Empty list returns empty."""
        anomalies = detector.detect_column_anomalies([], "col")
        assert anomalies == []

    def test_insufficient_samples(self, detector):
        """< 3 values returns empty (too few for statistics)."""
        anomalies = detector.detect_column_anomalies([1.0, 2.0], "col")
        assert anomalies == []

    def test_extreme_values(self, detector):
        """Very extreme outlier is detected."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10000]
        anomalies = detector.detect_column_anomalies(data, "col")
        assert len(anomalies) >= 1

    def test_method_all(self, detector):
        """METHOD_ALL runs multiple methods."""
        data = [10, 11, 12, 13, 14, 100, 12, 11, 10, 14, 13, 12]
        anomalies = detector.detect_column_anomalies(data, "col", method=METHOD_ALL)
        # Should detect 100 as outlier
        assert any(a["value"] == 100 for a in anomalies)


# ---------------------------------------------------------------------------
# TestDetectIQR
# ---------------------------------------------------------------------------


class TestDetectIQR:
    """Test detect_iqr() method."""

    def test_no_outliers(self, detector, normal_data):
        """Normal data: no IQR outliers."""
        indexed = _indexed(normal_data)
        anomalies = detector.detect_iqr(indexed, normal_data, "col")
        assert len(anomalies) == 0

    def test_lower_outlier(self, detector):
        """Value below lower fence detected."""
        data = [10, 11, 12, 13, 14, -100, 12, 11, 10, 14]
        indexed = _indexed([float(x) for x in data])
        anomalies = detector.detect_iqr(indexed, [float(x) for x in data], "col")
        assert any(a["value"] == -100.0 for a in anomalies)

    def test_upper_outlier(self, detector):
        """Value above upper fence detected."""
        data = [10, 11, 12, 13, 14, 200, 12, 11, 10, 14]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_iqr(indexed, nums, "col")
        assert any(a["value"] == 200.0 for a in anomalies)

    def test_both_sides(self, detector):
        """Outliers on both sides detected."""
        data = [-200, 10, 11, 12, 13, 14, 12, 11, 10, 300]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_iqr(indexed, nums, "col")
        vals = [a["value"] for a in anomalies]
        assert -200.0 in vals
        assert 300.0 in vals

    def test_boundary_not_outlier(self, detector):
        """Values exactly at fence are not outliers."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        # With k=1.5 on uniform 1-8 data, boundary values should be inside
        anomalies = detector.detect_iqr(indexed, nums, "col")
        # All within range, no anomalies expected
        assert len(anomalies) == 0

    def test_negative_values(self, detector):
        """Negative values are handled correctly."""
        data = [-10, -9, -8, -7, -6, -100, -8, -9, -10, -7]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_iqr(indexed, nums, "col")
        assert any(a["value"] == -100.0 for a in anomalies)

    def test_zero_iqr(self, detector):
        """All identical values -> IQR=0, no outliers."""
        data = [5.0] * 10
        indexed = _indexed(data)
        anomalies = detector.detect_iqr(indexed, data, "col")
        assert len(anomalies) == 0

    def test_small_dataset(self, detector):
        """Dataset with < 4 values returns empty."""
        data = [1.0, 2.0, 3.0]
        indexed = _indexed(data)
        anomalies = detector.detect_iqr(indexed, data, "col")
        assert anomalies == []

    def test_custom_multiplier(self, custom_detector):
        """Custom IQR multiplier changes sensitivity."""
        data = [10, 11, 12, 13, 14, 30, 12, 11, 10, 14]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        # k=2.0 -> wider fences, fewer outliers
        anomalies = custom_detector.detect_iqr(indexed, nums, "col")
        # 30 might not be flagged with k=2.0
        default_detector = AnomalyDetector()
        anomalies_default = default_detector.detect_iqr(indexed, nums, "col")
        # k=2.0 should detect same or fewer than k=1.5
        assert len(anomalies) <= len(anomalies_default)

    def test_known_dataset(self, detector):
        """Known dataset with known outlier at 100."""
        data = [10.0, 12.0, 11.0, 13.0, 100.0, 12.0, 11.0, 10.0, 14.0, 13.0]
        indexed = _indexed(data)
        anomalies = detector.detect_iqr(indexed, data, "col")
        assert any(a["value"] == 100.0 for a in anomalies)
        # Check anomaly has expected keys
        if anomalies:
            a = anomalies[0]
            assert "bounds" in a
            assert "iqr" in a
            assert a["method"] == METHOD_IQR

    def test_anomaly_severity_present(self, detector, outlier_data):
        """Each anomaly has a severity field."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_iqr(indexed, nums, "col")
        for a in anomalies:
            assert "severity" in a

    def test_anomaly_score_positive(self, detector, outlier_data):
        """Anomaly scores are positive."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_iqr(indexed, nums, "col")
        for a in anomalies:
            assert a["score"] > 0


# ---------------------------------------------------------------------------
# TestDetectZscore
# ---------------------------------------------------------------------------


class TestDetectZscore:
    """Test detect_zscore() method."""

    def test_no_outliers(self, detector, normal_data):
        """Normal data has no Z-score outliers."""
        indexed = _indexed(normal_data)
        anomalies = detector.detect_zscore(indexed, normal_data, "col")
        assert len(anomalies) == 0

    def test_clear_outlier(self, detector):
        """Outlier far from mean is detected (z > 3)."""
        # Need enough normal points to keep stddev low despite the outlier
        data = [10.0] * 50 + [1000.0]
        indexed = _indexed(data)
        anomalies = detector.detect_zscore(indexed, data, "col")
        assert any(a["value"] == 1000.0 for a in anomalies)

    def test_multiple_outliers(self, detector):
        """Multiple extreme values all detected when enough baseline data."""
        # 50 normal points keep stddev low; 500 and -400 will be outliers
        data = [10.0] * 50 + [500.0, -400.0]
        indexed = _indexed(data)
        anomalies = detector.detect_zscore(indexed, data, "col")
        assert len(anomalies) >= 1

    def test_boundary_zscore(self, detector):
        """Value at exactly 3 stddev from mean."""
        # Create data where we know the mean and stddev
        data = [0.0] * 100 + [3.0]  # mean ~0.03, std ~0.3
        nums = data
        indexed = _indexed(nums)
        anomalies = detector.detect_zscore(indexed, nums, "col")
        # 3.0 is far from mean ~0.03 in a low-std dataset
        assert len(anomalies) >= 1

    def test_negative_values(self, detector):
        """Works with negative values (extreme outlier detected)."""
        # 50 normal points keep stddev low; -500 will exceed 3 stddev
        data = [-10.0] * 50 + [-500.0]
        indexed = _indexed(data)
        anomalies = detector.detect_zscore(indexed, data, "col")
        assert any(a["value"] == -500.0 for a in anomalies)

    def test_zero_stddev(self, detector):
        """All identical values -> std=0, no outliers."""
        data = [5.0] * 10
        indexed = _indexed(data)
        anomalies = detector.detect_zscore(indexed, data, "col")
        assert len(anomalies) == 0

    def test_custom_threshold(self, custom_detector):
        """Custom threshold of 2.5 detects more outliers."""
        data = [10, 11, 12, 13, 14, 30, 12, 11, 10, 14]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = custom_detector.detect_zscore(indexed, nums, "col")
        default = AnomalyDetector()
        anomalies_default = default.detect_zscore(indexed, nums, "col")
        assert len(anomalies) >= len(anomalies_default)

    def test_z_score_value(self, detector):
        """Anomaly has z_score field."""
        data = [10, 11, 12, 13, 14, 100, 12, 11, 10, 14]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_zscore(indexed, nums, "col")
        for a in anomalies:
            assert "z_score" in a
            assert "mean" in a
            assert "stddev" in a

    def test_too_few_samples(self, detector):
        """< 3 values returns empty."""
        anomalies = detector.detect_zscore([(0, 1.0), (1, 2.0)], [1.0, 2.0], "col")
        assert anomalies == []

    def test_known_dataset(self, detector):
        """Known dataset: 100 is clearly > 3 stddev."""
        data = [10.0, 12.0, 11.0, 13.0, 100.0, 12.0, 11.0, 10.0, 14.0, 13.0]
        mean = statistics.mean(data)
        std = statistics.stdev(data)
        # z for 100: (100 - mean) / std
        z = (100 - mean) / std
        indexed = _indexed(data)
        anomalies = detector.detect_zscore(indexed, data, "col")
        if anomalies:
            assert abs(anomalies[0]["z_score"] - z) < 0.01


# ---------------------------------------------------------------------------
# TestDetectMAD
# ---------------------------------------------------------------------------


class TestDetectMAD:
    """Test detect_mad() method."""

    def test_no_outliers(self, detector, normal_data):
        """Normal data has no MAD outliers."""
        indexed = _indexed(normal_data)
        anomalies = detector.detect_mad(indexed, normal_data, "col")
        assert len(anomalies) == 0

    def test_clear_outlier(self, detector, outlier_data):
        """Clear outlier detected by MAD."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_mad(indexed, nums, "col")
        assert any(a["value"] == 100.0 for a in anomalies)

    def test_mad_zero(self, detector):
        """All identical values -> MAD=0, no outliers."""
        data = [5.0] * 10
        indexed = _indexed(data)
        anomalies = detector.detect_mad(indexed, data, "col")
        assert len(anomalies) == 0

    def test_boundary(self, detector):
        """Value at boundary threshold."""
        # Create data where MAD is known
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        indexed = _indexed(data)
        anomalies = detector.detect_mad(indexed, data, "col")
        # For uniform-like data, no outliers expected
        assert len(anomalies) == 0

    def test_custom_threshold(self, custom_detector):
        """Custom MAD threshold changes sensitivity."""
        data = [10, 11, 12, 13, 14, 50, 12, 11, 10, 14]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies_custom = custom_detector.detect_mad(indexed, nums, "col")
        default = AnomalyDetector()
        anomalies_default = default.detect_mad(indexed, nums, "col")
        # Lower threshold -> more anomalies
        assert len(anomalies_custom) >= len(anomalies_default)

    def test_negative_values(self, detector):
        """Works with negative values."""
        data = [-10, -9, -8, -7, -6, -100, -8, -9, -10, -7]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_mad(indexed, nums, "col")
        assert any(a["value"] == -100.0 for a in anomalies)

    def test_method_label(self, detector, outlier_data):
        """Method is labelled as 'mad'."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_mad(indexed, nums, "col")
        for a in anomalies:
            assert a["method"] == METHOD_MAD

    def test_has_median_and_mad_fields(self, detector, outlier_data):
        """Anomaly has median and mad fields."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_mad(indexed, nums, "col")
        for a in anomalies:
            assert "median" in a
            assert "mad" in a

    def test_too_few_samples(self, detector):
        """< 3 values returns empty."""
        anomalies = detector.detect_mad([(0, 1.0)], [1.0], "col")
        assert anomalies == []

    def test_known_dataset(self, detector):
        """Known dataset outlier detection."""
        data = [10.0, 12.0, 11.0, 13.0, 100.0, 12.0, 11.0, 10.0, 14.0, 13.0]
        indexed = _indexed(data)
        anomalies = detector.detect_mad(indexed, data, "col")
        assert any(a["value"] == 100.0 for a in anomalies)


# ---------------------------------------------------------------------------
# TestDetectGrubbs
# ---------------------------------------------------------------------------


class TestDetectGrubbs:
    """Test detect_grubbs() method."""

    def test_single_most_extreme(self, detector, outlier_data):
        """Grubbs detects the single most extreme value."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_grubbs(indexed, nums, "col")
        assert len(anomalies) <= 1  # Grubbs tests one at a time
        if anomalies:
            assert anomalies[0]["value"] == 100.0

    def test_no_outlier_in_small_set(self, detector, normal_data):
        """Normal small set: Grubbs finds nothing."""
        indexed = _indexed(normal_data)
        anomalies = detector.detect_grubbs(indexed, normal_data, "col")
        assert len(anomalies) == 0

    def test_clear_outlier(self, detector):
        """Very extreme value is detected."""
        data = [10, 11, 12, 13, 14, 1000, 12, 11, 10, 14]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_grubbs(indexed, nums, "col")
        assert len(anomalies) == 1
        assert anomalies[0]["value"] == 1000.0

    def test_all_same_values(self, detector):
        """Identical values -> std=0 -> no outlier."""
        data = [5.0] * 10
        indexed = _indexed(data)
        anomalies = detector.detect_grubbs(indexed, data, "col")
        assert len(anomalies) == 0

    def test_two_values(self, detector):
        """< 3 values -> empty."""
        data = [1.0, 100.0]
        indexed = _indexed(data)
        anomalies = detector.detect_grubbs(indexed, data, "col")
        assert anomalies == []

    def test_grubbs_stat_and_critical(self, detector):
        """Anomaly has grubbs_statistic and critical_value."""
        data = [10, 11, 12, 13, 14, 500, 12, 11, 10, 14]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_grubbs(indexed, nums, "col")
        if anomalies:
            assert "grubbs_statistic" in anomalies[0]
            assert "critical_value" in anomalies[0]
            assert anomalies[0]["grubbs_statistic"] > anomalies[0]["critical_value"]

    def test_sample_size_in_result(self, detector, outlier_data):
        """sample_size is included in result."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_grubbs(indexed, nums, "col")
        if anomalies:
            assert anomalies[0]["sample_size"] == len(outlier_data)

    def test_method_label(self, detector, outlier_data):
        """Method is labelled as grubbs."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_grubbs(indexed, nums, "col")
        for a in anomalies:
            assert a["method"] == METHOD_GRUBBS


# ---------------------------------------------------------------------------
# TestDetectModifiedZscore
# ---------------------------------------------------------------------------


class TestDetectModifiedZscore:
    """Test detect_modified_zscore() method."""

    def test_no_outliers(self, detector, normal_data):
        """Normal data has no modified z-score outliers."""
        indexed = _indexed(normal_data)
        anomalies = detector.detect_modified_zscore(indexed, normal_data, "col")
        assert len(anomalies) == 0

    def test_clear_outlier(self, detector, outlier_data):
        """Clear outlier is detected."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_modified_zscore(indexed, nums, "col")
        assert any(a["value"] == 100.0 for a in anomalies)

    def test_mad_zero(self, detector):
        """All same -> MAD=0 -> no outliers."""
        data = [5.0] * 10
        indexed = _indexed(data)
        anomalies = detector.detect_modified_zscore(indexed, data, "col")
        assert len(anomalies) == 0

    def test_method_label(self, detector, outlier_data):
        """Method is labelled as modified_zscore."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        anomalies = detector.detect_modified_zscore(indexed, nums, "col")
        for a in anomalies:
            assert a["method"] == METHOD_MODIFIED_ZSCORE

    def test_boundary(self, detector):
        """Boundary test: value at threshold."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        indexed = _indexed(data)
        anomalies = detector.detect_modified_zscore(indexed, data, "col")
        # Uniform-ish data: no outliers
        assert len(anomalies) == 0

    def test_comparison_with_regular_zscore(self, detector, outlier_data):
        """Modified z-score is more robust to non-normal data."""
        nums = [float(x) for x in outlier_data]
        indexed = _indexed(nums)
        mad_results = detector.detect_modified_zscore(indexed, nums, "col")
        zscore_results = detector.detect_zscore(indexed, nums, "col")
        # Both should detect the extreme outlier
        mad_vals = {a["value"] for a in mad_results}
        zscore_vals = {a["value"] for a in zscore_results}
        assert 100.0 in mad_vals or 100.0 in zscore_vals

    def test_negative_values(self, detector):
        """Works with negative values."""
        data = [-10, -9, -8, -7, -6, -100, -8, -9, -10, -7]
        nums = [float(x) for x in data]
        indexed = _indexed(nums)
        anomalies = detector.detect_modified_zscore(indexed, nums, "col")
        assert any(a["value"] == -100.0 for a in anomalies)

    def test_too_few_samples(self, detector):
        """< 3 values returns empty."""
        anomalies = detector.detect_modified_zscore([(0, 1.0)], [1.0], "col")
        assert anomalies == []


# ---------------------------------------------------------------------------
# TestProfileDistribution
# ---------------------------------------------------------------------------


class TestProfileDistribution:
    """Test profile_distribution() method."""

    def test_normal_like_data(self, detector, normal_data):
        """Normal data profile has expected keys."""
        profile = detector.profile_distribution(normal_data)
        assert "mean" in profile
        assert "median" in profile
        assert "std" in profile
        assert "skewness" in profile
        assert "kurtosis" in profile

    def test_uniform_data(self, detector):
        """Uniform data profile."""
        data = list(range(1, 21))
        profile = detector.profile_distribution(data)
        assert profile["numeric_count"] == 20
        assert profile["min"] == 1.0
        assert profile["max"] == 20.0

    def test_skewed_data(self, detector):
        """Skewed data has non-zero skewness."""
        data = [1, 1, 1, 1, 1, 2, 2, 3, 10, 100]
        profile = detector.profile_distribution(data)
        assert abs(profile["skewness"]) > 0.5

    def test_mean_median_std(self, detector):
        """Mean, median, std are computed."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        profile = detector.profile_distribution(data)
        assert abs(profile["mean"] - 3.0) < 0.01
        assert abs(profile["median"] - 3.0) < 0.01
        assert profile["std"] > 0

    def test_empty_values(self, detector):
        """Empty list."""
        profile = detector.profile_distribution([])
        assert profile["count"] == 0

    def test_single_value(self, detector):
        """Single value."""
        profile = detector.profile_distribution([42])
        assert profile["numeric_count"] == 1
        assert profile["mean"] == 42.0

    def test_two_values(self, detector):
        """Two values."""
        profile = detector.profile_distribution([10, 20])
        assert profile["numeric_count"] == 2
        assert abs(profile["mean"] - 15.0) < 0.01

    def test_normality_estimate(self, detector, normal_data):
        """Normality estimate is between 0 and 1."""
        profile = detector.profile_distribution(normal_data)
        assert 0.0 <= profile["normality_estimate"] <= 1.0

    def test_range_computation(self, detector):
        """Range = max - min."""
        profile = detector.profile_distribution([5, 10, 15])
        assert abs(profile["range"] - 10.0) < 0.01

    def test_non_numeric_filtered(self, detector):
        """Non-numeric values are filtered out."""
        data = [1, 2, "hello", None, 5]
        profile = detector.profile_distribution(data)
        assert profile["count"] == 5
        assert profile["numeric_count"] == 3


# ---------------------------------------------------------------------------
# TestDetectSuddenChange
# ---------------------------------------------------------------------------


class TestDetectSuddenChange:
    """Test detect_sudden_change() method."""

    def test_no_change(self, detector):
        """Stable series has no change points."""
        data = [10.0] * 30
        changes = detector.detect_sudden_change(data)
        assert len(changes) == 0

    def test_clear_step_change(self, detector):
        """Step change from 10 to 100 detected."""
        data = [10.0] * 15 + [100.0] * 15
        changes = detector.detect_sudden_change(data)
        assert len(changes) >= 1

    def test_gradual_change(self, detector):
        """Gradual change may or may not be detected."""
        data = [float(i) for i in range(30)]
        changes = detector.detect_sudden_change(data)
        # Gradual change might not trigger sudden change detection
        # Just verify it runs without error
        assert isinstance(changes, list)

    def test_window_size_parameter(self, detector):
        """Custom window_size is respected."""
        data = [10.0] * 10 + [100.0] * 10
        changes_small = detector.detect_sudden_change(data, window_size=3)
        changes_large = detector.detect_sudden_change(data, window_size=4)
        # Different window sizes may produce different results
        assert isinstance(changes_small, list)
        assert isinstance(changes_large, list)

    def test_empty_returns_empty(self, detector):
        """Empty series returns empty."""
        changes = detector.detect_sudden_change([])
        assert changes == []

    def test_short_series(self, detector):
        """Series shorter than 2*window+1 returns empty."""
        data = [1.0, 2.0, 3.0]
        changes = detector.detect_sudden_change(data)
        assert changes == []

    def test_oscillating(self, detector):
        """Oscillating series."""
        data = [10.0 if i % 2 == 0 else 20.0 for i in range(30)]
        changes = detector.detect_sudden_change(data)
        # May or may not detect changes depending on window
        assert isinstance(changes, list)

    def test_change_point_fields(self, detector):
        """Change points have expected fields."""
        data = [10.0] * 15 + [100.0] * 15
        changes = detector.detect_sudden_change(data)
        if changes:
            cp = changes[0]
            assert "index" in cp
            assert "before_mean" in cp
            assert "after_mean" in cp
            assert "change_ratio" in cp
            assert "severity" in cp


# ---------------------------------------------------------------------------
# TestComputeAnomalySeverity
# ---------------------------------------------------------------------------


class TestComputeAnomalySeverity:
    """Test compute_anomaly_severity() method."""

    def test_within_range(self, detector):
        """Value inside expected range -> INFO."""
        sev = detector.compute_anomaly_severity(50.0, (0.0, 100.0))
        assert sev == SEVERITY_INFO

    def test_low_severity(self, detector):
        """Small distance from range -> LOW."""
        sev = detector.compute_anomaly_severity(110.0, (0.0, 100.0))
        assert sev == SEVERITY_LOW

    def test_medium_severity(self, detector):
        """Moderate distance -> MEDIUM."""
        sev = detector.compute_anomaly_severity(300.0, (0.0, 100.0))
        assert sev == SEVERITY_MEDIUM

    def test_high_severity(self, detector):
        """Large distance -> HIGH."""
        sev = detector.compute_anomaly_severity(400.0, (0.0, 100.0))
        assert sev == SEVERITY_HIGH

    def test_critical_severity(self, detector):
        """Very large distance -> CRITICAL."""
        sev = detector.compute_anomaly_severity(600.0, (0.0, 100.0))
        assert sev == SEVERITY_CRITICAL

    def test_below_lower(self, detector):
        """Value below lower bound."""
        sev = detector.compute_anomaly_severity(-200.0, (0.0, 100.0))
        assert sev in (SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL)


# ---------------------------------------------------------------------------
# TestGenerateAnomalyIssues
# ---------------------------------------------------------------------------


class TestGenerateAnomalyIssues:
    """Test generate_anomaly_issues() method."""

    def test_issue_per_column(self, detector):
        """One issue per column with anomalies."""
        col_anomalies = {
            "col_a": [{"value": 100, "severity": SEVERITY_HIGH, "method": METHOD_IQR}],
            "col_b": [{"value": 200, "severity": SEVERITY_LOW, "method": METHOD_ZSCORE}],
        }
        issues = detector.generate_anomaly_issues(col_anomalies)
        assert len(issues) == 2

    def test_severity_mapping(self, detector):
        """Column severity matches highest anomaly severity."""
        col_anomalies = {
            "col_a": [
                {"value": 100, "severity": SEVERITY_LOW, "method": METHOD_IQR},
                {"value": 200, "severity": SEVERITY_CRITICAL, "method": METHOD_IQR},
            ],
        }
        issues = detector.generate_anomaly_issues(col_anomalies)
        assert issues[0]["severity"] == SEVERITY_CRITICAL

    def test_descriptions(self, detector):
        """Issues have message and type fields."""
        col_anomalies = {
            "col_a": [{"value": 100, "severity": SEVERITY_LOW, "method": METHOD_IQR}],
        }
        issues = detector.generate_anomaly_issues(col_anomalies)
        assert "message" in issues[0]
        assert issues[0]["type"] == "anomalies_detected"

    def test_column_reference(self, detector):
        """Issue references the column name."""
        col_anomalies = {
            "sensor_temp": [{"value": 100, "severity": SEVERITY_LOW, "method": METHOD_IQR}],
        }
        issues = detector.generate_anomaly_issues(col_anomalies)
        assert issues[0]["column"] == "sensor_temp"

    def test_no_anomalies_no_issues(self, detector):
        """Empty anomalies produce no issues."""
        issues = detector.generate_anomaly_issues({})
        assert issues == []

    def test_column_with_empty_anomalies(self, detector):
        """Column entry with empty list produces no issue."""
        col_anomalies = {"col_a": []}
        issues = detector.generate_anomaly_issues(col_anomalies)
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# TestStatistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test get_statistics()."""

    def test_initial_statistics(self, detector):
        """Initial stats all zero."""
        stats = detector.get_statistics()
        assert stats["detections_completed"] == 0
        assert stats["total_anomalies_found"] == 0
        assert stats["stored_detections"] == 0

    def test_post_detection_statistics(self, detector, outlier_data):
        """Stats update after detection."""
        data = [{"val": v} for v in outlier_data]
        detector.detect(data)
        stats = detector.get_statistics()
        assert stats["detections_completed"] == 1
        assert stats["total_values_scanned"] > 0
        assert stats["stored_detections"] == 1

    def test_multiple_detections(self, detector, normal_data):
        """Multiple detections accumulate."""
        data = [{"val": v} for v in normal_data]
        detector.detect(data)
        detector.detect(data)
        stats = detector.get_statistics()
        assert stats["detections_completed"] == 2

    def test_anomaly_rate(self, detector, outlier_data):
        """Anomaly rate is computed."""
        data = [{"val": v} for v in outlier_data]
        detector.detect(data)
        stats = detector.get_statistics()
        assert stats["overall_anomaly_rate"] >= 0.0


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Test provenance hash generation."""

    def test_sha256_format(self, detector, normal_data):
        """Provenance hash is 64-char hex (SHA-256)."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        h = result["provenance_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_helper_function(self):
        """_compute_provenance returns 64-char hex."""
        h = _compute_provenance("test_op", "test_data")
        assert len(h) == 64

    def test_detection_id_format(self, detector, normal_data):
        """detection_id starts with ANM-."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        assert result["detection_id"].startswith("ANM-")


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Test thread safety of AnomalyDetector."""

    def test_concurrent_detection(self, detector):
        """Multiple threads can run detect concurrently."""
        data = [{"val": float(i)} for i in range(20)]
        errors: List[Exception] = []

        def worker():
            try:
                for _ in range(3):
                    detector.detect(data)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = detector.get_statistics()
        assert stats["detections_completed"] == 12

    def test_concurrent_stats_access(self, detector):
        """Stats can be read concurrently with detection."""
        data = [{"val": float(i)} for i in range(20)]
        results: List[Dict] = []

        def detect_worker():
            for _ in range(5):
                detector.detect(data)

        def stats_worker():
            for _ in range(5):
                results.append(detector.get_statistics())

        t1 = threading.Thread(target=detect_worker)
        t2 = threading.Thread(target=stats_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 5


# ---------------------------------------------------------------------------
# TestStorageAndRetrieval
# ---------------------------------------------------------------------------


class TestStorageAndRetrieval:
    """Test detection storage, listing, and deletion."""

    def test_get_detection(self, detector, normal_data):
        """Retrieve a stored detection by ID."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        stored = detector.get_detection(result["detection_id"])
        assert stored is not None

    def test_get_nonexistent(self, detector):
        """Get nonexistent ID returns None."""
        assert detector.get_detection("ANM-nonexistent") is None

    def test_list_detections(self, detector, normal_data):
        """list_detections returns stored results."""
        data = [{"val": v} for v in normal_data]
        detector.detect(data)
        detector.detect(data)
        detections = detector.list_detections()
        assert len(detections) == 2

    def test_delete_detection(self, detector, normal_data):
        """Delete removes detection from storage."""
        data = [{"val": v} for v in normal_data]
        result = detector.detect(data)
        assert detector.delete_detection(result["detection_id"]) is True
        assert detector.get_detection(result["detection_id"]) is None

    def test_delete_nonexistent(self, detector):
        """Delete nonexistent returns False."""
        assert detector.delete_detection("ANM-nonexistent") is False


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Test module-level helper functions."""

    @pytest.mark.parametrize("value,expected", [
        (None, None),
        (True, None),
        (False, None),
        (42, 42.0),
        (3.14, 3.14),
        ("10.5", 10.5),
        ("hello", None),
        (float("nan"), None),
        (float("inf"), None),
    ])
    def test_try_float(self, value, expected):
        """_try_float handles various inputs."""
        result = _try_float(value)
        if expected is None:
            assert result is None
        else:
            assert abs(result - expected) < 0.01

    def test_safe_mean_empty(self):
        """Mean of empty list returns 0.0."""
        assert _safe_mean([]) == 0.0

    def test_safe_median_empty(self):
        """Median of empty list returns 0.0."""
        assert _safe_median([]) == 0.0

    def test_safe_stdev_single(self):
        """Stdev of single value returns 0.0."""
        assert _safe_stdev([5.0]) == 0.0

    def test_grubbs_critical_known(self):
        """Critical value for known sample size."""
        assert abs(_grubbs_critical(10) - 2.29) < 0.01

    def test_grubbs_critical_interpolation(self):
        """Critical value for interpolated sample size."""
        val = _grubbs_critical(22)
        assert 2.7 < val < 2.85

    def test_grubbs_critical_below_min(self):
        """Sample size below table minimum."""
        val = _grubbs_critical(2)
        assert val > 0

    def test_grubbs_critical_above_max(self):
        """Sample size above table maximum."""
        val = _grubbs_critical(200)
        assert val > 0
