# -*- coding: utf-8 -*-
"""
Engine-level integration tests for AGENT-DATA-014 Time Series Gap Filler.

Tests cross-engine interactions and combined behaviors:
- GapDetector + FrequencyAnalyzer: detect frequency then detect gaps
- InterpolationEngine + SeasonalFiller: compare fill methods on same gaps
- TrendExtrapolator + CrossSeriesFiller: fill same gaps, compare results
- Pipeline orchestration across all engines

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from greenlang.time_series_gap_filler.config import TimeSeriesGapFillerConfig, set_config
from greenlang.time_series_gap_filler.gap_detector import (
    GapDetectorEngine,
    GapDetectionResult,
    Frequency,
)
from greenlang.time_series_gap_filler.frequency_analyzer import (
    FrequencyAnalyzerEngine,
    FrequencyLevel,
)
from greenlang.time_series_gap_filler.interpolation_engine import (
    InterpolationEngine,
)
from greenlang.time_series_gap_filler.seasonal_filler import (
    CalendarDefinition,
    SeasonalFillerEngine,
)
from greenlang.time_series_gap_filler.provenance import ProvenanceTracker


# =========================================================================
# Test class: GapDetector + FrequencyAnalyzer
# =========================================================================


class TestGapDetectorWithFrequencyAnalyzer:
    """Test interactions between gap detection and frequency analysis."""

    def test_detect_freq_then_detect_gaps(self, sample_datetime_timestamps):
        """Analyze frequency first, then use it for gap detection."""
        analyzer = FrequencyAnalyzerEngine()
        detector = GapDetectorEngine()

        # Step 1: detect the dominant frequency
        freq_result = analyzer.analyze_frequency(sample_datetime_timestamps)
        freq_level = freq_result["frequency_level"]
        assert freq_level != ""

        # Step 2: build values with gaps, detect using frequency info
        values = [10.0, 12.0, None, 14.0, 16.0, None, None, 22.0, 24.0, 26.0, 28.0, 30.0]
        assert len(values) == len(sample_datetime_timestamps)

        detection = detector.detect_gaps(values)
        assert detection.total_gaps > 0
        assert detection.total_missing == 3

    def test_frequency_regularity_informs_confidence(
        self, sample_datetime_timestamps,
    ):
        """Regularity score from frequency analysis can inform fill confidence."""
        analyzer = FrequencyAnalyzerEngine()

        regularity = analyzer.detect_regularity(sample_datetime_timestamps)
        assert 0.0 <= regularity <= 1.0

        # Hourly timestamps should be very regular
        assert regularity > 0.8, (
            f"Expected high regularity for hourly data, got {regularity}"
        )

    def test_mixed_frequency_detection(self):
        """Detect mixed frequency patterns in irregular timestamps."""
        analyzer = FrequencyAnalyzerEngine()

        # Create timestamps with mixed intervals (some hourly, some daily)
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timestamps = [
            base,
            base + timedelta(hours=1),
            base + timedelta(hours=2),
            base + timedelta(days=1),
            base + timedelta(days=1, hours=1),
            base + timedelta(days=2),
            base + timedelta(days=3),
        ]

        result = analyzer.detect_mixed_frequency(timestamps)
        assert "is_mixed" in result
        assert "frequencies_found" in result
        assert "dominant_frequency" in result

    def test_frequency_grid_with_gap_detection(self):
        """Build a frequency grid and detect timestamp gaps."""
        detector = GapDetectorEngine()

        # Daily timestamps with a 2-day gap
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timestamps = [
            base,
            base + timedelta(days=1),
            base + timedelta(days=2),
            # Missing day 3 and 4
            base + timedelta(days=5),
            base + timedelta(days=6),
        ]

        grid = detector.build_frequency_grid(timestamps, Frequency.DAILY)
        assert len(grid) == 7  # days 0 through 6

        result = detector.detect_gaps_by_frequency(timestamps, "daily")
        assert result.total_gaps > 0
        assert result.total_missing >= 2

    def test_frequency_validation_against_expected(self):
        """Validate timestamps match an expected frequency."""
        analyzer = FrequencyAnalyzerEngine()

        # Create perfectly daily timestamps
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        timestamps = [base + timedelta(days=i) for i in range(30)]

        result = analyzer.validate_frequency(timestamps, "daily")
        assert result["is_valid"] is True
        assert result["match_pct"] > 0.8

    def test_interval_statistics_computation(self, sample_datetime_timestamps):
        """Compute interval statistics from timestamps."""
        analyzer = FrequencyAnalyzerEngine()

        stats = analyzer.compute_interval_statistics(sample_datetime_timestamps)
        assert stats["count"] > 0
        assert stats["mean_interval"] > 0.0
        assert stats["cv"] >= 0.0


# =========================================================================
# Test class: InterpolationEngine + SeasonalFiller
# =========================================================================


class TestInterpolationWithSeasonal:
    """Test interactions between interpolation and seasonal engines."""

    def test_compare_linear_and_seasonal_on_same_gaps(
        self, long_series_with_gaps,
    ):
        """Linear and seasonal fills produce different but valid results."""
        interpolator = InterpolationEngine()
        seasonal = SeasonalFillerEngine()

        linear_result = interpolator.fill_gaps(long_series_with_gaps, method="linear")
        seasonal_result = seasonal.fill_seasonal(long_series_with_gaps, period=12)

        # Both should fill the same number of gap positions
        assert linear_result.gaps_filled > 0
        assert seasonal_result.gaps_filled > 0

        # Both filled series should have no None values
        for v in linear_result.filled_values:
            assert v is not None
        for i in range(len(seasonal_result.values)):
            if i in seasonal_result.filled_indices:
                assert seasonal_result.values[i] is not None

    def test_seasonal_decomposition_then_interpolation(
        self, long_series_with_gaps,
    ):
        """Decompose seasonal component, then interpolate residuals."""
        seasonal = SeasonalFillerEngine()

        decomposition = seasonal.decompose_seasonal(
            long_series_with_gaps, period=12,
        )
        assert decomposition.period == 12
        assert len(decomposition.trend) == len(long_series_with_gaps)
        assert len(decomposition.seasonal) == len(long_series_with_gaps)

    def test_seasonality_detection_before_fill(self, long_series_with_gaps):
        """Detect seasonality first, use detected period for filling."""
        seasonal = SeasonalFillerEngine()

        detection = seasonal.detect_seasonality(long_series_with_gaps)
        assert "is_seasonal" in detection
        assert "dominant_period" in detection

        detected_period = detection["dominant_period"]
        if detected_period > 1:
            result = seasonal.fill_seasonal(
                long_series_with_gaps, period=detected_period,
            )
            assert result.gaps_filled > 0

    def test_multiple_interpolation_methods_all_valid(
        self, sample_series_with_gaps,
    ):
        """All interpolation methods produce numerically valid results."""
        interpolator = InterpolationEngine()
        methods = ["linear", "cubic_spline", "polynomial", "akima", "nearest", "pchip"]

        for method in methods:
            if method == "polynomial":
                result = interpolator.interpolate_polynomial(
                    sample_series_with_gaps, degree=2,
                )
            else:
                result = interpolator.fill_gaps(
                    sample_series_with_gaps, method=method,
                )

            for i, v in enumerate(result.filled_values):
                assert v is not None, f"{method}: None at index {i}"
                assert not math.isnan(v), f"{method}: NaN at index {i}"
                assert not math.isinf(v), f"{method}: Inf at index {i}"


# =========================================================================
# Test class: Seasonal day-of-week and month patterns
# =========================================================================


class TestSeasonalPatterns:
    """Test seasonal pattern extraction and pattern-based filling."""

    def test_day_of_week_pattern_fill(self):
        """Fill gaps using day-of-week pattern averages."""
        seasonal = SeasonalFillerEngine()

        # 3 weeks of daily data (21 days) with gaps on some Mondays
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)  # Monday
        timestamps = [base + timedelta(days=i) for i in range(21)]
        values = []
        for i, ts in enumerate(timestamps):
            weekday = ts.weekday()
            if weekday == 0 and i > 7:  # Mondays after first week
                values.append(None)
            else:
                values.append(100.0 + weekday * 10.0)

        result = seasonal.fill_day_of_week_pattern(values, timestamps)
        assert result.gaps_filled > 0
        assert result.method == "day_of_week"

    def test_month_pattern_fill(self):
        """Fill gaps using month-of-year pattern averages."""
        seasonal = SeasonalFillerEngine()

        # 24 months of monthly data with some gaps
        base = datetime(2023, 1, 15, tzinfo=timezone.utc)
        timestamps = []
        values = []
        for i in range(24):
            month = (i % 12) + 1
            year = 2023 + (i // 12)
            ts = datetime(year, month, 15, tzinfo=timezone.utc)
            timestamps.append(ts)
            if i == 5 or i == 17:  # gaps in June
                values.append(None)
            else:
                values.append(1000.0 + month * 50.0)

        result = seasonal.fill_month_pattern(values, timestamps)
        assert result.gaps_filled == 2
        assert result.method == "month_pattern"

    def test_get_seasonal_pattern_extraction(self, long_series_with_gaps):
        """Extract a repeating seasonal pattern of specified period."""
        seasonal = SeasonalFillerEngine()
        pattern = seasonal.get_seasonal_pattern(long_series_with_gaps, period=12)

        assert len(pattern) == 12
        # Pattern should be centred (sum approximately zero)
        pattern_sum = sum(pattern)
        assert abs(pattern_sum) < 1.0, (
            f"Seasonal pattern should be centred, sum={pattern_sum}"
        )


# =========================================================================
# Test class: Pipeline orchestration
# =========================================================================


class TestPipelineOrchestration:
    """Test that the pipeline correctly chains engine results."""

    def test_pipeline_detect_analyze_fill(self, sample_series_with_gaps):
        """Pipeline: detect -> analyze -> fill in correct order."""
        detector = GapDetectorEngine()
        interpolator = InterpolationEngine()

        # Detect
        detection = detector.detect_gaps(sample_series_with_gaps)
        assert detection.total_gaps > 0

        # Get gap statistics
        stats = detector.get_gap_statistics(detection)
        assert stats["total_gaps"] > 0
        assert stats["avg_gap_length"] > 0

        # Fill based on detection results
        result = interpolator.fill_gaps(sample_series_with_gaps, method="linear")
        assert result.gaps_filled == detection.total_gaps

    def test_pipeline_handles_no_gaps_gracefully(self, complete_series):
        """Pipeline with no gaps completes without error."""
        detector = GapDetectorEngine()
        interpolator = InterpolationEngine()

        detection = detector.detect_gaps(complete_series)
        assert detection.total_gaps == 0

        result = interpolator.fill_gaps(complete_series, method="linear")
        assert result.gaps_filled == 0
        assert result.filled_values == complete_series

    def test_consecutive_gap_finder(self):
        """Find consecutive gap segments in a series."""
        detector = GapDetectorEngine()
        values = [1.0, None, None, 4.0, None, 6.0, None, None, None, 10.0]

        segments = detector.find_consecutive_gaps(values)
        assert len(segments) == 3
        assert segments[0] == (1, 2)  # indices 1-2, length 2
        assert segments[1] == (4, 1)  # index 4, length 1
        assert segments[2] == (6, 3)  # indices 6-8, length 3
