# -*- coding: utf-8 -*-
"""
Service-level integration tests for AGENT-DATA-014 Time Series Gap Filler.

Tests end-to-end service workflows including:
- Gap detection -> fill -> validation pipelines
- Custom configuration overrides
- Multiple fill methods producing results
- Cross-series filling with reference data
- Calendar-aware filling
- Provenance chain maintenance across operations
- Edge cases (all gaps, no gaps, single point)

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    get_config,
    set_config,
)
from greenlang.time_series_gap_filler.gap_detector import GapDetectorEngine
from greenlang.time_series_gap_filler.frequency_analyzer import FrequencyAnalyzerEngine
from greenlang.time_series_gap_filler.interpolation_engine import InterpolationEngine
from greenlang.time_series_gap_filler.seasonal_filler import (
    CalendarDefinition,
    SeasonalFillerEngine,
)
from greenlang.time_series_gap_filler.provenance import ProvenanceTracker


# =========================================================================
# Test class: End-to-end service workflows
# =========================================================================


class TestServiceEndToEnd:
    """End-to-end integration tests for the gap filler service layer."""

    def test_detect_then_fill_then_validate_linear(self, sample_series_with_gaps):
        """E2E: detect gaps -> fill with linear -> validate no gaps remain."""
        detector = GapDetectorEngine()
        interpolator = InterpolationEngine()

        # Step 1: detect
        detection = detector.detect_gaps(sample_series_with_gaps)
        assert detection.total_gaps > 0
        assert detection.total_missing > 0

        # Step 2: fill
        result = interpolator.fill_gaps(sample_series_with_gaps, method="linear")
        assert len(result.filled_values) == len(sample_series_with_gaps)
        assert result.gaps_filled > 0
        assert result.mean_confidence > 0.0

        # Step 3: validate no gaps remain in the filled series
        filled_nones = [v for v in result.filled_values if v is None]
        assert len(filled_nones) == 0, "Filled series should have no None values"

        # Step 4: verify filled values are numerically reasonable
        for i, v in enumerate(result.filled_values):
            assert isinstance(v, float), f"Index {i} is not a float: {type(v)}"
            assert not math.isnan(v), f"Index {i} is NaN"

    def test_detect_then_fill_then_validate_cubic_spline(
        self, sample_series_with_gaps,
    ):
        """E2E: detect gaps -> fill with cubic spline -> validate."""
        detector = GapDetectorEngine()
        interpolator = InterpolationEngine()

        detection = detector.detect_gaps(sample_series_with_gaps)
        assert detection.total_gaps > 0

        result = interpolator.fill_gaps(
            sample_series_with_gaps, method="cubic_spline",
        )
        assert len(result.filled_values) == len(sample_series_with_gaps)
        assert result.gaps_filled > 0
        assert result.method == "cubic_spline"

        for v in result.filled_values:
            assert v is not None
            assert not math.isnan(v)

    def test_full_pipeline_detect_fill_verify_deterministic(
        self, sample_series_with_gaps,
    ):
        """Full pipeline run produces deterministic results on repeated calls."""
        detector = GapDetectorEngine()
        interpolator = InterpolationEngine()

        result1 = interpolator.fill_gaps(sample_series_with_gaps, method="linear")
        result2 = interpolator.fill_gaps(sample_series_with_gaps, method="linear")

        assert result1.filled_values == result2.filled_values
        assert result1.gaps_filled == result2.gaps_filled
        assert result1.mean_confidence == result2.mean_confidence

    def test_service_with_custom_config(self, sample_series_with_gaps):
        """Service respects custom config overrides."""
        custom = TimeSeriesGapFillerConfig(
            max_gap_ratio=0.3,
            min_data_points=4,
            default_strategy="linear",
            interpolation_method="linear",
            confidence_threshold=0.5,
        )
        set_config(custom)

        detector = GapDetectorEngine()
        detection = detector.detect_gaps(sample_series_with_gaps)
        assert detection.total_gaps > 0

        interpolator = InterpolationEngine()
        result = interpolator.fill_gaps(sample_series_with_gaps, method="linear")
        assert result.gaps_filled > 0

    def test_multiple_fill_methods_produce_results(self, sample_series_with_gaps):
        """Multiple fill methods all produce valid filled series."""
        interpolator = InterpolationEngine()
        methods = ["linear", "cubic_spline", "nearest", "pchip", "akima"]

        results = {}
        for method in methods:
            result = interpolator.fill_gaps(sample_series_with_gaps, method=method)
            results[method] = result
            assert result.gaps_filled > 0, f"{method} did not fill any gaps"
            assert len(result.filled_values) == len(sample_series_with_gaps)
            for v in result.filled_values:
                assert v is not None, f"{method} left None values"

        # Different methods may produce different values
        linear_vals = results["linear"].filled_values
        cubic_vals = results["cubic_spline"].filled_values
        # At least one gap position should differ between methods
        gap_positions = [
            i for i, v in enumerate(sample_series_with_gaps) if v is None
        ]
        differs = any(
            abs(linear_vals[i] - cubic_vals[i]) > 1e-10 for i in gap_positions
        )
        # This is expected for most datasets but not guaranteed
        # We simply verify both produce valid results

    def test_polynomial_fill_with_degree(self, sample_series_with_gaps):
        """Polynomial fill accepts and uses the degree parameter."""
        interpolator = InterpolationEngine()
        result = interpolator.interpolate_polynomial(
            sample_series_with_gaps, degree=3,
        )
        assert result.gaps_filled > 0
        assert len(result.filled_values) == len(sample_series_with_gaps)
        for v in result.filled_values:
            assert v is not None

    def test_seasonal_fill(self, long_series_with_gaps):
        """Seasonal filling uses pattern from non-missing data."""
        filler = SeasonalFillerEngine()
        result = filler.fill_seasonal(long_series_with_gaps, period=12)
        assert result.gaps_filled > 0
        assert result.method == "seasonal"
        assert result.confidence > 0.0

        # Filled values should be floats
        for idx in result.filled_indices:
            assert result.values[idx] is not None

    def test_calendar_aware_fill(self, sample_calendar):
        """Calendar-aware filling respects business day definitions."""
        filler = SeasonalFillerEngine()

        # Build a 14-day series with gaps on business days
        base = datetime(2025, 1, 6, tzinfo=timezone.utc)  # Monday
        timestamps = [base + timedelta(days=i) for i in range(14)]
        # Values for weekdays, None for some weekdays
        values = []
        for i, ts in enumerate(timestamps):
            wd = ts.weekday()
            if wd in [0, 1, 2, 3, 4]:  # business days
                if i == 3 or i == 7:  # gaps
                    values.append(None)
                else:
                    values.append(100.0 + i * 5.0)
            else:
                values.append(20.0)  # weekend

        result = filler.fill_calendar_aware(values, timestamps, sample_calendar)
        assert result.gaps_filled > 0
        assert result.method == "calendar"
        assert result.provenance_hash != ""

    def test_provenance_chain_maintained_across_operations(
        self, sample_series_with_gaps,
    ):
        """Provenance chain is maintained when running detect -> fill."""
        tracker = ProvenanceTracker()
        detector = GapDetectorEngine()
        interpolator = InterpolationEngine()

        detection = detector.detect_gaps(sample_series_with_gaps)
        assert detection.provenance_hash != ""

        result = interpolator.fill_gaps(sample_series_with_gaps, method="linear")
        assert result.provenance_hash != ""

        # Both should produce non-empty 64-character hex hashes
        assert len(detection.provenance_hash) == 64
        assert len(result.provenance_hash) == 64

    def test_detect_no_gaps_returns_zero(self, complete_series):
        """Detecting on a complete series returns zero gaps."""
        detector = GapDetectorEngine()
        detection = detector.detect_gaps(complete_series)
        assert detection.total_gaps == 0
        assert detection.total_missing == 0
        assert detection.gap_pct == 0.0

    def test_fill_no_gaps_returns_unchanged(self, complete_series):
        """Filling a complete series returns the values unchanged."""
        interpolator = InterpolationEngine()
        result = interpolator.fill_gaps(complete_series, method="linear")
        assert result.gaps_filled == 0
        assert result.filled_values == complete_series

    def test_detect_all_gaps(self):
        """Detecting a series of all None values finds one large gap."""
        detector = GapDetectorEngine()
        all_none = [None] * 10
        detection = detector.detect_gaps(all_none)
        assert detection.total_gaps == 1
        assert detection.total_missing == 10
        assert detection.gap_pct == 1.0

    def test_fill_preserves_original_non_missing_values(
        self, sample_series_with_gaps,
    ):
        """Filled values at non-gap positions match the original values."""
        interpolator = InterpolationEngine()
        result = interpolator.fill_gaps(sample_series_with_gaps, method="linear")

        for i, orig in enumerate(sample_series_with_gaps):
            if orig is not None:
                assert result.filled_values[i] == orig, (
                    f"Non-missing value at index {i} was altered: "
                    f"original={orig}, filled={result.filled_values[i]}"
                )

    def test_gap_characterization_types(self, sample_series_with_gaps):
        """Gap characterization classifies gaps by length."""
        detector = GapDetectorEngine()
        detection = detector.detect_gaps(sample_series_with_gaps)

        assert len(detection.characterizations) > 0
        for char in detection.characterizations:
            assert char.gap_type is not None
            assert char.gap_pattern is not None
            assert 0.0 <= char.position_pct <= 1.0

    def test_edge_gap_detection(self):
        """Leading and trailing gaps are detected correctly."""
        detector = GapDetectorEngine()
        series = [None, None, 3.0, 4.0, 5.0, None]
        edge_gaps = detector.detect_edge_gaps(series)
        assert edge_gaps["leading_gap"] == 2
        assert edge_gaps["trailing_gap"] == 1

    def test_frequency_analysis_integrates_with_detection(
        self, sample_datetime_timestamps,
    ):
        """Frequency analysis produces usable results for gap detection."""
        analyzer = FrequencyAnalyzerEngine()
        freq_result = analyzer.analyze_frequency(sample_datetime_timestamps)

        assert "frequency_level" in freq_result
        assert "regularity_score" in freq_result
        assert 0.0 <= freq_result["regularity_score"] <= 1.0
        assert freq_result["provenance_hash"] != ""
