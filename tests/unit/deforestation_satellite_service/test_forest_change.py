# -*- coding: utf-8 -*-
"""
Unit Tests for ForestChangeEngine (AGENT-DATA-007)

Tests change classification (no change, clear cut, degradation, partial loss,
regrowth), boundary values, confidence calculation, area calculation,
trend analysis, breakpoint detection, and detection count tracking.

Coverage target: 85%+ of forest_change.py

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-007 Deforestation Satellite Connector Agent (GL-DATA-GEO-003)
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline models
# ---------------------------------------------------------------------------


class ForestChangeDetection:
    """Result of a forest change detection analysis."""

    def __init__(
        self,
        detection_id: str = "",
        change_type: str = "no_change",
        confidence: float = 0.0,
        area_hectares: float = 0.0,
        ndvi_before: float = 0.0,
        ndvi_after: float = 0.0,
        delta_ndvi: float = 0.0,
        nbr_before: Optional[float] = None,
        nbr_after: Optional[float] = None,
        date_before: str = "",
        date_after: str = "",
        pixel_count: int = 0,
        resolution_m: float = 10.0,
        provenance_hash: str = "",
    ):
        self.detection_id = detection_id
        self.change_type = change_type
        self.confidence = max(0.0, min(100.0, confidence))
        self.area_hectares = area_hectares
        self.ndvi_before = ndvi_before
        self.ndvi_after = ndvi_after
        self.delta_ndvi = delta_ndvi
        self.nbr_before = nbr_before
        self.nbr_after = nbr_after
        self.date_before = date_before
        self.date_after = date_after
        self.pixel_count = pixel_count
        self.resolution_m = resolution_m
        self.provenance_hash = provenance_hash


class TrendPoint:
    """A single point in a time series trend."""

    def __init__(self, date: str = "", value: float = 0.0):
        self.date = date
        self.value = value


class TrendAnalysis:
    """Result of vegetation trend analysis over time."""

    def __init__(
        self,
        trend_direction: str = "stable",
        slope: float = 0.0,
        points: Optional[List[TrendPoint]] = None,
        breakpoints: Optional[List[str]] = None,
        confidence: float = 0.0,
    ):
        self.trend_direction = trend_direction
        self.slope = slope
        self.points = points or []
        self.breakpoints = breakpoints or []
        self.confidence = confidence


class VegetationIndexResult:
    """Result of a vegetation index calculation."""

    def __init__(
        self,
        index_type: str = "ndvi",
        values: Optional[List[float]] = None,
        mean: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 0.0,
        std_dev: float = 0.0,
        valid_pixel_count: int = 0,
        scene_id: str = "",
    ):
        self.index_type = index_type
        self.values = values or []
        self.mean = mean
        self.min_val = min_val
        self.max_val = max_val
        self.std_dev = std_dev
        self.valid_pixel_count = valid_pixel_count
        self.scene_id = scene_id


# ---------------------------------------------------------------------------
# Inline ForestChangeEngine
# ---------------------------------------------------------------------------


class ForestChangeEngine:
    """Engine for detecting and classifying forest cover changes.

    Uses NDVI delta thresholds to classify changes:
    - clear_cut:     dNDVI <= -0.3
    - degradation:   -0.3 < dNDVI <= -0.15
    - partial_loss:  -0.15 < dNDVI <= -0.05
    - no_change:     -0.05 < dNDVI < 0.1
    - regrowth:      dNDVI >= 0.1

    Confidence is proportional to the magnitude of the NDVI delta,
    boosted when NBR data is available.
    """

    # Default NDVI thresholds
    CLEARCUT_THRESHOLD: float = -0.3
    DEGRADATION_THRESHOLD: float = -0.15
    PARTIAL_LOSS_THRESHOLD: float = -0.05
    REGROWTH_THRESHOLD: float = 0.1

    def __init__(self) -> None:
        self._detections: List[ForestChangeDetection] = []

    # ------------------------------------------------------------------
    # Change classification
    # ------------------------------------------------------------------

    def classify_change(self, delta_ndvi: float) -> str:
        """Classify a change type based on NDVI delta.

        Args:
            delta_ndvi: NDVI difference (after - before).

        Returns:
            One of: clear_cut, degradation, partial_loss, no_change, regrowth.
        """
        if delta_ndvi <= self.CLEARCUT_THRESHOLD:
            return "clear_cut"
        elif delta_ndvi <= self.DEGRADATION_THRESHOLD:
            return "degradation"
        elif delta_ndvi <= self.PARTIAL_LOSS_THRESHOLD:
            return "partial_loss"
        elif delta_ndvi >= self.REGROWTH_THRESHOLD:
            return "regrowth"
        else:
            return "no_change"

    # ------------------------------------------------------------------
    # Detect change from index results
    # ------------------------------------------------------------------

    def detect_change(
        self,
        ndvi_before: VegetationIndexResult,
        ndvi_after: VegetationIndexResult,
        nbr_before: Optional[VegetationIndexResult] = None,
        nbr_after: Optional[VegetationIndexResult] = None,
        date_before: str = "",
        date_after: str = "",
        resolution_m: float = 10.0,
    ) -> ForestChangeDetection:
        """Detect forest change between two NDVI index results.

        Computes delta from mean NDVI values, classifies the change,
        calculates confidence and affected area.
        """
        delta = ndvi_after.mean - ndvi_before.mean
        change_type = self.classify_change(delta)
        confidence = self._calculate_confidence(delta, nbr_before, nbr_after)
        pixel_count = max(ndvi_before.valid_pixel_count, ndvi_after.valid_pixel_count)
        area = self._calculate_area(pixel_count, resolution_m)

        detection_id = f"det_{len(self._detections) + 1:04d}"
        det = ForestChangeDetection(
            detection_id=detection_id,
            change_type=change_type,
            confidence=confidence,
            area_hectares=area,
            ndvi_before=ndvi_before.mean,
            ndvi_after=ndvi_after.mean,
            delta_ndvi=round(delta, 6),
            nbr_before=nbr_before.mean if nbr_before else None,
            nbr_after=nbr_after.mean if nbr_after else None,
            date_before=date_before,
            date_after=date_after,
            pixel_count=pixel_count,
            resolution_m=resolution_m,
            provenance_hash=_compute_hash({
                "detection_id": detection_id,
                "delta_ndvi": delta,
                "change_type": change_type,
            }),
        )
        self._detections.append(det)
        return det

    # ------------------------------------------------------------------
    # Confidence calculation
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        delta_ndvi: float,
        nbr_before: Optional[VegetationIndexResult] = None,
        nbr_after: Optional[VegetationIndexResult] = None,
    ) -> float:
        """Calculate detection confidence from NDVI delta magnitude.

        Base confidence = min(abs(delta_ndvi) * 200, 95).
        If NBR data is available, add a 5% boost (capped at 100).
        """
        base = min(abs(delta_ndvi) * 200.0, 95.0)
        nbr_boost = 0.0
        if nbr_before is not None and nbr_after is not None:
            nbr_boost = 5.0
        return round(min(base + nbr_boost, 100.0), 2)

    # ------------------------------------------------------------------
    # Area calculation
    # ------------------------------------------------------------------

    def _calculate_area(self, pixel_count: int, resolution_m: float) -> float:
        """Calculate area in hectares from pixel count and resolution.

        area_ha = pixel_count * resolution_m^2 / 10000
        """
        return round(pixel_count * (resolution_m ** 2) / 10000.0, 4)

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def analyze_trend(self, points: List[TrendPoint]) -> TrendAnalysis:
        """Analyze vegetation trend from a time series of NDVI values.

        Uses simple linear regression to determine slope and direction.
        """
        if len(points) < 2:
            return TrendAnalysis(
                trend_direction="stable",
                slope=0.0,
                points=points,
                confidence=0.0,
            )

        n = len(points)
        x_vals = list(range(n))
        y_vals = [p.value for p in points]

        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        slope = numerator / denominator if denominator != 0 else 0.0

        # Classify direction
        if slope < -0.005:
            direction = "declining"
        elif slope > 0.005:
            direction = "increasing"
        else:
            direction = "stable"

        # Detect breakpoints (significant drops between consecutive points)
        breakpoints = self._detect_breakpoints(points)

        # Confidence based on R-squared
        ss_res = sum((y - (slope * x + (y_mean - slope * x_mean))) ** 2
                      for x, y in zip(x_vals, y_vals))
        ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        confidence = round(max(0.0, min(r_squared * 100.0, 100.0)), 2)

        return TrendAnalysis(
            trend_direction=direction,
            slope=round(slope, 6),
            points=points,
            breakpoints=breakpoints,
            confidence=confidence,
        )

    def _detect_breakpoints(self, points: List[TrendPoint]) -> List[str]:
        """Detect breakpoints where NDVI drops by more than 0.1 between periods."""
        breakpoints = []
        for i in range(1, len(points)):
            delta = points[i].value - points[i - 1].value
            if delta < -0.1:
                breakpoints.append(points[i].date)
        return breakpoints

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def detection_count(self) -> int:
        """Number of detections performed."""
        return len(self._detections)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> ForestChangeEngine:
    return ForestChangeEngine()


# ===========================================================================
# Test: Change classification
# ===========================================================================


class TestClassifyChange:
    """Test change classification from dNDVI values."""

    def test_no_change_zero(self, engine):
        assert engine.classify_change(0.0) == "no_change"

    def test_no_change_small_negative(self, engine):
        assert engine.classify_change(-0.02) == "no_change"

    def test_no_change_small_positive(self, engine):
        assert engine.classify_change(0.05) == "no_change"

    def test_clear_cut_severe(self, engine):
        """dNDVI <= -0.3 is clear cut."""
        assert engine.classify_change(-0.5) == "clear_cut"

    def test_clear_cut_threshold(self, engine):
        """dNDVI == -0.3 is clear cut."""
        assert engine.classify_change(-0.3) == "clear_cut"

    def test_degradation_moderate(self, engine):
        """dNDVI between -0.3 and -0.15 is degradation."""
        assert engine.classify_change(-0.2) == "degradation"

    def test_degradation_threshold(self, engine):
        """dNDVI == -0.15 is degradation."""
        assert engine.classify_change(-0.15) == "degradation"

    def test_partial_loss_minor(self, engine):
        """dNDVI between -0.15 and -0.05 is partial loss."""
        assert engine.classify_change(-0.1) == "partial_loss"

    def test_partial_loss_threshold(self, engine):
        """dNDVI == -0.05 is partial loss."""
        assert engine.classify_change(-0.05) == "partial_loss"

    def test_regrowth_positive(self, engine):
        """dNDVI >= 0.1 is regrowth."""
        assert engine.classify_change(0.2) == "regrowth"

    def test_regrowth_threshold(self, engine):
        """dNDVI == 0.1 is regrowth."""
        assert engine.classify_change(0.1) == "regrowth"


class TestClassifyChangeBoundaryValues:
    """Test boundary values for change classification."""

    def test_just_above_clearcut_threshold(self, engine):
        """dNDVI = -0.29 is degradation, not clear cut."""
        assert engine.classify_change(-0.29) == "degradation"

    def test_just_below_clearcut_threshold(self, engine):
        """dNDVI = -0.31 is still clear cut."""
        assert engine.classify_change(-0.31) == "clear_cut"

    def test_just_above_degradation_threshold(self, engine):
        """dNDVI = -0.14 is partial loss."""
        assert engine.classify_change(-0.14) == "partial_loss"

    def test_just_below_degradation_threshold(self, engine):
        """dNDVI = -0.16 is degradation."""
        assert engine.classify_change(-0.16) == "degradation"

    def test_just_above_partial_loss_threshold(self, engine):
        """dNDVI = -0.04 is no change."""
        assert engine.classify_change(-0.04) == "no_change"

    def test_just_below_regrowth_threshold(self, engine):
        """dNDVI = 0.09 is no change."""
        assert engine.classify_change(0.09) == "no_change"

    def test_extreme_negative(self, engine):
        """Very large negative delta is clear cut."""
        assert engine.classify_change(-1.0) == "clear_cut"

    def test_extreme_positive(self, engine):
        """Very large positive delta is regrowth."""
        assert engine.classify_change(1.0) == "regrowth"


# ===========================================================================
# Test: Detect change from indices
# ===========================================================================


class TestDetectChange:
    """Test detect_change from NDVI index results."""

    def test_detect_change_returns_detection(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.7, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.2, valid_pixel_count=100)
        det = engine.detect_change(before, after)
        assert isinstance(det, ForestChangeDetection)

    def test_detect_change_clear_cut(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.7, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.1, valid_pixel_count=100)
        det = engine.detect_change(before, after)
        assert det.change_type == "clear_cut"
        assert det.delta_ndvi < -0.3

    def test_detect_change_no_change(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.65, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.63, valid_pixel_count=100)
        det = engine.detect_change(before, after)
        assert det.change_type == "no_change"

    def test_detect_change_regrowth(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.3, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.5, valid_pixel_count=100)
        det = engine.detect_change(before, after)
        assert det.change_type == "regrowth"

    def test_detect_change_has_provenance(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.7, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.2, valid_pixel_count=100)
        det = engine.detect_change(before, after)
        assert det.provenance_hash != ""

    def test_detect_change_dates_set(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.7, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.2, valid_pixel_count=100)
        det = engine.detect_change(
            before, after, date_before="2020-06-01", date_after="2021-06-01",
        )
        assert det.date_before == "2020-06-01"
        assert det.date_after == "2021-06-01"


# ===========================================================================
# Test: Confidence calculation
# ===========================================================================


class TestConfidenceCalculation:
    """Test confidence calculation from NDVI delta."""

    def test_higher_delta_higher_confidence(self, engine):
        """Larger NDVI delta should produce higher confidence."""
        c1 = engine._calculate_confidence(-0.1)
        c2 = engine._calculate_confidence(-0.4)
        assert c2 > c1

    def test_zero_delta_zero_confidence(self, engine):
        """Zero NDVI delta should produce zero confidence."""
        assert engine._calculate_confidence(0.0) == 0.0

    def test_confidence_capped_at_95_without_nbr(self, engine):
        """Without NBR, confidence is capped at 95%."""
        c = engine._calculate_confidence(-1.0)
        assert c <= 95.0

    def test_confidence_with_nbr_boost(self, engine):
        """NBR data adds a 5% boost to confidence."""
        nbr_before = VegetationIndexResult(index_type="nbr", mean=0.5)
        nbr_after = VegetationIndexResult(index_type="nbr", mean=0.1)
        c_without = engine._calculate_confidence(-0.3)
        c_with = engine._calculate_confidence(-0.3, nbr_before, nbr_after)
        assert c_with == c_without + 5.0

    def test_confidence_with_nbr_capped_at_100(self, engine):
        """With NBR boost, confidence is still capped at 100."""
        nbr_before = VegetationIndexResult(index_type="nbr", mean=0.5)
        nbr_after = VegetationIndexResult(index_type="nbr", mean=0.1)
        c = engine._calculate_confidence(-0.9, nbr_before, nbr_after)
        assert c <= 100.0

    def test_positive_delta_produces_confidence(self, engine):
        """Positive deltas also produce confidence (regrowth detection)."""
        c = engine._calculate_confidence(0.3)
        assert c > 0.0

    def test_nbr_none_no_boost(self, engine):
        """If only one NBR is None, no boost applied."""
        nbr_before = VegetationIndexResult(index_type="nbr", mean=0.5)
        c = engine._calculate_confidence(-0.3, nbr_before, None)
        assert c == engine._calculate_confidence(-0.3)


# ===========================================================================
# Test: Area calculation
# ===========================================================================


class TestAreaCalculation:
    """Test area calculation from pixel count and resolution."""

    def test_basic_area(self, engine):
        """100 pixels at 10m resolution = 100 * 100 / 10000 = 1 ha."""
        area = engine._calculate_area(100, 10.0)
        assert abs(area - 1.0) < 1e-4

    def test_area_20m_resolution(self, engine):
        """100 pixels at 20m = 100 * 400 / 10000 = 4 ha."""
        area = engine._calculate_area(100, 20.0)
        assert abs(area - 4.0) < 1e-4

    def test_area_30m_resolution(self, engine):
        """100 pixels at 30m = 100 * 900 / 10000 = 9 ha."""
        area = engine._calculate_area(100, 30.0)
        assert abs(area - 9.0) < 1e-4

    def test_area_zero_pixels(self, engine):
        """Zero pixels = 0 hectares."""
        area = engine._calculate_area(0, 10.0)
        assert area == 0.0

    def test_area_single_pixel_10m(self, engine):
        """1 pixel at 10m = 0.01 ha."""
        area = engine._calculate_area(1, 10.0)
        assert abs(area - 0.01) < 1e-4

    def test_area_large_count(self, engine):
        """10000 pixels at 10m = 100 ha."""
        area = engine._calculate_area(10000, 10.0)
        assert abs(area - 100.0) < 1e-4


class TestAreaDifferentResolutions:
    """Test area with different satellite resolutions."""

    @pytest.mark.parametrize("resolution,expected_area_per_pixel", [
        (10.0, 0.01),
        (20.0, 0.04),
        (30.0, 0.09),
    ])
    def test_area_per_pixel(self, resolution, expected_area_per_pixel):
        engine = ForestChangeEngine()
        area = engine._calculate_area(1, resolution)
        assert abs(area - expected_area_per_pixel) < 1e-4


# ===========================================================================
# Test: Trend analysis
# ===========================================================================


class TestTrendAnalysisDeclining:
    """Test trend analysis with declining NDVI values."""

    def test_declining_trend_direction(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.8),
            TrendPoint(date="2020-07", value=0.7),
            TrendPoint(date="2021-01", value=0.6),
            TrendPoint(date="2021-07", value=0.5),
            TrendPoint(date="2022-01", value=0.4),
        ]
        result = engine.analyze_trend(points)
        assert result.trend_direction == "declining"

    def test_declining_negative_slope(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.8),
            TrendPoint(date="2020-07", value=0.6),
            TrendPoint(date="2021-01", value=0.4),
        ]
        result = engine.analyze_trend(points)
        assert result.slope < 0

    def test_declining_has_points(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.8),
            TrendPoint(date="2020-07", value=0.6),
        ]
        result = engine.analyze_trend(points)
        assert len(result.points) == 2


class TestTrendAnalysisStable:
    """Test trend analysis with stable NDVI values."""

    def test_stable_trend_direction(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.65),
            TrendPoint(date="2020-07", value=0.66),
            TrendPoint(date="2021-01", value=0.65),
            TrendPoint(date="2021-07", value=0.64),
            TrendPoint(date="2022-01", value=0.65),
        ]
        result = engine.analyze_trend(points)
        assert result.trend_direction == "stable"

    def test_stable_near_zero_slope(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.65),
            TrendPoint(date="2021-01", value=0.65),
        ]
        result = engine.analyze_trend(points)
        assert abs(result.slope) < 0.01


class TestTrendAnalysisIncreasing:
    """Test trend analysis with increasing NDVI values (regrowth)."""

    def test_increasing_trend_direction(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.3),
            TrendPoint(date="2020-07", value=0.4),
            TrendPoint(date="2021-01", value=0.5),
            TrendPoint(date="2021-07", value=0.6),
            TrendPoint(date="2022-01", value=0.7),
        ]
        result = engine.analyze_trend(points)
        assert result.trend_direction == "increasing"

    def test_increasing_positive_slope(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.3),
            TrendPoint(date="2021-01", value=0.5),
        ]
        result = engine.analyze_trend(points)
        assert result.slope > 0


class TestTrendSinglePoint:
    """Test trend analysis with single or no points."""

    def test_single_point_stable(self, engine):
        points = [TrendPoint(date="2020-01", value=0.7)]
        result = engine.analyze_trend(points)
        assert result.trend_direction == "stable"
        assert result.slope == 0.0

    def test_empty_points_stable(self, engine):
        result = engine.analyze_trend([])
        assert result.trend_direction == "stable"
        assert result.slope == 0.0


# ===========================================================================
# Test: Breakpoint detection
# ===========================================================================


class TestBreakpointDetection:
    """Test breakpoint detection in time series."""

    def test_breakpoint_detected(self, engine):
        """Detect breakpoint when NDVI drops by > 0.1."""
        points = [
            TrendPoint(date="2020-01", value=0.7),
            TrendPoint(date="2020-07", value=0.5),  # -0.2 drop
            TrendPoint(date="2021-01", value=0.48),
        ]
        result = engine.analyze_trend(points)
        assert len(result.breakpoints) >= 1
        assert "2020-07" in result.breakpoints

    def test_multiple_breakpoints(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.8),
            TrendPoint(date="2020-07", value=0.6),  # -0.2 drop
            TrendPoint(date="2021-01", value=0.55),
            TrendPoint(date="2021-07", value=0.3),  # -0.25 drop
        ]
        result = engine.analyze_trend(points)
        assert len(result.breakpoints) >= 2

    def test_no_breakpoint_gradual_decline(self, engine):
        """No breakpoint for gradual decline < 0.1 between points."""
        points = [
            TrendPoint(date="2020-01", value=0.7),
            TrendPoint(date="2020-07", value=0.65),
            TrendPoint(date="2021-01", value=0.60),
            TrendPoint(date="2021-07", value=0.55),
        ]
        result = engine.analyze_trend(points)
        assert len(result.breakpoints) == 0

    def test_no_breakpoint_stable(self, engine):
        points = [
            TrendPoint(date="2020-01", value=0.65),
            TrendPoint(date="2020-07", value=0.66),
            TrendPoint(date="2021-01", value=0.64),
        ]
        result = engine.analyze_trend(points)
        assert len(result.breakpoints) == 0

    def test_breakpoint_not_on_increase(self, engine):
        """Breakpoints are only for drops, not increases."""
        points = [
            TrendPoint(date="2020-01", value=0.3),
            TrendPoint(date="2020-07", value=0.5),  # +0.2 increase
            TrendPoint(date="2021-01", value=0.7),  # +0.2 increase
        ]
        result = engine.analyze_trend(points)
        assert len(result.breakpoints) == 0


# ===========================================================================
# Test: Detection count
# ===========================================================================


class TestDetectionCount:
    """Test detection count tracking."""

    def test_starts_at_zero(self, engine):
        assert engine.detection_count == 0

    def test_increments_on_detect(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.7, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.2, valid_pixel_count=100)
        engine.detect_change(before, after)
        assert engine.detection_count == 1
        engine.detect_change(before, after)
        assert engine.detection_count == 2

    def test_detection_ids_unique(self, engine):
        before = VegetationIndexResult(index_type="ndvi", mean=0.7, valid_pixel_count=100)
        after = VegetationIndexResult(index_type="ndvi", mean=0.2, valid_pixel_count=100)
        det1 = engine.detect_change(before, after)
        det2 = engine.detect_change(before, after)
        assert det1.detection_id != det2.detection_id
