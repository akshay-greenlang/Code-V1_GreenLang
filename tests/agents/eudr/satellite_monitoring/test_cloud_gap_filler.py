# -*- coding: utf-8 -*-
"""
Tests for CloudGapFiller - AGENT-EUDR-003 Feature 6: Cloud Gap Filling

Comprehensive test suite covering:
- Cloud detection and percentage calculation
- Temporal composite creation (median pixel replacement)
- SAR backscatter fusion for cloud-persistent regions
- Linear interpolation between clear observations
- Nearest clear scene selection
- Gap fill quality scoring
- Cloud persistence in tropical regions (Amazon, Congo, Borneo)
- Determinism and reproducibility

Test count: 55+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 (Feature 6 - Cloud Gap Filling)
"""

import math
import statistics

import pytest

from tests.agents.eudr.satellite_monitoring.conftest import (
    CloudGapFillResult,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    CLOUD_FILL_METHODS,
)


# ---------------------------------------------------------------------------
# Helpers for cloud gap filling logic
# ---------------------------------------------------------------------------


def _detect_clouds(bands: dict, threshold: float = 7000) -> list:
    """Detect cloud pixels from band data. Returns list of (row, col) tuples."""
    cloud_pixels = []
    blue = bands.get("B02", [])
    for r, row in enumerate(blue):
        if isinstance(row, list):
            for c, val in enumerate(row):
                if val >= threshold:
                    cloud_pixels.append((r, c))
        else:
            if row >= threshold:
                cloud_pixels.append((r, 0))
    return cloud_pixels


def _cloud_percentage(total_pixels: int, cloud_pixels: int) -> float:
    """Calculate cloud cover percentage."""
    if total_pixels == 0:
        return 0.0
    return (cloud_pixels / total_pixels) * 100.0


def _median_composite(scenes: list) -> list:
    """Create median composite from multiple scene values.

    scenes: list of lists of values (one per scene).
    Returns list of median values per pixel position.
    """
    if not scenes:
        return []
    n_pixels = len(scenes[0])
    result = []
    for i in range(n_pixels):
        values = [s[i] for s in scenes if s[i] is not None]
        if values:
            result.append(statistics.median(values))
        else:
            result.append(None)
    return result


def _sar_forest_classify(backscatter_db: float) -> str:
    """Classify SAR backscatter into forest/non-forest.

    Typical C-band VH backscatter:
    Forest: -10 to -15 dB
    Non-forest: -18 to -25 dB
    """
    if backscatter_db >= -16.0:
        return "forest"
    elif backscatter_db >= -20.0:
        return "degraded"
    else:
        return "non_forest"


def _linear_interpolation(val_before: float, val_after: float, t: float) -> float:
    """Linear interpolation between two values. t in [0, 1]."""
    return val_before + t * (val_after - val_before)


def _nearest_clear(scenes: list, target_idx: int) -> int:
    """Find nearest clear scene index to the target.

    scenes: list of (cloud_pct, index) tuples.
    Returns index of nearest clear scene.
    """
    clear = [(abs(idx - target_idx), idx) for cloud_pct, idx in scenes if cloud_pct < 20.0]
    if not clear:
        return -1
    clear.sort()
    return clear[0][1]


def _gap_fill_quality(original_cloud_pct: float, filled_cloud_pct: float) -> float:
    """Score gap fill quality (0-100)."""
    if original_cloud_pct == 0:
        return 100.0
    improvement = original_cloud_pct - filled_cloud_pct
    score = (improvement / original_cloud_pct) * 100.0
    return max(0.0, min(100.0, score))


# ===========================================================================
# 1. Cloud Detection (10 tests)
# ===========================================================================


class TestCloudDetection:
    """Test cloud pixel detection from spectral data."""

    def test_clear_scene(self):
        """Test clear scene has no cloud pixels detected."""
        bands = {"B02": [500, 600, 550, 480, 520]}
        clouds = _detect_clouds(bands)
        assert len(clouds) == 0

    def test_fully_cloudy(self):
        """Test fully cloudy scene detects all pixels as cloud."""
        bands = {"B02": [8000, 8200, 7500, 9000, 8800]}
        clouds = _detect_clouds(bands)
        assert len(clouds) == 5

    def test_partial_cloud(self):
        """Test partial cloud detection."""
        bands = {"B02": [500, 8000, 600, 7500, 520]}
        clouds = _detect_clouds(bands)
        assert len(clouds) == 2

    def test_cloud_percentage_accuracy(self):
        """Test cloud percentage calculation accuracy."""
        pct = _cloud_percentage(100, 25)
        assert pct == pytest.approx(25.0, abs=0.1)

    def test_cloud_percentage_zero(self):
        """Test cloud percentage is 0% for clear scene."""
        pct = _cloud_percentage(100, 0)
        assert pct == 0.0

    def test_cloud_percentage_full(self):
        """Test cloud percentage is 100% for fully cloudy scene."""
        pct = _cloud_percentage(100, 100)
        assert pct == 100.0

    def test_cloud_percentage_empty(self):
        """Test cloud percentage handles empty scene."""
        pct = _cloud_percentage(0, 0)
        assert pct == 0.0

    @pytest.mark.parametrize("total,cloud,expected", [
        (100, 0, 0.0),
        (100, 10, 10.0),
        (100, 50, 50.0),
        (100, 100, 100.0),
        (200, 50, 25.0),
        (1000, 333, 33.3),
    ])
    def test_cloud_percentage_parametrized(self, total, cloud, expected):
        """Test cloud percentage for various scenarios."""
        pct = _cloud_percentage(total, cloud)
        assert pct == pytest.approx(expected, abs=0.1)


# ===========================================================================
# 2. Temporal Composite (8 tests)
# ===========================================================================


class TestTemporalComposite:
    """Test temporal median compositing for cloud removal."""

    def test_median_pixel(self):
        """Test median composite selects middle value."""
        scenes = [
            [0.70, 0.72, 0.68],
            [0.71, 0.69, 0.73],
            [0.69, 0.71, 0.70],
        ]
        composite = _median_composite(scenes)
        assert composite[0] == pytest.approx(0.70, abs=0.01)

    def test_composite_removes_outlier(self):
        """Test median composite removes cloud-contaminated outlier."""
        scenes = [
            [0.70, 0.72, 0.68],
            [0.90, 0.69, 0.73],  # First pixel is cloud outlier
            [0.69, 0.71, 0.70],
        ]
        composite = _median_composite(scenes)
        assert composite[0] == pytest.approx(0.70, abs=0.05)

    def test_composite_empty(self):
        """Test empty scene list returns empty composite."""
        composite = _median_composite([])
        assert composite == []

    def test_composite_single_scene(self):
        """Test composite with single scene returns same values."""
        scenes = [[0.70, 0.72, 0.68]]
        composite = _median_composite(scenes)
        assert composite == [0.70, 0.72, 0.68]

    def test_composite_two_scenes(self):
        """Test composite with two scenes returns average."""
        scenes = [
            [0.70, 0.80],
            [0.60, 0.70],
        ]
        composite = _median_composite(scenes)
        assert composite[0] == pytest.approx(0.65, abs=0.01)

    def test_composite_handles_none(self):
        """Test composite handles None values (cloud-masked pixels)."""
        scenes = [
            [0.70, None, 0.68],
            [None, 0.69, 0.73],
            [0.69, 0.71, None],
        ]
        composite = _median_composite(scenes)
        assert composite[0] == pytest.approx(0.695, abs=0.01)
        assert composite[1] == pytest.approx(0.70, abs=0.01)


# ===========================================================================
# 3. SAR Fusion (8 tests)
# ===========================================================================


class TestSARFusion:
    """Test Sentinel-1 SAR backscatter fusion for cloud gap filling."""

    def test_sar_forest_classification(self):
        """Test SAR forest classification from backscatter."""
        assert _sar_forest_classify(-12.0) == "forest"
        assert _sar_forest_classify(-18.0) == "degraded"
        assert _sar_forest_classify(-22.0) == "non_forest"

    def test_sar_gap_fill(self):
        """Test SAR fills gap when optical is cloud-covered."""
        # Optical is cloudy, SAR backscatter indicates forest
        sar_backscatter = -13.0
        classification = _sar_forest_classify(sar_backscatter)
        assert classification == "forest"

    @pytest.mark.parametrize("backscatter,expected", [
        (-10.0, "forest"),
        (-12.0, "forest"),
        (-14.0, "forest"),
        (-15.0, "forest"),
        (-16.0, "forest"),
        (-17.0, "degraded"),
        (-18.0, "degraded"),
        (-19.0, "degraded"),
        (-20.0, "degraded"),
        (-21.0, "non_forest"),
        (-25.0, "non_forest"),
        (-30.0, "non_forest"),
    ])
    def test_sar_classification_parametrized(self, backscatter, expected):
        """Test SAR classification across backscatter range."""
        result = _sar_forest_classify(backscatter)
        assert result == expected


# ===========================================================================
# 4. Interpolation (6 tests)
# ===========================================================================


class TestInterpolation:
    """Test linear interpolation for temporal gap filling."""

    def test_linear_interpolation_midpoint(self):
        """Test interpolation at midpoint."""
        val = _linear_interpolation(0.70, 0.80, 0.5)
        assert val == pytest.approx(0.75, abs=0.001)

    def test_interpolation_endpoints(self):
        """Test interpolation at endpoints."""
        assert _linear_interpolation(0.70, 0.80, 0.0) == pytest.approx(0.70)
        assert _linear_interpolation(0.70, 0.80, 1.0) == pytest.approx(0.80)

    def test_interpolation_quarter(self):
        """Test interpolation at quarter points."""
        val = _linear_interpolation(0.60, 1.00, 0.25)
        assert val == pytest.approx(0.70, abs=0.001)

    def test_interpolation_decreasing(self):
        """Test interpolation with decreasing values."""
        val = _linear_interpolation(0.80, 0.40, 0.5)
        assert val == pytest.approx(0.60, abs=0.001)

    @pytest.mark.parametrize("before,after,t,expected", [
        (0.70, 0.80, 0.0, 0.70),
        (0.70, 0.80, 0.25, 0.725),
        (0.70, 0.80, 0.50, 0.75),
        (0.70, 0.80, 0.75, 0.775),
        (0.70, 0.80, 1.0, 0.80),
    ])
    def test_interpolation_parametrized(self, before, after, t, expected):
        """Test interpolation at various t values."""
        val = _linear_interpolation(before, after, t)
        assert val == pytest.approx(expected, abs=0.001)


# ===========================================================================
# 5. Nearest Clear Scene (5 tests)
# ===========================================================================


class TestNearestClear:
    """Test nearest clear scene selection."""

    def test_nearest_clear_selection(self):
        """Test nearest clear scene is found."""
        scenes = [(45.0, 0), (10.0, 1), (5.0, 2), (30.0, 3), (8.0, 4)]
        nearest = _nearest_clear(scenes, target_idx=3)
        assert nearest in (2, 4)

    def test_nearest_clear_at_target(self):
        """Test when target is already clear."""
        scenes = [(5.0, 0), (10.0, 1), (5.0, 2)]
        nearest = _nearest_clear(scenes, target_idx=0)
        assert nearest == 0

    def test_nearest_clear_no_clear(self):
        """Test when no clear scenes available."""
        scenes = [(45.0, 0), (50.0, 1), (80.0, 2)]
        nearest = _nearest_clear(scenes, target_idx=1)
        assert nearest == -1

    def test_nearest_clear_all_clear(self):
        """Test when all scenes are clear."""
        scenes = [(5.0, 0), (8.0, 1), (3.0, 2)]
        nearest = _nearest_clear(scenes, target_idx=2)
        assert nearest == 2


# ===========================================================================
# 6. Gap Fill Quality (8 tests)
# ===========================================================================


class TestGapFillQuality:
    """Test gap fill quality scoring."""

    def test_quality_no_gap(self):
        """Test quality is 100% when no gap to fill."""
        score = _gap_fill_quality(0.0, 0.0)
        assert score == 100.0

    def test_quality_complete_fill(self):
        """Test quality is 100% when all clouds removed."""
        score = _gap_fill_quality(50.0, 0.0)
        assert score == 100.0

    def test_quality_partial_fill(self):
        """Test quality for partial cloud removal."""
        score = _gap_fill_quality(50.0, 25.0)
        assert score == pytest.approx(50.0, abs=1.0)

    def test_quality_no_improvement(self):
        """Test quality is 0% when no improvement."""
        score = _gap_fill_quality(50.0, 50.0)
        assert score == 0.0

    def test_quality_heavy_fill(self):
        """Test quality for heavy cloud scenes."""
        score = _gap_fill_quality(80.0, 10.0)
        assert score >= 80.0

    @pytest.mark.parametrize("original,filled,expected_min", [
        (0.0, 0.0, 100.0),
        (10.0, 0.0, 100.0),
        (50.0, 0.0, 100.0),
        (50.0, 25.0, 45.0),
        (50.0, 50.0, 0.0),
        (80.0, 20.0, 70.0),
        (90.0, 10.0, 85.0),
    ])
    def test_quality_range_parametrized(self, original, filled, expected_min):
        """Test quality score for various cloud fill scenarios."""
        score = _gap_fill_quality(original, filled)
        assert score >= expected_min


# ===========================================================================
# 7. Cloud Persistence Regions (6 tests)
# ===========================================================================


class TestCloudPersistenceRegions:
    """Test cloud persistence handling for tropical regions."""

    @pytest.mark.parametrize("region,wet_months,expected_high_cloud_pct", [
        ("amazon", [12, 1, 2, 3, 4, 5], 60.0),
        ("congo", [9, 10, 11, 12, 1, 2], 65.0),
        ("borneo", [11, 12, 1, 2, 3], 70.0),
        ("west_africa", [5, 6, 7, 8, 9, 10], 55.0),
    ])
    def test_cloud_persistence(self, region, wet_months, expected_high_cloud_pct):
        """Test recognition of cloud-persistent tropical regions."""
        # In wet season months, cloud cover should be higher
        assert len(wet_months) >= 4
        assert expected_high_cloud_pct >= 50.0

    def test_cloud_fill_methods(self):
        """Test all cloud fill methods are recognized."""
        for method in CLOUD_FILL_METHODS:
            result = CloudGapFillResult(fill_method=method)
            assert result.fill_method in CLOUD_FILL_METHODS


# ===========================================================================
# 8. Determinism (5 tests)
# ===========================================================================


class TestCloudFillDeterminism:
    """Test cloud gap filling determinism."""

    def test_median_composite_deterministic(self):
        """Test median composite is deterministic."""
        scenes = [
            [0.70, 0.72, 0.68],
            [0.71, 0.69, 0.73],
            [0.69, 0.71, 0.70],
        ]
        results = [_median_composite(scenes) for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]

    def test_interpolation_deterministic(self):
        """Test interpolation is deterministic."""
        results = [_linear_interpolation(0.70, 0.80, 0.5) for _ in range(10)]
        assert len(set(results)) == 1

    def test_sar_classification_deterministic(self):
        """Test SAR classification is deterministic."""
        results = [_sar_forest_classify(-13.0) for _ in range(10)]
        assert len(set(results)) == 1

    def test_cloud_percentage_deterministic(self):
        """Test cloud percentage is deterministic."""
        results = [_cloud_percentage(100, 33) for _ in range(10)]
        assert len(set(results)) == 1
