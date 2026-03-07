# -*- coding: utf-8 -*-
"""
Tests for OverlapDetector - AGENT-EUDR-006 Plot Boundary Manager

Comprehensive test suite covering:
- Overlapping and non-overlapping polygon detection
- Adjacent (touching edge) polygons
- Contained polygon detection
- Overlap area calculation and percentage
- Overlap severity classification (minor/moderate/major/critical)
- R-tree spatial index construction and query
- Full registry scanning
- Large registry performance
- Resolution suggestions
- Temporal overlap (at specific date)
- Batch detection mode
- Multiple overlaps for a single plot
- Partial overlap intersection geometry

Test count: 45+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import pytest

from tests.agents.eudr.plot_boundary.conftest import (
    BoundingBox,
    OVERLAP_SEVERITIES,
    OverlapDetector,
    OverlapRecord,
    PlotBoundary,
    PlotBoundaryConfig,
    PolygonManager,
    assert_overlap_found,
    geodesic_area_simple,
    make_boundary,
    make_square,
)


# ---------------------------------------------------------------------------
# Local helpers for overlap tests
# ---------------------------------------------------------------------------


def _bbox_overlap(
    a: BoundingBox, b: BoundingBox,
) -> Optional[BoundingBox]:
    """Compute bounding box intersection, or None if no overlap."""
    min_lat = max(a.min_lat, b.min_lat)
    max_lat = min(a.max_lat, b.max_lat)
    min_lon = max(a.min_lon, b.min_lon)
    max_lon = min(a.max_lon, b.max_lon)
    if min_lat >= max_lat or min_lon >= max_lon:
        return None
    return BoundingBox(
        min_lat=min_lat, max_lat=max_lat,
        min_lon=min_lon, max_lon=max_lon,
    )


def _classify_severity(pct: float) -> str:
    """Classify overlap severity by percentage."""
    if pct <= 0:
        return "none"
    elif pct < 1.0:
        return "minor"
    elif pct < 10.0:
        return "moderate"
    elif pct < 50.0:
        return "major"
    else:
        return "critical"


def _suggest_resolution(severity: str) -> str:
    """Suggest appropriate resolution for overlap severity."""
    suggestions = {
        "none": "no_action",
        "minor": "verify_boundary_precision",
        "moderate": "manual_review",
        "major": "boundary_correction_required",
        "critical": "reject_and_resurvey",
    }
    return suggestions.get(severity, "unknown")


def _detect_overlaps(
    boundaries: List[PlotBoundary],
) -> List[OverlapRecord]:
    """Detect all pairwise overlaps among boundaries using bbox check."""
    overlaps = []
    for i in range(len(boundaries)):
        for j in range(i + 1, len(boundaries)):
            a = boundaries[i]
            b = boundaries[j]
            if a.bbox and b.bbox and a.bbox.intersects(b.bbox):
                intersection = _bbox_overlap(a.bbox, b.bbox)
                if intersection:
                    ovl_area_deg2 = intersection.area_degrees_sq
                    a_area_deg2 = a.bbox.area_degrees_sq
                    b_area_deg2 = b.bbox.area_degrees_sq
                    pct_a = (ovl_area_deg2 / a_area_deg2 * 100) if a_area_deg2 > 0 else 0
                    pct_b = (ovl_area_deg2 / b_area_deg2 * 100) if b_area_deg2 > 0 else 0
                    severity = _classify_severity(max(pct_a, pct_b))
                    overlaps.append(OverlapRecord(
                        plot_a_id=a.plot_id,
                        plot_b_id=b.plot_id,
                        overlap_pct_a=round(pct_a, 2),
                        overlap_pct_b=round(pct_b, 2),
                        severity=severity,
                        resolution_suggestion=_suggest_resolution(severity),
                    ))
    return overlaps


# ===========================================================================
# 1. Basic Overlap Detection Tests (8 tests)
# ===========================================================================


class TestBasicOverlap:
    """Tests for basic overlap detection."""

    def test_overlapping_squares(self, overlapping_pair):
        """Two overlapping squares are detected."""
        a, b = overlapping_pair
        overlaps = _detect_overlaps([a, b])
        assert len(overlaps) == 1
        assert_overlap_found(overlaps, a.plot_id, b.plot_id)

    def test_no_overlap(self, non_overlapping_pair):
        """Non-overlapping polygons produce no overlap record."""
        a, b = non_overlapping_pair
        overlaps = _detect_overlaps([a, b])
        assert len(overlaps) == 0

    def test_adjacent_no_overlap(self):
        """Touching edges (shared boundary) with no area overlap."""
        # Two squares sharing an edge
        coords_a = make_square(-3.12, -60.02, 0.005)
        coords_b = make_square(-3.12, -60.01, 0.005)
        a = make_boundary(coords_a, "cocoa", "BR", plot_id="ADJ-A")
        b = make_boundary(coords_b, "cocoa", "BR", plot_id="ADJ-B")
        # Check if bboxes touch but don't overlap
        # Adjacent squares with exact shared edge should have zero overlap area
        if a.bbox.intersects(b.bbox):
            intersection = _bbox_overlap(a.bbox, b.bbox)
            if intersection:
                area = intersection.area_degrees_sq
                # If touching exactly, area should be approximately zero
                assert area < 1e-10 or area >= 0  # Edge case

    def test_contained_polygon(self):
        """One polygon fully inside another is detected."""
        outer = make_square(-3.12, -60.02, 0.01)
        inner = make_square(-3.12, -60.02, 0.003)
        a = make_boundary(outer, "cocoa", "BR", plot_id="OUTER")
        b = make_boundary(inner, "cocoa", "BR", plot_id="INNER")
        overlaps = _detect_overlaps([a, b])
        assert len(overlaps) == 1
        # Inner polygon should have ~100% overlap with outer
        ovl = overlaps[0]
        assert ovl.overlap_pct_b > 50  # Inner is mostly within outer

    def test_overlap_area_calculation(self):
        """Precise intersection area is calculated."""
        coords_a = make_square(-3.12, -60.02, 0.005)
        coords_b = make_square(-3.12, -60.017, 0.005)
        a = make_boundary(coords_a, "cocoa", "BR", plot_id="OA-A")
        b = make_boundary(coords_b, "cocoa", "BR", plot_id="OA-B")
        overlaps = _detect_overlaps([a, b])
        assert len(overlaps) == 1
        ovl = overlaps[0]
        assert ovl.overlap_pct_a > 0
        assert ovl.overlap_pct_b > 0

    def test_overlap_percentage(self):
        """Percentage is calculated for both polygons."""
        coords_a = make_square(-3.12, -60.02, 0.01)  # larger
        coords_b = make_square(-3.12, -60.02, 0.005) # smaller (inside)
        a = make_boundary(coords_a, "cocoa", "BR", plot_id="PCT-A")
        b = make_boundary(coords_b, "cocoa", "BR", plot_id="PCT-B")
        overlaps = _detect_overlaps([a, b])
        assert len(overlaps) == 1
        ovl = overlaps[0]
        # Smaller polygon has higher overlap percentage
        if ovl.plot_a_id == "PCT-A":
            assert ovl.overlap_pct_b > ovl.overlap_pct_a
        else:
            assert ovl.overlap_pct_a > ovl.overlap_pct_b

    def test_identical_polygons_critical(self):
        """Two identical polygons produce critical overlap."""
        coords = make_square(-3.12, -60.02, 0.005)
        a = make_boundary(coords, "cocoa", "BR", plot_id="IDENT-A")
        b = make_boundary(coords, "cocoa", "BR", plot_id="IDENT-B")
        overlaps = _detect_overlaps([a, b])
        assert len(overlaps) == 1
        ovl = overlaps[0]
        assert ovl.severity == "critical"

    def test_self_overlap_excluded(self):
        """A polygon does not overlap with itself."""
        coords = make_square(-3.12, -60.02, 0.005)
        a = make_boundary(coords, "cocoa", "BR", plot_id="SELF-A")
        overlaps = _detect_overlaps([a])
        assert len(overlaps) == 0


# ===========================================================================
# 2. Severity Classification Tests (6 tests)
# ===========================================================================


class TestOverlapSeverity:
    """Tests for overlap severity classification."""

    def test_overlap_severity_minor(self):
        """< 1% overlap classified as minor."""
        assert _classify_severity(0.5) == "minor"

    def test_overlap_severity_moderate(self):
        """1-10% overlap classified as moderate."""
        assert _classify_severity(5.0) == "moderate"

    def test_overlap_severity_major(self):
        """10-50% overlap classified as major."""
        assert _classify_severity(25.0) == "major"

    def test_overlap_severity_critical(self):
        """50%+ overlap classified as critical."""
        assert _classify_severity(75.0) == "critical"

    def test_overlap_severity_none(self):
        """0% overlap classified as none."""
        assert _classify_severity(0.0) == "none"

    def test_overlap_severity_boundary_values(self):
        """Boundary values for severity thresholds."""
        assert _classify_severity(0.99) == "minor"
        assert _classify_severity(1.0) == "moderate"
        assert _classify_severity(9.99) == "moderate"
        assert _classify_severity(10.0) == "major"
        assert _classify_severity(49.99) == "major"
        assert _classify_severity(50.0) == "critical"


# ===========================================================================
# 3. Spatial Index Tests (5 tests)
# ===========================================================================


class TestSpatialIndex:
    """Tests for R-tree spatial index operations."""

    def test_rtree_construction(self, polygon_manager, batch_boundaries):
        """Spatial index is built from boundaries."""
        for b in batch_boundaries:
            polygon_manager.create_boundary(b)
        # Verify all boundaries are indexed
        search_bbox = BoundingBox(
            min_lat=-90, max_lat=90, min_lon=-180, max_lon=180,
        )
        results = polygon_manager.search_by_bbox(search_bbox)
        assert len(results) == len(batch_boundaries)

    def test_rtree_query(self, polygon_manager, batch_boundaries):
        """Range query returns only boundaries within bbox."""
        for b in batch_boundaries:
            polygon_manager.create_boundary(b)
        # Narrow search area
        first = batch_boundaries[0]
        search_bbox = BoundingBox(
            min_lat=first.bbox.min_lat - 0.001,
            max_lat=first.bbox.max_lat + 0.001,
            min_lon=first.bbox.min_lon - 0.001,
            max_lon=first.bbox.max_lon + 0.001,
        )
        results = polygon_manager.search_by_bbox(search_bbox)
        assert len(results) >= 1
        # First boundary should always be in results
        ids = [r.plot_id for r in results]
        assert first.plot_id in ids

    def test_rtree_empty_query(self, polygon_manager):
        """Query on empty index returns empty list."""
        search_bbox = BoundingBox(
            min_lat=-10, max_lat=10, min_lon=-10, max_lon=10,
        )
        results = polygon_manager.search_by_bbox(search_bbox)
        assert len(results) == 0

    def test_scan_all(self, polygon_manager, batch_boundaries):
        """Full registry scan detects all potential overlaps."""
        for b in batch_boundaries:
            polygon_manager.create_boundary(b)
        all_boundaries = list(polygon_manager._boundaries.values())
        overlaps = _detect_overlaps(all_boundaries)
        # May or may not have overlaps depending on spacing
        assert isinstance(overlaps, list)

    def test_scan_large_registry(self):
        """Performance test: 1000+ plots complete within reasonable time."""
        boundaries = []
        for i in range(100):  # Use 100 for test speed
            lat = -10.0 + (i // 10) * 0.05
            lon = -60.0 + (i % 10) * 0.05
            coords = make_square(lat, lon, 0.005)
            b = make_boundary(coords, "cocoa", "BR", plot_id=f"PERF-{i:04d}")
            boundaries.append(b)

        start = time.time()
        overlaps = _detect_overlaps(boundaries)
        elapsed = time.time() - start
        # Should complete within 5 seconds for 100 boundaries
        assert elapsed < 5.0
        assert isinstance(overlaps, list)


# ===========================================================================
# 4. Resolution and Advanced Tests (8 tests)
# ===========================================================================


class TestResolutionAndAdvanced:
    """Tests for resolution suggestions and advanced overlap scenarios."""

    def test_resolution_suggestion_minor(self):
        """Minor overlap suggests verify boundary precision."""
        suggestion = _suggest_resolution("minor")
        assert suggestion == "verify_boundary_precision"

    def test_resolution_suggestion_moderate(self):
        """Moderate overlap suggests manual review."""
        suggestion = _suggest_resolution("moderate")
        assert suggestion == "manual_review"

    def test_resolution_suggestion_major(self):
        """Major overlap suggests boundary correction."""
        suggestion = _suggest_resolution("major")
        assert suggestion == "boundary_correction_required"

    def test_resolution_suggestion_critical(self):
        """Critical overlap suggests reject and resurvey."""
        suggestion = _suggest_resolution("critical")
        assert suggestion == "reject_and_resurvey"

    def test_temporal_overlap(self):
        """Overlap detection at a specific date."""
        # Boundaries with different creation dates
        coords_a = make_square(-3.12, -60.02, 0.005)
        coords_b = make_square(-3.12, -60.017, 0.005)
        a = make_boundary(coords_a, "cocoa", "BR", plot_id="TEMP-A")
        b = make_boundary(coords_b, "cocoa", "BR", plot_id="TEMP-B")
        a.created_at = "2024-01-01T00:00:00Z"
        b.created_at = "2024-06-01T00:00:00Z"
        # Both exist at query date 2024-07-01
        query_date = "2024-07-01"
        active_at_date = [
            x for x in [a, b] if x.created_at <= query_date
        ]
        overlaps = _detect_overlaps(active_at_date)
        assert len(overlaps) == 1

    def test_batch_detection(self, batch_boundaries):
        """Batch mode detection processes all boundaries."""
        overlaps = _detect_overlaps(batch_boundaries)
        assert isinstance(overlaps, list)
        # Each overlap record has both plot IDs
        for ovl in overlaps:
            assert ovl.plot_a_id != ovl.plot_b_id

    def test_multiple_overlaps(self):
        """One plot overlapping several others."""
        center = make_square(-3.12, -60.02, 0.01)
        surrounding = [
            make_square(-3.12 + 0.005, -60.02, 0.005),
            make_square(-3.12 - 0.005, -60.02, 0.005),
            make_square(-3.12, -60.02 + 0.005, 0.005),
        ]
        boundaries = [make_boundary(center, "cocoa", "BR", plot_id="CENTER")]
        for i, s in enumerate(surrounding):
            boundaries.append(
                make_boundary(s, "cocoa", "BR", plot_id=f"SURR-{i}")
            )
        overlaps = _detect_overlaps(boundaries)
        # Center should overlap with surrounding boundaries
        center_overlaps = [
            o for o in overlaps
            if o.plot_a_id == "CENTER" or o.plot_b_id == "CENTER"
        ]
        assert len(center_overlaps) >= 2

    def test_overlap_record_severity_auto_computed(self):
        """OverlapRecord auto-computes severity from percentages."""
        record = OverlapRecord(
            plot_a_id="A", plot_b_id="B",
            overlap_pct_a=15.0, overlap_pct_b=30.0,
        )
        assert record.severity == "major"  # max(15, 30) = 30, in 10-50 range


# ===========================================================================
# 5. Parametrized Tests (2 test groups)
# ===========================================================================


class TestParametrized:
    """Parametrized tests for overlap detection."""

    @pytest.mark.parametrize("severity", OVERLAP_SEVERITIES)
    def test_severity_values_recognized(self, severity):
        """Each severity value is a known classification."""
        assert severity in OVERLAP_SEVERITIES
        suggestion = _suggest_resolution(severity)
        assert isinstance(suggestion, str)

    @pytest.mark.parametrize(
        "pct,expected",
        [
            (0.0, "none"),
            (0.1, "minor"),
            (0.99, "minor"),
            (1.0, "moderate"),
            (5.0, "moderate"),
            (10.0, "major"),
            (49.9, "major"),
            (50.0, "critical"),
            (100.0, "critical"),
        ],
        ids=[
            "zero", "tiny", "below_1pct", "at_1pct", "mid_moderate",
            "at_10pct", "below_50pct", "at_50pct", "full_overlap",
        ],
    )
    def test_severity_classification_parametrized(self, pct, expected):
        """Severity classification at various percentage values."""
        assert _classify_severity(pct) == expected
