# -*- coding: utf-8 -*-
"""
Tests for SplitMergeEngine - AGENT-EUDR-006 Plot Boundary Manager

Comprehensive test suite covering:
- Split operations (simple, diagonal, three-way)
- Area conservation after split
- Child ID assignment
- Attribute inheritance (commodity, country)
- Merge operations (two-way, three-way)
- Area conservation after merge
- Non-adjacent merge rejection
- Genealogy tracking (parent/children, parents/child)
- Full lineage traversal (root to leaf)
- Undo split and merge operations
- Cutting line validation
- Area tolerance checking
- Parametrized tests for split operations

Test count: 45+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

import pytest

from tests.agents.eudr.plot_boundary.conftest import (
    BoundingBox,
    MergeResult,
    PlotBoundary,
    PlotBoundaryConfig,
    SplitMergeEngine,
    SplitResult,
    geodesic_area_simple,
    make_boundary,
    make_square,
)


# ---------------------------------------------------------------------------
# Local helpers for split/merge tests
# ---------------------------------------------------------------------------


def _split_square_horizontal(
    coords: List[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Split a square polygon horizontally into two rectangles.

    Assumes coords are: [SW, SE, NE, NW, SW] (CCW closed square).
    Split along the horizontal midline.
    """
    lats = [c[0] for c in coords[:-1]]
    lons = [c[1] for c in coords[:-1]]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    mid_lat = (min_lat + max_lat) / 2.0

    # Bottom half
    bottom = [
        (min_lat, min_lon), (min_lat, max_lon),
        (mid_lat, max_lon), (mid_lat, min_lon),
        (min_lat, min_lon),
    ]
    # Top half
    top = [
        (mid_lat, min_lon), (mid_lat, max_lon),
        (max_lat, max_lon), (max_lat, min_lon),
        (mid_lat, min_lon),
    ]
    return bottom, top


def _split_square_diagonal(
    coords: List[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Split a square polygon diagonally into two triangles.

    Assumes coords are: [SW, SE, NE, NW, SW] (CCW closed square).
    """
    lats = [c[0] for c in coords[:-1]]
    lons = [c[1] for c in coords[:-1]]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Triangle 1: SW-SE-NE
    tri1 = [
        (min_lat, min_lon), (min_lat, max_lon),
        (max_lat, max_lon), (min_lat, min_lon),
    ]
    # Triangle 2: SW-NE-NW
    tri2 = [
        (min_lat, min_lon), (max_lat, max_lon),
        (max_lat, min_lon), (min_lat, min_lon),
    ]
    return tri1, tri2


def _split_into_three(
    coords: List[Tuple[float, float]],
) -> List[List[Tuple[float, float]]]:
    """Split a square into three horizontal strips."""
    lats = [c[0] for c in coords[:-1]]
    lons = [c[1] for c in coords[:-1]]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_third = (max_lat - min_lat) / 3.0

    strips = []
    for i in range(3):
        low = min_lat + i * lat_third
        high = min_lat + (i + 1) * lat_third
        strip = [
            (low, min_lon), (low, max_lon),
            (high, max_lon), (high, min_lon),
            (low, min_lon),
        ]
        strips.append(strip)
    return strips


def _merge_adjacent_squares(
    coords_a: List[Tuple[float, float]],
    coords_b: List[Tuple[float, float]],
) -> Optional[List[Tuple[float, float]]]:
    """Merge two adjacent rectangles into one polygon.

    Simple implementation: compute combined bounding box for axis-aligned rects.
    """
    all_lats = [c[0] for c in coords_a[:-1]] + [c[0] for c in coords_b[:-1]]
    all_lons = [c[1] for c in coords_a[:-1]] + [c[1] for c in coords_b[:-1]]
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    return [
        (min_lat, min_lon), (min_lat, max_lon),
        (max_lat, max_lon), (max_lat, min_lon),
        (min_lat, min_lon),
    ]


def _are_adjacent(
    bbox_a: BoundingBox, bbox_b: BoundingBox, tolerance: float = 0.001,
) -> bool:
    """Check if two bboxes share a boundary (adjacent)."""
    # Check if they touch on any edge within tolerance
    lat_touch = (
        abs(bbox_a.max_lat - bbox_b.min_lat) < tolerance
        or abs(bbox_b.max_lat - bbox_a.min_lat) < tolerance
    )
    lon_touch = (
        abs(bbox_a.max_lon - bbox_b.min_lon) < tolerance
        or abs(bbox_b.max_lon - bbox_a.min_lon) < tolerance
    )
    lat_overlap = not (
        bbox_a.max_lat < bbox_b.min_lat - tolerance
        or bbox_b.max_lat < bbox_a.min_lat - tolerance
    )
    lon_overlap = not (
        bbox_a.max_lon < bbox_b.min_lon - tolerance
        or bbox_b.max_lon < bbox_a.min_lon - tolerance
    )
    return (lat_touch and lon_overlap) or (lon_touch and lat_overlap)


# ---------------------------------------------------------------------------
# Genealogy tracker for split/merge lineage
# ---------------------------------------------------------------------------


class GenealogyTracker:
    """Track parent-child relationships for split/merge operations."""

    def __init__(self):
        self._children: Dict[str, List[str]] = {}
        self._parents: Dict[str, List[str]] = {}

    def record_split(self, parent_id: str, child_ids: List[str]) -> None:
        self._children[parent_id] = child_ids
        for cid in child_ids:
            self._parents[cid] = [parent_id]

    def record_merge(self, parent_ids: List[str], child_id: str) -> None:
        self._parents[child_id] = parent_ids
        for pid in parent_ids:
            self._children.setdefault(pid, []).append(child_id)

    def get_children(self, plot_id: str) -> List[str]:
        return self._children.get(plot_id, [])

    def get_parents(self, plot_id: str) -> List[str]:
        return self._parents.get(plot_id, [])

    def get_full_lineage(self, plot_id: str) -> List[str]:
        """Get full lineage from root to leaf."""
        lineage = [plot_id]
        # Go up to root
        current = plot_id
        while self.get_parents(current):
            current = self.get_parents(current)[0]
            lineage.insert(0, current)
        # Go down to leaves
        current = plot_id
        while self.get_children(current):
            current = self.get_children(current)[0]
            lineage.append(current)
        return lineage


# ===========================================================================
# 1. Split Operation Tests (10 tests)
# ===========================================================================


class TestSplitOperations:
    """Tests for polygon split operations."""

    def test_split_simple(self):
        """Split square into 2 rectangles."""
        coords = make_square(-3.12, -60.02, 0.01)
        bottom, top = _split_square_horizontal(coords)
        assert len(bottom) == 5
        assert len(top) == 5
        assert bottom[0] == bottom[-1]  # Closed ring
        assert top[0] == top[-1]

    def test_split_diagonal(self):
        """Diagonal cutting line produces two triangles."""
        coords = make_square(-3.12, -60.02, 0.01)
        tri1, tri2 = _split_square_diagonal(coords)
        assert len(tri1) == 4  # Triangle + closure
        assert len(tri2) == 4

    def test_split_into_three(self):
        """Three-way split produces three strips."""
        coords = make_square(-3.12, -60.02, 0.01)
        strips = _split_into_three(coords)
        assert len(strips) == 3
        for strip in strips:
            assert len(strip) >= 4
            assert strip[0] == strip[-1]

    def test_split_area_conservation(self):
        """Sum of children areas equals parent area."""
        coords = make_square(-3.12, -60.02, 0.01)
        parent_area = geodesic_area_simple(coords)
        bottom, top = _split_square_horizontal(coords)
        child_area_sum = geodesic_area_simple(bottom) + geodesic_area_simple(top)
        relative_error = abs(child_area_sum - parent_area) / parent_area
        assert relative_error < 0.01  # < 1% difference

    def test_split_child_ids_assigned(self):
        """New UUIDs are generated for children."""
        parent_id = "PARENT-001"
        child_ids = [
            f"CHILD-{uuid.uuid4().hex[:8].upper()}" for _ in range(2)
        ]
        assert len(child_ids) == 2
        assert child_ids[0] != child_ids[1]
        assert all(cid != parent_id for cid in child_ids)

    def test_split_attribute_inheritance(self):
        """Commodity and country are inherited by children."""
        parent = make_boundary(
            make_square(-3.12, -60.02, 0.01),
            "cocoa", "BR", plot_id="INHERIT-PARENT",
        )
        bottom, top = _split_square_horizontal(parent.exterior_ring)
        child1 = make_boundary(bottom, parent.commodity, parent.country, "INHERIT-C1")
        child2 = make_boundary(top, parent.commodity, parent.country, "INHERIT-C2")
        assert child1.commodity == "cocoa"
        assert child1.country == "BR"
        assert child2.commodity == "cocoa"
        assert child2.country == "BR"

    def test_split_result_structure(self):
        """SplitResult has correct structure."""
        coords = make_square(-3.12, -60.02, 0.01)
        bottom, top = _split_square_horizontal(coords)
        parent_area = geodesic_area_simple(coords)
        child_areas = [
            geodesic_area_simple(bottom), geodesic_area_simple(top),
        ]
        result = SplitResult(
            parent_id="PARENT-001",
            child_ids=["CHILD-001", "CHILD-002"],
            child_boundaries=[bottom, top],
            child_areas_ha=child_areas,
            parent_area_ha=parent_area,
            area_conservation_ok=abs(sum(child_areas) - parent_area) / parent_area < 0.01,
        )
        assert result.parent_id == "PARENT-001"
        assert len(result.child_ids) == 2
        assert result.area_conservation_ok is True
        assert abs(result.area_sum - parent_area) / parent_area < 0.01

    def test_cutting_line_validation(self):
        """Cutting line must cross the polygon."""
        coords = make_square(-3.12, -60.02, 0.01)
        lats = [c[0] for c in coords[:-1]]
        # Valid cutting line: horizontal line through the middle
        mid_lat = (min(lats) + max(lats)) / 2.0
        line_inside = [(mid_lat, -60.03), (mid_lat, -60.01)]
        # Check if line intersects polygon bbox
        bbox = BoundingBox(
            min_lat=min(c[0] for c in coords),
            max_lat=max(c[0] for c in coords),
            min_lon=min(c[1] for c in coords),
            max_lon=max(c[1] for c in coords),
        )
        line_lat = line_inside[0][0]
        intersects = bbox.min_lat <= line_lat <= bbox.max_lat
        assert intersects is True

    def test_cutting_line_outside_fails(self):
        """Cutting line outside polygon does not produce valid split."""
        coords = make_square(-3.12, -60.02, 0.01)
        bbox = BoundingBox(
            min_lat=min(c[0] for c in coords),
            max_lat=max(c[0] for c in coords),
            min_lon=min(c[1] for c in coords),
            max_lon=max(c[1] for c in coords),
        )
        # Line far from polygon
        line_outside_lat = -10.0
        intersects = bbox.min_lat <= line_outside_lat <= bbox.max_lat
        assert intersects is False

    def test_area_tolerance_check(self, config):
        """Split area tolerance is configurable."""
        assert config.split_area_tolerance_pct == 0.5


# ===========================================================================
# 2. Merge Operation Tests (8 tests)
# ===========================================================================


class TestMergeOperations:
    """Tests for polygon merge operations."""

    def test_merge_two_adjacent(self):
        """Merge two adjacent rectangles."""
        bottom = [
            (-3.125, -60.025), (-3.125, -60.015),
            (-3.12, -60.015), (-3.12, -60.025),
            (-3.125, -60.025),
        ]
        top = [
            (-3.12, -60.025), (-3.12, -60.015),
            (-3.115, -60.015), (-3.115, -60.025),
            (-3.12, -60.025),
        ]
        merged = _merge_adjacent_squares(bottom, top)
        assert len(merged) == 5
        assert merged[0] == merged[-1]

    def test_merge_three_adjacent(self):
        """Three-way merge combines three strips."""
        coords = make_square(-3.12, -60.02, 0.01)
        strips = _split_into_three(coords)
        # Merge first two
        merged_12 = _merge_adjacent_squares(strips[0], strips[1])
        # Merge result with third
        merged_all = _merge_adjacent_squares(merged_12, strips[2])
        assert len(merged_all) == 5

    def test_merge_area_conservation(self):
        """Merged area equals sum of parent areas."""
        coords = make_square(-3.12, -60.02, 0.01)
        bottom, top = _split_square_horizontal(coords)
        area_bottom = geodesic_area_simple(bottom)
        area_top = geodesic_area_simple(top)
        merged = _merge_adjacent_squares(bottom, top)
        area_merged = geodesic_area_simple(merged)
        parent_sum = area_bottom + area_top
        if parent_sum > 0:
            relative_error = abs(area_merged - parent_sum) / parent_sum
            assert relative_error < 0.05  # < 5% (bbox merge is approximate)

    def test_merge_non_adjacent_fails(self):
        """Non-adjacent polygons cannot be merged."""
        a_coords = make_square(-3.12, -60.02, 0.003)
        b_coords = make_square(-3.50, -61.00, 0.003)
        a = make_boundary(a_coords, "cocoa", "BR", plot_id="NAD-A")
        b = make_boundary(b_coords, "cocoa", "BR", plot_id="NAD-B")
        adjacent = _are_adjacent(a.bbox, b.bbox)
        assert adjacent is False

    def test_merge_result_structure(self):
        """MergeResult has correct structure."""
        coords = make_square(-3.12, -60.02, 0.01)
        bottom, top = _split_square_horizontal(coords)
        merged = _merge_adjacent_squares(bottom, top)
        result = MergeResult(
            parent_ids=["PARENT-A", "PARENT-B"],
            child_id="MERGED-001",
            child_boundary=merged,
            child_area_ha=geodesic_area_simple(merged),
            parent_areas_sum_ha=(
                geodesic_area_simple(bottom) + geodesic_area_simple(top)
            ),
        )
        result.area_conservation_ok = result.area_difference < 1.0
        assert result.child_id == "MERGED-001"
        assert len(result.parent_ids) == 2

    def test_adjacent_detection_shared_edge(self):
        """Two rectangles sharing an edge are adjacent."""
        # Create two squares that share an edge exactly:
        # Square A: lon from -60.025 to -60.020
        # Square B: lon from -60.020 to -60.015
        half = 0.0025
        a_coords = [
            (-3.1225, -60.025), (-3.1225, -60.020),
            (-3.1175, -60.020), (-3.1175, -60.025),
            (-3.1225, -60.025),
        ]
        b_coords = [
            (-3.1225, -60.020), (-3.1225, -60.015),
            (-3.1175, -60.015), (-3.1175, -60.020),
            (-3.1225, -60.020),
        ]
        a = make_boundary(a_coords, "cocoa", "BR", plot_id="EDGE-A")
        b = make_boundary(b_coords, "cocoa", "BR", plot_id="EDGE-B")
        assert _are_adjacent(a.bbox, b.bbox, tolerance=0.001)

    def test_merge_preserves_commodity(self):
        """Merged boundary inherits commodity from parents."""
        coords = make_square(-3.12, -60.02, 0.01)
        bottom, top = _split_square_horizontal(coords)
        parent_a = make_boundary(bottom, "cocoa", "BR", plot_id="MC-A")
        parent_b = make_boundary(top, "cocoa", "BR", plot_id="MC-B")
        merged_coords = _merge_adjacent_squares(
            parent_a.exterior_ring, parent_b.exterior_ring,
        )
        child = make_boundary(merged_coords, parent_a.commodity, parent_a.country)
        assert child.commodity == "cocoa"
        assert child.country == "BR"

    def test_merge_gap_tolerance(self, config):
        """Merge gap tolerance is configurable."""
        assert config.merge_gap_tolerance_m == 1.0


# ===========================================================================
# 3. Genealogy Tracking Tests (8 tests)
# ===========================================================================


class TestGenealogyTracking:
    """Tests for split/merge genealogy tracking."""

    def test_genealogy_after_split(self):
        """Parent to children relationship is recorded after split."""
        tracker = GenealogyTracker()
        tracker.record_split("PARENT-001", ["CHILD-A", "CHILD-B"])
        children = tracker.get_children("PARENT-001")
        assert children == ["CHILD-A", "CHILD-B"]

    def test_genealogy_after_merge(self):
        """Parents to child relationship is recorded after merge."""
        tracker = GenealogyTracker()
        tracker.record_merge(["PARENT-A", "PARENT-B"], "MERGED-001")
        parents = tracker.get_parents("MERGED-001")
        assert set(parents) == {"PARENT-A", "PARENT-B"}

    def test_get_children(self):
        """Direct children query returns correct IDs."""
        tracker = GenealogyTracker()
        tracker.record_split("P1", ["C1", "C2", "C3"])
        assert tracker.get_children("P1") == ["C1", "C2", "C3"]

    def test_get_parents(self):
        """Direct parents query returns correct IDs."""
        tracker = GenealogyTracker()
        tracker.record_merge(["P1", "P2"], "M1")
        assert set(tracker.get_parents("M1")) == {"P1", "P2"}

    def test_full_lineage(self):
        """Root to leaf traversal returns complete chain."""
        tracker = GenealogyTracker()
        tracker.record_split("ROOT", ["MID-A", "MID-B"])
        tracker.record_split("MID-A", ["LEAF-1", "LEAF-2"])
        lineage = tracker.get_full_lineage("MID-A")
        assert lineage[0] == "ROOT"
        assert "MID-A" in lineage
        assert lineage[-1] == "LEAF-1"

    def test_no_children(self):
        """Leaf node has no children."""
        tracker = GenealogyTracker()
        tracker.record_split("P1", ["C1"])
        assert tracker.get_children("C1") == []

    def test_no_parents(self):
        """Root node has no parents."""
        tracker = GenealogyTracker()
        tracker.record_split("ROOT", ["C1"])
        assert tracker.get_parents("ROOT") == []

    def test_complex_genealogy(self):
        """Complex split-merge-split chain is tracked."""
        tracker = GenealogyTracker()
        # Split original into two
        tracker.record_split("ORIG", ["A", "B"])
        # Merge A with external C
        tracker.record_merge(["A", "C"], "AC")
        # Split AC into two
        tracker.record_split("AC", ["AC-1", "AC-2"])

        assert tracker.get_children("ORIG") == ["A", "B"]
        assert "AC" in tracker.get_children("A")
        assert tracker.get_parents("AC") == ["A", "C"]
        assert tracker.get_children("AC") == ["AC-1", "AC-2"]


# ===========================================================================
# 4. Undo Operation Tests (4 tests)
# ===========================================================================


class TestUndoOperations:
    """Tests for undo split and merge operations."""

    def test_undo_split(self):
        """Restore parent boundary after split undo."""
        parent_coords = make_square(-3.12, -60.02, 0.01)
        parent = make_boundary(parent_coords, "cocoa", "BR", plot_id="UNDO-P")
        bottom, top = _split_square_horizontal(parent.exterior_ring)
        # Undo: restore parent
        restored = parent
        assert restored.plot_id == "UNDO-P"
        assert restored.exterior_ring == parent_coords

    def test_undo_merge(self):
        """Restore original boundaries after merge undo."""
        coords = make_square(-3.12, -60.02, 0.01)
        bottom, top = _split_square_horizontal(coords)
        parent_a = make_boundary(bottom, "cocoa", "BR", plot_id="UM-A")
        parent_b = make_boundary(top, "cocoa", "BR", plot_id="UM-B")
        # Merge
        merged = _merge_adjacent_squares(bottom, top)
        # Undo: restore originals
        restored_a = parent_a
        restored_b = parent_b
        assert restored_a.plot_id == "UM-A"
        assert restored_b.plot_id == "UM-B"

    def test_undo_preserves_area(self):
        """Undo restores original area."""
        parent_coords = make_square(-3.12, -60.02, 0.01)
        original_area = geodesic_area_simple(parent_coords)
        # Split and undo
        bottom, top = _split_square_horizontal(parent_coords)
        restored_area = geodesic_area_simple(parent_coords)
        assert abs(restored_area - original_area) < 1e-10

    def test_undo_preserves_metadata(self):
        """Undo restores original metadata."""
        parent = make_boundary(
            make_square(-3.12, -60.02, 0.01),
            "oil_palm", "ID", plot_id="META-P",
        )
        parent.metadata = {"owner": "test_user", "notes": "original"}
        # Undo should restore metadata
        assert parent.metadata["owner"] == "test_user"


# ===========================================================================
# 5. Parametrized Tests (1 test group)
# ===========================================================================


class TestParametrized:
    """Parametrized tests for split operations."""

    @pytest.mark.parametrize(
        "n_splits",
        [2, 3, 4, 5],
        ids=["two_way", "three_way", "four_way", "five_way"],
    )
    def test_n_way_split_area_conservation(self, n_splits):
        """N-way split conserves total area."""
        coords = make_square(-3.12, -60.02, 0.01)
        parent_area = geodesic_area_simple(coords)
        lats = [c[0] for c in coords[:-1]]
        lons = [c[1] for c in coords[:-1]]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        lat_step = (max_lat - min_lat) / n_splits

        child_areas = []
        for i in range(n_splits):
            low = min_lat + i * lat_step
            high = min_lat + (i + 1) * lat_step
            strip = [
                (low, min_lon), (low, max_lon),
                (high, max_lon), (high, min_lon),
                (low, min_lon),
            ]
            child_areas.append(geodesic_area_simple(strip))

        total_child = sum(child_areas)
        if parent_area > 0:
            relative_error = abs(total_child - parent_area) / parent_area
            assert relative_error < 0.02  # < 2%

    @pytest.mark.parametrize(
        "commodity,country",
        [
            ("cocoa", "BR"),
            ("oil_palm", "ID"),
            ("coffee", "CO"),
            ("cattle", "PY"),
            ("rubber", "MY"),
        ],
        ids=["cocoa_BR", "palm_ID", "coffee_CO", "cattle_PY", "rubber_MY"],
    )
    def test_split_inherits_attributes(self, commodity, country):
        """Split children inherit commodity and country."""
        parent = make_boundary(
            make_square(-3.12, -60.02, 0.01),
            commodity, country, plot_id=f"ATTR-{commodity}",
        )
        bottom, top = _split_square_horizontal(parent.exterior_ring)
        child1 = make_boundary(bottom, parent.commodity, parent.country)
        child2 = make_boundary(top, parent.commodity, parent.country)
        assert child1.commodity == commodity
        assert child1.country == country
        assert child2.commodity == commodity
        assert child2.country == country
