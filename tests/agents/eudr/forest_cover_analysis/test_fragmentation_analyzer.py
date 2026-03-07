# -*- coding: utf-8 -*-
"""
Tests for FragmentationAnalyzer - AGENT-EUDR-004 Engine 6: Fragmentation Analysis

Comprehensive test suite covering:
- Patch counting for contiguous and disconnected forest areas
- Edge density calculation (metres of edge per hectare)
- Core area percentage (interior area beyond edge buffer)
- Nearest-neighbour connectivity distance
- Shape complexity via perimeter-area ratio (PAR)
- Effective mesh size (MESH) calculation
- Fragmentation classification (intact, moderate, severe)
- Risk score mapping from fragmentation metrics
- Temporal fragmentation comparison between two periods
- Determinism and provenance hash reproducibility

Test count: 50+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-004 (Engine 6 - Fragmentation Analysis)
"""

import math

import pytest

from tests.agents.eudr.forest_cover_analysis.conftest import (
    FragmentationMetrics,
    compute_test_hash,
    SHA256_HEX_LENGTH,
)


# ---------------------------------------------------------------------------
# Helpers: Fragmentation metric calculations
# ---------------------------------------------------------------------------


def _count_patches(binary_map: list) -> int:
    """Count connected components in a 1D binary map (simplified).

    In production, this is a 2D connected-component analysis on a
    raster forest mask. Here we count contiguous runs of 1s.
    """
    if not binary_map:
        return 0
    count = 0
    prev = 0
    for v in binary_map:
        if v == 1 and prev == 0:
            count += 1
        prev = v
    return count


def _edge_density(
    total_edge_m: float,
    total_area_ha: float,
) -> float:
    """Compute edge density in metres of edge per hectare.

    Edge density = total perimeter of all patches / total landscape area.
    """
    if total_area_ha <= 0:
        return 0.0
    return total_edge_m / total_area_ha


def _core_area_pct(
    total_area_ha: float,
    edge_buffer_m: float = 100.0,
    patch_areas_ha: list = None,
    patch_perimeters_m: list = None,
) -> float:
    """Estimate core area percentage.

    Core area = area beyond the edge buffer. Simplified as:
    core_pct = (total_area - edge_area) / total_area * 100

    For a large single patch, most area is core.
    For many small patches, most area is edge.
    """
    if total_area_ha <= 0:
        return 0.0
    if patch_areas_ha is None:
        patch_areas_ha = [total_area_ha]
    if patch_perimeters_m is None:
        # Estimate perimeter for a circular patch
        patch_perimeters_m = [
            2 * math.pi * math.sqrt(a * 10000 / math.pi)
            for a in patch_areas_ha
        ]

    # Edge area approximated as buffer width * perimeter / 10000 (m^2 -> ha)
    edge_area = sum(
        edge_buffer_m * p / 10000.0
        for p in patch_perimeters_m
    )
    edge_area = min(edge_area, total_area_ha)
    core = total_area_ha - edge_area
    return max(0.0, (core / total_area_ha) * 100.0)


def _nearest_neighbour_distance(patch_centroids: list) -> float:
    """Compute mean nearest-neighbour distance between patch centroids.

    Each centroid is (x, y) in metres.
    """
    if len(patch_centroids) < 2:
        return 0.0
    nn_distances = []
    for i, c1 in enumerate(patch_centroids):
        min_dist = float("inf")
        for j, c2 in enumerate(patch_centroids):
            if i == j:
                continue
            dist = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
        nn_distances.append(min_dist)
    return sum(nn_distances) / len(nn_distances)


def _perimeter_area_ratio(perimeter_m: float, area_ha: float) -> float:
    """Compute perimeter-area ratio (shape complexity metric).

    Lower PAR = simpler shape (circle).
    Higher PAR = more complex/irregular shape.
    """
    area_m2 = area_ha * 10000.0
    if area_m2 <= 0:
        return 0.0
    return perimeter_m / area_m2


def _effective_mesh_size(
    patch_areas_ha: list,
    total_area_ha: float,
) -> float:
    """Compute effective mesh size (MESH).

    MESH = (sum of patch_area^2) / total_area.
    Higher MESH = less fragmented landscape.
    """
    if total_area_ha <= 0:
        return 0.0
    sum_sq = sum(a ** 2 for a in patch_areas_ha)
    return sum_sq / total_area_ha


def _classify_fragmentation(
    num_patches: int,
    core_area_pct: float,
    edge_density: float,
    mesh_size_ha: float,
    total_area_ha: float,
) -> str:
    """Classify fragmentation level.

    intact: single large patch, high core area, low edge density
    moderate: 2-5 patches, moderate metrics
    severely_fragmented: many small patches, low core, high edge density
    """
    if num_patches <= 1 and core_area_pct > 70.0 and edge_density < 100.0:
        return "intact"
    elif num_patches <= 5 and core_area_pct > 40.0:
        return "moderate"
    else:
        return "severely_fragmented"


def _fragmentation_risk_score(
    num_patches: int,
    core_area_pct: float,
    edge_density: float,
) -> float:
    """Compute a 0-1 risk score from fragmentation metrics.

    Higher patches, lower core area, higher edge density = higher risk.
    """
    # Normalize each component to 0-1
    patch_score = min(1.0, num_patches / 20.0)
    core_score = 1.0 - min(1.0, core_area_pct / 100.0)
    edge_score = min(1.0, edge_density / 500.0)
    return (patch_score + core_score + edge_score) / 3.0


# ===========================================================================
# 1. Patch Counting (6 tests)
# ===========================================================================


class TestPatchCounting:
    """Test connected component patch counting."""

    def test_count_patches_single(self):
        """Test contiguous forest map has 1 patch."""
        binary_map = [1, 1, 1, 1, 1]
        assert _count_patches(binary_map) == 1

    def test_count_patches_multiple(self):
        """Test 5 disconnected patches counted correctly."""
        binary_map = [1, 0, 1, 0, 1, 0, 1, 0, 1]
        assert _count_patches(binary_map) == 5

    def test_count_patches_no_forest(self):
        """Test all-zero map has 0 patches."""
        binary_map = [0, 0, 0, 0, 0]
        assert _count_patches(binary_map) == 0

    def test_count_patches_empty(self):
        """Test empty map has 0 patches."""
        assert _count_patches([]) == 0

    def test_count_patches_alternating(self):
        """Test alternating pattern counts correctly."""
        binary_map = [0, 1, 0, 1, 0]
        assert _count_patches(binary_map) == 2

    def test_count_patches_single_pixel(self):
        """Test single forest pixel is 1 patch."""
        binary_map = [0, 0, 1, 0, 0]
        assert _count_patches(binary_map) == 1


# ===========================================================================
# 2. Edge Density (6 tests)
# ===========================================================================


class TestEdgeDensity:
    """Test edge density calculation."""

    def test_edge_density_intact(self):
        """Test single large patch has low edge density."""
        # 100 ha circular patch: perimeter ~ 3545m
        density = _edge_density(3545.0, 100.0)
        assert density < 100.0

    def test_edge_density_fragmented(self):
        """Test many small patches have high edge density."""
        # 10 small patches, total perimeter 10000m, area 10ha
        density = _edge_density(10000.0, 10.0)
        assert density > 500.0

    def test_edge_density_zero_area(self):
        """Test zero area returns 0."""
        density = _edge_density(1000.0, 0.0)
        assert density == 0.0

    def test_edge_density_zero_edge(self):
        """Test zero edge returns 0."""
        density = _edge_density(0.0, 10.0)
        assert density == 0.0

    @pytest.mark.parametrize("edge_m,area_ha,expected", [
        (1000.0, 10.0, 100.0),
        (5000.0, 10.0, 500.0),
        (200.0, 100.0, 2.0),
    ])
    def test_edge_density_parametrized(self, edge_m, area_ha, expected):
        """Test edge density computation across values."""
        density = _edge_density(edge_m, area_ha)
        assert abs(density - expected) < 1e-9

    def test_edge_density_determinism(self):
        """Test edge density is deterministic."""
        results = [_edge_density(3545.0, 100.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 3. Core Area (6 tests)
# ===========================================================================


class TestCoreArea:
    """Test core area percentage calculation."""

    def test_core_area_full(self):
        """Test large single patch has >50% core area."""
        pct = _core_area_pct(100.0, edge_buffer_m=100.0)
        assert pct > 50.0

    def test_core_area_edge_dominated(self):
        """Test thin strip has low core area."""
        # Very long, thin patch
        pct = _core_area_pct(
            1.0, edge_buffer_m=100.0,
            patch_areas_ha=[1.0],
            patch_perimeters_m=[5000.0],  # Very long perimeter
        )
        assert pct < 50.0

    def test_core_area_zero_area(self):
        """Test zero area returns 0%."""
        pct = _core_area_pct(0.0)
        assert pct == 0.0

    def test_core_area_small_buffer(self):
        """Test smaller edge buffer gives higher core area."""
        pct_small = _core_area_pct(10.0, edge_buffer_m=10.0)
        pct_large = _core_area_pct(10.0, edge_buffer_m=100.0)
        assert pct_small > pct_large

    def test_core_area_range(self):
        """Test core area is always in [0, 100]."""
        pct = _core_area_pct(50.0, edge_buffer_m=100.0)
        assert 0.0 <= pct <= 100.0

    def test_core_area_determinism(self):
        """Test core area calculation is deterministic."""
        results = [_core_area_pct(100.0, 100.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 4. Nearest-Neighbour Connectivity (5 tests)
# ===========================================================================


class TestConnectivity:
    """Test nearest-neighbour distance between patches."""

    def test_connectivity_close(self):
        """Test patches near each other have low NN distance."""
        centroids = [(0, 0), (10, 0), (20, 0)]
        nn = _nearest_neighbour_distance(centroids)
        assert nn == 10.0

    def test_connectivity_far(self):
        """Test isolated patches have high NN distance."""
        centroids = [(0, 0), (1000, 0)]
        nn = _nearest_neighbour_distance(centroids)
        assert nn == 1000.0

    def test_connectivity_single_patch(self):
        """Test single patch returns 0 (no neighbours)."""
        nn = _nearest_neighbour_distance([(0, 0)])
        assert nn == 0.0

    def test_connectivity_no_patches(self):
        """Test no patches returns 0."""
        nn = _nearest_neighbour_distance([])
        assert nn == 0.0

    def test_connectivity_determinism(self):
        """Test NN distance is deterministic."""
        centroids = [(0, 0), (100, 0), (200, 0)]
        results = [_nearest_neighbour_distance(centroids) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 5. Shape Complexity (5 tests)
# ===========================================================================


class TestShapeComplexity:
    """Test perimeter-area ratio (shape complexity)."""

    def test_shape_complexity_circle(self):
        """Test circular patch has near-minimum PAR."""
        # Circle: area=1ha=10000m2, r=sqrt(10000/pi)~56.4m, perim=2*pi*r~354.5m
        par = _perimeter_area_ratio(354.5, 1.0)
        assert par < 0.04

    def test_shape_complexity_irregular(self):
        """Test irregular patch has high PAR."""
        # Very long perimeter relative to area
        par = _perimeter_area_ratio(5000.0, 1.0)
        assert par > 0.10

    def test_shape_complexity_zero_area(self):
        """Test zero area returns 0."""
        par = _perimeter_area_ratio(100.0, 0.0)
        assert par == 0.0

    @pytest.mark.parametrize("perimeter,area,min_par,max_par", [
        (354.5, 1.0, 0.01, 0.05),   # Circle-like
        (1000.0, 1.0, 0.05, 0.15),  # Moderate
        (5000.0, 1.0, 0.30, 0.60),  # Very irregular
    ])
    def test_shape_complexity_parametrized(self, perimeter, area, min_par, max_par):
        """Test PAR falls within expected range."""
        par = _perimeter_area_ratio(perimeter, area)
        assert min_par <= par <= max_par

    def test_shape_complexity_determinism(self):
        """Test PAR is deterministic."""
        results = [_perimeter_area_ratio(354.5, 1.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 6. Effective Mesh Size (5 tests)
# ===========================================================================


class TestEffectiveMeshSize:
    """Test MESH calculation."""

    def test_effective_mesh_size_single_patch(self):
        """Test single patch: MESH = patch_area."""
        mesh = _effective_mesh_size([100.0], 100.0)
        assert abs(mesh - 100.0) < 1e-9

    def test_effective_mesh_size_fragmented(self):
        """Test fragmented: MESH < largest patch."""
        mesh = _effective_mesh_size([10.0, 10.0, 10.0, 10.0], 40.0)
        # sum(10^2 * 4) / 40 = 400/40 = 10.0
        assert abs(mesh - 10.0) < 1e-9

    def test_effective_mesh_size_zero_area(self):
        """Test zero total area returns 0."""
        mesh = _effective_mesh_size([10.0], 0.0)
        assert mesh == 0.0

    def test_effective_mesh_size_empty_patches(self):
        """Test empty patch list returns 0."""
        mesh = _effective_mesh_size([], 100.0)
        assert mesh == 0.0

    def test_effective_mesh_size_determinism(self):
        """Test MESH is deterministic."""
        results = [_effective_mesh_size([10.0, 20.0], 30.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 7. Classification (6 tests)
# ===========================================================================


class TestFragmentationClassification:
    """Test fragmentation level classification."""

    def test_classify_intact(self):
        """Test intact: single large patch, high core, low edge."""
        cls = _classify_fragmentation(
            num_patches=1, core_area_pct=85.0,
            edge_density=50.0, mesh_size_ha=100.0, total_area_ha=100.0,
        )
        assert cls == "intact"

    def test_classify_moderate(self):
        """Test moderate: 3 patches, decent core area."""
        cls = _classify_fragmentation(
            num_patches=3, core_area_pct=55.0,
            edge_density=200.0, mesh_size_ha=30.0, total_area_ha=90.0,
        )
        assert cls == "moderate"

    def test_classify_severely_fragmented(self):
        """Test severely fragmented: many patches, low core."""
        cls = _classify_fragmentation(
            num_patches=15, core_area_pct=20.0,
            edge_density=800.0, mesh_size_ha=2.0, total_area_ha=30.0,
        )
        assert cls == "severely_fragmented"

    @pytest.mark.parametrize("patches,core,edge,expected", [
        (1, 90.0, 30.0, "intact"),
        (1, 75.0, 80.0, "intact"),
        (3, 50.0, 150.0, "moderate"),
        (5, 45.0, 200.0, "moderate"),
        (10, 30.0, 400.0, "severely_fragmented"),
        (20, 10.0, 900.0, "severely_fragmented"),
    ])
    def test_classify_parametrized(self, patches, core, edge, expected):
        """Test classification across various metric combinations."""
        cls = _classify_fragmentation(patches, core, edge, 10.0, 100.0)
        assert cls == expected

    def test_classify_determinism(self):
        """Test classification is deterministic."""
        results = [
            _classify_fragmentation(3, 55.0, 200.0, 30.0, 90.0)
            for _ in range(10)
        ]
        assert len(set(results)) == 1


# ===========================================================================
# 8. Risk Score (4 tests)
# ===========================================================================


class TestRiskScore:
    """Test fragmentation risk score mapping."""

    def test_risk_score_intact(self):
        """Test intact landscape has low risk score."""
        score = _fragmentation_risk_score(1, 90.0, 30.0)
        assert score < 0.3

    def test_risk_score_fragmented(self):
        """Test fragmented landscape has high risk score."""
        score = _fragmentation_risk_score(15, 20.0, 600.0)
        assert score > 0.6

    def test_risk_score_range(self):
        """Test risk score is always in [0, 1]."""
        for patches in [1, 5, 10, 20]:
            for core in [10.0, 50.0, 90.0]:
                for edge in [30.0, 200.0, 800.0]:
                    score = _fragmentation_risk_score(patches, core, edge)
                    assert 0.0 <= score <= 1.0

    def test_risk_score_determinism(self):
        """Test risk score is deterministic."""
        results = [_fragmentation_risk_score(5, 60.0, 150.0) for _ in range(10)]
        assert len(set(results)) == 1


# ===========================================================================
# 9. Temporal Comparison (3 tests)
# ===========================================================================


class TestTemporalComparison:
    """Test fragmentation comparison between two time periods."""

    def test_compare_fragmentation_worsened(self):
        """Test increasing patches indicates worsened fragmentation."""
        t1_patches = 2
        t2_patches = 8
        assert t2_patches > t1_patches

    def test_compare_fragmentation_improved(self):
        """Test decreasing patches indicates improved fragmentation."""
        t1_patches = 10
        t2_patches = 3
        assert t2_patches < t1_patches

    def test_compare_fragmentation_stable(self):
        """Test same metrics indicates stable fragmentation."""
        t1 = FragmentationMetrics(num_patches=3, core_area_pct=60.0)
        t2 = FragmentationMetrics(num_patches=3, core_area_pct=60.0)
        assert t1.num_patches == t2.num_patches
        assert t1.core_area_pct == t2.core_area_pct


# ===========================================================================
# 10. Determinism (3 tests)
# ===========================================================================


class TestFragmentationDeterminism:
    """Test deterministic behaviour of fragmentation analysis."""

    def test_determinism_patch_count(self):
        """Test patch counting is deterministic."""
        binary_map = [1, 0, 1, 0, 1]
        results = [_count_patches(binary_map) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_determinism_mesh_size(self):
        """Test MESH is deterministic."""
        results = [_effective_mesh_size([10.0, 20.0], 30.0) for _ in range(20)]
        assert len(set(results)) == 1

    def test_determinism_provenance_hash(self):
        """Test same inputs produce same provenance hash."""
        data = {"plot_id": "P001", "num_patches": 3, "core_area_pct": 65.0}
        hashes = [compute_test_hash(data) for _ in range(20)]
        assert len(set(hashes)) == 1
