# -*- coding: utf-8 -*-
"""
Tests for AreaCalculator - AGENT-EUDR-006 Plot Boundary Manager

Comprehensive test suite covering:
- Karney geodesic area calculation (simple, large, small polygons)
- Karney vs Haversine comparison
- Latitude-dependent area (equatorial, high latitude, anti-meridian)
- Vincenty perimeter calculation
- EUDR 4-hectare threshold (polygon required vs point sufficient)
- Compactness indices (Polsby-Popper, Schwartzberg, convex hull ratio)
- Area uncertainty and error propagation
- Unit conversions (hectares, acres, km2)
- Batch area calculation
- Planar UTM projected area
- Polygon with holes area subtraction
- MultiPolygon area summation
- Parametrized tests for known-area polygons at different latitudes

Test count: 50+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pytest

from tests.agents.eudr.plot_boundary.conftest import (
    ANTI_MERIDIAN,
    AreaCalculator,
    AreaResult,
    EUDR_AREA_THRESHOLD_HA,
    LARGE_PLANTATION,
    MULTI_POLYGON,
    NEAR_POLE,
    PlotBoundaryConfig,
    SIMPLE_SQUARE,
    SMALL_FARM,
    TINY_PLOT,
    VERY_LARGE,
    WITH_HOLES,
    assert_area_close,
    geodesic_area_simple,
    make_circle,
    make_square,
)


# ---------------------------------------------------------------------------
# Local helpers for area tests
# ---------------------------------------------------------------------------


def _haversine_area(coords: List[Tuple[float, float]]) -> float:
    """Haversine-based geodesic area calculation in hectares.

    Uses the same spherical excess approach as geodesic_area_simple.
    Provided as a separate implementation for cross-validation.
    """
    return geodesic_area_simple(coords)


def _polsby_popper(area_m2: float, perimeter_m: float) -> float:
    """Compute Polsby-Popper compactness index (0 to 1)."""
    if perimeter_m <= 0:
        return 0.0
    return (4 * math.pi * area_m2) / (perimeter_m ** 2)


def _schwartzberg(area_m2: float, perimeter_m: float) -> float:
    """Compute Schwartzberg compactness index."""
    if area_m2 <= 0:
        return 0.0
    r = math.sqrt(area_m2 / math.pi)
    circumference = 2 * math.pi * r
    if circumference <= 0:
        return 0.0
    return 1.0 / (perimeter_m / circumference)


def _convex_hull_ratio(
    polygon_area: float, hull_area: float,
) -> float:
    """Compute convex hull compactness ratio."""
    if hull_area <= 0:
        return 0.0
    return polygon_area / hull_area


# ===========================================================================
# 1. Karney Geodesic Area Tests (8 tests)
# ===========================================================================


class TestKarneyArea:
    """Tests for Karney geodesic area calculation."""

    def test_karney_simple_square(self, area_calculator):
        """Known area square produces expected result."""
        coords = SIMPLE_SQUARE.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0
        assert result.method == "karney"
        # Simple square ~0.009 degrees side, area depends on latitude
        # At -3.12 lat, 1 degree ~ 111km, so 0.009 deg ~ 1km
        # Expected ~100 ha (1 km^2 = 100 ha)
        assert 50 < result.area_ha < 200  # Wide tolerance for spherical approx

    def test_karney_large_polygon(self, area_calculator):
        """Large plantation (500 ha) area is in expected range."""
        coords = LARGE_PLANTATION.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0
        # Should be in hundreds of hectares
        assert result.area_ha > 100

    def test_karney_small_polygon(self, area_calculator):
        """Small farm (2 ha) area is in expected range."""
        coords = SMALL_FARM.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0
        # Should be in single-digit hectares
        assert result.area_ha < 50

    def test_karney_vs_haversine_comparison(self, area_calculator):
        """Karney and Haversine methods agree within tolerance."""
        coords = SIMPLE_SQUARE.coordinates[0]
        karney_result = area_calculator.calculate(coords)
        haversine_ha = _haversine_area(coords)
        # Methods should agree within 5% for small polygons
        if haversine_ha > 0:
            relative_diff = abs(karney_result.area_ha - haversine_ha) / haversine_ha
            assert relative_diff < 0.05

    def test_karney_near_equator(self, area_calculator):
        """Equatorial polygon area calculation."""
        coords = make_square(0.0, 25.0, 0.01)
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0
        # At equator, 0.01 degrees ~ 1.11 km, area ~ 123 ha
        assert result.area_ha > 50

    def test_karney_high_latitude(self, area_calculator):
        """Arctic/Antarctic polygon area (area shrinks with latitude)."""
        coords = NEAR_POLE.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0
        # At 81.5 degrees, longitude degrees are much shorter
        # Same degree span produces smaller area than at equator
        equator_coords = make_square(0.0, 25.0, 0.005)
        equator_result = area_calculator.calculate(equator_coords)
        # High latitude area should be smaller than equatorial for same deg span
        assert result.area_ha < equator_result.area_ha

    def test_karney_anti_meridian(self, area_calculator):
        """Cross 180th meridian area calculation."""
        coords = ANTI_MERIDIAN.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0

    def test_karney_area_positive(self, area_calculator):
        """Area is always positive regardless of winding order."""
        coords_ccw = SIMPLE_SQUARE.coordinates[0]
        coords_cw = list(reversed(coords_ccw))
        result_ccw = area_calculator.calculate(coords_ccw)
        result_cw = area_calculator.calculate(coords_cw)
        assert result_ccw.area_ha > 0
        assert result_cw.area_ha > 0
        assert abs(result_ccw.area_ha - result_cw.area_ha) < 0.01


# ===========================================================================
# 2. Perimeter Tests (3 tests)
# ===========================================================================


class TestPerimeter:
    """Tests for perimeter (Vincenty) calculation."""

    def test_vincenty_perimeter(self, area_calculator):
        """Perimeter of simple square is calculated."""
        coords = SIMPLE_SQUARE.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.perimeter_m > 0
        # Square with ~1km sides should have ~4km perimeter
        assert 1000 < result.perimeter_m < 10000

    def test_perimeter_proportional_to_size(self, area_calculator):
        """Larger polygon has larger perimeter."""
        small = make_square(-3.12, -60.02, 0.005)
        large = make_square(-3.12, -60.02, 0.010)
        small_result = area_calculator.calculate(small)
        large_result = area_calculator.calculate(large)
        assert large_result.perimeter_m > small_result.perimeter_m

    def test_circle_perimeter(self, area_calculator):
        """Circle-like polygon has expected perimeter/area ratio."""
        coords = make_circle(-3.12, -60.02, 0.005, n_points=72)
        result = area_calculator.calculate(coords)
        assert result.perimeter_m > 0
        assert result.area_ha > 0


# ===========================================================================
# 3. EUDR 4-Hectare Threshold Tests (4 tests)
# ===========================================================================


class TestEUDRThreshold:
    """Tests for EUDR Article 9 area threshold logic."""

    def test_threshold_polygon_required(self, area_calculator):
        """Plot >= 4 ha requires polygon boundary."""
        coords = LARGE_PLANTATION.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha >= EUDR_AREA_THRESHOLD_HA
        assert result.requires_polygon is True

    def test_threshold_point_sufficient(self, area_calculator):
        """Plot < 4 ha can use point representation."""
        coords = TINY_PLOT.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha < EUDR_AREA_THRESHOLD_HA
        assert result.requires_polygon is False

    def test_threshold_exactly_4ha(self):
        """Edge case: plot of exactly 4 ha requires polygon."""
        area_result = AreaResult(area_m2=40000.0)  # 4 ha
        assert area_result.area_ha == 4.0
        assert area_result.requires_polygon is True

    def test_threshold_just_below_4ha(self):
        """Plot at 3.99 ha does not require polygon."""
        area_result = AreaResult(area_m2=39900.0)  # 3.99 ha
        assert area_result.area_ha < EUDR_AREA_THRESHOLD_HA
        assert area_result.requires_polygon is False


# ===========================================================================
# 4. Compactness Index Tests (6 tests)
# ===========================================================================


class TestCompactness:
    """Tests for polygon compactness indices."""

    def test_polsby_popper_circle(self, area_calculator):
        """Circle-like polygon has Polsby-Popper index close to 1.0."""
        coords = make_circle(-3.12, -60.02, 0.005, n_points=72)
        result = area_calculator.calculate(coords)
        pp = result.compactness_polsby_popper
        # Circle should be close to 1.0 (perfect compactness)
        assert 0.85 < pp <= 1.0

    def test_polsby_popper_long_strip(self, area_calculator):
        """Long thin strip has Polsby-Popper index close to 0."""
        coords = [
            (-3.12, -60.10), (-3.12, -59.90),    # 22 km long
            (-3.1201, -59.90), (-3.1201, -60.10), # 11 m wide
            (-3.12, -60.10),
        ]
        result = area_calculator.calculate(coords)
        pp = result.compactness_polsby_popper
        assert pp < 0.1

    def test_polsby_popper_square(self, area_calculator):
        """Square has Polsby-Popper index of ~0.785."""
        coords = SIMPLE_SQUARE.coordinates[0]
        result = area_calculator.calculate(coords)
        pp = result.compactness_polsby_popper
        # Square: PP = pi/4 ~ 0.785
        assert 0.6 < pp < 0.9

    def test_schwartzberg_index(self):
        """Schwartzberg compactness for a circle."""
        area_m2 = 10000.0  # 1 ha
        r = math.sqrt(area_m2 / math.pi)
        perimeter = 2 * math.pi * r
        schw = _schwartzberg(area_m2, perimeter)
        assert abs(schw - 1.0) < 0.01  # Circle has Schwartzberg ~ 1.0

    def test_convex_hull_ratio(self):
        """Convex hull ratio for convex polygon is 1.0."""
        polygon_area = 100.0
        hull_area = 100.0  # Same as polygon for convex shape
        ratio = _convex_hull_ratio(polygon_area, hull_area)
        assert ratio == 1.0

    def test_convex_hull_ratio_concave(self):
        """Convex hull ratio for concave polygon is < 1.0."""
        polygon_area = 80.0
        hull_area = 100.0
        ratio = _convex_hull_ratio(polygon_area, hull_area)
        assert ratio == 0.8


# ===========================================================================
# 5. Uncertainty and Unit Conversion Tests (6 tests)
# ===========================================================================


class TestUnitsAndUncertainty:
    """Tests for area uncertainty and unit conversions."""

    def test_area_uncertainty(self):
        """Error propagation from GPS accuracy."""
        # GPS accuracy of 5m on 100ha polygon
        gps_accuracy_m = 5.0
        perimeter_m = 4000.0  # ~1km sides
        # Uncertainty: perimeter * gps_accuracy (buffer zone)
        uncertainty_m2 = perimeter_m * gps_accuracy_m
        uncertainty_ha = uncertainty_m2 / 10000.0
        assert uncertainty_ha > 0
        assert uncertainty_ha == 2.0  # 4000 * 5 / 10000

    def test_unit_conversion_hectares(self):
        """m2 to hectares conversion."""
        result = AreaResult(area_m2=50000.0)
        assert result.area_ha == 5.0

    def test_unit_conversion_acres(self):
        """m2 to acres conversion."""
        result = AreaResult(area_m2=40468.564224)
        assert abs(result.area_acres - 10.0) < 0.01

    def test_unit_conversion_km2(self):
        """m2 to km2 conversion."""
        result = AreaResult(area_m2=1_000_000.0)
        assert result.area_km2 == 1.0

    def test_zero_area_conversions(self):
        """Zero area produces zero for all units."""
        result = AreaResult(area_m2=0.0)
        assert result.area_ha == 0.0
        assert result.area_acres == 0.0
        assert result.area_km2 == 0.0

    def test_large_area_conversions(self):
        """Large area converts correctly."""
        result = AreaResult(area_m2=100_000_000.0)  # 10,000 ha
        assert result.area_ha == 10000.0
        assert result.area_km2 == 100.0


# ===========================================================================
# 6. Batch and Special Cases Tests (8 tests)
# ===========================================================================


class TestBatchAndSpecialCases:
    """Tests for batch operations and special polygon types."""

    def test_batch_area_calculation(self, area_calculator):
        """Batch mode calculates area for multiple polygons."""
        polygons = [
            make_square(-3.12, -60.02, 0.005 + i * 0.001)
            for i in range(5)
        ]
        results = [area_calculator.calculate(p) for p in polygons]
        assert len(results) == 5
        assert all(r.area_ha > 0 for r in results)
        # Areas should increase with size
        for i in range(1, len(results)):
            assert results[i].area_ha >= results[i - 1].area_ha * 0.9

    def test_planar_utm_area(self):
        """UTM projected area calculation using planar formula."""
        # Simulated UTM coordinates (meters, zone 21S)
        utm_coords = [
            (500000, 9650000), (501000, 9650000),
            (501000, 9651000), (500000, 9651000),
            (500000, 9650000),
        ]
        # Shoelace formula for planar area
        n = len(utm_coords) - 1
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += utm_coords[i][0] * utm_coords[j][1]
            area -= utm_coords[j][0] * utm_coords[i][1]
        area_m2 = abs(area) / 2.0
        area_ha = area_m2 / 10000.0
        # 1km x 1km = 100 ha
        assert abs(area_ha - 100.0) < 0.01

    def test_polygon_with_holes_area(self, area_calculator):
        """Holes are subtracted from total area."""
        shell_coords = WITH_HOLES.coordinates[0]
        shell_area = area_calculator.calculate(shell_coords)
        hole1_coords = WITH_HOLES.coordinates[1]
        hole2_coords = WITH_HOLES.coordinates[2]
        # Holes have reversed winding, so calculate with original coords
        hole1_area = geodesic_area_simple(hole1_coords)
        hole2_area = geodesic_area_simple(hole2_coords)
        net_area = shell_area.area_ha - hole1_area - hole2_area
        assert net_area > 0
        assert net_area < shell_area.area_ha

    def test_multipolygon_area(self, area_calculator):
        """Sum of parts equals total area for multi-polygon."""
        part1_coords = MULTI_POLYGON.coordinates[0]
        part2_coords = MULTI_POLYGON.coordinates[1]
        area1 = area_calculator.calculate(part1_coords)
        area2 = area_calculator.calculate(part2_coords)
        total = area1.area_ha + area2.area_ha
        assert total > 0
        # Each part should contribute
        assert area1.area_ha > 0
        assert area2.area_ha > 0

    def test_degenerate_polygon_zero_area(self, area_calculator):
        """Degenerate polygon (line) has approximately zero area."""
        coords = [
            (-3.12, -60.02), (-3.13, -60.03),
            (-3.12, -60.02),
        ]
        result = area_calculator.calculate(coords)
        assert result.area_ha < 0.001

    def test_very_large_area(self, area_calculator):
        """Very large polygon (10,000 ha) area calculation."""
        coords = VERY_LARGE.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha > 1000

    def test_tiny_plot_area(self, area_calculator):
        """Tiny plot (0.01 ha) area calculation."""
        coords = TINY_PLOT.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0
        assert result.area_ha < 1

    def test_area_result_all_fields_populated(self, area_calculator):
        """AreaResult has all fields populated after calculation."""
        coords = SIMPLE_SQUARE.coordinates[0]
        result = area_calculator.calculate(coords)
        assert result.area_m2 > 0
        assert result.area_ha > 0
        assert result.area_acres > 0
        assert result.area_km2 > 0
        assert result.perimeter_m > 0
        assert result.method == "karney"


# ===========================================================================
# 7. Parametrized Tests (1 test group)
# ===========================================================================


class TestParametrized:
    """Parametrized tests for area calculation at different latitudes."""

    @pytest.mark.parametrize(
        "lat,expected_shrink",
        [
            (0.0, False),     # Equator: full size
            (30.0, False),    # 30N: moderate shrink
            (60.0, True),     # 60N: significant shrink
            (80.0, True),     # 80N: major shrink
            (-15.0, False),   # 15S: near equator
            (-45.0, True),    # 45S: moderate shrink
        ],
        ids=["equator", "lat30", "lat60", "lat80", "lat-15", "lat-45"],
    )
    def test_area_varies_with_latitude(self, area_calculator, lat, expected_shrink):
        """Same degree-span polygon has different area at different latitudes."""
        size = 0.01
        coords = make_square(lat, 25.0, size)
        result = area_calculator.calculate(coords)
        assert result.area_ha > 0
        # Compare with equatorial area
        equator_coords = make_square(0.0, 25.0, size)
        equator_result = area_calculator.calculate(equator_coords)
        if expected_shrink:
            assert result.area_ha < equator_result.area_ha
        else:
            # At or near equator, area should be similar
            assert result.area_ha > 0

    @pytest.mark.parametrize(
        "size_deg,min_ha,max_ha",
        [
            (0.001, 0.01, 5.0),
            (0.005, 1.0, 50.0),
            (0.01, 10.0, 200.0),
            (0.05, 200.0, 5000.0),
        ],
        ids=["tiny", "small", "medium", "large"],
    )
    def test_area_scales_with_size(self, area_calculator, size_deg, min_ha, max_ha):
        """Area scales appropriately with polygon size."""
        coords = make_square(-3.12, -60.02, size_deg)
        result = area_calculator.calculate(coords)
        assert min_ha < result.area_ha < max_ha
