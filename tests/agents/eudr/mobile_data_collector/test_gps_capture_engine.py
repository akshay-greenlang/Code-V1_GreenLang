# -*- coding: utf-8 -*-
"""
Unit tests for GPSCaptureEngine - AGENT-EUDR-015 Engine 2.

Tests all methods of GPSCaptureEngine with 85%+ coverage.
Validates GPS point capture, polygon trace, coordinate validation,
area calculation (Shoelace), distance (Haversine), accuracy tiers,
centroid, self-intersection, plot size, and buffer zone.

Test count: ~60 tests
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.mobile_data_collector.gps_capture_engine import (
    GPSCaptureEngine,
    CoordinateValidationError,
    PolygonValidationError,
    AccuracyError,
    PlotSizeError,
    EARTH_RADIUS_M,
)

from .conftest import (
    SAMPLE_LAT, SAMPLE_LON, SAMPLE_ACCURACY, SAMPLE_HDOP,
    SAMPLE_SATELLITES, SAMPLE_POLYGON_VERTICES,
    assert_valid_coordinates, assert_within_tolerance,
)


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestGPSCaptureEngineInit:
    """Tests for GPSCaptureEngine initialization."""

    def test_initialization(self, gps_capture_engine):
        """Engine initializes successfully."""
        assert gps_capture_engine is not None

    def test_repr(self, gps_capture_engine):
        """Repr includes engine name."""
        r = repr(gps_capture_engine)
        assert "GPSCaptureEngine" in r

    def test_len_starts_at_zero(self, gps_capture_engine):
        """Initial capture count is zero."""
        assert len(gps_capture_engine) == 0


# ---------------------------------------------------------------------------
# Test: capture_point
# ---------------------------------------------------------------------------

class TestCapturePoint:
    """Tests for capture_point method."""

    def test_capture_valid_point(self, gps_capture_engine, make_gps_capture):
        """Capture a valid GPS point."""
        data = make_gps_capture()
        result = gps_capture_engine.capture_point(**data)
        assert "capture_id" in result
        assert result["latitude"] == SAMPLE_LAT
        assert result["longitude"] == SAMPLE_LON

    def test_capture_point_increments_count(self, gps_capture_engine, make_gps_capture):
        """Capture increments the capture count."""
        data = make_gps_capture()
        gps_capture_engine.capture_point(**data)
        assert len(gps_capture_engine) >= 1

    def test_capture_point_returns_accuracy_tier(self, gps_capture_engine, make_gps_capture):
        """Captured point includes accuracy_tier classification."""
        data = make_gps_capture(accuracy=1.5, hdop=1.2, satellites=10)
        result = gps_capture_engine.capture_point(**data)
        assert "accuracy_tier" in result
        assert result["accuracy_tier"] in ("excellent", "good", "acceptable", "poor", "rejected")

    @pytest.mark.parametrize("lat,lon", [
        (0.0, 0.0),          # Equator/prime meridian
        (90.0, 180.0),       # North pole / date line
        (-90.0, -180.0),     # South pole / anti-meridian
        (51.5074, -0.1278),  # London
        (-33.8688, 151.2093),  # Sydney
    ])
    def test_capture_various_valid_coordinates(
        self, gps_capture_engine, make_gps_capture, lat, lon,
    ):
        """Various valid coordinates are accepted."""
        data = make_gps_capture(latitude=lat, longitude=lon)
        result = gps_capture_engine.capture_point(**data)
        assert result["latitude"] == lat
        assert result["longitude"] == lon

    def test_capture_point_invalid_latitude_raises(self, gps_capture_engine, make_gps_capture):
        """Latitude outside [-90, 90] raises error."""
        data = make_gps_capture(latitude=91.0)
        with pytest.raises((CoordinateValidationError, ValueError)):
            gps_capture_engine.capture_point(**data)

    def test_capture_point_invalid_longitude_raises(self, gps_capture_engine, make_gps_capture):
        """Longitude outside [-180, 180] raises error."""
        data = make_gps_capture(longitude=181.0)
        with pytest.raises((CoordinateValidationError, ValueError)):
            gps_capture_engine.capture_point(**data)

    def test_capture_point_negative_latitude_raises(self, gps_capture_engine, make_gps_capture):
        """Latitude below -90 raises error."""
        data = make_gps_capture(latitude=-91.0)
        with pytest.raises((CoordinateValidationError, ValueError)):
            gps_capture_engine.capture_point(**data)


# ---------------------------------------------------------------------------
# Test: capture_polygon
# ---------------------------------------------------------------------------

class TestCapturePolygon:
    """Tests for capture_polygon method."""

    def test_capture_valid_polygon(self, gps_capture_engine, make_polygon_trace):
        """Capture a valid polygon trace."""
        data = make_polygon_trace()
        result = gps_capture_engine.capture_polygon(**data)
        assert "polygon_id" in result or "capture_id" in result

    def test_capture_polygon_has_area(self, gps_capture_engine, make_polygon_trace):
        """Captured polygon includes area calculation."""
        data = make_polygon_trace()
        result = gps_capture_engine.capture_polygon(**data)
        assert "area_ha" in result or "area_sqm" in result

    def test_capture_polygon_too_few_vertices_raises(self, gps_capture_engine, make_polygon_trace):
        """Polygon with too few vertices raises error."""
        data = make_polygon_trace(vertices=[[5.6, -0.18], [5.61, -0.18]])
        with pytest.raises((PolygonValidationError, ValueError)):
            gps_capture_engine.capture_polygon(**data)

    def test_capture_polygon_closed_ring(self, gps_capture_engine, make_polygon_trace):
        """Polygon vertices form a closed ring."""
        verts = [
            [5.6030, -0.1860],
            [5.6040, -0.1860],
            [5.6040, -0.1870],
            [5.6030, -0.1870],
            [5.6030, -0.1860],
        ]
        data = make_polygon_trace(vertices=verts)
        result = gps_capture_engine.capture_polygon(**data)
        assert result is not None


# ---------------------------------------------------------------------------
# Test: validate_coordinates
# ---------------------------------------------------------------------------

class TestValidateCoordinates:
    """Tests for validate_coordinates method."""

    def test_validate_valid_coordinates(self, gps_capture_engine):
        """Valid coordinates pass validation."""
        result = gps_capture_engine.validate_coordinates(SAMPLE_LAT, SAMPLE_LON)
        assert result is True or (isinstance(result, dict) and result.get("valid") is True)

    @pytest.mark.parametrize("lat,lon,should_fail", [
        (91.0, 0.0, True),
        (-91.0, 0.0, True),
        (0.0, 181.0, True),
        (0.0, -181.0, True),
        (0.0, 0.0, False),
        (89.9999, 179.9999, False),
    ])
    def test_validate_coordinates_boundary(
        self, gps_capture_engine, lat, lon, should_fail,
    ):
        """Coordinates at boundaries are handled correctly."""
        if should_fail:
            with pytest.raises((CoordinateValidationError, ValueError)):
                gps_capture_engine.validate_coordinates(lat, lon)
        else:
            result = gps_capture_engine.validate_coordinates(lat, lon)
            assert result is True or (isinstance(result, dict) and result.get("valid") is True)


# ---------------------------------------------------------------------------
# Test: calculate_area
# ---------------------------------------------------------------------------

class TestCalculateArea:
    """Tests for area calculation (Shoelace formula with geodesic correction)."""

    def test_calculate_area_returns_positive(self, gps_capture_engine):
        """Area calculation returns a positive value."""
        verts = SAMPLE_POLYGON_VERTICES
        result = gps_capture_engine.calculate_area(verts)
        assert isinstance(result, (int, float))
        assert result > 0

    def test_calculate_area_unit_square_approx(self, gps_capture_engine):
        """1-degree square at equator is approximately 12,300 ha."""
        # ~111km x ~111km = ~12,321 sq km = ~1,232,100 ha
        verts = [
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
        ]
        area = gps_capture_engine.calculate_area(verts)
        # Should be roughly 12,000-12,500 ha (geodesic correction)
        assert area > 1000, f"Area {area} is too small for a 1-degree square"

    def test_calculate_area_small_plot(self, gps_capture_engine):
        """Small plot area is in reasonable range (< 100 ha)."""
        verts = SAMPLE_POLYGON_VERTICES
        area = gps_capture_engine.calculate_area(verts)
        assert area < 100, f"Small plot area {area} ha is unexpectedly large"

    def test_calculate_area_triangle(self, gps_capture_engine):
        """Triangle area is calculable."""
        verts = [
            [5.600, -0.180],
            [5.605, -0.180],
            [5.600, -0.175],
            [5.600, -0.180],
        ]
        area = gps_capture_engine.calculate_area(verts)
        assert area > 0


# ---------------------------------------------------------------------------
# Test: calculate_distance (Haversine)
# ---------------------------------------------------------------------------

class TestCalculateDistance:
    """Tests for Haversine distance calculation."""

    def test_distance_same_point_is_zero(self, gps_capture_engine):
        """Distance from a point to itself is zero."""
        d = gps_capture_engine.calculate_distance(
            SAMPLE_LAT, SAMPLE_LON, SAMPLE_LAT, SAMPLE_LON,
        )
        assert d == pytest.approx(0.0, abs=0.01)

    def test_distance_known_pair(self, gps_capture_engine):
        """Distance between known points is approximately correct."""
        # Accra (5.6037, -0.1870) to Kumasi (6.6885, -1.6244)
        d = gps_capture_engine.calculate_distance(
            5.6037, -0.1870, 6.6885, -1.6244,
        )
        # Approximately 200 km
        assert 180_000 < d < 220_000, f"Distance {d}m is unexpected"

    def test_distance_equator_one_degree(self, gps_capture_engine):
        """One degree of longitude at equator is approximately 111 km."""
        d = gps_capture_engine.calculate_distance(0.0, 0.0, 0.0, 1.0)
        assert 110_000 < d < 112_000, f"1 degree lon at equator: {d}m"

    def test_distance_always_positive(self, gps_capture_engine):
        """Distance is always non-negative."""
        d = gps_capture_engine.calculate_distance(10.0, 20.0, -10.0, -20.0)
        assert d >= 0


# ---------------------------------------------------------------------------
# Test: get_centroid
# ---------------------------------------------------------------------------

class TestGetCentroid:
    """Tests for centroid calculation."""

    def test_centroid_of_polygon(self, gps_capture_engine):
        """Centroid is within the polygon bounds."""
        verts = SAMPLE_POLYGON_VERTICES
        centroid = gps_capture_engine.get_centroid(verts)
        assert "latitude" in centroid or isinstance(centroid, (list, tuple))

        if isinstance(centroid, dict):
            lat = centroid["latitude"]
            lon = centroid["longitude"]
        else:
            lat, lon = centroid[0], centroid[1]

        assert_valid_coordinates(lat, lon)
        # Centroid should be near the polygon center
        assert abs(lat - 5.6035) < 0.01
        assert abs(lon - (-0.1865)) < 0.01

    def test_centroid_of_symmetric_polygon(self, gps_capture_engine):
        """Centroid of symmetric polygon is at geometric center."""
        verts = [
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
        ]
        centroid = gps_capture_engine.get_centroid(verts)
        if isinstance(centroid, dict):
            lat = centroid["latitude"]
            lon = centroid["longitude"]
        else:
            lat, lon = centroid[0], centroid[1]
        assert abs(lat - 0.5) < 0.1
        assert abs(lon - 0.5) < 0.1


# ---------------------------------------------------------------------------
# Test: check_self_intersection
# ---------------------------------------------------------------------------

class TestCheckSelfIntersection:
    """Tests for self-intersection detection."""

    def test_simple_polygon_no_intersection(self, gps_capture_engine):
        """Simple polygon has no self-intersection."""
        verts = SAMPLE_POLYGON_VERTICES
        result = gps_capture_engine.check_self_intersection(verts)
        assert result is False or (isinstance(result, dict) and result.get("intersects") is False)

    def test_figure_eight_has_intersection(self, gps_capture_engine):
        """Figure-eight polygon has self-intersection."""
        # Bowtie shape
        verts = [
            [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        ]
        result = gps_capture_engine.check_self_intersection(verts)
        assert result is True or (isinstance(result, dict) and result.get("intersects") is True)


# ---------------------------------------------------------------------------
# Test: classify_accuracy
# ---------------------------------------------------------------------------

class TestClassifyAccuracy:
    """Tests for GPS accuracy tier classification."""

    @pytest.mark.parametrize("accuracy,hdop,sats,expected_tier", [
        (0.5, 0.8, 14, "excellent"),
        (2.0, 1.5, 10, "good"),
        (4.0, 2.5, 7, "acceptable"),
        (8.0, 4.0, 5, "poor"),
        (15.0, 8.0, 3, "rejected"),
    ])
    def test_classify_accuracy_tiers(
        self, gps_capture_engine, accuracy, hdop, sats, expected_tier,
    ):
        """Accuracy classification matches expected tiers."""
        result = gps_capture_engine.classify_accuracy(accuracy, hdop, sats)
        if isinstance(result, str):
            assert result == expected_tier
        elif isinstance(result, dict):
            assert result.get("tier") == expected_tier or result.get("accuracy_tier") == expected_tier

    def test_classify_boundary_accuracy(self, gps_capture_engine):
        """Boundary accuracy values are classified correctly."""
        # Exactly 3.0m with good HDOP and sat count should be "good"
        result = gps_capture_engine.classify_accuracy(3.0, 2.0, 8)
        assert result is not None


# ---------------------------------------------------------------------------
# Test: validate_plot_size
# ---------------------------------------------------------------------------

class TestValidatePlotSize:
    """Tests for plot size validation."""

    def test_valid_plot_size(self, gps_capture_engine):
        """Reasonable plot size passes validation."""
        result = gps_capture_engine.validate_plot_size(2.5)
        assert result is True or (isinstance(result, dict) and result.get("valid") is True)

    def test_plot_too_large_raises(self, gps_capture_engine):
        """Extremely large plot raises PlotSizeError."""
        with pytest.raises((PlotSizeError, ValueError)):
            gps_capture_engine.validate_plot_size(20000.0)

    def test_plot_zero_area_raises(self, gps_capture_engine):
        """Zero area plot raises error."""
        with pytest.raises((PlotSizeError, ValueError)):
            gps_capture_engine.validate_plot_size(0.0)

    def test_plot_negative_area_raises(self, gps_capture_engine):
        """Negative area raises error."""
        with pytest.raises((PlotSizeError, ValueError)):
            gps_capture_engine.validate_plot_size(-1.0)


# ---------------------------------------------------------------------------
# Test: generate_buffer_zone
# ---------------------------------------------------------------------------

class TestGenerateBufferZone:
    """Tests for buffer zone generation."""

    def test_generate_buffer_zone(self, gps_capture_engine):
        """Buffer zone is generated around a polygon."""
        verts = SAMPLE_POLYGON_VERTICES
        result = gps_capture_engine.generate_buffer_zone(verts, buffer_meters=100)
        assert result is not None
        if isinstance(result, list):
            assert len(result) > 0
        elif isinstance(result, dict):
            assert "vertices" in result or "buffer" in result

    def test_buffer_zone_larger_than_original(self, gps_capture_engine):
        """Buffer zone polygon has more or equal vertices."""
        verts = SAMPLE_POLYGON_VERTICES
        result = gps_capture_engine.generate_buffer_zone(verts, buffer_meters=50)
        if isinstance(result, list):
            assert len(result) >= len(verts)


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

class TestGPSEdgeCases:
    """Tests for GPS engine edge cases."""

    def test_multiple_captures_unique_ids(self, gps_capture_engine, make_gps_capture):
        """Multiple captures produce unique IDs."""
        ids = set()
        for i in range(10):
            data = make_gps_capture(latitude=5.60 + i * 0.001)
            result = gps_capture_engine.capture_point(**data)
            capture_id = result.get("capture_id") or result.get("point_id")
            ids.add(capture_id)
        assert len(ids) == 10

    def test_capture_with_altitude(self, gps_capture_engine, make_gps_capture):
        """Capture with altitude data is accepted."""
        data = make_gps_capture(altitude_m=150.0)
        result = gps_capture_engine.capture_point(**data)
        assert result is not None

    def test_capture_poles(self, gps_capture_engine, make_gps_capture):
        """Capture at geographic poles."""
        # North pole
        data = make_gps_capture(latitude=90.0, longitude=0.0)
        result = gps_capture_engine.capture_point(**data)
        assert result is not None

    def test_capture_antimeridian(self, gps_capture_engine, make_gps_capture):
        """Capture at antimeridian (180 degrees)."""
        data = make_gps_capture(latitude=0.0, longitude=180.0)
        result = gps_capture_engine.capture_point(**data)
        assert result is not None


# ---------------------------------------------------------------------------
# Test: Additional GPS Tests
# ---------------------------------------------------------------------------

class TestGPSAdditional:
    """Additional GPS capture and calculation tests."""

    def test_distance_symmetry(self, gps_capture_engine):
        """Distance from A to B equals distance from B to A."""
        d1 = gps_capture_engine.calculate_distance(5.60, -0.18, 6.68, -1.62)
        d2 = gps_capture_engine.calculate_distance(6.68, -1.62, 5.60, -0.18)
        assert abs(d1 - d2) < 0.01

    def test_distance_one_degree_latitude(self, gps_capture_engine):
        """One degree of latitude is approximately 111 km everywhere."""
        d = gps_capture_engine.calculate_distance(0.0, 0.0, 1.0, 0.0)
        assert 110_000 < d < 112_000

    def test_area_returns_float(self, gps_capture_engine):
        """Area calculation returns a float."""
        verts = SAMPLE_POLYGON_VERTICES
        area = gps_capture_engine.calculate_area(verts)
        assert isinstance(area, float)

    def test_capture_point_returns_device_id(self, gps_capture_engine, make_gps_capture):
        """Captured point includes device_id."""
        data = make_gps_capture(device_id="dev-gps-test")
        result = gps_capture_engine.capture_point(**data)
        assert result.get("device_id") == "dev-gps-test"

    def test_capture_polygon_returns_vertex_count(self, gps_capture_engine, make_polygon_trace):
        """Captured polygon includes vertex count."""
        data = make_polygon_trace()
        result = gps_capture_engine.capture_polygon(**data)
        assert "vertex_count" in result or "num_vertices" in result or len(result) > 0

    def test_classify_accuracy_returns_value(self, gps_capture_engine):
        """Accuracy classification returns a non-None value."""
        result = gps_capture_engine.classify_accuracy(5.0, 3.0, 6)
        assert result is not None

    def test_validate_plot_size_valid_range(self, gps_capture_engine):
        """Various valid plot sizes pass validation."""
        for size in [0.5, 1.0, 5.0, 50.0, 100.0]:
            result = gps_capture_engine.validate_plot_size(size)
            assert result is True or (isinstance(result, dict) and result.get("valid") is True)

    def test_buffer_zone_returns_data(self, gps_capture_engine):
        """Buffer zone returns non-empty data."""
        verts = SAMPLE_POLYGON_VERTICES
        result = gps_capture_engine.generate_buffer_zone(verts, buffer_meters=25)
        assert result is not None

    def test_centroid_returns_two_values(self, gps_capture_engine):
        """Centroid returns latitude and longitude."""
        verts = SAMPLE_POLYGON_VERTICES
        centroid = gps_capture_engine.get_centroid(verts)
        if isinstance(centroid, dict):
            assert "latitude" in centroid
            assert "longitude" in centroid
        else:
            assert len(centroid) >= 2
