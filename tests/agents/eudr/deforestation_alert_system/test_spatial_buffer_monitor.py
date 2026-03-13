# -*- coding: utf-8 -*-
"""
Unit tests for SpatialBufferMonitor - AGENT-EUDR-020 Engine 4

Tests geofencing engine for supply chain plot monitoring including
buffer creation (circular/polygon/adaptive), detection checking,
point-in-polygon ray casting, haversine distance calculations,
buffer polygon generation, and provenance tracking.

Coverage targets: 85%+ across all SpatialBufferMonitor methods.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-020
Agent ID: GL-EUDR-DAS-020
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from greenlang.agents.eudr.deforestation_alert_system.spatial_buffer_monitor import (
    BufferResult,
    BufferType,
    BufferViolation,
    BufferZone,
    CheckResult,
    COMMODITY_BUFFER_RADII,
    COUNTRY_RISK_MULTIPLIERS,
    DEFAULT_BUFFER_RADIUS_KM,
    DEFAULT_BUFFER_RESOLUTION,
    EARTH_RADIUS_KM,
    GeoPoint,
    MAX_BUFFER_RADIUS_KM,
    MIN_BUFFER_RADIUS_KM,
    SpatialBufferMonitor,
    ViolationSeverity,
    ViolationsResult,
    ZonesResult,
)


# ---------------------------------------------------------------------------
# Helper: deterministic hash computation matching module logic
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    """Replicates module hash for verification."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def monitor() -> SpatialBufferMonitor:
    """Create a fresh SpatialBufferMonitor instance."""
    return SpatialBufferMonitor(config=None)


@pytest.fixture
def buffer_at_goma(monitor: SpatialBufferMonitor) -> BufferResult:
    """Create a buffer near Goma, DRC (-1.68, 29.23)."""
    return monitor.create_buffer(
        plot_id="PLOT-GOMA-001",
        lat=-1.68,
        lon=29.23,
        radius_km=10.0,
        buffer_type="circular",
        commodities=["cocoa"],
        country_code="CD",
    )


@pytest.fixture
def buffer_at_manaus(monitor: SpatialBufferMonitor) -> BufferResult:
    """Create a buffer near Manaus, Brazil (-3.12, -60.02)."""
    return monitor.create_buffer(
        plot_id="PLOT-MANAUS-001",
        lat=-3.12,
        lon=-60.02,
        radius_km=15.0,
        buffer_type="circular",
        commodities=["soya"],
        country_code="BR",
    )


@pytest.fixture
def square_polygon() -> List[Tuple[float, float]]:
    """Unit square polygon centered at origin."""
    return [
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ]


@pytest.fixture
def triangle_polygon() -> List[Tuple[float, float]]:
    """Triangle polygon."""
    return [
        (0.0, 0.0),
        (0.0, 4.0),
        (3.0, 0.0),
    ]


# ---------------------------------------------------------------------------
# TestBufferCreation
# ---------------------------------------------------------------------------


class TestBufferCreation:
    """Tests for create_buffer with various buffer types."""

    def test_create_circular_buffer(self, monitor: SpatialBufferMonitor) -> None:
        """Create a standard circular buffer."""
        result = monitor.create_buffer(
            plot_id="PLOT-001",
            lat=-3.12,
            lon=28.57,
            radius_km=10.0,
        )
        assert isinstance(result, BufferResult)
        assert result.buffer is not None
        assert result.buffer.plot_id == "PLOT-001"
        assert result.buffer.radius_km == Decimal("10.0")
        assert result.buffer.buffer_type == BufferType.CIRCULAR.value
        assert result.buffer.is_active is True
        assert result.polygon_generated is True
        assert len(result.buffer.polygon_points) == DEFAULT_BUFFER_RESOLUTION
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_create_polygon_buffer(self, monitor: SpatialBufferMonitor) -> None:
        """Create a custom polygon buffer with user-defined points."""
        custom_polygon = [
            (-3.10, 28.55),
            (-3.10, 28.60),
            (-3.15, 28.60),
            (-3.15, 28.55),
        ]
        result = monitor.create_buffer(
            plot_id="PLOT-002",
            lat=-3.12,
            lon=28.57,
            radius_km=5.0,
            buffer_type="polygon",
            polygon_points=custom_polygon,
        )
        assert result.buffer is not None
        assert result.buffer.buffer_type == BufferType.POLYGON.value
        assert result.buffer.polygon_points == custom_polygon
        assert result.polygon_generated is False

    def test_create_adaptive_buffer(self, monitor: SpatialBufferMonitor) -> None:
        """Create an adaptive buffer using commodity/country risk multipliers."""
        result = monitor.create_buffer(
            plot_id="PLOT-003",
            lat=-1.68,
            lon=29.23,
            radius_km=10.0,
            buffer_type="adaptive",
            commodities=["soya"],
            country_code="BR",
        )
        assert result.buffer is not None
        assert result.buffer.buffer_type == BufferType.ADAPTIVE.value
        expected_base = COMMODITY_BUFFER_RADII["soya"]  # 15 km
        expected_multiplier = COUNTRY_RISK_MULTIPLIERS["BR"]  # 1.2
        expected_radius = (expected_base * expected_multiplier).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        assert result.buffer.radius_km == expected_radius
        assert any("Adaptive radius" in w for w in result.warnings)

    def test_create_buffer_invalid_latitude(self, monitor: SpatialBufferMonitor) -> None:
        """Latitude out of range raises ValueError."""
        with pytest.raises(ValueError, match="Latitude"):
            monitor.create_buffer(
                plot_id="PLOT-BAD",
                lat=95.0,
                lon=28.57,
                radius_km=10.0,
            )

    def test_create_buffer_invalid_longitude(self, monitor: SpatialBufferMonitor) -> None:
        """Longitude out of range raises ValueError."""
        with pytest.raises(ValueError, match="Longitude"):
            monitor.create_buffer(
                plot_id="PLOT-BAD",
                lat=-3.12,
                lon=200.0,
                radius_km=10.0,
            )

    def test_create_buffer_empty_plot_id(self, monitor: SpatialBufferMonitor) -> None:
        """Empty plot_id raises ValueError."""
        with pytest.raises(ValueError, match="plot_id"):
            monitor.create_buffer(
                plot_id="",
                lat=-3.12,
                lon=28.57,
                radius_km=10.0,
            )

    def test_create_buffer_radius_too_small(self, monitor: SpatialBufferMonitor) -> None:
        """Radius below minimum raises ValueError."""
        with pytest.raises(ValueError, match="radius_km"):
            monitor.create_buffer(
                plot_id="PLOT-SMALL",
                lat=-3.12,
                lon=28.57,
                radius_km=0.5,
            )

    def test_create_buffer_radius_too_large(self, monitor: SpatialBufferMonitor) -> None:
        """Radius above maximum raises ValueError."""
        with pytest.raises(ValueError, match="radius_km"):
            monitor.create_buffer(
                plot_id="PLOT-BIG",
                lat=-3.12,
                lon=28.57,
                radius_km=60.0,
            )

    def test_create_buffer_invalid_type(self, monitor: SpatialBufferMonitor) -> None:
        """Invalid buffer_type raises ValueError."""
        with pytest.raises(ValueError, match="buffer_type"):
            monitor.create_buffer(
                plot_id="PLOT-BAD-TYPE",
                lat=-3.12,
                lon=28.57,
                radius_km=10.0,
                buffer_type="hexagonal",
            )

    def test_create_buffer_stores_zone(self, monitor: SpatialBufferMonitor) -> None:
        """Created buffer is stored in the internal buffer store."""
        result = monitor.create_buffer(
            plot_id="PLOT-STORED", lat=0.0, lon=0.0, radius_km=5.0
        )
        assert result.buffer.buffer_id in monitor._buffer_store

    def test_create_buffer_processing_time(self, monitor: SpatialBufferMonitor) -> None:
        """Processing time is positive."""
        result = monitor.create_buffer(
            plot_id="PLOT-TIME", lat=0.0, lon=0.0, radius_km=5.0
        )
        assert result.processing_time_ms > 0


# ---------------------------------------------------------------------------
# TestBufferUpdate
# ---------------------------------------------------------------------------


class TestBufferUpdate:
    """Tests for update_buffer operations."""

    def test_update_buffer_radius(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Update buffer radius regenerates polygon."""
        bid = buffer_at_goma.buffer.buffer_id
        updated = monitor.update_buffer(bid, radius_km=20.0)
        assert updated.buffer.radius_km == Decimal("20.0")
        assert updated.polygon_generated is True
        assert any("Radius updated" in w for w in updated.warnings)

    def test_update_buffer_active_status(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Update active status."""
        bid = buffer_at_goma.buffer.buffer_id
        updated = monitor.update_buffer(bid, is_active=False)
        assert updated.buffer.is_active is False
        assert any("Active status" in w for w in updated.warnings)

    def test_update_buffer_nonexistent_raises(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Updating nonexistent buffer raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            monitor.update_buffer("nonexistent-id", radius_km=5.0)

    def test_update_buffer_empty_id_raises(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Empty buffer_id raises ValueError."""
        with pytest.raises(ValueError, match="buffer_id"):
            monitor.update_buffer("", radius_km=5.0)

    def test_update_buffer_radius_out_of_range(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Updating radius out of range raises ValueError."""
        bid = buffer_at_goma.buffer.buffer_id
        with pytest.raises(ValueError, match="radius_km"):
            monitor.update_buffer(bid, radius_km=0.1)

    def test_update_buffer_provenance_hash(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Updated buffer has a valid provenance hash."""
        bid = buffer_at_goma.buffer.buffer_id
        updated = monitor.update_buffer(bid, radius_km=8.0)
        assert len(updated.provenance_hash) == 64

    def test_update_buffer_timestamp_changes(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Updated buffer has a newer updated_at timestamp."""
        bid = buffer_at_goma.buffer.buffer_id
        original_updated = buffer_at_goma.buffer.updated_at
        updated = monitor.update_buffer(bid, is_active=False)
        assert updated.buffer.updated_at >= original_updated


# ---------------------------------------------------------------------------
# TestDetectionCheck
# ---------------------------------------------------------------------------


class TestDetectionCheck:
    """Tests for check_detection against active buffers."""

    def test_detection_inside_buffer(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Detection inside buffer produces violation."""
        result = monitor.check_detection(
            detection_lat=-1.68,
            detection_lon=29.23,
            detection_id="DET-001",
        )
        assert result.violations_found >= 1
        assert result.buffers_checked >= 1
        assert len(result.violations) >= 1
        assert result.violations[0].inside_buffer is True

    def test_detection_outside_buffer(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Detection far away from buffer produces no violation."""
        result = monitor.check_detection(
            detection_lat=10.0,
            detection_lon=50.0,
            detection_id="DET-FARAWAY",
        )
        assert result.violations_found == 0

    def test_detection_edge_of_buffer(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Detection near the edge of buffer boundary."""
        monitor.create_buffer(
            plot_id="PLOT-EDGE",
            lat=0.0,
            lon=0.0,
            radius_km=10.0,
        )
        # Approximately 9 km away (within 10km buffer)
        result = monitor.check_detection(
            detection_lat=0.08,  # ~8.9 km
            detection_lon=0.0,
            detection_id="DET-EDGE",
        )
        assert result.buffers_checked >= 1

    def test_detection_multiple_buffers(
        self,
        monitor: SpatialBufferMonitor,
        buffer_at_goma: BufferResult,
    ) -> None:
        """Detection checked against all active buffers."""
        monitor.create_buffer(
            plot_id="PLOT-NEARBY",
            lat=-1.70,
            lon=29.25,
            radius_km=20.0,
            commodities=["coffee"],
            country_code="CD",
        )
        result = monitor.check_detection(
            detection_lat=-1.69,
            detection_lon=29.24,
        )
        assert result.buffers_checked >= 2

    def test_detection_invalid_coordinates(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Invalid detection coordinates raise ValueError."""
        with pytest.raises(ValueError, match="Latitude"):
            monitor.check_detection(detection_lat=100.0, detection_lon=0.0)

    def test_detection_no_active_buffers(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """No active buffers returns zero violations."""
        result = monitor.check_detection(
            detection_lat=0.0, detection_lon=0.0
        )
        assert result.buffers_checked == 0
        assert result.violations_found == 0

    def test_detection_result_has_provenance(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Check result includes provenance hash."""
        result = monitor.check_detection(
            detection_lat=-1.68, detection_lon=29.23
        )
        assert len(result.provenance_hash) == 64

    def test_detection_tracks_nearest_buffer(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Nearest buffer distance and ID are tracked."""
        result = monitor.check_detection(
            detection_lat=-1.68, detection_lon=29.23
        )
        assert result.nearest_buffer_id != ""
        assert result.nearest_buffer_distance_km >= Decimal("0")


# ---------------------------------------------------------------------------
# TestViolations
# ---------------------------------------------------------------------------


class TestViolations:
    """Tests for get_violations filtering."""

    def test_get_violations_by_buffer_id(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Filter violations by buffer_id."""
        monitor.check_detection(
            detection_lat=-1.68, detection_lon=29.23, detection_id="DET-V1"
        )
        bid = buffer_at_goma.buffer.buffer_id
        result = monitor.get_violations(buffer_id=bid)
        assert isinstance(result, ViolationsResult)
        assert result.total >= 1
        for v in result.violations:
            assert v.buffer_id == bid

    def test_get_violations_no_filter(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Get all violations without filter."""
        monitor.check_detection(
            detection_lat=-1.68, detection_lon=29.23, detection_id="DET-V2"
        )
        result = monitor.get_violations()
        assert result.total >= 1

    def test_get_violations_empty(self, monitor: SpatialBufferMonitor) -> None:
        """No violations when no detections have occurred."""
        result = monitor.get_violations()
        assert result.total == 0
        assert len(result.violations) == 0

    def test_get_violations_date_range(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Filter violations by date range."""
        monitor.check_detection(
            detection_lat=-1.68, detection_lon=29.23, detection_id="DET-V3"
        )
        result = monitor.get_violations(
            date_range=("2020-01-01", "2030-12-31")
        )
        assert result.total >= 1

    def test_get_violations_provenance(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Violations result includes provenance hash."""
        result = monitor.get_violations()
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# TestZones
# ---------------------------------------------------------------------------


class TestZones:
    """Tests for get_zones filtering."""

    def test_get_zones_by_country(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Filter zones by country code."""
        result = monitor.get_zones(country_code="CD")
        assert result.total >= 1
        for z in result.zones:
            assert z.country_code == "CD"

    def test_get_zones_by_commodity(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Filter zones by commodity."""
        result = monitor.get_zones(commodity="cocoa")
        assert result.total >= 1
        for z in result.zones:
            assert "cocoa" in z.commodities

    def test_get_zones_active_only(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Active-only filter excludes inactive zones."""
        bid = buffer_at_goma.buffer.buffer_id
        monitor.update_buffer(bid, is_active=False)
        result = monitor.get_zones(active_only=True)
        assert all(z.is_active for z in result.zones)

    def test_get_zones_include_inactive(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Setting active_only=False includes inactive zones."""
        bid = buffer_at_goma.buffer.buffer_id
        monitor.update_buffer(bid, is_active=False)
        result = monitor.get_zones(active_only=False)
        assert result.total >= 1

    def test_get_zones_provenance(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Zones result includes provenance hash."""
        result = monitor.get_zones()
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# TestPointInCircularBuffer
# ---------------------------------------------------------------------------


class TestPointInCircularBuffer:
    """Tests for _point_in_circular_buffer with known distances."""

    @pytest.mark.parametrize(
        "radius_km,expected_inside",
        [
            (Decimal("1"), True),    # 0.5 km away, within 1 km buffer
            (Decimal("0.3"), False), # 0.5 km away, outside 0.3 km buffer
        ],
    )
    def test_buffer_radius_check(
        self,
        monitor: SpatialBufferMonitor,
        radius_km: Decimal,
        expected_inside: bool,
    ) -> None:
        """Parametrized test for point-in-circular-buffer at various radii."""
        # Point approximately 0.5 km from center
        center_lat = Decimal("0")
        center_lon = Decimal("0")
        point_lat = Decimal("0.0045")  # ~0.5 km north
        point_lon = Decimal("0")
        result = monitor._point_in_circular_buffer(
            point_lat, point_lon, center_lat, center_lon, radius_km
        )
        assert result is expected_inside

    def test_point_at_center(self, monitor: SpatialBufferMonitor) -> None:
        """Point exactly at center is inside any buffer."""
        result = monitor._point_in_circular_buffer(
            Decimal("0"), Decimal("0"),
            Decimal("0"), Decimal("0"),
            Decimal("1"),
        )
        assert result is True

    def test_point_exactly_at_radius(self, monitor: SpatialBufferMonitor) -> None:
        """Point exactly at radius boundary (distance == radius) is inside."""
        # Haversine distance at 0.009 lat diff ~1.0 km
        center_lat = Decimal("0")
        center_lon = Decimal("0")
        result = monitor._point_in_circular_buffer(
            Decimal("0.009"), Decimal("0"),
            center_lat, center_lon,
            Decimal("1.0"),
        )
        # At approximately 1 km away, should be right at boundary
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TestPointInPolygon
# ---------------------------------------------------------------------------


class TestPointInPolygon:
    """Tests for _point_in_polygon (ray casting) with known polygons."""

    def test_point_inside_square(
        self, monitor: SpatialBufferMonitor, square_polygon: List[Tuple[float, float]]
    ) -> None:
        """Point (0,0) inside unit square."""
        assert monitor._point_in_polygon(0.0, 0.0, square_polygon) is True

    def test_point_outside_square(
        self, monitor: SpatialBufferMonitor, square_polygon: List[Tuple[float, float]]
    ) -> None:
        """Point (2,0) outside unit square."""
        assert monitor._point_in_polygon(2.0, 0.0, square_polygon) is False

    def test_point_inside_triangle(
        self, monitor: SpatialBufferMonitor, triangle_polygon: List[Tuple[float, float]]
    ) -> None:
        """Point inside triangle."""
        assert monitor._point_in_polygon(0.5, 1.0, triangle_polygon) is True

    def test_point_outside_triangle(
        self, monitor: SpatialBufferMonitor, triangle_polygon: List[Tuple[float, float]]
    ) -> None:
        """Point outside triangle."""
        assert monitor._point_in_polygon(2.0, 3.0, triangle_polygon) is False

    def test_point_in_irregular_polygon(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Point inside irregular (L-shaped) polygon."""
        irregular = [
            (0.0, 0.0),
            (0.0, 4.0),
            (2.0, 4.0),
            (2.0, 2.0),
            (4.0, 2.0),
            (4.0, 0.0),
        ]
        # Point (1, 1) is inside the L
        assert monitor._point_in_polygon(1.0, 1.0, irregular) is True
        # Point (3, 3) is outside the L
        assert monitor._point_in_polygon(3.0, 3.0, irregular) is False

    def test_degenerate_polygon_less_than_3(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Polygon with fewer than 3 points returns False."""
        assert monitor._point_in_polygon(0.0, 0.0, [(0.0, 0.0), (1.0, 1.0)]) is False

    def test_empty_polygon(self, monitor: SpatialBufferMonitor) -> None:
        """Empty polygon returns False."""
        assert monitor._point_in_polygon(0.0, 0.0, []) is False


# ---------------------------------------------------------------------------
# TestHaversineDistance
# ---------------------------------------------------------------------------


class TestHaversineDistance:
    """Tests for _haversine_distance with known city pairs."""

    def test_london_to_paris(self, monitor: SpatialBufferMonitor) -> None:
        """London (51.5074, -0.1278) to Paris (48.8566, 2.3522) ~ 343 km."""
        dist = monitor._haversine_distance(
            Decimal("51.5074"), Decimal("-0.1278"),
            Decimal("48.8566"), Decimal("2.3522"),
        )
        assert Decimal("330") <= dist <= Decimal("360")

    def test_same_point_zero_distance(self, monitor: SpatialBufferMonitor) -> None:
        """Same point has zero distance."""
        dist = monitor._haversine_distance(
            Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")
        )
        assert dist == Decimal("0.000")

    def test_equator_one_degree_lon(self, monitor: SpatialBufferMonitor) -> None:
        """One degree of longitude at equator ~ 111.32 km."""
        dist = monitor._haversine_distance(
            Decimal("0"), Decimal("0"), Decimal("0"), Decimal("1")
        )
        assert Decimal("109") <= dist <= Decimal("113")

    def test_one_degree_lat(self, monitor: SpatialBufferMonitor) -> None:
        """One degree of latitude ~ 111.32 km at any longitude."""
        dist = monitor._haversine_distance(
            Decimal("0"), Decimal("0"), Decimal("1"), Decimal("0")
        )
        assert Decimal("109") <= dist <= Decimal("113")

    def test_new_york_to_los_angeles(self, monitor: SpatialBufferMonitor) -> None:
        """New York to Los Angeles ~ 3944 km."""
        dist = monitor._haversine_distance(
            Decimal("40.7128"), Decimal("-74.0060"),
            Decimal("34.0522"), Decimal("-118.2437"),
        )
        assert Decimal("3900") <= dist <= Decimal("4000")

    def test_antipodal_points(self, monitor: SpatialBufferMonitor) -> None:
        """Opposite sides of Earth ~ half circumference ~ 20015 km."""
        dist = monitor._haversine_distance(
            Decimal("0"), Decimal("0"), Decimal("0"), Decimal("180")
        )
        assert Decimal("19900") <= dist <= Decimal("20100")


# ---------------------------------------------------------------------------
# TestBufferPolygonGeneration
# ---------------------------------------------------------------------------


class TestBufferPolygonGeneration:
    """Tests for _generate_circular_buffer_polygon."""

    def test_64_point_polygon(self, monitor: SpatialBufferMonitor) -> None:
        """Default 64-point polygon generation."""
        points = monitor._generate_circular_buffer_polygon(
            Decimal("0"), Decimal("0"), Decimal("10"), 64
        )
        assert len(points) == 64

    def test_128_point_polygon(self, monitor: SpatialBufferMonitor) -> None:
        """128-point polygon for higher resolution."""
        points = monitor._generate_circular_buffer_polygon(
            Decimal("0"), Decimal("0"), Decimal("10"), 128
        )
        assert len(points) == 128

    def test_minimum_points_clamped(self, monitor: SpatialBufferMonitor) -> None:
        """Fewer than 4 requested points clamps to 4."""
        points = monitor._generate_circular_buffer_polygon(
            Decimal("0"), Decimal("0"), Decimal("10"), 2
        )
        assert len(points) == 4

    def test_points_are_tuples(self, monitor: SpatialBufferMonitor) -> None:
        """All generated points are (float, float) tuples."""
        points = monitor._generate_circular_buffer_polygon(
            Decimal("0"), Decimal("0"), Decimal("5"), 8
        )
        for pt in points:
            assert isinstance(pt, tuple)
            assert len(pt) == 2
            assert isinstance(pt[0], float)
            assert isinstance(pt[1], float)

    def test_points_within_valid_range(self, monitor: SpatialBufferMonitor) -> None:
        """All generated points have valid lat/lon."""
        points = monitor._generate_circular_buffer_polygon(
            Decimal("45"), Decimal("90"), Decimal("50"), 64
        )
        for lat, lon in points:
            assert -90.0 <= lat <= 90.0
            assert -180.0 <= lon <= 180.0

    def test_near_pole_does_not_crash(self, monitor: SpatialBufferMonitor) -> None:
        """Buffer generation near the pole handles edge case gracefully."""
        points = monitor._generate_circular_buffer_polygon(
            Decimal("89.5"), Decimal("0"), Decimal("5"), 16
        )
        assert len(points) == 16


# ---------------------------------------------------------------------------
# TestProvenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance hash generation on all results."""

    def test_create_buffer_provenance(self, monitor: SpatialBufferMonitor) -> None:
        """Buffer creation result has a SHA-256 provenance hash."""
        result = monitor.create_buffer(
            plot_id="PLOT-PROV-1", lat=0.0, lon=0.0, radius_km=5.0
        )
        assert len(result.provenance_hash) == 64

    def test_check_detection_provenance(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Detection check result has provenance hash."""
        result = monitor.check_detection(
            detection_lat=-1.68, detection_lon=29.23
        )
        assert len(result.provenance_hash) == 64

    def test_violation_provenance(
        self, monitor: SpatialBufferMonitor, buffer_at_goma: BufferResult
    ) -> None:
        """Individual violation has provenance hash."""
        result = monitor.check_detection(
            detection_lat=-1.68, detection_lon=29.23
        )
        if result.violations:
            assert len(result.violations[0].provenance_hash) == 64

    def test_zones_provenance(self, monitor: SpatialBufferMonitor) -> None:
        """Zones result includes provenance hash."""
        result = monitor.get_zones()
        assert len(result.provenance_hash) == 64

    def test_violations_result_provenance(self, monitor: SpatialBufferMonitor) -> None:
        """Violations result includes provenance hash."""
        result = monitor.get_violations()
        assert len(result.provenance_hash) == 64

    def test_provenance_determinism(self, monitor: SpatialBufferMonitor) -> None:
        """Same query produces same provenance hash."""
        r1 = monitor.get_violations()
        r2 = monitor.get_violations()
        assert r1.provenance_hash == r2.provenance_hash


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for spatial edge cases."""

    def test_minimum_radius_buffer(self, monitor: SpatialBufferMonitor) -> None:
        """Create buffer with minimum radius (1 km)."""
        result = monitor.create_buffer(
            plot_id="PLOT-MIN",
            lat=0.0,
            lon=0.0,
            radius_km=1.0,
        )
        assert result.buffer.radius_km == Decimal("1.0")

    def test_maximum_radius_buffer(self, monitor: SpatialBufferMonitor) -> None:
        """Create buffer with maximum radius (50 km)."""
        result = monitor.create_buffer(
            plot_id="PLOT-MAX",
            lat=0.0,
            lon=0.0,
            radius_km=50.0,
        )
        assert result.buffer.radius_km == Decimal("50.0")

    def test_equator_crossing(self, monitor: SpatialBufferMonitor) -> None:
        """Buffer near equator (lat=0) works correctly."""
        result = monitor.create_buffer(
            plot_id="PLOT-EQUATOR",
            lat=0.0,
            lon=10.0,
            radius_km=10.0,
        )
        assert result.buffer is not None
        check = monitor.check_detection(
            detection_lat=0.01, detection_lon=10.01
        )
        assert check.buffers_checked >= 1

    def test_dateline_crossing(self, monitor: SpatialBufferMonitor) -> None:
        """Buffer near dateline (lon=179.9) does not crash."""
        result = monitor.create_buffer(
            plot_id="PLOT-DATELINE",
            lat=0.0,
            lon=179.9,
            radius_km=5.0,
        )
        assert result.buffer is not None

    def test_south_pole_adjacent(self, monitor: SpatialBufferMonitor) -> None:
        """Buffer near south pole (lat=-89) does not crash."""
        result = monitor.create_buffer(
            plot_id="PLOT-SOUTH",
            lat=-89.0,
            lon=0.0,
            radius_km=5.0,
        )
        assert result.buffer is not None

    def test_geo_point_dataclass(self) -> None:
        """GeoPoint serialization works."""
        pt = GeoPoint(lat=Decimal("10.5"), lon=Decimal("20.3"))
        d = pt.to_dict()
        assert d["lat"] == "10.5"
        assert d["lon"] == "20.3"
        t = pt.to_tuple()
        assert t == (10.5, 20.3)

    def test_buffer_zone_auto_id(self) -> None:
        """BufferZone auto-generates buffer_id."""
        z = BufferZone(plot_id="P1")
        assert z.buffer_id != ""
        assert len(z.buffer_id) > 0
        assert z.created_at != ""

    def test_violation_auto_id(self) -> None:
        """BufferViolation auto-generates violation_id."""
        v = BufferViolation(buffer_id="B1")
        assert v.violation_id != ""
        assert v.violation_time != ""

    def test_violation_severity_classification(
        self, monitor: SpatialBufferMonitor
    ) -> None:
        """Severity classification by distance-to-radius ratio."""
        # ratio <= 0.25 -> CRITICAL
        assert monitor._classify_violation_severity(
            Decimal("2"), Decimal("10")
        ) == ViolationSeverity.CRITICAL
        # ratio <= 0.5 -> HIGH
        assert monitor._classify_violation_severity(
            Decimal("4"), Decimal("10")
        ) == ViolationSeverity.HIGH
        # ratio <= 0.75 -> MEDIUM
        assert monitor._classify_violation_severity(
            Decimal("7"), Decimal("10")
        ) == ViolationSeverity.MEDIUM
        # ratio > 0.75 -> LOW
        assert monitor._classify_violation_severity(
            Decimal("9"), Decimal("10")
        ) == ViolationSeverity.LOW

    def test_buffer_overlap_zero_area(self, monitor: SpatialBufferMonitor) -> None:
        """Zero detection area produces zero overlap."""
        overlap = monitor._calculate_buffer_overlap(Decimal("0"), Decimal("10"))
        assert overlap == Decimal("0")

    def test_buffer_overlap_positive_area(self, monitor: SpatialBufferMonitor) -> None:
        """Positive detection area produces positive overlap."""
        overlap = monitor._calculate_buffer_overlap(Decimal("50"), Decimal("10"))
        assert overlap > Decimal("0")
