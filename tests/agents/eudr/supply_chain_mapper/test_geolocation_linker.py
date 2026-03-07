# -*- coding: utf-8 -*-
"""
Tests for GeolocationLinker - AGENT-EUDR-001 Feature 3: Plot-Level Geolocation Integration

Comprehensive test suite covering:
- Coordinate validation (WGS84 bounds, precision)
- Polygon validation (EUDR Article 9(1)(d) compliance)
- Producer-to-plot linkage lifecycle
- Spatial index and bounding box queries (<100ms for 100K plots)
- Distance metrics and bearing calculations
- Gap analysis (missing plots, missing polygons, insufficient precision)
- Protected area cross-referencing
- Deforestation alert cross-referencing
- Bulk import (CSV, GeoJSON, Shapefile)
- Map data export (GeoJSON FeatureCollection)
- PostGIS query generation
- Integration with AGENT-DATA-005, 006, 007
- Edge cases and error handling

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 (Feature 3)
"""

import json
import math
import time
import pytest

from greenlang.agents.eudr.supply_chain_mapper.geolocation_linker import (
    GeolocationLinker,
    PostGISQueryBuilder,
    CoordinateValidation,
    PolygonValidation,
    DistanceMetric,
    LinkageStatus,
    GeolocationGapType,
    GeolocationGapSeverity,
    ProtectedAreaType,
    EARTH_RADIUS_M,
    MIN_COORDINATE_PRECISION,
    POLYGON_AREA_THRESHOLD_HA,
    WGS84_SRID,
    EUDR_CUTOFF_DATE,
    SUPPORTED_IMPORT_FORMATS,
    MAX_LOGISTICS_DISTANCE_M,
    _haversine_distance,
    _initial_bearing,
    _count_decimal_places,
    _geodesic_polygon_area,
    _point_in_polygon_ray,
    _SpatialGridIndex,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linker():
    """Create a default GeolocationLinker instance."""
    return GeolocationLinker()


@pytest.fixture
def linker_with_plots(linker):
    """Create a linker pre-loaded with sample plots."""
    # Brazilian cocoa farm (Para, Brazil)
    linker.link_producer_to_plot(
        producer_node_id="PROD-001",
        plot_id="PLOT-BR-001",
        latitude=-2.501234,
        longitude=-44.282345,
        area_hectares=3.5,
        commodity="cocoa",
        country_code="BR",
    )

    # Indonesian palm oil plantation (large plot with polygon)
    polygon_id = [
        [104.750000, -2.900000],
        [104.760000, -2.900000],
        [104.760000, -2.890000],
        [104.750000, -2.890000],
        [104.750000, -2.900000],
    ]
    linker.link_producer_to_plot(
        producer_node_id="PROD-002",
        plot_id="PLOT-ID-001",
        latitude=-2.895000,
        longitude=104.755000,
        polygon_coordinates=polygon_id,
        area_hectares=120.0,
        commodity="oil_palm",
        country_code="ID",
    )

    # Ghanaian cocoa farm
    linker.link_producer_to_plot(
        producer_node_id="PROD-003",
        plot_id="PLOT-GH-001",
        latitude=6.688234,
        longitude=-1.624567,
        area_hectares=2.1,
        commodity="cocoa",
        country_code="GH",
    )

    # Colombian coffee farm
    linker.link_producer_to_plot(
        producer_node_id="PROD-004",
        plot_id="PLOT-CO-001",
        latitude=4.710123,
        longitude=-75.532456,
        area_hectares=5.8,
        polygon_coordinates=[
            [-75.535, 4.708],
            [-75.530, 4.708],
            [-75.530, 4.712],
            [-75.535, 4.712],
            [-75.535, 4.708],
        ],
        commodity="coffee",
        country_code="CO",
    )

    return linker


# ===========================================================================
# 1. COORDINATE VALIDATION
# ===========================================================================


class TestCoordinateValidation:
    """Tests for WGS84 coordinate validation."""

    def test_valid_coordinates(self, linker):
        """Valid coordinates within WGS84 bounds pass validation."""
        result = linker.validate_coordinates(
            latitude=-2.501234,
            longitude=-44.282345,
        )
        assert result.is_valid is True
        assert result.within_bounds is True
        assert result.latitude == -2.501234
        assert result.longitude == -44.282345
        assert len(result.errors) == 0

    def test_precision_6_decimal_places(self, linker):
        """Coordinates with 6 decimal places meet EUDR precision requirement."""
        result = linker.validate_coordinates(
            latitude=51.507351,
            longitude=-0.127758,
        )
        assert result.meets_precision is True
        assert result.precision_lat == 6
        assert result.precision_lon == 6

    def test_insufficient_precision(self, linker):
        """Coordinates with fewer than 6 decimal places trigger warning."""
        result = linker.validate_coordinates(
            latitude=51.51,
            longitude=-0.13,
        )
        assert result.is_valid is True  # Still valid, just a warning
        assert result.meets_precision is False
        assert result.precision_lat < MIN_COORDINATE_PRECISION
        assert len(result.warnings) > 0
        assert "precision" in result.warnings[0].lower()

    def test_latitude_out_of_range_positive(self, linker):
        """Latitude > 90 is invalid."""
        result = linker.validate_coordinates(latitude=91.0, longitude=0.0)
        assert result.is_valid is False
        assert result.within_bounds is False
        assert any("Latitude" in e for e in result.errors)

    def test_latitude_out_of_range_negative(self, linker):
        """Latitude < -90 is invalid."""
        result = linker.validate_coordinates(latitude=-91.5, longitude=0.0)
        assert result.is_valid is False
        assert any("Latitude" in e for e in result.errors)

    def test_longitude_out_of_range_positive(self, linker):
        """Longitude > 180 is invalid."""
        result = linker.validate_coordinates(latitude=0.0, longitude=181.0)
        assert result.is_valid is False
        assert any("Longitude" in e for e in result.errors)

    def test_longitude_out_of_range_negative(self, linker):
        """Longitude < -180 is invalid."""
        result = linker.validate_coordinates(latitude=0.0, longitude=-180.5)
        assert result.is_valid is False
        assert any("Longitude" in e for e in result.errors)

    def test_boundary_values(self, linker):
        """Extreme but valid WGS84 boundary values."""
        for lat, lon in [(90.0, 180.0), (-90.0, -180.0), (0.0, 0.0)]:
            result = linker.validate_coordinates(latitude=lat, longitude=lon)
            assert result.within_bounds is True

    def test_null_island_warning(self, linker):
        """Coordinates at (0,0) trigger Null Island warning."""
        result = linker.validate_coordinates(latitude=0.0, longitude=0.0)
        assert result.is_valid is True
        assert any("Null Island" in w for w in result.warnings)

    def test_high_precision_coordinates(self, linker):
        """Coordinates with 8+ decimal places pass with full precision."""
        result = linker.validate_coordinates(
            latitude=-2.50123456,
            longitude=-44.28234567,
        )
        assert result.is_valid is True
        assert result.meets_precision is True
        assert result.precision_lat >= 6


# ===========================================================================
# 2. POLYGON VALIDATION (EUDR Article 9(1)(d))
# ===========================================================================


class TestPolygonValidation:
    """Tests for polygon validation per EUDR Article 9(1)(d)."""

    def test_small_plot_no_polygon_compliant(self, linker):
        """Plots <= 4 ha do not require polygon - compliant without one."""
        result = linker.validate_polygon(
            polygon_coordinates=None,
            area_hectares=3.5,
        )
        assert result.compliant is True
        assert result.requires_polygon is False
        assert result.has_polygon is False

    def test_large_plot_with_polygon_compliant(self, linker):
        """Plots > 4 ha with valid polygon are compliant."""
        polygon = [
            [-44.285, -2.500],
            [-44.280, -2.500],
            [-44.280, -2.505],
            [-44.285, -2.505],
            [-44.285, -2.500],
        ]
        result = linker.validate_polygon(
            polygon_coordinates=polygon,
            area_hectares=6.0,
        )
        assert result.compliant is True
        assert result.requires_polygon is True
        assert result.has_polygon is True
        assert result.is_valid is True

    def test_large_plot_without_polygon_non_compliant(self, linker):
        """Plots > 4 ha without polygon violate Article 9(1)(d)."""
        result = linker.validate_polygon(
            polygon_coordinates=None,
            area_hectares=5.5,
        )
        assert result.compliant is False
        assert result.requires_polygon is True
        assert result.has_polygon is False
        assert any("Article 9" in e for e in result.errors)

    def test_polygon_with_insufficient_vertices(self, linker):
        """Polygon with < 3 vertices is invalid."""
        result = linker.validate_polygon(
            polygon_coordinates=[[0, 0], [1, 1]],
            area_hectares=5.0,
        )
        assert result.compliant is False
        assert result.has_polygon is False

    def test_polygon_ring_closure(self, linker):
        """Polygon ring must be closed (first == last vertex)."""
        polygon = [
            [-44.285, -2.500],
            [-44.280, -2.500],
            [-44.280, -2.505],
            [-44.285, -2.505],
            [-44.285, -2.500],  # Closed ring
        ]
        result = linker.validate_polygon(
            polygon_coordinates=polygon,
            area_hectares=6.0,
        )
        assert result.is_closed is True

    def test_polygon_vertex_coordinate_validation(self, linker):
        """Invalid coordinates within polygon vertices are detected."""
        polygon = [
            [200.0, -2.500],  # Invalid longitude
            [-44.280, -2.500],
            [-44.280, -2.505],
            [200.0, -2.500],
        ]
        result = linker.validate_polygon(
            polygon_coordinates=polygon,
            area_hectares=6.0,
        )
        assert result.is_valid is False
        assert any("longitude" in e.lower() or "Longitude" in e for e in result.errors)

    def test_exactly_4ha_no_polygon_required(self, linker):
        """At exactly 4 ha, polygon is NOT required (threshold is >4 ha)."""
        result = linker.validate_polygon(
            polygon_coordinates=None,
            area_hectares=4.0,
        )
        assert result.requires_polygon is False
        assert result.compliant is True

    def test_polygon_area_computation(self, linker):
        """Polygon area is computed from ring coordinates."""
        # Approximately 1km x 1km square near equator
        polygon = [
            [0.0, 0.0],
            [0.009, 0.0],
            [0.009, 0.009],
            [0.0, 0.009],
            [0.0, 0.0],
        ]
        result = linker.validate_polygon(
            polygon_coordinates=polygon,
            area_hectares=100.0,
        )
        # Should compute some non-zero area
        assert result.area_hectares > 0


# ===========================================================================
# 3. PRODUCER-TO-PLOT LINKAGE
# ===========================================================================


class TestProducerToPlotLinkage:
    """Tests for the producer-to-plot linkage lifecycle."""

    def test_link_with_valid_data(self, linker):
        """Linking with valid coordinates creates a successful linkage."""
        result = linker.link_producer_to_plot(
            producer_node_id="PROD-A",
            plot_id="PLOT-A",
            latitude=-2.501234,
            longitude=-44.282345,
            area_hectares=3.5,
            commodity="cocoa",
            country_code="BR",
        )
        assert result["link_id"].startswith("GEO-LNK-")
        assert result["status"] == LinkageStatus.LINKED.value
        assert result["producer_node_id"] == "PROD-A"
        assert result["plot_id"] == "PLOT-A"
        assert len(result["validation_errors"]) == 0

    def test_link_with_invalid_coordinates_rejected(self, linker):
        """Linking with invalid coordinates results in REJECTED status."""
        result = linker.link_producer_to_plot(
            producer_node_id="PROD-B",
            plot_id="PLOT-B",
            latitude=100.0,  # Invalid
            longitude=-44.282345,
        )
        assert result["status"] == LinkageStatus.REJECTED.value
        assert len(result["validation_errors"]) > 0

    def test_link_without_coordinates_pending(self, linker):
        """Linking without coordinates creates PENDING_VALIDATION status."""
        result = linker.link_producer_to_plot(
            producer_node_id="PROD-C",
            plot_id="PLOT-C",
        )
        assert result["status"] == LinkageStatus.PENDING_VALIDATION.value

    def test_link_large_plot_without_polygon_rejected(self, linker):
        """Linking a >4ha plot without polygon is rejected."""
        result = linker.link_producer_to_plot(
            producer_node_id="PROD-D",
            plot_id="PLOT-D",
            latitude=-2.501234,
            longitude=-44.282345,
            area_hectares=6.0,
            polygon_coordinates=None,
        )
        assert result["status"] == LinkageStatus.REJECTED.value
        assert any("Article 9" in e for e in result["validation_errors"])

    def test_empty_producer_id_raises(self, linker):
        """Empty producer_node_id raises ValueError."""
        with pytest.raises(ValueError, match="producer_node_id"):
            linker.link_producer_to_plot(
                producer_node_id="",
                plot_id="PLOT-E",
            )

    def test_empty_plot_id_raises(self, linker):
        """Empty plot_id raises ValueError."""
        with pytest.raises(ValueError, match="plot_id"):
            linker.link_producer_to_plot(
                producer_node_id="PROD-E",
                plot_id="",
            )

    def test_unlink_producer_from_plot(self, linker):
        """Unlinking removes the linkage and spatial index entry."""
        linker.link_producer_to_plot(
            producer_node_id="PROD-F",
            plot_id="PLOT-F",
            latitude=5.123456,
            longitude=-1.234567,
        )
        assert linker.link_count == 1
        assert linker.plot_count == 1

        result = linker.unlink_producer_from_plot("PROD-F", "PLOT-F")
        assert result["removed_count"] == 1
        assert linker.link_count == 0
        assert linker.plot_count == 0

    def test_get_links_for_producer(self, linker_with_plots):
        """Retrieve all plot links for a specific producer."""
        links = linker_with_plots.get_links_for_producer("PROD-001")
        assert len(links) == 1
        assert links[0]["plot_id"] == "PLOT-BR-001"

    def test_get_links_for_plot(self, linker_with_plots):
        """Retrieve all producer links for a specific plot."""
        links = linker_with_plots.get_links_for_plot("PLOT-GH-001")
        assert len(links) == 1
        assert links[0]["producer_node_id"] == "PROD-003"

    def test_get_link_by_id(self, linker):
        """Retrieve a specific link by its ID."""
        result = linker.link_producer_to_plot(
            producer_node_id="PROD-G",
            plot_id="PLOT-G",
            latitude=10.123456,
            longitude=20.654321,
        )
        link = linker.get_link(result["link_id"])
        assert link is not None
        assert link["link_id"] == result["link_id"]

    def test_multiple_plots_per_producer(self, linker):
        """A producer can be linked to multiple plots."""
        linker.link_producer_to_plot(
            producer_node_id="PROD-MULTI",
            plot_id="PLOT-M1",
            latitude=1.123456, longitude=1.654321,
        )
        linker.link_producer_to_plot(
            producer_node_id="PROD-MULTI",
            plot_id="PLOT-M2",
            latitude=1.234567, longitude=1.765432,
        )
        links = linker.get_links_for_producer("PROD-MULTI")
        assert len(links) == 2

    def test_coordinate_validation_included(self, linker):
        """Link result includes coordinate validation details."""
        result = linker.link_producer_to_plot(
            producer_node_id="PROD-CV",
            plot_id="PLOT-CV",
            latitude=51.507351,
            longitude=-0.127758,
        )
        cv = result["coordinate_validation"]
        assert cv is not None
        assert cv["is_valid"] is True
        assert cv["meets_precision"] is True

    def test_polygon_validation_included(self, linker):
        """Link result includes polygon validation details."""
        result = linker.link_producer_to_plot(
            producer_node_id="PROD-PV",
            plot_id="PLOT-PV",
            latitude=1.123456,
            longitude=1.654321,
            area_hectares=5.0,
            polygon_coordinates=[
                [1.65, 1.12], [1.66, 1.12], [1.66, 1.13],
                [1.65, 1.13], [1.65, 1.12],
            ],
        )
        pv = result["polygon_validation"]
        assert pv is not None
        assert pv["requires_polygon"] is True
        assert pv["compliant"] is True


# ===========================================================================
# 4. SPATIAL INDEX AND BOUNDING BOX QUERIES
# ===========================================================================


class TestSpatialIndex:
    """Tests for the spatial grid index and bounding box queries."""

    def test_insert_and_query(self):
        """Basic insert and bounding box query."""
        idx = _SpatialGridIndex()
        idx.insert("P1", 10.0, 20.0)
        idx.insert("P2", 10.5, 20.5)
        idx.insert("P3", 50.0, 50.0)  # Outside bbox

        results = idx.query_bbox(9.0, 19.0, 11.0, 21.0)
        assert "P1" in results
        assert "P2" in results
        assert "P3" not in results

    def test_remove_from_index(self):
        """Removing a plot from the index excludes it from queries."""
        idx = _SpatialGridIndex()
        idx.insert("P1", 10.0, 20.0)
        assert idx.count == 1

        idx.remove("P1")
        assert idx.count == 0

        results = idx.query_bbox(9.0, 19.0, 11.0, 21.0)
        assert len(results) == 0

    def test_radius_query(self):
        """Radius query returns plots within distance, sorted."""
        idx = _SpatialGridIndex()
        idx.insert("P_CLOSE", 0.001, 0.001)
        idx.insert("P_FAR", 0.1, 0.1)
        idx.insert("P_OUTSIDE", 5.0, 5.0)

        results = idx.query_radius(0.0, 0.0, 50_000)  # 50km
        plot_ids = [r[0] for r in results]
        assert "P_CLOSE" in plot_ids
        assert "P_FAR" in plot_ids
        assert "P_OUTSIDE" not in plot_ids

        # P_CLOSE should be first (closer)
        if len(results) >= 2:
            assert results[0][1] < results[1][1]

    def test_find_plots_in_bbox(self, linker_with_plots):
        """Find plots within a bounding box returns matching results."""
        result = linker_with_plots.find_plots_in_bbox(
            min_lon=-45.0, min_lat=-3.0,
            max_lon=-44.0, max_lat=-2.0,
        )
        assert result["matching_count"] == 1
        assert result["matching_plots"][0]["plot_id"] == "PLOT-BR-001"
        assert result["postgis_query"] is not None
        assert result["elapsed_ms"] >= 0

    def test_find_plots_in_bbox_no_results(self, linker_with_plots):
        """BBox query with no matching plots returns empty list."""
        result = linker_with_plots.find_plots_in_bbox(
            min_lon=170.0, min_lat=80.0,
            max_lon=180.0, max_lat=90.0,
        )
        assert result["matching_count"] == 0
        assert len(result["matching_plots"]) == 0

    def test_find_plots_in_radius(self, linker_with_plots):
        """Radius query returns plots within distance, sorted."""
        result = linker_with_plots.find_plots_in_radius(
            longitude=-44.282345,
            latitude=-2.501234,
            radius_m=1000.0,  # 1km
        )
        assert result["matching_count"] >= 1
        assert result["matching_plots"][0]["plot_id"] == "PLOT-BR-001"
        assert "distance_metres" in result["matching_plots"][0]

    def test_performance_100k_plots(self, linker):
        """Bounding box lookup < 100ms for 100,000 plots."""
        # Insert 100,000 plots distributed globally
        # Use modular arithmetic to spread across lat/lon space
        for i in range(100_000):
            # Spread lat from -80 to +80 (avoiding poles)
            lat = ((i * 7) % 16000) / 100.0 - 80.0
            # Spread lon from -170 to +170
            lon = ((i * 13) % 34000) / 100.0 - 170.0
            linker._spatial_index.insert(f"PERF-{i}", lon, lat)

        assert linker._spatial_index.count == 100_000

        # Time the bounding box query (20x20 degree box near equator)
        start = time.monotonic()
        results = linker._spatial_index.query_bbox(-10.0, -10.0, 10.0, 10.0)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 100.0, f"BBox query took {elapsed_ms:.1f}ms, target <100ms"
        assert len(results) > 0, (
            f"Expected some plots in [-10,-10,10,10] bbox out of 100K distributed plots"
        )


# ===========================================================================
# 5. DISTANCE METRICS
# ===========================================================================


class TestDistanceMetrics:
    """Tests for geodesic distance and bearing calculations."""

    def test_haversine_distance_known_values(self):
        """Haversine distance matches known geographic distances."""
        # London (Big Ben) to Paris (Eiffel Tower): approximately 340km
        dist = _haversine_distance(-0.1276, 51.5074, 2.2945, 48.8584)
        assert 340_000 < dist < 345_000  # 340-345km

    def test_haversine_same_point_zero_distance(self):
        """Distance between same point is 0."""
        dist = _haversine_distance(10.0, 20.0, 10.0, 20.0)
        assert dist == 0.0

    def test_haversine_antipodal_points(self):
        """Distance between antipodal points is approximately half circumference."""
        dist = _haversine_distance(0.0, 0.0, 180.0, 0.0)
        expected = math.pi * EARTH_RADIUS_M
        assert abs(dist - expected) < 100  # Within 100m

    def test_initial_bearing(self):
        """Initial bearing is calculated correctly."""
        # Due north from equator
        bearing = _initial_bearing(0.0, 0.0, 0.0, 1.0)
        assert abs(bearing - 0.0) < 0.1

        # Due east from equator
        bearing = _initial_bearing(0.0, 0.0, 1.0, 0.0)
        assert abs(bearing - 90.0) < 0.1

    def test_calculate_distance_method(self, linker):
        """GeolocationLinker.calculate_distance returns DistanceMetric."""
        result = linker.calculate_distance(
            from_lon=-0.1276, from_lat=51.5074,
            to_lon=2.2945, to_lat=48.8584,
        )
        assert isinstance(result, DistanceMetric)
        assert 340_000 < result.distance_metres < 345_000
        assert 340 < result.distance_km < 345
        assert 0 <= result.bearing_degrees < 360
        assert result.within_logistics_range is True  # <500km

    def test_logistics_range_exceeded(self, linker):
        """Distances exceeding max logistics range are flagged."""
        # London to New York: approximately 5,500km
        result = linker.calculate_distance(
            from_lon=-0.1276, from_lat=51.5074,
            to_lon=-74.006, to_lat=40.7128,
        )
        assert result.within_logistics_range is False

    def test_calculate_node_distances(self, linker):
        """Node distance matrix calculation produces correct results."""
        nodes = [
            {"node_id": "N1", "longitude": 0.0, "latitude": 0.0},
            {"node_id": "N2", "longitude": 1.0, "latitude": 0.0},
            {"node_id": "N3", "longitude": 0.0, "latitude": 1.0},
        ]
        result = linker.calculate_node_distances(nodes)
        assert result["node_count"] == 3
        assert result["pair_count"] == 3
        assert "N1" in result["distance_matrix_km"]
        assert "N2" in result["distance_matrix_km"]["N1"]
        assert result["summary"]["max_distance_km"] > 0
        assert result["summary"]["min_distance_km"] > 0


# ===========================================================================
# 6. GAP ANALYSIS (Missing Plots)
# ===========================================================================


class TestGapAnalysis:
    """Tests for geolocation gap detection and compliance flagging."""

    def test_flag_missing_geolocation(self, linker_with_plots):
        """Producers without plots are flagged as missing geolocation."""
        result = linker_with_plots.flag_missing_geolocation(
            producer_node_ids=["PROD-001", "PROD-002", "PROD-UNKNOWN"],
        )
        assert result["total_producers_checked"] == 3
        assert result["linked_producers"] == 2
        assert result["unlinked_producers"] == 1
        assert result["total_gaps"] >= 1
        assert result["compliance_rate"] < 100.0

        # Check the flagged producer
        missing = result["missing_plot_gaps"]
        assert len(missing) == 1
        assert missing[0]["producer_node_id"] == "PROD-UNKNOWN"
        assert missing[0]["gap_type"] == GeolocationGapType.MISSING_PLOT.value
        assert missing[0]["severity"] == GeolocationGapSeverity.CRITICAL.value

    def test_all_producers_linked(self, linker_with_plots):
        """No gaps when all producers have plots."""
        result = linker_with_plots.flag_missing_geolocation(
            producer_node_ids=["PROD-001", "PROD-002"],
        )
        assert result["unlinked_producers"] == 0
        assert result["compliance_rate"] == 100.0

    def test_gap_includes_remediation(self, linker_with_plots):
        """Gap records include remediation guidance."""
        result = linker_with_plots.flag_missing_geolocation(
            producer_node_ids=["PROD-MISSING"],
        )
        gap = result["missing_plot_gaps"][0]
        assert "remediation" in gap
        assert "register" in gap["remediation"].lower() or "Register" in gap["remediation"]

    def test_get_gaps_for_producer(self, linker_with_plots):
        """Retrieve gaps for a specific producer."""
        linker_with_plots.flag_missing_geolocation(
            producer_node_ids=["PROD-NOGEO"],
        )
        gaps = linker_with_plots.get_gaps_for_producer("PROD-NOGEO")
        assert len(gaps) == 1
        assert gaps[0]["gap_type"] == GeolocationGapType.MISSING_PLOT.value

    def test_precision_gap_detection(self, linker):
        """Low-precision coordinates generate precision gap."""
        linker.link_producer_to_plot(
            producer_node_id="PROD-LP",
            plot_id="PLOT-LP",
            latitude=5.12,  # Only 2 decimal places
            longitude=-1.23,
        )
        result = linker.flag_missing_geolocation(
            producer_node_ids=["PROD-LP"],
        )
        assert result["precision_gaps"] >= 1


# ===========================================================================
# 7. PROTECTED AREA CROSS-REFERENCE
# ===========================================================================


class TestProtectedAreaCrossReference:
    """Tests for protected area overlap checking."""

    def test_no_overlap(self, linker_with_plots):
        """Plot outside protected areas returns no overlap."""
        result = linker_with_plots.check_protected_area_overlap(
            plot_id="PLOT-BR-001",
            protected_areas=[],
        )
        assert result["overlap_count"] == 0
        assert result["status"] == "no_overlap"

    def test_overlap_detected(self, linker_with_plots):
        """Plot inside a protected area polygon is detected."""
        # Create a protected area polygon that contains the Brazilian plot
        protected_area = {
            "area_id": "PA-001",
            "name": "Amazon Reserve",
            "area_type": ProtectedAreaType.NATURE_RESERVE.value,
            "polygon": [
                [-45.0, -3.0],
                [-44.0, -3.0],
                [-44.0, -2.0],
                [-45.0, -2.0],
                [-45.0, -3.0],
            ],
        }
        result = linker_with_plots.check_protected_area_overlap(
            plot_id="PLOT-BR-001",
            protected_areas=[protected_area],
        )
        assert result["overlap_count"] == 1
        assert result["status"] == "overlap_detected"
        assert result["overlaps"][0]["name"] == "Amazon Reserve"

    def test_plot_not_found(self, linker):
        """Missing plot_id returns plot_not_found status."""
        result = linker.check_protected_area_overlap(
            plot_id="NONEXISTENT",
        )
        assert result["status"] == "plot_not_found"

    def test_postgis_query_included(self, linker_with_plots):
        """Result includes PostGIS query for database-backed checks."""
        result = linker_with_plots.check_protected_area_overlap(
            plot_id="PLOT-BR-001",
        )
        assert "postgis_query" in result
        assert "protected_areas" in result["postgis_query"]


# ===========================================================================
# 8. DEFORESTATION ALERT CROSS-REFERENCE (AGENT-DATA-007)
# ===========================================================================


class TestDeforestationAlertCrossReference:
    """Tests for deforestation alert checking against AGENT-DATA-007."""

    def test_no_alerts(self, linker_with_plots):
        """Plot with no nearby alerts returns clear status."""
        result = linker_with_plots.check_deforestation_alerts(
            plot_id="PLOT-BR-001",
            alerts=[],
        )
        assert result["alerts_found"] == 0
        assert result["status"] == "clear"
        assert result["risk_level"] == "none"

    def test_pre_cutoff_alerts_only(self, linker_with_plots):
        """Pre-cutoff alerts are flagged but with low risk."""
        alert = {
            "alert_id": "ALT-001",
            "latitude": -2.502000,
            "longitude": -44.283000,
            "detection_date": "2019-06-15",
            "is_post_cutoff": False,
            "area_ha": 0.5,
        }
        result = linker_with_plots.check_deforestation_alerts(
            plot_id="PLOT-BR-001",
            alerts=[alert],
        )
        assert result["alerts_found"] == 1
        assert result["post_cutoff_alerts"] == 0
        assert result["status"] == "pre_cutoff_alerts_only"
        assert result["risk_level"] == "low"

    def test_post_cutoff_alerts_high_risk(self, linker_with_plots):
        """Post-EUDR-cutoff alerts trigger high risk assessment."""
        alerts = [
            {
                "alert_id": f"ALT-{i}",
                "latitude": -2.501 + i * 0.001,
                "longitude": -44.282 + i * 0.001,
                "detection_date": f"2021-0{i+1}-15",
                "is_post_cutoff": True,
                "area_ha": 0.3,
            }
            for i in range(3)
        ]
        result = linker_with_plots.check_deforestation_alerts(
            plot_id="PLOT-BR-001",
            alerts=alerts,
        )
        assert result["alerts_found"] == 3
        assert result["post_cutoff_alerts"] == 3
        assert result["status"] == "deforestation_risk_detected"
        assert result["risk_level"] == "high"

    def test_distant_alerts_not_matched(self, linker_with_plots):
        """Alerts beyond 5km radius are not matched to the plot."""
        alert = {
            "alert_id": "ALT-FAR",
            "latitude": -2.600000,  # Far away
            "longitude": -44.400000,
            "detection_date": "2022-01-15",
            "is_post_cutoff": True,
            "area_ha": 1.0,
        }
        result = linker_with_plots.check_deforestation_alerts(
            plot_id="PLOT-BR-001",
            alerts=[alert],
        )
        assert result["alerts_found"] == 0

    def test_plot_not_found(self, linker):
        """Missing plot returns plot_not_found status."""
        result = linker.check_deforestation_alerts(
            plot_id="NONEXISTENT",
        )
        assert result["status"] == "plot_not_found"


# ===========================================================================
# 9. BULK IMPORT (CSV, GeoJSON, Shapefile)
# ===========================================================================


class TestBulkImport:
    """Tests for bulk import from CSV, GeoJSON, and Shapefile."""

    def test_csv_import(self, linker):
        """CSV import parses and links plots correctly."""
        csv_data = (
            "latitude,longitude,plot_id,commodity,country_code,area_hectares\n"
            "-2.501234,-44.282345,CSV-PLOT-001,cocoa,BR,3.5\n"
            "6.688234,-1.624567,CSV-PLOT-002,cocoa,GH,2.1\n"
        )
        result = linker.bulk_import_plots(
            data=csv_data,
            format_type="csv",
            producer_node_id="BULK-PROD-001",
        )
        assert result["total_records"] == 2
        assert result["success"] == 2
        assert result["failed"] == 0
        assert linker.plot_count == 2

    def test_csv_import_missing_coordinates(self, linker):
        """CSV records without coordinates are skipped."""
        csv_data = (
            "latitude,longitude,plot_id\n"
            ",,-NO-COORDS\n"
            "5.123456,10.654321,HAS-COORDS\n"
        )
        result = linker.bulk_import_plots(
            data=csv_data,
            format_type="csv",
        )
        # Only records with lat/lon are parsed
        assert result["total_records"] == 1
        assert result["success"] == 1

    def test_geojson_import_points(self, linker):
        """GeoJSON FeatureCollection with Point geometries."""
        geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-44.282345, -2.501234],
                    },
                    "properties": {
                        "plot_id": "GJ-PLOT-001",
                        "commodity": "cocoa",
                        "country_code": "BR",
                    },
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-1.624567, 6.688234],
                    },
                    "properties": {
                        "plot_id": "GJ-PLOT-002",
                        "commodity": "cocoa",
                        "country_code": "GH",
                    },
                },
            ],
        })
        result = linker.bulk_import_plots(
            data=geojson,
            format_type="geojson",
            producer_node_id="GJ-PROD",
        )
        assert result["total_records"] == 2
        assert result["success"] == 2

    def test_geojson_import_polygons(self, linker):
        """GeoJSON with Polygon geometries extracts centroid and ring."""
        geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-44.285, -2.500],
                            [-44.280, -2.500],
                            [-44.280, -2.505],
                            [-44.285, -2.505],
                            [-44.285, -2.500],
                        ]],
                    },
                    "properties": {
                        "plot_id": "GJ-POLY-001",
                        "commodity": "cocoa",
                    },
                },
            ],
        })
        result = linker.bulk_import_plots(
            data=geojson,
            format_type="geojson",
            producer_node_id="GJ-POLY-PROD",
        )
        assert result["total_records"] == 1
        assert result["success"] == 1

    def test_shapefile_import(self, linker):
        """Shapefile JSON import (pre-converted via ogr2ogr)."""
        shapefile_json = json.dumps([
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [20.123456, 10.654321],
                },
                "properties": {
                    "plot_id": "SHP-001",
                    "commodity": "coffee",
                },
            },
        ])
        result = linker.bulk_import_plots(
            data=shapefile_json,
            format_type="shapefile",
            producer_node_id="SHP-PROD",
        )
        assert result["total_records"] == 1
        assert result["success"] == 1

    def test_unsupported_format_raises(self, linker):
        """Unsupported format type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            linker.bulk_import_plots(data="", format_type="xlsx")

    def test_csv_with_polygon_json(self, linker):
        """CSV with polygon_coordinates as JSON string."""
        poly_json = json.dumps([
            [1.65, 1.12], [1.66, 1.12], [1.66, 1.13],
            [1.65, 1.13], [1.65, 1.12],
        ])
        csv_data = (
            f"latitude,longitude,plot_id,area_hectares,polygon_coordinates\n"
            f"1.123456,1.654321,CSV-POLY-001,5.5,\"{poly_json}\"\n"
        )
        result = linker.bulk_import_plots(
            data=csv_data,
            format_type="csv",
            producer_node_id="CSV-POLY-PROD",
        )
        assert result["total_records"] == 1

    def test_invalid_geojson_returns_empty(self, linker):
        """Invalid GeoJSON data returns zero records."""
        result = linker.bulk_import_plots(
            data="not valid json {{{",
            format_type="geojson",
        )
        assert result["total_records"] == 0
        assert result["success"] == 0


# ===========================================================================
# 10. MAP DATA EXPORT (Interactive Visualization)
# ===========================================================================


class TestMapData:
    """Tests for GeoJSON map data export for visualization."""

    def test_get_all_map_data(self, linker_with_plots):
        """Get map data without filters returns all plots."""
        data = linker_with_plots.get_map_data()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 4
        assert data["metadata"]["total_features"] == 4

    def test_map_data_with_bbox_filter(self, linker_with_plots):
        """BBox filter limits map features to spatial extent."""
        data = linker_with_plots.get_map_data(
            min_lon=-45.0, min_lat=-3.0,
            max_lon=-44.0, max_lat=-2.0,
        )
        assert len(data["features"]) == 1
        assert data["features"][0]["properties"]["plot_id"] == "PLOT-BR-001"

    def test_map_data_with_commodity_filter(self, linker_with_plots):
        """Commodity filter returns only matching plots."""
        data = linker_with_plots.get_map_data(commodity_filter="cocoa")
        assert len(data["features"]) == 2
        for feat in data["features"]:
            assert feat["properties"]["commodity"] == "cocoa"

    def test_map_data_with_country_filter(self, linker_with_plots):
        """Country filter returns only matching plots."""
        data = linker_with_plots.get_map_data(country_filter="ID")
        assert len(data["features"]) == 1
        assert data["features"][0]["properties"]["country_code"] == "ID"

    def test_map_data_point_geometry(self, linker_with_plots):
        """Plots without polygon return Point geometry."""
        data = linker_with_plots.get_map_data(country_filter="BR")
        geom = data["features"][0]["geometry"]
        assert geom["type"] == "Point"
        assert len(geom["coordinates"]) == 2

    def test_map_data_polygon_geometry(self, linker_with_plots):
        """Plots with polygon return Polygon geometry."""
        data = linker_with_plots.get_map_data(country_filter="ID")
        geom = data["features"][0]["geometry"]
        assert geom["type"] == "Polygon"
        assert len(geom["coordinates"][0]) == 5

    def test_map_data_feature_properties(self, linker_with_plots):
        """Map features include required properties for visualization."""
        data = linker_with_plots.get_map_data()
        props = data["features"][0]["properties"]
        assert "plot_id" in props
        assert "producer_node_id" in props
        assert "commodity" in props
        assert "country_code" in props
        assert "linkage_status" in props
        assert "has_compliance_gaps" in props


# ===========================================================================
# 11. POSTGIS QUERY BUILDER
# ===========================================================================


class TestPostGISQueryBuilder:
    """Tests for PostGIS parameterized SQL query generation."""

    def test_bbox_query(self):
        """BBox query generates valid parameterized SQL."""
        builder = PostGISQueryBuilder()
        sql, params = builder.bbox_query()
        assert "ST_MakeEnvelope" in sql
        assert "%(min_lon)s" in sql
        assert "min_lon" in params
        assert "min_lat" in params
        assert "max_lon" in params
        assert "max_lat" in params

    def test_radius_query(self):
        """Radius query uses ST_DWithin."""
        builder = PostGISQueryBuilder()
        sql, params = builder.radius_query()
        assert "ST_DWithin" in sql
        assert "%(radius_m)s" in sql
        assert "radius_m" in params

    def test_contains_point_query(self):
        """Contains query uses ST_Contains."""
        builder = PostGISQueryBuilder()
        sql, params = builder.contains_point_query()
        assert "ST_Contains" in sql
        assert "lon" in params
        assert "lat" in params

    def test_protected_area_intersection_query(self):
        """Protected area query uses ST_Intersects."""
        builder = PostGISQueryBuilder()
        sql, params = builder.protected_area_intersection_query()
        assert "ST_Intersects" in sql
        assert "protected_areas" in sql
        assert "plot_id" in params

    def test_insert_plot_geometry(self):
        """Insert query uses ST_GeomFromGeoJSON."""
        builder = PostGISQueryBuilder()
        sql, params = builder.insert_plot_geometry()
        assert "INSERT INTO" in sql
        assert "ST_GeomFromGeoJSON" in sql
        assert "plot_id" in params
        assert "geojson" in params

    def test_create_spatial_index(self):
        """Spatial index DDL uses GIST."""
        builder = PostGISQueryBuilder()
        sql = builder.create_spatial_index()
        assert "GIST" in sql
        assert "CREATE INDEX" in sql

    def test_custom_table_and_column(self):
        """Custom table and column names are used in queries."""
        builder = PostGISQueryBuilder(
            table="my_plots",
            geom_column="geom",
        )
        sql, _ = builder.bbox_query()
        assert "my_plots" in sql
        assert "geom" in sql

    def test_custom_srid(self):
        """Custom SRID is applied to spatial queries."""
        builder = PostGISQueryBuilder(srid=3857)
        sql, _ = builder.bbox_query()
        assert "3857" in sql


# ===========================================================================
# 12. SPATIAL ANALYSIS (AGENT-DATA-006 Integration)
# ===========================================================================


class TestSpatialAnalysis:
    """Tests for spatial analysis between plots."""

    def test_spatial_analysis_between_plots(self, linker_with_plots):
        """Spatial analysis calculates distance between two plots."""
        result = linker_with_plots.spatial_analysis(
            plot_id_a="PLOT-BR-001",
            plot_id_b="PLOT-GH-001",
        )
        assert result["operation"] == "spatial_analysis"
        assert result["distance"]["km"] > 0

    def test_spatial_analysis_plot_not_found(self, linker):
        """Missing plot returns error status."""
        result = linker.spatial_analysis("MISSING-A", "MISSING-B")
        assert result["status"] == "plot_not_found"
        assert len(result["missing"]) == 2


# ===========================================================================
# 13. CORE SPATIAL FUNCTIONS
# ===========================================================================


class TestCoreSpatialFunctions:
    """Tests for exposed spatial utility functions."""

    def test_count_decimal_places(self):
        """Decimal place counting is accurate."""
        assert _count_decimal_places(1.123456) == 6
        assert _count_decimal_places(1.12) == 2
        assert _count_decimal_places(1.0) == 0
        assert _count_decimal_places(1) == 0
        assert _count_decimal_places(1.1234567890) >= 7

    def test_geodesic_polygon_area(self):
        """Geodesic area calculation produces reasonable values."""
        # 1-degree square at equator should be approximately 12,300 km2
        ring = [
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
            [0.0, 1.0], [0.0, 0.0],
        ]
        area_m2 = _geodesic_polygon_area(ring)
        area_km2 = area_m2 / 1_000_000.0
        assert 12_000 < area_km2 < 13_000

    def test_geodesic_polygon_area_empty(self):
        """Empty ring returns 0 area."""
        assert _geodesic_polygon_area([]) == 0.0
        assert _geodesic_polygon_area([[0, 0]]) == 0.0

    def test_point_in_polygon_ray(self):
        """Point-in-polygon ray casting is correct."""
        polygon = [
            [0.0, 0.0], [10.0, 0.0], [10.0, 10.0],
            [0.0, 10.0], [0.0, 0.0],
        ]
        # Inside
        assert _point_in_polygon_ray([5.0, 5.0], polygon) is True
        # Outside
        assert _point_in_polygon_ray([15.0, 5.0], polygon) is False
        # On edge (implementation-dependent, but should not crash)
        _point_in_polygon_ray([0.0, 5.0], polygon)


# ===========================================================================
# 14. STATISTICS AND DIAGNOSTICS
# ===========================================================================


class TestStatistics:
    """Tests for statistics and diagnostic properties."""

    def test_empty_linker_statistics(self, linker):
        """Empty linker has zero counts."""
        assert linker.link_count == 0
        assert linker.plot_count == 0
        assert linker.gap_count == 0

    def test_statistics_after_linking(self, linker_with_plots):
        """Statistics reflect linked plots."""
        assert linker_with_plots.link_count == 4
        assert linker_with_plots.plot_count == 4

    def test_get_statistics(self, linker_with_plots):
        """get_statistics returns comprehensive data."""
        stats = linker_with_plots.get_statistics()
        assert stats["total_links"] == 4
        assert stats["total_plots_indexed"] == 4
        assert "link_status_distribution" in stats
        assert "commodity_distribution" in stats
        assert "country_distribution" in stats
        assert stats["unique_producers_linked"] == 4
        assert stats["unique_plots_linked"] == 4


# ===========================================================================
# 15. ENUMERATIONS AND CONSTANTS
# ===========================================================================


class TestEnumerationsAndConstants:
    """Tests for enum values and module constants."""

    def test_linkage_status_values(self):
        """LinkageStatus has all expected values."""
        assert LinkageStatus.LINKED.value == "linked"
        assert LinkageStatus.UNLINKED.value == "unlinked"
        assert LinkageStatus.VALIDATED.value == "validated"
        assert LinkageStatus.REJECTED.value == "rejected"

    def test_gap_type_values(self):
        """GeolocationGapType has all expected values."""
        assert GeolocationGapType.MISSING_PLOT.value == "missing_plot"
        assert GeolocationGapType.MISSING_POLYGON.value == "missing_polygon"
        assert GeolocationGapType.DEFORESTATION_ALERT.value == "deforestation_alert"

    def test_constants(self):
        """Module constants have correct values."""
        assert EARTH_RADIUS_M == 6_371_000.0
        assert MIN_COORDINATE_PRECISION == 6
        assert POLYGON_AREA_THRESHOLD_HA == 4.0
        assert WGS84_SRID == 4326
        assert EUDR_CUTOFF_DATE == "2020-12-31"
        assert "csv" in SUPPORTED_IMPORT_FORMATS
        assert "geojson" in SUPPORTED_IMPORT_FORMATS
        assert "shapefile" in SUPPORTED_IMPORT_FORMATS

    def test_protected_area_types(self):
        """ProtectedAreaType enum has IUCN categories."""
        assert ProtectedAreaType.NATIONAL_PARK.value == "national_park"
        assert ProtectedAreaType.INDIGENOUS_TERRITORY.value == "indigenous_territory"
        assert ProtectedAreaType.KEY_BIODIVERSITY_AREA.value == "key_biodiversity_area"
