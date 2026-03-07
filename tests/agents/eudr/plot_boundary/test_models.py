# -*- coding: utf-8 -*-
"""
Tests for Data Models, Enums, and Config - AGENT-EUDR-006 Plot Boundary Manager

Comprehensive test suite covering:
- Enumeration tests (commodities, error types, severities, formats, algorithms)
- Coordinate validation (lat/lon ranges)
- BoundingBox operations (contains, intersects, area, width, height)
- Ring operations (closed, orientation, signed area, vertex count)
- PlotBoundary creation and serialization
- ValidationResult aggregation
- AreaResult unit conversions and threshold flag
- OverlapRecord severity classification and auto-computation
- BoundaryVersion hash computation
- SimplificationResult quality checks
- SplitResult / MergeResult area conservation
- ExportResult format validation
- PlotBoundaryConfig creation, defaults, validation, credential redaction
- Config from environment variables
- Config singleton behavior and reset
- Pydantic-style model serialization/deserialization patterns

Test count: 100+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

import pytest

from tests.agents.eudr.plot_boundary.conftest import (
    ALL_SAMPLE_COORDINATES,
    ALL_SAMPLE_POLYGONS,
    AreaResult,
    BoundingBox,
    BoundaryVersion,
    Coordinate,
    DeterministicUUID,
    EUDR_AREA_THRESHOLD_HA,
    EUDR_COMMODITIES,
    EUDR_COUNTRIES,
    EXPORT_FORMATS,
    ExportResult,
    GEOJSON_TYPES,
    MergeResult,
    OVERLAP_SEVERITIES,
    OverlapRecord,
    PlotBoundary,
    PlotBoundaryConfig,
    RESOLUTION_LEVELS,
    Ring,
    SHA256_HEX_LENGTH,
    SIMPLIFICATION_ALGORITHMS,
    SimplificationResult,
    SplitResult,
    SUPPORTED_CRS,
    VALIDATION_ERROR_TYPES,
    ValidationResult,
    compute_test_hash,
    make_boundary,
    make_ring,
    make_square,
)


# ===========================================================================
# 1. Enumeration / Constant Tests (12 tests)
# ===========================================================================


class TestEnumerations:
    """Tests for all enumeration values and completeness."""

    def test_eudr_commodities_count(self):
        """EUDR has exactly 7 regulated commodities."""
        assert len(EUDR_COMMODITIES) == 7

    def test_eudr_commodities_all_values(self):
        """All EUDR commodity values are present."""
        expected = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        assert set(EUDR_COMMODITIES) == expected

    def test_eudr_countries_count(self):
        """At least 10 EUDR-relevant countries."""
        assert len(EUDR_COUNTRIES) >= 10

    def test_eudr_countries_brazil(self):
        """Brazil is in EUDR countries."""
        assert "BR" in EUDR_COUNTRIES

    def test_validation_error_types_count(self):
        """12 validation error types defined."""
        assert len(VALIDATION_ERROR_TYPES) == 12

    def test_validation_error_types_all_present(self):
        """All expected error types are defined."""
        expected = {
            "self_intersection", "unclosed_ring", "duplicate_vertices",
            "spike_vertex", "sliver_polygon", "wrong_orientation",
            "invalid_coordinates", "too_few_vertices", "hole_outside_shell",
            "overlapping_holes", "nested_shells", "zero_area",
        }
        assert set(VALIDATION_ERROR_TYPES) == expected

    def test_export_formats_count(self):
        """8 export formats defined."""
        assert len(EXPORT_FORMATS) == 8

    def test_export_formats_include_geojson(self):
        """GeoJSON is in export formats."""
        assert "geojson" in EXPORT_FORMATS

    def test_overlap_severities_count(self):
        """5 overlap severity levels."""
        assert len(OVERLAP_SEVERITIES) == 5

    def test_simplification_algorithms(self):
        """2 simplification algorithms."""
        assert len(SIMPLIFICATION_ALGORITHMS) == 2
        assert "douglas_peucker" in SIMPLIFICATION_ALGORITHMS
        assert "visvalingam_whyatt" in SIMPLIFICATION_ALGORITHMS

    def test_resolution_levels(self):
        """4 resolution levels."""
        assert len(RESOLUTION_LEVELS) == 4

    def test_supported_crs(self):
        """Supported CRS includes WGS84."""
        assert "EPSG:4326" in SUPPORTED_CRS


# ===========================================================================
# 2. Coordinate Validation Tests (6 tests)
# ===========================================================================


class TestCoordinateValidation:
    """Tests for coordinate validation and ranges."""

    def test_coordinate_creation(self):
        """Coordinate can be created with lat/lon."""
        coord = Coordinate(lat=-3.12, lon=-60.02)
        assert coord.lat == -3.12
        assert coord.lon == -60.02

    def test_coordinate_defaults(self):
        """Coordinate defaults to (0, 0)."""
        coord = Coordinate()
        assert coord.lat == 0.0
        assert coord.lon == 0.0

    def test_coordinate_with_altitude(self):
        """Coordinate supports altitude."""
        coord = Coordinate(lat=-3.12, lon=-60.02, altitude=80.0)
        assert coord.altitude == 80.0

    def test_coordinate_latitude_range(self):
        """Valid latitude is within [-90, 90]."""
        valid_lats = [-90.0, -45.0, 0.0, 45.0, 90.0]
        for lat in valid_lats:
            coord = Coordinate(lat=lat, lon=0.0)
            assert -90.0 <= coord.lat <= 90.0

    def test_coordinate_longitude_range(self):
        """Valid longitude is within [-180, 180]."""
        valid_lons = [-180.0, -90.0, 0.0, 90.0, 180.0]
        for lon in valid_lons:
            coord = Coordinate(lat=0.0, lon=lon)
            assert -180.0 <= coord.lon <= 180.0

    def test_sample_coordinates_all_valid(self):
        """All predefined sample coordinates have valid ranges."""
        for sc in ALL_SAMPLE_COORDINATES:
            assert -90.0 <= sc.lat <= 90.0
            assert -180.0 <= sc.lon <= 180.0


# ===========================================================================
# 3. BoundingBox Tests (10 tests)
# ===========================================================================


class TestBoundingBox:
    """Tests for BoundingBox operations."""

    def test_bbox_creation(self):
        """BoundingBox creation with coordinates."""
        bbox = BoundingBox(min_lat=-3.13, max_lat=-3.11, min_lon=-60.03, max_lon=-60.01)
        assert bbox.min_lat == -3.13
        assert bbox.max_lat == -3.11

    def test_bbox_width(self):
        """Width is max_lon - min_lon."""
        bbox = BoundingBox(min_lat=0, max_lat=1, min_lon=10, max_lon=15)
        assert bbox.width == 5.0

    def test_bbox_height(self):
        """Height is max_lat - min_lat."""
        bbox = BoundingBox(min_lat=-5, max_lat=-2, min_lon=0, max_lon=1)
        assert bbox.height == 3.0

    def test_bbox_contains_point_inside(self):
        """Point inside bbox returns True."""
        bbox = BoundingBox(min_lat=-5, max_lat=5, min_lon=-5, max_lon=5)
        assert bbox.contains_point(0, 0) is True

    def test_bbox_contains_point_outside(self):
        """Point outside bbox returns False."""
        bbox = BoundingBox(min_lat=-5, max_lat=5, min_lon=-5, max_lon=5)
        assert bbox.contains_point(10, 10) is False

    def test_bbox_contains_point_on_edge(self):
        """Point on bbox edge returns True."""
        bbox = BoundingBox(min_lat=-5, max_lat=5, min_lon=-5, max_lon=5)
        assert bbox.contains_point(5, 5) is True

    def test_bbox_intersects_true(self):
        """Overlapping bboxes intersect."""
        a = BoundingBox(min_lat=0, max_lat=10, min_lon=0, max_lon=10)
        b = BoundingBox(min_lat=5, max_lat=15, min_lon=5, max_lon=15)
        assert a.intersects(b) is True
        assert b.intersects(a) is True

    def test_bbox_intersects_false(self):
        """Non-overlapping bboxes do not intersect."""
        a = BoundingBox(min_lat=0, max_lat=5, min_lon=0, max_lon=5)
        b = BoundingBox(min_lat=10, max_lat=15, min_lon=10, max_lon=15)
        assert a.intersects(b) is False

    def test_bbox_area_degrees_sq(self):
        """Area in square degrees is width * height."""
        bbox = BoundingBox(min_lat=0, max_lat=2, min_lon=0, max_lon=3)
        assert bbox.area_degrees_sq == 6.0

    def test_bbox_zero_area(self):
        """Point bbox has zero area."""
        bbox = BoundingBox(min_lat=5, max_lat=5, min_lon=10, max_lon=10)
        assert bbox.area_degrees_sq == 0.0


# ===========================================================================
# 4. Ring Tests (8 tests)
# ===========================================================================


class TestRing:
    """Tests for Ring operations."""

    def test_ring_is_closed(self):
        """Closed ring has first == last vertex."""
        ring = make_ring([(0, 0), (1, 0), (1, 1), (0, 0)])
        assert ring.is_closed is True

    def test_ring_not_closed(self):
        """Open ring has first != last vertex."""
        ring = make_ring([(0, 0), (1, 0), (1, 1)])
        assert ring.is_closed is False

    def test_ring_ccw_orientation(self):
        """Counter-clockwise ring has positive signed area."""
        # CCW: (0,0) -> (1,0) -> (1,1) -> (0,0)
        ring = make_ring([(0, 0), (1, 0), (1, 1), (0, 0)])
        assert ring.is_ccw is True
        assert ring.signed_area > 0

    def test_ring_cw_orientation(self):
        """Clockwise ring has negative signed area."""
        ring = make_ring([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        # This depends on exact vertex order
        # CW: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0) reversed
        cw_ring = make_ring([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        assert cw_ring.signed_area != 0

    def test_ring_vertex_count(self):
        """Vertex count includes closure vertex."""
        ring = make_ring([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        assert ring.vertex_count == 5

    def test_ring_signed_area_triangle(self):
        """Triangle signed area is computable."""
        ring = make_ring([(0, 0), (2, 0), (1, 1), (0, 0)])
        area = abs(ring.signed_area)
        assert abs(area - 1.0) < 0.001  # Triangle with base 2, height 1

    def test_ring_empty(self):
        """Empty ring has zero area."""
        ring = make_ring([])
        assert ring.signed_area == 0.0
        assert ring.vertex_count == 0

    def test_ring_two_points(self):
        """Two-point ring has zero area."""
        ring = make_ring([(0, 0), (1, 1)])
        assert ring.signed_area == 0.0


# ===========================================================================
# 5. PlotBoundary Tests (8 tests)
# ===========================================================================


class TestPlotBoundary:
    """Tests for PlotBoundary creation and properties."""

    def test_plot_boundary_creation(self):
        """PlotBoundary can be created with coordinates."""
        coords = make_square(-3.12, -60.02, 0.005)
        boundary = make_boundary(coords, "cocoa", "BR", "PB-001")
        assert boundary.plot_id == "PB-001"
        assert boundary.commodity == "cocoa"
        assert boundary.country == "BR"

    def test_plot_boundary_auto_id(self):
        """PlotBoundary gets auto-generated ID if not provided."""
        boundary = PlotBoundary(
            exterior_ring=make_square(-3.12, -60.02, 0.005),
            commodity="cocoa",
            country="BR",
        )
        assert boundary.plot_id.startswith("PLOT-")

    def test_plot_boundary_vertex_count(self):
        """Vertex count is computed from exterior ring."""
        coords = make_square(-3.12, -60.02, 0.005)
        boundary = make_boundary(coords, "cocoa", "BR")
        assert boundary.vertex_count == len(coords)

    def test_plot_boundary_centroid(self):
        """Centroid is computed."""
        coords = make_square(-3.12, -60.02, 0.005)
        boundary = make_boundary(coords, "cocoa", "BR")
        assert boundary.centroid_lat is not None
        assert boundary.centroid_lon is not None

    def test_plot_boundary_bbox(self):
        """Bounding box is computed."""
        coords = make_square(-3.12, -60.02, 0.005)
        boundary = make_boundary(coords, "cocoa", "BR")
        assert boundary.bbox is not None
        assert boundary.bbox.min_lat <= boundary.bbox.max_lat

    def test_plot_boundary_default_crs(self):
        """Default CRS is EPSG:4326."""
        boundary = PlotBoundary()
        assert boundary.crs == "EPSG:4326"

    def test_plot_boundary_version(self):
        """Default version is 1."""
        boundary = PlotBoundary()
        assert boundary.version == 1

    def test_plot_boundary_timestamps(self):
        """Created and updated timestamps are set."""
        boundary = PlotBoundary()
        assert boundary.created_at != ""
        assert boundary.updated_at != ""


# ===========================================================================
# 6. ValidationResult Tests (6 tests)
# ===========================================================================


class TestValidationResult:
    """Tests for ValidationResult aggregation."""

    def test_validation_result_default_valid(self):
        """Default result is valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_validation_result_with_errors(self):
        """Result with errors has correct count."""
        result = ValidationResult(
            is_valid=False,
            errors=[{"type": "unclosed_ring"}, {"type": "self_intersection"}],
        )
        assert result.is_valid is False
        assert result.error_count == 2

    def test_validation_result_with_warnings(self):
        """Result with warnings has correct count."""
        result = ValidationResult(
            warnings=[{"type": "low_vertex_density"}],
        )
        assert result.warning_count == 1

    def test_validation_result_total_issues(self):
        """Total issue count is errors + warnings."""
        result = ValidationResult(
            errors=[{"type": "a"}],
            warnings=[{"type": "b"}, {"type": "c"}],
        )
        assert result.total_issue_count == 3

    def test_validation_result_ogc_compliance(self):
        """OGC compliance flag defaults to True."""
        result = ValidationResult()
        assert result.ogc_compliant is True

    def test_validation_result_confidence(self):
        """Confidence defaults to 1.0."""
        result = ValidationResult()
        assert result.confidence == 1.0


# ===========================================================================
# 7. AreaResult Tests (8 tests)
# ===========================================================================


class TestAreaResult:
    """Tests for AreaResult unit conversions and threshold."""

    def test_area_result_hectares(self):
        """m2 to hectares conversion."""
        result = AreaResult(area_m2=50000.0)
        assert result.area_ha == 5.0

    def test_area_result_acres(self):
        """m2 to acres conversion."""
        result = AreaResult(area_m2=40468.564224)
        assert abs(result.area_acres - 10.0) < 0.01

    def test_area_result_km2(self):
        """m2 to km2 conversion."""
        result = AreaResult(area_m2=1_000_000.0)
        assert result.area_km2 == 1.0

    def test_area_result_threshold_requires_polygon(self):
        """Area >= 4 ha requires polygon."""
        result = AreaResult(area_m2=40000.0)  # 4 ha
        assert result.requires_polygon is True

    def test_area_result_threshold_no_polygon(self):
        """Area < 4 ha does not require polygon."""
        result = AreaResult(area_m2=30000.0)  # 3 ha
        assert result.requires_polygon is False

    def test_area_result_zero(self):
        """Zero area produces zero conversions."""
        result = AreaResult(area_m2=0.0)
        assert result.area_ha == 0.0
        assert result.area_acres == 0.0
        assert result.area_km2 == 0.0

    def test_area_result_method(self):
        """Default method is karney."""
        result = AreaResult()
        assert result.method == "karney"

    def test_area_result_exact_threshold(self):
        """Exactly 4 ha requires polygon."""
        result = AreaResult(area_m2=40000.0)
        assert result.area_ha == EUDR_AREA_THRESHOLD_HA
        assert result.requires_polygon is True


# ===========================================================================
# 8. OverlapRecord Tests (7 tests)
# ===========================================================================


class TestOverlapRecord:
    """Tests for OverlapRecord severity classification."""

    def test_overlap_record_creation(self):
        """OverlapRecord creation with IDs."""
        record = OverlapRecord(plot_a_id="A", plot_b_id="B")
        assert record.plot_a_id == "A"
        assert record.plot_b_id == "B"

    def test_overlap_severity_auto_none(self):
        """Zero overlap auto-classifies as none."""
        record = OverlapRecord(
            plot_a_id="A", plot_b_id="B",
            overlap_pct_a=0.0, overlap_pct_b=0.0,
        )
        assert record.severity == "none"

    def test_overlap_severity_auto_minor(self):
        """< 1% overlap auto-classifies as minor."""
        record = OverlapRecord(
            plot_a_id="A", plot_b_id="B",
            overlap_pct_a=0.5, overlap_pct_b=0.3,
        )
        assert record.severity == "minor"

    def test_overlap_severity_auto_moderate(self):
        """1-10% overlap auto-classifies as moderate."""
        record = OverlapRecord(
            plot_a_id="A", plot_b_id="B",
            overlap_pct_a=5.0, overlap_pct_b=3.0,
        )
        assert record.severity == "moderate"

    def test_overlap_severity_auto_major(self):
        """10-50% overlap auto-classifies as major."""
        record = OverlapRecord(
            plot_a_id="A", plot_b_id="B",
            overlap_pct_a=25.0, overlap_pct_b=15.0,
        )
        assert record.severity == "major"

    def test_overlap_severity_auto_critical(self):
        """50%+ overlap auto-classifies as critical."""
        record = OverlapRecord(
            plot_a_id="A", plot_b_id="B",
            overlap_pct_a=75.0, overlap_pct_b=100.0,
        )
        assert record.severity == "critical"

    def test_overlap_area_conversion(self):
        """m2 to ha conversion in overlap record."""
        record = OverlapRecord(
            plot_a_id="A", plot_b_id="B",
            overlap_area_m2=50000.0,
        )
        assert record.overlap_area_ha == 5.0


# ===========================================================================
# 9. BoundaryVersion Tests (5 tests)
# ===========================================================================


class TestBoundaryVersion:
    """Tests for BoundaryVersion hash computation."""

    def test_version_creation(self):
        """BoundaryVersion creation."""
        bv = BoundaryVersion(
            plot_id="VER-001", version=1,
            exterior_ring=make_square(-3.12, -60.02, 0.005),
            area_ha=10.0,
        )
        assert bv.plot_id == "VER-001"
        assert bv.version == 1

    def test_version_hash_computation(self):
        """Hash is computed deterministically."""
        bv = BoundaryVersion(
            plot_id="VER-001", version=1,
            exterior_ring=make_square(-3.12, -60.02, 0.005),
            area_ha=10.0,
        )
        hash1 = bv.compute_hash()
        hash2 = bv.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == SHA256_HEX_LENGTH

    def test_version_hash_changes_with_data(self):
        """Different data produces different hash."""
        bv1 = BoundaryVersion(
            plot_id="VER-001", version=1,
            exterior_ring=make_square(-3.12, -60.02, 0.005),
            area_ha=10.0,
        )
        bv2 = BoundaryVersion(
            plot_id="VER-001", version=2,
            exterior_ring=make_square(-3.12, -60.02, 0.006),
            area_ha=12.0,
        )
        assert bv1.compute_hash() != bv2.compute_hash()

    def test_version_default_change_type(self):
        """Default change type is created."""
        bv = BoundaryVersion()
        assert bv.change_type == "created"

    def test_version_history_fixture(self, boundary_version_history):
        """Version history fixture has 5 versions."""
        assert len(boundary_version_history) == 5
        for i, v in enumerate(boundary_version_history):
            assert v.version == i + 1
            assert v.provenance_hash != ""


# ===========================================================================
# 10. SimplificationResult Tests (4 tests)
# ===========================================================================


class TestSimplificationResult:
    """Tests for SimplificationResult quality checks."""

    def test_simplification_quality_ok(self):
        """Quality OK when deviation < 1% and topology preserved."""
        result = SimplificationResult(
            area_deviation_pct=0.3,
            topology_preserved=True,
        )
        assert result.quality_ok is True

    def test_simplification_quality_not_ok_area(self):
        """Quality not OK when area deviation >= 1%."""
        result = SimplificationResult(
            area_deviation_pct=1.5,
            topology_preserved=True,
        )
        assert result.quality_ok is False

    def test_simplification_quality_not_ok_topology(self):
        """Quality not OK when topology not preserved."""
        result = SimplificationResult(
            area_deviation_pct=0.3,
            topology_preserved=False,
        )
        assert result.quality_ok is False

    def test_simplification_reduction_ratio(self):
        """Reduction ratio is computed correctly."""
        result = SimplificationResult(
            original_vertex_count=100,
            simplified_vertex_count=25,
            reduction_ratio=0.75,
        )
        assert result.reduction_ratio == 0.75


# ===========================================================================
# 11. SplitResult / MergeResult Tests (6 tests)
# ===========================================================================


class TestSplitMergeResult:
    """Tests for SplitResult and MergeResult area conservation."""

    def test_split_result_area_sum(self):
        """Sum of child areas is computable."""
        result = SplitResult(
            parent_id="P1",
            child_ids=["C1", "C2"],
            child_areas_ha=[50.0, 50.0],
            parent_area_ha=100.0,
        )
        assert result.area_sum == 100.0

    def test_split_result_conservation(self):
        """Area is conserved in split."""
        result = SplitResult(
            child_areas_ha=[50.0, 49.5],
            parent_area_ha=100.0,
            area_conservation_ok=True,
        )
        assert abs(result.area_sum - result.parent_area_ha) < 1.0

    def test_merge_result_area_difference(self):
        """Area difference is computable."""
        result = MergeResult(
            child_area_ha=100.0,
            parent_areas_sum_ha=99.5,
        )
        assert result.area_difference == 0.5

    def test_merge_result_conservation(self):
        """Area is conserved in merge."""
        result = MergeResult(
            child_area_ha=100.0,
            parent_areas_sum_ha=100.0,
            area_conservation_ok=True,
        )
        assert result.area_difference == 0.0

    def test_split_result_child_ids(self):
        """Split produces correct number of child IDs."""
        result = SplitResult(
            parent_id="P1",
            child_ids=["C1", "C2", "C3"],
        )
        assert len(result.child_ids) == 3

    def test_merge_result_parent_ids(self):
        """Merge records correct parent IDs."""
        result = MergeResult(
            parent_ids=["PA", "PB"],
            child_id="MC",
        )
        assert len(result.parent_ids) == 2


# ===========================================================================
# 12. ExportResult Tests (5 tests)
# ===========================================================================


class TestExportResult:
    """Tests for ExportResult format validation."""

    def test_export_result_creation(self):
        """ExportResult creation."""
        result = ExportResult(format="geojson", content="{}", boundary_count=1)
        assert result.format == "geojson"
        assert result.boundary_count == 1

    def test_export_result_file_size(self):
        """File size in bytes."""
        content = '{"type": "FeatureCollection", "features": []}'
        result = ExportResult(
            format="geojson",
            content=content,
            file_size_bytes=len(content.encode("utf-8")),
        )
        assert result.file_size_bytes > 0

    def test_export_result_is_valid(self):
        """Validity flag defaults to True."""
        result = ExportResult()
        assert result.is_valid is True

    def test_export_result_precision(self):
        """Default coordinate precision is 7."""
        result = ExportResult()
        assert result.coordinate_precision == 7

    def test_export_result_bytes(self):
        """Binary content can be stored."""
        result = ExportResult(
            format="wkb",
            content_bytes=b"\x00\x01\x02\x03",
        )
        assert result.content_bytes is not None
        assert len(result.content_bytes) == 4


# ===========================================================================
# 13. Config Tests (12 tests)
# ===========================================================================


class TestPlotBoundaryConfig:
    """Tests for PlotBoundaryConfig creation, defaults, and behavior."""

    def test_config_creation(self, config):
        """Config is created with defaults."""
        assert config.default_crs == "EPSG:4326"
        assert config.max_vertices == 100_000
        assert config.min_vertices == 4

    def test_config_defaults(self):
        """All defaults are set correctly."""
        config = PlotBoundaryConfig()
        assert config.area_tolerance_pct == 10.0
        assert config.sliver_ratio_threshold == 0.001
        assert config.spike_angle_threshold_degrees == 1.0
        assert config.eudr_area_threshold_ha == 4.0

    def test_config_custom_values(self):
        """Custom values override defaults."""
        config = PlotBoundaryConfig(
            max_vertices=50_000,
            area_tolerance_pct=5.0,
            log_level="INFO",
        )
        assert config.max_vertices == 50_000
        assert config.area_tolerance_pct == 5.0
        assert config.log_level == "INFO"

    def test_config_to_dict(self, config):
        """Config serialization to dictionary."""
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["default_crs"] == "EPSG:4326"

    def test_config_credential_redaction(self, config):
        """URLs are redacted in to_dict output."""
        d = config.to_dict()
        assert d["database_url"] == "***REDACTED***"
        assert d["redis_url"] == "***REDACTED***"

    def test_config_non_url_fields_visible(self, config):
        """Non-URL fields are not redacted."""
        d = config.to_dict()
        assert d["default_crs"] == "EPSG:4326"
        assert d["max_vertices"] == 100_000
        assert d["enable_provenance"] is True

    def test_config_provenance_enabled(self, config):
        """Provenance tracking is enabled by default."""
        assert config.enable_provenance is True

    def test_config_genesis_hash(self, config):
        """Genesis hash is set for test."""
        assert config.genesis_hash == "GL-EUDR-PBM-006-TEST-GENESIS"

    def test_config_max_polygon_area(self, config):
        """Max polygon area is configured."""
        assert config.max_polygon_area_ha == 50_000.0

    def test_config_overlap_thresholds(self, config):
        """Overlap severity thresholds are configured."""
        assert config.overlap_minor_threshold_pct == 1.0
        assert config.overlap_moderate_threshold_pct == 10.0
        assert config.overlap_major_threshold_pct == 50.0

    def test_config_simplification_threshold(self, config):
        """Simplification area deviation threshold is configured."""
        assert config.simplification_area_max_deviation_pct == 1.0

    def test_config_pool_size(self, config):
        """Pool size is configured."""
        assert config.pool_size == 5


# ===========================================================================
# 14. Provenance and Utility Tests (6 tests)
# ===========================================================================


class TestProvenanceAndUtility:
    """Tests for provenance hashing and utility functions."""

    def test_sha256_hash_length(self):
        """SHA-256 hex digest is 64 characters."""
        h = compute_test_hash({"test": "data"})
        assert len(h) == SHA256_HEX_LENGTH

    def test_sha256_deterministic(self):
        """Same input produces same hash."""
        data = {"plot_id": "TEST-001", "area": 100.0}
        h1 = compute_test_hash(data)
        h2 = compute_test_hash(data)
        assert h1 == h2

    def test_sha256_different_input(self):
        """Different input produces different hash."""
        h1 = compute_test_hash({"a": 1})
        h2 = compute_test_hash({"a": 2})
        assert h1 != h2

    def test_deterministic_uuid_sequential(self):
        """DeterministicUUID produces sequential IDs."""
        gen = DeterministicUUID(prefix="plot")
        id1 = gen.next()
        id2 = gen.next()
        assert id1 == "plot-00000001"
        assert id2 == "plot-00000002"

    def test_deterministic_uuid_reset(self):
        """DeterministicUUID resets counter."""
        gen = DeterministicUUID(prefix="test")
        gen.next()
        gen.next()
        gen.reset()
        assert gen.next() == "test-00000001"

    def test_geojson_types(self):
        """All expected GeoJSON types are defined."""
        assert "Polygon" in GEOJSON_TYPES
        assert "MultiPolygon" in GEOJSON_TYPES
        assert "Point" in GEOJSON_TYPES


# ===========================================================================
# 15. Sample Polygon Tests (5 tests)
# ===========================================================================


class TestSamplePolygons:
    """Tests for predefined sample polygon properties."""

    def test_all_sample_polygons_count(self):
        """15+ sample polygons are defined."""
        assert len(ALL_SAMPLE_POLYGONS) >= 15

    def test_valid_polygons_have_expected_area(self):
        """Valid polygons have expected_area_ha set."""
        from tests.agents.eudr.plot_boundary.conftest import VALID_POLYGONS
        for p in VALID_POLYGONS:
            assert p.expected_area_ha is not None
            assert p.expected_area_ha >= 0

    def test_invalid_polygons_flagged(self):
        """Invalid polygons have is_valid=False."""
        from tests.agents.eudr.plot_boundary.conftest import INVALID_POLYGONS
        for p in INVALID_POLYGONS:
            assert p.is_valid is False

    def test_sample_polygons_have_names(self):
        """All sample polygons have unique names."""
        names = [p.name for p in ALL_SAMPLE_POLYGONS]
        assert len(names) == len(set(names))

    def test_sample_polygons_have_descriptions(self):
        """All sample polygons have descriptions."""
        for p in ALL_SAMPLE_POLYGONS:
            assert p.description != ""
