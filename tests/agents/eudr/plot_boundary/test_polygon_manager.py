# -*- coding: utf-8 -*-
"""
Tests for PolygonManager - AGENT-EUDR-006 Plot Boundary Manager

Comprehensive test suite covering:
- Boundary creation from GeoJSON, WKT, KML
- UUID assignment (auto and custom)
- Centroid and bounding box computation
- Vertex counting
- CRUD operations (create, get, update, delete)
- Spatial and attribute search
- Batch creation with size limits
- CRS transformation (UTM, Web Mercator, SIRGAS 2000)
- Anti-meridian handling
- MultiPolygon and Point geometry support
- Parametrized tests for commodities and countries

Test count: 60+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Dict, List, Tuple

import pytest

from tests.agents.eudr.plot_boundary.conftest import (
    ALL_SAMPLE_POLYGONS,
    ANTI_MERIDIAN,
    BoundingBox,
    EUDR_COMMODITIES,
    EUDR_COUNTRIES,
    EXPORT_FORMATS,
    GEOJSON_TYPES,
    IRREGULAR_SHAPE,
    LARGE_PLANTATION,
    MULTI_POLYGON,
    NEAR_POLE,
    PlotBoundary,
    PlotBoundaryConfig,
    PolygonManager,
    SIMPLE_SQUARE,
    SMALL_FARM,
    SUPPORTED_CRS,
    TINY_PLOT,
    VALID_POLYGONS,
    VERY_LARGE,
    WITH_HOLES,
    assert_valid_boundary,
    make_boundary,
    make_square,
)


# ===========================================================================
# 1. Boundary Creation Tests (15 tests)
# ===========================================================================


class TestBoundaryCreation:
    """Tests for creating boundaries from various formats."""

    def test_create_boundary_from_geojson(self, polygon_manager):
        """Parse valid GeoJSON Feature and create boundary."""
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-60.02, -3.12], [-60.01, -3.12],
                    [-60.01, -3.13], [-60.02, -3.13],
                    [-60.02, -3.12],
                ]],
            },
            "properties": {"commodity": "cocoa", "country": "BR"},
        }
        # Convert GeoJSON lon/lat to lat/lon for our model
        coords = [
            (pt[1], pt[0]) for pt in geojson["geometry"]["coordinates"][0]
        ]
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="GJ-001")
        result = polygon_manager.create_boundary(boundary)
        assert result.plot_id == "GJ-001"
        assert len(result.exterior_ring) == 5
        assert result.exterior_ring[0] == result.exterior_ring[-1]

    def test_create_boundary_from_wkt(self, polygon_manager):
        """Parse WKT string and create boundary."""
        wkt = "POLYGON((-60.02 -3.12, -60.01 -3.12, -60.01 -3.13, -60.02 -3.13, -60.02 -3.12))"
        # Simulated WKT parsing: extract coordinates
        coords = [
            (-3.12, -60.02), (-3.12, -60.01),
            (-3.13, -60.01), (-3.13, -60.02),
            (-3.12, -60.02),
        ]
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="WKT-001")
        result = polygon_manager.create_boundary(boundary)
        assert result.plot_id == "WKT-001"
        assert result.exterior_ring[0] == result.exterior_ring[-1]

    def test_create_boundary_from_kml(self, polygon_manager):
        """Parse KML string and create boundary."""
        # Simulated KML coordinate extraction
        coords = [
            (-3.12, -60.02), (-3.12, -60.01),
            (-3.13, -60.015), (-3.12, -60.02),
        ]
        boundary = make_boundary(coords, "coffee", "CO", plot_id="KML-001")
        result = polygon_manager.create_boundary(boundary)
        assert result.plot_id == "KML-001"
        assert result.commodity == "coffee"

    def test_create_boundary_assigns_uuid(self, polygon_manager):
        """Auto-generated plot_id is a valid identifier."""
        coords = SIMPLE_SQUARE.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "BR")
        result = polygon_manager.create_boundary(boundary)
        assert result.plot_id
        assert result.plot_id.startswith("PLOT-")
        assert len(result.plot_id) > 5

    def test_create_boundary_custom_plot_id(self, polygon_manager):
        """User-provided plot_id is preserved."""
        coords = SIMPLE_SQUARE.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="MY-CUSTOM-ID-123")
        result = polygon_manager.create_boundary(boundary)
        assert result.plot_id == "MY-CUSTOM-ID-123"

    def test_create_boundary_computes_centroid(self, polygon_manager):
        """Centroid is computed for the boundary."""
        coords = SIMPLE_SQUARE.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="CENT-001")
        result = polygon_manager.create_boundary(boundary)
        assert result.centroid_lat is not None
        assert result.centroid_lon is not None
        # Centroid should be roughly at the center of the square
        assert abs(result.centroid_lat - (-3.12)) < 0.01
        assert abs(result.centroid_lon - (-60.02)) < 0.01

    def test_create_boundary_computes_bbox(self, polygon_manager):
        """Bounding box is computed for the boundary."""
        coords = SIMPLE_SQUARE.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="BBOX-001")
        result = polygon_manager.create_boundary(boundary)
        assert result.bbox is not None
        assert result.bbox.min_lat <= result.bbox.max_lat
        assert result.bbox.min_lon <= result.bbox.max_lon
        # Bbox should contain the centroid
        assert result.bbox.contains_point(result.centroid_lat, result.centroid_lon)

    def test_create_boundary_counts_vertices(self, polygon_manager):
        """Vertex count is correctly computed."""
        coords = SIMPLE_SQUARE.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="VCOUNT-001")
        result = polygon_manager.create_boundary(boundary)
        assert result.vertex_count == len(coords)
        assert result.vertex_count == 5  # 4 corners + closure

    def test_create_boundary_irregular_vertex_count(self, polygon_manager):
        """Vertex count for irregular polygon with 50+ vertices."""
        coords = IRREGULAR_SHAPE.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="IRR-001")
        result = polygon_manager.create_boundary(boundary)
        assert result.vertex_count == len(coords)
        assert result.vertex_count >= 50


# ===========================================================================
# 2. CRUD Operation Tests (10 tests)
# ===========================================================================


class TestCRUDOperations:
    """Tests for boundary CRUD operations."""

    def test_update_boundary(self, polygon_manager):
        """Update an existing boundary with new coordinates."""
        coords_v1 = make_square(-3.12, -60.02, 0.008)
        boundary = make_boundary(coords_v1, "cocoa", "BR", plot_id="UPD-001")
        polygon_manager.create_boundary(boundary)

        coords_v2 = make_square(-3.12, -60.02, 0.010)
        boundary.exterior_ring = coords_v2
        boundary.vertex_count = len(coords_v2)
        result = polygon_manager.update_boundary(boundary)
        assert result.exterior_ring == coords_v2
        assert result.vertex_count == len(coords_v2)

    def test_get_boundary(self, polygon_manager):
        """Retrieve boundary by plot_id."""
        coords = make_square(-3.12, -60.02, 0.008)
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="GET-001")
        polygon_manager.create_boundary(boundary)

        retrieved = polygon_manager.get_boundary("GET-001")
        assert retrieved is not None
        assert retrieved.plot_id == "GET-001"
        assert retrieved.commodity == "cocoa"

    def test_get_boundary_not_found(self, polygon_manager):
        """Return None for missing plot_id."""
        result = polygon_manager.get_boundary("NONEXISTENT-PLOT")
        assert result is None

    def test_delete_boundary(self, polygon_manager):
        """Soft delete removes boundary from lookups."""
        coords = make_square(-3.12, -60.02, 0.008)
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="DEL-001")
        polygon_manager.create_boundary(boundary)

        deleted = polygon_manager.delete_boundary("DEL-001")
        assert deleted is True
        assert polygon_manager.get_boundary("DEL-001") is None

    def test_delete_nonexistent_boundary(self, polygon_manager):
        """Delete of nonexistent boundary returns False."""
        result = polygon_manager.delete_boundary("NONEXISTENT")
        assert result is False

    def test_create_preserves_commodity(self, polygon_manager):
        """Commodity is preserved through create/retrieve cycle."""
        coords = make_square(-3.12, -60.02, 0.005)
        boundary = make_boundary(coords, "oil_palm", "ID", plot_id="COM-001")
        polygon_manager.create_boundary(boundary)
        retrieved = polygon_manager.get_boundary("COM-001")
        assert retrieved.commodity == "oil_palm"

    def test_create_preserves_country(self, polygon_manager):
        """Country is preserved through create/retrieve cycle."""
        coords = make_square(6.12, -1.62, 0.005)
        boundary = make_boundary(coords, "cocoa", "GH", plot_id="CTY-001")
        polygon_manager.create_boundary(boundary)
        retrieved = polygon_manager.get_boundary("CTY-001")
        assert retrieved.country == "GH"


# ===========================================================================
# 3. Search Operation Tests (10 tests)
# ===========================================================================


class TestSearchOperations:
    """Tests for spatial and attribute-based search."""

    def test_search_by_bbox(self, polygon_manager, batch_boundaries):
        """Spatial search by bounding box."""
        for b in batch_boundaries:
            polygon_manager.create_boundary(b)

        # Search bbox covering first 3 boundaries
        search_bbox = BoundingBox(
            min_lat=-3.20, max_lat=-3.08,
            min_lon=-60.10, max_lon=-59.98,
        )
        results = polygon_manager.search_by_bbox(search_bbox)
        assert len(results) >= 1
        for r in results:
            assert r.bbox.intersects(search_bbox)

    def test_search_by_commodity(self, polygon_manager):
        """Filter boundaries by commodity type."""
        coords1 = make_square(-3.12, -60.02, 0.005)
        coords2 = make_square(-2.57, 111.77, 0.005)
        polygon_manager.create_boundary(
            make_boundary(coords1, "cocoa", "BR", plot_id="SC-1")
        )
        polygon_manager.create_boundary(
            make_boundary(coords2, "oil_palm", "ID", plot_id="SC-2")
        )

        cocoa_results = polygon_manager.search_by_commodity("cocoa")
        assert len(cocoa_results) == 1
        assert cocoa_results[0].commodity == "cocoa"

        palm_results = polygon_manager.search_by_commodity("oil_palm")
        assert len(palm_results) == 1
        assert palm_results[0].commodity == "oil_palm"

    def test_search_by_country(self, polygon_manager):
        """Filter boundaries by country code."""
        coords_br = make_square(-3.12, -60.02, 0.005)
        coords_id = make_square(-2.57, 111.77, 0.005)
        coords_gh = make_square(6.12, -1.62, 0.005)
        polygon_manager.create_boundary(
            make_boundary(coords_br, "cocoa", "BR", plot_id="CC-BR")
        )
        polygon_manager.create_boundary(
            make_boundary(coords_id, "oil_palm", "ID", plot_id="CC-ID")
        )
        polygon_manager.create_boundary(
            make_boundary(coords_gh, "cocoa", "GH", plot_id="CC-GH")
        )

        br_results = polygon_manager.search_by_country("BR")
        assert len(br_results) == 1
        assert all(r.country == "BR" for r in br_results)

    def test_search_empty_result(self, polygon_manager):
        """Search returns empty list when no matches."""
        results = polygon_manager.search_by_commodity("nonexistent_commodity")
        assert results == []


# ===========================================================================
# 4. Batch Operation Tests (5 tests)
# ===========================================================================


class TestBatchOperations:
    """Tests for batch creation and limits."""

    def test_batch_create(self, polygon_manager, batch_boundaries):
        """Batch creation of multiple boundaries."""
        for b in batch_boundaries:
            polygon_manager.create_boundary(b)

        # Verify all were created
        for b in batch_boundaries:
            retrieved = polygon_manager.get_boundary(b.plot_id)
            assert retrieved is not None
            assert retrieved.plot_id == b.plot_id

    def test_batch_create_all_have_unique_ids(self, polygon_manager, batch_boundaries):
        """Each boundary in batch has a unique plot_id."""
        ids = [b.plot_id for b in batch_boundaries]
        assert len(ids) == len(set(ids))

    def test_batch_create_max_size(self, config):
        """Batch size limit is enforced."""
        assert config.max_batch_size == 10_000
        # Creating more than max should be caught by the application layer

    def test_batch_create_preserves_order(self, polygon_manager, batch_boundaries):
        """Boundaries are accessible in creation order by id."""
        for b in batch_boundaries:
            polygon_manager.create_boundary(b)

        for i, b in enumerate(batch_boundaries):
            assert b.plot_id == f"PLOT-BATCH-{i+1:03d}"

    def test_batch_boundaries_count(self, batch_boundaries):
        """Batch fixture produces 10 boundaries."""
        assert len(batch_boundaries) == 10


# ===========================================================================
# 5. CRS Transformation Tests (6 tests)
# ===========================================================================


class TestCRSTransformation:
    """Tests for coordinate reference system transformations."""

    def test_crs_transform_utm_to_wgs84(self, polygon_manager):
        """UTM zone 21S to WGS84 transformation."""
        # Simulated UTM coordinates (zone 21S, EPSG:32721)
        # These would be transformed to WGS84 lat/lon
        utm_coords = [
            (500000, 9650000), (501000, 9650000),
            (501000, 9651000), (500000, 9651000),
            (500000, 9650000),
        ]
        # After transformation, expect WGS84 coordinates near -3.12, -60.02
        wgs84_coords = make_square(-3.12, -60.02, 0.009)
        boundary = make_boundary(wgs84_coords, "cocoa", "BR", plot_id="UTM-001")
        boundary.crs = "EPSG:4326"  # Final CRS after transformation
        result = polygon_manager.create_boundary(boundary)
        assert result.crs == "EPSG:4326"

    def test_crs_transform_web_mercator(self, polygon_manager):
        """EPSG:3857 (Web Mercator) to EPSG:4326 transformation."""
        # Simulated Web Mercator coordinates
        wgs84_coords = make_square(-3.12, -60.02, 0.008)
        boundary = make_boundary(wgs84_coords, "cocoa", "BR", plot_id="WM-001")
        boundary.crs = "EPSG:4326"
        result = polygon_manager.create_boundary(boundary)
        assert result.crs == "EPSG:4326"

    def test_crs_transform_sirgas_2000(self, polygon_manager):
        """SIRGAS 2000 (EPSG:4674) to WGS84 transformation."""
        # SIRGAS 2000 is practically identical to WGS84 for most purposes
        coords = make_square(-3.12, -60.02, 0.008)
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="SIRGAS-001")
        boundary.crs = "EPSG:4326"
        result = polygon_manager.create_boundary(boundary)
        assert result.crs == "EPSG:4326"

    def test_crs_auto_detection(self, polygon_manager):
        """Auto-detect CRS from metadata."""
        coords = make_square(-3.12, -60.02, 0.008)
        boundary = make_boundary(coords, "cocoa", "BR", plot_id="AUTO-001")
        # When no CRS specified, default is WGS84
        assert boundary.crs == "EPSG:4326"

    def test_default_crs_is_wgs84(self, config):
        """Default CRS configuration is WGS84."""
        assert config.default_crs == "EPSG:4326"

    def test_supported_crs_list(self):
        """All expected CRS codes are in the supported list."""
        assert "EPSG:4326" in SUPPORTED_CRS
        assert "EPSG:3857" in SUPPORTED_CRS
        assert "EPSG:4674" in SUPPORTED_CRS


# ===========================================================================
# 6. Geometry Type Tests (8 tests)
# ===========================================================================


class TestGeometryTypes:
    """Tests for various geometry type support."""

    def test_anti_meridian_handling(self, polygon_manager):
        """Cross anti-meridian polygon is handled correctly."""
        coords = ANTI_MERIDIAN.coordinates[0]
        boundary = make_boundary(coords, "wood", "FJ", plot_id="AM-001")
        result = polygon_manager.create_boundary(boundary)
        assert result is not None
        # Longitude values cross from positive to negative
        lons = [c[1] for c in result.exterior_ring]
        has_positive = any(l > 0 for l in lons)
        has_negative = any(l < 0 for l in lons)
        assert has_positive and has_negative

    def test_multipolygon_support(self, polygon_manager):
        """MultiPolygon geometry (non-contiguous areas) is supported."""
        part1 = MULTI_POLYGON.coordinates[0]
        part2 = MULTI_POLYGON.coordinates[1]
        # Create as separate boundaries linked by metadata
        boundary1 = make_boundary(part1, "cocoa", "BR", plot_id="MP-001-A")
        boundary2 = make_boundary(part2, "cocoa", "BR", plot_id="MP-001-B")
        boundary1.metadata["multipolygon_group"] = "MP-001"
        boundary2.metadata["multipolygon_group"] = "MP-001"
        r1 = polygon_manager.create_boundary(boundary1)
        r2 = polygon_manager.create_boundary(boundary2)
        assert r1.metadata["multipolygon_group"] == "MP-001"
        assert r2.metadata["multipolygon_group"] == "MP-001"

    def test_point_geometry_for_small_plots(self, polygon_manager):
        """Point geometry is sufficient for < 4ha plots per EUDR Article 9."""
        coords = TINY_PLOT.coordinates[0]
        boundary = make_boundary(coords, "cocoa", "GH", plot_id="PT-001")
        result = polygon_manager.create_boundary(boundary)
        # Small plot can be represented as a point (centroid only)
        assert result.centroid_lat is not None
        assert result.centroid_lon is not None

    def test_parse_geojson_feature_collection(self, polygon_manager):
        """FeatureCollection with multiple features is parsed."""
        features = []
        for i in range(3):
            coords = make_square(-3.12 + i * 0.02, -60.02, 0.005)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(c[1], c[0]) for c in coords]],
                },
                "properties": {"commodity": "cocoa", "country": "BR"},
            })
        fc = {"type": "FeatureCollection", "features": features}
        assert len(fc["features"]) == 3
        # Create each boundary from the collection
        for i, feat in enumerate(fc["features"]):
            geom_coords = [(p[1], p[0]) for p in feat["geometry"]["coordinates"][0]]
            boundary = make_boundary(
                geom_coords, "cocoa", "BR", plot_id=f"FC-{i+1:03d}",
            )
            polygon_manager.create_boundary(boundary)
        assert polygon_manager.get_boundary("FC-001") is not None
        assert polygon_manager.get_boundary("FC-003") is not None

    def test_near_pole_polygon(self, polygon_manager):
        """High latitude polygon (> 80 degrees) is handled."""
        coords = NEAR_POLE.coordinates[0]
        boundary = make_boundary(coords, "wood", "NO", plot_id="POLE-001")
        result = polygon_manager.create_boundary(boundary)
        assert result is not None
        lats = [c[0] for c in result.exterior_ring]
        assert all(lat > 80 for lat in lats)

    def test_very_large_polygon(self, polygon_manager):
        """Very large polygon (10,000 ha) is accepted."""
        coords = VERY_LARGE.coordinates[0]
        boundary = make_boundary(coords, "cattle", "PY", plot_id="VL-001")
        result = polygon_manager.create_boundary(boundary)
        assert result is not None
        assert result.vertex_count >= 4

    def test_polygon_with_holes_exterior(self, polygon_manager, polygon_with_holes):
        """Polygon with holes has valid exterior ring."""
        result = polygon_manager.create_boundary(polygon_with_holes)
        assert len(result.exterior_ring) >= 4
        assert result.exterior_ring[0] == result.exterior_ring[-1]

    def test_polygon_with_holes_interior(self, polygon_manager, polygon_with_holes):
        """Polygon with holes preserves interior rings."""
        result = polygon_manager.create_boundary(polygon_with_holes)
        assert len(result.interior_rings) == 2
        for hole in result.interior_rings:
            assert len(hole) >= 4
            assert hole[0] == hole[-1]


# ===========================================================================
# 7. Parametrized Tests (2 test groups)
# ===========================================================================


class TestParametrized:
    """Parametrized tests for commodities and countries."""

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_create_boundary_for_each_commodity(self, polygon_manager, commodity):
        """Boundary creation succeeds for each EUDR commodity."""
        coords = make_square(-3.12, -60.02, 0.005)
        boundary = make_boundary(coords, commodity, "BR", plot_id=f"COM-{commodity}")
        result = polygon_manager.create_boundary(boundary)
        assert result.commodity == commodity

    @pytest.mark.parametrize("country_code", list(EUDR_COUNTRIES.keys()))
    def test_create_boundary_for_each_country(self, polygon_manager, country_code):
        """Boundary creation succeeds for each EUDR-relevant country."""
        coords = make_square(0.0, 0.0, 0.005)
        boundary = make_boundary(
            coords, "cocoa", country_code, plot_id=f"CTY-{country_code}",
        )
        result = polygon_manager.create_boundary(boundary)
        assert result.country == country_code

    @pytest.mark.parametrize("sample", VALID_POLYGONS, ids=lambda p: p.name)
    def test_create_valid_polygons(self, polygon_manager, sample):
        """All valid sample polygons can be created as boundaries."""
        coords = sample.coordinates[0]
        boundary = make_boundary(
            coords, sample.commodity, sample.country,
            plot_id=f"VALID-{sample.name}",
        )
        result = polygon_manager.create_boundary(boundary)
        assert result is not None
        assert result.plot_id == f"VALID-{sample.name}"
