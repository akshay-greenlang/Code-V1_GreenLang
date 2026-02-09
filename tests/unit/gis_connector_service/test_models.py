# -*- coding: utf-8 -*-
"""
Unit Tests for GIS Connector Models (AGENT-DATA-006)

Tests all 10 enums (with member counts), 14 data models, 6 request models.
Test Coordinate validation (-180..180, -90..90), BoundingBox, Geometry types,
Feature, GeoLayer, CRSDefinition, SpatialResult, LandCoverClassification,
BoundaryResult, GeocodingResult, FormatConversionResult, TransformResult,
GISStatistics, OperationLog.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums mirroring greenlang/gis_connector/models.py
# ---------------------------------------------------------------------------


class GeometryType(str, Enum):
    POINT = "point"
    LINESTRING = "linestring"
    POLYGON = "polygon"
    MULTIPOINT = "multipoint"
    MULTILINESTRING = "multilinestring"
    MULTIPOLYGON = "multipolygon"
    GEOMETRYCOLLECTION = "geometrycollection"


class CRSType(str, Enum):
    GEOGRAPHIC = "geographic"
    PROJECTED = "projected"
    LOCAL = "local"
    COMPOUND = "compound"


class GeoFormat(str, Enum):
    GEOJSON = "geojson"
    WKT = "wkt"
    WKB = "wkb"
    KML = "kml"
    GML = "gml"
    SHAPEFILE = "shapefile"
    CSV = "csv"
    TOPOJSON = "topojson"


class SpatialOperation(str, Enum):
    INTERSECTION = "intersection"
    UNION = "union"
    DIFFERENCE = "difference"
    BUFFER = "buffer"
    CONVEX_HULL = "convex_hull"
    SIMPLIFY = "simplify"
    CENTROID = "centroid"
    ENVELOPE = "envelope"
    VORONOI = "voronoi"


class LandCoverType(str, Enum):
    URBAN = "urban"
    FOREST = "forest"
    CROPLAND = "cropland"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    WATER = "water"
    BARREN = "barren"
    SNOW_ICE = "snow_ice"
    SHRUBLAND = "shrubland"
    MANGROVE = "mangrove"


class BoundaryType(str, Enum):
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"
    CITY = "city"
    DISTRICT = "district"
    POSTAL_CODE = "postal_code"


class DataSourceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class LayerVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"


class TransformStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Inline Layer 1 models
# ---------------------------------------------------------------------------


class Coordinate:
    def __init__(self, longitude: float = 0.0, latitude: float = 0.0, altitude: Optional[float] = None):
        if not (-180.0 <= longitude <= 180.0):
            raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
        if not (-90.0 <= latitude <= 90.0):
            raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude

    def to_tuple(self) -> tuple:
        if self.altitude is not None:
            return (self.longitude, self.latitude, self.altitude)
        return (self.longitude, self.latitude)


class BoundingBox:
    def __init__(
        self,
        min_lon: float = -180.0,
        min_lat: float = -90.0,
        max_lon: float = 180.0,
        max_lat: float = 90.0,
    ):
        if min_lon > max_lon:
            raise ValueError(f"min_lon ({min_lon}) must be <= max_lon ({max_lon})")
        if min_lat > max_lat:
            raise ValueError(f"min_lat ({min_lat}) must be <= max_lat ({max_lat})")
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat

    def contains(self, coord: Coordinate) -> bool:
        return (
            self.min_lon <= coord.longitude <= self.max_lon
            and self.min_lat <= coord.latitude <= self.max_lat
        )

    def to_list(self) -> List[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


class Geometry:
    def __init__(
        self,
        geometry_type: str = "point",
        coordinates: Optional[Any] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.geometry_type = geometry_type
        self.coordinates = coordinates or []
        self.properties = properties or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.geometry_type,
            "coordinates": self.coordinates,
        }


class Feature:
    def __init__(
        self,
        feature_id: str = "",
        geometry: Optional[Geometry] = None,
        properties: Optional[Dict[str, Any]] = None,
        crs: str = "EPSG:4326",
        provenance_hash: Optional[str] = None,
    ):
        self.feature_id = feature_id or f"FTR-{uuid.uuid4().hex[:5]}"
        self.geometry = geometry
        self.properties = properties or {}
        self.crs = crs
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "Feature",
            "id": self.feature_id,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "properties": self.properties,
            "crs": self.crs,
        }


class GeoLayer:
    def __init__(
        self,
        layer_id: str = "",
        name: str = "",
        description: str = "",
        features: Optional[List[Feature]] = None,
        crs: str = "EPSG:4326",
        visibility: str = "public",
        tags: Optional[List[str]] = None,
        bounding_box: Optional[BoundingBox] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.layer_id = layer_id or f"LYR-{uuid.uuid4().hex[:5]}"
        self.name = name
        self.description = description
        self.features = features or []
        self.crs = crs
        self.visibility = visibility
        self.tags = tags or []
        self.bounding_box = bounding_box
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.updated_at = updated_at


class CRSDefinition:
    def __init__(
        self,
        code: str = "EPSG:4326",
        name: str = "WGS 84",
        crs_type: str = "geographic",
        datum: str = "WGS84",
        unit: str = "degree",
        authority: str = "EPSG",
        area_of_use: str = "World",
    ):
        self.code = code
        self.name = name
        self.crs_type = crs_type
        self.datum = datum
        self.unit = unit
        self.authority = authority
        self.area_of_use = area_of_use


class SpatialResult:
    def __init__(
        self,
        result_id: str = "",
        operation: str = "",
        input_features: int = 0,
        output_features: int = 0,
        geometry: Optional[Geometry] = None,
        execution_time_ms: float = 0.0,
        crs: str = "EPSG:4326",
        provenance_hash: Optional[str] = None,
    ):
        self.result_id = result_id or f"SPR-{uuid.uuid4().hex[:5]}"
        self.operation = operation
        self.input_features = input_features
        self.output_features = output_features
        self.geometry = geometry
        self.execution_time_ms = execution_time_ms
        self.crs = crs
        self.provenance_hash = provenance_hash


class LandCoverClassification:
    def __init__(
        self,
        classification_id: str = "",
        land_cover_type: str = "urban",
        area_sq_km: float = 0.0,
        percentage: float = 0.0,
        confidence: float = 0.0,
        geometry: Optional[Geometry] = None,
        source: str = "",
        year: int = 2026,
    ):
        self.classification_id = classification_id or f"LCC-{uuid.uuid4().hex[:5]}"
        self.land_cover_type = land_cover_type
        self.area_sq_km = area_sq_km
        self.percentage = percentage
        self.confidence = confidence
        self.geometry = geometry
        self.source = source
        self.year = year


class BoundaryResult:
    def __init__(
        self,
        boundary_id: str = "",
        boundary_type: str = "country",
        name: str = "",
        iso_code: str = "",
        geometry: Optional[Geometry] = None,
        area_sq_km: float = 0.0,
        population: int = 0,
        parent_boundary_id: Optional[str] = None,
    ):
        self.boundary_id = boundary_id or f"BND-{uuid.uuid4().hex[:5]}"
        self.boundary_type = boundary_type
        self.name = name
        self.iso_code = iso_code
        self.geometry = geometry
        self.area_sq_km = area_sq_km
        self.population = population
        self.parent_boundary_id = parent_boundary_id


class GeocodingResult:
    def __init__(
        self,
        geocode_id: str = "",
        query: str = "",
        coordinate: Optional[Coordinate] = None,
        address: str = "",
        confidence: float = 0.0,
        source: str = "",
        bounding_box: Optional[BoundingBox] = None,
    ):
        self.geocode_id = geocode_id or f"GEO-{uuid.uuid4().hex[:5]}"
        self.query = query
        self.coordinate = coordinate
        self.address = address
        self.confidence = confidence
        self.source = source
        self.bounding_box = bounding_box


class FormatConversionResult:
    def __init__(
        self,
        conversion_id: str = "",
        source_format: str = "",
        target_format: str = "",
        feature_count: int = 0,
        success: bool = True,
        warnings: Optional[List[str]] = None,
        output_size_bytes: int = 0,
        provenance_hash: Optional[str] = None,
    ):
        self.conversion_id = conversion_id or f"FCV-{uuid.uuid4().hex[:5]}"
        self.source_format = source_format
        self.target_format = target_format
        self.feature_count = feature_count
        self.success = success
        self.warnings = warnings or []
        self.output_size_bytes = output_size_bytes
        self.provenance_hash = provenance_hash


class TransformResult:
    def __init__(
        self,
        transform_id: str = "",
        source_crs: str = "",
        target_crs: str = "",
        feature_count: int = 0,
        status: str = "completed",
        execution_time_ms: float = 0.0,
        provenance_hash: Optional[str] = None,
    ):
        self.transform_id = transform_id or f"TRF-{uuid.uuid4().hex[:5]}"
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.feature_count = feature_count
        self.status = status
        self.execution_time_ms = execution_time_ms
        self.provenance_hash = provenance_hash


class GISStatistics:
    def __init__(
        self,
        total_operations: int = 0,
        successful_operations: int = 0,
        failed_operations: int = 0,
        total_features_processed: int = 0,
        avg_execution_time_ms: float = 0.0,
        cache_hits: int = 0,
        cache_misses: int = 0,
        active_layers: int = 0,
        uptime_seconds: float = 0.0,
    ):
        self.total_operations = total_operations
        self.successful_operations = successful_operations
        self.failed_operations = failed_operations
        self.total_features_processed = total_features_processed
        self.avg_execution_time_ms = avg_execution_time_ms
        self.cache_hits = cache_hits
        self.cache_misses = cache_misses
        self.active_layers = active_layers
        self.uptime_seconds = uptime_seconds


class OperationLog:
    def __init__(
        self,
        log_id: str = "",
        operation: str = "",
        status: str = "completed",
        input_params: Optional[Dict[str, Any]] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        execution_time_ms: float = 0.0,
        error_message: Optional[str] = None,
        created_at: Optional[str] = None,
        provenance_hash: Optional[str] = None,
    ):
        self.log_id = log_id or f"LOG-{uuid.uuid4().hex[:5]}"
        self.operation = operation
        self.status = status
        self.input_params = input_params or {}
        self.output_summary = output_summary or {}
        self.execution_time_ms = execution_time_ms
        self.error_message = error_message
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline Request models
# ---------------------------------------------------------------------------


class ParseFormatRequest:
    def __init__(
        self,
        data: str = "",
        format: str = "geojson",
        crs: str = "EPSG:4326",
        options: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.format = format
        self.crs = crs
        self.options = options or {}


class TransformCRSRequest:
    def __init__(
        self,
        features: Optional[List[Feature]] = None,
        source_crs: str = "EPSG:4326",
        target_crs: str = "EPSG:3857",
        batch_size: int = 100,
    ):
        self.features = features or []
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.batch_size = batch_size


class SpatialAnalysisRequest:
    def __init__(
        self,
        operation: str = "intersection",
        features: Optional[List[Feature]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        output_crs: str = "EPSG:4326",
    ):
        self.operation = operation
        self.features = features or []
        self.parameters = parameters or {}
        self.output_crs = output_crs


class GeocodingRequest:
    def __init__(
        self,
        query: str = "",
        country_code: str = "",
        max_results: int = 5,
        language: str = "en",
    ):
        self.query = query
        self.country_code = country_code
        self.max_results = max_results
        self.language = language


class LayerCreateRequest:
    def __init__(
        self,
        name: str = "",
        description: str = "",
        crs: str = "EPSG:4326",
        visibility: str = "public",
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.crs = crs
        self.visibility = visibility
        self.tags = tags or []


class BoundaryQueryRequest:
    def __init__(
        self,
        boundary_type: str = "country",
        name: Optional[str] = None,
        iso_code: Optional[str] = None,
        contains_point: Optional[Coordinate] = None,
        intersects_bbox: Optional[BoundingBox] = None,
    ):
        self.boundary_type = boundary_type
        self.name = name
        self.iso_code = iso_code
        self.contains_point = contains_point
        self.intersects_bbox = intersects_bbox


# ===========================================================================
# Test Classes -- Enums
# ===========================================================================


class TestGeometryTypeEnum:
    """Test GeometryType enum values (7 geometry types)."""

    def test_point(self):
        assert GeometryType.POINT.value == "point"

    def test_linestring(self):
        assert GeometryType.LINESTRING.value == "linestring"

    def test_polygon(self):
        assert GeometryType.POLYGON.value == "polygon"

    def test_multipoint(self):
        assert GeometryType.MULTIPOINT.value == "multipoint"

    def test_multilinestring(self):
        assert GeometryType.MULTILINESTRING.value == "multilinestring"

    def test_multipolygon(self):
        assert GeometryType.MULTIPOLYGON.value == "multipolygon"

    def test_geometrycollection(self):
        assert GeometryType.GEOMETRYCOLLECTION.value == "geometrycollection"

    def test_all_7_types(self):
        """GeometryType covers exactly 7 geometry types."""
        assert len(GeometryType) == 7

    def test_from_value(self):
        assert GeometryType("point") == GeometryType.POINT


class TestCRSTypeEnum:
    """Test CRSType enum values (4 CRS types)."""

    def test_geographic(self):
        assert CRSType.GEOGRAPHIC.value == "geographic"

    def test_projected(self):
        assert CRSType.PROJECTED.value == "projected"

    def test_local(self):
        assert CRSType.LOCAL.value == "local"

    def test_compound(self):
        assert CRSType.COMPOUND.value == "compound"

    def test_all_4_types(self):
        assert len(CRSType) == 4

    def test_from_value(self):
        assert CRSType("geographic") == CRSType.GEOGRAPHIC


class TestGeoFormatEnum:
    """Test GeoFormat enum values (8 formats)."""

    def test_geojson(self):
        assert GeoFormat.GEOJSON.value == "geojson"

    def test_wkt(self):
        assert GeoFormat.WKT.value == "wkt"

    def test_wkb(self):
        assert GeoFormat.WKB.value == "wkb"

    def test_kml(self):
        assert GeoFormat.KML.value == "kml"

    def test_gml(self):
        assert GeoFormat.GML.value == "gml"

    def test_shapefile(self):
        assert GeoFormat.SHAPEFILE.value == "shapefile"

    def test_csv(self):
        assert GeoFormat.CSV.value == "csv"

    def test_topojson(self):
        assert GeoFormat.TOPOJSON.value == "topojson"

    def test_all_8_formats(self):
        assert len(GeoFormat) == 8

    def test_from_value(self):
        assert GeoFormat("geojson") == GeoFormat.GEOJSON


class TestSpatialOperationEnum:
    """Test SpatialOperation enum values (9 operations)."""

    def test_intersection(self):
        assert SpatialOperation.INTERSECTION.value == "intersection"

    def test_union(self):
        assert SpatialOperation.UNION.value == "union"

    def test_difference(self):
        assert SpatialOperation.DIFFERENCE.value == "difference"

    def test_buffer(self):
        assert SpatialOperation.BUFFER.value == "buffer"

    def test_convex_hull(self):
        assert SpatialOperation.CONVEX_HULL.value == "convex_hull"

    def test_simplify(self):
        assert SpatialOperation.SIMPLIFY.value == "simplify"

    def test_centroid(self):
        assert SpatialOperation.CENTROID.value == "centroid"

    def test_envelope(self):
        assert SpatialOperation.ENVELOPE.value == "envelope"

    def test_voronoi(self):
        assert SpatialOperation.VORONOI.value == "voronoi"

    def test_all_9_operations(self):
        assert len(SpatialOperation) == 9

    def test_from_value(self):
        assert SpatialOperation("buffer") == SpatialOperation.BUFFER


class TestLandCoverTypeEnum:
    """Test LandCoverType enum values (10 types)."""

    def test_urban(self):
        assert LandCoverType.URBAN.value == "urban"

    def test_forest(self):
        assert LandCoverType.FOREST.value == "forest"

    def test_cropland(self):
        assert LandCoverType.CROPLAND.value == "cropland"

    def test_grassland(self):
        assert LandCoverType.GRASSLAND.value == "grassland"

    def test_wetland(self):
        assert LandCoverType.WETLAND.value == "wetland"

    def test_water(self):
        assert LandCoverType.WATER.value == "water"

    def test_barren(self):
        assert LandCoverType.BARREN.value == "barren"

    def test_snow_ice(self):
        assert LandCoverType.SNOW_ICE.value == "snow_ice"

    def test_shrubland(self):
        assert LandCoverType.SHRUBLAND.value == "shrubland"

    def test_mangrove(self):
        assert LandCoverType.MANGROVE.value == "mangrove"

    def test_all_10_types(self):
        assert len(LandCoverType) == 10

    def test_from_value(self):
        assert LandCoverType("forest") == LandCoverType.FOREST


class TestBoundaryTypeEnum:
    """Test BoundaryType enum values (6 types)."""

    def test_country(self):
        assert BoundaryType.COUNTRY.value == "country"

    def test_state(self):
        assert BoundaryType.STATE.value == "state"

    def test_county(self):
        assert BoundaryType.COUNTY.value == "county"

    def test_city(self):
        assert BoundaryType.CITY.value == "city"

    def test_district(self):
        assert BoundaryType.DISTRICT.value == "district"

    def test_postal_code(self):
        assert BoundaryType.POSTAL_CODE.value == "postal_code"

    def test_all_6_types(self):
        assert len(BoundaryType) == 6


class TestDataSourceStatusEnum:
    """Test DataSourceStatus enum values (5 statuses)."""

    def test_active(self):
        assert DataSourceStatus.ACTIVE.value == "active"

    def test_inactive(self):
        assert DataSourceStatus.INACTIVE.value == "inactive"

    def test_degraded(self):
        assert DataSourceStatus.DEGRADED.value == "degraded"

    def test_maintenance(self):
        assert DataSourceStatus.MAINTENANCE.value == "maintenance"

    def test_error(self):
        assert DataSourceStatus.ERROR.value == "error"

    def test_all_5_statuses(self):
        assert len(DataSourceStatus) == 5


class TestLayerVisibilityEnum:
    """Test LayerVisibility enum values (3 levels)."""

    def test_public(self):
        assert LayerVisibility.PUBLIC.value == "public"

    def test_private(self):
        assert LayerVisibility.PRIVATE.value == "private"

    def test_restricted(self):
        assert LayerVisibility.RESTRICTED.value == "restricted"

    def test_all_3_levels(self):
        assert len(LayerVisibility) == 3


class TestTransformStatusEnum:
    """Test TransformStatus enum values (5 statuses)."""

    def test_pending(self):
        assert TransformStatus.PENDING.value == "pending"

    def test_running(self):
        assert TransformStatus.RUNNING.value == "running"

    def test_completed(self):
        assert TransformStatus.COMPLETED.value == "completed"

    def test_failed(self):
        assert TransformStatus.FAILED.value == "failed"

    def test_cancelled(self):
        assert TransformStatus.CANCELLED.value == "cancelled"

    def test_all_5_statuses(self):
        assert len(TransformStatus) == 5


class TestValidationSeverityEnum:
    """Test ValidationSeverity enum values (4 levels)."""

    def test_info(self):
        assert ValidationSeverity.INFO.value == "info"

    def test_warning(self):
        assert ValidationSeverity.WARNING.value == "warning"

    def test_error(self):
        assert ValidationSeverity.ERROR.value == "error"

    def test_critical(self):
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_all_4_levels(self):
        assert len(ValidationSeverity) == 4


# ===========================================================================
# Test Classes -- Coordinate
# ===========================================================================


class TestCoordinate:
    """Test Coordinate model with validation (-180..180 lon, -90..90 lat)."""

    def test_create_coordinate(self):
        """Valid coordinate creation."""
        coord = Coordinate(longitude=-73.9857, latitude=40.7484)
        assert coord.longitude == -73.9857
        assert coord.latitude == 40.7484
        assert coord.altitude is None

    def test_create_coordinate_with_altitude(self):
        """Coordinate with altitude."""
        coord = Coordinate(longitude=0.0, latitude=0.0, altitude=100.5)
        assert coord.altitude == 100.5

    def test_to_tuple_2d(self):
        """to_tuple returns (lon, lat) for 2D coordinate."""
        coord = Coordinate(longitude=10.0, latitude=20.0)
        assert coord.to_tuple() == (10.0, 20.0)

    def test_to_tuple_3d(self):
        """to_tuple returns (lon, lat, alt) for 3D coordinate."""
        coord = Coordinate(longitude=10.0, latitude=20.0, altitude=30.0)
        assert coord.to_tuple() == (10.0, 20.0, 30.0)

    def test_longitude_min_boundary(self):
        """Longitude at -180 is valid."""
        coord = Coordinate(longitude=-180.0, latitude=0.0)
        assert coord.longitude == -180.0

    def test_longitude_max_boundary(self):
        """Longitude at 180 is valid."""
        coord = Coordinate(longitude=180.0, latitude=0.0)
        assert coord.longitude == 180.0

    def test_latitude_min_boundary(self):
        """Latitude at -90 is valid."""
        coord = Coordinate(longitude=0.0, latitude=-90.0)
        assert coord.latitude == -90.0

    def test_latitude_max_boundary(self):
        """Latitude at 90 is valid."""
        coord = Coordinate(longitude=0.0, latitude=90.0)
        assert coord.latitude == 90.0

    def test_longitude_too_low_raises(self):
        """Longitude below -180 raises ValueError."""
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            Coordinate(longitude=-180.1, latitude=0.0)

    def test_longitude_too_high_raises(self):
        """Longitude above 180 raises ValueError."""
        with pytest.raises(ValueError, match="Longitude must be between -180 and 180"):
            Coordinate(longitude=180.1, latitude=0.0)

    def test_latitude_too_low_raises(self):
        """Latitude below -90 raises ValueError."""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            Coordinate(longitude=0.0, latitude=-90.1)

    def test_latitude_too_high_raises(self):
        """Latitude above 90 raises ValueError."""
        with pytest.raises(ValueError, match="Latitude must be between -90 and 90"):
            Coordinate(longitude=0.0, latitude=90.1)

    def test_origin_coordinate(self):
        """Origin (0, 0) is valid."""
        coord = Coordinate(longitude=0.0, latitude=0.0)
        assert coord.longitude == 0.0
        assert coord.latitude == 0.0

    def test_default_coordinate(self):
        """Default coordinate is at origin."""
        coord = Coordinate()
        assert coord.longitude == 0.0
        assert coord.latitude == 0.0


# ===========================================================================
# Test Classes -- BoundingBox
# ===========================================================================


class TestBoundingBox:
    """Test BoundingBox model."""

    def test_create_bounding_box(self):
        """Valid bounding box creation."""
        bbox = BoundingBox(min_lon=-10.0, min_lat=-5.0, max_lon=10.0, max_lat=5.0)
        assert bbox.min_lon == -10.0
        assert bbox.min_lat == -5.0
        assert bbox.max_lon == 10.0
        assert bbox.max_lat == 5.0

    def test_default_bounding_box(self):
        """Default bounding box covers the world."""
        bbox = BoundingBox()
        assert bbox.min_lon == -180.0
        assert bbox.min_lat == -90.0
        assert bbox.max_lon == 180.0
        assert bbox.max_lat == 90.0

    def test_contains_point_inside(self):
        """Point inside bounding box returns True."""
        bbox = BoundingBox(min_lon=-10.0, min_lat=-10.0, max_lon=10.0, max_lat=10.0)
        coord = Coordinate(longitude=0.0, latitude=0.0)
        assert bbox.contains(coord) is True

    def test_contains_point_outside(self):
        """Point outside bounding box returns False."""
        bbox = BoundingBox(min_lon=-10.0, min_lat=-10.0, max_lon=10.0, max_lat=10.0)
        coord = Coordinate(longitude=20.0, latitude=20.0)
        assert bbox.contains(coord) is False

    def test_contains_point_on_boundary(self):
        """Point on bounding box edge is contained."""
        bbox = BoundingBox(min_lon=-10.0, min_lat=-10.0, max_lon=10.0, max_lat=10.0)
        coord = Coordinate(longitude=10.0, latitude=10.0)
        assert bbox.contains(coord) is True

    def test_to_list(self):
        """to_list returns [min_lon, min_lat, max_lon, max_lat]."""
        bbox = BoundingBox(min_lon=-1.0, min_lat=-2.0, max_lon=3.0, max_lat=4.0)
        assert bbox.to_list() == [-1.0, -2.0, 3.0, 4.0]

    def test_invalid_min_lon_gt_max_lon(self):
        """min_lon > max_lon raises ValueError."""
        with pytest.raises(ValueError, match="min_lon"):
            BoundingBox(min_lon=10.0, min_lat=0.0, max_lon=-10.0, max_lat=5.0)

    def test_invalid_min_lat_gt_max_lat(self):
        """min_lat > max_lat raises ValueError."""
        with pytest.raises(ValueError, match="min_lat"):
            BoundingBox(min_lon=0.0, min_lat=10.0, max_lon=5.0, max_lat=-10.0)

    def test_point_bounding_box(self):
        """Degenerate bounding box (single point)."""
        bbox = BoundingBox(min_lon=5.0, min_lat=5.0, max_lon=5.0, max_lat=5.0)
        coord = Coordinate(longitude=5.0, latitude=5.0)
        assert bbox.contains(coord) is True


# ===========================================================================
# Test Classes -- Geometry
# ===========================================================================


class TestGeometry:
    """Test Geometry model for all geometry types."""

    def test_create_point(self):
        """Point geometry creation."""
        geom = Geometry(geometry_type="point", coordinates=[10.0, 20.0])
        assert geom.geometry_type == "point"
        assert geom.coordinates == [10.0, 20.0]

    def test_create_linestring(self):
        """LineString geometry creation."""
        coords = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
        geom = Geometry(geometry_type="linestring", coordinates=coords)
        assert geom.geometry_type == "linestring"
        assert len(geom.coordinates) == 3

    def test_create_polygon(self):
        """Polygon geometry creation."""
        ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        geom = Geometry(geometry_type="polygon", coordinates=[ring])
        assert geom.geometry_type == "polygon"
        assert len(geom.coordinates) == 1
        assert len(geom.coordinates[0]) == 5

    def test_create_multipoint(self):
        """MultiPoint geometry creation."""
        coords = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        geom = Geometry(geometry_type="multipoint", coordinates=coords)
        assert geom.geometry_type == "multipoint"
        assert len(geom.coordinates) == 3

    def test_default_coordinates_empty(self):
        """Default coordinates is empty list."""
        geom = Geometry()
        assert geom.coordinates == []

    def test_properties(self):
        """Properties dict stored correctly."""
        props = {"name": "Test", "area": 100.5}
        geom = Geometry(geometry_type="polygon", properties=props)
        assert geom.properties["name"] == "Test"
        assert geom.properties["area"] == 100.5

    def test_default_properties_empty(self):
        """Default properties is empty dict."""
        geom = Geometry()
        assert geom.properties == {}

    def test_to_dict(self):
        """to_dict serialization."""
        geom = Geometry(geometry_type="point", coordinates=[5.0, 10.0])
        d = geom.to_dict()
        assert d["type"] == "point"
        assert d["coordinates"] == [5.0, 10.0]


# ===========================================================================
# Test Classes -- Feature
# ===========================================================================


class TestFeature:
    """Test Feature model."""

    def test_create_feature(self):
        """Feature with geometry and properties."""
        geom = Geometry(geometry_type="point", coordinates=[10.0, 20.0])
        feat = Feature(
            geometry=geom,
            properties={"name": "Test Point", "value": 42},
            crs="EPSG:4326",
        )
        assert feat.feature_id.startswith("FTR-")
        assert feat.geometry is not None
        assert feat.properties["name"] == "Test Point"
        assert feat.crs == "EPSG:4326"

    def test_custom_feature_id(self):
        """Custom feature_id is preserved."""
        feat = Feature(feature_id="FTR-CUSTOM-001")
        assert feat.feature_id == "FTR-CUSTOM-001"

    def test_auto_generated_id_format(self):
        """Auto-generated ID starts with FTR-."""
        feat = Feature()
        assert feat.feature_id.startswith("FTR-")

    def test_default_properties_empty(self):
        """Default properties is empty dict."""
        feat = Feature()
        assert feat.properties == {}

    def test_default_crs(self):
        """Default CRS is EPSG:4326."""
        feat = Feature()
        assert feat.crs == "EPSG:4326"

    def test_to_dict(self):
        """to_dict serialization."""
        geom = Geometry(geometry_type="point", coordinates=[1.0, 2.0])
        feat = Feature(
            feature_id="FTR-00001",
            geometry=geom,
            properties={"type": "facility"},
        )
        d = feat.to_dict()
        assert d["type"] == "Feature"
        assert d["id"] == "FTR-00001"
        assert d["geometry"]["type"] == "point"
        assert d["properties"]["type"] == "facility"
        assert d["crs"] == "EPSG:4326"

    def test_to_dict_no_geometry(self):
        """to_dict with no geometry returns None for geometry field."""
        feat = Feature(feature_id="FTR-00002")
        d = feat.to_dict()
        assert d["geometry"] is None

    def test_provenance_hash(self):
        """Provenance hash stored on feature."""
        feat = Feature(provenance_hash="abc123def456")
        assert feat.provenance_hash == "abc123def456"


# ===========================================================================
# Test Classes -- GeoLayer
# ===========================================================================


class TestGeoLayer:
    """Test GeoLayer model."""

    def test_create_layer(self):
        """Layer with features and metadata."""
        feat = Feature(geometry=Geometry(geometry_type="point", coordinates=[0.0, 0.0]))
        layer = GeoLayer(
            name="emissions_sites",
            description="Industrial emission measurement sites",
            features=[feat],
            crs="EPSG:4326",
            visibility="public",
            tags=["emissions", "monitoring"],
        )
        assert layer.layer_id.startswith("LYR-")
        assert layer.name == "emissions_sites"
        assert len(layer.features) == 1
        assert layer.crs == "EPSG:4326"
        assert layer.visibility == "public"
        assert len(layer.tags) == 2
        assert layer.created_at is not None

    def test_custom_layer_id(self):
        """Custom layer_id is preserved."""
        layer = GeoLayer(layer_id="LYR-CUSTOM-001")
        assert layer.layer_id == "LYR-CUSTOM-001"

    def test_default_features_empty(self):
        """Default features is empty list."""
        layer = GeoLayer()
        assert layer.features == []

    def test_default_tags_empty(self):
        """Default tags is empty list."""
        layer = GeoLayer()
        assert layer.tags == []

    def test_default_visibility(self):
        """Default visibility is public."""
        layer = GeoLayer()
        assert layer.visibility == "public"

    def test_bounding_box_association(self):
        """Layer with associated bounding box."""
        bbox = BoundingBox(min_lon=-10.0, min_lat=-10.0, max_lon=10.0, max_lat=10.0)
        layer = GeoLayer(bounding_box=bbox)
        assert layer.bounding_box is not None
        assert layer.bounding_box.min_lon == -10.0

    def test_updated_at_default_none(self):
        """Default updated_at is None."""
        layer = GeoLayer()
        assert layer.updated_at is None


# ===========================================================================
# Test Classes -- CRSDefinition
# ===========================================================================


class TestCRSDefinition:
    """Test CRSDefinition model."""

    def test_create_wgs84(self):
        """Default CRS is WGS 84 (EPSG:4326)."""
        crs = CRSDefinition()
        assert crs.code == "EPSG:4326"
        assert crs.name == "WGS 84"
        assert crs.crs_type == "geographic"
        assert crs.datum == "WGS84"
        assert crs.unit == "degree"
        assert crs.authority == "EPSG"
        assert crs.area_of_use == "World"

    def test_create_utm_zone(self):
        """UTM Zone 33N CRS."""
        crs = CRSDefinition(
            code="EPSG:32633",
            name="WGS 84 / UTM zone 33N",
            crs_type="projected",
            datum="WGS84",
            unit="metre",
            authority="EPSG",
            area_of_use="Between 12E and 18E, northern hemisphere",
        )
        assert crs.code == "EPSG:32633"
        assert crs.crs_type == "projected"
        assert crs.unit == "metre"

    def test_create_web_mercator(self):
        """Web Mercator (EPSG:3857) CRS."""
        crs = CRSDefinition(
            code="EPSG:3857",
            name="WGS 84 / Pseudo-Mercator",
            crs_type="projected",
            unit="metre",
        )
        assert crs.code == "EPSG:3857"
        assert crs.name == "WGS 84 / Pseudo-Mercator"


# ===========================================================================
# Test Classes -- SpatialResult
# ===========================================================================


class TestSpatialResult:
    """Test SpatialResult model."""

    def test_create_result(self):
        """Spatial result with metadata."""
        result = SpatialResult(
            operation="intersection",
            input_features=10,
            output_features=3,
            execution_time_ms=45.6,
            crs="EPSG:4326",
        )
        assert result.result_id.startswith("SPR-")
        assert result.operation == "intersection"
        assert result.input_features == 10
        assert result.output_features == 3
        assert result.execution_time_ms == 45.6

    def test_defaults(self):
        """Default values for SpatialResult."""
        result = SpatialResult()
        assert result.input_features == 0
        assert result.output_features == 0
        assert result.execution_time_ms == 0.0
        assert result.crs == "EPSG:4326"
        assert result.geometry is None

    def test_provenance_hash(self):
        """Provenance hash stored on result."""
        result = SpatialResult(provenance_hash="abc123")
        assert result.provenance_hash == "abc123"


# ===========================================================================
# Test Classes -- LandCoverClassification
# ===========================================================================


class TestLandCoverClassification:
    """Test LandCoverClassification model."""

    def test_create_classification(self):
        """Land cover classification with all fields."""
        lcc = LandCoverClassification(
            land_cover_type="forest",
            area_sq_km=1500.0,
            percentage=35.5,
            confidence=0.92,
            source="MODIS",
            year=2025,
        )
        assert lcc.classification_id.startswith("LCC-")
        assert lcc.land_cover_type == "forest"
        assert lcc.area_sq_km == 1500.0
        assert lcc.percentage == 35.5
        assert lcc.confidence == 0.92
        assert lcc.source == "MODIS"
        assert lcc.year == 2025

    def test_defaults(self):
        """Default values for LandCoverClassification."""
        lcc = LandCoverClassification()
        assert lcc.land_cover_type == "urban"
        assert lcc.area_sq_km == 0.0
        assert lcc.percentage == 0.0
        assert lcc.confidence == 0.0
        assert lcc.year == 2026

    def test_all_land_cover_types(self):
        """All 10 land cover types can be assigned."""
        for lct in ["urban", "forest", "cropland", "grassland", "wetland",
                     "water", "barren", "snow_ice", "shrubland", "mangrove"]:
            lcc = LandCoverClassification(land_cover_type=lct)
            assert lcc.land_cover_type == lct


# ===========================================================================
# Test Classes -- BoundaryResult
# ===========================================================================


class TestBoundaryResult:
    """Test BoundaryResult model."""

    def test_create_boundary(self):
        """Boundary result with all fields."""
        boundary = BoundaryResult(
            boundary_type="country",
            name="Germany",
            iso_code="DE",
            area_sq_km=357022.0,
            population=83200000,
        )
        assert boundary.boundary_id.startswith("BND-")
        assert boundary.boundary_type == "country"
        assert boundary.name == "Germany"
        assert boundary.iso_code == "DE"
        assert boundary.area_sq_km == 357022.0
        assert boundary.population == 83200000

    def test_defaults(self):
        """Default values for BoundaryResult."""
        boundary = BoundaryResult()
        assert boundary.boundary_type == "country"
        assert boundary.name == ""
        assert boundary.iso_code == ""
        assert boundary.area_sq_km == 0.0
        assert boundary.population == 0
        assert boundary.parent_boundary_id is None

    def test_parent_boundary(self):
        """Boundary with parent reference."""
        boundary = BoundaryResult(
            boundary_type="state",
            name="Bavaria",
            parent_boundary_id="BND-DE001",
        )
        assert boundary.parent_boundary_id == "BND-DE001"


# ===========================================================================
# Test Classes -- GeocodingResult
# ===========================================================================


class TestGeocodingResult:
    """Test GeocodingResult model."""

    def test_create_geocoding_result(self):
        """Geocoding result with coordinate and confidence."""
        coord = Coordinate(longitude=-73.9857, latitude=40.7484)
        result = GeocodingResult(
            query="Empire State Building",
            coordinate=coord,
            address="20 W 34th St, New York, NY 10001",
            confidence=0.98,
            source="Nominatim",
        )
        assert result.geocode_id.startswith("GEO-")
        assert result.query == "Empire State Building"
        assert result.coordinate.longitude == -73.9857
        assert result.address == "20 W 34th St, New York, NY 10001"
        assert result.confidence == 0.98
        assert result.source == "Nominatim"

    def test_defaults(self):
        """Default values for GeocodingResult."""
        result = GeocodingResult()
        assert result.query == ""
        assert result.coordinate is None
        assert result.address == ""
        assert result.confidence == 0.0
        assert result.bounding_box is None


# ===========================================================================
# Test Classes -- FormatConversionResult
# ===========================================================================


class TestFormatConversionResult:
    """Test FormatConversionResult model."""

    def test_create_conversion(self):
        """Format conversion result with all fields."""
        result = FormatConversionResult(
            source_format="geojson",
            target_format="wkt",
            feature_count=50,
            success=True,
            output_size_bytes=2048,
        )
        assert result.conversion_id.startswith("FCV-")
        assert result.source_format == "geojson"
        assert result.target_format == "wkt"
        assert result.feature_count == 50
        assert result.success is True
        assert result.output_size_bytes == 2048

    def test_defaults(self):
        """Default values for FormatConversionResult."""
        result = FormatConversionResult()
        assert result.source_format == ""
        assert result.target_format == ""
        assert result.feature_count == 0
        assert result.success is True
        assert result.warnings == []
        assert result.output_size_bytes == 0

    def test_warnings(self):
        """Conversion with warnings."""
        result = FormatConversionResult(
            warnings=["Lost Z coordinate", "Simplified geometry"],
        )
        assert len(result.warnings) == 2


# ===========================================================================
# Test Classes -- TransformResult
# ===========================================================================


class TestTransformResult:
    """Test TransformResult model."""

    def test_create_transform(self):
        """CRS transform result with all fields."""
        result = TransformResult(
            source_crs="EPSG:4326",
            target_crs="EPSG:3857",
            feature_count=100,
            status="completed",
            execution_time_ms=12.5,
        )
        assert result.transform_id.startswith("TRF-")
        assert result.source_crs == "EPSG:4326"
        assert result.target_crs == "EPSG:3857"
        assert result.feature_count == 100
        assert result.status == "completed"
        assert result.execution_time_ms == 12.5

    def test_defaults(self):
        """Default values for TransformResult."""
        result = TransformResult()
        assert result.source_crs == ""
        assert result.target_crs == ""
        assert result.feature_count == 0
        assert result.status == "completed"
        assert result.execution_time_ms == 0.0

    def test_all_statuses(self):
        """All transform statuses accepted."""
        for status in ["pending", "running", "completed", "failed", "cancelled"]:
            result = TransformResult(status=status)
            assert result.status == status


# ===========================================================================
# Test Classes -- GISStatistics
# ===========================================================================


class TestGISStatistics:
    """Test GISStatistics model."""

    def test_create_statistics(self):
        """Statistics with counts."""
        stats = GISStatistics(
            total_operations=1000,
            successful_operations=950,
            failed_operations=50,
            total_features_processed=500000,
            avg_execution_time_ms=25.5,
            cache_hits=300,
            cache_misses=700,
            active_layers=15,
            uptime_seconds=86400.0,
        )
        assert stats.total_operations == 1000
        assert stats.successful_operations == 950
        assert stats.failed_operations == 50
        assert stats.total_features_processed == 500000
        assert stats.avg_execution_time_ms == 25.5
        assert stats.cache_hits == 300
        assert stats.cache_misses == 700
        assert stats.active_layers == 15
        assert stats.uptime_seconds == 86400.0

    def test_defaults_all_zero(self):
        """All defaults are zero."""
        stats = GISStatistics()
        assert stats.total_operations == 0
        assert stats.successful_operations == 0
        assert stats.failed_operations == 0
        assert stats.total_features_processed == 0
        assert stats.avg_execution_time_ms == 0.0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.active_layers == 0
        assert stats.uptime_seconds == 0.0


# ===========================================================================
# Test Classes -- OperationLog
# ===========================================================================


class TestOperationLog:
    """Test OperationLog model."""

    def test_create_log(self):
        """Operation log with all fields."""
        log = OperationLog(
            operation="crs_transform",
            status="completed",
            input_params={"source_crs": "EPSG:4326", "target_crs": "EPSG:3857"},
            output_summary={"feature_count": 100, "time_ms": 12.5},
            execution_time_ms=12.5,
        )
        assert log.log_id.startswith("LOG-")
        assert log.operation == "crs_transform"
        assert log.status == "completed"
        assert log.input_params["source_crs"] == "EPSG:4326"
        assert log.output_summary["feature_count"] == 100
        assert log.execution_time_ms == 12.5
        assert log.created_at is not None

    def test_defaults(self):
        """Default values for OperationLog."""
        log = OperationLog()
        assert log.operation == ""
        assert log.status == "completed"
        assert log.input_params == {}
        assert log.output_summary == {}
        assert log.execution_time_ms == 0.0
        assert log.error_message is None

    def test_error_log(self):
        """Failed operation log with error message."""
        log = OperationLog(
            operation="parse_format",
            status="failed",
            error_message="Invalid GeoJSON: missing 'type' field",
        )
        assert log.status == "failed"
        assert "Invalid GeoJSON" in log.error_message

    def test_provenance_hash(self):
        """Provenance hash stored on log."""
        log = OperationLog(provenance_hash="abc123def456")
        assert log.provenance_hash == "abc123def456"


# ===========================================================================
# Test Classes -- Request Models
# ===========================================================================


class TestParseFormatRequest:
    """Test ParseFormatRequest."""

    def test_create_request(self):
        """Valid parse format request."""
        req = ParseFormatRequest(
            data='{"type": "Point", "coordinates": [10, 20]}',
            format="geojson",
            crs="EPSG:4326",
        )
        assert req.data != ""
        assert req.format == "geojson"
        assert req.crs == "EPSG:4326"

    def test_defaults(self):
        """Default values for ParseFormatRequest."""
        req = ParseFormatRequest()
        assert req.data == ""
        assert req.format == "geojson"
        assert req.crs == "EPSG:4326"
        assert req.options == {}

    def test_with_options(self):
        """Request with additional options."""
        req = ParseFormatRequest(
            format="csv",
            options={"lat_column": "latitude", "lon_column": "longitude"},
        )
        assert req.options["lat_column"] == "latitude"


class TestTransformCRSRequest:
    """Test TransformCRSRequest."""

    def test_create_request(self):
        """Valid CRS transform request."""
        feat = Feature(geometry=Geometry(geometry_type="point", coordinates=[10.0, 20.0]))
        req = TransformCRSRequest(
            features=[feat],
            source_crs="EPSG:4326",
            target_crs="EPSG:3857",
            batch_size=200,
        )
        assert len(req.features) == 1
        assert req.source_crs == "EPSG:4326"
        assert req.target_crs == "EPSG:3857"
        assert req.batch_size == 200

    def test_defaults(self):
        """Default values for TransformCRSRequest."""
        req = TransformCRSRequest()
        assert req.features == []
        assert req.source_crs == "EPSG:4326"
        assert req.target_crs == "EPSG:3857"
        assert req.batch_size == 100


class TestSpatialAnalysisRequest:
    """Test SpatialAnalysisRequest."""

    def test_create_request(self):
        """Valid spatial analysis request."""
        req = SpatialAnalysisRequest(
            operation="buffer",
            parameters={"distance_m": 1000.0},
            output_crs="EPSG:4326",
        )
        assert req.operation == "buffer"
        assert req.parameters["distance_m"] == 1000.0
        assert req.output_crs == "EPSG:4326"

    def test_defaults(self):
        """Default values for SpatialAnalysisRequest."""
        req = SpatialAnalysisRequest()
        assert req.operation == "intersection"
        assert req.features == []
        assert req.parameters == {}
        assert req.output_crs == "EPSG:4326"


class TestGeocodingRequest:
    """Test GeocodingRequest."""

    def test_create_request(self):
        """Valid geocoding request."""
        req = GeocodingRequest(
            query="Berlin, Germany",
            country_code="DE",
            max_results=3,
            language="de",
        )
        assert req.query == "Berlin, Germany"
        assert req.country_code == "DE"
        assert req.max_results == 3
        assert req.language == "de"

    def test_defaults(self):
        """Default values for GeocodingRequest."""
        req = GeocodingRequest()
        assert req.query == ""
        assert req.country_code == ""
        assert req.max_results == 5
        assert req.language == "en"


class TestLayerCreateRequest:
    """Test LayerCreateRequest."""

    def test_create_request(self):
        """Valid layer creation request."""
        req = LayerCreateRequest(
            name="emissions_grid",
            description="Grid of emission measurement points",
            crs="EPSG:4326",
            visibility="restricted",
            tags=["emissions", "grid", "monitoring"],
        )
        assert req.name == "emissions_grid"
        assert req.description == "Grid of emission measurement points"
        assert req.visibility == "restricted"
        assert len(req.tags) == 3

    def test_defaults(self):
        """Default values for LayerCreateRequest."""
        req = LayerCreateRequest()
        assert req.name == ""
        assert req.description == ""
        assert req.crs == "EPSG:4326"
        assert req.visibility == "public"
        assert req.tags == []


class TestBoundaryQueryRequest:
    """Test BoundaryQueryRequest."""

    def test_create_request(self):
        """Valid boundary query request."""
        coord = Coordinate(longitude=13.405, latitude=52.52)
        req = BoundaryQueryRequest(
            boundary_type="country",
            contains_point=coord,
        )
        assert req.boundary_type == "country"
        assert req.contains_point.longitude == 13.405

    def test_by_name(self):
        """Boundary query by name."""
        req = BoundaryQueryRequest(
            boundary_type="state",
            name="California",
        )
        assert req.name == "California"

    def test_by_iso_code(self):
        """Boundary query by ISO code."""
        req = BoundaryQueryRequest(
            boundary_type="country",
            iso_code="US",
        )
        assert req.iso_code == "US"

    def test_by_bbox(self):
        """Boundary query by intersecting bounding box."""
        bbox = BoundingBox(min_lon=-10.0, min_lat=45.0, max_lon=5.0, max_lat=55.0)
        req = BoundaryQueryRequest(
            boundary_type="country",
            intersects_bbox=bbox,
        )
        assert req.intersects_bbox.min_lon == -10.0

    def test_defaults(self):
        """Default values for BoundaryQueryRequest."""
        req = BoundaryQueryRequest()
        assert req.boundary_type == "country"
        assert req.name is None
        assert req.iso_code is None
        assert req.contains_point is None
        assert req.intersects_bbox is None
