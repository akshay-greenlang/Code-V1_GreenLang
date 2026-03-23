# -*- coding: utf-8 -*-
"""
GIS/Mapping Connector Agent Data Models - AGENT-DATA-006: GIS Connector

Pydantic v2 data models for the GIS/Mapping Connector Agent SDK. Defines all
enumerations, core data models, and request wrappers required for geospatial
data processing operations including format parsing, CRS transformation,
spatial analysis, land cover classification, boundary resolution, geocoding,
and layer management.

Models:
    - Enumerations: GeometryType, CRSType, GeoFormat, SpatialOperation,
        LandCoverType, BoundaryType, DataSourceStatus, LayerVisibility,
        TransformStatus, ValidationSeverity
    - Core models: Coordinate, BoundingBox, Geometry, Feature, GeoLayer,
        CRSDefinition, SpatialResult, LandCoverClassification,
        BoundaryResult, GeocodingResult, FormatConversionResult,
        TransformResult, GISStatistics, OperationLog
    - Request models: ParseFormatRequest, TransformCRSRequest,
        SpatialAnalysisRequest, GeocodingRequest, LayerCreateRequest,
        BoundaryQueryRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-006 GIS/Mapping Connector Agent
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# Enumerations
# =============================================================================


class GeometryType(str, Enum):
    """Types of geometries supported by the GIS connector.

    Covers the seven OGC Simple Features geometry types used in
    GeoJSON, WKT, and other geospatial data formats.
    """

    POINT = "point"
    LINESTRING = "linestring"
    POLYGON = "polygon"
    MULTIPOINT = "multipoint"
    MULTILINESTRING = "multilinestring"
    MULTIPOLYGON = "multipolygon"
    GEOMETRYCOLLECTION = "geometrycollection"


class CRSType(str, Enum):
    """Classification of Coordinate Reference Systems.

    Categorises CRS definitions by their projection type for
    appropriate transformation selection.
    """

    GEOGRAPHIC = "geographic"
    PROJECTED = "projected"
    LOCAL = "local"
    COMPOUND = "compound"


class GeoFormat(str, Enum):
    """Geospatial data interchange formats supported for parsing and export.

    Each value corresponds to a file / wire format that the
    FormatParserEngine can read and write.
    """

    GEOJSON = "geojson"
    WKT = "wkt"
    WKB = "wkb"
    KML = "kml"
    GML = "gml"
    SHAPEFILE = "shapefile"
    CSV = "csv"
    TOPOJSON = "topojson"


class SpatialOperation(str, Enum):
    """Spatial analysis operations available in the SpatialAnalyzerEngine.

    Operations are deterministic computational geometry algorithms
    with no ML/LLM components.
    """

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
    """CORINE-derived land cover classification types.

    Used by the LandCoverEngine for deterministic land cover
    classification and IPCC carbon stock estimation.
    """

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
    """Administrative boundary resolution levels.

    Defines the hierarchy of administrative divisions that the
    BoundaryResolverEngine can resolve coordinates to.
    """

    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"
    CITY = "city"
    DISTRICT = "district"
    POSTAL_CODE = "postal_code"


class DataSourceStatus(str, Enum):
    """Health status of geospatial data sources.

    Tracks the operational state of external GIS data providers
    and internal processing engines.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class LayerVisibility(str, Enum):
    """Visibility levels for managed geospatial layers.

    Controls access to layers within the LayerManagerEngine.
    """

    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"


class TransformStatus(str, Enum):
    """Lifecycle status of a CRS transformation or batch operation.

    Tracks the current state of a transform from submission
    through execution to completion or failure.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationSeverity(str, Enum):
    """Severity levels for geometry and data validation issues.

    Used by validation routines to classify the seriousness
    of detected problems in geospatial data.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Core Data Models
# =============================================================================


class Coordinate(BaseModel):
    """A geographic coordinate with longitude, latitude, and optional altitude.

    Validates that longitude is between -180 and 180 and latitude is
    between -90 and 90 (WGS-84 bounds).

    Attributes:
        longitude: Longitude in decimal degrees (-180 to 180).
        latitude: Latitude in decimal degrees (-90 to 90).
        altitude: Optional altitude in meters above WGS-84 ellipsoid.
    """

    longitude: float = Field(
        default=0.0,
        description="Longitude in decimal degrees (-180 to 180)",
    )
    latitude: float = Field(
        default=0.0,
        description="Latitude in decimal degrees (-90 to 90)",
    )
    altitude: Optional[float] = Field(
        None,
        description="Optional altitude in meters above WGS-84 ellipsoid",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        """Validate longitude is within -180 to 180."""
        if not (-180.0 <= v <= 180.0):
            raise ValueError(
                f"Longitude must be between -180 and 180, got {v}"
            )
        return v

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        """Validate latitude is within -90 to 90."""
        if not (-90.0 <= v <= 90.0):
            raise ValueError(
                f"Latitude must be between -90 and 90, got {v}"
            )
        return v

    def to_tuple(self) -> tuple:
        """Return coordinate as a tuple.

        Returns:
            (longitude, latitude) or (longitude, latitude, altitude) if
            altitude is set.
        """
        if self.altitude is not None:
            return (self.longitude, self.latitude, self.altitude)
        return (self.longitude, self.latitude)


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in geographic coordinates.

    Defines a rectangular geographic extent. Validates that min values
    do not exceed max values.

    Attributes:
        min_lon: Western boundary longitude.
        min_lat: Southern boundary latitude.
        max_lon: Eastern boundary longitude.
        max_lat: Northern boundary latitude.
    """

    min_lon: float = Field(
        default=-180.0,
        description="Western boundary longitude",
    )
    min_lat: float = Field(
        default=-90.0,
        description="Southern boundary latitude",
    )
    max_lon: float = Field(
        default=180.0,
        description="Eastern boundary longitude",
    )
    max_lat: float = Field(
        default=90.0,
        description="Northern boundary latitude",
    )

    model_config = ConfigDict(from_attributes=True)

    @field_validator("max_lon")
    @classmethod
    def validate_lon_range(cls, v: float, info: Any) -> float:
        """Validate min_lon <= max_lon."""
        min_lon = info.data.get("min_lon", -180.0)
        if min_lon is not None and min_lon > v:
            raise ValueError(
                f"min_lon ({min_lon}) must be <= max_lon ({v})"
            )
        return v

    @field_validator("max_lat")
    @classmethod
    def validate_lat_range(cls, v: float, info: Any) -> float:
        """Validate min_lat <= max_lat."""
        min_lat = info.data.get("min_lat", -90.0)
        if min_lat is not None and min_lat > v:
            raise ValueError(
                f"min_lat ({min_lat}) must be <= max_lat ({v})"
            )
        return v

    def contains(self, coord: Coordinate) -> bool:
        """Test whether a coordinate falls within this bounding box.

        Args:
            coord: Coordinate to test.

        Returns:
            True if the coordinate is inside or on the boundary.
        """
        return (
            self.min_lon <= coord.longitude <= self.max_lon
            and self.min_lat <= coord.latitude <= self.max_lat
        )

    def to_list(self) -> List[float]:
        """Return bounding box as [min_lon, min_lat, max_lon, max_lat].

        Returns:
            Four-element list of boundary values.
        """
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


class Geometry(BaseModel):
    """Geospatial geometry with type, coordinates, and optional properties.

    Represents any of the OGC Simple Features geometry types with
    associated coordinate arrays and arbitrary properties.

    Attributes:
        geometry_type: Type of geometry (point, linestring, polygon, etc.).
        coordinates: Coordinate array matching the geometry type.
        properties: Optional key-value properties attached to the geometry.
    """

    geometry_type: str = Field(
        default="point",
        description="Type of geometry (point, linestring, polygon, etc.)",
    )
    coordinates: Any = Field(
        default_factory=list,
        description="Coordinate array matching the geometry type",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key-value properties attached to the geometry",
    )

    model_config = ConfigDict(from_attributes=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the geometry as a GeoJSON-like dictionary.

        Returns:
            Dictionary with 'type' and 'coordinates' keys.
        """
        return {
            "type": self.geometry_type,
            "coordinates": self.coordinates,
        }


class Feature(BaseModel):
    """A geospatial feature combining geometry, properties, and CRS.

    Represents a single geographic entity with an associated geometry,
    arbitrary properties, and coordinate reference system identifier.

    Attributes:
        feature_id: Unique identifier for this feature (auto-generated).
        geometry: Optional geometry associated with the feature.
        properties: Arbitrary key-value properties for this feature.
        crs: Coordinate reference system identifier (e.g. EPSG:4326).
        provenance_hash: Optional SHA-256 provenance chain hash.
    """

    feature_id: str = Field(
        default="",
        description="Unique identifier for this feature",
    )
    geometry: Optional[Geometry] = Field(
        None,
        description="Optional geometry associated with the feature",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value properties for this feature",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate reference system identifier",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="Optional SHA-256 provenance chain hash",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.feature_id:
            self.feature_id = f"FTR-{uuid.uuid4().hex[:5]}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize feature as a GeoJSON-like dictionary.

        Returns:
            Dictionary with type, id, geometry, properties, and crs keys.
        """
        return {
            "type": "Feature",
            "id": self.feature_id,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "properties": self.properties,
            "crs": self.crs,
        }


class GeoLayer(BaseModel):
    """A managed layer containing a collection of geographic features.

    Provides a logical grouping of features with metadata, CRS,
    visibility controls, tagging, and bounding box extent.

    Attributes:
        layer_id: Unique identifier for this layer (auto-generated).
        name: Human-readable layer name.
        description: Description of the layer contents and purpose.
        features: List of features belonging to this layer.
        crs: Coordinate reference system identifier for the layer.
        visibility: Visibility level (public, private, restricted).
        tags: Tags for search and classification.
        bounding_box: Optional geographic extent of the layer.
        created_at: ISO-8601 timestamp of layer creation.
        updated_at: Optional ISO-8601 timestamp of last update.
    """

    layer_id: str = Field(
        default="",
        description="Unique identifier for this layer",
    )
    name: str = Field(
        default="",
        description="Human-readable layer name",
    )
    description: str = Field(
        default="",
        description="Description of the layer contents and purpose",
    )
    features: List[Feature] = Field(
        default_factory=list,
        description="List of features belonging to this layer",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate reference system identifier for the layer",
    )
    visibility: str = Field(
        default="public",
        description="Visibility level (public, private, restricted)",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for search and classification",
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Optional geographic extent of the layer",
    )
    created_at: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp of layer creation",
    )
    updated_at: Optional[str] = Field(
        None,
        description="Optional ISO-8601 timestamp of last update",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.layer_id:
            self.layer_id = f"LYR-{uuid.uuid4().hex[:5]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class CRSDefinition(BaseModel):
    """Coordinate Reference System definition with EPSG metadata.

    Stores the parameters of a CRS for use in coordinate
    transformation and validation operations.

    Attributes:
        code: CRS code (e.g. EPSG:4326).
        name: Human-readable CRS name.
        crs_type: Classification of the CRS (geographic, projected, etc.).
        datum: Geodetic datum name.
        unit: Measurement unit (degree, metre, etc.).
        authority: Issuing authority (e.g. EPSG).
        area_of_use: Geographic area of valid use.
    """

    code: str = Field(
        default="EPSG:4326",
        description="CRS code (e.g. EPSG:4326)",
    )
    name: str = Field(
        default="WGS 84",
        description="Human-readable CRS name",
    )
    crs_type: str = Field(
        default="geographic",
        description="Classification of the CRS (geographic, projected, etc.)",
    )
    datum: str = Field(
        default="WGS84",
        description="Geodetic datum name",
    )
    unit: str = Field(
        default="degree",
        description="Measurement unit (degree, metre, etc.)",
    )
    authority: str = Field(
        default="EPSG",
        description="Issuing authority (e.g. EPSG)",
    )
    area_of_use: str = Field(
        default="World",
        description="Geographic area of valid use",
    )

    model_config = ConfigDict(from_attributes=True)


class SpatialResult(BaseModel):
    """Result of a spatial analysis operation.

    Contains output geometry, feature counts, performance metrics,
    and provenance information from a spatial analysis execution.

    Attributes:
        result_id: Unique identifier for this result (auto-generated).
        operation: Spatial operation that produced this result.
        input_features: Number of input features processed.
        output_features: Number of output features produced.
        geometry: Optional output geometry.
        execution_time_ms: Execution time in milliseconds.
        crs: CRS of the output geometry.
        provenance_hash: Optional SHA-256 provenance chain hash.
    """

    result_id: str = Field(
        default="",
        description="Unique identifier for this result",
    )
    operation: str = Field(
        default="",
        description="Spatial operation that produced this result",
    )
    input_features: int = Field(
        default=0, ge=0,
        description="Number of input features processed",
    )
    output_features: int = Field(
        default=0, ge=0,
        description="Number of output features produced",
    )
    geometry: Optional[Geometry] = Field(
        None,
        description="Optional output geometry",
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Execution time in milliseconds",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="CRS of the output geometry",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="Optional SHA-256 provenance chain hash",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.result_id:
            self.result_id = f"SPR-{uuid.uuid4().hex[:5]}"


class LandCoverClassification(BaseModel):
    """Land cover classification result for a geographic area.

    Contains CORINE-derived classification type, area, confidence,
    and source metadata from the LandCoverEngine.

    Attributes:
        classification_id: Unique identifier (auto-generated).
        land_cover_type: CORINE land cover type.
        area_sq_km: Area of classified region in square kilometres.
        percentage: Percentage of the query area covered.
        confidence: Classification confidence score (0-1).
        geometry: Optional geometry of the classified area.
        source: Data source for the classification.
        year: Reference year for the classification data.
    """

    classification_id: str = Field(
        default="",
        description="Unique identifier for this classification",
    )
    land_cover_type: str = Field(
        default="urban",
        description="CORINE land cover type",
    )
    area_sq_km: float = Field(
        default=0.0, ge=0.0,
        description="Area of classified region in square kilometres",
    )
    percentage: float = Field(
        default=0.0, ge=0.0,
        description="Percentage of the query area covered",
    )
    confidence: float = Field(
        default=0.0, ge=0.0,
        description="Classification confidence score (0-1)",
    )
    geometry: Optional[Geometry] = Field(
        None,
        description="Optional geometry of the classified area",
    )
    source: str = Field(
        default="",
        description="Data source for the classification",
    )
    year: int = Field(
        default=2026,
        description="Reference year for the classification data",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.classification_id:
            self.classification_id = f"LCC-{uuid.uuid4().hex[:5]}"


class BoundaryResult(BaseModel):
    """Result of an administrative boundary resolution.

    Contains the resolved boundary name, ISO code, geometry, and
    hierarchical parent reference from the BoundaryResolverEngine.

    Attributes:
        boundary_id: Unique identifier for this result (auto-generated).
        boundary_type: Administrative level (country, state, etc.).
        name: Name of the resolved boundary.
        iso_code: ISO 3166 code if applicable.
        geometry: Optional boundary geometry.
        area_sq_km: Area of the boundary in square kilometres.
        population: Population within the boundary.
        parent_boundary_id: Optional parent boundary reference.
    """

    boundary_id: str = Field(
        default="",
        description="Unique identifier for this result",
    )
    boundary_type: str = Field(
        default="country",
        description="Administrative level (country, state, etc.)",
    )
    name: str = Field(
        default="",
        description="Name of the resolved boundary",
    )
    iso_code: str = Field(
        default="",
        description="ISO 3166 code if applicable",
    )
    geometry: Optional[Geometry] = Field(
        None,
        description="Optional boundary geometry",
    )
    area_sq_km: float = Field(
        default=0.0, ge=0.0,
        description="Area of the boundary in square kilometres",
    )
    population: int = Field(
        default=0, ge=0,
        description="Population within the boundary",
    )
    parent_boundary_id: Optional[str] = Field(
        None,
        description="Optional parent boundary reference",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.boundary_id:
            self.boundary_id = f"BND-{uuid.uuid4().hex[:5]}"


class GeocodingResult(BaseModel):
    """Result of a geocoding operation (forward or reverse).

    Contains the resolved coordinate, address, confidence score,
    and data source from the GeocoderEngine.

    Attributes:
        geocode_id: Unique identifier for this result (auto-generated).
        query: Original query string for forward geocoding.
        coordinate: Resolved geographic coordinate.
        address: Resolved address string for reverse geocoding.
        confidence: Geocoding confidence score (0-1).
        source: Data source used for geocoding.
        bounding_box: Optional bounding box of the resolved area.
    """

    geocode_id: str = Field(
        default="",
        description="Unique identifier for this result",
    )
    query: str = Field(
        default="",
        description="Original query string for forward geocoding",
    )
    coordinate: Optional[Coordinate] = Field(
        None,
        description="Resolved geographic coordinate",
    )
    address: str = Field(
        default="",
        description="Resolved address string for reverse geocoding",
    )
    confidence: float = Field(
        default=0.0, ge=0.0,
        description="Geocoding confidence score (0-1)",
    )
    source: str = Field(
        default="",
        description="Data source used for geocoding",
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Optional bounding box of the resolved area",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.geocode_id:
            self.geocode_id = f"GEO-{uuid.uuid4().hex[:5]}"


class FormatConversionResult(BaseModel):
    """Result of a geospatial format conversion operation.

    Contains source and target format information, feature counts,
    and conversion status from the FormatParserEngine.

    Attributes:
        conversion_id: Unique identifier for this result (auto-generated).
        source_format: Source data format.
        target_format: Target data format.
        feature_count: Number of features converted.
        success: Whether the conversion completed successfully.
        warnings: List of warning messages from conversion.
        output_size_bytes: Size of the converted output in bytes.
        provenance_hash: Optional SHA-256 provenance chain hash.
    """

    conversion_id: str = Field(
        default="",
        description="Unique identifier for this result",
    )
    source_format: str = Field(
        default="",
        description="Source data format",
    )
    target_format: str = Field(
        default="",
        description="Target data format",
    )
    feature_count: int = Field(
        default=0, ge=0,
        description="Number of features converted",
    )
    success: bool = Field(
        default=True,
        description="Whether the conversion completed successfully",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages from conversion",
    )
    output_size_bytes: int = Field(
        default=0, ge=0,
        description="Size of the converted output in bytes",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="Optional SHA-256 provenance chain hash",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.conversion_id:
            self.conversion_id = f"FCV-{uuid.uuid4().hex[:5]}"


class TransformResult(BaseModel):
    """Result of a CRS transformation operation.

    Contains source and target CRS, feature counts, execution
    status, and timing from the CRSTransformerEngine.

    Attributes:
        transform_id: Unique identifier for this result (auto-generated).
        source_crs: Source CRS identifier.
        target_crs: Target CRS identifier.
        feature_count: Number of features transformed.
        status: Execution status (pending, running, completed, failed, cancelled).
        execution_time_ms: Execution time in milliseconds.
        provenance_hash: Optional SHA-256 provenance chain hash.
    """

    transform_id: str = Field(
        default="",
        description="Unique identifier for this result",
    )
    source_crs: str = Field(
        default="",
        description="Source CRS identifier",
    )
    target_crs: str = Field(
        default="",
        description="Target CRS identifier",
    )
    feature_count: int = Field(
        default=0, ge=0,
        description="Number of features transformed",
    )
    status: str = Field(
        default="completed",
        description="Execution status",
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Execution time in milliseconds",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="Optional SHA-256 provenance chain hash",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.transform_id:
            self.transform_id = f"TRF-{uuid.uuid4().hex[:5]}"


class GISStatistics(BaseModel):
    """Aggregated statistics for the GIS Connector service.

    Provides high-level operational metrics for monitoring the
    overall health, performance, and activity of the service.

    Attributes:
        total_operations: Total number of operations processed.
        successful_operations: Number of successful operations.
        failed_operations: Number of failed operations.
        total_features_processed: Total features processed across all operations.
        avg_execution_time_ms: Average operation execution time in milliseconds.
        cache_hits: Total cache hits.
        cache_misses: Total cache misses.
        active_layers: Number of currently active layers.
        uptime_seconds: Service uptime in seconds.
    """

    total_operations: int = Field(
        default=0, ge=0,
        description="Total number of operations processed",
    )
    successful_operations: int = Field(
        default=0, ge=0,
        description="Number of successful operations",
    )
    failed_operations: int = Field(
        default=0, ge=0,
        description="Number of failed operations",
    )
    total_features_processed: int = Field(
        default=0, ge=0,
        description="Total features processed across all operations",
    )
    avg_execution_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Average operation execution time in milliseconds",
    )
    cache_hits: int = Field(
        default=0, ge=0,
        description="Total cache hits",
    )
    cache_misses: int = Field(
        default=0, ge=0,
        description="Total cache misses",
    )
    active_layers: int = Field(
        default=0, ge=0,
        description="Number of currently active layers",
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Service uptime in seconds",
    )

    model_config = ConfigDict(from_attributes=True)


class OperationLog(BaseModel):
    """Log entry for a single GIS connector operation.

    Records the operation type, parameters, result summary,
    timing, and provenance hash for audit trail purposes.

    Attributes:
        log_id: Unique identifier for this log entry (auto-generated).
        operation: Type of operation that was performed.
        status: Execution status (completed, failed, etc.).
        input_params: Input parameters for the operation.
        output_summary: Summary of the operation output.
        execution_time_ms: Execution time in milliseconds.
        error_message: Error message if the operation failed.
        created_at: ISO-8601 timestamp of log creation.
        provenance_hash: Optional SHA-256 provenance chain hash.
    """

    log_id: str = Field(
        default="",
        description="Unique identifier for this log entry",
    )
    operation: str = Field(
        default="",
        description="Type of operation that was performed",
    )
    status: str = Field(
        default="completed",
        description="Execution status (completed, failed, etc.)",
    )
    input_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for the operation",
    )
    output_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of the operation output",
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Execution time in milliseconds",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if the operation failed",
    )
    created_at: Optional[str] = Field(
        None,
        description="ISO-8601 timestamp of log creation",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="Optional SHA-256 provenance chain hash",
    )

    model_config = ConfigDict(from_attributes=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.log_id:
            self.log_id = f"LOG-{uuid.uuid4().hex[:5]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Request Models
# =============================================================================


class ParseFormatRequest(BaseModel):
    """Request body for parsing geospatial data from a given format.

    Attributes:
        data: Raw geospatial data string to parse.
        format: Data format hint (geojson, wkt, csv, kml, etc.).
        crs: Coordinate reference system of the input data.
        options: Additional parsing options.
    """

    data: str = Field(
        default="",
        description="Raw geospatial data string to parse",
    )
    format: str = Field(
        default="geojson",
        description="Data format hint (geojson, wkt, csv, kml, etc.)",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate reference system of the input data",
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parsing options",
    )

    model_config = ConfigDict(extra="forbid")


class TransformCRSRequest(BaseModel):
    """Request body for transforming features between CRS.

    Attributes:
        features: List of features to transform.
        source_crs: Source CRS identifier.
        target_crs: Target CRS identifier.
        batch_size: Number of features per processing batch.
    """

    features: List[Feature] = Field(
        default_factory=list,
        description="List of features to transform",
    )
    source_crs: str = Field(
        default="EPSG:4326",
        description="Source CRS identifier",
    )
    target_crs: str = Field(
        default="EPSG:3857",
        description="Target CRS identifier",
    )
    batch_size: int = Field(
        default=100, ge=1,
        description="Number of features per processing batch",
    )

    model_config = ConfigDict(extra="forbid")


class SpatialAnalysisRequest(BaseModel):
    """Request body for performing a spatial analysis operation.

    Attributes:
        operation: Spatial operation to perform (intersection, buffer, etc.).
        features: List of input features for the operation.
        parameters: Operation-specific parameters.
        output_crs: CRS for the output geometry.
    """

    operation: str = Field(
        default="intersection",
        description="Spatial operation to perform",
    )
    features: List[Feature] = Field(
        default_factory=list,
        description="List of input features for the operation",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters",
    )
    output_crs: str = Field(
        default="EPSG:4326",
        description="CRS for the output geometry",
    )

    model_config = ConfigDict(extra="forbid")


class GeocodingRequest(BaseModel):
    """Request body for a geocoding operation.

    Attributes:
        query: Address or place name to geocode.
        country_code: Optional ISO country code to restrict results.
        max_results: Maximum number of results to return.
        language: ISO language code for result labels.
    """

    query: str = Field(
        default="",
        description="Address or place name to geocode",
    )
    country_code: str = Field(
        default="",
        description="Optional ISO country code to restrict results",
    )
    max_results: int = Field(
        default=5, ge=1,
        description="Maximum number of results to return",
    )
    language: str = Field(
        default="en",
        description="ISO language code for result labels",
    )

    model_config = ConfigDict(extra="forbid")


class LayerCreateRequest(BaseModel):
    """Request body for creating a new geospatial layer.

    Attributes:
        name: Human-readable layer name.
        description: Description of the layer contents.
        crs: CRS identifier for the layer.
        visibility: Visibility level (public, private, restricted).
        tags: Tags for search and classification.
    """

    name: str = Field(
        default="",
        description="Human-readable layer name",
    )
    description: str = Field(
        default="",
        description="Description of the layer contents",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="CRS identifier for the layer",
    )
    visibility: str = Field(
        default="public",
        description="Visibility level (public, private, restricted)",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for search and classification",
    )

    model_config = ConfigDict(extra="forbid")


class BoundaryQueryRequest(BaseModel):
    """Request body for querying administrative boundaries.

    Attributes:
        boundary_type: Administrative level to query (country, state, etc.).
        name: Optional boundary name to search for.
        iso_code: Optional ISO code to search for.
        contains_point: Optional coordinate to find containing boundary.
        intersects_bbox: Optional bounding box for intersection query.
    """

    boundary_type: str = Field(
        default="country",
        description="Administrative level to query",
    )
    name: Optional[str] = Field(
        None,
        description="Optional boundary name to search for",
    )
    iso_code: Optional[str] = Field(
        None,
        description="Optional ISO code to search for",
    )
    contains_point: Optional[Coordinate] = Field(
        None,
        description="Optional coordinate to find containing boundary",
    )
    intersects_bbox: Optional[BoundingBox] = Field(
        None,
        description="Optional bounding box for intersection query",
    )

    model_config = ConfigDict(extra="forbid")


__all__ = [
    # Enumerations
    "GeometryType",
    "CRSType",
    "GeoFormat",
    "SpatialOperation",
    "LandCoverType",
    "BoundaryType",
    "DataSourceStatus",
    "LayerVisibility",
    "TransformStatus",
    "ValidationSeverity",
    # Core data models
    "Coordinate",
    "BoundingBox",
    "Geometry",
    "Feature",
    "GeoLayer",
    "CRSDefinition",
    "SpatialResult",
    "LandCoverClassification",
    "BoundaryResult",
    "GeocodingResult",
    "FormatConversionResult",
    "TransformResult",
    "GISStatistics",
    "OperationLog",
    # Request models
    "ParseFormatRequest",
    "TransformCRSRequest",
    "SpatialAnalysisRequest",
    "GeocodingRequest",
    "LayerCreateRequest",
    "BoundaryQueryRequest",
]
