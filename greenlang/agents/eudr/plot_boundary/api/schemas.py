# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-006 Plot Boundary Manager

Pydantic v2 request/response models for the Plot Boundary Manager REST API.
Covers boundary CRUD, polygon validation, geodetic area calculation, overlap
detection, immutable version history, multi-format export, and split/merge
operations with genealogy tracking.

Schema Groups:
    - Enums: API-layer mirrors of domain enums
    - Coordinate/Geometry: GeoJSON, WKT, KML input models
    - Boundary: CRUD request/response models with batch variants
    - Validation: Topology check and auto-repair schemas
    - Area: Geodetic area calculation and EUDR 4ha threshold
    - Overlap: Overlap detection, scanning, and resolution
    - Version: Immutable version history, diff, and lineage
    - Export: Multi-format export (GeoJSON, KML, Shapefile, EUDR XML)
    - Split/Merge: Boundary splitting, merging, and genealogy
    - Batch/Health: Job management and service health

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# Enums (API-layer mirrors)
# =============================================================================


class GeometryTypeSchema(str, enum.Enum):
    """Supported geometry types for boundary input."""

    POINT = "Point"
    POLYGON = "Polygon"
    MULTI_POLYGON = "MultiPolygon"


class ValidationErrorTypeSchema(str, enum.Enum):
    """Types of geometry validation errors."""

    RING_NOT_CLOSED = "ring_not_closed"
    SELF_INTERSECTION = "self_intersection"
    DUPLICATE_VERTEX = "duplicate_vertex"
    INSUFFICIENT_VERTICES = "insufficient_vertices"
    WINDING_ORDER = "winding_order"
    SLIVER_POLYGON = "sliver_polygon"
    SPIKE_VERTEX = "spike_vertex"
    HOLE_OUTSIDE_SHELL = "hole_outside_shell"
    NESTED_HOLES = "nested_holes"
    ZERO_AREA = "zero_area"
    EXCEEDS_MAX_AREA = "exceeds_max_area"
    BELOW_MIN_AREA = "below_min_area"
    EXCEEDS_MAX_VERTICES = "exceeds_max_vertices"
    INVALID_CRS = "invalid_crs"
    NON_PLANAR = "non_planar"
    COORDINATE_OUT_OF_RANGE = "coordinate_out_of_range"


class OverlapSeveritySchema(str, enum.Enum):
    """Overlap severity classification based on area fraction."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class VersionChangeReasonSchema(str, enum.Enum):
    """Reasons for boundary version changes."""

    INITIAL_REGISTRATION = "initial_registration"
    BOUNDARY_CORRECTION = "boundary_correction"
    SURVEY_UPDATE = "survey_update"
    SPLIT_OPERATION = "split_operation"
    MERGE_OPERATION = "merge_operation"
    REGULATORY_ADJUSTMENT = "regulatory_adjustment"
    OPERATOR_UPDATE = "operator_update"
    CERTIFICATION_CHANGE = "certification_change"
    ERROR_CORRECTION = "error_correction"


class SimplificationMethodSchema(str, enum.Enum):
    """Supported polygon simplification algorithms."""

    DOUGLAS_PEUCKER = "douglas_peucker"
    VISVALINGAM_WHYATT = "visvalingam_whyatt"
    TOPOLOGY_PRESERVING = "topology_preserving"


class ExportFormatSchema(str, enum.Enum):
    """Supported export formats for boundary data."""

    GEOJSON = "geojson"
    KML = "kml"
    WKT = "wkt"
    WKB = "wkb"
    SHAPEFILE = "shapefile"
    EUDR_XML = "eudr_xml"
    GPX = "gpx"
    GML = "gml"


class ThresholdClassificationSchema(str, enum.Enum):
    """EUDR Article 9 area threshold classification."""

    BELOW_THRESHOLD = "below_threshold"
    ABOVE_THRESHOLD = "above_threshold"
    AT_THRESHOLD = "at_threshold"


class BatchStatusSchema(str, enum.Enum):
    """Status of a batch processing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


# =============================================================================
# Coordinate and Geometry Schemas
# =============================================================================


class CoordinateSchema(BaseModel):
    """A single coordinate point in WGS84."""

    lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    altitude: Optional[float] = Field(
        None,
        description="Altitude in metres above WGS84 ellipsoid",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"lat": 6.6885, "lon": -1.6244, "altitude": 250.0},
            ]
        },
    )


class BoundingBoxSchema(BaseModel):
    """Axis-aligned bounding box in WGS84 coordinates."""

    min_lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Southern latitude bound",
    )
    min_lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Western longitude bound",
    )
    max_lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Northern latitude bound",
    )
    max_lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Eastern longitude bound",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("max_lat")
    @classmethod
    def validate_lat_range(cls, v: float, info: Any) -> float:
        """Ensure max_lat >= min_lat."""
        min_lat = info.data.get("min_lat")
        if min_lat is not None and v < min_lat:
            raise ValueError(
                f"max_lat ({v}) must be >= min_lat ({min_lat})"
            )
        return v

    @field_validator("max_lon")
    @classmethod
    def validate_lon_range(cls, v: float, info: Any) -> float:
        """Ensure max_lon >= min_lon."""
        min_lon = info.data.get("min_lon")
        if min_lon is not None and v < min_lon:
            raise ValueError(
                f"max_lon ({v}) must be >= min_lon ({min_lon})"
            )
        return v


class RingSchema(BaseModel):
    """A closed ring of coordinates forming part of a polygon."""

    coordinates: List[CoordinateSchema] = Field(
        ...,
        min_length=4,
        description="Ring vertices (minimum 4 for a closed triangle)",
    )
    is_exterior: bool = Field(
        True,
        description="True for exterior ring, False for interior (hole)",
    )

    model_config = ConfigDict(extra="forbid")


class GeoJSONGeometrySchema(BaseModel):
    """Standard GeoJSON geometry input.

    Supports Point, Polygon, and MultiPolygon types per RFC 7946.
    Coordinates follow GeoJSON ordering: [longitude, latitude, altitude?].
    """

    type: str = Field(
        ...,
        description="GeoJSON geometry type (Point, Polygon, MultiPolygon)",
    )
    coordinates: Any = Field(
        ...,
        description="GeoJSON coordinate array matching the geometry type",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-1.6244, 6.6885],
                            [-1.6230, 6.6885],
                            [-1.6230, 6.6900],
                            [-1.6244, 6.6900],
                            [-1.6244, 6.6885],
                        ]
                    ],
                }
            ]
        },
    )

    @field_validator("type")
    @classmethod
    def validate_geometry_type(cls, v: str) -> str:
        """Ensure geometry type is supported."""
        allowed = {"Point", "Polygon", "MultiPolygon"}
        if v not in allowed:
            raise ValueError(
                f"Geometry type must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


# =============================================================================
# Boundary Schemas
# =============================================================================


class CreateBoundaryRequestSchema(BaseModel):
    """Request to create a new plot boundary.

    At least one geometry input (geometry, wkt, or kml) must be provided.
    If plot_id is omitted, a UUID is generated automatically.
    """

    plot_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Plot identifier (auto-generated UUID if not provided)",
    )
    geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="GeoJSON geometry input",
    )
    wkt: Optional[str] = Field(
        None,
        max_length=1_000_000,
        description="Well-Known Text (WKT) geometry string",
    )
    kml: Optional[str] = Field(
        None,
        max_length=5_000_000,
        description="KML geometry string",
    )
    source_crs: str = Field(
        "EPSG:4326",
        max_length=32,
        description="Coordinate reference system of the input geometry",
    )
    commodity: str = Field(
        ...,
        max_length=50,
        description="EUDR commodity (palm_oil, cocoa, coffee, soya, rubber, cattle, wood)",
    )
    country_iso: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    owner_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Operator or owner identifier",
    )
    certification_id: Optional[str] = Field(
        None,
        max_length=256,
        description="Certification scheme identifier (e.g. RSPO, UTZ)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional key-value metadata for the boundary",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-1.6244, 6.6885],
                                [-1.6230, 6.6885],
                                [-1.6230, 6.6900],
                                [-1.6244, 6.6900],
                                [-1.6244, 6.6885],
                            ]
                        ],
                    },
                    "commodity": "cocoa",
                    "country_iso": "GH",
                    "owner_id": "operator-001",
                }
            ]
        },
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate commodity is EUDR-regulated."""
        allowed = {"palm_oil", "cocoa", "coffee", "soya", "rubber", "cattle", "wood"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"commodity must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @model_validator(mode="after")
    def validate_geometry_provided(self) -> "CreateBoundaryRequestSchema":
        """Ensure at least one geometry input is provided."""
        if not self.geometry and not self.wkt and not self.kml:
            raise ValueError(
                "At least one geometry input required: geometry (GeoJSON), wkt, or kml"
            )
        return self


class UpdateBoundaryRequestSchema(BaseModel):
    """Request to update an existing plot boundary.

    All fields are optional. Only provided fields are updated.
    """

    geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="Updated GeoJSON geometry",
    )
    wkt: Optional[str] = Field(
        None,
        max_length=1_000_000,
        description="Updated WKT geometry string",
    )
    kml: Optional[str] = Field(
        None,
        max_length=5_000_000,
        description="Updated KML geometry string",
    )
    source_crs: Optional[str] = Field(
        None,
        max_length=32,
        description="Coordinate reference system of the updated geometry",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="Updated EUDR commodity",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Updated ISO 3166-1 alpha-2 country code",
    )
    owner_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Updated owner identifier",
    )
    certification_id: Optional[str] = Field(
        None,
        max_length=256,
        description="Updated certification identifier",
    )
    change_reason: Optional[VersionChangeReasonSchema] = Field(
        None,
        description="Reason for the boundary update",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated metadata (merged with existing)",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate commodity if provided."""
        if v is None:
            return v
        allowed = {"palm_oil", "cocoa", "coffee", "soya", "rubber", "cattle", "wood"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"commodity must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code if provided."""
        if v is None:
            return v
        v = v.strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


class CentroidSchema(BaseModel):
    """Computed centroid of a polygon boundary."""

    lat: float = Field(..., description="Centroid latitude")
    lon: float = Field(..., description="Centroid longitude")

    model_config = ConfigDict(from_attributes=True)


class CompactnessMetricsSchema(BaseModel):
    """Compactness metrics for polygon shape analysis."""

    isoperimetric_quotient: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Isoperimetric quotient (1.0 = perfect circle)",
    )
    polsby_popper: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Polsby-Popper score (4*pi*area / perimeter^2)",
    )
    convexity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of area to convex hull area",
    )

    model_config = ConfigDict(from_attributes=True)


class BoundaryResponseSchema(BaseModel):
    """Full boundary response with computed fields."""

    plot_id: str = Field(..., description="Unique plot identifier")
    geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="Boundary geometry in GeoJSON format",
    )
    geometry_type: str = Field(
        ...,
        description="Geometry type (Point, Polygon, MultiPolygon)",
    )
    source_crs: str = Field(..., description="Original input CRS")
    stored_crs: str = Field(
        "EPSG:4326",
        description="Internal storage CRS (always WGS84)",
    )
    commodity: str = Field(..., description="EUDR commodity identifier")
    country_iso: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    owner_id: Optional[str] = Field(None, description="Operator/owner identifier")
    certification_id: Optional[str] = Field(None, description="Certification identifier")
    area_m2: float = Field(..., ge=0.0, description="Geodetic area in square metres")
    area_hectares: float = Field(..., ge=0.0, description="Geodetic area in hectares")
    perimeter_m: float = Field(..., ge=0.0, description="Perimeter in metres")
    centroid: Optional[CentroidSchema] = Field(None, description="Polygon centroid")
    vertex_count: int = Field(..., ge=0, description="Total vertex count")
    ring_count: int = Field(
        ...,
        ge=0,
        description="Number of rings (1 exterior + N holes)",
    )
    is_valid: bool = Field(..., description="OGC geometry validity status")
    threshold_classification: str = Field(
        ...,
        description="EUDR 4ha threshold classification",
    )
    version_number: int = Field(..., ge=1, description="Current version number")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Last update timestamp (UTC)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata")

    model_config = ConfigDict(from_attributes=True)


class BoundaryListResponseSchema(BaseModel):
    """Paginated list of plot boundaries."""

    items: List[BoundaryResponseSchema] = Field(
        default_factory=list,
        description="Page of boundary results",
    )
    total: int = Field(..., ge=0, description="Total matching boundaries")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")

    model_config = ConfigDict(from_attributes=True)


class BatchCreateRequestSchema(BaseModel):
    """Request to create multiple plot boundaries in a single batch."""

    boundaries: List[CreateBoundaryRequestSchema] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of boundary creation requests (max 10,000)",
    )

    model_config = ConfigDict(extra="forbid")


class BatchCreateResultSchema(BaseModel):
    """Result for a single boundary in a batch create operation."""

    plot_id: str = Field(..., description="Plot identifier")
    success: bool = Field(..., description="Whether creation succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    boundary: Optional[BoundaryResponseSchema] = Field(
        None,
        description="Created boundary if successful",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchCreateResponseSchema(BaseModel):
    """Response for a batch boundary creation."""

    created: int = Field(..., ge=0, description="Number of boundaries created")
    failed: int = Field(..., ge=0, description="Number of boundaries that failed")
    total: int = Field(..., ge=0, description="Total boundaries submitted")
    results: List[BatchCreateResultSchema] = Field(
        default_factory=list,
        description="Per-boundary creation results",
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Aggregated error details",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class SearchRequestSchema(BaseModel):
    """Request to search boundaries by spatial and attribute filters."""

    bbox: Optional[BoundingBoxSchema] = Field(
        None,
        description="Bounding box filter (WGS84)",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="Filter by EUDR commodity",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Filter by ISO 3166-1 alpha-2 country code",
    )
    owner_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Filter by owner/operator identifier",
    )
    is_valid: Optional[bool] = Field(
        None,
        description="Filter by geometry validity status",
    )
    min_area_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Minimum area filter in hectares",
    )
    max_area_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum area filter in hectares",
    )
    limit: int = Field(
        100,
        ge=1,
        le=1000,
        description="Maximum results to return",
    )
    offset: int = Field(
        0,
        ge=0,
        description="Number of results to skip",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate commodity if provided."""
        if v is None:
            return v
        allowed = {"palm_oil", "cocoa", "coffee", "soya", "rubber", "cattle", "wood"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"commodity must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code if provided."""
        if v is None:
            return v
        v = v.strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


# =============================================================================
# Validation Schemas
# =============================================================================


class ValidateRequestSchema(BaseModel):
    """Request to validate polygon topology.

    Provide geometry (GeoJSON), wkt, or plot_id to validate an
    existing boundary.
    """

    geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="GeoJSON geometry to validate",
    )
    wkt: Optional[str] = Field(
        None,
        max_length=1_000_000,
        description="WKT geometry string to validate",
    )
    plot_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Existing plot_id to validate",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_input_provided(self) -> "ValidateRequestSchema":
        """Ensure at least one input source is provided."""
        if not self.geometry and not self.wkt and not self.plot_id:
            raise ValueError(
                "At least one input required: geometry (GeoJSON), wkt, or plot_id"
            )
        return self


class RepairRequestSchema(BaseModel):
    """Request to validate and optionally auto-repair a polygon.

    Provide geometry (GeoJSON), wkt, or plot_id to validate and repair.
    """

    geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="GeoJSON geometry to validate and repair",
    )
    wkt: Optional[str] = Field(
        None,
        max_length=1_000_000,
        description="WKT geometry string to validate and repair",
    )
    plot_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Existing plot_id to validate and repair",
    )
    auto_repair: bool = Field(
        True,
        description="Attempt automatic repair of detected issues",
    )
    repair_self_intersections: bool = Field(
        True,
        description="Attempt to repair self-intersecting rings",
    )
    repair_winding_order: bool = Field(
        True,
        description="Enforce counter-clockwise exterior ring winding",
    )
    remove_duplicate_vertices: bool = Field(
        True,
        description="Remove consecutive duplicate vertices",
    )
    close_unclosed_rings: bool = Field(
        True,
        description="Close unclosed polygon rings",
    )
    remove_spikes: bool = Field(
        True,
        description="Remove spike vertices below angle threshold",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_input_provided(self) -> "RepairRequestSchema":
        """Ensure at least one input source is provided."""
        if not self.geometry and not self.wkt and not self.plot_id:
            raise ValueError(
                "At least one input required: geometry (GeoJSON), wkt, or plot_id"
            )
        return self


class ValidationErrorSchema(BaseModel):
    """A single geometry validation error or warning."""

    error_type: ValidationErrorTypeSchema = Field(
        ...,
        description="Error classification",
    )
    description: str = Field(
        ...,
        description="Human-readable error description",
    )
    location: Optional[str] = Field(
        None,
        description="Location within the geometry (e.g. ring index, vertex index)",
    )
    severity: str = Field(
        "error",
        description="Severity level: error, warning, or info",
    )
    auto_repairable: bool = Field(
        False,
        description="Whether this issue can be automatically repaired",
    )

    model_config = ConfigDict(from_attributes=True)


class RepairActionSchema(BaseModel):
    """A repair action taken during auto-repair."""

    action: str = Field(..., description="Repair action identifier")
    description: str = Field(..., description="Human-readable repair description")
    vertices_affected: int = Field(
        0,
        ge=0,
        description="Number of vertices affected by the repair",
    )
    area_change_m2: float = Field(
        0.0,
        description="Area change in square metres due to repair",
    )

    model_config = ConfigDict(from_attributes=True)


class ValidationResponseSchema(BaseModel):
    """Response for geometry validation or repair."""

    is_valid: bool = Field(..., description="Overall geometry validity")
    ogc_compliant: bool = Field(..., description="OGC Simple Features compliance")
    errors: List[ValidationErrorSchema] = Field(
        default_factory=list,
        description="Validation errors detected",
    )
    warnings: List[ValidationErrorSchema] = Field(
        default_factory=list,
        description="Validation warnings (non-blocking)",
    )
    repaired: bool = Field(
        False,
        description="Whether repairs were applied",
    )
    repair_actions: List[RepairActionSchema] = Field(
        default_factory=list,
        description="List of repair actions taken",
    )
    repaired_geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="Repaired geometry (if repairs were applied)",
    )
    repaired_wkt: Optional[str] = Field(
        None,
        description="Repaired geometry as WKT (if repairs were applied)",
    )
    vertex_count_before: int = Field(
        0,
        ge=0,
        description="Vertex count before repair",
    )
    vertex_count_after: int = Field(
        0,
        ge=0,
        description="Vertex count after repair",
    )
    confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the validation result (0.0-1.0)",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash of the validation",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchValidateRequestSchema(BaseModel):
    """Request to validate multiple geometries in batch."""

    plot_ids: Optional[List[str]] = Field(
        None,
        max_length=10000,
        description="List of existing plot_ids to validate",
    )
    geometries: Optional[List[GeoJSONGeometrySchema]] = Field(
        None,
        max_length=10000,
        description="List of GeoJSON geometries to validate",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_input_provided(self) -> "BatchValidateRequestSchema":
        """Ensure at least one input source is provided."""
        if not self.plot_ids and not self.geometries:
            raise ValueError(
                "At least one input required: plot_ids or geometries"
            )
        return self


class BatchValidateResultSchema(BaseModel):
    """Result for a single item in batch validation."""

    index: int = Field(..., ge=0, description="Index in the batch")
    plot_id: Optional[str] = Field(None, description="Plot ID if provided")
    is_valid: bool = Field(..., description="Validation result")
    error_count: int = Field(0, ge=0, description="Number of errors")
    warning_count: int = Field(0, ge=0, description="Number of warnings")
    repaired: bool = Field(False, description="Whether repairs were applied")
    errors: List[ValidationErrorSchema] = Field(
        default_factory=list,
        description="Validation errors",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchValidateResponseSchema(BaseModel):
    """Response for batch validation."""

    total: int = Field(..., ge=0, description="Total items validated")
    valid: int = Field(..., ge=0, description="Number of valid items")
    invalid: int = Field(..., ge=0, description="Number of invalid items")
    repaired: int = Field(0, ge=0, description="Number of items repaired")
    results: List[BatchValidateResultSchema] = Field(
        default_factory=list,
        description="Per-item validation results",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Area Schemas
# =============================================================================


class AreaCalculationRequestSchema(BaseModel):
    """Request to calculate geodetic area of a geometry.

    Provide geometry (GeoJSON), wkt, or plot_id.
    """

    geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="GeoJSON geometry for area calculation",
    )
    wkt: Optional[str] = Field(
        None,
        max_length=1_000_000,
        description="WKT geometry for area calculation",
    )
    plot_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Existing plot_id for area calculation",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_input_provided(self) -> "AreaCalculationRequestSchema":
        """Ensure at least one input source is provided."""
        if not self.geometry and not self.wkt and not self.plot_id:
            raise ValueError(
                "At least one input required: geometry (GeoJSON), wkt, or plot_id"
            )
        return self


class AreaResponseSchema(BaseModel):
    """Geodetic area calculation response with multiple unit conversions."""

    area_m2: float = Field(..., ge=0.0, description="Area in square metres")
    area_hectares: float = Field(..., ge=0.0, description="Area in hectares")
    area_acres: float = Field(..., ge=0.0, description="Area in acres")
    area_km2: float = Field(..., ge=0.0, description="Area in square kilometres")
    perimeter_m: float = Field(..., ge=0.0, description="Perimeter in metres")
    compactness: CompactnessMetricsSchema = Field(
        ...,
        description="Polygon compactness metrics",
    )
    threshold_classification: str = Field(
        ...,
        description="EUDR 4ha threshold classification",
    )
    polygon_required: bool = Field(
        ...,
        description="Whether full polygon boundary is required per EUDR Article 9",
    )
    method: str = Field(
        ...,
        description="Area computation method used (e.g. karney_geodesic, projected_utm)",
    )
    uncertainty_m2: float = Field(
        0.0,
        ge=0.0,
        description="Estimated area uncertainty in square metres",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )

    model_config = ConfigDict(from_attributes=True)


class ThresholdResponseSchema(BaseModel):
    """EUDR Article 9 area threshold classification response."""

    area_hectares: float = Field(
        ...,
        ge=0.0,
        description="Computed area in hectares",
    )
    threshold_hectares: float = Field(
        4.0,
        description="EUDR threshold value in hectares",
    )
    classification: ThresholdClassificationSchema = Field(
        ...,
        description="Threshold classification result",
    )
    polygon_required: bool = Field(
        ...,
        description="Whether full polygon is required by EUDR Article 9",
    )
    recommendation: str = Field(
        "",
        description="Compliance recommendation",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchAreaRequestSchema(BaseModel):
    """Request for batch area calculation."""

    plot_ids: Optional[List[str]] = Field(
        None,
        max_length=10000,
        description="List of existing plot_ids",
    )
    geometries: Optional[List[GeoJSONGeometrySchema]] = Field(
        None,
        max_length=10000,
        description="List of GeoJSON geometries",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_input_provided(self) -> "BatchAreaRequestSchema":
        """Ensure at least one input source is provided."""
        if not self.plot_ids and not self.geometries:
            raise ValueError(
                "At least one input required: plot_ids or geometries"
            )
        return self


class BatchAreaResultSchema(BaseModel):
    """Result for a single item in batch area calculation."""

    index: int = Field(..., ge=0, description="Index in the batch")
    plot_id: Optional[str] = Field(None, description="Plot ID if provided")
    area_m2: float = Field(0.0, ge=0.0, description="Area in square metres")
    area_hectares: float = Field(0.0, ge=0.0, description="Area in hectares")
    perimeter_m: float = Field(0.0, ge=0.0, description="Perimeter in metres")
    threshold_classification: str = Field(
        "",
        description="EUDR threshold classification",
    )
    success: bool = Field(True, description="Whether calculation succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")

    model_config = ConfigDict(from_attributes=True)


class BatchAreaResponseSchema(BaseModel):
    """Response for batch area calculation."""

    total: int = Field(..., ge=0, description="Total items processed")
    results: List[BatchAreaResultSchema] = Field(
        default_factory=list,
        description="Per-item area results",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Overlap Schemas
# =============================================================================


class OverlapDetectRequestSchema(BaseModel):
    """Request to detect overlaps for a specific plot boundary."""

    plot_id: str = Field(
        ...,
        max_length=128,
        description="Plot ID to check for overlaps",
    )
    search_radius_km: float = Field(
        10.0,
        gt=0.0,
        le=500.0,
        description="Search radius in kilometres for overlap candidates",
    )
    min_overlap_area_m2: Optional[float] = Field(
        None,
        ge=0.0,
        description="Minimum overlap area to report (overrides config default)",
    )

    model_config = ConfigDict(extra="forbid")


class OverlapScanRequestSchema(BaseModel):
    """Request for a full registry overlap scan."""

    region_bbox: Optional[BoundingBoxSchema] = Field(
        None,
        description="Bounding box to limit scan region",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="Filter by commodity",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Filter by country",
    )
    max_results: int = Field(
        1000,
        ge=1,
        le=100000,
        description="Maximum overlap pairs to return",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate commodity if provided."""
        if v is None:
            return v
        allowed = {"palm_oil", "cocoa", "coffee", "soya", "rubber", "cattle", "wood"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"commodity must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code if provided."""
        if v is None:
            return v
        v = v.strip().upper()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


class OverlapRecordSchema(BaseModel):
    """Record of a detected overlap between two plot boundaries."""

    overlap_id: str = Field(
        default_factory=lambda: f"ovr-{uuid.uuid4().hex[:12]}",
        description="Unique overlap record identifier",
    )
    plot_id_a: str = Field(..., description="First plot in the overlap pair")
    plot_id_b: str = Field(..., description="Second plot in the overlap pair")
    overlap_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Overlap area in square metres",
    )
    overlap_area_hectares: float = Field(
        ...,
        ge=0.0,
        description="Overlap area in hectares",
    )
    overlap_pct_a: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overlap as percentage of plot A area",
    )
    overlap_pct_b: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overlap as percentage of plot B area",
    )
    severity: OverlapSeveritySchema = Field(
        ...,
        description="Overlap severity classification",
    )
    overlap_geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="GeoJSON geometry of the overlap region",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Detection timestamp (UTC)",
    )

    model_config = ConfigDict(from_attributes=True)


class OverlapResponseSchema(BaseModel):
    """Response for overlap detection."""

    plot_id: str = Field(..., description="Subject plot identifier")
    total_overlaps: int = Field(
        ...,
        ge=0,
        description="Total number of overlaps detected",
    )
    overlaps: List[OverlapRecordSchema] = Field(
        default_factory=list,
        description="Detected overlap records",
    )
    search_radius_km: float = Field(
        ...,
        description="Search radius used in kilometres",
    )
    candidates_checked: int = Field(
        0,
        ge=0,
        description="Number of candidate boundaries evaluated",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class OverlapScanResponseSchema(BaseModel):
    """Response for a full registry overlap scan."""

    total_overlaps: int = Field(
        ...,
        ge=0,
        description="Total overlap pairs detected",
    )
    overlaps: List[OverlapRecordSchema] = Field(
        default_factory=list,
        description="Detected overlap records",
    )
    boundaries_scanned: int = Field(
        0,
        ge=0,
        description="Total boundaries included in the scan",
    )
    severity_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of overlaps by severity level",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class OverlapResolutionSchema(BaseModel):
    """Suggested resolution for a detected overlap."""

    overlap_id: str = Field(..., description="Overlap record identifier")
    suggested_resolution: str = Field(
        ...,
        description="Recommended resolution strategy",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resolution details and parameters",
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the suggested resolution",
    )
    alternative_resolutions: List[str] = Field(
        default_factory=list,
        description="Alternative resolution strategies",
    )

    model_config = ConfigDict(from_attributes=True)


class OverlapResolutionRequestSchema(BaseModel):
    """Request to resolve an overlap between plot boundaries."""

    overlap_id: str = Field(
        ...,
        max_length=128,
        description="Overlap record identifier to resolve",
    )
    plot_id_a: str = Field(..., description="First plot in the overlap pair")
    plot_id_b: str = Field(..., description="Second plot in the overlap pair")
    preferred_resolution: Optional[str] = Field(
        None,
        description="Preferred resolution strategy",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Version Schemas
# =============================================================================


class VersionSchema(BaseModel):
    """A single boundary version record."""

    plot_id: str = Field(..., description="Plot identifier")
    version_number: int = Field(..., ge=1, description="Version number (1-based)")
    change_reason: str = Field(
        ...,
        description="Reason for the boundary version change",
    )
    changed_by: str = Field(
        ...,
        description="User or system identifier that made the change",
    )
    changed_at: datetime = Field(
        ...,
        description="Timestamp of the version change (UTC)",
    )
    geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="Boundary geometry at this version",
    )
    area_m2: float = Field(0.0, ge=0.0, description="Area at this version")
    area_hectares: float = Field(
        0.0,
        ge=0.0,
        description="Area in hectares at this version",
    )
    area_diff_m2: float = Field(
        0.0,
        description="Area change from previous version in square metres",
    )
    area_diff_pct: float = Field(
        0.0,
        description="Area change from previous version as percentage",
    )
    vertex_count: int = Field(0, ge=0, description="Vertex count at this version")
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash for audit trail",
    )
    parent_provenance_hash: Optional[str] = Field(
        None,
        description="Provenance hash of the previous version",
    )

    model_config = ConfigDict(from_attributes=True)


class VersionHistoryResponseSchema(BaseModel):
    """Version history for a plot boundary."""

    plot_id: str = Field(..., description="Plot identifier")
    total_versions: int = Field(
        ...,
        ge=0,
        description="Total number of versions",
    )
    current_version: int = Field(
        ...,
        ge=1,
        description="Current (latest) version number",
    )
    versions: List[VersionSchema] = Field(
        default_factory=list,
        description="List of version records (newest first)",
    )

    model_config = ConfigDict(from_attributes=True)


class VersionAtDateRequestSchema(BaseModel):
    """Request to retrieve boundary version at a specific date."""

    date: str = Field(
        ...,
        description="ISO 8601 date string (e.g. 2024-12-31 or 2024-12-31T00:00:00Z)",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate ISO 8601 date format."""
        v = v.strip()
        if not v:
            raise ValueError("date must not be empty")
        # Accept both date-only and datetime formats
        from datetime import date as date_cls

        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                     "%Y-%m-%dT%H:%M:%S%z"):
            try:
                datetime.strptime(v.replace("+00:00", "Z").rstrip("Z"), fmt.rstrip("Z").rstrip("%z"))
                return v
            except ValueError:
                continue
        # Try basic date parse
        try:
            date_cls.fromisoformat(v[:10])
            return v
        except (ValueError, IndexError):
            raise ValueError(
                f"date must be a valid ISO 8601 string, got '{v}'"
            )


class VersionDiffRequestSchema(BaseModel):
    """Request to compute diff between two boundary versions."""

    version_a: int = Field(
        ...,
        ge=1,
        description="First version number for comparison",
    )
    version_b: int = Field(
        ...,
        ge=1,
        description="Second version number for comparison",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_different_versions(self) -> "VersionDiffRequestSchema":
        """Ensure version_a and version_b are different."""
        if self.version_a == self.version_b:
            raise ValueError(
                f"version_a and version_b must differ, both are {self.version_a}"
            )
        return self


class VersionDiffResponseSchema(BaseModel):
    """Response for a version diff between two boundary versions."""

    plot_id: str = Field(..., description="Plot identifier")
    version_a: int = Field(..., ge=1, description="First version number")
    version_b: int = Field(..., ge=1, description="Second version number")
    added_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Area added between versions (square metres)",
    )
    removed_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Area removed between versions (square metres)",
    )
    unchanged_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Area unchanged between versions (square metres)",
    )
    net_area_change_m2: float = Field(
        ...,
        description="Net area change (added - removed) in square metres",
    )
    net_area_change_pct: float = Field(
        ...,
        description="Net area change as percentage of version_a area",
    )
    geometry_a: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="Geometry at version_a",
    )
    geometry_b: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="Geometry at version_b",
    )
    diff_geometry: Optional[GeoJSONGeometrySchema] = Field(
        None,
        description="Symmetric difference geometry between versions",
    )

    model_config = ConfigDict(from_attributes=True)


class VersionLineageResponseSchema(BaseModel):
    """Complete version lineage for a plot boundary."""

    plot_id: str = Field(..., description="Plot identifier")
    total_versions: int = Field(
        ...,
        ge=0,
        description="Total versions in lineage",
    )
    lineage: List[VersionSchema] = Field(
        default_factory=list,
        description="Ordered version lineage (oldest first)",
    )
    provenance_chain_valid: bool = Field(
        ...,
        description="Whether the provenance hash chain is intact",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Export Schemas
# =============================================================================


class ExportRequestSchema(BaseModel):
    """Request to export plot boundaries in a specific format."""

    plot_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of plot IDs to export (max 10,000)",
    )
    format: ExportFormatSchema = Field(
        ...,
        description="Export output format",
    )
    precision: int = Field(
        8,
        ge=1,
        le=15,
        description="Coordinate precision (decimal places)",
    )
    simplify: bool = Field(
        False,
        description="Apply simplification before export",
    )
    simplification_tolerance: Optional[float] = Field(
        None,
        gt=0.0,
        description="Simplification tolerance in degrees (required if simplify=True)",
    )
    simplification_method: SimplificationMethodSchema = Field(
        SimplificationMethodSchema.DOUGLAS_PEUCKER,
        description="Simplification algorithm to use",
    )
    include_metadata: bool = Field(
        True,
        description="Include boundary metadata in export",
    )
    include_area: bool = Field(
        True,
        description="Include computed area in export properties",
    )

    model_config = ConfigDict(extra="forbid")


class ExportResponseSchema(BaseModel):
    """Response for a boundary export operation."""

    export_id: str = Field(
        default_factory=lambda: f"exp-{uuid.uuid4().hex[:12]}",
        description="Unique export identifier",
    )
    format: str = Field(..., description="Export format used")
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Size of the exported file in bytes",
    )
    plot_count: int = Field(..., ge=0, description="Number of plots exported")
    crs: str = Field(
        "EPSG:4326",
        description="Coordinate reference system of exported data",
    )
    download_url: Optional[str] = Field(
        None,
        description="Download URL for the exported file (for large exports)",
    )
    data: Optional[str] = Field(
        None,
        description="Inline export data (for small exports)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Export timestamp (UTC)",
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Expiration timestamp for download URL",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchExportRequestSchema(BaseModel):
    """Request to export boundaries in multiple formats."""

    plot_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of plot IDs to export",
    )
    formats: List[ExportFormatSchema] = Field(
        ...,
        min_length=1,
        max_length=8,
        description="List of export formats",
    )
    precision: int = Field(
        8,
        ge=1,
        le=15,
        description="Coordinate precision",
    )
    simplify: bool = Field(False, description="Apply simplification")
    simplification_tolerance: Optional[float] = Field(
        None,
        gt=0.0,
        description="Simplification tolerance in degrees",
    )

    model_config = ConfigDict(extra="forbid")


class BatchExportResponseSchema(BaseModel):
    """Response for a multi-format batch export."""

    batch_export_id: str = Field(
        default_factory=lambda: f"bexp-{uuid.uuid4().hex[:12]}",
        description="Batch export identifier",
    )
    exports: List[ExportResponseSchema] = Field(
        default_factory=list,
        description="Per-format export results",
    )
    total_formats: int = Field(
        ...,
        ge=0,
        description="Number of formats exported",
    )
    plot_count: int = Field(
        ...,
        ge=0,
        description="Number of plots exported",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceReportSchema(BaseModel):
    """EUDR compliance report for boundary data quality."""

    report_id: str = Field(
        default_factory=lambda: f"rpt-{uuid.uuid4().hex[:12]}",
        description="Unique report identifier",
    )
    total_plots: int = Field(..., ge=0, description="Total plots evaluated")
    valid_plots: int = Field(
        ...,
        ge=0,
        description="Plots with valid geometries",
    )
    invalid_plots: int = Field(
        ...,
        ge=0,
        description="Plots with invalid geometries",
    )
    polygon_required: int = Field(
        ...,
        ge=0,
        description="Plots above 4ha threshold requiring polygon boundaries",
    )
    point_sufficient: int = Field(
        ...,
        ge=0,
        description="Plots below 4ha threshold where point is sufficient",
    )
    commodity_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot count by commodity",
    )
    country_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Plot count by country",
    )
    overlap_count: int = Field(
        0,
        ge=0,
        description="Total overlapping boundary pairs",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Compliance improvement recommendations",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Report generation timestamp",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Split/Merge Schemas
# =============================================================================


class SplitRequestSchema(BaseModel):
    """Request to split a plot boundary along a cutting line."""

    plot_id: str = Field(
        ...,
        max_length=128,
        description="Plot identifier to split",
    )
    cutting_line: List[CoordinateSchema] = Field(
        ...,
        min_length=2,
        description="Cutting line vertices (minimum 2 points)",
    )
    change_reason: str = Field(
        "split_operation",
        description="Reason for the split operation",
    )

    model_config = ConfigDict(extra="forbid")


class MergeRequestSchema(BaseModel):
    """Request to merge multiple plot boundaries into one."""

    plot_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Plot identifiers to merge (minimum 2)",
    )
    merged_plot_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Identifier for the merged plot (auto-generated if omitted)",
    )
    change_reason: str = Field(
        "merge_operation",
        description="Reason for the merge operation",
    )

    model_config = ConfigDict(extra="forbid")


class AreaConservationSchema(BaseModel):
    """Area conservation check for split/merge operations."""

    original_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Total area of original boundary/boundaries",
    )
    result_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Total area of resulting boundary/boundaries",
    )
    difference_m2: float = Field(
        ...,
        description="Area difference (result - original) in square metres",
    )
    difference_pct: float = Field(
        ...,
        description="Area difference as percentage of original",
    )
    within_tolerance: bool = Field(
        ...,
        description="Whether difference is within configured tolerance",
    )

    model_config = ConfigDict(from_attributes=True)


class SplitResponseSchema(BaseModel):
    """Response for a boundary split operation."""

    parent_plot_id: str = Field(
        ...,
        description="Original plot that was split",
    )
    child_plot_ids: List[str] = Field(
        ...,
        description="Identifiers of resulting child plots",
    )
    child_boundaries: List[BoundaryResponseSchema] = Field(
        default_factory=list,
        description="Resulting child boundary details",
    )
    area_conservation: AreaConservationSchema = Field(
        ...,
        description="Area conservation verification",
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash of the split operation",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class MergeResponseSchema(BaseModel):
    """Response for a boundary merge operation."""

    parent_plot_ids: List[str] = Field(
        ...,
        description="Original plot identifiers that were merged",
    )
    merged_plot_id: str = Field(
        ...,
        description="Identifier of the resulting merged plot",
    )
    merged_boundary: Optional[BoundaryResponseSchema] = Field(
        None,
        description="Resulting merged boundary details",
    )
    area_conservation: AreaConservationSchema = Field(
        ...,
        description="Area conservation verification",
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash of the merge operation",
    )
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class GenealogyOperationSchema(BaseModel):
    """A single operation in the split/merge genealogy."""

    operation_id: str = Field(
        ...,
        description="Unique operation identifier",
    )
    operation_type: str = Field(
        ...,
        description="Operation type: 'split' or 'merge'",
    )
    input_plot_ids: List[str] = Field(
        ...,
        description="Plot IDs input to the operation",
    )
    output_plot_ids: List[str] = Field(
        ...,
        description="Plot IDs output from the operation",
    )
    performed_at: datetime = Field(
        ...,
        description="Operation timestamp (UTC)",
    )
    performed_by: str = Field(
        ...,
        description="User who performed the operation",
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 provenance hash of the operation",
    )

    model_config = ConfigDict(from_attributes=True)


class GenealogyResponseSchema(BaseModel):
    """Split/merge genealogy for a plot boundary."""

    plot_id: str = Field(..., description="Subject plot identifier")
    parents: List[str] = Field(
        default_factory=list,
        description="Parent plot IDs (if this plot resulted from split/merge)",
    )
    children: List[str] = Field(
        default_factory=list,
        description="Child plot IDs (if this plot was split)",
    )
    operations: List[GenealogyOperationSchema] = Field(
        default_factory=list,
        description="Ordered list of genealogy operations",
    )
    lineage_depth: int = Field(
        0,
        ge=0,
        description="Depth of the genealogy tree",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch/Health Schemas
# =============================================================================


class BatchJobRequestSchema(BaseModel):
    """Request to submit a batch processing job."""

    operation: str = Field(
        ...,
        description="Batch operation type (validate, area, overlap_scan, export)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation-specific parameters",
    )
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Job priority (1=highest, 10=lowest)",
    )
    callback_url: Optional[str] = Field(
        None,
        description="Webhook URL to notify upon completion",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation type."""
        allowed = {"validate", "area", "overlap_scan", "export", "repair"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"operation must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower


class BatchJobResponseSchema(BaseModel):
    """Response for a batch job submission or status query."""

    batch_id: str = Field(
        default_factory=lambda: f"batch-{uuid.uuid4().hex[:12]}",
        description="Unique batch job identifier",
    )
    status: BatchStatusSchema = Field(
        ...,
        description="Current batch job status",
    )
    operation: str = Field(..., description="Batch operation type")
    progress_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage",
    )
    total_items: int = Field(0, ge=0, description="Total items in the batch")
    processed_items: int = Field(0, ge=0, description="Items processed so far")
    failed_items: int = Field(0, ge=0, description="Items that failed")
    results_url: Optional[str] = Field(
        None,
        description="URL to download results when completed",
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow,
        description="Job submission timestamp",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Job completion timestamp",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if batch failed",
    )

    model_config = ConfigDict(from_attributes=True)


class EngineHealthSchema(BaseModel):
    """Health status of a single engine component."""

    name: str = Field(..., description="Engine name")
    status: str = Field(..., description="Engine status (healthy, degraded, error)")
    version: str = Field("1.0.0", description="Engine version")
    last_check: Optional[datetime] = Field(
        None,
        description="Last health check timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class HealthResponseSchema(BaseModel):
    """Service health response for load balancers and monitoring."""

    status: str = Field(..., description="Overall service status")
    agent: str = Field(
        "AGENT-EUDR-006",
        description="Agent identifier",
    )
    agent_name: str = Field(
        "Plot Boundary Manager",
        description="Human-readable agent name",
    )
    component: str = Field(
        "Plot Boundary Manager",
        description="Component identifier",
    )
    version: str = Field("1.0.0", description="API version")
    engines: Dict[str, str] = Field(
        default_factory=dict,
        description="Engine health status map",
    )
    boundary_count: int = Field(
        0,
        ge=0,
        description="Total boundaries in the registry",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Health check timestamp (UTC)",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# API Response Wrappers
# =============================================================================


class ApiResponse(BaseModel):
    """Standard API success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Response timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Pagination
# =============================================================================


class PaginatedMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "GeometryTypeSchema",
    "ValidationErrorTypeSchema",
    "OverlapSeveritySchema",
    "VersionChangeReasonSchema",
    "SimplificationMethodSchema",
    "ExportFormatSchema",
    "ThresholdClassificationSchema",
    "BatchStatusSchema",
    # Coordinate/Geometry
    "CoordinateSchema",
    "BoundingBoxSchema",
    "RingSchema",
    "GeoJSONGeometrySchema",
    # Boundary
    "CreateBoundaryRequestSchema",
    "UpdateBoundaryRequestSchema",
    "CentroidSchema",
    "CompactnessMetricsSchema",
    "BoundaryResponseSchema",
    "BoundaryListResponseSchema",
    "BatchCreateRequestSchema",
    "BatchCreateResultSchema",
    "BatchCreateResponseSchema",
    "SearchRequestSchema",
    # Validation
    "ValidateRequestSchema",
    "RepairRequestSchema",
    "ValidationErrorSchema",
    "RepairActionSchema",
    "ValidationResponseSchema",
    "BatchValidateRequestSchema",
    "BatchValidateResultSchema",
    "BatchValidateResponseSchema",
    # Area
    "AreaCalculationRequestSchema",
    "AreaResponseSchema",
    "ThresholdResponseSchema",
    "BatchAreaRequestSchema",
    "BatchAreaResultSchema",
    "BatchAreaResponseSchema",
    # Overlap
    "OverlapDetectRequestSchema",
    "OverlapScanRequestSchema",
    "OverlapRecordSchema",
    "OverlapResponseSchema",
    "OverlapScanResponseSchema",
    "OverlapResolutionSchema",
    "OverlapResolutionRequestSchema",
    # Version
    "VersionSchema",
    "VersionHistoryResponseSchema",
    "VersionAtDateRequestSchema",
    "VersionDiffRequestSchema",
    "VersionDiffResponseSchema",
    "VersionLineageResponseSchema",
    # Export
    "ExportRequestSchema",
    "ExportResponseSchema",
    "BatchExportRequestSchema",
    "BatchExportResponseSchema",
    "ComplianceReportSchema",
    # Split/Merge
    "SplitRequestSchema",
    "MergeRequestSchema",
    "AreaConservationSchema",
    "SplitResponseSchema",
    "MergeResponseSchema",
    "GenealogyOperationSchema",
    "GenealogyResponseSchema",
    # Batch/Health
    "BatchJobRequestSchema",
    "BatchJobResponseSchema",
    "EngineHealthSchema",
    "HealthResponseSchema",
    # Wrappers
    "ApiResponse",
    "ErrorResponse",
    "PaginatedMeta",
]
