# -*- coding: utf-8 -*-
"""
Plot Boundary Manager Data Models - AGENT-EUDR-006

Pydantic v2 data models for the Plot Boundary Manager Agent covering
polygon management, boundary validation, geodetic area calculation,
overlap detection, boundary versioning, simplification, split/merge
operations, and compliance export for EU Deforestation Regulation (EUDR)
Articles 9, 10, and 31 compliance.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all plot boundary management operations.

Enumerations (12):
    - GeometryType, CoordinateReferenceSystem, ValidationErrorType,
      RepairStrategy, OverlapSeverity, OverlapResolution,
      VersionChangeReason, SimplificationMethod, ExportFormat,
      ThresholdClassification, CompactnessIndex, BatchStatus

Core Models (14):
    - Coordinate, BoundingBox, Ring, PlotBoundary,
      ValidationError, ValidationResult, AreaResult,
      OverlapRecord, BoundaryVersion, SimplificationResult,
      SplitResult, MergeResult, ExportResult, BatchJob

Request Models (11):
    - CreateBoundaryRequest, UpdateBoundaryRequest, ValidateRequest,
      RepairRequest, AreaCalculationRequest, OverlapDetectionRequest,
      SimplifyRequest, SplitRequest, MergeRequest, ExportRequest,
      BatchBoundaryRequest

Response Models (8):
    - BoundaryResponse, ValidationResponse, AreaResponse,
      OverlapResponse, VersionResponse, SimplificationResponse,
      SplitMergeResponse, ExportResponse

Compatibility:
    Imports EUDRCommodity from greenlang.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, AGENT-EUDR-002
    Geolocation Verification, AGENT-EUDR-003 Satellite Monitoring,
    AGENT-EUDR-004 Forest Cover Analysis, and AGENT-EUDR-005
    Land Use Change Detector.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from greenlang.eudr_traceability.models import EUDRCommodity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR area threshold in hectares for polygon requirement (Article 9).
EUDR_AREA_THRESHOLD_HA: float = 4.0

#: Maximum number of boundaries in a single batch request.
MAX_BATCH_SIZE: int = 10000

#: Default coordinate precision (decimal places).
DEFAULT_COORDINATE_PRECISION: int = 8

#: WGS84 semi-major axis in metres.
WGS84_SEMI_MAJOR_AXIS: float = 6378137.0

#: WGS84 flattening.
WGS84_FLATTENING: float = 1.0 / 298.257223563

#: Minimum vertices for a valid polygon (closed triangle).
MIN_POLYGON_VERTICES: int = 4


# =============================================================================
# Enumerations
# =============================================================================


class GeometryType(str, Enum):
    """Geometry type classification for plot boundaries.

    Defines the supported geometry types for plot boundary
    representation per OGC Simple Features specification and EUDR
    requirements.

    POINT: A single coordinate pair (longitude, latitude). Used for
        plots below the 4-hectare EUDR threshold per Article 9 where
        a full polygon boundary is not required.
    POLYGON: A closed ring of coordinates forming a simple polygon with
        optional holes. The primary geometry type for EUDR plot
        boundaries above the 4-hectare threshold.
    MULTI_POLYGON: A collection of non-overlapping polygons treated as
        a single boundary. Used for plots with disjoint parcels or
        complex shapes with islands.
    LINE_STRING: An ordered sequence of coordinates forming a line.
        Used for cutting lines in split operations and boundary
        segments.
    GEOMETRY_COLLECTION: A heterogeneous collection of geometry types.
        Used for complex boundary representations requiring multiple
        geometry types.
    """

    POINT = "point"
    POLYGON = "polygon"
    MULTI_POLYGON = "multi_polygon"
    LINE_STRING = "line_string"
    GEOMETRY_COLLECTION = "geometry_collection"


class CoordinateReferenceSystem(str, Enum):
    """Coordinate reference system identifiers for boundary data.

    Enumerates the primary CRS families supported for input and output
    of plot boundary data. Individual EPSG codes within each family
    are validated against the config's supported_input_crs list.

    WGS84: World Geodetic System 1984 (EPSG:4326). The canonical CRS
        for GreenLang internal storage and EUDR regulatory submission.
    UTM_NORTH: Universal Transverse Mercator northern hemisphere zones.
        Used for metric area calculations in the northern hemisphere.
    UTM_SOUTH: Universal Transverse Mercator southern hemisphere zones.
        Used for metric area calculations in the southern hemisphere.
    WEB_MERCATOR: Web Mercator projection (EPSG:3857). Common in web
        mapping applications. Area calculations require reprojection.
    SIRGAS_2000: South American Datum 2000 (EPSG:4674). Used in Brazil,
        Colombia, and other South American countries.
    ETRS89: European Terrestrial Reference System 1989 (EPSG:4258).
        The standard datum for European Union member states.
    NAD83: North American Datum 1983 (EPSG:4269). Used in the United
        States, Canada, and Mexico.
    GDA2020: Geocentric Datum of Australia 2020 (EPSG:7844). The
        current official datum for Australia.
    JGD2011: Japanese Geodetic Datum 2011. Used in Japan for
        cadastral and land administration purposes.
    INDIAN_1975: Indian Datum 1975. Used in parts of South and
        Southeast Asia.
    HONG_KONG_1980: Hong Kong 1980 Grid System. Used for cadastral
        mapping in Hong Kong.
    NZGD2000: New Zealand Geodetic Datum 2000. Official datum for
        New Zealand.
    SAD69: South American Datum 1969. Legacy datum still used in
        some South American datasets.
    PULKOVO_1942: Pulkovo 1942 datum. Used in Russia and former
        Soviet states.
    BEIJING_1954: Beijing 1954 coordinate system. Legacy datum used
        in China.
    CGCS2000: China Geodetic Coordinate System 2000. Current official
        datum for China.
    PRS92: Philippine Reference System of 1992. Official datum for
        the Philippines.
    GRS80: Geodetic Reference System 1980. The reference ellipsoid
        underlying many modern datums.
    """

    WGS84 = "wgs84"
    UTM_NORTH = "utm_north"
    UTM_SOUTH = "utm_south"
    WEB_MERCATOR = "web_mercator"
    SIRGAS_2000 = "sirgas_2000"
    ETRS89 = "etrs89"
    NAD83 = "nad83"
    GDA2020 = "gda2020"
    JGD2011 = "jgd2011"
    INDIAN_1975 = "indian_1975"
    HONG_KONG_1980 = "hong_kong_1980"
    NZGD2000 = "nzgd2000"
    SAD69 = "sad69"
    PULKOVO_1942 = "pulkovo_1942"
    BEIJING_1954 = "beijing_1954"
    CGCS2000 = "cgcs2000"
    PRS92 = "prs92"
    GRS80 = "grs80"


class ValidationErrorType(str, Enum):
    """Geometry validation error type classification.

    Categorizes the types of geometric errors that can be detected
    during boundary validation per OGC Simple Features and ISO 19107
    standards.

    SELF_INTERSECTION: The polygon boundary crosses itself, creating
        an invalid topology. Common in hand-digitized boundaries.
    UNCLOSED_RING: The first and last coordinates of a ring do not
        match within the closure tolerance.
    DUPLICATE_VERTICES: Consecutive vertices are within the duplicate
        vertex tolerance, creating unnecessary complexity.
    SPIKE: An interior angle at a vertex is below the spike threshold,
        creating an extremely narrow protrusion.
    SLIVER: The polygon has an extremely high aspect ratio (length
        to width), indicating a narrow strip.
    WRONG_ORIENTATION: The exterior ring has clockwise orientation
        (should be counter-clockwise per OGC) or an interior ring has
        counter-clockwise orientation (should be clockwise).
    INVALID_COORDINATES: One or more coordinates are outside valid
        WGS84 ranges or contain NaN/Infinity values.
    TOO_FEW_VERTICES: The ring has fewer than the minimum required
        vertices (4 for a valid closed polygon).
    HOLE_OUTSIDE_SHELL: An interior ring (hole) extends partially or
        fully outside the exterior ring boundary.
    OVERLAPPING_HOLES: Two or more interior rings overlap each other,
        creating ambiguous interior topology.
    NESTED_SHELLS: Multiple exterior rings are nested inside each
        other when they should be separate polygons in a MultiPolygon.
    ZERO_AREA: The polygon has zero or near-zero area, typically
        caused by collinear vertices or a degenerate geometry.
    """

    SELF_INTERSECTION = "self_intersection"
    UNCLOSED_RING = "unclosed_ring"
    DUPLICATE_VERTICES = "duplicate_vertices"
    SPIKE = "spike"
    SLIVER = "sliver"
    WRONG_ORIENTATION = "wrong_orientation"
    INVALID_COORDINATES = "invalid_coordinates"
    TOO_FEW_VERTICES = "too_few_vertices"
    HOLE_OUTSIDE_SHELL = "hole_outside_shell"
    OVERLAPPING_HOLES = "overlapping_holes"
    NESTED_SHELLS = "nested_shells"
    ZERO_AREA = "zero_area"


class RepairStrategy(str, Enum):
    """Geometry repair strategy classification.

    Defines the automatic repair methods available for fixing
    detected geometry errors.

    NODE_INSERTION: Insert new vertices at self-intersection points
        to split the boundary into valid sub-polygons.
    RING_CLOSURE: Close an unclosed ring by appending a copy of the
        first vertex as the last vertex.
    VERTEX_REMOVAL: Remove duplicate or problematic vertices.
    SPIKE_REMOVAL: Remove vertices that form spikes (extremely
        narrow angles).
    ORIENTATION_REVERSAL: Reverse the vertex order of a ring to
        correct its orientation (CW to CCW or vice versa).
    HOLE_REMOVAL: Remove interior rings that violate containment
        rules.
    CONVEX_HULL_FALLBACK: Replace the invalid polygon with its
        convex hull. Last resort when other repairs fail.
    INTERPOLATION: Interpolate new vertices between existing ones
        to smooth irregular boundary sections.
    """

    NODE_INSERTION = "node_insertion"
    RING_CLOSURE = "ring_closure"
    VERTEX_REMOVAL = "vertex_removal"
    SPIKE_REMOVAL = "spike_removal"
    ORIENTATION_REVERSAL = "orientation_reversal"
    HOLE_REMOVAL = "hole_removal"
    CONVEX_HULL_FALLBACK = "convex_hull_fallback"
    INTERPOLATION = "interpolation"


class OverlapSeverity(str, Enum):
    """Overlap severity classification.

    Classifies the severity of detected boundary overlaps based
    on the overlap area as a fraction of the smaller polygon.

    MINOR: Overlap less than 1% of the smaller polygon area.
    MODERATE: Overlap between 1% and 10%.
    MAJOR: Overlap between 10% and 50%.
    CRITICAL: Overlap exceeds 50%.
    """

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class OverlapResolution(str, Enum):
    """Overlap resolution strategy.

    Defines methods for resolving detected boundary overlaps.

    BOUNDARY_ADJUSTMENT: Adjust the shared boundary to eliminate
        overlap by snapping to a common edge.
    PRIORITY_ASSIGNMENT: Assign the overlap area to one plot based
        on priority rules.
    ARBITRATION: Flag for manual arbitration by a qualified surveyor.
    SPLIT: Split the overlap area into separate parcels.
    MERGE: Merge the overlapping plots into a single boundary.
    MANUAL_REVIEW: Escalate to manual review without automated
        resolution.
    """

    BOUNDARY_ADJUSTMENT = "boundary_adjustment"
    PRIORITY_ASSIGNMENT = "priority_assignment"
    ARBITRATION = "arbitration"
    SPLIT = "split"
    MERGE = "merge"
    MANUAL_REVIEW = "manual_review"


class VersionChangeReason(str, Enum):
    """Boundary version change reason classification.

    Documents the reason for creating a new version of a plot
    boundary, supporting EUDR Article 31 audit trail requirements.

    SURVEY_UPDATE: New survey data provides a more accurate boundary.
    SPLIT: The plot was split into sub-plots.
    MERGE: Plots were merged into a single boundary.
    CORRECTION: Correction of a previously identified boundary error.
    SEASONAL: Seasonal boundary adjustment reflecting natural changes.
    INITIAL: Initial boundary creation.
    REPAIR: Automated geometry repair was applied.
    IMPORT: Boundary imported from an external data source.
    """

    SURVEY_UPDATE = "survey_update"
    SPLIT = "split"
    MERGE = "merge"
    CORRECTION = "correction"
    SEASONAL = "seasonal"
    INITIAL = "initial"
    REPAIR = "repair"
    IMPORT = "import"


class SimplificationMethod(str, Enum):
    """Polygon simplification algorithm selection.

    DOUGLAS_PEUCKER: Ramer-Douglas-Peucker algorithm. Iteratively
        removes vertices that deviate less than the tolerance.
    VISVALINGAM_WHYATT: Visvalingam-Whyatt algorithm. Iteratively
        removes the vertex forming the smallest area triangle.
    TOPOLOGY_PRESERVING: Simplification that preserves topological
        relationships between adjacent polygons.
    """

    DOUGLAS_PEUCKER = "douglas_peucker"
    VISVALINGAM_WHYATT = "visvalingam_whyatt"
    TOPOLOGY_PRESERVING = "topology_preserving"


class ExportFormat(str, Enum):
    """Boundary export format selection.

    GEOJSON: GeoJSON format (RFC 7946). Default exchange format.
    KML: Keyhole Markup Language for Google Earth compatibility.
    WKT: Well-Known Text per OGC Simple Features.
    WKB: Well-Known Binary per OGC Simple Features.
    SHAPEFILE: ESRI Shapefile format.
    EUDR_XML: EUDR-specific XML schema for regulatory submission.
    GPX: GPS Exchange Format.
    GML: Geography Markup Language (OGC standard).
    """

    GEOJSON = "geojson"
    KML = "kml"
    WKT = "wkt"
    WKB = "wkb"
    SHAPEFILE = "shapefile"
    EUDR_XML = "eudr_xml"
    GPX = "gpx"
    GML = "gml"


class ThresholdClassification(str, Enum):
    """EUDR area threshold classification result.

    POLYGON_REQUIRED: Plot area exceeds 4.0 hectares. Full polygon
        boundary required for DDS submission.
    POINT_SUFFICIENT: Plot area at or below 4.0 hectares. A single
        geolocation point is sufficient.
    """

    POLYGON_REQUIRED = "polygon_required"
    POINT_SUFFICIENT = "point_sufficient"


class CompactnessIndex(str, Enum):
    """Polygon compactness index measurement method.

    POLSBY_POPPER: 4 * pi * area / perimeter^2.
        1.0 = perfect circle.
    SCHWARTZBERG: perimeter / (2 * pi * sqrt(area / pi)).
        1.0 = perfect circle.
    CONVEX_HULL_RATIO: area / convex_hull_area.
        1.0 = convex polygon.
    """

    POLSBY_POPPER = "polsby_popper"
    SCHWARTZBERG = "schwartzberg"
    CONVEX_HULL_RATIO = "convex_hull_ratio"


class BatchStatus(str, Enum):
    """Batch boundary operation status.

    PENDING: Queued for execution.
    RUNNING: Currently being processed.
    COMPLETED: Finished successfully.
    FAILED: Fatal error encountered.
    CANCELLED: Cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Core Data Models
# =============================================================================


class Coordinate(BaseModel):
    """A geographic coordinate with optional metadata.

    Attributes:
        lat: Latitude in decimal degrees (-90.0 to 90.0).
        lon: Longitude in decimal degrees (-180.0 to 180.0).
        altitude: Optional altitude in metres above WGS84 ellipsoid.
        accuracy_m: Optional horizontal accuracy in metres (1-sigma).
        timestamp: Optional timestamp when the coordinate was captured.
    """

    model_config = ConfigDict(from_attributes=True)

    lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees (-90 to 90)",
    )
    lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees (-180 to 180)",
    )
    altitude: Optional[float] = Field(
        None,
        description="Altitude above WGS84 ellipsoid (metres)",
    )
    accuracy_m: Optional[float] = Field(
        None,
        ge=0.0,
        description="Horizontal accuracy (metres, 1-sigma)",
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Coordinate capture timestamp (UTC)",
    )

    @model_validator(mode="after")
    def _validate_coordinate_values(self) -> "Coordinate":
        """Validate coordinate values are finite numbers."""
        if not math.isfinite(self.lat):
            raise ValueError(f"lat must be finite, got {self.lat}")
        if not math.isfinite(self.lon):
            raise ValueError(f"lon must be finite, got {self.lon}")
        if self.altitude is not None and not math.isfinite(self.altitude):
            raise ValueError(
                f"altitude must be finite, got {self.altitude}"
            )
        return self


class BoundingBox(BaseModel):
    """Axis-aligned bounding box for spatial queries.

    Attributes:
        min_lat: Southern boundary latitude.
        min_lon: Western boundary longitude.
        max_lat: Northern boundary latitude.
        max_lon: Eastern boundary longitude.
    """

    model_config = ConfigDict(from_attributes=True)

    min_lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Southern boundary latitude",
    )
    min_lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Western boundary longitude",
    )
    max_lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Northern boundary latitude",
    )
    max_lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Eastern boundary longitude",
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> "BoundingBox":
        """Validate min <= max for latitude and longitude."""
        if self.min_lat > self.max_lat:
            raise ValueError(
                f"min_lat ({self.min_lat}) must be <= max_lat ({self.max_lat})"
            )
        if self.min_lon > self.max_lon:
            raise ValueError(
                f"min_lon ({self.min_lon}) must be <= max_lon ({self.max_lon})"
            )
        return self

    def contains(self, lat: float, lon: float) -> bool:
        """Check whether a point is within this bounding box.

        Args:
            lat: Latitude of the point.
            lon: Longitude of the point.

        Returns:
            True if the point is inside (inclusive) the bounding box.
        """
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Check whether this bounding box intersects another.

        Args:
            other: Another BoundingBox to test against.

        Returns:
            True if the two bounding boxes overlap.
        """
        return not (
            self.max_lat < other.min_lat
            or self.min_lat > other.max_lat
            or self.max_lon < other.min_lon
            or self.min_lon > other.max_lon
        )

    def area(self) -> float:
        """Approximate area of the bounding box in square degrees.

        Returns:
            Area in square degrees (lat * lon extent).
        """
        return (self.max_lat - self.min_lat) * (self.max_lon - self.min_lon)


class Ring(BaseModel):
    """A ring of coordinates forming a closed polygon boundary.

    Attributes:
        coordinates: Ordered list of coordinates forming the ring.
            The first and last coordinates must be identical (closed).
        is_exterior: Whether this is an exterior ring (True) or an
            interior ring / hole (False).
    """

    model_config = ConfigDict(from_attributes=True)

    coordinates: List[Coordinate] = Field(
        ...,
        min_length=4,
        description="Ordered coordinates forming the ring (min 4)",
    )
    is_exterior: bool = Field(
        default=True,
        description="True for exterior ring, False for hole",
    )

    def is_closed(self) -> bool:
        """Check whether the ring is closed (first == last coordinate).

        Returns:
            True if the first and last coordinates match exactly.
        """
        if len(self.coordinates) < 2:
            return False
        first = self.coordinates[0]
        last = self.coordinates[-1]
        return first.lat == last.lat and first.lon == last.lon

    def orientation(self) -> str:
        """Determine the orientation using the shoelace formula.

        Returns:
            'ccw' for counter-clockwise, 'cw' for clockwise,
            'collinear' if the signed area is zero.
        """
        signed = self.area_signed()
        if signed > 0:
            return "ccw"
        elif signed < 0:
            return "cw"
        return "collinear"

    def area_signed(self) -> float:
        """Compute the signed area using the shoelace formula.

        Positive for counter-clockwise, negative for clockwise.
        Units are square degrees (approximate).

        Returns:
            Signed area in square degrees.
        """
        coords = self.coordinates
        n = len(coords)
        if n < 3:
            return 0.0
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += coords[i].lon * coords[j].lat
            total -= coords[j].lon * coords[i].lat
        return total / 2.0


class PlotBoundary(BaseModel):
    """Complete plot boundary with geometry, metadata, and metrics.

    The primary data model for EUDR plot boundary management.

    Attributes:
        plot_id: Unique plot identifier (UUID or external reference).
        geometry_type: Type of geometry representing the boundary.
        exterior_ring: Exterior ring coordinates for the boundary.
        holes: List of interior rings (holes) within the boundary.
        crs: Coordinate reference system of the boundary data.
        metadata: Additional key-value metadata for the boundary.
        commodity: EUDR-regulated commodity grown on this plot.
        country_iso: ISO 3166-1 alpha-2 country code for the plot.
        owner_id: Owner or operator identifier for the plot.
        certification_id: External certification reference.
        created_at: UTC timestamp when the boundary was created.
        updated_at: UTC timestamp of the most recent boundary update.
        version: Boundary version number (incrementing integer).
        is_active: Whether this is the current active boundary version.
        centroid: Computed centroid of the boundary geometry.
        bounding_box: Axis-aligned bounding box of the boundary.
        area_hectares: Geodetic area of the boundary in hectares.
        perimeter_meters: Geodetic perimeter in metres.
        vertex_count: Total number of vertices in all rings.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        min_length=1,
        description="Unique plot identifier",
    )
    geometry_type: GeometryType = Field(
        default=GeometryType.POLYGON,
        description="Geometry type of the boundary",
    )
    exterior_ring: Optional[Ring] = Field(
        None,
        description="Exterior ring coordinates",
    )
    holes: List[Ring] = Field(
        default_factory=list,
        description="Interior rings (holes)",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate reference system",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata key-value pairs",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR-regulated commodity on this plot",
    )
    country_iso: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
    )
    owner_id: str = Field(
        default="",
        description="Owner or operator identifier",
    )
    certification_id: str = Field(
        default="",
        description="External certification reference",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Boundary creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Most recent update timestamp (UTC)",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Boundary version number",
    )
    is_active: bool = Field(
        default=True,
        description="Whether this is the active version",
    )
    centroid: Optional[Coordinate] = Field(
        None,
        description="Computed boundary centroid",
    )
    bounding_box: Optional[BoundingBox] = Field(
        None,
        description="Axis-aligned bounding box",
    )
    area_hectares: float = Field(
        default=0.0,
        ge=0.0,
        description="Geodetic area in hectares",
    )
    perimeter_meters: float = Field(
        default=0.0,
        ge=0.0,
        description="Geodetic perimeter in metres",
    )
    vertex_count: int = Field(
        default=0,
        ge=0,
        description="Total vertex count across all rings",
    )

    @model_validator(mode="after")
    def _validate_geometry_consistency(self) -> "PlotBoundary":
        """Validate geometry type is consistent with ring data."""
        if self.geometry_type == GeometryType.POLYGON:
            if self.exterior_ring is None:
                pass  # Allow deferred ring assignment
        if self.geometry_type == GeometryType.POINT:
            if self.exterior_ring is not None:
                pass  # Allow for backwards compatibility
        return self


class ValidationError(BaseModel):
    """A single geometry validation error or warning.

    Attributes:
        error_type: Type of validation error detected.
        description: Human-readable description of the error.
        location: Optional location description within the geometry.
        severity: Severity level (error or warning).
        auto_repairable: Whether automated repair is available.
        repair_strategy: Suggested repair strategy if auto-repairable.
    """

    model_config = ConfigDict(from_attributes=True)

    error_type: ValidationErrorType = Field(
        ...,
        description="Type of validation error",
    )
    description: str = Field(
        default="",
        description="Human-readable error description",
    )
    location: str = Field(
        default="",
        description="Location within the geometry",
    )
    severity: str = Field(
        default="error",
        description="Severity level: error or warning",
    )
    auto_repairable: bool = Field(
        default=False,
        description="Whether automated repair is available",
    )
    repair_strategy: Optional[RepairStrategy] = Field(
        None,
        description="Suggested repair strategy",
    )


class ValidationResult(BaseModel):
    """Complete validation result for a plot boundary.

    Attributes:
        is_valid: Whether the boundary passed all validation checks.
        errors: List of validation errors detected.
        warnings: List of validation warnings (non-blocking).
        repaired: Whether automatic repair was applied.
        repair_actions: List of repair actions that were applied.
        confidence_score: Confidence in the validation result (0-1).
        ogc_compliant: Whether the boundary is OGC compliant.
    """

    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(
        ...,
        description="Whether boundary passed validation",
    )
    errors: List[ValidationError] = Field(
        default_factory=list,
        description="Validation errors detected",
    )
    warnings: List[ValidationError] = Field(
        default_factory=list,
        description="Validation warnings (non-blocking)",
    )
    repaired: bool = Field(
        default=False,
        description="Whether automatic repair was applied",
    )
    repair_actions: List[str] = Field(
        default_factory=list,
        description="Repair actions applied",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Validation confidence score (0-1)",
    )
    ogc_compliant: bool = Field(
        default=False,
        description="OGC Simple Features compliant",
    )


class AreaResult(BaseModel):
    """Geodetic area calculation result for a plot boundary.

    Attributes:
        area_m2: Area in square metres.
        area_hectares: Area in hectares.
        area_acres: Area in acres.
        area_km2: Area in square kilometres.
        perimeter_m: Perimeter in metres.
        compactness: Compactness index value.
        threshold_classification: EUDR polygon vs point classification.
        method: Calculation method used.
        uncertainty_m2: Estimated area uncertainty in square metres.
    """

    model_config = ConfigDict(from_attributes=True)

    area_m2: float = Field(
        ...,
        ge=0.0,
        description="Area in square metres",
    )
    area_hectares: float = Field(
        ...,
        ge=0.0,
        description="Area in hectares",
    )
    area_acres: float = Field(
        ...,
        ge=0.0,
        description="Area in acres",
    )
    area_km2: float = Field(
        ...,
        ge=0.0,
        description="Area in square kilometres",
    )
    perimeter_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Perimeter in metres",
    )
    compactness: float = Field(
        default=0.0,
        ge=0.0,
        description="Compactness index value",
    )
    threshold_classification: ThresholdClassification = Field(
        ...,
        description="EUDR polygon vs point classification",
    )
    method: str = Field(
        default="karney",
        description="Calculation method used",
    )
    uncertainty_m2: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated area uncertainty (sq metres)",
    )


class OverlapRecord(BaseModel):
    """Detected overlap between two plot boundaries.

    Attributes:
        plot_id_a: First plot identifier in the overlap pair.
        plot_id_b: Second plot identifier in the overlap pair.
        overlap_area_m2: Area of the overlap region in square metres.
        overlap_percentage_a: Overlap as percentage of plot A area.
        overlap_percentage_b: Overlap as percentage of plot B area.
        severity: Overlap severity classification.
        intersection_geometry: WKT of the overlap geometry.
        detected_at: UTC timestamp when the overlap was detected.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id_a: str = Field(
        ...,
        min_length=1,
        description="First plot in overlap pair",
    )
    plot_id_b: str = Field(
        ...,
        min_length=1,
        description="Second plot in overlap pair",
    )
    overlap_area_m2: float = Field(
        ...,
        ge=0.0,
        description="Overlap area in square metres",
    )
    overlap_percentage_a: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overlap as percentage of plot A area",
    )
    overlap_percentage_b: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overlap as percentage of plot B area",
    )
    severity: OverlapSeverity = Field(
        ...,
        description="Overlap severity classification",
    )
    intersection_geometry: str = Field(
        default="",
        description="WKT of the overlap geometry",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Detection timestamp (UTC)",
    )

    @model_validator(mode="after")
    def _validate_distinct_plots(self) -> "OverlapRecord":
        """Validate that plot_id_a and plot_id_b are different."""
        if self.plot_id_a == self.plot_id_b:
            raise ValueError(
                f"plot_id_a and plot_id_b must be different, "
                f"both are '{self.plot_id_a}'"
            )
        return self


class BoundaryVersion(BaseModel):
    """A versioned snapshot of a plot boundary.

    Supports EUDR Article 31 record-keeping (5-year retention).

    Attributes:
        plot_id: Plot identifier this version belongs to.
        version_number: Version number (incrementing integer).
        boundary: Complete boundary state at this version.
        change_reason: Reason for creating this version.
        changed_by: Identifier of who made the change.
        changed_at: UTC timestamp when this version was created.
        previous_version: Previous version number (None for initial).
        area_diff_m2: Area difference from previous version (sq metres).
        provenance_hash: SHA-256 hash of the version content.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier",
    )
    version_number: int = Field(
        ...,
        ge=1,
        description="Version number",
    )
    boundary: PlotBoundary = Field(
        ...,
        description="Complete boundary at this version",
    )
    change_reason: VersionChangeReason = Field(
        ...,
        description="Reason for version creation",
    )
    changed_by: str = Field(
        default="system",
        description="User or system that made the change",
    )
    changed_at: datetime = Field(
        default_factory=_utcnow,
        description="Version creation timestamp (UTC)",
    )
    previous_version: Optional[int] = Field(
        None,
        ge=1,
        description="Previous version number",
    )
    area_diff_m2: float = Field(
        default=0.0,
        description="Area difference from previous version (sq metres)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of version content",
    )

    @model_validator(mode="after")
    def _validate_version_sequence(self) -> "BoundaryVersion":
        """Validate version numbering consistency."""
        if (
            self.previous_version is not None
            and self.previous_version >= self.version_number
        ):
            raise ValueError(
                f"previous_version ({self.previous_version}) must be "
                f"< version_number ({self.version_number})"
            )
        return self


class SimplificationResult(BaseModel):
    """Result of polygon boundary simplification.

    Attributes:
        original_vertices: Vertices in the original boundary.
        simplified_vertices: Vertices after simplification.
        reduction_ratio: Fraction of vertices removed (0-1).
        area_change_pct: Area change percentage from original.
        hausdorff_distance: Max distance between original and
            simplified in metres.
        method: Simplification algorithm used.
        tolerance: Tolerance parameter value used.
    """

    model_config = ConfigDict(from_attributes=True)

    original_vertices: int = Field(
        ...,
        ge=0,
        description="Vertices in original boundary",
    )
    simplified_vertices: int = Field(
        ...,
        ge=0,
        description="Vertices after simplification",
    )
    reduction_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of vertices removed (0-1)",
    )
    area_change_pct: float = Field(
        default=0.0,
        description="Area change percentage from original",
    )
    hausdorff_distance: float = Field(
        default=0.0,
        ge=0.0,
        description="Max distance between original and simplified (metres)",
    )
    method: SimplificationMethod = Field(
        ...,
        description="Simplification algorithm used",
    )
    tolerance: float = Field(
        ...,
        ge=0.0,
        description="Tolerance parameter value used",
    )

    @model_validator(mode="after")
    def _validate_vertex_counts(self) -> "SimplificationResult":
        """Validate simplified vertices <= original vertices."""
        if self.simplified_vertices > self.original_vertices:
            raise ValueError(
                f"simplified_vertices ({self.simplified_vertices}) must be "
                f"<= original_vertices ({self.original_vertices})"
            )
        return self


class SplitResult(BaseModel):
    """Result of a plot boundary split operation.

    Attributes:
        parent_plot_id: Identifier of the plot that was split.
        child_boundaries: Child boundaries created by the split.
        cutting_line: WKT of the cutting line used.
        area_conservation_check: Whether child areas sum to parent.
        provenance_hash: SHA-256 hash of the split operation.
    """

    model_config = ConfigDict(from_attributes=True)

    parent_plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot that was split",
    )
    child_boundaries: List[PlotBoundary] = Field(
        ...,
        min_length=2,
        description="Child boundaries from the split",
    )
    cutting_line: str = Field(
        default="",
        description="WKT of the cutting line",
    )
    area_conservation_check: bool = Field(
        default=False,
        description="Area sum matches parent within tolerance",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of split operation",
    )


class MergeResult(BaseModel):
    """Result of a plot boundary merge operation.

    Attributes:
        parent_plot_ids: Identifiers of the plots that were merged.
        merged_boundary: The combined boundary after merging.
        area_conservation_check: Whether merged area matches sum.
        provenance_hash: SHA-256 hash of the merge operation.
    """

    model_config = ConfigDict(from_attributes=True)

    parent_plot_ids: List[str] = Field(
        ...,
        min_length=2,
        description="Plots that were merged",
    )
    merged_boundary: PlotBoundary = Field(
        ...,
        description="Combined boundary after merging",
    )
    area_conservation_check: bool = Field(
        default=False,
        description="Area sum matches merged within tolerance",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of merge operation",
    )


class ExportResult(BaseModel):
    """Result of a boundary export operation.

    Attributes:
        format: Export format used.
        data: Exported data as string or base64.
        file_size_bytes: Size of the exported data in bytes.
        plot_count: Number of plots in the export.
        crs: CRS of the exported data.
        precision: Coordinate precision used.
        timestamp: UTC timestamp of the export.
        metadata: Additional export metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    format: ExportFormat = Field(
        ...,
        description="Export format used",
    )
    data: str = Field(
        default="",
        description="Exported data (text or base64-encoded binary)",
    )
    file_size_bytes: int = Field(
        default=0,
        ge=0,
        description="Exported data size in bytes",
    )
    plot_count: int = Field(
        default=0,
        ge=0,
        description="Number of plots in export",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="CRS of exported data",
    )
    precision: int = Field(
        default=8,
        ge=1,
        le=15,
        description="Coordinate precision (decimal places)",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Export timestamp (UTC)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional export metadata",
    )


class BatchJob(BaseModel):
    """Batch boundary processing job.

    Attributes:
        job_id: Unique batch job identifier (UUID).
        status: Current job status.
        plot_ids: List of plot identifiers in the batch.
        total_plots: Total number of plots in the batch.
        completed_plots: Number of plots completed.
        failed_plots: Number of plots that failed.
        operation: Type of batch operation being performed.
        priority: Processing priority (1-10).
        created_at: Job creation timestamp (UTC).
        started_at: Processing start timestamp (UTC).
        completed_at: Processing completion timestamp (UTC).
        error_messages: Errors keyed by plot_id.
        progress_pct: Completion percentage (0-100).
        estimated_remaining_seconds: Estimated time remaining.
        provenance_hash: SHA-256 hash of the batch job state.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier (UUID)",
    )
    status: BatchStatus = Field(
        default=BatchStatus.PENDING,
        description="Current job status",
    )
    plot_ids: List[str] = Field(
        default_factory=list,
        description="Plot identifiers in the batch",
    )
    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total plots in batch",
    )
    completed_plots: int = Field(
        default=0,
        ge=0,
        description="Plots completed successfully",
    )
    failed_plots: int = Field(
        default=0,
        ge=0,
        description="Plots that failed processing",
    )
    operation: str = Field(
        default="validate",
        description="Batch operation type",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Processing priority (1-10)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Job creation timestamp (UTC)",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Processing start timestamp (UTC)",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Processing completion timestamp (UTC)",
    )
    error_messages: Dict[str, str] = Field(
        default_factory=dict,
        description="Errors keyed by plot_id",
    )
    progress_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage (0-100)",
    )
    estimated_remaining_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated time remaining (seconds)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of batch job state",
    )

    @field_validator("plot_ids")
    @classmethod
    def _validate_batch_size(cls, v: List[str]) -> List[str]:
        """Validate batch size does not exceed maximum."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of {MAX_BATCH_SIZE}"
            )
        return v


# =============================================================================
# Request Models
# =============================================================================


class CreateBoundaryRequest(BaseModel):
    """Request to create a new plot boundary.

    Attributes:
        plot_id: Optional plot identifier (auto-generated if None).
        exterior_ring_coords: Exterior ring as [[lat, lon], ...].
        holes_coords: Holes as [[[lat, lon], ...], ...].
        crs: Input coordinate reference system.
        commodity: EUDR commodity on this plot.
        country_iso: ISO 3166-1 alpha-2 country code.
        owner_id: Owner or operator identifier.
        certification_id: External certification reference.
        metadata: Additional metadata.
        auto_repair: Auto-repair geometry errors.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: Optional[str] = Field(
        None,
        description="Optional plot identifier (auto-generated if None)",
    )
    exterior_ring_coords: List[List[float]] = Field(
        ...,
        min_length=4,
        description="Exterior ring as [[lat, lon], ...] pairs",
    )
    holes_coords: List[List[List[float]]] = Field(
        default_factory=list,
        description="Holes as [[[lat, lon], ...], ...] rings",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Input coordinate reference system",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="EUDR commodity on this plot",
    )
    country_iso: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code",
    )
    owner_id: str = Field(
        default="",
        description="Owner or operator identifier",
    )
    certification_id: str = Field(
        default="",
        description="External certification reference",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    auto_repair: bool = Field(
        default=True,
        description="Auto-repair geometry errors",
    )


class UpdateBoundaryRequest(BaseModel):
    """Request to update an existing plot boundary.

    Attributes:
        plot_id: Plot identifier to update.
        exterior_ring_coords: New exterior ring coordinate pairs.
        holes_coords: New hole coordinate pair lists.
        change_reason: Reason for the boundary update.
        changed_by: User or system making the change.
        auto_repair: Auto-repair geometry errors.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier to update",
    )
    exterior_ring_coords: List[List[float]] = Field(
        ...,
        min_length=4,
        description="New exterior ring [[lat, lon], ...] pairs",
    )
    holes_coords: List[List[List[float]]] = Field(
        default_factory=list,
        description="New holes [[[lat, lon], ...], ...] rings",
    )
    change_reason: VersionChangeReason = Field(
        default=VersionChangeReason.SURVEY_UPDATE,
        description="Reason for update",
    )
    changed_by: str = Field(
        default="system",
        description="User or system making the change",
    )
    auto_repair: bool = Field(
        default=True,
        description="Auto-repair geometry errors",
    )


class ValidateRequest(BaseModel):
    """Request to validate a plot boundary.

    Attributes:
        plot_id: Plot identifier to validate.
        polygon_wkt: Optional WKT boundary override.
        include_warnings: Include warning-level issues.
        strict_ogc: Enforce strict OGC compliance.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier to validate",
    )
    polygon_wkt: Optional[str] = Field(
        None,
        description="Optional WKT boundary override",
    )
    include_warnings: bool = Field(
        default=True,
        description="Include warning-level issues",
    )
    strict_ogc: bool = Field(
        default=True,
        description="Enforce strict OGC compliance",
    )


class RepairRequest(BaseModel):
    """Request to repair a plot boundary.

    Attributes:
        plot_id: Plot identifier to repair.
        strategies: Allowed repair strategies (empty = all).
        max_iterations: Maximum repair iterations.
        preserve_area: Prioritize area preservation.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier to repair",
    )
    strategies: List[RepairStrategy] = Field(
        default_factory=list,
        description="Allowed repair strategies (empty = all)",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum repair iterations",
    )
    preserve_area: bool = Field(
        default=True,
        description="Prioritize area preservation",
    )


class AreaCalculationRequest(BaseModel):
    """Request to calculate geodetic area for a plot boundary.

    Attributes:
        plot_id: Plot identifier.
        polygon_wkt: Optional WKT boundary override.
        method: Area calculation method preference.
        include_compactness: Compute compactness index.
        compactness_method: Compactness index method.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier",
    )
    polygon_wkt: Optional[str] = Field(
        None,
        description="Optional WKT boundary override",
    )
    method: str = Field(
        default="karney",
        description="Area calculation method (karney, spherical, utm)",
    )
    include_compactness: bool = Field(
        default=True,
        description="Compute compactness index",
    )
    compactness_method: CompactnessIndex = Field(
        default=CompactnessIndex.POLSBY_POPPER,
        description="Compactness index method",
    )


class OverlapDetectionRequest(BaseModel):
    """Request to detect overlaps for a plot boundary.

    Attributes:
        plot_id: Primary plot to check for overlaps.
        candidate_plot_ids: Candidate plots (empty = spatial scan).
        min_overlap_area_m2: Minimum overlap area override (sq metres).
        include_geometry: Include overlap geometry in response.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Primary plot to check for overlaps",
    )
    candidate_plot_ids: List[str] = Field(
        default_factory=list,
        description="Candidate plots (empty = spatial scan)",
    )
    min_overlap_area_m2: Optional[float] = Field(
        None,
        ge=0.0,
        description="Minimum overlap area override (sq metres)",
    )
    include_geometry: bool = Field(
        default=True,
        description="Include overlap geometry in response",
    )


class SimplifyRequest(BaseModel):
    """Request to simplify a plot boundary.

    Attributes:
        plot_id: Plot identifier to simplify.
        method: Simplification algorithm.
        tolerance: Tolerance parameter override.
        max_area_deviation: Maximum area deviation override.
        preserve_topology: Preserve topological relationships.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier to simplify",
    )
    method: SimplificationMethod = Field(
        default=SimplificationMethod.DOUGLAS_PEUCKER,
        description="Simplification algorithm",
    )
    tolerance: Optional[float] = Field(
        None,
        ge=0.0,
        description="Tolerance parameter override",
    )
    max_area_deviation: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Maximum area deviation override (fraction)",
    )
    preserve_topology: bool = Field(
        default=True,
        description="Preserve topological relationships",
    )


class SplitRequest(BaseModel):
    """Request to split a plot boundary.

    Attributes:
        plot_id: Plot identifier to split.
        cutting_line_wkt: WKT of the cutting line.
        child_metadata: Metadata for child boundaries.
        changed_by: User or system performing the split.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        ...,
        min_length=1,
        description="Plot identifier to split",
    )
    cutting_line_wkt: str = Field(
        ...,
        min_length=1,
        description="WKT of the cutting line",
    )
    child_metadata: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Metadata for child boundaries",
    )
    changed_by: str = Field(
        default="system",
        description="User or system performing the split",
    )


class MergeRequest(BaseModel):
    """Request to merge plot boundaries.

    Attributes:
        plot_ids: Plot identifiers to merge (min 2).
        merged_plot_id: Optional merged plot identifier.
        changed_by: User or system performing the merge.
        merged_metadata: Metadata for merged boundary.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_ids: List[str] = Field(
        ...,
        min_length=2,
        description="Plot identifiers to merge (min 2)",
    )
    merged_plot_id: Optional[str] = Field(
        None,
        description="Optional merged plot identifier",
    )
    changed_by: str = Field(
        default="system",
        description="User or system performing the merge",
    )
    merged_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata for merged boundary",
    )


class ExportRequest(BaseModel):
    """Request to export plot boundaries.

    Attributes:
        plot_ids: Plot identifiers to export.
        format: Export format.
        crs: Output CRS.
        precision: Coordinate precision (decimal places).
        include_metadata: Include boundary metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Plot identifiers to export",
    )
    format: ExportFormat = Field(
        default=ExportFormat.GEOJSON,
        description="Export format",
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Output CRS",
    )
    precision: int = Field(
        default=8,
        ge=1,
        le=15,
        description="Coordinate precision (decimal places)",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include boundary metadata in export",
    )


class BatchBoundaryRequest(BaseModel):
    """Request for batch boundary operations.

    Attributes:
        plot_ids: Plot identifiers to process.
        operation: Batch operation type.
        auto_repair: Auto-repair geometry errors.
        priority: Processing priority (1-10).
    """

    model_config = ConfigDict(from_attributes=True)

    plot_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Plot identifiers to process",
    )
    operation: str = Field(
        default="validate",
        description="Batch operation type",
    )
    auto_repair: bool = Field(
        default=True,
        description="Auto-repair geometry errors",
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Processing priority (1-10)",
    )

    @field_validator("plot_ids")
    @classmethod
    def _validate_batch_size(cls, v: List[str]) -> List[str]:
        """Validate batch size does not exceed maximum."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of {MAX_BATCH_SIZE}"
            )
        return v


# =============================================================================
# Response Models
# =============================================================================


class BoundaryResponse(BaseModel):
    """Response from a boundary create or update operation.

    Attributes:
        boundary: The created or updated boundary.
        validation: Validation result for the boundary.
        area: Computed area result.
        processing_time_ms: Processing time in milliseconds.
        warnings: List of warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    boundary: PlotBoundary = Field(
        ...,
        description="Created or updated boundary",
    )
    validation: Optional[ValidationResult] = Field(
        None,
        description="Validation result",
    )
    area: Optional[AreaResult] = Field(
        None,
        description="Computed area result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class ValidationResponse(BaseModel):
    """Response from a boundary validation request.

    Attributes:
        validation: The validation result.
        processing_time_ms: Processing time in milliseconds.
        repaired_boundary: Boundary after repair (if applied).
    """

    model_config = ConfigDict(from_attributes=True)

    validation: ValidationResult = Field(
        ...,
        description="Validation result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    repaired_boundary: Optional[PlotBoundary] = Field(
        None,
        description="Boundary after repair (if applied)",
    )


class AreaResponse(BaseModel):
    """Response from an area calculation request.

    Attributes:
        area: The area calculation result.
        processing_time_ms: Processing time in milliseconds.
        warnings: List of warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    area: AreaResult = Field(
        ...,
        description="Area calculation result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class OverlapResponse(BaseModel):
    """Response from an overlap detection request.

    Attributes:
        overlaps: Detected overlaps.
        total_overlap_area_m2: Total overlap area (sq metres).
        plots_checked: Number of candidate plots checked.
        processing_time_ms: Processing time in milliseconds.
        warnings: Warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    overlaps: List[OverlapRecord] = Field(
        default_factory=list,
        description="Detected overlaps",
    )
    total_overlap_area_m2: float = Field(
        default=0.0,
        ge=0.0,
        description="Total overlap area (sq metres)",
    )
    plots_checked: int = Field(
        default=0,
        ge=0,
        description="Number of candidate plots checked",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class VersionResponse(BaseModel):
    """Response from a version history query.

    Attributes:
        versions: Boundary version history.
        current_version: Current active version number.
        total_versions: Total number of versions.
        processing_time_ms: Processing time in milliseconds.
    """

    model_config = ConfigDict(from_attributes=True)

    versions: List[BoundaryVersion] = Field(
        default_factory=list,
        description="Boundary version history",
    )
    current_version: int = Field(
        default=1,
        ge=1,
        description="Current active version number",
    )
    total_versions: int = Field(
        default=0,
        ge=0,
        description="Total number of versions",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )


class SimplificationResponse(BaseModel):
    """Response from a simplification request.

    Attributes:
        result: Simplification metrics.
        simplified_boundary: Boundary after simplification.
        processing_time_ms: Processing time in milliseconds.
        warnings: Warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    result: SimplificationResult = Field(
        ...,
        description="Simplification metrics",
    )
    simplified_boundary: PlotBoundary = Field(
        ...,
        description="Boundary after simplification",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class SplitMergeResponse(BaseModel):
    """Response from a split or merge operation.

    Attributes:
        split_result: Split operation result (if split).
        merge_result: Merge operation result (if merge).
        processing_time_ms: Processing time in milliseconds.
        warnings: Warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    split_result: Optional[SplitResult] = Field(
        None,
        description="Split operation result",
    )
    merge_result: Optional[MergeResult] = Field(
        None,
        description="Merge operation result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


class ExportResponse(BaseModel):
    """Response from a boundary export operation.

    Attributes:
        export: Export result with data and metadata.
        processing_time_ms: Processing time in milliseconds.
        warnings: Warning messages.
    """

    model_config = ConfigDict(from_attributes=True)

    export: ExportResult = Field(
        ...,
        description="Export result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "EUDR_AREA_THRESHOLD_HA",
    "MAX_BATCH_SIZE",
    "DEFAULT_COORDINATE_PRECISION",
    "WGS84_SEMI_MAJOR_AXIS",
    "WGS84_FLATTENING",
    "MIN_POLYGON_VERTICES",
    # Re-exported
    "EUDRCommodity",
    # Enumerations
    "GeometryType",
    "CoordinateReferenceSystem",
    "ValidationErrorType",
    "RepairStrategy",
    "OverlapSeverity",
    "OverlapResolution",
    "VersionChangeReason",
    "SimplificationMethod",
    "ExportFormat",
    "ThresholdClassification",
    "CompactnessIndex",
    "BatchStatus",
    # Core models
    "Coordinate",
    "BoundingBox",
    "Ring",
    "PlotBoundary",
    "ValidationError",
    "ValidationResult",
    "AreaResult",
    "OverlapRecord",
    "BoundaryVersion",
    "SimplificationResult",
    "SplitResult",
    "MergeResult",
    "ExportResult",
    "BatchJob",
    # Request models
    "CreateBoundaryRequest",
    "UpdateBoundaryRequest",
    "ValidateRequest",
    "RepairRequest",
    "AreaCalculationRequest",
    "OverlapDetectionRequest",
    "SimplifyRequest",
    "SplitRequest",
    "MergeRequest",
    "ExportRequest",
    "BatchBoundaryRequest",
    # Response models
    "BoundaryResponse",
    "ValidationResponse",
    "AreaResponse",
    "OverlapResponse",
    "VersionResponse",
    "SimplificationResponse",
    "SplitMergeResponse",
    "ExportResponse",
]
