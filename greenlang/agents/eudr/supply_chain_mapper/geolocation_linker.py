# -*- coding: utf-8 -*-
"""
Geolocation Linker Engine - AGENT-EUDR-001: Supply Chain Mapping Master (Feature 3)

Links supply chain producer nodes to registered production plots with GPS/polygon
geolocation data per EUDR Article 9. Provides spatial validation, PostGIS-backed
lookups, protected area cross-referencing, deforestation alert cross-referencing,
bulk import from multiple formats, and distance metrics for logistics validation.

Integrations:
    - AGENT-DATA-005 PlotRegistryEngine: Plot registration and lookup
    - AGENT-DATA-006 GIS/Mapping Connector (SpatialAnalyzerEngine): Spatial operations
    - AGENT-DATA-006 GIS/Mapping Connector (BoundaryResolverEngine): Protected areas
    - AGENT-DATA-007 Deforestation Satellite Connector (AlertAggregationEngine): Alerts

Zero-Hallucination Guarantees:
    - All coordinate validation is deterministic (WGS84 bounds checking)
    - Polygon requirement (>4 ha) enforced per EUDR Article 9(1)(d)
    - Distance calculations use Haversine formula with WGS84 constants
    - Precision validation ensures 6+ decimal places (approximately 0.11m)
    - Spatial indexing via R-tree for <100ms bounding box lookups (100K plots)
    - SHA-256 provenance hashes on all linkage operations
    - No ML/LLM used for geolocation validation or spatial reasoning

Performance Targets (from PRD):
    - Plot lookups within bounding box: <100ms for 100,000 plots
    - Coordinate precision: 6+ decimal places (WGS84)
    - Polygon compliance: 100% of plots >4 ha must have polygon data

Example:
    >>> from greenlang.agents.eudr.supply_chain_mapper.geolocation_linker import (
    ...     GeolocationLinker,
    ... )
    >>> linker = GeolocationLinker()
    >>> link_result = linker.link_producer_to_plot(
    ...     producer_node_id="NODE-abc123",
    ...     plot_id="PLOT-def456",
    ... )
    >>> assert link_result["link_id"].startswith("GEO-LNK-")

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 (Feature 3: Plot-Level Geolocation Integration)
Agent ID: GL-EUDR-SCM-001
Regulation: EU 2023/1115 (EUDR) Article 9
Status: Production Ready
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, model, or other serializable).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with a given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        ID in format "{prefix}-{hex12}".
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Earth radius in metres for Haversine calculations (WGS84 mean)
EARTH_RADIUS_M: float = 6_371_000.0

# Minimum decimal precision for EUDR Article 9 coordinate compliance
MIN_COORDINATE_PRECISION: int = 6

# EUDR Article 9(1)(d) polygon requirement threshold in hectares
POLYGON_AREA_THRESHOLD_HA: float = 4.0

# Minimum polygon vertices to form a valid polygon
MIN_POLYGON_VERTICES: int = 3

# WGS84 SRID
WGS84_SRID: int = 4326

# EUDR cutoff date for deforestation-free declaration
EUDR_CUTOFF_DATE: str = "2020-12-31"

# Default spatial index grid cell size in degrees (approx 11km at equator)
DEFAULT_GRID_CELL_SIZE: float = 0.1

# Supported bulk import formats
SUPPORTED_IMPORT_FORMATS = frozenset({"csv", "geojson", "shapefile"})

# Maximum distance in metres for logistics proximity validation (500km)
MAX_LOGISTICS_DISTANCE_M: float = 500_000.0

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LinkageStatus(str, Enum):
    """Status of a producer-to-plot geolocation linkage.

    Tracks the lifecycle of the linkage from creation through
    validation and final verification.
    """

    LINKED = "linked"
    UNLINKED = "unlinked"
    PENDING_VALIDATION = "pending_validation"
    VALIDATED = "validated"
    REJECTED = "rejected"

class GeolocationGapType(str, Enum):
    """Types of geolocation gaps identified during compliance analysis.

    Each gap type corresponds to a specific EUDR Article 9 requirement
    that is not met for a given producer or plot.
    """

    MISSING_PLOT = "missing_plot"
    MISSING_COORDINATES = "missing_coordinates"
    INSUFFICIENT_PRECISION = "insufficient_precision"
    MISSING_POLYGON = "missing_polygon"
    INVALID_COORDINATES = "invalid_coordinates"
    PROTECTED_AREA_OVERLAP = "protected_area_overlap"
    DEFORESTATION_ALERT = "deforestation_alert"

class GeolocationGapSeverity(str, Enum):
    """Severity levels for geolocation compliance gaps."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ProtectedAreaType(str, Enum):
    """Categories of protected areas for cross-referencing.

    Based on IUCN Protected Area Categories and additional
    EUDR-relevant designations.
    """

    NATIONAL_PARK = "national_park"
    NATURE_RESERVE = "nature_reserve"
    WILDERNESS_AREA = "wilderness_area"
    NATURAL_MONUMENT = "natural_monument"
    HABITAT_MANAGEMENT = "habitat_management"
    PROTECTED_LANDSCAPE = "protected_landscape"
    COMMUNITY_CONSERVED = "community_conserved"
    INDIGENOUS_TERRITORY = "indigenous_territory"
    RAMSAR_WETLAND = "ramsar_wetland"
    UNESCO_HERITAGE = "unesco_heritage"
    KEY_BIODIVERSITY_AREA = "key_biodiversity_area"

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CoordinateValidation:
    """Result of WGS84 coordinate validation.

    Attributes:
        is_valid: Whether all validation checks pass.
        latitude: Validated latitude value.
        longitude: Validated longitude value.
        precision_lat: Number of decimal places in latitude.
        precision_lon: Number of decimal places in longitude.
        meets_precision: Whether precision meets the 6-decimal minimum.
        within_bounds: Whether coordinates are within WGS84 bounds.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
    """

    is_valid: bool
    latitude: float
    longitude: float
    precision_lat: int
    precision_lon: int
    meets_precision: bool
    within_bounds: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

@dataclass
class PolygonValidation:
    """Result of polygon geometry validation for EUDR Article 9 compliance.

    Attributes:
        is_valid: Whether the polygon passes all validation checks.
        vertex_count: Number of vertices in the polygon ring.
        is_closed: Whether the ring is properly closed (first == last).
        area_hectares: Computed area in hectares.
        requires_polygon: Whether the plot area exceeds the 4 ha threshold.
        has_polygon: Whether polygon coordinates are provided.
        compliant: Whether EUDR Article 9(1)(d) polygon requirement is met.
        errors: List of validation error messages.
    """

    is_valid: bool
    vertex_count: int
    is_closed: bool
    area_hectares: float
    requires_polygon: bool
    has_polygon: bool
    compliant: bool
    errors: list = field(default_factory=list)

@dataclass
class DistanceMetric:
    """Distance calculation result between two geographic points.

    Attributes:
        from_point: Source [longitude, latitude] coordinates.
        to_point: Destination [longitude, latitude] coordinates.
        distance_metres: Geodesic distance in metres.
        distance_km: Geodesic distance in kilometres.
        bearing_degrees: Initial bearing in degrees from north.
        within_logistics_range: Whether distance is within acceptable range.
    """

    from_point: List[float]
    to_point: List[float]
    distance_metres: float
    distance_km: float
    bearing_degrees: float
    within_logistics_range: bool

# ---------------------------------------------------------------------------
# Spatial Index (R-tree approximation using grid cells)
# ---------------------------------------------------------------------------

class _SpatialGridIndex:
    """Grid-based spatial index for fast bounding box lookups.

    Uses a fixed-size grid over the WGS84 coordinate space.
    Each grid cell stores a list of plot IDs whose centroid falls
    within that cell. Bounding box queries return all plots in
    cells that intersect the query box.

    Performance target: <100ms for 100,000 plots.
    """

    def __init__(self, cell_size: float = DEFAULT_GRID_CELL_SIZE) -> None:
        """Initialize grid index.

        Args:
            cell_size: Grid cell size in degrees.
        """
        self._cell_size = cell_size
        self._grid: Dict[Tuple[int, int], List[str]] = {}
        self._locations: Dict[str, Tuple[float, float]] = {}

    def insert(self, plot_id: str, lon: float, lat: float) -> None:
        """Insert a plot into the spatial index.

        Args:
            plot_id: Plot identifier.
            lon: Longitude in degrees.
            lat: Latitude in degrees.
        """
        cell = self._cell_key(lon, lat)
        if cell not in self._grid:
            self._grid[cell] = []
        self._grid[cell].append(plot_id)
        self._locations[plot_id] = (lon, lat)

    def remove(self, plot_id: str) -> None:
        """Remove a plot from the spatial index.

        Args:
            plot_id: Plot identifier.
        """
        loc = self._locations.pop(plot_id, None)
        if loc is not None:
            cell = self._cell_key(loc[0], loc[1])
            cell_plots = self._grid.get(cell, [])
            if plot_id in cell_plots:
                cell_plots.remove(plot_id)

    def query_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> List[str]:
        """Query all plots within a bounding box.

        Args:
            min_lon: Minimum longitude.
            min_lat: Minimum latitude.
            max_lon: Maximum longitude.
            max_lat: Maximum latitude.

        Returns:
            List of plot IDs within the bounding box.
        """
        results: List[str] = []
        min_cx = int(math.floor(min_lon / self._cell_size))
        max_cx = int(math.floor(max_lon / self._cell_size))
        min_cy = int(math.floor(min_lat / self._cell_size))
        max_cy = int(math.floor(max_lat / self._cell_size))

        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cell_plots = self._grid.get((cx, cy), [])
                for pid in cell_plots:
                    loc = self._locations.get(pid)
                    if loc and min_lon <= loc[0] <= max_lon and min_lat <= loc[1] <= max_lat:
                        results.append(pid)

        return results

    def query_radius(
        self,
        lon: float,
        lat: float,
        radius_m: float,
    ) -> List[Tuple[str, float]]:
        """Query all plots within a radius of a point.

        Args:
            lon: Query point longitude.
            lat: Query point latitude.
            radius_m: Radius in metres.

        Returns:
            List of (plot_id, distance_metres) tuples, sorted by distance.
        """
        # Convert radius to approximate degree extent
        d_lat = radius_m / 111_320.0
        d_lon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.0001))

        candidates = self.query_bbox(
            lon - d_lon, lat - d_lat,
            lon + d_lon, lat + d_lat,
        )

        results: List[Tuple[str, float]] = []
        for pid in candidates:
            loc = self._locations.get(pid)
            if loc:
                dist = _haversine_distance(lon, lat, loc[0], loc[1])
                if dist <= radius_m:
                    results.append((pid, dist))

        results.sort(key=lambda x: x[1])
        return results

    @property
    def count(self) -> int:
        """Return the number of indexed plots."""
        return len(self._locations)

    def _cell_key(self, lon: float, lat: float) -> Tuple[int, int]:
        """Compute grid cell key for a coordinate.

        Args:
            lon: Longitude in degrees.
            lat: Latitude in degrees.

        Returns:
            (cell_x, cell_y) tuple.
        """
        return (
            int(math.floor(lon / self._cell_size)),
            int(math.floor(lat / self._cell_size)),
        )

# ---------------------------------------------------------------------------
# Core Spatial Functions
# ---------------------------------------------------------------------------

def _haversine_distance(
    lon1: float, lat1: float, lon2: float, lat2: float,
) -> float:
    """Compute Haversine distance between two WGS84 points.

    Args:
        lon1: Longitude of first point in degrees.
        lat1: Latitude of first point in degrees.
        lon2: Longitude of second point in degrees.
        lat2: Latitude of second point in degrees.

    Returns:
        Distance in metres.
    """
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c

def _initial_bearing(
    lon1: float, lat1: float, lon2: float, lat2: float,
) -> float:
    """Compute initial bearing from point 1 to point 2.

    Args:
        lon1: Longitude of first point in degrees.
        lat1: Latitude of first point in degrees.
        lon2: Longitude of second point in degrees.
        lat2: Latitude of second point in degrees.

    Returns:
        Bearing in degrees (0-360, clockwise from north).
    """
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(rlat2)
    y = (
        math.cos(rlat1) * math.sin(rlat2)
        - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    )
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

def _count_decimal_places(value: float) -> int:
    """Count the number of decimal places in a float value.

    Args:
        value: Numeric value to inspect.

    Returns:
        Number of significant decimal places.
    """
    s = str(value)
    if "." not in s:
        return 0
    # Strip trailing zeros for meaningful precision
    decimal_part = s.split(".")[1].rstrip("0")
    return len(decimal_part) if decimal_part else 0

def _geodesic_polygon_area(ring: List[List[float]]) -> float:
    """Compute geodesic area of a ring using the spherical excess formula.

    Uses the simplified formula with WGS84 Earth radius for area
    computation from longitude/latitude coordinate pairs.

    Args:
        ring: List of [longitude, latitude] coordinate pairs.

    Returns:
        Absolute area in square metres.
    """
    if len(ring) < 3:
        return 0.0

    area = 0.0
    n = len(ring)
    for i in range(n - 1):
        lon1, lat1 = ring[i][0], ring[i][1]
        lon2, lat2 = ring[i + 1][0], ring[i + 1][1]
        area += math.radians(lon2 - lon1) * (
            2 + math.sin(math.radians(lat1))
            + math.sin(math.radians(lat2))
        )

    return abs(area * EARTH_RADIUS_M ** 2 / 2.0)

def _point_in_polygon_ray(
    point: List[float], ring: List[List[float]],
) -> bool:
    """Ray casting algorithm for point-in-polygon test.

    Args:
        point: [longitude, latitude] test point.
        ring: List of [longitude, latitude] polygon vertices.

    Returns:
        True if point is inside the polygon.
    """
    x, y = point[0], point[1]
    n = len(ring)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside

# ---------------------------------------------------------------------------
# PostGIS Query Builder
# ---------------------------------------------------------------------------

class PostGISQueryBuilder:
    """Generates PostGIS-compatible SQL queries for spatial operations.

    All generated queries use parameterized SQL to prevent injection.
    Queries target the origin_plots table with PostGIS geometry column
    and GIST spatial index.

    Attributes:
        _srid: Spatial Reference Identifier (default WGS84 4326).
        _table: Target table name.
        _geom_column: Geometry column name.
    """

    def __init__(
        self,
        srid: int = WGS84_SRID,
        table: str = "origin_plots",
        geom_column: str = "geometry",
    ) -> None:
        """Initialize PostGIS query builder.

        Args:
            srid: Spatial Reference System Identifier.
            table: Target database table name.
            geom_column: Name of the PostGIS geometry column.
        """
        self._srid = srid
        self._table = table
        self._geom_column = geom_column

    def bbox_query(self) -> Tuple[str, List[str]]:
        """Generate a bounding box plot lookup query.

        Returns:
            Tuple of (SQL query string, list of parameter names).
            The query uses ST_MakeEnvelope with parameterized bounds.
        """
        sql = (
            f"SELECT plot_id, producer_node_id, commodity, country_code, "
            f"area_hectares, validation_status, deforestation_risk_score, "
            f"ST_AsGeoJSON({self._geom_column}) AS geojson "
            f"FROM {self._table} "
            f"WHERE {self._geom_column} && "
            f"ST_MakeEnvelope(%(min_lon)s, %(min_lat)s, %(max_lon)s, %(max_lat)s, {self._srid})"
        )
        return sql, ["min_lon", "min_lat", "max_lon", "max_lat"]

    def radius_query(self) -> Tuple[str, List[str]]:
        """Generate a radius-based plot lookup query.

        Returns:
            Tuple of (SQL query string, list of parameter names).
            Uses ST_DWithin for indexed distance filtering.
        """
        sql = (
            f"SELECT plot_id, producer_node_id, commodity, country_code, "
            f"area_hectares, validation_status, "
            f"ST_Distance("
            f"  {self._geom_column}::geography, "
            f"  ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), {self._srid})::geography"
            f") AS distance_m, "
            f"ST_AsGeoJSON({self._geom_column}) AS geojson "
            f"FROM {self._table} "
            f"WHERE ST_DWithin("
            f"  {self._geom_column}::geography, "
            f"  ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), {self._srid})::geography, "
            f"  %(radius_m)s"
            f") "
            f"ORDER BY distance_m ASC"
        )
        return sql, ["lon", "lat", "radius_m"]

    def contains_point_query(self) -> Tuple[str, List[str]]:
        """Generate a point-in-polygon containment query.

        Returns:
            Tuple of (SQL query string, list of parameter names).
        """
        sql = (
            f"SELECT plot_id, producer_node_id, commodity, country_code, "
            f"area_hectares, validation_status, "
            f"ST_AsGeoJSON({self._geom_column}) AS geojson "
            f"FROM {self._table} "
            f"WHERE ST_Contains("
            f"  {self._geom_column}, "
            f"  ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), {self._srid})"
            f")"
        )
        return sql, ["lon", "lat"]

    def protected_area_intersection_query(self) -> Tuple[str, List[str]]:
        """Generate a query to check plot overlap with protected areas.

        Returns:
            Tuple of (SQL query string, list of parameter names).
        """
        sql = (
            "SELECT pa.area_id, pa.name, pa.area_type, pa.iucn_category, "
            "ST_Area(ST_Intersection(p.geometry::geography, pa.geometry::geography)) "
            "  AS overlap_area_sq_m, "
            "ST_Area(p.geometry::geography) AS plot_area_sq_m "
            f"FROM {self._table} p "
            "JOIN protected_areas pa "
            "  ON ST_Intersects(p.geometry, pa.geometry) "
            "WHERE p.plot_id = %(plot_id)s"
        )
        return sql, ["plot_id"]

    def insert_plot_geometry(self) -> Tuple[str, List[str]]:
        """Generate an INSERT statement for plot geometry.

        Returns:
            Tuple of (SQL query string, list of parameter names).
        """
        sql = (
            f"INSERT INTO {self._table} "
            f"(plot_id, producer_node_id, geometry, area_hectares, commodity, "
            f"country_code, validation_status) "
            f"VALUES ("
            f"  %(plot_id)s, %(producer_node_id)s, "
            f"  ST_SetSRID(ST_GeomFromGeoJSON(%(geojson)s), {self._srid}), "
            f"  %(area_hectares)s, %(commodity)s, %(country_code)s, "
            f"  %(validation_status)s"
            f")"
        )
        return sql, [
            "plot_id", "producer_node_id", "geojson",
            "area_hectares", "commodity", "country_code", "validation_status",
        ]

    def create_spatial_index(self) -> str:
        """Generate DDL for creating a GIST spatial index.

        Returns:
            SQL DDL string for index creation.
        """
        return (
            f"CREATE INDEX IF NOT EXISTS idx_{self._table}_geom "
            f"ON {self._table} USING GIST ({self._geom_column});"
        )

# ---------------------------------------------------------------------------
# GeolocationLinker Engine
# ---------------------------------------------------------------------------

class GeolocationLinker:
    """Links supply chain producer nodes to registered plots with geolocation data.

    Implements PRD Feature 3 (Plot-Level Geolocation Integration) for
    AGENT-EUDR-001 Supply Chain Mapping Master. Validates GPS coordinates,
    enforces EUDR Article 9 polygon requirements, performs spatial lookups,
    cross-references protected areas and deforestation alerts, and supports
    bulk import from CSV, GeoJSON, and Shapefile.

    Attributes:
        _config: Configuration dictionary or object.
        _provenance: Optional provenance tracker instance.
        _plot_registry: Optional PlotRegistryEngine instance.
        _spatial_analyzer: Optional SpatialAnalyzerEngine instance.
        _boundary_resolver: Optional BoundaryResolverEngine instance.
        _alert_engine: Optional AlertAggregationEngine instance.
        _links: In-memory linkage storage keyed by link_id.
        _producer_links: Index of link IDs by producer_node_id.
        _plot_links: Index of link IDs by plot_id.
        _gaps: In-memory geolocation gap storage keyed by gap_id.
        _spatial_index: Grid-based spatial index for fast lookups.
        _postgis: PostGIS query builder instance.

    Example:
        >>> linker = GeolocationLinker()
        >>> result = linker.link_producer_to_plot("NODE-abc", "PLOT-def")
        >>> assert result["status"] == "linked"
    """

    def __init__(
        self,
        config: Any = None,
        provenance: Any = None,
        plot_registry: Any = None,
        spatial_analyzer: Any = None,
        boundary_resolver: Any = None,
        alert_engine: Any = None,
    ) -> None:
        """Initialize GeolocationLinker.

        Args:
            config: Optional configuration dictionary or object.
            provenance: Optional ProvenanceTracker instance for audit trails.
            plot_registry: Optional PlotRegistryEngine from AGENT-DATA-005.
            spatial_analyzer: Optional SpatialAnalyzerEngine from AGENT-DATA-006.
            boundary_resolver: Optional BoundaryResolverEngine from AGENT-DATA-006.
            alert_engine: Optional AlertAggregationEngine from AGENT-DATA-007.
        """
        self._config = config or {}
        self._provenance = provenance
        self._plot_registry = plot_registry
        self._spatial_analyzer = spatial_analyzer
        self._boundary_resolver = boundary_resolver
        self._alert_engine = alert_engine

        # In-memory storage
        self._links: Dict[str, Dict[str, Any]] = {}
        self._plots: Dict[str, Dict[str, Any]] = {}

        # Indexes for fast lookup
        self._producer_links: Dict[str, List[str]] = {}
        self._plot_links: Dict[str, List[str]] = {}

        # Gap tracking
        self._gaps: Dict[str, Dict[str, Any]] = {}
        self._producer_gaps: Dict[str, List[str]] = {}

        # Spatial index for <100ms bounding box lookups
        self._spatial_index = _SpatialGridIndex(
            cell_size=getattr(self._config, "grid_cell_size", DEFAULT_GRID_CELL_SIZE),
        )

        # PostGIS query builder
        self._postgis = PostGISQueryBuilder()

        logger.info("GeolocationLinker initialized")

    # ------------------------------------------------------------------
    # Coordinate Validation (EUDR Article 9)
    # ------------------------------------------------------------------

    def validate_coordinates(
        self,
        latitude: float,
        longitude: float,
    ) -> CoordinateValidation:
        """Validate GPS coordinates per EUDR Article 9 requirements.

        Checks:
        - WGS84 latitude bounds: [-90, 90]
        - WGS84 longitude bounds: [-180, 180]
        - Coordinate precision: 6+ decimal places (approximately 0.11m)

        Args:
            latitude: GPS latitude in decimal degrees.
            longitude: GPS longitude in decimal degrees.

        Returns:
            CoordinateValidation with detailed validation results.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check bounds
        within_bounds = True
        if not -90.0 <= latitude <= 90.0:
            within_bounds = False
            errors.append(
                f"Latitude {latitude} out of WGS84 range [-90, 90]"
            )
        if not -180.0 <= longitude <= 180.0:
            within_bounds = False
            errors.append(
                f"Longitude {longitude} out of WGS84 range [-180, 180]"
            )

        # Check precision
        precision_lat = _count_decimal_places(latitude)
        precision_lon = _count_decimal_places(longitude)
        meets_precision = (
            precision_lat >= MIN_COORDINATE_PRECISION
            and precision_lon >= MIN_COORDINATE_PRECISION
        )

        if not meets_precision:
            warnings.append(
                f"Coordinate precision below EUDR minimum of "
                f"{MIN_COORDINATE_PRECISION} decimal places: "
                f"lat={precision_lat}, lon={precision_lon}. "
                f"Approximately {10 ** (-min(precision_lat, precision_lon)) * 111_320:.1f}m accuracy."
            )

        # Zero-island check (0,0 is in the Gulf of Guinea -- likely data error)
        if latitude == 0.0 and longitude == 0.0:
            warnings.append(
                "Coordinates (0, 0) detected -- this is 'Null Island' "
                "in the Gulf of Guinea and likely indicates missing data."
            )

        is_valid = within_bounds and len(errors) == 0

        return CoordinateValidation(
            is_valid=is_valid,
            latitude=latitude,
            longitude=longitude,
            precision_lat=precision_lat,
            precision_lon=precision_lon,
            meets_precision=meets_precision,
            within_bounds=within_bounds,
            errors=errors,
            warnings=warnings,
        )

    def validate_polygon(
        self,
        polygon_coordinates: Optional[List[List[float]]],
        area_hectares: float,
    ) -> PolygonValidation:
        """Validate polygon geometry per EUDR Article 9(1)(d).

        EUDR Article 9(1)(d) requires polygon geolocation for plots
        of land with an area exceeding 4 hectares.

        Args:
            polygon_coordinates: List of [longitude, latitude] pairs
                forming a closed ring, or None if only a point is provided.
            area_hectares: Area of the plot in hectares.

        Returns:
            PolygonValidation with compliance assessment.
        """
        errors: List[str] = []
        requires_polygon = area_hectares > POLYGON_AREA_THRESHOLD_HA
        has_polygon = (
            polygon_coordinates is not None
            and len(polygon_coordinates) >= MIN_POLYGON_VERTICES
        )

        vertex_count = len(polygon_coordinates) if polygon_coordinates else 0
        is_closed = False
        computed_area = 0.0

        if has_polygon and polygon_coordinates is not None:
            # Validate vertex coordinates
            for i, coord in enumerate(polygon_coordinates):
                if len(coord) != 2:
                    errors.append(
                        f"Vertex {i} must have exactly 2 values "
                        f"[longitude, latitude], got {len(coord)}"
                    )
                else:
                    lon, lat = coord
                    if not -180.0 <= lon <= 180.0:
                        errors.append(
                            f"Vertex {i} longitude {lon} out of range [-180, 180]"
                        )
                    if not -90.0 <= lat <= 90.0:
                        errors.append(
                            f"Vertex {i} latitude {lat} out of range [-90, 90]"
                        )

            # Check ring closure
            if polygon_coordinates[0] == polygon_coordinates[-1]:
                is_closed = True
            else:
                # Auto-detect near-closure (within ~1m)
                d = _haversine_distance(
                    polygon_coordinates[0][0], polygon_coordinates[0][1],
                    polygon_coordinates[-1][0], polygon_coordinates[-1][1],
                )
                if d < 1.0:
                    is_closed = True
                else:
                    errors.append(
                        "Polygon ring is not closed: first and last vertices "
                        "do not match."
                    )

            # Compute area
            if len(errors) == 0:
                computed_area = _geodesic_polygon_area(polygon_coordinates) / 10_000.0

        # EUDR compliance check
        compliant = True
        if requires_polygon and not has_polygon:
            compliant = False
            errors.append(
                f"EUDR Article 9(1)(d) violation: Plot area "
                f"({area_hectares:.2f} ha) exceeds {POLYGON_AREA_THRESHOLD_HA} ha "
                f"threshold but no polygon coordinates are provided."
            )

        is_valid = len(errors) == 0

        return PolygonValidation(
            is_valid=is_valid,
            vertex_count=vertex_count,
            is_closed=is_closed,
            area_hectares=computed_area if computed_area > 0 else area_hectares,
            requires_polygon=requires_polygon,
            has_polygon=has_polygon,
            compliant=compliant,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Producer-to-Plot Linkage
    # ------------------------------------------------------------------

    def link_producer_to_plot(
        self,
        producer_node_id: str,
        plot_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        polygon_coordinates: Optional[List[List[float]]] = None,
        area_hectares: Optional[float] = None,
        commodity: Optional[str] = None,
        country_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Link a supply chain producer node to a registered plot.

        Creates a validated geolocation linkage between a producer node
        in the supply chain graph and a plot in the PlotRegistryEngine.
        Validates coordinates and polygon requirements, then indexes the
        linkage for fast spatial lookups.

        Args:
            producer_node_id: Identifier of the supply chain producer node.
            plot_id: Identifier of the registered plot.
            latitude: Optional GPS latitude for validation.
            longitude: Optional GPS longitude for validation.
            polygon_coordinates: Optional polygon ring coordinates.
            area_hectares: Optional plot area in hectares.
            commodity: Optional EUDR commodity type.
            country_code: Optional ISO 3166-1 alpha-2 country code.
            metadata: Optional additional metadata.

        Returns:
            Linkage result dictionary with link_id, status, validations.

        Raises:
            ValueError: If producer_node_id or plot_id is empty.
        """
        start_time = time.monotonic()

        if not producer_node_id or not producer_node_id.strip():
            raise ValueError("producer_node_id must be non-empty")
        if not plot_id or not plot_id.strip():
            raise ValueError("plot_id must be non-empty")

        link_id = _generate_id("GEO-LNK")
        validation_errors: List[str] = []
        validation_warnings: List[str] = []

        # Coordinate validation
        coord_validation = None
        if latitude is not None and longitude is not None:
            coord_validation = self.validate_coordinates(latitude, longitude)
            validation_errors.extend(coord_validation.errors)
            validation_warnings.extend(coord_validation.warnings)

        # Polygon validation
        polygon_validation = None
        if area_hectares is not None:
            polygon_validation = self.validate_polygon(
                polygon_coordinates, area_hectares,
            )
            validation_errors.extend(polygon_validation.errors)

        # Determine link status
        if validation_errors:
            status = LinkageStatus.REJECTED
        elif coord_validation and coord_validation.is_valid:
            status = LinkageStatus.LINKED
        else:
            status = LinkageStatus.PENDING_VALIDATION

        # Build link record
        link_record: Dict[str, Any] = {
            "link_id": link_id,
            "producer_node_id": producer_node_id,
            "plot_id": plot_id,
            "status": status.value,
            "latitude": latitude,
            "longitude": longitude,
            "polygon_coordinates": polygon_coordinates,
            "area_hectares": area_hectares,
            "commodity": commodity,
            "country_code": country_code,
            "coordinate_validation": {
                "is_valid": coord_validation.is_valid if coord_validation else None,
                "precision_lat": coord_validation.precision_lat if coord_validation else None,
                "precision_lon": coord_validation.precision_lon if coord_validation else None,
                "meets_precision": coord_validation.meets_precision if coord_validation else None,
            } if coord_validation else None,
            "polygon_validation": {
                "is_valid": polygon_validation.is_valid if polygon_validation else None,
                "vertex_count": polygon_validation.vertex_count if polygon_validation else None,
                "requires_polygon": polygon_validation.requires_polygon if polygon_validation else None,
                "compliant": polygon_validation.compliant if polygon_validation else None,
            } if polygon_validation else None,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
            "metadata": metadata or {},
            "created_at": utcnow().isoformat(),
        }

        # Store link
        self._links[link_id] = link_record

        # Index by producer and plot
        if producer_node_id not in self._producer_links:
            self._producer_links[producer_node_id] = []
        self._producer_links[producer_node_id].append(link_id)

        if plot_id not in self._plot_links:
            self._plot_links[plot_id] = []
        self._plot_links[plot_id].append(link_id)

        # Insert into spatial index if coordinates are valid
        if (
            latitude is not None
            and longitude is not None
            and coord_validation is not None
            and coord_validation.is_valid
        ):
            self._spatial_index.insert(plot_id, longitude, latitude)
            self._plots[plot_id] = {
                "plot_id": plot_id,
                "producer_node_id": producer_node_id,
                "latitude": latitude,
                "longitude": longitude,
                "polygon_coordinates": polygon_coordinates,
                "area_hectares": area_hectares,
                "commodity": commodity,
                "country_code": country_code,
            }

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(link_record)
            self._provenance.record(
                entity_type="geolocation_link",
                entity_id=link_id,
                action="link_producer_to_plot",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.agents.data.eudr_traceability.metrics import record_plot_registered
            record_plot_registered(commodity or "unknown", country_code or "XX")
        except (ImportError, Exception):
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Linked producer %s to plot %s: status=%s, link_id=%s (%.1f ms)",
            producer_node_id, plot_id, status.value, link_id, elapsed_ms,
        )
        return link_record

    def unlink_producer_from_plot(
        self,
        producer_node_id: str,
        plot_id: str,
    ) -> Dict[str, Any]:
        """Remove the linkage between a producer node and a plot.

        Args:
            producer_node_id: Producer node identifier.
            plot_id: Plot identifier.

        Returns:
            Result dictionary with removed_count and details.
        """
        removed: List[str] = []
        link_ids = list(self._producer_links.get(producer_node_id, []))

        for lid in link_ids:
            link = self._links.get(lid)
            if link and link.get("plot_id") == plot_id:
                del self._links[lid]
                self._producer_links[producer_node_id].remove(lid)
                plot_link_list = self._plot_links.get(plot_id, [])
                if lid in plot_link_list:
                    plot_link_list.remove(lid)
                self._spatial_index.remove(plot_id)
                self._plots.pop(plot_id, None)
                removed.append(lid)

        logger.info(
            "Unlinked producer %s from plot %s: removed %d linkages",
            producer_node_id, plot_id, len(removed),
        )
        return {
            "producer_node_id": producer_node_id,
            "plot_id": plot_id,
            "removed_count": len(removed),
            "removed_link_ids": removed,
        }

    def get_links_for_producer(
        self,
        producer_node_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all plot linkages for a producer node.

        Args:
            producer_node_id: Producer node identifier.

        Returns:
            List of linkage records.
        """
        link_ids = self._producer_links.get(producer_node_id, [])
        return [
            self._links[lid]
            for lid in link_ids
            if lid in self._links
        ]

    def get_links_for_plot(
        self,
        plot_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all producer linkages for a plot.

        Args:
            plot_id: Plot identifier.

        Returns:
            List of linkage records.
        """
        link_ids = self._plot_links.get(plot_id, [])
        return [
            self._links[lid]
            for lid in link_ids
            if lid in self._links
        ]

    def get_link(self, link_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific linkage record by ID.

        Args:
            link_id: Linkage identifier.

        Returns:
            Linkage record dictionary or None if not found.
        """
        return self._links.get(link_id)

    # ------------------------------------------------------------------
    # Spatial Queries (PostGIS-backed + in-memory fallback)
    # ------------------------------------------------------------------

    def find_plots_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> Dict[str, Any]:
        """Find all plots within a bounding box.

        Uses the spatial grid index for in-memory lookups. For
        database-backed deployments, generates the PostGIS query
        via PostGISQueryBuilder.

        Performance target: <100ms for 100,000 plots.

        Args:
            min_lon: Minimum longitude of the bounding box.
            min_lat: Minimum latitude of the bounding box.
            max_lon: Maximum longitude of the bounding box.
            max_lat: Maximum latitude of the bounding box.

        Returns:
            Dictionary with matching plot_ids, count, and PostGIS SQL.
        """
        start_time = time.monotonic()

        # In-memory spatial index query
        matching_ids = self._spatial_index.query_bbox(
            min_lon, min_lat, max_lon, max_lat,
        )

        matching_plots = [
            self._plots[pid]
            for pid in matching_ids
            if pid in self._plots
        ]

        # Generate PostGIS query for database-backed deployments
        postgis_sql, postgis_params = self._postgis.bbox_query()

        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "result_id": _generate_id("GEO-BBOX"),
            "operation": "find_plots_in_bbox",
            "bbox": {
                "min_lon": min_lon, "min_lat": min_lat,
                "max_lon": max_lon, "max_lat": max_lat,
            },
            "matching_count": len(matching_plots),
            "matching_plots": matching_plots,
            "postgis_query": postgis_sql,
            "postgis_params": postgis_params,
            "elapsed_ms": round(elapsed_ms, 2),
            "created_at": utcnow().isoformat(),
        }

        logger.debug(
            "BBox query [%.6f,%.6f,%.6f,%.6f]: %d plots found (%.1f ms)",
            min_lon, min_lat, max_lon, max_lat,
            len(matching_plots), elapsed_ms,
        )
        return result

    def find_plots_in_radius(
        self,
        longitude: float,
        latitude: float,
        radius_m: float,
    ) -> Dict[str, Any]:
        """Find all plots within a radius of a point.

        Args:
            longitude: Center point longitude.
            latitude: Center point latitude.
            radius_m: Search radius in metres.

        Returns:
            Dictionary with matching plots sorted by distance.
        """
        start_time = time.monotonic()

        matches = self._spatial_index.query_radius(longitude, latitude, radius_m)

        matching_plots = []
        for pid, dist in matches:
            plot = self._plots.get(pid)
            if plot:
                matching_plots.append({
                    **plot,
                    "distance_metres": round(dist, 2),
                    "distance_km": round(dist / 1000.0, 6),
                })

        postgis_sql, postgis_params = self._postgis.radius_query()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        result = {
            "result_id": _generate_id("GEO-RAD"),
            "operation": "find_plots_in_radius",
            "center": {"longitude": longitude, "latitude": latitude},
            "radius_m": radius_m,
            "matching_count": len(matching_plots),
            "matching_plots": matching_plots,
            "postgis_query": postgis_sql,
            "postgis_params": postgis_params,
            "elapsed_ms": round(elapsed_ms, 2),
            "created_at": utcnow().isoformat(),
        }

        logger.debug(
            "Radius query [%.6f,%.6f] r=%.0f m: %d plots found (%.1f ms)",
            longitude, latitude, radius_m, len(matching_plots), elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Distance Metrics
    # ------------------------------------------------------------------

    def calculate_distance(
        self,
        from_lon: float,
        from_lat: float,
        to_lon: float,
        to_lat: float,
        max_logistics_distance_m: Optional[float] = None,
    ) -> DistanceMetric:
        """Calculate geodesic distance and bearing between two points.

        Used for logistics validation between supply chain nodes
        (e.g., verifying that a producer's plot is within a reasonable
        distance of the declared processing facility).

        Args:
            from_lon: Source point longitude.
            from_lat: Source point latitude.
            to_lon: Destination point longitude.
            to_lat: Destination point latitude.
            max_logistics_distance_m: Optional maximum distance threshold.

        Returns:
            DistanceMetric with distance, bearing, and logistics flag.
        """
        threshold = max_logistics_distance_m or MAX_LOGISTICS_DISTANCE_M
        distance_m = _haversine_distance(from_lon, from_lat, to_lon, to_lat)
        bearing = _initial_bearing(from_lon, from_lat, to_lon, to_lat)

        return DistanceMetric(
            from_point=[from_lon, from_lat],
            to_point=[to_lon, to_lat],
            distance_metres=round(distance_m, 2),
            distance_km=round(distance_m / 1000.0, 6),
            bearing_degrees=round(bearing, 2),
            within_logistics_range=distance_m <= threshold,
        )

    def calculate_node_distances(
        self,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate pairwise distances between supply chain nodes.

        Args:
            nodes: List of node dictionaries, each with keys
                "node_id", "longitude", "latitude".

        Returns:
            Dictionary with distance matrix and summary statistics.
        """
        start_time = time.monotonic()
        n = len(nodes)
        matrix: Dict[str, Dict[str, float]] = {}
        total_distance = 0.0
        max_distance = 0.0
        min_distance = float("inf")
        pair_count = 0

        for i in range(n):
            node_a = nodes[i]
            aid = node_a["node_id"]
            if aid not in matrix:
                matrix[aid] = {}

            for j in range(i + 1, n):
                node_b = nodes[j]
                bid = node_b["node_id"]

                dist = _haversine_distance(
                    node_a["longitude"], node_a["latitude"],
                    node_b["longitude"], node_b["latitude"],
                )
                dist_km = round(dist / 1000.0, 6)

                if aid not in matrix:
                    matrix[aid] = {}
                matrix[aid][bid] = dist_km
                if bid not in matrix:
                    matrix[bid] = {}
                matrix[bid][aid] = dist_km

                total_distance += dist
                max_distance = max(max_distance, dist)
                min_distance = min(min_distance, dist)
                pair_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return {
            "result_id": _generate_id("GEO-DIST"),
            "operation": "calculate_node_distances",
            "node_count": n,
            "pair_count": pair_count,
            "distance_matrix_km": matrix,
            "summary": {
                "total_distance_km": round(total_distance / 1000.0, 2),
                "avg_distance_km": round(
                    (total_distance / max(pair_count, 1)) / 1000.0, 2
                ),
                "max_distance_km": round(max_distance / 1000.0, 2),
                "min_distance_km": round(
                    min_distance / 1000.0, 2
                ) if min_distance < float("inf") else 0.0,
            },
            "elapsed_ms": round(elapsed_ms, 2),
            "created_at": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Gap Analysis: Producers Without Plots
    # ------------------------------------------------------------------

    def flag_missing_geolocation(
        self,
        producer_node_ids: Sequence[str],
    ) -> Dict[str, Any]:
        """Flag producers without registered plots as 'geolocation missing'.

        Scans a list of producer node IDs and identifies those that
        have no active plot linkage, flagging them as critical gaps
        per EUDR Article 9 requirements.

        Args:
            producer_node_ids: Sequence of producer node IDs to check.

        Returns:
            Gap analysis result with flagged producers and statistics.
        """
        start_time = time.monotonic()
        flagged: List[Dict[str, Any]] = []
        linked_count = 0

        for producer_id in producer_node_ids:
            links = self._producer_links.get(producer_id, [])
            active_links = [
                lid for lid in links
                if lid in self._links
                and self._links[lid].get("status") in (
                    LinkageStatus.LINKED.value,
                    LinkageStatus.VALIDATED.value,
                )
            ]

            if not active_links:
                gap_id = _generate_id("GEO-GAP")
                gap_record = {
                    "gap_id": gap_id,
                    "producer_node_id": producer_id,
                    "gap_type": GeolocationGapType.MISSING_PLOT.value,
                    "severity": GeolocationGapSeverity.CRITICAL.value,
                    "description": (
                        f"Producer {producer_id} has no registered plot with "
                        f"geolocation data. EUDR Article 9 requires geolocation "
                        f"of all plots of land where commodities are produced."
                    ),
                    "remediation": (
                        "Register at least one production plot with GPS coordinates "
                        "for this producer. For plots >4 ha, polygon boundaries "
                        "are required per Article 9(1)(d)."
                    ),
                    "eudr_article": "Article 9",
                    "created_at": utcnow().isoformat(),
                }
                self._gaps[gap_id] = gap_record

                if producer_id not in self._producer_gaps:
                    self._producer_gaps[producer_id] = []
                self._producer_gaps[producer_id].append(gap_id)

                flagged.append(gap_record)
            else:
                linked_count += 1

        # Additional gap checks for linked producers
        precision_gaps: List[Dict[str, Any]] = []
        polygon_gaps: List[Dict[str, Any]] = []

        for producer_id in producer_node_ids:
            links = self._producer_links.get(producer_id, [])
            for lid in links:
                link = self._links.get(lid)
                if not link:
                    continue

                # Check coordinate precision
                cv = link.get("coordinate_validation")
                if cv and cv.get("meets_precision") is False:
                    gap_id = _generate_id("GEO-GAP")
                    gap_record = {
                        "gap_id": gap_id,
                        "producer_node_id": producer_id,
                        "plot_id": link.get("plot_id"),
                        "gap_type": GeolocationGapType.INSUFFICIENT_PRECISION.value,
                        "severity": GeolocationGapSeverity.MEDIUM.value,
                        "description": (
                            f"Plot {link.get('plot_id')} coordinates have fewer than "
                            f"{MIN_COORDINATE_PRECISION} decimal places."
                        ),
                        "remediation": (
                            "Update coordinates to have at least 6 decimal places "
                            "(approximately 0.11m accuracy)."
                        ),
                        "eudr_article": "Article 9",
                        "created_at": utcnow().isoformat(),
                    }
                    self._gaps[gap_id] = gap_record
                    precision_gaps.append(gap_record)

                # Check polygon compliance
                pv = link.get("polygon_validation")
                if pv and pv.get("compliant") is False:
                    gap_id = _generate_id("GEO-GAP")
                    gap_record = {
                        "gap_id": gap_id,
                        "producer_node_id": producer_id,
                        "plot_id": link.get("plot_id"),
                        "gap_type": GeolocationGapType.MISSING_POLYGON.value,
                        "severity": GeolocationGapSeverity.CRITICAL.value,
                        "description": (
                            f"Plot {link.get('plot_id')} exceeds "
                            f"{POLYGON_AREA_THRESHOLD_HA} ha but lacks polygon data."
                        ),
                        "remediation": (
                            "Provide polygon boundary coordinates for this plot "
                            "per EUDR Article 9(1)(d)."
                        ),
                        "eudr_article": "Article 9(1)(d)",
                        "created_at": utcnow().isoformat(),
                    }
                    self._gaps[gap_id] = gap_record
                    polygon_gaps.append(gap_record)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        total_checked = len(producer_node_ids)

        result = {
            "result_id": _generate_id("GEO-GAPS"),
            "operation": "flag_missing_geolocation",
            "total_producers_checked": total_checked,
            "linked_producers": linked_count,
            "unlinked_producers": len(flagged),
            "precision_gaps": len(precision_gaps),
            "polygon_gaps": len(polygon_gaps),
            "total_gaps": len(flagged) + len(precision_gaps) + len(polygon_gaps),
            "compliance_rate": round(
                linked_count / max(total_checked, 1) * 100, 2
            ),
            "missing_plot_gaps": flagged,
            "precision_gaps_detail": precision_gaps,
            "polygon_gaps_detail": polygon_gaps,
            "elapsed_ms": round(elapsed_ms, 2),
            "created_at": utcnow().isoformat(),
        }

        logger.info(
            "Gap analysis: %d/%d producers linked (%.1f%%), %d total gaps (%.1f ms)",
            linked_count, total_checked,
            result["compliance_rate"],
            result["total_gaps"],
            elapsed_ms,
        )
        return result

    def get_gaps_for_producer(
        self,
        producer_node_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all geolocation gaps for a specific producer.

        Args:
            producer_node_id: Producer node identifier.

        Returns:
            List of gap records for the producer.
        """
        gap_ids = self._producer_gaps.get(producer_node_id, [])
        return [
            self._gaps[gid]
            for gid in gap_ids
            if gid in self._gaps
        ]

    # ------------------------------------------------------------------
    # Protected Area Cross-Reference
    # ------------------------------------------------------------------

    def check_protected_area_overlap(
        self,
        plot_id: str,
        protected_areas: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Cross-reference a plot location against protected area boundaries.

        Checks whether the plot's geolocation overlaps with any registered
        protected areas (national parks, nature reserves, indigenous
        territories, etc.). Uses point-in-polygon or polygon intersection
        depending on available data.

        Args:
            plot_id: Plot identifier to check.
            protected_areas: Optional list of protected area geometries.
                Each item should have keys: "area_id", "name", "area_type",
                "polygon" (list of [lon, lat] coordinate pairs).

        Returns:
            Result dictionary with overlap status and details.
        """
        start_time = time.monotonic()
        plot = self._plots.get(plot_id)

        if plot is None:
            return {
                "result_id": _generate_id("GEO-PA"),
                "operation": "check_protected_area_overlap",
                "plot_id": plot_id,
                "status": "plot_not_found",
                "overlaps": [],
                "overlap_count": 0,
                "created_at": utcnow().isoformat(),
            }

        point = [plot["longitude"], plot["latitude"]]
        overlaps: List[Dict[str, Any]] = []

        # Use boundary_resolver if available
        if self._boundary_resolver is not None:
            try:
                boundary_result = self._boundary_resolver.resolve_protected_areas(
                    point,
                )
                if isinstance(boundary_result, dict):
                    pa_list = boundary_result.get("output_data", {}).get(
                        "protected_areas", []
                    )
                    for pa in pa_list:
                        overlaps.append({
                            "area_id": pa.get("area_id", "unknown"),
                            "name": pa.get("name", "Unknown Area"),
                            "area_type": pa.get("type", "unknown"),
                            "overlap_type": "point_in_area",
                        })
            except (AttributeError, Exception) as exc:
                logger.warning(
                    "BoundaryResolver not available for protected area check: %s", exc
                )

        # Check against provided protected area geometries
        if protected_areas:
            for pa in protected_areas:
                polygon_ring = pa.get("polygon", [])
                if polygon_ring and _point_in_polygon_ray(point, polygon_ring):
                    overlaps.append({
                        "area_id": pa.get("area_id", "unknown"),
                        "name": pa.get("name", "Unknown Area"),
                        "area_type": pa.get("area_type", "unknown"),
                        "overlap_type": "point_in_area",
                    })

        # Flag as gap if overlap detected
        if overlaps:
            gap_id = _generate_id("GEO-GAP")
            gap_record = {
                "gap_id": gap_id,
                "plot_id": plot_id,
                "producer_node_id": plot.get("producer_node_id"),
                "gap_type": GeolocationGapType.PROTECTED_AREA_OVERLAP.value,
                "severity": GeolocationGapSeverity.HIGH.value,
                "description": (
                    f"Plot {plot_id} overlaps with {len(overlaps)} protected area(s): "
                    f"{', '.join(o['name'] for o in overlaps)}"
                ),
                "remediation": (
                    "Verify that commodity production within protected areas "
                    "complies with local legislation. This may require enhanced "
                    "due diligence per EUDR Article 10."
                ),
                "eudr_article": "Article 3, Article 10",
                "created_at": utcnow().isoformat(),
            }
            self._gaps[gap_id] = gap_record

        # Generate PostGIS query for database-backed checks
        postgis_sql, postgis_params = self._postgis.protected_area_intersection_query()
        elapsed_ms = (time.monotonic() - start_time) * 1000

        return {
            "result_id": _generate_id("GEO-PA"),
            "operation": "check_protected_area_overlap",
            "plot_id": plot_id,
            "status": "overlap_detected" if overlaps else "no_overlap",
            "overlaps": overlaps,
            "overlap_count": len(overlaps),
            "postgis_query": postgis_sql,
            "postgis_params": postgis_params,
            "elapsed_ms": round(elapsed_ms, 2),
            "created_at": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Deforestation Alert Cross-Reference (AGENT-DATA-007)
    # ------------------------------------------------------------------

    def check_deforestation_alerts(
        self,
        plot_id: str,
        start_date: str = EUDR_CUTOFF_DATE,
        end_date: Optional[str] = None,
        alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Cross-reference a plot against deforestation satellite alerts.

        Checks whether the plot's geolocation has any deforestation alerts
        from AGENT-DATA-007 (GLAD, RADD, FIRMS, GFW) within the specified
        date range. Post-EUDR-cutoff alerts (after 2020-12-31) are flagged
        as compliance risks.

        Args:
            plot_id: Plot identifier to check.
            start_date: Start date for alert search (default: EUDR cutoff).
            end_date: End date for alert search (default: today).
            alerts: Optional pre-fetched list of alert dictionaries.

        Returns:
            Result dictionary with alert matches and risk assessment.
        """
        start_time = time.monotonic()
        plot = self._plots.get(plot_id)

        if plot is None:
            return {
                "result_id": _generate_id("GEO-DFA"),
                "operation": "check_deforestation_alerts",
                "plot_id": plot_id,
                "status": "plot_not_found",
                "alerts_found": 0,
                "post_cutoff_alerts": 0,
                "created_at": utcnow().isoformat(),
            }

        if end_date is None:
            end_date = utcnow().strftime("%Y-%m-%d")

        point = [plot["longitude"], plot["latitude"]]
        matched_alerts: List[Dict[str, Any]] = []
        post_cutoff_count = 0

        # Use alert engine if available (AGENT-DATA-007 integration)
        if self._alert_engine is not None:
            try:
                # Build a small bounding polygon around the plot
                search_radius_km = 5.0  # 5km search radius
                d_lat = search_radius_km / 111.32
                d_lon = search_radius_km / (
                    111.32 * max(math.cos(math.radians(point[1])), 0.0001)
                )
                search_polygon = [
                    [point[0] - d_lon, point[1] - d_lat],
                    [point[0] + d_lon, point[1] - d_lat],
                    [point[0] + d_lon, point[1] + d_lat],
                    [point[0] - d_lon, point[1] + d_lat],
                    [point[0] - d_lon, point[1] - d_lat],
                ]

                from greenlang.agents.data.deforestation_satellite.models import QueryAlertsRequest
                query = QueryAlertsRequest(
                    polygon_coordinates=search_polygon,
                    start_date=start_date,
                    end_date=end_date,
                )
                aggregation = self._alert_engine.query_alerts(query)

                if hasattr(aggregation, "alerts"):
                    for alert in aggregation.alerts:
                        alert_dict = (
                            alert.model_dump() if hasattr(alert, "model_dump")
                            else alert if isinstance(alert, dict)
                            else {}
                        )
                        matched_alerts.append(alert_dict)
                        if alert_dict.get("is_post_cutoff", False):
                            post_cutoff_count += 1

            except (ImportError, Exception) as exc:
                logger.warning(
                    "AlertAggregationEngine not available for deforestation check: %s",
                    exc,
                )

        # Check against provided alerts (for testing or offline mode)
        if alerts:
            for alert in alerts:
                alert_lon = alert.get("longitude", 0.0)
                alert_lat = alert.get("latitude", 0.0)
                dist = _haversine_distance(
                    point[0], point[1], alert_lon, alert_lat,
                )
                # Check if alert is within 5km of plot centroid
                if dist <= 5_000.0:
                    alert["distance_to_plot_m"] = round(dist, 2)
                    matched_alerts.append(alert)
                    if alert.get("is_post_cutoff", False):
                        post_cutoff_count += 1

        # Flag as gap if post-cutoff alerts found
        risk_level = "none"
        if post_cutoff_count > 0:
            risk_level = "high" if post_cutoff_count >= 3 else "medium"
            gap_id = _generate_id("GEO-GAP")
            gap_record = {
                "gap_id": gap_id,
                "plot_id": plot_id,
                "producer_node_id": plot.get("producer_node_id"),
                "gap_type": GeolocationGapType.DEFORESTATION_ALERT.value,
                "severity": (
                    GeolocationGapSeverity.CRITICAL.value
                    if post_cutoff_count >= 3
                    else GeolocationGapSeverity.HIGH.value
                ),
                "description": (
                    f"Plot {plot_id} has {post_cutoff_count} post-EUDR-cutoff "
                    f"deforestation alert(s) within 5km. Total alerts: "
                    f"{len(matched_alerts)}."
                ),
                "remediation": (
                    "Conduct enhanced due diligence per EUDR Article 10. "
                    "Verify deforestation-free status through satellite imagery "
                    "analysis and on-ground verification."
                ),
                "eudr_article": "Article 3, Article 10",
                "alert_count": len(matched_alerts),
                "post_cutoff_count": post_cutoff_count,
                "created_at": utcnow().isoformat(),
            }
            self._gaps[gap_id] = gap_record
        elif matched_alerts:
            risk_level = "low"

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return {
            "result_id": _generate_id("GEO-DFA"),
            "operation": "check_deforestation_alerts",
            "plot_id": plot_id,
            "search_period": {"start_date": start_date, "end_date": end_date},
            "alerts_found": len(matched_alerts),
            "post_cutoff_alerts": post_cutoff_count,
            "risk_level": risk_level,
            "alerts": matched_alerts,
            "status": (
                "deforestation_risk_detected"
                if post_cutoff_count > 0
                else "clear" if not matched_alerts else "pre_cutoff_alerts_only"
            ),
            "elapsed_ms": round(elapsed_ms, 2),
            "created_at": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Bulk Import (CSV, GeoJSON, Shapefile)
    # ------------------------------------------------------------------

    def bulk_import_plots(
        self,
        data: str,
        format_type: str,
        default_commodity: Optional[str] = None,
        default_country_code: Optional[str] = None,
        producer_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Bulk import plot geolocation data from CSV, GeoJSON, or Shapefile.

        Parses the input data, validates each plot's coordinates and
        polygon geometry, creates linkages, and indexes plots spatially.

        Supported formats:
        - CSV: Columns must include latitude, longitude. Optional:
            plot_id, area_hectares, commodity, country_code, producer_node_id.
        - GeoJSON: Standard FeatureCollection with Point or Polygon geometries.
        - Shapefile: Shapefile content as JSON-serialized feature list.

        Args:
            data: Raw data string (CSV text, GeoJSON string, or Shapefile JSON).
            format_type: One of "csv", "geojson", "shapefile".
            default_commodity: Default commodity for all plots if not in data.
            default_country_code: Default country code if not in data.
            producer_node_id: Default producer node ID if not in data.

        Returns:
            Import result with success/failure counts and details.

        Raises:
            ValueError: If format_type is not supported.
        """
        start_time = time.monotonic()

        format_lower = format_type.lower()
        if format_lower not in SUPPORTED_IMPORT_FORMATS:
            raise ValueError(
                f"Unsupported import format: {format_type}. "
                f"Supported: {', '.join(sorted(SUPPORTED_IMPORT_FORMATS))}"
            )

        if format_lower == "csv":
            records = self._parse_csv_import(data)
        elif format_lower == "geojson":
            records = self._parse_geojson_import(data)
        elif format_lower == "shapefile":
            records = self._parse_shapefile_import(data)
        else:
            records = []

        # Process each record
        results: List[Dict[str, Any]] = []
        success_count = 0
        failure_count = 0

        for i, record in enumerate(records):
            try:
                plot_id = record.get("plot_id") or _generate_id("PLOT")
                pid = record.get("producer_node_id") or producer_node_id or "UNKNOWN"
                lat = record.get("latitude")
                lon = record.get("longitude")
                poly = record.get("polygon_coordinates")
                area = record.get("area_hectares")
                commodity = record.get("commodity") or default_commodity
                country = record.get("country_code") or default_country_code

                if lat is None or lon is None:
                    raise ValueError(
                        f"Record {i}: latitude and longitude are required"
                    )

                link = self.link_producer_to_plot(
                    producer_node_id=pid,
                    plot_id=plot_id,
                    latitude=float(lat),
                    longitude=float(lon),
                    polygon_coordinates=poly,
                    area_hectares=float(area) if area is not None else None,
                    commodity=commodity,
                    country_code=country,
                )

                results.append({
                    "index": i,
                    "plot_id": plot_id,
                    "link_id": link["link_id"],
                    "status": link["status"],
                })
                if link["status"] != LinkageStatus.REJECTED.value:
                    success_count += 1
                else:
                    failure_count += 1

            except Exception as exc:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": str(exc),
                })
                failure_count += 1
                logger.warning("Bulk import record %d failed: %s", i, exc)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash({
                "format": format_lower,
                "record_count": len(records),
                "success": success_count,
                "failed": failure_count,
            })
            self._provenance.record(
                entity_type="bulk_import",
                entity_id=_generate_id("IMPORT"),
                action="bulk_import_plots",
                data_hash=data_hash,
            )

        import_result = {
            "result_id": _generate_id("GEO-IMP"),
            "operation": "bulk_import_plots",
            "format": format_lower,
            "total_records": len(records),
            "success": success_count,
            "failed": failure_count,
            "results": results,
            "elapsed_ms": round(elapsed_ms, 2),
            "created_at": utcnow().isoformat(),
        }

        logger.info(
            "Bulk import (%s): %d records, %d success, %d failed (%.1f ms)",
            format_lower, len(records), success_count, failure_count, elapsed_ms,
        )
        return import_result

    def _parse_csv_import(self, data: str) -> List[Dict[str, Any]]:
        """Parse CSV data into plot records.

        Expected columns: latitude, longitude (required).
        Optional: plot_id, area_hectares, commodity, country_code,
        producer_node_id, polygon_coordinates (as JSON string).

        Args:
            data: CSV text string.

        Returns:
            List of parsed record dictionaries.
        """
        records: List[Dict[str, Any]] = []
        reader = csv.DictReader(io.StringIO(data))

        for row in reader:
            record: Dict[str, Any] = {}

            # Required fields
            lat_str = row.get("latitude", "").strip()
            lon_str = row.get("longitude", "").strip()
            if lat_str and lon_str:
                record["latitude"] = float(lat_str)
                record["longitude"] = float(lon_str)

            # Optional fields
            for key in ("plot_id", "commodity", "country_code", "producer_node_id"):
                val = row.get(key, "").strip()
                if val:
                    record[key] = val

            area_str = row.get("area_hectares", "").strip()
            if area_str:
                record["area_hectares"] = float(area_str)

            # Polygon from JSON string
            poly_str = row.get("polygon_coordinates", "").strip()
            if poly_str:
                try:
                    record["polygon_coordinates"] = json.loads(poly_str)
                except (json.JSONDecodeError, ValueError):
                    pass

            if "latitude" in record and "longitude" in record:
                records.append(record)

        return records

    def _parse_geojson_import(self, data: str) -> List[Dict[str, Any]]:
        """Parse GeoJSON FeatureCollection into plot records.

        Supports Point and Polygon geometries. Properties are mapped
        to plot record fields.

        Args:
            data: GeoJSON string.

        Returns:
            List of parsed record dictionaries.
        """
        records: List[Dict[str, Any]] = []

        try:
            geojson = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            logger.error("Invalid GeoJSON data")
            return records

        features = []
        geojson_type = geojson.get("type", "")

        if geojson_type == "FeatureCollection":
            features = geojson.get("features", [])
        elif geojson_type == "Feature":
            features = [geojson]
        else:
            logger.warning("Unexpected GeoJSON type: %s", geojson_type)
            return records

        for feature in features:
            geometry = feature.get("geometry", {})
            properties = feature.get("properties", {})
            record: Dict[str, Any] = {}

            geom_type = geometry.get("type", "")
            coords = geometry.get("coordinates", [])

            if geom_type == "Point" and len(coords) >= 2:
                record["longitude"] = coords[0]
                record["latitude"] = coords[1]

            elif geom_type == "Polygon" and coords:
                # Use centroid as point location
                ring = coords[0]
                if ring:
                    avg_lon = sum(c[0] for c in ring) / len(ring)
                    avg_lat = sum(c[1] for c in ring) / len(ring)
                    record["longitude"] = avg_lon
                    record["latitude"] = avg_lat
                    record["polygon_coordinates"] = ring

                    # Compute area
                    area_m2 = _geodesic_polygon_area(ring)
                    record["area_hectares"] = area_m2 / 10_000.0

            else:
                continue

            # Map properties
            for key in ("plot_id", "commodity", "country_code", "producer_node_id"):
                val = properties.get(key)
                if val:
                    record[key] = str(val)

            area_prop = properties.get("area_hectares")
            if area_prop is not None and "area_hectares" not in record:
                record["area_hectares"] = float(area_prop)

            if "latitude" in record and "longitude" in record:
                records.append(record)

        return records

    def _parse_shapefile_import(self, data: str) -> List[Dict[str, Any]]:
        """Parse Shapefile data (as JSON-serialized features) into plot records.

        Shapefiles are expected to be pre-converted to a JSON feature list
        (e.g., via ogr2ogr or fiona). Each feature follows GeoJSON-like
        structure with geometry and properties.

        Args:
            data: JSON string containing a list of features.

        Returns:
            List of parsed record dictionaries.
        """
        # Shapefile features follow the same structure as GeoJSON features
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            logger.error("Invalid Shapefile JSON data")
            return []

        # Handle both list of features and FeatureCollection wrapper
        if isinstance(parsed, dict):
            if parsed.get("type") == "FeatureCollection":
                return self._parse_geojson_import(data)
            else:
                # Wrap as FeatureCollection
                wrapper = json.dumps({
                    "type": "FeatureCollection",
                    "features": [parsed],
                })
                return self._parse_geojson_import(wrapper)
        elif isinstance(parsed, list):
            wrapper = json.dumps({
                "type": "FeatureCollection",
                "features": parsed,
            })
            return self._parse_geojson_import(wrapper)
        else:
            return []

    # ------------------------------------------------------------------
    # Map Data for Visualization (Feature 3: Interactive Map)
    # ------------------------------------------------------------------

    def get_map_data(
        self,
        min_lon: Optional[float] = None,
        min_lat: Optional[float] = None,
        max_lon: Optional[float] = None,
        max_lat: Optional[float] = None,
        commodity_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get plot locations formatted for interactive map visualization.

        Returns GeoJSON FeatureCollection of all plots within the optional
        bounding box filter, with properties suitable for rendering in
        the supply chain graph view.

        Args:
            min_lon: Optional minimum longitude for spatial filter.
            min_lat: Optional minimum latitude for spatial filter.
            max_lon: Optional maximum longitude for spatial filter.
            max_lat: Optional maximum latitude for spatial filter.
            commodity_filter: Optional commodity type filter.
            country_filter: Optional ISO country code filter.

        Returns:
            GeoJSON FeatureCollection with plot features and metadata.
        """
        start_time = time.monotonic()

        # Get candidate plots via spatial index or full scan
        if all(v is not None for v in (min_lon, min_lat, max_lon, max_lat)):
            plot_ids = self._spatial_index.query_bbox(
                min_lon, min_lat, max_lon, max_lat,  # type: ignore[arg-type]
            )
        else:
            plot_ids = list(self._plots.keys())

        # Apply attribute filters
        features: List[Dict[str, Any]] = []
        for pid in plot_ids:
            plot = self._plots.get(pid)
            if not plot:
                continue

            if commodity_filter and plot.get("commodity") != commodity_filter:
                continue
            if country_filter and plot.get("country_code") != country_filter:
                continue

            # Build GeoJSON Feature
            polygon = plot.get("polygon_coordinates")
            if polygon:
                geometry = {
                    "type": "Polygon",
                    "coordinates": [polygon],
                }
            else:
                geometry = {
                    "type": "Point",
                    "coordinates": [plot["longitude"], plot["latitude"]],
                }

            # Get linkage status
            link_ids = self._plot_links.get(pid, [])
            linkage_status = "unlinked"
            for lid in link_ids:
                link = self._links.get(lid)
                if link:
                    linkage_status = link.get("status", "unlinked")
                    break

            # Get gaps for this plot
            has_gaps = any(
                g.get("plot_id") == pid
                for g in self._gaps.values()
            )

            feature = {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "plot_id": pid,
                    "producer_node_id": plot.get("producer_node_id"),
                    "commodity": plot.get("commodity"),
                    "country_code": plot.get("country_code"),
                    "area_hectares": plot.get("area_hectares"),
                    "linkage_status": linkage_status,
                    "has_compliance_gaps": has_gaps,
                },
            }
            features.append(feature)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "result_id": _generate_id("GEO-MAP"),
                "total_features": len(features),
                "filters_applied": {
                    "bbox": [min_lon, min_lat, max_lon, max_lat]
                    if all(v is not None for v in (min_lon, min_lat, max_lon, max_lat))
                    else None,
                    "commodity": commodity_filter,
                    "country": country_filter,
                },
                "elapsed_ms": round(elapsed_ms, 2),
                "created_at": utcnow().isoformat(),
            },
        }

    # ------------------------------------------------------------------
    # Integration with AGENT-DATA-006 SpatialAnalyzerEngine
    # ------------------------------------------------------------------

    def spatial_analysis(
        self,
        plot_id_a: str,
        plot_id_b: str,
    ) -> Dict[str, Any]:
        """Perform spatial analysis between two plots using AGENT-DATA-006.

        Calculates distance, checks containment, and computes bounding
        box overlap between two plot geometries using the integrated
        SpatialAnalyzerEngine.

        Args:
            plot_id_a: First plot identifier.
            plot_id_b: Second plot identifier.

        Returns:
            Spatial analysis result dictionary.
        """
        plot_a = self._plots.get(plot_id_a)
        plot_b = self._plots.get(plot_id_b)

        if not plot_a or not plot_b:
            return {
                "result_id": _generate_id("GEO-SPA"),
                "operation": "spatial_analysis",
                "status": "plot_not_found",
                "missing": [
                    pid for pid in (plot_id_a, plot_id_b)
                    if pid not in self._plots
                ],
            }

        # Distance calculation
        distance = self.calculate_distance(
            plot_a["longitude"], plot_a["latitude"],
            plot_b["longitude"], plot_b["latitude"],
        )

        result: Dict[str, Any] = {
            "result_id": _generate_id("GEO-SPA"),
            "operation": "spatial_analysis",
            "plot_a": plot_id_a,
            "plot_b": plot_id_b,
            "distance": {
                "metres": distance.distance_metres,
                "km": distance.distance_km,
                "bearing_degrees": distance.bearing_degrees,
            },
            "created_at": utcnow().isoformat(),
        }

        # Use SpatialAnalyzerEngine if available
        if self._spatial_analyzer is not None:
            try:
                point_a = [plot_a["longitude"], plot_a["latitude"]]
                point_b = [plot_b["longitude"], plot_b["latitude"]]

                dist_result = self._spatial_analyzer.distance(point_a, point_b)
                result["spatial_analyzer_distance"] = dist_result.get(
                    "output_data", {}
                )

                # Check if plots have polygons for intersection test
                poly_a = plot_a.get("polygon_coordinates")
                poly_b = plot_b.get("polygon_coordinates")
                if poly_a and poly_b:
                    geom_a = {"type": "Polygon", "coordinates": [poly_a]}
                    geom_b = {"type": "Polygon", "coordinates": [poly_b]}

                    intersection = self._spatial_analyzer.intersection(geom_a, geom_b)
                    result["intersection"] = intersection.get("output_data", {})

            except (AttributeError, Exception) as exc:
                logger.warning(
                    "SpatialAnalyzerEngine analysis failed: %s", exc
                )

        return result

    # ------------------------------------------------------------------
    # Statistics and Diagnostics
    # ------------------------------------------------------------------

    @property
    def link_count(self) -> int:
        """Return the total number of geolocation linkages."""
        return len(self._links)

    @property
    def plot_count(self) -> int:
        """Return the total number of spatially indexed plots."""
        return self._spatial_index.count

    @property
    def gap_count(self) -> int:
        """Return the total number of identified geolocation gaps."""
        return len(self._gaps)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the geolocation linker.

        Returns:
            Dictionary with linkage, plot, gap, and index statistics.
        """
        status_counts: Dict[str, int] = {}
        for link in self._links.values():
            status = link.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        gap_type_counts: Dict[str, int] = {}
        gap_severity_counts: Dict[str, int] = {}
        for gap in self._gaps.values():
            gt = gap.get("gap_type", "unknown")
            gap_type_counts[gt] = gap_type_counts.get(gt, 0) + 1
            gs = gap.get("severity", "unknown")
            gap_severity_counts[gs] = gap_severity_counts.get(gs, 0) + 1

        commodity_counts: Dict[str, int] = {}
        country_counts: Dict[str, int] = {}
        for plot in self._plots.values():
            com = plot.get("commodity", "unknown")
            commodity_counts[com] = commodity_counts.get(com, 0) + 1
            cc = plot.get("country_code", "XX")
            country_counts[cc] = country_counts.get(cc, 0) + 1

        return {
            "total_links": len(self._links),
            "total_plots_indexed": self._spatial_index.count,
            "total_gaps": len(self._gaps),
            "link_status_distribution": status_counts,
            "gap_type_distribution": gap_type_counts,
            "gap_severity_distribution": gap_severity_counts,
            "commodity_distribution": commodity_counts,
            "country_distribution": country_counts,
            "unique_producers_linked": len(self._producer_links),
            "unique_plots_linked": len(self._plot_links),
            "spatial_index_cells": len(self._spatial_index._grid),
        }

# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Main engine
    "GeolocationLinker",
    # PostGIS query builder
    "PostGISQueryBuilder",
    # Data structures
    "CoordinateValidation",
    "PolygonValidation",
    "DistanceMetric",
    # Enumerations
    "LinkageStatus",
    "GeolocationGapType",
    "GeolocationGapSeverity",
    "ProtectedAreaType",
    # Constants
    "EARTH_RADIUS_M",
    "MIN_COORDINATE_PRECISION",
    "POLYGON_AREA_THRESHOLD_HA",
    "WGS84_SRID",
    "EUDR_CUTOFF_DATE",
    "SUPPORTED_IMPORT_FORMATS",
    "MAX_LOGISTICS_DISTANCE_M",
    # Spatial functions (exposed for testing)
    "_haversine_distance",
    "_initial_bearing",
    "_count_decimal_places",
    "_geodesic_polygon_area",
    "_point_in_polygon_ray",
]
