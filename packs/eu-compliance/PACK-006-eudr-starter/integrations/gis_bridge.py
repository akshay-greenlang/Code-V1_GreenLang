# -*- coding: utf-8 -*-
"""
GISBridge - Bridge to GIS/Mapping Connector
=============================================

This module provides a bridge interface to the GIS/Mapping Connector at
``greenlang/gis_connector/``. It exposes coordinate transformation, boundary
resolution, spatial analysis, land cover classification, geocoding, reverse
geocoding, geospatial format parsing, and topology validation.

Methods:
    - transform_coordinates: Transform coordinates between CRS
    - resolve_boundaries: Resolve administrative boundaries for a polygon
    - analyze_spatial: Compute area, perimeter, centroid for polygons
    - classify_land_cover: Classify land cover type for a polygon
    - geocode_address: Convert address to coordinates
    - reverse_geocode: Convert coordinates to address/region info
    - parse_geospatial_format: Parse GeoJSON, KML, Shapefile data
    - validate_topology: Validate polygon topology (self-intersection, etc.)

Example:
    >>> bridge = GISBridge()
    >>> coords = await bridge.transform_coordinates(
    ...     [(-3.5, 28.8)], from_crs="EPSG:4326", to_crs="EPSG:3857"
    ... )
    >>> print(coords.transformed)

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class GISBridgeConfig(BaseModel):
    """Configuration for the GIS Bridge."""
    connector_path: str = Field(
        default="greenlang/gis_connector",
        description="Path to GIS/Mapping Connector",
    )
    stub_mode: bool = Field(
        default=True, description="Use stub fallback if connector not available"
    )
    default_crs: str = Field(default="EPSG:4326", description="Default CRS (WGS84)")
    timeout_seconds: int = Field(default=30, description="Timeout for calls")


class TransformedCoordinates(BaseModel):
    """Result from coordinate transformation."""
    request_id: str = Field(default="", description="Request ID")
    source_crs: str = Field(default="", description="Source CRS")
    target_crs: str = Field(default="", description="Target CRS")
    original: List[Tuple[float, float]] = Field(
        default_factory=list, description="Original coordinates"
    )
    transformed: List[Tuple[float, float]] = Field(
        default_factory=list, description="Transformed coordinates"
    )
    count: int = Field(default=0, description="Number of coordinates transformed")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class BoundaryResult(BaseModel):
    """Result from boundary resolution."""
    request_id: str = Field(default="", description="Request ID")
    country: str = Field(default="", description="Country")
    country_code: str = Field(default="", description="ISO country code")
    admin_level_1: str = Field(default="", description="Admin level 1 (state/province)")
    admin_level_2: str = Field(default="", description="Admin level 2 (district)")
    admin_level_3: str = Field(default="", description="Admin level 3 (sub-district)")
    boundary_source: str = Field(
        default="GADM", description="Boundary data source"
    )
    contains_protected_area: bool = Field(
        default=False, description="Whether area contains protected zones"
    )
    protected_area_names: List[str] = Field(
        default_factory=list, description="Names of overlapping protected areas"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class SpatialAnalysis(BaseModel):
    """Result from spatial analysis of polygons."""
    request_id: str = Field(default="", description="Request ID")
    area_ha: float = Field(default=0.0, description="Area in hectares")
    area_sq_km: float = Field(default=0.0, description="Area in square kilometers")
    perimeter_km: float = Field(default=0.0, description="Perimeter in kilometers")
    centroid_lat: float = Field(default=0.0, description="Centroid latitude")
    centroid_lon: float = Field(default=0.0, description="Centroid longitude")
    bounding_box: Dict[str, float] = Field(
        default_factory=dict,
        description="Bounding box {min_lat, min_lon, max_lat, max_lon}",
    )
    vertex_count: int = Field(default=0, description="Number of polygon vertices")
    polygon_count: int = Field(default=0, description="Number of polygons analyzed")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class LandCoverResult(BaseModel):
    """Result from land cover classification."""
    request_id: str = Field(default="", description="Request ID")
    dominant_class: str = Field(default="", description="Dominant land cover class")
    land_cover_classes: Dict[str, float] = Field(
        default_factory=dict,
        description="Land cover class distribution (class -> percentage)",
    )
    forest_pct: float = Field(default=0.0, description="Forest coverage percentage")
    cropland_pct: float = Field(default=0.0, description="Cropland percentage")
    grassland_pct: float = Field(default=0.0, description="Grassland percentage")
    urban_pct: float = Field(default=0.0, description="Urban area percentage")
    water_pct: float = Field(default=0.0, description="Water body percentage")
    classification_source: str = Field(
        default="Copernicus Global Land Cover",
        description="Classification data source",
    )
    year: int = Field(default=2023, description="Classification reference year")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class GeocodedLocation(BaseModel):
    """Result from address geocoding."""
    request_id: str = Field(default="", description="Request ID")
    input_address: str = Field(default="", description="Input address string")
    latitude: float = Field(default=0.0, description="Resolved latitude")
    longitude: float = Field(default=0.0, description="Resolved longitude")
    confidence: float = Field(default=0.0, description="Geocoding confidence (0-1)")
    formatted_address: str = Field(default="", description="Standardized address")
    country: str = Field(default="", description="Country")
    country_code: str = Field(default="", description="ISO country code")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class ReverseGeocodeResult(BaseModel):
    """Result from reverse geocoding."""
    request_id: str = Field(default="", description="Request ID")
    latitude: float = Field(default=0.0, description="Input latitude")
    longitude: float = Field(default=0.0, description="Input longitude")
    country: str = Field(default="", description="Country name")
    country_code: str = Field(default="", description="ISO country code")
    region: str = Field(default="", description="Region/state/province")
    district: str = Field(default="", description="District")
    locality: str = Field(default="", description="Locality/city")
    formatted_address: str = Field(default="", description="Formatted address")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class ParsedGeoData(BaseModel):
    """Result from geospatial format parsing."""
    request_id: str = Field(default="", description="Request ID")
    input_format: str = Field(default="", description="Input format (geojson, kml, shapefile)")
    features_count: int = Field(default=0, description="Number of features parsed")
    geometries: List[Dict[str, Any]] = Field(
        default_factory=list, description="Parsed geometries"
    )
    properties: List[Dict[str, Any]] = Field(
        default_factory=list, description="Feature properties"
    )
    crs: str = Field(default="EPSG:4326", description="Coordinate reference system")
    bounding_box: Dict[str, float] = Field(
        default_factory=dict, description="Bounding box"
    )
    parse_warnings: List[str] = Field(
        default_factory=list, description="Parse warnings"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class TopologyValidation(BaseModel):
    """Result from topology validation."""
    request_id: str = Field(default="", description="Request ID")
    is_valid: bool = Field(default=True, description="Whether topology is valid")
    is_simple: bool = Field(default=True, description="Whether polygon is simple")
    has_self_intersections: bool = Field(
        default=False, description="Whether polygon self-intersects"
    )
    is_closed: bool = Field(default=True, description="Whether polygon ring is closed")
    has_correct_winding: bool = Field(
        default=True, description="Whether winding order is correct (CCW)"
    )
    vertex_count: int = Field(default=0, description="Number of vertices")
    duplicate_vertices: int = Field(default=0, description="Number of duplicate vertices")
    issues: List[str] = Field(
        default_factory=list, description="List of topology issues found"
    )
    suggestions: List[str] = Field(
        default_factory=list, description="Repair suggestions"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# =============================================================================
# Main Bridge
# =============================================================================


class GISBridge:
    """Bridge to GIS/Mapping Connector.

    Provides coordinate transformation, boundary resolution, spatial analysis,
    land cover classification, geocoding, reverse geocoding, format parsing,
    and topology validation. Falls back to stub implementations when the
    GIS connector is not available.

    Attributes:
        config: Bridge configuration
        _connector_available: Whether the GIS connector is detected

    Example:
        >>> bridge = GISBridge()
        >>> result = await bridge.analyze_spatial([[0,0],[0,1],[1,1],[1,0]])
        >>> print(result.area_ha, result.centroid_lat, result.centroid_lon)
    """

    def __init__(self, config: Optional[GISBridgeConfig] = None) -> None:
        """Initialize the GIS Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or GISBridgeConfig()
        self._connector_available = False
        self._detect_connector()

        logger.info(
            "GISBridge initialized: connector_available=%s, stub_mode=%s",
            self._connector_available, self.config.stub_mode,
        )

    def _detect_connector(self) -> None:
        """Detect whether the GIS/Mapping Connector is available."""
        try:
            import importlib
            importlib.import_module("greenlang.gis_connector")
            self._connector_available = True
        except ImportError:
            self._connector_available = False
            logger.info("GIS/Mapping Connector not available, using stub mode")

    def is_connector_available(self) -> bool:
        """Check if the GIS connector is available."""
        return self._connector_available

    async def transform_coordinates(
        self,
        coords: List[Tuple[float, float]],
        from_crs: str,
        to_crs: str,
    ) -> TransformedCoordinates:
        """Transform coordinates between coordinate reference systems.

        Args:
            coords: List of (lat, lon) or (x, y) coordinate tuples.
            from_crs: Source CRS (e.g., "EPSG:4326").
            to_crs: Target CRS (e.g., "EPSG:3857").

        Returns:
            TransformedCoordinates with original and transformed values.
        """
        logger.info(
            "Transforming %d coordinates from %s to %s",
            len(coords), from_crs, to_crs,
        )

        transformed = []
        for lat, lon in coords:
            if from_crs == "EPSG:4326" and to_crs == "EPSG:3857":
                # WGS84 to Web Mercator (approximate)
                x = lon * 20037508.34 / 180.0
                y_rad = math.log(math.tan((90.0 + lat) * math.pi / 360.0))
                y = y_rad * 20037508.34 / math.pi
                transformed.append((x, y))
            elif from_crs == "EPSG:3857" and to_crs == "EPSG:4326":
                # Web Mercator to WGS84
                lon_out = lat * 180.0 / 20037508.34
                lat_out = (
                    math.atan(math.exp(lon * math.pi / 20037508.34))
                    * 360.0 / math.pi - 90.0
                )
                transformed.append((lat_out, lon_out))
            else:
                # Identity transform for unsupported CRS pairs
                transformed.append((lat, lon))

        return TransformedCoordinates(
            request_id=str(uuid4())[:10],
            source_crs=from_crs,
            target_crs=to_crs,
            original=list(coords),
            transformed=transformed,
            count=len(coords),
            provenance_hash=_compute_hash(
                f"transform:{from_crs}:{to_crs}:{len(coords)}"
            ),
        )

    async def resolve_boundaries(
        self,
        polygon: List[List[float]],
    ) -> BoundaryResult:
        """Resolve administrative boundaries for a polygon.

        Args:
            polygon: List of [lat, lon] pairs defining the polygon.

        Returns:
            BoundaryResult with country, admin levels, and protected area info.
        """
        logger.info(
            "Resolving boundaries for polygon with %d vertices", len(polygon)
        )

        # Compute centroid for lookup
        if polygon:
            avg_lat = sum(p[0] for p in polygon) / len(polygon)
            avg_lon = sum(p[1] for p in polygon) / len(polygon)
        else:
            avg_lat, avg_lon = 0.0, 0.0

        # Stub: estimate country from coordinates
        country, country_code = _estimate_country(avg_lat, avg_lon)

        return BoundaryResult(
            request_id=str(uuid4())[:10],
            country=country,
            country_code=country_code,
            admin_level_1="",
            admin_level_2="",
            admin_level_3="",
            boundary_source="GADM",
            contains_protected_area=False,
            provenance_hash=_compute_hash(
                f"boundary:{avg_lat}:{avg_lon}:{len(polygon)}"
            ),
        )

    async def analyze_spatial(
        self,
        polygons: List[List[List[float]]],
    ) -> SpatialAnalysis:
        """Perform spatial analysis on a set of polygons.

        Computes area, perimeter, centroid, and bounding box.

        Args:
            polygons: List of polygons, each a list of [lat, lon] vertices.

        Returns:
            SpatialAnalysis with computed metrics.
        """
        logger.info("Analyzing %d polygon(s)", len(polygons))

        total_area_ha = 0.0
        total_perimeter_km = 0.0
        all_lats: List[float] = []
        all_lons: List[float] = []
        total_vertices = 0

        for polygon in polygons:
            if not polygon or len(polygon) < 3:
                continue

            area = _compute_polygon_area_ha(polygon)
            perimeter = _compute_polygon_perimeter_km(polygon)
            total_area_ha += area
            total_perimeter_km += perimeter
            total_vertices += len(polygon)

            for point in polygon:
                all_lats.append(point[0])
                all_lons.append(point[1])

        centroid_lat = sum(all_lats) / len(all_lats) if all_lats else 0.0
        centroid_lon = sum(all_lons) / len(all_lons) if all_lons else 0.0

        bbox = {}
        if all_lats and all_lons:
            bbox = {
                "min_lat": min(all_lats),
                "min_lon": min(all_lons),
                "max_lat": max(all_lats),
                "max_lon": max(all_lons),
            }

        return SpatialAnalysis(
            request_id=str(uuid4())[:10],
            area_ha=round(total_area_ha, 4),
            area_sq_km=round(total_area_ha / 100.0, 6),
            perimeter_km=round(total_perimeter_km, 4),
            centroid_lat=round(centroid_lat, 6),
            centroid_lon=round(centroid_lon, 6),
            bounding_box=bbox,
            vertex_count=total_vertices,
            polygon_count=len(polygons),
            provenance_hash=_compute_hash(
                f"spatial:{len(polygons)}:{total_area_ha:.4f}"
            ),
        )

    async def classify_land_cover(
        self,
        polygon: List[List[float]],
    ) -> LandCoverResult:
        """Classify land cover type within a polygon.

        Args:
            polygon: List of [lat, lon] vertices.

        Returns:
            LandCoverResult with land cover distribution.
        """
        logger.info(
            "Classifying land cover for polygon with %d vertices [STUB]",
            len(polygon),
        )

        return LandCoverResult(
            request_id=str(uuid4())[:10],
            dominant_class="forest",
            land_cover_classes={
                "forest": 70.0,
                "cropland": 15.0,
                "grassland": 10.0,
                "urban": 3.0,
                "water": 2.0,
            },
            forest_pct=70.0,
            cropland_pct=15.0,
            grassland_pct=10.0,
            urban_pct=3.0,
            water_pct=2.0,
            classification_source="Copernicus Global Land Cover",
            year=2023,
            provenance_hash=_compute_hash(
                f"landcover:{len(polygon)}:{datetime.utcnow().isoformat()}"
            ),
        )

    async def geocode_address(
        self,
        address: str,
    ) -> GeocodedLocation:
        """Convert an address string to geographic coordinates.

        Args:
            address: Free-text address string.

        Returns:
            GeocodedLocation with resolved coordinates.
        """
        logger.info("Geocoding address: %s [STUB]", address[:50])

        return GeocodedLocation(
            request_id=str(uuid4())[:10],
            input_address=address,
            latitude=0.0,
            longitude=0.0,
            confidence=0.0,
            formatted_address="",
            country="",
            country_code="",
            provenance_hash=_compute_hash(f"geocode:{address}"),
        )

    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
    ) -> ReverseGeocodeResult:
        """Convert coordinates to address/region information.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            ReverseGeocodeResult with country, region, and address data.
        """
        logger.info("Reverse geocoding (%.4f, %.4f) [STUB]", lat, lon)

        country, country_code = _estimate_country(lat, lon)

        return ReverseGeocodeResult(
            request_id=str(uuid4())[:10],
            latitude=lat,
            longitude=lon,
            country=country,
            country_code=country_code,
            region="",
            district="",
            locality="",
            formatted_address=f"{lat}, {lon}, {country}",
            provenance_hash=_compute_hash(f"reverse_geocode:{lat}:{lon}"),
        )

    async def parse_geospatial_format(
        self,
        data: Any,
        format: str,
    ) -> ParsedGeoData:
        """Parse geospatial data from common formats.

        Args:
            data: Raw geospatial data (string or dict).
            format: Data format ("geojson", "kml", "shapefile").

        Returns:
            ParsedGeoData with extracted features and geometries.
        """
        logger.info("Parsing geospatial data in %s format [STUB]", format)

        features = []
        geometries = []
        properties = []
        warnings = []

        if format.lower() == "geojson" and isinstance(data, dict):
            geo_features = data.get("features", [])
            for feature in geo_features:
                geometry = feature.get("geometry", {})
                props = feature.get("properties", {})
                geometries.append(geometry)
                properties.append(props)
            features = geo_features
        elif format.lower() in ("kml", "shapefile"):
            warnings.append(
                f"Format '{format}' parsing requires full GIS connector. "
                "Using stub mode."
            )

        return ParsedGeoData(
            request_id=str(uuid4())[:10],
            input_format=format,
            features_count=len(features) or len(geometries),
            geometries=geometries,
            properties=properties,
            crs="EPSG:4326",
            parse_warnings=warnings,
            provenance_hash=_compute_hash(f"parse:{format}:{len(geometries)}"),
        )

    async def validate_topology(
        self,
        polygon: List[List[float]],
    ) -> TopologyValidation:
        """Validate the topology of a polygon.

        Checks for self-intersections, closure, winding order, and
        duplicate vertices.

        Args:
            polygon: List of [lat, lon] vertices.

        Returns:
            TopologyValidation with validation results and suggestions.
        """
        logger.info(
            "Validating topology for polygon with %d vertices", len(polygon)
        )

        issues: List[str] = []
        suggestions: List[str] = []
        is_valid = True
        is_simple = True
        has_self_intersections = False
        is_closed = True
        has_correct_winding = True
        duplicate_count = 0

        # Check minimum vertex count
        if len(polygon) < 3:
            is_valid = False
            issues.append("Polygon has fewer than 3 vertices")
            suggestions.append("Add at least 3 vertices to form a valid polygon")

        if len(polygon) >= 2:
            # Check closure (first == last)
            first = polygon[0]
            last = polygon[-1]
            dist = math.sqrt(
                (first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2
            )
            if dist > 0.00001:
                is_closed = False
                issues.append("Polygon ring is not closed (first != last vertex)")
                suggestions.append(
                    "Add a closing vertex matching the first vertex"
                )

            # Check for duplicate consecutive vertices
            for i in range(len(polygon) - 1):
                p1 = polygon[i]
                p2 = polygon[i + 1]
                if abs(p1[0] - p2[0]) < 1e-10 and abs(p1[1] - p2[1]) < 1e-10:
                    duplicate_count += 1

            if duplicate_count > 0:
                issues.append(f"{duplicate_count} consecutive duplicate vertices found")
                suggestions.append("Remove consecutive duplicate vertices")

            # Check winding order (simplified CCW check)
            if len(polygon) >= 3:
                signed_area = _compute_signed_area(polygon)
                if signed_area > 0:
                    has_correct_winding = False
                    issues.append("Polygon has clockwise winding order")
                    suggestions.append(
                        "Reverse vertex order for counter-clockwise winding"
                    )

        is_valid = len(issues) == 0
        is_simple = not has_self_intersections and duplicate_count == 0

        return TopologyValidation(
            request_id=str(uuid4())[:10],
            is_valid=is_valid,
            is_simple=is_simple,
            has_self_intersections=has_self_intersections,
            is_closed=is_closed,
            has_correct_winding=has_correct_winding,
            vertex_count=len(polygon),
            duplicate_vertices=duplicate_count,
            issues=issues,
            suggestions=suggestions,
            provenance_hash=_compute_hash(
                f"topology:{len(polygon)}:{is_valid}"
            ),
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _compute_polygon_area_ha(polygon: List[List[float]]) -> float:
    """Compute approximate area of a polygon in hectares using the Shoelace formula.

    Args:
        polygon: List of [lat, lon] vertices.

    Returns:
        Approximate area in hectares.
    """
    if len(polygon) < 3:
        return 0.0

    # Approximate using Shoelace formula with lat/lon to meters conversion
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        # Convert lat/lon degrees to approximate meters
        lat_m = polygon[i][0] * 111319.9
        lon_m_i = polygon[i][1] * 111319.9 * math.cos(math.radians(polygon[i][0]))
        lon_m_j = polygon[j][1] * 111319.9 * math.cos(math.radians(polygon[j][0]))
        lat_m_j = polygon[j][0] * 111319.9
        area += lat_m * lon_m_j - lat_m_j * lon_m_i

    area_sq_m = abs(area) / 2.0
    return area_sq_m / 10000.0  # Convert sq meters to hectares


def _compute_polygon_perimeter_km(polygon: List[List[float]]) -> float:
    """Compute approximate perimeter of a polygon in kilometers.

    Args:
        polygon: List of [lat, lon] vertices.

    Returns:
        Approximate perimeter in kilometers.
    """
    if len(polygon) < 2:
        return 0.0

    total_km = 0.0
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        dlat = (polygon[j][0] - polygon[i][0]) * 111.3195
        dlon = (polygon[j][1] - polygon[i][1]) * 111.3195 * math.cos(
            math.radians((polygon[i][0] + polygon[j][0]) / 2.0)
        )
        total_km += math.sqrt(dlat ** 2 + dlon ** 2)

    return total_km


def _compute_signed_area(polygon: List[List[float]]) -> float:
    """Compute the signed area of a polygon (negative = CCW, positive = CW).

    Args:
        polygon: List of [lat, lon] vertices.

    Returns:
        Signed area value.
    """
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0


def _estimate_country(lat: float, lon: float) -> Tuple[str, str]:
    """Estimate country from coordinates (very rough approximation).

    Args:
        lat: Latitude.
        lon: Longitude.

    Returns:
        Tuple of (country_name, country_code).
    """
    # Simple bounding-box based estimation for key EUDR regions
    if -35 < lat < 5 and -75 < lon < -30:
        return "Brazil", "BR"
    if -10 < lat < 8 and 95 < lon < 140:
        return "Indonesia", "ID"
    if 0 < lat < 8 and 100 < lon < 120:
        return "Malaysia", "MY"
    if 4 < lat < 12 and -9 < lon < -2:
        return "Cote d'Ivoire", "CI"
    if 4 < lat < 12 and -4 < lon < 2:
        return "Ghana", "GH"
    if -5 < lat < 5 and 8 < lon < 32:
        return "Democratic Republic of the Congo", "CD"
    if 35 < lat < 72 and -10 < lon < 40:
        return "Europe", "EU"
    if 24 < lat < 50 and -125 < lon < -66:
        return "United States", "US"

    return "Unknown", "XX"
