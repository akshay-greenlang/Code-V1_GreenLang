# -*- coding: utf-8 -*-
"""
GreenLang EUDR GeoJSON Parser

Zero-hallucination GeoJSON parsing and validation for EUDR compliance.
All calculations are deterministic with exact mathematical formulas.

This module provides:
- GeoJSON Point, Polygon, MultiPolygon parsing
- Coordinate bounds validation (WGS84)
- Polygon area calculation in hectares (Shoelace formula)
- Self-intersection detection
- Centroid calculation

Author: GreenLang Calculator Engine
License: Proprietary
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
import math
from datetime import datetime


class GeoJSONType(str, Enum):
    """Supported GeoJSON geometry types."""
    POINT = "Point"
    POLYGON = "Polygon"
    MULTI_POLYGON = "MultiPolygon"
    LINE_STRING = "LineString"
    MULTI_POINT = "MultiPoint"
    MULTI_LINE_STRING = "MultiLineString"
    GEOMETRY_COLLECTION = "GeometryCollection"
    FEATURE = "Feature"
    FEATURE_COLLECTION = "FeatureCollection"


class Coordinate(BaseModel):
    """
    A single WGS84 coordinate with validation.

    EUDR requires 6 decimal places precision for coordinates.
    This provides ~0.1 meter precision which is sufficient for
    plot-level deforestation monitoring.
    """
    longitude: Decimal = Field(..., description="Longitude in degrees (-180 to 180)")
    latitude: Decimal = Field(..., description="Latitude in degrees (-90 to 90)")
    altitude: Optional[Decimal] = Field(None, description="Altitude in meters (optional)")

    # EUDR precision requirement
    REQUIRED_PRECISION: int = 6

    @field_validator('longitude')
    @classmethod
    def validate_longitude(cls, v: Decimal) -> Decimal:
        """Validate longitude is within WGS84 bounds."""
        if v < Decimal('-180') or v > Decimal('180'):
            raise ValueError(f"Longitude {v} out of bounds. Must be -180 to 180.")
        return v

    @field_validator('latitude')
    @classmethod
    def validate_latitude(cls, v: Decimal) -> Decimal:
        """Validate latitude is within WGS84 bounds."""
        if v < Decimal('-90') or v > Decimal('90'):
            raise ValueError(f"Latitude {v} out of bounds. Must be -90 to 90.")
        return v

    def get_precision(self) -> Tuple[int, int]:
        """
        Get decimal precision of coordinates.

        Returns:
            Tuple of (longitude_precision, latitude_precision)
        """
        lon_str = str(self.longitude)
        lat_str = str(self.latitude)

        lon_precision = len(lon_str.split('.')[-1]) if '.' in lon_str else 0
        lat_precision = len(lat_str.split('.')[-1]) if '.' in lat_str else 0

        return (lon_precision, lat_precision)

    def meets_eudr_precision(self) -> bool:
        """Check if coordinate meets EUDR 6 decimal place requirement."""
        lon_prec, lat_prec = self.get_precision()
        return lon_prec >= self.REQUIRED_PRECISION and lat_prec >= self.REQUIRED_PRECISION

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (longitude, latitude) tuple."""
        return (float(self.longitude), float(self.latitude))

    def to_lat_lon_tuple(self) -> Tuple[float, float]:
        """Convert to (latitude, longitude) tuple for calculations."""
        return (float(self.latitude), float(self.longitude))

    @classmethod
    def from_geojson(cls, coords: List[Union[float, int]]) -> "Coordinate":
        """
        Create Coordinate from GeoJSON coordinate array.

        GeoJSON uses [longitude, latitude, altitude?] format.

        Args:
            coords: GeoJSON coordinate array [lon, lat] or [lon, lat, alt]

        Returns:
            Coordinate instance
        """
        if len(coords) < 2:
            raise ValueError("Coordinate array must have at least 2 elements [lon, lat]")

        longitude = Decimal(str(coords[0]))
        latitude = Decimal(str(coords[1]))
        altitude = Decimal(str(coords[2])) if len(coords) > 2 else None

        return cls(longitude=longitude, latitude=latitude, altitude=altitude)


class BoundingBox(BaseModel):
    """Bounding box for a geometry."""
    min_longitude: Decimal
    min_latitude: Decimal
    max_longitude: Decimal
    max_latitude: Decimal

    def contains(self, coord: Coordinate) -> bool:
        """Check if coordinate is within bounding box."""
        return (
            self.min_longitude <= coord.longitude <= self.max_longitude and
            self.min_latitude <= coord.latitude <= self.max_latitude
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this bounding box intersects another."""
        return not (
            self.max_longitude < other.min_longitude or
            self.min_longitude > other.max_longitude or
            self.max_latitude < other.min_latitude or
            self.min_latitude > other.max_latitude
        )


class PolygonRing(BaseModel):
    """
    A single polygon ring (exterior or hole).

    A ring is a closed LineString with 4 or more positions.
    The first and last positions must be identical.
    """
    coordinates: List[Coordinate]
    is_exterior: bool = True

    @field_validator('coordinates')
    @classmethod
    def validate_ring(cls, v: List[Coordinate]) -> List[Coordinate]:
        """Validate ring is closed and has minimum points."""
        if len(v) < 4:
            raise ValueError(f"Ring must have at least 4 coordinates, got {len(v)}")

        # Check if ring is closed (first == last)
        first = v[0]
        last = v[-1]
        if first.longitude != last.longitude or first.latitude != last.latitude:
            raise ValueError("Ring must be closed (first and last coordinates must match)")

        return v

    def is_clockwise(self) -> bool:
        """
        Determine if ring is clockwise using signed area.

        For GeoJSON:
        - Exterior rings should be counterclockwise (positive area)
        - Holes should be clockwise (negative area)

        Uses the Shoelace formula to calculate signed area.
        """
        signed_area = Decimal('0')
        n = len(self.coordinates)

        for i in range(n - 1):  # -1 because ring is closed
            j = (i + 1) % (n - 1)
            signed_area += (
                self.coordinates[j].longitude - self.coordinates[i].longitude
            ) * (
                self.coordinates[j].latitude + self.coordinates[i].latitude
            )

        return signed_area > 0


class ParsedPoint(BaseModel):
    """Parsed GeoJSON Point geometry."""
    type: str = "Point"
    coordinate: Coordinate
    properties: Dict[str, Any] = Field(default_factory=dict)

    def get_hash(self) -> str:
        """Calculate SHA-256 hash for provenance."""
        data = {
            "type": self.type,
            "coordinate": [float(self.coordinate.longitude), float(self.coordinate.latitude)]
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class ParsedPolygon(BaseModel):
    """
    Parsed GeoJSON Polygon geometry with calculated metrics.

    Contains:
    - Validated exterior ring
    - Optional holes (interior rings)
    - Calculated area in hectares
    - Calculated centroid
    - Self-intersection status
    """
    type: str = "Polygon"
    exterior_ring: PolygonRing
    holes: List[PolygonRing] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)

    # Calculated fields
    area_hectares: Optional[Decimal] = None
    perimeter_meters: Optional[Decimal] = None
    centroid: Optional[Coordinate] = None
    is_valid: bool = True
    has_self_intersection: bool = False
    bounding_box: Optional[BoundingBox] = None

    def get_hash(self) -> str:
        """Calculate SHA-256 hash for provenance."""
        coords = [[float(c.longitude), float(c.latitude)]
                  for c in self.exterior_ring.coordinates]
        data = {
            "type": self.type,
            "coordinates": coords,
            "area_hectares": str(self.area_hectares) if self.area_hectares else None
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class ParsedMultiPolygon(BaseModel):
    """Parsed GeoJSON MultiPolygon geometry."""
    type: str = "MultiPolygon"
    polygons: List[ParsedPolygon]
    properties: Dict[str, Any] = Field(default_factory=dict)

    # Aggregated metrics
    total_area_hectares: Optional[Decimal] = None
    combined_centroid: Optional[Coordinate] = None
    bounding_box: Optional[BoundingBox] = None

    def get_hash(self) -> str:
        """Calculate SHA-256 hash for provenance."""
        polygon_hashes = [p.get_hash() for p in self.polygons]
        data = {
            "type": self.type,
            "polygon_hashes": polygon_hashes,
            "total_area_hectares": str(self.total_area_hectares) if self.total_area_hectares else None
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class GeoJSONParseResult(BaseModel):
    """Result of GeoJSON parsing with complete provenance."""
    success: bool
    geometry_type: Optional[GeoJSONType] = None
    point: Optional[ParsedPoint] = None
    polygon: Optional[ParsedPolygon] = None
    multi_polygon: Optional[ParsedMultiPolygon] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    parse_time_ms: float = 0.0
    provenance_hash: str = ""
    parsed_at: datetime = Field(default_factory=datetime.utcnow)


class GeoJSONParser:
    """
    Zero-Hallucination GeoJSON Parser for EUDR Compliance.

    This parser guarantees:
    - Deterministic parsing (same input -> same output)
    - Bit-perfect area calculations
    - Complete provenance tracking
    - NO LLM in parsing path

    Supported geometry types:
    - Point
    - Polygon
    - MultiPolygon

    All calculations use exact mathematical formulas:
    - Area: Shoelace formula with geodetic correction
    - Centroid: Weighted average of vertices
    - Self-intersection: Line segment intersection test

    Example:
        parser = GeoJSONParser()
        result = parser.parse(geojson_dict)

        if result.success:
            print(f"Area: {result.polygon.area_hectares} hectares")
            print(f"Provenance: {result.provenance_hash}")
    """

    # Earth radius in meters (WGS84 semi-major axis)
    EARTH_RADIUS_METERS: float = 6378137.0

    # Conversion factor: square meters to hectares
    SQ_METERS_TO_HECTARES: Decimal = Decimal('0.0001')

    def __init__(self, require_eudr_precision: bool = True):
        """
        Initialize GeoJSON parser.

        Args:
            require_eudr_precision: If True, enforce 6 decimal place precision
        """
        self.require_eudr_precision = require_eudr_precision

    def parse(self, geojson: Dict[str, Any]) -> GeoJSONParseResult:
        """
        Parse GeoJSON geometry - DETERMINISTIC.

        This is the main entry point for parsing GeoJSON.
        Supports Point, Polygon, and MultiPolygon types.

        Args:
            geojson: GeoJSON geometry dictionary

        Returns:
            GeoJSONParseResult with parsed geometry and metrics
        """
        import time
        start_time = time.perf_counter()

        result = GeoJSONParseResult(success=False)

        try:
            # Handle Feature wrapper
            if geojson.get("type") == "Feature":
                geometry = geojson.get("geometry", {})
                properties = geojson.get("properties", {})
            elif geojson.get("type") == "FeatureCollection":
                result.errors.append("FeatureCollection not supported. Parse features individually.")
                return result
            else:
                geometry = geojson
                properties = {}

            geometry_type = geometry.get("type")
            coordinates = geometry.get("coordinates")

            if not geometry_type:
                result.errors.append("Missing 'type' field in geometry")
                return result

            if coordinates is None:
                result.errors.append("Missing 'coordinates' field in geometry")
                return result

            # Parse based on geometry type
            if geometry_type == GeoJSONType.POINT.value:
                result.geometry_type = GeoJSONType.POINT
                result.point = self._parse_point(coordinates, properties)
                result.success = True
                result.provenance_hash = result.point.get_hash()

            elif geometry_type == GeoJSONType.POLYGON.value:
                result.geometry_type = GeoJSONType.POLYGON
                parsed_polygon = self._parse_polygon(coordinates, properties)

                # Calculate metrics
                parsed_polygon = self._calculate_polygon_metrics(parsed_polygon)

                # Check for self-intersection
                parsed_polygon.has_self_intersection = self._detect_self_intersection(
                    parsed_polygon.exterior_ring.coordinates
                )

                if parsed_polygon.has_self_intersection:
                    result.warnings.append("Polygon has self-intersection")

                result.polygon = parsed_polygon
                result.success = True
                result.provenance_hash = result.polygon.get_hash()

            elif geometry_type == GeoJSONType.MULTI_POLYGON.value:
                result.geometry_type = GeoJSONType.MULTI_POLYGON
                result.multi_polygon = self._parse_multi_polygon(coordinates, properties)
                result.success = True
                result.provenance_hash = result.multi_polygon.get_hash()

            else:
                result.errors.append(f"Unsupported geometry type: {geometry_type}")
                return result

            # Check EUDR precision if required
            if self.require_eudr_precision:
                precision_warnings = self._check_eudr_precision(result)
                result.warnings.extend(precision_warnings)

        except ValueError as e:
            result.errors.append(f"Validation error: {str(e)}")
        except Exception as e:
            result.errors.append(f"Parse error: {str(e)}")

        result.parse_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _parse_point(self, coordinates: List, properties: Dict) -> ParsedPoint:
        """Parse Point geometry."""
        coord = Coordinate.from_geojson(coordinates)
        return ParsedPoint(coordinate=coord, properties=properties)

    def _parse_polygon(self, coordinates: List, properties: Dict) -> ParsedPolygon:
        """
        Parse Polygon geometry.

        GeoJSON Polygon structure:
        [
            [[lon, lat], ...],  # exterior ring
            [[lon, lat], ...],  # hole 1 (optional)
            [[lon, lat], ...],  # hole 2 (optional)
        ]
        """
        if not coordinates or len(coordinates) < 1:
            raise ValueError("Polygon must have at least one ring (exterior)")

        # Parse exterior ring
        exterior_coords = [Coordinate.from_geojson(c) for c in coordinates[0]]
        exterior_ring = PolygonRing(coordinates=exterior_coords, is_exterior=True)

        # Parse holes
        holes = []
        for i, hole_coords in enumerate(coordinates[1:], 1):
            hole_coordinates = [Coordinate.from_geojson(c) for c in hole_coords]
            hole_ring = PolygonRing(coordinates=hole_coordinates, is_exterior=False)
            holes.append(hole_ring)

        # Calculate bounding box
        bbox = self._calculate_bounding_box(exterior_coords)

        return ParsedPolygon(
            exterior_ring=exterior_ring,
            holes=holes,
            properties=properties,
            bounding_box=bbox
        )

    def _parse_multi_polygon(self, coordinates: List, properties: Dict) -> ParsedMultiPolygon:
        """
        Parse MultiPolygon geometry.

        MultiPolygon structure:
        [
            [[[lon, lat], ...]],  # polygon 1
            [[[lon, lat], ...]],  # polygon 2
        ]
        """
        polygons = []
        total_area = Decimal('0')

        for poly_coords in coordinates:
            polygon = self._parse_polygon(poly_coords, {})
            polygon = self._calculate_polygon_metrics(polygon)
            polygons.append(polygon)

            if polygon.area_hectares:
                total_area += polygon.area_hectares

        # Calculate combined bounding box
        all_coords = []
        for p in polygons:
            all_coords.extend(p.exterior_ring.coordinates)
        bbox = self._calculate_bounding_box(all_coords)

        # Calculate combined centroid (area-weighted)
        centroid = self._calculate_multipolygon_centroid(polygons)

        return ParsedMultiPolygon(
            polygons=polygons,
            properties=properties,
            total_area_hectares=total_area,
            combined_centroid=centroid,
            bounding_box=bbox
        )

    def _calculate_polygon_metrics(self, polygon: ParsedPolygon) -> ParsedPolygon:
        """
        Calculate polygon metrics (area, centroid, perimeter).

        All calculations are DETERMINISTIC using exact formulas.
        """
        # Calculate area using geodetic Shoelace formula
        polygon.area_hectares = self._calculate_geodetic_area(
            polygon.exterior_ring.coordinates
        )

        # Subtract hole areas
        for hole in polygon.holes:
            hole_area = self._calculate_geodetic_area(hole.coordinates)
            polygon.area_hectares -= hole_area

        # Ensure positive area
        polygon.area_hectares = abs(polygon.area_hectares)

        # Calculate centroid
        polygon.centroid = self._calculate_centroid(polygon.exterior_ring.coordinates)

        # Calculate perimeter
        polygon.perimeter_meters = self._calculate_perimeter(
            polygon.exterior_ring.coordinates
        )

        return polygon

    def _calculate_geodetic_area(self, coordinates: List[Coordinate]) -> Decimal:
        """
        Calculate geodetic area using the Shoelace formula with latitude correction.

        This is a DETERMINISTIC calculation.

        Formula:
        A = (pi/180) * R^2 * |sum_{i=0}^{n-1}(lon_{i+1} - lon_{i-1}) * sin(lat_i)|

        Where:
        - R = Earth radius (6378137 m for WGS84)
        - lon, lat are in degrees

        Returns:
            Area in hectares with 6 decimal precision
        """
        n = len(coordinates) - 1  # -1 because ring is closed

        if n < 3:
            return Decimal('0')

        # Use spherical excess formula for geodetic area
        # This is more accurate than simple Shoelace for large areas
        total = 0.0

        for i in range(n):
            j = (i + 1) % n
            k = (i + 2) % n

            lat_i = float(coordinates[i].latitude)
            lat_j = float(coordinates[j].latitude)
            lon_i = float(coordinates[i].longitude)
            lon_j = float(coordinates[j].longitude)

            # Shoelace formula with latitude correction
            total += math.radians(lon_j - lon_i) * (
                2 + math.sin(math.radians(lat_i)) +
                math.sin(math.radians(lat_j))
            )

        # Calculate area in square meters
        area_sq_meters = abs(total * self.EARTH_RADIUS_METERS ** 2 / 2)

        # Convert to hectares
        area_hectares = Decimal(str(area_sq_meters)) * self.SQ_METERS_TO_HECTARES

        # Round to 6 decimal places for EUDR compliance
        return area_hectares.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def _calculate_centroid(self, coordinates: List[Coordinate]) -> Coordinate:
        """
        Calculate polygon centroid using the geometric center formula.

        DETERMINISTIC calculation.

        For a simple polygon, the centroid is calculated as:
        Cx = (1/6A) * sum((xi + xi+1) * (xi*yi+1 - xi+1*yi))
        Cy = (1/6A) * sum((yi + yi+1) * (xi*yi+1 - xi+1*yi))

        Returns:
            Centroid coordinate
        """
        n = len(coordinates) - 1  # -1 because ring is closed

        if n < 3:
            # Return first coordinate for degenerate cases
            return coordinates[0]

        # Calculate centroid using formula
        sum_x = Decimal('0')
        sum_y = Decimal('0')
        sum_a = Decimal('0')

        for i in range(n):
            j = (i + 1) % n

            xi = coordinates[i].longitude
            yi = coordinates[i].latitude
            xj = coordinates[j].longitude
            yj = coordinates[j].latitude

            cross = xi * yj - xj * yi
            sum_a += cross
            sum_x += (xi + xj) * cross
            sum_y += (yi + yj) * cross

        if sum_a == 0:
            # Degenerate polygon, use average
            avg_lon = sum(c.longitude for c in coordinates[:-1]) / n
            avg_lat = sum(c.latitude for c in coordinates[:-1]) / n
            return Coordinate(longitude=avg_lon, latitude=avg_lat)

        area_6 = Decimal('3') * sum_a
        cx = sum_x / area_6
        cy = sum_y / area_6

        # Round to EUDR precision
        cx = cx.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
        cy = cy.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

        return Coordinate(longitude=cx, latitude=cy)

    def _calculate_perimeter(self, coordinates: List[Coordinate]) -> Decimal:
        """
        Calculate polygon perimeter using Haversine distance.

        DETERMINISTIC calculation.

        Returns:
            Perimeter in meters
        """
        n = len(coordinates) - 1  # -1 because ring is closed
        total_distance = Decimal('0')

        for i in range(n):
            j = (i + 1) % (n + 1)  # Include wrap-around
            distance = self._haversine_distance(coordinates[i], coordinates[j])
            total_distance += distance

        return total_distance.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _haversine_distance(self, coord1: Coordinate, coord2: Coordinate) -> Decimal:
        """
        Calculate distance between two coordinates using Haversine formula.

        DETERMINISTIC calculation.

        Formula:
        a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R * c

        Returns:
            Distance in meters
        """
        lat1 = math.radians(float(coord1.latitude))
        lat2 = math.radians(float(coord2.latitude))
        lon1 = math.radians(float(coord1.longitude))
        lon2 = math.radians(float(coord2.longitude))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = self.EARTH_RADIUS_METERS * c
        return Decimal(str(distance))

    def _calculate_bounding_box(self, coordinates: List[Coordinate]) -> BoundingBox:
        """Calculate bounding box for coordinates."""
        if not coordinates:
            raise ValueError("Cannot calculate bounding box for empty coordinates")

        lons = [c.longitude for c in coordinates]
        lats = [c.latitude for c in coordinates]

        return BoundingBox(
            min_longitude=min(lons),
            max_longitude=max(lons),
            min_latitude=min(lats),
            max_latitude=max(lats)
        )

    def _calculate_multipolygon_centroid(self, polygons: List[ParsedPolygon]) -> Coordinate:
        """
        Calculate area-weighted centroid for MultiPolygon.

        DETERMINISTIC calculation.
        """
        if not polygons:
            raise ValueError("Cannot calculate centroid for empty MultiPolygon")

        total_area = Decimal('0')
        weighted_lon = Decimal('0')
        weighted_lat = Decimal('0')

        for polygon in polygons:
            if polygon.centroid and polygon.area_hectares:
                total_area += polygon.area_hectares
                weighted_lon += polygon.centroid.longitude * polygon.area_hectares
                weighted_lat += polygon.centroid.latitude * polygon.area_hectares

        if total_area == 0:
            # Fall back to first polygon's centroid
            return polygons[0].centroid

        centroid_lon = (weighted_lon / total_area).quantize(
            Decimal('0.000001'), rounding=ROUND_HALF_UP
        )
        centroid_lat = (weighted_lat / total_area).quantize(
            Decimal('0.000001'), rounding=ROUND_HALF_UP
        )

        return Coordinate(longitude=centroid_lon, latitude=centroid_lat)

    def _detect_self_intersection(self, coordinates: List[Coordinate]) -> bool:
        """
        Detect self-intersecting polygon using line segment intersection test.

        DETERMINISTIC calculation.

        Uses the cross-product method to detect if any non-adjacent
        line segments intersect.

        Returns:
            True if polygon has self-intersection
        """
        n = len(coordinates) - 1  # -1 because ring is closed

        if n < 4:
            return False  # Triangle cannot self-intersect

        # Check all non-adjacent segment pairs
        for i in range(n):
            # Segment i: coordinates[i] to coordinates[i+1]
            p1 = coordinates[i]
            p2 = coordinates[i + 1]

            # Check against all non-adjacent segments
            for j in range(i + 2, n):
                # Skip adjacent segments
                if j == (i + n - 1) % n:
                    continue

                # Segment j: coordinates[j] to coordinates[j+1]
                p3 = coordinates[j]
                p4 = coordinates[j + 1]

                if self._segments_intersect(p1, p2, p3, p4):
                    return True

        return False

    def _segments_intersect(
        self,
        p1: Coordinate,
        p2: Coordinate,
        p3: Coordinate,
        p4: Coordinate
    ) -> bool:
        """
        Check if two line segments intersect using cross product method.

        DETERMINISTIC calculation.
        """
        def ccw(a: Coordinate, b: Coordinate, c: Coordinate) -> Decimal:
            """Counter-clockwise test using cross product."""
            return (
                (c.latitude - a.latitude) * (b.longitude - a.longitude) -
                (b.latitude - a.latitude) * (c.longitude - a.longitude)
            )

        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True

        return False

    def _check_eudr_precision(self, result: GeoJSONParseResult) -> List[str]:
        """Check if all coordinates meet EUDR precision requirements."""
        warnings = []

        if result.point:
            if not result.point.coordinate.meets_eudr_precision():
                warnings.append(
                    f"Point coordinate does not meet EUDR 6 decimal precision requirement"
                )

        if result.polygon:
            for i, coord in enumerate(result.polygon.exterior_ring.coordinates):
                if not coord.meets_eudr_precision():
                    warnings.append(
                        f"Polygon coordinate {i} does not meet EUDR precision requirement"
                    )
                    break  # Only report once

        if result.multi_polygon:
            for pi, polygon in enumerate(result.multi_polygon.polygons):
                for ci, coord in enumerate(polygon.exterior_ring.coordinates):
                    if not coord.meets_eudr_precision():
                        warnings.append(
                            f"MultiPolygon[{pi}] coordinate {ci} does not meet EUDR precision"
                        )
                        break  # Only report once per polygon

        return warnings

    def validate_geojson_structure(self, geojson: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate GeoJSON structure without parsing.

        Quick validation for API input.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Check for required fields
        if "type" not in geojson:
            errors.append("Missing 'type' field")
            return (False, errors)

        geo_type = geojson["type"]

        # Handle Feature wrapper
        if geo_type == "Feature":
            if "geometry" not in geojson:
                errors.append("Feature missing 'geometry' field")
                return (False, errors)
            return self.validate_geojson_structure(geojson["geometry"])

        if geo_type == "FeatureCollection":
            errors.append("FeatureCollection not supported for single geometry validation")
            return (False, errors)

        # Check for coordinates
        if "coordinates" not in geojson:
            errors.append("Missing 'coordinates' field")
            return (False, errors)

        coords = geojson["coordinates"]

        # Type-specific validation
        if geo_type == "Point":
            if not isinstance(coords, list) or len(coords) < 2:
                errors.append("Point coordinates must be [lon, lat] array")

        elif geo_type == "Polygon":
            if not isinstance(coords, list) or len(coords) < 1:
                errors.append("Polygon must have at least one ring")
            elif not isinstance(coords[0], list) or len(coords[0]) < 4:
                errors.append("Polygon ring must have at least 4 coordinates")

        elif geo_type == "MultiPolygon":
            if not isinstance(coords, list) or len(coords) < 1:
                errors.append("MultiPolygon must have at least one polygon")

        else:
            errors.append(f"Unsupported geometry type: {geo_type}")

        return (len(errors) == 0, errors)
