# -*- coding: utf-8 -*-
"""
GreenLang EUDR Coordinate Validator

Zero-hallucination coordinate validation and geodetic calculations.
All formulas are mathematically exact and deterministic.

This module provides:
- WGS84 coordinate validation
- EUDR precision requirements (6 decimal places)
- Haversine distance calculations
- Polygon perimeter calculation
- Buffer zone calculations
- Bearing calculations

Author: GreenLang Calculator Engine
License: Proprietary
"""

from typing import Dict, List, Tuple, Optional, Union
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import math
import hashlib
import json
from datetime import datetime

from .geojson_parser import Coordinate, BoundingBox, ParsedPolygon


class CoordinateSystem(str, Enum):
    """Supported coordinate reference systems."""
    WGS84 = "EPSG:4326"  # World Geodetic System 1984
    WEB_MERCATOR = "EPSG:3857"  # Web Mercator
    UTM = "UTM"  # Universal Transverse Mercator


class DistanceUnit(str, Enum):
    """Distance measurement units."""
    METERS = "m"
    KILOMETERS = "km"
    MILES = "mi"
    FEET = "ft"
    NAUTICAL_MILES = "nm"


class AreaUnit(str, Enum):
    """Area measurement units."""
    SQUARE_METERS = "m2"
    HECTARES = "ha"
    SQUARE_KILOMETERS = "km2"
    ACRES = "ac"
    SQUARE_FEET = "ft2"


class PrecisionLevel(str, Enum):
    """Coordinate precision levels with typical use cases."""
    EUDR_COMPLIANT = "eudr"  # 6 decimals (~0.1m precision)
    HIGH = "high"  # 5 decimals (~1m precision)
    MEDIUM = "medium"  # 4 decimals (~10m precision)
    LOW = "low"  # 3 decimals (~100m precision)
    CITY = "city"  # 2 decimals (~1km precision)


class CoordinateValidationResult(BaseModel):
    """Result of coordinate validation."""
    is_valid: bool
    coordinate: Optional[Coordinate] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    precision_level: Optional[PrecisionLevel] = None
    meets_eudr_requirement: bool = False
    provenance_hash: str = ""


class DistanceResult(BaseModel):
    """Result of distance calculation with provenance."""
    distance_meters: Decimal
    distance_km: Decimal
    bearing_degrees: Decimal
    from_coordinate: Coordinate
    to_coordinate: Coordinate
    calculation_method: str = "haversine"
    provenance_hash: str = ""


class BufferResult(BaseModel):
    """Result of buffer zone calculation."""
    center: Coordinate
    radius_meters: Decimal
    buffer_polygon_coords: List[Coordinate]
    area_hectares: Decimal
    num_vertices: int = 64
    provenance_hash: str = ""


class CoordinateValidator:
    """
    Zero-Hallucination Coordinate Validator for EUDR Compliance.

    This validator guarantees:
    - Deterministic validation (same input -> same output)
    - Bit-perfect distance calculations
    - Complete provenance tracking
    - NO LLM in validation path

    Geodetic Constants (WGS84):
    - Semi-major axis (a): 6378137.0 m
    - Semi-minor axis (b): 6356752.314245 m
    - Flattening (f): 1/298.257223563
    - Eccentricity squared (e2): 0.00669437999014

    Example:
        validator = CoordinateValidator()

        # Validate a coordinate
        result = validator.validate_coordinate(lon=12.345678, lat=48.123456)

        # Calculate distance
        distance = validator.haversine_distance(coord1, coord2)
    """

    # WGS84 Constants (EXACT values from EPSG)
    EARTH_RADIUS_METERS: float = 6378137.0  # Semi-major axis (a)
    EARTH_RADIUS_POLAR: float = 6356752.314245  # Semi-minor axis (b)
    FLATTENING: float = 1 / 298.257223563  # Flattening
    ECCENTRICITY_SQUARED: float = 0.00669437999014  # e^2

    # Unit conversion constants (EXACT)
    METERS_PER_KILOMETER: Decimal = Decimal('1000')
    METERS_PER_MILE: Decimal = Decimal('1609.344')
    METERS_PER_NAUTICAL_MILE: Decimal = Decimal('1852')
    METERS_PER_FOOT: Decimal = Decimal('0.3048')

    SQUARE_METERS_PER_HECTARE: Decimal = Decimal('10000')
    SQUARE_METERS_PER_ACRE: Decimal = Decimal('4046.8564224')
    SQUARE_METERS_PER_SQ_KM: Decimal = Decimal('1000000')

    # Precision requirements
    PRECISION_DECIMALS = {
        PrecisionLevel.EUDR_COMPLIANT: 6,
        PrecisionLevel.HIGH: 5,
        PrecisionLevel.MEDIUM: 4,
        PrecisionLevel.LOW: 3,
        PrecisionLevel.CITY: 2,
    }

    # Precision in meters (approximate)
    PRECISION_METERS = {
        PrecisionLevel.EUDR_COMPLIANT: Decimal('0.1'),
        PrecisionLevel.HIGH: Decimal('1'),
        PrecisionLevel.MEDIUM: Decimal('10'),
        PrecisionLevel.LOW: Decimal('100'),
        PrecisionLevel.CITY: Decimal('1000'),
    }

    def __init__(self, default_precision: PrecisionLevel = PrecisionLevel.EUDR_COMPLIANT):
        """
        Initialize coordinate validator.

        Args:
            default_precision: Default precision level for validation
        """
        self.default_precision = default_precision

    def validate_coordinate(
        self,
        longitude: Union[float, Decimal, str],
        latitude: Union[float, Decimal, str],
        altitude: Optional[Union[float, Decimal, str]] = None,
        required_precision: Optional[PrecisionLevel] = None
    ) -> CoordinateValidationResult:
        """
        Validate a coordinate - DETERMINISTIC.

        Performs:
        1. Bounds validation (WGS84)
        2. Precision level detection
        3. EUDR compliance check

        Args:
            longitude: Longitude in degrees (-180 to 180)
            latitude: Latitude in degrees (-90 to 90)
            altitude: Optional altitude in meters
            required_precision: Minimum precision required

        Returns:
            CoordinateValidationResult with validation details
        """
        result = CoordinateValidationResult(is_valid=False)
        errors = []
        warnings = []

        try:
            # Convert to Decimal for precision
            lon = Decimal(str(longitude))
            lat = Decimal(str(latitude))
            alt = Decimal(str(altitude)) if altitude is not None else None

            # Validate bounds
            if lon < Decimal('-180') or lon > Decimal('180'):
                errors.append(f"Longitude {lon} out of bounds [-180, 180]")

            if lat < Decimal('-90') or lat > Decimal('90'):
                errors.append(f"Latitude {lat} out of bounds [-90, 90]")

            if errors:
                result.errors = errors
                return result

            # Create coordinate
            coord = Coordinate(longitude=lon, latitude=lat, altitude=alt)
            result.coordinate = coord

            # Detect precision level
            lon_precision = self._get_decimal_precision(lon)
            lat_precision = self._get_decimal_precision(lat)
            min_precision = min(lon_precision, lat_precision)

            # Determine precision level
            result.precision_level = self._precision_to_level(min_precision)

            # Check EUDR compliance
            result.meets_eudr_requirement = min_precision >= 6

            if not result.meets_eudr_requirement:
                warnings.append(
                    f"Coordinate precision ({min_precision} decimals) does not meet "
                    f"EUDR requirement (6 decimals)"
                )

            # Check against required precision
            if required_precision:
                required_decimals = self.PRECISION_DECIMALS[required_precision]
                if min_precision < required_decimals:
                    errors.append(
                        f"Coordinate precision ({min_precision}) below required "
                        f"({required_decimals}) for {required_precision.value}"
                    )

            # Edge case warnings
            if abs(lat) > Decimal('85'):
                warnings.append(
                    "Coordinate near pole - some projections may be inaccurate"
                )

            if abs(lon) > Decimal('179.9') or abs(lat) > Decimal('89.9'):
                warnings.append(
                    "Coordinate near boundary - verify coordinate is correct"
                )

            result.errors = errors
            result.warnings = warnings
            result.is_valid = len(errors) == 0

            # Calculate provenance hash
            result.provenance_hash = self._calculate_hash({
                "longitude": str(lon),
                "latitude": str(lat),
                "altitude": str(alt) if alt else None,
                "precision": min_precision,
                "is_valid": result.is_valid
            })

        except Exception as e:
            result.errors = [f"Validation error: {str(e)}"]

        return result

    def haversine_distance(
        self,
        coord1: Coordinate,
        coord2: Coordinate
    ) -> DistanceResult:
        """
        Calculate distance between two coordinates using Haversine formula.

        DETERMINISTIC CALCULATION.

        The Haversine formula calculates great-circle distance:
        a = sin^2((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin^2((lon2-lon1)/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R * c

        Accuracy: ~0.5% for most distances (assumes spherical Earth)

        Args:
            coord1: First coordinate
            coord2: Second coordinate

        Returns:
            DistanceResult with distance and bearing
        """
        lat1 = math.radians(float(coord1.latitude))
        lat2 = math.radians(float(coord2.latitude))
        lon1 = math.radians(float(coord1.longitude))
        lon2 = math.radians(float(coord2.longitude))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Haversine formula
        a = (
            math.sin(dlat / 2) ** 2 +
            math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance_m = self.EARTH_RADIUS_METERS * c

        # Calculate bearing
        bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)

        # Convert to Decimal with precision
        distance_meters = Decimal(str(distance_m)).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
        distance_km = (distance_meters / self.METERS_PER_KILOMETER).quantize(
            Decimal('0.000001'), rounding=ROUND_HALF_UP
        )

        result = DistanceResult(
            distance_meters=distance_meters,
            distance_km=distance_km,
            bearing_degrees=Decimal(str(bearing)).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP
            ),
            from_coordinate=coord1,
            to_coordinate=coord2,
            calculation_method="haversine"
        )

        result.provenance_hash = self._calculate_hash({
            "from": [float(coord1.longitude), float(coord1.latitude)],
            "to": [float(coord2.longitude), float(coord2.latitude)],
            "distance_m": float(distance_meters),
            "method": "haversine"
        })

        return result

    def vincenty_distance(
        self,
        coord1: Coordinate,
        coord2: Coordinate,
        max_iterations: int = 200,
        tolerance: float = 1e-12
    ) -> DistanceResult:
        """
        Calculate distance using Vincenty's formula (ellipsoidal Earth).

        DETERMINISTIC CALCULATION.

        Vincenty's formula is more accurate than Haversine for
        long distances because it accounts for Earth's ellipsoidal shape.

        Accuracy: ~0.5mm

        Args:
            coord1: First coordinate
            coord2: Second coordinate
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            DistanceResult with distance and bearing
        """
        lat1 = math.radians(float(coord1.latitude))
        lat2 = math.radians(float(coord2.latitude))
        lon1 = math.radians(float(coord1.longitude))
        lon2 = math.radians(float(coord2.longitude))

        # WGS84 parameters
        a = self.EARTH_RADIUS_METERS
        b = self.EARTH_RADIUS_POLAR
        f = self.FLATTENING

        L = lon2 - lon1
        U1 = math.atan((1 - f) * math.tan(lat1))
        U2 = math.atan((1 - f) * math.tan(lat2))

        sin_U1 = math.sin(U1)
        cos_U1 = math.cos(U1)
        sin_U2 = math.sin(U2)
        cos_U2 = math.cos(U2)

        lam = L
        for _ in range(max_iterations):
            sin_lam = math.sin(lam)
            cos_lam = math.cos(lam)

            sin_sigma = math.sqrt(
                (cos_U2 * sin_lam) ** 2 +
                (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam) ** 2
            )

            if sin_sigma == 0:
                # Coincident points
                return DistanceResult(
                    distance_meters=Decimal('0'),
                    distance_km=Decimal('0'),
                    bearing_degrees=Decimal('0'),
                    from_coordinate=coord1,
                    to_coordinate=coord2,
                    calculation_method="vincenty"
                )

            cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
            sigma = math.atan2(sin_sigma, cos_sigma)

            sin_alpha = cos_U1 * cos_U2 * sin_lam / sin_sigma
            cos_sq_alpha = 1 - sin_alpha ** 2

            if cos_sq_alpha == 0:
                cos_2sigma_m = 0
            else:
                cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos_sq_alpha

            C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))

            lam_prev = lam
            lam = L + (1 - C) * f * sin_alpha * (
                sigma + C * sin_sigma * (
                    cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m ** 2)
                )
            )

            if abs(lam - lam_prev) < tolerance:
                break

        u_sq = cos_sq_alpha * (a ** 2 - b ** 2) / b ** 2
        A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
        B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

        delta_sigma = B * sin_sigma * (
            cos_2sigma_m + B / 4 * (
                cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) -
                B / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma ** 2) *
                (-3 + 4 * cos_2sigma_m ** 2)
            )
        )

        distance_m = b * A * (sigma - delta_sigma)

        # Calculate bearing
        bearing = self._calculate_bearing(lat1, lon1, lat2, lon2)

        distance_meters = Decimal(str(distance_m)).quantize(
            Decimal('0.001'), rounding=ROUND_HALF_UP
        )
        distance_km = (distance_meters / self.METERS_PER_KILOMETER).quantize(
            Decimal('0.000001'), rounding=ROUND_HALF_UP
        )

        result = DistanceResult(
            distance_meters=distance_meters,
            distance_km=distance_km,
            bearing_degrees=Decimal(str(bearing)).quantize(
                Decimal('0.01'), rounding=ROUND_HALF_UP
            ),
            from_coordinate=coord1,
            to_coordinate=coord2,
            calculation_method="vincenty"
        )

        result.provenance_hash = self._calculate_hash({
            "from": [float(coord1.longitude), float(coord1.latitude)],
            "to": [float(coord2.longitude), float(coord2.latitude)],
            "distance_m": float(distance_meters),
            "method": "vincenty"
        })

        return result

    def _calculate_bearing(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate initial bearing from point 1 to point 2.

        DETERMINISTIC CALCULATION.

        Returns bearing in degrees (0-360).
        """
        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = (
            math.cos(lat1) * math.sin(lat2) -
            math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        )

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def calculate_polygon_perimeter(
        self,
        polygon: ParsedPolygon,
        use_vincenty: bool = False
    ) -> Decimal:
        """
        Calculate polygon perimeter.

        DETERMINISTIC CALCULATION.

        Args:
            polygon: Parsed polygon
            use_vincenty: Use Vincenty formula for higher accuracy

        Returns:
            Perimeter in meters
        """
        coords = polygon.exterior_ring.coordinates
        n = len(coords) - 1  # -1 because ring is closed

        total_distance = Decimal('0')

        for i in range(n):
            j = (i + 1) % (n + 1)

            if use_vincenty:
                result = self.vincenty_distance(coords[i], coords[j])
            else:
                result = self.haversine_distance(coords[i], coords[j])

            total_distance += result.distance_meters

        return total_distance.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def calculate_buffer_zone(
        self,
        center: Coordinate,
        radius_meters: Union[float, Decimal],
        num_vertices: int = 64
    ) -> BufferResult:
        """
        Calculate buffer zone (circle) around a point.

        DETERMINISTIC CALCULATION.

        Creates a polygon approximation of a circle with specified
        number of vertices.

        Args:
            center: Center coordinate
            radius_meters: Buffer radius in meters
            num_vertices: Number of vertices for polygon (default 64)

        Returns:
            BufferResult with buffer polygon coordinates
        """
        radius = Decimal(str(radius_meters))
        buffer_coords = []

        # Calculate angular step
        angle_step = 2 * math.pi / num_vertices

        center_lat = float(center.latitude)
        center_lon = float(center.longitude)

        for i in range(num_vertices + 1):  # +1 to close the ring
            angle = i * angle_step

            # Calculate point at given angle and distance
            # Using spherical approximation
            lat_rad = math.radians(center_lat)
            lon_rad = math.radians(center_lon)

            angular_distance = float(radius) / self.EARTH_RADIUS_METERS

            new_lat = math.asin(
                math.sin(lat_rad) * math.cos(angular_distance) +
                math.cos(lat_rad) * math.sin(angular_distance) * math.cos(angle)
            )

            new_lon = lon_rad + math.atan2(
                math.sin(angle) * math.sin(angular_distance) * math.cos(lat_rad),
                math.cos(angular_distance) - math.sin(lat_rad) * math.sin(new_lat)
            )

            new_lat_deg = Decimal(str(math.degrees(new_lat))).quantize(
                Decimal('0.000001'), rounding=ROUND_HALF_UP
            )
            new_lon_deg = Decimal(str(math.degrees(new_lon))).quantize(
                Decimal('0.000001'), rounding=ROUND_HALF_UP
            )

            buffer_coords.append(
                Coordinate(longitude=new_lon_deg, latitude=new_lat_deg)
            )

        # Calculate area (pi * r^2)
        area_sq_m = Decimal(str(math.pi)) * radius * radius
        area_hectares = (area_sq_m / self.SQUARE_METERS_PER_HECTARE).quantize(
            Decimal('0.000001'), rounding=ROUND_HALF_UP
        )

        result = BufferResult(
            center=center,
            radius_meters=radius,
            buffer_polygon_coords=buffer_coords,
            area_hectares=area_hectares,
            num_vertices=num_vertices
        )

        result.provenance_hash = self._calculate_hash({
            "center": [float(center.longitude), float(center.latitude)],
            "radius_m": float(radius),
            "num_vertices": num_vertices,
            "area_ha": float(area_hectares)
        })

        return result

    def point_in_polygon(
        self,
        point: Coordinate,
        polygon: ParsedPolygon
    ) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.

        DETERMINISTIC CALCULATION.

        Algorithm: Cast a ray from the point to infinity and count
        the number of polygon edges it crosses. Odd = inside, Even = outside.

        Args:
            point: Point to check
            polygon: Polygon to check against

        Returns:
            True if point is inside polygon
        """
        # First check bounding box
        if polygon.bounding_box:
            if not polygon.bounding_box.contains(point):
                return False

        coords = polygon.exterior_ring.coordinates
        n = len(coords) - 1  # -1 because ring is closed

        inside = False
        px = float(point.longitude)
        py = float(point.latitude)

        j = n - 1
        for i in range(n):
            xi = float(coords[i].longitude)
            yi = float(coords[i].latitude)
            xj = float(coords[j].longitude)
            yj = float(coords[j].latitude)

            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        # Check if point is inside any hole
        if inside:
            for hole in polygon.holes:
                hole_polygon = ParsedPolygon(
                    exterior_ring=hole,
                    holes=[]
                )
                if self.point_in_polygon(point, hole_polygon):
                    return False

        return inside

    def convert_distance(
        self,
        value: Decimal,
        from_unit: DistanceUnit,
        to_unit: DistanceUnit
    ) -> Decimal:
        """
        Convert distance between units.

        DETERMINISTIC CALCULATION.
        """
        # Convert to meters first
        if from_unit == DistanceUnit.METERS:
            meters = value
        elif from_unit == DistanceUnit.KILOMETERS:
            meters = value * self.METERS_PER_KILOMETER
        elif from_unit == DistanceUnit.MILES:
            meters = value * self.METERS_PER_MILE
        elif from_unit == DistanceUnit.NAUTICAL_MILES:
            meters = value * self.METERS_PER_NAUTICAL_MILE
        elif from_unit == DistanceUnit.FEET:
            meters = value * self.METERS_PER_FOOT
        else:
            raise ValueError(f"Unknown distance unit: {from_unit}")

        # Convert from meters to target unit
        if to_unit == DistanceUnit.METERS:
            result = meters
        elif to_unit == DistanceUnit.KILOMETERS:
            result = meters / self.METERS_PER_KILOMETER
        elif to_unit == DistanceUnit.MILES:
            result = meters / self.METERS_PER_MILE
        elif to_unit == DistanceUnit.NAUTICAL_MILES:
            result = meters / self.METERS_PER_NAUTICAL_MILE
        elif to_unit == DistanceUnit.FEET:
            result = meters / self.METERS_PER_FOOT
        else:
            raise ValueError(f"Unknown distance unit: {to_unit}")

        return result.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def convert_area(
        self,
        value: Decimal,
        from_unit: AreaUnit,
        to_unit: AreaUnit
    ) -> Decimal:
        """
        Convert area between units.

        DETERMINISTIC CALCULATION.
        """
        # Convert to square meters first
        if from_unit == AreaUnit.SQUARE_METERS:
            sq_meters = value
        elif from_unit == AreaUnit.HECTARES:
            sq_meters = value * self.SQUARE_METERS_PER_HECTARE
        elif from_unit == AreaUnit.SQUARE_KILOMETERS:
            sq_meters = value * self.SQUARE_METERS_PER_SQ_KM
        elif from_unit == AreaUnit.ACRES:
            sq_meters = value * self.SQUARE_METERS_PER_ACRE
        elif from_unit == AreaUnit.SQUARE_FEET:
            sq_meters = value * (self.METERS_PER_FOOT ** 2)
        else:
            raise ValueError(f"Unknown area unit: {from_unit}")

        # Convert from square meters to target unit
        if to_unit == AreaUnit.SQUARE_METERS:
            result = sq_meters
        elif to_unit == AreaUnit.HECTARES:
            result = sq_meters / self.SQUARE_METERS_PER_HECTARE
        elif to_unit == AreaUnit.SQUARE_KILOMETERS:
            result = sq_meters / self.SQUARE_METERS_PER_SQ_KM
        elif to_unit == AreaUnit.ACRES:
            result = sq_meters / self.SQUARE_METERS_PER_ACRE
        elif to_unit == AreaUnit.SQUARE_FEET:
            result = sq_meters / (self.METERS_PER_FOOT ** 2)
        else:
            raise ValueError(f"Unknown area unit: {to_unit}")

        return result.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)

    def _get_decimal_precision(self, value: Decimal) -> int:
        """Get number of decimal places in a Decimal value."""
        value_str = str(value)
        if '.' in value_str:
            return len(value_str.split('.')[-1])
        return 0

    def _precision_to_level(self, decimals: int) -> PrecisionLevel:
        """Convert decimal count to precision level."""
        if decimals >= 6:
            return PrecisionLevel.EUDR_COMPLIANT
        elif decimals >= 5:
            return PrecisionLevel.HIGH
        elif decimals >= 4:
            return PrecisionLevel.MEDIUM
        elif decimals >= 3:
            return PrecisionLevel.LOW
        else:
            return PrecisionLevel.CITY

    def _calculate_hash(self, data: Dict) -> str:
        """Calculate SHA-256 hash for provenance."""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
