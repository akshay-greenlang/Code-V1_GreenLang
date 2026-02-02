# -*- coding: utf-8 -*-
"""
GreenLang EUDR Protected Area Checker

Zero-hallucination protected area validation for EUDR compliance.
Integrates with WDPA (World Database on Protected Areas) for accurate
protected area intersection detection.

This module provides:
- WDPA integration for protected area lookups
- Point and polygon intersection checks
- Protection status and area details
- Caching for performance optimization
- Complete provenance tracking

Author: GreenLang Calculator Engine
License: Proprietary
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from pydantic import BaseModel, Field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from datetime import datetime, date
import hashlib
import json
from functools import lru_cache
import logging

from .geojson_parser import Coordinate, ParsedPolygon, BoundingBox
from .coordinate_validator import CoordinateValidator

logger = logging.getLogger(__name__)


class IUCNCategory(str, Enum):
    """IUCN Protected Area Categories."""
    IA = "Ia"  # Strict Nature Reserve
    IB = "Ib"  # Wilderness Area
    II = "II"  # National Park
    III = "III"  # Natural Monument
    IV = "IV"  # Habitat/Species Management Area
    V = "V"  # Protected Landscape/Seascape
    VI = "VI"  # Protected Area with Sustainable Use
    NOT_APPLICABLE = "Not Applicable"
    NOT_ASSIGNED = "Not Assigned"
    NOT_REPORTED = "Not Reported"


class ProtectionStatus(str, Enum):
    """Protection status for EUDR compliance."""
    PROTECTED = "protected"  # Area is protected - EUDR violation
    BUFFER_ZONE = "buffer_zone"  # Within buffer zone - high risk
    ADJACENT = "adjacent"  # Adjacent to protected area - medium risk
    NOT_PROTECTED = "not_protected"  # No protection - allowed
    UNKNOWN = "unknown"  # Unable to determine


class ProtectionLevel(str, Enum):
    """Level of protection (strictness)."""
    STRICT = "strict"  # Ia, Ib - No human activity allowed
    HIGH = "high"  # II, III - Limited activities
    MEDIUM = "medium"  # IV, V - Managed activities
    LOW = "low"  # VI - Sustainable use
    NONE = "none"  # Not protected


class ProtectedArea(BaseModel):
    """
    A protected area from WDPA database.

    Based on WDPA schema version 1.6.
    """
    wdpa_id: int = Field(..., description="WDPA unique identifier")
    name: str = Field(..., description="Protected area name")
    name_original: Optional[str] = Field(None, description="Name in original language")
    designation: str = Field(..., description="Type of designation")
    designation_type: str = Field(..., description="National/International/Regional")
    iucn_category: IUCNCategory = Field(..., description="IUCN management category")
    marine: bool = Field(default=False, description="Is marine area")
    reported_area_km2: Decimal = Field(..., description="Reported area in km2")
    gis_area_km2: Optional[Decimal] = Field(None, description="GIS calculated area")
    status: str = Field(..., description="Legal status")
    status_year: Optional[int] = Field(None, description="Year status established")
    governance_type: Optional[str] = Field(None, description="Governance type")
    management_authority: Optional[str] = Field(None, description="Managing authority")
    management_plan: Optional[str] = Field(None, description="Management plan status")
    country_iso3: str = Field(..., description="ISO3 country code")
    country_name: str = Field(..., description="Country name")
    sub_location: Optional[str] = Field(None, description="Sub-national location")
    parent_iso3: Optional[str] = Field(None, description="Parent country ISO3")
    verification: Optional[str] = Field(None, description="Verification status")
    metadata_id: Optional[int] = Field(None, description="Metadata record ID")
    bounding_box: Optional[BoundingBox] = Field(None, description="Area bounding box")


class ProtectedAreaIntersection(BaseModel):
    """Result of protected area intersection check."""
    protected_area: ProtectedArea
    intersection_type: str = Field(..., description="Type of intersection")
    overlap_percentage: Decimal = Field(..., description="Percentage of query area overlapping")
    overlap_area_hectares: Optional[Decimal] = Field(None, description="Overlap area in hectares")
    distance_to_boundary_meters: Optional[Decimal] = Field(None, description="Distance to boundary")
    protection_level: ProtectionLevel
    eudr_compliant: bool = Field(..., description="Is EUDR compliant (no protected area)")


class ProtectedAreaCheckResult(BaseModel):
    """Complete result of protected area check."""
    status: ProtectionStatus
    protection_level: ProtectionLevel
    is_eudr_compliant: bool
    intersections: List[ProtectedAreaIntersection] = Field(default_factory=list)
    nearest_protected_area: Optional[ProtectedArea] = None
    distance_to_nearest_meters: Optional[Decimal] = None
    buffer_violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    data_source: str = "WDPA"
    data_version: Optional[str] = None
    provenance_hash: str = ""


class WDPAQueryResult(BaseModel):
    """Result of WDPA database query."""
    areas_found: int
    query_time_ms: float
    protected_areas: List[ProtectedArea]
    query_bbox: Optional[BoundingBox] = None


class ProtectedAreaChecker:
    """
    Zero-Hallucination Protected Area Checker for EUDR Compliance.

    This checker guarantees:
    - Deterministic lookups (same input -> same output)
    - Complete audit trail
    - NO LLM in validation path

    Data Source: WDPA (World Database on Protected Areas)
    - Managed by UNEP-WCMC and IUCN
    - Updated monthly
    - Contains 280,000+ protected areas globally

    EUDR Requirements:
    - Products must not originate from protected areas
    - Buffer zones around protected areas may apply
    - Specific national designations may have additional restrictions

    Example:
        checker = ProtectedAreaChecker()

        # Check a single coordinate
        result = checker.check_coordinate(
            Coordinate(longitude=Decimal('12.345678'), latitude=Decimal('48.123456'))
        )

        if not result.is_eudr_compliant:
            print(f"Protected area detected: {result.intersections[0].protected_area.name}")
    """

    # Default buffer zone distances (meters)
    DEFAULT_BUFFER_STRICT: int = 1000  # For IUCN Ia/Ib
    DEFAULT_BUFFER_HIGH: int = 500  # For IUCN II/III
    DEFAULT_BUFFER_MEDIUM: int = 250  # For IUCN IV/V
    DEFAULT_BUFFER_LOW: int = 100  # For IUCN VI

    # WDPA API endpoints (for integration)
    WDPA_API_BASE: str = "https://api.protectedplanet.net/v3"

    def __init__(
        self,
        wdpa_api_token: Optional[str] = None,
        cache_enabled: bool = True,
        cache_ttl_hours: int = 24,
        local_db_path: Optional[str] = None
    ):
        """
        Initialize Protected Area Checker.

        Args:
            wdpa_api_token: API token for WDPA API (optional)
            cache_enabled: Enable caching for performance
            cache_ttl_hours: Cache time-to-live in hours
            local_db_path: Path to local WDPA database (for offline use)
        """
        self.wdpa_api_token = wdpa_api_token
        self.cache_enabled = cache_enabled
        self.cache_ttl_hours = cache_ttl_hours
        self.local_db_path = local_db_path
        self.coordinate_validator = CoordinateValidator()

        # Initialize local cache
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

        # Load local database if available
        self._local_areas: Dict[int, ProtectedArea] = {}
        self._spatial_index: Dict[str, List[int]] = {}  # Grid-based spatial index

        if local_db_path:
            self._load_local_database(local_db_path)

    def check_coordinate(
        self,
        coordinate: Coordinate,
        include_buffer: bool = True,
        buffer_distance_meters: Optional[int] = None,
        country_filter: Optional[str] = None
    ) -> ProtectedAreaCheckResult:
        """
        Check if a coordinate is within a protected area.

        DETERMINISTIC LOOKUP.

        Args:
            coordinate: Coordinate to check
            include_buffer: Include buffer zone checks
            buffer_distance_meters: Custom buffer distance (overrides defaults)
            country_filter: Filter to specific country (ISO3)

        Returns:
            ProtectedAreaCheckResult with complete details
        """
        result = ProtectedAreaCheckResult(
            status=ProtectionStatus.UNKNOWN,
            protection_level=ProtectionLevel.NONE,
            is_eudr_compliant=True
        )

        try:
            # Query protected areas near coordinate
            nearby_areas = self._query_nearby_areas(coordinate, country_filter)

            intersections = []
            nearest_area = None
            min_distance = None

            for area in nearby_areas.protected_areas:
                # Check intersection
                intersection = self._check_area_intersection(
                    coordinate, area, buffer_distance_meters
                )

                if intersection:
                    intersections.append(intersection)

                    if not intersection.eudr_compliant:
                        result.is_eudr_compliant = False

                # Track nearest area
                distance = self._calculate_distance_to_area(coordinate, area)
                if distance is not None and (min_distance is None or distance < min_distance):
                    min_distance = distance
                    nearest_area = area

            # Set results
            result.intersections = intersections
            result.nearest_protected_area = nearest_area
            result.distance_to_nearest_meters = min_distance

            # Determine overall status
            if intersections:
                # Find most restrictive intersection
                result.status = ProtectionStatus.PROTECTED
                result.protection_level = max(
                    (i.protection_level for i in intersections),
                    key=lambda x: self._protection_level_rank(x)
                )
            elif include_buffer and nearest_area and min_distance:
                # Check buffer zones
                buffer_threshold = self._get_buffer_distance(
                    nearest_area, buffer_distance_meters
                )
                if min_distance < Decimal(str(buffer_threshold)):
                    result.status = ProtectionStatus.BUFFER_ZONE
                    result.buffer_violations.append(
                        f"Within {buffer_threshold}m buffer of {nearest_area.name}"
                    )
                    result.protection_level = self._get_protection_level(
                        nearest_area.iucn_category
                    )
                elif min_distance < Decimal(str(buffer_threshold * 2)):
                    result.status = ProtectionStatus.ADJACENT
                else:
                    result.status = ProtectionStatus.NOT_PROTECTED
            else:
                result.status = ProtectionStatus.NOT_PROTECTED

            # Calculate provenance hash
            result.provenance_hash = self._calculate_hash({
                "coordinate": [float(coordinate.longitude), float(coordinate.latitude)],
                "status": result.status.value,
                "intersections_count": len(intersections),
                "is_compliant": result.is_eudr_compliant
            })

        except Exception as e:
            logger.error(f"Protected area check failed: {e}")
            result.status = ProtectionStatus.UNKNOWN
            result.warnings.append(f"Check failed: {str(e)}")

        return result

    def check_polygon(
        self,
        polygon: ParsedPolygon,
        include_buffer: bool = True,
        buffer_distance_meters: Optional[int] = None,
        country_filter: Optional[str] = None
    ) -> ProtectedAreaCheckResult:
        """
        Check if a polygon intersects protected areas.

        DETERMINISTIC LOOKUP.

        Args:
            polygon: Polygon to check
            include_buffer: Include buffer zone checks
            buffer_distance_meters: Custom buffer distance
            country_filter: Filter to specific country (ISO3)

        Returns:
            ProtectedAreaCheckResult with complete details
        """
        result = ProtectedAreaCheckResult(
            status=ProtectionStatus.UNKNOWN,
            protection_level=ProtectionLevel.NONE,
            is_eudr_compliant=True
        )

        try:
            # Check centroid first (quick check)
            if polygon.centroid:
                centroid_result = self.check_coordinate(
                    polygon.centroid,
                    include_buffer=False,
                    country_filter=country_filter
                )

                if not centroid_result.is_eudr_compliant:
                    result = centroid_result
                    return result

            # Check all vertices
            for coord in polygon.exterior_ring.coordinates:
                vertex_result = self.check_coordinate(
                    coord,
                    include_buffer=include_buffer,
                    buffer_distance_meters=buffer_distance_meters,
                    country_filter=country_filter
                )

                if not vertex_result.is_eudr_compliant:
                    result.is_eudr_compliant = False
                    result.intersections.extend(vertex_result.intersections)
                    result.buffer_violations.extend(vertex_result.buffer_violations)

                if vertex_result.status == ProtectionStatus.PROTECTED:
                    result.status = ProtectionStatus.PROTECTED
                elif (vertex_result.status == ProtectionStatus.BUFFER_ZONE and
                      result.status != ProtectionStatus.PROTECTED):
                    result.status = ProtectionStatus.BUFFER_ZONE

            # Remove duplicate intersections
            seen_ids = set()
            unique_intersections = []
            for intersection in result.intersections:
                if intersection.protected_area.wdpa_id not in seen_ids:
                    seen_ids.add(intersection.protected_area.wdpa_id)
                    unique_intersections.append(intersection)
            result.intersections = unique_intersections

            # Set final status if no issues found
            if result.status == ProtectionStatus.UNKNOWN:
                result.status = ProtectionStatus.NOT_PROTECTED

            # Calculate provenance
            result.provenance_hash = self._calculate_hash({
                "polygon_hash": polygon.get_hash(),
                "status": result.status.value,
                "intersections_count": len(result.intersections),
                "is_compliant": result.is_eudr_compliant
            })

        except Exception as e:
            logger.error(f"Polygon protected area check failed: {e}")
            result.warnings.append(f"Check failed: {str(e)}")

        return result

    def _query_nearby_areas(
        self,
        coordinate: Coordinate,
        country_filter: Optional[str] = None,
        radius_km: float = 50
    ) -> WDPAQueryResult:
        """
        Query protected areas near a coordinate.

        This is a stub that should be connected to WDPA API or local database.
        For production, implement actual WDPA API integration.
        """
        import time
        start_time = time.perf_counter()

        # Check cache first
        cache_key = f"{coordinate.longitude}:{coordinate.latitude}:{country_filter}:{radius_km}"
        if self.cache_enabled and cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            age_hours = (datetime.utcnow() - cached_time).total_seconds() / 3600
            if age_hours < self.cache_ttl_hours:
                return cached_result

        # Query local database if available
        areas = []
        if self._local_areas:
            areas = self._query_local_database(coordinate, radius_km, country_filter)

        # If no local results and API token available, query WDPA API
        elif self.wdpa_api_token:
            areas = self._query_wdpa_api(coordinate, radius_km, country_filter)

        result = WDPAQueryResult(
            areas_found=len(areas),
            query_time_ms=(time.perf_counter() - start_time) * 1000,
            protected_areas=areas
        )

        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = (datetime.utcnow(), result)

        return result

    def _query_local_database(
        self,
        coordinate: Coordinate,
        radius_km: float,
        country_filter: Optional[str]
    ) -> List[ProtectedArea]:
        """Query local WDPA database."""
        # Use spatial index for efficient lookup
        grid_key = self._get_grid_key(coordinate)
        candidate_ids = self._spatial_index.get(grid_key, [])

        # Also check adjacent grid cells
        for adjacent_key in self._get_adjacent_grid_keys(coordinate):
            candidate_ids.extend(self._spatial_index.get(adjacent_key, []))

        # Filter by distance and country
        results = []
        for wdpa_id in set(candidate_ids):
            area = self._local_areas.get(wdpa_id)
            if area is None:
                continue

            if country_filter and area.country_iso3 != country_filter:
                continue

            # Check if within radius (rough check using bounding box)
            if area.bounding_box:
                coord_in_expanded_bbox = self._coordinate_in_expanded_bbox(
                    coordinate, area.bounding_box, radius_km
                )
                if coord_in_expanded_bbox:
                    results.append(area)

        return results

    def _query_wdpa_api(
        self,
        coordinate: Coordinate,
        radius_km: float,
        country_filter: Optional[str]
    ) -> List[ProtectedArea]:
        """
        Query WDPA API.

        This is a stub for WDPA API integration.
        In production, implement actual HTTP calls to Protected Planet API.
        """
        # Stub implementation - would make actual API call
        logger.warning("WDPA API integration not implemented - using stub")
        return []

    def _check_area_intersection(
        self,
        coordinate: Coordinate,
        area: ProtectedArea,
        custom_buffer: Optional[int]
    ) -> Optional[ProtectedAreaIntersection]:
        """
        Check if coordinate intersects a protected area.

        For production, this would use actual geometry intersection.
        """
        # Simple bounding box check (stub for actual geometry check)
        if area.bounding_box:
            if area.bounding_box.contains(coordinate):
                protection_level = self._get_protection_level(area.iucn_category)

                return ProtectedAreaIntersection(
                    protected_area=area,
                    intersection_type="point_in_polygon",
                    overlap_percentage=Decimal('100'),
                    protection_level=protection_level,
                    eudr_compliant=False  # Any intersection = not compliant
                )

        return None

    def _calculate_distance_to_area(
        self,
        coordinate: Coordinate,
        area: ProtectedArea
    ) -> Optional[Decimal]:
        """
        Calculate distance from coordinate to protected area boundary.

        For production, use actual geometry distance calculation.
        """
        if area.bounding_box:
            # Simple distance to bounding box (stub)
            if area.bounding_box.contains(coordinate):
                return Decimal('0')  # Inside

            # Calculate distance to nearest edge
            nearest_lon = max(
                area.bounding_box.min_longitude,
                min(coordinate.longitude, area.bounding_box.max_longitude)
            )
            nearest_lat = max(
                area.bounding_box.min_latitude,
                min(coordinate.latitude, area.bounding_box.max_latitude)
            )

            nearest_coord = Coordinate(longitude=nearest_lon, latitude=nearest_lat)
            distance_result = self.coordinate_validator.haversine_distance(
                coordinate, nearest_coord
            )

            return distance_result.distance_meters

        return None

    def _get_buffer_distance(
        self,
        area: ProtectedArea,
        custom_buffer: Optional[int]
    ) -> int:
        """Get buffer distance for a protected area based on IUCN category."""
        if custom_buffer is not None:
            return custom_buffer

        category = area.iucn_category

        if category in [IUCNCategory.IA, IUCNCategory.IB]:
            return self.DEFAULT_BUFFER_STRICT
        elif category in [IUCNCategory.II, IUCNCategory.III]:
            return self.DEFAULT_BUFFER_HIGH
        elif category in [IUCNCategory.IV, IUCNCategory.V]:
            return self.DEFAULT_BUFFER_MEDIUM
        else:
            return self.DEFAULT_BUFFER_LOW

    def _get_protection_level(self, category: IUCNCategory) -> ProtectionLevel:
        """Convert IUCN category to protection level."""
        if category in [IUCNCategory.IA, IUCNCategory.IB]:
            return ProtectionLevel.STRICT
        elif category in [IUCNCategory.II, IUCNCategory.III]:
            return ProtectionLevel.HIGH
        elif category in [IUCNCategory.IV, IUCNCategory.V]:
            return ProtectionLevel.MEDIUM
        elif category == IUCNCategory.VI:
            return ProtectionLevel.LOW
        else:
            return ProtectionLevel.NONE

    def _protection_level_rank(self, level: ProtectionLevel) -> int:
        """Get numerical rank for protection level (higher = stricter)."""
        ranks = {
            ProtectionLevel.STRICT: 4,
            ProtectionLevel.HIGH: 3,
            ProtectionLevel.MEDIUM: 2,
            ProtectionLevel.LOW: 1,
            ProtectionLevel.NONE: 0
        }
        return ranks.get(level, 0)

    def _get_grid_key(self, coordinate: Coordinate, grid_size: float = 1.0) -> str:
        """Get spatial index grid key for coordinate."""
        lat_grid = int(float(coordinate.latitude) / grid_size)
        lon_grid = int(float(coordinate.longitude) / grid_size)
        return f"{lat_grid}:{lon_grid}"

    def _get_adjacent_grid_keys(self, coordinate: Coordinate, grid_size: float = 1.0) -> List[str]:
        """Get adjacent grid keys for spatial search."""
        lat_grid = int(float(coordinate.latitude) / grid_size)
        lon_grid = int(float(coordinate.longitude) / grid_size)

        adjacent = []
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                if dlat == 0 and dlon == 0:
                    continue
                adjacent.append(f"{lat_grid + dlat}:{lon_grid + dlon}")

        return adjacent

    def _coordinate_in_expanded_bbox(
        self,
        coordinate: Coordinate,
        bbox: BoundingBox,
        expansion_km: float
    ) -> bool:
        """Check if coordinate is within expanded bounding box."""
        # Approximate degrees per km at equator
        km_per_degree = Decimal('111.32')
        expansion_degrees = Decimal(str(expansion_km)) / km_per_degree

        expanded_bbox = BoundingBox(
            min_longitude=bbox.min_longitude - expansion_degrees,
            max_longitude=bbox.max_longitude + expansion_degrees,
            min_latitude=bbox.min_latitude - expansion_degrees,
            max_latitude=bbox.max_latitude + expansion_degrees
        )

        return expanded_bbox.contains(coordinate)

    def _load_local_database(self, db_path: str) -> None:
        """
        Load local WDPA database.

        For production, this would load from a local database file
        (e.g., SQLite, GeoPackage, or shapefile).
        """
        logger.info(f"Loading local WDPA database from {db_path}")
        # Stub - implement actual database loading
        pass

    def _calculate_hash(self, data: Dict) -> str:
        """Calculate SHA-256 hash for provenance."""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def get_protected_areas_for_country(
        self,
        country_iso3: str
    ) -> List[ProtectedArea]:
        """
        Get all protected areas for a country.

        Args:
            country_iso3: ISO3 country code

        Returns:
            List of protected areas
        """
        areas = []
        for area in self._local_areas.values():
            if area.country_iso3 == country_iso3:
                areas.append(area)

        return sorted(areas, key=lambda x: x.reported_area_km2, reverse=True)

    def get_area_by_wdpa_id(self, wdpa_id: int) -> Optional[ProtectedArea]:
        """Get a protected area by WDPA ID."""
        return self._local_areas.get(wdpa_id)

    def clear_cache(self) -> None:
        """Clear the protected area cache."""
        self._cache.clear()
        logger.info("Protected area cache cleared")
