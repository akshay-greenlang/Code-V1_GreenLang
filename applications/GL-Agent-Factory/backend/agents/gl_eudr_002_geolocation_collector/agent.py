"""
GL-EUDR-002: Geolocation Collector Agent

This module implements the Geolocation Collector Agent for EUDR compliance,
providing collection, validation, and management of production plot coordinates.

The agent supports:
- Multi-channel data collection (GPS, web form, bulk upload, API)
- Deterministic coordinate validation (zero hallucination)
- Plot geometry management (Point and Polygon)
- Data enrichment (geocoding, admin regions, biomes)
- PostGIS integration for spatial operations

Regulatory Reference:
    EU Regulation 2023/1115 (EUDR) - Article 9
    Geolocation Requirements: WGS-84, 6 decimal precision

Example:
    >>> agent = GeolocationCollectorAgent()
    >>> result = agent.run(GeolocationInput(
    ...     operation=OperationType.VALIDATE_COORDINATES,
    ...     coordinates=PointCoordinates(latitude=-4.123456, longitude=102.654321),
    ...     country_code="ID",
    ...     commodity=CommodityType.PALM_OIL
    ... ))
    >>> print(f"Valid: {result.valid}, Errors: {len(result.errors)}")
"""

import hashlib
import json
import logging
import math
import re
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

# Import spatial validation services
from .spatial import (
    SpatialValidationService,
    CountryBoundaryService,
    WaterBodyService,
    ProtectedAreaService,
    UrbanAreaService,
    SpatialQueryResult,
    normalize_longitude,
    crosses_dateline,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CommodityType(str, Enum):
    """EUDR regulated commodity categories."""
    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    PALM_OIL = "PALM_OIL"
    RUBBER = "RUBBER"
    SOY = "SOY"
    WOOD = "WOOD"


class GeometryType(str, Enum):
    """Supported geometry types."""
    POINT = "POINT"
    POLYGON = "POLYGON"


class ValidationStatus(str, Enum):
    """Plot validation status."""
    VALID = "VALID"
    INVALID = "INVALID"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PENDING = "PENDING"


class ValidationSeverity(str, Enum):
    """Validation issue severity."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class CollectionMethod(str, Enum):
    """Data collection method."""
    GPS = "GPS"
    MANUAL = "MANUAL"
    UPLOAD = "UPLOAD"
    API = "API"


class OperationType(str, Enum):
    """Agent operation types."""
    VALIDATE_COORDINATES = "VALIDATE_COORDINATES"
    SUBMIT_PLOT = "SUBMIT_PLOT"
    REVALIDATE_PLOT = "REVALIDATE_PLOT"
    BULK_UPLOAD = "BULK_UPLOAD"
    GET_PLOT = "GET_PLOT"
    LIST_PLOTS = "LIST_PLOTS"
    ENRICH_PLOT = "ENRICH_PLOT"
    EXPORT_PLOTS = "EXPORT_PLOTS"


class BulkUploadFormat(str, Enum):
    """Supported bulk upload formats."""
    CSV = "csv"
    GEOJSON = "geojson"
    KML = "kml"
    SHAPEFILE = "shapefile"


class BulkJobStatus(str, Enum):
    """Bulk upload job status."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# =============================================================================
# VALIDATION ERROR CODES
# =============================================================================

class ErrorCode(str, Enum):
    """Validation error codes per PRD specification."""
    # Precision errors
    INSUFFICIENT_LAT_PRECISION = "INSUFFICIENT_LAT_PRECISION"
    INSUFFICIENT_LON_PRECISION = "INSUFFICIENT_LON_PRECISION"

    # Range errors
    INVALID_LAT_RANGE = "INVALID_LAT_RANGE"
    INVALID_LON_RANGE = "INVALID_LON_RANGE"

    # Geographic errors
    NOT_IN_COUNTRY = "NOT_IN_COUNTRY"
    IN_WATER_BODY = "IN_WATER_BODY"

    # Polygon errors
    INVALID_POLYGON = "INVALID_POLYGON"
    SELF_INTERSECTING = "SELF_INTERSECTING"
    AREA_TOO_SMALL = "AREA_TOO_SMALL"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"

    # Warning codes
    IN_PROTECTED_AREA = "IN_PROTECTED_AREA"
    IN_URBAN_AREA = "IN_URBAN_AREA"
    AREA_MISMATCH = "AREA_MISMATCH"
    NEEDS_POLYGON = "NEEDS_POLYGON"
    POOR_GPS_ACCURACY = "POOR_GPS_ACCURACY"

    # Overlap detection codes (FR-024/FR-025)
    PLOT_OVERLAP = "PLOT_OVERLAP"
    DUPLICATE_COORDINATES = "DUPLICATE_COORDINATES"
    COUNTRY_MISMATCH = "COUNTRY_MISMATCH"


# Error code severity mapping
ERROR_SEVERITY = {
    ErrorCode.INSUFFICIENT_LAT_PRECISION: ValidationSeverity.ERROR,
    ErrorCode.INSUFFICIENT_LON_PRECISION: ValidationSeverity.ERROR,
    ErrorCode.INVALID_LAT_RANGE: ValidationSeverity.ERROR,
    ErrorCode.INVALID_LON_RANGE: ValidationSeverity.ERROR,
    ErrorCode.NOT_IN_COUNTRY: ValidationSeverity.ERROR,
    ErrorCode.IN_WATER_BODY: ValidationSeverity.ERROR,
    ErrorCode.INVALID_POLYGON: ValidationSeverity.ERROR,
    ErrorCode.SELF_INTERSECTING: ValidationSeverity.ERROR,
    ErrorCode.AREA_TOO_SMALL: ValidationSeverity.ERROR,
    ErrorCode.INSUFFICIENT_POINTS: ValidationSeverity.ERROR,
    ErrorCode.IN_PROTECTED_AREA: ValidationSeverity.WARNING,
    ErrorCode.IN_URBAN_AREA: ValidationSeverity.WARNING,
    ErrorCode.AREA_MISMATCH: ValidationSeverity.WARNING,
    ErrorCode.NEEDS_POLYGON: ValidationSeverity.WARNING,
    ErrorCode.POOR_GPS_ACCURACY: ValidationSeverity.WARNING,
    # Overlap detection (FR-024/FR-025)
    ErrorCode.PLOT_OVERLAP: ValidationSeverity.WARNING,
    ErrorCode.DUPLICATE_COORDINATES: ValidationSeverity.ERROR,
    ErrorCode.COUNTRY_MISMATCH: ValidationSeverity.ERROR,
}


# =============================================================================
# COORDINATE MODELS
# =============================================================================

class PointCoordinates(BaseModel):
    """Point coordinates (latitude, longitude)."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    @validator('latitude', 'longitude')
    def validate_precision(cls, v):
        """Ensure coordinate is a valid float."""
        return float(v)

    def to_tuple(self) -> Tuple[float, float]:
        """Return as (lat, lon) tuple."""
        return (self.latitude, self.longitude)

    def to_geojson(self) -> Dict[str, Any]:
        """Return as GeoJSON Point."""
        return {
            "type": "Point",
            "coordinates": [self.longitude, self.latitude]  # GeoJSON is [lon, lat]
        }


class PolygonCoordinates(BaseModel):
    """Polygon coordinates (list of points forming a closed ring)."""
    coordinates: List[List[float]] = Field(
        ...,
        min_items=4,
        description="List of [longitude, latitude] pairs. First and last must match."
    )

    @validator('coordinates')
    def validate_polygon(cls, v):
        """Validate polygon structure."""
        if len(v) < 4:
            raise ValueError("Polygon requires at least 4 points (including closing point)")

        # Check each coordinate pair
        for i, coord in enumerate(v):
            if len(coord) != 2:
                raise ValueError(f"Coordinate {i} must have exactly 2 values [lon, lat]")
            lon, lat = coord
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} at position {i} out of range")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} at position {i} out of range")

        # Check closure (first == last)
        if v[0] != v[-1]:
            raise ValueError("Polygon must be closed (first point must equal last point)")

        return v

    def to_geojson(self) -> Dict[str, Any]:
        """Return as GeoJSON Polygon."""
        return {
            "type": "Polygon",
            "coordinates": [self.coordinates]  # GeoJSON polygon is array of rings
        }

    def get_centroid(self) -> Tuple[float, float]:
        """Calculate polygon centroid."""
        n = len(self.coordinates) - 1  # Exclude closing point
        sum_lon = sum(c[0] for c in self.coordinates[:n])
        sum_lat = sum(c[1] for c in self.coordinates[:n])
        return (sum_lat / n, sum_lon / n)


# =============================================================================
# VALIDATION MODELS
# =============================================================================

class ValidationError(BaseModel):
    """A single validation error or warning."""
    code: ErrorCode
    message: str
    severity: ValidationSeverity
    coordinate: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of coordinate/plot validation."""
    valid: bool
    status: ValidationStatus
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


class StageResult(BaseModel):
    """Result from a validation stage."""
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_fatal: bool = False


# =============================================================================
# PLOT MODELS
# =============================================================================

class PlotSubmission(BaseModel):
    """Input for plot submission."""
    supplier_id: UUID
    external_id: Optional[str] = None
    coordinates: Union[PointCoordinates, PolygonCoordinates]
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")
    commodity: CommodityType
    declared_area_hectares: Optional[float] = Field(None, gt=0)
    collection_method: CollectionMethod = CollectionMethod.API
    collection_device: Optional[str] = None
    collection_accuracy_m: Optional[float] = Field(None, ge=0)
    collection_date: Optional[date] = None
    collected_by: Optional[str] = None
    crop_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class Plot(BaseModel):
    """A production plot with validated geolocation."""
    plot_id: UUID = Field(default_factory=uuid.uuid4)
    supplier_id: UUID
    external_id: Optional[str] = None

    # Geometry
    geometry_type: GeometryType
    coordinates: Union[PointCoordinates, PolygonCoordinates]
    centroid: Optional[PointCoordinates] = None
    bounding_box: Optional[List[float]] = None  # [min_lon, min_lat, max_lon, max_lat]
    area_hectares: Optional[Decimal] = Field(None, gt=0)
    perimeter_km: Optional[Decimal] = None

    # Location
    country_code: str = Field(..., pattern=r"^[A-Z]{2}$")
    admin_level_1: Optional[str] = None  # State/Province
    admin_level_2: Optional[str] = None  # District/County
    admin_level_3: Optional[str] = None  # Municipality
    nearest_place: Optional[str] = None

    # Commodity
    commodity: CommodityType
    crop_type: Optional[str] = None

    # Validation
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_errors: List[ValidationError] = Field(default_factory=list)
    validation_warnings: List[ValidationError] = Field(default_factory=list)
    precision_lat: Optional[int] = None
    precision_lon: Optional[int] = None
    last_validated_at: Optional[datetime] = None

    # Collection metadata
    collection_method: CollectionMethod
    collection_device: Optional[str] = None
    collection_accuracy_m: Optional[float] = None
    collection_date: Optional[date] = None
    collected_by: Optional[str] = None

    # Enrichment
    biome: Optional[str] = None
    ecosystem: Optional[str] = None
    elevation_m: Optional[int] = None
    slope_degrees: Optional[float] = None
    in_protected_area: bool = False
    protected_area_name: Optional[str] = None
    in_urban_area: bool = False

    # Version tracking
    version: int = 1
    previous_version_id: Optional[UUID] = None

    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    organization_id: Optional[UUID] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class PlotValidationHistory(BaseModel):
    """Historical validation record for a plot."""
    validation_id: UUID = Field(default_factory=uuid.uuid4)
    plot_id: UUID
    validation_date: datetime = Field(default_factory=datetime.utcnow)
    status: ValidationStatus
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[ValidationError] = Field(default_factory=list)
    validated_by: Optional[str] = None
    validation_method: str = "AUTO"  # AUTO, MANUAL, REVIEW

    class Config:
        use_enum_values = True


# =============================================================================
# BULK UPLOAD MODELS
# =============================================================================

class BulkUploadJob(BaseModel):
    """Bulk upload job tracking."""
    job_id: UUID = Field(default_factory=uuid.uuid4)
    supplier_id: UUID
    file_format: BulkUploadFormat
    file_name: str
    file_size_bytes: int
    status: BulkJobStatus = BulkJobStatus.QUEUED

    # Progress
    total_count: int = 0
    processed_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    warning_count: int = 0

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    report_url: Optional[str] = None
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    organization_id: Optional[UUID] = None

    class Config:
        use_enum_values = True


class BulkUploadResult(BaseModel):
    """Result of bulk upload processing."""
    job_id: UUID
    status: BulkJobStatus
    total_count: int
    processed_count: int
    valid_count: int
    invalid_count: int
    warning_count: int
    processing_time_ms: int
    plots: List[Plot] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class GeolocationInput(BaseModel):
    """Input for Geolocation Collector Agent."""
    operation: OperationType

    # For validation/submission
    coordinates: Optional[Union[PointCoordinates, PolygonCoordinates]] = None
    country_code: Optional[str] = Field(None, pattern=r"^[A-Z]{2}$")
    commodity: Optional[CommodityType] = None
    declared_area_hectares: Optional[float] = None
    supplier_id: Optional[UUID] = None

    # For plot operations
    plot_id: Optional[UUID] = None
    plot_ids: Optional[List[UUID]] = None

    # For bulk upload
    file_path: Optional[str] = None
    file_format: Optional[BulkUploadFormat] = None
    job_id: Optional[UUID] = None

    # Collection metadata
    collection_method: CollectionMethod = CollectionMethod.API
    collection_accuracy_m: Optional[float] = None

    # Filters
    validation_status: Optional[ValidationStatus] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)

    class Config:
        use_enum_values = True


class GeolocationOutput(BaseModel):
    """Output from Geolocation Collector Agent."""
    success: bool
    operation: OperationType

    # Validation results
    validation_result: Optional[ValidationResult] = None

    # Plot results
    plot: Optional[Plot] = None
    plots: List[Plot] = Field(default_factory=list)
    total_count: int = 0

    # Bulk upload results
    bulk_job: Optional[BulkUploadJob] = None
    bulk_result: Optional[BulkUploadResult] = None

    # Metadata
    processing_time_ms: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


# =============================================================================
# GEOLOCATION VALIDATOR
# =============================================================================

class GeolocationValidator:
    """
    Deterministic coordinate validation engine.
    Zero hallucination - all rules are explicit and auditable.
    """

    # Minimum required decimal precision
    MIN_PRECISION = 6

    # Minimum plot area in hectares
    MIN_AREA_HECTARES = 0.01

    # Area threshold for polygon requirement
    POLYGON_THRESHOLD_HECTARES = 4.0

    # GPS accuracy warning threshold (meters)
    GPS_ACCURACY_WARNING_THRESHOLD = 10.0

    # Maximum polygon vertices
    MAX_POLYGON_VERTICES = 10000

    def __init__(
        self,
        spatial_service: Optional[SpatialValidationService] = None,
        gadm_path: Optional[str] = None,
        water_path: Optional[str] = None,
        wdpa_path: Optional[str] = None,
        urban_path: Optional[str] = None
    ):
        """
        Initialize validator with spatial services.

        Args:
            spatial_service: Pre-configured SpatialValidationService
            gadm_path: Path to GADM country boundary data
            water_path: Path to OSM water body data
            wdpa_path: Path to WDPA protected area data
            urban_path: Path to OSM urban area data
        """
        if spatial_service:
            self.spatial_service = spatial_service
        else:
            self.spatial_service = SpatialValidationService()
            # Initialize with provided paths if any
            if any([gadm_path, water_path, wdpa_path, urban_path]):
                self.spatial_service.initialize(
                    gadm_path=gadm_path,
                    water_path=water_path,
                    wdpa_path=wdpa_path,
                    urban_path=urban_path
                )

        # Keep legacy attributes for backwards compatibility
        self.country_boundaries = self.spatial_service.country_service
        self.water_bodies = self.spatial_service.water_service
        self.protected_areas = self.spatial_service.protected_service
        self.urban_areas = self.spatial_service.urban_service

    def validate(
        self,
        coordinates: Union[PointCoordinates, PolygonCoordinates],
        country_code: str,
        commodity: CommodityType,
        declared_area: Optional[float] = None,
        gps_accuracy_m: Optional[float] = None
    ) -> ValidationResult:
        """
        Main validation entry point.

        Args:
            coordinates: Point or Polygon coordinates
            country_code: ISO 2-letter country code
            commodity: EUDR commodity type
            declared_area: Supplier-declared area in hectares
            gps_accuracy_m: GPS device accuracy in meters

        Returns:
            ValidationResult with status, errors, and warnings
        """
        all_errors = []
        all_warnings = []
        all_metadata = {}

        # Stage 1: Format and range validation
        format_result = self._validate_format_and_range(coordinates)
        all_errors.extend(format_result.errors)
        all_metadata.update(format_result.metadata)

        if format_result.is_fatal:
            return ValidationResult(
                valid=False,
                status=ValidationStatus.INVALID,
                errors=all_errors,
                warnings=all_warnings,
                metadata=all_metadata
            )

        # Stage 2: Precision validation
        precision_result = self._validate_precision(coordinates)
        all_errors.extend(precision_result.errors)
        all_metadata.update(precision_result.metadata)

        # Stage 3: Geographic validation (country, water)
        geo_result = self._validate_geography(coordinates, country_code)
        all_errors.extend(geo_result.errors)
        all_warnings.extend(geo_result.warnings)
        all_metadata.update(geo_result.metadata)

        # Stage 4: Geometry-specific validation
        if isinstance(coordinates, PolygonCoordinates):
            poly_result = self._validate_polygon(coordinates, declared_area)
            all_errors.extend(poly_result.errors)
            all_warnings.extend(poly_result.warnings)
            all_metadata.update(poly_result.metadata)
        else:
            # Check if point should be polygon
            if declared_area and declared_area >= self.POLYGON_THRESHOLD_HECTARES:
                all_warnings.append(ValidationError(
                    code=ErrorCode.NEEDS_POLYGON,
                    message=f"Plot >= {self.POLYGON_THRESHOLD_HECTARES} ha should have polygon geometry",
                    severity=ValidationSeverity.WARNING,
                    metadata={"declared_area": declared_area}
                ))

        # Stage 5: GPS accuracy check
        if gps_accuracy_m and gps_accuracy_m > self.GPS_ACCURACY_WARNING_THRESHOLD:
            all_warnings.append(ValidationError(
                code=ErrorCode.POOR_GPS_ACCURACY,
                message=f"GPS accuracy {gps_accuracy_m}m exceeds {self.GPS_ACCURACY_WARNING_THRESHOLD}m threshold",
                severity=ValidationSeverity.WARNING,
                metadata={"gps_accuracy_m": gps_accuracy_m}
            ))

        # Determine final status
        has_errors = len(all_errors) > 0
        has_warnings = len(all_warnings) > 0

        if has_errors:
            status = ValidationStatus.INVALID
        elif has_warnings:
            status = ValidationStatus.NEEDS_REVIEW
        else:
            status = ValidationStatus.VALID

        return ValidationResult(
            valid=not has_errors,
            status=status,
            errors=all_errors,
            warnings=all_warnings,
            metadata=all_metadata
        )

    def _validate_format_and_range(
        self,
        coordinates: Union[PointCoordinates, PolygonCoordinates]
    ) -> StageResult:
        """Validate coordinate format and ranges."""
        errors = []
        metadata = {"geometry_type": "POINT" if isinstance(coordinates, PointCoordinates) else "POLYGON"}

        if isinstance(coordinates, PointCoordinates):
            lat, lon = coordinates.latitude, coordinates.longitude

            if not (-90 <= lat <= 90):
                errors.append(ValidationError(
                    code=ErrorCode.INVALID_LAT_RANGE,
                    message=f"Latitude {lat} outside valid range [-90, 90]",
                    severity=ValidationSeverity.ERROR,
                    coordinate=(lat, lon)
                ))

            if not (-180 <= lon <= 180):
                errors.append(ValidationError(
                    code=ErrorCode.INVALID_LON_RANGE,
                    message=f"Longitude {lon} outside valid range [-180, 180]",
                    severity=ValidationSeverity.ERROR,
                    coordinate=(lat, lon)
                ))
        else:
            # Polygon coordinates already validated by Pydantic
            metadata["point_count"] = len(coordinates.coordinates)

            if len(coordinates.coordinates) > self.MAX_POLYGON_VERTICES:
                errors.append(ValidationError(
                    code=ErrorCode.INVALID_POLYGON,
                    message=f"Polygon has {len(coordinates.coordinates)} vertices, max is {self.MAX_POLYGON_VERTICES}",
                    severity=ValidationSeverity.ERROR
                ))

        return StageResult(
            errors=errors,
            metadata=metadata,
            is_fatal=len(errors) > 0
        )

    def _validate_precision(
        self,
        coordinates: Union[PointCoordinates, PolygonCoordinates]
    ) -> StageResult:
        """
        Validate coordinate precision (>= 6 decimal places).
        EUDR Article 9 requires minimum 6 decimal places (~0.11m precision).
        """
        errors = []
        metadata = {}

        if isinstance(coordinates, PointCoordinates):
            points = [(coordinates.latitude, coordinates.longitude)]
        else:
            points = [(c[1], c[0]) for c in coordinates.coordinates]  # [lon, lat] -> (lat, lon)

        min_lat_precision = 99
        min_lon_precision = 99

        for lat, lon in points:
            lat_precision = self._count_decimal_places(lat)
            lon_precision = self._count_decimal_places(lon)

            min_lat_precision = min(min_lat_precision, lat_precision)
            min_lon_precision = min(min_lon_precision, lon_precision)

            if lat_precision < self.MIN_PRECISION:
                errors.append(ValidationError(
                    code=ErrorCode.INSUFFICIENT_LAT_PRECISION,
                    message=f"Latitude has {lat_precision} decimal places, requires {self.MIN_PRECISION}",
                    severity=ValidationSeverity.ERROR,
                    coordinate=(lat, lon),
                    metadata={"precision": lat_precision, "required": self.MIN_PRECISION}
                ))
                break  # Only report first error for polygons

            if lon_precision < self.MIN_PRECISION:
                errors.append(ValidationError(
                    code=ErrorCode.INSUFFICIENT_LON_PRECISION,
                    message=f"Longitude has {lon_precision} decimal places, requires {self.MIN_PRECISION}",
                    severity=ValidationSeverity.ERROR,
                    coordinate=(lat, lon),
                    metadata={"precision": lon_precision, "required": self.MIN_PRECISION}
                ))
                break

        metadata["min_lat_precision"] = min_lat_precision
        metadata["min_lon_precision"] = min_lon_precision

        return StageResult(errors=errors, metadata=metadata)

    def _validate_geography(
        self,
        coordinates: Union[PointCoordinates, PolygonCoordinates],
        country_code: str
    ) -> StageResult:
        """Validate geographic placement (country, water bodies)."""
        errors = []
        warnings = []
        metadata = {"country_code": country_code}

        # Get centroid for checks
        if isinstance(coordinates, PointCoordinates):
            lat, lon = coordinates.latitude, coordinates.longitude
            centroid = (lat, lon)
        else:
            centroid = coordinates.get_centroid()
            lat, lon = centroid

        # Normalize longitude for dateline handling
        lon = normalize_longitude(lon)
        metadata["centroid"] = (lat, lon)

        # Check for dateline crossing in polygons
        if isinstance(coordinates, PolygonCoordinates):
            if crosses_dateline(coordinates.coordinates):
                metadata["crosses_dateline"] = True
                warnings.append(ValidationError(
                    code=ErrorCode.IN_PROTECTED_AREA,  # Using as general warning
                    message="Polygon crosses international date line - verify coordinates",
                    severity=ValidationSeverity.WARNING,
                    metadata={"crosses_dateline": True}
                ))

        # Use spatial validation service for comprehensive checks
        spatial_result = self.spatial_service.validate_location(
            lat, lon, country_code
        )

        # Check country boundary
        if not spatial_result["in_expected_country"]:
            detected_country = spatial_result.get("detected_country")
            error_msg = f"Coordinates ({lat:.6f}, {lon:.6f}) not within {country_code} boundaries"
            if detected_country and detected_country != country_code:
                error_msg += f" - detected in {detected_country}"

            errors.append(ValidationError(
                code=ErrorCode.NOT_IN_COUNTRY,
                message=error_msg,
                severity=ValidationSeverity.ERROR,
                coordinate=centroid,
                metadata={
                    "country_code": country_code,
                    "detected_country": detected_country
                }
            ))
            metadata["detected_country"] = detected_country

        # Check water bodies
        if spatial_result["in_water"]:
            errors.append(ValidationError(
                code=ErrorCode.IN_WATER_BODY,
                message="Coordinates located in water body",
                severity=ValidationSeverity.ERROR,
                coordinate=centroid
            ))
            metadata["in_water"] = True

        # Check protected areas
        if spatial_result["in_protected_area"]:
            protected_area = spatial_result.get("protected_area_name", "Unknown")
            warnings.append(ValidationError(
                code=ErrorCode.IN_PROTECTED_AREA,
                message=f"Plot overlaps with protected area: {protected_area}",
                severity=ValidationSeverity.WARNING,
                coordinate=centroid,
                metadata={"protected_area": protected_area}
            ))
            metadata["protected_area"] = protected_area
            metadata["in_protected_area"] = True

        # For polygons, check for protected area overlaps
        if isinstance(coordinates, PolygonCoordinates):
            overlaps = spatial_result.get("protected_area_overlaps", [])
            if overlaps:
                for overlap in overlaps:
                    if overlap["name"] != metadata.get("protected_area"):
                        warnings.append(ValidationError(
                            code=ErrorCode.IN_PROTECTED_AREA,
                            message=f"Polygon overlaps protected area: {overlap['name']}",
                            severity=ValidationSeverity.WARNING,
                            metadata=overlap
                        ))
                metadata["protected_area_overlaps"] = overlaps

        # Check urban areas
        if spatial_result["in_urban_area"]:
            warnings.append(ValidationError(
                code=ErrorCode.IN_URBAN_AREA,
                message="Coordinates located in urban area (suspicious for agriculture)",
                severity=ValidationSeverity.WARNING,
                coordinate=centroid
            ))
            metadata["in_urban_area"] = True

        return StageResult(errors=errors, warnings=warnings, metadata=metadata)

    def _validate_polygon(
        self,
        polygon: PolygonCoordinates,
        declared_area: Optional[float] = None
    ) -> StageResult:
        """Validate polygon geometry."""
        errors = []
        warnings = []
        metadata = {}

        coords = polygon.coordinates

        # Check minimum points (4 including closing)
        if len(coords) < 4:
            errors.append(ValidationError(
                code=ErrorCode.INSUFFICIENT_POINTS,
                message=f"Polygon requires at least 3 distinct points, got {len(coords) - 1}",
                severity=ValidationSeverity.ERROR
            ))
            return StageResult(errors=errors, is_fatal=True)

        # Check for self-intersection (simplified check)
        if self._is_self_intersecting(coords):
            errors.append(ValidationError(
                code=ErrorCode.SELF_INTERSECTING,
                message="Polygon has self-intersecting edges",
                severity=ValidationSeverity.ERROR
            ))

        # Calculate area
        area_hectares = self._calculate_area_hectares(coords)
        metadata["area_hectares"] = area_hectares

        if area_hectares < self.MIN_AREA_HECTARES:
            errors.append(ValidationError(
                code=ErrorCode.AREA_TOO_SMALL,
                message=f"Plot area {area_hectares:.4f} ha below minimum {self.MIN_AREA_HECTARES} ha",
                severity=ValidationSeverity.ERROR,
                metadata={"area_hectares": area_hectares, "minimum": self.MIN_AREA_HECTARES}
            ))

        # Check declared vs calculated area
        if declared_area and area_hectares > 0:
            diff_percent = abs(area_hectares - declared_area) / declared_area * 100
            if diff_percent > 20:
                warnings.append(ValidationError(
                    code=ErrorCode.AREA_MISMATCH,
                    message=f"Calculated area {area_hectares:.2f} ha differs from declared {declared_area:.2f} ha by {diff_percent:.1f}%",
                    severity=ValidationSeverity.WARNING,
                    metadata={
                        "calculated_area": area_hectares,
                        "declared_area": declared_area,
                        "difference_percent": diff_percent
                    }
                ))

        # Calculate additional metadata
        metadata["centroid"] = polygon.get_centroid()
        metadata["perimeter_km"] = self._calculate_perimeter_km(coords)
        metadata["bounding_box"] = self._calculate_bounding_box(coords)

        return StageResult(errors=errors, warnings=warnings, metadata=metadata)

    @staticmethod
    def _count_decimal_places(value: float) -> int:
        """
        Count significant decimal places in a coordinate.
        Critical for EUDR compliance (requires 6 decimal places).
        """
        # Convert to string with high precision
        str_value = f"{value:.15f}"

        # Split on decimal point
        if '.' not in str_value:
            return 0

        integer_part, decimal_part = str_value.split('.')

        # Find last non-zero digit
        last_nonzero = len(decimal_part.rstrip('0'))

        return last_nonzero

    def _point_in_country(self, point: Tuple[float, float], country_code: str) -> bool:
        """
        Check if point is within country boundaries.

        Uses the CountryBoundaryService for actual validation.
        Falls back to bounding box check if full GADM data not loaded.
        """
        lat, lon = point
        return self.spatial_service.country_service.is_point_in_country(
            lat, lon, country_code
        )

    def _point_in_water(self, point: Tuple[float, float]) -> bool:
        """
        Check if point is in water body.

        Uses WaterBodyService for validation against OSM data.
        """
        lat, lon = point
        return self.spatial_service.water_service.is_point_in_water(lat, lon)

    def _check_protected_area(
        self,
        coordinates: Union[PointCoordinates, PolygonCoordinates]
    ) -> Optional[str]:
        """
        Check if coordinates overlap with protected area.

        Uses ProtectedAreaService for validation against WDPA data.
        """
        if isinstance(coordinates, PointCoordinates):
            lat, lon = coordinates.latitude, coordinates.longitude
        else:
            lat, lon = coordinates.get_centroid()

        result = self.spatial_service.protected_service.check_protected_area(lat, lon)
        if result.found:
            return result.feature_name
        return None

    def _point_in_urban(self, point: Tuple[float, float]) -> bool:
        """
        Check if point is in urban area.

        Uses UrbanAreaService for validation against OSM data.
        """
        lat, lon = point
        return self.spatial_service.urban_service.is_point_in_urban(lat, lon)

    @staticmethod
    def _is_self_intersecting(coords: List[List[float]]) -> bool:
        """
        Check if polygon has self-intersecting edges.
        Simplified check - full implementation would use Shapely.
        """
        n = len(coords) - 1  # Exclude closing point

        if n < 4:
            return False

        # Check each pair of non-adjacent edges
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Skip adjacent edges at closure

                # Get edge endpoints
                p1 = coords[i]
                p2 = coords[i + 1]
                p3 = coords[j]
                p4 = coords[(j + 1) % n]

                if GeolocationValidator._edges_intersect(p1, p2, p3, p4):
                    return True

        return False

    @staticmethod
    def _edges_intersect(p1, p2, p3, p4) -> bool:
        """Check if two line segments intersect."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    @staticmethod
    def _calculate_area_hectares(coords: List[List[float]]) -> float:
        """
        Calculate polygon area in hectares using Shoelace formula.
        Assumes coordinates in decimal degrees.
        """
        n = len(coords) - 1  # Exclude closing point

        if n < 3:
            return 0.0

        # Shoelace formula for area in degrees
        area_deg = 0.0
        for i in range(n):
            j = (i + 1) % n
            area_deg += coords[i][0] * coords[j][1]
            area_deg -= coords[j][0] * coords[i][1]
        area_deg = abs(area_deg) / 2.0

        # Convert to square meters (approximate at centroid latitude)
        centroid_lat = sum(c[1] for c in coords[:n]) / n

        # Meters per degree at this latitude
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * math.radians(centroid_lat))
        m_per_deg_lon = 111412.84 * math.cos(math.radians(centroid_lat))

        area_m2 = area_deg * m_per_deg_lat * m_per_deg_lon
        area_hectares = area_m2 / 10000

        return round(area_hectares, 4)

    @staticmethod
    def _calculate_perimeter_km(coords: List[List[float]]) -> float:
        """Calculate polygon perimeter in kilometers."""
        n = len(coords) - 1
        total_km = 0.0

        for i in range(n):
            j = (i + 1) % (n + 1)
            total_km += GeolocationValidator._haversine_distance(
                coords[i][1], coords[i][0],
                coords[j][1], coords[j][0]
            )

        return round(total_km, 4)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km using Haversine formula."""
        R = 6371  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = math.sin(delta_lat / 2) ** 2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    def _calculate_bounding_box(coords: List[List[float]]) -> List[float]:
        """Calculate bounding box [min_lon, min_lat, max_lon, max_lat]."""
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [min(lons), min(lats), max(lons), max(lats)]


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class GeolocationCollectorAgent:
    """
    GL-EUDR-002: Geolocation Collector Agent

    Collects, validates, and manages production plot geolocation data
    for EUDR compliance. Provides deterministic validation with zero
    hallucination guarantees.
    """

    def __init__(
        self,
        db_session: Optional[Any] = None,
        validator: Optional[GeolocationValidator] = None,
        llm_service: Optional[Any] = None,
        geocoding_service: Optional[Any] = None
    ) -> None:
        """
        Initialize the Geolocation Collector Agent.

        Args:
            db_session: Database session for persistence
            validator: Custom validator instance (creates default if None)
            llm_service: LLM service for address parsing/explanations
            geocoding_service: Service for reverse geocoding
        """
        self.db = db_session
        self.validator = validator or GeolocationValidator()
        self.llm = llm_service
        self.geocoding = geocoding_service

        # In-memory storage for testing
        self._plots: Dict[UUID, Plot] = {}
        self._validation_history: Dict[UUID, List[PlotValidationHistory]] = {}
        self._bulk_jobs: Dict[UUID, BulkUploadJob] = {}

    def run(self, input_data: GeolocationInput) -> GeolocationOutput:
        """Execute the requested operation."""
        start_time = time.time()

        try:
            if input_data.operation == OperationType.VALIDATE_COORDINATES:
                result = self._validate_coordinates(input_data)
            elif input_data.operation == OperationType.SUBMIT_PLOT:
                result = self._submit_plot(input_data)
            elif input_data.operation == OperationType.REVALIDATE_PLOT:
                result = self._revalidate_plot(input_data)
            elif input_data.operation == OperationType.GET_PLOT:
                result = self._get_plot(input_data)
            elif input_data.operation == OperationType.LIST_PLOTS:
                result = self._list_plots(input_data)
            elif input_data.operation == OperationType.ENRICH_PLOT:
                result = self._enrich_plot(input_data)
            elif input_data.operation == OperationType.BULK_UPLOAD:
                result = self._initiate_bulk_upload(input_data)
            else:
                result = GeolocationOutput(
                    success=False,
                    operation=input_data.operation,
                    errors=[f"Unknown operation: {input_data.operation}"]
                )

            result.processing_time_ms = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.exception(f"Error in operation {input_data.operation}")
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=[str(e)],
                processing_time_ms=int((time.time() - start_time) * 1000)
            )

    # =========================================================================
    # VALIDATION OPERATIONS
    # =========================================================================

    def _validate_coordinates(
        self,
        input_data: GeolocationInput
    ) -> GeolocationOutput:
        """Validate coordinates without storing."""
        if not input_data.coordinates:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Coordinates are required"]
            )

        if not input_data.country_code:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Country code is required"]
            )

        if not input_data.commodity:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Commodity is required"]
            )

        validation_result = self.validator.validate(
            coordinates=input_data.coordinates,
            country_code=input_data.country_code,
            commodity=input_data.commodity,
            declared_area=input_data.declared_area_hectares,
            gps_accuracy_m=input_data.collection_accuracy_m
        )

        return GeolocationOutput(
            success=True,
            operation=input_data.operation,
            validation_result=validation_result
        )

    def _submit_plot(self, input_data: GeolocationInput) -> GeolocationOutput:
        """Submit and validate a new plot."""
        # Validate required fields
        if not all([
            input_data.coordinates,
            input_data.country_code,
            input_data.commodity,
            input_data.supplier_id
        ]):
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Missing required fields: coordinates, country_code, commodity, supplier_id"]
            )

        # Run validation
        validation_result = self.validator.validate(
            coordinates=input_data.coordinates,
            country_code=input_data.country_code,
            commodity=input_data.commodity,
            declared_area=input_data.declared_area_hectares,
            gps_accuracy_m=input_data.collection_accuracy_m
        )

        # Check for overlaps with existing plots (FR-024/FR-025)
        overlap_issues = self.check_for_overlaps(
            coordinates=input_data.coordinates,
            supplier_id=input_data.supplier_id
        )

        # Add overlap issues to validation result
        for issue in overlap_issues:
            if issue.severity == ValidationSeverity.ERROR:
                validation_result.errors.append(issue)
                validation_result.valid = False
                validation_result.status = ValidationStatus.INVALID
            else:
                validation_result.warnings.append(issue)
                if validation_result.status == ValidationStatus.VALID:
                    validation_result.status = ValidationStatus.NEEDS_REVIEW

        # Determine geometry type
        is_polygon = isinstance(input_data.coordinates, PolygonCoordinates)
        geometry_type = GeometryType.POLYGON if is_polygon else GeometryType.POINT

        # Calculate centroid and area
        centroid = None
        area_hectares = None
        perimeter_km = None
        bounding_box = None

        if is_polygon:
            coords = input_data.coordinates.coordinates
            centroid_tuple = input_data.coordinates.get_centroid()
            centroid = PointCoordinates(latitude=centroid_tuple[0], longitude=centroid_tuple[1])
            area_hectares = Decimal(str(validation_result.metadata.get("area_hectares", 0)))
            perimeter_km = Decimal(str(validation_result.metadata.get("perimeter_km", 0)))
            bounding_box = validation_result.metadata.get("bounding_box")
        else:
            centroid = input_data.coordinates

        # Create plot
        plot = Plot(
            supplier_id=input_data.supplier_id,
            geometry_type=geometry_type,
            coordinates=input_data.coordinates,
            centroid=centroid,
            bounding_box=bounding_box,
            area_hectares=area_hectares,
            perimeter_km=perimeter_km,
            country_code=input_data.country_code,
            commodity=input_data.commodity,
            validation_status=validation_result.status,
            validation_errors=validation_result.errors,
            validation_warnings=validation_result.warnings,
            precision_lat=validation_result.metadata.get("min_lat_precision"),
            precision_lon=validation_result.metadata.get("min_lon_precision"),
            last_validated_at=datetime.utcnow(),
            collection_method=input_data.collection_method,
            collection_accuracy_m=input_data.collection_accuracy_m,
            in_protected_area=validation_result.metadata.get("protected_area") is not None,
            protected_area_name=validation_result.metadata.get("protected_area"),
            in_urban_area=validation_result.metadata.get("in_urban_area", False)
        )

        # Store plot
        self._plots[plot.plot_id] = plot

        # Create validation history entry
        history = PlotValidationHistory(
            plot_id=plot.plot_id,
            status=validation_result.status,
            errors=validation_result.errors,
            warnings=validation_result.warnings,
            validation_method="AUTO"
        )
        if plot.plot_id not in self._validation_history:
            self._validation_history[plot.plot_id] = []
        self._validation_history[plot.plot_id].append(history)

        return GeolocationOutput(
            success=True,
            operation=input_data.operation,
            validation_result=validation_result,
            plot=plot
        )

    def _revalidate_plot(self, input_data: GeolocationInput) -> GeolocationOutput:
        """Re-validate an existing plot."""
        if not input_data.plot_id:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Plot ID is required"]
            )

        plot = self._plots.get(input_data.plot_id)
        if not plot:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=[f"Plot {input_data.plot_id} not found"]
            )

        # Re-run validation
        validation_result = self.validator.validate(
            coordinates=plot.coordinates,
            country_code=plot.country_code,
            commodity=plot.commodity,
            declared_area=float(plot.area_hectares) if plot.area_hectares else None,
            gps_accuracy_m=plot.collection_accuracy_m
        )

        # Update plot
        plot.validation_status = validation_result.status
        plot.validation_errors = validation_result.errors
        plot.validation_warnings = validation_result.warnings
        plot.last_validated_at = datetime.utcnow()
        plot.updated_at = datetime.utcnow()

        # Create validation history entry
        history = PlotValidationHistory(
            plot_id=plot.plot_id,
            status=validation_result.status,
            errors=validation_result.errors,
            warnings=validation_result.warnings,
            validation_method="AUTO"
        )
        self._validation_history[plot.plot_id].append(history)

        return GeolocationOutput(
            success=True,
            operation=input_data.operation,
            validation_result=validation_result,
            plot=plot
        )

    # =========================================================================
    # PLOT CRUD OPERATIONS
    # =========================================================================

    def _get_plot(self, input_data: GeolocationInput) -> GeolocationOutput:
        """Get a plot by ID."""
        if not input_data.plot_id:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Plot ID is required"]
            )

        plot = self._plots.get(input_data.plot_id)
        if not plot:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=[f"Plot {input_data.plot_id} not found"]
            )

        return GeolocationOutput(
            success=True,
            operation=input_data.operation,
            plot=plot
        )

    def _list_plots(self, input_data: GeolocationInput) -> GeolocationOutput:
        """List plots with optional filtering."""
        plots = list(self._plots.values())

        # Filter by supplier
        if input_data.supplier_id:
            plots = [p for p in plots if p.supplier_id == input_data.supplier_id]

        # Filter by validation status
        if input_data.validation_status:
            plots = [p for p in plots if p.validation_status == input_data.validation_status]

        # Filter by commodity
        if input_data.commodity:
            plots = [p for p in plots if p.commodity == input_data.commodity]

        # Apply pagination
        total_count = len(plots)
        plots = plots[input_data.offset:input_data.offset + input_data.limit]

        return GeolocationOutput(
            success=True,
            operation=input_data.operation,
            plots=plots,
            total_count=total_count
        )

    # =========================================================================
    # ENRICHMENT OPERATIONS
    # =========================================================================

    def _enrich_plot(self, input_data: GeolocationInput) -> GeolocationOutput:
        """Enrich plot with geocoding and other data."""
        if not input_data.plot_id:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Plot ID is required"]
            )

        plot = self._plots.get(input_data.plot_id)
        if not plot:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=[f"Plot {input_data.plot_id} not found"]
            )

        # Perform reverse geocoding if service available
        if self.geocoding and plot.centroid:
            try:
                geocode_result = self.geocoding.reverse(
                    plot.centroid.latitude,
                    plot.centroid.longitude
                )
                if geocode_result:
                    plot.admin_level_1 = geocode_result.get("state")
                    plot.admin_level_2 = geocode_result.get("county")
                    plot.admin_level_3 = geocode_result.get("city")
                    plot.nearest_place = geocode_result.get("place")
            except Exception as e:
                logger.warning(f"Geocoding failed for plot {plot.plot_id}: {e}")

        plot.updated_at = datetime.utcnow()

        return GeolocationOutput(
            success=True,
            operation=input_data.operation,
            plot=plot
        )

    # =========================================================================
    # BULK UPLOAD OPERATIONS
    # =========================================================================

    def _initiate_bulk_upload(self, input_data: GeolocationInput) -> GeolocationOutput:
        """Initiate async bulk upload processing."""
        if not input_data.file_path:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["File path is required"]
            )

        if not input_data.file_format:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["File format is required"]
            )

        if not input_data.supplier_id:
            return GeolocationOutput(
                success=False,
                operation=input_data.operation,
                errors=["Supplier ID is required"]
            )

        # Create bulk job
        job = BulkUploadJob(
            supplier_id=input_data.supplier_id,
            file_format=input_data.file_format,
            file_name=input_data.file_path.split("/")[-1],
            file_size_bytes=0  # Would get actual size
        )

        self._bulk_jobs[job.job_id] = job

        # In production, would queue job for async processing
        # For now, return the job info

        return GeolocationOutput(
            success=True,
            operation=input_data.operation,
            bulk_job=job
        )

    def get_bulk_job_status(self, job_id: UUID) -> Optional[BulkUploadJob]:
        """Get bulk upload job status."""
        return self._bulk_jobs.get(job_id)

    # =========================================================================
    # OVERLAP DETECTION (FR-024/FR-025)
    # =========================================================================

    def check_for_overlaps(
        self,
        coordinates: Union[PointCoordinates, PolygonCoordinates],
        supplier_id: UUID,
        exclude_plot_id: Optional[UUID] = None
    ) -> List[ValidationError]:
        """
        Check for overlapping or duplicate plots (FR-024/FR-025).

        Detects:
        - Exact duplicate coordinates
        - Overlapping polygons (>50% overlap)
        - Points within existing polygons

        Args:
            coordinates: New plot coordinates
            supplier_id: Supplier ID to scope search
            exclude_plot_id: Plot ID to exclude (for updates)

        Returns:
            List of ValidationError for any detected overlaps
        """
        overlap_warnings = []

        # Get existing plots for this supplier
        existing_plots = [
            p for p in self._plots.values()
            if p.supplier_id == supplier_id and p.plot_id != exclude_plot_id
        ]

        if not existing_plots:
            return overlap_warnings

        if isinstance(coordinates, PointCoordinates):
            # Check for exact duplicate points
            for plot in existing_plots:
                if isinstance(plot.coordinates, PointCoordinates):
                    if (abs(plot.coordinates.latitude - coordinates.latitude) < 1e-7 and
                        abs(plot.coordinates.longitude - coordinates.longitude) < 1e-7):
                        overlap_warnings.append(ValidationError(
                            code=ErrorCode.DUPLICATE_COORDINATES,
                            message=f"Exact duplicate of existing plot {plot.plot_id}",
                            severity=ValidationSeverity.ERROR,
                            coordinate=(coordinates.latitude, coordinates.longitude),
                            metadata={"existing_plot_id": str(plot.plot_id)}
                        ))
                elif isinstance(plot.coordinates, PolygonCoordinates):
                    # Check if point is inside existing polygon
                    if self._point_in_polygon(coordinates, plot.coordinates):
                        overlap_warnings.append(ValidationError(
                            code=ErrorCode.PLOT_OVERLAP,
                            message=f"Point is inside existing polygon plot {plot.plot_id}",
                            severity=ValidationSeverity.WARNING,
                            coordinate=(coordinates.latitude, coordinates.longitude),
                            metadata={"existing_plot_id": str(plot.plot_id)}
                        ))

        elif isinstance(coordinates, PolygonCoordinates):
            # Check for polygon overlaps
            for plot in existing_plots:
                if isinstance(plot.coordinates, PolygonCoordinates):
                    overlap_pct = self._calculate_polygon_overlap(
                        coordinates, plot.coordinates
                    )
                    if overlap_pct > 0.5:  # >50% overlap threshold
                        overlap_warnings.append(ValidationError(
                            code=ErrorCode.PLOT_OVERLAP,
                            message=f"Polygon overlaps {overlap_pct*100:.1f}% with existing plot {plot.plot_id}",
                            severity=ValidationSeverity.WARNING,
                            metadata={
                                "existing_plot_id": str(plot.plot_id),
                                "overlap_percentage": overlap_pct * 100
                            }
                        ))

        return overlap_warnings

    def _point_in_polygon(
        self,
        point: PointCoordinates,
        polygon: PolygonCoordinates
    ) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm.
        Simple implementation for in-memory checks.
        """
        x, y = point.longitude, point.latitude
        coords = polygon.coordinates
        n = len(coords)
        inside = False

        p1x, p1y = coords[0]
        for i in range(1, n + 1):
            p2x, p2y = coords[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _calculate_polygon_overlap(
        self,
        poly1: PolygonCoordinates,
        poly2: PolygonCoordinates
    ) -> float:
        """
        Calculate overlap percentage between two polygons.
        Returns approximate overlap using bounding box intersection.

        For production use PostGIS ST_Intersection.
        """
        # Get bounding boxes
        def get_bbox(coords):
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            return min(lons), min(lats), max(lons), max(lats)

        bbox1 = get_bbox(poly1.coordinates)
        bbox2 = get_bbox(poly2.coordinates)

        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0  # No intersection

        # Calculate areas (approximate using bbox)
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

        if bbox1_area == 0:
            return 0.0

        return min(1.0, intersection_area / bbox1_area)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def get_validation_history(self, plot_id: UUID) -> List[PlotValidationHistory]:
        """Get validation history for a plot."""
        return self._validation_history.get(plot_id, [])

    def get_all_plots(self) -> List[Plot]:
        """Get all plots (for testing)."""
        return list(self._plots.values())


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "pack_id": "greenlang/eudr-geolocation-collector",
    "version": "1.0.0",
    "name": "EUDR Geolocation Collector",
    "description": "Collects and validates production plot geolocation data for EUDR compliance",
    "agent_family": "EUDRTraceabilityFamily",
    "layer": "Supply Chain Traceability",
    "domains": ["geolocation", "validation", "eudr"],
    "inputs": {
        "coordinates": "Point or Polygon coordinates",
        "country_code": "ISO 2-letter country code",
        "commodity": "EUDR commodity type"
    },
    "outputs": {
        "validation_result": "Validation status with errors/warnings",
        "plot": "Validated and enriched plot data"
    },
    "regulatory_reference": "EU Regulation 2023/1115 - Article 9",
    "precision_requirement": "6 decimal places (~0.11m)"
}
