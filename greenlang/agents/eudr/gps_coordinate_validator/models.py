# -*- coding: utf-8 -*-
"""
GPS Coordinate Validator Data Models - AGENT-EUDR-007

Pydantic v2 data models for the GPS Coordinate Validator Agent covering
multi-format coordinate parsing, geodetic datum transformation, precision
analysis, format validation, spatial plausibility checking, reverse
geocoding, accuracy assessment, and EUDR Article 9 compliance reporting.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all GPS coordinate validation operations.

Enumerations (14):
    - CoordinateFormat, GeodeticDatum, PrecisionLevel,
      ValidationErrorType, PlausibilityCheckResult, AccuracyTier,
      CorrectionType, ReportFormat, ComplianceStatus, SourceType,
      LandUseContext, BatchStatus, ElevationSource, HemisphereIndicator

Core Models (7):
    - RawCoordinate, ParsedCoordinate, NormalizedCoordinate,
      CoordinateValidationError, ValidationResult, PrecisionResult,
      DatumTransformResult

Result Models (4):
    - PlausibilityResult, ReverseGeocodeResult, AccuracyScore,
      ComplianceCertificate

Request Models (8):
    - ParseCoordinateRequest, ValidateCoordinateRequest,
      TransformDatumRequest, AnalyzePrecisionRequest,
      CheckPlausibilityRequest, ReverseGeocodeRequest,
      AssessAccuracyRequest, GenerateReportRequest

Response Models (8):
    - ParseCoordinateResponse, ValidateCoordinateResponse,
      TransformDatumResponse, AnalyzePrecisionResponse,
      CheckPlausibilityResponse, ReverseGeocodeResponse,
      AssessAccuracyResponse, BatchValidationResult

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-001 Supply Chain Mapper.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GCV-007)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from greenlang.agents.data.eudr_traceability.models import EUDRCommodity
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Maximum number of coordinates in a single batch validation request.
MAX_BATCH_SIZE: int = 50_000

#: Default accuracy score component weights (must sum to 1.0).
DEFAULT_QUALITY_WEIGHTS: Dict[str, float] = {
    "precision": 0.25,
    "plausibility": 0.25,
    "consistency": 0.25,
    "source": 0.25,
}

#: Accuracy tier thresholds mapping tier to (min_score, max_score).
ACCURACY_TIER_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "gold": (90.0, 100.0),
    "silver": (70.0, 89.99),
    "bronze": (50.0, 69.99),
    "unverified": (0.0, 49.99),
}

#: Minimum decimal places required for EUDR compliance (5 = ~1.1m).
EUDR_MIN_DECIMAL_PLACES: int = 5

#: WGS84 semi-major axis in meters.
WGS84_SEMI_MAJOR_AXIS: float = 6378137.0

#: WGS84 inverse flattening.
WGS84_INV_FLATTENING: float = 298.257223563

#: WGS84 flattening.
WGS84_FLATTENING: float = 1.0 / WGS84_INV_FLATTENING

# =============================================================================
# Enumerations
# =============================================================================

class CoordinateFormat(str, Enum):
    """Supported GPS coordinate input formats.

    The GPS Coordinate Validator accepts coordinates in multiple common
    formats and normalizes them to decimal degrees (DD) in WGS84 for
    EUDR compliance processing.

    DECIMAL_DEGREES: Standard decimal format (e.g., -3.4653, -62.2159).
    DMS: Degrees, minutes, seconds (e.g., 3d27'55"S 62d12'57"W).
    DDM: Degrees and decimal minutes (e.g., 3d27.917'S 62d12.954'W).
    UTM: Universal Transverse Mercator (e.g., 20M 584123 9616789).
    MGRS: Military Grid Reference System (e.g., 20MNA8412316789).
    SIGNED_DD: Signed decimal degrees with explicit +/- (e.g., -3.4653).
    DD_SUFFIX: Decimal degrees with N/S/E/W suffix (e.g., 3.4653S).
    UNKNOWN: Format could not be determined from the input string.
    """

    DECIMAL_DEGREES = "decimal_degrees"
    DMS = "dms"
    DDM = "ddm"
    UTM = "utm"
    MGRS = "mgrs"
    SIGNED_DD = "signed_dd"
    DD_SUFFIX = "dd_suffix"
    UNKNOWN = "unknown"

class GeodeticDatum(str, Enum):
    """Geodetic reference datums supported for coordinate transformation.

    The validator can accept coordinates in any of these datums and
    transform them to WGS84 (EPSG:4326) using Helmert 7-parameter
    or Molodensky 3-parameter transformations.

    WGS84: World Geodetic System 1984 (canonical EUDR datum).
    NAD27: North American Datum 1927 (Clarke 1866 ellipsoid).
    NAD83: North American Datum 1983 (GRS 1980 ellipsoid).
    ED50: European Datum 1950 (International 1924 ellipsoid).
    ETRS89: European Terrestrial Reference System 1989.
    OSGB36: Ordnance Survey Great Britain 1936.
    SIRGAS_2000: South American Geocentric Reference System 2000.
    INDIAN_1975: Indian national datum 1975 (Everest 1830).
    ARC_1960: Arc 1960 datum used in East Africa.
    PULKOVO_1942: Soviet geodetic datum used in former USSR.
    TOKYO: Tokyo datum used in Japan and Korea.
    GDA94: Geocentric Datum of Australia 1994.
    GDA2020: Geocentric Datum of Australia 2020.
    NZGD2000: New Zealand Geodetic Datum 2000.
    CAPE: Cape Datum used in South Africa.
    HERMANNSKOGEL: Datum used in Austria and Central Europe.
    POTSDAM: Rauenberg datum used in Germany.
    ROME_1940: Monte Mario datum used in Italy.
    BESSEL_1841: Bessel 1841 ellipsoid datum (Central Europe).
    KERTAU_1948: Kertau 1948 datum (Malaysia, Singapore).
    LUZON_1911: Luzon 1911 datum (Philippines).
    TIMBALAI_1948: Timbalai 1948 datum (Brunei, East Malaysia).
    EVEREST_1956: Everest 1956 datum (India, Nepal).
    KALIANPUR_1975: Kalianpur 1975 datum (India).
    HONG_KONG_1980: Hong Kong 1980 datum.
    SOUTH_AMERICAN_1969: South American Datum 1969.
    BOGOTA_1975: Bogota datum (Colombia).
    CAMPO_INCHAUSPE: Campo Inchauspe datum (Argentina).
    CHUA_ASTRO: Chua Astro datum (Paraguay).
    CORREGO_ALEGRE: Corrego Alegre datum (Brazil).
    YACARE: Yacare datum (Uruguay).
    ZANDERIJ: Zanderij datum (Suriname).
    ADINDAN: Adindan datum (North/East Africa).
    MINNA: Minna datum (Nigeria).
    CAMACUPA: Camacupa datum (Angola).
    SCHWARZECK: Schwarzeck datum (Namibia).
    HARTEBEESTHOEK94: Hartebeesthoek 94 (South Africa).
    UNKNOWN: Datum could not be determined.
    """

    WGS84 = "wgs84"
    NAD27 = "nad27"
    NAD83 = "nad83"
    ED50 = "ed50"
    ETRS89 = "etrs89"
    OSGB36 = "osgb36"
    SIRGAS_2000 = "sirgas_2000"
    INDIAN_1975 = "indian_1975"
    ARC_1960 = "arc_1960"
    PULKOVO_1942 = "pulkovo_1942"
    TOKYO = "tokyo"
    GDA94 = "gda94"
    GDA2020 = "gda2020"
    NZGD2000 = "nzgd2000"
    CAPE = "cape"
    HERMANNSKOGEL = "hermannskogel"
    POTSDAM = "potsdam"
    ROME_1940 = "rome_1940"
    BESSEL_1841 = "bessel_1841"
    KERTAU_1948 = "kertau_1948"
    LUZON_1911 = "luzon_1911"
    TIMBALAI_1948 = "timbalai_1948"
    EVEREST_1956 = "everest_1956"
    KALIANPUR_1975 = "kalianpur_1975"
    HONG_KONG_1980 = "hong_kong_1980"
    SOUTH_AMERICAN_1969 = "south_american_1969"
    BOGOTA_1975 = "bogota_1975"
    CAMPO_INCHAUSPE = "campo_inchauspe"
    CHUA_ASTRO = "chua_astro"
    CORREGO_ALEGRE = "corrego_alegre"
    YACARE = "yacare"
    ZANDERIJ = "zanderij"
    ADINDAN = "adindan"
    MINNA = "minna"
    CAMACUPA = "camacupa"
    SCHWARZECK = "schwarzeck"
    HARTEBEESTHOEK94 = "hartebeesthoek94"
    UNKNOWN = "unknown"

class PrecisionLevel(str, Enum):
    """Classification of coordinate precision based on ground resolution.

    Maps the number of decimal places to an approximate ground
    resolution in meters at the equator.

    SURVEY_GRADE: <= 1m ground resolution (>= 5 decimal places).
        Meets EUDR Article 9 requirements without reservation.
    HIGH: <= 10m ground resolution (4 decimal places).
        Meets EUDR requirements with minor caveats.
    MODERATE: <= 100m ground resolution (3 decimal places).
        May be acceptable for large plots (> 20 hectares).
    LOW: <= 1000m ground resolution (2 decimal places).
        Generally insufficient for EUDR compliance.
    INADEQUATE: > 1000m ground resolution (< 2 decimal places).
        Cannot be used for EUDR compliance.
    """

    SURVEY_GRADE = "survey_grade"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INADEQUATE = "inadequate"

class ValidationErrorType(str, Enum):
    """Classification of coordinate validation errors.

    Each error type corresponds to a specific validation check
    performed by the GPS Coordinate Validator.

    OUT_OF_RANGE: Coordinate value exceeds WGS84 valid ranges
        (latitude not in [-90, 90] or longitude not in [-180, 180]).
    SWAPPED_LAT_LON: Latitude and longitude values appear to be
        swapped based on range analysis and contextual heuristics.
    SIGN_ERROR: Sign of latitude or longitude is likely incorrect
        for the declared country (e.g., positive latitude for Brazil).
    HEMISPHERE_ERROR: Hemisphere indicator (N/S/E/W) does not match
        the numeric sign of the coordinate value.
    NULL_ISLAND: Coordinate is at or near (0.0, 0.0), which is
        typically a default/missing value, not a real location.
    NAN_VALUE: Coordinate contains NaN (Not a Number) values.
    INF_VALUE: Coordinate contains infinity values.
    NULL_VALUE: Coordinate is null/None/empty.
    DUPLICATE: Exact duplicate of another coordinate in the dataset.
    NEAR_DUPLICATE: Very close to another coordinate within the
        configured distance threshold.
    TRUNCATED: Coordinate appears to have been truncated (sudden
        loss of decimal precision in a series).
    ARTIFICIALLY_ROUNDED: Coordinate appears to have been rounded
        to an unlikely number of decimal places (e.g., whole degrees).
    FORMAT_ERROR: Input string could not be parsed as a valid
        coordinate in any supported format.
    """

    OUT_OF_RANGE = "out_of_range"
    SWAPPED_LAT_LON = "swapped_lat_lon"
    SIGN_ERROR = "sign_error"
    HEMISPHERE_ERROR = "hemisphere_error"
    NULL_ISLAND = "null_island"
    NAN_VALUE = "nan_value"
    INF_VALUE = "inf_value"
    NULL_VALUE = "null_value"
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    TRUNCATED = "truncated"
    ARTIFICIALLY_ROUNDED = "artificially_rounded"
    FORMAT_ERROR = "format_error"

class PlausibilityCheckResult(str, Enum):
    """Classification of spatial plausibility check outcomes.

    Each value represents a specific plausibility assessment that
    can be performed on a validated coordinate.

    LAND: Coordinate falls on land (passed ocean check).
    OCEAN: Coordinate falls in an ocean or large water body.
    COUNTRY_MATCH: Coordinate falls within the declared country.
    COUNTRY_MISMATCH: Coordinate falls outside the declared country.
    COMMODITY_PLAUSIBLE: Coordinate is in a region where the declared
        commodity is known to be produced.
    COMMODITY_IMPLAUSIBLE: Coordinate is in a region where the declared
        commodity production is highly unlikely.
    ELEVATION_PLAUSIBLE: Elevation at the coordinate is within the
        expected range for the declared commodity.
    ELEVATION_IMPLAUSIBLE: Elevation at the coordinate is outside
        the expected range for the declared commodity.
    URBAN_AREA: Coordinate falls within a mapped urban area.
    PROTECTED_AREA: Coordinate falls within a protected area boundary.
    """

    LAND = "land"
    OCEAN = "ocean"
    COUNTRY_MATCH = "country_match"
    COUNTRY_MISMATCH = "country_mismatch"
    COMMODITY_PLAUSIBLE = "commodity_plausible"
    COMMODITY_IMPLAUSIBLE = "commodity_implausible"
    ELEVATION_PLAUSIBLE = "elevation_plausible"
    ELEVATION_IMPLAUSIBLE = "elevation_implausible"
    URBAN_AREA = "urban_area"
    PROTECTED_AREA = "protected_area"

class AccuracyTier(str, Enum):
    """Accuracy tier classification based on composite quality score.

    Each tier has specific regulatory implications for Due Diligence
    Statement (DDS) eligibility under EUDR Article 9.

    GOLD: Score 90-100. Fully validated, all checks passed.
        DDS-ready with minimal additional due diligence required.
    SILVER: Score 70-89. Mostly validated with minor issues.
        DDS-eligible with noted limitations.
    BRONZE: Score 50-69. Partially validated with significant issues.
        Enhanced due diligence required before DDS submission.
    UNVERIFIED: Score 0-49. Insufficient validation completed.
        Not eligible for DDS submission; remediation required.
    """

    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"
    UNVERIFIED = "unverified"

class CorrectionType(str, Enum):
    """Classification of auto-correction actions for coordinate errors.

    Each correction type maps to a specific transformation that can
    be automatically applied when the confidence threshold is met.

    SWAP_LAT_LON: Swap latitude and longitude values.
    NEGATE_LAT: Negate the latitude (flip hemisphere N/S).
    NEGATE_LON: Negate the longitude (flip hemisphere E/W).
    ADD_HEMISPHERE: Add missing hemisphere indicator.
    REMOVE_HEMISPHERE: Remove duplicate/conflicting hemisphere indicator.
    DATUM_TRANSFORM: Apply datum transformation to WGS84.
    PRECISION_ENHANCE: Flag for precision enhancement (no auto-fix).
    NO_CORRECTION: No auto-correction available or applicable.
    """

    SWAP_LAT_LON = "swap_lat_lon"
    NEGATE_LAT = "negate_lat"
    NEGATE_LON = "negate_lon"
    ADD_HEMISPHERE = "add_hemisphere"
    REMOVE_HEMISPHERE = "remove_hemisphere"
    DATUM_TRANSFORM = "datum_transform"
    PRECISION_ENHANCE = "precision_enhance"
    NO_CORRECTION = "no_correction"

class ComplianceStatus(str, Enum):
    """EUDR compliance status for a validated coordinate.

    COMPLIANT: Coordinate meets all EUDR Article 9 requirements
        for geolocation data quality and precision.
    NON_COMPLIANT: Coordinate fails one or more EUDR requirements
        and must be remediated before DDS submission.
    NEEDS_REVIEW: Coordinate has borderline results that require
        human review before a compliance determination.
    INSUFFICIENT_DATA: Not enough information available to make
        a compliance determination (missing country, commodity, etc.).
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    INSUFFICIENT_DATA = "insufficient_data"

class SourceType(str, Enum):
    """Classification of coordinate data source for quality weighting.

    The source type affects the source quality component of the
    accuracy score. Higher-quality sources receive higher weights.

    GNSS_SURVEY: Professional GNSS survey equipment (RTK, DGPS).
        Highest accuracy, typically sub-meter precision.
    MOBILE_GPS: Consumer-grade mobile device GPS (smartphone, tablet).
        Typical accuracy 3-10 meters.
    MANUAL_ENTRY: Coordinates typed manually by a human operator.
        Prone to transcription errors and formatting issues.
    DIGITIZED_MAP: Coordinates extracted from digitized paper maps.
        Accuracy depends on map scale and georeferencing quality.
    ERP_EXPORT: Coordinates exported from an ERP or supply chain
        management system. May have been through multiple transformations.
    CERTIFICATION_DB: Coordinates from a certification database
        (e.g., Rainforest Alliance, UTZ, organic certification).
    GOVERNMENT_REGISTRY: Coordinates from an official government
        land registry or cadastral database.
    UNKNOWN: Source of coordinate data is not known.
    """

    GNSS_SURVEY = "gnss_survey"
    MOBILE_GPS = "mobile_gps"
    MANUAL_ENTRY = "manual_entry"
    DIGITIZED_MAP = "digitized_map"
    ERP_EXPORT = "erp_export"
    CERTIFICATION_DB = "certification_db"
    GOVERNMENT_REGISTRY = "government_registry"
    UNKNOWN = "unknown"

class LandUseContext(str, Enum):
    """Land use classification at the coordinate location.

    Used for plausibility checking to determine whether the declared
    commodity is consistent with the observed land use.

    FOREST: Forested area (natural or plantation).
    AGRICULTURAL: Active agricultural land.
    URBAN: Urban or built-up area.
    WATER: Open water body (ocean, lake, river).
    DESERT: Arid or desert region.
    GRASSLAND: Grassland or savanna.
    WETLAND: Wetland or marsh area.
    UNKNOWN: Land use could not be determined.
    """

    FOREST = "forest"
    AGRICULTURAL = "agricultural"
    URBAN = "urban"
    WATER = "water"
    DESERT = "desert"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    UNKNOWN = "unknown"

class BatchStatus(str, Enum):
    """Processing status for a batch coordinate validation job.

    PENDING: Job has been submitted but not yet started.
    RUNNING: Job is currently being processed.
    COMPLETED: Job has finished successfully.
    FAILED: Job encountered a fatal error and could not complete.
    CANCELLED: Job was cancelled by the operator.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ElevationSource(str, Enum):
    """Source of elevation data used for plausibility checking.

    SRTM: Shuttle Radar Topography Mission (NASA, ~30m resolution).
    ASTER: Advanced Spaceborne Thermal Emission and Reflection
        Radiometer GDEM (~30m resolution).
    COPERNICUS_DEM: Copernicus DEM (ESA, ~30m global, ~10m Europe).
    MANUAL: Elevation provided by the data submitter.
    ESTIMATED: Elevation estimated from nearby known points.
    """

    SRTM = "srtm"
    ASTER = "aster"
    COPERNICUS_DEM = "copernicus_dem"
    MANUAL = "manual"
    ESTIMATED = "estimated"

class HemisphereIndicator(str, Enum):
    """Cardinal hemisphere indicator for DMS/DDM coordinate formats.

    NORTH: Northern hemisphere (positive latitude).
    SOUTH: Southern hemisphere (negative latitude).
    EAST: Eastern hemisphere (positive longitude).
    WEST: Western hemisphere (negative longitude).
    """

    NORTH = "N"
    SOUTH = "S"
    EAST = "E"
    WEST = "W"

# =============================================================================
# Core Data Models
# =============================================================================

class RawCoordinate(GreenLangBase):
    """A raw, unparsed GPS coordinate input for validation.

    Represents the original coordinate data as received from any
    upstream source before any parsing, normalization, or validation.

    Attributes:
        raw_input: Original coordinate string exactly as received.
        format_hint: Optional hint about the expected coordinate format.
        datum_hint: Optional hint about the geodetic datum.
        commodity: Optional EUDR commodity for context-aware validation.
        country_iso: Optional ISO 3166-1 alpha-2 country code.
        source_type: How the coordinate data was originally captured.
        metadata: Optional dictionary of additional contextual fields.
    """

    model_config = ConfigDict(from_attributes=True)

    raw_input: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Original coordinate string as received",
    )
    format_hint: Optional[CoordinateFormat] = Field(
        None,
        description="Optional hint about expected coordinate format",
    )
    datum_hint: Optional[GeodeticDatum] = Field(
        None,
        description="Optional hint about geodetic datum",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=100,
        description="EUDR commodity for context-aware validation",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    source_type: SourceType = Field(
        default=SourceType.UNKNOWN,
        description="How the coordinate data was captured",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual fields",
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code to uppercase ISO alpha-2."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

class ParsedCoordinate(GreenLangBase):
    """A coordinate parsed from its raw format into numeric values.

    Contains the extracted latitude and longitude in decimal degrees,
    the detected format, parse confidence, and any warnings generated
    during parsing.

    Attributes:
        latitude: Parsed latitude in decimal degrees.
        longitude: Parsed longitude in decimal degrees.
        altitude: Optional parsed altitude in meters above ellipsoid.
        detected_format: Coordinate format detected during parsing.
        confidence: Parser confidence in the result (0.0-1.0).
        original_input: Original input string that was parsed.
        datum: Geodetic datum of the parsed coordinate.
        parse_warnings: Warnings generated during parsing.
    """

    model_config = ConfigDict(from_attributes=True)

    latitude: float = Field(
        ...,
        description="Parsed latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        description="Parsed longitude in decimal degrees",
    )
    altitude: Optional[float] = Field(
        None,
        description="Altitude in meters above ellipsoid",
    )
    detected_format: CoordinateFormat = Field(
        default=CoordinateFormat.UNKNOWN,
        description="Coordinate format detected during parsing",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Parser confidence in the result (0.0-1.0)",
    )
    original_input: str = Field(
        default="",
        description="Original input string that was parsed",
    )
    datum: GeodeticDatum = Field(
        default=GeodeticDatum.WGS84,
        description="Geodetic datum of the parsed coordinate",
    )
    parse_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during parsing",
    )

class NormalizedCoordinate(GreenLangBase):
    """A coordinate normalized to WGS84 decimal degrees.

    Contains the final latitude and longitude values after format
    normalization and optional datum transformation, along with
    precision metadata for quality assessment.

    Attributes:
        latitude: Normalized latitude in WGS84 decimal degrees.
        longitude: Normalized longitude in WGS84 decimal degrees.
        altitude: Optional altitude in meters above WGS84 ellipsoid.
        datum: Geodetic datum (always WGS84 after normalization).
        decimal_places_lat: Number of significant decimal places in latitude.
        decimal_places_lon: Number of significant decimal places in longitude.
        ground_resolution_m: Approximate ground resolution in meters
            based on the number of decimal places at this latitude.
        source_datum: Original datum before transformation (if any).
        transformation_displacement_m: Distance in meters the coordinate
            was shifted during datum transformation.
    """

    model_config = ConfigDict(from_attributes=True)

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Normalized latitude in WGS84 decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Normalized longitude in WGS84 decimal degrees",
    )
    altitude: Optional[float] = Field(
        None,
        description="Altitude in meters above WGS84 ellipsoid",
    )
    datum: GeodeticDatum = Field(
        default=GeodeticDatum.WGS84,
        description="Geodetic datum (always WGS84 after normalization)",
    )
    decimal_places_lat: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Significant decimal places in latitude",
    )
    decimal_places_lon: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Significant decimal places in longitude",
    )
    ground_resolution_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Approximate ground resolution in meters",
    )
    source_datum: Optional[GeodeticDatum] = Field(
        None,
        description="Original datum before transformation",
    )
    transformation_displacement_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Displacement from datum transformation in meters",
    )

class CoordinateValidationError(GreenLangBase):
    """A single error or issue detected during coordinate validation.

    Represents one specific problem found with a GPS coordinate,
    including the error classification, severity, description,
    auto-correctability assessment, and suggested correction if
    available.

    Attributes:
        error_type: Classification of the validation error.
        description: Human-readable explanation of the error.
        severity: How critical this error is (error, warning, info).
        auto_correctable: Whether this error can be automatically fixed.
        correction_type: Suggested correction type if auto-correctable.
        corrected_value: Suggested corrected coordinate string.
        confidence: Confidence in the suggested correction (0.0-1.0).
    """

    model_config = ConfigDict(from_attributes=True)

    error_type: ValidationErrorType = Field(
        ...,
        description="Classification of the validation error",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the error",
    )
    severity: str = Field(
        default="error",
        description="Issue severity: 'error', 'warning', or 'info'",
    )
    auto_correctable: bool = Field(
        default=False,
        description="Whether this error can be automatically fixed",
    )
    correction_type: Optional[CorrectionType] = Field(
        None,
        description="Suggested correction type if auto-correctable",
    )
    corrected_value: Optional[str] = Field(
        None,
        description="Suggested corrected coordinate string",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the suggested correction (0.0-1.0)",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is error, warning, or info."""
        v = v.lower().strip()
        if v not in ("error", "warning", "info"):
            raise ValueError(
                f"severity must be 'error', 'warning', or 'info', got '{v}'"
            )
        return v

class ValidationResult(GreenLangBase):
    """Complete validation result for a single coordinate.

    Aggregates all validation checks (range, format, precision,
    plausibility) into a single result with overall pass/fail status,
    detected errors, applied auto-corrections, and provenance hash.

    Attributes:
        is_valid: Overall validation result (True if no errors).
        coordinate: Normalized coordinate after validation.
        errors: List of validation errors detected.
        warnings: List of warning messages.
        auto_corrections: List of auto-corrections applied.
        provenance_hash: SHA-256 hash of the validation inputs and
            outputs for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    is_valid: bool = Field(
        ...,
        description="Overall validation result (True if no errors)",
    )
    coordinate: Optional[NormalizedCoordinate] = Field(
        None,
        description="Normalized coordinate after validation",
    )
    errors: List[CoordinateValidationError] = Field(
        default_factory=list,
        description="Validation errors detected",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages",
    )
    auto_corrections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Auto-corrections applied",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class PrecisionResult(GreenLangBase):
    """Result of coordinate precision analysis.

    Provides detailed assessment of coordinate precision including
    decimal place counts, ground resolution, EUDR adequacy, and
    truncation/rounding detection.

    Attributes:
        decimal_places_lat: Number of significant decimal places in latitude.
        decimal_places_lon: Number of significant decimal places in longitude.
        ground_resolution_lat_m: Ground resolution of latitude in meters.
        ground_resolution_lon_m: Ground resolution of longitude in meters.
        level: Precision level classification.
        eudr_adequate: Whether precision meets EUDR Article 9 requirements.
        is_truncated: Whether the coordinate appears to be truncated.
        is_artificially_rounded: Whether the coordinate appears to be
            artificially rounded to whole degrees or similar.
    """

    model_config = ConfigDict(from_attributes=True)

    decimal_places_lat: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Significant decimal places in latitude",
    )
    decimal_places_lon: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Significant decimal places in longitude",
    )
    ground_resolution_lat_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Ground resolution of latitude in meters",
    )
    ground_resolution_lon_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Ground resolution of longitude in meters",
    )
    level: PrecisionLevel = Field(
        default=PrecisionLevel.INADEQUATE,
        description="Precision level classification",
    )
    eudr_adequate: bool = Field(
        default=False,
        description="Whether precision meets EUDR requirements",
    )
    is_truncated: bool = Field(
        default=False,
        description="Whether coordinate appears truncated",
    )
    is_artificially_rounded: bool = Field(
        default=False,
        description="Whether coordinate appears artificially rounded",
    )

class DatumTransformResult(GreenLangBase):
    """Result of a geodetic datum transformation.

    Contains the transformed coordinate, transformation parameters used,
    and quality metrics including displacement distance and residual error.

    Attributes:
        source_datum: Original geodetic datum.
        target_datum: Target geodetic datum (typically WGS84).
        source_lat: Original latitude in source datum.
        source_lon: Original longitude in source datum.
        target_lat: Transformed latitude in target datum.
        target_lon: Transformed longitude in target datum.
        displacement_m: Distance the coordinate moved in meters.
        method: Transformation method used (helmert_7p, molodensky_3p).
        residual_error_m: Estimated residual error in meters.
        parameters_used: Transformation parameters applied.
    """

    model_config = ConfigDict(from_attributes=True)

    source_datum: GeodeticDatum = Field(
        ...,
        description="Original geodetic datum",
    )
    target_datum: GeodeticDatum = Field(
        default=GeodeticDatum.WGS84,
        description="Target geodetic datum",
    )
    source_lat: float = Field(
        ...,
        description="Original latitude in source datum",
    )
    source_lon: float = Field(
        ...,
        description="Original longitude in source datum",
    )
    target_lat: float = Field(
        ...,
        description="Transformed latitude in target datum",
    )
    target_lon: float = Field(
        ...,
        description="Transformed longitude in target datum",
    )
    displacement_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Displacement from transformation in meters",
    )
    method: str = Field(
        default="helmert_7p",
        description="Transformation method (helmert_7p, molodensky_3p)",
    )
    residual_error_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated residual error in meters",
    )
    parameters_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Transformation parameters applied",
    )

# =============================================================================
# Result Models
# =============================================================================

class PlausibilityResult(GreenLangBase):
    """Result of spatial plausibility checking for a coordinate.

    Contains the outcomes of all plausibility checks including
    land/ocean, country match, commodity suitability, elevation,
    urban area, and protected area assessments.

    Attributes:
        is_on_land: Whether the coordinate falls on land.
        detected_country_iso: ISO country code detected at coordinate.
        country_match: Whether detected country matches declared country.
        commodity_plausible: Whether commodity production is plausible.
        elevation_plausible: Whether elevation is within expected range.
        is_urban: Whether the coordinate is in an urban area.
        is_protected_area: Whether the coordinate is in a protected area.
        land_use: Detected land use context at the coordinate.
        distance_to_coast_km: Distance to nearest coastline in km.
        details: Additional plausibility check details.
    """

    model_config = ConfigDict(from_attributes=True)

    is_on_land: bool = Field(
        default=True,
        description="Whether coordinate falls on land",
    )
    detected_country_iso: Optional[str] = Field(
        None,
        description="ISO country code detected at coordinate",
    )
    country_match: bool = Field(
        default=True,
        description="Whether detected country matches declared",
    )
    commodity_plausible: bool = Field(
        default=True,
        description="Whether commodity production is plausible",
    )
    elevation_plausible: bool = Field(
        default=True,
        description="Whether elevation is within expected range",
    )
    is_urban: bool = Field(
        default=False,
        description="Whether coordinate is in an urban area",
    )
    is_protected_area: bool = Field(
        default=False,
        description="Whether coordinate is in a protected area",
    )
    land_use: LandUseContext = Field(
        default=LandUseContext.UNKNOWN,
        description="Detected land use context at coordinate",
    )
    distance_to_coast_km: Optional[float] = Field(
        None,
        ge=0.0,
        description="Distance to nearest coastline in km",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional plausibility check details",
    )

class ReverseGeocodeResult(GreenLangBase):
    """Result of reverse geocoding a coordinate to a location.

    Contains geographic context information including country,
    administrative region, nearest place name, land use, coast
    distance, commodity zone, and elevation.

    Attributes:
        country_iso: ISO 3166-1 alpha-2 country code.
        country_name: Full country name in English.
        admin_region: Administrative region or province name.
        nearest_place: Name of the nearest populated place.
        land_use: Detected land use context.
        distance_to_coast_km: Distance to nearest coastline in km.
        commodity_zone: Name of the commodity production zone if known.
        elevation_m: Elevation at the coordinate in meters.
        elevation_source: Source of the elevation data.
    """

    model_config = ConfigDict(from_attributes=True)

    country_iso: Optional[str] = Field(
        None,
        description="ISO 3166-1 alpha-2 country code",
    )
    country_name: Optional[str] = Field(
        None,
        description="Full country name in English",
    )
    admin_region: Optional[str] = Field(
        None,
        description="Administrative region or province",
    )
    nearest_place: Optional[str] = Field(
        None,
        description="Name of the nearest populated place",
    )
    land_use: LandUseContext = Field(
        default=LandUseContext.UNKNOWN,
        description="Detected land use context",
    )
    distance_to_coast_km: Optional[float] = Field(
        None,
        ge=0.0,
        description="Distance to nearest coastline in km",
    )
    commodity_zone: Optional[str] = Field(
        None,
        description="Commodity production zone name",
    )
    elevation_m: Optional[float] = Field(
        None,
        description="Elevation in meters",
    )
    elevation_source: Optional[ElevationSource] = Field(
        None,
        description="Source of elevation data",
    )

class AccuracyScore(GreenLangBase):
    """Composite GPS coordinate accuracy score.

    Provides a weighted composite score from 0-100 based on four
    quality dimensions: precision, plausibility, consistency, and
    source quality. Each dimension is scored independently and
    combined using configurable weights.

    Attributes:
        overall_score: Composite accuracy score (0.0-100.0).
        tier: Accuracy tier classification based on overall_score.
        precision_score: Precision component score (0.0-100.0).
        plausibility_score: Plausibility component score (0.0-100.0).
        consistency_score: Consistency component score (0.0-100.0).
        source_score: Source quality component score (0.0-100.0).
        confidence_interval_m: Estimated confidence interval in meters.
        explanations: Breakdown of scoring rationale per component.
    """

    model_config = ConfigDict(from_attributes=True)

    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Composite accuracy score (0-100)",
    )
    tier: AccuracyTier = Field(
        default=AccuracyTier.UNVERIFIED,
        description="Accuracy tier classification",
    )
    precision_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Precision component score (0-100)",
    )
    plausibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Plausibility component score (0-100)",
    )
    consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Consistency component score (0-100)",
    )
    source_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Source quality component score (0-100)",
    )
    confidence_interval_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated confidence interval in meters",
    )
    explanations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scoring rationale per component",
    )

class ComplianceCertificate(GreenLangBase):
    """EUDR compliance certificate for a validated GPS coordinate.

    Represents a formal certification that a GPS coordinate meets
    EUDR Article 9 geolocation data quality requirements. Includes
    the validated coordinate, accuracy score, validation result,
    validity period, and provenance hash.

    Attributes:
        cert_id: Unique certificate identifier (UUID).
        coordinate: Validated and normalized coordinate.
        status: EUDR compliance status.
        accuracy_score: Composite accuracy score assessment.
        validation_result: Full validation result details.
        issued_at: UTC timestamp when certificate was issued.
        valid_until: UTC timestamp when certificate expires.
        provenance_hash: SHA-256 hash of the complete certificate
            for tamper detection.
    """

    model_config = ConfigDict(from_attributes=True)

    cert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique certificate identifier",
    )
    coordinate: NormalizedCoordinate = Field(
        ...,
        description="Validated and normalized coordinate",
    )
    status: ComplianceStatus = Field(
        default=ComplianceStatus.INSUFFICIENT_DATA,
        description="EUDR compliance status",
    )
    accuracy_score: AccuracyScore = Field(
        ...,
        description="Composite accuracy score assessment",
    )
    validation_result: ValidationResult = Field(
        ...,
        description="Full validation result details",
    )
    issued_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when certificate was issued",
    )
    valid_until: Optional[datetime] = Field(
        None,
        description="UTC timestamp when certificate expires",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for tamper detection",
    )

# =============================================================================
# Request Models
# =============================================================================

class ParseCoordinateRequest(GreenLangBase):
    """Request body for parsing a raw coordinate string.

    Attributes:
        raw_input: Raw coordinate string to parse.
        format_hint: Optional format hint to guide parsing.
        datum_hint: Optional datum hint for the input coordinate.
    """
    raw_input: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Raw coordinate string to parse",
    )
    format_hint: Optional[CoordinateFormat] = Field(
        None,
        description="Optional format hint",
    )
    datum_hint: Optional[GeodeticDatum] = Field(
        None,
        description="Optional datum hint",
    )

class ValidateCoordinateRequest(GreenLangBase):
    """Request body for validating a single coordinate.

    Attributes:
        raw_coordinate: Raw coordinate input to validate.
        enable_auto_correction: Whether to apply auto-corrections.
        correction_confidence_threshold: Minimum confidence for
            auto-correction (0.0-1.0).
    """
    raw_coordinate: RawCoordinate = Field(
        ...,
        description="Raw coordinate input to validate",
    )
    enable_auto_correction: bool = Field(
        default=False,
        description="Whether to apply auto-corrections",
    )
    correction_confidence_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for auto-correction",
    )

class TransformDatumRequest(GreenLangBase):
    """Request body for transforming a coordinate to a different datum.

    Attributes:
        latitude: Latitude in the source datum.
        longitude: Longitude in the source datum.
        source_datum: Source geodetic datum.
        target_datum: Target geodetic datum.
        method: Transformation method preference.
    """
    latitude: float = Field(
        ...,
        description="Latitude in source datum",
    )
    longitude: float = Field(
        ...,
        description="Longitude in source datum",
    )
    source_datum: GeodeticDatum = Field(
        ...,
        description="Source geodetic datum",
    )
    target_datum: GeodeticDatum = Field(
        default=GeodeticDatum.WGS84,
        description="Target geodetic datum",
    )
    method: str = Field(
        default="auto",
        description="Transformation method: auto, helmert_7p, molodensky_3p",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate transformation method."""
        v = v.lower().strip()
        if v not in ("auto", "helmert_7p", "molodensky_3p"):
            raise ValueError(
                f"method must be 'auto', 'helmert_7p', or 'molodensky_3p', "
                f"got '{v}'"
            )
        return v

class AnalyzePrecisionRequest(GreenLangBase):
    """Request body for analyzing coordinate precision.

    Attributes:
        latitude: Latitude value to analyze.
        longitude: Longitude value to analyze.
        original_format: Original coordinate format string.
    """
    latitude: float = Field(
        ...,
        description="Latitude value to analyze",
    )
    longitude: float = Field(
        ...,
        description="Longitude value to analyze",
    )
    original_format: Optional[str] = Field(
        None,
        description="Original coordinate format string",
    )

class CheckPlausibilityRequest(GreenLangBase):
    """Request body for checking spatial plausibility.

    Attributes:
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        declared_country_iso: Declared ISO country code.
        commodity: Declared commodity for context checking.
        enable_ocean_check: Whether to check for ocean location.
        enable_country_check: Whether to verify country match.
        enable_commodity_check: Whether to check commodity plausibility.
        enable_elevation_check: Whether to check elevation plausibility.
    """
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees",
    )
    declared_country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Declared ISO 3166-1 alpha-2 country code",
    )
    commodity: Optional[str] = Field(
        None,
        description="Declared commodity for context checking",
    )
    enable_ocean_check: bool = Field(
        default=True,
        description="Whether to check for ocean location",
    )
    enable_country_check: bool = Field(
        default=True,
        description="Whether to verify country match",
    )
    enable_commodity_check: bool = Field(
        default=True,
        description="Whether to check commodity plausibility",
    )
    enable_elevation_check: bool = Field(
        default=True,
        description="Whether to check elevation plausibility",
    )

    @field_validator("declared_country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code if provided."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "declared_country_iso must be a two-letter "
                "ISO 3166-1 alpha-2 code"
            )
        return v

class ReverseGeocodeRequest(GreenLangBase):
    """Request body for reverse geocoding a coordinate.

    Attributes:
        latitude: WGS84 latitude in decimal degrees.
        longitude: WGS84 longitude in decimal degrees.
        include_elevation: Whether to include elevation data.
        include_land_use: Whether to include land use classification.
    """
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude in decimal degrees",
    )
    include_elevation: bool = Field(
        default=True,
        description="Whether to include elevation data",
    )
    include_land_use: bool = Field(
        default=True,
        description="Whether to include land use classification",
    )

class AssessAccuracyRequest(GreenLangBase):
    """Request body for assessing GPS coordinate accuracy.

    Attributes:
        coordinate: Normalized coordinate to assess.
        precision_result: Precision analysis result.
        plausibility_result: Plausibility check result.
        source_type: How the coordinate was originally captured.
        weights: Optional custom quality dimension weights.
    """
    coordinate: NormalizedCoordinate = Field(
        ...,
        description="Normalized coordinate to assess",
    )
    precision_result: Optional[PrecisionResult] = Field(
        None,
        description="Precision analysis result",
    )
    plausibility_result: Optional[PlausibilityResult] = Field(
        None,
        description="Plausibility check result",
    )
    source_type: SourceType = Field(
        default=SourceType.UNKNOWN,
        description="How the coordinate was captured",
    )
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional custom quality dimension weights",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> AssessAccuracyRequest:
        """Validate that custom weights sum to 1.0 if provided."""
        if self.weights is not None:
            expected_keys = {"precision", "plausibility", "consistency", "source"}
            actual_keys = set(self.weights.keys())
            if actual_keys != expected_keys:
                raise ValueError(
                    f"weights must have keys {sorted(expected_keys)}, "
                    f"got {sorted(actual_keys)}"
                )
            weight_sum = sum(self.weights.values())
            if abs(weight_sum - 1.0) > 0.001:
                raise ValueError(
                    f"weights must sum to 1.0, got {weight_sum:.4f}"
                )
        return self

class GenerateReportRequest(GreenLangBase):
    """Request body for generating a GPS validation compliance report.

    Attributes:
        operator_id: Operator ID for the report.
        commodity: Optional commodity filter.
        country_code: Optional country filter.
        report_format: Desired output format.
        include_details: Whether to include per-coordinate details.
        include_certificates: Whether to include compliance certificates.
    """
    operator_id: str = Field(
        ...,
        min_length=1,
        description="Operator ID for the report",
    )
    commodity: Optional[str] = Field(
        None,
        description="Optional commodity filter",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Optional country filter",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Desired output format",
    )
    include_details: bool = Field(
        default=True,
        description="Whether to include per-coordinate details",
    )
    include_certificates: bool = Field(
        default=False,
        description="Whether to include compliance certificates",
    )

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v.strip()

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize country code if provided."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

# =============================================================================
# Response Models
# =============================================================================

class ParseCoordinateResponse(GreenLangBase):
    """Response for a coordinate parsing operation.

    Attributes:
        success: Whether parsing succeeded.
        parsed: Parsed coordinate result if successful.
        error: Error message if parsing failed.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        default=True,
        description="Whether parsing succeeded",
    )
    parsed: Optional[ParsedCoordinate] = Field(
        None,
        description="Parsed coordinate if successful",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if parsing failed",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class ValidateCoordinateResponse(GreenLangBase):
    """Response for a coordinate validation operation.

    Attributes:
        validation_id: Unique identifier for this validation run.
        result: Full validation result.
        accuracy_score: Accuracy score if assessment was performed.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    validation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this validation run",
    )
    result: ValidationResult = Field(
        ...,
        description="Full validation result",
    )
    accuracy_score: Optional[AccuracyScore] = Field(
        None,
        description="Accuracy score if assessment was performed",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class TransformDatumResponse(GreenLangBase):
    """Response for a datum transformation operation.

    Attributes:
        success: Whether transformation succeeded.
        result: Datum transformation result if successful.
        error: Error message if transformation failed.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        default=True,
        description="Whether transformation succeeded",
    )
    result: Optional[DatumTransformResult] = Field(
        None,
        description="Transformation result if successful",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if transformation failed",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class AnalyzePrecisionResponse(GreenLangBase):
    """Response for a precision analysis operation.

    Attributes:
        result: Precision analysis result.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    result: PrecisionResult = Field(
        ...,
        description="Precision analysis result",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class CheckPlausibilityResponse(GreenLangBase):
    """Response for a plausibility check operation.

    Attributes:
        result: Plausibility check result.
        geocode: Optional reverse geocode result.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    result: PlausibilityResult = Field(
        ...,
        description="Plausibility check result",
    )
    geocode: Optional[ReverseGeocodeResult] = Field(
        None,
        description="Reverse geocode result if available",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class ReverseGeocodeResponse(GreenLangBase):
    """Response for a reverse geocoding operation.

    Attributes:
        success: Whether reverse geocoding succeeded.
        result: Reverse geocode result if successful.
        error: Error message if geocoding failed.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(
        default=True,
        description="Whether reverse geocoding succeeded",
    )
    result: Optional[ReverseGeocodeResult] = Field(
        None,
        description="Reverse geocode result if successful",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if geocoding failed",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class AssessAccuracyResponse(GreenLangBase):
    """Response for an accuracy assessment operation.

    Attributes:
        score: Accuracy score assessment.
        certificate: Optional compliance certificate if score meets threshold.
        processing_time_ms: Processing time in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """

    model_config = ConfigDict(from_attributes=True)

    score: AccuracyScore = Field(
        ...,
        description="Accuracy score assessment",
    )
    certificate: Optional[ComplianceCertificate] = Field(
        None,
        description="Compliance certificate if score meets threshold",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail",
    )

class BatchValidationResult(GreenLangBase):
    """Complete result of a batch coordinate validation job.

    Provides aggregate statistics and individual coordinate validation
    results for a completed or partially completed batch job.

    Attributes:
        batch_id: Unique identifier for the batch job.
        status: Current batch processing status.
        total: Total number of coordinates in the batch.
        valid: Number of coordinates that passed validation.
        invalid: Number of coordinates that failed validation.
        warnings: Number of coordinates with warnings.
        auto_corrected: Number of coordinates that were auto-corrected.
        results: List of individual validation results.
        summary: Aggregate summary statistics.
        started_at: UTC timestamp when batch started.
        completed_at: UTC timestamp when batch completed.
        duration_seconds: Total elapsed time in seconds.
        provenance_hash: SHA-256 hash of the complete batch result.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the batch job",
    )
    status: BatchStatus = Field(
        default=BatchStatus.PENDING,
        description="Current batch processing status",
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total coordinates in the batch",
    )
    valid: int = Field(
        default=0,
        ge=0,
        description="Coordinates that passed validation",
    )
    invalid: int = Field(
        default=0,
        ge=0,
        description="Coordinates that failed validation",
    )
    warnings: int = Field(
        default=0,
        ge=0,
        description="Coordinates with warnings",
    )
    auto_corrected: int = Field(
        default=0,
        ge=0,
        description="Coordinates that were auto-corrected",
    )
    results: List[ValidationResult] = Field(
        default_factory=list,
        description="Individual validation results",
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate summary statistics",
    )
    started_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when batch started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when batch completed",
    )
    duration_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Total elapsed time in seconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the complete batch result",
    )

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "VERSION",
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "DEFAULT_QUALITY_WEIGHTS",
    "ACCURACY_TIER_THRESHOLDS",
    "EUDR_MIN_DECIMAL_PLACES",
    "WGS84_SEMI_MAJOR_AXIS",
    "WGS84_INV_FLATTENING",
    "WGS84_FLATTENING",
    # Re-export for convenience
    "EUDRCommodity",
    # Enumerations
    "CoordinateFormat",
    "GeodeticDatum",
    "PrecisionLevel",
    "ValidationErrorType",
    "PlausibilityCheckResult",
    "AccuracyTier",
    "CorrectionType",
    "ReportFormat",
    "ComplianceStatus",
    "SourceType",
    "LandUseContext",
    "BatchStatus",
    "ElevationSource",
    "HemisphereIndicator",
    # Core models
    "RawCoordinate",
    "ParsedCoordinate",
    "NormalizedCoordinate",
    "CoordinateValidationError",
    "ValidationResult",
    "PrecisionResult",
    "DatumTransformResult",
    # Result models
    "PlausibilityResult",
    "ReverseGeocodeResult",
    "AccuracyScore",
    "ComplianceCertificate",
    # Request models
    "ParseCoordinateRequest",
    "ValidateCoordinateRequest",
    "TransformDatumRequest",
    "AnalyzePrecisionRequest",
    "CheckPlausibilityRequest",
    "ReverseGeocodeRequest",
    "AssessAccuracyRequest",
    "GenerateReportRequest",
    # Response models
    "ParseCoordinateResponse",
    "ValidateCoordinateResponse",
    "TransformDatumResponse",
    "AnalyzePrecisionResponse",
    "CheckPlausibilityResponse",
    "ReverseGeocodeResponse",
    "AssessAccuracyResponse",
    "BatchValidationResult",
]
