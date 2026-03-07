# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-007 GPS Coordinate Validator

Pydantic v2 request/response models for the GPS Coordinate Validator REST API.
Covers coordinate parsing, validation, plausibility analysis, accuracy
assessment, compliance reporting, datum transformation, reverse geocoding,
and batch processing operations.

Core domain models are imported from the main models module; this file
defines API-level request wrappers, response envelopes, and batch schemas.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EUDR_COMMODITIES: List[str] = [
    "cattle",
    "cocoa",
    "coffee",
    "oil_palm",
    "rubber",
    "soy",
    "wood",
]

SUPPORTED_DATUMS: List[str] = [
    "wgs84", "nad27", "nad83", "ed50", "etrs89", "osgb36", "tokyo",
    "indian_1975", "pulkovo_1942", "agd66", "agd84", "gda94", "gda2020",
    "sad69", "sirgas2000", "hartebeesthoek94", "arc1960", "cape", "adindan",
    "minna", "camacupa", "schwarzeck", "massawa", "merchich", "egypt_1907",
    "lome", "accra", "jakarta", "kalianpur", "kertau", "luzon_1911",
    "timbalai_1948", "nzgd49",
]

VALID_SOURCE_TYPES: List[str] = [
    "gnss_survey", "rtk_gps", "mobile_gps", "handheld_gps",
    "manual_entry", "digitized_map", "geocoded", "satellite_derived",
    "unknown",
]


# =============================================================================
# Pagination
# =============================================================================


class PaginatedMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""

    items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Page of result items"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


class PaginationParams(BaseModel):
    """Standard pagination query parameters."""

    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


# =============================================================================
# Response Wrappers
# =============================================================================


class ApiResponse(BaseModel):
    """Standard API success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


# =============================================================================
# Coordinate Input Schemas
# =============================================================================


class RawCoordinateSchema(BaseModel):
    """Raw coordinate input before parsing.

    Accepts any string-format GPS coordinate (DD, DMS, DDM, UTM, MGRS) with
    optional hints for format detection, datum context, and commodity scope.
    """

    input: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw coordinate string (e.g., '5d36\\'13.4\"N 0d11\\'13.1\"W')",
    )
    format_hint: Optional[str] = Field(
        None,
        description=(
            "Hint for coordinate format: decimal_degrees, dms, ddm, "
            "utm, mgrs, signed_dd, dd_suffix"
        ),
    )
    datum: Optional[str] = Field(
        None,
        description="Source geodetic datum if known (e.g., wgs84, nad27, ed50)",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="EUDR commodity for plausibility checks",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    source_type: Optional[str] = Field(
        None,
        description="GPS data source type (e.g., mobile_gps, gnss_survey)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "input": "5d36'13.4\"N 0d11'13.1\"W",
                    "format_hint": "dms",
                    "datum": "wgs84",
                    "commodity": "cocoa",
                    "country_iso": "GH",
                    "source_type": "mobile_gps",
                }
            ]
        },
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code to uppercase."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("format_hint")
    @classmethod
    def validate_format_hint(cls, v: Optional[str]) -> Optional[str]:
        """Validate format hint is a supported format."""
        if v is None:
            return v
        v = v.lower().strip()
        allowed = {
            "decimal_degrees", "dms", "ddm", "utm", "mgrs",
            "signed_dd", "dd_suffix",
        }
        if v not in allowed:
            raise ValueError(
                f"format_hint must be one of {sorted(allowed)}, got '{v}'"
            )
        return v

    @field_validator("datum")
    @classmethod
    def validate_datum_field(cls, v: Optional[str]) -> Optional[str]:
        """Validate datum is supported."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in SUPPORTED_DATUMS:
            raise ValueError(
                f"datum '{v}' is not supported. "
                f"Supported datums: {SUPPORTED_DATUMS[:10]}... ({len(SUPPORTED_DATUMS)} total)"
            )
        return v


class CoordinatePairSchema(BaseModel):
    """Pre-parsed coordinate pair with optional context.

    Standard WGS84 coordinate pair used throughout the validation,
    plausibility, and assessment endpoints.
    """

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    altitude: Optional[float] = Field(
        None,
        description="Altitude in metres above ellipsoid",
    )
    datum: str = Field(
        default="WGS84",
        description="Geodetic datum of the coordinate",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="EUDR commodity for context",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    source_type: str = Field(
        default="unknown",
        description="GPS data source type",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.603716,
                    "longitude": -0.186964,
                    "altitude": 215.0,
                    "datum": "WGS84",
                    "commodity": "cocoa",
                    "country_iso": "GH",
                    "source_type": "mobile_gps",
                }
            ]
        },
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code to uppercase."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


# =============================================================================
# Parse Schemas
# =============================================================================


class ParseRequestSchema(BaseModel):
    """Request to parse a raw coordinate string into decimal degrees.

    Supports DD, DMS, DDM, UTM, and MGRS formats with automatic detection
    or explicit format hint.
    """

    input: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw coordinate string to parse",
    )
    format_hint: Optional[str] = Field(
        None,
        description="Optional format hint to guide parser",
    )
    datum: Optional[str] = Field(
        None,
        description="Source datum for datum-aware parsing",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "input": "5d36'13.4\"N 0d11'13.1\"W",
                    "format_hint": "dms",
                    "datum": "wgs84",
                }
            ]
        },
    )

    @field_validator("format_hint")
    @classmethod
    def validate_format_hint(cls, v: Optional[str]) -> Optional[str]:
        """Validate format hint."""
        if v is None:
            return v
        v = v.lower().strip()
        allowed = {
            "decimal_degrees", "dms", "ddm", "utm", "mgrs",
            "signed_dd", "dd_suffix",
        }
        if v not in allowed:
            raise ValueError(
                f"format_hint must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


class ParseResponseSchema(BaseModel):
    """Response from coordinate parsing."""

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Parsed latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Parsed longitude in decimal degrees",
    )
    detected_format: str = Field(
        ...,
        description="Detected coordinate format",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Format detection confidence",
    )
    datum: str = Field(
        default="wgs84",
        description="Datum of the parsed coordinate",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Parse warnings or notes",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    parsed_at: datetime = Field(
        default_factory=_utcnow,
        description="Parse timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchParseRequestSchema(BaseModel):
    """Request to parse multiple raw coordinate strings in batch."""

    coordinates: List[RawCoordinateSchema] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of raw coordinates to parse (max 10,000)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "coordinates": [
                        {"input": "5.603716, -0.186964", "commodity": "cocoa"},
                        {"input": "6d41'18\"N 1d37'28\"W", "format_hint": "dms"},
                    ]
                }
            ]
        },
    )


class BatchParseResponseSchema(BaseModel):
    """Response from batch coordinate parsing."""

    total: int = Field(..., ge=0, description="Total coordinates submitted")
    successful: int = Field(..., ge=0, description="Successfully parsed")
    failed: int = Field(..., ge=0, description="Failed to parse")
    results: List[ParseResponseSchema] = Field(
        default_factory=list,
        description="Per-coordinate parse results",
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parse errors with index and reason",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 batch provenance hash",
    )
    parsed_at: datetime = Field(
        default_factory=_utcnow,
        description="Batch parse timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class FormatDetectionResponseSchema(BaseModel):
    """Response from coordinate format detection."""

    detected_format: str = Field(
        ...,
        description="Most likely coordinate format",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence for primary format",
    )
    alternatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Alternative format candidates with confidence scores",
    )
    input_analyzed: str = Field(
        default="",
        description="The input string that was analyzed",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Detection timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class NormalizeResponseSchema(BaseModel):
    """Response from coordinate normalization to WGS84."""

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Normalized WGS84 latitude",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Normalized WGS84 longitude",
    )
    original_input: str = Field(
        default="",
        description="Original input string",
    )
    source_datum: str = Field(
        default="wgs84",
        description="Source datum before normalization",
    )
    target_datum: str = Field(
        default="wgs84",
        description="Target datum (always WGS84)",
    )
    displacement_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Displacement from datum transformation in metres",
    )
    format_detected: str = Field(
        default="unknown",
        description="Detected input format",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    normalized_at: datetime = Field(
        default_factory=_utcnow,
        description="Normalization timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Validation Schemas
# =============================================================================


class ValidateRequestSchema(BaseModel):
    """Request to validate a coordinate pair."""

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code for boundary check",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="EUDR commodity for plausibility context",
    )
    source_type: Optional[str] = Field(
        None,
        description="GPS data source type",
    )
    altitude: Optional[float] = Field(
        None,
        description="Altitude in metres for elevation plausibility",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.603716,
                    "longitude": -0.186964,
                    "country_iso": "GH",
                    "commodity": "cocoa",
                    "source_type": "mobile_gps",
                    "altitude": 215.0,
                }
            ]
        },
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code to uppercase."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate EUDR commodity."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {EUDR_COMMODITIES}, got '{v}'"
            )
        return v


class ValidationErrorSchema(BaseModel):
    """A single validation error or warning."""

    error_type: str = Field(
        ...,
        description="Error classification (e.g., out_of_range, swapped, null_island)",
    )
    description: str = Field(
        ...,
        description="Human-readable error description",
    )
    severity: str = Field(
        default="error",
        description="Severity: critical, error, warning, info",
    )
    auto_correctable: bool = Field(
        default=False,
        description="Whether this error can be auto-corrected",
    )
    correction: Optional[Dict[str, Any]] = Field(
        None,
        description="Suggested correction values if auto-correctable",
    )

    model_config = ConfigDict(from_attributes=True)


class ValidationResponseSchema(BaseModel):
    """Response from coordinate validation."""

    is_valid: bool = Field(
        ...,
        description="Overall validation pass/fail",
    )
    errors: List[ValidationErrorSchema] = Field(
        default_factory=list,
        description="List of validation errors detected",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings",
    )
    auto_corrections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Applied or suggested auto-corrections",
    )
    normalized: CoordinatePairSchema = Field(
        ...,
        description="Normalized coordinate (with corrections applied if any)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    validated_at: datetime = Field(
        default_factory=_utcnow,
        description="Validation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchValidateRequestSchema(BaseModel):
    """Request to validate multiple coordinate pairs."""

    coordinates: List[CoordinatePairSchema] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of coordinates to validate (max 10,000)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "coordinates": [
                        {"latitude": 5.603716, "longitude": -0.186964, "commodity": "cocoa"},
                        {"latitude": 6.688500, "longitude": -1.624400, "commodity": "cocoa"},
                    ]
                }
            ]
        },
    )


class BatchValidateResponseSchema(BaseModel):
    """Response from batch coordinate validation."""

    total: int = Field(..., ge=0, description="Total coordinates submitted")
    valid: int = Field(..., ge=0, description="Coordinates that passed")
    invalid: int = Field(..., ge=0, description="Coordinates that failed")
    warnings: int = Field(
        default=0,
        ge=0,
        description="Coordinates with warnings only",
    )
    auto_corrected: int = Field(
        default=0,
        ge=0,
        description="Coordinates auto-corrected",
    )
    results: List[ValidationResponseSchema] = Field(
        default_factory=list,
        description="Per-coordinate validation results",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 batch provenance hash",
    )
    validated_at: datetime = Field(
        default_factory=_utcnow,
        description="Batch validation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class RangeCheckResponseSchema(BaseModel):
    """Response from coordinate range check."""

    latitude: float = Field(..., description="Checked latitude")
    longitude: float = Field(..., description="Checked longitude")
    latitude_in_range: bool = Field(
        ...,
        description="Whether latitude is within [-90, 90]",
    )
    longitude_in_range: bool = Field(
        ...,
        description="Whether longitude is within [-180, 180]",
    )
    is_null_island: bool = Field(
        default=False,
        description="Whether coordinate is at or near (0, 0)",
    )
    is_nan: bool = Field(
        default=False,
        description="Whether any value is NaN",
    )
    is_boundary: bool = Field(
        default=False,
        description="Whether coordinate is exactly at a boundary value",
    )
    details: List[str] = Field(
        default_factory=list,
        description="Diagnostic details",
    )

    model_config = ConfigDict(from_attributes=True)


class SwapDetectionRequestSchema(BaseModel):
    """Request to detect swapped latitude/longitude."""

    latitude: float = Field(
        ...,
        description="Latitude value to check for swap",
    )
    longitude: float = Field(
        ...,
        description="Longitude value to check for swap",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Expected country for swap validation",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": -0.186964,
                    "longitude": 5.603716,
                    "country_iso": "GH",
                }
            ]
        },
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code to uppercase."""
        if v is None:
            return v
        return v.upper().strip()


class SwapDetectionResponseSchema(BaseModel):
    """Response from swap detection analysis."""

    is_swapped: bool = Field(
        ...,
        description="Whether lat/lon appear to be swapped",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in swap detection",
    )
    corrected: Optional[CoordinatePairSchema] = Field(
        None,
        description="Corrected coordinate if swap was detected",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of swap detection reasoning",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Detection timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class DuplicateDetectionRequestSchema(BaseModel):
    """Request to detect duplicate coordinates in a set."""

    coordinates: List[CoordinatePairSchema] = Field(
        ...,
        min_length=2,
        max_length=10000,
        description="Coordinates to check for duplicates",
    )
    threshold_m: float = Field(
        default=1.0,
        ge=0.0,
        le=10000.0,
        description="Distance threshold in metres for near-duplicate detection",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "coordinates": [
                        {"latitude": 5.603716, "longitude": -0.186964},
                        {"latitude": 5.603716, "longitude": -0.186964},
                        {"latitude": 5.603720, "longitude": -0.186960},
                    ],
                    "threshold_m": 1.0,
                }
            ]
        },
    )


class DuplicateDetectionResponseSchema(BaseModel):
    """Response from duplicate detection analysis."""

    total_coordinates: int = Field(
        ...,
        ge=0,
        description="Total coordinates analyzed",
    )
    exact_duplicates: int = Field(
        default=0,
        ge=0,
        description="Number of exact duplicate pairs",
    )
    near_duplicates: int = Field(
        default=0,
        ge=0,
        description="Number of near-duplicate pairs",
    )
    duplicate_pairs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Duplicate pair details (indices, distance)",
    )
    near_duplicate_pairs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Near-duplicate pair details (indices, distance_m)",
    )
    threshold_m: float = Field(
        default=1.0,
        description="Distance threshold used",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Detection timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Plausibility Schemas
# =============================================================================


class PlausibilityRequestSchema(BaseModel):
    """Request for full plausibility analysis of a coordinate."""

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="EUDR commodity for plausibility check",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Declared country ISO code",
    )
    altitude: Optional[float] = Field(
        None,
        description="Altitude in metres for elevation check",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.603716,
                    "longitude": -0.186964,
                    "commodity": "cocoa",
                    "country_iso": "GH",
                    "altitude": 215.0,
                }
            ]
        },
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code to uppercase."""
        if v is None:
            return v
        return v.upper().strip()

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate EUDR commodity."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {EUDR_COMMODITIES}, got '{v}'"
            )
        return v


class PlausibilityResponseSchema(BaseModel):
    """Response from full plausibility analysis."""

    is_on_land: bool = Field(
        ...,
        description="Whether coordinate falls on land",
    )
    detected_country: Optional[str] = Field(
        None,
        description="ISO code of detected country",
    )
    country_match: Optional[bool] = Field(
        None,
        description="Whether detected country matches declared country",
    )
    commodity_plausible: Optional[bool] = Field(
        None,
        description="Whether commodity is plausible at this location",
    )
    elevation_plausible: Optional[bool] = Field(
        None,
        description="Whether elevation is plausible for the commodity",
    )
    is_urban: bool = Field(
        default=False,
        description="Whether location is in an urban area",
    )
    is_protected: bool = Field(
        default=False,
        description="Whether location is in a protected area",
    )
    land_use: Optional[str] = Field(
        None,
        description="Detected land use type",
    )
    distance_to_coast_km: Optional[float] = Field(
        None,
        ge=0.0,
        description="Distance to nearest coastline in kilometres",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional plausibility analysis details",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    analyzed_at: datetime = Field(
        default_factory=_utcnow,
        description="Analysis timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class LandOceanResponseSchema(BaseModel):
    """Response from land/ocean detection."""

    is_on_land: bool = Field(
        ...,
        description="Whether coordinate is on land",
    )
    nearest_coast_km: float = Field(
        ...,
        ge=0.0,
        description="Distance to nearest coastline in kilometres",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence",
    )
    data_source: str = Field(
        default="internal",
        description="Data source for land/ocean determination",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Check timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class CountryResponseSchema(BaseModel):
    """Response from country detection and matching."""

    detected_iso: Optional[str] = Field(
        None,
        description="ISO 3166-1 alpha-2 code of detected country",
    )
    detected_name: Optional[str] = Field(
        None,
        description="Full name of detected country",
    )
    matches_declared: Optional[bool] = Field(
        None,
        description="Whether detected country matches declared country",
    )
    declared_iso: Optional[str] = Field(
        None,
        description="ISO code that was declared for comparison",
    )
    distance_to_border_km: Optional[float] = Field(
        None,
        ge=0.0,
        description="Distance to nearest international border in km",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Check timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class CommodityResponseSchema(BaseModel):
    """Response from commodity plausibility check."""

    is_plausible: bool = Field(
        ...,
        description="Whether commodity is plausible at this location",
    )
    reason: str = Field(
        default="",
        description="Explanation of plausibility assessment",
    )
    latitude_range: Optional[Dict[str, float]] = Field(
        None,
        description="Expected latitude range for commodity (min, max)",
    )
    elevation_range: Optional[Dict[str, float]] = Field(
        None,
        description="Expected elevation range for commodity (min_m, max_m)",
    )
    known_growing_regions: List[str] = Field(
        default_factory=list,
        description="Known growing regions for this commodity",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Check timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class ElevationResponseSchema(BaseModel):
    """Response from elevation plausibility check."""

    is_plausible: bool = Field(
        ...,
        description="Whether elevation is plausible for the commodity",
    )
    elevation_m: float = Field(
        ...,
        description="Estimated or declared elevation in metres",
    )
    commodity_range: Optional[Dict[str, float]] = Field(
        None,
        description="Expected elevation range for commodity (min_m, max_m)",
    )
    data_source: str = Field(
        default="srtm",
        description="Elevation data source (srtm, aster, declared)",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Elevation confidence",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Check timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Assessment Schemas
# =============================================================================


class AssessmentRequestSchema(BaseModel):
    """Request for full accuracy assessment of a coordinate."""

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees",
    )
    source_type: Optional[str] = Field(
        None,
        description="GPS data source type for source scoring",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="EUDR commodity for context",
    )
    country_iso: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Declared country ISO code",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.603716,
                    "longitude": -0.186964,
                    "source_type": "mobile_gps",
                    "commodity": "cocoa",
                    "country_iso": "GH",
                }
            ]
        },
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code to uppercase."""
        if v is None:
            return v
        return v.upper().strip()


class AccuracyScoreSchema(BaseModel):
    """Accuracy scoring breakdown for a coordinate."""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall accuracy score (0-100)",
    )
    tier: str = Field(
        ...,
        description="Quality tier: gold, silver, bronze, fail",
    )
    precision_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Precision sub-score",
    )
    plausibility_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Plausibility sub-score",
    )
    consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Consistency sub-score",
    )
    source_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Source reliability sub-score",
    )
    confidence_interval_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Confidence interval in metres",
    )
    explanations: List[str] = Field(
        default_factory=list,
        description="Score component explanations",
    )

    model_config = ConfigDict(from_attributes=True)


class AssessmentResponseSchema(BaseModel):
    """Response from full accuracy assessment."""

    assessment_id: str = Field(
        ...,
        description="Unique assessment identifier",
    )
    coordinate: CoordinatePairSchema = Field(
        ...,
        description="Assessed coordinate",
    )
    accuracy: AccuracyScoreSchema = Field(
        ...,
        description="Accuracy scoring breakdown",
    )
    validation: ValidationResponseSchema = Field(
        ...,
        description="Validation results",
    )
    precision: "PrecisionResponseSchema" = Field(
        ...,
        description="Precision analysis results",
    )
    plausibility: PlausibilityResponseSchema = Field(
        ...,
        description="Plausibility analysis results",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchAssessmentRequestSchema(BaseModel):
    """Request for batch accuracy assessment."""

    coordinates: List[AssessmentRequestSchema] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Coordinates to assess (max 5,000)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "coordinates": [
                        {
                            "latitude": 5.603716,
                            "longitude": -0.186964,
                            "commodity": "cocoa",
                        }
                    ]
                }
            ]
        },
    )


class BatchAssessmentResponseSchema(BaseModel):
    """Response from batch accuracy assessment."""

    total: int = Field(..., ge=0, description="Total coordinates assessed")
    results: List[AssessmentResponseSchema] = Field(
        default_factory=list,
        description="Per-coordinate assessment results",
    )
    summary: "BatchSummaryResponseSchema" = Field(
        ...,
        description="Aggregate assessment summary",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 batch provenance hash",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Batch assessment timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class PrecisionRequestSchema(BaseModel):
    """Request for precision analysis only."""

    latitude: float = Field(
        ...,
        description="Latitude value to analyze precision",
    )
    longitude: float = Field(
        ...,
        description="Longitude value to analyze precision",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"latitude": 5.603716, "longitude": -0.186964}
            ]
        },
    )


class PrecisionResponseSchema(BaseModel):
    """Response from precision analysis."""

    decimal_places_lat: int = Field(
        ...,
        ge=0,
        description="Decimal places in latitude",
    )
    decimal_places_lon: int = Field(
        ...,
        ge=0,
        description="Decimal places in longitude",
    )
    ground_resolution_lat_m: float = Field(
        ...,
        ge=0.0,
        description="Ground resolution in latitude direction (metres)",
    )
    ground_resolution_lon_m: float = Field(
        ...,
        ge=0.0,
        description="Ground resolution in longitude direction (metres)",
    )
    level: str = Field(
        ...,
        description="Precision level: survey_grade, high, moderate, low, inadequate",
    )
    eudr_adequate: bool = Field(
        ...,
        description="Whether precision meets EUDR requirements",
    )
    is_truncated: bool = Field(
        default=False,
        description="Whether coordinate appears truncated",
    )
    is_rounded: bool = Field(
        default=False,
        description="Whether coordinate appears artificially rounded",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    analyzed_at: datetime = Field(
        default_factory=_utcnow,
        description="Analysis timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Report Schemas
# =============================================================================


class ComplianceCertRequestSchema(BaseModel):
    """Request to generate a compliance certificate for a coordinate."""

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees",
    )
    commodity: str = Field(
        ...,
        max_length=50,
        description="EUDR commodity",
    )
    country_iso: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    source_type: str = Field(
        default="unknown",
        description="GPS data source type",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.603716,
                    "longitude": -0.186964,
                    "commodity": "cocoa",
                    "country_iso": "GH",
                    "source_type": "mobile_gps",
                }
            ]
        },
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {EUDR_COMMODITIES}, got '{v}'"
            )
        return v


class ComplianceCertResponseSchema(BaseModel):
    """Response with generated compliance certificate."""

    cert_id: str = Field(
        ...,
        description="Unique certificate identifier",
    )
    status: str = Field(
        ...,
        description="Certificate status: compliant, non_compliant, conditional",
    )
    accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Accuracy score achieved",
    )
    coordinate: CoordinatePairSchema = Field(
        ...,
        description="Certified coordinate",
    )
    issued_at: datetime = Field(
        default_factory=_utcnow,
        description="Certificate issuance timestamp",
    )
    valid_until: datetime = Field(
        ...,
        description="Certificate validity expiry",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    checks_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of all checks performed",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchSummaryResponseSchema(BaseModel):
    """Aggregate summary for batch operations."""

    total: int = Field(..., ge=0, description="Total coordinates processed")
    valid: int = Field(..., ge=0, description="Valid coordinates")
    invalid: int = Field(..., ge=0, description="Invalid coordinates")
    warning_count: int = Field(
        default=0, ge=0, description="Total warnings"
    )
    error_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Error counts by error type",
    )
    precision_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Coordinate counts by precision level",
    )
    tier_distribution: Dict[str, int] = Field(
        default_factory=lambda: {
            "gold": 0, "silver": 0, "bronze": 0, "fail": 0,
        },
        description="Coordinate counts by quality tier",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Summary generation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class RemediationItemSchema(BaseModel):
    """A single remediation action item."""

    index: int = Field(
        ...,
        ge=0,
        description="Index of the coordinate in the original batch",
    )
    error_type: str = Field(
        ...,
        description="Type of error requiring remediation",
    )
    severity: str = Field(
        default="error",
        description="Severity of the issue",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the issue",
    )
    suggested_action: str = Field(
        default="",
        description="Suggested remediation action",
    )
    auto_fixable: bool = Field(
        default=False,
        description="Whether this can be auto-fixed",
    )
    fix_value: Optional[Dict[str, Any]] = Field(
        None,
        description="Auto-fix values if applicable",
    )

    model_config = ConfigDict(from_attributes=True)


class RemediationResponseSchema(BaseModel):
    """Response with remediation plan for failed coordinates."""

    total_errors: int = Field(
        ...,
        ge=0,
        description="Total errors found requiring remediation",
    )
    auto_fixable_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors that can be auto-fixed",
    )
    manual_review_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors requiring manual review",
    )
    remediation_items: List[RemediationItemSchema] = Field(
        default_factory=list,
        description="Detailed remediation items",
    )
    summary_by_error_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Error counts grouped by type",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Remediation plan generation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class ReportResponseSchema(BaseModel):
    """Response for retrieving a stored report."""

    report_id: str = Field(
        ...,
        description="Unique report identifier",
    )
    report_type: str = Field(
        ...,
        description="Type of report: compliance_cert, batch_summary, remediation",
    )
    status: str = Field(
        default="generated",
        description="Report status: generating, generated, failed",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report data payload",
    )
    format: str = Field(
        default="json",
        description="Report format: json, pdf, csv",
    )
    download_url: Optional[str] = Field(
        None,
        description="Download URL for PDF/CSV format",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Report generation timestamp",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Datum Schemas
# =============================================================================


class DatumTransformRequestSchema(BaseModel):
    """Request to transform coordinates between geodetic datums."""

    latitude: float = Field(
        ...,
        description="Latitude in source datum",
    )
    longitude: float = Field(
        ...,
        description="Longitude in source datum",
    )
    source_datum: str = Field(
        ...,
        description="Source geodetic datum (e.g., nad27, ed50)",
    )
    target_datum: str = Field(
        default="wgs84",
        description="Target geodetic datum (default: wgs84)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.603716,
                    "longitude": -0.186964,
                    "source_datum": "accra",
                    "target_datum": "wgs84",
                }
            ]
        },
    )

    @field_validator("source_datum", "target_datum")
    @classmethod
    def validate_datum(cls, v: str) -> str:
        """Validate datum is supported."""
        v = v.lower().strip()
        if v not in SUPPORTED_DATUMS:
            raise ValueError(
                f"datum '{v}' is not supported. "
                f"Use GET /datums for the full list."
            )
        return v


class DatumTransformResponseSchema(BaseModel):
    """Response from datum transformation."""

    latitude: float = Field(
        ...,
        description="Transformed latitude in target datum",
    )
    longitude: float = Field(
        ...,
        description="Transformed longitude in target datum",
    )
    source_datum: str = Field(
        ...,
        description="Source datum used",
    )
    target_datum: str = Field(
        ...,
        description="Target datum used",
    )
    displacement_m: float = Field(
        ...,
        ge=0.0,
        description="Displacement from transformation in metres",
    )
    transformation_method: str = Field(
        default="helmert_7_param",
        description="Transformation method used",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    transformed_at: datetime = Field(
        default_factory=_utcnow,
        description="Transformation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchDatumTransformRequestSchema(BaseModel):
    """Request to transform multiple coordinates between datums."""

    coordinates: List[DatumTransformRequestSchema] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Coordinates to transform (max 10,000)",
    )

    model_config = ConfigDict(extra="forbid")


class BatchDatumTransformResponseSchema(BaseModel):
    """Response from batch datum transformation."""

    total: int = Field(..., ge=0, description="Total coordinates transformed")
    results: List[DatumTransformResponseSchema] = Field(
        default_factory=list,
        description="Per-coordinate transformation results",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds",
    )
    transformed_at: datetime = Field(
        default_factory=_utcnow,
        description="Batch transformation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class DatumInfoSchema(BaseModel):
    """Information about a supported geodetic datum."""

    code: str = Field(..., description="Datum code identifier")
    name: str = Field(..., description="Full datum name")
    epsg: Optional[int] = Field(None, description="EPSG code if applicable")
    region: str = Field(default="", description="Primary geographic region")
    description: str = Field(default="", description="Datum description")

    model_config = ConfigDict(from_attributes=True)


class DatumListResponseSchema(BaseModel):
    """Response listing all supported datums."""

    datums: List[DatumInfoSchema] = Field(
        default_factory=list,
        description="List of supported datums",
    )
    total: int = Field(
        default=0,
        ge=0,
        description="Total supported datums",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Reverse Geocoding Schemas
# =============================================================================


class ReverseGeocodeRequestSchema(BaseModel):
    """Request for reverse geocoding of a coordinate."""

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Longitude in decimal degrees",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"latitude": 5.603716, "longitude": -0.186964}
            ]
        },
    )


class ReverseGeocodeResponseSchema(BaseModel):
    """Response from reverse geocoding."""

    country_iso: Optional[str] = Field(
        None,
        description="ISO 3166-1 alpha-2 country code",
    )
    country_name: Optional[str] = Field(
        None,
        description="Full country name",
    )
    admin_region: Optional[str] = Field(
        None,
        description="Administrative region / state / province",
    )
    nearest_place: Optional[str] = Field(
        None,
        description="Nearest named place",
    )
    land_use: Optional[str] = Field(
        None,
        description="Detected land use classification",
    )
    distance_to_coast_km: Optional[float] = Field(
        None,
        ge=0.0,
        description="Distance to nearest coastline in kilometres",
    )
    commodity_zone: Optional[str] = Field(
        None,
        description="EUDR commodity growing zone if applicable",
    )
    elevation_m: Optional[float] = Field(
        None,
        description="Estimated elevation in metres",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )
    geocoded_at: datetime = Field(
        default_factory=_utcnow,
        description="Geocoding timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchReverseGeocodeRequestSchema(BaseModel):
    """Request for batch reverse geocoding."""

    coordinates: List[ReverseGeocodeRequestSchema] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Coordinates to reverse geocode (max 5,000)",
    )

    model_config = ConfigDict(extra="forbid")


class BatchReverseGeocodeResponseSchema(BaseModel):
    """Response from batch reverse geocoding."""

    total: int = Field(..., ge=0, description="Total coordinates geocoded")
    results: List[ReverseGeocodeResponseSchema] = Field(
        default_factory=list,
        description="Per-coordinate geocoding results",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds",
    )
    geocoded_at: datetime = Field(
        default_factory=_utcnow,
        description="Batch geocoding timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class CountryLookupResponseSchema(BaseModel):
    """Response from country lookup for a coordinate."""

    latitude: float = Field(..., description="Queried latitude")
    longitude: float = Field(..., description="Queried longitude")
    country_iso: Optional[str] = Field(
        None,
        description="Detected country ISO code",
    )
    country_name: Optional[str] = Field(
        None,
        description="Full country name",
    )
    is_on_land: bool = Field(
        default=True,
        description="Whether coordinate is on land",
    )
    admin_regions: List[str] = Field(
        default_factory=list,
        description="Administrative region hierarchy",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Lookup timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Job Schemas
# =============================================================================


class BatchJobRequestSchema(BaseModel):
    """Request to submit a large batch processing job."""

    coordinates: List[CoordinatePairSchema] = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Coordinates for batch processing (max 50,000)",
    )
    operations: List[str] = Field(
        default_factory=lambda: ["validate", "plausibility", "precision"],
        description="Operations to perform: validate, plausibility, precision, assess",
    )
    priority: str = Field(
        default="normal",
        description="Job priority: low, normal, high",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "coordinates": [
                        {"latitude": 5.603716, "longitude": -0.186964},
                    ],
                    "operations": ["validate", "plausibility"],
                    "priority": "normal",
                }
            ]
        },
    )

    @field_validator("operations")
    @classmethod
    def validate_operations(cls, v: List[str]) -> List[str]:
        """Validate operations list."""
        allowed = {"validate", "plausibility", "precision", "assess"}
        for op in v:
            op_lower = op.lower().strip()
            if op_lower not in allowed:
                raise ValueError(
                    f"operation '{op}' not supported. "
                    f"Allowed: {sorted(allowed)}"
                )
        return [op.lower().strip() for op in v]

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate priority."""
        v = v.lower().strip()
        allowed = {"low", "normal", "high"}
        if v not in allowed:
            raise ValueError(
                f"priority must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


class BatchJobResponseSchema(BaseModel):
    """Response after submitting a batch job."""

    job_id: str = Field(
        ...,
        description="Unique batch job identifier",
    )
    status: str = Field(
        default="accepted",
        description="Job status: accepted, queued, processing, completed, failed, cancelled",
    )
    total_coordinates: int = Field(
        ...,
        ge=0,
        description="Total coordinates in job",
    )
    operations: List[str] = Field(
        default_factory=list,
        description="Operations being performed",
    )
    priority: str = Field(
        default="normal",
        description="Job priority",
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow,
        description="Job submission timestamp",
    )
    estimated_completion_seconds: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated completion time in seconds",
    )
    progress_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp when done",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchJobCancelResponseSchema(BaseModel):
    """Response after cancelling a batch job."""

    job_id: str = Field(
        ...,
        description="Cancelled job identifier",
    )
    status: str = Field(
        default="cancelled",
        description="Job status after cancellation",
    )
    completed_coordinates: int = Field(
        default=0,
        ge=0,
        description="Coordinates completed before cancellation",
    )
    total_coordinates: int = Field(
        default=0,
        ge=0,
        description="Total coordinates in job",
    )
    cancelled_at: datetime = Field(
        default_factory=_utcnow,
        description="Cancellation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Health Check
# =============================================================================


class HealthResponseSchema(BaseModel):
    """Health check response for GPS Coordinate Validator API."""

    status: str = Field(default="healthy", description="Service health status")
    agent_id: str = Field(
        default="GL-EUDR-GPS-007",
        description="Agent identifier",
    )
    agent_name: str = Field(
        default="EUDR GPS Coordinate Validator",
        description="Agent display name",
    )
    version: str = Field(default="1.0.0", description="API version")
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Current server timestamp",
    )
    supported_formats: List[str] = Field(
        default_factory=lambda: [
            "decimal_degrees", "dms", "ddm", "utm", "mgrs",
            "signed_dd", "dd_suffix",
        ],
        description="Supported coordinate formats",
    )
    supported_datums_count: int = Field(
        default=33,
        description="Number of supported geodetic datums",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Constants
    "EUDR_COMMODITIES",
    "SUPPORTED_DATUMS",
    "VALID_SOURCE_TYPES",
    # Pagination
    "PaginatedMeta",
    "PaginatedResponse",
    "PaginationParams",
    # Response wrappers
    "ApiResponse",
    "ErrorResponse",
    # Coordinate inputs
    "RawCoordinateSchema",
    "CoordinatePairSchema",
    # Parse
    "ParseRequestSchema",
    "ParseResponseSchema",
    "BatchParseRequestSchema",
    "BatchParseResponseSchema",
    "FormatDetectionResponseSchema",
    "NormalizeResponseSchema",
    # Validation
    "ValidateRequestSchema",
    "ValidationErrorSchema",
    "ValidationResponseSchema",
    "BatchValidateRequestSchema",
    "BatchValidateResponseSchema",
    "RangeCheckResponseSchema",
    "SwapDetectionRequestSchema",
    "SwapDetectionResponseSchema",
    "DuplicateDetectionRequestSchema",
    "DuplicateDetectionResponseSchema",
    # Plausibility
    "PlausibilityRequestSchema",
    "PlausibilityResponseSchema",
    "LandOceanResponseSchema",
    "CountryResponseSchema",
    "CommodityResponseSchema",
    "ElevationResponseSchema",
    # Assessment
    "AssessmentRequestSchema",
    "AccuracyScoreSchema",
    "AssessmentResponseSchema",
    "BatchAssessmentRequestSchema",
    "BatchAssessmentResponseSchema",
    "PrecisionRequestSchema",
    "PrecisionResponseSchema",
    # Report
    "ComplianceCertRequestSchema",
    "ComplianceCertResponseSchema",
    "BatchSummaryResponseSchema",
    "RemediationItemSchema",
    "RemediationResponseSchema",
    "ReportResponseSchema",
    # Datum
    "DatumTransformRequestSchema",
    "DatumTransformResponseSchema",
    "BatchDatumTransformRequestSchema",
    "BatchDatumTransformResponseSchema",
    "DatumInfoSchema",
    "DatumListResponseSchema",
    # Reverse geocoding
    "ReverseGeocodeRequestSchema",
    "ReverseGeocodeResponseSchema",
    "BatchReverseGeocodeRequestSchema",
    "BatchReverseGeocodeResponseSchema",
    "CountryLookupResponseSchema",
    # Batch jobs
    "BatchJobRequestSchema",
    "BatchJobResponseSchema",
    "BatchJobCancelResponseSchema",
    # Health
    "HealthResponseSchema",
]
