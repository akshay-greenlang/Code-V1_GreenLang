# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-002 Geolocation Verification

Pydantic v2 request/response models specific to the REST API layer.
Core domain models are imported from the main models module; this file
defines API-level request wrappers, paginated list responses, batch
submission models, and compliance report schemas.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.agents.eudr.geolocation_verification.models import (
    IssueSeverity,
    QualityTier,
)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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
    timestamp: datetime = Field(default_factory=_utcnow, description="Response timestamp")

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


# =============================================================================
# Coordinate Validation Schemas
# =============================================================================


class CoordinateValidationRequest(BaseModel):
    """Request to validate a single coordinate pair."""

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
    declared_country_code: str = Field(
        default="",
        min_length=0,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code declared by operator",
    )
    commodity: str = Field(
        default="",
        max_length=50,
        description="EUDR commodity identifier (e.g., cocoa, palm_oil, soy)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "lat": 6.6885,
                    "lon": -1.6244,
                    "declared_country_code": "GH",
                    "commodity": "cocoa",
                }
            ]
        },
    )

    @field_validator("declared_country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        if not v:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "declared_country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


class BatchCoordinateRequest(BaseModel):
    """Request to validate multiple coordinate pairs in a batch."""

    coordinates: List[CoordinateValidationRequest] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of coordinates to validate (max 10,000 per batch)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "coordinates": [
                        {
                            "lat": 6.6885,
                            "lon": -1.6244,
                            "declared_country_code": "GH",
                            "commodity": "cocoa",
                        },
                        {
                            "lat": 5.5600,
                            "lon": -0.2057,
                            "declared_country_code": "GH",
                            "commodity": "cocoa",
                        },
                    ]
                }
            ]
        },
    )


class CoordinateValidationResponse(BaseModel):
    """Response for a single coordinate validation."""

    validation_id: str = Field(..., description="Unique validation identifier")
    lat: float = Field(..., description="Validated latitude")
    lon: float = Field(..., description="Validated longitude")
    is_valid: bool = Field(..., description="Overall validation result")
    wgs84_valid: bool = Field(True, description="Within WGS84 bounds")
    precision_decimal_places: int = Field(0, ge=0, description="Detected decimal places")
    precision_score: float = Field(0.0, ge=0.0, le=1.0, description="Precision quality score")
    transposition_detected: bool = Field(False, description="Whether lat/lon appear swapped")
    country_match: bool = Field(True, description="Coordinate falls within declared country")
    resolved_country: Optional[str] = Field(None, description="Country resolved from coordinate")
    is_on_land: bool = Field(True, description="Coordinate falls on land")
    is_duplicate: bool = Field(False, description="Duplicates another coordinate in batch")
    elevation_m: Optional[float] = Field(None, description="Estimated elevation in metres")
    elevation_plausible: bool = Field(True, description="Elevation plausible for commodity")
    cluster_anomaly: bool = Field(False, description="Coordinate is a cluster anomaly")
    issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of validation issues"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    validated_at: datetime = Field(default_factory=_utcnow, description="Validation timestamp")

    model_config = ConfigDict(from_attributes=True)


class BatchCoordinateResponse(BaseModel):
    """Response for batch coordinate validation."""

    batch_id: str = Field(..., description="Unique batch identifier")
    total_coordinates: int = Field(..., ge=0, description="Total coordinates submitted")
    valid_count: int = Field(..., ge=0, description="Number of valid coordinates")
    invalid_count: int = Field(..., ge=0, description="Number of invalid coordinates")
    results: List[CoordinateValidationResponse] = Field(
        default_factory=list, description="Per-coordinate validation results"
    )
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total processing time in ms")
    validated_at: datetime = Field(default_factory=_utcnow, description="Batch validation timestamp")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Polygon Verification Schemas
# =============================================================================


class PolygonVerificationRequest(BaseModel):
    """Request to verify a polygon's topology and geometry."""

    vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        max_length=100000,
        description="List of (lat, lon) tuples forming the polygon ring",
    )
    declared_area_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Operator-declared area in hectares for tolerance check",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "vertices": [
                        [6.6885, -1.6244],
                        [6.6895, -1.6234],
                        [6.6875, -1.6224],
                        [6.6885, -1.6244],
                    ],
                    "declared_area_hectares": 2.5,
                }
            ]
        },
    )


class PolygonRepairRequest(BaseModel):
    """Request to attempt auto-repair of polygon topology issues."""

    vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        max_length=100000,
        description="List of (lat, lon) tuples forming the polygon ring",
    )
    issues_to_repair: List[str] = Field(
        default_factory=list,
        description=(
            "Issue codes to repair (e.g., RING_NOT_CLOSED, WINDING_ORDER_CW, "
            "SELF_INTERSECTION). If empty, attempts all known repairs."
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "vertices": [
                        [6.6885, -1.6244],
                        [6.6895, -1.6234],
                        [6.6875, -1.6224],
                    ],
                    "issues_to_repair": ["RING_NOT_CLOSED", "WINDING_ORDER_CW"],
                }
            ]
        },
    )


class PolygonVerificationResponse(BaseModel):
    """Response from polygon topology verification."""

    verification_id: str = Field(..., description="Unique verification identifier")
    is_valid: bool = Field(..., description="Overall polygon validity")
    ring_closed: bool = Field(True, description="Polygon ring properly closed")
    winding_order_ccw: bool = Field(True, description="Vertices in CCW order")
    has_self_intersection: bool = Field(False, description="Polygon self-intersects")
    vertex_count: int = Field(0, ge=0, description="Number of vertices")
    calculated_area_ha: float = Field(0.0, ge=0.0, description="Geodesic area in hectares")
    declared_area_ha: Optional[float] = Field(None, description="Declared area in hectares")
    area_within_tolerance: bool = Field(True, description="Calculated matches declared area")
    area_tolerance_pct: float = Field(10.0, description="Tolerance percentage used")
    is_sliver: bool = Field(False, description="Polygon is a degenerate sliver")
    has_spikes: bool = Field(False, description="Contains spike vertices")
    spike_vertex_indices: List[int] = Field(
        default_factory=list, description="Indices of spike vertices"
    )
    vertex_density_ok: bool = Field(True, description="Vertices have adequate spacing")
    max_area_ok: bool = Field(True, description="Area within commodity max limit")
    issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected verification issues"
    )
    repair_suggestions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Suggested repair actions"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    verified_at: datetime = Field(default_factory=_utcnow, description="Verification timestamp")

    model_config = ConfigDict(from_attributes=True)


class PolygonRepairResponse(BaseModel):
    """Response from polygon auto-repair attempt."""

    verification_id: str = Field(..., description="Unique verification identifier")
    original_issues: List[str] = Field(
        default_factory=list, description="Issues present before repair"
    )
    repaired_issues: List[str] = Field(
        default_factory=list, description="Issues successfully repaired"
    )
    remaining_issues: List[str] = Field(
        default_factory=list, description="Issues that could not be auto-repaired"
    )
    repaired_vertices: List[Tuple[float, float]] = Field(
        default_factory=list, description="Repaired polygon vertices"
    )
    is_valid_after_repair: bool = Field(
        False, description="Whether polygon passes validation after repair"
    )
    verification_result: Optional[PolygonVerificationResponse] = Field(
        None, description="Full verification result after repair"
    )
    repaired_at: datetime = Field(default_factory=_utcnow, description="Repair timestamp")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Protected Area Screening Schemas
# =============================================================================


class ProtectedAreaScreenRequest(BaseModel):
    """Request to screen a plot against protected areas."""

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
    polygon_vertices: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Optional polygon vertices for overlap calculation",
    )
    buffer_km: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Buffer zone radius in kilometres around the location",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "lat": 6.6885,
                    "lon": -1.6244,
                    "buffer_km": 5.0,
                }
            ]
        },
    )


class NearbyProtectedAreasRequest(BaseModel):
    """Request parameters for listing nearby protected areas."""

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
    radius_km: float = Field(
        default=50.0,
        ge=0.1,
        le=500.0,
        description="Search radius in kilometres",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "lat": 6.6885,
                    "lon": -1.6244,
                    "radius_km": 50.0,
                }
            ]
        },
    )


class ProtectedAreaScreenResponse(BaseModel):
    """Response from protected area screening."""

    overlaps_protected: bool = Field(
        False, description="Whether location overlaps a protected area"
    )
    protected_area_name: Optional[str] = Field(
        None, description="Name of overlapping protected area"
    )
    protected_area_type: Optional[str] = Field(
        None, description="IUCN category or protection type"
    )
    overlap_percentage: float = Field(
        0.0, ge=0.0, le=100.0, description="Percentage of polygon overlapping protected area"
    )
    buffer_km_used: float = Field(5.0, description="Buffer zone radius used")
    nearby_areas: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Protected areas within buffer zone",
    )
    screened_at: datetime = Field(default_factory=_utcnow, description="Screening timestamp")

    model_config = ConfigDict(from_attributes=True)


class NearbyProtectedAreasResponse(BaseModel):
    """Response listing nearby protected areas."""

    lat: float = Field(..., description="Query latitude")
    lon: float = Field(..., description="Query longitude")
    radius_km: float = Field(..., description="Search radius used")
    total_found: int = Field(0, ge=0, description="Total protected areas found")
    areas: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of nearby protected areas with name, type, distance_km, "
            "area_ha, and IUCN category"
        ),
    )
    queried_at: datetime = Field(default_factory=_utcnow, description="Query timestamp")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Deforestation Verification Schemas
# =============================================================================


class DeforestationVerifyRequest(BaseModel):
    """Request to verify deforestation status for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Plot centroid latitude (WGS84)",
    )
    lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Plot centroid longitude (WGS84)",
    )
    polygon_vertices: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Optional polygon vertices for precise area analysis",
    )
    commodity: str = Field(
        default="",
        max_length=50,
        description="EUDR commodity identifier",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "lat": 6.6885,
                    "lon": -1.6244,
                    "commodity": "cocoa",
                }
            ]
        },
    )


class DeforestationVerifyResponse(BaseModel):
    """Response from deforestation status verification."""

    plot_id: str = Field(..., description="Plot identifier")
    deforestation_detected: bool = Field(
        False, description="Whether deforestation detected post-cutoff"
    )
    alert_count: int = Field(0, ge=0, description="Number of deforestation alerts")
    forest_loss_ha: float = Field(0.0, ge=0.0, description="Estimated forest loss in hectares")
    cutoff_date: str = Field("2020-12-31", description="EUDR cutoff date used")
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Detection confidence level"
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Satellite data sources consulted (GFW, JRC, GLAD)",
    )
    verified_at: datetime = Field(default_factory=_utcnow, description="Verification timestamp")

    model_config = ConfigDict(from_attributes=True)


class DeforestationEvidenceResponse(BaseModel):
    """Response with deforestation evidence package for a plot."""

    plot_id: str = Field(..., description="Plot identifier")
    evidence_id: str = Field(..., description="Unique evidence package identifier")
    alert_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual deforestation alerts with dates, coordinates, confidence",
    )
    satellite_imagery: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Satellite imagery references (source, date, resolution)",
    )
    forest_cover_timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical forest cover percentages by year",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    generated_at: datetime = Field(default_factory=_utcnow, description="Evidence generation timestamp")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Full Plot Verification Schemas
# =============================================================================


class PlotVerificationRequest(BaseModel):
    """Request for full verification of a single production plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    lat: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="Plot centroid latitude (WGS84)",
    )
    lon: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="Plot centroid longitude (WGS84)",
    )
    polygon_vertices: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Polygon boundary vertices (lat, lon tuples)",
    )
    declared_area_hectares: Optional[float] = Field(
        None,
        ge=0.0,
        description="Operator-declared area in hectares",
    )
    declared_country_code: str = Field(
        default="",
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity: str = Field(
        default="",
        max_length=50,
        description="EUDR commodity identifier",
    )
    verification_level: str = Field(
        default="standard",
        description="Verification depth: quick, standard, or deep",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "lat": 6.6885,
                    "lon": -1.6244,
                    "polygon_vertices": [
                        [6.6885, -1.6244],
                        [6.6895, -1.6234],
                        [6.6875, -1.6224],
                        [6.6885, -1.6244],
                    ],
                    "declared_area_hectares": 2.5,
                    "declared_country_code": "GH",
                    "commodity": "cocoa",
                    "verification_level": "standard",
                }
            ]
        },
    )

    @field_validator("declared_country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        if not v:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "declared_country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("verification_level")
    @classmethod
    def validate_verification_level(cls, v: str) -> str:
        """Validate verification level is one of the allowed values."""
        v = v.lower().strip()
        allowed = {"quick", "standard", "deep"}
        if v not in allowed:
            raise ValueError(
                f"verification_level must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


class PlotVerificationResponse(BaseModel):
    """Response from full plot verification."""

    verification_id: str = Field(..., description="Unique verification identifier")
    plot_id: str = Field(..., description="Plot identifier")
    verification_level: str = Field("standard", description="Verification depth applied")
    overall_pass: bool = Field(
        ..., description="Whether the plot passes all verification checks"
    )
    coordinate_result: Optional[CoordinateValidationResponse] = Field(
        None, description="Coordinate validation result"
    )
    polygon_result: Optional[PolygonVerificationResponse] = Field(
        None, description="Polygon verification result (if vertices provided)"
    )
    protected_area_result: Optional[ProtectedAreaScreenResponse] = Field(
        None, description="Protected area screening result"
    )
    deforestation_result: Optional[DeforestationVerifyResponse] = Field(
        None, description="Deforestation verification result"
    )
    accuracy_score: Optional[Dict[str, Any]] = Field(
        None, description="Composite accuracy score breakdown"
    )
    quality_tier: Optional[str] = Field(
        None, description="Quality tier classification (gold/silver/bronze/fail)"
    )
    issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="All issues across verification checks"
    )
    processing_time_ms: float = Field(0.0, ge=0.0, description="Total processing time in ms")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    verified_at: datetime = Field(default_factory=_utcnow, description="Verification timestamp")

    model_config = ConfigDict(from_attributes=True)


class PlotVerificationHistoryResponse(BaseModel):
    """Response listing verification history for a plot."""

    plot_id: str = Field(..., description="Plot identifier")
    total_verifications: int = Field(0, ge=0, description="Total verifications performed")
    verifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical verification summaries",
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Verification Schemas
# =============================================================================


class BatchVerificationSubmitRequest(BaseModel):
    """Request to submit a batch of plots for verification."""

    plots: List[PlotVerificationRequest] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of plots to verify (max 10,000 per batch)",
    )
    verification_level: str = Field(
        default="standard",
        description="Verification depth: quick, standard, or deep",
    )
    priority_sort: bool = Field(
        default=True,
        description=(
            "Sort plots by risk priority before processing. "
            "High-risk commodities and countries are verified first."
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plots": [
                        {
                            "plot_id": "plot-gh-001",
                            "lat": 6.6885,
                            "lon": -1.6244,
                            "declared_country_code": "GH",
                            "commodity": "cocoa",
                        },
                        {
                            "plot_id": "plot-gh-002",
                            "lat": 5.5600,
                            "lon": -0.2057,
                            "declared_country_code": "GH",
                            "commodity": "cocoa",
                        },
                    ],
                    "verification_level": "standard",
                    "priority_sort": True,
                }
            ]
        },
    )

    @field_validator("verification_level")
    @classmethod
    def validate_verification_level(cls, v: str) -> str:
        """Validate verification level."""
        v = v.lower().strip()
        allowed = {"quick", "standard", "deep"}
        if v not in allowed:
            raise ValueError(
                f"verification_level must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


class BatchVerificationResponse(BaseModel):
    """Response after submitting a batch verification job."""

    batch_id: str = Field(..., description="Unique batch job identifier")
    status: str = Field(
        default="accepted",
        description="Job status: accepted, processing, completed, failed, cancelled",
    )
    total_plots: int = Field(..., ge=0, description="Total plots submitted")
    verification_level: str = Field("standard", description="Verification depth")
    priority_sort: bool = Field(True, description="Whether priority sorting is enabled")
    submitted_at: datetime = Field(default_factory=_utcnow, description="Submission timestamp")
    estimated_completion_seconds: Optional[float] = Field(
        None, description="Estimated time to complete in seconds"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchStatusResponse(BaseModel):
    """Response with batch job status and results."""

    batch_id: str = Field(..., description="Batch job identifier")
    status: str = Field(
        ..., description="Job status: accepted, processing, completed, failed, cancelled"
    )
    total_plots: int = Field(..., ge=0, description="Total plots in batch")
    completed_plots: int = Field(0, ge=0, description="Plots completed")
    failed_plots: int = Field(0, ge=0, description="Plots that failed verification")
    passed_plots: int = Field(0, ge=0, description="Plots that passed all checks")
    average_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Average accuracy score"
    )
    quality_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
        description="Distribution of quality tiers",
    )
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-plot verification results (populated when completed)",
    )
    started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    processing_time_ms: Optional[float] = Field(
        None, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchProgressResponse(BaseModel):
    """Real-time progress of a batch verification job."""

    batch_id: str = Field(..., description="Batch job identifier")
    status: str = Field(..., description="Current job status")
    total_plots: int = Field(..., ge=0, description="Total plots in batch")
    completed_plots: int = Field(0, ge=0, description="Plots completed so far")
    progress_percent: float = Field(
        0.0, ge=0.0, le=100.0, description="Completion percentage"
    )
    current_plot_id: Optional[str] = Field(
        None, description="Plot currently being processed"
    )
    elapsed_seconds: float = Field(0.0, ge=0.0, description="Elapsed time in seconds")
    estimated_remaining_seconds: Optional[float] = Field(
        None, description="Estimated time remaining in seconds"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchCancelResponse(BaseModel):
    """Response after cancelling a batch verification job."""

    batch_id: str = Field(..., description="Cancelled batch job identifier")
    status: str = Field(default="cancelled", description="Job status after cancellation")
    completed_plots: int = Field(0, ge=0, description="Plots completed before cancellation")
    total_plots: int = Field(0, ge=0, description="Total plots that were in the batch")
    cancelled_at: datetime = Field(default_factory=_utcnow, description="Cancellation timestamp")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Accuracy Scoring Schemas
# =============================================================================


class ScoreWeightsUpdateRequest(BaseModel):
    """Request to update accuracy score component weights (admin only)."""

    weights: Dict[str, float] = Field(
        ...,
        description=(
            "Score component weights. Required keys: precision, polygon, "
            "country, protected, deforestation, temporal. Must sum to 1.0."
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "weights": {
                        "precision": 0.20,
                        "polygon": 0.20,
                        "country": 0.15,
                        "protected": 0.15,
                        "deforestation": 0.15,
                        "temporal": 0.15,
                    }
                }
            ]
        },
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate score weights keys and sum."""
        required_keys = {
            "precision", "polygon", "country",
            "protected", "deforestation", "temporal",
        }
        if set(v.keys()) != required_keys:
            raise ValueError(
                f"weights must contain exactly: {sorted(required_keys)}, "
                f"got {sorted(v.keys())}"
            )
        for key, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"weights['{key}'] must be in [0.0, 1.0], got {value}"
                )
        weight_sum = sum(v.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(
                f"weights must sum to 1.0, got {weight_sum:.4f}"
            )
        return v


class AccuracyScoreResponse(BaseModel):
    """Response with a plot's accuracy score breakdown."""

    score_id: str = Field(..., description="Unique score identifier")
    plot_id: str = Field(..., description="Plot identifier")
    total_score: float = Field(
        ..., ge=0.0, le=100.0, description="Composite accuracy score (0-100)"
    )
    coordinate_precision_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Coordinate precision sub-score"
    )
    polygon_quality_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Polygon quality sub-score"
    )
    country_match_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Country match sub-score"
    )
    protected_area_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Protected area sub-score"
    )
    deforestation_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Deforestation sub-score"
    )
    temporal_consistency_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Temporal consistency sub-score"
    )
    quality_tier: str = Field(
        ..., description="Quality tier: gold, silver, bronze, fail"
    )
    weights_used: Dict[str, float] = Field(
        default_factory=dict, description="Weight configuration used"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    scored_at: datetime = Field(default_factory=_utcnow, description="Scoring timestamp")

    model_config = ConfigDict(from_attributes=True)


class ScoreHistoryResponse(BaseModel):
    """Response listing score history for a plot."""

    plot_id: str = Field(..., description="Plot identifier")
    total_scores: int = Field(0, ge=0, description="Total scores recorded")
    scores: List[AccuracyScoreResponse] = Field(
        default_factory=list, description="Historical scores"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


class ScoreSummaryResponse(BaseModel):
    """Response with aggregate score statistics."""

    total_plots_scored: int = Field(0, ge=0, description="Total plots scored")
    average_score: float = Field(0.0, ge=0.0, le=100.0, description="Average accuracy score")
    median_score: float = Field(0.0, ge=0.0, le=100.0, description="Median accuracy score")
    min_score: float = Field(0.0, ge=0.0, le=100.0, description="Minimum accuracy score")
    max_score: float = Field(0.0, ge=0.0, le=100.0, description="Maximum accuracy score")
    quality_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
        description="Distribution of quality tiers",
    )
    average_sub_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Average sub-scores by component",
    )
    current_weights: Dict[str, float] = Field(
        default_factory=dict, description="Current score weights"
    )
    generated_at: datetime = Field(default_factory=_utcnow, description="Summary generation timestamp")

    model_config = ConfigDict(from_attributes=True)


class ScoreWeightsResponse(BaseModel):
    """Response after updating score weights."""

    status: str = Field(default="updated", description="Update status")
    previous_weights: Dict[str, float] = Field(
        default_factory=dict, description="Previous weight values"
    )
    new_weights: Dict[str, float] = Field(
        default_factory=dict, description="Updated weight values"
    )
    updated_at: datetime = Field(default_factory=_utcnow, description="Update timestamp")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Compliance Reporting Schemas
# =============================================================================


class ComplianceReportRequest(BaseModel):
    """Request to generate an Article 9 compliance report."""

    operator_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Operator identifier for report scope",
    )
    commodity: Optional[str] = Field(
        None,
        max_length=50,
        description="Filter report to specific EUDR commodity",
    )
    format: str = Field(
        default="json",
        description="Report output format: json, pdf, csv",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "operator_id": "OP-GH-001",
                    "commodity": "cocoa",
                    "format": "json",
                }
            ]
        },
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate report format."""
        v = v.lower().strip()
        allowed = {"json", "pdf", "csv"}
        if v not in allowed:
            raise ValueError(
                f"format must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


class ComplianceReportResponse(BaseModel):
    """Response with generated Article 9 compliance report."""

    report_id: str = Field(..., description="Unique report identifier")
    operator_id: str = Field(..., description="Operator identifier")
    commodity: Optional[str] = Field(None, description="Commodity filter applied")
    format: str = Field("json", description="Report output format")
    status: str = Field(
        default="generated",
        description="Report status: generating, generated, failed",
    )
    total_plots: int = Field(0, ge=0, description="Total plots in scope")
    compliant_plots: int = Field(0, ge=0, description="Plots meeting compliance")
    non_compliant_plots: int = Field(0, ge=0, description="Plots failing compliance")
    compliance_rate: float = Field(
        0.0, ge=0.0, le=100.0, description="Compliance percentage"
    )
    average_accuracy_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Average accuracy score"
    )
    quality_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
        description="Quality tier distribution",
    )
    issues_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Issue counts by severity",
    )
    report_data: Optional[Dict[str, Any]] = Field(
        None, description="Full report data (for JSON format)"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL (for PDF/CSV format)"
    )
    generated_at: datetime = Field(default_factory=_utcnow, description="Report generation timestamp")

    model_config = ConfigDict(from_attributes=True)


class ComplianceSummaryResponse(BaseModel):
    """Response with compliance dashboard summary data."""

    operator_id: Optional[str] = Field(None, description="Operator filter applied")
    total_operators: int = Field(0, ge=0, description="Total operators in scope")
    total_plots: int = Field(0, ge=0, description="Total production plots")
    verified_plots: int = Field(0, ge=0, description="Plots with completed verification")
    compliant_plots: int = Field(0, ge=0, description="Plots meeting EUDR compliance")
    overall_compliance_rate: float = Field(
        0.0, ge=0.0, le=100.0, description="Overall compliance percentage"
    )
    average_accuracy_score: float = Field(
        0.0, ge=0.0, le=100.0, description="Average accuracy score across all plots"
    )
    quality_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
        description="Quality tier distribution",
    )
    top_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Most common issues with counts",
    )
    by_commodity: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Compliance breakdown by commodity",
    )
    by_country: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Compliance breakdown by country",
    )
    generated_at: datetime = Field(default_factory=_utcnow, description="Summary generation timestamp")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    agent_id: str = Field(default="GL-EUDR-GEO-002")
    agent_name: str = Field(default="EUDR Geolocation Verification Agent")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Pagination
    "PaginatedMeta",
    "PaginatedResponse",
    "PaginationParams",
    # Response wrappers
    "ApiResponse",
    "ErrorResponse",
    # Coordinate validation
    "CoordinateValidationRequest",
    "BatchCoordinateRequest",
    "CoordinateValidationResponse",
    "BatchCoordinateResponse",
    # Polygon verification
    "PolygonVerificationRequest",
    "PolygonRepairRequest",
    "PolygonVerificationResponse",
    "PolygonRepairResponse",
    # Protected area screening
    "ProtectedAreaScreenRequest",
    "NearbyProtectedAreasRequest",
    "ProtectedAreaScreenResponse",
    "NearbyProtectedAreasResponse",
    # Deforestation verification
    "DeforestationVerifyRequest",
    "DeforestationVerifyResponse",
    "DeforestationEvidenceResponse",
    # Full plot verification
    "PlotVerificationRequest",
    "PlotVerificationResponse",
    "PlotVerificationHistoryResponse",
    # Batch verification
    "BatchVerificationSubmitRequest",
    "BatchVerificationResponse",
    "BatchStatusResponse",
    "BatchProgressResponse",
    "BatchCancelResponse",
    # Accuracy scoring
    "ScoreWeightsUpdateRequest",
    "AccuracyScoreResponse",
    "ScoreHistoryResponse",
    "ScoreSummaryResponse",
    "ScoreWeightsResponse",
    # Compliance reporting
    "ComplianceReportRequest",
    "ComplianceReportResponse",
    "ComplianceSummaryResponse",
    # Health
    "HealthResponse",
]
