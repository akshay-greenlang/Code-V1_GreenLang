# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-004 Forest Cover Analysis

Pydantic v2 request/response models for the Forest Cover Analysis REST API
covering canopy density analysis, forest type classification, historical
cover reconstruction, deforestation-free verification, canopy height
estimation, fragmentation analysis, biomass estimation, compliance report
generation, and batch analysis.

Core domain models are referenced via the analysis engines; this file
defines API-level request wrappers, response envelopes, pagination, and
batch schemas with full Pydantic v2 validation.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _request_id() -> str:
    """Generate a unique request ID."""
    return f"req-{uuid.uuid4().hex[:16]}"


# =============================================================================
# Enumerations
# =============================================================================


class DensityMethod(str, Enum):
    """Canopy density estimation method."""

    SPECTRAL = "spectral"
    LIDAR = "lidar"
    RADAR = "radar"
    FUSION = "fusion"
    HANSEN = "hansen"


class ClassificationMethod(str, Enum):
    """Forest type classification method."""

    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    OBJECT_BASED = "object_based"
    DEEP_LEARNING = "deep_learning"
    RULE_BASED = "rule_based"


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodity types."""

    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    SOY = "soy"
    RUBBER = "rubber"
    CATTLE = "cattle"
    WOOD = "wood"


class HeightSource(str, Enum):
    """Canopy height data source."""

    GEDI = "gedi"
    ICESAT2 = "icesat2"
    LIDAR = "lidar"
    PHOTOGRAMMETRY = "photogrammetry"
    RADAR = "radar"


class BiomassSource(str, Enum):
    """Above-ground biomass data source."""

    GEDI_L4A = "gedi_l4a"
    ESA_CCI = "esa_cci"
    GLOBBIOMASS = "globbiomass"
    LIDAR = "lidar"
    ALLOMETRIC = "allometric"


class ReportFormat(str, Enum):
    """Compliance report output format."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XLSX = "xlsx"


class ForestType(str, Enum):
    """Forest type classification result."""

    TROPICAL_MOIST = "tropical_moist"
    TROPICAL_DRY = "tropical_dry"
    SUBTROPICAL = "subtropical"
    TEMPERATE_BROADLEAF = "temperate_broadleaf"
    TEMPERATE_CONIFER = "temperate_conifer"
    BOREAL = "boreal"
    MANGROVE = "mangrove"
    PLANTATION = "plantation"
    AGROFORESTRY = "agroforestry"
    SECONDARY_GROWTH = "secondary_growth"
    NON_FOREST = "non_forest"
    UNKNOWN = "unknown"


class DeforestationVerdict(str, Enum):
    """Deforestation-free verification verdict."""

    DEFORESTATION_FREE = "deforestation_free"
    DEFORESTATION_DETECTED = "deforestation_detected"
    DEGRADATION_DETECTED = "degradation_detected"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_DATA = "insufficient_data"


class AnalysisStatus(str, Enum):
    """Analysis job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_VALID_BIOMES = frozenset({
    "tropical_moist", "tropical_dry", "subtropical",
    "temperate", "boreal", "mangrove", "savanna",
})

_VALID_SORT_FIELDS = frozenset({
    "created_at", "updated_at", "plot_id", "density_pct",
    "confidence", "processing_time_ms",
})

_VALID_SORT_ORDERS = frozenset({"asc", "desc"})


# =============================================================================
# WKT Polygon Validator
# =============================================================================


_WKT_POLYGON_RE = re.compile(
    r"^POLYGON\s*\(\s*\(.*\)\s*\)$",
    re.IGNORECASE | re.DOTALL,
)


def _validate_polygon_wkt(v: str) -> str:
    """Validate that a WKT string is a valid POLYGON geometry.

    Args:
        v: WKT string to validate.

    Returns:
        Cleaned WKT string.

    Raises:
        ValueError: If the string is not a valid WKT POLYGON.
    """
    v = v.strip()
    if not _WKT_POLYGON_RE.match(v):
        raise ValueError(
            "polygon_wkt must be a valid WKT POLYGON, "
            "e.g. 'POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))'"
        )
    return v


def _validate_plot_id(v: str) -> str:
    """Validate plot ID format.

    Args:
        v: Plot ID string to validate.

    Returns:
        Cleaned plot ID string.

    Raises:
        ValueError: If the plot ID is empty or too long.
    """
    v = v.strip()
    if not v:
        raise ValueError("plot_id must not be empty")
    if len(v) > 200:
        raise ValueError("plot_id must not exceed 200 characters")
    return v


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
# Base Response Wrappers
# =============================================================================


class ApiResponse(BaseModel):
    """Standard API success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")
    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
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
# Density Schemas - Request
# =============================================================================


class AnalyzeDensityRequest(BaseModel):
    """Request to analyze canopy density for a single plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    imagery_date: Optional[date] = Field(
        None,
        description="Target imagery date (YYYY-MM-DD). Defaults to latest available.",
    )
    method: DensityMethod = Field(
        default=DensityMethod.FUSION,
        description="Density estimation method: spectral, lidar, radar, fusion, hansen",
    )
    biome: Optional[str] = Field(
        None,
        description="Biome context for threshold adjustment",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "method": "fusion",
                    "biome": "tropical_moist",
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)

    @field_validator("biome")
    @classmethod
    def validate_biome(cls, v: Optional[str]) -> Optional[str]:
        """Validate biome identifier."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in _VALID_BIOMES:
            raise ValueError(
                f"biome must be one of {sorted(_VALID_BIOMES)}, got '{v}'"
            )
        return v


class BatchDensityRequest(BaseModel):
    """Request for batch canopy density analysis."""

    plots: List[AnalyzeDensityRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots to analyze (max 5000 per batch)",
    )

    model_config = ConfigDict(extra="forbid")


class CompareDensityRequest(BaseModel):
    """Request to compare canopy density between two dates."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    date_before: date = Field(
        ...,
        description="Earlier comparison date (YYYY-MM-DD)",
    )
    date_after: date = Field(
        ...,
        description="Later comparison date (YYYY-MM-DD)",
    )
    method: DensityMethod = Field(
        default=DensityMethod.FUSION,
        description="Density estimation method",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "date_before": "2020-06-01",
                    "date_after": "2024-06-01",
                    "method": "fusion",
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)

    @field_validator("date_after")
    @classmethod
    def validate_date_order(cls, v: date, info) -> date:
        """Validate date_after is after date_before."""
        before = info.data.get("date_before")
        if before and v <= before:
            raise ValueError(
                f"date_after ({v}) must be after date_before ({before})"
            )
        return v


# =============================================================================
# Density Schemas - Response
# =============================================================================


class CanopyDensityResponse(BaseModel):
    """Response from canopy density analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    density_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Canopy density percentage"
    )
    density_class: str = Field(
        default="unknown",
        description="Density classification: dense, moderate, sparse, open, non_forest",
    )
    method: str = Field(default="fusion", description="Method used for estimation")
    pixel_count: int = Field(default=0, ge=0, description="Pixels analyzed")
    area_ha: float = Field(default=0.0, ge=0.0, description="Plot area in hectares")
    cloud_cover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cloud cover in analyzed imagery",
    )
    imagery_date: Optional[date] = Field(
        None, description="Date of imagery used"
    )
    biome: Optional[str] = Field(None, description="Biome context applied")
    fao_threshold_met: bool = Field(
        default=False,
        description="Whether density meets FAO forest threshold (>=10%)",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Analysis confidence score"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Analysis timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class DensityComparisonResponse(BaseModel):
    """Response from density comparison between two dates."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    density_before_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Canopy density at earlier date"
    )
    density_after_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Canopy density at later date"
    )
    density_change_pct: float = Field(
        ..., ge=-100.0, le=100.0,
        description="Density change (after - before) in percentage points",
    )
    density_change_relative_pct: float = Field(
        default=0.0,
        description="Relative change as percentage of original density",
    )
    date_before: date = Field(..., description="Earlier comparison date")
    date_after: date = Field(..., description="Later comparison date")
    change_classification: str = Field(
        default="no_change",
        description="Change type: gain, loss, degradation, no_change",
    )
    method: str = Field(default="fusion", description="Method used")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Comparison confidence"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Comparison timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class DensityHistoryEntry(BaseModel):
    """Single entry in density history time series."""

    date: date = Field(..., description="Observation date")
    density_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Canopy density percentage"
    )
    method: str = Field(default="fusion", description="Method used")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )
    source: str = Field(default="", description="Primary data source")

    model_config = ConfigDict(from_attributes=True)


class DensityHistoryResponse(BaseModel):
    """Response with density history over time for a plot."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    entries: List[DensityHistoryEntry] = Field(
        default_factory=list, description="Density time series entries"
    )
    trend: str = Field(
        default="stable",
        description="Overall trend: increasing, decreasing, stable, volatile",
    )
    mean_density_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Mean density across all observations",
    )
    min_density_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum observed density",
    )
    max_density_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Maximum observed density",
    )
    total_observations: int = Field(
        default=0, ge=0, description="Total observations in history"
    )
    meta: Optional[PaginatedMeta] = Field(
        None, description="Pagination metadata"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Query timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Classification Schemas - Request
# =============================================================================


class ClassifyForestRequest(BaseModel):
    """Request to classify forest type for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    date_range_start: date = Field(
        ...,
        description="Classification window start date",
    )
    date_range_end: date = Field(
        ...,
        description="Classification window end date",
    )
    methods: Optional[List[ClassificationMethod]] = Field(
        None,
        description="Classification methods to use. Defaults to all available.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "date_range_start": "2024-01-01",
                    "date_range_end": "2024-06-30",
                    "methods": ["supervised", "deep_learning"],
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)

    @field_validator("date_range_end")
    @classmethod
    def validate_date_range(cls, v: date, info) -> date:
        """Validate end date is after start date."""
        start = info.data.get("date_range_start")
        if start and v < start:
            raise ValueError(
                f"date_range_end ({v}) must be on or after date_range_start ({start})"
            )
        return v


class BatchClassifyRequest(BaseModel):
    """Request for batch forest type classification."""

    plots: List[ClassifyForestRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots to classify (max 5000 per batch)",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Classification Schemas - Response
# =============================================================================


class ForestClassificationResponse(BaseModel):
    """Response from forest type classification."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    primary_type: ForestType = Field(
        ..., description="Primary forest type classification"
    )
    secondary_type: Optional[ForestType] = Field(
        None, description="Secondary forest type (if mixed)"
    )
    type_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across forest types",
    )
    methods_used: List[str] = Field(
        default_factory=list, description="Classification methods applied"
    )
    dominant_species_group: Optional[str] = Field(
        None, description="Dominant tree species group if identifiable"
    )
    canopy_structure: str = Field(
        default="unknown",
        description="Canopy structure: single_layer, multi_layer, emergent",
    )
    is_primary_forest: bool = Field(
        default=False,
        description="Whether classified as primary/old-growth forest",
    )
    is_plantation: bool = Field(
        default=False,
        description="Whether classified as plantation forest",
    )
    area_ha: float = Field(default=0.0, ge=0.0, description="Plot area in hectares")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Classification confidence"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    scenes_analyzed: int = Field(
        default=0, ge=0, description="Number of scenes analyzed"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Classification timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class ForestTypeInfo(BaseModel):
    """Information about a single forest type."""

    type_id: str = Field(..., description="Forest type identifier")
    name: str = Field(..., description="Forest type display name")
    description: str = Field(default="", description="Type description")
    typical_canopy_cover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Typical canopy cover percentage",
    )
    typical_height_m: float = Field(
        default=0.0, ge=0.0, description="Typical canopy height in metres"
    )
    typical_biomass_t_ha: float = Field(
        default=0.0, ge=0.0,
        description="Typical above-ground biomass in tonnes per hectare",
    )
    climate_zones: List[str] = Field(
        default_factory=list, description="Typical climate zones"
    )
    eudr_relevance: str = Field(
        default="medium",
        description="Relevance to EUDR compliance: high, medium, low",
    )

    model_config = ConfigDict(from_attributes=True)


class ForestTypesListResponse(BaseModel):
    """Response listing all forest types with descriptions."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    types: List[ForestTypeInfo] = Field(
        default_factory=list, description="Available forest types"
    )
    total: int = Field(default=0, ge=0, description="Total types")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Historical Reconstruction Schemas - Request
# =============================================================================


class ReconstructHistoryRequest(BaseModel):
    """Request to reconstruct forest cover at EUDR cutoff date."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    target_date: date = Field(
        default=date(2020, 12, 31),
        description="Target reconstruction date. Defaults to EUDR cutoff (2020-12-31).",
    )
    window_years: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Years of imagery to composite for reconstruction (1-10)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "target_date": "2020-12-31",
                    "window_years": 3,
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


class BatchReconstructRequest(BaseModel):
    """Request for batch historical reconstruction."""

    plots: List[ReconstructHistoryRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots to reconstruct (max 5000 per batch)",
    )

    model_config = ConfigDict(extra="forbid")


class CompareHistoricalRequest(BaseModel):
    """Request to compare cutoff vs current forest cover."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    cutoff_date: date = Field(
        default=date(2020, 12, 31),
        description="EUDR cutoff date for baseline comparison",
    )
    current_date: Optional[date] = Field(
        None,
        description="Current comparison date. Defaults to latest available.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "cutoff_date": "2020-12-31",
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


# =============================================================================
# Historical Reconstruction Schemas - Response
# =============================================================================


class HistoricalCoverResponse(BaseModel):
    """Response from historical forest cover reconstruction."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    reconstruction_id: str = Field(..., description="Unique reconstruction identifier")
    plot_id: str = Field(..., description="Plot identifier")
    target_date: date = Field(..., description="Target reconstruction date")
    window_start: date = Field(..., description="Composite window start date")
    window_end: date = Field(..., description="Composite window end date")
    forest_cover_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Reconstructed forest cover percentage at target date",
    )
    canopy_density_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Reconstructed canopy density percentage",
    )
    forest_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Forest area in hectares at target date",
    )
    non_forest_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Non-forest area in hectares at target date",
    )
    ndvi_mean: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Mean NDVI at target date",
    )
    forest_type: Optional[ForestType] = Field(
        None, description="Dominant forest type at target date"
    )
    scenes_composited: int = Field(
        default=0, ge=0, description="Number of scenes composited"
    )
    cloud_free_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cloud-free coverage in composite",
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Reconstruction confidence"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Reconstruction timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class HistoricalComparisonResponse(BaseModel):
    """Response from cutoff vs current forest cover comparison."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    cutoff_date: date = Field(..., description="Baseline cutoff date")
    current_date: date = Field(..., description="Current comparison date")
    cutoff_forest_cover_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Forest cover at cutoff date",
    )
    current_forest_cover_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Current forest cover",
    )
    forest_cover_change_pct: float = Field(
        ..., ge=-100.0, le=100.0,
        description="Forest cover change (current - cutoff) in percentage points",
    )
    cutoff_density_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Canopy density at cutoff date",
    )
    current_density_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Current canopy density",
    )
    deforestation_detected: bool = Field(
        default=False,
        description="Whether significant forest loss detected post-cutoff",
    )
    degradation_detected: bool = Field(
        default=False,
        description="Whether significant degradation detected post-cutoff",
    )
    change_classification: str = Field(
        default="no_change",
        description="Classification: no_change, deforestation, degradation, regrowth",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Comparison confidence"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Comparison timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class DataSourceInfo(BaseModel):
    """Information about a data source used in reconstruction."""

    source_id: str = Field(..., description="Source identifier")
    source_type: str = Field(
        ..., description="Source type: satellite, lidar, radar, field_survey"
    )
    provider: str = Field(default="", description="Data provider name")
    acquisition_date: Optional[date] = Field(
        None, description="Data acquisition date"
    )
    spatial_resolution_m: float = Field(
        default=0.0, ge=0.0, description="Spatial resolution in metres"
    )
    cloud_cover_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Cloud cover percentage"
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Quality score"
    )
    bands_used: List[str] = Field(
        default_factory=list, description="Spectral bands used"
    )

    model_config = ConfigDict(from_attributes=True)


class DataSourcesResponse(BaseModel):
    """Response listing data sources used for a reconstruction."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    sources: List[DataSourceInfo] = Field(
        default_factory=list, description="Data sources used"
    )
    total_sources: int = Field(default=0, ge=0, description="Total sources")
    primary_source: Optional[str] = Field(
        None, description="Primary data source"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Query timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Verification Schemas - Request
# =============================================================================


class VerifyDeforestationFreeRequest(BaseModel):
    """Request to verify deforestation-free status for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR-regulated commodity",
    )
    include_evidence: bool = Field(
        default=True,
        description="Include supporting evidence in response",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "commodity": "cocoa",
                    "include_evidence": True,
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


class BatchVerifyRequest(BaseModel):
    """Request for batch deforestation-free verification."""

    plots: List[VerifyDeforestationFreeRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots to verify (max 5000 per batch)",
    )

    model_config = ConfigDict(extra="forbid")


class CompletePlotAnalysisRequest(BaseModel):
    """Request for complete plot analysis (all engines + verdict)."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR-regulated commodity",
    )
    include_all: bool = Field(
        default=True,
        description="Include all analysis engines (density, classification, "
        "historical, height, fragmentation, biomass)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "commodity": "cocoa",
                    "include_all": True,
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


# =============================================================================
# Verification Schemas - Response
# =============================================================================


class EvidenceItem(BaseModel):
    """Single piece of evidence supporting a verification verdict."""

    evidence_type: str = Field(
        ..., description="Evidence type: satellite_image, ndvi_series, "
        "height_profile, biomass_estimate, classification_result"
    )
    description: str = Field(default="", description="Evidence description")
    date: Optional[date] = Field(None, description="Evidence date")
    source: str = Field(default="", description="Evidence source")
    value: Optional[float] = Field(None, description="Numeric evidence value")
    unit: Optional[str] = Field(None, description="Value unit")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Evidence confidence"
    )

    model_config = ConfigDict(from_attributes=True)


class DeforestationFreeResponse(BaseModel):
    """Response from deforestation-free verification."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    verification_id: str = Field(..., description="Unique verification identifier")
    plot_id: str = Field(..., description="Plot identifier")
    commodity: str = Field(..., description="EUDR commodity verified")
    verdict: DeforestationVerdict = Field(
        ..., description="Deforestation-free verification verdict"
    )
    cutoff_date: str = Field(
        default="2020-12-31", description="EUDR cutoff date applied"
    )
    forest_cover_at_cutoff_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Forest cover at cutoff date",
    )
    forest_cover_current_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Current forest cover",
    )
    forest_cover_change_pct: float = Field(
        default=0.0, ge=-100.0, le=100.0,
        description="Forest cover change (current - cutoff)",
    )
    deforestation_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Deforestation area in hectares since cutoff",
    )
    degradation_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Degradation area in hectares since cutoff",
    )
    engines_used: List[str] = Field(
        default_factory=list,
        description="Analysis engines used for verification",
    )
    engine_agreement: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Agreement ratio across engines",
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Supporting evidence items",
    )
    risk_level: str = Field(
        default="unknown",
        description="Risk level: negligible, low, standard, high, benchmark",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall verification confidence",
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Verification timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Analysis Schemas - Request (Height, Fragmentation, Biomass)
# =============================================================================


class EstimateHeightRequest(BaseModel):
    """Request to estimate canopy height for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    sources: Optional[List[HeightSource]] = Field(
        None,
        description="Canopy height data sources. Defaults to all available.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "sources": ["gedi", "icesat2"],
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


class AnalyzeFragmentationRequest(BaseModel):
    """Request to analyze forest fragmentation for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    edge_buffer_m: float = Field(
        default=100.0,
        ge=10.0,
        le=1000.0,
        description="Edge buffer distance in metres for edge effect analysis",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "edge_buffer_m": 100.0,
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


class EstimateBiomassRequest(BaseModel):
    """Request to estimate above-ground biomass for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    sources: Optional[List[BiomassSource]] = Field(
        None,
        description="Biomass data sources. Defaults to all available.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "sources": ["gedi_l4a", "esa_cci"],
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


class CompareAnalysisRequest(BaseModel):
    """Request to compare metrics between two dates."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )
    date_before: date = Field(
        ...,
        description="Earlier comparison date",
    )
    date_after: date = Field(
        ...,
        description="Later comparison date",
    )
    metrics: Optional[List[str]] = Field(
        None,
        description="Metrics to compare: height, biomass, fragmentation. Defaults to all.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                    "date_before": "2020-06-01",
                    "date_after": "2024-06-01",
                    "metrics": ["height", "biomass"],
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)

    @field_validator("date_after")
    @classmethod
    def validate_date_order(cls, v: date, info) -> date:
        """Validate date_after is after date_before."""
        before = info.data.get("date_before")
        if before and v <= before:
            raise ValueError(
                f"date_after ({v}) must be after date_before ({before})"
            )
        return v


# =============================================================================
# Analysis Schemas - Response (Height, Fragmentation, Biomass)
# =============================================================================


class CanopyHeightResponse(BaseModel):
    """Response from canopy height estimation."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    mean_height_m: float = Field(
        ..., ge=0.0, description="Mean canopy height in metres"
    )
    median_height_m: float = Field(
        default=0.0, ge=0.0, description="Median canopy height in metres"
    )
    max_height_m: float = Field(
        default=0.0, ge=0.0, description="Maximum canopy height in metres"
    )
    min_height_m: float = Field(
        default=0.0, ge=0.0, description="Minimum canopy height in metres"
    )
    std_dev_m: float = Field(
        default=0.0, ge=0.0, description="Height standard deviation in metres"
    )
    p95_height_m: float = Field(
        default=0.0, ge=0.0, description="95th percentile height in metres"
    )
    fao_threshold_met: bool = Field(
        default=False,
        description="Whether height meets FAO forest threshold (>=5m)",
    )
    height_distribution: Optional[Dict[str, int]] = Field(
        None,
        description="Height distribution histogram (bin_label -> count)",
    )
    sources_used: List[str] = Field(
        default_factory=list, description="Height data sources used"
    )
    footprint_count: int = Field(
        default=0, ge=0, description="Number of height footprints/samples"
    )
    area_ha: float = Field(default=0.0, ge=0.0, description="Plot area in hectares")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimation confidence"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Estimation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class FragmentationResponse(BaseModel):
    """Response from forest fragmentation analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    total_patches: int = Field(
        default=0, ge=0, description="Number of forest patches"
    )
    largest_patch_ha: float = Field(
        default=0.0, ge=0.0, description="Largest patch area in hectares"
    )
    mean_patch_ha: float = Field(
        default=0.0, ge=0.0, description="Mean patch area in hectares"
    )
    edge_density_m_per_ha: float = Field(
        default=0.0, ge=0.0,
        description="Edge density in metres per hectare",
    )
    core_area_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Core area percentage (interior beyond edge buffer)",
    )
    edge_area_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Edge area percentage (within edge buffer)",
    )
    shape_index: float = Field(
        default=0.0, ge=0.0,
        description="Mean patch shape index (1.0 = circle, higher = complex)",
    )
    connectivity_index: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Landscape connectivity index (0-1)",
    )
    fragmentation_class: str = Field(
        default="unknown",
        description="Fragmentation level: intact, perforated, fragmented, "
        "patch, relictual",
    )
    edge_buffer_m: float = Field(
        default=100.0, ge=0.0, description="Edge buffer used in metres"
    )
    area_ha: float = Field(default=0.0, ge=0.0, description="Total plot area")
    forest_area_ha: float = Field(
        default=0.0, ge=0.0, description="Forested area within plot"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Analysis confidence"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Analysis timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class BiomassResponse(BaseModel):
    """Response from above-ground biomass estimation."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    agb_mean_t_ha: float = Field(
        ..., ge=0.0,
        description="Mean above-ground biomass in tonnes per hectare",
    )
    agb_total_t: float = Field(
        default=0.0, ge=0.0,
        description="Total above-ground biomass in tonnes",
    )
    agb_median_t_ha: float = Field(
        default=0.0, ge=0.0,
        description="Median AGB in tonnes per hectare",
    )
    agb_std_dev_t_ha: float = Field(
        default=0.0, ge=0.0,
        description="AGB standard deviation in tonnes per hectare",
    )
    carbon_stock_t_ha: float = Field(
        default=0.0, ge=0.0,
        description="Estimated carbon stock in tonnes per hectare (AGB * 0.47)",
    )
    carbon_stock_total_t: float = Field(
        default=0.0, ge=0.0,
        description="Total estimated carbon stock in tonnes",
    )
    sources_used: List[str] = Field(
        default_factory=list, description="Biomass data sources used"
    )
    area_ha: float = Field(default=0.0, ge=0.0, description="Plot area in hectares")
    sample_count: int = Field(
        default=0, ge=0, description="Number of biomass samples"
    )
    uncertainty_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Estimation uncertainty as percentage",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Estimation confidence"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Estimation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class PlotProfileResponse(BaseModel):
    """Complete plot profile combining all analysis results."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    area_ha: float = Field(default=0.0, ge=0.0, description="Plot area in hectares")
    density: Optional[CanopyDensityResponse] = Field(
        None, description="Latest canopy density analysis"
    )
    classification: Optional[ForestClassificationResponse] = Field(
        None, description="Latest forest type classification"
    )
    historical: Optional[HistoricalCoverResponse] = Field(
        None, description="Historical cover reconstruction"
    )
    height: Optional[CanopyHeightResponse] = Field(
        None, description="Latest canopy height estimation"
    )
    fragmentation: Optional[FragmentationResponse] = Field(
        None, description="Latest fragmentation analysis"
    )
    biomass: Optional[BiomassResponse] = Field(
        None, description="Latest biomass estimation"
    )
    verification: Optional[DeforestationFreeResponse] = Field(
        None, description="Deforestation-free verification"
    )
    fao_forest_status: str = Field(
        default="unknown",
        description="FAO forest status: forest, non_forest, borderline",
    )
    overall_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted average confidence across analyses",
    )
    last_updated: datetime = Field(
        default_factory=_utcnow, description="Last analysis update"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Profile generation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class MetricComparison(BaseModel):
    """Comparison of a single metric between two dates."""

    metric_name: str = Field(..., description="Metric name")
    value_before: float = Field(..., description="Value at earlier date")
    value_after: float = Field(..., description="Value at later date")
    absolute_change: float = Field(..., description="Absolute change")
    relative_change_pct: float = Field(
        default=0.0, description="Relative change as percentage"
    )
    unit: str = Field(default="", description="Measurement unit")

    model_config = ConfigDict(from_attributes=True)


class AnalysisComparisonResponse(BaseModel):
    """Response comparing metrics between two dates."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    date_before: date = Field(..., description="Earlier comparison date")
    date_after: date = Field(..., description="Later comparison date")
    comparisons: List[MetricComparison] = Field(
        default_factory=list, description="Metric comparisons"
    )
    overall_change_direction: str = Field(
        default="stable",
        description="Overall change: improving, degrading, stable",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Comparison confidence"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Comparison timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Report Schemas - Request
# =============================================================================


class GenerateReportRequest(BaseModel):
    """Request to generate a compliance report."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    report_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Report type: eudr_compliance, due_diligence, "
        "risk_assessment, monitoring_summary",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Report output format: json, csv, pdf, xlsx",
    )
    include_evidence: bool = Field(
        default=True,
        description="Include supporting evidence in the report",
    )
    include_maps: bool = Field(
        default=False,
        description="Include map visualizations (PDF/XLSX only)",
    )
    operator_id: Optional[str] = Field(
        None,
        max_length=200,
        description="Operator ID for report attribution",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "report_type": "eudr_compliance",
                    "format": "json",
                    "include_evidence": True,
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        """Validate report type."""
        valid_types = frozenset({
            "eudr_compliance", "due_diligence",
            "risk_assessment", "monitoring_summary",
        })
        v = v.lower().strip()
        if v not in valid_types:
            raise ValueError(
                f"report_type must be one of {sorted(valid_types)}, got '{v}'"
            )
        return v


class BatchReportRequest(BaseModel):
    """Request for batch report generation."""

    reports: List[GenerateReportRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of reports to generate (max 1000 per batch)",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Report Schemas - Response
# =============================================================================


class ComplianceReportResponse(BaseModel):
    """Response from compliance report generation."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    report_id: str = Field(..., description="Unique report identifier")
    plot_id: str = Field(..., description="Plot identifier")
    report_type: str = Field(..., description="Report type")
    format: str = Field(default="json", description="Output format")
    status: str = Field(
        default="generated",
        description="Report status: generating, generated, failed",
    )
    title: str = Field(default="", description="Report title")
    summary: str = Field(default="", description="Executive summary")
    verdict: Optional[DeforestationVerdict] = Field(
        None, description="EUDR compliance verdict if applicable"
    )
    risk_level: Optional[str] = Field(
        None, description="Risk level: negligible, low, standard, high"
    )
    sections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Report sections with headings and content",
    )
    evidence_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Supporting evidence items",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Compliance recommendations",
    )
    operator_id: Optional[str] = Field(
        None, description="Operator ID for attribution"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL for non-JSON formats"
    )
    generated_by: str = Field(
        default="GL-EUDR-FCA-004",
        description="Agent that generated the report",
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources referenced"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Report generation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Analysis Schemas
# =============================================================================


class BatchPlotEntry(BaseModel):
    """Single plot entry for batch analysis."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_wkt: str = Field(
        ...,
        min_length=10,
        max_length=100000,
        description="Plot boundary as WKT POLYGON geometry",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: str) -> str:
        """Validate plot ID format."""
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: str) -> str:
        """Validate WKT polygon format."""
        return _validate_polygon_wkt(v)


class BatchAnalysisRequest(BaseModel):
    """Request for multi-plot batch analysis."""

    plots: List[BatchPlotEntry] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots with polygon boundaries (max 5000)",
    )
    analysis_types: List[str] = Field(
        ...,
        min_length=1,
        description="Analysis types: density, classification, historical, "
        "verification, height, fragmentation, biomass",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity for verification context",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plots": [
                        {
                            "plot_id": "plot-gh-001",
                            "polygon_wkt": "POLYGON((-1.624 6.688, -1.623 6.689, -1.622 6.687, -1.624 6.688))",
                        },
                    ],
                    "analysis_types": ["density", "verification"],
                    "commodity": "cocoa",
                }
            ]
        },
    )

    @field_validator("analysis_types")
    @classmethod
    def validate_analysis_types(cls, v: List[str]) -> List[str]:
        """Validate analysis type identifiers."""
        valid_types = frozenset({
            "density", "classification", "historical",
            "verification", "height", "fragmentation", "biomass",
        })
        normalized = []
        for t in v:
            t = t.lower().strip()
            if t not in valid_types:
                raise ValueError(
                    f"Invalid analysis type '{t}'. Valid types: {sorted(valid_types)}"
                )
            normalized.append(t)
        return normalized


class BatchPlotResult(BaseModel):
    """Result for a single plot in a batch analysis."""

    plot_id: str = Field(..., description="Plot identifier")
    status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETED, description="Analysis status"
    )
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-analysis-type results",
    )
    error: Optional[str] = Field(
        None, description="Error message if failed"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchAnalysisResponse(BaseModel):
    """Response from batch analysis submission."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    batch_id: str = Field(..., description="Unique batch job identifier")
    status: AnalysisStatus = Field(
        default=AnalysisStatus.PENDING, description="Batch job status"
    )
    total_plots: int = Field(..., ge=0, description="Total plots submitted")
    completed_plots: int = Field(
        default=0, ge=0, description="Plots completed"
    )
    failed_plots: int = Field(default=0, ge=0, description="Plots failed")
    analysis_types: List[str] = Field(
        default_factory=list, description="Analysis types requested"
    )
    commodity: str = Field(default="", description="EUDR commodity")
    results: List[BatchPlotResult] = Field(
        default_factory=list,
        description="Per-plot results (populated when completed)",
    )
    progress_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Completion percentage"
    )
    estimated_completion_seconds: Optional[float] = Field(
        None, description="Estimated time to complete in seconds"
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow, description="Submission timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Completion timestamp"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Dashboard / Summary Schemas
# =============================================================================


class AnalysisSummaryResponse(BaseModel):
    """Summary statistics across all analyses."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    total_plots_analyzed: int = Field(
        default=0, ge=0, description="Total plots analyzed"
    )
    total_area_ha: float = Field(
        default=0.0, ge=0.0, description="Total area analyzed in hectares"
    )
    mean_density_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Mean canopy density"
    )
    mean_height_m: float = Field(
        default=0.0, ge=0.0, description="Mean canopy height"
    )
    mean_biomass_t_ha: float = Field(
        default=0.0, ge=0.0, description="Mean above-ground biomass"
    )
    deforestation_free_count: int = Field(
        default=0, ge=0, description="Plots verified deforestation-free"
    )
    deforestation_detected_count: int = Field(
        default=0, ge=0, description="Plots with deforestation detected"
    )
    inconclusive_count: int = Field(
        default=0, ge=0, description="Plots with inconclusive results"
    )
    by_commodity: Dict[str, int] = Field(
        default_factory=dict, description="Analysis counts by commodity"
    )
    by_forest_type: Dict[str, int] = Field(
        default_factory=dict, description="Plot counts by forest type"
    )
    by_verdict: Dict[str, int] = Field(
        default_factory=dict, description="Plot counts by verdict"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Summary timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


class DashboardResponse(BaseModel):
    """Dashboard overview with key metrics and recent activity."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    summary: AnalysisSummaryResponse = Field(
        ..., description="Overall analysis summary"
    )
    recent_analyses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent analysis results (last 10)",
    )
    recent_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent deforestation alerts (last 10)",
    )
    pending_verifications: int = Field(
        default=0, ge=0, description="Pending verification count"
    )
    system_health: Dict[str, Any] = Field(
        default_factory=dict,
        description="System health metrics",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Dashboard generation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    agent_id: str = Field(default="GL-EUDR-FCA-004")
    agent_name: str = Field(default="EUDR Forest Cover Analysis Agent")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


class VersionResponse(BaseModel):
    """Version information response."""

    agent_id: str = Field(default="GL-EUDR-FCA-004")
    agent_name: str = Field(default="EUDR Forest Cover Analysis Agent")
    version: str = Field(default="1.0.0")
    api_version: str = Field(default="v1")
    engines: List[str] = Field(
        default_factory=lambda: [
            "canopy_density",
            "forest_classification",
            "historical_reconstruction",
            "deforestation_verification",
            "canopy_height",
            "fragmentation_analysis",
            "biomass_estimation",
        ],
        description="Available analysis engines",
    )
    supported_commodities: List[str] = Field(
        default_factory=lambda: [c.value for c in EUDRCommodity],
        description="Supported EUDR commodities",
    )
    fao_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "canopy_cover_pct": 10.0,
            "tree_height_m": 5.0,
            "min_area_ha": 0.5,
        },
        description="FAO forest definition thresholds",
    )
    eudr_cutoff_date: str = Field(
        default="2020-12-31", description="EUDR deforestation cutoff date"
    )
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "DensityMethod",
    "ClassificationMethod",
    "EUDRCommodity",
    "HeightSource",
    "BiomassSource",
    "ReportFormat",
    "ForestType",
    "DeforestationVerdict",
    "AnalysisStatus",
    # Pagination
    "PaginatedMeta",
    "PaginatedResponse",
    "PaginationParams",
    # Response wrappers
    "ApiResponse",
    "ErrorResponse",
    # Density
    "AnalyzeDensityRequest",
    "BatchDensityRequest",
    "CompareDensityRequest",
    "CanopyDensityResponse",
    "DensityComparisonResponse",
    "DensityHistoryEntry",
    "DensityHistoryResponse",
    # Classification
    "ClassifyForestRequest",
    "BatchClassifyRequest",
    "ForestClassificationResponse",
    "ForestTypeInfo",
    "ForestTypesListResponse",
    # Historical Reconstruction
    "ReconstructHistoryRequest",
    "BatchReconstructRequest",
    "CompareHistoricalRequest",
    "HistoricalCoverResponse",
    "HistoricalComparisonResponse",
    "DataSourceInfo",
    "DataSourcesResponse",
    # Verification
    "VerifyDeforestationFreeRequest",
    "BatchVerifyRequest",
    "CompletePlotAnalysisRequest",
    "EvidenceItem",
    "DeforestationFreeResponse",
    # Analysis (Height, Fragmentation, Biomass)
    "EstimateHeightRequest",
    "AnalyzeFragmentationRequest",
    "EstimateBiomassRequest",
    "CompareAnalysisRequest",
    "CanopyHeightResponse",
    "FragmentationResponse",
    "BiomassResponse",
    "PlotProfileResponse",
    "MetricComparison",
    "AnalysisComparisonResponse",
    # Reports
    "GenerateReportRequest",
    "BatchReportRequest",
    "ComplianceReportResponse",
    # Batch
    "BatchPlotEntry",
    "BatchAnalysisRequest",
    "BatchPlotResult",
    "BatchAnalysisResponse",
    # Dashboard
    "AnalysisSummaryResponse",
    "DashboardResponse",
    # Health / Version
    "HealthResponse",
    "VersionResponse",
]
