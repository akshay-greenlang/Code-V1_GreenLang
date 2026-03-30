# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-003 Satellite Monitoring

Pydantic v2 request/response models specific to the REST API layer for
satellite monitoring operations including imagery search, spectral analysis,
baseline management, change detection, multi-source fusion, continuous
monitoring, alert management, evidence packaging, and batch analysis.

Core domain models are imported from the main engine modules; this file
defines API-level request wrappers, paginated list responses, and
evidence/batch schemas.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-003 Satellite Monitoring Agent (GL-EUDR-SAT-003)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ConfigDict, Field, field_validator

from greenlang.schemas import GreenLangBase, utcnow

# =============================================================================
# Enumerations (string literals)
# =============================================================================

_VALID_SOURCES = frozenset({
    "sentinel2", "landsat8", "landsat9", "gfw", "planet", "all",
})

_VALID_INDEX_TYPES = frozenset({
    "ndvi", "evi", "nbr", "ndmi", "savi", "msavi",
})

_VALID_COMMODITIES = frozenset({
    "cocoa", "coffee", "palm_oil", "soy", "rubber", "cattle", "wood",
})

_VALID_BIOMES = frozenset({
    "tropical_moist", "tropical_dry", "subtropical",
    "temperate", "boreal", "mangrove", "savanna",
})

_VALID_PRIORITIES = frozenset({"low", "medium", "high", "critical"})

_VALID_SEVERITIES = frozenset({"low", "medium", "high", "critical"})

_VALID_INTERVALS = frozenset({
    "daily", "weekly", "biweekly", "monthly", "quarterly",
})

_VALID_FORMATS = frozenset({"json", "csv", "pdf"})

_VALID_ANALYSIS_LEVELS = frozenset({"quick", "standard", "deep"})

_VALID_BANDS = frozenset({
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12", "SCL",
    "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7",
    "QA_PIXEL",
})

# =============================================================================
# Pagination
# =============================================================================

class PaginatedMeta(GreenLangBase):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")

class PaginatedResponse(GreenLangBase):
    """Generic paginated response wrapper."""

    items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Page of result items"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)

class PaginationParams(GreenLangBase):
    """Standard pagination query parameters."""

    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")

# =============================================================================
# Response Wrappers
# =============================================================================

class ApiResponse(GreenLangBase):
    """Standard API success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")

# =============================================================================
# Imagery Schemas
# =============================================================================

class SearchScenesApiRequest(GreenLangBase):
    """Request to search available satellite scenes for a polygon area."""

    polygon_vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        max_length=100000,
        description="Polygon boundary vertices as (lat, lon) tuples",
    )
    start_date: date = Field(
        ...,
        description="Search window start date (YYYY-MM-DD)",
    )
    end_date: date = Field(
        ...,
        description="Search window end date (YYYY-MM-DD)",
    )
    source: str = Field(
        default="all",
        description="Satellite source: sentinel2, landsat8, landsat9, gfw, planet, all",
    )
    cloud_cover_max: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable cloud cover percentage",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "polygon_vertices": [
                        [6.6885, -1.6244],
                        [6.6895, -1.6234],
                        [6.6875, -1.6224],
                        [6.6885, -1.6244],
                    ],
                    "start_date": "2024-01-01",
                    "end_date": "2024-06-30",
                    "source": "sentinel2",
                    "cloud_cover_max": 20.0,
                }
            ]
        },
    )

    @field_validator("polygon_vertices")
    @classmethod
    def validate_polygon_vertices(
        cls, v: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Validate polygon vertex coordinates are within WGS84 bounds."""
        for i, (lat, lon) in enumerate(v):
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Vertex {i}: latitude {lat} out of range [-90, 90]"
                )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Vertex {i}: longitude {lon} out of range [-180, 180]"
                )
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate satellite source."""
        v = v.lower().strip()
        if v not in _VALID_SOURCES:
            raise ValueError(
                f"source must be one of {sorted(_VALID_SOURCES)}, got '{v}'"
            )
        return v

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: date, info) -> date:
        """Validate end_date is after start_date."""
        start = info.data.get("start_date")
        if start and v < start:
            raise ValueError(
                f"end_date ({v}) must be on or after start_date ({start})"
            )
        return v

class SceneMetadataResponse(GreenLangBase):
    """Metadata for a single satellite scene."""

    scene_id: str = Field(..., description="Unique scene identifier")
    source: str = Field(..., description="Satellite source (sentinel2, landsat8, etc)")
    acquisition_date: datetime = Field(..., description="Scene acquisition timestamp")
    cloud_cover_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Cloud cover percentage"
    )
    spatial_resolution_m: float = Field(
        ..., ge=0.0, description="Spatial resolution in metres"
    )
    tile_id: str = Field(default="", description="MGRS tile ID or WRS path/row")
    bounds: Dict[str, float] = Field(
        default_factory=dict,
        description="Bounding box: min_lat, max_lat, min_lon, max_lon",
    )
    available_bands: List[str] = Field(
        default_factory=list, description="Available spectral bands"
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall quality score"
    )
    file_size_mb: Optional[float] = Field(
        None, ge=0.0, description="File size in megabytes"
    )
    processing_level: str = Field(
        default="L2A", description="Processing level (L1C, L2A, etc)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)

class SearchScenesApiResponse(GreenLangBase):
    """Response from satellite scene search."""

    total_scenes: int = Field(..., ge=0, description="Total scenes found")
    source_filter: str = Field(default="all", description="Source filter applied")
    cloud_cover_max: float = Field(
        default=20.0, description="Cloud cover filter applied"
    )
    scenes: List[SceneMetadataResponse] = Field(
        default_factory=list, description="Matching scenes sorted by date"
    )
    search_area_ha: float = Field(
        default=0.0, ge=0.0, description="Search polygon area in hectares"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Search processing time in ms"
    )
    searched_at: datetime = Field(
        default_factory=utcnow, description="Search timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class DownloadBandsApiRequest(GreenLangBase):
    """Request to download specific bands from a satellite scene."""

    scene_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Scene identifier to download from",
    )
    bands: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of band identifiers to download",
    )
    polygon_clip: Optional[List[Tuple[float, float]]] = Field(
        None,
        description="Optional polygon to clip the download area",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "scene_id": "S2A_MSIL2A_20240315T103021_N0510_R108_T30NUN",
                    "bands": ["B04", "B08", "B11"],
                }
            ]
        },
    )

    @field_validator("bands")
    @classmethod
    def validate_bands(cls, v: List[str]) -> List[str]:
        """Validate band identifiers."""
        normalized = []
        for band in v:
            band = band.strip().upper()
            if band not in _VALID_BANDS:
                raise ValueError(
                    f"Invalid band '{band}'. Valid bands: {sorted(_VALID_BANDS)}"
                )
            normalized.append(band)
        return normalized

class DownloadBandsApiResponse(GreenLangBase):
    """Response from band download request."""

    scene_id: str = Field(..., description="Scene identifier")
    bands_downloaded: List[str] = Field(
        default_factory=list, description="Successfully downloaded bands"
    )
    band_metadata: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-band metadata (resolution, data type, dimensions)",
    )
    total_size_mb: float = Field(
        default=0.0, ge=0.0, description="Total download size in MB"
    )
    download_status: str = Field(
        default="completed", description="Download status: completed, partial, failed"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Download processing time in ms"
    )
    downloaded_at: datetime = Field(
        default_factory=utcnow, description="Download timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class AvailabilityResponse(GreenLangBase):
    """Data availability summary for a location."""

    lat: float = Field(..., description="Query latitude")
    lon: float = Field(..., description="Query longitude")
    start_date: date = Field(..., description="Start date queried")
    end_date: date = Field(..., description="End date queried")
    total_scenes: int = Field(default=0, ge=0, description="Total available scenes")
    by_source: Dict[str, int] = Field(
        default_factory=dict,
        description="Scene counts per source (sentinel2, landsat8, etc)",
    )
    cloud_free_scenes: int = Field(
        default=0, ge=0, description="Scenes with <20% cloud cover"
    )
    best_scene_id: Optional[str] = Field(
        None, description="ID of scene with lowest cloud cover"
    )
    queried_at: datetime = Field(
        default_factory=utcnow, description="Query timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Analysis Schemas - Spectral Index
# =============================================================================

class CalculateIndexApiRequest(GreenLangBase):
    """Request to calculate a spectral vegetation index."""

    red_band: List[List[float]] = Field(
        ...,
        min_length=1,
        description="Red band reflectance values as 2D array",
    )
    nir_band: List[List[float]] = Field(
        ...,
        min_length=1,
        description="Near-infrared band reflectance values as 2D array",
    )
    index_type: str = Field(
        default="ndvi",
        description="Spectral index type: ndvi, evi, nbr, ndmi, savi, msavi",
    )
    biome: Optional[str] = Field(
        None,
        description="Biome context for threshold classification",
    )
    swir_band: Optional[List[List[float]]] = Field(
        None,
        description="SWIR band (required for NBR and NDMI calculations)",
    )
    blue_band: Optional[List[List[float]]] = Field(
        None,
        description="Blue band (required for EVI calculation)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "red_band": [[0.08, 0.09], [0.07, 0.10]],
                    "nir_band": [[0.45, 0.50], [0.48, 0.42]],
                    "index_type": "ndvi",
                    "biome": "tropical_moist",
                }
            ]
        },
    )

    @field_validator("index_type")
    @classmethod
    def validate_index_type(cls, v: str) -> str:
        """Validate spectral index type."""
        v = v.lower().strip()
        if v not in _VALID_INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(_VALID_INDEX_TYPES)}, got '{v}'"
            )
        return v

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

class SpectralIndexApiResponse(GreenLangBase):
    """Response from spectral index calculation."""

    index_type: str = Field(..., description="Computed index type")
    mean_value: float = Field(..., description="Mean index value across pixels")
    min_value: float = Field(..., description="Minimum index value")
    max_value: float = Field(..., description="Maximum index value")
    std_dev: float = Field(default=0.0, ge=0.0, description="Standard deviation")
    pixel_count: int = Field(default=0, ge=0, description="Number of pixels processed")
    classification: str = Field(
        default="unknown",
        description="Vegetation classification: dense_forest, forest, degraded, non_forest, water",
    )
    biome_threshold_used: Optional[float] = Field(
        None, description="Biome-specific threshold applied"
    )
    index_values: Optional[List[List[float]]] = Field(
        None, description="Full 2D index array (returned for small areas)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Analysis Schemas - Baseline
# =============================================================================

class EstablishBaselineApiRequest(GreenLangBase):
    """Request to establish a Dec 2020 baseline snapshot for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        max_length=100000,
        description="Polygon boundary vertices as (lat, lon) tuples",
    )
    commodity: str = Field(
        ...,
        max_length=50,
        description="EUDR commodity: cocoa, coffee, palm_oil, soy, rubber, cattle, wood",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    biome: Optional[str] = Field(
        None,
        description="Biome classification for threshold adjustment",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_vertices": [
                        [6.6885, -1.6244],
                        [6.6895, -1.6234],
                        [6.6875, -1.6224],
                        [6.6885, -1.6244],
                    ],
                    "commodity": "cocoa",
                    "country_code": "GH",
                    "biome": "tropical_moist",
                }
            ]
        },
    )

    @field_validator("polygon_vertices")
    @classmethod
    def validate_polygon_vertices(
        cls, v: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Validate polygon vertex coordinates."""
        for i, (lat, lon) in enumerate(v):
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Vertex {i}: latitude {lat} out of range [-90, 90]"
                )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Vertex {i}: longitude {lon} out of range [-180, 180]"
                )
        return v

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in _VALID_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {sorted(_VALID_COMMODITIES)}, got '{v}'"
            )
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

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

class BaselineApiResponse(GreenLangBase):
    """Response from baseline establishment."""

    baseline_id: str = Field(..., description="Unique baseline snapshot identifier")
    plot_id: str = Field(..., description="Plot identifier")
    cutoff_date: str = Field(
        default="2020-12-31", description="EUDR cutoff date used"
    )
    baseline_ndvi: float = Field(
        ..., ge=-1.0, le=1.0, description="Baseline mean NDVI value"
    )
    baseline_evi: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Baseline mean EVI value"
    )
    forest_cover_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Forest cover percentage at baseline"
    )
    canopy_density_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Canopy density percentage"
    )
    area_ha: float = Field(
        ..., ge=0.0, description="Plot area in hectares"
    )
    commodity: str = Field(..., description="EUDR commodity")
    country_code: str = Field(..., description="Country code")
    biome: Optional[str] = Field(None, description="Biome classification")
    scenes_used: int = Field(
        default=0, ge=0, description="Number of scenes composited"
    )
    cloud_free_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cloud-free coverage percentage in composited baseline",
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Satellite sources used"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    established_at: datetime = Field(
        default_factory=utcnow, description="Baseline establishment timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Analysis Schemas - Change Detection
# =============================================================================

class DetectChangeApiRequest(GreenLangBase):
    """Request to run deforestation change detection on a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        max_length=100000,
        description="Polygon boundary vertices as (lat, lon) tuples",
    )
    commodity: str = Field(
        ...,
        max_length=50,
        description="EUDR commodity identifier",
    )
    analysis_date: Optional[date] = Field(
        None,
        description="Analysis target date (defaults to today)",
    )
    analysis_level: str = Field(
        default="standard",
        description="Analysis depth: quick, standard, deep",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_vertices": [
                        [6.6885, -1.6244],
                        [6.6895, -1.6234],
                        [6.6875, -1.6224],
                        [6.6885, -1.6244],
                    ],
                    "commodity": "cocoa",
                    "analysis_date": "2025-03-01",
                    "analysis_level": "standard",
                }
            ]
        },
    )

    @field_validator("polygon_vertices")
    @classmethod
    def validate_polygon_vertices(
        cls, v: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Validate polygon vertex coordinates."""
        for i, (lat, lon) in enumerate(v):
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Vertex {i}: latitude {lat} out of range [-90, 90]"
                )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Vertex {i}: longitude {lon} out of range [-180, 180]"
                )
        return v

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in _VALID_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {sorted(_VALID_COMMODITIES)}, got '{v}'"
            )
        return v

    @field_validator("analysis_level")
    @classmethod
    def validate_analysis_level(cls, v: str) -> str:
        """Validate analysis level."""
        v = v.lower().strip()
        if v not in _VALID_ANALYSIS_LEVELS:
            raise ValueError(
                f"analysis_level must be one of {sorted(_VALID_ANALYSIS_LEVELS)}, got '{v}'"
            )
        return v

class ChangeDetectionApiResponse(GreenLangBase):
    """Response from change detection analysis."""

    detection_id: str = Field(..., description="Unique detection identifier")
    plot_id: str = Field(..., description="Plot identifier")
    deforestation_detected: bool = Field(
        ..., description="Whether deforestation detected post-cutoff"
    )
    change_classification: str = Field(
        default="no_change",
        description="Classification: no_change, degradation, deforestation, regrowth",
    )
    ndvi_baseline: float = Field(
        ..., ge=-1.0, le=1.0, description="Baseline NDVI value"
    )
    ndvi_current: float = Field(
        ..., ge=-1.0, le=1.0, description="Current NDVI value"
    )
    ndvi_delta: float = Field(
        ..., ge=-2.0, le=2.0, description="NDVI change (current - baseline)"
    )
    forest_loss_ha: float = Field(
        default=0.0, ge=0.0, description="Estimated forest loss area in hectares"
    )
    forest_loss_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Forest loss as percentage of plot area",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score"
    )
    analysis_level: str = Field(
        default="standard", description="Analysis depth applied"
    )
    analysis_date: date = Field(..., description="Analysis target date")
    cutoff_date: str = Field(default="2020-12-31", description="EUDR cutoff date")
    data_sources: List[str] = Field(
        default_factory=list, description="Satellite data sources used"
    )
    change_pixels: Optional[int] = Field(
        None, ge=0, description="Number of pixels showing change"
    )
    total_pixels: Optional[int] = Field(
        None, ge=0, description="Total pixels in analysis area"
    )
    seasonal_adjusted: bool = Field(
        default=False, description="Whether seasonal adjustment was applied"
    )
    alerts_generated: int = Field(
        default=0, ge=0, description="Number of alerts generated"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Analysis processing time in ms"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    detected_at: datetime = Field(
        default_factory=utcnow, description="Detection timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Analysis Schemas - Multi-Source Fusion
# =============================================================================

class FusionSourceResult(GreenLangBase):
    """Single source result for fusion input."""

    source: str = Field(..., description="Data source: sentinel2, landsat, gfw")
    ndvi_delta: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="NDVI delta from this source"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Source confidence"
    )
    deforestation_detected: bool = Field(
        default=False, description="Detection result from this source"
    )
    forest_loss_ha: float = Field(
        default=0.0, ge=0.0, description="Forest loss from this source"
    )
    alert_count: int = Field(
        default=0, ge=0, description="Alert count (GFW specific)"
    )
    acquisition_date: Optional[date] = Field(
        None, description="Most recent acquisition date"
    )

    model_config = ConfigDict(extra="forbid")

class FusionApiRequest(GreenLangBase):
    """Request to run multi-source data fusion analysis."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Plot identifier for fusion context",
    )
    sentinel2_result: Optional[FusionSourceResult] = Field(
        None, description="Sentinel-2 analysis result"
    )
    landsat_result: Optional[FusionSourceResult] = Field(
        None, description="Landsat analysis result"
    )
    gfw_result: Optional[FusionSourceResult] = Field(
        None, description="GFW alert data result"
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Optional custom fusion weights overriding config defaults",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "sentinel2_result": {
                        "source": "sentinel2",
                        "ndvi_delta": -0.18,
                        "confidence": 0.85,
                        "deforestation_detected": True,
                        "forest_loss_ha": 1.2,
                    },
                    "landsat_result": {
                        "source": "landsat",
                        "ndvi_delta": -0.16,
                        "confidence": 0.78,
                        "deforestation_detected": True,
                        "forest_loss_ha": 1.1,
                    },
                    "gfw_result": {
                        "source": "gfw",
                        "confidence": 0.70,
                        "deforestation_detected": True,
                        "forest_loss_ha": 1.0,
                        "alert_count": 5,
                    },
                }
            ]
        },
    )

    @field_validator("custom_weights")
    @classmethod
    def validate_custom_weights(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """Validate custom fusion weights sum to 1.0."""
        if v is None:
            return v
        for key, weight in v.items():
            if not (0.0 <= weight <= 1.0):
                raise ValueError(
                    f"Weight for '{key}' must be in [0.0, 1.0], got {weight}"
                )
        weight_sum = sum(v.values())
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(
                f"Custom weights must sum to 1.0, got {weight_sum:.4f}"
            )
        return v

class FusionApiResponse(GreenLangBase):
    """Response from multi-source fusion analysis."""

    fusion_id: str = Field(..., description="Unique fusion result identifier")
    plot_id: str = Field(..., description="Plot identifier")
    fused_deforestation_detected: bool = Field(
        ..., description="Fused detection result"
    )
    fused_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Fused confidence score"
    )
    fused_ndvi_delta: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="Weighted fused NDVI delta"
    )
    fused_forest_loss_ha: float = Field(
        default=0.0, ge=0.0, description="Fused forest loss estimate in hectares"
    )
    sources_used: int = Field(
        default=0, ge=0, description="Number of sources in fusion"
    )
    source_agreement: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Agreement ratio across sources (1.0 = all agree)",
    )
    weights_applied: Dict[str, float] = Field(
        default_factory=dict, description="Fusion weights used"
    )
    per_source_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-source summary for audit trail",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    fused_at: datetime = Field(
        default_factory=utcnow, description="Fusion timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Monitoring Schemas
# =============================================================================

class CreateMonitoringApiRequest(GreenLangBase):
    """Request to create a continuous monitoring schedule for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier to monitor",
    )
    polygon_vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        max_length=100000,
        description="Polygon boundary vertices as (lat, lon) tuples",
    )
    commodity: str = Field(
        ...,
        max_length=50,
        description="EUDR commodity identifier",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    interval: str = Field(
        default="monthly",
        description="Monitoring interval: daily, weekly, biweekly, monthly, quarterly",
    )
    priority: str = Field(
        default="medium",
        description="Monitoring priority: low, medium, high, critical",
    )
    analysis_level: str = Field(
        default="standard",
        description="Analysis depth: quick, standard, deep",
    )
    alert_on_change: bool = Field(
        default=True,
        description="Generate alerts on detected changes",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "polygon_vertices": [
                        [6.6885, -1.6244],
                        [6.6895, -1.6234],
                        [6.6875, -1.6224],
                        [6.6885, -1.6244],
                    ],
                    "commodity": "cocoa",
                    "country_code": "GH",
                    "interval": "monthly",
                    "priority": "high",
                }
            ]
        },
    )

    @field_validator("polygon_vertices")
    @classmethod
    def validate_polygon_vertices(
        cls, v: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Validate polygon vertex coordinates."""
        for i, (lat, lon) in enumerate(v):
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Vertex {i}: latitude {lat} out of range [-90, 90]"
                )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Vertex {i}: longitude {lon} out of range [-180, 180]"
                )
        return v

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in _VALID_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {sorted(_VALID_COMMODITIES)}, got '{v}'"
            )
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate monitoring interval."""
        v = v.lower().strip()
        if v not in _VALID_INTERVALS:
            raise ValueError(
                f"interval must be one of {sorted(_VALID_INTERVALS)}, got '{v}'"
            )
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate monitoring priority."""
        v = v.lower().strip()
        if v not in _VALID_PRIORITIES:
            raise ValueError(
                f"priority must be one of {sorted(_VALID_PRIORITIES)}, got '{v}'"
            )
        return v

    @field_validator("analysis_level")
    @classmethod
    def validate_analysis_level(cls, v: str) -> str:
        """Validate analysis level."""
        v = v.lower().strip()
        if v not in _VALID_ANALYSIS_LEVELS:
            raise ValueError(
                f"analysis_level must be one of {sorted(_VALID_ANALYSIS_LEVELS)}, got '{v}'"
            )
        return v

class UpdateMonitoringApiRequest(GreenLangBase):
    """Request to update an existing monitoring schedule."""

    interval: Optional[str] = Field(
        None, description="Updated monitoring interval"
    )
    priority: Optional[str] = Field(
        None, description="Updated monitoring priority"
    )
    analysis_level: Optional[str] = Field(
        None, description="Updated analysis depth"
    )
    active: Optional[bool] = Field(
        None, description="Enable or disable the schedule"
    )
    alert_on_change: Optional[bool] = Field(
        None, description="Enable or disable alert generation"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "interval": "weekly",
                    "priority": "critical",
                    "active": True,
                }
            ]
        },
    )

    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: Optional[str]) -> Optional[str]:
        """Validate monitoring interval."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in _VALID_INTERVALS:
            raise ValueError(
                f"interval must be one of {sorted(_VALID_INTERVALS)}, got '{v}'"
            )
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: Optional[str]) -> Optional[str]:
        """Validate monitoring priority."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in _VALID_PRIORITIES:
            raise ValueError(
                f"priority must be one of {sorted(_VALID_PRIORITIES)}, got '{v}'"
            )
        return v

    @field_validator("analysis_level")
    @classmethod
    def validate_analysis_level(cls, v: Optional[str]) -> Optional[str]:
        """Validate analysis level."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in _VALID_ANALYSIS_LEVELS:
            raise ValueError(
                f"analysis_level must be one of {sorted(_VALID_ANALYSIS_LEVELS)}, got '{v}'"
            )
        return v

class MonitoringScheduleResponse(GreenLangBase):
    """Response with monitoring schedule details."""

    schedule_id: str = Field(..., description="Unique schedule identifier")
    plot_id: str = Field(..., description="Monitored plot identifier")
    commodity: str = Field(..., description="EUDR commodity")
    country_code: str = Field(..., description="Country code")
    interval: str = Field(..., description="Monitoring interval")
    priority: str = Field(..., description="Monitoring priority")
    analysis_level: str = Field(default="standard", description="Analysis depth")
    active: bool = Field(default=True, description="Whether schedule is active")
    alert_on_change: bool = Field(
        default=True, description="Alert generation enabled"
    )
    next_execution: Optional[datetime] = Field(
        None, description="Next scheduled execution timestamp"
    )
    last_execution: Optional[datetime] = Field(
        None, description="Last execution timestamp"
    )
    total_executions: int = Field(
        default=0, ge=0, description="Total executions performed"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Schedule creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=utcnow, description="Last update timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class MonitoringResultResponse(GreenLangBase):
    """Response with a single monitoring execution result."""

    result_id: str = Field(..., description="Unique result identifier")
    schedule_id: str = Field(..., description="Parent schedule identifier")
    plot_id: str = Field(..., description="Plot identifier")
    execution_date: datetime = Field(..., description="Execution timestamp")
    deforestation_detected: bool = Field(
        ..., description="Whether deforestation detected"
    )
    change_classification: str = Field(
        default="no_change",
        description="Classification: no_change, degradation, deforestation, regrowth",
    )
    ndvi_current: float = Field(
        ..., ge=-1.0, le=1.0, description="Current NDVI value"
    )
    ndvi_delta: float = Field(
        ..., ge=-2.0, le=2.0, description="NDVI delta from baseline"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence"
    )
    forest_loss_ha: float = Field(
        default=0.0, ge=0.0, description="Forest loss in hectares"
    )
    alerts_generated: int = Field(
        default=0, ge=0, description="Alerts generated from this execution"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Sources used"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in ms"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)

class MonitoringExecuteRequest(GreenLangBase):
    """Request to trigger manual monitoring execution."""

    schedule_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Schedule identifier to execute",
    )

    model_config = ConfigDict(extra="forbid")

# =============================================================================
# Alert Schemas
# =============================================================================

class AlertDetailResponse(GreenLangBase):
    """Detailed satellite monitoring alert."""

    alert_id: str = Field(..., description="Unique alert identifier")
    plot_id: str = Field(..., description="Plot identifier")
    schedule_id: Optional[str] = Field(
        None, description="Monitoring schedule that triggered the alert"
    )
    severity: str = Field(
        ..., description="Alert severity: low, medium, high, critical"
    )
    alert_type: str = Field(
        ...,
        description="Alert type: deforestation, degradation, anomaly, data_gap",
    )
    title: str = Field(..., description="Alert title")
    description: str = Field(default="", description="Detailed alert description")
    commodity: str = Field(default="", description="EUDR commodity")
    country_code: str = Field(default="", description="Country code")
    ndvi_delta: Optional[float] = Field(
        None, ge=-2.0, le=2.0, description="NDVI change that triggered alert"
    )
    forest_loss_ha: float = Field(
        default=0.0, ge=0.0, description="Estimated forest loss in hectares"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Detection confidence"
    )
    acknowledged: bool = Field(
        default=False, description="Whether alert has been acknowledged"
    )
    acknowledged_by: Optional[str] = Field(
        None, description="User who acknowledged the alert"
    )
    acknowledged_at: Optional[datetime] = Field(
        None, description="Acknowledgement timestamp"
    )
    acknowledgement_notes: Optional[str] = Field(
        None, description="Notes from acknowledgement"
    )
    detection_date: date = Field(..., description="Date of detected change")
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources that contributed"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    created_at: datetime = Field(
        default_factory=utcnow, description="Alert creation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class AlertListResponse(GreenLangBase):
    """Paginated list of satellite monitoring alerts."""

    alerts: List[AlertDetailResponse] = Field(
        default_factory=list, description="Alert items"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)

class AcknowledgeAlertRequest(GreenLangBase):
    """Request to acknowledge a satellite alert."""

    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional acknowledgement notes",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "notes": "Verified via field inspection. Alert is a false positive due to seasonal leaf drop.",
                }
            ]
        },
    )

class AlertSummaryResponse(GreenLangBase):
    """Alert summary statistics."""

    total_alerts: int = Field(default=0, ge=0, description="Total alerts")
    unacknowledged: int = Field(
        default=0, ge=0, description="Unacknowledged alert count"
    )
    acknowledged: int = Field(
        default=0, ge=0, description="Acknowledged alert count"
    )
    by_severity: Dict[str, int] = Field(
        default_factory=lambda: {
            "critical": 0, "high": 0, "medium": 0, "low": 0,
        },
        description="Alert counts by severity",
    )
    by_alert_type: Dict[str, int] = Field(
        default_factory=lambda: {
            "deforestation": 0, "degradation": 0, "anomaly": 0, "data_gap": 0,
        },
        description="Alert counts by type",
    )
    by_commodity: Dict[str, int] = Field(
        default_factory=dict, description="Alert counts by commodity"
    )
    by_country: Dict[str, int] = Field(
        default_factory=dict, description="Alert counts by country"
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average confidence across alerts"
    )
    total_forest_loss_ha: float = Field(
        default=0.0, ge=0.0, description="Total forest loss across all alerts"
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Summary generation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Evidence Schemas
# =============================================================================

class GenerateEvidenceApiRequest(GreenLangBase):
    """Request to generate an evidence package for a plot."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Plot identifier for evidence compilation",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Operator identifier for report attribution",
    )
    format: str = Field(
        default="json",
        description="Evidence output format: json, csv, pdf",
    )
    include_time_series: bool = Field(
        default=True,
        description="Include historical NDVI time series data",
    )
    include_imagery_refs: bool = Field(
        default=True,
        description="Include satellite imagery references",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "plot_id": "plot-gh-001",
                    "operator_id": "OP-GH-001",
                    "format": "json",
                    "include_time_series": True,
                }
            ]
        },
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate evidence format."""
        v = v.lower().strip()
        if v not in _VALID_FORMATS:
            raise ValueError(
                f"format must be one of {sorted(_VALID_FORMATS)}, got '{v}'"
            )
        return v

class EvidencePackageResponse(GreenLangBase):
    """Response with generated evidence package."""

    package_id: str = Field(..., description="Unique evidence package identifier")
    plot_id: str = Field(..., description="Plot identifier")
    operator_id: str = Field(..., description="Operator identifier")
    format: str = Field(default="json", description="Output format")
    status: str = Field(
        default="generated",
        description="Package status: generating, generated, failed",
    )
    baseline_snapshot: Optional[Dict[str, Any]] = Field(
        None, description="Baseline NDVI/forest cover snapshot"
    )
    change_detections: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical change detection results",
    )
    time_series: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="NDVI time series data points (date, ndvi, source)",
    )
    imagery_references: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Satellite imagery references (scene_id, date, source, cloud_cover)",
    )
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Related alerts for the plot",
    )
    compliance_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="EUDR compliance assessment summary",
    )
    total_monitoring_events: int = Field(
        default=0, ge=0, description="Total monitoring events in package"
    )
    date_range_start: Optional[date] = Field(
        None, description="Earliest data point date"
    )
    date_range_end: Optional[date] = Field(
        None, description="Latest data point date"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL for PDF/CSV format"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    generated_at: datetime = Field(
        default_factory=utcnow, description="Package generation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Batch Analysis Schemas
# =============================================================================

class BatchPlotEntry(GreenLangBase):
    """Single plot entry in a batch analysis request."""

    plot_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique plot identifier",
    )
    polygon_vertices: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        max_length=100000,
        description="Polygon boundary vertices as (lat, lon) tuples",
    )
    commodity: str = Field(
        ...,
        max_length=50,
        description="EUDR commodity identifier",
    )
    country_code: str = Field(
        default="",
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("polygon_vertices")
    @classmethod
    def validate_polygon_vertices(
        cls, v: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Validate polygon vertex coordinates."""
        for i, (lat, lon) in enumerate(v):
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Vertex {i}: latitude {lat} out of range [-90, 90]"
                )
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Vertex {i}: longitude {lon} out of range [-180, 180]"
                )
        return v

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in _VALID_COMMODITIES:
            raise ValueError(
                f"commodity must be one of {sorted(_VALID_COMMODITIES)}, got '{v}'"
            )
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        if not v:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

class BatchAnalysisApiRequest(GreenLangBase):
    """Request to submit a batch satellite analysis job."""

    operator_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Operator identifier",
    )
    plots: List[BatchPlotEntry] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of plots to analyze (max 10,000 per batch)",
    )
    analysis_level: str = Field(
        default="standard",
        description="Analysis depth: quick, standard, deep",
    )
    include_baseline: bool = Field(
        default=True,
        description="Establish baseline if not already present",
    )
    include_fusion: bool = Field(
        default=True,
        description="Run multi-source fusion analysis",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "operator_id": "OP-GH-001",
                    "plots": [
                        {
                            "plot_id": "plot-gh-001",
                            "polygon_vertices": [
                                [6.6885, -1.6244],
                                [6.6895, -1.6234],
                                [6.6875, -1.6224],
                                [6.6885, -1.6244],
                            ],
                            "commodity": "cocoa",
                            "country_code": "GH",
                        },
                    ],
                    "analysis_level": "standard",
                }
            ]
        },
    )

    @field_validator("analysis_level")
    @classmethod
    def validate_analysis_level(cls, v: str) -> str:
        """Validate analysis level."""
        v = v.lower().strip()
        if v not in _VALID_ANALYSIS_LEVELS:
            raise ValueError(
                f"analysis_level must be one of {sorted(_VALID_ANALYSIS_LEVELS)}, got '{v}'"
            )
        return v

class BatchAnalysisApiResponse(GreenLangBase):
    """Response after submitting a batch analysis job."""

    batch_id: str = Field(..., description="Unique batch job identifier")
    status: str = Field(
        default="accepted",
        description="Job status: accepted, processing, completed, failed, cancelled",
    )
    total_plots: int = Field(..., ge=0, description="Total plots submitted")
    operator_id: str = Field(..., description="Operator identifier")
    analysis_level: str = Field(default="standard", description="Analysis depth")
    submitted_at: datetime = Field(
        default_factory=utcnow, description="Submission timestamp"
    )
    estimated_completion_seconds: Optional[float] = Field(
        None, description="Estimated time to complete in seconds"
    )

    model_config = ConfigDict(from_attributes=True)

class BatchResultsResponse(GreenLangBase):
    """Response with batch job status and results."""

    batch_id: str = Field(..., description="Batch job identifier")
    status: str = Field(
        ..., description="Job status: accepted, processing, completed, failed, cancelled"
    )
    total_plots: int = Field(..., ge=0, description="Total plots in batch")
    completed_plots: int = Field(default=0, ge=0, description="Plots completed")
    failed_plots: int = Field(default=0, ge=0, description="Plots that failed analysis")
    deforestation_detected_count: int = Field(
        default=0, ge=0, description="Plots with deforestation detected"
    )
    no_change_count: int = Field(
        default=0, ge=0, description="Plots with no change detected"
    )
    average_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Average detection confidence"
    )
    total_forest_loss_ha: float = Field(
        default=0.0, ge=0.0, description="Total forest loss across all plots"
    )
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-plot results (populated when completed)",
    )
    started_at: Optional[datetime] = Field(
        None, description="Processing start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Processing completion timestamp"
    )
    processing_time_ms: Optional[float] = Field(
        None, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class BatchProgressResponse(GreenLangBase):
    """Real-time progress of a batch analysis job."""

    batch_id: str = Field(..., description="Batch job identifier")
    status: str = Field(..., description="Current job status")
    total_plots: int = Field(..., ge=0, description="Total plots in batch")
    completed_plots: int = Field(default=0, ge=0, description="Plots completed so far")
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Completion percentage"
    )
    current_plot_id: Optional[str] = Field(
        None, description="Plot currently being processed"
    )
    elapsed_seconds: float = Field(
        default=0.0, ge=0.0, description="Elapsed time in seconds"
    )
    estimated_remaining_seconds: Optional[float] = Field(
        None, description="Estimated time remaining in seconds"
    )

    model_config = ConfigDict(from_attributes=True)

class BatchCancelResponse(GreenLangBase):
    """Response after cancelling a batch analysis job."""

    batch_id: str = Field(..., description="Cancelled batch job identifier")
    status: str = Field(default="cancelled", description="Job status after cancellation")
    completed_plots: int = Field(
        default=0, ge=0, description="Plots completed before cancellation"
    )
    total_plots: int = Field(
        default=0, ge=0, description="Total plots that were in the batch"
    )
    cancelled_at: datetime = Field(
        default_factory=utcnow, description="Cancellation timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Health Check
# =============================================================================

class HealthResponse(GreenLangBase):
    """Health check response."""

    status: str = Field(default="healthy")
    agent_id: str = Field(default="GL-EUDR-SAT-003")
    agent_name: str = Field(default="EUDR Satellite Monitoring Agent")
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=utcnow)

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
    # Imagery
    "SearchScenesApiRequest",
    "SearchScenesApiResponse",
    "SceneMetadataResponse",
    "DownloadBandsApiRequest",
    "DownloadBandsApiResponse",
    "AvailabilityResponse",
    # Analysis - Spectral Index
    "CalculateIndexApiRequest",
    "SpectralIndexApiResponse",
    # Analysis - Baseline
    "EstablishBaselineApiRequest",
    "BaselineApiResponse",
    # Analysis - Change Detection
    "DetectChangeApiRequest",
    "ChangeDetectionApiResponse",
    # Analysis - Fusion
    "FusionSourceResult",
    "FusionApiRequest",
    "FusionApiResponse",
    # Monitoring
    "CreateMonitoringApiRequest",
    "UpdateMonitoringApiRequest",
    "MonitoringScheduleResponse",
    "MonitoringResultResponse",
    "MonitoringExecuteRequest",
    # Alerts
    "AlertDetailResponse",
    "AlertListResponse",
    "AcknowledgeAlertRequest",
    "AlertSummaryResponse",
    # Evidence
    "GenerateEvidenceApiRequest",
    "EvidencePackageResponse",
    # Batch
    "BatchPlotEntry",
    "BatchAnalysisApiRequest",
    "BatchAnalysisApiResponse",
    "BatchResultsResponse",
    "BatchProgressResponse",
    "BatchCancelResponse",
    # Health
    "HealthResponse",
]
