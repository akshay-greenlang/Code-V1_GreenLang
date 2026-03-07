# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-005 Land Use Change Detector

Pydantic v2 request/response models for the Land Use Change Detector REST API
covering multi-class land use classification, temporal transition detection,
trajectory analysis, EUDR cutoff date compliance verification, conversion risk
assessment, urban encroachment analysis, compliance report generation, and
batch processing.

Core domain models are referenced via the analysis engines; this file
defines API-level request wrappers, response envelopes, pagination, and
batch schemas with full Pydantic v2 validation.

Schema Groups:
    - Enumerations: LandUseCategory, ClassificationMethod, TransitionType,
      TrajectoryType, ComplianceVerdict, EUDRCommodity, RiskTier, ReportFormat,
      ReportType, BatchJobStatus, AnalysisStatus
    - Classification: ClassifyRequest, ClassifyBatchRequest,
      ClassifyCompareRequest, ClassificationResult, ClassificationBatchResponse,
      ClassificationCompareResponse, ClassificationHistoryEntry,
      ClassificationHistoryResponse
    - Transition: TransitionDetectRequest, TransitionBatchRequest,
      TransitionMatrixRequest, TransitionResult, TransitionBatchResponse,
      TransitionMatrixResponse, TransitionTypeInfo, TransitionTypesListResponse
    - Trajectory: TrajectoryAnalyzeRequest, TrajectoryBatchRequest,
      TrajectoryResult, TrajectoryBatchResponse, NDVIDataPoint
    - Verification: VerifyCutoffRequest, VerifyBatchRequest,
      VerifyCompleteRequest, CutoffVerificationResult,
      VerificationBatchResponse, EvidenceItem, EvidencePackage
    - Risk: RiskAssessRequest, RiskBatchRequest, RiskResult, RiskBatchResponse,
      RiskFactor, UrbanAnalyzeRequest, UrbanBatchRequest, UrbanResult,
      UrbanBatchResponse, InfrastructureFeature, PressureCorridor
    - Reports: ReportGenerateRequest, ReportBatchRequest, ReportConfigItem,
      ReportResult, ReportBatchResponse
    - Batch Jobs: BatchJobSubmitRequest, BatchJobResponse
    - System: HealthResponse, VersionResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

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


class LandUseCategory(str, Enum):
    """IPCC-aligned land use classification categories.

    Ten primary land use categories following IPCC Guidelines for
    National Greenhouse Gas Inventories (Volume 4: AFOLU), extended
    with sub-categories relevant to EUDR compliance monitoring.
    """

    FOREST = "forest"
    SHRUBLAND = "shrubland"
    GRASSLAND = "grassland"
    CROPLAND = "cropland"
    WETLAND = "wetland"
    WATER = "water"
    URBAN = "urban"
    BARE_SOIL = "bare_soil"
    SNOW_ICE = "snow_ice"
    OTHER = "other"


class LandUseSubCategory(str, Enum):
    """Detailed sub-categories for land use classification."""

    # Forest sub-categories
    TROPICAL_MOIST_FOREST = "tropical_moist_forest"
    TROPICAL_DRY_FOREST = "tropical_dry_forest"
    TEMPERATE_FOREST = "temperate_forest"
    BOREAL_FOREST = "boreal_forest"
    MANGROVE = "mangrove"
    PLANTATION = "plantation"
    AGROFORESTRY = "agroforestry"
    SECONDARY_GROWTH = "secondary_growth"
    # Cropland sub-categories
    ANNUAL_CROPLAND = "annual_cropland"
    PERENNIAL_CROPLAND = "perennial_cropland"
    PALM_OIL_PLANTATION = "palm_oil_plantation"
    COCOA_PLANTATION = "cocoa_plantation"
    COFFEE_PLANTATION = "coffee_plantation"
    SOY_FIELD = "soy_field"
    RUBBER_PLANTATION = "rubber_plantation"
    # Grassland sub-categories
    NATURAL_GRASSLAND = "natural_grassland"
    PASTURE = "pasture"
    DEGRADED_GRASSLAND = "degraded_grassland"
    # Urban sub-categories
    BUILT_UP = "built_up"
    INFRASTRUCTURE = "infrastructure"
    MINING = "mining"
    # Other
    UNKNOWN = "unknown"
    MIXED = "mixed"


class ClassificationMethod(str, Enum):
    """Land use classification method."""

    SPECTRAL = "spectral"
    VEGETATION_INDEX = "vegetation_index"
    PHENOLOGY = "phenology"
    TEXTURE = "texture"
    ENSEMBLE = "ensemble"


class TransitionType(str, Enum):
    """Land use transition type classification.

    Categories follow IPCC land use change matrix conventions with
    EUDR-specific deforestation and degradation sub-types.
    """

    DEFORESTATION = "deforestation"
    DEGRADATION = "degradation"
    AFFORESTATION = "afforestation"
    REFORESTATION = "reforestation"
    AGRICULTURAL_EXPANSION = "agricultural_expansion"
    URBANIZATION = "urbanization"
    WETLAND_CONVERSION = "wetland_conversion"
    CROPLAND_ABANDONMENT = "cropland_abandonment"
    INTENSIFICATION = "intensification"
    STABLE = "stable"
    UNKNOWN = "unknown"


class TrajectoryType(str, Enum):
    """Temporal change trajectory classification."""

    STABLE = "stable"
    ABRUPT_CHANGE = "abrupt_change"
    GRADUAL_CHANGE = "gradual_change"
    OSCILLATING = "oscillating"
    RECOVERY = "recovery"


class ComplianceVerdict(str, Enum):
    """EUDR cutoff date compliance verdict."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    DEGRADED = "degraded"
    INCONCLUSIVE = "inconclusive"
    INSUFFICIENT_DATA = "insufficient_data"


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodity types per Article 1(1) of EU 2023/1115."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class RiskTier(str, Enum):
    """Conversion risk tier classification."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ReportFormat(str, Enum):
    """Compliance report output format."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XLSX = "xlsx"


class ReportType(str, Enum):
    """Land use change report type."""

    CLASSIFICATION_SUMMARY = "classification_summary"
    TRANSITION_ANALYSIS = "transition_analysis"
    TRAJECTORY_REPORT = "trajectory_report"
    CUTOFF_VERIFICATION = "cutoff_verification"
    RISK_ASSESSMENT = "risk_assessment"
    URBAN_ENCROACHMENT = "urban_encroachment"
    COMPREHENSIVE = "comprehensive"


class BatchJobStatus(str, Enum):
    """Batch job processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class AnalysisStatus(str, Enum):
    """Individual analysis status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobType(str, Enum):
    """Type of batch job to submit."""

    CLASSIFICATION = "classification"
    TRANSITION = "transition"
    TRAJECTORY = "trajectory"
    VERIFICATION = "verification"
    RISK_ASSESSMENT = "risk_assessment"
    URBAN_ANALYSIS = "urban_analysis"
    REPORT_GENERATION = "report_generation"


# =============================================================================
# WKT and validation helpers
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
            "e.g. 'POLYGON((-1.624 6.688, -1.623 6.689, "
            "-1.622 6.687, -1.624 6.688))'"
        )
    return v


def _validate_plot_id(v: str) -> str:
    """Validate plot ID format.

    Args:
        v: Plot ID string to validate.

    Returns:
        Cleaned plot ID string.

    Raises:
        ValueError: If the plot ID is empty or exceeds 200 characters.
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
    request_id: Optional[str] = Field(
        None, description="Request correlation ID"
    )


# =============================================================================
# Spectral Index Data Model
# =============================================================================


class SpectralIndicesData(BaseModel):
    """Spectral vegetation indices computed for classification."""

    ndvi: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Normalized Difference Vegetation Index (-1 to 1)",
    )
    evi: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Enhanced Vegetation Index (-1 to 1)",
    )
    savi: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Soil-Adjusted Vegetation Index (-1 to 1)",
    )
    ndwi: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Normalized Difference Water Index (-1 to 1)",
    )
    nbr: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Normalized Burn Ratio (-1 to 1)",
    )
    bsi: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Bare Soil Index (-1 to 1)",
    )
    ndbi: Optional[float] = Field(
        None, ge=-1.0, le=1.0,
        description="Normalized Difference Built-up Index (-1 to 1)",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Classification Schemas - Request
# =============================================================================


class PlotCoordinate(BaseModel):
    """Single plot coordinate for point-based analysis."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )

    model_config = ConfigDict(extra="forbid")


class ClassifyRequest(BaseModel):
    """Request to classify land use for a single plot.

    Supports both coordinate-based (latitude/longitude) and polygon-based
    (polygon_wkt) spatial references. At least one must be provided.
    """

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    date: date = Field(
        ...,
        description="Target classification date (YYYY-MM-DD)",
    )
    method: ClassificationMethod = Field(
        default=ClassificationMethod.ENSEMBLE,
        description=(
            "Classification method: spectral, vegetation_index, "
            "phenology, texture, ensemble"
        ),
    )
    commodity_context: Optional[EUDRCommodity] = Field(
        None,
        description=(
            "EUDR commodity context for targeted classification. "
            "When provided, the classifier applies commodity-specific "
            "spectral signatures and thresholds."
        ),
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "date": "2024-06-15",
                    "method": "ensemble",
                    "commodity_context": "cocoa",
                }
            ]
        },
    )

    @field_validator("plot_id")
    @classmethod
    def validate_plot_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional plot ID format."""
        if v is None:
            return v
        return _validate_plot_id(v)

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


class ClassifyBatchRequest(BaseModel):
    """Request for batch land use classification."""

    plots: List[ClassifyRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots to classify (max 5000 per batch)",
    )
    date: Optional[date] = Field(
        None,
        description=(
            "Global classification date. Overrides individual plot dates "
            "if provided."
        ),
    )
    method: ClassificationMethod = Field(
        default=ClassificationMethod.ENSEMBLE,
        description="Classification method applied to all plots",
    )

    model_config = ConfigDict(extra="forbid")


class ClassifyCompareRequest(BaseModel):
    """Request to compare land use classification between two dates."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    date1: date = Field(
        ...,
        description="First classification date (earlier, YYYY-MM-DD)",
    )
    date2: date = Field(
        ...,
        description="Second classification date (later, YYYY-MM-DD)",
    )
    method: ClassificationMethod = Field(
        default=ClassificationMethod.ENSEMBLE,
        description="Classification method for both dates",
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "date1": "2020-06-15",
                    "date2": "2024-06-15",
                    "method": "ensemble",
                }
            ]
        },
    )

    @field_validator("date2")
    @classmethod
    def validate_date_order(cls, v: date, info) -> date:
        """Validate date2 is after date1."""
        date1 = info.data.get("date1")
        if date1 and v <= date1:
            raise ValueError(
                f"date2 ({v}) must be after date1 ({date1})"
            )
        return v

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


# =============================================================================
# Classification Schemas - Response
# =============================================================================


class ClassificationResult(BaseModel):
    """Result from a single land use classification analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    land_use_category: LandUseCategory = Field(
        ..., description="Primary IPCC land use category"
    )
    sub_category: Optional[LandUseSubCategory] = Field(
        None, description="Detailed sub-category within the primary class"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence (0-1)"
    )
    method: ClassificationMethod = Field(
        ..., description="Classification method used"
    )
    class_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across all land use categories",
    )
    spectral_indices: Optional[SpectralIndicesData] = Field(
        None, description="Computed spectral vegetation indices"
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Plot latitude (WGS84)"
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Plot longitude (WGS84)"
    )
    area_ha: Optional[float] = Field(
        None, ge=0.0, description="Plot area in hectares (if polygon provided)"
    )
    imagery_date: Optional[date] = Field(
        None, description="Date of satellite imagery used"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used for classification"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Classification timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class ClassificationBatchResponse(BaseModel):
    """Response from batch land use classification."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    results: List[ClassificationResult] = Field(
        default_factory=list, description="Individual classification results"
    )
    total: int = Field(..., ge=0, description="Total plots submitted")
    successful: int = Field(
        default=0, ge=0, description="Successfully classified plots"
    )
    failed: int = Field(default=0, ge=0, description="Failed classifications")
    category_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of plots across land use categories",
    )
    mean_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean classification confidence across all results",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Batch completion timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class ClassificationCompareResponse(BaseModel):
    """Response from comparing classification between two dates."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    date1_result: ClassificationResult = Field(
        ..., description="Classification result at date1"
    )
    date2_result: ClassificationResult = Field(
        ..., description="Classification result at date2"
    )
    transition_detected: bool = Field(
        ...,
        description="Whether a land use transition was detected between dates",
    )
    transition_type: Optional[TransitionType] = Field(
        None,
        description="Type of transition detected (if any)",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the comparison result",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Comparison timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class ClassificationHistoryEntry(BaseModel):
    """Single entry in classification history time series."""

    date: date = Field(..., description="Classification observation date")
    land_use_category: LandUseCategory = Field(
        ..., description="Classified land use category"
    )
    sub_category: Optional[LandUseSubCategory] = Field(
        None, description="Detailed sub-category"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    method: str = Field(default="ensemble", description="Method used")
    source: str = Field(default="", description="Primary data source")

    model_config = ConfigDict(from_attributes=True)


class ClassificationHistoryResponse(BaseModel):
    """Response with classification history over time for a plot."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    entries: List[ClassificationHistoryEntry] = Field(
        default_factory=list, description="Classification time series entries"
    )
    dominant_category: Optional[LandUseCategory] = Field(
        None, description="Most frequent land use category in history"
    )
    transitions_count: int = Field(
        default=0, ge=0, description="Number of transitions detected in history"
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
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Transition Schemas - Request
# =============================================================================


class TransitionDetectRequest(BaseModel):
    """Request to detect land use transition for a single plot."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    date_from: date = Field(
        ...,
        description="Transition detection start date (YYYY-MM-DD)",
    )
    date_to: date = Field(
        ...,
        description="Transition detection end date (YYYY-MM-DD)",
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )
    min_area_ha: Optional[float] = Field(
        None, ge=0.0, le=10000.0,
        description=(
            "Minimum area threshold in hectares for transition detection. "
            "Overrides the global configuration default."
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "date_from": "2020-01-01",
                    "date_to": "2024-12-31",
                }
            ]
        },
    )

    @field_validator("date_to")
    @classmethod
    def validate_date_order(cls, v: date, info) -> date:
        """Validate date_to is after date_from."""
        date_from = info.data.get("date_from")
        if date_from and v <= date_from:
            raise ValueError(
                f"date_to ({v}) must be after date_from ({date_from})"
            )
        return v

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


class TransitionBatchRequest(BaseModel):
    """Request for batch transition detection."""

    plots: List[TransitionDetectRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots for transition detection (max 5000)",
    )
    date_from: Optional[date] = Field(
        None,
        description=(
            "Global start date. Overrides individual plot dates if provided."
        ),
    )
    date_to: Optional[date] = Field(
        None,
        description=(
            "Global end date. Overrides individual plot dates if provided."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("date_to")
    @classmethod
    def validate_date_order(cls, v: Optional[date], info) -> Optional[date]:
        """Validate global date_to is after date_from if both provided."""
        if v is None:
            return v
        date_from = info.data.get("date_from")
        if date_from and v <= date_from:
            raise ValueError(
                f"date_to ({v}) must be after date_from ({date_from})"
            )
        return v


class TransitionMatrixRequest(BaseModel):
    """Request to generate a land use transition matrix for a region."""

    region_bounds: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "Bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84 "
            "decimal degrees"
        ),
    )
    date_from: date = Field(
        ...,
        description="Transition matrix start date (YYYY-MM-DD)",
    )
    date_to: date = Field(
        ...,
        description="Transition matrix end date (YYYY-MM-DD)",
    )
    resolution_m: int = Field(
        default=30, ge=10, le=1000,
        description="Spatial resolution in meters for matrix computation",
    )
    categories: Optional[List[LandUseCategory]] = Field(
        None,
        description="Subset of categories to include. Defaults to all.",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "region_bounds": [-1.700, 6.600, -1.500, 6.800],
                    "date_from": "2020-01-01",
                    "date_to": "2024-12-31",
                    "resolution_m": 30,
                }
            ]
        },
    )

    @field_validator("region_bounds")
    @classmethod
    def validate_bounds(cls, v: List[float]) -> List[float]:
        """Validate bounding box coordinates."""
        if len(v) != 4:
            raise ValueError("region_bounds must have exactly 4 values")
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180.0 <= min_lon <= 180.0):
            raise ValueError(f"min_lon must be in [-180, 180], got {min_lon}")
        if not (-90.0 <= min_lat <= 90.0):
            raise ValueError(f"min_lat must be in [-90, 90], got {min_lat}")
        if not (-180.0 <= max_lon <= 180.0):
            raise ValueError(f"max_lon must be in [-180, 180], got {max_lon}")
        if not (-90.0 <= max_lat <= 90.0):
            raise ValueError(f"max_lat must be in [-90, 90], got {max_lat}")
        if max_lon <= min_lon:
            raise ValueError(
                f"max_lon ({max_lon}) must be greater than min_lon ({min_lon})"
            )
        if max_lat <= min_lat:
            raise ValueError(
                f"max_lat ({max_lat}) must be greater than min_lat ({min_lat})"
            )
        return v

    @field_validator("date_to")
    @classmethod
    def validate_date_order(cls, v: date, info) -> date:
        """Validate date_to is after date_from."""
        date_from = info.data.get("date_from")
        if date_from and v <= date_from:
            raise ValueError(
                f"date_to ({v}) must be after date_from ({date_from})"
            )
        return v


# =============================================================================
# Transition Schemas - Response
# =============================================================================


class TransitionEvidence(BaseModel):
    """Evidence supporting a detected land use transition."""

    evidence_type: str = Field(
        ..., description="Type of evidence (spectral, temporal, spatial)"
    )
    description: str = Field(
        ..., description="Human-readable description of the evidence"
    )
    date: Optional[date] = Field(None, description="Date of the evidence")
    source: str = Field(default="", description="Data source")
    value: Optional[float] = Field(None, description="Numeric evidence value")
    unit: Optional[str] = Field(None, description="Unit of the value")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Evidence confidence"
    )

    model_config = ConfigDict(from_attributes=True)


class TransitionResult(BaseModel):
    """Result from a single transition detection analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    from_class: LandUseCategory = Field(
        ..., description="Land use category at start date"
    )
    to_class: LandUseCategory = Field(
        ..., description="Land use category at end date"
    )
    transition_type: TransitionType = Field(
        ..., description="Classified transition type"
    )
    date_range: Dict[str, str] = Field(
        ...,
        description="Date range as {'from': 'YYYY-MM-DD', 'to': 'YYYY-MM-DD'}",
    )
    estimated_transition_date: Optional[date] = Field(
        None, description="Estimated date when transition occurred"
    )
    transition_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Estimated area of transition in hectares",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence (0-1)"
    )
    evidence: List[TransitionEvidence] = Field(
        default_factory=list,
        description="Evidence items supporting the transition detection",
    )
    is_eudr_relevant: bool = Field(
        default=False,
        description=(
            "Whether this transition is relevant to EUDR compliance "
            "(deforestation or degradation post-cutoff)"
        ),
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Plot latitude (WGS84)"
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Plot longitude (WGS84)"
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
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class TransitionBatchResponse(BaseModel):
    """Response from batch transition detection."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    results: List[TransitionResult] = Field(
        default_factory=list, description="Individual transition results"
    )
    total: int = Field(..., ge=0, description="Total plots submitted")
    successful: int = Field(
        default=0, ge=0, description="Successfully processed plots"
    )
    failed: int = Field(default=0, ge=0, description="Failed detections")
    deforestation_count: int = Field(
        default=0, ge=0,
        description="Plots where deforestation was detected",
    )
    degradation_count: int = Field(
        default=0, ge=0,
        description="Plots where forest degradation was detected",
    )
    stable_count: int = Field(
        default=0, ge=0,
        description="Plots with no transition detected",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Batch completion timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class TransitionMatrixCell(BaseModel):
    """Single cell in a transition matrix."""

    from_class: str = Field(..., description="Source land use category")
    to_class: str = Field(..., description="Destination land use category")
    area_ha: float = Field(
        ..., ge=0.0, description="Transition area in hectares"
    )
    pixel_count: int = Field(
        default=0, ge=0, description="Number of pixels in transition"
    )
    percentage: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of total study area",
    )

    model_config = ConfigDict(from_attributes=True)


class TopTransition(BaseModel):
    """Top transition by area in a transition matrix."""

    from_class: str = Field(..., description="Source land use category")
    to_class: str = Field(..., description="Destination land use category")
    area_ha: float = Field(
        ..., ge=0.0, description="Transition area in hectares"
    )
    percentage: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of total transitions",
    )

    model_config = ConfigDict(from_attributes=True)


class TransitionMatrixResponse(BaseModel):
    """Response from transition matrix generation."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    matrix: List[TransitionMatrixCell] = Field(
        default_factory=list,
        description="Transition matrix cells (from-to pairs with areas)",
    )
    total_area_ha: float = Field(
        ..., ge=0.0, description="Total study area in hectares"
    )
    transitions_detected: int = Field(
        ..., ge=0, description="Number of distinct transitions detected"
    )
    top_transitions: List[TopTransition] = Field(
        default_factory=list,
        description="Top transitions ranked by area",
    )
    date_from: date = Field(..., description="Analysis start date")
    date_to: date = Field(..., description="Analysis end date")
    resolution_m: int = Field(
        default=30, description="Spatial resolution used"
    )
    region_bounds: List[float] = Field(
        ..., description="Bounding box used for analysis"
    )
    deforestation_area_ha: float = Field(
        default=0.0, ge=0.0,
        description="Total area of forest-to-non-forest transitions",
    )
    net_forest_change_ha: float = Field(
        default=0.0,
        description="Net forest area change (positive = gain, negative = loss)",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Analysis timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class TransitionTypeInfo(BaseModel):
    """Information about a supported transition type."""

    type_id: str = Field(..., description="Transition type identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Detailed description")
    from_categories: List[str] = Field(
        default_factory=list,
        description="Source land use categories for this transition",
    )
    to_categories: List[str] = Field(
        default_factory=list,
        description="Destination land use categories for this transition",
    )
    eudr_relevance: str = Field(
        default="low",
        description="EUDR relevance level: high, medium, low",
    )
    severity: str = Field(
        default="low",
        description="Environmental severity: critical, high, medium, low",
    )

    model_config = ConfigDict(from_attributes=True)


class TransitionTypesListResponse(BaseModel):
    """Response listing all supported transition types."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    types: List[TransitionTypeInfo] = Field(
        ..., description="List of supported transition types"
    )
    total: int = Field(..., ge=0, description="Total number of types")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Trajectory Schemas - Request
# =============================================================================


class TrajectoryAnalyzeRequest(BaseModel):
    """Request to analyze temporal land use change trajectory."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    date_from: date = Field(
        ...,
        description="Trajectory analysis start date (YYYY-MM-DD)",
    )
    date_to: date = Field(
        ...,
        description="Trajectory analysis end date (YYYY-MM-DD)",
    )
    time_step_months: int = Field(
        default=3, ge=1, le=12,
        description=(
            "Time step interval in months for trajectory sampling. "
            "Smaller values provide finer temporal resolution."
        ),
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "date_from": "2018-01-01",
                    "date_to": "2024-12-31",
                    "time_step_months": 3,
                }
            ]
        },
    )

    @field_validator("date_to")
    @classmethod
    def validate_date_order(cls, v: date, info) -> date:
        """Validate date_to is after date_from."""
        date_from = info.data.get("date_from")
        if date_from and v <= date_from:
            raise ValueError(
                f"date_to ({v}) must be after date_from ({date_from})"
            )
        return v

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


class TrajectoryBatchRequest(BaseModel):
    """Request for batch trajectory analysis."""

    plots: List[TrajectoryAnalyzeRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots for trajectory analysis (max 5000)",
    )
    date_from: Optional[date] = Field(
        None,
        description="Global start date. Overrides individual plot dates.",
    )
    date_to: Optional[date] = Field(
        None,
        description="Global end date. Overrides individual plot dates.",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("date_to")
    @classmethod
    def validate_date_order(cls, v: Optional[date], info) -> Optional[date]:
        """Validate global date_to is after date_from if both provided."""
        if v is None:
            return v
        date_from = info.data.get("date_from")
        if date_from and v <= date_from:
            raise ValueError(
                f"date_to ({v}) must be after date_from ({date_from})"
            )
        return v


# =============================================================================
# Trajectory Schemas - Response
# =============================================================================


class NDVIDataPoint(BaseModel):
    """Single NDVI observation in a time series."""

    date: date = Field(..., description="Observation date")
    ndvi: float = Field(
        ..., ge=-1.0, le=1.0, description="NDVI value (-1 to 1)"
    )
    quality_flag: str = Field(
        default="good",
        description="Quality flag: good, cloudy, shadowed, interpolated",
    )
    source: str = Field(default="", description="Data source identifier")

    model_config = ConfigDict(from_attributes=True)


class ChangeDate(BaseModel):
    """Detected change point in a trajectory."""

    date: date = Field(..., description="Estimated date of change")
    from_category: LandUseCategory = Field(
        ..., description="Land use before change"
    )
    to_category: LandUseCategory = Field(
        ..., description="Land use after change"
    )
    magnitude: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Change magnitude (0 = minimal, 1 = complete conversion)",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Change detection confidence"
    )

    model_config = ConfigDict(from_attributes=True)


class VisualizationData(BaseModel):
    """Data formatted for trajectory visualization."""

    time_labels: List[str] = Field(
        default_factory=list, description="Time axis labels (ISO dates)"
    )
    ndvi_values: List[float] = Field(
        default_factory=list, description="NDVI values for time axis"
    )
    category_labels: List[str] = Field(
        default_factory=list,
        description="Land use category at each time step",
    )
    change_markers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Change point markers for visualization",
    )

    model_config = ConfigDict(from_attributes=True)


class TrajectoryResult(BaseModel):
    """Result from a temporal trajectory analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    trajectory_type: TrajectoryType = Field(
        ..., description="Classified trajectory type"
    )
    change_dates: List[ChangeDate] = Field(
        default_factory=list,
        description="Detected change points in the trajectory",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Trajectory classification confidence"
    )
    ndvi_series: List[NDVIDataPoint] = Field(
        default_factory=list, description="NDVI time series data"
    )
    mean_ndvi: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Mean NDVI across the analysis period",
    )
    ndvi_trend: float = Field(
        default=0.0,
        description="Linear NDVI trend (slope per year, positive = greening)",
    )
    ndvi_variance: float = Field(
        default=0.0, ge=0.0,
        description="NDVI temporal variance (indicator of instability)",
    )
    visualization_data: Optional[VisualizationData] = Field(
        None, description="Formatted data for trajectory visualization"
    )
    date_from: date = Field(..., description="Analysis start date")
    date_to: date = Field(..., description="Analysis end date")
    time_steps: int = Field(
        default=0, ge=0, description="Number of time steps analyzed"
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Plot latitude (WGS84)"
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Plot longitude (WGS84)"
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
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class TrajectoryBatchResponse(BaseModel):
    """Response from batch trajectory analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    results: List[TrajectoryResult] = Field(
        default_factory=list, description="Individual trajectory results"
    )
    total: int = Field(..., ge=0, description="Total plots submitted")
    successful: int = Field(
        default=0, ge=0, description="Successfully analyzed plots"
    )
    failed: int = Field(default=0, ge=0, description="Failed analyses")
    stable_count: int = Field(
        default=0, ge=0, description="Plots with stable trajectories"
    )
    changed_count: int = Field(
        default=0, ge=0,
        description="Plots with detected changes (abrupt or gradual)",
    )
    trajectory_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of plots across trajectory types",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Batch completion timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Verification Schemas - Request
# =============================================================================


class VerifyCutoffRequest(BaseModel):
    """Request to verify EUDR cutoff date compliance for a single plot."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR-regulated commodity being sourced from this plot",
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to include supporting evidence in the response",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "commodity": "cocoa",
                    "include_evidence": True,
                }
            ]
        },
    )

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


class VerifyBatchPlot(BaseModel):
    """Single plot entry in a batch verification request."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    commodity: EUDRCommodity = Field(
        ..., description="EUDR-regulated commodity"
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


class VerifyBatchRequest(BaseModel):
    """Request for batch cutoff date verification."""

    plots: List[VerifyBatchPlot] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots for verification (max 5000)",
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to include evidence for each plot",
    )

    model_config = ConfigDict(extra="forbid")


class VerifyCompleteRequest(BaseModel):
    """Request for complete verification pipeline (classification + transition
    + trajectory + cutoff verification combined)."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR-regulated commodity being sourced from this plot",
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to include full evidence chain",
    )
    include_trajectory: bool = Field(
        default=True,
        description="Whether to include temporal trajectory analysis",
    )
    include_risk: bool = Field(
        default=True,
        description="Whether to include risk assessment",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "commodity": "cocoa",
                    "include_evidence": True,
                    "include_trajectory": True,
                    "include_risk": True,
                }
            ]
        },
    )

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


# =============================================================================
# Verification Schemas - Response
# =============================================================================


class EvidenceItem(BaseModel):
    """Single evidence item supporting a verification verdict."""

    evidence_type: str = Field(
        ...,
        description=(
            "Type of evidence: spectral_analysis, transition_detection, "
            "trajectory_analysis, satellite_imagery, historical_comparison"
        ),
    )
    description: str = Field(
        ..., description="Human-readable evidence description"
    )
    date: Optional[date] = Field(None, description="Evidence observation date")
    source: str = Field(default="", description="Data source identifier")
    value: Optional[float] = Field(
        None, description="Numeric evidence value"
    )
    unit: Optional[str] = Field(None, description="Unit of the value")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Evidence confidence"
    )

    model_config = ConfigDict(from_attributes=True)


class EvidencePackage(BaseModel):
    """Complete evidence package for a verification verdict."""

    plot_id: str = Field(..., description="Plot identifier")
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list, description="All evidence items"
    )
    classification_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence from land use classification",
    )
    transition_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence from transition detection",
    )
    trajectory_evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence from trajectory analysis",
    )
    satellite_imagery_refs: List[str] = Field(
        default_factory=list,
        description="References to satellite imagery used",
    )
    total_evidence_items: int = Field(
        default=0, ge=0, description="Total number of evidence items"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash of evidence package"
    )

    model_config = ConfigDict(from_attributes=True)


class CutoffVerificationResult(BaseModel):
    """Result from EUDR cutoff date compliance verification."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    verification_id: str = Field(
        ..., description="Unique verification identifier"
    )
    plot_id: str = Field(..., description="Plot identifier")
    verdict: ComplianceVerdict = Field(
        ..., description="EUDR compliance verdict"
    )
    commodity: str = Field(
        ..., description="EUDR commodity under verification"
    )
    cutoff_date: str = Field(
        default="2020-12-31",
        description="EUDR cutoff date used for verification",
    )
    cutoff_classification: LandUseCategory = Field(
        ..., description="Land use classification at the cutoff date"
    )
    current_classification: LandUseCategory = Field(
        ..., description="Current land use classification"
    )
    transition_detected: bool = Field(
        ...,
        description="Whether a land use transition was detected since cutoff",
    )
    transition_type: Optional[TransitionType] = Field(
        None, description="Type of transition detected (if any)"
    )
    trajectory: Optional[TrajectoryResult] = Field(
        None, description="Full trajectory analysis (if requested)"
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items supporting the verdict",
    )
    risk_assessment: Optional[Dict[str, Any]] = Field(
        None, description="Risk assessment summary (if requested)"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Overall verification confidence",
    )
    engines_used: List[str] = Field(
        default_factory=list,
        description="Analysis engines used for verification",
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources consulted"
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Plot latitude (WGS84)"
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Plot longitude (WGS84)"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Verification timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class VerificationBatchResponse(BaseModel):
    """Response from batch cutoff date verification."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    results: List[CutoffVerificationResult] = Field(
        default_factory=list, description="Individual verification results"
    )
    total: int = Field(..., ge=0, description="Total plots submitted")
    successful: int = Field(
        default=0, ge=0, description="Successfully verified plots"
    )
    failed: int = Field(default=0, ge=0, description="Failed verifications")
    compliant_count: int = Field(
        default=0, ge=0, description="Plots verified as compliant"
    )
    non_compliant_count: int = Field(
        default=0, ge=0, description="Plots verified as non-compliant"
    )
    degraded_count: int = Field(
        default=0, ge=0, description="Plots with degradation detected"
    )
    inconclusive_count: int = Field(
        default=0, ge=0, description="Plots with inconclusive results"
    )
    verdict_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of plots across verdict categories",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Batch completion timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Risk Schemas - Request
# =============================================================================


class RiskAssessRequest(BaseModel):
    """Request to assess conversion risk for a single plot."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    commodity: EUDRCommodity = Field(
        ..., description="EUDR-regulated commodity context"
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )
    include_factors: bool = Field(
        default=True,
        description="Whether to include individual risk factor scores",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "commodity": "cocoa",
                    "include_factors": True,
                }
            ]
        },
    )

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


class RiskBatchRequest(BaseModel):
    """Request for batch risk assessment."""

    plots: List[RiskAssessRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots for risk assessment (max 5000)",
    )

    model_config = ConfigDict(extra="forbid")


class UrbanAnalyzeRequest(BaseModel):
    """Request to analyze urban encroachment around a plot."""

    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees (WGS84)",
    )
    buffer_km: float = Field(
        default=10.0, ge=0.1, le=50.0,
        description="Analysis buffer radius in kilometres",
    )
    date_from: date = Field(
        ...,
        description="Analysis start date (YYYY-MM-DD)",
    )
    date_to: date = Field(
        ...,
        description="Analysis end date (YYYY-MM-DD)",
    )
    plot_id: Optional[str] = Field(
        None, max_length=200,
        description="Optional plot identifier for result tracking",
    )
    polygon_wkt: Optional[str] = Field(
        None, max_length=100000,
        description="Optional plot boundary as WKT POLYGON geometry",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "buffer_km": 10.0,
                    "date_from": "2020-01-01",
                    "date_to": "2024-12-31",
                }
            ]
        },
    )

    @field_validator("date_to")
    @classmethod
    def validate_date_order(cls, v: date, info) -> date:
        """Validate date_to is after date_from."""
        date_from = info.data.get("date_from")
        if date_from and v <= date_from:
            raise ValueError(
                f"date_to ({v}) must be after date_from ({date_from})"
            )
        return v

    @field_validator("polygon_wkt")
    @classmethod
    def validate_polygon_wkt(cls, v: Optional[str]) -> Optional[str]:
        """Validate optional WKT polygon format."""
        if v is None:
            return v
        return _validate_polygon_wkt(v)


class UrbanBatchRequest(BaseModel):
    """Request for batch urban encroachment analysis."""

    plots: List[UrbanAnalyzeRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of plots for urban analysis (max 5000)",
    )
    buffer_km: Optional[float] = Field(
        None, ge=0.1, le=50.0,
        description=(
            "Global buffer radius in km. Overrides individual plot buffers."
        ),
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Risk Schemas - Response
# =============================================================================


class RiskFactor(BaseModel):
    """Individual risk factor assessment."""

    factor_name: str = Field(
        ..., description="Risk factor identifier"
    )
    display_name: str = Field(
        ..., description="Human-readable factor name"
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Factor score (0 = low, 1 = high)"
    )
    weight: float = Field(
        ..., ge=0.0, le=1.0, description="Factor weight in composite score"
    )
    weighted_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weighted contribution to composite score",
    )
    description: str = Field(
        default="", description="Factor assessment description"
    )
    data_quality: str = Field(
        default="medium",
        description="Data quality for this factor: high, medium, low",
    )

    model_config = ConfigDict(from_attributes=True)


class RiskResult(BaseModel):
    """Result from conversion risk assessment."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    composite_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Composite risk score (0 = lowest, 1 = highest)",
    )
    risk_tier: RiskTier = Field(
        ..., description="Risk tier classification"
    )
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Individual risk factor assessments (8 factors)",
    )
    probability_6m: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of conversion within 6 months",
    )
    probability_12m: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of conversion within 12 months",
    )
    probability_24m: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probability of conversion within 24 months",
    )
    commodity: str = Field(
        ..., description="Commodity context for risk assessment"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Risk mitigation recommendations",
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Plot latitude (WGS84)"
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Plot longitude (WGS84)"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Assessment timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class RiskBatchResponse(BaseModel):
    """Response from batch risk assessment."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    results: List[RiskResult] = Field(
        default_factory=list, description="Individual risk results"
    )
    total: int = Field(..., ge=0, description="Total plots submitted")
    successful: int = Field(
        default=0, ge=0, description="Successfully assessed plots"
    )
    failed: int = Field(default=0, ge=0, description="Failed assessments")
    by_tier: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of plots by risk tier",
    )
    mean_composite_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean composite risk score across all results",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Batch completion timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class InfrastructureFeature(BaseModel):
    """Detected infrastructure feature near a plot."""

    feature_type: str = Field(
        ...,
        description=(
            "Infrastructure type: road, highway, railway, building, "
            "industrial, commercial, residential"
        ),
    )
    distance_km: float = Field(
        ..., ge=0.0, description="Distance from plot in kilometres"
    )
    bearing_degrees: float = Field(
        default=0.0, ge=0.0, le=360.0,
        description="Compass bearing from plot to feature",
    )
    growth_rate_pct_year: Optional[float] = Field(
        None,
        description="Annual growth rate of this feature type (%/year)",
    )

    model_config = ConfigDict(from_attributes=True)


class PressureCorridor(BaseModel):
    """Urban expansion pressure corridor identified near a plot."""

    corridor_id: str = Field(
        ..., description="Corridor identifier"
    )
    direction: str = Field(
        ..., description="Expansion direction (N, NE, E, SE, S, SW, W, NW)"
    )
    width_km: float = Field(
        ..., ge=0.0, description="Corridor width in kilometres"
    )
    expansion_rate_ha_year: float = Field(
        ..., ge=0.0,
        description="Annual expansion rate in hectares per year",
    )
    distance_to_plot_km: float = Field(
        ..., ge=0.0, description="Distance from corridor front to plot in km"
    )
    estimated_arrival_months: Optional[int] = Field(
        None, ge=0,
        description="Estimated months until corridor reaches plot",
    )

    model_config = ConfigDict(from_attributes=True)


class UrbanResult(BaseModel):
    """Result from urban encroachment analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    plot_id: str = Field(..., description="Plot identifier")
    encroachment_detected: bool = Field(
        ...,
        description="Whether urban encroachment is detected in the buffer zone",
    )
    urban_expansion_rate_ha_year: float = Field(
        default=0.0, ge=0.0,
        description="Urban expansion rate in hectares per year within buffer",
    )
    urban_area_start_ha: float = Field(
        default=0.0, ge=0.0,
        description="Urban area within buffer at start date (hectares)",
    )
    urban_area_end_ha: float = Field(
        default=0.0, ge=0.0,
        description="Urban area within buffer at end date (hectares)",
    )
    infrastructure_types: List[InfrastructureFeature] = Field(
        default_factory=list,
        description="Infrastructure features detected near the plot",
    )
    pressure_corridors: List[PressureCorridor] = Field(
        default_factory=list,
        description="Identified urban expansion pressure corridors",
    )
    time_to_conversion_months: Optional[int] = Field(
        None, ge=0,
        description=(
            "Estimated months until urban conversion reaches the plot "
            "(None if no imminent threat)"
        ),
    )
    buffer_km: float = Field(
        ..., ge=0.0, description="Analysis buffer radius used (km)"
    )
    date_from: date = Field(..., description="Analysis start date")
    date_to: date = Field(..., description="Analysis end date")
    latitude: float = Field(
        ..., ge=-90.0, le=90.0, description="Plot latitude (WGS84)"
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Plot longitude (WGS84)"
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
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class UrbanBatchResponse(BaseModel):
    """Response from batch urban encroachment analysis."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    results: List[UrbanResult] = Field(
        default_factory=list, description="Individual urban analysis results"
    )
    total: int = Field(..., ge=0, description="Total plots submitted")
    successful: int = Field(
        default=0, ge=0, description="Successfully analyzed plots"
    )
    failed: int = Field(default=0, ge=0, description="Failed analyses")
    encroachment_count: int = Field(
        default=0, ge=0,
        description="Plots where urban encroachment was detected",
    )
    mean_expansion_rate: float = Field(
        default=0.0, ge=0.0,
        description="Mean urban expansion rate across all results (ha/year)",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Batch completion timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Report Schemas - Request
# =============================================================================


class ReportOptions(BaseModel):
    """Options for report generation."""

    include_maps: bool = Field(
        default=True,
        description="Whether to include map visualizations",
    )
    include_charts: bool = Field(
        default=True,
        description="Whether to include charts and graphs",
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to include evidence packages",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Whether to include actionable recommendations",
    )
    language: str = Field(
        default="en",
        description="Report language (ISO 639-1 code)",
    )
    operator_id: Optional[str] = Field(
        None, description="Operator identifier for branding"
    )

    model_config = ConfigDict(extra="forbid")


class ReportGenerateRequest(BaseModel):
    """Request to generate a land use change report."""

    report_type: ReportType = Field(
        ..., description="Type of report to generate"
    )
    plot_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Plot identifiers to include in the report",
    )
    format: ReportFormat = Field(
        default=ReportFormat.PDF,
        description="Output format for the report",
    )
    options: Optional[ReportOptions] = Field(
        None,
        description="Report generation options",
    )
    title: Optional[str] = Field(
        None, max_length=500,
        description="Custom report title",
    )
    date_range_start: Optional[date] = Field(
        None, description="Report coverage start date"
    )
    date_range_end: Optional[date] = Field(
        None, description="Report coverage end date"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "report_type": "cutoff_verification",
                    "plot_ids": ["plot-gh-001", "plot-gh-002"],
                    "format": "pdf",
                    "options": {
                        "include_maps": True,
                        "include_evidence": True,
                    },
                }
            ]
        },
    )

    @field_validator("date_range_end")
    @classmethod
    def validate_date_order(cls, v: Optional[date], info) -> Optional[date]:
        """Validate end date is after start date."""
        if v is None:
            return v
        start = info.data.get("date_range_start")
        if start and v <= start:
            raise ValueError(
                f"date_range_end ({v}) must be after "
                f"date_range_start ({start})"
            )
        return v


class ReportConfigItem(BaseModel):
    """Single report configuration in a batch request."""

    report_type: ReportType = Field(
        ..., description="Type of report to generate"
    )
    plot_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Plot identifiers for this report",
    )
    format: ReportFormat = Field(
        default=ReportFormat.PDF,
        description="Output format",
    )
    options: Optional[ReportOptions] = Field(
        None, description="Report options"
    )
    title: Optional[str] = Field(
        None, max_length=500, description="Custom title"
    )

    model_config = ConfigDict(extra="forbid")


class ReportBatchRequest(BaseModel):
    """Request to generate multiple reports in batch."""

    report_configs: List[ReportConfigItem] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of report configurations (max 50)",
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Report Schemas - Response
# =============================================================================


class ReportSection(BaseModel):
    """Single section within a generated report."""

    section_id: str = Field(..., description="Section identifier")
    title: str = Field(..., description="Section title")
    content_type: str = Field(
        default="text",
        description="Content type: text, table, chart, map",
    )
    content: Optional[Any] = Field(
        None, description="Section content"
    )
    order: int = Field(default=0, ge=0, description="Display order")

    model_config = ConfigDict(from_attributes=True)


class ReportResult(BaseModel):
    """Result from report generation."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    report_id: str = Field(..., description="Unique report identifier")
    report_type: str = Field(..., description="Report type generated")
    format: str = Field(..., description="Output format")
    status: str = Field(
        default="generated",
        description="Report status: generated, pending, failed",
    )
    title: str = Field(default="", description="Report title")
    summary: str = Field(default="", description="Report executive summary")
    sections: List[ReportSection] = Field(
        default_factory=list, description="Report sections"
    )
    plot_count: int = Field(
        default=0, ge=0, description="Number of plots included"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL for non-JSON formats"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="Report file size in bytes"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Report creation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Download URL expiration timestamp"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


class ReportBatchResponse(BaseModel):
    """Response from batch report generation."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    results: List[ReportResult] = Field(
        default_factory=list, description="Individual report results"
    )
    total: int = Field(..., ge=0, description="Total reports requested")
    successful: int = Field(
        default=0, ge=0, description="Successfully generated reports"
    )
    failed: int = Field(default=0, ge=0, description="Failed generations")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Batch completion timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Job Schemas
# =============================================================================


class BatchJobSubmitRequest(BaseModel):
    """Request to submit an asynchronous batch job."""

    job_type: BatchJobType = Field(
        ..., description="Type of batch job to execute"
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description=(
            "Job parameters. Structure depends on job_type. "
            "For classification: {plots: [...], method: '...'}, "
            "For verification: {plots: [...], commodity: '...'}"
        ),
    )
    priority: int = Field(
        default=5, ge=1, le=10,
        description="Job priority (1 = lowest, 10 = highest)",
    )
    callback_url: Optional[str] = Field(
        None,
        description="Webhook URL for job completion notification",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for job organization and filtering",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "job_type": "verification",
                    "parameters": {
                        "plots": [
                            {
                                "latitude": 5.6037,
                                "longitude": -0.1870,
                                "commodity": "cocoa",
                            }
                        ]
                    },
                    "priority": 7,
                }
            ]
        },
    )


class BatchJobResponse(BaseModel):
    """Response for batch job operations."""

    request_id: str = Field(
        default_factory=_request_id, description="Request correlation ID"
    )
    job_id: str = Field(..., description="Unique batch job identifier")
    job_type: str = Field(..., description="Type of batch job")
    status: BatchJobStatus = Field(
        ..., description="Current job status"
    )
    progress_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Job progress percentage (0-100)",
    )
    total_items: int = Field(
        default=0, ge=0, description="Total items to process"
    )
    completed_items: int = Field(
        default=0, ge=0, description="Items processed so far"
    )
    failed_items: int = Field(
        default=0, ge=0, description="Items that failed processing"
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow, description="Job submission timestamp"
    )
    started_at: Optional[datetime] = Field(
        None, description="Job processing start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    result_url: Optional[str] = Field(
        None, description="URL to download results when complete"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if job failed"
    )
    tags: List[str] = Field(
        default_factory=list, description="Job tags"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# System Schemas
# =============================================================================


class EngineStatus(BaseModel):
    """Status of a single analysis engine."""

    engine_name: str = Field(..., description="Engine identifier")
    status: str = Field(
        default="healthy",
        description="Engine status: healthy, degraded, unavailable",
    )
    version: str = Field(default="1.0.0", description="Engine version")
    last_used: Optional[datetime] = Field(
        None, description="Last usage timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field(
        default="healthy",
        description="Overall service status: healthy, degraded, unhealthy",
    )
    agent_id: str = Field(
        default="GL-EUDR-LUC-005",
        description="Agent identifier",
    )
    agent_name: str = Field(
        default="EUDR Land Use Change Detector Agent",
        description="Agent display name",
    )
    version: str = Field(
        default="1.0.0", description="API version"
    )
    engines_status: List[EngineStatus] = Field(
        default_factory=list,
        description="Status of individual analysis engines",
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Service uptime in seconds"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Health check timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class VersionResponse(BaseModel):
    """API version information response."""

    agent_id: str = Field(
        default="GL-EUDR-LUC-005",
        description="Agent identifier",
    )
    agent_name: str = Field(
        default="EUDR Land Use Change Detector Agent",
        description="Agent display name",
    )
    version: str = Field(default="1.0.0", description="API version")
    engines: List[str] = Field(
        default_factory=lambda: [
            "LandUseClassifier",
            "TransitionDetector",
            "TemporalTrajectoryAnalyzer",
            "CutoffDateVerifier",
        ],
        description="Available analysis engines",
    )
    supported_commodities: List[str] = Field(
        default_factory=lambda: [e.value for e in EUDRCommodity],
        description="Supported EUDR commodities",
    )
    classification_methods: List[str] = Field(
        default_factory=lambda: [e.value for e in ClassificationMethod],
        description="Supported classification methods",
    )
    land_use_categories: List[str] = Field(
        default_factory=lambda: [e.value for e in LandUseCategory],
        description="Supported land use categories",
    )
    transition_types: List[str] = Field(
        default_factory=lambda: [e.value for e in TransitionType],
        description="Supported transition types",
    )
    trajectory_types: List[str] = Field(
        default_factory=lambda: [e.value for e in TrajectoryType],
        description="Supported trajectory types",
    )
    eudr_cutoff_date: str = Field(
        default="2020-12-31",
        description="EUDR deforestation cutoff date",
    )
    report_formats: List[str] = Field(
        default_factory=lambda: [e.value for e in ReportFormat],
        description="Supported report output formats",
    )
    risk_factors: List[str] = Field(
        default_factory=lambda: [
            "transition_magnitude",
            "proximity_to_forest",
            "historical_deforestation_rate",
            "commodity_pressure",
            "governance_score",
            "protected_area_proximity",
            "road_infrastructure_proximity",
            "population_density_change",
        ],
        description="Risk assessment factors",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "AnalysisStatus",
    "BatchJobStatus",
    "BatchJobType",
    "ClassificationMethod",
    "ComplianceVerdict",
    "EUDRCommodity",
    "LandUseCategory",
    "LandUseSubCategory",
    "ReportFormat",
    "ReportType",
    "RiskTier",
    "TrajectoryType",
    "TransitionType",
    # Base models
    "ApiResponse",
    "ErrorResponse",
    "PaginatedMeta",
    "PaginatedResponse",
    "SpectralIndicesData",
    "PlotCoordinate",
    # Classification - Request
    "ClassifyRequest",
    "ClassifyBatchRequest",
    "ClassifyCompareRequest",
    # Classification - Response
    "ClassificationResult",
    "ClassificationBatchResponse",
    "ClassificationCompareResponse",
    "ClassificationHistoryEntry",
    "ClassificationHistoryResponse",
    # Transition - Request
    "TransitionDetectRequest",
    "TransitionBatchRequest",
    "TransitionMatrixRequest",
    # Transition - Response
    "TransitionResult",
    "TransitionBatchResponse",
    "TransitionEvidence",
    "TransitionMatrixCell",
    "TransitionMatrixResponse",
    "TopTransition",
    "TransitionTypeInfo",
    "TransitionTypesListResponse",
    # Trajectory - Request
    "TrajectoryAnalyzeRequest",
    "TrajectoryBatchRequest",
    # Trajectory - Response
    "NDVIDataPoint",
    "ChangeDate",
    "VisualizationData",
    "TrajectoryResult",
    "TrajectoryBatchResponse",
    # Verification - Request
    "VerifyCutoffRequest",
    "VerifyBatchPlot",
    "VerifyBatchRequest",
    "VerifyCompleteRequest",
    # Verification - Response
    "EvidenceItem",
    "EvidencePackage",
    "CutoffVerificationResult",
    "VerificationBatchResponse",
    # Risk - Request
    "RiskAssessRequest",
    "RiskBatchRequest",
    "UrbanAnalyzeRequest",
    "UrbanBatchRequest",
    # Risk - Response
    "RiskFactor",
    "RiskResult",
    "RiskBatchResponse",
    "InfrastructureFeature",
    "PressureCorridor",
    "UrbanResult",
    "UrbanBatchResponse",
    # Report - Request
    "ReportOptions",
    "ReportGenerateRequest",
    "ReportConfigItem",
    "ReportBatchRequest",
    # Report - Response
    "ReportSection",
    "ReportResult",
    "ReportBatchResponse",
    # Batch Jobs
    "BatchJobSubmitRequest",
    "BatchJobResponse",
    # System
    "EngineStatus",
    "HealthResponse",
    "VersionResponse",
]
