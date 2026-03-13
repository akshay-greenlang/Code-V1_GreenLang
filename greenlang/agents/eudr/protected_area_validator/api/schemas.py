# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-022 Protected Area Validator

Pydantic v2 request/response models for the REST API layer covering all
8 engine domains: protected area management, overlap detection, buffer zone
monitoring, designation validation, risk scoring, violation detection,
compliance assessment, and PADDD event monitoring.

All numeric fields use ``Decimal`` for precision (zero-hallucination).
All date/time fields use UTC-aware ``datetime``.
All geospatial coordinates use Decimal for regulatory-grade precision.

Schema Groups (8 domains + common):
    Common: ProvenanceInfo, MetadataSchema, PaginatedMeta, ErrorResponse,
            GeoPointSchema, GeoPolygonSchema, HealthResponse
    1. Protected Area: ProtectedAreaCreateRequest, ProtectedAreaResponse,
       ProtectedAreaUpdateRequest, ProtectedAreaListResponse,
       ProtectedAreaSearchRequest, ProtectedAreaSearchResponse
    2. Overlap: OverlapDetectRequest/Response, OverlapAnalyzeRequest/Response,
       OverlapBulkRequest/Response, OverlapByPlotResponse, OverlapByAreaResponse
    3. Buffer Zone: BufferZoneMonitorRequest/Response,
       BufferZoneViolationsResponse, BufferZoneAnalyzeRequest/Response,
       BufferZoneBulkRequest/Response
    4. Designation: DesignationValidateRequest/Response,
       DesignationStatusResponse, DesignationHistoryResponse
    5. Risk: RiskScoreRequest/Response, RiskHeatmapResponse,
       RiskSummaryResponse, ProximityAlertEntry, ProximityAlertsResponse
    6. Violation: ViolationDetectRequest/Response, ViolationListResponse,
       ViolationResolveRequest/Response, ViolationEscalateRequest/Response
    7. Compliance: ComplianceAssessRequest/Response,
       ComplianceReportResponse, ComplianceAuditTrailResponse
    8. PADDD: PADDDMonitorRequest/Response, PADDDEventsResponse,
       PADDDImpactAssessmentRequest/Response

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_id() -> str:
    """Generate a new UUID4 string identifier."""
    return str(uuid.uuid4())


# =============================================================================
# Enumerations (API-level mirrors for OpenAPI documentation)
# =============================================================================


class ProtectedAreaTypeEnum(str, Enum):
    """IUCN protected area management categories."""

    STRICT_NATURE_RESERVE = "strict_nature_reserve"           # Ia
    WILDERNESS_AREA = "wilderness_area"                       # Ib
    NATIONAL_PARK = "national_park"                           # II
    NATURAL_MONUMENT = "natural_monument"                     # III
    HABITAT_MANAGEMENT = "habitat_management"                 # IV
    PROTECTED_LANDSCAPE = "protected_landscape"               # V
    MANAGED_RESOURCE = "managed_resource"                     # VI
    UNESCO_WORLD_HERITAGE = "unesco_world_heritage"
    RAMSAR_WETLAND = "ramsar_wetland"
    BIOSPHERE_RESERVE = "biosphere_reserve"
    INDIGENOUS_TERRITORY = "indigenous_territory"
    KEY_BIODIVERSITY_AREA = "key_biodiversity_area"
    OTHER = "other"


class DesignationStatusEnum(str, Enum):
    """Protected area designation lifecycle status."""

    DESIGNATED = "designated"
    PROPOSED = "proposed"
    INSCRIBED = "inscribed"
    ADOPTED = "adopted"
    DOWNGRADED = "downgraded"
    DOWNSIZED = "downsized"
    DEGAZETTED = "degazetted"
    UNKNOWN = "unknown"


class OverlapTypeEnum(str, Enum):
    """Types of spatial overlap between plot and protected area."""

    FULL = "full"
    PARTIAL = "partial"
    BUFFER_ONLY = "buffer_only"
    ADJACENT = "adjacent"
    NONE = "none"


class RiskLevelEnum(str, Enum):
    """Risk level classification for protected area proximity."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ViolationTypeEnum(str, Enum):
    """Types of protected area violations."""

    ENCROACHMENT = "encroachment"
    BUFFER_BREACH = "buffer_breach"
    ILLEGAL_CLEARING = "illegal_clearing"
    BOUNDARY_VIOLATION = "boundary_violation"
    DESIGNATION_NON_COMPLIANCE = "designation_non_compliance"
    ACTIVITY_RESTRICTION = "activity_restriction"
    OTHER = "other"


class ViolationStatusEnum(str, Enum):
    """Violation lifecycle status."""

    DETECTED = "detected"
    CONFIRMED = "confirmed"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
    FALSE_POSITIVE = "false_positive"


class PADDDEventTypeEnum(str, Enum):
    """PADDD (Protected Area Downgrading, Downsizing, Degazettement) event types."""

    DOWNGRADE = "downgrade"
    DOWNSIZE = "downsize"
    DEGAZETTE = "degazette"
    PROPOSED_DOWNGRADE = "proposed_downgrade"
    PROPOSED_DOWNSIZE = "proposed_downsize"
    PROPOSED_DEGAZETTE = "proposed_degazette"
    REVERSED = "reversed"


class ComplianceOutcomeEnum(str, Enum):
    """Compliance assessment outcome for protected area validation."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AT_RISK = "at_risk"
    REQUIRES_INVESTIGATION = "requires_investigation"
    REMEDIATION_REQUIRED = "remediation_required"


class EUDRCommodityEnum(str, Enum):
    """EUDR-regulated commodity types per Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class DataSourceEnum(str, Enum):
    """Protected area data source identifiers."""

    WDPA = "wdpa"
    OECM = "oecm"
    NATIONAL_REGISTRY = "national_registry"
    IUCN_RED_LIST = "iucn_red_list"
    UNESCO = "unesco"
    RAMSAR = "ramsar"
    CUSTOM = "custom"


class EscalationLevelEnum(str, Enum):
    """Escalation level for violations."""

    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"


class AuditActionEnum(str, Enum):
    """Audit trail action types."""

    AREA_REGISTERED = "area_registered"
    AREA_UPDATED = "area_updated"
    AREA_ARCHIVED = "area_archived"
    OVERLAP_DETECTED = "overlap_detected"
    VIOLATION_DETECTED = "violation_detected"
    VIOLATION_RESOLVED = "violation_resolved"
    VIOLATION_ESCALATED = "violation_escalated"
    COMPLIANCE_ASSESSED = "compliance_assessed"
    DESIGNATION_VALIDATED = "designation_validated"
    RISK_SCORED = "risk_scored"
    PADDD_DETECTED = "paddd_detected"
    BUFFER_VIOLATION = "buffer_violation"


# =============================================================================
# Common / Shared Schemas
# =============================================================================


class ProvenanceInfo(BaseModel):
    """Provenance tracking information for audit trail."""

    model_config = ConfigDict(populate_by_name=True)

    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail verification"
    )
    processing_time_ms: Decimal = Field(
        ..., description="Processing duration in milliseconds"
    )
    agent_id: str = Field(
        default="GL-EUDR-PAV-022",
        description="Agent identifier",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of operation",
    )


class MetadataSchema(BaseModel):
    """Response metadata for traceability."""

    model_config = ConfigDict(populate_by_name=True)

    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used in this response",
    )
    regulation: str = Field(
        default="EU 2023/1115 (EUDR)",
        description="Applicable regulation",
    )
    articles: List[str] = Field(
        default_factory=lambda: ["Art. 2", "Art. 3", "Art. 9", "Art. 10", "Art. 29"],
        description="Applicable regulatory articles",
    )
    api_version: str = Field(
        default="1.0.0", description="API version"
    )


class PaginatedMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of records")
    limit: int = Field(..., ge=1, description="Records per page")
    offset: int = Field(..., ge=0, description="Number of records skipped")
    has_more: bool = Field(..., description="Whether more pages exist")


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


class GeoPointSchema(BaseModel):
    """Geographic coordinate point with regulatory-grade precision."""

    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="WGS84 latitude in decimal degrees",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="WGS84 longitude in decimal degrees",
    )


class GeoPolygonSchema(BaseModel):
    """Geographic polygon defined by ordered coordinate points."""

    coordinates: List[GeoPointSchema] = Field(
        ...,
        min_length=3,
        description="Ordered list of polygon vertices (minimum 3 points)",
    )

    @field_validator("coordinates")
    @classmethod
    def validate_polygon_closure(cls, v: List[GeoPointSchema]) -> List[GeoPointSchema]:
        """Validate polygon has at least 3 distinct vertices."""
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        return v


class GeoBoundingBoxSchema(BaseModel):
    """Geographic bounding box for spatial queries."""

    min_latitude: Decimal = Field(
        ..., ge=Decimal("-90"), le=Decimal("90"),
        description="Southern boundary latitude",
    )
    max_latitude: Decimal = Field(
        ..., ge=Decimal("-90"), le=Decimal("90"),
        description="Northern boundary latitude",
    )
    min_longitude: Decimal = Field(
        ..., ge=Decimal("-180"), le=Decimal("180"),
        description="Western boundary longitude",
    )
    max_longitude: Decimal = Field(
        ..., ge=Decimal("-180"), le=Decimal("180"),
        description="Eastern boundary longitude",
    )

    @field_validator("max_latitude")
    @classmethod
    def validate_lat_range(cls, v: Decimal, info: Any) -> Decimal:
        """Validate max_latitude >= min_latitude."""
        min_lat = info.data.get("min_latitude")
        if min_lat is not None and v < min_lat:
            raise ValueError("max_latitude must be >= min_latitude")
        return v


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Service health status")
    agent_id: str = Field(
        default="GL-EUDR-PAV-022", description="Agent identifier"
    )
    component: str = Field(
        default="protected-area-validator", description="Component name"
    )
    version: str = Field(default="1.0.0", description="API version")


# =============================================================================
# 1. Protected Area Schemas
# =============================================================================


class ProtectedAreaCreateRequest(BaseModel):
    """Request to register a new protected area."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Protected area name",
    )
    area_type: ProtectedAreaTypeEnum = Field(
        ..., description="IUCN management category or designation type",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    wdpa_id: Optional[str] = Field(
        None, description="World Database on Protected Areas identifier",
    )
    designation_status: DesignationStatusEnum = Field(
        default=DesignationStatusEnum.DESIGNATED,
        description="Current designation status",
    )
    designation_date: Optional[date] = Field(
        None, description="Date of official designation",
    )
    total_area_km2: Decimal = Field(
        ..., gt=Decimal("0"),
        description="Total area in square kilometers",
    )
    marine_area_km2: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Marine portion area in square kilometers",
    )
    boundary: GeoPolygonSchema = Field(
        ..., description="Protected area boundary polygon",
    )
    centroid: Optional[GeoPointSchema] = Field(
        None, description="Protected area centroid point",
    )
    buffer_zone_km: Decimal = Field(
        default=Decimal("5"),
        gt=Decimal("0"),
        le=Decimal("50"),
        description="Buffer zone radius in kilometers (default 5 km)",
    )
    governance_type: Optional[str] = Field(
        None, max_length=255,
        description="Governance type (government, private, community, shared)",
    )
    management_authority: Optional[str] = Field(
        None, max_length=500,
        description="Managing authority or organization",
    )
    data_source: DataSourceEnum = Field(
        default=DataSourceEnum.WDPA,
        description="Data source for this protected area",
    )
    iucn_category: Optional[str] = Field(
        None, max_length=10,
        description="IUCN management category code (Ia, Ib, II-VI)",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Additional notes or metadata",
    )


class ProtectedAreaEntry(BaseModel):
    """A single protected area record."""

    area_id: str = Field(..., description="Unique protected area identifier")
    name: str = Field(..., description="Protected area name")
    area_type: ProtectedAreaTypeEnum = Field(
        ..., description="Area type classification",
    )
    country_code: str = Field(..., description="Country code")
    wdpa_id: Optional[str] = Field(None, description="WDPA identifier")
    designation_status: DesignationStatusEnum = Field(
        ..., description="Designation status",
    )
    designation_date: Optional[date] = Field(
        None, description="Designation date",
    )
    total_area_km2: Decimal = Field(
        ..., description="Total area in km2",
    )
    buffer_zone_km: Decimal = Field(
        ..., description="Buffer zone radius in km",
    )
    centroid_latitude: Optional[Decimal] = Field(
        None, description="Centroid latitude",
    )
    centroid_longitude: Optional[Decimal] = Field(
        None, description="Centroid longitude",
    )
    governance_type: Optional[str] = Field(
        None, description="Governance type",
    )
    management_authority: Optional[str] = Field(
        None, description="Management authority",
    )
    data_source: DataSourceEnum = Field(
        ..., description="Data source",
    )
    iucn_category: Optional[str] = Field(
        None, description="IUCN category",
    )
    is_active: bool = Field(
        default=True, description="Whether area record is active",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp",
    )
    updated_at: Optional[datetime] = Field(
        None, description="Last update timestamp",
    )


class ProtectedAreaResponse(BaseModel):
    """Response for a single protected area."""

    model_config = ConfigDict(populate_by_name=True)

    area: ProtectedAreaEntry = Field(
        ..., description="Protected area details",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class ProtectedAreaUpdateRequest(BaseModel):
    """Request to update a protected area."""

    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = Field(
        None, min_length=1, max_length=500,
        description="Updated name",
    )
    designation_status: Optional[DesignationStatusEnum] = Field(
        None, description="Updated designation status",
    )
    total_area_km2: Optional[Decimal] = Field(
        None, gt=Decimal("0"),
        description="Updated total area",
    )
    buffer_zone_km: Optional[Decimal] = Field(
        None, gt=Decimal("0"), le=Decimal("50"),
        description="Updated buffer zone radius",
    )
    boundary: Optional[GeoPolygonSchema] = Field(
        None, description="Updated boundary polygon",
    )
    governance_type: Optional[str] = Field(
        None, max_length=255,
        description="Updated governance type",
    )
    management_authority: Optional[str] = Field(
        None, max_length=500,
        description="Updated management authority",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Updated notes",
    )


class ProtectedAreaListResponse(BaseModel):
    """Paginated list of protected areas."""

    model_config = ConfigDict(populate_by_name=True)

    areas: List[ProtectedAreaEntry] = Field(
        default_factory=list, description="List of protected areas",
    )
    total_areas: int = Field(
        default=0, ge=0, description="Total matching areas",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class ProtectedAreaSearchRequest(BaseModel):
    """Request for advanced spatial search of protected areas."""

    model_config = ConfigDict(populate_by_name=True)

    center: Optional[GeoPointSchema] = Field(
        None, description="Search center point for radius query",
    )
    radius_km: Optional[Decimal] = Field(
        None, gt=Decimal("0"), le=Decimal("500"),
        description="Search radius in kilometers",
    )
    bounding_box: Optional[GeoBoundingBoxSchema] = Field(
        None, description="Bounding box for spatial search",
    )
    polygon: Optional[GeoPolygonSchema] = Field(
        None, description="Custom polygon for intersection search",
    )
    country_codes: Optional[List[str]] = Field(
        None, description="Filter by country codes",
    )
    area_types: Optional[List[ProtectedAreaTypeEnum]] = Field(
        None, description="Filter by area types",
    )
    designation_statuses: Optional[List[DesignationStatusEnum]] = Field(
        None, description="Filter by designation statuses",
    )
    data_sources: Optional[List[DataSourceEnum]] = Field(
        None, description="Filter by data sources",
    )
    min_area_km2: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Minimum area filter (km2)",
    )
    max_area_km2: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Maximum area filter (km2)",
    )
    include_buffer: bool = Field(
        default=True,
        description="Include buffer zones in spatial search",
    )


class ProtectedAreaSearchResponse(BaseModel):
    """Response for spatial search of protected areas."""

    model_config = ConfigDict(populate_by_name=True)

    areas: List[ProtectedAreaEntry] = Field(
        default_factory=list, description="Matching protected areas",
    )
    total_results: int = Field(
        default=0, ge=0, description="Total matching results",
    )
    search_area_km2: Optional[Decimal] = Field(
        None, description="Total search area in km2",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# 2. Overlap Detection Schemas
# =============================================================================


class OverlapDetectRequest(BaseModel):
    """Request to detect overlaps between a plot and protected areas."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(
        ..., description="Supply chain plot identifier",
    )
    plot_boundary: GeoPolygonSchema = Field(
        ..., description="Plot boundary polygon",
    )
    plot_center: Optional[GeoPointSchema] = Field(
        None, description="Plot centroid for proximity calculations",
    )
    include_buffer_zones: bool = Field(
        default=True,
        description="Include buffer zone overlaps in detection",
    )
    max_distance_km: Decimal = Field(
        default=Decimal("50"),
        gt=Decimal("0"),
        le=Decimal("500"),
        description="Maximum search distance in km",
    )
    area_types: Optional[List[ProtectedAreaTypeEnum]] = Field(
        None, description="Filter by protected area types",
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities produced on the plot",
    )


class OverlapEntry(BaseModel):
    """A single overlap detection result."""

    overlap_id: str = Field(
        default_factory=_new_id, description="Unique overlap identifier",
    )
    area_id: str = Field(
        ..., description="Protected area identifier",
    )
    area_name: str = Field(
        ..., description="Protected area name",
    )
    area_type: ProtectedAreaTypeEnum = Field(
        ..., description="Protected area type",
    )
    country_code: str = Field(
        ..., description="Country code",
    )
    overlap_type: OverlapTypeEnum = Field(
        ..., description="Type of spatial overlap",
    )
    overlap_area_km2: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Area of overlap in km2",
    )
    overlap_percentage: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of plot area overlapping",
    )
    distance_km: Decimal = Field(
        ..., ge=Decimal("0"),
        description="Distance from plot center to area boundary in km",
    )
    buffer_zone_km: Decimal = Field(
        ..., description="Buffer zone radius of the protected area",
    )
    is_within_buffer: bool = Field(
        ..., description="Whether plot is within the buffer zone",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk level based on overlap",
    )
    designation_status: DesignationStatusEnum = Field(
        ..., description="Current designation status",
    )
    iucn_category: Optional[str] = Field(
        None, description="IUCN category",
    )


class OverlapDetectResponse(BaseModel):
    """Response for overlap detection."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(
        ..., description="Analyzed plot identifier",
    )
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Detected overlaps",
    )
    total_overlaps: int = Field(
        default=0, ge=0, description="Total overlaps detected",
    )
    has_direct_overlap: bool = Field(
        default=False,
        description="Whether any direct (non-buffer) overlap exists",
    )
    has_buffer_overlap: bool = Field(
        default=False,
        description="Whether any buffer zone overlap exists",
    )
    highest_risk_level: RiskLevelEnum = Field(
        default=RiskLevelEnum.NEGLIGIBLE,
        description="Highest risk level among all overlaps",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class OverlapAnalyzeRequest(BaseModel):
    """Request for detailed overlap analysis between plot and protected areas."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(
        ..., description="Supply chain plot identifier",
    )
    area_id: str = Field(
        ..., description="Protected area identifier for detailed analysis",
    )
    include_boundary_detail: bool = Field(
        default=True,
        description="Include detailed boundary intersection geometry",
    )
    include_historical: bool = Field(
        default=False,
        description="Include historical overlap changes",
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities produced on the plot",
    )


class OverlapAnalyzeResponse(BaseModel):
    """Response for detailed overlap analysis."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Plot identifier")
    area_id: str = Field(..., description="Protected area identifier")
    area_name: str = Field(..., description="Protected area name")
    overlap_type: OverlapTypeEnum = Field(
        ..., description="Type of overlap",
    )
    overlap_area_km2: Decimal = Field(
        ..., description="Overlap area in km2",
    )
    overlap_percentage_plot: Decimal = Field(
        ..., description="Percentage of plot within protected area",
    )
    overlap_percentage_area: Decimal = Field(
        ..., description="Percentage of protected area overlapped by plot",
    )
    distance_to_boundary_km: Decimal = Field(
        ..., description="Distance from plot center to nearest boundary",
    )
    distance_to_core_km: Optional[Decimal] = Field(
        None, description="Distance to protected area core zone",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Assessed risk level",
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Contributing risk factors",
    )
    regulatory_implications: List[str] = Field(
        default_factory=list,
        description="Regulatory implications of this overlap",
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended next steps",
    )
    historical_changes: Optional[List[Dict[str, Any]]] = Field(
        None, description="Historical overlap changes",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class OverlapBulkRequest(BaseModel):
    """Request for bulk overlap detection across multiple plots."""

    model_config = ConfigDict(populate_by_name=True)

    plots: List[OverlapDetectRequest] = Field(
        ..., min_length=1, max_length=500,
        description="Plots to check for overlaps (max 500)",
    )
    include_buffer_zones: bool = Field(
        default=True,
        description="Include buffer zone overlaps",
    )


class OverlapBulkResultEntry(BaseModel):
    """Result for a single plot in bulk overlap detection."""

    plot_id: str = Field(..., description="Plot identifier")
    total_overlaps: int = Field(
        default=0, ge=0, description="Total overlaps for this plot",
    )
    has_direct_overlap: bool = Field(
        default=False, description="Has direct overlap",
    )
    highest_risk_level: RiskLevelEnum = Field(
        default=RiskLevelEnum.NEGLIGIBLE,
        description="Highest risk level",
    )
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Overlaps detected",
    )
    error: Optional[str] = Field(
        None, description="Error message if processing failed",
    )


class OverlapBulkResponse(BaseModel):
    """Response for bulk overlap detection."""

    model_config = ConfigDict(populate_by_name=True)

    results: List[OverlapBulkResultEntry] = Field(
        default_factory=list, description="Per-plot results",
    )
    total_plots_processed: int = Field(
        default=0, ge=0, description="Total plots processed",
    )
    total_overlaps_found: int = Field(
        default=0, ge=0, description="Total overlaps across all plots",
    )
    plots_with_overlaps: int = Field(
        default=0, ge=0, description="Number of plots with overlaps",
    )
    plots_with_direct_overlap: int = Field(
        default=0, ge=0, description="Plots with direct (non-buffer) overlap",
    )
    failed_count: int = Field(
        default=0, ge=0, description="Number of failed processing",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class OverlapByPlotResponse(BaseModel):
    """Response for overlaps by plot."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Plot identifier")
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Overlaps for this plot",
    )
    total_overlaps: int = Field(
        default=0, ge=0, description="Total overlaps",
    )
    highest_risk_level: RiskLevelEnum = Field(
        default=RiskLevelEnum.NEGLIGIBLE, description="Highest risk level",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class OverlapByAreaResponse(BaseModel):
    """Response for overlaps by protected area."""

    model_config = ConfigDict(populate_by_name=True)

    area_id: str = Field(..., description="Protected area identifier")
    area_name: str = Field(..., description="Protected area name")
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Overlaps for this area",
    )
    total_overlaps: int = Field(
        default=0, ge=0, description="Total overlaps",
    )
    total_affected_plots: int = Field(
        default=0, ge=0, description="Total affected plots",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# 3. Buffer Zone Schemas
# =============================================================================


class BufferZoneMonitorRequest(BaseModel):
    """Request to monitor buffer zone compliance for plots."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(
        ..., description="Supply chain plot identifier",
    )
    plot_center: GeoPointSchema = Field(
        ..., description="Plot centroid coordinates",
    )
    plot_boundary: Optional[GeoPolygonSchema] = Field(
        None, description="Plot boundary for precise analysis",
    )
    buffer_threshold_km: Decimal = Field(
        default=Decimal("5"),
        gt=Decimal("0"),
        le=Decimal("50"),
        description="Buffer distance threshold in km",
    )
    area_types: Optional[List[ProtectedAreaTypeEnum]] = Field(
        None, description="Filter by protected area types",
    )


class BufferZoneMonitorEntry(BaseModel):
    """Buffer zone monitoring result for a single protected area."""

    area_id: str = Field(..., description="Protected area ID")
    area_name: str = Field(..., description="Protected area name")
    area_type: ProtectedAreaTypeEnum = Field(
        ..., description="Protected area type",
    )
    buffer_zone_km: Decimal = Field(
        ..., description="Buffer zone radius in km",
    )
    distance_km: Decimal = Field(
        ..., ge=Decimal("0"),
        description="Distance from plot to area boundary",
    )
    is_within_buffer: bool = Field(
        ..., description="Whether plot is within the buffer zone",
    )
    is_compliant: bool = Field(
        ..., description="Whether buffer zone compliance is met",
    )
    violation_type: Optional[ViolationTypeEnum] = Field(
        None, description="Violation type if non-compliant",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk level assessment",
    )


class BufferZoneMonitorResponse(BaseModel):
    """Response for buffer zone monitoring."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Monitored plot ID")
    results: List[BufferZoneMonitorEntry] = Field(
        default_factory=list, description="Monitoring results",
    )
    total_areas_checked: int = Field(
        default=0, ge=0, description="Total areas checked",
    )
    violations_detected: int = Field(
        default=0, ge=0, description="Number of violations",
    )
    is_compliant: bool = Field(
        default=True, description="Overall buffer compliance",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class BufferZoneViolationEntry(BaseModel):
    """A buffer zone violation record."""

    violation_id: str = Field(
        default_factory=_new_id, description="Violation identifier",
    )
    plot_id: str = Field(..., description="Violating plot ID")
    area_id: str = Field(..., description="Affected protected area ID")
    area_name: str = Field(..., description="Protected area name")
    distance_km: Decimal = Field(
        ..., description="Distance from plot to area boundary",
    )
    buffer_zone_km: Decimal = Field(
        ..., description="Required buffer zone distance",
    )
    penetration_km: Optional[Decimal] = Field(
        None, description="How far into the buffer zone (buffer_km - distance_km)",
    )
    violation_type: ViolationTypeEnum = Field(
        ..., description="Type of buffer violation",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk level",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp",
    )
    status: ViolationStatusEnum = Field(
        default=ViolationStatusEnum.DETECTED,
        description="Violation status",
    )


class BufferZoneViolationsResponse(BaseModel):
    """Paginated list of buffer zone violations."""

    model_config = ConfigDict(populate_by_name=True)

    violations: List[BufferZoneViolationEntry] = Field(
        default_factory=list, description="Buffer zone violations",
    )
    total_violations: int = Field(
        default=0, ge=0, description="Total violations",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class BufferZoneAnalyzeRequest(BaseModel):
    """Request for proximity analysis between a point and protected areas."""

    model_config = ConfigDict(populate_by_name=True)

    center: GeoPointSchema = Field(
        ..., description="Point coordinates for proximity analysis",
    )
    radius_km: Decimal = Field(
        default=Decimal("50"),
        gt=Decimal("0"),
        le=Decimal("500"),
        description="Search radius in km",
    )
    area_types: Optional[List[ProtectedAreaTypeEnum]] = Field(
        None, description="Filter by area types",
    )
    include_distances: bool = Field(
        default=True, description="Include distance calculations",
    )


class BufferZoneAnalyzeEntry(BaseModel):
    """Proximity analysis result for a single protected area."""

    area_id: str = Field(..., description="Protected area ID")
    area_name: str = Field(..., description="Protected area name")
    area_type: ProtectedAreaTypeEnum = Field(
        ..., description="Area type",
    )
    distance_km: Decimal = Field(
        ..., ge=Decimal("0"), description="Distance in km",
    )
    buffer_zone_km: Decimal = Field(
        ..., description="Buffer zone radius",
    )
    is_within_buffer: bool = Field(
        ..., description="Whether point is within buffer",
    )
    bearing_degrees: Optional[Decimal] = Field(
        None, description="Bearing from point to area in degrees",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk assessment",
    )


class BufferZoneAnalyzeResponse(BaseModel):
    """Response for proximity analysis."""

    model_config = ConfigDict(populate_by_name=True)

    center: GeoPointSchema = Field(
        ..., description="Analysis center point",
    )
    nearby_areas: List[BufferZoneAnalyzeEntry] = Field(
        default_factory=list, description="Nearby protected areas",
    )
    total_areas_found: int = Field(
        default=0, ge=0, description="Total areas in range",
    )
    areas_within_buffer: int = Field(
        default=0, ge=0, description="Areas whose buffer zone encompasses the point",
    )
    nearest_area_km: Optional[Decimal] = Field(
        None, description="Distance to nearest protected area",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class BufferZoneBulkRequest(BaseModel):
    """Request for bulk buffer zone monitoring."""

    model_config = ConfigDict(populate_by_name=True)

    plots: List[BufferZoneMonitorRequest] = Field(
        ..., min_length=1, max_length=500,
        description="Plots to monitor (max 500)",
    )


class BufferZoneBulkResultEntry(BaseModel):
    """Result for a single plot in bulk buffer monitoring."""

    plot_id: str = Field(..., description="Plot identifier")
    is_compliant: bool = Field(
        default=True, description="Buffer compliance",
    )
    violations_detected: int = Field(
        default=0, ge=0, description="Violation count",
    )
    nearest_area_km: Optional[Decimal] = Field(
        None, description="Distance to nearest area",
    )
    error: Optional[str] = Field(
        None, description="Error if processing failed",
    )


class BufferZoneBulkResponse(BaseModel):
    """Response for bulk buffer zone monitoring."""

    model_config = ConfigDict(populate_by_name=True)

    results: List[BufferZoneBulkResultEntry] = Field(
        default_factory=list, description="Per-plot results",
    )
    total_plots_processed: int = Field(
        default=0, ge=0, description="Total plots processed",
    )
    compliant_count: int = Field(
        default=0, ge=0, description="Compliant plots count",
    )
    non_compliant_count: int = Field(
        default=0, ge=0, description="Non-compliant plots count",
    )
    total_violations: int = Field(
        default=0, ge=0, description="Total violations across all plots",
    )
    failed_count: int = Field(
        default=0, ge=0, description="Failed processing count",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# 4. Designation Validation Schemas
# =============================================================================


class DesignationValidateRequest(BaseModel):
    """Request to validate designation status of protected areas affecting a plot."""

    model_config = ConfigDict(populate_by_name=True)

    area_ids: Optional[List[str]] = Field(
        None, description="Specific area IDs to validate",
    )
    plot_id: Optional[str] = Field(
        None, description="Plot ID to find related areas",
    )
    plot_center: Optional[GeoPointSchema] = Field(
        None, description="Plot center for spatial lookup",
    )
    radius_km: Optional[Decimal] = Field(
        None, gt=Decimal("0"), le=Decimal("500"),
        description="Search radius for spatial lookup",
    )
    check_paddd: bool = Field(
        default=True,
        description="Check for PADDD events affecting designation",
    )
    check_legal_status: bool = Field(
        default=True,
        description="Verify current legal designation status",
    )


class DesignationValidationEntry(BaseModel):
    """Validation result for a single protected area designation."""

    area_id: str = Field(..., description="Protected area ID")
    area_name: str = Field(..., description="Protected area name")
    designation_status: DesignationStatusEnum = Field(
        ..., description="Current designation status",
    )
    is_valid: bool = Field(
        ..., description="Whether designation is currently valid",
    )
    designation_date: Optional[date] = Field(
        None, description="Original designation date",
    )
    last_verified: Optional[datetime] = Field(
        None, description="Last verification date",
    )
    paddd_events: int = Field(
        default=0, ge=0,
        description="Number of PADDD events affecting this area",
    )
    has_active_paddd: bool = Field(
        default=False,
        description="Whether active PADDD events exist",
    )
    legal_basis: Optional[str] = Field(
        None, description="Legal basis for designation",
    )
    verification_notes: Optional[str] = Field(
        None, description="Notes from verification",
    )


class DesignationValidateResponse(BaseModel):
    """Response for designation validation."""

    model_config = ConfigDict(populate_by_name=True)

    validations: List[DesignationValidationEntry] = Field(
        default_factory=list, description="Validation results",
    )
    total_validated: int = Field(
        default=0, ge=0, description="Total areas validated",
    )
    valid_count: int = Field(
        default=0, ge=0, description="Areas with valid designation",
    )
    invalid_count: int = Field(
        default=0, ge=0, description="Areas with invalid designation",
    )
    paddd_affected_count: int = Field(
        default=0, ge=0, description="Areas affected by PADDD",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class DesignationStatusResponse(BaseModel):
    """Response for current designation status of a protected area."""

    model_config = ConfigDict(populate_by_name=True)

    area_id: str = Field(..., description="Protected area ID")
    area_name: str = Field(..., description="Protected area name")
    designation_status: DesignationStatusEnum = Field(
        ..., description="Current status",
    )
    is_valid: bool = Field(
        ..., description="Whether designation is valid",
    )
    designation_date: Optional[date] = Field(
        None, description="Designation date",
    )
    area_type: ProtectedAreaTypeEnum = Field(
        ..., description="Area type",
    )
    iucn_category: Optional[str] = Field(
        None, description="IUCN category",
    )
    governance_type: Optional[str] = Field(
        None, description="Governance type",
    )
    management_authority: Optional[str] = Field(
        None, description="Management authority",
    )
    has_active_paddd: bool = Field(
        default=False, description="Active PADDD events",
    )
    last_verified: Optional[datetime] = Field(
        None, description="Last verification timestamp",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class DesignationHistoryEntry(BaseModel):
    """A single designation history event."""

    event_date: date = Field(
        ..., description="Event date",
    )
    previous_status: Optional[DesignationStatusEnum] = Field(
        None, description="Previous designation status",
    )
    new_status: DesignationStatusEnum = Field(
        ..., description="New designation status",
    )
    event_type: str = Field(
        ..., description="Event type (designation, amendment, paddd)",
    )
    legal_reference: Optional[str] = Field(
        None, description="Legal reference for the change",
    )
    notes: Optional[str] = Field(
        None, description="Event notes",
    )


class DesignationHistoryResponse(BaseModel):
    """Response for designation history of a protected area."""

    model_config = ConfigDict(populate_by_name=True)

    area_id: str = Field(..., description="Protected area ID")
    area_name: str = Field(..., description="Protected area name")
    current_status: DesignationStatusEnum = Field(
        ..., description="Current designation status",
    )
    history: List[DesignationHistoryEntry] = Field(
        default_factory=list, description="Designation history events",
    )
    total_events: int = Field(
        default=0, ge=0, description="Total history events",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# 5. Risk Scoring Schemas
# =============================================================================


class RiskScoreRequest(BaseModel):
    """Request to calculate protected area risk score for a plot."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(
        ..., description="Supply chain plot identifier",
    )
    plot_center: GeoPointSchema = Field(
        ..., description="Plot centroid coordinates",
    )
    plot_boundary: Optional[GeoPolygonSchema] = Field(
        None, description="Plot boundary for precise analysis",
    )
    plot_area_ha: Optional[Decimal] = Field(
        None, gt=Decimal("0"),
        description="Plot area in hectares",
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities produced on plot",
    )
    include_breakdown: bool = Field(
        default=True,
        description="Include detailed score breakdown",
    )
    custom_weights: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Custom scoring weights (proximity, overlap, designation, iucn, biodiversity)",
    )


class RiskScoreBreakdown(BaseModel):
    """Detailed risk score component breakdown."""

    proximity_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Proximity to protected area score (0-1)",
    )
    proximity_weight: Decimal = Field(
        ..., description="Proximity component weight",
    )
    overlap_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Spatial overlap score (0-1)",
    )
    overlap_weight: Decimal = Field(
        ..., description="Overlap component weight",
    )
    designation_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Designation strength score (0-1)",
    )
    designation_weight: Decimal = Field(
        ..., description="Designation component weight",
    )
    iucn_category_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="IUCN category restrictiveness score (0-1)",
    )
    iucn_weight: Decimal = Field(
        ..., description="IUCN component weight",
    )
    biodiversity_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Biodiversity value score (0-1)",
    )
    biodiversity_weight: Decimal = Field(
        ..., description="Biodiversity component weight",
    )
    weighted_total: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Weighted total score (0-1)",
    )
    multiplier_applied: Optional[Decimal] = Field(
        None, description="Score multiplier applied",
    )
    final_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Final risk score after multipliers",
    )


class RiskScoreResponse(BaseModel):
    """Response for risk score calculation."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Plot identifier")
    risk_level: RiskLevelEnum = Field(
        ..., description="Overall risk level",
    )
    risk_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Risk score (0-1)",
    )
    breakdown: Optional[RiskScoreBreakdown] = Field(
        None, description="Detailed score breakdown",
    )
    nearest_area_name: Optional[str] = Field(
        None, description="Name of nearest protected area",
    )
    nearest_area_distance_km: Optional[Decimal] = Field(
        None, description="Distance to nearest area in km",
    )
    total_areas_in_range: int = Field(
        default=0, ge=0, description="Protected areas in range",
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Contributing risk factors",
    )
    classification_reason: str = Field(
        default="", description="Human-readable classification rationale",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class RiskHeatmapCell(BaseModel):
    """A single cell in the risk heatmap grid."""

    latitude: Decimal = Field(..., description="Cell center latitude")
    longitude: Decimal = Field(..., description="Cell center longitude")
    risk_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Risk score for this cell",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk level",
    )
    nearest_area_km: Optional[Decimal] = Field(
        None, description="Distance to nearest area",
    )
    protected_areas_count: int = Field(
        default=0, ge=0, description="Protected areas affecting this cell",
    )


class RiskHeatmapResponse(BaseModel):
    """Response for risk heatmap data."""

    model_config = ConfigDict(populate_by_name=True)

    cells: List[RiskHeatmapCell] = Field(
        default_factory=list, description="Heatmap grid cells",
    )
    total_cells: int = Field(
        default=0, ge=0, description="Total grid cells",
    )
    grid_resolution_km: Decimal = Field(
        ..., description="Grid cell resolution in km",
    )
    bounding_box: GeoBoundingBoxSchema = Field(
        ..., description="Heatmap bounding box",
    )
    high_risk_cells: int = Field(
        default=0, ge=0, description="Cells with high/critical risk",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class RiskSummaryByCategory(BaseModel):
    """Risk summary grouped by a category."""

    category: str = Field(..., description="Category name")
    count: int = Field(default=0, ge=0, description="Item count")
    average_risk_score: Optional[Decimal] = Field(
        None, description="Average risk score",
    )
    percentage: Optional[Decimal] = Field(
        None, description="Percentage of total",
    )


class RiskSummaryResponse(BaseModel):
    """Response for risk summary statistics."""

    model_config = ConfigDict(populate_by_name=True)

    total_plots_assessed: int = Field(
        default=0, ge=0, description="Total plots assessed",
    )
    average_risk_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Overall average risk score",
    )
    by_risk_level: List[RiskSummaryByCategory] = Field(
        default_factory=list, description="Distribution by risk level",
    )
    by_area_type: List[RiskSummaryByCategory] = Field(
        default_factory=list, description="Distribution by area type",
    )
    by_country: List[RiskSummaryByCategory] = Field(
        default_factory=list, description="Distribution by country",
    )
    critical_plots_count: int = Field(
        default=0, ge=0, description="Plots with critical risk",
    )
    high_risk_plots_count: int = Field(
        default=0, ge=0, description="Plots with high risk",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class ProximityAlertEntry(BaseModel):
    """A high-risk proximity alert."""

    alert_id: str = Field(
        default_factory=_new_id, description="Alert identifier",
    )
    plot_id: str = Field(..., description="Plot identifier")
    area_id: str = Field(..., description="Protected area ID")
    area_name: str = Field(..., description="Protected area name")
    distance_km: Decimal = Field(
        ..., ge=Decimal("0"), description="Distance in km",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk level",
    )
    risk_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Risk score",
    )
    alert_reason: str = Field(
        ..., description="Reason for the alert",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp",
    )


class ProximityAlertsResponse(BaseModel):
    """Response for proximity alerts."""

    model_config = ConfigDict(populate_by_name=True)

    alerts: List[ProximityAlertEntry] = Field(
        default_factory=list, description="Proximity alerts",
    )
    total_alerts: int = Field(
        default=0, ge=0, description="Total alerts",
    )
    critical_count: int = Field(
        default=0, ge=0, description="Critical alerts",
    )
    high_count: int = Field(
        default=0, ge=0, description="High-risk alerts",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# 6. Violation Detection Schemas
# =============================================================================


class ViolationDetectRequest(BaseModel):
    """Request to detect protected area violations for a plot."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(
        ..., description="Supply chain plot identifier",
    )
    plot_boundary: GeoPolygonSchema = Field(
        ..., description="Plot boundary polygon",
    )
    plot_center: Optional[GeoPointSchema] = Field(
        None, description="Plot centroid",
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities produced on plot",
    )
    include_buffer_violations: bool = Field(
        default=True, description="Include buffer zone violations",
    )
    include_designation_violations: bool = Field(
        default=True, description="Include designation non-compliance",
    )


class ViolationEntry(BaseModel):
    """A single violation detection result."""

    violation_id: str = Field(
        default_factory=_new_id, description="Violation identifier",
    )
    plot_id: str = Field(..., description="Plot identifier")
    area_id: str = Field(..., description="Protected area ID")
    area_name: str = Field(..., description="Protected area name")
    violation_type: ViolationTypeEnum = Field(
        ..., description="Type of violation",
    )
    status: ViolationStatusEnum = Field(
        default=ViolationStatusEnum.DETECTED,
        description="Violation status",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk level",
    )
    overlap_area_km2: Optional[Decimal] = Field(
        None, description="Violation overlap area in km2",
    )
    distance_km: Optional[Decimal] = Field(
        None, description="Distance to boundary",
    )
    description: str = Field(
        default="", description="Violation description",
    )
    regulatory_reference: Optional[str] = Field(
        None, description="Applicable regulatory reference",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp",
    )
    resolved_at: Optional[datetime] = Field(
        None, description="Resolution timestamp",
    )


class ViolationDetectResponse(BaseModel):
    """Response for violation detection."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Analyzed plot ID")
    violations: List[ViolationEntry] = Field(
        default_factory=list, description="Detected violations",
    )
    total_violations: int = Field(
        default=0, ge=0, description="Total violations",
    )
    has_violations: bool = Field(
        default=False, description="Whether violations detected",
    )
    highest_risk_level: RiskLevelEnum = Field(
        default=RiskLevelEnum.NEGLIGIBLE,
        description="Highest risk level",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class ViolationListResponse(BaseModel):
    """Paginated list of violations."""

    model_config = ConfigDict(populate_by_name=True)

    violations: List[ViolationEntry] = Field(
        default_factory=list, description="Violations",
    )
    total_violations: int = Field(
        default=0, ge=0, description="Total violations",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class ViolationResolveRequest(BaseModel):
    """Request to resolve a violation."""

    model_config = ConfigDict(populate_by_name=True)

    resolution: str = Field(
        ..., max_length=2000,
        description="Resolution description (confirmed, remediated, false_positive)",
    )
    root_cause: Optional[str] = Field(
        None, max_length=2000, description="Root cause analysis",
    )
    corrective_actions: Optional[List[str]] = Field(
        None, description="Corrective actions taken",
    )
    evidence_urls: Optional[List[str]] = Field(
        None, description="Supporting evidence URLs",
    )
    is_false_positive: bool = Field(
        default=False, description="Mark as false positive",
    )


class ViolationResolveResponse(BaseModel):
    """Response for violation resolution."""

    model_config = ConfigDict(populate_by_name=True)

    violation_id: str = Field(..., description="Resolved violation ID")
    previous_status: ViolationStatusEnum = Field(
        ..., description="Previous status",
    )
    new_status: ViolationStatusEnum = Field(
        ..., description="New status",
    )
    resolved_by: str = Field(
        ..., description="User who resolved",
    )
    resolved_at: datetime = Field(
        default_factory=_utcnow, description="Resolution timestamp",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class ViolationEscalateRequest(BaseModel):
    """Request to escalate a violation."""

    model_config = ConfigDict(populate_by_name=True)

    escalation_level: EscalationLevelEnum = Field(
        ..., description="Escalation level",
    )
    reason: str = Field(
        ..., max_length=2000, description="Escalation reason",
    )
    escalate_to: Optional[str] = Field(
        None, description="User/team to escalate to",
    )
    requires_authority_notification: bool = Field(
        default=False,
        description="Whether competent authority notification is required",
    )


class ViolationEscalateResponse(BaseModel):
    """Response for violation escalation."""

    model_config = ConfigDict(populate_by_name=True)

    violation_id: str = Field(..., description="Escalated violation ID")
    previous_status: ViolationStatusEnum = Field(
        ..., description="Previous status",
    )
    new_status: ViolationStatusEnum = Field(
        ..., description="New status (escalated)",
    )
    escalation_level: EscalationLevelEnum = Field(
        ..., description="Escalation level set",
    )
    escalated_by: str = Field(
        ..., description="User who escalated",
    )
    escalated_at: datetime = Field(
        default_factory=_utcnow, description="Escalation timestamp",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# 7. Compliance Assessment Schemas
# =============================================================================


class ComplianceAssessRequest(BaseModel):
    """Request for full protected area compliance assessment."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(
        ..., description="Supply chain plot identifier",
    )
    plot_boundary: GeoPolygonSchema = Field(
        ..., description="Plot boundary polygon",
    )
    plot_center: Optional[GeoPointSchema] = Field(
        None, description="Plot centroid",
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities produced on the plot",
    )
    operator_id: Optional[str] = Field(
        None, description="EUDR operator identifier",
    )
    include_overlap_analysis: bool = Field(
        default=True, description="Include overlap analysis",
    )
    include_buffer_analysis: bool = Field(
        default=True, description="Include buffer zone analysis",
    )
    include_designation_check: bool = Field(
        default=True, description="Include designation validation",
    )
    include_paddd_check: bool = Field(
        default=True, description="Include PADDD event check",
    )
    include_risk_score: bool = Field(
        default=True, description="Include risk score calculation",
    )


class ComplianceAssessResponse(BaseModel):
    """Response for full compliance assessment."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Assessed plot ID")
    compliance_outcome: ComplianceOutcomeEnum = Field(
        ..., description="Compliance assessment outcome",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Overall risk level",
    )
    risk_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("1"),
        description="Overall risk score",
    )
    total_overlaps: int = Field(
        default=0, ge=0, description="Protected area overlaps found",
    )
    total_violations: int = Field(
        default=0, ge=0, description="Violations detected",
    )
    buffer_violations: int = Field(
        default=0, ge=0, description="Buffer zone violations",
    )
    has_direct_overlap: bool = Field(
        default=False, description="Direct overlap with protected area",
    )
    designation_issues: int = Field(
        default=0, ge=0, description="Designation issues found",
    )
    paddd_events: int = Field(
        default=0, ge=0, description="Active PADDD events",
    )
    nearest_area_name: Optional[str] = Field(
        None, description="Nearest protected area name",
    )
    nearest_area_distance_km: Optional[Decimal] = Field(
        None, description="Distance to nearest area",
    )
    regulatory_articles: List[str] = Field(
        default_factory=lambda: ["Art. 2", "Art. 3", "Art. 9", "Art. 10", "Art. 29"],
        description="Applicable regulatory articles",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Compliance recommendations",
    )
    assessment_rationale: str = Field(
        default="", description="Compliance assessment rationale",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class ComplianceReportResponse(BaseModel):
    """Response for compliance report generation."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Plot identifier")
    report_id: str = Field(
        default_factory=_new_id, description="Report identifier",
    )
    compliance_outcome: ComplianceOutcomeEnum = Field(
        ..., description="Compliance outcome",
    )
    risk_level: RiskLevelEnum = Field(
        ..., description="Risk level",
    )
    overlaps_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Overlap analysis summary",
    )
    violations_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Violations summary",
    )
    buffer_zone_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Buffer zone analysis summary",
    )
    designation_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Designation validation summary",
    )
    risk_score_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Risk scoring summary",
    )
    paddd_summary: Dict[str, Any] = Field(
        default_factory=dict, description="PADDD events summary",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow, description="Report generation timestamp",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class AuditTrailEntry(BaseModel):
    """A single audit trail record."""

    entry_id: str = Field(
        default_factory=_new_id, description="Audit entry ID",
    )
    action: AuditActionEnum = Field(
        ..., description="Action performed",
    )
    entity_type: str = Field(
        ..., description="Entity type (plot, area, violation)",
    )
    entity_id: str = Field(
        ..., description="Entity identifier",
    )
    performed_by: str = Field(
        ..., description="User who performed the action",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Action timestamp",
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional details",
    )
    ip_address: Optional[str] = Field(
        None, description="Client IP address",
    )


class ComplianceAuditTrailResponse(BaseModel):
    """Response for compliance audit trail."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Plot identifier")
    entries: List[AuditTrailEntry] = Field(
        default_factory=list, description="Audit trail entries",
    )
    total_entries: int = Field(
        default=0, ge=0, description="Total entries",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# 8. PADDD Monitoring Schemas
# =============================================================================


class PADDDMonitorRequest(BaseModel):
    """Request to monitor PADDD events for protected areas near plots."""

    model_config = ConfigDict(populate_by_name=True)

    plot_ids: Optional[List[str]] = Field(
        None, description="Plot IDs to check for PADDD impact",
    )
    area_ids: Optional[List[str]] = Field(
        None, description="Protected area IDs to monitor",
    )
    country_codes: Optional[List[str]] = Field(
        None, description="Country codes to monitor",
    )
    center: Optional[GeoPointSchema] = Field(
        None, description="Center point for spatial search",
    )
    radius_km: Optional[Decimal] = Field(
        None, gt=Decimal("0"), le=Decimal("500"),
        description="Search radius in km",
    )
    event_types: Optional[List[PADDDEventTypeEnum]] = Field(
        None, description="Filter by PADDD event types",
    )
    since_date: Optional[date] = Field(
        None, description="Only events after this date",
    )


class PADDDEventEntry(BaseModel):
    """A single PADDD event record."""

    event_id: str = Field(
        default_factory=_new_id, description="Event identifier",
    )
    area_id: str = Field(..., description="Affected protected area ID")
    area_name: str = Field(..., description="Protected area name")
    country_code: str = Field(..., description="Country code")
    event_type: PADDDEventTypeEnum = Field(
        ..., description="PADDD event type",
    )
    event_date: Optional[date] = Field(
        None, description="Event date",
    )
    enacted_date: Optional[date] = Field(
        None, description="Date event was enacted",
    )
    area_affected_km2: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Area affected by PADDD event in km2",
    )
    percentage_affected: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of area affected",
    )
    legal_mechanism: Optional[str] = Field(
        None, description="Legal mechanism for the PADDD event",
    )
    description: Optional[str] = Field(
        None, max_length=2000, description="Event description",
    )
    source: Optional[str] = Field(
        None, description="Data source for this event",
    )
    is_reversed: bool = Field(
        default=False, description="Whether event has been reversed",
    )


class PADDDMonitorResponse(BaseModel):
    """Response for PADDD event monitoring."""

    model_config = ConfigDict(populate_by_name=True)

    events: List[PADDDEventEntry] = Field(
        default_factory=list, description="PADDD events detected",
    )
    total_events: int = Field(
        default=0, ge=0, description="Total events",
    )
    areas_affected: int = Field(
        default=0, ge=0, description="Number of areas affected",
    )
    active_events: int = Field(
        default=0, ge=0, description="Active (non-reversed) events",
    )
    by_event_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by type",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class PADDDEventsResponse(BaseModel):
    """Paginated list of PADDD events."""

    model_config = ConfigDict(populate_by_name=True)

    events: List[PADDDEventEntry] = Field(
        default_factory=list, description="PADDD events",
    )
    total_events: int = Field(
        default=0, ge=0, description="Total events",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


class PADDDImpactAssessmentRequest(BaseModel):
    """Request for PADDD impact assessment on supply chain plots."""

    model_config = ConfigDict(populate_by_name=True)

    event_id: Optional[str] = Field(
        None, description="Specific PADDD event to assess",
    )
    area_id: Optional[str] = Field(
        None, description="Protected area to assess impact for",
    )
    plot_ids: Optional[List[str]] = Field(
        None, description="Specific plots to assess",
    )
    include_supply_chain_impact: bool = Field(
        default=True,
        description="Include supply chain impact analysis",
    )
    include_risk_reassessment: bool = Field(
        default=True,
        description="Include risk score reassessment",
    )


class PADDDImpactAssessmentResponse(BaseModel):
    """Response for PADDD impact assessment."""

    model_config = ConfigDict(populate_by_name=True)

    assessment_id: str = Field(
        default_factory=_new_id, description="Assessment identifier",
    )
    event_id: Optional[str] = Field(
        None, description="PADDD event assessed",
    )
    area_id: Optional[str] = Field(
        None, description="Protected area assessed",
    )
    plots_affected: int = Field(
        default=0, ge=0, description="Number of plots affected",
    )
    risk_change_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Risk level changes (increased, decreased, unchanged)",
    )
    supply_chain_impact: Optional[Dict[str, Any]] = Field(
        None, description="Supply chain impact analysis",
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Impact mitigation recommendations",
    )
    assessment_rationale: str = Field(
        default="", description="Assessment rationale",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata",
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "ProtectedAreaTypeEnum",
    "DesignationStatusEnum",
    "OverlapTypeEnum",
    "RiskLevelEnum",
    "ViolationTypeEnum",
    "ViolationStatusEnum",
    "PADDDEventTypeEnum",
    "ComplianceOutcomeEnum",
    "EUDRCommodityEnum",
    "DataSourceEnum",
    "EscalationLevelEnum",
    "AuditActionEnum",
    # Common
    "ProvenanceInfo",
    "MetadataSchema",
    "PaginatedMeta",
    "ErrorResponse",
    "GeoPointSchema",
    "GeoPolygonSchema",
    "GeoBoundingBoxSchema",
    "HealthResponse",
    # Protected Area
    "ProtectedAreaCreateRequest",
    "ProtectedAreaEntry",
    "ProtectedAreaResponse",
    "ProtectedAreaUpdateRequest",
    "ProtectedAreaListResponse",
    "ProtectedAreaSearchRequest",
    "ProtectedAreaSearchResponse",
    # Overlap
    "OverlapDetectRequest",
    "OverlapEntry",
    "OverlapDetectResponse",
    "OverlapAnalyzeRequest",
    "OverlapAnalyzeResponse",
    "OverlapBulkRequest",
    "OverlapBulkResultEntry",
    "OverlapBulkResponse",
    "OverlapByPlotResponse",
    "OverlapByAreaResponse",
    # Buffer Zone
    "BufferZoneMonitorRequest",
    "BufferZoneMonitorEntry",
    "BufferZoneMonitorResponse",
    "BufferZoneViolationEntry",
    "BufferZoneViolationsResponse",
    "BufferZoneAnalyzeRequest",
    "BufferZoneAnalyzeEntry",
    "BufferZoneAnalyzeResponse",
    "BufferZoneBulkRequest",
    "BufferZoneBulkResultEntry",
    "BufferZoneBulkResponse",
    # Designation
    "DesignationValidateRequest",
    "DesignationValidationEntry",
    "DesignationValidateResponse",
    "DesignationStatusResponse",
    "DesignationHistoryEntry",
    "DesignationHistoryResponse",
    # Risk
    "RiskScoreRequest",
    "RiskScoreBreakdown",
    "RiskScoreResponse",
    "RiskHeatmapCell",
    "RiskHeatmapResponse",
    "RiskSummaryByCategory",
    "RiskSummaryResponse",
    "ProximityAlertEntry",
    "ProximityAlertsResponse",
    # Violation
    "ViolationDetectRequest",
    "ViolationEntry",
    "ViolationDetectResponse",
    "ViolationListResponse",
    "ViolationResolveRequest",
    "ViolationResolveResponse",
    "ViolationEscalateRequest",
    "ViolationEscalateResponse",
    # Compliance
    "ComplianceAssessRequest",
    "ComplianceAssessResponse",
    "ComplianceReportResponse",
    "AuditTrailEntry",
    "ComplianceAuditTrailResponse",
    # PADDD
    "PADDDMonitorRequest",
    "PADDDEventEntry",
    "PADDDMonitorResponse",
    "PADDDEventsResponse",
    "PADDDImpactAssessmentRequest",
    "PADDDImpactAssessmentResponse",
]
