# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-020 Deforestation Alert System

Pydantic v2 request/response models for the REST API layer covering all
8 engine domains: satellite change detection, alert management, severity
classification, spatial buffer monitoring, EUDR cutoff date verification,
historical baseline comparison, alert workflow management, and compliance
impact assessment.

All numeric fields use ``Decimal`` for precision (zero-hallucination).
All date/time fields use UTC-aware ``datetime``.
All geospatial coordinates use Decimal for regulatory-grade precision.

Schema Groups (8 domains + common):
    Common: ProvenanceInfo, MetadataSchema, PaginatedMeta, ErrorResponse,
            GeoPointSchema, HealthResponse
    1. Satellite: SatelliteDetectionRequest/Response, SatelliteScanRequest/Response,
       SatelliteSourcesResponse, SatelliteImageryResponse
    2. Alert: AlertCreateRequest, AlertListResponse, AlertDetailResponse,
       AlertBatchRequest/Response, AlertSummaryResponse, AlertStatisticsResponse
    3. Severity: SeverityClassifyRequest/Response, SeverityReclassifyRequest/Response,
       SeverityThresholdsResponse, SeverityDistributionResponse
    4. Buffer: BufferCreateRequest/Response, BufferUpdateRequest, BufferCheckRequest/Response,
       BufferViolationsResponse, BufferZonesResponse
    5. Cutoff: CutoffVerifyRequest/Response, CutoffBatchVerifyRequest/Response,
       CutoffEvidenceResponse, CutoffTimelineResponse
    6. Baseline: BaselineEstablishRequest/Response, BaselineCompareRequest/Response,
       BaselineUpdateRequest, BaselineCoverageResponse
    7. Workflow: WorkflowTriageRequest/Response, WorkflowAssignRequest,
       WorkflowInvestigateRequest, WorkflowResolveRequest, WorkflowEscalateRequest,
       WorkflowSLAResponse
    8. Compliance: ComplianceAssessRequest/Response, AffectedProductsResponse,
       ComplianceRecommendationsResponse, RemediationRequest/Response

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-020 Deforestation Alert System (GL-EUDR-DAS-020)
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


class SatelliteSourceEnum(str, Enum):
    """Satellite data source identifiers."""

    SENTINEL2 = "sentinel2"
    LANDSAT = "landsat"
    GLAD = "glad"
    HANSEN_GFC = "hansen_gfc"
    RADD = "radd"
    PLANET = "planet"
    CUSTOM = "custom"


class ChangeTypeEnum(str, Enum):
    """Types of detected land cover change."""

    DEFORESTATION = "deforestation"
    DEGRADATION = "degradation"
    DISTURBANCE = "disturbance"
    RECOVERY = "recovery"
    NO_CHANGE = "no_change"
    UNKNOWN = "unknown"


class AlertSeverityEnum(str, Enum):
    """Alert severity levels for deforestation events."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class AlertStatusEnum(str, Enum):
    """Alert lifecycle status values."""

    NEW = "new"
    TRIAGED = "triaged"
    ASSIGNED = "assigned"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
    FALSE_POSITIVE = "false_positive"


class BufferTypeEnum(str, Enum):
    """Spatial buffer geometry types."""

    CIRCULAR = "circular"
    POLYGON = "polygon"
    ADAPTIVE = "adaptive"


class CutoffResultEnum(str, Enum):
    """EUDR cutoff date verification outcomes."""

    PRE_CUTOFF = "pre_cutoff"
    POST_CUTOFF = "post_cutoff"
    GRACE_PERIOD = "grace_period"
    INCONCLUSIVE = "inconclusive"


class ComplianceOutcomeEnum(str, Enum):
    """Compliance impact assessment outcomes."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AT_RISK = "at_risk"
    REQUIRES_INVESTIGATION = "requires_investigation"
    REMEDIATION_REQUIRED = "remediation_required"


class WorkflowActionEnum(str, Enum):
    """Alert workflow actions."""

    TRIAGE = "triage"
    ASSIGN = "assign"
    INVESTIGATE = "investigate"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    DISMISS = "dismiss"
    REOPEN = "reopen"


class SpectralIndexEnum(str, Enum):
    """Spectral vegetation indices for change detection."""

    NDVI = "ndvi"
    EVI = "evi"
    NBR = "nbr"
    NDMI = "ndmi"
    SAVI = "savi"


class EUDRCommodityEnum(str, Enum):
    """EUDR-regulated commodity types per Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class EvidenceQualityEnum(str, Enum):
    """Quality classification for temporal evidence."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class RemediationActionEnum(str, Enum):
    """Types of remediation actions for non-compliant alerts."""

    SUPPLIER_SUSPENSION = "supplier_suspension"
    ALTERNATIVE_SOURCING = "alternative_sourcing"
    ENHANCED_MONITORING = "enhanced_monitoring"
    CERTIFICATION_REVIEW = "certification_review"
    SUPPLY_CHAIN_AUDIT = "supply_chain_audit"
    MARKET_WITHDRAWAL = "market_withdrawal"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"
    CORRECTIVE_ACTION_PLAN = "corrective_action_plan"


class WorkflowPriorityEnum(str, Enum):
    """Workflow processing priority levels."""

    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SLAStatusEnum(str, Enum):
    """SLA compliance status."""

    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    COMPLETED = "completed"


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
        default="GL-EUDR-DAS-020",
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
        default_factory=lambda: ["Art. 2", "Art. 9", "Art. 10", "Art. 11", "Art. 31"],
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


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Service health status")
    agent_id: str = Field(
        default="GL-EUDR-DAS-020", description="Agent identifier"
    )
    component: str = Field(
        default="deforestation-alert-system", description="Component name"
    )
    version: str = Field(default="1.0.0", description="API version")


# =============================================================================
# 1. Satellite Schemas
# =============================================================================


class SatelliteDetectionRequest(BaseModel):
    """Request to trigger satellite change detection for an area."""

    model_config = ConfigDict(populate_by_name=True)

    center: GeoPointSchema = Field(
        ..., description="Center point of the detection area"
    )
    radius_km: Decimal = Field(
        default=Decimal("10"),
        gt=Decimal("0"),
        le=Decimal("50"),
        description="Detection radius in kilometers (1-50 km)",
    )
    sources: Optional[List[SatelliteSourceEnum]] = Field(
        None,
        description="Satellite sources to use (default: all enabled)",
    )
    spectral_indices: Optional[List[SpectralIndexEnum]] = Field(
        None,
        description="Spectral indices to compute (default: all)",
    )
    start_date: Optional[date] = Field(
        None, description="Analysis start date (default: 30 days ago)"
    )
    end_date: Optional[date] = Field(
        None, description="Analysis end date (default: today)"
    )
    min_confidence: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Minimum confidence threshold (0-1)",
    )
    max_cloud_cover_pct: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Maximum cloud cover percentage (0-100)",
    )
    plot_ids: Optional[List[str]] = Field(
        None, description="Specific plot IDs to monitor"
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: Optional[date], info: Any) -> Optional[date]:
        """Validate end_date >= start_date when both provided."""
        if v is not None and info.data.get("start_date") is not None:
            if v < info.data["start_date"]:
                raise ValueError("end_date must be >= start_date")
        return v


class DetectionEntry(BaseModel):
    """A single satellite detection result."""

    detection_id: str = Field(
        default_factory=_new_id, description="Unique detection identifier"
    )
    source: SatelliteSourceEnum = Field(
        ..., description="Satellite data source"
    )
    latitude: Decimal = Field(..., description="Detection center latitude")
    longitude: Decimal = Field(..., description="Detection center longitude")
    area_ha: Decimal = Field(
        ..., ge=Decimal("0"), description="Affected area in hectares"
    )
    change_type: ChangeTypeEnum = Field(
        ..., description="Type of land cover change detected"
    )
    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Detection confidence score (0-1)",
    )
    detection_date: date = Field(..., description="Date of detection")
    spectral_changes: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Spectral index changes (e.g. ndvi_change: -0.25)",
    )
    cloud_cover_pct: Optional[int] = Field(
        None, description="Cloud cover percentage at detection"
    )
    resolution_m: Optional[int] = Field(
        None, description="Spatial resolution in meters"
    )


class SatelliteDetectionResponse(BaseModel):
    """Response for satellite change detection operation."""

    model_config = ConfigDict(populate_by_name=True)

    detection_id: str = Field(
        default_factory=_new_id,
        description="Unique identifier for this detection run",
    )
    detections: List[DetectionEntry] = Field(
        default_factory=list, description="List of detected changes"
    )
    total_detections: int = Field(
        default=0, ge=0, description="Total number of detections"
    )
    sources_queried: List[SatelliteSourceEnum] = Field(
        default_factory=list, description="Satellite sources queried"
    )
    area_scanned_km2: Optional[Decimal] = Field(
        None, description="Total area scanned in square kilometers"
    )
    deforestation_detected: bool = Field(
        default=False, description="Whether deforestation was detected"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class SatelliteScanRequest(BaseModel):
    """Request to scan a specific area with a specific satellite source."""

    model_config = ConfigDict(populate_by_name=True)

    source: SatelliteSourceEnum = Field(
        ..., description="Satellite source to use"
    )
    polygon: Optional[GeoPolygonSchema] = Field(
        None, description="Polygon area to scan"
    )
    center: Optional[GeoPointSchema] = Field(
        None, description="Center point for circular scan"
    )
    radius_km: Optional[Decimal] = Field(
        None,
        gt=Decimal("0"),
        le=Decimal("50"),
        description="Scan radius in kilometers",
    )
    date_range_start: date = Field(
        ..., description="Scan period start date"
    )
    date_range_end: date = Field(
        ..., description="Scan period end date"
    )
    spectral_indices: List[SpectralIndexEnum] = Field(
        default_factory=lambda: [SpectralIndexEnum.NDVI],
        description="Spectral indices to compute",
    )
    max_cloud_cover_pct: int = Field(
        default=20, ge=0, le=100, description="Maximum cloud cover"
    )


class SatelliteScanResponse(BaseModel):
    """Response for a targeted satellite scan."""

    model_config = ConfigDict(populate_by_name=True)

    scan_id: str = Field(
        default_factory=_new_id, description="Unique scan identifier"
    )
    source: SatelliteSourceEnum = Field(
        ..., description="Satellite source used"
    )
    scenes_analyzed: int = Field(
        default=0, ge=0, description="Number of satellite scenes analyzed"
    )
    detections: List[DetectionEntry] = Field(
        default_factory=list, description="Detected changes"
    )
    total_detections: int = Field(
        default=0, ge=0, description="Total detections count"
    )
    area_scanned_km2: Decimal = Field(
        default=Decimal("0"), description="Total area scanned (km2)"
    )
    cloud_cover_avg_pct: Optional[Decimal] = Field(
        None, description="Average cloud cover across scenes"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class SatelliteSourceInfo(BaseModel):
    """Information about an available satellite data source."""

    source: SatelliteSourceEnum = Field(
        ..., description="Satellite source identifier"
    )
    name: str = Field(..., description="Human-readable source name")
    resolution_m: int = Field(
        ..., description="Spatial resolution in meters"
    )
    revisit_days: int = Field(
        ..., description="Revisit period in days"
    )
    enabled: bool = Field(
        default=True, description="Whether this source is enabled"
    )
    coverage: str = Field(
        default="global", description="Geographic coverage"
    )
    data_type: str = Field(
        default="optical", description="Data type (optical, radar, etc.)"
    )


class SatelliteSourcesResponse(BaseModel):
    """Response listing available satellite data sources."""

    model_config = ConfigDict(populate_by_name=True)

    sources: List[SatelliteSourceInfo] = Field(
        default_factory=list, description="Available satellite sources"
    )
    total_sources: int = Field(
        default=0, ge=0, description="Total number of sources"
    )
    enabled_count: int = Field(
        default=0, ge=0, description="Number of enabled sources"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ImageryMetadata(BaseModel):
    """Metadata for a satellite imagery scene."""

    scene_id: str = Field(..., description="Unique scene identifier")
    source: SatelliteSourceEnum = Field(
        ..., description="Satellite source"
    )
    acquisition_date: date = Field(
        ..., description="Date of image acquisition"
    )
    cloud_cover_pct: Decimal = Field(
        ..., description="Cloud cover percentage"
    )
    resolution_m: int = Field(
        ..., description="Spatial resolution in meters"
    )
    bands: Optional[List[str]] = Field(
        None, description="Available spectral bands"
    )
    thumbnail_url: Optional[str] = Field(
        None, description="URL to thumbnail preview"
    )


class SatelliteImageryResponse(BaseModel):
    """Response with imagery metadata for a detection."""

    model_config = ConfigDict(populate_by_name=True)

    detection_id: str = Field(
        ..., description="Detection identifier"
    )
    imagery: List[ImageryMetadata] = Field(
        default_factory=list, description="Associated imagery scenes"
    )
    total_scenes: int = Field(
        default=0, ge=0, description="Total number of scenes"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 2. Alert Schemas
# =============================================================================


class AlertCreateRequest(BaseModel):
    """Request to create a manual deforestation alert."""

    model_config = ConfigDict(populate_by_name=True)

    detection_id: Optional[str] = Field(
        None, description="Associated satellite detection ID"
    )
    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Alert location latitude",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Alert location longitude",
    )
    area_ha: Decimal = Field(
        ..., gt=Decimal("0"), description="Affected area in hectares"
    )
    change_type: ChangeTypeEnum = Field(
        default=ChangeTypeEnum.DEFORESTATION,
        description="Type of change detected",
    )
    severity: Optional[AlertSeverityEnum] = Field(
        None, description="Manual severity classification"
    )
    country_code: Optional[str] = Field(
        None, description="ISO 3166-1 alpha-2 country code"
    )
    description: Optional[str] = Field(
        None, max_length=2000, description="Alert description"
    )
    source: Optional[SatelliteSourceEnum] = Field(
        None, description="Primary satellite source"
    )
    confidence: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Detection confidence (0-1)",
    )
    affected_plot_ids: Optional[List[str]] = Field(
        None, description="Associated supply chain plot IDs"
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Affected EUDR-regulated commodities"
    )


class AlertEntry(BaseModel):
    """Summary of a single deforestation alert."""

    alert_id: str = Field(..., description="Unique alert identifier")
    detection_id: Optional[str] = Field(
        None, description="Associated detection ID"
    )
    latitude: Decimal = Field(..., description="Alert location latitude")
    longitude: Decimal = Field(..., description="Alert location longitude")
    area_ha: Decimal = Field(..., description="Affected area in hectares")
    change_type: ChangeTypeEnum = Field(
        ..., description="Type of change"
    )
    severity: AlertSeverityEnum = Field(
        ..., description="Alert severity level"
    )
    status: AlertStatusEnum = Field(
        ..., description="Alert lifecycle status"
    )
    country_code: Optional[str] = Field(
        None, description="Country code"
    )
    confidence: Optional[Decimal] = Field(
        None, description="Detection confidence"
    )
    created_at: datetime = Field(
        ..., description="Alert creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        None, description="Last update timestamp"
    )


class AlertDetailResponse(BaseModel):
    """Detailed response for a single alert."""

    model_config = ConfigDict(populate_by_name=True)

    alert: AlertEntry = Field(..., description="Alert details")
    detection: Optional[DetectionEntry] = Field(
        None, description="Associated satellite detection"
    )
    severity_score: Optional[Dict[str, Decimal]] = Field(
        None, description="Severity scoring breakdown"
    )
    affected_plot_ids: List[str] = Field(
        default_factory=list, description="Affected supply chain plots"
    )
    commodities: List[EUDRCommodityEnum] = Field(
        default_factory=list, description="Affected commodities"
    )
    workflow_history: Optional[List[Dict[str, Any]]] = Field(
        None, description="Workflow state transition history"
    )
    cutoff_status: Optional[CutoffResultEnum] = Field(
        None, description="EUDR cutoff date verification result"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class AlertListResponse(BaseModel):
    """Paginated list of deforestation alerts."""

    model_config = ConfigDict(populate_by_name=True)

    alerts: List[AlertEntry] = Field(
        default_factory=list, description="List of alerts"
    )
    total_alerts: int = Field(
        default=0, ge=0, description="Total matching alerts"
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class AlertBatchRequest(BaseModel):
    """Request to batch-create alerts from satellite detections."""

    model_config = ConfigDict(populate_by_name=True)

    detection_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Detection IDs to create alerts from (max 1000)",
    )
    auto_classify_severity: bool = Field(
        default=True,
        description="Automatically classify severity for each alert",
    )
    auto_triage: bool = Field(
        default=True, description="Automatically triage generated alerts"
    )


class AlertBatchResponse(BaseModel):
    """Response for batch alert creation."""

    model_config = ConfigDict(populate_by_name=True)

    alerts_created: int = Field(
        default=0, ge=0, description="Number of alerts created"
    )
    alerts_deduplicated: int = Field(
        default=0, ge=0, description="Number of duplicates filtered"
    )
    alerts_failed: int = Field(
        default=0, ge=0, description="Number of creation failures"
    )
    alert_ids: List[str] = Field(
        default_factory=list, description="Created alert IDs"
    )
    errors: Optional[List[Dict[str, str]]] = Field(
        None, description="Per-detection error details"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class AlertSummaryByCategory(BaseModel):
    """Alert count by a category dimension."""

    category: str = Field(..., description="Category name")
    count: int = Field(default=0, ge=0, description="Alert count")
    percentage: Optional[Decimal] = Field(
        None, description="Percentage of total"
    )


class AlertSummaryResponse(BaseModel):
    """Alert summary statistics grouped by country/severity."""

    model_config = ConfigDict(populate_by_name=True)

    total_alerts: int = Field(
        default=0, ge=0, description="Total alert count"
    )
    by_severity: List[AlertSummaryByCategory] = Field(
        default_factory=list, description="Alerts by severity"
    )
    by_country: List[AlertSummaryByCategory] = Field(
        default_factory=list, description="Alerts by country"
    )
    by_status: List[AlertSummaryByCategory] = Field(
        default_factory=list, description="Alerts by status"
    )
    by_commodity: List[AlertSummaryByCategory] = Field(
        default_factory=list, description="Alerts by commodity"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class AlertStatisticsResponse(BaseModel):
    """Detailed alert statistics with trends."""

    model_config = ConfigDict(populate_by_name=True)

    total_alerts: int = Field(default=0, ge=0, description="Total alerts")
    active_alerts: int = Field(
        default=0, ge=0, description="Currently active alerts"
    )
    resolved_alerts: int = Field(
        default=0, ge=0, description="Resolved alerts"
    )
    false_positives: int = Field(
        default=0, ge=0, description="False positive count"
    )
    false_positive_rate: Optional[Decimal] = Field(
        None, description="False positive rate (0-1)"
    )
    average_resolution_hours: Optional[Decimal] = Field(
        None, description="Average time to resolution in hours"
    )
    total_affected_area_ha: Decimal = Field(
        default=Decimal("0"), description="Total affected area (hectares)"
    )
    alerts_last_24h: int = Field(
        default=0, ge=0, description="Alerts in last 24 hours"
    )
    alerts_last_7d: int = Field(
        default=0, ge=0, description="Alerts in last 7 days"
    )
    alerts_last_30d: int = Field(
        default=0, ge=0, description="Alerts in last 30 days"
    )
    sla_compliance_rate: Optional[Decimal] = Field(
        None, description="SLA compliance rate (0-1)"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 3. Severity Schemas
# =============================================================================


class SeverityClassifyRequest(BaseModel):
    """Request to classify the severity of an alert."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert to classify")
    area_ha: Decimal = Field(
        ..., gt=Decimal("0"), description="Deforestation area in hectares"
    )
    deforestation_rate_ha_per_day: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Deforestation rate in hectares per day",
    )
    proximity_km: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Distance to nearest supply chain plot in km",
    )
    in_protected_area: bool = Field(
        default=False, description="Whether within a protected area"
    )
    is_post_cutoff: bool = Field(
        default=False,
        description="Whether event is after EUDR cutoff (2020-12-31)",
    )
    confidence: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Detection confidence",
    )
    custom_weights: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Custom severity weights override (area, rate, proximity, protected, timing)",
    )


class SeverityScoreBreakdown(BaseModel):
    """Detailed severity score component breakdown."""

    area_score: Decimal = Field(
        ..., description="Area component score (0-1)"
    )
    area_weight: Decimal = Field(
        ..., description="Area component weight"
    )
    rate_score: Decimal = Field(
        ..., description="Deforestation rate score (0-1)"
    )
    rate_weight: Decimal = Field(
        ..., description="Rate component weight"
    )
    proximity_score: Decimal = Field(
        ..., description="Proximity score (0-1)"
    )
    proximity_weight: Decimal = Field(
        ..., description="Proximity component weight"
    )
    protected_score: Decimal = Field(
        ..., description="Protected area score (0-1)"
    )
    protected_weight: Decimal = Field(
        ..., description="Protected area weight"
    )
    timing_score: Decimal = Field(
        ..., description="Post-cutoff timing score (0-1)"
    )
    timing_weight: Decimal = Field(
        ..., description="Timing component weight"
    )
    weighted_total: Decimal = Field(
        ..., description="Weighted total score (0-1)"
    )
    multiplier_applied: Optional[Decimal] = Field(
        None, description="Score multiplier (protected/post-cutoff)"
    )
    final_score: Decimal = Field(
        ..., description="Final severity score after multipliers"
    )


class SeverityClassifyResponse(BaseModel):
    """Response for severity classification."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Classified alert ID")
    severity: AlertSeverityEnum = Field(
        ..., description="Assigned severity level"
    )
    score: Decimal = Field(
        ..., description="Final severity score (0-1)"
    )
    breakdown: SeverityScoreBreakdown = Field(
        ..., description="Detailed score breakdown"
    )
    previous_severity: Optional[AlertSeverityEnum] = Field(
        None, description="Previous severity if reclassified"
    )
    classification_reason: str = Field(
        default="", description="Human-readable classification rationale"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class SeverityReclassifyRequest(BaseModel):
    """Request to reclassify an existing alert's severity."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert to reclassify")
    reason: str = Field(
        ..., max_length=2000, description="Reclassification reason"
    )
    new_area_ha: Optional[Decimal] = Field(
        None, gt=Decimal("0"), description="Updated area"
    )
    new_proximity_km: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Updated proximity"
    )
    force_severity: Optional[AlertSeverityEnum] = Field(
        None, description="Force a specific severity level"
    )


class SeverityReclassifyResponse(BaseModel):
    """Response for severity reclassification."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Reclassified alert ID")
    previous_severity: AlertSeverityEnum = Field(
        ..., description="Previous severity level"
    )
    new_severity: AlertSeverityEnum = Field(
        ..., description="New severity level"
    )
    score: Decimal = Field(..., description="New severity score")
    reason: str = Field(..., description="Reclassification reason")
    reclassified_by: str = Field(
        default="system", description="User who reclassified"
    )
    reclassified_at: datetime = Field(
        default_factory=_utcnow, description="Reclassification timestamp"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class SeverityThresholdEntry(BaseModel):
    """A single severity threshold configuration."""

    severity: AlertSeverityEnum = Field(
        ..., description="Severity level"
    )
    area_threshold_ha: Optional[Decimal] = Field(
        None, description="Area threshold in hectares"
    )
    proximity_threshold_km: Optional[Decimal] = Field(
        None, description="Proximity threshold in km"
    )
    score_range_min: Decimal = Field(
        ..., description="Minimum score for this severity"
    )
    score_range_max: Decimal = Field(
        ..., description="Maximum score for this severity"
    )


class SeverityThresholdsResponse(BaseModel):
    """Current severity classification threshold configuration."""

    model_config = ConfigDict(populate_by_name=True)

    thresholds: List[SeverityThresholdEntry] = Field(
        ..., description="Severity threshold configuration"
    )
    weights: Dict[str, Decimal] = Field(
        ..., description="Current severity weights"
    )
    multipliers: Dict[str, Decimal] = Field(
        ..., description="Score multipliers (protected_area, post_cutoff)"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class SeverityDistributionEntry(BaseModel):
    """Distribution entry for a severity level."""

    severity: AlertSeverityEnum = Field(
        ..., description="Severity level"
    )
    count: int = Field(default=0, ge=0, description="Alert count")
    percentage: Decimal = Field(
        default=Decimal("0"), description="Percentage of total"
    )
    average_area_ha: Optional[Decimal] = Field(
        None, description="Average area for this severity"
    )
    average_score: Optional[Decimal] = Field(
        None, description="Average severity score"
    )


class SeverityDistributionResponse(BaseModel):
    """Distribution of alerts across severity levels."""

    model_config = ConfigDict(populate_by_name=True)

    distribution: List[SeverityDistributionEntry] = Field(
        default_factory=list, description="Severity distribution"
    )
    total_alerts: int = Field(
        default=0, ge=0, description="Total alert count"
    )
    average_severity_score: Optional[Decimal] = Field(
        None, description="Overall average severity score"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 4. Buffer Schemas
# =============================================================================


class BufferCreateRequest(BaseModel):
    """Request to create a spatial buffer zone."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Supply chain plot ID")
    center: GeoPointSchema = Field(
        ..., description="Buffer center point"
    )
    radius_km: Decimal = Field(
        ...,
        gt=Decimal("0"),
        le=Decimal("50"),
        description="Buffer radius in kilometers (1-50 km)",
    )
    buffer_type: BufferTypeEnum = Field(
        default=BufferTypeEnum.CIRCULAR,
        description="Buffer geometry type",
    )
    polygon: Optional[GeoPolygonSchema] = Field(
        None, description="Custom polygon for polygon-type buffers"
    )
    resolution_points: int = Field(
        default=64, ge=4, le=256, description="Buffer geometry resolution"
    )
    name: Optional[str] = Field(
        None, max_length=255, description="Buffer zone name"
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities produced in the plot"
    )


class BufferZoneEntry(BaseModel):
    """A spatial buffer zone record."""

    buffer_id: str = Field(..., description="Unique buffer zone identifier")
    plot_id: str = Field(..., description="Associated plot ID")
    center_latitude: Decimal = Field(
        ..., description="Buffer center latitude"
    )
    center_longitude: Decimal = Field(
        ..., description="Buffer center longitude"
    )
    radius_km: Decimal = Field(..., description="Buffer radius in km")
    buffer_type: BufferTypeEnum = Field(
        ..., description="Buffer geometry type"
    )
    area_km2: Optional[Decimal] = Field(
        None, description="Buffer area in square kilometers"
    )
    active: bool = Field(default=True, description="Whether buffer is active")
    name: Optional[str] = Field(None, description="Buffer zone name")
    commodities: List[EUDRCommodityEnum] = Field(
        default_factory=list, description="Commodities in plot"
    )
    violation_count: int = Field(
        default=0, ge=0, description="Total violations detected"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )


class BufferCreateResponse(BaseModel):
    """Response for buffer zone creation."""

    model_config = ConfigDict(populate_by_name=True)

    buffer: BufferZoneEntry = Field(
        ..., description="Created buffer zone"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class BufferUpdateRequest(BaseModel):
    """Request to update an existing buffer zone."""

    model_config = ConfigDict(populate_by_name=True)

    radius_km: Optional[Decimal] = Field(
        None,
        gt=Decimal("0"),
        le=Decimal("50"),
        description="Updated radius",
    )
    active: Optional[bool] = Field(
        None, description="Activate/deactivate buffer"
    )
    name: Optional[str] = Field(
        None, max_length=255, description="Updated name"
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Updated commodities"
    )
    resolution_points: Optional[int] = Field(
        None, ge=4, le=256, description="Updated resolution"
    )


class BufferCheckRequest(BaseModel):
    """Request to check if a point falls within any active buffer zones."""

    model_config = ConfigDict(populate_by_name=True)

    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Point latitude to check",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Point longitude to check",
    )
    detection_id: Optional[str] = Field(
        None, description="Associated detection ID"
    )
    include_distance: bool = Field(
        default=True, description="Include distance to buffer center"
    )


class BufferCheckResult(BaseModel):
    """Result of checking a point against a single buffer zone."""

    buffer_id: str = Field(..., description="Buffer zone ID")
    plot_id: str = Field(..., description="Associated plot ID")
    is_within_buffer: bool = Field(
        ..., description="Whether point is within buffer"
    )
    distance_km: Optional[Decimal] = Field(
        None, description="Distance to buffer center in km"
    )
    buffer_radius_km: Decimal = Field(
        ..., description="Buffer radius in km"
    )
    commodities: List[EUDRCommodityEnum] = Field(
        default_factory=list, description="Commodities at risk"
    )


class BufferCheckResponse(BaseModel):
    """Response for spatial buffer check."""

    model_config = ConfigDict(populate_by_name=True)

    point_latitude: Decimal = Field(..., description="Checked latitude")
    point_longitude: Decimal = Field(..., description="Checked longitude")
    is_within_any_buffer: bool = Field(
        ..., description="Whether point is within any buffer"
    )
    buffers_affected: List[BufferCheckResult] = Field(
        default_factory=list, description="Affected buffer zones"
    )
    total_buffers_checked: int = Field(
        default=0, ge=0, description="Total buffers checked"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class BufferViolationEntry(BaseModel):
    """A buffer zone violation record."""

    violation_id: str = Field(
        default_factory=_new_id, description="Violation identifier"
    )
    buffer_id: str = Field(..., description="Violated buffer zone ID")
    plot_id: str = Field(..., description="Associated plot ID")
    alert_id: Optional[str] = Field(
        None, description="Associated alert ID"
    )
    detection_id: Optional[str] = Field(
        None, description="Associated detection ID"
    )
    latitude: Decimal = Field(..., description="Violation latitude")
    longitude: Decimal = Field(..., description="Violation longitude")
    distance_km: Decimal = Field(
        ..., description="Distance to buffer center"
    )
    area_ha: Optional[Decimal] = Field(
        None, description="Affected area in hectares"
    )
    severity: Optional[AlertSeverityEnum] = Field(
        None, description="Violation severity"
    )
    detected_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp"
    )


class BufferViolationsResponse(BaseModel):
    """List of buffer zone violations."""

    model_config = ConfigDict(populate_by_name=True)

    violations: List[BufferViolationEntry] = Field(
        default_factory=list, description="Buffer violations"
    )
    total_violations: int = Field(
        default=0, ge=0, description="Total violations"
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class BufferZonesResponse(BaseModel):
    """List of active buffer zones."""

    model_config = ConfigDict(populate_by_name=True)

    zones: List[BufferZoneEntry] = Field(
        default_factory=list, description="Buffer zones"
    )
    total_zones: int = Field(
        default=0, ge=0, description="Total zones"
    )
    active_zones: int = Field(
        default=0, ge=0, description="Active zones count"
    )
    total_area_km2: Optional[Decimal] = Field(
        None, description="Total monitored area in km2"
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 5. Cutoff Schemas
# =============================================================================


class CutoffVerifyRequest(BaseModel):
    """Request to verify a detection against the EUDR cutoff date."""

    model_config = ConfigDict(populate_by_name=True)

    detection_id: str = Field(
        ..., description="Detection to verify against cutoff"
    )
    detection_date: date = Field(
        ..., description="Date of the detected change"
    )
    latitude: Decimal = Field(
        ...,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Detection latitude",
    )
    longitude: Decimal = Field(
        ...,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Detection longitude",
    )
    sources: Optional[List[SatelliteSourceEnum]] = Field(
        None, description="Evidence sources to query"
    )
    confidence_threshold: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Override confidence threshold",
    )


class TemporalEvidenceEntry(BaseModel):
    """A single piece of temporal evidence for cutoff verification."""

    source: SatelliteSourceEnum = Field(
        ..., description="Evidence source"
    )
    observation_date: date = Field(
        ..., description="Date of observation"
    )
    forest_cover_pct: Optional[Decimal] = Field(
        None, description="Forest cover percentage at observation"
    )
    ndvi_value: Optional[Decimal] = Field(
        None, description="NDVI value at observation"
    )
    quality: EvidenceQualityEnum = Field(
        default=EvidenceQualityEnum.MEDIUM,
        description="Evidence quality classification",
    )
    confidence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Evidence confidence",
    )


class CutoffVerifyResponse(BaseModel):
    """Response for EUDR cutoff date verification."""

    model_config = ConfigDict(populate_by_name=True)

    detection_id: str = Field(..., description="Verified detection ID")
    cutoff_result: CutoffResultEnum = Field(
        ..., description="Cutoff verification result"
    )
    cutoff_date: date = Field(
        default=date(2020, 12, 31),
        description="EUDR cutoff date (Art. 2(1))",
    )
    detection_date: date = Field(
        ..., description="Detection date"
    )
    days_from_cutoff: int = Field(
        ..., description="Days relative to cutoff (negative = before)"
    )
    confidence: Decimal = Field(
        ..., description="Verification confidence (0-1)"
    )
    evidence: List[TemporalEvidenceEntry] = Field(
        default_factory=list, description="Temporal evidence used"
    )
    evidence_count: int = Field(
        default=0, ge=0, description="Total evidence sources"
    )
    is_compliant: bool = Field(
        ...,
        description="Whether detection is EUDR-compliant (pre-cutoff or inconclusive)",
    )
    regulatory_reference: str = Field(
        default="EU 2023/1115 Article 2(1)",
        description="Applicable regulatory reference",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class CutoffBatchVerifyRequest(BaseModel):
    """Request to batch-verify multiple detections against cutoff."""

    model_config = ConfigDict(populate_by_name=True)

    verifications: List[CutoffVerifyRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Detections to verify (max 500)",
    )


class CutoffBatchResultEntry(BaseModel):
    """A single result in batch cutoff verification."""

    detection_id: str = Field(..., description="Detection ID")
    cutoff_result: CutoffResultEnum = Field(
        ..., description="Verification result"
    )
    confidence: Decimal = Field(..., description="Confidence")
    is_compliant: bool = Field(
        ..., description="Cutoff compliance status"
    )
    error: Optional[str] = Field(
        None, description="Error message if verification failed"
    )


class CutoffBatchVerifyResponse(BaseModel):
    """Response for batch cutoff verification."""

    model_config = ConfigDict(populate_by_name=True)

    results: List[CutoffBatchResultEntry] = Field(
        default_factory=list, description="Verification results"
    )
    total_verified: int = Field(
        default=0, ge=0, description="Total verified"
    )
    pre_cutoff_count: int = Field(
        default=0, ge=0, description="Pre-cutoff count"
    )
    post_cutoff_count: int = Field(
        default=0, ge=0, description="Post-cutoff count"
    )
    inconclusive_count: int = Field(
        default=0, ge=0, description="Inconclusive count"
    )
    failed_count: int = Field(
        default=0, ge=0, description="Verification failures"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class CutoffEvidenceResponse(BaseModel):
    """Temporal evidence details for a detection."""

    model_config = ConfigDict(populate_by_name=True)

    detection_id: str = Field(..., description="Detection ID")
    evidence: List[TemporalEvidenceEntry] = Field(
        default_factory=list, description="Temporal evidence records"
    )
    total_evidence: int = Field(
        default=0, ge=0, description="Total evidence records"
    )
    earliest_observation: Optional[date] = Field(
        None, description="Earliest observation date"
    )
    latest_observation: Optional[date] = Field(
        None, description="Latest observation date"
    )
    average_confidence: Optional[Decimal] = Field(
        None, description="Average evidence confidence"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ForestStateEntry(BaseModel):
    """Forest state at a point in time for timeline construction."""

    observation_date: date = Field(
        ..., description="Observation date"
    )
    forest_cover_pct: Decimal = Field(
        ..., description="Forest cover percentage"
    )
    canopy_density: Optional[Decimal] = Field(
        None, description="Canopy density metric"
    )
    change_from_prior: Optional[Decimal] = Field(
        None, description="Change from prior observation"
    )
    source: SatelliteSourceEnum = Field(
        ..., description="Observation source"
    )
    is_cutoff_period: bool = Field(
        default=False,
        description="Whether this observation is near cutoff date",
    )


class CutoffTimelineResponse(BaseModel):
    """Forest state timeline for a detection location."""

    model_config = ConfigDict(populate_by_name=True)

    detection_id: str = Field(..., description="Detection ID")
    timeline: List[ForestStateEntry] = Field(
        default_factory=list, description="Forest state timeline"
    )
    cutoff_date: date = Field(
        default=date(2020, 12, 31), description="EUDR cutoff date"
    )
    forest_cover_at_cutoff: Optional[Decimal] = Field(
        None, description="Estimated forest cover at cutoff date"
    )
    current_forest_cover: Optional[Decimal] = Field(
        None, description="Current forest cover"
    )
    total_change_pct: Optional[Decimal] = Field(
        None, description="Total change in forest cover since cutoff"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 6. Baseline Schemas
# =============================================================================


class BaselineEstablishRequest(BaseModel):
    """Request to establish a historical baseline for an area."""

    model_config = ConfigDict(populate_by_name=True)

    plot_id: str = Field(..., description="Plot to establish baseline for")
    center: GeoPointSchema = Field(
        ..., description="Baseline area center point"
    )
    radius_km: Optional[Decimal] = Field(
        None,
        gt=Decimal("0"),
        le=Decimal("50"),
        description="Baseline area radius",
    )
    polygon: Optional[GeoPolygonSchema] = Field(
        None, description="Custom polygon for baseline area"
    )
    reference_start_year: int = Field(
        default=2018, ge=2000, le=2025, description="Reference period start"
    )
    reference_end_year: int = Field(
        default=2020, ge=2000, le=2025, description="Reference period end"
    )
    sources: Optional[List[SatelliteSourceEnum]] = Field(
        None, description="Sources for baseline data"
    )
    min_samples: int = Field(
        default=3, ge=1, description="Minimum observation samples required"
    )


class BaselineDataEntry(BaseModel):
    """A baseline observation data point."""

    observation_date: date = Field(
        ..., description="Observation date"
    )
    canopy_cover_pct: Decimal = Field(
        ..., description="Canopy cover percentage"
    )
    forest_area_ha: Optional[Decimal] = Field(
        None, description="Forest area in hectares"
    )
    ndvi_mean: Optional[Decimal] = Field(
        None, description="Mean NDVI value"
    )
    source: SatelliteSourceEnum = Field(
        ..., description="Data source"
    )


class BaselineEstablishResponse(BaseModel):
    """Response for baseline establishment."""

    model_config = ConfigDict(populate_by_name=True)

    baseline_id: str = Field(
        default_factory=_new_id, description="Baseline identifier"
    )
    plot_id: str = Field(..., description="Associated plot ID")
    reference_period: str = Field(
        ..., description="Reference period (e.g. 2018-2020)"
    )
    sample_count: int = Field(
        ..., ge=0, description="Number of baseline samples"
    )
    average_canopy_cover_pct: Decimal = Field(
        ..., description="Average canopy cover during reference"
    )
    average_forest_area_ha: Optional[Decimal] = Field(
        None, description="Average forest area during reference"
    )
    baseline_data: List[BaselineDataEntry] = Field(
        default_factory=list, description="Baseline observation data"
    )
    canopy_cover_threshold_pct: Decimal = Field(
        default=Decimal("10"),
        description="Canopy cover threshold for forest classification",
    )
    is_forested: bool = Field(
        ..., description="Whether area was classified as forest"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class BaselineCompareRequest(BaseModel):
    """Request to compare current state against a baseline."""

    model_config = ConfigDict(populate_by_name=True)

    baseline_id: str = Field(
        ..., description="Baseline to compare against"
    )
    current_observation_date: Optional[date] = Field(
        None, description="Date of current observation (default: latest)"
    )
    include_timeline: bool = Field(
        default=False, description="Include full timeline comparison"
    )


class BaselineCompareResponse(BaseModel):
    """Response for baseline comparison."""

    model_config = ConfigDict(populate_by_name=True)

    baseline_id: str = Field(..., description="Baseline ID")
    plot_id: str = Field(..., description="Associated plot ID")
    baseline_canopy_cover_pct: Decimal = Field(
        ..., description="Baseline average canopy cover"
    )
    current_canopy_cover_pct: Decimal = Field(
        ..., description="Current canopy cover"
    )
    change_pct: Decimal = Field(
        ..., description="Change in canopy cover (negative = loss)"
    )
    change_area_ha: Optional[Decimal] = Field(
        None, description="Estimated area change in hectares"
    )
    deforestation_detected: bool = Field(
        ..., description="Whether significant deforestation detected"
    )
    change_significance: str = Field(
        default="not_significant",
        description="Change significance (significant, moderate, not_significant)",
    )
    comparison_date: date = Field(
        ..., description="Date of comparison observation"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class BaselineUpdateRequest(BaseModel):
    """Request to update an existing baseline."""

    model_config = ConfigDict(populate_by_name=True)

    add_samples: Optional[List[BaselineDataEntry]] = Field(
        None, description="Additional baseline samples to include"
    )
    reference_start_year: Optional[int] = Field(
        None, ge=2000, le=2025, description="Updated start year"
    )
    reference_end_year: Optional[int] = Field(
        None, ge=2000, le=2025, description="Updated end year"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Update notes"
    )


class BaselineCoverageEntry(BaseModel):
    """Coverage statistics for a single baseline."""

    baseline_id: str = Field(..., description="Baseline ID")
    plot_id: str = Field(..., description="Plot ID")
    reference_period: str = Field(
        ..., description="Reference period"
    )
    sample_count: int = Field(..., description="Sample count")
    is_adequate: bool = Field(
        ..., description="Whether coverage meets minimum requirements"
    )
    average_canopy_cover_pct: Decimal = Field(
        ..., description="Average canopy cover"
    )
    last_updated: datetime = Field(
        ..., description="Last update timestamp"
    )


class BaselineCoverageResponse(BaseModel):
    """Response for baseline coverage statistics."""

    model_config = ConfigDict(populate_by_name=True)

    baselines: List[BaselineCoverageEntry] = Field(
        default_factory=list, description="Baseline coverage entries"
    )
    total_baselines: int = Field(
        default=0, ge=0, description="Total baselines"
    )
    adequate_coverage_count: int = Field(
        default=0, ge=0, description="Baselines with adequate coverage"
    )
    inadequate_coverage_count: int = Field(
        default=0, ge=0, description="Baselines needing more samples"
    )
    coverage_rate: Optional[Decimal] = Field(
        None, description="Percentage with adequate coverage"
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 7. Workflow Schemas
# =============================================================================


class WorkflowTriageRequest(BaseModel):
    """Request to triage a deforestation alert."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert to triage")
    priority: WorkflowPriorityEnum = Field(
        default=WorkflowPriorityEnum.MEDIUM,
        description="Triage priority assignment",
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Triage notes"
    )
    auto_assign: bool = Field(
        default=False, description="Automatically assign after triage"
    )
    assignee_id: Optional[str] = Field(
        None, description="Assignee for auto-assignment"
    )


class WorkflowTriageResponse(BaseModel):
    """Response for alert triage operation."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Triaged alert ID")
    previous_status: AlertStatusEnum = Field(
        ..., description="Previous alert status"
    )
    new_status: AlertStatusEnum = Field(
        ..., description="New alert status (triaged)"
    )
    priority: WorkflowPriorityEnum = Field(
        ..., description="Assigned priority"
    )
    triaged_by: str = Field(
        default="system", description="User who performed triage"
    )
    triaged_at: datetime = Field(
        default_factory=_utcnow, description="Triage timestamp"
    )
    sla_deadline: Optional[datetime] = Field(
        None, description="SLA deadline for next action"
    )
    assigned_to: Optional[str] = Field(
        None, description="Auto-assigned investigator"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class WorkflowAssignRequest(BaseModel):
    """Request to assign an alert to an investigator."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert to assign")
    assignee_id: str = Field(
        ..., description="Investigator user ID to assign to"
    )
    priority: Optional[WorkflowPriorityEnum] = Field(
        None, description="Override priority"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Assignment notes"
    )
    due_date: Optional[datetime] = Field(
        None, description="Custom due date for investigation"
    )


class WorkflowInvestigateRequest(BaseModel):
    """Request to start investigation on an alert."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(
        ..., description="Alert to start investigating"
    )
    investigation_type: Optional[str] = Field(
        None,
        description="Type of investigation (field_visit, remote_sensing, document_review)",
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Investigation notes"
    )
    evidence_urls: Optional[List[str]] = Field(
        None, description="URLs to supporting evidence"
    )


class WorkflowResolveRequest(BaseModel):
    """Request to resolve a deforestation alert."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert to resolve")
    resolution: str = Field(
        ...,
        description="Resolution type (confirmed, false_positive, remediated, inconclusive)",
    )
    root_cause: Optional[str] = Field(
        None, max_length=2000, description="Root cause analysis"
    )
    findings: Optional[str] = Field(
        None, max_length=5000, description="Investigation findings"
    )
    evidence_urls: Optional[List[str]] = Field(
        None, description="Supporting evidence URLs"
    )
    remediation_actions: Optional[List[RemediationActionEnum]] = Field(
        None, description="Remediation actions taken"
    )
    is_false_positive: bool = Field(
        default=False, description="Mark as false positive"
    )


class WorkflowEscalateRequest(BaseModel):
    """Request to escalate a deforestation alert."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert to escalate")
    escalation_level: int = Field(
        ..., ge=1, le=3, description="Escalation level (1-3)"
    )
    reason: str = Field(
        ..., max_length=2000, description="Escalation reason"
    )
    escalate_to: Optional[str] = Field(
        None, description="User/team to escalate to"
    )
    requires_external_review: bool = Field(
        default=False,
        description="Whether external review (e.g. competent authority) needed",
    )


class WorkflowTransitionResponse(BaseModel):
    """Generic response for workflow state transitions."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert ID")
    action: WorkflowActionEnum = Field(
        ..., description="Action performed"
    )
    previous_status: AlertStatusEnum = Field(
        ..., description="Previous status"
    )
    new_status: AlertStatusEnum = Field(
        ..., description="New status"
    )
    performed_by: str = Field(
        default="system", description="User who performed action"
    )
    performed_at: datetime = Field(
        default_factory=_utcnow, description="Action timestamp"
    )
    sla_deadline: Optional[datetime] = Field(
        None, description="SLA deadline for next action"
    )
    notes: Optional[str] = Field(None, description="Action notes")
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class SLAEntry(BaseModel):
    """SLA status for a single alert."""

    alert_id: str = Field(..., description="Alert ID")
    current_status: AlertStatusEnum = Field(
        ..., description="Current alert status"
    )
    sla_stage: str = Field(
        ..., description="Current SLA stage (triage, investigation, resolution)"
    )
    deadline: datetime = Field(..., description="SLA deadline")
    sla_status: SLAStatusEnum = Field(
        ..., description="SLA compliance status"
    )
    hours_remaining: Optional[Decimal] = Field(
        None, description="Hours remaining until deadline"
    )
    hours_elapsed: Optional[Decimal] = Field(
        None, description="Hours elapsed since stage start"
    )
    escalation_level: int = Field(
        default=0, ge=0, description="Current escalation level"
    )


class WorkflowSLAResponse(BaseModel):
    """Response for SLA status query."""

    model_config = ConfigDict(populate_by_name=True)

    sla_entries: List[SLAEntry] = Field(
        default_factory=list, description="SLA status entries"
    )
    total_tracked: int = Field(
        default=0, ge=0, description="Total alerts tracked"
    )
    on_track_count: int = Field(
        default=0, ge=0, description="Alerts on track"
    )
    at_risk_count: int = Field(
        default=0, ge=0, description="Alerts at risk of breach"
    )
    breached_count: int = Field(
        default=0, ge=0, description="Alerts with SLA breach"
    )
    sla_compliance_rate: Optional[Decimal] = Field(
        None, description="Overall SLA compliance rate"
    )
    sla_config: Dict[str, int] = Field(
        default_factory=lambda: {
            "triage_hours": 4,
            "investigation_hours": 48,
            "resolution_hours": 168,
        },
        description="SLA configuration (hours per stage)",
    )
    pagination: PaginatedMeta = Field(
        ..., description="Pagination information"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 8. Compliance Schemas
# =============================================================================


class ComplianceAssessRequest(BaseModel):
    """Request to assess compliance impact of a deforestation alert."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(
        ..., description="Alert to assess compliance impact for"
    )
    include_affected_products: bool = Field(
        default=True, description="Include affected product analysis"
    )
    include_financial_impact: bool = Field(
        default=True, description="Include financial impact estimation"
    )
    include_recommendations: bool = Field(
        default=True, description="Include remediation recommendations"
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Specific commodities to assess"
    )
    operator_id: Optional[str] = Field(
        None, description="Operator ID for impact scoping"
    )


class AffectedSupplierEntry(BaseModel):
    """An affected supplier in compliance assessment."""

    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: Optional[str] = Field(
        None, description="Supplier name"
    )
    country_code: Optional[str] = Field(
        None, description="Supplier country"
    )
    risk_level: str = Field(
        default="medium", description="Supplier risk level"
    )
    commodities: List[EUDRCommodityEnum] = Field(
        default_factory=list, description="Supplier commodities"
    )
    estimated_volume_affected: Optional[Decimal] = Field(
        None, description="Estimated volume affected (tonnes)"
    )


class AffectedProductEntry(BaseModel):
    """An affected product in compliance assessment."""

    product_id: str = Field(..., description="Product identifier")
    product_name: Optional[str] = Field(
        None, description="Product name"
    )
    commodity: EUDRCommodityEnum = Field(
        ..., description="EUDR commodity type"
    )
    cn_code: Optional[str] = Field(
        None, description="EU Combined Nomenclature code"
    )
    estimated_value_eur: Optional[Decimal] = Field(
        None, description="Estimated value in EUR"
    )
    market_restriction_risk: str = Field(
        default="low", description="Market restriction risk"
    )


class ComplianceRecommendationEntry(BaseModel):
    """A compliance remediation recommendation."""

    recommendation_id: str = Field(
        default_factory=_new_id, description="Recommendation ID"
    )
    action: RemediationActionEnum = Field(
        ..., description="Recommended remediation action"
    )
    priority: WorkflowPriorityEnum = Field(
        default=WorkflowPriorityEnum.MEDIUM,
        description="Recommendation priority",
    )
    description: str = Field(
        ..., max_length=2000, description="Detailed recommendation"
    )
    estimated_timeline_days: Optional[int] = Field(
        None, description="Estimated implementation timeline"
    )
    regulatory_reference: Optional[str] = Field(
        None, description="Applicable EUDR article reference"
    )


class ComplianceAssessResponse(BaseModel):
    """Response for compliance impact assessment."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Assessed alert ID")
    compliance_outcome: ComplianceOutcomeEnum = Field(
        ..., description="Compliance assessment outcome"
    )
    severity: AlertSeverityEnum = Field(
        ..., description="Alert severity level"
    )
    is_post_cutoff: bool = Field(
        ..., description="Whether event is post-cutoff"
    )
    affected_suppliers: List[AffectedSupplierEntry] = Field(
        default_factory=list, description="Affected suppliers"
    )
    affected_products: List[AffectedProductEntry] = Field(
        default_factory=list, description="Affected products"
    )
    total_affected_suppliers: int = Field(
        default=0, ge=0, description="Total affected suppliers"
    )
    total_affected_products: int = Field(
        default=0, ge=0, description="Total affected products"
    )
    estimated_financial_impact_eur: Optional[Decimal] = Field(
        None, description="Estimated financial impact in EUR"
    )
    market_restriction_triggered: bool = Field(
        default=False, description="Whether market restrictions triggered"
    )
    recommendations: List[ComplianceRecommendationEntry] = Field(
        default_factory=list, description="Remediation recommendations"
    )
    regulatory_articles: List[str] = Field(
        default_factory=lambda: ["Art. 2", "Art. 9", "Art. 10"],
        description="Applicable regulatory articles",
    )
    assessment_rationale: str = Field(
        default="", description="Compliance assessment rationale"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class AffectedProductsResponse(BaseModel):
    """Response for affected products query."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert ID")
    affected_products: List[AffectedProductEntry] = Field(
        default_factory=list, description="Affected products"
    )
    affected_suppliers: List[AffectedSupplierEntry] = Field(
        default_factory=list, description="Affected suppliers"
    )
    total_products: int = Field(
        default=0, ge=0, description="Total affected products"
    )
    total_suppliers: int = Field(
        default=0, ge=0, description="Total affected suppliers"
    )
    total_estimated_value_eur: Optional[Decimal] = Field(
        None, description="Total estimated value at risk"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ComplianceRecommendationsResponse(BaseModel):
    """Response for compliance recommendations query."""

    model_config = ConfigDict(populate_by_name=True)

    recommendations: List[ComplianceRecommendationEntry] = Field(
        default_factory=list, description="Compliance recommendations"
    )
    total_recommendations: int = Field(
        default=0, ge=0, description="Total recommendations"
    )
    urgent_count: int = Field(
        default=0, ge=0, description="Urgent recommendations"
    )
    estimated_total_timeline_days: Optional[int] = Field(
        None, description="Total estimated implementation timeline"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class RemediationPlanRequest(BaseModel):
    """Request to create a remediation plan for a compliance incident."""

    model_config = ConfigDict(populate_by_name=True)

    alert_id: str = Field(..., description="Alert to remediate")
    actions: List[RemediationActionEnum] = Field(
        ...,
        min_length=1,
        description="Planned remediation actions",
    )
    target_completion_date: date = Field(
        ..., description="Target completion date"
    )
    responsible_party: str = Field(
        ..., description="Party responsible for remediation"
    )
    description: str = Field(
        ..., max_length=5000, description="Remediation plan description"
    )
    affected_supplier_ids: Optional[List[str]] = Field(
        None, description="Affected supplier IDs"
    )
    affected_product_ids: Optional[List[str]] = Field(
        None, description="Affected product IDs"
    )
    estimated_cost_eur: Optional[Decimal] = Field(
        None, description="Estimated remediation cost (EUR)"
    )


class RemediationPlanResponse(BaseModel):
    """Response for remediation plan creation."""

    model_config = ConfigDict(populate_by_name=True)

    plan_id: str = Field(
        default_factory=_new_id, description="Remediation plan ID"
    )
    alert_id: str = Field(..., description="Associated alert ID")
    status: str = Field(
        default="active", description="Plan status"
    )
    actions: List[RemediationActionEnum] = Field(
        ..., description="Planned actions"
    )
    target_completion_date: date = Field(
        ..., description="Target completion date"
    )
    responsible_party: str = Field(
        ..., description="Responsible party"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "SatelliteSourceEnum",
    "ChangeTypeEnum",
    "AlertSeverityEnum",
    "AlertStatusEnum",
    "BufferTypeEnum",
    "CutoffResultEnum",
    "ComplianceOutcomeEnum",
    "WorkflowActionEnum",
    "SpectralIndexEnum",
    "EUDRCommodityEnum",
    "EvidenceQualityEnum",
    "RemediationActionEnum",
    "WorkflowPriorityEnum",
    "SLAStatusEnum",
    # Common
    "ProvenanceInfo",
    "MetadataSchema",
    "PaginatedMeta",
    "ErrorResponse",
    "GeoPointSchema",
    "GeoPolygonSchema",
    "HealthResponse",
    # Satellite
    "SatelliteDetectionRequest",
    "DetectionEntry",
    "SatelliteDetectionResponse",
    "SatelliteScanRequest",
    "SatelliteScanResponse",
    "SatelliteSourceInfo",
    "SatelliteSourcesResponse",
    "ImageryMetadata",
    "SatelliteImageryResponse",
    # Alert
    "AlertCreateRequest",
    "AlertEntry",
    "AlertDetailResponse",
    "AlertListResponse",
    "AlertBatchRequest",
    "AlertBatchResponse",
    "AlertSummaryByCategory",
    "AlertSummaryResponse",
    "AlertStatisticsResponse",
    # Severity
    "SeverityClassifyRequest",
    "SeverityScoreBreakdown",
    "SeverityClassifyResponse",
    "SeverityReclassifyRequest",
    "SeverityReclassifyResponse",
    "SeverityThresholdEntry",
    "SeverityThresholdsResponse",
    "SeverityDistributionEntry",
    "SeverityDistributionResponse",
    # Buffer
    "BufferCreateRequest",
    "BufferZoneEntry",
    "BufferCreateResponse",
    "BufferUpdateRequest",
    "BufferCheckRequest",
    "BufferCheckResult",
    "BufferCheckResponse",
    "BufferViolationEntry",
    "BufferViolationsResponse",
    "BufferZonesResponse",
    # Cutoff
    "CutoffVerifyRequest",
    "TemporalEvidenceEntry",
    "CutoffVerifyResponse",
    "CutoffBatchVerifyRequest",
    "CutoffBatchResultEntry",
    "CutoffBatchVerifyResponse",
    "CutoffEvidenceResponse",
    "ForestStateEntry",
    "CutoffTimelineResponse",
    # Baseline
    "BaselineEstablishRequest",
    "BaselineDataEntry",
    "BaselineEstablishResponse",
    "BaselineCompareRequest",
    "BaselineCompareResponse",
    "BaselineUpdateRequest",
    "BaselineCoverageEntry",
    "BaselineCoverageResponse",
    # Workflow
    "WorkflowTriageRequest",
    "WorkflowTriageResponse",
    "WorkflowAssignRequest",
    "WorkflowInvestigateRequest",
    "WorkflowResolveRequest",
    "WorkflowEscalateRequest",
    "WorkflowTransitionResponse",
    "SLAEntry",
    "WorkflowSLAResponse",
    # Compliance
    "ComplianceAssessRequest",
    "AffectedSupplierEntry",
    "AffectedProductEntry",
    "ComplianceRecommendationEntry",
    "ComplianceAssessResponse",
    "AffectedProductsResponse",
    "ComplianceRecommendationsResponse",
    "RemediationPlanRequest",
    "RemediationPlanResponse",
]
