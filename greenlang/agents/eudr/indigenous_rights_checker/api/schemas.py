# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-021 Indigenous Rights Checker

Pydantic v2 request/response models for the REST API layer covering all
7 route domains: territory management, FPIC verification, land rights
overlap analysis, community consultation tracking, rights violation
detection, indigenous community registry, and compliance reporting.

All numeric fields use ``Decimal`` for precision (zero-hallucination).
All date/time fields use UTC-aware ``datetime``.
All geospatial coordinates use Decimal for regulatory-grade precision.

Schema Groups (7 domains + common):
    Common: ProvenanceInfo, MetadataSchema, PaginatedMeta, ErrorResponse,
            GeoPointSchema, GeoPolygonSchema, HealthResponse
    1. Territory: TerritoryCreateRequest, TerritoryUpdateRequest,
       TerritoryResponse, TerritoryListResponse
    2. FPIC: FPICVerifyRequest, FPICVerifyResponse, FPICDocumentResponse,
       FPICDocumentListResponse, FPICScoreRequest, FPICScoreResponse
    3. Overlap: OverlapAnalyzeRequest, OverlapAnalyzeResponse,
       OverlapListResponse, OverlapBulkRequest, OverlapBulkResponse
    4. Consultation: ConsultationCreateRequest, ConsultationResponse,
       ConsultationListResponse
    5. Violation: ViolationDetectRequest, ViolationDetectResponse,
       ViolationResponse, ViolationListResponse, ViolationResolveRequest,
       ViolationResolveResponse
    6. Registry: CommunityRegisterRequest, CommunityResponse,
       CommunityListResponse
    7. Compliance: ComplianceReportResponse, ComplianceAssessRequest,
       ComplianceAssessResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
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


class TerritoryStatusEnum(str, Enum):
    """Territory registration lifecycle status."""

    ACTIVE = "active"
    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    ARCHIVED = "archived"


class TerritoryTypeEnum(str, Enum):
    """Types of indigenous/traditional territories."""

    INDIGENOUS_LAND = "indigenous_land"
    TRADITIONAL_TERRITORY = "traditional_territory"
    COMMUNITY_RESERVE = "community_reserve"
    PROTECTED_AREA = "protected_area"
    ANCESTRAL_DOMAIN = "ancestral_domain"
    COLLECTIVE_TITLE = "collective_title"
    CUSTOMARY_LAND = "customary_land"


class RecognitionLevelEnum(str, Enum):
    """Legal recognition level for a territory."""

    LEGALLY_TITLED = "legally_titled"
    FORMALLY_RECOGNIZED = "formally_recognized"
    PENDING_RECOGNITION = "pending_recognition"
    CUSTOMARY_ONLY = "customary_only"
    DISPUTED = "disputed"
    UNRECOGNIZED = "unrecognized"


class FPICStatusEnum(str, Enum):
    """FPIC (Free, Prior and Informed Consent) verification status."""

    OBTAINED = "obtained"
    PENDING = "pending"
    DENIED = "denied"
    EXPIRED = "expired"
    CONDITIONAL = "conditional"
    NOT_REQUIRED = "not_required"
    INSUFFICIENT = "insufficient"


class FPICDocumentTypeEnum(str, Enum):
    """Types of FPIC documentation."""

    CONSENT_AGREEMENT = "consent_agreement"
    MEETING_MINUTES = "meeting_minutes"
    COMMUNITY_RESOLUTION = "community_resolution"
    SIGNED_MOU = "signed_mou"
    AUDIO_RECORDING = "audio_recording"
    VIDEO_EVIDENCE = "video_evidence"
    WITNESS_STATEMENT = "witness_statement"
    GOVERNMENT_APPROVAL = "government_approval"
    LEGAL_OPINION = "legal_opinion"


class OverlapTypeEnum(str, Enum):
    """Types of plot-territory spatial overlap."""

    FULL_OVERLAP = "full_overlap"
    PARTIAL_OVERLAP = "partial_overlap"
    BOUNDARY_OVERLAP = "boundary_overlap"
    BUFFER_ZONE_OVERLAP = "buffer_zone_overlap"
    NO_OVERLAP = "no_overlap"


class OverlapSeverityEnum(str, Enum):
    """Severity level of a detected overlap."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ConsultationTypeEnum(str, Enum):
    """Types of community consultation activities."""

    PUBLIC_HEARING = "public_hearing"
    COMMUNITY_MEETING = "community_meeting"
    FIELD_VISIT = "field_visit"
    WRITTEN_CONSULTATION = "written_consultation"
    VIRTUAL_MEETING = "virtual_meeting"
    ELDER_COUNCIL = "elder_council"
    TRADITIONAL_ASSEMBLY = "traditional_assembly"
    FOCUS_GROUP = "focus_group"


class ConsultationStatusEnum(str, Enum):
    """Consultation activity status."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FOLLOW_UP_REQUIRED = "follow_up_required"


class ViolationTypeEnum(str, Enum):
    """Types of indigenous rights violations."""

    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MISSING_FPIC = "missing_fpic"
    EXPIRED_CONSENT = "expired_consent"
    LAND_ENCROACHMENT = "land_encroachment"
    CULTURAL_SITE_DAMAGE = "cultural_site_damage"
    FORCED_DISPLACEMENT = "forced_displacement"
    RESOURCE_EXTRACTION = "resource_extraction"
    BOUNDARY_VIOLATION = "boundary_violation"
    CONSULTATION_FAILURE = "consultation_failure"
    BENEFIT_SHARING_BREACH = "benefit_sharing_breach"


class ViolationSeverityEnum(str, Enum):
    """Severity level of a rights violation."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ViolationStatusEnum(str, Enum):
    """Violation lifecycle status."""

    DETECTED = "detected"
    CONFIRMED = "confirmed"
    UNDER_INVESTIGATION = "under_investigation"
    REMEDIATION_PLANNED = "remediation_planned"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"


class CommunityStatusEnum(str, Enum):
    """Indigenous community registry status."""

    ACTIVE = "active"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    INACTIVE = "inactive"


class ComplianceStatusEnum(str, Enum):
    """Overall compliance status for a plot or supplier."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AT_RISK = "at_risk"
    REQUIRES_ASSESSMENT = "requires_assessment"
    REMEDIATION_REQUIRED = "remediation_required"
    EXEMPT = "exempt"


class EUDRCommodityEnum(str, Enum):
    """EUDR-regulated commodity types per Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class SortOrderEnum(str, Enum):
    """Sort order for list endpoints."""

    ASC = "asc"
    DESC = "desc"


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
        default="GL-EUDR-IRC-021",
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
        default_factory=lambda: [
            "Art. 2", "Art. 3", "Art. 9", "Art. 10", "Art. 29",
        ],
        description="Applicable regulatory articles (incl. indigenous rights references)",
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
    srid: int = Field(
        default=4326,
        description="Spatial Reference System Identifier (default WGS84)",
    )

    @field_validator("coordinates")
    @classmethod
    def validate_polygon_closure(cls, v: List[GeoPointSchema]) -> List[GeoPointSchema]:
        """Validate polygon has at least 3 distinct points."""
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 coordinate points")
        return v


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy", description="Service health status")
    agent_id: str = Field(default="GL-EUDR-IRC-021", description="Agent identifier")
    agent: str = Field(default="EUDR-021", description="Agent code")
    component: str = Field(
        default="indigenous-rights-checker", description="Component name"
    )
    version: str = Field(default="1.0.0", description="API version")


# =============================================================================
# 1. Territory Schemas
# =============================================================================


class TerritoryCreateRequest(BaseModel):
    """Request to register an indigenous territory."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Kayapo Indigenous Territory",
                "country_code": "BR",
                "territory_type": "indigenous_land",
                "recognition_level": "legally_titled",
                "community_id": "comm-kayapo-001",
                "boundary": {
                    "coordinates": [
                        {"latitude": "-7.5000", "longitude": "-51.5000"},
                        {"latitude": "-7.5000", "longitude": "-51.0000"},
                        {"latitude": "-8.0000", "longitude": "-51.0000"},
                        {"latitude": "-8.0000", "longitude": "-51.5000"},
                    ],
                    "srid": 4326,
                },
                "area_ha": "1250000.50",
                "description": "Kayapo indigenous territory in Para state, Brazil",
                "legal_reference": "FUNAI Decree 2019/1234",
            }
        }
    )

    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Territory name",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    territory_type: TerritoryTypeEnum = Field(
        ..., description="Type of indigenous territory",
    )
    recognition_level: RecognitionLevelEnum = Field(
        ..., description="Legal recognition level",
    )
    community_id: Optional[str] = Field(
        None, description="Associated community identifier",
    )
    boundary: GeoPolygonSchema = Field(
        ..., description="Territory boundary polygon",
    )
    area_ha: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Territory area in hectares",
    )
    description: Optional[str] = Field(
        None, max_length=5000,
        description="Territory description and context",
    )
    legal_reference: Optional[str] = Field(
        None, max_length=1000,
        description="Legal reference or decree number",
    )
    established_date: Optional[date] = Field(
        None, description="Date territory was legally established",
    )
    tags: Optional[List[str]] = Field(
        None, description="Classification tags",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        normalized = v.strip().upper()
        if not normalized.isalpha() or len(normalized) != 2:
            raise ValueError("Country code must be 2-letter ISO 3166-1 alpha-2")
        return normalized


class TerritoryUpdateRequest(BaseModel):
    """Request to update an existing territory."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Kayapo Indigenous Territory (Updated)",
                "recognition_level": "legally_titled",
                "area_ha": "1260000.00",
                "description": "Updated boundary after 2026 survey",
            }
        }
    )

    name: Optional[str] = Field(
        None, min_length=1, max_length=500,
        description="Updated territory name",
    )
    territory_type: Optional[TerritoryTypeEnum] = Field(
        None, description="Updated territory type",
    )
    recognition_level: Optional[RecognitionLevelEnum] = Field(
        None, description="Updated legal recognition level",
    )
    community_id: Optional[str] = Field(
        None, description="Updated community identifier",
    )
    boundary: Optional[GeoPolygonSchema] = Field(
        None, description="Updated boundary polygon",
    )
    area_ha: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Updated area in hectares",
    )
    status: Optional[TerritoryStatusEnum] = Field(
        None, description="Updated territory status",
    )
    description: Optional[str] = Field(
        None, max_length=5000,
        description="Updated description",
    )
    legal_reference: Optional[str] = Field(
        None, max_length=1000,
        description="Updated legal reference",
    )
    tags: Optional[List[str]] = Field(
        None, description="Updated classification tags",
    )


class TerritoryEntry(BaseModel):
    """Single territory record in responses."""

    territory_id: str = Field(..., description="Unique territory identifier")
    name: str = Field(..., description="Territory name")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    territory_type: TerritoryTypeEnum = Field(..., description="Territory type")
    recognition_level: RecognitionLevelEnum = Field(
        ..., description="Legal recognition level"
    )
    status: TerritoryStatusEnum = Field(..., description="Territory status")
    community_id: Optional[str] = Field(None, description="Associated community ID")
    area_ha: Optional[Decimal] = Field(None, description="Area in hectares")
    boundary: Optional[GeoPolygonSchema] = Field(None, description="Boundary polygon")
    description: Optional[str] = Field(None, description="Territory description")
    legal_reference: Optional[str] = Field(None, description="Legal reference")
    established_date: Optional[date] = Field(None, description="Establishment date")
    tags: Optional[List[str]] = Field(None, description="Classification tags")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class TerritoryResponse(BaseModel):
    """Single territory detail response."""

    territory: TerritoryEntry = Field(..., description="Territory details")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class TerritoryListResponse(BaseModel):
    """Paginated territory list response."""

    territories: List[TerritoryEntry] = Field(
        default_factory=list, description="Territory entries"
    )
    total_territories: int = Field(0, ge=0, description="Total count")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 2. FPIC Schemas
# =============================================================================


class FPICVerifyRequest(BaseModel):
    """Request to verify FPIC documentation for a plot/territory pair."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plot_id": "plot-br-001",
                "territory_id": "terr-kayapo-001",
                "supplier_id": "supplier-abc",
                "document_ids": ["doc-fpic-001", "doc-fpic-002"],
                "commodity": "soya",
                "verification_date": "2026-03-01",
            }
        }
    )

    plot_id: str = Field(..., description="Plot identifier to verify FPIC for")
    territory_id: str = Field(..., description="Indigenous territory identifier")
    supplier_id: Optional[str] = Field(None, description="Associated supplier ID")
    document_ids: Optional[List[str]] = Field(
        None, description="FPIC document IDs to verify"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity type for context"
    )
    verification_date: Optional[date] = Field(
        None, description="Target verification date"
    )


class FPICVerifyResponse(BaseModel):
    """FPIC verification result."""

    verification_id: str = Field(
        default_factory=_new_id, description="Unique verification identifier"
    )
    plot_id: str = Field(..., description="Verified plot ID")
    territory_id: str = Field(..., description="Verified territory ID")
    fpic_status: FPICStatusEnum = Field(..., description="FPIC verification outcome")
    compliance_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="FPIC compliance score (0-100)",
    )
    documents_verified: int = Field(0, ge=0, description="Number of documents verified")
    documents_valid: int = Field(0, ge=0, description="Number of valid documents")
    issues: List[str] = Field(
        default_factory=list, description="Identified verification issues"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    expiry_date: Optional[date] = Field(
        None, description="FPIC consent expiry date"
    )
    verified_at: datetime = Field(
        default_factory=_utcnow, description="Verification timestamp"
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class FPICDocumentEntry(BaseModel):
    """Single FPIC document record."""

    document_id: str = Field(..., description="Unique document identifier")
    territory_id: str = Field(..., description="Associated territory ID")
    community_id: Optional[str] = Field(None, description="Associated community ID")
    document_type: FPICDocumentTypeEnum = Field(
        ..., description="Type of FPIC document"
    )
    title: str = Field(..., description="Document title")
    status: FPICStatusEnum = Field(..., description="Document status")
    issue_date: Optional[date] = Field(None, description="Date document was issued")
    expiry_date: Optional[date] = Field(None, description="Document expiry date")
    signatories: Optional[List[str]] = Field(
        None, description="Document signatories"
    )
    language: Optional[str] = Field(None, description="Document language code")
    storage_url: Optional[str] = Field(None, description="Document storage URL")
    verification_notes: Optional[str] = Field(
        None, description="Verification notes"
    )
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class FPICDocumentResponse(BaseModel):
    """Single FPIC document detail response."""

    document: FPICDocumentEntry = Field(..., description="Document details")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class FPICDocumentListResponse(BaseModel):
    """Paginated FPIC document list response."""

    documents: List[FPICDocumentEntry] = Field(
        default_factory=list, description="FPIC document entries"
    )
    total_documents: int = Field(0, ge=0, description="Total document count")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class FPICScoreRequest(BaseModel):
    """Request to calculate FPIC compliance score for a territory."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "territory_id": "terr-kayapo-001",
                "include_expired": False,
                "weight_document_quality": True,
            }
        }
    )

    territory_id: str = Field(..., description="Territory to score")
    plot_id: Optional[str] = Field(None, description="Optional plot for context")
    include_expired: bool = Field(
        False, description="Include expired documents in scoring"
    )
    weight_document_quality: bool = Field(
        True, description="Apply quality weighting to documents"
    )


class FPICScoreResponse(BaseModel):
    """FPIC compliance score response."""

    territory_id: str = Field(..., description="Scored territory ID")
    overall_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Overall FPIC compliance score (0-100)",
    )
    fpic_status: FPICStatusEnum = Field(
        ..., description="Derived FPIC status from score"
    )
    document_count: int = Field(0, ge=0, description="Total documents assessed")
    valid_document_count: int = Field(0, ge=0, description="Valid documents count")
    expired_document_count: int = Field(
        0, ge=0, description="Expired documents count"
    )
    score_breakdown: Optional[Dict[str, Decimal]] = Field(
        None, description="Score breakdown by category"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Identified risk factors"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended improvements"
    )
    scored_at: datetime = Field(
        default_factory=_utcnow, description="Scoring timestamp"
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 3. Overlap Schemas
# =============================================================================


class OverlapAnalyzeRequest(BaseModel):
    """Request to analyze overlap between a plot and indigenous territories."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plot_id": "plot-br-001",
                "plot_boundary": {
                    "coordinates": [
                        {"latitude": "-7.6000", "longitude": "-51.3000"},
                        {"latitude": "-7.6000", "longitude": "-51.2000"},
                        {"latitude": "-7.7000", "longitude": "-51.2000"},
                        {"latitude": "-7.7000", "longitude": "-51.3000"},
                    ],
                    "srid": 4326,
                },
                "buffer_km": "5.0",
                "include_buffer_zones": True,
            }
        }
    )

    plot_id: str = Field(..., description="Plot identifier to check for overlaps")
    plot_boundary: Optional[GeoPolygonSchema] = Field(
        None, description="Plot boundary polygon (optional if plot is registered)"
    )
    territory_ids: Optional[List[str]] = Field(
        None, description="Specific territory IDs to check (all if omitted)"
    )
    buffer_km: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Buffer distance in kilometers for proximity analysis",
    )
    include_buffer_zones: bool = Field(
        True, description="Include buffer zone overlaps in analysis"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity for risk context"
    )


class OverlapEntry(BaseModel):
    """Single overlap analysis result."""

    overlap_id: str = Field(
        default_factory=_new_id, description="Unique overlap identifier"
    )
    plot_id: str = Field(..., description="Plot identifier")
    territory_id: str = Field(..., description="Overlapping territory ID")
    territory_name: Optional[str] = Field(
        None, description="Territory name"
    )
    overlap_type: OverlapTypeEnum = Field(..., description="Type of overlap")
    overlap_severity: OverlapSeverityEnum = Field(
        ..., description="Overlap severity level"
    )
    overlap_area_ha: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Overlapping area in hectares",
    )
    overlap_percentage: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
        description="Percentage of plot overlapping territory",
    )
    distance_km: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Distance to nearest territory boundary in km",
    )
    fpic_status: Optional[FPICStatusEnum] = Field(
        None, description="FPIC status for the overlapping territory"
    )
    community_name: Optional[str] = Field(
        None, description="Name of affected community"
    )
    requires_fpic: bool = Field(
        True, description="Whether FPIC is required for this overlap"
    )
    detected_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp"
    )


class OverlapAnalyzeResponse(BaseModel):
    """Overlap analysis result for a single plot."""

    plot_id: str = Field(..., description="Analyzed plot ID")
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Detected overlaps"
    )
    total_overlaps: int = Field(0, ge=0, description="Total overlaps detected")
    has_critical_overlap: bool = Field(
        False, description="Whether any critical overlaps were detected"
    )
    requires_fpic: bool = Field(
        False, description="Whether FPIC is required based on overlaps"
    )
    risk_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
        description="Aggregate indigenous rights risk score (0-100)",
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class OverlapListResponse(BaseModel):
    """Paginated overlap list response."""

    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Overlap entries"
    )
    total_overlaps: int = Field(0, ge=0, description="Total overlaps")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class OverlapBulkRequest(BaseModel):
    """Request for bulk overlap analysis of multiple plots."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plot_ids": ["plot-br-001", "plot-br-002", "plot-br-003"],
                "buffer_km": "10.0",
                "include_buffer_zones": True,
            }
        }
    )

    plot_ids: List[str] = Field(
        ..., min_length=1, max_length=500,
        description="Plot IDs to analyze (1-500)",
    )
    buffer_km: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Buffer distance in kilometers",
    )
    include_buffer_zones: bool = Field(
        True, description="Include buffer zone overlaps"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity for risk context"
    )


class OverlapBulkResultEntry(BaseModel):
    """Single plot result within a bulk overlap analysis."""

    plot_id: str = Field(..., description="Plot identifier")
    total_overlaps: int = Field(0, ge=0, description="Overlaps detected for this plot")
    has_critical_overlap: bool = Field(False, description="Critical overlaps found")
    requires_fpic: bool = Field(False, description="FPIC required")
    risk_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
        description="Risk score for this plot",
    )
    overlaps: List[OverlapEntry] = Field(
        default_factory=list, description="Overlap details"
    )


class OverlapBulkResponse(BaseModel):
    """Bulk overlap analysis response."""

    total_plots: int = Field(..., ge=0, description="Total plots analyzed")
    plots_with_overlaps: int = Field(
        0, ge=0, description="Plots that have overlaps"
    )
    plots_requiring_fpic: int = Field(
        0, ge=0, description="Plots that require FPIC"
    )
    results: List[OverlapBulkResultEntry] = Field(
        default_factory=list, description="Per-plot results"
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 4. Consultation Schemas
# =============================================================================


class ConsultationCreateRequest(BaseModel):
    """Request to record a community consultation activity."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "territory_id": "terr-kayapo-001",
                "community_id": "comm-kayapo-001",
                "consultation_type": "community_meeting",
                "title": "FPIC Consent Discussion for Plot BR-001",
                "description": "Community meeting to discuss land use for soya production",
                "scheduled_date": "2026-03-15",
                "location": "Kayapo Community Center, Para, Brazil",
                "attendees": ["Chief Leader", "Community Elder", "Company Rep"],
                "language": "pt-BR",
            }
        }
    )

    territory_id: str = Field(..., description="Associated territory ID")
    community_id: str = Field(..., description="Associated community ID")
    consultation_type: ConsultationTypeEnum = Field(
        ..., description="Type of consultation"
    )
    title: str = Field(
        ..., min_length=1, max_length=500,
        description="Consultation title",
    )
    description: Optional[str] = Field(
        None, max_length=5000,
        description="Consultation description and agenda",
    )
    scheduled_date: Optional[date] = Field(
        None, description="Scheduled consultation date"
    )
    completed_date: Optional[date] = Field(
        None, description="Actual completion date"
    )
    location: Optional[str] = Field(
        None, max_length=1000,
        description="Physical or virtual location",
    )
    attendees: Optional[List[str]] = Field(
        None, description="List of attendee names/roles"
    )
    language: Optional[str] = Field(
        None, max_length=10,
        description="Primary language used (ISO 639 code)",
    )
    outcomes: Optional[List[str]] = Field(
        None, description="Consultation outcomes and decisions"
    )
    follow_up_actions: Optional[List[str]] = Field(
        None, description="Required follow-up actions"
    )
    documents: Optional[List[str]] = Field(
        None, description="Related document IDs"
    )


class ConsultationEntry(BaseModel):
    """Single consultation record."""

    consultation_id: str = Field(..., description="Unique consultation identifier")
    territory_id: str = Field(..., description="Associated territory ID")
    community_id: str = Field(..., description="Associated community ID")
    consultation_type: ConsultationTypeEnum = Field(
        ..., description="Consultation type"
    )
    title: str = Field(..., description="Consultation title")
    status: ConsultationStatusEnum = Field(..., description="Consultation status")
    description: Optional[str] = Field(None, description="Description")
    scheduled_date: Optional[date] = Field(None, description="Scheduled date")
    completed_date: Optional[date] = Field(None, description="Completion date")
    location: Optional[str] = Field(None, description="Location")
    attendees: Optional[List[str]] = Field(None, description="Attendees")
    attendee_count: Optional[int] = Field(None, ge=0, description="Attendee count")
    language: Optional[str] = Field(None, description="Primary language")
    outcomes: Optional[List[str]] = Field(None, description="Outcomes")
    follow_up_actions: Optional[List[str]] = Field(None, description="Follow-ups")
    documents: Optional[List[str]] = Field(None, description="Document IDs")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class ConsultationResponse(BaseModel):
    """Single consultation detail response."""

    consultation: ConsultationEntry = Field(..., description="Consultation details")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ConsultationListResponse(BaseModel):
    """Paginated consultation list response."""

    consultations: List[ConsultationEntry] = Field(
        default_factory=list, description="Consultation entries"
    )
    total_consultations: int = Field(0, ge=0, description="Total consultation count")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 5. Violation Schemas
# =============================================================================


class ViolationDetectRequest(BaseModel):
    """Request to detect rights violations for a plot or territory."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plot_id": "plot-br-001",
                "territory_id": "terr-kayapo-001",
                "supplier_id": "supplier-abc",
                "check_types": ["missing_fpic", "land_encroachment", "expired_consent"],
                "commodity": "soya",
            }
        }
    )

    plot_id: Optional[str] = Field(None, description="Plot to check for violations")
    territory_id: Optional[str] = Field(
        None, description="Territory to check for violations"
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier to check"
    )
    check_types: Optional[List[ViolationTypeEnum]] = Field(
        None, description="Specific violation types to check (all if omitted)"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity context"
    )
    include_resolved: bool = Field(
        False, description="Include previously resolved violations"
    )


class ViolationEntry(BaseModel):
    """Single violation record."""

    violation_id: str = Field(..., description="Unique violation identifier")
    violation_type: ViolationTypeEnum = Field(..., description="Violation type")
    severity: ViolationSeverityEnum = Field(..., description="Violation severity")
    status: ViolationStatusEnum = Field(..., description="Violation status")
    plot_id: Optional[str] = Field(None, description="Affected plot ID")
    territory_id: Optional[str] = Field(None, description="Affected territory ID")
    community_id: Optional[str] = Field(None, description="Affected community ID")
    supplier_id: Optional[str] = Field(None, description="Responsible supplier ID")
    description: str = Field(..., description="Violation description")
    evidence: Optional[List[str]] = Field(None, description="Evidence references")
    remediation_actions: Optional[List[str]] = Field(
        None, description="Required remediation actions"
    )
    detected_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp"
    )
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")


class ViolationDetectResponse(BaseModel):
    """Violation detection result."""

    violations: List[ViolationEntry] = Field(
        default_factory=list, description="Detected violations"
    )
    total_violations: int = Field(0, ge=0, description="Total violations detected")
    critical_count: int = Field(0, ge=0, description="Critical violations count")
    high_count: int = Field(0, ge=0, description="High severity count")
    compliance_impact: ComplianceStatusEnum = Field(
        ..., description="Overall compliance impact"
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ViolationResponse(BaseModel):
    """Single violation detail response."""

    violation: ViolationEntry = Field(..., description="Violation details")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ViolationListResponse(BaseModel):
    """Paginated violation list response."""

    violations: List[ViolationEntry] = Field(
        default_factory=list, description="Violation entries"
    )
    total_violations: int = Field(0, ge=0, description="Total violation count")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ViolationResolveRequest(BaseModel):
    """Request to resolve a violation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "resolution_notes": "FPIC obtained from Kayapo community council on 2026-03-10",
                "remediation_actions_taken": [
                    "Community meeting held",
                    "FPIC consent document signed",
                    "Benefit sharing agreement executed",
                ],
                "evidence_ids": ["doc-fpic-003", "doc-fpic-004"],
            }
        }
    )

    resolution_notes: str = Field(
        ..., min_length=1, max_length=5000,
        description="Explanation of how the violation was resolved",
    )
    remediation_actions_taken: Optional[List[str]] = Field(
        None, description="Actions taken to remediate the violation"
    )
    evidence_ids: Optional[List[str]] = Field(
        None, description="Evidence document IDs supporting resolution"
    )


class ViolationResolveResponse(BaseModel):
    """Violation resolution result."""

    violation: ViolationEntry = Field(..., description="Updated violation record")
    resolution_accepted: bool = Field(
        ..., description="Whether the resolution was accepted"
    )
    remaining_actions: Optional[List[str]] = Field(
        None, description="Any remaining actions required"
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 6. Community Registry Schemas
# =============================================================================


class CommunityRegisterRequest(BaseModel):
    """Request to register an indigenous community."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Kayapo People",
                "country_code": "BR",
                "region": "Para State",
                "population": 12000,
                "language": "Kayapo",
                "territory_ids": ["terr-kayapo-001", "terr-kayapo-002"],
                "contact_info": "FUNAI Regional Office, Para",
                "recognition_status": "legally_titled",
            }
        }
    )

    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Community name",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: Optional[str] = Field(
        None, max_length=500,
        description="Administrative region or state",
    )
    population: Optional[int] = Field(
        None, ge=0, description="Estimated community population"
    )
    language: Optional[str] = Field(
        None, max_length=200,
        description="Primary language spoken",
    )
    secondary_languages: Optional[List[str]] = Field(
        None, description="Additional languages spoken"
    )
    territory_ids: Optional[List[str]] = Field(
        None, description="Associated territory IDs"
    )
    contact_info: Optional[str] = Field(
        None, max_length=2000,
        description="Community or representative contact information",
    )
    recognition_status: Optional[RecognitionLevelEnum] = Field(
        None, description="Legal recognition status"
    )
    description: Optional[str] = Field(
        None, max_length=5000,
        description="Community description and cultural context",
    )
    tags: Optional[List[str]] = Field(
        None, description="Classification tags"
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        normalized = v.strip().upper()
        if not normalized.isalpha() or len(normalized) != 2:
            raise ValueError("Country code must be 2-letter ISO 3166-1 alpha-2")
        return normalized


class CommunityEntry(BaseModel):
    """Single community record."""

    community_id: str = Field(..., description="Unique community identifier")
    name: str = Field(..., description="Community name")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    status: CommunityStatusEnum = Field(..., description="Registry status")
    region: Optional[str] = Field(None, description="Region or state")
    population: Optional[int] = Field(None, ge=0, description="Estimated population")
    language: Optional[str] = Field(None, description="Primary language")
    secondary_languages: Optional[List[str]] = Field(
        None, description="Additional languages"
    )
    territory_ids: Optional[List[str]] = Field(
        None, description="Associated territory IDs"
    )
    territory_count: Optional[int] = Field(
        None, ge=0, description="Number of associated territories"
    )
    contact_info: Optional[str] = Field(None, description="Contact information")
    recognition_status: Optional[RecognitionLevelEnum] = Field(
        None, description="Legal recognition"
    )
    description: Optional[str] = Field(None, description="Description")
    tags: Optional[List[str]] = Field(None, description="Classification tags")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class CommunityResponse(BaseModel):
    """Single community detail response."""

    community: CommunityEntry = Field(..., description="Community details")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class CommunityListResponse(BaseModel):
    """Paginated community list response."""

    communities: List[CommunityEntry] = Field(
        default_factory=list, description="Community entries"
    )
    total_communities: int = Field(0, ge=0, description="Total community count")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# 7. Compliance Schemas
# =============================================================================


class ComplianceReportResponse(BaseModel):
    """Compliance report for a specific plot."""

    report_id: str = Field(
        default_factory=_new_id, description="Unique report identifier"
    )
    plot_id: str = Field(..., description="Assessed plot ID")
    compliance_status: ComplianceStatusEnum = Field(
        ..., description="Overall compliance status"
    )
    overall_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Overall compliance score (0-100)",
    )
    territory_overlaps: int = Field(
        0, ge=0, description="Number of territory overlaps"
    )
    fpic_status: FPICStatusEnum = Field(
        ..., description="FPIC status for the plot"
    )
    active_violations: int = Field(
        0, ge=0, description="Active violations count"
    )
    consultations_completed: int = Field(
        0, ge=0, description="Completed consultations count"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Identified compliance risk factors"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Compliance recommendations"
    )
    affected_communities: List[str] = Field(
        default_factory=list, description="Affected community names"
    )
    generated_at: datetime = Field(
        default_factory=_utcnow, description="Report generation timestamp"
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


class ComplianceAssessRequest(BaseModel):
    """Request for a full compliance assessment."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plot_ids": ["plot-br-001", "plot-br-002"],
                "supplier_id": "supplier-abc",
                "commodity": "soya",
                "include_overlap_analysis": True,
                "include_fpic_verification": True,
                "include_violation_check": True,
                "include_consultation_review": True,
            }
        }
    )

    plot_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="Plot IDs to assess (1-100)",
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier identifier for context"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity type for context"
    )
    include_overlap_analysis: bool = Field(
        True, description="Include territory overlap analysis"
    )
    include_fpic_verification: bool = Field(
        True, description="Include FPIC verification"
    )
    include_violation_check: bool = Field(
        True, description="Include violation detection"
    )
    include_consultation_review: bool = Field(
        True, description="Include consultation status review"
    )


class CompliancePlotSummary(BaseModel):
    """Per-plot compliance summary within an assessment."""

    plot_id: str = Field(..., description="Plot identifier")
    compliance_status: ComplianceStatusEnum = Field(
        ..., description="Compliance status"
    )
    score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Compliance score (0-100)",
    )
    overlap_count: int = Field(0, ge=0, description="Territory overlaps")
    fpic_status: Optional[FPICStatusEnum] = Field(None, description="FPIC status")
    violation_count: int = Field(0, ge=0, description="Active violations")
    issues: List[str] = Field(
        default_factory=list, description="Identified issues"
    )


class ComplianceAssessResponse(BaseModel):
    """Full compliance assessment response."""

    assessment_id: str = Field(
        default_factory=_new_id, description="Unique assessment identifier"
    )
    overall_compliance: ComplianceStatusEnum = Field(
        ..., description="Aggregate compliance status"
    )
    overall_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Aggregate compliance score (0-100)",
    )
    total_plots_assessed: int = Field(0, ge=0, description="Total plots assessed")
    compliant_plots: int = Field(0, ge=0, description="Compliant plots count")
    non_compliant_plots: int = Field(
        0, ge=0, description="Non-compliant plots count"
    )
    at_risk_plots: int = Field(0, ge=0, description="At-risk plots count")
    plots: List[CompliancePlotSummary] = Field(
        default_factory=list, description="Per-plot summaries"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Aggregate risk factors"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Aggregate recommendations"
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow, description="Assessment timestamp"
    )
    provenance: ProvenanceInfo = Field(..., description="Audit trail provenance")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums
    "CommunityStatusEnum",
    "ComplianceStatusEnum",
    "ConsultationStatusEnum",
    "ConsultationTypeEnum",
    "EUDRCommodityEnum",
    "FPICDocumentTypeEnum",
    "FPICStatusEnum",
    "OverlapSeverityEnum",
    "OverlapTypeEnum",
    "RecognitionLevelEnum",
    "SortOrderEnum",
    "TerritoryStatusEnum",
    "TerritoryTypeEnum",
    "ViolationSeverityEnum",
    "ViolationStatusEnum",
    "ViolationTypeEnum",
    # Common
    "ErrorResponse",
    "GeoPointSchema",
    "GeoPolygonSchema",
    "HealthResponse",
    "MetadataSchema",
    "PaginatedMeta",
    "ProvenanceInfo",
    # Territory
    "TerritoryCreateRequest",
    "TerritoryEntry",
    "TerritoryListResponse",
    "TerritoryResponse",
    "TerritoryUpdateRequest",
    # FPIC
    "FPICDocumentEntry",
    "FPICDocumentListResponse",
    "FPICDocumentResponse",
    "FPICScoreRequest",
    "FPICScoreResponse",
    "FPICVerifyRequest",
    "FPICVerifyResponse",
    # Overlap
    "OverlapAnalyzeRequest",
    "OverlapAnalyzeResponse",
    "OverlapBulkRequest",
    "OverlapBulkResponse",
    "OverlapBulkResultEntry",
    "OverlapEntry",
    "OverlapListResponse",
    # Consultation
    "ConsultationCreateRequest",
    "ConsultationEntry",
    "ConsultationListResponse",
    "ConsultationResponse",
    # Violation
    "ViolationDetectRequest",
    "ViolationDetectResponse",
    "ViolationEntry",
    "ViolationListResponse",
    "ViolationResolveRequest",
    "ViolationResolveResponse",
    "ViolationResponse",
    # Registry
    "CommunityEntry",
    "CommunityListResponse",
    "CommunityRegisterRequest",
    "CommunityResponse",
    # Compliance
    "ComplianceAssessRequest",
    "ComplianceAssessResponse",
    "CompliancePlotSummary",
    "ComplianceReportResponse",
]
