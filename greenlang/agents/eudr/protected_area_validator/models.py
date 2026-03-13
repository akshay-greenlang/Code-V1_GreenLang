# -*- coding: utf-8 -*-
"""
Protected Area Validator Data Models - AGENT-EUDR-022

Pydantic v2 data models for the Protected Area Validator Agent covering
WDPA integration (270,000+ protected areas), IUCN-category-aware spatial
overlap detection, configurable buffer zone monitoring (1-50 km), protected
area designation validation (international/national/regional/local),
high-risk proximity alerting (UNESCO WH, Ramsar, AZE, KBA), compliance
tracking with remediation workflows, conservation status assessment, and
audit-ready compliance reporting in 8 formats and 5 languages.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all protected area validation operations per
EU 2023/1115 Articles 2, 9, 10, 11, 29, and 31.

Enumerations (18):
    - IUCNCategory, PAOverlapType, DesignationLevel, PALegalStatus,
      GovernanceType, BufferProximityTier, PAComplianceStatus,
      AlertSeverity, RiskLevel, ViolationType, DetectionMethod,
      EncroachmentTrend, ReportType, ReportFormat, ReportLanguage,
      DataSource, GISQuality, EUDRCommodity

Core Models (14):
    - ProtectedArea, ProtectedAreaOverlap, BufferZoneResult,
      ProximityAlert, PAComplianceRecord, ConservationAssessment,
      DesignationValidation, RiskScoreBreakdown, BatchScreeningJob,
      ComplianceReport, IntegrationEvent, AuditLogEntry,
      ProtectedAreaVersion, BufferZoneExemption

Request Models (7):
    - CheckOverlapRequest, BatchOverlapRequest, BufferAnalysisRequest,
      ValidateDesignationRequest, ProximityAlertRequest,
      ComplianceReportRequest, ConservationAssessmentRequest

Response Models (7):
    - CheckOverlapResponse, BatchOverlapResponse, BufferAnalysisResponse,
      ValidateDesignationResponse, ProximityAlertResponse,
      ComplianceReportResponse, ConservationAssessmentResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 Protected Area Validator (GL-EUDR-PAV-022)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION: str = "1.0.0"
EUDR_CUTOFF_DATE: str = "2020-12-31"
MAX_BUFFER_RADIUS_KM: int = 50
MIN_BUFFER_RADIUS_KM: int = 1
MAX_BATCH_SIZE: int = 10000
EUDR_RETENTION_YEARS: int = 5
DEFAULT_BUFFER_RESOLUTION: int = 64

SUPPORTED_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

SUPPORTED_BUFFER_RADII: List[int] = [1, 5, 10, 25, 50]


# ---------------------------------------------------------------------------
# Enumerations (18)
# ---------------------------------------------------------------------------

class IUCNCategory(str, Enum):
    """IUCN Management Categories for protected areas.

    Classification system from the International Union for Conservation
    of Nature. Categories range from Ia (strictest) to VI (sustainable
    use) with an NR option for areas where the category is not reported.
    """
    IA = "Ia"
    """Strict Nature Reserve: absolutely prohibited commodity production."""
    IB = "Ib"
    """Wilderness Area: absolutely prohibited commodity production."""
    II = "II"
    """National Park: prohibited except limited traditional use."""
    III = "III"
    """Natural Monument: prohibited commodity production."""
    IV = "IV"
    """Habitat/Species Management Area: generally prohibited."""
    V = "V"
    """Protected Landscape/Seascape: conditionally allowed."""
    VI = "VI"
    """Protected Area with Sustainable Use: allowed under management plan."""
    NOT_REPORTED = "NR"
    """Not Reported: IUCN category not assigned, treated as HIGH default."""


class PAOverlapType(str, Enum):
    """Classification of overlap between a production plot and a protected area."""
    INSIDE = "inside"
    """Plot entirely within protected area."""
    PARTIAL = "partial"
    """Polygon intersection with partial overlap."""
    BOUNDARY = "boundary"
    """Plot boundary touches PA boundary."""
    BUFFER = "buffer"
    """Plot within buffer zone only, not overlapping."""
    CLEAR = "clear"
    """Plot outside all buffer zones."""


class DesignationLevel(str, Enum):
    """Protected area designation level tier."""
    INTERNATIONAL = "international"
    """UNESCO World Heritage, Ramsar, MAB Biosphere Reserve."""
    NATIONAL = "national"
    """National park, national reserve, wildlife sanctuary."""
    REGIONAL = "regional"
    """State/provincial protected area, regional park."""
    LOCAL = "local"
    """Municipal reserve, community conserved area, private reserve."""
    PROPOSED = "proposed"
    """Under consideration, not yet gazetted."""


class PALegalStatus(str, Enum):
    """Protected area legal status from WDPA."""
    DESIGNATED = "designated"
    """Formally gazetted under national law."""
    PROPOSED = "proposed"
    """Under consideration, not yet gazetted."""
    INSCRIBED = "inscribed"
    """Internationally inscribed (e.g., World Heritage)."""
    ADOPTED = "adopted"
    """Adopted but not yet enforced."""
    ESTABLISHED = "established"
    """Established without formal gazettement."""


class GovernanceType(str, Enum):
    """Protected area governance type from WDPA."""
    FEDERAL = "federal"
    SUBNATIONAL = "subnational"
    COLLABORATIVE = "collaborative"
    PRIVATE = "private"
    INDIGENOUS = "indigenous"
    NOT_REPORTED = "not_reported"


class BufferProximityTier(str, Enum):
    """Buffer zone proximity tier classification."""
    IMMEDIATE = "immediate"
    """0-1 km from protected area boundary."""
    CLOSE = "close"
    """1-5 km from protected area boundary."""
    MODERATE = "moderate"
    """5-10 km from protected area boundary."""
    DISTANT = "distant"
    """10-25 km from protected area boundary."""
    PERIPHERAL = "peripheral"
    """25-50 km from protected area boundary."""


class PAComplianceStatus(str, Enum):
    """Compliance status lifecycle for plot-protected area pairs."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    VIOLATION_CONFIRMED = "violation_confirmed"
    REMEDIATION_PLANNED = "remediation_planned"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    REMEDIATED = "remediated"
    EXEMPTION_GRANTED = "exemption_granted"
    FALSE_POSITIVE = "false_positive"


class AlertSeverity(str, Enum):
    """Severity levels for proximity alerts."""
    CRITICAL = "critical"
    """Inside or < 1 km from high-risk protected area."""
    SEVERE = "severe"
    """1-5 km from high-risk protected area."""
    HIGH = "high"
    """5-10 km from high-risk protected area."""
    ELEVATED = "elevated"
    """10-25 km from high-risk protected area."""
    STANDARD = "standard"
    """> 25 km or low-risk protected area."""


class RiskLevel(str, Enum):
    """Protected area risk level classification."""
    CRITICAL = "critical"
    """Risk score >= 80."""
    HIGH = "high"
    """60 <= Risk score < 80."""
    MEDIUM = "medium"
    """40 <= Risk score < 60."""
    LOW = "low"
    """20 <= Risk score < 40."""
    CLEAR = "clear"
    """Risk score < 20."""


class ViolationType(str, Enum):
    """Type of protected area violation."""
    ENCROACHMENT = "encroachment"
    ILLEGAL_CLEARING = "illegal_clearing"
    UNAUTHORIZED_CONSTRUCTION = "unauthorized_construction"
    RESOURCE_EXTRACTION = "resource_extraction"


class DetectionMethod(str, Enum):
    """Method by which violation was detected."""
    WDPA_OVERLAP = "wdpa_overlap"
    SATELLITE_ALERT = "satellite_alert"
    FIELD_INSPECTION = "field_inspection"
    THIRD_PARTY_REPORT = "third_party_report"


class EncroachmentTrend(str, Enum):
    """Encroachment trend direction."""
    APPROACHING = "approaching"
    STABLE = "stable"
    RETREATING = "retreating"


class ReportType(str, Enum):
    """Protected area compliance report types."""
    FULL_COMPLIANCE = "full_compliance"
    EXECUTIVE_SUMMARY = "executive_summary"
    SUPPLIER_SCORECARD = "supplier_scorecard"
    COMMODITY_ANALYSIS = "commodity_analysis"
    DDS_SECTION = "dds_section"
    CERTIFICATION_AUDIT = "certification_audit"
    TREND_ANALYSIS = "trend_analysis"
    BI_EXPORT = "bi_export"


class ReportFormat(str, Enum):
    """Output formats for compliance reports."""
    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    XLSX = "xlsx"


class ReportLanguage(str, Enum):
    """Supported languages for report generation."""
    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    PT = "pt"


class DataSource(str, Enum):
    """Data source for protected area records."""
    WDPA = "wdpa"
    PROTECTED_PLANET = "protected_planet"
    NATIONAL_REGISTRY = "national_registry"
    KBA_DATABASE = "kba_database"
    UNESCO = "unesco"
    RAMSAR = "ramsar"


class GISQuality(str, Enum):
    """GIS boundary quality level."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodities per Article 1."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


# ---------------------------------------------------------------------------
# Core Models (14)
# ---------------------------------------------------------------------------

class ProtectedArea(BaseModel):
    """Protected area record from WDPA and supplementary data sources.

    Represents a single protected area with spatial boundary, IUCN
    classification, designation status, and management metadata. This
    is the primary reference entity for all overlap detection and risk
    scoring operations.

    Attributes:
        wdpa_id: WDPA unique identifier.
        name: Protected area official name.
        orig_name: Name in local language.
        designation: Designation type name.
        designation_level: International/National/Regional/Local.
        iucn_category: IUCN management category.
        country_code: ISO 3166-1 alpha-3 country code.
        iso3: ISO3 code.
        area_hectares: Total reported area.
        marine_area_hectares: Marine component area.
        legal_status: Legal designation status.
        status_year: Year of current status.
        governance_type: Governance classification.
        management_authority: Managing body name.
        mett_score: METT management effectiveness score (0-100).
        boundary_geojson: GeoJSON polygon/multipolygon.
        is_world_heritage: UNESCO World Heritage Site.
        is_ramsar: Ramsar Wetland designation.
        is_biosphere: MAB Biosphere Reserve.
        is_kba: Key Biodiversity Area.
        is_aze: Alliance for Zero Extinction site.
        wdpa_version: WDPA release version.
        data_source: Source database identifier.
        gis_quality: Boundary accuracy level.
        confidence: Overall data confidence.
        provenance_hash: SHA-256 hash for audit trail.
        last_verified: Last verification timestamp.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    wdpa_id: int = Field(..., description="WDPA unique identifier")
    name: str = Field(..., min_length=1, max_length=500, description="Protected area name")
    orig_name: Optional[str] = Field(None, max_length=500, description="Name in local language")
    designation: str = Field(..., max_length=500, description="Designation type name")
    designation_level: DesignationLevel = Field(..., description="Designation tier")
    iucn_category: IUCNCategory = Field(..., description="IUCN management category")
    country_code: str = Field(..., min_length=2, max_length=3, description="ISO country code")
    iso3: str = Field(..., min_length=3, max_length=3, description="ISO3 code")
    area_hectares: Decimal = Field(Decimal("0"), ge=Decimal("0"), description="Total area (ha)")
    marine_area_hectares: Decimal = Field(Decimal("0"), ge=Decimal("0"), description="Marine area (ha)")
    legal_status: PALegalStatus = Field(..., description="Legal designation status")
    status_year: Optional[int] = Field(None, ge=1800, le=2100, description="Status year")
    governance_type: GovernanceType = Field(
        GovernanceType.NOT_REPORTED, description="Governance classification"
    )
    management_authority: Optional[str] = Field(None, max_length=500)
    management_plan: Optional[str] = Field(None, max_length=50)
    mett_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"), description="METT score 0-100"
    )
    boundary_geojson: Optional[Dict[str, Any]] = Field(None, description="GeoJSON boundary")
    is_world_heritage: bool = Field(False, description="UNESCO WH designation")
    is_ramsar: bool = Field(False, description="Ramsar Wetland")
    is_biosphere: bool = Field(False, description="MAB Biosphere Reserve")
    is_kba: bool = Field(False, description="Key Biodiversity Area")
    is_aze: bool = Field(False, description="Alliance for Zero Extinction site")
    wdpa_version: str = Field(..., description="WDPA release version")
    data_source: str = Field("WDPA", description="Data source identifier")
    gis_quality: GISQuality = Field(GISQuality.MEDIUM, description="Boundary accuracy")
    confidence: GISQuality = Field(GISQuality.MEDIUM, description="Data confidence")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    last_verified: datetime = Field(default_factory=_utcnow, description="Last verified")
    created_at: datetime = Field(default_factory=_utcnow, description="Created timestamp")

    @field_validator("country_code", "iso3")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country codes are uppercase."""
        return v.upper()


class ProtectedAreaOverlap(BaseModel):
    """Overlap analysis result between a production plot and a protected area.

    Records the spatial relationship, overlap metrics, risk scoring, and
    cross-reference correlations for a single plot-PA pair.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    overlap_id: str = Field(default_factory=_new_uuid, description="Overlap record UUID")
    plot_id: str = Field(..., min_length=1, description="Production plot identifier")
    wdpa_id: int = Field(..., description="WDPA protected area ID")
    protected_area_name: str = Field(..., description="Protected area name")
    iucn_category: IUCNCategory = Field(..., description="IUCN category")
    designation_level: DesignationLevel = Field(..., description="Designation tier")
    overlap_type: PAOverlapType = Field(..., description="Overlap classification")
    overlap_area_hectares: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Overlap area (ha)"
    )
    overlap_pct_of_plot: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"), description="% of plot area"
    )
    overlap_pct_of_pa: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"), description="% of PA area"
    )
    distance_meters: Decimal = Field(
        ..., ge=Decimal("0"), description="Min distance to PA boundary (m)"
    )
    bearing_degrees: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("360"), description="Bearing to nearest PA boundary"
    )
    risk_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"), description="Composite risk score 0-100"
    )
    risk_level: RiskLevel = Field(..., description="Risk classification")
    designation_strength: Decimal = Field(
        Decimal("50"), ge=Decimal("0"), le=Decimal("100"), description="Designation strength 0-100"
    )
    deforestation_correlation: bool = Field(False, description="Cross-ref with EUDR-020")
    indigenous_rights_correlation: bool = Field(False, description="Cross-ref with EUDR-021")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    detected_at: datetime = Field(default_factory=_utcnow, description="Detection timestamp")


class BufferZoneResult(BaseModel):
    """Buffer zone analysis result for a plot near a protected area."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    buffer_id: str = Field(default_factory=_new_uuid, description="Buffer record UUID")
    plot_id: str = Field(..., min_length=1, description="Production plot identifier")
    wdpa_id: int = Field(..., description="WDPA protected area ID")
    proximity_tier: BufferProximityTier = Field(..., description="Proximity classification")
    distance_meters: Decimal = Field(..., ge=Decimal("0"), description="Exact distance (m)")
    buffer_radius_km: Decimal = Field(..., ge=Decimal("0"), description="Configured buffer (km)")
    iucn_category: IUCNCategory = Field(..., description="PA IUCN category")
    encroachment_trend: EncroachmentTrend = Field(
        EncroachmentTrend.STABLE, description="Encroachment direction"
    )
    national_buffer_required: bool = Field(False, description="National law mandates buffer")
    national_buffer_km: Optional[Decimal] = Field(None, ge=Decimal("0"))
    compliant_with_national: bool = Field(True, description="Complies with national buffer")
    plots_in_buffer_count: int = Field(0, ge=0, description="Plot count in this buffer tier")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    detected_at: datetime = Field(default_factory=_utcnow, description="Detection timestamp")


class ProximityAlert(BaseModel):
    """High-risk proximity alert for a plot near a significant protected area."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    alert_id: str = Field(default_factory=_new_uuid, description="Alert UUID")
    plot_id: str = Field(..., min_length=1, description="Production plot identifier")
    wdpa_id: int = Field(..., description="WDPA protected area ID")
    protected_area_name: str = Field(..., description="Protected area name")
    iucn_category: IUCNCategory = Field(..., description="IUCN category")
    high_risk_designations: List[str] = Field(
        default_factory=list, description="UNESCO_WH, Ramsar, AZE, KBA"
    )
    distance_meters: Decimal = Field(..., ge=Decimal("0"), description="Distance (m)")
    proximity_risk_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"), description="Proximity risk 0-100"
    )
    alert_severity: AlertSeverity = Field(..., description="Alert severity")
    supply_chain_impact: Optional[Dict[str, Any]] = Field(None, description="Impact details")
    enhanced_dd_required: bool = Field(False, description="Enhanced due diligence needed")
    enhanced_dd_deadline: Optional[datetime] = Field(None, description="EDD response deadline")
    compound_risk_indigenous: bool = Field(False, description="Also near indigenous territory")
    deforestation_trend: Optional[str] = Field(None, description="Deforestation trend")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    created_at: datetime = Field(default_factory=_utcnow, description="Creation timestamp")


class PAComplianceRecord(BaseModel):
    """Compliance tracking record for a plot-protected area violation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    compliance_id: str = Field(default_factory=_new_uuid, description="Compliance record UUID")
    plot_id: str = Field(..., min_length=1, description="Production plot identifier")
    wdpa_id: int = Field(..., description="WDPA protected area ID")
    overlap_id: str = Field(..., description="Overlap record reference")
    compliance_status: PAComplianceStatus = Field(..., description="Current status")
    violation_type: Optional[ViolationType] = Field(None, description="Violation type")
    affected_area_hectares: Optional[Decimal] = Field(None, ge=Decimal("0"))
    detection_method: DetectionMethod = Field(
        DetectionMethod.WDPA_OVERLAP, description="How violation was detected"
    )
    remediation_plan: Optional[str] = Field(None, max_length=5000)
    remediation_deadline: Optional[datetime] = Field(None)
    exemption_permit: Optional[Dict[str, Any]] = Field(None)
    compliance_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"), description="Compliance score 0-100"
    )
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ConservationAssessment(BaseModel):
    """Conservation status assessment for a protected area."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    assessment_id: str = Field(default_factory=_new_uuid, description="Assessment UUID")
    wdpa_id: int = Field(..., description="WDPA protected area ID")
    threatened_species_count: int = Field(0, ge=0, description="CR/EN/VU species count")
    cites_species_count: int = Field(0, ge=0, description="CITES-listed species count")
    aze_trigger_present: bool = Field(False, description="AZE trigger species present")
    habitat_rarity_score: Decimal = Field(
        Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="Habitat rarity 0-100"
    )
    connectivity_index: Decimal = Field(
        Decimal("50"), ge=Decimal("0"), le=Decimal("100"), description="Connectivity 0-100"
    )
    biodiversity_score: Decimal = Field(
        Decimal("0"), ge=Decimal("0"), le=Decimal("100"), description="Composite biodiversity 0-100"
    )
    fragmentation_index: Optional[Decimal] = Field(None, ge=Decimal("0"), le=Decimal("100"))
    carbon_storage_tco2e: Optional[Decimal] = Field(None, ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    assessed_at: datetime = Field(default_factory=_utcnow)


class DesignationValidation(BaseModel):
    """Validation result for a protected area's legal designation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    validation_id: str = Field(default_factory=_new_uuid, description="Validation UUID")
    wdpa_id: int = Field(..., description="WDPA protected area ID")
    designation_level: DesignationLevel = Field(..., description="Designation tier")
    legal_status: PALegalStatus = Field(..., description="Legal status")
    governance_type: GovernanceType = Field(..., description="Governance type")
    designation_level_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"), description="Designation level score"
    )
    legal_status_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"), description="Legal status score"
    )
    management_effectiveness: Decimal = Field(
        Decimal("50"), ge=Decimal("0"), le=Decimal("100"), description="METT or default"
    )
    enforcement_assessment: Decimal = Field(
        Decimal("50"), ge=Decimal("0"), le=Decimal("100"), description="Enforcement score"
    )
    designation_strength: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"), description="Composite designation strength"
    )
    high_designation_alert: bool = Field(False, description="UNESCO/Ramsar/AZE trigger")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    validated_at: datetime = Field(default_factory=_utcnow)


class RiskScoreBreakdown(BaseModel):
    """Detailed breakdown of protected area risk score calculation."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    score_id: str = Field(default_factory=_new_uuid, description="Score UUID")
    plot_id: str = Field(..., description="Plot identifier")
    wdpa_id: int = Field(..., description="WDPA protected area ID")
    iucn_category_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    overlap_type_multiplier: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    designation_level_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    management_effectiveness_gap: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    country_enforcement_gap: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    weighted_iucn_component: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    weighted_designation_component: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    weighted_management_component: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    weighted_enforcement_component: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    total_risk_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    risk_level: RiskLevel = Field(..., description="Classified risk level")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    computed_at: datetime = Field(default_factory=_utcnow)


class BatchScreeningJob(BaseModel):
    """Batch overlap screening job status."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    job_id: str = Field(default_factory=_new_uuid, description="Job UUID")
    total_plots: int = Field(..., ge=1, description="Total plots to screen")
    processed_plots: int = Field(0, ge=0, description="Plots processed so far")
    overlaps_found: int = Field(0, ge=0, description="Overlaps detected")
    status: str = Field("pending", description="pending/running/completed/failed")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    processing_time_ms: Optional[Decimal] = Field(None, ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    created_at: datetime = Field(default_factory=_utcnow)


class ComplianceReport(BaseModel):
    """Generated compliance report metadata."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    report_id: str = Field(default_factory=_new_uuid, description="Report UUID")
    report_type: ReportType = Field(..., description="Report type")
    report_format: ReportFormat = Field(ReportFormat.PDF, description="Output format")
    language: ReportLanguage = Field(ReportLanguage.EN, description="Report language")
    title: str = Field(..., min_length=1, max_length=500, description="Report title")
    version: str = Field("1.0", description="Report version")
    total_plots_screened: int = Field(0, ge=0)
    overlaps_by_iucn: Dict[str, int] = Field(default_factory=dict)
    compliance_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100")
    )
    file_size_bytes: Optional[int] = Field(None, ge=0)
    file_path: Optional[str] = Field(None)
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    generated_at: datetime = Field(default_factory=_utcnow)


class IntegrationEvent(BaseModel):
    """Event published to the GreenLang event bus for cross-agent integration."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    event_id: str = Field(default_factory=_new_uuid, description="Event UUID")
    event_type: str = Field(..., description="Event type identifier")
    source_agent: str = Field("GL-EUDR-PAV-022", description="Source agent ID")
    target_agents: List[str] = Field(default_factory=list, description="Target agent IDs")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    published_at: datetime = Field(default_factory=_utcnow)


class AuditLogEntry(BaseModel):
    """Audit log entry for EUDR Article 31 compliance."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    entry_id: str = Field(default_factory=_new_uuid, description="Audit entry UUID")
    operation: str = Field(..., min_length=1, max_length=100, description="Operation performed")
    entity_type: str = Field(..., min_length=1, max_length=100)
    entity_id: str = Field(..., description="Affected entity identifier")
    actor: str = Field("system", description="Actor identifier")
    details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    created_at: datetime = Field(default_factory=_utcnow)


class ProtectedAreaVersion(BaseModel):
    """WDPA data version tracking record."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    version_id: str = Field(default_factory=_new_uuid, description="Version UUID")
    wdpa_version: str = Field(..., description="WDPA release identifier (e.g. Oct2025)")
    release_date: Optional[datetime] = Field(None, description="WDPA release date")
    total_records: int = Field(0, ge=0, description="Total protected area records")
    records_added: int = Field(0, ge=0)
    records_modified: int = Field(0, ge=0)
    records_removed: int = Field(0, ge=0)
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    ingested_at: datetime = Field(default_factory=_utcnow)


class BufferZoneExemption(BaseModel):
    """Buffer zone exemption for plots with valid legal permits."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    exemption_id: str = Field(default_factory=_new_uuid, description="Exemption UUID")
    plot_id: str = Field(..., min_length=1)
    wdpa_id: int = Field(...)
    permit_type: str = Field(..., max_length=200)
    issuing_authority: str = Field(..., max_length=500)
    permit_number: str = Field(..., max_length=200)
    effective_date: datetime = Field(...)
    expiry_date: Optional[datetime] = Field(None)
    conditions: Optional[str] = Field(None, max_length=5000)
    document_hash: Optional[str] = Field(None, description="Document SHA-256")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 hash")
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Request Models (7)
# ---------------------------------------------------------------------------

class CheckOverlapRequest(BaseModel):
    """Request to check overlap between a single plot and protected areas."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plot_id: str = Field(..., min_length=1, description="Plot identifier")
    latitude: Decimal = Field(..., ge=Decimal("-90"), le=Decimal("90"))
    longitude: Decimal = Field(..., ge=Decimal("-180"), le=Decimal("180"))
    geometry_wkt: Optional[str] = Field(None, description="Plot polygon WKT")
    buffer_radii_km: Optional[List[Decimal]] = Field(None, description="Buffer radii to check")
    include_conservation: bool = Field(False, description="Include conservation assessment")
    request_id: Optional[str] = Field(None)


class BatchOverlapRequest(BaseModel):
    """Request for batch overlap screening of multiple plots."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plots: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=10000,
        description="List of {plot_id, latitude, longitude, geometry_wkt}"
    )
    buffer_radii_km: Optional[List[Decimal]] = Field(None)
    request_id: Optional[str] = Field(None)


class BufferAnalysisRequest(BaseModel):
    """Request for buffer zone analysis around a protected area."""

    model_config = ConfigDict(str_strip_whitespace=True)

    wdpa_id: int = Field(..., description="WDPA protected area ID")
    radii_km: List[Decimal] = Field(
        default_factory=lambda: [Decimal("1"), Decimal("5"), Decimal("10"), Decimal("25"), Decimal("50")]
    )
    include_plot_density: bool = Field(True)
    request_id: Optional[str] = Field(None)


class ValidateDesignationRequest(BaseModel):
    """Request to validate a protected area's legal designation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    wdpa_id: int = Field(..., description="WDPA protected area ID")
    country_enforcement_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"), description="From EUDR-016"
    )
    request_id: Optional[str] = Field(None)


class ProximityAlertRequest(BaseModel):
    """Request to check for high-risk proximity alerts."""

    model_config = ConfigDict(str_strip_whitespace=True)

    plot_id: str = Field(..., min_length=1)
    latitude: Decimal = Field(..., ge=Decimal("-90"), le=Decimal("90"))
    longitude: Decimal = Field(..., ge=Decimal("-180"), le=Decimal("180"))
    radius_km: Decimal = Field(Decimal("25"), ge=Decimal("1"), le=Decimal("50"))
    commodities: Optional[List[EUDRCommodity]] = Field(None)
    request_id: Optional[str] = Field(None)


class ComplianceReportRequest(BaseModel):
    """Request to generate a compliance report."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report_type: ReportType = Field(ReportType.FULL_COMPLIANCE)
    report_format: ReportFormat = Field(ReportFormat.PDF)
    language: ReportLanguage = Field(ReportLanguage.EN)
    operator_id: Optional[str] = Field(None)
    supplier_ids: Optional[List[str]] = Field(None)
    country_codes: Optional[List[str]] = Field(None)
    commodity_filter: Optional[List[EUDRCommodity]] = Field(None)
    request_id: Optional[str] = Field(None)


class ConservationAssessmentRequest(BaseModel):
    """Request to assess conservation status of a protected area."""

    model_config = ConfigDict(str_strip_whitespace=True)

    wdpa_id: int = Field(..., description="WDPA protected area ID")
    include_species_data: bool = Field(True)
    include_fragmentation: bool = Field(True)
    request_id: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Response Models (7)
# ---------------------------------------------------------------------------

class CheckOverlapResponse(BaseModel):
    """Response from single-plot overlap check."""

    model_config = ConfigDict(str_strip_whitespace=True)

    overlaps: List[ProtectedAreaOverlap] = Field(default_factory=list)
    buffer_zones: List[BufferZoneResult] = Field(default_factory=list)
    highest_risk_level: RiskLevel = Field(RiskLevel.CLEAR)
    total_overlaps: int = Field(0, ge=0)
    processing_time_ms: Decimal = Field(Decimal("0"), ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)


class BatchOverlapResponse(BaseModel):
    """Response from batch overlap screening."""

    model_config = ConfigDict(str_strip_whitespace=True)

    job: BatchScreeningJob = Field(...)
    results: List[CheckOverlapResponse] = Field(default_factory=list)
    total_plots_screened: int = Field(0, ge=0)
    total_overlaps_found: int = Field(0, ge=0)
    processing_time_ms: Decimal = Field(Decimal("0"), ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)


class BufferAnalysisResponse(BaseModel):
    """Response from buffer zone analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    buffer_zones: List[BufferZoneResult] = Field(default_factory=list)
    plots_by_tier: Dict[str, int] = Field(default_factory=dict)
    total_plots_in_buffers: int = Field(0, ge=0)
    processing_time_ms: Decimal = Field(Decimal("0"), ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)


class ValidateDesignationResponse(BaseModel):
    """Response from designation validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    validation: DesignationValidation = Field(...)
    processing_time_ms: Decimal = Field(Decimal("0"), ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)


class ProximityAlertResponse(BaseModel):
    """Response from proximity alert check."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alerts: List[ProximityAlert] = Field(default_factory=list)
    total_alerts: int = Field(0, ge=0)
    highest_severity: AlertSeverity = Field(AlertSeverity.STANDARD)
    processing_time_ms: Decimal = Field(Decimal("0"), ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)


class ComplianceReportResponse(BaseModel):
    """Response from compliance report generation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report: ComplianceReport = Field(...)
    processing_time_ms: Decimal = Field(Decimal("0"), ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)


class ConservationAssessmentResponse(BaseModel):
    """Response from conservation status assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessment: ConservationAssessment = Field(...)
    processing_time_ms: Decimal = Field(Decimal("0"), ge=Decimal("0"))
    provenance_hash: Optional[str] = Field(None)
    request_id: Optional[str] = Field(None)
