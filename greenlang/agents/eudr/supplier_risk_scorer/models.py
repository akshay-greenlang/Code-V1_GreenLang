# -*- coding: utf-8 -*-
"""
Supplier Risk Scorer Data Models - AGENT-EUDR-017

Pydantic v2 data models for the Supplier Risk Scorer Agent covering
composite supplier risk scoring with 8 weighted factors (geographic
sourcing 20%, compliance history 15%, documentation quality 15%,
certification status 15%, traceability completeness 10%, financial
stability 10%, environmental performance 10%, social compliance 5%);
due diligence tracking with non-conformance categorization (minor,
major, critical) and corrective action management; documentation
analysis with EUDR-required document validation (geolocation, DDS
reference, product description, quantity declaration, harvest date,
compliance declaration); certification validation supporting 8 major
schemes (FSC, PEFC, RSPO, Rainforest Alliance, UTZ, Organic, Fair
Trade, ISCC) with chain-of-custody verification; geographic sourcing
analysis with country risk integration (AGENT-EUDR-016), deforestation
overlay, and concentration risk calculation; supplier network analysis
with multi-tier risk propagation, sub-supplier evaluation, and circular
dependency detection; continuous monitoring with configurable frequency
and multi-severity alerting (info, warning, high, critical); and risk
report generation in multiple formats (PDF, JSON, HTML, Excel) with
DDS package assembly and audit trail documentation.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all supplier risk evaluation operations per
EU 2023/1115 Articles 4, 8, 9, 10, 11, and 31.

Enumerations (18):
    - RiskLevel, SupplierType, CommodityType, CertificationScheme,
      CertificationStatus, DocumentType, DocumentStatus, DDLevel,
      DDStatus, NonConformanceType, AlertSeverity, AlertType,
      ReportType, ReportFormat, MonitoringFrequency, TrendDirection,
      DDActivityType, NonConformanceStatus

Core Models (16):
    - SupplierRiskAssessment, DueDiligenceRecord, DocumentationProfile,
      CertificationRecord, GeographicSourcingProfile, SupplierNetwork,
      MonitoringConfig, SupplierAlert, RiskReport, SupplierProfile,
      FactorScore, AuditLogEntry, DDActivity, NonConformance,
      CorrectiveActionPlan, SupplierDocument

Request Models (15):
    - AssessSupplierRequest, TrackDueDiligenceRequest,
      AnalyzeDocumentationRequest, ValidateCertificationRequest,
      AnalyzeGeographicSourcingRequest, AnalyzeNetworkRequest,
      ConfigureMonitoringRequest, GenerateAlertRequest,
      GenerateReportRequest, GetSupplierProfileRequest,
      CompareSupplierRequest, GetTrendRequest, BatchAssessmentRequest,
      SearchSupplierRequest, HealthRequest

Response Models (15):
    - SupplierRiskResponse, DueDiligenceResponse, DocumentationResponse,
      CertificationResponse, GeographicSourcingResponse, NetworkResponse,
      MonitoringResponse, AlertResponse, ReportResponse, ProfileResponse,
      ComparisonResponse, TrendResponse, BatchResponse, SearchResponse,
      HealthResponse

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-001 Supply Chain Mapping Master.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Cross-agent commodity import (graceful fallback)
# ---------------------------------------------------------------------------

from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import AlertSeverity, ReportFormat, RiskLevel

try:
    from greenlang.agents.data.eudr_traceability.models import (
        EUDRCommodity as _ExternalEUDRCommodity,
    )
except ImportError:
    _ExternalEUDRCommodity = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: Maximum risk score value.
MAX_RISK_SCORE: int = 100

#: Minimum risk score value.
MIN_RISK_SCORE: int = 0

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 500

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Default EUDR commodities (EU 2023/1115 Article 1).
SUPPORTED_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

#: Supported certification schemes.
SUPPORTED_SCHEMES: List[str] = [
    "FSC", "PEFC", "RSPO", "RAINFOREST_ALLIANCE",
    "UTZ", "ORGANIC", "FAIR_TRADE", "ISCC",
]

#: Supported report output formats.
SUPPORTED_OUTPUT_FORMATS: List[str] = [
    "pdf", "json", "html", "excel", "csv",
]

#: Supported report languages.
SUPPORTED_REPORT_LANGUAGES: List[str] = [
    "en", "fr", "de", "es", "pt",
]

#: Default factor weights for composite risk scoring.
DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "geographic_sourcing": 0.20,
    "compliance_history": 0.15,
    "documentation_quality": 0.15,
    "certification_status": 0.15,
    "traceability_completeness": 0.10,
    "financial_stability": 0.10,
    "environmental_performance": 0.10,
    "social_compliance": 0.05,
}

# =============================================================================
# Enumerations
# =============================================================================

class SupplierType(str, Enum):
    """Supplier role classification in the EUDR supply chain.

    PRODUCER: Primary producer of EUDR commodities (farms, plantations).
    TRADER: Trader or aggregator of EUDR commodities.
    PROCESSOR: Processor or manufacturer of EUDR-derived products.
    EXPORTER: Exporter from production country.
    IMPORTER: Importer into EU market.
    BROKER: Broker or intermediary without physical handling.
    COOPERATIVE: Producer cooperative or farmer association.
    """

    PRODUCER = "producer"
    TRADER = "trader"
    PROCESSOR = "processor"
    EXPORTER = "exporter"
    IMPORTER = "importer"
    BROKER = "broker"
    COOPERATIVE = "cooperative"

class CommodityType(str, Enum):
    """EUDR-regulated commodities per Article 1 and Annex I.

    Seven commodity groups subject to the EUDR deforestation-free
    requirement. Each commodity has specific HS code mappings,
    derived product definitions, and production characteristics
    that affect risk scoring.
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

class CertificationScheme(str, Enum):
    """Supported third-party certification schemes for EUDR compliance.

    FSC: Forest Stewardship Council (wood, paper).
    PEFC: Programme for the Endorsement of Forest Certification (wood, paper).
    RSPO: Roundtable on Sustainable Palm Oil (palm oil).
    RAINFOREST_ALLIANCE: Rainforest Alliance (coffee, cocoa, tea).
    UTZ: UTZ Certified (coffee, cocoa, tea, hazelnuts).
    ORGANIC: Organic certification (various commodities).
    FAIR_TRADE: Fair Trade certification (coffee, cocoa, bananas).
    ISCC: International Sustainability and Carbon Certification (palm oil, soya).
    """

    FSC = "FSC"
    PEFC = "PEFC"
    RSPO = "RSPO"
    RAINFOREST_ALLIANCE = "RAINFOREST_ALLIANCE"
    UTZ = "UTZ"
    ORGANIC = "ORGANIC"
    FAIR_TRADE = "FAIR_TRADE"
    ISCC = "ISCC"

class CertificationStatus(str, Enum):
    """Certification validity status.

    VALID: Certification is valid and within validity period.
    EXPIRED: Certification has expired beyond validity period.
    SUSPENDED: Certification is suspended by certification body.
    REVOKED: Certification has been permanently revoked.
    PENDING: Certification application is pending approval.
    """

    VALID = "valid"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    PENDING = "pending"

class DocumentType(str, Enum):
    """EUDR-required document types per Articles 8-9.

    GEOLOCATION: Geolocation coordinates of production plots.
    DDS_REFERENCE: Due Diligence Statement reference number.
    PRODUCT_DESCRIPTION: Product description and HS/CN code.
    QUANTITY_DECLARATION: Quantity declaration (mass, volume, units).
    HARVEST_DATE: Harvest or production date declaration.
    COMPLIANCE_DECLARATION: EUDR compliance declaration.
    CERTIFICATE: Third-party certification certificate.
    TRADE_LICENSE: Trade or export license.
    PHYTOSANITARY: Phytosanitary certificate.
    OTHER: Other supporting documentation.
    """

    GEOLOCATION = "geolocation"
    DDS_REFERENCE = "dds_reference"
    PRODUCT_DESCRIPTION = "product_description"
    QUANTITY_DECLARATION = "quantity_declaration"
    HARVEST_DATE = "harvest_date"
    COMPLIANCE_DECLARATION = "compliance_declaration"
    CERTIFICATE = "certificate"
    TRADE_LICENSE = "trade_license"
    PHYTOSANITARY = "phytosanitary"
    OTHER = "other"

class DocumentStatus(str, Enum):
    """Document validation status.

    SUBMITTED: Document has been submitted for validation.
    VERIFIED: Document has been verified and accepted.
    REJECTED: Document has been rejected due to validation failure.
    EXPIRED: Document has expired beyond validity period.
    MISSING: Required document is missing.
    """

    SUBMITTED = "submitted"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    MISSING = "missing"

class DDLevel(str, Enum):
    """Due diligence level classification per EUDR Articles 10-13.

    SIMPLIFIED: Reduced due diligence for low-risk suppliers.
    STANDARD: Standard due diligence for medium-risk suppliers.
    ENHANCED: Enhanced due diligence for high/critical-risk suppliers.
    """

    SIMPLIFIED = "simplified"
    STANDARD = "standard"
    ENHANCED = "enhanced"

class DDStatus(str, Enum):
    """Due diligence tracking status.

    NOT_STARTED: Due diligence has not been started.
    IN_PROGRESS: Due diligence is in progress.
    COMPLETED: Due diligence has been completed.
    OVERDUE: Due diligence is overdue.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"

class NonConformanceType(str, Enum):
    """Non-conformance severity classification.

    MINOR: Minor non-conformance, low impact, corrective action recommended.
    MAJOR: Major non-conformance, significant impact, corrective action required.
    CRITICAL: Critical non-conformance, severe impact, immediate action required.
    """

    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

class AlertType(str, Enum):
    """Alert type classification.

    RISK_THRESHOLD: Supplier risk score exceeded threshold.
    CERTIFICATION_EXPIRY: Certification expiring or expired.
    DOCUMENT_MISSING: Required document is missing.
    DD_OVERDUE: Due diligence is overdue.
    SANCTION_HIT: Supplier matched sanction list.
    BEHAVIOR_CHANGE: Significant behavior change detected.
    """

    RISK_THRESHOLD = "risk_threshold"
    CERTIFICATION_EXPIRY = "certification_expiry"
    DOCUMENT_MISSING = "document_missing"
    DD_OVERDUE = "dd_overdue"
    SANCTION_HIT = "sanction_hit"
    BEHAVIOR_CHANGE = "behavior_change"

class ReportType(str, Enum):
    """Risk report type classification.

    INDIVIDUAL: Individual supplier risk assessment report.
    PORTFOLIO: Portfolio-level supplier risk aggregation report.
    COMPARATIVE: Comparative analysis of multiple suppliers.
    TREND: Trend analysis over time for supplier or portfolio.
    AUDIT_PACKAGE: Audit package with all supporting documentation.
    EXECUTIVE: Executive summary report for senior management.
    """

    INDIVIDUAL = "individual"
    PORTFOLIO = "portfolio"
    COMPARATIVE = "comparative"
    TREND = "trend"
    AUDIT_PACKAGE = "audit_package"
    EXECUTIVE = "executive"

class MonitoringFrequency(str, Enum):
    """Monitoring frequency for continuous supplier monitoring.

    DAILY: Daily monitoring checks.
    WEEKLY: Weekly monitoring checks.
    BIWEEKLY: Biweekly (every 2 weeks) monitoring checks.
    MONTHLY: Monthly monitoring checks.
    QUARTERLY: Quarterly monitoring checks.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class TrendDirection(str, Enum):
    """Direction of risk score trend over time.

    IMPROVING: Risk score decreasing (positive trend, lower risk).
    STABLE: Risk score stable within threshold over analysis window.
    DETERIORATING: Risk score increasing (negative trend, higher risk).
    """

    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"

class DDActivityType(str, Enum):
    """Due diligence activity type classification.

    AUDIT: On-site or remote audit by qualified auditor.
    SITE_VISIT: Physical inspection of production/processing sites.
    DOCUMENT_REVIEW: Review of EUDR-required documentation package.
    QUESTIONNAIRE: Supplier self-assessment questionnaire response.
    SCREENING: Screening against sanctions lists, deforestation databases.
    VERIFICATION: Third-party verification of claims or certifications.
    TRAINING: Supplier capacity building or training session.
    INTERVIEW: Stakeholder or worker interviews.
    """

    AUDIT = "audit"
    SITE_VISIT = "site_visit"
    DOCUMENT_REVIEW = "document_review"
    QUESTIONNAIRE = "questionnaire"
    SCREENING = "screening"
    VERIFICATION = "verification"
    TRAINING = "training"
    INTERVIEW = "interview"

class NonConformanceStatus(str, Enum):
    """Non-conformance status classification.

    OPEN: Non-conformance identified, action pending.
    IN_PROGRESS: Corrective action in progress.
    RESOLVED: Corrective action completed, pending verification.
    VERIFIED: Corrective action verified and closed.
    OVERDUE: Corrective action deadline exceeded.
    """

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    VERIFIED = "verified"
    OVERDUE = "overdue"

# =============================================================================
# Core Models
# =============================================================================

class FactorScore(GreenLangBase):
    """Individual risk factor score with weighting and normalization.

    Attributes:
        factor_name: Risk factor name (e.g., geographic_sourcing).
        raw_score: Raw score before normalization (0-100).
        normalized_score: Normalized score (0-100).
        weight: Factor weight in composite score (0.0-1.0, sum=1.0).
        weighted_score: Weighted score contribution to composite (0-100).
        data_sources: List of data sources used for scoring.
        confidence: Confidence in factor score (0.0-1.0).
        last_updated: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    factor_name: str = Field(..., min_length=1, max_length=100)
    raw_score: Decimal = Field(..., ge=0, le=100)
    normalized_score: Decimal = Field(..., ge=0, le=100)
    weight: Decimal = Field(..., ge=0, le=1)
    weighted_score: Decimal = Field(..., ge=0, le=100)
    data_sources: List[str] = Field(default_factory=list)
    confidence: Decimal = Field(..., ge=0, le=1)
    last_updated: datetime = Field(default_factory=utcnow)

class SupplierProfile(GreenLangBase):
    """Supplier profile with basic information.

    Attributes:
        supplier_id: Unique supplier identifier.
        name: Supplier legal name.
        type: Supplier type classification.
        country: ISO 3166-1 alpha-2 country code.
        commodities: List of commodities supplied.
        registration_date: Supplier registration date.
        tax_id: Tax identification number (optional).
        address: Physical address (optional).
        contact_email: Contact email (optional).
        contact_phone: Contact phone (optional).
        website: Website URL (optional).
        active: Supplier active status.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    type: SupplierType
    country: str = Field(..., min_length=2, max_length=2)
    commodities: List[CommodityType]
    registration_date: datetime
    tax_id: Optional[str] = Field(None, max_length=50)
    address: Optional[str] = Field(None, max_length=500)
    contact_email: Optional[str] = Field(None, max_length=255)
    contact_phone: Optional[str] = Field(None, max_length=50)
    website: Optional[str] = Field(None, max_length=255)
    active: bool = True

class SupplierRiskAssessment(GreenLangBase):
    """Composite supplier risk assessment with 8-factor scoring.

    Attributes:
        assessment_id: Unique assessment identifier.
        supplier_id: Supplier identifier.
        risk_score: Composite risk score (0-100).
        risk_level: Risk level classification.
        factor_scores: Individual factor scores (8 factors).
        confidence: Overall confidence in assessment (0.0-1.0).
        trend: Risk trend direction (increasing, stable, decreasing).
        assessed_at: Assessment timestamp.
        assessed_by: User or system identifier.
        version: Assessment version number.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    assessment_id: str = Field(
        default_factory=lambda: f"sra-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    risk_score: Decimal = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    factor_scores: List[FactorScore]
    confidence: Decimal = Field(..., ge=0, le=1)
    trend: Optional[str] = Field(None, max_length=50)
    assessed_at: datetime = Field(default_factory=utcnow)
    assessed_by: str = Field(..., min_length=1, max_length=100)
    version: int = Field(default=1, ge=1)

    @field_validator("factor_scores")
    @classmethod
    def validate_factor_scores(cls, v: List[FactorScore]) -> List[FactorScore]:
        """Validate that factor scores contain all 8 required factors."""
        if len(v) != 8:
            raise ValueError(f"Expected 8 factor scores, got {len(v)}")
        return v

class DueDiligenceRecord(GreenLangBase):
    """Due diligence tracking record with non-conformances and corrective actions.

    Attributes:
        dd_id: Unique due diligence record identifier.
        supplier_id: Supplier identifier.
        dd_level: Due diligence level (simplified, standard, enhanced).
        status: Due diligence status.
        activities: List of due diligence activities performed.
        non_conformances: List of non-conformances detected.
        corrective_actions: List of corrective actions taken.
        start_date: Due diligence start date.
        completion_date: Due diligence completion date (if completed).
        next_review_date: Next review date.
        auditor: Auditor identifier.
        notes: Additional notes.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    dd_id: str = Field(
        default_factory=lambda: f"dd-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    dd_level: DDLevel
    status: DDStatus
    activities: List[str] = Field(default_factory=list)
    non_conformances: List[Dict[str, Any]] = Field(default_factory=list)
    corrective_actions: List[Dict[str, Any]] = Field(default_factory=list)
    start_date: datetime
    completion_date: Optional[datetime] = None
    next_review_date: Optional[datetime] = None
    auditor: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=2000)

class DocumentationProfile(GreenLangBase):
    """Supplier documentation profile with completeness and quality scoring.

    Attributes:
        profile_id: Unique profile identifier.
        supplier_id: Supplier identifier.
        documents: List of documents with status.
        completeness_score: Documentation completeness score (0.0-1.0).
        gaps: List of missing or incomplete documents.
        quality_score: Documentation quality score (0-100).
        expiring_soon: List of documents expiring within warning period.
        last_updated: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    profile_id: str = Field(
        default_factory=lambda: f"doc-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    completeness_score: Decimal = Field(..., ge=0, le=1)
    gaps: List[str] = Field(default_factory=list)
    quality_score: Decimal = Field(..., ge=0, le=100)
    expiring_soon: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=utcnow)

class CertificationRecord(GreenLangBase):
    """Third-party certification record with chain-of-custody validation.

    Attributes:
        cert_id: Unique certification identifier.
        supplier_id: Supplier identifier.
        scheme: Certification scheme.
        status: Certification status.
        certificate_number: Certificate number.
        scope: Certification scope (commodities, locations).
        issue_date: Certificate issue date.
        expiry_date: Certificate expiry date.
        chain_of_custody: Chain-of-custody validated (bool).
        multi_site: Multi-site certification (bool).
        certification_body: Certification body name.
        verification_url: Verification URL (optional).
        last_verified: Last verification timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    cert_id: str = Field(
        default_factory=lambda: f"cert-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    scheme: CertificationScheme
    status: CertificationStatus
    certificate_number: str = Field(..., min_length=1, max_length=100)
    scope: List[str] = Field(default_factory=list)
    issue_date: datetime
    expiry_date: datetime
    chain_of_custody: bool = False
    multi_site: bool = False
    certification_body: str = Field(..., min_length=1, max_length=255)
    verification_url: Optional[str] = Field(None, max_length=500)
    last_verified: datetime = Field(default_factory=utcnow)

class GeographicSourcingProfile(GreenLangBase):
    """Geographic sourcing profile with country risk integration.

    Attributes:
        profile_id: Unique profile identifier.
        supplier_id: Supplier identifier.
        sourcing_locations: List of sourcing locations with country codes.
        risk_zones: List of high-risk zones identified.
        concentration_index: Geographic concentration index (HHI, 0.0-1.0).
        deforestation_overlay: Deforestation overlay detected (bool).
        protected_area_overlap: Protected area overlap detected (bool).
        indigenous_territory_overlap: Indigenous territory overlap detected (bool).
        country_risk_scores: Country risk scores from AGENT-EUDR-016.
        last_updated: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    profile_id: str = Field(
        default_factory=lambda: f"geo-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    sourcing_locations: List[Dict[str, Any]] = Field(default_factory=list)
    risk_zones: List[Dict[str, Any]] = Field(default_factory=list)
    concentration_index: Decimal = Field(..., ge=0, le=1)
    deforestation_overlay: bool = False
    protected_area_overlap: bool = False
    indigenous_territory_overlap: bool = False
    country_risk_scores: Dict[str, Decimal] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=utcnow)

class SupplierNetwork(GreenLangBase):
    """Supplier network graph with multi-tier risk propagation.

    Attributes:
        network_id: Unique network identifier.
        supplier_id: Primary supplier identifier.
        sub_suppliers: List of sub-supplier identifiers.
        intermediaries: List of intermediary identifiers.
        depth: Network depth (number of tiers).
        risk_propagation_score: Aggregated risk propagation score (0-100).
        circular_dependencies: Circular dependencies detected (bool).
        relationship_strengths: Relationship strength scores.
        last_updated: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    network_id: str = Field(
        default_factory=lambda: f"net-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    sub_suppliers: List[str] = Field(default_factory=list)
    intermediaries: List[str] = Field(default_factory=list)
    depth: int = Field(..., ge=0, le=10)
    risk_propagation_score: Decimal = Field(..., ge=0, le=100)
    circular_dependencies: bool = False
    relationship_strengths: Dict[str, Decimal] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=utcnow)

class MonitoringConfig(GreenLangBase):
    """Continuous monitoring configuration for supplier.

    Attributes:
        config_id: Unique configuration identifier.
        supplier_id: Supplier identifier.
        frequency: Monitoring frequency.
        alert_thresholds: Alert thresholds by severity.
        watchlist_status: Watchlist status (bool).
        enabled: Monitoring enabled (bool).
        last_check: Last monitoring check timestamp.
        next_check: Next monitoring check timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    config_id: str = Field(
        default_factory=lambda: f"mon-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    frequency: MonitoringFrequency
    alert_thresholds: Dict[str, int] = Field(default_factory=dict)
    watchlist_status: bool = False
    enabled: bool = True
    last_check: Optional[datetime] = None
    next_check: Optional[datetime] = None

class SupplierAlert(GreenLangBase):
    """Supplier risk alert with severity classification.

    Attributes:
        alert_id: Unique alert identifier.
        supplier_id: Supplier identifier.
        alert_type: Alert type classification.
        severity: Alert severity.
        message: Alert message.
        triggered_at: Alert trigger timestamp.
        acknowledged: Alert acknowledged (bool).
        acknowledged_at: Acknowledgement timestamp (if acknowledged).
        acknowledged_by: User identifier (if acknowledged).
        resolved: Alert resolved (bool).
        resolved_at: Resolution timestamp (if resolved).
        resolved_by: User identifier (if resolved).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    alert_id: str = Field(
        default_factory=lambda: f"alert-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    alert_type: AlertType
    severity: AlertSeverity
    message: str = Field(..., min_length=1, max_length=1000)
    triggered_at: datetime = Field(default_factory=utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = Field(None, max_length=100)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = Field(None, max_length=100)

class RiskReport(GreenLangBase):
    """Risk report generation record with format and content tracking.

    Attributes:
        report_id: Unique report identifier.
        report_type: Report type classification.
        format: Report output format.
        supplier_ids: List of supplier IDs included in report.
        content_hash: SHA-256 hash of report content.
        file_path: File path or URL (optional).
        file_size_bytes: File size in bytes.
        generated_at: Report generation timestamp.
        generated_by: User or system identifier.
        language: Report language code.
        retention_until: Retention expiry date.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    report_id: str = Field(
        default_factory=lambda: f"rpt-{uuid.uuid4().hex[:12]}"
    )
    report_type: ReportType
    format: ReportFormat
    supplier_ids: List[str] = Field(default_factory=list)
    content_hash: str = Field(..., min_length=64, max_length=64)
    file_path: Optional[str] = Field(None, max_length=500)
    file_size_bytes: int = Field(..., ge=0)
    generated_at: datetime = Field(default_factory=utcnow)
    generated_by: str = Field(..., min_length=1, max_length=100)
    language: str = Field(default="en", min_length=2, max_length=2)
    retention_until: datetime

class AuditLogEntry(GreenLangBase):
    """Audit log entry for supplier risk assessment operations.

    Attributes:
        log_id: Unique log entry identifier.
        entity_type: Entity type (e.g., supplier_assessment).
        entity_id: Entity identifier.
        action: Action performed (e.g., assess, track, analyze).
        actor: User or system identifier.
        timestamp: Action timestamp.
        provenance_hash: SHA-256 provenance hash.
        details: Additional details (optional).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    log_id: str = Field(
        default_factory=lambda: f"log-{uuid.uuid4().hex[:12]}"
    )
    entity_type: str = Field(..., min_length=1, max_length=100)
    entity_id: str = Field(..., min_length=1, max_length=100)
    action: str = Field(..., min_length=1, max_length=100)
    actor: str = Field(..., min_length=1, max_length=100)
    timestamp: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(..., min_length=64, max_length=64)
    details: Optional[Dict[str, Any]] = None

class DDActivity(GreenLangBase):
    """Due diligence activity record.

    Attributes:
        activity_id: Unique activity identifier.
        supplier_id: Supplier identifier.
        activity_type: Activity type classification.
        activity_date: Activity execution date.
        conducted_by: Person or entity who conducted the activity.
        findings: Activity findings summary.
        documents_reviewed: List of document identifiers reviewed.
        sites_visited: List of site locations visited.
        stakeholders_interviewed: List of stakeholders interviewed.
        cost_usd: Activity cost in USD.
        duration_hours: Activity duration in hours.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    activity_id: str = Field(..., min_length=1, max_length=100)
    supplier_id: str = Field(..., min_length=1, max_length=100)
    activity_type: DDActivityType
    activity_date: datetime
    conducted_by: str = Field(..., min_length=1, max_length=100)
    findings: str = Field(default="", max_length=2000)
    documents_reviewed: List[str] = Field(default_factory=list)
    sites_visited: List[str] = Field(default_factory=list)
    stakeholders_interviewed: List[str] = Field(default_factory=list)
    cost_usd: float = Field(default=0.0, ge=0)
    duration_hours: float = Field(default=0.0, ge=0)

class NonConformance(GreenLangBase):
    """Non-conformance finding record.

    Attributes:
        nc_id: Unique non-conformance identifier.
        record_id: Parent due diligence record identifier.
        supplier_id: Supplier identifier.
        nc_type: Non-conformance type classification.
        description: Non-conformance description.
        severity_score: Severity score (0-100).
        detected_date: Detection date.
        status: Non-conformance status.
        resolved_date: Resolution date (if resolved).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    nc_id: str = Field(..., min_length=1, max_length=100)
    record_id: str = Field(..., min_length=1, max_length=100)
    supplier_id: str = Field(..., min_length=1, max_length=100)
    nc_type: NonConformanceType
    description: str = Field(..., min_length=1, max_length=2000)
    severity_score: Decimal = Field(..., ge=0, le=100)
    detected_date: datetime
    status: NonConformanceStatus
    resolved_date: Optional[datetime] = None

class CorrectiveActionPlan(GreenLangBase):
    """Corrective action plan for non-conformance.

    Attributes:
        cap_id: Unique corrective action plan identifier.
        nc_id: Non-conformance identifier.
        action_description: Description of corrective action.
        responsible_party: Person or entity responsible.
        deadline: Completion deadline.
        resources_required: Required resources description.
        created_date: Plan creation date.
        status: Action plan status.
        completed_date: Completion date (if completed).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    cap_id: str = Field(..., min_length=1, max_length=100)
    nc_id: str = Field(..., min_length=1, max_length=100)
    action_description: str = Field(..., min_length=1, max_length=2000)
    responsible_party: str = Field(..., min_length=1, max_length=100)
    deadline: datetime
    resources_required: str = Field(default="", max_length=1000)
    created_date: datetime
    status: NonConformanceStatus
    completed_date: Optional[datetime] = None

class SupplierDocument(GreenLangBase):
    """Supplier document record.

    Attributes:
        document_id: Unique document identifier.
        supplier_id: Supplier identifier.
        document_type: Document type classification.
        file_name: Original file name.
        file_format: File format (pdf, xlsx, jpg, etc.).
        file_size_bytes: File size in bytes.
        content_hash: SHA-256 content hash.
        submission_date: Document submission date.
        expiry_date: Document expiry date (if applicable).
        version: Document version number.
        status: Document validation status.
        language_code: Document language code (ISO 639-1).
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    document_id: str = Field(..., min_length=1, max_length=100)
    supplier_id: str = Field(..., min_length=1, max_length=100)
    document_type: DocumentType
    file_name: str = Field(..., min_length=1, max_length=255)
    file_format: str = Field(..., min_length=1, max_length=50)
    file_size_bytes: int = Field(..., ge=0)
    content_hash: str = Field(..., min_length=64, max_length=64)
    submission_date: datetime
    expiry_date: Optional[datetime] = None
    version: int = Field(default=1, ge=1)
    status: DocumentStatus
    language_code: str = Field(default="en", min_length=2, max_length=2)

# =============================================================================
# Request Models
# =============================================================================

class AssessSupplierRequest(GreenLangBase):
    """Request to assess supplier risk score."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    custom_weights: Optional[Dict[str, float]] = None
    include_trend: bool = True
    assessed_by: str = Field(..., min_length=1, max_length=100)

class TrackDueDiligenceRequest(GreenLangBase):
    """Request to track due diligence activities."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    dd_level: DDLevel
    activities: List[str] = Field(default_factory=list)
    auditor: Optional[str] = Field(None, max_length=100)

class AnalyzeDocumentationRequest(GreenLangBase):
    """Request to analyze supplier documentation."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    include_quality_score: bool = True
    include_gap_detection: bool = True

class ValidateCertificationRequest(GreenLangBase):
    """Request to validate supplier certifications."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    scheme: Optional[CertificationScheme] = None
    verify_chain_of_custody: bool = True

class AnalyzeGeographicSourcingRequest(GreenLangBase):
    """Request to analyze geographic sourcing profile."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    include_country_risk: bool = True
    include_deforestation_overlay: bool = True

class AnalyzeNetworkRequest(GreenLangBase):
    """Request to analyze supplier network."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    max_depth: int = Field(default=3, ge=1, le=10)
    include_risk_propagation: bool = True

class ConfigureMonitoringRequest(GreenLangBase):
    """Request to configure supplier monitoring."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    frequency: MonitoringFrequency
    alert_thresholds: Optional[Dict[str, int]] = None
    watchlist_status: bool = False

class GenerateAlertRequest(GreenLangBase):
    """Request to generate supplier alert."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    alert_type: AlertType
    severity: AlertSeverity
    message: str = Field(..., min_length=1, max_length=1000)

class GenerateReportRequest(GreenLangBase):
    """Request to generate risk report."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    report_type: ReportType
    format: ReportFormat
    supplier_ids: List[str] = Field(..., min_items=1)
    language: str = Field(default="en", min_length=2, max_length=2)
    generated_by: str = Field(..., min_length=1, max_length=100)

class GetSupplierProfileRequest(GreenLangBase):
    """Request to get supplier profile."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)

class CompareSupplierRequest(GreenLangBase):
    """Request to compare multiple suppliers."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_ids: List[str] = Field(..., min_items=2)
    comparison_factors: Optional[List[str]] = None

class GetTrendRequest(GreenLangBase):
    """Request to get risk trend analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    period_months: int = Field(default=12, ge=1, le=60)

class BatchAssessmentRequest(GreenLangBase):
    """Request to perform batch supplier assessment."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_ids: List[str] = Field(..., min_items=1, max_items=500)
    assessed_by: str = Field(..., min_length=1, max_length=100)

class SearchSupplierRequest(GreenLangBase):
    """Request to search suppliers by criteria."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    risk_level: Optional[RiskLevel] = None
    supplier_type: Optional[SupplierType] = None
    commodity: Optional[CommodityType] = None
    country: Optional[str] = Field(None, min_length=2, max_length=2)
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)

class HealthRequest(GreenLangBase):
    """Request to check service health."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    include_database: bool = True
    include_cache: bool = True

# =============================================================================
# Response Models
# =============================================================================

class SupplierRiskResponse(GreenLangBase):
    """Response for supplier risk assessment."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    assessment: SupplierRiskAssessment
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class DueDiligenceResponse(GreenLangBase):
    """Response for due diligence tracking."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    dd_record: DueDiligenceRecord
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class DocumentationResponse(GreenLangBase):
    """Response for documentation analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    profile: DocumentationProfile
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class CertificationResponse(GreenLangBase):
    """Response for certification validation."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    certifications: List[CertificationRecord]
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class GeographicSourcingResponse(GreenLangBase):
    """Response for geographic sourcing analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    profile: GeographicSourcingProfile
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class NetworkResponse(GreenLangBase):
    """Response for network analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    network: SupplierNetwork
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class MonitoringResponse(GreenLangBase):
    """Response for monitoring configuration."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    config: MonitoringConfig
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class AlertResponse(GreenLangBase):
    """Response for alert generation."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    alert: SupplierAlert
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class ReportResponse(GreenLangBase):
    """Response for report generation."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    report: RiskReport
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class ProfileResponse(GreenLangBase):
    """Response for supplier profile retrieval."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    profile: SupplierProfile
    processing_time_ms: float

class ComparisonResponse(GreenLangBase):
    """Response for supplier comparison."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    assessments: List[SupplierRiskAssessment]
    comparison_matrix: Dict[str, Any]
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class TrendResponse(GreenLangBase):
    """Response for trend analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str
    trend_data: List[Dict[str, Any]]
    trend_direction: str = Field(..., min_length=1, max_length=50)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class BatchResponse(GreenLangBase):
    """Response for batch assessment."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    assessments: List[SupplierRiskAssessment]
    total_count: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)

class SearchResponse(GreenLangBase):
    """Response for supplier search."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    suppliers: List[SupplierProfile]
    total_count: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    processing_time_ms: float

class HealthResponse(GreenLangBase):
    """Response for health check."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    status: str = Field(..., min_length=1, max_length=50)
    database_status: Optional[str] = Field(None, max_length=50)
    cache_status: Optional[str] = Field(None, max_length=50)
    version: str = Field(default=VERSION)
    timestamp: datetime = Field(default_factory=utcnow)
