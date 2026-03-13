# -*- coding: utf-8 -*-
"""
Legal Compliance Verifier Data Models - AGENT-EUDR-023

Pydantic v2 data models for the Legal Compliance Verifier Agent covering
12 enumerations, 10 core models, 12 request models, and 12 response models
aligned with EUDR Article 2(40) eight legislation categories.

Enumerations (12):
    LegislationCategory, ComplianceStatus, DocumentValidityStatus,
    CertificationScheme, RedFlagSeverity, RedFlagCategory,
    AuditFindingCategory, AuditFindingStatus, ReportType,
    ReportFormat, RiskLevel, CommodityType

Core Models (10):
    LegalFramework, ComplianceDocument, CertificationRecord,
    RedFlagAlert, AuditReport, ComplianceAssessment,
    LegalRequirement, ComplianceReport, AuditLogEntry, AuditFinding

Request Models (12):
    QueryLegalFrameworkRequest, VerifyDocumentRequest,
    ValidateCertificationRequest, ScanRedFlagsRequest,
    AssessComplianceRequest, SubmitAuditReportRequest,
    GenerateReportRequest, BatchAssessmentRequest,
    AcknowledgeRedFlagRequest, UpdateFrameworkRequest,
    ExpiringDocumentsRequest, CountryComplianceRequest

Response Models (12):
    LegalFrameworkResponse, DocumentVerificationResponse,
    CertificationValidationResponse, RedFlagScanResponse,
    ComplianceAssessmentResponse, AuditReportResponse,
    ComplianceReportResponse, BatchAssessmentResponse,
    ExpiringDocumentsResponse, CountryComplianceSummaryResponse,
    HealthCheckResponse, AdminStatsResponse

Zero-Hallucination:
    - All numeric fields use Decimal for deterministic arithmetic
    - Frozen models prevent mutation after creation
    - Comprehensive field validation via Pydantic v2 validators
    - Provenance hash on all core models for audit trail

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

VERSION = "1.0.0"
EUDR_CUTOFF_DATE = date(2020, 12, 31)
MAX_BATCH_SIZE = 1000
EUDR_RETENTION_YEARS = 5

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ===========================================================================
# Enumerations (12)
# ===========================================================================


class LegislationCategory(str, Enum):
    """EUDR Article 2(40) legislation categories."""
    LAND_USE_RIGHTS = "land_use_rights"
    ENVIRONMENTAL_PROTECTION = "environmental_protection"
    FOREST_RELATED_RULES = "forest_related_rules"
    THIRD_PARTY_RIGHTS = "third_party_rights"
    LABOUR_RIGHTS = "labour_rights"
    TAX_AND_ROYALTY = "tax_and_royalty"
    TRADE_AND_CUSTOMS = "trade_and_customs"
    ANTI_CORRUPTION = "anti_corruption"


class ComplianceStatus(str, Enum):
    """Compliance determination states."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_DATA = "insufficient_data"


class DocumentValidityStatus(str, Enum):
    """Document validity states."""
    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    UNVERIFIABLE = "unverifiable"


class CertificationScheme(str, Enum):
    """Supported certification schemes."""
    FSC_FM = "fsc_fm"
    FSC_COC = "fsc_coc"
    FSC_CW = "fsc_cw"
    PEFC_SFM = "pefc_sfm"
    PEFC_COC = "pefc_coc"
    RSPO_PC = "rspo_pc"
    RSPO_SCC = "rspo_scc"
    RSPO_IS = "rspo_is"
    RA_SA = "ra_sa"
    RA_COC = "ra_coc"
    ISCC_EU = "iscc_eu"
    ISCC_PLUS = "iscc_plus"


class RedFlagSeverity(str, Enum):
    """Red flag severity classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RedFlagCategory(str, Enum):
    """Red flag indicator categories."""
    CORRUPTION_BRIBERY = "corruption_bribery"
    ILLEGAL_LOGGING = "illegal_logging"
    LAND_RIGHTS_VIOLATION = "land_rights_violation"
    LABOUR_VIOLATION = "labour_violation"
    TAX_EVASION = "tax_evasion"
    DOCUMENT_FRAUD = "document_fraud"


class AuditFindingCategory(str, Enum):
    """Audit finding classification."""
    MAJOR_NON_CONFORMITY = "major_non_conformity"
    MINOR_NON_CONFORMITY = "minor_non_conformity"
    OBSERVATION = "observation"
    POSITIVE_PRACTICE = "positive_practice"
    NOT_APPLICABLE = "not_applicable"


class AuditFindingStatus(str, Enum):
    """Corrective action follow-up status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"
    OVERDUE = "overdue"


class ReportType(str, Enum):
    """Compliance report types."""
    FULL_ASSESSMENT = "full_assessment"
    CATEGORY_SPECIFIC = "category_specific"
    SUPPLIER_SCORECARD = "supplier_scorecard"
    RED_FLAG_SUMMARY = "red_flag_summary"
    DOCUMENT_STATUS = "document_status"
    CERTIFICATION_VALIDITY = "certification_validity"
    COUNTRY_FRAMEWORK = "country_framework"
    DDS_ANNEX = "dds_annex"


class ReportFormat(str, Enum):
    """Report output formats."""
    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    XBRL = "xbrl"
    XML = "xml"


class RiskLevel(str, Enum):
    """General risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class CommodityType(str, Enum):
    """EUDR Article 1 regulated commodities."""
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


# ===========================================================================
# Core Models (10)
# ===========================================================================


class LegalFramework(BaseModel):
    """Country-specific legal framework record for one legislation category.

    Represents a single piece of legislation applicable to EUDR-regulated
    commodity production in a specific country for a specific Article 2(40)
    legislation category.

    Example:
        >>> fw = LegalFramework(
        ...     country_code="BR",
        ...     category=LegislationCategory.FOREST_RELATED_RULES,
        ...     law_name="Forest Code",
        ...     law_reference="Lei 12.651/2012",
        ...     enacted_date=date(2012, 5, 25),
        ...     source_database="national_portal",
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    country_code: str = Field(..., min_length=2, max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code")
    category: LegislationCategory
    law_name: str = Field(..., max_length=500)
    law_reference: str = Field(..., max_length=200,
        description="Official gazette/statute reference number")
    enacted_date: date
    last_amended_date: Optional[date] = None
    enforcement_status: str = Field(default="active",
        pattern=r"^(active|suspended|repealed|pending)$")
    applicable_commodities: List[CommodityType] = Field(default_factory=list)
    source_database: str = Field(...,
        pattern=r"^(faolex|ecolex|natlex|national_portal|eu_oj|wto)$")
    source_url: Optional[str] = None
    requirements: List[str] = Field(default_factory=list,
        description="List of specific legal requirements")
    provenance_hash: Optional[str] = Field(None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ComplianceDocument(BaseModel):
    """A permit, license, certificate, or legal document for verification.

    Example:
        >>> doc = ComplianceDocument(
        ...     supplier_id=uuid.uuid4(),
        ...     document_type="forest_harvesting_permit",
        ...     document_number="FP-2024-001",
        ...     issuing_authority="IBAMA",
        ...     issuing_country="BR",
        ...     issue_date=date(2024, 1, 15),
        ...     legislation_category=LegislationCategory.FOREST_RELATED_RULES,
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    document_type: str = Field(..., max_length=100)
    document_number: str = Field(..., max_length=200)
    issuing_authority: str = Field(..., max_length=300)
    issuing_country: str = Field(..., min_length=2, max_length=3)
    issue_date: date
    expiry_date: Optional[date] = None
    validity_status: DocumentValidityStatus = DocumentValidityStatus.VALID
    verification_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"))
    verification_details: Optional[Dict[str, Any]] = None
    s3_document_key: Optional[str] = Field(None, max_length=500)
    checksum_sha256: Optional[str] = Field(None, max_length=64)
    legislation_category: LegislationCategory
    linked_framework_id: Optional[uuid.UUID] = None
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class CertificationRecord(BaseModel):
    """Certification scheme record for a supplier or site.

    Example:
        >>> cert = CertificationRecord(
        ...     supplier_id=uuid.uuid4(),
        ...     scheme=CertificationScheme.FSC_FM,
        ...     certificate_number="FSC-C123456",
        ...     certification_body="SGS SA",
        ...     issue_date=date(2023, 6, 1),
        ...     expiry_date=date(2028, 5, 31),
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    scheme: CertificationScheme
    certificate_number: str = Field(..., max_length=100)
    certification_body: str = Field(..., max_length=300)
    cb_accreditation_number: Optional[str] = Field(None, max_length=100)
    scope_description: str = Field(default="", max_length=2000)
    covered_commodities: List[CommodityType] = Field(default_factory=list)
    covered_sites: List[str] = Field(default_factory=list)
    coc_model: Optional[str] = Field(None, max_length=50)
    issue_date: date
    expiry_date: date
    last_audit_date: Optional[date] = None
    next_audit_date: Optional[date] = None
    validity_status: DocumentValidityStatus = DocumentValidityStatus.VALID
    non_conformities_open: int = Field(default=0, ge=0)
    eudr_equivalence_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"))
    eudr_categories_covered: List[LegislationCategory] = Field(
        default_factory=list)
    provenance_hash: Optional[str] = Field(None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class RedFlagAlert(BaseModel):
    """A triggered red flag indicator for a supplier.

    Example:
        >>> flag = RedFlagAlert(
        ...     supplier_id=uuid.uuid4(),
        ...     country_code="CD",
        ...     flag_code="RF-001",
        ...     flag_category=RedFlagCategory.CORRUPTION_BRIBERY,
        ...     flag_description="Supplier in country with CPI < 30",
        ...     severity=RedFlagSeverity.HIGH,
        ...     base_weight=Decimal("0.20"),
        ...     triggering_data={"cpi_score": 19},
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    flag_code: str = Field(..., max_length=10,
        description="Red flag identifier (RF-001 through RF-040)")
    flag_category: RedFlagCategory
    flag_description: str = Field(..., max_length=500)
    severity: RedFlagSeverity
    base_weight: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    country_multiplier: Decimal = Field(default=Decimal("1.0"),
        ge=Decimal("0.5"), le=Decimal("3.0"))
    commodity_multiplier: Decimal = Field(default=Decimal("1.0"),
        ge=Decimal("0.5"), le=Decimal("2.0"))
    weighted_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"))
    triggering_data: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)


class AuditFinding(BaseModel):
    """Individual finding extracted from an audit report.

    Example:
        >>> finding = AuditFinding(
        ...     audit_report_id=uuid.uuid4(),
        ...     finding_reference="NC-01",
        ...     finding_category=AuditFindingCategory.MAJOR_NON_CONFORMITY,
        ...     description="Inadequate forest management plan",
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    audit_report_id: uuid.UUID
    finding_reference: str = Field(..., max_length=50)
    finding_category: AuditFindingCategory
    applicable_requirement: Optional[str] = Field(None, max_length=200)
    description: str = Field(..., max_length=2000)
    evidence_observed: Optional[str] = Field(None, max_length=2000)
    root_cause: Optional[str] = Field(None, max_length=1000)
    corrective_action_required: Optional[str] = Field(None, max_length=1000)
    corrective_action_deadline: Optional[date] = None
    follow_up_status: AuditFindingStatus = AuditFindingStatus.OPEN
    closed_date: Optional[date] = None
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class AuditReport(BaseModel):
    """Third-party audit report with extracted findings.

    Example:
        >>> report = AuditReport(
        ...     supplier_id=uuid.uuid4(),
        ...     audit_type="FSC",
        ...     auditor_organization="Control Union",
        ...     audit_date=date(2025, 6, 15),
        ...     report_date=date(2025, 7, 1),
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    audit_type: str = Field(..., max_length=100)
    auditor_organization: str = Field(..., max_length=300)
    lead_auditor: Optional[str] = Field(None, max_length=200)
    audit_date: date
    report_date: date
    scope: str = Field(default="", max_length=2000)
    overall_conclusion: str = Field(default="",
        pattern=r"^(conformant|minor_nc|major_nc|suspended|withdrawn|)$")
    major_non_conformities: int = Field(default=0, ge=0)
    minor_non_conformities: int = Field(default=0, ge=0)
    observations: int = Field(default=0, ge=0)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    corrective_action_deadline: Optional[date] = None
    follow_up_audit_date: Optional[date] = None
    s3_report_key: Optional[str] = Field(None, max_length=500)
    provenance_hash: Optional[str] = Field(None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ComplianceAssessment(BaseModel):
    """Full 8-category compliance assessment for a supplier-commodity pair.

    Example:
        >>> assessment = ComplianceAssessment(
        ...     supplier_id=uuid.uuid4(),
        ...     country_code="BR",
        ...     commodity=CommodityType.SOYA,
        ...     overall_status=ComplianceStatus.PARTIALLY_COMPLIANT,
        ...     overall_score=Decimal("72.5"),
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    assessment_date: datetime = Field(default_factory=_utcnow)
    overall_status: ComplianceStatus
    overall_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("100"))
    category_scores: Dict[str, Decimal] = Field(default_factory=dict)
    category_statuses: Dict[str, str] = Field(default_factory=dict)
    requirements_total: int = Field(default=0, ge=0)
    requirements_met: int = Field(default=0, ge=0)
    requirements_unmet: int = Field(default=0, ge=0)
    requirements_insufficient_data: int = Field(default=0, ge=0)
    gap_analysis: List[Dict[str, Any]] = Field(default_factory=list)
    red_flag_count: int = Field(default=0, ge=0)
    red_flag_score: Decimal = Field(default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"))
    documents_verified: int = Field(default=0, ge=0)
    certifications_validated: int = Field(default=0, ge=0)
    risk_level: RiskLevel = RiskLevel.MODERATE
    provenance_hash: Optional[str] = Field(None, max_length=64)
    chain_hash: Optional[str] = Field(None, max_length=64)
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)


class LegalRequirement(BaseModel):
    """A specific legal requirement derived from a legal framework.

    Example:
        >>> req = LegalRequirement(
        ...     framework_id=uuid.uuid4(),
        ...     country_code="BR",
        ...     category=LegislationCategory.FOREST_RELATED_RULES,
        ...     requirement_code="BR-FR-001",
        ...     description="Valid DOF for timber transport",
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    framework_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    category: LegislationCategory
    requirement_code: str = Field(..., max_length=50)
    description: str = Field(..., max_length=1000)
    applicable_commodities: List[CommodityType] = Field(default_factory=list)
    mandatory: bool = True
    evidence_types: List[str] = Field(default_factory=list)
    verification_method: str = Field(default="document_check",
        pattern=r"^(document_check|certification_check|field_audit|"
                r"self_declaration|database_query)$")
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID


class ComplianceReport(BaseModel):
    """Generated compliance report metadata.

    Example:
        >>> report = ComplianceReport(
        ...     assessment_id=uuid.uuid4(),
        ...     report_type=ReportType.FULL_ASSESSMENT,
        ...     report_format=ReportFormat.PDF,
        ...     s3_report_key="reports/2026/03/full-001.pdf",
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    assessment_id: uuid.UUID
    report_type: ReportType
    report_format: ReportFormat
    language: str = Field(default="en", pattern=r"^(en|fr|de|es|pt)$")
    s3_report_key: str = Field(..., max_length=500)
    file_size_bytes: int = Field(default=0, ge=0)
    generated_at: datetime = Field(default_factory=_utcnow)
    digital_signature: Optional[str] = Field(None, max_length=512)
    provenance_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID


class AuditLogEntry(BaseModel):
    """Audit trail entry for all LCV operations.

    Example:
        >>> entry = AuditLogEntry(
        ...     entity_type="compliance_assessment",
        ...     entity_id=uuid.uuid4(),
        ...     action="create",
        ...     actor="user@example.com",
        ...     tenant_id=uuid.uuid4(),
        ... )
    """
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    entity_type: str = Field(..., max_length=50)
    entity_id: uuid.UUID
    action: str = Field(..., max_length=50)
    actor: str = Field(..., max_length=200)
    details: Optional[Dict[str, Any]] = None
    provenance_hash: Optional[str] = Field(None, max_length=64)
    chain_hash: Optional[str] = Field(None, max_length=64)
    tenant_id: uuid.UUID
    created_at: datetime = Field(default_factory=_utcnow)


# ===========================================================================
# Request Models (12)
# ===========================================================================


class QueryLegalFrameworkRequest(BaseModel):
    """Request to query legal frameworks for a country."""
    country_code: str = Field(..., min_length=2, max_length=3)
    category: Optional[LegislationCategory] = None
    commodity: Optional[CommodityType] = None
    include_repealed: bool = False


class VerifyDocumentRequest(BaseModel):
    """Request to verify a compliance document."""
    supplier_id: uuid.UUID
    document_type: str = Field(..., max_length=100)
    document_number: str = Field(..., max_length=200)
    issuing_authority: str = Field(..., max_length=300)
    issuing_country: str = Field(..., min_length=2, max_length=3)
    issue_date: date
    expiry_date: Optional[date] = None
    legislation_category: LegislationCategory
    s3_document_key: Optional[str] = None


class ValidateCertificationRequest(BaseModel):
    """Request to validate a certification."""
    supplier_id: uuid.UUID
    scheme: CertificationScheme
    certificate_number: str = Field(..., max_length=100)
    certification_body: Optional[str] = None


class ScanRedFlagsRequest(BaseModel):
    """Request to scan a supplier for red flags."""
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    include_categories: Optional[List[RedFlagCategory]] = None


class AssessComplianceRequest(BaseModel):
    """Request to run full compliance assessment."""
    supplier_id: uuid.UUID
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    categories: Optional[List[LegislationCategory]] = None
    include_red_flags: bool = True
    include_certifications: bool = True


class SubmitAuditReportRequest(BaseModel):
    """Request to submit an audit report for processing."""
    supplier_id: uuid.UUID
    audit_type: str = Field(..., max_length=100)
    auditor_organization: str = Field(..., max_length=300)
    audit_date: date
    report_date: date
    s3_report_key: str = Field(..., max_length=500)


class GenerateReportRequest(BaseModel):
    """Request to generate a compliance report."""
    assessment_id: uuid.UUID
    report_type: ReportType
    report_format: ReportFormat = ReportFormat.PDF
    language: str = Field(default="en", pattern=r"^(en|fr|de|es|pt)$")


class BatchAssessmentRequest(BaseModel):
    """Request for batch compliance assessment."""
    supplier_ids: List[uuid.UUID] = Field(..., max_length=1000)
    commodity: CommodityType
    categories: Optional[List[LegislationCategory]] = None
    include_red_flags: bool = True


class AcknowledgeRedFlagRequest(BaseModel):
    """Request to acknowledge a red flag."""
    flag_id: uuid.UUID
    acknowledged_by: str = Field(..., max_length=200)
    justification: Optional[str] = Field(None, max_length=1000)


class UpdateFrameworkRequest(BaseModel):
    """Request to create/update a legal framework record."""
    country_code: str = Field(..., min_length=2, max_length=3)
    category: LegislationCategory
    law_name: str = Field(..., max_length=500)
    law_reference: str = Field(..., max_length=200)
    enacted_date: date
    source_database: str
    requirements: List[str] = Field(default_factory=list)


class ExpiringDocumentsRequest(BaseModel):
    """Request to list expiring documents."""
    days_ahead: int = Field(default=30, ge=1, le=365)
    country_code: Optional[str] = None
    document_type: Optional[str] = None


class CountryComplianceRequest(BaseModel):
    """Request for country compliance summary."""
    country_code: str = Field(..., min_length=2, max_length=3)
    commodity: CommodityType
    supplier_ids: Optional[List[uuid.UUID]] = None


# ===========================================================================
# Response Models (12)
# ===========================================================================


class LegalFrameworkResponse(BaseModel):
    """Response for legal framework queries."""
    frameworks: List[LegalFramework]
    total_count: int
    country_code: str
    categories_covered: List[str]
    provenance_hash: str


class DocumentVerificationResponse(BaseModel):
    """Response for document verification."""
    document: ComplianceDocument
    verification_passed: bool
    verification_steps: List[Dict[str, Any]]
    warnings: List[str]
    provenance_hash: str


class CertificationValidationResponse(BaseModel):
    """Response for certification validation."""
    certification: CertificationRecord
    validation_passed: bool
    validation_checks: List[Dict[str, Any]]
    eudr_coverage_summary: Dict[str, bool]
    provenance_hash: str


class RedFlagScanResponse(BaseModel):
    """Response for red flag scan."""
    supplier_id: uuid.UUID
    flags_triggered: List[RedFlagAlert]
    total_flags: int
    aggregate_score: Decimal
    risk_level: RiskLevel
    category_breakdown: Dict[str, int]
    provenance_hash: str


class ComplianceAssessmentResponse(BaseModel):
    """Response for compliance assessment."""
    assessment: ComplianceAssessment
    frameworks_checked: int
    documents_verified: int
    certifications_validated: int
    red_flags_detected: int
    gap_analysis: List[Dict[str, Any]]
    recommendations: List[str]
    provenance_hash: str


class AuditReportResponse(BaseModel):
    """Response for audit report processing."""
    report: AuditReport
    findings_extracted: int
    compliance_impact: Dict[str, Any]
    provenance_hash: str


class ComplianceReportResponse(BaseModel):
    """Response for compliance report generation."""
    report: ComplianceReport
    download_url: str
    provenance_hash: str


class BatchAssessmentResponse(BaseModel):
    """Response for batch compliance assessment."""
    job_id: uuid.UUID
    total_suppliers: int
    completed: int
    failed: int
    results: List[ComplianceAssessmentResponse]
    provenance_hash: str


class ExpiringDocumentsResponse(BaseModel):
    """Response for expiring documents query."""
    documents: List[ComplianceDocument]
    total_expiring: int
    by_category: Dict[str, int]
    by_country: Dict[str, int]


class CountryComplianceSummaryResponse(BaseModel):
    """Response for country compliance summary."""
    country_code: str
    commodity: CommodityType
    framework_completeness: Decimal
    average_supplier_score: Decimal
    category_averages: Dict[str, Decimal]
    non_compliant_count: int
    total_assessed: int
    provenance_hash: str


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    engines: Dict[str, str]
    database_connected: bool
    redis_connected: bool
    uptime_seconds: float
    provenance_chain_valid: bool


class AdminStatsResponse(BaseModel):
    """Admin statistics response."""
    total_frameworks: int
    total_documents: int
    total_certifications: int
    total_assessments: int
    total_red_flags: int
    total_audit_reports: int
    countries_covered: int
    assessments_last_24h: int
