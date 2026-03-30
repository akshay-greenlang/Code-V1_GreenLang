# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-023 Legal Compliance Verifier

Pydantic v2 request/response models for the REST API layer covering all
8 route domains: legal framework management, document verification,
certification validation, red flag detection, compliance assessment,
audit integration, reporting, and batch processing.

All numeric fields use ``Decimal`` for precision (zero-hallucination).
All date/time fields use UTC-aware ``datetime``.

Schema Groups (8 domains + common):
    Common: ProvenanceInfo, MetadataSchema, PaginatedMeta, ErrorResponse,
            HealthResponse
    1. Framework: FrameworkRegisterRequest/Response, FrameworkDetailResponse,
       FrameworkUpdateRequest, FrameworkListResponse, FrameworkSearchRequest/Response
    2. Document: DocumentVerifyRequest/Response, DocumentListResponse,
       DocumentDetailResponse, ValidityCheckRequest/Response,
       ExpiringDocumentsResponse
    3. Certification: CertValidateRequest/Response, CertListResponse,
       CertDetailResponse, EUDREquivalenceRequest/Response
    4. RedFlag: RedFlagDetectRequest/Response, RedFlagListResponse,
       RedFlagDetailResponse, RedFlagSuppressRequest/Response
    5. Compliance: ComplianceAssessRequest/Response, CategoryCheckRequest/Response,
       ComplianceListResponse, ComplianceDetailResponse, ComplianceHistoryResponse
    6. Audit: AuditIngestRequest/Response, AuditListResponse,
       AuditFindingsResponse, CorrectiveActionsRequest/Response
    7. Report: ReportGenerateRequest/Response, ReportListResponse,
       ReportDownloadResponse, ReportScheduleRequest/Response
    8. Batch: BatchAssessRequest/Response, BatchVerifyRequest/Response,
       BatchStatusResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-023 Legal Compliance Verifier (GL-EUDR-LCV-023)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator

from greenlang.schemas import GreenLangBase, utcnow

def _new_id() -> str:
    """Generate a new UUID4 string identifier."""
    return str(uuid.uuid4())

# =============================================================================
# Enumerations (API-level mirrors for OpenAPI documentation)
# =============================================================================

class JurisdictionTypeEnum(str, Enum):
    """Legal jurisdiction types."""

    EU = "eu"
    NATIONAL = "national"
    REGIONAL = "regional"
    BILATERAL = "bilateral"
    INTERNATIONAL = "international"

class FrameworkStatusEnum(str, Enum):
    """Legal framework status."""

    ACTIVE = "active"
    DRAFT = "draft"
    REPEALED = "repealed"
    AMENDED = "amended"
    PENDING = "pending"

class DocumentTypeEnum(str, Enum):
    """Document types for legal verification."""

    DUE_DILIGENCE_STATEMENT = "due_diligence_statement"
    IMPORT_LICENSE = "import_license"
    EXPORT_PERMIT = "export_permit"
    PHYTOSANITARY_CERTIFICATE = "phytosanitary_certificate"
    CERTIFICATE_OF_ORIGIN = "certificate_of_origin"
    CUSTOMS_DECLARATION = "customs_declaration"
    OPERATOR_REGISTRATION = "operator_registration"
    SUSTAINABILITY_CERTIFICATE = "sustainability_certificate"
    AUDIT_REPORT = "audit_report"
    CORRECTIVE_ACTION_PLAN = "corrective_action_plan"
    OTHER = "other"

class DocumentStatusEnum(str, Enum):
    """Document verification status."""

    VERIFIED = "verified"
    PENDING = "pending"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNDER_REVIEW = "under_review"

class CertificationTypeEnum(str, Enum):
    """Certification types for EUDR equivalence."""

    FSC = "fsc"
    PEFC = "pefc"
    RSPO = "rspo"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    UTZ = "utz"
    FAIRTRADE = "fairtrade"
    ISCC = "iscc"
    BONSUCRO = "bonsucro"
    ROUND_TABLE_SOY = "round_table_soy"
    EU_ORGANIC = "eu_organic"
    NATIONAL_SCHEME = "national_scheme"
    OTHER = "other"

class CertStatusEnum(str, Enum):
    """Certification validation status."""

    VALID = "valid"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    PENDING_RENEWAL = "pending_renewal"
    NOT_RECOGNIZED = "not_recognized"

class EquivalenceResultEnum(str, Enum):
    """EUDR equivalence assessment outcome."""

    EQUIVALENT = "equivalent"
    PARTIALLY_EQUIVALENT = "partially_equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    UNDER_REVIEW = "under_review"

class RedFlagSeverityEnum(str, Enum):
    """Red flag severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class RedFlagCategoryEnum(str, Enum):
    """Red flag categories."""

    DOCUMENT_FORGERY = "document_forgery"
    EXPIRED_CERTIFICATION = "expired_certification"
    SANCTIONED_ENTITY = "sanctioned_entity"
    HIGH_RISK_COUNTRY = "high_risk_country"
    DEFORESTATION_LINK = "deforestation_link"
    MISSING_DUE_DILIGENCE = "missing_due_diligence"
    INCONSISTENT_DATA = "inconsistent_data"
    UNAUTHORIZED_OPERATOR = "unauthorized_operator"
    SUPPLY_CHAIN_GAP = "supply_chain_gap"
    REGULATORY_VIOLATION = "regulatory_violation"
    CUSTOMS_DISCREPANCY = "customs_discrepancy"
    OTHER = "other"

class RedFlagStatusEnum(str, Enum):
    """Red flag lifecycle status."""

    ACTIVE = "active"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    SUPPRESSED = "suppressed"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"

class ComplianceOutcomeEnum(str, Enum):
    """Compliance assessment outcomes."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    AT_RISK = "at_risk"
    REQUIRES_INVESTIGATION = "requires_investigation"
    PENDING = "pending"

class ComplianceCategoryEnum(str, Enum):
    """Compliance assessment categories per EUDR structure."""

    DEFORESTATION_FREE = "deforestation_free"
    LEGALITY = "legality"
    DUE_DILIGENCE = "due_diligence"
    TRACEABILITY = "traceability"
    DOCUMENTATION = "documentation"
    RISK_ASSESSMENT = "risk_assessment"
    RISK_MITIGATION = "risk_mitigation"
    MONITORING = "monitoring"
    REPORTING = "reporting"

class AuditTypeEnum(str, Enum):
    """Audit report types."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    REGULATORY = "regulatory"
    THIRD_PARTY = "third_party"
    SELF_ASSESSMENT = "self_assessment"

class AuditStatusEnum(str, Enum):
    """Audit lifecycle status."""

    INGESTED = "ingested"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    FINDINGS_ISSUED = "findings_issued"
    CORRECTIVE_ACTIONS_PENDING = "corrective_actions_pending"
    CLOSED = "closed"

class FindingSeverityEnum(str, Enum):
    """Audit finding severity."""

    MAJOR_NON_CONFORMITY = "major_non_conformity"
    MINOR_NON_CONFORMITY = "minor_non_conformity"
    OBSERVATION = "observation"
    OPPORTUNITY_FOR_IMPROVEMENT = "opportunity_for_improvement"

class CorrectiveActionStatusEnum(str, Enum):
    """Corrective action status."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    OVERDUE = "overdue"

class ReportTypeEnum(str, Enum):
    """Report types."""

    COMPLIANCE_SUMMARY = "compliance_summary"
    DUE_DILIGENCE_REPORT = "due_diligence_report"
    RED_FLAG_REPORT = "red_flag_report"
    AUDIT_SUMMARY = "audit_summary"
    CERTIFICATION_STATUS = "certification_status"
    REGULATORY_FILING = "regulatory_filing"
    RISK_ASSESSMENT = "risk_assessment"

class ReportFormatEnum(str, Enum):
    """Report output formats."""

    PDF = "pdf"
    XLSX = "xlsx"
    CSV = "csv"
    JSON = "json"
    HTML = "html"

class ReportStatusEnum(str, Enum):
    """Report generation status."""

    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"

class ScheduleFrequencyEnum(str, Enum):
    """Report schedule frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

class BatchStatusEnum(str, Enum):
    """Batch processing status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"

class EUDRCommodityEnum(str, Enum):
    """EUDR-regulated commodity types per Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

# =============================================================================
# Common / Shared Schemas
# =============================================================================

class ProvenanceInfo(GreenLangBase):
    """Provenance tracking information for audit trail."""

    model_config = ConfigDict(populate_by_name=True)

    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail verification"
    )
    processing_time_ms: Decimal = Field(
        ..., description="Processing duration in milliseconds"
    )
    agent_id: str = Field(
        default="GL-EUDR-LCV-023",
        description="Agent identifier",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of operation",
    )

class MetadataSchema(GreenLangBase):
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
            "Art. 2", "Art. 3", "Art. 4", "Art. 8",
            "Art. 9", "Art. 10", "Art. 11", "Art. 12",
        ],
        description="Applicable regulatory articles",
    )
    api_version: str = Field(
        default="1.0.0", description="API version"
    )

class PaginatedMeta(GreenLangBase):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of records")
    limit: int = Field(..., ge=1, description="Records per page")
    offset: int = Field(..., ge=0, description="Number of records skipped")
    has_more: bool = Field(..., description="Whether more pages exist")

class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")

class HealthResponse(GreenLangBase):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Service health status")
    agent_id: str = Field(
        default="GL-EUDR-LCV-023", description="Agent identifier"
    )
    component: str = Field(
        default="legal-compliance-verifier", description="Component name"
    )
    version: str = Field(default="1.0.0", description="API version")

# =============================================================================
# 1. Legal Framework Schemas
# =============================================================================

class FrameworkRegisterRequest(GreenLangBase):
    """Request to register a new legal framework."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        ..., min_length=1, max_length=500,
        description="Legal framework name (e.g. 'EU 2023/1115 EUDR')",
    )
    jurisdiction: JurisdictionTypeEnum = Field(
        ..., description="Jurisdiction type"
    )
    country_codes: List[str] = Field(
        default_factory=list,
        description="ISO 3166-1 alpha-2 country codes where applicable",
    )
    effective_date: date = Field(
        ..., description="Date the framework takes effect"
    )
    expiry_date: Optional[date] = Field(
        None, description="Date the framework expires (if applicable)"
    )
    description: Optional[str] = Field(
        None, max_length=5000, description="Framework description"
    )
    reference_url: Optional[str] = Field(
        None, max_length=2000, description="URL to official regulation text"
    )
    articles: Optional[List[str]] = Field(
        None, description="Key article references"
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="EUDR commodities covered by this framework"
    )
    parent_framework_id: Optional[str] = Field(
        None, description="Parent framework ID for amendments"
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Validate all country codes are 2-letter uppercase."""
        for code in v:
            if len(code) != 2 or not code.isalpha():
                raise ValueError(f"Invalid country code: {code}")
        return [c.upper() for c in v]

class FrameworkEntry(GreenLangBase):
    """Summary of a single legal framework."""

    framework_id: str = Field(..., description="Unique framework identifier")
    name: str = Field(..., description="Framework name")
    jurisdiction: JurisdictionTypeEnum = Field(
        ..., description="Jurisdiction type"
    )
    country_codes: List[str] = Field(
        default_factory=list, description="Applicable countries"
    )
    status: FrameworkStatusEnum = Field(
        ..., description="Framework status"
    )
    effective_date: date = Field(..., description="Effective date")
    expiry_date: Optional[date] = Field(None, description="Expiry date")
    commodities: List[EUDRCommodityEnum] = Field(
        default_factory=list, description="Covered commodities"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(None, description="Last update")

class FrameworkRegisterResponse(GreenLangBase):
    """Response for framework registration."""

    model_config = ConfigDict(populate_by_name=True)

    framework: FrameworkEntry = Field(
        ..., description="Registered framework"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class FrameworkDetailResponse(GreenLangBase):
    """Detailed response for a single legal framework."""

    model_config = ConfigDict(populate_by_name=True)

    framework: FrameworkEntry = Field(
        ..., description="Framework details"
    )
    description: Optional[str] = Field(
        None, description="Full description"
    )
    reference_url: Optional[str] = Field(
        None, description="Official URL"
    )
    articles: List[str] = Field(
        default_factory=list, description="Key articles"
    )
    parent_framework_id: Optional[str] = Field(
        None, description="Parent framework if amendment"
    )
    amendment_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Amendment history"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class FrameworkUpdateRequest(GreenLangBase):
    """Request to update an existing legal framework."""

    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = Field(
        None, min_length=1, max_length=500, description="Updated name"
    )
    status: Optional[FrameworkStatusEnum] = Field(
        None, description="Updated status"
    )
    effective_date: Optional[date] = Field(
        None, description="Updated effective date"
    )
    expiry_date: Optional[date] = Field(
        None, description="Updated expiry date"
    )
    description: Optional[str] = Field(
        None, max_length=5000, description="Updated description"
    )
    reference_url: Optional[str] = Field(
        None, max_length=2000, description="Updated URL"
    )
    articles: Optional[List[str]] = Field(
        None, description="Updated articles"
    )
    country_codes: Optional[List[str]] = Field(
        None, description="Updated country codes"
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Updated commodities"
    )

class FrameworkListResponse(GreenLangBase):
    """Paginated list of legal frameworks."""

    model_config = ConfigDict(populate_by_name=True)

    frameworks: List[FrameworkEntry] = Field(
        default_factory=list, description="Legal frameworks"
    )
    total_frameworks: int = Field(
        default=0, ge=0, description="Total matching frameworks"
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

class FrameworkSearchRequest(GreenLangBase):
    """Request for advanced framework search."""

    model_config = ConfigDict(populate_by_name=True)

    query: Optional[str] = Field(
        None, max_length=500, description="Free-text search query"
    )
    jurisdiction: Optional[JurisdictionTypeEnum] = Field(
        None, description="Filter by jurisdiction"
    )
    country_code: Optional[str] = Field(
        None, description="Filter by country code"
    )
    status: Optional[FrameworkStatusEnum] = Field(
        None, description="Filter by status"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Filter by commodity"
    )
    effective_after: Optional[date] = Field(
        None, description="Frameworks effective after this date"
    )
    effective_before: Optional[date] = Field(
        None, description="Frameworks effective before this date"
    )

class FrameworkSearchResponse(GreenLangBase):
    """Response for advanced framework search."""

    model_config = ConfigDict(populate_by_name=True)

    frameworks: List[FrameworkEntry] = Field(
        default_factory=list, description="Matching frameworks"
    )
    total_results: int = Field(
        default=0, ge=0, description="Total results"
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
# 2. Document Verification Schemas
# =============================================================================

class DocumentVerifyRequest(GreenLangBase):
    """Request to verify a legal document."""

    model_config = ConfigDict(populate_by_name=True)

    document_type: DocumentTypeEnum = Field(
        ..., description="Type of document to verify"
    )
    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference number or ID",
    )
    issuing_authority: Optional[str] = Field(
        None, max_length=500, description="Authority that issued the document"
    )
    issuing_country: Optional[str] = Field(
        None, description="ISO 3166-1 alpha-2 country code of issuer"
    )
    issue_date: Optional[date] = Field(
        None, description="Document issue date"
    )
    expiry_date: Optional[date] = Field(
        None, description="Document expiry date"
    )
    operator_id: Optional[str] = Field(
        None, description="Associated operator ID"
    )
    supplier_id: Optional[str] = Field(
        None, description="Associated supplier ID"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Associated commodity"
    )
    file_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the document file"
    )
    file_url: Optional[str] = Field(
        None, max_length=2000, description="URL to document file"
    )
    additional_data: Optional[Dict[str, Any]] = Field(
        None, description="Additional document-specific data"
    )

class DocumentEntry(GreenLangBase):
    """Summary of a verified document."""

    document_id: str = Field(..., description="Unique document identifier")
    document_type: DocumentTypeEnum = Field(
        ..., description="Document type"
    )
    document_reference: str = Field(
        ..., description="Document reference number"
    )
    status: DocumentStatusEnum = Field(
        ..., description="Verification status"
    )
    issuing_authority: Optional[str] = Field(
        None, description="Issuing authority"
    )
    issuing_country: Optional[str] = Field(
        None, description="Issuing country code"
    )
    issue_date: Optional[date] = Field(None, description="Issue date")
    expiry_date: Optional[date] = Field(None, description="Expiry date")
    operator_id: Optional[str] = Field(None, description="Operator ID")
    supplier_id: Optional[str] = Field(None, description="Supplier ID")
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Associated commodity"
    )
    verification_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
        description="Verification confidence score (0-100)",
    )
    verified_at: Optional[datetime] = Field(
        None, description="Verification timestamp"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Record creation timestamp"
    )

class DocumentVerifyResponse(GreenLangBase):
    """Response for document verification."""

    model_config = ConfigDict(populate_by_name=True)

    document: DocumentEntry = Field(
        ..., description="Verified document record"
    )
    verification_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed verification checks performed",
    )
    red_flags_detected: int = Field(
        default=0, ge=0, description="Number of red flags detected"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class DocumentListResponse(GreenLangBase):
    """Paginated list of documents."""

    model_config = ConfigDict(populate_by_name=True)

    documents: List[DocumentEntry] = Field(
        default_factory=list, description="Document records"
    )
    total_documents: int = Field(
        default=0, ge=0, description="Total documents"
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

class DocumentDetailResponse(GreenLangBase):
    """Detailed response for a single document."""

    model_config = ConfigDict(populate_by_name=True)

    document: DocumentEntry = Field(
        ..., description="Document details"
    )
    verification_details: Dict[str, Any] = Field(
        default_factory=dict, description="Verification details"
    )
    file_hash: Optional[str] = Field(
        None, description="Document file hash"
    )
    file_url: Optional[str] = Field(
        None, description="Document file URL"
    )
    additional_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    related_documents: List[str] = Field(
        default_factory=list, description="Related document IDs"
    )
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list, description="Document audit trail"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class ValidityCheckRequest(GreenLangBase):
    """Request to check document validity status."""

    model_config = ConfigDict(populate_by_name=True)

    document_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="Document IDs to check validity (max 100)",
    )
    check_date: Optional[date] = Field(
        None, description="Date to check validity against (default: today)"
    )
    include_expiry_warnings: bool = Field(
        default=True,
        description="Include warnings for documents expiring within 30 days",
    )

class DocumentValidityEntry(GreenLangBase):
    """Validity status for a single document."""

    document_id: str = Field(..., description="Document ID")
    document_type: DocumentTypeEnum = Field(
        ..., description="Document type"
    )
    is_valid: bool = Field(..., description="Whether document is currently valid")
    status: DocumentStatusEnum = Field(
        ..., description="Current status"
    )
    expiry_date: Optional[date] = Field(None, description="Expiry date")
    days_until_expiry: Optional[int] = Field(
        None, description="Days until expiry (negative if expired)"
    )
    expiry_warning: bool = Field(
        default=False, description="True if expiring within 30 days"
    )
    issues: List[str] = Field(
        default_factory=list, description="Validity issues found"
    )

class ValidityCheckResponse(GreenLangBase):
    """Response for document validity check."""

    model_config = ConfigDict(populate_by_name=True)

    results: List[DocumentValidityEntry] = Field(
        default_factory=list, description="Validity results"
    )
    total_checked: int = Field(
        default=0, ge=0, description="Total documents checked"
    )
    valid_count: int = Field(
        default=0, ge=0, description="Valid documents"
    )
    invalid_count: int = Field(
        default=0, ge=0, description="Invalid documents"
    )
    expiring_soon_count: int = Field(
        default=0, ge=0, description="Documents expiring within 30 days"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class ExpiringDocumentEntry(GreenLangBase):
    """An expiring document entry."""

    document_id: str = Field(..., description="Document ID")
    document_type: DocumentTypeEnum = Field(
        ..., description="Document type"
    )
    document_reference: str = Field(
        ..., description="Document reference"
    )
    operator_id: Optional[str] = Field(None, description="Operator ID")
    supplier_id: Optional[str] = Field(None, description="Supplier ID")
    expiry_date: date = Field(..., description="Expiry date")
    days_until_expiry: int = Field(
        ..., description="Days until expiry"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Associated commodity"
    )
    status: DocumentStatusEnum = Field(
        ..., description="Current status"
    )

class ExpiringDocumentsResponse(GreenLangBase):
    """Response for expiring documents query."""

    model_config = ConfigDict(populate_by_name=True)

    documents: List[ExpiringDocumentEntry] = Field(
        default_factory=list, description="Expiring documents"
    )
    total_expiring: int = Field(
        default=0, ge=0, description="Total expiring documents"
    )
    expiring_within_7_days: int = Field(
        default=0, ge=0, description="Expiring within 7 days"
    )
    expiring_within_30_days: int = Field(
        default=0, ge=0, description="Expiring within 30 days"
    )
    expiring_within_90_days: int = Field(
        default=0, ge=0, description="Expiring within 90 days"
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
# 3. Certification Schemas
# =============================================================================

class CertValidateRequest(GreenLangBase):
    """Request to validate a certification."""

    model_config = ConfigDict(populate_by_name=True)

    certification_type: CertificationTypeEnum = Field(
        ..., description="Type of certification"
    )
    certificate_number: str = Field(
        ..., min_length=1, max_length=200,
        description="Certificate number or identifier",
    )
    holder_name: Optional[str] = Field(
        None, max_length=500, description="Certificate holder name"
    )
    holder_country: Optional[str] = Field(
        None, description="Certificate holder country code"
    )
    issue_date: Optional[date] = Field(
        None, description="Certificate issue date"
    )
    expiry_date: Optional[date] = Field(
        None, description="Certificate expiry date"
    )
    scope: Optional[str] = Field(
        None, max_length=2000, description="Certification scope description"
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities covered by certification"
    )
    supplier_id: Optional[str] = Field(
        None, description="Associated supplier ID"
    )
    verification_url: Optional[str] = Field(
        None, max_length=2000,
        description="URL for online certificate verification",
    )

class CertificationEntry(GreenLangBase):
    """Summary of a validated certification."""

    cert_id: str = Field(..., description="Unique certification record ID")
    certification_type: CertificationTypeEnum = Field(
        ..., description="Certification type"
    )
    certificate_number: str = Field(
        ..., description="Certificate number"
    )
    status: CertStatusEnum = Field(
        ..., description="Validation status"
    )
    holder_name: Optional[str] = Field(
        None, description="Certificate holder"
    )
    holder_country: Optional[str] = Field(
        None, description="Holder country"
    )
    issue_date: Optional[date] = Field(None, description="Issue date")
    expiry_date: Optional[date] = Field(None, description="Expiry date")
    scope: Optional[str] = Field(None, description="Certification scope")
    commodities: List[EUDRCommodityEnum] = Field(
        default_factory=list, description="Covered commodities"
    )
    eudr_equivalence: Optional[EquivalenceResultEnum] = Field(
        None, description="EUDR equivalence status"
    )
    validation_score: Optional[Decimal] = Field(
        None, ge=Decimal("0"), le=Decimal("100"),
        description="Validation confidence score (0-100)",
    )
    validated_at: Optional[datetime] = Field(
        None, description="Validation timestamp"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Record creation timestamp"
    )

class CertValidateResponse(GreenLangBase):
    """Response for certification validation."""

    model_config = ConfigDict(populate_by_name=True)

    certification: CertificationEntry = Field(
        ..., description="Validated certification"
    )
    validation_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed validation checks"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class CertListResponse(GreenLangBase):
    """Paginated list of certifications."""

    model_config = ConfigDict(populate_by_name=True)

    certifications: List[CertificationEntry] = Field(
        default_factory=list, description="Certification records"
    )
    total_certifications: int = Field(
        default=0, ge=0, description="Total certifications"
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

class CertDetailResponse(GreenLangBase):
    """Detailed response for a single certification."""

    model_config = ConfigDict(populate_by_name=True)

    certification: CertificationEntry = Field(
        ..., description="Certification details"
    )
    validation_details: Dict[str, Any] = Field(
        default_factory=dict, description="Validation details"
    )
    verification_url: Optional[str] = Field(
        None, description="Online verification URL"
    )
    supplier_id: Optional[str] = Field(
        None, description="Associated supplier"
    )
    related_documents: List[str] = Field(
        default_factory=list, description="Related document IDs"
    )
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list, description="Certification audit trail"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class EUDREquivalenceRequest(GreenLangBase):
    """Request to check EUDR equivalence of a certification."""

    model_config = ConfigDict(populate_by_name=True)

    certification_type: CertificationTypeEnum = Field(
        ..., description="Type of certification to assess"
    )
    certificate_number: Optional[str] = Field(
        None, description="Specific certificate to assess"
    )
    commodities: List[EUDRCommodityEnum] = Field(
        ..., min_length=1,
        description="Commodities to assess equivalence for",
    )
    country_code: Optional[str] = Field(
        None, description="Country context for equivalence"
    )
    include_gap_analysis: bool = Field(
        default=True,
        description="Include detailed gap analysis against EUDR requirements",
    )

class EquivalenceGapEntry(GreenLangBase):
    """A gap identified in EUDR equivalence analysis."""

    requirement: str = Field(
        ..., description="EUDR requirement not fully covered"
    )
    eudr_article: str = Field(
        ..., description="Applicable EUDR article"
    )
    gap_severity: str = Field(
        default="medium", description="Gap severity (critical, major, minor)"
    )
    description: str = Field(
        ..., description="Detailed gap description"
    )
    remediation_suggestion: Optional[str] = Field(
        None, description="Suggested remediation"
    )

class EUDREquivalenceResponse(GreenLangBase):
    """Response for EUDR equivalence check."""

    model_config = ConfigDict(populate_by_name=True)

    certification_type: CertificationTypeEnum = Field(
        ..., description="Assessed certification type"
    )
    equivalence_result: EquivalenceResultEnum = Field(
        ..., description="Equivalence assessment result"
    )
    equivalence_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Equivalence score (0-100)",
    )
    commodities_assessed: List[EUDRCommodityEnum] = Field(
        default_factory=list, description="Commodities assessed"
    )
    requirements_met: int = Field(
        default=0, ge=0, description="EUDR requirements met"
    )
    requirements_total: int = Field(
        default=0, ge=0, description="Total EUDR requirements"
    )
    gaps: List[EquivalenceGapEntry] = Field(
        default_factory=list, description="Identified gaps"
    )
    regulatory_reference: str = Field(
        default="EU 2023/1115 Article 4, Article 10",
        description="Applicable regulatory reference",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

# =============================================================================
# 4. Red Flag Schemas
# =============================================================================

class RedFlagDetectRequest(GreenLangBase):
    """Request to detect red flags for an entity."""

    model_config = ConfigDict(populate_by_name=True)

    operator_id: Optional[str] = Field(
        None, description="Operator ID to screen"
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier ID to screen"
    )
    document_ids: Optional[List[str]] = Field(
        None, description="Document IDs to screen"
    )
    certification_ids: Optional[List[str]] = Field(
        None, description="Certification IDs to screen"
    )
    country_code: Optional[str] = Field(
        None, description="Country to check for regulatory risks"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity to check"
    )
    categories: Optional[List[RedFlagCategoryEnum]] = Field(
        None, description="Specific red flag categories to check"
    )
    include_cross_references: bool = Field(
        default=True,
        description="Cross-reference with external databases (sanctions, etc.)",
    )

class RedFlagEntry(GreenLangBase):
    """Summary of a detected red flag."""

    flag_id: str = Field(..., description="Unique red flag identifier")
    category: RedFlagCategoryEnum = Field(
        ..., description="Red flag category"
    )
    severity: RedFlagSeverityEnum = Field(
        ..., description="Red flag severity"
    )
    status: RedFlagStatusEnum = Field(
        ..., description="Red flag status"
    )
    title: str = Field(..., description="Red flag title")
    description: str = Field(..., description="Red flag description")
    entity_type: str = Field(
        ..., description="Entity type (operator, supplier, document, etc.)"
    )
    entity_id: str = Field(..., description="Entity ID that triggered the flag")
    country_code: Optional[str] = Field(
        None, description="Associated country"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Associated commodity"
    )
    regulatory_reference: Optional[str] = Field(
        None, description="Applicable regulation article"
    )
    detected_at: datetime = Field(
        default_factory=utcnow, description="Detection timestamp"
    )
    suppressed: bool = Field(
        default=False, description="Whether suppressed as false positive"
    )

class RedFlagDetectResponse(GreenLangBase):
    """Response for red flag detection."""

    model_config = ConfigDict(populate_by_name=True)

    red_flags: List[RedFlagEntry] = Field(
        default_factory=list, description="Detected red flags"
    )
    total_flags: int = Field(
        default=0, ge=0, description="Total red flags detected"
    )
    critical_count: int = Field(
        default=0, ge=0, description="Critical severity count"
    )
    high_count: int = Field(
        default=0, ge=0, description="High severity count"
    )
    requires_immediate_action: bool = Field(
        default=False,
        description="Whether any flag requires immediate action",
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class RedFlagListResponse(GreenLangBase):
    """Paginated list of red flags."""

    model_config = ConfigDict(populate_by_name=True)

    red_flags: List[RedFlagEntry] = Field(
        default_factory=list, description="Red flag records"
    )
    total_flags: int = Field(
        default=0, ge=0, description="Total flags"
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

class RedFlagDetailResponse(GreenLangBase):
    """Detailed response for a single red flag."""

    model_config = ConfigDict(populate_by_name=True)

    red_flag: RedFlagEntry = Field(
        ..., description="Red flag details"
    )
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list, description="Supporting evidence"
    )
    related_flags: List[str] = Field(
        default_factory=list, description="Related red flag IDs"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list, description="Flag audit trail"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class RedFlagSuppressRequest(GreenLangBase):
    """Request to suppress a red flag as false positive."""

    model_config = ConfigDict(populate_by_name=True)

    reason: str = Field(
        ..., min_length=10, max_length=2000,
        description="Reason for suppression (minimum 10 characters)",
    )
    evidence_urls: Optional[List[str]] = Field(
        None, description="URLs to supporting evidence for suppression"
    )
    reviewed_by: Optional[str] = Field(
        None, description="Reviewer identifier"
    )

class RedFlagSuppressResponse(GreenLangBase):
    """Response for red flag suppression."""

    model_config = ConfigDict(populate_by_name=True)

    flag_id: str = Field(..., description="Suppressed red flag ID")
    previous_status: RedFlagStatusEnum = Field(
        ..., description="Previous status"
    )
    new_status: RedFlagStatusEnum = Field(
        ..., description="New status (suppressed/false_positive)"
    )
    suppressed_by: str = Field(
        ..., description="User who suppressed the flag"
    )
    suppressed_at: datetime = Field(
        default_factory=utcnow, description="Suppression timestamp"
    )
    reason: str = Field(..., description="Suppression reason")
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

# =============================================================================
# 5. Compliance Assessment Schemas
# =============================================================================

class ComplianceAssessRequest(GreenLangBase):
    """Request to perform a full compliance assessment."""

    model_config = ConfigDict(populate_by_name=True)

    operator_id: Optional[str] = Field(
        None, description="Operator to assess"
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier to assess"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity context for assessment"
    )
    country_code: Optional[str] = Field(
        None, description="Country context"
    )
    include_documents: bool = Field(
        default=True, description="Include document compliance check"
    )
    include_certifications: bool = Field(
        default=True, description="Include certification compliance check"
    )
    include_red_flags: bool = Field(
        default=True, description="Include red flag screening"
    )
    include_recommendations: bool = Field(
        default=True, description="Include compliance recommendations"
    )
    assessment_scope: Optional[List[ComplianceCategoryEnum]] = Field(
        None, description="Specific categories to assess (default: all)"
    )
    framework_ids: Optional[List[str]] = Field(
        None, description="Specific frameworks to assess against"
    )

class CategoryAssessmentEntry(GreenLangBase):
    """Assessment result for a single compliance category."""

    category: ComplianceCategoryEnum = Field(
        ..., description="Compliance category"
    )
    outcome: ComplianceOutcomeEnum = Field(
        ..., description="Category assessment outcome"
    )
    score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Category compliance score (0-100)",
    )
    requirements_met: int = Field(
        default=0, ge=0, description="Requirements met"
    )
    requirements_total: int = Field(
        default=0, ge=0, description="Total requirements"
    )
    issues: List[str] = Field(
        default_factory=list, description="Identified issues"
    )
    regulatory_reference: Optional[str] = Field(
        None, description="Applicable article"
    )

class ComplianceRecommendationEntry(GreenLangBase):
    """A compliance recommendation."""

    recommendation_id: str = Field(
        default_factory=_new_id, description="Recommendation ID"
    )
    category: ComplianceCategoryEnum = Field(
        ..., description="Related compliance category"
    )
    priority: str = Field(
        default="medium", description="Priority (critical, high, medium, low)"
    )
    description: str = Field(
        ..., max_length=2000, description="Recommendation description"
    )
    regulatory_reference: Optional[str] = Field(
        None, description="Applicable EUDR article"
    )
    estimated_effort_days: Optional[int] = Field(
        None, description="Estimated implementation effort in days"
    )

class ComplianceAssessResponse(GreenLangBase):
    """Response for full compliance assessment."""

    model_config = ConfigDict(populate_by_name=True)

    assessment_id: str = Field(
        default_factory=_new_id, description="Assessment identifier"
    )
    overall_outcome: ComplianceOutcomeEnum = Field(
        ..., description="Overall compliance outcome"
    )
    overall_score: Decimal = Field(
        ..., ge=Decimal("0"), le=Decimal("100"),
        description="Overall compliance score (0-100)",
    )
    operator_id: Optional[str] = Field(
        None, description="Assessed operator"
    )
    supplier_id: Optional[str] = Field(
        None, description="Assessed supplier"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity context"
    )
    country_code: Optional[str] = Field(
        None, description="Country context"
    )
    category_assessments: List[CategoryAssessmentEntry] = Field(
        default_factory=list, description="Per-category assessments"
    )
    total_requirements_met: int = Field(
        default=0, ge=0, description="Total requirements met"
    )
    total_requirements: int = Field(
        default=0, ge=0, description="Total requirements"
    )
    red_flags_found: int = Field(
        default=0, ge=0, description="Red flags detected during assessment"
    )
    recommendations: List[ComplianceRecommendationEntry] = Field(
        default_factory=list, description="Recommendations"
    )
    frameworks_assessed: List[str] = Field(
        default_factory=list, description="Framework IDs assessed against"
    )
    assessed_at: datetime = Field(
        default_factory=utcnow, description="Assessment timestamp"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class CategoryCheckRequest(GreenLangBase):
    """Request to check a single compliance category."""

    model_config = ConfigDict(populate_by_name=True)

    category: ComplianceCategoryEnum = Field(
        ..., description="Category to check"
    )
    operator_id: Optional[str] = Field(
        None, description="Operator to check"
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier to check"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity context"
    )
    country_code: Optional[str] = Field(
        None, description="Country context"
    )

class CategoryCheckResponse(GreenLangBase):
    """Response for single-category compliance check."""

    model_config = ConfigDict(populate_by_name=True)

    category_assessment: CategoryAssessmentEntry = Field(
        ..., description="Category assessment result"
    )
    recommendations: List[ComplianceRecommendationEntry] = Field(
        default_factory=list, description="Category-specific recommendations"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class ComplianceListEntry(GreenLangBase):
    """Summary entry in compliance assessment list."""

    assessment_id: str = Field(..., description="Assessment ID")
    overall_outcome: ComplianceOutcomeEnum = Field(
        ..., description="Overall outcome"
    )
    overall_score: Decimal = Field(
        ..., description="Overall score"
    )
    operator_id: Optional[str] = Field(None, description="Operator ID")
    supplier_id: Optional[str] = Field(None, description="Supplier ID")
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity"
    )
    assessed_at: datetime = Field(
        ..., description="Assessment timestamp"
    )

class ComplianceListResponse(GreenLangBase):
    """Paginated list of compliance assessments."""

    model_config = ConfigDict(populate_by_name=True)

    assessments: List[ComplianceListEntry] = Field(
        default_factory=list, description="Assessment records"
    )
    total_assessments: int = Field(
        default=0, ge=0, description="Total assessments"
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

class ComplianceDetailResponse(GreenLangBase):
    """Detailed response for a single compliance assessment."""

    model_config = ConfigDict(populate_by_name=True)

    assessment_id: str = Field(..., description="Assessment ID")
    overall_outcome: ComplianceOutcomeEnum = Field(
        ..., description="Overall outcome"
    )
    overall_score: Decimal = Field(
        ..., description="Overall score"
    )
    operator_id: Optional[str] = Field(None, description="Operator")
    supplier_id: Optional[str] = Field(None, description="Supplier")
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity"
    )
    country_code: Optional[str] = Field(None, description="Country")
    category_assessments: List[CategoryAssessmentEntry] = Field(
        default_factory=list, description="Per-category assessments"
    )
    recommendations: List[ComplianceRecommendationEntry] = Field(
        default_factory=list, description="Recommendations"
    )
    red_flags_found: int = Field(default=0, ge=0, description="Red flags")
    frameworks_assessed: List[str] = Field(
        default_factory=list, description="Frameworks"
    )
    assessed_at: datetime = Field(
        ..., description="Assessment timestamp"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class ComplianceHistoryEntry(GreenLangBase):
    """A single entry in compliance assessment history."""

    assessment_id: str = Field(..., description="Assessment ID")
    overall_outcome: ComplianceOutcomeEnum = Field(
        ..., description="Outcome at time of assessment"
    )
    overall_score: Decimal = Field(
        ..., description="Score at time of assessment"
    )
    assessed_at: datetime = Field(
        ..., description="Assessment timestamp"
    )
    change_from_previous: Optional[Decimal] = Field(
        None, description="Score change from previous assessment"
    )

class ComplianceHistoryResponse(GreenLangBase):
    """Response for compliance assessment history."""

    model_config = ConfigDict(populate_by_name=True)

    assessment_id: str = Field(
        ..., description="Current assessment ID"
    )
    history: List[ComplianceHistoryEntry] = Field(
        default_factory=list, description="Assessment history"
    )
    total_assessments: int = Field(
        default=0, ge=0, description="Total historical assessments"
    )
    trend: Optional[str] = Field(
        None, description="Trend direction (improving, declining, stable)"
    )
    average_score: Optional[Decimal] = Field(
        None, description="Average historical score"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

# =============================================================================
# 6. Audit Integration Schemas
# =============================================================================

class AuditIngestRequest(GreenLangBase):
    """Request to ingest an audit report."""

    model_config = ConfigDict(populate_by_name=True)

    audit_type: AuditTypeEnum = Field(
        ..., description="Type of audit report"
    )
    auditor_name: str = Field(
        ..., min_length=1, max_length=500,
        description="Name of auditor or auditing firm",
    )
    auditor_accreditation: Optional[str] = Field(
        None, max_length=500,
        description="Auditor accreditation reference",
    )
    audit_date: date = Field(
        ..., description="Date of the audit"
    )
    operator_id: Optional[str] = Field(
        None, description="Audited operator ID"
    )
    supplier_id: Optional[str] = Field(
        None, description="Audited supplier ID"
    )
    scope: str = Field(
        ..., min_length=1, max_length=5000,
        description="Audit scope description",
    )
    commodities: Optional[List[EUDRCommodityEnum]] = Field(
        None, description="Commodities covered"
    )
    country_code: Optional[str] = Field(
        None, description="Country of audited entity"
    )
    findings_summary: Optional[str] = Field(
        None, max_length=10000,
        description="Summary of audit findings",
    )
    file_url: Optional[str] = Field(
        None, max_length=2000,
        description="URL to the audit report document",
    )
    file_hash: Optional[str] = Field(
        None, description="SHA-256 hash of audit report file"
    )

class AuditEntry(GreenLangBase):
    """Summary of an ingested audit report."""

    audit_id: str = Field(..., description="Unique audit record ID")
    audit_type: AuditTypeEnum = Field(
        ..., description="Audit type"
    )
    auditor_name: str = Field(
        ..., description="Auditor name"
    )
    audit_date: date = Field(
        ..., description="Audit date"
    )
    status: AuditStatusEnum = Field(
        ..., description="Audit status"
    )
    operator_id: Optional[str] = Field(None, description="Operator ID")
    supplier_id: Optional[str] = Field(None, description="Supplier ID")
    country_code: Optional[str] = Field(None, description="Country code")
    total_findings: int = Field(
        default=0, ge=0, description="Total findings"
    )
    major_findings: int = Field(
        default=0, ge=0, description="Major non-conformities"
    )
    minor_findings: int = Field(
        default=0, ge=0, description="Minor non-conformities"
    )
    corrective_actions_pending: int = Field(
        default=0, ge=0, description="Pending corrective actions"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Ingestion timestamp"
    )

class AuditIngestResponse(GreenLangBase):
    """Response for audit report ingestion."""

    model_config = ConfigDict(populate_by_name=True)

    audit: AuditEntry = Field(
        ..., description="Ingested audit record"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class AuditListResponse(GreenLangBase):
    """Paginated list of audit reports."""

    model_config = ConfigDict(populate_by_name=True)

    audits: List[AuditEntry] = Field(
        default_factory=list, description="Audit records"
    )
    total_audits: int = Field(
        default=0, ge=0, description="Total audits"
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

class FindingEntry(GreenLangBase):
    """A single audit finding."""

    finding_id: str = Field(
        default_factory=_new_id, description="Finding identifier"
    )
    severity: FindingSeverityEnum = Field(
        ..., description="Finding severity"
    )
    category: Optional[ComplianceCategoryEnum] = Field(
        None, description="Related compliance category"
    )
    title: str = Field(..., description="Finding title")
    description: str = Field(
        ..., max_length=5000, description="Finding description"
    )
    regulatory_reference: Optional[str] = Field(
        None, description="EUDR article reference"
    )
    corrective_action_required: bool = Field(
        default=False, description="Whether corrective action required"
    )
    corrective_action_deadline: Optional[date] = Field(
        None, description="Corrective action deadline"
    )
    corrective_action_status: Optional[CorrectiveActionStatusEnum] = Field(
        None, description="Corrective action status"
    )

class AuditFindingsResponse(GreenLangBase):
    """Response for audit findings."""

    model_config = ConfigDict(populate_by_name=True)

    audit_id: str = Field(..., description="Audit record ID")
    findings: List[FindingEntry] = Field(
        default_factory=list, description="Audit findings"
    )
    total_findings: int = Field(
        default=0, ge=0, description="Total findings"
    )
    major_count: int = Field(
        default=0, ge=0, description="Major non-conformities"
    )
    minor_count: int = Field(
        default=0, ge=0, description="Minor non-conformities"
    )
    observations_count: int = Field(
        default=0, ge=0, description="Observations"
    )
    corrective_actions_required: int = Field(
        default=0, ge=0, description="Corrective actions required"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class CorrectiveActionEntry(GreenLangBase):
    """A corrective action for an audit finding."""

    action_id: str = Field(
        default_factory=_new_id, description="Action identifier"
    )
    finding_id: str = Field(..., description="Related finding ID")
    description: str = Field(
        ..., max_length=5000, description="Action description"
    )
    responsible_party: str = Field(
        ..., description="Responsible party"
    )
    deadline: date = Field(..., description="Action deadline")
    status: CorrectiveActionStatusEnum = Field(
        default=CorrectiveActionStatusEnum.OPEN, description="Action status"
    )
    completion_date: Optional[date] = Field(
        None, description="Actual completion date"
    )
    evidence_urls: List[str] = Field(
        default_factory=list, description="Evidence of completion"
    )
    notes: Optional[str] = Field(
        None, max_length=5000, description="Action notes"
    )

class CorrectiveActionsRequest(GreenLangBase):
    """Request to update corrective actions for an audit."""

    model_config = ConfigDict(populate_by_name=True)

    actions: List[CorrectiveActionEntry] = Field(
        ..., min_length=1, max_length=50,
        description="Corrective actions to create/update (max 50)",
    )

class CorrectiveActionsResponse(GreenLangBase):
    """Response for corrective actions update."""

    model_config = ConfigDict(populate_by_name=True)

    audit_id: str = Field(..., description="Audit record ID")
    actions: List[CorrectiveActionEntry] = Field(
        default_factory=list, description="Updated corrective actions"
    )
    total_actions: int = Field(
        default=0, ge=0, description="Total actions"
    )
    open_count: int = Field(
        default=0, ge=0, description="Open actions"
    )
    in_progress_count: int = Field(
        default=0, ge=0, description="In-progress actions"
    )
    completed_count: int = Field(
        default=0, ge=0, description="Completed actions"
    )
    overdue_count: int = Field(
        default=0, ge=0, description="Overdue actions"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

# =============================================================================
# 7. Reporting Schemas
# =============================================================================

class ReportGenerateRequest(GreenLangBase):
    """Request to generate a compliance report."""

    model_config = ConfigDict(populate_by_name=True)

    report_type: ReportTypeEnum = Field(
        ..., description="Type of report to generate"
    )
    format: ReportFormatEnum = Field(
        default=ReportFormatEnum.PDF,
        description="Output format",
    )
    operator_id: Optional[str] = Field(
        None, description="Operator to report on"
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier to report on"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity filter"
    )
    country_code: Optional[str] = Field(
        None, description="Country filter"
    )
    date_from: Optional[date] = Field(
        None, description="Report period start date"
    )
    date_to: Optional[date] = Field(
        None, description="Report period end date"
    )
    include_details: bool = Field(
        default=True, description="Include detailed breakdown"
    )
    title: Optional[str] = Field(
        None, max_length=500, description="Custom report title"
    )

    @field_validator("date_to")
    @classmethod
    def validate_date_range(cls, v: Optional[date], info: Any) -> Optional[date]:
        """Validate date_to >= date_from when both provided."""
        if v is not None and info.data.get("date_from") is not None:
            if v < info.data["date_from"]:
                raise ValueError("date_to must be >= date_from")
        return v

class ReportEntry(GreenLangBase):
    """Summary of a generated report."""

    report_id: str = Field(..., description="Unique report identifier")
    report_type: ReportTypeEnum = Field(
        ..., description="Report type"
    )
    format: ReportFormatEnum = Field(
        ..., description="Report format"
    )
    status: ReportStatusEnum = Field(
        ..., description="Report generation status"
    )
    title: Optional[str] = Field(None, description="Report title")
    operator_id: Optional[str] = Field(None, description="Operator")
    supplier_id: Optional[str] = Field(None, description="Supplier")
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="Generated file size"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Completion timestamp"
    )

class ReportGenerateResponse(GreenLangBase):
    """Response for report generation."""

    model_config = ConfigDict(populate_by_name=True)

    report: ReportEntry = Field(
        ..., description="Generated report record"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class ReportListResponse(GreenLangBase):
    """Paginated list of reports."""

    model_config = ConfigDict(populate_by_name=True)

    reports: List[ReportEntry] = Field(
        default_factory=list, description="Report records"
    )
    total_reports: int = Field(
        default=0, ge=0, description="Total reports"
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

class ReportDownloadResponse(GreenLangBase):
    """Response for report download."""

    model_config = ConfigDict(populate_by_name=True)

    report_id: str = Field(..., description="Report ID")
    download_url: str = Field(
        ..., description="Pre-signed download URL"
    )
    expires_at: datetime = Field(
        ..., description="URL expiration timestamp"
    )
    format: ReportFormatEnum = Field(
        ..., description="Report format"
    )
    file_size_bytes: Optional[int] = Field(
        None, description="File size in bytes"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class ReportScheduleRequest(GreenLangBase):
    """Request to schedule a recurring report."""

    model_config = ConfigDict(populate_by_name=True)

    report_type: ReportTypeEnum = Field(
        ..., description="Type of report to schedule"
    )
    format: ReportFormatEnum = Field(
        default=ReportFormatEnum.PDF, description="Output format"
    )
    frequency: ScheduleFrequencyEnum = Field(
        ..., description="Report generation frequency"
    )
    operator_id: Optional[str] = Field(
        None, description="Operator filter"
    )
    supplier_id: Optional[str] = Field(
        None, description="Supplier filter"
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity filter"
    )
    recipients: List[str] = Field(
        ..., min_length=1, max_length=20,
        description="Email recipients (max 20)",
    )
    title: Optional[str] = Field(
        None, max_length=500, description="Custom report title"
    )
    start_date: Optional[date] = Field(
        None, description="Schedule start date (default: today)"
    )
    end_date: Optional[date] = Field(
        None, description="Schedule end date (default: indefinite)"
    )

class ScheduleEntry(GreenLangBase):
    """Summary of a report schedule."""

    schedule_id: str = Field(..., description="Schedule identifier")
    report_type: ReportTypeEnum = Field(
        ..., description="Report type"
    )
    frequency: ScheduleFrequencyEnum = Field(
        ..., description="Generation frequency"
    )
    format: ReportFormatEnum = Field(
        ..., description="Output format"
    )
    recipients: List[str] = Field(
        default_factory=list, description="Email recipients"
    )
    next_run: Optional[datetime] = Field(
        None, description="Next scheduled generation"
    )
    active: bool = Field(default=True, description="Whether schedule is active")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )

class ReportScheduleResponse(GreenLangBase):
    """Response for report schedule creation."""

    model_config = ConfigDict(populate_by_name=True)

    schedule: ScheduleEntry = Field(
        ..., description="Created schedule"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

# =============================================================================
# 8. Batch Processing Schemas
# =============================================================================

class BatchAssessRequest(GreenLangBase):
    """Request for batch compliance assessment."""

    model_config = ConfigDict(populate_by_name=True)

    entity_ids: List[str] = Field(
        ..., min_length=1, max_length=500,
        description="Operator or supplier IDs to assess (max 500)",
    )
    entity_type: str = Field(
        default="supplier",
        description="Entity type: 'operator' or 'supplier'",
    )
    commodity: Optional[EUDRCommodityEnum] = Field(
        None, description="Commodity context"
    )
    assessment_scope: Optional[List[ComplianceCategoryEnum]] = Field(
        None, description="Categories to assess (default: all)"
    )

class BatchAssessResultEntry(GreenLangBase):
    """Result for a single entity in batch assessment."""

    entity_id: str = Field(..., description="Entity ID")
    assessment_id: Optional[str] = Field(
        None, description="Generated assessment ID"
    )
    overall_outcome: Optional[ComplianceOutcomeEnum] = Field(
        None, description="Assessment outcome"
    )
    overall_score: Optional[Decimal] = Field(
        None, description="Assessment score"
    )
    error: Optional[str] = Field(
        None, description="Error message if assessment failed"
    )

class BatchAssessResponse(GreenLangBase):
    """Response for batch compliance assessment."""

    model_config = ConfigDict(populate_by_name=True)

    batch_id: str = Field(
        default_factory=_new_id, description="Batch identifier"
    )
    status: BatchStatusEnum = Field(
        ..., description="Batch processing status"
    )
    total_submitted: int = Field(
        default=0, ge=0, description="Total entities submitted"
    )
    completed_count: int = Field(
        default=0, ge=0, description="Assessments completed"
    )
    failed_count: int = Field(
        default=0, ge=0, description="Assessments failed"
    )
    results: List[BatchAssessResultEntry] = Field(
        default_factory=list, description="Per-entity results"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class BatchVerifyRequest(GreenLangBase):
    """Request for batch document verification."""

    model_config = ConfigDict(populate_by_name=True)

    documents: List[DocumentVerifyRequest] = Field(
        ..., min_length=1, max_length=200,
        description="Documents to verify (max 200)",
    )

class BatchVerifyResultEntry(GreenLangBase):
    """Result for a single document in batch verification."""

    document_reference: str = Field(
        ..., description="Document reference"
    )
    document_id: Optional[str] = Field(
        None, description="Generated document ID"
    )
    status: Optional[DocumentStatusEnum] = Field(
        None, description="Verification status"
    )
    verification_score: Optional[Decimal] = Field(
        None, description="Verification score"
    )
    red_flags_detected: int = Field(
        default=0, ge=0, description="Red flags detected"
    )
    error: Optional[str] = Field(
        None, description="Error message if verification failed"
    )

class BatchVerifyResponse(GreenLangBase):
    """Response for batch document verification."""

    model_config = ConfigDict(populate_by_name=True)

    batch_id: str = Field(
        default_factory=_new_id, description="Batch identifier"
    )
    status: BatchStatusEnum = Field(
        ..., description="Batch processing status"
    )
    total_submitted: int = Field(
        default=0, ge=0, description="Total documents submitted"
    )
    verified_count: int = Field(
        default=0, ge=0, description="Documents verified"
    )
    failed_count: int = Field(
        default=0, ge=0, description="Verifications failed"
    )
    total_red_flags: int = Field(
        default=0, ge=0, description="Total red flags detected"
    )
    results: List[BatchVerifyResultEntry] = Field(
        default_factory=list, description="Per-document results"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking information"
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema, description="Response metadata"
    )

class BatchStatusResponse(GreenLangBase):
    """Response for batch processing status."""

    model_config = ConfigDict(populate_by_name=True)

    batch_id: str = Field(..., description="Batch identifier")
    status: BatchStatusEnum = Field(
        ..., description="Current batch status"
    )
    batch_type: str = Field(
        ..., description="Batch type (assess or verify)"
    )
    total_submitted: int = Field(
        default=0, ge=0, description="Total items submitted"
    )
    completed_count: int = Field(
        default=0, ge=0, description="Items completed"
    )
    failed_count: int = Field(
        default=0, ge=0, description="Items failed"
    )
    progress_percent: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"), le=Decimal("100"),
        description="Processing progress (0-100)",
    )
    started_at: Optional[datetime] = Field(
        None, description="Processing start time"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Processing completion time"
    )
    estimated_remaining_seconds: Optional[int] = Field(
        None, description="Estimated seconds remaining"
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
    "JurisdictionTypeEnum",
    "FrameworkStatusEnum",
    "DocumentTypeEnum",
    "DocumentStatusEnum",
    "CertificationTypeEnum",
    "CertStatusEnum",
    "EquivalenceResultEnum",
    "RedFlagSeverityEnum",
    "RedFlagCategoryEnum",
    "RedFlagStatusEnum",
    "ComplianceOutcomeEnum",
    "ComplianceCategoryEnum",
    "AuditTypeEnum",
    "AuditStatusEnum",
    "FindingSeverityEnum",
    "CorrectiveActionStatusEnum",
    "ReportTypeEnum",
    "ReportFormatEnum",
    "ReportStatusEnum",
    "ScheduleFrequencyEnum",
    "BatchStatusEnum",
    "EUDRCommodityEnum",
    # Common
    "ProvenanceInfo",
    "MetadataSchema",
    "PaginatedMeta",
    "ErrorResponse",
    "HealthResponse",
    # Framework
    "FrameworkRegisterRequest",
    "FrameworkEntry",
    "FrameworkRegisterResponse",
    "FrameworkDetailResponse",
    "FrameworkUpdateRequest",
    "FrameworkListResponse",
    "FrameworkSearchRequest",
    "FrameworkSearchResponse",
    # Document
    "DocumentVerifyRequest",
    "DocumentEntry",
    "DocumentVerifyResponse",
    "DocumentListResponse",
    "DocumentDetailResponse",
    "ValidityCheckRequest",
    "DocumentValidityEntry",
    "ValidityCheckResponse",
    "ExpiringDocumentEntry",
    "ExpiringDocumentsResponse",
    # Certification
    "CertValidateRequest",
    "CertificationEntry",
    "CertValidateResponse",
    "CertListResponse",
    "CertDetailResponse",
    "EUDREquivalenceRequest",
    "EquivalenceGapEntry",
    "EUDREquivalenceResponse",
    # Red Flag
    "RedFlagDetectRequest",
    "RedFlagEntry",
    "RedFlagDetectResponse",
    "RedFlagListResponse",
    "RedFlagDetailResponse",
    "RedFlagSuppressRequest",
    "RedFlagSuppressResponse",
    # Compliance
    "ComplianceAssessRequest",
    "CategoryAssessmentEntry",
    "ComplianceRecommendationEntry",
    "ComplianceAssessResponse",
    "CategoryCheckRequest",
    "CategoryCheckResponse",
    "ComplianceListEntry",
    "ComplianceListResponse",
    "ComplianceDetailResponse",
    "ComplianceHistoryEntry",
    "ComplianceHistoryResponse",
    # Audit
    "AuditIngestRequest",
    "AuditEntry",
    "AuditIngestResponse",
    "AuditListResponse",
    "FindingEntry",
    "AuditFindingsResponse",
    "CorrectiveActionEntry",
    "CorrectiveActionsRequest",
    "CorrectiveActionsResponse",
    # Report
    "ReportGenerateRequest",
    "ReportEntry",
    "ReportGenerateResponse",
    "ReportListResponse",
    "ReportDownloadResponse",
    "ReportScheduleRequest",
    "ScheduleEntry",
    "ReportScheduleResponse",
    # Batch
    "BatchAssessRequest",
    "BatchAssessResultEntry",
    "BatchAssessResponse",
    "BatchVerifyRequest",
    "BatchVerifyResultEntry",
    "BatchVerifyResponse",
    "BatchStatusResponse",
]
