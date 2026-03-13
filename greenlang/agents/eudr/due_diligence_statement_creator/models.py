# -*- coding: utf-8 -*-
"""
Due Diligence Statement Creator Models - AGENT-EUDR-037

Pydantic v2 models for Due Diligence Statement creation, assembly,
validation, geolocation formatting, risk data integration, supply chain
compilation, compliance verification, document packaging, version
control, digital signing, and EU Information System submission.

All models use Decimal for numeric values to ensure deterministic,
bit-perfect reproducibility in compliance data submissions.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-037 Due Diligence Statement Creator (GL-EUDR-DDSC-037)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 14, 31, 33
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums (12)
# ---------------------------------------------------------------------------


class DDSStatus(str, enum.Enum):
    """Due Diligence Statement lifecycle status."""

    DRAFT = "draft"
    ASSEMBLING = "assembling"
    ASSEMBLED = "assembled"
    VALIDATING = "validating"
    VALIDATED = "validated"
    SIGNING = "signing"
    SIGNED = "signed"
    PACKAGING = "packaging"
    PACKAGED = "packaged"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMENDED = "amended"
    WITHDRAWN = "withdrawn"


class CommodityType(str, enum.Enum):
    """EUDR regulated commodities per Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class RiskLevel(str, enum.Enum):
    """Risk classification levels per EUDR Article 29."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, enum.Enum):
    """Overall compliance status of a DDS."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    PENDING = "pending"


class DocumentType(str, enum.Enum):
    """Types of supporting documents in a DDS package."""

    CERTIFICATE_OF_ORIGIN = "certificate_of_origin"
    GEOLOCATION_DATA = "geolocation_data"
    SATELLITE_IMAGERY = "satellite_imagery"
    RISK_ASSESSMENT_REPORT = "risk_assessment_report"
    SUPPLY_CHAIN_MAP = "supply_chain_map"
    CUSTOMS_DECLARATION = "customs_declaration"
    PHYTOSANITARY_CERTIFICATE = "phytosanitary_certificate"
    CERTIFICATION_SCHEME = "certification_scheme"
    AUDIT_REPORT = "audit_report"
    STAKEHOLDER_CONSULTATION = "stakeholder_consultation"
    FPIC_DOCUMENTATION = "fpic_documentation"
    LEGAL_COMPLIANCE_PROOF = "legal_compliance_proof"
    DEFORESTATION_FREE_EVIDENCE = "deforestation_free_evidence"
    OTHER = "other"


class SignatureType(str, enum.Enum):
    """Digital signature types for DDS authentication."""

    SIMPLE_ELECTRONIC = "simple_electronic"
    ADVANCED_ELECTRONIC = "advanced_electronic"
    QUALIFIED_ELECTRONIC = "qualified_electronic"


class ValidationResult(str, enum.Enum):
    """Outcome of a DDS validation check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


class SubmissionStatus(str, enum.Enum):
    """Status of DDS submission to EU Information System."""

    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AmendmentReason(str, enum.Enum):
    """Reasons for amending a submitted DDS."""

    CORRECTION_OF_ERROR = "correction_of_error"
    ADDITIONAL_INFORMATION = "additional_information"
    UPDATED_RISK_ASSESSMENT = "updated_risk_assessment"
    COMPETENT_AUTHORITY_REQUEST = "competent_authority_request"
    SUPPLY_CHAIN_CHANGE = "supply_chain_change"
    GEOLOCATION_UPDATE = "geolocation_update"
    QUANTITY_ADJUSTMENT = "quantity_adjustment"
    REGULATORY_CHANGE = "regulatory_change"


class GeolocationMethod(str, enum.Enum):
    """Methods used for geolocation data capture."""

    GPS_FIELD_SURVEY = "gps_field_survey"
    SATELLITE_DERIVED = "satellite_derived"
    CADASTRAL_MAP = "cadastral_map"
    DRONE_SURVEY = "drone_survey"
    REMOTE_SENSING = "remote_sensing"
    SELF_REPORTED = "self_reported"
    VERIFIED_BY_THIRD_PARTY = "verified_by_third_party"


class LanguageCode(str, enum.Enum):
    """EU official languages for DDS translation."""

    BG = "bg"
    CS = "cs"
    DA = "da"
    DE = "de"
    EL = "el"
    EN = "en"
    ES = "es"
    ET = "et"
    FI = "fi"
    FR = "fr"
    GA = "ga"
    HR = "hr"
    HU = "hu"
    IT = "it"
    LT = "lt"
    LV = "lv"
    MT = "mt"
    NL = "nl"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    SK = "sk"
    SL = "sl"
    SV = "sv"


class StatementType(str, enum.Enum):
    """Type of DDS per EUDR Article 4."""

    PLACING = "placing"
    MAKING_AVAILABLE = "making_available"
    EXPORT = "export"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-DDSC-037"
AGENT_VERSION = "1.0.0"

EUDR_REGULATED_COMMODITIES: List[CommodityType] = [
    CommodityType.CATTLE,
    CommodityType.COCOA,
    CommodityType.COFFEE,
    CommodityType.OIL_PALM,
    CommodityType.RUBBER,
    CommodityType.SOYA,
    CommodityType.WOOD,
]

ARTICLE_4_MANDATORY_FIELDS: List[str] = [
    "operator_name",
    "operator_address",
    "operator_eori_number",
    "commodity_type",
    "product_description",
    "hs_code",
    "country_of_production",
    "geolocation_of_plots",
    "quantity",
    "supplier_information",
    "compliance_declaration",
    "risk_assessment_outcome",
    "risk_mitigation_measures",
    "date_of_statement",
]

EU_OFFICIAL_LANGUAGES: List[str] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et",
    "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv",
    "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
]


# ---------------------------------------------------------------------------
# Pydantic Models (15+)
# ---------------------------------------------------------------------------


class GeolocationData(BaseModel):
    """Geolocation data for a production plot per Article 9.

    Stores coordinates as latitude/longitude for plots under 4ha,
    or as polygon boundaries for plots 4ha or larger.
    """

    plot_id: str = Field(..., description="Unique plot identifier")
    latitude: Decimal = Field(
        ..., ge=-90, le=90,
        description="Latitude of plot centroid (WGS84)",
    )
    longitude: Decimal = Field(
        ..., ge=-180, le=180,
        description="Longitude of plot centroid (WGS84)",
    )
    area_hectares: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Plot area in hectares",
    )
    polygon_coordinates: List[List[Decimal]] = Field(
        default_factory=list,
        description="Polygon boundary coordinates [[lat,lon],...] for plots >= 4ha",
    )
    country_code: str = Field(
        default="", max_length=3,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: str = Field(default="", description="Sub-national region")
    collection_method: GeolocationMethod = GeolocationMethod.GPS_FIELD_SURVEY
    collection_date: Optional[datetime] = None
    accuracy_meters: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Geolocation accuracy in meters",
    )
    verified: bool = Field(
        default=False, description="Whether geolocation is verified"
    )
    verification_source: str = Field(
        default="", description="Source of verification"
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class RiskReference(BaseModel):
    """Reference to a risk assessment from upstream EUDR agents.

    Captures the outcome of risk evaluations from EUDR-016 through
    EUDR-025 for inclusion in the DDS.
    """

    risk_id: str = Field(..., description="Risk assessment identifier")
    source_agent: str = Field(
        ..., description="Source EUDR agent ID (e.g., EUDR-016)"
    )
    risk_category: str = Field(
        ..., description="Risk category (country, supplier, commodity, etc.)"
    )
    risk_level: RiskLevel = RiskLevel.STANDARD
    risk_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Normalized risk score (0-100)",
    )
    assessment_date: Optional[datetime] = None
    factors: List[str] = Field(
        default_factory=list,
        description="Risk factors identified",
    )
    mitigation_measures: List[str] = Field(
        default_factory=list,
        description="Mitigation measures applied",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used in assessment",
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class SupplyChainData(BaseModel):
    """Aggregated supply chain traceability data from EUDR-001 to 015.

    Provides the complete supply chain picture required for
    DDS section on supplier information and chain of custody.
    """

    supply_chain_id: str = Field(
        ..., description="Supply chain identifier"
    )
    operator_id: str = Field(..., description="EUDR operator identifier")
    commodity: CommodityType
    tier_count: int = Field(
        default=0, ge=0,
        description="Number of supply chain tiers mapped",
    )
    supplier_count: int = Field(
        default=0, ge=0,
        description="Total number of suppliers in chain",
    )
    suppliers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Supplier details with tier, country, status",
    )
    chain_of_custody_model: str = Field(
        default="segregation",
        description="Chain of custody model (segregation/mass_balance/identity_preserved)",
    )
    plot_count: int = Field(
        default=0, ge=0,
        description="Number of production plots in chain",
    )
    countries_of_production: List[str] = Field(
        default_factory=list,
        description="ISO country codes of production countries",
    )
    traceability_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Overall traceability completeness score",
    )
    last_updated: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class ComplianceCheck(BaseModel):
    """Individual compliance check result for Article 4 validation.

    Each mandatory field and regulatory requirement generates a
    separate ComplianceCheck record.
    """

    check_id: str = Field(..., description="Unique check identifier")
    field_name: str = Field(
        ..., description="Field or requirement being checked"
    )
    article_reference: str = Field(
        default="", description="EUDR article reference (e.g., Art. 4(2)(a))"
    )
    result: ValidationResult = ValidationResult.PASS
    message: str = Field(default="", description="Check result message")
    severity: str = Field(
        default="info",
        description="Severity level (info/warning/error/critical)",
    )
    suggested_fix: str = Field(
        default="", description="Suggested remediation"
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class DocumentPackage(BaseModel):
    """A supporting document in the DDS evidence package.

    References certificates, satellite imagery, risk reports, and
    other evidence bundled with the DDS for submission.
    """

    document_id: str = Field(
        ..., description="Unique document identifier"
    )
    document_type: DocumentType
    filename: str = Field(..., description="Original filename")
    mime_type: str = Field(
        default="application/pdf", description="MIME type"
    )
    size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    hash_sha256: str = Field(
        default="", description="SHA-256 hash of file content"
    )
    description: str = Field(default="", description="Document description")
    issuing_authority: str = Field(
        default="", description="Authority that issued the document"
    )
    issue_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    language: str = Field(default="en", description="Document language")
    storage_path: str = Field(default="", description="Storage location")
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class StatementVersion(BaseModel):
    """Version record for DDS amendment tracking.

    Maintains complete version history for regulatory auditability
    per EUDR Article 31 record-keeping requirements.
    """

    version_id: str = Field(..., description="Unique version identifier")
    statement_id: str = Field(
        ..., description="Parent DDS identifier"
    )
    version_number: int = Field(
        ..., ge=1, description="Sequential version number"
    )
    status: DDSStatus = DDSStatus.DRAFT
    amendment_reason: Optional[AmendmentReason] = None
    amendment_description: str = Field(
        default="", description="Description of changes"
    )
    changes_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured summary of changes per section",
    )
    created_by: str = Field(default="", description="User who created version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    supersedes_version: Optional[str] = Field(
        default=None, description="Version ID this supersedes"
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DigitalSignature(BaseModel):
    """Digital signature applied to a DDS.

    Supports simple, advanced, and qualified electronic signatures
    per eIDAS Regulation (EU) 910/2014.
    """

    signature_id: str = Field(
        ..., description="Unique signature identifier"
    )
    statement_id: str = Field(
        ..., description="DDS being signed"
    )
    signer_name: str = Field(..., description="Name of the signer")
    signer_role: str = Field(
        default="", description="Role of the signer"
    )
    signer_organization: str = Field(
        default="", description="Signer's organization"
    )
    signature_type: SignatureType = SignatureType.QUALIFIED_ELECTRONIC
    algorithm: str = Field(
        default="RSA-SHA256", description="Signature algorithm"
    )
    certificate_issuer: str = Field(
        default="", description="Certificate issuing authority"
    )
    certificate_serial: str = Field(
        default="", description="Certificate serial number"
    )
    signature_value: str = Field(
        default="", description="Base64-encoded signature value"
    )
    signed_hash: str = Field(
        default="", description="SHA-256 hash of signed content"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    is_valid: bool = Field(
        default=False, description="Whether signature is currently valid"
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DDSValidationReport(BaseModel):
    """Complete validation report for a DDS.

    Aggregates all individual compliance checks and produces an
    overall pass/fail determination.
    """

    report_id: str = Field(..., description="Unique report identifier")
    statement_id: str = Field(
        ..., description="DDS being validated"
    )
    overall_result: ValidationResult = ValidationResult.PASS
    total_checks: int = Field(default=0, ge=0)
    passed_checks: int = Field(default=0, ge=0)
    failed_checks: int = Field(default=0, ge=0)
    warning_checks: int = Field(default=0, ge=0)
    checks: List[ComplianceCheck] = Field(
        default_factory=list,
        description="Individual compliance check results",
    )
    mandatory_fields_complete: bool = Field(
        default=False,
        description="Whether all Article 4 mandatory fields present",
    )
    geolocation_valid: bool = Field(
        default=False,
        description="Whether geolocation data meets Article 9 requirements",
    )
    risk_assessment_included: bool = Field(
        default=False,
        description="Whether risk assessment data is included",
    )
    supply_chain_complete: bool = Field(
        default=False,
        description="Whether supply chain data is complete",
    )
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class SubmissionPackage(BaseModel):
    """Package prepared for submission to EU Information System.

    Contains the finalized DDS along with all supporting documents,
    digital signature, and metadata required for EU IS acceptance.
    """

    package_id: str = Field(..., description="Unique package identifier")
    statement_id: str = Field(
        ..., description="DDS identifier"
    )
    operator_id: str = Field(
        ..., description="EUDR operator identifier"
    )
    submission_status: SubmissionStatus = SubmissionStatus.PENDING
    eu_is_reference: str = Field(
        default="", description="EU Information System reference number"
    )
    eu_is_receipt_timestamp: Optional[datetime] = None
    document_count: int = Field(
        default=0, ge=0,
        description="Number of documents in the package",
    )
    total_size_bytes: int = Field(
        default=0, ge=0,
        description="Total package size in bytes",
    )
    validation_passed: bool = Field(
        default=False,
        description="Whether pre-submission validation passed",
    )
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    rejection_reason: str = Field(
        default="", description="Reason for rejection if applicable"
    )
    retry_count: int = Field(default=0, ge=0)
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class AmendmentRecord(BaseModel):
    """Record of a DDS amendment operation.

    Tracks the before/after state and rationale for every amendment
    for Article 31 audit trail compliance.
    """

    amendment_id: str = Field(
        ..., description="Unique amendment identifier"
    )
    statement_id: str = Field(
        ..., description="DDS being amended"
    )
    reason: AmendmentReason
    description: str = Field(
        ..., description="Detailed description of amendment"
    )
    previous_version: int = Field(
        ..., ge=1, description="Version number before amendment"
    )
    new_version: int = Field(
        ..., ge=2, description="Version number after amendment"
    )
    changed_fields: List[str] = Field(
        default_factory=list,
        description="List of fields that were changed",
    )
    changed_by: str = Field(
        default="", description="User who made the amendment"
    )
    approved_by: str = Field(
        default="", description="User who approved the amendment"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class TemplateConfig(BaseModel):
    """Configuration for DDS document template rendering.

    Controls the structure, language, and formatting of generated
    DDS documents.
    """

    template_id: str = Field(
        ..., description="Unique template identifier"
    )
    template_version: str = Field(
        default="1.0", description="Template version"
    )
    language: LanguageCode = LanguageCode.EN
    sections: List[str] = Field(
        default_factory=list,
        description="Ordered list of sections to include",
    )
    include_annexes: bool = Field(
        default=True, description="Whether to include annexes"
    )
    include_maps: bool = Field(
        default=True, description="Whether to include geolocation maps"
    )
    output_format: str = Field(
        default="pdf", description="Output format (pdf/html/json)"
    )
    header_logo_path: str = Field(
        default="", description="Path to header logo"
    )
    footer_text: str = Field(
        default="", description="Footer text for each page"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional template metadata",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class LanguageTranslation(BaseModel):
    """Translation record for a DDS field or section.

    Supports rendering DDS in any of the 24 EU official languages.
    """

    translation_id: str = Field(
        ..., description="Unique translation identifier"
    )
    source_language: LanguageCode = LanguageCode.EN
    target_language: LanguageCode
    field_key: str = Field(
        ..., description="Field or section key being translated"
    )
    source_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text")
    translator: str = Field(
        default="system",
        description="Translator identity (system/human/certified)",
    )
    verified: bool = Field(
        default=False, description="Whether translation is verified"
    )
    verified_by: str = Field(default="", description="Verifier identity")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class StatementSummary(BaseModel):
    """Summary view of a DDS for listing and search.

    Lightweight representation of a DDS for dashboard display,
    search results, and batch operations.
    """

    statement_id: str = Field(
        ..., description="Unique statement identifier"
    )
    reference_number: str = Field(
        ..., description="DDS reference number"
    )
    operator_id: str = Field(..., description="Operator identifier")
    operator_name: str = Field(default="", description="Operator name")
    statement_type: StatementType = StatementType.PLACING
    status: DDSStatus = DDSStatus.DRAFT
    commodities: List[CommodityType] = Field(
        default_factory=list,
        description="Commodities covered by this statement",
    )
    countries_of_production: List[str] = Field(
        default_factory=list,
        description="Production countries",
    )
    overall_risk_level: RiskLevel = RiskLevel.STANDARD
    compliance_status: ComplianceStatus = ComplianceStatus.PENDING
    version_number: int = Field(default=1, ge=1)
    document_count: int = Field(default=0, ge=0)
    total_quantity: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total commodity quantity (metric tonnes)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    submitted_at: Optional[datetime] = None
    language: str = Field(default="en", description="Primary language")
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DDSStatement(BaseModel):
    """Complete Due Diligence Statement record.

    The primary data model for AGENT-EUDR-037, representing a full
    DDS with all sections required by EUDR Article 4.
    """

    statement_id: str = Field(
        ..., description="Unique statement identifier"
    )
    reference_number: str = Field(
        ..., description="DDS reference number (GL-DDS-YYYYMMDD-XXXX)"
    )
    operator_id: str = Field(..., description="EUDR operator identifier")
    operator_name: str = Field(..., description="Operator legal name")
    operator_address: str = Field(
        default="", description="Operator registered address"
    )
    operator_eori_number: str = Field(
        default="",
        description="Operator EORI number (Economic Operators Registration)",
    )
    statement_type: StatementType = StatementType.PLACING
    status: DDSStatus = DDSStatus.DRAFT
    version_number: int = Field(default=1, ge=1)

    # Commodity information
    commodities: List[CommodityType] = Field(
        ..., min_length=1,
        description="EUDR regulated commodities covered",
    )
    product_descriptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Product descriptions with HS codes",
    )
    hs_codes: List[str] = Field(
        default_factory=list,
        description="Harmonized System codes",
    )
    total_quantity: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total quantity in metric tonnes",
    )
    quantity_unit: str = Field(
        default="metric_tonnes", description="Unit of measurement"
    )

    # Geolocation
    countries_of_production: List[str] = Field(
        default_factory=list,
        description="ISO 3166-1 country codes of production",
    )
    geolocations: List[GeolocationData] = Field(
        default_factory=list,
        description="Geolocation data per Article 9",
    )

    # Supply chain
    supply_chain_data: Optional[SupplyChainData] = None
    supplier_count: int = Field(default=0, ge=0)

    # Risk assessment
    risk_references: List[RiskReference] = Field(
        default_factory=list,
        description="Risk assessment references from EUDR-016 to 025",
    )
    overall_risk_level: RiskLevel = RiskLevel.STANDARD
    risk_mitigation_measures: List[str] = Field(
        default_factory=list,
        description="Risk mitigation measures applied",
    )

    # Compliance
    compliance_status: ComplianceStatus = ComplianceStatus.PENDING
    compliance_declaration: str = Field(
        default="",
        description="Operator compliance declaration text",
    )
    deforestation_free: bool = Field(
        default=False,
        description="Declaration that products are deforestation-free",
    )
    legally_produced: bool = Field(
        default=False,
        description="Declaration that products are legally produced",
    )

    # Documents
    supporting_documents: List[DocumentPackage] = Field(
        default_factory=list,
        description="Supporting evidence documents",
    )

    # Signature
    digital_signature: Optional[DigitalSignature] = None

    # Versioning
    versions: List[StatementVersion] = Field(
        default_factory=list,
        description="Version history",
    )
    amendments: List[AmendmentRecord] = Field(
        default_factory=list,
        description="Amendment records",
    )

    # Submission
    submission_package: Optional[SubmissionPackage] = None
    eu_is_reference: str = Field(
        default="",
        description="EU Information System reference number",
    )

    # Language
    language: str = Field(default="en", description="Primary language")
    translations: List[LanguageTranslation] = Field(
        default_factory=list,
        description="Available translations",
    )

    # Metadata
    date_of_statement: Optional[datetime] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    submitted_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(BaseModel):
    """Health check response for the DDS Creator."""

    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}
