# -*- coding: utf-8 -*-
"""
Multi-Tier Supplier Tracker Data Models - AGENT-EUDR-008

Pydantic v2 data models for the Multi-Tier Supplier Tracker Agent covering
supplier hierarchy discovery, profile management, tier depth tracking,
relationship lifecycle management, risk propagation, compliance monitoring,
gap analysis, and audit reporting.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all multi-tier supplier tracking operations.

Enumerations (13):
    - SupplierTier, RelationshipStatus, RelationshipConfidence,
      RiskCategory, ComplianceStatus, GapSeverity, CertificationType,
      SupplierType, CommodityType, ReportFormat, BatchStatus,
      RiskPropagationMethod, DiscoverySource

Core Models (9):
    - SupplierProfile, SupplierRelationship, TierDepthResult,
      RiskScore, RiskPropagationResult, ComplianceCheckResult,
      DataGap, RemediationPlan, CertificationRecord

Request Models (7):
    - DiscoverSuppliersRequest, CreateSupplierRequest,
      AssessTierDepthRequest, AssessRiskRequest,
      CheckComplianceRequest, AnalyzeGapsRequest,
      GenerateReportRequest

Response Models (8):
    - DiscoverSuppliersResponse, CreateSupplierResponse,
      AssessTierDepthResponse, AssessRiskResponse,
      CheckComplianceResponse, AnalyzeGapsResponse,
      GenerateReportResponse, BatchResult

Compatibility:
    Imports EUDRCommodity from greenlang.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, and AGENT-EUDR-007
    GPS Coordinate Validator.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from greenlang.eudr_traceability.models import EUDRCommodity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Maximum number of supplier records in a single batch.
MAX_BATCH_SIZE: int = 100_000

#: Maximum supported supplier tier depth.
MAX_TIER_DEPTH: int = 15

#: Default profile completeness weights (must sum to 1.0).
DEFAULT_PROFILE_COMPLETENESS_WEIGHTS: Dict[str, float] = {
    "legal_identity": 0.25,
    "location": 0.20,
    "commodity": 0.15,
    "certification": 0.15,
    "compliance": 0.15,
    "contact": 0.10,
}

#: Default risk category weights (must sum to 1.0).
DEFAULT_RISK_CATEGORY_WEIGHTS: Dict[str, float] = {
    "deforestation_proximity": 0.30,
    "country_risk": 0.20,
    "certification_gap": 0.15,
    "compliance_history": 0.15,
    "data_quality": 0.10,
    "concentration_risk": 0.10,
}

#: Compliance status thresholds.
COMPLIANCE_STATUS_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "compliant": (90.0, 100.0),
    "conditionally_compliant": (70.0, 89.99),
    "non_compliant": (0.0, 69.99),
}

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Typical supply chain depths by commodity.
TYPICAL_CHAIN_DEPTHS: Dict[str, Tuple[int, int]] = {
    "cattle": (3, 5),
    "cocoa": (6, 8),
    "coffee": (5, 7),
    "oil_palm": (5, 7),
    "rubber": (5, 7),
    "soya": (4, 6),
    "wood": (4, 6),
}


# =============================================================================
# Enumerations
# =============================================================================


class SupplierTier(str, Enum):
    """Supplier tier level in the supply chain hierarchy.

    Represents the position of a supplier relative to the EU operator.
    Tier 1 is the direct supplier; higher tiers are deeper in the
    supply chain closer to the production origin.

    TIER_1: Direct supplier (exporter/trader to EU operator).
    TIER_2: Processor, refiner, or aggregator.
    TIER_3: Regional collector or cooperative.
    TIER_4: Local aggregation point.
    TIER_5: Sub-regional supplier.
    TIER_6: District-level supplier.
    TIER_7: Village-level cooperative or mill.
    TIER_8: Primary processor.
    TIER_9: Collection point.
    TIER_10: Farmer group.
    TIER_11: Individual farm or plot.
    TIER_12: Sub-plot level.
    TIER_13: Deep supply chain intermediary.
    TIER_14: Extended supply chain intermediary.
    TIER_15: Maximum tracked depth.
    UNKNOWN: Tier level not yet determined.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"
    TIER_5 = "tier_5"
    TIER_6 = "tier_6"
    TIER_7 = "tier_7"
    TIER_8 = "tier_8"
    TIER_9 = "tier_9"
    TIER_10 = "tier_10"
    TIER_11 = "tier_11"
    TIER_12 = "tier_12"
    TIER_13 = "tier_13"
    TIER_14 = "tier_14"
    TIER_15 = "tier_15"
    UNKNOWN = "unknown"


class RelationshipStatus(str, Enum):
    """Lifecycle status of a supplier relationship.

    Tracks the current state of the commercial relationship between
    a buyer and supplier within the supply chain hierarchy.

    PROSPECTIVE: Identified but not yet engaged.
    ONBOARDING: Currently being onboarded and verified.
    ACTIVE: Verified and actively supplying.
    SUSPENDED: Temporarily suspended due to compliance or risk.
    TERMINATED: Permanently terminated.
    """

    PROSPECTIVE = "prospective"
    ONBOARDING = "onboarding"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class RelationshipConfidence(str, Enum):
    """Confidence level of a discovered supplier relationship.

    Indicates how the relationship was established and the degree of
    certainty in the link between buyer and supplier.

    VERIFIED: Confirmed through documentation or direct verification.
    DECLARED: Declared by the supplier or buyer in a questionnaire.
    INFERRED: Inferred from shipping documents, invoices, or ERP data.
    SUSPECTED: Suspected based on pattern analysis or third-party data.
    """

    VERIFIED = "verified"
    DECLARED = "declared"
    INFERRED = "inferred"
    SUSPECTED = "suspected"


class RiskCategory(str, Enum):
    """Risk assessment category for supplier evaluation.

    Six risk dimensions used to compute composite supplier risk scores
    per PRD Section 5.5 / Appendix B.

    DEFORESTATION_PROXIMITY: Distance to recent deforestation events.
    COUNTRY_RISK: Country-level deforestation and governance risk.
    CERTIFICATION_GAP: Missing or expired certifications.
    COMPLIANCE_HISTORY: Historical compliance violations.
    DATA_QUALITY: Profile completeness and data freshness.
    CONCENTRATION_RISK: Single-source or geographic concentration.
    """

    DEFORESTATION_PROXIMITY = "deforestation_proximity"
    COUNTRY_RISK = "country_risk"
    CERTIFICATION_GAP = "certification_gap"
    COMPLIANCE_HISTORY = "compliance_history"
    DATA_QUALITY = "data_quality"
    CONCENTRATION_RISK = "concentration_risk"


class ComplianceStatus(str, Enum):
    """EUDR compliance status for a supplier.

    Determines whether a supplier can be included in a Due Diligence
    Statement (DDS) submission per Appendix C.

    COMPLIANT: All checks pass; valid DDS, certifications, GPS.
    CONDITIONALLY_COMPLIANT: Minor gaps; remediation in progress.
    NON_COMPLIANT: Critical gaps; failed checks.
    UNVERIFIED: Not yet assessed; insufficient data.
    EXPIRED: Previously compliant; certifications or DDS expired.
    """

    COMPLIANT = "compliant"
    CONDITIONALLY_COMPLIANT = "conditionally_compliant"
    NON_COMPLIANT = "non_compliant"
    UNVERIFIED = "unverified"
    EXPIRED = "expired"


class GapSeverity(str, Enum):
    """Severity classification for data gaps.

    CRITICAL: Blocks DDS submission entirely.
    MAJOR: Creates regulatory risk but does not block DDS.
    MINOR: Data quality issue; does not impact compliance.
    """

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class CertificationType(str, Enum):
    """Sustainability certification types relevant to EUDR commodities.

    FSC: Forest Stewardship Council (wood/timber).
    RSPO: Roundtable on Sustainable Palm Oil.
    UTZ: UTZ Certified (now Rainforest Alliance merged).
    RAINFOREST_ALLIANCE: Rainforest Alliance certification.
    ISO_14001: ISO 14001 Environmental Management System.
    ORGANIC: Organic farming certification.
    FAIRTRADE: Fairtrade International certification.
    OTHER: Other certification types not listed above.
    """

    FSC = "fsc"
    RSPO = "rspo"
    UTZ = "utz"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    ISO_14001 = "iso_14001"
    ORGANIC = "organic"
    FAIRTRADE = "fairtrade"
    OTHER = "other"


class SupplierType(str, Enum):
    """Type of entity in the commodity supply chain.

    Categorizes the role a supplier plays in the supply chain from
    production origin to EU market placement.

    FARMER: Individual smallholder farmer or plantation.
    COOPERATIVE: Farmer cooperative or association.
    AGGREGATOR: Regional commodity aggregator.
    PROCESSOR: Processing facility (mill, refinery, roaster).
    REFINERY: Specialized refining operation.
    TRADER: Commodity trading company.
    EXPORTER: Export operation in origin country.
    IMPORTER: Import operation in destination country.
    RETAILER: End retail or consumer-facing entity.
    """

    FARMER = "farmer"
    COOPERATIVE = "cooperative"
    AGGREGATOR = "aggregator"
    PROCESSOR = "processor"
    REFINERY = "refinery"
    TRADER = "trader"
    EXPORTER = "exporter"
    IMPORTER = "importer"
    RETAILER = "retailer"


class CommodityType(str, Enum):
    """EUDR-regulated commodity types and their derived products.

    The seven primary EUDR commodities plus a category for derived
    products that trace back to one or more primary commodities.

    CATTLE: Cattle and bovine products (leather, beef).
    COCOA: Cocoa beans and derived products (chocolate, cocoa butter).
    COFFEE: Coffee beans and derived products.
    OIL_PALM: Oil palm and derived products (palm oil, palm kernel oil).
    RUBBER: Natural rubber and derived products.
    SOYA: Soy beans and derived products (soy meal, soy oil).
    WOOD: Wood and derived products (timber, pulp, paper, furniture).
    DERIVED: Derived product traceable to one or more primary commodities.
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"
    DERIVED = "derived"


class ReportFormat(str, Enum):
    """Output format for audit and compliance reports.

    JSON: Machine-readable JSON format.
    PDF: Human-readable PDF report.
    CSV: Tabular CSV data export.
    EUDR_XML: EU Information System compatible XML format.
    """

    JSON = "json"
    PDF = "pdf"
    CSV = "csv"
    EUDR_XML = "eudr_xml"


class BatchStatus(str, Enum):
    """Status of a batch processing job.

    PENDING: Batch job submitted but not yet started.
    RUNNING: Batch job currently executing.
    COMPLETED: Batch job finished successfully.
    FAILED: Batch job terminated with errors.
    CANCELLED: Batch job cancelled by operator.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RiskPropagationMethod(str, Enum):
    """Method for propagating risk through the supplier hierarchy.

    MAX: Use the maximum risk score from any upstream supplier.
    WEIGHTED_AVERAGE: Volume-independent weighted average of upstream.
    VOLUME_WEIGHTED: Weight upstream risk by commodity volume share.
    """

    MAX = "max"
    WEIGHTED_AVERAGE = "weighted_average"
    VOLUME_WEIGHTED = "volume_weighted"


class DiscoverySource(str, Enum):
    """Source from which a supplier relationship was discovered.

    DECLARATION: Direct supplier declaration or self-registration.
    QUESTIONNAIRE: Supplier questionnaire response.
    SHIPPING_DOC: Extracted from shipping documents (BL, packing list).
    CERTIFICATION_DB: Cross-reference from certification database.
    ERP: Extracted from ERP or procurement system.
    MANUAL: Manually entered by operator.
    INFERRED: Algorithmically inferred from data patterns.
    """

    DECLARATION = "declaration"
    QUESTIONNAIRE = "questionnaire"
    SHIPPING_DOC = "shipping_doc"
    CERTIFICATION_DB = "certification_db"
    ERP = "erp"
    MANUAL = "manual"
    INFERRED = "inferred"


# =============================================================================
# Core Models
# =============================================================================


class CertificationRecord(BaseModel):
    """A sustainability certification held by a supplier.

    Tracks certification type, validity, and linkage to the certifying
    body for EUDR compliance verification.

    Attributes:
        certification_id: Unique identifier for this certification record.
        supplier_id: ID of the supplier holding this certification.
        certification_type: Type of certification (FSC, RSPO, etc.).
        certificate_number: Official certificate number or reference.
        certifying_body: Name of the organization that issued the cert.
        issue_date: Date the certification was issued.
        expiry_date: Date the certification expires.
        scope: Description of what the certification covers.
        status: Current status (active, expired, suspended, revoked).
        commodities: Commodities covered by this certification.
        verification_url: URL to verify the certificate online.
        last_verified_at: Last time the certificate was independently verified.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    certification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique certification record identifier",
    )
    supplier_id: str = Field(
        ...,
        description="ID of the supplier holding this certification",
    )
    certification_type: CertificationType = Field(
        ...,
        description="Type of sustainability certification",
    )
    certificate_number: str = Field(
        ...,
        description="Official certificate number or reference",
    )
    certifying_body: str = Field(
        ...,
        description="Issuing certification organization",
    )
    issue_date: datetime = Field(
        ...,
        description="Date the certification was issued",
    )
    expiry_date: datetime = Field(
        ...,
        description="Date the certification expires",
    )
    scope: Optional[str] = Field(
        None,
        description="Scope description of the certification",
    )
    status: str = Field(
        "active",
        description="Current status: active, expired, suspended, revoked",
    )
    commodities: List[CommodityType] = Field(
        default_factory=list,
        description="Commodities covered by this certification",
    )
    verification_url: Optional[str] = Field(
        None,
        description="URL to verify the certificate online",
    )
    last_verified_at: Optional[datetime] = Field(
        None,
        description="Last independent verification timestamp",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate certification status value."""
        valid_statuses = {"active", "expired", "suspended", "revoked"}
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"status must be one of {sorted(valid_statuses)}, got '{v}'"
            )
        return v.lower()


class SupplierProfile(BaseModel):
    """Comprehensive supplier profile with EUDR compliance metadata.

    Maintains all required supplier information per EUDR Article 9
    including legal entity, location, commodity, certification,
    compliance status, and contact data.

    Attributes:
        supplier_id: Unique supplier identifier.
        legal_name: Official registered legal entity name.
        registration_id: Business registration number.
        tax_id: Tax identification number.
        duns_number: Dun and Bradstreet DUNS number.
        supplier_type: Role in supply chain hierarchy.
        tier: Position in supply chain relative to EU operator.
        country_iso: ISO 3166-1 alpha-2 country code.
        admin_region: Administrative region or state.
        address: Full postal address.
        latitude: GPS latitude of supplier location.
        longitude: GPS longitude of supplier location.
        commodities: List of EUDR commodities handled.
        annual_volume_tonnes: Annual commodity volume in metric tonnes.
        processing_capacity_tonnes: Processing capacity in tonnes/year.
        upstream_supplier_count: Number of known upstream suppliers.
        certifications: List of certification records.
        dds_references: List of linked DDS IDs from EU Information System.
        deforestation_free_status: Whether verified deforestation-free.
        compliance_status: Current EUDR compliance status.
        compliance_score: Composite compliance score (0-100).
        profile_completeness_score: Profile completeness (0-100).
        risk_score: Composite risk score (0-100).
        primary_contact_name: Name of primary contact.
        primary_contact_email: Email of primary contact.
        primary_contact_phone: Phone of primary contact.
        compliance_contact_name: Name of compliance contact.
        compliance_contact_email: Email of compliance contact.
        discovery_source: How this supplier was discovered.
        is_active: Whether the supplier profile is active.
        version: Profile version number for change tracking.
        notes: Free-text notes about the supplier.
        metadata: Additional structured metadata.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    supplier_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique supplier identifier",
    )
    legal_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Official registered legal entity name",
    )
    registration_id: Optional[str] = Field(
        None,
        description="Business registration number",
    )
    tax_id: Optional[str] = Field(
        None,
        description="Tax identification number",
    )
    duns_number: Optional[str] = Field(
        None,
        description="Dun and Bradstreet DUNS number",
    )
    supplier_type: SupplierType = Field(
        ...,
        description="Role in supply chain hierarchy",
    )
    tier: SupplierTier = Field(
        SupplierTier.UNKNOWN,
        description="Position in supply chain relative to EU operator",
    )
    country_iso: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    admin_region: Optional[str] = Field(
        None,
        description="Administrative region or state",
    )
    address: Optional[str] = Field(
        None,
        description="Full postal address",
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="GPS latitude of supplier location",
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="GPS longitude of supplier location",
    )
    commodities: List[CommodityType] = Field(
        default_factory=list,
        description="EUDR commodities handled by this supplier",
    )
    annual_volume_tonnes: Optional[float] = Field(
        None,
        ge=0.0,
        description="Annual commodity volume in metric tonnes",
    )
    processing_capacity_tonnes: Optional[float] = Field(
        None,
        ge=0.0,
        description="Processing capacity in tonnes per year",
    )
    upstream_supplier_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of known upstream suppliers",
    )
    certifications: List[CertificationRecord] = Field(
        default_factory=list,
        description="Sustainability certifications held",
    )
    dds_references: List[str] = Field(
        default_factory=list,
        description="Linked DDS IDs from EU Information System",
    )
    deforestation_free_status: Optional[bool] = Field(
        None,
        description="Whether verified deforestation-free",
    )
    compliance_status: ComplianceStatus = Field(
        ComplianceStatus.UNVERIFIED,
        description="Current EUDR compliance status",
    )
    compliance_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Composite compliance score (0-100)",
    )
    profile_completeness_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Profile completeness score (0-100)",
    )
    risk_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Composite risk score (0-100)",
    )
    primary_contact_name: Optional[str] = Field(
        None,
        description="Name of primary contact person",
    )
    primary_contact_email: Optional[str] = Field(
        None,
        description="Email of primary contact person",
    )
    primary_contact_phone: Optional[str] = Field(
        None,
        description="Phone number of primary contact",
    )
    compliance_contact_name: Optional[str] = Field(
        None,
        description="Name of compliance contact person",
    )
    compliance_contact_email: Optional[str] = Field(
        None,
        description="Email of compliance contact person",
    )
    discovery_source: DiscoverySource = Field(
        DiscoverySource.MANUAL,
        description="How this supplier was discovered",
    )
    is_active: bool = Field(
        True,
        description="Whether the supplier profile is active",
    )
    version: int = Field(
        1,
        ge=1,
        description="Profile version number for change tracking",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes about the supplier",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp",
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: str) -> str:
        """Normalize country ISO code to uppercase."""
        return v.upper()


class SupplierRelationship(BaseModel):
    """A relationship link between two suppliers in the hierarchy.

    Represents a buyer-supplier connection with lifecycle status,
    confidence level, commodity flow, and volume data for supply
    chain mapping.

    Attributes:
        relationship_id: Unique relationship identifier.
        buyer_id: Supplier ID of the buying entity.
        supplier_id: Supplier ID of the supplying entity.
        buyer_tier: Tier of the buyer in the hierarchy.
        supplier_tier: Tier of the supplier in the hierarchy.
        status: Lifecycle status of the relationship.
        confidence: Confidence level of the relationship link.
        commodity: Primary commodity in this relationship.
        volume_tonnes: Annual volume in metric tonnes.
        volume_share_pct: Percentage of buyer's total volume.
        start_date: When the relationship began.
        end_date: When the relationship ended (if terminated).
        is_exclusive: Whether this is an exclusive supply relationship.
        is_seasonal: Whether the relationship is seasonal.
        season_start_month: Start month of seasonal supply (1-12).
        season_end_month: End month of seasonal supply (1-12).
        discovery_source: How this relationship was discovered.
        frequency: Transaction frequency (daily, weekly, monthly, etc.).
        strength_score: Relationship strength score (0-100).
        reason_code: Reason for current status (e.g., suspension reason).
        dds_reference: DDS reference linked to this relationship.
        metadata: Additional structured metadata.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    relationship_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique relationship identifier",
    )
    buyer_id: str = Field(
        ...,
        description="Supplier ID of the buying entity",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier ID of the supplying entity",
    )
    buyer_tier: SupplierTier = Field(
        SupplierTier.UNKNOWN,
        description="Tier of the buyer in the hierarchy",
    )
    supplier_tier: SupplierTier = Field(
        SupplierTier.UNKNOWN,
        description="Tier of the supplier in the hierarchy",
    )
    status: RelationshipStatus = Field(
        RelationshipStatus.PROSPECTIVE,
        description="Lifecycle status of the relationship",
    )
    confidence: RelationshipConfidence = Field(
        RelationshipConfidence.SUSPECTED,
        description="Confidence level of the relationship link",
    )
    commodity: CommodityType = Field(
        ...,
        description="Primary commodity in this relationship",
    )
    volume_tonnes: Optional[float] = Field(
        None,
        ge=0.0,
        description="Annual volume in metric tonnes",
    )
    volume_share_pct: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Percentage of buyer total volume",
    )
    start_date: Optional[datetime] = Field(
        None,
        description="When the relationship began",
    )
    end_date: Optional[datetime] = Field(
        None,
        description="When the relationship ended",
    )
    is_exclusive: bool = Field(
        False,
        description="Whether this is an exclusive supply relationship",
    )
    is_seasonal: bool = Field(
        False,
        description="Whether the relationship is seasonal",
    )
    season_start_month: Optional[int] = Field(
        None,
        ge=1,
        le=12,
        description="Start month of seasonal supply (1-12)",
    )
    season_end_month: Optional[int] = Field(
        None,
        ge=1,
        le=12,
        description="End month of seasonal supply (1-12)",
    )
    discovery_source: DiscoverySource = Field(
        DiscoverySource.MANUAL,
        description="How this relationship was discovered",
    )
    frequency: Optional[str] = Field(
        None,
        description="Transaction frequency (daily, weekly, monthly, etc.)",
    )
    strength_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Relationship strength score (0-100)",
    )
    reason_code: Optional[str] = Field(
        None,
        description="Reason for current status",
    )
    dds_reference: Optional[str] = Field(
        None,
        description="DDS reference linked to this relationship",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp",
    )

    @model_validator(mode="after")
    def validate_relationship(self) -> SupplierRelationship:
        """Validate buyer and supplier are different entities."""
        if self.buyer_id == self.supplier_id:
            raise ValueError("buyer_id and supplier_id must be different")
        return self


class TierDepthResult(BaseModel):
    """Result of a tier depth assessment for a supply chain.

    Quantifies how deep the supply chain visibility extends and
    identifies where visibility drops off.

    Attributes:
        assessment_id: Unique assessment identifier.
        root_supplier_id: ID of the root supplier (Tier 1).
        commodity: Commodity being traced.
        max_tier_reached: Deepest tier with known suppliers.
        total_suppliers_mapped: Total suppliers discovered in chain.
        tier_counts: Number of suppliers at each tier level.
        visibility_scores: Visibility score per tier (0.0-1.0).
        coverage_score: Overall coverage score (0-100).
        gap_tiers: Tiers with missing or incomplete visibility.
        depth_vs_benchmark: Comparison to industry average depth.
        first_unknown_tier: First tier where visibility drops off.
        chain_completeness_pct: Percentage of chain fully traced.
        assessed_at: Assessment timestamp.
        metadata: Additional assessment metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    assessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique assessment identifier",
    )
    root_supplier_id: str = Field(
        ...,
        description="ID of the root supplier (Tier 1)",
    )
    commodity: CommodityType = Field(
        ...,
        description="Commodity being traced",
    )
    max_tier_reached: int = Field(
        ...,
        ge=0,
        le=15,
        description="Deepest tier with known suppliers",
    )
    total_suppliers_mapped: int = Field(
        ...,
        ge=0,
        description="Total suppliers discovered in chain",
    )
    tier_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of suppliers at each tier level",
    )
    visibility_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Visibility score per tier (0.0-1.0)",
    )
    coverage_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall coverage score (0-100)",
    )
    gap_tiers: List[str] = Field(
        default_factory=list,
        description="Tiers with missing or incomplete visibility",
    )
    depth_vs_benchmark: Optional[float] = Field(
        None,
        description="Depth compared to industry average (ratio)",
    )
    first_unknown_tier: Optional[str] = Field(
        None,
        description="First tier where visibility drops off",
    )
    chain_completeness_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of chain fully traced to origin",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional assessment metadata",
    )


class RiskScore(BaseModel):
    """Composite risk assessment for a supplier.

    Contains individual risk category scores and the weighted composite
    score per PRD Section 5.5 and Appendix B.

    Attributes:
        risk_id: Unique risk assessment identifier.
        supplier_id: Supplier being assessed.
        composite_score: Weighted composite risk score (0-100).
        category_scores: Individual scores per risk category.
        deforestation_proximity_score: Deforestation proximity (0-100).
        country_risk_score: Country-level risk (0-100).
        certification_gap_score: Certification gap risk (0-100).
        compliance_history_score: Compliance history risk (0-100).
        data_quality_score: Data quality risk (0-100).
        concentration_risk_score: Concentration risk (0-100).
        risk_level: Risk level classification.
        trend: Risk trend direction (improving, stable, degrading).
        contributing_factors: Key factors driving the risk score.
        assessed_at: Assessment timestamp.
        valid_until: When this assessment expires.
        metadata: Additional risk metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    risk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique risk assessment identifier",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier being assessed",
    )
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Weighted composite risk score (0-100)",
    )
    category_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual scores per risk category (0-100)",
    )
    deforestation_proximity_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Deforestation proximity risk score (0-100)",
    )
    country_risk_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Country-level risk score (0-100)",
    )
    certification_gap_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Certification gap risk score (0-100)",
    )
    compliance_history_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Compliance history risk score (0-100)",
    )
    data_quality_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Data quality risk score (0-100)",
    )
    concentration_risk_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Concentration risk score (0-100)",
    )
    risk_level: str = Field(
        "unknown",
        description="Risk level: low, medium, high, critical, unknown",
    )
    trend: str = Field(
        "stable",
        description="Risk trend: improving, stable, degrading",
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Key factors driving the risk score",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp",
    )
    valid_until: Optional[datetime] = Field(
        None,
        description="When this assessment expires",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional risk metadata",
    )


class RiskPropagationResult(BaseModel):
    """Result of propagating risk through the supplier hierarchy.

    Shows how deep-tier risk flows upstream to Tier 1 and the operator.

    Attributes:
        propagation_id: Unique propagation run identifier.
        root_supplier_id: The root supplier receiving propagated risk.
        method: Propagation method used (max, weighted_avg, volume_weighted).
        original_score: Root supplier's own risk score before propagation.
        propagated_score: Risk score after upstream propagation.
        propagation_path: Ordered list of supplier IDs in propagation path.
        tier_risk_breakdown: Risk scores at each tier level.
        highest_risk_supplier_id: Supplier with the highest upstream risk.
        highest_risk_score: Score of the highest-risk upstream supplier.
        suppliers_assessed: Total suppliers included in propagation.
        propagated_at: Propagation timestamp.
        metadata: Additional propagation metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    propagation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique propagation run identifier",
    )
    root_supplier_id: str = Field(
        ...,
        description="Root supplier receiving propagated risk",
    )
    method: RiskPropagationMethod = Field(
        ...,
        description="Propagation method used",
    )
    original_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Root supplier own risk before propagation",
    )
    propagated_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Risk score after upstream propagation",
    )
    propagation_path: List[str] = Field(
        default_factory=list,
        description="Ordered supplier IDs in propagation path",
    )
    tier_risk_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Risk scores at each tier level",
    )
    highest_risk_supplier_id: Optional[str] = Field(
        None,
        description="Supplier with the highest upstream risk",
    )
    highest_risk_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Score of the highest-risk upstream supplier",
    )
    suppliers_assessed: int = Field(
        ...,
        ge=0,
        description="Total suppliers included in propagation",
    )
    propagated_at: datetime = Field(
        default_factory=_utcnow,
        description="Propagation timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional propagation metadata",
    )


class ComplianceCheckResult(BaseModel):
    """Result of a compliance status check for a supplier.

    Evaluates the supplier against EUDR compliance dimensions: DDS
    validity, certification status, geolocation coverage, and
    deforestation-free verification.

    Attributes:
        check_id: Unique compliance check identifier.
        supplier_id: Supplier being checked.
        overall_status: Resulting compliance status.
        overall_score: Composite compliance score (0-100).
        dds_valid: Whether DDS reference is valid and current.
        dds_expiry_date: DDS expiry date if applicable.
        dds_days_until_expiry: Days until DDS expires.
        certification_valid: Whether certifications are valid.
        certification_expiry_date: Earliest cert expiry date.
        cert_days_until_expiry: Days until earliest cert expires.
        geolocation_coverage_pct: Percentage of volume GPS-verified.
        deforestation_free_verified: Deforestation-free status.
        dimension_scores: Score per compliance dimension (0-100).
        alerts: List of compliance alert messages.
        previous_status: Status before this check.
        status_changed: Whether status changed from previous.
        checked_at: Check timestamp.
        next_check_due: When the next check should occur.
        metadata: Additional check metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    check_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique compliance check identifier",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier being checked",
    )
    overall_status: ComplianceStatus = Field(
        ...,
        description="Resulting compliance status",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite compliance score (0-100)",
    )
    dds_valid: bool = Field(
        False,
        description="Whether DDS reference is valid and current",
    )
    dds_expiry_date: Optional[datetime] = Field(
        None,
        description="DDS expiry date if applicable",
    )
    dds_days_until_expiry: Optional[int] = Field(
        None,
        description="Days until DDS expires",
    )
    certification_valid: bool = Field(
        False,
        description="Whether certifications are valid and current",
    )
    certification_expiry_date: Optional[datetime] = Field(
        None,
        description="Earliest certification expiry date",
    )
    cert_days_until_expiry: Optional[int] = Field(
        None,
        description="Days until earliest certification expires",
    )
    geolocation_coverage_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of volume with GPS-verified origin",
    )
    deforestation_free_verified: Optional[bool] = Field(
        None,
        description="Whether deforestation-free status is verified",
    )
    dimension_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Score per compliance dimension (0-100)",
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Compliance alert messages",
    )
    previous_status: Optional[ComplianceStatus] = Field(
        None,
        description="Status before this check",
    )
    status_changed: bool = Field(
        False,
        description="Whether status changed from previous check",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Check timestamp",
    )
    next_check_due: Optional[datetime] = Field(
        None,
        description="When the next compliance check should occur",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional check metadata",
    )


class DataGap(BaseModel):
    """A data gap identified in supplier profile or supply chain.

    Represents missing, incomplete, or outdated data that impacts
    EUDR compliance readiness.

    Attributes:
        gap_id: Unique gap identifier.
        supplier_id: Supplier with the data gap.
        gap_type: Type of gap (missing_gps, missing_certification, etc.).
        severity: Gap severity classification.
        field_name: Name of the missing or incomplete field.
        description: Human-readable gap description.
        impact: How this gap impacts compliance.
        remediation_action: Recommended remediation action.
        remediation_deadline: Deadline for remediation.
        is_resolved: Whether the gap has been resolved.
        resolved_at: When the gap was resolved.
        resolved_by: Who resolved the gap.
        tier: Tier level where the gap exists.
        commodity: Commodity affected by the gap.
        detected_at: When the gap was detected.
        metadata: Additional gap metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    gap_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique gap identifier",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier with the data gap",
    )
    gap_type: str = Field(
        ...,
        description="Type: missing_gps, missing_certification, "
        "missing_legal_entity, missing_dds, outdated_data, "
        "missing_contact, missing_commodity, tier_gap",
    )
    severity: GapSeverity = Field(
        ...,
        description="Gap severity classification",
    )
    field_name: str = Field(
        ...,
        description="Name of the missing or incomplete field",
    )
    description: str = Field(
        ...,
        description="Human-readable gap description",
    )
    impact: str = Field(
        ...,
        description="How this gap impacts compliance",
    )
    remediation_action: str = Field(
        ...,
        description="Recommended remediation action",
    )
    remediation_deadline: Optional[datetime] = Field(
        None,
        description="Deadline for remediation",
    )
    is_resolved: bool = Field(
        False,
        description="Whether the gap has been resolved",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="When the gap was resolved",
    )
    resolved_by: Optional[str] = Field(
        None,
        description="Who resolved the gap",
    )
    tier: Optional[SupplierTier] = Field(
        None,
        description="Tier level where the gap exists",
    )
    commodity: Optional[CommodityType] = Field(
        None,
        description="Commodity affected by the gap",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="When the gap was detected",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional gap metadata",
    )


class RemediationPlan(BaseModel):
    """Action plan for remediating supplier data gaps.

    Contains prioritized steps, progress tracking, and completion
    metrics for closing identified data gaps.

    Attributes:
        plan_id: Unique remediation plan identifier.
        supplier_id: Supplier this plan targets.
        gap_ids: List of gap IDs addressed by this plan.
        title: Plan title summary.
        description: Detailed plan description.
        priority: Plan priority (1=highest).
        steps: Ordered list of remediation steps.
        total_steps: Total number of steps.
        completed_steps: Number of completed steps.
        completion_pct: Completion percentage (0-100).
        status: Plan status (draft, active, completed, cancelled).
        assigned_to: Person or team assigned to execute.
        due_date: Plan completion deadline.
        started_at: When execution began.
        completed_at: When the plan was completed.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
        metadata: Additional plan metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    plan_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique remediation plan identifier",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier this plan targets",
    )
    gap_ids: List[str] = Field(
        default_factory=list,
        description="Gap IDs addressed by this plan",
    )
    title: str = Field(
        ...,
        description="Plan title summary",
    )
    description: str = Field(
        ...,
        description="Detailed plan description",
    )
    priority: int = Field(
        1,
        ge=1,
        le=5,
        description="Priority level (1=highest, 5=lowest)",
    )
    steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of remediation steps",
    )
    total_steps: int = Field(
        0,
        ge=0,
        description="Total number of steps",
    )
    completed_steps: int = Field(
        0,
        ge=0,
        description="Number of completed steps",
    )
    completion_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Completion percentage (0-100)",
    )
    status: str = Field(
        "draft",
        description="Plan status: draft, active, completed, cancelled",
    )
    assigned_to: Optional[str] = Field(
        None,
        description="Person or team assigned to execute",
    )
    due_date: Optional[datetime] = Field(
        None,
        description="Plan completion deadline",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="When execution began",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="When the plan was completed",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional plan metadata",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate plan status value."""
        valid_statuses = {"draft", "active", "completed", "cancelled"}
        if v.lower() not in valid_statuses:
            raise ValueError(
                f"status must be one of {sorted(valid_statuses)}, got '{v}'"
            )
        return v.lower()


# =============================================================================
# Request Models
# =============================================================================


class DiscoverSuppliersRequest(BaseModel):
    """Request to discover sub-tier suppliers from various sources.

    Attributes:
        root_supplier_id: Starting supplier for discovery.
        commodity: Commodity to trace.
        max_depth: Maximum discovery depth.
        sources: Discovery sources to query.
        confidence_threshold: Minimum relationship confidence.
        deduplication_enabled: Whether to deduplicate results.
        include_inactive: Whether to include inactive suppliers.
        metadata: Additional request metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    root_supplier_id: str = Field(
        ...,
        description="Starting supplier for discovery",
    )
    commodity: CommodityType = Field(
        ...,
        description="Commodity to trace through supply chain",
    )
    max_depth: int = Field(
        15,
        ge=1,
        le=50,
        description="Maximum discovery depth",
    )
    sources: List[DiscoverySource] = Field(
        default_factory=lambda: [
            DiscoverySource.DECLARATION,
            DiscoverySource.QUESTIONNAIRE,
            DiscoverySource.SHIPPING_DOC,
            DiscoverySource.CERTIFICATION_DB,
            DiscoverySource.ERP,
        ],
        description="Discovery sources to query",
    )
    confidence_threshold: float = Field(
        0.70,
        ge=0.0,
        le=1.0,
        description="Minimum relationship confidence",
    )
    deduplication_enabled: bool = Field(
        True,
        description="Whether to deduplicate discovered suppliers",
    )
    include_inactive: bool = Field(
        False,
        description="Whether to include inactive suppliers",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )


class CreateSupplierRequest(BaseModel):
    """Request to create a new supplier profile.

    Attributes:
        legal_name: Official registered legal entity name.
        supplier_type: Role in supply chain hierarchy.
        country_iso: ISO 3166-1 alpha-2 country code.
        tier: Position in supply chain.
        registration_id: Business registration number.
        tax_id: Tax identification number.
        duns_number: DUNS number.
        admin_region: Administrative region.
        address: Postal address.
        latitude: GPS latitude.
        longitude: GPS longitude.
        commodities: Commodities handled.
        annual_volume_tonnes: Annual volume in tonnes.
        discovery_source: How the supplier was discovered.
        primary_contact_name: Primary contact name.
        primary_contact_email: Primary contact email.
        notes: Free-text notes.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    legal_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Official registered legal entity name",
    )
    supplier_type: SupplierType = Field(
        ...,
        description="Role in supply chain hierarchy",
    )
    country_iso: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    tier: SupplierTier = Field(
        SupplierTier.UNKNOWN,
        description="Position in supply chain",
    )
    registration_id: Optional[str] = Field(
        None, description="Business registration number",
    )
    tax_id: Optional[str] = Field(
        None, description="Tax identification number",
    )
    duns_number: Optional[str] = Field(
        None, description="DUNS number",
    )
    admin_region: Optional[str] = Field(
        None, description="Administrative region",
    )
    address: Optional[str] = Field(
        None, description="Postal address",
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="GPS latitude",
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="GPS longitude",
    )
    commodities: List[CommodityType] = Field(
        default_factory=list, description="Commodities handled",
    )
    annual_volume_tonnes: Optional[float] = Field(
        None, ge=0.0, description="Annual volume in tonnes",
    )
    discovery_source: DiscoverySource = Field(
        DiscoverySource.MANUAL, description="Discovery source",
    )
    primary_contact_name: Optional[str] = Field(
        None, description="Primary contact name",
    )
    primary_contact_email: Optional[str] = Field(
        None, description="Primary contact email",
    )
    notes: Optional[str] = Field(
        None, description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata",
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: str) -> str:
        """Normalize country ISO code to uppercase."""
        return v.upper()


class AssessTierDepthRequest(BaseModel):
    """Request to assess tier depth for a supply chain.

    Attributes:
        root_supplier_id: Starting supplier for assessment.
        commodity: Commodity to assess.
        include_inactive_relationships: Include inactive relationships.
        metadata: Additional request metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    root_supplier_id: str = Field(
        ...,
        description="Starting supplier for assessment",
    )
    commodity: CommodityType = Field(
        ...,
        description="Commodity to assess",
    )
    include_inactive_relationships: bool = Field(
        False,
        description="Include inactive relationships in assessment",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )


class AssessRiskRequest(BaseModel):
    """Request to assess risk for a supplier.

    Attributes:
        supplier_id: Supplier to assess.
        propagation_method: Risk propagation method.
        include_upstream: Whether to include upstream supplier risk.
        commodity: Commodity context for risk assessment.
        metadata: Additional request metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    supplier_id: str = Field(
        ...,
        description="Supplier to assess",
    )
    propagation_method: RiskPropagationMethod = Field(
        RiskPropagationMethod.MAX,
        description="Risk propagation method",
    )
    include_upstream: bool = Field(
        True,
        description="Whether to include upstream supplier risk",
    )
    commodity: Optional[CommodityType] = Field(
        None,
        description="Commodity context for risk assessment",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )


class CheckComplianceRequest(BaseModel):
    """Request to check compliance status for a supplier.

    Attributes:
        supplier_id: Supplier to check.
        check_dds: Whether to check DDS validity.
        check_certifications: Whether to check certifications.
        check_geolocation: Whether to check GPS coverage.
        check_deforestation: Whether to check deforestation status.
        metadata: Additional request metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    supplier_id: str = Field(
        ...,
        description="Supplier to check",
    )
    check_dds: bool = Field(
        True,
        description="Whether to check DDS validity",
    )
    check_certifications: bool = Field(
        True,
        description="Whether to check certifications",
    )
    check_geolocation: bool = Field(
        True,
        description="Whether to check GPS coverage",
    )
    check_deforestation: bool = Field(
        True,
        description="Whether to check deforestation-free status",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )


class AnalyzeGapsRequest(BaseModel):
    """Request to analyze data gaps for a supplier or supply chain.

    Attributes:
        supplier_id: Supplier to analyze (if single supplier).
        root_supplier_id: Root supplier for chain-wide analysis.
        commodity: Commodity context.
        include_resolved: Whether to include resolved gaps.
        generate_remediation: Whether to generate remediation plans.
        metadata: Additional request metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    supplier_id: Optional[str] = Field(
        None,
        description="Supplier to analyze (single supplier)",
    )
    root_supplier_id: Optional[str] = Field(
        None,
        description="Root supplier for chain-wide analysis",
    )
    commodity: Optional[CommodityType] = Field(
        None,
        description="Commodity context",
    )
    include_resolved: bool = Field(
        False,
        description="Whether to include already resolved gaps",
    )
    generate_remediation: bool = Field(
        True,
        description="Whether to generate remediation plans",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )

    @model_validator(mode="after")
    def validate_target(self) -> AnalyzeGapsRequest:
        """Validate that at least one target is specified."""
        if not self.supplier_id and not self.root_supplier_id:
            raise ValueError(
                "Either supplier_id or root_supplier_id must be provided"
            )
        return self


class GenerateReportRequest(BaseModel):
    """Request to generate an audit or compliance report.

    Attributes:
        root_supplier_id: Root supplier for the report.
        commodity: Commodity context.
        report_type: Type of report to generate.
        format: Output format.
        include_risk_details: Include risk assessment details.
        include_compliance_details: Include compliance check details.
        include_gap_details: Include gap analysis details.
        include_tier_details: Include tier depth details.
        date_range_start: Start of reporting period.
        date_range_end: End of reporting period.
        metadata: Additional request metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    root_supplier_id: str = Field(
        ...,
        description="Root supplier for the report",
    )
    commodity: Optional[CommodityType] = Field(
        None,
        description="Commodity context",
    )
    report_type: str = Field(
        "audit",
        description="Report type: audit, tier_summary, gaps, "
        "risk_propagation, dds_readiness",
    )
    format: ReportFormat = Field(
        ReportFormat.JSON,
        description="Output format",
    )
    include_risk_details: bool = Field(
        True,
        description="Include risk assessment details",
    )
    include_compliance_details: bool = Field(
        True,
        description="Include compliance check details",
    )
    include_gap_details: bool = Field(
        True,
        description="Include gap analysis details",
    )
    include_tier_details: bool = Field(
        True,
        description="Include tier depth details",
    )
    date_range_start: Optional[datetime] = Field(
        None,
        description="Start of reporting period",
    )
    date_range_end: Optional[datetime] = Field(
        None,
        description="End of reporting period",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        """Validate report type value."""
        valid_types = {
            "audit", "tier_summary", "gaps",
            "risk_propagation", "dds_readiness",
        }
        if v.lower() not in valid_types:
            raise ValueError(
                f"report_type must be one of {sorted(valid_types)}, "
                f"got '{v}'"
            )
        return v.lower()


# =============================================================================
# Response Models
# =============================================================================


class DiscoverSuppliersResponse(BaseModel):
    """Response from supplier discovery operation.

    Attributes:
        request_id: Unique request identifier.
        root_supplier_id: Starting supplier for discovery.
        commodity: Commodity traced.
        suppliers_discovered: Total new suppliers discovered.
        relationships_created: Total relationships created.
        max_depth_reached: Deepest tier discovered.
        discovered_suppliers: List of discovered supplier profiles.
        discovered_relationships: List of discovered relationships.
        duplicates_merged: Number of duplicate suppliers merged.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional response metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    root_supplier_id: str = Field(
        ...,
        description="Starting supplier for discovery",
    )
    commodity: CommodityType = Field(
        ...,
        description="Commodity traced",
    )
    suppliers_discovered: int = Field(
        0,
        ge=0,
        description="Total new suppliers discovered",
    )
    relationships_created: int = Field(
        0,
        ge=0,
        description="Total relationships created",
    )
    max_depth_reached: int = Field(
        0,
        ge=0,
        description="Deepest tier discovered",
    )
    discovered_suppliers: List[SupplierProfile] = Field(
        default_factory=list,
        description="Discovered supplier profiles",
    )
    discovered_relationships: List[SupplierRelationship] = Field(
        default_factory=list,
        description="Discovered relationships",
    )
    duplicates_merged: int = Field(
        0,
        ge=0,
        description="Number of duplicate suppliers merged",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class CreateSupplierResponse(BaseModel):
    """Response from supplier profile creation.

    Attributes:
        request_id: Unique request identifier.
        supplier_id: ID of the created supplier.
        profile: Created supplier profile.
        profile_completeness_score: Completeness score (0-100).
        missing_fields: List of missing required fields.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional response metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    supplier_id: str = Field(
        ...,
        description="ID of the created supplier",
    )
    profile: SupplierProfile = Field(
        ...,
        description="Created supplier profile",
    )
    profile_completeness_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Profile completeness score (0-100)",
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="Missing required fields for full completeness",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class AssessTierDepthResponse(BaseModel):
    """Response from tier depth assessment.

    Attributes:
        request_id: Unique request identifier.
        result: Tier depth assessment result.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional response metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    result: TierDepthResult = Field(
        ...,
        description="Tier depth assessment result",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class AssessRiskResponse(BaseModel):
    """Response from risk assessment.

    Attributes:
        request_id: Unique request identifier.
        risk_score: Risk assessment result.
        propagation_result: Risk propagation result if upstream included.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional response metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    risk_score: RiskScore = Field(
        ...,
        description="Risk assessment result",
    )
    propagation_result: Optional[RiskPropagationResult] = Field(
        None,
        description="Risk propagation result if upstream included",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class CheckComplianceResponse(BaseModel):
    """Response from compliance status check.

    Attributes:
        request_id: Unique request identifier.
        result: Compliance check result.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional response metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    result: ComplianceCheckResult = Field(
        ...,
        description="Compliance check result",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class AnalyzeGapsResponse(BaseModel):
    """Response from gap analysis.

    Attributes:
        request_id: Unique request identifier.
        gaps: List of identified data gaps.
        remediation_plans: Generated remediation plans.
        total_gaps: Total number of gaps found.
        critical_gaps: Number of critical gaps.
        major_gaps: Number of major gaps.
        minor_gaps: Number of minor gaps.
        coverage_score: Data coverage score (0-100).
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional response metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    gaps: List[DataGap] = Field(
        default_factory=list,
        description="Identified data gaps",
    )
    remediation_plans: List[RemediationPlan] = Field(
        default_factory=list,
        description="Generated remediation plans",
    )
    total_gaps: int = Field(
        0,
        ge=0,
        description="Total number of gaps found",
    )
    critical_gaps: int = Field(
        0,
        ge=0,
        description="Number of critical gaps",
    )
    major_gaps: int = Field(
        0,
        ge=0,
        description="Number of major gaps",
    )
    minor_gaps: int = Field(
        0,
        ge=0,
        description="Number of minor gaps",
    )
    coverage_score: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Data coverage score (0-100)",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class GenerateReportResponse(BaseModel):
    """Response from report generation.

    Attributes:
        request_id: Unique request identifier.
        report_id: Unique report identifier.
        report_type: Type of report generated.
        format: Output format used.
        content: Report content (JSON string or base64 for binary).
        file_path: Path to generated report file.
        file_size_bytes: Size of the report file in bytes.
        supplier_count: Number of suppliers in the report.
        tier_count: Number of tiers covered.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 provenance hash.
        generated_at: Report generation timestamp.
        metadata: Additional response metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier",
    )
    report_type: str = Field(
        ...,
        description="Type of report generated",
    )
    format: ReportFormat = Field(
        ...,
        description="Output format used",
    )
    content: Optional[str] = Field(
        None,
        description="Report content (JSON string or base64 for binary)",
    )
    file_path: Optional[str] = Field(
        None,
        description="Path to generated report file",
    )
    file_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Size of the report file in bytes",
    )
    supplier_count: int = Field(
        0,
        ge=0,
        description="Number of suppliers in the report",
    )
    tier_count: int = Field(
        0,
        ge=0,
        description="Number of tiers covered",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Report generation timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )


class BatchResult(BaseModel):
    """Result of a batch processing job.

    Attributes:
        batch_id: Unique batch job identifier.
        status: Current batch job status.
        operation: Type of batch operation performed.
        total_records: Total records in the batch.
        processed_records: Number of records processed.
        succeeded_records: Number of records that succeeded.
        failed_records: Number of records that failed.
        progress_pct: Progress percentage (0-100).
        errors: List of error messages for failed records.
        results: List of individual operation results.
        processing_time_ms: Total processing duration in ms.
        provenance_hash: SHA-256 provenance hash.
        started_at: Batch job start timestamp.
        completed_at: Batch job completion timestamp.
        metadata: Additional batch metadata.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier",
    )
    status: BatchStatus = Field(
        BatchStatus.PENDING,
        description="Current batch job status",
    )
    operation: str = Field(
        ...,
        description="Type of batch operation: discover, create, "
        "assess_risk, check_compliance, analyze_gaps",
    )
    total_records: int = Field(
        0,
        ge=0,
        description="Total records in the batch",
    )
    processed_records: int = Field(
        0,
        ge=0,
        description="Number of records processed",
    )
    succeeded_records: int = Field(
        0,
        ge=0,
        description="Number of records that succeeded",
    )
    failed_records: int = Field(
        0,
        ge=0,
        description="Number of records that failed",
    )
    progress_pct: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages for failed records",
    )
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Individual operation results",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Total processing duration in milliseconds",
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 provenance hash",
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Batch job start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Batch job completion timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional batch metadata",
    )
