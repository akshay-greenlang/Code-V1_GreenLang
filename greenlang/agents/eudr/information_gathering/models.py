# -*- coding: utf-8 -*-
"""
Information Gathering Agent Models - AGENT-EUDR-027

Pydantic v2 models for information gathering operations, external database
queries, certification verification, supplier profiles, completeness
validation, normalization records, and information packages.

All models use Decimal for numeric scores to ensure deterministic,
bit-perfect reproducibility in compliance calculations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-027 Information Gathering Agent (GL-EUDR-IGA-027)
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 12, 13, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExternalDatabaseSource(str, enum.Enum):
    """Supported external database sources."""

    EU_TRACES = "eu_traces"
    CITES = "cites"
    FLEGT_VPA = "flegt_vpa"
    UN_COMTRADE = "un_comtrade"
    FAO_STAT = "fao_stat"
    GLOBAL_FOREST_WATCH = "global_forest_watch"
    WORLD_BANK_WGI = "world_bank_wgi"
    TRANSPARENCY_CPI = "transparency_cpi"
    EU_SANCTIONS = "eu_sanctions"
    NATIONAL_CUSTOMS = "national_customs"
    NATIONAL_LAND_REGISTRY = "national_land_registry"


class CertificationBody(str, enum.Enum):
    """Supported certification body connectors."""

    FSC = "fsc"
    RSPO = "rspo"
    PEFC = "pefc"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    UTZ = "utz"
    EU_ORGANIC = "eu_organic"


class EUDRCommodity(str, enum.Enum):
    """EUDR regulated commodities (Article 1)."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class QueryStatus(str, enum.Enum):
    """Status of an external database query."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CACHED = "cached"
    TIMEOUT = "timeout"


class CertVerificationStatus(str, enum.Enum):
    """Certificate verification result status."""

    VALID = "valid"
    EXPIRED = "expired"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"
    NOT_FOUND = "not_found"
    ERROR = "error"


class CompletenessClassification(str, enum.Enum):
    """Information completeness classification."""

    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    COMPLETE = "complete"


class NormalizationType(str, enum.Enum):
    """Data normalization transformation types."""

    UNIT = "unit"
    DATE = "date"
    COORDINATE = "coordinate"
    CURRENCY = "currency"
    COUNTRY_CODE = "country_code"
    PRODUCT_CODE = "product_code"
    ADDRESS = "address"
    CERTIFICATE_ID = "certificate_id"


class GatheringOperationStatus(str, enum.Enum):
    """Information gathering operation lifecycle status."""

    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FreshnessStatus(str, enum.Enum):
    """Data freshness classification."""

    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class DataSourcePriority(str, enum.Enum):
    """Source authority ranking for conflict resolution."""

    GOVERNMENT_REGISTRY = "government_registry"
    CERTIFICATION_BODY = "certification_body"
    CUSTOMS_RECORD = "customs_record"
    TRADE_DATABASE = "trade_database"
    SUPPLIER_SELF_DECLARED = "supplier_self_declared"
    PUBLIC_DATABASE = "public_database"


class Article9ElementName(str, enum.Enum):
    """The 10 mandatory Article 9 information elements."""

    PRODUCT_DESCRIPTION = "product_description"
    QUANTITY = "quantity"
    COUNTRY_OF_PRODUCTION = "country_of_production"
    GEOLOCATION = "geolocation"
    PRODUCTION_DATE_RANGE = "production_date_range"
    SUPPLIER_IDENTIFICATION = "supplier_identification"
    BUYER_IDENTIFICATION = "buyer_identification"
    DEFORESTATION_FREE_EVIDENCE = "deforestation_free_evidence"
    LEGAL_PRODUCTION_EVIDENCE = "legal_production_evidence"
    SUPPLY_CHAIN_INFORMATION = "supply_chain_information"


class ElementStatus(str, enum.Enum):
    """Per-element status in completeness validation."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTICLE_9_ELEMENTS: List[str] = [e.value for e in Article9ElementName]

ARTICLE_9_DEFAULT_WEIGHTS: Dict[str, Decimal] = {
    e.value: Decimal("0.10") for e in Article9ElementName
}

SUPPORTED_COMMODITIES: List[str] = [c.value for c in EUDRCommodity]

CERTIFICATION_COMMODITY_MAP: Dict[str, List[str]] = {
    CertificationBody.FSC.value: [EUDRCommodity.WOOD.value],
    CertificationBody.RSPO.value: [EUDRCommodity.OIL_PALM.value],
    CertificationBody.PEFC.value: [EUDRCommodity.WOOD.value],
    CertificationBody.RAINFOREST_ALLIANCE.value: [
        EUDRCommodity.COCOA.value,
        EUDRCommodity.COFFEE.value,
    ],
    CertificationBody.UTZ.value: [
        EUDRCommodity.COCOA.value,
        EUDRCommodity.COFFEE.value,
    ],
    CertificationBody.EU_ORGANIC.value: [c.value for c in EUDRCommodity],
}

#: Countries with FLEGT VPA partnerships requiring FLEGT license verification.
FLEGT_VPA_COUNTRIES: List[str] = [
    "CM", "CF", "GH", "GY", "HN", "ID", "LA", "LR", "MY",
    "CG", "CD", "TH", "VN",
]

#: Low-risk countries eligible for simplified due diligence (Article 13).
LOW_RISK_COUNTRIES: List[str] = []  # Populated from EC benchmark publications


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """Result of an external database query."""

    query_id: str
    source: ExternalDatabaseSource
    query_parameters: Dict[str, Any] = Field(default_factory=dict)
    status: QueryStatus = QueryStatus.SUCCESS
    records: List[Dict[str, Any]] = Field(default_factory=list)
    record_count: int = 0
    query_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: int = 0
    cached: bool = False
    cache_age_seconds: Optional[int] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class CertificateVerificationResult(BaseModel):
    """Result of verifying a single certificate."""

    certificate_id: str
    certification_body: CertificationBody
    holder_name: str = ""
    verification_status: CertVerificationStatus = CertVerificationStatus.NOT_FOUND
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    scope: List[str] = Field(default_factory=list)
    commodity_scope: List[EUDRCommodity] = Field(default_factory=list)
    chain_of_custody_model: Optional[str] = None
    days_until_expiry: Optional[int] = None
    last_verified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""


class DataDiscrepancy(BaseModel):
    """A discrepancy detected between two data sources for the same field."""

    field_name: str
    source_a: str
    value_a: str
    source_b: str
    value_b: str
    severity: str = "medium"  # low, medium, high, critical
    recommendation: str = ""


class SupplierProfile(BaseModel):
    """Unified supplier information profile aggregated from multiple sources."""

    supplier_id: str
    name: str
    alternative_names: List[str] = Field(default_factory=list)
    postal_address: str = ""
    country_code: str = ""
    email: Optional[str] = None
    registration_number: Optional[str] = None
    commodities: List[EUDRCommodity] = Field(default_factory=list)
    certifications: List[CertificateVerificationResult] = Field(default_factory=list)
    plot_ids: List[str] = Field(default_factory=list)
    tier_depth: int = 0
    data_sources: List[str] = Field(default_factory=list)
    completeness_score: Decimal = Decimal("0")
    confidence_score: Decimal = Decimal("0")
    discrepancies: List[DataDiscrepancy] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""


class Article9ElementStatus(BaseModel):
    """Status of a single Article 9 information element."""

    element_name: str
    status: ElementStatus = ElementStatus.MISSING
    source: str = ""
    value_summary: str = ""
    confidence: Decimal = Decimal("0")
    last_updated: Optional[datetime] = None


class NormalizationRecord(BaseModel):
    """Audit record for a single data normalization transformation."""

    field_name: str
    source_value: str
    normalized_value: str
    normalization_type: NormalizationType
    confidence: Decimal = Decimal("1.0")


class GapReportItem(BaseModel):
    """A single gap identified in the information package."""

    element_name: str
    gap_type: str  # missing, partial, stale, inconsistent
    severity: str  # critical, high, medium, low
    remediation_action: str
    estimated_effort: str = ""  # e.g., "1 day", "send questionnaire"


class GapReport(BaseModel):
    """Aggregated gap analysis report for an information package."""

    total_gaps: int = 0
    critical_gaps: int = 0
    high_gaps: int = 0
    medium_gaps: int = 0
    low_gaps: int = 0
    items: List[GapReportItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvidenceArtifact(BaseModel):
    """A single evidence artifact within an information package."""

    artifact_id: str
    article_9_element: str
    source: str
    format: str = "json"  # json, pdf, csv, xml
    content_hash: str = ""
    s3_path: Optional[str] = None
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProvenanceEntry(BaseModel):
    """A single step in the provenance chain."""

    step: str
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str = "AGENT-EUDR-027"
    input_hash: str = ""
    output_hash: str = ""


class HarvestResult(BaseModel):
    """Result of a public data harvest operation."""

    source: ExternalDatabaseSource
    data_type: str
    country_code: Optional[str] = None
    commodity: Optional[str] = None
    records_harvested: int = 0
    data_timestamp: Optional[datetime] = None
    is_incremental: bool = False
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH
    provenance_hash: str = ""
    harvested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DataFreshnessRecord(BaseModel):
    """Freshness status for a single data source."""

    source: str
    data_type: str
    last_updated: datetime
    next_expected_update: Optional[datetime] = None
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH
    max_age_hours: int = 24


class CompletenessReport(BaseModel):
    """Full Article 9 completeness validation report."""

    operation_id: str
    commodity: EUDRCommodity
    elements: List[Article9ElementStatus] = Field(default_factory=list)
    completeness_score: Decimal = Decimal("0")
    completeness_classification: CompletenessClassification = (
        CompletenessClassification.INSUFFICIENT
    )
    gap_report: GapReport = Field(default_factory=GapReport)
    is_simplified_dd: bool = False
    validated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""


class InformationPackage(BaseModel):
    """Assembled information package for DDS submission."""

    package_id: str
    operator_id: str
    commodity: EUDRCommodity
    version: int = 1
    article_9_elements: Dict[str, Article9ElementStatus] = Field(default_factory=dict)
    completeness_score: Decimal = Decimal("0")
    completeness_classification: str = "insufficient"
    supplier_profiles: List[SupplierProfile] = Field(default_factory=list)
    external_data: Dict[str, List[QueryResult]] = Field(default_factory=dict)
    certification_results: List[CertificateVerificationResult] = Field(
        default_factory=list
    )
    public_data: Dict[str, Any] = Field(default_factory=dict)
    normalization_log: List[NormalizationRecord] = Field(default_factory=list)
    gap_report: GapReport = Field(default_factory=GapReport)
    evidence_artifacts: List[EvidenceArtifact] = Field(default_factory=list)
    provenance_chain: List[ProvenanceEntry] = Field(default_factory=list)
    package_hash: str = ""
    assembled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None


class GatheringOperation(BaseModel):
    """Top-level information gathering operation tracking."""

    operation_id: str
    operator_id: str
    commodity: EUDRCommodity
    workflow_id: Optional[str] = None
    status: GatheringOperationStatus = GatheringOperationStatus.INITIATED
    sources_queried: List[str] = Field(default_factory=list)
    sources_completed: List[str] = Field(default_factory=list)
    sources_failed: List[str] = Field(default_factory=list)
    completeness_score: Decimal = Decimal("0")
    completeness_classification: Optional[str] = None
    total_records_collected: int = 0
    total_suppliers_resolved: int = 0
    total_certificates_verified: int = 0
    package_id: Optional[str] = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    provenance_hash: str = ""


class PackageDiff(BaseModel):
    """Difference between two information package versions."""

    package_a_id: str
    package_b_id: str
    added_elements: List[str] = Field(default_factory=list)
    removed_elements: List[str] = Field(default_factory=list)
    changed_elements: List[str] = Field(default_factory=list)
    score_delta: Decimal = Decimal("0")
    compared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
