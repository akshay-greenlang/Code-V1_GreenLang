# -*- coding: utf-8 -*-
"""
Spend Data Categorizer Service Data Models - AGENT-DATA-009

Pydantic v2 data models for the Spend Data Categorizer SDK. Re-exports the
Layer 1 enumerations and models from foundation agents (Scope 3 Category
Mapper, ERP Connector, Procurement Carbon Footprint), and defines additional
SDK models for spend ingestion, taxonomy classification, Scope 3 assignment,
emission factor lookup, emission calculation, category rules, aggregation,
hotspot analysis, trend tracking, reporting, and batch management.

Re-exported Layer 1 sources:
    - greenlang.agents.mrv.scope3_category_mapper: Scope3Category,
        DataSourceType, CalculationApproach, SpendRecord (as L1SpendRecord),
        PurchaseOrder (as L1PurchaseOrder), BOMItem, CategoryMappingResult,
        NAICS_TO_CATEGORY, SPEND_KEYWORDS_TO_CATEGORY
    - greenlang.agents.data.erp_connector_agent: ERPSystem,
        SpendCategory (as ERPSpendCategory), TransactionType,
        SPEND_TO_SCOPE3_MAPPING, DEFAULT_EMISSION_FACTORS
    - greenlang.agents.procurement.procurement_carbon_footprint:
        CalculationMethod, ProcurementItem, EmissionCalculation

New enumerations (12):
    - TaxonomySystem, IngestionSource, RecordStatus,
      ClassificationConfidence, EmissionFactorSource, EmissionFactorUnit,
      CurrencyCode, AnalyticsTimeframe, ReportFormat, RuleMatchType,
      RulePriority, HotspotType

New SDK models (15):
    - SpendRecord, NormalizedSpendRecord, TaxonomyCode,
      TaxonomyClassification, Scope3Assignment, EmissionFactor,
      EmissionCalculationResult, CategoryRule, SpendAggregate,
      HotspotResult, TrendDataPoint, CategorizationReport,
      IngestionBatch, SpendCategorizerStatistics, VendorProfile

Request models (7):
    - IngestSpendRequest, ClassifyRequest, MapScope3Request,
      CalculateEmissionsRequest, CreateRuleRequest, AnalyticsRequest,
      GenerateReportRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 enumerations from scope3_category_mapper
# ---------------------------------------------------------------------------

from greenlang.agents.mrv.scope3_category_mapper import (
    Scope3Category,
    DataSourceType,
    CalculationApproach,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from scope3_category_mapper (aliased to avoid
# collision with this module's own SpendRecord / PurchaseOrder)
# ---------------------------------------------------------------------------

from greenlang.agents.mrv.scope3_category_mapper import (
    SpendRecord as L1SpendRecord,
    PurchaseOrder as L1PurchaseOrder,
    BOMItem,
    CategoryMappingResult,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 constants from scope3_category_mapper
# ---------------------------------------------------------------------------

from greenlang.agents.mrv.scope3_category_mapper import (
    NAICS_TO_CATEGORY,
    SPEND_KEYWORDS_TO_CATEGORY,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 enumerations from erp_connector_agent
# ---------------------------------------------------------------------------

from greenlang.agents.data.erp_connector_agent import (
    ERPSystem,
    TransactionType,
)

# Re-export SpendCategory from ERP connector aliased to avoid collision with
# procurement_carbon_footprint.SpendCategory
from greenlang.agents.data.erp_connector_agent import (
    SpendCategory as ERPSpendCategory,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 constants from erp_connector_agent
# ---------------------------------------------------------------------------

from greenlang.agents.data.erp_connector_agent import (
    SPEND_TO_SCOPE3_MAPPING,
    DEFAULT_EMISSION_FACTORS,
)

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from procurement_carbon_footprint
# ---------------------------------------------------------------------------

from greenlang.agents.procurement.procurement_carbon_footprint import (
    CalculationMethod,
    ProcurementItem,
    EmissionCalculation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# New Enumerations (12)
# =============================================================================


class TaxonomySystem(str, Enum):
    """Taxonomy classification systems for spend categorization.

    Each system provides a hierarchical code structure for classifying
    goods and services in procurement and spend data.
    """

    UNSPSC = "unspsc"
    NAICS = "naics"
    ECLASS = "eclass"
    ISIC = "isic"
    SIC = "sic"
    CPV = "cpv"
    HS_CN = "hs_cn"


class IngestionSource(str, Enum):
    """Source types for spend data ingestion.

    Identifies the origin system or file format from which spend
    records were extracted or uploaded.
    """

    ERP_EXTRACT = "erp_extract"
    CSV_FILE = "csv_file"
    EXCEL_FILE = "excel_file"
    API_FEED = "api_feed"
    MANUAL_ENTRY = "manual_entry"
    PROCUREMENT_PLATFORM = "procurement_platform"


class RecordStatus(str, Enum):
    """Lifecycle status of a spend record through the pipeline.

    Records progress through these stages sequentially from
    ingestion through final export.
    """

    RAW = "raw"
    NORMALIZED = "normalized"
    CLASSIFIED = "classified"
    MAPPED = "mapped"
    CALCULATED = "calculated"
    VALIDATED = "validated"
    EXPORTED = "exported"


class ClassificationConfidence(str, Enum):
    """Classification confidence levels based on score thresholds.

    HIGH: confidence >= 0.8
    MEDIUM: confidence >= 0.5
    LOW: confidence >= 0.3
    UNCLASSIFIED: confidence < 0.3
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCLASSIFIED = "unclassified"


class EmissionFactorSource(str, Enum):
    """Sources for emission factor data.

    Identifies the authoritative database or method used to obtain
    the emission factor applied to a spend record.
    """

    EPA_EEIO = "epa_eeio"
    EXIOBASE = "exiobase"
    DEFRA = "defra"
    ECOINVENT = "ecoinvent"
    CUSTOM = "custom"
    SUPPLIER_SPECIFIC = "supplier_specific"


class EmissionFactorUnit(str, Enum):
    """Units for emission factor values.

    Defines the denominator unit for kgCO2e emission factors
    used in spend-based and activity-based calculations.
    """

    KG_CO2E_PER_USD = "kgCO2e/USD"
    KG_CO2E_PER_EUR = "kgCO2e/EUR"
    KG_CO2E_PER_KG = "kgCO2e/kg"
    KG_CO2E_PER_UNIT = "kgCO2e/unit"
    KG_CO2E_PER_KWH = "kgCO2e/kWh"


class CurrencyCode(str, Enum):
    """ISO 4217 currency codes for the 12 major trading currencies.

    Used for spend amount denomination and currency conversion
    operations throughout the categorizer pipeline.
    """

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CNY = "CNY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    INR = "INR"
    BRL = "BRL"
    KRW = "KRW"
    MXN = "MXN"


class AnalyticsTimeframe(str, Enum):
    """Timeframe options for spend and emissions analytics.

    Defines the aggregation period for analytics queries
    and trend analysis.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Output format for categorization reports.

    Defines the serialization format for generated reports
    including analytics summaries and compliance outputs.
    """

    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


class RuleMatchType(str, Enum):
    """Match types for category classification rules.

    Defines how the rule pattern is matched against spend
    record fields (description, vendor name, category).
    """

    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"
    FUZZY = "fuzzy"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


class RulePriority(str, Enum):
    """Priority levels for classification rules.

    Higher priority rules are evaluated first during
    classification. CRITICAL rules override all others.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFAULT = "default"


class HotspotType(str, Enum):
    """Types of hotspot analysis for spend and emissions.

    Identifies the dimension used to rank categories or
    vendors in hotspot identification.
    """

    TOP_SPEND = "top_spend"
    TOP_EMISSIONS = "top_emissions"
    TOP_INTENSITY = "top_intensity"
    RISING_TREND = "rising_trend"


# =============================================================================
# SDK Data Models (15)
# =============================================================================


class SpendRecord(BaseModel):
    """Ingested spend record with all fields for categorization.

    Represents a single financial transaction or spend line item
    as received from ERP systems, CSV files, or manual entry,
    prior to normalization and classification.

    Attributes:
        record_id: Unique identifier for this spend record.
        ingestion_source: Source system or file from which record was ingested.
        transaction_date: Date of the financial transaction.
        transaction_type: Type of financial transaction.
        vendor_id: Vendor or supplier identifier.
        vendor_name: Vendor or supplier display name.
        description: Transaction description or line item text.
        amount: Spend amount in original currency.
        currency: ISO 4217 currency code for the amount.
        amount_usd: Amount converted to USD (if available).
        cost_center: Cost center or department code.
        gl_account: General ledger account code.
        material_group: Material or product group code.
        po_number: Associated purchase order number.
        invoice_number: Associated invoice number.
        naics_code: NAICS industry code for the vendor.
        unspsc_code: UNSPSC product code for the line item.
        status: Current lifecycle status of the record.
        tenant_id: Tenant identifier for multi-tenant isolation.
        created_at: Timestamp when the record was ingested.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this spend record",
    )
    ingestion_source: IngestionSource = Field(
        default=IngestionSource.MANUAL_ENTRY,
        description="Source system or file from which record was ingested",
    )
    transaction_date: date = Field(
        ..., description="Date of the financial transaction",
    )
    transaction_type: Optional[TransactionType] = Field(
        None, description="Type of financial transaction",
    )
    vendor_id: str = Field(
        ..., description="Vendor or supplier identifier",
    )
    vendor_name: str = Field(
        ..., description="Vendor or supplier display name",
    )
    description: Optional[str] = Field(
        None, description="Transaction description or line item text",
    )
    amount: float = Field(
        ..., gt=0.0, description="Spend amount in original currency",
    )
    currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="ISO 4217 currency code for the amount",
    )
    amount_usd: Optional[float] = Field(
        None, ge=0.0, description="Amount converted to USD (if available)",
    )
    cost_center: Optional[str] = Field(
        None, description="Cost center or department code",
    )
    gl_account: Optional[str] = Field(
        None, description="General ledger account code",
    )
    material_group: Optional[str] = Field(
        None, description="Material or product group code",
    )
    po_number: Optional[str] = Field(
        None, description="Associated purchase order number",
    )
    invoice_number: Optional[str] = Field(
        None, description="Associated invoice number",
    )
    naics_code: Optional[str] = Field(
        None, description="NAICS industry code for the vendor",
    )
    unspsc_code: Optional[str] = Field(
        None, description="UNSPSC product code for the line item",
    )
    status: RecordStatus = Field(
        default=RecordStatus.RAW,
        description="Current lifecycle status of the record",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the record was ingested",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("vendor_id")
    @classmethod
    def validate_vendor_id(cls, v: str) -> str:
        """Validate vendor_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("vendor_id must be non-empty")
        return v

    @field_validator("vendor_name")
    @classmethod
    def validate_vendor_name(cls, v: str) -> str:
        """Validate vendor_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("vendor_name must be non-empty")
        return v


class NormalizedSpendRecord(BaseModel):
    """Spend record after normalization and deduplication.

    Extends the raw spend record with normalized fields including
    standardized vendor names, resolved currency amounts, and
    deduplication metadata.

    Attributes:
        record_id: Unique identifier (same as source SpendRecord).
        original_record_id: Reference to the raw SpendRecord ID.
        normalized_vendor_name: Standardized vendor name after normalization.
        normalized_description: Cleaned and standardized description text.
        amount_usd: Spend amount converted to USD.
        exchange_rate: Exchange rate used for currency conversion.
        is_duplicate: Whether this record was flagged as a duplicate.
        duplicate_of: Record ID of the primary record if this is a duplicate.
        dedup_similarity: Similarity score used in deduplication matching.
        status: Current lifecycle status (NORMALIZED or later).
        normalized_at: Timestamp when normalization was performed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    record_id: str = Field(
        ..., description="Unique identifier (same as source SpendRecord)",
    )
    original_record_id: str = Field(
        ..., description="Reference to the raw SpendRecord ID",
    )
    normalized_vendor_name: str = Field(
        ..., description="Standardized vendor name after normalization",
    )
    normalized_description: Optional[str] = Field(
        None, description="Cleaned and standardized description text",
    )
    amount_usd: float = Field(
        ..., ge=0.0, description="Spend amount converted to USD",
    )
    exchange_rate: float = Field(
        default=1.0, gt=0.0,
        description="Exchange rate used for currency conversion",
    )
    is_duplicate: bool = Field(
        default=False,
        description="Whether this record was flagged as a duplicate",
    )
    duplicate_of: Optional[str] = Field(
        None,
        description="Record ID of the primary record if this is a duplicate",
    )
    dedup_similarity: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Similarity score used in deduplication matching",
    )
    status: RecordStatus = Field(
        default=RecordStatus.NORMALIZED,
        description="Current lifecycle status (NORMALIZED or later)",
    )
    normalized_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when normalization was performed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, v: str) -> str:
        """Validate record_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_id must be non-empty")
        return v

    @field_validator("original_record_id")
    @classmethod
    def validate_original_record_id(cls, v: str) -> str:
        """Validate original_record_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("original_record_id must be non-empty")
        return v

    @field_validator("normalized_vendor_name")
    @classmethod
    def validate_normalized_vendor_name(cls, v: str) -> str:
        """Validate normalized_vendor_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("normalized_vendor_name must be non-empty")
        return v


class TaxonomyCode(BaseModel):
    """A taxonomy code from a classification system.

    Represents a single node in a hierarchical taxonomy tree
    such as UNSPSC, NAICS, eCl@ss, or ISIC. Supports multi-level
    code resolution with parent references.

    Attributes:
        code: Taxonomy code value (e.g. '43211500' for UNSPSC).
        system: Taxonomy system this code belongs to.
        level: Depth level in the taxonomy hierarchy (1 = top).
        description: Human-readable description of the code.
        parent_code: Code of the parent node in the hierarchy.
    """

    code: str = Field(
        ..., description="Taxonomy code value",
    )
    system: TaxonomySystem = Field(
        ..., description="Taxonomy system this code belongs to",
    )
    level: int = Field(
        ..., ge=1, le=8,
        description="Depth level in the taxonomy hierarchy (1 = top)",
    )
    description: str = Field(
        default="", description="Human-readable description of the code",
    )
    parent_code: Optional[str] = Field(
        None, description="Code of the parent node in the hierarchy",
    )

    model_config = {"extra": "forbid"}

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Validate code is non-empty."""
        if not v or not v.strip():
            raise ValueError("code must be non-empty")
        return v


class TaxonomyClassification(BaseModel):
    """Result of taxonomy classification for a spend record.

    Links a normalized spend record to one or more taxonomy codes
    with confidence scoring and the rule or method used for
    classification.

    Attributes:
        classification_id: Unique identifier for this classification.
        record_id: Spend record that was classified.
        primary_code: Primary taxonomy code assigned.
        secondary_codes: Additional taxonomy codes if multi-classified.
        confidence: Classification confidence score (0.0 to 1.0).
        confidence_level: Confidence level label (HIGH/MEDIUM/LOW/UNCLASSIFIED).
        classification_method: Method used for classification.
        rule_id: ID of the CategoryRule used (if rule-based).
        classified_at: Timestamp when classification was performed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    classification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this classification",
    )
    record_id: str = Field(
        ..., description="Spend record that was classified",
    )
    primary_code: TaxonomyCode = Field(
        ..., description="Primary taxonomy code assigned",
    )
    secondary_codes: List[TaxonomyCode] = Field(
        default_factory=list,
        description="Additional taxonomy codes if multi-classified",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Classification confidence score (0.0 to 1.0)",
    )
    confidence_level: ClassificationConfidence = Field(
        ..., description="Confidence level label",
    )
    classification_method: str = Field(
        default="rule_based",
        description="Method used for classification",
    )
    rule_id: Optional[str] = Field(
        None, description="ID of the CategoryRule used (if rule-based)",
    )
    classified_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when classification was performed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, v: str) -> str:
        """Validate record_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_id must be non-empty")
        return v


class Scope3Assignment(BaseModel):
    """Result of Scope 3 category assignment for a spend record.

    Maps a classified spend record to a GHG Protocol Scope 3
    category with confidence scoring, the rule used, and the
    recommended calculation approach.

    Attributes:
        assignment_id: Unique identifier for this assignment.
        record_id: Spend record that was assigned.
        scope3_category: GHG Protocol Scope 3 category assigned.
        category_number: Scope 3 category number (1 through 15).
        category_name: Human-readable Scope 3 category name.
        confidence: Assignment confidence score (0.0 to 1.0).
        confidence_level: Confidence level label.
        mapping_rule: Rule or method used for the mapping.
        recommended_approach: Recommended calculation approach.
        taxonomy_code: Taxonomy code used for the mapping (if any).
        assigned_at: Timestamp when assignment was performed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    assignment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this assignment",
    )
    record_id: str = Field(
        ..., description="Spend record that was assigned",
    )
    scope3_category: Scope3Category = Field(
        ..., description="GHG Protocol Scope 3 category assigned",
    )
    category_number: int = Field(
        ..., ge=1, le=15,
        description="Scope 3 category number (1 through 15)",
    )
    category_name: str = Field(
        ..., description="Human-readable Scope 3 category name",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Assignment confidence score (0.0 to 1.0)",
    )
    confidence_level: ClassificationConfidence = Field(
        ..., description="Confidence level label",
    )
    mapping_rule: str = Field(
        default="default",
        description="Rule or method used for the mapping",
    )
    recommended_approach: CalculationApproach = Field(
        default=CalculationApproach.SPEND_BASED,
        description="Recommended calculation approach",
    )
    taxonomy_code: Optional[str] = Field(
        None, description="Taxonomy code used for the mapping (if any)",
    )
    assigned_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when assignment was performed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, v: str) -> str:
        """Validate record_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_id must be non-empty")
        return v

    @field_validator("category_name")
    @classmethod
    def validate_category_name(cls, v: str) -> str:
        """Validate category_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("category_name must be non-empty")
        return v


class EmissionFactor(BaseModel):
    """An emission factor from an authoritative source.

    Represents a single emission conversion factor used to estimate
    greenhouse gas emissions from spend or activity data, with full
    source attribution and regional/temporal applicability.

    Attributes:
        factor_id: Unique identifier for this emission factor.
        taxonomy_code: Taxonomy code this factor applies to.
        taxonomy_system: Taxonomy system for the code.
        source: Authoritative source of the factor.
        source_version: Version of the source database.
        value: Emission factor value.
        unit: Unit of measurement for the factor.
        region: Geographic region the factor applies to.
        year: Reference year for the factor.
        description: Human-readable description of the factor.
        data_quality_score: Quality score for this factor (0.0 to 1.0).
        valid_from: Start date of factor validity period.
        valid_to: End date of factor validity period.
    """

    factor_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this emission factor",
    )
    taxonomy_code: str = Field(
        ..., description="Taxonomy code this factor applies to",
    )
    taxonomy_system: TaxonomySystem = Field(
        default=TaxonomySystem.NAICS,
        description="Taxonomy system for the code",
    )
    source: EmissionFactorSource = Field(
        ..., description="Authoritative source of the factor",
    )
    source_version: str = Field(
        default="", description="Version of the source database",
    )
    value: float = Field(
        ..., ge=0.0, description="Emission factor value",
    )
    unit: EmissionFactorUnit = Field(
        default=EmissionFactorUnit.KG_CO2E_PER_USD,
        description="Unit of measurement for the factor",
    )
    region: str = Field(
        default="global",
        description="Geographic region the factor applies to",
    )
    year: int = Field(
        default=2024, ge=1990, le=2100,
        description="Reference year for the factor",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the factor",
    )
    data_quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Quality score for this factor (0.0 to 1.0)",
    )
    valid_from: Optional[date] = Field(
        None, description="Start date of factor validity period",
    )
    valid_to: Optional[date] = Field(
        None, description="End date of factor validity period",
    )

    model_config = {"extra": "forbid"}

    @field_validator("taxonomy_code")
    @classmethod
    def validate_taxonomy_code(cls, v: str) -> str:
        """Validate taxonomy_code is non-empty."""
        if not v or not v.strip():
            raise ValueError("taxonomy_code must be non-empty")
        return v


class EmissionCalculationResult(BaseModel):
    """Result of an emissions calculation for a single spend record.

    Contains the calculated emissions estimate with full traceability
    to the source spend record, emission factor, calculation method,
    and provenance chain for audit purposes.

    Attributes:
        result_id: Unique identifier for this calculation result.
        record_id: Source spend record identifier.
        vendor_id: Vendor associated with the spend.
        amount_usd: Spend amount in USD used for calculation.
        emission_factor_id: Emission factor used for calculation.
        emission_factor_value: Numeric value of the emission factor.
        emission_factor_source: Source of the emission factor.
        emission_factor_unit: Unit of the emission factor.
        calculation_method: Method used for calculation.
        emissions_kgco2e: Calculated emissions in kilograms CO2 equivalent.
        emissions_tco2e: Calculated emissions in tonnes CO2 equivalent.
        data_quality_score: Quality score of the calculation (0.0 to 1.0).
        scope3_category: Scope 3 category for the emissions.
        calculated_at: Timestamp when calculation was performed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    result_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this calculation result",
    )
    record_id: str = Field(
        ..., description="Source spend record identifier",
    )
    vendor_id: str = Field(
        ..., description="Vendor associated with the spend",
    )
    amount_usd: float = Field(
        ..., ge=0.0,
        description="Spend amount in USD used for calculation",
    )
    emission_factor_id: Optional[str] = Field(
        None, description="Emission factor used for calculation",
    )
    emission_factor_value: float = Field(
        ..., ge=0.0,
        description="Numeric value of the emission factor",
    )
    emission_factor_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA_EEIO,
        description="Source of the emission factor",
    )
    emission_factor_unit: EmissionFactorUnit = Field(
        default=EmissionFactorUnit.KG_CO2E_PER_USD,
        description="Unit of the emission factor",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.SPEND_BASED,
        description="Method used for calculation",
    )
    emissions_kgco2e: float = Field(
        ..., ge=0.0,
        description="Calculated emissions in kilograms CO2 equivalent",
    )
    emissions_tco2e: float = Field(
        ..., ge=0.0,
        description="Calculated emissions in tonnes CO2 equivalent",
    )
    data_quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Quality score of the calculation (0.0 to 1.0)",
    )
    scope3_category: Optional[Scope3Category] = Field(
        None, description="Scope 3 category for the emissions",
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when calculation was performed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, v: str) -> str:
        """Validate record_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("record_id must be non-empty")
        return v

    @field_validator("vendor_id")
    @classmethod
    def validate_vendor_id(cls, v: str) -> str:
        """Validate vendor_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("vendor_id must be non-empty")
        return v


class CategoryRule(BaseModel):
    """A classification rule for automated spend categorization.

    Defines a pattern-based rule that maps spend record fields
    (description, vendor name, GL account) to taxonomy codes
    and Scope 3 categories for deterministic classification.

    Attributes:
        rule_id: Unique identifier for this rule.
        name: Human-readable rule name.
        description: Detailed description of the rule purpose.
        match_type: Pattern matching strategy.
        match_field: Field to match against (description, vendor_name, etc.).
        pattern: Pattern string to match.
        target_taxonomy_system: Taxonomy system for the target code.
        target_taxonomy_code: Target taxonomy code to assign on match.
        target_scope3_category: Target Scope 3 category to assign on match.
        priority: Rule evaluation priority.
        confidence_boost: Confidence score adjustment on match.
        is_active: Whether this rule is currently active.
        created_by: User or system that created the rule.
        created_at: Timestamp when the rule was created.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    rule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this rule",
    )
    name: str = Field(
        ..., description="Human-readable rule name",
    )
    description: str = Field(
        default="", description="Detailed description of the rule purpose",
    )
    match_type: RuleMatchType = Field(
        ..., description="Pattern matching strategy",
    )
    match_field: str = Field(
        default="description",
        description="Field to match against (description, vendor_name, etc.)",
    )
    pattern: str = Field(
        ..., description="Pattern string to match",
    )
    target_taxonomy_system: Optional[TaxonomySystem] = Field(
        None, description="Taxonomy system for the target code",
    )
    target_taxonomy_code: Optional[str] = Field(
        None, description="Target taxonomy code to assign on match",
    )
    target_scope3_category: Optional[Scope3Category] = Field(
        None, description="Target Scope 3 category to assign on match",
    )
    priority: RulePriority = Field(
        default=RulePriority.MEDIUM,
        description="Rule evaluation priority",
    )
    confidence_boost: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Confidence score adjustment on match",
    )
    is_active: bool = Field(
        default=True, description="Whether this rule is currently active",
    )
    created_by: str = Field(
        default="system",
        description="User or system that created the rule",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the rule was created",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate pattern is non-empty."""
        if not v or not v.strip():
            raise ValueError("pattern must be non-empty")
        return v


class SpendAggregate(BaseModel):
    """Aggregated spend and emissions data for a single category.

    Provides summary statistics for a taxonomy code, Scope 3
    category, or vendor grouping including spend totals, emissions,
    and intensity metrics.

    Attributes:
        category: Category identifier (taxonomy code or Scope 3 name).
        category_label: Human-readable category label.
        total_spend_usd: Total spend amount in USD for this category.
        total_emissions_kgco2e: Total emissions in kgCO2e for this category.
        record_count: Number of spend records in this category.
        vendor_count: Number of unique vendors in this category.
        avg_intensity_kgco2e_per_usd: Average emissions intensity.
        percentage_of_total_spend: Percentage of total spend this represents.
        percentage_of_total_emissions: Percentage of total emissions.
    """

    category: str = Field(
        ..., description="Category identifier",
    )
    category_label: str = Field(
        default="", description="Human-readable category label",
    )
    total_spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Total spend amount in USD for this category",
    )
    total_emissions_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total emissions in kgCO2e for this category",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Number of spend records in this category",
    )
    vendor_count: int = Field(
        default=0, ge=0,
        description="Number of unique vendors in this category",
    )
    avg_intensity_kgco2e_per_usd: float = Field(
        default=0.0, ge=0.0,
        description="Average emissions intensity (kgCO2e per USD)",
    )
    percentage_of_total_spend: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of total spend this represents",
    )
    percentage_of_total_emissions: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of total emissions this represents",
    )

    model_config = {"extra": "forbid"}

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate category is non-empty."""
        if not v or not v.strip():
            raise ValueError("category must be non-empty")
        return v


class HotspotResult(BaseModel):
    """Result of hotspot analysis identifying high-impact categories.

    Identifies categories, vendors, or spend areas that represent
    disproportionately high spend, emissions, or emissions intensity
    for targeted reduction efforts.

    Attributes:
        hotspot_id: Unique identifier for this hotspot result.
        hotspot_type: Dimension used for hotspot ranking.
        category: Category or vendor identifier.
        category_label: Human-readable label for the hotspot.
        spend_usd: Total spend in USD for this hotspot.
        emissions_kgco2e: Total emissions in kgCO2e for this hotspot.
        intensity_kgco2e_per_usd: Emissions intensity for this hotspot.
        rank: Rank position within the hotspot type (1 = highest).
        percentile: Percentile ranking (0.0 to 100.0).
        record_count: Number of underlying spend records.
        recommendation: Suggested action for this hotspot.
    """

    hotspot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this hotspot result",
    )
    hotspot_type: HotspotType = Field(
        ..., description="Dimension used for hotspot ranking",
    )
    category: str = Field(
        ..., description="Category or vendor identifier",
    )
    category_label: str = Field(
        default="", description="Human-readable label for the hotspot",
    )
    spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Total spend in USD for this hotspot",
    )
    emissions_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total emissions in kgCO2e for this hotspot",
    )
    intensity_kgco2e_per_usd: float = Field(
        default=0.0, ge=0.0,
        description="Emissions intensity for this hotspot",
    )
    rank: int = Field(
        ..., ge=1,
        description="Rank position within the hotspot type (1 = highest)",
    )
    percentile: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentile ranking (0.0 to 100.0)",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Number of underlying spend records",
    )
    recommendation: str = Field(
        default="",
        description="Suggested action for this hotspot",
    )

    model_config = {"extra": "forbid"}

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate category is non-empty."""
        if not v or not v.strip():
            raise ValueError("category must be non-empty")
        return v


class TrendDataPoint(BaseModel):
    """A single data point in a spend or emissions trend series.

    Represents spend and emissions values for a single time period
    with delta and direction metadata for trend visualization.

    Attributes:
        period: Period label (e.g. '2024-Q1', '2024-01').
        period_start: Start date of the period.
        period_end: End date of the period.
        spend_usd: Total spend in USD for the period.
        emissions_kgco2e: Total emissions in kgCO2e for the period.
        intensity_kgco2e_per_usd: Emissions intensity for the period.
        delta_spend_pct: Percentage change in spend from previous period.
        delta_emissions_pct: Percentage change in emissions from previous.
        direction: Trend direction ('up', 'down', 'flat').
        record_count: Number of spend records in the period.
    """

    period: str = Field(
        ..., description="Period label (e.g. '2024-Q1', '2024-01')",
    )
    period_start: date = Field(
        ..., description="Start date of the period",
    )
    period_end: date = Field(
        ..., description="End date of the period",
    )
    spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Total spend in USD for the period",
    )
    emissions_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total emissions in kgCO2e for the period",
    )
    intensity_kgco2e_per_usd: float = Field(
        default=0.0, ge=0.0,
        description="Emissions intensity for the period",
    )
    delta_spend_pct: float = Field(
        default=0.0,
        description="Percentage change in spend from previous period",
    )
    delta_emissions_pct: float = Field(
        default=0.0,
        description="Percentage change in emissions from previous period",
    )
    direction: str = Field(
        default="flat",
        description="Trend direction ('up', 'down', 'flat')",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Number of spend records in the period",
    )

    model_config = {"extra": "forbid"}

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        """Validate period label is non-empty."""
        if not v or not v.strip():
            raise ValueError("period must be non-empty")
        return v

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Validate direction is one of the allowed values."""
        allowed = {"up", "down", "flat"}
        if v not in allowed:
            raise ValueError(f"direction must be one of {allowed}, got '{v}'")
        return v


class CategorizationReport(BaseModel):
    """A generated categorization report with analytics sections.

    Contains a complete report output including title, period,
    summary statistics, detailed sections, and format metadata
    for compliance and analytics reporting.

    Attributes:
        report_id: Unique identifier for this report.
        title: Report title.
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        report_format: Output format of the report.
        summary: High-level summary statistics.
        sections: Detailed report sections as key-value pairs.
        aggregates: Category aggregation results.
        hotspots: Hotspot analysis results.
        trends: Trend analysis data points.
        total_spend_usd: Total spend in USD covered by the report.
        total_emissions_kgco2e: Total emissions covered by the report.
        total_records: Total number of spend records in the report.
        generated_at: Timestamp when the report was generated.
        generated_by: User or system that generated the report.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    title: str = Field(
        ..., description="Report title",
    )
    period_start: date = Field(
        ..., description="Start date of the reporting period",
    )
    period_end: date = Field(
        ..., description="End date of the reporting period",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format of the report",
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level summary statistics",
    )
    sections: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed report sections as key-value pairs",
    )
    aggregates: List[SpendAggregate] = Field(
        default_factory=list,
        description="Category aggregation results",
    )
    hotspots: List[HotspotResult] = Field(
        default_factory=list,
        description="Hotspot analysis results",
    )
    trends: List[TrendDataPoint] = Field(
        default_factory=list,
        description="Trend analysis data points",
    )
    total_spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Total spend in USD covered by the report",
    )
    total_emissions_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total emissions covered by the report",
    )
    total_records: int = Field(
        default=0, ge=0,
        description="Total number of spend records in the report",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the report was generated",
    )
    generated_by: str = Field(
        default="system",
        description="User or system that generated the report",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title is non-empty."""
        if not v or not v.strip():
            raise ValueError("title must be non-empty")
        return v


class IngestionBatch(BaseModel):
    """Metadata for a batch of ingested spend records.

    Tracks the lifecycle, size, source, and error details of
    a single ingestion operation for monitoring and retry.

    Attributes:
        batch_id: Unique identifier for this ingestion batch.
        source: Ingestion source type.
        source_reference: File name, API endpoint, or ERP connection ID.
        record_count: Total number of records in the batch.
        records_accepted: Number of records accepted after validation.
        records_rejected: Number of records rejected during validation.
        status: Current batch processing status.
        errors: List of error messages encountered during ingestion.
        warnings: List of warning messages.
        started_at: Timestamp when ingestion started.
        completed_at: Timestamp when ingestion completed.
        duration_seconds: Total ingestion duration in seconds.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this ingestion batch",
    )
    source: IngestionSource = Field(
        ..., description="Ingestion source type",
    )
    source_reference: str = Field(
        default="",
        description="File name, API endpoint, or ERP connection ID",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Total number of records in the batch",
    )
    records_accepted: int = Field(
        default=0, ge=0,
        description="Number of records accepted after validation",
    )
    records_rejected: int = Field(
        default=0, ge=0,
        description="Number of records rejected during validation",
    )
    status: RecordStatus = Field(
        default=RecordStatus.RAW,
        description="Current batch processing status",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error messages encountered during ingestion",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of warning messages",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when ingestion started",
    )
    completed_at: Optional[datetime] = Field(
        None, description="Timestamp when ingestion completed",
    )
    duration_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Total ingestion duration in seconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}


class SpendCategorizerStatistics(BaseModel):
    """Aggregated statistics for the spend categorizer service.

    Provides high-level operational metrics for monitoring the
    overall health, coverage, and activity of the categorizer.

    Attributes:
        total_records_ingested: Total number of spend records ingested.
        total_records_classified: Number of records with taxonomy codes.
        total_records_mapped: Number of records mapped to Scope 3.
        total_records_calculated: Number of records with emissions values.
        total_spend_usd: Total spend amount in USD processed.
        total_emissions_kgco2e: Total estimated emissions in kgCO2e.
        avg_classification_confidence: Average classification confidence.
        classification_coverage_pct: Percentage of records classified.
        scope3_coverage_pct: Percentage of records mapped to Scope 3.
        emissions_coverage_pct: Percentage of records with emissions.
        total_vendors: Total number of unique vendors.
        total_rules: Total number of active classification rules.
        total_batches: Total number of ingestion batches processed.
        avg_batch_duration_seconds: Average batch processing duration.
    """

    total_records_ingested: int = Field(
        default=0, ge=0,
        description="Total number of spend records ingested",
    )
    total_records_classified: int = Field(
        default=0, ge=0,
        description="Number of records with taxonomy codes assigned",
    )
    total_records_mapped: int = Field(
        default=0, ge=0,
        description="Number of records mapped to Scope 3 categories",
    )
    total_records_calculated: int = Field(
        default=0, ge=0,
        description="Number of records with emissions values calculated",
    )
    total_spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Total spend amount in USD processed",
    )
    total_emissions_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Total estimated emissions in kgCO2e",
    )
    avg_classification_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average classification confidence score",
    )
    classification_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of records successfully classified",
    )
    scope3_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of records mapped to Scope 3 categories",
    )
    emissions_coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of records with emissions calculated",
    )
    total_vendors: int = Field(
        default=0, ge=0,
        description="Total number of unique vendors",
    )
    total_rules: int = Field(
        default=0, ge=0,
        description="Total number of active classification rules",
    )
    total_batches: int = Field(
        default=0, ge=0,
        description="Total number of ingestion batches processed",
    )
    avg_batch_duration_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Average batch processing duration in seconds",
    )

    model_config = {"extra": "forbid"}


class VendorProfile(BaseModel):
    """Profile of a vendor with categorization metadata.

    Aggregates vendor-level information including normalized name,
    known aliases, default category assignments, and total spend
    for vendor management and classification.

    Attributes:
        vendor_id: Unique vendor identifier.
        normalized_name: Standardized vendor name after normalization.
        aliases: Known alternative names or spellings for the vendor.
        erp_spend_category: High-level ERP spend category for the vendor.
        scope3_category: Default Scope 3 category assignment.
        taxonomy_code: Default taxonomy code for the vendor.
        taxonomy_system: Taxonomy system for the default code.
        total_spend_usd: Cumulative total spend in USD with this vendor.
        total_emissions_kgco2e: Cumulative total emissions for this vendor.
        record_count: Total number of spend records from this vendor.
        emission_factor_override: Vendor-specific emission factor override.
        emission_factor_source: Source of the vendor-specific factor.
        is_strategic: Whether this is a strategic or key vendor.
        first_seen: Date the vendor first appeared in spend data.
        last_seen: Date of the most recent spend record from this vendor.
        tenant_id: Tenant identifier for multi-tenant isolation.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    vendor_id: str = Field(
        ..., description="Unique vendor identifier",
    )
    normalized_name: str = Field(
        ..., description="Standardized vendor name after normalization",
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Known alternative names or spellings for the vendor",
    )
    erp_spend_category: Optional[ERPSpendCategory] = Field(
        None, description="High-level ERP spend category for the vendor",
    )
    scope3_category: Optional[Scope3Category] = Field(
        None, description="Default Scope 3 category assignment",
    )
    taxonomy_code: Optional[str] = Field(
        None, description="Default taxonomy code for the vendor",
    )
    taxonomy_system: Optional[TaxonomySystem] = Field(
        None, description="Taxonomy system for the default code",
    )
    total_spend_usd: float = Field(
        default=0.0, ge=0.0,
        description="Cumulative total spend in USD with this vendor",
    )
    total_emissions_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="Cumulative total emissions for this vendor",
    )
    record_count: int = Field(
        default=0, ge=0,
        description="Total number of spend records from this vendor",
    )
    emission_factor_override: Optional[float] = Field(
        None, ge=0.0,
        description="Vendor-specific emission factor override (kgCO2e/USD)",
    )
    emission_factor_source: Optional[EmissionFactorSource] = Field(
        None, description="Source of the vendor-specific emission factor",
    )
    is_strategic: bool = Field(
        default=False,
        description="Whether this is a strategic or key vendor",
    )
    first_seen: Optional[date] = Field(
        None, description="Date the vendor first appeared in spend data",
    )
    last_seen: Optional[date] = Field(
        None,
        description="Date of the most recent spend record from this vendor",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("vendor_id")
    @classmethod
    def validate_vendor_id(cls, v: str) -> str:
        """Validate vendor_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("vendor_id must be non-empty")
        return v

    @field_validator("normalized_name")
    @classmethod
    def validate_normalized_name(cls, v: str) -> str:
        """Validate normalized_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("normalized_name must be non-empty")
        return v


# =============================================================================
# Request Models (7)
# =============================================================================


class IngestSpendRequest(BaseModel):
    """Request body for ingesting a batch of spend records.

    Attributes:
        source: Ingestion source type.
        source_reference: File name, API endpoint, or connection ID.
        records: List of spend records to ingest.
        enable_deduplication: Whether to deduplicate on ingestion.
        enable_normalization: Whether to normalize vendor names.
        default_currency: Default currency if not specified per record.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    source: IngestionSource = Field(
        ..., description="Ingestion source type",
    )
    source_reference: str = Field(
        default="",
        description="File name, API endpoint, or connection ID",
    )
    records: List[SpendRecord] = Field(
        ..., min_length=1,
        description="List of spend records to ingest",
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Whether to deduplicate on ingestion",
    )
    enable_normalization: bool = Field(
        default=True,
        description="Whether to normalize vendor names",
    )
    default_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Default currency if not specified per record",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}


class ClassifyRequest(BaseModel):
    """Request body for classifying spend records with taxonomy codes.

    Attributes:
        record_ids: List of record IDs to classify (empty = all pending).
        taxonomy_system: Taxonomy system to use for classification.
        min_confidence: Minimum confidence threshold for classification.
        use_rules: Whether to apply user-defined classification rules.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    record_ids: List[str] = Field(
        default_factory=list,
        description="List of record IDs to classify (empty = all pending)",
    )
    taxonomy_system: TaxonomySystem = Field(
        default=TaxonomySystem.UNSPSC,
        description="Taxonomy system to use for classification",
    )
    min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum confidence threshold for classification",
    )
    use_rules: bool = Field(
        default=True,
        description="Whether to apply user-defined classification rules",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}


class MapScope3Request(BaseModel):
    """Request body for mapping classified records to Scope 3 categories.

    Attributes:
        record_ids: List of record IDs to map (empty = all classified).
        mapping_strategy: Strategy for Scope 3 mapping.
        use_naics_lookup: Whether to use NAICS code lookup tables.
        use_keyword_matching: Whether to use keyword-based matching.
        min_confidence: Minimum confidence threshold for mapping.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    record_ids: List[str] = Field(
        default_factory=list,
        description="List of record IDs to map (empty = all classified)",
    )
    mapping_strategy: str = Field(
        default="rule_based",
        description="Strategy for Scope 3 mapping",
    )
    use_naics_lookup: bool = Field(
        default=True,
        description="Whether to use NAICS code lookup tables",
    )
    use_keyword_matching: bool = Field(
        default=True,
        description="Whether to use keyword-based matching",
    )
    min_confidence: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum confidence threshold for mapping",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}


class CalculateEmissionsRequest(BaseModel):
    """Request body for calculating emissions from categorized spend.

    Attributes:
        record_ids: List of record IDs to calculate (empty = all mapped).
        calculation_method: Preferred emission calculation method.
        emission_factor_sources: Ordered list of factor sources to use.
        default_currency: Currency for spend amounts.
        include_data_quality: Whether to include data quality scores.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    record_ids: List[str] = Field(
        default_factory=list,
        description="List of record IDs to calculate (empty = all mapped)",
    )
    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.SPEND_BASED,
        description="Preferred emission calculation method",
    )
    emission_factor_sources: List[EmissionFactorSource] = Field(
        default_factory=lambda: [
            EmissionFactorSource.SUPPLIER_SPECIFIC,
            EmissionFactorSource.EPA_EEIO,
            EmissionFactorSource.EXIOBASE,
            EmissionFactorSource.DEFRA,
        ],
        description="Ordered list of factor sources to use (priority order)",
    )
    default_currency: CurrencyCode = Field(
        default=CurrencyCode.USD,
        description="Currency for spend amounts",
    )
    include_data_quality: bool = Field(
        default=True,
        description="Whether to include data quality scores in results",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}


class CreateRuleRequest(BaseModel):
    """Request body for creating a new classification rule.

    Attributes:
        name: Human-readable rule name.
        description: Detailed description of the rule purpose.
        match_type: Pattern matching strategy.
        match_field: Field to match against.
        pattern: Pattern string to match.
        target_taxonomy_system: Taxonomy system for the target code.
        target_taxonomy_code: Target taxonomy code to assign on match.
        target_scope3_category: Target Scope 3 category on match.
        priority: Rule evaluation priority.
        confidence_boost: Confidence score adjustment on match.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    name: str = Field(
        ..., description="Human-readable rule name",
    )
    description: str = Field(
        default="", description="Detailed description of the rule purpose",
    )
    match_type: RuleMatchType = Field(
        ..., description="Pattern matching strategy",
    )
    match_field: str = Field(
        default="description",
        description="Field to match against (description, vendor_name, etc.)",
    )
    pattern: str = Field(
        ..., description="Pattern string to match",
    )
    target_taxonomy_system: Optional[TaxonomySystem] = Field(
        None, description="Taxonomy system for the target code",
    )
    target_taxonomy_code: Optional[str] = Field(
        None, description="Target taxonomy code to assign on match",
    )
    target_scope3_category: Optional[Scope3Category] = Field(
        None, description="Target Scope 3 category to assign on match",
    )
    priority: RulePriority = Field(
        default=RulePriority.MEDIUM,
        description="Rule evaluation priority",
    )
    confidence_boost: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Confidence score adjustment on match",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate pattern is non-empty."""
        if not v or not v.strip():
            raise ValueError("pattern must be non-empty")
        return v


class AnalyticsRequest(BaseModel):
    """Request body for running spend and emissions analytics.

    Attributes:
        timeframe: Aggregation timeframe for analytics.
        start_date: Start date for the analytics period.
        end_date: End date for the analytics period.
        group_by: Field to group results by (category, vendor, scope3).
        include_hotspots: Whether to include hotspot analysis.
        include_trends: Whether to include trend analysis.
        hotspot_types: Hotspot types to include in analysis.
        top_n: Number of top items to return in rankings.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    timeframe: AnalyticsTimeframe = Field(
        default=AnalyticsTimeframe.QUARTERLY,
        description="Aggregation timeframe for analytics",
    )
    start_date: date = Field(
        ..., description="Start date for the analytics period",
    )
    end_date: date = Field(
        ..., description="End date for the analytics period",
    )
    group_by: str = Field(
        default="scope3_category",
        description="Field to group results by (category, vendor, scope3)",
    )
    include_hotspots: bool = Field(
        default=True,
        description="Whether to include hotspot analysis",
    )
    include_trends: bool = Field(
        default=True,
        description="Whether to include trend analysis",
    )
    hotspot_types: List[HotspotType] = Field(
        default_factory=lambda: [
            HotspotType.TOP_SPEND,
            HotspotType.TOP_EMISSIONS,
            HotspotType.TOP_INTENSITY,
        ],
        description="Hotspot types to include in analysis",
    )
    top_n: int = Field(
        default=10, ge=1, le=100,
        description="Number of top items to return in rankings",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}


class GenerateReportRequest(BaseModel):
    """Request body for generating a categorization report.

    Attributes:
        title: Report title.
        start_date: Start date for the reporting period.
        end_date: End date for the reporting period.
        report_format: Desired output format.
        include_aggregates: Whether to include category aggregations.
        include_hotspots: Whether to include hotspot analysis.
        include_trends: Whether to include trend analysis.
        scope3_categories: Scope 3 categories to include (empty = all).
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    title: str = Field(
        ..., description="Report title",
    )
    start_date: date = Field(
        ..., description="Start date for the reporting period",
    )
    end_date: date = Field(
        ..., description="End date for the reporting period",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Desired output format",
    )
    include_aggregates: bool = Field(
        default=True,
        description="Whether to include category aggregations",
    )
    include_hotspots: bool = Field(
        default=True,
        description="Whether to include hotspot analysis",
    )
    include_trends: bool = Field(
        default=True,
        description="Whether to include trend analysis",
    )
    scope3_categories: List[Scope3Category] = Field(
        default_factory=list,
        description="Scope 3 categories to include (empty = all)",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )

    model_config = {"extra": "forbid"}

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate title is non-empty."""
        if not v or not v.strip():
            raise ValueError("title must be non-empty")
        return v


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Re-exported Layer 1 enumerations (scope3_category_mapper)
    # -------------------------------------------------------------------------
    "Scope3Category",
    "DataSourceType",
    "CalculationApproach",
    # -------------------------------------------------------------------------
    # Re-exported Layer 1 models (scope3_category_mapper, aliased)
    # -------------------------------------------------------------------------
    "L1SpendRecord",
    "L1PurchaseOrder",
    "BOMItem",
    "CategoryMappingResult",
    # -------------------------------------------------------------------------
    # Re-exported Layer 1 constants (scope3_category_mapper)
    # -------------------------------------------------------------------------
    "NAICS_TO_CATEGORY",
    "SPEND_KEYWORDS_TO_CATEGORY",
    # -------------------------------------------------------------------------
    # Re-exported Layer 1 enumerations (erp_connector_agent)
    # -------------------------------------------------------------------------
    "ERPSystem",
    "ERPSpendCategory",
    "TransactionType",
    # -------------------------------------------------------------------------
    # Re-exported Layer 1 constants (erp_connector_agent)
    # -------------------------------------------------------------------------
    "SPEND_TO_SCOPE3_MAPPING",
    "DEFAULT_EMISSION_FACTORS",
    # -------------------------------------------------------------------------
    # Re-exported Layer 1 models (procurement_carbon_footprint)
    # -------------------------------------------------------------------------
    "CalculationMethod",
    "ProcurementItem",
    "EmissionCalculation",
    # -------------------------------------------------------------------------
    # New enumerations (12)
    # -------------------------------------------------------------------------
    "TaxonomySystem",
    "IngestionSource",
    "RecordStatus",
    "ClassificationConfidence",
    "EmissionFactorSource",
    "EmissionFactorUnit",
    "CurrencyCode",
    "AnalyticsTimeframe",
    "ReportFormat",
    "RuleMatchType",
    "RulePriority",
    "HotspotType",
    # -------------------------------------------------------------------------
    # SDK data models (15)
    # -------------------------------------------------------------------------
    "SpendRecord",
    "NormalizedSpendRecord",
    "TaxonomyCode",
    "TaxonomyClassification",
    "Scope3Assignment",
    "EmissionFactor",
    "EmissionCalculationResult",
    "CategoryRule",
    "SpendAggregate",
    "HotspotResult",
    "TrendDataPoint",
    "CategorizationReport",
    "IngestionBatch",
    "SpendCategorizerStatistics",
    "VendorProfile",
    # -------------------------------------------------------------------------
    # Request models (7)
    # -------------------------------------------------------------------------
    "IngestSpendRequest",
    "ClassifyRequest",
    "MapScope3Request",
    "CalculateEmissionsRequest",
    "CreateRuleRequest",
    "AnalyticsRequest",
    "GenerateReportRequest",
]
