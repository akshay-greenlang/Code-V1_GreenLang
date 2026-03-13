# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-018 Commodity Risk Analyzer

Pydantic v2 request/response models for the REST API layer covering all
8 engine domains: commodity profiling, derived product analysis, price
volatility, production forecasting, substitution risk, regulatory compliance,
due diligence workflows, and portfolio aggregation.

All numeric risk and financial fields use ``Decimal`` for precision.
All date/time fields use UTC-aware ``datetime``.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
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


class CommodityTypeEnum(str, Enum):
    """EUDR-regulated commodity types per EU 2023/1115 Article 1."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class RiskLevelEnum(str, Enum):
    """Risk classification levels for EUDR commodity assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketConditionEnum(str, Enum):
    """Market condition classifications for price volatility."""

    STABLE = "stable"
    VOLATILE = "volatile"
    DISRUPTED = "disrupted"
    CRISIS = "crisis"


class VolatilityLevelEnum(str, Enum):
    """Volatility classification for price analysis."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class ComplianceStatusEnum(str, Enum):
    """Regulatory compliance status values."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    PENDING = "pending"


class DDWorkflowStatusEnum(str, Enum):
    """Due diligence workflow status values."""

    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    EVIDENCE_REVIEW = "evidence_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvidenceTypeEnum(str, Enum):
    """Types of evidence for due diligence workflows."""

    CERTIFICATE = "certificate"
    SATELLITE_IMAGE = "satellite_image"
    GPS_COORDINATES = "gps_coordinates"
    SUPPLIER_DECLARATION = "supplier_declaration"
    AUDIT_REPORT = "audit_report"
    CUSTOMS_DOCUMENT = "customs_document"
    LAB_TEST = "lab_test"
    FIELD_INSPECTION = "field_inspection"
    TRADE_DOCUMENT = "trade_document"
    OTHER = "other"


class VolatilityTrendEnum(str, Enum):
    """Trend direction for volatility analysis."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class PenaltyCategoryEnum(str, Enum):
    """EUDR penalty categories per Article 25."""

    MINOR = "minor"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CRITICAL = "critical"


class DDTriggerEnum(str, Enum):
    """Triggers for initiating a due diligence workflow."""

    INITIAL_ASSESSMENT = "initial_assessment"
    PERIODIC_REVIEW = "periodic_review"
    RISK_ESCALATION = "risk_escalation"
    REGULATORY_REQUEST = "regulatory_request"
    SUBSTITUTION_ALERT = "substitution_alert"
    SUPPLIER_CHANGE = "supplier_change"
    MANUAL = "manual"


class SeveritySummaryEnum(str, Enum):
    """Severity classification for substitution alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Common / Shared Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response for the Commodity Risk Analyzer API."""

    status: str = Field(
        default="healthy",
        description="Service health status",
        examples=["healthy"],
    )
    agent_id: str = Field(
        default="GL-EUDR-CRA-018",
        description="Agent identifier",
    )
    agent_name: str = Field(
        default="EUDR Commodity Risk Analyzer",
        description="Human-readable agent name",
    )
    version: str = Field(
        default="1.0.0",
        description="API version",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Current server time in UTC",
    )

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(
        ...,
        description="Error type identifier",
        examples=["validation_error"],
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Invalid commodity type provided"],
    )
    detail: Optional[str] = Field(
        None,
        description="Additional error context or stack trace reference",
    )
    request_id: Optional[str] = Field(
        None,
        description="Request correlation ID for tracing",
    )

    model_config = ConfigDict(from_attributes=True)


class PaginationParams(BaseModel):
    """Standard pagination query parameters."""

    limit: int = Field(
        default=50, ge=1, le=1000,
        description="Maximum number of results to return",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Number of results to skip",
    )


class PaginatedMeta(BaseModel):
    """Pagination metadata included in list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")


class ProvenanceInfo(BaseModel):
    """Provenance tracking metadata for audit trails."""

    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash of input+output for data lineage",
        examples=["a1b2c3d4e5f6..."],
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when provenance was recorded",
    )
    source_agent: str = Field(
        default="GL-EUDR-CRA-018",
        description="Agent that produced this record",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        description="Processing duration in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class CountryData(BaseModel):
    """Country-level data for commodity analysis."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
        examples=["GH"],
    )
    share: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Share of total sourcing (0.0 to 1.0)",
        examples=[Decimal("0.45")],
    )
    risk_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Country-level risk score (0-100)",
    )
    deforestation_rate: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Annual deforestation rate (hectares/year)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class SupplyChainData(BaseModel):
    """Supply chain metadata for commodity profiling."""

    depth: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of tiers in the supply chain",
        examples=[4],
    )
    total_suppliers: int = Field(
        default=0,
        ge=0,
        description="Total number of known suppliers",
        examples=[25],
    )
    certified_suppliers: int = Field(
        default=0,
        ge=0,
        description="Number of certified suppliers",
        examples=[18],
    )
    traceability_coverage: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Fraction of supply chain with full traceability",
        examples=[Decimal("0.72")],
    )

    model_config = ConfigDict(from_attributes=True)


class ProfileOptions(BaseModel):
    """Options for commodity profiling requests."""

    include_price_data: bool = Field(
        default=True,
        description="Include current price and volatility data",
    )
    include_production_data: bool = Field(
        default=True,
        description="Include production volume and yield data",
    )
    include_regulatory_data: bool = Field(
        default=True,
        description="Include regulatory compliance status",
    )
    risk_weights: Optional[Dict[str, Decimal]] = Field(
        None,
        description="Custom risk dimension weights (must sum to 1.0)",
    )

    @field_validator("risk_weights")
    @classmethod
    def validate_risk_weights(
        cls, v: Optional[Dict[str, Decimal]]
    ) -> Optional[Dict[str, Decimal]]:
        """Validate risk weights sum to 1.0 if provided."""
        if v is None:
            return v
        weight_sum = sum(v.values())
        if abs(float(weight_sum) - 1.0) > 0.001:
            raise ValueError(
                f"risk_weights must sum to 1.0, got {weight_sum}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Commodity Profile Schemas
# =============================================================================


class CommodityProfileRequest(BaseModel):
    """Request to profile a single EUDR commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="EUDR-regulated commodity type",
        examples=["cocoa"],
    )
    country_data: List[CountryData] = Field(
        ...,
        min_length=1,
        description="Country distribution for this commodity",
    )
    supply_chain_data: Optional[SupplyChainData] = Field(
        None,
        description="Supply chain depth and coverage metadata",
    )
    options: Optional[ProfileOptions] = Field(
        None,
        description="Profiling options and custom weights",
    )

    @field_validator("country_data")
    @classmethod
    def validate_country_data(cls, v: List[CountryData]) -> List[CountryData]:
        """Validate that country shares sum to approximately 1.0."""
        total_share = sum(c.share for c in v)
        if abs(float(total_share) - 1.0) > 0.01:
            raise ValueError(
                f"Country shares must sum to 1.0, got {total_share}"
            )
        return v

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity_type": "cocoa",
                    "country_data": [
                        {"country_code": "GH", "share": 0.45},
                        {"country_code": "CI", "share": 0.55},
                    ],
                    "supply_chain_data": {
                        "depth": 4,
                        "total_suppliers": 25,
                        "certified_suppliers": 18,
                        "traceability_coverage": 0.72,
                    },
                }
            ]
        },
    )


class CommodityProfileBatchRequest(BaseModel):
    """Request to profile multiple EUDR commodities in a single batch."""

    commodities: List[CommodityProfileRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of commodity profiles to analyze (max 50)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodities": [
                        {
                            "commodity_type": "cocoa",
                            "country_data": [
                                {"country_code": "GH", "share": 1.0},
                            ],
                        },
                        {
                            "commodity_type": "coffee",
                            "country_data": [
                                {"country_code": "BR", "share": 0.6},
                                {"country_code": "CO", "share": 0.4},
                            ],
                        },
                    ]
                }
            ]
        },
    )


class CommodityProfileResponse(BaseModel):
    """Response containing a complete commodity risk profile."""

    profile_id: str = Field(
        default_factory=_new_id,
        description="Unique profile identifier",
    )
    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="EUDR-regulated commodity type",
    )
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Composite risk score (0-100)",
        examples=[Decimal("68.5")],
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Risk classification",
        examples=["high"],
    )
    deforestation_risk: RiskLevelEnum = Field(
        ...,
        description="Deforestation-specific risk level",
        examples=["high"],
    )
    supply_chain_complexity: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Supply chain complexity index (0-1)",
    )
    traceability_score: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="End-to-end traceability score (0-1)",
    )
    price_volatility_index: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Current price volatility index (0-1)",
    )
    production_volume: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Annual production volume in metric tonnes",
    )
    country_distribution: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Country code to sourcing share mapping",
    )
    processing_chains: List[str] = Field(
        default_factory=list,
        description="Known processing chain stages",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Audit trail provenance metadata",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Profile creation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class CommodityProfileBatchResponse(BaseModel):
    """Response for batch commodity profiling."""

    profiles: List[CommodityProfileResponse] = Field(
        default_factory=list,
        description="List of generated commodity profiles",
    )
    total_processed: int = Field(
        default=0,
        ge=0,
        description="Number of commodities successfully profiled",
    )
    total_failed: int = Field(
        default=0,
        ge=0,
        description="Number of commodities that failed profiling",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        description="Total batch processing time in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class CommodityRiskHistoryEntry(BaseModel):
    """Single point in a commodity risk history time series."""

    date: date = Field(..., description="Assessment date")
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Risk score on this date",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk level on this date")
    deforestation_risk: RiskLevelEnum = Field(
        ...,
        description="Deforestation risk on this date",
    )

    model_config = ConfigDict(from_attributes=True)


class CommodityRiskHistoryResponse(BaseModel):
    """Risk history time series for a commodity."""

    commodity_id: str = Field(..., description="Commodity identifier")
    commodity_type: CommodityTypeEnum = Field(..., description="Commodity type")
    history: List[CommodityRiskHistoryEntry] = Field(
        default_factory=list,
        description="Time series of risk assessments",
    )
    period_start: date = Field(..., description="Start of history period")
    period_end: date = Field(..., description="End of history period")
    trend: VolatilityTrendEnum = Field(
        ...,
        description="Overall risk trend direction",
    )

    model_config = ConfigDict(from_attributes=True)


class CommodityComparisonEntry(BaseModel):
    """Risk comparison data for a single commodity."""

    commodity_type: CommodityTypeEnum = Field(..., description="Commodity type")
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Composite risk score",
    )
    deforestation_risk: RiskLevelEnum = Field(..., description="Deforestation risk")
    supply_chain_complexity: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Complexity index",
    )
    traceability_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Traceability score",
    )
    price_volatility: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Price volatility index",
    )

    model_config = ConfigDict(from_attributes=True)


class CommodityComparisonResponse(BaseModel):
    """Response for comparing multiple commodities side by side."""

    commodities: List[CommodityComparisonEntry] = Field(
        ...,
        description="Comparison entries for each commodity",
    )
    comparison_matrix: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Pairwise risk differential matrix",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Risk mitigation recommendations based on comparison",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Comparison generation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class CommoditySummaryEntry(BaseModel):
    """Summary data for a single commodity in the overview."""

    commodity_type: CommodityTypeEnum = Field(..., description="Commodity type")
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Current risk score",
    )
    risk_level: RiskLevelEnum = Field(..., description="Current risk level")
    active_suppliers: int = Field(default=0, ge=0, description="Active supplier count")
    compliance_rate: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Compliance rate (0-1)",
    )

    model_config = ConfigDict(from_attributes=True)


class CommoditySummaryResponse(BaseModel):
    """Summary overview of all monitored commodities."""

    commodities: List[CommoditySummaryEntry] = Field(
        default_factory=list,
        description="Summary entries for each commodity",
    )
    total_commodities: int = Field(default=0, ge=0, description="Total tracked")
    average_risk_score: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Average risk across all commodities",
    )
    highest_risk_commodity: Optional[CommodityTypeEnum] = Field(
        None,
        description="Commodity with highest risk score",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Summary generation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Derived Product Schemas
# =============================================================================


class ProcessingStageItem(BaseModel):
    """A single stage in a derived product processing chain."""

    stage_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Processing stage name",
        examples=["fermentation"],
    )
    stage_order: int = Field(
        ...,
        ge=1,
        description="Order in the processing chain",
    )
    risk_contribution: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Risk contribution of this stage (0-1)",
    )
    location_country: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Country where this stage occurs",
    )

    model_config = ConfigDict(from_attributes=True)


class DerivedProductAnalyzeRequest(BaseModel):
    """Request to analyze a derived product for EUDR Annex I compliance."""

    product_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique product identifier",
        examples=["PROD-CHOC-001"],
    )
    source_commodity: CommodityTypeEnum = Field(
        ...,
        description="Source EUDR commodity type",
        examples=["cocoa"],
    )
    processing_stages: List[ProcessingStageItem] = Field(
        ...,
        min_length=1,
        description="Processing stages from raw commodity to final product",
    )
    product_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Human-readable product name",
        examples=["Dark Chocolate 70%"],
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "product_id": "PROD-CHOC-001",
                    "source_commodity": "cocoa",
                    "processing_stages": [
                        {"stage_name": "fermentation", "stage_order": 1},
                        {"stage_name": "drying", "stage_order": 2},
                        {"stage_name": "roasting", "stage_order": 3},
                        {"stage_name": "conching", "stage_order": 4},
                    ],
                    "product_name": "Dark Chocolate 70%",
                }
            ]
        },
    )


class DerivedProductTraceRequest(BaseModel):
    """Request to trace a derived product back to its raw commodity origin."""

    product_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Product identifier to trace",
    )

    model_config = ConfigDict(extra="forbid")


class DerivedProductResponse(BaseModel):
    """Response containing derived product analysis results."""

    product_id: str = Field(..., description="Product identifier")
    source_commodity: CommodityTypeEnum = Field(
        ...,
        description="Source EUDR commodity",
    )
    product_name: Optional[str] = Field(None, description="Product name")
    processing_chain: List[ProcessingStageItem] = Field(
        default_factory=list,
        description="Complete processing chain",
    )
    risk_multiplier: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0.0"),
        le=Decimal("10.0"),
        description="Risk multiplier applied due to processing complexity",
    )
    traceability_score: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Product-level traceability score (0-1)",
    )
    annex_i_reference: Optional[str] = Field(
        None,
        description="EUDR Annex I reference code for this product",
        examples=["1806.31", "1806.32"],
    )
    overall_risk_score: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Overall risk score incorporating processing chain",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Audit trail provenance metadata",
    )
    analyzed_at: datetime = Field(
        default_factory=_utcnow,
        description="Analysis timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class ProcessingChainResponse(BaseModel):
    """Response containing a complete processing chain analysis."""

    chain_id: str = Field(
        default_factory=_new_id,
        description="Unique chain identifier",
    )
    source_commodity: CommodityTypeEnum = Field(
        ...,
        description="Raw commodity at chain origin",
    )
    final_product: str = Field(
        ...,
        description="Final derived product name",
    )
    stages: List[ProcessingStageItem] = Field(
        default_factory=list,
        description="Ordered processing stages",
    )
    total_risk: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Cumulative risk score across chain",
    )
    transformation_ratio: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0.0"),
        description="Input-to-output mass ratio",
    )
    chain_length: int = Field(
        default=0,
        ge=0,
        description="Number of processing stages",
    )

    model_config = ConfigDict(from_attributes=True)


class AnnexIMappingEntry(BaseModel):
    """Mapping entry between EUDR Annex I product codes and commodity types."""

    hs_code: str = Field(
        ...,
        description="Harmonized System tariff code",
        examples=["1806.31"],
    )
    description: str = Field(
        ...,
        description="Product description from Annex I",
    )
    source_commodity: CommodityTypeEnum = Field(
        ...,
        description="Source EUDR commodity",
    )
    product_category: str = Field(
        ...,
        description="Derived product category",
        examples=["chocolate"],
    )

    model_config = ConfigDict(from_attributes=True)


class AnnexIMappingResponse(BaseModel):
    """Response containing EUDR Annex I product-to-commodity mappings."""

    mappings: List[AnnexIMappingEntry] = Field(
        default_factory=list,
        description="List of Annex I mapping entries",
    )
    total_entries: int = Field(default=0, ge=0, description="Total entries")
    commodity_filter: Optional[CommodityTypeEnum] = Field(
        None,
        description="Applied commodity filter",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Price Schemas
# =============================================================================


class PriceResponse(BaseModel):
    """Current price data for a commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    price: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Current price per metric tonne",
        examples=[Decimal("2450.00")],
    )
    currency: str = Field(
        default="USD",
        description="Price currency (ISO 4217)",
        examples=["USD"],
    )
    price_date: date = Field(
        ...,
        description="Price observation date",
    )
    volatility_30d: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="30-day rolling volatility index (0-1)",
    )
    volatility_90d: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="90-day rolling volatility index (0-1)",
    )
    market_condition: MarketConditionEnum = Field(
        default=MarketConditionEnum.STABLE,
        description="Current market condition classification",
    )
    exchange: Optional[str] = Field(
        None,
        description="Commodity exchange source",
        examples=["ICE", "CBOT"],
    )

    model_config = ConfigDict(from_attributes=True)


class PriceHistoryEntry(BaseModel):
    """Single price data point in a historical series."""

    price_date: date = Field(..., description="Observation date")
    price: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Price per metric tonne",
    )
    currency: str = Field(default="USD", description="Price currency")
    volume: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Trading volume on this date",
    )

    model_config = ConfigDict(from_attributes=True)


class PriceHistoryResponse(BaseModel):
    """Historical price data for a commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    prices: List[PriceHistoryEntry] = Field(
        default_factory=list,
        description="Historical price data points",
    )
    period_start: date = Field(..., description="Start of price history period")
    period_end: date = Field(..., description="End of price history period")
    currency: str = Field(default="USD", description="Price currency")
    data_points: int = Field(default=0, ge=0, description="Number of data points")

    model_config = ConfigDict(from_attributes=True)


class VolatilityResponse(BaseModel):
    """Volatility analysis results for a commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    volatility: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Volatility index (0-1)",
    )
    window_days: int = Field(
        ...,
        ge=1,
        description="Rolling window size in days",
    )
    trend: VolatilityTrendEnum = Field(
        ...,
        description="Volatility trend direction",
    )
    risk_level: VolatilityLevelEnum = Field(
        ...,
        description="Volatility risk classification",
    )
    percentile_rank: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Historical percentile rank of current volatility",
    )
    analyzed_at: datetime = Field(
        default_factory=_utcnow,
        description="Analysis timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class MarketDisruptionEntry(BaseModel):
    """Single market disruption event."""

    disruption_id: str = Field(
        default_factory=_new_id,
        description="Unique disruption identifier",
    )
    event_type: str = Field(
        ...,
        description="Type of disruption event",
        examples=["supply_shortage", "export_ban", "weather_event"],
    )
    description: str = Field(
        ...,
        description="Human-readable disruption description",
    )
    severity: SeveritySummaryEnum = Field(
        ...,
        description="Disruption severity",
    )
    start_date: date = Field(..., description="Disruption start date")
    end_date: Optional[date] = Field(None, description="Disruption end date")
    price_impact_pct: Optional[Decimal] = Field(
        None,
        description="Estimated price impact as percentage",
    )
    affected_countries: List[str] = Field(
        default_factory=list,
        description="ISO country codes affected",
    )

    model_config = ConfigDict(from_attributes=True)


class MarketDisruptionResponse(BaseModel):
    """Market disruption analysis for a commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    disruptions: List[MarketDisruptionEntry] = Field(
        default_factory=list,
        description="Active and recent disruption events",
    )
    severity: SeveritySummaryEnum = Field(
        default=SeveritySummaryEnum.LOW,
        description="Overall market severity",
    )
    total_disruptions: int = Field(
        default=0,
        ge=0,
        description="Total number of disruptions",
    )

    model_config = ConfigDict(from_attributes=True)


class PriceForecastRequest(BaseModel):
    """Request for commodity price forecasting."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type to forecast",
        examples=["cocoa"],
    )
    horizon_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Forecast horizon in days",
        examples=[90],
    )
    include_confidence: bool = Field(
        default=True,
        description="Include confidence intervals in forecast",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity_type": "cocoa",
                    "horizon_days": 90,
                    "include_confidence": True,
                }
            ]
        },
    )


class ForecastPoint(BaseModel):
    """Single forecast data point."""

    forecast_date: date = Field(..., description="Forecast target date")
    price: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Forecasted price per metric tonne",
    )
    lower_bound: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Lower confidence bound",
    )
    upper_bound: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Upper confidence bound",
    )
    confidence: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Prediction confidence (0-1)",
    )

    model_config = ConfigDict(from_attributes=True)


class PriceForecastResponse(BaseModel):
    """Price forecast results for a commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    forecast: List[ForecastPoint] = Field(
        default_factory=list,
        description="Forecasted price data points",
    )
    confidence_intervals: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Confidence interval metadata (e.g. {'90pct_lower': ..., '90pct_upper': ...})",
    )
    horizon_days: int = Field(..., ge=1, description="Forecast horizon in days")
    currency: str = Field(default="USD", description="Forecast currency")
    model_version: str = Field(
        default="1.0.0",
        description="Forecasting model version",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Forecast generation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Production Schemas
# =============================================================================


class ProductionForecastRequest(BaseModel):
    """Request for production volume forecasting."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type to forecast",
        examples=["cocoa"],
    )
    region: Optional[str] = Field(
        None,
        max_length=100,
        description="Region or country code to scope the forecast",
        examples=["GH"],
    )
    horizon_months: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Forecast horizon in months",
        examples=[12],
    )
    include_climate_impact: bool = Field(
        default=True,
        description="Include climate impact adjustment factors",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity_type": "cocoa",
                    "region": "GH",
                    "horizon_months": 12,
                    "include_climate_impact": True,
                }
            ]
        },
    )


class ProductionForecastEntry(BaseModel):
    """Single month production forecast data point."""

    month: str = Field(
        ...,
        description="Forecast month (YYYY-MM)",
        examples=["2026-04"],
    )
    production_volume: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Forecasted production volume in metric tonnes",
    )
    confidence_lower: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Lower confidence bound",
    )
    confidence_upper: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Upper confidence bound",
    )

    model_config = ConfigDict(from_attributes=True)


class ClimateImpactData(BaseModel):
    """Climate impact data affecting production forecasts."""

    impact_factor: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0.0"),
        le=Decimal("2.0"),
        description="Climate impact adjustment factor (1.0 = no impact)",
    )
    drought_risk: RiskLevelEnum = Field(
        default=RiskLevelEnum.LOW,
        description="Drought risk level",
    )
    flood_risk: RiskLevelEnum = Field(
        default=RiskLevelEnum.LOW,
        description="Flood risk level",
    )
    temperature_anomaly: Optional[Decimal] = Field(
        None,
        description="Temperature anomaly in degrees Celsius",
    )
    rainfall_anomaly: Optional[Decimal] = Field(
        None,
        description="Rainfall anomaly as percentage deviation from average",
    )

    model_config = ConfigDict(from_attributes=True)


class SeasonalFactorEntry(BaseModel):
    """Monthly seasonal coefficient for production analysis."""

    month: int = Field(..., ge=1, le=12, description="Month number (1-12)")
    month_name: str = Field(..., description="Month name", examples=["January"])
    seasonal_coefficient: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("2.0"),
        description="Seasonal coefficient (1.0 = average)",
    )
    is_peak: bool = Field(
        default=False,
        description="Whether this is a peak production month",
    )

    model_config = ConfigDict(from_attributes=True)


class ProductionForecastResponse(BaseModel):
    """Production forecast results for a commodity and region."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    region: Optional[str] = Field(None, description="Region scope")
    forecasts: List[ProductionForecastEntry] = Field(
        default_factory=list,
        description="Monthly production forecasts",
    )
    climate_impact: Optional[ClimateImpactData] = Field(
        None,
        description="Climate impact analysis data",
    )
    seasonal_factors: List[SeasonalFactorEntry] = Field(
        default_factory=list,
        description="Monthly seasonal coefficients",
    )
    total_forecast_volume: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        description="Sum of forecasted production over the horizon",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Forecast generation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class YieldResponse(BaseModel):
    """Yield estimation for a commodity in a specific country and year."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    year: int = Field(
        ...,
        ge=2000,
        le=2050,
        description="Assessment year",
    )
    yield_estimate: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Yield estimate in metric tonnes per hectare",
    )
    confidence_interval: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Confidence interval (e.g. {'lower': 1.2, 'upper': 1.8})",
    )
    historical_average: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Historical average yield for comparison",
    )

    model_config = ConfigDict(from_attributes=True)


class SeasonalPatternResponse(BaseModel):
    """Seasonal production patterns for a commodity in a region."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    region: str = Field(..., description="Region or country code")
    monthly_patterns: List[SeasonalFactorEntry] = Field(
        default_factory=list,
        description="Monthly seasonal coefficients",
    )
    peak_months: List[int] = Field(
        default_factory=list,
        description="Peak production month numbers",
    )
    low_months: List[int] = Field(
        default_factory=list,
        description="Low production month numbers",
    )

    model_config = ConfigDict(from_attributes=True)


class ProductionSummaryEntry(BaseModel):
    """Production summary for a single commodity."""

    commodity_type: CommodityTypeEnum = Field(..., description="Commodity type")
    total_production: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        description="Total annual production in metric tonnes",
    )
    top_producing_countries: List[str] = Field(
        default_factory=list,
        description="Top producing country codes",
    )
    year_over_year_change: Optional[Decimal] = Field(
        None,
        description="Year-over-year production change as percentage",
    )

    model_config = ConfigDict(from_attributes=True)


class ProductionSummaryResponse(BaseModel):
    """Production summary across all tracked commodities."""

    commodities: List[ProductionSummaryEntry] = Field(
        default_factory=list,
        description="Production summaries per commodity",
    )
    total_commodities: int = Field(default=0, ge=0)
    generated_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Substitution Schemas
# =============================================================================


class CommodityHistoryEntry(BaseModel):
    """Historical commodity declaration for substitution detection."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Declared commodity type",
    )
    declaration_date: date = Field(
        ...,
        description="Date of declaration",
    )
    quantity: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Declared quantity in metric tonnes",
    )
    origin_country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Declared country of origin",
    )

    model_config = ConfigDict(from_attributes=True)


class CurrentDeclaration(BaseModel):
    """Current commodity declaration for substitution verification."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Currently declared commodity type",
    )
    quantity: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Declared quantity in metric tonnes",
    )
    origin_country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Declared country of origin",
    )
    declaration_date: date = Field(
        ...,
        description="Date of current declaration",
    )

    model_config = ConfigDict(from_attributes=True)


class SubstitutionDetectRequest(BaseModel):
    """Request to detect commodity substitution for a supplier."""

    supplier_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Supplier identifier",
        examples=["SUP-GH-001"],
    )
    commodity_history: List[CommodityHistoryEntry] = Field(
        ...,
        min_length=1,
        description="Historical commodity declarations from this supplier",
    )
    current_declaration: CurrentDeclaration = Field(
        ...,
        description="Current commodity declaration to verify",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "supplier_id": "SUP-GH-001",
                    "commodity_history": [
                        {
                            "commodity_type": "cocoa",
                            "declaration_date": "2025-06-01",
                            "quantity": 500.0,
                            "origin_country": "GH",
                        },
                        {
                            "commodity_type": "cocoa",
                            "declaration_date": "2025-09-01",
                            "quantity": 480.0,
                            "origin_country": "GH",
                        },
                    ],
                    "current_declaration": {
                        "commodity_type": "cocoa",
                        "quantity": 520.0,
                        "origin_country": "CI",
                        "declaration_date": "2026-01-15",
                    },
                }
            ]
        },
    )


class SubstitutionVerifyRequest(BaseModel):
    """Request to verify a specific commodity declaration against evidence."""

    declaration: CurrentDeclaration = Field(
        ...,
        description="Declaration to verify",
    )
    supporting_evidence: List[str] = Field(
        ...,
        min_length=1,
        description="Evidence document IDs supporting the declaration",
    )
    supplier_id: Optional[str] = Field(
        None,
        description="Supplier identifier for context",
    )

    model_config = ConfigDict(extra="forbid")


class DetectedSwitch(BaseModel):
    """A detected commodity substitution event."""

    switch_id: str = Field(
        default_factory=_new_id,
        description="Unique switch event identifier",
    )
    from_commodity: CommodityTypeEnum = Field(
        ...,
        description="Original declared commodity",
    )
    to_commodity: CommodityTypeEnum = Field(
        ...,
        description="Substituted commodity",
    )
    from_country: str = Field(..., description="Original origin country")
    to_country: str = Field(..., description="New origin country")
    detection_confidence: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Detection confidence score (0-1)",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Detection timestamp",
    )
    risk_impact: RiskLevelEnum = Field(
        ...,
        description="Risk impact of this substitution",
    )

    model_config = ConfigDict(from_attributes=True)


class SubstitutionResponse(BaseModel):
    """Response from substitution risk detection."""

    supplier_id: str = Field(..., description="Supplier identifier")
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Substitution risk score (0-100)",
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Overall substitution risk level",
    )
    detected_switches: List[DetectedSwitch] = Field(
        default_factory=list,
        description="Detected substitution events",
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Alert messages for detected substitutions",
    )
    total_declarations_analyzed: int = Field(
        default=0,
        ge=0,
        description="Total declarations analyzed",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Audit trail provenance metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class SubstitutionAlertEntry(BaseModel):
    """Single substitution alert."""

    alert_id: str = Field(
        default_factory=_new_id,
        description="Unique alert identifier",
    )
    supplier_id: str = Field(..., description="Supplier identifier")
    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Affected commodity type",
    )
    severity: SeveritySummaryEnum = Field(
        ...,
        description="Alert severity",
    )
    message: str = Field(..., description="Alert message")
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Alert creation timestamp",
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether the alert has been acknowledged",
    )

    model_config = ConfigDict(from_attributes=True)


class SubstitutionAlertResponse(BaseModel):
    """Response listing active substitution alerts."""

    alerts: List[SubstitutionAlertEntry] = Field(
        default_factory=list,
        description="Active substitution alerts",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of alerts",
    )
    severity_summary: Dict[str, int] = Field(
        default_factory=lambda: {
            "low": 0, "medium": 0, "high": 0, "critical": 0
        },
        description="Alert count by severity",
    )

    model_config = ConfigDict(from_attributes=True)


class SubstitutionHistoryEntry(BaseModel):
    """Historical substitution event for a supplier."""

    event_date: date = Field(..., description="Event date")
    from_commodity: CommodityTypeEnum = Field(..., description="Original commodity")
    to_commodity: CommodityTypeEnum = Field(..., description="Substituted commodity")
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Risk score at time of event",
    )
    resolution: Optional[str] = Field(
        None,
        description="Resolution status or notes",
    )

    model_config = ConfigDict(from_attributes=True)


class SubstitutionHistoryResponse(BaseModel):
    """Substitution switching history for a supplier."""

    supplier_id: str = Field(..., description="Supplier identifier")
    history: List[SubstitutionHistoryEntry] = Field(
        default_factory=list,
        description="Historical substitution events",
    )
    total_switches: int = Field(
        default=0,
        ge=0,
        description="Total number of detected switches",
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


class SubstitutionPatternEntry(BaseModel):
    """Detected substitution pattern across suppliers."""

    pattern_id: str = Field(
        default_factory=_new_id,
        description="Unique pattern identifier",
    )
    pattern_type: str = Field(
        ...,
        description="Type of pattern (e.g. 'origin_switch', 'commodity_swap')",
    )
    frequency: int = Field(
        default=0,
        ge=0,
        description="Number of occurrences",
    )
    affected_suppliers: int = Field(
        default=0,
        ge=0,
        description="Number of suppliers exhibiting this pattern",
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Pattern risk level",
    )
    description: str = Field(..., description="Pattern description")

    model_config = ConfigDict(from_attributes=True)


class SubstitutionPatternResponse(BaseModel):
    """Response containing detected substitution patterns."""

    patterns: List[SubstitutionPatternEntry] = Field(
        default_factory=list,
        description="Detected substitution patterns",
    )
    total_patterns: int = Field(default=0, ge=0, description="Total patterns found")
    analysis_period_start: date = Field(..., description="Analysis start date")
    analysis_period_end: date = Field(..., description="Analysis end date")

    model_config = ConfigDict(from_attributes=True)


class SubstitutionVerifyResponse(BaseModel):
    """Response from declaration verification against evidence."""

    verified: bool = Field(
        ...,
        description="Whether the declaration is verified",
    )
    confidence: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Verification confidence score",
    )
    discrepancies: List[str] = Field(
        default_factory=list,
        description="Identified discrepancies",
    )
    evidence_reviewed: int = Field(
        default=0,
        ge=0,
        description="Number of evidence documents reviewed",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Regulatory Schemas
# =============================================================================


class ComplianceCheckRequest(BaseModel):
    """Request to check regulatory compliance for a commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="EUDR commodity type to check",
        examples=["cocoa"],
    )
    supplier_data: Dict[str, Any] = Field(
        ...,
        description="Supplier metadata including certifications and documentation",
    )
    documentation: List[str] = Field(
        default_factory=list,
        description="Document IDs available for compliance verification",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity_type": "cocoa",
                    "supplier_data": {
                        "supplier_id": "SUP-GH-001",
                        "country": "GH",
                        "certifications": ["RA-2024-GH-001"],
                    },
                    "documentation": ["DOC-001", "DOC-002", "DOC-003"],
                }
            ]
        },
    )


class ComplianceGap(BaseModel):
    """A single compliance gap identified during assessment."""

    gap_id: str = Field(
        default_factory=_new_id,
        description="Unique gap identifier",
    )
    article_reference: str = Field(
        ...,
        description="EUDR article reference",
        examples=["Article 4(2)(f)"],
    )
    requirement: str = Field(
        ...,
        description="Compliance requirement description",
    )
    severity: SeveritySummaryEnum = Field(
        ...,
        description="Gap severity",
    )
    status: str = Field(
        default="open",
        description="Gap resolution status",
    )

    model_config = ConfigDict(from_attributes=True)


class RemediationStep(BaseModel):
    """A recommended remediation step for a compliance gap."""

    step_number: int = Field(..., ge=1, description="Step order")
    action: str = Field(..., description="Remediation action description")
    priority: SeveritySummaryEnum = Field(..., description="Priority level")
    estimated_effort: Optional[str] = Field(
        None,
        description="Estimated effort (e.g. '2-4 hours')",
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceCheckResponse(BaseModel):
    """Response from a regulatory compliance check."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Assessed commodity type",
    )
    compliance_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Overall compliance score (0-100)",
    )
    compliance_status: ComplianceStatusEnum = Field(
        ...,
        description="Overall compliance status",
    )
    gaps: List[ComplianceGap] = Field(
        default_factory=list,
        description="Identified compliance gaps",
    )
    remediation_steps: List[RemediationStep] = Field(
        default_factory=list,
        description="Recommended remediation actions",
    )
    articles_assessed: List[str] = Field(
        default_factory=list,
        description="EUDR articles assessed",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Audit trail provenance metadata",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class ArticleRequirement(BaseModel):
    """EUDR article requirement for a specific commodity."""

    article_id: str = Field(
        ...,
        description="EUDR article identifier",
        examples=["Art. 4(2)(f)"],
    )
    article_title: str = Field(
        ...,
        description="Article title",
    )
    requirement_text: str = Field(
        ...,
        description="Specific requirement text",
    )
    mandatory: bool = Field(
        default=True,
        description="Whether this requirement is mandatory",
    )
    documentation_types: List[str] = Field(
        default_factory=list,
        description="Types of documentation that satisfy this requirement",
    )

    model_config = ConfigDict(from_attributes=True)


class RegulatoryRequirementsResponse(BaseModel):
    """EUDR regulatory requirements for a specific commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    articles: List[ArticleRequirement] = Field(
        default_factory=list,
        description="Applicable EUDR article requirements",
    )
    documentation_needed: List[str] = Field(
        default_factory=list,
        description="Complete list of required documentation types",
    )
    total_requirements: int = Field(
        default=0,
        ge=0,
        description="Total number of requirements",
    )

    model_config = ConfigDict(from_attributes=True)


class PenaltyRiskFactor(BaseModel):
    """A factor contributing to penalty risk."""

    factor_name: str = Field(..., description="Risk factor name")
    factor_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Factor risk score",
    )
    description: str = Field(..., description="Factor description")

    model_config = ConfigDict(from_attributes=True)


class PenaltyRiskResponse(BaseModel):
    """Penalty risk assessment per EUDR Article 25."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Assessed commodity type",
    )
    penalty_category: PenaltyCategoryEnum = Field(
        ...,
        description="Penalty severity category per Article 25",
    )
    estimated_fine_range: Dict[str, Decimal] = Field(
        ...,
        description="Estimated fine range (e.g. {'min': 10000, 'max': 500000})",
    )
    risk_factors: List[PenaltyRiskFactor] = Field(
        default_factory=list,
        description="Contributing risk factors",
    )
    overall_penalty_risk: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Overall penalty risk score (0-100)",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class RegulatoryUpdateEntry(BaseModel):
    """A single regulatory update or change."""

    update_id: str = Field(
        default_factory=_new_id,
        description="Unique update identifier",
    )
    title: str = Field(..., description="Update title")
    summary: str = Field(..., description="Update summary")
    effective_date: date = Field(..., description="Date update takes effect")
    impact_level: SeveritySummaryEnum = Field(
        ...,
        description="Impact level of this update",
    )
    affected_commodities: List[CommodityTypeEnum] = Field(
        default_factory=list,
        description="Commodities affected by this update",
    )

    model_config = ConfigDict(from_attributes=True)


class RegulatoryUpdatesResponse(BaseModel):
    """Response listing recent regulatory updates."""

    updates: List[RegulatoryUpdateEntry] = Field(
        default_factory=list,
        description="Recent regulatory updates",
    )
    total_updates: int = Field(default=0, ge=0, description="Total updates")

    model_config = ConfigDict(from_attributes=True)


class DocumentationRequirementEntry(BaseModel):
    """Documentation requirement for EUDR compliance."""

    document_type: str = Field(
        ...,
        description="Type of document required",
        examples=["geolocation_data", "supplier_declaration"],
    )
    description: str = Field(
        ...,
        description="Requirement description",
    )
    mandatory: bool = Field(
        default=True,
        description="Whether this document is mandatory",
    )
    applicable_articles: List[str] = Field(
        default_factory=list,
        description="EUDR articles requiring this document",
    )
    accepted_formats: List[str] = Field(
        default_factory=list,
        description="Accepted file formats",
    )

    model_config = ConfigDict(from_attributes=True)


class DocumentationRequirementsResponse(BaseModel):
    """Response listing documentation requirements for EUDR compliance."""

    commodity_type: Optional[CommodityTypeEnum] = Field(
        None,
        description="Commodity filter applied",
    )
    requirements: List[DocumentationRequirementEntry] = Field(
        default_factory=list,
        description="Documentation requirements",
    )
    total_requirements: int = Field(
        default=0,
        ge=0,
        description="Total documentation requirements",
    )
    mandatory_count: int = Field(
        default=0,
        ge=0,
        description="Number of mandatory documents",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Due Diligence Schemas
# =============================================================================


class DDInitiateRequest(BaseModel):
    """Request to initiate a commodity-specific due diligence workflow."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="EUDR commodity type for this workflow",
        examples=["cocoa"],
    )
    supplier_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Supplier identifier",
        examples=["SUP-GH-001"],
    )
    trigger: DDTriggerEnum = Field(
        default=DDTriggerEnum.INITIAL_ASSESSMENT,
        description="Trigger reason for initiating the workflow",
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional notes for the workflow",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity_type": "cocoa",
                    "supplier_id": "SUP-GH-001",
                    "trigger": "initial_assessment",
                    "notes": "Annual due diligence review for Q1 2026",
                }
            ]
        },
    )


class DDEvidenceSubmitRequest(BaseModel):
    """Request to submit evidence to a due diligence workflow."""

    evidence_type: EvidenceTypeEnum = Field(
        ...,
        description="Type of evidence being submitted",
        examples=["certificate"],
    )
    evidence_data: Dict[str, Any] = Field(
        ...,
        description="Evidence payload (varies by evidence type)",
    )
    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Notes about this evidence submission",
    )
    document_ids: List[str] = Field(
        default_factory=list,
        description="Associated document identifiers",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "evidence_type": "certificate",
                    "evidence_data": {
                        "certificate_id": "RA-2026-GH-001",
                        "issuer": "Rainforest Alliance",
                        "valid_until": "2027-03-01",
                    },
                    "notes": "Annual certification renewal",
                    "document_ids": ["DOC-CERT-001"],
                }
            ]
        },
    )


class DDEvidenceItem(BaseModel):
    """An evidence item within a due diligence workflow."""

    evidence_id: str = Field(
        default_factory=_new_id,
        description="Unique evidence identifier",
    )
    evidence_type: EvidenceTypeEnum = Field(
        ...,
        description="Type of evidence",
    )
    status: str = Field(
        default="pending_review",
        description="Evidence review status",
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow,
        description="Submission timestamp",
    )
    reviewed_at: Optional[datetime] = Field(
        None,
        description="Review completion timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class DDNextStep(BaseModel):
    """A next step in the due diligence workflow."""

    step_id: str = Field(
        default_factory=_new_id,
        description="Step identifier",
    )
    step_name: str = Field(..., description="Step name")
    description: str = Field(..., description="Step description")
    required: bool = Field(default=True, description="Whether step is required")
    completed: bool = Field(default=False, description="Whether step is completed")

    model_config = ConfigDict(from_attributes=True)


class DDWorkflowResponse(BaseModel):
    """Response containing due diligence workflow status and details."""

    workflow_id: str = Field(
        default_factory=_new_id,
        description="Unique workflow identifier",
    )
    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type for this workflow",
    )
    supplier_id: str = Field(..., description="Supplier identifier")
    status: DDWorkflowStatusEnum = Field(
        ...,
        description="Current workflow status",
    )
    completion_pct: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Workflow completion percentage (0-100)",
    )
    evidence_items: List[DDEvidenceItem] = Field(
        default_factory=list,
        description="Evidence items collected in this workflow",
    )
    next_steps: List[DDNextStep] = Field(
        default_factory=list,
        description="Remaining workflow steps",
    )
    trigger: DDTriggerEnum = Field(
        ...,
        description="Trigger that initiated this workflow",
    )
    initiated_at: datetime = Field(
        default_factory=_utcnow,
        description="Workflow initiation timestamp",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Workflow completion timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class DDPendingWorkflowEntry(BaseModel):
    """Summary of a pending due diligence workflow."""

    workflow_id: str = Field(..., description="Workflow identifier")
    commodity_type: CommodityTypeEnum = Field(..., description="Commodity type")
    supplier_id: str = Field(..., description="Supplier identifier")
    status: DDWorkflowStatusEnum = Field(..., description="Current status")
    completion_pct: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Completion percentage",
    )
    initiated_at: datetime = Field(..., description="Initiation timestamp")

    model_config = ConfigDict(from_attributes=True)


class DDPendingResponse(BaseModel):
    """Response listing pending due diligence workflows."""

    workflows: List[DDPendingWorkflowEntry] = Field(
        default_factory=list,
        description="Pending due diligence workflows",
    )
    total_count: int = Field(
        default=0,
        ge=0,
        description="Total number of pending workflows",
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


class DDCompleteResponse(BaseModel):
    """Response after completing a due diligence workflow."""

    workflow_id: str = Field(..., description="Completed workflow identifier")
    status: DDWorkflowStatusEnum = Field(
        default=DDWorkflowStatusEnum.COMPLETED,
        description="Final status",
    )
    completion_pct: Decimal = Field(
        default=Decimal("100.0"),
        description="Final completion percentage",
    )
    total_evidence_items: int = Field(
        default=0,
        ge=0,
        description="Total evidence items collected",
    )
    completed_at: datetime = Field(
        default_factory=_utcnow,
        description="Completion timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Portfolio Schemas
# =============================================================================


class CommodityPosition(BaseModel):
    """A single commodity position within a portfolio."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="EUDR commodity type",
    )
    volume: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Volume in metric tonnes",
        examples=[Decimal("5000.0")],
    )
    value: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Value in USD",
    )
    primary_origin: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Primary sourcing country code",
    )
    risk_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Pre-computed risk score for this position",
    )

    model_config = ConfigDict(from_attributes=True)


class PortfolioAnalyzeRequest(BaseModel):
    """Request to analyze a multi-commodity portfolio."""

    commodity_positions: List[CommodityPosition] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Commodity positions in the portfolio",
    )
    portfolio_name: Optional[str] = Field(
        None,
        max_length=500,
        description="Human-readable portfolio name",
        examples=["Q1 2026 Sourcing Portfolio"],
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity_positions": [
                        {
                            "commodity_type": "cocoa",
                            "volume": 5000.0,
                            "value": 12250000.0,
                            "primary_origin": "GH",
                        },
                        {
                            "commodity_type": "coffee",
                            "volume": 3000.0,
                            "value": 7500000.0,
                            "primary_origin": "BR",
                        },
                        {
                            "commodity_type": "oil_palm",
                            "volume": 8000.0,
                            "value": 6400000.0,
                            "primary_origin": "ID",
                        },
                    ],
                    "portfolio_name": "Q1 2026 Sourcing Portfolio",
                }
            ]
        },
    )


class CommodityBreakdownEntry(BaseModel):
    """Portfolio breakdown for a single commodity."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    volume: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Volume in metric tonnes",
    )
    share_pct: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Share of total portfolio volume (%)",
    )
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Risk score for this commodity",
    )
    risk_contribution: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Contribution to total portfolio risk (%)",
    )

    model_config = ConfigDict(from_attributes=True)


class PortfolioResponse(BaseModel):
    """Response from portfolio risk analysis."""

    portfolio_id: str = Field(
        default_factory=_new_id,
        description="Unique portfolio analysis identifier",
    )
    portfolio_name: Optional[str] = Field(None, description="Portfolio name")
    concentration_index: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Herfindahl-Hirschman Index (0-1, higher = more concentrated)",
    )
    diversification_score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Portfolio diversification score (0-1, higher = more diversified)",
    )
    total_risk_exposure: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Total portfolio risk exposure (0-100)",
    )
    commodity_breakdown: List[CommodityBreakdownEntry] = Field(
        default_factory=list,
        description="Risk breakdown by commodity",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Portfolio optimization recommendations",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Audit trail provenance metadata",
    )
    analyzed_at: datetime = Field(
        default_factory=_utcnow,
        description="Analysis timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


class CommodityShareEntry(BaseModel):
    """Commodity share within concentration analysis."""

    commodity_type: CommodityTypeEnum = Field(
        ...,
        description="Commodity type",
    )
    share_pct: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Share of total portfolio (%)",
    )
    volume: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Volume in metric tonnes",
    )

    model_config = ConfigDict(from_attributes=True)


class ConcentrationResponse(BaseModel):
    """Portfolio concentration analysis using Herfindahl-Hirschman Index."""

    hhi_index: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="HHI value (0-1, higher = more concentrated)",
    )
    classification: str = Field(
        ...,
        description="Concentration classification (diversified, moderate, concentrated, highly_concentrated)",
        examples=["moderate"],
    )
    commodity_shares: List[CommodityShareEntry] = Field(
        default_factory=list,
        description="Commodity volume shares",
    )
    top_commodity: Optional[CommodityTypeEnum] = Field(
        None,
        description="Most concentrated commodity",
    )

    model_config = ConfigDict(from_attributes=True)


class DiversificationSuggestion(BaseModel):
    """Suggestion for improving portfolio diversification."""

    suggestion_id: str = Field(
        default_factory=_new_id,
        description="Unique suggestion identifier",
    )
    action: str = Field(..., description="Recommended action")
    impact: str = Field(
        ...,
        description="Expected impact description",
    )
    priority: SeveritySummaryEnum = Field(
        ...,
        description="Suggestion priority",
    )

    model_config = ConfigDict(from_attributes=True)


class DiversificationResponse(BaseModel):
    """Portfolio diversification analysis results."""

    score: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Diversification score (0-1)",
    )
    current_state: str = Field(
        ...,
        description="Current diversification state description",
        examples=["Moderately diversified across 3 commodities"],
    )
    improvement_suggestions: List[DiversificationSuggestion] = Field(
        default_factory=list,
        description="Suggestions for improving diversification",
    )
    commodity_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct commodities in portfolio",
    )

    model_config = ConfigDict(from_attributes=True)


class PortfolioSummaryResponse(BaseModel):
    """High-level portfolio summary."""

    portfolio_name: Optional[str] = Field(None, description="Portfolio name")
    total_volume: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        description="Total portfolio volume in metric tonnes",
    )
    total_value: Optional[Decimal] = Field(
        None,
        ge=Decimal("0.0"),
        description="Total portfolio value in USD",
    )
    commodity_count: int = Field(
        default=0,
        ge=0,
        description="Number of distinct commodities",
    )
    average_risk_score: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("100.0"),
        description="Volume-weighted average risk score",
    )
    hhi_index: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Concentration index",
    )
    diversification_score: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        le=Decimal("1.0"),
        description="Diversification score",
    )
    highest_risk_commodity: Optional[CommodityTypeEnum] = Field(
        None,
        description="Commodity with highest risk score",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Summary generation timestamp",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "CommodityTypeEnum",
    "RiskLevelEnum",
    "MarketConditionEnum",
    "VolatilityLevelEnum",
    "ComplianceStatusEnum",
    "DDWorkflowStatusEnum",
    "EvidenceTypeEnum",
    "VolatilityTrendEnum",
    "PenaltyCategoryEnum",
    "DDTriggerEnum",
    "SeveritySummaryEnum",
    # Common
    "HealthResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedMeta",
    "ProvenanceInfo",
    "CountryData",
    "SupplyChainData",
    "ProfileOptions",
    # Commodity Profile
    "CommodityProfileRequest",
    "CommodityProfileBatchRequest",
    "CommodityProfileResponse",
    "CommodityProfileBatchResponse",
    "CommodityRiskHistoryEntry",
    "CommodityRiskHistoryResponse",
    "CommodityComparisonEntry",
    "CommodityComparisonResponse",
    "CommoditySummaryEntry",
    "CommoditySummaryResponse",
    # Derived Product
    "ProcessingStageItem",
    "DerivedProductAnalyzeRequest",
    "DerivedProductTraceRequest",
    "DerivedProductResponse",
    "ProcessingChainResponse",
    "AnnexIMappingEntry",
    "AnnexIMappingResponse",
    # Price
    "PriceResponse",
    "PriceHistoryEntry",
    "PriceHistoryResponse",
    "VolatilityResponse",
    "MarketDisruptionEntry",
    "MarketDisruptionResponse",
    "PriceForecastRequest",
    "ForecastPoint",
    "PriceForecastResponse",
    # Production
    "ProductionForecastRequest",
    "ProductionForecastEntry",
    "ClimateImpactData",
    "SeasonalFactorEntry",
    "ProductionForecastResponse",
    "YieldResponse",
    "SeasonalPatternResponse",
    "ProductionSummaryEntry",
    "ProductionSummaryResponse",
    # Substitution
    "CommodityHistoryEntry",
    "CurrentDeclaration",
    "SubstitutionDetectRequest",
    "SubstitutionVerifyRequest",
    "DetectedSwitch",
    "SubstitutionResponse",
    "SubstitutionAlertEntry",
    "SubstitutionAlertResponse",
    "SubstitutionHistoryEntry",
    "SubstitutionHistoryResponse",
    "SubstitutionPatternEntry",
    "SubstitutionPatternResponse",
    "SubstitutionVerifyResponse",
    # Regulatory
    "ComplianceCheckRequest",
    "ComplianceGap",
    "RemediationStep",
    "ComplianceCheckResponse",
    "ArticleRequirement",
    "RegulatoryRequirementsResponse",
    "PenaltyRiskFactor",
    "PenaltyRiskResponse",
    "RegulatoryUpdateEntry",
    "RegulatoryUpdatesResponse",
    "DocumentationRequirementEntry",
    "DocumentationRequirementsResponse",
    # Due Diligence
    "DDInitiateRequest",
    "DDEvidenceSubmitRequest",
    "DDEvidenceItem",
    "DDNextStep",
    "DDWorkflowResponse",
    "DDPendingWorkflowEntry",
    "DDPendingResponse",
    "DDCompleteResponse",
    # Portfolio
    "CommodityPosition",
    "PortfolioAnalyzeRequest",
    "CommodityBreakdownEntry",
    "PortfolioResponse",
    "CommodityShareEntry",
    "ConcentrationResponse",
    "DiversificationSuggestion",
    "DiversificationResponse",
    "PortfolioSummaryResponse",
]
