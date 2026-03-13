# -*- coding: utf-8 -*-
"""
Commodity Risk Analyzer Data Models - AGENT-EUDR-018

Pydantic v2 data models for the Commodity Risk Analyzer Agent covering
all 7 EUDR-regulated commodities (cattle, cocoa, coffee, oil palm,
rubber, soya, wood) and their Annex I derived products with commodity
risk profiling, derived product traceability, price volatility monitoring,
production forecasting, substitution risk detection, regulatory compliance
mapping, commodity-specific due diligence workflows, and portfolio risk
aggregation.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all commodity risk analysis operations per
EU 2023/1115 Articles 1, 2, 3, 4, 8, 9, 10, and Annex I.

Enumerations (12):
    - CommodityType, DerivedProductCategory, ProcessingStage,
      RiskLevel, MarketCondition, VolatilityLevel, SeasonalPhase,
      ComplianceStatus, DDWorkflowStatus, EvidenceType,
      PortfolioStrategy, ReportFormat

Core Models (10):
    - CommodityProfile, DerivedProduct, PriceData, ProductionForecast,
      SubstitutionEvent, RegulatoryRequirement, DDWorkflow,
      PortfolioAnalysis, CommodityRiskScore, AuditLogEntry

Request Models (12):
    - ProfileCommodityRequest, AnalyzeDerivedProductRequest,
      QueryPriceVolatilityRequest, GenerateForecastRequest,
      DetectSubstitutionRequest, CheckComplianceRequest,
      InitiateDDWorkflowRequest, AggregatePortfolioRequest,
      BatchCommodityAnalysisRequest, CompareCommoditiesRequest,
      GetTrendRequest, HealthRequest

Response Models (12):
    - CommodityProfileResponse, DerivedProductResponse,
      PriceVolatilityResponse, ProductionForecastResponse,
      SubstitutionRiskResponse, RegulatoryComplianceResponse,
      DDWorkflowResponse, PortfolioAnalysisResponse,
      BatchAnalysisResponse, ComparisonResponse, TrendResponse,
      HealthResponse

Compatibility:
    Imports EUDRCommodity from greenlang.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-001 Supply Chain Mapping Master.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018 Commodity Risk Analyzer (GL-EUDR-CRA-018)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Cross-agent commodity import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.eudr_traceability.models import (
        EUDRCommodity as _ExternalEUDRCommodity,
    )
except ImportError:
    _ExternalEUDRCommodity = None  # type: ignore[assignment,misc]


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

#: Supported derived product categories (Annex I).
SUPPORTED_DERIVED_CATEGORIES: List[str] = [
    "beef_fresh", "beef_frozen", "beef_processed", "leather_raw",
    "leather_finished", "tallow", "gelatin", "bone_meal",
    "cocoa_beans", "cocoa_paste", "cocoa_butter", "cocoa_powder",
    "chocolate", "chocolate_confectionery",
    "coffee_beans_green", "coffee_roasted", "coffee_extracts",
    "coffee_preparations",
    "palm_oil_crude", "palm_oil_refined", "palm_kernel_oil",
    "biodiesel", "margarine", "glycerol", "oleochemicals",
    "natural_rubber_raw", "natural_rubber_smoked", "latex",
    "tires", "rubber_goods", "rubber_footwear",
    "soybeans", "soy_oil", "soy_meal", "soy_flour",
    "soy_protein", "animal_feed",
    "logs", "sawnwood", "plywood", "particle_board", "fibreboard",
    "veneer", "charcoal", "paper_pulp", "paper", "printed_matter",
    "furniture", "cork",
]

#: Supported report output formats.
SUPPORTED_OUTPUT_FORMATS: List[str] = [
    "pdf", "json", "html", "excel", "csv",
]

#: Default commodity risk scoring weights.
DEFAULT_COMMODITY_WEIGHTS: Dict[str, float] = {
    "deforestation_risk": 0.25,
    "supply_chain_complexity": 0.20,
    "price_volatility": 0.15,
    "regulatory_pressure": 0.15,
    "geographic_concentration": 0.15,
    "production_stability": 0.10,
}


# =============================================================================
# Enumerations
# =============================================================================


class CommodityType(str, Enum):
    """EUDR-regulated commodities per Article 1 and Annex I.

    Seven commodity groups subject to the EUDR deforestation-free
    requirement. Each commodity has specific HS code mappings,
    derived product definitions, and production characteristics
    that affect commodity risk analysis.
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class DerivedProductCategory(str, Enum):
    """Derived product categories per EUDR Annex I.

    Product categories derived from the 7 base EUDR commodities
    that are subject to due diligence requirements. Each category
    maps to specific HS/CN codes and has defined processing chains.
    """

    # Cattle-derived
    BEEF_FRESH = "beef_fresh"
    BEEF_FROZEN = "beef_frozen"
    BEEF_PROCESSED = "beef_processed"
    LEATHER_RAW = "leather_raw"
    LEATHER_FINISHED = "leather_finished"
    TALLOW = "tallow"
    GELATIN = "gelatin"
    BONE_MEAL = "bone_meal"
    # Cocoa-derived
    COCOA_BEANS = "cocoa_beans"
    COCOA_PASTE = "cocoa_paste"
    COCOA_BUTTER = "cocoa_butter"
    COCOA_POWDER = "cocoa_powder"
    CHOCOLATE = "chocolate"
    CHOCOLATE_CONFECTIONERY = "chocolate_confectionery"
    # Coffee-derived
    COFFEE_BEANS_GREEN = "coffee_beans_green"
    COFFEE_ROASTED = "coffee_roasted"
    COFFEE_EXTRACTS = "coffee_extracts"
    COFFEE_PREPARATIONS = "coffee_preparations"
    # Oil palm-derived
    PALM_OIL_CRUDE = "palm_oil_crude"
    PALM_OIL_REFINED = "palm_oil_refined"
    PALM_KERNEL_OIL = "palm_kernel_oil"
    BIODIESEL = "biodiesel"
    MARGARINE = "margarine"
    GLYCEROL = "glycerol"
    OLEOCHEMICALS = "oleochemicals"
    # Rubber-derived
    NATURAL_RUBBER_RAW = "natural_rubber_raw"
    NATURAL_RUBBER_SMOKED = "natural_rubber_smoked"
    LATEX = "latex"
    TIRES = "tires"
    RUBBER_GOODS = "rubber_goods"
    RUBBER_FOOTWEAR = "rubber_footwear"
    # Soya-derived
    SOYBEANS = "soybeans"
    SOY_OIL = "soy_oil"
    SOY_MEAL = "soy_meal"
    SOY_FLOUR = "soy_flour"
    SOY_PROTEIN = "soy_protein"
    ANIMAL_FEED = "animal_feed"
    # Wood-derived
    LOGS = "logs"
    SAWNWOOD = "sawnwood"
    PLYWOOD = "plywood"
    PARTICLE_BOARD = "particle_board"
    FIBREBOARD = "fibreboard"
    VENEER = "veneer"
    CHARCOAL = "charcoal"
    PAPER_PULP = "paper_pulp"
    PAPER = "paper"
    PRINTED_MATTER = "printed_matter"
    FURNITURE = "furniture"
    CORK = "cork"


class ProcessingStage(str, Enum):
    """Processing stages in the commodity transformation chain.

    RAW: Raw commodity as harvested or extracted.
    PRIMARY: Primary processing (cleaning, sorting, grading).
    SECONDARY: Secondary processing (milling, pressing, refining).
    TERTIARY: Tertiary processing (mixing, formulating, assembling).
    FINISHED: Finished product ready for market.
    PACKAGED: Packaged product ready for distribution.
    """

    RAW = "raw"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    FINISHED = "finished"
    PACKAGED = "packaged"


class RiskLevel(str, Enum):
    """Commodity risk classification per composite risk scoring.

    LOW: Risk score 0-25. Low risk, reduced monitoring frequency.
    MEDIUM: Risk score 26-50. Standard risk, standard monitoring.
    HIGH: Risk score 51-75. High risk, enhanced monitoring.
    CRITICAL: Risk score 76-100. Critical risk, immediate action required.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketCondition(str, Enum):
    """Market condition classification for commodity price analysis.

    STABLE: Market conditions are stable with normal price movements.
    VOLATILE: Market shows above-normal price fluctuations.
    DISRUPTED: Market is experiencing supply or demand disruption.
    CRISIS: Market is in crisis with extreme price movements.
    """

    STABLE = "stable"
    VOLATILE = "volatile"
    DISRUPTED = "disrupted"
    CRISIS = "crisis"


class VolatilityLevel(str, Enum):
    """Price volatility classification.

    LOW: Volatility index below low threshold.
    MODERATE: Volatility index between low and moderate thresholds.
    HIGH: Volatility index between moderate and high thresholds.
    EXTREME: Volatility index above extreme threshold.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class SeasonalPhase(str, Enum):
    """Seasonal production phase for commodity forecasting.

    PLANTING: Planting/seeding season.
    GROWING: Growing/maturation season.
    HARVEST: Harvest season.
    OFF_SEASON: Off-season (dormant period).
    """

    PLANTING = "planting"
    GROWING = "growing"
    HARVEST = "harvest"
    OFF_SEASON = "off_season"


class ComplianceStatus(str, Enum):
    """Regulatory compliance status per commodity per EUDR article.

    COMPLIANT: Fully compliant with all requirements.
    PARTIALLY_COMPLIANT: Some requirements met, gaps identified.
    NON_COMPLIANT: Not compliant, corrective action required.
    UNDER_REVIEW: Compliance status under review.
    NOT_ASSESSED: Compliance not yet assessed.
    """

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_ASSESSED = "not_assessed"


class DDWorkflowStatus(str, Enum):
    """Due diligence workflow execution status.

    NOT_STARTED: Workflow has not been initiated.
    IN_PROGRESS: Workflow is actively being executed.
    PENDING_REVIEW: Workflow completed, pending review.
    APPROVED: Workflow reviewed and approved.
    REJECTED: Workflow reviewed and rejected.
    OVERDUE: Workflow has exceeded deadline.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    OVERDUE = "overdue"


class EvidenceType(str, Enum):
    """Evidence types for commodity due diligence.

    COMPLIANCE_DECLARATION: EUDR compliance declaration.
    PRODUCT_DESCRIPTION: Product description with HS/CN code.
    QUANTITY_DECLARATION: Quantity declaration (mass, volume, units).
    GEOLOCATION_DATA: Geolocation coordinates of production plots.
    HARVEST_DATE: Harvest or production date declaration.
    DDS_REFERENCE: Due Diligence Statement reference number.
    SATELLITE_IMAGERY: Satellite imagery verification.
    FIELD_VERIFICATION: On-site field verification report.
    THIRD_PARTY_AUDIT: Third-party audit report.
    CERTIFICATION_DOCUMENT: Certification scheme document.
    """

    COMPLIANCE_DECLARATION = "compliance_declaration"
    PRODUCT_DESCRIPTION = "product_description"
    QUANTITY_DECLARATION = "quantity_declaration"
    GEOLOCATION_DATA = "geolocation_data"
    HARVEST_DATE = "harvest_date"
    DDS_REFERENCE = "dds_reference"
    SATELLITE_IMAGERY = "satellite_imagery"
    FIELD_VERIFICATION = "field_verification"
    THIRD_PARTY_AUDIT = "third_party_audit"
    CERTIFICATION_DOCUMENT = "certification_document"


class PortfolioStrategy(str, Enum):
    """Portfolio risk management strategy classification.

    CONSERVATIVE: Minimize risk, prioritize low-risk commodities.
    BALANCED: Balance risk and exposure across commodities.
    DIVERSIFIED: Maximize diversification across commodities and regions.
    CONCENTRATED: Accept concentration risk for strategic commodities.
    """

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    DIVERSIFIED = "diversified"
    CONCENTRATED = "concentrated"


class ReportFormat(str, Enum):
    """Report output format.

    JSON: JSON format for programmatic access.
    HTML: HTML format for web display.
    PDF: PDF format for printing and archiving.
    EXCEL: Excel format for data analysis.
    CSV: CSV format for data export.
    """

    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"


# =============================================================================
# Core Models
# =============================================================================


class CommodityRiskScore(BaseModel):
    """Individual commodity risk factor score with weighting.

    Attributes:
        factor_name: Risk factor name (e.g., deforestation_risk).
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
    last_updated: datetime = Field(default_factory=_utcnow)


class CommodityProfile(BaseModel):
    """Commodity risk profile with comprehensive risk scoring.

    Attributes:
        profile_id: Unique profile identifier.
        commodity_type: EUDR commodity type.
        risk_score: Composite risk score (0-100).
        risk_level: Risk level classification.
        supply_chain_depth: Number of tiers in supply chain (1-10).
        deforestation_risk: Deforestation risk classification.
        price_volatility_index: Price volatility index (0.0-1.0).
        production_volume: Production volume in metric tonnes.
        country_distribution: Country code to share mapping (sum=1.0).
        processing_chains: List of processing chain identifiers.
        factor_scores: Individual risk factor scores.
        provenance_hash: SHA-256 provenance hash.
        created_at: Profile creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    profile_id: str = Field(
        default_factory=lambda: f"cpf-{uuid.uuid4().hex[:12]}"
    )
    commodity_type: CommodityType
    risk_score: Decimal = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    supply_chain_depth: int = Field(..., ge=1, le=10)
    deforestation_risk: RiskLevel
    price_volatility_index: Decimal = Field(..., ge=0, le=1)
    production_volume: Decimal = Field(..., ge=0)
    country_distribution: Dict[str, Decimal] = Field(default_factory=dict)
    processing_chains: List[str] = Field(default_factory=list)
    factor_scores: List[CommodityRiskScore] = Field(default_factory=list)
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("country_distribution")
    @classmethod
    def validate_country_distribution(
        cls, v: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """Validate country distribution sums to approximately 1.0."""
        if v:
            total = sum(v.values())
            if not (Decimal("0.99") <= total <= Decimal("1.01")):
                raise ValueError(
                    f"Country distribution must sum to ~1.0, got {total}"
                )
        return v


class DerivedProduct(BaseModel):
    """Derived product from an EUDR commodity with traceability scoring.

    Attributes:
        product_id: Unique product identifier.
        source_commodity: Source EUDR commodity type.
        category: Derived product category (Annex I).
        processing_stages: Ordered list of processing stages.
        transformation_ratio: Input-to-output transformation ratio.
        risk_multiplier: Risk multiplier from processing complexity.
        traceability_score: Traceability score (0.0-1.0).
        hs_codes: Associated HS/CN codes.
        provenance_hash: SHA-256 provenance hash.
        created_at: Product record creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    product_id: str = Field(
        default_factory=lambda: f"dpd-{uuid.uuid4().hex[:12]}"
    )
    source_commodity: CommodityType
    category: DerivedProductCategory
    processing_stages: List[ProcessingStage] = Field(default_factory=list)
    transformation_ratio: Decimal = Field(..., gt=0, le=100)
    risk_multiplier: Decimal = Field(..., ge=0, le=10)
    traceability_score: Decimal = Field(..., ge=0, le=1)
    hs_codes: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class PriceData(BaseModel):
    """Commodity price data point with volatility metrics.

    Attributes:
        price_id: Unique price record identifier.
        commodity: Commodity type.
        price_date: Price observation date.
        price: Price value in specified currency.
        currency: ISO 4217 currency code.
        exchange: Commodity exchange identifier.
        volatility_30d: 30-day rolling volatility (0.0-1.0).
        volatility_90d: 90-day rolling volatility (0.0-1.0).
        volatility_level: Volatility classification.
        market_condition: Market condition classification.
        volume: Trading volume (optional).
        provenance_hash: SHA-256 provenance hash.
        created_at: Record creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    price_id: str = Field(
        default_factory=lambda: f"prc-{uuid.uuid4().hex[:12]}"
    )
    commodity: CommodityType
    price_date: date
    price: Decimal = Field(..., gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    exchange: str = Field(default="", max_length=100)
    volatility_30d: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    volatility_90d: Decimal = Field(default=Decimal("0"), ge=0, le=1)
    volatility_level: VolatilityLevel = VolatilityLevel.LOW
    market_condition: MarketCondition = MarketCondition.STABLE
    volume: Optional[Decimal] = Field(None, ge=0)
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ProductionForecast(BaseModel):
    """Production forecast for a commodity in a specific region.

    Attributes:
        forecast_id: Unique forecast identifier.
        commodity: Commodity type.
        region: ISO 3166-1 alpha-2 country or region code.
        forecast_date: Target forecast date.
        yield_estimate: Estimated yield in metric tonnes.
        confidence_lower: Lower bound of confidence interval.
        confidence_upper: Upper bound of confidence interval.
        confidence_level: Confidence level (e.g., 0.95 for 95%).
        climate_adjustment: Climate adjustment factor applied.
        seasonal_factor: Seasonal coefficient applied.
        seasonal_phase: Current seasonal phase.
        provenance_hash: SHA-256 provenance hash.
        created_at: Forecast creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    forecast_id: str = Field(
        default_factory=lambda: f"fcst-{uuid.uuid4().hex[:12]}"
    )
    commodity: CommodityType
    region: str = Field(..., min_length=2, max_length=10)
    forecast_date: date
    yield_estimate: Decimal = Field(..., ge=0)
    confidence_lower: Decimal = Field(..., ge=0)
    confidence_upper: Decimal = Field(..., ge=0)
    confidence_level: Decimal = Field(default=Decimal("0.95"), gt=0, le=1)
    climate_adjustment: Decimal = Field(default=Decimal("1.0"), gt=0, le=2)
    seasonal_factor: Decimal = Field(default=Decimal("1.0"), gt=0, le=3)
    seasonal_phase: SeasonalPhase = SeasonalPhase.GROWING
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def validate_confidence_bounds(self) -> "ProductionForecast":
        """Validate confidence interval bounds are consistent."""
        if self.confidence_lower > self.yield_estimate:
            raise ValueError(
                f"confidence_lower ({self.confidence_lower}) cannot exceed "
                f"yield_estimate ({self.yield_estimate})"
            )
        if self.confidence_upper < self.yield_estimate:
            raise ValueError(
                f"confidence_upper ({self.confidence_upper}) cannot be less "
                f"than yield_estimate ({self.yield_estimate})"
            )
        return self


class SubstitutionEvent(BaseModel):
    """Commodity substitution event detection record.

    Attributes:
        event_id: Unique event identifier.
        supplier_id: Supplier identifier where substitution detected.
        from_commodity: Original commodity type.
        to_commodity: Substituted commodity type.
        detection_date: Date substitution was detected.
        confidence: Detection confidence score (0.0-1.0).
        risk_impact: Quantified risk impact (0-100).
        evidence_items: Supporting evidence for detection.
        provenance_hash: SHA-256 provenance hash.
        created_at: Record creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    event_id: str = Field(
        default_factory=lambda: f"sub-{uuid.uuid4().hex[:12]}"
    )
    supplier_id: str = Field(..., min_length=1, max_length=100)
    from_commodity: CommodityType
    to_commodity: CommodityType
    detection_date: date
    confidence: Decimal = Field(..., ge=0, le=1)
    risk_impact: Decimal = Field(..., ge=0, le=100)
    evidence_items: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("to_commodity")
    @classmethod
    def validate_different_commodity(
        cls, v: CommodityType, info: Any
    ) -> CommodityType:
        """Validate substitution is between different commodities."""
        from_val = info.data.get("from_commodity")
        if from_val is not None and v == from_val:
            raise ValueError(
                "to_commodity must differ from from_commodity"
            )
        return v


class RegulatoryRequirement(BaseModel):
    """Regulatory requirement per commodity per EUDR article.

    Attributes:
        requirement_id: Unique requirement identifier.
        commodity: Applicable commodity type.
        eudr_article: EUDR article reference (e.g., "art4", "art8").
        requirement_type: Type of requirement (documentation, verification, etc.).
        description: Requirement description text.
        documentation_needed: List of required document types.
        evidence_standard: Required evidence standard (e.g., "photo", "gps").
        compliance_status: Current compliance status.
        provenance_hash: SHA-256 provenance hash.
        created_at: Record creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    requirement_id: str = Field(
        default_factory=lambda: f"req-{uuid.uuid4().hex[:12]}"
    )
    commodity: CommodityType
    eudr_article: str = Field(..., min_length=1, max_length=20)
    requirement_type: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=2000)
    documentation_needed: List[str] = Field(default_factory=list)
    evidence_standard: str = Field(default="standard", max_length=100)
    compliance_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class DDWorkflow(BaseModel):
    """Commodity-specific due diligence workflow.

    Attributes:
        workflow_id: Unique workflow identifier.
        commodity: Commodity type for this workflow.
        supplier_id: Supplier identifier (optional, for supplier-specific DD).
        status: Workflow execution status.
        dd_level: Due diligence level (simplified, standard, enhanced).
        evidence_items: Collected evidence items with types and statuses.
        verification_steps: Ordered verification steps with completion.
        completion_percentage: Overall completion percentage (0.0-1.0).
        deadline: Workflow completion deadline.
        started_at: Workflow start timestamp.
        completed_at: Workflow completion timestamp (if completed).
        provenance_hash: SHA-256 provenance hash.
        created_at: Record creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    workflow_id: str = Field(
        default_factory=lambda: f"ddw-{uuid.uuid4().hex[:12]}"
    )
    commodity: CommodityType
    supplier_id: Optional[str] = Field(None, max_length=100)
    status: DDWorkflowStatus
    dd_level: str = Field(default="standard", max_length=20)
    evidence_items: List[Dict[str, Any]] = Field(default_factory=list)
    verification_steps: List[Dict[str, Any]] = Field(default_factory=list)
    completion_percentage: Decimal = Field(
        default=Decimal("0"), ge=0, le=1
    )
    deadline: Optional[datetime] = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class PortfolioAnalysis(BaseModel):
    """Multi-commodity portfolio risk aggregation analysis.

    Attributes:
        analysis_id: Unique analysis identifier.
        commodities: List of commodity profiles in the portfolio.
        concentration_index: HHI concentration index (0.0-1.0).
        diversification_score: Diversification score (0.0-1.0).
        total_risk_exposure: Total portfolio risk exposure (0-100).
        strategy: Portfolio risk management strategy.
        commodity_shares: Per-commodity portfolio share mapping.
        risk_breakdown: Per-commodity risk contribution breakdown.
        correlation_matrix: Cross-commodity correlation matrix.
        provenance_hash: SHA-256 provenance hash.
        created_at: Analysis creation timestamp.
        updated_at: Last update timestamp.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    analysis_id: str = Field(
        default_factory=lambda: f"pfa-{uuid.uuid4().hex[:12]}"
    )
    commodities: List[CommodityType] = Field(default_factory=list)
    concentration_index: Decimal = Field(..., ge=0, le=1)
    diversification_score: Decimal = Field(..., ge=0, le=1)
    total_risk_exposure: Decimal = Field(..., ge=0, le=100)
    strategy: PortfolioStrategy = PortfolioStrategy.BALANCED
    commodity_shares: Dict[str, Decimal] = Field(default_factory=dict)
    risk_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict
    )
    provenance_hash: str = Field(default="", max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class AuditLogEntry(BaseModel):
    """Audit log entry for commodity risk analysis operations.

    Attributes:
        log_id: Unique log entry identifier.
        entity_type: Entity type (e.g., commodity_profile, derived_product).
        entity_id: Entity identifier.
        action: Action performed (e.g., profile, analyze, forecast).
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
    timestamp: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(..., min_length=64, max_length=64)
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Request Models
# =============================================================================


class ProfileCommodityRequest(BaseModel):
    """Request to create or update a commodity risk profile."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodity_type: CommodityType
    region: Optional[str] = Field(None, min_length=2, max_length=10)
    include_derived_products: bool = True
    include_price_data: bool = True
    requested_by: str = Field(..., min_length=1, max_length=100)


class AnalyzeDerivedProductRequest(BaseModel):
    """Request to analyze a derived product."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    source_commodity: CommodityType
    category: DerivedProductCategory
    processing_stages: Optional[List[ProcessingStage]] = None
    include_traceability: bool = True


class QueryPriceVolatilityRequest(BaseModel):
    """Request to query price volatility data."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodity: CommodityType
    start_date: date
    end_date: date
    currency: str = Field(default="USD", min_length=3, max_length=3)
    include_forecast: bool = False

    @model_validator(mode="after")
    def validate_date_range(self) -> "QueryPriceVolatilityRequest":
        """Validate that start_date is before end_date."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class GenerateForecastRequest(BaseModel):
    """Request to generate a production forecast."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodity: CommodityType
    region: str = Field(..., min_length=2, max_length=10)
    horizon_months: int = Field(default=12, ge=1, le=24)
    include_climate_adjustment: bool = True
    include_seasonal_factor: bool = True


class DetectSubstitutionRequest(BaseModel):
    """Request to detect commodity substitution events."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    supplier_id: str = Field(..., min_length=1, max_length=100)
    commodities: Optional[List[CommodityType]] = None
    lookback_days: int = Field(default=180, ge=1, le=365)
    min_confidence: Decimal = Field(default=Decimal("0.70"), ge=0, le=1)


class CheckComplianceRequest(BaseModel):
    """Request to check regulatory compliance for a commodity."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodity: CommodityType
    eudr_articles: Optional[List[str]] = None
    supplier_id: Optional[str] = Field(None, max_length=100)


class InitiateDDWorkflowRequest(BaseModel):
    """Request to initiate a commodity due diligence workflow."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodity: CommodityType
    supplier_id: Optional[str] = Field(None, max_length=100)
    dd_level: str = Field(default="standard", max_length=20)
    deadline_days: int = Field(default=30, ge=1, le=365)
    initiated_by: str = Field(..., min_length=1, max_length=100)


class AggregatePortfolioRequest(BaseModel):
    """Request to aggregate portfolio risk across commodities."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodities: List[CommodityType] = Field(..., min_length=1)
    strategy: PortfolioStrategy = PortfolioStrategy.BALANCED
    include_correlation: bool = True
    include_diversification: bool = True


class BatchCommodityAnalysisRequest(BaseModel):
    """Request to perform batch commodity analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodities: List[CommodityType] = Field(..., min_length=1, max_length=7)
    analysis_types: List[str] = Field(default_factory=lambda: ["profile"])
    requested_by: str = Field(..., min_length=1, max_length=100)


class CompareCommoditiesRequest(BaseModel):
    """Request to compare multiple commodities."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodities: List[CommodityType] = Field(..., min_length=2, max_length=7)
    comparison_factors: Optional[List[str]] = None
    include_trend: bool = True


class GetTrendRequest(BaseModel):
    """Request to get commodity risk trend analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodity: CommodityType
    period_months: int = Field(default=12, ge=1, le=60)
    include_forecast: bool = False


class HealthRequest(BaseModel):
    """Request to check service health."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    include_database: bool = True
    include_cache: bool = True


# =============================================================================
# Response Models
# =============================================================================


class CommodityProfileResponse(BaseModel):
    """Response for commodity profiling."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    profile: CommodityProfile
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class DerivedProductResponse(BaseModel):
    """Response for derived product analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    product: DerivedProduct
    related_products: List[DerivedProduct] = Field(default_factory=list)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class PriceVolatilityResponse(BaseModel):
    """Response for price volatility query."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    price_data: List[PriceData]
    current_volatility: Decimal = Field(..., ge=0, le=1)
    volatility_level: VolatilityLevel
    market_condition: MarketCondition
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class ProductionForecastResponse(BaseModel):
    """Response for production forecast."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    forecast: ProductionForecast
    historical_data: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class SubstitutionRiskResponse(BaseModel):
    """Response for substitution risk detection."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    events: List[SubstitutionEvent]
    total_events: int = Field(..., ge=0)
    pattern_detected: bool = False
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class RegulatoryComplianceResponse(BaseModel):
    """Response for regulatory compliance check."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    requirements: List[RegulatoryRequirement]
    overall_status: ComplianceStatus
    compliance_gaps: List[str] = Field(default_factory=list)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class DDWorkflowResponse(BaseModel):
    """Response for due diligence workflow operations."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    workflow: DDWorkflow
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class PortfolioAnalysisResponse(BaseModel):
    """Response for portfolio risk aggregation."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    analysis: PortfolioAnalysis
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class BatchAnalysisResponse(BaseModel):
    """Response for batch commodity analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    profiles: List[CommodityProfile]
    total_count: int = Field(..., ge=0)
    success_count: int = Field(..., ge=0)
    failure_count: int = Field(..., ge=0)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class ComparisonResponse(BaseModel):
    """Response for commodity comparison."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    profiles: List[CommodityProfile]
    comparison_matrix: Dict[str, Any]
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class TrendResponse(BaseModel):
    """Response for trend analysis."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    commodity: CommodityType
    trend_data: List[Dict[str, Any]]
    trend_direction: str = Field(..., min_length=1, max_length=50)
    processing_time_ms: float
    provenance_hash: str = Field(..., min_length=64, max_length=64)


class HealthResponse(BaseModel):
    """Response for health check."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    status: str = Field(..., min_length=1, max_length=50)
    database_status: Optional[str] = Field(None, max_length=50)
    cache_status: Optional[str] = Field(None, max_length=50)
    version: str = Field(default=VERSION)
    timestamp: datetime = Field(default_factory=_utcnow)
