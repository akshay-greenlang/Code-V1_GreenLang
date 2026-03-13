# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-019 Corruption Index Monitor

Pydantic v2 request/response models for the REST API layer covering all
8 engine domains: CPI monitoring, WGI analysis, bribery risk assessment,
institutional quality scoring, trend analysis, deforestation-corruption
correlation, alert management, and compliance impact assessment.

All numeric risk and financial fields use ``Decimal`` for precision.
All date/time fields use UTC-aware ``datetime``.

Schema Groups (8 domains):
    1. CPI Schemas: CPIScoreRequest/Response, CPIHistoryRequest/Response,
       CPIRankingsRequest/Response, CPIRegionalRequest/Response,
       CPIBatchRequest/Response, CPISummaryResponse
    2. WGI Schemas: WGIIndicatorsRequest/Response, WGIHistoryRequest/Response,
       WGIDimensionRequest/Response, WGIComparisonRequest/Response,
       WGIRankingsRequest/Response
    3. Bribery Schemas: BriberyAssessmentRequest/Response, BriberyProfileResponse,
       SectorRiskRequest/Response, HighRiskCountriesRequest/Response,
       SectorExposureRequest/Response
    4. Institutional Schemas: InstitutionalQualityRequest/Response,
       GovernanceProfileResponse, StrengthAssessmentRequest/Response,
       ForestGovernanceRequest/Response, InstitutionalComparisonRequest/Response
    5. Trend Schemas: TrendAnalysisRequest/Response, TrajectoryRequest/Response,
       PredictionRequest/Response, ImprovingCountriesRequest/Response,
       DeterioratingCountriesRequest/Response
    6. Correlation Schemas: CorrelationAnalysisRequest/Response,
       DeforestationLinkRequest/Response, RegressionRequest/Response,
       HeatmapRequest/Response, CausalPathwayRequest/Response
    7. Alert Schemas: AlertListRequest/Response, AlertDetailResponse,
       AlertConfigRequest/Response, AlertAcknowledgeRequest/Response,
       AlertSummaryRequest/Response
    8. Compliance Schemas: ComplianceImpactRequest/Response,
       CountryImpactResponse, DDRecommendationsRequest/Response,
       CountryClassificationRequest/Response

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
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


class RiskLevelEnum(str, Enum):
    """Risk classification levels for corruption assessment."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class WGIDimensionEnum(str, Enum):
    """World Bank Worldwide Governance Indicators dimensions."""

    VOICE_ACCOUNTABILITY = "voice_accountability"
    POLITICAL_STABILITY = "political_stability"
    GOVERNMENT_EFFECTIVENESS = "government_effectiveness"
    REGULATORY_QUALITY = "regulatory_quality"
    RULE_OF_LAW = "rule_of_law"
    CONTROL_OF_CORRUPTION = "control_of_corruption"


class BriberySectorEnum(str, Enum):
    """Sector-specific bribery risk assessment sectors."""

    FORESTRY = "forestry"
    CUSTOMS = "customs"
    AGRICULTURE = "agriculture"
    MINING = "mining"
    EXTRACTION = "extraction"
    JUDICIARY = "judiciary"


class TrendDirectionEnum(str, Enum):
    """Trend direction classification for temporal analysis."""

    IMPROVING = "improving"
    DETERIORATING = "deteriorating"
    STABLE = "stable"
    VOLATILE = "volatile"


class AlertSeverityEnum(str, Enum):
    """Alert severity levels for corruption index changes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertTypeEnum(str, Enum):
    """Types of corruption index alerts."""

    CPI_CHANGE = "cpi_change"
    WGI_CHANGE = "wgi_change"
    TREND_REVERSAL = "trend_reversal"
    THRESHOLD_BREACH = "threshold_breach"
    COUNTRY_RECLASSIFICATION = "country_reclassification"
    BRIBERY_RISK_ESCALATION = "bribery_risk_escalation"
    INSTITUTIONAL_DEGRADATION = "institutional_degradation"


class AlertStatusEnum(str, Enum):
    """Alert lifecycle status values."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"
    SUPPRESSED = "suppressed"


class ComplianceLevelEnum(str, Enum):
    """EUDR compliance due diligence levels."""

    SIMPLIFIED = "simplified"
    STANDARD = "standard"
    ENHANCED = "enhanced"


class CountryClassificationEnum(str, Enum):
    """EUDR Article 29 country classification levels."""

    LOW_RISK = "low_risk"
    STANDARD_RISK = "standard_risk"
    HIGH_RISK = "high_risk"


class CorrelationStrengthEnum(str, Enum):
    """Correlation strength classification."""

    NEGLIGIBLE = "negligible"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class GovernanceRatingEnum(str, Enum):
    """Governance quality rating classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    CRITICAL = "critical"


class DataSourceEnum(str, Enum):
    """Data source identifiers for provenance tracking."""

    TRANSPARENCY_INTERNATIONAL = "transparency_international"
    WORLD_BANK = "world_bank"
    TRACE_MATRIX = "trace_matrix"
    GLOBAL_FOREST_WATCH = "global_forest_watch"
    INTERNAL = "internal"
    COMPOSITE = "composite"


class RegionEnum(str, Enum):
    """Regional classification for CPI analysis."""

    AFRICA = "africa"
    AMERICAS = "americas"
    ASIA_PACIFIC = "asia_pacific"
    EASTERN_EUROPE_CENTRAL_ASIA = "eastern_europe_central_asia"
    EU_WESTERN_EUROPE = "eu_western_europe"
    MIDDLE_EAST_NORTH_AFRICA = "middle_east_north_africa"
    SUB_SAHARAN_AFRICA = "sub_saharan_africa"


class PredictionConfidenceEnum(str, Enum):
    """Confidence level classification for predictions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRELIABLE = "unreliable"


# =============================================================================
# Common / Shared Schemas
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response for the Corruption Index Monitor API."""

    status: str = Field(
        default="healthy",
        description="Service health status",
        examples=["healthy"],
    )
    agent_id: str = Field(
        default="GL-EUDR-CIM-019",
        description="Agent identifier",
    )
    agent_name: str = Field(
        default="EUDR Corruption Index Monitor",
        description="Human-readable agent name",
    )
    version: str = Field(
        default="1.0.0",
        description="API version",
    )
    component: str = Field(
        default="corruption-index-monitor",
        description="Component identifier",
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
        examples=["Invalid country code provided"],
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
        default="GL-EUDR-CIM-019",
        description="Agent that produced this record",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0.0"),
        ge=Decimal("0.0"),
        description="Processing duration in milliseconds",
    )

    model_config = ConfigDict(from_attributes=True)


class MetadataSchema(BaseModel):
    """Generic metadata schema for enriched responses."""

    request_id: str = Field(
        default_factory=_new_id,
        description="Unique request identifier",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Response timestamp in UTC",
    )
    agent_id: str = Field(
        default="GL-EUDR-CIM-019",
        description="Agent that produced this response",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources consulted for this response",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether response was served from cache",
    )


# =============================================================================
# 1. CPI Schemas - Corruption Perceptions Index Monitoring
# =============================================================================


class CPIScoreEntry(BaseModel):
    """Single CPI score data point for a country."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
        examples=["BR"],
    )
    country_name: str = Field(
        default="",
        description="Human-readable country name",
        examples=["Brazil"],
    )
    year: int = Field(
        ...,
        ge=1995,
        le=2030,
        description="CPI score year",
        examples=[2024],
    )
    score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CPI score on 0-100 scale (0=most corrupt, 100=cleanest)",
        examples=[Decimal("38")],
    )
    rank: Optional[int] = Field(
        None,
        ge=1,
        description="Global ranking position (1=cleanest)",
        examples=[104],
    )
    percentile: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentile rank (0-100)",
        examples=[Decimal("42.2")],
    )
    region: Optional[RegionEnum] = Field(
        None,
        description="Regional classification",
    )
    year_over_year_change: Optional[Decimal] = Field(
        None,
        description="Change from previous year (positive=improvement)",
    )
    data_source: DataSourceEnum = Field(
        default=DataSourceEnum.TRANSPARENCY_INTERNATIONAL,
        description="Data source for this score",
    )

    model_config = ConfigDict(from_attributes=True)

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


class CPIScoreResponse(BaseModel):
    """Response model for single CPI score retrieval."""

    score: CPIScoreEntry = Field(
        ...,
        description="CPI score data for the requested country and year",
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Risk classification based on CPI score",
    )
    eudr_classification: CountryClassificationEnum = Field(
        ...,
        description="EUDR Article 29 country classification",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIHistoryEntry(BaseModel):
    """Single historical CPI data point."""

    year: int = Field(..., ge=1995, le=2030, description="CPI score year")
    score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CPI score for the year",
    )
    rank: Optional[int] = Field(None, ge=1, description="Global ranking")
    change_from_prior: Optional[Decimal] = Field(
        None,
        description="Change from previous year",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIHistoryResponse(BaseModel):
    """Response model for CPI score history retrieval."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    history: List[CPIHistoryEntry] = Field(
        ...,
        description="Historical CPI scores",
    )
    period_start: int = Field(..., description="Start year of history")
    period_end: int = Field(..., description="End year of history")
    average_score: Decimal = Field(
        ...,
        description="Average CPI score over the period",
    )
    trend_direction: TrendDirectionEnum = Field(
        ...,
        description="Overall trend direction over the period",
    )
    total_change: Decimal = Field(
        ...,
        description="Total score change from start to end",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIRankingEntry(BaseModel):
    """Single entry in the CPI rankings list."""

    rank: int = Field(..., ge=1, description="Global ranking position")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CPI score",
    )
    region: Optional[RegionEnum] = Field(None, description="Regional classification")
    year_over_year_change: Optional[Decimal] = Field(
        None,
        description="Change from previous year",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")

    model_config = ConfigDict(from_attributes=True)


class CPIRankingsResponse(BaseModel):
    """Response model for global/regional CPI rankings."""

    year: int = Field(..., description="Rankings year")
    region: Optional[RegionEnum] = Field(
        None,
        description="Region filter applied (None=global)",
    )
    rankings: List[CPIRankingEntry] = Field(
        ...,
        description="Ranked list of countries",
    )
    total_countries: int = Field(
        ...,
        ge=0,
        description="Total countries in rankings",
    )
    global_average: Decimal = Field(
        ...,
        description="Average CPI score across all ranked countries",
    )
    pagination: PaginatedMeta = Field(
        ...,
        description="Pagination metadata",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIRegionalStats(BaseModel):
    """Statistical summary for a region."""

    region: RegionEnum = Field(..., description="Region identifier")
    country_count: int = Field(..., ge=0, description="Countries in region")
    average_score: Decimal = Field(..., description="Average CPI score")
    median_score: Decimal = Field(..., description="Median CPI score")
    min_score: Decimal = Field(..., description="Minimum CPI score in region")
    max_score: Decimal = Field(..., description="Maximum CPI score in region")
    std_deviation: Decimal = Field(..., description="Standard deviation")
    high_risk_count: int = Field(
        ...,
        ge=0,
        description="Number of high-risk countries",
    )
    low_risk_count: int = Field(
        ...,
        ge=0,
        description="Number of low-risk countries",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIRegionalResponse(BaseModel):
    """Response model for regional CPI analysis."""

    year: int = Field(..., description="Analysis year")
    region: RegionEnum = Field(..., description="Region analyzed")
    stats: CPIRegionalStats = Field(..., description="Regional statistics")
    top_performers: List[CPIRankingEntry] = Field(
        ...,
        description="Top performing countries in region",
    )
    bottom_performers: List[CPIRankingEntry] = Field(
        ...,
        description="Lowest performing countries in region",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIBatchCountryEntry(BaseModel):
    """Single country entry in a CPI batch query request."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    year: Optional[int] = Field(
        None,
        ge=1995,
        le=2030,
        description="Specific year (default: latest)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class CPIBatchRequest(BaseModel):
    """Request model for batch CPI score retrieval."""

    countries: List[CPIBatchCountryEntry] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of countries to query (max 100)",
    )
    include_history: bool = Field(
        default=False,
        description="Include historical data for each country",
    )
    year: Optional[int] = Field(
        None,
        ge=1995,
        le=2030,
        description="Default year for all countries (overridable per country)",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIBatchResultEntry(BaseModel):
    """Single result entry in a CPI batch response."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    score: Optional[CPIScoreEntry] = Field(
        None,
        description="CPI score data (None if not available)",
    )
    risk_level: Optional[RiskLevelEnum] = Field(
        None,
        description="Risk classification",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if query failed for this country",
    )

    model_config = ConfigDict(from_attributes=True)


class CPIBatchResponse(BaseModel):
    """Response model for batch CPI score retrieval."""

    results: List[CPIBatchResultEntry] = Field(
        ...,
        description="Batch query results",
    )
    total_queried: int = Field(..., ge=0, description="Total countries queried")
    total_succeeded: int = Field(..., ge=0, description="Successful queries")
    total_failed: int = Field(..., ge=0, description="Failed queries")
    processing_time_ms: Decimal = Field(
        ...,
        ge=Decimal("0.0"),
        description="Total processing time in milliseconds",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CPISummaryResponse(BaseModel):
    """Response model for CPI summary statistics."""

    year: int = Field(..., description="Summary year")
    total_countries: int = Field(
        ...,
        ge=0,
        description="Total countries monitored",
    )
    global_average: Decimal = Field(
        ...,
        description="Global average CPI score",
    )
    global_median: Decimal = Field(
        ...,
        description="Global median CPI score",
    )
    high_risk_countries: int = Field(
        ...,
        ge=0,
        description="Countries classified as high risk",
    )
    moderate_risk_countries: int = Field(
        ...,
        ge=0,
        description="Countries classified as moderate risk",
    )
    low_risk_countries: int = Field(
        ...,
        ge=0,
        description="Countries classified as low risk",
    )
    regional_averages: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Average CPI scores by region",
    )
    year_over_year_change: Decimal = Field(
        ...,
        description="Average global change from prior year",
    )
    improving_countries: int = Field(
        ...,
        ge=0,
        description="Countries with improved scores",
    )
    deteriorating_countries: int = Field(
        ...,
        ge=0,
        description="Countries with worsened scores",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# 2. WGI Schemas - Worldwide Governance Indicators Analysis
# =============================================================================


class WGIDimensionScore(BaseModel):
    """Single WGI dimension score for a country."""

    dimension: WGIDimensionEnum = Field(
        ...,
        description="WGI governance dimension",
    )
    estimate: Decimal = Field(
        ...,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Governance estimate on -2.5 to +2.5 scale",
    )
    standard_error: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Standard error of the estimate",
    )
    percentile_rank: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentile rank (0-100)",
    )
    num_sources: Optional[int] = Field(
        None,
        ge=0,
        description="Number of data sources used",
    )

    model_config = ConfigDict(from_attributes=True)


class WGIIndicatorsResponse(BaseModel):
    """Response model for all 6 WGI dimension indicators for a country."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    year: int = Field(..., description="Indicator year")
    dimensions: List[WGIDimensionScore] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="All 6 WGI dimension scores",
    )
    composite_score: Decimal = Field(
        ...,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Weighted composite governance score",
    )
    governance_rating: GovernanceRatingEnum = Field(
        ...,
        description="Overall governance quality rating",
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Risk classification based on WGI composite",
    )
    data_source: DataSourceEnum = Field(
        default=DataSourceEnum.WORLD_BANK,
        description="Data source identifier",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class WGIHistoryEntry(BaseModel):
    """Single historical WGI data point for a dimension."""

    year: int = Field(..., ge=1996, le=2030, description="Indicator year")
    estimate: Decimal = Field(
        ...,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Governance estimate",
    )
    percentile_rank: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentile rank",
    )
    change_from_prior: Optional[Decimal] = Field(
        None,
        description="Change from previous year",
    )

    model_config = ConfigDict(from_attributes=True)


class WGIHistoryResponse(BaseModel):
    """Response model for WGI indicator history."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    dimension: WGIDimensionEnum = Field(..., description="WGI dimension queried")
    history: List[WGIHistoryEntry] = Field(
        ...,
        description="Historical WGI indicator values",
    )
    period_start: int = Field(..., description="Start year of history")
    period_end: int = Field(..., description="End year of history")
    average_estimate: Decimal = Field(..., description="Average estimate over period")
    trend_direction: TrendDirectionEnum = Field(
        ...,
        description="Trend direction over the period",
    )
    total_change: Decimal = Field(
        ...,
        description="Total change from start to end",
    )
    provenance: ProvenanceInfo = Field(
        ...,
        description="Provenance tracking metadata",
    )
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class WGIDimensionCountryEntry(BaseModel):
    """Single country entry for cross-country dimension analysis."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    estimate: Decimal = Field(
        ...,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Governance estimate",
    )
    percentile_rank: Optional[Decimal] = Field(
        None,
        description="Percentile rank",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")

    model_config = ConfigDict(from_attributes=True)


class WGIDimensionResponse(BaseModel):
    """Response model for cross-country WGI dimension analysis."""

    dimension: WGIDimensionEnum = Field(..., description="WGI dimension analyzed")
    year: int = Field(..., description="Analysis year")
    countries: List[WGIDimensionCountryEntry] = Field(
        ...,
        description="Country scores for this dimension",
    )
    global_average: Decimal = Field(..., description="Global average for dimension")
    global_median: Decimal = Field(..., description="Global median for dimension")
    total_countries: int = Field(..., ge=0, description="Total countries analyzed")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class WGIComparisonRequest(BaseModel):
    """Request model for WGI country comparison."""

    country_codes: List[str] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="ISO 3166-1 alpha-2 country codes to compare (2-20)",
    )
    year: Optional[int] = Field(
        None,
        ge=1996,
        le=2030,
        description="Comparison year (default: latest)",
    )
    dimensions: Optional[List[WGIDimensionEnum]] = Field(
        None,
        description="Specific dimensions to compare (default: all 6)",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Normalize all country codes to uppercase."""
        return [cc.upper().strip() for cc in v]

    model_config = ConfigDict(from_attributes=True)


class WGIComparisonCountryEntry(BaseModel):
    """Single country entry in WGI comparison results."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    dimensions: List[WGIDimensionScore] = Field(
        ...,
        description="WGI dimension scores",
    )
    composite_score: Decimal = Field(
        ...,
        description="Weighted composite governance score",
    )
    governance_rating: GovernanceRatingEnum = Field(
        ...,
        description="Governance quality rating",
    )

    model_config = ConfigDict(from_attributes=True)


class WGIComparisonResponse(BaseModel):
    """Response model for WGI country comparison."""

    year: int = Field(..., description="Comparison year")
    countries: List[WGIComparisonCountryEntry] = Field(
        ...,
        description="Country comparison results",
    )
    differential_matrix: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Pairwise composite score differential matrix",
    )
    best_performer: Optional[str] = Field(
        None,
        description="Country code of the best performer",
    )
    worst_performer: Optional[str] = Field(
        None,
        description="Country code of the worst performer",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class WGIRankingEntry(BaseModel):
    """Single entry in WGI rankings."""

    rank: int = Field(..., ge=1, description="Ranking position")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    estimate: Decimal = Field(
        ...,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Governance estimate",
    )
    percentile_rank: Optional[Decimal] = Field(None, description="Percentile rank")

    model_config = ConfigDict(from_attributes=True)


class WGIRankingsResponse(BaseModel):
    """Response model for WGI rankings by dimension."""

    dimension: WGIDimensionEnum = Field(..., description="Ranked dimension")
    year: int = Field(..., description="Rankings year")
    rankings: List[WGIRankingEntry] = Field(..., description="Ranked country list")
    total_countries: int = Field(..., ge=0, description="Total countries ranked")
    global_average: Decimal = Field(..., description="Global average estimate")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# 3. Bribery Schemas - Sector-Specific Bribery Risk Assessment
# =============================================================================


class BriberyAssessmentRequest(BaseModel):
    """Request model for bribery risk assessment."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    sectors: Optional[List[BriberySectorEnum]] = Field(
        None,
        description="Specific sectors to assess (default: all 6)",
    )
    include_mitigation: bool = Field(
        default=True,
        description="Include mitigation recommendations",
    )
    commodity_type: Optional[str] = Field(
        None,
        description="EUDR commodity context for sector weighting",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()

    model_config = ConfigDict(from_attributes=True)


class BriberySectorScore(BaseModel):
    """Bribery risk score for a specific sector."""

    sector: BriberySectorEnum = Field(..., description="Sector assessed")
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Bribery risk score for sector (0=low risk, 100=high risk)",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")
    weight: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Sector weight in composite score",
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Key factors contributing to the risk score",
    )
    mitigation_measures: List[str] = Field(
        default_factory=list,
        description="Recommended mitigation measures",
    )

    model_config = ConfigDict(from_attributes=True)


class BriberyAssessmentResponse(BaseModel):
    """Response model for bribery risk assessment."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    composite_bribery_risk: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted composite bribery risk score",
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Overall bribery risk classification",
    )
    sector_scores: List[BriberySectorScore] = Field(
        ...,
        description="Per-sector bribery risk scores",
    )
    highest_risk_sector: BriberySectorEnum = Field(
        ...,
        description="Sector with highest bribery risk",
    )
    cpi_correlation: Optional[Decimal] = Field(
        None,
        description="Correlation with CPI score",
    )
    data_source: DataSourceEnum = Field(
        default=DataSourceEnum.TRACE_MATRIX,
        description="Primary data source",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class BriberyProfileResponse(BaseModel):
    """Response model for country bribery risk profile."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    overall_bribery_risk: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Overall bribery risk score",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")
    sector_breakdown: List[BriberySectorScore] = Field(
        ...,
        description="Detailed per-sector breakdown",
    )
    historical_trend: TrendDirectionEnum = Field(
        ...,
        description="Historical bribery risk trend",
    )
    peer_comparison_percentile: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentile among regional peers",
    )
    last_updated: datetime = Field(
        default_factory=_utcnow,
        description="Last data update timestamp",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class SectorRiskEntry(BaseModel):
    """Single sector risk entry for a country."""

    sector: BriberySectorEnum = Field(..., description="Sector")
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Sector risk score",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")
    eudr_relevance: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="EUDR relevance weight for this sector",
    )

    model_config = ConfigDict(from_attributes=True)


class SectorRiskResponse(BaseModel):
    """Response model for sector-specific bribery risks by country."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    sectors: List[SectorRiskEntry] = Field(
        ...,
        description="Sector-specific risk breakdown",
    )
    highest_risk_sector: BriberySectorEnum = Field(
        ...,
        description="Highest risk sector",
    )
    forestry_risk_detail: Optional[BriberySectorScore] = Field(
        None,
        description="Detailed forestry bribery risk (EUDR-critical)",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class HighRiskCountryEntry(BaseModel):
    """Single high-risk country entry."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    composite_bribery_risk: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Composite bribery risk score",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")
    highest_risk_sector: BriberySectorEnum = Field(
        ...,
        description="Highest risk sector",
    )
    eudr_commodity_exposure: List[str] = Field(
        default_factory=list,
        description="EUDR commodities sourced from this country",
    )

    model_config = ConfigDict(from_attributes=True)


class HighRiskCountriesResponse(BaseModel):
    """Response model for high-risk bribery countries list."""

    threshold: Decimal = Field(
        ...,
        description="Bribery risk score threshold used for filtering",
    )
    countries: List[HighRiskCountryEntry] = Field(
        ...,
        description="Countries exceeding the bribery risk threshold",
    )
    total_high_risk: int = Field(
        ...,
        ge=0,
        description="Total number of high-risk countries",
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class SectorExposureCountryEntry(BaseModel):
    """Country entry for cross-country sector analysis."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    sector_risk_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Sector risk score for this country",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")

    model_config = ConfigDict(from_attributes=True)


class SectorExposureResponse(BaseModel):
    """Response model for cross-country sector bribery risk analysis."""

    sector: BriberySectorEnum = Field(..., description="Sector analyzed")
    year: int = Field(..., description="Analysis year")
    countries: List[SectorExposureCountryEntry] = Field(
        ...,
        description="Country-level sector risks",
    )
    sector_global_average: Decimal = Field(
        ...,
        description="Global average risk for this sector",
    )
    high_risk_count: int = Field(
        ...,
        ge=0,
        description="Countries with high sector risk",
    )
    total_countries: int = Field(..., ge=0, description="Total countries analyzed")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# 4. Institutional Schemas - Institutional Quality Assessment
# =============================================================================


class InstitutionalDimensionScore(BaseModel):
    """Score for a single institutional quality dimension."""

    dimension: str = Field(
        ...,
        description="Institutional dimension name",
        examples=["judicial_independence"],
    )
    score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Dimension score (0-100)",
    )
    weight: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Weight in composite score",
    )
    rating: GovernanceRatingEnum = Field(
        ...,
        description="Dimension quality rating",
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="Contributing indicators for this dimension",
    )

    model_config = ConfigDict(from_attributes=True)


class InstitutionalQualityResponse(BaseModel):
    """Response model for institutional quality assessment."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    composite_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted composite institutional quality score",
    )
    governance_rating: GovernanceRatingEnum = Field(
        ...,
        description="Overall governance rating",
    )
    dimensions: List[InstitutionalDimensionScore] = Field(
        ...,
        description="Per-dimension quality scores",
    )
    strongest_dimension: str = Field(
        ...,
        description="Strongest institutional dimension",
    )
    weakest_dimension: str = Field(
        ...,
        description="Weakest institutional dimension",
    )
    risk_level: RiskLevelEnum = Field(..., description="Risk classification")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class GovernanceProfileResponse(BaseModel):
    """Response model for detailed governance profile."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    governance_rating: GovernanceRatingEnum = Field(
        ...,
        description="Overall governance rating",
    )
    institutional_quality_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Composite institutional quality score",
    )
    cpi_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Current CPI score for context",
    )
    wgi_composite: Optional[Decimal] = Field(
        None,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Current WGI composite score for context",
    )
    dimensions: List[InstitutionalDimensionScore] = Field(
        ...,
        description="Detailed dimension scores",
    )
    judicial_independence_detail: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed judicial independence metrics",
    )
    regulatory_enforcement_detail: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed regulatory enforcement metrics",
    )
    forest_governance_detail: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed forest governance metrics",
    )
    law_enforcement_detail: Optional[Dict[str, Any]] = Field(
        None,
        description="Detailed law enforcement metrics",
    )
    historical_trend: TrendDirectionEnum = Field(
        ...,
        description="Historical institutional quality trend",
    )
    peer_countries: List[str] = Field(
        default_factory=list,
        description="Peer countries for benchmarking",
    )
    last_updated: datetime = Field(
        default_factory=_utcnow,
        description="Last data update timestamp",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class StrengthAssessmentRequest(BaseModel):
    """Request model for institutional strength assessment."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    dimensions: Optional[List[str]] = Field(
        None,
        description="Specific dimensions to assess (default: all 4)",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include improvement recommendations",
    )
    benchmark_region: Optional[RegionEnum] = Field(
        None,
        description="Region for peer benchmarking",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()

    model_config = ConfigDict(from_attributes=True)


class StrengthAssessmentResponse(BaseModel):
    """Response model for institutional strength assessment."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    composite_strength: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Composite institutional strength score",
    )
    governance_rating: GovernanceRatingEnum = Field(
        ...,
        description="Governance quality rating",
    )
    dimensions: List[InstitutionalDimensionScore] = Field(
        ...,
        description="Per-dimension strength scores",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key institutional strengths",
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Key institutional weaknesses",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations",
    )
    benchmark_percentile: Optional[Decimal] = Field(
        None,
        description="Percentile among benchmark peers",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class ForestGovernanceResponse(BaseModel):
    """Response model for forest governance assessment."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    forest_governance_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Forest governance quality score",
    )
    governance_rating: GovernanceRatingEnum = Field(
        ...,
        description="Forest governance quality rating",
    )
    legal_framework_strength: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Strength of forest legal framework",
    )
    enforcement_capacity: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Law enforcement capacity for forest protection",
    )
    monitoring_capability: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Forest monitoring and surveillance capability",
    )
    transparency_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Transparency in forest governance",
    )
    corruption_vulnerability: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Vulnerability to corruption in forest sector",
    )
    eudr_readiness: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Readiness for EUDR compliance demands",
    )
    key_risks: List[str] = Field(
        default_factory=list,
        description="Key forest governance risks identified",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class InstitutionalComparisonRequest(BaseModel):
    """Request model for institutional quality comparison."""

    country_codes: List[str] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="ISO 3166-1 alpha-2 country codes to compare (2-20)",
    )
    dimensions: Optional[List[str]] = Field(
        None,
        description="Specific dimensions to compare (default: all 4)",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Normalize all country codes to uppercase."""
        return [cc.upper().strip() for cc in v]

    model_config = ConfigDict(from_attributes=True)


class InstitutionalComparisonCountryEntry(BaseModel):
    """Single country entry in institutional comparison."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    composite_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Composite institutional quality score",
    )
    governance_rating: GovernanceRatingEnum = Field(
        ...,
        description="Governance quality rating",
    )
    dimensions: List[InstitutionalDimensionScore] = Field(
        ...,
        description="Per-dimension scores",
    )

    model_config = ConfigDict(from_attributes=True)


class InstitutionalComparisonResponse(BaseModel):
    """Response model for institutional quality comparison."""

    countries: List[InstitutionalComparisonCountryEntry] = Field(
        ...,
        description="Country comparison results",
    )
    differential_matrix: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Pairwise composite score differential matrix",
    )
    best_performer: Optional[str] = Field(
        None,
        description="Country code of the best performer",
    )
    worst_performer: Optional[str] = Field(
        None,
        description="Country code of the worst performer",
    )
    dimension_leaders: Dict[str, str] = Field(
        default_factory=dict,
        description="Best-performing country per dimension",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# 5. Trend Schemas - Corruption Trend Analysis
# =============================================================================


class TrendAnalysisRequest(BaseModel):
    """Request model for corruption trend analysis."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    index_type: str = Field(
        default="cpi",
        description="Index type to analyze: cpi, wgi, or composite",
    )
    start_year: Optional[int] = Field(
        None,
        ge=1995,
        le=2030,
        description="Analysis start year",
    )
    end_year: Optional[int] = Field(
        None,
        ge=1995,
        le=2030,
        description="Analysis end year",
    )
    wgi_dimension: Optional[WGIDimensionEnum] = Field(
        None,
        description="Specific WGI dimension (if index_type=wgi)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()

    @field_validator("index_type")
    @classmethod
    def validate_index_type(cls, v: str) -> str:
        """Validate index type."""
        allowed = {"cpi", "wgi", "composite"}
        if v.lower() not in allowed:
            raise ValueError(f"index_type must be one of {allowed}")
        return v.lower()

    model_config = ConfigDict(from_attributes=True)


class TrendDataPoint(BaseModel):
    """Single data point in a trend analysis."""

    year: int = Field(..., description="Year")
    value: Decimal = Field(..., description="Index value")
    predicted: bool = Field(
        default=False,
        description="Whether this is a predicted (not observed) value",
    )
    confidence_lower: Optional[Decimal] = Field(
        None,
        description="Lower bound of confidence interval",
    )
    confidence_upper: Optional[Decimal] = Field(
        None,
        description="Upper bound of confidence interval",
    )

    model_config = ConfigDict(from_attributes=True)


class TrendAnalysisResponse(BaseModel):
    """Response model for corruption trend analysis."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    index_type: str = Field(..., description="Index type analyzed")
    trend_direction: TrendDirectionEnum = Field(
        ...,
        description="Overall trend direction",
    )
    slope: Decimal = Field(
        ...,
        description="Linear regression slope (annual rate of change)",
    )
    r_squared: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="R-squared goodness of fit",
    )
    trend_reliable: bool = Field(
        ...,
        description="Whether trend meets min R-squared threshold",
    )
    data_points: List[TrendDataPoint] = Field(
        ...,
        description="Historical data points used for analysis",
    )
    period_start: int = Field(..., description="Analysis start year")
    period_end: int = Field(..., description="Analysis end year")
    total_change: Decimal = Field(
        ...,
        description="Total change over the analysis period",
    )
    annualized_change: Decimal = Field(
        ...,
        description="Annualized rate of change",
    )
    trend_reversal_detected: bool = Field(
        default=False,
        description="Whether a trend reversal was detected",
    )
    reversal_year: Optional[int] = Field(
        None,
        description="Year of detected trend reversal",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class TrajectoryResponse(BaseModel):
    """Response model for country corruption trajectory."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    trajectory_direction: TrendDirectionEnum = Field(
        ...,
        description="Current trajectory direction",
    )
    trajectory_strength: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Strength of the trajectory (0=weak, 1=strong)",
    )
    cpi_trajectory: Optional[TrendDataPoint] = Field(
        None,
        description="Latest CPI trajectory point",
    )
    wgi_trajectory: Optional[TrendDataPoint] = Field(
        None,
        description="Latest WGI trajectory point",
    )
    momentum: Decimal = Field(
        ...,
        description="Momentum indicator (positive=improving, negative=worsening)",
    )
    acceleration: Decimal = Field(
        ...,
        description="Rate of change of momentum",
    )
    phase: str = Field(
        ...,
        description="Trajectory phase (accelerating, decelerating, stable, inflecting)",
    )
    risk_outlook: RiskLevelEnum = Field(
        ...,
        description="Forward-looking risk outlook",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class PredictionRequest(BaseModel):
    """Request model for corruption index prediction."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    index_type: str = Field(
        default="cpi",
        description="Index type to predict: cpi, wgi, or composite",
    )
    horizon_years: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of years to predict forward",
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.95"),
        ge=Decimal("0.5"),
        le=Decimal("0.99"),
        description="Confidence level for prediction intervals",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    """Response model for corruption index prediction."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    index_type: str = Field(..., description="Index type predicted")
    predictions: List[TrendDataPoint] = Field(
        ...,
        description="Predicted future data points with confidence intervals",
    )
    prediction_confidence: PredictionConfidenceEnum = Field(
        ...,
        description="Overall confidence in predictions",
    )
    model_r_squared: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Model R-squared from historical fit",
    )
    base_year: int = Field(..., description="Last observed year used as base")
    base_value: Decimal = Field(..., description="Last observed value")
    predicted_risk_trajectory: TrendDirectionEnum = Field(
        ...,
        description="Predicted risk trajectory",
    )
    warning_flags: List[str] = Field(
        default_factory=list,
        description="Warning flags about prediction reliability",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class TrendCountryEntry(BaseModel):
    """Single country entry for improving/deteriorating lists."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    current_score: Decimal = Field(..., description="Current index score")
    previous_score: Decimal = Field(..., description="Previous period score")
    change: Decimal = Field(..., description="Score change (absolute)")
    change_percent: Decimal = Field(..., description="Score change (percentage)")
    trend_direction: TrendDirectionEnum = Field(
        ...,
        description="Trend direction",
    )
    region: Optional[RegionEnum] = Field(None, description="Country region")

    model_config = ConfigDict(from_attributes=True)


class ImprovingCountriesResponse(BaseModel):
    """Response model for countries with improving corruption indices."""

    period: str = Field(
        ...,
        description="Analysis period description",
        examples=["2020-2024"],
    )
    index_type: str = Field(..., description="Index type analyzed")
    countries: List[TrendCountryEntry] = Field(
        ...,
        description="Countries with improving trends, sorted by magnitude",
    )
    total_improving: int = Field(
        ...,
        ge=0,
        description="Total number of improving countries",
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class DeterioratingCountriesResponse(BaseModel):
    """Response model for countries with deteriorating corruption indices."""

    period: str = Field(
        ...,
        description="Analysis period description",
        examples=["2020-2024"],
    )
    index_type: str = Field(..., description="Index type analyzed")
    countries: List[TrendCountryEntry] = Field(
        ...,
        description="Countries with deteriorating trends, sorted by magnitude",
    )
    total_deteriorating: int = Field(
        ...,
        ge=0,
        description="Total number of deteriorating countries",
    )
    eudr_high_risk_overlap: int = Field(
        ...,
        ge=0,
        description="Countries that are both deteriorating and EUDR high-risk",
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# 6. Correlation Schemas - Deforestation-Corruption Correlation
# =============================================================================


class CorrelationAnalysisRequest(BaseModel):
    """Request model for corruption-deforestation correlation analysis."""

    country_codes: Optional[List[str]] = Field(
        None,
        description="Specific countries to analyze (default: all available)",
    )
    index_type: str = Field(
        default="cpi",
        description="Corruption index type: cpi, wgi, or composite",
    )
    deforestation_metric: str = Field(
        default="annual_loss_hectares",
        description="Deforestation metric for correlation",
    )
    start_year: Optional[int] = Field(
        None,
        ge=2000,
        le=2030,
        description="Analysis start year",
    )
    end_year: Optional[int] = Field(
        None,
        ge=2000,
        le=2030,
        description="Analysis end year",
    )
    min_data_points: int = Field(
        default=10,
        ge=3,
        description="Minimum data points required for valid correlation",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Normalize all country codes to uppercase."""
        if v is not None:
            return [cc.upper().strip() for cc in v]
        return v

    model_config = ConfigDict(from_attributes=True)


class CorrelationResultEntry(BaseModel):
    """Single correlation result."""

    variable_pair: str = Field(
        ...,
        description="Variable pair description",
        examples=["CPI vs Annual Forest Loss"],
    )
    pearson_r: Decimal = Field(
        ...,
        ge=Decimal("-1.0"),
        le=Decimal("1.0"),
        description="Pearson correlation coefficient",
    )
    p_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Statistical significance p-value",
    )
    significant: bool = Field(
        ...,
        description="Whether correlation is statistically significant",
    )
    strength: CorrelationStrengthEnum = Field(
        ...,
        description="Correlation strength classification",
    )
    n_observations: int = Field(
        ...,
        ge=0,
        description="Number of observations used",
    )
    direction: str = Field(
        ...,
        description="Correlation direction: positive, negative, or none",
    )

    model_config = ConfigDict(from_attributes=True)


class CorrelationAnalysisResponse(BaseModel):
    """Response model for correlation analysis."""

    correlations: List[CorrelationResultEntry] = Field(
        ...,
        description="Correlation analysis results",
    )
    primary_correlation: CorrelationResultEntry = Field(
        ...,
        description="Primary corruption-deforestation correlation",
    )
    analysis_period: str = Field(
        ...,
        description="Analysis period description",
    )
    total_countries_analyzed: int = Field(
        ...,
        ge=0,
        description="Total countries included in analysis",
    )
    data_quality_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Overall data quality score for the analysis",
    )
    interpretation: str = Field(
        ...,
        description="Human-readable interpretation of correlation results",
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Methodological caveats and limitations",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class DeforestationLinkResponse(BaseModel):
    """Response model for country-specific deforestation-corruption link."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    corruption_score: Decimal = Field(
        ...,
        description="Current corruption index score",
    )
    deforestation_rate: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Current annual deforestation rate (hectares)",
    )
    correlation: CorrelationResultEntry = Field(
        ...,
        description="Country-specific correlation result",
    )
    risk_amplification_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Factor by which corruption amplifies deforestation risk",
    )
    historical_comparison: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Historical comparison of corruption vs deforestation trends",
    )
    peer_benchmarks: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Benchmarks against regional peers",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class RegressionRequest(BaseModel):
    """Request model for regression model building."""

    dependent_variable: str = Field(
        default="deforestation_rate",
        description="Dependent variable (Y axis)",
    )
    independent_variables: List[str] = Field(
        default_factory=lambda: ["cpi_score"],
        description="Independent variables (X axes)",
    )
    country_codes: Optional[List[str]] = Field(
        None,
        description="Countries to include (default: all available)",
    )
    model_type: str = Field(
        default="linear",
        description="Regression model type: linear, polynomial, or logistic",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Normalize all country codes to uppercase."""
        if v is not None:
            return [cc.upper().strip() for cc in v]
        return v

    model_config = ConfigDict(from_attributes=True)


class RegressionResponse(BaseModel):
    """Response model for regression analysis."""

    model_type: str = Field(..., description="Regression model type used")
    dependent_variable: str = Field(..., description="Dependent variable")
    independent_variables: List[str] = Field(
        ...,
        description="Independent variables",
    )
    coefficients: Dict[str, Decimal] = Field(
        ...,
        description="Regression coefficients (variable: value)",
    )
    intercept: Decimal = Field(..., description="Model intercept")
    r_squared: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="R-squared goodness of fit",
    )
    adjusted_r_squared: Decimal = Field(
        ...,
        description="Adjusted R-squared",
    )
    f_statistic: Optional[Decimal] = Field(
        None,
        description="F-statistic for model significance",
    )
    p_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Model p-value",
    )
    n_observations: int = Field(..., ge=0, description="Number of observations")
    residual_std_error: Optional[Decimal] = Field(
        None,
        description="Residual standard error",
    )
    interpretation: str = Field(
        ...,
        description="Human-readable model interpretation",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class HeatmapCell(BaseModel):
    """Single cell in a corruption-deforestation heatmap."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    corruption_score: Decimal = Field(..., description="Corruption index value")
    deforestation_rate: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Annual deforestation rate",
    )
    risk_quadrant: str = Field(
        ...,
        description="Risk quadrant: high_corruption_high_deforestation, etc.",
    )
    risk_level: RiskLevelEnum = Field(..., description="Combined risk level")

    model_config = ConfigDict(from_attributes=True)


class HeatmapResponse(BaseModel):
    """Response model for corruption-deforestation heatmap."""

    cells: List[HeatmapCell] = Field(
        ...,
        description="Heatmap data cells (one per country)",
    )
    corruption_axis_label: str = Field(
        ...,
        description="Label for corruption axis",
    )
    deforestation_axis_label: str = Field(
        ...,
        description="Label for deforestation axis",
    )
    quadrant_counts: Dict[str, int] = Field(
        ...,
        description="Number of countries in each quadrant",
    )
    total_countries: int = Field(..., ge=0, description="Total countries in heatmap")
    high_risk_quadrant_countries: List[str] = Field(
        default_factory=list,
        description="Countries in high-corruption + high-deforestation quadrant",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CausalPathwayStep(BaseModel):
    """Single step in a causal pathway."""

    step_number: int = Field(..., ge=1, description="Step position in pathway")
    mechanism: str = Field(
        ...,
        description="Causal mechanism description",
    )
    evidence_strength: CorrelationStrengthEnum = Field(
        ...,
        description="Strength of evidence for this mechanism",
    )
    supporting_data: List[str] = Field(
        default_factory=list,
        description="Supporting data references",
    )

    model_config = ConfigDict(from_attributes=True)


class CausalPathway(BaseModel):
    """Complete causal pathway from corruption to deforestation."""

    pathway_id: str = Field(
        default_factory=_new_id,
        description="Unique pathway identifier",
    )
    pathway_name: str = Field(
        ...,
        description="Pathway name",
        examples=["Weak Enforcement Pathway"],
    )
    steps: List[CausalPathwayStep] = Field(
        ...,
        description="Sequential steps in the causal pathway",
    )
    overall_evidence_strength: CorrelationStrengthEnum = Field(
        ...,
        description="Overall evidence strength for this pathway",
    )
    relevance_to_eudr: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1.0"),
        description="Relevance to EUDR compliance (0-1)",
    )

    model_config = ConfigDict(from_attributes=True)


class CausalPathwayResponse(BaseModel):
    """Response model for causal pathway analysis."""

    pathways: List[CausalPathway] = Field(
        ...,
        description="Identified causal pathways",
    )
    primary_pathway: str = Field(
        ...,
        description="ID of the strongest causal pathway",
    )
    total_pathways: int = Field(
        ...,
        ge=0,
        description="Total pathways identified",
    )
    methodology_notes: str = Field(
        ...,
        description="Methodology used for causal analysis",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Analysis limitations and caveats",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# 7. Alert Schemas - Corruption Index Alert Management
# =============================================================================


class AlertEntry(BaseModel):
    """Single alert entry in the alert list."""

    alert_id: str = Field(
        default_factory=_new_id,
        description="Unique alert identifier",
    )
    alert_type: AlertTypeEnum = Field(..., description="Type of alert")
    severity: AlertSeverityEnum = Field(..., description="Alert severity level")
    status: AlertStatusEnum = Field(
        default=AlertStatusEnum.ACTIVE,
        description="Alert lifecycle status",
    )
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed alert description")
    index_type: str = Field(
        ...,
        description="Index that triggered the alert (cpi, wgi, bribery, etc.)",
    )
    previous_value: Optional[Decimal] = Field(
        None,
        description="Previous index value",
    )
    current_value: Optional[Decimal] = Field(
        None,
        description="Current index value triggering the alert",
    )
    change_magnitude: Optional[Decimal] = Field(
        None,
        description="Magnitude of the change",
    )
    threshold_breached: Optional[Decimal] = Field(
        None,
        description="Threshold value that was breached",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Alert creation timestamp",
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="Alert acknowledgement timestamp",
    )
    acknowledged_by: Optional[str] = Field(
        None,
        description="User who acknowledged the alert",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="Alert resolution timestamp",
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Alert expiration timestamp",
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions for this alert",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertListResponse(BaseModel):
    """Response model for alert list with pagination."""

    alerts: List[AlertEntry] = Field(
        ...,
        description="List of alerts matching the query",
    )
    total_alerts: int = Field(..., ge=0, description="Total alerts matching filters")
    active_count: int = Field(..., ge=0, description="Active alerts")
    critical_count: int = Field(
        ...,
        ge=0,
        description="Critical severity active alerts",
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertDetailResponse(BaseModel):
    """Response model for individual alert detail."""

    alert: AlertEntry = Field(..., description="Full alert details")
    related_alerts: List[str] = Field(
        default_factory=list,
        description="IDs of related alerts for the same country",
    )
    country_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Country context data (CPI, WGI, classification)",
    )
    historical_alerts: int = Field(
        ...,
        ge=0,
        description="Total historical alerts for this country",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertConfigRequest(BaseModel):
    """Request model for alert rule configuration."""

    alert_type: AlertTypeEnum = Field(..., description="Alert type to configure")
    enabled: bool = Field(default=True, description="Whether the alert rule is active")
    severity_override: Optional[AlertSeverityEnum] = Field(
        None,
        description="Override default severity for this rule",
    )
    threshold_value: Optional[Decimal] = Field(
        None,
        description="Custom threshold value for triggering",
    )
    country_codes: Optional[List[str]] = Field(
        None,
        description="Specific countries (None=all countries)",
    )
    cooldown_hours: int = Field(
        default=24,
        ge=0,
        le=720,
        description="Cooldown period between duplicate alerts (hours)",
    )
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email", "dashboard"],
        description="Notification channels for this rule",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertConfigResponse(BaseModel):
    """Response model for alert rule configuration."""

    config_id: str = Field(
        default_factory=_new_id,
        description="Configuration identifier",
    )
    alert_type: AlertTypeEnum = Field(..., description="Alert type configured")
    enabled: bool = Field(..., description="Whether rule is active")
    severity: Optional[AlertSeverityEnum] = Field(
        None,
        description="Severity override (None=use default)",
    )
    threshold_value: Optional[Decimal] = Field(
        None,
        description="Custom threshold value",
    )
    country_scope: str = Field(
        ...,
        description="Country scope description (all or specific list)",
    )
    cooldown_hours: int = Field(..., description="Cooldown period in hours")
    notification_channels: List[str] = Field(
        ...,
        description="Active notification channels",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Configuration creation timestamp",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertAcknowledgeRequest(BaseModel):
    """Request model for alert acknowledgement."""

    notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Acknowledgement notes",
    )
    action_taken: Optional[str] = Field(
        None,
        max_length=1000,
        description="Action taken in response to the alert",
    )
    resolve: bool = Field(
        default=False,
        description="Also resolve the alert (not just acknowledge)",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertAcknowledgeResponse(BaseModel):
    """Response model for alert acknowledgement."""

    alert_id: str = Field(..., description="Acknowledged alert ID")
    status: AlertStatusEnum = Field(
        ...,
        description="New alert status after acknowledgement",
    )
    acknowledged_at: datetime = Field(
        default_factory=_utcnow,
        description="Acknowledgement timestamp",
    )
    acknowledged_by: str = Field(..., description="User who acknowledged")
    resolved: bool = Field(..., description="Whether alert was also resolved")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertSeverityCount(BaseModel):
    """Alert count grouped by severity."""

    severity: AlertSeverityEnum = Field(..., description="Severity level")
    count: int = Field(..., ge=0, description="Number of alerts")

    model_config = ConfigDict(from_attributes=True)


class AlertTypeSummary(BaseModel):
    """Alert count grouped by type."""

    alert_type: AlertTypeEnum = Field(..., description="Alert type")
    count: int = Field(..., ge=0, description="Number of alerts")
    latest: Optional[datetime] = Field(
        None,
        description="Timestamp of most recent alert of this type",
    )

    model_config = ConfigDict(from_attributes=True)


class AlertSummaryResponse(BaseModel):
    """Response model for alert summary statistics."""

    total_active: int = Field(..., ge=0, description="Total active alerts")
    total_acknowledged: int = Field(
        ...,
        ge=0,
        description="Total acknowledged (unresolved) alerts",
    )
    total_resolved: int = Field(
        ...,
        ge=0,
        description="Total resolved alerts",
    )
    severity_breakdown: List[AlertSeverityCount] = Field(
        ...,
        description="Active alerts grouped by severity",
    )
    type_breakdown: List[AlertTypeSummary] = Field(
        ...,
        description="Active alerts grouped by type",
    )
    top_affected_countries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Countries with most active alerts",
    )
    alerts_last_24h: int = Field(
        ...,
        ge=0,
        description="Alerts generated in the last 24 hours",
    )
    alerts_last_7d: int = Field(
        ...,
        ge=0,
        description="Alerts generated in the last 7 days",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# 8. Compliance Schemas - EUDR Compliance Impact Assessment
# =============================================================================


class ComplianceImpactRequest(BaseModel):
    """Request model for EUDR compliance impact assessment."""

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity_types: Optional[List[str]] = Field(
        None,
        description="EUDR commodity types to consider (default: all)",
    )
    include_cost_estimates: bool = Field(
        default=True,
        description="Include due diligence cost estimates",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Include compliance recommendations",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()

    model_config = ConfigDict(from_attributes=True)


class DueDiligenceCostEstimate(BaseModel):
    """Cost estimate for due diligence activities."""

    dd_level: ComplianceLevelEnum = Field(
        ...,
        description="Due diligence level",
    )
    estimated_cost_eur: Decimal = Field(
        ...,
        ge=Decimal("0"),
        description="Estimated cost in EUR",
    )
    audit_frequency_months: int = Field(
        ...,
        ge=1,
        description="Recommended audit frequency in months",
    )
    estimated_duration_days: int = Field(
        ...,
        ge=1,
        description="Estimated duration in days",
    )
    required_resources: List[str] = Field(
        default_factory=list,
        description="Resources required for due diligence",
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceImpactResponse(BaseModel):
    """Response model for comprehensive EUDR compliance impact assessment."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    eudr_classification: CountryClassificationEnum = Field(
        ...,
        description="EUDR Article 29 country classification",
    )
    required_dd_level: ComplianceLevelEnum = Field(
        ...,
        description="Required due diligence level",
    )
    cpi_score: Optional[Decimal] = Field(
        None,
        description="CPI score used for classification",
    )
    wgi_composite: Optional[Decimal] = Field(
        None,
        description="WGI composite used for classification",
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Key risk factors affecting classification",
    )
    mitigating_factors: List[str] = Field(
        default_factory=list,
        description="Mitigating factors",
    )
    cost_estimates: Optional[DueDiligenceCostEstimate] = Field(
        None,
        description="Due diligence cost estimates",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Compliance recommendations",
    )
    regulatory_articles: List[str] = Field(
        default_factory=lambda: ["Art. 10", "Art. 11", "Art. 29"],
        description="Applicable EUDR articles",
    )
    classification_rationale: str = Field(
        ...,
        description="Rationale for the classification decision",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CountryImpactResponse(BaseModel):
    """Response model for country-specific compliance impact profile."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    eudr_classification: CountryClassificationEnum = Field(
        ...,
        description="Current EUDR classification",
    )
    previous_classification: Optional[CountryClassificationEnum] = Field(
        None,
        description="Previous classification (if changed)",
    )
    classification_changed: bool = Field(
        default=False,
        description="Whether classification recently changed",
    )
    change_date: Optional[date] = Field(
        None,
        description="Date of most recent classification change",
    )
    required_dd_level: ComplianceLevelEnum = Field(
        ...,
        description="Required due diligence level",
    )
    corruption_indices: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Current corruption index values (cpi, wgi, bribery, etc.)",
    )
    risk_trajectory: TrendDirectionEnum = Field(
        ...,
        description="Risk trajectory direction",
    )
    next_review_date: Optional[date] = Field(
        None,
        description="Estimated next review date for classification",
    )
    active_alerts: int = Field(
        ...,
        ge=0,
        description="Number of active alerts for this country",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class DDRecommendationEntry(BaseModel):
    """Single due diligence recommendation."""

    recommendation_id: str = Field(
        default_factory=_new_id,
        description="Recommendation identifier",
    )
    category: str = Field(
        ...,
        description="Recommendation category",
        examples=["enhanced_monitoring", "supplier_audit", "documentation"],
    )
    priority: str = Field(
        ...,
        description="Priority level: critical, high, medium, low",
    )
    title: str = Field(..., description="Recommendation title")
    description: str = Field(
        ...,
        description="Detailed recommendation description",
    )
    estimated_effort: str = Field(
        ...,
        description="Estimated effort to implement",
        examples=["2-4 weeks", "1-2 months"],
    )
    applicable_articles: List[str] = Field(
        default_factory=list,
        description="Applicable EUDR articles",
    )

    model_config = ConfigDict(from_attributes=True)


class DDRecommendationsResponse(BaseModel):
    """Response model for due diligence recommendations."""

    country_code: Optional[str] = Field(
        None,
        description="Country code (if country-specific)",
    )
    dd_level: ComplianceLevelEnum = Field(
        ...,
        description="Due diligence level these recommendations apply to",
    )
    recommendations: List[DDRecommendationEntry] = Field(
        ...,
        description="Prioritized recommendations",
    )
    total_recommendations: int = Field(
        ...,
        ge=0,
        description="Total number of recommendations",
    )
    critical_count: int = Field(
        ...,
        ge=0,
        description="Number of critical-priority recommendations",
    )
    implementation_timeline: str = Field(
        ...,
        description="Estimated overall implementation timeline",
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


class CountryClassificationEntry(BaseModel):
    """Single country classification entry."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    country_name: str = Field(default="", description="Country name")
    classification: CountryClassificationEnum = Field(
        ...,
        description="EUDR Article 29 classification",
    )
    required_dd_level: ComplianceLevelEnum = Field(
        ...,
        description="Required due diligence level",
    )
    cpi_score: Optional[Decimal] = Field(None, description="Current CPI score")
    wgi_composite: Optional[Decimal] = Field(
        None,
        description="Current WGI composite",
    )
    risk_level: RiskLevelEnum = Field(..., description="Overall risk classification")
    region: Optional[RegionEnum] = Field(None, description="Regional classification")

    model_config = ConfigDict(from_attributes=True)


class CountryClassificationResponse(BaseModel):
    """Response model for EUDR country classifications."""

    classifications: List[CountryClassificationEntry] = Field(
        ...,
        description="Country classifications",
    )
    total_countries: int = Field(
        ...,
        ge=0,
        description="Total countries classified",
    )
    low_risk_count: int = Field(
        ...,
        ge=0,
        description="Countries classified as low risk",
    )
    standard_risk_count: int = Field(
        ...,
        ge=0,
        description="Countries classified as standard risk",
    )
    high_risk_count: int = Field(
        ...,
        ge=0,
        description="Countries classified as high risk",
    )
    classification_date: date = Field(
        ...,
        description="Date of the classification assessment",
    )
    next_review_date: Optional[date] = Field(
        None,
        description="Estimated next review date",
    )
    methodology: str = Field(
        default="CPI + WGI composite per EUDR Article 29",
        description="Classification methodology",
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking metadata")
    metadata: MetadataSchema = Field(
        default_factory=MetadataSchema,
        description="Response metadata",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "RiskLevelEnum",
    "WGIDimensionEnum",
    "BriberySectorEnum",
    "TrendDirectionEnum",
    "AlertSeverityEnum",
    "AlertTypeEnum",
    "AlertStatusEnum",
    "ComplianceLevelEnum",
    "CountryClassificationEnum",
    "CorrelationStrengthEnum",
    "GovernanceRatingEnum",
    "DataSourceEnum",
    "RegionEnum",
    "PredictionConfidenceEnum",
    # Common
    "HealthResponse",
    "ErrorResponse",
    "PaginatedMeta",
    "ProvenanceInfo",
    "MetadataSchema",
    # CPI
    "CPIScoreEntry",
    "CPIScoreResponse",
    "CPIHistoryEntry",
    "CPIHistoryResponse",
    "CPIRankingEntry",
    "CPIRankingsResponse",
    "CPIRegionalStats",
    "CPIRegionalResponse",
    "CPIBatchCountryEntry",
    "CPIBatchRequest",
    "CPIBatchResultEntry",
    "CPIBatchResponse",
    "CPISummaryResponse",
    # WGI
    "WGIDimensionScore",
    "WGIIndicatorsResponse",
    "WGIHistoryEntry",
    "WGIHistoryResponse",
    "WGIDimensionCountryEntry",
    "WGIDimensionResponse",
    "WGIComparisonRequest",
    "WGIComparisonCountryEntry",
    "WGIComparisonResponse",
    "WGIRankingEntry",
    "WGIRankingsResponse",
    # Bribery
    "BriberyAssessmentRequest",
    "BriberySectorScore",
    "BriberyAssessmentResponse",
    "BriberyProfileResponse",
    "SectorRiskEntry",
    "SectorRiskResponse",
    "HighRiskCountryEntry",
    "HighRiskCountriesResponse",
    "SectorExposureCountryEntry",
    "SectorExposureResponse",
    # Institutional
    "InstitutionalDimensionScore",
    "InstitutionalQualityResponse",
    "GovernanceProfileResponse",
    "StrengthAssessmentRequest",
    "StrengthAssessmentResponse",
    "ForestGovernanceResponse",
    "InstitutionalComparisonRequest",
    "InstitutionalComparisonCountryEntry",
    "InstitutionalComparisonResponse",
    # Trend
    "TrendAnalysisRequest",
    "TrendDataPoint",
    "TrendAnalysisResponse",
    "TrajectoryResponse",
    "PredictionRequest",
    "PredictionResponse",
    "TrendCountryEntry",
    "ImprovingCountriesResponse",
    "DeterioratingCountriesResponse",
    # Correlation
    "CorrelationAnalysisRequest",
    "CorrelationResultEntry",
    "CorrelationAnalysisResponse",
    "DeforestationLinkResponse",
    "RegressionRequest",
    "RegressionResponse",
    "HeatmapCell",
    "HeatmapResponse",
    "CausalPathwayStep",
    "CausalPathway",
    "CausalPathwayResponse",
    # Alert
    "AlertEntry",
    "AlertListResponse",
    "AlertDetailResponse",
    "AlertConfigRequest",
    "AlertConfigResponse",
    "AlertAcknowledgeRequest",
    "AlertAcknowledgeResponse",
    "AlertSeverityCount",
    "AlertTypeSummary",
    "AlertSummaryResponse",
    # Compliance
    "ComplianceImpactRequest",
    "DueDiligenceCostEstimate",
    "ComplianceImpactResponse",
    "CountryImpactResponse",
    "DDRecommendationEntry",
    "DDRecommendationsResponse",
    "CountryClassificationEntry",
    "CountryClassificationResponse",
]
