# -*- coding: utf-8 -*-
"""
Corruption Index Monitor Data Models - AGENT-EUDR-019

Pydantic v2 data models for the Corruption Index Monitor Agent covering
Transparency International CPI monitoring (0-100 scale), World Bank WGI
analysis (6 dimensions, -2.5 to +2.5 scale), sector-specific bribery risk
assessment, institutional quality scoring, trend analysis with linear
regression and prediction, deforestation-corruption correlation analysis,
alert generation for significant index changes, and compliance impact
assessment mapping corruption indices to EUDR Article 29 country
classifications with due diligence level determination.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all corruption index monitoring operations per
EU 2023/1115 Articles 10, 11, 13, 29, and 31.

Enumerations (10):
    - WGIDimension, RiskLevel, TrendDirection, AlertSeverity,
      ComplianceLevel, BriberySector, CountryClassification,
      CorrelationStrength, GovernanceRating, DataSource

Core Models (10):
    - CPIScore, WGIIndicator, BriberyRiskAssessment,
      InstitutionalQualityScore, TrendAnalysis, DeforestationCorrelation,
      Alert, ComplianceImpact, CountryProfile, AuditLogEntry

Request Models (10):
    - QueryCPIRequest, QueryWGIRequest, AssessBriberyRiskRequest,
      EvaluateInstitutionalQualityRequest, AnalyzeTrendRequest,
      AnalyzeCorrelationRequest, GenerateAlertRequest,
      AssessComplianceImpactRequest, BuildCountryProfileRequest,
      HealthCheckRequest

Response Models (10):
    - CPIScoreResponse, WGIIndicatorResponse, BriberyRiskResponse,
      InstitutionalQualityResponse, TrendAnalysisResponse,
      CorrelationResponse, AlertResponse, ComplianceImpactResponse,
      CountryProfileResponse, HealthCheckResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
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

#: Maximum CPI score value (cleanest).
MAX_CPI_SCORE: int = 100

#: Minimum CPI score value (most corrupt).
MIN_CPI_SCORE: int = 0

#: Maximum WGI estimate value.
MAX_WGI_ESTIMATE: float = 2.5

#: Minimum WGI estimate value.
MIN_WGI_ESTIMATE: float = -2.5

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 500

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Supported regions for CPI analysis.
SUPPORTED_REGIONS: List[str] = [
    "africa",
    "americas",
    "asia_pacific",
    "eastern_europe_central_asia",
    "eu_western_europe",
    "middle_east_north_africa",
    "sub_saharan_africa",
]

#: WGI dimensions list.
WGI_DIMENSIONS_LIST: List[str] = [
    "voice_accountability",
    "political_stability",
    "government_effectiveness",
    "regulatory_quality",
    "rule_of_law",
    "control_of_corruption",
]


# ---------------------------------------------------------------------------
# Enumerations (10)
# ---------------------------------------------------------------------------


class WGIDimension(str, Enum):
    """World Bank Worldwide Governance Indicators dimensions.

    The WGI captures six broad dimensions of governance for over
    200 countries and territories. Scores range from -2.5 (weakest)
    to +2.5 (strongest governance).

    Reference: https://info.worldbank.org/governance/wgi/
    """

    VOICE_ACCOUNTABILITY = "voice_accountability"
    """Voice and Accountability: citizen participation, free media, civil liberties."""

    POLITICAL_STABILITY = "political_stability"
    """Political Stability and Absence of Violence/Terrorism."""

    GOVERNMENT_EFFECTIVENESS = "government_effectiveness"
    """Quality of public services, policy formulation and implementation."""

    REGULATORY_QUALITY = "regulatory_quality"
    """Sound policies and regulations permitting private sector development."""

    RULE_OF_LAW = "rule_of_law"
    """Confidence in and abidance by rules of society, contract enforcement."""

    CONTROL_OF_CORRUPTION = "control_of_corruption"
    """Extent to which public power is exercised for private gain."""


class RiskLevel(str, Enum):
    """Risk level classification for corruption and governance assessment.

    Four-tier risk classification aligned with EUDR Article 29
    country benchmarking methodology.
    """

    LOW = "low"
    """Low risk: CPI > 70, strong governance indicators."""

    MODERATE = "moderate"
    """Moderate risk: CPI 51-70, acceptable governance indicators."""

    HIGH = "high"
    """High risk: CPI 31-50, concerning governance indicators."""

    CRITICAL = "critical"
    """Critical risk: CPI <= 30, severe governance deficiencies."""


class TrendDirection(str, Enum):
    """Direction of trend in corruption or governance indices over time.

    Determined by linear regression slope and R-squared analysis
    over a configurable time window (default 5-10 years).
    """

    IMPROVING = "improving"
    """Index values are improving (CPI increasing, corruption decreasing)."""

    STABLE = "stable"
    """Index values show no statistically significant change."""

    DETERIORATING = "deteriorating"
    """Index values are worsening (CPI decreasing, corruption increasing)."""

    VOLATILE = "volatile"
    """Index values show high variance with no clear direction."""


class AlertSeverity(str, Enum):
    """Severity levels for corruption index monitoring alerts.

    Aligned with GreenLang platform alerting standards for
    consistent notification routing and escalation.
    """

    LOW = "low"
    """Low severity: informational, no immediate action required."""

    MEDIUM = "medium"
    """Medium severity: review recommended within 7 days."""

    HIGH = "high"
    """High severity: action required within 48 hours."""

    CRITICAL = "critical"
    """Critical severity: immediate action required, possible reclassification."""


class ComplianceLevel(str, Enum):
    """EUDR due diligence compliance level based on country risk.

    Maps directly to EUDR Articles 10-13 due diligence requirements
    where risk determines the depth and frequency of compliance measures.
    """

    SIMPLIFIED = "simplified"
    """Simplified DD for low-risk countries per Article 13."""

    STANDARD = "standard"
    """Standard DD for standard-risk countries per Article 10."""

    ENHANCED = "enhanced"
    """Enhanced DD for high-risk countries per Article 11."""


class BriberySector(str, Enum):
    """Sectors assessed for bribery and corruption risk relevant to EUDR.

    These sectors are specifically relevant to forest commodity
    supply chains and EUDR compliance monitoring.
    """

    FORESTRY = "forestry"
    """Forestry sector: logging permits, forest management, timber trade."""

    CUSTOMS = "customs"
    """Customs sector: import/export clearance, border controls."""

    AGRICULTURE = "agriculture"
    """Agriculture sector: land permits, crop management, subsidies."""

    MINING = "mining"
    """Mining sector: extraction permits, environmental approvals."""

    EXTRACTION = "extraction"
    """Natural resource extraction sector: oil, gas, minerals."""

    JUDICIARY = "judiciary"
    """Judiciary sector: legal enforcement, court system integrity."""


class CountryClassification(str, Enum):
    """EUDR Article 29 country risk classification.

    The European Commission classifies countries into three tiers
    based on deforestation risk and governance quality. This
    classification determines the level of due diligence required.
    """

    LOW = "low"
    """Low risk: strong governance, low deforestation, simplified DD allowed."""

    STANDARD = "standard"
    """Standard risk: adequate governance, standard DD required."""

    HIGH = "high"
    """High risk: weak governance, high deforestation, enhanced DD mandatory."""


class CorrelationStrength(str, Enum):
    """Strength classification of statistical correlation.

    Based on absolute value of Pearson correlation coefficient (|r|).
    Used to classify the relationship between corruption indices
    and deforestation rates.
    """

    STRONG = "strong"
    """Strong correlation: |r| >= 0.7."""

    MODERATE = "moderate"
    """Moderate correlation: 0.4 <= |r| < 0.7."""

    WEAK = "weak"
    """Weak correlation: 0.2 <= |r| < 0.4."""

    NONE = "none"
    """No meaningful correlation: |r| < 0.2."""


class GovernanceRating(str, Enum):
    """Composite governance quality rating derived from multiple indicators.

    Combines CPI, WGI, and institutional quality scores into a
    single letter-grade assessment for quick reference.
    """

    A = "A"
    """Excellent governance: strong institutions, low corruption."""

    B = "B"
    """Good governance: adequate institutions, moderate corruption."""

    C = "C"
    """Fair governance: some institutional weaknesses."""

    D = "D"
    """Poor governance: significant institutional deficiencies."""

    F = "F"
    """Failed governance: severe institutional collapse, high corruption."""


class DataSource(str, Enum):
    """Data sources for corruption and governance indices.

    Identifies the provenance of index data for audit trail
    and data quality assessment.
    """

    TRANSPARENCY_INTERNATIONAL = "transparency_international"
    """Transparency International CPI data."""

    WORLD_BANK = "world_bank"
    """World Bank Worldwide Governance Indicators data."""

    TRACE_INTERNATIONAL = "trace_international"
    """TRACE International Bribery Risk Matrix."""

    GLOBAL_FOREST_WATCH = "global_forest_watch"
    """Global Forest Watch deforestation data."""

    FAO = "fao"
    """UN FAO Forest Resources Assessment data."""

    ITTO = "itto"
    """International Tropical Timber Organization data."""

    CUSTOM = "custom"
    """Custom/internal data source."""


# ---------------------------------------------------------------------------
# Core Models (10)
# ---------------------------------------------------------------------------


class CPIScore(BaseModel):
    """Transparency International Corruption Perceptions Index score.

    Represents a single country's CPI score for a given year.
    CPI scores range from 0 (highly corrupt) to 100 (very clean).
    Published annually by Transparency International since 1995.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code (e.g., "BR", "ID").
        year: Calendar year of the CPI score.
        score: CPI score 0-100 (Decimal for precision).
        rank: Global rank (1 = least corrupt).
        percentile: Percentile rank (0-100).
        region: Geographic region classification.
        data_source: Source of the CPI data.
        standard_error: Optional standard error of the score estimate.
        sources_count: Number of independent data sources used.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "country_code": "BR",
                    "year": 2024,
                    "score": "38",
                    "rank": 104,
                    "percentile": "42.2",
                    "region": "americas",
                    "data_source": "transparency_international",
                }
            ]
        },
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    year: int = Field(
        ...,
        ge=1995,
        le=2030,
        description="Calendar year of the CPI score",
    )
    score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="CPI score 0-100 (0=most corrupt, 100=cleanest)",
    )
    rank: Optional[int] = Field(
        None,
        ge=1,
        le=250,
        description="Global rank (1 = least corrupt)",
    )
    percentile: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentile rank (0-100)",
    )
    region: Optional[str] = Field(
        None,
        description="Geographic region classification",
    )
    data_source: str = Field(
        "transparency_international",
        description="Source of the CPI data",
    )
    standard_error: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Standard error of the score estimate",
    )
    sources_count: Optional[int] = Field(
        None,
        ge=1,
        description="Number of independent data sources used",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class WGIIndicator(BaseModel):
    """World Bank Worldwide Governance Indicator for a single dimension.

    Represents one of the six WGI dimensions for a country-year pair.
    Estimates range from approximately -2.5 (weakest) to +2.5 (strongest
    governance performance). Standard error and percentile rank provide
    statistical context.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        year: Calendar year of the WGI data.
        dimension: WGI dimension (one of 6).
        estimate: Governance estimate (-2.5 to +2.5).
        std_error: Standard error of the estimate.
        percentile_rank: Percentile rank among all countries (0-100).
        governance_score: Normalized governance score (0-100).
        num_sources: Number of data sources used for this estimate.
        data_source: Source of the WGI data.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    year: int = Field(
        ...,
        ge=1996,
        le=2030,
        description="Calendar year of the WGI data",
    )
    dimension: WGIDimension = Field(
        ...,
        description="WGI dimension (one of 6)",
    )
    estimate: Decimal = Field(
        ...,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Governance estimate (-2.5 weakest to +2.5 strongest)",
    )
    std_error: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Standard error of the estimate",
    )
    percentile_rank: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentile rank among all countries (0-100)",
    )
    governance_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Normalized governance score (0-100)",
    )
    num_sources: Optional[int] = Field(
        None,
        ge=1,
        description="Number of data sources used for this estimate",
    )
    data_source: str = Field(
        "world_bank",
        description="Source of the WGI data",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class BriberyRiskAssessment(BaseModel):
    """Sector-specific bribery risk assessment for a country.

    Evaluates the risk of bribery and corruption in specific economic
    sectors relevant to EUDR commodity supply chains. Combines multiple
    risk indicators with sector-specific weights to produce a composite
    bribery risk score.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        sector: Bribery risk sector being assessed.
        risk_score: Composite bribery risk score (0-100, 100=highest risk).
        risk_level: Classified risk level.
        contributing_factors: Factors contributing to the risk score.
        mitigation_measures: Recommended mitigation measures.
        regulatory_framework_score: Quality of anti-bribery regulations (0-100).
        enforcement_score: Effectiveness of enforcement (0-100).
        data_sources: Sources used for the assessment.
        assessment_date: Date of the assessment.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    sector: BriberySector = Field(
        ...,
        description="Bribery risk sector being assessed",
    )
    risk_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Composite bribery risk score (0-100, 100=highest risk)",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Classified risk level",
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to the risk score",
    )
    mitigation_measures: List[str] = Field(
        default_factory=list,
        description="Recommended mitigation measures",
    )
    regulatory_framework_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Quality of anti-bribery regulations (0-100)",
    )
    enforcement_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Effectiveness of enforcement (0-100)",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Sources used for the assessment",
    )
    assessment_date: date = Field(
        default_factory=lambda: date.today(),
        description="Date of the assessment",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class InstitutionalQualityScore(BaseModel):
    """Composite institutional quality score for a country.

    Aggregates multiple dimensions of institutional quality relevant to
    EUDR compliance: judicial independence, regulatory enforcement
    effectiveness, forest governance quality, and law enforcement
    capacity. Each dimension is scored 0-100 and weighted to produce
    an overall score.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        overall_score: Weighted composite institutional quality score (0-100).
        judicial_independence: Judicial independence score (0-100).
        regulatory_enforcement: Regulatory enforcement effectiveness (0-100).
        forest_governance: Forest governance quality score (0-100).
        law_enforcement_capacity: Law enforcement capacity score (0-100).
        governance_rating: Letter-grade governance rating.
        year: Assessment year.
        data_sources: Sources used for the assessment.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    overall_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Weighted composite institutional quality score (0-100)",
    )
    judicial_independence: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Judicial independence score (0-100)",
    )
    regulatory_enforcement: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Regulatory enforcement effectiveness (0-100)",
    )
    forest_governance: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Forest governance quality score (0-100)",
    )
    law_enforcement_capacity: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Law enforcement capacity score (0-100)",
    )
    governance_rating: Optional[GovernanceRating] = Field(
        None,
        description="Letter-grade governance rating (A-F)",
    )
    year: Optional[int] = Field(
        None,
        ge=2000,
        le=2030,
        description="Assessment year",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Sources used for the assessment",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class TrendAnalysis(BaseModel):
    """Trend analysis results for a corruption or governance index.

    Contains linear regression results, trend direction classification,
    prediction values, and confidence intervals for corruption or
    governance index trajectories over time.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        index_type: Type of index analyzed (cpi, wgi, bribery, institutional).
        dimension: Optional WGI dimension if index_type is wgi.
        direction: Overall trend direction.
        slope: Linear regression slope (change per year).
        intercept: Linear regression intercept.
        r_squared: R-squared goodness of fit (0-1).
        data_points: Number of data points used in analysis.
        start_year: First year in the analysis window.
        end_year: Last year in the analysis window.
        prediction: Predicted value for the next period.
        prediction_year: Year of the prediction.
        confidence_lower: Lower bound of prediction confidence interval.
        confidence_upper: Upper bound of prediction confidence interval.
        is_reversal: Whether a trend reversal was detected.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    index_type: str = Field(
        ...,
        description="Type of index analyzed (cpi, wgi, bribery, institutional)",
    )
    dimension: Optional[WGIDimension] = Field(
        None,
        description="WGI dimension if index_type is wgi",
    )
    direction: TrendDirection = Field(
        ...,
        description="Overall trend direction",
    )
    slope: Decimal = Field(
        ...,
        description="Linear regression slope (change per year)",
    )
    intercept: Decimal = Field(
        ...,
        description="Linear regression intercept",
    )
    r_squared: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="R-squared goodness of fit (0-1)",
    )
    data_points: int = Field(
        ...,
        ge=2,
        description="Number of data points used in analysis",
    )
    start_year: int = Field(
        ...,
        ge=1995,
        le=2030,
        description="First year in the analysis window",
    )
    end_year: int = Field(
        ...,
        ge=1995,
        le=2030,
        description="Last year in the analysis window",
    )
    prediction: Optional[Decimal] = Field(
        None,
        description="Predicted value for the next period",
    )
    prediction_year: Optional[int] = Field(
        None,
        ge=2020,
        le=2035,
        description="Year of the prediction",
    )
    confidence_lower: Optional[Decimal] = Field(
        None,
        description="Lower bound of prediction confidence interval",
    )
    confidence_upper: Optional[Decimal] = Field(
        None,
        description="Upper bound of prediction confidence interval",
    )
    is_reversal: bool = Field(
        False,
        description="Whether a trend reversal was detected",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()

    @model_validator(mode="after")
    def validate_year_range(self) -> "TrendAnalysis":
        """Ensure start_year <= end_year."""
        if self.start_year > self.end_year:
            raise ValueError(
                f"start_year ({self.start_year}) must be <= "
                f"end_year ({self.end_year})"
            )
        return self


class DeforestationCorrelation(BaseModel):
    """Correlation between corruption indices and deforestation rates.

    Captures the statistical relationship between a corruption or
    governance index and deforestation rates for a country, using
    Pearson correlation analysis with significance testing.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        corruption_index: Which corruption index was correlated (cpi, wgi, etc.).
        deforestation_metric: Deforestation metric used (tree_cover_loss_km2, etc.).
        correlation_coefficient: Pearson correlation coefficient (-1 to +1).
        p_value: Statistical significance p-value.
        correlation_strength: Classified correlation strength.
        regression_slope: Regression line slope.
        regression_intercept: Regression line intercept.
        data_points: Number of paired observations.
        start_year: First year in the correlation window.
        end_year: Last year in the correlation window.
        is_significant: Whether the correlation is statistically significant.
        interpretation: Human-readable interpretation of the correlation.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    corruption_index: str = Field(
        ...,
        description="Which corruption index was correlated (cpi, wgi, etc.)",
    )
    deforestation_metric: str = Field(
        "tree_cover_loss_km2",
        description="Deforestation metric used",
    )
    correlation_coefficient: Decimal = Field(
        ...,
        ge=Decimal("-1"),
        le=Decimal("1"),
        description="Pearson correlation coefficient (-1 to +1)",
    )
    p_value: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Statistical significance p-value",
    )
    correlation_strength: CorrelationStrength = Field(
        ...,
        description="Classified correlation strength",
    )
    regression_slope: Optional[Decimal] = Field(
        None,
        description="Regression line slope",
    )
    regression_intercept: Optional[Decimal] = Field(
        None,
        description="Regression line intercept",
    )
    data_points: int = Field(
        ...,
        ge=3,
        description="Number of paired observations",
    )
    start_year: int = Field(
        ...,
        ge=1995,
        le=2030,
        description="First year in the correlation window",
    )
    end_year: int = Field(
        ...,
        ge=1995,
        le=2030,
        description="Last year in the correlation window",
    )
    is_significant: bool = Field(
        ...,
        description="Whether the correlation is statistically significant",
    )
    interpretation: Optional[str] = Field(
        None,
        max_length=1000,
        description="Human-readable interpretation of the correlation",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class Alert(BaseModel):
    """Alert for significant corruption index changes or events.

    Generated when corruption indices change beyond configured thresholds,
    trend reversals are detected, or country reclassifications may be
    warranted. Alerts drive compliance workflow updates.

    Attributes:
        alert_id: Unique alert identifier (UUID).
        country_code: ISO 3166-1 alpha-2 country code.
        alert_type: Type of alert (cpi_change, wgi_change, trend_reversal, etc.).
        severity: Alert severity level.
        description: Human-readable alert description.
        old_value: Previous index value before the change.
        new_value: New index value after the change.
        change_magnitude: Absolute magnitude of the change.
        index_type: Which index triggered the alert (cpi, wgi, etc.).
        dimension: Optional WGI dimension if applicable.
        recommended_action: Recommended follow-up action.
        is_acknowledged: Whether the alert has been acknowledged.
        acknowledged_by: Who acknowledged the alert.
        acknowledged_at: When the alert was acknowledged.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique alert identifier (UUID)",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    alert_type: str = Field(
        ...,
        description="Type of alert (cpi_change, wgi_change, trend_reversal, "
                    "threshold_breach, reclassification)",
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity level",
    )
    description: str = Field(
        ...,
        max_length=2000,
        description="Human-readable alert description",
    )
    old_value: Optional[Decimal] = Field(
        None,
        description="Previous index value before the change",
    )
    new_value: Optional[Decimal] = Field(
        None,
        description="New index value after the change",
    )
    change_magnitude: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        description="Absolute magnitude of the change",
    )
    index_type: Optional[str] = Field(
        None,
        description="Which index triggered the alert (cpi, wgi, etc.)",
    )
    dimension: Optional[WGIDimension] = Field(
        None,
        description="WGI dimension if applicable",
    )
    recommended_action: Optional[str] = Field(
        None,
        max_length=1000,
        description="Recommended follow-up action",
    )
    is_acknowledged: bool = Field(
        False,
        description="Whether the alert has been acknowledged",
    )
    acknowledged_by: Optional[str] = Field(
        None,
        description="Who acknowledged the alert",
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="When the alert was acknowledged",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class ComplianceImpact(BaseModel):
    """Compliance impact assessment mapping corruption to EUDR requirements.

    Maps CPI and WGI scores to EUDR Article 29 country classifications
    and determines the appropriate due diligence level, cost implications,
    and audit frequency for a given country.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        cpi_score: Current CPI score (0-100).
        wgi_composite: Composite WGI score (-2.5 to +2.5).
        article_29_classification: EUDR Article 29 classification.
        dd_level: Required due diligence level.
        risk_adjustment: Risk adjustment factor (0.5-2.0).
        estimated_dd_cost_eur: Estimated DD cost in EUR.
        audit_frequency_months: Recommended audit interval in months.
        risk_factors: Contributing risk factors.
        mitigating_factors: Factors that reduce risk.
        assessment_confidence: Confidence in the assessment (0-1).
        effective_date: Date from which this assessment is effective.
        expiry_date: Date after which reassessment is required.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    cpi_score: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Current CPI score (0-100)",
    )
    wgi_composite: Optional[Decimal] = Field(
        None,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Composite WGI score (-2.5 to +2.5)",
    )
    article_29_classification: CountryClassification = Field(
        ...,
        description="EUDR Article 29 country classification",
    )
    dd_level: ComplianceLevel = Field(
        ...,
        description="Required due diligence level",
    )
    risk_adjustment: Decimal = Field(
        Decimal("1.0"),
        ge=Decimal("0.5"),
        le=Decimal("2.0"),
        description="Risk adjustment factor (0.5-2.0)",
    )
    estimated_dd_cost_eur: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated DD cost in EUR",
    )
    audit_frequency_months: Optional[int] = Field(
        None,
        ge=1,
        le=60,
        description="Recommended audit interval in months",
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Contributing risk factors",
    )
    mitigating_factors: List[str] = Field(
        default_factory=list,
        description="Factors that reduce risk",
    )
    assessment_confidence: Optional[Decimal] = Field(
        None,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence in the assessment (0-1)",
    )
    effective_date: Optional[date] = Field(
        None,
        description="Date from which this assessment is effective",
    )
    expiry_date: Optional[date] = Field(
        None,
        description="Date after which reassessment is required",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class CountryProfile(BaseModel):
    """Comprehensive country governance and corruption profile.

    Aggregates CPI scores, WGI indicators, bribery risk assessments,
    institutional quality scores, trend analyses, and compliance impact
    into a single comprehensive country profile for EUDR compliance
    decision-making.

    Attributes:
        country_code: ISO 3166-1 alpha-2 country code.
        country_name: Full country name.
        latest_cpi: Most recent CPI score.
        cpi_trend: CPI trend direction.
        wgi_indicators: Latest WGI indicators per dimension.
        wgi_composite: Composite WGI score.
        bribery_assessments: Sector-specific bribery risk assessments.
        institutional_quality: Latest institutional quality score.
        article_29_classification: EUDR Article 29 classification.
        dd_level: Required due diligence level.
        governance_rating: Composite governance rating.
        risk_level: Overall risk level.
        active_alerts: Number of active (unacknowledged) alerts.
        deforestation_correlation: Correlation with deforestation.
        last_assessment_date: Date of last comprehensive assessment.
        next_reassessment_date: Date of next required reassessment.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
        updated_at: Record last update timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    country_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code",
    )
    country_name: str = Field(
        ...,
        max_length=200,
        description="Full country name",
    )
    latest_cpi: Optional[CPIScore] = Field(
        None,
        description="Most recent CPI score",
    )
    cpi_trend: Optional[TrendDirection] = Field(
        None,
        description="CPI trend direction",
    )
    wgi_indicators: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Latest WGI indicators per dimension (dimension -> estimate)",
    )
    wgi_composite: Optional[Decimal] = Field(
        None,
        ge=Decimal("-2.5"),
        le=Decimal("2.5"),
        description="Composite WGI score",
    )
    bribery_assessments: List[BriberyRiskAssessment] = Field(
        default_factory=list,
        description="Sector-specific bribery risk assessments",
    )
    institutional_quality: Optional[InstitutionalQualityScore] = Field(
        None,
        description="Latest institutional quality score",
    )
    article_29_classification: Optional[CountryClassification] = Field(
        None,
        description="EUDR Article 29 classification",
    )
    dd_level: Optional[ComplianceLevel] = Field(
        None,
        description="Required due diligence level",
    )
    governance_rating: Optional[GovernanceRating] = Field(
        None,
        description="Composite governance rating (A-F)",
    )
    risk_level: Optional[RiskLevel] = Field(
        None,
        description="Overall risk level",
    )
    active_alerts: int = Field(
        0,
        ge=0,
        description="Number of active (unacknowledged) alerts",
    )
    deforestation_correlation: Optional[CorrelationStrength] = Field(
        None,
        description="Strength of corruption-deforestation correlation",
    )
    last_assessment_date: Optional[date] = Field(
        None,
        description="Date of last comprehensive assessment",
    )
    next_reassessment_date: Optional[date] = Field(
        None,
        description="Date of next required reassessment",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Record last update timestamp (UTC)",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country_code is uppercase."""
        return v.upper()


class AuditLogEntry(BaseModel):
    """Audit log entry for corruption index monitor operations.

    Records all significant operations for EUDR Article 31 compliance
    with five-year retention requirement.

    Attributes:
        entry_id: Unique entry identifier (UUID).
        operation: Operation performed.
        entity_type: Type of entity affected.
        entity_id: Identifier of the affected entity.
        actor: User or system identifier.
        details: Operation details.
        provenance_hash: SHA-256 hash for audit trail.
        created_at: Record creation timestamp.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
    )

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique entry identifier (UUID)",
    )
    operation: str = Field(
        ...,
        description="Operation performed",
    )
    entity_type: str = Field(
        ...,
        description="Type of entity affected",
    )
    entity_id: str = Field(
        ...,
        description="Identifier of the affected entity",
    )
    actor: str = Field(
        "system",
        description="User or system identifier",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operation details",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Record creation timestamp (UTC)",
    )


# ---------------------------------------------------------------------------
# Request Models (10)
# ---------------------------------------------------------------------------


class QueryCPIRequest(BaseModel):
    """Request model for querying CPI scores."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    year: Optional[int] = Field(None, ge=1995, le=2030, description="Specific year")
    start_year: Optional[int] = Field(None, ge=1995, le=2030, description="Range start")
    end_year: Optional[int] = Field(None, ge=1995, le=2030, description="Range end")
    include_regional: bool = Field(False, description="Include regional comparison")


class QueryWGIRequest(BaseModel):
    """Request model for querying WGI indicators."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    dimension: Optional[WGIDimension] = Field(None, description="Specific WGI dimension")
    year: Optional[int] = Field(None, ge=1996, le=2030, description="Specific year")
    start_year: Optional[int] = Field(None, ge=1996, le=2030, description="Range start")
    end_year: Optional[int] = Field(None, ge=1996, le=2030, description="Range end")
    include_composite: bool = Field(True, description="Include composite WGI score")


class AssessBriberyRiskRequest(BaseModel):
    """Request model for bribery risk assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    sectors: Optional[List[BriberySector]] = Field(None, description="Sectors to assess")
    include_mitigation: bool = Field(True, description="Include mitigation recommendations")


class EvaluateInstitutionalQualityRequest(BaseModel):
    """Request model for institutional quality evaluation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    year: Optional[int] = Field(None, ge=2000, le=2030, description="Assessment year")
    include_breakdown: bool = Field(True, description="Include dimension breakdown")


class AnalyzeTrendRequest(BaseModel):
    """Request model for trend analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    index_type: str = Field("cpi", description="Index type (cpi, wgi, bribery)")
    dimension: Optional[WGIDimension] = Field(None, description="WGI dimension")
    min_years: Optional[int] = Field(None, ge=2, description="Override min years")
    include_prediction: bool = Field(True, description="Include prediction")


class AnalyzeCorrelationRequest(BaseModel):
    """Request model for deforestation-corruption correlation analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    corruption_index: str = Field("cpi", description="Corruption index to correlate")
    deforestation_metric: str = Field(
        "tree_cover_loss_km2", description="Deforestation metric"
    )
    min_data_points: Optional[int] = Field(None, ge=3, description="Override min points")


class GenerateAlertRequest(BaseModel):
    """Request model for alert generation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    alert_type: str = Field(
        ..., description="Alert type (cpi_change, wgi_change, trend_reversal)"
    )
    old_value: Optional[Decimal] = Field(None, description="Previous value")
    new_value: Optional[Decimal] = Field(None, description="New value")
    index_type: Optional[str] = Field(None, description="Index type")
    dimension: Optional[WGIDimension] = Field(None, description="WGI dimension")


class AssessComplianceImpactRequest(BaseModel):
    """Request model for compliance impact assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    include_cost_estimate: bool = Field(True, description="Include DD cost estimate")
    include_audit_frequency: bool = Field(True, description="Include audit frequency")


class BuildCountryProfileRequest(BaseModel):
    """Request model for building comprehensive country profile."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, max_length=3, description="Country code")
    include_trends: bool = Field(True, description="Include trend analysis")
    include_correlations: bool = Field(True, description="Include correlation analysis")
    include_bribery: bool = Field(True, description="Include bribery assessments")
    include_alerts: bool = Field(True, description="Include active alerts")


class HealthCheckRequest(BaseModel):
    """Request model for health check."""

    model_config = ConfigDict(str_strip_whitespace=True)

    include_details: bool = Field(False, description="Include detailed diagnostics")


# ---------------------------------------------------------------------------
# Response Models (10)
# ---------------------------------------------------------------------------


class CPIScoreResponse(BaseModel):
    """Response model for CPI score queries."""

    model_config = ConfigDict(str_strip_whitespace=True)

    scores: List[CPIScore] = Field(default_factory=list, description="CPI scores")
    count: int = Field(0, ge=0, description="Number of results")
    regional_average: Optional[Decimal] = Field(None, description="Regional average")
    global_average: Optional[Decimal] = Field(None, description="Global average")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class WGIIndicatorResponse(BaseModel):
    """Response model for WGI indicator queries."""

    model_config = ConfigDict(str_strip_whitespace=True)

    indicators: List[WGIIndicator] = Field(default_factory=list, description="WGI indicators")
    composite_score: Optional[Decimal] = Field(None, description="Composite WGI score")
    count: int = Field(0, ge=0, description="Number of results")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class BriberyRiskResponse(BaseModel):
    """Response model for bribery risk assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessments: List[BriberyRiskAssessment] = Field(
        default_factory=list, description="Bribery risk assessments"
    )
    composite_risk_score: Optional[Decimal] = Field(None, description="Composite score")
    highest_risk_sector: Optional[str] = Field(None, description="Highest risk sector")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class InstitutionalQualityResponse(BaseModel):
    """Response model for institutional quality evaluation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    score: Optional[InstitutionalQualityScore] = Field(None, description="IQ score")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class TrendAnalysisResponse(BaseModel):
    """Response model for trend analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    analysis: Optional[TrendAnalysis] = Field(None, description="Trend analysis result")
    data_sufficient: bool = Field(True, description="Whether enough data was available")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class CorrelationResponse(BaseModel):
    """Response model for deforestation-corruption correlation analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    correlation: Optional[DeforestationCorrelation] = Field(
        None, description="Correlation result"
    )
    data_sufficient: bool = Field(True, description="Whether enough data was available")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class AlertResponse(BaseModel):
    """Response model for alert operations."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alerts: List[Alert] = Field(default_factory=list, description="Generated alerts")
    count: int = Field(0, ge=0, description="Number of alerts")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class ComplianceImpactResponse(BaseModel):
    """Response model for compliance impact assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    impact: Optional[ComplianceImpact] = Field(None, description="Compliance impact")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class CountryProfileResponse(BaseModel):
    """Response model for comprehensive country profile."""

    model_config = ConfigDict(str_strip_whitespace=True)

    profile: Optional[CountryProfile] = Field(None, description="Country profile")
    processing_time_ms: float = Field(0.0, description="Processing time in ms")
    provenance_hash: Optional[str] = Field(None, description="Response provenance hash")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str = Field("healthy", description="Service status")
    version: str = Field(VERSION, description="Service version")
    agent_id: str = Field("GL-EUDR-CIM-019", description="Agent identifier")
    uptime_seconds: float = Field(0.0, description="Service uptime in seconds")
    monitored_countries: int = Field(0, ge=0, description="Countries being monitored")
    active_alerts: int = Field(0, ge=0, description="Active alert count")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed diagnostics")
