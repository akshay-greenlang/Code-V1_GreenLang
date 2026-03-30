# -*- coding: utf-8 -*-
"""
Country Risk Evaluator API Schemas - AGENT-EUDR-016

Pydantic v2 request/response schemas for the Country Risk Evaluator
REST API. All schemas are designed for OpenAPI/Swagger documentation
with comprehensive field descriptions, validation constraints, and
JSON schema examples.

Schema Groups:
    - Common: PaginationSchema, ErrorSchema, HealthSchema, SuccessSchema
    - Country: AssessCountrySchema, CountryRiskSchema, CountryListSchema,
      CountryCompareSchema, TrendSchema, TrendPointSchema
    - Commodity: AnalyzeCommoditySchema, CommodityProfileSchema,
      CommodityListSchema, RiskMatrixSchema, RiskMatrixEntrySchema,
      CorrelationSchema, CorrelationEntrySchema
    - Hotspot: DetectHotspotSchema, HotspotSchema, HotspotListSchema,
      AlertSchema, AlertListSchema, ClusteringSchema, ClusteringResultSchema
    - Governance: EvaluateGovernanceSchema, GovernanceIndexSchema,
      GovernanceListSchema, GovernanceCompareSchema
    - Due Diligence: ClassifySchema, ClassificationSchema,
      ClassificationListSchema, CostEstimateSchema, CostEstimateResultSchema,
      AuditFrequencySchema, AuditFrequencyResultSchema
    - Trade Flow: AnalyzeFlowSchema, TradeFlowSchema, TradeFlowListSchema,
      RouteSchema, RouteListSchema, ReExportRiskSchema, ReExportRiskResultSchema
    - Report: GenerateReportSchema, ReportSchema, ReportListSchema,
      DownloadSchema, ExecutiveSummarySchema, ExecutiveSummaryResultSchema
    - Regulatory: TrackUpdateSchema, UpdateSchema, UpdateListSchema,
      ImpactAssessmentSchema, ImpactAssessmentResultSchema

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator
from greenlang.schemas import GreenLangBase


# =============================================================================
# Common Schemas
# =============================================================================


class PaginationSchema(GreenLangBase):
    """Standard pagination metadata for list responses."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total: int = Field(
        default=0, ge=0,
        description="Total number of records matching the query.",
    )
    limit: int = Field(
        default=50, ge=1, le=500,
        description="Maximum number of records per page.",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Number of records skipped from the start.",
    )
    has_more: bool = Field(
        default=False,
        description="Whether additional pages are available.",
    )


class ErrorSchema(GreenLangBase):
    """Structured error response schema per GreenLang API standards."""

    model_config = ConfigDict(str_strip_whitespace=True)

    error: str = Field(
        ...,
        description="Machine-readable error code (e.g., 'validation_error').",
    )
    message: str = Field(
        ...,
        description="Human-readable error description.",
    )
    details: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Additional error details (field-level validation errors).",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing and support.",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the error occurred (UTC).",
    )


class HealthSchema(GreenLangBase):
    """Health check response for the Country Risk Evaluator service."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str = Field(
        default="healthy",
        description="Service health status (healthy/degraded/unhealthy).",
    )
    version: str = Field(
        default="1.0.0",
        description="Service version.",
    )
    agent_id: str = Field(
        default="GL-EUDR-CRE-016",
        description="Agent identifier.",
    )
    countries_assessed: int = Field(
        default=0, ge=0,
        description="Number of countries with active risk assessments.",
    )
    high_risk_countries: int = Field(
        default=0, ge=0,
        description="Number of countries classified as high risk.",
    )
    active_hotspots: int = Field(
        default=0, ge=0,
        description="Number of active deforestation hotspots.",
    )
    database_connected: bool = Field(
        default=False,
        description="Whether database connection is healthy.",
    )
    cache_connected: bool = Field(
        default=False,
        description="Whether Redis cache connection is healthy.",
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Service uptime in seconds.",
    )
    checked_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of health check (UTC).",
    )


class SuccessSchema(GreenLangBase):
    """Generic success response."""

    model_config = ConfigDict(str_strip_whitespace=True)

    success: bool = Field(
        default=True,
        description="Whether the operation succeeded.",
    )
    message: str = Field(
        default="Operation completed successfully.",
        description="Human-readable success message.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing.",
    )


# =============================================================================
# Country Schemas
# =============================================================================


class AssessCountrySchema(GreenLangBase):
    """Request schema for running a country risk assessment.

    Accepts one or more ISO 3166-1 alpha-2 country codes and optional
    custom factor weights for the 6-factor composite score. Custom
    weights must sum to 1.0 if provided.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_codes": ["BR", "ID", "CO"],
                "custom_weights": None,
                "include_trend": True,
                "include_regional_context": True,
            }
        },
    )

    country_codes: List[str] = Field(
        ..., min_length=1, max_length=50,
        description=(
            "List of ISO 3166-1 alpha-2 country codes to assess. "
            "Maximum 50 countries per request."
        ),
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Optional custom factor weights. Keys: deforestation_rate, "
            "governance_index, enforcement_score, corruption_index, "
            "forest_law_compliance, historical_trend. Values must sum to 1.0."
        ),
    )
    include_trend: bool = Field(
        default=True,
        description="Whether to include historical trend analysis.",
    )
    include_regional_context: bool = Field(
        default=True,
        description=(
            "Whether to include regional comparison with "
            "neighboring countries."
        ),
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Normalize country codes to uppercase."""
        return [c.upper().strip() for c in v]


class RiskFactorSchema(GreenLangBase):
    """Individual risk factor within a composite score."""

    model_config = ConfigDict(str_strip_whitespace=True)

    factor_name: str = Field(
        ...,
        description="Risk factor name (e.g., 'deforestation_rate').",
    )
    weight: float = Field(
        ..., ge=0.0, le=1.0,
        description="Factor weight in composite calculation.",
    )
    raw_value: float = Field(
        ...,
        description="Raw factor value before normalization.",
    )
    normalized_value: float = Field(
        ..., ge=0.0, le=100.0,
        description="Normalized factor value (0-100).",
    )
    data_source: str = Field(
        default="",
        description="Primary data source for this factor.",
    )
    data_date: Optional[datetime] = Field(
        default=None,
        description="Publication date of the source data.",
    )


class CountryRiskSchema(GreenLangBase):
    """Response schema for a single country risk assessment.

    Contains the composite 6-factor risk score (0-100), classification
    (low/standard/high), trend direction, confidence level, and full
    factor breakdown with data source attribution.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "assessment_id": "cra-550e8400-e29b-41d4-a716-446655440000",
                "country_code": "BR",
                "country_name": "Brazil",
                "risk_level": "high",
                "risk_score": 72.5,
                "confidence": "high",
                "trend": "deteriorating",
                "percentile_rank": 85.0,
                "ec_benchmark_aligned": True,
            }
        },
    )

    assessment_id: str = Field(
        ...,
        description="Unique assessment identifier (UUID-based).",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    country_name: str = Field(
        ...,
        description="Full country name.",
    )
    risk_level: str = Field(
        ...,
        description="Three-tier risk classification: low, standard, high.",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Composite risk score (0-100, higher = riskier).",
    )
    composite_factors: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Individual factor scores. Keys: deforestation_rate, "
            "governance_index, enforcement_score, corruption_index, "
            "forest_law_compliance, historical_trend."
        ),
    )
    factor_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights applied to each factor (sum to 1.0).",
    )
    risk_factors: List[RiskFactorSchema] = Field(
        default_factory=list,
        description="Detailed risk factor breakdown.",
    )
    confidence: str = Field(
        default="medium",
        description=(
            "Confidence level: very_low, low, medium, high, very_high."
        ),
    )
    trend: str = Field(
        default="stable",
        description=(
            "Risk score trend: improving, stable, deteriorating, "
            "insufficient_data."
        ),
    )
    percentile_rank: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Percentile rank among all assessed countries.",
    )
    regional_average: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Average risk score of neighboring countries.",
    )
    ec_benchmark_aligned: bool = Field(
        default=True,
        description=(
            "Whether agent classification matches EC benchmark."
        ),
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="List of data sources used for this assessment.",
    )
    assessed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )


class CountryListSchema(GreenLangBase):
    """Paginated list response for country risk assessments."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessments: List[CountryRiskSchema] = Field(
        default_factory=list,
        description="List of country risk assessments.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class CountryCompareSchema(GreenLangBase):
    """Request schema for comparing multiple countries side-by-side.

    Requires at least 2 country codes. Optionally filter by commodities
    and include governance comparison.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_codes": ["BR", "ID", "CO", "GH"],
                "commodities": ["cocoa", "coffee"],
                "include_governance": True,
            }
        },
    )

    country_codes: List[str] = Field(
        ..., min_length=2, max_length=20,
        description="Countries to compare (minimum 2, maximum 20).",
    )
    commodities: Optional[List[str]] = Field(
        default=None,
        description=(
            "Commodities to compare across (None = all 7 EUDR "
            "commodities)."
        ),
    )
    include_governance: bool = Field(
        default=True,
        description="Whether to include governance comparison.",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Normalize country codes to uppercase."""
        return [c.upper().strip() for c in v]


class CountryCompareResultSchema(GreenLangBase):
    """Response schema for country comparison."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessments: List[CountryRiskSchema] = Field(
        default_factory=list,
        description="Assessments for compared countries.",
    )
    ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Countries ranked by risk score. Each entry: "
            "{country_code, country_name, risk_score, risk_level, rank}."
        ),
    )
    governance_comparison: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Governance scores side-by-side, if requested.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class TrendPointSchema(GreenLangBase):
    """Single data point in a risk trend time series."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessed_at: datetime = Field(
        ...,
        description="Timestamp of assessment (UTC).",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Risk score at this point.",
    )
    risk_level: str = Field(
        ...,
        description="Risk classification at this point.",
    )
    change_reason: str = Field(
        default="",
        description="Reason for score change, if any.",
    )
    previous_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Previous risk score before this change.",
    )


class TrendSchema(GreenLangBase):
    """Response schema for country risk trend analysis.

    Contains historical risk scores with trend direction over
    the configured analysis window (default 5 years).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "BR",
                "trend_direction": "deteriorating",
                "years_analyzed": 5,
                "data_points": 20,
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    country_name: str = Field(
        default="",
        description="Full country name.",
    )
    trend_direction: str = Field(
        default="stable",
        description=(
            "Overall trend direction: improving, stable, "
            "deteriorating, insufficient_data."
        ),
    )
    years_analyzed: int = Field(
        default=5, ge=1,
        description="Number of years analyzed.",
    )
    history: List[TrendPointSchema] = Field(
        default_factory=list,
        description="Historical risk score records.",
    )
    data_points: int = Field(
        default=0, ge=0,
        description="Total number of data points in the trend.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Commodity Schemas
# =============================================================================


class AnalyzeCommoditySchema(GreenLangBase):
    """Request schema for commodity-specific risk analysis.

    Analyzes one or more EUDR commodities for a specific country,
    with optional seasonal and certification analysis.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "ID",
                "commodities": ["oil_palm", "rubber"],
                "include_seasonal": True,
                "include_certifications": True,
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodities: Optional[List[str]] = Field(
        default=None,
        description=(
            "Commodities to analyze. Valid values: cattle, cocoa, "
            "coffee, oil_palm, rubber, soya, wood. None = all 7."
        ),
    )
    include_seasonal: bool = Field(
        default=True,
        description="Whether to include seasonal risk variation analysis.",
    )
    include_certifications: bool = Field(
        default=True,
        description="Whether to include certification effectiveness.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class CommodityProfileSchema(GreenLangBase):
    """Response schema for a single commodity risk profile.

    Contains commodity-specific risk score, deforestation correlation,
    production data, certification effectiveness, and seasonal factors.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "profile_id": "crp-550e8400-e29b-41d4-a716-446655440000",
                "country_code": "ID",
                "commodity_type": "oil_palm",
                "risk_score": 78.5,
                "risk_level": "high",
                "deforestation_correlation": 0.85,
            }
        },
    )

    profile_id: str = Field(
        ...,
        description="Unique profile identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: str = Field(
        ...,
        description="EUDR commodity type.",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Commodity-specific risk score (0-100).",
    )
    risk_level: str = Field(
        ...,
        description="Commodity-specific risk classification.",
    )
    production_volume_tonnes: Optional[float] = Field(
        default=None, ge=0.0,
        description="Annual production volume in tonnes.",
    )
    deforestation_correlation: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "Correlation coefficient between commodity production "
            "expansion and forest loss (0.0-1.0)."
        ),
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="Active certification schemes.",
    )
    certification_effectiveness: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Certification scheme effectiveness score (0-100).",
    )
    seasonal_factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Monthly seasonal risk multipliers.",
    )
    supply_chain_complexity: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Supply chain complexity score (1-10).",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used for this profile.",
    )
    assessed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class CommodityListSchema(GreenLangBase):
    """Paginated list response for commodity risk profiles."""

    model_config = ConfigDict(str_strip_whitespace=True)

    profiles: List[CommodityProfileSchema] = Field(
        default_factory=list,
        description="List of commodity risk profiles.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of profiles.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class RiskMatrixEntrySchema(GreenLangBase):
    """Single entry in the country-commodity risk matrix."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    country_name: str = Field(
        default="",
        description="Full country name.",
    )
    commodity_scores: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-commodity risk scores. Keys are commodity types, "
            "values are 0-100 risk scores."
        ),
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Average risk score across commodities.",
    )
    risk_level: str = Field(
        default="standard",
        description="Overall risk classification.",
    )


class RiskMatrixSchema(GreenLangBase):
    """Response schema for the country-commodity risk matrix.

    A heatmap-style matrix of risk scores for multiple countries
    and commodities, suitable for dashboard rendering.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    matrix: List[RiskMatrixEntrySchema] = Field(
        default_factory=list,
        description="Risk matrix entries (one per country).",
    )
    countries_count: int = Field(
        default=0, ge=0,
        description="Number of countries in the matrix.",
    )
    commodities_count: int = Field(
        default=0, ge=0,
        description="Number of commodities in the matrix.",
    )
    commodities: List[str] = Field(
        default_factory=list,
        description="Commodity types included in the matrix.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class CorrelationEntrySchema(GreenLangBase):
    """Single correlation pair between two commodities."""

    model_config = ConfigDict(str_strip_whitespace=True)

    commodity_a: str = Field(
        ...,
        description="First commodity type.",
    )
    commodity_b: str = Field(
        ...,
        description="Second commodity type.",
    )
    correlation_coefficient: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Pearson correlation coefficient.",
    )
    p_value: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Statistical significance p-value.",
    )
    sample_size: int = Field(
        default=0, ge=0,
        description="Number of countries in the sample.",
    )


class CorrelationSchema(GreenLangBase):
    """Response schema for cross-commodity risk correlations.

    Shows how risk scores correlate across different EUDR commodities,
    identifying which commodities tend to be co-risky.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    correlations: List[CorrelationEntrySchema] = Field(
        default_factory=list,
        description="Pairwise commodity correlation entries.",
    )
    total_pairs: int = Field(
        default=0, ge=0,
        description="Total number of commodity pairs analyzed.",
    )
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Correlation threshold applied for filtering.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Hotspot Schemas
# =============================================================================


class DetectHotspotSchema(GreenLangBase):
    """Request schema for deforestation hotspot detection.

    Triggers spatial analysis to detect sub-national deforestation
    hotspots using DBSCAN-like clustering of GFW alerts and Hansen
    tree cover loss data.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "BR",
                "min_severity": "medium",
                "include_fire_correlation": True,
                "include_protected_areas": True,
                "temporal_window_months": 12,
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    min_severity: str = Field(
        default="low",
        description=(
            "Minimum severity to include: low, medium, high, critical."
        ),
    )
    include_fire_correlation: bool = Field(
        default=True,
        description=(
            "Whether to include FIRMS/VIIRS fire alert correlation."
        ),
    )
    include_protected_areas: bool = Field(
        default=True,
        description="Whether to include protected area proximity analysis.",
    )
    temporal_window_months: int = Field(
        default=12, ge=1, le=60,
        description="Temporal window for alert clustering (months).",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class HotspotSchema(GreenLangBase):
    """Response schema for a single deforestation hotspot.

    Contains geographic coordinates, severity, drivers, tree cover
    loss, fire correlation, and protected area proximity data.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "hotspot_id": "dhs-550e8400-e29b-41d4-a716-446655440000",
                "country_code": "BR",
                "region": "Para",
                "latitude": -3.4168,
                "longitude": -52.2053,
                "area_km2": 125.0,
                "severity": "critical",
            }
        },
    )

    hotspot_id: str = Field(
        ...,
        description="Unique hotspot identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    region: str = Field(
        ...,
        description="Sub-national region/province/state name.",
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Centroid latitude (WGS84).",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Centroid longitude (WGS84).",
    )
    area_km2: float = Field(
        ..., gt=0.0,
        description="Hotspot area in square kilometers.",
    )
    severity: str = Field(
        ...,
        description="Severity classification: low, medium, high, critical.",
    )
    drivers: List[str] = Field(
        default_factory=list,
        description=(
            "Primary deforestation drivers: agriculture, logging, "
            "mining, infrastructure, fire, urbanization."
        ),
    )
    tree_cover_loss_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Tree cover loss since EUDR cutoff (Dec 31, 2020).",
    )
    fire_correlation: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Correlation with FIRMS/VIIRS fire alerts.",
    )
    protected_area_overlap_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Percentage overlapping protected areas.",
    )
    protected_area_distance_km: Optional[float] = Field(
        default=None, ge=0.0,
        description="Distance to nearest protected area boundary (km).",
    )
    indigenous_territory_overlap: bool = Field(
        default=False,
        description="Whether hotspot overlaps indigenous territory.",
    )
    trend: str = Field(
        default="stable",
        description="Deforestation trend direction.",
    )
    alert_count: int = Field(
        default=0, ge=0,
        description="Number of deforestation alerts in cluster.",
    )
    detected_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of detection (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class HotspotListSchema(GreenLangBase):
    """Paginated list response for deforestation hotspots."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hotspots: List[HotspotSchema] = Field(
        default_factory=list,
        description="List of detected hotspots.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class AlertSchema(GreenLangBase):
    """Active deforestation alert schema."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alert_id: str = Field(
        ...,
        description="Unique alert identifier.",
    )
    hotspot_id: str = Field(
        ...,
        description="Associated hotspot identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    region: str = Field(
        default="",
        description="Sub-national region.",
    )
    severity: str = Field(
        ...,
        description="Alert severity: low, medium, high, critical.",
    )
    message: str = Field(
        default="",
        description="Human-readable alert description.",
    )
    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Alert latitude (WGS84).",
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Alert longitude (WGS84).",
    )
    deforestation_rate: Optional[float] = Field(
        default=None, ge=0.0,
        description="Annual deforestation rate (%).",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Alert creation timestamp (UTC).",
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether the alert has been acknowledged.",
    )


class AlertListSchema(GreenLangBase):
    """List response for active deforestation alerts."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alerts: List[AlertSchema] = Field(
        default_factory=list,
        description="List of active alerts.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of active alerts.",
    )
    critical_count: int = Field(
        default=0, ge=0,
        description="Number of critical-severity alerts.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )


class ClusteringSchema(GreenLangBase):
    """Request schema for spatial clustering analysis.

    Configures DBSCAN parameters for hotspot clustering.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "BR",
                "min_points": 10,
                "radius_km": 5.0,
                "temporal_window_months": 12,
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    min_points: int = Field(
        default=10, ge=1, le=1000,
        description="Minimum cluster points (DBSCAN min_samples).",
    )
    radius_km: float = Field(
        default=5.0, gt=0.0, le=500.0,
        description="Clustering radius in km (DBSCAN epsilon).",
    )
    temporal_window_months: int = Field(
        default=12, ge=1, le=60,
        description="Temporal window for clustering (months).",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class ClusteringResultSchema(GreenLangBase):
    """Response schema for spatial clustering analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    clusters: List[HotspotSchema] = Field(
        default_factory=list,
        description="Identified clusters as hotspot objects.",
    )
    total_clusters: int = Field(
        default=0, ge=0,
        description="Total number of clusters identified.",
    )
    noise_points: int = Field(
        default=0, ge=0,
        description="Number of alerts not assigned to any cluster.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="DBSCAN parameters used for clustering.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Governance Schemas
# =============================================================================


class EvaluateGovernanceSchema(GreenLangBase):
    """Request schema for governance quality evaluation.

    Evaluates governance using World Bank WGI, Transparency International
    CPI, and FAO/ITTO forest governance frameworks.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_codes": ["BR", "ID", "GH"],
                "include_legal_framework": True,
                "include_enforcement": True,
                "include_indigenous_rights": True,
            }
        },
    )

    country_codes: List[str] = Field(
        ..., min_length=1, max_length=50,
        description="Countries to evaluate (ISO 3166-1 alpha-2).",
    )
    include_legal_framework: bool = Field(
        default=True,
        description="Whether to include legal framework scoring.",
    )
    include_enforcement: bool = Field(
        default=True,
        description="Whether to include enforcement analysis.",
    )
    include_indigenous_rights: bool = Field(
        default=True,
        description="Whether to include indigenous rights assessment.",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Normalize country codes to uppercase."""
        return [c.upper().strip() for c in v]


class GovernanceIndexSchema(GreenLangBase):
    """Response schema for a single governance index.

    Contains composite governance score (0-100, higher = better),
    WGI indicators, CPI score, forest governance, legal framework,
    and enforcement effectiveness.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "index_id": "gix-550e8400-e29b-41d4-a716-446655440000",
                "country_code": "BR",
                "overall_score": 48.5,
                "cpi_score": 38.0,
            }
        },
    )

    index_id: str = Field(
        ...,
        description="Unique governance index identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    country_name: str = Field(
        default="",
        description="Full country name.",
    )
    overall_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Composite governance score (0-100, higher = better).",
    )
    indicators: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Individual WGI indicator scores: rule_of_law, "
            "regulatory_quality, control_of_corruption, "
            "government_effectiveness, voice_accountability, "
            "political_stability."
        ),
    )
    cpi_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Transparency International CPI score (0-100).",
    )
    forest_governance_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="FAO/ITTO forest governance score (0-100).",
    )
    legal_framework_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Legal framework strength score (0-100).",
    )
    enforcement_effectiveness: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Environmental enforcement effectiveness (0-100).",
    )
    indigenous_rights_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Indigenous peoples rights recognition (0-100).",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used for this index.",
    )
    assessed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class GovernanceListSchema(GreenLangBase):
    """Paginated list response for governance indices."""

    model_config = ConfigDict(str_strip_whitespace=True)

    indices: List[GovernanceIndexSchema] = Field(
        default_factory=list,
        description="List of governance indices.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class GovernanceCompareSchema(GreenLangBase):
    """Response schema for governance comparison across countries."""

    model_config = ConfigDict(str_strip_whitespace=True)

    indices: List[GovernanceIndexSchema] = Field(
        default_factory=list,
        description="Governance indices for compared countries.",
    )
    ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Countries ranked by governance score (descending). "
            "Each entry: {country_code, country_name, overall_score, rank}."
        ),
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Due Diligence Schemas
# =============================================================================


class ClassifySchema(GreenLangBase):
    """Request schema for due diligence classification.

    Determines the required EUDR due diligence level (simplified,
    standard, enhanced) based on country risk score, commodity type,
    and certification credits.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "BR",
                "commodity_type": "soya",
                "region": "Mato Grosso",
                "certification_schemes": ["iscc"],
                "include_cost_estimate": True,
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: Optional[str] = Field(
        default=None,
        description=(
            "EUDR commodity type. None = country-level classification."
        ),
    )
    region: Optional[str] = Field(
        default=None,
        description="Sub-national region for override checks.",
    )
    certification_schemes: List[str] = Field(
        default_factory=list,
        description=(
            "Active certification schemes for credit: fsc, pefc, rspo, "
            "rainforest_alliance, fairtrade, organic, bonsucro, iscc."
        ),
    )
    include_cost_estimate: bool = Field(
        default=True,
        description="Whether to include cost estimation.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class ClassificationSchema(GreenLangBase):
    """Response schema for due diligence classification result.

    Contains the classified level, risk score, certification credit,
    audit frequency, cost estimates, and regulatory requirements.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "classification_id": "ddc-550e8400-e29b-41d4-a716-446655440000",
                "country_code": "BR",
                "commodity_type": "soya",
                "level": "enhanced",
                "risk_score": 72.5,
                "satellite_required": True,
            }
        },
    )

    classification_id: str = Field(
        ...,
        description="Unique classification identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    country_name: str = Field(
        default="",
        description="Full country name.",
    )
    commodity_type: Optional[str] = Field(
        default=None,
        description="EUDR commodity type, if commodity-specific.",
    )
    level: str = Field(
        ...,
        description="Due diligence level: simplified, standard, enhanced.",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Risk score driving classification.",
    )
    certification_credit: float = Field(
        default=0.0, ge=0.0, le=50.0,
        description="Certification-based risk credit applied.",
    )
    effective_risk_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Risk score after certification credit.",
    )
    audit_frequency: str = Field(
        default="annual",
        description="Recommended audit frequency.",
    )
    satellite_required: bool = Field(
        default=False,
        description="Whether satellite verification is required.",
    )
    cost_estimate_min_eur: Optional[float] = Field(
        default=None, ge=0.0,
        description="Minimum estimated DD cost (EUR per shipment).",
    )
    cost_estimate_max_eur: Optional[float] = Field(
        default=None, ge=0.0,
        description="Maximum estimated DD cost (EUR per shipment).",
    )
    time_to_compliance_days: Optional[int] = Field(
        default=None, ge=0,
        description="Estimated days to achieve compliance.",
    )
    regulatory_requirements: List[str] = Field(
        default_factory=list,
        description="Specific regulatory requirements for this level.",
    )
    classified_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of classification (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )


class ClassificationListSchema(GreenLangBase):
    """Paginated list response for due diligence classifications."""

    model_config = ConfigDict(str_strip_whitespace=True)

    classifications: List[ClassificationSchema] = Field(
        default_factory=list,
        description="List of classifications.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class CostEstimateSchema(GreenLangBase):
    """Request schema for due diligence cost estimation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "BR",
                "commodity_type": "soya",
                "shipments_per_year": 24,
                "certification_schemes": ["iscc"],
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: str = Field(
        ...,
        description="EUDR commodity type.",
    )
    shipments_per_year: int = Field(
        default=1, ge=1, le=10000,
        description="Number of expected shipments per year.",
    )
    certification_schemes: List[str] = Field(
        default_factory=list,
        description="Active certification schemes.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class CostEstimateResultSchema(GreenLangBase):
    """Response schema for due diligence cost estimation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: str = Field(
        ...,
        description="EUDR commodity type.",
    )
    dd_level: str = Field(
        ...,
        description="Classified due diligence level.",
    )
    cost_per_shipment_min_eur: float = Field(
        ..., ge=0.0,
        description="Minimum cost per shipment (EUR).",
    )
    cost_per_shipment_max_eur: float = Field(
        ..., ge=0.0,
        description="Maximum cost per shipment (EUR).",
    )
    annual_cost_min_eur: float = Field(
        ..., ge=0.0,
        description="Minimum annual cost (EUR).",
    )
    annual_cost_max_eur: float = Field(
        ..., ge=0.0,
        description="Maximum annual cost (EUR).",
    )
    shipments_per_year: int = Field(
        ..., ge=1,
        description="Number of shipments per year used in calculation.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class AuditFrequencySchema(GreenLangBase):
    """Response schema for audit frequency recommendation.

    Provides recommended audit frequency based on risk level and
    certification status for a country-commodity pair.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: Optional[str] = Field(
        default=None,
        description="EUDR commodity type.",
    )
    dd_level: str = Field(
        ...,
        description="Due diligence level.",
    )
    audit_frequency: str = Field(
        ...,
        description="Recommended audit frequency (annual, semi_annual, quarterly).",
    )
    audits_per_year: int = Field(
        ..., ge=1,
        description="Number of audits per year.",
    )
    next_audit_due: Optional[datetime] = Field(
        default=None,
        description="Next audit due date.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )


# =============================================================================
# Trade Flow Schemas
# =============================================================================


class AnalyzeFlowSchema(GreenLangBase):
    """Request schema for trade flow analysis.

    Analyzes bilateral commodity trade flows between countries with
    optional re-export detection and sanction overlay.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "origin_country": "BR",
                "destination_country": "NL",
                "commodity_type": "soya",
                "include_re_export_detection": True,
                "include_sanction_overlay": True,
                "period": "2025-Q4",
            }
        },
    )

    origin_country: Optional[str] = Field(
        default=None, min_length=2, max_length=2,
        description="Origin country filter (ISO 3166-1 alpha-2).",
    )
    destination_country: Optional[str] = Field(
        default=None, min_length=2, max_length=2,
        description="Destination country filter (ISO 3166-1 alpha-2).",
    )
    commodity_type: Optional[str] = Field(
        default=None,
        description="Commodity type filter.",
    )
    include_re_export_detection: bool = Field(
        default=True,
        description="Whether to include re-export risk detection.",
    )
    include_sanction_overlay: bool = Field(
        default=True,
        description="Whether to include EU sanction overlay.",
    )
    period: Optional[str] = Field(
        default=None,
        description="Trade period filter (e.g., '2025-Q4').",
    )


class TradeFlowSchema(GreenLangBase):
    """Response schema for a single trade flow record."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "flow_id": "tfl-550e8400-e29b-41d4-a716-446655440000",
                "origin_country": "BR",
                "destination_country": "NL",
                "commodity_type": "soya",
                "volume_tonnes": 50000.0,
                "value_usd": 25000000.0,
                "route_risk_score": 65.0,
                "re_export_risk": 0.15,
            }
        },
    )

    flow_id: str = Field(
        ...,
        description="Unique trade flow identifier.",
    )
    origin_country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 origin country code.",
    )
    destination_country: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 destination country code.",
    )
    commodity_type: str = Field(
        ...,
        description="EUDR commodity type.",
    )
    direction: str = Field(
        default="export",
        description="Trade direction: import, export, re_export, transit.",
    )
    volume_tonnes: Optional[float] = Field(
        default=None, ge=0.0,
        description="Trade volume in tonnes.",
    )
    value_usd: Optional[float] = Field(
        default=None, ge=0.0,
        description="Trade value in USD.",
    )
    route_risk_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Trade route risk score (0-100).",
    )
    re_export_risk: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Re-export risk indicator (0.0-1.0).",
    )
    transshipment_countries: List[str] = Field(
        default_factory=list,
        description="Intermediate countries along the trade route.",
    )
    hs_codes: List[str] = Field(
        default_factory=list,
        description="HS/CN codes for the traded products.",
    )
    quarter: Optional[str] = Field(
        default=None,
        description="Trade period (e.g., '2025-Q4').",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources for this trade flow.",
    )
    recorded_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of record (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class TradeFlowListSchema(GreenLangBase):
    """Paginated list response for trade flows."""

    model_config = ConfigDict(str_strip_whitespace=True)

    flows: List[TradeFlowSchema] = Field(
        default_factory=list,
        description="List of trade flows.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )
    re_export_alerts: List[str] = Field(
        default_factory=list,
        description="Re-export risk alert messages.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class RouteSchema(GreenLangBase):
    """Trade route with risk assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    route_id: str = Field(
        ...,
        description="Unique route identifier.",
    )
    origin_country: str = Field(
        ..., min_length=2, max_length=2,
        description="Route origin country code.",
    )
    destination_country: str = Field(
        ..., min_length=2, max_length=2,
        description="Route destination country code.",
    )
    waypoints: List[str] = Field(
        default_factory=list,
        description="Intermediate country codes along the route.",
    )
    commodity_type: str = Field(
        ...,
        description="EUDR commodity type.",
    )
    route_risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Aggregate route risk score (0-100).",
    )
    transshipment_risk: bool = Field(
        default=False,
        description="Whether transshipment risk is detected.",
    )
    total_volume_tonnes: Optional[float] = Field(
        default=None, ge=0.0,
        description="Total volume through this route (tonnes).",
    )
    flow_count: int = Field(
        default=0, ge=0,
        description="Number of individual flows on this route.",
    )


class RouteListSchema(GreenLangBase):
    """List response for trade routes."""

    model_config = ConfigDict(str_strip_whitespace=True)

    routes: List[RouteSchema] = Field(
        default_factory=list,
        description="List of trade routes.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of routes.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class ReExportRiskSchema(GreenLangBase):
    """Request schema for re-export risk detection.

    Detects potential commodity laundering through re-export by
    analyzing export/production ratios and transshipment patterns.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "SG",
                "commodity_type": "oil_palm",
                "threshold": 0.7,
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="Country to check for re-export risk.",
    )
    commodity_type: Optional[str] = Field(
        default=None,
        description="Commodity type filter (None = all).",
    )
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Re-export risk threshold (0.0-1.0).",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class ReExportRiskResultSchema(GreenLangBase):
    """Response schema for re-export risk detection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="Country analyzed.",
    )
    flagged: bool = Field(
        default=False,
        description="Whether re-export risk exceeds threshold.",
    )
    re_export_ratio: Optional[float] = Field(
        default=None, ge=0.0,
        description="Export-to-production ratio.",
    )
    commodities_at_risk: List[str] = Field(
        default_factory=list,
        description="Commodities exceeding re-export threshold.",
    )
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-commodity re-export analysis details.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Report Schemas
# =============================================================================


class GenerateReportSchema(GreenLangBase):
    """Request schema for risk report generation.

    Generates a formatted risk report in PDF, JSON, HTML, CSV, or
    Excel format with multi-language support (en, fr, de, es, pt).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "report_type": "country_profile",
                "format": "pdf",
                "countries": ["BR", "ID"],
                "commodities": ["soya", "oil_palm"],
                "language": "en",
                "include_charts": True,
            }
        },
    )

    report_type: str = Field(
        ...,
        description=(
            "Report type: country_profile, commodity_matrix, "
            "comparative, trend, due_diligence, executive_summary."
        ),
    )
    format: str = Field(
        default="pdf",
        description="Output format: pdf, json, html, csv, excel.",
    )
    countries: List[str] = Field(
        default_factory=list,
        description="Countries to include in report (ISO 3166-1 alpha-2).",
    )
    commodities: List[str] = Field(
        default_factory=list,
        description="Commodities to include in report.",
    )
    language: str = Field(
        default="en",
        description="Report language: en, fr, de, es, pt.",
    )
    include_charts: bool = Field(
        default=True,
        description="Whether to include visual charts and graphics.",
    )


class ReportSchema(GreenLangBase):
    """Response schema for a generated report."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "report_id": "rpt-550e8400-e29b-41d4-a716-446655440000",
                "report_type": "country_profile",
                "format": "pdf",
                "title": "Country Risk Profile: Brazil & Indonesia",
                "language": "en",
                "file_size_bytes": 1048576,
            }
        },
    )

    report_id: str = Field(
        ...,
        description="Unique report identifier.",
    )
    report_type: str = Field(
        ...,
        description="Type of risk report.",
    )
    format: str = Field(
        ...,
        description="Report output format.",
    )
    title: str = Field(
        default="",
        description="Report title.",
    )
    language: str = Field(
        default="en",
        description="Report language code.",
    )
    countries: List[str] = Field(
        default_factory=list,
        description="Countries covered by this report.",
    )
    commodities: List[str] = Field(
        default_factory=list,
        description="Commodities covered by this report.",
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of report content for integrity.",
    )
    file_size_bytes: Optional[int] = Field(
        default=None, ge=0,
        description="Report file size in bytes.",
    )
    storage_path: Optional[str] = Field(
        default=None,
        description="Storage path or URL for the report file.",
    )
    generated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of report generation (UTC).",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Report expiry date per retention policy.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )


class ReportListSchema(GreenLangBase):
    """Paginated list response for reports."""

    model_config = ConfigDict(str_strip_whitespace=True)

    reports: List[ReportSchema] = Field(
        default_factory=list,
        description="List of reports.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )


class DownloadSchema(GreenLangBase):
    """Response schema for report download.

    Provides a signed download URL or direct download metadata for
    a generated report.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    report_id: str = Field(
        ...,
        description="Report identifier.",
    )
    download_url: Optional[str] = Field(
        default=None,
        description="Signed URL for report download.",
    )
    format: str = Field(
        ...,
        description="Report format (pdf, json, html, csv, excel).",
    )
    file_size_bytes: Optional[int] = Field(
        default=None, ge=0,
        description="File size in bytes.",
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 content hash for integrity verification.",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Download URL expiry time.",
    )


class ExecutiveSummarySchema(GreenLangBase):
    """Request schema for executive summary generation.

    Generates a high-level KPI summary suitable for leadership
    briefings, covering overall risk posture, key changes, and
    action items.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "countries": ["BR", "ID", "CO", "GH"],
                "commodities": ["soya", "cocoa", "oil_palm"],
                "language": "en",
                "period": "2025-Q4",
            }
        },
    )

    countries: List[str] = Field(
        default_factory=list,
        description="Countries to include in summary.",
    )
    commodities: List[str] = Field(
        default_factory=list,
        description="Commodities to include in summary.",
    )
    language: str = Field(
        default="en",
        description="Summary language.",
    )
    period: Optional[str] = Field(
        default=None,
        description="Reporting period (e.g., '2025-Q4').",
    )


class ExecutiveSummaryResultSchema(GreenLangBase):
    """Response schema for executive summary."""

    model_config = ConfigDict(str_strip_whitespace=True)

    summary_id: str = Field(
        ...,
        description="Unique summary identifier.",
    )
    title: str = Field(
        default="",
        description="Summary title.",
    )
    overall_risk_posture: str = Field(
        default="standard",
        description="Aggregate risk posture: low, standard, high.",
    )
    countries_analyzed: int = Field(
        default=0, ge=0,
        description="Number of countries analyzed.",
    )
    high_risk_countries: List[str] = Field(
        default_factory=list,
        description="Countries classified as high risk.",
    )
    key_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Key risk changes since last period.",
    )
    action_items: List[str] = Field(
        default_factory=list,
        description="Recommended action items.",
    )
    kpi_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key performance indicator metrics.",
    )
    generated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of generation (UTC).",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Regulatory Schemas
# =============================================================================


class TrackUpdateSchema(GreenLangBase):
    """Request schema for regulatory update tracking.

    Tracks EC benchmarking list updates, country reclassifications,
    implementing regulations, and enforcement actions.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "BR",
                "change_types": ["reclassification", "enforcement_action"],
                "since": "2025-01-01T00:00:00Z",
                "include_impact_assessment": True,
            }
        },
    )

    country_code: Optional[str] = Field(
        default=None, min_length=2, max_length=2,
        description="Country code filter.",
    )
    change_types: List[str] = Field(
        default_factory=list,
        description=(
            "Change type filters: reclassification, amendment, "
            "enforcement_action, new_guidance."
        ),
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only include updates since this date.",
    )
    include_impact_assessment: bool = Field(
        default=True,
        description="Whether to include impact assessment.",
    )


class UpdateSchema(GreenLangBase):
    """Response schema for a single regulatory update."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "update_id": "reg-550e8400-e29b-41d4-a716-446655440000",
                "regulation": "EU 2023/1115",
                "country_code": "BR",
                "change_type": "reclassification",
                "status": "adopted",
            }
        },
    )

    update_id: str = Field(
        ...,
        description="Unique regulatory update identifier.",
    )
    regulation: str = Field(
        default="EU 2023/1115",
        description="Regulation identifier.",
    )
    country_code: Optional[str] = Field(
        default=None,
        description="Affected country (ISO 3166-1 alpha-2).",
    )
    change_type: str = Field(
        ...,
        description=(
            "Type of change: reclassification, amendment, "
            "enforcement_action, new_guidance."
        ),
    )
    status: str = Field(
        default="adopted",
        description=(
            "Status: proposed, adopted, enforced, amended, repealed."
        ),
    )
    effective_date: Optional[datetime] = Field(
        default=None,
        description="Date the change becomes effective.",
    )
    impact_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Impact score of this change (0-100).",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the change.",
    )
    reference_url: Optional[str] = Field(
        default=None,
        description="URL to official source document.",
    )
    previous_classification: Optional[str] = Field(
        default=None,
        description="Previous risk level for reclassifications.",
    )
    new_classification: Optional[str] = Field(
        default=None,
        description="New risk level for reclassifications.",
    )
    affected_imports_count: Optional[int] = Field(
        default=None, ge=0,
        description="Number of active imports affected.",
    )
    tracked_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of tracking (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class UpdateListSchema(GreenLangBase):
    """Paginated list response for regulatory updates."""

    model_config = ConfigDict(str_strip_whitespace=True)

    updates: List[UpdateSchema] = Field(
        default_factory=list,
        description="List of regulatory updates.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class ImpactAssessmentSchema(GreenLangBase):
    """Request schema for regulatory change impact assessment.

    Assesses the impact of a proposed country reclassification on
    active imports, costs, and compliance timelines.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "country_code": "CO",
                "new_risk_level": "high",
                "affected_commodities": ["coffee", "cocoa"],
                "effective_date": "2026-07-01T00:00:00Z",
            }
        },
    )

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    new_risk_level: str = Field(
        ...,
        description="Proposed new risk level: low, standard, high.",
    )
    affected_commodities: Optional[List[str]] = Field(
        default=None,
        description="Commodities affected by change (None = all).",
    )
    effective_date: Optional[datetime] = Field(
        default=None,
        description="Effective date of the change.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper().strip()


class ImpactAssessmentResultSchema(GreenLangBase):
    """Response schema for regulatory change impact assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    country_name: str = Field(
        default="",
        description="Full country name.",
    )
    current_level: str = Field(
        ...,
        description="Current risk level.",
    )
    proposed_level: str = Field(
        ...,
        description="Proposed new risk level.",
    )
    affected_imports: int = Field(
        default=0, ge=0,
        description="Number of active imports affected.",
    )
    affected_commodities: List[str] = Field(
        default_factory=list,
        description="Commodities affected by the change.",
    )
    cost_impact_eur: Optional[float] = Field(
        default=None,
        description="Estimated annual cost impact (EUR).",
    )
    cost_change_pct: Optional[float] = Field(
        default=None,
        description="Cost change percentage vs current level.",
    )
    action_timeline_days: Optional[int] = Field(
        default=None, ge=0,
        description="Recommended compliance action timeline (days).",
    )
    dd_level_change: Optional[str] = Field(
        default=None,
        description="Due diligence level change (e.g., 'standard -> enhanced').",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Compliance recommendations.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Common
    "PaginationSchema",
    "ErrorSchema",
    "HealthSchema",
    "SuccessSchema",
    # Country
    "AssessCountrySchema",
    "RiskFactorSchema",
    "CountryRiskSchema",
    "CountryListSchema",
    "CountryCompareSchema",
    "CountryCompareResultSchema",
    "TrendPointSchema",
    "TrendSchema",
    # Commodity
    "AnalyzeCommoditySchema",
    "CommodityProfileSchema",
    "CommodityListSchema",
    "RiskMatrixEntrySchema",
    "RiskMatrixSchema",
    "CorrelationEntrySchema",
    "CorrelationSchema",
    # Hotspot
    "DetectHotspotSchema",
    "HotspotSchema",
    "HotspotListSchema",
    "AlertSchema",
    "AlertListSchema",
    "ClusteringSchema",
    "ClusteringResultSchema",
    # Governance
    "EvaluateGovernanceSchema",
    "GovernanceIndexSchema",
    "GovernanceListSchema",
    "GovernanceCompareSchema",
    # Due Diligence
    "ClassifySchema",
    "ClassificationSchema",
    "ClassificationListSchema",
    "CostEstimateSchema",
    "CostEstimateResultSchema",
    "AuditFrequencySchema",
    # Trade Flow
    "AnalyzeFlowSchema",
    "TradeFlowSchema",
    "TradeFlowListSchema",
    "RouteSchema",
    "RouteListSchema",
    "ReExportRiskSchema",
    "ReExportRiskResultSchema",
    # Report
    "GenerateReportSchema",
    "ReportSchema",
    "ReportListSchema",
    "DownloadSchema",
    "ExecutiveSummarySchema",
    "ExecutiveSummaryResultSchema",
    # Regulatory
    "TrackUpdateSchema",
    "UpdateSchema",
    "UpdateListSchema",
    "ImpactAssessmentSchema",
    "ImpactAssessmentResultSchema",
]
