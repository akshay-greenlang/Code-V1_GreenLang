# -*- coding: utf-8 -*-
"""
Country Risk Evaluator Data Models - AGENT-EUDR-016

Pydantic v2 data models for the Country Risk Evaluator Agent covering
composite country risk scoring with 6 weighted factors (deforestation
rate, governance index, enforcement score, corruption perception,
forest law compliance, historical trend); commodity-specific risk
analysis for all 7 EUDR commodities (cattle, cocoa, coffee, oil palm,
rubber, soya, wood); sub-national deforestation hotspot detection with
DBSCAN spatial clustering, fire correlation, and protected area
proximity; governance index engine integrating World Bank WGI,
Transparency International CPI, and FAO/ITTO forest governance
frameworks; automated 3-tier due diligence classification per EUDR
Articles 10-13 (simplified, standard, enhanced); bilateral trade flow
analysis with re-export risk detection and commodity laundering
identification; audit-ready risk report generation in PDF/JSON/HTML
with multi-language support; and EC regulatory update tracking with
country reclassification impact assessment.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all country risk evaluation operations per
EU 2023/1115 Articles 10, 11, 13, 29, and 31.

Enumerations (15):
    - RiskLevel, DueDiligenceLevel, CommodityType, ForestType,
      GovernanceIndicator, HotspotSeverity, DeforestationDriver,
      TradeFlowDirection, ReportFormat, ReportType, RegulatoryStatus,
      AssessmentConfidence, TrendDirection, CertificationScheme,
      DataSource

Core Models (12):
    - CountryRiskAssessment, CommodityRiskProfile, DeforestationHotspot,
      GovernanceIndex, DueDiligenceClassification, TradeFlow,
      RiskReport, RegulatoryUpdate, RiskFactor, RiskHistory,
      CertificationRecord, AuditLogEntry

Request Models (15):
    - AssessCountryRequest, AnalyzeCommodityRequest,
      DetectHotspotsRequest, EvaluateGovernanceRequest,
      ClassifyDueDiligenceRequest, AnalyzeTradeFlowRequest,
      GenerateReportRequest, TrackRegulatoryRequest,
      CompareCountriesRequest, GetTrendsRequest,
      CostEstimateRequest, MatrixRequest, ClusteringRequest,
      ImpactAssessmentRequest, SearchRequest

Response Models (15):
    - CountryRiskResponse, CommodityRiskResponse, HotspotResponse,
      GovernanceResponse, DueDiligenceResponse, TradeFlowResponse,
      ReportResponse, RegulatoryResponse, ComparisonResponse,
      TrendResponse, CostEstimateResponse, MatrixResponse,
      ClusteringResponse, ImpactResponse, HealthResponse

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector and AGENT-EUDR-001 Supply Chain Mapping Master.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
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
    from greenlang.agents.data.eudr_traceability.models import (
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

#: EC benchmarking portal URL (Article 29).
EC_BENCHMARK_URL: str = (
    "https://environment.ec.europa.eu/topics/"
    "forests/deforestation/regulation/benchmarking_en"
)

#: EUDR enforcement date for large operators.
EUDR_ENFORCEMENT_DATE: str = "2025-12-30"

#: EUDR enforcement date for SMEs.
EUDR_SME_ENFORCEMENT_DATE: str = "2026-06-30"

#: Default EUDR commodities (EU 2023/1115 Article 1).
SUPPORTED_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

#: Supported report output formats.
SUPPORTED_OUTPUT_FORMATS: List[str] = [
    "pdf", "json", "html", "csv", "excel",
]

#: Supported report languages.
SUPPORTED_REPORT_LANGUAGES: List[str] = [
    "en", "fr", "de", "es", "pt",
]

#: Default factor weights for composite risk scoring.
DEFAULT_FACTOR_WEIGHTS: Dict[str, float] = {
    "deforestation_rate": 0.30,
    "governance_index": 0.20,
    "enforcement_score": 0.15,
    "corruption_index": 0.15,
    "forest_law_compliance": 0.10,
    "historical_trend": 0.10,
}

#: ISO 3166-1 alpha-2 codes for 200+ supported countries.
SUPPORTED_COUNTRIES: List[str] = [
    "AF", "AL", "DZ", "AD", "AO", "AG", "AR", "AM", "AU", "AT",
    "AZ", "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BT",
    "BO", "BA", "BW", "BR", "BN", "BG", "BF", "BI", "CV", "KH",
    "CM", "CA", "CF", "TD", "CL", "CN", "CO", "KM", "CG", "CD",
    "CR", "CI", "HR", "CU", "CY", "CZ", "DK", "DJ", "DM", "DO",
    "EC", "EG", "SV", "GQ", "ER", "EE", "SZ", "ET", "FJ", "FI",
    "FR", "GA", "GM", "GE", "DE", "GH", "GR", "GD", "GT", "GN",
    "GW", "GY", "HT", "HN", "HU", "IS", "IN", "ID", "IR", "IQ",
    "IE", "IL", "IT", "JM", "JP", "JO", "KZ", "KE", "KI", "KP",
    "KR", "KW", "KG", "LA", "LV", "LB", "LS", "LR", "LY", "LI",
    "LT", "LU", "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MR",
    "MU", "MX", "FM", "MD", "MC", "MN", "ME", "MA", "MZ", "MM",
    "NA", "NR", "NP", "NL", "NZ", "NI", "NE", "NG", "MK", "NO",
    "OM", "PK", "PW", "PA", "PG", "PY", "PE", "PH", "PL", "PT",
    "QA", "RO", "RU", "RW", "KN", "LC", "VC", "WS", "SM", "ST",
    "SA", "SN", "RS", "SC", "SL", "SG", "SK", "SI", "SB", "SO",
    "ZA", "SS", "ES", "LK", "SD", "SR", "SE", "CH", "SY", "TW",
    "TJ", "TZ", "TH", "TL", "TG", "TO", "TT", "TN", "TR", "TM",
    "TV", "UG", "UA", "AE", "GB", "US", "UY", "UZ", "VU", "VE",
    "VN", "YE", "ZM", "ZW", "HK", "MO", "PS", "XK",
]


# =============================================================================
# Enumerations
# =============================================================================


class RiskLevel(str, Enum):
    """Country risk classification per EUDR Article 29.

    LOW: Country benchmarked as low risk by the EC. Eligible for
        simplified due diligence per Article 13. Score 0-30.
    STANDARD: Default classification. Standard due diligence per
        Articles 10-11. Score 31-65.
    HIGH: Country benchmarked as high risk. Enhanced due diligence
        with mandatory satellite verification per Article 11.
        Score 66-100.
    """

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class DueDiligenceLevel(str, Enum):
    """Due diligence classification per EUDR Articles 10-13.

    SIMPLIFIED: Reduced due diligence for low-risk country imports
        per Article 13. Basic documentation and supplier declarations.
    STANDARD: Full due diligence per Articles 10-11 for standard-risk
        imports. Documentation, risk assessment, supplier verification.
    ENHANCED: Enhanced due diligence per Article 11 for high-risk
        imports. Mandatory satellite verification, independent audit,
        supplier site visits.
    """

    SIMPLIFIED = "simplified"
    STANDARD = "standard"
    ENHANCED = "enhanced"


class CommodityType(str, Enum):
    """EUDR-regulated commodities per Article 1 and Annex I.

    Seven commodity groups subject to the EUDR deforestation-free
    requirement. Each commodity has specific HS code mappings,
    derived product definitions, and production characteristics
    that affect risk scoring.
    """

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class ForestType(str, Enum):
    """Forest ecosystem classification for deforestation analysis.

    Categories aligned with FAO Global Forest Resources Assessment
    (FRA) forest type classifications used in deforestation risk
    scoring and hotspot detection.
    """

    TROPICAL_RAINFOREST = "tropical_rainforest"
    TROPICAL_DRY = "tropical_dry"
    TEMPERATE = "temperate"
    BOREAL = "boreal"
    MANGROVE = "mangrove"
    PLANTATION = "plantation"
    SAVANNA = "savanna"


class GovernanceIndicator(str, Enum):
    """World Bank Worldwide Governance Indicators (WGI) dimensions.

    Six aggregate governance indicators compiled from over 30
    underlying data sources, capturing perceptions of governance
    quality across countries. Normalized to 0-100 scale.
    """

    RULE_OF_LAW = "rule_of_law"
    REGULATORY_QUALITY = "regulatory_quality"
    CONTROL_OF_CORRUPTION = "control_of_corruption"
    GOVERNMENT_EFFECTIVENESS = "government_effectiveness"
    VOICE_ACCOUNTABILITY = "voice_accountability"
    POLITICAL_STABILITY = "political_stability"


class HotspotSeverity(str, Enum):
    """Deforestation hotspot severity classification.

    LOW: Minor deforestation activity, below alert threshold.
    MEDIUM: Moderate deforestation activity, approaching alert
        threshold.
    HIGH: Significant deforestation activity, above alert threshold.
    CRITICAL: Severe, rapid deforestation activity requiring
        immediate attention and mandatory enhanced DD.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeforestationDriver(str, Enum):
    """Primary drivers of deforestation per IPCC and FAO classification.

    Controlled vocabulary for attributing deforestation cause at
    the hotspot level using deterministic rule-based classification
    based on commodity production data and land use patterns.
    """

    AGRICULTURE = "agriculture"
    LOGGING = "logging"
    MINING = "mining"
    INFRASTRUCTURE = "infrastructure"
    FIRE = "fire"
    URBANIZATION = "urbanization"


class TradeFlowDirection(str, Enum):
    """Direction of commodity trade flow.

    IMPORT: Commodity entering an EU member state.
    EXPORT: Commodity leaving origin country.
    RE_EXPORT: Commodity re-exported through an intermediary country.
    TRANSIT: Commodity transiting through a country without import.
    """

    IMPORT = "import"
    EXPORT = "export"
    RE_EXPORT = "re_export"
    TRANSIT = "transit"


class ReportFormat(str, Enum):
    """Output format for risk assessment reports.

    PDF: Portable Document Format for regulatory submission.
    JSON: Machine-readable structured data for API integration.
    HTML: Web-viewable report for dashboards.
    CSV: Tabular data for spreadsheet analysis.
    EXCEL: Microsoft Excel workbook with formatted sheets.
    """

    PDF = "pdf"
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    EXCEL = "excel"


class ReportType(str, Enum):
    """Type of risk assessment report.

    COUNTRY_PROFILE: Comprehensive single-country risk assessment.
    COMMODITY_MATRIX: Multi-country, multi-commodity risk heatmap.
    COMPARATIVE: Country-vs-country peer comparison.
    TREND: Historical risk score evolution over time.
    DUE_DILIGENCE: Due diligence requirement specification.
    EXECUTIVE_SUMMARY: High-level KPI summary for leadership.
    """

    COUNTRY_PROFILE = "country_profile"
    COMMODITY_MATRIX = "commodity_matrix"
    COMPARATIVE = "comparative"
    TREND = "trend"
    DUE_DILIGENCE = "due_diligence"
    EXECUTIVE_SUMMARY = "executive_summary"


class RegulatoryStatus(str, Enum):
    """Lifecycle status of a regulatory instrument.

    PROPOSED: Regulation is proposed but not yet adopted.
    ADOPTED: Regulation adopted but not yet enforced.
    ENFORCED: Regulation is actively enforced.
    AMENDED: Regulation has been amended since adoption.
    REPEALED: Regulation has been repealed and is no longer in force.
    """

    PROPOSED = "proposed"
    ADOPTED = "adopted"
    ENFORCED = "enforced"
    AMENDED = "amended"
    REPEALED = "repealed"


class AssessmentConfidence(str, Enum):
    """Confidence level for a risk assessment score.

    VERY_LOW: Insufficient data, score based on global defaults.
    LOW: Data > 12 months old or estimated from regional average.
    MEDIUM: Data 6-12 months old, secondary sources.
    HIGH: Data < 6 months old, primary authoritative sources.
    VERY_HIGH: Data < 3 months old, multiple corroborating primary
        sources.
    """

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TrendDirection(str, Enum):
    """Direction of risk score trend over time.

    IMPROVING: Risk score decreasing (positive trend).
    STABLE: Risk score stable within +/-5% over analysis window.
    DETERIORATING: Risk score increasing (negative trend).
    INSUFFICIENT_DATA: Not enough historical data for trend analysis.
    """

    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    INSUFFICIENT_DATA = "insufficient_data"


class CertificationScheme(str, Enum):
    """Recognized certification schemes for EUDR commodities.

    Each scheme has varying levels of effectiveness per commodity
    and per country, scored 0-100 by the Commodity Risk Analyzer.
    """

    FSC = "fsc"
    PEFC = "pefc"
    RSPO = "rspo"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    FAIRTRADE = "fairtrade"
    ORGANIC = "organic"
    BONSUCRO = "bonsucro"
    ISCC = "iscc"


class DataSource(str, Enum):
    """Authoritative data sources for risk factor calculation.

    Each risk factor score must be traceable to one or more of
    these authoritative data sources for audit compliance.
    """

    EC_BENCHMARKING = "ec_benchmarking"
    FAO = "fao"
    GFW = "gfw"
    WRI = "wri"
    WORLD_BANK = "world_bank"
    TRANSPARENCY_INTL = "transparency_intl"
    ITTO = "itto"
    FIRMS = "firms"


# =============================================================================
# Core Models (12)
# =============================================================================


class CountryRiskAssessment(BaseModel):
    """Composite country risk assessment per EUDR Article 29.

    Represents a complete risk evaluation for a single country,
    combining 6 weighted risk factors into a composite score with
    confidence level, trend direction, and data source attribution.
    All scores use 0-100 scale with higher values indicating higher
    risk.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        frozen=False,
        json_schema_extra={
            "example": {
                "assessment_id": "cra-550e8400-e29b-41d4-a716-446655440000",
                "country_code": "BR",
                "country_name": "Brazil",
                "risk_level": "high",
                "risk_score": 72.5,
                "composite_factors": {
                    "deforestation_rate": 85.0,
                    "governance_index": 55.0,
                    "enforcement_score": 45.0,
                    "corruption_index": 62.0,
                    "forest_law_compliance": 50.0,
                    "historical_trend": 70.0,
                },
            }
        },
    )

    assessment_id: str = Field(
        default_factory=lambda: f"cra-{uuid.uuid4()}",
        description="Unique assessment identifier (UUID).",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    country_name: str = Field(
        ..., min_length=1, max_length=200,
        description="Full country name.",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Three-tier risk classification (low/standard/high).",
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
        default_factory=lambda: dict(DEFAULT_FACTOR_WEIGHTS),
        description="Weights applied to each factor (sum to 1.0).",
    )
    confidence: AssessmentConfidence = Field(
        default=AssessmentConfidence.MEDIUM,
        description="Confidence level based on data freshness and quality.",
    )
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Risk score trend direction over analysis window.",
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
        description="Whether agent classification matches EC benchmark.",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="List of data sources used for this assessment.",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class CommodityRiskProfile(BaseModel):
    """Commodity-specific risk profile per country per EUDR Article 29.

    Provides per-commodity risk assessment including deforestation
    correlation, production volume, certification effectiveness,
    and seasonal factors.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    profile_id: str = Field(
        default_factory=lambda: f"crp-{uuid.uuid4()}",
        description="Unique profile identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: CommodityType = Field(
        ...,
        description="EUDR commodity type.",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Commodity-specific risk score (0-100).",
    )
    risk_level: RiskLevel = Field(
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
        description="Active certification schemes for this commodity-country.",
    )
    certification_effectiveness: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Certification scheme effectiveness score (0-100).",
    )
    seasonal_factors: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Monthly seasonal risk factors (key: month name, "
            "value: risk multiplier)."
        ),
    )
    supply_chain_complexity: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="Supply chain complexity score (1-10).",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used for this profile.",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class DeforestationHotspot(BaseModel):
    """Sub-national deforestation hotspot detected via spatial analysis.

    Represents a geographic area with concentrated deforestation
    activity, identified through DBSCAN-like clustering of GFW
    alerts and Hansen tree cover loss data.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    hotspot_id: str = Field(
        default_factory=lambda: f"dhs-{uuid.uuid4()}",
        description="Unique hotspot identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    region: str = Field(
        ..., min_length=1, max_length=200,
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
    severity: HotspotSeverity = Field(
        ...,
        description="Hotspot severity classification.",
    )
    drivers: List[DeforestationDriver] = Field(
        default_factory=list,
        description="Primary deforestation drivers identified.",
    )
    tree_cover_loss_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description=(
            "Tree cover loss percentage since EUDR cutoff "
            "(Dec 31, 2020)."
        ),
    )
    fire_correlation: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Correlation with FIRMS/VIIRS fire alerts.",
    )
    protected_area_overlap_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Percentage of hotspot overlapping protected areas.",
    )
    protected_area_distance_km: Optional[float] = Field(
        default=None, ge=0.0,
        description="Distance to nearest protected area boundary (km).",
    )
    indigenous_territory_overlap: bool = Field(
        default=False,
        description="Whether hotspot overlaps indigenous territory.",
    )
    trend: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Deforestation trend direction.",
    )
    alert_count: int = Field(
        default=0, ge=0,
        description="Number of deforestation alerts in cluster.",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of detection (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class GovernanceIndex(BaseModel):
    """Governance quality assessment per country.

    Integrates World Bank WGI, Transparency International CPI,
    and FAO/ITTO forest governance framework scores into a
    composite governance quality index (0-100, higher = better
    governance).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    index_id: str = Field(
        default_factory=lambda: f"gix-{uuid.uuid4()}",
        description="Unique governance index identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    overall_score: float = Field(
        ..., ge=0.0, le=100.0,
        description=(
            "Composite governance score (0-100, higher = better "
            "governance)."
        ),
    )
    indicators: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Individual governance indicator scores. Keys map to "
            "GovernanceIndicator enum values."
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
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class DueDiligenceClassification(BaseModel):
    """Due diligence level classification per EUDR Articles 10-13.

    Determines the required level of due diligence for a specific
    country-commodity combination based on risk score, with
    certification credits, cost estimation, and audit frequency.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    classification_id: str = Field(
        default_factory=lambda: f"ddc-{uuid.uuid4()}",
        description="Unique classification identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: Optional[CommodityType] = Field(
        default=None,
        description=(
            "EUDR commodity type. None for country-level "
            "classification."
        ),
    )
    level: DueDiligenceLevel = Field(
        ...,
        description="Required due diligence level.",
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
        description=(
            "Recommended audit frequency (annual, semi_annual, "
            "quarterly)."
        ),
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
        description="List of specific regulatory requirements for this level.",
    )
    classified_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of classification (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class TradeFlow(BaseModel):
    """Bilateral trade flow record for EUDR commodity analysis.

    Represents a commodity trade flow between an origin and
    destination country, with volume, value, and risk scoring
    for re-export detection and route risk assessment.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    flow_id: str = Field(
        default_factory=lambda: f"tfl-{uuid.uuid4()}",
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
    commodity_type: CommodityType = Field(
        ...,
        description="EUDR commodity type.",
    )
    direction: TradeFlowDirection = Field(
        default=TradeFlowDirection.EXPORT,
        description="Trade flow direction.",
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
        description="Countries along the trade route.",
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
    recorded_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of record (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("origin_country", "destination_country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country codes are uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class RiskReport(BaseModel):
    """Generated risk assessment report.

    Represents a risk report document generated by the Risk Report
    Generator engine in one of the supported formats (PDF, JSON,
    HTML, CSV, Excel).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    report_id: str = Field(
        default_factory=lambda: f"rpt-{uuid.uuid4()}",
        description="Unique report identifier.",
    )
    report_type: ReportType = Field(
        ...,
        description="Type of risk report.",
    )
    format: ReportFormat = Field(
        default=ReportFormat.PDF,
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
    generated_at: datetime = Field(
        default_factory=_utcnow,
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


class RegulatoryUpdate(BaseModel):
    """Regulatory change record for EUDR compliance tracking.

    Tracks EC benchmarking list updates, country reclassifications,
    implementing regulations, and enforcement actions.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    update_id: str = Field(
        default_factory=lambda: f"reg-{uuid.uuid4()}",
        description="Unique regulatory update identifier.",
    )
    regulation: str = Field(
        default="EU 2023/1115",
        description="Regulation identifier.",
    )
    country_code: Optional[str] = Field(
        default=None,
        description="Affected country (ISO 3166-1 alpha-2), if applicable.",
    )
    change_type: str = Field(
        ...,
        description=(
            "Type of change (reclassification, amendment, "
            "enforcement_action, new_guidance)."
        ),
    )
    status: RegulatoryStatus = Field(
        default=RegulatoryStatus.ADOPTED,
        description="Status of the regulatory instrument.",
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
    tracked_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of tracking (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class RiskFactor(BaseModel):
    """Individual risk factor within a composite risk score.

    Represents a single factor (e.g., deforestation rate, governance
    index) with its weight, raw value, normalized value, and
    data source attribution.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    factor_name: str = Field(
        ..., min_length=1,
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
        description="Normalized factor value (0-100 scale).",
    )
    data_source: str = Field(
        default="",
        description="Primary data source for this factor.",
    )
    data_date: Optional[datetime] = Field(
        default=None,
        description="Publication date of the source data.",
    )
    last_updated: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of last update (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class RiskHistory(BaseModel):
    """Historical risk score record for trend analysis.

    Tracks risk score changes over time for a country, supporting
    5-year trend analysis with annotated change reasons.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    history_id: str = Field(
        default_factory=lambda: f"rhs-{uuid.uuid4()}",
        description="Unique history record identifier.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Risk score at this point in time.",
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Risk classification at this point in time.",
    )
    change_reason: str = Field(
        default="",
        description="Reason for score change.",
    )
    previous_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Previous risk score before this change.",
    )
    previous_level: Optional[RiskLevel] = Field(
        default=None,
        description="Previous risk level before this change.",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class CertificationRecord(BaseModel):
    """Certification scheme effectiveness record.

    Tracks the effectiveness and coverage of a certification scheme
    for a specific commodity in a specific country.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    record_id: str = Field(
        default_factory=lambda: f"cer-{uuid.uuid4()}",
        description="Unique certification record identifier.",
    )
    scheme: CertificationScheme = Field(
        ...,
        description="Certification scheme.",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: CommodityType = Field(
        ...,
        description="EUDR commodity type.",
    )
    effectiveness_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Certification effectiveness score (0-100).",
    )
    coverage_pct: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Percentage of commodity covered by certification.",
    )
    audit_rigor_score: Optional[float] = Field(
        default=None, ge=0.0, le=100.0,
        description="Audit rigor assessment score (0-100).",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources for this record.",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of assessment (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase ISO 3166-1 alpha-2."""
        return v.upper().strip()


class AuditLogEntry(BaseModel):
    """Immutable audit log entry for risk assessment operations.

    Records all significant operations for EUDR Article 31
    record-keeping compliance (5-year retention). Every entry
    is append-only and includes entity reference, action type,
    actor identification, and change details.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    entry_id: str = Field(
        default_factory=lambda: f"aud-{uuid.uuid4()}",
        description="Unique audit log entry identifier.",
    )
    entity_type: str = Field(
        ...,
        description=(
            "Type of entity (country_assessment, commodity_profile, "
            "hotspot, governance_index, classification, trade_flow, "
            "report, regulatory_update)."
        ),
    )
    entity_id: str = Field(
        ...,
        description="Identifier of the affected entity.",
    )
    action: str = Field(
        ...,
        description=(
            "Action performed (create, update, delete, classify, "
            "generate, archive)."
        ),
    )
    actor: str = Field(
        default="system",
        description="Actor who performed the action.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the action.",
    )
    previous_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values before the change (for updates).",
    )
    new_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Values after the change (for updates).",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of the action (UTC).",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


# =============================================================================
# Request Models (15)
# =============================================================================


class AssessCountryRequest(BaseModel):
    """Request to assess risk for one or more countries."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_codes: List[str] = Field(
        ..., min_length=1,
        description="List of ISO 3166-1 alpha-2 country codes to assess.",
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional custom factor weights (must sum to 1.0).",
    )
    include_trend: bool = Field(
        default=True,
        description="Whether to include trend analysis.",
    )
    include_regional_context: bool = Field(
        default=True,
        description="Whether to include regional comparison.",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Ensure all country codes are uppercase."""
        return [c.upper().strip() for c in v]


class AnalyzeCommodityRequest(BaseModel):
    """Request to analyze commodity-specific risk."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodities: Optional[List[CommodityType]] = Field(
        default=None,
        description="Commodities to analyze (None = all 7).",
    )
    include_seasonal: bool = Field(
        default=True,
        description="Whether to include seasonal analysis.",
    )
    include_certifications: bool = Field(
        default=True,
        description="Whether to include certification assessment.",
    )


class DetectHotspotsRequest(BaseModel):
    """Request to detect deforestation hotspots."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    min_severity: HotspotSeverity = Field(
        default=HotspotSeverity.LOW,
        description="Minimum severity to include.",
    )
    include_fire_correlation: bool = Field(
        default=True,
        description="Whether to include fire alert correlation.",
    )
    include_protected_areas: bool = Field(
        default=True,
        description="Whether to include protected area analysis.",
    )
    temporal_window_months: int = Field(
        default=12, ge=1, le=60,
        description="Temporal window for alert clustering (months).",
    )


class EvaluateGovernanceRequest(BaseModel):
    """Request to evaluate governance quality."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_codes: List[str] = Field(
        ..., min_length=1,
        description="List of country codes to evaluate.",
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
        """Ensure all country codes are uppercase."""
        return [c.upper().strip() for c in v]


class ClassifyDueDiligenceRequest(BaseModel):
    """Request to classify due diligence level."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: Optional[CommodityType] = Field(
        default=None,
        description="Commodity type (None = country-level).",
    )
    region: Optional[str] = Field(
        default=None,
        description="Sub-national region for override checks.",
    )
    certification_schemes: List[str] = Field(
        default_factory=list,
        description="Active certification schemes for credit.",
    )
    include_cost_estimate: bool = Field(
        default=True,
        description="Whether to include cost estimation.",
    )


class AnalyzeTradeFlowRequest(BaseModel):
    """Request to analyze trade flows."""

    model_config = ConfigDict(str_strip_whitespace=True)

    origin_country: Optional[str] = Field(
        default=None,
        description="Origin country filter (ISO alpha-2).",
    )
    destination_country: Optional[str] = Field(
        default=None,
        description="Destination country filter (ISO alpha-2).",
    )
    commodity_type: Optional[CommodityType] = Field(
        default=None,
        description="Commodity type filter.",
    )
    include_re_export_detection: bool = Field(
        default=True,
        description="Whether to include re-export risk detection.",
    )
    include_sanction_overlay: bool = Field(
        default=True,
        description="Whether to include sanction overlay.",
    )
    period: Optional[str] = Field(
        default=None,
        description="Trade period filter (e.g., '2025-Q4').",
    )


class GenerateReportRequest(BaseModel):
    """Request to generate a risk report."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report_type: ReportType = Field(
        ...,
        description="Type of report to generate.",
    )
    format: ReportFormat = Field(
        default=ReportFormat.PDF,
        description="Output format.",
    )
    countries: List[str] = Field(
        default_factory=list,
        description="Countries to include in report.",
    )
    commodities: List[CommodityType] = Field(
        default_factory=list,
        description="Commodities to include in report.",
    )
    language: str = Field(
        default="en",
        description="Report language.",
    )
    include_charts: bool = Field(
        default=True,
        description="Whether to include visual charts.",
    )


class TrackRegulatoryRequest(BaseModel):
    """Request to track regulatory updates."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: Optional[str] = Field(
        default=None,
        description="Country code filter.",
    )
    change_types: List[str] = Field(
        default_factory=list,
        description="Change type filters.",
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only include updates since this date.",
    )
    include_impact_assessment: bool = Field(
        default=True,
        description="Whether to include impact assessment.",
    )


class CompareCountriesRequest(BaseModel):
    """Request to compare risk across countries."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_codes: List[str] = Field(
        ..., min_length=2,
        description="Countries to compare (minimum 2).",
    )
    commodities: Optional[List[CommodityType]] = Field(
        default=None,
        description="Commodities to compare across.",
    )
    include_governance: bool = Field(
        default=True,
        description="Whether to include governance comparison.",
    )

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(cls, v: List[str]) -> List[str]:
        """Ensure all country codes are uppercase."""
        return [c.upper().strip() for c in v]


class GetTrendsRequest(BaseModel):
    """Request to get risk score trends."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    years: int = Field(
        default=5, ge=1, le=20,
        description="Number of years for trend analysis.",
    )
    commodity_type: Optional[CommodityType] = Field(
        default=None,
        description="Optional commodity-specific trends.",
    )


class CostEstimateRequest(BaseModel):
    """Request to estimate due diligence costs."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: CommodityType = Field(
        ...,
        description="EUDR commodity type.",
    )
    shipments_per_year: int = Field(
        default=1, ge=1,
        description="Number of expected shipments per year.",
    )
    certification_schemes: List[str] = Field(
        default_factory=list,
        description="Active certification schemes.",
    )


class MatrixRequest(BaseModel):
    """Request to generate a country-commodity risk matrix."""

    model_config = ConfigDict(str_strip_whitespace=True)

    countries: Optional[List[str]] = Field(
        default=None,
        description="Countries to include (None = all).",
    )
    commodities: Optional[List[CommodityType]] = Field(
        default=None,
        description="Commodities to include (None = all 7).",
    )
    sort_by: str = Field(
        default="risk_score",
        description="Sort criteria for the matrix.",
    )
    format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format.",
    )


class ClusteringRequest(BaseModel):
    """Request to perform hotspot clustering analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    min_points: int = Field(
        default=10, ge=1,
        description="Minimum cluster points (DBSCAN min_samples).",
    )
    radius_km: float = Field(
        default=5.0, gt=0.0,
        description="Clustering radius in km (DBSCAN epsilon).",
    )
    temporal_window_months: int = Field(
        default=12, ge=1, le=60,
        description="Temporal window for clustering (months).",
    )


class ImpactAssessmentRequest(BaseModel):
    """Request for regulatory change impact assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    new_risk_level: RiskLevel = Field(
        ...,
        description="Proposed new risk level.",
    )
    affected_commodities: Optional[List[CommodityType]] = Field(
        default=None,
        description="Commodities affected by change.",
    )
    effective_date: Optional[datetime] = Field(
        default=None,
        description="Effective date of the change.",
    )


class SearchRequest(BaseModel):
    """General search request for risk data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ..., min_length=1,
        description="Search query string.",
    )
    entity_types: List[str] = Field(
        default_factory=list,
        description="Entity types to search across.",
    )
    country_code: Optional[str] = Field(
        default=None,
        description="Country code filter.",
    )
    commodity_type: Optional[CommodityType] = Field(
        default=None,
        description="Commodity type filter.",
    )
    limit: int = Field(
        default=50, ge=1, le=500,
        description="Maximum results to return.",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Offset for pagination.",
    )


# =============================================================================
# Response Models (15)
# =============================================================================


class CountryRiskResponse(BaseModel):
    """Response for country risk assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessments: List[CountryRiskAssessment] = Field(
        default_factory=list,
        description="List of country risk assessments.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of assessments returned.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class CommodityRiskResponse(BaseModel):
    """Response for commodity risk analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    profiles: List[CommodityRiskProfile] = Field(
        default_factory=list,
        description="List of commodity risk profiles.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of profiles returned.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class HotspotResponse(BaseModel):
    """Response for hotspot detection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hotspots: List[DeforestationHotspot] = Field(
        default_factory=list,
        description="List of detected hotspots.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of hotspots detected.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class GovernanceResponse(BaseModel):
    """Response for governance evaluation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    indices: List[GovernanceIndex] = Field(
        default_factory=list,
        description="List of governance indices.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of indices returned.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class DueDiligenceResponse(BaseModel):
    """Response for due diligence classification."""

    model_config = ConfigDict(str_strip_whitespace=True)

    classification: Optional[DueDiligenceClassification] = Field(
        default=None,
        description="Due diligence classification result.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class TradeFlowResponse(BaseModel):
    """Response for trade flow analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    flows: List[TradeFlow] = Field(
        default_factory=list,
        description="List of trade flows.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of flows returned.",
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


class ReportResponse(BaseModel):
    """Response for report generation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report: Optional[RiskReport] = Field(
        default=None,
        description="Generated report metadata.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class RegulatoryResponse(BaseModel):
    """Response for regulatory update tracking."""

    model_config = ConfigDict(str_strip_whitespace=True)

    updates: List[RegulatoryUpdate] = Field(
        default_factory=list,
        description="List of regulatory updates.",
    )
    total_count: int = Field(
        default=0, ge=0,
        description="Total number of updates returned.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class ComparisonResponse(BaseModel):
    """Response for country comparison."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessments: List[CountryRiskAssessment] = Field(
        default_factory=list,
        description="Assessments for compared countries.",
    )
    ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Countries ranked by risk score.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class TrendResponse(BaseModel):
    """Response for trend analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code.",
    )
    history: List[RiskHistory] = Field(
        default_factory=list,
        description="Historical risk score records.",
    )
    trend_direction: TrendDirection = Field(
        default=TrendDirection.STABLE,
        description="Overall trend direction.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class CostEstimateResponse(BaseModel):
    """Response for due diligence cost estimation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code.",
    )
    commodity_type: CommodityType = Field(
        ...,
        description="EUDR commodity type.",
    )
    dd_level: DueDiligenceLevel = Field(
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
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class MatrixResponse(BaseModel):
    """Response for country-commodity risk matrix."""

    model_config = ConfigDict(str_strip_whitespace=True)

    matrix: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Risk matrix data. Each entry contains country_code, "
            "country_name, and per-commodity risk scores."
        ),
    )
    countries_count: int = Field(
        default=0, ge=0,
        description="Number of countries in the matrix.",
    )
    commodities_count: int = Field(
        default=0, ge=0,
        description="Number of commodities in the matrix.",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class ClusteringResponse(BaseModel):
    """Response for hotspot clustering analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    clusters: List[DeforestationHotspot] = Field(
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
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class ImpactResponse(BaseModel):
    """Response for regulatory change impact assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(
        ...,
        description="ISO 3166-1 alpha-2 country code.",
    )
    current_level: RiskLevel = Field(
        ...,
        description="Current risk level.",
    )
    proposed_level: RiskLevel = Field(
        ...,
        description="Proposed new risk level.",
    )
    affected_imports: int = Field(
        default=0, ge=0,
        description="Number of active imports affected.",
    )
    cost_impact_eur: Optional[float] = Field(
        default=None,
        description="Estimated annual cost impact (EUR).",
    )
    action_timeline_days: Optional[int] = Field(
        default=None, ge=0,
        description="Recommended compliance action timeline (days).",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0,
        description="Processing time in milliseconds.",
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 provenance hash for audit trail.",
    )


class HealthResponse(BaseModel):
    """Health check response for the Country Risk Evaluator service.

    Returns service status, version, and key operational metrics.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str = Field(
        default="healthy",
        description="Service health status.",
    )
    version: str = Field(
        default=VERSION,
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
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of health check (UTC).",
    )
