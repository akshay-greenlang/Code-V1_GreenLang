# -*- coding: utf-8 -*-
"""
RiskScoringEngine - PACK-006 EUDR Starter Engine 3
====================================================

Multi-source weighted risk aggregation engine per EUDR Articles 10-11.
Calculates composite risk scores from country, supplier, commodity, and
document risk factors for Due Diligence compliance.

Key Capabilities:
    - Country risk scoring based on deforestation rates, governance, and forest cover
    - Supplier risk assessment from due diligence history and certifications
    - Commodity-specific risk profiles for all 7 EUDR commodities
    - Document completeness and quality risk scoring
    - Composite weighted risk aggregation (country 35%, supplier 25%,
      commodity 20%, document 20%)
    - Article 29 country benchmarking (LOW/STANDARD/HIGH risk)
    - Simplified due diligence eligibility assessment
    - Risk trend analysis across reporting periods

EUDR Risk Framework:
    - Article 10: Risk assessment requirements
    - Article 11: Risk assessment criteria (country, deforestation, supplier)
    - Article 29: Country benchmarking for simplified DD eligibility

Zero-Hallucination:
    - All scores computed from deterministic weighted formulas
    - No LLM involvement in any risk calculation path
    - SHA-256 provenance hashing on every output
    - Pydantic validation at all input/output boundaries

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-006 EUDR Starter
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a value to [low, high] range."""
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Risk classification levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Article29Benchmark(str, Enum):
    """Article 29 country risk benchmark classification."""

    LOW = "LOW"
    STANDARD = "STANDARD"
    HIGH = "HIGH"


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodity categories."""

    CATTLE = "CATTLE"
    COCOA = "COCOA"
    COFFEE = "COFFEE"
    OIL_PALM = "OIL_PALM"
    RUBBER = "RUBBER"
    SOYA = "SOYA"
    WOOD = "WOOD"


class RiskFactorCategory(str, Enum):
    """Categories of risk factors."""

    COUNTRY = "COUNTRY"
    SUPPLIER = "SUPPLIER"
    COMMODITY = "COMMODITY"
    DOCUMENT = "DOCUMENT"
    GEOLOCATION = "GEOLOCATION"
    TEMPORAL = "TEMPORAL"


class TrendDirection(str, Enum):
    """Risk trend direction."""

    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class RiskFactor(BaseModel):
    """Individual risk factor contributing to a score."""

    factor_id: str = Field(default_factory=_new_uuid, description="Factor identifier")
    category: RiskFactorCategory = Field(..., description="Factor category")
    name: str = Field(..., description="Factor name")
    description: str = Field(default="", description="Factor description")
    score: float = Field(..., ge=0, le=100, description="Factor score (0-100)")
    weight: float = Field(default=1.0, ge=0, le=1.0, description="Factor weight")
    weighted_score: float = Field(default=0.0, description="Score * weight")
    data_source: Optional[str] = Field(None, description="Source of the risk data")
    article_reference: Optional[str] = Field(None, description="EUDR article reference")


class CountryRiskScore(BaseModel):
    """Country-level risk score per EUDR Article 11."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 code")
    country_name: str = Field(default="", description="Country name")
    overall_score: float = Field(..., ge=0, le=100, description="Overall country risk 0-100")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    deforestation_rate_score: float = Field(default=0.0, ge=0, le=100, description="Deforestation rate risk")
    governance_score: float = Field(default=0.0, ge=0, le=100, description="Governance risk (inverted)")
    forest_cover_score: float = Field(default=0.0, ge=0, le=100, description="Forest cover risk")
    enforcement_score: float = Field(default=0.0, ge=0, le=100, description="Law enforcement risk")
    article_29_benchmark: Article29Benchmark = Field(
        default=Article29Benchmark.STANDARD, description="Article 29 benchmark"
    )
    factors: List[RiskFactor] = Field(default_factory=list, description="Contributing factors")
    assessed_at: datetime = Field(default_factory=_utcnow, description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class SupplierRiskScore(BaseModel):
    """Supplier-level risk score."""

    supplier_id: str = Field(default="", description="Supplier identifier")
    supplier_name: str = Field(default="", description="Supplier name")
    overall_score: float = Field(..., ge=0, le=100, description="Overall supplier risk 0-100")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    dd_history_score: float = Field(default=0.0, ge=0, le=100, description="Due diligence history risk")
    certification_score: float = Field(default=0.0, ge=0, le=100, description="Certification risk")
    transparency_score: float = Field(default=0.0, ge=0, le=100, description="Transparency risk")
    compliance_history_score: float = Field(default=0.0, ge=0, le=100, description="Compliance history risk")
    tier_depth_score: float = Field(default=0.0, ge=0, le=100, description="Supply chain depth risk")
    factors: List[RiskFactor] = Field(default_factory=list, description="Contributing factors")
    assessed_at: datetime = Field(default_factory=_utcnow, description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CommodityRiskScore(BaseModel):
    """Commodity-specific risk score."""

    commodity: str = Field(..., description="Commodity category")
    country_code: str = Field(default="", description="Country of origin")
    overall_score: float = Field(..., ge=0, le=100, description="Overall commodity risk 0-100")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    base_commodity_risk: float = Field(default=0.0, ge=0, le=100, description="Inherent commodity risk")
    country_commodity_risk: float = Field(default=0.0, ge=0, le=100, description="Country-commodity interaction")
    deforestation_association: float = Field(default=0.0, ge=0, le=100, description="Deforestation association risk")
    supply_chain_complexity: float = Field(default=0.0, ge=0, le=100, description="Supply chain complexity risk")
    factors: List[RiskFactor] = Field(default_factory=list, description="Contributing factors")
    assessed_at: datetime = Field(default_factory=_utcnow, description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class DocumentRiskScore(BaseModel):
    """Document completeness and quality risk score."""

    overall_score: float = Field(..., ge=0, le=100, description="Overall document risk 0-100")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    completeness_score: float = Field(default=0.0, ge=0, le=100, description="Document completeness risk")
    authenticity_score: float = Field(default=0.0, ge=0, le=100, description="Authenticity risk")
    currency_score: float = Field(default=0.0, ge=0, le=100, description="Document currency risk")
    consistency_score: float = Field(default=0.0, ge=0, le=100, description="Cross-document consistency risk")
    total_documents: int = Field(default=0, description="Total documents assessed")
    missing_documents: List[str] = Field(default_factory=list, description="Missing required documents")
    factors: List[RiskFactor] = Field(default_factory=list, description="Contributing factors")
    assessed_at: datetime = Field(default_factory=_utcnow, description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CompositeRiskScore(BaseModel):
    """Composite risk score aggregating all risk dimensions."""

    composite_id: str = Field(default_factory=_new_uuid, description="Composite score identifier")
    overall_score: float = Field(..., ge=0, le=100, description="Overall composite risk 0-100")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    country_component: float = Field(default=0.0, description="Weighted country risk contribution")
    supplier_component: float = Field(default=0.0, description="Weighted supplier risk contribution")
    commodity_component: float = Field(default=0.0, description="Weighted commodity risk contribution")
    document_component: float = Field(default=0.0, description="Weighted document risk contribution")
    weights: Dict[str, float] = Field(default_factory=dict, description="Applied weights")
    country_risk: Optional[CountryRiskScore] = Field(None, description="Country risk details")
    supplier_risk: Optional[SupplierRiskScore] = Field(None, description="Supplier risk details")
    commodity_risk: Optional[CommodityRiskScore] = Field(None, description="Commodity risk details")
    document_risk: Optional[DocumentRiskScore] = Field(None, description="Document risk details")
    all_factors: List[RiskFactor] = Field(default_factory=list, description="All contributing factors")
    requires_enhanced_dd: bool = Field(default=False, description="Whether enhanced DD is required")
    simplified_dd_eligible: bool = Field(default=False, description="Whether simplified DD is possible")
    assessed_at: datetime = Field(default_factory=_utcnow, description="Assessment timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class CountryBenchmark(BaseModel):
    """Article 29 country benchmarking result."""

    country_code: str = Field(..., description="ISO 3166-1 alpha-2 code")
    country_name: str = Field(default="", description="Country name")
    benchmark: Article29Benchmark = Field(..., description="Benchmark classification")
    deforestation_rate: float = Field(default=0.0, description="Annual deforestation rate %")
    governance_index: float = Field(default=0.0, description="Governance index 0-100")
    forest_cover_pct: float = Field(default=0.0, description="Forest cover percentage")
    rationale: str = Field(default="", description="Benchmarking rationale")
    effective_date: Optional[datetime] = Field(None, description="When benchmark became effective")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class SimplifiedDDEligibility(BaseModel):
    """Simplified due diligence eligibility assessment."""

    is_eligible: bool = Field(default=False, description="Whether simplified DD is eligible")
    country_code: str = Field(default="", description="Country code assessed")
    commodity: str = Field(default="", description="Commodity assessed")
    country_benchmark: Article29Benchmark = Field(
        default=Article29Benchmark.STANDARD, description="Country benchmark"
    )
    reasons: List[str] = Field(default_factory=list, description="Eligibility reasons")
    disqualifying_factors: List[str] = Field(default_factory=list, description="Disqualifying factors")
    article_reference: str = Field(default="Article 13", description="EUDR article reference")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class RiskTrendPoint(BaseModel):
    """A single point in a risk trend timeline."""

    period: str = Field(..., description="Period identifier (e.g., '2025-Q1')")
    score: float = Field(..., ge=0, le=100, description="Risk score for this period")
    risk_level: RiskLevel = Field(..., description="Risk classification for this period")


class RiskTrend(BaseModel):
    """Risk trend analysis over multiple periods."""

    entity_id: str = Field(..., description="Entity being tracked")
    entity_type: str = Field(default="", description="Type of entity (supplier, country, etc.)")
    direction: TrendDirection = Field(..., description="Overall trend direction")
    trend_points: List[RiskTrendPoint] = Field(default_factory=list, description="Trend data points")
    average_score: float = Field(default=0.0, description="Average score across periods")
    score_change: float = Field(default=0.0, description="Score change from first to last period")
    periods_analyzed: int = Field(default=0, description="Number of periods analyzed")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Country Risk Database
# ---------------------------------------------------------------------------

# Format: {country_code: (deforestation_rate, governance_index, forest_cover_pct, article_29_benchmark)}
# deforestation_rate: annual % (higher = worse)
# governance_index: 0-100 (higher = better governance)
# forest_cover_pct: % of land area with forest
# article_29_benchmark: LOW, STANDARD, or HIGH risk
COUNTRY_RISK_DB: Dict[str, Tuple[float, float, float, str]] = {
    # Africa
    "AO": (0.25, 28, 46.4, "HIGH"),
    "BF": (1.00, 35, 19.3, "HIGH"),
    "CD": (0.45, 12, 67.3, "HIGH"),
    "CF": (0.10, 8, 35.6, "HIGH"),
    "CG": (0.15, 20, 65.4, "HIGH"),
    "CI": (2.50, 32, 8.9, "HIGH"),
    "CM": (0.90, 30, 39.3, "HIGH"),
    "EG": (0.00, 42, 0.1, "STANDARD"),
    "ET": (1.10, 30, 12.5, "HIGH"),
    "GA": (0.08, 35, 88.0, "STANDARD"),
    "GH": (2.00, 48, 35.0, "HIGH"),
    "GN": (0.50, 25, 25.9, "HIGH"),
    "KE": (0.30, 42, 6.1, "STANDARD"),
    "LR": (0.40, 22, 43.4, "HIGH"),
    "MG": (0.40, 30, 21.4, "HIGH"),
    "ML": (0.60, 28, 3.4, "HIGH"),
    "MZ": (0.30, 25, 43.1, "HIGH"),
    "NG": (3.70, 28, 7.7, "HIGH"),
    "RW": (0.20, 55, 12.4, "STANDARD"),
    "SL": (0.30, 30, 38.1, "HIGH"),
    "SN": (0.50, 45, 42.8, "STANDARD"),
    "TG": (0.80, 32, 3.5, "HIGH"),
    "TZ": (0.80, 38, 48.1, "HIGH"),
    "UG": (2.60, 35, 9.7, "HIGH"),
    "ZA": (0.05, 62, 7.6, "STANDARD"),
    "ZM": (0.40, 35, 60.0, "HIGH"),
    "ZW": (0.50, 22, 36.4, "HIGH"),
    # Americas
    "AR": (0.80, 55, 9.9, "STANDARD"),
    "BO": (0.50, 35, 50.6, "HIGH"),
    "BR": (0.50, 50, 59.4, "HIGH"),
    "BZ": (0.60, 52, 58.9, "STANDARD"),
    "CA": (0.01, 90, 38.7, "LOW"),
    "CL": (0.05, 72, 24.1, "LOW"),
    "CO": (0.40, 48, 52.7, "HIGH"),
    "CR": (0.03, 68, 51.4, "LOW"),
    "DO": (0.10, 45, 40.8, "STANDARD"),
    "EC": (0.60, 40, 50.0, "HIGH"),
    "GT": (1.00, 35, 33.0, "HIGH"),
    "GY": (0.05, 48, 84.3, "STANDARD"),
    "HN": (1.30, 30, 40.8, "HIGH"),
    "MX": (0.25, 48, 33.9, "STANDARD"),
    "NI": (1.00, 32, 25.9, "HIGH"),
    "PA": (0.30, 52, 57.1, "STANDARD"),
    "PE": (0.20, 45, 57.8, "HIGH"),
    "PY": (2.00, 40, 38.6, "HIGH"),
    "SR": (0.02, 45, 93.0, "STANDARD"),
    "SV": (0.15, 40, 12.8, "STANDARD"),
    "US": (0.01, 88, 33.9, "LOW"),
    "UY": (0.00, 72, 10.5, "LOW"),
    "VE": (0.30, 18, 52.4, "HIGH"),
    # Asia
    "CN": (0.00, 48, 22.3, "STANDARD"),
    "ID": (0.70, 42, 49.1, "HIGH"),
    "IN": (0.05, 52, 24.3, "STANDARD"),
    "JP": (0.00, 85, 68.5, "LOW"),
    "KH": (1.50, 28, 47.0, "HIGH"),
    "KR": (0.00, 78, 63.7, "LOW"),
    "LA": (0.70, 25, 58.0, "HIGH"),
    "LK": (0.15, 48, 29.7, "STANDARD"),
    "MM": (1.20, 18, 42.9, "HIGH"),
    "MY": (0.40, 60, 58.5, "STANDARD"),
    "PG": (0.50, 22, 74.1, "HIGH"),
    "PH": (0.30, 45, 24.1, "STANDARD"),
    "PK": (0.20, 32, 2.0, "STANDARD"),
    "SG": (0.00, 90, 23.1, "LOW"),
    "TH": (0.10, 52, 31.6, "STANDARD"),
    "VN": (0.30, 42, 47.6, "STANDARD"),
    # Europe
    "AD": (0.00, 75, 34.0, "LOW"),
    "AT": (0.00, 88, 46.8, "LOW"),
    "BE": (0.00, 85, 22.6, "LOW"),
    "CH": (0.00, 92, 31.7, "LOW"),
    "CZ": (0.00, 78, 34.5, "LOW"),
    "DE": (0.00, 90, 32.7, "LOW"),
    "DK": (0.00, 92, 14.6, "LOW"),
    "EE": (0.00, 82, 52.7, "LOW"),
    "ES": (0.00, 75, 36.8, "LOW"),
    "FI": (0.00, 95, 73.1, "LOW"),
    "FR": (0.00, 82, 31.0, "LOW"),
    "GB": (0.00, 85, 13.0, "LOW"),
    "HR": (0.00, 62, 34.4, "LOW"),
    "HU": (0.00, 65, 22.7, "LOW"),
    "IE": (0.00, 88, 11.0, "LOW"),
    "IT": (0.00, 68, 31.6, "LOW"),
    "LT": (0.00, 72, 34.8, "LOW"),
    "LU": (0.00, 90, 33.5, "LOW"),
    "LV": (0.00, 72, 54.0, "LOW"),
    "NL": (0.00, 90, 11.2, "LOW"),
    "NO": (0.00, 95, 33.2, "LOW"),
    "PL": (0.00, 68, 30.8, "LOW"),
    "PT": (0.00, 72, 35.3, "LOW"),
    "RO": (0.50, 55, 29.8, "STANDARD"),
    "RU": (0.02, 30, 49.4, "STANDARD"),
    "SE": (0.00, 95, 68.9, "LOW"),
    "SI": (0.00, 78, 62.0, "LOW"),
    "SK": (0.00, 68, 40.3, "LOW"),
    "UA": (0.10, 35, 16.7, "STANDARD"),
    # Oceania
    "AU": (0.10, 85, 16.3, "LOW"),
    "NZ": (0.00, 92, 31.4, "LOW"),
}

# Country names for display
COUNTRY_NAMES: Dict[str, str] = {
    "AD": "Andorra", "AO": "Angola", "AR": "Argentina", "AT": "Austria",
    "AU": "Australia", "BE": "Belgium", "BF": "Burkina Faso", "BO": "Bolivia",
    "BR": "Brazil", "BZ": "Belize", "CA": "Canada", "CD": "DR Congo",
    "CF": "Central African Republic", "CG": "Republic of Congo", "CH": "Switzerland",
    "CI": "Cote d'Ivoire", "CL": "Chile", "CM": "Cameroon", "CN": "China",
    "CO": "Colombia", "CR": "Costa Rica", "CZ": "Czechia", "DE": "Germany",
    "DK": "Denmark", "DO": "Dominican Republic", "EC": "Ecuador", "EE": "Estonia",
    "EG": "Egypt", "ES": "Spain", "ET": "Ethiopia", "FI": "Finland", "FR": "France",
    "GA": "Gabon", "GB": "United Kingdom", "GH": "Ghana", "GN": "Guinea",
    "GT": "Guatemala", "GY": "Guyana", "HN": "Honduras", "HR": "Croatia",
    "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IN": "India",
    "IT": "Italy", "JP": "Japan", "KE": "Kenya", "KH": "Cambodia",
    "KR": "South Korea", "LA": "Laos", "LK": "Sri Lanka", "LR": "Liberia",
    "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MG": "Madagascar",
    "ML": "Mali", "MM": "Myanmar", "MX": "Mexico", "MY": "Malaysia",
    "MZ": "Mozambique", "NG": "Nigeria", "NI": "Nicaragua", "NL": "Netherlands",
    "NO": "Norway", "NZ": "New Zealand", "PA": "Panama", "PE": "Peru",
    "PG": "Papua New Guinea", "PH": "Philippines", "PK": "Pakistan",
    "PL": "Poland", "PT": "Portugal", "PY": "Paraguay", "RO": "Romania",
    "RU": "Russia", "RW": "Rwanda", "SE": "Sweden", "SG": "Singapore",
    "SI": "Slovenia", "SK": "Slovakia", "SL": "Sierra Leone", "SN": "Senegal",
    "SR": "Suriname", "SV": "El Salvador", "TG": "Togo", "TH": "Thailand",
    "TZ": "Tanzania", "UA": "Ukraine", "UG": "Uganda", "US": "United States",
    "UY": "Uruguay", "VE": "Venezuela", "VN": "Vietnam", "ZA": "South Africa",
    "ZM": "Zambia", "ZW": "Zimbabwe",
}

# Commodity base risk scores (inherent risk by commodity type)
COMMODITY_BASE_RISK: Dict[str, float] = {
    "CATTLE": 55.0,
    "COCOA": 60.0,
    "COFFEE": 45.0,
    "OIL_PALM": 70.0,
    "RUBBER": 50.0,
    "SOYA": 55.0,
    "WOOD": 40.0,
}

# Commodity-country interaction risk multipliers
# Certain commodity-country combinations have elevated risk
COMMODITY_COUNTRY_RISK: Dict[str, Dict[str, float]] = {
    "CATTLE": {"BR": 1.5, "AR": 1.2, "PY": 1.4, "BO": 1.3, "CO": 1.2},
    "COCOA": {"CI": 1.6, "GH": 1.5, "CM": 1.3, "NG": 1.3, "ID": 1.1},
    "COFFEE": {"BR": 1.1, "VN": 1.1, "CO": 1.0, "ET": 1.3, "HN": 1.2},
    "OIL_PALM": {"ID": 1.6, "MY": 1.3, "PG": 1.4, "CO": 1.2, "NG": 1.2},
    "RUBBER": {"TH": 1.0, "ID": 1.3, "VN": 1.1, "MY": 1.1, "CI": 1.3},
    "SOYA": {"BR": 1.5, "AR": 1.2, "PY": 1.4, "BO": 1.3, "US": 0.8},
    "WOOD": {"BR": 1.3, "ID": 1.3, "RU": 1.1, "CM": 1.3, "CD": 1.5},
}

# Required documents for due diligence
REQUIRED_DOCUMENTS: List[str] = [
    "supplier_declaration",
    "geolocation_data",
    "certificate_of_origin",
    "customs_declaration",
    "transport_documents",
    "satellite_imagery",
    "land_title_or_permit",
    "risk_assessment_report",
]


# ---------------------------------------------------------------------------
# Default Weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "country": 0.35,
    "supplier": 0.25,
    "commodity": 0.20,
    "document": 0.20,
}

# Sub-weights within country risk
COUNTRY_SUB_WEIGHTS: Dict[str, float] = {
    "deforestation_rate": 0.35,
    "governance": 0.30,
    "forest_cover": 0.20,
    "enforcement": 0.15,
}

# Sub-weights within supplier risk
SUPPLIER_SUB_WEIGHTS: Dict[str, float] = {
    "dd_history": 0.25,
    "certification": 0.25,
    "transparency": 0.20,
    "compliance_history": 0.20,
    "tier_depth": 0.10,
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RiskScoringEngine:
    """
    Multi-Source Weighted Risk Aggregation Engine.

    Calculates composite risk scores from country, supplier, commodity,
    and document risk factors per EUDR Articles 10-11. Uses configurable
    weights and deterministic formulas for all calculations.

    Default weights:
        - Country: 35%
        - Supplier: 25%
        - Commodity: 20%
        - Document: 20%

    Risk classifications:
        - LOW: 0-25
        - MEDIUM: 26-50
        - HIGH: 51-75
        - CRITICAL: 76-100

    Attributes:
        config: Optional engine configuration
        weights: Risk dimension weights

    Example:
        >>> engine = RiskScoringEngine()
        >>> country_risk = engine.calculate_country_risk("BR")
        >>> assert country_risk.risk_level in list(RiskLevel)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RiskScoringEngine.

        Args:
            config: Optional configuration dictionary with keys:
                - weights: Custom weights dict (country, supplier, commodity, document)
                - risk_thresholds: Custom thresholds for risk levels
                - enhanced_dd_threshold: Score above which enhanced DD is required
        """
        self.config = config or {}
        self.weights: Dict[str, float] = self.config.get("weights", DEFAULT_WEIGHTS.copy())
        self._enhanced_dd_threshold: float = self.config.get("enhanced_dd_threshold", 50.0)
        self._scoring_count: int = 0
        logger.info("RiskScoringEngine initialized (version=%s, weights=%s)",
                     _MODULE_VERSION, self.weights)

    # -------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------

    def calculate_country_risk(self, country_code: str) -> CountryRiskScore:
        """Calculate country-level risk score per EUDR Article 11.

        Uses deforestation rate, governance index, forest cover percentage,
        and law enforcement metrics to compute a weighted country risk score.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            CountryRiskScore with component scores and risk level.
        """
        code = country_code.upper()
        country_name = COUNTRY_NAMES.get(code, code)
        logger.debug("Calculating country risk for %s (%s)", code, country_name)

        data = COUNTRY_RISK_DB.get(code)
        if data is None:
            # Unknown country defaults to STANDARD risk
            result = CountryRiskScore(
                country_code=code,
                country_name=country_name,
                overall_score=50.0,
                risk_level=RiskLevel.MEDIUM,
                deforestation_rate_score=50.0,
                governance_score=50.0,
                forest_cover_score=50.0,
                enforcement_score=50.0,
                article_29_benchmark=Article29Benchmark.STANDARD,
                factors=[RiskFactor(
                    category=RiskFactorCategory.COUNTRY,
                    name="unknown_country",
                    description=f"Country {code} not in risk database, defaulting to STANDARD",
                    score=50.0,
                    weight=1.0,
                    weighted_score=50.0,
                )],
            )
            result.provenance_hash = _compute_hash(result)
            return result

        deforestation_rate, governance_index, forest_cover, benchmark_str = data

        # Compute sub-scores
        # Deforestation rate: higher rate = higher risk
        deforestation_score = self._deforestation_rate_to_score(deforestation_rate)

        # Governance: higher index = lower risk (invert)
        governance_risk = _clamp(100.0 - governance_index)

        # Forest cover: higher cover + high deforestation = higher risk
        forest_risk = self._forest_cover_to_risk(forest_cover, deforestation_rate)

        # Enforcement: derived from governance with adjustment
        enforcement_risk = _clamp(100.0 - governance_index * 0.8)

        # Weighted average
        overall = (
            deforestation_score * COUNTRY_SUB_WEIGHTS["deforestation_rate"]
            + governance_risk * COUNTRY_SUB_WEIGHTS["governance"]
            + forest_risk * COUNTRY_SUB_WEIGHTS["forest_cover"]
            + enforcement_risk * COUNTRY_SUB_WEIGHTS["enforcement"]
        )
        overall = round(_clamp(overall), 2)

        benchmark = Article29Benchmark(benchmark_str)
        risk_level = self.classify_risk(overall)

        factors = [
            RiskFactor(
                category=RiskFactorCategory.COUNTRY,
                name="deforestation_rate",
                description=f"Annual deforestation rate: {deforestation_rate}%",
                score=round(deforestation_score, 2),
                weight=COUNTRY_SUB_WEIGHTS["deforestation_rate"],
                weighted_score=round(deforestation_score * COUNTRY_SUB_WEIGHTS["deforestation_rate"], 2),
                data_source="EUDR_COUNTRY_DB",
                article_reference="Article 11(1)(a)",
            ),
            RiskFactor(
                category=RiskFactorCategory.COUNTRY,
                name="governance_index",
                description=f"Governance index: {governance_index}/100",
                score=round(governance_risk, 2),
                weight=COUNTRY_SUB_WEIGHTS["governance"],
                weighted_score=round(governance_risk * COUNTRY_SUB_WEIGHTS["governance"], 2),
                data_source="EUDR_COUNTRY_DB",
                article_reference="Article 11(1)(b)",
            ),
            RiskFactor(
                category=RiskFactorCategory.COUNTRY,
                name="forest_cover",
                description=f"Forest cover: {forest_cover}%",
                score=round(forest_risk, 2),
                weight=COUNTRY_SUB_WEIGHTS["forest_cover"],
                weighted_score=round(forest_risk * COUNTRY_SUB_WEIGHTS["forest_cover"], 2),
                data_source="EUDR_COUNTRY_DB",
                article_reference="Article 11(1)(c)",
            ),
            RiskFactor(
                category=RiskFactorCategory.COUNTRY,
                name="enforcement",
                description=f"Law enforcement capacity score",
                score=round(enforcement_risk, 2),
                weight=COUNTRY_SUB_WEIGHTS["enforcement"],
                weighted_score=round(enforcement_risk * COUNTRY_SUB_WEIGHTS["enforcement"], 2),
                data_source="EUDR_COUNTRY_DB",
                article_reference="Article 11(1)(d)",
            ),
        ]

        result = CountryRiskScore(
            country_code=code,
            country_name=country_name,
            overall_score=overall,
            risk_level=risk_level,
            deforestation_rate_score=round(deforestation_score, 2),
            governance_score=round(governance_risk, 2),
            forest_cover_score=round(forest_risk, 2),
            enforcement_score=round(enforcement_risk, 2),
            article_29_benchmark=benchmark,
            factors=factors,
        )
        result.provenance_hash = _compute_hash(result)
        self._scoring_count += 1
        return result

    def calculate_supplier_risk(self, supplier_data: Dict[str, Any]) -> SupplierRiskScore:
        """Calculate supplier-level risk score.

        Evaluates due diligence history, certifications, transparency,
        compliance track record, and supply chain tier depth.

        Args:
            supplier_data: Dictionary with keys:
                - supplier_id, supplier_name
                - dd_completed (bool), dd_history_years (int)
                - certifications (list), active_certifications (int)
                - transparency_rating (0-100)
                - compliance_violations (int), compliance_years (int)
                - tier (int, 1-10)

        Returns:
            SupplierRiskScore with component scores and risk level.
        """
        supplier_id = supplier_data.get("supplier_id", _new_uuid())
        supplier_name = supplier_data.get("supplier_name", "Unknown")

        # DD History risk
        dd_completed = supplier_data.get("dd_completed", False)
        dd_years = supplier_data.get("dd_history_years", 0)
        if not dd_completed:
            dd_risk = 80.0
        elif dd_years >= 3:
            dd_risk = 10.0
        elif dd_years >= 1:
            dd_risk = 30.0
        else:
            dd_risk = 50.0

        # Certification risk
        certs = supplier_data.get("certifications", [])
        active_certs = supplier_data.get("active_certifications", len(certs))
        recognized_certs = {"FSC", "PEFC", "RSPO", "Rainforest Alliance", "UTZ",
                           "Fairtrade", "ISO14001", "ISCC", "Bonsucro", "GlobalGAP"}
        recognized_count = sum(1 for c in certs if c in recognized_certs)
        if recognized_count >= 2:
            cert_risk = 10.0
        elif recognized_count == 1:
            cert_risk = 30.0
        elif active_certs > 0:
            cert_risk = 50.0
        else:
            cert_risk = 75.0

        # Transparency risk
        transparency_rating = supplier_data.get("transparency_rating", 50)
        transparency_risk = _clamp(100.0 - transparency_rating)

        # Compliance history risk
        violations = supplier_data.get("compliance_violations", 0)
        compliance_years = supplier_data.get("compliance_years", 0)
        if violations == 0 and compliance_years >= 2:
            compliance_risk = 10.0
        elif violations == 0:
            compliance_risk = 30.0
        elif violations <= 2:
            compliance_risk = 60.0
        else:
            compliance_risk = 85.0

        # Tier depth risk (deeper tiers = higher risk)
        tier = supplier_data.get("tier", 1)
        tier_risk = _clamp(min(tier * 15.0, 100.0))

        # Weighted average
        overall = (
            dd_risk * SUPPLIER_SUB_WEIGHTS["dd_history"]
            + cert_risk * SUPPLIER_SUB_WEIGHTS["certification"]
            + transparency_risk * SUPPLIER_SUB_WEIGHTS["transparency"]
            + compliance_risk * SUPPLIER_SUB_WEIGHTS["compliance_history"]
            + tier_risk * SUPPLIER_SUB_WEIGHTS["tier_depth"]
        )
        overall = round(_clamp(overall), 2)
        risk_level = self.classify_risk(overall)

        factors = [
            RiskFactor(
                category=RiskFactorCategory.SUPPLIER,
                name="dd_history",
                description=f"DD completed: {dd_completed}, years: {dd_years}",
                score=dd_risk, weight=SUPPLIER_SUB_WEIGHTS["dd_history"],
                weighted_score=round(dd_risk * SUPPLIER_SUB_WEIGHTS["dd_history"], 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.SUPPLIER,
                name="certification",
                description=f"Recognized certifications: {recognized_count}, active: {active_certs}",
                score=cert_risk, weight=SUPPLIER_SUB_WEIGHTS["certification"],
                weighted_score=round(cert_risk * SUPPLIER_SUB_WEIGHTS["certification"], 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.SUPPLIER,
                name="transparency",
                description=f"Transparency rating: {transparency_rating}/100",
                score=transparency_risk, weight=SUPPLIER_SUB_WEIGHTS["transparency"],
                weighted_score=round(transparency_risk * SUPPLIER_SUB_WEIGHTS["transparency"], 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.SUPPLIER,
                name="compliance_history",
                description=f"Violations: {violations} over {compliance_years} years",
                score=compliance_risk, weight=SUPPLIER_SUB_WEIGHTS["compliance_history"],
                weighted_score=round(compliance_risk * SUPPLIER_SUB_WEIGHTS["compliance_history"], 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.SUPPLIER,
                name="tier_depth",
                description=f"Supply chain tier: {tier}",
                score=tier_risk, weight=SUPPLIER_SUB_WEIGHTS["tier_depth"],
                weighted_score=round(tier_risk * SUPPLIER_SUB_WEIGHTS["tier_depth"], 2),
            ),
        ]

        result = SupplierRiskScore(
            supplier_id=supplier_id,
            supplier_name=supplier_name,
            overall_score=overall,
            risk_level=risk_level,
            dd_history_score=dd_risk,
            certification_score=cert_risk,
            transparency_score=transparency_risk,
            compliance_history_score=compliance_risk,
            tier_depth_score=tier_risk,
            factors=factors,
        )
        result.provenance_hash = _compute_hash(result)
        self._scoring_count += 1
        return result

    def calculate_commodity_risk(
        self, commodity: str, country: str
    ) -> CommodityRiskScore:
        """Calculate commodity-specific risk score.

        Combines the inherent commodity risk with country-commodity
        interaction risk factors.

        Args:
            commodity: EUDR commodity name (e.g., 'OIL_PALM', 'COCOA').
            country: ISO 3166-1 alpha-2 country code.

        Returns:
            CommodityRiskScore with component scores and risk level.
        """
        commodity_upper = commodity.upper()
        country_upper = country.upper()

        # Base commodity risk
        base_risk = COMMODITY_BASE_RISK.get(commodity_upper, 50.0)

        # Country-commodity interaction
        country_multiplier = (
            COMMODITY_COUNTRY_RISK
            .get(commodity_upper, {})
            .get(country_upper, 1.0)
        )
        interaction_score = _clamp(base_risk * country_multiplier)

        # Deforestation association from country data
        country_data = COUNTRY_RISK_DB.get(country_upper)
        deforestation_assoc = 50.0
        if country_data:
            defo_rate = country_data[0]
            deforestation_assoc = self._deforestation_rate_to_score(defo_rate)

        # Supply chain complexity (commodity-specific)
        complexity_scores: Dict[str, float] = {
            "CATTLE": 60.0, "COCOA": 65.0, "COFFEE": 50.0,
            "OIL_PALM": 55.0, "RUBBER": 45.0, "SOYA": 50.0, "WOOD": 70.0,
        }
        complexity = complexity_scores.get(commodity_upper, 50.0)

        # Weighted average
        overall = (
            base_risk * 0.30
            + interaction_score * 0.30
            + deforestation_assoc * 0.25
            + complexity * 0.15
        )
        overall = round(_clamp(overall), 2)
        risk_level = self.classify_risk(overall)

        factors = [
            RiskFactor(
                category=RiskFactorCategory.COMMODITY,
                name="base_commodity_risk",
                description=f"Inherent risk for {commodity_upper}",
                score=base_risk, weight=0.30,
                weighted_score=round(base_risk * 0.30, 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.COMMODITY,
                name="country_interaction",
                description=f"Country-commodity risk for {commodity_upper} in {country_upper} "
                            f"(multiplier: {country_multiplier})",
                score=round(interaction_score, 2), weight=0.30,
                weighted_score=round(interaction_score * 0.30, 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.COMMODITY,
                name="deforestation_association",
                description=f"Deforestation association for {country_upper}",
                score=round(deforestation_assoc, 2), weight=0.25,
                weighted_score=round(deforestation_assoc * 0.25, 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.COMMODITY,
                name="supply_chain_complexity",
                description=f"Supply chain complexity for {commodity_upper}",
                score=complexity, weight=0.15,
                weighted_score=round(complexity * 0.15, 2),
            ),
        ]

        result = CommodityRiskScore(
            commodity=commodity_upper,
            country_code=country_upper,
            overall_score=overall,
            risk_level=risk_level,
            base_commodity_risk=base_risk,
            country_commodity_risk=round(interaction_score, 2),
            deforestation_association=round(deforestation_assoc, 2),
            supply_chain_complexity=complexity,
            factors=factors,
        )
        result.provenance_hash = _compute_hash(result)
        self._scoring_count += 1
        return result

    def calculate_document_risk(
        self, documents: List[Dict[str, Any]]
    ) -> DocumentRiskScore:
        """Calculate document completeness and quality risk score.

        Evaluates available documents against required documentation for
        EUDR compliance, checking completeness, authenticity indicators,
        currency (expiry), and cross-document consistency.

        Args:
            documents: List of document dictionaries with keys:
                - document_type (str), title (str)
                - is_authentic (bool), issue_date (str/datetime)
                - expiry_date (str/datetime), issuing_authority (str)

        Returns:
            DocumentRiskScore with component scores and risk level.
        """
        total_docs = len(documents)

        # Completeness: check required document types
        provided_types = {d.get("document_type", "").lower() for d in documents}
        missing = [
            req for req in REQUIRED_DOCUMENTS
            if req not in provided_types
        ]
        completeness_ratio = 1.0 - (len(missing) / len(REQUIRED_DOCUMENTS))
        completeness_risk = _clamp((1.0 - completeness_ratio) * 100.0)

        # Authenticity: check flags
        auth_flags = [d.get("is_authentic", True) for d in documents]
        if total_docs > 0:
            authentic_ratio = sum(1 for a in auth_flags if a) / total_docs
        else:
            authentic_ratio = 0.0
        authenticity_risk = _clamp((1.0 - authentic_ratio) * 100.0)

        # Currency: check expiry dates
        now = _utcnow()
        expired_count = 0
        for d in documents:
            expiry = d.get("expiry_date")
            if expiry:
                if isinstance(expiry, str):
                    try:
                        expiry = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                if isinstance(expiry, datetime) and expiry < now:
                    expired_count += 1

        if total_docs > 0:
            currency_risk = _clamp((expired_count / total_docs) * 100.0)
        else:
            currency_risk = 80.0

        # Consistency: simple check for matching data across documents
        consistency_risk = 20.0  # Base low risk; elevate if inconsistencies found
        authorities = {d.get("issuing_authority", "") for d in documents if d.get("issuing_authority")}
        if not authorities:
            consistency_risk = 60.0

        # Weighted average
        overall = (
            completeness_risk * 0.40
            + authenticity_risk * 0.25
            + currency_risk * 0.20
            + consistency_risk * 0.15
        )
        overall = round(_clamp(overall), 2)
        risk_level = self.classify_risk(overall)

        factors = [
            RiskFactor(
                category=RiskFactorCategory.DOCUMENT,
                name="completeness",
                description=f"Document completeness: {len(REQUIRED_DOCUMENTS) - len(missing)}"
                            f"/{len(REQUIRED_DOCUMENTS)} required documents",
                score=round(completeness_risk, 2), weight=0.40,
                weighted_score=round(completeness_risk * 0.40, 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.DOCUMENT,
                name="authenticity",
                description=f"Document authenticity ratio: {authentic_ratio:.0%}",
                score=round(authenticity_risk, 2), weight=0.25,
                weighted_score=round(authenticity_risk * 0.25, 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.DOCUMENT,
                name="currency",
                description=f"Expired documents: {expired_count}/{total_docs}",
                score=round(currency_risk, 2), weight=0.20,
                weighted_score=round(currency_risk * 0.20, 2),
            ),
            RiskFactor(
                category=RiskFactorCategory.DOCUMENT,
                name="consistency",
                description=f"Cross-document consistency check",
                score=round(consistency_risk, 2), weight=0.15,
                weighted_score=round(consistency_risk * 0.15, 2),
            ),
        ]

        result = DocumentRiskScore(
            overall_score=overall,
            risk_level=risk_level,
            completeness_score=round(completeness_risk, 2),
            authenticity_score=round(authenticity_risk, 2),
            currency_score=round(currency_risk, 2),
            consistency_score=round(consistency_risk, 2),
            total_documents=total_docs,
            missing_documents=missing,
            factors=factors,
        )
        result.provenance_hash = _compute_hash(result)
        self._scoring_count += 1
        return result

    def calculate_composite_risk(
        self,
        country_risk: CountryRiskScore,
        supplier_risk: SupplierRiskScore,
        commodity_risk: CommodityRiskScore,
        document_risk: DocumentRiskScore,
    ) -> CompositeRiskScore:
        """Calculate composite risk score from all risk dimensions.

        Applies configurable weights to aggregate country, supplier,
        commodity, and document risk into a single composite score.

        Args:
            country_risk: Country risk score.
            supplier_risk: Supplier risk score.
            commodity_risk: Commodity risk score.
            document_risk: Document risk score.

        Returns:
            CompositeRiskScore with weighted components and risk level.
        """
        w = self.weights
        country_component = country_risk.overall_score * w.get("country", 0.35)
        supplier_component = supplier_risk.overall_score * w.get("supplier", 0.25)
        commodity_component = commodity_risk.overall_score * w.get("commodity", 0.20)
        document_component = document_risk.overall_score * w.get("document", 0.20)

        overall = country_component + supplier_component + commodity_component + document_component
        overall = round(_clamp(overall), 2)
        risk_level = self.classify_risk(overall)

        # Aggregate all factors
        all_factors = (
            country_risk.factors
            + supplier_risk.factors
            + commodity_risk.factors
            + document_risk.factors
        )

        requires_enhanced = overall > self._enhanced_dd_threshold
        simplified_eligible = (
            country_risk.article_29_benchmark == Article29Benchmark.LOW
            and overall <= 25.0
        )

        result = CompositeRiskScore(
            overall_score=overall,
            risk_level=risk_level,
            country_component=round(country_component, 2),
            supplier_component=round(supplier_component, 2),
            commodity_component=round(commodity_component, 2),
            document_component=round(document_component, 2),
            weights=w,
            country_risk=country_risk,
            supplier_risk=supplier_risk,
            commodity_risk=commodity_risk,
            document_risk=document_risk,
            all_factors=all_factors,
            requires_enhanced_dd=requires_enhanced,
            simplified_dd_eligible=simplified_eligible,
        )
        result.provenance_hash = _compute_hash(result)
        self._scoring_count += 1
        return result

    def classify_risk(self, score: float) -> RiskLevel:
        """Classify a risk score into a risk level.

        Thresholds:
            - LOW: 0-25
            - MEDIUM: 26-50
            - HIGH: 51-75
            - CRITICAL: 76-100

        Args:
            score: Risk score from 0 to 100.

        Returns:
            RiskLevel classification.
        """
        if score <= 25.0:
            return RiskLevel.LOW
        elif score <= 50.0:
            return RiskLevel.MEDIUM
        elif score <= 75.0:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def get_risk_factors(self, score: CompositeRiskScore) -> List[RiskFactor]:
        """Get all contributing risk factors from a composite score.

        Sorts factors by weighted score descending to highlight the
        most significant risk contributors.

        Args:
            score: CompositeRiskScore to extract factors from.

        Returns:
            List of RiskFactor sorted by weighted_score descending.
        """
        factors = list(score.all_factors)
        factors.sort(key=lambda f: f.weighted_score, reverse=True)
        return factors

    def check_article_29_benchmark(self, country: str) -> CountryBenchmark:
        """Check Article 29 country benchmarking classification.

        Determines whether a country is classified as LOW, STANDARD,
        or HIGH risk per Article 29 benchmarking criteria.

        Args:
            country: ISO 3166-1 alpha-2 country code.

        Returns:
            CountryBenchmark with classification and supporting data.
        """
        code = country.upper()
        country_name = COUNTRY_NAMES.get(code, code)

        data = COUNTRY_RISK_DB.get(code)
        if data is None:
            result = CountryBenchmark(
                country_code=code,
                country_name=country_name,
                benchmark=Article29Benchmark.STANDARD,
                rationale=f"Country {code} not in benchmark database, defaulting to STANDARD",
            )
            result.provenance_hash = _compute_hash(result)
            return result

        deforestation_rate, governance_index, forest_cover, benchmark_str = data
        benchmark = Article29Benchmark(benchmark_str)

        if benchmark == Article29Benchmark.LOW:
            rationale = (
                f"{country_name} benchmarked as LOW risk: deforestation rate {deforestation_rate}%, "
                f"governance index {governance_index}/100, forest cover {forest_cover}%"
            )
        elif benchmark == Article29Benchmark.HIGH:
            rationale = (
                f"{country_name} benchmarked as HIGH risk: deforestation rate {deforestation_rate}%, "
                f"governance index {governance_index}/100, elevated deforestation concerns"
            )
        else:
            rationale = (
                f"{country_name} benchmarked as STANDARD risk: moderate deforestation and governance metrics"
            )

        result = CountryBenchmark(
            country_code=code,
            country_name=country_name,
            benchmark=benchmark,
            deforestation_rate=deforestation_rate,
            governance_index=governance_index,
            forest_cover_pct=forest_cover,
            rationale=rationale,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def assess_simplified_dd_eligibility(
        self, country: str, commodity: str
    ) -> SimplifiedDDEligibility:
        """Assess eligibility for simplified due diligence per Article 13.

        Simplified DD is only available when all source countries are
        benchmarked as LOW risk under Article 29.

        Args:
            country: ISO 3166-1 alpha-2 country code.
            commodity: EUDR commodity name.

        Returns:
            SimplifiedDDEligibility with eligibility status and reasons.
        """
        code = country.upper()
        commodity_upper = commodity.upper()

        benchmark = self.check_article_29_benchmark(code)
        reasons: List[str] = []
        disqualifying: List[str] = []

        is_eligible = benchmark.benchmark == Article29Benchmark.LOW

        if is_eligible:
            reasons.append(
                f"Country {code} is benchmarked as LOW risk under Article 29"
            )
            reasons.append(
                f"Simplified DD available per Article 13 for {commodity_upper}"
            )
        else:
            disqualifying.append(
                f"Country {code} is benchmarked as {benchmark.benchmark.value} "
                f"(must be LOW for simplified DD)"
            )

        # Check commodity-country interaction risk
        interaction_risk = (
            COMMODITY_COUNTRY_RISK
            .get(commodity_upper, {})
            .get(code, 1.0)
        )
        if interaction_risk > 1.2:
            is_eligible = False
            disqualifying.append(
                f"Elevated commodity-country risk for {commodity_upper} in {code} "
                f"(interaction multiplier: {interaction_risk})"
            )

        result = SimplifiedDDEligibility(
            is_eligible=is_eligible,
            country_code=code,
            commodity=commodity_upper,
            country_benchmark=benchmark.benchmark,
            reasons=reasons,
            disqualifying_factors=disqualifying,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def batch_risk_assessment(
        self, pairs: List[Dict[str, Any]]
    ) -> List[CompositeRiskScore]:
        """Perform batch risk assessments for multiple country-commodity-supplier combinations.

        Args:
            pairs: List of dictionaries with keys:
                - country (str), commodity (str)
                - supplier_data (dict), documents (list)

        Returns:
            List of CompositeRiskScore for each pair.
        """
        logger.info("Batch risk assessment for %d pairs", len(pairs))
        results: List[CompositeRiskScore] = []

        for pair in pairs:
            try:
                country_risk = self.calculate_country_risk(pair.get("country", ""))
                supplier_risk = self.calculate_supplier_risk(pair.get("supplier_data", {}))
                commodity_risk = self.calculate_commodity_risk(
                    pair.get("commodity", ""), pair.get("country", "")
                )
                document_risk = self.calculate_document_risk(pair.get("documents", []))

                composite = self.calculate_composite_risk(
                    country_risk, supplier_risk, commodity_risk, document_risk
                )
                results.append(composite)
            except Exception as exc:
                logger.warning("Batch risk assessment failed for pair: %s", str(exc))

        return results

    def get_risk_trend(
        self, entity_id: str, periods: List[Dict[str, Any]]
    ) -> RiskTrend:
        """Analyze risk score trends across multiple periods.

        Args:
            entity_id: Identifier of the entity being tracked.
            periods: List of period dictionaries with keys:
                - period (str, e.g., '2025-Q1')
                - score (float, 0-100)

        Returns:
            RiskTrend with direction and trend data points.
        """
        if not periods:
            return RiskTrend(
                entity_id=entity_id,
                direction=TrendDirection.STABLE,
                periods_analyzed=0,
            )

        trend_points: List[RiskTrendPoint] = []
        for p in periods:
            score = float(p.get("score", 50.0))
            trend_points.append(RiskTrendPoint(
                period=p.get("period", ""),
                score=score,
                risk_level=self.classify_risk(score),
            ))

        scores = [tp.score for tp in trend_points]
        avg_score = round(sum(scores) / len(scores), 2)
        score_change = round(scores[-1] - scores[0], 2) if len(scores) > 1 else 0.0

        if score_change < -5:
            direction = TrendDirection.IMPROVING
        elif score_change > 5:
            direction = TrendDirection.DETERIORATING
        else:
            direction = TrendDirection.STABLE

        result = RiskTrend(
            entity_id=entity_id,
            direction=direction,
            trend_points=trend_points,
            average_score=avg_score,
            score_change=score_change,
            periods_analyzed=len(periods),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------
    # Private: Scoring Helpers
    # -------------------------------------------------------------------

    def _deforestation_rate_to_score(self, rate: float) -> float:
        """Convert annual deforestation rate to a 0-100 risk score.

        Uses a non-linear mapping where small increases in high rates
        produce larger score increases.

        Args:
            rate: Annual deforestation rate as percentage.

        Returns:
            Risk score from 0 to 100.
        """
        if rate <= 0.0:
            return 0.0
        elif rate <= 0.05:
            return 10.0
        elif rate <= 0.10:
            return 20.0
        elif rate <= 0.25:
            return 35.0
        elif rate <= 0.50:
            return 50.0
        elif rate <= 1.00:
            return 65.0
        elif rate <= 2.00:
            return 80.0
        elif rate <= 3.00:
            return 90.0
        else:
            return 95.0

    def _forest_cover_to_risk(self, cover_pct: float, deforestation_rate: float) -> float:
        """Convert forest cover percentage and deforestation rate to risk score.

        Higher forest cover combined with high deforestation = highest risk.
        Low forest cover = lower risk (less to lose).

        Args:
            cover_pct: Forest cover as percentage of land area.
            deforestation_rate: Annual deforestation rate.

        Returns:
            Risk score from 0 to 100.
        """
        if cover_pct <= 5.0:
            base = 10.0
        elif cover_pct <= 20.0:
            base = 25.0
        elif cover_pct <= 40.0:
            base = 40.0
        elif cover_pct <= 60.0:
            base = 55.0
        else:
            base = 65.0

        # Boost risk if high deforestation rate with substantial forest cover
        if deforestation_rate > 0.5 and cover_pct > 30.0:
            base = min(base + 20.0, 95.0)

        return _clamp(base)
