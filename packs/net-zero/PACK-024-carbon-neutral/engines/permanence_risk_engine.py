# -*- coding: utf-8 -*-
"""
PermanenceRiskEngine - PACK-024 Carbon Neutral Engine 10
=========================================================

8-category permanence risk assessment engine with reversal, leakage,
technology failure, political instability, market failure, monitoring
gaps, buffer pool adequacy, and natural disaster risk categories,
5-tier classification (very high permanence to very low), credit-level
and portfolio-level risk scoring, and mitigation recommendations.

This engine evaluates the permanence risk of carbon credits used for
carbon neutral claims, ensuring that claimed emission reductions or
removals are durable and unlikely to be reversed. It applies the
ICVCM CCP permanence criteria and provides a comprehensive risk
assessment suitable for ISO 14068-1:2023 Section 8.2 compliance.

Calculation Methodology:
    8 Risk Categories:
        1. Reversal Risk: Risk of stored carbon being released
        2. Leakage Risk: Emissions displaced to other locations
        3. Technology Failure: Technology reliability and maturity
        4. Political Instability: Governance and regulatory risks
        5. Market Failure: Economic viability and market risks
        6. Monitoring Gaps: Monitoring and reporting adequacy
        7. Buffer Pool Adequacy: Sufficiency of buffer contributions
        8. Natural Disasters: Climate and natural hazard exposure

    Each category scored 0-10 (0 = no risk, 10 = extreme risk):
        Category weights (sum to 1.0):
            reversal:            0.20
            leakage:             0.15
            technology_failure:  0.12
            political:           0.10
            market:              0.10
            monitoring:          0.10
            buffer_pool:         0.13
            natural_disaster:    0.10

    Overall Risk Score:
        risk_score = sum(category_score * category_weight) * 10
        Score 0-100: 0 = no risk, 100 = maximum risk

    5-Tier Classification:
        VERY_HIGH_PERMANENCE:  risk_score <= 10  (geological storage, 1000+ years)
        HIGH_PERMANENCE:       risk_score <= 25  (durable removal, 100+ years)
        MODERATE_PERMANENCE:   risk_score <= 45  (managed forests, 40-100 years)
        LOW_PERMANENCE:        risk_score <= 65  (soil carbon, 20-40 years)
        VERY_LOW_PERMANENCE:   risk_score > 65   (temporary, <20 years)

    Permanence-Adjusted Value:
        discount_factor = 1 - (risk_score / 100 * max_discount)
        adjusted_tco2e = nominal_tco2e * discount_factor
        max_discount: 0.50 (50% maximum discount for highest risk)

    Buffer Pool Assessment:
        buffer_adequate = contribution_pct >= required_pct
        required_pct varies by project type:
            Nature-based removal: 20-40%
            Technology removal: 5-10%
            Avoidance: 10-15%

Regulatory References:
    - ICVCM Core Carbon Principles V1.0 (2023) - Criterion 2: Permanence
    - ICVCM Assessment Framework V1.0 (2023) - Permanence requirements
    - ISO 14068-1:2023 - Section 8.2: Credit quality (permanence)
    - Verra VCS AFOLU Non-Permanence Risk Tool (2023)
    - Gold Standard Permanence Requirements (2022)
    - Oxford Principles for Net Zero Aligned Carbon Offsetting (2020)
    - IPCC AR6 WG3 (2022) - Carbon removal permanence

Zero-Hallucination:
    - Risk categories from ICVCM Assessment Framework V1.0
    - Buffer pool requirements from Verra VCS AFOLU Risk Tool
    - Permanence tiers from published carbon removal literature
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-024 Carbon Neutral
Engine:  10 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RiskCategory(str, Enum):
    """Permanence risk categories.

    8 categories covering all major permanence risk factors.
    """
    REVERSAL = "reversal"
    LEAKAGE = "leakage"
    TECHNOLOGY_FAILURE = "technology_failure"
    POLITICAL = "political"
    MARKET = "market"
    MONITORING = "monitoring"
    BUFFER_POOL = "buffer_pool"
    NATURAL_DISASTER = "natural_disaster"

class PermanenceTier(str, Enum):
    """5-tier permanence classification.

    VERY_HIGH: Geological storage, 1000+ years (risk <= 10).
    HIGH: Durable removal, 100+ years (risk <= 25).
    MODERATE: Managed forests, 40-100 years (risk <= 45).
    LOW: Soil carbon, 20-40 years (risk <= 65).
    VERY_LOW: Temporary storage, <20 years (risk > 65).
    """
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"

class ProjectCategory(str, Enum):
    """Project category for risk profiling.

    GEOLOGICAL_STORAGE: CCS, DACCS with geological storage.
    MINERALIZATION: Enhanced weathering, mineral carbonation.
    BIOCHAR: Biochar production and application.
    AFFORESTATION: Tree planting, reforestation.
    FOREST_CONSERVATION: REDD+, avoided deforestation.
    SOIL_CARBON: Regenerative agriculture, soil management.
    BLUE_CARBON: Mangrove, seagrass, coastal wetland.
    OCEAN_BASED: Ocean alkalinity enhancement, direct ocean capture.
    BECCS: Bioenergy with CCS.
    RENEWABLE_ENERGY: Avoidance credits from renewables.
    COOKSTOVE: Clean cooking solutions.
    METHANE_CAPTURE: Landfill gas, methane avoidance.
    """
    GEOLOGICAL_STORAGE = "geological_storage"
    MINERALIZATION = "mineralization"
    BIOCHAR = "biochar"
    AFFORESTATION = "afforestation"
    FOREST_CONSERVATION = "forest_conservation"
    SOIL_CARBON = "soil_carbon"
    BLUE_CARBON = "blue_carbon"
    OCEAN_BASED = "ocean_based"
    BECCS = "beccs"
    RENEWABLE_ENERGY = "renewable_energy"
    COOKSTOVE = "cookstove"
    METHANE_CAPTURE = "methane_capture"

class RiskLevel(str, Enum):
    """Risk level for individual categories."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Risk category weights (sum to 1.0).
RISK_CATEGORY_WEIGHTS: Dict[str, Decimal] = {
    RiskCategory.REVERSAL.value: Decimal("0.20"),
    RiskCategory.LEAKAGE.value: Decimal("0.15"),
    RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("0.12"),
    RiskCategory.POLITICAL.value: Decimal("0.10"),
    RiskCategory.MARKET.value: Decimal("0.10"),
    RiskCategory.MONITORING.value: Decimal("0.10"),
    RiskCategory.BUFFER_POOL.value: Decimal("0.13"),
    RiskCategory.NATURAL_DISASTER.value: Decimal("0.10"),
}

# Permanence tier thresholds (risk score).
TIER_THRESHOLDS: List[Tuple[Decimal, str]] = [
    (Decimal("10"), PermanenceTier.VERY_HIGH.value),
    (Decimal("25"), PermanenceTier.HIGH.value),
    (Decimal("45"), PermanenceTier.MODERATE.value),
    (Decimal("65"), PermanenceTier.LOW.value),
    (Decimal("100"), PermanenceTier.VERY_LOW.value),
]

# Risk level thresholds (per category score 0-10).
RISK_LEVEL_THRESHOLDS: List[Tuple[Decimal, str]] = [
    (Decimal("2"), RiskLevel.NEGLIGIBLE.value),
    (Decimal("4"), RiskLevel.LOW.value),
    (Decimal("6"), RiskLevel.MODERATE.value),
    (Decimal("8"), RiskLevel.HIGH.value),
    (Decimal("10"), RiskLevel.VERY_HIGH.value),
]

# Maximum permanence discount (50%).
MAX_PERMANENCE_DISCOUNT: Decimal = Decimal("0.50")

# Default risk profiles by project category.
# Each value is a dict of category -> typical score (0-10).
DEFAULT_RISK_PROFILES: Dict[str, Dict[str, Decimal]] = {
    ProjectCategory.GEOLOGICAL_STORAGE.value: {
        RiskCategory.REVERSAL.value: Decimal("1"),
        RiskCategory.LEAKAGE.value: Decimal("1"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("2"),
        RiskCategory.POLITICAL.value: Decimal("2"),
        RiskCategory.MARKET.value: Decimal("2"),
        RiskCategory.MONITORING.value: Decimal("2"),
        RiskCategory.BUFFER_POOL.value: Decimal("1"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("1"),
    },
    ProjectCategory.AFFORESTATION.value: {
        RiskCategory.REVERSAL.value: Decimal("6"),
        RiskCategory.LEAKAGE.value: Decimal("5"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("2"),
        RiskCategory.POLITICAL.value: Decimal("5"),
        RiskCategory.MARKET.value: Decimal("4"),
        RiskCategory.MONITORING.value: Decimal("5"),
        RiskCategory.BUFFER_POOL.value: Decimal("5"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("7"),
    },
    ProjectCategory.FOREST_CONSERVATION.value: {
        RiskCategory.REVERSAL.value: Decimal("7"),
        RiskCategory.LEAKAGE.value: Decimal("6"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("2"),
        RiskCategory.POLITICAL.value: Decimal("6"),
        RiskCategory.MARKET.value: Decimal("5"),
        RiskCategory.MONITORING.value: Decimal("5"),
        RiskCategory.BUFFER_POOL.value: Decimal("5"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("7"),
    },
    ProjectCategory.SOIL_CARBON.value: {
        RiskCategory.REVERSAL.value: Decimal("7"),
        RiskCategory.LEAKAGE.value: Decimal("4"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("3"),
        RiskCategory.POLITICAL.value: Decimal("4"),
        RiskCategory.MARKET.value: Decimal("5"),
        RiskCategory.MONITORING.value: Decimal("6"),
        RiskCategory.BUFFER_POOL.value: Decimal("6"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("5"),
    },
    ProjectCategory.BIOCHAR.value: {
        RiskCategory.REVERSAL.value: Decimal("2"),
        RiskCategory.LEAKAGE.value: Decimal("2"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("3"),
        RiskCategory.POLITICAL.value: Decimal("3"),
        RiskCategory.MARKET.value: Decimal("4"),
        RiskCategory.MONITORING.value: Decimal("3"),
        RiskCategory.BUFFER_POOL.value: Decimal("2"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("2"),
    },
    ProjectCategory.MINERALIZATION.value: {
        RiskCategory.REVERSAL.value: Decimal("1"),
        RiskCategory.LEAKAGE.value: Decimal("1"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("4"),
        RiskCategory.POLITICAL.value: Decimal("3"),
        RiskCategory.MARKET.value: Decimal("4"),
        RiskCategory.MONITORING.value: Decimal("3"),
        RiskCategory.BUFFER_POOL.value: Decimal("1"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("1"),
    },
    ProjectCategory.BLUE_CARBON.value: {
        RiskCategory.REVERSAL.value: Decimal("5"),
        RiskCategory.LEAKAGE.value: Decimal("4"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("3"),
        RiskCategory.POLITICAL.value: Decimal("5"),
        RiskCategory.MARKET.value: Decimal("5"),
        RiskCategory.MONITORING.value: Decimal("6"),
        RiskCategory.BUFFER_POOL.value: Decimal("5"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("8"),
    },
    ProjectCategory.RENEWABLE_ENERGY.value: {
        RiskCategory.REVERSAL.value: Decimal("1"),
        RiskCategory.LEAKAGE.value: Decimal("2"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("2"),
        RiskCategory.POLITICAL.value: Decimal("3"),
        RiskCategory.MARKET.value: Decimal("3"),
        RiskCategory.MONITORING.value: Decimal("2"),
        RiskCategory.BUFFER_POOL.value: Decimal("1"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("2"),
    },
    ProjectCategory.COOKSTOVE.value: {
        RiskCategory.REVERSAL.value: Decimal("1"),
        RiskCategory.LEAKAGE.value: Decimal("3"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("4"),
        RiskCategory.POLITICAL.value: Decimal("4"),
        RiskCategory.MARKET.value: Decimal("4"),
        RiskCategory.MONITORING.value: Decimal("5"),
        RiskCategory.BUFFER_POOL.value: Decimal("2"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("2"),
    },
    ProjectCategory.METHANE_CAPTURE.value: {
        RiskCategory.REVERSAL.value: Decimal("2"),
        RiskCategory.LEAKAGE.value: Decimal("3"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("3"),
        RiskCategory.POLITICAL.value: Decimal("3"),
        RiskCategory.MARKET.value: Decimal("3"),
        RiskCategory.MONITORING.value: Decimal("3"),
        RiskCategory.BUFFER_POOL.value: Decimal("2"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("3"),
    },
    ProjectCategory.BECCS.value: {
        RiskCategory.REVERSAL.value: Decimal("2"),
        RiskCategory.LEAKAGE.value: Decimal("2"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("5"),
        RiskCategory.POLITICAL.value: Decimal("4"),
        RiskCategory.MARKET.value: Decimal("5"),
        RiskCategory.MONITORING.value: Decimal("3"),
        RiskCategory.BUFFER_POOL.value: Decimal("2"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("2"),
    },
    ProjectCategory.OCEAN_BASED.value: {
        RiskCategory.REVERSAL.value: Decimal("4"),
        RiskCategory.LEAKAGE.value: Decimal("4"),
        RiskCategory.TECHNOLOGY_FAILURE.value: Decimal("7"),
        RiskCategory.POLITICAL.value: Decimal("5"),
        RiskCategory.MARKET.value: Decimal("6"),
        RiskCategory.MONITORING.value: Decimal("7"),
        RiskCategory.BUFFER_POOL.value: Decimal("5"),
        RiskCategory.NATURAL_DISASTER.value: Decimal("5"),
    },
}

# Buffer pool requirements by project category (minimum %).
BUFFER_REQUIREMENTS: Dict[str, Decimal] = {
    ProjectCategory.AFFORESTATION.value: Decimal("25"),
    ProjectCategory.FOREST_CONSERVATION.value: Decimal("30"),
    ProjectCategory.SOIL_CARBON.value: Decimal("20"),
    ProjectCategory.BLUE_CARBON.value: Decimal("25"),
    ProjectCategory.BIOCHAR.value: Decimal("5"),
    ProjectCategory.GEOLOGICAL_STORAGE.value: Decimal("5"),
    ProjectCategory.MINERALIZATION.value: Decimal("5"),
    ProjectCategory.BECCS.value: Decimal("10"),
    ProjectCategory.OCEAN_BASED.value: Decimal("15"),
    ProjectCategory.RENEWABLE_ENERGY.value: Decimal("0"),
    ProjectCategory.COOKSTOVE.value: Decimal("10"),
    ProjectCategory.METHANE_CAPTURE.value: Decimal("10"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class RiskCategoryInput(BaseModel):
    """Input for a single risk category assessment.

    Attributes:
        category: Risk category identifier.
        score: Risk score (0-10, 0 = no risk, 10 = extreme risk).
        evidence: Evidence supporting the score.
        mitigations: Existing mitigation measures.
        notes: Additional notes.
    """
    category: str = Field(..., description="Risk category")
    score: Decimal = Field(
        default=Decimal("5"), ge=0, le=Decimal("10"),
        description="Risk score (0-10)"
    )
    evidence: str = Field(default="", description="Evidence")
    mitigations: List[str] = Field(
        default_factory=list, description="Mitigation measures"
    )
    notes: str = Field(default="", description="Notes")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = {c.value for c in RiskCategory}
        if v not in valid:
            raise ValueError(f"Unknown risk category '{v}'.")
        return v

class CreditRiskInput(BaseModel):
    """Input for a single credit's permanence risk assessment.

    Attributes:
        credit_id: Credit identifier.
        project_name: Project name.
        project_category: Project category for default risk profile.
        quantity_tco2e: Credit quantity.
        vintage_year: Vintage year.
        country: Project country.
        expected_permanence_years: Expected permanence duration.
        buffer_pool_pct: Buffer pool contribution percentage.
        risk_scores: Per-category risk scores (override defaults).
        has_insurance: Whether insurance or replacement commitment exists.
        has_monitoring: Whether long-term monitoring plan exists.
        is_removal: Whether this is a carbon removal credit.
        standard: Certification standard.
    """
    credit_id: str = Field(default_factory=_new_uuid, description="Credit ID")
    project_name: str = Field(default="", max_length=300, description="Project name")
    project_category: str = Field(
        default=ProjectCategory.AFFORESTATION.value,
        description="Project category"
    )
    quantity_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Quantity (tCO2e)"
    )
    vintage_year: int = Field(default=0, ge=0, le=2060, description="Vintage year")
    country: str = Field(default="", max_length=2, description="Country")
    expected_permanence_years: int = Field(
        default=0, ge=0, description="Expected permanence (years)"
    )
    buffer_pool_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Buffer pool contribution (%)"
    )
    risk_scores: List[RiskCategoryInput] = Field(
        default_factory=list, description="Per-category risk scores"
    )
    has_insurance: bool = Field(default=False, description="Has insurance")
    has_monitoring: bool = Field(default=False, description="Has monitoring plan")
    is_removal: bool = Field(default=False, description="Is removal credit")
    standard: str = Field(default="", max_length=50, description="Standard")

    @field_validator("project_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = {c.value for c in ProjectCategory}
        if v not in valid:
            raise ValueError(f"Unknown project category '{v}'.")
        return v

class PermanenceRiskInput(BaseModel):
    """Complete input for permanence risk assessment.

    Attributes:
        entity_name: Entity name.
        assessment_year: Assessment year.
        credits: Credits to assess.
        use_default_profiles: Whether to use default risk profiles.
        include_portfolio: Whether to include portfolio-level analysis.
        include_mitigations: Whether to generate mitigation recommendations.
        max_acceptable_risk: Maximum acceptable risk score (0-100).
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    assessment_year: int = Field(
        ..., ge=2015, le=2060, description="Assessment year"
    )
    credits: List[CreditRiskInput] = Field(
        default_factory=list, description="Credits to assess"
    )
    use_default_profiles: bool = Field(
        default=True, description="Use default risk profiles"
    )
    include_portfolio: bool = Field(
        default=True, description="Include portfolio analysis"
    )
    include_mitigations: bool = Field(
        default=True, description="Generate mitigations"
    )
    max_acceptable_risk: Decimal = Field(
        default=Decimal("45"), ge=0, le=Decimal("100"),
        description="Max acceptable risk"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class RiskCategoryResult(BaseModel):
    """Result for a single risk category.

    Attributes:
        category: Risk category.
        category_name: Human-readable name.
        score: Risk score (0-10).
        weight: Category weight.
        weighted_score: score * weight.
        risk_level: Risk classification.
        mitigations_in_place: Existing mitigations.
        recommended_mitigations: Recommended additional mitigations.
    """
    category: str = Field(default="")
    category_name: str = Field(default="")
    score: Decimal = Field(default=Decimal("0"))
    weight: Decimal = Field(default=Decimal("0"))
    weighted_score: Decimal = Field(default=Decimal("0"))
    risk_level: str = Field(default=RiskLevel.MODERATE.value)
    mitigations_in_place: List[str] = Field(default_factory=list)
    recommended_mitigations: List[str] = Field(default_factory=list)

class CreditRiskResult(BaseModel):
    """Permanence risk result for a single credit.

    Attributes:
        credit_id: Credit identifier.
        project_name: Project name.
        project_category: Project category.
        quantity_tco2e: Nominal quantity.
        adjusted_tco2e: Permanence-adjusted quantity.
        discount_pct: Permanence discount applied.
        overall_risk_score: Overall risk score (0-100).
        permanence_tier: 5-tier classification.
        expected_permanence_years: Expected permanence.
        category_results: Per-category risk results.
        buffer_pool_adequate: Whether buffer pool is adequate.
        buffer_required_pct: Required buffer contribution.
        buffer_actual_pct: Actual buffer contribution.
        has_insurance: Whether insurance exists.
        has_monitoring: Whether monitoring exists.
        acceptable: Whether risk is within acceptable limits.
        issues: Issues identified.
        recommendations: Recommendations.
    """
    credit_id: str = Field(default="")
    project_name: str = Field(default="")
    project_category: str = Field(default="")
    quantity_tco2e: Decimal = Field(default=Decimal("0"))
    adjusted_tco2e: Decimal = Field(default=Decimal("0"))
    discount_pct: Decimal = Field(default=Decimal("0"))
    overall_risk_score: Decimal = Field(default=Decimal("0"))
    permanence_tier: str = Field(default=PermanenceTier.MODERATE.value)
    expected_permanence_years: int = Field(default=0)
    category_results: List[RiskCategoryResult] = Field(default_factory=list)
    buffer_pool_adequate: bool = Field(default=False)
    buffer_required_pct: Decimal = Field(default=Decimal("0"))
    buffer_actual_pct: Decimal = Field(default=Decimal("0"))
    has_insurance: bool = Field(default=False)
    has_monitoring: bool = Field(default=False)
    acceptable: bool = Field(default=True)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class PortfolioRiskSummary(BaseModel):
    """Portfolio-level permanence risk summary.

    Attributes:
        total_credits: Number of credits assessed.
        total_nominal_tco2e: Total nominal quantity.
        total_adjusted_tco2e: Total after permanence adjustment.
        portfolio_discount_pct: Portfolio-level discount.
        weighted_risk_score: Quantity-weighted risk score.
        tier_distribution: Credits by permanence tier.
        category_distribution: Credits by project category.
        avg_permanence_years: Average expected permanence.
        pct_with_insurance: Percentage with insurance.
        pct_with_monitoring: Percentage with monitoring.
        pct_acceptable: Percentage within acceptable risk.
        highest_risk_credits: IDs of highest-risk credits.
        message: Human-readable summary.
    """
    total_credits: int = Field(default=0)
    total_nominal_tco2e: Decimal = Field(default=Decimal("0"))
    total_adjusted_tco2e: Decimal = Field(default=Decimal("0"))
    portfolio_discount_pct: Decimal = Field(default=Decimal("0"))
    weighted_risk_score: Decimal = Field(default=Decimal("0"))
    tier_distribution: Dict[str, int] = Field(default_factory=dict)
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_permanence_years: int = Field(default=0)
    pct_with_insurance: Decimal = Field(default=Decimal("0"))
    pct_with_monitoring: Decimal = Field(default=Decimal("0"))
    pct_acceptable: Decimal = Field(default=Decimal("0"))
    highest_risk_credits: List[str] = Field(default_factory=list)
    message: str = Field(default="")

class PermanenceRiskResult(BaseModel):
    """Complete permanence risk assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        entity_name: Entity name.
        assessment_year: Assessment year.
        credit_results: Per-credit risk assessments.
        portfolio_summary: Portfolio-level summary.
        max_acceptable_risk: Maximum acceptable risk.
        all_acceptable: Whether all credits are acceptable.
        recommendations: Overall recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    assessment_year: int = Field(default=0)
    credit_results: List[CreditRiskResult] = Field(default_factory=list)
    portfolio_summary: Optional[PortfolioRiskSummary] = Field(default=None)
    max_acceptable_risk: Decimal = Field(default=Decimal("45"))
    all_acceptable: bool = Field(default=False)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Risk Category Names
# ---------------------------------------------------------------------------

RISK_CATEGORY_NAMES: Dict[str, str] = {
    RiskCategory.REVERSAL.value: "Reversal Risk",
    RiskCategory.LEAKAGE.value: "Leakage Risk",
    RiskCategory.TECHNOLOGY_FAILURE.value: "Technology Failure Risk",
    RiskCategory.POLITICAL.value: "Political Instability Risk",
    RiskCategory.MARKET.value: "Market Failure Risk",
    RiskCategory.MONITORING.value: "Monitoring Gaps Risk",
    RiskCategory.BUFFER_POOL.value: "Buffer Pool Adequacy Risk",
    RiskCategory.NATURAL_DISASTER.value: "Natural Disaster Risk",
}

# Mitigation recommendations by category.
MITIGATION_RECOMMENDATIONS: Dict[str, List[str]] = {
    RiskCategory.REVERSAL.value: [
        "Require long-term contractual commitments (40+ years) for reversal liability.",
        "Include reversal insurance or replacement commitments in credit agreements.",
        "Favour project types with inherently low reversal risk (geological, biochar).",
    ],
    RiskCategory.LEAKAGE.value: [
        "Require jurisdiction-wide or landscape-level accounting.",
        "Include leakage deduction factors in quantification methodology.",
        "Favour projects with robust leakage monitoring protocols.",
    ],
    RiskCategory.TECHNOLOGY_FAILURE.value: [
        "Prioritise mature, proven technologies with operational track records.",
        "Require technology performance guarantees or warranties.",
        "Diversify portfolio across multiple technology types.",
    ],
    RiskCategory.POLITICAL.value: [
        "Diversify credits across multiple jurisdictions.",
        "Favour countries with stable governance and rule of law.",
        "Include political risk insurance where available.",
    ],
    RiskCategory.MARKET.value: [
        "Include economic sustainability assessments in credit selection.",
        "Favour projects with diversified revenue streams.",
        "Secure long-term offtake agreements at fixed prices.",
    ],
    RiskCategory.MONITORING.value: [
        "Require comprehensive MRV (Monitoring, Reporting, Verification) plans.",
        "Include satellite/remote sensing for nature-based solutions.",
        "Specify minimum monitoring duration and frequency.",
    ],
    RiskCategory.BUFFER_POOL.value: [
        "Ensure buffer pool contributions meet or exceed standard requirements.",
        "Favour registries with robust buffer pool management.",
        "Consider additional voluntary buffer contributions for high-risk projects.",
    ],
    RiskCategory.NATURAL_DISASTER.value: [
        "Assess climate-related physical risks for project locations.",
        "Require disaster preparedness and response plans.",
        "Diversify geographic exposure to reduce concentration risk.",
    ],
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PermanenceRiskEngine:
    """8-category permanence risk assessment engine.

    Evaluates permanence risk of carbon credits across 8 risk categories
    with 5-tier classification and permanence-adjusted value calculation.

    Usage::

        engine = PermanenceRiskEngine()
        result = engine.assess(input_data)
        for cr in result.credit_results:
            print(f"{cr.project_name}: {cr.permanence_tier} ({cr.overall_risk_score}/100)")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._max_discount = _decimal(
            self.config.get("max_permanence_discount", MAX_PERMANENCE_DISCOUNT)
        )
        logger.info("PermanenceRiskEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def assess(
        self, data: PermanenceRiskInput,
    ) -> PermanenceRiskResult:
        """Perform permanence risk assessment.

        Args:
            data: Validated risk input.

        Returns:
            PermanenceRiskResult with comprehensive assessment.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Assess each credit
        credit_results: List[CreditRiskResult] = []
        for credit in data.credits:
            cr = self._assess_credit(
                credit, data.use_default_profiles,
                data.max_acceptable_risk, data.include_mitigations
            )
            credit_results.append(cr)

        # Step 2: Portfolio summary
        portfolio: Optional[PortfolioRiskSummary] = None
        if data.include_portfolio and credit_results:
            portfolio = self._build_portfolio_summary(
                credit_results, data.max_acceptable_risk
            )

        # Step 3: Overall assessment
        all_acceptable = all(cr.acceptable for cr in credit_results) if credit_results else False

        # Step 4: Recommendations
        recommendations: List[str] = []
        if data.include_mitigations:
            for cr in credit_results:
                if not cr.acceptable:
                    recommendations.append(
                        f"Credit '{cr.project_name}' ({cr.credit_id[:8]}) has risk score "
                        f"{cr.overall_risk_score}/100 ({cr.permanence_tier}). "
                        f"Consider replacing or adding mitigations."
                    )
            if not all_acceptable:
                recommendations.insert(0,
                    "ATTENTION: Some credits exceed the maximum acceptable permanence risk. "
                    "Review portfolio composition."
                )

        if not credit_results:
            warnings.append("No credits provided for assessment.")

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = PermanenceRiskResult(
            entity_name=data.entity_name,
            assessment_year=data.assessment_year,
            credit_results=credit_results,
            portfolio_summary=portfolio,
            max_acceptable_risk=data.max_acceptable_risk,
            all_acceptable=all_acceptable,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Permanence risk assessment complete: credits=%d, all_acceptable=%s, hash=%s",
            len(credit_results), all_acceptable, result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _assess_credit(
        self,
        credit: CreditRiskInput,
        use_defaults: bool,
        max_risk: Decimal,
        include_mitigations: bool,
    ) -> CreditRiskResult:
        """Assess permanence risk for a single credit."""
        # Build risk score map
        input_scores = {rs.category: rs for rs in credit.risk_scores}
        default_profile = DEFAULT_RISK_PROFILES.get(
            credit.project_category, {}
        ) if use_defaults else {}

        category_results: List[RiskCategoryResult] = []
        overall_weighted = Decimal("0")

        for cat in RiskCategory:
            weight = RISK_CATEGORY_WEIGHTS.get(cat.value, Decimal("0.10"))
            cat_name = RISK_CATEGORY_NAMES.get(cat.value, cat.value)

            # Get score: input > default > 5
            if cat.value in input_scores:
                score = input_scores[cat.value].score
                mitigations = input_scores[cat.value].mitigations
            elif cat.value in default_profile:
                score = default_profile[cat.value]
                mitigations = []
            else:
                score = Decimal("5")
                mitigations = []

            # Apply insurance/monitoring bonuses
            if cat.value == RiskCategory.REVERSAL.value and credit.has_insurance:
                score = max(Decimal("0"), score - Decimal("2"))
            if cat.value == RiskCategory.MONITORING.value and credit.has_monitoring:
                score = max(Decimal("0"), score - Decimal("2"))
            if cat.value == RiskCategory.BUFFER_POOL.value:
                req_pct = BUFFER_REQUIREMENTS.get(credit.project_category, Decimal("10"))
                if credit.buffer_pool_pct >= req_pct:
                    score = max(Decimal("0"), score - Decimal("3"))

            weighted = score * weight
            overall_weighted += weighted

            risk_level = self._risk_level(score)

            recs: List[str] = []
            if include_mitigations and score >= Decimal("6"):
                recs = MITIGATION_RECOMMENDATIONS.get(cat.value, [])[:2]

            category_results.append(RiskCategoryResult(
                category=cat.value,
                category_name=cat_name,
                score=score,
                weight=weight,
                weighted_score=_round_val(weighted, 4),
                risk_level=risk_level,
                mitigations_in_place=mitigations,
                recommended_mitigations=recs,
            ))

        # Overall risk score (0-100)
        risk_score = _round_val(overall_weighted * Decimal("10"), 2)

        # Permanence tier
        tier = self._permanence_tier(risk_score)

        # Permanence discount
        discount_pct = _round_val(
            risk_score / Decimal("100") * self._max_discount * Decimal("100"), 2
        )
        adjusted = _round_val(
            credit.quantity_tco2e * (Decimal("100") - discount_pct) / Decimal("100")
        )

        # Buffer pool
        req_buffer = BUFFER_REQUIREMENTS.get(credit.project_category, Decimal("10"))
        buffer_ok = credit.buffer_pool_pct >= req_buffer

        # Acceptable
        acceptable = risk_score <= max_risk

        issues: List[str] = []
        recs_overall: List[str] = []

        if not acceptable:
            issues.append(
                f"Risk score {risk_score}/100 exceeds maximum acceptable {max_risk}."
            )
        if not buffer_ok:
            issues.append(
                f"Buffer pool {credit.buffer_pool_pct}% is below required {req_buffer}%."
            )
        if not credit.has_monitoring:
            issues.append("No long-term monitoring plan in place.")

        return CreditRiskResult(
            credit_id=credit.credit_id,
            project_name=credit.project_name,
            project_category=credit.project_category,
            quantity_tco2e=credit.quantity_tco2e,
            adjusted_tco2e=adjusted,
            discount_pct=discount_pct,
            overall_risk_score=risk_score,
            permanence_tier=tier,
            expected_permanence_years=credit.expected_permanence_years,
            category_results=category_results,
            buffer_pool_adequate=buffer_ok,
            buffer_required_pct=req_buffer,
            buffer_actual_pct=credit.buffer_pool_pct,
            has_insurance=credit.has_insurance,
            has_monitoring=credit.has_monitoring,
            acceptable=acceptable,
            issues=issues,
            recommendations=recs_overall,
        )

    def _build_portfolio_summary(
        self,
        results: List[CreditRiskResult],
        max_risk: Decimal,
    ) -> PortfolioRiskSummary:
        """Build portfolio-level risk summary."""
        total = len(results)
        nominal = sum((cr.quantity_tco2e for cr in results), Decimal("0"))
        adjusted = sum((cr.adjusted_tco2e for cr in results), Decimal("0"))
        discount = _safe_pct(nominal - adjusted, nominal) if nominal > Decimal("0") else Decimal("0")

        # Weighted risk
        weighted_risk = Decimal("0")
        if nominal > Decimal("0"):
            weighted_risk = sum(
                (cr.overall_risk_score * cr.quantity_tco2e for cr in results),
                Decimal("0"),
            ) / nominal

        # Tier distribution
        tier_dist: Dict[str, int] = {}
        for cr in results:
            tier_dist[cr.permanence_tier] = tier_dist.get(cr.permanence_tier, 0) + 1

        # Category distribution
        cat_dist: Dict[str, int] = {}
        for cr in results:
            cat_dist[cr.project_category] = cat_dist.get(cr.project_category, 0) + 1

        # Average permanence
        perms = [cr.expected_permanence_years for cr in results if cr.expected_permanence_years > 0]
        avg_perm = int(sum(perms) / len(perms)) if perms else 0

        # Insurance/monitoring
        with_ins = sum(1 for cr in results if cr.has_insurance)
        with_mon = sum(1 for cr in results if cr.has_monitoring)
        acceptable = sum(1 for cr in results if cr.acceptable)

        pct_ins = _safe_pct(_decimal(with_ins), _decimal(total))
        pct_mon = _safe_pct(_decimal(with_mon), _decimal(total))
        pct_acc = _safe_pct(_decimal(acceptable), _decimal(total))

        # Highest risk
        sorted_by_risk = sorted(results, key=lambda c: c.overall_risk_score, reverse=True)
        highest = [cr.credit_id for cr in sorted_by_risk[:3] if cr.overall_risk_score > max_risk]

        msg = (
            f"Portfolio of {total} credits: weighted risk {_round_val(weighted_risk, 1)}/100, "
            f"permanence discount {_round_val(discount, 1)}%, "
            f"{_round_val(pct_acc, 0)}% within acceptable risk limits."
        )

        return PortfolioRiskSummary(
            total_credits=total,
            total_nominal_tco2e=_round_val(nominal),
            total_adjusted_tco2e=_round_val(adjusted),
            portfolio_discount_pct=_round_val(discount, 2),
            weighted_risk_score=_round_val(weighted_risk, 2),
            tier_distribution=tier_dist,
            category_distribution=cat_dist,
            avg_permanence_years=avg_perm,
            pct_with_insurance=_round_val(pct_ins, 2),
            pct_with_monitoring=_round_val(pct_mon, 2),
            pct_acceptable=_round_val(pct_acc, 2),
            highest_risk_credits=highest,
            message=msg,
        )

    def _permanence_tier(self, risk_score: Decimal) -> str:
        """Determine permanence tier from risk score."""
        for threshold, tier in TIER_THRESHOLDS:
            if risk_score <= threshold:
                return tier
        return PermanenceTier.VERY_LOW.value

    def _risk_level(self, score: Decimal) -> str:
        """Determine risk level from category score."""
        for threshold, level in RISK_LEVEL_THRESHOLDS:
            if score <= threshold:
                return level
        return RiskLevel.VERY_HIGH.value
