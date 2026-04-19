# -*- coding: utf-8 -*-
"""
FIPortfolioEngine - PACK-023 SBTi Alignment Engine 9
======================================================

Financial Institutions Net-Zero (FINZ V1.0) portfolio target engine
supporting 8 asset classes with PCAF data quality scoring (1-5 scale),
portfolio coverage calculation, engagement target tracking, temperature
alignment per asset class, and comprehensive gap analysis for SBTi FI
target submission.

This engine implements the SBTi Financial Institutions Framework V1.0
and the FINZ Standard for setting portfolio-level targets aligned with
net-zero by 2050.  It handles multiple asset classes each with distinct
target-setting methodologies, coverage requirements, and data quality
expectations per PCAF (Partnership for Carbon Accounting Financials).

Calculation Methodology:
    Financed Emissions Attribution:
        FE_entity = sum(attribution_factor_i * emissions_i)
        attribution_factor = outstanding_amount / entity_value

    Portfolio Weighted Temperature Score:
        WATS = sum(portfolio_weight_i * temp_score_i)
        portfolio_weight_i = outstanding_i / total_portfolio

    PCAF Data Quality Score:
        portfolio_dq = sum(weight_i * dq_score_i) / sum(weight_i)
        Score 1 = verified specific (best)
        Score 5 = estimated/proxy (worst)

    Portfolio Coverage:
        coverage_pct = (FE_with_targets / FE_total) * 100
        engagement_pct = (FE_engaged / FE_total) * 100

    Temperature Alignment:
        temp_score = f(target_ambition, target_year, coverage)
        aligned if temp_score <= 1.5 (C)

    Engagement Target:
        engagement_target_met = pct_engaged >= required_pct
        SBTi requires engagement of portfolio companies

Regulatory References:
    - SBTi Financial Institutions Net-Zero Standard V1.0 (2024)
    - SBTi FI Guidance V2.0 (2024) - Target-setting methods
    - PCAF Global GHG Accounting Standard V3.0 (2023)
    - PCAF Data Quality Framework (2023) - Scores 1-5
    - TCFD Recommendations (2017, updated 2022)
    - NZBA Guidelines V2.0 (2024)
    - SBTi Portfolio Coverage Approach (PCA) Guidance
    - SBTi Temperature Rating Methodology V3.0
    - Paris Agreement Art. 2.1(c) - Financial flows alignment
    - ISO 14097:2021 - Climate finance assessment

Zero-Hallucination:
    - All thresholds from SBTi FI Standard V1.0
    - PCAF scores from PCAF Global Standard V3.0
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
Engine:  9 of 10
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
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    """FINZ V1.0 asset classes for portfolio target-setting.

    Each asset class has specific target-setting methodologies,
    coverage requirements, and data quality expectations per
    PCAF Global Standard V3.0.
    """
    CORPORATE_LOANS = "corporate_loans"
    LISTED_EQUITY = "listed_equity"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    MOTOR_VEHICLE_LOANS = "motor_vehicle_loans"
    SOVEREIGN_BONDS = "sovereign_bonds"
    CORPORATE_BONDS = "corporate_bonds"

class PcafDataQuality(int, Enum):
    """PCAF data quality scores (1-5 scale).

    Score 1: Verified emissions from audited reports (highest quality).
    Score 2: Reported, unverified emissions data.
    Score 3: Estimated using physical activity data.
    Score 4: Estimated using economic activity data.
    Score 5: Estimated using sector averages (lowest quality with data).
    """
    SCORE_1 = 1
    SCORE_2 = 2
    SCORE_3 = 3
    SCORE_4 = 4
    SCORE_5 = 5

class TargetMethodology(str, Enum):
    """Target-setting methodology per FINZ V1.0.

    SDA:            Sectoral Decarbonisation Approach (physical intensity).
    PCA:            Portfolio Coverage Approach (% with SBTi targets).
    TEMP_RATING:    Temperature Rating methodology.
    ABSOLUTE:       Absolute emissions reduction.
    INTENSITY:      Economic intensity reduction.
    ENGAGEMENT:     Portfolio engagement target.
    NOT_SET:        No methodology selected.
    """
    SDA = "sda"
    PCA = "pca"
    TEMP_RATING = "temperature_rating"
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"
    ENGAGEMENT = "engagement"
    NOT_SET = "not_set"

class PortfolioCoverageStatus(str, Enum):
    """Coverage status for portfolio targets.

    FULL:     All entities in scope have targets.
    PARTIAL:  Some entities covered.
    MINIMAL:  Below minimum coverage threshold.
    NONE:     No coverage.
    """
    FULL = "full"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    NONE = "none"

class EngagementStatus(str, Enum):
    """Engagement status for portfolio entities.

    ACTIVE:           Active engagement ongoing.
    COMMITTED:        Entity committed to set SBTi targets.
    TARGET_SET:       Entity has validated SBTi targets.
    ESCALATION:       Engagement escalated (no response).
    NOT_ENGAGED:      No engagement initiated.
    DIVESTED:         Position divested.
    """
    ACTIVE = "active"
    COMMITTED = "committed"
    TARGET_SET = "target_set"
    ESCALATION = "escalation"
    NOT_ENGAGED = "not_engaged"
    DIVESTED = "divested"

class TemperatureAlignment(str, Enum):
    """Temperature alignment classification.

    BELOW_1_5C:     Below 1.5 degrees C.
    AT_1_5C:        At 1.5 degrees C.
    BETWEEN_1_5_2C: Between 1.5 and 2 degrees C.
    AT_2C:          At 2 degrees C.
    ABOVE_2C:       Above 2 degrees C.
    NOT_ASSESSED:   Not yet assessed.
    """
    BELOW_1_5C = "below_1.5c"
    AT_1_5C = "1.5c"
    BETWEEN_1_5_2C = "1.5c_to_2c"
    AT_2C = "2c"
    ABOVE_2C = "above_2c"
    NOT_ASSESSED = "not_assessed"

class AssetClassRisk(str, Enum):
    """Climate risk level for an asset class.

    HIGH:   Significant transition risk exposure.
    MEDIUM: Moderate transition risk.
    LOW:    Limited transition risk.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# ---------------------------------------------------------------------------
# Constants -- SBTi FINZ V1.0 Thresholds
# ---------------------------------------------------------------------------

# Portfolio coverage minimum: near-term target.
# Source: SBTi FI Standard V1.0, Section 5.3.
PORTFOLIO_COVERAGE_NT_MIN_PCT: Decimal = Decimal("67.0")

# Portfolio coverage minimum: long-term target.
# Source: SBTi FI Standard V1.0, Section 5.4.
PORTFOLIO_COVERAGE_LT_MIN_PCT: Decimal = Decimal("90.0")

# Minimum engagement percentage for PCA approach.
# Source: SBTi PCA Guidance, Section 3.
PCA_ENGAGEMENT_MIN_PCT: Decimal = Decimal("50.0")

# Target year for near-term PCA.
PCA_NEAR_TERM_YEARS: int = 5

# Temperature rating threshold for 1.5C alignment.
TEMP_RATING_1_5C: Decimal = Decimal("1.5")

# Temperature rating threshold for well-below 2C.
TEMP_RATING_WB2C: Decimal = Decimal("2.0")

# PCAF maximum acceptable data quality score for target-setting.
PCAF_MAX_ACCEPTABLE_DQ: int = 4

# PCAF target data quality score.
PCAF_TARGET_DQ: int = 3

# Minimum entities in portfolio for meaningful analysis.
MIN_PORTFOLIO_ENTITIES: int = 5

# Maximum entities supported per analysis batch.
MAX_PORTFOLIO_ENTITIES: int = 500

# Minimum portfolio coverage for any target type.
MIN_COVERAGE_FOR_TARGET_PCT: Decimal = Decimal("25.0")

# Asset class reference data with typical methodologies and thresholds.
ASSET_CLASS_REFERENCE: Dict[str, Dict[str, Any]] = {
    AssetClass.CORPORATE_LOANS.value: {
        "name": "Corporate Loans",
        "description": "Loans to corporate entities for general purposes or specific projects",
        "eligible_methodologies": ["sda", "pca", "temperature_rating", "absolute", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.1",
        "attribution_method": "outstanding_amount / (debt + equity)",
        "typical_risk": "high",
        "finz_required": True,
        "near_term_required": True,
        "long_term_required": True,
    },
    AssetClass.LISTED_EQUITY.value: {
        "name": "Listed Equity and Bonds",
        "description": "Publicly traded equity investments and corporate bonds",
        "eligible_methodologies": ["sda", "pca", "temperature_rating", "absolute", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.2",
        "attribution_method": "outstanding_amount / (debt + equity)",
        "typical_risk": "high",
        "finz_required": True,
        "near_term_required": True,
        "long_term_required": True,
    },
    AssetClass.PROJECT_FINANCE.value: {
        "name": "Project Finance",
        "description": "Financing for specific projects (e.g. renewable energy, infrastructure)",
        "eligible_methodologies": ["sda", "absolute", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.3",
        "attribution_method": "outstanding_amount / total_project_cost",
        "typical_risk": "medium",
        "finz_required": True,
        "near_term_required": True,
        "long_term_required": False,
    },
    AssetClass.COMMERCIAL_REAL_ESTATE.value: {
        "name": "Commercial Real Estate",
        "description": "Loans and investments in commercial properties",
        "eligible_methodologies": ["sda", "absolute", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.4",
        "attribution_method": "outstanding_amount / property_value",
        "typical_risk": "medium",
        "finz_required": True,
        "near_term_required": True,
        "long_term_required": True,
    },
    AssetClass.MORTGAGES.value: {
        "name": "Residential Mortgages",
        "description": "Mortgage loans for residential properties",
        "eligible_methodologies": ["sda", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.5",
        "attribution_method": "outstanding_amount / property_value",
        "typical_risk": "low",
        "finz_required": True,
        "near_term_required": True,
        "long_term_required": False,
    },
    AssetClass.MOTOR_VEHICLE_LOANS.value: {
        "name": "Motor Vehicle Loans",
        "description": "Loans for vehicle purchases (cars, trucks, fleet)",
        "eligible_methodologies": ["sda", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.6",
        "attribution_method": "outstanding_amount / vehicle_value",
        "typical_risk": "medium",
        "finz_required": False,
        "near_term_required": False,
        "long_term_required": False,
    },
    AssetClass.SOVEREIGN_BONDS.value: {
        "name": "Sovereign Bonds",
        "description": "Government bonds and sovereign debt instruments",
        "eligible_methodologies": ["absolute", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.7",
        "attribution_method": "outstanding_amount / total_sovereign_debt",
        "typical_risk": "low",
        "finz_required": False,
        "near_term_required": False,
        "long_term_required": False,
    },
    AssetClass.CORPORATE_BONDS.value: {
        "name": "Corporate Bonds",
        "description": "Fixed-income instruments issued by corporations",
        "eligible_methodologies": ["sda", "pca", "temperature_rating", "absolute", "intensity"],
        "pcaf_coverage": "PCAF Standard Section 5.2",
        "attribution_method": "outstanding_amount / (debt + equity)",
        "typical_risk": "high",
        "finz_required": True,
        "near_term_required": True,
        "long_term_required": True,
    },
}

# PCAF data quality descriptions per score level.
PCAF_DQ_DESCRIPTIONS: Dict[int, Dict[str, str]] = {
    1: {
        "name": "Verified",
        "description": "Audited, verified emissions data from the entity",
        "data_source": "Entity-specific, third-party verified",
        "uncertainty": "Very low",
    },
    2: {
        "name": "Reported",
        "description": "Reported but unverified emissions data",
        "data_source": "Entity-specific, self-reported",
        "uncertainty": "Low",
    },
    3: {
        "name": "Physical Activity",
        "description": "Estimated using physical activity data and emission factors",
        "data_source": "Asset-level, activity-based",
        "uncertainty": "Medium",
    },
    4: {
        "name": "Economic Activity",
        "description": "Estimated using economic/financial activity data",
        "data_source": "Revenue or asset-based estimation",
        "uncertainty": "High",
    },
    5: {
        "name": "Sector Average",
        "description": "Estimated using sector-level average data",
        "data_source": "Sector averages, proxy data",
        "uncertainty": "Very high",
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class PortfolioEntityInput(BaseModel):
    """Input data for a single entity in the portfolio.

    Attributes:
        entity_id: Unique entity identifier.
        entity_name: Entity or counterparty name.
        asset_class: Asset class classification.
        outstanding_amount: Outstanding exposure amount (currency units).
        entity_total_value: Entity's total value (debt + equity).
        financed_emissions_tco2e: Attributed financed emissions (tCO2e).
        entity_revenue: Entity annual revenue (for intensity).
        pcaf_data_quality: PCAF data quality score (1-5).
        has_sbti_target: Whether entity has validated SBTi target.
        sbti_committed: Whether entity is committed to SBTi.
        engagement_status: Current engagement status.
        entity_sector: Entity's sector classification.
        entity_temperature_score: Entity's implied temperature score (C).
        entity_scope1_tco2e: Entity's Scope 1 emissions.
        entity_scope2_tco2e: Entity's Scope 2 emissions.
        entity_scope3_tco2e: Entity's Scope 3 emissions.
        notes: Additional notes.
    """
    entity_id: str = Field(
        default_factory=_new_uuid,
        description="Unique entity identifier"
    )
    entity_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Entity or counterparty name"
    )
    asset_class: str = Field(
        ..., description="Asset class classification"
    )
    outstanding_amount: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Outstanding exposure (currency units)"
    )
    entity_total_value: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Entity total value (debt + equity)"
    )
    financed_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Attributed financed emissions (tCO2e)"
    )
    entity_revenue: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Entity annual revenue"
    )
    pcaf_data_quality: int = Field(
        default=5, ge=1, le=5,
        description="PCAF data quality score (1=best, 5=worst)"
    )
    has_sbti_target: bool = Field(
        default=False,
        description="Entity has validated SBTi target"
    )
    sbti_committed: bool = Field(
        default=False,
        description="Entity is committed to SBTi"
    )
    engagement_status: str = Field(
        default=EngagementStatus.NOT_ENGAGED.value,
        description="Current engagement status"
    )
    entity_sector: str = Field(
        default="general",
        description="Entity's sector classification"
    )
    entity_temperature_score: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Entity implied temperature score (degrees C)"
    )
    entity_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Entity Scope 1 emissions"
    )
    entity_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Entity Scope 2 emissions"
    )
    entity_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Entity Scope 3 emissions"
    )
    notes: str = Field(
        default="",
        description="Additional notes"
    )

    @field_validator("asset_class")
    @classmethod
    def validate_asset_class(cls, v: str) -> str:
        """Validate asset class is known."""
        valid = {a.value for a in AssetClass}
        if v not in valid:
            raise ValueError(
                f"Unknown asset class '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("engagement_status")
    @classmethod
    def validate_engagement_status(cls, v: str) -> str:
        """Validate engagement status."""
        valid = {e.value for e in EngagementStatus}
        if v not in valid:
            raise ValueError(
                f"Unknown engagement status '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

class PortfolioTargetInput(BaseModel):
    """Input for a portfolio-level target.

    Attributes:
        asset_class: Asset class for this target.
        methodology: Target-setting methodology.
        base_year: Target base year.
        target_year: Target year.
        target_value: Target value (depends on methodology).
        target_unit: Unit for target value (%, tCO2e, C, etc.).
        current_value: Current value for progress tracking.
    """
    asset_class: str = Field(
        ..., description="Asset class for this target"
    )
    methodology: str = Field(
        default=TargetMethodology.NOT_SET.value,
        description="Target-setting methodology"
    )
    base_year: int = Field(
        default=0, ge=0, le=2030,
        description="Target base year"
    )
    target_year: int = Field(
        default=0, ge=0, le=2060,
        description="Target year"
    )
    target_value: Decimal = Field(
        default=Decimal("0"),
        description="Target value"
    )
    target_unit: str = Field(
        default="",
        description="Unit for target value"
    )
    current_value: Decimal = Field(
        default=Decimal("0"),
        description="Current value for progress"
    )

    @field_validator("asset_class")
    @classmethod
    def validate_asset_class(cls, v: str) -> str:
        """Validate asset class is known."""
        valid = {a.value for a in AssetClass}
        if v not in valid:
            raise ValueError(
                f"Unknown asset class '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("methodology")
    @classmethod
    def validate_methodology(cls, v: str) -> str:
        """Validate methodology is known."""
        valid = {m.value for m in TargetMethodology}
        if v not in valid:
            raise ValueError(
                f"Unknown methodology '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

class FIPortfolioInput(BaseModel):
    """Complete FI portfolio analysis input.

    Attributes:
        institution_name: Financial institution name.
        base_year: Portfolio base year.
        reporting_year: Current reporting year.
        total_portfolio_value: Total portfolio value.
        currency: Reporting currency (ISO 4217).
        entities: Portfolio entity data (up to 500).
        targets: Portfolio-level targets per asset class.
        engagement_target_pct: Target % of portfolio engaged.
        engagement_target_year: Year for engagement target.
        include_temperature_alignment: Assess temperature alignment.
        include_coverage_analysis: Perform coverage analysis.
        include_engagement_tracking: Track engagement progress.
        include_data_quality_assessment: Assess PCAF data quality.
        include_recommendations: Generate recommendations.
    """
    institution_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Financial institution name"
    )
    base_year: int = Field(
        ..., ge=2015, le=2030,
        description="Portfolio base year"
    )
    reporting_year: int = Field(
        default=0, ge=0, le=2030,
        description="Current reporting year (0 = auto)"
    )
    total_portfolio_value: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total portfolio value"
    )
    currency: str = Field(
        default="USD", max_length=3,
        description="Reporting currency (ISO 4217)"
    )
    entities: List[PortfolioEntityInput] = Field(
        default_factory=list,
        description="Portfolio entities (max 500)"
    )
    targets: List[PortfolioTargetInput] = Field(
        default_factory=list,
        description="Portfolio-level targets"
    )
    engagement_target_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Target % of portfolio engaged"
    )
    engagement_target_year: int = Field(
        default=0, ge=0, le=2060,
        description="Year for engagement target"
    )
    include_temperature_alignment: bool = Field(
        default=True,
        description="Assess temperature alignment"
    )
    include_coverage_analysis: bool = Field(
        default=True,
        description="Perform portfolio coverage analysis"
    )
    include_engagement_tracking: bool = Field(
        default=True,
        description="Track engagement progress"
    )
    include_data_quality_assessment: bool = Field(
        default=True,
        description="Assess PCAF data quality"
    )
    include_recommendations: bool = Field(
        default=True,
        description="Generate recommendations"
    )

    @field_validator("reporting_year")
    @classmethod
    def validate_reporting_year(cls, v: int, info: Any) -> int:
        """Default reporting year to base year if zero."""
        if v == 0:
            base = info.data.get("base_year", 2023)
            return base
        return v

    @field_validator("entities")
    @classmethod
    def validate_entities_count(cls, v: List[PortfolioEntityInput]) -> List[PortfolioEntityInput]:
        """Validate entity count within limits."""
        if len(v) > MAX_PORTFOLIO_ENTITIES:
            raise ValueError(
                f"Maximum {MAX_PORTFOLIO_ENTITIES} entities supported. "
                f"Received {len(v)}."
            )
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class AssetClassSummary(BaseModel):
    """Summary metrics for a single asset class.

    Attributes:
        asset_class: Asset class identifier.
        asset_class_name: Human-readable name.
        entity_count: Number of entities in this class.
        total_outstanding: Total outstanding exposure.
        total_financed_emissions_tco2e: Total financed emissions.
        pct_of_portfolio: Percentage of total portfolio.
        pct_of_financed_emissions: Percentage of total financed emissions.
        weighted_pcaf_score: Emissions-weighted PCAF data quality.
        entities_with_sbti: Count with SBTi targets.
        entities_committed: Count committed to SBTi.
        entities_engaged: Count actively engaged.
        coverage_pct: Portfolio coverage percentage.
        engagement_pct: Engagement percentage.
        weighted_temperature_score: Weighted temperature score (C).
        temperature_alignment: Temperature alignment classification.
        eligible_methodologies: Eligible target-setting methods.
        target_set: Whether a target is set for this class.
        target_methodology: Methodology used (if target set).
        is_finz_required: Whether FINZ target is required.
    """
    asset_class: str = Field(default="")
    asset_class_name: str = Field(default="")
    entity_count: int = Field(default=0)
    total_outstanding: Decimal = Field(default=Decimal("0"))
    total_financed_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    pct_of_portfolio: Decimal = Field(default=Decimal("0"))
    pct_of_financed_emissions: Decimal = Field(default=Decimal("0"))
    weighted_pcaf_score: Decimal = Field(default=Decimal("0"))
    entities_with_sbti: int = Field(default=0)
    entities_committed: int = Field(default=0)
    entities_engaged: int = Field(default=0)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    engagement_pct: Decimal = Field(default=Decimal("0"))
    weighted_temperature_score: Decimal = Field(default=Decimal("0"))
    temperature_alignment: str = Field(
        default=TemperatureAlignment.NOT_ASSESSED.value
    )
    eligible_methodologies: List[str] = Field(default_factory=list)
    target_set: bool = Field(default=False)
    target_methodology: str = Field(default=TargetMethodology.NOT_SET.value)
    is_finz_required: bool = Field(default=False)

class PortfolioCoverageResult(BaseModel):
    """Portfolio coverage assessment result.

    Attributes:
        total_financed_emissions_tco2e: Total portfolio financed emissions.
        covered_emissions_tco2e: Emissions from entities with SBTi targets.
        committed_emissions_tco2e: Emissions from committed entities.
        coverage_pct: Coverage percentage (targets only).
        coverage_incl_committed_pct: Coverage including committed.
        near_term_required_pct: SBTi near-term threshold (67%).
        long_term_required_pct: SBTi long-term threshold (90%).
        meets_near_term: Whether near-term coverage is met.
        meets_long_term: Whether long-term coverage is met.
        gap_to_near_term_pct: Gap to near-term threshold.
        gap_to_long_term_pct: Gap to long-term threshold.
        total_entities: Total entities in portfolio.
        entities_with_targets: Entities with SBTi targets.
        entities_committed: Entities committed to SBTi.
        entity_coverage_pct: Entity-count based coverage.
        coverage_status: Coverage status classification.
        message: Human-readable coverage assessment.
    """
    total_financed_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    covered_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    committed_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    coverage_pct: Decimal = Field(default=Decimal("0"))
    coverage_incl_committed_pct: Decimal = Field(default=Decimal("0"))
    near_term_required_pct: Decimal = Field(default=Decimal("67.0"))
    long_term_required_pct: Decimal = Field(default=Decimal("90.0"))
    meets_near_term: bool = Field(default=False)
    meets_long_term: bool = Field(default=False)
    gap_to_near_term_pct: Decimal = Field(default=Decimal("0"))
    gap_to_long_term_pct: Decimal = Field(default=Decimal("0"))
    total_entities: int = Field(default=0)
    entities_with_targets: int = Field(default=0)
    entities_committed: int = Field(default=0)
    entity_coverage_pct: Decimal = Field(default=Decimal("0"))
    coverage_status: str = Field(default=PortfolioCoverageStatus.NONE.value)
    message: str = Field(default="")

class EngagementTrackingResult(BaseModel):
    """Engagement tracking assessment.

    Attributes:
        total_entities: Total entities in portfolio.
        entities_engaged: Entities with active engagement.
        entities_target_set: Entities with SBTi targets.
        entities_committed: Entities committed to SBTi.
        entities_escalated: Entities in escalation.
        entities_not_engaged: Entities not yet engaged.
        entities_divested: Entities divested.
        engagement_pct: Engagement percentage by emissions.
        engagement_by_count_pct: Engagement percentage by entity count.
        target_pct: Target engagement percentage.
        target_year: Target year for engagement.
        meets_target: Whether engagement target is met.
        gap_to_target_pct: Gap to engagement target.
        emissions_engaged_tco2e: Emissions covered by engagement.
        emissions_not_engaged_tco2e: Emissions not covered.
        top_engagement_priorities: Entities to prioritise.
        message: Human-readable engagement summary.
    """
    total_entities: int = Field(default=0)
    entities_engaged: int = Field(default=0)
    entities_target_set: int = Field(default=0)
    entities_committed: int = Field(default=0)
    entities_escalated: int = Field(default=0)
    entities_not_engaged: int = Field(default=0)
    entities_divested: int = Field(default=0)
    engagement_pct: Decimal = Field(default=Decimal("0"))
    engagement_by_count_pct: Decimal = Field(default=Decimal("0"))
    target_pct: Decimal = Field(default=Decimal("0"))
    target_year: int = Field(default=0)
    meets_target: bool = Field(default=False)
    gap_to_target_pct: Decimal = Field(default=Decimal("0"))
    emissions_engaged_tco2e: Decimal = Field(default=Decimal("0"))
    emissions_not_engaged_tco2e: Decimal = Field(default=Decimal("0"))
    top_engagement_priorities: List[str] = Field(default_factory=list)
    message: str = Field(default="")

class PcafDataQualityAssessment(BaseModel):
    """PCAF data quality assessment across the portfolio.

    Attributes:
        portfolio_weighted_score: Emissions-weighted PCAF score.
        simple_average_score: Simple average PCAF score.
        score_distribution: Count of entities at each score level.
        emissions_by_score: Financed emissions at each score level.
        pct_at_score_1_2: Percentage at high quality (1-2).
        pct_at_score_3: Percentage at medium quality (3).
        pct_at_score_4_5: Percentage at low quality (4-5).
        meets_target_quality: Whether portfolio meets target.
        target_score: Target PCAF score.
        improvement_priorities: Entities to improve first.
        message: Human-readable quality assessment.
    """
    portfolio_weighted_score: Decimal = Field(default=Decimal("0"))
    simple_average_score: Decimal = Field(default=Decimal("0"))
    score_distribution: Dict[str, int] = Field(default_factory=dict)
    emissions_by_score: Dict[str, Decimal] = Field(default_factory=dict)
    pct_at_score_1_2: Decimal = Field(default=Decimal("0"))
    pct_at_score_3: Decimal = Field(default=Decimal("0"))
    pct_at_score_4_5: Decimal = Field(default=Decimal("0"))
    meets_target_quality: bool = Field(default=False)
    target_score: int = Field(default=PCAF_TARGET_DQ)
    improvement_priorities: List[str] = Field(default_factory=list)
    message: str = Field(default="")

class TemperatureAlignmentResult(BaseModel):
    """Temperature alignment assessment for the portfolio.

    Attributes:
        portfolio_temperature_score: Weighted average temperature (C).
        temperature_alignment: Alignment classification.
        entities_below_1_5c: Count below 1.5C.
        entities_at_1_5c: Count at 1.5C.
        entities_between_1_5_2c: Count between 1.5 and 2C.
        entities_at_2c: Count at 2C.
        entities_above_2c: Count above 2C.
        entities_not_assessed: Count not assessed.
        emissions_weighted_temp: Emissions-weighted temperature.
        hotspot_entities: Entities with highest temperature scores.
        pct_aligned_1_5c: Percentage aligned to 1.5C.
        pct_aligned_2c: Percentage aligned to 2C.
        gap_to_1_5c: Gap to 1.5C alignment (C).
        by_asset_class: Temperature by asset class.
        message: Human-readable temperature assessment.
    """
    portfolio_temperature_score: Decimal = Field(default=Decimal("0"))
    temperature_alignment: str = Field(
        default=TemperatureAlignment.NOT_ASSESSED.value
    )
    entities_below_1_5c: int = Field(default=0)
    entities_at_1_5c: int = Field(default=0)
    entities_between_1_5_2c: int = Field(default=0)
    entities_at_2c: int = Field(default=0)
    entities_above_2c: int = Field(default=0)
    entities_not_assessed: int = Field(default=0)
    emissions_weighted_temp: Decimal = Field(default=Decimal("0"))
    hotspot_entities: List[str] = Field(default_factory=list)
    pct_aligned_1_5c: Decimal = Field(default=Decimal("0"))
    pct_aligned_2c: Decimal = Field(default=Decimal("0"))
    gap_to_1_5c: Decimal = Field(default=Decimal("0"))
    by_asset_class: Dict[str, Decimal] = Field(default_factory=dict)
    message: str = Field(default="")

class FIRecommendation(BaseModel):
    """A single FI-specific recommendation.

    Attributes:
        recommendation_id: Unique recommendation identifier.
        priority: Priority level (immediate/short/medium/long).
        category: Recommendation category.
        asset_class: Related asset class (or 'all').
        action: Description of recommended action.
        rationale: Why this action is recommended.
        estimated_impact: Expected impact description.
        timeline_months: Estimated implementation time.
    """
    recommendation_id: str = Field(default_factory=_new_uuid)
    priority: str = Field(default="medium_term")
    category: str = Field(default="general")
    asset_class: str = Field(default="all")
    action: str = Field(default="")
    rationale: str = Field(default="")
    estimated_impact: str = Field(default="")
    timeline_months: int = Field(default=12)

class FIPortfolioResult(BaseModel):
    """Complete FI portfolio analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        institution_name: Institution name.
        base_year: Portfolio base year.
        total_entities: Total portfolio entities analysed.
        total_financed_emissions_tco2e: Total financed emissions.
        total_portfolio_value: Total portfolio value.
        currency: Reporting currency.
        asset_class_summaries: Per-asset-class summaries.
        coverage: Portfolio coverage assessment.
        engagement: Engagement tracking.
        data_quality: PCAF data quality assessment.
        temperature: Temperature alignment assessment.
        recommendations: FI-specific recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    institution_name: str = Field(default="")
    base_year: int = Field(default=0)
    total_entities: int = Field(default=0)
    total_financed_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_portfolio_value: Decimal = Field(default=Decimal("0"))
    currency: str = Field(default="USD")
    asset_class_summaries: List[AssetClassSummary] = Field(
        default_factory=list
    )
    coverage: Optional[PortfolioCoverageResult] = Field(None)
    engagement: Optional[EngagementTrackingResult] = Field(None)
    data_quality: Optional[PcafDataQualityAssessment] = Field(None)
    temperature: Optional[TemperatureAlignmentResult] = Field(None)
    recommendations: List[FIRecommendation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FIPortfolioEngine:
    """SBTi Financial Institutions portfolio target engine.

    Performs FINZ V1.0 portfolio analysis including:
      - 8 asset class segmentation and metrics
      - PCAF data quality scoring (1-5 scale)
      - Portfolio coverage calculation (67%/90% thresholds)
      - Engagement target tracking and gap analysis
      - Temperature alignment per asset class and portfolio
      - Financed emissions attribution and analysis

    All calculations use deterministic Decimal arithmetic with SHA-256
    provenance hashing.  No LLM involvement in any calculation path.

    Usage::

        engine = FIPortfolioEngine()
        result = engine.analyse(input_data)
        print(f"Coverage: {result.coverage.coverage_pct}%")
        for ac in result.asset_class_summaries:
            print(f"  {ac.asset_class_name}: {ac.total_financed_emissions_tco2e}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise FIPortfolioEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - coverage_nt_min_pct (Decimal)
                - coverage_lt_min_pct (Decimal)
                - pca_engagement_min_pct (Decimal)
                - pcaf_target_dq (int)
                - temp_1_5c_threshold (Decimal)
        """
        self.config = config or {}
        self._coverage_nt = _decimal(
            self.config.get("coverage_nt_min_pct", PORTFOLIO_COVERAGE_NT_MIN_PCT)
        )
        self._coverage_lt = _decimal(
            self.config.get("coverage_lt_min_pct", PORTFOLIO_COVERAGE_LT_MIN_PCT)
        )
        self._pca_min = _decimal(
            self.config.get("pca_engagement_min_pct", PCA_ENGAGEMENT_MIN_PCT)
        )
        self._pcaf_target = int(
            self.config.get("pcaf_target_dq", PCAF_TARGET_DQ)
        )
        self._temp_1_5c = _decimal(
            self.config.get("temp_1_5c_threshold", TEMP_RATING_1_5C)
        )
        logger.info(
            "FIPortfolioEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyse(self, data: FIPortfolioInput) -> FIPortfolioResult:
        """Perform complete FI portfolio analysis.

        Orchestrates the full portfolio pipeline: segments by asset
        class, calculates coverage, tracks engagement, assesses
        PCAF data quality, evaluates temperature alignment, and
        produces prioritised recommendations.

        Args:
            data: Validated portfolio input.

        Returns:
            FIPortfolioResult with all assessments.
        """
        t0 = time.perf_counter()
        logger.info(
            "FI portfolio analysis: institution=%s, entities=%d, base=%d",
            data.institution_name, len(data.entities), data.base_year,
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Validate minimum data
        if len(data.entities) < MIN_PORTFOLIO_ENTITIES:
            warnings.append(
                f"Portfolio has {len(data.entities)} entities, below "
                f"the recommended minimum of {MIN_PORTFOLIO_ENTITIES}."
            )
        if len(data.entities) == 0:
            warnings.append(
                "No portfolio entities provided. Analysis will be limited."
            )

        # Step 2: Calculate totals
        total_fe = sum(
            (e.financed_emissions_tco2e for e in data.entities),
            Decimal("0"),
        )
        total_outstanding = sum(
            (e.outstanding_amount for e in data.entities),
            Decimal("0"),
        )
        portfolio_value = data.total_portfolio_value
        if portfolio_value <= Decimal("0"):
            portfolio_value = total_outstanding

        # Step 3: Build asset class summaries
        ac_summaries = self._build_asset_class_summaries(
            data.entities, data.targets, total_fe, portfolio_value
        )

        # Step 4: Portfolio coverage
        coverage: Optional[PortfolioCoverageResult] = None
        if data.include_coverage_analysis:
            coverage = self._assess_coverage(data.entities, total_fe)

        # Step 5: Engagement tracking
        engagement: Optional[EngagementTrackingResult] = None
        if data.include_engagement_tracking:
            engagement = self._track_engagement(
                data.entities, total_fe,
                data.engagement_target_pct,
                data.engagement_target_year,
            )

        # Step 6: PCAF data quality
        dq: Optional[PcafDataQualityAssessment] = None
        if data.include_data_quality_assessment:
            dq = self._assess_data_quality(data.entities, total_fe)

        # Step 7: Temperature alignment
        temperature: Optional[TemperatureAlignmentResult] = None
        if data.include_temperature_alignment:
            temperature = self._assess_temperature(
                data.entities, total_fe, portfolio_value
            )

        # Step 8: Recommendations
        recommendations: List[FIRecommendation] = []
        if data.include_recommendations:
            recommendations = self._generate_recommendations(
                ac_summaries, coverage, engagement, dq,
                temperature, data,
            )

        # Step 9: Warnings for FINZ-required asset classes without targets
        for acs in ac_summaries:
            if acs.is_finz_required and not acs.target_set:
                warnings.append(
                    f"FINZ-required asset class '{acs.asset_class_name}' "
                    f"does not have a target set."
                )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = FIPortfolioResult(
            institution_name=data.institution_name,
            base_year=data.base_year,
            total_entities=len(data.entities),
            total_financed_emissions_tco2e=_round_val(total_fe, 2),
            total_portfolio_value=_round_val(portfolio_value, 2),
            currency=data.currency,
            asset_class_summaries=ac_summaries,
            coverage=coverage,
            engagement=engagement,
            data_quality=dq,
            temperature=temperature,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "FI portfolio analysis complete: entities=%d, FE=%.0f tCO2e, "
            "coverage=%.1f%%, PCAF=%.1f, temp=%.1fC, hash=%s",
            len(data.entities),
            float(total_fe),
            float(coverage.coverage_pct) if coverage else 0.0,
            float(dq.portfolio_weighted_score) if dq else 0.0,
            float(temperature.portfolio_temperature_score) if temperature else 0.0,
            result.provenance_hash[:16],
        )
        return result

    def calculate_attribution_factor(
        self,
        outstanding_amount: Decimal,
        entity_total_value: Decimal,
    ) -> Decimal:
        """Calculate PCAF attribution factor.

        Attribution factor = outstanding_amount / entity_total_value
        Per PCAF Global Standard V3.0.

        Args:
            outstanding_amount: Outstanding exposure.
            entity_total_value: Entity total value (debt + equity).

        Returns:
            Attribution factor (0-1 range).
        """
        factor = _safe_divide(outstanding_amount, entity_total_value)
        return _round_val(min(factor, Decimal("1")), 6)

    def calculate_financed_emissions(
        self,
        attribution_factor: Decimal,
        entity_emissions_tco2e: Decimal,
    ) -> Decimal:
        """Calculate attributed financed emissions.

        FE = attribution_factor * entity_emissions
        Per PCAF Global Standard V3.0.

        Args:
            attribution_factor: PCAF attribution factor.
            entity_emissions_tco2e: Entity's total emissions.

        Returns:
            Attributed financed emissions (tCO2e).
        """
        return _round_val(attribution_factor * entity_emissions_tco2e, 4)

    def get_asset_class_reference(self) -> List[Dict[str, Any]]:
        """Return reference data for all 8 asset classes.

        Returns:
            List of dicts with asset class definitions.
        """
        result: List[Dict[str, Any]] = []
        for ac_enum in AssetClass:
            ref = ASSET_CLASS_REFERENCE.get(ac_enum.value, {})
            result.append({
                "asset_class": ac_enum.value,
                "name": ref.get("name", ""),
                "description": ref.get("description", ""),
                "eligible_methodologies": ref.get("eligible_methodologies", []),
                "pcaf_coverage": ref.get("pcaf_coverage", ""),
                "attribution_method": ref.get("attribution_method", ""),
                "finz_required": ref.get("finz_required", False),
            })
        return result

    def get_pcaf_quality_reference(self) -> Dict[int, Dict[str, str]]:
        """Return PCAF data quality tier definitions.

        Returns:
            Dict mapping score (1-5) to description.
        """
        return PCAF_DQ_DESCRIPTIONS

    # ------------------------------------------------------------------ #
    # Internal: Asset Class Summaries                                      #
    # ------------------------------------------------------------------ #

    def _build_asset_class_summaries(
        self,
        entities: List[PortfolioEntityInput],
        targets: List[PortfolioTargetInput],
        total_fe: Decimal,
        portfolio_value: Decimal,
    ) -> List[AssetClassSummary]:
        """Build per-asset-class summary metrics.

        Args:
            entities: Portfolio entities.
            targets: Portfolio targets.
            total_fe: Total financed emissions.
            portfolio_value: Total portfolio value.

        Returns:
            List of AssetClassSummary objects.
        """
        # Group entities by asset class
        ac_entities: Dict[str, List[PortfolioEntityInput]] = {}
        for e in entities:
            ac_entities.setdefault(e.asset_class, []).append(e)

        # Build target lookup
        target_lookup: Dict[str, PortfolioTargetInput] = {}
        for t in targets:
            target_lookup[t.asset_class] = t

        summaries: List[AssetClassSummary] = []

        for ac_enum in AssetClass:
            ac = ac_enum.value
            ref = ASSET_CLASS_REFERENCE.get(ac, {})
            ents = ac_entities.get(ac, [])

            if not ents:
                # Include empty asset class only if FINZ-required
                if ref.get("finz_required", False):
                    summaries.append(AssetClassSummary(
                        asset_class=ac,
                        asset_class_name=ref.get("name", ac),
                        entity_count=0,
                        eligible_methodologies=ref.get("eligible_methodologies", []),
                        is_finz_required=ref.get("finz_required", False),
                    ))
                continue

            ac_outstanding = sum(
                (e.outstanding_amount for e in ents), Decimal("0")
            )
            ac_fe = sum(
                (e.financed_emissions_tco2e for e in ents), Decimal("0")
            )
            ac_sbti = sum(1 for e in ents if e.has_sbti_target)
            ac_committed = sum(1 for e in ents if e.sbti_committed)
            ac_engaged = sum(
                1 for e in ents
                if e.engagement_status in (
                    EngagementStatus.ACTIVE.value,
                    EngagementStatus.COMMITTED.value,
                    EngagementStatus.TARGET_SET.value,
                )
            )

            # Portfolio percentage
            pct_portfolio = _safe_pct(ac_outstanding, portfolio_value)
            pct_fe = _safe_pct(ac_fe, total_fe)

            # Coverage: emissions from entities with SBTi targets
            covered_fe = sum(
                (e.financed_emissions_tco2e for e in ents if e.has_sbti_target),
                Decimal("0"),
            )
            cov_pct = _safe_pct(covered_fe, ac_fe)

            # Engagement
            engaged_fe = sum(
                (
                    e.financed_emissions_tco2e for e in ents
                    if e.engagement_status in (
                        EngagementStatus.ACTIVE.value,
                        EngagementStatus.COMMITTED.value,
                        EngagementStatus.TARGET_SET.value,
                    )
                ),
                Decimal("0"),
            )
            eng_pct = _safe_pct(engaged_fe, ac_fe)

            # PCAF weighted quality
            dq_weighted_sum = Decimal("0")
            dq_weight_total = Decimal("0")
            for e in ents:
                if e.financed_emissions_tco2e > Decimal("0"):
                    dq_weighted_sum += (
                        _decimal(e.pcaf_data_quality)
                        * e.financed_emissions_tco2e
                    )
                    dq_weight_total += e.financed_emissions_tco2e
            weighted_dq = _safe_divide(dq_weighted_sum, dq_weight_total)

            # Temperature: weighted average
            temp_weighted_sum = Decimal("0")
            temp_weight_total = Decimal("0")
            for e in ents:
                if (
                    e.entity_temperature_score > Decimal("0")
                    and e.outstanding_amount > Decimal("0")
                ):
                    temp_weighted_sum += (
                        e.entity_temperature_score * e.outstanding_amount
                    )
                    temp_weight_total += e.outstanding_amount
            weighted_temp = _safe_divide(temp_weighted_sum, temp_weight_total)
            temp_align = self._classify_temperature(weighted_temp)

            # Target info
            target = target_lookup.get(ac)
            target_set = target is not None and target.methodology != TargetMethodology.NOT_SET.value
            target_method = target.methodology if target else TargetMethodology.NOT_SET.value

            summaries.append(AssetClassSummary(
                asset_class=ac,
                asset_class_name=ref.get("name", ac),
                entity_count=len(ents),
                total_outstanding=_round_val(ac_outstanding, 2),
                total_financed_emissions_tco2e=_round_val(ac_fe, 2),
                pct_of_portfolio=_round_val(pct_portfolio, 2),
                pct_of_financed_emissions=_round_val(pct_fe, 2),
                weighted_pcaf_score=_round_val(weighted_dq, 2),
                entities_with_sbti=ac_sbti,
                entities_committed=ac_committed,
                entities_engaged=ac_engaged,
                coverage_pct=_round_val(cov_pct, 2),
                engagement_pct=_round_val(eng_pct, 2),
                weighted_temperature_score=_round_val(weighted_temp, 2),
                temperature_alignment=temp_align,
                eligible_methodologies=ref.get("eligible_methodologies", []),
                target_set=target_set,
                target_methodology=target_method,
                is_finz_required=ref.get("finz_required", False),
            ))

        # Sort by financed emissions descending
        summaries.sort(
            key=lambda s: s.total_financed_emissions_tco2e, reverse=True
        )

        return summaries

    # ------------------------------------------------------------------ #
    # Internal: Portfolio Coverage                                         #
    # ------------------------------------------------------------------ #

    def _assess_coverage(
        self,
        entities: List[PortfolioEntityInput],
        total_fe: Decimal,
    ) -> PortfolioCoverageResult:
        """Assess portfolio coverage against SBTi thresholds.

        Coverage is measured as the percentage of total financed
        emissions from entities that have validated SBTi targets.

        Args:
            entities: Portfolio entities.
            total_fe: Total financed emissions.

        Returns:
            PortfolioCoverageResult with gap analysis.
        """
        covered_fe = sum(
            (e.financed_emissions_tco2e for e in entities if e.has_sbti_target),
            Decimal("0"),
        )
        committed_fe = sum(
            (e.financed_emissions_tco2e for e in entities if e.sbti_committed),
            Decimal("0"),
        )

        cov_pct = _round_val(_safe_pct(covered_fe, total_fe), 2)
        cov_incl_pct = _round_val(
            _safe_pct(covered_fe + committed_fe, total_fe), 2
        )

        nt_pct = self._coverage_nt
        lt_pct = self._coverage_lt

        meets_nt = cov_pct >= nt_pct
        meets_lt = cov_pct >= lt_pct

        gap_nt = _round_val(max(Decimal("0"), nt_pct - cov_pct), 2)
        gap_lt = _round_val(max(Decimal("0"), lt_pct - cov_pct), 2)

        total_ents = len(entities)
        ents_target = sum(1 for e in entities if e.has_sbti_target)
        ents_committed = sum(1 for e in entities if e.sbti_committed)
        ent_cov_pct = _round_val(
            _safe_pct(_decimal(ents_target), _decimal(total_ents)), 2
        )

        # Coverage status
        if cov_pct >= lt_pct:
            status = PortfolioCoverageStatus.FULL.value
        elif cov_pct >= nt_pct:
            status = PortfolioCoverageStatus.PARTIAL.value
        elif cov_pct >= MIN_COVERAGE_FOR_TARGET_PCT:
            status = PortfolioCoverageStatus.MINIMAL.value
        else:
            status = PortfolioCoverageStatus.NONE.value

        if meets_lt:
            msg = (
                f"Portfolio coverage of {cov_pct}% meets both near-term "
                f"({nt_pct}%) and long-term ({lt_pct}%) SBTi thresholds."
            )
        elif meets_nt:
            msg = (
                f"Portfolio coverage of {cov_pct}% meets near-term "
                f"({nt_pct}%) but not long-term ({lt_pct}%). "
                f"Gap to long-term: {gap_lt}%."
            )
        else:
            msg = (
                f"Portfolio coverage of {cov_pct}% is below near-term "
                f"threshold ({nt_pct}%). Gap: {gap_nt}%. "
                f"Including committed entities: {cov_incl_pct}%."
            )

        return PortfolioCoverageResult(
            total_financed_emissions_tco2e=_round_val(total_fe, 2),
            covered_emissions_tco2e=_round_val(covered_fe, 2),
            committed_emissions_tco2e=_round_val(committed_fe, 2),
            coverage_pct=cov_pct,
            coverage_incl_committed_pct=cov_incl_pct,
            near_term_required_pct=nt_pct,
            long_term_required_pct=lt_pct,
            meets_near_term=meets_nt,
            meets_long_term=meets_lt,
            gap_to_near_term_pct=gap_nt,
            gap_to_long_term_pct=gap_lt,
            total_entities=total_ents,
            entities_with_targets=ents_target,
            entities_committed=ents_committed,
            entity_coverage_pct=ent_cov_pct,
            coverage_status=status,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    # Internal: Engagement Tracking                                        #
    # ------------------------------------------------------------------ #

    def _track_engagement(
        self,
        entities: List[PortfolioEntityInput],
        total_fe: Decimal,
        target_pct: Decimal,
        target_year: int,
    ) -> EngagementTrackingResult:
        """Track portfolio engagement progress.

        Args:
            entities: Portfolio entities.
            total_fe: Total financed emissions.
            target_pct: Engagement target percentage.
            target_year: Engagement target year.

        Returns:
            EngagementTrackingResult with progress metrics.
        """
        target_set_count = 0
        committed_count = 0
        active_count = 0
        escalated_count = 0
        not_engaged_count = 0
        divested_count = 0

        engaged_emissions = Decimal("0")

        for e in entities:
            if e.engagement_status == EngagementStatus.TARGET_SET.value:
                target_set_count += 1
                engaged_emissions += e.financed_emissions_tco2e
            elif e.engagement_status == EngagementStatus.COMMITTED.value:
                committed_count += 1
                engaged_emissions += e.financed_emissions_tco2e
            elif e.engagement_status == EngagementStatus.ACTIVE.value:
                active_count += 1
                engaged_emissions += e.financed_emissions_tco2e
            elif e.engagement_status == EngagementStatus.ESCALATION.value:
                escalated_count += 1
            elif e.engagement_status == EngagementStatus.DIVESTED.value:
                divested_count += 1
            else:
                not_engaged_count += 1

        total_engaged = target_set_count + committed_count + active_count
        not_engaged_emissions = total_fe - engaged_emissions

        eng_pct = _round_val(_safe_pct(engaged_emissions, total_fe), 2)
        total_ents = len(entities)
        eng_count_pct = _round_val(
            _safe_pct(_decimal(total_engaged), _decimal(total_ents)), 2
        )

        eff_target = target_pct if target_pct > Decimal("0") else self._pca_min
        meets = eng_pct >= eff_target
        gap = _round_val(max(Decimal("0"), eff_target - eng_pct), 2)

        # Top priorities: largest not-engaged entities by emissions
        not_engaged_ents = [
            e for e in entities
            if e.engagement_status == EngagementStatus.NOT_ENGAGED.value
        ]
        not_engaged_ents.sort(
            key=lambda e: e.financed_emissions_tco2e, reverse=True
        )
        top_priorities = [
            e.entity_name for e in not_engaged_ents[:10]
        ]

        if meets:
            msg = (
                f"Engagement target met: {eng_pct}% of portfolio "
                f"emissions are engaged (target: {eff_target}%)."
            )
        else:
            msg = (
                f"Engagement at {eng_pct}% of portfolio emissions, "
                f"below target of {eff_target}%. Gap: {gap}%. "
                f"{not_engaged_count} entities not yet engaged."
            )

        return EngagementTrackingResult(
            total_entities=total_ents,
            entities_engaged=total_engaged,
            entities_target_set=target_set_count,
            entities_committed=committed_count,
            entities_escalated=escalated_count,
            entities_not_engaged=not_engaged_count,
            entities_divested=divested_count,
            engagement_pct=eng_pct,
            engagement_by_count_pct=eng_count_pct,
            target_pct=eff_target,
            target_year=target_year,
            meets_target=meets,
            gap_to_target_pct=gap,
            emissions_engaged_tco2e=_round_val(engaged_emissions, 2),
            emissions_not_engaged_tco2e=_round_val(not_engaged_emissions, 2),
            top_engagement_priorities=top_priorities,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    # Internal: PCAF Data Quality                                          #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(
        self,
        entities: List[PortfolioEntityInput],
        total_fe: Decimal,
    ) -> PcafDataQualityAssessment:
        """Assess PCAF data quality across the portfolio.

        Args:
            entities: Portfolio entities.
            total_fe: Total financed emissions.

        Returns:
            PcafDataQualityAssessment with scores and priorities.
        """
        # Score distribution
        score_dist: Dict[str, int] = {
            "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
        }
        emissions_by_score: Dict[str, Decimal] = {
            "1": Decimal("0"), "2": Decimal("0"),
            "3": Decimal("0"), "4": Decimal("0"),
            "5": Decimal("0"),
        }

        dq_sum = Decimal("0")
        weighted_sum = Decimal("0")
        weighted_den = Decimal("0")

        for e in entities:
            score_key = str(e.pcaf_data_quality)
            score_dist[score_key] = score_dist.get(score_key, 0) + 1
            emissions_by_score[score_key] = (
                emissions_by_score.get(score_key, Decimal("0"))
                + e.financed_emissions_tco2e
            )
            dq_sum += _decimal(e.pcaf_data_quality)

            if e.financed_emissions_tco2e > Decimal("0"):
                weighted_sum += (
                    _decimal(e.pcaf_data_quality)
                    * e.financed_emissions_tco2e
                )
                weighted_den += e.financed_emissions_tco2e

        total_ents = len(entities)
        simple_avg = _safe_divide(dq_sum, _decimal(total_ents))
        weighted_avg = _safe_divide(weighted_sum, weighted_den)

        # Percentage at each quality tier
        ents_1_2 = score_dist.get("1", 0) + score_dist.get("2", 0)
        ents_3 = score_dist.get("3", 0)
        ents_4_5 = score_dist.get("4", 0) + score_dist.get("5", 0)

        pct_1_2 = _round_val(
            _safe_pct(_decimal(ents_1_2), _decimal(total_ents)), 2
        )
        pct_3 = _round_val(
            _safe_pct(_decimal(ents_3), _decimal(total_ents)), 2
        )
        pct_4_5 = _round_val(
            _safe_pct(_decimal(ents_4_5), _decimal(total_ents)), 2
        )

        meets_target = weighted_avg <= _decimal(self._pcaf_target)

        # Improvement priorities: largest entities with score 4-5
        low_quality = [
            e for e in entities if e.pcaf_data_quality >= 4
        ]
        low_quality.sort(
            key=lambda e: e.financed_emissions_tco2e, reverse=True
        )
        priorities = [e.entity_name for e in low_quality[:10]]

        if meets_target:
            msg = (
                f"Portfolio PCAF weighted score of "
                f"{_round_val(weighted_avg, 1)} meets the target "
                f"of {self._pcaf_target}. "
                f"{pct_1_2}% of entities at score 1-2 (high quality)."
            )
        else:
            msg = (
                f"Portfolio PCAF weighted score of "
                f"{_round_val(weighted_avg, 1)} exceeds the target "
                f"of {self._pcaf_target}. "
                f"{pct_4_5}% of entities at score 4-5 (low quality). "
                f"Data quality improvement needed."
            )

        return PcafDataQualityAssessment(
            portfolio_weighted_score=_round_val(weighted_avg, 2),
            simple_average_score=_round_val(simple_avg, 2),
            score_distribution=score_dist,
            emissions_by_score={
                k: _round_val(v, 2) for k, v in emissions_by_score.items()
            },
            pct_at_score_1_2=pct_1_2,
            pct_at_score_3=pct_3,
            pct_at_score_4_5=pct_4_5,
            meets_target_quality=meets_target,
            target_score=self._pcaf_target,
            improvement_priorities=priorities,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    # Internal: Temperature Alignment                                      #
    # ------------------------------------------------------------------ #

    def _assess_temperature(
        self,
        entities: List[PortfolioEntityInput],
        total_fe: Decimal,
        portfolio_value: Decimal,
    ) -> TemperatureAlignmentResult:
        """Assess temperature alignment across the portfolio.

        Calculates weighted average temperature score using portfolio
        weights (outstanding / total) and classifies alignment.

        Args:
            entities: Portfolio entities.
            total_fe: Total financed emissions.
            portfolio_value: Total portfolio value.

        Returns:
            TemperatureAlignmentResult with classification.
        """
        below_1_5 = 0
        at_1_5 = 0
        between_1_5_2 = 0
        at_2 = 0
        above_2 = 0
        not_assessed = 0

        temp_weighted_sum = Decimal("0")
        temp_weight_total = Decimal("0")
        emissions_temp_sum = Decimal("0")
        emissions_temp_den = Decimal("0")

        # By asset class
        ac_temp_sums: Dict[str, Decimal] = {}
        ac_temp_weights: Dict[str, Decimal] = {}

        hotspot_entities: List[Tuple[Decimal, str]] = []

        for e in entities:
            ts = e.entity_temperature_score
            if ts <= Decimal("0"):
                not_assessed += 1
                continue

            # Classify
            if ts < self._temp_1_5c:
                below_1_5 += 1
            elif ts == self._temp_1_5c:
                at_1_5 += 1
            elif ts < TEMP_RATING_WB2C:
                between_1_5_2 += 1
            elif ts == TEMP_RATING_WB2C:
                at_2 += 1
            else:
                above_2 += 1
                hotspot_entities.append((ts, e.entity_name))

            # Portfolio-weighted temperature
            if e.outstanding_amount > Decimal("0"):
                temp_weighted_sum += ts * e.outstanding_amount
                temp_weight_total += e.outstanding_amount

            # Emissions-weighted temperature
            if e.financed_emissions_tco2e > Decimal("0"):
                emissions_temp_sum += ts * e.financed_emissions_tco2e
                emissions_temp_den += e.financed_emissions_tco2e

            # By asset class
            ac = e.asset_class
            ac_temp_sums[ac] = ac_temp_sums.get(ac, Decimal("0")) + (
                ts * e.outstanding_amount
            )
            ac_temp_weights[ac] = ac_temp_weights.get(ac, Decimal("0")) + (
                e.outstanding_amount
            )

        portfolio_temp = _safe_divide(temp_weighted_sum, temp_weight_total)
        emissions_temp = _safe_divide(emissions_temp_sum, emissions_temp_den)
        alignment = self._classify_temperature(portfolio_temp)

        # Asset class temperatures
        by_ac: Dict[str, Decimal] = {}
        for ac, temp_sum in ac_temp_sums.items():
            weight = ac_temp_weights.get(ac, Decimal("0"))
            by_ac[ac] = _round_val(_safe_divide(temp_sum, weight), 2)

        # Alignment percentages
        assessed = (
            below_1_5 + at_1_5 + between_1_5_2 + at_2 + above_2
        )
        pct_1_5 = _round_val(
            _safe_pct(
                _decimal(below_1_5 + at_1_5), _decimal(assessed)
            ), 2
        ) if assessed > 0 else Decimal("0")
        pct_2 = _round_val(
            _safe_pct(
                _decimal(below_1_5 + at_1_5 + between_1_5_2 + at_2),
                _decimal(assessed),
            ), 2
        ) if assessed > 0 else Decimal("0")

        gap_1_5 = _round_val(
            max(Decimal("0"), portfolio_temp - self._temp_1_5c), 2
        )

        # Sort hotspots by temperature descending
        hotspot_entities.sort(reverse=True)
        hotspot_names = [name for _, name in hotspot_entities[:10]]

        if alignment in (
            TemperatureAlignment.BELOW_1_5C.value,
            TemperatureAlignment.AT_1_5C.value,
        ):
            msg = (
                f"Portfolio is 1.5C-aligned with weighted temperature "
                f"of {_round_val(portfolio_temp, 2)}C. "
                f"{pct_1_5}% of entities aligned to 1.5C."
            )
        elif alignment == TemperatureAlignment.BETWEEN_1_5_2C.value:
            msg = (
                f"Portfolio temperature of {_round_val(portfolio_temp, 2)}C "
                f"is between 1.5C and 2C. Gap to 1.5C: {gap_1_5}C. "
                f"{above_2} entities above 2C."
            )
        elif alignment == TemperatureAlignment.AT_2C.value:
            msg = (
                f"Portfolio temperature of {_round_val(portfolio_temp, 2)}C "
                f"is at 2C alignment. Improvement needed for 1.5C. "
                f"Gap: {gap_1_5}C."
            )
        elif alignment == TemperatureAlignment.ABOVE_2C.value:
            msg = (
                f"Portfolio temperature of {_round_val(portfolio_temp, 2)}C "
                f"exceeds 2C. Significant decarbonisation needed. "
                f"{above_2} entities above 2C contributing to misalignment."
            )
        else:
            msg = "Temperature alignment not assessed (insufficient data)."

        return TemperatureAlignmentResult(
            portfolio_temperature_score=_round_val(portfolio_temp, 2),
            temperature_alignment=alignment,
            entities_below_1_5c=below_1_5,
            entities_at_1_5c=at_1_5,
            entities_between_1_5_2c=between_1_5_2,
            entities_at_2c=at_2,
            entities_above_2c=above_2,
            entities_not_assessed=not_assessed,
            emissions_weighted_temp=_round_val(emissions_temp, 2),
            hotspot_entities=hotspot_names,
            pct_aligned_1_5c=pct_1_5,
            pct_aligned_2c=pct_2,
            gap_to_1_5c=gap_1_5,
            by_asset_class=by_ac,
            message=msg,
        )

    def _classify_temperature(self, temp: Decimal) -> str:
        """Classify temperature alignment.

        Args:
            temp: Temperature score in degrees C.

        Returns:
            TemperatureAlignment value string.
        """
        if temp <= Decimal("0"):
            return TemperatureAlignment.NOT_ASSESSED.value
        if temp < self._temp_1_5c:
            return TemperatureAlignment.BELOW_1_5C.value
        if temp == self._temp_1_5c:
            return TemperatureAlignment.AT_1_5C.value
        if temp < TEMP_RATING_WB2C:
            return TemperatureAlignment.BETWEEN_1_5_2C.value
        if temp == TEMP_RATING_WB2C:
            return TemperatureAlignment.AT_2C.value
        return TemperatureAlignment.ABOVE_2C.value

    # ------------------------------------------------------------------ #
    # Internal: Recommendations                                            #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        ac_summaries: List[AssetClassSummary],
        coverage: Optional[PortfolioCoverageResult],
        engagement: Optional[EngagementTrackingResult],
        dq: Optional[PcafDataQualityAssessment],
        temperature: Optional[TemperatureAlignmentResult],
        data: FIPortfolioInput,
    ) -> List[FIRecommendation]:
        """Generate FI-specific recommendations.

        Args:
            ac_summaries: Asset class summaries.
            coverage: Coverage assessment.
            engagement: Engagement tracking.
            dq: Data quality assessment.
            temperature: Temperature alignment.
            data: Original input data.

        Returns:
            List of FIRecommendation sorted by priority.
        """
        recs: List[FIRecommendation] = []

        # R1: FINZ-required asset classes without targets
        for acs in ac_summaries:
            if acs.is_finz_required and not acs.target_set:
                recs.append(FIRecommendation(
                    priority="immediate",
                    category="target_setting",
                    asset_class=acs.asset_class,
                    action=(
                        f"Set FINZ target for {acs.asset_class_name}. "
                        f"This asset class is required under SBTi FI "
                        f"Standard V1.0 and represents "
                        f"{acs.pct_of_financed_emissions}% of financed "
                        f"emissions."
                    ),
                    rationale=(
                        "SBTi FINZ V1.0 requires targets for all "
                        "material asset classes in the portfolio."
                    ),
                    estimated_impact=(
                        f"Covers {acs.pct_of_financed_emissions}% "
                        f"of financed emissions"
                    ),
                    timeline_months=6,
                ))

        # R2: Coverage gaps
        if coverage and not coverage.meets_near_term:
            recs.append(FIRecommendation(
                priority="immediate",
                category="coverage",
                asset_class="all",
                action=(
                    f"Increase portfolio coverage from "
                    f"{coverage.coverage_pct}% to at least "
                    f"{coverage.near_term_required_pct}% (near-term). "
                    f"Gap: {coverage.gap_to_near_term_pct}%."
                ),
                rationale=(
                    "SBTi FI Standard V1.0 requires portfolio coverage "
                    f"of at least {coverage.near_term_required_pct}% "
                    f"of financed emissions for near-term targets."
                ),
                estimated_impact="Meets SBTi near-term coverage requirement",
                timeline_months=12,
            ))

        if coverage and coverage.meets_near_term and not coverage.meets_long_term:
            recs.append(FIRecommendation(
                priority="short_term",
                category="coverage",
                asset_class="all",
                action=(
                    f"Plan coverage expansion to "
                    f"{coverage.long_term_required_pct}% for long-term. "
                    f"Current: {coverage.coverage_pct}%. "
                    f"Gap: {coverage.gap_to_long_term_pct}%."
                ),
                rationale=(
                    "SBTi long-term targets require portfolio coverage "
                    f"of at least {coverage.long_term_required_pct}% "
                    f"of financed emissions."
                ),
                estimated_impact="Meets SBTi long-term coverage requirement",
                timeline_months=24,
            ))

        # R3: Engagement gaps
        if engagement and not engagement.meets_target:
            recs.append(FIRecommendation(
                priority="short_term",
                category="engagement",
                asset_class="all",
                action=(
                    f"Expand engagement programme to reach "
                    f"{engagement.target_pct}% of portfolio emissions. "
                    f"Current: {engagement.engagement_pct}%. "
                    f"Gap: {engagement.gap_to_target_pct}%. "
                    f"{engagement.entities_not_engaged} entities not engaged."
                ),
                rationale=(
                    "SBTi PCA approach requires engaging portfolio "
                    "companies to set their own science-based targets."
                ),
                estimated_impact="Increased portfolio coverage via engagement",
                timeline_months=18,
            ))

        # R4: Engagement priorities
        if engagement and engagement.top_engagement_priorities:
            top_5 = engagement.top_engagement_priorities[:5]
            recs.append(FIRecommendation(
                priority="short_term",
                category="engagement",
                asset_class="all",
                action=(
                    f"Prioritise engagement with top emitting entities: "
                    f"{', '.join(top_5)}."
                ),
                rationale=(
                    "Engaging the largest emitters first maximises "
                    "portfolio coverage improvement per engagement effort."
                ),
                estimated_impact="Maximum coverage per engagement effort",
                timeline_months=6,
            ))

        # R5: PCAF data quality
        if dq and not dq.meets_target_quality:
            recs.append(FIRecommendation(
                priority="short_term",
                category="data_quality",
                asset_class="all",
                action=(
                    f"Improve PCAF data quality from weighted score "
                    f"{dq.portfolio_weighted_score} to target of "
                    f"{dq.target_score}. {dq.pct_at_score_4_5}% of "
                    f"entities at score 4-5."
                ),
                rationale=(
                    "Higher PCAF data quality improves financed emissions "
                    "accuracy and supports more robust target-setting."
                ),
                estimated_impact="More accurate financed emissions measurement",
                timeline_months=12,
            ))

        if dq and dq.improvement_priorities:
            top_priorities = dq.improvement_priorities[:5]
            recs.append(FIRecommendation(
                priority="short_term",
                category="data_quality",
                asset_class="all",
                action=(
                    f"Upgrade data quality for high-emitting entities: "
                    f"{', '.join(top_priorities)}."
                ),
                rationale=(
                    "Improving data quality for the largest emitters "
                    "has the greatest impact on portfolio-level PCAF score."
                ),
                estimated_impact="Portfolio PCAF score improvement",
                timeline_months=6,
            ))

        # R6: Temperature alignment
        if temperature and temperature.temperature_alignment == TemperatureAlignment.ABOVE_2C.value:
            recs.append(FIRecommendation(
                priority="immediate",
                category="temperature",
                asset_class="all",
                action=(
                    f"Portfolio temperature of "
                    f"{temperature.portfolio_temperature_score}C exceeds "
                    f"2C. Develop decarbonisation strategy for portfolio. "
                    f"{temperature.entities_above_2c} entities above 2C."
                ),
                rationale=(
                    "Paris Agreement alignment requires portfolio "
                    "temperature well below 2C, preferably 1.5C."
                ),
                estimated_impact="Paris-aligned portfolio temperature",
                timeline_months=12,
            ))

        if temperature and temperature.hotspot_entities:
            hotspots = temperature.hotspot_entities[:5]
            recs.append(FIRecommendation(
                priority="short_term",
                category="temperature",
                asset_class="all",
                action=(
                    f"Address temperature hotspots: {', '.join(hotspots)}. "
                    f"These entities have the highest temperature scores."
                ),
                rationale=(
                    "Reducing temperature scores of the highest-scoring "
                    "entities has the greatest impact on portfolio alignment."
                ),
                estimated_impact="Reduced portfolio temperature score",
                timeline_months=12,
            ))

        # R7: Asset class methodology recommendations
        for acs in ac_summaries:
            if (
                acs.entity_count > 0
                and not acs.target_set
                and acs.pct_of_financed_emissions > Decimal("5")
            ):
                eligible = acs.eligible_methodologies
                if eligible:
                    recs.append(FIRecommendation(
                        priority="medium_term",
                        category="methodology",
                        asset_class=acs.asset_class,
                        action=(
                            f"Select target methodology for "
                            f"{acs.asset_class_name}. Eligible: "
                            f"{', '.join(eligible)}."
                        ),
                        rationale=(
                            f"{acs.asset_class_name} represents "
                            f"{acs.pct_of_financed_emissions}% of financed "
                            f"emissions and should have a methodology-specific "
                            f"target."
                        ),
                        estimated_impact="Structured reduction pathway",
                        timeline_months=6,
                    ))

        # R8: Annual review
        recs.append(FIRecommendation(
            priority="medium_term",
            category="governance",
            asset_class="all",
            action=(
                "Establish annual portfolio alignment review process "
                "covering coverage, engagement, data quality, and "
                "temperature metrics."
            ),
            rationale=(
                "SBTi FI Standard requires annual progress reporting "
                "and regular target review."
            ),
            estimated_impact="Ongoing FINZ compliance and reporting",
            timeline_months=3,
        ))

        # Sort by priority
        priority_order = {
            "immediate": 0,
            "short_term": 1,
            "medium_term": 2,
            "long_term": 3,
        }
        recs.sort(key=lambda r: priority_order.get(r.priority, 4))

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                      #
    # ------------------------------------------------------------------ #

    def get_summary(self, result: FIPortfolioResult) -> Dict[str, Any]:
        """Generate concise summary from portfolio result.

        Args:
            result: FI portfolio result to summarise.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "institution_name": result.institution_name,
            "base_year": result.base_year,
            "total_entities": result.total_entities,
            "total_financed_emissions_tco2e": str(
                result.total_financed_emissions_tco2e
            ),
            "total_portfolio_value": str(result.total_portfolio_value),
            "currency": result.currency,
            "asset_classes_with_data": sum(
                1 for acs in result.asset_class_summaries
                if acs.entity_count > 0
            ),
        }

        if result.coverage:
            summary["coverage_pct"] = str(result.coverage.coverage_pct)
            summary["meets_near_term"] = result.coverage.meets_near_term
            summary["meets_long_term"] = result.coverage.meets_long_term

        if result.engagement:
            summary["engagement_pct"] = str(result.engagement.engagement_pct)
            summary["meets_engagement_target"] = result.engagement.meets_target

        if result.data_quality:
            summary["pcaf_weighted_score"] = str(
                result.data_quality.portfolio_weighted_score
            )
            summary["meets_pcaf_target"] = (
                result.data_quality.meets_target_quality
            )

        if result.temperature:
            summary["portfolio_temperature"] = str(
                result.temperature.portfolio_temperature_score
            )
            summary["temperature_alignment"] = (
                result.temperature.temperature_alignment
            )

        summary["recommendations_count"] = len(result.recommendations)
        summary["warnings_count"] = len(result.warnings)
        summary["provenance_hash"] = _compute_hash(summary)
        return summary

    def validate_methodology_for_asset_class(
        self,
        asset_class: str,
        methodology: str,
    ) -> Tuple[bool, str]:
        """Validate whether a methodology is eligible for an asset class.

        Args:
            asset_class: Asset class identifier.
            methodology: Proposed methodology.

        Returns:
            Tuple of (is_valid, message).
        """
        ref = ASSET_CLASS_REFERENCE.get(asset_class)
        if ref is None:
            return False, f"Unknown asset class: {asset_class}"

        eligible = ref.get("eligible_methodologies", [])
        ac_name = ref.get("name", asset_class)

        if methodology in eligible:
            return True, (
                f"{methodology} is a valid methodology for {ac_name}."
            )

        return False, (
            f"{methodology} is not eligible for {ac_name}. "
            f"Eligible methodologies: {', '.join(eligible)}."
        )

    def calculate_portfolio_emissions_intensity(
        self,
        entities: List[PortfolioEntityInput],
    ) -> Decimal:
        """Calculate portfolio-level emissions intensity.

        Intensity = total financed emissions / total outstanding amount
        In tCO2e per million currency units.

        Args:
            entities: Portfolio entities.

        Returns:
            Portfolio emissions intensity (tCO2e/million).
        """
        total_fe = sum(
            (e.financed_emissions_tco2e for e in entities),
            Decimal("0"),
        )
        total_outstanding = sum(
            (e.outstanding_amount for e in entities),
            Decimal("0"),
        )

        if total_outstanding <= Decimal("0"):
            return Decimal("0")

        # Intensity per million
        intensity = (total_fe / total_outstanding) * Decimal("1000000")
        return _round_val(intensity, 2)

    def estimate_coverage_with_engagement(
        self,
        entities: List[PortfolioEntityInput],
        engagement_success_rate: Decimal = Decimal("0.50"),
    ) -> Dict[str, Any]:
        """Estimate projected coverage if engagement succeeds.

        Args:
            entities: Portfolio entities.
            engagement_success_rate: Expected success rate (0-1).

        Returns:
            Dict with projected coverage metrics.
        """
        total_fe = sum(
            (e.financed_emissions_tco2e for e in entities),
            Decimal("0"),
        )

        current_covered = sum(
            (e.financed_emissions_tco2e for e in entities if e.has_sbti_target),
            Decimal("0"),
        )

        engaged_not_covered = sum(
            (
                e.financed_emissions_tco2e for e in entities
                if not e.has_sbti_target
                and e.engagement_status in (
                    EngagementStatus.ACTIVE.value,
                    EngagementStatus.COMMITTED.value,
                )
            ),
            Decimal("0"),
        )

        projected_additional = engaged_not_covered * engagement_success_rate
        projected_total = current_covered + projected_additional
        projected_pct = _safe_pct(projected_total, total_fe)

        return {
            "current_coverage_pct": str(_round_val(
                _safe_pct(current_covered, total_fe), 2
            )),
            "engaged_not_covered_tco2e": str(_round_val(engaged_not_covered, 2)),
            "engagement_success_rate": str(engagement_success_rate),
            "projected_additional_tco2e": str(_round_val(projected_additional, 2)),
            "projected_coverage_pct": str(_round_val(projected_pct, 2)),
            "meets_near_term": projected_pct >= self._coverage_nt,
            "meets_long_term": projected_pct >= self._coverage_lt,
        }
